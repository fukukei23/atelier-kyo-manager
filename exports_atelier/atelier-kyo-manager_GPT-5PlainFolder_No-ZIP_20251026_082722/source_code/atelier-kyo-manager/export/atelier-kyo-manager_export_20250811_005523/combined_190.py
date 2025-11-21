
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\test_constructors.py ===
from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)
from functools import partial
from operator import attrgetter

import dateutil
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz

from pandas._libs.tslibs import (
    OutOfBoundsDatetime,
    astype_overflowsafe,
    timezones,
)

import pandas as pd
from pandas import (
    DatetimeIndex,
    Index,
    Timestamp,
    date_range,
    offsets,
    to_datetime,
)
import pandas._testing as tm
from pandas.core.arrays import period_array


class TestDatetimeIndex:
    def test_closed_deprecated(self):
        # GH#52628
        msg = "The 'closed' keyword"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            DatetimeIndex([], closed=True)

    def test_normalize_deprecated(self):
        # GH#52628
        msg = "The 'normalize' keyword"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            DatetimeIndex([], normalize=True)

    def test_from_dt64_unsupported_unit(self):
        # GH#49292
        val = np.datetime64(1, "D")
        result = DatetimeIndex([val], tz="US/Pacific")

        expected = DatetimeIndex([val.astype("M8[s]")], tz="US/Pacific")
        tm.assert_index_equal(result, expected)

    def test_explicit_tz_none(self):
        # GH#48659
        dti = date_range("2016-01-01", periods=10, tz="UTC")

        msg = "Passed data is timezone-aware, incompatible with 'tz=None'"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(dti, tz=None)

        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(np.array(dti), tz=None)

        msg = "Cannot pass both a timezone-aware dtype and tz=None"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex([], dtype="M8[ns, UTC]", tz=None)

    def test_freq_validation_with_nat(self):
        # GH#11587 make sure we get a useful error message when generate_range
        #  raises
        msg = (
            "Inferred frequency None from passed values does not conform "
            "to passed frequency D"
        )
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex([pd.NaT, Timestamp("2011-01-01")], freq="D")
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex([pd.NaT, Timestamp("2011-01-01")._value], freq="D")

    # TODO: better place for tests shared by DTI/TDI?
    @pytest.mark.parametrize(
        "index",
        [
            date_range("2016-01-01", periods=5, tz="US/Pacific"),
            pd.timedelta_range("1 Day", periods=5),
        ],
    )
    def test_shallow_copy_inherits_array_freq(self, index):
        # If we pass a DTA/TDA to shallow_copy and dont specify a freq,
        #  we should inherit the array's freq, not our own.
        array = index._data

        arr = array[[0, 3, 2, 4, 1]]
        assert arr.freq is None

        result = index._shallow_copy(arr)
        assert result.freq is None

    def test_categorical_preserves_tz(self):
        # GH#18664 retain tz when going DTI-->Categorical-->DTI
        dti = DatetimeIndex(
            [pd.NaT, "2015-01-01", "1999-04-06 15:14:13", "2015-01-01"], tz="US/Eastern"
        )

        for dtobj in [dti, dti._data]:
            # works for DatetimeIndex or DatetimeArray

            ci = pd.CategoricalIndex(dtobj)
            carr = pd.Categorical(dtobj)
            cser = pd.Series(ci)

            for obj in [ci, carr, cser]:
                result = DatetimeIndex(obj)
                tm.assert_index_equal(result, dti)

    def test_dti_with_period_data_raises(self):
        # GH#23675
        data = pd.PeriodIndex(["2016Q1", "2016Q2"], freq="Q")

        with pytest.raises(TypeError, match="PeriodDtype data is invalid"):
            DatetimeIndex(data)

        with pytest.raises(TypeError, match="PeriodDtype data is invalid"):
            to_datetime(data)

        with pytest.raises(TypeError, match="PeriodDtype data is invalid"):
            DatetimeIndex(period_array(data))

        with pytest.raises(TypeError, match="PeriodDtype data is invalid"):
            to_datetime(period_array(data))

    def test_dti_with_timedelta64_data_raises(self):
        # GH#23675 deprecated, enforrced in GH#29794
        data = np.array([0], dtype="m8[ns]")
        msg = r"timedelta64\[ns\] cannot be converted to datetime64"
        with pytest.raises(TypeError, match=msg):
            DatetimeIndex(data)

        with pytest.raises(TypeError, match=msg):
            to_datetime(data)

        with pytest.raises(TypeError, match=msg):
            DatetimeIndex(pd.TimedeltaIndex(data))

        with pytest.raises(TypeError, match=msg):
            to_datetime(pd.TimedeltaIndex(data))

    def test_constructor_from_sparse_array(self):
        # https://github.com/pandas-dev/pandas/issues/35843
        values = [
            Timestamp("2012-05-01T01:00:00.000000"),
            Timestamp("2016-05-01T01:00:00.000000"),
        ]
        arr = pd.arrays.SparseArray(values)
        result = Index(arr)
        assert type(result) is Index
        assert result.dtype == arr.dtype

    def test_construction_caching(self):
        df = pd.DataFrame(
            {
                "dt": date_range("20130101", periods=3),
                "dttz": date_range("20130101", periods=3, tz="US/Eastern"),
                "dt_with_null": [
                    Timestamp("20130101"),
                    pd.NaT,
                    Timestamp("20130103"),
                ],
                "dtns": date_range("20130101", periods=3, freq="ns"),
            }
        )
        assert df.dttz.dtype.tz.zone == "US/Eastern"

    @pytest.mark.parametrize(
        "kwargs",
        [{"tz": "dtype.tz"}, {"dtype": "dtype"}, {"dtype": "dtype", "tz": "dtype.tz"}],
    )
    def test_construction_with_alt(self, kwargs, tz_aware_fixture):
        tz = tz_aware_fixture
        i = date_range("20130101", periods=5, freq="h", tz=tz)
        kwargs = {key: attrgetter(val)(i) for key, val in kwargs.items()}
        result = DatetimeIndex(i, **kwargs)
        tm.assert_index_equal(i, result)

    @pytest.mark.parametrize(
        "kwargs",
        [{"tz": "dtype.tz"}, {"dtype": "dtype"}, {"dtype": "dtype", "tz": "dtype.tz"}],
    )
    def test_construction_with_alt_tz_localize(self, kwargs, tz_aware_fixture):
        tz = tz_aware_fixture
        i = date_range("20130101", periods=5, freq="h", tz=tz)
        i = i._with_freq(None)
        kwargs = {key: attrgetter(val)(i) for key, val in kwargs.items()}

        if "tz" in kwargs:
            result = DatetimeIndex(i.asi8, tz="UTC").tz_convert(kwargs["tz"])

            expected = DatetimeIndex(i, **kwargs)
            tm.assert_index_equal(result, expected)

        # localize into the provided tz
        i2 = DatetimeIndex(i.tz_localize(None).asi8, tz="UTC")
        expected = i.tz_localize(None).tz_localize("UTC")
        tm.assert_index_equal(i2, expected)

        # incompat tz/dtype
        msg = "cannot supply both a tz and a dtype with a tz"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(i.tz_localize(None).asi8, dtype=i.dtype, tz="US/Pacific")

    def test_construction_index_with_mixed_timezones(self):
        # gh-11488: no tz results in DatetimeIndex
        result = Index([Timestamp("2011-01-01"), Timestamp("2011-01-02")], name="idx")
        exp = DatetimeIndex(
            [Timestamp("2011-01-01"), Timestamp("2011-01-02")], name="idx"
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is None

        # same tz results in DatetimeIndex
        result = Index(
            [
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                Timestamp("2011-01-02 10:00", tz="Asia/Tokyo"),
            ],
            name="idx",
        )
        exp = DatetimeIndex(
            [Timestamp("2011-01-01 10:00"), Timestamp("2011-01-02 10:00")],
            tz="Asia/Tokyo",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz

        # same tz results in DatetimeIndex (DST)
        result = Index(
            [
                Timestamp("2011-01-01 10:00", tz="US/Eastern"),
                Timestamp("2011-08-01 10:00", tz="US/Eastern"),
            ],
            name="idx",
        )
        exp = DatetimeIndex(
            [Timestamp("2011-01-01 10:00"), Timestamp("2011-08-01 10:00")],
            tz="US/Eastern",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz

        # Different tz results in Index(dtype=object)
        result = Index(
            [
                Timestamp("2011-01-01 10:00"),
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            name="idx",
        )
        exp = Index(
            [
                Timestamp("2011-01-01 10:00"),
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            dtype="object",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert not isinstance(result, DatetimeIndex)

        result = Index(
            [
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            name="idx",
        )
        exp = Index(
            [
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            dtype="object",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert not isinstance(result, DatetimeIndex)

        msg = "DatetimeIndex has mixed timezones"
        msg_depr = "parsing datetimes with mixed time zones will raise an error"
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=msg_depr):
                DatetimeIndex(["2013-11-02 22:00-05:00", "2013-11-03 22:00-06:00"])

        # length = 1
        result = Index([Timestamp("2011-01-01")], name="idx")
        exp = DatetimeIndex([Timestamp("2011-01-01")], name="idx")
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is None

        # length = 1 with tz
        result = Index([Timestamp("2011-01-01 10:00", tz="Asia/Tokyo")], name="idx")
        exp = DatetimeIndex(
            [Timestamp("2011-01-01 10:00")], tz="Asia/Tokyo", name="idx"
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz

    def test_construction_index_with_mixed_timezones_with_NaT(self):
        # see gh-11488
        result = Index(
            [pd.NaT, Timestamp("2011-01-01"), pd.NaT, Timestamp("2011-01-02")],
            name="idx",
        )
        exp = DatetimeIndex(
            [pd.NaT, Timestamp("2011-01-01"), pd.NaT, Timestamp("2011-01-02")],
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is None

        # Same tz results in DatetimeIndex
        result = Index(
            [
                pd.NaT,
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                pd.NaT,
                Timestamp("2011-01-02 10:00", tz="Asia/Tokyo"),
            ],
            name="idx",
        )
        exp = DatetimeIndex(
            [
                pd.NaT,
                Timestamp("2011-01-01 10:00"),
                pd.NaT,
                Timestamp("2011-01-02 10:00"),
            ],
            tz="Asia/Tokyo",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz

        # same tz results in DatetimeIndex (DST)
        result = Index(
            [
                Timestamp("2011-01-01 10:00", tz="US/Eastern"),
                pd.NaT,
                Timestamp("2011-08-01 10:00", tz="US/Eastern"),
            ],
            name="idx",
        )
        exp = DatetimeIndex(
            [Timestamp("2011-01-01 10:00"), pd.NaT, Timestamp("2011-08-01 10:00")],
            tz="US/Eastern",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz

        # different tz results in Index(dtype=object)
        result = Index(
            [
                pd.NaT,
                Timestamp("2011-01-01 10:00"),
                pd.NaT,
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            name="idx",
        )
        exp = Index(
            [
                pd.NaT,
                Timestamp("2011-01-01 10:00"),
                pd.NaT,
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            dtype="object",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert not isinstance(result, DatetimeIndex)

        result = Index(
            [
                pd.NaT,
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                pd.NaT,
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            name="idx",
        )
        exp = Index(
            [
                pd.NaT,
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                pd.NaT,
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            dtype="object",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert not isinstance(result, DatetimeIndex)

        # all NaT
        result = Index([pd.NaT, pd.NaT], name="idx")
        exp = DatetimeIndex([pd.NaT, pd.NaT], name="idx")
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is None

    def test_construction_dti_with_mixed_timezones(self):
        # GH 11488 (not changed, added explicit tests)

        # no tz results in DatetimeIndex
        result = DatetimeIndex(
            [Timestamp("2011-01-01"), Timestamp("2011-01-02")], name="idx"
        )
        exp = DatetimeIndex(
            [Timestamp("2011-01-01"), Timestamp("2011-01-02")], name="idx"
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)

        # same tz results in DatetimeIndex
        result = DatetimeIndex(
            [
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                Timestamp("2011-01-02 10:00", tz="Asia/Tokyo"),
            ],
            name="idx",
        )
        exp = DatetimeIndex(
            [Timestamp("2011-01-01 10:00"), Timestamp("2011-01-02 10:00")],
            tz="Asia/Tokyo",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)

        # same tz results in DatetimeIndex (DST)
        result = DatetimeIndex(
            [
                Timestamp("2011-01-01 10:00", tz="US/Eastern"),
                Timestamp("2011-08-01 10:00", tz="US/Eastern"),
            ],
            name="idx",
        )
        exp = DatetimeIndex(
            [Timestamp("2011-01-01 10:00"), Timestamp("2011-08-01 10:00")],
            tz="US/Eastern",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)

        # tz mismatch affecting to tz-aware raises TypeError/ValueError

        msg = "cannot be converted to datetime64"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(
                [
                    Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                    Timestamp("2011-01-02 10:00", tz="US/Eastern"),
                ],
                name="idx",
            )

        # pre-2.0 this raised bc of awareness mismatch. in 2.0 with a tz#
        #  specified we behave as if this was called pointwise, so
        #  the naive Timestamp is treated as a wall time.
        dti = DatetimeIndex(
            [
                Timestamp("2011-01-01 10:00"),
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            tz="Asia/Tokyo",
            name="idx",
        )
        expected = DatetimeIndex(
            [
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                Timestamp("2011-01-02 10:00", tz="US/Eastern").tz_convert("Asia/Tokyo"),
            ],
            tz="Asia/Tokyo",
            name="idx",
        )
        tm.assert_index_equal(dti, expected)

        # pre-2.0 mixed-tz scalars raised even if a tz/dtype was specified.
        #  as of 2.0 we successfully return the requested tz/dtype
        dti = DatetimeIndex(
            [
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            tz="US/Eastern",
            name="idx",
        )
        expected = DatetimeIndex(
            [
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo").tz_convert("US/Eastern"),
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            tz="US/Eastern",
            name="idx",
        )
        tm.assert_index_equal(dti, expected)

        # same thing but pass dtype instead of tz
        dti = DatetimeIndex(
            [
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            dtype="M8[ns, US/Eastern]",
            name="idx",
        )
        tm.assert_index_equal(dti, expected)

    def test_construction_base_constructor(self):
        arr = [Timestamp("2011-01-01"), pd.NaT, Timestamp("2011-01-03")]
        tm.assert_index_equal(Index(arr), DatetimeIndex(arr))
        tm.assert_index_equal(Index(np.array(arr)), DatetimeIndex(np.array(arr)))

        arr = [np.nan, pd.NaT, Timestamp("2011-01-03")]
        tm.assert_index_equal(Index(arr), DatetimeIndex(arr))
        tm.assert_index_equal(Index(np.array(arr)), DatetimeIndex(np.array(arr)))

    def test_construction_outofbounds(self):
        # GH 13663
        dates = [
            datetime(3000, 1, 1),
            datetime(4000, 1, 1),
            datetime(5000, 1, 1),
            datetime(6000, 1, 1),
        ]
        exp = Index(dates, dtype=object)
        # coerces to object
        tm.assert_index_equal(Index(dates), exp)

        msg = "^Out of bounds nanosecond timestamp: 3000-01-01 00:00:00, at position 0$"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            # can't create DatetimeIndex
            DatetimeIndex(dates)

    @pytest.mark.parametrize("data", [["1400-01-01"], [datetime(1400, 1, 1)]])
    def test_dti_date_out_of_range(self, data):
        # GH#1475
        msg = (
            "^Out of bounds nanosecond timestamp: "
            "1400-01-01( 00:00:00)?, at position 0$"
        )
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            DatetimeIndex(data)

    def test_construction_with_ndarray(self):
        # GH 5152
        dates = [datetime(2013, 10, 7), datetime(2013, 10, 8), datetime(2013, 10, 9)]
        data = DatetimeIndex(dates, freq=offsets.BDay()).values
        result = DatetimeIndex(data, freq=offsets.BDay())
        expected = DatetimeIndex(["2013-10-07", "2013-10-08", "2013-10-09"], freq="B")
        tm.assert_index_equal(result, expected)

    def test_integer_values_and_tz_interpreted_as_utc(self):
        # GH-24559
        val = np.datetime64("2000-01-01 00:00:00", "ns")
        values = np.array([val.view("i8")])

        result = DatetimeIndex(values).tz_localize("US/Central")

        expected = DatetimeIndex(["2000-01-01T00:00:00"], dtype="M8[ns, US/Central]")
        tm.assert_index_equal(result, expected)

        # but UTC is *not* deprecated.
        with tm.assert_produces_warning(None):
            result = DatetimeIndex(values, tz="UTC")
        expected = DatetimeIndex(["2000-01-01T00:00:00"], dtype="M8[ns, UTC]")
        tm.assert_index_equal(result, expected)

    def test_constructor_coverage(self):
        msg = r"DatetimeIndex\(\.\.\.\) must be called with a collection"
        with pytest.raises(TypeError, match=msg):
            DatetimeIndex("1/1/2000")

        # generator expression
        gen = (datetime(2000, 1, 1) + timedelta(i) for i in range(10))
        result = DatetimeIndex(gen)
        expected = DatetimeIndex(
            [datetime(2000, 1, 1) + timedelta(i) for i in range(10)]
        )
        tm.assert_index_equal(result, expected)

        # NumPy string array
        strings = np.array(["2000-01-01", "2000-01-02", "2000-01-03"])
        result = DatetimeIndex(strings)
        expected = DatetimeIndex(strings.astype("O"))
        tm.assert_index_equal(result, expected)

        from_ints = DatetimeIndex(expected.asi8)
        tm.assert_index_equal(from_ints, expected)

        # string with NaT
        strings = np.array(["2000-01-01", "2000-01-02", "NaT"])
        result = DatetimeIndex(strings)
        expected = DatetimeIndex(strings.astype("O"))
        tm.assert_index_equal(result, expected)

        from_ints = DatetimeIndex(expected.asi8)
        tm.assert_index_equal(from_ints, expected)

        # non-conforming
        msg = (
            "Inferred frequency None from passed values does not conform "
            "to passed frequency D"
        )
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-04"], freq="D")

    @pytest.mark.parametrize("freq", ["YS", "W-SUN"])
    def test_constructor_datetime64_tzformat(self, freq):
        # see GH#6572: ISO 8601 format results in stdlib timezone object
        idx = date_range(
            "2013-01-01T00:00:00-05:00", "2016-01-01T23:59:59-05:00", freq=freq
        )
        expected = date_range(
            "2013-01-01T00:00:00",
            "2016-01-01T23:59:59",
            freq=freq,
            tz=timezone(timedelta(minutes=-300)),
        )
        tm.assert_index_equal(idx, expected)
        # Unable to use `US/Eastern` because of DST
        expected_i8 = date_range(
            "2013-01-01T00:00:00", "2016-01-01T23:59:59", freq=freq, tz="America/Lima"
        )
        tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)

        idx = date_range(
            "2013-01-01T00:00:00+09:00", "2016-01-01T23:59:59+09:00", freq=freq
        )
        expected = date_range(
            "2013-01-01T00:00:00",
            "2016-01-01T23:59:59",
            freq=freq,
            tz=timezone(timedelta(minutes=540)),
        )
        tm.assert_index_equal(idx, expected)
        expected_i8 = date_range(
            "2013-01-01T00:00:00", "2016-01-01T23:59:59", freq=freq, tz="Asia/Tokyo"
        )
        tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)

        # Non ISO 8601 format results in dateutil.tz.tzoffset
        idx = date_range("2013/1/1 0:00:00-5:00", "2016/1/1 23:59:59-5:00", freq=freq)
        expected = date_range(
            "2013-01-01T00:00:00",
            "2016-01-01T23:59:59",
            freq=freq,
            tz=timezone(timedelta(minutes=-300)),
        )
        tm.assert_index_equal(idx, expected)
        # Unable to use `US/Eastern` because of DST
        expected_i8 = date_range(
            "2013-01-01T00:00:00", "2016-01-01T23:59:59", freq=freq, tz="America/Lima"
        )
        tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)

        idx = date_range("2013/1/1 0:00:00+9:00", "2016/1/1 23:59:59+09:00", freq=freq)
        expected = date_range(
            "2013-01-01T00:00:00",
            "2016-01-01T23:59:59",
            freq=freq,
            tz=timezone(timedelta(minutes=540)),
        )
        tm.assert_index_equal(idx, expected)
        expected_i8 = date_range(
            "2013-01-01T00:00:00", "2016-01-01T23:59:59", freq=freq, tz="Asia/Tokyo"
        )
        tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)

    def test_constructor_dtype(self):
        # passing a dtype with a tz should localize
        idx = DatetimeIndex(
            ["2013-01-01", "2013-01-02"], dtype="datetime64[ns, US/Eastern]"
        )
        expected = (
            DatetimeIndex(["2013-01-01", "2013-01-02"])
            .as_unit("ns")
            .tz_localize("US/Eastern")
        )
        tm.assert_index_equal(idx, expected)

        idx = DatetimeIndex(["2013-01-01", "2013-01-02"], tz="US/Eastern").as_unit("ns")
        tm.assert_index_equal(idx, expected)

    def test_constructor_dtype_tz_mismatch_raises(self):
        # if we already have a tz and its not the same, then raise
        idx = DatetimeIndex(
            ["2013-01-01", "2013-01-02"], dtype="datetime64[ns, US/Eastern]"
        )

        msg = (
            "cannot supply both a tz and a timezone-naive dtype "
            r"\(i\.e\. datetime64\[ns\]\)"
        )
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(idx, dtype="datetime64[ns]")

        # this is effectively trying to convert tz's
        msg = "data is already tz-aware US/Eastern, unable to set specified tz: CET"
        with pytest.raises(TypeError, match=msg):
            DatetimeIndex(idx, dtype="datetime64[ns, CET]")
        msg = "cannot supply both a tz and a dtype with a tz"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(idx, tz="CET", dtype="datetime64[ns, US/Eastern]")

        result = DatetimeIndex(idx, dtype="datetime64[ns, US/Eastern]")
        tm.assert_index_equal(idx, result)

    @pytest.mark.parametrize("dtype", [object, np.int32, np.int64])
    def test_constructor_invalid_dtype_raises(self, dtype):
        # GH 23986
        msg = "Unexpected value for 'dtype'"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex([1, 2], dtype=dtype)

    def test_000constructor_resolution(self):
        # 2252
        t1 = Timestamp((1352934390 * 1000000000) + 1000000 + 1000 + 1)
        idx = DatetimeIndex([t1])

        assert idx.nanosecond[0] == t1.nanosecond

    def test_disallow_setting_tz(self):
        # GH 3746
        dti = DatetimeIndex(["2010"], tz="UTC")
        msg = "Cannot directly set timezone"
        with pytest.raises(AttributeError, match=msg):
            dti.tz = pytz.timezone("US/Pacific")

    @pytest.mark.parametrize(
        "tz",
        [
            None,
            "America/Los_Angeles",
            pytz.timezone("America/Los_Angeles"),
            Timestamp("2000", tz="America/Los_Angeles").tz,
        ],
    )
    def test_constructor_start_end_with_tz(self, tz):
        # GH 18595
        start = Timestamp("2013-01-01 06:00:00", tz="America/Los_Angeles")
        end = Timestamp("2013-01-02 06:00:00", tz="America/Los_Angeles")
        result = date_range(freq="D", start=start, end=end, tz=tz)
        expected = DatetimeIndex(
            ["2013-01-01 06:00:00", "2013-01-02 06:00:00"],
            dtype="M8[ns, America/Los_Angeles]",
            freq="D",
        )
        tm.assert_index_equal(result, expected)
        # Especially assert that the timezone is consistent for pytz
        assert pytz.timezone("America/Los_Angeles") is result.tz

    @pytest.mark.parametrize("tz", ["US/Pacific", "US/Eastern", "Asia/Tokyo"])
    def test_constructor_with_non_normalized_pytz(self, tz):
        # GH 18595
        non_norm_tz = Timestamp("2010", tz=tz).tz
        result = DatetimeIndex(["2010"], tz=non_norm_tz)
        assert pytz.timezone(tz) is result.tz

    def test_constructor_timestamp_near_dst(self):
        # GH 20854
        ts = [
            Timestamp("2016-10-30 03:00:00+0300", tz="Europe/Helsinki"),
            Timestamp("2016-10-30 03:00:00+0200", tz="Europe/Helsinki"),
        ]
        result = DatetimeIndex(ts)
        expected = DatetimeIndex([ts[0].to_pydatetime(), ts[1].to_pydatetime()])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("klass", [Index, DatetimeIndex])
    @pytest.mark.parametrize("box", [np.array, partial(np.array, dtype=object), list])
    @pytest.mark.parametrize(
        "tz, dtype",
        [("US/Pacific", "datetime64[ns, US/Pacific]"), (None, "datetime64[ns]")],
    )
    def test_constructor_with_int_tz(self, klass, box, tz, dtype):
        # GH 20997, 20964
        ts = Timestamp("2018-01-01", tz=tz).as_unit("ns")
        result = klass(box([ts._value]), dtype=dtype)
        expected = klass([ts])
        assert result == expected

    def test_construction_int_rountrip(self, tz_naive_fixture):
        # GH 12619, GH#24559
        tz = tz_naive_fixture

        result = 1293858000000000000
        expected = DatetimeIndex([result], tz=tz).asi8[0]
        assert result == expected

    def test_construction_from_replaced_timestamps_with_dst(self):
        # GH 18785
        index = date_range(
            Timestamp(2000, 12, 31),
            Timestamp(2005, 12, 31),
            freq="YE-DEC",
            tz="Australia/Melbourne",
        )
        result = DatetimeIndex([x.replace(month=6, day=1) for x in index])
        expected = DatetimeIndex(
            [
                "2000-06-01 00:00:00",
                "2001-06-01 00:00:00",
                "2002-06-01 00:00:00",
                "2003-06-01 00:00:00",
                "2004-06-01 00:00:00",
                "2005-06-01 00:00:00",
            ],
            tz="Australia/Melbourne",
        )
        tm.assert_index_equal(result, expected)

    def test_construction_with_tz_and_tz_aware_dti(self):
        # GH 23579
        dti = date_range("2016-01-01", periods=3, tz="US/Central")
        msg = "data is already tz-aware US/Central, unable to set specified tz"
        with pytest.raises(TypeError, match=msg):
            DatetimeIndex(dti, tz="Asia/Tokyo")

    def test_construction_with_nat_and_tzlocal(self):
        tz = dateutil.tz.tzlocal()
        result = DatetimeIndex(["2018", "NaT"], tz=tz)
        expected = DatetimeIndex([Timestamp("2018", tz=tz), pd.NaT])
        tm.assert_index_equal(result, expected)

    def test_constructor_with_ambiguous_keyword_arg(self):
        # GH 35297

        expected = DatetimeIndex(
            ["2020-11-01 01:00:00", "2020-11-02 01:00:00"],
            dtype="datetime64[ns, America/New_York]",
            freq="D",
            ambiguous=False,
        )

        # ambiguous keyword in start
        timezone = "America/New_York"
        start = Timestamp(year=2020, month=11, day=1, hour=1).tz_localize(
            timezone, ambiguous=False
        )
        result = date_range(start=start, periods=2, ambiguous=False)
        tm.assert_index_equal(result, expected)

        # ambiguous keyword in end
        timezone = "America/New_York"
        end = Timestamp(year=2020, month=11, day=2, hour=1).tz_localize(
            timezone, ambiguous=False
        )
        result = date_range(end=end, periods=2, ambiguous=False)
        tm.assert_index_equal(result, expected)

    def test_constructor_with_nonexistent_keyword_arg(self, warsaw):
        # GH 35297
        timezone = warsaw

        # nonexistent keyword in start
        start = Timestamp("2015-03-29 02:30:00").tz_localize(
            timezone, nonexistent="shift_forward"
        )
        result = date_range(start=start, periods=2, freq="h")
        expected = DatetimeIndex(
            [
                Timestamp("2015-03-29 03:00:00+02:00", tz=timezone),
                Timestamp("2015-03-29 04:00:00+02:00", tz=timezone),
            ]
        )

        tm.assert_index_equal(result, expected)

        # nonexistent keyword in end
        end = start
        result = date_range(end=end, periods=2, freq="h")
        expected = DatetimeIndex(
            [
                Timestamp("2015-03-29 01:00:00+01:00", tz=timezone),
                Timestamp("2015-03-29 03:00:00+02:00", tz=timezone),
            ]
        )

        tm.assert_index_equal(result, expected)

    def test_constructor_no_precision_raises(self):
        # GH-24753, GH-24739

        msg = "with no precision is not allowed"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(["2000"], dtype="datetime64")

        msg = "The 'datetime64' dtype has no unit. Please pass in"
        with pytest.raises(ValueError, match=msg):
            Index(["2000"], dtype="datetime64")

    def test_constructor_wrong_precision_raises(self):
        dti = DatetimeIndex(["2000"], dtype="datetime64[us]")
        assert dti.dtype == "M8[us]"
        assert dti[0] == Timestamp(2000, 1, 1)

    def test_index_constructor_with_numpy_object_array_and_timestamp_tz_with_nan(self):
        # GH 27011
        result = Index(np.array([Timestamp("2019", tz="UTC"), np.nan], dtype=object))
        expected = DatetimeIndex([Timestamp("2019", tz="UTC"), pd.NaT])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("tz", [pytz.timezone("US/Eastern"), gettz("US/Eastern")])
    def test_dti_from_tzaware_datetime(self, tz):
        d = [datetime(2012, 8, 19, tzinfo=tz)]

        index = DatetimeIndex(d)
        assert timezones.tz_compare(index.tz, tz)

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_dti_tz_constructors(self, tzstr):
        """Test different DatetimeIndex constructions with timezone
        Follow-up of GH#4229
        """
        arr = ["11/10/2005 08:00:00", "11/10/2005 09:00:00"]

        idx1 = to_datetime(arr).tz_localize(tzstr)
        idx2 = date_range(start="2005-11-10 08:00:00", freq="h", periods=2, tz=tzstr)
        idx2 = idx2._with_freq(None)  # the others all have freq=None
        idx3 = DatetimeIndex(arr, tz=tzstr)
        idx4 = DatetimeIndex(np.array(arr), tz=tzstr)

        for other in [idx2, idx3, idx4]:
            tm.assert_index_equal(idx1, other)

    def test_dti_construction_idempotent(self, unit):
        rng = date_range(
            "03/12/2012 00:00", periods=10, freq="W-FRI", tz="US/Eastern", unit=unit
        )
        rng2 = DatetimeIndex(data=rng, tz="US/Eastern")
        tm.assert_index_equal(rng, rng2)

    @pytest.mark.parametrize("prefix", ["", "dateutil/"])
    def test_dti_constructor_static_tzinfo(self, prefix):
        # it works!
        index = DatetimeIndex([datetime(2012, 1, 1)], tz=prefix + "EST")
        index.hour
        index[0]

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_dti_convert_datetime_list(self, tzstr):
        dr = date_range("2012-06-02", periods=10, tz=tzstr, name="foo")
        dr2 = DatetimeIndex(list(dr), name="foo", freq="D")
        tm.assert_index_equal(dr, dr2)

    @pytest.mark.parametrize(
        "tz",
        [
            pytz.timezone("US/Eastern"),
            gettz("US/Eastern"),
        ],
    )
    @pytest.mark.parametrize("use_str", [True, False])
    @pytest.mark.parametrize("box_cls", [Timestamp, DatetimeIndex])
    def test_dti_ambiguous_matches_timestamp(self, tz, use_str, box_cls, request):
        # GH#47471 check that we get the same raising behavior in the DTI
        # constructor and Timestamp constructor
        dtstr = "2013-11-03 01:59:59.999999"
        item = dtstr
        if not use_str:
            item = Timestamp(dtstr).to_pydatetime()
        if box_cls is not Timestamp:
            item = [item]

        if not use_str and isinstance(tz, dateutil.tz.tzfile):
            # FIXME: The Timestamp constructor here behaves differently than all
            #  the other cases bc with dateutil/zoneinfo tzinfos we implicitly
            #  get fold=0. Having this raise is not important, but having the
            #  behavior be consistent across cases is.
            mark = pytest.mark.xfail(reason="We implicitly get fold=0.")
            request.applymarker(mark)

        with pytest.raises(pytz.AmbiguousTimeError, match=dtstr):
            box_cls(item, tz=tz)

    @pytest.mark.parametrize("tz", [None, "UTC", "US/Pacific"])
    def test_dti_constructor_with_non_nano_dtype(self, tz):
        # GH#55756, GH#54620
        ts = Timestamp("2999-01-01")
        dtype = "M8[us]"
        if tz is not None:
            dtype = f"M8[us, {tz}]"
        vals = [ts, "2999-01-02 03:04:05.678910", 2500]
        result = DatetimeIndex(vals, dtype=dtype)
        # The 2500 is interpreted as microseconds, consistent with what
        #  we would get if we created DatetimeIndexes from vals[:2] and vals[2:]
        #  and concated the results.
        pointwise = [
            vals[0].tz_localize(tz),
            Timestamp(vals[1], tz=tz),
            to_datetime(vals[2], unit="us", utc=True).tz_convert(tz),
        ]
        exp_vals = [x.as_unit("us").asm8 for x in pointwise]
        exp_arr = np.array(exp_vals, dtype="M8[us]")
        expected = DatetimeIndex(exp_arr, dtype="M8[us]")
        if tz is not None:
            expected = expected.tz_localize("UTC").tz_convert(tz)
        tm.assert_index_equal(result, expected)

        result2 = DatetimeIndex(np.array(vals, dtype=object), dtype=dtype)
        tm.assert_index_equal(result2, expected)

    def test_dti_constructor_with_non_nano_now_today(self):
        # GH#55756
        now = Timestamp.now()
        today = Timestamp.today()
        result = DatetimeIndex(["now", "today"], dtype="M8[s]")
        assert result.dtype == "M8[s]"

        # result may not exactly match [now, today] so we'll test it up to a tolerance.
        #  (it *may* match exactly due to rounding)
        tolerance = pd.Timedelta(microseconds=1)

        diff0 = result[0] - now.as_unit("s")
        assert diff0 >= pd.Timedelta(0)
        assert diff0 < tolerance

        diff1 = result[1] - today.as_unit("s")
        assert diff1 >= pd.Timedelta(0)
        assert diff1 < tolerance

    def test_dti_constructor_object_float_matches_float_dtype(self):
        # GH#55780
        arr = np.array([0, np.nan], dtype=np.float64)
        arr2 = arr.astype(object)

        dti1 = DatetimeIndex(arr, tz="CET")
        dti2 = DatetimeIndex(arr2, tz="CET")
        tm.assert_index_equal(dti1, dti2)

    @pytest.mark.parametrize("dtype", ["M8[us]", "M8[us, US/Pacific]"])
    def test_dti_constructor_with_dtype_object_int_matches_int_dtype(self, dtype):
        # Going through the object path should match the non-object path

        vals1 = np.arange(5, dtype="i8") * 1000
        vals1[0] = pd.NaT.value

        vals2 = vals1.astype(np.float64)
        vals2[0] = np.nan

        vals3 = vals1.astype(object)
        # change lib.infer_dtype(vals3) from "integer" so we go through
        #  array_to_datetime in _sequence_to_dt64
        vals3[0] = pd.NaT

        vals4 = vals2.astype(object)

        res1 = DatetimeIndex(vals1, dtype=dtype)
        res2 = DatetimeIndex(vals2, dtype=dtype)
        res3 = DatetimeIndex(vals3, dtype=dtype)
        res4 = DatetimeIndex(vals4, dtype=dtype)

        expected = DatetimeIndex(vals1.view("M8[us]"))
        if res1.tz is not None:
            expected = expected.tz_localize("UTC").tz_convert(res1.tz)
        tm.assert_index_equal(res1, expected)
        tm.assert_index_equal(res2, expected)
        tm.assert_index_equal(res3, expected)
        tm.assert_index_equal(res4, expected)


class TestTimeSeries:
    def test_dti_constructor_preserve_dti_freq(self):
        rng = date_range("1/1/2000", "1/2/2000", freq="5min")

        rng2 = DatetimeIndex(rng)
        assert rng.freq == rng2.freq

    def test_explicit_none_freq(self):
        # Explicitly passing freq=None is respected
        rng = date_range("1/1/2000", "1/2/2000", freq="5min")

        result = DatetimeIndex(rng, freq=None)
        assert result.freq is None

        result = DatetimeIndex(rng._data, freq=None)
        assert result.freq is None

    def test_dti_constructor_small_int(self, any_int_numpy_dtype):
        # see gh-13721
        exp = DatetimeIndex(
            [
                "1970-01-01 00:00:00.00000000",
                "1970-01-01 00:00:00.00000001",
                "1970-01-01 00:00:00.00000002",
            ]
        )

        arr = np.array([0, 10, 20], dtype=any_int_numpy_dtype)
        tm.assert_index_equal(DatetimeIndex(arr), exp)

    def test_ctor_str_intraday(self):
        rng = DatetimeIndex(["1-1-2000 00:00:01"])
        assert rng[0].second == 1

    def test_index_cast_datetime64_other_units(self):
        arr = np.arange(0, 100, 10, dtype=np.int64).view("M8[D]")
        idx = Index(arr)

        assert (idx.values == astype_overflowsafe(arr, dtype=np.dtype("M8[ns]"))).all()

    def test_constructor_int64_nocopy(self):
        # GH#1624
        arr = np.arange(1000, dtype=np.int64)
        index = DatetimeIndex(arr)

        arr[50:100] = -1
        assert (index.asi8[50:100] == -1).all()

        arr = np.arange(1000, dtype=np.int64)
        index = DatetimeIndex(arr, copy=True)

        arr[50:100] = -1
        assert (index.asi8[50:100] != -1).all()

    @pytest.mark.parametrize(
        "freq",
        ["ME", "QE", "YE", "D", "B", "bh", "min", "s", "ms", "us", "h", "ns", "C"],
    )
    def test_from_freq_recreate_from_data(self, freq):
        org = date_range(start="2001/02/01 09:00", freq=freq, periods=1)
        idx = DatetimeIndex(org, freq=freq)
        tm.assert_index_equal(idx, org)

        org = date_range(
            start="2001/02/01 09:00", freq=freq, tz="US/Pacific", periods=1
        )
        idx = DatetimeIndex(org, freq=freq, tz="US/Pacific")
        tm.assert_index_equal(idx, org)

    def test_datetimeindex_constructor_misc(self):
        arr = ["1/1/2005", "1/2/2005", "Jn 3, 2005", "2005-01-04"]
        msg = r"(\(')?Unknown datetime string format(:', 'Jn 3, 2005'\))?"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(arr)

        arr = ["1/1/2005", "1/2/2005", "1/3/2005", "2005-01-04"]
        idx1 = DatetimeIndex(arr)

        arr = [datetime(2005, 1, 1), "1/2/2005", "1/3/2005", "2005-01-04"]
        idx2 = DatetimeIndex(arr)

        arr = [Timestamp(datetime(2005, 1, 1)), "1/2/2005", "1/3/2005", "2005-01-04"]
        idx3 = DatetimeIndex(arr)

        arr = np.array(["1/1/2005", "1/2/2005", "1/3/2005", "2005-01-04"], dtype="O")
        idx4 = DatetimeIndex(arr)

        idx5 = DatetimeIndex(["12/05/2007", "25/01/2008"], dayfirst=True)
        idx6 = DatetimeIndex(
            ["2007/05/12", "2008/01/25"], dayfirst=False, yearfirst=True
        )
        tm.assert_index_equal(idx5, idx6)

        for other in [idx2, idx3, idx4]:
            assert (idx1.values == other.values).all()

    def test_dti_constructor_object_dtype_dayfirst_yearfirst_with_tz(self):
        # GH#55813
        val = "5/10/16"

        dfirst = Timestamp(2016, 10, 5, tz="US/Pacific")
        yfirst = Timestamp(2005, 10, 16, tz="US/Pacific")

        result1 = DatetimeIndex([val], tz="US/Pacific", dayfirst=True)
        expected1 = DatetimeIndex([dfirst])
        tm.assert_index_equal(result1, expected1)

        result2 = DatetimeIndex([val], tz="US/Pacific", yearfirst=True)
        expected2 = DatetimeIndex([yfirst])
        tm.assert_index_equal(result2, expected2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\offsets\test_offsets.py ===
"""
Tests of pandas.tseries.offsets
"""
from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas._libs.tslibs import (
    NaT,
    Timedelta,
    Timestamp,
    conversion,
    timezones,
)
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import (
    _get_offset,
    _offset_map,
    to_offset,
)
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import PerformanceWarning

from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    date_range,
)
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import WeekDay

from pandas.tseries import offsets
from pandas.tseries.offsets import (
    FY5253,
    BDay,
    BMonthEnd,
    BusinessHour,
    CustomBusinessDay,
    CustomBusinessHour,
    CustomBusinessMonthBegin,
    CustomBusinessMonthEnd,
    DateOffset,
    Easter,
    FY5253Quarter,
    LastWeekOfMonth,
    MonthBegin,
    Nano,
    Tick,
    Week,
    WeekOfMonth,
)

_ARITHMETIC_DATE_OFFSET = [
    "years",
    "months",
    "weeks",
    "days",
    "hours",
    "minutes",
    "seconds",
    "milliseconds",
    "microseconds",
]


def _create_offset(klass, value=1, normalize=False):
    # create instance from offset class
    if klass is FY5253:
        klass = klass(
            n=value,
            startingMonth=1,
            weekday=1,
            variation="last",
            normalize=normalize,
        )
    elif klass is FY5253Quarter:
        klass = klass(
            n=value,
            startingMonth=1,
            weekday=1,
            qtr_with_extra_week=1,
            variation="last",
            normalize=normalize,
        )
    elif klass is LastWeekOfMonth:
        klass = klass(n=value, weekday=5, normalize=normalize)
    elif klass is WeekOfMonth:
        klass = klass(n=value, week=1, weekday=5, normalize=normalize)
    elif klass is Week:
        klass = klass(n=value, weekday=5, normalize=normalize)
    elif klass is DateOffset:
        klass = klass(days=value, normalize=normalize)
    else:
        klass = klass(value, normalize=normalize)
    return klass


@pytest.fixture(
    params=[
        getattr(offsets, o)
        for o in offsets.__all__
        if issubclass(getattr(offsets, o), liboffsets.MonthOffset)
        and o != "MonthOffset"
    ]
)
def month_classes(request):
    """
    Fixture for month based datetime offsets available for a time series.
    """
    return request.param


@pytest.fixture(
    params=[
        getattr(offsets, o) for o in offsets.__all__ if o not in ("Tick", "BaseOffset")
    ]
)
def offset_types(request):
    """
    Fixture for all the datetime offsets available for a time series.
    """
    return request.param


@pytest.fixture
def dt():
    return Timestamp(datetime(2008, 1, 2))


@pytest.fixture
def expecteds():
    # executed value created by _create_offset
    # are applied to 2011/01/01 09:00 (Saturday)
    # used for .apply and .rollforward
    return {
        "Day": Timestamp("2011-01-02 09:00:00"),
        "DateOffset": Timestamp("2011-01-02 09:00:00"),
        "BusinessDay": Timestamp("2011-01-03 09:00:00"),
        "CustomBusinessDay": Timestamp("2011-01-03 09:00:00"),
        "CustomBusinessMonthEnd": Timestamp("2011-01-31 09:00:00"),
        "CustomBusinessMonthBegin": Timestamp("2011-01-03 09:00:00"),
        "MonthBegin": Timestamp("2011-02-01 09:00:00"),
        "BusinessMonthBegin": Timestamp("2011-01-03 09:00:00"),
        "MonthEnd": Timestamp("2011-01-31 09:00:00"),
        "SemiMonthEnd": Timestamp("2011-01-15 09:00:00"),
        "SemiMonthBegin": Timestamp("2011-01-15 09:00:00"),
        "BusinessMonthEnd": Timestamp("2011-01-31 09:00:00"),
        "YearBegin": Timestamp("2012-01-01 09:00:00"),
        "BYearBegin": Timestamp("2011-01-03 09:00:00"),
        "YearEnd": Timestamp("2011-12-31 09:00:00"),
        "BYearEnd": Timestamp("2011-12-30 09:00:00"),
        "QuarterBegin": Timestamp("2011-03-01 09:00:00"),
        "BQuarterBegin": Timestamp("2011-03-01 09:00:00"),
        "QuarterEnd": Timestamp("2011-03-31 09:00:00"),
        "BQuarterEnd": Timestamp("2011-03-31 09:00:00"),
        "BusinessHour": Timestamp("2011-01-03 10:00:00"),
        "CustomBusinessHour": Timestamp("2011-01-03 10:00:00"),
        "WeekOfMonth": Timestamp("2011-01-08 09:00:00"),
        "LastWeekOfMonth": Timestamp("2011-01-29 09:00:00"),
        "FY5253Quarter": Timestamp("2011-01-25 09:00:00"),
        "FY5253": Timestamp("2011-01-25 09:00:00"),
        "Week": Timestamp("2011-01-08 09:00:00"),
        "Easter": Timestamp("2011-04-24 09:00:00"),
        "Hour": Timestamp("2011-01-01 10:00:00"),
        "Minute": Timestamp("2011-01-01 09:01:00"),
        "Second": Timestamp("2011-01-01 09:00:01"),
        "Milli": Timestamp("2011-01-01 09:00:00.001000"),
        "Micro": Timestamp("2011-01-01 09:00:00.000001"),
        "Nano": Timestamp("2011-01-01T09:00:00.000000001"),
    }


class TestCommon:
    def test_immutable(self, offset_types):
        # GH#21341 check that __setattr__ raises
        offset = _create_offset(offset_types)
        msg = "objects is not writable|DateOffset objects are immutable"
        with pytest.raises(AttributeError, match=msg):
            offset.normalize = True
        with pytest.raises(AttributeError, match=msg):
            offset.n = 91

    def test_return_type(self, offset_types):
        offset = _create_offset(offset_types)

        # make sure that we are returning a Timestamp
        result = Timestamp("20080101") + offset
        assert isinstance(result, Timestamp)

        # make sure that we are returning NaT
        assert NaT + offset is NaT
        assert offset + NaT is NaT

        assert NaT - offset is NaT
        assert (-offset)._apply(NaT) is NaT

    def test_offset_n(self, offset_types):
        offset = _create_offset(offset_types)
        assert offset.n == 1

        neg_offset = offset * -1
        assert neg_offset.n == -1

        mul_offset = offset * 3
        assert mul_offset.n == 3

    def test_offset_timedelta64_arg(self, offset_types):
        # check that offset._validate_n raises TypeError on a timedelt64
        #  object
        off = _create_offset(offset_types)

        td64 = np.timedelta64(4567, "s")
        with pytest.raises(TypeError, match="argument must be an integer"):
            type(off)(n=td64, **off.kwds)

    def test_offset_mul_ndarray(self, offset_types):
        off = _create_offset(offset_types)

        expected = np.array([[off, off * 2], [off * 3, off * 4]])

        result = np.array([[1, 2], [3, 4]]) * off
        tm.assert_numpy_array_equal(result, expected)

        result = off * np.array([[1, 2], [3, 4]])
        tm.assert_numpy_array_equal(result, expected)

    def test_offset_freqstr(self, offset_types):
        offset = _create_offset(offset_types)

        freqstr = offset.freqstr
        if freqstr not in ("<Easter>", "<DateOffset: days=1>", "LWOM-SAT"):
            code = _get_offset(freqstr)
            assert offset.rule_code == code

    def _check_offsetfunc_works(self, offset, funcname, dt, expected, normalize=False):
        if normalize and issubclass(offset, Tick):
            # normalize=True disallowed for Tick subclasses GH#21427
            return

        offset_s = _create_offset(offset, normalize=normalize)
        func = getattr(offset_s, funcname)

        result = func(dt)
        assert isinstance(result, Timestamp)
        assert result == expected

        result = func(Timestamp(dt))
        assert isinstance(result, Timestamp)
        assert result == expected

        # see gh-14101
        ts = Timestamp(dt) + Nano(5)
        # test nanosecond is preserved
        with tm.assert_produces_warning(None):
            result = func(ts)

        assert isinstance(result, Timestamp)
        if normalize is False:
            assert result == expected + Nano(5)
        else:
            assert result == expected

        if isinstance(dt, np.datetime64):
            # test tz when input is datetime or Timestamp
            return

        for tz in [
            None,
            "UTC",
            "Asia/Tokyo",
            "US/Eastern",
            "dateutil/Asia/Tokyo",
            "dateutil/US/Pacific",
        ]:
            expected_localize = expected.tz_localize(tz)
            tz_obj = timezones.maybe_get_tz(tz)
            dt_tz = conversion.localize_pydatetime(dt, tz_obj)

            result = func(dt_tz)
            assert isinstance(result, Timestamp)
            assert result == expected_localize

            result = func(Timestamp(dt, tz=tz))
            assert isinstance(result, Timestamp)
            assert result == expected_localize

            # see gh-14101
            ts = Timestamp(dt, tz=tz) + Nano(5)
            # test nanosecond is preserved
            with tm.assert_produces_warning(None):
                result = func(ts)
            assert isinstance(result, Timestamp)
            if normalize is False:
                assert result == expected_localize + Nano(5)
            else:
                assert result == expected_localize

    def test_apply(self, offset_types, expecteds):
        sdt = datetime(2011, 1, 1, 9, 0)
        ndt = np.datetime64("2011-01-01 09:00")

        expected = expecteds[offset_types.__name__]
        expected_norm = Timestamp(expected.date())

        for dt in [sdt, ndt]:
            self._check_offsetfunc_works(offset_types, "_apply", dt, expected)

            self._check_offsetfunc_works(
                offset_types, "_apply", dt, expected_norm, normalize=True
            )

    def test_rollforward(self, offset_types, expecteds):
        expecteds = expecteds.copy()

        # result will not be changed if the target is on the offset
        no_changes = [
            "Day",
            "MonthBegin",
            "SemiMonthBegin",
            "YearBegin",
            "Week",
            "Hour",
            "Minute",
            "Second",
            "Milli",
            "Micro",
            "Nano",
            "DateOffset",
        ]
        for n in no_changes:
            expecteds[n] = Timestamp("2011/01/01 09:00")

        expecteds["BusinessHour"] = Timestamp("2011-01-03 09:00:00")
        expecteds["CustomBusinessHour"] = Timestamp("2011-01-03 09:00:00")

        # but be changed when normalize=True
        norm_expected = expecteds.copy()
        for k in norm_expected:
            norm_expected[k] = Timestamp(norm_expected[k].date())

        normalized = {
            "Day": Timestamp("2011-01-02 00:00:00"),
            "DateOffset": Timestamp("2011-01-02 00:00:00"),
            "MonthBegin": Timestamp("2011-02-01 00:00:00"),
            "SemiMonthBegin": Timestamp("2011-01-15 00:00:00"),
            "YearBegin": Timestamp("2012-01-01 00:00:00"),
            "Week": Timestamp("2011-01-08 00:00:00"),
            "Hour": Timestamp("2011-01-01 00:00:00"),
            "Minute": Timestamp("2011-01-01 00:00:00"),
            "Second": Timestamp("2011-01-01 00:00:00"),
            "Milli": Timestamp("2011-01-01 00:00:00"),
            "Micro": Timestamp("2011-01-01 00:00:00"),
        }
        norm_expected.update(normalized)

        sdt = datetime(2011, 1, 1, 9, 0)
        ndt = np.datetime64("2011-01-01 09:00")

        for dt in [sdt, ndt]:
            expected = expecteds[offset_types.__name__]
            self._check_offsetfunc_works(offset_types, "rollforward", dt, expected)
            expected = norm_expected[offset_types.__name__]
            self._check_offsetfunc_works(
                offset_types, "rollforward", dt, expected, normalize=True
            )

    def test_rollback(self, offset_types):
        expecteds = {
            "BusinessDay": Timestamp("2010-12-31 09:00:00"),
            "CustomBusinessDay": Timestamp("2010-12-31 09:00:00"),
            "CustomBusinessMonthEnd": Timestamp("2010-12-31 09:00:00"),
            "CustomBusinessMonthBegin": Timestamp("2010-12-01 09:00:00"),
            "BusinessMonthBegin": Timestamp("2010-12-01 09:00:00"),
            "MonthEnd": Timestamp("2010-12-31 09:00:00"),
            "SemiMonthEnd": Timestamp("2010-12-31 09:00:00"),
            "BusinessMonthEnd": Timestamp("2010-12-31 09:00:00"),
            "BYearBegin": Timestamp("2010-01-01 09:00:00"),
            "YearEnd": Timestamp("2010-12-31 09:00:00"),
            "BYearEnd": Timestamp("2010-12-31 09:00:00"),
            "QuarterBegin": Timestamp("2010-12-01 09:00:00"),
            "BQuarterBegin": Timestamp("2010-12-01 09:00:00"),
            "QuarterEnd": Timestamp("2010-12-31 09:00:00"),
            "BQuarterEnd": Timestamp("2010-12-31 09:00:00"),
            "BusinessHour": Timestamp("2010-12-31 17:00:00"),
            "CustomBusinessHour": Timestamp("2010-12-31 17:00:00"),
            "WeekOfMonth": Timestamp("2010-12-11 09:00:00"),
            "LastWeekOfMonth": Timestamp("2010-12-25 09:00:00"),
            "FY5253Quarter": Timestamp("2010-10-26 09:00:00"),
            "FY5253": Timestamp("2010-01-26 09:00:00"),
            "Easter": Timestamp("2010-04-04 09:00:00"),
        }

        # result will not be changed if the target is on the offset
        for n in [
            "Day",
            "MonthBegin",
            "SemiMonthBegin",
            "YearBegin",
            "Week",
            "Hour",
            "Minute",
            "Second",
            "Milli",
            "Micro",
            "Nano",
            "DateOffset",
        ]:
            expecteds[n] = Timestamp("2011/01/01 09:00")

        # but be changed when normalize=True
        norm_expected = expecteds.copy()
        for k in norm_expected:
            norm_expected[k] = Timestamp(norm_expected[k].date())

        normalized = {
            "Day": Timestamp("2010-12-31 00:00:00"),
            "DateOffset": Timestamp("2010-12-31 00:00:00"),
            "MonthBegin": Timestamp("2010-12-01 00:00:00"),
            "SemiMonthBegin": Timestamp("2010-12-15 00:00:00"),
            "YearBegin": Timestamp("2010-01-01 00:00:00"),
            "Week": Timestamp("2010-12-25 00:00:00"),
            "Hour": Timestamp("2011-01-01 00:00:00"),
            "Minute": Timestamp("2011-01-01 00:00:00"),
            "Second": Timestamp("2011-01-01 00:00:00"),
            "Milli": Timestamp("2011-01-01 00:00:00"),
            "Micro": Timestamp("2011-01-01 00:00:00"),
        }
        norm_expected.update(normalized)

        sdt = datetime(2011, 1, 1, 9, 0)
        ndt = np.datetime64("2011-01-01 09:00")

        for dt in [sdt, ndt]:
            expected = expecteds[offset_types.__name__]
            self._check_offsetfunc_works(offset_types, "rollback", dt, expected)

            expected = norm_expected[offset_types.__name__]
            self._check_offsetfunc_works(
                offset_types, "rollback", dt, expected, normalize=True
            )

    def test_is_on_offset(self, offset_types, expecteds):
        dt = expecteds[offset_types.__name__]
        offset_s = _create_offset(offset_types)
        assert offset_s.is_on_offset(dt)

        # when normalize=True, is_on_offset checks time is 00:00:00
        if issubclass(offset_types, Tick):
            # normalize=True disallowed for Tick subclasses GH#21427
            return
        offset_n = _create_offset(offset_types, normalize=True)
        assert not offset_n.is_on_offset(dt)

        if offset_types in (BusinessHour, CustomBusinessHour):
            # In default BusinessHour (9:00-17:00), normalized time
            # cannot be in business hour range
            return
        date = datetime(dt.year, dt.month, dt.day)
        assert offset_n.is_on_offset(date)

    def test_add(self, offset_types, tz_naive_fixture, expecteds):
        tz = tz_naive_fixture
        dt = datetime(2011, 1, 1, 9, 0)

        offset_s = _create_offset(offset_types)
        expected = expecteds[offset_types.__name__]

        result_dt = dt + offset_s
        result_ts = Timestamp(dt) + offset_s
        for result in [result_dt, result_ts]:
            assert isinstance(result, Timestamp)
            assert result == expected

        expected_localize = expected.tz_localize(tz)
        result = Timestamp(dt, tz=tz) + offset_s
        assert isinstance(result, Timestamp)
        assert result == expected_localize

        # normalize=True, disallowed for Tick subclasses GH#21427
        if issubclass(offset_types, Tick):
            return
        offset_s = _create_offset(offset_types, normalize=True)
        expected = Timestamp(expected.date())

        result_dt = dt + offset_s
        result_ts = Timestamp(dt) + offset_s
        for result in [result_dt, result_ts]:
            assert isinstance(result, Timestamp)
            assert result == expected

        expected_localize = expected.tz_localize(tz)
        result = Timestamp(dt, tz=tz) + offset_s
        assert isinstance(result, Timestamp)
        assert result == expected_localize

    def test_add_empty_datetimeindex(self, offset_types, tz_naive_fixture):
        # GH#12724, GH#30336
        offset_s = _create_offset(offset_types)

        dti = DatetimeIndex([], tz=tz_naive_fixture).as_unit("ns")

        warn = None
        if isinstance(
            offset_s,
            (
                Easter,
                WeekOfMonth,
                LastWeekOfMonth,
                CustomBusinessDay,
                BusinessHour,
                CustomBusinessHour,
                CustomBusinessMonthBegin,
                CustomBusinessMonthEnd,
                FY5253,
                FY5253Quarter,
            ),
        ):
            # We don't have an optimized apply_index
            warn = PerformanceWarning

        # stacklevel checking is slow, and we have ~800 of variants of this
        #  test, so let's only check the stacklevel in a subset of them
        check_stacklevel = tz_naive_fixture is None
        with tm.assert_produces_warning(warn, check_stacklevel=check_stacklevel):
            result = dti + offset_s
        tm.assert_index_equal(result, dti)
        with tm.assert_produces_warning(warn, check_stacklevel=check_stacklevel):
            result = offset_s + dti
        tm.assert_index_equal(result, dti)

        dta = dti._data
        with tm.assert_produces_warning(warn, check_stacklevel=check_stacklevel):
            result = dta + offset_s
        tm.assert_equal(result, dta)
        with tm.assert_produces_warning(warn, check_stacklevel=check_stacklevel):
            result = offset_s + dta
        tm.assert_equal(result, dta)

    def test_pickle_roundtrip(self, offset_types):
        off = _create_offset(offset_types)
        res = tm.round_trip_pickle(off)
        assert off == res
        if type(off) is not DateOffset:
            for attr in off._attributes:
                if attr == "calendar":
                    # np.busdaycalendar __eq__ will return False;
                    #  we check holidays and weekmask attrs so are OK
                    continue
                # Make sure nothings got lost from _params (which __eq__) is based on
                assert getattr(off, attr) == getattr(res, attr)

    def test_pickle_dateoffset_odd_inputs(self):
        # GH#34511
        off = DateOffset(months=12)
        res = tm.round_trip_pickle(off)
        assert off == res

        base_dt = datetime(2020, 1, 1)
        assert base_dt + off == base_dt + res

    def test_offsets_hashable(self, offset_types):
        # GH: 37267
        off = _create_offset(offset_types)
        assert hash(off) is not None

    # TODO: belongs in arithmetic tests?
    @pytest.mark.filterwarnings(
        "ignore:Non-vectorized DateOffset being applied to Series or DatetimeIndex"
    )
    @pytest.mark.parametrize("unit", ["s", "ms", "us"])
    def test_add_dt64_ndarray_non_nano(self, offset_types, unit):
        # check that the result with non-nano matches nano
        off = _create_offset(offset_types)

        dti = date_range("2016-01-01", periods=35, freq="D", unit=unit)

        result = (dti + off)._with_freq(None)

        exp_unit = unit
        if isinstance(off, Tick) and off._creso > dti._data._creso:
            # cast to higher reso like we would with Timedelta scalar
            exp_unit = Timedelta(off).unit
        # TODO(GH#55564): as_unit will be unnecessary
        expected = DatetimeIndex([x + off for x in dti]).as_unit(exp_unit)

        tm.assert_index_equal(result, expected)


class TestDateOffset:
    def setup_method(self):
        _offset_map.clear()

    def test_repr(self):
        repr(DateOffset())
        repr(DateOffset(2))
        repr(2 * DateOffset())
        repr(2 * DateOffset(months=2))

    def test_mul(self):
        assert DateOffset(2) == 2 * DateOffset(1)
        assert DateOffset(2) == DateOffset(1) * 2

    @pytest.mark.parametrize("kwd", sorted(liboffsets._relativedelta_kwds))
    def test_constructor(self, kwd, request):
        if kwd == "millisecond":
            request.applymarker(
                pytest.mark.xfail(
                    raises=NotImplementedError,
                    reason="Constructing DateOffset object with `millisecond` is not "
                    "yet supported.",
                )
            )
        offset = DateOffset(**{kwd: 2})
        assert offset.kwds == {kwd: 2}
        assert getattr(offset, kwd) == 2

    def test_default_constructor(self, dt):
        assert (dt + DateOffset(2)) == datetime(2008, 1, 4)

    def test_is_anchored(self):
        msg = "DateOffset.is_anchored is deprecated "

        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert not DateOffset(2).is_anchored()
            assert DateOffset(1).is_anchored()

    def test_copy(self):
        assert DateOffset(months=2).copy() == DateOffset(months=2)
        assert DateOffset(milliseconds=1).copy() == DateOffset(milliseconds=1)

    @pytest.mark.parametrize(
        "arithmatic_offset_type, expected",
        zip(
            _ARITHMETIC_DATE_OFFSET,
            [
                "2009-01-02",
                "2008-02-02",
                "2008-01-09",
                "2008-01-03",
                "2008-01-02 01:00:00",
                "2008-01-02 00:01:00",
                "2008-01-02 00:00:01",
                "2008-01-02 00:00:00.001000000",
                "2008-01-02 00:00:00.000001000",
            ],
        ),
    )
    def test_add(self, arithmatic_offset_type, expected, dt):
        assert DateOffset(**{arithmatic_offset_type: 1}) + dt == Timestamp(expected)
        assert dt + DateOffset(**{arithmatic_offset_type: 1}) == Timestamp(expected)

    @pytest.mark.parametrize(
        "arithmatic_offset_type, expected",
        zip(
            _ARITHMETIC_DATE_OFFSET,
            [
                "2007-01-02",
                "2007-12-02",
                "2007-12-26",
                "2008-01-01",
                "2008-01-01 23:00:00",
                "2008-01-01 23:59:00",
                "2008-01-01 23:59:59",
                "2008-01-01 23:59:59.999000000",
                "2008-01-01 23:59:59.999999000",
            ],
        ),
    )
    def test_sub(self, arithmatic_offset_type, expected, dt):
        assert dt - DateOffset(**{arithmatic_offset_type: 1}) == Timestamp(expected)
        with pytest.raises(TypeError, match="Cannot subtract datetime from offset"):
            DateOffset(**{arithmatic_offset_type: 1}) - dt

    @pytest.mark.parametrize(
        "arithmatic_offset_type, n, expected",
        zip(
            _ARITHMETIC_DATE_OFFSET,
            range(1, 10),
            [
                "2009-01-02",
                "2008-03-02",
                "2008-01-23",
                "2008-01-06",
                "2008-01-02 05:00:00",
                "2008-01-02 00:06:00",
                "2008-01-02 00:00:07",
                "2008-01-02 00:00:00.008000000",
                "2008-01-02 00:00:00.000009000",
            ],
        ),
    )
    def test_mul_add(self, arithmatic_offset_type, n, expected, dt):
        assert DateOffset(**{arithmatic_offset_type: 1}) * n + dt == Timestamp(expected)
        assert n * DateOffset(**{arithmatic_offset_type: 1}) + dt == Timestamp(expected)
        assert dt + DateOffset(**{arithmatic_offset_type: 1}) * n == Timestamp(expected)
        assert dt + n * DateOffset(**{arithmatic_offset_type: 1}) == Timestamp(expected)

    @pytest.mark.parametrize(
        "arithmatic_offset_type, n, expected",
        zip(
            _ARITHMETIC_DATE_OFFSET,
            range(1, 10),
            [
                "2007-01-02",
                "2007-11-02",
                "2007-12-12",
                "2007-12-29",
                "2008-01-01 19:00:00",
                "2008-01-01 23:54:00",
                "2008-01-01 23:59:53",
                "2008-01-01 23:59:59.992000000",
                "2008-01-01 23:59:59.999991000",
            ],
        ),
    )
    def test_mul_sub(self, arithmatic_offset_type, n, expected, dt):
        assert dt - DateOffset(**{arithmatic_offset_type: 1}) * n == Timestamp(expected)
        assert dt - n * DateOffset(**{arithmatic_offset_type: 1}) == Timestamp(expected)

    def test_leap_year(self):
        d = datetime(2008, 1, 31)
        assert (d + DateOffset(months=1)) == datetime(2008, 2, 29)

    def test_eq(self):
        offset1 = DateOffset(days=1)
        offset2 = DateOffset(days=365)

        assert offset1 != offset2

        assert DateOffset(milliseconds=3) != DateOffset(milliseconds=7)

    @pytest.mark.parametrize(
        "offset_kwargs, expected_arg",
        [
            ({"microseconds": 1, "milliseconds": 1}, "2022-01-01 00:00:00.001001"),
            ({"seconds": 1, "milliseconds": 1}, "2022-01-01 00:00:01.001"),
            ({"minutes": 1, "milliseconds": 1}, "2022-01-01 00:01:00.001"),
            ({"hours": 1, "milliseconds": 1}, "2022-01-01 01:00:00.001"),
            ({"days": 1, "milliseconds": 1}, "2022-01-02 00:00:00.001"),
            ({"weeks": 1, "milliseconds": 1}, "2022-01-08 00:00:00.001"),
            ({"months": 1, "milliseconds": 1}, "2022-02-01 00:00:00.001"),
            ({"years": 1, "milliseconds": 1}, "2023-01-01 00:00:00.001"),
        ],
    )
    def test_milliseconds_combination(self, offset_kwargs, expected_arg):
        # GH 49897
        offset = DateOffset(**offset_kwargs)
        ts = Timestamp("2022-01-01")
        result = ts + offset
        expected = Timestamp(expected_arg)

        assert result == expected

    def test_offset_invalid_arguments(self):
        msg = "^Invalid argument/s or bad combination of arguments"
        with pytest.raises(ValueError, match=msg):
            DateOffset(picoseconds=1)


class TestOffsetNames:
    def test_get_offset_name(self):
        assert BDay().freqstr == "B"
        assert BDay(2).freqstr == "2B"
        assert BMonthEnd().freqstr == "BME"
        assert Week(weekday=0).freqstr == "W-MON"
        assert Week(weekday=1).freqstr == "W-TUE"
        assert Week(weekday=2).freqstr == "W-WED"
        assert Week(weekday=3).freqstr == "W-THU"
        assert Week(weekday=4).freqstr == "W-FRI"

        assert LastWeekOfMonth(weekday=WeekDay.SUN).freqstr == "LWOM-SUN"


def test_get_offset():
    with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
        _get_offset("gibberish")
    with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
        _get_offset("QS-JAN-B")

    pairs = [
        ("B", BDay()),
        ("b", BDay()),
        ("bme", BMonthEnd()),
        ("Bme", BMonthEnd()),
        ("W-MON", Week(weekday=0)),
        ("W-TUE", Week(weekday=1)),
        ("W-WED", Week(weekday=2)),
        ("W-THU", Week(weekday=3)),
        ("W-FRI", Week(weekday=4)),
    ]

    for name, expected in pairs:
        offset = _get_offset(name)
        assert offset == expected, (
            f"Expected {repr(name)} to yield {repr(expected)} "
            f"(actual: {repr(offset)})"
        )


def test_get_offset_legacy():
    pairs = [("w@Sat", Week(weekday=5))]
    for name, expected in pairs:
        with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
            _get_offset(name)


class TestOffsetAliases:
    def setup_method(self):
        _offset_map.clear()

    def test_alias_equality(self):
        for k, v in _offset_map.items():
            if v is None:
                continue
            assert k == v.copy()

    def test_rule_code(self):
        lst = ["ME", "MS", "BME", "BMS", "D", "B", "h", "min", "s", "ms", "us"]
        for k in lst:
            assert k == _get_offset(k).rule_code
            # should be cached - this is kind of an internals test...
            assert k in _offset_map
            assert k == (_get_offset(k) * 3).rule_code

        suffix_lst = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
        base = "W"
        for v in suffix_lst:
            alias = "-".join([base, v])
            assert alias == _get_offset(alias).rule_code
            assert alias == (_get_offset(alias) * 5).rule_code

        suffix_lst = [
            "JAN",
            "FEB",
            "MAR",
            "APR",
            "MAY",
            "JUN",
            "JUL",
            "AUG",
            "SEP",
            "OCT",
            "NOV",
            "DEC",
        ]
        base_lst = ["YE", "YS", "BYE", "BYS", "QE", "QS", "BQE", "BQS"]
        for base in base_lst:
            for v in suffix_lst:
                alias = "-".join([base, v])
                assert alias == _get_offset(alias).rule_code
                assert alias == (_get_offset(alias) * 5).rule_code


def test_freq_offsets():
    off = BDay(1, offset=timedelta(0, 1800))
    assert off.freqstr == "B+30Min"

    off = BDay(1, offset=timedelta(0, -1800))
    assert off.freqstr == "B-30Min"


class TestReprNames:
    def test_str_for_named_is_name(self):
        # look at all the amazing combinations!
        month_prefixes = ["YE", "YS", "BYE", "BYS", "QE", "BQE", "BQS", "QS"]
        names = [
            prefix + "-" + month
            for prefix in month_prefixes
            for month in [
                "JAN",
                "FEB",
                "MAR",
                "APR",
                "MAY",
                "JUN",
                "JUL",
                "AUG",
                "SEP",
                "OCT",
                "NOV",
                "DEC",
            ]
        ]
        days = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
        names += ["W-" + day for day in days]
        names += ["WOM-" + week + day for week in ("1", "2", "3", "4") for day in days]
        _offset_map.clear()
        for name in names:
            offset = _get_offset(name)
            assert offset.freqstr == name


# ---------------------------------------------------------------------


def test_valid_default_arguments(offset_types):
    # GH#19142 check that the calling the constructors without passing
    # any keyword arguments produce valid offsets
    cls = offset_types
    cls()


@pytest.mark.parametrize("kwd", sorted(liboffsets._relativedelta_kwds))
def test_valid_month_attributes(kwd, month_classes):
    # GH#18226
    cls = month_classes
    # check that we cannot create e.g. MonthEnd(weeks=3)
    msg = rf"__init__\(\) got an unexpected keyword argument '{kwd}'"
    with pytest.raises(TypeError, match=msg):
        cls(**{kwd: 3})


def test_month_offset_name(month_classes):
    # GH#33757 off.name with n != 1 should not raise AttributeError
    obj = month_classes(1)
    obj2 = month_classes(2)
    assert obj2.name == obj.name


@pytest.mark.parametrize("kwd", sorted(liboffsets._relativedelta_kwds))
def test_valid_relativedelta_kwargs(kwd, request):
    if kwd == "millisecond":
        request.applymarker(
            pytest.mark.xfail(
                raises=NotImplementedError,
                reason="Constructing DateOffset object with `millisecond` is not "
                "yet supported.",
            )
        )
    # Check that all the arguments specified in liboffsets._relativedelta_kwds
    # are in fact valid relativedelta keyword args
    DateOffset(**{kwd: 1})


@pytest.mark.parametrize("kwd", sorted(liboffsets._relativedelta_kwds))
def test_valid_tick_attributes(kwd, tick_classes):
    # GH#18226
    cls = tick_classes
    # check that we cannot create e.g. Hour(weeks=3)
    msg = rf"__init__\(\) got an unexpected keyword argument '{kwd}'"
    with pytest.raises(TypeError, match=msg):
        cls(**{kwd: 3})


def test_validate_n_error():
    with pytest.raises(TypeError, match="argument must be an integer"):
        DateOffset(n="Doh!")

    with pytest.raises(TypeError, match="argument must be an integer"):
        MonthBegin(n=timedelta(1))

    with pytest.raises(TypeError, match="argument must be an integer"):
        BDay(n=np.array([1, 2], dtype=np.int64))


def test_require_integers(offset_types):
    cls = offset_types
    with pytest.raises(ValueError, match="argument must be an integer"):
        cls(n=1.5)


def test_tick_normalize_raises(tick_classes):
    # check that trying to create a Tick object with normalize=True raises
    # GH#21427
    cls = tick_classes
    msg = "Tick offset with `normalize=True` are not allowed."
    with pytest.raises(ValueError, match=msg):
        cls(n=3, normalize=True)


@pytest.mark.parametrize(
    "offset_kwargs, expected_arg",
    [
        ({"nanoseconds": 1}, "1970-01-01 00:00:00.000000001"),
        ({"nanoseconds": 5}, "1970-01-01 00:00:00.000000005"),
        ({"nanoseconds": -1}, "1969-12-31 23:59:59.999999999"),
        ({"microseconds": 1}, "1970-01-01 00:00:00.000001"),
        ({"microseconds": -1}, "1969-12-31 23:59:59.999999"),
        ({"seconds": 1}, "1970-01-01 00:00:01"),
        ({"seconds": -1}, "1969-12-31 23:59:59"),
        ({"minutes": 1}, "1970-01-01 00:01:00"),
        ({"minutes": -1}, "1969-12-31 23:59:00"),
        ({"hours": 1}, "1970-01-01 01:00:00"),
        ({"hours": -1}, "1969-12-31 23:00:00"),
        ({"days": 1}, "1970-01-02 00:00:00"),
        ({"days": -1}, "1969-12-31 00:00:00"),
        ({"weeks": 1}, "1970-01-08 00:00:00"),
        ({"weeks": -1}, "1969-12-25 00:00:00"),
        ({"months": 1}, "1970-02-01 00:00:00"),
        ({"months": -1}, "1969-12-01 00:00:00"),
        ({"years": 1}, "1971-01-01 00:00:00"),
        ({"years": -1}, "1969-01-01 00:00:00"),
    ],
)
def test_dateoffset_add_sub(offset_kwargs, expected_arg):
    offset = DateOffset(**offset_kwargs)
    ts = Timestamp(0)
    result = ts + offset
    expected = Timestamp(expected_arg)
    assert result == expected
    result -= offset
    assert result == ts
    result = offset + ts
    assert result == expected


def test_dateoffset_add_sub_timestamp_with_nano():
    offset = DateOffset(minutes=2, nanoseconds=9)
    ts = Timestamp(4)
    result = ts + offset
    expected = Timestamp("1970-01-01 00:02:00.000000013")
    assert result == expected
    result -= offset
    assert result == ts
    result = offset + ts
    assert result == expected

    offset2 = DateOffset(minutes=2, nanoseconds=9, hour=1)
    assert offset2._use_relativedelta
    with tm.assert_produces_warning(None):
        # no warning about Discarding nonzero nanoseconds
        result2 = ts + offset2
    expected2 = Timestamp("1970-01-01 01:02:00.000000013")
    assert result2 == expected2


@pytest.mark.parametrize(
    "attribute",
    [
        "hours",
        "days",
        "weeks",
        "months",
        "years",
    ],
)
def test_dateoffset_immutable(attribute):
    offset = DateOffset(**{attribute: 0})
    msg = "DateOffset objects are immutable"
    with pytest.raises(AttributeError, match=msg):
        setattr(offset, attribute, 5)


def test_dateoffset_misc():
    oset = offsets.DateOffset(months=2, days=4)
    # it works
    oset.freqstr

    assert not offsets.DateOffset(months=2) == 2


@pytest.mark.parametrize("n", [-1, 1, 3])
def test_construct_int_arg_no_kwargs_assumed_days(n):
    # GH 45890, 45643
    offset = DateOffset(n)
    assert offset._offset == timedelta(1)
    result = Timestamp(2022, 1, 2) + offset
    expected = Timestamp(2022, 1, 2 + n)
    assert result == expected


@pytest.mark.parametrize(
    "offset, expected",
    [
        (
            DateOffset(minutes=7, nanoseconds=18),
            Timestamp("2022-01-01 00:07:00.000000018"),
        ),
        (DateOffset(nanoseconds=3), Timestamp("2022-01-01 00:00:00.000000003")),
    ],
)
def test_dateoffset_add_sub_timestamp_series_with_nano(offset, expected):
    # GH 47856
    start_time = Timestamp("2022-01-01")
    teststamp = start_time
    testseries = Series([start_time])
    testseries = testseries + offset
    assert testseries[0] == expected
    testseries -= offset
    assert testseries[0] == teststamp
    testseries = offset + testseries
    assert testseries[0] == expected


@pytest.mark.parametrize(
    "n_months, scaling_factor, start_timestamp, expected_timestamp",
    [
        (1, 2, "2020-01-30", "2020-03-30"),
        (2, 1, "2020-01-30", "2020-03-30"),
        (1, 0, "2020-01-30", "2020-01-30"),
        (2, 0, "2020-01-30", "2020-01-30"),
        (1, -1, "2020-01-30", "2019-12-30"),
        (2, -1, "2020-01-30", "2019-11-30"),
    ],
)
def test_offset_multiplication(
    n_months, scaling_factor, start_timestamp, expected_timestamp
):
    # GH 47953
    mo1 = DateOffset(months=n_months)

    startscalar = Timestamp(start_timestamp)
    startarray = Series([startscalar])

    resultscalar = startscalar + (mo1 * scaling_factor)
    resultarray = startarray + (mo1 * scaling_factor)

    expectedscalar = Timestamp(expected_timestamp)
    expectedarray = Series([expectedscalar])
    assert resultscalar == expectedscalar

    tm.assert_series_equal(resultarray, expectedarray)


def test_dateoffset_operations_on_dataframes():
    # GH 47953
    df = DataFrame({"T": [Timestamp("2019-04-30")], "D": [DateOffset(months=1)]})
    frameresult1 = df["T"] + 26 * df["D"]
    df2 = DataFrame(
        {
            "T": [Timestamp("2019-04-30"), Timestamp("2019-04-30")],
            "D": [DateOffset(months=1), DateOffset(months=1)],
        }
    )
    expecteddate = Timestamp("2021-06-30")
    with tm.assert_produces_warning(PerformanceWarning):
        frameresult2 = df2["T"] + 26 * df2["D"]

    assert frameresult1[0] == expecteddate
    assert frameresult2[0] == expecteddate


def test_is_yqm_start_end():
    freq_m = to_offset("ME")
    bm = to_offset("BME")
    qfeb = to_offset("QE-FEB")
    qsfeb = to_offset("QS-FEB")
    bq = to_offset("BQE")
    bqs_apr = to_offset("BQS-APR")
    as_nov = to_offset("YS-NOV")

    tests = [
        (freq_m.is_month_start(Timestamp("2013-06-01")), 1),
        (bm.is_month_start(Timestamp("2013-06-01")), 0),
        (freq_m.is_month_start(Timestamp("2013-06-03")), 0),
        (bm.is_month_start(Timestamp("2013-06-03")), 1),
        (qfeb.is_month_end(Timestamp("2013-02-28")), 1),
        (qfeb.is_quarter_end(Timestamp("2013-02-28")), 1),
        (qfeb.is_year_end(Timestamp("2013-02-28")), 1),
        (qfeb.is_month_start(Timestamp("2013-03-01")), 1),
        (qfeb.is_quarter_start(Timestamp("2013-03-01")), 1),
        (qfeb.is_year_start(Timestamp("2013-03-01")), 1),
        (qsfeb.is_month_end(Timestamp("2013-03-31")), 1),
        (qsfeb.is_quarter_end(Timestamp("2013-03-31")), 0),
        (qsfeb.is_year_end(Timestamp("2013-03-31")), 0),
        (qsfeb.is_month_start(Timestamp("2013-02-01")), 1),
        (qsfeb.is_quarter_start(Timestamp("2013-02-01")), 1),
        (qsfeb.is_year_start(Timestamp("2013-02-01")), 1),
        (bq.is_month_end(Timestamp("2013-06-30")), 0),
        (bq.is_quarter_end(Timestamp("2013-06-30")), 0),
        (bq.is_year_end(Timestamp("2013-06-30")), 0),
        (bq.is_month_end(Timestamp("2013-06-28")), 1),
        (bq.is_quarter_end(Timestamp("2013-06-28")), 1),
        (bq.is_year_end(Timestamp("2013-06-28")), 0),
        (bqs_apr.is_month_end(Timestamp("2013-06-30")), 0),
        (bqs_apr.is_quarter_end(Timestamp("2013-06-30")), 0),
        (bqs_apr.is_year_end(Timestamp("2013-06-30")), 0),
        (bqs_apr.is_month_end(Timestamp("2013-06-28")), 1),
        (bqs_apr.is_quarter_end(Timestamp("2013-06-28")), 1),
        (bqs_apr.is_year_end(Timestamp("2013-03-29")), 1),
        (as_nov.is_year_start(Timestamp("2013-11-01")), 1),
        (as_nov.is_year_end(Timestamp("2013-10-31")), 1),
        (Timestamp("2012-02-01").days_in_month, 29),
        (Timestamp("2013-02-01").days_in_month, 28),
    ]

    for ts, value in tests:
        assert ts == value

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\reflection.py ===
# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

# This code is meant to work on Python 2.4 and above only.

"""Contains a metaclass and helper functions used to create
protocol message classes from Descriptor objects at runtime.

Recall that a metaclass is the "type" of a class.
(A class is to a metaclass what an instance is to a class.)

In this case, we use the GeneratedProtocolMessageType metaclass
to inject all the useful functionality into the classes
output by the protocol compiler at compile-time.

The upshot of all this is that the real implementation
details for ALL pure-Python protocol buffers are *here in
this file*.
"""

__author__ = 'robinson@google.com (Will Robinson)'

import warnings

from google.protobuf import message_factory
from google.protobuf import symbol_database

# The type of all Message classes.
# Part of the public interface, but normally only used by message factories.
GeneratedProtocolMessageType = message_factory._GENERATED_PROTOCOL_MESSAGE_TYPE

MESSAGE_CLASS_CACHE = {}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_pyinstaller\hook-numpy.py ===
"""This hook should collect all binary files and any hidden modules that numpy
needs.

Our (some-what inadequate) docs for writing PyInstaller hooks are kept here:
https://pyinstaller.readthedocs.io/en/stable/hooks.html

"""
from PyInstaller.compat import is_pure_conda
from PyInstaller.utils.hooks import collect_dynamic_libs

# Collect all DLLs inside numpy's installation folder, dump them into built
# app's root.
binaries = collect_dynamic_libs("numpy", ".")

# If using Conda without any non-conda virtual environment manager:
if is_pure_conda:
    # Assume running the NumPy from Conda-forge and collect it's DLLs from the
    # communal Conda bin directory. DLLs from NumPy's dependencies must also be
    # collected to capture MKL, OpenBlas, OpenMP, etc.
    from PyInstaller.utils.hooks import conda_support
    datas = conda_support.collect_dynamic_libs("numpy", dependencies=True)

# Submodules PyInstaller cannot detect.  `_dtype_ctypes` is only imported
# from C and `_multiarray_tests` is used in tests (which are not packed).
hiddenimports = ['numpy._core._dtype_ctypes', 'numpy._core._multiarray_tests']

# Remove testing and building code and packages that are referenced throughout
# NumPy but are not really dependencies.
excludedimports = [
    "scipy",
    "pytest",
    "f2py",
    "setuptools",
    "distutils",
    "numpy.distutils",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\workbook\function_group.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Sequence,
    String,
    Integer,
)

class FunctionGroup(Serialisable):

    tagname = "functionGroup"

    name = String()

    def __init__(self,
                 name=None,
                ):
        self.name = name


class FunctionGroupList(Serialisable):

    tagname = "functionGroups"

    builtInGroupCount = Integer(allow_none=True)
    functionGroup = Sequence(expected_type=FunctionGroup, allow_none=True)

    __elements__ = ('functionGroup',)

    def __init__(self,
                 builtInGroupCount=16,
                 functionGroup=(),
                ):
        self.builtInGroupCount = builtInGroupCount
        self.functionGroup = functionGroup

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\timedelta\test_arithmetic.py ===
"""
Tests for scalar Timedelta arithmetic ops
"""
from datetime import (
    datetime,
    timedelta,
)
import operator

import numpy as np
import pytest

from pandas.errors import OutOfBoundsTimedelta

import pandas as pd
from pandas import (
    NaT,
    Timedelta,
    Timestamp,
    offsets,
)
import pandas._testing as tm
from pandas.core import ops


class TestTimedeltaAdditionSubtraction:
    """
    Tests for Timedelta methods:

        __add__, __radd__,
        __sub__, __rsub__
    """

    @pytest.mark.parametrize(
        "ten_seconds",
        [
            Timedelta(10, unit="s"),
            timedelta(seconds=10),
            np.timedelta64(10, "s"),
            np.timedelta64(10000000000, "ns"),
            offsets.Second(10),
        ],
    )
    def test_td_add_sub_ten_seconds(self, ten_seconds):
        # GH#6808
        base = Timestamp("20130101 09:01:12.123456")
        expected_add = Timestamp("20130101 09:01:22.123456")
        expected_sub = Timestamp("20130101 09:01:02.123456")

        result = base + ten_seconds
        assert result == expected_add

        result = base - ten_seconds
        assert result == expected_sub

    @pytest.mark.parametrize(
        "one_day_ten_secs",
        [
            Timedelta("1 day, 00:00:10"),
            Timedelta("1 days, 00:00:10"),
            timedelta(days=1, seconds=10),
            np.timedelta64(1, "D") + np.timedelta64(10, "s"),
            offsets.Day() + offsets.Second(10),
        ],
    )
    def test_td_add_sub_one_day_ten_seconds(self, one_day_ten_secs):
        # GH#6808
        base = Timestamp("20130102 09:01:12.123456")
        expected_add = Timestamp("20130103 09:01:22.123456")
        expected_sub = Timestamp("20130101 09:01:02.123456")

        result = base + one_day_ten_secs
        assert result == expected_add

        result = base - one_day_ten_secs
        assert result == expected_sub

    @pytest.mark.parametrize("op", [operator.add, ops.radd])
    def test_td_add_datetimelike_scalar(self, op):
        # GH#19738
        td = Timedelta(10, unit="d")

        result = op(td, datetime(2016, 1, 1))
        if op is operator.add:
            # datetime + Timedelta does _not_ call Timedelta.__radd__,
            # so we get a datetime back instead of a Timestamp
            assert isinstance(result, Timestamp)
        assert result == Timestamp(2016, 1, 11)

        result = op(td, Timestamp("2018-01-12 18:09"))
        assert isinstance(result, Timestamp)
        assert result == Timestamp("2018-01-22 18:09")

        result = op(td, np.datetime64("2018-01-12"))
        assert isinstance(result, Timestamp)
        assert result == Timestamp("2018-01-22")

        result = op(td, NaT)
        assert result is NaT

    def test_td_add_timestamp_overflow(self):
        ts = Timestamp("1700-01-01").as_unit("ns")
        msg = "Cannot cast 259987 from D to 'ns' without overflow."
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            ts + Timedelta(13 * 19999, unit="D")

        msg = "Cannot cast 259987 days 00:00:00 to unit='ns' without overflow"
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            ts + timedelta(days=13 * 19999)

    @pytest.mark.parametrize("op", [operator.add, ops.radd])
    def test_td_add_td(self, op):
        td = Timedelta(10, unit="d")

        result = op(td, Timedelta(days=10))
        assert isinstance(result, Timedelta)
        assert result == Timedelta(days=20)

    @pytest.mark.parametrize("op", [operator.add, ops.radd])
    def test_td_add_pytimedelta(self, op):
        td = Timedelta(10, unit="d")
        result = op(td, timedelta(days=9))
        assert isinstance(result, Timedelta)
        assert result == Timedelta(days=19)

    @pytest.mark.parametrize("op", [operator.add, ops.radd])
    def test_td_add_timedelta64(self, op):
        td = Timedelta(10, unit="d")
        result = op(td, np.timedelta64(-4, "D"))
        assert isinstance(result, Timedelta)
        assert result == Timedelta(days=6)

    @pytest.mark.parametrize("op", [operator.add, ops.radd])
    def test_td_add_offset(self, op):
        td = Timedelta(10, unit="d")

        result = op(td, offsets.Hour(6))
        assert isinstance(result, Timedelta)
        assert result == Timedelta(days=10, hours=6)

    def test_td_sub_td(self):
        td = Timedelta(10, unit="d")
        expected = Timedelta(0, unit="ns")
        result = td - td
        assert isinstance(result, Timedelta)
        assert result == expected

    def test_td_sub_pytimedelta(self):
        td = Timedelta(10, unit="d")
        expected = Timedelta(0, unit="ns")

        result = td - td.to_pytimedelta()
        assert isinstance(result, Timedelta)
        assert result == expected

        result = td.to_pytimedelta() - td
        assert isinstance(result, Timedelta)
        assert result == expected

    def test_td_sub_timedelta64(self):
        td = Timedelta(10, unit="d")
        expected = Timedelta(0, unit="ns")

        result = td - td.to_timedelta64()
        assert isinstance(result, Timedelta)
        assert result == expected

        result = td.to_timedelta64() - td
        assert isinstance(result, Timedelta)
        assert result == expected

    def test_td_sub_nat(self):
        # In this context pd.NaT is treated as timedelta-like
        td = Timedelta(10, unit="d")
        result = td - NaT
        assert result is NaT

    def test_td_sub_td64_nat(self):
        td = Timedelta(10, unit="d")
        td_nat = np.timedelta64("NaT")

        result = td - td_nat
        assert result is NaT

        result = td_nat - td
        assert result is NaT

    def test_td_sub_offset(self):
        td = Timedelta(10, unit="d")
        result = td - offsets.Hour(1)
        assert isinstance(result, Timedelta)
        assert result == Timedelta(239, unit="h")

    def test_td_add_sub_numeric_raises(self):
        td = Timedelta(10, unit="d")
        msg = "unsupported operand type"
        for other in [2, 2.0, np.int64(2), np.float64(2)]:
            with pytest.raises(TypeError, match=msg):
                td + other
            with pytest.raises(TypeError, match=msg):
                other + td
            with pytest.raises(TypeError, match=msg):
                td - other
            with pytest.raises(TypeError, match=msg):
                other - td

    def test_td_add_sub_int_ndarray(self):
        td = Timedelta("1 day")
        other = np.array([1])

        msg = r"unsupported operand type\(s\) for \+: 'Timedelta' and 'int'"
        with pytest.raises(TypeError, match=msg):
            td + np.array([1])

        msg = "|".join(
            [
                (
                    r"unsupported operand type\(s\) for \+: 'numpy.ndarray' "
                    "and 'Timedelta'"
                ),
                # This message goes on to say "Please do not rely on this error;
                #  it may not be given on all Python implementations"
                "Concatenation operation is not implemented for NumPy arrays",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            other + td
        msg = r"unsupported operand type\(s\) for -: 'Timedelta' and 'int'"
        with pytest.raises(TypeError, match=msg):
            td - other
        msg = r"unsupported operand type\(s\) for -: 'numpy.ndarray' and 'Timedelta'"
        with pytest.raises(TypeError, match=msg):
            other - td

    def test_td_rsub_nat(self):
        td = Timedelta(10, unit="d")
        result = NaT - td
        assert result is NaT

        result = np.datetime64("NaT") - td
        assert result is NaT

    def test_td_rsub_offset(self):
        result = offsets.Hour(1) - Timedelta(10, unit="d")
        assert isinstance(result, Timedelta)
        assert result == Timedelta(-239, unit="h")

    def test_td_sub_timedeltalike_object_dtype_array(self):
        # GH#21980
        arr = np.array([Timestamp("20130101 9:01"), Timestamp("20121230 9:02")])
        exp = np.array([Timestamp("20121231 9:01"), Timestamp("20121229 9:02")])
        res = arr - Timedelta("1D")
        tm.assert_numpy_array_equal(res, exp)

    def test_td_sub_mixed_most_timedeltalike_object_dtype_array(self):
        # GH#21980
        now = Timestamp("2021-11-09 09:54:00")
        arr = np.array([now, Timedelta("1D"), np.timedelta64(2, "h")])
        exp = np.array(
            [
                now - Timedelta("1D"),
                Timedelta("0D"),
                np.timedelta64(2, "h") - Timedelta("1D"),
            ]
        )
        res = arr - Timedelta("1D")
        tm.assert_numpy_array_equal(res, exp)

    def test_td_rsub_mixed_most_timedeltalike_object_dtype_array(self):
        # GH#21980
        now = Timestamp("2021-11-09 09:54:00")
        arr = np.array([now, Timedelta("1D"), np.timedelta64(2, "h")])
        msg = r"unsupported operand type\(s\) for \-: 'Timedelta' and 'Timestamp'"
        with pytest.raises(TypeError, match=msg):
            Timedelta("1D") - arr

    @pytest.mark.parametrize("op", [operator.add, ops.radd])
    def test_td_add_timedeltalike_object_dtype_array(self, op):
        # GH#21980
        arr = np.array([Timestamp("20130101 9:01"), Timestamp("20121230 9:02")])
        exp = np.array([Timestamp("20130102 9:01"), Timestamp("20121231 9:02")])
        res = op(arr, Timedelta("1D"))
        tm.assert_numpy_array_equal(res, exp)

    @pytest.mark.parametrize("op", [operator.add, ops.radd])
    def test_td_add_mixed_timedeltalike_object_dtype_array(self, op):
        # GH#21980
        now = Timestamp("2021-11-09 09:54:00")
        arr = np.array([now, Timedelta("1D")])
        exp = np.array([now + Timedelta("1D"), Timedelta("2D")])
        res = op(arr, Timedelta("1D"))
        tm.assert_numpy_array_equal(res, exp)

    def test_td_add_sub_td64_ndarray(self):
        td = Timedelta("1 day")

        other = np.array([td.to_timedelta64()])
        expected = np.array([Timedelta("2 Days").to_timedelta64()])

        result = td + other
        tm.assert_numpy_array_equal(result, expected)
        result = other + td
        tm.assert_numpy_array_equal(result, expected)

        result = td - other
        tm.assert_numpy_array_equal(result, expected * 0)
        result = other - td
        tm.assert_numpy_array_equal(result, expected * 0)

    def test_td_add_sub_dt64_ndarray(self):
        td = Timedelta("1 day")
        other = np.array(["2000-01-01"], dtype="M8[ns]")

        expected = np.array(["2000-01-02"], dtype="M8[ns]")
        tm.assert_numpy_array_equal(td + other, expected)
        tm.assert_numpy_array_equal(other + td, expected)

        expected = np.array(["1999-12-31"], dtype="M8[ns]")
        tm.assert_numpy_array_equal(-td + other, expected)
        tm.assert_numpy_array_equal(other - td, expected)

    def test_td_add_sub_ndarray_0d(self):
        td = Timedelta("1 day")
        other = np.array(td.asm8)

        result = td + other
        assert isinstance(result, Timedelta)
        assert result == 2 * td

        result = other + td
        assert isinstance(result, Timedelta)
        assert result == 2 * td

        result = other - td
        assert isinstance(result, Timedelta)
        assert result == 0 * td

        result = td - other
        assert isinstance(result, Timedelta)
        assert result == 0 * td


class TestTimedeltaMultiplicationDivision:
    """
    Tests for Timedelta methods:

        __mul__, __rmul__,
        __div__, __rdiv__,
        __truediv__, __rtruediv__,
        __floordiv__, __rfloordiv__,
        __mod__, __rmod__,
        __divmod__, __rdivmod__
    """

    # ---------------------------------------------------------------
    # Timedelta.__mul__, __rmul__

    @pytest.mark.parametrize(
        "td_nat", [NaT, np.timedelta64("NaT", "ns"), np.timedelta64("NaT")]
    )
    @pytest.mark.parametrize("op", [operator.mul, ops.rmul])
    def test_td_mul_nat(self, op, td_nat):
        # GH#19819
        td = Timedelta(10, unit="d")
        typs = "|".join(["numpy.timedelta64", "NaTType", "Timedelta"])
        msg = "|".join(
            [
                rf"unsupported operand type\(s\) for \*: '{typs}' and '{typs}'",
                r"ufunc '?multiply'? cannot use operands with types",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            op(td, td_nat)

    @pytest.mark.parametrize("nan", [np.nan, np.float64("NaN"), float("nan")])
    @pytest.mark.parametrize("op", [operator.mul, ops.rmul])
    def test_td_mul_nan(self, op, nan):
        # np.float64('NaN') has a 'dtype' attr, avoid treating as array
        td = Timedelta(10, unit="d")
        result = op(td, nan)
        assert result is NaT

    @pytest.mark.parametrize("op", [operator.mul, ops.rmul])
    def test_td_mul_scalar(self, op):
        # GH#19738
        td = Timedelta(minutes=3)

        result = op(td, 2)
        assert result == Timedelta(minutes=6)

        result = op(td, 1.5)
        assert result == Timedelta(minutes=4, seconds=30)

        assert op(td, np.nan) is NaT

        assert op(-1, td)._value == -1 * td._value
        assert op(-1.0, td)._value == -1.0 * td._value

        msg = "unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            # timedelta * datetime is gibberish
            op(td, Timestamp(2016, 1, 2))

        with pytest.raises(TypeError, match=msg):
            # invalid multiply with another timedelta
            op(td, td)

    def test_td_mul_numeric_ndarray(self):
        td = Timedelta("1 day")
        other = np.array([2])
        expected = np.array([Timedelta("2 Days").to_timedelta64()])

        result = td * other
        tm.assert_numpy_array_equal(result, expected)

        result = other * td
        tm.assert_numpy_array_equal(result, expected)

    def test_td_mul_numeric_ndarray_0d(self):
        td = Timedelta("1 day")
        other = np.array(2, dtype=np.int64)
        assert other.ndim == 0
        expected = Timedelta("2 days")

        res = td * other
        assert type(res) is Timedelta
        assert res == expected

        res = other * td
        assert type(res) is Timedelta
        assert res == expected

    def test_td_mul_td64_ndarray_invalid(self):
        td = Timedelta("1 day")
        other = np.array([Timedelta("2 Days").to_timedelta64()])

        msg = (
            "ufunc '?multiply'? cannot use operands with types "
            rf"dtype\('{tm.ENDIAN}m8\[ns\]'\) and dtype\('{tm.ENDIAN}m8\[ns\]'\)"
        )
        with pytest.raises(TypeError, match=msg):
            td * other
        with pytest.raises(TypeError, match=msg):
            other * td

    # ---------------------------------------------------------------
    # Timedelta.__div__, __truediv__

    def test_td_div_timedeltalike_scalar(self):
        # GH#19738
        td = Timedelta(10, unit="d")

        result = td / offsets.Hour(1)
        assert result == 240

        assert td / td == 1
        assert td / np.timedelta64(60, "h") == 4

        assert np.isnan(td / NaT)

    def test_td_div_td64_non_nano(self):
        # truediv
        td = Timedelta("1 days 2 hours 3 ns")
        result = td / np.timedelta64(1, "D")
        assert result == td._value / (86400 * 10**9)
        result = td / np.timedelta64(1, "s")
        assert result == td._value / 10**9
        result = td / np.timedelta64(1, "ns")
        assert result == td._value

        # floordiv
        td = Timedelta("1 days 2 hours 3 ns")
        result = td // np.timedelta64(1, "D")
        assert result == 1
        result = td // np.timedelta64(1, "s")
        assert result == 93600
        result = td // np.timedelta64(1, "ns")
        assert result == td._value

    def test_td_div_numeric_scalar(self):
        # GH#19738
        td = Timedelta(10, unit="d")

        result = td / 2
        assert isinstance(result, Timedelta)
        assert result == Timedelta(days=5)

        result = td / 5
        assert isinstance(result, Timedelta)
        assert result == Timedelta(days=2)

    @pytest.mark.parametrize(
        "nan",
        [
            np.nan,
            np.float64("NaN"),
            float("nan"),
        ],
    )
    def test_td_div_nan(self, nan):
        # np.float64('NaN') has a 'dtype' attr, avoid treating as array
        td = Timedelta(10, unit="d")
        result = td / nan
        assert result is NaT

        result = td // nan
        assert result is NaT

    def test_td_div_td64_ndarray(self):
        td = Timedelta("1 day")

        other = np.array([Timedelta("2 Days").to_timedelta64()])
        expected = np.array([0.5])

        result = td / other
        tm.assert_numpy_array_equal(result, expected)

        result = other / td
        tm.assert_numpy_array_equal(result, expected * 4)

    def test_td_div_ndarray_0d(self):
        td = Timedelta("1 day")

        other = np.array(1)
        res = td / other
        assert isinstance(res, Timedelta)
        assert res == td

    # ---------------------------------------------------------------
    # Timedelta.__rdiv__

    def test_td_rdiv_timedeltalike_scalar(self):
        # GH#19738
        td = Timedelta(10, unit="d")
        result = offsets.Hour(1) / td
        assert result == 1 / 240.0

        assert np.timedelta64(60, "h") / td == 0.25

    def test_td_rdiv_na_scalar(self):
        # GH#31869 None gets cast to NaT
        td = Timedelta(10, unit="d")

        result = NaT / td
        assert np.isnan(result)

        result = None / td
        assert np.isnan(result)

        result = np.timedelta64("NaT") / td
        assert np.isnan(result)

        msg = r"unsupported operand type\(s\) for /: 'numpy.datetime64' and 'Timedelta'"
        with pytest.raises(TypeError, match=msg):
            np.datetime64("NaT") / td

        msg = r"unsupported operand type\(s\) for /: 'float' and 'Timedelta'"
        with pytest.raises(TypeError, match=msg):
            np.nan / td

    def test_td_rdiv_ndarray(self):
        td = Timedelta(10, unit="d")

        arr = np.array([td], dtype=object)
        result = arr / td
        expected = np.array([1], dtype=np.float64)
        tm.assert_numpy_array_equal(result, expected)

        arr = np.array([None])
        result = arr / td
        expected = np.array([np.nan])
        tm.assert_numpy_array_equal(result, expected)

        arr = np.array([np.nan], dtype=object)
        msg = r"unsupported operand type\(s\) for /: 'float' and 'Timedelta'"
        with pytest.raises(TypeError, match=msg):
            arr / td

        arr = np.array([np.nan], dtype=np.float64)
        msg = "cannot use operands with types dtype"
        with pytest.raises(TypeError, match=msg):
            arr / td

    def test_td_rdiv_ndarray_0d(self):
        td = Timedelta(10, unit="d")

        arr = np.array(td.asm8)

        assert arr / td == 1

    # ---------------------------------------------------------------
    # Timedelta.__floordiv__

    def test_td_floordiv_timedeltalike_scalar(self):
        # GH#18846
        td = Timedelta(hours=3, minutes=4)
        scalar = Timedelta(hours=3, minutes=3)

        assert td // scalar == 1
        assert -td // scalar.to_pytimedelta() == -2
        assert (2 * td) // scalar.to_timedelta64() == 2

    def test_td_floordiv_null_scalar(self):
        # GH#18846
        td = Timedelta(hours=3, minutes=4)

        assert td // np.nan is NaT
        assert np.isnan(td // NaT)
        assert np.isnan(td // np.timedelta64("NaT"))

    def test_td_floordiv_offsets(self):
        # GH#19738
        td = Timedelta(hours=3, minutes=4)
        assert td // offsets.Hour(1) == 3
        assert td // offsets.Minute(2) == 92

    def test_td_floordiv_invalid_scalar(self):
        # GH#18846
        td = Timedelta(hours=3, minutes=4)

        msg = "|".join(
            [
                r"Invalid dtype datetime64\[D\] for __floordiv__",
                "'dtype' is an invalid keyword argument for this function",
                "this function got an unexpected keyword argument 'dtype'",
                r"ufunc '?floor_divide'? cannot use operands with types",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            td // np.datetime64("2016-01-01", dtype="datetime64[us]")

    def test_td_floordiv_numeric_scalar(self):
        # GH#18846
        td = Timedelta(hours=3, minutes=4)

        expected = Timedelta(hours=1, minutes=32)
        assert td // 2 == expected
        assert td // 2.0 == expected
        assert td // np.float64(2.0) == expected
        assert td // np.int32(2.0) == expected
        assert td // np.uint8(2.0) == expected

    def test_td_floordiv_timedeltalike_array(self):
        # GH#18846
        td = Timedelta(hours=3, minutes=4)
        scalar = Timedelta(hours=3, minutes=3)

        # Array-like others
        assert td // np.array(scalar.to_timedelta64()) == 1

        res = (3 * td) // np.array([scalar.to_timedelta64()])
        expected = np.array([3], dtype=np.int64)
        tm.assert_numpy_array_equal(res, expected)

        res = (10 * td) // np.array([scalar.to_timedelta64(), np.timedelta64("NaT")])
        expected = np.array([10, np.nan])
        tm.assert_numpy_array_equal(res, expected)

    def test_td_floordiv_numeric_series(self):
        # GH#18846
        td = Timedelta(hours=3, minutes=4)
        ser = pd.Series([1], dtype=np.int64)
        res = td // ser
        assert res.dtype.kind == "m"

    # ---------------------------------------------------------------
    # Timedelta.__rfloordiv__

    def test_td_rfloordiv_timedeltalike_scalar(self):
        # GH#18846
        td = Timedelta(hours=3, minutes=3)
        scalar = Timedelta(hours=3, minutes=4)

        # scalar others
        # x // Timedelta is defined only for timedelta-like x. int-like,
        # float-like, and date-like, in particular, should all either
        # a) raise TypeError directly or
        # b) return NotImplemented, following which the reversed
        #    operation will raise TypeError.
        assert td.__rfloordiv__(scalar) == 1
        assert (-td).__rfloordiv__(scalar.to_pytimedelta()) == -2
        assert (2 * td).__rfloordiv__(scalar.to_timedelta64()) == 0

    def test_td_rfloordiv_null_scalar(self):
        # GH#18846
        td = Timedelta(hours=3, minutes=3)

        assert np.isnan(td.__rfloordiv__(NaT))
        assert np.isnan(td.__rfloordiv__(np.timedelta64("NaT")))

    def test_td_rfloordiv_offsets(self):
        # GH#19738
        assert offsets.Hour(1) // Timedelta(minutes=25) == 2

    def test_td_rfloordiv_invalid_scalar(self):
        # GH#18846
        td = Timedelta(hours=3, minutes=3)

        dt64 = np.datetime64("2016-01-01", "us")

        assert td.__rfloordiv__(dt64) is NotImplemented

        msg = (
            r"unsupported operand type\(s\) for //: 'numpy.datetime64' and 'Timedelta'"
        )
        with pytest.raises(TypeError, match=msg):
            dt64 // td

    def test_td_rfloordiv_numeric_scalar(self):
        # GH#18846
        td = Timedelta(hours=3, minutes=3)

        assert td.__rfloordiv__(np.nan) is NotImplemented
        assert td.__rfloordiv__(3.5) is NotImplemented
        assert td.__rfloordiv__(2) is NotImplemented
        assert td.__rfloordiv__(np.float64(2.0)) is NotImplemented
        assert td.__rfloordiv__(np.uint8(9)) is NotImplemented
        assert td.__rfloordiv__(np.int32(2.0)) is NotImplemented

        msg = r"unsupported operand type\(s\) for //: '.*' and 'Timedelta"
        with pytest.raises(TypeError, match=msg):
            np.float64(2.0) // td
        with pytest.raises(TypeError, match=msg):
            np.uint8(9) // td
        with pytest.raises(TypeError, match=msg):
            # deprecated GH#19761, enforced GH#29797
            np.int32(2.0) // td

    def test_td_rfloordiv_timedeltalike_array(self):
        # GH#18846
        td = Timedelta(hours=3, minutes=3)
        scalar = Timedelta(hours=3, minutes=4)

        # Array-like others
        assert td.__rfloordiv__(np.array(scalar.to_timedelta64())) == 1

        res = td.__rfloordiv__(np.array([(3 * scalar).to_timedelta64()]))
        expected = np.array([3], dtype=np.int64)
        tm.assert_numpy_array_equal(res, expected)

        arr = np.array([(10 * scalar).to_timedelta64(), np.timedelta64("NaT")])
        res = td.__rfloordiv__(arr)
        expected = np.array([10, np.nan])
        tm.assert_numpy_array_equal(res, expected)

    def test_td_rfloordiv_intarray(self):
        # deprecated GH#19761, enforced GH#29797
        ints = np.array([1349654400, 1349740800, 1349827200, 1349913600]) * 10**9

        msg = "Invalid dtype"
        with pytest.raises(TypeError, match=msg):
            ints // Timedelta(1, unit="s")

    def test_td_rfloordiv_numeric_series(self):
        # GH#18846
        td = Timedelta(hours=3, minutes=3)
        ser = pd.Series([1], dtype=np.int64)
        res = td.__rfloordiv__(ser)
        assert res is NotImplemented

        msg = "Invalid dtype"
        with pytest.raises(TypeError, match=msg):
            # Deprecated GH#19761, enforced GH#29797
            ser // td

    # ----------------------------------------------------------------
    # Timedelta.__mod__, __rmod__

    def test_mod_timedeltalike(self):
        # GH#19365
        td = Timedelta(hours=37)

        # Timedelta-like others
        result = td % Timedelta(hours=6)
        assert isinstance(result, Timedelta)
        assert result == Timedelta(hours=1)

        result = td % timedelta(minutes=60)
        assert isinstance(result, Timedelta)
        assert result == Timedelta(0)

        result = td % NaT
        assert result is NaT

    def test_mod_timedelta64_nat(self):
        # GH#19365
        td = Timedelta(hours=37)

        result = td % np.timedelta64("NaT", "ns")
        assert result is NaT

    def test_mod_timedelta64(self):
        # GH#19365
        td = Timedelta(hours=37)

        result = td % np.timedelta64(2, "h")
        assert isinstance(result, Timedelta)
        assert result == Timedelta(hours=1)

    def test_mod_offset(self):
        # GH#19365
        td = Timedelta(hours=37)

        result = td % offsets.Hour(5)
        assert isinstance(result, Timedelta)
        assert result == Timedelta(hours=2)

    def test_mod_numeric(self):
        # GH#19365
        td = Timedelta(hours=37)

        # Numeric Others
        result = td % 2
        assert isinstance(result, Timedelta)
        assert result == Timedelta(0)

        result = td % 1e12
        assert isinstance(result, Timedelta)
        assert result == Timedelta(minutes=3, seconds=20)

        result = td % int(1e12)
        assert isinstance(result, Timedelta)
        assert result == Timedelta(minutes=3, seconds=20)

    def test_mod_invalid(self):
        # GH#19365
        td = Timedelta(hours=37)
        msg = "unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            td % Timestamp("2018-01-22")

        with pytest.raises(TypeError, match=msg):
            td % []

    def test_rmod_pytimedelta(self):
        # GH#19365
        td = Timedelta(minutes=3)

        result = timedelta(minutes=4) % td
        assert isinstance(result, Timedelta)
        assert result == Timedelta(minutes=1)

    def test_rmod_timedelta64(self):
        # GH#19365
        td = Timedelta(minutes=3)
        result = np.timedelta64(5, "m") % td
        assert isinstance(result, Timedelta)
        assert result == Timedelta(minutes=2)

    def test_rmod_invalid(self):
        # GH#19365
        td = Timedelta(minutes=3)

        msg = "unsupported operand"
        with pytest.raises(TypeError, match=msg):
            Timestamp("2018-01-22") % td

        with pytest.raises(TypeError, match=msg):
            15 % td

        with pytest.raises(TypeError, match=msg):
            16.0 % td

        msg = "Invalid dtype int"
        with pytest.raises(TypeError, match=msg):
            np.array([22, 24]) % td

    # ----------------------------------------------------------------
    # Timedelta.__divmod__, __rdivmod__

    def test_divmod_numeric(self):
        # GH#19365
        td = Timedelta(days=2, hours=6)

        result = divmod(td, 53 * 3600 * 1e9)
        assert result[0] == Timedelta(1, unit="ns")
        assert isinstance(result[1], Timedelta)
        assert result[1] == Timedelta(hours=1)

        assert result
        result = divmod(td, np.nan)
        assert result[0] is NaT
        assert result[1] is NaT

    def test_divmod(self):
        # GH#19365
        td = Timedelta(days=2, hours=6)

        result = divmod(td, timedelta(days=1))
        assert result[0] == 2
        assert isinstance(result[1], Timedelta)
        assert result[1] == Timedelta(hours=6)

        result = divmod(td, 54)
        assert result[0] == Timedelta(hours=1)
        assert isinstance(result[1], Timedelta)
        assert result[1] == Timedelta(0)

        result = divmod(td, NaT)
        assert np.isnan(result[0])
        assert result[1] is NaT

    def test_divmod_offset(self):
        # GH#19365
        td = Timedelta(days=2, hours=6)

        result = divmod(td, offsets.Hour(-4))
        assert result[0] == -14
        assert isinstance(result[1], Timedelta)
        assert result[1] == Timedelta(hours=-2)

    def test_divmod_invalid(self):
        # GH#19365
        td = Timedelta(days=2, hours=6)

        msg = r"unsupported operand type\(s\) for //: 'Timedelta' and 'Timestamp'"
        with pytest.raises(TypeError, match=msg):
            divmod(td, Timestamp("2018-01-22"))

    def test_rdivmod_pytimedelta(self):
        # GH#19365
        result = divmod(timedelta(days=2, hours=6), Timedelta(days=1))
        assert result[0] == 2
        assert isinstance(result[1], Timedelta)
        assert result[1] == Timedelta(hours=6)

    def test_rdivmod_offset(self):
        result = divmod(offsets.Hour(54), Timedelta(hours=-4))
        assert result[0] == -14
        assert isinstance(result[1], Timedelta)
        assert result[1] == Timedelta(hours=-2)

    def test_rdivmod_invalid(self):
        # GH#19365
        td = Timedelta(minutes=3)
        msg = "unsupported operand type"

        with pytest.raises(TypeError, match=msg):
            divmod(Timestamp("2018-01-22"), td)

        with pytest.raises(TypeError, match=msg):
            divmod(15, td)

        with pytest.raises(TypeError, match=msg):
            divmod(16.0, td)

        msg = "Invalid dtype int"
        with pytest.raises(TypeError, match=msg):
            divmod(np.array([22, 24]), td)

    # ----------------------------------------------------------------

    @pytest.mark.parametrize(
        "op", [operator.mul, ops.rmul, operator.truediv, ops.rdiv, ops.rsub]
    )
    @pytest.mark.parametrize(
        "arr",
        [
            np.array([Timestamp("20130101 9:01"), Timestamp("20121230 9:02")]),
            np.array([Timestamp("2021-11-09 09:54:00"), Timedelta("1D")]),
        ],
    )
    def test_td_op_timedelta_timedeltalike_array(self, op, arr):
        msg = "unsupported operand type|cannot use operands with types"
        with pytest.raises(TypeError, match=msg):
            op(arr, Timedelta("1D"))


class TestTimedeltaComparison:
    @pytest.mark.skip_ubsan
    def test_compare_pytimedelta_bounds(self):
        # GH#49021 don't overflow on comparison with very large pytimedeltas

        for unit in ["ns", "us"]:
            tdmax = Timedelta.max.as_unit(unit).max
            tdmin = Timedelta.min.as_unit(unit).min

            assert tdmax < timedelta.max
            assert tdmax <= timedelta.max
            assert not tdmax > timedelta.max
            assert not tdmax >= timedelta.max
            assert tdmax != timedelta.max
            assert not tdmax == timedelta.max

            assert tdmin > timedelta.min
            assert tdmin >= timedelta.min
            assert not tdmin < timedelta.min
            assert not tdmin <= timedelta.min
            assert tdmin != timedelta.min
            assert not tdmin == timedelta.min

        # But the "ms" and "s"-reso bounds extend pass pytimedelta
        for unit in ["ms", "s"]:
            tdmax = Timedelta.max.as_unit(unit).max
            tdmin = Timedelta.min.as_unit(unit).min

            assert tdmax > timedelta.max
            assert tdmax >= timedelta.max
            assert not tdmax < timedelta.max
            assert not tdmax <= timedelta.max
            assert tdmax != timedelta.max
            assert not tdmax == timedelta.max

            assert tdmin < timedelta.min
            assert tdmin <= timedelta.min
            assert not tdmin > timedelta.min
            assert not tdmin >= timedelta.min
            assert tdmin != timedelta.min
            assert not tdmin == timedelta.min

    def test_compare_pytimedelta_bounds2(self):
        # a pytimedelta outside the microsecond bounds
        pytd = timedelta(days=999999999, seconds=86399)
        # NB: np.timedelta64(td, "s"") incorrectly overflows
        td64 = np.timedelta64(pytd.days, "D") + np.timedelta64(pytd.seconds, "s")
        td = Timedelta(td64)
        assert td.days == pytd.days
        assert td.seconds == pytd.seconds

        assert td == pytd
        assert not td != pytd
        assert not td < pytd
        assert not td > pytd
        assert td <= pytd
        assert td >= pytd

        td2 = td - Timedelta(seconds=1).as_unit("s")
        assert td2 != pytd
        assert not td2 == pytd
        assert td2 < pytd
        assert td2 <= pytd
        assert not td2 > pytd
        assert not td2 >= pytd

    def test_compare_tick(self, tick_classes):
        cls = tick_classes

        off = cls(4)
        td = off._as_pd_timedelta
        assert isinstance(td, Timedelta)

        assert td == off
        assert not td != off
        assert td <= off
        assert td >= off
        assert not td < off
        assert not td > off

        assert not td == 2 * off
        assert td != 2 * off
        assert td <= 2 * off
        assert td < 2 * off
        assert not td >= 2 * off
        assert not td > 2 * off

    def test_comparison_object_array(self):
        # analogous to GH#15183
        td = Timedelta("2 days")
        other = Timedelta("3 hours")

        arr = np.array([other, td], dtype=object)
        res = arr == td
        expected = np.array([False, True], dtype=bool)
        assert (res == expected).all()

        # 2D case
        arr = np.array([[other, td], [td, other]], dtype=object)
        res = arr != td
        expected = np.array([[True, False], [False, True]], dtype=bool)
        assert res.shape == expected.shape
        assert (res == expected).all()

    def test_compare_timedelta_ndarray(self):
        # GH#11835
        periods = [Timedelta("0 days 01:00:00"), Timedelta("0 days 01:00:00")]
        arr = np.array(periods)
        result = arr[0] > arr
        expected = np.array([False, False])
        tm.assert_numpy_array_equal(result, expected)

    def test_compare_td64_ndarray(self):
        # GG#33441
        arr = np.arange(5).astype("timedelta64[ns]")
        td = Timedelta(arr[1])

        expected = np.array([False, True, False, False, False], dtype=bool)

        result = td == arr
        tm.assert_numpy_array_equal(result, expected)

        result = arr == td
        tm.assert_numpy_array_equal(result, expected)

        result = td != arr
        tm.assert_numpy_array_equal(result, ~expected)

        result = arr != td
        tm.assert_numpy_array_equal(result, ~expected)

    def test_compare_custom_object(self):
        """
        Make sure non supported operations on Timedelta returns NonImplemented
        and yields to other operand (GH#20829).
        """

        class CustomClass:
            def __init__(self, cmp_result=None) -> None:
                self.cmp_result = cmp_result

            def generic_result(self):
                if self.cmp_result is None:
                    return NotImplemented
                else:
                    return self.cmp_result

            def __eq__(self, other):
                return self.generic_result()

            def __gt__(self, other):
                return self.generic_result()

        t = Timedelta("1s")

        assert t != "string"
        assert t != 1
        assert t != CustomClass()
        assert t != CustomClass(cmp_result=False)

        assert t < CustomClass(cmp_result=True)
        assert not t < CustomClass(cmp_result=False)

        assert t == CustomClass(cmp_result=True)

    @pytest.mark.parametrize("val", ["string", 1])
    def test_compare_unknown_type(self, val):
        # GH#20829
        t = Timedelta("1s")
        msg = "not supported between instances of 'Timedelta' and '(int|str)'"
        with pytest.raises(TypeError, match=msg):
            t >= val
        with pytest.raises(TypeError, match=msg):
            t > val
        with pytest.raises(TypeError, match=msg):
            t <= val
        with pytest.raises(TypeError, match=msg):
            t < val


def test_ops_notimplemented():
    class Other:
        pass

    other = Other()

    td = Timedelta("1 day")
    assert td.__add__(other) is NotImplemented
    assert td.__sub__(other) is NotImplemented
    assert td.__truediv__(other) is NotImplemented
    assert td.__mul__(other) is NotImplemented
    assert td.__floordiv__(other) is NotImplemented


def test_ops_error_str():
    # GH#13624
    td = Timedelta("1 day")

    for left, right in [(td, "a"), ("a", td)]:
        msg = "|".join(
            [
                "unsupported operand type",
                r'can only concatenate str \(not "Timedelta"\) to str',
                "must be str, not Timedelta",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            left + right

        msg = "not supported between instances of"
        with pytest.raises(TypeError, match=msg):
            left > right

        assert not left == right  # pylint: disable=unneeded-not
        assert left != right

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\truststore\__init__.py ===
"""Verify certificates using native system trust stores"""

import sys as _sys

if _sys.version_info < (3, 10):
    raise ImportError("truststore requires Python 3.10 or later")

# Detect Python runtimes which don't implement SSLObject.get_unverified_chain() API
# This API only became public in Python 3.13 but was available in CPython and PyPy since 3.10.
if _sys.version_info < (3, 13) and _sys.implementation.name not in ("cpython", "pypy"):
    try:
        import ssl as _ssl
    except ImportError:
        raise ImportError("truststore requires the 'ssl' module")
    else:
        _sslmem = _ssl.MemoryBIO()
        _sslobj = _ssl.create_default_context().wrap_bio(
            _sslmem,
            _sslmem,
        )
        try:
            while not hasattr(_sslobj, "get_unverified_chain"):
                _sslobj = _sslobj._sslobj  # type: ignore[attr-defined]
        except AttributeError:
            raise ImportError(
                "truststore requires peer certificate chain APIs to be available"
            ) from None

        del _ssl, _sslobj, _sslmem  # noqa: F821

from ._api import SSLContext, extract_from_ssl, inject_into_ssl  # noqa: E402

del _api, _sys  # type: ignore[name-defined] # noqa: F821

__all__ = ["SSLContext", "inject_into_ssl", "extract_from_ssl"]
__version__ = "0.10.1"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\contrib\_appengine_environ.py ===
"""
This module provides means to detect the App Engine environment.
"""

import os


def is_appengine():
    return is_local_appengine() or is_prod_appengine()


def is_appengine_sandbox():
    """Reports if the app is running in the first generation sandbox.

    The second generation runtimes are technically still in a sandbox, but it
    is much less restrictive, so generally you shouldn't need to check for it.
    see https://cloud.google.com/appengine/docs/standard/runtimes
    """
    return is_appengine() and os.environ["APPENGINE_RUNTIME"] == "python27"


def is_local_appengine():
    return "APPENGINE_RUNTIME" in os.environ and os.environ.get(
        "SERVER_SOFTWARE", ""
    ).startswith("Development/")


def is_prod_appengine():
    return "APPENGINE_RUNTIME" in os.environ and os.environ.get(
        "SERVER_SOFTWARE", ""
    ).startswith("Google App Engine/")


def is_prod_appengine_mvms():
    """Deprecated."""
    return False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\actions\wheel_actions.py ===
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

from typing import Optional

from .interaction import Interaction
from .wheel_input import WheelInput


class WheelActions(Interaction):
    def __init__(self, source: Optional[WheelInput] = None):
        if source is None:
            source = WheelInput("wheel")
        super().__init__(source)

    def pause(self, duration: float = 0):
        self.source.create_pause(duration)
        return self

    def scroll(self, x=0, y=0, delta_x=0, delta_y=0, duration=0, origin="viewport"):
        self.source.create_scroll(x, y, delta_x, delta_y, duration, origin)
        return self

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\autolev\test-examples\ruletest6.py ===
import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

q1, q2 = _me.dynamicsymbols('q1 q2')
x, y, z = _me.dynamicsymbols('x y z')
e = q1+q2
a = (e).subs({q1:x**2+y**2, q2:x-y})
e2 = _sm.cos(x)
e3 = _sm.cos(x*y)
a = (e2).series(x, 0, 2).removeO()
b = (e3).series(x, 0, 2).removeO().series(y, 0, 2).removeO()
e = ((x+y)**2).expand()
a = (e).subs({q1:x**2+y**2,q2:x-y}).subs({x:1,y:z})
bm = _sm.Matrix([i.subs({x:1,y:z}) for i in _sm.Matrix([e,2*e]).reshape(2, 1)]).reshape((_sm.Matrix([e,2*e]).reshape(2, 1)).shape[0], (_sm.Matrix([e,2*e]).reshape(2, 1)).shape[1])
e = q1+q2
a = (e).subs({q1:x**2+y**2,q2:x-y}).subs({x:2,y:z**2})
j, k, l = _sm.symbols('j k l', real=True)
p1 = _sm.Poly(_sm.Matrix([j,k,l]).reshape(1, 3), x)
p2 = _sm.Poly(j*x+k, x)
root1 = [i.evalf() for i in _sm.solve(p1, x)]
root2 = [i.evalf() for i in _sm.solve(_sm.Poly(_sm.Matrix([1,2,3]).reshape(3, 1), x),x)]
m = _sm.Matrix([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape(4, 4)
am = (m).T+m
bm = _sm.Matrix([i.evalf() for i in (m).eigenvals().keys()])
c1 = _sm.diag(1,1,1,1)
c2 = _sm.Matrix([2 if i==j else 0 for i in range(3) for j in range(4)]).reshape(3, 4)
dm = (m+c1)**(-1)
e = (m+c1).det()+(_sm.Matrix([1,0,0,1]).reshape(2, 2)).trace()
f = (m)[1,2]
a = (m).cols
bm = (m).col(0)
cm = _sm.Matrix([(m).T.row(0),(m).T.row(1),(m).T.row(2),(m).T.row(3),(m).T.row(2)])
dm = (m).row(0)
em = _sm.Matrix([(m).row(0),(m).row(1),(m).row(2),(m).row(3),(m).row(2)])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\autolev\test-examples\pydy-example-repo\non_min_pendulum.py ===
import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

q1, q2 = _me.dynamicsymbols('q1 q2')
q1_d, q2_d = _me.dynamicsymbols('q1_ q2_', 1)
q1_dd, q2_dd = _me.dynamicsymbols('q1_ q2_', 2)
l, m, g = _sm.symbols('l m g', real=True)
frame_n = _me.ReferenceFrame('n')
point_pn = _me.Point('pn')
point_pn.set_vel(frame_n, 0)
theta1 = _sm.atan(q2/q1)
frame_a = _me.ReferenceFrame('a')
frame_a.orient(frame_n, 'Axis', [theta1, frame_n.z])
particle_p = _me.Particle('p', _me.Point('p_pt'), _sm.Symbol('m'))
particle_p.point.set_pos(point_pn, q1*frame_n.x+q2*frame_n.y)
particle_p.mass = m
particle_p.point.set_vel(frame_n, (point_pn.pos_from(particle_p.point)).dt(frame_n))
f_v = _me.dot((particle_p.point.vel(frame_n)).express(frame_a), frame_a.x)
force_p = particle_p.mass*(g*frame_n.x)
dependent = _sm.Matrix([[0]])
dependent[0] = f_v
velocity_constraints = [i for i in dependent]
u_q1_d = _me.dynamicsymbols('u_q1_d')
u_q2_d = _me.dynamicsymbols('u_q2_d')
kd_eqs = [q1_d-u_q1_d, q2_d-u_q2_d]
forceList = [(particle_p.point,particle_p.mass*(g*frame_n.x))]
kane = _me.KanesMethod(frame_n, q_ind=[q1,q2], u_ind=[u_q2_d], u_dependent=[u_q1_d], kd_eqs = kd_eqs, velocity_constraints = velocity_constraints)
fr, frstar = kane.kanes_equations([particle_p], forceList)
zero = fr+frstar
f_c = point_pn.pos_from(particle_p.point).magnitude()-l
config = _sm.Matrix([[0]])
config[0] = f_c
zero = zero.row_insert(zero.shape[0], _sm.Matrix([[0]]))
zero[zero.shape[0]-1] = config[0]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\vector\__init__.py ===
__all__ = [
    'CoordinateSym', 'ReferenceFrame',

    'Dyadic',

    'Vector',

    'Point',

    'cross', 'dot', 'express', 'time_derivative', 'outer',
    'kinematic_equations', 'get_motion_params', 'partial_velocity',
    'dynamicsymbols',

    'vprint', 'vsstrrepr', 'vsprint', 'vpprint', 'vlatex', 'init_vprinting',

    'curl', 'divergence', 'gradient', 'is_conservative', 'is_solenoidal',
    'scalar_potential', 'scalar_potential_difference',

]
from .frame import CoordinateSym, ReferenceFrame

from .dyadic import Dyadic

from .vector import Vector

from .point import Point

from .functions import (cross, dot, express, time_derivative, outer,
        kinematic_equations, get_motion_params, partial_velocity,
        dynamicsymbols)

from .printing import (vprint, vsstrrepr, vsprint, vpprint, vlatex,
        init_vprinting)

from .fieldfunctions import (curl, divergence, gradient, is_conservative,
        is_solenoidal, scalar_potential, scalar_potential_difference)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\sets\__init__.py ===
from .sets import (Set, Interval, Union, FiniteSet, ProductSet,
        Intersection, imageset, Complement, SymmetricDifference,
        DisjointUnion)

from .fancysets import ImageSet, Range, ComplexRegion
from .contains import Contains
from .conditionset import ConditionSet
from .ordinals import Ordinal, OmegaPower, ord0
from .powerset import PowerSet
from ..core.singleton import S
from .handlers.comparison import _eval_is_eq  # noqa:F401
Complexes = S.Complexes
EmptySet = S.EmptySet
Integers = S.Integers
Naturals = S.Naturals
Naturals0 = S.Naturals0
Rationals = S.Rationals
Reals = S.Reals
UniversalSet = S.UniversalSet

__all__ = [
    'Set', 'Interval', 'Union', 'EmptySet', 'FiniteSet', 'ProductSet',
    'Intersection', 'imageset', 'Complement', 'SymmetricDifference', 'DisjointUnion',

    'ImageSet', 'Range', 'ComplexRegion', 'Reals',

    'Contains',

    'ConditionSet',

    'Ordinal', 'OmegaPower', 'ord0',

    'PowerSet',

    'Reals', 'Naturals', 'Naturals0', 'UniversalSet', 'Integers', 'Rationals',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\testing\_network.py ===
from .. import socket as tsocket
from .._highlevel_socket import SocketListener, SocketStream


async def open_stream_to_socket_listener(
    socket_listener: SocketListener,
) -> SocketStream:
    """Connect to the given :class:`~trio.SocketListener`.

    This is particularly useful in tests when you want to let a server pick
    its own port, and then connect to it::

        listeners = await trio.open_tcp_listeners(0)
        client = await trio.testing.open_stream_to_socket_listener(listeners[0])

    Args:
      socket_listener (~trio.SocketListener): The
          :class:`~trio.SocketListener` to connect to.

    Returns:
      SocketStream: a stream connected to the given listener.

    """
    family = socket_listener.socket.family
    sockaddr = socket_listener.socket.getsockname()
    if family in (tsocket.AF_INET, tsocket.AF_INET6):
        sockaddr = list(sockaddr)
        if sockaddr[0] == "0.0.0.0":
            sockaddr[0] = "127.0.0.1"
        if sockaddr[0] == "::":
            sockaddr[0] = "::1"
        sockaddr = tuple(sockaddr)

    sock = tsocket.socket(family=family)
    await sock.connect(sockaddr)
    return SocketStream(sock)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\formats\test_to_html.py ===
from datetime import datetime
from io import StringIO
import itertools
import re
import textwrap

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    get_option,
    option_context,
)
import pandas._testing as tm

import pandas.io.formats.format as fmt

lorem_ipsum = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim "
    "veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex "
    "ea commodo consequat. Duis aute irure dolor in reprehenderit in "
    "voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur "
    "sint occaecat cupidatat non proident, sunt in culpa qui officia "
    "deserunt mollit anim id est laborum."
)


def expected_html(datapath, name):
    """
    Read HTML file from formats data directory.

    Parameters
    ----------
    datapath : pytest fixture
        The datapath fixture injected into a test by pytest.
    name : str
        The name of the HTML file without the suffix.

    Returns
    -------
    str : contents of HTML file.
    """
    filename = ".".join([name, "html"])
    filepath = datapath("io", "formats", "data", "html", filename)
    with open(filepath, encoding="utf-8") as f:
        html = f.read()
    return html.rstrip()


@pytest.fixture(params=["mixed", "empty"])
def biggie_df_fixture(request):
    """Fixture for a big mixed Dataframe and an empty Dataframe"""
    if request.param == "mixed":
        df = DataFrame(
            {
                "A": np.random.default_rng(2).standard_normal(200),
                "B": Index([f"{i}?!" for i in range(200)]),
            },
            index=np.arange(200),
        )
        df.loc[:20, "A"] = np.nan
        df.loc[:20, "B"] = np.nan
        return df
    elif request.param == "empty":
        df = DataFrame(index=np.arange(200))
        return df


@pytest.fixture(params=fmt.VALID_JUSTIFY_PARAMETERS)
def justify(request):
    return request.param


@pytest.mark.parametrize("col_space", [30, 50])
def test_to_html_with_col_space(col_space):
    df = DataFrame(np.random.default_rng(2).random(size=(1, 3)))
    # check that col_space affects HTML generation
    # and be very brittle about it.
    result = df.to_html(col_space=col_space)
    hdrs = [x for x in result.split(r"\n") if re.search(r"<th[>\s]", x)]
    assert len(hdrs) > 0
    for h in hdrs:
        assert "min-width" in h
        assert str(col_space) in h


def test_to_html_with_column_specific_col_space_raises():
    df = DataFrame(
        np.random.default_rng(2).random(size=(3, 3)), columns=["a", "b", "c"]
    )

    msg = (
        "Col_space length\\(\\d+\\) should match "
        "DataFrame number of columns\\(\\d+\\)"
    )
    with pytest.raises(ValueError, match=msg):
        df.to_html(col_space=[30, 40])

    with pytest.raises(ValueError, match=msg):
        df.to_html(col_space=[30, 40, 50, 60])

    msg = "unknown column"
    with pytest.raises(ValueError, match=msg):
        df.to_html(col_space={"a": "foo", "b": 23, "d": 34})


def test_to_html_with_column_specific_col_space():
    df = DataFrame(
        np.random.default_rng(2).random(size=(3, 3)), columns=["a", "b", "c"]
    )

    result = df.to_html(col_space={"a": "2em", "b": 23})
    hdrs = [x for x in result.split("\n") if re.search(r"<th[>\s]", x)]
    assert 'min-width: 2em;">a</th>' in hdrs[1]
    assert 'min-width: 23px;">b</th>' in hdrs[2]
    assert "<th>c</th>" in hdrs[3]

    result = df.to_html(col_space=["1em", 2, 3])
    hdrs = [x for x in result.split("\n") if re.search(r"<th[>\s]", x)]
    assert 'min-width: 1em;">a</th>' in hdrs[1]
    assert 'min-width: 2px;">b</th>' in hdrs[2]
    assert 'min-width: 3px;">c</th>' in hdrs[3]


def test_to_html_with_empty_string_label():
    # GH 3547, to_html regards empty string labels as repeated labels
    data = {"c1": ["a", "b"], "c2": ["a", ""], "data": [1, 2]}
    df = DataFrame(data).set_index(["c1", "c2"])
    result = df.to_html()
    assert "rowspan" not in result


@pytest.mark.parametrize(
    "df,expected",
    [
        (DataFrame({"\u03c3": np.arange(10.0)}), "unicode_1"),
        (DataFrame({"A": ["\u03c3"]}), "unicode_2"),
    ],
)
def test_to_html_unicode(df, expected, datapath):
    expected = expected_html(datapath, expected)
    result = df.to_html()
    assert result == expected


def test_to_html_encoding(float_frame, tmp_path):
    # GH 28663
    path = tmp_path / "test.html"
    float_frame.to_html(path, encoding="gbk")
    with open(str(path), encoding="gbk") as f:
        assert float_frame.to_html() == f.read()


def test_to_html_decimal(datapath):
    # GH 12031
    df = DataFrame({"A": [6.0, 3.1, 2.2]})
    result = df.to_html(decimal=",")
    expected = expected_html(datapath, "gh12031_expected_output")
    assert result == expected


@pytest.mark.parametrize(
    "kwargs,string,expected",
    [
        ({}, "<type 'str'>", "escaped"),
        ({"escape": False}, "<b>bold</b>", "escape_disabled"),
    ],
)
def test_to_html_escaped(kwargs, string, expected, datapath):
    a = "str<ing1 &amp;"
    b = "stri>ng2 &amp;"

    test_dict = {"co<l1": {a: string, b: string}, "co>l2": {a: string, b: string}}
    result = DataFrame(test_dict).to_html(**kwargs)
    expected = expected_html(datapath, expected)
    assert result == expected


@pytest.mark.parametrize("index_is_named", [True, False])
def test_to_html_multiindex_index_false(index_is_named, datapath):
    # GH 8452
    df = DataFrame(
        {"a": range(2), "b": range(3, 5), "c": range(5, 7), "d": range(3, 5)}
    )
    df.columns = MultiIndex.from_product([["a", "b"], ["c", "d"]])
    if index_is_named:
        df.index = Index(df.index.values, name="idx")
    result = df.to_html(index=False)
    expected = expected_html(datapath, "gh8452_expected_output")
    assert result == expected


@pytest.mark.parametrize(
    "multi_sparse,expected",
    [
        (False, "multiindex_sparsify_false_multi_sparse_1"),
        (False, "multiindex_sparsify_false_multi_sparse_2"),
        (True, "multiindex_sparsify_1"),
        (True, "multiindex_sparsify_2"),
    ],
)
def test_to_html_multiindex_sparsify(multi_sparse, expected, datapath):
    index = MultiIndex.from_arrays([[0, 0, 1, 1], [0, 1, 0, 1]], names=["foo", None])
    df = DataFrame([[0, 1], [2, 3], [4, 5], [6, 7]], index=index)
    if expected.endswith("2"):
        df.columns = index[::2]
    with option_context("display.multi_sparse", multi_sparse):
        result = df.to_html()
    expected = expected_html(datapath, expected)
    assert result == expected


@pytest.mark.parametrize(
    "max_rows,expected",
    [
        (60, "gh14882_expected_output_1"),
        # Test that ... appears in a middle level
        (56, "gh14882_expected_output_2"),
    ],
)
def test_to_html_multiindex_odd_even_truncate(max_rows, expected, datapath):
    # GH 14882 - Issue on truncation with odd length DataFrame
    index = MultiIndex.from_product(
        [[100, 200, 300], [10, 20, 30], [1, 2, 3, 4, 5, 6, 7]], names=["a", "b", "c"]
    )
    df = DataFrame({"n": range(len(index))}, index=index)
    result = df.to_html(max_rows=max_rows)
    expected = expected_html(datapath, expected)
    assert result == expected


@pytest.mark.parametrize(
    "df,formatters,expected",
    [
        (
            DataFrame(
                [[0, 1], [2, 3], [4, 5], [6, 7]],
                columns=Index(["foo", None], dtype=object),
                index=np.arange(4),
            ),
            {"__index__": lambda x: "abcd"[x]},
            "index_formatter",
        ),
        (
            DataFrame({"months": [datetime(2016, 1, 1), datetime(2016, 2, 2)]}),
            {"months": lambda x: x.strftime("%Y-%m")},
            "datetime64_monthformatter",
        ),
        (
            DataFrame(
                {
                    "hod": pd.to_datetime(
                        ["10:10:10.100", "12:12:12.120"], format="%H:%M:%S.%f"
                    )
                }
            ),
            {"hod": lambda x: x.strftime("%H:%M")},
            "datetime64_hourformatter",
        ),
        (
            DataFrame(
                {
                    "i": pd.Series([1, 2], dtype="int64"),
                    "f": pd.Series([1, 2], dtype="float64"),
                    "I": pd.Series([1, 2], dtype="Int64"),
                    "s": pd.Series([1, 2], dtype="string"),
                    "b": pd.Series([True, False], dtype="boolean"),
                    "c": pd.Series(["a", "b"], dtype=pd.CategoricalDtype(["a", "b"])),
                    "o": pd.Series([1, "2"], dtype=object),
                }
            ),
            [lambda x: "formatted"] * 7,
            "various_dtypes_formatted",
        ),
    ],
)
def test_to_html_formatters(df, formatters, expected, datapath):
    expected = expected_html(datapath, expected)
    result = df.to_html(formatters=formatters)
    assert result == expected


def test_to_html_regression_GH6098():
    df = DataFrame(
        {
            "clé1": ["a", "a", "b", "b", "a"],
            "clé2": ["1er", "2ème", "1er", "2ème", "1er"],
            "données1": np.random.default_rng(2).standard_normal(5),
            "données2": np.random.default_rng(2).standard_normal(5),
        }
    )

    # it works
    df.pivot_table(index=["clé1"], columns=["clé2"])._repr_html_()


def test_to_html_truncate(datapath):
    index = pd.date_range(start="20010101", freq="D", periods=20)
    df = DataFrame(index=index, columns=range(20))
    result = df.to_html(max_rows=8, max_cols=4)
    expected = expected_html(datapath, "truncate")
    assert result == expected


@pytest.mark.parametrize("size", [1, 5])
def test_html_invalid_formatters_arg_raises(size):
    # issue-28469
    df = DataFrame(columns=["a", "b", "c"])
    msg = "Formatters length({}) should match DataFrame number of columns(3)"
    with pytest.raises(ValueError, match=re.escape(msg.format(size))):
        df.to_html(formatters=["{}".format] * size)


def test_to_html_truncate_formatter(datapath):
    # issue-25955
    data = [
        {"A": 1, "B": 2, "C": 3, "D": 4},
        {"A": 5, "B": 6, "C": 7, "D": 8},
        {"A": 9, "B": 10, "C": 11, "D": 12},
        {"A": 13, "B": 14, "C": 15, "D": 16},
    ]

    df = DataFrame(data)
    fmt = lambda x: str(x) + "_mod"
    formatters = [fmt, fmt, None, None]
    result = df.to_html(formatters=formatters, max_cols=3)
    expected = expected_html(datapath, "truncate_formatter")
    assert result == expected


@pytest.mark.parametrize(
    "sparsify,expected",
    [(True, "truncate_multi_index"), (False, "truncate_multi_index_sparse_off")],
)
def test_to_html_truncate_multi_index(sparsify, expected, datapath):
    arrays = [
        ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
        ["one", "two", "one", "two", "one", "two", "one", "two"],
    ]
    df = DataFrame(index=arrays, columns=arrays)
    result = df.to_html(max_rows=7, max_cols=7, sparsify=sparsify)
    expected = expected_html(datapath, expected)
    assert result == expected


@pytest.mark.parametrize(
    "option,result,expected",
    [
        (None, lambda df: df.to_html(), "1"),
        (None, lambda df: df.to_html(border=2), "2"),
        (2, lambda df: df.to_html(), "2"),
        (2, lambda df: df._repr_html_(), "2"),
    ],
)
def test_to_html_border(option, result, expected):
    df = DataFrame({"A": [1, 2]})
    if option is None:
        result = result(df)
    else:
        with option_context("display.html.border", option):
            result = result(df)
    expected = f'border="{expected}"'
    assert expected in result


@pytest.mark.parametrize("biggie_df_fixture", ["mixed"], indirect=True)
def test_to_html(biggie_df_fixture):
    # TODO: split this test
    df = biggie_df_fixture
    s = df.to_html()

    buf = StringIO()
    retval = df.to_html(buf=buf)
    assert retval is None
    assert buf.getvalue() == s

    assert isinstance(s, str)

    df.to_html(columns=["B", "A"], col_space=17)
    df.to_html(columns=["B", "A"], formatters={"A": lambda x: f"{x:.1f}"})

    df.to_html(columns=["B", "A"], float_format=str)
    df.to_html(columns=["B", "A"], col_space=12, float_format=str)


@pytest.mark.parametrize("biggie_df_fixture", ["empty"], indirect=True)
def test_to_html_empty_dataframe(biggie_df_fixture):
    df = biggie_df_fixture
    df.to_html()


def test_to_html_filename(biggie_df_fixture, tmpdir):
    df = biggie_df_fixture
    expected = df.to_html()
    path = tmpdir.join("test.html")
    df.to_html(path)
    result = path.read()
    assert result == expected


def test_to_html_with_no_bold():
    df = DataFrame({"x": np.random.default_rng(2).standard_normal(5)})
    html = df.to_html(bold_rows=False)
    result = html[html.find("</thead>")]
    assert "<strong" not in result


def test_to_html_columns_arg(float_frame):
    result = float_frame.to_html(columns=["A"])
    assert "<th>B</th>" not in result


@pytest.mark.parametrize(
    "columns,justify,expected",
    [
        (
            MultiIndex.from_arrays(
                [np.arange(2).repeat(2), np.mod(range(4), 2)],
                names=["CL0", "CL1"],
            ),
            "left",
            "multiindex_1",
        ),
        (
            MultiIndex.from_arrays([np.arange(4), np.mod(range(4), 2)]),
            "right",
            "multiindex_2",
        ),
    ],
)
def test_to_html_multiindex(columns, justify, expected, datapath):
    df = DataFrame([list("abcd"), list("efgh")], columns=columns)
    result = df.to_html(justify=justify)
    expected = expected_html(datapath, expected)
    assert result == expected


def test_to_html_justify(justify, datapath):
    df = DataFrame(
        {"A": [6, 30000, 2], "B": [1, 2, 70000], "C": [223442, 0, 1]},
        columns=["A", "B", "C"],
    )
    result = df.to_html(justify=justify)
    expected = expected_html(datapath, "justify").format(justify=justify)
    assert result == expected


@pytest.mark.parametrize(
    "justify", ["super-right", "small-left", "noinherit", "tiny", "pandas"]
)
def test_to_html_invalid_justify(justify):
    # GH 17527
    df = DataFrame()
    msg = "Invalid value for justify parameter"

    with pytest.raises(ValueError, match=msg):
        df.to_html(justify=justify)


class TestHTMLIndex:
    @pytest.fixture
    def df(self):
        index = ["foo", "bar", "baz"]
        df = DataFrame(
            {"A": [1, 2, 3], "B": [1.2, 3.4, 5.6], "C": ["one", "two", np.nan]},
            columns=["A", "B", "C"],
            index=index,
        )
        return df

    @pytest.fixture
    def expected_without_index(self, datapath):
        return expected_html(datapath, "index_2")

    def test_to_html_flat_index_without_name(
        self, datapath, df, expected_without_index
    ):
        expected_with_index = expected_html(datapath, "index_1")
        assert df.to_html() == expected_with_index

        result = df.to_html(index=False)
        for i in df.index:
            assert i not in result
        assert result == expected_without_index

    def test_to_html_flat_index_with_name(self, datapath, df, expected_without_index):
        df.index = Index(["foo", "bar", "baz"], name="idx")
        expected_with_index = expected_html(datapath, "index_3")
        assert df.to_html() == expected_with_index
        assert df.to_html(index=False) == expected_without_index

    def test_to_html_multiindex_without_names(
        self, datapath, df, expected_without_index
    ):
        tuples = [("foo", "car"), ("foo", "bike"), ("bar", "car")]
        df.index = MultiIndex.from_tuples(tuples)

        expected_with_index = expected_html(datapath, "index_4")
        assert df.to_html() == expected_with_index

        result = df.to_html(index=False)
        for i in ["foo", "bar", "car", "bike"]:
            assert i not in result
        # must be the same result as normal index
        assert result == expected_without_index

    def test_to_html_multiindex_with_names(self, datapath, df, expected_without_index):
        tuples = [("foo", "car"), ("foo", "bike"), ("bar", "car")]
        df.index = MultiIndex.from_tuples(tuples, names=["idx1", "idx2"])
        expected_with_index = expected_html(datapath, "index_5")
        assert df.to_html() == expected_with_index
        assert df.to_html(index=False) == expected_without_index


@pytest.mark.parametrize("classes", ["sortable draggable", ["sortable", "draggable"]])
def test_to_html_with_classes(classes, datapath):
    df = DataFrame()
    expected = expected_html(datapath, "with_classes")
    result = df.to_html(classes=classes)
    assert result == expected


def test_to_html_no_index_max_rows(datapath):
    # GH 14998
    df = DataFrame({"A": [1, 2, 3, 4]})
    result = df.to_html(index=False, max_rows=1)
    expected = expected_html(datapath, "gh14998_expected_output")
    assert result == expected


def test_to_html_multiindex_max_cols(datapath):
    # GH 6131
    index = MultiIndex(
        levels=[["ba", "bb", "bc"], ["ca", "cb", "cc"]],
        codes=[[0, 1, 2], [0, 1, 2]],
        names=["b", "c"],
    )
    columns = MultiIndex(
        levels=[["d"], ["aa", "ab", "ac"]],
        codes=[[0, 0, 0], [0, 1, 2]],
        names=[None, "a"],
    )
    data = np.array(
        [[1.0, np.nan, np.nan], [np.nan, 2.0, np.nan], [np.nan, np.nan, 3.0]]
    )
    df = DataFrame(data, index, columns)
    result = df.to_html(max_cols=2)
    expected = expected_html(datapath, "gh6131_expected_output")
    assert result == expected


def test_to_html_multi_indexes_index_false(datapath):
    # GH 22579
    df = DataFrame(
        {"a": range(10), "b": range(10, 20), "c": range(10, 20), "d": range(10, 20)}
    )
    df.columns = MultiIndex.from_product([["a", "b"], ["c", "d"]])
    df.index = MultiIndex.from_product([["a", "b"], ["c", "d", "e", "f", "g"]])
    result = df.to_html(index=False)
    expected = expected_html(datapath, "gh22579_expected_output")
    assert result == expected


@pytest.mark.parametrize("index_names", [True, False])
@pytest.mark.parametrize("header", [True, False])
@pytest.mark.parametrize("index", [True, False])
@pytest.mark.parametrize(
    "column_index, column_type",
    [
        (Index([0, 1]), "unnamed_standard"),
        (Index([0, 1], name="columns.name"), "named_standard"),
        (MultiIndex.from_product([["a"], ["b", "c"]]), "unnamed_multi"),
        (
            MultiIndex.from_product(
                [["a"], ["b", "c"]], names=["columns.name.0", "columns.name.1"]
            ),
            "named_multi",
        ),
    ],
)
@pytest.mark.parametrize(
    "row_index, row_type",
    [
        (Index([0, 1]), "unnamed_standard"),
        (Index([0, 1], name="index.name"), "named_standard"),
        (MultiIndex.from_product([["a"], ["b", "c"]]), "unnamed_multi"),
        (
            MultiIndex.from_product(
                [["a"], ["b", "c"]], names=["index.name.0", "index.name.1"]
            ),
            "named_multi",
        ),
    ],
)
def test_to_html_basic_alignment(
    datapath, row_index, row_type, column_index, column_type, index, header, index_names
):
    # GH 22747, GH 22579
    df = DataFrame(np.zeros((2, 2), dtype=int), index=row_index, columns=column_index)
    result = df.to_html(index=index, header=header, index_names=index_names)

    if not index:
        row_type = "none"
    elif not index_names and row_type.startswith("named"):
        row_type = "un" + row_type

    if not header:
        column_type = "none"
    elif not index_names and column_type.startswith("named"):
        column_type = "un" + column_type

    filename = "index_" + row_type + "_columns_" + column_type
    expected = expected_html(datapath, filename)
    assert result == expected


@pytest.mark.parametrize("index_names", [True, False])
@pytest.mark.parametrize("header", [True, False])
@pytest.mark.parametrize("index", [True, False])
@pytest.mark.parametrize(
    "column_index, column_type",
    [
        (Index(np.arange(8)), "unnamed_standard"),
        (Index(np.arange(8), name="columns.name"), "named_standard"),
        (
            MultiIndex.from_product([["a", "b"], ["c", "d"], ["e", "f"]]),
            "unnamed_multi",
        ),
        (
            MultiIndex.from_product(
                [["a", "b"], ["c", "d"], ["e", "f"]], names=["foo", None, "baz"]
            ),
            "named_multi",
        ),
    ],
)
@pytest.mark.parametrize(
    "row_index, row_type",
    [
        (Index(np.arange(8)), "unnamed_standard"),
        (Index(np.arange(8), name="index.name"), "named_standard"),
        (
            MultiIndex.from_product([["a", "b"], ["c", "d"], ["e", "f"]]),
            "unnamed_multi",
        ),
        (
            MultiIndex.from_product(
                [["a", "b"], ["c", "d"], ["e", "f"]], names=["foo", None, "baz"]
            ),
            "named_multi",
        ),
    ],
)
def test_to_html_alignment_with_truncation(
    datapath, row_index, row_type, column_index, column_type, index, header, index_names
):
    # GH 22747, GH 22579
    df = DataFrame(np.arange(64).reshape(8, 8), index=row_index, columns=column_index)
    result = df.to_html(
        max_rows=4, max_cols=4, index=index, header=header, index_names=index_names
    )

    if not index:
        row_type = "none"
    elif not index_names and row_type.startswith("named"):
        row_type = "un" + row_type

    if not header:
        column_type = "none"
    elif not index_names and column_type.startswith("named"):
        column_type = "un" + column_type

    filename = "trunc_df_index_" + row_type + "_columns_" + column_type
    expected = expected_html(datapath, filename)
    assert result == expected


@pytest.mark.parametrize("index", [False, 0])
def test_to_html_truncation_index_false_max_rows(datapath, index):
    # GH 15019
    data = [
        [1.764052, 0.400157],
        [0.978738, 2.240893],
        [1.867558, -0.977278],
        [0.950088, -0.151357],
        [-0.103219, 0.410599],
    ]
    df = DataFrame(data)
    result = df.to_html(max_rows=4, index=index)
    expected = expected_html(datapath, "gh15019_expected_output")
    assert result == expected


@pytest.mark.parametrize("index", [False, 0])
@pytest.mark.parametrize(
    "col_index_named, expected_output",
    [(False, "gh22783_expected_output"), (True, "gh22783_named_columns_index")],
)
def test_to_html_truncation_index_false_max_cols(
    datapath, index, col_index_named, expected_output
):
    # GH 22783
    data = [
        [1.764052, 0.400157, 0.978738, 2.240893, 1.867558],
        [-0.977278, 0.950088, -0.151357, -0.103219, 0.410599],
    ]
    df = DataFrame(data)
    if col_index_named:
        df.columns.rename("columns.name", inplace=True)
    result = df.to_html(max_cols=4, index=index)
    expected = expected_html(datapath, expected_output)
    assert result == expected


@pytest.mark.parametrize("notebook", [True, False])
def test_to_html_notebook_has_style(notebook):
    df = DataFrame({"A": [1, 2, 3]})
    result = df.to_html(notebook=notebook)

    if notebook:
        assert "tbody tr th:only-of-type" in result
        assert "vertical-align: middle;" in result
        assert "thead th" in result
    else:
        assert "tbody tr th:only-of-type" not in result
        assert "vertical-align: middle;" not in result
        assert "thead th" not in result


def test_to_html_with_index_names_false():
    # GH 16493
    df = DataFrame({"A": [1, 2]}, index=Index(["a", "b"], name="myindexname"))
    result = df.to_html(index_names=False)
    assert "myindexname" not in result


def test_to_html_with_id():
    # GH 8496
    df = DataFrame({"A": [1, 2]}, index=Index(["a", "b"], name="myindexname"))
    result = df.to_html(index_names=False, table_id="TEST_ID")
    assert ' id="TEST_ID"' in result


@pytest.mark.parametrize(
    "value,float_format,expected",
    [
        (0.19999, "%.3f", "gh21625_expected_output"),
        (100.0, "%.0f", "gh22270_expected_output"),
    ],
)
def test_to_html_float_format_no_fixed_width(value, float_format, expected, datapath):
    # GH 21625, GH 22270
    df = DataFrame({"x": [value]})
    expected = expected_html(datapath, expected)
    result = df.to_html(float_format=float_format)
    assert result == expected


@pytest.mark.parametrize(
    "render_links,expected",
    [(True, "render_links_true"), (False, "render_links_false")],
)
def test_to_html_render_links(render_links, expected, datapath):
    # GH 2679
    data = [
        [0, "https://pandas.pydata.org/?q1=a&q2=b", "pydata.org"],
        [0, "www.pydata.org", "pydata.org"],
    ]
    df = DataFrame(data, columns=Index(["foo", "bar", None], dtype=object))

    result = df.to_html(render_links=render_links)
    expected = expected_html(datapath, expected)
    assert result == expected


@pytest.mark.parametrize(
    "method,expected",
    [
        ("to_html", lambda x: lorem_ipsum),
        ("_repr_html_", lambda x: lorem_ipsum[: x - 4] + "..."),  # regression case
    ],
)
@pytest.mark.parametrize("max_colwidth", [10, 20, 50, 100])
def test_ignore_display_max_colwidth(method, expected, max_colwidth):
    # see gh-17004
    df = DataFrame([lorem_ipsum])
    with option_context("display.max_colwidth", max_colwidth):
        result = getattr(df, method)()
    expected = expected(max_colwidth)
    assert expected in result


@pytest.mark.parametrize("classes", [True, 0])
def test_to_html_invalid_classes_type(classes):
    # GH 25608
    df = DataFrame()
    msg = "classes must be a string, list, or tuple"

    with pytest.raises(TypeError, match=msg):
        df.to_html(classes=classes)


def test_to_html_round_column_headers():
    # GH 17280
    df = DataFrame([1], columns=[0.55555])
    with option_context("display.precision", 3):
        html = df.to_html(notebook=False)
        notebook = df.to_html(notebook=True)
    assert "0.55555" in html
    assert "0.556" in notebook


@pytest.mark.parametrize("unit", ["100px", "10%", "5em", 150])
def test_to_html_with_col_space_units(unit):
    # GH 25941
    df = DataFrame(np.random.default_rng(2).random(size=(1, 3)))
    result = df.to_html(col_space=unit)
    result = result.split("tbody")[0]
    hdrs = [x for x in result.split("\n") if re.search(r"<th[>\s]", x)]
    if isinstance(unit, int):
        unit = str(unit) + "px"
    for h in hdrs:
        expected = f'<th style="min-width: {unit};">'
        assert expected in h


class TestReprHTML:
    def test_html_repr_min_rows_default(self, datapath):
        # gh-27991

        # default setting no truncation even if above min_rows
        df = DataFrame({"a": range(20)})
        result = df._repr_html_()
        expected = expected_html(datapath, "html_repr_min_rows_default_no_truncation")
        assert result == expected

        # default of max_rows 60 triggers truncation if above
        df = DataFrame({"a": range(61)})
        result = df._repr_html_()
        expected = expected_html(datapath, "html_repr_min_rows_default_truncated")
        assert result == expected

    @pytest.mark.parametrize(
        "max_rows,min_rows,expected",
        [
            # truncated after first two rows
            (10, 4, "html_repr_max_rows_10_min_rows_4"),
            # when set to None, follow value of max_rows
            (12, None, "html_repr_max_rows_12_min_rows_None"),
            # when set value higher as max_rows, use the minimum
            (10, 12, "html_repr_max_rows_10_min_rows_12"),
            # max_rows of None -> never truncate
            (None, 12, "html_repr_max_rows_None_min_rows_12"),
        ],
    )
    def test_html_repr_min_rows(self, datapath, max_rows, min_rows, expected):
        # gh-27991

        df = DataFrame({"a": range(61)})
        expected = expected_html(datapath, expected)
        with option_context("display.max_rows", max_rows, "display.min_rows", min_rows):
            result = df._repr_html_()
        assert result == expected

    def test_repr_html_ipython_config(self, ip):
        code = textwrap.dedent(
            """\
        from pandas import DataFrame
        df = DataFrame({"A": [1, 2]})
        df._repr_html_()

        cfg = get_ipython().config
        cfg['IPKernelApp']['parent_appname']
        df._repr_html_()
        """
        )
        result = ip.run_cell(code, silent=True)
        assert not result.error_in_exec

    def test_info_repr_html(self):
        max_rows = 60
        max_cols = 20
        # Long
        h, w = max_rows + 1, max_cols - 1
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        assert r"&lt;class" not in df._repr_html_()
        with option_context("display.large_repr", "info"):
            assert r"&lt;class" in df._repr_html_()

        # Wide
        h, w = max_rows - 1, max_cols + 1
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        assert "<class" not in df._repr_html_()
        with option_context(
            "display.large_repr", "info", "display.max_columns", max_cols
        ):
            assert "&lt;class" in df._repr_html_()

    def test_fake_qtconsole_repr_html(self, float_frame):
        df = float_frame

        def get_ipython():
            return {"config": {"KernelApp": {"parent_appname": "ipython-qtconsole"}}}

        repstr = df._repr_html_()
        assert repstr is not None

        with option_context("display.max_rows", 5, "display.max_columns", 2):
            repstr = df._repr_html_()

        assert "class" in repstr  # info fallback

    def test_repr_html(self, float_frame):
        df = float_frame
        df._repr_html_()

        with option_context("display.max_rows", 1, "display.max_columns", 1):
            df._repr_html_()

        with option_context("display.notebook_repr_html", False):
            df._repr_html_()

        df = DataFrame([[1, 2], [3, 4]])
        with option_context("display.show_dimensions", True):
            assert "2 rows" in df._repr_html_()
        with option_context("display.show_dimensions", False):
            assert "2 rows" not in df._repr_html_()

    def test_repr_html_mathjax(self):
        df = DataFrame([[1, 2], [3, 4]])
        assert "tex2jax_ignore" not in df._repr_html_()

        with option_context("display.html.use_mathjax", False):
            assert "tex2jax_ignore" in df._repr_html_()

    def test_repr_html_wide(self):
        max_cols = 20
        df = DataFrame([["a" * 25] * (max_cols - 1)] * 10)
        with option_context("display.max_rows", 60, "display.max_columns", 20):
            assert "..." not in df._repr_html_()

        wide_df = DataFrame([["a" * 25] * (max_cols + 1)] * 10)
        with option_context("display.max_rows", 60, "display.max_columns", 20):
            assert "..." in wide_df._repr_html_()

    def test_repr_html_wide_multiindex_cols(self):
        max_cols = 20

        mcols = MultiIndex.from_product(
            [np.arange(max_cols // 2), ["foo", "bar"]], names=["first", "second"]
        )
        df = DataFrame([["a" * 25] * len(mcols)] * 10, columns=mcols)
        reg_repr = df._repr_html_()
        assert "..." not in reg_repr

        mcols = MultiIndex.from_product(
            (np.arange(1 + (max_cols // 2)), ["foo", "bar"]), names=["first", "second"]
        )
        df = DataFrame([["a" * 25] * len(mcols)] * 10, columns=mcols)
        with option_context("display.max_rows", 60, "display.max_columns", 20):
            assert "..." in df._repr_html_()

    def test_repr_html_long(self):
        with option_context("display.max_rows", 60):
            max_rows = get_option("display.max_rows")
            h = max_rows - 1
            df = DataFrame({"A": np.arange(1, 1 + h), "B": np.arange(41, 41 + h)})
            reg_repr = df._repr_html_()
            assert ".." not in reg_repr
            assert str(41 + max_rows // 2) in reg_repr

            h = max_rows + 1
            df = DataFrame({"A": np.arange(1, 1 + h), "B": np.arange(41, 41 + h)})
            long_repr = df._repr_html_()
            assert ".." in long_repr
            assert str(41 + max_rows // 2) not in long_repr
            assert f"{h} rows " in long_repr
            assert "2 columns" in long_repr

    def test_repr_html_float(self):
        with option_context("display.max_rows", 60):
            max_rows = get_option("display.max_rows")
            h = max_rows - 1
            df = DataFrame(
                {
                    "idx": np.linspace(-10, 10, h),
                    "A": np.arange(1, 1 + h),
                    "B": np.arange(41, 41 + h),
                }
            ).set_index("idx")
            reg_repr = df._repr_html_()
            assert ".." not in reg_repr
            assert f"<td>{40 + h}</td>" in reg_repr

            h = max_rows + 1
            df = DataFrame(
                {
                    "idx": np.linspace(-10, 10, h),
                    "A": np.arange(1, 1 + h),
                    "B": np.arange(41, 41 + h),
                }
            ).set_index("idx")
            long_repr = df._repr_html_()
            assert ".." in long_repr
            assert "<td>31</td>" not in long_repr
            assert f"{h} rows " in long_repr
            assert "2 columns" in long_repr

    def test_repr_html_long_multiindex(self):
        max_rows = 60
        max_L1 = max_rows // 2

        tuples = list(itertools.product(np.arange(max_L1), ["foo", "bar"]))
        idx = MultiIndex.from_tuples(tuples, names=["first", "second"])
        df = DataFrame(
            np.random.default_rng(2).standard_normal((max_L1 * 2, 2)),
            index=idx,
            columns=["A", "B"],
        )
        with option_context("display.max_rows", 60, "display.max_columns", 20):
            reg_repr = df._repr_html_()
        assert "..." not in reg_repr

        tuples = list(itertools.product(np.arange(max_L1 + 1), ["foo", "bar"]))
        idx = MultiIndex.from_tuples(tuples, names=["first", "second"])
        df = DataFrame(
            np.random.default_rng(2).standard_normal(((max_L1 + 1) * 2, 2)),
            index=idx,
            columns=["A", "B"],
        )
        long_repr = df._repr_html_()
        assert "..." in long_repr

    def test_repr_html_long_and_wide(self):
        max_cols = 20
        max_rows = 60

        h, w = max_rows - 1, max_cols - 1
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        with option_context("display.max_rows", 60, "display.max_columns", 20):
            assert "..." not in df._repr_html_()

        h, w = max_rows + 1, max_cols + 1
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        with option_context("display.max_rows", 60, "display.max_columns", 20):
            assert "..." in df._repr_html_()


def test_to_html_multilevel(multiindex_year_month_day_dataframe_random_data):
    ymd = multiindex_year_month_day_dataframe_random_data

    ymd.columns.name = "foo"
    ymd.to_html()
    ymd.T.to_html()


@pytest.mark.parametrize("na_rep", ["NaN", "Ted"])
def test_to_html_na_rep_and_float_format(na_rep, datapath):
    # https://github.com/pandas-dev/pandas/issues/13828
    df = DataFrame(
        [
            ["A", 1.2225],
            ["A", None],
        ],
        columns=["Group", "Data"],
    )
    result = df.to_html(na_rep=na_rep, float_format="{:.2f}".format)
    expected = expected_html(datapath, "gh13828_expected_output")
    expected = expected.format(na_rep=na_rep)
    assert result == expected


def test_to_html_na_rep_non_scalar_data(datapath):
    # GH47103
    df = DataFrame([{"a": 1, "b": [1, 2, 3]}])
    result = df.to_html(na_rep="-")
    expected = expected_html(datapath, "gh47103_expected_output")
    assert result == expected


def test_to_html_float_format_object_col(datapath):
    # GH#40024
    df = DataFrame(data={"x": [1000.0, "test"]})
    result = df.to_html(float_format=lambda x: f"{x:,.0f}")
    expected = expected_html(datapath, "gh40024_expected_output")
    assert result == expected


def test_to_html_multiindex_col_with_colspace():
    # GH#53885
    df = DataFrame([[1, 2]])
    df.columns = MultiIndex.from_tuples([(1, 1), (2, 1)])
    result = df.to_html(col_space=100)
    expected = (
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        "    <tr>\n"
        '      <th style="min-width: 100px;"></th>\n'
        '      <th style="min-width: 100px;">1</th>\n'
        '      <th style="min-width: 100px;">2</th>\n'
        "    </tr>\n"
        "    <tr>\n"
        '      <th style="min-width: 100px;"></th>\n'
        '      <th style="min-width: 100px;">1</th>\n'
        '      <th style="min-width: 100px;">1</th>\n'
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th>0</th>\n"
        "      <td>1</td>\n"
        "      <td>2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>"
    )
    assert result == expected


def test_to_html_tuple_col_with_colspace():
    # GH#53885
    df = DataFrame({("a", "b"): [1], "b": [2]})
    result = df.to_html(col_space=100)
    expected = (
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        '      <th style="min-width: 100px;"></th>\n'
        '      <th style="min-width: 100px;">(a, b)</th>\n'
        '      <th style="min-width: 100px;">b</th>\n'
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th>0</th>\n"
        "      <td>1</td>\n"
        "      <td>2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>"
    )
    assert result == expected


def test_to_html_empty_complex_array():
    # GH#54167
    df = DataFrame({"x": np.array([], dtype="complex")})
    result = df.to_html(col_space=100)
    expected = (
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        '      <th style="min-width: 100px;"></th>\n'
        '      <th style="min-width: 100px;">x</th>\n'
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "  </tbody>\n"
        "</table>"
    )
    assert result == expected


def test_to_html_pos_args_deprecation():
    # GH-54229
    df = DataFrame({"a": [1, 2, 3]})
    msg = (
        r"Starting with pandas version 3.0 all arguments of to_html except for the "
        r"argument 'buf' will be keyword-only."
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.to_html(None, None)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_scalarmath.py ===
import contextlib
import itertools
import operator
import platform
import sys
import warnings

import pytest
from hypothesis import given, settings
from hypothesis.extra import numpy as hynp
from hypothesis.strategies import sampled_from
from numpy._core._rational_tests import rational

import numpy as np
from numpy._utils import _pep440
from numpy.exceptions import ComplexWarning
from numpy.testing import (
    IS_PYPY,
    _gen_alignment_data,
    assert_,
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_raises,
    check_support_sve,
    suppress_warnings,
)

types = [np.bool, np.byte, np.ubyte, np.short, np.ushort, np.intc, np.uintc,
         np.int_, np.uint, np.longlong, np.ulonglong,
         np.single, np.double, np.longdouble, np.csingle,
         np.cdouble, np.clongdouble]

floating_types = np.floating.__subclasses__()
complex_floating_types = np.complexfloating.__subclasses__()

objecty_things = [object(), None, np.array(None, dtype=object)]

binary_operators_for_scalars = [
    operator.lt, operator.le, operator.eq, operator.ne, operator.ge,
    operator.gt, operator.add, operator.floordiv, operator.mod,
    operator.mul, operator.pow, operator.sub, operator.truediv
]
binary_operators_for_scalar_ints = binary_operators_for_scalars + [
    operator.xor, operator.or_, operator.and_
]


# This compares scalarmath against ufuncs.

class TestTypes:
    def test_types(self):
        for atype in types:
            a = atype(1)
            assert_(a == 1, f"error with {atype!r}: got {a!r}")

    def test_type_add(self):
        # list of types
        for k, atype in enumerate(types):
            a_scalar = atype(3)
            a_array = np.array([3], dtype=atype)
            for l, btype in enumerate(types):
                b_scalar = btype(1)
                b_array = np.array([1], dtype=btype)
                c_scalar = a_scalar + b_scalar
                c_array = a_array + b_array
                # It was comparing the type numbers, but the new ufunc
                # function-finding mechanism finds the lowest function
                # to which both inputs can be cast - which produces 'l'
                # when you do 'q' + 'b'.  The old function finding mechanism
                # skipped ahead based on the first argument, but that
                # does not produce properly symmetric results...
                assert_equal(c_scalar.dtype, c_array.dtype,
                           "error with types (%d/'%c' + %d/'%c')" %
                            (k, np.dtype(atype).char, l, np.dtype(btype).char))

    def test_type_create(self):
        for atype in types:
            a = np.array([1, 2, 3], atype)
            b = atype([1, 2, 3])
            assert_equal(a, b)

    def test_leak(self):
        # test leak of scalar objects
        # a leak would show up in valgrind as still-reachable of ~2.6MB
        for i in range(200000):
            np.add(1, 1)


def check_ufunc_scalar_equivalence(op, arr1, arr2):
    scalar1 = arr1[()]
    scalar2 = arr2[()]
    assert isinstance(scalar1, np.generic)
    assert isinstance(scalar2, np.generic)

    if arr1.dtype.kind == "c" or arr2.dtype.kind == "c":
        comp_ops = {operator.ge, operator.gt, operator.le, operator.lt}
        if op in comp_ops and (np.isnan(scalar1) or np.isnan(scalar2)):
            pytest.xfail("complex comp ufuncs use sort-order, scalars do not.")
    if op == operator.pow and arr2.item() in [-1, 0, 0.5, 1, 2]:
        # array**scalar special case can have different result dtype
        # (Other powers may have issues also, but are not hit here.)
        # TODO: It would be nice to resolve this issue.
        pytest.skip("array**2 can have incorrect/weird result dtype")

    # ignore fpe's since they may just mismatch for integers anyway.
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        # Comparisons DeprecationWarnings replacing errors (2022-03):
        warnings.simplefilter("error", DeprecationWarning)
        try:
            res = op(arr1, arr2)
        except Exception as e:
            with pytest.raises(type(e)):
                op(scalar1, scalar2)
        else:
            scalar_res = op(scalar1, scalar2)
            assert_array_equal(scalar_res, res, strict=True)


@pytest.mark.slow
@settings(max_examples=10000, deadline=2000)
@given(sampled_from(binary_operators_for_scalars),
       hynp.arrays(dtype=hynp.scalar_dtypes(), shape=()),
       hynp.arrays(dtype=hynp.scalar_dtypes(), shape=()))
def test_array_scalar_ufunc_equivalence(op, arr1, arr2):
    """
    This is a thorough test attempting to cover important promotion paths
    and ensuring that arrays and scalars stay as aligned as possible.
    However, if it creates troubles, it should maybe just be removed.
    """
    check_ufunc_scalar_equivalence(op, arr1, arr2)


@pytest.mark.slow
@given(sampled_from(binary_operators_for_scalars),
       hynp.scalar_dtypes(), hynp.scalar_dtypes())
def test_array_scalar_ufunc_dtypes(op, dt1, dt2):
    # Same as above, but don't worry about sampling weird values so that we
    # do not have to sample as much
    arr1 = np.array(2, dtype=dt1)
    arr2 = np.array(3, dtype=dt2)  # some power do weird things.

    check_ufunc_scalar_equivalence(op, arr1, arr2)


@pytest.mark.parametrize("fscalar", [np.float16, np.float32])
def test_int_float_promotion_truediv(fscalar):
    # Promotion for mixed int and float32/float16 must not go to float64
    i = np.int8(1)
    f = fscalar(1)
    expected = np.result_type(i, f)
    assert (i / f).dtype == expected
    assert (f / i).dtype == expected
    # But normal int / int true division goes to float64:
    assert (i / i).dtype == np.dtype("float64")
    # For int16, result has to be ast least float32 (takes ufunc path):
    assert (np.int16(1) / f).dtype == np.dtype("float32")


class TestBaseMath:
    @pytest.mark.xfail(check_support_sve(), reason="gh-22982")
    def test_blocked(self):
        # test alignments offsets for simd instructions
        # alignments for vz + 2 * (vs - 1) + 1
        for dt, sz in [(np.float32, 11), (np.float64, 7), (np.int32, 11)]:
            for out, inp1, inp2, msg in _gen_alignment_data(dtype=dt,
                                                            type='binary',
                                                            max_size=sz):
                exp1 = np.ones_like(inp1)
                inp1[...] = np.ones_like(inp1)
                inp2[...] = np.zeros_like(inp2)
                assert_almost_equal(np.add(inp1, inp2), exp1, err_msg=msg)
                assert_almost_equal(np.add(inp1, 2), exp1 + 2, err_msg=msg)
                assert_almost_equal(np.add(1, inp2), exp1, err_msg=msg)

                np.add(inp1, inp2, out=out)
                assert_almost_equal(out, exp1, err_msg=msg)

                inp2[...] += np.arange(inp2.size, dtype=dt) + 1
                assert_almost_equal(np.square(inp2),
                                    np.multiply(inp2, inp2), err_msg=msg)
                # skip true divide for ints
                if dt != np.int32:
                    assert_almost_equal(np.reciprocal(inp2),
                                        np.divide(1, inp2), err_msg=msg)

                inp1[...] = np.ones_like(inp1)
                np.add(inp1, 2, out=out)
                assert_almost_equal(out, exp1 + 2, err_msg=msg)
                inp2[...] = np.ones_like(inp2)
                np.add(2, inp2, out=out)
                assert_almost_equal(out, exp1 + 2, err_msg=msg)

    def test_lower_align(self):
        # check data that is not aligned to element size
        # i.e doubles are aligned to 4 bytes on i386
        d = np.zeros(23 * 8, dtype=np.int8)[4:-4].view(np.float64)
        o = np.zeros(23 * 8, dtype=np.int8)[4:-4].view(np.float64)
        assert_almost_equal(d + d, d * 2)
        np.add(d, d, out=o)
        np.add(np.ones_like(d), d, out=o)
        np.add(d, np.ones_like(d), out=o)
        np.add(np.ones_like(d), d)
        np.add(d, np.ones_like(d))


class TestPower:
    def test_small_types(self):
        for t in [np.int8, np.int16, np.float16]:
            a = t(3)
            b = a ** 4
            assert_(b == 81, f"error with {t!r}: got {b!r}")

    def test_large_types(self):
        for t in [np.int32, np.int64, np.float32, np.float64, np.longdouble]:
            a = t(51)
            b = a ** 4
            msg = f"error with {t!r}: got {b!r}"
            if np.issubdtype(t, np.integer):
                assert_(b == 6765201, msg)
            else:
                assert_almost_equal(b, 6765201, err_msg=msg)

    def test_integers_to_negative_integer_power(self):
        # Note that the combination of uint64 with a signed integer
        # has common type np.float64. The other combinations should all
        # raise a ValueError for integer ** negative integer.
        exp = [np.array(-1, dt)[()] for dt in 'bhilq']

        # 1 ** -1 possible special case
        base = [np.array(1, dt)[()] for dt in 'bhilqBHILQ']
        for i1, i2 in itertools.product(base, exp):
            if i1.dtype != np.uint64:
                assert_raises(ValueError, operator.pow, i1, i2)
            else:
                res = operator.pow(i1, i2)
                assert_(res.dtype.type is np.float64)
                assert_almost_equal(res, 1.)

        # -1 ** -1 possible special case
        base = [np.array(-1, dt)[()] for dt in 'bhilq']
        for i1, i2 in itertools.product(base, exp):
            if i1.dtype != np.uint64:
                assert_raises(ValueError, operator.pow, i1, i2)
            else:
                res = operator.pow(i1, i2)
                assert_(res.dtype.type is np.float64)
                assert_almost_equal(res, -1.)

        # 2 ** -1 perhaps generic
        base = [np.array(2, dt)[()] for dt in 'bhilqBHILQ']
        for i1, i2 in itertools.product(base, exp):
            if i1.dtype != np.uint64:
                assert_raises(ValueError, operator.pow, i1, i2)
            else:
                res = operator.pow(i1, i2)
                assert_(res.dtype.type is np.float64)
                assert_almost_equal(res, .5)

    def test_mixed_types(self):
        typelist = [np.int8, np.int16, np.float16,
                    np.float32, np.float64, np.int8,
                    np.int16, np.int32, np.int64]
        for t1 in typelist:
            for t2 in typelist:
                a = t1(3)
                b = t2(2)
                result = a**b
                msg = f"error with {t1!r} and {t2!r}:got {result!r}, expected {9!r}"
                if np.issubdtype(np.dtype(result), np.integer):
                    assert_(result == 9, msg)
                else:
                    assert_almost_equal(result, 9, err_msg=msg)

    def test_modular_power(self):
        # modular power is not implemented, so ensure it errors
        a = 5
        b = 4
        c = 10
        expected = pow(a, b, c)  # noqa: F841
        for t in (np.int32, np.float32, np.complex64):
            # note that 3-operand power only dispatches on the first argument
            assert_raises(TypeError, operator.pow, t(a), b, c)
            assert_raises(TypeError, operator.pow, np.array(t(a)), b, c)


def floordiv_and_mod(x, y):
    return (x // y, x % y)


def _signs(dt):
    if dt in np.typecodes['UnsignedInteger']:
        return (+1,)
    else:
        return (+1, -1)


class TestModulus:

    def test_modulus_basic(self):
        dt = np.typecodes['AllInteger'] + np.typecodes['Float']
        for op in [floordiv_and_mod, divmod]:
            for dt1, dt2 in itertools.product(dt, dt):
                for sg1, sg2 in itertools.product(_signs(dt1), _signs(dt2)):
                    fmt = 'op: %s, dt1: %s, dt2: %s, sg1: %s, sg2: %s'
                    msg = fmt % (op.__name__, dt1, dt2, sg1, sg2)
                    a = np.array(sg1 * 71, dtype=dt1)[()]
                    b = np.array(sg2 * 19, dtype=dt2)[()]
                    div, rem = op(a, b)
                    assert_equal(div * b + rem, a, err_msg=msg)
                    if sg2 == -1:
                        assert_(b < rem <= 0, msg)
                    else:
                        assert_(b > rem >= 0, msg)

    def test_float_modulus_exact(self):
        # test that float results are exact for small integers. This also
        # holds for the same integers scaled by powers of two.
        nlst = list(range(-127, 0))
        plst = list(range(1, 128))
        dividend = nlst + [0] + plst
        divisor = nlst + plst
        arg = list(itertools.product(dividend, divisor))
        tgt = [divmod(*t) for t in arg]

        a, b = np.array(arg, dtype=int).T
        # convert exact integer results from Python to float so that
        # signed zero can be used, it is checked.
        tgtdiv, tgtrem = np.array(tgt, dtype=float).T
        tgtdiv = np.where((tgtdiv == 0.0) & ((b < 0) ^ (a < 0)), -0.0, tgtdiv)
        tgtrem = np.where((tgtrem == 0.0) & (b < 0), -0.0, tgtrem)

        for op in [floordiv_and_mod, divmod]:
            for dt in np.typecodes['Float']:
                msg = f'op: {op.__name__}, dtype: {dt}'
                fa = a.astype(dt)
                fb = b.astype(dt)
                # use list comprehension so a_ and b_ are scalars
                div, rem = zip(*[op(a_, b_) for a_, b_ in zip(fa, fb)])
                assert_equal(div, tgtdiv, err_msg=msg)
                assert_equal(rem, tgtrem, err_msg=msg)

    def test_float_modulus_roundoff(self):
        # gh-6127
        dt = np.typecodes['Float']
        for op in [floordiv_and_mod, divmod]:
            for dt1, dt2 in itertools.product(dt, dt):
                for sg1, sg2 in itertools.product((+1, -1), (+1, -1)):
                    fmt = 'op: %s, dt1: %s, dt2: %s, sg1: %s, sg2: %s'
                    msg = fmt % (op.__name__, dt1, dt2, sg1, sg2)
                    a = np.array(sg1 * 78 * 6e-8, dtype=dt1)[()]
                    b = np.array(sg2 * 6e-8, dtype=dt2)[()]
                    div, rem = op(a, b)
                    # Equal assertion should hold when fmod is used
                    assert_equal(div * b + rem, a, err_msg=msg)
                    if sg2 == -1:
                        assert_(b < rem <= 0, msg)
                    else:
                        assert_(b > rem >= 0, msg)

    def test_float_modulus_corner_cases(self):
        # Check remainder magnitude.
        for dt in np.typecodes['Float']:
            b = np.array(1.0, dtype=dt)
            a = np.nextafter(np.array(0.0, dtype=dt), -b)
            rem = operator.mod(a, b)
            assert_(rem <= b, f'dt: {dt}')
            rem = operator.mod(-a, -b)
            assert_(rem >= -b, f'dt: {dt}')

        # Check nans, inf
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in remainder")
            sup.filter(RuntimeWarning, "divide by zero encountered in remainder")
            sup.filter(RuntimeWarning, "divide by zero encountered in floor_divide")
            sup.filter(RuntimeWarning, "divide by zero encountered in divmod")
            sup.filter(RuntimeWarning, "invalid value encountered in divmod")
            for dt in np.typecodes['Float']:
                fone = np.array(1.0, dtype=dt)
                fzer = np.array(0.0, dtype=dt)
                finf = np.array(np.inf, dtype=dt)
                fnan = np.array(np.nan, dtype=dt)
                rem = operator.mod(fone, fzer)
                assert_(np.isnan(rem), f'dt: {dt}')
                # MSVC 2008 returns NaN here, so disable the check.
                #rem = operator.mod(fone, finf)
                #assert_(rem == fone, 'dt: %s' % dt)
                rem = operator.mod(fone, fnan)
                assert_(np.isnan(rem), f'dt: {dt}')
                rem = operator.mod(finf, fone)
                assert_(np.isnan(rem), f'dt: {dt}')
                for op in [floordiv_and_mod, divmod]:
                    div, mod = op(fone, fzer)
                    assert_(np.isinf(div)) and assert_(np.isnan(mod))

    def test_inplace_floordiv_handling(self):
        # issue gh-12927
        # this only applies to in-place floordiv //=, because the output type
        # promotes to float which does not fit
        a = np.array([1, 2], np.int64)
        b = np.array([1, 2], np.uint64)
        with pytest.raises(TypeError,
                match=r"Cannot cast ufunc 'floor_divide' output from"):
            a //= b

class TestComparison:
    def test_comparision_different_types(self):
        x = np.array(1)
        y = np.array('s')
        eq = x == y
        neq = x != y
        assert eq is np.bool_(False)
        assert neq is np.bool_(True)


class TestComplexDivision:
    def test_zero_division(self):
        with np.errstate(all="ignore"):
            for t in [np.complex64, np.complex128]:
                a = t(0.0)
                b = t(1.0)
                assert_(np.isinf(b / a))
                b = t(complex(np.inf, np.inf))
                assert_(np.isinf(b / a))
                b = t(complex(np.inf, np.nan))
                assert_(np.isinf(b / a))
                b = t(complex(np.nan, np.inf))
                assert_(np.isinf(b / a))
                b = t(complex(np.nan, np.nan))
                assert_(np.isnan(b / a))
                b = t(0.)
                assert_(np.isnan(b / a))

    def test_signed_zeros(self):
        with np.errstate(all="ignore"):
            for t in [np.complex64, np.complex128]:
                # tupled (numerator, denominator, expected)
                # for testing as expected == numerator/denominator
                data = (
                    (( 0.0, -1.0), ( 0.0, 1.0), (-1.0, -0.0)),
                    (( 0.0, -1.0), ( 0.0, -1.0), ( 1.0, -0.0)),
                    (( 0.0, -1.0), (-0.0, -1.0), ( 1.0, 0.0)),
                    (( 0.0, -1.0), (-0.0, 1.0), (-1.0, 0.0)),
                    (( 0.0, 1.0), ( 0.0, -1.0), (-1.0, 0.0)),
                    (( 0.0, -1.0), ( 0.0, -1.0), ( 1.0, -0.0)),
                    ((-0.0, -1.0), ( 0.0, -1.0), ( 1.0, -0.0)),
                    ((-0.0, 1.0), ( 0.0, -1.0), (-1.0, -0.0))
                )
                for cases in data:
                    n = cases[0]
                    d = cases[1]
                    ex = cases[2]
                    result = t(complex(n[0], n[1])) / t(complex(d[0], d[1]))
                    # check real and imag parts separately to avoid comparison
                    # in array context, which does not account for signed zeros
                    assert_equal(result.real, ex[0])
                    assert_equal(result.imag, ex[1])

    def test_branches(self):
        with np.errstate(all="ignore"):
            for t in [np.complex64, np.complex128]:
                # tupled (numerator, denominator, expected)
                # for testing as expected == numerator/denominator
                data = []

                # trigger branch: real(fabs(denom)) > imag(fabs(denom))
                # followed by else condition as neither are == 0
                data.append((( 2.0, 1.0), ( 2.0, 1.0), (1.0, 0.0)))

                # trigger branch: real(fabs(denom)) > imag(fabs(denom))
                # followed by if condition as both are == 0
                # is performed in test_zero_division(), so this is skipped

                # trigger else if branch: real(fabs(denom)) < imag(fabs(denom))
                data.append(((1.0, 2.0), (1.0, 2.0), (1.0, 0.0)))

                for cases in data:
                    n = cases[0]
                    d = cases[1]
                    ex = cases[2]
                    result = t(complex(n[0], n[1])) / t(complex(d[0], d[1]))
                    # check real and imag parts separately to avoid comparison
                    # in array context, which does not account for signed zeros
                    assert_equal(result.real, ex[0])
                    assert_equal(result.imag, ex[1])


class TestConversion:
    def test_int_from_long(self):
        l = [1e6, 1e12, 1e18, -1e6, -1e12, -1e18]
        li = [10**6, 10**12, 10**18, -10**6, -10**12, -10**18]
        for T in [None, np.float64, np.int64]:
            a = np.array(l, dtype=T)
            assert_equal([int(_m) for _m in a], li)

        a = np.array(l[:3], dtype=np.uint64)
        assert_equal([int(_m) for _m in a], li[:3])

    def test_iinfo_long_values(self):
        for code in 'bBhH':
            with pytest.raises(OverflowError):
                np.array(np.iinfo(code).max + 1, dtype=code)

        for code in np.typecodes['AllInteger']:
            res = np.array(np.iinfo(code).max, dtype=code)
            tgt = np.iinfo(code).max
            assert_(res == tgt)

        for code in np.typecodes['AllInteger']:
            res = np.dtype(code).type(np.iinfo(code).max)
            tgt = np.iinfo(code).max
            assert_(res == tgt)

    def test_int_raise_behaviour(self):
        def overflow_error_func(dtype):
            dtype(np.iinfo(dtype).max + 1)

        for code in [np.int_, np.uint, np.longlong, np.ulonglong]:
            assert_raises(OverflowError, overflow_error_func, code)

    def test_int_from_infinite_longdouble(self):
        # gh-627
        x = np.longdouble(np.inf)
        assert_raises(OverflowError, int, x)
        with suppress_warnings() as sup:
            sup.record(ComplexWarning)
            x = np.clongdouble(np.inf)
            assert_raises(OverflowError, int, x)
            assert_equal(len(sup.log), 1)

    @pytest.mark.skipif(not IS_PYPY, reason="Test is PyPy only (gh-9972)")
    def test_int_from_infinite_longdouble___int__(self):
        x = np.longdouble(np.inf)
        assert_raises(OverflowError, x.__int__)
        with suppress_warnings() as sup:
            sup.record(ComplexWarning)
            x = np.clongdouble(np.inf)
            assert_raises(OverflowError, x.__int__)
            assert_equal(len(sup.log), 1)

    @pytest.mark.skipif(np.finfo(np.double) == np.finfo(np.longdouble),
                        reason="long double is same as double")
    @pytest.mark.skipif(platform.machine().startswith("ppc"),
                        reason="IBM double double")
    def test_int_from_huge_longdouble(self):
        # Produce a longdouble that would overflow a double,
        # use exponent that avoids bug in Darwin pow function.
        exp = np.finfo(np.double).maxexp - 1
        huge_ld = 2 * 1234 * np.longdouble(2) ** exp
        huge_i = 2 * 1234 * 2 ** exp
        assert_(huge_ld != np.inf)
        assert_equal(int(huge_ld), huge_i)

    def test_int_from_longdouble(self):
        x = np.longdouble(1.5)
        assert_equal(int(x), 1)
        x = np.longdouble(-10.5)
        assert_equal(int(x), -10)

    def test_numpy_scalar_relational_operators(self):
        # All integer
        for dt1 in np.typecodes['AllInteger']:
            assert_(1 > np.array(0, dtype=dt1)[()], f"type {dt1} failed")
            assert_(not 1 < np.array(0, dtype=dt1)[()], f"type {dt1} failed")

            for dt2 in np.typecodes['AllInteger']:
                assert_(np.array(1, dtype=dt1)[()] > np.array(0, dtype=dt2)[()],
                        f"type {dt1} and {dt2} failed")
                assert_(not np.array(1, dtype=dt1)[()] < np.array(0, dtype=dt2)[()],
                        f"type {dt1} and {dt2} failed")

        # Unsigned integers
        for dt1 in 'BHILQP':
            assert_(-1 < np.array(1, dtype=dt1)[()], f"type {dt1} failed")
            assert_(not -1 > np.array(1, dtype=dt1)[()], f"type {dt1} failed")
            assert_(-1 != np.array(1, dtype=dt1)[()], f"type {dt1} failed")

            # unsigned vs signed
            for dt2 in 'bhilqp':
                assert_(np.array(1, dtype=dt1)[()] > np.array(-1, dtype=dt2)[()],
                        f"type {dt1} and {dt2} failed")
                assert_(not np.array(1, dtype=dt1)[()] < np.array(-1, dtype=dt2)[()],
                        f"type {dt1} and {dt2} failed")
                assert_(np.array(1, dtype=dt1)[()] != np.array(-1, dtype=dt2)[()],
                        f"type {dt1} and {dt2} failed")

        # Signed integers and floats
        for dt1 in 'bhlqp' + np.typecodes['Float']:
            assert_(1 > np.array(-1, dtype=dt1)[()], f"type {dt1} failed")
            assert_(not 1 < np.array(-1, dtype=dt1)[()], f"type {dt1} failed")
            assert_(-1 == np.array(-1, dtype=dt1)[()], f"type {dt1} failed")

            for dt2 in 'bhlqp' + np.typecodes['Float']:
                assert_(np.array(1, dtype=dt1)[()] > np.array(-1, dtype=dt2)[()],
                        f"type {dt1} and {dt2} failed")
                assert_(not np.array(1, dtype=dt1)[()] < np.array(-1, dtype=dt2)[()],
                        f"type {dt1} and {dt2} failed")
                assert_(np.array(-1, dtype=dt1)[()] == np.array(-1, dtype=dt2)[()],
                        f"type {dt1} and {dt2} failed")

    def test_scalar_comparison_to_none(self):
        # Scalars should just return False and not give a warnings.
        # The comparisons are flagged by pep8, ignore that.
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', FutureWarning)
            assert_(not np.float32(1) == None)  # noqa: E711
            assert_(not np.str_('test') == None)  # noqa: E711
            # This is dubious (see below):
            assert_(not np.datetime64('NaT') == None)  # noqa: E711

            assert_(np.float32(1) != None)  # noqa: E711
            assert_(np.str_('test') != None)  # noqa: E711
            # This is dubious (see below):
            assert_(np.datetime64('NaT') != None)  # noqa: E711
        assert_(len(w) == 0)

        # For documentation purposes, this is why the datetime is dubious.
        # At the time of deprecation this was no behaviour change, but
        # it has to be considered when the deprecations are done.
        assert_(np.equal(np.datetime64('NaT'), None))


#class TestRepr:
#    def test_repr(self):
#        for t in types:
#            val = t(1197346475.0137341)
#            val_repr = repr(val)
#            val2 = eval(val_repr)
#            assert_equal( val, val2 )


class TestRepr:
    def _test_type_repr(self, t):
        finfo = np.finfo(t)
        last_fraction_bit_idx = finfo.nexp + finfo.nmant
        last_exponent_bit_idx = finfo.nexp
        storage_bytes = np.dtype(t).itemsize * 8
        # could add some more types to the list below
        for which in ['small denorm', 'small norm']:
            # Values from https://en.wikipedia.org/wiki/IEEE_754
            constr = np.array([0x00] * storage_bytes, dtype=np.uint8)
            if which == 'small denorm':
                byte = last_fraction_bit_idx // 8
                bytebit = 7 - (last_fraction_bit_idx % 8)
                constr[byte] = 1 << bytebit
            elif which == 'small norm':
                byte = last_exponent_bit_idx // 8
                bytebit = 7 - (last_exponent_bit_idx % 8)
                constr[byte] = 1 << bytebit
            else:
                raise ValueError('hmm')
            val = constr.view(t)[0]
            val_repr = repr(val)
            val2 = t(eval(val_repr))
            if not (val2 == 0 and val < 1e-100):
                assert_equal(val, val2)

    def test_float_repr(self):
        # long double test cannot work, because eval goes through a python
        # float
        for t in [np.float32, np.float64]:
            self._test_type_repr(t)


if not IS_PYPY:
    # sys.getsizeof() is not valid on PyPy
    class TestSizeOf:

        def test_equal_nbytes(self):
            for type in types:
                x = type(0)
                assert_(sys.getsizeof(x) > x.nbytes)

        def test_error(self):
            d = np.float32()
            assert_raises(TypeError, d.__sizeof__, "a")


class TestMultiply:
    def test_seq_repeat(self):
        # Test that basic sequences get repeated when multiplied with
        # numpy integers. And errors are raised when multiplied with others.
        # Some of this behaviour may be controversial and could be open for
        # change.
        accepted_types = set(np.typecodes["AllInteger"])
        deprecated_types = {'?'}
        forbidden_types = (
            set(np.typecodes["All"]) - accepted_types - deprecated_types)
        forbidden_types -= {'V'}  # can't default-construct void scalars

        for seq_type in (list, tuple):
            seq = seq_type([1, 2, 3])
            for numpy_type in accepted_types:
                i = np.dtype(numpy_type).type(2)
                assert_equal(seq * i, seq * int(i))
                assert_equal(i * seq, int(i) * seq)

            for numpy_type in deprecated_types:
                i = np.dtype(numpy_type).type()
                with assert_raises(TypeError):
                    operator.mul(seq, i)

            for numpy_type in forbidden_types:
                i = np.dtype(numpy_type).type()
                assert_raises(TypeError, operator.mul, seq, i)
                assert_raises(TypeError, operator.mul, i, seq)

    def test_no_seq_repeat_basic_array_like(self):
        # Test that an array-like which does not know how to be multiplied
        # does not attempt sequence repeat (raise TypeError).
        # See also gh-7428.
        class ArrayLike:
            def __init__(self, arr):
                self.arr = arr

            def __array__(self, dtype=None, copy=None):
                return self.arr

        # Test for simple ArrayLike above and memoryviews (original report)
        for arr_like in (ArrayLike(np.ones(3)), memoryview(np.ones(3))):
            assert_array_equal(arr_like * np.float32(3.), np.full(3, 3.))
            assert_array_equal(np.float32(3.) * arr_like, np.full(3, 3.))
            assert_array_equal(arr_like * np.int_(3), np.full(3, 3))
            assert_array_equal(np.int_(3) * arr_like, np.full(3, 3))


class TestNegative:
    def test_exceptions(self):
        a = np.ones((), dtype=np.bool)[()]
        assert_raises(TypeError, operator.neg, a)

    def test_result(self):
        types = np.typecodes['AllInteger'] + np.typecodes['AllFloat']
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            for dt in types:
                a = np.ones((), dtype=dt)[()]
                if dt in np.typecodes['UnsignedInteger']:
                    st = np.dtype(dt).type
                    max = st(np.iinfo(dt).max)
                    assert_equal(operator.neg(a), max)
                else:
                    assert_equal(operator.neg(a) + a, 0)

class TestSubtract:
    def test_exceptions(self):
        a = np.ones((), dtype=np.bool)[()]
        assert_raises(TypeError, operator.sub, a, a)

    def test_result(self):
        types = np.typecodes['AllInteger'] + np.typecodes['AllFloat']
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            for dt in types:
                a = np.ones((), dtype=dt)[()]
                assert_equal(operator.sub(a, a), 0)


class TestAbs:
    def _test_abs_func(self, absfunc, test_dtype):
        x = test_dtype(-1.5)
        assert_equal(absfunc(x), 1.5)
        x = test_dtype(0.0)
        res = absfunc(x)
        # assert_equal() checks zero signedness
        assert_equal(res, 0.0)
        x = test_dtype(-0.0)
        res = absfunc(x)
        assert_equal(res, 0.0)

        x = test_dtype(np.finfo(test_dtype).max)
        assert_equal(absfunc(x), x.real)

        with suppress_warnings() as sup:
            sup.filter(UserWarning)
            x = test_dtype(np.finfo(test_dtype).tiny)
            assert_equal(absfunc(x), x.real)

        x = test_dtype(np.finfo(test_dtype).min)
        assert_equal(absfunc(x), -x.real)

    @pytest.mark.parametrize("dtype", floating_types + complex_floating_types)
    def test_builtin_abs(self, dtype):
        if (
                sys.platform == "cygwin" and dtype == np.clongdouble and
                (
                    _pep440.parse(platform.release().split("-")[0])
                    < _pep440.Version("3.3.0")
                )
        ):
            pytest.xfail(
                reason="absl is computed in double precision on cygwin < 3.3"
            )
        self._test_abs_func(abs, dtype)

    @pytest.mark.parametrize("dtype", floating_types + complex_floating_types)
    def test_numpy_abs(self, dtype):
        if (
                sys.platform == "cygwin" and dtype == np.clongdouble and
                (
                    _pep440.parse(platform.release().split("-")[0])
                    < _pep440.Version("3.3.0")
                )
        ):
            pytest.xfail(
                reason="absl is computed in double precision on cygwin < 3.3"
            )
        self._test_abs_func(np.abs, dtype)

class TestBitShifts:

    @pytest.mark.parametrize('type_code', np.typecodes['AllInteger'])
    @pytest.mark.parametrize('op',
        [operator.rshift, operator.lshift], ids=['>>', '<<'])
    def test_shift_all_bits(self, type_code, op):
        """Shifts where the shift amount is the width of the type or wider """
        # gh-2449
        dt = np.dtype(type_code)
        nbits = dt.itemsize * 8
        for val in [5, -5]:
            for shift in [nbits, nbits + 4]:
                val_scl = np.array(val).astype(dt)[()]
                shift_scl = dt.type(shift)
                res_scl = op(val_scl, shift_scl)
                if val_scl < 0 and op is operator.rshift:
                    # sign bit is preserved
                    assert_equal(res_scl, -1)
                else:
                    assert_equal(res_scl, 0)

                # Result on scalars should be the same as on arrays
                val_arr = np.array([val_scl] * 32, dtype=dt)
                shift_arr = np.array([shift] * 32, dtype=dt)
                res_arr = op(val_arr, shift_arr)
                assert_equal(res_arr, res_scl)


class TestHash:
    @pytest.mark.parametrize("type_code", np.typecodes['AllInteger'])
    def test_integer_hashes(self, type_code):
        scalar = np.dtype(type_code).type
        for i in range(128):
            assert hash(i) == hash(scalar(i))

    @pytest.mark.parametrize("type_code", np.typecodes['AllFloat'])
    def test_float_and_complex_hashes(self, type_code):
        scalar = np.dtype(type_code).type
        for val in [np.pi, np.inf, 3, 6.]:
            numpy_val = scalar(val)
            # Cast back to Python, in case the NumPy scalar has less precision
            if numpy_val.dtype.kind == 'c':
                val = complex(numpy_val)
            else:
                val = float(numpy_val)
            assert val == numpy_val
            assert hash(val) == hash(numpy_val)

        if hash(float(np.nan)) != hash(float(np.nan)):
            # If Python distinguishes different NaNs we do so too (gh-18833)
            assert hash(scalar(np.nan)) != hash(scalar(np.nan))

    @pytest.mark.parametrize("type_code", np.typecodes['Complex'])
    def test_complex_hashes(self, type_code):
        # Test some complex valued hashes specifically:
        scalar = np.dtype(type_code).type
        for val in [np.pi + 1j, np.inf - 3j, 3j, 6. + 1j]:
            numpy_val = scalar(val)
            assert hash(complex(numpy_val)) == hash(numpy_val)


@contextlib.contextmanager
def recursionlimit(n):
    o = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(n)
        yield
    finally:
        sys.setrecursionlimit(o)


@given(sampled_from(objecty_things),
       sampled_from(binary_operators_for_scalar_ints),
       sampled_from(types + [rational]))
def test_operator_object_left(o, op, type_):
    try:
        with recursionlimit(200):
            op(o, type_(1))
    except TypeError:
        pass


@given(sampled_from(objecty_things),
       sampled_from(binary_operators_for_scalar_ints),
       sampled_from(types + [rational]))
def test_operator_object_right(o, op, type_):
    try:
        with recursionlimit(200):
            op(type_(1), o)
    except TypeError:
        pass


@given(sampled_from(binary_operators_for_scalars),
       sampled_from(types),
       sampled_from(types))
def test_operator_scalars(op, type1, type2):
    try:
        op(type1(1), type2(1))
    except TypeError:
        pass


@pytest.mark.parametrize("op", binary_operators_for_scalars)
@pytest.mark.parametrize("sctype", [np.longdouble, np.clongdouble])
def test_longdouble_operators_with_obj(sctype, op):
    # This is/used to be tricky, because NumPy generally falls back to
    # using the ufunc via `np.asarray()`, this effectively might do:
    # longdouble + None
    #   -> asarray(longdouble) + np.array(None, dtype=object)
    #   -> asarray(longdouble).astype(object) + np.array(None, dtype=object)
    # And after getting the scalars in the inner loop:
    #   -> longdouble + None
    #
    # That would recurse infinitely.  Other scalars return the python object
    # on cast, so this type of things works OK.
    #
    # As of NumPy 2.1, this has been consolidated into the np.generic binops
    # and now checks `.item()`.  That also allows the below path to work now.
    try:
        op(sctype(3), None)
    except TypeError:
        pass
    try:
        op(None, sctype(3))
    except TypeError:
        pass


@pytest.mark.parametrize("op", [operator.add, operator.pow, operator.sub])
@pytest.mark.parametrize("sctype", [np.longdouble, np.clongdouble])
def test_longdouble_with_arrlike(sctype, op):
    # As of NumPy 2.1, longdouble behaves like other types and can coerce
    # e.g. lists.  (Not necessarily better, but consistent.)
    assert_array_equal(op(sctype(3), [1, 2]), op(3, np.array([1, 2])))
    assert_array_equal(op([1, 2], sctype(3)), op(np.array([1, 2]), 3))


@pytest.mark.parametrize("op", binary_operators_for_scalars)
@pytest.mark.parametrize("sctype", [np.longdouble, np.clongdouble])
@np.errstate(all="ignore")
def test_longdouble_operators_with_large_int(sctype, op):
    # (See `test_longdouble_operators_with_obj` for why longdouble is special)
    # NEP 50 means that the result is clearly a (c)longdouble here:
    if sctype == np.clongdouble and op in [operator.mod, operator.floordiv]:
        # The above operators are not support for complex though...
        with pytest.raises(TypeError):
            op(sctype(3), 2**64)
        with pytest.raises(TypeError):
            op(sctype(3), 2**64)
    else:
        assert op(sctype(3), -2**64) == op(sctype(3), sctype(-2**64))
        assert op(2**64, sctype(3)) == op(sctype(2**64), sctype(3))


@pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
@pytest.mark.parametrize("operation", [
        lambda min, max: max + max,
        lambda min, max: min - max,
        lambda min, max: max * max], ids=["+", "-", "*"])
def test_scalar_integer_operation_overflow(dtype, operation):
    st = np.dtype(dtype).type
    min = st(np.iinfo(dtype).min)
    max = st(np.iinfo(dtype).max)

    with pytest.warns(RuntimeWarning, match="overflow encountered"):
        operation(min, max)


@pytest.mark.parametrize("dtype", np.typecodes["Integer"])
@pytest.mark.parametrize("operation", [
        lambda min, neg_1: -min,
        lambda min, neg_1: abs(min),
        lambda min, neg_1: min * neg_1,
        pytest.param(lambda min, neg_1: min // neg_1,
            marks=pytest.mark.skip(reason="broken on some platforms"))],
        ids=["neg", "abs", "*", "//"])
def test_scalar_signed_integer_overflow(dtype, operation):
    # The minimum signed integer can "overflow" for some additional operations
    st = np.dtype(dtype).type
    min = st(np.iinfo(dtype).min)
    neg_1 = st(-1)

    with pytest.warns(RuntimeWarning, match="overflow encountered"):
        operation(min, neg_1)


@pytest.mark.parametrize("dtype", np.typecodes["UnsignedInteger"])
def test_scalar_unsigned_integer_overflow(dtype):
    val = np.dtype(dtype).type(8)
    with pytest.warns(RuntimeWarning, match="overflow encountered"):
        -val

    zero = np.dtype(dtype).type(0)
    -zero  # does not warn

@pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
@pytest.mark.parametrize("operation", [
        lambda val, zero: val // zero,
        lambda val, zero: val % zero, ], ids=["//", "%"])
def test_scalar_integer_operation_divbyzero(dtype, operation):
    st = np.dtype(dtype).type
    val = st(100)
    zero = st(0)

    with pytest.warns(RuntimeWarning, match="divide by zero"):
        operation(val, zero)


ops_with_names = [
    ("__lt__", "__gt__", operator.lt, True),
    ("__le__", "__ge__", operator.le, True),
    ("__eq__", "__eq__", operator.eq, True),
    # Note __op__ and __rop__ may be identical here:
    ("__ne__", "__ne__", operator.ne, True),
    ("__gt__", "__lt__", operator.gt, True),
    ("__ge__", "__le__", operator.ge, True),
    ("__floordiv__", "__rfloordiv__", operator.floordiv, False),
    ("__truediv__", "__rtruediv__", operator.truediv, False),
    ("__add__", "__radd__", operator.add, False),
    ("__mod__", "__rmod__", operator.mod, False),
    ("__mul__", "__rmul__", operator.mul, False),
    ("__pow__", "__rpow__", operator.pow, False),
    ("__sub__", "__rsub__", operator.sub, False),
]


@pytest.mark.parametrize(["__op__", "__rop__", "op", "cmp"], ops_with_names)
@pytest.mark.parametrize("sctype", [np.float32, np.float64, np.longdouble])
def test_subclass_deferral(sctype, __op__, __rop__, op, cmp):
    """
    This test covers scalar subclass deferral.  Note that this is exceedingly
    complicated, especially since it tends to fall back to the array paths and
    these additionally add the "array priority" mechanism.

    The behaviour was modified subtly in 1.22 (to make it closer to how Python
    scalars work).  Due to its complexity and the fact that subclassing NumPy
    scalars is probably a bad idea to begin with.  There is probably room
    for adjustments here.
    """
    class myf_simple1(sctype):
        pass

    class myf_simple2(sctype):
        pass

    def op_func(self, other):
        return __op__

    def rop_func(self, other):
        return __rop__

    myf_op = type("myf_op", (sctype,), {__op__: op_func, __rop__: rop_func})

    # inheritance has to override, or this is correctly lost:
    res = op(myf_simple1(1), myf_simple2(2))
    assert type(res) == sctype or type(res) == np.bool
    assert op(myf_simple1(1), myf_simple2(2)) == op(1, 2)  # inherited

    # Two independent subclasses do not really define an order.  This could
    # be attempted, but we do not since Python's `int` does neither:
    assert op(myf_op(1), myf_simple1(2)) == __op__
    assert op(myf_simple1(1), myf_op(2)) == op(1, 2)  # inherited


def test_longdouble_complex():
    # Simple test to check longdouble and complex combinations, since these
    # need to go through promotion, which longdouble needs to be careful about.
    x = np.longdouble(1)
    assert x + 1j == 1 + 1j
    assert 1j + x == 1 + 1j


@pytest.mark.parametrize(["__op__", "__rop__", "op", "cmp"], ops_with_names)
@pytest.mark.parametrize("subtype", [float, int, complex, np.float16])
def test_pyscalar_subclasses(subtype, __op__, __rop__, op, cmp):
    # This tests that python scalar subclasses behave like a float64 (if they
    # don't override it).
    # In an earlier version of NEP 50, they behaved like the Python buildins.
    def op_func(self, other):
        return __op__

    def rop_func(self, other):
        return __rop__

    # Check that deferring is indicated using `__array_ufunc__`:
    myt = type("myt", (subtype,),
               {__op__: op_func, __rop__: rop_func, "__array_ufunc__": None})

    # Just like normally, we should never presume we can modify the float.
    assert op(myt(1), np.float64(2)) == __op__
    assert op(np.float64(1), myt(2)) == __rop__

    if op in {operator.mod, operator.floordiv} and subtype == complex:
        return  # module is not support for complex.  Do not test.

    if __rop__ == __op__:
        return

    # When no deferring is indicated, subclasses are handled normally.
    myt = type("myt", (subtype,), {__rop__: rop_func})
    behaves_like = lambda x: np.array(subtype(x))[()]

    # Check for float32, as a float subclass float64 may behave differently
    res = op(myt(1), np.float16(2))
    expected = op(behaves_like(1), np.float16(2))
    assert res == expected
    assert type(res) == type(expected)
    res = op(np.float32(2), myt(1))
    expected = op(np.float32(2), behaves_like(1))
    assert res == expected
    assert type(res) == type(expected)

    # Same check for longdouble (compare via dtype to accept float64 when
    # longdouble has the identical size), which is currently not perfectly
    # consistent.
    res = op(myt(1), np.longdouble(2))
    expected = op(behaves_like(1), np.longdouble(2))
    assert res == expected
    assert np.dtype(type(res)) == np.dtype(type(expected))
    res = op(np.float32(2), myt(1))
    expected = op(np.float32(2), behaves_like(1))
    assert res == expected
    assert np.dtype(type(res)) == np.dtype(type(expected))


def test_truediv_int():
    # This should work, as the result is float:
    assert np.uint8(3) / 123454 == np.float64(3) / 123454


@pytest.mark.slow
@pytest.mark.parametrize("op",
    # TODO: Power is a bit special, but here mostly bools seem to behave oddly
    [op for op in binary_operators_for_scalars if op is not operator.pow])
@pytest.mark.parametrize("sctype", types)
@pytest.mark.parametrize("other_type", [float, int, complex])
@pytest.mark.parametrize("rop", [True, False])
def test_scalar_matches_array_op_with_pyscalar(op, sctype, other_type, rop):
    # Check that the ufunc path matches by coercing to an array explicitly
    val1 = sctype(2)
    val2 = other_type(2)

    if rop:
        _op = op
        op = lambda x, y: _op(y, x)

    try:
        res = op(val1, val2)
    except TypeError:
        try:
            expected = op(np.asarray(val1), val2)
            raise AssertionError("ufunc didn't raise.")
        except TypeError:
            return
    else:
        expected = op(np.asarray(val1), val2)

    # Note that we only check dtype equivalency, as ufuncs may pick the lower
    # dtype if they are equivalent.
    assert res == expected
    if isinstance(val1, float) and other_type is complex and rop:
        # Python complex accepts float subclasses, so we don't get a chance
        # and the result may be a Python complex (thus, the `np.array()``)
        assert np.array(res).dtype == expected.dtype
    else:
        assert res.dtype == expected.dtype

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\test_reductions.py ===
import builtins
import datetime as dt
from string import ascii_lowercase

import numpy as np
import pytest

from pandas._libs.tslibs import iNaT

from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.missing import na_value_for_dtype

import pandas as pd
from pandas import (
    DataFrame,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
    isna,
)
import pandas._testing as tm
from pandas.util import _test_decorators as td


@pytest.mark.parametrize("agg_func", ["any", "all"])
@pytest.mark.parametrize(
    "vals",
    [
        ["foo", "bar", "baz"],
        ["foo", "", ""],
        ["", "", ""],
        [1, 2, 3],
        [1, 0, 0],
        [0, 0, 0],
        [1.0, 2.0, 3.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [True, True, True],
        [True, False, False],
        [False, False, False],
        [np.nan, np.nan, np.nan],
    ],
)
def test_groupby_bool_aggs(skipna, agg_func, vals):
    df = DataFrame({"key": ["a"] * 3 + ["b"] * 3, "val": vals * 2})

    # Figure out expectation using Python builtin
    exp = getattr(builtins, agg_func)(vals)

    # edge case for missing data with skipna and 'any'
    if skipna and all(isna(vals)) and agg_func == "any":
        exp = False

    expected = DataFrame(
        [exp] * 2, columns=["val"], index=pd.Index(["a", "b"], name="key")
    )
    result = getattr(df.groupby("key"), agg_func)(skipna=skipna)
    tm.assert_frame_equal(result, expected)


def test_any():
    df = DataFrame(
        [[1, 2, "foo"], [1, np.nan, "bar"], [3, np.nan, "baz"]],
        columns=["A", "B", "C"],
    )
    expected = DataFrame(
        [[True, True], [False, True]], columns=["B", "C"], index=[1, 3]
    )
    expected.index.name = "A"
    result = df.groupby("A").any()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("bool_agg_func", ["any", "all"])
def test_bool_aggs_dup_column_labels(bool_agg_func):
    # GH#21668
    df = DataFrame([[True, True]], columns=["a", "a"])
    grp_by = df.groupby([0])
    result = getattr(grp_by, bool_agg_func)()

    expected = df.set_axis(np.array([0]))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("bool_agg_func", ["any", "all"])
@pytest.mark.parametrize(
    "data",
    [
        [False, False, False],
        [True, True, True],
        [pd.NA, pd.NA, pd.NA],
        [False, pd.NA, False],
        [True, pd.NA, True],
        [True, pd.NA, False],
    ],
)
def test_masked_kleene_logic(bool_agg_func, skipna, data):
    # GH#37506
    ser = Series(data, dtype="boolean")

    # The result should match aggregating on the whole series. Correctness
    # there is verified in test_reductions.py::test_any_all_boolean_kleene_logic
    expected_data = getattr(ser, bool_agg_func)(skipna=skipna)
    expected = Series(expected_data, index=np.array([0]), dtype="boolean")

    result = ser.groupby([0, 0, 0]).agg(bool_agg_func, skipna=skipna)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "dtype1,dtype2,exp_col1,exp_col2",
    [
        (
            "float",
            "Float64",
            np.array([True], dtype=bool),
            pd.array([pd.NA], dtype="boolean"),
        ),
        (
            "Int64",
            "float",
            pd.array([pd.NA], dtype="boolean"),
            np.array([True], dtype=bool),
        ),
        (
            "Int64",
            "Int64",
            pd.array([pd.NA], dtype="boolean"),
            pd.array([pd.NA], dtype="boolean"),
        ),
        (
            "Float64",
            "boolean",
            pd.array([pd.NA], dtype="boolean"),
            pd.array([pd.NA], dtype="boolean"),
        ),
    ],
)
def test_masked_mixed_types(dtype1, dtype2, exp_col1, exp_col2):
    # GH#37506
    data = [1.0, np.nan]
    df = DataFrame(
        {"col1": pd.array(data, dtype=dtype1), "col2": pd.array(data, dtype=dtype2)}
    )
    result = df.groupby([1, 1]).agg("all", skipna=False)

    expected = DataFrame({"col1": exp_col1, "col2": exp_col2}, index=np.array([1]))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("bool_agg_func", ["any", "all"])
@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
def test_masked_bool_aggs_skipna(bool_agg_func, dtype, skipna, frame_or_series):
    # GH#40585
    obj = frame_or_series([pd.NA, 1], dtype=dtype)
    expected_res = True
    if not skipna and bool_agg_func == "all":
        expected_res = pd.NA
    expected = frame_or_series([expected_res], index=np.array([1]), dtype="boolean")

    result = obj.groupby([1, 1]).agg(bool_agg_func, skipna=skipna)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "bool_agg_func,data,expected_res",
    [
        ("any", [pd.NA, np.nan], False),
        ("any", [pd.NA, 1, np.nan], True),
        ("all", [pd.NA, pd.NaT], True),
        ("all", [pd.NA, False, pd.NaT], False),
    ],
)
def test_object_type_missing_vals(bool_agg_func, data, expected_res, frame_or_series):
    # GH#37501
    obj = frame_or_series(data, dtype=object)
    result = obj.groupby([1] * len(data)).agg(bool_agg_func)
    expected = frame_or_series([expected_res], index=np.array([1]), dtype="bool")
    tm.assert_equal(result, expected)


@pytest.mark.parametrize("bool_agg_func", ["any", "all"])
def test_object_NA_raises_with_skipna_false(bool_agg_func):
    # GH#37501
    ser = Series([pd.NA], dtype=object)
    with pytest.raises(TypeError, match="boolean value of NA is ambiguous"):
        ser.groupby([1]).agg(bool_agg_func, skipna=False)


@pytest.mark.parametrize("bool_agg_func", ["any", "all"])
def test_empty(frame_or_series, bool_agg_func):
    # GH 45231
    kwargs = {"columns": ["a"]} if frame_or_series is DataFrame else {"name": "a"}
    obj = frame_or_series(**kwargs, dtype=object)
    result = getattr(obj.groupby(obj.index), bool_agg_func)()
    expected = frame_or_series(**kwargs, dtype=bool)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize("how", ["idxmin", "idxmax"])
def test_idxmin_idxmax_extremes(how, any_real_numpy_dtype):
    # GH#57040
    if any_real_numpy_dtype is int or any_real_numpy_dtype is float:
        # No need to test
        return
    info = np.iinfo if "int" in any_real_numpy_dtype else np.finfo
    min_value = info(any_real_numpy_dtype).min
    max_value = info(any_real_numpy_dtype).max
    df = DataFrame(
        {"a": [2, 1, 1, 2], "b": [min_value, max_value, max_value, min_value]},
        dtype=any_real_numpy_dtype,
    )
    gb = df.groupby("a")
    result = getattr(gb, how)()
    expected = DataFrame(
        {"b": [1, 0]}, index=pd.Index([1, 2], name="a", dtype=any_real_numpy_dtype)
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("how", ["idxmin", "idxmax"])
def test_idxmin_idxmax_extremes_skipna(skipna, how, float_numpy_dtype):
    # GH#57040
    min_value = np.finfo(float_numpy_dtype).min
    max_value = np.finfo(float_numpy_dtype).max
    df = DataFrame(
        {
            "a": Series(np.repeat(range(1, 6), repeats=2), dtype="intp"),
            "b": Series(
                [
                    np.nan,
                    min_value,
                    np.nan,
                    max_value,
                    min_value,
                    np.nan,
                    max_value,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                dtype=float_numpy_dtype,
            ),
        },
    )
    gb = df.groupby("a")

    warn = None if skipna else FutureWarning
    msg = f"The behavior of DataFrameGroupBy.{how} with all-NA values"
    with tm.assert_produces_warning(warn, match=msg):
        result = getattr(gb, how)(skipna=skipna)
    if skipna:
        values = [1, 3, 4, 6, np.nan]
    else:
        values = np.nan
    expected = DataFrame(
        {"b": values}, index=pd.Index(range(1, 6), name="a", dtype="intp")
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "func, values",
    [
        ("idxmin", {"c_int": [0, 2], "c_float": [1, 3], "c_date": [1, 2]}),
        ("idxmax", {"c_int": [1, 3], "c_float": [0, 2], "c_date": [0, 3]}),
    ],
)
@pytest.mark.parametrize("numeric_only", [True, False])
def test_idxmin_idxmax_returns_int_types(func, values, numeric_only):
    # GH 25444
    df = DataFrame(
        {
            "name": ["A", "A", "B", "B"],
            "c_int": [1, 2, 3, 4],
            "c_float": [4.02, 3.03, 2.04, 1.05],
            "c_date": ["2019", "2018", "2016", "2017"],
        }
    )
    df["c_date"] = pd.to_datetime(df["c_date"])
    df["c_date_tz"] = df["c_date"].dt.tz_localize("US/Pacific")
    df["c_timedelta"] = df["c_date"] - df["c_date"].iloc[0]
    df["c_period"] = df["c_date"].dt.to_period("W")
    df["c_Integer"] = df["c_int"].astype("Int64")
    df["c_Floating"] = df["c_float"].astype("Float64")

    result = getattr(df.groupby("name"), func)(numeric_only=numeric_only)

    expected = DataFrame(values, index=pd.Index(["A", "B"], name="name"))
    if numeric_only:
        expected = expected.drop(columns=["c_date"])
    else:
        expected["c_date_tz"] = expected["c_date"]
        expected["c_timedelta"] = expected["c_date"]
        expected["c_period"] = expected["c_date"]
    expected["c_Integer"] = expected["c_int"]
    expected["c_Floating"] = expected["c_float"]

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        (
            Timestamp("2011-01-15 12:50:28.502376"),
            Timestamp("2011-01-20 12:50:28.593448"),
        ),
        (24650000000000001, 24650000000000002),
    ],
)
@pytest.mark.parametrize("method", ["count", "min", "max", "first", "last"])
def test_groupby_non_arithmetic_agg_int_like_precision(method, data):
    # GH#6620, GH#9311
    df = DataFrame({"a": [1, 1], "b": data})

    grouped = df.groupby("a")
    result = getattr(grouped, method)()
    if method == "count":
        expected_value = 2
    elif method == "first":
        expected_value = data[0]
    elif method == "last":
        expected_value = data[1]
    else:
        expected_value = getattr(df["b"], method)()
    expected = DataFrame({"b": [expected_value]}, index=pd.Index([1], name="a"))

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("how", ["first", "last"])
def test_first_last_skipna(any_real_nullable_dtype, sort, skipna, how):
    # GH#57019
    na_value = na_value_for_dtype(pandas_dtype(any_real_nullable_dtype))
    df = DataFrame(
        {
            "a": [2, 1, 1, 2, 3, 3],
            "b": [na_value, 3.0, na_value, 4.0, np.nan, np.nan],
            "c": [na_value, 3.0, na_value, 4.0, np.nan, np.nan],
        },
        dtype=any_real_nullable_dtype,
    )
    gb = df.groupby("a", sort=sort)
    method = getattr(gb, how)
    result = method(skipna=skipna)

    ilocs = {
        ("first", True): [3, 1, 4],
        ("first", False): [0, 1, 4],
        ("last", True): [3, 1, 5],
        ("last", False): [3, 2, 5],
    }[how, skipna]
    expected = df.iloc[ilocs].set_index("a")
    if sort:
        expected = expected.sort_index()
    tm.assert_frame_equal(result, expected)


def test_idxmin_idxmax_axis1():
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)), columns=["A", "B", "C", "D"]
    )
    df["A"] = [1, 2, 3, 1, 2, 3, 1, 2, 3, 4]

    gb = df.groupby("A")

    warn_msg = "DataFrameGroupBy.idxmax with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=warn_msg):
        res = gb.idxmax(axis=1)

    alt = df.iloc[:, 1:].idxmax(axis=1)
    indexer = res.index.get_level_values(1)

    tm.assert_series_equal(alt[indexer], res.droplevel("A"))

    df["E"] = date_range("2016-01-01", periods=10)
    gb2 = df.groupby("A")

    msg = "'>' not supported between instances of 'Timestamp' and 'float'"
    with pytest.raises(TypeError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=warn_msg):
            gb2.idxmax(axis=1)


def test_groupby_mean_no_overflow():
    # Regression test for (#22487)
    df = DataFrame(
        {
            "user": ["A", "A", "A", "A", "A"],
            "connections": [4970, 4749, 4719, 4704, 18446744073699999744],
        }
    )
    assert df.groupby("user")["connections"].mean()["A"] == 3689348814740003840


def test_mean_on_timedelta():
    # GH 17382
    df = DataFrame({"time": pd.to_timedelta(range(10)), "cat": ["A", "B"] * 5})
    result = df.groupby("cat")["time"].mean()
    expected = Series(
        pd.to_timedelta([4, 5]), name="time", index=pd.Index(["A", "B"], name="cat")
    )
    tm.assert_series_equal(result, expected)


def test_cython_median():
    arr = np.random.default_rng(2).standard_normal(1000)
    arr[::2] = np.nan
    df = DataFrame(arr)

    labels = np.random.default_rng(2).integers(0, 50, size=1000).astype(float)
    labels[::17] = np.nan

    result = df.groupby(labels).median()
    msg = "using DataFrameGroupBy.median"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        exp = df.groupby(labels).agg(np.nanmedian)
    tm.assert_frame_equal(result, exp)

    df = DataFrame(np.random.default_rng(2).standard_normal((1000, 5)))
    msg = "using DataFrameGroupBy.median"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rs = df.groupby(labels).agg(np.median)
    xp = df.groupby(labels).median()
    tm.assert_frame_equal(rs, xp)


def test_median_empty_bins(observed):
    df = DataFrame(np.random.default_rng(2).integers(0, 44, 500))

    grps = range(0, 55, 5)
    bins = pd.cut(df[0], grps)

    result = df.groupby(bins, observed=observed).median()
    expected = df.groupby(bins, observed=observed).agg(lambda x: x.median())
    tm.assert_frame_equal(result, expected)


def test_max_min_non_numeric():
    # #2700
    aa = DataFrame({"nn": [11, 11, 22, 22], "ii": [1, 2, 3, 4], "ss": 4 * ["mama"]})

    result = aa.groupby("nn").max()
    assert "ss" in result

    result = aa.groupby("nn").max(numeric_only=False)
    assert "ss" in result

    result = aa.groupby("nn").min()
    assert "ss" in result

    result = aa.groupby("nn").min(numeric_only=False)
    assert "ss" in result


def test_max_min_object_multiple_columns(using_array_manager, using_infer_string):
    # GH#41111 case where the aggregation is valid for some columns but not
    # others; we split object blocks column-wise, consistent with
    # DataFrame._reduce

    df = DataFrame(
        {
            "A": [1, 1, 2, 2, 3],
            "B": [1, "foo", 2, "bar", False],
            "C": ["a", "b", "c", "d", "e"],
        }
    )
    df._consolidate_inplace()  # should already be consolidate, but double-check
    if not using_array_manager:
        assert len(df._mgr.blocks) == 3 if using_infer_string else 2

    gb = df.groupby("A")

    result = gb[["C"]].max()
    # "max" is valid for column "C" but not for "B"
    ei = pd.Index([1, 2, 3], name="A")
    expected = DataFrame({"C": ["b", "d", "e"]}, index=ei)
    tm.assert_frame_equal(result, expected)

    result = gb[["C"]].min()
    # "min" is valid for column "C" but not for "B"
    ei = pd.Index([1, 2, 3], name="A")
    expected = DataFrame({"C": ["a", "c", "e"]}, index=ei)
    tm.assert_frame_equal(result, expected)


def test_min_date_with_nans():
    # GH26321
    dates = pd.to_datetime(
        Series(["2019-05-09", "2019-05-09", "2019-05-09"]), format="%Y-%m-%d"
    ).dt.date
    df = DataFrame({"a": [np.nan, "1", np.nan], "b": [0, 1, 1], "c": dates})

    result = df.groupby("b", as_index=False)["c"].min()["c"]
    expected = pd.to_datetime(
        Series(["2019-05-09", "2019-05-09"], name="c"), format="%Y-%m-%d"
    ).dt.date
    tm.assert_series_equal(result, expected)

    result = df.groupby("b")["c"].min()
    expected.index.name = "b"
    tm.assert_series_equal(result, expected)


def test_max_inat():
    # GH#40767 dont interpret iNaT as NaN
    ser = Series([1, iNaT])
    key = np.array([1, 1], dtype=np.int64)
    gb = ser.groupby(key)

    result = gb.max(min_count=2)
    expected = Series({1: 1}, dtype=np.int64)
    tm.assert_series_equal(result, expected, check_exact=True)

    result = gb.min(min_count=2)
    expected = Series({1: iNaT}, dtype=np.int64)
    tm.assert_series_equal(result, expected, check_exact=True)

    # not enough entries -> gets masked to NaN
    result = gb.min(min_count=3)
    expected = Series({1: np.nan})
    tm.assert_series_equal(result, expected, check_exact=True)


def test_max_inat_not_all_na():
    # GH#40767 dont interpret iNaT as NaN

    # make sure we dont round iNaT+1 to iNaT
    ser = Series([1, iNaT, 2, iNaT + 1])
    gb = ser.groupby([1, 2, 3, 3])
    result = gb.min(min_count=2)

    # Note: in converting to float64, the iNaT + 1 maps to iNaT, i.e. is lossy
    expected = Series({1: np.nan, 2: np.nan, 3: iNaT + 1})
    expected.index = expected.index.astype(int)
    tm.assert_series_equal(result, expected, check_exact=True)


@pytest.mark.parametrize("func", ["min", "max"])
def test_groupby_aggregate_period_column(func):
    # GH 31471
    groups = [1, 2]
    periods = pd.period_range("2020", periods=2, freq="Y")
    df = DataFrame({"a": groups, "b": periods})

    result = getattr(df.groupby("a")["b"], func)()
    idx = pd.Index([1, 2], name="a")
    expected = Series(periods, index=idx, name="b")

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", ["min", "max"])
def test_groupby_aggregate_period_frame(func):
    # GH 31471
    groups = [1, 2]
    periods = pd.period_range("2020", periods=2, freq="Y")
    df = DataFrame({"a": groups, "b": periods})

    result = getattr(df.groupby("a"), func)()
    idx = pd.Index([1, 2], name="a")
    expected = DataFrame({"b": periods}, index=idx)

    tm.assert_frame_equal(result, expected)


def test_aggregate_numeric_object_dtype():
    # https://github.com/pandas-dev/pandas/issues/39329
    # simplified case: multiple object columns where one is all-NaN
    # -> gets split as the all-NaN is inferred as float
    df = DataFrame(
        {"key": ["A", "A", "B", "B"], "col1": list("abcd"), "col2": [np.nan] * 4},
    ).astype(object)
    result = df.groupby("key").min()
    expected = (
        DataFrame(
            {"key": ["A", "B"], "col1": ["a", "c"], "col2": [np.nan, np.nan]},
        )
        .set_index("key")
        .astype(object)
    )
    tm.assert_frame_equal(result, expected)

    # same but with numbers
    df = DataFrame(
        {"key": ["A", "A", "B", "B"], "col1": list("abcd"), "col2": range(4)},
    ).astype(object)
    result = df.groupby("key").min()
    expected = (
        DataFrame({"key": ["A", "B"], "col1": ["a", "c"], "col2": [0, 2]})
        .set_index("key")
        .astype(object)
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("func", ["min", "max"])
def test_aggregate_categorical_lost_index(func: str):
    # GH: 28641 groupby drops index, when grouping over categorical column with min/max
    ds = Series(["b"], dtype="category").cat.as_ordered()
    df = DataFrame({"A": [1997], "B": ds})
    result = df.groupby("A").agg({"B": func})
    expected = DataFrame({"B": ["b"]}, index=pd.Index([1997], name="A"))

    # ordered categorical dtype should be preserved
    expected["B"] = expected["B"].astype(ds.dtype)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", ["Int64", "Int32", "Float64", "Float32", "boolean"])
def test_groupby_min_max_nullable(dtype):
    if dtype == "Int64":
        # GH#41743 avoid precision loss
        ts = 1618556707013635762
    elif dtype == "boolean":
        ts = 0
    else:
        ts = 4.0

    df = DataFrame({"id": [2, 2], "ts": [ts, ts + 1]})
    df["ts"] = df["ts"].astype(dtype)

    gb = df.groupby("id")

    result = gb.min()
    expected = df.iloc[:1].set_index("id")
    tm.assert_frame_equal(result, expected)

    res_max = gb.max()
    expected_max = df.iloc[1:].set_index("id")
    tm.assert_frame_equal(res_max, expected_max)

    result2 = gb.min(min_count=3)
    expected2 = DataFrame({"ts": [pd.NA]}, index=expected.index, dtype=dtype)
    tm.assert_frame_equal(result2, expected2)

    res_max2 = gb.max(min_count=3)
    tm.assert_frame_equal(res_max2, expected2)

    # Case with NA values
    df2 = DataFrame({"id": [2, 2, 2], "ts": [ts, pd.NA, ts + 1]})
    df2["ts"] = df2["ts"].astype(dtype)
    gb2 = df2.groupby("id")

    result3 = gb2.min()
    tm.assert_frame_equal(result3, expected)

    res_max3 = gb2.max()
    tm.assert_frame_equal(res_max3, expected_max)

    result4 = gb2.min(min_count=100)
    tm.assert_frame_equal(result4, expected2)

    res_max4 = gb2.max(min_count=100)
    tm.assert_frame_equal(res_max4, expected2)


def test_min_max_nullable_uint64_empty_group():
    # don't raise NotImplementedError from libgroupby
    cat = pd.Categorical([0] * 10, categories=[0, 1])
    df = DataFrame({"A": cat, "B": pd.array(np.arange(10, dtype=np.uint64))})
    gb = df.groupby("A", observed=False)

    res = gb.min()

    idx = pd.CategoricalIndex([0, 1], dtype=cat.dtype, name="A")
    expected = DataFrame({"B": pd.array([0, pd.NA], dtype="UInt64")}, index=idx)
    tm.assert_frame_equal(res, expected)

    res = gb.max()
    expected.iloc[0, 0] = 9
    tm.assert_frame_equal(res, expected)


@pytest.mark.parametrize("func", ["first", "last", "min", "max"])
def test_groupby_min_max_categorical(func):
    # GH: 52151
    df = DataFrame(
        {
            "col1": pd.Categorical(["A"], categories=list("AB"), ordered=True),
            "col2": pd.Categorical([1], categories=[1, 2], ordered=True),
            "value": 0.1,
        }
    )
    result = getattr(df.groupby("col1", observed=False), func)()

    idx = pd.CategoricalIndex(data=["A", "B"], name="col1", ordered=True)
    expected = DataFrame(
        {
            "col2": pd.Categorical([1, None], categories=[1, 2], ordered=True),
            "value": [0.1, None],
        },
        index=idx,
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("func", ["min", "max"])
def test_min_empty_string_dtype(func, string_dtype_no_object):
    # GH#55619
    dtype = string_dtype_no_object
    df = DataFrame({"a": ["a"], "b": "a", "c": "a"}, dtype=dtype).iloc[:0]
    result = getattr(df.groupby("a"), func)()
    expected = DataFrame(
        columns=["b", "c"], dtype=dtype, index=pd.Index([], dtype=dtype, name="a")
    )
    tm.assert_frame_equal(result, expected)


def test_max_nan_bug():
    df = DataFrame(
        {
            "Unnamed: 0": ["-04-23", "-05-06", "-05-07"],
            "Date": [
                "2013-04-23 00:00:00",
                "2013-05-06 00:00:00",
                "2013-05-07 00:00:00",
            ],
            "app": Series([np.nan, np.nan, "OE"]),
            "File": ["log080001.log", "log.log", "xlsx"],
        }
    )
    gb = df.groupby("Date")
    r = gb[["File"]].max()
    e = gb["File"].max().to_frame()
    tm.assert_frame_equal(r, e)
    assert not r["File"].isna().any()


@pytest.mark.slow
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("dropna", [False, True])
@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize("with_nan", [True, False])
@pytest.mark.parametrize("keys", [["joe"], ["joe", "jim"]])
def test_series_groupby_nunique(sort, dropna, as_index, with_nan, keys):
    n = 100
    m = 10
    days = date_range("2015-08-23", periods=10)
    df = DataFrame(
        {
            "jim": np.random.default_rng(2).choice(list(ascii_lowercase), n),
            "joe": np.random.default_rng(2).choice(days, n),
            "julie": np.random.default_rng(2).integers(0, m, n),
        }
    )
    if with_nan:
        df = df.astype({"julie": float})  # Explicit cast to avoid implicit cast below
        df.loc[1::17, "jim"] = None
        df.loc[3::37, "joe"] = None
        df.loc[7::19, "julie"] = None
        df.loc[8::19, "julie"] = None
        df.loc[9::19, "julie"] = None
    original_df = df.copy()
    gr = df.groupby(keys, as_index=as_index, sort=sort)
    left = gr["julie"].nunique(dropna=dropna)

    gr = df.groupby(keys, as_index=as_index, sort=sort)
    right = gr["julie"].apply(Series.nunique, dropna=dropna)
    if not as_index:
        right = right.reset_index(drop=True)

    if as_index:
        tm.assert_series_equal(left, right, check_names=False)
    else:
        tm.assert_frame_equal(left, right, check_names=False)
    tm.assert_frame_equal(df, original_df)


def test_nunique():
    df = DataFrame({"A": list("abbacc"), "B": list("abxacc"), "C": list("abbacx")})

    expected = DataFrame({"A": list("abc"), "B": [1, 2, 1], "C": [1, 1, 2]})
    result = df.groupby("A", as_index=False).nunique()
    tm.assert_frame_equal(result, expected)

    # as_index
    expected.index = list("abc")
    expected.index.name = "A"
    expected = expected.drop(columns="A")
    result = df.groupby("A").nunique()
    tm.assert_frame_equal(result, expected)

    # with na
    result = df.replace({"x": None}).groupby("A").nunique(dropna=False)
    tm.assert_frame_equal(result, expected)

    # dropna
    expected = DataFrame({"B": [1] * 3, "C": [1] * 3}, index=list("abc"))
    expected.index.name = "A"
    result = df.replace({"x": None}).groupby("A").nunique()
    tm.assert_frame_equal(result, expected)


def test_nunique_with_object():
    # GH 11077
    data = DataFrame(
        [
            [100, 1, "Alice"],
            [200, 2, "Bob"],
            [300, 3, "Charlie"],
            [-400, 4, "Dan"],
            [500, 5, "Edith"],
        ],
        columns=["amount", "id", "name"],
    )

    result = data.groupby(["id", "amount"])["name"].nunique()
    index = MultiIndex.from_arrays([data.id, data.amount])
    expected = Series([1] * 5, name="name", index=index)
    tm.assert_series_equal(result, expected)


def test_nunique_with_empty_series():
    # GH 12553
    data = Series(name="name", dtype=object)
    result = data.groupby(level=0).nunique()
    expected = Series(name="name", dtype="int64")
    tm.assert_series_equal(result, expected)


def test_nunique_with_timegrouper():
    # GH 13453
    test = DataFrame(
        {
            "time": [
                Timestamp("2016-06-28 09:35:35"),
                Timestamp("2016-06-28 16:09:30"),
                Timestamp("2016-06-28 16:46:28"),
            ],
            "data": ["1", "2", "3"],
        }
    ).set_index("time")
    result = test.groupby(pd.Grouper(freq="h"))["data"].nunique()
    expected = test.groupby(pd.Grouper(freq="h"))["data"].apply(Series.nunique)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "key, data, dropna, expected",
    [
        (
            ["x", "x", "x"],
            [Timestamp("2019-01-01"), pd.NaT, Timestamp("2019-01-01")],
            True,
            Series([1], index=pd.Index(["x"], name="key"), name="data"),
        ),
        (
            ["x", "x", "x"],
            [dt.date(2019, 1, 1), pd.NaT, dt.date(2019, 1, 1)],
            True,
            Series([1], index=pd.Index(["x"], name="key"), name="data"),
        ),
        (
            ["x", "x", "x", "y", "y"],
            [
                dt.date(2019, 1, 1),
                pd.NaT,
                dt.date(2019, 1, 1),
                pd.NaT,
                dt.date(2019, 1, 1),
            ],
            False,
            Series([2, 2], index=pd.Index(["x", "y"], name="key"), name="data"),
        ),
        (
            ["x", "x", "x", "x", "y"],
            [
                dt.date(2019, 1, 1),
                pd.NaT,
                dt.date(2019, 1, 1),
                pd.NaT,
                dt.date(2019, 1, 1),
            ],
            False,
            Series([2, 1], index=pd.Index(["x", "y"], name="key"), name="data"),
        ),
    ],
)
def test_nunique_with_NaT(key, data, dropna, expected):
    # GH 27951
    df = DataFrame({"key": key, "data": data})
    result = df.groupby(["key"])["data"].nunique(dropna=dropna)
    tm.assert_series_equal(result, expected)


def test_nunique_preserves_column_level_names():
    # GH 23222
    test = DataFrame([1, 2, 2], columns=pd.Index(["A"], name="level_0"))
    result = test.groupby([0, 0, 0]).nunique()
    expected = DataFrame([2], index=np.array([0]), columns=test.columns)
    tm.assert_frame_equal(result, expected)


def test_nunique_transform_with_datetime():
    # GH 35109 - transform with nunique on datetimes results in integers
    df = DataFrame(date_range("2008-12-31", "2009-01-02"), columns=["date"])
    result = df.groupby([0, 0, 1])["date"].transform("nunique")
    expected = Series([2, 2, 1], name="date")
    tm.assert_series_equal(result, expected)


def test_empty_categorical(observed):
    # GH#21334
    cat = Series([1]).astype("category")
    ser = cat[:0]
    gb = ser.groupby(ser, observed=observed)
    result = gb.nunique()
    if observed:
        expected = Series([], index=cat[:0], dtype="int64")
    else:
        expected = Series([0], index=cat, dtype="int64")
    tm.assert_series_equal(result, expected)


def test_intercept_builtin_sum():
    s = Series([1.0, 2.0, np.nan, 3.0])
    grouped = s.groupby([0, 1, 2, 2])

    msg = "using SeriesGroupBy.sum"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        # GH#53425
        result = grouped.agg(builtins.sum)
    msg = "using np.sum"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        # GH#53425
        result2 = grouped.apply(builtins.sum)
    expected = grouped.sum()
    tm.assert_series_equal(result, expected)
    tm.assert_series_equal(result2, expected)


@pytest.mark.parametrize("min_count", [0, 10])
def test_groupby_sum_mincount_boolean(min_count):
    b = True
    a = False
    na = np.nan
    dfg = pd.array([b, b, na, na, a, a, b], dtype="boolean")

    df = DataFrame({"A": [1, 1, 2, 2, 3, 3, 1], "B": dfg})
    result = df.groupby("A").sum(min_count=min_count)
    if min_count == 0:
        expected = DataFrame(
            {"B": pd.array([3, 0, 0], dtype="Int64")},
            index=pd.Index([1, 2, 3], name="A"),
        )
        tm.assert_frame_equal(result, expected)
    else:
        expected = DataFrame(
            {"B": pd.array([pd.NA] * 3, dtype="Int64")},
            index=pd.Index([1, 2, 3], name="A"),
        )
        tm.assert_frame_equal(result, expected)


def test_groupby_sum_below_mincount_nullable_integer():
    # https://github.com/pandas-dev/pandas/issues/32861
    df = DataFrame({"a": [0, 1, 2], "b": [0, 1, 2], "c": [0, 1, 2]}, dtype="Int64")
    grouped = df.groupby("a")
    idx = pd.Index([0, 1, 2], name="a", dtype="Int64")

    result = grouped["b"].sum(min_count=2)
    expected = Series([pd.NA] * 3, dtype="Int64", index=idx, name="b")
    tm.assert_series_equal(result, expected)

    result = grouped.sum(min_count=2)
    expected = DataFrame({"b": [pd.NA] * 3, "c": [pd.NA] * 3}, dtype="Int64", index=idx)
    tm.assert_frame_equal(result, expected)


def test_groupby_sum_timedelta_with_nat():
    # GH#42659
    df = DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [pd.Timedelta("1d"), pd.Timedelta("2d"), pd.Timedelta("3d"), pd.NaT],
        }
    )
    td3 = pd.Timedelta(days=3)

    gb = df.groupby("a")

    res = gb.sum()
    expected = DataFrame({"b": [td3, td3]}, index=pd.Index([1, 2], name="a"))
    tm.assert_frame_equal(res, expected)

    res = gb["b"].sum()
    tm.assert_series_equal(res, expected["b"])

    res = gb["b"].sum(min_count=2)
    expected = Series([td3, pd.NaT], dtype="m8[ns]", name="b", index=expected.index)
    tm.assert_series_equal(res, expected)


@pytest.mark.parametrize(
    "dtype", ["int8", "int16", "int32", "int64", "float32", "float64", "uint64"]
)
@pytest.mark.parametrize(
    "method,data",
    [
        ("first", {"df": [{"a": 1, "b": 1}, {"a": 2, "b": 3}]}),
        ("last", {"df": [{"a": 1, "b": 2}, {"a": 2, "b": 4}]}),
        ("min", {"df": [{"a": 1, "b": 1}, {"a": 2, "b": 3}]}),
        ("max", {"df": [{"a": 1, "b": 2}, {"a": 2, "b": 4}]}),
        ("count", {"df": [{"a": 1, "b": 2}, {"a": 2, "b": 2}], "out_type": "int64"}),
    ],
)
def test_groupby_non_arithmetic_agg_types(dtype, method, data):
    # GH9311, GH6620
    df = DataFrame(
        [{"a": 1, "b": 1}, {"a": 1, "b": 2}, {"a": 2, "b": 3}, {"a": 2, "b": 4}]
    )

    df["b"] = df.b.astype(dtype)

    if "args" not in data:
        data["args"] = []

    if "out_type" in data:
        out_type = data["out_type"]
    else:
        out_type = dtype

    exp = data["df"]
    df_out = DataFrame(exp)

    df_out["b"] = df_out.b.astype(out_type)
    df_out.set_index("a", inplace=True)

    grpd = df.groupby("a")
    t = getattr(grpd, method)(*data["args"])
    tm.assert_frame_equal(t, df_out)


def scipy_sem(*args, **kwargs):
    from scipy.stats import sem

    return sem(*args, ddof=1, **kwargs)


@pytest.mark.parametrize(
    "op,targop",
    [
        ("mean", np.mean),
        ("median", np.median),
        ("std", np.std),
        ("var", np.var),
        ("sum", np.sum),
        ("prod", np.prod),
        ("min", np.min),
        ("max", np.max),
        ("first", lambda x: x.iloc[0]),
        ("last", lambda x: x.iloc[-1]),
        ("count", np.size),
        pytest.param("sem", scipy_sem, marks=td.skip_if_no("scipy")),
    ],
)
def test_ops_general(op, targop):
    df = DataFrame(np.random.default_rng(2).standard_normal(1000))
    labels = np.random.default_rng(2).integers(0, 50, size=1000).astype(float)

    result = getattr(df.groupby(labels), op)()
    warn = None if op in ("first", "last", "count", "sem") else FutureWarning
    msg = f"using DataFrameGroupBy.{op}"
    with tm.assert_produces_warning(warn, match=msg):
        expected = df.groupby(labels).agg(targop)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "values",
    [
        {
            "a": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "b": [1, pd.NA, 2, 1, pd.NA, 2, 1, pd.NA, 2],
        },
        {"a": [1, 1, 2, 2, 3, 3], "b": [1, 2, 1, 2, 1, 2]},
    ],
)
@pytest.mark.parametrize("function", ["mean", "median", "var"])
def test_apply_to_nullable_integer_returns_float(values, function):
    # https://github.com/pandas-dev/pandas/issues/32219
    output = 0.5 if function == "var" else 1.5
    arr = np.array([output] * 3, dtype=float)
    idx = pd.Index([1, 2, 3], name="a", dtype="Int64")
    expected = DataFrame({"b": arr}, index=idx).astype("Float64")

    groups = DataFrame(values, dtype="Int64").groupby("a")

    result = getattr(groups, function)()
    tm.assert_frame_equal(result, expected)

    result = groups.agg(function)
    tm.assert_frame_equal(result, expected)

    result = groups.agg([function])
    expected.columns = MultiIndex.from_tuples([("b", function)])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "op",
    [
        "sum",
        "prod",
        "min",
        "max",
        "median",
        "mean",
        "skew",
        "std",
        "var",
        "sem",
    ],
)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("sort", [True, False])
def test_regression_allowlist_methods(op, axis, skipna, sort):
    # GH6944
    # GH 17537
    # explicitly test the allowlist methods
    raw_frame = DataFrame([0])
    if axis == 0:
        frame = raw_frame
        msg = "The 'axis' keyword in DataFrame.groupby is deprecated and will be"
    else:
        frame = raw_frame.T
        msg = "DataFrame.groupby with axis=1 is deprecated"

    with tm.assert_produces_warning(FutureWarning, match=msg):
        grouped = frame.groupby(level=0, axis=axis, sort=sort)

    if op == "skew":
        # skew has skipna
        result = getattr(grouped, op)(skipna=skipna)
        expected = frame.groupby(level=0).apply(
            lambda h: getattr(h, op)(axis=axis, skipna=skipna)
        )
        if sort:
            expected = expected.sort_index(axis=axis)
        tm.assert_frame_equal(result, expected)
    else:
        result = getattr(grouped, op)()
        expected = frame.groupby(level=0).apply(lambda h: getattr(h, op)(axis=axis))
        if sort:
            expected = expected.sort_index(axis=axis)
        tm.assert_frame_equal(result, expected)


def test_groupby_prod_with_int64_dtype():
    # GH#46573
    data = [
        [1, 11],
        [1, 41],
        [1, 17],
        [1, 37],
        [1, 7],
        [1, 29],
        [1, 31],
        [1, 2],
        [1, 3],
        [1, 43],
        [1, 5],
        [1, 47],
        [1, 19],
        [1, 88],
    ]
    df = DataFrame(data, columns=["A", "B"], dtype="int64")
    result = df.groupby(["A"]).prod().reset_index()
    expected = DataFrame({"A": [1], "B": [180970905912331920]}, dtype="int64")
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\test_indexing.py ===
""" test fancy indexing & misc """

import array
from datetime import datetime
import re
import weakref

import numpy as np
import pytest

from pandas.errors import IndexingError

from pandas.core.dtypes.common import (
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
)

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    NaT,
    Series,
    date_range,
    offsets,
    timedelta_range,
)
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
from pandas.tests.indexing.test_floats import gen_obj

# ------------------------------------------------------------------------
# Indexing test cases


class TestFancy:
    """pure get/set item & fancy indexing"""

    def test_setitem_ndarray_1d(self):
        # GH5508

        # len of indexer vs length of the 1d ndarray
        df = DataFrame(index=Index(np.arange(1, 11), dtype=np.int64))
        df["foo"] = np.zeros(10, dtype=np.float64)
        df["bar"] = np.zeros(10, dtype=complex)

        # invalid
        msg = "Must have equal len keys and value when setting with an iterable"
        with pytest.raises(ValueError, match=msg):
            df.loc[df.index[2:5], "bar"] = np.array([2.33j, 1.23 + 0.1j, 2.2, 1.0])

        # valid
        df.loc[df.index[2:6], "bar"] = np.array([2.33j, 1.23 + 0.1j, 2.2, 1.0])

        result = df.loc[df.index[2:6], "bar"]
        expected = Series(
            [2.33j, 1.23 + 0.1j, 2.2, 1.0], index=[3, 4, 5, 6], name="bar"
        )
        tm.assert_series_equal(result, expected)

    def test_setitem_ndarray_1d_2(self):
        # GH5508

        # dtype getting changed?
        df = DataFrame(index=Index(np.arange(1, 11)))
        df["foo"] = np.zeros(10, dtype=np.float64)
        df["bar"] = np.zeros(10, dtype=complex)

        msg = "Must have equal len keys and value when setting with an iterable"
        with pytest.raises(ValueError, match=msg):
            df[2:5] = np.arange(1, 4) * 1j

    @pytest.mark.filterwarnings(
        "ignore:Series.__getitem__ treating keys as positions is deprecated:"
        "FutureWarning"
    )
    def test_getitem_ndarray_3d(
        self, index, frame_or_series, indexer_sli, using_array_manager
    ):
        # GH 25567
        obj = gen_obj(frame_or_series, index)
        idxr = indexer_sli(obj)
        nd3 = np.random.default_rng(2).integers(5, size=(2, 2, 2))

        msgs = []
        if frame_or_series is Series and indexer_sli in [tm.setitem, tm.iloc]:
            msgs.append(r"Wrong number of dimensions. values.ndim > ndim \[3 > 1\]")
            if using_array_manager:
                msgs.append("Passed array should be 1-dimensional")
        if frame_or_series is Series or indexer_sli is tm.iloc:
            msgs.append(r"Buffer has wrong number of dimensions \(expected 1, got 3\)")
            if using_array_manager:
                msgs.append("indexer should be 1-dimensional")
        if indexer_sli is tm.loc or (
            frame_or_series is Series and indexer_sli is tm.setitem
        ):
            msgs.append("Cannot index with multidimensional key")
        if frame_or_series is DataFrame and indexer_sli is tm.setitem:
            msgs.append("Index data must be 1-dimensional")
        if isinstance(index, pd.IntervalIndex) and indexer_sli is tm.iloc:
            msgs.append("Index data must be 1-dimensional")
        if isinstance(index, (pd.TimedeltaIndex, pd.DatetimeIndex, pd.PeriodIndex)):
            msgs.append("Data must be 1-dimensional")
        if len(index) == 0 or isinstance(index, pd.MultiIndex):
            msgs.append("positional indexers are out-of-bounds")
        if type(index) is Index and not isinstance(index._values, np.ndarray):
            # e.g. Int64
            msgs.append("values must be a 1D array")

            # string[pyarrow]
            msgs.append("only handle 1-dimensional arrays")

        msg = "|".join(msgs)

        potential_errors = (IndexError, ValueError, NotImplementedError)
        with pytest.raises(potential_errors, match=msg):
            idxr[nd3]

    @pytest.mark.filterwarnings(
        "ignore:Series.__setitem__ treating keys as positions is deprecated:"
        "FutureWarning"
    )
    def test_setitem_ndarray_3d(self, index, frame_or_series, indexer_sli):
        # GH 25567
        obj = gen_obj(frame_or_series, index)
        idxr = indexer_sli(obj)
        nd3 = np.random.default_rng(2).integers(5, size=(2, 2, 2))

        if indexer_sli is tm.iloc:
            err = ValueError
            msg = f"Cannot set values with ndim > {obj.ndim}"
        else:
            err = ValueError
            msg = "|".join(
                [
                    r"Buffer has wrong number of dimensions \(expected 1, got 3\)",
                    "Cannot set values with ndim > 1",
                    "Index data must be 1-dimensional",
                    "Data must be 1-dimensional",
                    "Array conditional must be same shape as self",
                ]
            )

        with pytest.raises(err, match=msg):
            idxr[nd3] = 0

    def test_getitem_ndarray_0d(self):
        # GH#24924
        key = np.array(0)

        # dataframe __getitem__
        df = DataFrame([[1, 2], [3, 4]])
        result = df[key]
        expected = Series([1, 3], name=0)
        tm.assert_series_equal(result, expected)

        # series __getitem__
        ser = Series([1, 2])
        result = ser[key]
        assert result == 1

    def test_inf_upcast(self):
        # GH 16957
        # We should be able to use np.inf as a key
        # np.inf should cause an index to convert to float

        # Test with np.inf in rows
        df = DataFrame(columns=[0])
        df.loc[1] = 1
        df.loc[2] = 2
        df.loc[np.inf] = 3

        # make sure we can look up the value
        assert df.loc[np.inf, 0] == 3

        result = df.index
        expected = Index([1, 2, np.inf], dtype=np.float64)
        tm.assert_index_equal(result, expected)

    def test_setitem_dtype_upcast(self):
        # GH3216
        df = DataFrame([{"a": 1}, {"a": 3, "b": 2}])
        df["c"] = np.nan
        assert df["c"].dtype == np.float64

        with tm.assert_produces_warning(
            FutureWarning, match="item of incompatible dtype"
        ):
            df.loc[0, "c"] = "foo"
        expected = DataFrame(
            {"a": [1, 3], "b": [np.nan, 2], "c": Series(["foo", np.nan], dtype=object)}
        )
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("val", [3.14, "wxyz"])
    def test_setitem_dtype_upcast2(self, val):
        # GH10280
        df = DataFrame(
            np.arange(6, dtype="int64").reshape(2, 3),
            index=list("ab"),
            columns=["foo", "bar", "baz"],
        )

        left = df.copy()
        with tm.assert_produces_warning(
            FutureWarning, match="item of incompatible dtype"
        ):
            left.loc["a", "bar"] = val
        right = DataFrame(
            [[0, val, 2], [3, 4, 5]],
            index=list("ab"),
            columns=["foo", "bar", "baz"],
        )

        tm.assert_frame_equal(left, right)
        assert is_integer_dtype(left["foo"])
        assert is_integer_dtype(left["baz"])

    def test_setitem_dtype_upcast3(self):
        left = DataFrame(
            np.arange(6, dtype="int64").reshape(2, 3) / 10.0,
            index=list("ab"),
            columns=["foo", "bar", "baz"],
        )
        with tm.assert_produces_warning(
            FutureWarning, match="item of incompatible dtype"
        ):
            left.loc["a", "bar"] = "wxyz"

        right = DataFrame(
            [[0, "wxyz", 0.2], [0.3, 0.4, 0.5]],
            index=list("ab"),
            columns=["foo", "bar", "baz"],
        )

        tm.assert_frame_equal(left, right)
        assert is_float_dtype(left["foo"])
        assert is_float_dtype(left["baz"])

    def test_dups_fancy_indexing(self):
        # GH 3455

        df = DataFrame(np.eye(3), columns=["a", "a", "b"])
        result = df[["b", "a"]].columns
        expected = Index(["b", "a", "a"])
        tm.assert_index_equal(result, expected)

    def test_dups_fancy_indexing_across_dtypes(self):
        # across dtypes
        df = DataFrame([[1, 2, 1.0, 2.0, 3.0, "foo", "bar"]], columns=list("aaaaaaa"))
        result = DataFrame([[1, 2, 1.0, 2.0, 3.0, "foo", "bar"]])
        result.columns = list("aaaaaaa")  # GH#3468

        # GH#3509 smoke tests for indexing with duplicate columns
        df.iloc[:, 4]
        result.iloc[:, 4]

        tm.assert_frame_equal(df, result)

    def test_dups_fancy_indexing_not_in_order(self):
        # GH 3561, dups not in selected order
        df = DataFrame(
            {"test": [5, 7, 9, 11], "test1": [4.0, 5, 6, 7], "other": list("abcd")},
            index=["A", "A", "B", "C"],
        )
        rows = ["C", "B"]
        expected = DataFrame(
            {"test": [11, 9], "test1": [7.0, 6], "other": ["d", "c"]}, index=rows
        )
        result = df.loc[rows]
        tm.assert_frame_equal(result, expected)

        result = df.loc[Index(rows)]
        tm.assert_frame_equal(result, expected)

        rows = ["C", "B", "E"]
        with pytest.raises(KeyError, match="not in index"):
            df.loc[rows]

        # see GH5553, make sure we use the right indexer
        rows = ["F", "G", "H", "C", "B", "E"]
        with pytest.raises(KeyError, match="not in index"):
            df.loc[rows]

    def test_dups_fancy_indexing_only_missing_label(self, using_infer_string):
        # List containing only missing label
        dfnu = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)), index=list("AABCD")
        )
        if using_infer_string:
            with pytest.raises(
                KeyError,
                match=re.escape(
                    "\"None of [Index(['E'], dtype='str')] are in the [index]\""
                ),
            ):
                dfnu.loc[["E"]]
        else:
            with pytest.raises(
                KeyError,
                match=re.escape(
                    "\"None of [Index(['E'], dtype='object')] are in the [index]\""
                ),
            ):
                dfnu.loc[["E"]]

    @pytest.mark.parametrize("vals", [[0, 1, 2], list("abc")])
    def test_dups_fancy_indexing_missing_label(self, vals):
        # GH 4619; duplicate indexer with missing label
        df = DataFrame({"A": vals})
        with pytest.raises(KeyError, match="not in index"):
            df.loc[[0, 8, 0]]

    def test_dups_fancy_indexing_non_unique(self):
        # non unique with non unique selector
        df = DataFrame({"test": [5, 7, 9, 11]}, index=["A", "A", "B", "C"])
        with pytest.raises(KeyError, match="not in index"):
            df.loc[["A", "A", "E"]]

    def test_dups_fancy_indexing2(self):
        # GH 5835
        # dups on index and missing values
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 5)),
            columns=["A", "B", "B", "B", "A"],
        )

        with pytest.raises(KeyError, match="not in index"):
            df.loc[:, ["A", "B", "C"]]

    def test_dups_fancy_indexing3(self):
        # GH 6504, multi-axis indexing
        df = DataFrame(
            np.random.default_rng(2).standard_normal((9, 2)),
            index=[1, 1, 1, 2, 2, 2, 3, 3, 3],
            columns=["a", "b"],
        )

        expected = df.iloc[0:6]
        result = df.loc[[1, 2]]
        tm.assert_frame_equal(result, expected)

        expected = df
        result = df.loc[:, ["a", "b"]]
        tm.assert_frame_equal(result, expected)

        expected = df.iloc[0:6, :]
        result = df.loc[[1, 2], ["a", "b"]]
        tm.assert_frame_equal(result, expected)

    def test_duplicate_int_indexing(self, indexer_sl):
        # GH 17347
        ser = Series(range(3), index=[1, 1, 3])
        expected = Series(range(2), index=[1, 1])
        result = indexer_sl(ser)[[1]]
        tm.assert_series_equal(result, expected)

    def test_indexing_mixed_frame_bug(self):
        # GH3492
        df = DataFrame(
            {"a": {1: "aaa", 2: "bbb", 3: "ccc"}, "b": {1: 111, 2: 222, 3: 333}}
        )

        # this works, new column is created correctly
        df["test"] = df["a"].apply(lambda x: "_" if x == "aaa" else x)

        # this does not work, ie column test is not changed
        idx = df["test"] == "_"
        temp = df.loc[idx, "a"].apply(lambda x: "-----" if x == "aaa" else x)
        df.loc[idx, "test"] = temp
        assert df.iloc[0, 2] == "-----"

    def test_multitype_list_index_access(self):
        # GH 10610
        df = DataFrame(
            np.random.default_rng(2).random((10, 5)), columns=["a"] + [20, 21, 22, 23]
        )

        with pytest.raises(KeyError, match=re.escape("'[26, -8] not in index'")):
            df[[22, 26, -8]]
        assert df[21].shape[0] == df.shape[0]

    def test_set_index_nan(self):
        # GH 3586
        df = DataFrame(
            {
                "PRuid": {
                    17: "nonQC",
                    18: "nonQC",
                    19: "nonQC",
                    20: "10",
                    21: "11",
                    22: "12",
                    23: "13",
                    24: "24",
                    25: "35",
                    26: "46",
                    27: "47",
                    28: "48",
                    29: "59",
                    30: "10",
                },
                "QC": {
                    17: 0.0,
                    18: 0.0,
                    19: 0.0,
                    20: np.nan,
                    21: np.nan,
                    22: np.nan,
                    23: np.nan,
                    24: 1.0,
                    25: np.nan,
                    26: np.nan,
                    27: np.nan,
                    28: np.nan,
                    29: np.nan,
                    30: np.nan,
                },
                "data": {
                    17: 7.9544899999999998,
                    18: 8.0142609999999994,
                    19: 7.8591520000000008,
                    20: 0.86140349999999999,
                    21: 0.87853110000000001,
                    22: 0.8427041999999999,
                    23: 0.78587700000000005,
                    24: 0.73062459999999996,
                    25: 0.81668560000000001,
                    26: 0.81927080000000008,
                    27: 0.80705009999999999,
                    28: 0.81440240000000008,
                    29: 0.80140849999999997,
                    30: 0.81307740000000006,
                },
                "year": {
                    17: 2006,
                    18: 2007,
                    19: 2008,
                    20: 1985,
                    21: 1985,
                    22: 1985,
                    23: 1985,
                    24: 1985,
                    25: 1985,
                    26: 1985,
                    27: 1985,
                    28: 1985,
                    29: 1985,
                    30: 1986,
                },
            }
        ).reset_index()

        result = (
            df.set_index(["year", "PRuid", "QC"])
            .reset_index()
            .reindex(columns=df.columns)
        )
        tm.assert_frame_equal(result, df)

    def test_multi_assign(self):
        # GH 3626, an assignment of a sub-df to a df
        # set float64 to avoid upcast when setting nan
        df = DataFrame(
            {
                "FC": ["a", "b", "a", "b", "a", "b"],
                "PF": [0, 0, 0, 0, 1, 1],
                "col1": list(range(6)),
                "col2": list(range(6, 12)),
            }
        ).astype({"col2": "float64"})
        df.iloc[1, 0] = np.nan
        df2 = df.copy()

        mask = ~df2.FC.isna()
        cols = ["col1", "col2"]

        dft = df2 * 2
        dft.iloc[3, 3] = np.nan

        expected = DataFrame(
            {
                "FC": ["a", np.nan, "a", "b", "a", "b"],
                "PF": [0, 0, 0, 0, 1, 1],
                "col1": Series([0, 1, 4, 6, 8, 10]),
                "col2": [12, 7, 16, np.nan, 20, 22],
            }
        )

        # frame on rhs
        df2.loc[mask, cols] = dft.loc[mask, cols]
        tm.assert_frame_equal(df2, expected)

        # with an ndarray on rhs
        # coerces to float64 because values has float64 dtype
        # GH 14001
        expected = DataFrame(
            {
                "FC": ["a", np.nan, "a", "b", "a", "b"],
                "PF": [0, 0, 0, 0, 1, 1],
                "col1": [0, 1, 4, 6, 8, 10],
                "col2": [12, 7, 16, np.nan, 20, 22],
            }
        )
        df2 = df.copy()
        df2.loc[mask, cols] = dft.loc[mask, cols].values
        tm.assert_frame_equal(df2, expected)

    def test_multi_assign_broadcasting_rhs(self):
        # broadcasting on the rhs is required
        df = DataFrame(
            {
                "A": [1, 2, 0, 0, 0],
                "B": [0, 0, 0, 10, 11],
                "C": [0, 0, 0, 10, 11],
                "D": [3, 4, 5, 6, 7],
            }
        )

        expected = df.copy()
        mask = expected["A"] == 0
        for col in ["A", "B"]:
            expected.loc[mask, col] = df["D"]

        df.loc[df["A"] == 0, ["A", "B"]] = df["D"].copy()
        tm.assert_frame_equal(df, expected)

    def test_setitem_list(self):
        # GH 6043
        # iloc with a list
        df = DataFrame(index=[0, 1], columns=[0])
        df.iloc[1, 0] = [1, 2, 3]
        df.iloc[1, 0] = [1, 2]

        result = DataFrame(index=[0, 1], columns=[0])
        result.iloc[1, 0] = [1, 2]

        tm.assert_frame_equal(result, df)

    def test_string_slice(self):
        # GH 14424
        # string indexing against datetimelike with object
        # dtype should properly raises KeyError
        df = DataFrame([1], Index([pd.Timestamp("2011-01-01")], dtype=object))
        assert df.index._is_all_dates
        with pytest.raises(KeyError, match="'2011'"):
            df["2011"]

        with pytest.raises(KeyError, match="'2011'"):
            df.loc["2011", 0]

    def test_string_slice_empty(self):
        # GH 14424

        df = DataFrame()
        assert not df.index._is_all_dates
        with pytest.raises(KeyError, match="'2011'"):
            df["2011"]

        with pytest.raises(KeyError, match="^0$"):
            df.loc["2011", 0]

    def test_astype_assignment(self, using_infer_string):
        # GH4312 (iloc)
        df_orig = DataFrame(
            [["1", "2", "3", ".4", 5, 6.0, "foo"]], columns=list("ABCDEFG")
        )
        df_orig[list("ABCDG")] = df_orig[list("ABCDG")].astype(object)

        df = df_orig.copy()

        # with the enforcement of GH#45333 in 2.0, this setting is attempted inplace,
        #  so object dtype is retained
        df.iloc[:, 0:2] = df.iloc[:, 0:2].astype(np.int64)
        expected = DataFrame(
            [[1, 2, "3", ".4", 5, 6.0, "foo"]], columns=list("ABCDEFG")
        )
        expected[list("CDG")] = expected[list("CDG")].astype(object)
        expected["A"] = expected["A"].astype(object)
        expected["B"] = expected["B"].astype(object)
        tm.assert_frame_equal(df, expected)

        # GH5702 (loc)
        df = df_orig.copy()
        df.loc[:, "A"] = df.loc[:, "A"].astype(np.int64)
        expected = DataFrame(
            [[1, "2", "3", ".4", 5, 6.0, "foo"]], columns=list("ABCDEFG")
        )
        expected[list("ABCDG")] = expected[list("ABCDG")].astype(object)
        tm.assert_frame_equal(df, expected)

        df = df_orig.copy()

        df.loc[:, ["B", "C"]] = df.loc[:, ["B", "C"]].astype(np.int64)
        expected = DataFrame(
            [["1", 2, 3, ".4", 5, 6.0, "foo"]], columns=list("ABCDEFG")
        )
        expected[list("ABCDG")] = expected[list("ABCDG")].astype(object)
        tm.assert_frame_equal(df, expected)

    def test_astype_assignment_full_replacements(self):
        # full replacements / no nans
        df = DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})

        # With the enforcement of GH#45333 in 2.0, this assignment occurs inplace,
        #  so float64 is retained
        df.iloc[:, 0] = df["A"].astype(np.int64)
        expected = DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})
        tm.assert_frame_equal(df, expected)

        df = DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})
        df.loc[:, "A"] = df["A"].astype(np.int64)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("indexer", [tm.getitem, tm.loc])
    def test_index_type_coercion(self, indexer):
        # GH 11836
        # if we have an index type and set it with something that looks
        # to numpy like the same, but is actually, not
        # (e.g. setting with a float or string '0')
        # then we need to coerce to object

        # integer indexes
        for s in [Series(range(5)), Series(range(5), index=range(1, 6))]:
            assert is_integer_dtype(s.index)

            s2 = s.copy()
            indexer(s2)[0.1] = 0
            assert is_float_dtype(s2.index)
            assert indexer(s2)[0.1] == 0

            s2 = s.copy()
            indexer(s2)[0.0] = 0
            exp = s.index
            if 0 not in s:
                exp = Index(s.index.tolist() + [0])
            tm.assert_index_equal(s2.index, exp)

            s2 = s.copy()
            indexer(s2)["0"] = 0
            assert is_object_dtype(s2.index)

        for s in [Series(range(5), index=np.arange(5.0))]:
            assert is_float_dtype(s.index)

            s2 = s.copy()
            indexer(s2)[0.1] = 0
            assert is_float_dtype(s2.index)
            assert indexer(s2)[0.1] == 0

            s2 = s.copy()
            indexer(s2)[0.0] = 0
            tm.assert_index_equal(s2.index, s.index)

            s2 = s.copy()
            indexer(s2)["0"] = 0
            assert is_object_dtype(s2.index)


class TestMisc:
    def test_float_index_to_mixed(self):
        df = DataFrame(
            {
                0.0: np.random.default_rng(2).random(10),
                1.0: np.random.default_rng(2).random(10),
            }
        )
        df["a"] = 10

        expected = DataFrame({0.0: df[0.0], 1.0: df[1.0], "a": [10] * 10})
        tm.assert_frame_equal(expected, df)

    def test_float_index_non_scalar_assignment(self):
        df = DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]}, index=[1.0, 2.0, 3.0])
        df.loc[df.index[:2]] = 1
        expected = DataFrame({"a": [1, 1, 3], "b": [1, 1, 5]}, index=df.index)
        tm.assert_frame_equal(expected, df)

    def test_loc_setitem_fullindex_views(self):
        df = DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]}, index=[1.0, 2.0, 3.0])
        df2 = df.copy()
        df.loc[df.index] = df.loc[df.index]
        tm.assert_frame_equal(df, df2)

    def test_rhs_alignment(self, using_infer_string):
        # GH8258, tests that both rows & columns are aligned to what is
        # assigned to. covers both uniform data-type & multi-type cases
        def run_tests(df, rhs, right_loc, right_iloc):
            # label, index, slice
            lbl_one, idx_one, slice_one = list("bcd"), [1, 2, 3], slice(1, 4)
            lbl_two, idx_two, slice_two = ["joe", "jolie"], [1, 2], slice(1, 3)

            left = df.copy()
            left.loc[lbl_one, lbl_two] = rhs
            tm.assert_frame_equal(left, right_loc)

            left = df.copy()
            left.iloc[idx_one, idx_two] = rhs
            tm.assert_frame_equal(left, right_iloc)

            left = df.copy()
            left.iloc[slice_one, slice_two] = rhs
            tm.assert_frame_equal(left, right_iloc)

        xs = np.arange(20).reshape(5, 4)
        cols = ["jim", "joe", "jolie", "joline"]
        df = DataFrame(xs, columns=cols, index=list("abcde"), dtype="int64")

        # right hand side; permute the indices and multiplpy by -2
        rhs = -2 * df.iloc[3:0:-1, 2:0:-1]

        # expected `right` result; just multiply by -2
        right_iloc = df.copy()
        right_iloc["joe"] = [1, 14, 10, 6, 17]
        right_iloc["jolie"] = [2, 13, 9, 5, 18]
        right_iloc.iloc[1:4, 1:3] *= -2
        right_loc = df.copy()
        right_loc.iloc[1:4, 1:3] *= -2

        # run tests with uniform dtypes
        run_tests(df, rhs, right_loc, right_iloc)

        # make frames multi-type & re-run tests
        for frame in [df, rhs, right_loc, right_iloc]:
            frame["joe"] = frame["joe"].astype("float64")
            frame["jolie"] = frame["jolie"].map(lambda x: f"@{x}")
        right_iloc["joe"] = [1.0, "@-28", "@-20", "@-12", 17.0]
        right_iloc["jolie"] = ["@2", -26.0, -18.0, -10.0, "@18"]
        if using_infer_string:
            with pytest.raises(TypeError, match="Invalid value"):
                with tm.assert_produces_warning(
                    FutureWarning, match="incompatible dtype"
                ):
                    run_tests(df, rhs, right_loc, right_iloc)
        else:
            with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
                run_tests(df, rhs, right_loc, right_iloc)

    @pytest.mark.parametrize(
        "idx", [_mklbl("A", 20), np.arange(20) + 100, np.linspace(100, 150, 20)]
    )
    def test_str_label_slicing_with_negative_step(self, idx):
        SLC = pd.IndexSlice

        idx = Index(idx)
        ser = Series(np.arange(20), index=idx)
        tm.assert_indexing_slices_equivalent(ser, SLC[idx[9] :: -1], SLC[9::-1])
        tm.assert_indexing_slices_equivalent(ser, SLC[: idx[9] : -1], SLC[:8:-1])
        tm.assert_indexing_slices_equivalent(
            ser, SLC[idx[13] : idx[9] : -1], SLC[13:8:-1]
        )
        tm.assert_indexing_slices_equivalent(ser, SLC[idx[9] : idx[13] : -1], SLC[:0])

    def test_slice_with_zero_step_raises(self, index, indexer_sl, frame_or_series):
        obj = frame_or_series(np.arange(len(index)), index=index)
        with pytest.raises(ValueError, match="slice step cannot be zero"):
            indexer_sl(obj)[::0]

    def test_loc_setitem_indexing_assignment_dict_already_exists(self):
        index = Index([-5, 0, 5], name="z")
        df = DataFrame({"x": [1, 2, 6], "y": [2, 2, 8]}, index=index)
        expected = df.copy()
        rhs = {"x": 9, "y": 99}
        df.loc[5] = rhs
        expected.loc[5] = [9, 99]
        tm.assert_frame_equal(df, expected)

        # GH#38335 same thing, mixed dtypes
        df = DataFrame({"x": [1, 2, 6], "y": [2.0, 2.0, 8.0]}, index=index)
        df.loc[5] = rhs
        expected = DataFrame({"x": [1, 2, 9], "y": [2.0, 2.0, 99.0]}, index=index)
        tm.assert_frame_equal(df, expected)

    def test_iloc_getitem_indexing_dtypes_on_empty(self):
        # Check that .iloc returns correct dtypes GH9983
        df = DataFrame({"a": [1, 2, 3], "b": ["b", "b2", "b3"]})
        df2 = df.iloc[[], :]

        assert df2.loc[:, "a"].dtype == np.int64
        tm.assert_series_equal(df2.loc[:, "a"], df2.iloc[:, 0])

    @pytest.mark.parametrize("size", [5, 999999, 1000000])
    def test_loc_range_in_series_indexing(self, size):
        # range can cause an indexing error
        # GH 11652
        s = Series(index=range(size), dtype=np.float64)
        s.loc[range(1)] = 42
        tm.assert_series_equal(s.loc[range(1)], Series(42.0, index=[0]))

        s.loc[range(2)] = 43
        tm.assert_series_equal(s.loc[range(2)], Series(43.0, index=[0, 1]))

    def test_partial_boolean_frame_indexing(self):
        # GH 17170
        df = DataFrame(
            np.arange(9.0).reshape(3, 3), index=list("abc"), columns=list("ABC")
        )
        index_df = DataFrame(1, index=list("ab"), columns=list("AB"))
        result = df[index_df.notnull()]
        expected = DataFrame(
            np.array([[0.0, 1.0, np.nan], [3.0, 4.0, np.nan], [np.nan] * 3]),
            index=list("abc"),
            columns=list("ABC"),
        )
        tm.assert_frame_equal(result, expected)

    def test_no_reference_cycle(self):
        df = DataFrame({"a": [0, 1], "b": [2, 3]})
        for name in ("loc", "iloc", "at", "iat"):
            getattr(df, name)
        wr = weakref.ref(df)
        del df
        assert wr() is None

    def test_label_indexing_on_nan(self, nulls_fixture):
        # GH 32431
        df = Series([1, "{1,2}", 1, nulls_fixture])
        vc = df.value_counts(dropna=False)
        result1 = vc.loc[nulls_fixture]
        result2 = vc[nulls_fixture]

        expected = 1
        assert result1 == expected
        assert result2 == expected


class TestDataframeNoneCoercion:
    EXPECTED_SINGLE_ROW_RESULTS = [
        # For numeric series, we should coerce to NaN.
        ([1, 2, 3], [np.nan, 2, 3], FutureWarning),
        ([1.0, 2.0, 3.0], [np.nan, 2.0, 3.0], None),
        # For datetime series, we should coerce to NaT.
        (
            [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
            [NaT, datetime(2000, 1, 2), datetime(2000, 1, 3)],
            None,
        ),
        # For objects, we should preserve the None value.
        (["foo", "bar", "baz"], [None, "bar", "baz"], None),
    ]

    @pytest.mark.parametrize("expected", EXPECTED_SINGLE_ROW_RESULTS)
    def test_coercion_with_loc(self, expected):
        start_data, expected_result, warn = expected

        start_dataframe = DataFrame({"foo": start_data})
        start_dataframe.loc[0, ["foo"]] = None

        expected_dataframe = DataFrame({"foo": expected_result})
        tm.assert_frame_equal(start_dataframe, expected_dataframe)

    @pytest.mark.parametrize("expected", EXPECTED_SINGLE_ROW_RESULTS)
    def test_coercion_with_setitem_and_dataframe(self, expected):
        start_data, expected_result, warn = expected

        start_dataframe = DataFrame({"foo": start_data})
        start_dataframe[start_dataframe["foo"] == start_dataframe["foo"][0]] = None

        expected_dataframe = DataFrame({"foo": expected_result})
        tm.assert_frame_equal(start_dataframe, expected_dataframe)

    @pytest.mark.parametrize("expected", EXPECTED_SINGLE_ROW_RESULTS)
    def test_none_coercion_loc_and_dataframe(self, expected):
        start_data, expected_result, warn = expected

        start_dataframe = DataFrame({"foo": start_data})
        start_dataframe.loc[start_dataframe["foo"] == start_dataframe["foo"][0]] = None

        expected_dataframe = DataFrame({"foo": expected_result})
        tm.assert_frame_equal(start_dataframe, expected_dataframe)

    def test_none_coercion_mixed_dtypes(self):
        start_dataframe = DataFrame(
            {
                "a": [1, 2, 3],
                "b": [1.0, 2.0, 3.0],
                "c": [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
                "d": ["a", "b", "c"],
            }
        )
        start_dataframe.iloc[0] = None

        exp = DataFrame(
            {
                "a": [np.nan, 2, 3],
                "b": [np.nan, 2.0, 3.0],
                "c": [NaT, datetime(2000, 1, 2), datetime(2000, 1, 3)],
                "d": [None, "b", "c"],
            }
        )
        tm.assert_frame_equal(start_dataframe, exp)


class TestDatetimelikeCoercion:
    def test_setitem_dt64_string_scalar(self, tz_naive_fixture, indexer_sli):
        # dispatching _can_hold_element to underlying DatetimeArray
        tz = tz_naive_fixture

        dti = date_range("2016-01-01", periods=3, tz=tz)
        ser = Series(dti.copy(deep=True))

        values = ser._values

        newval = "2018-01-01"
        values._validate_setitem_value(newval)

        indexer_sli(ser)[0] = newval

        if tz is None:
            # TODO(EA2D): we can make this no-copy in tz-naive case too
            assert ser.dtype == dti.dtype
            assert ser._values._ndarray is values._ndarray
        else:
            assert ser._values is values

    @pytest.mark.parametrize("box", [list, np.array, pd.array, pd.Categorical, Index])
    @pytest.mark.parametrize(
        "key", [[0, 1], slice(0, 2), np.array([True, True, False])]
    )
    def test_setitem_dt64_string_values(self, tz_naive_fixture, indexer_sli, key, box):
        # dispatching _can_hold_element to underling DatetimeArray
        tz = tz_naive_fixture

        if isinstance(key, slice) and indexer_sli is tm.loc:
            key = slice(0, 1)

        dti = date_range("2016-01-01", periods=3, tz=tz)
        ser = Series(dti.copy(deep=True))

        values = ser._values

        newvals = box(["2019-01-01", "2010-01-02"])
        values._validate_setitem_value(newvals)

        indexer_sli(ser)[key] = newvals

        if tz is None:
            # TODO(EA2D): we can make this no-copy in tz-naive case too
            assert ser.dtype == dti.dtype
            assert ser._values._ndarray is values._ndarray
        else:
            assert ser._values is values

    @pytest.mark.parametrize("scalar", ["3 Days", offsets.Hour(4)])
    def test_setitem_td64_scalar(self, indexer_sli, scalar):
        # dispatching _can_hold_element to underling TimedeltaArray
        tdi = timedelta_range("1 Day", periods=3)
        ser = Series(tdi.copy(deep=True))

        values = ser._values
        values._validate_setitem_value(scalar)

        indexer_sli(ser)[0] = scalar
        assert ser._values._ndarray is values._ndarray

    @pytest.mark.parametrize("box", [list, np.array, pd.array, pd.Categorical, Index])
    @pytest.mark.parametrize(
        "key", [[0, 1], slice(0, 2), np.array([True, True, False])]
    )
    def test_setitem_td64_string_values(self, indexer_sli, key, box):
        # dispatching _can_hold_element to underling TimedeltaArray
        if isinstance(key, slice) and indexer_sli is tm.loc:
            key = slice(0, 1)

        tdi = timedelta_range("1 Day", periods=3)
        ser = Series(tdi.copy(deep=True))

        values = ser._values

        newvals = box(["10 Days", "44 hours"])
        values._validate_setitem_value(newvals)

        indexer_sli(ser)[key] = newvals
        assert ser._values._ndarray is values._ndarray


def test_extension_array_cross_section():
    # A cross-section of a homogeneous EA should be an EA
    df = DataFrame(
        {
            "A": pd.array([1, 2], dtype="Int64"),
            "B": pd.array([3, 4], dtype="Int64"),
        },
        index=["a", "b"],
    )
    expected = Series(pd.array([1, 3], dtype="Int64"), index=["A", "B"], name="a")
    result = df.loc["a"]
    tm.assert_series_equal(result, expected)

    result = df.iloc[0]
    tm.assert_series_equal(result, expected)


def test_extension_array_cross_section_converts():
    # all numeric columns -> numeric series
    df = DataFrame(
        {
            "A": pd.array([1, 2], dtype="Int64"),
            "B": np.array([1, 2], dtype="int64"),
        },
        index=["a", "b"],
    )
    result = df.loc["a"]
    expected = Series([1, 1], dtype="Int64", index=["A", "B"], name="a")
    tm.assert_series_equal(result, expected)

    result = df.iloc[0]
    tm.assert_series_equal(result, expected)

    # mixed columns -> object series
    df = DataFrame(
        {"A": pd.array([1, 2], dtype="Int64"), "B": np.array(["a", "b"])},
        index=["a", "b"],
    )
    result = df.loc["a"]
    expected = Series([1, "a"], dtype=object, index=["A", "B"], name="a")
    tm.assert_series_equal(result, expected)

    result = df.iloc[0]
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "ser, keys",
    [(Series([10]), (0, 0)), (Series([1, 2, 3], index=list("abc")), (0, 1))],
)
def test_ser_tup_indexer_exceeds_dimensions(ser, keys, indexer_li):
    # GH#13831
    exp_err, exp_msg = IndexingError, "Too many indexers"
    with pytest.raises(exp_err, match=exp_msg):
        indexer_li(ser)[keys]

    if indexer_li == tm.iloc:
        # For iloc.__setitem__ we let numpy handle the error reporting.
        exp_err, exp_msg = IndexError, "too many indices for array"

    with pytest.raises(exp_err, match=exp_msg):
        indexer_li(ser)[keys] = 0


def test_ser_list_indexer_exceeds_dimensions(indexer_li):
    # GH#13831
    # Make sure an exception is raised when a tuple exceeds the dimension of the series,
    # but not list when a list is used.
    ser = Series([10])
    res = indexer_li(ser)[[0, 0]]
    exp = Series([10, 10], index=Index([0, 0]))
    tm.assert_series_equal(res, exp)


@pytest.mark.parametrize(
    "value", [(0, 1), [0, 1], np.array([0, 1]), array.array("b", [0, 1])]
)
def test_scalar_setitem_with_nested_value(value):
    # For numeric data, we try to unpack and thus raise for mismatching length
    df = DataFrame({"A": [1, 2, 3]})
    msg = "|".join(
        [
            "Must have equal len keys and value",
            "setting an array element with a sequence",
        ]
    )
    with pytest.raises(ValueError, match=msg):
        df.loc[0, "B"] = value

    # TODO For object dtype this happens as well, but should we rather preserve
    # the nested data and set as such?
    df = DataFrame({"A": [1, 2, 3], "B": np.array([1, "a", "b"], dtype=object)})
    with pytest.raises(ValueError, match="Must have equal len keys and value"):
        df.loc[0, "B"] = value
    # if isinstance(value, np.ndarray):
    #     assert (df.loc[0, "B"] == value).all()
    # else:
    #     assert df.loc[0, "B"] == value


@pytest.mark.parametrize(
    "value", [(0, 1), [0, 1], np.array([0, 1]), array.array("b", [0, 1])]
)
def test_scalar_setitem_series_with_nested_value(value, indexer_sli):
    # For numeric data, we try to unpack and thus raise for mismatching length
    ser = Series([1, 2, 3])
    with pytest.raises(ValueError, match="setting an array element with a sequence"):
        indexer_sli(ser)[0] = value

    # but for object dtype we preserve the nested data and set as such
    ser = Series([1, "a", "b"], dtype=object)
    indexer_sli(ser)[0] = value
    if isinstance(value, np.ndarray):
        assert (ser.loc[0] == value).all()
    else:
        assert ser.loc[0] == value


@pytest.mark.parametrize(
    "value", [(0.0,), [0.0], np.array([0.0]), array.array("d", [0.0])]
)
def test_scalar_setitem_with_nested_value_length1(value):
    # https://github.com/pandas-dev/pandas/issues/46268

    # For numeric data, assigning length-1 array to scalar position gets unpacked
    df = DataFrame({"A": [1, 2, 3]})
    df.loc[0, "B"] = value
    expected = DataFrame({"A": [1, 2, 3], "B": [0.0, np.nan, np.nan]})
    tm.assert_frame_equal(df, expected)

    # but for object dtype we preserve the nested data
    df = DataFrame({"A": [1, 2, 3], "B": np.array([1, "a", "b"], dtype=object)})
    df.loc[0, "B"] = value
    if isinstance(value, np.ndarray):
        assert (df.loc[0, "B"] == value).all()
    else:
        assert df.loc[0, "B"] == value


@pytest.mark.parametrize(
    "value", [(0.0,), [0.0], np.array([0.0]), array.array("d", [0.0])]
)
def test_scalar_setitem_series_with_nested_value_length1(value, indexer_sli):
    # For numeric data, assigning length-1 array to scalar position gets unpacked
    # TODO this only happens in case of ndarray, should we make this consistent
    # for all list-likes? (as happens for DataFrame.(i)loc, see test above)
    ser = Series([1.0, 2.0, 3.0])
    if isinstance(value, np.ndarray):
        indexer_sli(ser)[0] = value
        expected = Series([0.0, 2.0, 3.0])
        tm.assert_series_equal(ser, expected)
    else:
        with pytest.raises(
            ValueError, match="setting an array element with a sequence"
        ):
            indexer_sli(ser)[0] = value

    # but for object dtype we preserve the nested data
    ser = Series([1, "a", "b"], dtype=object)
    indexer_sli(ser)[0] = value
    if isinstance(value, np.ndarray):
        assert (ser.loc[0] == value).all()
    else:
        assert ser.loc[0] == value


def test_object_dtype_series_set_series_element():
    # GH 48933
    s1 = Series(dtype="O", index=["a", "b"])

    s1["a"] = Series()
    s1.loc["b"] = Series()

    tm.assert_series_equal(s1.loc["a"], Series())
    tm.assert_series_equal(s1.loc["b"], Series())

    s2 = Series(dtype="O", index=["a", "b"])

    s2.iloc[1] = Series()
    tm.assert_series_equal(s2.iloc[1], Series())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_fillna.py ===
from datetime import (
    datetime,
    timedelta,
    timezone,
)

import numpy as np
import pytest
import pytz

from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    NaT,
    Period,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    isna,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.arrays import period_array


@pytest.mark.filterwarnings(
    "ignore:(Series|DataFrame).fillna with 'method' is deprecated:FutureWarning"
)
class TestSeriesFillNA:
    def test_fillna_nat(self):
        series = Series([0, 1, 2, NaT._value], dtype="M8[ns]")

        filled = series.fillna(method="pad")
        filled2 = series.fillna(value=series.values[2])

        expected = series.copy()
        expected.iloc[3] = expected.iloc[2]

        tm.assert_series_equal(filled, expected)
        tm.assert_series_equal(filled2, expected)

        df = DataFrame({"A": series})
        filled = df.fillna(method="pad")
        filled2 = df.fillna(value=series.values[2])
        expected = DataFrame({"A": expected})
        tm.assert_frame_equal(filled, expected)
        tm.assert_frame_equal(filled2, expected)

        series = Series([NaT._value, 0, 1, 2], dtype="M8[ns]")

        filled = series.fillna(method="bfill")
        filled2 = series.fillna(value=series[1])

        expected = series.copy()
        expected[0] = expected[1]

        tm.assert_series_equal(filled, expected)
        tm.assert_series_equal(filled2, expected)

        df = DataFrame({"A": series})
        filled = df.fillna(method="bfill")
        filled2 = df.fillna(value=series[1])
        expected = DataFrame({"A": expected})
        tm.assert_frame_equal(filled, expected)
        tm.assert_frame_equal(filled2, expected)

    def test_fillna_value_or_method(self, datetime_series):
        msg = "Cannot specify both 'value' and 'method'"
        with pytest.raises(ValueError, match=msg):
            datetime_series.fillna(value=0, method="ffill")

    def test_fillna(self):
        ts = Series(
            [0.0, 1.0, 2.0, 3.0, 4.0], index=date_range("2020-01-01", periods=5)
        )

        tm.assert_series_equal(ts, ts.fillna(method="ffill"))

        ts.iloc[2] = np.nan

        exp = Series([0.0, 1.0, 1.0, 3.0, 4.0], index=ts.index)
        tm.assert_series_equal(ts.fillna(method="ffill"), exp)

        exp = Series([0.0, 1.0, 3.0, 3.0, 4.0], index=ts.index)
        tm.assert_series_equal(ts.fillna(method="backfill"), exp)

        exp = Series([0.0, 1.0, 5.0, 3.0, 4.0], index=ts.index)
        tm.assert_series_equal(ts.fillna(value=5), exp)

        msg = "Must specify a fill 'value' or 'method'"
        with pytest.raises(ValueError, match=msg):
            ts.fillna()

    def test_fillna_nonscalar(self):
        # GH#5703
        s1 = Series([np.nan])
        s2 = Series([1])
        result = s1.fillna(s2)
        expected = Series([1.0])
        tm.assert_series_equal(result, expected)
        result = s1.fillna({})
        tm.assert_series_equal(result, s1)
        result = s1.fillna(Series((), dtype=object))
        tm.assert_series_equal(result, s1)
        result = s2.fillna(s1)
        tm.assert_series_equal(result, s2)
        result = s1.fillna({0: 1})
        tm.assert_series_equal(result, expected)
        result = s1.fillna({1: 1})
        tm.assert_series_equal(result, Series([np.nan]))
        result = s1.fillna({0: 1, 1: 1})
        tm.assert_series_equal(result, expected)
        result = s1.fillna(Series({0: 1, 1: 1}))
        tm.assert_series_equal(result, expected)
        result = s1.fillna(Series({0: 1, 1: 1}, index=[4, 5]))
        tm.assert_series_equal(result, s1)

    def test_fillna_aligns(self):
        s1 = Series([0, 1, 2], list("abc"))
        s2 = Series([0, np.nan, 2], list("bac"))
        result = s2.fillna(s1)
        expected = Series([0, 0, 2.0], list("bac"))
        tm.assert_series_equal(result, expected)

    def test_fillna_limit(self):
        ser = Series(np.nan, index=[0, 1, 2])
        result = ser.fillna(999, limit=1)
        expected = Series([999, np.nan, np.nan], index=[0, 1, 2])
        tm.assert_series_equal(result, expected)

        result = ser.fillna(999, limit=2)
        expected = Series([999, 999, np.nan], index=[0, 1, 2])
        tm.assert_series_equal(result, expected)

    def test_fillna_dont_cast_strings(self):
        # GH#9043
        # make sure a string representation of int/float values can be filled
        # correctly without raising errors or being converted
        vals = ["0", "1.5", "-0.3"]
        for val in vals:
            ser = Series([0, 1, np.nan, np.nan, 4], dtype="float64")
            result = ser.fillna(val)
            expected = Series([0, 1, val, val, 4], dtype="object")
            tm.assert_series_equal(result, expected)

    def test_fillna_consistency(self):
        # GH#16402
        # fillna with a tz aware to a tz-naive, should result in object

        ser = Series([Timestamp("20130101"), NaT])

        result = ser.fillna(Timestamp("20130101", tz="US/Eastern"))
        expected = Series(
            [Timestamp("20130101"), Timestamp("2013-01-01", tz="US/Eastern")],
            dtype="object",
        )
        tm.assert_series_equal(result, expected)

        result = ser.where([True, False], Timestamp("20130101", tz="US/Eastern"))
        tm.assert_series_equal(result, expected)

        result = ser.where([True, False], Timestamp("20130101", tz="US/Eastern"))
        tm.assert_series_equal(result, expected)

        # with a non-datetime
        result = ser.fillna("foo")
        expected = Series([Timestamp("20130101"), "foo"])
        tm.assert_series_equal(result, expected)

        # assignment
        ser2 = ser.copy()
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            ser2[1] = "foo"
        tm.assert_series_equal(ser2, expected)

    def test_fillna_downcast(self):
        # GH#15277
        # infer int64 from float64
        ser = Series([1.0, np.nan])
        msg = "The 'downcast' keyword in fillna is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = ser.fillna(0, downcast="infer")
        expected = Series([1, 0])
        tm.assert_series_equal(result, expected)

        # infer int64 from float64 when fillna value is a dict
        ser = Series([1.0, np.nan])
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = ser.fillna({1: 0}, downcast="infer")
        expected = Series([1, 0])
        tm.assert_series_equal(result, expected)

    def test_fillna_downcast_infer_objects_to_numeric(self):
        # GH#44241 if we have object-dtype, 'downcast="infer"' should
        #  _actually_ infer

        arr = np.arange(5).astype(object)
        arr[3] = np.nan

        ser = Series(arr)

        msg = "The 'downcast' keyword in fillna is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = ser.fillna(3, downcast="infer")
        expected = Series(np.arange(5), dtype=np.int64)
        tm.assert_series_equal(res, expected)

        msg = "The 'downcast' keyword in ffill is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = ser.ffill(downcast="infer")
        expected = Series([0, 1, 2, 2, 4], dtype=np.int64)
        tm.assert_series_equal(res, expected)

        msg = "The 'downcast' keyword in bfill is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = ser.bfill(downcast="infer")
        expected = Series([0, 1, 2, 4, 4], dtype=np.int64)
        tm.assert_series_equal(res, expected)

        # with a non-round float present, we will downcast to float64
        ser[2] = 2.5

        expected = Series([0, 1, 2.5, 3, 4], dtype=np.float64)
        msg = "The 'downcast' keyword in fillna is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = ser.fillna(3, downcast="infer")
        tm.assert_series_equal(res, expected)

        msg = "The 'downcast' keyword in ffill is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = ser.ffill(downcast="infer")
        expected = Series([0, 1, 2.5, 2.5, 4], dtype=np.float64)
        tm.assert_series_equal(res, expected)

        msg = "The 'downcast' keyword in bfill is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = ser.bfill(downcast="infer")
        expected = Series([0, 1, 2.5, 4, 4], dtype=np.float64)
        tm.assert_series_equal(res, expected)

    def test_timedelta_fillna(self, frame_or_series, unit):
        # GH#3371
        ser = Series(
            [
                Timestamp("20130101"),
                Timestamp("20130101"),
                Timestamp("20130102"),
                Timestamp("20130103 9:01:01"),
            ],
            dtype=f"M8[{unit}]",
        )
        td = ser.diff()
        obj = frame_or_series(td).copy()

        # reg fillna
        result = obj.fillna(Timedelta(seconds=0))
        expected = Series(
            [
                timedelta(0),
                timedelta(0),
                timedelta(1),
                timedelta(days=1, seconds=9 * 3600 + 60 + 1),
            ],
            dtype=f"m8[{unit}]",
        )
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)

        # GH#45746 pre-1.? ints were interpreted as seconds.  then that was
        #  deprecated and changed to raise. In 2.0 it casts to common dtype,
        #  consistent with every other dtype's behavior
        res = obj.fillna(1)
        expected = obj.astype(object).fillna(1)
        tm.assert_equal(res, expected)

        result = obj.fillna(Timedelta(seconds=1))
        expected = Series(
            [
                timedelta(seconds=1),
                timedelta(0),
                timedelta(1),
                timedelta(days=1, seconds=9 * 3600 + 60 + 1),
            ],
            dtype=f"m8[{unit}]",
        )
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)

        result = obj.fillna(timedelta(days=1, seconds=1))
        expected = Series(
            [
                timedelta(days=1, seconds=1),
                timedelta(0),
                timedelta(1),
                timedelta(days=1, seconds=9 * 3600 + 60 + 1),
            ],
            dtype=f"m8[{unit}]",
        )
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)

        result = obj.fillna(np.timedelta64(10**9))
        expected = Series(
            [
                timedelta(seconds=1),
                timedelta(0),
                timedelta(1),
                timedelta(days=1, seconds=9 * 3600 + 60 + 1),
            ],
            dtype=f"m8[{unit}]",
        )
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)

        result = obj.fillna(NaT)
        expected = Series(
            [
                NaT,
                timedelta(0),
                timedelta(1),
                timedelta(days=1, seconds=9 * 3600 + 60 + 1),
            ],
            dtype=f"m8[{unit}]",
        )
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)

        # ffill
        td[2] = np.nan
        obj = frame_or_series(td).copy()
        result = obj.ffill()
        expected = td.fillna(Timedelta(seconds=0))
        expected[0] = np.nan
        expected = frame_or_series(expected)

        tm.assert_equal(result, expected)

        # bfill
        td[2] = np.nan
        obj = frame_or_series(td)
        result = obj.bfill()
        expected = td.fillna(Timedelta(seconds=0))
        expected[2] = timedelta(days=1, seconds=9 * 3600 + 60 + 1)
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)

    def test_datetime64_fillna(self):
        ser = Series(
            [
                Timestamp("20130101"),
                Timestamp("20130101"),
                Timestamp("20130102"),
                Timestamp("20130103 9:01:01"),
            ]
        )
        ser[2] = np.nan

        # ffill
        result = ser.ffill()
        expected = Series(
            [
                Timestamp("20130101"),
                Timestamp("20130101"),
                Timestamp("20130101"),
                Timestamp("20130103 9:01:01"),
            ]
        )
        tm.assert_series_equal(result, expected)

        # bfill
        result = ser.bfill()
        expected = Series(
            [
                Timestamp("20130101"),
                Timestamp("20130101"),
                Timestamp("20130103 9:01:01"),
                Timestamp("20130103 9:01:01"),
            ]
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "scalar",
        [
            False,
            pytest.param(
                True,
                marks=pytest.mark.xfail(
                    reason="GH#56410 scalar case not yet addressed"
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_datetime64_fillna_mismatched_reso_no_rounding(self, tz, scalar):
        # GH#56410
        dti = date_range("2016-01-01", periods=3, unit="s", tz=tz)
        item = Timestamp("2016-02-03 04:05:06.789", tz=tz)
        vec = date_range(item, periods=3, unit="ms")

        exp_dtype = "M8[ms]" if tz is None else "M8[ms, UTC]"
        expected = Series([item, dti[1], dti[2]], dtype=exp_dtype)

        ser = Series(dti)
        ser[0] = NaT
        ser2 = ser.copy()

        res = ser.fillna(item)
        res2 = ser2.fillna(Series(vec))

        if scalar:
            tm.assert_series_equal(res, expected)
        else:
            tm.assert_series_equal(res2, expected)

    @pytest.mark.parametrize(
        "scalar",
        [
            False,
            pytest.param(
                True,
                marks=pytest.mark.xfail(
                    reason="GH#56410 scalar case not yet addressed"
                ),
            ),
        ],
    )
    def test_timedelta64_fillna_mismatched_reso_no_rounding(self, scalar):
        # GH#56410
        tdi = date_range("2016-01-01", periods=3, unit="s") - Timestamp("1970-01-01")
        item = Timestamp("2016-02-03 04:05:06.789") - Timestamp("1970-01-01")
        vec = timedelta_range(item, periods=3, unit="ms")

        expected = Series([item, tdi[1], tdi[2]], dtype="m8[ms]")

        ser = Series(tdi)
        ser[0] = NaT
        ser2 = ser.copy()

        res = ser.fillna(item)
        res2 = ser2.fillna(Series(vec))

        if scalar:
            tm.assert_series_equal(res, expected)
        else:
            tm.assert_series_equal(res2, expected)

    def test_datetime64_fillna_backfill(self):
        # GH#6587
        # make sure that we are treating as integer when filling
        ser = Series([NaT, NaT, "2013-08-05 15:30:00.000001"], dtype="M8[ns]")

        expected = Series(
            [
                "2013-08-05 15:30:00.000001",
                "2013-08-05 15:30:00.000001",
                "2013-08-05 15:30:00.000001",
            ],
            dtype="M8[ns]",
        )
        result = ser.fillna(method="backfill")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("tz", ["US/Eastern", "Asia/Tokyo"])
    def test_datetime64_tz_fillna(self, tz, unit):
        # DatetimeLikeBlock
        ser = Series(
            [
                Timestamp("2011-01-01 10:00"),
                NaT,
                Timestamp("2011-01-03 10:00"),
                NaT,
            ],
            dtype=f"M8[{unit}]",
        )
        null_loc = Series([False, True, False, True])

        result = ser.fillna(Timestamp("2011-01-02 10:00"))
        expected = Series(
            [
                Timestamp("2011-01-01 10:00"),
                Timestamp("2011-01-02 10:00"),
                Timestamp("2011-01-03 10:00"),
                Timestamp("2011-01-02 10:00"),
            ],
            dtype=f"M8[{unit}]",
        )
        tm.assert_series_equal(expected, result)
        # check s is not changed
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna(Timestamp("2011-01-02 10:00", tz=tz))
        expected = Series(
            [
                Timestamp("2011-01-01 10:00"),
                Timestamp("2011-01-02 10:00", tz=tz),
                Timestamp("2011-01-03 10:00"),
                Timestamp("2011-01-02 10:00", tz=tz),
            ]
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna("AAA")
        expected = Series(
            [
                Timestamp("2011-01-01 10:00"),
                "AAA",
                Timestamp("2011-01-03 10:00"),
                "AAA",
            ],
            dtype=object,
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna(
            {
                1: Timestamp("2011-01-02 10:00", tz=tz),
                3: Timestamp("2011-01-04 10:00"),
            }
        )
        expected = Series(
            [
                Timestamp("2011-01-01 10:00"),
                Timestamp("2011-01-02 10:00", tz=tz),
                Timestamp("2011-01-03 10:00"),
                Timestamp("2011-01-04 10:00"),
            ]
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna(
            {1: Timestamp("2011-01-02 10:00"), 3: Timestamp("2011-01-04 10:00")}
        )
        expected = Series(
            [
                Timestamp("2011-01-01 10:00"),
                Timestamp("2011-01-02 10:00"),
                Timestamp("2011-01-03 10:00"),
                Timestamp("2011-01-04 10:00"),
            ],
            dtype=f"M8[{unit}]",
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        # DatetimeTZBlock
        idx = DatetimeIndex(
            ["2011-01-01 10:00", NaT, "2011-01-03 10:00", NaT], tz=tz
        ).as_unit(unit)
        ser = Series(idx)
        assert ser.dtype == f"datetime64[{unit}, {tz}]"
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna(Timestamp("2011-01-02 10:00"))
        expected = Series(
            [
                Timestamp("2011-01-01 10:00", tz=tz),
                Timestamp("2011-01-02 10:00"),
                Timestamp("2011-01-03 10:00", tz=tz),
                Timestamp("2011-01-02 10:00"),
            ]
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna(Timestamp("2011-01-02 10:00", tz=tz))
        idx = DatetimeIndex(
            [
                "2011-01-01 10:00",
                "2011-01-02 10:00",
                "2011-01-03 10:00",
                "2011-01-02 10:00",
            ],
            tz=tz,
        ).as_unit(unit)
        expected = Series(idx)
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna(Timestamp("2011-01-02 10:00", tz=tz).to_pydatetime())
        idx = DatetimeIndex(
            [
                "2011-01-01 10:00",
                "2011-01-02 10:00",
                "2011-01-03 10:00",
                "2011-01-02 10:00",
            ],
            tz=tz,
        ).as_unit(unit)
        expected = Series(idx)
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna("AAA")
        expected = Series(
            [
                Timestamp("2011-01-01 10:00", tz=tz),
                "AAA",
                Timestamp("2011-01-03 10:00", tz=tz),
                "AAA",
            ],
            dtype=object,
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna(
            {
                1: Timestamp("2011-01-02 10:00", tz=tz),
                3: Timestamp("2011-01-04 10:00"),
            }
        )
        expected = Series(
            [
                Timestamp("2011-01-01 10:00", tz=tz),
                Timestamp("2011-01-02 10:00", tz=tz),
                Timestamp("2011-01-03 10:00", tz=tz),
                Timestamp("2011-01-04 10:00"),
            ]
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna(
            {
                1: Timestamp("2011-01-02 10:00", tz=tz),
                3: Timestamp("2011-01-04 10:00", tz=tz),
            }
        )
        expected = Series(
            [
                Timestamp("2011-01-01 10:00", tz=tz),
                Timestamp("2011-01-02 10:00", tz=tz),
                Timestamp("2011-01-03 10:00", tz=tz),
                Timestamp("2011-01-04 10:00", tz=tz),
            ]
        ).dt.as_unit(unit)
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        # filling with a naive/other zone, coerce to object
        result = ser.fillna(Timestamp("20130101"))
        expected = Series(
            [
                Timestamp("2011-01-01 10:00", tz=tz),
                Timestamp("2013-01-01"),
                Timestamp("2011-01-03 10:00", tz=tz),
                Timestamp("2013-01-01"),
            ]
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        # pre-2.0 fillna with mixed tzs would cast to object, in 2.0
        #  it retains dtype.
        result = ser.fillna(Timestamp("20130101", tz="US/Pacific"))
        expected = Series(
            [
                Timestamp("2011-01-01 10:00", tz=tz),
                Timestamp("2013-01-01", tz="US/Pacific").tz_convert(tz),
                Timestamp("2011-01-03 10:00", tz=tz),
                Timestamp("2013-01-01", tz="US/Pacific").tz_convert(tz),
            ]
        ).dt.as_unit(unit)
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

    def test_fillna_dt64tz_with_method(self):
        # with timezone
        # GH#15855
        ser = Series([Timestamp("2012-11-11 00:00:00+01:00"), NaT])
        exp = Series(
            [
                Timestamp("2012-11-11 00:00:00+01:00"),
                Timestamp("2012-11-11 00:00:00+01:00"),
            ]
        )
        tm.assert_series_equal(ser.fillna(method="pad"), exp)

        ser = Series([NaT, Timestamp("2012-11-11 00:00:00+01:00")])
        exp = Series(
            [
                Timestamp("2012-11-11 00:00:00+01:00"),
                Timestamp("2012-11-11 00:00:00+01:00"),
            ]
        )
        tm.assert_series_equal(ser.fillna(method="bfill"), exp)

    def test_fillna_pytimedelta(self):
        # GH#8209
        ser = Series([np.nan, Timedelta("1 days")], index=["A", "B"])

        result = ser.fillna(timedelta(1))
        expected = Series(Timedelta("1 days"), index=["A", "B"])
        tm.assert_series_equal(result, expected)

    def test_fillna_period(self):
        # GH#13737
        ser = Series([Period("2011-01", freq="M"), Period("NaT", freq="M")])

        res = ser.fillna(Period("2012-01", freq="M"))
        exp = Series([Period("2011-01", freq="M"), Period("2012-01", freq="M")])
        tm.assert_series_equal(res, exp)
        assert res.dtype == "Period[M]"

    def test_fillna_dt64_timestamp(self, frame_or_series):
        ser = Series(
            [
                Timestamp("20130101"),
                Timestamp("20130101"),
                Timestamp("20130102"),
                Timestamp("20130103 9:01:01"),
            ]
        )
        ser[2] = np.nan
        obj = frame_or_series(ser)

        # reg fillna
        result = obj.fillna(Timestamp("20130104"))
        expected = Series(
            [
                Timestamp("20130101"),
                Timestamp("20130101"),
                Timestamp("20130104"),
                Timestamp("20130103 9:01:01"),
            ]
        )
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)

        result = obj.fillna(NaT)
        expected = obj
        tm.assert_equal(result, expected)

    def test_fillna_dt64_non_nao(self):
        # GH#27419
        ser = Series([Timestamp("2010-01-01"), NaT, Timestamp("2000-01-01")])
        val = np.datetime64("1975-04-05", "ms")

        result = ser.fillna(val)
        expected = Series(
            [Timestamp("2010-01-01"), Timestamp("1975-04-05"), Timestamp("2000-01-01")]
        )
        tm.assert_series_equal(result, expected)

    def test_fillna_numeric_inplace(self):
        x = Series([np.nan, 1.0, np.nan, 3.0, np.nan], ["z", "a", "b", "c", "d"])
        y = x.copy()

        return_value = y.fillna(value=0, inplace=True)
        assert return_value is None

        expected = x.fillna(value=0)
        tm.assert_series_equal(y, expected)

    # ---------------------------------------------------------------
    # CategoricalDtype

    @pytest.mark.parametrize(
        "fill_value, expected_output",
        [
            ("a", ["a", "a", "b", "a", "a"]),
            ({1: "a", 3: "b", 4: "b"}, ["a", "a", "b", "b", "b"]),
            ({1: "a"}, ["a", "a", "b", np.nan, np.nan]),
            ({1: "a", 3: "b"}, ["a", "a", "b", "b", np.nan]),
            (Series("a"), ["a", np.nan, "b", np.nan, np.nan]),
            (Series("a", index=[1]), ["a", "a", "b", np.nan, np.nan]),
            (Series({1: "a", 3: "b"}), ["a", "a", "b", "b", np.nan]),
            (Series(["a", "b"], index=[3, 4]), ["a", np.nan, "b", "a", "b"]),
        ],
    )
    def test_fillna_categorical(self, fill_value, expected_output):
        # GH#17033
        # Test fillna for a Categorical series
        data = ["a", np.nan, "b", np.nan, np.nan]
        ser = Series(Categorical(data, categories=["a", "b"]))
        exp = Series(Categorical(expected_output, categories=["a", "b"]))
        result = ser.fillna(fill_value)
        tm.assert_series_equal(result, exp)

    @pytest.mark.parametrize(
        "fill_value, expected_output",
        [
            (Series(["a", "b", "c", "d", "e"]), ["a", "b", "b", "d", "e"]),
            (Series(["b", "d", "a", "d", "a"]), ["a", "d", "b", "d", "a"]),
            (
                Series(
                    Categorical(
                        ["b", "d", "a", "d", "a"], categories=["b", "c", "d", "e", "a"]
                    )
                ),
                ["a", "d", "b", "d", "a"],
            ),
        ],
    )
    def test_fillna_categorical_with_new_categories(self, fill_value, expected_output):
        # GH#26215
        data = ["a", np.nan, "b", np.nan, np.nan]
        ser = Series(Categorical(data, categories=["a", "b", "c", "d", "e"]))
        exp = Series(Categorical(expected_output, categories=["a", "b", "c", "d", "e"]))
        result = ser.fillna(fill_value)
        tm.assert_series_equal(result, exp)

    def test_fillna_categorical_raises(self):
        data = ["a", np.nan, "b", np.nan, np.nan]
        ser = Series(Categorical(data, categories=["a", "b"]))
        cat = ser._values

        msg = "Cannot setitem on a Categorical with a new category"
        with pytest.raises(TypeError, match=msg):
            ser.fillna("d")

        msg2 = "Length of 'value' does not match."
        with pytest.raises(ValueError, match=msg2):
            cat.fillna(Series("d"))

        with pytest.raises(TypeError, match=msg):
            ser.fillna({1: "d", 3: "a"})

        msg = '"value" parameter must be a scalar or dict, but you passed a "list"'
        with pytest.raises(TypeError, match=msg):
            ser.fillna(["a", "b"])

        msg = '"value" parameter must be a scalar or dict, but you passed a "tuple"'
        with pytest.raises(TypeError, match=msg):
            ser.fillna(("a", "b"))

        msg = (
            '"value" parameter must be a scalar, dict '
            'or Series, but you passed a "DataFrame"'
        )
        with pytest.raises(TypeError, match=msg):
            ser.fillna(DataFrame({1: ["a"], 3: ["b"]}))

    @pytest.mark.parametrize("dtype", [float, "float32", "float64"])
    @pytest.mark.parametrize("fill_type", tm.ALL_REAL_NUMPY_DTYPES)
    @pytest.mark.parametrize("scalar", [True, False])
    def test_fillna_float_casting(self, dtype, fill_type, scalar):
        # GH-43424
        ser = Series([np.nan, 1.2], dtype=dtype)
        fill_values = Series([2, 2], dtype=fill_type)
        if scalar:
            fill_values = fill_values.dtype.type(2)

        result = ser.fillna(fill_values)
        expected = Series([2.0, 1.2], dtype=dtype)
        tm.assert_series_equal(result, expected)

        ser = Series([np.nan, 1.2], dtype=dtype)
        mask = ser.isna().to_numpy()
        ser[mask] = fill_values
        tm.assert_series_equal(ser, expected)

        ser = Series([np.nan, 1.2], dtype=dtype)
        ser.mask(mask, fill_values, inplace=True)
        tm.assert_series_equal(ser, expected)

        ser = Series([np.nan, 1.2], dtype=dtype)
        res = ser.where(~mask, fill_values)
        tm.assert_series_equal(res, expected)

    def test_fillna_f32_upcast_with_dict(self):
        # GH-43424
        ser = Series([np.nan, 1.2], dtype=np.float32)
        result = ser.fillna({0: 1})
        expected = Series([1.0, 1.2], dtype=np.float32)
        tm.assert_series_equal(result, expected)

    # ---------------------------------------------------------------
    # Invalid Usages

    def test_fillna_invalid_method(self, datetime_series):
        try:
            datetime_series.fillna(method="ffil")
        except ValueError as inst:
            assert "ffil" in str(inst)

    def test_fillna_listlike_invalid(self):
        ser = Series(np.random.default_rng(2).integers(-100, 100, 50))
        msg = '"value" parameter must be a scalar or dict, but you passed a "list"'
        with pytest.raises(TypeError, match=msg):
            ser.fillna([1, 2])

        msg = '"value" parameter must be a scalar or dict, but you passed a "tuple"'
        with pytest.raises(TypeError, match=msg):
            ser.fillna((1, 2))

    def test_fillna_method_and_limit_invalid(self):
        # related GH#9217, make sure limit is an int and greater than 0
        ser = Series([1, 2, 3, None])
        msg = "|".join(
            [
                r"Cannot specify both 'value' and 'method'\.",
                "Limit must be greater than 0",
                "Limit must be an integer",
            ]
        )
        for limit in [-1, 0, 1.0, 2.0]:
            for method in ["backfill", "bfill", "pad", "ffill", None]:
                with pytest.raises(ValueError, match=msg):
                    ser.fillna(1, limit=limit, method=method)

    def test_fillna_datetime64_with_timezone_tzinfo(self):
        # https://github.com/pandas-dev/pandas/issues/38851
        # different tzinfos representing UTC treated as equal
        ser = Series(date_range("2020", periods=3, tz="UTC"))
        expected = ser.copy()
        ser[1] = NaT
        result = ser.fillna(datetime(2020, 1, 2, tzinfo=timezone.utc))
        tm.assert_series_equal(result, expected)

        # pre-2.0 we cast to object with mixed tzs, in 2.0 we retain dtype
        ts = Timestamp("2000-01-01", tz="US/Pacific")
        ser2 = Series(ser._values.tz_convert("dateutil/US/Pacific"))
        assert ser2.dtype.kind == "M"
        result = ser2.fillna(ts)
        expected = Series(
            [ser2[0], ts.tz_convert(ser2.dtype.tz), ser2[2]],
            dtype=ser2.dtype,
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "input, input_fillna, expected_data, expected_categories",
        [
            (["A", "B", None, "A"], "B", ["A", "B", "B", "A"], ["A", "B"]),
            (["A", "B", np.nan, "A"], "B", ["A", "B", "B", "A"], ["A", "B"]),
        ],
    )
    def test_fillna_categorical_accept_same_type(
        self, input, input_fillna, expected_data, expected_categories
    ):
        # GH32414
        cat = Categorical(input)
        ser = Series(cat).fillna(input_fillna)
        filled = cat.fillna(ser)
        result = cat.fillna(filled)
        expected = Categorical(expected_data, categories=expected_categories)
        tm.assert_categorical_equal(result, expected)


@pytest.mark.filterwarnings(
    "ignore:Series.fillna with 'method' is deprecated:FutureWarning"
)
class TestFillnaPad:
    def test_fillna_bug(self):
        ser = Series([np.nan, 1.0, np.nan, 3.0, np.nan], ["z", "a", "b", "c", "d"])
        filled = ser.fillna(method="ffill")
        expected = Series([np.nan, 1.0, 1.0, 3.0, 3.0], ser.index)
        tm.assert_series_equal(filled, expected)

        filled = ser.fillna(method="bfill")
        expected = Series([1.0, 1.0, 3.0, 3.0, np.nan], ser.index)
        tm.assert_series_equal(filled, expected)

    def test_ffill(self):
        ts = Series(
            [0.0, 1.0, 2.0, 3.0, 4.0], index=date_range("2020-01-01", periods=5)
        )
        ts.iloc[2] = np.nan
        tm.assert_series_equal(ts.ffill(), ts.fillna(method="ffill"))

    def test_ffill_mixed_dtypes_without_missing_data(self):
        # GH#14956
        series = Series([datetime(2015, 1, 1, tzinfo=pytz.utc), 1])
        result = series.ffill()
        tm.assert_series_equal(series, result)

    def test_bfill(self):
        ts = Series(
            [0.0, 1.0, 2.0, 3.0, 4.0], index=date_range("2020-01-01", periods=5)
        )
        ts.iloc[2] = np.nan
        tm.assert_series_equal(ts.bfill(), ts.fillna(method="bfill"))

    def test_pad_nan(self):
        x = Series(
            [np.nan, 1.0, np.nan, 3.0, np.nan], ["z", "a", "b", "c", "d"], dtype=float
        )

        return_value = x.fillna(method="pad", inplace=True)
        assert return_value is None

        expected = Series(
            [np.nan, 1.0, 1.0, 3.0, 3.0], ["z", "a", "b", "c", "d"], dtype=float
        )
        tm.assert_series_equal(x[1:], expected[1:])
        assert np.isnan(x.iloc[0]), np.isnan(expected.iloc[0])

    def test_series_fillna_limit(self):
        index = np.arange(10)
        s = Series(np.random.default_rng(2).standard_normal(10), index=index)

        result = s[:2].reindex(index)
        result = result.fillna(method="pad", limit=5)

        expected = s[:2].reindex(index).fillna(method="pad")
        expected[-3:] = np.nan
        tm.assert_series_equal(result, expected)

        result = s[-2:].reindex(index)
        result = result.fillna(method="bfill", limit=5)

        expected = s[-2:].reindex(index).fillna(method="backfill")
        expected[:3] = np.nan
        tm.assert_series_equal(result, expected)

    def test_series_pad_backfill_limit(self):
        index = np.arange(10)
        s = Series(np.random.default_rng(2).standard_normal(10), index=index)

        result = s[:2].reindex(index, method="pad", limit=5)

        expected = s[:2].reindex(index).fillna(method="pad")
        expected[-3:] = np.nan
        tm.assert_series_equal(result, expected)

        result = s[-2:].reindex(index, method="backfill", limit=5)

        expected = s[-2:].reindex(index).fillna(method="backfill")
        expected[:3] = np.nan
        tm.assert_series_equal(result, expected)

    def test_fillna_int(self):
        ser = Series(np.random.default_rng(2).integers(-100, 100, 50))
        return_value = ser.fillna(method="ffill", inplace=True)
        assert return_value is None
        tm.assert_series_equal(ser.fillna(method="ffill", inplace=False), ser)

    def test_datetime64tz_fillna_round_issue(self):
        # GH#14872

        data = Series(
            [NaT, NaT, datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=pytz.utc)]
        )

        filled = data.bfill()

        expected = Series(
            [
                datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=pytz.utc),
                datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=pytz.utc),
                datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=pytz.utc),
            ]
        )

        tm.assert_series_equal(filled, expected)

    def test_fillna_parr(self):
        # GH-24537
        dti = date_range(
            Timestamp.max - Timedelta(nanoseconds=10), periods=5, freq="ns"
        )
        ser = Series(dti.to_period("ns"))
        ser[2] = NaT
        arr = period_array(
            [
                Timestamp("2262-04-11 23:47:16.854775797"),
                Timestamp("2262-04-11 23:47:16.854775798"),
                Timestamp("2262-04-11 23:47:16.854775798"),
                Timestamp("2262-04-11 23:47:16.854775800"),
                Timestamp("2262-04-11 23:47:16.854775801"),
            ],
            freq="ns",
        )
        expected = Series(arr)

        filled = ser.ffill()

        tm.assert_series_equal(filled, expected)

    @pytest.mark.parametrize("func", ["pad", "backfill"])
    def test_pad_backfill_deprecated(self, func):
        # GH#33396
        ser = Series([1, 2, 3])
        with tm.assert_produces_warning(FutureWarning):
            getattr(ser, func)()


@pytest.mark.parametrize(
    "data, expected_data, method, kwargs",
    (
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, 3.0, 3.0, 3.0, 7.0, np.nan, np.nan],
            "ffill",
            {"limit_area": "inside"},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, 3.0, np.nan, np.nan, 7.0, np.nan, np.nan],
            "ffill",
            {"limit_area": "inside", "limit": 1},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, 7.0],
            "ffill",
            {"limit_area": "outside"},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, np.nan],
            "ffill",
            {"limit_area": "outside", "limit": 1},
        ),
        (
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "ffill",
            {"limit_area": "outside", "limit": 1},
        ),
        (
            range(5),
            range(5),
            "ffill",
            {"limit_area": "outside", "limit": 1},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, 7.0, 7.0, 7.0, 7.0, np.nan, np.nan],
            "bfill",
            {"limit_area": "inside"},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, np.nan, np.nan, 7.0, 7.0, np.nan, np.nan],
            "bfill",
            {"limit_area": "inside", "limit": 1},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [3.0, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan, np.nan],
            "bfill",
            {"limit_area": "outside"},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan, np.nan],
            "bfill",
            {"limit_area": "outside", "limit": 1},
        ),
    ),
)
def test_ffill_bfill_limit_area(data, expected_data, method, kwargs):
    # GH#56492
    s = Series(data)
    expected = Series(expected_data)
    result = getattr(s, method)(**kwargs)
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\period\test_period.py ===
from datetime import (
    date,
    datetime,
    timedelta,
)
import re

import numpy as np
import pytest

from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.ccalendar import (
    DAYS,
    MONTHS,
)
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas._libs.tslibs.parsing import DateParseError
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG

from pandas import (
    NaT,
    Period,
    Timedelta,
    Timestamp,
    offsets,
)
import pandas._testing as tm

bday_msg = "Period with BDay freq is deprecated"


class TestPeriodDisallowedFreqs:
    @pytest.mark.parametrize(
        "freq, freq_msg",
        [
            (offsets.BYearBegin(), "BYearBegin"),
            (offsets.YearBegin(2), "YearBegin"),
            (offsets.QuarterBegin(startingMonth=12), "QuarterBegin"),
            (offsets.BusinessMonthEnd(2), "BusinessMonthEnd"),
        ],
    )
    def test_offsets_not_supported(self, freq, freq_msg):
        # GH#55785
        msg = re.escape(f"{freq} is not supported as period frequency")
        with pytest.raises(ValueError, match=msg):
            Period(year=2014, freq=freq)

    def test_custom_business_day_freq_raises(self):
        # GH#52534
        msg = "C is not supported as period frequency"
        with pytest.raises(ValueError, match=msg):
            Period("2023-04-10", freq="C")
        msg = f"{offsets.CustomBusinessDay().base} is not supported as period frequency"
        with pytest.raises(ValueError, match=msg):
            Period("2023-04-10", freq=offsets.CustomBusinessDay())

    def test_invalid_frequency_error_message(self):
        msg = "WOM-1MON is not supported as period frequency"
        with pytest.raises(ValueError, match=msg):
            Period("2012-01-02", freq="WOM-1MON")

    def test_invalid_frequency_period_error_message(self):
        msg = "for Period, please use 'M' instead of 'ME'"
        with pytest.raises(ValueError, match=msg):
            Period("2012-01-02", freq="ME")


class TestPeriodConstruction:
    def test_from_td64nat_raises(self):
        # GH#44507
        td = NaT.to_numpy("m8[ns]")

        msg = "Value must be Period, string, integer, or datetime"
        with pytest.raises(ValueError, match=msg):
            Period(td)

        with pytest.raises(ValueError, match=msg):
            Period(td, freq="D")

    def test_construction(self):
        i1 = Period("1/1/2005", freq="M")
        i2 = Period("Jan 2005")

        assert i1 == i2

        # GH#54105 - Period can be confusingly instantiated with lowercase freq
        # TODO: raise in the future an error when passing lowercase freq
        i1 = Period("2005", freq="Y")
        i2 = Period("2005")

        assert i1 == i2

        i4 = Period("2005", freq="M")
        assert i1 != i4

        i1 = Period.now(freq="Q")
        i2 = Period(datetime.now(), freq="Q")

        assert i1 == i2

        # Pass in freq as a keyword argument sometimes as a test for
        # https://github.com/pandas-dev/pandas/issues/53369
        i1 = Period.now(freq="D")
        i2 = Period(datetime.now(), freq="D")
        i3 = Period.now(offsets.Day())

        assert i1 == i2
        assert i1 == i3

        i1 = Period("1982", freq="min")
        msg = "'MIN' is deprecated and will be removed in a future version."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            i2 = Period("1982", freq="MIN")
        assert i1 == i2

        i1 = Period(year=2005, month=3, day=1, freq="D")
        i2 = Period("3/1/2005", freq="D")
        assert i1 == i2

        i3 = Period(year=2005, month=3, day=1, freq="d")
        assert i1 == i3

        i1 = Period("2007-01-01 09:00:00.001")
        expected = Period(datetime(2007, 1, 1, 9, 0, 0, 1000), freq="ms")
        assert i1 == expected

        expected = Period("2007-01-01 09:00:00.001", freq="ms")
        assert i1 == expected

        i1 = Period("2007-01-01 09:00:00.00101")
        expected = Period(datetime(2007, 1, 1, 9, 0, 0, 1010), freq="us")
        assert i1 == expected

        expected = Period("2007-01-01 09:00:00.00101", freq="us")
        assert i1 == expected

        msg = "Must supply freq for ordinal value"
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=200701)

        msg = "Invalid frequency: X"
        with pytest.raises(ValueError, match=msg):
            Period("2007-1-1", freq="X")

    def test_tuple_freq_disallowed(self):
        # GH#34703 tuple freq disallowed
        with pytest.raises(TypeError, match="pass as a string instead"):
            Period("1982", freq=("Min", 1))

        with pytest.raises(TypeError, match="pass as a string instead"):
            Period("2006-12-31", ("w", 1))

    def test_construction_from_timestamp_nanos(self):
        # GH#46811 don't drop nanos from Timestamp
        ts = Timestamp("2022-04-20 09:23:24.123456789")
        per = Period(ts, freq="ns")

        # should losslessly round-trip, not lose the 789
        rt = per.to_timestamp()
        assert rt == ts

        # same thing but from a datetime64 object
        dt64 = ts.asm8
        per2 = Period(dt64, freq="ns")
        rt2 = per2.to_timestamp()
        assert rt2.asm8 == dt64

    def test_construction_bday(self):
        # Biz day construction, roll forward if non-weekday
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            i1 = Period("3/10/12", freq="B")
            i2 = Period("3/10/12", freq="D")
            assert i1 == i2.asfreq("B")
            i2 = Period("3/11/12", freq="D")
            assert i1 == i2.asfreq("B")
            i2 = Period("3/12/12", freq="D")
            assert i1 == i2.asfreq("B")

            i3 = Period("3/10/12", freq="b")
            assert i1 == i3

            i1 = Period(year=2012, month=3, day=10, freq="B")
            i2 = Period("3/12/12", freq="B")
            assert i1 == i2

    def test_construction_quarter(self):
        i1 = Period(year=2005, quarter=1, freq="Q")
        i2 = Period("1/1/2005", freq="Q")
        assert i1 == i2

        i1 = Period(year=2005, quarter=3, freq="Q")
        i2 = Period("9/1/2005", freq="Q")
        assert i1 == i2

        i1 = Period("2005Q1")
        i2 = Period(year=2005, quarter=1, freq="Q")
        i3 = Period("2005q1")
        assert i1 == i2
        assert i1 == i3

        i1 = Period("05Q1")
        assert i1 == i2
        lower = Period("05q1")
        assert i1 == lower

        i1 = Period("1Q2005")
        assert i1 == i2
        lower = Period("1q2005")
        assert i1 == lower

        i1 = Period("1Q05")
        assert i1 == i2
        lower = Period("1q05")
        assert i1 == lower

        i1 = Period("4Q1984")
        assert i1.year == 1984
        lower = Period("4q1984")
        assert i1 == lower

    def test_construction_month(self):
        expected = Period("2007-01", freq="M")
        i1 = Period("200701", freq="M")
        assert i1 == expected

        i1 = Period("200701", freq="M")
        assert i1 == expected

        i1 = Period(200701, freq="M")
        assert i1 == expected

        i1 = Period(ordinal=200701, freq="M")
        assert i1.year == 18695

        i1 = Period(datetime(2007, 1, 1), freq="M")
        i2 = Period("200701", freq="M")
        assert i1 == i2

        i1 = Period(date(2007, 1, 1), freq="M")
        i2 = Period(datetime(2007, 1, 1), freq="M")
        i3 = Period(np.datetime64("2007-01-01"), freq="M")
        i4 = Period("2007-01-01 00:00:00", freq="M")
        i5 = Period("2007-01-01 00:00:00.000", freq="M")
        assert i1 == i2
        assert i1 == i3
        assert i1 == i4
        assert i1 == i5

    def test_period_constructor_offsets(self):
        assert Period("1/1/2005", freq=offsets.MonthEnd()) == Period(
            "1/1/2005", freq="M"
        )
        assert Period("2005", freq=offsets.YearEnd()) == Period("2005", freq="Y")
        assert Period("2005", freq=offsets.MonthEnd()) == Period("2005", freq="M")
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert Period("3/10/12", freq=offsets.BusinessDay()) == Period(
                "3/10/12", freq="B"
            )
        assert Period("3/10/12", freq=offsets.Day()) == Period("3/10/12", freq="D")

        assert Period(
            year=2005, quarter=1, freq=offsets.QuarterEnd(startingMonth=12)
        ) == Period(year=2005, quarter=1, freq="Q")
        assert Period(
            year=2005, quarter=2, freq=offsets.QuarterEnd(startingMonth=12)
        ) == Period(year=2005, quarter=2, freq="Q")

        assert Period(year=2005, month=3, day=1, freq=offsets.Day()) == Period(
            year=2005, month=3, day=1, freq="D"
        )
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert Period(year=2012, month=3, day=10, freq=offsets.BDay()) == Period(
                year=2012, month=3, day=10, freq="B"
            )

        expected = Period("2005-03-01", freq="3D")
        assert Period(year=2005, month=3, day=1, freq=offsets.Day(3)) == expected
        assert Period(year=2005, month=3, day=1, freq="3D") == expected

        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert Period(year=2012, month=3, day=10, freq=offsets.BDay(3)) == Period(
                year=2012, month=3, day=10, freq="3B"
            )

        assert Period(200701, freq=offsets.MonthEnd()) == Period(200701, freq="M")

        i1 = Period(ordinal=200701, freq=offsets.MonthEnd())
        i2 = Period(ordinal=200701, freq="M")
        assert i1 == i2
        assert i1.year == 18695
        assert i2.year == 18695

        i1 = Period(datetime(2007, 1, 1), freq="M")
        i2 = Period("200701", freq="M")
        assert i1 == i2

        i1 = Period(date(2007, 1, 1), freq="M")
        i2 = Period(datetime(2007, 1, 1), freq="M")
        i3 = Period(np.datetime64("2007-01-01"), freq="M")
        i4 = Period("2007-01-01 00:00:00", freq="M")
        i5 = Period("2007-01-01 00:00:00.000", freq="M")
        assert i1 == i2
        assert i1 == i3
        assert i1 == i4
        assert i1 == i5

        i1 = Period("2007-01-01 09:00:00.001")
        expected = Period(datetime(2007, 1, 1, 9, 0, 0, 1000), freq="ms")
        assert i1 == expected

        expected = Period("2007-01-01 09:00:00.001", freq="ms")
        assert i1 == expected

        i1 = Period("2007-01-01 09:00:00.00101")
        expected = Period(datetime(2007, 1, 1, 9, 0, 0, 1010), freq="us")
        assert i1 == expected

        expected = Period("2007-01-01 09:00:00.00101", freq="us")
        assert i1 == expected

    def test_invalid_arguments(self):
        msg = "Must supply freq for datetime value"
        with pytest.raises(ValueError, match=msg):
            Period(datetime.now())
        with pytest.raises(ValueError, match=msg):
            Period(datetime.now().date())

        msg = "Value must be Period, string, integer, or datetime"
        with pytest.raises(ValueError, match=msg):
            Period(1.6, freq="D")
        msg = "Ordinal must be an integer"
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=1.6, freq="D")
        msg = "Only value or ordinal but not both should be given but not both"
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=2, value=1, freq="D")

        msg = "If value is None, freq cannot be None"
        with pytest.raises(ValueError, match=msg):
            Period(month=1)

        msg = '^Given date string "-2000" not likely a datetime$'
        with pytest.raises(ValueError, match=msg):
            Period("-2000", "Y")
        msg = "day is out of range for month"
        with pytest.raises(DateParseError, match=msg):
            Period("0", "Y")
        msg = "Unknown datetime string format, unable to parse"
        with pytest.raises(DateParseError, match=msg):
            Period("1/1/-2000", "Y")

    def test_constructor_corner(self):
        expected = Period("2007-01", freq="2M")
        assert Period(year=2007, month=1, freq="2M") == expected

        assert Period(None) is NaT

        p = Period("2007-01-01", freq="D")

        result = Period(p, freq="Y")
        exp = Period("2007", freq="Y")
        assert result == exp

    def test_constructor_infer_freq(self):
        p = Period("2007-01-01")
        assert p.freq == "D"

        p = Period("2007-01-01 07")
        assert p.freq == "h"

        p = Period("2007-01-01 07:10")
        assert p.freq == "min"

        p = Period("2007-01-01 07:10:15")
        assert p.freq == "s"

        p = Period("2007-01-01 07:10:15.123")
        assert p.freq == "ms"

        # We see that there are 6 digits after the decimal, so get microsecond
        #  even though they are all zeros.
        p = Period("2007-01-01 07:10:15.123000")
        assert p.freq == "us"

        p = Period("2007-01-01 07:10:15.123400")
        assert p.freq == "us"

    def test_multiples(self):
        result1 = Period("1989", freq="2Y")
        result2 = Period("1989", freq="Y")
        assert result1.ordinal == result2.ordinal
        assert result1.freqstr == "2Y-DEC"
        assert result2.freqstr == "Y-DEC"
        assert result1.freq == offsets.YearEnd(2)
        assert result2.freq == offsets.YearEnd()

        assert (result1 + 1).ordinal == result1.ordinal + 2
        assert (1 + result1).ordinal == result1.ordinal + 2
        assert (result1 - 1).ordinal == result2.ordinal - 2
        assert (-1 + result1).ordinal == result2.ordinal - 2

    @pytest.mark.parametrize("month", MONTHS)
    def test_period_cons_quarterly(self, month):
        # bugs in scikits.timeseries
        freq = f"Q-{month}"
        exp = Period("1989Q3", freq=freq)
        assert "1989Q3" in str(exp)
        stamp = exp.to_timestamp("D", how="end")
        p = Period(stamp, freq=freq)
        assert p == exp

        stamp = exp.to_timestamp("3D", how="end")
        p = Period(stamp, freq=freq)
        assert p == exp

    @pytest.mark.parametrize("month", MONTHS)
    def test_period_cons_annual(self, month):
        # bugs in scikits.timeseries
        freq = f"Y-{month}"
        exp = Period("1989", freq=freq)
        stamp = exp.to_timestamp("D", how="end") + timedelta(days=30)
        p = Period(stamp, freq=freq)

        assert p == exp + 1
        assert isinstance(p, Period)

    @pytest.mark.parametrize("day", DAYS)
    @pytest.mark.parametrize("num", range(10, 17))
    def test_period_cons_weekly(self, num, day):
        daystr = f"2011-02-{num}"
        freq = f"W-{day}"

        result = Period(daystr, freq=freq)
        expected = Period(daystr, freq="D").asfreq(freq)
        assert result == expected
        assert isinstance(result, Period)

    def test_parse_week_str_roundstrip(self):
        # GH#50803
        per = Period("2017-01-23/2017-01-29")
        assert per.freq.freqstr == "W-SUN"

        per = Period("2017-01-24/2017-01-30")
        assert per.freq.freqstr == "W-MON"

        msg = "Could not parse as weekly-freq Period"
        with pytest.raises(ValueError, match=msg):
            # not 6 days apart
            Period("2016-01-23/2017-01-29")

    def test_period_from_ordinal(self):
        p = Period("2011-01", freq="M")
        res = Period._from_ordinal(p.ordinal, freq=p.freq)
        assert p == res
        assert isinstance(res, Period)

    @pytest.mark.parametrize("freq", ["Y", "M", "D", "h"])
    def test_construct_from_nat_string_and_freq(self, freq):
        per = Period("NaT", freq=freq)
        assert per is NaT

        per = Period("NaT", freq="2" + freq)
        assert per is NaT

        per = Period("NaT", freq="3" + freq)
        assert per is NaT

    def test_period_cons_nat(self):
        p = Period("nat", freq="W-SUN")
        assert p is NaT

        p = Period(iNaT, freq="D")
        assert p is NaT

        p = Period(iNaT, freq="3D")
        assert p is NaT

        p = Period(iNaT, freq="1D1h")
        assert p is NaT

        p = Period("NaT")
        assert p is NaT

        p = Period(iNaT)
        assert p is NaT

    def test_period_cons_mult(self):
        p1 = Period("2011-01", freq="3M")
        p2 = Period("2011-01", freq="M")
        assert p1.ordinal == p2.ordinal

        assert p1.freq == offsets.MonthEnd(3)
        assert p1.freqstr == "3M"

        assert p2.freq == offsets.MonthEnd()
        assert p2.freqstr == "M"

        result = p1 + 1
        assert result.ordinal == (p2 + 3).ordinal

        assert result.freq == p1.freq
        assert result.freqstr == "3M"

        result = p1 - 1
        assert result.ordinal == (p2 - 3).ordinal
        assert result.freq == p1.freq
        assert result.freqstr == "3M"

        msg = "Frequency must be positive, because it represents span: -3M"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="-3M")

        msg = "Frequency must be positive, because it represents span: 0M"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="0M")

    def test_period_cons_combined(self):
        p = [
            (
                Period("2011-01", freq="1D1h"),
                Period("2011-01", freq="1h1D"),
                Period("2011-01", freq="h"),
            ),
            (
                Period(ordinal=1, freq="1D1h"),
                Period(ordinal=1, freq="1h1D"),
                Period(ordinal=1, freq="h"),
            ),
        ]

        for p1, p2, p3 in p:
            assert p1.ordinal == p3.ordinal
            assert p2.ordinal == p3.ordinal

            assert p1.freq == offsets.Hour(25)
            assert p1.freqstr == "25h"

            assert p2.freq == offsets.Hour(25)
            assert p2.freqstr == "25h"

            assert p3.freq == offsets.Hour()
            assert p3.freqstr == "h"

            result = p1 + 1
            assert result.ordinal == (p3 + 25).ordinal
            assert result.freq == p1.freq
            assert result.freqstr == "25h"

            result = p2 + 1
            assert result.ordinal == (p3 + 25).ordinal
            assert result.freq == p2.freq
            assert result.freqstr == "25h"

            result = p1 - 1
            assert result.ordinal == (p3 - 25).ordinal
            assert result.freq == p1.freq
            assert result.freqstr == "25h"

            result = p2 - 1
            assert result.ordinal == (p3 - 25).ordinal
            assert result.freq == p2.freq
            assert result.freqstr == "25h"

        msg = "Frequency must be positive, because it represents span: -25h"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="-1D1h")
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="-1h1D")
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=1, freq="-1D1h")
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=1, freq="-1h1D")

        msg = "Frequency must be positive, because it represents span: 0D"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="0D0h")
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=1, freq="0D0h")

        # You can only combine together day and intraday offsets
        msg = "Invalid frequency: 1W1D"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="1W1D")
        msg = "Invalid frequency: 1D1W"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="1D1W")

    @pytest.mark.parametrize("day", ["1970/01/01 ", "2020-12-31 ", "1981/09/13 "])
    @pytest.mark.parametrize("hour", ["00:00:00", "00:00:01", "23:59:59", "12:00:59"])
    @pytest.mark.parametrize(
        "sec_float, expected",
        [
            (".000000001", 1),
            (".000000999", 999),
            (".123456789", 789),
            (".999999999", 999),
            (".999999000", 0),
            # Test femtoseconds, attoseconds, picoseconds are dropped like Timestamp
            (".999999001123", 1),
            (".999999001123456", 1),
            (".999999001123456789", 1),
        ],
    )
    def test_period_constructor_nanosecond(self, day, hour, sec_float, expected):
        # GH 34621

        assert Period(day + hour + sec_float).start_time.nanosecond == expected

    @pytest.mark.parametrize("hour", range(24))
    def test_period_large_ordinal(self, hour):
        # Issue #36430
        # Integer overflow for Period over the maximum timestamp
        p = Period(ordinal=2562048 + hour, freq="1h")
        assert p.hour == hour


class TestPeriodMethods:
    def test_round_trip(self):
        p = Period("2000Q1")
        new_p = tm.round_trip_pickle(p)
        assert new_p == p

    def test_hash(self):
        assert hash(Period("2011-01", freq="M")) == hash(Period("2011-01", freq="M"))

        assert hash(Period("2011-01-01", freq="D")) != hash(Period("2011-01", freq="M"))

        assert hash(Period("2011-01", freq="3M")) != hash(Period("2011-01", freq="2M"))

        assert hash(Period("2011-01", freq="M")) != hash(Period("2011-02", freq="M"))

    # --------------------------------------------------------------
    # to_timestamp

    def test_to_timestamp_mult(self):
        p = Period("2011-01", freq="M")
        assert p.to_timestamp(how="S") == Timestamp("2011-01-01")
        expected = Timestamp("2011-02-01") - Timedelta(1, "ns")
        assert p.to_timestamp(how="E") == expected

        p = Period("2011-01", freq="3M")
        assert p.to_timestamp(how="S") == Timestamp("2011-01-01")
        expected = Timestamp("2011-04-01") - Timedelta(1, "ns")
        assert p.to_timestamp(how="E") == expected

    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    def test_to_timestamp(self):
        p = Period("1982", freq="Y")
        start_ts = p.to_timestamp(how="S")
        aliases = ["s", "StarT", "BEGIn"]
        for a in aliases:
            assert start_ts == p.to_timestamp("D", how=a)
            # freq with mult should not affect to the result
            assert start_ts == p.to_timestamp("3D", how=a)

        end_ts = p.to_timestamp(how="E")
        aliases = ["e", "end", "FINIsH"]
        for a in aliases:
            assert end_ts == p.to_timestamp("D", how=a)
            assert end_ts == p.to_timestamp("3D", how=a)

        from_lst = ["Y", "Q", "M", "W", "B", "D", "h", "Min", "s"]

        def _ex(p):
            if p.freq == "B":
                return p.start_time + Timedelta(days=1, nanoseconds=-1)
            return Timestamp((p + p.freq).start_time._value - 1)

        for fcode in from_lst:
            p = Period("1982", freq=fcode)
            result = p.to_timestamp().to_period(fcode)
            assert result == p

            assert p.start_time == p.to_timestamp(how="S")

            assert p.end_time == _ex(p)

        # Frequency other than daily

        p = Period("1985", freq="Y")

        result = p.to_timestamp("h", how="end")
        expected = Timestamp(1986, 1, 1) - Timedelta(1, "ns")
        assert result == expected
        result = p.to_timestamp("3h", how="end")
        assert result == expected

        result = p.to_timestamp("min", how="end")
        expected = Timestamp(1986, 1, 1) - Timedelta(1, "ns")
        assert result == expected
        result = p.to_timestamp("2min", how="end")
        assert result == expected

        result = p.to_timestamp(how="end")
        expected = Timestamp(1986, 1, 1) - Timedelta(1, "ns")
        assert result == expected

        expected = datetime(1985, 1, 1)
        result = p.to_timestamp("h", how="start")
        assert result == expected
        result = p.to_timestamp("min", how="start")
        assert result == expected
        result = p.to_timestamp("s", how="start")
        assert result == expected
        result = p.to_timestamp("3h", how="start")
        assert result == expected
        result = p.to_timestamp("5s", how="start")
        assert result == expected

    def test_to_timestamp_business_end(self):
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            per = Period("1990-01-05", "B")  # Friday
            result = per.to_timestamp("B", how="E")

        expected = Timestamp("1990-01-06") - Timedelta(nanoseconds=1)
        assert result == expected

    @pytest.mark.parametrize(
        "ts, expected",
        [
            ("1970-01-01 00:00:00", 0),
            ("1970-01-01 00:00:00.000001", 1),
            ("1970-01-01 00:00:00.00001", 10),
            ("1970-01-01 00:00:00.499", 499000),
            ("1999-12-31 23:59:59.999", 999000),
            ("1999-12-31 23:59:59.999999", 999999),
            ("2050-12-31 23:59:59.5", 500000),
            ("2050-12-31 23:59:59.500001", 500001),
            ("2050-12-31 23:59:59.123456", 123456),
        ],
    )
    @pytest.mark.parametrize("freq", [None, "us", "ns"])
    def test_to_timestamp_microsecond(self, ts, expected, freq):
        # GH 24444
        result = Period(ts).to_timestamp(freq=freq).microsecond
        assert result == expected

    # --------------------------------------------------------------
    # Rendering: __repr__, strftime, etc

    @pytest.mark.parametrize(
        "str_ts,freq,str_res,str_freq",
        (
            ("Jan-2000", None, "2000-01", "M"),
            ("2000-12-15", None, "2000-12-15", "D"),
            (
                "2000-12-15 13:45:26.123456789",
                "ns",
                "2000-12-15 13:45:26.123456789",
                "ns",
            ),
            ("2000-12-15 13:45:26.123456789", "us", "2000-12-15 13:45:26.123456", "us"),
            ("2000-12-15 13:45:26.123456", None, "2000-12-15 13:45:26.123456", "us"),
            ("2000-12-15 13:45:26.123456789", "ms", "2000-12-15 13:45:26.123", "ms"),
            ("2000-12-15 13:45:26.123", None, "2000-12-15 13:45:26.123", "ms"),
            ("2000-12-15 13:45:26", "s", "2000-12-15 13:45:26", "s"),
            ("2000-12-15 13:45:26", "min", "2000-12-15 13:45", "min"),
            ("2000-12-15 13:45:26", "h", "2000-12-15 13:00", "h"),
            ("2000-12-15", "Y", "2000", "Y-DEC"),
            ("2000-12-15", "Q", "2000Q4", "Q-DEC"),
            ("2000-12-15", "M", "2000-12", "M"),
            ("2000-12-15", "W", "2000-12-11/2000-12-17", "W-SUN"),
            ("2000-12-15", "D", "2000-12-15", "D"),
            ("2000-12-15", "B", "2000-12-15", "B"),
        ),
    )
    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    def test_repr(self, str_ts, freq, str_res, str_freq):
        p = Period(str_ts, freq=freq)
        assert str(p) == str_res
        assert repr(p) == f"Period('{str_res}', '{str_freq}')"

    def test_repr_nat(self):
        p = Period("nat", freq="M")
        assert repr(NaT) in repr(p)

    def test_strftime(self):
        # GH#3363
        p = Period("2000-1-1 12:34:12", freq="s")
        res = p.strftime("%Y-%m-%d %H:%M:%S")
        assert res == "2000-01-01 12:34:12"
        assert isinstance(res, str)


class TestPeriodProperties:
    """Test properties such as year, month, weekday, etc...."""

    @pytest.mark.parametrize("freq", ["Y", "M", "D", "h"])
    def test_is_leap_year(self, freq):
        # GH 13727
        p = Period("2000-01-01 00:00:00", freq=freq)
        assert p.is_leap_year
        assert isinstance(p.is_leap_year, bool)

        p = Period("1999-01-01 00:00:00", freq=freq)
        assert not p.is_leap_year

        p = Period("2004-01-01 00:00:00", freq=freq)
        assert p.is_leap_year

        p = Period("2100-01-01 00:00:00", freq=freq)
        assert not p.is_leap_year

    def test_quarterly_negative_ordinals(self):
        p = Period(ordinal=-1, freq="Q-DEC")
        assert p.year == 1969
        assert p.quarter == 4
        assert isinstance(p, Period)

        p = Period(ordinal=-2, freq="Q-DEC")
        assert p.year == 1969
        assert p.quarter == 3
        assert isinstance(p, Period)

        p = Period(ordinal=-2, freq="M")
        assert p.year == 1969
        assert p.month == 11
        assert isinstance(p, Period)

    def test_freq_str(self):
        i1 = Period("1982", freq="Min")
        assert i1.freq == offsets.Minute()
        assert i1.freqstr == "min"

    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    def test_period_deprecated_freq(self):
        cases = {
            "M": ["MTH", "MONTH", "MONTHLY", "Mth", "month", "monthly"],
            "B": ["BUS", "BUSINESS", "BUSINESSLY", "WEEKDAY", "bus"],
            "D": ["DAY", "DLY", "DAILY", "Day", "Dly", "Daily"],
            "h": ["HR", "HOUR", "HRLY", "HOURLY", "hr", "Hour", "HRly"],
            "min": ["minute", "MINUTE", "MINUTELY", "minutely"],
            "s": ["sec", "SEC", "SECOND", "SECONDLY", "second"],
            "ms": ["MILLISECOND", "MILLISECONDLY", "millisecond"],
            "us": ["MICROSECOND", "MICROSECONDLY", "microsecond"],
            "ns": ["NANOSECOND", "NANOSECONDLY", "nanosecond"],
        }

        msg = INVALID_FREQ_ERR_MSG
        for exp, freqs in cases.items():
            for freq in freqs:
                with pytest.raises(ValueError, match=msg):
                    Period("2016-03-01 09:00", freq=freq)
                with pytest.raises(ValueError, match=msg):
                    Period(ordinal=1, freq=freq)

            # check supported freq-aliases still works
            p1 = Period("2016-03-01 09:00", freq=exp)
            p2 = Period(ordinal=1, freq=exp)
            assert isinstance(p1, Period)
            assert isinstance(p2, Period)

    @staticmethod
    def _period_constructor(bound, offset):
        return Period(
            year=bound.year,
            month=bound.month,
            day=bound.day,
            hour=bound.hour,
            minute=bound.minute,
            second=bound.second + offset,
            freq="us",
        )

    @pytest.mark.parametrize("bound, offset", [(Timestamp.min, -1), (Timestamp.max, 1)])
    @pytest.mark.parametrize("period_property", ["start_time", "end_time"])
    def test_outer_bounds_start_and_end_time(self, bound, offset, period_property):
        # GH #13346
        period = TestPeriodProperties._period_constructor(bound, offset)
        with pytest.raises(OutOfBoundsDatetime, match="Out of bounds nanosecond"):
            getattr(period, period_property)

    @pytest.mark.parametrize("bound, offset", [(Timestamp.min, -1), (Timestamp.max, 1)])
    @pytest.mark.parametrize("period_property", ["start_time", "end_time"])
    def test_inner_bounds_start_and_end_time(self, bound, offset, period_property):
        # GH #13346
        period = TestPeriodProperties._period_constructor(bound, -offset)
        expected = period.to_timestamp().round(freq="s")
        assert getattr(period, period_property).round(freq="s") == expected
        expected = (bound - offset * Timedelta(1, unit="s")).floor("s")
        assert getattr(period, period_property).floor("s") == expected

    def test_start_time(self):
        freq_lst = ["Y", "Q", "M", "D", "h", "min", "s"]
        xp = datetime(2012, 1, 1)
        for f in freq_lst:
            p = Period("2012", freq=f)
            assert p.start_time == xp
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert Period("2012", freq="B").start_time == datetime(2012, 1, 2)
        assert Period("2012", freq="W").start_time == datetime(2011, 12, 26)

    def test_end_time(self):
        p = Period("2012", freq="Y")

        def _ex(*args):
            return Timestamp(Timestamp(datetime(*args)).as_unit("ns")._value - 1)

        xp = _ex(2013, 1, 1)
        assert xp == p.end_time

        p = Period("2012", freq="Q")
        xp = _ex(2012, 4, 1)
        assert xp == p.end_time

        p = Period("2012", freq="M")
        xp = _ex(2012, 2, 1)
        assert xp == p.end_time

        p = Period("2012", freq="D")
        xp = _ex(2012, 1, 2)
        assert xp == p.end_time

        p = Period("2012", freq="h")
        xp = _ex(2012, 1, 1, 1)
        assert xp == p.end_time

        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            p = Period("2012", freq="B")
            xp = _ex(2012, 1, 3)
            assert xp == p.end_time

        p = Period("2012", freq="W")
        xp = _ex(2012, 1, 2)
        assert xp == p.end_time

        # Test for GH 11738
        p = Period("2012", freq="15D")
        xp = _ex(2012, 1, 16)
        assert xp == p.end_time

        p = Period("2012", freq="1D1h")
        xp = _ex(2012, 1, 2, 1)
        assert xp == p.end_time

        p = Period("2012", freq="1h1D")
        xp = _ex(2012, 1, 2, 1)
        assert xp == p.end_time

    def test_end_time_business_friday(self):
        # GH#34449
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            per = Period("1990-01-05", "B")
            result = per.end_time

        expected = Timestamp("1990-01-06") - Timedelta(nanoseconds=1)
        assert result == expected

    def test_anchor_week_end_time(self):
        def _ex(*args):
            return Timestamp(Timestamp(datetime(*args)).as_unit("ns")._value - 1)

        p = Period("2013-1-1", "W-SAT")
        xp = _ex(2013, 1, 6)
        assert p.end_time == xp

    def test_properties_annually(self):
        # Test properties on Periods with annually frequency.
        a_date = Period(freq="Y", year=2007)
        assert a_date.year == 2007

    def test_properties_quarterly(self):
        # Test properties on Periods with daily frequency.
        qedec_date = Period(freq="Q-DEC", year=2007, quarter=1)
        qejan_date = Period(freq="Q-JAN", year=2007, quarter=1)
        qejun_date = Period(freq="Q-JUN", year=2007, quarter=1)
        #
        for x in range(3):
            for qd in (qedec_date, qejan_date, qejun_date):
                assert (qd + x).qyear == 2007
                assert (qd + x).quarter == x + 1

    def test_properties_monthly(self):
        # Test properties on Periods with daily frequency.
        m_date = Period(freq="M", year=2007, month=1)
        for x in range(11):
            m_ival_x = m_date + x
            assert m_ival_x.year == 2007
            if 1 <= x + 1 <= 3:
                assert m_ival_x.quarter == 1
            elif 4 <= x + 1 <= 6:
                assert m_ival_x.quarter == 2
            elif 7 <= x + 1 <= 9:
                assert m_ival_x.quarter == 3
            elif 10 <= x + 1 <= 12:
                assert m_ival_x.quarter == 4
            assert m_ival_x.month == x + 1

    def test_properties_weekly(self):
        # Test properties on Periods with daily frequency.
        w_date = Period(freq="W", year=2007, month=1, day=7)
        #
        assert w_date.year == 2007
        assert w_date.quarter == 1
        assert w_date.month == 1
        assert w_date.week == 1
        assert (w_date - 1).week == 52
        assert w_date.days_in_month == 31
        assert Period(freq="W", year=2012, month=2, day=1).days_in_month == 29

    def test_properties_weekly_legacy(self):
        # Test properties on Periods with daily frequency.
        w_date = Period(freq="W", year=2007, month=1, day=7)
        assert w_date.year == 2007
        assert w_date.quarter == 1
        assert w_date.month == 1
        assert w_date.week == 1
        assert (w_date - 1).week == 52
        assert w_date.days_in_month == 31

        exp = Period(freq="W", year=2012, month=2, day=1)
        assert exp.days_in_month == 29

        msg = INVALID_FREQ_ERR_MSG
        with pytest.raises(ValueError, match=msg):
            Period(freq="WK", year=2007, month=1, day=7)

    def test_properties_daily(self):
        # Test properties on Periods with daily frequency.
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            b_date = Period(freq="B", year=2007, month=1, day=1)
        #
        assert b_date.year == 2007
        assert b_date.quarter == 1
        assert b_date.month == 1
        assert b_date.day == 1
        assert b_date.weekday == 0
        assert b_date.dayofyear == 1
        assert b_date.days_in_month == 31
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert Period(freq="B", year=2012, month=2, day=1).days_in_month == 29

        d_date = Period(freq="D", year=2007, month=1, day=1)

        assert d_date.year == 2007
        assert d_date.quarter == 1
        assert d_date.month == 1
        assert d_date.day == 1
        assert d_date.weekday == 0
        assert d_date.dayofyear == 1
        assert d_date.days_in_month == 31
        assert Period(freq="D", year=2012, month=2, day=1).days_in_month == 29

    def test_properties_hourly(self):
        # Test properties on Periods with hourly frequency.
        h_date1 = Period(freq="h", year=2007, month=1, day=1, hour=0)
        h_date2 = Period(freq="2h", year=2007, month=1, day=1, hour=0)

        for h_date in [h_date1, h_date2]:
            assert h_date.year == 2007
            assert h_date.quarter == 1
            assert h_date.month == 1
            assert h_date.day == 1
            assert h_date.weekday == 0
            assert h_date.dayofyear == 1
            assert h_date.hour == 0
            assert h_date.days_in_month == 31
            assert (
                Period(freq="h", year=2012, month=2, day=1, hour=0).days_in_month == 29
            )

    def test_properties_minutely(self):
        # Test properties on Periods with minutely frequency.
        t_date = Period(freq="Min", year=2007, month=1, day=1, hour=0, minute=0)
        #
        assert t_date.quarter == 1
        assert t_date.month == 1
        assert t_date.day == 1
        assert t_date.weekday == 0
        assert t_date.dayofyear == 1
        assert t_date.hour == 0
        assert t_date.minute == 0
        assert t_date.days_in_month == 31
        assert (
            Period(freq="D", year=2012, month=2, day=1, hour=0, minute=0).days_in_month
            == 29
        )

    def test_properties_secondly(self):
        # Test properties on Periods with secondly frequency.
        s_date = Period(
            freq="Min", year=2007, month=1, day=1, hour=0, minute=0, second=0
        )
        #
        assert s_date.year == 2007
        assert s_date.quarter == 1
        assert s_date.month == 1
        assert s_date.day == 1
        assert s_date.weekday == 0
        assert s_date.dayofyear == 1
        assert s_date.hour == 0
        assert s_date.minute == 0
        assert s_date.second == 0
        assert s_date.days_in_month == 31
        assert (
            Period(
                freq="Min", year=2012, month=2, day=1, hour=0, minute=0, second=0
            ).days_in_month
            == 29
        )


class TestPeriodComparisons:
    def test_sort_periods(self):
        jan = Period("2000-01", "M")
        feb = Period("2000-02", "M")
        mar = Period("2000-03", "M")
        periods = [mar, jan, feb]
        correctPeriods = [jan, feb, mar]
        assert sorted(periods) == correctPeriods


def test_period_immutable():
    # see gh-17116
    msg = "not writable"

    per = Period("2014Q1")
    with pytest.raises(AttributeError, match=msg):
        per.ordinal = 14

    freq = per.freq
    with pytest.raises(AttributeError, match=msg):
        per.freq = 2 * freq


def test_small_year_parsing():
    per1 = Period("0001-01-07", "D")
    assert per1.year == 1
    assert per1.day == 7


def test_negone_ordinals():
    freqs = ["Y", "M", "Q", "D", "h", "min", "s"]

    period = Period(ordinal=-1, freq="D")
    for freq in freqs:
        repr(period.asfreq(freq))

    for freq in freqs:
        period = Period(ordinal=-1, freq=freq)
        repr(period)
        assert period.year == 1969

    with tm.assert_produces_warning(FutureWarning, match=bday_msg):
        period = Period(ordinal=-1, freq="B")
    repr(period)
    period = Period(ordinal=-1, freq="W")
    repr(period)