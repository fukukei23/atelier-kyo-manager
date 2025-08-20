
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_fillna.py ===
import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    NaT,
    PeriodIndex,
    Series,
    TimedeltaIndex,
    Timestamp,
    date_range,
    to_datetime,
)
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float


class TestFillNA:
    def test_fillna_dict_inplace_nonunique_columns(
        self, using_copy_on_write, warn_copy_on_write
    ):
        df = DataFrame(
            {"A": [np.nan] * 3, "B": [NaT, Timestamp(1), NaT], "C": [np.nan, "foo", 2]}
        )
        df.columns = ["A", "A", "A"]
        orig = df[:]

        # TODO(CoW-warn) better warning message
        with tm.assert_cow_warning(warn_copy_on_write):
            df.fillna({"A": 2}, inplace=True)
        # The first and third columns can be set inplace, while the second cannot.

        expected = DataFrame(
            {"A": [2.0] * 3, "B": [2, Timestamp(1), 2], "C": [2, "foo", 2]}
        )
        expected.columns = ["A", "A", "A"]
        tm.assert_frame_equal(df, expected)

        # TODO: what's the expected/desired behavior with CoW?
        if not using_copy_on_write:
            assert tm.shares_memory(df.iloc[:, 0], orig.iloc[:, 0])
        assert not tm.shares_memory(df.iloc[:, 1], orig.iloc[:, 1])
        if not using_copy_on_write:
            assert tm.shares_memory(df.iloc[:, 2], orig.iloc[:, 2])

    @td.skip_array_manager_not_yet_implemented
    def test_fillna_on_column_view(self, using_copy_on_write):
        # GH#46149 avoid unnecessary copies
        arr = np.full((40, 50), np.nan)
        df = DataFrame(arr, copy=False)

        if using_copy_on_write:
            with tm.raises_chained_assignment_error():
                df[0].fillna(-1, inplace=True)
            assert np.isnan(arr[:, 0]).all()
        else:
            with tm.assert_produces_warning(FutureWarning, match="inplace method"):
                df[0].fillna(-1, inplace=True)
            assert (arr[:, 0] == -1).all()

        # i.e. we didn't create a new 49-column block
        assert len(df._mgr.arrays) == 1
        assert np.shares_memory(df.values, arr)

    def test_fillna_datetime(self, datetime_frame):
        tf = datetime_frame
        tf.loc[tf.index[:5], "A"] = np.nan
        tf.loc[tf.index[-5:], "A"] = np.nan

        zero_filled = datetime_frame.fillna(0)
        assert (zero_filled.loc[zero_filled.index[:5], "A"] == 0).all()

        msg = "DataFrame.fillna with 'method' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            padded = datetime_frame.fillna(method="pad")
        assert np.isnan(padded.loc[padded.index[:5], "A"]).all()
        assert (
            padded.loc[padded.index[-5:], "A"] == padded.loc[padded.index[-5], "A"]
        ).all()

        msg = "Must specify a fill 'value' or 'method'"
        with pytest.raises(ValueError, match=msg):
            datetime_frame.fillna()
        msg = "Cannot specify both 'value' and 'method'"
        with pytest.raises(ValueError, match=msg):
            datetime_frame.fillna(5, method="ffill")

    def test_fillna_mixed_type(self, float_string_frame):
        mf = float_string_frame
        mf.loc[mf.index[5:20], "foo"] = np.nan
        mf.loc[mf.index[-10:], "A"] = np.nan
        # TODO: make stronger assertion here, GH 25640
        mf.fillna(value=0)
        msg = "DataFrame.fillna with 'method' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            mf.fillna(method="pad")

    def test_fillna_mixed_float(self, mixed_float_frame):
        # mixed numeric (but no float16)
        mf = mixed_float_frame.reindex(columns=["A", "B", "D"])
        mf.loc[mf.index[-10:], "A"] = np.nan
        result = mf.fillna(value=0)
        _check_mixed_float(result, dtype={"C": None})

        msg = "DataFrame.fillna with 'method' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = mf.fillna(method="pad")
        _check_mixed_float(result, dtype={"C": None})

    def test_fillna_empty(self, using_copy_on_write):
        if using_copy_on_write:
            pytest.skip("condition is unnecessary complex and is deprecated anyway")
        # empty frame (GH#2778)
        df = DataFrame(columns=["x"])
        for m in ["pad", "backfill"]:
            msg = "Series.fillna with 'method' is deprecated"
            with tm.assert_produces_warning(FutureWarning, match=msg):
                df.x.fillna(method=m, inplace=True)
                df.x.fillna(method=m)

    def test_fillna_different_dtype(self):
        # with different dtype (GH#3386)
        df = DataFrame(
            [["a", "a", np.nan, "a"], ["b", "b", np.nan, "b"], ["c", "c", np.nan, "c"]]
        )

        result = df.fillna({2: "foo"})
        expected = DataFrame(
            [["a", "a", "foo", "a"], ["b", "b", "foo", "b"], ["c", "c", "foo", "c"]]
        )
        # column is originally float (all-NaN) -> filling with string gives object dtype
        expected[2] = expected[2].astype("object")
        tm.assert_frame_equal(result, expected)

        return_value = df.fillna({2: "foo"}, inplace=True)
        tm.assert_frame_equal(df, expected)
        assert return_value is None

    def test_fillna_limit_and_value(self):
        # limit and value
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 3)))
        df.iloc[2:7, 0] = np.nan
        df.iloc[3:5, 2] = np.nan

        expected = df.copy()
        expected.iloc[2, 0] = 999
        expected.iloc[3, 2] = 999
        result = df.fillna(999, limit=1)
        tm.assert_frame_equal(result, expected)

    def test_fillna_datelike(self):
        # with datelike
        # GH#6344
        df = DataFrame(
            {
                "Date": [NaT, Timestamp("2014-1-1")],
                "Date2": [Timestamp("2013-1-1"), NaT],
            }
        )

        expected = df.copy()
        expected["Date"] = expected["Date"].fillna(df.loc[df.index[0], "Date2"])
        result = df.fillna(value={"Date": df["Date2"]})
        tm.assert_frame_equal(result, expected)

    def test_fillna_tzaware(self):
        # with timezone
        # GH#15855
        df = DataFrame({"A": [Timestamp("2012-11-11 00:00:00+01:00"), NaT]})
        exp = DataFrame(
            {
                "A": [
                    Timestamp("2012-11-11 00:00:00+01:00"),
                    Timestamp("2012-11-11 00:00:00+01:00"),
                ]
            }
        )
        msg = "DataFrame.fillna with 'method' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = df.fillna(method="pad")
        tm.assert_frame_equal(res, exp)

        df = DataFrame({"A": [NaT, Timestamp("2012-11-11 00:00:00+01:00")]})
        exp = DataFrame(
            {
                "A": [
                    Timestamp("2012-11-11 00:00:00+01:00"),
                    Timestamp("2012-11-11 00:00:00+01:00"),
                ]
            }
        )
        msg = "DataFrame.fillna with 'method' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = df.fillna(method="bfill")
        tm.assert_frame_equal(res, exp)

    def test_fillna_tzaware_different_column(self):
        # with timezone in another column
        # GH#15522
        df = DataFrame(
            {
                "A": date_range("20130101", periods=4, tz="US/Eastern"),
                "B": [1, 2, np.nan, np.nan],
            }
        )
        msg = "DataFrame.fillna with 'method' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.fillna(method="pad")
        expected = DataFrame(
            {
                "A": date_range("20130101", periods=4, tz="US/Eastern"),
                "B": [1.0, 2.0, 2.0, 2.0],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_na_actions_categorical(self):
        cat = Categorical([1, 2, 3, np.nan], categories=[1, 2, 3])
        vals = ["a", "b", np.nan, "d"]
        df = DataFrame({"cats": cat, "vals": vals})
        cat2 = Categorical([1, 2, 3, 3], categories=[1, 2, 3])
        vals2 = ["a", "b", "b", "d"]
        df_exp_fill = DataFrame({"cats": cat2, "vals": vals2})
        cat3 = Categorical([1, 2, 3], categories=[1, 2, 3])
        vals3 = ["a", "b", np.nan]
        df_exp_drop_cats = DataFrame({"cats": cat3, "vals": vals3})
        cat4 = Categorical([1, 2], categories=[1, 2, 3])
        vals4 = ["a", "b"]
        df_exp_drop_all = DataFrame({"cats": cat4, "vals": vals4})

        # fillna
        res = df.fillna(value={"cats": 3, "vals": "b"})
        tm.assert_frame_equal(res, df_exp_fill)

        msg = "Cannot setitem on a Categorical with a new category"
        with pytest.raises(TypeError, match=msg):
            df.fillna(value={"cats": 4, "vals": "c"})

        msg = "DataFrame.fillna with 'method' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = df.fillna(method="pad")
        tm.assert_frame_equal(res, df_exp_fill)

        # dropna
        res = df.dropna(subset=["cats"])
        tm.assert_frame_equal(res, df_exp_drop_cats)

        res = df.dropna()
        tm.assert_frame_equal(res, df_exp_drop_all)

        # make sure that fillna takes missing values into account
        c = Categorical([np.nan, "b", np.nan], categories=["a", "b"])
        df = DataFrame({"cats": c, "vals": [1, 2, 3]})

        cat_exp = Categorical(["a", "b", "a"], categories=["a", "b"])
        df_exp = DataFrame({"cats": cat_exp, "vals": [1, 2, 3]})

        res = df.fillna("a")
        tm.assert_frame_equal(res, df_exp)

    def test_fillna_categorical_nan(self):
        # GH#14021
        # np.nan should always be a valid filler
        cat = Categorical([np.nan, 2, np.nan])
        val = Categorical([np.nan, np.nan, np.nan])
        df = DataFrame({"cats": cat, "vals": val})

        # GH#32950 df.median() is poorly behaved because there is no
        #  Categorical.median
        median = Series({"cats": 2.0, "vals": np.nan})

        res = df.fillna(median)
        v_exp = [np.nan, np.nan, np.nan]
        df_exp = DataFrame({"cats": [2, 2, 2], "vals": v_exp}, dtype="category")
        tm.assert_frame_equal(res, df_exp)

        result = df.cats.fillna(np.nan)
        tm.assert_series_equal(result, df.cats)

        result = df.vals.fillna(np.nan)
        tm.assert_series_equal(result, df.vals)

        idx = DatetimeIndex(
            ["2011-01-01 09:00", "2016-01-01 23:45", "2011-01-01 09:00", NaT, NaT]
        )
        df = DataFrame({"a": Categorical(idx)})
        tm.assert_frame_equal(df.fillna(value=NaT), df)

        idx = PeriodIndex(["2011-01", "2011-01", "2011-01", NaT, NaT], freq="M")
        df = DataFrame({"a": Categorical(idx)})
        tm.assert_frame_equal(df.fillna(value=NaT), df)

        idx = TimedeltaIndex(["1 days", "2 days", "1 days", NaT, NaT])
        df = DataFrame({"a": Categorical(idx)})
        tm.assert_frame_equal(df.fillna(value=NaT), df)

    def test_fillna_downcast(self):
        # GH#15277
        # infer int64 from float64
        df = DataFrame({"a": [1.0, np.nan]})
        msg = "The 'downcast' keyword in fillna is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.fillna(0, downcast="infer")
        expected = DataFrame({"a": [1, 0]})
        tm.assert_frame_equal(result, expected)

        # infer int64 from float64 when fillna value is a dict
        df = DataFrame({"a": [1.0, np.nan]})
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.fillna({"a": 0}, downcast="infer")
        expected = DataFrame({"a": [1, 0]})
        tm.assert_frame_equal(result, expected)

    def test_fillna_downcast_false(self, frame_or_series):
        # GH#45603 preserve object dtype with downcast=False
        obj = frame_or_series([1, 2, 3], dtype="object")
        msg = "The 'downcast' keyword in fillna"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = obj.fillna("", downcast=False)
        tm.assert_equal(result, obj)

    def test_fillna_downcast_noop(self, frame_or_series):
        # GH#45423
        # Two relevant paths:
        #  1) not _can_hold_na (e.g. integer)
        #  2) _can_hold_na + noop + not can_hold_element

        obj = frame_or_series([1, 2, 3], dtype=np.int64)

        msg = "The 'downcast' keyword in fillna"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # GH#40988
            res = obj.fillna("foo", downcast=np.dtype(np.int32))
        expected = obj.astype(np.int32)
        tm.assert_equal(res, expected)

        obj2 = obj.astype(np.float64)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res2 = obj2.fillna("foo", downcast="infer")
        expected2 = obj  # get back int64
        tm.assert_equal(res2, expected2)

        with tm.assert_produces_warning(FutureWarning, match=msg):
            # GH#40988
            res3 = obj2.fillna("foo", downcast=np.dtype(np.int32))
        tm.assert_equal(res3, expected)

    @pytest.mark.parametrize("columns", [["A", "A", "B"], ["A", "A"]])
    def test_fillna_dictlike_value_duplicate_colnames(self, columns):
        # GH#43476
        df = DataFrame(np.nan, index=[0, 1], columns=columns)
        with tm.assert_produces_warning(None):
            result = df.fillna({"A": 0})

        expected = df.copy()
        expected["A"] = 0.0
        tm.assert_frame_equal(result, expected)

    def test_fillna_dtype_conversion(self, using_infer_string):
        # make sure that fillna on an empty frame works
        df = DataFrame(index=["A", "B", "C"], columns=[1, 2, 3, 4, 5])
        result = df.dtypes
        expected = Series([np.dtype("object")] * 5, index=[1, 2, 3, 4, 5])
        tm.assert_series_equal(result, expected)

        msg = "Downcasting object dtype arrays"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.fillna(1)
        expected = DataFrame(1, index=["A", "B", "C"], columns=[1, 2, 3, 4, 5])
        tm.assert_frame_equal(result, expected)

        # empty block
        df = DataFrame(index=range(3), columns=["A", "B"], dtype="float64")
        result = df.fillna("nan")
        expected = DataFrame("nan", index=range(3), columns=["A", "B"], dtype=object)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("val", ["", 1, np.nan, 1.0])
    def test_fillna_dtype_conversion_equiv_replace(self, val):
        df = DataFrame({"A": [1, np.nan], "B": [1.0, 2.0]})
        expected = df.replace(np.nan, val)
        result = df.fillna(val)
        tm.assert_frame_equal(result, expected)

    def test_fillna_datetime_columns(self):
        # GH#7095
        df = DataFrame(
            {
                "A": [-1, -2, np.nan],
                "B": date_range("20130101", periods=3),
                "C": ["foo", "bar", None],
                "D": ["foo2", "bar2", None],
            },
            index=date_range("20130110", periods=3),
        )
        result = df.fillna("?")
        expected = DataFrame(
            {
                "A": [-1, -2, "?"],
                "B": date_range("20130101", periods=3),
                "C": ["foo", "bar", "?"],
                "D": ["foo2", "bar2", "?"],
            },
            index=date_range("20130110", periods=3),
        )
        tm.assert_frame_equal(result, expected)

        df = DataFrame(
            {
                "A": [-1, -2, np.nan],
                "B": [Timestamp("2013-01-01"), Timestamp("2013-01-02"), NaT],
                "C": ["foo", "bar", None],
                "D": ["foo2", "bar2", None],
            },
            index=date_range("20130110", periods=3),
        )
        result = df.fillna("?")
        expected = DataFrame(
            {
                "A": [-1, -2, "?"],
                "B": [Timestamp("2013-01-01"), Timestamp("2013-01-02"), "?"],
                "C": ["foo", "bar", "?"],
                "D": ["foo2", "bar2", "?"],
            },
            index=date_range("20130110", periods=3),
        )
        tm.assert_frame_equal(result, expected)

    def test_ffill(self, datetime_frame):
        datetime_frame.loc[datetime_frame.index[:5], "A"] = np.nan
        datetime_frame.loc[datetime_frame.index[-5:], "A"] = np.nan

        msg = "DataFrame.fillna with 'method' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            alt = datetime_frame.fillna(method="ffill")
        tm.assert_frame_equal(datetime_frame.ffill(), alt)

    def test_bfill(self, datetime_frame):
        datetime_frame.loc[datetime_frame.index[:5], "A"] = np.nan
        datetime_frame.loc[datetime_frame.index[-5:], "A"] = np.nan

        msg = "DataFrame.fillna with 'method' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            alt = datetime_frame.fillna(method="bfill")

        tm.assert_frame_equal(datetime_frame.bfill(), alt)

    def test_frame_pad_backfill_limit(self):
        index = np.arange(10)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), index=index)

        result = df[:2].reindex(index, method="pad", limit=5)

        msg = "DataFrame.fillna with 'method' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = df[:2].reindex(index).fillna(method="pad")
        expected.iloc[-3:] = np.nan
        tm.assert_frame_equal(result, expected)

        result = df[-2:].reindex(index, method="backfill", limit=5)

        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = df[-2:].reindex(index).fillna(method="backfill")
        expected.iloc[:3] = np.nan
        tm.assert_frame_equal(result, expected)

    def test_frame_fillna_limit(self):
        index = np.arange(10)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), index=index)

        result = df[:2].reindex(index)
        msg = "DataFrame.fillna with 'method' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = result.fillna(method="pad", limit=5)

        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = df[:2].reindex(index).fillna(method="pad")
        expected.iloc[-3:] = np.nan
        tm.assert_frame_equal(result, expected)

        result = df[-2:].reindex(index)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = result.fillna(method="backfill", limit=5)

        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = df[-2:].reindex(index).fillna(method="backfill")
        expected.iloc[:3] = np.nan
        tm.assert_frame_equal(result, expected)

    def test_fillna_skip_certain_blocks(self):
        # don't try to fill boolean, int blocks

        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)).astype(int))

        # it works!
        df.fillna(np.nan)

    @pytest.mark.parametrize("type", [int, float])
    def test_fillna_positive_limit(self, type):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4))).astype(type)

        msg = "Limit must be greater than 0"
        with pytest.raises(ValueError, match=msg):
            df.fillna(0, limit=-5)

    @pytest.mark.parametrize("type", [int, float])
    def test_fillna_integer_limit(self, type):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4))).astype(type)

        msg = "Limit must be an integer"
        with pytest.raises(ValueError, match=msg):
            df.fillna(0, limit=0.5)

    def test_fillna_inplace(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        df.loc[:4, 1] = np.nan
        df.loc[-4:, 3] = np.nan

        expected = df.fillna(value=0)
        assert expected is not df

        df.fillna(value=0, inplace=True)
        tm.assert_frame_equal(df, expected)

        expected = df.fillna(value={0: 0}, inplace=True)
        assert expected is None

        df.loc[:4, 1] = np.nan
        df.loc[-4:, 3] = np.nan
        msg = "DataFrame.fillna with 'method' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = df.fillna(method="ffill")
        assert expected is not df

        with tm.assert_produces_warning(FutureWarning, match=msg):
            df.fillna(method="ffill", inplace=True)
        tm.assert_frame_equal(df, expected)

    def test_fillna_dict_series(self):
        df = DataFrame(
            {
                "a": [np.nan, 1, 2, np.nan, np.nan],
                "b": [1, 2, 3, np.nan, np.nan],
                "c": [np.nan, 1, 2, 3, 4],
            }
        )

        result = df.fillna({"a": 0, "b": 5})

        expected = df.copy()
        expected["a"] = expected["a"].fillna(0)
        expected["b"] = expected["b"].fillna(5)
        tm.assert_frame_equal(result, expected)

        # it works
        result = df.fillna({"a": 0, "b": 5, "d": 7})

        # Series treated same as dict
        result = df.fillna(df.max())
        expected = df.fillna(df.max().to_dict())
        tm.assert_frame_equal(result, expected)

        # disable this for now
        with pytest.raises(NotImplementedError, match="column by column"):
            df.fillna(df.max(1), axis=1)

    def test_fillna_dataframe(self):
        # GH#8377
        df = DataFrame(
            {
                "a": [np.nan, 1, 2, np.nan, np.nan],
                "b": [1, 2, 3, np.nan, np.nan],
                "c": [np.nan, 1, 2, 3, 4],
            },
            index=list("VWXYZ"),
        )

        # df2 may have different index and columns
        df2 = DataFrame(
            {
                "a": [np.nan, 10, 20, 30, 40],
                "b": [50, 60, 70, 80, 90],
                "foo": ["bar"] * 5,
            },
            index=list("VWXuZ"),
        )

        result = df.fillna(df2)

        # only those columns and indices which are shared get filled
        expected = DataFrame(
            {
                "a": [np.nan, 1, 2, np.nan, 40],
                "b": [1, 2, 3, np.nan, 90],
                "c": [np.nan, 1, 2, 3, 4],
            },
            index=list("VWXYZ"),
        )

        tm.assert_frame_equal(result, expected)

    def test_fillna_columns(self):
        arr = np.random.default_rng(2).standard_normal((10, 10))
        arr[:, ::2] = np.nan
        df = DataFrame(arr)

        msg = "DataFrame.fillna with 'method' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.fillna(method="ffill", axis=1)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = df.T.fillna(method="pad").T
        tm.assert_frame_equal(result, expected)

        df.insert(6, "foo", 5)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.fillna(method="ffill", axis=1)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = df.astype(float).fillna(method="ffill", axis=1)
        tm.assert_frame_equal(result, expected)

    def test_fillna_invalid_method(self, float_frame):
        with pytest.raises(ValueError, match="ffil"):
            float_frame.fillna(method="ffil")

    def test_fillna_invalid_value(self, float_frame):
        # list
        msg = '"value" parameter must be a scalar or dict, but you passed a "{}"'
        with pytest.raises(TypeError, match=msg.format("list")):
            float_frame.fillna([1, 2])
        # tuple
        with pytest.raises(TypeError, match=msg.format("tuple")):
            float_frame.fillna((1, 2))
        # frame with series
        msg = (
            '"value" parameter must be a scalar, dict or Series, but you '
            'passed a "DataFrame"'
        )
        with pytest.raises(TypeError, match=msg):
            float_frame.iloc[:, 0].fillna(float_frame)

    def test_fillna_col_reordering(self):
        cols = ["COL." + str(i) for i in range(5, 0, -1)]
        data = np.random.default_rng(2).random((20, 5))
        df = DataFrame(index=range(20), columns=cols, data=data)
        msg = "DataFrame.fillna with 'method' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            filled = df.fillna(method="ffill")
        assert df.columns.tolist() == filled.columns.tolist()

    def test_fill_empty(self, float_frame):
        df = float_frame.reindex(columns=[])
        result = df.fillna(value=0)
        tm.assert_frame_equal(result, df)

    def test_fillna_downcast_dict(self):
        # GH#40809
        df = DataFrame({"col1": [1, np.nan]})

        msg = "The 'downcast' keyword in fillna"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.fillna({"col1": 2}, downcast={"col1": "int64"})
        expected = DataFrame({"col1": [1, 2]})
        tm.assert_frame_equal(result, expected)

    def test_fillna_with_columns_and_limit(self):
        # GH40989
        df = DataFrame(
            [
                [np.nan, 2, np.nan, 0],
                [3, 4, np.nan, 1],
                [np.nan, np.nan, np.nan, 5],
                [np.nan, 3, np.nan, 4],
            ],
            columns=list("ABCD"),
        )
        result = df.fillna(axis=1, value=100, limit=1)
        result2 = df.fillna(axis=1, value=100, limit=2)

        expected = DataFrame(
            {
                "A": Series([100, 3, 100, 100], dtype="float64"),
                "B": [2, 4, np.nan, 3],
                "C": [np.nan, 100, np.nan, np.nan],
                "D": Series([0, 1, 5, 4], dtype="float64"),
            },
            index=[0, 1, 2, 3],
        )
        expected2 = DataFrame(
            {
                "A": Series([100, 3, 100, 100], dtype="float64"),
                "B": Series([2, 4, 100, 3], dtype="float64"),
                "C": [100, 100, np.nan, 100],
                "D": Series([0, 1, 5, 4], dtype="float64"),
            },
            index=[0, 1, 2, 3],
        )

        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected2)

    def test_fillna_datetime_inplace(self):
        # GH#48863
        df = DataFrame(
            {
                "date1": to_datetime(["2018-05-30", None]),
                "date2": to_datetime(["2018-09-30", None]),
            }
        )
        expected = df.copy()
        df.fillna(np.nan, inplace=True)
        tm.assert_frame_equal(df, expected)

    def test_fillna_inplace_with_columns_limit_and_value(self):
        # GH40989
        df = DataFrame(
            [
                [np.nan, 2, np.nan, 0],
                [3, 4, np.nan, 1],
                [np.nan, np.nan, np.nan, 5],
                [np.nan, 3, np.nan, 4],
            ],
            columns=list("ABCD"),
        )

        expected = df.fillna(axis=1, value=100, limit=1)
        assert expected is not df

        df.fillna(axis=1, value=100, limit=1, inplace=True)
        tm.assert_frame_equal(df, expected)

    @td.skip_array_manager_invalid_test
    @pytest.mark.parametrize("val", [-1, {"x": -1, "y": -1}])
    def test_inplace_dict_update_view(
        self, val, using_copy_on_write, warn_copy_on_write
    ):
        # GH#47188
        df = DataFrame({"x": [np.nan, 2], "y": [np.nan, 2]})
        df_orig = df.copy()
        result_view = df[:]
        with tm.assert_cow_warning(warn_copy_on_write):
            df.fillna(val, inplace=True)
        expected = DataFrame({"x": [-1, 2.0], "y": [-1.0, 2]})
        tm.assert_frame_equal(df, expected)
        if using_copy_on_write:
            tm.assert_frame_equal(result_view, df_orig)
        else:
            tm.assert_frame_equal(result_view, expected)

    def test_single_block_df_with_horizontal_axis(self):
        # GH 47713
        df = DataFrame(
            {
                "col1": [5, 0, np.nan, 10, np.nan],
                "col2": [7, np.nan, np.nan, 5, 3],
                "col3": [12, np.nan, 1, 2, 0],
                "col4": [np.nan, 1, 1, np.nan, 18],
            }
        )
        result = df.fillna(50, limit=1, axis=1)
        expected = DataFrame(
            [
                [5.0, 7.0, 12.0, 50.0],
                [0.0, 50.0, np.nan, 1.0],
                [50.0, np.nan, 1.0, 1.0],
                [10.0, 5.0, 2.0, 50.0],
                [50.0, 3.0, 0.0, 18.0],
            ],
            columns=["col1", "col2", "col3", "col4"],
        )
        tm.assert_frame_equal(result, expected)

    def test_fillna_with_multi_index_frame(self):
        # GH 47649
        pdf = DataFrame(
            {
                ("x", "a"): [np.nan, 2.0, 3.0],
                ("x", "b"): [1.0, 2.0, np.nan],
                ("y", "c"): [1.0, 2.0, np.nan],
            }
        )
        expected = DataFrame(
            {
                ("x", "a"): [-1.0, 2.0, 3.0],
                ("x", "b"): [1.0, 2.0, -1.0],
                ("y", "c"): [1.0, 2.0, np.nan],
            }
        )
        tm.assert_frame_equal(pdf.fillna({"x": -1}), expected)
        tm.assert_frame_equal(pdf.fillna({"x": -1, ("x", "b"): -2}), expected)

        expected = DataFrame(
            {
                ("x", "a"): [-1.0, 2.0, 3.0],
                ("x", "b"): [1.0, 2.0, -2.0],
                ("y", "c"): [1.0, 2.0, np.nan],
            }
        )
        tm.assert_frame_equal(pdf.fillna({("x", "b"): -2, "x": -1}), expected)


def test_fillna_nonconsolidated_frame():
    # https://github.com/pandas-dev/pandas/issues/36495
    df = DataFrame(
        [
            [1, 1, 1, 1.0],
            [2, 2, 2, 2.0],
            [3, 3, 3, 3.0],
        ],
        columns=["i1", "i2", "i3", "f1"],
    )
    df_nonconsol = df.pivot(index="i1", columns="i2")
    result = df_nonconsol.fillna(0)
    assert result.isna().sum().sum() == 0


def test_fillna_nones_inplace():
    # GH 48480
    df = DataFrame(
        [[None, None], [None, None]],
        columns=["A", "B"],
    )
    msg = "Downcasting object dtype arrays"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.fillna(value={"A": 1, "B": 2}, inplace=True)

    expected = DataFrame([[1, 2], [1, 2]], columns=["A", "B"])
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize("func", ["pad", "backfill"])
def test_pad_backfill_deprecated(func):
    # GH#33396
    df = DataFrame({"a": [1, 2, 3]})
    with tm.assert_produces_warning(FutureWarning):
        getattr(df, func)()


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
    df = DataFrame(data)
    expected = DataFrame(expected_data)
    result = getattr(df, method)(**kwargs)
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_array_coercion.py ===
"""
Tests for array coercion, mainly through testing `np.array` results directly.
Note that other such tests exist, e.g., in `test_api.py` and many corner-cases
are tested (sometimes indirectly) elsewhere.
"""

from itertools import permutations, product

import numpy._core._multiarray_umath as ncu
import pytest
from numpy._core._rational_tests import rational
from pytest import param

import numpy as np
from numpy.testing import IS_64BIT, IS_PYPY, assert_array_equal


def arraylikes():
    """
    Generator for functions converting an array into various array-likes.
    If full is True (default) it includes array-likes not capable of handling
    all dtypes.
    """
    # base array:
    def ndarray(a):
        return a

    yield param(ndarray, id="ndarray")

    # subclass:
    class MyArr(np.ndarray):
        pass

    def subclass(a):
        return a.view(MyArr)

    yield subclass

    class _SequenceLike:
        # Older NumPy versions, sometimes cared whether a protocol array was
        # also _SequenceLike.  This shouldn't matter, but keep it for now
        # for __array__ and not the others.
        def __len__(self):
            raise TypeError

        def __getitem__(self, _, /):
            raise TypeError

    # Array-interface
    class ArrayDunder(_SequenceLike):
        def __init__(self, a):
            self.a = a

        def __array__(self, dtype=None, copy=None):
            if dtype is None:
                return self.a
            return self.a.astype(dtype)

    yield param(ArrayDunder, id="__array__")

    # memory-view
    yield param(memoryview, id="memoryview")

    # Array-interface
    class ArrayInterface:
        def __init__(self, a):
            self.a = a  # need to hold on to keep interface valid
            self.__array_interface__ = a.__array_interface__

    yield param(ArrayInterface, id="__array_interface__")

    # Array-Struct
    class ArrayStruct:
        def __init__(self, a):
            self.a = a  # need to hold on to keep struct valid
            self.__array_struct__ = a.__array_struct__

    yield param(ArrayStruct, id="__array_struct__")


def scalar_instances(times=True, extended_precision=True, user_dtype=True):
    # Hard-coded list of scalar instances.
    # Floats:
    yield param(np.sqrt(np.float16(5)), id="float16")
    yield param(np.sqrt(np.float32(5)), id="float32")
    yield param(np.sqrt(np.float64(5)), id="float64")
    if extended_precision:
        yield param(np.sqrt(np.longdouble(5)), id="longdouble")

    # Complex:
    yield param(np.sqrt(np.complex64(2 + 3j)), id="complex64")
    yield param(np.sqrt(np.complex128(2 + 3j)), id="complex128")
    if extended_precision:
        yield param(np.sqrt(np.clongdouble(2 + 3j)), id="clongdouble")

    # Bool:
    # XFAIL: Bool should be added, but has some bad properties when it
    # comes to strings, see also gh-9875
    # yield param(np.bool(0), id="bool")

    # Integers:
    yield param(np.int8(2), id="int8")
    yield param(np.int16(2), id="int16")
    yield param(np.int32(2), id="int32")
    yield param(np.int64(2), id="int64")

    yield param(np.uint8(2), id="uint8")
    yield param(np.uint16(2), id="uint16")
    yield param(np.uint32(2), id="uint32")
    yield param(np.uint64(2), id="uint64")

    # Rational:
    if user_dtype:
        yield param(rational(1, 2), id="rational")

    # Cannot create a structured void scalar directly:
    structured = np.array([(1, 3)], "i,i")[0]
    assert isinstance(structured, np.void)
    assert structured.dtype == np.dtype("i,i")
    yield param(structured, id="structured")

    if times:
        # Datetimes and timedelta
        yield param(np.timedelta64(2), id="timedelta64[generic]")
        yield param(np.timedelta64(23, "s"), id="timedelta64[s]")
        yield param(np.timedelta64("NaT", "s"), id="timedelta64[s](NaT)")

        yield param(np.datetime64("NaT"), id="datetime64[generic](NaT)")
        yield param(np.datetime64("2020-06-07 12:43", "ms"), id="datetime64[ms]")

    # Strings and unstructured void:
    yield param(np.bytes_(b"1234"), id="bytes")
    yield param(np.str_("2345"), id="unicode")
    yield param(np.void(b"4321"), id="unstructured_void")


def is_parametric_dtype(dtype):
    """Returns True if the dtype is a parametric legacy dtype (itemsize
    is 0, or a datetime without units)
    """
    if dtype.itemsize == 0:
        return True
    if issubclass(dtype.type, (np.datetime64, np.timedelta64)):
        if dtype.name.endswith("64"):
            # Generic time units
            return True
    return False


class TestStringDiscovery:
    @pytest.mark.parametrize("obj",
            [object(), 1.2, 10**43, None, "string"],
            ids=["object", "1.2", "10**43", "None", "string"])
    def test_basic_stringlength(self, obj):
        length = len(str(obj))
        expected = np.dtype(f"S{length}")

        assert np.array(obj, dtype="S").dtype == expected
        assert np.array([obj], dtype="S").dtype == expected

        # A nested array is also discovered correctly
        arr = np.array(obj, dtype="O")
        assert np.array(arr, dtype="S").dtype == expected
        # Also if we use the dtype class
        assert np.array(arr, dtype=type(expected)).dtype == expected
        # Check that .astype() behaves identical
        assert arr.astype("S").dtype == expected
        # The DType class is accepted by `.astype()`
        assert arr.astype(type(np.dtype("S"))).dtype == expected

    @pytest.mark.parametrize("obj",
            [object(), 1.2, 10**43, None, "string"],
            ids=["object", "1.2", "10**43", "None", "string"])
    def test_nested_arrays_stringlength(self, obj):
        length = len(str(obj))
        expected = np.dtype(f"S{length}")
        arr = np.array(obj, dtype="O")
        assert np.array([arr, arr], dtype="S").dtype == expected

    @pytest.mark.parametrize("arraylike", arraylikes())
    def test_unpack_first_level(self, arraylike):
        # We unpack exactly one level of array likes
        obj = np.array([None])
        obj[0] = np.array(1.2)
        # the length of the included item, not of the float dtype
        length = len(str(obj[0]))
        expected = np.dtype(f"S{length}")

        obj = arraylike(obj)
        # casting to string usually calls str(obj)
        arr = np.array([obj], dtype="S")
        assert arr.shape == (1, 1)
        assert arr.dtype == expected


class TestScalarDiscovery:
    def test_void_special_case(self):
        # Void dtypes with structures discover tuples as elements
        arr = np.array((1, 2, 3), dtype="i,i,i")
        assert arr.shape == ()
        arr = np.array([(1, 2, 3)], dtype="i,i,i")
        assert arr.shape == (1,)

    def test_char_special_case(self):
        arr = np.array("string", dtype="c")
        assert arr.shape == (6,)
        assert arr.dtype.char == "c"
        arr = np.array(["string"], dtype="c")
        assert arr.shape == (1, 6)
        assert arr.dtype.char == "c"

    def test_char_special_case_deep(self):
        # Check that the character special case errors correctly if the
        # array is too deep:
        nested = ["string"]  # 2 dimensions (due to string being sequence)
        for i in range(ncu.MAXDIMS - 2):
            nested = [nested]

        arr = np.array(nested, dtype='c')
        assert arr.shape == (1,) * (ncu.MAXDIMS - 1) + (6,)
        with pytest.raises(ValueError):
            np.array([nested], dtype="c")

    def test_unknown_object(self):
        arr = np.array(object())
        assert arr.shape == ()
        assert arr.dtype == np.dtype("O")

    @pytest.mark.parametrize("scalar", scalar_instances())
    def test_scalar(self, scalar):
        arr = np.array(scalar)
        assert arr.shape == ()
        assert arr.dtype == scalar.dtype

        arr = np.array([[scalar, scalar]])
        assert arr.shape == (1, 2)
        assert arr.dtype == scalar.dtype

    # Additionally to string this test also runs into a corner case
    # with datetime promotion (the difference is the promotion order).
    @pytest.mark.filterwarnings("ignore:Promotion of numbers:FutureWarning")
    def test_scalar_promotion(self):
        for sc1, sc2 in product(scalar_instances(), scalar_instances()):
            sc1, sc2 = sc1.values[0], sc2.values[0]
            # test all combinations:
            try:
                arr = np.array([sc1, sc2])
            except (TypeError, ValueError):
                # The promotion between two times can fail
                # XFAIL (ValueError): Some object casts are currently undefined
                continue
            assert arr.shape == (2,)
            try:
                dt1, dt2 = sc1.dtype, sc2.dtype
                expected_dtype = np.promote_types(dt1, dt2)
                assert arr.dtype == expected_dtype
            except TypeError as e:
                # Will currently always go to object dtype
                assert arr.dtype == np.dtype("O")

    @pytest.mark.parametrize("scalar", scalar_instances())
    def test_scalar_coercion(self, scalar):
        # This tests various scalar coercion paths, mainly for the numerical
        # types. It includes some paths not directly related to `np.array`.
        if isinstance(scalar, np.inexact):
            # Ensure we have a full-precision number if available
            scalar = type(scalar)((scalar * 2)**0.5)

        if type(scalar) is rational:
            # Rational generally fails due to a missing cast. In the future
            # object casts should automatically be defined based on `setitem`.
            pytest.xfail("Rational to object cast is undefined currently.")

        # Use casting from object:
        arr = np.array(scalar, dtype=object).astype(scalar.dtype)

        # Test various ways to create an array containing this scalar:
        arr1 = np.array(scalar).reshape(1)
        arr2 = np.array([scalar])
        arr3 = np.empty(1, dtype=scalar.dtype)
        arr3[0] = scalar
        arr4 = np.empty(1, dtype=scalar.dtype)
        arr4[:] = [scalar]
        # All of these methods should yield the same results
        assert_array_equal(arr, arr1)
        assert_array_equal(arr, arr2)
        assert_array_equal(arr, arr3)
        assert_array_equal(arr, arr4)

    @pytest.mark.xfail(IS_PYPY, reason="`int(np.complex128(3))` fails on PyPy")
    @pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning")
    @pytest.mark.parametrize("cast_to", scalar_instances())
    def test_scalar_coercion_same_as_cast_and_assignment(self, cast_to):
        """
        Test that in most cases:
           * `np.array(scalar, dtype=dtype)`
           * `np.empty((), dtype=dtype)[()] = scalar`
           * `np.array(scalar).astype(dtype)`
        should behave the same.  The only exceptions are parametric dtypes
        (mainly datetime/timedelta without unit) and void without fields.
        """
        dtype = cast_to.dtype  # use to parametrize only the target dtype

        for scalar in scalar_instances(times=False):
            scalar = scalar.values[0]

            if dtype.type == np.void:
                if scalar.dtype.fields is not None and dtype.fields is None:
                    # Here, coercion to "V6" works, but the cast fails.
                    # Since the types are identical, SETITEM takes care of
                    # this, but has different rules than the cast.
                    with pytest.raises(TypeError):
                        np.array(scalar).astype(dtype)
                    np.array(scalar, dtype=dtype)
                    np.array([scalar], dtype=dtype)
                    continue

            # The main test, we first try to use casting and if it succeeds
            # continue below testing that things are the same, otherwise
            # test that the alternative paths at least also fail.
            try:
                cast = np.array(scalar).astype(dtype)
            except (TypeError, ValueError, RuntimeError):
                # coercion should also raise (error type may change)
                with pytest.raises(Exception):  # noqa: B017
                    np.array(scalar, dtype=dtype)

                if (isinstance(scalar, rational) and
                        np.issubdtype(dtype, np.signedinteger)):
                    return

                with pytest.raises(Exception):  # noqa: B017
                    np.array([scalar], dtype=dtype)
                # assignment should also raise
                res = np.zeros((), dtype=dtype)
                with pytest.raises(Exception):  # noqa: B017
                    res[()] = scalar

                return

            # Non error path:
            arr = np.array(scalar, dtype=dtype)
            assert_array_equal(arr, cast)
            # assignment behaves the same
            ass = np.zeros((), dtype=dtype)
            ass[()] = scalar
            assert_array_equal(ass, cast)

    @pytest.mark.parametrize("pyscalar", [10, 10.32, 10.14j, 10**100])
    def test_pyscalar_subclasses(self, pyscalar):
        """NumPy arrays are read/write which means that anything but invariant
        behaviour is on thin ice.  However, we currently are happy to discover
        subclasses of Python float, int, complex the same as the base classes.
        This should potentially be deprecated.
        """
        class MyScalar(type(pyscalar)):
            pass

        res = np.array(MyScalar(pyscalar))
        expected = np.array(pyscalar)
        assert_array_equal(res, expected)

    @pytest.mark.parametrize("dtype_char", np.typecodes["All"])
    def test_default_dtype_instance(self, dtype_char):
        if dtype_char in "SU":
            dtype = np.dtype(dtype_char + "1")
        elif dtype_char == "V":
            # Legacy behaviour was to use V8. The reason was float64 being the
            # default dtype and that having 8 bytes.
            dtype = np.dtype("V8")
        else:
            dtype = np.dtype(dtype_char)

        discovered_dtype, _ = ncu._discover_array_parameters([], type(dtype))

        assert discovered_dtype == dtype
        assert discovered_dtype.itemsize == dtype.itemsize

    @pytest.mark.parametrize("dtype", np.typecodes["Integer"])
    @pytest.mark.parametrize(["scalar", "error"],
            [(np.float64(np.nan), ValueError),
             (np.array(-1).astype(np.ulonglong)[()], OverflowError)])
    def test_scalar_to_int_coerce_does_not_cast(self, dtype, scalar, error):
        """
        Signed integers are currently different in that they do not cast other
        NumPy scalar, but instead use scalar.__int__(). The hardcoded
        exception to this rule is `np.array(scalar, dtype=integer)`.
        """
        dtype = np.dtype(dtype)

        # This is a special case using casting logic. It warns for the NaN
        # but allows the cast (giving undefined behaviour).
        with np.errstate(invalid="ignore"):
            coerced = np.array(scalar, dtype=dtype)
            cast = np.array(scalar).astype(dtype)
        assert_array_equal(coerced, cast)

        # However these fail:
        with pytest.raises(error):
            np.array([scalar], dtype=dtype)
        with pytest.raises(error):
            cast[()] = scalar


class TestTimeScalars:
    @pytest.mark.parametrize("dtype", [np.int64, np.float32])
    @pytest.mark.parametrize("scalar",
            [param(np.timedelta64("NaT", "s"), id="timedelta64[s](NaT)"),
             param(np.timedelta64(123, "s"), id="timedelta64[s]"),
             param(np.datetime64("NaT", "generic"), id="datetime64[generic](NaT)"),
             param(np.datetime64(1, "D"), id="datetime64[D]")],)
    def test_coercion_basic(self, dtype, scalar):
        # Note the `[scalar]` is there because np.array(scalar) uses stricter
        # `scalar.__int__()` rules for backward compatibility right now.
        arr = np.array(scalar, dtype=dtype)
        cast = np.array(scalar).astype(dtype)
        assert_array_equal(arr, cast)

        ass = np.ones((), dtype=dtype)
        if issubclass(dtype, np.integer):
            with pytest.raises(TypeError):
                # raises, as would np.array([scalar], dtype=dtype), this is
                # conversion from times, but behaviour of integers.
                ass[()] = scalar
        else:
            ass[()] = scalar
            assert_array_equal(ass, cast)

    @pytest.mark.parametrize("dtype", [np.int64, np.float32])
    @pytest.mark.parametrize("scalar",
            [param(np.timedelta64(123, "ns"), id="timedelta64[ns]"),
             param(np.timedelta64(12, "generic"), id="timedelta64[generic]")])
    def test_coercion_timedelta_convert_to_number(self, dtype, scalar):
        # Only "ns" and "generic" timedeltas can be converted to numbers
        # so these are slightly special.
        arr = np.array(scalar, dtype=dtype)
        cast = np.array(scalar).astype(dtype)
        ass = np.ones((), dtype=dtype)
        ass[()] = scalar  # raises, as would np.array([scalar], dtype=dtype)

        assert_array_equal(arr, cast)
        assert_array_equal(cast, cast)

    @pytest.mark.parametrize("dtype", ["S6", "U6"])
    @pytest.mark.parametrize(["val", "unit"],
            [param(123, "s", id="[s]"), param(123, "D", id="[D]")])
    def test_coercion_assignment_datetime(self, val, unit, dtype):
        # String from datetime64 assignment is currently special cased to
        # never use casting.  This is because casting will error in this
        # case, and traditionally in most cases the behaviour is maintained
        # like this.  (`np.array(scalar, dtype="U6")` would have failed before)
        # TODO: This discrepancy _should_ be resolved, either by relaxing the
        #       cast, or by deprecating the first part.
        scalar = np.datetime64(val, unit)
        dtype = np.dtype(dtype)
        cut_string = dtype.type(str(scalar)[:6])

        arr = np.array(scalar, dtype=dtype)
        assert arr[()] == cut_string
        ass = np.ones((), dtype=dtype)
        ass[()] = scalar
        assert ass[()] == cut_string

        with pytest.raises(RuntimeError):
            # However, unlike the above assignment using `str(scalar)[:6]`
            # due to being handled by the string DType and not be casting
            # the explicit cast fails:
            np.array(scalar).astype(dtype)

    @pytest.mark.parametrize(["val", "unit"],
            [param(123, "s", id="[s]"), param(123, "D", id="[D]")])
    def test_coercion_assignment_timedelta(self, val, unit):
        scalar = np.timedelta64(val, unit)

        # Unlike datetime64, timedelta allows the unsafe cast:
        np.array(scalar, dtype="S6")
        cast = np.array(scalar).astype("S6")
        ass = np.ones((), dtype="S6")
        ass[()] = scalar
        expected = scalar.astype("S")[:6]
        assert cast[()] == expected
        assert ass[()] == expected

class TestNested:
    def test_nested_simple(self):
        initial = [1.2]
        nested = initial
        for i in range(ncu.MAXDIMS - 1):
            nested = [nested]

        arr = np.array(nested, dtype="float64")
        assert arr.shape == (1,) * ncu.MAXDIMS
        with pytest.raises(ValueError):
            np.array([nested], dtype="float64")

        with pytest.raises(ValueError, match=".*would exceed the maximum"):
            np.array([nested])  # user must ask for `object` explicitly

        arr = np.array([nested], dtype=object)
        assert arr.dtype == np.dtype("O")
        assert arr.shape == (1,) * ncu.MAXDIMS
        assert arr.item() is initial

    def test_pathological_self_containing(self):
        # Test that this also works for two nested sequences
        l = []
        l.append(l)
        arr = np.array([l, l, l], dtype=object)
        assert arr.shape == (3,) + (1,) * (ncu.MAXDIMS - 1)

        # Also check a ragged case:
        arr = np.array([l, [None], l], dtype=object)
        assert arr.shape == (3, 1)

    @pytest.mark.parametrize("arraylike", arraylikes())
    def test_nested_arraylikes(self, arraylike):
        # We try storing an array like into an array, but the array-like
        # will have too many dimensions.  This means the shape discovery
        # decides that the array-like must be treated as an object (a special
        # case of ragged discovery).  The result will be an array with one
        # dimension less than the maximum dimensions, and the array being
        # assigned to it (which does work for object or if `float(arraylike)`
        # works).
        initial = arraylike(np.ones((1, 1)))

        nested = initial
        for i in range(ncu.MAXDIMS - 1):
            nested = [nested]

        with pytest.raises(ValueError, match=".*would exceed the maximum"):
            # It will refuse to assign the array into
            np.array(nested, dtype="float64")

        # If this is object, we end up assigning a (1, 1) array into (1,)
        # (due to running out of dimensions), this is currently supported but
        # a special case which is not ideal.
        arr = np.array(nested, dtype=object)
        assert arr.shape == (1,) * ncu.MAXDIMS
        assert arr.item() == np.array(initial).item()

    @pytest.mark.parametrize("arraylike", arraylikes())
    def test_uneven_depth_ragged(self, arraylike):
        arr = np.arange(4).reshape((2, 2))
        arr = arraylike(arr)

        # Array is ragged in the second dimension already:
        out = np.array([arr, [arr]], dtype=object)
        assert out.shape == (2,)
        assert out[0] is arr
        assert type(out[1]) is list

        # Array is ragged in the third dimension:
        with pytest.raises(ValueError):
            # This is a broadcast error during assignment, because
            # the array shape would be (2, 2, 2) but `arr[0, 0] = arr` fails.
            np.array([arr, [arr, arr]], dtype=object)

    def test_empty_sequence(self):
        arr = np.array([[], [1], [[1]]], dtype=object)
        assert arr.shape == (3,)

        # The empty sequence stops further dimension discovery, so the
        # result shape will be (0,) which leads to an error during:
        with pytest.raises(ValueError):
            np.array([[], np.empty((0, 1))], dtype=object)

    def test_array_of_different_depths(self):
        # When multiple arrays (or array-likes) are included in a
        # sequences and have different depth, we currently discover
        # as many dimensions as they share. (see also gh-17224)
        arr = np.zeros((3, 2))
        mismatch_first_dim = np.zeros((1, 2))
        mismatch_second_dim = np.zeros((3, 3))

        dtype, shape = ncu._discover_array_parameters(
            [arr, mismatch_second_dim], dtype=np.dtype("O"))
        assert shape == (2, 3)

        dtype, shape = ncu._discover_array_parameters(
            [arr, mismatch_first_dim], dtype=np.dtype("O"))
        assert shape == (2,)
        # The second case is currently supported because the arrays
        # can be stored as objects:
        res = np.asarray([arr, mismatch_first_dim], dtype=np.dtype("O"))
        assert res[0] is arr
        assert res[1] is mismatch_first_dim


class TestBadSequences:
    # These are tests for bad objects passed into `np.array`, in general
    # these have undefined behaviour.  In the old code they partially worked
    # when now they will fail.  We could (and maybe should) create a copy
    # of all sequences to be safe against bad-actors.

    def test_growing_list(self):
        # List to coerce, `mylist` will append to it during coercion
        obj = []

        class mylist(list):
            def __len__(self):
                obj.append([1, 2])
                return super().__len__()

        obj.append(mylist([1, 2]))

        with pytest.raises(RuntimeError):
            np.array(obj)

    # Note: We do not test a shrinking list.  These do very evil things
    #       and the only way to fix them would be to copy all sequences.
    #       (which may be a real option in the future).

    def test_mutated_list(self):
        # List to coerce, `mylist` will mutate the first element
        obj = []

        class mylist(list):
            def __len__(self):
                obj[0] = [2, 3]  # replace with a different list.
                return super().__len__()

        obj.append([2, 3])
        obj.append(mylist([1, 2]))
        # Does not crash:
        np.array(obj)

    def test_replace_0d_array(self):
        # List to coerce, `mylist` will mutate the first element
        obj = []

        class baditem:
            def __len__(self):
                obj[0][0] = 2  # replace with a different list.
                raise ValueError("not actually a sequence!")

            def __getitem__(self, _, /):
                pass

        # Runs into a corner case in the new code, the `array(2)` is cached
        # so replacing it invalidates the cache.
        obj.append([np.array(2), baditem()])
        with pytest.raises(RuntimeError):
            np.array(obj)


class TestArrayLikes:
    @pytest.mark.parametrize("arraylike", arraylikes())
    def test_0d_object_special_case(self, arraylike):
        arr = np.array(0.)
        obj = arraylike(arr)
        # A single array-like is always converted:
        res = np.array(obj, dtype=object)
        assert_array_equal(arr, res)

        # But a single 0-D nested array-like never:
        res = np.array([obj], dtype=object)
        assert res[0] is obj

    @pytest.mark.parametrize("arraylike", arraylikes())
    @pytest.mark.parametrize("arr", [np.array(0.), np.arange(4)])
    def test_object_assignment_special_case(self, arraylike, arr):
        obj = arraylike(arr)
        empty = np.arange(1, dtype=object)
        empty[:] = [obj]
        assert empty[0] is obj

    def test_0d_generic_special_case(self):
        class ArraySubclass(np.ndarray):
            def __float__(self):
                raise TypeError("e.g. quantities raise on this")

        arr = np.array(0.)
        obj = arr.view(ArraySubclass)
        res = np.array(obj)
        # The subclass is simply cast:
        assert_array_equal(arr, res)

        # If the 0-D array-like is included, __float__ is currently
        # guaranteed to be used.  We may want to change that, quantities
        # and masked arrays half make use of this.
        with pytest.raises(TypeError):
            np.array([obj])

        # The same holds for memoryview:
        obj = memoryview(arr)
        res = np.array(obj)
        assert_array_equal(arr, res)
        with pytest.raises(ValueError):
            # The error type does not matter much here.
            np.array([obj])

    def test_arraylike_classes(self):
        # The classes of array-likes should generally be acceptable to be
        # stored inside a numpy (object) array.  This tests all of the
        # special attributes (since all are checked during coercion).
        arr = np.array(np.int64)
        assert arr[()] is np.int64
        arr = np.array([np.int64])
        assert arr[0] is np.int64

        # This also works for properties/unbound methods:
        class ArrayLike:
            @property
            def __array_interface__(self):
                pass

            @property
            def __array_struct__(self):
                pass

            def __array__(self, dtype=None, copy=None):
                pass

        arr = np.array(ArrayLike)
        assert arr[()] is ArrayLike
        arr = np.array([ArrayLike])
        assert arr[0] is ArrayLike

    @pytest.mark.skipif(not IS_64BIT, reason="Needs 64bit platform")
    def test_too_large_array_error_paths(self):
        """Test the error paths, including for memory leaks"""
        arr = np.array(0, dtype="uint8")
        # Guarantees that a contiguous copy won't work:
        arr = np.broadcast_to(arr, 2**62)

        for i in range(5):
            # repeat, to ensure caching cannot have an effect:
            with pytest.raises(MemoryError):
                np.array(arr)
            with pytest.raises(MemoryError):
                np.array([arr])

    @pytest.mark.parametrize("attribute",
        ["__array_interface__", "__array__", "__array_struct__"])
    @pytest.mark.parametrize("error", [RecursionError, MemoryError])
    def test_bad_array_like_attributes(self, attribute, error):
        # RecursionError and MemoryError are considered fatal. All errors
        # (except AttributeError) should probably be raised in the future,
        # but shapely made use of it, so it will require a deprecation.

        class BadInterface:
            def __getattr__(self, attr):
                if attr == attribute:
                    raise error
                super().__getattr__(attr)

        with pytest.raises(error):
            np.array(BadInterface())

    @pytest.mark.parametrize("error", [RecursionError, MemoryError])
    def test_bad_array_like_bad_length(self, error):
        # RecursionError and MemoryError are considered "critical" in
        # sequences. We could expand this more generally though. (NumPy 1.20)
        class BadSequence:
            def __len__(self):
                raise error

            def __getitem__(self, _, /):
                # must have getitem to be a Sequence
                return 1

        with pytest.raises(error):
            np.array(BadSequence())

    def test_array_interface_descr_optional(self):
        # The descr should be optional regression test for gh-27249
        arr = np.ones(10, dtype="V10")
        iface = arr.__array_interface__
        iface.pop("descr")

        class MyClass:
            __array_interface__ = iface

        assert_array_equal(np.asarray(MyClass), arr)


class TestAsArray:
    """Test expected behaviors of ``asarray``."""

    def test_dtype_identity(self):
        """Confirm the intended behavior for *dtype* kwarg.

        The result of ``asarray()`` should have the dtype provided through the
        keyword argument, when used. This forces unique array handles to be
        produced for unique np.dtype objects, but (for equivalent dtypes), the
        underlying data (the base object) is shared with the original array
        object.

        Ref https://github.com/numpy/numpy/issues/1468
        """
        int_array = np.array([1, 2, 3], dtype='i')
        assert np.asarray(int_array) is int_array

        # The character code resolves to the singleton dtype object provided
        # by the numpy package.
        assert np.asarray(int_array, dtype='i') is int_array

        # Derive a dtype from n.dtype('i'), but add a metadata object to force
        # the dtype to be distinct.
        unequal_type = np.dtype('i', metadata={'spam': True})
        annotated_int_array = np.asarray(int_array, dtype=unequal_type)
        assert annotated_int_array is not int_array
        assert annotated_int_array.base is int_array
        # Create an equivalent descriptor with a new and distinct dtype
        # instance.
        equivalent_requirement = np.dtype('i', metadata={'spam': True})
        annotated_int_array_alt = np.asarray(annotated_int_array,
                                             dtype=equivalent_requirement)
        assert unequal_type == equivalent_requirement
        assert unequal_type is not equivalent_requirement
        assert annotated_int_array_alt is not annotated_int_array
        assert annotated_int_array_alt.dtype is equivalent_requirement

        # Check the same logic for a pair of C types whose equivalence may vary
        # between computing environments.
        # Find an equivalent pair.
        integer_type_codes = ('i', 'l', 'q')
        integer_dtypes = [np.dtype(code) for code in integer_type_codes]
        typeA = None
        typeB = None
        for typeA, typeB in permutations(integer_dtypes, r=2):
            if typeA == typeB:
                assert typeA is not typeB
                break
        assert isinstance(typeA, np.dtype) and isinstance(typeB, np.dtype)

        # These ``asarray()`` calls may produce a new view or a copy,
        # but never the same object.
        long_int_array = np.asarray(int_array, dtype='l')
        long_long_int_array = np.asarray(int_array, dtype='q')
        assert long_int_array is not int_array
        assert long_long_int_array is not int_array
        assert np.asarray(long_int_array, dtype='q') is not long_int_array
        array_a = np.asarray(int_array, dtype=typeA)
        assert typeA == typeB
        assert typeA is not typeB
        assert array_a.dtype is typeA
        assert array_a is not np.asarray(array_a, dtype=typeB)
        assert np.asarray(array_a, dtype=typeB).dtype is typeB
        assert array_a is np.asarray(array_a, dtype=typeB).base


class TestSpecialAttributeLookupFailure:
    # An exception was raised while fetching the attribute

    class WeirdArrayLike:
        @property
        def __array__(self, dtype=None, copy=None):  # noqa: PLR0206
            raise RuntimeError("oops!")

    class WeirdArrayInterface:
        @property
        def __array_interface__(self):
            raise RuntimeError("oops!")

    def test_deprecated(self):
        with pytest.raises(RuntimeError):
            np.array(self.WeirdArrayLike())
        with pytest.raises(RuntimeError):
            np.array(self.WeirdArrayInterface())


def test_subarray_from_array_construction():
    # Arrays are more complex, since they "broadcast" on success:
    arr = np.array([1, 2])

    res = arr.astype("2i")
    assert_array_equal(res, [[1, 1], [2, 2]])

    res = np.array(arr, dtype="(2,)i")

    assert_array_equal(res, [[1, 1], [2, 2]])

    res = np.array([[(1,), (2,)], arr], dtype="2i")
    assert_array_equal(res, [[[1, 1], [2, 2]], [[1, 1], [2, 2]]])

    # Also try a multi-dimensional example:
    arr = np.arange(5 * 2).reshape(5, 2)
    expected = np.broadcast_to(arr[:, :, np.newaxis, np.newaxis], (5, 2, 2, 2))

    res = arr.astype("(2,2)f")
    assert_array_equal(res, expected)

    res = np.array(arr, dtype="(2,2)f")
    assert_array_equal(res, expected)


def test_empty_string():
    # Empty strings are unfortunately often converted to S1 and we need to
    # make sure we are filling the S1 and not the (possibly) detected S0
    # result.  This should likely just return S0 and if not maybe the decision
    # to return S1 should be moved.
    res = np.array([""] * 10, dtype="S")
    assert_array_equal(res, np.array("\0", "S1"))
    assert res.dtype == "S1"

    arr = np.array([""] * 10, dtype=object)

    res = arr.astype("S")
    assert_array_equal(res, b"")
    assert res.dtype == "S1"

    res = np.array(arr, dtype="S")
    assert_array_equal(res, b"")
    # TODO: This is arguably weird/wrong, but seems old:
    assert res.dtype == f"S{np.dtype('O').itemsize}"

    res = np.array([[""] * 10, arr], dtype="S")
    assert_array_equal(res, b"")
    assert res.shape == (2, 10)
    assert res.dtype == "S1"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\json\test_normalize.py ===
import json

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
    json_normalize,
)
import pandas._testing as tm

from pandas.io.json._normalize import nested_to_record


@pytest.fixture
def deep_nested():
    # deeply nested data
    return [
        {
            "country": "USA",
            "states": [
                {
                    "name": "California",
                    "cities": [
                        {"name": "San Francisco", "pop": 12345},
                        {"name": "Los Angeles", "pop": 12346},
                    ],
                },
                {
                    "name": "Ohio",
                    "cities": [
                        {"name": "Columbus", "pop": 1234},
                        {"name": "Cleveland", "pop": 1236},
                    ],
                },
            ],
        },
        {
            "country": "Germany",
            "states": [
                {"name": "Bayern", "cities": [{"name": "Munich", "pop": 12347}]},
                {
                    "name": "Nordrhein-Westfalen",
                    "cities": [
                        {"name": "Duesseldorf", "pop": 1238},
                        {"name": "Koeln", "pop": 1239},
                    ],
                },
            ],
        },
    ]


@pytest.fixture
def state_data():
    return [
        {
            "counties": [
                {"name": "Dade", "population": 12345},
                {"name": "Broward", "population": 40000},
                {"name": "Palm Beach", "population": 60000},
            ],
            "info": {"governor": "Rick Scott"},
            "shortname": "FL",
            "state": "Florida",
        },
        {
            "counties": [
                {"name": "Summit", "population": 1234},
                {"name": "Cuyahoga", "population": 1337},
            ],
            "info": {"governor": "John Kasich"},
            "shortname": "OH",
            "state": "Ohio",
        },
    ]


@pytest.fixture
def author_missing_data():
    return [
        {"info": None},
        {
            "info": {"created_at": "11/08/1993", "last_updated": "26/05/2012"},
            "author_name": {"first": "Jane", "last_name": "Doe"},
        },
    ]


@pytest.fixture
def missing_metadata():
    return [
        {
            "name": "Alice",
            "addresses": [
                {
                    "number": 9562,
                    "street": "Morris St.",
                    "city": "Massillon",
                    "state": "OH",
                    "zip": 44646,
                }
            ],
            "previous_residences": {"cities": [{"city_name": "Foo York City"}]},
        },
        {
            "addresses": [
                {
                    "number": 8449,
                    "street": "Spring St.",
                    "city": "Elizabethton",
                    "state": "TN",
                    "zip": 37643,
                }
            ],
            "previous_residences": {"cities": [{"city_name": "Barmingham"}]},
        },
    ]


@pytest.fixture
def max_level_test_input_data():
    """
    input data to test json_normalize with max_level param
    """
    return [
        {
            "CreatedBy": {"Name": "User001"},
            "Lookup": {
                "TextField": "Some text",
                "UserField": {"Id": "ID001", "Name": "Name001"},
            },
            "Image": {"a": "b"},
        }
    ]


class TestJSONNormalize:
    def test_simple_records(self):
        recs = [
            {"a": 1, "b": 2, "c": 3},
            {"a": 4, "b": 5, "c": 6},
            {"a": 7, "b": 8, "c": 9},
            {"a": 10, "b": 11, "c": 12},
        ]

        result = json_normalize(recs)
        expected = DataFrame(recs)

        tm.assert_frame_equal(result, expected)

    def test_simple_normalize(self, state_data):
        result = json_normalize(state_data[0], "counties")
        expected = DataFrame(state_data[0]["counties"])
        tm.assert_frame_equal(result, expected)

        result = json_normalize(state_data, "counties")

        expected = []
        for rec in state_data:
            expected.extend(rec["counties"])
        expected = DataFrame(expected)

        tm.assert_frame_equal(result, expected)

        result = json_normalize(state_data, "counties", meta="state")
        expected["state"] = np.array(["Florida", "Ohio"]).repeat([3, 2])

        tm.assert_frame_equal(result, expected)

    def test_fields_list_type_normalize(self):
        parse_metadata_fields_list_type = [
            {"values": [1, 2, 3], "metadata": {"listdata": [1, 2]}}
        ]
        result = json_normalize(
            parse_metadata_fields_list_type,
            record_path=["values"],
            meta=[["metadata", "listdata"]],
        )
        expected = DataFrame(
            {0: [1, 2, 3], "metadata.listdata": [[1, 2], [1, 2], [1, 2]]}
        )
        tm.assert_frame_equal(result, expected)

    def test_empty_array(self):
        result = json_normalize([])
        expected = DataFrame()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "data, record_path, exception_type",
        [
            ([{"a": 0}, {"a": 1}], None, None),
            ({"a": [{"a": 0}, {"a": 1}]}, "a", None),
            ('{"a": [{"a": 0}, {"a": 1}]}', None, NotImplementedError),
            (None, None, NotImplementedError),
        ],
    )
    def test_accepted_input(self, data, record_path, exception_type):
        if exception_type is not None:
            with pytest.raises(exception_type, match=""):
                json_normalize(data, record_path=record_path)
        else:
            result = json_normalize(data, record_path=record_path)
            expected = DataFrame([0, 1], columns=["a"])
            tm.assert_frame_equal(result, expected)

    def test_simple_normalize_with_separator(self, deep_nested):
        # GH 14883
        result = json_normalize({"A": {"A": 1, "B": 2}})
        expected = DataFrame([[1, 2]], columns=["A.A", "A.B"])
        tm.assert_frame_equal(result.reindex_like(expected), expected)

        result = json_normalize({"A": {"A": 1, "B": 2}}, sep="_")
        expected = DataFrame([[1, 2]], columns=["A_A", "A_B"])
        tm.assert_frame_equal(result.reindex_like(expected), expected)

        result = json_normalize({"A": {"A": 1, "B": 2}}, sep="\u03c3")
        expected = DataFrame([[1, 2]], columns=["A\u03c3A", "A\u03c3B"])
        tm.assert_frame_equal(result.reindex_like(expected), expected)

        result = json_normalize(
            deep_nested,
            ["states", "cities"],
            meta=["country", ["states", "name"]],
            sep="_",
        )
        expected = Index(["name", "pop", "country", "states_name"]).sort_values()
        assert result.columns.sort_values().equals(expected)

    def test_normalize_with_multichar_separator(self):
        # GH #43831
        data = {"a": [1, 2], "b": {"b_1": 2, "b_2": (3, 4)}}
        result = json_normalize(data, sep="__")
        expected = DataFrame([[[1, 2], 2, (3, 4)]], columns=["a", "b__b_1", "b__b_2"])
        tm.assert_frame_equal(result, expected)

    def test_value_array_record_prefix(self):
        # GH 21536
        result = json_normalize({"A": [1, 2]}, "A", record_prefix="Prefix.")
        expected = DataFrame([[1], [2]], columns=["Prefix.0"])
        tm.assert_frame_equal(result, expected)

    def test_nested_object_record_path(self):
        # GH 22706
        data = {
            "state": "Florida",
            "info": {
                "governor": "Rick Scott",
                "counties": [
                    {"name": "Dade", "population": 12345},
                    {"name": "Broward", "population": 40000},
                    {"name": "Palm Beach", "population": 60000},
                ],
            },
        }
        result = json_normalize(data, record_path=["info", "counties"])
        expected = DataFrame(
            [["Dade", 12345], ["Broward", 40000], ["Palm Beach", 60000]],
            columns=["name", "population"],
        )
        tm.assert_frame_equal(result, expected)

    def test_more_deeply_nested(self, deep_nested):
        result = json_normalize(
            deep_nested, ["states", "cities"], meta=["country", ["states", "name"]]
        )
        ex_data = {
            "country": ["USA"] * 4 + ["Germany"] * 3,
            "states.name": [
                "California",
                "California",
                "Ohio",
                "Ohio",
                "Bayern",
                "Nordrhein-Westfalen",
                "Nordrhein-Westfalen",
            ],
            "name": [
                "San Francisco",
                "Los Angeles",
                "Columbus",
                "Cleveland",
                "Munich",
                "Duesseldorf",
                "Koeln",
            ],
            "pop": [12345, 12346, 1234, 1236, 12347, 1238, 1239],
        }

        expected = DataFrame(ex_data, columns=result.columns)
        tm.assert_frame_equal(result, expected)

    def test_shallow_nested(self):
        data = [
            {
                "state": "Florida",
                "shortname": "FL",
                "info": {"governor": "Rick Scott"},
                "counties": [
                    {"name": "Dade", "population": 12345},
                    {"name": "Broward", "population": 40000},
                    {"name": "Palm Beach", "population": 60000},
                ],
            },
            {
                "state": "Ohio",
                "shortname": "OH",
                "info": {"governor": "John Kasich"},
                "counties": [
                    {"name": "Summit", "population": 1234},
                    {"name": "Cuyahoga", "population": 1337},
                ],
            },
        ]

        result = json_normalize(
            data, "counties", ["state", "shortname", ["info", "governor"]]
        )
        ex_data = {
            "name": ["Dade", "Broward", "Palm Beach", "Summit", "Cuyahoga"],
            "state": ["Florida"] * 3 + ["Ohio"] * 2,
            "shortname": ["FL", "FL", "FL", "OH", "OH"],
            "info.governor": ["Rick Scott"] * 3 + ["John Kasich"] * 2,
            "population": [12345, 40000, 60000, 1234, 1337],
        }
        expected = DataFrame(ex_data, columns=result.columns)
        tm.assert_frame_equal(result, expected)

    def test_nested_meta_path_with_nested_record_path(self, state_data):
        # GH 27220
        result = json_normalize(
            data=state_data,
            record_path=["counties"],
            meta=["state", "shortname", ["info", "governor"]],
            errors="ignore",
        )

        ex_data = {
            "name": ["Dade", "Broward", "Palm Beach", "Summit", "Cuyahoga"],
            "population": [12345, 40000, 60000, 1234, 1337],
            "state": ["Florida"] * 3 + ["Ohio"] * 2,
            "shortname": ["FL"] * 3 + ["OH"] * 2,
            "info.governor": ["Rick Scott"] * 3 + ["John Kasich"] * 2,
        }

        expected = DataFrame(ex_data)
        tm.assert_frame_equal(result, expected)

    def test_meta_name_conflict(self):
        data = [
            {
                "foo": "hello",
                "bar": "there",
                "data": [
                    {"foo": "something", "bar": "else"},
                    {"foo": "something2", "bar": "else2"},
                ],
            }
        ]

        msg = r"Conflicting metadata name (foo|bar), need distinguishing prefix"
        with pytest.raises(ValueError, match=msg):
            json_normalize(data, "data", meta=["foo", "bar"])

        result = json_normalize(data, "data", meta=["foo", "bar"], meta_prefix="meta")

        for val in ["metafoo", "metabar", "foo", "bar"]:
            assert val in result

    def test_meta_parameter_not_modified(self):
        # GH 18610
        data = [
            {
                "foo": "hello",
                "bar": "there",
                "data": [
                    {"foo": "something", "bar": "else"},
                    {"foo": "something2", "bar": "else2"},
                ],
            }
        ]

        COLUMNS = ["foo", "bar"]
        result = json_normalize(data, "data", meta=COLUMNS, meta_prefix="meta")

        assert COLUMNS == ["foo", "bar"]
        for val in ["metafoo", "metabar", "foo", "bar"]:
            assert val in result

    def test_record_prefix(self, state_data):
        result = json_normalize(state_data[0], "counties")
        expected = DataFrame(state_data[0]["counties"])
        tm.assert_frame_equal(result, expected)

        result = json_normalize(
            state_data, "counties", meta="state", record_prefix="county_"
        )

        expected = []
        for rec in state_data:
            expected.extend(rec["counties"])
        expected = DataFrame(expected)
        expected = expected.rename(columns=lambda x: "county_" + x)
        expected["state"] = np.array(["Florida", "Ohio"]).repeat([3, 2])

        tm.assert_frame_equal(result, expected)

    def test_non_ascii_key(self):
        testjson = (
            b'[{"\xc3\x9cnic\xc3\xb8de":0,"sub":{"A":1, "B":2}},'
            b'{"\xc3\x9cnic\xc3\xb8de":1,"sub":{"A":3, "B":4}}]'
        ).decode("utf8")

        testdata = {
            b"\xc3\x9cnic\xc3\xb8de".decode("utf8"): [0, 1],
            "sub.A": [1, 3],
            "sub.B": [2, 4],
        }
        expected = DataFrame(testdata)

        result = json_normalize(json.loads(testjson))
        tm.assert_frame_equal(result, expected)

    def test_missing_field(self, author_missing_data):
        # GH20030:
        result = json_normalize(author_missing_data)
        ex_data = [
            {
                "info": np.nan,
                "info.created_at": np.nan,
                "info.last_updated": np.nan,
                "author_name.first": np.nan,
                "author_name.last_name": np.nan,
            },
            {
                "info": None,
                "info.created_at": "11/08/1993",
                "info.last_updated": "26/05/2012",
                "author_name.first": "Jane",
                "author_name.last_name": "Doe",
            },
        ]
        expected = DataFrame(ex_data)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "max_level,expected",
        [
            (
                0,
                [
                    {
                        "TextField": "Some text",
                        "UserField": {"Id": "ID001", "Name": "Name001"},
                        "CreatedBy": {"Name": "User001"},
                        "Image": {"a": "b"},
                    },
                    {
                        "TextField": "Some text",
                        "UserField": {"Id": "ID001", "Name": "Name001"},
                        "CreatedBy": {"Name": "User001"},
                        "Image": {"a": "b"},
                    },
                ],
            ),
            (
                1,
                [
                    {
                        "TextField": "Some text",
                        "UserField.Id": "ID001",
                        "UserField.Name": "Name001",
                        "CreatedBy": {"Name": "User001"},
                        "Image": {"a": "b"},
                    },
                    {
                        "TextField": "Some text",
                        "UserField.Id": "ID001",
                        "UserField.Name": "Name001",
                        "CreatedBy": {"Name": "User001"},
                        "Image": {"a": "b"},
                    },
                ],
            ),
        ],
    )
    def test_max_level_with_records_path(self, max_level, expected):
        # GH23843: Enhanced JSON normalize
        test_input = [
            {
                "CreatedBy": {"Name": "User001"},
                "Lookup": [
                    {
                        "TextField": "Some text",
                        "UserField": {"Id": "ID001", "Name": "Name001"},
                    },
                    {
                        "TextField": "Some text",
                        "UserField": {"Id": "ID001", "Name": "Name001"},
                    },
                ],
                "Image": {"a": "b"},
                "tags": [
                    {"foo": "something", "bar": "else"},
                    {"foo": "something2", "bar": "else2"},
                ],
            }
        ]

        result = json_normalize(
            test_input,
            record_path=["Lookup"],
            meta=[["CreatedBy"], ["Image"]],
            max_level=max_level,
        )
        expected_df = DataFrame(data=expected, columns=result.columns.values)
        tm.assert_equal(expected_df, result)

    def test_nested_flattening_consistent(self):
        # see gh-21537
        df1 = json_normalize([{"A": {"B": 1}}])
        df2 = json_normalize({"dummy": [{"A": {"B": 1}}]}, "dummy")

        # They should be the same.
        tm.assert_frame_equal(df1, df2)

    def test_nonetype_record_path(self, nulls_fixture):
        # see gh-30148
        # should not raise TypeError
        result = json_normalize(
            [
                {"state": "Texas", "info": nulls_fixture},
                {"state": "Florida", "info": [{"i": 2}]},
            ],
            record_path=["info"],
        )
        expected = DataFrame({"i": 2}, index=[0])
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("value", ["false", "true", "{}", "1", '"text"'])
    def test_non_list_record_path_errors(self, value):
        # see gh-30148, GH 26284
        parsed_value = json.loads(value)
        test_input = {"state": "Texas", "info": parsed_value}
        test_path = "info"
        msg = (
            f"{test_input} has non list value {parsed_value} for path {test_path}. "
            "Must be list or null."
        )
        with pytest.raises(TypeError, match=msg):
            json_normalize([test_input], record_path=[test_path])

    def test_meta_non_iterable(self):
        # GH 31507
        data = """[{"id": 99, "data": [{"one": 1, "two": 2}]}]"""

        result = json_normalize(json.loads(data), record_path=["data"], meta=["id"])
        expected = DataFrame(
            {"one": [1], "two": [2], "id": np.array([99], dtype=object)}
        )
        tm.assert_frame_equal(result, expected)

    def test_generator(self, state_data):
        # GH35923 Fix pd.json_normalize to not skip the first element of a
        # generator input
        def generator_data():
            yield from state_data[0]["counties"]

        result = json_normalize(generator_data())
        expected = DataFrame(state_data[0]["counties"])

        tm.assert_frame_equal(result, expected)

    def test_top_column_with_leading_underscore(self):
        # 49861
        data = {"_id": {"a1": 10, "l2": {"l3": 0}}, "gg": 4}
        result = json_normalize(data, sep="_")
        expected = DataFrame([[4, 10, 0]], columns=["gg", "_id_a1", "_id_l2_l3"])

        tm.assert_frame_equal(result, expected)


class TestNestedToRecord:
    def test_flat_stays_flat(self):
        recs = [{"flat1": 1, "flat2": 2}, {"flat3": 3, "flat2": 4}]
        result = nested_to_record(recs)
        expected = recs
        assert result == expected

    def test_one_level_deep_flattens(self):
        data = {"flat1": 1, "dict1": {"c": 1, "d": 2}}

        result = nested_to_record(data)
        expected = {"dict1.c": 1, "dict1.d": 2, "flat1": 1}

        assert result == expected

    def test_nested_flattens(self):
        data = {
            "flat1": 1,
            "dict1": {"c": 1, "d": 2},
            "nested": {"e": {"c": 1, "d": 2}, "d": 2},
        }

        result = nested_to_record(data)
        expected = {
            "dict1.c": 1,
            "dict1.d": 2,
            "flat1": 1,
            "nested.d": 2,
            "nested.e.c": 1,
            "nested.e.d": 2,
        }

        assert result == expected

    def test_json_normalize_errors(self, missing_metadata):
        # GH14583:
        # If meta keys are not always present a new option to set
        # errors='ignore' has been implemented

        msg = (
            "Key 'name' not found. To replace missing values of "
            "'name' with np.nan, pass in errors='ignore'"
        )
        with pytest.raises(KeyError, match=msg):
            json_normalize(
                data=missing_metadata,
                record_path="addresses",
                meta="name",
                errors="raise",
            )

    def test_missing_meta(self, missing_metadata):
        # GH25468
        # If metadata is nullable with errors set to ignore, the null values
        # should be numpy.nan values
        result = json_normalize(
            data=missing_metadata, record_path="addresses", meta="name", errors="ignore"
        )
        ex_data = [
            [9562, "Morris St.", "Massillon", "OH", 44646, "Alice"],
            [8449, "Spring St.", "Elizabethton", "TN", 37643, np.nan],
        ]
        columns = ["number", "street", "city", "state", "zip", "name"]
        expected = DataFrame(ex_data, columns=columns)
        tm.assert_frame_equal(result, expected)

    def test_missing_nested_meta(self):
        # GH44312
        # If errors="ignore" and nested metadata is null, we should return nan
        data = {"meta": "foo", "nested_meta": None, "value": [{"rec": 1}, {"rec": 2}]}
        result = json_normalize(
            data,
            record_path="value",
            meta=["meta", ["nested_meta", "leaf"]],
            errors="ignore",
        )
        ex_data = [[1, "foo", np.nan], [2, "foo", np.nan]]
        columns = ["rec", "meta", "nested_meta.leaf"]
        expected = DataFrame(ex_data, columns=columns).astype(
            {"nested_meta.leaf": object}
        )
        tm.assert_frame_equal(result, expected)

        # If errors="raise" and nested metadata is null, we should raise with the
        # key of the first missing level
        with pytest.raises(KeyError, match="'leaf' not found"):
            json_normalize(
                data,
                record_path="value",
                meta=["meta", ["nested_meta", "leaf"]],
                errors="raise",
            )

    def test_missing_meta_multilevel_record_path_errors_raise(self, missing_metadata):
        # GH41876
        # Ensure errors='raise' works as intended even when a record_path of length
        # greater than one is passed in
        msg = (
            "Key 'name' not found. To replace missing values of "
            "'name' with np.nan, pass in errors='ignore'"
        )
        with pytest.raises(KeyError, match=msg):
            json_normalize(
                data=missing_metadata,
                record_path=["previous_residences", "cities"],
                meta="name",
                errors="raise",
            )

    def test_missing_meta_multilevel_record_path_errors_ignore(self, missing_metadata):
        # GH41876
        # Ensure errors='ignore' works as intended even when a record_path of length
        # greater than one is passed in
        result = json_normalize(
            data=missing_metadata,
            record_path=["previous_residences", "cities"],
            meta="name",
            errors="ignore",
        )
        ex_data = [
            ["Foo York City", "Alice"],
            ["Barmingham", np.nan],
        ]
        columns = ["city_name", "name"]
        expected = DataFrame(ex_data, columns=columns)
        tm.assert_frame_equal(result, expected)

    def test_donot_drop_nonevalues(self):
        # GH21356
        data = [
            {"info": None, "author_name": {"first": "Smith", "last_name": "Appleseed"}},
            {
                "info": {"created_at": "11/08/1993", "last_updated": "26/05/2012"},
                "author_name": {"first": "Jane", "last_name": "Doe"},
            },
        ]
        result = nested_to_record(data)
        expected = [
            {
                "info": None,
                "author_name.first": "Smith",
                "author_name.last_name": "Appleseed",
            },
            {
                "author_name.first": "Jane",
                "author_name.last_name": "Doe",
                "info.created_at": "11/08/1993",
                "info.last_updated": "26/05/2012",
            },
        ]

        assert result == expected

    def test_nonetype_top_level_bottom_level(self):
        # GH21158: If inner level json has a key with a null value
        # make sure it does not do a new_d.pop twice and except
        data = {
            "id": None,
            "location": {
                "country": {
                    "state": {
                        "id": None,
                        "town.info": {
                            "id": None,
                            "region": None,
                            "x": 49.151580810546875,
                            "y": -33.148521423339844,
                            "z": 27.572303771972656,
                        },
                    }
                }
            },
        }
        result = nested_to_record(data)
        expected = {
            "id": None,
            "location.country.state.id": None,
            "location.country.state.town.info.id": None,
            "location.country.state.town.info.region": None,
            "location.country.state.town.info.x": 49.151580810546875,
            "location.country.state.town.info.y": -33.148521423339844,
            "location.country.state.town.info.z": 27.572303771972656,
        }
        assert result == expected

    def test_nonetype_multiple_levels(self):
        # GH21158: If inner level json has a key with a null value
        # make sure it does not do a new_d.pop twice and except
        data = {
            "id": None,
            "location": {
                "id": None,
                "country": {
                    "id": None,
                    "state": {
                        "id": None,
                        "town.info": {
                            "region": None,
                            "x": 49.151580810546875,
                            "y": -33.148521423339844,
                            "z": 27.572303771972656,
                        },
                    },
                },
            },
        }
        result = nested_to_record(data)
        expected = {
            "id": None,
            "location.id": None,
            "location.country.id": None,
            "location.country.state.id": None,
            "location.country.state.town.info.region": None,
            "location.country.state.town.info.x": 49.151580810546875,
            "location.country.state.town.info.y": -33.148521423339844,
            "location.country.state.town.info.z": 27.572303771972656,
        }
        assert result == expected

    @pytest.mark.parametrize(
        "max_level, expected",
        [
            (
                None,
                [
                    {
                        "CreatedBy.Name": "User001",
                        "Lookup.TextField": "Some text",
                        "Lookup.UserField.Id": "ID001",
                        "Lookup.UserField.Name": "Name001",
                        "Image.a": "b",
                    }
                ],
            ),
            (
                0,
                [
                    {
                        "CreatedBy": {"Name": "User001"},
                        "Lookup": {
                            "TextField": "Some text",
                            "UserField": {"Id": "ID001", "Name": "Name001"},
                        },
                        "Image": {"a": "b"},
                    }
                ],
            ),
            (
                1,
                [
                    {
                        "CreatedBy.Name": "User001",
                        "Lookup.TextField": "Some text",
                        "Lookup.UserField": {"Id": "ID001", "Name": "Name001"},
                        "Image.a": "b",
                    }
                ],
            ),
        ],
    )
    def test_with_max_level(self, max_level, expected, max_level_test_input_data):
        # GH23843: Enhanced JSON normalize
        output = nested_to_record(max_level_test_input_data, max_level=max_level)
        assert output == expected

    def test_with_large_max_level(self):
        # GH23843: Enhanced JSON normalize
        max_level = 100
        input_data = [
            {
                "CreatedBy": {
                    "user": {
                        "name": {"firstname": "Leo", "LastName": "Thomson"},
                        "family_tree": {
                            "father": {
                                "name": "Father001",
                                "father": {
                                    "Name": "Father002",
                                    "father": {
                                        "name": "Father003",
                                        "father": {"Name": "Father004"},
                                    },
                                },
                            }
                        },
                    }
                }
            }
        ]
        expected = [
            {
                "CreatedBy.user.name.firstname": "Leo",
                "CreatedBy.user.name.LastName": "Thomson",
                "CreatedBy.user.family_tree.father.name": "Father001",
                "CreatedBy.user.family_tree.father.father.Name": "Father002",
                "CreatedBy.user.family_tree.father.father.father.name": "Father003",
                "CreatedBy.user.family_tree.father.father.father.father.Name": "Father004",  # noqa: E501
            }
        ]
        output = nested_to_record(input_data, max_level=max_level)
        assert output == expected

    def test_series_non_zero_index(self):
        # GH 19020
        data = {
            0: {"id": 1, "name": "Foo", "elements": {"a": 1}},
            1: {"id": 2, "name": "Bar", "elements": {"b": 2}},
            2: {"id": 3, "name": "Baz", "elements": {"c": 3}},
        }
        s = Series(data)
        s.index = [1, 2, 3]
        result = json_normalize(s)
        expected = DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Foo", "Bar", "Baz"],
                "elements.a": [1.0, np.nan, np.nan],
                "elements.b": [np.nan, 2.0, np.nan],
                "elements.c": [np.nan, np.nan, 3.0],
            }
        )
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_printing.py ===
# -*- encoding: utf-8 -*-
"""
TODO:
* Address Issue 2251, printing of spin states
"""
from __future__ import annotations
from typing import Any

from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.cg import CG, Wigner3j, Wigner6j, Wigner9j
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.gate import CGate, CNotGate, IdentityGate, UGate, XGate
from sympy.physics.quantum.hilbert import ComplexSpace, FockSpace, HilbertSpace, L2
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.operator import Operator, OuterProduct, DifferentialOperator
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.qubit import Qubit, IntQubit
from sympy.physics.quantum.spin import Jz, J2, JzBra, JzBraCoupled, JzKet, JzKetCoupled, Rotation, WignerD
from sympy.physics.quantum.state import Bra, Ket, TimeDepBra, TimeDepKet
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.sho1d import RaisingOp

from sympy.core.function import (Derivative, Function)
from sympy.core.numbers import oo
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.matrices.dense import Matrix
from sympy.sets.sets import Interval
from sympy.testing.pytest import XFAIL

# Imports used in srepr strings
from sympy.physics.quantum.spin import JzOp

from sympy.printing import srepr
from sympy.printing.pretty import pretty as xpretty
from sympy.printing.latex import latex

MutableDenseMatrix = Matrix


ENV: dict[str, Any] = {}
exec('from sympy import *', ENV)
exec('from sympy.physics.quantum import *', ENV)
exec('from sympy.physics.quantum.cg import *', ENV)
exec('from sympy.physics.quantum.spin import *', ENV)
exec('from sympy.physics.quantum.hilbert import *', ENV)
exec('from sympy.physics.quantum.qubit import *', ENV)
exec('from sympy.physics.quantum.qexpr import *', ENV)
exec('from sympy.physics.quantum.gate import *', ENV)
exec('from sympy.physics.quantum.constants import *', ENV)


def sT(expr, string):
    """
    sT := sreprTest
    from sympy/printing/tests/test_repr.py
    """
    assert srepr(expr) == string
    assert eval(string, ENV) == expr


def pretty(expr):
    """ASCII pretty-printing"""
    return xpretty(expr, use_unicode=False, wrap_line=False)


def upretty(expr):
    """Unicode pretty-printing"""
    return xpretty(expr, use_unicode=True, wrap_line=False)


def test_anticommutator():
    A = Operator('A')
    B = Operator('B')
    ac = AntiCommutator(A, B)
    ac_tall = AntiCommutator(A**2, B)
    assert str(ac) == '{A,B}'
    assert pretty(ac) == '{A,B}'
    assert upretty(ac) == '{A,B}'
    assert latex(ac) == r'\left\{A,B\right\}'
    sT(ac, "AntiCommutator(Operator(Symbol('A')),Operator(Symbol('B')))")
    assert str(ac_tall) == '{A**2,B}'
    ascii_str = \
"""\
/ 2  \\\n\
<A ,B>\n\
\\    /\
"""
    ucode_str = \
"""\
⎧ 2  ⎫\n\
⎨A ,B⎬\n\
⎩    ⎭\
"""
    assert pretty(ac_tall) == ascii_str
    assert upretty(ac_tall) == ucode_str
    assert latex(ac_tall) == r'\left\{A^{2},B\right\}'
    sT(ac_tall, "AntiCommutator(Pow(Operator(Symbol('A')), Integer(2)),Operator(Symbol('B')))")


def test_cg():
    cg = CG(1, 2, 3, 4, 5, 6)
    wigner3j = Wigner3j(1, 2, 3, 4, 5, 6)
    wigner6j = Wigner6j(1, 2, 3, 4, 5, 6)
    wigner9j = Wigner9j(1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert str(cg) == 'CG(1, 2, 3, 4, 5, 6)'
    ascii_str = \
"""\
 5,6    \n\
C       \n\
 1,2,3,4\
"""
    ucode_str = \
"""\
 5,6    \n\
C       \n\
 1,2,3,4\
"""
    assert pretty(cg) == ascii_str
    assert upretty(cg) == ucode_str
    assert latex(cg) == 'C^{5,6}_{1,2,3,4}'
    assert latex(cg ** 2) == R'\left(C^{5,6}_{1,2,3,4}\right)^{2}'
    sT(cg, "CG(Integer(1), Integer(2), Integer(3), Integer(4), Integer(5), Integer(6))")
    assert str(wigner3j) == 'Wigner3j(1, 2, 3, 4, 5, 6)'
    ascii_str = \
"""\
/1  3  5\\\n\
|       |\n\
\\2  4  6/\
"""
    ucode_str = \
"""\
⎛1  3  5⎞\n\
⎜       ⎟\n\
⎝2  4  6⎠\
"""
    assert pretty(wigner3j) == ascii_str
    assert upretty(wigner3j) == ucode_str
    assert latex(wigner3j) == \
        r'\left(\begin{array}{ccc} 1 & 3 & 5 \\ 2 & 4 & 6 \end{array}\right)'
    sT(wigner3j, "Wigner3j(Integer(1), Integer(2), Integer(3), Integer(4), Integer(5), Integer(6))")
    assert str(wigner6j) == 'Wigner6j(1, 2, 3, 4, 5, 6)'
    ascii_str = \
"""\
/1  2  3\\\n\
<       >\n\
\\4  5  6/\
"""
    ucode_str = \
"""\
⎧1  2  3⎫\n\
⎨       ⎬\n\
⎩4  5  6⎭\
"""
    assert pretty(wigner6j) == ascii_str
    assert upretty(wigner6j) == ucode_str
    assert latex(wigner6j) == \
        r'\left\{\begin{array}{ccc} 1 & 2 & 3 \\ 4 & 5 & 6 \end{array}\right\}'
    sT(wigner6j, "Wigner6j(Integer(1), Integer(2), Integer(3), Integer(4), Integer(5), Integer(6))")
    assert str(wigner9j) == 'Wigner9j(1, 2, 3, 4, 5, 6, 7, 8, 9)'
    ascii_str = \
"""\
/1  2  3\\\n\
|       |\n\
<4  5  6>\n\
|       |\n\
\\7  8  9/\
"""
    ucode_str = \
"""\
⎧1  2  3⎫\n\
⎪       ⎪\n\
⎨4  5  6⎬\n\
⎪       ⎪\n\
⎩7  8  9⎭\
"""
    assert pretty(wigner9j) == ascii_str
    assert upretty(wigner9j) == ucode_str
    assert latex(wigner9j) == \
        r'\left\{\begin{array}{ccc} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{array}\right\}'
    sT(wigner9j, "Wigner9j(Integer(1), Integer(2), Integer(3), Integer(4), Integer(5), Integer(6), Integer(7), Integer(8), Integer(9))")


def test_commutator():
    A = Operator('A')
    B = Operator('B')
    c = Commutator(A, B)
    c_tall = Commutator(A**2, B)
    assert str(c) == '[A,B]'
    assert pretty(c) == '[A,B]'
    assert upretty(c) == '[A,B]'
    assert latex(c) == r'\left[A,B\right]'
    sT(c, "Commutator(Operator(Symbol('A')),Operator(Symbol('B')))")
    assert str(c_tall) == '[A**2,B]'
    ascii_str = \
"""\
[ 2  ]\n\
[A ,B]\
"""
    ucode_str = \
"""\
⎡ 2  ⎤\n\
⎣A ,B⎦\
"""
    assert pretty(c_tall) == ascii_str
    assert upretty(c_tall) == ucode_str
    assert latex(c_tall) == r'\left[A^{2},B\right]'
    sT(c_tall, "Commutator(Pow(Operator(Symbol('A')), Integer(2)),Operator(Symbol('B')))")


def test_constants():
    assert str(hbar) == 'hbar'
    assert pretty(hbar) == 'hbar'
    assert upretty(hbar) == 'ℏ'
    assert latex(hbar) == r'\hbar'
    sT(hbar, "HBar()")


def test_dagger():
    x = symbols('x', commutative=False)
    expr = Dagger(x)
    assert str(expr) == 'Dagger(x)'
    ascii_str = \
"""\
 +\n\
x \
"""
    ucode_str = \
"""\
 †\n\
x \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    assert latex(expr) == r'x^{\dagger}'
    sT(expr, "Dagger(Symbol('x', commutative=False))")


@XFAIL
def test_gate_failing():
    a, b, c, d = symbols('a,b,c,d')
    uMat = Matrix([[a, b], [c, d]])
    g = UGate((0,), uMat)
    assert str(g) == 'U(0)'


def test_gate():
    a, b, c, d = symbols('a,b,c,d')
    uMat = Matrix([[a, b], [c, d]])
    q = Qubit(1, 0, 1, 0, 1)
    g1 = IdentityGate(2)
    g2 = CGate((3, 0), XGate(1))
    g3 = CNotGate(1, 0)
    g4 = UGate((0,), uMat)
    assert str(g1) == '1(2)'
    assert pretty(g1) == '1 \n 2'
    assert upretty(g1) == '1 \n 2'
    assert latex(g1) == r'1_{2}'
    sT(g1, "IdentityGate(Integer(2))")
    assert str(g1*q) == '1(2)*|10101>'
    ascii_str = \
"""\
1 *|10101>\n\
 2        \
"""
    ucode_str = \
"""\
1 ⋅❘10101⟩\n\
 2        \
"""
    assert pretty(g1*q) == ascii_str
    assert upretty(g1*q) == ucode_str
    assert latex(g1*q) == r'1_{2} {\left|10101\right\rangle }'
    sT(g1*q, "Mul(IdentityGate(Integer(2)), Qubit(Integer(1),Integer(0),Integer(1),Integer(0),Integer(1)))")
    assert str(g2) == 'C((3,0),X(1))'
    ascii_str = \
"""\
C   /X \\\n\
 3,0\\ 1/\
"""
    ucode_str = \
"""\
C   ⎛X ⎞\n\
 3,0⎝ 1⎠\
"""
    assert pretty(g2) == ascii_str
    assert upretty(g2) == ucode_str
    assert latex(g2) == r'C_{3,0}{\left(X_{1}\right)}'
    sT(g2, "CGate(Tuple(Integer(3), Integer(0)),XGate(Integer(1)))")
    assert str(g3) == 'CNOT(1,0)'
    ascii_str = \
"""\
CNOT   \n\
    1,0\
"""
    ucode_str = \
"""\
CNOT   \n\
    1,0\
"""
    assert pretty(g3) == ascii_str
    assert upretty(g3) == ucode_str
    assert latex(g3) == r'\text{CNOT}_{1,0}'
    sT(g3, "CNotGate(Integer(1),Integer(0))")
    ascii_str = \
"""\
U \n\
 0\
"""
    ucode_str = \
"""\
U \n\
 0\
"""
    assert str(g4) == \
"""\
U((0,),Matrix([\n\
[a, b],\n\
[c, d]]))\
"""
    assert pretty(g4) == ascii_str
    assert upretty(g4) == ucode_str
    assert latex(g4) == r'U_{0}'
    sT(g4, "UGate(Tuple(Integer(0)),ImmutableDenseMatrix([[Symbol('a'), Symbol('b')], [Symbol('c'), Symbol('d')]]))")


def test_hilbert():
    h1 = HilbertSpace()
    h2 = ComplexSpace(2)
    h3 = FockSpace()
    h4 = L2(Interval(0, oo))
    assert str(h1) == 'H'
    assert pretty(h1) == 'H'
    assert upretty(h1) == 'H'
    assert latex(h1) == r'\mathcal{H}'
    sT(h1, "HilbertSpace()")
    assert str(h2) == 'C(2)'
    ascii_str = \
"""\
 2\n\
C \
"""
    ucode_str = \
"""\
 2\n\
C \
"""
    assert pretty(h2) == ascii_str
    assert upretty(h2) == ucode_str
    assert latex(h2) == r'\mathcal{C}^{2}'
    sT(h2, "ComplexSpace(Integer(2))")
    assert str(h3) == 'F'
    assert pretty(h3) == 'F'
    assert upretty(h3) == 'F'
    assert latex(h3) == r'\mathcal{F}'
    sT(h3, "FockSpace()")
    assert str(h4) == 'L2(Interval(0, oo))'
    ascii_str = \
"""\
 2\n\
L \
"""
    ucode_str = \
"""\
 2\n\
L \
"""
    assert pretty(h4) == ascii_str
    assert upretty(h4) == ucode_str
    assert latex(h4) == r'{\mathcal{L}^2}\left( \left[0, \infty\right) \right)'
    sT(h4, "L2(Interval(Integer(0), oo, false, true))")
    assert str(h1 + h2) == 'H+C(2)'
    ascii_str = \
"""\
     2\n\
H + C \
"""
    ucode_str = \
"""\
     2\n\
H ⊕ C \
"""
    assert pretty(h1 + h2) == ascii_str
    assert upretty(h1 + h2) == ucode_str
    assert latex(h1 + h2)
    sT(h1 + h2, "DirectSumHilbertSpace(HilbertSpace(),ComplexSpace(Integer(2)))")
    assert str(h1*h2) == "H*C(2)"
    ascii_str = \
"""\
     2\n\
H x C \
"""
    ucode_str = \
"""\
     2\n\
H ⨂ C \
"""
    assert pretty(h1*h2) == ascii_str
    assert upretty(h1*h2) == ucode_str
    assert latex(h1*h2)
    sT(h1*h2,
       "TensorProductHilbertSpace(HilbertSpace(),ComplexSpace(Integer(2)))")
    assert str(h1**2) == 'H**2'
    ascii_str = \
"""\
 x2\n\
H  \
"""
    ucode_str = \
"""\
 ⨂2\n\
H  \
"""
    assert pretty(h1**2) == ascii_str
    assert upretty(h1**2) == ucode_str
    assert latex(h1**2) == r'{\mathcal{H}}^{\otimes 2}'
    sT(h1**2, "TensorPowerHilbertSpace(HilbertSpace(),Integer(2))")


def test_innerproduct():
    x = symbols('x')
    ip1 = InnerProduct(Bra(), Ket())
    ip2 = InnerProduct(TimeDepBra(), TimeDepKet())
    ip3 = InnerProduct(JzBra(1, 1), JzKet(1, 1))
    ip4 = InnerProduct(JzBraCoupled(1, 1, (1, 1)), JzKetCoupled(1, 1, (1, 1)))
    ip_tall1 = InnerProduct(Bra(x/2), Ket(x/2))
    ip_tall2 = InnerProduct(Bra(x), Ket(x/2))
    ip_tall3 = InnerProduct(Bra(x/2), Ket(x))
    assert str(ip1) == '<psi|psi>'
    assert pretty(ip1) == '<psi|psi>'
    assert upretty(ip1) == '⟨ψ❘ψ⟩'
    assert latex(
        ip1) == r'\left\langle \psi \right. {\left|\psi\right\rangle }'
    sT(ip1, "InnerProduct(Bra(Symbol('psi')),Ket(Symbol('psi')))")
    assert str(ip2) == '<psi;t|psi;t>'
    assert pretty(ip2) == '<psi;t|psi;t>'
    assert upretty(ip2) == '⟨ψ;t❘ψ;t⟩'
    assert latex(ip2) == \
        r'\left\langle \psi;t \right. {\left|\psi;t\right\rangle }'
    sT(ip2, "InnerProduct(TimeDepBra(Symbol('psi'),Symbol('t')),TimeDepKet(Symbol('psi'),Symbol('t')))")
    assert str(ip3) == "<1,1|1,1>"
    assert pretty(ip3) == '<1,1|1,1>'
    assert upretty(ip3) == '⟨1,1❘1,1⟩'
    assert latex(ip3) == r'\left\langle 1,1 \right. {\left|1,1\right\rangle }'
    sT(ip3, "InnerProduct(JzBra(Integer(1),Integer(1)),JzKet(Integer(1),Integer(1)))")
    assert str(ip4) == "<1,1,j1=1,j2=1|1,1,j1=1,j2=1>"
    assert pretty(ip4) == '<1,1,j1=1,j2=1|1,1,j1=1,j2=1>'
    assert upretty(ip4) == '⟨1,1,j₁=1,j₂=1❘1,1,j₁=1,j₂=1⟩'
    assert latex(ip4) == \
        r'\left\langle 1,1,j_{1}=1,j_{2}=1 \right. {\left|1,1,j_{1}=1,j_{2}=1\right\rangle }'
    sT(ip4, "InnerProduct(JzBraCoupled(Integer(1),Integer(1),Tuple(Integer(1), Integer(1)),Tuple(Tuple(Integer(1), Integer(2), Integer(1)))),JzKetCoupled(Integer(1),Integer(1),Tuple(Integer(1), Integer(1)),Tuple(Tuple(Integer(1), Integer(2), Integer(1)))))")
    assert str(ip_tall1) == '<x/2|x/2>'
    ascii_str = \
"""\
 / | \\ \n\
/ x|x \\\n\
\\ -|- /\n\
 \\2|2/ \
"""
    ucode_str = \
"""\
 ╱ │ ╲ \n\
╱ x│x ╲\n\
╲ ─│─ ╱\n\
 ╲2│2╱ \
"""
    assert pretty(ip_tall1) == ascii_str
    assert upretty(ip_tall1) == ucode_str
    assert latex(ip_tall1) == \
        r'\left\langle \frac{x}{2} \right. {\left|\frac{x}{2}\right\rangle }'
    sT(ip_tall1, "InnerProduct(Bra(Mul(Rational(1, 2), Symbol('x'))),Ket(Mul(Rational(1, 2), Symbol('x'))))")
    assert str(ip_tall2) == '<x|x/2>'
    ascii_str = \
"""\
 / | \\ \n\
/  |x \\\n\
\\ x|- /\n\
 \\ |2/ \
"""
    ucode_str = \
"""\
 ╱ │ ╲ \n\
╱  │x ╲\n\
╲ x│─ ╱\n\
 ╲ │2╱ \
"""
    assert pretty(ip_tall2) == ascii_str
    assert upretty(ip_tall2) == ucode_str
    assert latex(ip_tall2) == \
        r'\left\langle x \right. {\left|\frac{x}{2}\right\rangle }'
    sT(ip_tall2,
       "InnerProduct(Bra(Symbol('x')),Ket(Mul(Rational(1, 2), Symbol('x'))))")
    assert str(ip_tall3) == '<x/2|x>'
    ascii_str = \
"""\
 / | \\ \n\
/ x|  \\\n\
\\ -|x /\n\
 \\2| / \
"""
    ucode_str = \
"""\
 ╱ │ ╲ \n\
╱ x│  ╲\n\
╲ ─│x ╱\n\
 ╲2│ ╱ \
"""
    assert pretty(ip_tall3) == ascii_str
    assert upretty(ip_tall3) == ucode_str
    assert latex(ip_tall3) == \
        r'\left\langle \frac{x}{2} \right. {\left|x\right\rangle }'
    sT(ip_tall3,
       "InnerProduct(Bra(Mul(Rational(1, 2), Symbol('x'))),Ket(Symbol('x')))")


def test_operator():
    a = Operator('A')
    b = Operator('B', Symbol('t'), S.Half)
    inv = a.inv()
    f = Function('f')
    x = symbols('x')
    d = DifferentialOperator(Derivative(f(x), x), f(x))
    op = OuterProduct(Ket(), Bra())
    assert str(a) == 'A'
    assert pretty(a) == 'A'
    assert upretty(a) == 'A'
    assert latex(a) == 'A'
    sT(a, "Operator(Symbol('A'))")
    assert str(inv) == 'A**(-1)'
    ascii_str = \
"""\
 -1\n\
A  \
"""
    ucode_str = \
"""\
 -1\n\
A  \
"""
    assert pretty(inv) == ascii_str
    assert upretty(inv) == ucode_str
    assert latex(inv) == r'A^{-1}'
    sT(inv, "Pow(Operator(Symbol('A')), Integer(-1))")
    assert str(d) == 'DifferentialOperator(Derivative(f(x), x),f(x))'
    ascii_str = \
"""\
                    /d            \\\n\
DifferentialOperator|--(f(x)),f(x)|\n\
                    \\dx           /\
"""
    ucode_str = \
"""\
                    ⎛d            ⎞\n\
DifferentialOperator⎜──(f(x)),f(x)⎟\n\
                    ⎝dx           ⎠\
"""
    assert pretty(d) == ascii_str
    assert upretty(d) == ucode_str
    assert latex(d) == \
        r'DifferentialOperator\left(\frac{d}{d x} f{\left(x \right)},f{\left(x \right)}\right)'
    sT(d, "DifferentialOperator(Derivative(Function('f')(Symbol('x')), Tuple(Symbol('x'), Integer(1))),Function('f')(Symbol('x')))")
    assert str(b) == 'Operator(B,t,1/2)'
    assert pretty(b) == 'Operator(B,t,1/2)'
    assert upretty(b) == 'Operator(B,t,1/2)'
    assert latex(b) == r'Operator\left(B,t,\frac{1}{2}\right)'
    sT(b, "Operator(Symbol('B'),Symbol('t'),Rational(1, 2))")
    assert str(op) == '|psi><psi|'
    assert pretty(op) == '|psi><psi|'
    assert upretty(op) == '❘ψ⟩⟨ψ❘'
    assert latex(op) == r'{\left|\psi\right\rangle }{\left\langle \psi\right|}'
    sT(op, "OuterProduct(Ket(Symbol('psi')),Bra(Symbol('psi')))")


def test_qexpr():
    q = QExpr('q')
    assert str(q) == 'q'
    assert pretty(q) == 'q'
    assert upretty(q) == 'q'
    assert latex(q) == r'q'
    sT(q, "QExpr(Symbol('q'))")


def test_qubit():
    q1 = Qubit('0101')
    q2 = IntQubit(8)
    assert str(q1) == '|0101>'
    assert pretty(q1) == '|0101>'
    assert upretty(q1) == '❘0101⟩'
    assert latex(q1) == r'{\left|0101\right\rangle }'
    sT(q1, "Qubit(Integer(0),Integer(1),Integer(0),Integer(1))")
    assert str(q2) == '|8>'
    assert pretty(q2) == '|8>'
    assert upretty(q2) == '❘8⟩'
    assert latex(q2) == r'{\left|8\right\rangle }'
    sT(q2, "IntQubit(8)")


def test_spin():
    lz = JzOp('L')
    ket = JzKet(1, 0)
    bra = JzBra(1, 0)
    cket = JzKetCoupled(1, 0, (1, 2))
    cbra = JzBraCoupled(1, 0, (1, 2))
    cket_big = JzKetCoupled(1, 0, (1, 2, 3))
    cbra_big = JzBraCoupled(1, 0, (1, 2, 3))
    rot = Rotation(1, 2, 3)
    bigd = WignerD(1, 2, 3, 4, 5, 6)
    smalld = WignerD(1, 2, 3, 0, 4, 0)
    assert str(lz) == 'Lz'
    ascii_str = \
"""\
L \n\
 z\
"""
    ucode_str = \
"""\
L \n\
 z\
"""
    assert pretty(lz) == ascii_str
    assert upretty(lz) == ucode_str
    assert latex(lz) == 'L_z'
    sT(lz, "JzOp(Symbol('L'))")
    assert str(J2) == 'J2'
    ascii_str = \
"""\
 2\n\
J \
"""
    ucode_str = \
"""\
 2\n\
J \
"""
    assert pretty(J2) == ascii_str
    assert upretty(J2) == ucode_str
    assert latex(J2) == r'J^2'
    sT(J2, "J2Op(Symbol('J'))")
    assert str(Jz) == 'Jz'
    ascii_str = \
"""\
J \n\
 z\
"""
    ucode_str = \
"""\
J \n\
 z\
"""
    assert pretty(Jz) == ascii_str
    assert upretty(Jz) == ucode_str
    assert latex(Jz) == 'J_z'
    sT(Jz, "JzOp(Symbol('J'))")
    assert str(ket) == '|1,0>'
    assert pretty(ket) == '|1,0>'
    assert upretty(ket) == '❘1,0⟩'
    assert latex(ket) == r'{\left|1,0\right\rangle }'
    sT(ket, "JzKet(Integer(1),Integer(0))")
    assert str(bra) == '<1,0|'
    assert pretty(bra) == '<1,0|'
    assert upretty(bra) == '⟨1,0❘'
    assert latex(bra) == r'{\left\langle 1,0\right|}'
    sT(bra, "JzBra(Integer(1),Integer(0))")
    assert str(cket) == '|1,0,j1=1,j2=2>'
    assert pretty(cket) == '|1,0,j1=1,j2=2>'
    assert upretty(cket) == '❘1,0,j₁=1,j₂=2⟩'
    assert latex(cket) == r'{\left|1,0,j_{1}=1,j_{2}=2\right\rangle }'
    sT(cket, "JzKetCoupled(Integer(1),Integer(0),Tuple(Integer(1), Integer(2)),Tuple(Tuple(Integer(1), Integer(2), Integer(1))))")
    assert str(cbra) == '<1,0,j1=1,j2=2|'
    assert pretty(cbra) == '<1,0,j1=1,j2=2|'
    assert upretty(cbra) == '⟨1,0,j₁=1,j₂=2❘'
    assert latex(cbra) == r'{\left\langle 1,0,j_{1}=1,j_{2}=2\right|}'
    sT(cbra, "JzBraCoupled(Integer(1),Integer(0),Tuple(Integer(1), Integer(2)),Tuple(Tuple(Integer(1), Integer(2), Integer(1))))")
    assert str(cket_big) == '|1,0,j1=1,j2=2,j3=3,j(1,2)=3>'
    # TODO: Fix non-unicode pretty printing
    # i.e. j1,2 -> j(1,2)
    assert pretty(cket_big) == '|1,0,j1=1,j2=2,j3=3,j1,2=3>'
    assert upretty(cket_big) == '❘1,0,j₁=1,j₂=2,j₃=3,j₁,₂=3⟩'
    assert latex(cket_big) == \
        r'{\left|1,0,j_{1}=1,j_{2}=2,j_{3}=3,j_{1,2}=3\right\rangle }'
    sT(cket_big, "JzKetCoupled(Integer(1),Integer(0),Tuple(Integer(1), Integer(2), Integer(3)),Tuple(Tuple(Integer(1), Integer(2), Integer(3)), Tuple(Integer(1), Integer(3), Integer(1))))")
    assert str(cbra_big) == '<1,0,j1=1,j2=2,j3=3,j(1,2)=3|'
    assert pretty(cbra_big) == '<1,0,j1=1,j2=2,j3=3,j1,2=3|'
    assert upretty(cbra_big) == '⟨1,0,j₁=1,j₂=2,j₃=3,j₁,₂=3❘'
    assert latex(cbra_big) == \
        r'{\left\langle 1,0,j_{1}=1,j_{2}=2,j_{3}=3,j_{1,2}=3\right|}'
    sT(cbra_big, "JzBraCoupled(Integer(1),Integer(0),Tuple(Integer(1), Integer(2), Integer(3)),Tuple(Tuple(Integer(1), Integer(2), Integer(3)), Tuple(Integer(1), Integer(3), Integer(1))))")
    assert str(rot) == 'R(1,2,3)'
    assert pretty(rot) == 'R (1,2,3)'
    assert upretty(rot) == 'ℛ (1,2,3)'
    assert latex(rot) == r'\mathcal{R}\left(1,2,3\right)'
    sT(rot, "Rotation(Integer(1),Integer(2),Integer(3))")
    assert str(bigd) == 'WignerD(1, 2, 3, 4, 5, 6)'
    ascii_str = \
"""\
 1         \n\
D   (4,5,6)\n\
 2,3       \
"""
    ucode_str = \
"""\
 1         \n\
D   (4,5,6)\n\
 2,3       \
"""
    assert pretty(bigd) == ascii_str
    assert upretty(bigd) == ucode_str
    assert latex(bigd) == r'D^{1}_{2,3}\left(4,5,6\right)'
    sT(bigd, "WignerD(Integer(1), Integer(2), Integer(3), Integer(4), Integer(5), Integer(6))")
    assert str(smalld) == 'WignerD(1, 2, 3, 0, 4, 0)'
    ascii_str = \
"""\
 1     \n\
d   (4)\n\
 2,3   \
"""
    ucode_str = \
"""\
 1     \n\
d   (4)\n\
 2,3   \
"""
    assert pretty(smalld) == ascii_str
    assert upretty(smalld) == ucode_str
    assert latex(smalld) == r'd^{1}_{2,3}\left(4\right)'
    sT(smalld, "WignerD(Integer(1), Integer(2), Integer(3), Integer(0), Integer(4), Integer(0))")


def test_state():
    x = symbols('x')
    bra = Bra()
    ket = Ket()
    bra_tall = Bra(x/2)
    ket_tall = Ket(x/2)
    tbra = TimeDepBra()
    tket = TimeDepKet()
    assert str(bra) == '<psi|'
    assert pretty(bra) == '<psi|'
    assert upretty(bra) == '⟨ψ❘'
    assert latex(bra) == r'{\left\langle \psi\right|}'
    sT(bra, "Bra(Symbol('psi'))")
    assert str(ket) == '|psi>'
    assert pretty(ket) == '|psi>'
    assert upretty(ket) == '❘ψ⟩'
    assert latex(ket) == r'{\left|\psi\right\rangle }'
    sT(ket, "Ket(Symbol('psi'))")
    assert str(bra_tall) == '<x/2|'
    ascii_str = \
"""\
 / |\n\
/ x|\n\
\\ -|\n\
 \\2|\
"""
    ucode_str = \
"""\
 ╱ │\n\
╱ x│\n\
╲ ─│\n\
 ╲2│\
"""
    assert pretty(bra_tall) == ascii_str
    assert upretty(bra_tall) == ucode_str
    assert latex(bra_tall) == r'{\left\langle \frac{x}{2}\right|}'
    sT(bra_tall, "Bra(Mul(Rational(1, 2), Symbol('x')))")
    assert str(ket_tall) == '|x/2>'
    ascii_str = \
"""\
| \\ \n\
|x \\\n\
|- /\n\
|2/ \
"""
    ucode_str = \
"""\
│ ╲ \n\
│x ╲\n\
│─ ╱\n\
│2╱ \
"""
    assert pretty(ket_tall) == ascii_str
    assert upretty(ket_tall) == ucode_str
    assert latex(ket_tall) == r'{\left|\frac{x}{2}\right\rangle }'
    sT(ket_tall, "Ket(Mul(Rational(1, 2), Symbol('x')))")
    assert str(tbra) == '<psi;t|'
    assert pretty(tbra) == '<psi;t|'
    assert upretty(tbra) == '⟨ψ;t❘'
    assert latex(tbra) == r'{\left\langle \psi;t\right|}'
    sT(tbra, "TimeDepBra(Symbol('psi'),Symbol('t'))")
    assert str(tket) == '|psi;t>'
    assert pretty(tket) == '|psi;t>'
    assert upretty(tket) == '❘ψ;t⟩'
    assert latex(tket) == r'{\left|\psi;t\right\rangle }'
    sT(tket, "TimeDepKet(Symbol('psi'),Symbol('t'))")


def test_tensorproduct():
    tp = TensorProduct(JzKet(1, 1), JzKet(1, 0))
    assert str(tp) == '|1,1>x|1,0>'
    assert pretty(tp) == '|1,1>x |1,0>'
    assert upretty(tp) == '❘1,1⟩⨂ ❘1,0⟩'
    assert latex(tp) == \
        r'{{\left|1,1\right\rangle }}\otimes {{\left|1,0\right\rangle }}'
    sT(tp, "TensorProduct(JzKet(Integer(1),Integer(1)), JzKet(Integer(1),Integer(0)))")


def test_big_expr():
    f = Function('f')
    x = symbols('x')
    e1 = Dagger(AntiCommutator(Operator('A') + Operator('B'), Pow(DifferentialOperator(Derivative(f(x), x), f(x)), 3))*TensorProduct(Jz**2, Operator('A') + Operator('B')))*(JzBra(1, 0) + JzBra(1, 1))*(JzKet(0, 0) + JzKet(1, -1))
    e2 = Commutator(Jz**2, Operator('A') + Operator('B'))*AntiCommutator(Dagger(Operator('C')*Operator('D')), Operator('E').inv()**2)*Dagger(Commutator(Jz, J2))
    e3 = Wigner3j(1, 2, 3, 4, 5, 6)*TensorProduct(Commutator(Operator('A') + Dagger(Operator('B')), Operator('C') + Operator('D')), Jz - J2)*Dagger(OuterProduct(Dagger(JzBra(1, 1)), JzBra(1, 0)))*TensorProduct(JzKetCoupled(1, 1, (1, 1)) + JzKetCoupled(1, 0, (1, 1)), JzKetCoupled(1, -1, (1, 1)))
    e4 = (ComplexSpace(1)*ComplexSpace(2) + FockSpace()**2)*(L2(Interval(
        0, oo)) + HilbertSpace())
    assert str(e1) == '(Jz**2)x(Dagger(A) + Dagger(B))*{Dagger(DifferentialOperator(Derivative(f(x), x),f(x)))**3,Dagger(A) + Dagger(B)}*(<1,0| + <1,1|)*(|0,0> + |1,-1>)'
    ascii_str = \
"""\
                 /                                      3        \\                                 \n\
                 |/                                   +\\         |                                 \n\
    2  / +    +\\ <|                    /d            \\ |   +    +>                                 \n\
/J \\ x \\A  + B /*||DifferentialOperator|--(f(x)),f(x)| | ,A  + B |*(<1,0| + <1,1|)*(|0,0> + |1,-1>)\n\
\\ z/             \\\\                    \\dx           / /         /                                 \
"""
    ucode_str = \
"""\
                 ⎧                                      3        ⎫                                 \n\
                 ⎪⎛                                   †⎞         ⎪                                 \n\
    2  ⎛ †    †⎞ ⎨⎜                    ⎛d            ⎞ ⎟   †    †⎬                                 \n\
⎛J ⎞ ⨂ ⎝A  + B ⎠⋅⎪⎜DifferentialOperator⎜──(f(x)),f(x)⎟ ⎟ ,A  + B ⎪⋅(⟨1,0❘ + ⟨1,1❘)⋅(❘0,0⟩ + ❘1,-1⟩)\n\
⎝ z⎠             ⎩⎝                    ⎝dx           ⎠ ⎠         ⎭                                 \
"""
    assert pretty(e1) == ascii_str
    assert upretty(e1) == ucode_str
    assert latex(e1) == \
        r'{J_z^{2}}\otimes \left({A^{\dagger} + B^{\dagger}}\right) \left\{\left(DifferentialOperator\left(\frac{d}{d x} f{\left(x \right)},f{\left(x \right)}\right)^{\dagger}\right)^{3},A^{\dagger} + B^{\dagger}\right\} \left({\left\langle 1,0\right|} + {\left\langle 1,1\right|}\right) \left({\left|0,0\right\rangle } + {\left|1,-1\right\rangle }\right)'
    sT(e1, "Mul(TensorProduct(Pow(JzOp(Symbol('J')), Integer(2)), Add(Dagger(Operator(Symbol('A'))), Dagger(Operator(Symbol('B'))))), AntiCommutator(Pow(Dagger(DifferentialOperator(Derivative(Function('f')(Symbol('x')), Tuple(Symbol('x'), Integer(1))),Function('f')(Symbol('x')))), Integer(3)),Add(Dagger(Operator(Symbol('A'))), Dagger(Operator(Symbol('B'))))), Add(JzBra(Integer(1),Integer(0)), JzBra(Integer(1),Integer(1))), Add(JzKet(Integer(0),Integer(0)), JzKet(Integer(1),Integer(-1))))")
    assert str(e2) == '[Jz**2,A + B]*{E**(-2),Dagger(D)*Dagger(C)}*[J2,Jz]'
    ascii_str = \
"""\
[    2      ] / -2  +  +\\ [ 2   ]\n\
[/J \\ ,A + B]*<E  ,D *C >*[J ,J ]\n\
[\\ z/       ] \\         / [    z]\
"""
    ucode_str = \
"""\
⎡    2      ⎤ ⎧ -2  †  †⎫ ⎡ 2   ⎤\n\
⎢⎛J ⎞ ,A + B⎥⋅⎨E  ,D ⋅C ⎬⋅⎢J ,J ⎥\n\
⎣⎝ z⎠       ⎦ ⎩         ⎭ ⎣    z⎦\
"""
    assert pretty(e2) == ascii_str
    assert upretty(e2) == ucode_str
    assert latex(e2) == \
        r'\left[J_z^{2},A + B\right] \left\{E^{-2},D^{\dagger} C^{\dagger}\right\} \left[J^2,J_z\right]'
    sT(e2, "Mul(Commutator(Pow(JzOp(Symbol('J')), Integer(2)),Add(Operator(Symbol('A')), Operator(Symbol('B')))), AntiCommutator(Pow(Operator(Symbol('E')), Integer(-2)),Mul(Dagger(Operator(Symbol('D'))), Dagger(Operator(Symbol('C'))))), Commutator(J2Op(Symbol('J')),JzOp(Symbol('J'))))")
    assert str(e3) == \
        "Wigner3j(1, 2, 3, 4, 5, 6)*[Dagger(B) + A,C + D]x(-J2 + Jz)*|1,0><1,1|*(|1,0,j1=1,j2=1> + |1,1,j1=1,j2=1>)x|1,-1,j1=1,j2=1>"
    ascii_str = \
"""\
          [ +          ]  /   2     \\                                                                 \n\
/1  3  5\\*[B  + A,C + D]x |- J  + J |*|1,0><1,1|*(|1,0,j1=1,j2=1> + |1,1,j1=1,j2=1>)x |1,-1,j1=1,j2=1>\n\
|       |                 \\        z/                                                                 \n\
\\2  4  6/                                                                                             \
"""
    ucode_str = \
"""\
          ⎡ †          ⎤  ⎛   2     ⎞                                                                 \n\
⎛1  3  5⎞⋅⎣B  + A,C + D⎦⨂ ⎜- J  + J ⎟⋅❘1,0⟩⟨1,1❘⋅(❘1,0,j₁=1,j₂=1⟩ + ❘1,1,j₁=1,j₂=1⟩)⨂ ❘1,-1,j₁=1,j₂=1⟩\n\
⎜       ⎟                 ⎝        z⎠                                                                 \n\
⎝2  4  6⎠                                                                                             \
"""
    assert pretty(e3) == ascii_str
    assert upretty(e3) == ucode_str
    assert latex(e3) == \
        r'\left(\begin{array}{ccc} 1 & 3 & 5 \\ 2 & 4 & 6 \end{array}\right) {\left[B^{\dagger} + A,C + D\right]}\otimes \left({- J^2 + J_z}\right) {\left|1,0\right\rangle }{\left\langle 1,1\right|} \left({{\left|1,0,j_{1}=1,j_{2}=1\right\rangle } + {\left|1,1,j_{1}=1,j_{2}=1\right\rangle }}\right)\otimes {{\left|1,-1,j_{1}=1,j_{2}=1\right\rangle }}'
    sT(e3, "Mul(Wigner3j(Integer(1), Integer(2), Integer(3), Integer(4), Integer(5), Integer(6)), TensorProduct(Commutator(Add(Dagger(Operator(Symbol('B'))), Operator(Symbol('A'))),Add(Operator(Symbol('C')), Operator(Symbol('D')))), Add(Mul(Integer(-1), J2Op(Symbol('J'))), JzOp(Symbol('J')))), OuterProduct(JzKet(Integer(1),Integer(0)),JzBra(Integer(1),Integer(1))), TensorProduct(Add(JzKetCoupled(Integer(1),Integer(0),Tuple(Integer(1), Integer(1)),Tuple(Tuple(Integer(1), Integer(2), Integer(1)))), JzKetCoupled(Integer(1),Integer(1),Tuple(Integer(1), Integer(1)),Tuple(Tuple(Integer(1), Integer(2), Integer(1))))), JzKetCoupled(Integer(1),Integer(-1),Tuple(Integer(1), Integer(1)),Tuple(Tuple(Integer(1), Integer(2), Integer(1))))))")
    assert str(e4) == '(C(1)*C(2)+F**2)*(L2(Interval(0, oo))+H)'
    ascii_str = \
"""\
// 1    2\\    x2\\   / 2    \\\n\
\\\\C  x C / + F  / x \\L  + H/\
"""
    ucode_str = \
"""\
⎛⎛ 1    2⎞    ⨂2⎞   ⎛ 2    ⎞\n\
⎝⎝C  ⨂ C ⎠ ⊕ F  ⎠ ⨂ ⎝L  ⊕ H⎠\
"""
    assert pretty(e4) == ascii_str
    assert upretty(e4) == ucode_str
    assert latex(e4) == \
        r'\left(\left(\mathcal{C}^{1}\otimes \mathcal{C}^{2}\right)\oplus {\mathcal{F}}^{\otimes 2}\right)\otimes \left({\mathcal{L}^2}\left( \left[0, \infty\right) \right)\oplus \mathcal{H}\right)'
    sT(e4, "TensorProductHilbertSpace((DirectSumHilbertSpace(TensorProductHilbertSpace(ComplexSpace(Integer(1)),ComplexSpace(Integer(2))),TensorPowerHilbertSpace(FockSpace(),Integer(2)))),(DirectSumHilbertSpace(L2(Interval(Integer(0), oo, false, true)),HilbertSpace())))")


def _test_sho1d():
    ad = RaisingOp('a')
    assert pretty(ad) == ' \N{DAGGER}\na '
    assert latex(ad) == 'a^{\\dagger}'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\core\_internal.py ===
from numpy._core import _internal


# Build a new array from the information in a pickle.
# Note that the name numpy.core._internal._reconstruct is embedded in
# pickles of ndarrays made with NumPy before release 1.0
# so don't remove the name here, or you'll
# break backward compatibility.
def _reconstruct(subtype, shape, dtype):
    from numpy import ndarray
    return ndarray.__new__(subtype, shape, dtype)


# Pybind11 (in versions <= 2.11.1) imports _dtype_from_pep3118 from the
# _internal submodule, therefore it must be importable without a warning.
_dtype_from_pep3118 = _internal._dtype_from_pep3118

def __getattr__(attr_name):
    from numpy._core import _internal

    from ._utils import _raise_warning
    ret = getattr(_internal, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core._internal' has no attribute {attr_name}")
    _raise_warning(attr_name, "_internal")
    return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\__init__.py ===
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys

# need to add the path to the ORT flatbuffers python module before we import anything else here.
# we also auto-magically adjust to whether we're running from the ORT repo, or from within the ORT python package
script_dir = os.path.dirname(os.path.realpath(__file__))
fbs_py_schema_dirname = "ort_flatbuffers_py"
if os.path.isdir(os.path.join(script_dir, fbs_py_schema_dirname)):
    # fbs bindings are in this directory, so we're running in the ORT python package
    ort_fbs_py_parent_dir = script_dir
else:
    # running directly from ORT repo, so fbs bindings are under onnxruntime/core/flatbuffers
    ort_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
    ort_fbs_py_parent_dir = os.path.join(ort_root, "onnxruntime", "core", "flatbuffers")

sys.path.append(ort_fbs_py_parent_dir)

from .operator_type_usage_processors import (  # noqa: E402
    GloballyAllowedTypesOpTypeImplFilter,  # noqa: F401
    OperatorTypeUsageManager,  # noqa: F401
    OpTypeImplFilterInterface,  # noqa: F401
)
from .ort_model_processor import OrtFormatModelProcessor  # noqa: E402, F401
from .utils import create_config_from_models  # noqa: E402, F401

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\_numba\kernels\__init__.py ===
from pandas.core._numba.kernels.mean_ import (
    grouped_mean,
    sliding_mean,
)
from pandas.core._numba.kernels.min_max_ import (
    grouped_min_max,
    sliding_min_max,
)
from pandas.core._numba.kernels.sum_ import (
    grouped_sum,
    sliding_sum,
)
from pandas.core._numba.kernels.var_ import (
    grouped_var,
    sliding_var,
)

__all__ = [
    "sliding_mean",
    "grouped_mean",
    "sliding_sum",
    "grouped_sum",
    "sliding_var",
    "grouped_var",
    "sliding_min_max",
    "grouped_min_max",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\_libs\__init__.py ===
__all__ = [
    "NaT",
    "NaTType",
    "OutOfBoundsDatetime",
    "Period",
    "Timedelta",
    "Timestamp",
    "iNaT",
    "Interval",
]


# Below imports needs to happen first to ensure pandas top level
# module gets monkeypatched with the pandas_datetime_CAPI
# see pandas_datetime_exec in pd_datetime.c
import pandas._libs.pandas_parser  # isort: skip # type: ignore[reportUnusedImport]
import pandas._libs.pandas_datetime  # noqa: F401 # isort: skip # type: ignore[reportUnusedImport]
from pandas._libs.interval import Interval
from pandas._libs.tslibs import (
    NaT,
    NaTType,
    OutOfBoundsDatetime,
    Period,
    Timedelta,
    Timestamp,
    iNaT,
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\cli\command_context.py ===
from contextlib import ExitStack, contextmanager
from typing import ContextManager, Generator, TypeVar

_T = TypeVar("_T", covariant=True)


class CommandContextMixIn:
    def __init__(self) -> None:
        super().__init__()
        self._in_main_context = False
        self._main_context = ExitStack()

    @contextmanager
    def main_context(self) -> Generator[None, None, None]:
        assert not self._in_main_context

        self._in_main_context = True
        try:
            with self._main_context:
                yield
        finally:
            self._in_main_context = False

    def enter_context(self, context_provider: ContextManager[_T]) -> _T:
        assert self._in_main_context

        return self._main_context.enter_context(context_provider)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\resolvelib\__init__.py ===
__all__ = [
    "__version__",
    "AbstractProvider",
    "AbstractResolver",
    "BaseReporter",
    "InconsistentCandidate",
    "Resolver",
    "RequirementsConflicted",
    "ResolutionError",
    "ResolutionImpossible",
    "ResolutionTooDeep",
]

__version__ = "1.1.0"


from .providers import AbstractProvider
from .reporters import BaseReporter
from .resolvers import (
    AbstractResolver,
    InconsistentCandidate,
    RequirementsConflicted,
    ResolutionError,
    ResolutionImpossible,
    ResolutionTooDeep,
    Resolver,
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\resolvelib\resolvers\__init__.py ===
from ..structs import RequirementInformation
from .abstract import AbstractResolver, Result
from .criterion import Criterion
from .exceptions import (
    InconsistentCandidate,
    RequirementsConflicted,
    ResolutionError,
    ResolutionImpossible,
    ResolutionTooDeep,
    ResolverException,
)
from .resolution import Resolution, Resolver

__all__ = [
    "AbstractResolver",
    "InconsistentCandidate",
    "Resolver",
    "Resolution",
    "RequirementsConflicted",
    "ResolutionError",
    "ResolutionImpossible",
    "ResolutionTooDeep",
    "RequirementInformation",
    "ResolverException",
    "Result",
    "Criterion",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\clipboard\api.py ===
# -*- coding: utf-8 -*-
# *****************************************************************************
#     Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#     Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************

from pyreadline3.py3k_compat import is_ironpython

if is_ironpython:
    try:
        from .ironpython_clipboard import get_clipboard_text, set_clipboard_text
    except ImportError:
        from .no_clipboard import get_clipboard_text, set_clipboard_text

else:
    try:
        from .win32_clipboard import get_clipboard_text, set_clipboard_text
    except ImportError:
        from .no_clipboard import get_clipboard_text, set_clipboard_text

__all__ = [
    "get_clipboard_text",
    "set_clipboard_text",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\remote\utils.py ===
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

import json
from typing import Any, Union


def dump_json(json_struct: Any) -> str:
    return json.dumps(json_struct)


def load_json(s: Union[str, bytes]) -> Any:
    return json.loads(s)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_subs.py ===
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.function import (Derivative, Function, Lambda, Subs)
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, oo, pi, zoo)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.core.sympify import SympifyError
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan2, cos, cot, sin, tan)
from sympy.matrices.dense import (Matrix, zeros)
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.polys.polytools import factor
from sympy.polys.rootoftools import RootOf
from sympy.simplify.cse_main import cse
from sympy.simplify.simplify import nsimplify
from sympy.core.basic import _aresame
from sympy.testing.pytest import XFAIL, raises
from sympy.abc import a, x, y, z, t


def test_subs():
    n3 = Rational(3)
    e = x
    e = e.subs(x, n3)
    assert e == Rational(3)

    e = 2*x
    assert e == 2*x
    e = e.subs(x, n3)
    assert e == Rational(6)


def test_subs_Matrix():
    z = zeros(2)
    z1 = ZeroMatrix(2, 2)
    assert (x*y).subs({x:z, y:0}) in [z, z1]
    assert (x*y).subs({y:z, x:0}) == 0
    assert (x*y).subs({y:z, x:0}, simultaneous=True) in [z, z1]
    assert (x + y).subs({x: z, y: z}, simultaneous=True) in [z, z1]
    assert (x + y).subs({x: z, y: z}) in [z, z1]

    # Issue #15528
    assert Mul(Matrix([[3]]), x).subs(x, 2.0) == Matrix([[6.0]])
    # Does not raise a TypeError, see comment on the MatAdd postprocessor
    assert Add(Matrix([[3]]), x).subs(x, 2.0) == Add(Matrix([[3]]), 2.0)


def test_subs_AccumBounds():
    e = x
    e = e.subs(x, AccumBounds(1, 3))
    assert e == AccumBounds(1, 3)

    e = 2*x
    e = e.subs(x, AccumBounds(1, 3))
    assert e == AccumBounds(2, 6)

    e = x + x**2
    e = e.subs(x, AccumBounds(-1, 1))
    assert e == AccumBounds(-1, 2)


def test_trigonometric():
    n3 = Rational(3)
    e = (sin(x)**2).diff(x)
    assert e == 2*sin(x)*cos(x)
    e = e.subs(x, n3)
    assert e == 2*cos(n3)*sin(n3)

    e = (sin(x)**2).diff(x)
    assert e == 2*sin(x)*cos(x)
    e = e.subs(sin(x), cos(x))
    assert e == 2*cos(x)**2

    assert exp(pi).subs(exp, sin) == 0
    assert cos(exp(pi)).subs(exp, sin) == 1

    i = Symbol('i', integer=True)
    zoo = S.ComplexInfinity
    assert tan(x).subs(x, pi/2) is zoo
    assert cot(x).subs(x, pi) is zoo
    assert cot(i*x).subs(x, pi) is zoo
    assert tan(i*x).subs(x, pi/2) == tan(i*pi/2)
    assert tan(i*x).subs(x, pi/2).subs(i, 1) is zoo
    o = Symbol('o', odd=True)
    assert tan(o*x).subs(x, pi/2) == tan(o*pi/2)


def test_powers():
    assert sqrt(1 - sqrt(x)).subs(x, 4) == I
    assert (sqrt(1 - x**2)**3).subs(x, 2) == - 3*I*sqrt(3)
    assert (x**Rational(1, 3)).subs(x, 27) == 3
    assert (x**Rational(1, 3)).subs(x, -27) == 3*(-1)**Rational(1, 3)
    assert ((-x)**Rational(1, 3)).subs(x, 27) == 3*(-1)**Rational(1, 3)
    n = Symbol('n', negative=True)
    assert (x**n).subs(x, 0) is S.ComplexInfinity
    assert exp(-1).subs(S.Exp1, 0) is S.ComplexInfinity
    assert (x**(4.0*y)).subs(x**(2.0*y), n) == n**2.0
    assert (2**(x + 2)).subs(2, 3) == 3**(x + 3)


def test_logexppow():   # no eval()
    x = Symbol('x', real=True)
    w = Symbol('w')
    e = (3**(1 + x) + 2**(1 + x))/(3**x + 2**x)
    assert e.subs(2**x, w) != e
    assert e.subs(exp(x*log(Rational(2))), w) != e


def test_bug():
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    y = x1*x2
    assert y.subs(x1, Float(3.0)) == Float(3.0)*x2


def test_subbug1():
    # see that they don't fail
    (x**x).subs(x, 1)
    (x**x).subs(x, 1.0)


def test_subbug2():
    # Ensure this does not cause infinite recursion
    assert Float(7.7).epsilon_eq(abs(x).subs(x, -7.7))


def test_dict_set():
    a, b, c = map(Wild, 'abc')

    f = 3*cos(4*x)
    r = f.match(a*cos(b*x))
    assert r == {a: 3, b: 4}
    e = a/b*sin(b*x)
    assert e.subs(r) == r[a]/r[b]*sin(r[b]*x)
    assert e.subs(r) == 3*sin(4*x) / 4
    s = set(r.items())
    assert e.subs(s) == r[a]/r[b]*sin(r[b]*x)
    assert e.subs(s) == 3*sin(4*x) / 4

    assert e.subs(r) == r[a]/r[b]*sin(r[b]*x)
    assert e.subs(r) == 3*sin(4*x) / 4
    assert x.subs(Dict((x, 1))) == 1


def test_dict_ambigous():   # see issue 3566
    f = x*exp(x)
    g = z*exp(z)

    df = {x: y, exp(x): y}
    dg = {z: y, exp(z): y}

    assert f.subs(df) == y**2
    assert g.subs(dg) == y**2

    # and this is how order can affect the result
    assert f.subs(x, y).subs(exp(x), y) == y*exp(y)
    assert f.subs(exp(x), y).subs(x, y) == y**2

    # length of args and count_ops are the same so
    # default_sort_key resolves ordering...if one
    # doesn't want this result then an unordered
    # sequence should not be used.
    e = 1 + x*y
    assert e.subs({x: y, y: 2}) == 5
    # here, there are no obviously clashing keys or values
    # but the results depend on the order
    assert exp(x/2 + y).subs({exp(y + 1): 2, x: 2}) == exp(y + 1)


def test_deriv_sub_bug3():
    f = Function('f')
    pat = Derivative(f(x), x, x)
    assert pat.subs(y, y**2) == Derivative(f(x), x, x)
    assert pat.subs(y, y**2) != Derivative(f(x), x)


def test_equality_subs1():
    f = Function('f')
    eq = Eq(f(x)**2, x)
    res = Eq(Integer(16), x)
    assert eq.subs(f(x), 4) == res


def test_equality_subs2():
    f = Function('f')
    eq = Eq(f(x)**2, 16)
    assert bool(eq.subs(f(x), 3)) is False
    assert bool(eq.subs(f(x), 4)) is True


def test_issue_3742():
    e = sqrt(x)*exp(y)
    assert e.subs(sqrt(x), 1) == exp(y)


def test_subs_dict1():
    assert (1 + x*y).subs(x, pi) == 1 + pi*y
    assert (1 + x*y).subs({x: pi, y: 2}) == 1 + 2*pi

    c2, c3, q1p, q2p, c1, s1, s2, s3 = symbols('c2 c3 q1p q2p c1 s1 s2 s3')
    test = (c2**2*q2p*c3 + c1**2*s2**2*q2p*c3 + s1**2*s2**2*q2p*c3
            - c1**2*q1p*c2*s3 - s1**2*q1p*c2*s3)
    assert (test.subs({c1**2: 1 - s1**2, c2**2: 1 - s2**2, c3**3: 1 - s3**2})
        == c3*q2p*(1 - s2**2) + c3*q2p*s2**2*(1 - s1**2)
            - c2*q1p*s3*(1 - s1**2) + c3*q2p*s1**2*s2**2 - c2*q1p*s3*s1**2)


def test_mul():
    x, y, z, a, b, c = symbols('x y z a b c')
    A, B, C = symbols('A B C', commutative=0)
    assert (x*y*z).subs(z*x, y) == y**2
    assert (z*x).subs(1/x, z) == 1
    assert (x*y/z).subs(1/z, a) == a*x*y
    assert (x*y/z).subs(x/z, a) == a*y
    assert (x*y/z).subs(y/z, a) == a*x
    assert (x*y/z).subs(x/z, 1/a) == y/a
    assert (x*y/z).subs(x, 1/a) == y/(z*a)
    assert (2*x*y).subs(5*x*y, z) != z*Rational(2, 5)
    assert (x*y*A).subs(x*y, a) == a*A
    assert (x**2*y**(x*Rational(3, 2))).subs(x*y**(x/2), 2) == 4*y**(x/2)
    assert (x*exp(x*2)).subs(x*exp(x), 2) == 2*exp(x)
    assert ((x**(2*y))**3).subs(x**y, 2) == 64
    assert (x*A*B).subs(x*A, y) == y*B
    assert (x*y*(1 + x)*(1 + x*y)).subs(x*y, 2) == 6*(1 + x)
    assert ((1 + A*B)*A*B).subs(A*B, x*A*B)
    assert (x*a/z).subs(x/z, A) == a*A
    assert (x**3*A).subs(x**2*A, a) == a*x
    assert (x**2*A*B).subs(x**2*B, a) == a*A
    assert (x**2*A*B).subs(x**2*A, a) == a*B
    assert (b*A**3/(a**3*c**3)).subs(a**4*c**3*A**3/b**4, z) == \
        b*A**3/(a**3*c**3)
    assert (6*x).subs(2*x, y) == 3*y
    assert (y*exp(x*Rational(3, 2))).subs(y*exp(x), 2) == 2*exp(x/2)
    assert (y*exp(x*Rational(3, 2))).subs(y*exp(x), 2) == 2*exp(x/2)
    assert (A**2*B*A**2*B*A**2).subs(A*B*A, C) == A*C**2*A
    assert (x*A**3).subs(x*A, y) == y*A**2
    assert (x**2*A**3).subs(x*A, y) == y**2*A
    assert (x*A**3).subs(x*A, B) == B*A**2
    assert (x*A*B*A*exp(x*A*B)).subs(x*A, B) == B**2*A*exp(B*B)
    assert (x**2*A*B*A*exp(x*A*B)).subs(x*A, B) == B**3*exp(B**2)
    assert (x**3*A*exp(x*A*B)*A*exp(x*A*B)).subs(x*A, B) == \
        x*B*exp(B**2)*B*exp(B**2)
    assert (x*A*B*C*A*B).subs(x*A*B, C) == C**2*A*B
    assert (-I*a*b).subs(a*b, 2) == -2*I

    # issue 6361
    assert (-8*I*a).subs(-2*a, 1) == 4*I
    assert (-I*a).subs(-a, 1) == I

    # issue 6441
    assert (4*x**2).subs(2*x, y) == y**2
    assert (2*4*x**2).subs(2*x, y) == 2*y**2
    assert (-x**3/9).subs(-x/3, z) == -z**2*x
    assert (-x**3/9).subs(x/3, z) == -z**2*x
    assert (-2*x**3/9).subs(x/3, z) == -2*x*z**2
    assert (-2*x**3/9).subs(-x/3, z) == -2*x*z**2
    assert (-2*x**3/9).subs(-2*x, z) == z*x**2/9
    assert (-2*x**3/9).subs(2*x, z) == -z*x**2/9
    assert (2*(3*x/5/7)**2).subs(3*x/5, z) == 2*(Rational(1, 7))**2*z**2
    assert (4*x).subs(-2*x, z) == 4*x  # try keep subs literal


def test_subs_simple():
    a = symbols('a', commutative=True)
    x = symbols('x', commutative=False)

    assert (2*a).subs(1, 3) == 2*a
    assert (2*a).subs(2, 3) == 3*a
    assert (2*a).subs(a, 3) == 6
    assert sin(2).subs(1, 3) == sin(2)
    assert sin(2).subs(2, 3) == sin(3)
    assert sin(a).subs(a, 3) == sin(3)

    assert (2*x).subs(1, 3) == 2*x
    assert (2*x).subs(2, 3) == 3*x
    assert (2*x).subs(x, 3) == 6
    assert sin(x).subs(x, 3) == sin(3)


def test_subs_constants():
    a, b = symbols('a b', commutative=True)
    x, y = symbols('x y', commutative=False)

    assert (a*b).subs(2*a, 1) == a*b
    assert (1.5*a*b).subs(a, 1) == 1.5*b
    assert (2*a*b).subs(2*a, 1) == b
    assert (2*a*b).subs(4*a, 1) == 2*a*b

    assert (x*y).subs(2*x, 1) == x*y
    assert (1.5*x*y).subs(x, 1) == 1.5*y
    assert (2*x*y).subs(2*x, 1) == y
    assert (2*x*y).subs(4*x, 1) == 2*x*y


def test_subs_commutative():
    a, b, c, d, K = symbols('a b c d K', commutative=True)

    assert (a*b).subs(a*b, K) == K
    assert (a*b*a*b).subs(a*b, K) == K**2
    assert (a*a*b*b).subs(a*b, K) == K**2
    assert (a*b*c*d).subs(a*b*c, K) == d*K
    assert (a*b**c).subs(a, K) == K*b**c
    assert (a*b**c).subs(b, K) == a*K**c
    assert (a*b**c).subs(c, K) == a*b**K
    assert (a*b*c*b*a).subs(a*b, K) == c*K**2
    assert (a**3*b**2*a).subs(a*b, K) == a**2*K**2


def test_subs_noncommutative():
    w, x, y, z, L = symbols('w x y z L', commutative=False)
    alpha = symbols('alpha', commutative=True)
    someint = symbols('someint', commutative=True, integer=True)

    assert (x*y).subs(x*y, L) == L
    assert (w*y*x).subs(x*y, L) == w*y*x
    assert (w*x*y*z).subs(x*y, L) == w*L*z
    assert (x*y*x*y).subs(x*y, L) == L**2
    assert (x*x*y).subs(x*y, L) == x*L
    assert (x*x*y*y).subs(x*y, L) == x*L*y
    assert (w*x*y).subs(x*y*z, L) == w*x*y
    assert (x*y**z).subs(x, L) == L*y**z
    assert (x*y**z).subs(y, L) == x*L**z
    assert (x*y**z).subs(z, L) == x*y**L
    assert (w*x*y*z*x*y).subs(x*y*z, L) == w*L*x*y
    assert (w*x*y*y*w*x*x*y*x*y*y*x*y).subs(x*y, L) == w*L*y*w*x*L**2*y*L

    # Check fractional power substitutions. It should not do
    # substitutions that choose a value for noncommutative log,
    # or inverses that don't already appear in the expressions.
    assert (x*x*x).subs(x*x, L) == L*x
    assert (x*x*x*y*x*x*x*x).subs(x*x, L) == L*x*y*L**2
    for p in range(1, 5):
        for k in range(10):
            assert (y * x**k).subs(x**p, L) == y * L**(k//p) * x**(k % p)
    assert (x**Rational(3, 2)).subs(x**S.Half, L) == x**Rational(3, 2)
    assert (x**S.Half).subs(x**S.Half, L) == L
    assert (x**Rational(-1, 2)).subs(x**S.Half, L) == x**Rational(-1, 2)
    assert (x**Rational(-1, 2)).subs(x**Rational(-1, 2), L) == L

    assert (x**(2*someint)).subs(x**someint, L) == L**2
    assert (x**(2*someint + 3)).subs(x**someint, L) == L**2*x**3
    assert (x**(3*someint + 3)).subs(x**someint, L) == L**3*x**3
    assert (x**(3*someint)).subs(x**(2*someint), L) == L * x**someint
    assert (x**(4*someint)).subs(x**(2*someint), L) == L**2
    assert (x**(4*someint + 1)).subs(x**(2*someint), L) == L**2 * x
    assert (x**(4*someint)).subs(x**(3*someint), L) == L * x**someint
    assert (x**(4*someint + 1)).subs(x**(3*someint), L) == L * x**(someint + 1)

    assert (x**(2*alpha)).subs(x**alpha, L) == x**(2*alpha)
    assert (x**(2*alpha + 2)).subs(x**2, L) == x**(2*alpha + 2)
    assert ((2*z)**alpha).subs(z**alpha, y) == (2*z)**alpha
    assert (x**(2*someint*alpha)).subs(x**someint, L) == x**(2*someint*alpha)
    assert (x**(2*someint + alpha)).subs(x**someint, L) == x**(2*someint + alpha)

    # This could in principle be substituted, but is not currently
    # because it requires recognizing that someint**2 is divisible by
    # someint.
    assert (x**(someint**2 + 3)).subs(x**someint, L) == x**(someint**2 + 3)

    # alpha**z := exp(log(alpha) z) is usually well-defined
    assert (4**z).subs(2**z, y) == y**2

    # Negative powers
    assert (x**(-1)).subs(x**3, L) == x**(-1)
    assert (x**(-2)).subs(x**3, L) == x**(-2)
    assert (x**(-3)).subs(x**3, L) == L**(-1)
    assert (x**(-4)).subs(x**3, L) == L**(-1) * x**(-1)
    assert (x**(-5)).subs(x**3, L) == L**(-1) * x**(-2)

    assert (x**(-1)).subs(x**(-3), L) == x**(-1)
    assert (x**(-2)).subs(x**(-3), L) == x**(-2)
    assert (x**(-3)).subs(x**(-3), L) == L
    assert (x**(-4)).subs(x**(-3), L) == L * x**(-1)
    assert (x**(-5)).subs(x**(-3), L) == L * x**(-2)

    assert (x**1).subs(x**(-3), L) == x
    assert (x**2).subs(x**(-3), L) == x**2
    assert (x**3).subs(x**(-3), L) == L**(-1)
    assert (x**4).subs(x**(-3), L) == L**(-1) * x
    assert (x**5).subs(x**(-3), L) == L**(-1) * x**2


def test_subs_basic_funcs():
    a, b, c, d, K = symbols('a b c d K', commutative=True)
    w, x, y, z, L = symbols('w x y z L', commutative=False)

    assert (x + y).subs(x + y, L) == L
    assert (x - y).subs(x - y, L) == L
    assert (x/y).subs(x, L) == L/y
    assert (x**y).subs(x, L) == L**y
    assert (x**y).subs(y, L) == x**L
    assert ((a - c)/b).subs(b, K) == (a - c)/K
    assert (exp(x*y - z)).subs(x*y, L) == exp(L - z)
    assert (a*exp(x*y - w*z) + b*exp(x*y + w*z)).subs(z, 0) == \
        a*exp(x*y) + b*exp(x*y)
    assert ((a - b)/(c*d - a*b)).subs(c*d - a*b, K) == (a - b)/K
    assert (w*exp(a*b - c)*x*y/4).subs(x*y, L) == w*exp(a*b - c)*L/4


def test_subs_wild():
    R, S, T, U = symbols('R S T U', cls=Wild)

    assert (R*S).subs(R*S, T) == T
    assert (S*R).subs(R*S, T) == T
    assert (R + S).subs(R + S, T) == T
    assert (R**S).subs(R, T) == T**S
    assert (R**S).subs(S, T) == R**T
    assert (R*S**T).subs(R, U) == U*S**T
    assert (R*S**T).subs(S, U) == R*U**T
    assert (R*S**T).subs(T, U) == R*S**U


def test_subs_mixed():
    a, b, c, d, K = symbols('a b c d K', commutative=True)
    w, x, y, z, L = symbols('w x y z L', commutative=False)
    R, S, T, U = symbols('R S T U', cls=Wild)

    assert (a*x*y).subs(x*y, L) == a*L
    assert (a*b*x*y*x).subs(x*y, L) == a*b*L*x
    assert (R*x*y*exp(x*y)).subs(x*y, L) == R*L*exp(L)
    assert (a*x*y*y*x - x*y*z*exp(a*b)).subs(x*y, L) == a*L*y*x - L*z*exp(a*b)
    e = c*y*x*y*x**(R*S - a*b) - T*(a*R*b*S)
    assert e.subs(x*y, L).subs(a*b, K).subs(R*S, U) == \
        c*y*L*x**(U - K) - T*(U*K)


def test_division():
    a, b, c = symbols('a b c', commutative=True)
    x, y, z = symbols('x y z', commutative=True)

    assert (1/a).subs(a, c) == 1/c
    assert (1/a**2).subs(a, c) == 1/c**2
    assert (1/a**2).subs(a, -2) == Rational(1, 4)
    assert (-(1/a**2)).subs(a, -2) == Rational(-1, 4)

    assert (1/x).subs(x, z) == 1/z
    assert (1/x**2).subs(x, z) == 1/z**2
    assert (1/x**2).subs(x, -2) == Rational(1, 4)
    assert (-(1/x**2)).subs(x, -2) == Rational(-1, 4)

    #issue 5360
    assert (1/x).subs(x, 0) == 1/S.Zero


def test_add():
    a, b, c, d, x, y, t = symbols('a b c d x y t')

    assert (a**2 - b - c).subs(a**2 - b, d) in [d - c, a**2 - b - c]
    assert (a**2 - c).subs(a**2 - c, d) == d
    assert (a**2 - b - c).subs(a**2 - c, d) in [d - b, a**2 - b - c]
    assert (a**2 - x - c).subs(a**2 - c, d) in [d - x, a**2 - x - c]
    assert (a**2 - b - sqrt(a)).subs(a**2 - sqrt(a), c) == c - b
    assert (a + b + exp(a + b)).subs(a + b, c) == c + exp(c)
    assert (c + b + exp(c + b)).subs(c + b, a) == a + exp(a)
    assert (a + b + c + d).subs(b + c, x) == a + d + x
    assert (a + b + c + d).subs(-b - c, x) == a + d - x
    assert ((x + 1)*y).subs(x + 1, t) == t*y
    assert ((-x - 1)*y).subs(x + 1, t) == -t*y
    assert ((x - 1)*y).subs(x + 1, t) == y*(t - 2)
    assert ((-x + 1)*y).subs(x + 1, t) == y*(-t + 2)

    # this should work every time:
    e = a**2 - b - c
    assert e.subs(Add(*e.args[:2]), d) == d + e.args[2]
    assert e.subs(a**2 - c, d) == d - b

    # the fallback should recognize when a change has
    # been made; while .1 == Rational(1, 10) they are not the same
    # and the change should be made
    assert (0.1 + a).subs(0.1, Rational(1, 10)) == Rational(1, 10) + a

    e = (-x*(-y + 1) - y*(y - 1))
    ans = (-x*(x) - y*(-x)).expand()
    assert e.subs(-y + 1, x) == ans

    #Test issue 18747
    assert (exp(x) + cos(x)).subs(x, oo) == oo
    assert Add(*[AccumBounds(-1, 1), oo]) == oo
    assert Add(*[oo, AccumBounds(-1, 1)]) == oo


def test_subs_issue_4009():
    assert (I*Symbol('a')).subs(1, 2) == I*Symbol('a')


def test_functions_subs():
    f, g = symbols('f g', cls=Function)
    l = Lambda((x, y), sin(x) + y)
    assert (g(y, x) + cos(x)).subs(g, l) == sin(y) + x + cos(x)
    assert (f(x)**2).subs(f, sin) == sin(x)**2
    assert (f(x, y)).subs(f, log) == log(x, y)
    assert (f(x, y)).subs(f, sin) == f(x, y)
    assert (sin(x) + atan2(x, y)).subs([[atan2, f], [sin, g]]) == \
        f(x, y) + g(x)
    assert (g(f(x + y, x))).subs([[f, l], [g, exp]]) == exp(x + sin(x + y))


def test_derivative_subs():
    f = Function('f')
    g = Function('g')
    assert Derivative(f(x), x).subs(f(x), y) != 0
    # need xreplace to put the function back, see #13803
    assert Derivative(f(x), x).subs(f(x), y).xreplace({y: f(x)}) == \
        Derivative(f(x), x)
    # issues 5085, 5037
    assert cse(Derivative(f(x), x) + f(x))[1][0].has(Derivative)
    assert cse(Derivative(f(x, y), x) +
               Derivative(f(x, y), y))[1][0].has(Derivative)
    eq = Derivative(g(x), g(x))
    assert eq.subs(g, f) == Derivative(f(x), f(x))
    assert eq.subs(g(x), f(x)) == Derivative(f(x), f(x))
    assert eq.subs(g, cos) == Subs(Derivative(y, y), y, cos(x))


def test_derivative_subs2():
    f_func, g_func = symbols('f g', cls=Function)
    f, g = f_func(x, y, z), g_func(x, y, z)
    assert Derivative(f, x, y).subs(Derivative(f, x, y), g) == g
    assert Derivative(f, y, x).subs(Derivative(f, x, y), g) == g
    assert Derivative(f, x, y).subs(Derivative(f, x), g) == Derivative(g, y)
    assert Derivative(f, x, y).subs(Derivative(f, y), g) == Derivative(g, x)
    assert (Derivative(f, x, y, z).subs(
                Derivative(f, x, z), g) == Derivative(g, y))
    assert (Derivative(f, x, y, z).subs(
                Derivative(f, z, y), g) == Derivative(g, x))
    assert (Derivative(f, x, y, z).subs(
                Derivative(f, z, y, x), g) == g)

    # Issue 9135
    assert (Derivative(f, x, x, y).subs(
                Derivative(f, y, y), g) == Derivative(f, x, x, y))
    assert (Derivative(f, x, y, y, z).subs(
                Derivative(f, x, y, y, y), g) == Derivative(f, x, y, y, z))

    assert Derivative(f, x, y).subs(Derivative(f_func(x), x, y), g) == Derivative(f, x, y)


def test_derivative_subs3():
    dex = Derivative(exp(x), x)
    assert Derivative(dex, x).subs(dex, exp(x)) == dex
    assert dex.subs(exp(x), dex) == Derivative(exp(x), x, x)


def test_issue_5284():
    A, B = symbols('A B', commutative=False)
    assert (x*A).subs(x**2*A, B) == x*A
    assert (A**2).subs(A**3, B) == A**2
    assert (A**6).subs(A**3, B) == B**2


def test_subs_iter():
    assert x.subs(reversed([[x, y]])) == y
    it = iter([[x, y]])
    assert x.subs(it) == y
    assert x.subs(Tuple((x, y))) == y


def test_subs_dict():
    a, b, c, d, e = symbols('a b c d e')

    assert (2*x + y + z).subs({"x": 1, "y": 2}) == 4 + z

    l = [(sin(x), 2), (x, 1)]
    assert (sin(x)).subs(l) == \
           (sin(x)).subs(dict(l)) == 2
    assert sin(x).subs(reversed(l)) == sin(1)

    expr = sin(2*x) + sqrt(sin(2*x))*cos(2*x)*sin(exp(x)*x)
    reps = {sin(2*x): c,
               sqrt(sin(2*x)): a,
               cos(2*x): b,
               exp(x): e,
               x: d,}
    assert expr.subs(reps) == c + a*b*sin(d*e)

    l = [(x, 3), (y, x**2)]
    assert (x + y).subs(l) == 3 + x**2
    assert (x + y).subs(reversed(l)) == 12

    # If changes are made to convert lists into dictionaries and do
    # a dictionary-lookup replacement, these tests will help to catch
    # some logical errors that might occur
    l = [(y, z + 2), (1 + z, 5), (z, 2)]
    assert (y - 1 + 3*x).subs(l) == 5 + 3*x
    l = [(y, z + 2), (z, 3)]
    assert (y - 2).subs(l) == 3


def test_no_arith_subs_on_floats():
    assert (x + 3).subs(x + 3, a) == a
    assert (x + 3).subs(x + 2, a) == a + 1

    assert (x + y + 3).subs(x + 3, a) == a + y
    assert (x + y + 3).subs(x + 2, a) == a + y + 1

    assert (x + 3.0).subs(x + 3.0, a) == a
    assert (x + 3.0).subs(x + 2.0, a) == x + 3.0

    assert (x + y + 3.0).subs(x + 3.0, a) == a + y
    assert (x + y + 3.0).subs(x + 2.0, a) == x + y + 3.0


def test_issue_5651():
    a, b, c, K = symbols('a b c K', commutative=True)
    assert (a/(b*c)).subs(b*c, K) == a/K
    assert (a/(b**2*c**3)).subs(b*c, K) == a/(c*K**2)
    assert (1/(x*y)).subs(x*y, 2) == S.Half
    assert ((1 + x*y)/(x*y)).subs(x*y, 1) == 2
    assert (x*y*z).subs(x*y, 2) == 2*z
    assert ((1 + x*y)/(x*y)/z).subs(x*y, 1) == 2/z


def test_issue_6075():
    assert Tuple(1, True).subs(1, 2) == Tuple(2, True)


def test_issue_6079():
    # since x + 2.0 == x + 2 we can't do a simple equality test
    assert _aresame((x + 2.0).subs(2, 3), x + 2.0)
    assert _aresame((x + 2.0).subs(2.0, 3), x + 3)
    assert not _aresame(x + 2, x + 2.0)
    assert not _aresame(Basic(cos(x), S(1)), Basic(cos(x), S(1.)))
    assert _aresame(cos, cos)
    assert not _aresame(1, S.One)
    assert not _aresame(x, symbols('x', positive=True))


def test_issue_4680():
    N = Symbol('N')
    assert N.subs({"N": 3}) == 3


def test_issue_6158():
    assert (x - 1).subs(1, y) == x - y
    assert (x - 1).subs(-1, y) == x + y
    assert (x - oo).subs(oo, y) == x - y
    assert (x - oo).subs(-oo, y) == x + y


def test_Function_subs():
    f, g, h, i = symbols('f g h i', cls=Function)
    p = Piecewise((g(f(x, y)), x < -1), (g(x), x <= 1))
    assert p.subs(g, h) == Piecewise((h(f(x, y)), x < -1), (h(x), x <= 1))
    assert (f(y) + g(x)).subs({f: h, g: i}) == i(x) + h(y)


def test_simultaneous_subs():
    reps = {x: 0, y: 0}
    assert (x/y).subs(reps) != (y/x).subs(reps)
    assert (x/y).subs(reps, simultaneous=True) == \
        (y/x).subs(reps, simultaneous=True)
    reps = reps.items()
    assert (x/y).subs(reps) != (y/x).subs(reps)
    assert (x/y).subs(reps, simultaneous=True) == \
        (y/x).subs(reps, simultaneous=True)
    assert Derivative(x, y, z).subs(reps, simultaneous=True) == \
        Subs(Derivative(0, y, z), y, 0)


def test_issue_6419_6421():
    assert (1/(1 + x/y)).subs(x/y, x) == 1/(1 + x)
    assert (-2*I).subs(2*I, x) == -x
    assert (-I*x).subs(I*x, x) == -x
    assert (-3*I*y**4).subs(3*I*y**2, x) == -x*y**2


def test_issue_6559():
    assert (-12*x + y).subs(-x, 1) == 12 + y
    # though this involves cse it generated a failure in Mul._eval_subs
    x0, x1 = symbols('x0 x1')
    e = -log(-12*sqrt(2) + 17)/24 - log(-2*sqrt(2) + 3)/12 + sqrt(2)/3
    # XXX modify cse so x1 is eliminated and x0 = -sqrt(2)?
    assert cse(e) == (
        [(x0, sqrt(2))], [x0/3 - log(-12*x0 + 17)/24 - log(-2*x0 + 3)/12])


def test_issue_5261():
    x = symbols('x', real=True)
    e = I*x
    assert exp(e).subs(exp(x), y) == y**I
    assert (2**e).subs(2**x, y) == y**I
    eq = (-2)**e
    assert eq.subs((-2)**x, y) == eq


def test_issue_6923():
    assert (-2*x*sqrt(2)).subs(2*x, y) == -sqrt(2)*y


def test_2arg_hack():
    N = Symbol('N', commutative=False)
    ans = Mul(2, y + 1, evaluate=False)
    assert (2*x*(y + 1)).subs(x, 1, hack2=True) == ans
    assert (2*(y + 1 + N)).subs(N, 0, hack2=True) == ans


@XFAIL
def test_mul2():
    """When this fails, remove things labelled "2-arg hack"
    1) remove special handling in the fallback of subs that
    was added in the same commit as this test
    2) remove the special handling in Mul.flatten
    """
    assert (2*(x + 1)).is_Mul


def test_noncommutative_subs():
    x,y = symbols('x,y', commutative=False)
    assert (x*y*x).subs([(x, x*y), (y, x)], simultaneous=True) == (x*y*x**2*y)


def test_issue_2877():
    f = Float(2.0)
    assert (x + f).subs({f: 2}) == x + 2

    def r(a, b, c):
        return factor(a*x**2 + b*x + c)
    e = r(5.0/6, 10, 5)
    assert nsimplify(e) == 5*x**2/6 + 10*x + 5


def test_issue_5910():
    t = Symbol('t')
    assert (1/(1 - t)).subs(t, 1) is zoo
    n = t
    d = t - 1
    assert (n/d).subs(t, 1) is zoo
    assert (-n/-d).subs(t, 1) is zoo


def test_issue_5217():
    s = Symbol('s')
    z = (1 - 2*x*x)
    w = (1 + 2*x*x)
    q = 2*x*x*2*y*y
    sub = {2*x*x: s}
    assert w.subs(sub) == 1 + s
    assert z.subs(sub) == 1 - s
    assert q == 4*x**2*y**2
    assert q.subs(sub) == 2*y**2*s


def test_issue_10829():
    assert (4**x).subs(2**x, y) == y**2
    assert (9**x).subs(3**x, y) == y**2


def test_pow_eval_subs_no_cache():
    # Tests pull request 9376 is working
    from sympy.core.cache import clear_cache

    s = 1/sqrt(x**2)
    # This bug only appeared when the cache was turned off.
    # We need to approximate running this test without the cache.
    # This creates approximately the same situation.
    clear_cache()

    # This used to fail with a wrong result.
    # It incorrectly returned 1/sqrt(x**2) before this pull request.
    result = s.subs(sqrt(x**2), y)
    assert result == 1/y


def test_RootOf_issue_10092():
    x = Symbol('x', real=True)
    eq = x**3 - 17*x**2 + 81*x - 118
    r = RootOf(eq, 0)
    assert (x < r).subs(x, r) is S.false


def test_issue_8886():
    from sympy.physics.mechanics import ReferenceFrame as R
    # if something can't be sympified we assume that it
    # doesn't play well with SymPy and disallow the
    # substitution
    v = R('A').x
    raises(SympifyError, lambda: x.subs(x, v))
    raises(SympifyError, lambda: v.subs(v, x))
    assert v.__eq__(x) is False


def test_issue_12657():
    # treat -oo like the atom that it is
    reps = [(-oo, 1), (oo, 2)]
    assert (x < -oo).subs(reps) == (x < 1)
    assert (x < -oo).subs(list(reversed(reps))) == (x < 1)
    reps = [(-oo, 2), (oo, 1)]
    assert (x < oo).subs(reps) == (x < 1)
    assert (x < oo).subs(list(reversed(reps))) == (x < 1)


def test_recurse_Application_args():
    F = Lambda((x, y), exp(2*x + 3*y))
    f = Function('f')
    A = f(x, f(x, x))
    C = F(x, F(x, x))
    assert A.subs(f, F) == A.replace(f, F) == C


def test_Subs_subs():
    assert Subs(x*y, x, x).subs(x, y) == Subs(x*y, x, y)
    assert Subs(x*y, x, x + 1).subs(x, y) == \
        Subs(x*y, x, y + 1)
    assert Subs(x*y, y, x + 1).subs(x, y) == \
        Subs(y**2, y, y + 1)
    a = Subs(x*y*z, (y, x, z), (x + 1, x + z, x))
    b = Subs(x*y*z, (y, x, z), (x + 1, y + z, y))
    assert a.subs(x, y) == b and \
        a.doit().subs(x, y) == a.subs(x, y).doit()
    f = Function('f')
    g = Function('g')
    assert Subs(2*f(x, y) + g(x), f(x, y), 1).subs(y, 2) == Subs(
        2*f(x, y) + g(x), (f(x, y), y), (1, 2))


def test_issue_13333():
    eq = 1/x
    assert eq.subs({"x": '1/2'}) == 2
    assert eq.subs({"x": '(1/2)'}) == 2


def test_issue_15234():
    x, y = symbols('x y', real=True)
    p = 6*x**5 + x**4 - 4*x**3 + 4*x**2 - 2*x + 3
    p_subbed = 6*x**5 - 4*x**3 - 2*x + y**4 + 4*y**2 + 3
    assert p.subs([(x**i, y**i) for i in [2, 4]]) == p_subbed
    x, y = symbols('x y', complex=True)
    p = 6*x**5 + x**4 - 4*x**3 + 4*x**2 - 2*x + 3
    p_subbed = 6*x**5 - 4*x**3 - 2*x + y**4 + 4*y**2 + 3
    assert p.subs([(x**i, y**i) for i in [2, 4]]) == p_subbed


def test_issue_6976():
    x, y = symbols('x y')
    assert (sqrt(x)**3 + sqrt(x) + x + x**2).subs(sqrt(x), y) == \
        y**4 + y**3 + y**2 + y
    assert (x**4 + x**3 + x**2 + x + sqrt(x)).subs(x**2, y) == \
        sqrt(x) + x**3 + x + y**2 + y
    assert x.subs(x**3, y) == x
    assert x.subs(x**Rational(1, 3), y) == y**3

    # More substitutions are possible with nonnegative symbols
    x, y = symbols('x y', nonnegative=True)
    assert (x**4 + x**3 + x**2 + x + sqrt(x)).subs(x**2, y) == \
        y**Rational(1, 4) + y**Rational(3, 2) + sqrt(y) + y**2 + y
    assert x.subs(x**3, y) == y**Rational(1, 3)


def test_issue_11746():
    assert (1/x).subs(x**2, 1) == 1/x
    assert (1/(x**3)).subs(x**2, 1) == x**(-3)
    assert (1/(x**4)).subs(x**2, 1) == 1
    assert (1/(x**3)).subs(x**4, 1) == x**(-3)
    assert (1/(y**5)).subs(x**5, 1) == y**(-5)


def test_issue_17823():
    from sympy.physics.mechanics import dynamicsymbols
    q1, q2 = dynamicsymbols('q1, q2')
    expr = q1.diff().diff()**2*q1 + q1.diff()*q2.diff()
    reps={q1: a, q1.diff(): a*x*y, q1.diff().diff(): z}
    assert expr.subs(reps) == a*x*y*Derivative(q2, t) + a*z**2


def test_issue_19326():
    x, y = [i(t) for i in map(Function, 'xy')]
    assert (x*y).subs({x: 1 + x, y: x}) == (1 + x)*x


def test_issue_19558():
    e = (7*x*cos(x) - 12*log(x)**3)*(-log(x)**4 + 2*sin(x) + 1)**2/ \
    (2*(x*cos(x) - 2*log(x)**3)*(3*log(x)**4 - 7*sin(x) + 3)**2)

    assert e.subs(x, oo) == AccumBounds(-oo, oo)
    assert (sin(x) + cos(x)).subs(x, oo) == AccumBounds(-2, 2)


def test_issue_22033():
    xr = Symbol('xr', real=True)
    e = (1/xr)
    assert e.subs(xr**2, y) == e


def test_guard_against_indeterminate_evaluation():
    eq = x**y
    assert eq.subs([(x, 1), (y, oo)]) == 1  # because 1**y == 1
    assert eq.subs([(y, oo), (x, 1)]) is S.NaN
    assert eq.subs({x: 1, y: oo}) is S.NaN
    assert eq.subs([(x, 1), (y, oo)], simultaneous=True) is S.NaN

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\units\systems\natural.py ===
"""
Naturalunit system.

The natural system comes from "setting c = 1, hbar = 1". From the computer
point of view it means that we use velocity and action instead of length and
time. Moreover instead of mass we use energy.
"""

from sympy.physics.units import DimensionSystem
from sympy.physics.units.definitions import c, eV, hbar
from sympy.physics.units.definitions.dimension_definitions import (
    action, energy, force, frequency, length, mass, momentum,
    power, time, velocity)
from sympy.physics.units.prefixes import PREFIXES, prefix_unit
from sympy.physics.units.unitsystem import UnitSystem


# dimension system
_natural_dim = DimensionSystem(
    base_dims=(action, energy, velocity),
    derived_dims=(length, mass, time, momentum, force, power, frequency)
)

units = prefix_unit(eV, PREFIXES)

# unit system
natural = UnitSystem(base_units=(hbar, eV, c), units=units, name="Natural system")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\numberfields\__init__.py ===
"""Computational algebraic field theory. """

__all__ = [
    'minpoly', 'minimal_polynomial',

    'field_isomorphism', 'primitive_element', 'to_number_field',

    'isolate',

    'round_two',

    'prime_decomp', 'prime_valuation',

    'galois_group',
]

from .minpoly import minpoly, minimal_polynomial

from .subfield import field_isomorphism, primitive_element, to_number_field

from .utilities import isolate

from .basis import round_two

from .primes import prime_decomp, prime_valuation

from .galoisgroups import galois_group

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_sympify.py ===
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import (Function, Lambda)
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, pi, oo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.logic.boolalg import (false, Or, true, Xor)
from sympy.matrices.dense import Matrix
from sympy.parsing.sympy_parser import null
from sympy.polys.polytools import Poly
from sympy.printing.repr import srepr
from sympy.sets.fancysets import Range
from sympy.sets.sets import Interval
from sympy.abc import x, y
from sympy.core.sympify import (sympify, _sympify, SympifyError, kernS,
    CantSympify, converter)
from sympy.core.decorators import _sympifyit
from sympy.external import import_module
from sympy.testing.pytest import raises, XFAIL, skip
from sympy.utilities.decorator import conserve_mpmath_dps
from sympy.geometry import Point, Line
from sympy.functions.combinatorial.factorials import factorial, factorial2
from sympy.abc import _clash, _clash1, _clash2
from sympy.external.gmpy import gmpy as _gmpy, flint as _flint
from sympy.sets import FiniteSet, EmptySet
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray

import mpmath
from collections import defaultdict, OrderedDict


numpy = import_module('numpy')


def test_issue_3538():
    v = sympify("exp(x)")
    assert v == exp(x)
    assert type(v) == type(exp(x))
    assert str(type(v)) == str(type(exp(x)))


def test_sympify1():
    assert sympify("x") == Symbol("x")
    assert sympify("   x") == Symbol("x")
    assert sympify("   x   ") == Symbol("x")
    # issue 4877
    assert sympify('--.5') == 0.5
    assert sympify('-1/2') == -S.Half
    assert sympify('-+--.5') == -0.5
    assert sympify('-.[3]') == Rational(-1, 3)
    assert sympify('.[3]') == Rational(1, 3)
    assert sympify('+.[3]') == Rational(1, 3)
    assert sympify('+0.[3]*10**-2') == Rational(1, 300)
    assert sympify('.[052631578947368421]') == Rational(1, 19)
    assert sympify('.0[526315789473684210]') == Rational(1, 19)
    assert sympify('.034[56]') == Rational(1711, 49500)
    # options to make reals into rationals
    assert sympify('1.22[345]', rational=True) == \
        1 + Rational(22, 100) + Rational(345, 99900)
    assert sympify('2/2.6', rational=True) == Rational(10, 13)
    assert sympify('2.6/2', rational=True) == Rational(13, 10)
    assert sympify('2.6e2/17', rational=True) == Rational(260, 17)
    assert sympify('2.6e+2/17', rational=True) == Rational(260, 17)
    assert sympify('2.6e-2/17', rational=True) == Rational(26, 17000)
    assert sympify('2.1+3/4', rational=True) == \
        Rational(21, 10) + Rational(3, 4)
    assert sympify('2.234456', rational=True) == Rational(279307, 125000)
    assert sympify('2.234456e23', rational=True) == 223445600000000000000000
    assert sympify('2.234456e-23', rational=True) == \
        Rational(279307, 12500000000000000000000000000)
    assert sympify('-2.234456e-23', rational=True) == \
        Rational(-279307, 12500000000000000000000000000)
    assert sympify('12345678901/17', rational=True) == \
        Rational(12345678901, 17)
    assert sympify('1/.3 + x', rational=True) == Rational(10, 3) + x
    # make sure longs in fractions work
    assert sympify('222222222222/11111111111') == \
        Rational(222222222222, 11111111111)
    # ... even if they come from repetend notation
    assert sympify('1/.2[123456789012]') == Rational(333333333333, 70781892967)
    # ... or from high precision reals
    assert sympify('.1234567890123456', rational=True) == \
        Rational(19290123283179, 156250000000000)


def test_sympify_Fraction():
    try:
        import fractions
    except ImportError:
        pass
    else:
        value = sympify(fractions.Fraction(101, 127))
        assert value == Rational(101, 127) and type(value) is Rational


def test_sympify_gmpy():
    if _gmpy is not None:
        import gmpy2

        value = sympify(gmpy2.mpz(1000001))
        assert value == Integer(1000001) and type(value) is Integer

        value = sympify(gmpy2.mpq(101, 127))
        assert value == Rational(101, 127) and type(value) is Rational


def test_sympify_flint():
    if _flint is not None:
        import flint

        value = sympify(flint.fmpz(1000001))
        assert value == Integer(1000001) and type(value) is Integer

        value = sympify(flint.fmpq(101, 127))
        assert value == Rational(101, 127) and type(value) is Rational


@conserve_mpmath_dps
def test_sympify_mpmath():
    value = sympify(mpmath.mpf(1.0))
    assert value == Float(1.0) and type(value) is Float

    mpmath.mp.dps = 12
    assert sympify(
        mpmath.pi).epsilon_eq(Float("3.14159265359"), Float("1e-12")) == True
    assert sympify(
        mpmath.pi).epsilon_eq(Float("3.14159265359"), Float("1e-13")) == False

    mpmath.mp.dps = 6
    assert sympify(
        mpmath.pi).epsilon_eq(Float("3.14159"), Float("1e-5")) == True
    assert sympify(
        mpmath.pi).epsilon_eq(Float("3.14159"), Float("1e-6")) == False

    mpmath.mp.dps = 15
    assert sympify(mpmath.mpc(1.0 + 2.0j)) == Float(1.0) + Float(2.0)*I


def test_sympify2():
    class A:
        def _sympy_(self):
            return Symbol("x")**3

    a = A()

    assert _sympify(a) == x**3
    assert sympify(a) == x**3
    assert a == x**3


def test_sympify3():
    assert sympify("x**3") == x**3
    assert sympify("x^3") == x**3
    assert sympify("1/2") == Integer(1)/2

    raises(SympifyError, lambda: _sympify('x**3'))
    raises(SympifyError, lambda: _sympify('1/2'))


def test_sympify_keywords():
    raises(SympifyError, lambda: sympify('if'))
    raises(SympifyError, lambda: sympify('for'))
    raises(SympifyError, lambda: sympify('while'))
    raises(SympifyError, lambda: sympify('lambda'))


def test_sympify_float():
    assert sympify("1e-64") != 0
    assert sympify("1e-20000") != 0


def test_sympify_bool():
    assert sympify(True) is true
    assert sympify(False) is false


def test_sympify_iterables():
    ans = [Rational(3, 10), Rational(1, 5)]
    assert sympify(['.3', '.2'], rational=True) == ans
    assert sympify({"x": 0, "y": 1}) == {x: 0, y: 1}
    assert sympify(['1', '2', ['3', '4']]) == [S(1), S(2), [S(3), S(4)]]


@XFAIL
def test_issue_16772():
    # because there is a converter for tuple, the
    # args are only sympified without the flags being passed
    # along; list, on the other hand, is not converted
    # with a converter so its args are traversed later
    ans = [Rational(3, 10), Rational(1, 5)]
    assert sympify(('.3', '.2'), rational=True) == Tuple(*ans)


def test_issue_16859():
    class no(float, CantSympify):
        pass
    raises(SympifyError, lambda: sympify(no(1.2)))


def test_sympify4():
    class A:
        def _sympy_(self):
            return Symbol("x")

    a = A()

    assert _sympify(a)**3 == x**3
    assert sympify(a)**3 == x**3
    assert a == x


def test_sympify_text():
    assert sympify('some') == Symbol('some')
    assert sympify('core') == Symbol('core')

    assert sympify('True') is True
    assert sympify('False') is False

    assert sympify('Poly') == Poly
    assert sympify('sin') == sin


def test_sympify_function():
    assert sympify('factor(x**2-1, x)') == -(1 - x)*(x + 1)
    assert sympify('sin(pi/2)*cos(pi)') == -Integer(1)


def test_sympify_poly():
    p = Poly(x**2 + x + 1, x)

    assert _sympify(p) is p
    assert sympify(p) is p


def test_sympify_factorial():
    assert sympify('x!') == factorial(x)
    assert sympify('(x+1)!') == factorial(x + 1)
    assert sympify('(1 + y*(x + 1))!') == factorial(1 + y*(x + 1))
    assert sympify('(1 + y*(x + 1)!)^2') == (1 + y*factorial(x + 1))**2
    assert sympify('y*x!') == y*factorial(x)
    assert sympify('x!!') == factorial2(x)
    assert sympify('(x+1)!!') == factorial2(x + 1)
    assert sympify('(1 + y*(x + 1))!!') == factorial2(1 + y*(x + 1))
    assert sympify('(1 + y*(x + 1)!!)^2') == (1 + y*factorial2(x + 1))**2
    assert sympify('y*x!!') == y*factorial2(x)
    assert sympify('factorial2(x)!') == factorial(factorial2(x))

    raises(SympifyError, lambda: sympify("+!!"))
    raises(SympifyError, lambda: sympify(")!!"))
    raises(SympifyError, lambda: sympify("!"))
    raises(SympifyError, lambda: sympify("(!)"))
    raises(SympifyError, lambda: sympify("x!!!"))


def test_issue_3595():
    assert sympify("a_") == Symbol("a_")
    assert sympify("_a") == Symbol("_a")


def test_lambda():
    x = Symbol('x')
    assert sympify('lambda: 1') == Lambda((), 1)
    assert sympify('lambda x: x') == Lambda(x, x)
    assert sympify('lambda x: 2*x') == Lambda(x, 2*x)
    assert sympify('lambda x, y: 2*x+y') == Lambda((x, y), 2*x + y)


def test_lambda_raises():
    raises(SympifyError, lambda: sympify("lambda *args: args")) # args argument error
    raises(SympifyError, lambda: sympify("lambda **kwargs: kwargs[0]")) # kwargs argument error
    raises(SympifyError, lambda: sympify("lambda x = 1: x"))    # Keyword argument error
    with raises(SympifyError):
        _sympify('lambda: 1')


def test_sympify_raises():
    raises(SympifyError, lambda: sympify("fx)"))

    class A:
        def __str__(self):
            return 'x'

    raises(SympifyError, lambda: sympify(A()))


def test__sympify():
    x = Symbol('x')
    f = Function('f')

    # positive _sympify
    assert _sympify(x) is x
    assert _sympify(1) == Integer(1)
    assert _sympify(0.5) == Float("0.5")
    assert _sympify(1 + 1j) == 1.0 + I*1.0

    # Function f is not Basic and can't sympify to Basic. We allow it to pass
    # with sympify but not with _sympify.
    # https://github.com/sympy/sympy/issues/20124
    assert sympify(f) is f
    raises(SympifyError, lambda: _sympify(f))

    class A:
        def _sympy_(self):
            return Integer(5)

    a = A()
    assert _sympify(a) == Integer(5)

    # negative _sympify
    raises(SympifyError, lambda: _sympify('1'))
    raises(SympifyError, lambda: _sympify([1, 2, 3]))


def test_sympifyit():
    x = Symbol('x')
    y = Symbol('y')

    @_sympifyit('b', NotImplemented)
    def add(a, b):
        return a + b

    assert add(x, 1) == x + 1
    assert add(x, 0.5) == x + Float('0.5')
    assert add(x, y) == x + y

    assert add(x, '1') == NotImplemented

    @_sympifyit('b')
    def add_raises(a, b):
        return a + b

    assert add_raises(x, 1) == x + 1
    assert add_raises(x, 0.5) == x + Float('0.5')
    assert add_raises(x, y) == x + y

    raises(SympifyError, lambda: add_raises(x, '1'))


def test_int_float():
    class F1_1:
        def __float__(self):
            return 1.1

    class F1_1b:
        """
        This class is still a float, even though it also implements __int__().
        """
        def __float__(self):
            return 1.1

        def __int__(self):
            return 1

    class F1_1c:
        """
        This class is still a float, because it implements _sympy_()
        """
        def __float__(self):
            return 1.1

        def __int__(self):
            return 1

        def _sympy_(self):
            return Float(1.1)

    class I5:
        def __int__(self):
            return 5

    class I5b:
        """
        This class implements both __int__() and __float__(), so it will be
        treated as Float in SymPy. One could change this behavior, by using
        float(a) == int(a), but deciding that integer-valued floats represent
        exact numbers is arbitrary and often not correct, so we do not do it.
        If, in the future, we decide to do it anyway, the tests for I5b need to
        be changed.
        """
        def __float__(self):
            return 5.0

        def __int__(self):
            return 5

    class I5c:
        """
        This class implements both __int__() and __float__(), but also
        a _sympy_() method, so it will be Integer.
        """
        def __float__(self):
            return 5.0

        def __int__(self):
            return 5

        def _sympy_(self):
            return Integer(5)

    i5 = I5()
    i5b = I5b()
    i5c = I5c()
    f1_1 = F1_1()
    f1_1b = F1_1b()
    f1_1c = F1_1c()
    assert sympify(i5) == 5
    assert isinstance(sympify(i5), Integer)
    assert sympify(i5b) == 5.0
    assert isinstance(sympify(i5b), Float)
    assert sympify(i5c) == 5
    assert isinstance(sympify(i5c), Integer)
    assert abs(sympify(f1_1) - 1.1) < 1e-5
    assert abs(sympify(f1_1b) - 1.1) < 1e-5
    assert abs(sympify(f1_1c) - 1.1) < 1e-5

    assert _sympify(i5) == 5
    assert isinstance(_sympify(i5), Integer)
    assert _sympify(i5b) == 5.0
    assert isinstance(_sympify(i5b), Float)
    assert _sympify(i5c) == 5
    assert isinstance(_sympify(i5c), Integer)
    assert abs(_sympify(f1_1) - 1.1) < 1e-5
    assert abs(_sympify(f1_1b) - 1.1) < 1e-5
    assert abs(_sympify(f1_1c) - 1.1) < 1e-5


def test_evaluate_false():
    cases = {
        '2 + 3': Add(2, 3, evaluate=False),
        '2**2 / 3': Mul(Pow(2, 2, evaluate=False), Pow(3, -1, evaluate=False), evaluate=False),
        '2 + 3 * 5': Add(2, Mul(3, 5, evaluate=False), evaluate=False),
        '2 - 3 * 5': Add(2, Mul(-1, Mul(3, 5,evaluate=False), evaluate=False), evaluate=False),
        '1 / 3': Mul(1, Pow(3, -1, evaluate=False), evaluate=False),
        'True | False': Or(True, False, evaluate=False),
        '1 + 2 + 3 + 5*3 + integrate(x)': Add(1, 2, 3, Mul(5, 3, evaluate=False), x**2/2, evaluate=False),
        '2 * 4 * 6 + 8': Add(Mul(2, 4, 6, evaluate=False), 8, evaluate=False),
        '2 - 8 / 4': Add(2, Mul(-1, Mul(8, Pow(4, -1, evaluate=False), evaluate=False), evaluate=False), evaluate=False),
        '2 - 2**2': Add(2, Mul(-1, Pow(2, 2, evaluate=False), evaluate=False), evaluate=False),
    }
    for case, result in cases.items():
        assert sympify(case, evaluate=False) == result


def test_issue_4133():
    a = sympify('Integer(4)')

    assert a == Integer(4)
    assert a.is_Integer


def test_issue_3982():
    a = [3, 2.0]
    assert sympify(a) == [Integer(3), Float(2.0)]
    assert sympify(tuple(a)) == Tuple(Integer(3), Float(2.0))
    assert sympify(set(a)) == FiniteSet(Integer(3), Float(2.0))


def test_S_sympify():
    assert S(1)/2 == sympify(1)/2 == S.Half
    assert (-2)**(S(1)/2) == sqrt(2)*I


def test_issue_4788():
    assert srepr(S(1.0 + 0J)) == srepr(S(1.0)) == srepr(Float(1.0))


def test_issue_4798_None():
    assert S(None) is None


def test_issue_3218():
    assert sympify("x+\ny") == x + y

def test_issue_19399():
    if not numpy:
        skip("numpy not installed.")

    a = numpy.array(Rational(1, 2))
    b = Rational(1, 3)
    assert (a * b, type(a * b)) == (b * a, type(b * a))


def test_issue_4988_builtins():
    C = Symbol('C')
    vars = {'C': C}
    exp1 = sympify('C')
    assert exp1 == C  # Make sure it did not get mixed up with sympy.C

    exp2 = sympify('C', vars)
    assert exp2 == C  # Make sure it did not get mixed up with sympy.C


def test_geometry():
    p = sympify(Point(0, 1))
    assert p == Point(0, 1) and isinstance(p, Point)
    L = sympify(Line(p, (1, 0)))
    assert L == Line((0, 1), (1, 0)) and isinstance(L, Line)


def test_kernS():
    s =   '-1 - 2*(-(-x + 1/x)/(x*(x - 1/x)**2) - 1/(x*(x - 1/x)))'
    # when 1497 is fixed, this no longer should pass: the expression
    # should be unchanged
    assert -1 - 2*(-(-x + 1/x)/(x*(x - 1/x)**2) - 1/(x*(x - 1/x))) == -1
    # sympification should not allow the constant to enter a Mul
    # or else the structure can change dramatically
    ss = kernS(s)
    assert ss != -1 and ss.simplify() == -1
    s = '-1 - 2*(-(-x + 1/x)/(x*(x - 1/x)**2) - 1/(x*(x - 1/x)))'.replace(
        'x', '_kern')
    ss = kernS(s)
    assert ss != -1 and ss.simplify() == -1
    # issue 6687
    assert (kernS('Interval(-1,-2 - 4*(-3))')
        == Interval(-1, Add(-2, Mul(12, 1, evaluate=False), evaluate=False)))
    assert kernS('_kern') == Symbol('_kern')
    assert kernS('E**-(x)') == exp(-x)
    e = 2*(x + y)*y
    assert kernS(['2*(x + y)*y', ('2*(x + y)*y',)]) == [e, (e,)]
    assert kernS('-(2*sin(x)**2 + 2*sin(x)*cos(x))*y/2') == \
        -y*(2*sin(x)**2 + 2*sin(x)*cos(x))/2
    # issue 15132
    assert kernS('(1 - x)/(1 - x*(1-y))') == kernS('(1-x)/(1-(1-y)*x)')
    assert kernS('(1-2**-(4+1)*(1-y)*x)') == (1 - x*(1 - y)/32)
    assert kernS('(1-2**(4+1)*(1-y)*x)') == (1 - 32*x*(1 - y))
    assert kernS('(1-2.*(1-y)*x)') == 1 - 2.*x*(1 - y)
    one = kernS('x - (x - 1)')
    assert one != 1 and one.expand() == 1
    assert kernS("(2*x)/(x-1)") == 2*x/(x-1)


def test_issue_6540_6552():
    assert S('[[1/3,2], (2/5,)]') == [[Rational(1, 3), 2], (Rational(2, 5),)]
    assert S('[[2/6,2], (2/4,)]') == [[Rational(1, 3), 2], (S.Half,)]
    assert S('[[[2*(1)]]]') == [[[2]]]
    assert S('Matrix([2*(1)])') == Matrix([2])


def test_issue_6046():
    assert str(S("Q & C", locals=_clash1)) == 'C & Q'
    assert str(S('pi(x)', locals=_clash2)) == 'pi(x)'
    locals = {}
    exec("from sympy.abc import Q, C", locals)
    assert str(S('C&Q', locals)) == 'C & Q'
    # clash can act as Symbol or Function
    assert str(S('pi(C, Q)', locals=_clash)) == 'pi(C, Q)'
    assert len(S('pi + x', locals=_clash2).free_symbols) == 2
    # but not both
    raises(TypeError, lambda: S('pi + pi(x)', locals=_clash2))
    assert all(set(i.values()) == {null} for i in (
        _clash, _clash1, _clash2))


def test_issue_8821_highprec_from_str():
    s = str(pi.evalf(128))
    p = sympify(s)
    assert Abs(sin(p)) < 1e-127


def test_issue_10295():
    if not numpy:
        skip("numpy not installed.")

    A = numpy.array([[1, 3, -1],
                     [0, 1, 7]])
    sA = S(A)
    assert sA.shape == (2, 3)
    for (ri, ci), val in numpy.ndenumerate(A):
        assert sA[ri, ci] == val

    B = numpy.array([-7, x, 3*y**2])
    sB = S(B)
    assert sB.shape == (3,)
    assert B[0] == sB[0] == -7
    assert B[1] == sB[1] == x
    assert B[2] == sB[2] == 3*y**2

    C = numpy.arange(0, 24)
    C.resize(2,3,4)
    sC = S(C)
    assert sC[0, 0, 0].is_integer
    assert sC[0, 0, 0] == 0

    a1 = numpy.array([1, 2, 3])
    a2 = numpy.array(list(range(24)))
    a2.resize(2, 4, 3)
    assert sympify(a1) == ImmutableDenseNDimArray([1, 2, 3])
    assert sympify(a2) == ImmutableDenseNDimArray(list(range(24)), (2, 4, 3))


def test_Range():
    # Only works in Python 3 where range returns a range type
    assert sympify(range(10)) == Range(10)
    assert _sympify(range(10)) == Range(10)


def test_sympify_set():
    n = Symbol('n')
    assert sympify({n}) == FiniteSet(n)
    assert sympify(set()) == EmptySet


def test_sympify_numpy():
    if not numpy:
        skip('numpy not installed. Abort numpy tests.')
    np = numpy

    def equal(x, y):
        return x == y and type(x) == type(y)

    assert sympify(np.bool_(1)) is S(True)
    try:
        assert equal(
            sympify(np.int_(1234567891234567891)), S(1234567891234567891))
        assert equal(
            sympify(np.intp(1234567891234567891)), S(1234567891234567891))
    except OverflowError:
        # May fail on 32-bit systems: Python int too large to convert to C long
        pass
    assert equal(sympify(np.intc(1234567891)), S(1234567891))
    assert equal(sympify(np.int8(-123)), S(-123))
    assert equal(sympify(np.int16(-12345)), S(-12345))
    assert equal(sympify(np.int32(-1234567891)), S(-1234567891))
    assert equal(
        sympify(np.int64(-1234567891234567891)), S(-1234567891234567891))
    assert equal(sympify(np.uint8(123)), S(123))
    assert equal(sympify(np.uint16(12345)), S(12345))
    assert equal(sympify(np.uint32(1234567891)), S(1234567891))
    assert equal(
        sympify(np.uint64(1234567891234567891)), S(1234567891234567891))
    assert equal(sympify(np.float32(1.123456)), Float(1.123456, precision=24))
    assert equal(sympify(np.float64(1.1234567891234)),
                Float(1.1234567891234, precision=53))

    # The exact precision of np.longdouble, npfloat128 and other extended
    # precision dtypes is platform dependent.
    ldprec = np.finfo(np.longdouble(1)).nmant + 1
    assert equal(sympify(np.longdouble(1.123456789)),
                 Float(1.123456789, precision=ldprec))

    assert equal(sympify(np.complex64(1 + 2j)), S(1.0 + 2.0*I))
    assert equal(sympify(np.complex128(1 + 2j)), S(1.0 + 2.0*I))

    lcprec = np.finfo(np.clongdouble(1)).nmant + 1
    assert equal(sympify(np.clongdouble(1 + 2j)),
                Float(1.0, precision=lcprec) + Float(2.0, precision=lcprec)*I)

    #float96 does not exist on all platforms
    if hasattr(np, 'float96'):
        f96prec = np.finfo(np.float96(1)).nmant + 1
        assert equal(sympify(np.float96(1.123456789)),
                    Float(1.123456789, precision=f96prec))

    #float128 does not exist on all platforms
    if hasattr(np, 'float128'):
        f128prec = np.finfo(np.float128(1)).nmant + 1
        assert equal(sympify(np.float128(1.123456789123)),
                    Float(1.123456789123, precision=f128prec))


@XFAIL
def test_sympify_rational_numbers_set():
    ans = [Rational(3, 10), Rational(1, 5)]
    assert sympify({'.3', '.2'}, rational=True) == FiniteSet(*ans)


def test_sympify_mro():
    """Tests the resolution order for classes that implement _sympy_"""
    class a:
        def _sympy_(self):
            return Integer(1)
    class b(a):
        def _sympy_(self):
            return Integer(2)
    class c(a):
        pass

    assert sympify(a()) == Integer(1)
    assert sympify(b()) == Integer(2)
    assert sympify(c()) == Integer(1)


def test_sympify_converter():
    """Tests the resolution order for classes in converter"""
    class a:
        pass
    class b(a):
        pass
    class c(a):
        pass

    converter[a] = lambda x: Integer(1)
    converter[b] = lambda x: Integer(2)

    assert sympify(a()) == Integer(1)
    assert sympify(b()) == Integer(2)
    assert sympify(c()) == Integer(1)

    class MyInteger(Integer):
        pass

    if int in converter:
        int_converter = converter[int]
    else:
        int_converter = None

    try:
        converter[int] = MyInteger
        assert sympify(1) == MyInteger(1)
    finally:
        if int_converter is None:
            del converter[int]
        else:
            converter[int] = int_converter


def test_issue_13924():
    if not numpy:
        skip("numpy not installed.")

    a = sympify(numpy.array([1]))
    assert isinstance(a, ImmutableDenseNDimArray)
    assert a[0] == 1


def test_numpy_sympify_args():
    # Issue 15098. Make sure sympify args work with numpy types (like numpy.str_)
    if not numpy:
        skip("numpy not installed.")

    a = sympify(numpy.str_('a'))
    assert type(a) is Symbol
    assert a == Symbol('a')

    class CustomSymbol(Symbol):
        pass

    a = sympify(numpy.str_('a'), {"Symbol": CustomSymbol})
    assert isinstance(a, CustomSymbol)

    a = sympify(numpy.str_('x^y'))
    assert a == x**y
    a = sympify(numpy.str_('x^y'), convert_xor=False)
    assert a == Xor(x, y)

    raises(SympifyError, lambda: sympify(numpy.str_('x'), strict=True))

    a = sympify(numpy.str_('1.1'))
    assert isinstance(a, Float)
    assert a == 1.1

    a = sympify(numpy.str_('1.1'), rational=True)
    assert isinstance(a, Rational)
    assert a == Rational(11, 10)

    a = sympify(numpy.str_('x + x'))
    assert isinstance(a, Mul)
    assert a == 2*x

    a = sympify(numpy.str_('x + x'), evaluate=False)
    assert isinstance(a, Add)
    assert a == Add(x, x, evaluate=False)


def test_issue_5939():
    a = Symbol('a')
    b = Symbol('b')
    assert sympify('''a+\nb''') == a + b


def test_issue_16759():
    d = sympify({.5: 1})
    assert S.Half not in d
    assert Float(.5) in d
    assert d[.5] is S.One
    d = sympify(OrderedDict({.5: 1}))
    assert S.Half not in d
    assert Float(.5) in d
    assert d[.5] is S.One
    d = sympify(defaultdict(int, {.5: 1}))
    assert S.Half not in d
    assert Float(.5) in d
    assert d[.5] is S.One


def test_issue_17811():
    a = Function('a')
    assert sympify('a(x)*5', evaluate=False) == Mul(a(x), 5, evaluate=False)


def test_issue_8439():
    assert sympify(float('inf')) == oo
    assert x + float('inf') == x + oo
    assert S(float('inf')) == oo


def test_issue_14706():
    if not numpy:
        skip("numpy not installed.")

    z1 = numpy.zeros((1, 1), dtype=numpy.float64)
    z2 = numpy.zeros((2, 2), dtype=numpy.float64)
    z3 = numpy.zeros((), dtype=numpy.float64)

    y1 = numpy.ones((1, 1), dtype=numpy.float64)
    y2 = numpy.ones((2, 2), dtype=numpy.float64)
    y3 = numpy.ones((), dtype=numpy.float64)

    assert numpy.all(x + z1 == numpy.full((1, 1), x))
    assert numpy.all(x + z2 == numpy.full((2, 2), x))
    assert numpy.all(z1 + x == numpy.full((1, 1), x))
    assert numpy.all(z2 + x == numpy.full((2, 2), x))
    for z in [z3,
              numpy.int64(0),
              numpy.float64(0),
              numpy.complex64(0)]:
        assert x + z == x
        assert z + x == x
        assert isinstance(x + z, Symbol)
        assert isinstance(z + x, Symbol)

    # If these tests fail, then it means that numpy has finally
    # fixed the issue of scalar conversion for rank>0 arrays
    # which is mentioned in numpy/numpy#10404. In that case,
    # some changes have to be made in sympify.py.
    # Note: For future reference, for anyone who takes up this
    # issue when numpy has finally fixed their side of the problem,
    # the changes for this temporary fix were introduced in PR 18651
    assert numpy.all(x + y1 == numpy.full((1, 1), x + 1.0))
    assert numpy.all(x + y2 == numpy.full((2, 2), x + 1.0))
    assert numpy.all(y1 + x == numpy.full((1, 1), x + 1.0))
    assert numpy.all(y2 + x == numpy.full((2, 2), x + 1.0))
    for y_ in [y3,
              numpy.int64(1),
              numpy.float64(1),
              numpy.complex64(1)]:
        assert x + y_ == y_ + x
        assert isinstance(x + y_, Add)
        assert isinstance(y_ + x, Add)

    assert x + numpy.array(x) == 2 * x
    assert x + numpy.array([x]) == numpy.array([2*x], dtype=object)

    assert sympify(numpy.array([1])) == ImmutableDenseNDimArray([1], 1)
    assert sympify(numpy.array([[[1]]])) == ImmutableDenseNDimArray([1], (1, 1, 1))
    assert sympify(z1) == ImmutableDenseNDimArray([0.0], (1, 1))
    assert sympify(z2) == ImmutableDenseNDimArray([0.0, 0.0, 0.0, 0.0], (2, 2))
    assert sympify(z3) == ImmutableDenseNDimArray([0.0], ())
    assert sympify(z3, strict=True) == 0.0

    raises(SympifyError, lambda: sympify(numpy.array([1]), strict=True))
    raises(SympifyError, lambda: sympify(z1, strict=True))
    raises(SympifyError, lambda: sympify(z2, strict=True))


def test_issue_21536():
    #test to check evaluate=False in case of iterable input
    u = sympify("x+3*x+2", evaluate=False)
    v = sympify("2*x+4*x+2+4", evaluate=False)

    assert u.is_Add and set(u.args) == {x, 3*x, 2}
    assert v.is_Add and set(v.args) == {2*x, 4*x, 2, 4}
    assert sympify(["x+3*x+2", "2*x+4*x+2+4"], evaluate=False) == [u, v]

    #test to check evaluate=True in case of iterable input
    u = sympify("x+3*x+2", evaluate=True)
    v = sympify("2*x+4*x+2+4", evaluate=True)

    assert u.is_Add and set(u.args) == {4*x, 2}
    assert v.is_Add and set(v.args) == {6*x, 6}
    assert sympify(["x+3*x+2", "2*x+4*x+2+4"], evaluate=True) == [u, v]

    #test to check evaluate with no input in case of iterable input
    u = sympify("x+3*x+2")
    v = sympify("2*x+4*x+2+4")

    assert u.is_Add and set(u.args) == {4*x, 2}
    assert v.is_Add and set(v.args) == {6*x, 6}
    assert sympify(["x+3*x+2", "2*x+4*x+2+4"]) == [u, v]

def test_issue_27284():
    if not numpy:
        skip("numpy not installed.")

    assert Float(numpy.float32(float('inf'))) == S.Infinity
    assert Float(numpy.float32(float('-inf'))) == S.NegativeInfinity

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_c.py ===
from sympy.core import (
    S, pi, oo, Symbol, symbols, Rational, Integer, Float, Function, Mod, GoldenRatio, EulerGamma, Catalan,
    Lambda, Dummy, nan, Mul, Pow, UnevaluatedExpr
)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.functions import (
    Abs, acos, acosh, asin, asinh, atan, atanh, atan2, ceiling, cos, cosh, erf,
    erfc, exp, floor, gamma, log, loggamma, Max, Min, Piecewise, sign, sin, sinh,
    sqrt, tan, tanh, fibonacci, lucas
)
from sympy.sets import Range
from sympy.logic import ITE, Implies, Equivalent
from sympy.codegen import For, aug_assign, Assignment
from sympy.testing.pytest import raises, XFAIL
from sympy.printing.codeprinter import PrintMethodNotImplementedError
from sympy.printing.c import C89CodePrinter, C99CodePrinter, get_math_macros
from sympy.codegen.ast import (
    AddAugmentedAssignment, Element, Type, FloatType, Declaration, Pointer, Variable, value_const, pointer_const,
    While, Scope, Print, FunctionPrototype, FunctionDefinition, FunctionCall, Return,
    real, float32, float64, float80, float128, intc, Comment, CodeBlock, stderr, QuotedString
)
from sympy.codegen.cfunctions import expm1, log1p, exp2, log2, fma, log10, Cbrt, hypot, Sqrt, isnan, isinf
from sympy.codegen.cnodes import restrict
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol, SparseMatrix

from sympy.printing.codeprinter import ccode

x, y, z = symbols('x,y,z')


def test_printmethod():
    class fabs(Abs):
        def _ccode(self, printer):
            return "fabs(%s)" % printer._print(self.args[0])

    assert ccode(fabs(x)) == "fabs(x)"


def test_ccode_sqrt():
    assert ccode(sqrt(x)) == "sqrt(x)"
    assert ccode(x**0.5) == "sqrt(x)"
    assert ccode(sqrt(x)) == "sqrt(x)"


def test_ccode_Pow():
    assert ccode(x**3) == "pow(x, 3)"
    assert ccode(x**(y**3)) == "pow(x, pow(y, 3))"
    g = implemented_function('g', Lambda(x, 2*x))
    assert ccode(1/(g(x)*3.5)**(x - y**x)/(x**2 + y)) == \
        "pow(3.5*2*x, -x + pow(y, x))/(pow(x, 2) + y)"
    assert ccode(x**-1.0) == '1.0/x'
    assert ccode(x**Rational(2, 3)) == 'pow(x, 2.0/3.0)'
    assert ccode(x**Rational(2, 3), type_aliases={real: float80}) == 'powl(x, 2.0L/3.0L)'
    _cond_cfunc = [(lambda base, exp: exp.is_integer, "dpowi"),
                   (lambda base, exp: not exp.is_integer, "pow")]
    assert ccode(x**3, user_functions={'Pow': _cond_cfunc}) == 'dpowi(x, 3)'
    assert ccode(x**0.5, user_functions={'Pow': _cond_cfunc}) == 'pow(x, 0.5)'
    assert ccode(x**Rational(16, 5), user_functions={'Pow': _cond_cfunc}) == 'pow(x, 16.0/5.0)'
    _cond_cfunc2 = [(lambda base, exp: base == 2, lambda base, exp: 'exp2(%s)' % exp),
                    (lambda base, exp: base != 2, 'pow')]
    # Related to gh-11353
    assert ccode(2**x, user_functions={'Pow': _cond_cfunc2}) == 'exp2(x)'
    assert ccode(x**2, user_functions={'Pow': _cond_cfunc2}) == 'pow(x, 2)'
    # For issue 14160
    assert ccode(Mul(-2, x, Pow(Mul(y,y,evaluate=False), -1, evaluate=False),
                                                evaluate=False)) == '-2*x/(y*y)'


def test_ccode_Max():
    # Test for gh-11926
    assert ccode(Max(x,x*x),user_functions={"Max":"my_max", "Pow":"my_pow"}) == 'my_max(x, my_pow(x, 2))'


def test_ccode_Min_performance():
    #Shouldn't take more than a few seconds
    big_min = Min(*symbols('a[0:50]'))
    for curr_standard in ('c89', 'c99', 'c11'):
        output = ccode(big_min, standard=curr_standard)
        assert output.count('(') == output.count(')')


def test_ccode_constants_mathh():
    assert ccode(exp(1)) == "M_E"
    assert ccode(pi) == "M_PI"
    assert ccode(oo, standard='c89') == "HUGE_VAL"
    assert ccode(-oo, standard='c89') == "-HUGE_VAL"
    assert ccode(oo) == "INFINITY"
    assert ccode(-oo, standard='c99') == "-INFINITY"
    assert ccode(pi, type_aliases={real: float80}) == "M_PIl"


def test_ccode_constants_other():
    assert ccode(2*GoldenRatio) == "const double GoldenRatio = %s;\n2*GoldenRatio" % GoldenRatio.evalf(17)
    assert ccode(
        2*Catalan) == "const double Catalan = %s;\n2*Catalan" % Catalan.evalf(17)
    assert ccode(2*EulerGamma) == "const double EulerGamma = %s;\n2*EulerGamma" % EulerGamma.evalf(17)


def test_ccode_Rational():
    assert ccode(Rational(3, 7)) == "3.0/7.0"
    assert ccode(Rational(3, 7), type_aliases={real: float80}) == "3.0L/7.0L"
    assert ccode(Rational(18, 9)) == "2"
    assert ccode(Rational(3, -7)) == "-3.0/7.0"
    assert ccode(Rational(3, -7), type_aliases={real: float80}) == "-3.0L/7.0L"
    assert ccode(Rational(-3, -7)) == "3.0/7.0"
    assert ccode(Rational(-3, -7), type_aliases={real: float80}) == "3.0L/7.0L"
    assert ccode(x + Rational(3, 7)) == "x + 3.0/7.0"
    assert ccode(x + Rational(3, 7), type_aliases={real: float80}) == "x + 3.0L/7.0L"
    assert ccode(Rational(3, 7)*x) == "(3.0/7.0)*x"
    assert ccode(Rational(3, 7)*x, type_aliases={real: float80}) == "(3.0L/7.0L)*x"


def test_ccode_Integer():
    assert ccode(Integer(67)) == "67"
    assert ccode(Integer(-1)) == "-1"


def test_ccode_functions():
    assert ccode(sin(x) ** cos(x)) == "pow(sin(x), cos(x))"


def test_ccode_inline_function():
    x = symbols('x')
    g = implemented_function('g', Lambda(x, 2*x))
    assert ccode(g(x)) == "2*x"
    g = implemented_function('g', Lambda(x, 2*x/Catalan))
    assert ccode(
        g(x)) == "const double Catalan = %s;\n2*x/Catalan" % Catalan.evalf(17)
    A = IndexedBase('A')
    i = Idx('i', symbols('n', integer=True))
    g = implemented_function('g', Lambda(x, x*(1 + x)*(2 + x)))
    assert ccode(g(A[i]), assign_to=A[i]) == (
        "for (int i=0; i<n; i++){\n"
        "   A[i] = (A[i] + 1)*(A[i] + 2)*A[i];\n"
        "}"
    )


def test_ccode_exceptions():
    assert ccode(gamma(x), standard='C99') == "tgamma(x)"
    with raises(PrintMethodNotImplementedError):
        ccode(gamma(x), standard='C89')
    with raises(PrintMethodNotImplementedError):
        ccode(gamma(x), standard='C89', allow_unknown_functions=False)

    ccode(gamma(x), standard='C89', allow_unknown_functions=True)



def test_ccode_functions2():
    assert ccode(ceiling(x)) == "ceil(x)"
    assert ccode(Abs(x)) == "fabs(x)"
    assert ccode(gamma(x)) == "tgamma(x)"
    r, s = symbols('r,s', real=True)
    assert ccode(Mod(ceiling(r), ceiling(s))) == '((ceil(r) % ceil(s)) + '\
                                                 'ceil(s)) % ceil(s)'
    assert ccode(Mod(r, s)) == "fmod(r, s)"
    p1, p2 = symbols('p1 p2', integer=True, positive=True)
    assert ccode(Mod(p1, p2)) == 'p1 % p2'
    assert ccode(Mod(p1, p2 + 3)) == 'p1 % (p2 + 3)'
    assert ccode(Mod(-3, -7, evaluate=False)) == '(-3) % (-7)'
    assert ccode(-Mod(3, 7, evaluate=False)) == '-(3 % 7)'
    assert ccode(r*Mod(p1, p2)) == 'r*(p1 % p2)'
    assert ccode(Mod(p1, p2)**s) == 'pow(p1 % p2, s)'
    n = symbols('n', integer=True, negative=True)
    assert ccode(Mod(-n, p2)) == '(-n) % p2'
    assert ccode(fibonacci(n)) == '((1.0/5.0)*pow(2, -n)*sqrt(5)*(-pow(1 - sqrt(5), n) + pow(1 + sqrt(5), n)))'
    assert ccode(lucas(n)) == '(pow(2, -n)*(pow(1 - sqrt(5), n) + pow(1 + sqrt(5), n)))'


def test_ccode_user_functions():
    x = symbols('x', integer=False)
    n = symbols('n', integer=True)
    custom_functions = {
        "ceiling": "ceil",
        "Abs": [(lambda x: not x.is_integer, "fabs"), (lambda x: x.is_integer, "abs")],
    }
    assert ccode(ceiling(x), user_functions=custom_functions) == "ceil(x)"
    assert ccode(Abs(x), user_functions=custom_functions) == "fabs(x)"
    assert ccode(Abs(n), user_functions=custom_functions) == "abs(n)"

    expr = Symbol('a')
    muladd = Function('muladd')
    for i in range(0, 100):
        # the large number of terms acts as a regression test for gh-23839
        expr = muladd(Rational(1, 2), Symbol(f'a{i}'), expr)
    out = ccode(expr, user_functions={'muladd':'muladd'})
    assert 'a99' in out
    assert out.count('muladd') == 100


def test_ccode_boolean():
    assert ccode(True) == "true"
    assert ccode(S.true) == "true"
    assert ccode(False) == "false"
    assert ccode(S.false) == "false"
    assert ccode(x & y) == "x && y"
    assert ccode(x | y) == "x || y"
    assert ccode(~x) == "!x"
    assert ccode(x & y & z) == "x && y && z"
    assert ccode(x | y | z) == "x || y || z"
    assert ccode((x & y) | z) == "z || x && y"
    assert ccode((x | y) & z) == "z && (x || y)"
    # Automatic rewrites
    assert ccode(x ^ y) == '(x || y) && (!x || !y)'
    assert ccode((x ^ y) ^ z) == '(x || y || z) && (x || !y || !z) && (y || !x || !z) && (z || !x || !y)'
    assert ccode(Implies(x, y)) == 'y || !x'
    assert ccode(Equivalent(x, z ^ y, Implies(z, x))) == '(x || (y || !z) && (z || !y)) && (z && !x || (y || z) && (!y || !z))'


def test_ccode_Relational():
    assert ccode(Eq(x, y)) == "x == y"
    assert ccode(Ne(x, y)) == "x != y"
    assert ccode(Le(x, y)) == "x <= y"
    assert ccode(Lt(x, y)) == "x < y"
    assert ccode(Gt(x, y)) == "x > y"
    assert ccode(Ge(x, y)) == "x >= y"


def test_ccode_Piecewise():
    expr = Piecewise((x, x < 1), (x**2, True))
    assert ccode(expr) == (
            "((x < 1) ? (\n"
            "   x\n"
            ")\n"
            ": (\n"
            "   pow(x, 2)\n"
            "))")
    assert ccode(expr, assign_to="c") == (
            "if (x < 1) {\n"
            "   c = x;\n"
            "}\n"
            "else {\n"
            "   c = pow(x, 2);\n"
            "}")
    expr = Piecewise((x, x < 1), (x + 1, x < 2), (x**2, True))
    assert ccode(expr) == (
            "((x < 1) ? (\n"
            "   x\n"
            ")\n"
            ": ((x < 2) ? (\n"
            "   x + 1\n"
            ")\n"
            ": (\n"
            "   pow(x, 2)\n"
            ")))")
    assert ccode(expr, assign_to='c') == (
            "if (x < 1) {\n"
            "   c = x;\n"
            "}\n"
            "else if (x < 2) {\n"
            "   c = x + 1;\n"
            "}\n"
            "else {\n"
            "   c = pow(x, 2);\n"
            "}")
    # Check that Piecewise without a True (default) condition error
    expr = Piecewise((x, x < 1), (x**2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: ccode(expr))


def test_ccode_sinc():
    from sympy.functions.elementary.trigonometric import sinc
    expr = sinc(x)
    assert ccode(expr) == (
            "(((x != 0) ? (\n"
            "   sin(x)/x\n"
            ")\n"
            ": (\n"
            "   1\n"
            ")))")


def test_ccode_Piecewise_deep():
    p = ccode(2*Piecewise((x, x < 1), (x + 1, x < 2), (x**2, True)))
    assert p == (
            "2*((x < 1) ? (\n"
            "   x\n"
            ")\n"
            ": ((x < 2) ? (\n"
            "   x + 1\n"
            ")\n"
            ": (\n"
            "   pow(x, 2)\n"
            ")))")
    expr = x*y*z + x**2 + y**2 + Piecewise((0, x < 0.5), (1, True)) + cos(z) - 1
    assert ccode(expr) == (
            "pow(x, 2) + x*y*z + pow(y, 2) + ((x < 0.5) ? (\n"
            "   0\n"
            ")\n"
            ": (\n"
            "   1\n"
            ")) + cos(z) - 1")
    assert ccode(expr, assign_to='c') == (
            "c = pow(x, 2) + x*y*z + pow(y, 2) + ((x < 0.5) ? (\n"
            "   0\n"
            ")\n"
            ": (\n"
            "   1\n"
            ")) + cos(z) - 1;")


def test_ccode_ITE():
    expr = ITE(x < 1, y, z)
    assert ccode(expr) == (
            "((x < 1) ? (\n"
            "   y\n"
            ")\n"
            ": (\n"
            "   z\n"
            "))")


def test_ccode_settings():
    raises(TypeError, lambda: ccode(sin(x), method="garbage"))


def test_ccode_Indexed():
    s, n, m, o = symbols('s n m o', integer=True)
    i, j, k = Idx('i', n), Idx('j', m), Idx('k', o)

    x = IndexedBase('x')[j]
    A = IndexedBase('A')[i, j]
    B = IndexedBase('B')[i, j, k]

    p = C99CodePrinter()

    assert p._print_Indexed(x) == 'x[j]'
    assert p._print_Indexed(A) == 'A[%s]' % (m*i+j)
    assert p._print_Indexed(B) == 'B[%s]' % (i*o*m+j*o+k)

    A = IndexedBase('A', shape=(5,3))[i, j]
    assert p._print_Indexed(A) == 'A[%s]' % (3*i + j)

    A = IndexedBase('A', shape=(5,3), strides='F')[i, j]
    assert ccode(A) == 'A[%s]' % (i + 5*j)

    A = IndexedBase('A', shape=(29,29), strides=(1, s), offset=o)[i, j]
    assert ccode(A) == 'A[o + s*j + i]'

    Abase = IndexedBase('A', strides=(s, m, n), offset=o)
    assert ccode(Abase[i, j, k]) == 'A[m*j + n*k + o + s*i]'
    assert ccode(Abase[2, 3, k]) == 'A[3*m + n*k + o + 2*s]'


def test_Element():
    assert ccode(Element('x', 'ij')) == 'x[i][j]'
    assert ccode(Element('x', 'ij', strides='kl', offset='o')) == 'x[i*k + j*l + o]'
    assert ccode(Element('x', (3,))) == 'x[3]'
    assert ccode(Element('x', (3,4,5))) == 'x[3][4][5]'


def test_ccode_Indexed_without_looking_for_contraction():
    len_y = 5
    y = IndexedBase('y', shape=(len_y,))
    x = IndexedBase('x', shape=(len_y,))
    Dy = IndexedBase('Dy', shape=(len_y-1,))
    i = Idx('i', len_y-1)
    e = Eq(Dy[i], (y[i+1]-y[i])/(x[i+1]-x[i]))
    code0 = ccode(e.rhs, assign_to=e.lhs, contract=False)
    assert code0 == 'Dy[i] = (y[%s] - y[i])/(x[%s] - x[i]);' % (i + 1, i + 1)


def test_ccode_loops_matrix_vector():
    n, m = symbols('n m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)

    s = (
        'for (int i=0; i<m; i++){\n'
        '   y[i] = 0;\n'
        '}\n'
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      y[i] = A[%s]*x[j] + y[i];\n' % (i*n + j) +\
        '   }\n'
        '}'
    )
    assert ccode(A[i, j]*x[j], assign_to=y[i]) == s


def test_dummy_loops():
    i, m = symbols('i m', integer=True, cls=Dummy)
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx(i, m)

    expected = (
        'for (int i_%(icount)i=0; i_%(icount)i<m_%(mcount)i; i_%(icount)i++){\n'
        '   y[i_%(icount)i] = x[i_%(icount)i];\n'
        '}'
    ) % {'icount': i.label.dummy_index, 'mcount': m.dummy_index}

    assert ccode(x[i], assign_to=y[i]) == expected


def test_ccode_loops_add():
    n, m = symbols('n m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    z = IndexedBase('z')
    i = Idx('i', m)
    j = Idx('j', n)

    s = (
        'for (int i=0; i<m; i++){\n'
        '   y[i] = x[i] + z[i];\n'
        '}\n'
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      y[i] = A[%s]*x[j] + y[i];\n' % (i*n + j) +\
        '   }\n'
        '}'
    )
    assert ccode(A[i, j]*x[j] + x[i] + z[i], assign_to=y[i]) == s


def test_ccode_loops_multiple_contractions():
    n, m, o, p = symbols('n m o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)

    s = (
        'for (int i=0; i<m; i++){\n'
        '   y[i] = 0;\n'
        '}\n'
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      for (int k=0; k<o; k++){\n'
        '         for (int l=0; l<p; l++){\n'
        '            y[i] = a[%s]*b[%s] + y[i];\n' % (i*n*o*p + j*o*p + k*p + l, j*o*p + k*p + l) +\
        '         }\n'
        '      }\n'
        '   }\n'
        '}'
    )
    assert ccode(b[j, k, l]*a[i, j, k, l], assign_to=y[i]) == s


def test_ccode_loops_addfactor():
    n, m, o, p = symbols('n m o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)

    s = (
        'for (int i=0; i<m; i++){\n'
        '   y[i] = 0;\n'
        '}\n'
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      for (int k=0; k<o; k++){\n'
        '         for (int l=0; l<p; l++){\n'
        '            y[i] = (a[%s] + b[%s])*c[%s] + y[i];\n' % (i*n*o*p + j*o*p + k*p + l, i*n*o*p + j*o*p + k*p + l, j*o*p + k*p + l) +\
        '         }\n'
        '      }\n'
        '   }\n'
        '}'
    )
    assert ccode((a[i, j, k, l] + b[i, j, k, l])*c[j, k, l], assign_to=y[i]) == s


def test_ccode_loops_multiple_terms():
    n, m, o, p = symbols('n m o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)

    s0 = (
        'for (int i=0; i<m; i++){\n'
        '   y[i] = 0;\n'
        '}\n'
    )
    s1 = (
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      for (int k=0; k<o; k++){\n'
        '         y[i] = b[j]*b[k]*c[%s] + y[i];\n' % (i*n*o + j*o + k) +\
        '      }\n'
        '   }\n'
        '}\n'
    )
    s2 = (
        'for (int i=0; i<m; i++){\n'
        '   for (int k=0; k<o; k++){\n'
        '      y[i] = a[%s]*b[k] + y[i];\n' % (i*o + k) +\
        '   }\n'
        '}\n'
    )
    s3 = (
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      y[i] = a[%s]*b[j] + y[i];\n' % (i*n + j) +\
        '   }\n'
        '}\n'
    )
    c = ccode(b[j]*a[i, j] + b[k]*a[i, k] + b[j]*b[k]*c[i, j, k], assign_to=y[i])
    assert (c == s0 + s1 + s2 + s3[:-1] or
            c == s0 + s1 + s3 + s2[:-1] or
            c == s0 + s2 + s1 + s3[:-1] or
            c == s0 + s2 + s3 + s1[:-1] or
            c == s0 + s3 + s1 + s2[:-1] or
            c == s0 + s3 + s2 + s1[:-1])


def test_dereference_printing():
    expr = x + y + sin(z) + z
    assert ccode(expr, dereference=[z]) == "x + y + (*z) + sin((*z))"


def test_Matrix_printing():
    # Test returning a Matrix
    mat = Matrix([x*y, Piecewise((2 + x, y>0), (y, True)), sin(z)])
    A = MatrixSymbol('A', 3, 1)
    assert ccode(mat, A) == (
        "A[0] = x*y;\n"
        "if (y > 0) {\n"
        "   A[1] = x + 2;\n"
        "}\n"
        "else {\n"
        "   A[1] = y;\n"
        "}\n"
        "A[2] = sin(z);")
    # Test using MatrixElements in expressions
    expr = Piecewise((2*A[2, 0], x > 0), (A[2, 0], True)) + sin(A[1, 0]) + A[0, 0]
    assert ccode(expr) == (
        "((x > 0) ? (\n"
        "   2*A[2]\n"
        ")\n"
        ": (\n"
        "   A[2]\n"
        ")) + sin(A[1]) + A[0]")
    # Test using MatrixElements in a Matrix
    q = MatrixSymbol('q', 5, 1)
    M = MatrixSymbol('M', 3, 3)
    m = Matrix([[sin(q[1,0]), 0, cos(q[2,0])],
        [q[1,0] + q[2,0], q[3, 0], 5],
        [2*q[4, 0]/q[1,0], sqrt(q[0,0]) + 4, 0]])
    assert ccode(m, M) == (
        "M[0] = sin(q[1]);\n"
        "M[1] = 0;\n"
        "M[2] = cos(q[2]);\n"
        "M[3] = q[1] + q[2];\n"
        "M[4] = q[3];\n"
        "M[5] = 5;\n"
        "M[6] = 2*q[4]/q[1];\n"
        "M[7] = sqrt(q[0]) + 4;\n"
        "M[8] = 0;")


def test_sparse_matrix():
    # gh-15791
    with raises(PrintMethodNotImplementedError):
        ccode(SparseMatrix([[1, 2, 3]]))

    assert 'Not supported in C' in C89CodePrinter({'strict': False}).doprint(SparseMatrix([[1, 2, 3]]))



def test_ccode_reserved_words():
    x, y = symbols('x, if')
    with raises(ValueError):
        ccode(y**2, error_on_reserved=True, standard='C99')
    assert ccode(y**2) == 'pow(if_, 2)'
    assert ccode(x * y**2, dereference=[y]) == 'pow((*if_), 2)*x'
    assert ccode(y**2, reserved_word_suffix='_unreserved') == 'pow(if_unreserved, 2)'


def test_ccode_sign():
    expr1, ref1 = sign(x) * y, 'y*(((x) > 0) - ((x) < 0))'
    expr2, ref2 = sign(cos(x)), '(((cos(x)) > 0) - ((cos(x)) < 0))'
    expr3, ref3 = sign(2 * x + x**2) * x + x**2, 'pow(x, 2) + x*(((pow(x, 2) + 2*x) > 0) - ((pow(x, 2) + 2*x) < 0))'
    assert ccode(expr1) == ref1
    assert ccode(expr1, 'z') == 'z = %s;' % ref1
    assert ccode(expr2) == ref2
    assert ccode(expr3) == ref3

def test_ccode_Assignment():
    assert ccode(Assignment(x, y + z)) == 'x = y + z;'
    assert ccode(aug_assign(x, '+', y + z)) == 'x += y + z;'


def test_ccode_For():
    f = For(x, Range(0, 10, 2), [aug_assign(y, '*', x)])
    assert ccode(f) == ("for (x = 0; x < 10; x += 2) {\n"
                        "   y *= x;\n"
                        "}")

def test_ccode_Max_Min():
    assert ccode(Max(x, 0), standard='C89') == '((0 > x) ? 0 : x)'
    assert ccode(Max(x, 0), standard='C99') == 'fmax(0, x)'
    assert ccode(Min(x, 0, sqrt(x)), standard='c89') == (
        '((0 < ((x < sqrt(x)) ? x : sqrt(x))) ? 0 : ((x < sqrt(x)) ? x : sqrt(x)))'
    )

def test_ccode_standard():
    assert ccode(expm1(x), standard='c99') == 'expm1(x)'
    assert ccode(nan, standard='c99') == 'NAN'
    assert ccode(float('nan'), standard='c99') == 'NAN'


def test_C89CodePrinter():
    c89printer = C89CodePrinter()
    assert c89printer.language == 'C'
    assert c89printer.standard == 'C89'
    assert 'void' in c89printer.reserved_words
    assert 'template' not in c89printer.reserved_words
    assert c89printer.doprint(log10(x)) == 'log10(x)'


def test_C99CodePrinter():
    assert C99CodePrinter().doprint(expm1(x)) == 'expm1(x)'
    assert C99CodePrinter().doprint(log1p(x)) == 'log1p(x)'
    assert C99CodePrinter().doprint(exp2(x)) == 'exp2(x)'
    assert C99CodePrinter().doprint(log2(x)) == 'log2(x)'
    assert C99CodePrinter().doprint(fma(x, y, -z)) == 'fma(x, y, -z)'
    assert C99CodePrinter().doprint(log10(x)) == 'log10(x)'
    assert C99CodePrinter().doprint(Cbrt(x)) == 'cbrt(x)'  # note Cbrt due to cbrt already taken.
    assert C99CodePrinter().doprint(hypot(x, y)) == 'hypot(x, y)'
    assert C99CodePrinter().doprint(loggamma(x)) == 'lgamma(x)'
    assert C99CodePrinter().doprint(Max(x, 3, x**2)) == 'fmax(3, fmax(x, pow(x, 2)))'
    assert C99CodePrinter().doprint(Min(x, 3)) == 'fmin(3, x)'
    c99printer = C99CodePrinter()
    assert c99printer.language == 'C'
    assert c99printer.standard == 'C99'
    assert 'restrict' in c99printer.reserved_words
    assert 'using' not in c99printer.reserved_words


@XFAIL
def test_C99CodePrinter__precision_f80():
    f80_printer = C99CodePrinter({"type_aliases": {real: float80}})
    assert f80_printer.doprint(sin(x + Float('2.1'))) == 'sinl(x + 2.1L)'


def test_C99CodePrinter__precision():
    n = symbols('n', integer=True)
    p = symbols('p', integer=True, positive=True)
    f32_printer = C99CodePrinter({"type_aliases": {real: float32}})
    f64_printer = C99CodePrinter({"type_aliases": {real: float64}})
    f80_printer = C99CodePrinter({"type_aliases": {real: float80}})
    assert f32_printer.doprint(sin(x+2.1)) == 'sinf(x + 2.1F)'
    assert f64_printer.doprint(sin(x+2.1)) == 'sin(x + 2.1000000000000001)'
    assert f80_printer.doprint(sin(x+Float('2.0'))) == 'sinl(x + 2.0L)'

    for printer, suffix in zip([f32_printer, f64_printer, f80_printer], ['f', '', 'l']):
        def check(expr, ref):
            assert printer.doprint(expr) == ref.format(s=suffix, S=suffix.upper())
        check(Abs(n), 'abs(n)')
        check(Abs(x + 2.0), 'fabs{s}(x + 2.0{S})')
        check(sin(x + 4.0)**cos(x - 2.0), 'pow{s}(sin{s}(x + 4.0{S}), cos{s}(x - 2.0{S}))')
        check(exp(x*8.0), 'exp{s}(8.0{S}*x)')
        check(exp2(x), 'exp2{s}(x)')
        check(expm1(x*4.0), 'expm1{s}(4.0{S}*x)')
        check(Mod(p, 2), 'p % 2')
        check(Mod(2*p + 3, 3*p + 5, evaluate=False), '(2*p + 3) % (3*p + 5)')
        check(Mod(x + 2.0, 3.0), 'fmod{s}(1.0{S}*x + 2.0{S}, 3.0{S})')
        check(Mod(x, 2.0*x + 3.0), 'fmod{s}(1.0{S}*x, 2.0{S}*x + 3.0{S})')
        check(log(x/2), 'log{s}((1.0{S}/2.0{S})*x)')
        check(log10(3*x/2), 'log10{s}((3.0{S}/2.0{S})*x)')
        check(log2(x*8.0), 'log2{s}(8.0{S}*x)')
        check(log1p(x), 'log1p{s}(x)')
        check(2**x, 'pow{s}(2, x)')
        check(2.0**x, 'pow{s}(2.0{S}, x)')
        check(x**3, 'pow{s}(x, 3)')
        check(x**4.0, 'pow{s}(x, 4.0{S})')
        check(sqrt(3+x), 'sqrt{s}(x + 3)')
        check(Cbrt(x-2.0), 'cbrt{s}(x - 2.0{S})')
        check(hypot(x, y), 'hypot{s}(x, y)')
        check(sin(3.*x + 2.), 'sin{s}(3.0{S}*x + 2.0{S})')
        check(cos(3.*x - 1.), 'cos{s}(3.0{S}*x - 1.0{S})')
        check(tan(4.*y + 2.), 'tan{s}(4.0{S}*y + 2.0{S})')
        check(asin(3.*x + 2.), 'asin{s}(3.0{S}*x + 2.0{S})')
        check(acos(3.*x + 2.), 'acos{s}(3.0{S}*x + 2.0{S})')
        check(atan(3.*x + 2.), 'atan{s}(3.0{S}*x + 2.0{S})')
        check(atan2(3.*x, 2.*y), 'atan2{s}(3.0{S}*x, 2.0{S}*y)')

        check(sinh(3.*x + 2.), 'sinh{s}(3.0{S}*x + 2.0{S})')
        check(cosh(3.*x - 1.), 'cosh{s}(3.0{S}*x - 1.0{S})')
        check(tanh(4.0*y + 2.), 'tanh{s}(4.0{S}*y + 2.0{S})')
        check(asinh(3.*x + 2.), 'asinh{s}(3.0{S}*x + 2.0{S})')
        check(acosh(3.*x + 2.), 'acosh{s}(3.0{S}*x + 2.0{S})')
        check(atanh(3.*x + 2.), 'atanh{s}(3.0{S}*x + 2.0{S})')
        check(erf(42.*x), 'erf{s}(42.0{S}*x)')
        check(erfc(42.*x), 'erfc{s}(42.0{S}*x)')
        check(gamma(x), 'tgamma{s}(x)')
        check(loggamma(x), 'lgamma{s}(x)')

        check(ceiling(x + 2.), "ceil{s}(x) + 2")
        check(floor(x + 2.), "floor{s}(x) + 2")
        check(fma(x, y, -z), 'fma{s}(x, y, -z)')
        check(Max(x, 8.0, x**4.0), 'fmax{s}(8.0{S}, fmax{s}(x, pow{s}(x, 4.0{S})))')
        check(Min(x, 2.0), 'fmin{s}(2.0{S}, x)')


def test_get_math_macros():
    macros = get_math_macros()
    assert macros[exp(1)] == 'M_E'
    assert macros[1/Sqrt(2)] == 'M_SQRT1_2'


def test_ccode_Declaration():
    i = symbols('i', integer=True)
    var1 = Variable(i, type=Type.from_expr(i))
    dcl1 = Declaration(var1)
    assert ccode(dcl1) == 'int i'

    var2 = Variable(x, type=float32, attrs={value_const})
    dcl2a = Declaration(var2)
    assert ccode(dcl2a) == 'const float x'
    dcl2b = var2.as_Declaration(value=pi)
    assert ccode(dcl2b) == 'const float x = M_PI'

    var3 = Variable(y, type=Type('bool'))
    dcl3 = Declaration(var3)
    printer = C89CodePrinter()
    assert 'stdbool.h' not in printer.headers
    assert printer.doprint(dcl3) == 'bool y'
    assert 'stdbool.h' in printer.headers

    u = symbols('u', real=True)
    ptr4 = Pointer.deduced(u, attrs={pointer_const, restrict})
    dcl4 = Declaration(ptr4)
    assert ccode(dcl4) == 'double * const restrict u'

    var5 = Variable(x, Type('__float128'), attrs={value_const})
    dcl5a = Declaration(var5)
    assert ccode(dcl5a) == 'const __float128 x'
    var5b = Variable(var5.symbol, var5.type, pi, attrs=var5.attrs)
    dcl5b = Declaration(var5b)
    assert ccode(dcl5b) == 'const __float128 x = M_PI'


def test_C99CodePrinter_custom_type():
    # We will look at __float128 (new in glibc 2.26)
    f128 = FloatType('_Float128', float128.nbits, float128.nmant, float128.nexp)
    p128 = C99CodePrinter({
        "type_aliases": {real: f128},
        "type_literal_suffixes": {f128: 'Q'},
        "type_func_suffixes": {f128: 'f128'},
        "type_math_macro_suffixes": {
            real: 'f128',
            f128: 'f128'
        },
        "type_macros": {
            f128: ('__STDC_WANT_IEC_60559_TYPES_EXT__',)
        }
    })
    assert p128.doprint(x) == 'x'
    assert not p128.headers
    assert not p128.libraries
    assert not p128.macros
    assert p128.doprint(2.0) == '2.0Q'
    assert not p128.headers
    assert not p128.libraries
    assert p128.macros == {'__STDC_WANT_IEC_60559_TYPES_EXT__'}

    assert p128.doprint(Rational(1, 2)) == '1.0Q/2.0Q'
    assert p128.doprint(sin(x)) == 'sinf128(x)'
    assert p128.doprint(cos(2., evaluate=False)) == 'cosf128(2.0Q)'
    assert p128.doprint(x**-1.0) == '1.0Q/x'

    var5 = Variable(x, f128, attrs={value_const})

    dcl5a = Declaration(var5)
    assert ccode(dcl5a) == 'const _Float128 x'
    var5b = Variable(x, f128, pi, attrs={value_const})
    dcl5b = Declaration(var5b)
    assert p128.doprint(dcl5b) == 'const _Float128 x = M_PIf128'
    var5b = Variable(x, f128, value=Catalan.evalf(38), attrs={value_const})
    dcl5c = Declaration(var5b)
    assert p128.doprint(dcl5c) == 'const _Float128 x = %sQ' % Catalan.evalf(f128.decimal_dig)


def test_MatrixElement_printing():
    # test cases for issue #11821
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)
    C = MatrixSymbol("C", 1, 3)

    assert(ccode(A[0, 0]) == "A[0]")
    assert(ccode(3 * A[0, 0]) == "3*A[0]")

    F = C[0, 0].subs(C, A - B)
    assert(ccode(F) == "(A - B)[0]")

def test_ccode_math_macros():
    assert ccode(z + exp(1)) == 'z + M_E'
    assert ccode(z + log2(exp(1))) == 'z + M_LOG2E'
    assert ccode(z + 1/log(2)) == 'z + M_LOG2E'
    assert ccode(z + log(2)) == 'z + M_LN2'
    assert ccode(z + log(10)) == 'z + M_LN10'
    assert ccode(z + pi) == 'z + M_PI'
    assert ccode(z + pi/2) == 'z + M_PI_2'
    assert ccode(z + pi/4) == 'z + M_PI_4'
    assert ccode(z + 1/pi) == 'z + M_1_PI'
    assert ccode(z + 2/pi) == 'z + M_2_PI'
    assert ccode(z + 2/sqrt(pi)) == 'z + M_2_SQRTPI'
    assert ccode(z + 2/Sqrt(pi)) == 'z + M_2_SQRTPI'
    assert ccode(z + sqrt(2)) == 'z + M_SQRT2'
    assert ccode(z + Sqrt(2)) == 'z + M_SQRT2'
    assert ccode(z + 1/sqrt(2)) == 'z + M_SQRT1_2'
    assert ccode(z + 1/Sqrt(2)) == 'z + M_SQRT1_2'


def test_ccode_Type():
    assert ccode(Type('float')) == 'float'
    assert ccode(intc) == 'int'


def test_ccode_codegen_ast():
    # Note that C only allows comments of the form /* ... */, double forward
    # slash is not standard C, and some C compilers will grind to a halt upon
    # encountering them.
    assert ccode(Comment("this is a comment")) == "/* this is a comment */"  # not //
    assert ccode(While(abs(x) > 1, [aug_assign(x, '-', 1)])) == (
        'while (fabs(x) > 1) {\n'
        '   x -= 1;\n'
        '}'
    )
    assert ccode(Scope([AddAugmentedAssignment(x, 1)])) == (
        '{\n'
        '   x += 1;\n'
        '}'
    )
    inp_x = Declaration(Variable(x, type=real))
    assert ccode(FunctionPrototype(real, 'pwer', [inp_x])) == 'double pwer(double x)'
    assert ccode(FunctionDefinition(real, 'pwer', [inp_x], [Assignment(x, x**2)])) == (
        'double pwer(double x){\n'
        '   x = pow(x, 2);\n'
        '}'
    )

    # Elements of CodeBlock are formatted as statements:
    block = CodeBlock(
        x,
        Print([x, y], "%d %d"),
        Print([QuotedString('hello'), y], "%s %d", file=stderr),
        FunctionCall('pwer', [x]),
        Return(x),
    )
    assert ccode(block) == '\n'.join([
        'x;',
        'printf("%d %d", x, y);',
        'fprintf(stderr, "%s %d", "hello", y);',
        'pwer(x);',
        'return x;',
    ])

def test_ccode_UnevaluatedExpr():
    assert ccode(UnevaluatedExpr(y * x) + z) == "z + x*y"
    assert ccode(UnevaluatedExpr(y + x) + z) == "z + (x + y)"  # gh-21955
    w = symbols('w')
    assert ccode(UnevaluatedExpr(y + x) + UnevaluatedExpr(z + w)) == "(w + z) + (x + y)"

    p, q, r = symbols("p q r", real=True)
    q_r = UnevaluatedExpr(q + r)
    expr = abs(exp(p+q_r))
    assert ccode(expr) == "exp(p + (q + r))"


def test_ccode_array_like_containers():
    assert ccode([2,3,4]) == "{2, 3, 4}"
    assert ccode((2,3,4)) == "{2, 3, 4}"

def test_ccode__isinf_isnan():
    assert ccode(isinf(x)) == 'isinf(x)'
    assert ccode(isnan(x)) == 'isnan(x)'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\test_crosstab.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    CategoricalDtype,
    CategoricalIndex,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    crosstab,
)
import pandas._testing as tm


@pytest.fixture
def df():
    df = DataFrame(
        {
            "A": [
                "foo",
                "foo",
                "foo",
                "foo",
                "bar",
                "bar",
                "bar",
                "bar",
                "foo",
                "foo",
                "foo",
            ],
            "B": [
                "one",
                "one",
                "one",
                "two",
                "one",
                "one",
                "one",
                "two",
                "two",
                "two",
                "one",
            ],
            "C": [
                "dull",
                "dull",
                "shiny",
                "dull",
                "dull",
                "shiny",
                "shiny",
                "dull",
                "shiny",
                "shiny",
                "shiny",
            ],
            "D": np.random.default_rng(2).standard_normal(11),
            "E": np.random.default_rng(2).standard_normal(11),
            "F": np.random.default_rng(2).standard_normal(11),
        }
    )

    return pd.concat([df, df], ignore_index=True)


class TestCrosstab:
    def test_crosstab_single(self, df):
        result = crosstab(df["A"], df["C"])
        expected = df.groupby(["A", "C"]).size().unstack()
        tm.assert_frame_equal(result, expected.fillna(0).astype(np.int64))

    def test_crosstab_multiple(self, df):
        result = crosstab(df["A"], [df["B"], df["C"]])
        expected = df.groupby(["A", "B", "C"]).size()
        expected = expected.unstack("B").unstack("C").fillna(0).astype(np.int64)
        tm.assert_frame_equal(result, expected)

        result = crosstab([df["B"], df["C"]], df["A"])
        expected = df.groupby(["B", "C", "A"]).size()
        expected = expected.unstack("A").fillna(0).astype(np.int64)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("box", [np.array, list, tuple])
    def test_crosstab_ndarray(self, box):
        # GH 44076
        a = box(np.random.default_rng(2).integers(0, 5, size=100))
        b = box(np.random.default_rng(2).integers(0, 3, size=100))
        c = box(np.random.default_rng(2).integers(0, 10, size=100))

        df = DataFrame({"a": a, "b": b, "c": c})

        result = crosstab(a, [b, c], rownames=["a"], colnames=("b", "c"))
        expected = crosstab(df["a"], [df["b"], df["c"]])
        tm.assert_frame_equal(result, expected)

        result = crosstab([b, c], a, colnames=["a"], rownames=("b", "c"))
        expected = crosstab([df["b"], df["c"]], df["a"])
        tm.assert_frame_equal(result, expected)

        # assign arbitrary names
        result = crosstab(a, c)
        expected = crosstab(df["a"], df["c"])
        expected.index.names = ["row_0"]
        expected.columns.names = ["col_0"]
        tm.assert_frame_equal(result, expected)

    def test_crosstab_non_aligned(self):
        # GH 17005
        a = Series([0, 1, 1], index=["a", "b", "c"])
        b = Series([3, 4, 3, 4, 3], index=["a", "b", "c", "d", "f"])
        c = np.array([3, 4, 3], dtype=np.int64)

        expected = DataFrame(
            [[1, 0], [1, 1]],
            index=Index([0, 1], name="row_0"),
            columns=Index([3, 4], name="col_0"),
        )

        result = crosstab(a, b)
        tm.assert_frame_equal(result, expected)

        result = crosstab(a, c)
        tm.assert_frame_equal(result, expected)

    def test_crosstab_margins(self):
        a = np.random.default_rng(2).integers(0, 7, size=100)
        b = np.random.default_rng(2).integers(0, 3, size=100)
        c = np.random.default_rng(2).integers(0, 5, size=100)

        df = DataFrame({"a": a, "b": b, "c": c})

        result = crosstab(a, [b, c], rownames=["a"], colnames=("b", "c"), margins=True)

        assert result.index.names == ("a",)
        assert result.columns.names == ["b", "c"]

        all_cols = result["All", ""]
        exp_cols = df.groupby(["a"]).size().astype("i8")
        # to keep index.name
        exp_margin = Series([len(df)], index=Index(["All"], name="a"))
        exp_cols = pd.concat([exp_cols, exp_margin])
        exp_cols.name = ("All", "")

        tm.assert_series_equal(all_cols, exp_cols)

        all_rows = result.loc["All"]
        exp_rows = df.groupby(["b", "c"]).size().astype("i8")
        exp_rows = pd.concat([exp_rows, Series([len(df)], index=[("All", "")])])
        exp_rows.name = "All"

        exp_rows = exp_rows.reindex(all_rows.index)
        exp_rows = exp_rows.fillna(0).astype(np.int64)
        tm.assert_series_equal(all_rows, exp_rows)

    def test_crosstab_margins_set_margin_name(self):
        # GH 15972
        a = np.random.default_rng(2).integers(0, 7, size=100)
        b = np.random.default_rng(2).integers(0, 3, size=100)
        c = np.random.default_rng(2).integers(0, 5, size=100)

        df = DataFrame({"a": a, "b": b, "c": c})

        result = crosstab(
            a,
            [b, c],
            rownames=["a"],
            colnames=("b", "c"),
            margins=True,
            margins_name="TOTAL",
        )

        assert result.index.names == ("a",)
        assert result.columns.names == ["b", "c"]

        all_cols = result["TOTAL", ""]
        exp_cols = df.groupby(["a"]).size().astype("i8")
        # to keep index.name
        exp_margin = Series([len(df)], index=Index(["TOTAL"], name="a"))
        exp_cols = pd.concat([exp_cols, exp_margin])
        exp_cols.name = ("TOTAL", "")

        tm.assert_series_equal(all_cols, exp_cols)

        all_rows = result.loc["TOTAL"]
        exp_rows = df.groupby(["b", "c"]).size().astype("i8")
        exp_rows = pd.concat([exp_rows, Series([len(df)], index=[("TOTAL", "")])])
        exp_rows.name = "TOTAL"

        exp_rows = exp_rows.reindex(all_rows.index)
        exp_rows = exp_rows.fillna(0).astype(np.int64)
        tm.assert_series_equal(all_rows, exp_rows)

        msg = "margins_name argument must be a string"
        for margins_name in [666, None, ["a", "b"]]:
            with pytest.raises(ValueError, match=msg):
                crosstab(
                    a,
                    [b, c],
                    rownames=["a"],
                    colnames=("b", "c"),
                    margins=True,
                    margins_name=margins_name,
                )

    def test_crosstab_pass_values(self):
        a = np.random.default_rng(2).integers(0, 7, size=100)
        b = np.random.default_rng(2).integers(0, 3, size=100)
        c = np.random.default_rng(2).integers(0, 5, size=100)
        values = np.random.default_rng(2).standard_normal(100)

        table = crosstab(
            [a, b], c, values, aggfunc="sum", rownames=["foo", "bar"], colnames=["baz"]
        )

        df = DataFrame({"foo": a, "bar": b, "baz": c, "values": values})

        expected = df.pivot_table(
            "values", index=["foo", "bar"], columns="baz", aggfunc="sum"
        )
        tm.assert_frame_equal(table, expected)

    def test_crosstab_dropna(self):
        # GH 3820
        a = np.array(["foo", "foo", "foo", "bar", "bar", "foo", "foo"], dtype=object)
        b = np.array(["one", "one", "two", "one", "two", "two", "two"], dtype=object)
        c = np.array(
            ["dull", "dull", "dull", "dull", "dull", "shiny", "shiny"], dtype=object
        )
        res = crosstab(a, [b, c], rownames=["a"], colnames=["b", "c"], dropna=False)
        m = MultiIndex.from_tuples(
            [("one", "dull"), ("one", "shiny"), ("two", "dull"), ("two", "shiny")],
            names=["b", "c"],
        )
        tm.assert_index_equal(res.columns, m)

    def test_crosstab_no_overlap(self):
        # GS 10291

        s1 = Series([1, 2, 3], index=[1, 2, 3])
        s2 = Series([4, 5, 6], index=[4, 5, 6])

        actual = crosstab(s1, s2)
        expected = DataFrame(
            index=Index([], dtype="int64", name="row_0"),
            columns=Index([], dtype="int64", name="col_0"),
        )

        tm.assert_frame_equal(actual, expected)

    def test_margin_dropna(self):
        # GH 12577
        # pivot_table counts null into margin ('All')
        # when margins=true and dropna=true

        df = DataFrame({"a": [1, 2, 2, 2, 2, np.nan], "b": [3, 3, 4, 4, 4, 4]})
        actual = crosstab(df.a, df.b, margins=True, dropna=True)
        expected = DataFrame([[1, 0, 1], [1, 3, 4], [2, 3, 5]])
        expected.index = Index([1.0, 2.0, "All"], name="a")
        expected.columns = Index([3, 4, "All"], name="b")
        tm.assert_frame_equal(actual, expected)

    def test_margin_dropna2(self):
        df = DataFrame(
            {"a": [1, np.nan, np.nan, np.nan, 2, np.nan], "b": [3, np.nan, 4, 4, 4, 4]}
        )
        actual = crosstab(df.a, df.b, margins=True, dropna=True)
        expected = DataFrame([[1, 0, 1], [0, 1, 1], [1, 1, 2]])
        expected.index = Index([1.0, 2.0, "All"], name="a")
        expected.columns = Index([3.0, 4.0, "All"], name="b")
        tm.assert_frame_equal(actual, expected)

    def test_margin_dropna3(self):
        df = DataFrame(
            {"a": [1, np.nan, np.nan, np.nan, np.nan, 2], "b": [3, 3, 4, 4, 4, 4]}
        )
        actual = crosstab(df.a, df.b, margins=True, dropna=True)
        expected = DataFrame([[1, 0, 1], [0, 1, 1], [1, 1, 2]])
        expected.index = Index([1.0, 2.0, "All"], name="a")
        expected.columns = Index([3, 4, "All"], name="b")
        tm.assert_frame_equal(actual, expected)

    def test_margin_dropna4(self):
        # GH 12642
        # _add_margins raises KeyError: Level None not found
        # when margins=True and dropna=False
        # GH: 10772: Keep np.nan in result with dropna=False
        df = DataFrame({"a": [1, 2, 2, 2, 2, np.nan], "b": [3, 3, 4, 4, 4, 4]})
        actual = crosstab(df.a, df.b, margins=True, dropna=False)
        expected = DataFrame([[1, 0, 1.0], [1, 3, 4.0], [0, 1, np.nan], [2, 4, 6.0]])
        expected.index = Index([1.0, 2.0, np.nan, "All"], name="a")
        expected.columns = Index([3, 4, "All"], name="b")
        tm.assert_frame_equal(actual, expected)

    def test_margin_dropna5(self):
        # GH: 10772: Keep np.nan in result with dropna=False
        df = DataFrame(
            {"a": [1, np.nan, np.nan, np.nan, 2, np.nan], "b": [3, np.nan, 4, 4, 4, 4]}
        )
        actual = crosstab(df.a, df.b, margins=True, dropna=False)
        expected = DataFrame(
            [[1, 0, 0, 1.0], [0, 1, 0, 1.0], [0, 3, 1, np.nan], [1, 4, 0, 6.0]]
        )
        expected.index = Index([1.0, 2.0, np.nan, "All"], name="a")
        expected.columns = Index([3.0, 4.0, np.nan, "All"], name="b")
        tm.assert_frame_equal(actual, expected)

    def test_margin_dropna6(self):
        # GH: 10772: Keep np.nan in result with dropna=False
        a = np.array(["foo", "foo", "foo", "bar", "bar", "foo", "foo"], dtype=object)
        b = np.array(["one", "one", "two", "one", "two", np.nan, "two"], dtype=object)
        c = np.array(
            ["dull", "dull", "dull", "dull", "dull", "shiny", "shiny"], dtype=object
        )

        actual = crosstab(
            a, [b, c], rownames=["a"], colnames=["b", "c"], margins=True, dropna=False
        )
        m = MultiIndex.from_arrays(
            [
                ["one", "one", "two", "two", np.nan, np.nan, "All"],
                ["dull", "shiny", "dull", "shiny", "dull", "shiny", ""],
            ],
            names=["b", "c"],
        )
        expected = DataFrame(
            [[1, 0, 1, 0, 0, 0, 2], [2, 0, 1, 1, 0, 1, 5], [3, 0, 2, 1, 0, 0, 7]],
            columns=m,
        )
        expected.index = Index(["bar", "foo", "All"], name="a")
        tm.assert_frame_equal(actual, expected)

        actual = crosstab(
            [a, b], c, rownames=["a", "b"], colnames=["c"], margins=True, dropna=False
        )
        m = MultiIndex.from_arrays(
            [
                ["bar", "bar", "bar", "foo", "foo", "foo", "All"],
                ["one", "two", np.nan, "one", "two", np.nan, ""],
            ],
            names=["a", "b"],
        )
        expected = DataFrame(
            [
                [1, 0, 1.0],
                [1, 0, 1.0],
                [0, 0, np.nan],
                [2, 0, 2.0],
                [1, 1, 2.0],
                [0, 1, np.nan],
                [5, 2, 7.0],
            ],
            index=m,
        )
        expected.columns = Index(["dull", "shiny", "All"], name="c")
        tm.assert_frame_equal(actual, expected)

        actual = crosstab(
            [a, b], c, rownames=["a", "b"], colnames=["c"], margins=True, dropna=True
        )
        m = MultiIndex.from_arrays(
            [["bar", "bar", "foo", "foo", "All"], ["one", "two", "one", "two", ""]],
            names=["a", "b"],
        )
        expected = DataFrame(
            [[1, 0, 1], [1, 0, 1], [2, 0, 2], [1, 1, 2], [5, 1, 6]], index=m
        )
        expected.columns = Index(["dull", "shiny", "All"], name="c")
        tm.assert_frame_equal(actual, expected)

    def test_crosstab_normalize(self):
        # Issue 12578
        df = DataFrame(
            {"a": [1, 2, 2, 2, 2], "b": [3, 3, 4, 4, 4], "c": [1, 1, np.nan, 1, 1]}
        )

        rindex = Index([1, 2], name="a")
        cindex = Index([3, 4], name="b")
        full_normal = DataFrame([[0.2, 0], [0.2, 0.6]], index=rindex, columns=cindex)
        row_normal = DataFrame([[1.0, 0], [0.25, 0.75]], index=rindex, columns=cindex)
        col_normal = DataFrame([[0.5, 0], [0.5, 1.0]], index=rindex, columns=cindex)

        # Check all normalize args
        tm.assert_frame_equal(crosstab(df.a, df.b, normalize="all"), full_normal)
        tm.assert_frame_equal(crosstab(df.a, df.b, normalize=True), full_normal)
        tm.assert_frame_equal(crosstab(df.a, df.b, normalize="index"), row_normal)
        tm.assert_frame_equal(crosstab(df.a, df.b, normalize="columns"), col_normal)
        tm.assert_frame_equal(
            crosstab(df.a, df.b, normalize=1),
            crosstab(df.a, df.b, normalize="columns"),
        )
        tm.assert_frame_equal(
            crosstab(df.a, df.b, normalize=0), crosstab(df.a, df.b, normalize="index")
        )

        row_normal_margins = DataFrame(
            [[1.0, 0], [0.25, 0.75], [0.4, 0.6]],
            index=Index([1, 2, "All"], name="a", dtype="object"),
            columns=Index([3, 4], name="b", dtype="object"),
        )
        col_normal_margins = DataFrame(
            [[0.5, 0, 0.2], [0.5, 1.0, 0.8]],
            index=Index([1, 2], name="a", dtype="object"),
            columns=Index([3, 4, "All"], name="b", dtype="object"),
        )

        all_normal_margins = DataFrame(
            [[0.2, 0, 0.2], [0.2, 0.6, 0.8], [0.4, 0.6, 1]],
            index=Index([1, 2, "All"], name="a", dtype="object"),
            columns=Index([3, 4, "All"], name="b", dtype="object"),
        )
        tm.assert_frame_equal(
            crosstab(df.a, df.b, normalize="index", margins=True), row_normal_margins
        )
        tm.assert_frame_equal(
            crosstab(df.a, df.b, normalize="columns", margins=True), col_normal_margins
        )
        tm.assert_frame_equal(
            crosstab(df.a, df.b, normalize=True, margins=True), all_normal_margins
        )

    def test_crosstab_normalize_arrays(self):
        # GH#12578
        df = DataFrame(
            {"a": [1, 2, 2, 2, 2], "b": [3, 3, 4, 4, 4], "c": [1, 1, np.nan, 1, 1]}
        )

        # Test arrays
        crosstab(
            [np.array([1, 1, 2, 2]), np.array([1, 2, 1, 2])], np.array([1, 2, 1, 2])
        )

        # Test with aggfunc
        norm_counts = DataFrame(
            [[0.25, 0, 0.25], [0.25, 0.5, 0.75], [0.5, 0.5, 1]],
            index=Index([1, 2, "All"], name="a", dtype="object"),
            columns=Index([3, 4, "All"], name="b"),
        )
        test_case = crosstab(
            df.a, df.b, df.c, aggfunc="count", normalize="all", margins=True
        )
        tm.assert_frame_equal(test_case, norm_counts)

        df = DataFrame(
            {"a": [1, 2, 2, 2, 2], "b": [3, 3, 4, 4, 4], "c": [0, 4, np.nan, 3, 3]}
        )

        norm_sum = DataFrame(
            [[0, 0, 0.0], [0.4, 0.6, 1], [0.4, 0.6, 1]],
            index=Index([1, 2, "All"], name="a", dtype="object"),
            columns=Index([3, 4, "All"], name="b", dtype="object"),
        )
        msg = "using DataFrameGroupBy.sum"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            test_case = crosstab(
                df.a, df.b, df.c, aggfunc=np.sum, normalize="all", margins=True
            )
        tm.assert_frame_equal(test_case, norm_sum)

    def test_crosstab_with_empties(self, using_array_manager):
        # Check handling of empties
        df = DataFrame(
            {
                "a": [1, 2, 2, 2, 2],
                "b": [3, 3, 4, 4, 4],
                "c": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )

        empty = DataFrame(
            [[0.0, 0.0], [0.0, 0.0]],
            index=Index([1, 2], name="a", dtype="int64"),
            columns=Index([3, 4], name="b"),
        )

        for i in [True, "index", "columns"]:
            calculated = crosstab(df.a, df.b, values=df.c, aggfunc="count", normalize=i)
            tm.assert_frame_equal(empty, calculated)

        nans = DataFrame(
            [[0.0, np.nan], [0.0, 0.0]],
            index=Index([1, 2], name="a", dtype="int64"),
            columns=Index([3, 4], name="b"),
        )
        if using_array_manager:
            # INFO(ArrayManager) column without NaNs can preserve int dtype
            nans[3] = nans[3].astype("int64")

        calculated = crosstab(df.a, df.b, values=df.c, aggfunc="count", normalize=False)
        tm.assert_frame_equal(nans, calculated)

    def test_crosstab_errors(self):
        # Issue 12578

        df = DataFrame(
            {"a": [1, 2, 2, 2, 2], "b": [3, 3, 4, 4, 4], "c": [1, 1, np.nan, 1, 1]}
        )

        error = "values cannot be used without an aggfunc."
        with pytest.raises(ValueError, match=error):
            crosstab(df.a, df.b, values=df.c)

        error = "aggfunc cannot be used without values"
        with pytest.raises(ValueError, match=error):
            crosstab(df.a, df.b, aggfunc=np.mean)

        error = "Not a valid normalize argument"
        with pytest.raises(ValueError, match=error):
            crosstab(df.a, df.b, normalize="42")

        with pytest.raises(ValueError, match=error):
            crosstab(df.a, df.b, normalize=42)

        error = "Not a valid margins argument"
        with pytest.raises(ValueError, match=error):
            crosstab(df.a, df.b, normalize="all", margins=42)

    def test_crosstab_with_categorial_columns(self):
        # GH 8860
        df = DataFrame(
            {
                "MAKE": ["Honda", "Acura", "Tesla", "Honda", "Honda", "Acura"],
                "MODEL": ["Sedan", "Sedan", "Electric", "Pickup", "Sedan", "Sedan"],
            }
        )
        categories = ["Sedan", "Electric", "Pickup"]
        df["MODEL"] = df["MODEL"].astype("category").cat.set_categories(categories)
        result = crosstab(df["MAKE"], df["MODEL"])

        expected_index = Index(["Acura", "Honda", "Tesla"], name="MAKE")
        expected_columns = CategoricalIndex(
            categories, categories=categories, ordered=False, name="MODEL"
        )
        expected_data = [[2, 0, 0], [2, 0, 1], [0, 1, 0]]
        expected = DataFrame(
            expected_data, index=expected_index, columns=expected_columns
        )
        tm.assert_frame_equal(result, expected)

    def test_crosstab_with_numpy_size(self):
        # GH 4003
        df = DataFrame(
            {
                "A": ["one", "one", "two", "three"] * 6,
                "B": ["A", "B", "C"] * 8,
                "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 4,
                "D": np.random.default_rng(2).standard_normal(24),
                "E": np.random.default_rng(2).standard_normal(24),
            }
        )
        result = crosstab(
            index=[df["A"], df["B"]],
            columns=[df["C"]],
            margins=True,
            aggfunc=np.size,
            values=df["D"],
        )
        expected_index = MultiIndex(
            levels=[["All", "one", "three", "two"], ["", "A", "B", "C"]],
            codes=[[1, 1, 1, 2, 2, 2, 3, 3, 3, 0], [1, 2, 3, 1, 2, 3, 1, 2, 3, 0]],
            names=["A", "B"],
        )
        expected_column = Index(["bar", "foo", "All"], name="C")
        expected_data = np.array(
            [
                [2.0, 2.0, 4.0],
                [2.0, 2.0, 4.0],
                [2.0, 2.0, 4.0],
                [2.0, np.nan, 2.0],
                [np.nan, 2.0, 2.0],
                [2.0, np.nan, 2.0],
                [np.nan, 2.0, 2.0],
                [2.0, np.nan, 2.0],
                [np.nan, 2.0, 2.0],
                [12.0, 12.0, 24.0],
            ]
        )
        expected = DataFrame(
            expected_data, index=expected_index, columns=expected_column
        )
        # aggfunc is np.size, resulting in integers
        expected["All"] = expected["All"].astype("int64")
        tm.assert_frame_equal(result, expected)

    def test_crosstab_duplicate_names(self):
        # GH 13279 / 22529

        s1 = Series(range(3), name="foo")
        s2_foo = Series(range(1, 4), name="foo")
        s2_bar = Series(range(1, 4), name="bar")
        s3 = Series(range(3), name="waldo")

        # check result computed with duplicate labels against
        # result computed with unique labels, then relabelled
        mapper = {"bar": "foo"}

        # duplicate row, column labels
        result = crosstab(s1, s2_foo)
        expected = crosstab(s1, s2_bar).rename_axis(columns=mapper, axis=1)
        tm.assert_frame_equal(result, expected)

        # duplicate row, unique column labels
        result = crosstab([s1, s2_foo], s3)
        expected = crosstab([s1, s2_bar], s3).rename_axis(index=mapper, axis=0)
        tm.assert_frame_equal(result, expected)

        # unique row, duplicate column labels
        result = crosstab(s3, [s1, s2_foo])
        expected = crosstab(s3, [s1, s2_bar]).rename_axis(columns=mapper, axis=1)

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("names", [["a", ("b", "c")], [("a", "b"), "c"]])
    def test_crosstab_tuple_name(self, names):
        s1 = Series(range(3), name=names[0])
        s2 = Series(range(1, 4), name=names[1])

        mi = MultiIndex.from_arrays([range(3), range(1, 4)], names=names)
        expected = Series(1, index=mi).unstack(1, fill_value=0)

        result = crosstab(s1, s2)
        tm.assert_frame_equal(result, expected)

    def test_crosstab_both_tuple_names(self):
        # GH 18321
        s1 = Series(range(3), name=("a", "b"))
        s2 = Series(range(3), name=("c", "d"))

        expected = DataFrame(
            np.eye(3, dtype="int64"),
            index=Index(range(3), name=("a", "b")),
            columns=Index(range(3), name=("c", "d")),
        )
        result = crosstab(s1, s2)
        tm.assert_frame_equal(result, expected)

    def test_crosstab_unsorted_order(self):
        df = DataFrame({"b": [3, 1, 2], "a": [5, 4, 6]}, index=["C", "A", "B"])
        result = crosstab(df.index, [df.b, df.a])
        e_idx = Index(["A", "B", "C"], name="row_0")
        e_columns = MultiIndex.from_tuples([(1, 4), (2, 6), (3, 5)], names=["b", "a"])
        expected = DataFrame(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], index=e_idx, columns=e_columns
        )
        tm.assert_frame_equal(result, expected)

    def test_crosstab_normalize_multiple_columns(self):
        # GH 15150
        df = DataFrame(
            {
                "A": ["one", "one", "two", "three"] * 6,
                "B": ["A", "B", "C"] * 8,
                "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 4,
                "D": [0] * 24,
                "E": [0] * 24,
            }
        )

        msg = "using DataFrameGroupBy.sum"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = crosstab(
                [df.A, df.B],
                df.C,
                values=df.D,
                aggfunc=np.sum,
                normalize=True,
                margins=True,
            )
        expected = DataFrame(
            np.array([0] * 29 + [1], dtype=float).reshape(10, 3),
            columns=Index(["bar", "foo", "All"], name="C"),
            index=MultiIndex.from_tuples(
                [
                    ("one", "A"),
                    ("one", "B"),
                    ("one", "C"),
                    ("three", "A"),
                    ("three", "B"),
                    ("three", "C"),
                    ("two", "A"),
                    ("two", "B"),
                    ("two", "C"),
                    ("All", ""),
                ],
                names=["A", "B"],
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_margin_normalize(self):
        # GH 27500
        df = DataFrame(
            {
                "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                "C": [
                    "small",
                    "large",
                    "large",
                    "small",
                    "small",
                    "large",
                    "small",
                    "small",
                    "large",
                ],
                "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
            }
        )
        # normalize on index
        result = crosstab(
            [df.A, df.B], df.C, margins=True, margins_name="Sub-Total", normalize=0
        )
        expected = DataFrame(
            [[0.5, 0.5], [0.5, 0.5], [0.666667, 0.333333], [0, 1], [0.444444, 0.555556]]
        )
        expected.index = MultiIndex(
            levels=[["Sub-Total", "bar", "foo"], ["", "one", "two"]],
            codes=[[1, 1, 2, 2, 0], [1, 2, 1, 2, 0]],
            names=["A", "B"],
        )
        expected.columns = Index(["large", "small"], name="C")
        tm.assert_frame_equal(result, expected)

        # normalize on columns
        result = crosstab(
            [df.A, df.B], df.C, margins=True, margins_name="Sub-Total", normalize=1
        )
        expected = DataFrame(
            [
                [0.25, 0.2, 0.222222],
                [0.25, 0.2, 0.222222],
                [0.5, 0.2, 0.333333],
                [0, 0.4, 0.222222],
            ]
        )
        expected.columns = Index(["large", "small", "Sub-Total"], name="C")
        expected.index = MultiIndex(
            levels=[["bar", "foo"], ["one", "two"]],
            codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
            names=["A", "B"],
        )
        tm.assert_frame_equal(result, expected)

        # normalize on both index and column
        result = crosstab(
            [df.A, df.B], df.C, margins=True, margins_name="Sub-Total", normalize=True
        )
        expected = DataFrame(
            [
                [0.111111, 0.111111, 0.222222],
                [0.111111, 0.111111, 0.222222],
                [0.222222, 0.111111, 0.333333],
                [0.000000, 0.222222, 0.222222],
                [0.444444, 0.555555, 1],
            ]
        )
        expected.columns = Index(["large", "small", "Sub-Total"], name="C")
        expected.index = MultiIndex(
            levels=[["Sub-Total", "bar", "foo"], ["", "one", "two"]],
            codes=[[1, 1, 2, 2, 0], [1, 2, 1, 2, 0]],
            names=["A", "B"],
        )
        tm.assert_frame_equal(result, expected)

    def test_margin_normalize_multiple_columns(self):
        # GH 35144
        # use multiple columns with margins and normalization
        df = DataFrame(
            {
                "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                "C": [
                    "small",
                    "large",
                    "large",
                    "small",
                    "small",
                    "large",
                    "small",
                    "small",
                    "large",
                ],
                "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
            }
        )
        result = crosstab(
            index=df.C,
            columns=[df.A, df.B],
            margins=True,
            margins_name="margin",
            normalize=True,
        )
        expected = DataFrame(
            [
                [0.111111, 0.111111, 0.222222, 0.000000, 0.444444],
                [0.111111, 0.111111, 0.111111, 0.222222, 0.555556],
                [0.222222, 0.222222, 0.333333, 0.222222, 1.0],
            ],
            index=["large", "small", "margin"],
        )
        expected.columns = MultiIndex(
            levels=[["bar", "foo", "margin"], ["", "one", "two"]],
            codes=[[0, 0, 1, 1, 2], [1, 2, 1, 2, 0]],
            names=["A", "B"],
        )
        expected.index.name = "C"
        tm.assert_frame_equal(result, expected)

    def test_margin_support_Float(self):
        # GH 50313
        # use Float64 formats and function aggfunc with margins
        df = DataFrame(
            {"A": [1, 2, 2, 1], "B": [3, 3, 4, 5], "C": [-1.0, 10.0, 1.0, 10.0]},
            dtype="Float64",
        )
        result = crosstab(
            df["A"],
            df["B"],
            values=df["C"],
            aggfunc="sum",
            margins=True,
        )
        expected = DataFrame(
            [
                [-1.0, pd.NA, 10.0, 9.0],
                [10.0, 1.0, pd.NA, 11.0],
                [9.0, 1.0, 10.0, 20.0],
            ],
            index=Index([1.0, 2.0, "All"], dtype="object", name="A"),
            columns=Index([3.0, 4.0, 5.0, "All"], dtype="object", name="B"),
            dtype="Float64",
        )
        tm.assert_frame_equal(result, expected)

    def test_margin_with_ordered_categorical_column(self):
        # GH 25278
        df = DataFrame(
            {
                "First": ["B", "B", "C", "A", "B", "C"],
                "Second": ["C", "B", "B", "B", "C", "A"],
            }
        )
        df["First"] = df["First"].astype(CategoricalDtype(ordered=True))
        customized_categories_order = ["C", "A", "B"]
        df["First"] = df["First"].cat.reorder_categories(customized_categories_order)
        result = crosstab(df["First"], df["Second"], margins=True)

        expected_index = Index(["C", "A", "B", "All"], name="First")
        expected_columns = Index(["A", "B", "C", "All"], name="Second")
        expected_data = [[1, 1, 0, 2], [0, 1, 0, 1], [0, 1, 2, 3], [1, 3, 2, 6]]
        expected = DataFrame(
            expected_data, index=expected_index, columns=expected_columns
        )
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("a_dtype", ["category", "int64"])
@pytest.mark.parametrize("b_dtype", ["category", "int64"])
def test_categoricals(a_dtype, b_dtype):
    # https://github.com/pandas-dev/pandas/issues/37465
    g = np.random.default_rng(2)
    a = Series(g.integers(0, 3, size=100)).astype(a_dtype)
    b = Series(g.integers(0, 2, size=100)).astype(b_dtype)
    result = crosstab(a, b, margins=True, dropna=False)
    columns = Index([0, 1, "All"], dtype="object", name="col_0")
    index = Index([0, 1, 2, "All"], dtype="object", name="row_0")
    values = [[10, 18, 28], [23, 16, 39], [17, 16, 33], [50, 50, 100]]
    expected = DataFrame(values, index, columns)
    tm.assert_frame_equal(result, expected)

    # Verify when categorical does not have all values present
    a.loc[a == 1] = 2
    a_is_cat = isinstance(a.dtype, CategoricalDtype)
    assert not a_is_cat or a.value_counts().loc[1] == 0
    result = crosstab(a, b, margins=True, dropna=False)
    values = [[10, 18, 28], [0, 0, 0], [40, 32, 72], [50, 50, 100]]
    expected = DataFrame(values, index, columns)
    if not a_is_cat:
        expected = expected.loc[[0, 2, "All"]]
        expected["All"] = expected["All"].astype("int64")
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_shape_base.py ===
import pytest

import numpy as np
from numpy._core import (
    arange,
    array,
    atleast_1d,
    atleast_2d,
    atleast_3d,
    block,
    concatenate,
    hstack,
    newaxis,
    stack,
    vstack,
)
from numpy._core.shape_base import (
    _block_concatenate,
    _block_dispatcher,
    _block_setup,
    _block_slicing,
)
from numpy.exceptions import AxisError
from numpy.testing import (
    IS_PYPY,
    assert_,
    assert_array_equal,
    assert_equal,
    assert_raises,
    assert_raises_regex,
)


class TestAtleast1d:
    def test_0D_array(self):
        a = array(1)
        b = array(2)
        res = [atleast_1d(a), atleast_1d(b)]
        desired = [array([1]), array([2])]
        assert_array_equal(res, desired)

    def test_1D_array(self):
        a = array([1, 2])
        b = array([2, 3])
        res = [atleast_1d(a), atleast_1d(b)]
        desired = [array([1, 2]), array([2, 3])]
        assert_array_equal(res, desired)

    def test_2D_array(self):
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        res = [atleast_1d(a), atleast_1d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)

    def test_3D_array(self):
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        a = array([a, a])
        b = array([b, b])
        res = [atleast_1d(a), atleast_1d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)

    def test_r1array(self):
        """ Test to make sure equivalent Travis O's r1array function
        """
        assert_(atleast_1d(3).shape == (1,))
        assert_(atleast_1d(3j).shape == (1,))
        assert_(atleast_1d(3.0).shape == (1,))
        assert_(atleast_1d([[2, 3], [4, 5]]).shape == (2, 2))


class TestAtleast2d:
    def test_0D_array(self):
        a = array(1)
        b = array(2)
        res = [atleast_2d(a), atleast_2d(b)]
        desired = [array([[1]]), array([[2]])]
        assert_array_equal(res, desired)

    def test_1D_array(self):
        a = array([1, 2])
        b = array([2, 3])
        res = [atleast_2d(a), atleast_2d(b)]
        desired = [array([[1, 2]]), array([[2, 3]])]
        assert_array_equal(res, desired)

    def test_2D_array(self):
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        res = [atleast_2d(a), atleast_2d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)

    def test_3D_array(self):
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        a = array([a, a])
        b = array([b, b])
        res = [atleast_2d(a), atleast_2d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)

    def test_r2array(self):
        """ Test to make sure equivalent Travis O's r2array function
        """
        assert_(atleast_2d(3).shape == (1, 1))
        assert_(atleast_2d([3j, 1]).shape == (1, 2))
        assert_(atleast_2d([[[3, 1], [4, 5]], [[3, 5], [1, 2]]]).shape == (2, 2, 2))


class TestAtleast3d:
    def test_0D_array(self):
        a = array(1)
        b = array(2)
        res = [atleast_3d(a), atleast_3d(b)]
        desired = [array([[[1]]]), array([[[2]]])]
        assert_array_equal(res, desired)

    def test_1D_array(self):
        a = array([1, 2])
        b = array([2, 3])
        res = [atleast_3d(a), atleast_3d(b)]
        desired = [array([[[1], [2]]]), array([[[2], [3]]])]
        assert_array_equal(res, desired)

    def test_2D_array(self):
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        res = [atleast_3d(a), atleast_3d(b)]
        desired = [a[:, :, newaxis], b[:, :, newaxis]]
        assert_array_equal(res, desired)

    def test_3D_array(self):
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        a = array([a, a])
        b = array([b, b])
        res = [atleast_3d(a), atleast_3d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)


class TestHstack:
    def test_non_iterable(self):
        assert_raises(TypeError, hstack, 1)

    def test_empty_input(self):
        assert_raises(ValueError, hstack, ())

    def test_0D_array(self):
        a = array(1)
        b = array(2)
        res = hstack([a, b])
        desired = array([1, 2])
        assert_array_equal(res, desired)

    def test_1D_array(self):
        a = array([1])
        b = array([2])
        res = hstack([a, b])
        desired = array([1, 2])
        assert_array_equal(res, desired)

    def test_2D_array(self):
        a = array([[1], [2]])
        b = array([[1], [2]])
        res = hstack([a, b])
        desired = array([[1, 1], [2, 2]])
        assert_array_equal(res, desired)

    def test_generator(self):
        with pytest.raises(TypeError, match="arrays to stack must be"):
            hstack(np.arange(3) for _ in range(2))
        with pytest.raises(TypeError, match="arrays to stack must be"):
            hstack(x for x in np.ones((3, 2)))

    def test_casting_and_dtype(self):
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        res = np.hstack((a, b), casting="unsafe", dtype=np.int64)
        expected_res = np.array([1, 2, 3, 2, 3, 4])
        assert_array_equal(res, expected_res)

    def test_casting_and_dtype_type_error(self):
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        with pytest.raises(TypeError):
            hstack((a, b), casting="safe", dtype=np.int64)


class TestVstack:
    def test_non_iterable(self):
        assert_raises(TypeError, vstack, 1)

    def test_empty_input(self):
        assert_raises(ValueError, vstack, ())

    def test_0D_array(self):
        a = array(1)
        b = array(2)
        res = vstack([a, b])
        desired = array([[1], [2]])
        assert_array_equal(res, desired)

    def test_1D_array(self):
        a = array([1])
        b = array([2])
        res = vstack([a, b])
        desired = array([[1], [2]])
        assert_array_equal(res, desired)

    def test_2D_array(self):
        a = array([[1], [2]])
        b = array([[1], [2]])
        res = vstack([a, b])
        desired = array([[1], [2], [1], [2]])
        assert_array_equal(res, desired)

    def test_2D_array2(self):
        a = array([1, 2])
        b = array([1, 2])
        res = vstack([a, b])
        desired = array([[1, 2], [1, 2]])
        assert_array_equal(res, desired)

    def test_generator(self):
        with pytest.raises(TypeError, match="arrays to stack must be"):
            vstack(np.arange(3) for _ in range(2))

    def test_casting_and_dtype(self):
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        res = np.vstack((a, b), casting="unsafe", dtype=np.int64)
        expected_res = np.array([[1, 2, 3], [2, 3, 4]])
        assert_array_equal(res, expected_res)

    def test_casting_and_dtype_type_error(self):
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        with pytest.raises(TypeError):
            vstack((a, b), casting="safe", dtype=np.int64)


class TestConcatenate:
    def test_returns_copy(self):
        a = np.eye(3)
        b = np.concatenate([a])
        b[0, 0] = 2
        assert b[0, 0] != a[0, 0]

    def test_exceptions(self):
        # test axis must be in bounds
        for ndim in [1, 2, 3]:
            a = np.ones((1,) * ndim)
            np.concatenate((a, a), axis=0)  # OK
            assert_raises(AxisError, np.concatenate, (a, a), axis=ndim)
            assert_raises(AxisError, np.concatenate, (a, a), axis=-(ndim + 1))

        # Scalars cannot be concatenated
        assert_raises(ValueError, concatenate, (0,))
        assert_raises(ValueError, concatenate, (np.array(0),))

        # dimensionality must match
        assert_raises_regex(
            ValueError,
            r"all the input arrays must have same number of dimensions, but "
            r"the array at index 0 has 1 dimension\(s\) and the array at "
            r"index 1 has 2 dimension\(s\)",
            np.concatenate, (np.zeros(1), np.zeros((1, 1))))

        # test shapes must match except for concatenation axis
        a = np.ones((1, 2, 3))
        b = np.ones((2, 2, 3))
        axis = list(range(3))
        for i in range(3):
            np.concatenate((a, b), axis=axis[0])  # OK
            assert_raises_regex(
                ValueError,
                "all the input array dimensions except for the concatenation axis "
                f"must match exactly, but along dimension {i}, the array at "
                "index 0 has size 1 and the array at index 1 has size 2",
                np.concatenate, (a, b), axis=axis[1])
            assert_raises(ValueError, np.concatenate, (a, b), axis=axis[2])
            a = np.moveaxis(a, -1, 0)
            b = np.moveaxis(b, -1, 0)
            axis.append(axis.pop(0))

        # No arrays to concatenate raises ValueError
        assert_raises(ValueError, concatenate, ())

    def test_concatenate_axis_None(self):
        a = np.arange(4, dtype=np.float64).reshape((2, 2))
        b = list(range(3))
        c = ['x']
        r = np.concatenate((a, a), axis=None)
        assert_equal(r.dtype, a.dtype)
        assert_equal(r.ndim, 1)
        r = np.concatenate((a, b), axis=None)
        assert_equal(r.size, a.size + len(b))
        assert_equal(r.dtype, a.dtype)
        r = np.concatenate((a, b, c), axis=None, dtype="U")
        d = array(['0.0', '1.0', '2.0', '3.0',
                   '0', '1', '2', 'x'])
        assert_array_equal(r, d)

        out = np.zeros(a.size + len(b))
        r = np.concatenate((a, b), axis=None)
        rout = np.concatenate((a, b), axis=None, out=out)
        assert_(out is rout)
        assert_equal(r, rout)

    def test_large_concatenate_axis_None(self):
        # When no axis is given, concatenate uses flattened versions.
        # This also had a bug with many arrays (see gh-5979).
        x = np.arange(1, 100)
        r = np.concatenate(x, None)
        assert_array_equal(x, r)

        # Once upon a time, this was the same as `axis=None` now it fails
        # (with an unspecified error, as multiple things are wrong here)
        with pytest.raises(ValueError):
            np.concatenate(x, 100)

    def test_concatenate(self):
        # Test concatenate function
        # One sequence returns unmodified (but as array)
        r4 = list(range(4))
        assert_array_equal(concatenate((r4,)), r4)
        # Any sequence
        assert_array_equal(concatenate((tuple(r4),)), r4)
        assert_array_equal(concatenate((array(r4),)), r4)
        # 1D default concatenation
        r3 = list(range(3))
        assert_array_equal(concatenate((r4, r3)), r4 + r3)
        # Mixed sequence types
        assert_array_equal(concatenate((tuple(r4), r3)), r4 + r3)
        assert_array_equal(concatenate((array(r4), r3)), r4 + r3)
        # Explicit axis specification
        assert_array_equal(concatenate((r4, r3), 0), r4 + r3)
        # Including negative
        assert_array_equal(concatenate((r4, r3), -1), r4 + r3)
        # 2D
        a23 = array([[10, 11, 12], [13, 14, 15]])
        a13 = array([[0, 1, 2]])
        res = array([[10, 11, 12], [13, 14, 15], [0, 1, 2]])
        assert_array_equal(concatenate((a23, a13)), res)
        assert_array_equal(concatenate((a23, a13), 0), res)
        assert_array_equal(concatenate((a23.T, a13.T), 1), res.T)
        assert_array_equal(concatenate((a23.T, a13.T), -1), res.T)
        # Arrays much match shape
        assert_raises(ValueError, concatenate, (a23.T, a13.T), 0)
        # 3D
        res = arange(2 * 3 * 7).reshape((2, 3, 7))
        a0 = res[..., :4]
        a1 = res[..., 4:6]
        a2 = res[..., 6:]
        assert_array_equal(concatenate((a0, a1, a2), 2), res)
        assert_array_equal(concatenate((a0, a1, a2), -1), res)
        assert_array_equal(concatenate((a0.T, a1.T, a2.T), 0), res.T)

        out = res.copy()
        rout = concatenate((a0, a1, a2), 2, out=out)
        assert_(out is rout)
        assert_equal(res, rout)

    @pytest.mark.skipif(IS_PYPY, reason="PYPY handles sq_concat, nb_add differently than cpython")
    def test_operator_concat(self):
        import operator
        a = array([1, 2])
        b = array([3, 4])
        n = [1, 2]
        res = array([1, 2, 3, 4])
        assert_raises(TypeError, operator.concat, a, b)
        assert_raises(TypeError, operator.concat, a, n)
        assert_raises(TypeError, operator.concat, n, a)
        assert_raises(TypeError, operator.concat, a, 1)
        assert_raises(TypeError, operator.concat, 1, a)

    def test_bad_out_shape(self):
        a = array([1, 2])
        b = array([3, 4])

        assert_raises(ValueError, concatenate, (a, b), out=np.empty(5))
        assert_raises(ValueError, concatenate, (a, b), out=np.empty((4, 1)))
        assert_raises(ValueError, concatenate, (a, b), out=np.empty((1, 4)))
        concatenate((a, b), out=np.empty(4))

    @pytest.mark.parametrize("axis", [None, 0])
    @pytest.mark.parametrize("out_dtype", ["c8", "f4", "f8", ">f8", "i8", "S4"])
    @pytest.mark.parametrize("casting",
            ['no', 'equiv', 'safe', 'same_kind', 'unsafe'])
    def test_out_and_dtype(self, axis, out_dtype, casting):
        # Compare usage of `out=out` with `dtype=out.dtype`
        out = np.empty(4, dtype=out_dtype)
        to_concat = (array([1.1, 2.2]), array([3.3, 4.4]))

        if not np.can_cast(to_concat[0], out_dtype, casting=casting):
            with assert_raises(TypeError):
                concatenate(to_concat, out=out, axis=axis, casting=casting)
            with assert_raises(TypeError):
                concatenate(to_concat, dtype=out.dtype,
                            axis=axis, casting=casting)
        else:
            res_out = concatenate(to_concat, out=out,
                                  axis=axis, casting=casting)
            res_dtype = concatenate(to_concat, dtype=out.dtype,
                                    axis=axis, casting=casting)
            assert res_out is out
            assert_array_equal(out, res_dtype)
            assert res_dtype.dtype == out_dtype

        with assert_raises(TypeError):
            concatenate(to_concat, out=out, dtype=out_dtype, axis=axis)

    @pytest.mark.parametrize("axis", [None, 0])
    @pytest.mark.parametrize("string_dt", ["S", "U", "S0", "U0"])
    @pytest.mark.parametrize("arrs",
            [([0.],), ([0.], [1]), ([0], ["string"], [1.])])
    def test_dtype_with_promotion(self, arrs, string_dt, axis):
        # Note that U0 and S0 should be deprecated eventually and changed to
        # actually give the empty string result (together with `np.array`)
        res = np.concatenate(arrs, axis=axis, dtype=string_dt, casting="unsafe")
        # The actual dtype should be identical to a cast (of a double array):
        assert res.dtype == np.array(1.).astype(string_dt).dtype

    @pytest.mark.parametrize("axis", [None, 0])
    def test_string_dtype_does_not_inspect(self, axis):
        with pytest.raises(TypeError):
            np.concatenate(([None], [1]), dtype="S", axis=axis)
        with pytest.raises(TypeError):
            np.concatenate(([None], [1]), dtype="U", axis=axis)

    @pytest.mark.parametrize("axis", [None, 0])
    def test_subarray_error(self, axis):
        with pytest.raises(TypeError, match=".*subarray dtype"):
            np.concatenate(([1], [1]), dtype="(2,)i", axis=axis)


def test_stack():
    # non-iterable input
    assert_raises(TypeError, stack, 1)

    # 0d input
    for input_ in [(1, 2, 3),
                   [np.int32(1), np.int32(2), np.int32(3)],
                   [np.array(1), np.array(2), np.array(3)]]:
        assert_array_equal(stack(input_), [1, 2, 3])
    # 1d input examples
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    r1 = array([[1, 2, 3], [4, 5, 6]])
    assert_array_equal(np.stack((a, b)), r1)
    assert_array_equal(np.stack((a, b), axis=1), r1.T)
    # all input types
    assert_array_equal(np.stack([a, b]), r1)
    assert_array_equal(np.stack(array([a, b])), r1)
    # all shapes for 1d input
    arrays = [np.random.randn(3) for _ in range(10)]
    axes = [0, 1, -1, -2]
    expected_shapes = [(10, 3), (3, 10), (3, 10), (10, 3)]
    for axis, expected_shape in zip(axes, expected_shapes):
        assert_equal(np.stack(arrays, axis).shape, expected_shape)
    assert_raises_regex(AxisError, 'out of bounds', stack, arrays, axis=2)
    assert_raises_regex(AxisError, 'out of bounds', stack, arrays, axis=-3)
    # all shapes for 2d input
    arrays = [np.random.randn(3, 4) for _ in range(10)]
    axes = [0, 1, 2, -1, -2, -3]
    expected_shapes = [(10, 3, 4), (3, 10, 4), (3, 4, 10),
                       (3, 4, 10), (3, 10, 4), (10, 3, 4)]
    for axis, expected_shape in zip(axes, expected_shapes):
        assert_equal(np.stack(arrays, axis).shape, expected_shape)
    # empty arrays
    assert_(stack([[], [], []]).shape == (3, 0))
    assert_(stack([[], [], []], axis=1).shape == (0, 3))
    # out
    out = np.zeros_like(r1)
    np.stack((a, b), out=out)
    assert_array_equal(out, r1)
    # edge cases
    assert_raises_regex(ValueError, 'need at least one array', stack, [])
    assert_raises_regex(ValueError, 'must have the same shape',
                        stack, [1, np.arange(3)])
    assert_raises_regex(ValueError, 'must have the same shape',
                        stack, [np.arange(3), 1])
    assert_raises_regex(ValueError, 'must have the same shape',
                        stack, [np.arange(3), 1], axis=1)
    assert_raises_regex(ValueError, 'must have the same shape',
                        stack, [np.zeros((3, 3)), np.zeros(3)], axis=1)
    assert_raises_regex(ValueError, 'must have the same shape',
                        stack, [np.arange(2), np.arange(3)])

    # do not accept generators
    with pytest.raises(TypeError, match="arrays to stack must be"):
        stack(x for x in range(3))

    # casting and dtype test
    a = np.array([1, 2, 3])
    b = np.array([2.5, 3.5, 4.5])
    res = np.stack((a, b), axis=1, casting="unsafe", dtype=np.int64)
    expected_res = np.array([[1, 2], [2, 3], [3, 4]])
    assert_array_equal(res, expected_res)
    # casting and dtype with TypeError
    with assert_raises(TypeError):
        stack((a, b), dtype=np.int64, axis=1, casting="safe")


def test_unstack():
    a = np.arange(24).reshape((2, 3, 4))

    for stacks in [np.unstack(a),
                   np.unstack(a, axis=0),
                   np.unstack(a, axis=-3)]:
        assert isinstance(stacks, tuple)
        assert len(stacks) == 2
        assert_array_equal(stacks[0], a[0])
        assert_array_equal(stacks[1], a[1])

    for stacks in [np.unstack(a, axis=1),
                   np.unstack(a, axis=-2)]:
        assert isinstance(stacks, tuple)
        assert len(stacks) == 3
        assert_array_equal(stacks[0], a[:, 0])
        assert_array_equal(stacks[1], a[:, 1])
        assert_array_equal(stacks[2], a[:, 2])

    for stacks in [np.unstack(a, axis=2),
                   np.unstack(a, axis=-1)]:
        assert isinstance(stacks, tuple)
        assert len(stacks) == 4
        assert_array_equal(stacks[0], a[:, :, 0])
        assert_array_equal(stacks[1], a[:, :, 1])
        assert_array_equal(stacks[2], a[:, :, 2])
        assert_array_equal(stacks[3], a[:, :, 3])

    assert_raises(ValueError, np.unstack, a, axis=3)
    assert_raises(ValueError, np.unstack, a, axis=-4)
    assert_raises(ValueError, np.unstack, np.array(0), axis=0)


@pytest.mark.parametrize("axis", [0])
@pytest.mark.parametrize("out_dtype", ["c8", "f4", "f8", ">f8", "i8"])
@pytest.mark.parametrize("casting",
                         ['no', 'equiv', 'safe', 'same_kind', 'unsafe'])
def test_stack_out_and_dtype(axis, out_dtype, casting):
    to_concat = (array([1, 2]), array([3, 4]))
    res = array([[1, 2], [3, 4]])
    out = np.zeros_like(res)

    if not np.can_cast(to_concat[0], out_dtype, casting=casting):
        with assert_raises(TypeError):
            stack(to_concat, dtype=out_dtype,
                  axis=axis, casting=casting)
    else:
        res_out = stack(to_concat, out=out,
                        axis=axis, casting=casting)
        res_dtype = stack(to_concat, dtype=out_dtype,
                          axis=axis, casting=casting)
        assert res_out is out
        assert_array_equal(out, res_dtype)
        assert res_dtype.dtype == out_dtype

    with assert_raises(TypeError):
        stack(to_concat, out=out, dtype=out_dtype, axis=axis)


class TestBlock:
    @pytest.fixture(params=['block', 'force_concatenate', 'force_slicing'])
    def block(self, request):
        # blocking small arrays and large arrays go through different paths.
        # the algorithm is triggered depending on the number of element
        # copies required.
        # We define a test fixture that forces most tests to go through
        # both code paths.
        # Ultimately, this should be removed if a single algorithm is found
        # to be faster for both small and large arrays.
        def _block_force_concatenate(arrays):
            arrays, list_ndim, result_ndim, _ = _block_setup(arrays)
            return _block_concatenate(arrays, list_ndim, result_ndim)

        def _block_force_slicing(arrays):
            arrays, list_ndim, result_ndim, _ = _block_setup(arrays)
            return _block_slicing(arrays, list_ndim, result_ndim)

        if request.param == 'force_concatenate':
            return _block_force_concatenate
        elif request.param == 'force_slicing':
            return _block_force_slicing
        elif request.param == 'block':
            return block
        else:
            raise ValueError('Unknown blocking request. There is a typo in the tests.')

    def test_returns_copy(self, block):
        a = np.eye(3)
        b = block(a)
        b[0, 0] = 2
        assert b[0, 0] != a[0, 0]

    def test_block_total_size_estimate(self, block):
        _, _, _, total_size = _block_setup([1])
        assert total_size == 1

        _, _, _, total_size = _block_setup([[1]])
        assert total_size == 1

        _, _, _, total_size = _block_setup([[1, 1]])
        assert total_size == 2

        _, _, _, total_size = _block_setup([[1], [1]])
        assert total_size == 2

        _, _, _, total_size = _block_setup([[1, 2], [3, 4]])
        assert total_size == 4

    def test_block_simple_row_wise(self, block):
        a_2d = np.ones((2, 2))
        b_2d = 2 * a_2d
        desired = np.array([[1, 1, 2, 2],
                            [1, 1, 2, 2]])
        result = block([a_2d, b_2d])
        assert_equal(desired, result)

    def test_block_simple_column_wise(self, block):
        a_2d = np.ones((2, 2))
        b_2d = 2 * a_2d
        expected = np.array([[1, 1],
                             [1, 1],
                             [2, 2],
                             [2, 2]])
        result = block([[a_2d], [b_2d]])
        assert_equal(expected, result)

    def test_block_with_1d_arrays_row_wise(self, block):
        # # # 1-D vectors are treated as row arrays
        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        expected = np.array([1, 2, 3, 2, 3, 4])
        result = block([a, b])
        assert_equal(expected, result)

    def test_block_with_1d_arrays_multiple_rows(self, block):
        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        expected = np.array([[1, 2, 3, 2, 3, 4],
                             [1, 2, 3, 2, 3, 4]])
        result = block([[a, b], [a, b]])
        assert_equal(expected, result)

    def test_block_with_1d_arrays_column_wise(self, block):
        # # # 1-D vectors are treated as row arrays
        a_1d = np.array([1, 2, 3])
        b_1d = np.array([2, 3, 4])
        expected = np.array([[1, 2, 3],
                             [2, 3, 4]])
        result = block([[a_1d], [b_1d]])
        assert_equal(expected, result)

    def test_block_mixed_1d_and_2d(self, block):
        a_2d = np.ones((2, 2))
        b_1d = np.array([2, 2])
        result = block([[a_2d], [b_1d]])
        expected = np.array([[1, 1],
                             [1, 1],
                             [2, 2]])
        assert_equal(expected, result)

    def test_block_complicated(self, block):
        # a bit more complicated
        one_2d = np.array([[1, 1, 1]])
        two_2d = np.array([[2, 2, 2]])
        three_2d = np.array([[3, 3, 3, 3, 3, 3]])
        four_1d = np.array([4, 4, 4, 4, 4, 4])
        five_0d = np.array(5)
        six_1d = np.array([6, 6, 6, 6, 6])
        zero_2d = np.zeros((2, 6))

        expected = np.array([[1, 1, 1, 2, 2, 2],
                             [3, 3, 3, 3, 3, 3],
                             [4, 4, 4, 4, 4, 4],
                             [5, 6, 6, 6, 6, 6],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]])

        result = block([[one_2d, two_2d],
                        [three_2d],
                        [four_1d],
                        [five_0d, six_1d],
                        [zero_2d]])
        assert_equal(result, expected)

    def test_nested(self, block):
        one = np.array([1, 1, 1])
        two = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
        three = np.array([3, 3, 3])
        four = np.array([4, 4, 4])
        five = np.array(5)
        six = np.array([6, 6, 6, 6, 6])
        zero = np.zeros((2, 6))

        result = block([
            [
                block([
                   [one],
                   [three],
                   [four]
                ]),
                two
            ],
            [five, six],
            [zero]
        ])
        expected = np.array([[1, 1, 1, 2, 2, 2],
                             [3, 3, 3, 2, 2, 2],
                             [4, 4, 4, 2, 2, 2],
                             [5, 6, 6, 6, 6, 6],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]])

        assert_equal(result, expected)

    def test_3d(self, block):
        a000 = np.ones((2, 2, 2), int) * 1

        a100 = np.ones((3, 2, 2), int) * 2
        a010 = np.ones((2, 3, 2), int) * 3
        a001 = np.ones((2, 2, 3), int) * 4

        a011 = np.ones((2, 3, 3), int) * 5
        a101 = np.ones((3, 2, 3), int) * 6
        a110 = np.ones((3, 3, 2), int) * 7

        a111 = np.ones((3, 3, 3), int) * 8

        result = block([
            [
                [a000, a001],
                [a010, a011],
            ],
            [
                [a100, a101],
                [a110, a111],
            ]
        ])
        expected = array([[[1, 1, 4, 4, 4],
                           [1, 1, 4, 4, 4],
                           [3, 3, 5, 5, 5],
                           [3, 3, 5, 5, 5],
                           [3, 3, 5, 5, 5]],

                          [[1, 1, 4, 4, 4],
                           [1, 1, 4, 4, 4],
                           [3, 3, 5, 5, 5],
                           [3, 3, 5, 5, 5],
                           [3, 3, 5, 5, 5]],

                          [[2, 2, 6, 6, 6],
                           [2, 2, 6, 6, 6],
                           [7, 7, 8, 8, 8],
                           [7, 7, 8, 8, 8],
                           [7, 7, 8, 8, 8]],

                          [[2, 2, 6, 6, 6],
                           [2, 2, 6, 6, 6],
                           [7, 7, 8, 8, 8],
                           [7, 7, 8, 8, 8],
                           [7, 7, 8, 8, 8]],

                          [[2, 2, 6, 6, 6],
                           [2, 2, 6, 6, 6],
                           [7, 7, 8, 8, 8],
                           [7, 7, 8, 8, 8],
                           [7, 7, 8, 8, 8]]])

        assert_array_equal(result, expected)

    def test_block_with_mismatched_shape(self, block):
        a = np.array([0, 0])
        b = np.eye(2)
        assert_raises(ValueError, block, [a, b])
        assert_raises(ValueError, block, [b, a])

        to_block = [[np.ones((2, 3)), np.ones((2, 2))],
                    [np.ones((2, 2)), np.ones((2, 2))]]
        assert_raises(ValueError, block, to_block)

    def test_no_lists(self, block):
        assert_equal(block(1),         np.array(1))
        assert_equal(block(np.eye(3)), np.eye(3))

    def test_invalid_nesting(self, block):
        msg = 'depths are mismatched'
        assert_raises_regex(ValueError, msg, block, [1, [2]])
        assert_raises_regex(ValueError, msg, block, [1, []])
        assert_raises_regex(ValueError, msg, block, [[1], 2])
        assert_raises_regex(ValueError, msg, block, [[], 2])
        assert_raises_regex(ValueError, msg, block, [
            [[1], [2]],
            [[3, 4]],
            [5]  # missing brackets
        ])

    def test_empty_lists(self, block):
        assert_raises_regex(ValueError, 'empty', block, [])
        assert_raises_regex(ValueError, 'empty', block, [[]])
        assert_raises_regex(ValueError, 'empty', block, [[1], []])

    def test_tuple(self, block):
        assert_raises_regex(TypeError, 'tuple', block, ([1, 2], [3, 4]))
        assert_raises_regex(TypeError, 'tuple', block, [(1, 2), (3, 4)])

    def test_different_ndims(self, block):
        a = 1.
        b = 2 * np.ones((1, 2))
        c = 3 * np.ones((1, 1, 3))

        result = block([a, b, c])
        expected = np.array([[[1., 2., 2., 3., 3., 3.]]])

        assert_equal(result, expected)

    def test_different_ndims_depths(self, block):
        a = 1.
        b = 2 * np.ones((1, 2))
        c = 3 * np.ones((1, 2, 3))

        result = block([[a, b], [c]])
        expected = np.array([[[1., 2., 2.],
                              [3., 3., 3.],
                              [3., 3., 3.]]])

        assert_equal(result, expected)

    def test_block_memory_order(self, block):
        # 3D
        arr_c = np.zeros((3,) * 3, order='C')
        arr_f = np.zeros((3,) * 3, order='F')

        b_c = [[[arr_c, arr_c],
                [arr_c, arr_c]],
               [[arr_c, arr_c],
                [arr_c, arr_c]]]

        b_f = [[[arr_f, arr_f],
                [arr_f, arr_f]],
               [[arr_f, arr_f],
                [arr_f, arr_f]]]

        assert block(b_c).flags['C_CONTIGUOUS']
        assert block(b_f).flags['F_CONTIGUOUS']

        arr_c = np.zeros((3, 3), order='C')
        arr_f = np.zeros((3, 3), order='F')
        # 2D
        b_c = [[arr_c, arr_c],
               [arr_c, arr_c]]

        b_f = [[arr_f, arr_f],
               [arr_f, arr_f]]

        assert block(b_c).flags['C_CONTIGUOUS']
        assert block(b_f).flags['F_CONTIGUOUS']


def test_block_dispatcher():
    class ArrayLike:
        pass
    a = ArrayLike()
    b = ArrayLike()
    c = ArrayLike()
    assert_equal(list(_block_dispatcher(a)), [a])
    assert_equal(list(_block_dispatcher([a])), [a])
    assert_equal(list(_block_dispatcher([a, b])), [a, b])
    assert_equal(list(_block_dispatcher([[a], [b, [c]]])), [a, b, c])
    # don't recurse into non-lists
    assert_equal(list(_block_dispatcher((a, b))), [(a, b)])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\ode\tests\test_riccati.py ===
from sympy.core.random import randint
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.polys.polytools import Poly
from sympy.simplify.ratsimp import ratsimp
from sympy.solvers.ode.subscheck import checkodesol
from sympy.testing.pytest import slow
from sympy.solvers.ode.riccati import (riccati_normal, riccati_inverse_normal,
    riccati_reduced, match_riccati, inverse_transform_poly, limit_at_inf,
    check_necessary_conds, val_at_inf, construct_c_case_1,
    construct_c_case_2, construct_c_case_3, construct_d_case_4,
    construct_d_case_5, construct_d_case_6, rational_laurent_series,
    solve_riccati)

f = Function('f')
x = symbols('x')

# These are the functions used to generate the tests
# SHOULD NOT BE USED DIRECTLY IN TESTS

def rand_rational(maxint):
    return Rational(randint(-maxint, maxint), randint(1, maxint))


def rand_poly(x, degree, maxint):
    return Poly([rand_rational(maxint) for _ in range(degree+1)], x)


def rand_rational_function(x, degree, maxint):
    degnum = randint(1, degree)
    degden = randint(1, degree)
    num = rand_poly(x, degnum, maxint)
    den = rand_poly(x, degden, maxint)
    while den == Poly(0, x):
        den = rand_poly(x, degden, maxint)
    return num / den


def find_riccati_ode(ratfunc, x, yf):
    y = ratfunc
    yp = y.diff(x)
    q1 = rand_rational_function(x, 1, 3)
    q2 = rand_rational_function(x, 1, 3)
    while q2 == 0:
        q2 = rand_rational_function(x, 1, 3)
    q0 = ratsimp(yp - q1*y - q2*y**2)
    eq = Eq(yf.diff(), q0 + q1*yf + q2*yf**2)
    sol = Eq(yf, y)
    assert checkodesol(eq, sol) == (True, 0)
    return eq, q0, q1, q2


# Testing functions start

def test_riccati_transformation():
    """
    This function tests the transformation of the
    solution of a Riccati ODE to the solution of
    its corresponding normal Riccati ODE.

    Each test case 4 values -

    1. w - The solution to be transformed
    2. b1 - The coefficient of f(x) in the ODE.
    3. b2 - The coefficient of f(x)**2 in the ODE.
    4. y - The solution to the normal Riccati ODE.
    """
    tests = [
    (
        x/(x - 1),
        (x**2 + 7)/3*x,
        x,
        -x**2/(x - 1) - x*(x**2/3 + S(7)/3)/2 - 1/(2*x)
    ),
    (
        (2*x + 3)/(2*x + 2),
        (3 - 3*x)/(x + 1),
        5*x,
        -5*x*(2*x + 3)/(2*x + 2) - (3 - 3*x)/(Mul(2, x + 1, evaluate=False)) - 1/(2*x)
    ),
    (
        -1/(2*x**2 - 1),
        0,
        (2 - x)/(4*x - 2),
        (2 - x)/((4*x - 2)*(2*x**2 - 1)) - (4*x - 2)*(Mul(-4, 2 - x, evaluate=False)/(4*x - \
        2)**2 - 1/(4*x - 2))/(Mul(2, 2 - x, evaluate=False))
    ),
    (
        x,
        (8*x - 12)/(12*x + 9),
        x**3/(6*x - 9),
        -x**4/(6*x - 9) - (8*x - 12)/(Mul(2, 12*x + 9, evaluate=False)) - (6*x - 9)*(-6*x**3/(6*x \
        - 9)**2 + 3*x**2/(6*x - 9))/(2*x**3)
    )]
    for w, b1, b2, y in tests:
        assert y == riccati_normal(w, x, b1, b2)
        assert w == riccati_inverse_normal(y, x, b1, b2).cancel()

    # Test bp parameter in riccati_inverse_normal
    tests = [
    (
        (-2*x - 1)/(2*x**2 + 2*x - 2),
        -2/x,
        (-x - 1)/(4*x),
        8*x**2*(1/(4*x) + (-x - 1)/(4*x**2))/(-x - 1)**2 + 4/(-x - 1),
        -2*x*(-1/(4*x) - (-x - 1)/(4*x**2))/(-x - 1) - (-2*x - 1)*(-x - 1)/(4*x*(2*x**2 + 2*x \
        - 2)) + 1/x
    ),
    (
        3/(2*x**2),
        -2/x,
        (-x - 1)/(4*x),
        8*x**2*(1/(4*x) + (-x - 1)/(4*x**2))/(-x - 1)**2 + 4/(-x - 1),
        -2*x*(-1/(4*x) - (-x - 1)/(4*x**2))/(-x - 1) + 1/x - Mul(3, -x - 1, evaluate=False)/(8*x**3)
    )]
    for w, b1, b2, bp, y in tests:
        assert y == riccati_normal(w, x, b1, b2)
        assert w == riccati_inverse_normal(y, x, b1, b2, bp).cancel()


def test_riccati_reduced():
    """
    This function tests the transformation of a
    Riccati ODE to its normal Riccati ODE.

    Each test case 2 values -

    1. eq - A Riccati ODE.
    2. normal_eq - The normal Riccati ODE of eq.
    """
    tests = [
    (
        f(x).diff(x) - x**2 - x*f(x) - x*f(x)**2,

        f(x).diff(x) + f(x)**2 + x**3 - x**2/4 - 3/(4*x**2)
    ),
    (
        6*x/(2*x + 9) + f(x).diff(x) - (x + 1)*f(x)**2/x,

        -3*x**2*(1/x + (-x - 1)/x**2)**2/(4*(-x - 1)**2) + Mul(6, \
        -x - 1, evaluate=False)/(2*x + 9) + f(x)**2 + f(x).diff(x) \
        - (-1 + (x + 1)/x)/(x*(-x - 1))
    ),
    (
        f(x)**2 + f(x).diff(x) - (x - 1)*f(x)/(-x - S(1)/2),

        -(2*x - 2)**2/(4*(2*x + 1)**2) + (2*x - 2)/(2*x + 1)**2 + \
        f(x)**2 + f(x).diff(x) - 1/(2*x + 1)
    ),
    (
        f(x).diff(x) - f(x)**2/x,

        f(x)**2 + f(x).diff(x) + 1/(4*x**2)
    ),
    (
        -3*(-x**2 - x + 1)/(x**2 + 6*x + 1) + f(x).diff(x) + f(x)**2/x,

        f(x)**2 + f(x).diff(x) + (3*x**2/(x**2 + 6*x + 1) + 3*x/(x**2 \
        + 6*x + 1) - 3/(x**2 + 6*x + 1))/x + 1/(4*x**2)
    ),
    (
        6*x/(2*x + 9) + f(x).diff(x) - (x + 1)*f(x)/x,

        False
    ),
    (
        f(x)*f(x).diff(x) - 1/x + f(x)/3 + f(x)**2/(x**2 - 2),

        False
    )]
    for eq, normal_eq in tests:
        assert normal_eq == riccati_reduced(eq, f, x)


def test_match_riccati():
    """
    This function tests if an ODE is Riccati or not.

    Each test case has 5 values -

    1. eq - The Riccati ODE.
    2. match - Boolean indicating if eq is a Riccati ODE.
    3. b0 -
    4. b1 - Coefficient of f(x) in eq.
    5. b2 - Coefficient of f(x)**2 in eq.
    """
    tests = [
    # Test Rational Riccati ODEs
    (
        f(x).diff(x) - (405*x**3 - 882*x**2 - 78*x + 92)/(243*x**4 \
        - 945*x**3 + 846*x**2 + 180*x - 72) - 2 - f(x)**2/(3*x + 1) \
        - (S(1)/3 - x)*f(x)/(S(1)/3 - 3*x/2),

        True,

        45*x**3/(27*x**4 - 105*x**3 + 94*x**2 + 20*x - 8) - 98*x**2/ \
        (27*x**4 - 105*x**3 + 94*x**2 + 20*x - 8) - 26*x/(81*x**4 - \
        315*x**3 + 282*x**2 + 60*x - 24) + 2 + 92/(243*x**4 - 945*x**3 \
        + 846*x**2 + 180*x - 72),

        Mul(-1, 2 - 6*x, evaluate=False)/(9*x - 2),

        1/(3*x + 1)
    ),
    (
        f(x).diff(x) + 4*x/27 - (x/3 - 1)*f(x)**2 - (2*x/3 + \
        1)*f(x)/(3*x + 2) - S(10)/27 - (265*x**2 + 423*x + 162) \
        /(324*x**3 + 216*x**2),

        True,

        -4*x/27 + S(10)/27 + 3/(6*x**3 + 4*x**2) + 47/(36*x**2 \
        + 24*x) + 265/(324*x + 216),

        Mul(-1, -2*x - 3, evaluate=False)/(9*x + 6),

        x/3 - 1
    ),
    (
        f(x).diff(x) - (304*x**5 - 745*x**4 + 631*x**3 - 876*x**2 \
        + 198*x - 108)/(36*x**6 - 216*x**5 + 477*x**4 - 567*x**3 + \
        360*x**2 - 108*x) - S(17)/9 - (x - S(3)/2)*f(x)/(x/2 - \
        S(3)/2) - (x/3 - 3)*f(x)**2/(3*x),

        True,

        304*x**4/(36*x**5 - 216*x**4 + 477*x**3 - 567*x**2 + 360*x - \
        108) - 745*x**3/(36*x**5 - 216*x**4 + 477*x**3 - 567*x**2 + \
        360*x - 108) + 631*x**2/(36*x**5 - 216*x**4 + 477*x**3 - 567* \
        x**2 + 360*x - 108) - 292*x/(12*x**5 - 72*x**4 + 159*x**3 - \
        189*x**2 + 120*x - 36) + S(17)/9 - 12/(4*x**6 - 24*x**5 + \
        53*x**4 - 63*x**3 + 40*x**2 - 12*x) + 22/(4*x**5 - 24*x**4 \
        + 53*x**3 - 63*x**2 + 40*x - 12),

        Mul(-1, 3 - 2*x, evaluate=False)/(x - 3),

        Mul(-1, 9 - x, evaluate=False)/(9*x)
    ),
    # Test Non-Rational Riccati ODEs
    (
        f(x).diff(x) - x**(S(3)/2)/(x**(S(1)/2) - 2) + x**2*f(x) + \
        x*f(x)**2/(x**(S(3)/4)),
        False, 0, 0, 0
    ),
    (
        f(x).diff(x) - sin(x**2) + exp(x)*f(x) + log(x)*f(x)**2,
        False, 0, 0, 0
    ),
    (
        f(x).diff(x) - tanh(x + sqrt(x)) + f(x) + x**4*f(x)**2,
        False, 0, 0, 0
    ),
    # Test Non-Riccati ODEs
    (
        (1 - x**2)*f(x).diff(x, 2) - 2*x*f(x).diff(x) + 20*f(x),
        False, 0, 0, 0
    ),
    (
        f(x).diff(x) - x**2 + x**3*f(x) + (x**2/(x + 1))*f(x)**3,
        False, 0, 0, 0
    ),
    (
        f(x).diff(x)*f(x)**2 + (x**2 - 1)/(x**3 + 1)*f(x) + 1/(2*x \
        + 3) + f(x)**2,
        False, 0, 0, 0
    )]
    for eq, res, b0, b1, b2 in tests:
        match, funcs = match_riccati(eq, f, x)
        assert match == res
        if res:
            assert [b0, b1, b2] == funcs


def test_val_at_inf():
    """
    This function tests the valuation of rational
    function at oo.

    Each test case has 3 values -

    1. num - Numerator of rational function.
    2. den - Denominator of rational function.
    3. val_inf - Valuation of rational function at oo
    """
    tests = [
    # degree(denom) > degree(numer)
    (
        Poly(10*x**3 + 8*x**2 - 13*x + 6, x),
        Poly(-13*x**10 - x**9 + 5*x**8 + 7*x**7 + 10*x**6 + 6*x**5 - 7*x**4 + 11*x**3 - 8*x**2 + 5*x + 13, x),
         7
    ),
    (
        Poly(1, x),
        Poly(-9*x**4 + 3*x**3 + 15*x**2 - 6*x - 14, x),
         4
    ),
    # degree(denom) == degree(numer)
    (
        Poly(-6*x**3 - 8*x**2 + 8*x - 6, x),
        Poly(-5*x**3 + 12*x**2 - 6*x - 9, x),
         0
    ),
    # degree(denom) < degree(numer)
    (
        Poly(12*x**8 - 12*x**7 - 11*x**6 + 8*x**5 + 3*x**4 - x**3 + x**2 - 11*x, x),
        Poly(-14*x**2 + x, x),
         -6
    ),
    (
        Poly(5*x**6 + 9*x**5 - 11*x**4 - 9*x**3 + x**2 - 4*x + 4, x),
        Poly(15*x**4 + 3*x**3 - 8*x**2 + 15*x + 12, x),
         -2
    )]
    for num, den, val in tests:
        assert val_at_inf(num, den, x) == val


def test_necessary_conds():
    """
    This function tests the necessary conditions for
    a Riccati ODE to have a rational particular solution.
    """
    # Valuation at Infinity is an odd negative integer
    assert check_necessary_conds(-3, [1, 2, 4]) == False
    # Valuation at Infinity is a positive integer lesser than 2
    assert check_necessary_conds(1, [1, 2, 4]) == False
    # Multiplicity of a pole is an odd integer greater than 1
    assert check_necessary_conds(2, [3, 1, 6]) == False
    # All values are correct
    assert check_necessary_conds(-10, [1, 2, 8, 12]) == True


def test_inverse_transform_poly():
    """
    This function tests the substitution x -> 1/x
    in rational functions represented using Poly.
    """
    fns = [
    (15*x**3 - 8*x**2 - 2*x - 6)/(18*x + 6),

    (180*x**5 + 40*x**4 + 80*x**3 + 30*x**2 - 60*x - 80)/(180*x**3 - 150*x**2 + 75*x + 12),

    (-15*x**5 - 36*x**4 + 75*x**3 - 60*x**2 - 80*x - 60)/(80*x**4 + 60*x**3 + 60*x**2 + 60*x - 80),

    (60*x**7 + 24*x**6 - 15*x**5 - 20*x**4 + 30*x**2 + 100*x - 60)/(240*x**2 - 20*x - 30),

    (30*x**6 - 12*x**5 + 15*x**4 - 15*x**2 + 10*x + 60)/(3*x**10 - 45*x**9 + 15*x**5 + 15*x**4 - 5*x**3 \
    + 15*x**2 + 45*x - 15)
    ]
    for f in fns:
        num, den = [Poly(e, x) for e in f.as_numer_denom()]
        num, den = inverse_transform_poly(num, den, x)
        assert f.subs(x, 1/x).cancel() == num/den


def test_limit_at_inf():
    """
    This function tests the limit at oo of a
    rational function.

    Each test case has 3 values -

    1. num - Numerator of rational function.
    2. den - Denominator of rational function.
    3. limit_at_inf - Limit of rational function at oo
    """
    tests = [
    # deg(denom) > deg(numer)
    (
        Poly(-12*x**2 + 20*x + 32, x),
        Poly(32*x**3 + 72*x**2 + 3*x - 32, x),
        0
    ),
    # deg(denom) < deg(numer)
    (
        Poly(1260*x**4 - 1260*x**3 - 700*x**2 - 1260*x + 1400, x),
        Poly(6300*x**3 - 1575*x**2 + 756*x - 540, x),
        oo
    ),
    # deg(denom) < deg(numer), one of the leading coefficients is negative
    (
        Poly(-735*x**8 - 1400*x**7 + 1680*x**6 - 315*x**5 - 600*x**4 + 840*x**3 - 525*x**2 \
        + 630*x + 3780, x),
        Poly(1008*x**7 - 2940*x**6 - 84*x**5 + 2940*x**4 - 420*x**3 + 1512*x**2 + 105*x + 168, x),
        -oo
    ),
    # deg(denom) == deg(numer)
    (
        Poly(105*x**7 - 960*x**6 + 60*x**5 + 60*x**4 - 80*x**3 + 45*x**2 + 120*x + 15, x),
        Poly(735*x**7 + 525*x**6 + 720*x**5 + 720*x**4 - 8400*x**3 - 2520*x**2 + 2800*x + 280, x),
        S(1)/7
    ),
    (
        Poly(288*x**4 - 450*x**3 + 280*x**2 - 900*x - 90, x),
        Poly(607*x**4 + 840*x**3 - 1050*x**2 + 420*x + 420, x),
        S(288)/607
    )]
    for num, den, lim in tests:
        assert limit_at_inf(num, den, x) == lim


def test_construct_c_case_1():
    """
    This function tests the Case 1 in the step
    to calculate coefficients of c-vectors.

    Each test case has 4 values -

    1. num - Numerator of the rational function a(x).
    2. den - Denominator of the rational function a(x).
    3. pole - Pole of a(x) for which c-vector is being
       calculated.
    4. c - The c-vector for the pole.
    """
    tests = [
    (
        Poly(-3*x**3 + 3*x**2 + 4*x - 5, x, extension=True),
        Poly(4*x**8 + 16*x**7 + 9*x**5 + 12*x**4 + 6*x**3 + 12*x**2, x, extension=True),
        S(0),
        [[S(1)/2 + sqrt(6)*I/6], [S(1)/2 - sqrt(6)*I/6]]
    ),
    (
        Poly(1200*x**3 + 1440*x**2 + 816*x + 560, x, extension=True),
        Poly(128*x**5 - 656*x**4 + 1264*x**3 - 1125*x**2 + 385*x + 49, x, extension=True),
        S(7)/4,
        [[S(1)/2 + sqrt(16367978)/634], [S(1)/2 - sqrt(16367978)/634]]
    ),
    (
        Poly(4*x + 2, x, extension=True),
        Poly(18*x**4 + (2 - 18*sqrt(3))*x**3 + (14 - 11*sqrt(3))*x**2 + (4 - 6*sqrt(3))*x \
            + 8*sqrt(3) + 16, x, domain='QQ<sqrt(3)>'),
        (S(1) + sqrt(3))/2,
        [[S(1)/2 + sqrt(Mul(4, 2*sqrt(3) + 4, evaluate=False)/(19*sqrt(3) + 44) + 1)/2], \
            [S(1)/2 - sqrt(Mul(4, 2*sqrt(3) + 4, evaluate=False)/(19*sqrt(3) + 44) + 1)/2]]
    )]
    for num, den, pole, c in tests:
        assert construct_c_case_1(num, den, x, pole) == c


def test_construct_c_case_2():
    """
    This function tests the Case 2 in the step
    to calculate coefficients of c-vectors.

    Each test case has 5 values -

    1. num - Numerator of the rational function a(x).
    2. den - Denominator of the rational function a(x).
    3. pole - Pole of a(x) for which c-vector is being
       calculated.
    4. mul - The multiplicity of the pole.
    5. c - The c-vector for the pole.
    """
    tests = [
    # Testing poles with multiplicity 2
    (
        Poly(1, x, extension=True),
        Poly((x - 1)**2*(x - 2), x, extension=True),
        1, 2,
        [[-I*(-1 - I)/2], [I*(-1 + I)/2]]
    ),
    (
        Poly(3*x**5 - 12*x**4 - 7*x**3 + 1, x, extension=True),
        Poly((3*x - 1)**2*(x + 2)**2, x, extension=True),
        S(1)/3, 2,
        [[-S(89)/98], [-S(9)/98]]
    ),
    # Testing poles with multiplicity 4
    (
        Poly(x**3 - x**2 + 4*x, x, extension=True),
        Poly((x - 2)**4*(x + 5)**2, x, extension=True),
        2, 4,
        [[7*sqrt(3)*(S(60)/343 - 4*sqrt(3)/7)/12, 2*sqrt(3)/7], \
        [-7*sqrt(3)*(S(60)/343 + 4*sqrt(3)/7)/12, -2*sqrt(3)/7]]
    ),
    (
        Poly(3*x**5 + x**4 + 3, x, extension=True),
        Poly((4*x + 1)**4*(x + 2), x, extension=True),
        -S(1)/4, 4,
        [[128*sqrt(439)*(-sqrt(439)/128 - S(55)/14336)/439, sqrt(439)/256], \
        [-128*sqrt(439)*(sqrt(439)/128 - S(55)/14336)/439, -sqrt(439)/256]]
    ),
    # Testing poles with multiplicity 6
    (
        Poly(x**3 + 2, x, extension=True),
        Poly((3*x - 1)**6*(x**2 + 1), x, extension=True),
        S(1)/3, 6,
        [[27*sqrt(66)*(-sqrt(66)/54 - S(131)/267300)/22, -2*sqrt(66)/1485, sqrt(66)/162], \
        [-27*sqrt(66)*(sqrt(66)/54 - S(131)/267300)/22, 2*sqrt(66)/1485, -sqrt(66)/162]]
    ),
    (
        Poly(x**2 + 12, x, extension=True),
        Poly((x - sqrt(2))**6, x, extension=True),
        sqrt(2), 6,
        [[sqrt(14)*(S(6)/7 - 3*sqrt(14))/28, sqrt(7)/7, sqrt(14)], \
        [-sqrt(14)*(S(6)/7 + 3*sqrt(14))/28, -sqrt(7)/7, -sqrt(14)]]
    )]
    for num, den, pole, mul, c in tests:
        assert construct_c_case_2(num, den, x, pole, mul) == c


def test_construct_c_case_3():
    """
    This function tests the Case 3 in the step
    to calculate coefficients of c-vectors.
    """
    assert construct_c_case_3() == [[1]]


def test_construct_d_case_4():
    """
    This function tests the Case 4 in the step
    to calculate coefficients of the d-vector.

    Each test case has 4 values -

    1. num - Numerator of the rational function a(x).
    2. den - Denominator of the rational function a(x).
    3. mul - Multiplicity of oo as a pole.
    4. d - The d-vector.
    """
    tests = [
    # Tests with multiplicity at oo = 2
    (
        Poly(-x**5 - 2*x**4 + 4*x**3 + 2*x + 5, x, extension=True),
        Poly(9*x**3 - 2*x**2 + 10*x - 2, x, extension=True),
        2,
        [[10*I/27, I/3, -3*I*(S(158)/243 - I/3)/2], \
        [-10*I/27, -I/3, 3*I*(S(158)/243 + I/3)/2]]
    ),
    (
        Poly(-x**6 + 9*x**5 + 5*x**4 + 6*x**3 + 5*x**2 + 6*x + 7, x, extension=True),
        Poly(x**4 + 3*x**3 + 12*x**2 - x + 7, x, extension=True),
        2,
        [[-6*I, I, -I*(17 - I)/2], [6*I, -I, I*(17 + I)/2]]
    ),
    # Tests with multiplicity at oo = 4
    (
        Poly(-2*x**6 - x**5 - x**4 - 2*x**3 - x**2 - 3*x - 3, x, extension=True),
        Poly(3*x**2 + 10*x + 7, x, extension=True),
        4,
        [[269*sqrt(6)*I/288, -17*sqrt(6)*I/36, sqrt(6)*I/3, -sqrt(6)*I*(S(16969)/2592 \
        - 2*sqrt(6)*I/3)/4], [-269*sqrt(6)*I/288, 17*sqrt(6)*I/36, -sqrt(6)*I/3, \
        sqrt(6)*I*(S(16969)/2592 + 2*sqrt(6)*I/3)/4]]
    ),
    (
        Poly(-3*x**5 - 3*x**4 - 3*x**3 - x**2 - 1, x, extension=True),
        Poly(12*x - 2, x, extension=True),
        4,
        [[41*I/192, 7*I/24, I/2, -I*(-S(59)/6912 - I)], \
        [-41*I/192, -7*I/24, -I/2, I*(-S(59)/6912 + I)]]
    ),
    # Tests with multiplicity at oo = 4
    (
        Poly(-x**7 - x**5 - x**4 - x**2 - x, x, extension=True),
        Poly(x + 2, x, extension=True),
        6,
        [[-5*I/2, 2*I, -I, I, -I*(-9 - 3*I)/2], [5*I/2, -2*I, I, -I, I*(-9 + 3*I)/2]]
    ),
    (
        Poly(-x**7 - x**6 - 2*x**5 - 2*x**4 - x**3 - x**2 + 2*x - 2, x, extension=True),
        Poly(2*x - 2, x, extension=True),
        6,
        [[3*sqrt(2)*I/4, 3*sqrt(2)*I/4, sqrt(2)*I/2, sqrt(2)*I/2, -sqrt(2)*I*(-S(7)/8 - \
        3*sqrt(2)*I/2)/2], [-3*sqrt(2)*I/4, -3*sqrt(2)*I/4, -sqrt(2)*I/2, -sqrt(2)*I/2, \
        sqrt(2)*I*(-S(7)/8 + 3*sqrt(2)*I/2)/2]]
    )]
    for num, den, mul, d in tests:
        ser = rational_laurent_series(num, den, x, oo, mul, 1)
        assert construct_d_case_4(ser, mul//2) == d


def test_construct_d_case_5():
    """
    This function tests the Case 5 in the step
    to calculate coefficients of the d-vector.

    Each test case has 3 values -

    1. num - Numerator of the rational function a(x).
    2. den - Denominator of the rational function a(x).
    3. d - The d-vector.
    """
    tests = [
    (
        Poly(2*x**3 + x**2 + x - 2, x, extension=True),
        Poly(9*x**3 + 5*x**2 + 2*x - 1, x, extension=True),
        [[sqrt(2)/3, -sqrt(2)/108], [-sqrt(2)/3, sqrt(2)/108]]
    ),
    (
        Poly(3*x**5 + x**4 - x**3 + x**2 - 2*x - 2, x, domain='ZZ'),
        Poly(9*x**5 + 7*x**4 + 3*x**3 + 2*x**2 + 5*x + 7, x, domain='ZZ'),
        [[sqrt(3)/3, -2*sqrt(3)/27], [-sqrt(3)/3, 2*sqrt(3)/27]]
    ),
    (
        Poly(x**2 - x + 1, x, domain='ZZ'),
        Poly(3*x**2 + 7*x + 3, x, domain='ZZ'),
        [[sqrt(3)/3, -5*sqrt(3)/9], [-sqrt(3)/3, 5*sqrt(3)/9]]
    )]
    for num, den, d in tests:
        # Multiplicity of oo is 0
        ser = rational_laurent_series(num, den, x, oo, 0, 1)
        assert construct_d_case_5(ser) == d


def test_construct_d_case_6():
    """
    This function tests the Case 6 in the step
    to calculate coefficients of the d-vector.

    Each test case has 3 values -

    1. num - Numerator of the rational function a(x).
    2. den - Denominator of the rational function a(x).
    3. d - The d-vector.
    """
    tests = [
    (
        Poly(-2*x**2 - 5, x, domain='ZZ'),
        Poly(4*x**4 + 2*x**2 + 10*x + 2, x, domain='ZZ'),
        [[S(1)/2 + I/2], [S(1)/2 - I/2]]
    ),
    (
        Poly(-2*x**3 - 4*x**2 - 2*x - 5, x, domain='ZZ'),
        Poly(x**6 - x**5 + 2*x**4 - 4*x**3 - 5*x**2 - 5*x + 9, x, domain='ZZ'),
        [[1], [0]]
    ),
    (
        Poly(-5*x**3 + x**2 + 11*x + 12, x, domain='ZZ'),
        Poly(6*x**8 - 26*x**7 - 27*x**6 - 10*x**5 - 44*x**4 - 46*x**3 - 34*x**2 \
        - 27*x - 42, x, domain='ZZ'),
        [[1], [0]]
    )]
    for num, den, d in tests:
        assert construct_d_case_6(num, den, x) == d


def test_rational_laurent_series():
    """
    This function tests the computation of coefficients
    of Laurent series of a rational function.

    Each test case has 5 values -

    1. num - Numerator of the rational function.
    2. den - Denominator of the rational function.
    3. x0 - Point about which Laurent series is to
       be calculated.
    4. mul - Multiplicity of x0 if x0 is a pole of
       the rational function (0 otherwise).
    5. n - Number of terms upto which the series
       is to be calculated.
    """
    tests = [
    # Laurent series about simple pole (Multiplicity = 1)
    (
        Poly(x**2 - 3*x + 9, x, extension=True),
        Poly(x**2 - x, x, extension=True),
        S(1), 1, 6,
        {1: 7, 0: -8, -1: 9, -2: -9, -3: 9, -4: -9}
    ),
    # Laurent series about multiple pole (Multiplicity > 1)
    (
        Poly(64*x**3 - 1728*x + 1216, x, extension=True),
        Poly(64*x**4 - 80*x**3 - 831*x**2 + 1809*x - 972, x, extension=True),
        S(9)/8, 2, 3,
        {0: S(32177152)/46521675, 2: S(1019)/984, -1: S(11947565056)/28610830125, \
        1: S(209149)/75645}
    ),
    (
        Poly(1, x, extension=True),
        Poly(x**5 + (-4*sqrt(2) - 1)*x**4 + (4*sqrt(2) + 12)*x**3 + (-12 - 8*sqrt(2))*x**2 \
        + (4 + 8*sqrt(2))*x - 4, x, extension=True),
        sqrt(2), 4, 6,
        {4: 1 + sqrt(2), 3: -3 - 2*sqrt(2), 2: Mul(-1, -3 - 2*sqrt(2), evaluate=False)/(-1 \
        + sqrt(2)), 1: (-3 - 2*sqrt(2))/(-1 + sqrt(2))**2, 0: Mul(-1, -3 - 2*sqrt(2), evaluate=False \
        )/(-1 + sqrt(2))**3, -1: (-3 - 2*sqrt(2))/(-1 + sqrt(2))**4}
    ),
    # Laurent series about oo
    (
        Poly(x**5 - 4*x**3 + 6*x**2 + 10*x - 13, x, extension=True),
        Poly(x**2 - 5, x, extension=True),
        oo, 3, 6,
        {3: 1, 2: 0, 1: 1, 0: 6, -1: 15, -2: 17}
    ),
    # Laurent series at x0 where x0 is not a pole of the function
    # Using multiplicity as 0 (as x0 will not be a pole)
    (
        Poly(3*x**3 + 6*x**2 - 2*x + 5, x, extension=True),
        Poly(9*x**4 - x**3 - 3*x**2 + 4*x + 4, x, extension=True),
        S(2)/5, 0, 1,
        {0: S(3345)/3304, -1: S(399325)/2729104, -2: S(3926413375)/4508479808, \
        -3: S(-5000852751875)/1862002160704, -4: S(-6683640101653125)/6152055138966016}
    ),
    (
        Poly(-7*x**2 + 2*x - 4, x, extension=True),
        Poly(7*x**5 + 9*x**4 + 8*x**3 + 3*x**2 + 6*x + 9, x, extension=True),
        oo, 0, 6,
        {0: 0, -2: 0, -5: -S(71)/49, -1: 0, -3: -1, -4: S(11)/7}
    )]
    for num, den, x0, mul, n, ser in tests:
        assert ser == rational_laurent_series(num, den, x, x0, mul, n)


def check_dummy_sol(eq, solse, dummy_sym):
    """
    Helper function to check if actual solution
    matches expected solution if actual solution
    contains dummy symbols.
    """
    if isinstance(eq, Eq):
        eq = eq.lhs - eq.rhs
    _, funcs = match_riccati(eq, f, x)

    sols = solve_riccati(f(x), x, *funcs)
    C1 = Dummy('C1')
    sols = [sol.subs(C1, dummy_sym) for sol in sols]

    assert all(x[0] for x in checkodesol(eq, sols))
    assert all(s1.dummy_eq(s2, dummy_sym) for s1, s2 in zip(sols, solse))


def test_solve_riccati():
    """
    This function tests the computation of rational
    particular solutions for a Riccati ODE.

    Each test case has 2 values -

    1. eq - Riccati ODE to be solved.
    2. sol - Expected solution to the equation.

    Some examples have been taken from the paper - "Statistical Investigation of
    First-Order Algebraic ODEs and their Rational General Solutions" by
    Georg Grasegger, N. Thieu Vo, Franz Winkler

    https://www3.risc.jku.at/publications/download/risc_5197/RISCReport15-19.pdf
    """
    C0 = Dummy('C0')
    # Type: 1st Order Rational Riccati, dy/dx = a + b*y + c*y**2,
    # a, b, c are rational functions of x

    tests = [
    # a(x) is a constant
    (
        Eq(f(x).diff(x) + f(x)**2 - 2, 0),
        [Eq(f(x), sqrt(2)), Eq(f(x), -sqrt(2))]
    ),
    # a(x) is a constant
    (
        f(x)**2 + f(x).diff(x) + 4*f(x)/x + 2/x**2,
        [Eq(f(x), (-2*C0 - x)/(C0*x + x**2))]
    ),
    # a(x) is a constant
    (
        2*x**2*f(x).diff(x) - x*(4*f(x) + f(x).diff(x) - 4) + (f(x) - 1)*f(x),
        [Eq(f(x), (C0 + 2*x**2)/(C0 + x))]
    ),
    # Pole with multiplicity 1
    (
        Eq(f(x).diff(x), -f(x)**2 - 2/(x**3 - x**2)),
        [Eq(f(x), 1/(x**2 - x))]
    ),
    # One pole of multiplicity 2
    (
        x**2 - (2*x + 1/x)*f(x) + f(x)**2 + f(x).diff(x),
        [Eq(f(x), (C0*x + x**3 + 2*x)/(C0 + x**2)), Eq(f(x), x)]
    ),
    (
        x**4*f(x).diff(x) + x**2 - x*(2*f(x)**2 + f(x).diff(x)) + f(x),
        [Eq(f(x), (C0*x**2 + x)/(C0 + x**2)), Eq(f(x), x**2)]
    ),
    # Multiple poles of multiplicity 2
    (
        -f(x)**2 + f(x).diff(x) + (15*x**2 - 20*x + 7)/((x - 1)**2*(2*x \
            - 1)**2),
        [Eq(f(x), (9*C0*x - 6*C0 - 15*x**5 + 60*x**4 - 94*x**3 + 72*x**2 \
        - 30*x + 6)/(6*C0*x**2 - 9*C0*x + 3*C0 + 6*x**6 - 29*x**5 + \
        57*x**4 - 58*x**3 + 30*x**2 - 6*x)), Eq(f(x), (3*x - 2)/(2*x**2 \
        - 3*x + 1))]
    ),
    # Regression: Poles with even multiplicity > 2 fixed
    (
        f(x)**2 + f(x).diff(x) - (4*x**6 - 8*x**5 + 12*x**4 + 4*x**3 + \
            7*x**2 - 20*x + 4)/(4*x**4),
        [Eq(f(x), (2*x**5 - 2*x**4 - x**3 + 4*x**2 + 3*x - 2)/(2*x**4 \
            - 2*x**2))]
    ),
    # Regression: Poles with even multiplicity > 2 fixed
    (
        Eq(f(x).diff(x), (-x**6 + 15*x**4 - 40*x**3 + 45*x**2 - 24*x + 4)/\
            (x**12 - 12*x**11 + 66*x**10 - 220*x**9 + 495*x**8 - 792*x**7 + 924*x**6 - \
            792*x**5 + 495*x**4 - 220*x**3 + 66*x**2 - 12*x + 1) + f(x)**2 + f(x)),
        [Eq(f(x), 1/(x**6 - 6*x**5 + 15*x**4 - 20*x**3 + 15*x**2 - 6*x + 1))]
    ),
    # More than 2 poles with multiplicity 2
    # Regression: Fixed mistake in necessary conditions
    (
        Eq(f(x).diff(x), x*f(x) + 2*x + (3*x - 2)*f(x)**2/(4*x + 2) + \
            (8*x**2 - 7*x + 26)/(16*x**3 - 24*x**2 + 8) - S(3)/2),
        [Eq(f(x), (1 - 4*x)/(2*x - 2))]
    ),
    # Regression: Fixed mistake in necessary conditions
    (
        Eq(f(x).diff(x), (-12*x**2 - 48*x - 15)/(24*x**3 - 40*x**2 + 8*x + 8) \
            + 3*f(x)**2/(6*x + 2)),
        [Eq(f(x), (2*x + 1)/(2*x - 2))]
    ),
    # Imaginary poles
    (
        f(x).diff(x) + (3*x**2 + 1)*f(x)**2/x + (6*x**2 - x + 3)*f(x)/(x*(x \
            - 1)) + (3*x**2 - 2*x + 2)/(x*(x - 1)**2),
        [Eq(f(x), (-C0 - x**3 + x**2 - 2*x)/(C0*x - C0 + x**4 - x**3 + x**2 \
            - x)), Eq(f(x), -1/(x - 1))],
    ),
    # Imaginary coefficients in equation
    (
        f(x).diff(x) - 2*I*(f(x)**2 + 1)/x,
        [Eq(f(x), (-I*C0 + I*x**4)/(C0 + x**4)), Eq(f(x), -I)]
    ),
    # Regression: linsolve returning empty solution
    # Large value of m (> 10)
    (
        Eq(f(x).diff(x), x*f(x)/(S(3)/2 - 2*x) + (x/2 - S(1)/3)*f(x)**2/\
            (2*x/3 - S(1)/2) - S(5)/4 + (281*x**2 - 1260*x + 756)/(16*x**3 - 12*x**2)),
        [Eq(f(x), (9 - x)/x), Eq(f(x), (40*x**14 + 28*x**13 + 420*x**12 + 2940*x**11 + \
            18480*x**10 + 103950*x**9 + 519750*x**8 + 2286900*x**7 + 8731800*x**6 + 28378350*\
            x**5 + 76403250*x**4 + 163721250*x**3 + 261954000*x**2 + 278326125*x + 147349125)/\
            ((24*x**14 + 140*x**13 + 840*x**12 + 4620*x**11 + 23100*x**10 + 103950*x**9 + \
            415800*x**8 + 1455300*x**7 + 4365900*x**6 + 10914750*x**5 + 21829500*x**4 + 32744250\
            *x**3 + 32744250*x**2 + 16372125*x)))]
    ),
    # Regression: Fixed bug due to a typo in paper
    (
        Eq(f(x).diff(x), 18*x**3 + 18*x**2 + (-x/2 - S(1)/2)*f(x)**2 + 6),
        [Eq(f(x), 6*x)]
    ),
    # Regression: Fixed bug due to a typo in paper
    (
        Eq(f(x).diff(x), -3*x**3/4 + 15*x/2 + (x/3 - S(4)/3)*f(x)**2 \
            + 9 + (1 - x)*f(x)/x + 3/x),
        [Eq(f(x), -3*x/2 - 3)]
    )]
    for eq, sol in tests:
        check_dummy_sol(eq, sol, C0)


@slow
def test_solve_riccati_slow():
    """
    This function tests the computation of rational
    particular solutions for a Riccati ODE.

    Each test case has 2 values -

    1. eq - Riccati ODE to be solved.
    2. sol - Expected solution to the equation.
    """
    C0 = Dummy('C0')
    tests = [
    # Very large values of m (989 and 991)
    (
        Eq(f(x).diff(x), (1 - x)*f(x)/(x - 3) + (2 - 12*x)*f(x)**2/(2*x - 9) + \
            (54924*x**3 - 405264*x**2 + 1084347*x - 1087533)/(8*x**4 - 132*x**3 + 810*x**2 - \
            2187*x + 2187) + 495),
        [Eq(f(x), (18*x + 6)/(2*x - 9))]
    )]
    for eq, sol in tests:
        check_dummy_sol(eq, sol, C0)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_galoistools.py ===
from sympy.polys.galoistools import (
    gf_crt, gf_crt1, gf_crt2, gf_int,
    gf_degree, gf_strip, gf_trunc, gf_normal,
    gf_from_dict, gf_to_dict,
    gf_from_int_poly, gf_to_int_poly,
    gf_neg, gf_add_ground, gf_sub_ground, gf_mul_ground,
    gf_add, gf_sub, gf_add_mul, gf_sub_mul, gf_mul, gf_sqr,
    gf_div, gf_rem, gf_quo, gf_exquo,
    gf_lshift, gf_rshift, gf_expand,
    gf_pow, gf_pow_mod,
    gf_gcdex, gf_gcd, gf_lcm, gf_cofactors,
    gf_LC, gf_TC, gf_monic,
    gf_eval, gf_multi_eval,
    gf_compose, gf_compose_mod,
    gf_trace_map,
    gf_diff,
    gf_irreducible, gf_irreducible_p,
    gf_irred_p_ben_or, gf_irred_p_rabin,
    gf_sqf_list, gf_sqf_part, gf_sqf_p,
    gf_Qmatrix, gf_Qbasis,
    gf_ddf_zassenhaus, gf_ddf_shoup,
    gf_edf_zassenhaus, gf_edf_shoup,
    gf_berlekamp,
    gf_factor_sqf, gf_factor,
    gf_value, linear_congruence, _csolve_prime_las_vegas,
    csolve_prime, gf_csolve, gf_frobenius_map, gf_frobenius_monomial_base
)

from sympy.polys.polyerrors import (
    ExactQuotientFailed,
)

from sympy.polys import polyconfig as config

from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises


def test_gf_crt():
    U = [49, 76, 65]
    M = [99, 97, 95]

    p = 912285
    u = 639985

    assert gf_crt(U, M, ZZ) == u

    E = [9215, 9405, 9603]
    S = [62, 24, 12]

    assert gf_crt1(M, ZZ) == (p, E, S)
    assert gf_crt2(U, M, p, E, S, ZZ) == u


def test_gf_int():
    assert gf_int(0, 5) == 0
    assert gf_int(1, 5) == 1
    assert gf_int(2, 5) == 2
    assert gf_int(3, 5) == -2
    assert gf_int(4, 5) == -1
    assert gf_int(5, 5) == 0


def test_gf_degree():
    assert gf_degree([]) == -1
    assert gf_degree([1]) == 0
    assert gf_degree([1, 0]) == 1
    assert gf_degree([1, 0, 0, 0, 1]) == 4


def test_gf_strip():
    assert gf_strip([]) == []
    assert gf_strip([0]) == []
    assert gf_strip([0, 0, 0]) == []

    assert gf_strip([1]) == [1]
    assert gf_strip([0, 1]) == [1]
    assert gf_strip([0, 0, 0, 1]) == [1]

    assert gf_strip([1, 2, 0]) == [1, 2, 0]
    assert gf_strip([0, 1, 2, 0]) == [1, 2, 0]
    assert gf_strip([0, 0, 0, 1, 2, 0]) == [1, 2, 0]


def test_gf_trunc():
    assert gf_trunc([], 11) == []
    assert gf_trunc([1], 11) == [1]
    assert gf_trunc([22], 11) == []
    assert gf_trunc([12], 11) == [1]

    assert gf_trunc([11, 22, 17, 1, 0], 11) == [6, 1, 0]
    assert gf_trunc([12, 23, 17, 1, 0], 11) == [1, 1, 6, 1, 0]


def test_gf_normal():
    assert gf_normal([11, 22, 17, 1, 0], 11, ZZ) == [6, 1, 0]


def test_gf_from_to_dict():
    f = {11: 12, 6: 2, 0: 25}
    F = {11: 1, 6: 2, 0: 3}
    g = [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3]

    assert gf_from_dict(f, 11, ZZ) == g
    assert gf_to_dict(g, 11) == F

    f = {11: -5, 4: 0, 3: 1, 0: 12}
    F = {11: -5, 3: 1, 0: 1}
    g = [6, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]

    assert gf_from_dict(f, 11, ZZ) == g
    assert gf_to_dict(g, 11) == F

    assert gf_to_dict([10], 11, symmetric=True) == {0: -1}
    assert gf_to_dict([10], 11, symmetric=False) == {0: 10}


def test_gf_from_to_int_poly():
    assert gf_from_int_poly([1, 0, 7, 2, 20], 5) == [1, 0, 2, 2, 0]
    assert gf_to_int_poly([1, 0, 4, 2, 3], 5) == [1, 0, -1, 2, -2]

    assert gf_to_int_poly([10], 11, symmetric=True) == [-1]
    assert gf_to_int_poly([10], 11, symmetric=False) == [10]


def test_gf_LC():
    assert gf_LC([], ZZ) == 0
    assert gf_LC([1], ZZ) == 1
    assert gf_LC([1, 2], ZZ) == 1


def test_gf_TC():
    assert gf_TC([], ZZ) == 0
    assert gf_TC([1], ZZ) == 1
    assert gf_TC([1, 2], ZZ) == 2


def test_gf_monic():
    assert gf_monic(ZZ.map([]), 11, ZZ) == (0, [])

    assert gf_monic(ZZ.map([1]), 11, ZZ) == (1, [1])
    assert gf_monic(ZZ.map([2]), 11, ZZ) == (2, [1])

    assert gf_monic(ZZ.map([1, 2, 3, 4]), 11, ZZ) == (1, [1, 2, 3, 4])
    assert gf_monic(ZZ.map([2, 3, 4, 5]), 11, ZZ) == (2, [1, 7, 2, 8])


def test_gf_arith():
    assert gf_neg([], 11, ZZ) == []
    assert gf_neg([1], 11, ZZ) == [10]
    assert gf_neg([1, 2, 3], 11, ZZ) == [10, 9, 8]

    assert gf_add_ground([], 0, 11, ZZ) == []
    assert gf_sub_ground([], 0, 11, ZZ) == []

    assert gf_add_ground([], 3, 11, ZZ) == [3]
    assert gf_sub_ground([], 3, 11, ZZ) == [8]

    assert gf_add_ground([1], 3, 11, ZZ) == [4]
    assert gf_sub_ground([1], 3, 11, ZZ) == [9]

    assert gf_add_ground([8], 3, 11, ZZ) == []
    assert gf_sub_ground([3], 3, 11, ZZ) == []

    assert gf_add_ground([1, 2, 3], 3, 11, ZZ) == [1, 2, 6]
    assert gf_sub_ground([1, 2, 3], 3, 11, ZZ) == [1, 2, 0]

    assert gf_mul_ground([], 0, 11, ZZ) == []
    assert gf_mul_ground([], 1, 11, ZZ) == []

    assert gf_mul_ground([1], 0, 11, ZZ) == []
    assert gf_mul_ground([1], 1, 11, ZZ) == [1]

    assert gf_mul_ground([1, 2, 3], 0, 11, ZZ) == []
    assert gf_mul_ground([1, 2, 3], 1, 11, ZZ) == [1, 2, 3]
    assert gf_mul_ground([1, 2, 3], 7, 11, ZZ) == [7, 3, 10]

    assert gf_add([], [], 11, ZZ) == []
    assert gf_add([1], [], 11, ZZ) == [1]
    assert gf_add([], [1], 11, ZZ) == [1]
    assert gf_add([1], [1], 11, ZZ) == [2]
    assert gf_add([1], [2], 11, ZZ) == [3]

    assert gf_add([1, 2], [1], 11, ZZ) == [1, 3]
    assert gf_add([1], [1, 2], 11, ZZ) == [1, 3]

    assert gf_add([1, 2, 3], [8, 9, 10], 11, ZZ) == [9, 0, 2]

    assert gf_sub([], [], 11, ZZ) == []
    assert gf_sub([1], [], 11, ZZ) == [1]
    assert gf_sub([], [1], 11, ZZ) == [10]
    assert gf_sub([1], [1], 11, ZZ) == []
    assert gf_sub([1], [2], 11, ZZ) == [10]

    assert gf_sub([1, 2], [1], 11, ZZ) == [1, 1]
    assert gf_sub([1], [1, 2], 11, ZZ) == [10, 10]

    assert gf_sub([3, 2, 1], [8, 9, 10], 11, ZZ) == [6, 4, 2]

    assert gf_add_mul(
        [1, 5, 6], [7, 3], [8, 0, 6, 1], 11, ZZ) == [1, 2, 10, 8, 9]
    assert gf_sub_mul(
        [1, 5, 6], [7, 3], [8, 0, 6, 1], 11, ZZ) == [10, 9, 3, 2, 3]

    assert gf_mul([], [], 11, ZZ) == []
    assert gf_mul([], [1], 11, ZZ) == []
    assert gf_mul([1], [], 11, ZZ) == []
    assert gf_mul([1], [1], 11, ZZ) == [1]
    assert gf_mul([5], [7], 11, ZZ) == [2]

    assert gf_mul([3, 0, 0, 6, 1, 2], [4, 0, 1, 0], 11, ZZ) == [1, 0,
                  3, 2, 4, 3, 1, 2, 0]
    assert gf_mul([4, 0, 1, 0], [3, 0, 0, 6, 1, 2], 11, ZZ) == [1, 0,
                  3, 2, 4, 3, 1, 2, 0]

    assert gf_mul([2, 0, 0, 1, 7], [2, 0, 0, 1, 7], 11, ZZ) == [4, 0,
                  0, 4, 6, 0, 1, 3, 5]

    assert gf_sqr([], 11, ZZ) == []
    assert gf_sqr([2], 11, ZZ) == [4]
    assert gf_sqr([1, 2], 11, ZZ) == [1, 4, 4]

    assert gf_sqr([2, 0, 0, 1, 7], 11, ZZ) == [4, 0, 0, 4, 6, 0, 1, 3, 5]


def test_gf_division():
    raises(ZeroDivisionError, lambda: gf_div([1, 2, 3], [], 11, ZZ))
    raises(ZeroDivisionError, lambda: gf_rem([1, 2, 3], [], 11, ZZ))
    raises(ZeroDivisionError, lambda: gf_quo([1, 2, 3], [], 11, ZZ))
    raises(ZeroDivisionError, lambda: gf_quo([1, 2, 3], [], 11, ZZ))

    assert gf_div([1], [1, 2, 3], 7, ZZ) == ([], [1])
    assert gf_rem([1], [1, 2, 3], 7, ZZ) == [1]
    assert gf_quo([1], [1, 2, 3], 7, ZZ) == []

    f = ZZ.map([5, 4, 3, 2, 1, 0])
    g = ZZ.map([1, 2, 3])
    q = [5, 1, 0, 6]
    r = [3, 3]

    assert gf_div(f, g, 7, ZZ) == (q, r)
    assert gf_rem(f, g, 7, ZZ) == r
    assert gf_quo(f, g, 7, ZZ) == q

    raises(ExactQuotientFailed, lambda: gf_exquo(f, g, 7, ZZ))

    f = ZZ.map([5, 4, 3, 2, 1, 0])
    g = ZZ.map([1, 2, 3, 0])
    q = [5, 1, 0]
    r = [6, 1, 0]

    assert gf_div(f, g, 7, ZZ) == (q, r)
    assert gf_rem(f, g, 7, ZZ) == r
    assert gf_quo(f, g, 7, ZZ) == q

    raises(ExactQuotientFailed, lambda: gf_exquo(f, g, 7, ZZ))

    assert gf_quo(ZZ.map([1, 2, 1]), ZZ.map([1, 1]), 11, ZZ) == [1, 1]


def test_gf_shift():
    f = [1, 2, 3, 4, 5]

    assert gf_lshift([], 5, ZZ) == []
    assert gf_rshift([], 5, ZZ) == ([], [])

    assert gf_lshift(f, 1, ZZ) == [1, 2, 3, 4, 5, 0]
    assert gf_lshift(f, 2, ZZ) == [1, 2, 3, 4, 5, 0, 0]

    assert gf_rshift(f, 0, ZZ) == (f, [])
    assert gf_rshift(f, 1, ZZ) == ([1, 2, 3, 4], [5])
    assert gf_rshift(f, 3, ZZ) == ([1, 2], [3, 4, 5])
    assert gf_rshift(f, 5, ZZ) == ([], f)


def test_gf_expand():
    F = [([1, 1], 2), ([1, 2], 3)]

    assert gf_expand(F, 11, ZZ) == [1, 8, 3, 5, 6, 8]
    assert gf_expand((4, F), 11, ZZ) == [4, 10, 1, 9, 2, 10]


def test_gf_powering():
    assert gf_pow([1, 0, 0, 1, 8], 0, 11, ZZ) == [1]
    assert gf_pow([1, 0, 0, 1, 8], 1, 11, ZZ) == [1, 0, 0, 1, 8]
    assert gf_pow([1, 0, 0, 1, 8], 2, 11, ZZ) == [1, 0, 0, 2, 5, 0, 1, 5, 9]

    assert gf_pow([1, 0, 0, 1, 8], 5, 11, ZZ) == \
        [1, 0, 0, 5, 7, 0, 10, 6, 2, 10, 9, 6, 10, 6, 6, 0, 5, 2, 5, 9, 10]

    assert gf_pow([1, 0, 0, 1, 8], 8, 11, ZZ) == \
        [1, 0, 0, 8, 9, 0, 6, 8, 10, 1, 2, 5, 10, 7, 7, 9, 1, 2, 0, 0, 6, 2,
         5, 2, 5, 7, 7, 9, 10, 10, 7, 5, 5]

    assert gf_pow([1, 0, 0, 1, 8], 45, 11, ZZ) == \
        [ 1, 0, 0,  1,  8, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0,
          0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0,  4, 0, 0,  4, 10, 0, 0, 0, 0, 0, 0,
         10, 0, 0, 10,  3, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0,
          6, 0, 0,  6,  4, 0, 0, 0, 0, 0, 0,  8, 0, 0,  8,  9, 0, 0, 0, 0, 0, 0,
         10, 0, 0, 10,  3, 0, 0, 0, 0, 0, 0,  4, 0, 0,  4, 10, 0, 0, 0, 0, 0, 0,
          8, 0, 0,  8,  9, 0, 0, 0, 0, 0, 0,  9, 0, 0,  9,  6, 0, 0, 0, 0, 0, 0,
          3, 0, 0,  3,  2, 0, 0, 0, 0, 0, 0, 10, 0, 0, 10,  3, 0, 0, 0, 0, 0, 0,
         10, 0, 0, 10,  3, 0, 0, 0, 0, 0, 0,  2, 0, 0,  2,  5, 0, 0, 0, 0, 0, 0,
          4, 0, 0,  4, 10]

    assert gf_pow_mod(ZZ.map([1, 0, 0, 1, 8]), 0, ZZ.map([2, 0, 7]), 11, ZZ) == [1]
    assert gf_pow_mod(ZZ.map([1, 0, 0, 1, 8]), 1, ZZ.map([2, 0, 7]), 11, ZZ) == [1, 1]
    assert gf_pow_mod(ZZ.map([1, 0, 0, 1, 8]), 2, ZZ.map([2, 0, 7]), 11, ZZ) == [2, 3]
    assert gf_pow_mod(ZZ.map([1, 0, 0, 1, 8]), 5, ZZ.map([2, 0, 7]), 11, ZZ) == [7, 8]
    assert gf_pow_mod(ZZ.map([1, 0, 0, 1, 8]), 8, ZZ.map([2, 0, 7]), 11, ZZ) == [1, 5]
    assert gf_pow_mod(ZZ.map([1, 0, 0, 1, 8]), 45, ZZ.map([2, 0, 7]), 11, ZZ) == [5, 4]


def test_gf_gcdex():
    assert gf_gcdex(ZZ.map([]), ZZ.map([]), 11, ZZ) == ([1], [], [])
    assert gf_gcdex(ZZ.map([2]), ZZ.map([]), 11, ZZ) == ([6], [], [1])
    assert gf_gcdex(ZZ.map([]), ZZ.map([2]), 11, ZZ) == ([], [6], [1])
    assert gf_gcdex(ZZ.map([2]), ZZ.map([2]), 11, ZZ) == ([], [6], [1])

    assert gf_gcdex(ZZ.map([]), ZZ.map([3, 0]), 11, ZZ) == ([], [4], [1, 0])
    assert gf_gcdex(ZZ.map([3, 0]), ZZ.map([]), 11, ZZ) == ([4], [], [1, 0])

    assert gf_gcdex(ZZ.map([3, 0]), ZZ.map([3, 0]), 11, ZZ) == ([], [4], [1, 0])

    assert gf_gcdex(ZZ.map([1, 8, 7]), ZZ.map([1, 7, 1, 7]), 11, ZZ) == ([5, 6], [6], [1, 7])


def test_gf_gcd():
    assert gf_gcd(ZZ.map([]), ZZ.map([]), 11, ZZ) == []
    assert gf_gcd(ZZ.map([2]), ZZ.map([]), 11, ZZ) == [1]
    assert gf_gcd(ZZ.map([]), ZZ.map([2]), 11, ZZ) == [1]
    assert gf_gcd(ZZ.map([2]), ZZ.map([2]), 11, ZZ) == [1]

    assert gf_gcd(ZZ.map([]), ZZ.map([1, 0]), 11, ZZ) == [1, 0]
    assert gf_gcd(ZZ.map([1, 0]), ZZ.map([]), 11, ZZ) == [1, 0]

    assert gf_gcd(ZZ.map([3, 0]), ZZ.map([3, 0]), 11, ZZ) == [1, 0]
    assert gf_gcd(ZZ.map([1, 8, 7]), ZZ.map([1, 7, 1, 7]), 11, ZZ) == [1, 7]


def test_gf_lcm():
    assert gf_lcm(ZZ.map([]), ZZ.map([]), 11, ZZ) == []
    assert gf_lcm(ZZ.map([2]), ZZ.map([]), 11, ZZ) == []
    assert gf_lcm(ZZ.map([]), ZZ.map([2]), 11, ZZ) == []
    assert gf_lcm(ZZ.map([2]), ZZ.map([2]), 11, ZZ) == [1]

    assert gf_lcm(ZZ.map([]), ZZ.map([1, 0]), 11, ZZ) == []
    assert gf_lcm(ZZ.map([1, 0]), ZZ.map([]), 11, ZZ) == []

    assert gf_lcm(ZZ.map([3, 0]), ZZ.map([3, 0]), 11, ZZ) == [1, 0]
    assert gf_lcm(ZZ.map([1, 8, 7]), ZZ.map([1, 7, 1, 7]), 11, ZZ) == [1, 8, 8, 8, 7]


def test_gf_cofactors():
    assert gf_cofactors(ZZ.map([]), ZZ.map([]), 11, ZZ) == ([], [], [])
    assert gf_cofactors(ZZ.map([2]), ZZ.map([]), 11, ZZ) == ([1], [2], [])
    assert gf_cofactors(ZZ.map([]), ZZ.map([2]), 11, ZZ) == ([1], [], [2])
    assert gf_cofactors(ZZ.map([2]), ZZ.map([2]), 11, ZZ) == ([1], [2], [2])

    assert gf_cofactors(ZZ.map([]), ZZ.map([1, 0]), 11, ZZ) == ([1, 0], [], [1])
    assert gf_cofactors(ZZ.map([1, 0]), ZZ.map([]), 11, ZZ) == ([1, 0], [1], [])

    assert gf_cofactors(ZZ.map([3, 0]), ZZ.map([3, 0]), 11, ZZ) == (
        [1, 0], [3], [3])
    assert gf_cofactors(ZZ.map([1, 8, 7]), ZZ.map([1, 7, 1, 7]), 11, ZZ) == (
        ([1, 7], [1, 1], [1, 0, 1]))


def test_gf_diff():
    assert gf_diff([], 11, ZZ) == []
    assert gf_diff([7], 11, ZZ) == []

    assert gf_diff([7, 3], 11, ZZ) == [7]
    assert gf_diff([7, 3, 1], 11, ZZ) == [3, 3]

    assert gf_diff([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 11, ZZ) == []


def test_gf_eval():
    assert gf_eval([], 4, 11, ZZ) == 0
    assert gf_eval([], 27, 11, ZZ) == 0
    assert gf_eval([7], 4, 11, ZZ) == 7
    assert gf_eval([7], 27, 11, ZZ) == 7

    assert gf_eval([1, 0, 3, 2, 4, 3, 1, 2, 0], 0, 11, ZZ) == 0
    assert gf_eval([1, 0, 3, 2, 4, 3, 1, 2, 0], 4, 11, ZZ) == 9
    assert gf_eval([1, 0, 3, 2, 4, 3, 1, 2, 0], 27, 11, ZZ) == 5

    assert gf_eval([4, 0, 0, 4, 6, 0, 1, 3, 5], 0, 11, ZZ) == 5
    assert gf_eval([4, 0, 0, 4, 6, 0, 1, 3, 5], 4, 11, ZZ) == 3
    assert gf_eval([4, 0, 0, 4, 6, 0, 1, 3, 5], 27, 11, ZZ) == 9

    assert gf_multi_eval([3, 2, 1], [0, 1, 2, 3], 11, ZZ) == [1, 6, 6, 1]


def test_gf_compose():
    assert gf_compose([], [1, 0], 11, ZZ) == []
    assert gf_compose_mod([], [1, 0], [1, 0], 11, ZZ) == []

    assert gf_compose([1], [], 11, ZZ) == [1]
    assert gf_compose([1, 0], [], 11, ZZ) == []
    assert gf_compose([1, 0], [1, 0], 11, ZZ) == [1, 0]

    f = ZZ.map([1, 1, 4, 9, 1])
    g = ZZ.map([1, 1, 1])
    h = ZZ.map([1, 0, 0, 2])

    assert gf_compose(g, h, 11, ZZ) == [1, 0, 0, 5, 0, 0, 7]
    assert gf_compose_mod(g, h, f, 11, ZZ) == [3, 9, 6, 10]


def test_gf_trace_map():
    f = ZZ.map([1, 1, 4, 9, 1])
    a = [1, 1, 1]
    c = ZZ.map([1, 0])
    b = gf_pow_mod(c, 11, f, 11, ZZ)

    assert gf_trace_map(a, b, c, 0, f, 11, ZZ) == \
        ([1, 1, 1], [1, 1, 1])
    assert gf_trace_map(a, b, c, 1, f, 11, ZZ) == \
        ([5, 2, 10, 3], [5, 3, 0, 4])
    assert gf_trace_map(a, b, c, 2, f, 11, ZZ) == \
        ([5, 9, 5, 3], [10, 1, 5, 7])
    assert gf_trace_map(a, b, c, 3, f, 11, ZZ) == \
        ([1, 10, 6, 0], [7])
    assert gf_trace_map(a, b, c, 4, f, 11, ZZ) == \
        ([1, 1, 1], [1, 1, 8])
    assert gf_trace_map(a, b, c, 5, f, 11, ZZ) == \
        ([5, 2, 10, 3], [5, 3, 0, 0])
    assert gf_trace_map(a, b, c, 11, f, 11, ZZ) == \
        ([1, 10, 6, 0], [10])


def test_gf_irreducible():
    assert gf_irreducible_p(gf_irreducible(1, 11, ZZ), 11, ZZ) is True
    assert gf_irreducible_p(gf_irreducible(2, 11, ZZ), 11, ZZ) is True
    assert gf_irreducible_p(gf_irreducible(3, 11, ZZ), 11, ZZ) is True
    assert gf_irreducible_p(gf_irreducible(4, 11, ZZ), 11, ZZ) is True
    assert gf_irreducible_p(gf_irreducible(5, 11, ZZ), 11, ZZ) is True
    assert gf_irreducible_p(gf_irreducible(6, 11, ZZ), 11, ZZ) is True
    assert gf_irreducible_p(gf_irreducible(7, 11, ZZ), 11, ZZ) is True


def test_gf_irreducible_p():
    assert gf_irred_p_ben_or(ZZ.map([7]), 11, ZZ) is True
    assert gf_irred_p_ben_or(ZZ.map([7, 3]), 11, ZZ) is True
    assert gf_irred_p_ben_or(ZZ.map([7, 3, 1]), 11, ZZ) is False

    assert gf_irred_p_rabin(ZZ.map([7]), 11, ZZ) is True
    assert gf_irred_p_rabin(ZZ.map([7, 3]), 11, ZZ) is True
    assert gf_irred_p_rabin(ZZ.map([7, 3, 1]), 11, ZZ) is False

    config.setup('GF_IRRED_METHOD', 'ben-or')

    assert gf_irreducible_p(ZZ.map([7]), 11, ZZ) is True
    assert gf_irreducible_p(ZZ.map([7, 3]), 11, ZZ) is True
    assert gf_irreducible_p(ZZ.map([7, 3, 1]), 11, ZZ) is False

    config.setup('GF_IRRED_METHOD', 'rabin')

    assert gf_irreducible_p(ZZ.map([7]), 11, ZZ) is True
    assert gf_irreducible_p(ZZ.map([7, 3]), 11, ZZ) is True
    assert gf_irreducible_p(ZZ.map([7, 3, 1]), 11, ZZ) is False

    config.setup('GF_IRRED_METHOD', 'other')
    raises(KeyError, lambda: gf_irreducible_p([7], 11, ZZ))
    config.setup('GF_IRRED_METHOD')

    f = ZZ.map([1, 9, 9, 13, 16, 15, 6, 7, 7, 7, 10])
    g = ZZ.map([1, 7, 16, 7, 15, 13, 13, 11, 16, 10, 9])

    h = gf_mul(f, g, 17, ZZ)

    assert gf_irred_p_ben_or(f, 17, ZZ) is True
    assert gf_irred_p_ben_or(g, 17, ZZ) is True

    assert gf_irred_p_ben_or(h, 17, ZZ) is False

    assert gf_irred_p_rabin(f, 17, ZZ) is True
    assert gf_irred_p_rabin(g, 17, ZZ) is True

    assert gf_irred_p_rabin(h, 17, ZZ) is False


def test_gf_squarefree():
    assert gf_sqf_list([], 11, ZZ) == (0, [])
    assert gf_sqf_list([1], 11, ZZ) == (1, [])
    assert gf_sqf_list([1, 1], 11, ZZ) == (1, [([1, 1], 1)])

    assert gf_sqf_p([], 11, ZZ) is True
    assert gf_sqf_p([1], 11, ZZ) is True
    assert gf_sqf_p([1, 1], 11, ZZ) is True

    f = gf_from_dict({11: 1, 0: 1}, 11, ZZ)

    assert gf_sqf_p(f, 11, ZZ) is False

    assert gf_sqf_list(f, 11, ZZ) == \
        (1, [([1, 1], 11)])

    f = [1, 5, 8, 4]

    assert gf_sqf_p(f, 11, ZZ) is False

    assert gf_sqf_list(f, 11, ZZ) == \
        (1, [([1, 1], 1),
             ([1, 2], 2)])

    assert gf_sqf_part(f, 11, ZZ) == [1, 3, 2]

    f = [1, 0, 0, 2, 0, 0, 2, 0, 0, 1, 0]

    assert gf_sqf_list(f, 3, ZZ) == \
        (1, [([1, 0], 1),
             ([1, 1], 3),
             ([1, 2], 6)])

def test_gf_frobenius_map():
    f = ZZ.map([2, 0, 1, 0, 2, 2, 0, 2, 2, 2])
    g = ZZ.map([1,1,0,2,0,1,0,2,0,1])
    p = 3
    b = gf_frobenius_monomial_base(g, p, ZZ)
    h = gf_frobenius_map(f, g, b, p, ZZ)
    h1 = gf_pow_mod(f, p, g, p, ZZ)
    assert h == h1


def test_gf_berlekamp():
    f = gf_from_int_poly([1, -3, 1, -3, -1, -3, 1], 11)

    Q = [[1, 0, 0, 0, 0, 0],
         [3, 5, 8, 8, 6, 5],
         [3, 6, 6, 1, 10, 0],
         [9, 4, 10, 3, 7, 9],
         [7, 8, 10, 0, 0, 8],
         [8, 10, 7, 8, 10, 8]]

    V = [[1, 0, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 0],
         [0, 0, 7, 9, 0, 1]]

    assert gf_Qmatrix(f, 11, ZZ) == Q
    assert gf_Qbasis(Q, 11, ZZ) == V

    assert gf_berlekamp(f, 11, ZZ) == \
        [[1, 1], [1, 5, 3], [1, 2, 3, 4]]

    f = ZZ.map([1, 0, 1, 0, 10, 10, 8, 2, 8])

    Q = ZZ.map([[1, 0, 0, 0, 0, 0, 0, 0],
         [2, 1, 7, 11, 10, 12, 5, 11],
         [3, 6, 4, 3, 0, 4, 7, 2],
         [4, 3, 6, 5, 1, 6, 2, 3],
         [2, 11, 8, 8, 3, 1, 3, 11],
         [6, 11, 8, 6, 2, 7, 10, 9],
         [5, 11, 7, 10, 0, 11, 7, 12],
         [3, 3, 12, 5, 0, 11, 9, 12]])

    V = [[1, 0, 0, 0, 0, 0, 0, 0],
         [0, 5, 5, 0, 9, 5, 1, 0],
         [0, 9, 11, 9, 10, 12, 0, 1]]

    assert gf_Qmatrix(f, 13, ZZ) == Q
    assert gf_Qbasis(Q, 13, ZZ) == V

    assert gf_berlekamp(f, 13, ZZ) == \
        [[1, 3], [1, 8, 4, 12], [1, 2, 3, 4, 6]]


def test_gf_ddf():
    f = gf_from_dict({15: ZZ(1), 0: ZZ(-1)}, 11, ZZ)
    g = [([1, 0, 0, 0, 0, 10], 1),
         ([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], 2)]

    assert gf_ddf_zassenhaus(f, 11, ZZ) == g
    assert gf_ddf_shoup(f, 11, ZZ) == g

    f = gf_from_dict({63: ZZ(1), 0: ZZ(1)}, 2, ZZ)
    g = [([1, 1], 1),
         ([1, 1, 1], 2),
         ([1, 1, 1, 1, 1, 1, 1], 3),
         ([1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1], 6)]

    assert gf_ddf_zassenhaus(f, 2, ZZ) == g
    assert gf_ddf_shoup(f, 2, ZZ) == g

    f = gf_from_dict({6: ZZ(1), 5: ZZ(-1), 4: ZZ(1), 3: ZZ(1), 1: ZZ(-1)}, 3, ZZ)
    g = [([1, 1, 0], 1),
         ([1, 1, 0, 1, 2], 2)]

    assert gf_ddf_zassenhaus(f, 3, ZZ) == g
    assert gf_ddf_shoup(f, 3, ZZ) == g

    f = ZZ.map([1, 2, 5, 26, 677, 436, 791, 325, 456, 24, 577])
    g = [([1, 701], 1),
         ([1, 110, 559, 532, 694, 151, 110, 70, 735, 122], 9)]

    assert gf_ddf_zassenhaus(f, 809, ZZ) == g
    assert gf_ddf_shoup(f, 809, ZZ) == g

    p = ZZ(nextprime(int((2**15 * pi).evalf())))
    f = gf_from_dict({15: 1, 1: 1, 0: 1}, p, ZZ)
    g = [([1, 22730, 68144], 2),
         ([1, 64876, 83977, 10787, 12561, 68608, 52650, 88001, 84356], 4),
         ([1, 15347, 95022, 84569, 94508, 92335], 5)]

    assert gf_ddf_zassenhaus(f, p, ZZ) == g
    assert gf_ddf_shoup(f, p, ZZ) == g


def test_gf_edf():
    f = ZZ.map([1, 1, 0, 1, 2])
    g = ZZ.map([[1, 0, 1], [1, 1, 2]])

    assert gf_edf_zassenhaus(f, 2, 3, ZZ) == g
    assert gf_edf_shoup(f, 2, 3, ZZ) == g


def test_issue_23174():
    f = ZZ.map([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    g = ZZ.map([[1, 0, 0, 1, 1, 1, 0, 0, 1], [1, 1, 1, 0, 1, 0, 1, 1, 1]])

    assert gf_edf_zassenhaus(f, 8, 2, ZZ) == g


def test_gf_factor():
    assert gf_factor([], 11, ZZ) == (0, [])
    assert gf_factor([1], 11, ZZ) == (1, [])
    assert gf_factor([1, 1], 11, ZZ) == (1, [([1, 1], 1)])

    assert gf_factor_sqf([], 11, ZZ) == (0, [])
    assert gf_factor_sqf([1], 11, ZZ) == (1, [])
    assert gf_factor_sqf([1, 1], 11, ZZ) == (1, [[1, 1]])

    config.setup('GF_FACTOR_METHOD', 'berlekamp')

    assert gf_factor_sqf([], 11, ZZ) == (0, [])
    assert gf_factor_sqf([1], 11, ZZ) == (1, [])
    assert gf_factor_sqf([1, 1], 11, ZZ) == (1, [[1, 1]])

    config.setup('GF_FACTOR_METHOD', 'zassenhaus')

    assert gf_factor_sqf([], 11, ZZ) == (0, [])
    assert gf_factor_sqf([1], 11, ZZ) == (1, [])
    assert gf_factor_sqf([1, 1], 11, ZZ) == (1, [[1, 1]])

    config.setup('GF_FACTOR_METHOD', 'shoup')

    assert gf_factor_sqf(ZZ.map([]), 11, ZZ) == (0, [])
    assert gf_factor_sqf(ZZ.map([1]), 11, ZZ) == (1, [])
    assert gf_factor_sqf(ZZ.map([1, 1]), 11, ZZ) == (1, [[1, 1]])

    f, p = ZZ.map([1, 0, 0, 1, 0]), 2

    g = (1, [([1, 0], 1),
             ([1, 1], 1),
             ([1, 1, 1], 1)])

    config.setup('GF_FACTOR_METHOD', 'berlekamp')
    assert gf_factor(f, p, ZZ) == g

    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor(f, p, ZZ) == g

    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor(f, p, ZZ) == g

    g = (1, [[1, 0],
             [1, 1],
             [1, 1, 1]])

    config.setup('GF_FACTOR_METHOD', 'berlekamp')
    assert gf_factor_sqf(f, p, ZZ) == g

    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor_sqf(f, p, ZZ) == g

    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor_sqf(f, p, ZZ) == g

    f, p = gf_from_int_poly([1, -3, 1, -3, -1, -3, 1], 11), 11

    g = (1, [([1, 1], 1),
             ([1, 5, 3], 1),
             ([1, 2, 3, 4], 1)])

    config.setup('GF_FACTOR_METHOD', 'berlekamp')
    assert gf_factor(f, p, ZZ) == g

    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor(f, p, ZZ) == g

    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor(f, p, ZZ) == g

    f, p = [1, 5, 8, 4], 11

    g = (1, [([1, 1], 1), ([1, 2], 2)])

    config.setup('GF_FACTOR_METHOD', 'berlekamp')
    assert gf_factor(f, p, ZZ) == g

    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor(f, p, ZZ) == g

    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor(f, p, ZZ) == g

    f, p = [1, 1, 10, 1, 0, 10, 10, 10, 0, 0], 11

    g = (1, [([1, 0], 2), ([1, 9, 5], 1), ([1, 3, 0, 8, 5, 2], 1)])

    config.setup('GF_FACTOR_METHOD', 'berlekamp')
    assert gf_factor(f, p, ZZ) == g

    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor(f, p, ZZ) == g

    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor(f, p, ZZ) == g

    f, p = gf_from_dict({32: 1, 0: 1}, 11, ZZ), 11

    g = (1, [([1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 10], 1),
             ([1, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 10], 1)])

    config.setup('GF_FACTOR_METHOD', 'berlekamp')
    assert gf_factor(f, p, ZZ) == g

    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor(f, p, ZZ) == g

    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor(f, p, ZZ) == g

    f, p = gf_from_dict({32: ZZ(8), 0: ZZ(5)}, 11, ZZ), 11

    g = (8, [([1, 3], 1),
             ([1, 8], 1),
             ([1, 0, 9], 1),
             ([1, 2, 2], 1),
             ([1, 9, 2], 1),
             ([1, 0, 5, 0, 7], 1),
             ([1, 0, 6, 0, 7], 1),
             ([1, 0, 0, 0, 1, 0, 0, 0, 6], 1),
             ([1, 0, 0, 0, 10, 0, 0, 0, 6], 1)])

    config.setup('GF_FACTOR_METHOD', 'berlekamp')
    assert gf_factor(f, p, ZZ) == g

    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor(f, p, ZZ) == g

    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor(f, p, ZZ) == g

    f, p = gf_from_dict({63: ZZ(8), 0: ZZ(5)}, 11, ZZ), 11

    g = (8, [([1, 7], 1),
             ([1, 4, 5], 1),
             ([1, 6, 8, 2], 1),
             ([1, 9, 9, 2], 1),
             ([1, 0, 0, 9, 0, 0, 4], 1),
             ([1, 2, 0, 8, 4, 6, 4], 1),
             ([1, 2, 3, 8, 0, 6, 4], 1),
             ([1, 2, 6, 0, 8, 4, 4], 1),
             ([1, 3, 3, 1, 6, 8, 4], 1),
             ([1, 5, 6, 0, 8, 6, 4], 1),
             ([1, 6, 2, 7, 9, 8, 4], 1),
             ([1, 10, 4, 7, 10, 7, 4], 1),
             ([1, 10, 10, 1, 4, 9, 4], 1)])

    config.setup('GF_FACTOR_METHOD', 'berlekamp')
    assert gf_factor(f, p, ZZ) == g

    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor(f, p, ZZ) == g

    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor(f, p, ZZ) == g

    # Gathen polynomials: x**n + x + 1 (mod p > 2**n * pi)

    p = ZZ(nextprime(int((2**15 * pi).evalf())))
    f = gf_from_dict({15: 1, 1: 1, 0: 1}, p, ZZ)

    assert gf_sqf_p(f, p, ZZ) is True

    g = (1, [([1, 22730, 68144], 1),
             ([1, 81553, 77449, 86810, 4724], 1),
             ([1, 86276, 56779, 14859, 31575], 1),
             ([1, 15347, 95022, 84569, 94508, 92335], 1)])

    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor(f, p, ZZ) == g

    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor(f, p, ZZ) == g

    g = (1, [[1, 22730, 68144],
             [1, 81553, 77449, 86810, 4724],
             [1, 86276, 56779, 14859, 31575],
             [1, 15347, 95022, 84569, 94508, 92335]])

    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor_sqf(f, p, ZZ) == g

    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor_sqf(f, p, ZZ) == g

    # Shoup polynomials: f = a_0 x**n + a_1 x**(n-1) + ... + a_n
    # (mod p > 2**(n-2) * pi), where a_n = a_{n-1}**2 + 1, a_0 = 1

    p = ZZ(nextprime(int((2**4 * pi).evalf())))
    f = ZZ.map([1, 2, 5, 26, 41, 39, 38])

    assert gf_sqf_p(f, p, ZZ) is True

    g = (1, [([1, 44, 26], 1),
             ([1, 11, 25, 18, 30], 1)])

    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor(f, p, ZZ) == g

    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor(f, p, ZZ) == g

    g = (1, [[1, 44, 26],
             [1, 11, 25, 18, 30]])

    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor_sqf(f, p, ZZ) == g

    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor_sqf(f, p, ZZ) == g

    config.setup('GF_FACTOR_METHOD', 'other')
    raises(KeyError, lambda: gf_factor([1, 1], 11, ZZ))
    config.setup('GF_FACTOR_METHOD')


def test_gf_csolve():
    assert gf_value([1, 7, 2, 4], 11) == 2204

    assert linear_congruence(4, 3, 5) == [2]
    assert linear_congruence(0, 3, 5) == []
    assert linear_congruence(6, 1, 4) == []
    assert linear_congruence(0, 5, 5) == [0, 1, 2, 3, 4]
    assert linear_congruence(3, 12, 15) == [4, 9, 14]
    assert linear_congruence(6, 0, 18) == [0, 3, 6, 9, 12, 15]
    # _csolve_prime_las_vegas
    assert _csolve_prime_las_vegas([2, 3, 1], 5) == [2, 4]
    assert _csolve_prime_las_vegas([2, 0, 1], 5) == []
    from sympy.ntheory import primerange
    for p in primerange(2, 100):
        # f = x**(p-1) - 1
        f = gf_sub_ground(gf_pow([1, 0], p - 1, p, ZZ), 1, p, ZZ)
        assert _csolve_prime_las_vegas(f, p) == list(range(1, p))
    # with power = 1
    assert csolve_prime([1, 3, 2, 17], 7) == [3]
    assert csolve_prime([1, 3, 1, 5], 5) == [0, 1]
    assert csolve_prime([3, 6, 9, 3], 3) == [0, 1, 2]
    # with power > 1
    assert csolve_prime(
        [1, 1, 223], 3, 4) == [4, 13, 22, 31, 40, 49, 58, 67, 76]
    assert csolve_prime([3, 5, 2, 25], 5, 3) == [16, 50, 99]
    assert csolve_prime([3, 2, 2, 49], 7, 3) == [147, 190, 234]

    assert gf_csolve([1, 1, 7], 189) == [13, 49, 76, 112, 139, 175]
    assert gf_csolve([1, 3, 4, 1, 30], 60) == [10, 30]
    assert gf_csolve([1, 1, 7], 15) == []