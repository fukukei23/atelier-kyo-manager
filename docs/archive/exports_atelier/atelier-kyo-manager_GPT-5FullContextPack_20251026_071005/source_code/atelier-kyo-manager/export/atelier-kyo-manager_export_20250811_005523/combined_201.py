
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\concat\test_append_common.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm


@pytest.fixture(
    params=list(
        {
            "bool": [True, False, True],
            "int64": [1, 2, 3],
            "float64": [1.1, np.nan, 3.3],
            "category": Categorical(["X", "Y", "Z"]),
            "object": ["a", "b", "c"],
            "datetime64[ns]": [
                pd.Timestamp("2011-01-01"),
                pd.Timestamp("2011-01-02"),
                pd.Timestamp("2011-01-03"),
            ],
            "datetime64[ns, US/Eastern]": [
                pd.Timestamp("2011-01-01", tz="US/Eastern"),
                pd.Timestamp("2011-01-02", tz="US/Eastern"),
                pd.Timestamp("2011-01-03", tz="US/Eastern"),
            ],
            "timedelta64[ns]": [
                pd.Timedelta("1 days"),
                pd.Timedelta("2 days"),
                pd.Timedelta("3 days"),
            ],
            "period[M]": [
                pd.Period("2011-01", freq="M"),
                pd.Period("2011-02", freq="M"),
                pd.Period("2011-03", freq="M"),
            ],
        }.items()
    )
)
def item(request):
    key, data = request.param
    return key, data


@pytest.fixture
def item2(item):
    return item


class TestConcatAppendCommon:
    """
    Test common dtype coercion rules between concat and append.
    """

    def test_dtypes(self, item, index_or_series, using_infer_string):
        # to confirm test case covers intended dtypes
        typ, vals = item
        obj = index_or_series(vals)
        if typ == "object" and using_infer_string:
            typ = "string"
        if isinstance(obj, Index):
            assert obj.dtype == typ
        elif isinstance(obj, Series):
            if typ.startswith("period"):
                assert obj.dtype == "Period[M]"
            else:
                assert obj.dtype == typ

    def test_concatlike_same_dtypes(self, item):
        # GH 13660
        typ1, vals1 = item

        vals2 = vals1
        vals3 = vals1

        if typ1 == "category":
            exp_data = Categorical(list(vals1) + list(vals2))
            exp_data3 = Categorical(list(vals1) + list(vals2) + list(vals3))
        else:
            exp_data = vals1 + vals2
            exp_data3 = vals1 + vals2 + vals3

        # ----- Index ----- #

        # index.append
        res = Index(vals1).append(Index(vals2))
        exp = Index(exp_data)
        tm.assert_index_equal(res, exp)

        # 3 elements
        res = Index(vals1).append([Index(vals2), Index(vals3)])
        exp = Index(exp_data3)
        tm.assert_index_equal(res, exp)

        # index.append name mismatch
        i1 = Index(vals1, name="x")
        i2 = Index(vals2, name="y")
        res = i1.append(i2)
        exp = Index(exp_data)
        tm.assert_index_equal(res, exp)

        # index.append name match
        i1 = Index(vals1, name="x")
        i2 = Index(vals2, name="x")
        res = i1.append(i2)
        exp = Index(exp_data, name="x")
        tm.assert_index_equal(res, exp)

        # cannot append non-index
        with pytest.raises(TypeError, match="all inputs must be Index"):
            Index(vals1).append(vals2)

        with pytest.raises(TypeError, match="all inputs must be Index"):
            Index(vals1).append([Index(vals2), vals3])

        # ----- Series ----- #

        # series.append
        res = Series(vals1)._append(Series(vals2), ignore_index=True)
        exp = Series(exp_data)
        tm.assert_series_equal(res, exp, check_index_type=True)

        # concat
        res = pd.concat([Series(vals1), Series(vals2)], ignore_index=True)
        tm.assert_series_equal(res, exp, check_index_type=True)

        # 3 elements
        res = Series(vals1)._append([Series(vals2), Series(vals3)], ignore_index=True)
        exp = Series(exp_data3)
        tm.assert_series_equal(res, exp)

        res = pd.concat(
            [Series(vals1), Series(vals2), Series(vals3)],
            ignore_index=True,
        )
        tm.assert_series_equal(res, exp)

        # name mismatch
        s1 = Series(vals1, name="x")
        s2 = Series(vals2, name="y")
        res = s1._append(s2, ignore_index=True)
        exp = Series(exp_data)
        tm.assert_series_equal(res, exp, check_index_type=True)

        res = pd.concat([s1, s2], ignore_index=True)
        tm.assert_series_equal(res, exp, check_index_type=True)

        # name match
        s1 = Series(vals1, name="x")
        s2 = Series(vals2, name="x")
        res = s1._append(s2, ignore_index=True)
        exp = Series(exp_data, name="x")
        tm.assert_series_equal(res, exp, check_index_type=True)

        res = pd.concat([s1, s2], ignore_index=True)
        tm.assert_series_equal(res, exp, check_index_type=True)

        # cannot append non-index
        msg = (
            r"cannot concatenate object of type '.+'; "
            "only Series and DataFrame objs are valid"
        )
        with pytest.raises(TypeError, match=msg):
            Series(vals1)._append(vals2)

        with pytest.raises(TypeError, match=msg):
            Series(vals1)._append([Series(vals2), vals3])

        with pytest.raises(TypeError, match=msg):
            pd.concat([Series(vals1), vals2])

        with pytest.raises(TypeError, match=msg):
            pd.concat([Series(vals1), Series(vals2), vals3])

    def test_concatlike_dtypes_coercion(self, item, item2, request):
        # GH 13660
        typ1, vals1 = item
        typ2, vals2 = item2

        vals3 = vals2

        # basically infer
        exp_index_dtype = None
        exp_series_dtype = None

        if typ1 == typ2:
            pytest.skip("same dtype is tested in test_concatlike_same_dtypes")
        elif typ1 == "category" or typ2 == "category":
            pytest.skip("categorical type tested elsewhere")

        # specify expected dtype
        if typ1 == "bool" and typ2 in ("int64", "float64"):
            # series coerces to numeric based on numpy rule
            # index doesn't because bool is object dtype
            exp_series_dtype = typ2
            mark = pytest.mark.xfail(reason="GH#39187 casting to object")
            request.applymarker(mark)
        elif typ2 == "bool" and typ1 in ("int64", "float64"):
            exp_series_dtype = typ1
            mark = pytest.mark.xfail(reason="GH#39187 casting to object")
            request.applymarker(mark)
        elif typ1 in {"datetime64[ns, US/Eastern]", "timedelta64[ns]"} or typ2 in {
            "datetime64[ns, US/Eastern]",
            "timedelta64[ns]",
        }:
            exp_index_dtype = object
            exp_series_dtype = object

        exp_data = vals1 + vals2
        exp_data3 = vals1 + vals2 + vals3

        # ----- Index ----- #

        # index.append
        # GH#39817
        res = Index(vals1).append(Index(vals2))
        exp = Index(exp_data, dtype=exp_index_dtype)
        tm.assert_index_equal(res, exp)

        # 3 elements
        res = Index(vals1).append([Index(vals2), Index(vals3)])
        exp = Index(exp_data3, dtype=exp_index_dtype)
        tm.assert_index_equal(res, exp)

        # ----- Series ----- #

        # series._append
        # GH#39817
        res = Series(vals1)._append(Series(vals2), ignore_index=True)
        exp = Series(exp_data, dtype=exp_series_dtype)
        tm.assert_series_equal(res, exp, check_index_type=True)

        # concat
        # GH#39817
        res = pd.concat([Series(vals1), Series(vals2)], ignore_index=True)
        tm.assert_series_equal(res, exp, check_index_type=True)

        # 3 elements
        # GH#39817
        res = Series(vals1)._append([Series(vals2), Series(vals3)], ignore_index=True)
        exp = Series(exp_data3, dtype=exp_series_dtype)
        tm.assert_series_equal(res, exp)

        # GH#39817
        res = pd.concat(
            [Series(vals1), Series(vals2), Series(vals3)],
            ignore_index=True,
        )
        tm.assert_series_equal(res, exp)

    def test_concatlike_common_coerce_to_pandas_object(self):
        # GH 13626
        # result must be Timestamp/Timedelta, not datetime.datetime/timedelta
        dti = pd.DatetimeIndex(["2011-01-01", "2011-01-02"])
        tdi = pd.TimedeltaIndex(["1 days", "2 days"])

        exp = Index(
            [
                pd.Timestamp("2011-01-01"),
                pd.Timestamp("2011-01-02"),
                pd.Timedelta("1 days"),
                pd.Timedelta("2 days"),
            ]
        )

        res = dti.append(tdi)
        tm.assert_index_equal(res, exp)
        assert isinstance(res[0], pd.Timestamp)
        assert isinstance(res[-1], pd.Timedelta)

        dts = Series(dti)
        tds = Series(tdi)
        res = dts._append(tds)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))
        assert isinstance(res.iloc[0], pd.Timestamp)
        assert isinstance(res.iloc[-1], pd.Timedelta)

        res = pd.concat([dts, tds])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))
        assert isinstance(res.iloc[0], pd.Timestamp)
        assert isinstance(res.iloc[-1], pd.Timedelta)

    def test_concatlike_datetimetz(self, tz_aware_fixture):
        tz = tz_aware_fixture
        # GH 7795
        dti1 = pd.DatetimeIndex(["2011-01-01", "2011-01-02"], tz=tz)
        dti2 = pd.DatetimeIndex(["2012-01-01", "2012-01-02"], tz=tz)

        exp = pd.DatetimeIndex(
            ["2011-01-01", "2011-01-02", "2012-01-01", "2012-01-02"], tz=tz
        )

        res = dti1.append(dti2)
        tm.assert_index_equal(res, exp)

        dts1 = Series(dti1)
        dts2 = Series(dti2)
        res = dts1._append(dts2)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

        res = pd.concat([dts1, dts2])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

    @pytest.mark.parametrize("tz", ["UTC", "US/Eastern", "Asia/Tokyo", "EST5EDT"])
    def test_concatlike_datetimetz_short(self, tz):
        # GH#7795
        ix1 = pd.date_range(start="2014-07-15", end="2014-07-17", freq="D", tz=tz)
        ix2 = pd.DatetimeIndex(["2014-07-11", "2014-07-21"], tz=tz)
        df1 = DataFrame(0, index=ix1, columns=["A", "B"])
        df2 = DataFrame(0, index=ix2, columns=["A", "B"])

        exp_idx = pd.DatetimeIndex(
            ["2014-07-15", "2014-07-16", "2014-07-17", "2014-07-11", "2014-07-21"],
            tz=tz,
        ).as_unit("ns")
        exp = DataFrame(0, index=exp_idx, columns=["A", "B"])

        tm.assert_frame_equal(df1._append(df2), exp)
        tm.assert_frame_equal(pd.concat([df1, df2]), exp)

    def test_concatlike_datetimetz_to_object(self, tz_aware_fixture):
        tz = tz_aware_fixture
        # GH 13660

        # different tz coerces to object
        dti1 = pd.DatetimeIndex(["2011-01-01", "2011-01-02"], tz=tz)
        dti2 = pd.DatetimeIndex(["2012-01-01", "2012-01-02"])

        exp = Index(
            [
                pd.Timestamp("2011-01-01", tz=tz),
                pd.Timestamp("2011-01-02", tz=tz),
                pd.Timestamp("2012-01-01"),
                pd.Timestamp("2012-01-02"),
            ],
            dtype=object,
        )

        res = dti1.append(dti2)
        tm.assert_index_equal(res, exp)

        dts1 = Series(dti1)
        dts2 = Series(dti2)
        res = dts1._append(dts2)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

        res = pd.concat([dts1, dts2])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

        # different tz
        dti3 = pd.DatetimeIndex(["2012-01-01", "2012-01-02"], tz="US/Pacific")

        exp = Index(
            [
                pd.Timestamp("2011-01-01", tz=tz),
                pd.Timestamp("2011-01-02", tz=tz),
                pd.Timestamp("2012-01-01", tz="US/Pacific"),
                pd.Timestamp("2012-01-02", tz="US/Pacific"),
            ],
            dtype=object,
        )

        res = dti1.append(dti3)
        tm.assert_index_equal(res, exp)

        dts1 = Series(dti1)
        dts3 = Series(dti3)
        res = dts1._append(dts3)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

        res = pd.concat([dts1, dts3])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

    def test_concatlike_common_period(self):
        # GH 13660
        pi1 = pd.PeriodIndex(["2011-01", "2011-02"], freq="M")
        pi2 = pd.PeriodIndex(["2012-01", "2012-02"], freq="M")

        exp = pd.PeriodIndex(["2011-01", "2011-02", "2012-01", "2012-02"], freq="M")

        res = pi1.append(pi2)
        tm.assert_index_equal(res, exp)

        ps1 = Series(pi1)
        ps2 = Series(pi2)
        res = ps1._append(ps2)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

        res = pd.concat([ps1, ps2])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

    def test_concatlike_common_period_diff_freq_to_object(self):
        # GH 13221
        pi1 = pd.PeriodIndex(["2011-01", "2011-02"], freq="M")
        pi2 = pd.PeriodIndex(["2012-01-01", "2012-02-01"], freq="D")

        exp = Index(
            [
                pd.Period("2011-01", freq="M"),
                pd.Period("2011-02", freq="M"),
                pd.Period("2012-01-01", freq="D"),
                pd.Period("2012-02-01", freq="D"),
            ],
            dtype=object,
        )

        res = pi1.append(pi2)
        tm.assert_index_equal(res, exp)

        ps1 = Series(pi1)
        ps2 = Series(pi2)
        res = ps1._append(ps2)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

        res = pd.concat([ps1, ps2])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

    def test_concatlike_common_period_mixed_dt_to_object(self):
        # GH 13221
        # different datetimelike
        pi1 = pd.PeriodIndex(["2011-01", "2011-02"], freq="M")
        tdi = pd.TimedeltaIndex(["1 days", "2 days"])
        exp = Index(
            [
                pd.Period("2011-01", freq="M"),
                pd.Period("2011-02", freq="M"),
                pd.Timedelta("1 days"),
                pd.Timedelta("2 days"),
            ],
            dtype=object,
        )

        res = pi1.append(tdi)
        tm.assert_index_equal(res, exp)

        ps1 = Series(pi1)
        tds = Series(tdi)
        res = ps1._append(tds)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

        res = pd.concat([ps1, tds])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

        # inverse
        exp = Index(
            [
                pd.Timedelta("1 days"),
                pd.Timedelta("2 days"),
                pd.Period("2011-01", freq="M"),
                pd.Period("2011-02", freq="M"),
            ],
            dtype=object,
        )

        res = tdi.append(pi1)
        tm.assert_index_equal(res, exp)

        ps1 = Series(pi1)
        tds = Series(tdi)
        res = tds._append(ps1)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

        res = pd.concat([tds, ps1])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

    def test_concat_categorical(self):
        # GH 13524

        # same categories -> category
        s1 = Series([1, 2, np.nan], dtype="category")
        s2 = Series([2, 1, 2], dtype="category")

        exp = Series([1, 2, np.nan, 2, 1, 2], dtype="category")
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)

        # partially different categories => not-category
        s1 = Series([3, 2], dtype="category")
        s2 = Series([2, 1], dtype="category")

        exp = Series([3, 2, 2, 1])
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)

        # completely different categories (same dtype) => not-category
        s1 = Series([10, 11, np.nan], dtype="category")
        s2 = Series([np.nan, 1, 3, 2], dtype="category")

        exp = Series([10, 11, np.nan, np.nan, 1, 3, 2], dtype=np.float64)
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)

    def test_union_categorical_same_categories_different_order(self):
        # https://github.com/pandas-dev/pandas/issues/19096
        a = Series(Categorical(["a", "b", "c"], categories=["a", "b", "c"]))
        b = Series(Categorical(["a", "b", "c"], categories=["b", "a", "c"]))
        result = pd.concat([a, b], ignore_index=True)
        expected = Series(
            Categorical(["a", "b", "c", "a", "b", "c"], categories=["a", "b", "c"])
        )
        tm.assert_series_equal(result, expected)

    def test_concat_categorical_coercion(self):
        # GH 13524

        # category + not-category => not-category
        s1 = Series([1, 2, np.nan], dtype="category")
        s2 = Series([2, 1, 2])

        exp = Series([1, 2, np.nan, 2, 1, 2], dtype=np.float64)
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)

        # result shouldn't be affected by 1st elem dtype
        exp = Series([2, 1, 2, 1, 2, np.nan], dtype=np.float64)
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
        tm.assert_series_equal(s2._append(s1, ignore_index=True), exp)

        # all values are not in category => not-category
        s1 = Series([3, 2], dtype="category")
        s2 = Series([2, 1])

        exp = Series([3, 2, 2, 1])
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)

        exp = Series([2, 1, 3, 2])
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
        tm.assert_series_equal(s2._append(s1, ignore_index=True), exp)

        # completely different categories => not-category
        s1 = Series([10, 11, np.nan], dtype="category")
        s2 = Series([1, 3, 2])

        exp = Series([10, 11, np.nan, 1, 3, 2], dtype=np.float64)
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)

        exp = Series([1, 3, 2, 10, 11, np.nan], dtype=np.float64)
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
        tm.assert_series_equal(s2._append(s1, ignore_index=True), exp)

        # different dtype => not-category
        s1 = Series([10, 11, np.nan], dtype="category")
        s2 = Series(["a", "b", "c"])

        exp = Series([10, 11, np.nan, "a", "b", "c"])
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)

        exp = Series(["a", "b", "c", 10, 11, np.nan])
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
        tm.assert_series_equal(s2._append(s1, ignore_index=True), exp)

        # if normal series only contains NaN-likes => not-category
        s1 = Series([10, 11], dtype="category")
        s2 = Series([np.nan, np.nan, np.nan])

        exp = Series([10, 11, np.nan, np.nan, np.nan])
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)

        exp = Series([np.nan, np.nan, np.nan, 10, 11])
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
        tm.assert_series_equal(s2._append(s1, ignore_index=True), exp)

    def test_concat_categorical_3elem_coercion(self):
        # GH 13524

        # mixed dtypes => not-category
        s1 = Series([1, 2, np.nan], dtype="category")
        s2 = Series([2, 1, 2], dtype="category")
        s3 = Series([1, 2, 1, 2, np.nan])

        exp = Series([1, 2, np.nan, 2, 1, 2, 1, 2, 1, 2, np.nan], dtype="float")
        tm.assert_series_equal(pd.concat([s1, s2, s3], ignore_index=True), exp)
        tm.assert_series_equal(s1._append([s2, s3], ignore_index=True), exp)

        exp = Series([1, 2, 1, 2, np.nan, 1, 2, np.nan, 2, 1, 2], dtype="float")
        tm.assert_series_equal(pd.concat([s3, s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s3._append([s1, s2], ignore_index=True), exp)

        # values are all in either category => not-category
        s1 = Series([4, 5, 6], dtype="category")
        s2 = Series([1, 2, 3], dtype="category")
        s3 = Series([1, 3, 4])

        exp = Series([4, 5, 6, 1, 2, 3, 1, 3, 4])
        tm.assert_series_equal(pd.concat([s1, s2, s3], ignore_index=True), exp)
        tm.assert_series_equal(s1._append([s2, s3], ignore_index=True), exp)

        exp = Series([1, 3, 4, 4, 5, 6, 1, 2, 3])
        tm.assert_series_equal(pd.concat([s3, s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s3._append([s1, s2], ignore_index=True), exp)

        # values are all in either category => not-category
        s1 = Series([4, 5, 6], dtype="category")
        s2 = Series([1, 2, 3], dtype="category")
        s3 = Series([10, 11, 12])

        exp = Series([4, 5, 6, 1, 2, 3, 10, 11, 12])
        tm.assert_series_equal(pd.concat([s1, s2, s3], ignore_index=True), exp)
        tm.assert_series_equal(s1._append([s2, s3], ignore_index=True), exp)

        exp = Series([10, 11, 12, 4, 5, 6, 1, 2, 3])
        tm.assert_series_equal(pd.concat([s3, s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s3._append([s1, s2], ignore_index=True), exp)

    def test_concat_categorical_multi_coercion(self):
        # GH 13524

        s1 = Series([1, 3], dtype="category")
        s2 = Series([3, 4], dtype="category")
        s3 = Series([2, 3])
        s4 = Series([2, 2], dtype="category")
        s5 = Series([1, np.nan])
        s6 = Series([1, 3, 2], dtype="category")

        # mixed dtype, values are all in categories => not-category
        exp = Series([1, 3, 3, 4, 2, 3, 2, 2, 1, np.nan, 1, 3, 2])
        res = pd.concat([s1, s2, s3, s4, s5, s6], ignore_index=True)
        tm.assert_series_equal(res, exp)
        res = s1._append([s2, s3, s4, s5, s6], ignore_index=True)
        tm.assert_series_equal(res, exp)

        exp = Series([1, 3, 2, 1, np.nan, 2, 2, 2, 3, 3, 4, 1, 3])
        res = pd.concat([s6, s5, s4, s3, s2, s1], ignore_index=True)
        tm.assert_series_equal(res, exp)
        res = s6._append([s5, s4, s3, s2, s1], ignore_index=True)
        tm.assert_series_equal(res, exp)

    def test_concat_categorical_ordered(self):
        # GH 13524

        s1 = Series(Categorical([1, 2, np.nan], ordered=True))
        s2 = Series(Categorical([2, 1, 2], ordered=True))

        exp = Series(Categorical([1, 2, np.nan, 2, 1, 2], ordered=True))
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)

        exp = Series(Categorical([1, 2, np.nan, 2, 1, 2, 1, 2, np.nan], ordered=True))
        tm.assert_series_equal(pd.concat([s1, s2, s1], ignore_index=True), exp)
        tm.assert_series_equal(s1._append([s2, s1], ignore_index=True), exp)

    def test_concat_categorical_coercion_nan(self):
        # GH 13524

        # some edge cases
        # category + not-category => not category
        s1 = Series(np.array([np.nan, np.nan], dtype=np.float64), dtype="category")
        s2 = Series([np.nan, 1])

        exp = Series([np.nan, np.nan, np.nan, 1])
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)

        s1 = Series([1, np.nan], dtype="category")
        s2 = Series([np.nan, np.nan])

        exp = Series([1, np.nan, np.nan, np.nan], dtype="float")
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)

        # mixed dtype, all nan-likes => not-category
        s1 = Series([np.nan, np.nan], dtype="category")
        s2 = Series([np.nan, np.nan])

        exp = Series([np.nan, np.nan, np.nan, np.nan])
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
        tm.assert_series_equal(s2._append(s1, ignore_index=True), exp)

        # all category nan-likes => category
        s1 = Series([np.nan, np.nan], dtype="category")
        s2 = Series([np.nan, np.nan], dtype="category")

        exp = Series([np.nan, np.nan, np.nan, np.nan], dtype="category")

        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)

    def test_concat_categorical_empty(self):
        # GH 13524

        s1 = Series([], dtype="category")
        s2 = Series([1, 2], dtype="category")

        msg = "The behavior of array concatenation with empty entries is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), s2)
            tm.assert_series_equal(s1._append(s2, ignore_index=True), s2)

        with tm.assert_produces_warning(FutureWarning, match=msg):
            tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), s2)
            tm.assert_series_equal(s2._append(s1, ignore_index=True), s2)

        s1 = Series([], dtype="category")
        s2 = Series([], dtype="category")

        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), s2)
        tm.assert_series_equal(s1._append(s2, ignore_index=True), s2)

        s1 = Series([], dtype="category")
        s2 = Series([], dtype="object")

        # different dtype => not-category
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), s2)
        tm.assert_series_equal(s1._append(s2, ignore_index=True), s2)
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), s2)
        tm.assert_series_equal(s2._append(s1, ignore_index=True), s2)

        s1 = Series([], dtype="category")
        s2 = Series([np.nan, np.nan])

        # empty Series is ignored
        exp = Series([np.nan, np.nan])
        with tm.assert_produces_warning(FutureWarning, match=msg):
            tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
            tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)

        with tm.assert_produces_warning(FutureWarning, match=msg):
            tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
            tm.assert_series_equal(s2._append(s1, ignore_index=True), exp)

    def test_categorical_concat_append(self):
        cat = Categorical(["a", "b"], categories=["a", "b"])
        vals = [1, 2]
        df = DataFrame({"cats": cat, "vals": vals})
        cat2 = Categorical(["a", "b", "a", "b"], categories=["a", "b"])
        vals2 = [1, 2, 1, 2]
        exp = DataFrame({"cats": cat2, "vals": vals2}, index=Index([0, 1, 0, 1]))

        tm.assert_frame_equal(pd.concat([df, df]), exp)
        tm.assert_frame_equal(df._append(df), exp)

        # GH 13524 can concat different categories
        cat3 = Categorical(["a", "b"], categories=["a", "b", "c"])
        vals3 = [1, 2]
        df_different_categories = DataFrame({"cats": cat3, "vals": vals3})

        res = pd.concat([df, df_different_categories], ignore_index=True)
        exp = DataFrame({"cats": list("abab"), "vals": [1, 2, 1, 2]})
        tm.assert_frame_equal(res, exp)

        res = df._append(df_different_categories, ignore_index=True)
        tm.assert_frame_equal(res, exp)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\plotting\frame\test_frame_subplots.py ===
""" Test cases for DataFrame.plot """

import string

import numpy as np
import pytest

from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm
from pandas.tests.plotting.common import (
    _check_axes_shape,
    _check_box_return_type,
    _check_legend_labels,
    _check_ticks_props,
    _check_visible,
    _flatten_visible,
)

from pandas.io.formats.printing import pprint_thing

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")


class TestDataFramePlotsSubplots:
    @pytest.mark.slow
    @pytest.mark.parametrize("kind", ["bar", "barh", "line", "area"])
    def test_subplots(self, kind):
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )

        axes = df.plot(kind=kind, subplots=True, sharex=True, legend=True)
        _check_axes_shape(axes, axes_num=3, layout=(3, 1))
        assert axes.shape == (3,)

        for ax, column in zip(axes, df.columns):
            _check_legend_labels(ax, labels=[pprint_thing(column)])

        for ax in axes[:-2]:
            _check_visible(ax.xaxis)  # xaxis must be visible for grid
            _check_visible(ax.get_xticklabels(), visible=False)
            if kind != "bar":
                # change https://github.com/pandas-dev/pandas/issues/26714
                _check_visible(ax.get_xticklabels(minor=True), visible=False)
            _check_visible(ax.xaxis.get_label(), visible=False)
            _check_visible(ax.get_yticklabels())

        _check_visible(axes[-1].xaxis)
        _check_visible(axes[-1].get_xticklabels())
        _check_visible(axes[-1].get_xticklabels(minor=True))
        _check_visible(axes[-1].xaxis.get_label())
        _check_visible(axes[-1].get_yticklabels())

    @pytest.mark.slow
    @pytest.mark.parametrize("kind", ["bar", "barh", "line", "area"])
    def test_subplots_no_share_x(self, kind):
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )
        axes = df.plot(kind=kind, subplots=True, sharex=False)
        for ax in axes:
            _check_visible(ax.xaxis)
            _check_visible(ax.get_xticklabels())
            _check_visible(ax.get_xticklabels(minor=True))
            _check_visible(ax.xaxis.get_label())
            _check_visible(ax.get_yticklabels())

    @pytest.mark.slow
    @pytest.mark.parametrize("kind", ["bar", "barh", "line", "area"])
    def test_subplots_no_legend(self, kind):
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )
        axes = df.plot(kind=kind, subplots=True, legend=False)
        for ax in axes:
            assert ax.get_legend() is None

    @pytest.mark.parametrize("kind", ["line", "area"])
    def test_subplots_timeseries(self, kind):
        idx = date_range(start="2014-07-01", freq="ME", periods=10)
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=idx)

        axes = df.plot(kind=kind, subplots=True, sharex=True)
        _check_axes_shape(axes, axes_num=3, layout=(3, 1))

        for ax in axes[:-2]:
            # GH 7801
            _check_visible(ax.xaxis)  # xaxis must be visible for grid
            _check_visible(ax.get_xticklabels(), visible=False)
            _check_visible(ax.get_xticklabels(minor=True), visible=False)
            _check_visible(ax.xaxis.get_label(), visible=False)
            _check_visible(ax.get_yticklabels())

        _check_visible(axes[-1].xaxis)
        _check_visible(axes[-1].get_xticklabels())
        _check_visible(axes[-1].get_xticklabels(minor=True))
        _check_visible(axes[-1].xaxis.get_label())
        _check_visible(axes[-1].get_yticklabels())
        _check_ticks_props(axes, xrot=0)

    @pytest.mark.parametrize("kind", ["line", "area"])
    def test_subplots_timeseries_rot(self, kind):
        idx = date_range(start="2014-07-01", freq="ME", periods=10)
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=idx)
        axes = df.plot(kind=kind, subplots=True, sharex=False, rot=45, fontsize=7)
        for ax in axes:
            _check_visible(ax.xaxis)
            _check_visible(ax.get_xticklabels())
            _check_visible(ax.get_xticklabels(minor=True))
            _check_visible(ax.xaxis.get_label())
            _check_visible(ax.get_yticklabels())
            _check_ticks_props(ax, xlabelsize=7, xrot=45, ylabelsize=7)

    @pytest.mark.parametrize(
        "col", ["numeric", "timedelta", "datetime_no_tz", "datetime_all_tz"]
    )
    def test_subplots_timeseries_y_axis(self, col):
        # GH16953
        data = {
            "numeric": np.array([1, 2, 5]),
            "timedelta": [
                pd.Timedelta(-10, unit="s"),
                pd.Timedelta(10, unit="m"),
                pd.Timedelta(10, unit="h"),
            ],
            "datetime_no_tz": [
                pd.to_datetime("2017-08-01 00:00:00"),
                pd.to_datetime("2017-08-01 02:00:00"),
                pd.to_datetime("2017-08-02 00:00:00"),
            ],
            "datetime_all_tz": [
                pd.to_datetime("2017-08-01 00:00:00", utc=True),
                pd.to_datetime("2017-08-01 02:00:00", utc=True),
                pd.to_datetime("2017-08-02 00:00:00", utc=True),
            ],
            "text": ["This", "should", "fail"],
        }
        testdata = DataFrame(data)

        ax = testdata.plot(y=col)
        result = ax.get_lines()[0].get_data()[1]
        expected = testdata[col].values
        assert (result == expected).all()

    def test_subplots_timeseries_y_text_error(self):
        # GH16953
        data = {
            "numeric": np.array([1, 2, 5]),
            "text": ["This", "should", "fail"],
        }
        testdata = DataFrame(data)
        msg = "no numeric data to plot"
        with pytest.raises(TypeError, match=msg):
            testdata.plot(y="text")

    @pytest.mark.xfail(reason="not support for period, categorical, datetime_mixed_tz")
    def test_subplots_timeseries_y_axis_not_supported(self):
        """
        This test will fail for:
            period:
                since period isn't yet implemented in ``select_dtypes``
                and because it will need a custom value converter +
                tick formatter (as was done for x-axis plots)

            categorical:
                 because it will need a custom value converter +
                 tick formatter (also doesn't work for x-axis, as of now)

            datetime_mixed_tz:
                because of the way how pandas handles ``Series`` of
                ``datetime`` objects with different timezone,
                generally converting ``datetime`` objects in a tz-aware
                form could help with this problem
        """
        data = {
            "numeric": np.array([1, 2, 5]),
            "period": [
                pd.Period("2017-08-01 00:00:00", freq="H"),
                pd.Period("2017-08-01 02:00", freq="H"),
                pd.Period("2017-08-02 00:00:00", freq="H"),
            ],
            "categorical": pd.Categorical(
                ["c", "b", "a"], categories=["a", "b", "c"], ordered=False
            ),
            "datetime_mixed_tz": [
                pd.to_datetime("2017-08-01 00:00:00", utc=True),
                pd.to_datetime("2017-08-01 02:00:00"),
                pd.to_datetime("2017-08-02 00:00:00"),
            ],
        }
        testdata = DataFrame(data)
        ax_period = testdata.plot(x="numeric", y="period")
        assert (
            ax_period.get_lines()[0].get_data()[1] == testdata["period"].values
        ).all()
        ax_categorical = testdata.plot(x="numeric", y="categorical")
        assert (
            ax_categorical.get_lines()[0].get_data()[1]
            == testdata["categorical"].values
        ).all()
        ax_datetime_mixed_tz = testdata.plot(x="numeric", y="datetime_mixed_tz")
        assert (
            ax_datetime_mixed_tz.get_lines()[0].get_data()[1]
            == testdata["datetime_mixed_tz"].values
        ).all()

    @pytest.mark.parametrize(
        "layout, exp_layout",
        [
            [(2, 2), (2, 2)],
            [(-1, 2), (2, 2)],
            [(2, -1), (2, 2)],
            [(1, 4), (1, 4)],
            [(-1, 4), (1, 4)],
            [(4, -1), (4, 1)],
        ],
    )
    def test_subplots_layout_multi_column(self, layout, exp_layout):
        # GH 6667
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )

        axes = df.plot(subplots=True, layout=layout)
        _check_axes_shape(axes, axes_num=3, layout=exp_layout)
        assert axes.shape == exp_layout

    def test_subplots_layout_multi_column_error(self):
        # GH 6667
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )
        msg = "Layout of 1x1 must be larger than required size 3"

        with pytest.raises(ValueError, match=msg):
            df.plot(subplots=True, layout=(1, 1))

        msg = "At least one dimension of layout must be positive"
        with pytest.raises(ValueError, match=msg):
            df.plot(subplots=True, layout=(-1, -1))

    @pytest.mark.parametrize(
        "kwargs, expected_axes_num, expected_layout, expected_shape",
        [
            ({}, 1, (1, 1), (1,)),
            ({"layout": (3, 3)}, 1, (3, 3), (3, 3)),
        ],
    )
    def test_subplots_layout_single_column(
        self, kwargs, expected_axes_num, expected_layout, expected_shape
    ):
        # GH 6667
        df = DataFrame(
            np.random.default_rng(2).random((10, 1)),
            index=list(string.ascii_letters[:10]),
        )
        axes = df.plot(subplots=True, **kwargs)
        _check_axes_shape(
            axes,
            axes_num=expected_axes_num,
            layout=expected_layout,
        )
        assert axes.shape == expected_shape

    @pytest.mark.slow
    @pytest.mark.parametrize("idx", [range(5), date_range("1/1/2000", periods=5)])
    def test_subplots_warnings(self, idx):
        # GH 9464
        with tm.assert_produces_warning(None):
            df = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), index=idx)
            df.plot(subplots=True, layout=(3, 2))

    def test_subplots_multiple_axes(self):
        # GH 5353, 6970, GH 7069
        fig, axes = mpl.pyplot.subplots(2, 3)
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )

        returned = df.plot(subplots=True, ax=axes[0], sharex=False, sharey=False)
        _check_axes_shape(returned, axes_num=3, layout=(1, 3))
        assert returned.shape == (3,)
        assert returned[0].figure is fig
        # draw on second row
        returned = df.plot(subplots=True, ax=axes[1], sharex=False, sharey=False)
        _check_axes_shape(returned, axes_num=3, layout=(1, 3))
        assert returned.shape == (3,)
        assert returned[0].figure is fig
        _check_axes_shape(axes, axes_num=6, layout=(2, 3))

    def test_subplots_multiple_axes_error(self):
        # GH 5353, 6970, GH 7069
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )
        msg = "The number of passed axes must be 3, the same as the output plot"
        _, axes = mpl.pyplot.subplots(2, 3)

        with pytest.raises(ValueError, match=msg):
            # pass different number of axes from required
            df.plot(subplots=True, ax=axes)

    @pytest.mark.parametrize(
        "layout, exp_layout",
        [
            [(2, 1), (2, 2)],
            [(2, -1), (2, 2)],
            [(-1, 2), (2, 2)],
        ],
    )
    def test_subplots_multiple_axes_2_dim(self, layout, exp_layout):
        # GH 5353, 6970, GH 7069
        # pass 2-dim axes and invalid layout
        # invalid lauout should not affect to input and return value
        # (show warning is tested in
        # TestDataFrameGroupByPlots.test_grouped_box_multiple_axes
        _, axes = mpl.pyplot.subplots(2, 2)
        df = DataFrame(
            np.random.default_rng(2).random((10, 4)),
            index=list(string.ascii_letters[:10]),
        )
        with tm.assert_produces_warning(UserWarning):
            returned = df.plot(
                subplots=True, ax=axes, layout=layout, sharex=False, sharey=False
            )
            _check_axes_shape(returned, axes_num=4, layout=exp_layout)
            assert returned.shape == (4,)

    def test_subplots_multiple_axes_single_col(self):
        # GH 5353, 6970, GH 7069
        # single column
        _, axes = mpl.pyplot.subplots(1, 1)
        df = DataFrame(
            np.random.default_rng(2).random((10, 1)),
            index=list(string.ascii_letters[:10]),
        )

        axes = df.plot(subplots=True, ax=[axes], sharex=False, sharey=False)
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))
        assert axes.shape == (1,)

    def test_subplots_ts_share_axes(self):
        # GH 3964
        _, axes = mpl.pyplot.subplots(3, 3, sharex=True, sharey=True)
        mpl.pyplot.subplots_adjust(left=0.05, right=0.95, hspace=0.3, wspace=0.3)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 9)),
            index=date_range(start="2014-07-01", freq="ME", periods=10),
        )
        for i, ax in enumerate(axes.ravel()):
            df[i].plot(ax=ax, fontsize=5)

        # Rows other than bottom should not be visible
        for ax in axes[0:-1].ravel():
            _check_visible(ax.get_xticklabels(), visible=False)

        # Bottom row should be visible
        for ax in axes[-1].ravel():
            _check_visible(ax.get_xticklabels(), visible=True)

        # First column should be visible
        for ax in axes[[0, 1, 2], [0]].ravel():
            _check_visible(ax.get_yticklabels(), visible=True)

        # Other columns should not be visible
        for ax in axes[[0, 1, 2], [1]].ravel():
            _check_visible(ax.get_yticklabels(), visible=False)
        for ax in axes[[0, 1, 2], [2]].ravel():
            _check_visible(ax.get_yticklabels(), visible=False)

    def test_subplots_sharex_axes_existing_axes(self):
        # GH 9158
        d = {"A": [1.0, 2.0, 3.0, 4.0], "B": [4.0, 3.0, 2.0, 1.0], "C": [5, 1, 3, 4]}
        df = DataFrame(d, index=date_range("2014 10 11", "2014 10 14"))

        axes = df[["A", "B"]].plot(subplots=True)
        df["C"].plot(ax=axes[0], secondary_y=True)

        _check_visible(axes[0].get_xticklabels(), visible=False)
        _check_visible(axes[1].get_xticklabels(), visible=True)
        for ax in axes.ravel():
            _check_visible(ax.get_yticklabels(), visible=True)

    def test_subplots_dup_columns(self):
        # GH 10962
        df = DataFrame(np.random.default_rng(2).random((5, 5)), columns=list("aaaaa"))
        axes = df.plot(subplots=True)
        for ax in axes:
            _check_legend_labels(ax, labels=["a"])
            assert len(ax.lines) == 1

    def test_subplots_dup_columns_secondary_y(self):
        # GH 10962
        df = DataFrame(np.random.default_rng(2).random((5, 5)), columns=list("aaaaa"))
        axes = df.plot(subplots=True, secondary_y="a")
        for ax in axes:
            # (right) is only attached when subplots=False
            _check_legend_labels(ax, labels=["a"])
            assert len(ax.lines) == 1

    def test_subplots_dup_columns_secondary_y_no_subplot(self):
        # GH 10962
        df = DataFrame(np.random.default_rng(2).random((5, 5)), columns=list("aaaaa"))
        ax = df.plot(secondary_y="a")
        _check_legend_labels(ax, labels=["a (right)"] * 5)
        assert len(ax.lines) == 0
        assert len(ax.right_ax.lines) == 5

    @pytest.mark.xfail(
        np_version_gte1p24 and is_platform_linux(),
        reason="Weird rounding problems",
        strict=False,
    )
    def test_bar_log_no_subplots(self):
        # GH3254, GH3298 matplotlib/matplotlib#1882, #1892
        # regressions in 1.2.1
        expected = np.array([0.1, 1.0, 10.0, 100])

        # no subplots
        df = DataFrame({"A": [3] * 5, "B": list(range(1, 6))}, index=range(5))
        ax = df.plot.bar(grid=True, log=True)
        tm.assert_numpy_array_equal(ax.yaxis.get_ticklocs(), expected)

    @pytest.mark.xfail(
        np_version_gte1p24 and is_platform_linux(),
        reason="Weird rounding problems",
        strict=False,
    )
    def test_bar_log_subplots(self):
        expected = np.array([0.1, 1.0, 10.0, 100.0, 1000.0, 1e4])

        ax = DataFrame([Series([200, 300]), Series([300, 500])]).plot.bar(
            log=True, subplots=True
        )

        tm.assert_numpy_array_equal(ax[0].yaxis.get_ticklocs(), expected)
        tm.assert_numpy_array_equal(ax[1].yaxis.get_ticklocs(), expected)

    def test_boxplot_subplots_return_type_default(self, hist_df):
        df = hist_df

        # normal style: return_type=None
        result = df.plot.box(subplots=True)
        assert isinstance(result, Series)
        _check_box_return_type(
            result, None, expected_keys=["height", "weight", "category"]
        )

    @pytest.mark.parametrize("rt", ["dict", "axes", "both"])
    def test_boxplot_subplots_return_type(self, hist_df, rt):
        df = hist_df
        returned = df.plot.box(return_type=rt, subplots=True)
        _check_box_return_type(
            returned,
            rt,
            expected_keys=["height", "weight", "category"],
            check_ax_title=False,
        )

    def test_df_subplots_patterns_minorticks(self):
        # GH 10657
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)),
            index=date_range("1/1/2000", periods=10),
            columns=list("AB"),
        )

        # shared subplots
        _, axes = plt.subplots(2, 1, sharex=True)
        axes = df.plot(subplots=True, ax=axes)
        for ax in axes:
            assert len(ax.lines) == 1
            _check_visible(ax.get_yticklabels(), visible=True)
        # xaxis of 1st ax must be hidden
        _check_visible(axes[0].get_xticklabels(), visible=False)
        _check_visible(axes[0].get_xticklabels(minor=True), visible=False)
        _check_visible(axes[1].get_xticklabels(), visible=True)
        _check_visible(axes[1].get_xticklabels(minor=True), visible=True)

    def test_df_subplots_patterns_minorticks_1st_ax_hidden(self):
        # GH 10657
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)),
            index=date_range("1/1/2000", periods=10),
            columns=list("AB"),
        )
        _, axes = plt.subplots(2, 1)
        with tm.assert_produces_warning(UserWarning):
            axes = df.plot(subplots=True, ax=axes, sharex=True)
        for ax in axes:
            assert len(ax.lines) == 1
            _check_visible(ax.get_yticklabels(), visible=True)
        # xaxis of 1st ax must be hidden
        _check_visible(axes[0].get_xticklabels(), visible=False)
        _check_visible(axes[0].get_xticklabels(minor=True), visible=False)
        _check_visible(axes[1].get_xticklabels(), visible=True)
        _check_visible(axes[1].get_xticklabels(minor=True), visible=True)

    def test_df_subplots_patterns_minorticks_not_shared(self):
        # GH 10657
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)),
            index=date_range("1/1/2000", periods=10),
            columns=list("AB"),
        )
        # not shared
        _, axes = plt.subplots(2, 1)
        axes = df.plot(subplots=True, ax=axes)
        for ax in axes:
            assert len(ax.lines) == 1
            _check_visible(ax.get_yticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(minor=True), visible=True)

    def test_subplots_sharex_false(self):
        # test when sharex is set to False, two plots should have different
        # labels, GH 25160
        df = DataFrame(np.random.default_rng(2).random((10, 2)))
        df.iloc[5:, 1] = np.nan
        df.iloc[:5, 0] = np.nan

        _, axs = mpl.pyplot.subplots(2, 1)
        df.plot.line(ax=axs, subplots=True, sharex=False)

        expected_ax1 = np.arange(4.5, 10, 0.5)
        expected_ax2 = np.arange(-0.5, 5, 0.5)

        tm.assert_numpy_array_equal(axs[0].get_xticks(), expected_ax1)
        tm.assert_numpy_array_equal(axs[1].get_xticks(), expected_ax2)

    def test_subplots_constrained_layout(self):
        # GH 25261
        idx = date_range(start="now", periods=10)
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=idx)
        kwargs = {}
        if hasattr(mpl.pyplot.Figure, "get_constrained_layout"):
            kwargs["constrained_layout"] = True
        _, axes = mpl.pyplot.subplots(2, **kwargs)
        with tm.assert_produces_warning(None):
            df.plot(ax=axes[0])
            with tm.ensure_clean(return_filelike=True) as path:
                mpl.pyplot.savefig(path)

    @pytest.mark.parametrize(
        "index_name, old_label, new_label",
        [
            (None, "", "new"),
            ("old", "old", "new"),
            (None, "", ""),
            (None, "", 1),
            (None, "", [1, 2]),
        ],
    )
    @pytest.mark.parametrize("kind", ["line", "area", "bar"])
    def test_xlabel_ylabel_dataframe_subplots(
        self, kind, index_name, old_label, new_label
    ):
        # GH 9093
        df = DataFrame([[1, 2], [2, 5]], columns=["Type A", "Type B"])
        df.index.name = index_name

        # default is the ylabel is not shown and xlabel is index name
        axes = df.plot(kind=kind, subplots=True)
        assert all(ax.get_ylabel() == "" for ax in axes)
        assert all(ax.get_xlabel() == old_label for ax in axes)

        # old xlabel will be overridden and assigned ylabel will be used as ylabel
        axes = df.plot(kind=kind, ylabel=new_label, xlabel=new_label, subplots=True)
        assert all(ax.get_ylabel() == str(new_label) for ax in axes)
        assert all(ax.get_xlabel() == str(new_label) for ax in axes)

    @pytest.mark.parametrize(
        "kwargs",
        [
            # stacked center
            {"kind": "bar", "stacked": True},
            {"kind": "bar", "stacked": True, "width": 0.9},
            {"kind": "barh", "stacked": True},
            {"kind": "barh", "stacked": True, "width": 0.9},
            # center
            {"kind": "bar", "stacked": False},
            {"kind": "bar", "stacked": False, "width": 0.9},
            {"kind": "barh", "stacked": False},
            {"kind": "barh", "stacked": False, "width": 0.9},
            # subplots center
            {"kind": "bar", "subplots": True},
            {"kind": "bar", "subplots": True, "width": 0.9},
            {"kind": "barh", "subplots": True},
            {"kind": "barh", "subplots": True, "width": 0.9},
            # align edge
            {"kind": "bar", "stacked": True, "align": "edge"},
            {"kind": "bar", "stacked": True, "width": 0.9, "align": "edge"},
            {"kind": "barh", "stacked": True, "align": "edge"},
            {"kind": "barh", "stacked": True, "width": 0.9, "align": "edge"},
            {"kind": "bar", "stacked": False, "align": "edge"},
            {"kind": "bar", "stacked": False, "width": 0.9, "align": "edge"},
            {"kind": "barh", "stacked": False, "align": "edge"},
            {"kind": "barh", "stacked": False, "width": 0.9, "align": "edge"},
            {"kind": "bar", "subplots": True, "align": "edge"},
            {"kind": "bar", "subplots": True, "width": 0.9, "align": "edge"},
            {"kind": "barh", "subplots": True, "align": "edge"},
            {"kind": "barh", "subplots": True, "width": 0.9, "align": "edge"},
        ],
    )
    def test_bar_align_multiple_columns(self, kwargs):
        # GH2157
        df = DataFrame({"A": [3] * 5, "B": list(range(5))}, index=range(5))
        self._check_bar_alignment(df, **kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"kind": "bar", "stacked": False},
            {"kind": "bar", "stacked": True},
            {"kind": "barh", "stacked": False},
            {"kind": "barh", "stacked": True},
            {"kind": "bar", "subplots": True},
            {"kind": "barh", "subplots": True},
        ],
    )
    def test_bar_align_single_column(self, kwargs):
        df = DataFrame(np.random.default_rng(2).standard_normal(5))
        self._check_bar_alignment(df, **kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"kind": "bar", "stacked": False},
            {"kind": "bar", "stacked": True},
            {"kind": "barh", "stacked": False},
            {"kind": "barh", "stacked": True},
            {"kind": "bar", "subplots": True},
            {"kind": "barh", "subplots": True},
        ],
    )
    def test_bar_barwidth_position(self, kwargs):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        self._check_bar_alignment(df, width=0.9, position=0.2, **kwargs)

    @pytest.mark.parametrize("w", [1, 1.0])
    def test_bar_barwidth_position_int(self, w):
        # GH 12979
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot.bar(stacked=True, width=w)
        ticks = ax.xaxis.get_ticklocs()
        tm.assert_numpy_array_equal(ticks, np.array([0, 1, 2, 3, 4]))
        assert ax.get_xlim() == (-0.75, 4.75)
        # check left-edge of bars
        assert ax.patches[0].get_x() == -0.5
        assert ax.patches[-1].get_x() == 3.5

    @pytest.mark.parametrize(
        "kind, kwargs",
        [
            ["bar", {"stacked": True}],
            ["barh", {"stacked": False}],
            ["barh", {"stacked": True}],
            ["bar", {"subplots": True}],
            ["barh", {"subplots": True}],
        ],
    )
    def test_bar_barwidth_position_int_width_1(self, kind, kwargs):
        # GH 12979
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        self._check_bar_alignment(df, kind=kind, width=1, **kwargs)

    def _check_bar_alignment(
        self,
        df,
        kind="bar",
        stacked=False,
        subplots=False,
        align="center",
        width=0.5,
        position=0.5,
    ):
        axes = df.plot(
            kind=kind,
            stacked=stacked,
            subplots=subplots,
            align=align,
            width=width,
            position=position,
            grid=True,
        )

        axes = _flatten_visible(axes)

        for ax in axes:
            if kind == "bar":
                axis = ax.xaxis
                ax_min, ax_max = ax.get_xlim()
                min_edge = min(p.get_x() for p in ax.patches)
                max_edge = max(p.get_x() + p.get_width() for p in ax.patches)
            elif kind == "barh":
                axis = ax.yaxis
                ax_min, ax_max = ax.get_ylim()
                min_edge = min(p.get_y() for p in ax.patches)
                max_edge = max(p.get_y() + p.get_height() for p in ax.patches)
            else:
                raise ValueError

            # GH 7498
            # compare margins between lim and bar edges
            tm.assert_almost_equal(ax_min, min_edge - 0.25)
            tm.assert_almost_equal(ax_max, max_edge + 0.25)

            p = ax.patches[0]
            if kind == "bar" and (stacked is True or subplots is True):
                edge = p.get_x()
                center = edge + p.get_width() * position
            elif kind == "bar" and stacked is False:
                center = p.get_x() + p.get_width() * len(df.columns) * position
                edge = p.get_x()
            elif kind == "barh" and (stacked is True or subplots is True):
                center = p.get_y() + p.get_height() * position
                edge = p.get_y()
            elif kind == "barh" and stacked is False:
                center = p.get_y() + p.get_height() * len(df.columns) * position
                edge = p.get_y()
            else:
                raise ValueError

            # Check the ticks locates on integer
            assert (axis.get_ticklocs() == np.arange(len(df))).all()

            if align == "center":
                # Check whether the bar locates on center
                tm.assert_almost_equal(axis.get_ticklocs()[0], center)
            elif align == "edge":
                # Check whether the bar's edge starts from the tick
                tm.assert_almost_equal(axis.get_ticklocs()[0], edge)
            else:
                raise ValueError

        return axes

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\numberfields\tests\test_modules.py ===
from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
    ClosureFailure, MissingUnityError, StructureError
)
from sympy.polys.numberfields.modules import (
    Module, ModuleElement, ModuleEndomorphism, PowerBasis, PowerBasisElement,
    find_min_poly, is_sq_maxrank_HNF, make_mod_elt, to_col,
)
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises


def test_to_col():
    c = [1, 2, 3, 4]
    m = to_col(c)
    assert m.domain.is_ZZ
    assert m.shape == (4, 1)
    assert m.flat() == c


def test_Module_NotImplemented():
    M = Module()
    raises(NotImplementedError, lambda: M.n)
    raises(NotImplementedError, lambda: M.mult_tab())
    raises(NotImplementedError, lambda: M.represent(None))
    raises(NotImplementedError, lambda: M.starts_with_unity())
    raises(NotImplementedError, lambda: M.element_from_rational(QQ(2, 3)))


def test_Module_ancestors():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    C = B.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    D = B.submodule_from_matrix(5 * DomainMatrix.eye(4, ZZ))
    assert C.ancestors(include_self=True) == [A, B, C]
    assert D.ancestors(include_self=True) == [A, B, D]
    assert C.power_basis_ancestor() == A
    assert C.nearest_common_ancestor(D) == B
    M = Module()
    assert M.power_basis_ancestor() is None


def test_Module_compat_col():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    col = to_col([1, 2, 3, 4])
    row = col.transpose()
    assert A.is_compat_col(col) is True
    assert A.is_compat_col(row) is False
    assert A.is_compat_col(1) is False
    assert A.is_compat_col(DomainMatrix.eye(3, ZZ)[:, 0]) is False
    assert A.is_compat_col(DomainMatrix.eye(4, QQ)[:, 0]) is False
    assert A.is_compat_col(DomainMatrix.eye(4, ZZ)[:, 0]) is True


def test_Module_call():
    T = Poly(cyclotomic_poly(5, x))
    B = PowerBasis(T)
    assert B(0).col.flat() == [1, 0, 0, 0]
    assert B(1).col.flat() == [0, 1, 0, 0]
    col = DomainMatrix.eye(4, ZZ)[:, 2]
    assert B(col).col == col
    raises(ValueError, lambda: B(-1))


def test_Module_starts_with_unity():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    assert A.starts_with_unity() is True
    assert B.starts_with_unity() is False


def test_Module_basis_elements():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    basis = B.basis_elements()
    bp = B.basis_element_pullbacks()
    for i, (e, p) in enumerate(zip(basis, bp)):
        c = [0] * 4
        assert e.module == B
        assert p.module == A
        c[i] = 1
        assert e == B(to_col(c))
        c[i] = 2
        assert p == A(to_col(c))


def test_Module_zero():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    assert A.zero().col.flat() == [0, 0, 0, 0]
    assert A.zero().module == A
    assert B.zero().col.flat() == [0, 0, 0, 0]
    assert B.zero().module == B


def test_Module_one():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    assert A.one().col.flat() == [1, 0, 0, 0]
    assert A.one().module == A
    assert B.one().col.flat() == [1, 0, 0, 0]
    assert B.one().module == A


def test_Module_element_from_rational():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    rA = A.element_from_rational(QQ(22, 7))
    rB = B.element_from_rational(QQ(22, 7))
    assert rA.coeffs == [22, 0, 0, 0]
    assert rA.denom == 7
    assert rA.module == A
    assert rB.coeffs == [22, 0, 0, 0]
    assert rB.denom == 7
    assert rB.module == A


def test_Module_submodule_from_gens():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    gens = [2*A(0), 2*A(1), 6*A(0), 6*A(1)]
    B = A.submodule_from_gens(gens)
    # Because the 3rd and 4th generators do not add anything new, we expect
    # the cols of the matrix of B to just reproduce the first two gens:
    M = gens[0].column().hstack(gens[1].column())
    assert B.matrix == M
    # At least one generator must be provided:
    raises(ValueError, lambda: A.submodule_from_gens([]))
    # All generators must belong to A:
    raises(ValueError, lambda: A.submodule_from_gens([3*A(0), B(0)]))


def test_Module_submodule_from_matrix():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    e = B(to_col([1, 2, 3, 4]))
    f = e.to_parent()
    assert f.col.flat() == [2, 4, 6, 8]
    # Matrix must be over ZZ:
    raises(ValueError, lambda: A.submodule_from_matrix(DomainMatrix.eye(4, QQ)))
    # Number of rows of matrix must equal number of generators of module A:
    raises(ValueError, lambda: A.submodule_from_matrix(2 * DomainMatrix.eye(5, ZZ)))


def test_Module_whole_submodule():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.whole_submodule()
    e = B(to_col([1, 2, 3, 4]))
    f = e.to_parent()
    assert f.col.flat() == [1, 2, 3, 4]
    e0, e1, e2, e3 = B(0), B(1), B(2), B(3)
    assert e2 * e3 == e0
    assert e3 ** 2 == e1


def test_PowerBasis_repr():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    assert repr(A) == 'PowerBasis(x**4 + x**3 + x**2 + x + 1)'


def test_PowerBasis_eq():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = PowerBasis(T)
    assert A == B


def test_PowerBasis_mult_tab():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    M = A.mult_tab()
    exp = {0: {0: [1, 0, 0, 0], 1: [0, 1, 0, 0], 2: [0, 0, 1, 0], 3: [0, 0, 0, 1]},
           1: {1: [0, 0, 1, 0], 2: [0, 0, 0, 1], 3: [-1, -1, -1, -1]},
           2: {2: [-1, -1, -1, -1], 3: [1, 0, 0, 0]},
           3: {3: [0, 1, 0, 0]}}
    # We get the table we expect:
    assert M == exp
    # And all entries are of expected type:
    assert all(is_int(c) for u in M for v in M[u] for c in M[u][v])


def test_PowerBasis_represent():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    col = to_col([1, 2, 3, 4])
    a = A(col)
    assert A.represent(a) == col
    b = A(col, denom=2)
    raises(ClosureFailure, lambda: A.represent(b))


def test_PowerBasis_element_from_poly():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    f = Poly(1 + 2*x)
    g = Poly(x**4)
    h = Poly(0, x)
    assert A.element_from_poly(f).coeffs == [1, 2, 0, 0]
    assert A.element_from_poly(g).coeffs == [-1, -1, -1, -1]
    assert A.element_from_poly(h).coeffs == [0, 0, 0, 0]


def test_PowerBasis_element__conversions():
    k = QQ.cyclotomic_field(5)
    L = QQ.cyclotomic_field(7)
    B = PowerBasis(k)

    # ANP --> PowerBasisElement
    a = k([QQ(1, 2), QQ(1, 3), 5, 7])
    e = B.element_from_ANP(a)
    assert e.coeffs == [42, 30, 2, 3]
    assert e.denom == 6

    # PowerBasisElement --> ANP
    assert e.to_ANP() == a

    # Cannot convert ANP from different field
    d = L([QQ(1, 2), QQ(1, 3), 5, 7])
    raises(UnificationFailed, lambda: B.element_from_ANP(d))

    # AlgebraicNumber --> PowerBasisElement
    alpha = k.to_alg_num(a)
    eps = B.element_from_alg_num(alpha)
    assert eps.coeffs == [42, 30, 2, 3]
    assert eps.denom == 6

    # PowerBasisElement --> AlgebraicNumber
    assert eps.to_alg_num() == alpha

    # Cannot convert AlgebraicNumber from different field
    delta = L.to_alg_num(d)
    raises(UnificationFailed, lambda: B.element_from_alg_num(delta))

    # When we don't know the field:
    C = PowerBasis(k.ext.minpoly)
    # Can convert from AlgebraicNumber:
    eps = C.element_from_alg_num(alpha)
    assert eps.coeffs == [42, 30, 2, 3]
    assert eps.denom == 6
    # But can't convert back:
    raises(StructureError, lambda: eps.to_alg_num())


def test_Submodule_repr():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ), denom=3)
    assert repr(B) == 'Submodule[[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]]/3'


def test_Submodule_reduced():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    C = A.submodule_from_matrix(6 * DomainMatrix.eye(4, ZZ), denom=3)
    D = C.reduced()
    assert D.denom == 1 and D == C == B


def test_Submodule_discard_before():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    B.compute_mult_tab()
    C = B.discard_before(2)
    assert C.parent == B.parent
    assert B.is_sq_maxrank_HNF() and not C.is_sq_maxrank_HNF()
    assert C.matrix == B.matrix[:, 2:]
    assert C.mult_tab() == {0: {0: [-2, -2], 1: [0, 0]}, 1: {1: [0, 0]}}


def test_Submodule_QQ_matrix():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    C = A.submodule_from_matrix(6 * DomainMatrix.eye(4, ZZ), denom=3)
    assert C.QQ_matrix == B.QQ_matrix


def test_Submodule_represent():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    C = B.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    a0 = A(to_col([6, 12, 18, 24]))
    a1 = A(to_col([2, 4, 6, 8]))
    a2 = A(to_col([1, 3, 5, 7]))

    b1 = B.represent(a1)
    assert b1.flat() == [1, 2, 3, 4]

    c0 = C.represent(a0)
    assert c0.flat() == [1, 2, 3, 4]

    Y = A.submodule_from_matrix(DomainMatrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ], (3, 4), ZZ).transpose())

    U = Poly(cyclotomic_poly(7, x))
    Z = PowerBasis(U)
    z0 = Z(to_col([1, 2, 3, 4, 5, 6]))

    raises(ClosureFailure, lambda: Y.represent(A(3)))
    raises(ClosureFailure, lambda: B.represent(a2))
    raises(ClosureFailure, lambda: B.represent(z0))


def test_Submodule_is_compat_submodule():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    D = C.submodule_from_matrix(5 * DomainMatrix.eye(4, ZZ))
    assert B.is_compat_submodule(C) is True
    assert B.is_compat_submodule(A) is False
    assert B.is_compat_submodule(D) is False


def test_Submodule_eq():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    C = A.submodule_from_matrix(6 * DomainMatrix.eye(4, ZZ), denom=3)
    assert C == B


def test_Submodule_add():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(DomainMatrix([
        [4, 0, 0, 0],
        [0, 4, 0, 0],
    ], (2, 4), ZZ).transpose(), denom=6)
    C = A.submodule_from_matrix(DomainMatrix([
        [0, 10, 0, 0],
        [0,  0, 7, 0],
    ], (2, 4), ZZ).transpose(), denom=15)
    D = A.submodule_from_matrix(DomainMatrix([
        [20,  0,  0, 0],
        [ 0, 20,  0, 0],
        [ 0,  0, 14, 0],
    ], (3, 4), ZZ).transpose(), denom=30)
    assert B + C == D

    U = Poly(cyclotomic_poly(7, x))
    Z = PowerBasis(U)
    Y = Z.submodule_from_gens([Z(0), Z(1)])
    raises(TypeError, lambda: B + Y)


def test_Submodule_mul():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    C = A.submodule_from_matrix(DomainMatrix([
        [0, 10, 0, 0],
        [0, 0, 7, 0],
    ], (2, 4), ZZ).transpose(), denom=15)
    C1 = A.submodule_from_matrix(DomainMatrix([
        [0, 20, 0, 0],
        [0, 0, 14, 0],
    ], (2, 4), ZZ).transpose(), denom=3)
    C2 = A.submodule_from_matrix(DomainMatrix([
        [0, 0, 10, 0],
        [0, 0,  0, 7],
    ], (2, 4), ZZ).transpose(), denom=15)
    C3_unred = A.submodule_from_matrix(DomainMatrix([
        [0, 0, 100, 0],
        [0, 0, 0, 70],
        [0, 0, 0, 70],
        [-49, -49, -49, -49]
    ], (4, 4), ZZ).transpose(), denom=225)
    C3 = A.submodule_from_matrix(DomainMatrix([
        [4900, 4900, 0, 0],
        [4410, 4410, 10, 0],
        [2107, 2107, 7, 7]
    ], (3, 4), ZZ).transpose(), denom=225)
    assert C * 1 == C
    assert C ** 1 == C
    assert C * 10 == C1
    assert C * A(1) == C2
    assert C.mul(C, hnf=False) == C3_unred
    assert C * C == C3
    assert C ** 2 == C3


def test_Submodule_reduce_element():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.whole_submodule()
    b = B(to_col([90, 84, 80, 75]), denom=120)

    C = B.submodule_from_matrix(DomainMatrix.eye(4, ZZ), denom=2)
    b_bar_expected = B(to_col([30, 24, 20, 15]), denom=120)
    b_bar = C.reduce_element(b)
    assert b_bar == b_bar_expected

    C = B.submodule_from_matrix(DomainMatrix.eye(4, ZZ), denom=4)
    b_bar_expected = B(to_col([0, 24, 20, 15]), denom=120)
    b_bar = C.reduce_element(b)
    assert b_bar == b_bar_expected

    C = B.submodule_from_matrix(DomainMatrix.eye(4, ZZ), denom=8)
    b_bar_expected = B(to_col([0, 9, 5, 0]), denom=120)
    b_bar = C.reduce_element(b)
    assert b_bar == b_bar_expected

    a = A(to_col([1, 2, 3, 4]))
    raises(NotImplementedError, lambda: C.reduce_element(a))

    C = B.submodule_from_matrix(DomainMatrix([
        [5, 4, 3, 2],
        [0, 8, 7, 6],
        [0, 0,11,12],
        [0, 0, 0, 1]
    ], (4, 4), ZZ).transpose())
    raises(StructureError, lambda: C.reduce_element(b))


def test_is_HNF():
    M = DM([
        [3, 2, 1],
        [0, 2, 1],
        [0, 0, 1]
    ], ZZ)
    M1 = DM([
        [3, 2, 1],
        [0, -2, 1],
        [0, 0, 1]
    ], ZZ)
    M2 = DM([
        [3, 2, 3],
        [0, 2, 1],
        [0, 0, 1]
    ], ZZ)
    assert is_sq_maxrank_HNF(M) is True
    assert is_sq_maxrank_HNF(M1) is False
    assert is_sq_maxrank_HNF(M2) is False


def test_make_mod_elt():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    col = to_col([1, 2, 3, 4])
    eA = make_mod_elt(A, col)
    eB = make_mod_elt(B, col)
    assert isinstance(eA, PowerBasisElement)
    assert not isinstance(eB, PowerBasisElement)


def test_ModuleElement_repr():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    e = A(to_col([1, 2, 3, 4]), denom=2)
    assert repr(e) == '[1, 2, 3, 4]/2'


def test_ModuleElement_reduced():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    e = A(to_col([2, 4, 6, 8]), denom=2)
    f = e.reduced()
    assert f.denom == 1 and f == e


def test_ModuleElement_reduced_mod_p():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    e = A(to_col([20, 40, 60, 80]))
    f = e.reduced_mod_p(7)
    assert f.coeffs == [-1, -2, -3, 3]


def test_ModuleElement_from_int_list():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    c = [1, 2, 3, 4]
    assert ModuleElement.from_int_list(A, c).coeffs == c


def test_ModuleElement_len():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    e = A(0)
    assert len(e) == 4


def test_ModuleElement_column():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    e = A(0)
    col1 = e.column()
    assert col1 == e.col and col1 is not e.col
    col2 = e.column(domain=FF(5))
    assert col2.domain.is_FF


def test_ModuleElement_QQ_col():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    e = A(to_col([1, 2, 3, 4]), denom=1)
    f = A(to_col([3, 6, 9, 12]), denom=3)
    assert e.QQ_col == f.QQ_col


def test_ModuleElement_to_ancestors():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    C = B.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    D = C.submodule_from_matrix(5 * DomainMatrix.eye(4, ZZ))
    eD = D(0)
    eC = eD.to_parent()
    eB = eD.to_ancestor(B)
    eA = eD.over_power_basis()
    assert eC.module is C and eC.coeffs == [5, 0, 0, 0]
    assert eB.module is B and eB.coeffs == [15, 0, 0, 0]
    assert eA.module is A and eA.coeffs == [30, 0, 0, 0]

    a = A(0)
    raises(ValueError, lambda: a.to_parent())


def test_ModuleElement_compatibility():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    C = B.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    D = B.submodule_from_matrix(5 * DomainMatrix.eye(4, ZZ))
    assert C(0).is_compat(C(1)) is True
    assert C(0).is_compat(D(0)) is False
    u, v = C(0).unify(D(0))
    assert u.module is B and v.module is B
    assert C(C.represent(u)) == C(0) and D(D.represent(v)) == D(0)

    u, v = C(0).unify(C(1))
    assert u == C(0) and v == C(1)

    U = Poly(cyclotomic_poly(7, x))
    Z = PowerBasis(U)
    raises(UnificationFailed, lambda: C(0).unify(Z(1)))


def test_ModuleElement_eq():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    e = A(to_col([1, 2, 3, 4]), denom=1)
    f = A(to_col([3, 6, 9, 12]), denom=3)
    assert e == f

    U = Poly(cyclotomic_poly(7, x))
    Z = PowerBasis(U)
    assert e != Z(0)
    assert e != 3.14


def test_ModuleElement_equiv():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    e = A(to_col([1, 2, 3, 4]), denom=1)
    f = A(to_col([3, 6, 9, 12]), denom=3)
    assert e.equiv(f)

    C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    g = C(to_col([1, 2, 3, 4]), denom=1)
    h = A(to_col([3, 6, 9, 12]), denom=1)
    assert g.equiv(h)
    assert C(to_col([5, 0, 0, 0]), denom=7).equiv(QQ(15, 7))

    U = Poly(cyclotomic_poly(7, x))
    Z = PowerBasis(U)
    raises(UnificationFailed, lambda: e.equiv(Z(0)))

    assert e.equiv(3.14) is False


def test_ModuleElement_add():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    e = A(to_col([1, 2, 3, 4]), denom=6)
    f = A(to_col([5, 6, 7, 8]), denom=10)
    g = C(to_col([1, 1, 1, 1]), denom=2)
    assert e + f == A(to_col([10, 14, 18, 22]), denom=15)
    assert e - f == A(to_col([-5, -4, -3, -2]), denom=15)
    assert e + g == A(to_col([10, 11, 12, 13]), denom=6)
    assert e + QQ(7, 10) == A(to_col([26, 10, 15, 20]), denom=30)
    assert g + QQ(7, 10) == A(to_col([22, 15, 15, 15]), denom=10)

    U = Poly(cyclotomic_poly(7, x))
    Z = PowerBasis(U)
    raises(TypeError, lambda: e + Z(0))
    raises(TypeError, lambda: e + 3.14)


def test_ModuleElement_mul():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    e = A(to_col([0, 2, 0, 0]), denom=3)
    f = A(to_col([0, 0, 0, 7]), denom=5)
    g = C(to_col([0, 0, 0, 1]), denom=2)
    h = A(to_col([0, 0, 3, 1]), denom=7)
    assert e * f == A(to_col([-14, -14, -14, -14]), denom=15)
    assert e * g == A(to_col([-1, -1, -1, -1]))
    assert e * h == A(to_col([-2, -2, -2, 4]), denom=21)
    assert e * QQ(6, 5) == A(to_col([0, 4, 0, 0]), denom=5)
    assert (g * QQ(10, 21)).equiv(A(to_col([0, 0, 0, 5]), denom=7))
    assert e // QQ(6, 5) == A(to_col([0, 5, 0, 0]), denom=9)

    U = Poly(cyclotomic_poly(7, x))
    Z = PowerBasis(U)
    raises(TypeError, lambda: e * Z(0))
    raises(TypeError, lambda: e * 3.14)
    raises(TypeError, lambda: e // 3.14)
    raises(ZeroDivisionError, lambda: e // 0)


def test_ModuleElement_div():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    e = A(to_col([0, 2, 0, 0]), denom=3)
    f = A(to_col([0, 0, 0, 7]), denom=5)
    g = C(to_col([1, 1, 1, 1]))
    assert e // f == 10*A(3)//21
    assert e // g == -2*A(2)//9
    assert 3 // g == -A(1)


def test_ModuleElement_pow():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    e = A(to_col([0, 2, 0, 0]), denom=3)
    g = C(to_col([0, 0, 0, 1]), denom=2)
    assert e ** 3 == A(to_col([0, 0, 0, 8]), denom=27)
    assert g ** 2 == C(to_col([0, 3, 0, 0]), denom=4)
    assert e ** 0 == A(to_col([1, 0, 0, 0]))
    assert g ** 0 == A(to_col([1, 0, 0, 0]))
    assert e ** 1 == e
    assert g ** 1 == g


def test_ModuleElement_mod():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    e = A(to_col([1, 15, 8, 0]), denom=2)
    assert e % 7 == A(to_col([1, 1, 8, 0]), denom=2)
    assert e % QQ(1, 2) == A.zero()
    assert e % QQ(1, 3) == A(to_col([1, 1, 0, 0]), denom=6)

    B = A.submodule_from_gens([A(0), 5*A(1), 3*A(2), A(3)])
    assert e % B == A(to_col([1, 5, 2, 0]), denom=2)

    C = B.whole_submodule()
    raises(TypeError, lambda: e % C)


def test_PowerBasisElement_polys():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    e = A(to_col([1, 15, 8, 0]), denom=2)
    assert e.numerator(x=zeta) == Poly(8 * zeta ** 2 + 15 * zeta + 1, domain=ZZ)
    assert e.poly(x=zeta) == Poly(4 * zeta ** 2 + QQ(15, 2) * zeta + QQ(1, 2), domain=QQ)


def test_PowerBasisElement_norm():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    lam = A(to_col([1, -1, 0, 0]))
    assert lam.norm() == 5


def test_PowerBasisElement_inverse():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    e = A(to_col([1, 1, 1, 1]))
    assert 2 // e == -2*A(1)
    assert e ** -3 == -A(3)


def test_ModuleHomomorphism_matrix():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    phi = ModuleEndomorphism(A, lambda a: a ** 2)
    M = phi.matrix()
    assert M == DomainMatrix([
        [1, 0, -1, 0],
        [0, 0, -1, 1],
        [0, 1, -1, 0],
        [0, 0, -1, 0]
    ], (4, 4), ZZ)


def test_ModuleHomomorphism_kernel():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    phi = ModuleEndomorphism(A, lambda a: a ** 5)
    N = phi.kernel()
    assert N.n == 3


def test_EndomorphismRing_represent():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    R = A.endomorphism_ring()
    phi = R.inner_endomorphism(A(1))
    col = R.represent(phi)
    assert col.transpose() == DomainMatrix([
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -1, -1, -1, -1]
    ], (1, 16), ZZ)

    B = A.submodule_from_matrix(DomainMatrix.zeros((4, 0), ZZ))
    S = B.endomorphism_ring()
    psi = S.inner_endomorphism(A(1))
    col = S.represent(psi)
    assert col == DomainMatrix([], (0, 0), ZZ)

    raises(NotImplementedError, lambda: R.represent(3.14))


def test_find_min_poly():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    powers = []
    m = find_min_poly(A(1), QQ, x=x, powers=powers)
    assert m == Poly(T, domain=QQ)
    assert len(powers) == 5

    # powers list need not be passed
    m = find_min_poly(A(1), QQ, x=x)
    assert m == Poly(T, domain=QQ)

    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    raises(MissingUnityError, lambda: find_min_poly(B(1), QQ))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\libs\test_hashtable.py ===
from collections.abc import Generator
from contextlib import contextmanager
import re
import struct
import tracemalloc

import numpy as np
import pytest

from pandas._libs import hashtable as ht

import pandas as pd
import pandas._testing as tm
from pandas.core.algorithms import isin


@contextmanager
def activated_tracemalloc() -> Generator[None, None, None]:
    tracemalloc.start()
    try:
        yield
    finally:
        tracemalloc.stop()


def get_allocated_khash_memory():
    snapshot = tracemalloc.take_snapshot()
    snapshot = snapshot.filter_traces(
        (tracemalloc.DomainFilter(True, ht.get_hashtable_trace_domain()),)
    )
    return sum(x.size for x in snapshot.traces)


@pytest.mark.parametrize(
    "table_type, dtype",
    [
        (ht.PyObjectHashTable, np.object_),
        (ht.Complex128HashTable, np.complex128),
        (ht.Int64HashTable, np.int64),
        (ht.UInt64HashTable, np.uint64),
        (ht.Float64HashTable, np.float64),
        (ht.Complex64HashTable, np.complex64),
        (ht.Int32HashTable, np.int32),
        (ht.UInt32HashTable, np.uint32),
        (ht.Float32HashTable, np.float32),
        (ht.Int16HashTable, np.int16),
        (ht.UInt16HashTable, np.uint16),
        (ht.Int8HashTable, np.int8),
        (ht.UInt8HashTable, np.uint8),
        (ht.IntpHashTable, np.intp),
    ],
)
class TestHashTable:
    def test_get_set_contains_len(self, table_type, dtype):
        index = 5
        table = table_type(55)
        assert len(table) == 0
        assert index not in table

        table.set_item(index, 42)
        assert len(table) == 1
        assert index in table
        assert table.get_item(index) == 42

        table.set_item(index + 1, 41)
        assert index in table
        assert index + 1 in table
        assert len(table) == 2
        assert table.get_item(index) == 42
        assert table.get_item(index + 1) == 41

        table.set_item(index, 21)
        assert index in table
        assert index + 1 in table
        assert len(table) == 2
        assert table.get_item(index) == 21
        assert table.get_item(index + 1) == 41
        assert index + 2 not in table

        table.set_item(index + 1, 21)
        assert index in table
        assert index + 1 in table
        assert len(table) == 2
        assert table.get_item(index) == 21
        assert table.get_item(index + 1) == 21

        with pytest.raises(KeyError, match=str(index + 2)):
            table.get_item(index + 2)

    def test_get_set_contains_len_mask(self, table_type, dtype):
        if table_type == ht.PyObjectHashTable:
            pytest.skip("Mask not supported for object")
        index = 5
        table = table_type(55, uses_mask=True)
        assert len(table) == 0
        assert index not in table

        table.set_item(index, 42)
        assert len(table) == 1
        assert index in table
        assert table.get_item(index) == 42
        with pytest.raises(KeyError, match="NA"):
            table.get_na()

        table.set_item(index + 1, 41)
        table.set_na(41)
        assert pd.NA in table
        assert index in table
        assert index + 1 in table
        assert len(table) == 3
        assert table.get_item(index) == 42
        assert table.get_item(index + 1) == 41
        assert table.get_na() == 41

        table.set_na(21)
        assert index in table
        assert index + 1 in table
        assert len(table) == 3
        assert table.get_item(index + 1) == 41
        assert table.get_na() == 21
        assert index + 2 not in table

        with pytest.raises(KeyError, match=str(index + 2)):
            table.get_item(index + 2)

    def test_map_keys_to_values(self, table_type, dtype, writable):
        # only Int64HashTable has this method
        if table_type == ht.Int64HashTable:
            N = 77
            table = table_type()
            keys = np.arange(N).astype(dtype)
            vals = np.arange(N).astype(np.int64) + N
            keys.flags.writeable = writable
            vals.flags.writeable = writable
            table.map_keys_to_values(keys, vals)
            for i in range(N):
                assert table.get_item(keys[i]) == i + N

    def test_map_locations(self, table_type, dtype, writable):
        N = 8
        table = table_type()
        keys = (np.arange(N) + N).astype(dtype)
        keys.flags.writeable = writable
        table.map_locations(keys)
        for i in range(N):
            assert table.get_item(keys[i]) == i

    def test_map_locations_mask(self, table_type, dtype, writable):
        if table_type == ht.PyObjectHashTable:
            pytest.skip("Mask not supported for object")
        N = 3
        table = table_type(uses_mask=True)
        keys = (np.arange(N) + N).astype(dtype)
        keys.flags.writeable = writable
        table.map_locations(keys, np.array([False, False, True]))
        for i in range(N - 1):
            assert table.get_item(keys[i]) == i

        with pytest.raises(KeyError, match=re.escape(str(keys[N - 1]))):
            table.get_item(keys[N - 1])

        assert table.get_na() == 2

    def test_lookup(self, table_type, dtype, writable):
        N = 3
        table = table_type()
        keys = (np.arange(N) + N).astype(dtype)
        keys.flags.writeable = writable
        table.map_locations(keys)
        result = table.lookup(keys)
        expected = np.arange(N)
        tm.assert_numpy_array_equal(result.astype(np.int64), expected.astype(np.int64))

    def test_lookup_wrong(self, table_type, dtype):
        if dtype in (np.int8, np.uint8):
            N = 100
        else:
            N = 512
        table = table_type()
        keys = (np.arange(N) + N).astype(dtype)
        table.map_locations(keys)
        wrong_keys = np.arange(N).astype(dtype)
        result = table.lookup(wrong_keys)
        assert np.all(result == -1)

    def test_lookup_mask(self, table_type, dtype, writable):
        if table_type == ht.PyObjectHashTable:
            pytest.skip("Mask not supported for object")
        N = 3
        table = table_type(uses_mask=True)
        keys = (np.arange(N) + N).astype(dtype)
        mask = np.array([False, True, False])
        keys.flags.writeable = writable
        table.map_locations(keys, mask)
        result = table.lookup(keys, mask)
        expected = np.arange(N)
        tm.assert_numpy_array_equal(result.astype(np.int64), expected.astype(np.int64))

        result = table.lookup(np.array([1 + N]).astype(dtype), np.array([False]))
        tm.assert_numpy_array_equal(
            result.astype(np.int64), np.array([-1], dtype=np.int64)
        )

    def test_unique(self, table_type, dtype, writable):
        if dtype in (np.int8, np.uint8):
            N = 88
        else:
            N = 1000
        table = table_type()
        expected = (np.arange(N) + N).astype(dtype)
        keys = np.repeat(expected, 5)
        keys.flags.writeable = writable
        unique = table.unique(keys)
        tm.assert_numpy_array_equal(unique, expected)

    def test_tracemalloc_works(self, table_type, dtype):
        if dtype in (np.int8, np.uint8):
            N = 256
        else:
            N = 30000
        keys = np.arange(N).astype(dtype)
        with activated_tracemalloc():
            table = table_type()
            table.map_locations(keys)
            used = get_allocated_khash_memory()
            my_size = table.sizeof()
            assert used == my_size
            del table
            assert get_allocated_khash_memory() == 0

    def test_tracemalloc_for_empty(self, table_type, dtype):
        with activated_tracemalloc():
            table = table_type()
            used = get_allocated_khash_memory()
            my_size = table.sizeof()
            assert used == my_size
            del table
            assert get_allocated_khash_memory() == 0

    def test_get_state(self, table_type, dtype):
        table = table_type(1000)
        state = table.get_state()
        assert state["size"] == 0
        assert state["n_occupied"] == 0
        assert "n_buckets" in state
        assert "upper_bound" in state

    @pytest.mark.parametrize("N", range(1, 110))
    def test_no_reallocation(self, table_type, dtype, N):
        keys = np.arange(N).astype(dtype)
        preallocated_table = table_type(N)
        n_buckets_start = preallocated_table.get_state()["n_buckets"]
        preallocated_table.map_locations(keys)
        n_buckets_end = preallocated_table.get_state()["n_buckets"]
        # original number of buckets was enough:
        assert n_buckets_start == n_buckets_end
        # check with clean table (not too much preallocated)
        clean_table = table_type()
        clean_table.map_locations(keys)
        assert n_buckets_start == clean_table.get_state()["n_buckets"]


class TestHashTableUnsorted:
    # TODO: moved from test_algos; may be redundancies with other tests
    def test_string_hashtable_set_item_signature(self):
        # GH#30419 fix typing in StringHashTable.set_item to prevent segfault
        tbl = ht.StringHashTable()

        tbl.set_item("key", 1)
        assert tbl.get_item("key") == 1

        with pytest.raises(TypeError, match="'key' has incorrect type"):
            # key arg typed as string, not object
            tbl.set_item(4, 6)
        with pytest.raises(TypeError, match="'val' has incorrect type"):
            tbl.get_item(4)

    def test_lookup_nan(self, writable):
        # GH#21688 ensure we can deal with readonly memory views
        xs = np.array([2.718, 3.14, np.nan, -7, 5, 2, 3])
        xs.setflags(write=writable)
        m = ht.Float64HashTable()
        m.map_locations(xs)
        tm.assert_numpy_array_equal(m.lookup(xs), np.arange(len(xs), dtype=np.intp))

    def test_add_signed_zeros(self):
        # GH#21866 inconsistent hash-function for float64
        # default hash-function would lead to different hash-buckets
        # for 0.0 and -0.0 if there are more than 2^30 hash-buckets
        # but this would mean 16GB
        N = 4  # 12 * 10**8 would trigger the error, if you have enough memory
        m = ht.Float64HashTable(N)
        m.set_item(0.0, 0)
        m.set_item(-0.0, 0)
        assert len(m) == 1  # 0.0 and -0.0 are equivalent

    def test_add_different_nans(self):
        # GH#21866 inconsistent hash-function for float64
        # create different nans from bit-patterns:
        NAN1 = struct.unpack("d", struct.pack("=Q", 0x7FF8000000000000))[0]
        NAN2 = struct.unpack("d", struct.pack("=Q", 0x7FF8000000000001))[0]
        assert NAN1 != NAN1
        assert NAN2 != NAN2
        # default hash function would lead to different hash-buckets
        # for NAN1 and NAN2 even if there are only 4 buckets:
        m = ht.Float64HashTable()
        m.set_item(NAN1, 0)
        m.set_item(NAN2, 0)
        assert len(m) == 1  # NAN1 and NAN2 are equivalent

    def test_lookup_overflow(self, writable):
        xs = np.array([1, 2, 2**63], dtype=np.uint64)
        # GH 21688 ensure we can deal with readonly memory views
        xs.setflags(write=writable)
        m = ht.UInt64HashTable()
        m.map_locations(xs)
        tm.assert_numpy_array_equal(m.lookup(xs), np.arange(len(xs), dtype=np.intp))

    @pytest.mark.parametrize("nvals", [0, 10])  # resizing to 0 is special case
    @pytest.mark.parametrize(
        "htable, uniques, dtype, safely_resizes",
        [
            (ht.PyObjectHashTable, ht.ObjectVector, "object", False),
            (ht.StringHashTable, ht.ObjectVector, "object", True),
            (ht.Float64HashTable, ht.Float64Vector, "float64", False),
            (ht.Int64HashTable, ht.Int64Vector, "int64", False),
            (ht.Int32HashTable, ht.Int32Vector, "int32", False),
            (ht.UInt64HashTable, ht.UInt64Vector, "uint64", False),
        ],
    )
    def test_vector_resize(
        self, writable, htable, uniques, dtype, safely_resizes, nvals
    ):
        # Test for memory errors after internal vector
        # reallocations (GH 7157)
        # Changed from using np.random.default_rng(2).rand to range
        # which could cause flaky CI failures when safely_resizes=False
        vals = np.array(range(1000), dtype=dtype)

        # GH 21688 ensures we can deal with read-only memory views
        vals.setflags(write=writable)

        # initialise instances; cannot initialise in parametrization,
        # as otherwise external views would be held on the array (which is
        # one of the things this test is checking)
        htable = htable()
        uniques = uniques()

        # get_labels may append to uniques
        htable.get_labels(vals[:nvals], uniques, 0, -1)
        # to_array() sets an external_view_exists flag on uniques.
        tmp = uniques.to_array()
        oldshape = tmp.shape

        # subsequent get_labels() calls can no longer append to it
        # (except for StringHashTables + ObjectVector)
        if safely_resizes:
            htable.get_labels(vals, uniques, 0, -1)
        else:
            with pytest.raises(ValueError, match="external reference.*"):
                htable.get_labels(vals, uniques, 0, -1)

        uniques.to_array()  # should not raise here
        assert tmp.shape == oldshape

    @pytest.mark.parametrize(
        "hashtable",
        [
            ht.PyObjectHashTable,
            ht.StringHashTable,
            ht.Float64HashTable,
            ht.Int64HashTable,
            ht.Int32HashTable,
            ht.UInt64HashTable,
        ],
    )
    def test_hashtable_large_sizehint(self, hashtable):
        # GH#22729 smoketest for not raising when passing a large size_hint
        size_hint = np.iinfo(np.uint32).max + 1
        hashtable(size_hint=size_hint)


class TestPyObjectHashTableWithNans:
    def test_nan_float(self):
        nan1 = float("nan")
        nan2 = float("nan")
        assert nan1 is not nan2
        table = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42

    def test_nan_complex_both(self):
        nan1 = complex(float("nan"), float("nan"))
        nan2 = complex(float("nan"), float("nan"))
        assert nan1 is not nan2
        table = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42

    def test_nan_complex_real(self):
        nan1 = complex(float("nan"), 1)
        nan2 = complex(float("nan"), 1)
        other = complex(float("nan"), 2)
        assert nan1 is not nan2
        table = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42
        with pytest.raises(KeyError, match=None) as error:
            table.get_item(other)
        assert str(error.value) == str(other)

    def test_nan_complex_imag(self):
        nan1 = complex(1, float("nan"))
        nan2 = complex(1, float("nan"))
        other = complex(2, float("nan"))
        assert nan1 is not nan2
        table = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42
        with pytest.raises(KeyError, match=None) as error:
            table.get_item(other)
        assert str(error.value) == str(other)

    def test_nan_in_tuple(self):
        nan1 = (float("nan"),)
        nan2 = (float("nan"),)
        assert nan1[0] is not nan2[0]
        table = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42

    def test_nan_in_nested_tuple(self):
        nan1 = (1, (2, (float("nan"),)))
        nan2 = (1, (2, (float("nan"),)))
        other = (1, 2)
        table = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42
        with pytest.raises(KeyError, match=None) as error:
            table.get_item(other)
        assert str(error.value) == str(other)


def test_hash_equal_tuple_with_nans():
    a = (float("nan"), (float("nan"), float("nan")))
    b = (float("nan"), (float("nan"), float("nan")))
    assert ht.object_hash(a) == ht.object_hash(b)
    assert ht.objects_are_equal(a, b)


def test_get_labels_groupby_for_Int64(writable):
    table = ht.Int64HashTable()
    vals = np.array([1, 2, -1, 2, 1, -1], dtype=np.int64)
    vals.flags.writeable = writable
    arr, unique = table.get_labels_groupby(vals)
    expected_arr = np.array([0, 1, -1, 1, 0, -1], dtype=np.intp)
    expected_unique = np.array([1, 2], dtype=np.int64)
    tm.assert_numpy_array_equal(arr, expected_arr)
    tm.assert_numpy_array_equal(unique, expected_unique)


def test_tracemalloc_works_for_StringHashTable():
    N = 1000
    keys = np.arange(N).astype(np.str_).astype(np.object_)
    with activated_tracemalloc():
        table = ht.StringHashTable()
        table.map_locations(keys)
        used = get_allocated_khash_memory()
        my_size = table.sizeof()
        assert used == my_size
        del table
        assert get_allocated_khash_memory() == 0


def test_tracemalloc_for_empty_StringHashTable():
    with activated_tracemalloc():
        table = ht.StringHashTable()
        used = get_allocated_khash_memory()
        my_size = table.sizeof()
        assert used == my_size
        del table
        assert get_allocated_khash_memory() == 0


@pytest.mark.parametrize("N", range(1, 110))
def test_no_reallocation_StringHashTable(N):
    keys = np.arange(N).astype(np.str_).astype(np.object_)
    preallocated_table = ht.StringHashTable(N)
    n_buckets_start = preallocated_table.get_state()["n_buckets"]
    preallocated_table.map_locations(keys)
    n_buckets_end = preallocated_table.get_state()["n_buckets"]
    # original number of buckets was enough:
    assert n_buckets_start == n_buckets_end
    # check with clean table (not too much preallocated)
    clean_table = ht.StringHashTable()
    clean_table.map_locations(keys)
    assert n_buckets_start == clean_table.get_state()["n_buckets"]


@pytest.mark.parametrize(
    "table_type, dtype",
    [
        (ht.Float64HashTable, np.float64),
        (ht.Float32HashTable, np.float32),
        (ht.Complex128HashTable, np.complex128),
        (ht.Complex64HashTable, np.complex64),
    ],
)
class TestHashTableWithNans:
    def test_get_set_contains_len(self, table_type, dtype):
        index = float("nan")
        table = table_type()
        assert index not in table

        table.set_item(index, 42)
        assert len(table) == 1
        assert index in table
        assert table.get_item(index) == 42

        table.set_item(index, 41)
        assert len(table) == 1
        assert index in table
        assert table.get_item(index) == 41

    def test_map_locations(self, table_type, dtype):
        N = 10
        table = table_type()
        keys = np.full(N, np.nan, dtype=dtype)
        table.map_locations(keys)
        assert len(table) == 1
        assert table.get_item(np.nan) == N - 1

    def test_unique(self, table_type, dtype):
        N = 1020
        table = table_type()
        keys = np.full(N, np.nan, dtype=dtype)
        unique = table.unique(keys)
        assert np.all(np.isnan(unique)) and len(unique) == 1


def test_unique_for_nan_objects_floats():
    table = ht.PyObjectHashTable()
    keys = np.array([float("nan") for i in range(50)], dtype=np.object_)
    unique = table.unique(keys)
    assert len(unique) == 1


def test_unique_for_nan_objects_complex():
    table = ht.PyObjectHashTable()
    keys = np.array([complex(float("nan"), 1.0) for i in range(50)], dtype=np.object_)
    unique = table.unique(keys)
    assert len(unique) == 1


def test_unique_for_nan_objects_tuple():
    table = ht.PyObjectHashTable()
    keys = np.array(
        [1] + [(1.0, (float("nan"), 1.0)) for i in range(50)], dtype=np.object_
    )
    unique = table.unique(keys)
    assert len(unique) == 2


@pytest.mark.parametrize(
    "dtype",
    [
        np.object_,
        np.complex128,
        np.int64,
        np.uint64,
        np.float64,
        np.complex64,
        np.int32,
        np.uint32,
        np.float32,
        np.int16,
        np.uint16,
        np.int8,
        np.uint8,
        np.intp,
    ],
)
class TestHelpFunctions:
    def test_value_count(self, dtype, writable):
        N = 43
        expected = (np.arange(N) + N).astype(dtype)
        values = np.repeat(expected, 5)
        values.flags.writeable = writable
        keys, counts, _ = ht.value_count(values, False)
        tm.assert_numpy_array_equal(np.sort(keys), expected)
        assert np.all(counts == 5)

    def test_value_count_mask(self, dtype):
        if dtype == np.object_:
            pytest.skip("mask not implemented for object dtype")
        values = np.array([1] * 5, dtype=dtype)
        mask = np.zeros((5,), dtype=np.bool_)
        mask[1] = True
        mask[4] = True
        keys, counts, na_counter = ht.value_count(values, False, mask=mask)
        assert len(keys) == 2
        assert na_counter == 2

    def test_value_count_stable(self, dtype, writable):
        # GH12679
        values = np.array([2, 1, 5, 22, 3, -1, 8]).astype(dtype)
        values.flags.writeable = writable
        keys, counts, _ = ht.value_count(values, False)
        tm.assert_numpy_array_equal(keys, values)
        assert np.all(counts == 1)

    def test_duplicated_first(self, dtype, writable):
        N = 100
        values = np.repeat(np.arange(N).astype(dtype), 5)
        values.flags.writeable = writable
        result = ht.duplicated(values)
        expected = np.ones_like(values, dtype=np.bool_)
        expected[::5] = False
        tm.assert_numpy_array_equal(result, expected)

    def test_ismember_yes(self, dtype, writable):
        N = 127
        arr = np.arange(N).astype(dtype)
        values = np.arange(N).astype(dtype)
        arr.flags.writeable = writable
        values.flags.writeable = writable
        result = ht.ismember(arr, values)
        expected = np.ones_like(values, dtype=np.bool_)
        tm.assert_numpy_array_equal(result, expected)

    def test_ismember_no(self, dtype):
        N = 17
        arr = np.arange(N).astype(dtype)
        values = (np.arange(N) + N).astype(dtype)
        result = ht.ismember(arr, values)
        expected = np.zeros_like(values, dtype=np.bool_)
        tm.assert_numpy_array_equal(result, expected)

    def test_mode(self, dtype, writable):
        if dtype in (np.int8, np.uint8):
            N = 53
        else:
            N = 11111
        values = np.repeat(np.arange(N).astype(dtype), 5)
        values[0] = 42
        values.flags.writeable = writable
        result = ht.mode(values, False)[0]
        assert result == 42

    def test_mode_stable(self, dtype, writable):
        values = np.array([2, 1, 5, 22, 3, -1, 8]).astype(dtype)
        values.flags.writeable = writable
        keys = ht.mode(values, False)[0]
        tm.assert_numpy_array_equal(keys, values)


def test_modes_with_nans():
    # GH42688, nans aren't mangled
    nulls = [pd.NA, np.nan, pd.NaT, None]
    values = np.array([True] + nulls * 2, dtype=np.object_)
    modes = ht.mode(values, False)[0]
    assert modes.size == len(nulls)


def test_unique_label_indices_intp(writable):
    keys = np.array([1, 2, 2, 2, 1, 3], dtype=np.intp)
    keys.flags.writeable = writable
    result = ht.unique_label_indices(keys)
    expected = np.array([0, 1, 5], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)


def test_unique_label_indices():
    a = np.random.default_rng(2).integers(1, 1 << 10, 1 << 15).astype(np.intp)

    left = ht.unique_label_indices(a)
    right = np.unique(a, return_index=True)[1]

    tm.assert_numpy_array_equal(left, right, check_dtype=False)

    a[np.random.default_rng(2).choice(len(a), 10)] = -1
    left = ht.unique_label_indices(a)
    right = np.unique(a, return_index=True)[1][1:]
    tm.assert_numpy_array_equal(left, right, check_dtype=False)


@pytest.mark.parametrize(
    "dtype",
    [
        np.float64,
        np.float32,
        np.complex128,
        np.complex64,
    ],
)
class TestHelpFunctionsWithNans:
    def test_value_count(self, dtype):
        values = np.array([np.nan, np.nan, np.nan], dtype=dtype)
        keys, counts, _ = ht.value_count(values, True)
        assert len(keys) == 0
        keys, counts, _ = ht.value_count(values, False)
        assert len(keys) == 1 and np.all(np.isnan(keys))
        assert counts[0] == 3

    def test_duplicated_first(self, dtype):
        values = np.array([np.nan, np.nan, np.nan], dtype=dtype)
        result = ht.duplicated(values)
        expected = np.array([False, True, True])
        tm.assert_numpy_array_equal(result, expected)

    def test_ismember_yes(self, dtype):
        arr = np.array([np.nan, np.nan, np.nan], dtype=dtype)
        values = np.array([np.nan, np.nan], dtype=dtype)
        result = ht.ismember(arr, values)
        expected = np.array([True, True, True], dtype=np.bool_)
        tm.assert_numpy_array_equal(result, expected)

    def test_ismember_no(self, dtype):
        arr = np.array([np.nan, np.nan, np.nan], dtype=dtype)
        values = np.array([1], dtype=dtype)
        result = ht.ismember(arr, values)
        expected = np.array([False, False, False], dtype=np.bool_)
        tm.assert_numpy_array_equal(result, expected)

    def test_mode(self, dtype):
        values = np.array([42, np.nan, np.nan, np.nan], dtype=dtype)
        assert ht.mode(values, True)[0] == 42
        assert np.isnan(ht.mode(values, False)[0])


def test_ismember_tuple_with_nans():
    # GH-41836
    values = [("a", float("nan")), ("b", 1)]
    comps = [("a", float("nan"))]

    msg = "isin with argument that is not not a Series"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = isin(values, comps)
    expected = np.array([True, False], dtype=np.bool_)
    tm.assert_numpy_array_equal(result, expected)


def test_float_complex_int_are_equal_as_objects():
    values = ["a", 5, 5.0, 5.0 + 0j]
    comps = list(range(129))
    result = isin(np.array(values, dtype=object), np.asarray(comps))
    expected = np.array([False, True, True, True], dtype=np.bool_)
    tm.assert_numpy_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\tests\test_sparse.py ===
from sympy.core.numbers import (Float, I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.polys.polytools import PurePoly
from sympy.matrices import \
    Matrix, MutableSparseMatrix, ImmutableSparseMatrix, SparseMatrix, eye, \
    ones, zeros, ShapeError, NonSquareMatrixError
from sympy.testing.pytest import raises


def test_sparse_creation():
    a = SparseMatrix(2, 2, {(0, 0): [[1, 2], [3, 4]]})
    assert a == SparseMatrix([[1, 2], [3, 4]])
    a = SparseMatrix(2, 2, {(0, 0): [[1, 2]]})
    assert a == SparseMatrix([[1, 2], [0, 0]])
    a = SparseMatrix(2, 2, {(0, 0): [1, 2]})
    assert a == SparseMatrix([[1, 0], [2, 0]])


def test_sparse_matrix():
    def sparse_eye(n):
        return SparseMatrix.eye(n)

    def sparse_zeros(n):
        return SparseMatrix.zeros(n)

    # creation args
    raises(TypeError, lambda: SparseMatrix(1, 2))

    a = SparseMatrix((
        (1, 0),
        (0, 1)
    ))
    assert SparseMatrix(a) == a

    from sympy.matrices import MutableDenseMatrix
    a = MutableSparseMatrix([])
    b = MutableDenseMatrix([1, 2])
    assert a.row_join(b) == b
    assert a.col_join(b) == b
    assert type(a.row_join(b)) == type(a)
    assert type(a.col_join(b)) == type(a)

    # make sure 0 x n matrices get stacked correctly
    sparse_matrices = [SparseMatrix.zeros(0, n) for n in range(4)]
    assert SparseMatrix.hstack(*sparse_matrices) == Matrix(0, 6, [])
    sparse_matrices = [SparseMatrix.zeros(n, 0) for n in range(4)]
    assert SparseMatrix.vstack(*sparse_matrices) == Matrix(6, 0, [])

    # test element assignment
    a = SparseMatrix((
        (1, 0),
        (0, 1)
    ))

    a[3] = 4
    assert a[1, 1] == 4
    a[3] = 1

    a[0, 0] = 2
    assert a == SparseMatrix((
        (2, 0),
        (0, 1)
    ))
    a[1, 0] = 5
    assert a == SparseMatrix((
        (2, 0),
        (5, 1)
    ))
    a[1, 1] = 0
    assert a == SparseMatrix((
        (2, 0),
        (5, 0)
    ))
    assert a.todok() == {(0, 0): 2, (1, 0): 5}

    # test_multiplication
    a = SparseMatrix((
        (1, 2),
        (3, 1),
        (0, 6),
    ))

    b = SparseMatrix((
        (1, 2),
        (3, 0),
    ))

    c = a*b
    assert c[0, 0] == 7
    assert c[0, 1] == 2
    assert c[1, 0] == 6
    assert c[1, 1] == 6
    assert c[2, 0] == 18
    assert c[2, 1] == 0

    try:
        eval('c = a @ b')
    except SyntaxError:
        pass
    else:
        assert c[0, 0] == 7
        assert c[0, 1] == 2
        assert c[1, 0] == 6
        assert c[1, 1] == 6
        assert c[2, 0] == 18
        assert c[2, 1] == 0

    x = Symbol("x")

    c = b * Symbol("x")
    assert isinstance(c, SparseMatrix)
    assert c[0, 0] == x
    assert c[0, 1] == 2*x
    assert c[1, 0] == 3*x
    assert c[1, 1] == 0

    c = 5 * b
    assert isinstance(c, SparseMatrix)
    assert c[0, 0] == 5
    assert c[0, 1] == 2*5
    assert c[1, 0] == 3*5
    assert c[1, 1] == 0

    #test_power
    A = SparseMatrix([[2, 3], [4, 5]])
    assert (A**5)[:] == [6140, 8097, 10796, 14237]
    A = SparseMatrix([[2, 1, 3], [4, 2, 4], [6, 12, 1]])
    assert (A**3)[:] == [290, 262, 251, 448, 440, 368, 702, 954, 433]

    # test_creation
    x = Symbol("x")
    a = SparseMatrix([[x, 0], [0, 0]])
    m = a
    assert m.cols == m.rows
    assert m.cols == 2
    assert m[:] == [x, 0, 0, 0]
    b = SparseMatrix(2, 2, [x, 0, 0, 0])
    m = b
    assert m.cols == m.rows
    assert m.cols == 2
    assert m[:] == [x, 0, 0, 0]

    assert a == b
    S = sparse_eye(3)
    S.row_del(1)
    assert S == SparseMatrix([
                             [1, 0, 0],
    [0, 0, 1]])
    S = sparse_eye(3)
    S.col_del(1)
    assert S == SparseMatrix([
                             [1, 0],
    [0, 0],
    [0, 1]])
    S = SparseMatrix.eye(3)
    S[2, 1] = 2
    S.col_swap(1, 0)
    assert S == SparseMatrix([
        [0, 1, 0],
        [1, 0, 0],
        [2, 0, 1]])
    S.row_swap(0, 1)
    assert S == SparseMatrix([
        [1, 0, 0],
        [0, 1, 0],
        [2, 0, 1]])

    a = SparseMatrix(1, 2, [1, 2])
    b = a.copy()
    c = a.copy()
    assert a[0] == 1
    a.row_del(0)
    assert a == SparseMatrix(0, 2, [])
    b.col_del(1)
    assert b == SparseMatrix(1, 1, [1])

    assert SparseMatrix([[1, 2, 3], [1, 2], [1]]) == Matrix([
        [1, 2, 3],
        [1, 2, 0],
        [1, 0, 0]])
    assert SparseMatrix(4, 4, {(1, 1): sparse_eye(2)}) == Matrix([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0]])
    raises(ValueError, lambda: SparseMatrix(1, 1, {(1, 1): 1}))
    assert SparseMatrix(1, 2, [1, 2]).tolist() == [[1, 2]]
    assert SparseMatrix(2, 2, [1, [2, 3]]).tolist() == [[1, 0], [2, 3]]
    raises(ValueError, lambda: SparseMatrix(2, 2, [1]))
    raises(ValueError, lambda: SparseMatrix(1, 1, [[1, 2]]))
    assert SparseMatrix([.1]).has(Float)
    # autosizing
    assert SparseMatrix(None, {(0, 1): 0}).shape == (0, 0)
    assert SparseMatrix(None, {(0, 1): 1}).shape == (1, 2)
    assert SparseMatrix(None, None, {(0, 1): 1}).shape == (1, 2)
    raises(ValueError, lambda: SparseMatrix(None, 1, [[1, 2]]))
    raises(ValueError, lambda: SparseMatrix(1, None, [[1, 2]]))
    raises(ValueError, lambda: SparseMatrix(3, 3, {(0, 0): ones(2), (1, 1): 2}))

    # test_determinant
    x, y = Symbol('x'), Symbol('y')

    assert SparseMatrix(1, 1, [0]).det() == 0

    assert SparseMatrix([[1]]).det() == 1

    assert SparseMatrix(((-3, 2), (8, -5))).det() == -1

    assert SparseMatrix(((x, 1), (y, 2*y))).det() == 2*x*y - y

    assert SparseMatrix(( (1, 1, 1),
                          (1, 2, 3),
                          (1, 3, 6) )).det() == 1

    assert SparseMatrix(( ( 3, -2,  0, 5),
                          (-2,  1, -2, 2),
                          ( 0, -2,  5, 0),
                          ( 5,  0,  3, 4) )).det() == -289

    assert SparseMatrix(( ( 1,  2,  3,  4),
                          ( 5,  6,  7,  8),
                          ( 9, 10, 11, 12),
                          (13, 14, 15, 16) )).det() == 0

    assert SparseMatrix(( (3, 2, 0, 0, 0),
                          (0, 3, 2, 0, 0),
                          (0, 0, 3, 2, 0),
                          (0, 0, 0, 3, 2),
                          (2, 0, 0, 0, 3) )).det() == 275

    assert SparseMatrix(( (1, 0,  1,  2, 12),
                          (2, 0,  1,  1,  4),
                          (2, 1,  1, -1,  3),
                          (3, 2, -1,  1,  8),
                          (1, 1,  1,  0,  6) )).det() == -55

    assert SparseMatrix(( (-5,  2,  3,  4,  5),
                          ( 1, -4,  3,  4,  5),
                          ( 1,  2, -3,  4,  5),
                          ( 1,  2,  3, -2,  5),
                          ( 1,  2,  3,  4, -1) )).det() == 11664

    assert SparseMatrix(( ( 3,  0,  0, 0),
                          (-2,  1,  0, 0),
                          ( 0, -2,  5, 0),
                          ( 5,  0,  3, 4) )).det() == 60

    assert SparseMatrix(( ( 1,  0,  0,  0),
                          ( 5,  0,  0,  0),
                          ( 9, 10, 11, 0),
                          (13, 14, 15, 16) )).det() == 0

    assert SparseMatrix(( (3, 2, 0, 0, 0),
                          (0, 3, 2, 0, 0),
                          (0, 0, 3, 2, 0),
                          (0, 0, 0, 3, 2),
                          (0, 0, 0, 0, 3) )).det() == 243

    assert SparseMatrix(( ( 2,  7, -1, 3, 2),
                          ( 0,  0,  1, 0, 1),
                          (-2,  0,  7, 0, 2),
                          (-3, -2,  4, 5, 3),
                          ( 1,  0,  0, 0, 1) )).det() == 123

    # test_slicing
    m0 = sparse_eye(4)
    assert m0[:3, :3] == sparse_eye(3)
    assert m0[2:4, 0:2] == sparse_zeros(2)

    m1 = SparseMatrix(3, 3, lambda i, j: i + j)
    assert m1[0, :] == SparseMatrix(1, 3, (0, 1, 2))
    assert m1[1:3, 1] == SparseMatrix(2, 1, (2, 3))

    m2 = SparseMatrix(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
    assert m2[:, -1] == SparseMatrix(4, 1, [3, 7, 11, 15])
    assert m2[-2:, :] == SparseMatrix([[8, 9, 10, 11], [12, 13, 14, 15]])

    assert SparseMatrix([[1, 2], [3, 4]])[[1], [1]] == Matrix([[4]])

    # test_submatrix_assignment
    m = sparse_zeros(4)
    m[2:4, 2:4] = sparse_eye(2)
    assert m == SparseMatrix([(0, 0, 0, 0),
                              (0, 0, 0, 0),
                              (0, 0, 1, 0),
                              (0, 0, 0, 1)])
    assert len(m.todok()) == 2
    m[:2, :2] = sparse_eye(2)
    assert m == sparse_eye(4)
    m[:, 0] = SparseMatrix(4, 1, (1, 2, 3, 4))
    assert m == SparseMatrix([(1, 0, 0, 0),
                              (2, 1, 0, 0),
                              (3, 0, 1, 0),
                              (4, 0, 0, 1)])
    m[:, :] = sparse_zeros(4)
    assert m == sparse_zeros(4)
    m[:, :] = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16))
    assert m == SparseMatrix((( 1,  2,  3,  4),
                              ( 5,  6,  7,  8),
                              ( 9, 10, 11, 12),
                              (13, 14, 15, 16)))
    m[:2, 0] = [0, 0]
    assert m == SparseMatrix((( 0,  2,  3,  4),
                              ( 0,  6,  7,  8),
                              ( 9, 10, 11, 12),
                              (13, 14, 15, 16)))

    # test_reshape
    m0 = sparse_eye(3)
    assert m0.reshape(1, 9) == SparseMatrix(1, 9, (1, 0, 0, 0, 1, 0, 0, 0, 1))
    m1 = SparseMatrix(3, 4, lambda i, j: i + j)
    assert m1.reshape(4, 3) == \
        SparseMatrix([(0, 1, 2), (3, 1, 2), (3, 4, 2), (3, 4, 5)])
    assert m1.reshape(2, 6) == \
        SparseMatrix([(0, 1, 2, 3, 1, 2), (3, 4, 2, 3, 4, 5)])

    # test_applyfunc
    m0 = sparse_eye(3)
    assert m0.applyfunc(lambda x: 2*x) == sparse_eye(3)*2
    assert m0.applyfunc(lambda x: 0 ) == sparse_zeros(3)

    # test__eval_Abs
    assert abs(SparseMatrix(((x, 1), (y, 2*y)))) == SparseMatrix(((Abs(x), 1), (Abs(y), 2*Abs(y))))

    # test_LUdecomp
    testmat = SparseMatrix([[ 0, 2, 5, 3],
                            [ 3, 3, 7, 4],
                            [ 8, 4, 0, 2],
                            [-2, 6, 3, 4]])
    L, U, p = testmat.LUdecomposition()
    assert L.is_lower
    assert U.is_upper
    assert (L*U).permute_rows(p, 'backward') - testmat == sparse_zeros(4)

    testmat = SparseMatrix([[ 6, -2, 7, 4],
                            [ 0,  3, 6, 7],
                            [ 1, -2, 7, 4],
                            [-9,  2, 6, 3]])
    L, U, p = testmat.LUdecomposition()
    assert L.is_lower
    assert U.is_upper
    assert (L*U).permute_rows(p, 'backward') - testmat == sparse_zeros(4)

    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    M = Matrix(((1, x, 1), (2, y, 0), (y, 0, z)))
    L, U, p = M.LUdecomposition()
    assert L.is_lower
    assert U.is_upper
    assert (L*U).permute_rows(p, 'backward') - M == sparse_zeros(3)

    # test_LUsolve
    A = SparseMatrix([[2, 3, 5],
                      [3, 6, 2],
                      [8, 3, 6]])
    x = SparseMatrix(3, 1, [3, 7, 5])
    b = A*x
    soln = A.LUsolve(b)
    assert soln == x
    A = SparseMatrix([[0, -1, 2],
                      [5, 10, 7],
                      [8,  3, 4]])
    x = SparseMatrix(3, 1, [-1, 2, 5])
    b = A*x
    soln = A.LUsolve(b)
    assert soln == x

    # test_inverse
    A = sparse_eye(4)
    assert A.inv() == sparse_eye(4)
    assert A.inv(method="CH") == sparse_eye(4)
    assert A.inv(method="LDL") == sparse_eye(4)

    A = SparseMatrix([[2, 3, 5],
                      [3, 6, 2],
                      [7, 2, 6]])
    Ainv = SparseMatrix(Matrix(A).inv())
    assert A*Ainv == sparse_eye(3)
    assert A.inv(method="CH") == Ainv
    assert A.inv(method="LDL") == Ainv

    A = SparseMatrix([[2, 3, 5],
                      [3, 6, 2],
                      [5, 2, 6]])
    Ainv = SparseMatrix(Matrix(A).inv())
    assert A*Ainv == sparse_eye(3)
    assert A.inv(method="CH") == Ainv
    assert A.inv(method="LDL") == Ainv

    # test_cross
    v1 = Matrix(1, 3, [1, 2, 3])
    v2 = Matrix(1, 3, [3, 4, 5])
    assert v1.cross(v2) == Matrix(1, 3, [-2, 4, -2])
    assert v1.norm(2)**2 == 14

    # conjugate
    a = SparseMatrix(((1, 2 + I), (3, 4)))
    assert a.C == SparseMatrix([
        [1, 2 - I],
        [3,     4]
    ])

    # mul
    assert a*Matrix(2, 2, [1, 0, 0, 1]) == a
    assert a + Matrix(2, 2, [1, 1, 1, 1]) == SparseMatrix([
        [2, 3 + I],
        [4,     5]
    ])

    # col join
    assert a.col_join(sparse_eye(2)) == SparseMatrix([
        [1, 2 + I],
        [3,     4],
        [1,     0],
        [0,     1]
    ])

    # row insert
    assert a.row_insert(2, sparse_eye(2)) == SparseMatrix([
        [1, 2 + I],
        [3,     4],
        [1,     0],
        [0,     1]
    ])

    # col insert
    assert a.col_insert(2, SparseMatrix.zeros(2, 1)) == SparseMatrix([
        [1, 2 + I, 0],
        [3,     4, 0],
    ])

    # symmetric
    assert not a.is_symmetric(simplify=False)

    # col op
    M = SparseMatrix.eye(3)*2
    M[1, 0] = -1
    M.col_op(1, lambda v, i: v + 2*M[i, 0])
    assert M == SparseMatrix([
        [ 2, 4, 0],
        [-1, 0, 0],
        [ 0, 0, 2]
    ])

    # fill
    M = SparseMatrix.eye(3)
    M.fill(2)
    assert M == SparseMatrix([
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
    ])

    # test_cofactor
    assert sparse_eye(3) == sparse_eye(3).cofactor_matrix()
    test = SparseMatrix([[1, 3, 2], [2, 6, 3], [2, 3, 6]])
    assert test.cofactor_matrix() == \
        SparseMatrix([[27, -6, -6], [-12, 2, 3], [-3, 1, 0]])
    test = SparseMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert test.cofactor_matrix() == \
        SparseMatrix([[-3, 6, -3], [6, -12, 6], [-3, 6, -3]])

    # test_jacobian
    x = Symbol('x')
    y = Symbol('y')
    L = SparseMatrix(1, 2, [x**2*y, 2*y**2 + x*y])
    syms = [x, y]
    assert L.jacobian(syms) == Matrix([[2*x*y, x**2], [y, 4*y + x]])

    L = SparseMatrix(1, 2, [x, x**2*y**3])
    assert L.jacobian(syms) == SparseMatrix([[1, 0], [2*x*y**3, x**2*3*y**2]])

    # test_QR
    A = Matrix([[1, 2], [2, 3]])
    Q, S = A.QRdecomposition()
    R = Rational
    assert Q == Matrix([
        [  5**R(-1, 2),  (R(2)/5)*(R(1)/5)**R(-1, 2)],
        [2*5**R(-1, 2), (-R(1)/5)*(R(1)/5)**R(-1, 2)]])
    assert S == Matrix([
        [5**R(1, 2),     8*5**R(-1, 2)],
        [         0, (R(1)/5)**R(1, 2)]])
    assert Q*S == A
    assert Q.T * Q == sparse_eye(2)

    R = Rational
    # test nullspace
    # first test reduced row-ech form

    M = SparseMatrix([[5, 7, 2, 1],
               [1, 6, 2, -1]])
    out, tmp = M.rref()
    assert out == Matrix([[1, 0, -R(2)/23, R(13)/23],
                          [0, 1,  R(8)/23, R(-6)/23]])

    M = SparseMatrix([[ 1,  3, 0,  2,  6, 3, 1],
                      [-2, -6, 0, -2, -8, 3, 1],
                      [ 3,  9, 0,  0,  6, 6, 2],
                      [-1, -3, 0,  1,  0, 9, 3]])

    out, tmp = M.rref()
    assert out == Matrix([[1, 3, 0, 0, 2, 0, 0],
                          [0, 0, 0, 1, 2, 0, 0],
                          [0, 0, 0, 0, 0, 1, R(1)/3],
                          [0, 0, 0, 0, 0, 0, 0]])
    # now check the vectors
    basis = M.nullspace()
    assert basis[0] == Matrix([-3, 1, 0, 0, 0, 0, 0])
    assert basis[1] == Matrix([0, 0, 1, 0, 0, 0, 0])
    assert basis[2] == Matrix([-2, 0, 0, -2, 1, 0, 0])
    assert basis[3] == Matrix([0, 0, 0, 0, 0, R(-1)/3, 1])

    # test eigen
    x = Symbol('x')
    y = Symbol('y')
    sparse_eye3 = sparse_eye(3)
    assert sparse_eye3.charpoly(x) == PurePoly((x - 1)**3)
    assert sparse_eye3.charpoly(y) == PurePoly((y - 1)**3)

    # test values
    M = Matrix([( 0, 1, -1),
                ( 1, 1,  0),
                (-1, 0,  1)])
    vals = M.eigenvals()
    assert sorted(vals.keys()) == [-1, 1, 2]

    R = Rational
    M = Matrix([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
    assert M.eigenvects() == [(1, 3, [
        Matrix([1, 0, 0]),
        Matrix([0, 1, 0]),
        Matrix([0, 0, 1])])]
    M = Matrix([[5, 0, 2],
                [3, 2, 0],
                [0, 0, 1]])
    assert M.eigenvects() == [(1, 1, [Matrix([R(-1)/2, R(3)/2, 1])]),
                              (2, 1, [Matrix([0, 1, 0])]),
                              (5, 1, [Matrix([1, 1, 0])])]

    assert M.zeros(3, 5) == SparseMatrix(3, 5, {})
    A =  SparseMatrix(10, 10, {(0, 0): 18, (0, 9): 12, (1, 4): 18, (2, 7): 16, (3, 9): 12, (4, 2): 19, (5, 7): 16, (6, 2): 12, (9, 7): 18})
    assert A.row_list() == [(0, 0, 18), (0, 9, 12), (1, 4, 18), (2, 7, 16), (3, 9, 12), (4, 2, 19), (5, 7, 16), (6, 2, 12), (9, 7, 18)]
    assert A.col_list() == [(0, 0, 18), (4, 2, 19), (6, 2, 12), (1, 4, 18), (2, 7, 16), (5, 7, 16), (9, 7, 18), (0, 9, 12), (3, 9, 12)]
    assert SparseMatrix.eye(2).nnz() == 2


def test_scalar_multiply():
    assert SparseMatrix([[1, 2]]).scalar_multiply(3) == SparseMatrix([[3, 6]])


def test_transpose():
    assert SparseMatrix(((1, 2), (3, 4))).transpose() == \
        SparseMatrix(((1, 3), (2, 4)))


def test_trace():
    assert SparseMatrix(((1, 2), (3, 4))).trace() == 5
    assert SparseMatrix(((0, 0), (0, 4))).trace() == 4


def test_CL_RL():
    assert SparseMatrix(((1, 2), (3, 4))).row_list() == \
        [(0, 0, 1), (0, 1, 2), (1, 0, 3), (1, 1, 4)]
    assert SparseMatrix(((1, 2), (3, 4))).col_list() == \
        [(0, 0, 1), (1, 0, 3), (0, 1, 2), (1, 1, 4)]


def test_add():
    assert SparseMatrix(((1, 0), (0, 1))) + SparseMatrix(((0, 1), (1, 0))) == \
        SparseMatrix(((1, 1), (1, 1)))
    a = SparseMatrix(100, 100, lambda i, j: int(j != 0 and i % j == 0))
    b = SparseMatrix(100, 100, lambda i, j: int(i != 0 and j % i == 0))
    assert (len(a.todok()) + len(b.todok()) - len((a + b).todok()) > 0)


def test_errors():
    raises(ValueError, lambda: SparseMatrix(1.4, 2, lambda i, j: 0))
    raises(TypeError, lambda: SparseMatrix([1, 2, 3], [1, 2]))
    raises(ValueError, lambda: SparseMatrix([[1, 2], [3, 4]])[(1, 2, 3)])
    raises(IndexError, lambda: SparseMatrix([[1, 2], [3, 4]])[5])
    raises(ValueError, lambda: SparseMatrix([[1, 2], [3, 4]])[1, 2, 3])
    raises(TypeError,
        lambda: SparseMatrix([[1, 2], [3, 4]]).copyin_list([0, 1], set()))
    raises(
        IndexError, lambda: SparseMatrix([[1, 2], [3, 4]])[1, 2])
    raises(TypeError, lambda: SparseMatrix([1, 2, 3]).cross(1))
    raises(IndexError, lambda: SparseMatrix(1, 2, [1, 2])[3])
    raises(ShapeError,
        lambda: SparseMatrix(1, 2, [1, 2]) + SparseMatrix(2, 1, [2, 1]))


def test_len():
    assert not SparseMatrix()
    assert SparseMatrix() == SparseMatrix([])
    assert SparseMatrix() == SparseMatrix([[]])


def test_sparse_zeros_sparse_eye():
    assert SparseMatrix.eye(3) == eye(3, cls=SparseMatrix)
    assert len(SparseMatrix.eye(3).todok()) == 3
    assert SparseMatrix.zeros(3) == zeros(3, cls=SparseMatrix)
    assert len(SparseMatrix.zeros(3).todok()) == 0


def test_copyin():
    s = SparseMatrix(3, 3, {})
    s[1, 0] = 1
    assert s[:, 0] == SparseMatrix(Matrix([0, 1, 0]))
    assert s[3] == 1
    assert s[3: 4] == [1]
    s[1, 1] = 42
    assert s[1, 1] == 42
    assert s[1, 1:] == SparseMatrix([[42, 0]])
    s[1, 1:] = Matrix([[5, 6]])
    assert s[1, :] == SparseMatrix([[1, 5, 6]])
    s[1, 1:] = [[42, 43]]
    assert s[1, :] == SparseMatrix([[1, 42, 43]])
    s[0, 0] = 17
    assert s[:, :1] == SparseMatrix([17, 1, 0])
    s[0, 0] = [1, 1, 1]
    assert s[:, 0] == SparseMatrix([1, 1, 1])
    s[0, 0] = Matrix([1, 1, 1])
    assert s[:, 0] == SparseMatrix([1, 1, 1])
    s[0, 0] = SparseMatrix([1, 1, 1])
    assert s[:, 0] == SparseMatrix([1, 1, 1])


def test_sparse_solve():
    A = SparseMatrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))
    assert A.cholesky() == Matrix([
        [ 5, 0, 0],
        [ 3, 3, 0],
        [-1, 1, 3]])
    assert A.cholesky() * A.cholesky().T == Matrix([
        [25, 15, -5],
        [15, 18, 0],
        [-5, 0, 11]])

    A = SparseMatrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))
    L, D = A.LDLdecomposition()
    assert 15*L == Matrix([
        [15, 0, 0],
        [ 9, 15, 0],
        [-3, 5, 15]])
    assert D == Matrix([
        [25, 0, 0],
        [ 0, 9, 0],
        [ 0, 0, 9]])
    assert L * D * L.T == A

    A = SparseMatrix(((3, 0, 2), (0, 0, 1), (1, 2, 0)))
    assert A.inv() * A == SparseMatrix(eye(3))

    A = SparseMatrix([
        [ 2, -1, 0],
        [-1, 2, -1],
        [ 0, 0, 2]])
    ans = SparseMatrix([
        [Rational(2, 3), Rational(1, 3), Rational(1, 6)],
        [Rational(1, 3), Rational(2, 3), Rational(1, 3)],
        [             0,              0,        S.Half]])
    assert A.inv(method='CH') == ans
    assert A.inv(method='LDL') == ans
    assert A * ans == SparseMatrix(eye(3))

    s = A.solve(A[:, 0], 'LDL')
    assert A*s == A[:, 0]
    s = A.solve(A[:, 0], 'CH')
    assert A*s == A[:, 0]
    A = A.col_join(A)
    s = A.solve_least_squares(A[:, 0], 'CH')
    assert A*s == A[:, 0]
    s = A.solve_least_squares(A[:, 0], 'LDL')
    assert A*s == A[:, 0]


def test_lower_triangular_solve():
    raises(NonSquareMatrixError, lambda:
        SparseMatrix([[1, 2]]).lower_triangular_solve(Matrix([[1, 2]])))
    raises(ShapeError, lambda:
        SparseMatrix([[1, 2], [0, 4]]).lower_triangular_solve(Matrix([1])))
    raises(ValueError, lambda:
        SparseMatrix([[1, 2], [3, 4]]).lower_triangular_solve(Matrix([[1, 2], [3, 4]])))

    a, b, c, d = symbols('a:d')
    u, v, w, x = symbols('u:x')

    A = SparseMatrix([[a, 0], [c, d]])
    B = MutableSparseMatrix([[u, v], [w, x]])
    C = ImmutableSparseMatrix([[u, v], [w, x]])

    sol = Matrix([[u/a, v/a], [(w - c*u/a)/d, (x - c*v/a)/d]])
    assert A.lower_triangular_solve(B) == sol
    assert A.lower_triangular_solve(C) == sol


def test_upper_triangular_solve():
    raises(NonSquareMatrixError, lambda:
        SparseMatrix([[1, 2]]).upper_triangular_solve(Matrix([[1, 2]])))
    raises(ShapeError, lambda:
        SparseMatrix([[1, 2], [0, 4]]).upper_triangular_solve(Matrix([1])))
    raises(TypeError, lambda:
        SparseMatrix([[1, 2], [3, 4]]).upper_triangular_solve(Matrix([[1, 2], [3, 4]])))

    a, b, c, d = symbols('a:d')
    u, v, w, x = symbols('u:x')

    A = SparseMatrix([[a, b], [0, d]])
    B = MutableSparseMatrix([[u, v], [w, x]])
    C = ImmutableSparseMatrix([[u, v], [w, x]])

    sol = Matrix([[(u - b*w/d)/a, (v - b*x/d)/a], [w/d, x/d]])
    assert A.upper_triangular_solve(B) == sol
    assert A.upper_triangular_solve(C) == sol


def test_diagonal_solve():
    a, d = symbols('a d')
    u, v, w, x = symbols('u:x')

    A = SparseMatrix([[a, 0], [0, d]])
    B = MutableSparseMatrix([[u, v], [w, x]])
    C = ImmutableSparseMatrix([[u, v], [w, x]])

    sol = Matrix([[u/a, v/a], [w/d, x/d]])
    assert A.diagonal_solve(B) == sol
    assert A.diagonal_solve(C) == sol


def test_hermitian():
    x = Symbol('x')
    a = SparseMatrix([[0, I], [-I, 0]])
    assert a.is_hermitian
    a = SparseMatrix([[1, I], [-I, 1]])
    assert a.is_hermitian
    a[0, 0] = 2*I
    assert a.is_hermitian is False
    a[0, 0] = x
    assert a.is_hermitian is None
    a[0, 1] = a[1, 0]*I
    assert a.is_hermitian is False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\tests\test_gamma_functions.py ===
from sympy.core.function import expand_func, Subs
from sympy.core import EulerGamma
from sympy.core.numbers import (I, Rational, nan, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.combinatorial.numbers import harmonic
from sympy.functions.elementary.complexes import (Abs, conjugate, im, re)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, atan)
from sympy.functions.special.error_functions import (Ei, erf, erfc)
from sympy.functions.special.gamma_functions import (digamma, gamma, loggamma, lowergamma, multigamma, polygamma, trigamma, uppergamma)
from sympy.functions.special.zeta_functions import zeta
from sympy.series.order import O

from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.testing.pytest import raises
from sympy.core.random import (test_derivative_numerically as td,
                                      random_complex_number as randcplx,
                                      verify_numerically as tn)

x = Symbol('x')
y = Symbol('y')
n = Symbol('n', integer=True)
w = Symbol('w', real=True)

def test_gamma():
    assert gamma(nan) is nan
    assert gamma(oo) is oo

    assert gamma(-100) is zoo
    assert gamma(0) is zoo
    assert gamma(-100.0) is zoo

    assert gamma(1) == 1
    assert gamma(2) == 1
    assert gamma(3) == 2

    assert gamma(102) == factorial(101)

    assert gamma(S.Half) == sqrt(pi)

    assert gamma(Rational(3, 2)) == sqrt(pi)*S.Half
    assert gamma(Rational(5, 2)) == sqrt(pi)*Rational(3, 4)
    assert gamma(Rational(7, 2)) == sqrt(pi)*Rational(15, 8)

    assert gamma(Rational(-1, 2)) == -2*sqrt(pi)
    assert gamma(Rational(-3, 2)) == sqrt(pi)*Rational(4, 3)
    assert gamma(Rational(-5, 2)) == sqrt(pi)*Rational(-8, 15)

    assert gamma(Rational(-15, 2)) == sqrt(pi)*Rational(256, 2027025)

    assert gamma(Rational(
        -11, 8)).expand(func=True) == Rational(64, 33)*gamma(Rational(5, 8))
    assert gamma(Rational(
        -10, 3)).expand(func=True) == Rational(81, 280)*gamma(Rational(2, 3))
    assert gamma(Rational(
        14, 3)).expand(func=True) == Rational(880, 81)*gamma(Rational(2, 3))
    assert gamma(Rational(
        17, 7)).expand(func=True) == Rational(30, 49)*gamma(Rational(3, 7))
    assert gamma(Rational(
        19, 8)).expand(func=True) == Rational(33, 64)*gamma(Rational(3, 8))

    assert gamma(x).diff(x) == gamma(x)*polygamma(0, x)

    assert gamma(x - 1).expand(func=True) == gamma(x)/(x - 1)
    assert gamma(x + 2).expand(func=True, mul=False) == x*(x + 1)*gamma(x)

    assert conjugate(gamma(x)) == gamma(conjugate(x))

    assert expand_func(gamma(x + Rational(3, 2))) == \
        (x + S.Half)*gamma(x + S.Half)

    assert expand_func(gamma(x - S.Half)) == \
        gamma(S.Half + x)/(x - S.Half)

    # Test a bug:
    assert expand_func(gamma(x + Rational(3, 4))) == gamma(x + Rational(3, 4))

    # XXX: Not sure about these tests. I can fix them by defining e.g.
    # exp_polar.is_integer but I'm not sure if that makes sense.
    assert gamma(3*exp_polar(I*pi)/4).is_nonnegative is False
    assert gamma(3*exp_polar(I*pi)/4).is_extended_nonpositive is True

    y = Symbol('y', nonpositive=True, integer=True)
    assert gamma(y).is_real == False
    y = Symbol('y', positive=True, noninteger=True)
    assert gamma(y).is_real == True

    assert gamma(-1.0, evaluate=False).is_real == False
    assert gamma(0, evaluate=False).is_real == False
    assert gamma(-2, evaluate=False).is_real == False


def test_gamma_rewrite():
    assert gamma(n).rewrite(factorial) == factorial(n - 1)


def test_gamma_series():
    assert gamma(x + 1).series(x, 0, 3) == \
        1 - EulerGamma*x + x**2*(EulerGamma**2/2 + pi**2/12) + O(x**3)
    assert gamma(x).series(x, -1, 3) == \
        -1/(x + 1) + EulerGamma - 1 + (x + 1)*(-1 - pi**2/12 - EulerGamma**2/2 + \
       EulerGamma) + (x + 1)**2*(-1 - pi**2/12 - EulerGamma**2/2 + EulerGamma**3/6 - \
       polygamma(2, 1)/6 + EulerGamma*pi**2/12 + EulerGamma) + O((x + 1)**3, (x, -1))


def tn_branch(s, func):
    from sympy.core.random import uniform
    c = uniform(1, 5)
    expr = func(s, c*exp_polar(I*pi)) - func(s, c*exp_polar(-I*pi))
    eps = 1e-15
    expr2 = func(s + eps, -c + eps*I) - func(s + eps, -c - eps*I)
    return abs(expr.n() - expr2.n()).n() < 1e-10


def test_lowergamma():
    from sympy.functions.special.error_functions import expint
    from sympy.functions.special.hyper import meijerg
    assert lowergamma(x, 0) == 0
    assert lowergamma(x, y).diff(y) == y**(x - 1)*exp(-y)
    assert td(lowergamma(randcplx(), y), y)
    assert td(lowergamma(x, randcplx()), x)
    assert lowergamma(x, y).diff(x) == \
        gamma(x)*digamma(x) - uppergamma(x, y)*log(y) \
        - meijerg([], [1, 1], [0, 0, x], [], y)

    assert lowergamma(S.Half, x) == sqrt(pi)*erf(sqrt(x))
    assert not lowergamma(S.Half - 3, x).has(lowergamma)
    assert not lowergamma(S.Half + 3, x).has(lowergamma)
    assert lowergamma(S.Half, x, evaluate=False).has(lowergamma)
    assert tn(lowergamma(S.Half + 3, x, evaluate=False),
              lowergamma(S.Half + 3, x), x)
    assert tn(lowergamma(S.Half - 3, x, evaluate=False),
              lowergamma(S.Half - 3, x), x)

    assert tn_branch(-3, lowergamma)
    assert tn_branch(-4, lowergamma)
    assert tn_branch(Rational(1, 3), lowergamma)
    assert tn_branch(pi, lowergamma)
    assert lowergamma(3, exp_polar(4*pi*I)*x) == lowergamma(3, x)
    assert lowergamma(y, exp_polar(5*pi*I)*x) == \
        exp(4*I*pi*y)*lowergamma(y, x*exp_polar(pi*I))
    assert lowergamma(-2, exp_polar(5*pi*I)*x) == \
        lowergamma(-2, x*exp_polar(I*pi)) + 2*pi*I

    assert conjugate(lowergamma(x, y)) == lowergamma(conjugate(x), conjugate(y))
    assert conjugate(lowergamma(x, 0)) == 0
    assert unchanged(conjugate, lowergamma(x, -oo))

    assert lowergamma(0, x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(S(1)/3, x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(1, x, evaluate=False)._eval_is_meromorphic(x, 0) == True
    assert lowergamma(x, x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(x + 1, x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(1/x, x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(0, x + 1)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(S(1)/3, x + 1)._eval_is_meromorphic(x, 0) == True
    assert lowergamma(1, x + 1, evaluate=False)._eval_is_meromorphic(x, 0) == True
    assert lowergamma(x, x + 1)._eval_is_meromorphic(x, 0) == True
    assert lowergamma(x + 1, x + 1)._eval_is_meromorphic(x, 0) == True
    assert lowergamma(1/x, x + 1)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(0, 1/x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(S(1)/3, 1/x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(1, 1/x, evaluate=False)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(x, 1/x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(x + 1, 1/x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(1/x, 1/x)._eval_is_meromorphic(x, 0) == False

    assert lowergamma(x, 2).series(x, oo, 3) == \
        2**x*(1 + 2/(x + 1))*exp(-2)/x + O(exp(x*log(2))/x**3, (x, oo))

    assert lowergamma(
        x, y).rewrite(expint) == -y**x*expint(-x + 1, y) + gamma(x)
    k = Symbol('k', integer=True)
    assert lowergamma(
        k, y).rewrite(expint) == -y**k*expint(-k + 1, y) + gamma(k)
    k = Symbol('k', integer=True, positive=False)
    assert lowergamma(k, y).rewrite(expint) == lowergamma(k, y)
    assert lowergamma(x, y).rewrite(uppergamma) == gamma(x) - uppergamma(x, y)

    assert lowergamma(70, 6) == factorial(69) - 69035724522603011058660187038367026272747334489677105069435923032634389419656200387949342530805432320 * exp(-6)
    assert (lowergamma(S(77) / 2, 6) - lowergamma(S(77) / 2, 6, evaluate=False)).evalf() < 1e-16
    assert (lowergamma(-S(77) / 2, 6) - lowergamma(-S(77) / 2, 6, evaluate=False)).evalf() < 1e-16


def test_uppergamma():
    from sympy.functions.special.error_functions import expint
    from sympy.functions.special.hyper import meijerg
    assert uppergamma(4, 0) == 6
    assert uppergamma(x, y).diff(y) == -y**(x - 1)*exp(-y)
    assert td(uppergamma(randcplx(), y), y)
    assert uppergamma(x, y).diff(x) == \
        uppergamma(x, y)*log(y) + meijerg([], [1, 1], [0, 0, x], [], y)
    assert td(uppergamma(x, randcplx()), x)

    p = Symbol('p', positive=True)
    assert uppergamma(0, p) == -Ei(-p)
    assert uppergamma(p, 0) == gamma(p)
    assert uppergamma(S.Half, x) == sqrt(pi)*erfc(sqrt(x))
    assert not uppergamma(S.Half - 3, x).has(uppergamma)
    assert not uppergamma(S.Half + 3, x).has(uppergamma)
    assert uppergamma(S.Half, x, evaluate=False).has(uppergamma)
    assert tn(uppergamma(S.Half + 3, x, evaluate=False),
              uppergamma(S.Half + 3, x), x)
    assert tn(uppergamma(S.Half - 3, x, evaluate=False),
              uppergamma(S.Half - 3, x), x)

    assert unchanged(uppergamma, x, -oo)
    assert unchanged(uppergamma, x, 0)

    assert tn_branch(-3, uppergamma)
    assert tn_branch(-4, uppergamma)
    assert tn_branch(Rational(1, 3), uppergamma)
    assert tn_branch(pi, uppergamma)
    assert uppergamma(3, exp_polar(4*pi*I)*x) == uppergamma(3, x)
    assert uppergamma(y, exp_polar(5*pi*I)*x) == \
        exp(4*I*pi*y)*uppergamma(y, x*exp_polar(pi*I)) + \
        gamma(y)*(1 - exp(4*pi*I*y))
    assert uppergamma(-2, exp_polar(5*pi*I)*x) == \
        uppergamma(-2, x*exp_polar(I*pi)) - 2*pi*I

    assert uppergamma(-2, x) == expint(3, x)/x**2

    assert conjugate(uppergamma(x, y)) == uppergamma(conjugate(x), conjugate(y))
    assert unchanged(conjugate, uppergamma(x, -oo))

    assert uppergamma(x, y).rewrite(expint) == y**x*expint(-x + 1, y)
    assert uppergamma(x, y).rewrite(lowergamma) == gamma(x) - lowergamma(x, y)

    assert uppergamma(70, 6) == 69035724522603011058660187038367026272747334489677105069435923032634389419656200387949342530805432320*exp(-6)
    assert (uppergamma(S(77) / 2, 6) - uppergamma(S(77) / 2, 6, evaluate=False)).evalf() < 1e-16
    assert (uppergamma(-S(77) / 2, 6) - uppergamma(-S(77) / 2, 6, evaluate=False)).evalf() < 1e-16


def test_polygamma():
    assert polygamma(n, nan) is nan

    assert polygamma(0, oo) is oo
    assert polygamma(0, -oo) is oo
    assert polygamma(0, I*oo) is oo
    assert polygamma(0, -I*oo) is oo
    assert polygamma(1, oo) == 0
    assert polygamma(5, oo) == 0

    assert polygamma(0, -9) is zoo

    assert polygamma(0, -9) is zoo
    assert polygamma(0, -1) is zoo
    assert polygamma(Rational(3, 2), -1) is zoo

    assert polygamma(0, 0) is zoo

    assert polygamma(0, 1) == -EulerGamma
    assert polygamma(0, 7) == Rational(49, 20) - EulerGamma

    assert polygamma(1, 1) == pi**2/6
    assert polygamma(1, 2) == pi**2/6 - 1
    assert polygamma(1, 3) == pi**2/6 - Rational(5, 4)
    assert polygamma(3, 1) == pi**4 / 15
    assert polygamma(3, 5) == 6*(Rational(-22369, 20736) + pi**4/90)
    assert polygamma(5, 1) == 8 * pi**6 / 63

    assert polygamma(1, S.Half) == pi**2 / 2
    assert polygamma(2, S.Half) == -14*zeta(3)
    assert polygamma(11, S.Half) == 176896*pi**12

    def t(m, n):
        x = S(m)/n
        r = polygamma(0, x)
        if r.has(polygamma):
            return False
        return abs(polygamma(0, x.n()).n() - r.n()).n() < 1e-10
    assert t(1, 2)
    assert t(3, 2)
    assert t(-1, 2)
    assert t(1, 4)
    assert t(-3, 4)
    assert t(1, 3)
    assert t(4, 3)
    assert t(3, 4)
    assert t(2, 3)
    assert t(123, 5)

    assert polygamma(0, x).rewrite(zeta) == polygamma(0, x)
    assert polygamma(1, x).rewrite(zeta) == zeta(2, x)
    assert polygamma(2, x).rewrite(zeta) == -2*zeta(3, x)
    assert polygamma(I, 2).rewrite(zeta) == polygamma(I, 2)
    n1 = Symbol('n1')
    n2 = Symbol('n2', real=True)
    n3 = Symbol('n3', integer=True)
    n4 = Symbol('n4', positive=True)
    n5 = Symbol('n5', positive=True, integer=True)
    assert polygamma(n1, x).rewrite(zeta) == polygamma(n1, x)
    assert polygamma(n2, x).rewrite(zeta) == polygamma(n2, x)
    assert polygamma(n3, x).rewrite(zeta) == polygamma(n3, x)
    assert polygamma(n4, x).rewrite(zeta) == polygamma(n4, x)
    assert polygamma(n5, x).rewrite(zeta) == (-1)**(n5 + 1) * factorial(n5) * zeta(n5 + 1, x)

    assert polygamma(3, 7*x).diff(x) == 7*polygamma(4, 7*x)

    assert polygamma(0, x).rewrite(harmonic) == harmonic(x - 1) - EulerGamma
    assert polygamma(2, x).rewrite(harmonic) == 2*harmonic(x - 1, 3) - 2*zeta(3)
    ni = Symbol("n", integer=True)
    assert polygamma(ni, x).rewrite(harmonic) == (-1)**(ni + 1)*(-harmonic(x - 1, ni + 1)
                                                                 + zeta(ni + 1))*factorial(ni)

    # Polygamma of non-negative integer order is unbranched:
    k = Symbol('n', integer=True, nonnegative=True)
    assert polygamma(k, exp_polar(2*I*pi)*x) == polygamma(k, x)

    # but negative integers are branched!
    k = Symbol('n', integer=True)
    assert polygamma(k, exp_polar(2*I*pi)*x).args == (k, exp_polar(2*I*pi)*x)

    # Polygamma of order -1 is loggamma:
    assert polygamma(-1, x) == loggamma(x) - log(2*pi) / 2

    # But smaller orders are iterated integrals and don't have a special name
    assert polygamma(-2, x).func is polygamma

    # Test a bug
    assert polygamma(0, -x).expand(func=True) == polygamma(0, -x)

    assert polygamma(2, 2.5).is_positive == False
    assert polygamma(2, -2.5).is_positive == False
    assert polygamma(3, 2.5).is_positive == True
    assert polygamma(3, -2.5).is_positive is True
    assert polygamma(-2, -2.5).is_positive is None
    assert polygamma(-3, -2.5).is_positive is None

    assert polygamma(2, 2.5).is_negative == True
    assert polygamma(3, 2.5).is_negative == False
    assert polygamma(3, -2.5).is_negative == False
    assert polygamma(2, -2.5).is_negative is True
    assert polygamma(-2, -2.5).is_negative is None
    assert polygamma(-3, -2.5).is_negative is None

    assert polygamma(I, 2).is_positive is None
    assert polygamma(I, 3).is_negative is None

    # issue 17350
    assert (I*polygamma(I, pi)).as_real_imag() == \
           (-im(polygamma(I, pi)), re(polygamma(I, pi)))
    assert (tanh(polygamma(I, 1))).rewrite(exp) == \
           (exp(polygamma(I, 1)) - exp(-polygamma(I, 1)))/(exp(polygamma(I, 1)) + exp(-polygamma(I, 1)))
    assert (I / polygamma(I, 4)).rewrite(exp) == \
           I*exp(-I*atan(im(polygamma(I, 4))/re(polygamma(I, 4))))/Abs(polygamma(I, 4))

    # issue 12569
    assert unchanged(im, polygamma(0, I))
    assert polygamma(Symbol('a', positive=True), Symbol('b', positive=True)).is_real is True
    assert polygamma(0, I).is_real is None

    assert str(polygamma(pi, 3).evalf(n=10)) == "0.1169314564"
    assert str(polygamma(2.3, 1.0).evalf(n=10)) == "-3.003302909"
    assert str(polygamma(-1, 1).evalf(n=10)) == "-0.9189385332" # not zero
    assert str(polygamma(I, 1).evalf(n=10)) == "-3.109856569 + 1.89089016*I"
    assert str(polygamma(1, I).evalf(n=10)) == "-0.5369999034 - 0.7942335428*I"
    assert str(polygamma(I, I).evalf(n=10)) == "6.332362889 + 45.92828268*I"


def test_polygamma_expand_func():
    assert polygamma(0, x).expand(func=True) == polygamma(0, x)
    assert polygamma(0, 2*x).expand(func=True) == \
        polygamma(0, x)/2 + polygamma(0, S.Half + x)/2 + log(2)
    assert polygamma(1, 2*x).expand(func=True) == \
        polygamma(1, x)/4 + polygamma(1, S.Half + x)/4
    assert polygamma(2, x).expand(func=True) == \
        polygamma(2, x)
    assert polygamma(0, -1 + x).expand(func=True) == \
        polygamma(0, x) - 1/(x - 1)
    assert polygamma(0, 1 + x).expand(func=True) == \
        1/x + polygamma(0, x )
    assert polygamma(0, 2 + x).expand(func=True) == \
        1/x + 1/(1 + x) + polygamma(0, x)
    assert polygamma(0, 3 + x).expand(func=True) == \
        polygamma(0, x) + 1/x + 1/(1 + x) + 1/(2 + x)
    assert polygamma(0, 4 + x).expand(func=True) == \
        polygamma(0, x) + 1/x + 1/(1 + x) + 1/(2 + x) + 1/(3 + x)
    assert polygamma(1, 1 + x).expand(func=True) == \
        polygamma(1, x) - 1/x**2
    assert polygamma(1, 2 + x).expand(func=True, multinomial=False) == \
        polygamma(1, x) - 1/x**2 - 1/(1 + x)**2
    assert polygamma(1, 3 + x).expand(func=True, multinomial=False) == \
        polygamma(1, x) - 1/x**2 - 1/(1 + x)**2 - 1/(2 + x)**2
    assert polygamma(1, 4 + x).expand(func=True, multinomial=False) == \
        polygamma(1, x) - 1/x**2 - 1/(1 + x)**2 - \
        1/(2 + x)**2 - 1/(3 + x)**2
    assert polygamma(0, x + y).expand(func=True) == \
        polygamma(0, x + y)
    assert polygamma(1, x + y).expand(func=True) == \
        polygamma(1, x + y)
    assert polygamma(1, 3 + 4*x + y).expand(func=True, multinomial=False) == \
        polygamma(1, y + 4*x) - 1/(y + 4*x)**2 - \
        1/(1 + y + 4*x)**2 - 1/(2 + y + 4*x)**2
    assert polygamma(3, 3 + 4*x + y).expand(func=True, multinomial=False) == \
        polygamma(3, y + 4*x) - 6/(y + 4*x)**4 - \
        6/(1 + y + 4*x)**4 - 6/(2 + y + 4*x)**4
    assert polygamma(3, 4*x + y + 1).expand(func=True, multinomial=False) == \
        polygamma(3, y + 4*x) - 6/(y + 4*x)**4
    e = polygamma(3, 4*x + y + Rational(3, 2))
    assert e.expand(func=True) == e
    e = polygamma(3, x + y + Rational(3, 4))
    assert e.expand(func=True, basic=False) == e

    assert polygamma(-1, x, evaluate=False).expand(func=True) == \
        loggamma(x) - log(pi)/2 - log(2)/2
    p2 = polygamma(-2, x).expand(func=True) + x**2/2 - x/2 + S(1)/12
    assert isinstance(p2, Subs)
    assert p2.point == (-1,)


def test_digamma():
    assert digamma(nan) == nan

    assert digamma(oo) == oo
    assert digamma(-oo) == oo
    assert digamma(I*oo) == oo
    assert digamma(-I*oo) == oo

    assert digamma(-9) == zoo

    assert digamma(-9) == zoo
    assert digamma(-1) == zoo

    assert digamma(0) == zoo

    assert digamma(1) == -EulerGamma
    assert digamma(7) == Rational(49, 20) - EulerGamma

    def t(m, n):
        x = S(m)/n
        r = digamma(x)
        if r.has(digamma):
            return False
        return abs(digamma(x.n()).n() - r.n()).n() < 1e-10
    assert t(1, 2)
    assert t(3, 2)
    assert t(-1, 2)
    assert t(1, 4)
    assert t(-3, 4)
    assert t(1, 3)
    assert t(4, 3)
    assert t(3, 4)
    assert t(2, 3)
    assert t(123, 5)

    assert digamma(x).rewrite(zeta) == polygamma(0, x)

    assert digamma(x).rewrite(harmonic) == harmonic(x - 1) - EulerGamma

    assert digamma(I).is_real is None

    assert digamma(x,evaluate=False).fdiff() == polygamma(1, x)

    assert digamma(x,evaluate=False).is_real is None

    assert digamma(x,evaluate=False).is_positive is None

    assert digamma(x,evaluate=False).is_negative is None

    assert digamma(x,evaluate=False).rewrite(polygamma) == polygamma(0, x)


def test_digamma_expand_func():
    assert digamma(x).expand(func=True) == polygamma(0, x)
    assert digamma(2*x).expand(func=True) == \
        polygamma(0, x)/2 + polygamma(0, Rational(1, 2) + x)/2 + log(2)
    assert digamma(-1 + x).expand(func=True) == \
        polygamma(0, x) - 1/(x - 1)
    assert digamma(1 + x).expand(func=True) == \
        1/x + polygamma(0, x )
    assert digamma(2 + x).expand(func=True) == \
        1/x + 1/(1 + x) + polygamma(0, x)
    assert digamma(3 + x).expand(func=True) == \
        polygamma(0, x) + 1/x + 1/(1 + x) + 1/(2 + x)
    assert digamma(4 + x).expand(func=True) == \
        polygamma(0, x) + 1/x + 1/(1 + x) + 1/(2 + x) + 1/(3 + x)
    assert digamma(x + y).expand(func=True) == \
        polygamma(0, x + y)

def test_trigamma():
    assert trigamma(nan) == nan

    assert trigamma(oo) == 0

    assert trigamma(1) == pi**2/6
    assert trigamma(2) == pi**2/6 - 1
    assert trigamma(3) == pi**2/6 - Rational(5, 4)

    assert trigamma(x, evaluate=False).rewrite(zeta) == zeta(2, x)
    assert trigamma(x, evaluate=False).rewrite(harmonic) == \
        trigamma(x).rewrite(polygamma).rewrite(harmonic)

    assert trigamma(x,evaluate=False).fdiff() == polygamma(2, x)

    assert trigamma(x,evaluate=False).is_real is None

    assert trigamma(x,evaluate=False).is_positive is None

    assert trigamma(x,evaluate=False).is_negative is None

    assert trigamma(x,evaluate=False).rewrite(polygamma) == polygamma(1, x)

def test_trigamma_expand_func():
    assert trigamma(2*x).expand(func=True) == \
        polygamma(1, x)/4 + polygamma(1, Rational(1, 2) + x)/4
    assert trigamma(1 + x).expand(func=True) == \
        polygamma(1, x) - 1/x**2
    assert trigamma(2 + x).expand(func=True, multinomial=False) == \
        polygamma(1, x) - 1/x**2 - 1/(1 + x)**2
    assert trigamma(3 + x).expand(func=True, multinomial=False) == \
        polygamma(1, x) - 1/x**2 - 1/(1 + x)**2 - 1/(2 + x)**2
    assert trigamma(4 + x).expand(func=True, multinomial=False) == \
        polygamma(1, x) - 1/x**2 - 1/(1 + x)**2 - \
        1/(2 + x)**2 - 1/(3 + x)**2
    assert trigamma(x + y).expand(func=True) == \
        polygamma(1, x + y)
    assert trigamma(3 + 4*x + y).expand(func=True, multinomial=False) == \
        polygamma(1, y + 4*x) - 1/(y + 4*x)**2 - \
        1/(1 + y + 4*x)**2 - 1/(2 + y + 4*x)**2

def test_loggamma():
    raises(TypeError, lambda: loggamma(2, 3))
    raises(ArgumentIndexError, lambda: loggamma(x).fdiff(2))

    assert loggamma(-1) is oo
    assert loggamma(-2) is oo
    assert loggamma(0) is oo
    assert loggamma(1) == 0
    assert loggamma(2) == 0
    assert loggamma(3) == log(2)
    assert loggamma(4) == log(6)

    n = Symbol("n", integer=True, positive=True)
    assert loggamma(n) == log(gamma(n))
    assert loggamma(-n) is oo
    assert loggamma(n/2) == log(2**(-n + 1)*sqrt(pi)*gamma(n)/gamma(n/2 + S.Half))

    assert loggamma(oo) is oo
    assert loggamma(-oo) is zoo
    assert loggamma(I*oo) is zoo
    assert loggamma(-I*oo) is zoo
    assert loggamma(zoo) is zoo
    assert loggamma(nan) is nan

    L = loggamma(Rational(16, 3))
    E = -5*log(3) + loggamma(Rational(1, 3)) + log(4) + log(7) + log(10) + log(13)
    assert expand_func(L).doit() == E
    assert L.n() == E.n()

    L = loggamma(Rational(19, 4))
    E = -4*log(4) + loggamma(Rational(3, 4)) + log(3) + log(7) + log(11) + log(15)
    assert expand_func(L).doit() == E
    assert L.n() == E.n()

    L = loggamma(Rational(23, 7))
    E = -3*log(7) + log(2) + loggamma(Rational(2, 7)) + log(9) + log(16)
    assert expand_func(L).doit() == E
    assert L.n() == E.n()

    L = loggamma(Rational(19, 4) - 7)
    E = -log(9) - log(5) + loggamma(Rational(3, 4)) + 3*log(4) - 3*I*pi
    assert expand_func(L).doit() == E
    assert L.n() == E.n()

    L = loggamma(Rational(23, 7) - 6)
    E = -log(19) - log(12) - log(5) + loggamma(Rational(2, 7)) + 3*log(7) - 3*I*pi
    assert expand_func(L).doit() == E
    assert L.n() == E.n()

    assert loggamma(x).diff(x) == polygamma(0, x)
    s1 = loggamma(1/(x + sin(x)) + cos(x)).nseries(x, n=4)
    s2 = (-log(2*x) - 1)/(2*x) - log(x/pi)/2 + (4 - log(2*x))*x/24 + O(x**2) + \
        log(x)*x**2/2
    assert (s1 - s2).expand(force=True).removeO() == 0
    s1 = loggamma(1/x).series(x)
    s2 = (1/x - S.Half)*log(1/x) - 1/x + log(2*pi)/2 + \
        x/12 - x**3/360 + x**5/1260 + O(x**7)
    assert ((s1 - s2).expand(force=True)).removeO() == 0

    assert loggamma(x).rewrite('intractable') == log(gamma(x))

    s1 = loggamma(x).series(x).cancel()
    assert s1 == -log(x) - EulerGamma*x + pi**2*x**2/12 + x**3*polygamma(2, 1)/6 + \
        pi**4*x**4/360 + x**5*polygamma(4, 1)/120 + O(x**6)
    assert s1 == loggamma(x).rewrite('intractable').series(x).cancel()

    assert conjugate(loggamma(x)) == loggamma(conjugate(x))
    assert conjugate(loggamma(0)) is oo
    assert conjugate(loggamma(1)) == loggamma(conjugate(1))
    assert conjugate(loggamma(-oo)) == conjugate(zoo)

    assert loggamma(Symbol('v', positive=True)).is_real is True
    assert loggamma(Symbol('v', zero=True)).is_real is False
    assert loggamma(Symbol('v', negative=True)).is_real is False
    assert loggamma(Symbol('v', nonpositive=True)).is_real is False
    assert loggamma(Symbol('v', nonnegative=True)).is_real is None
    assert loggamma(Symbol('v', imaginary=True)).is_real is None
    assert loggamma(Symbol('v', real=True)).is_real is None
    assert loggamma(Symbol('v')).is_real is None

    assert loggamma(S.Half).is_real is True
    assert loggamma(0).is_real is False
    assert loggamma(Rational(-1, 2)).is_real is False
    assert loggamma(I).is_real is None
    assert loggamma(2 + 3*I).is_real is None

    def tN(N, M):
        assert loggamma(1/x)._eval_nseries(x, n=N).getn() == M
    tN(0, 0)
    tN(1, 1)
    tN(2, 2)
    tN(3, 3)
    tN(4, 4)
    tN(5, 5)


def test_polygamma_expansion():
    # A. & S., pa. 259 and 260
    assert polygamma(0, 1/x).nseries(x, n=3) == \
        -log(x) - x/2 - x**2/12 + O(x**3)
    assert polygamma(1, 1/x).series(x, n=5) == \
        x + x**2/2 + x**3/6 + O(x**5)
    assert polygamma(3, 1/x).nseries(x, n=11) == \
        2*x**3 + 3*x**4 + 2*x**5 - x**7 + 4*x**9/3 + O(x**11)


def test_polygamma_leading_term():
    expr = -log(1/x) + polygamma(0, 1 + 1/x) + S.EulerGamma
    assert expr.as_leading_term(x, logx=-y) == S.EulerGamma


def test_issue_8657():
    n = Symbol('n', negative=True, integer=True)
    m = Symbol('m', integer=True)
    o = Symbol('o', positive=True)
    p = Symbol('p', negative=True, integer=False)
    assert gamma(n).is_real is False
    assert gamma(m).is_real is None
    assert gamma(o).is_real is True
    assert gamma(p).is_real is True
    assert gamma(w).is_real is None


def test_issue_8524():
    x = Symbol('x', positive=True)
    y = Symbol('y', negative=True)
    z = Symbol('z', positive=False)
    p = Symbol('p', negative=False)
    q = Symbol('q', integer=True)
    r = Symbol('r', integer=False)
    e = Symbol('e', even=True, negative=True)
    assert gamma(x).is_positive is True
    assert gamma(y).is_positive is None
    assert gamma(z).is_positive is None
    assert gamma(p).is_positive is None
    assert gamma(q).is_positive is None
    assert gamma(r).is_positive is None
    assert gamma(e + S.Half).is_positive is True
    assert gamma(e - S.Half).is_positive is False

def test_issue_14450():
    assert uppergamma(Rational(3, 8), x).evalf() == uppergamma(Rational(3, 8), x)
    assert lowergamma(x, Rational(3, 8)).evalf() == lowergamma(x, Rational(3, 8))
    # some values from Wolfram Alpha for comparison
    assert abs(uppergamma(Rational(3, 8), 2).evalf() - 0.07105675881) < 1e-9
    assert abs(lowergamma(Rational(3, 8), 2).evalf() - 2.2993794256) < 1e-9

def test_issue_14528():
    k = Symbol('k', integer=True, nonpositive=True)
    assert isinstance(gamma(k), gamma)

def test_multigamma():
    from sympy.concrete.products import Product
    p = Symbol('p')
    _k = Dummy('_k')

    assert multigamma(x, p).dummy_eq(pi**(p*(p - 1)/4)*\
        Product(gamma(x + (1 - _k)/2), (_k, 1, p)))

    assert conjugate(multigamma(x, p)).dummy_eq(pi**((conjugate(p) - 1)*\
        conjugate(p)/4)*Product(gamma(conjugate(x) + (1-conjugate(_k))/2), (_k, 1, p)))
    assert conjugate(multigamma(x, 1)) == gamma(conjugate(x))

    p = Symbol('p', positive=True)
    assert conjugate(multigamma(x, p)).dummy_eq(pi**((p - 1)*p/4)*\
        Product(gamma(conjugate(x) + (1-conjugate(_k))/2), (_k, 1, p)))

    assert multigamma(nan, 1) is nan
    assert multigamma(oo, 1).doit() is oo

    assert multigamma(1, 1) == 1
    assert multigamma(2, 1) == 1
    assert multigamma(3, 1) == 2

    assert multigamma(102, 1) == factorial(101)
    assert multigamma(S.Half, 1) == sqrt(pi)

    assert multigamma(1, 2) == pi
    assert multigamma(2, 2) == pi/2

    assert multigamma(1, 3) is zoo
    assert multigamma(2, 3) == pi**2/2
    assert multigamma(3, 3) == 3*pi**2/2

    assert multigamma(x, 1).diff(x) == gamma(x)*polygamma(0, x)
    assert multigamma(x, 2).diff(x) == sqrt(pi)*gamma(x)*gamma(x - S.Half)*\
        polygamma(0, x) + sqrt(pi)*gamma(x)*gamma(x - S.Half)*polygamma(0, x - S.Half)

    assert multigamma(x - 1, 1).expand(func=True) == gamma(x)/(x - 1)
    assert multigamma(x + 2, 1).expand(func=True, mul=False) == x*(x + 1)*\
        gamma(x)
    assert multigamma(x - 1, 2).expand(func=True) == sqrt(pi)*gamma(x)*\
        gamma(x + S.Half)/(x**3 - 3*x**2 + x*Rational(11, 4) - Rational(3, 4))
    assert multigamma(x - 1, 3).expand(func=True) == pi**Rational(3, 2)*gamma(x)**2*\
        gamma(x + S.Half)/(x**5 - 6*x**4 + 55*x**3/4 - 15*x**2 + x*Rational(31, 4) - Rational(3, 2))

    assert multigamma(n, 1).rewrite(factorial) == factorial(n - 1)
    assert multigamma(n, 2).rewrite(factorial) == sqrt(pi)*\
        factorial(n - Rational(3, 2))*factorial(n - 1)
    assert multigamma(n, 3).rewrite(factorial) == pi**Rational(3, 2)*\
        factorial(n - 2)*factorial(n - Rational(3, 2))*factorial(n - 1)

    assert multigamma(Rational(-1, 2), 3, evaluate=False).is_real == False
    assert multigamma(S.Half, 3, evaluate=False).is_real == False
    assert multigamma(0, 1, evaluate=False).is_real == False
    assert multigamma(1, 3, evaluate=False).is_real == False
    assert multigamma(-1.0, 3, evaluate=False).is_real == False
    assert multigamma(0.7, 3, evaluate=False).is_real == True
    assert multigamma(3, 3, evaluate=False).is_real == True

def test_gamma_as_leading_term():
    assert gamma(x).as_leading_term(x) == 1/x
    assert gamma(2 + x).as_leading_term(x) == S(1)
    assert gamma(cos(x)).as_leading_term(x) == S(1)
    assert gamma(sin(x)).as_leading_term(x) == 1/x

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\test_get_dummies.py ===
import re
import unicodedata

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas.core.dtypes.common import is_integer_dtype

import pandas as pd
from pandas import (
    ArrowDtype,
    Categorical,
    CategoricalDtype,
    CategoricalIndex,
    DataFrame,
    Index,
    RangeIndex,
    Series,
    SparseDtype,
    get_dummies,
)
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray

try:
    import pyarrow as pa
except ImportError:
    pa = None


class TestGetDummies:
    @pytest.fixture
    def df(self):
        return DataFrame({"A": ["a", "b", "a"], "B": ["b", "b", "c"], "C": [1, 2, 3]})

    @pytest.fixture(params=["uint8", "i8", np.float64, bool, None])
    def dtype(self, request):
        return np.dtype(request.param)

    @pytest.fixture(params=["dense", "sparse"])
    def sparse(self, request):
        # params are strings to simplify reading test results,
        # e.g. TestGetDummies::test_basic[uint8-sparse] instead of [uint8-True]
        return request.param == "sparse"

    def effective_dtype(self, dtype):
        if dtype is None:
            return np.uint8
        return dtype

    def test_get_dummies_raises_on_dtype_object(self, df):
        msg = "dtype=object is not a valid dtype for get_dummies"
        with pytest.raises(ValueError, match=msg):
            get_dummies(df, dtype="object")

    def test_get_dummies_basic(self, sparse, dtype):
        s_list = list("abc")
        s_series = Series(s_list)
        s_series_index = Series(s_list, list("ABC"))

        expected = DataFrame(
            {"a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1]},
            dtype=self.effective_dtype(dtype),
        )
        if sparse:
            if dtype.kind == "b":
                expected = expected.apply(SparseArray, fill_value=False)
            else:
                expected = expected.apply(SparseArray, fill_value=0.0)
        result = get_dummies(s_list, sparse=sparse, dtype=dtype)
        tm.assert_frame_equal(result, expected)

        result = get_dummies(s_series, sparse=sparse, dtype=dtype)
        tm.assert_frame_equal(result, expected)

        expected.index = list("ABC")
        result = get_dummies(s_series_index, sparse=sparse, dtype=dtype)
        tm.assert_frame_equal(result, expected)

    def test_get_dummies_basic_types(self, sparse, dtype, using_infer_string):
        # GH 10531
        s_list = list("abc")
        s_series = Series(s_list)
        s_df = DataFrame(
            {"a": [0, 1, 0, 1, 2], "b": ["A", "A", "B", "C", "C"], "c": [2, 3, 3, 3, 2]}
        )

        expected = DataFrame(
            {"a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1]},
            dtype=self.effective_dtype(dtype),
            columns=list("abc"),
        )
        if sparse:
            if is_integer_dtype(dtype):
                fill_value = 0
            elif dtype == bool:
                fill_value = False
            else:
                fill_value = 0.0

            expected = expected.apply(SparseArray, fill_value=fill_value)
        result = get_dummies(s_list, sparse=sparse, dtype=dtype)
        tm.assert_frame_equal(result, expected)

        result = get_dummies(s_series, sparse=sparse, dtype=dtype)
        tm.assert_frame_equal(result, expected)

        result = get_dummies(s_df, columns=s_df.columns, sparse=sparse, dtype=dtype)
        if sparse:
            dtype_name = f"Sparse[{self.effective_dtype(dtype).name}, {fill_value}]"
        else:
            dtype_name = self.effective_dtype(dtype).name

        expected = Series({dtype_name: 8}, name="count")
        result = result.dtypes.value_counts()
        result.index = [str(i) for i in result.index]
        tm.assert_series_equal(result, expected)

        result = get_dummies(s_df, columns=["a"], sparse=sparse, dtype=dtype)

        key = "str" if using_infer_string else "object"
        expected_counts = {"int64": 1, key: 1}
        expected_counts[dtype_name] = 3 + expected_counts.get(dtype_name, 0)

        expected = Series(expected_counts, name="count").sort_index()
        result = result.dtypes.value_counts()
        result.index = [str(i) for i in result.index]
        result = result.sort_index()
        tm.assert_series_equal(result, expected)

    def test_get_dummies_just_na(self, sparse):
        just_na_list = [np.nan]
        just_na_series = Series(just_na_list)
        just_na_series_index = Series(just_na_list, index=["A"])

        res_list = get_dummies(just_na_list, sparse=sparse)
        res_series = get_dummies(just_na_series, sparse=sparse)
        res_series_index = get_dummies(just_na_series_index, sparse=sparse)

        assert res_list.empty
        assert res_series.empty
        assert res_series_index.empty

        assert res_list.index.tolist() == [0]
        assert res_series.index.tolist() == [0]
        assert res_series_index.index.tolist() == ["A"]

    def test_get_dummies_include_na(self, sparse, dtype):
        s = ["a", "b", np.nan]
        res = get_dummies(s, sparse=sparse, dtype=dtype)
        exp = DataFrame(
            {"a": [1, 0, 0], "b": [0, 1, 0]}, dtype=self.effective_dtype(dtype)
        )
        if sparse:
            if dtype.kind == "b":
                exp = exp.apply(SparseArray, fill_value=False)
            else:
                exp = exp.apply(SparseArray, fill_value=0.0)
        tm.assert_frame_equal(res, exp)

        # Sparse dataframes do not allow nan labelled columns, see #GH8822
        res_na = get_dummies(s, dummy_na=True, sparse=sparse, dtype=dtype)
        exp_na = DataFrame(
            {np.nan: [0, 0, 1], "a": [1, 0, 0], "b": [0, 1, 0]},
            dtype=self.effective_dtype(dtype),
        )
        exp_na = exp_na.reindex(["a", "b", np.nan], axis=1)
        # hack (NaN handling in assert_index_equal)
        exp_na.columns = res_na.columns
        if sparse:
            if dtype.kind == "b":
                exp_na = exp_na.apply(SparseArray, fill_value=False)
            else:
                exp_na = exp_na.apply(SparseArray, fill_value=0.0)
        tm.assert_frame_equal(res_na, exp_na)

        res_just_na = get_dummies([np.nan], dummy_na=True, sparse=sparse, dtype=dtype)
        exp_just_na = DataFrame(
            Series(1, index=[0]), columns=[np.nan], dtype=self.effective_dtype(dtype)
        )
        tm.assert_numpy_array_equal(res_just_na.values, exp_just_na.values)

    def test_get_dummies_unicode(self, sparse):
        # See GH 6885 - get_dummies chokes on unicode values
        e = "e"
        eacute = unicodedata.lookup("LATIN SMALL LETTER E WITH ACUTE")
        s = [e, eacute, eacute]
        res = get_dummies(s, prefix="letter", sparse=sparse)
        exp = DataFrame(
            {"letter_e": [True, False, False], f"letter_{eacute}": [False, True, True]}
        )
        if sparse:
            exp = exp.apply(SparseArray, fill_value=False)
        tm.assert_frame_equal(res, exp)

    def test_dataframe_dummies_all_obj(self, df, sparse):
        df = df[["A", "B"]]
        result = get_dummies(df, sparse=sparse)
        expected = DataFrame(
            {"A_a": [1, 0, 1], "A_b": [0, 1, 0], "B_b": [1, 1, 0], "B_c": [0, 0, 1]},
            dtype=bool,
        )
        if sparse:
            expected = DataFrame(
                {
                    "A_a": SparseArray([1, 0, 1], dtype="bool"),
                    "A_b": SparseArray([0, 1, 0], dtype="bool"),
                    "B_b": SparseArray([1, 1, 0], dtype="bool"),
                    "B_c": SparseArray([0, 0, 1], dtype="bool"),
                }
            )

        tm.assert_frame_equal(result, expected)

    def test_dataframe_dummies_string_dtype(self, df, any_string_dtype):
        # GH44965
        df = df[["A", "B"]]
        df = df.astype({"A": "str", "B": any_string_dtype})
        result = get_dummies(df)
        expected = DataFrame(
            {
                "A_a": [1, 0, 1],
                "A_b": [0, 1, 0],
                "B_b": [1, 1, 0],
                "B_c": [0, 0, 1],
            },
            dtype=bool,
        )
        if any_string_dtype == "string" and any_string_dtype.na_value is pd.NA:
            expected[["B_b", "B_c"]] = expected[["B_b", "B_c"]].astype("boolean")
        tm.assert_frame_equal(result, expected)

    def test_dataframe_dummies_mix_default(self, df, sparse, dtype):
        result = get_dummies(df, sparse=sparse, dtype=dtype)
        if sparse:
            arr = SparseArray
            if dtype.kind == "b":
                typ = SparseDtype(dtype, False)
            else:
                typ = SparseDtype(dtype, 0)
        else:
            arr = np.array
            typ = dtype
        expected = DataFrame(
            {
                "C": [1, 2, 3],
                "A_a": arr([1, 0, 1], dtype=typ),
                "A_b": arr([0, 1, 0], dtype=typ),
                "B_b": arr([1, 1, 0], dtype=typ),
                "B_c": arr([0, 0, 1], dtype=typ),
            }
        )
        expected = expected[["C", "A_a", "A_b", "B_b", "B_c"]]
        tm.assert_frame_equal(result, expected)

    def test_dataframe_dummies_prefix_list(self, df, sparse):
        prefixes = ["from_A", "from_B"]
        result = get_dummies(df, prefix=prefixes, sparse=sparse)
        expected = DataFrame(
            {
                "C": [1, 2, 3],
                "from_A_a": [True, False, True],
                "from_A_b": [False, True, False],
                "from_B_b": [True, True, False],
                "from_B_c": [False, False, True],
            },
        )
        expected[["C"]] = df[["C"]]
        cols = ["from_A_a", "from_A_b", "from_B_b", "from_B_c"]
        expected = expected[["C"] + cols]

        typ = SparseArray if sparse else Series
        expected[cols] = expected[cols].apply(lambda x: typ(x))
        tm.assert_frame_equal(result, expected)

    def test_dataframe_dummies_prefix_str(self, df, sparse):
        # not that you should do this...
        result = get_dummies(df, prefix="bad", sparse=sparse)
        bad_columns = ["bad_a", "bad_b", "bad_b", "bad_c"]
        expected = DataFrame(
            [
                [1, True, False, True, False],
                [2, False, True, True, False],
                [3, True, False, False, True],
            ],
            columns=["C"] + bad_columns,
        )
        expected = expected.astype({"C": np.int64})
        if sparse:
            # work around astyping & assigning with duplicate columns
            # https://github.com/pandas-dev/pandas/issues/14427
            expected = pd.concat(
                [
                    Series([1, 2, 3], name="C"),
                    Series([True, False, True], name="bad_a", dtype="Sparse[bool]"),
                    Series([False, True, False], name="bad_b", dtype="Sparse[bool]"),
                    Series([True, True, False], name="bad_b", dtype="Sparse[bool]"),
                    Series([False, False, True], name="bad_c", dtype="Sparse[bool]"),
                ],
                axis=1,
            )

        tm.assert_frame_equal(result, expected)

    def test_dataframe_dummies_subset(self, df, sparse):
        result = get_dummies(df, prefix=["from_A"], columns=["A"], sparse=sparse)
        expected = DataFrame(
            {
                "B": ["b", "b", "c"],
                "C": [1, 2, 3],
                "from_A_a": [1, 0, 1],
                "from_A_b": [0, 1, 0],
            },
        )
        cols = expected.columns
        expected[cols[1:]] = expected[cols[1:]].astype(bool)
        expected[["C"]] = df[["C"]]
        if sparse:
            cols = ["from_A_a", "from_A_b"]
            expected[cols] = expected[cols].astype(SparseDtype("bool", False))
        tm.assert_frame_equal(result, expected)

    def test_dataframe_dummies_prefix_sep(self, df, sparse):
        result = get_dummies(df, prefix_sep="..", sparse=sparse)
        expected = DataFrame(
            {
                "C": [1, 2, 3],
                "A..a": [True, False, True],
                "A..b": [False, True, False],
                "B..b": [True, True, False],
                "B..c": [False, False, True],
            },
        )
        expected[["C"]] = df[["C"]]
        expected = expected[["C", "A..a", "A..b", "B..b", "B..c"]]
        if sparse:
            cols = ["A..a", "A..b", "B..b", "B..c"]
            expected[cols] = expected[cols].astype(SparseDtype("bool", False))

        tm.assert_frame_equal(result, expected)

        result = get_dummies(df, prefix_sep=["..", "__"], sparse=sparse)
        expected = expected.rename(columns={"B..b": "B__b", "B..c": "B__c"})
        tm.assert_frame_equal(result, expected)

        result = get_dummies(df, prefix_sep={"A": "..", "B": "__"}, sparse=sparse)
        tm.assert_frame_equal(result, expected)

    def test_dataframe_dummies_prefix_bad_length(self, df, sparse):
        msg = re.escape(
            "Length of 'prefix' (1) did not match the length of the columns being "
            "encoded (2)"
        )
        with pytest.raises(ValueError, match=msg):
            get_dummies(df, prefix=["too few"], sparse=sparse)

    def test_dataframe_dummies_prefix_sep_bad_length(self, df, sparse):
        msg = re.escape(
            "Length of 'prefix_sep' (1) did not match the length of the columns being "
            "encoded (2)"
        )
        with pytest.raises(ValueError, match=msg):
            get_dummies(df, prefix_sep=["bad"], sparse=sparse)

    def test_dataframe_dummies_prefix_dict(self, sparse):
        prefixes = {"A": "from_A", "B": "from_B"}
        df = DataFrame({"C": [1, 2, 3], "A": ["a", "b", "a"], "B": ["b", "b", "c"]})
        result = get_dummies(df, prefix=prefixes, sparse=sparse)

        expected = DataFrame(
            {
                "C": [1, 2, 3],
                "from_A_a": [1, 0, 1],
                "from_A_b": [0, 1, 0],
                "from_B_b": [1, 1, 0],
                "from_B_c": [0, 0, 1],
            }
        )

        columns = ["from_A_a", "from_A_b", "from_B_b", "from_B_c"]
        expected[columns] = expected[columns].astype(bool)
        if sparse:
            expected[columns] = expected[columns].astype(SparseDtype("bool", False))

        tm.assert_frame_equal(result, expected)

    def test_dataframe_dummies_with_na(self, df, sparse, dtype):
        df.loc[3, :] = [np.nan, np.nan, np.nan]
        result = get_dummies(df, dummy_na=True, sparse=sparse, dtype=dtype).sort_index(
            axis=1
        )

        if sparse:
            arr = SparseArray
            if dtype.kind == "b":
                typ = SparseDtype(dtype, False)
            else:
                typ = SparseDtype(dtype, 0)
        else:
            arr = np.array
            typ = dtype

        expected = DataFrame(
            {
                "C": [1, 2, 3, np.nan],
                "A_a": arr([1, 0, 1, 0], dtype=typ),
                "A_b": arr([0, 1, 0, 0], dtype=typ),
                "A_nan": arr([0, 0, 0, 1], dtype=typ),
                "B_b": arr([1, 1, 0, 0], dtype=typ),
                "B_c": arr([0, 0, 1, 0], dtype=typ),
                "B_nan": arr([0, 0, 0, 1], dtype=typ),
            }
        ).sort_index(axis=1)

        tm.assert_frame_equal(result, expected)

        result = get_dummies(df, dummy_na=False, sparse=sparse, dtype=dtype)
        expected = expected[["C", "A_a", "A_b", "B_b", "B_c"]]
        tm.assert_frame_equal(result, expected)

    def test_dataframe_dummies_with_categorical(self, df, sparse, dtype):
        df["cat"] = Categorical(["x", "y", "y"])
        result = get_dummies(df, sparse=sparse, dtype=dtype).sort_index(axis=1)
        if sparse:
            arr = SparseArray
            if dtype.kind == "b":
                typ = SparseDtype(dtype, False)
            else:
                typ = SparseDtype(dtype, 0)
        else:
            arr = np.array
            typ = dtype

        expected = DataFrame(
            {
                "C": [1, 2, 3],
                "A_a": arr([1, 0, 1], dtype=typ),
                "A_b": arr([0, 1, 0], dtype=typ),
                "B_b": arr([1, 1, 0], dtype=typ),
                "B_c": arr([0, 0, 1], dtype=typ),
                "cat_x": arr([1, 0, 0], dtype=typ),
                "cat_y": arr([0, 1, 1], dtype=typ),
            }
        ).sort_index(axis=1)

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "get_dummies_kwargs,expected",
        [
            (
                {"data": DataFrame({"ä": ["a"]})},
                DataFrame({"ä_a": [True]}),
            ),
            (
                {"data": DataFrame({"x": ["ä"]})},
                DataFrame({"x_ä": [True]}),
            ),
            (
                {"data": DataFrame({"x": ["a"]}), "prefix": "ä"},
                DataFrame({"ä_a": [True]}),
            ),
            (
                {"data": DataFrame({"x": ["a"]}), "prefix_sep": "ä"},
                DataFrame({"xäa": [True]}),
            ),
        ],
    )
    def test_dataframe_dummies_unicode(self, get_dummies_kwargs, expected):
        # GH22084 get_dummies incorrectly encodes unicode characters
        # in dataframe column names
        result = get_dummies(**get_dummies_kwargs)
        tm.assert_frame_equal(result, expected)

    def test_get_dummies_basic_drop_first(self, sparse):
        # GH12402 Add a new parameter `drop_first` to avoid collinearity
        # Basic case
        s_list = list("abc")
        s_series = Series(s_list)
        s_series_index = Series(s_list, list("ABC"))

        expected = DataFrame({"b": [0, 1, 0], "c": [0, 0, 1]}, dtype=bool)

        result = get_dummies(s_list, drop_first=True, sparse=sparse)
        if sparse:
            expected = expected.apply(SparseArray, fill_value=False)
        tm.assert_frame_equal(result, expected)

        result = get_dummies(s_series, drop_first=True, sparse=sparse)
        tm.assert_frame_equal(result, expected)

        expected.index = list("ABC")
        result = get_dummies(s_series_index, drop_first=True, sparse=sparse)
        tm.assert_frame_equal(result, expected)

    def test_get_dummies_basic_drop_first_one_level(self, sparse):
        # Test the case that categorical variable only has one level.
        s_list = list("aaa")
        s_series = Series(s_list)
        s_series_index = Series(s_list, list("ABC"))

        expected = DataFrame(index=RangeIndex(3))

        result = get_dummies(s_list, drop_first=True, sparse=sparse)
        tm.assert_frame_equal(result, expected)

        result = get_dummies(s_series, drop_first=True, sparse=sparse)
        tm.assert_frame_equal(result, expected)

        expected = DataFrame(index=list("ABC"))
        result = get_dummies(s_series_index, drop_first=True, sparse=sparse)
        tm.assert_frame_equal(result, expected)

    def test_get_dummies_basic_drop_first_NA(self, sparse):
        # Test NA handling together with drop_first
        s_NA = ["a", "b", np.nan]
        res = get_dummies(s_NA, drop_first=True, sparse=sparse)
        exp = DataFrame({"b": [0, 1, 0]}, dtype=bool)
        if sparse:
            exp = exp.apply(SparseArray, fill_value=False)

        tm.assert_frame_equal(res, exp)

        res_na = get_dummies(s_NA, dummy_na=True, drop_first=True, sparse=sparse)
        exp_na = DataFrame({"b": [0, 1, 0], np.nan: [0, 0, 1]}, dtype=bool).reindex(
            ["b", np.nan], axis=1
        )
        if sparse:
            exp_na = exp_na.apply(SparseArray, fill_value=False)
        tm.assert_frame_equal(res_na, exp_na)

        res_just_na = get_dummies(
            [np.nan], dummy_na=True, drop_first=True, sparse=sparse
        )
        exp_just_na = DataFrame(index=RangeIndex(1))
        tm.assert_frame_equal(res_just_na, exp_just_na)

    def test_dataframe_dummies_drop_first(self, df, sparse):
        df = df[["A", "B"]]
        result = get_dummies(df, drop_first=True, sparse=sparse)
        expected = DataFrame({"A_b": [0, 1, 0], "B_c": [0, 0, 1]}, dtype=bool)
        if sparse:
            expected = expected.apply(SparseArray, fill_value=False)
        tm.assert_frame_equal(result, expected)

    def test_dataframe_dummies_drop_first_with_categorical(self, df, sparse, dtype):
        df["cat"] = Categorical(["x", "y", "y"])
        result = get_dummies(df, drop_first=True, sparse=sparse)
        expected = DataFrame(
            {"C": [1, 2, 3], "A_b": [0, 1, 0], "B_c": [0, 0, 1], "cat_y": [0, 1, 1]}
        )
        cols = ["A_b", "B_c", "cat_y"]
        expected[cols] = expected[cols].astype(bool)
        expected = expected[["C", "A_b", "B_c", "cat_y"]]
        if sparse:
            for col in cols:
                expected[col] = SparseArray(expected[col])
        tm.assert_frame_equal(result, expected)

    def test_dataframe_dummies_drop_first_with_na(self, df, sparse):
        df.loc[3, :] = [np.nan, np.nan, np.nan]
        result = get_dummies(
            df, dummy_na=True, drop_first=True, sparse=sparse
        ).sort_index(axis=1)
        expected = DataFrame(
            {
                "C": [1, 2, 3, np.nan],
                "A_b": [0, 1, 0, 0],
                "A_nan": [0, 0, 0, 1],
                "B_c": [0, 0, 1, 0],
                "B_nan": [0, 0, 0, 1],
            }
        )
        cols = ["A_b", "A_nan", "B_c", "B_nan"]
        expected[cols] = expected[cols].astype(bool)
        expected = expected.sort_index(axis=1)
        if sparse:
            for col in cols:
                expected[col] = SparseArray(expected[col])

        tm.assert_frame_equal(result, expected)

        result = get_dummies(df, dummy_na=False, drop_first=True, sparse=sparse)
        expected = expected[["C", "A_b", "B_c"]]
        tm.assert_frame_equal(result, expected)

    def test_get_dummies_int_int(self):
        data = Series([1, 2, 1])
        result = get_dummies(data)
        expected = DataFrame([[1, 0], [0, 1], [1, 0]], columns=[1, 2], dtype=bool)
        tm.assert_frame_equal(result, expected)

        data = Series(Categorical(["a", "b", "a"]))
        result = get_dummies(data)
        expected = DataFrame(
            [[1, 0], [0, 1], [1, 0]], columns=Categorical(["a", "b"]), dtype=bool
        )
        tm.assert_frame_equal(result, expected)

    def test_get_dummies_int_df(self, dtype):
        data = DataFrame(
            {
                "A": [1, 2, 1],
                "B": Categorical(["a", "b", "a"]),
                "C": [1, 2, 1],
                "D": [1.0, 2.0, 1.0],
            }
        )
        columns = ["C", "D", "A_1", "A_2", "B_a", "B_b"]
        expected = DataFrame(
            [[1, 1.0, 1, 0, 1, 0], [2, 2.0, 0, 1, 0, 1], [1, 1.0, 1, 0, 1, 0]],
            columns=columns,
        )
        expected[columns[2:]] = expected[columns[2:]].astype(dtype)
        result = get_dummies(data, columns=["A", "B"], dtype=dtype)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("ordered", [True, False])
    def test_dataframe_dummies_preserve_categorical_dtype(self, dtype, ordered):
        # GH13854
        cat = Categorical(list("xy"), categories=list("xyz"), ordered=ordered)
        result = get_dummies(cat, dtype=dtype)

        data = np.array([[1, 0, 0], [0, 1, 0]], dtype=self.effective_dtype(dtype))
        cols = CategoricalIndex(
            cat.categories, categories=cat.categories, ordered=ordered
        )
        expected = DataFrame(data, columns=cols, dtype=self.effective_dtype(dtype))

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("sparse", [True, False])
    def test_get_dummies_dont_sparsify_all_columns(self, sparse):
        # GH18914
        df = DataFrame.from_dict({"GDP": [1, 2], "Nation": ["AB", "CD"]})
        df = get_dummies(df, columns=["Nation"], sparse=sparse)
        df2 = df.reindex(columns=["GDP"])

        tm.assert_frame_equal(df[["GDP"]], df2)

    def test_get_dummies_duplicate_columns(self, df):
        # GH20839
        df.columns = ["A", "A", "A"]
        result = get_dummies(df).sort_index(axis=1)

        expected = DataFrame(
            [
                [1, True, False, True, False],
                [2, False, True, True, False],
                [3, True, False, False, True],
            ],
            columns=["A", "A_a", "A_b", "A_b", "A_c"],
        ).sort_index(axis=1)

        expected = expected.astype({"A": np.int64})

        tm.assert_frame_equal(result, expected)

    def test_get_dummies_all_sparse(self):
        df = DataFrame({"A": [1, 2]})
        result = get_dummies(df, columns=["A"], sparse=True)
        dtype = SparseDtype("bool", False)
        expected = DataFrame(
            {
                "A_1": SparseArray([1, 0], dtype=dtype),
                "A_2": SparseArray([0, 1], dtype=dtype),
            }
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("values", ["baz"])
    def test_get_dummies_with_string_values(self, values):
        # issue #28383
        df = DataFrame(
            {
                "bar": [1, 2, 3, 4, 5, 6],
                "foo": ["one", "one", "one", "two", "two", "two"],
                "baz": ["A", "B", "C", "A", "B", "C"],
                "zoo": ["x", "y", "z", "q", "w", "t"],
            }
        )

        msg = "Input must be a list-like for parameter `columns`"

        with pytest.raises(TypeError, match=msg):
            get_dummies(df, columns=values)

    def test_get_dummies_ea_dtype_series(self, any_numeric_ea_and_arrow_dtype):
        # GH#32430
        ser = Series(list("abca"))
        result = get_dummies(ser, dtype=any_numeric_ea_and_arrow_dtype)
        expected = DataFrame(
            {"a": [1, 0, 0, 1], "b": [0, 1, 0, 0], "c": [0, 0, 1, 0]},
            dtype=any_numeric_ea_and_arrow_dtype,
        )
        tm.assert_frame_equal(result, expected)

    def test_get_dummies_ea_dtype_dataframe(self, any_numeric_ea_and_arrow_dtype):
        # GH#32430
        df = DataFrame({"x": list("abca")})
        result = get_dummies(df, dtype=any_numeric_ea_and_arrow_dtype)
        expected = DataFrame(
            {"x_a": [1, 0, 0, 1], "x_b": [0, 1, 0, 0], "x_c": [0, 0, 1, 0]},
            dtype=any_numeric_ea_and_arrow_dtype,
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dtype_type", ["string", "category"])
    def test_get_dummies_ea_dtype(self, dtype_type, string_dtype_no_object):
        # GH#56273
        dtype = string_dtype_no_object
        exp_dtype = "boolean" if dtype.na_value is pd.NA else "bool"
        if dtype_type == "category":
            dtype = CategoricalDtype(Index(["a"], dtype))
        df = DataFrame({"name": Series(["a"], dtype=dtype), "x": 1})
        result = get_dummies(df)
        expected = DataFrame({"x": 1, "name_a": Series([True], dtype=exp_dtype)})
        tm.assert_frame_equal(result, expected)

    @td.skip_if_no("pyarrow")
    def test_get_dummies_arrow_dtype(self):
        # GH#56273
        df = DataFrame({"name": Series(["a"], dtype=ArrowDtype(pa.string())), "x": 1})
        result = get_dummies(df)
        expected = DataFrame({"x": 1, "name_a": Series([True], dtype="bool[pyarrow]")})
        tm.assert_frame_equal(result, expected)

        df = DataFrame(
            {
                "name": Series(
                    ["a"],
                    dtype=CategoricalDtype(Index(["a"], dtype=ArrowDtype(pa.string()))),
                ),
                "x": 1,
            }
        )
        result = get_dummies(df)
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_evalf.py ===
import math

from sympy.concrete.products import (Product, product)
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.evalf import N
from sympy.core.function import (Function, nfloat)
from sympy.core.mul import Mul
from sympy.core import (GoldenRatio)
from sympy.core.numbers import (AlgebraicNumber, E, Float, I, Rational,
                                oo, zoo, nan, pi)
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.combinatorial.numbers import fibonacci
from sympy.functions.elementary.complexes import (Abs, re, im)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, cosh)
from sympy.functions.elementary.integers import (ceiling, floor)
from sympy.functions.elementary.miscellaneous import (Max, sqrt)
from sympy.functions.elementary.trigonometric import (acos, atan, cos, sin, tan)
from sympy.integrals.integrals import (Integral, integrate)
from sympy.polys.polytools import factor
from sympy.polys.rootoftools import CRootOf
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.printing import srepr
from sympy.printing.str import sstr
from sympy.simplify.simplify import simplify
from sympy.core.numbers import comp
from sympy.core.evalf import (complex_accuracy, PrecisionExhausted,
                              scaled_zero, get_integer_part, as_mpmath, evalf, _evalf_with_bounded_error)
from mpmath import inf, ninf, make_mpc
from mpmath.libmp.libmpf import from_float, fzero
from sympy.core.expr import unchanged
from sympy.testing.pytest import raises, XFAIL
from sympy.abc import n, x, y


def NS(e, n=15, **options):
    return sstr(sympify(e).evalf(n, **options), full_prec=True)


def test_evalf_helpers():
    from mpmath.libmp import finf
    assert complex_accuracy((from_float(2.0), None, 35, None)) == 35
    assert complex_accuracy((from_float(2.0), from_float(10.0), 35, 100)) == 37
    assert complex_accuracy(
        (from_float(2.0), from_float(1000.0), 35, 100)) == 43
    assert complex_accuracy((from_float(2.0), from_float(10.0), 100, 35)) == 35
    assert complex_accuracy(
        (from_float(2.0), from_float(1000.0), 100, 35)) == 35
    assert complex_accuracy(finf) == math.inf
    assert complex_accuracy(zoo) == math.inf
    raises(ValueError, lambda: get_integer_part(zoo, 1, {}))


def test_evalf_basic():
    assert NS('pi', 15) == '3.14159265358979'
    assert NS('2/3', 10) == '0.6666666667'
    assert NS('355/113-pi', 6) == '2.66764e-7'
    assert NS('16*atan(1/5)-4*atan(1/239)', 15) == '3.14159265358979'


def test_cancellation():
    assert NS(Add(pi, Rational(1, 10**1000), -pi, evaluate=False), 15,
              maxn=1200) == '1.00000000000000e-1000'


def test_evalf_powers():
    assert NS('pi**(10**20)', 10) == '1.339148777e+49714987269413385435'
    assert NS(pi**(10**100), 10) == ('4.946362032e+4971498726941338543512682882'
          '9089887365167832438044244613405349992494711208'
          '95526746555473864642912223')
    assert NS('2**(1/10**50)', 15) == '1.00000000000000'
    assert NS('2**(1/10**50)-1', 15) == '6.93147180559945e-51'

# Evaluation of Rump's ill-conditioned polynomial


def test_evalf_rump():
    a = 1335*y**6/4 + x**2*(11*x**2*y**2 - y**6 - 121*y**4 - 2) + 11*y**8/2 + x/(2*y)
    assert NS(a, 15, subs={x: 77617, y: 33096}) == '-0.827396059946821'


def test_evalf_complex():
    assert NS('2*sqrt(pi)*I', 10) == '3.544907702*I'
    assert NS('3+3*I', 15) == '3.00000000000000 + 3.00000000000000*I'
    assert NS('E+pi*I', 15) == '2.71828182845905 + 3.14159265358979*I'
    assert NS('pi * (3+4*I)', 15) == '9.42477796076938 + 12.5663706143592*I'
    assert NS('I*(2+I)', 15) == '-1.00000000000000 + 2.00000000000000*I'


@XFAIL
def test_evalf_complex_bug():
    assert NS('(pi+E*I)*(E+pi*I)', 15) in ('0.e-15 + 17.25866050002*I',
              '0.e-17 + 17.25866050002*I', '-0.e-17 + 17.25866050002*I')


def test_evalf_complex_powers():
    assert NS('(E+pi*I)**100000000000000000') == \
        '-3.58896782867793e+61850354284995199 + 4.58581754997159e+61850354284995199*I'
    # XXX: rewrite if a+a*I simplification introduced in SymPy
    #assert NS('(pi + pi*I)**2') in ('0.e-15 + 19.7392088021787*I', '0.e-16 + 19.7392088021787*I')
    assert NS('(pi + pi*I)**2', chop=True) == '19.7392088021787*I'
    assert NS(
        '(pi + 1/10**8 + pi*I)**2') == '6.2831853e-8 + 19.7392088650106*I'
    assert NS('(pi + 1/10**12 + pi*I)**2') == '6.283e-12 + 19.7392088021850*I'
    assert NS('(pi + pi*I)**4', chop=True) == '-389.636364136010'
    assert NS(
        '(pi + 1/10**8 + pi*I)**4') == '-389.636366616512 + 2.4805021e-6*I'
    assert NS('(pi + 1/10**12 + pi*I)**4') == '-389.636364136258 + 2.481e-10*I'
    assert NS(
        '(10000*pi + 10000*pi*I)**4', chop=True) == '-3.89636364136010e+18'


@XFAIL
def test_evalf_complex_powers_bug():
    assert NS('(pi + pi*I)**4') == '-389.63636413601 + 0.e-14*I'


def test_evalf_exponentiation():
    assert NS(sqrt(-pi)) == '1.77245385090552*I'
    assert NS(Pow(pi*I, Rational(
        1, 2), evaluate=False)) == '1.25331413731550 + 1.25331413731550*I'
    assert NS(pi**I) == '0.413292116101594 + 0.910598499212615*I'
    assert NS(pi**(E + I/3)) == '20.8438653991931 + 8.36343473930031*I'
    assert NS((pi + I/3)**(E + I/3)) == '17.2442906093590 + 13.6839376767037*I'
    assert NS(exp(pi)) == '23.1406926327793'
    assert NS(exp(pi + E*I)) == '-21.0981542849657 + 9.50576358282422*I'
    assert NS(pi**pi) == '36.4621596072079'
    assert NS((-pi)**pi) == '-32.9138577418939 - 15.6897116534332*I'
    assert NS((-pi)**(-pi)) == '-0.0247567717232697 + 0.0118013091280262*I'

# An example from Smith, "Multiple Precision Complex Arithmetic and Functions"


def test_evalf_complex_cancellation():
    A = Rational('63287/100000')
    B = Rational('52498/100000')
    C = Rational('69301/100000')
    D = Rational('83542/100000')
    F = Rational('2231321613/2500000000')
    # XXX: the number of returned mantissa digits in the real part could
    # change with the implementation. What matters is that the returned digits are
    # correct; those that are showing now are correct.
    # >>> ((A+B*I)*(C+D*I)).expand()
    # 64471/10000000000 + 2231321613*I/2500000000
    # >>> 2231321613*4
    # 8925286452L
    assert NS((A + B*I)*(C + D*I), 6) == '6.44710e-6 + 0.892529*I'
    assert NS((A + B*I)*(C + D*I), 10) == '6.447100000e-6 + 0.8925286452*I'
    assert NS((A + B*I)*(
        C + D*I) - F*I, 5) in ('6.4471e-6 + 0.e-14*I', '6.4471e-6 - 0.e-14*I')


def test_evalf_logs():
    assert NS("log(3+pi*I)", 15) == '1.46877619736226 + 0.808448792630022*I'
    assert NS("log(pi*I)", 15) == '1.14472988584940 + 1.57079632679490*I'
    assert NS('log(-1 + 0.00001)', 2) == '-1.0e-5 + 3.1*I'
    assert NS('log(100, 10, evaluate=False)', 15) == '2.00000000000000'
    assert NS('-2*I*log(-(-1)**(S(1)/9))', 15) == '-5.58505360638185'


def test_evalf_trig():
    assert NS('sin(1)', 15) == '0.841470984807897'
    assert NS('cos(1)', 15) == '0.540302305868140'
    assert NS('tan(1)', 15) == '1.55740772465490'
    assert NS('sin(10**-6)', 15) == '9.99999999999833e-7'
    assert NS('cos(10**-6)', 15) == '0.999999999999500'
    assert NS('tan(10**-6)', 15) == '1.00000000000033e-6'
    assert NS('sin(E*10**100)', 15) == '0.409160531722613'
    assert NS('tan(I)',15) =='0.761594155955765*I'
    assert NS('tan(1000*I)',15)== '1.00000000000000*I'
    # Some input near roots
    assert NS(sin(exp(pi*sqrt(163))*pi), 15) == '-2.35596641936785e-12'
    assert NS(sin(pi*10**100 + Rational(7, 10**5), evaluate=False), 15, maxn=120) == \
        '6.99999999428333e-5'
    assert NS(sin(Rational(7, 10**5), evaluate=False), 15) == \
        '6.99999999428333e-5'

# Check detection of various false identities


def test_evalf_near_integers():
    # Binet's formula
    f = lambda n: ((1 + sqrt(5))**n)/(2**n * sqrt(5))
    assert NS(f(5000) - fibonacci(5000), 10, maxn=1500) == '5.156009964e-1046'
    # Some near-integer identities from
    # http://mathworld.wolfram.com/AlmostInteger.html
    assert NS('sin(2017*2**(1/5))', 15) == '-1.00000000000000'
    assert NS('sin(2017*2**(1/5))', 20) == '-0.99999999999999997857'
    assert NS('1+sin(2017*2**(1/5))', 15) == '2.14322287389390e-17'
    assert NS('45 - 613*E/37 + 35/991', 15) == '6.03764498766326e-11'


def test_evalf_ramanujan():
    assert NS(exp(pi*sqrt(163)) - 640320**3 - 744, 10) == '-7.499274028e-13'
    # A related identity
    A = 262537412640768744*exp(-pi*sqrt(163))
    B = 196884*exp(-2*pi*sqrt(163))
    C = 103378831900730205293632*exp(-3*pi*sqrt(163))
    assert NS(1 - A - B + C, 10) == '1.613679005e-59'

# Input that for various reasons have failed at some point


def test_evalf_bugs():
    assert NS(sin(1) + exp(-10**10), 10) == NS(sin(1), 10)
    assert NS(exp(10**10) + sin(1), 10) == NS(exp(10**10), 10)
    assert NS('expand_log(log(1+1/10**50))', 20) == '1.0000000000000000000e-50'
    assert NS('log(10**100,10)', 10) == '100.0000000'
    assert NS('log(2)', 10) == '0.6931471806'
    assert NS(
        '(sin(x)-x)/x**3', 15, subs={x: '1/10**50'}) == '-0.166666666666667'
    assert NS(sin(1) + Rational(
        1, 10**100)*I, 15) == '0.841470984807897 + 1.00000000000000e-100*I'
    assert x.evalf() == x
    assert NS((1 + I)**2*I, 6) == '-2.00000'
    d = {n: (
        -1)**Rational(6, 7), y: (-1)**Rational(4, 7), x: (-1)**Rational(2, 7)}
    assert NS((x*(1 + y*(1 + n))).subs(d).evalf(), 6) == '0.346011 + 0.433884*I'
    assert NS(((-I - sqrt(2)*I)**2).evalf()) == '-5.82842712474619'
    assert NS((1 + I)**2*I, 15) == '-2.00000000000000'
    # issue 4758 (1/2):
    assert NS(pi.evalf(69) - pi) == '-4.43863937855894e-71'
    # issue 4758 (2/2): With the bug present, this still only fails if the
    # terms are in the order given here. This is not generally the case,
    # because the order depends on the hashes of the terms.
    assert NS(20 - 5008329267844*n**25 - 477638700*n**37 - 19*n,
              subs={n: .01}) == '19.8100000000000'
    assert NS(((x - 1)*(1 - x)**1000).n()
              ) == '(1.00000000000000 - x)**1000*(x - 1.00000000000000)'
    assert NS((-x).n()) == '-x'
    assert NS((-2*x).n()) == '-2.00000000000000*x'
    assert NS((-2*x*y).n()) == '-2.00000000000000*x*y'
    assert cos(x).n(subs={x: 1+I}) == cos(x).subs(x, 1+I).n()
    # issue 6660. Also NaN != mpmath.nan
    # In this order:
    # 0*nan, 0/nan, 0*inf, 0/inf
    # 0+nan, 0-nan, 0+inf, 0-inf
    # >>> n = Some Number
    # n*nan, n/nan, n*inf, n/inf
    # n+nan, n-nan, n+inf, n-inf
    assert (0*E**(oo)).n() is S.NaN
    assert (0/E**(oo)).n() is S.Zero

    assert (0+E**(oo)).n() is S.Infinity
    assert (0-E**(oo)).n() is S.NegativeInfinity

    assert (5*E**(oo)).n() is S.Infinity
    assert (5/E**(oo)).n() is S.Zero

    assert (5+E**(oo)).n() is S.Infinity
    assert (5-E**(oo)).n() is S.NegativeInfinity

    #issue 7416
    assert as_mpmath(0.0, 10, {'chop': True}) == 0

    #issue 5412
    assert ((oo*I).n() == S.Infinity*I)
    assert ((oo+oo*I).n() == S.Infinity + S.Infinity*I)

    #issue 11518
    assert NS(2*x**2.5, 5) == '2.0000*x**2.5000'

    #issue 13076
    assert NS(Mul(Max(0, y), x, evaluate=False).evalf()) == 'x*Max(0, y)'

    #issue 18516
    assert NS(log(S(3273390607896141870013189696827599152216642046043064789483291368096133796404674554883270092325904157150886684127560071009217256545885393053328527589376)/36360291795869936842385267079543319118023385026001623040346035832580600191583895484198508262979388783308179702534403855752855931517013066142992430916562025780021771247847643450125342836565813209972590371590152578728008385990139795377610001).evalf(15, chop=True)) == '-oo'


def test_evalf_integer_parts():
    a = floor(log(8)/log(2) - exp(-1000), evaluate=False)
    b = floor(log(8)/log(2), evaluate=False)
    assert a.evalf() == 3.0
    assert b.evalf() == 3.0
    # equals, as a fallback, can still fail but it might succeed as here
    assert ceiling(10*(sin(1)**2 + cos(1)**2)) == 10

    assert int(floor(factorial(50)/E, evaluate=False).evalf(70)) == \
        int(11188719610782480504630258070757734324011354208865721592720336800)
    assert int(ceiling(factorial(50)/E, evaluate=False).evalf(70)) == \
        int(11188719610782480504630258070757734324011354208865721592720336801)
    assert int(floor(GoldenRatio**999 / sqrt(5) + S.Half)
               .evalf(1000)) == fibonacci(999)
    assert int(floor(GoldenRatio**1000 / sqrt(5) + S.Half)
               .evalf(1000)) == fibonacci(1000)

    assert ceiling(x).evalf(subs={x: 3}) == 3.0
    assert ceiling(x).evalf(subs={x: 3*I}) == 3.0*I
    assert ceiling(x).evalf(subs={x: 2 + 3*I}) == 2.0 + 3.0*I
    assert ceiling(x).evalf(subs={x: 3.}) == 3.0
    assert ceiling(x).evalf(subs={x: 3.*I}) == 3.0*I
    assert ceiling(x).evalf(subs={x: 2. + 3*I}) == 2.0 + 3.0*I

    assert float((floor(1.5, evaluate=False)+1/9).evalf()) == 1 + 1/9
    assert float((floor(0.5, evaluate=False)+20).evalf()) == 20

    # issue 19991
    n = 1169809367327212570704813632106852886389036911
    r = 744723773141314414542111064094745678855643068

    assert floor(n / (pi / 2)) == r
    assert floor(80782 * sqrt(2)) == 114242

    # issue 20076
    assert 260515 - floor(260515/pi + 1/2) * pi == atan(tan(260515))

    assert floor(x).evalf(subs={x: sqrt(2)}) == 1.0


def test_evalf_trig_zero_detection():
    a = sin(160*pi, evaluate=False)
    t = a.evalf(maxn=100)
    assert abs(t) < 1e-100
    assert t._prec < 2
    assert a.evalf(chop=True) == 0
    raises(PrecisionExhausted, lambda: a.evalf(strict=True))


def test_evalf_sum():
    assert Sum(n,(n,1,2)).evalf() == 3.
    assert Sum(n,(n,1,2)).doit().evalf() == 3.
    # the next test should return instantly
    assert Sum(1/n,(n,1,2)).evalf() == 1.5

    # issue 8219
    assert Sum(E/factorial(n), (n, 0, oo)).evalf() == (E*E).evalf()
    # issue 8254
    assert Sum(2**n*n/factorial(n), (n, 0, oo)).evalf() == (2*E*E).evalf()
    # issue 8411
    s = Sum(1/x**2, (x, 100, oo))
    assert s.n() == s.doit().n()


def test_evalf_divergent_series():
    raises(ValueError, lambda: Sum(1/n, (n, 1, oo)).evalf())
    raises(ValueError, lambda: Sum(n/(n**2 + 1), (n, 1, oo)).evalf())
    raises(ValueError, lambda: Sum((-1)**n, (n, 1, oo)).evalf())
    raises(ValueError, lambda: Sum((-1)**n, (n, 1, oo)).evalf())
    raises(ValueError, lambda: Sum(n**2, (n, 1, oo)).evalf())
    raises(ValueError, lambda: Sum(2**n, (n, 1, oo)).evalf())
    raises(ValueError, lambda: Sum((-2)**n, (n, 1, oo)).evalf())
    raises(ValueError, lambda: Sum((2*n + 3)/(3*n**2 + 4), (n, 0, oo)).evalf())
    raises(ValueError, lambda: Sum((0.5*n**3)/(n**4 + 1), (n, 0, oo)).evalf())


def test_evalf_product():
    assert Product(n, (n, 1, 10)).evalf() == 3628800.
    assert comp(Product(1 - S.Half**2/n**2, (n, 1, oo)).n(5), 0.63662)
    assert Product(n, (n, -1, 3)).evalf() == 0


def test_evalf_py_methods():
    assert abs(float(pi + 1) - 4.1415926535897932) < 1e-10
    assert abs(complex(pi + 1) - 4.1415926535897932) < 1e-10
    assert abs(
        complex(pi + E*I) - (3.1415926535897931 + 2.7182818284590451j)) < 1e-10
    raises(TypeError, lambda: float(pi + x))


def test_evalf_power_subs_bugs():
    assert (x**2).evalf(subs={x: 0}) == 0
    assert sqrt(x).evalf(subs={x: 0}) == 0
    assert (x**Rational(2, 3)).evalf(subs={x: 0}) == 0
    assert (x**x).evalf(subs={x: 0}) == 1.0
    assert (3**x).evalf(subs={x: 0}) == 1.0
    assert exp(x).evalf(subs={x: 0}) == 1.0
    assert ((2 + I)**x).evalf(subs={x: 0}) == 1.0
    assert (0**x).evalf(subs={x: 0}) == 1.0


def test_evalf_arguments():
    raises(TypeError, lambda: pi.evalf(method="garbage"))


def test_implemented_function_evalf():
    from sympy.utilities.lambdify import implemented_function
    f = Function('f')
    f = implemented_function(f, lambda x: x + 1)
    assert str(f(x)) == "f(x)"
    assert str(f(2)) == "f(2)"
    assert f(2).evalf() == 3.0
    assert f(x).evalf() == f(x)
    f = implemented_function(Function('sin'), lambda x: x + 1)
    assert f(2).evalf() != sin(2)
    del f._imp_     # XXX: due to caching _imp_ would influence all other tests


def test_evaluate_false():
    for no in [0, False]:
        assert Add(3, 2, evaluate=no).is_Add
        assert Mul(3, 2, evaluate=no).is_Mul
        assert Pow(3, 2, evaluate=no).is_Pow
    assert Pow(y, 2, evaluate=True) - Pow(y, 2, evaluate=True) == 0


def test_evalf_relational():
    assert Eq(x/5, y/10).evalf() == Eq(0.2*x, 0.1*y)
    # if this first assertion fails it should be replaced with
    # one that doesn't
    assert unchanged(Eq, (3 - I)**2/2 + I, 0)
    assert Eq((3 - I)**2/2 + I, 0).n() is S.false
    assert nfloat(Eq((3 - I)**2 + I, 0)) == S.false


def test_issue_5486():
    assert not cos(sqrt(0.5 + I)).n().is_Function


def test_issue_5486_bug():
    from sympy.core.expr import Expr
    from sympy.core.numbers import I
    assert abs(Expr._from_mpmath(I._to_mpmath(15), 15) - I) < 1.0e-15


def test_bugs():
    from sympy.functions.elementary.complexes import (polar_lift, re)

    assert abs(re((1 + I)**2)) < 1e-15

    # anything that evalf's to 0 will do in place of polar_lift
    assert abs(polar_lift(0)).n() == 0


def test_subs():
    assert NS('besseli(-x, y) - besseli(x, y)', subs={x: 3.5, y: 20.0}) == \
        '-4.92535585957223e-10'
    assert NS('Piecewise((x, x>0)) + Piecewise((1-x, x>0))', subs={x: 0.1}) == \
        '1.00000000000000'
    raises(TypeError, lambda: x.evalf(subs=(x, 1)))


def test_issue_4956_5204():
    # issue 4956
    v = S('''(-27*12**(1/3)*sqrt(31)*I +
    27*2**(2/3)*3**(1/3)*sqrt(31)*I)/(-2511*2**(2/3)*3**(1/3) +
    (29*18**(1/3) + 9*2**(1/3)*3**(2/3)*sqrt(31)*I +
    87*2**(1/3)*3**(1/6)*I)**2)''')
    assert NS(v, 1) == '0.e-118 - 0.e-118*I'

    # issue 5204
    v = S('''-(357587765856 + 18873261792*249**(1/2) + 56619785376*I*83**(1/2) +
    108755765856*I*3**(1/2) + 41281887168*6**(1/3)*(1422 +
    54*249**(1/2))**(1/3) - 1239810624*6**(1/3)*249**(1/2)*(1422 +
    54*249**(1/2))**(1/3) - 3110400000*I*6**(1/3)*83**(1/2)*(1422 +
    54*249**(1/2))**(1/3) + 13478400000*I*3**(1/2)*6**(1/3)*(1422 +
    54*249**(1/2))**(1/3) + 1274950152*6**(2/3)*(1422 +
    54*249**(1/2))**(2/3) + 32347944*6**(2/3)*249**(1/2)*(1422 +
    54*249**(1/2))**(2/3) - 1758790152*I*3**(1/2)*6**(2/3)*(1422 +
    54*249**(1/2))**(2/3) - 304403832*I*6**(2/3)*83**(1/2)*(1422 +
    4*249**(1/2))**(2/3))/(175732658352 + (1106028 + 25596*249**(1/2) +
    76788*I*83**(1/2))**2)''')
    assert NS(v, 5) == '0.077284 + 1.1104*I'
    assert NS(v, 1) == '0.08 + 1.*I'


def test_old_docstring():
    a = (E + pi*I)*(E - pi*I)
    assert NS(a) == '17.2586605000200'
    assert a.n() == 17.25866050002001


def test_issue_4806():
    assert integrate(atan(x)**2, (x, -1, 1)).evalf().round(1) == Float(0.5, 1)
    assert atan(0, evaluate=False).n() == 0


def test_evalf_mul():
    # SymPy should not try to expand this; it should be handled term-wise
    # in evalf through mpmath
    assert NS(product(1 + sqrt(n)*I, (n, 1, 500)), 1) == '5.e+567 + 2.e+568*I'


def test_scaled_zero():
    a, b = (([0], 1, 100, 1), -1)
    assert scaled_zero(100) == (a, b)
    assert scaled_zero(a) == (0, 1, 100, 1)
    a, b = (([1], 1, 100, 1), -1)
    assert scaled_zero(100, -1) == (a, b)
    assert scaled_zero(a) == (1, 1, 100, 1)
    raises(ValueError, lambda: scaled_zero(scaled_zero(100)))
    raises(ValueError, lambda: scaled_zero(100, 2))
    raises(ValueError, lambda: scaled_zero(100, 0))
    raises(ValueError, lambda: scaled_zero((1, 5, 1, 3)))


def test_chop_value():
    for i in range(-27, 28):
        assert (Pow(10, i)*2).n(chop=10**i) and not (Pow(10, i)).n(chop=10**i)


def test_infinities():
    assert oo.evalf(chop=True) == inf
    assert (-oo).evalf(chop=True) == ninf


def test_to_mpmath():
    assert sqrt(3)._to_mpmath(20)._mpf_ == (0, int(908093), -19, 20)
    assert S(3.2)._to_mpmath(20)._mpf_ == (0, int(838861), -18, 20)


def test_issue_6632_evalf():
    add = (-100000*sqrt(2500000001) + 5000000001)
    assert add.n() == 9.999999998e-11
    assert (add*add).n() == 9.999999996e-21


def test_issue_4945():
    from sympy.abc import H
    assert (H/0).evalf(subs={H:1}) == zoo


def test_evalf_integral():
    # test that workprec has to increase in order to get a result other than 0
    eps = Rational(1, 1000000)
    assert Integral(sin(x), (x, -pi, pi + eps)).n(2)._prec == 10


def test_issue_8821_highprec_from_str():
    s = str(pi.evalf(128))
    p = N(s)
    assert Abs(sin(p)) < 1e-15
    p = N(s, 64)
    assert Abs(sin(p)) < 1e-64


def test_issue_8853():
    p = Symbol('x', even=True, positive=True)
    assert floor(-p - S.Half).is_even == False
    assert floor(-p + S.Half).is_even == True
    assert ceiling(p - S.Half).is_even == True
    assert ceiling(p + S.Half).is_even == False

    assert get_integer_part(S.Half, -1, {}, True) == (0, 0)
    assert get_integer_part(S.Half, 1, {}, True) == (1, 0)
    assert get_integer_part(Rational(-1, 2), -1, {}, True) == (-1, 0)
    assert get_integer_part(Rational(-1, 2), 1, {}, True) == (0, 0)


def test_issue_17681():
    class identity_func(Function):

        def _eval_evalf(self, *args, **kwargs):
            return self.args[0].evalf(*args, **kwargs)

    assert floor(identity_func(S(0))) == 0
    assert get_integer_part(S(0), 1, {}, True) == (0, 0)


def test_issue_9326():
    from sympy.core.symbol import Dummy
    d1 = Dummy('d')
    d2 = Dummy('d')
    e = d1 + d2
    assert e.evalf(subs = {d1: 1, d2: 2}) == 3.0


def test_issue_10323():
    assert ceiling(sqrt(2**30 + 1)) == 2**15 + 1


def test_AssocOp_Function():
    # the first arg of Min is not comparable in the imaginary part
    raises(ValueError, lambda: S('''
    Min(-sqrt(3)*cos(pi/18)/6 + re(1/((-1/2 - sqrt(3)*I/2)*(1/6 +
    sqrt(3)*I/18)**(1/3)))/3 + sin(pi/18)/2 + 2 + I*(-cos(pi/18)/2 -
    sqrt(3)*sin(pi/18)/6 + im(1/((-1/2 - sqrt(3)*I/2)*(1/6 +
    sqrt(3)*I/18)**(1/3)))/3), re(1/((-1/2 + sqrt(3)*I/2)*(1/6 +
    sqrt(3)*I/18)**(1/3)))/3 - sqrt(3)*cos(pi/18)/6 - sin(pi/18)/2 + 2 +
    I*(im(1/((-1/2 + sqrt(3)*I/2)*(1/6 + sqrt(3)*I/18)**(1/3)))/3 -
    sqrt(3)*sin(pi/18)/6 + cos(pi/18)/2))'''))
    # if that is changed so a non-comparable number remains as
    # an arg, then the Min/Max instantiation needs to be changed
    # to watch out for non-comparable args when making simplifications
    # and the following test should be added instead (with e being
    # the sympified expression above):
    # raises(ValueError, lambda: e._eval_evalf(2))


def test_issue_10395():
    eq = x*Max(0, y)
    assert nfloat(eq) == eq
    eq = x*Max(y, -1.1)
    assert nfloat(eq) == eq
    assert Max(y, 4).n() == Max(4.0, y)


def test_issue_13098():
    assert floor(log(S('9.'+'9'*20), 10)) == 0
    assert ceiling(log(S('9.'+'9'*20), 10)) == 1
    assert floor(log(20 - S('9.'+'9'*20), 10)) == 1
    assert ceiling(log(20 - S('9.'+'9'*20), 10)) == 2


def test_issue_14601():
    e = 5*x*y/2 - y*(35*(x**3)/2 - 15*x/2)
    subst = {x:0.0, y:0.0}
    e2 = e.evalf(subs=subst)
    assert float(e2) == 0.0
    assert float((x + x*(x**2 + x)).evalf(subs={x: 0.0})) == 0.0


def test_issue_11151():
    z = S.Zero
    e = Sum(z, (x, 1, 2))
    assert e != z  # it shouldn't evaluate
    # when it does evaluate, this is what it should give
    assert evalf(e, 15, {}) == \
        evalf(z, 15, {}) == (None, None, 15, None)
    # so this shouldn't fail
    assert (e/2).n() == 0
    # this was where the issue appeared
    expr0 = Sum(x**2 + x, (x, 1, 2))
    expr1 = Sum(0, (x, 1, 2))
    expr2 = expr1/expr0
    assert simplify(factor(expr2) - expr2) == 0


def test_issue_13425():
    assert N('2**.5', 30) == N('sqrt(2)', 30)
    assert N('x - x', 30) == 0
    assert abs((N('pi*.1', 22)*10 - pi).n()) < 1e-22


def test_issue_17421():
    assert N(acos(-I + acosh(cosh(cosh(1) + I)))) == 1.0*I


def test_issue_20291():
    from sympy.sets import EmptySet, Reals
    from sympy.sets.sets import (Complement, FiniteSet, Intersection)
    a = Symbol('a')
    b = Symbol('b')
    A = FiniteSet(a, b)
    assert A.evalf(subs={a: 1, b: 2}) == FiniteSet(1.0, 2.0)
    B = FiniteSet(a-b, 1)
    assert B.evalf(subs={a: 1, b: 2}) == FiniteSet(-1.0, 1.0)

    sol = Complement(Intersection(FiniteSet(-b/2 - sqrt(b**2-4*pi)/2), Reals), FiniteSet(0))
    assert sol.evalf(subs={b: 1}) == EmptySet


def test_evalf_with_zoo():
    assert (1/x).evalf(subs={x: 0}) == zoo  # issue 8242
    assert (-1/x).evalf(subs={x: 0}) == zoo  # PR 16150
    assert (0 ** x).evalf(subs={x: -1}) == zoo  # PR 16150
    assert (0 ** x).evalf(subs={x: -1 + I}) == nan
    assert Mul(2, Pow(0, -1, evaluate=False), evaluate=False).evalf() == zoo  # issue 21147
    assert Mul(x, 1/x, evaluate=False).evalf(subs={x: 0}) == Mul(x, 1/x, evaluate=False).subs(x, 0) == nan
    assert Mul(1/x, 1/x, evaluate=False).evalf(subs={x: 0}) == zoo
    assert Mul(1/x, Abs(1/x), evaluate=False).evalf(subs={x: 0}) == zoo
    assert Abs(zoo, evaluate=False).evalf() == oo
    assert re(zoo, evaluate=False).evalf() == nan
    assert im(zoo, evaluate=False).evalf() == nan
    assert Add(zoo, zoo, evaluate=False).evalf() == nan
    assert Add(oo, zoo, evaluate=False).evalf() == nan
    assert Pow(zoo, -1, evaluate=False).evalf() == 0
    assert Pow(zoo, Rational(-1, 3), evaluate=False).evalf() == 0
    assert Pow(zoo, Rational(1, 3), evaluate=False).evalf() == zoo
    assert Pow(zoo, S.Half, evaluate=False).evalf() == zoo
    assert Pow(zoo, 2, evaluate=False).evalf() == zoo
    assert Pow(0, zoo, evaluate=False).evalf() == nan
    assert log(zoo, evaluate=False).evalf() == zoo
    assert zoo.evalf(chop=True) == zoo
    assert x.evalf(subs={x: zoo}) == zoo


def test_evalf_with_bounded_error():
    cases = [
        # zero
        (Rational(0), None, 1),
        # zero im part
        (pi, None, 10),
        # zero real part
        (pi*I, None, 10),
        # re and im nonzero
        (2-3*I, None, 5),
        # similar tests again, but using eps instead of m
        (Rational(0), Rational(1, 2), None),
        (pi, Rational(1, 1000), None),
        (pi * I, Rational(1, 1000), None),
        (2 - 3 * I, Rational(1, 1000), None),
        # very large eps
        (2 - 3 * I, Rational(1000), None),
        # case where x already small, hence some cancellation in p = m + n - 1
        (Rational(1234, 10**8), Rational(1, 10**12), None),
    ]
    for x0, eps, m in cases:
        a, b, _, _ = evalf(x0, 53, {})
        c, d, _, _ = _evalf_with_bounded_error(x0, eps, m)
        if eps is None:
            eps = 2**(-m)
        z = make_mpc((a or fzero, b or fzero))
        w = make_mpc((c or fzero, d or fzero))
        assert abs(w - z) < eps

    # eps must be positive
    raises(ValueError, lambda: _evalf_with_bounded_error(pi, Rational(0)))
    raises(ValueError, lambda: _evalf_with_bounded_error(pi, -pi))
    raises(ValueError, lambda: _evalf_with_bounded_error(pi, I))


def test_issue_22849():
    a = -8 + 3 * sqrt(3)
    x = AlgebraicNumber(a)
    assert evalf(a, 1, {}) == evalf(x, 1, {})


def test_evalf_real_alg_num():
    # This test demonstrates why the entry for `AlgebraicNumber` in
    # `sympy.core.evalf._create_evalf_table()` has to use `x.to_root()`,
    # instead of `x.as_expr()`. If the latter is used, then `z` will be
    # a complex number with `0.e-20` for imaginary part, even though `a5`
    # is a real number.
    zeta = Symbol('zeta')
    a5 = AlgebraicNumber(CRootOf(cyclotomic_poly(5), -1), [-1, -1, 0, 0], alias=zeta)
    z = a5.evalf()
    assert isinstance(z, Float)
    assert not hasattr(z, '_mpc_')
    assert hasattr(z, '_mpf_')


def test_issue_20733():
    expr = 1/((x - 9)*(x - 8)*(x - 7)*(x - 4)**2*(x - 3)**3*(x - 2))
    assert str(expr.evalf(1, subs={x:1})) == '-4.e-5'
    assert str(expr.evalf(2, subs={x:1})) == '-4.1e-5'
    assert str(expr.evalf(11, subs={x:1})) == '-4.1335978836e-5'
    assert str(expr.evalf(20, subs={x:1})) == '-0.000041335978835978835979'

    expr = Mul(*((x - i) for i in range(2, 1000)))
    assert srepr(expr.evalf(2, subs={x: 1})) == "Float('4.0271e+2561', precision=10)"
    assert srepr(expr.evalf(10, subs={x: 1})) == "Float('4.02790050126e+2561', precision=37)"
    assert srepr(expr.evalf(53, subs={x: 1})) == "Float('4.0279005012722099453824067459760158730668154575647110393e+2561', precision=179)"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\matrices\tests\test_rref.py ===
from sympy import ZZ, QQ, ZZ_I, EX, Matrix, eye, zeros, symbols
from sympy.polys.matrices import DM, DomainMatrix
from sympy.polys.matrices.dense import ddm_irref_den, ddm_irref
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.sdm import SDM, sdm_irref, sdm_rref_den

import pytest


#
# The dense and sparse implementations of rref_den are ddm_irref_den and
# sdm_irref_den. These can give results that differ by some factor and also
# give different results if the order of the rows is changed. The tests below
# show all results on lowest terms as should be returned by cancel_denom.
#
# The EX domain is also a case where the dense and sparse implementations
# can give results in different forms: the results should be equivalent but
# are not canonical because EX does not have a canonical form.
#


a, b, c, d = symbols('a, b, c, d')


qq_large_1 = DM([
[  (1,2),  (1,3),  (1,5),  (1,7), (1,11), (1,13), (1,17), (1,19), (1,23), (1,29), (1,31)],
[ (1,37), (1,41), (1,43), (1,47), (1,53), (1,59), (1,61), (1,67), (1,71), (1,73), (1,79)],
[ (1,83), (1,89), (1,97),(1,101),(1,103),(1,107),(1,109),(1,113),(1,127),(1,131),(1,137)],
[(1,139),(1,149),(1,151),(1,157),(1,163),(1,167),(1,173),(1,179),(1,181),(1,191),(1,193)],
[(1,197),(1,199),(1,211),(1,223),(1,227),(1,229),(1,233),(1,239),(1,241),(1,251),(1,257)],
[(1,263),(1,269),(1,271),(1,277),(1,281),(1,283),(1,293),(1,307),(1,311),(1,313),(1,317)],
[(1,331),(1,337),(1,347),(1,349),(1,353),(1,359),(1,367),(1,373),(1,379),(1,383),(1,389)],
[(1,397),(1,401),(1,409),(1,419),(1,421),(1,431),(1,433),(1,439),(1,443),(1,449),(1,457)],
[(1,461),(1,463),(1,467),(1,479),(1,487),(1,491),(1,499),(1,503),(1,509),(1,521),(1,523)],
[(1,541),(1,547),(1,557),(1,563),(1,569),(1,571),(1,577),(1,587),(1,593),(1,599),(1,601)],
[(1,607),(1,613),(1,617),(1,619),(1,631),(1,641),(1,643),(1,647),(1,653),(1,659),(1,661)]],
        QQ)

qq_large_2 = qq_large_1 + 10**100 * DomainMatrix.eye(11, QQ)


RREF_EXAMPLES = [
    (
        'zz_1',
         DM([[1, 2, 3]], ZZ),
         DM([[1, 2, 3]], ZZ),
         ZZ(1),
    ),

    (
        'zz_2',
         DomainMatrix([], (0, 0), ZZ),
         DomainMatrix([], (0, 0), ZZ),
         ZZ(1),
    ),

    (
        'zz_3',
        DM([[1, 2],
            [3, 4]], ZZ),
        DM([[1, 0],
            [0, 1]], ZZ),
        ZZ(1),
    ),

    (
        'zz_4',
        DM([[1, 0],
            [3, 4]], ZZ),
        DM([[1, 0],
            [0, 1]], ZZ),
        ZZ(1),
    ),

    (
        'zz_5',
        DM([[0, 2],
            [3, 4]], ZZ),
        DM([[1, 0],
            [0, 1]], ZZ),
        ZZ(1),
    ),

    (
        'zz_6',
        DM([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]], ZZ),
        DM([[1, 0, -1],
            [0, 1,  2],
            [0, 0,  0]], ZZ),
        ZZ(1),
    ),

    (
        'zz_7',
        DM([[0, 0, 0],
            [0, 0, 0],
            [1, 0, 0]], ZZ),
        DM([[1, 0, 0],
            [0, 0, 0],
            [0, 0, 0]], ZZ),
        ZZ(1),
    ),

    (
        'zz_8',
        DM([[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]], ZZ),
        DM([[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]], ZZ),
        ZZ(1),
    ),

    (
        'zz_9',
        DM([[1, 1, 0],
            [0, 0, 2],
            [0, 0, 0]], ZZ),
        DM([[1, 1, 0],
            [0, 0, 1],
            [0, 0, 0]], ZZ),
        ZZ(1),
    ),

    (
        'zz_10',
        DM([[2, 2, 0],
            [0, 0, 2],
            [0, 0, 0]], ZZ),
        DM([[1, 1, 0],
            [0, 0, 1],
            [0, 0, 0]], ZZ),
        ZZ(1),
    ),

    (
        'zz_11',
        DM([[2, 2, 0],
            [0, 2, 2],
            [0, 0, 2]], ZZ),
        DM([[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]], ZZ),
        ZZ(1),
    ),

    (
        'zz_12',
        DM([[ 1,  2,  3],
            [ 4,  5,  6],
            [ 7,  8,  9],
            [10, 11, 12]], ZZ),
        DM([[1,  0, -1],
            [0,  1,  2],
            [0,  0,  0],
            [0,  0,  0]], ZZ),
        ZZ(1),
    ),

    (
        'zz_13',
        DM([[ 1,  2,  3],
            [ 4,  5,  6],
            [ 7,  8,  9],
            [10, 11, 13]], ZZ),
        DM([[ 1,  0,  0],
            [ 0,  1,  0],
            [ 0,  0,  1],
            [ 0,  0,  0]], ZZ),
        ZZ(1),
    ),

    (
        'zz_14',
        DM([[1, 2,  4, 3],
            [4, 5, 10, 6],
            [7, 8, 16, 9]], ZZ),
        DM([[1, 0, 0, -1],
            [0, 1, 2,  2],
            [0, 0, 0,  0]], ZZ),
        ZZ(1),
    ),

    (
        'zz_15',
        DM([[1, 2,  4, 3],
            [4, 5, 10, 6],
            [7, 8, 17, 9]], ZZ),
        DM([[1, 0, 0, -1],
            [0, 1, 0,  2],
            [0, 0, 1,  0]], ZZ),
        ZZ(1),
    ),

    (
        'zz_16',
        DM([[1, 2, 0, 1],
            [1, 1, 9, 0]], ZZ),
        DM([[1, 0, 18, -1],
            [0, 1, -9,  1]], ZZ),
        ZZ(1),
    ),

    (
        'zz_17',
        DM([[1, 1, 1],
            [1, 2, 2]], ZZ),
        DM([[1, 0, 0],
            [0, 1, 1]], ZZ),
        ZZ(1),
    ),

    (
        # Here the sparse implementation and dense implementation give very
        # different denominators: 4061232 and -1765176.
        'zz_18',
        DM([[94, 24,  0, 27, 0],
            [79,  0,  0,  0, 0],
            [85, 16, 71, 81, 0],
            [ 0,  0, 72, 77, 0],
            [21,  0, 34,  0, 0]], ZZ),
        DM([[ 1,  0,  0,  0, 0],
            [ 0,  1,  0,  0, 0],
            [ 0,  0,  1,  0, 0],
            [ 0,  0,  0,  1, 0],
            [ 0,  0,  0,  0, 0]], ZZ),
        ZZ(1),
    ),

    (
        # Let's have a denominator that cannot be cancelled.
        'zz_19',
        DM([[1, 2, 4],
            [4, 5, 6]], ZZ),
        DM([[3, 0, -8],
            [0, 3, 10]], ZZ),
        ZZ(3),
    ),

    (
        'zz_20',
        DM([[0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 4]], ZZ),
        DM([[0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]], ZZ),
        ZZ(1),
    ),

    (
        'zz_21',
        DM([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]], ZZ),
        DM([[1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]], ZZ),
        ZZ(1),
    ),

    (
        'zz_22',
        DM([[1, 1, 1, 0, 1],
            [1, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [1, 1, 0, 1, 0],
            [1, 0, 0, 0, 0]], ZZ),
        DM([[1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0]], ZZ),
        ZZ(1),
    ),

    (
        'zz_large_1',
        DM([
[ 0,  0,  0, 81,  0,  0, 75,  0,  0,  0,  0,  0,  0, 27,  0,  0,  0,  0,  0,  0],
[ 0,  0,  0,  0,  0, 86,  0, 92, 79, 54,  0,  7,  0,  0,  0,  0, 79,  0,  0,  0],
[89, 54, 81,  0,  0, 20,  0,  0,  0,  0,  0,  0, 51,  0, 94,  0,  0, 77,  0,  0],
[ 0,  0,  0, 96,  0,  0,  0,  0,  0,  0,  0,  0, 48, 29,  0,  0,  5,  0, 32,  0],
[ 0, 70,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 60,  0,  0,  0, 11],
[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 37,  0, 43,  0,  0],
[ 0,  0,  0,  0,  0, 38, 91,  0,  0,  0,  0, 38,  0,  0,  0,  0,  0, 26,  0,  0],
[69,  0,  0,  0,  0,  0, 94,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 55],
[ 0, 13, 18, 49, 49, 88,  0,  0, 35, 54,  0,  0, 51,  0,  0,  0,  0,  0,  0, 87],
[ 0,  0,  0,  0, 31,  0, 40,  0,  0,  0,  0,  0,  0, 50,  0,  0,  0,  0, 88,  0],
[ 0,  0,  0,  0,  0,  0,  0,  0, 98,  0,  0,  0, 15, 53,  0, 92,  0,  0,  0,  0],
[ 0,  0,  0, 95,  0,  0,  0, 36,  0,  0,  0,  0,  0, 72,  0,  0,  0,  0, 73, 19],
[ 0, 65, 14, 96,  0,  0,  0,  0,  0,  0,  0,  0,  0, 90,  0,  0,  0, 34,  0,  0],
[ 0,  0,  0, 16, 39, 44,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 51,  0,  0],
[ 0, 17,  0,  0,  0, 99, 84, 13, 50, 84,  0,  0,  0,  0, 95,  0, 43, 33, 20,  0],
[79,  0, 17, 52, 99, 12, 69,  0, 98,  0, 68,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[ 0,  0,  0, 82,  0, 44,  0,  0,  0, 97,  0,  0,  0,  0,  0, 10,  0,  0, 31,  0],
[ 0,  0, 21,  0, 67,  0,  0,  0,  0,  0,  4,  0, 50,  0,  0,  0, 33,  0,  0,  0],
[ 0,  0,  0,  0,  9, 42,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  8],
[ 0, 77,  0,  0,  0,  0,  0,  0,  0,  0, 34, 93,  0,  0,  0,  0, 47,  0,  0,  0]],
           ZZ),
        DM([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], ZZ),
        ZZ(1),
    ),

    (
        'zz_large_2',
        DM([
[ 0,  0,  0,  0, 50,  0,  6, 81,  0,  1, 86,  0,  0, 98, 82, 94,  4,  0,  0, 29],
[ 0, 44, 43,  0, 62,  0,  0,  0, 60,  0,  0,  0,  0, 71,  9,  0, 57, 41,  0, 93],
[ 0,  0, 28,  0, 74, 89, 42,  0, 28,  0,  6,  0,  0,  0, 44,  0,  0,  0, 77, 19],
[ 0, 21, 82,  0, 30, 88,  0, 89, 68,  0,  0,  0, 79, 41,  0,  0, 99,  0,  0,  0],
[31,  0,  0,  0, 19, 64,  0,  0, 79,  0,  5,  0, 72, 10, 60, 32, 64, 59,  0, 24],
[ 0,  0,  0,  0,  0, 57,  0, 94,  0, 83, 20,  0,  0,  9, 31,  0, 49, 26, 58,  0],
[ 0, 65, 56, 31, 64,  0,  0,  0,  0,  0,  0, 52, 85,  0,  0,  0,  0, 51,  0,  0],
[ 0, 35,  0,  0,  0, 69,  0,  0, 64,  0,  0,  0,  0, 70,  0,  0, 90,  0, 75, 76],
[69,  7,  0, 90,  0,  0, 84,  0, 47, 69, 19, 20, 42,  0,  0, 32, 71, 35,  0,  0],
[39,  0, 90,  0,  0,  4, 85,  0,  0, 55,  0,  0,  0, 35, 67, 40,  0, 40,  0, 77],
[98, 63,  0, 71,  0, 50,  0,  2, 61,  0, 38,  0,  0,  0,  0, 75,  0, 40, 33, 56],
[ 0, 73,  0, 64,  0, 38,  0, 35, 61,  0,  0, 52,  0,  7,  0, 51,  0,  0,  0, 34],
[ 0,  0, 28,  0, 34,  5, 63, 45, 14, 42, 60, 16, 76, 54, 99,  0, 28, 30,  0,  0],
[58, 37, 14,  0,  0,  0, 94,  0,  0, 90,  0,  0,  0,  0,  0,  0,  0,  8, 90, 53],
[86, 74, 94,  0, 49, 10, 60,  0, 40, 18,  0,  0,  0, 31, 60, 24,  0,  1,  0, 29],
[53,  0,  0, 97,  0,  0, 58,  0,  0, 39, 44, 47,  0,  0,  0, 12, 50,  0,  0, 11],
[ 4,  0, 92, 10, 28,  0,  0, 89,  0,  0, 18, 54, 23, 39,  0,  2,  0, 48,  0, 92],
[ 0,  0, 90, 77, 95, 33,  0,  0, 49, 22, 39,  0,  0,  0,  0,  0,  0, 40,  0,  0],
[96,  0,  0,  0,  0, 38, 86,  0, 22, 76,  0,  0,  0,  0, 83, 88, 95, 65, 72,  0],
[81, 65,  0,  4, 60,  0, 19,  0,  0, 68,  0,  0, 89,  0, 67, 22,  0,  0, 55, 33]],
           ZZ),
        DM([
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
           ZZ),
        ZZ(1),
    ),

    (
        'zz_large_3',
        DM([
[62,35,89,58,22,47,30,28,52,72,17,56,80,26,64,21,10,35,24,42,96,32,23,50,92,37,76,94,63,66],
[20,47,96,34,10,98,19,6,29,2,19,92,61,94,38,41,32,9,5,94,31,58,27,41,72,85,61,62,40,46],
[69,26,35,68,25,52,94,13,38,65,81,10,29,15,5,4,13,99,85,0,80,51,60,60,26,77,85,2,87,25],
[99,58,69,15,52,12,18,7,27,56,12,54,21,92,38,95,33,83,28,1,44,8,29,84,92,12,2,25,46,46],
[93,13,55,48,35,87,24,40,23,35,25,32,0,19,0,85,4,79,26,11,46,75,7,96,76,11,7,57,99,75],
[128,85,26,51,161,173,77,78,85,103,123,58,91,147,38,91,161,36,123,81,102,25,75,59,17,150,112,65,77,143],
[15,59,61,82,12,83,34,8,94,71,66,7,91,21,48,69,26,12,64,38,97,87,38,15,51,33,93,43,66,89],
[74,74,53,39,69,90,41,80,32,66,40,83,87,87,61,38,12,80,24,49,37,90,19,33,56,0,46,57,56,60],
[82,11,0,25,56,58,39,49,92,93,80,38,19,62,33,85,19,61,14,30,45,91,97,34,97,53,92,28,33,43],
[83,79,41,16,95,35,53,45,26,4,71,76,61,69,69,72,87,92,59,72,54,11,22,83,8,57,77,55,19,22],
[49,34,13,31,72,77,52,70,46,41,37,6,42,66,35,6,75,33,62,57,30,14,26,31,9,95,89,13,12,90],
[29,3,49,30,51,32,77,41,38,50,16,1,87,81,93,88,58,91,83,0,38,67,29,64,60,84,5,60,23,28],
[79,51,13,20,89,96,25,8,39,62,86,52,49,81,3,85,86,3,61,24,72,11,49,28,8,55,23,52,65,53],
[96,86,73,20,41,20,37,18,10,61,85,24,40,83,69,41,4,92,23,99,64,33,18,36,32,56,60,98,39,24],
[32,62,47,80,51,66,17,1,9,30,65,75,75,88,99,92,64,53,53,86,38,51,41,14,35,18,39,25,26,32],
[39,21,8,16,33,6,35,85,75,62,43,34,18,68,71,28,32,18,12,0,81,53,1,99,3,5,45,99,35,33],
[19,95,89,45,75,94,92,5,84,93,34,17,50,56,79,98,68,82,65,81,51,90,5,95,33,71,46,61,14,7],
[53,92,8,49,67,84,21,79,49,95,66,48,36,14,62,97,26,45,58,31,83,48,11,89,67,72,91,34,56,89],
[56,76,99,92,40,8,0,16,15,48,35,72,91,46,81,14,86,60,51,7,33,12,53,78,48,21,3,89,15,79],
[81,43,33,49,6,49,36,32,57,74,87,91,17,37,31,17,67,1,40,38,69,8,3,48,59,37,64,97,11,3],
[98,48,77,16,2,48,57,38,63,59,79,35,16,71,60,86,71,41,14,76,80,97,77,69,4,58,22,55,26,73],
[80,47,78,44,31,48,47,29,29,62,19,21,17,24,19,3,53,93,97,57,13,54,12,10,77,66,60,75,32,21],
[86,63,2,13,71,38,86,23,18,15,91,65,77,65,9,92,50,0,17,42,99,80,99,27,10,99,92,9,87,84],
[66,27,72,13,13,15,72,75,39,3,14,71,15,68,10,19,49,54,11,29,47,20,63,13,97,47,24,62,16,96],
[42,63,83,60,49,68,9,53,75,87,40,25,12,63,0,12,0,95,46,46,55,25,89,1,51,1,1,96,80,52],
[35,9,97,13,86,39,66,48,41,57,23,38,11,9,35,72,88,13,41,60,10,64,71,23,1,5,23,57,6,19],
[70,61,5,50,72,60,77,13,41,94,1,45,52,22,99,47,27,18,99,42,16,48,26,9,88,77,10,94,11,92],
[55,68,58,2,72,56,81,52,79,37,1,40,21,46,27,60,37,13,97,42,85,98,69,60,76,44,42,46,29,73],
[73,0,43,17,89,97,45,2,68,14,55,60,95,2,74,85,88,68,93,76,38,76,2,51,45,76,50,79,56,18],
[72,58,41,39,24,80,23,79,44,7,98,75,30,6,85,60,20,58,77,71,90,51,38,80,30,15,33,10,82,8]],
            ZZ),
        Matrix([
    [eye(29) * 2028539767964472550625641331179545072876560857886207583101,
     Matrix([ 4260575808093245475167216057435155595594339172099000182569,
              169148395880755256182802335904188369274227936894862744452,
              4915975976683942569102447281579134986891620721539038348914,
              6113916866367364958834844982578214901958429746875633283248,
              5585689617819894460378537031623265659753379011388162534838,
              359776822829880747716695359574308645968094838905181892423,
              -2800926112141776386671436511182421432449325232461665113305,
              941642292388230001722444876624818265766384442910688463158,
              3648811843256146649321864698600908938933015862008642023935,
              -4104526163246702252932955226754097174212129127510547462419,
              -704814955438106792441896903238080197619233342348191408078,
              1640882266829725529929398131287244562048075707575030019335,
              -4068330845192910563212155694231438198040299927120544468520,
              136589038308366497790495711534532612862715724187671166593,
              2544937011460702462290799932536905731142196510605191645593,
              755591839174293940486133926192300657264122907519174116472,
              -3683838489869297144348089243628436188645897133242795965021,
              -522207137101161299969706310062775465103537953077871128403,
              -2260451796032703984456606059649402832441331339246756656334,
              -6476809325293587953616004856993300606040336446656916663680,
              3521944238996782387785653800944972787867472610035040989081,
              2270762115788407950241944504104975551914297395787473242379,
              -3259947194628712441902262570532921252128444706733549251156,
              -5624569821491886970999097239695637132075823246850431083557,
              -3262698255682055804320585332902837076064075936601504555698,
              5786719943788937667411185880136324396357603606944869545501,
              -955257841973865996077323863289453200904051299086000660036,
              -1294235552446355326174641248209752679127075717918392702116,
              -3718353510747301598130831152458342785269166356215331448279,
             ]),],
        [zeros(1, 29), zeros(1, 1)],
        ]).to_DM().to_dense(),
        ZZ(2028539767964472550625641331179545072876560857886207583101),
    ),


    (
        'qq_1',
        DM([[(1,2), 0], [0, 2]], QQ),
        DM([[1, 0], [0, 1]], QQ),
        QQ(1),
    ),

    (
        # Standard square case
        'qq_2',
        DM([[0, 1],
            [1, 1]], QQ),
        DM([[1, 0],
            [0, 1]], QQ),
        QQ(1),
    ),

    (
        # m < n  case
        'qq_3',
        DM([[1, 2, 1],
            [3, 4, 1]], QQ),
        DM([[1, 0, -1],
            [0, 1,  1]], QQ),
        QQ(1),
    ),

    (
        # same m < n  but reversed
        'qq_4',
        DM([[3, 4, 1],
            [1, 2, 1]], QQ),
        DM([[1, 0, -1],
            [0, 1,  1]], QQ),
        QQ(1),
    ),

    (
        # m > n case
        'qq_5',
        DM([[1, 0],
            [1, 3],
            [0, 1]], QQ),
        DM([[1, 0],
            [0, 1],
            [0, 0]], QQ),
        QQ(1),
    ),

    (
        # Example with missing pivot
        'qq_6',
        DM([[1, 0, 1],
            [3, 0, 1]], QQ),
        DM([[1, 0, 0],
            [0, 0, 1]], QQ),
        QQ(1),
    ),

    (
        # This is intended to trigger the threshold where we give up on
        # clearing denominators.
        'qq_large_1',
        qq_large_1,
        DomainMatrix.eye(11, QQ).to_dense(),
        QQ(1),
    ),

    (
        # This is intended to trigger the threshold where we use rref_den over
        # QQ.
        'qq_large_2',
        qq_large_2,
        DomainMatrix.eye(11, QQ).to_dense(),
        QQ(1),
    ),

    (
        # Example with missing pivot and no replacement

        # This example is just enough to show a different result from the dense
        # and sparse versions of the algorithm:
        #
        #   >>> A = Matrix([[0, 1], [0, 2], [1, 0]])
        #   >>> A.to_DM().to_sparse().rref_den()[0].to_Matrix()
        #   Matrix([
        #   [1, 0],
        #   [0, 1],
        #   [0, 0]])
        #   >>> A.to_DM().to_dense().rref_den()[0].to_Matrix()
        #   Matrix([
        #   [2, 0],
        #   [0, 2],
        #   [0, 0]])
        #
        'qq_7',
        DM([[0, 1],
            [0, 2],
            [1, 0]], QQ),
        DM([[1, 0],
            [0, 1],
            [0, 0]], QQ),
        QQ(1),
    ),

    (
        # Gaussian integers
        'zz_i_1',
        DM([[(0,1), 1, 1],
            [    1, 1, 1]], ZZ_I),
        DM([[1, 0, 0],
            [0, 1, 1]], ZZ_I),
        ZZ_I(1),
    ),

    (
        # EX: test_issue_23718
        'EX_1',
        DM([
        [a, b, 1],
        [c, d, 1]], EX),
        DM([[a*d - b*c,         0, -b + d],
            [        0, a*d - b*c,  a - c]], EX),
        EX(a*d - b*c),
    ),

]


def _to_DM(A, ans):
    """Convert the answer to DomainMatrix."""
    if isinstance(A, DomainMatrix):
        return A.to_dense()
    elif isinstance(A, Matrix):
        return A.to_DM(ans.domain).to_dense()

    if not (hasattr(A, 'shape') and hasattr(A, 'domain')):
        shape, domain = ans.shape, ans.domain
    else:
        shape, domain = A.shape, A.domain

    if isinstance(A, (DDM, list)):
        return DomainMatrix(list(A), shape, domain).to_dense()
    elif isinstance(A, (SDM, dict)):
        return DomainMatrix(dict(A), shape, domain).to_dense()
    else:
        assert False # pragma: no cover


def _pivots(A_rref):
    """Return the pivots from the rref of A."""
    return tuple(sorted(map(min, A_rref.to_sdm().values())))


def _check_cancel(result, rref_ans, den_ans):
    """Check the cancelled result."""
    rref, den, pivots = result
    if isinstance(rref, (DDM, SDM, list, dict)):
        assert type(pivots) is list
        pivots = tuple(pivots)
    rref = _to_DM(rref, rref_ans)
    rref2, den2 = rref.cancel_denom(den)
    assert rref2 == rref_ans
    assert den2 == den_ans
    assert pivots == _pivots(rref)


def _check_divide(result, rref_ans, den_ans):
    """Check the divided result."""
    rref, pivots = result
    if isinstance(rref, (DDM, SDM, list, dict)):
        assert type(pivots) is list
        pivots = tuple(pivots)
    rref_ans = rref_ans.to_field() / den_ans
    rref = _to_DM(rref, rref_ans)
    assert rref == rref_ans
    assert _pivots(rref) == pivots


@pytest.mark.parametrize('name, A, A_rref, den', RREF_EXAMPLES)
def test_Matrix_rref(name, A, A_rref, den):
    K = A.domain
    A = A.to_Matrix()
    A_rref_found, pivots = A.rref()
    if K.is_EX:
        A_rref_found = A_rref_found.expand()
    _check_divide((A_rref_found, pivots), A_rref, den)


@pytest.mark.parametrize('name, A, A_rref, den', RREF_EXAMPLES)
def test_dm_dense_rref(name, A, A_rref, den):
    A = A.to_field()
    _check_divide(A.rref(), A_rref, den)


@pytest.mark.parametrize('name, A, A_rref, den', RREF_EXAMPLES)
def test_dm_dense_rref_den(name, A, A_rref, den):
    _check_cancel(A.rref_den(), A_rref, den)


@pytest.mark.parametrize('name, A, A_rref, den', RREF_EXAMPLES)
def test_dm_sparse_rref(name, A, A_rref, den):
    A = A.to_field().to_sparse()
    _check_divide(A.rref(), A_rref, den)


@pytest.mark.parametrize('name, A, A_rref, den', RREF_EXAMPLES)
def test_dm_sparse_rref_den(name, A, A_rref, den):
    A = A.to_sparse()
    _check_cancel(A.rref_den(), A_rref, den)


@pytest.mark.parametrize('name, A, A_rref, den', RREF_EXAMPLES)
def test_dm_sparse_rref_den_keep_domain(name, A, A_rref, den):
    A = A.to_sparse()
    A_rref_f, den_f, pivots_f = A.rref_den(keep_domain=False)
    A_rref_f = A_rref_f.to_field() / den_f
    _check_divide((A_rref_f, pivots_f), A_rref, den)


@pytest.mark.parametrize('name, A, A_rref, den', RREF_EXAMPLES)
def test_dm_sparse_rref_den_keep_domain_CD(name, A, A_rref, den):
    A = A.to_sparse()
    A_rref_f, den_f, pivots_f = A.rref_den(keep_domain=False, method='CD')
    A_rref_f = A_rref_f.to_field() / den_f
    _check_divide((A_rref_f, pivots_f), A_rref, den)


@pytest.mark.parametrize('name, A, A_rref, den', RREF_EXAMPLES)
def test_dm_sparse_rref_den_keep_domain_GJ(name, A, A_rref, den):
    A = A.to_sparse()
    A_rref_f, den_f, pivots_f = A.rref_den(keep_domain=False, method='GJ')
    A_rref_f = A_rref_f.to_field() / den_f
    _check_divide((A_rref_f, pivots_f), A_rref, den)


@pytest.mark.parametrize('name, A, A_rref, den', RREF_EXAMPLES)
def test_ddm_rref_den(name, A, A_rref, den):
    A = A.to_ddm()
    _check_cancel(A.rref_den(), A_rref, den)


@pytest.mark.parametrize('name, A, A_rref, den', RREF_EXAMPLES)
def test_sdm_rref_den(name, A, A_rref, den):
    A = A.to_sdm()
    _check_cancel(A.rref_den(), A_rref, den)


@pytest.mark.parametrize('name, A, A_rref, den', RREF_EXAMPLES)
def test_ddm_rref(name, A, A_rref, den):
    A = A.to_field().to_ddm()
    _check_divide(A.rref(), A_rref, den)


@pytest.mark.parametrize('name, A, A_rref, den', RREF_EXAMPLES)
def test_sdm_rref(name, A, A_rref, den):
    A = A.to_field().to_sdm()
    _check_divide(A.rref(), A_rref, den)


@pytest.mark.parametrize('name, A, A_rref, den', RREF_EXAMPLES)
def test_ddm_irref(name, A, A_rref, den):
    A = A.to_field().to_ddm().copy()
    pivots_found = ddm_irref(A)
    _check_divide((A, pivots_found), A_rref, den)


@pytest.mark.parametrize('name, A, A_rref, den', RREF_EXAMPLES)
def test_ddm_irref_den(name, A, A_rref, den):
    A = A.to_ddm().copy()
    (den_found, pivots_found) = ddm_irref_den(A, A.domain)
    result = (A, den_found, pivots_found)
    _check_cancel(result, A_rref, den)


@pytest.mark.parametrize('name, A, A_rref, den', RREF_EXAMPLES)
def test_sparse_sdm_rref(name, A, A_rref, den):
    A = A.to_field().to_sdm()
    _check_divide(sdm_irref(A)[:2], A_rref, den)


@pytest.mark.parametrize('name, A, A_rref, den', RREF_EXAMPLES)
def test_sparse_sdm_rref_den(name, A, A_rref, den):
    A = A.to_sdm().copy()
    K = A.domain
    _check_cancel(sdm_rref_den(A, K), A_rref, den)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\testing\__init__.py ===
"""Common test support for all numpy test scripts.

This single module should provide all the common functionality for numpy tests
in a single location, so that test scripts can just import it and work right
away.

"""
from unittest import TestCase

from . import _private, overrides
from ._private import extbuild
from ._private.utils import *
from ._private.utils import _assert_valid_refcount, _gen_alignment_data

__all__ = (
    _private.utils.__all__ + ['TestCase', 'overrides']
)

from numpy._pytesttester import PytestTester

test = PytestTester(__name__)
del PytestTester

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\operators\qdq_base_operator.py ===
import itertools

from ..quant_utils import QuantizedValue, QuantizedValueType, attribute_to_kwarg, quantize_nparray  # noqa: F401
from .base_operator import QuantOperatorBase  # noqa: F401


class QDQOperatorBase:
    def __init__(self, onnx_quantizer, onnx_node):
        self.quantizer = onnx_quantizer
        self.node = onnx_node
        self.disable_qdq_for_node_output = onnx_node.op_type in onnx_quantizer.op_types_to_exclude_output_quantization

    def quantize(self):
        node = self.node

        if self.disable_qdq_for_node_output:
            tensors_to_quantize = node.input
        else:
            tensors_to_quantize = itertools.chain(node.input, node.output)

        for tensor_name in tensors_to_quantize:
            self.quantizer.quantize_activation_tensor(tensor_name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\utils\protection.py ===
# Copyright (c) 2010-2024 openpyxl


def hash_password(plaintext_password=''):
    """
    Create a password hash from a given string for protecting a worksheet
    only. This will not work for encrypting a workbook.

    This method is based on the algorithm provided by
    Daniel Rentz of OpenOffice and the PEAR package
    Spreadsheet_Excel_Writer by Xavier Noguer <xnoguer@rezebra.com>.
    See also http://blogs.msdn.com/b/ericwhite/archive/2008/02/23/the-legacy-hashing-algorithm-in-open-xml.aspx
    """
    password = 0x0000
    for idx, char in enumerate(plaintext_password, 1):
        value = ord(char) << idx
        rotated_bits = value >> 15
        value &= 0x7fff
        password ^= (value | rotated_bits)
    password ^= len(plaintext_password)
    password ^= 0xCE4B
    return str(hex(password)).upper()[2:]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\util\queue.py ===
import collections

from ..packages import six
from ..packages.six.moves import queue

if six.PY2:
    # Queue is imported for side effects on MS Windows. See issue #229.
    import Queue as _unused_module_Queue  # noqa: F401


class LifoQueue(queue.Queue):
    def _init(self, _):
        self.queue = collections.deque()

    def _qsize(self, len=len):
        return len(self.queue)

    def _put(self, item):
        self.queue.append(item)

    def _get(self):
        return self.queue.pop()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\get_doc.py ===
import sys
import textwrap

from .py3k_compat import is_callable

rlmain = sys.modules["readline"]
rl = rlmain.rl


def get_doc(rl_):
    methods = [(x, getattr(rl_, x)) for x in dir(rl_) if is_callable(getattr(rl_, x))]
    return [(x, m.__doc__) for x, m in methods if m.__doc__]


def get_rest(rl_):
    q = get_doc(rl_)
    out = []
    for funcname, doc in q:
        out.append(funcname)
        out.append("\n".join(textwrap.wrap(doc, 80, initial_indent="   ")))
        out.append("")
    return out

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\autolev\test-examples\ruletest2.py ===
import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

x1, x2 = _me.dynamicsymbols('x1 x2')
f1 = x1*x2+3*x1**2
f2 = x1*_me.dynamicsymbols._t+x2*_me.dynamicsymbols._t**2
x, y = _me.dynamicsymbols('x y')
x_d, y_d = _me.dynamicsymbols('x_ y_', 1)
y_dd = _me.dynamicsymbols('y_', 2)
q1, q2, q3, u1, u2 = _me.dynamicsymbols('q1 q2 q3 u1 u2')
p1, p2 = _me.dynamicsymbols('p1 p2')
p1_d, p2_d = _me.dynamicsymbols('p1_ p2_', 1)
w1, w2, w3, r1, r2 = _me.dynamicsymbols('w1 w2 w3 r1 r2')
w1_d, w2_d, w3_d, r1_d, r2_d = _me.dynamicsymbols('w1_ w2_ w3_ r1_ r2_', 1)
r1_dd, r2_dd = _me.dynamicsymbols('r1_ r2_', 2)
c11, c12, c21, c22 = _me.dynamicsymbols('c11 c12 c21 c22')
d11, d12, d13 = _me.dynamicsymbols('d11 d12 d13')
j1, j2 = _me.dynamicsymbols('j1 j2')
n = _sm.symbols('n')
n = _sm.I

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\__init__.py ===
from .plot import plot_backends
from .plot_implicit import plot_implicit
from .textplot import textplot
from .pygletplot import PygletPlot
from .plot import PlotGrid
from .plot import (plot, plot_parametric, plot3d, plot3d_parametric_surface,
                  plot3d_parametric_line, plot_contour)

__all__ = [
    'plot_backends',

    'plot_implicit',

    'textplot',

    'PygletPlot',

    'PlotGrid',

    'plot', 'plot_parametric', 'plot3d', 'plot3d_parametric_surface',
    'plot3d_parametric_line', 'plot_contour'
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\pythonrational.py ===
"""
Rational number type based on Python integers.

The PythonRational class from here has been moved to
sympy.external.pythonmpq

This module is just left here for backwards compatibility.
"""


from sympy.core.numbers import Rational
from sympy.core.sympify import _sympy_converter
from sympy.utilities import public
from sympy.external.pythonmpq import PythonMPQ


PythonRational = public(PythonMPQ)


def sympify_pythonrational(arg):
    return Rational(arg.numerator, arg.denominator)
_sympy_converter[PythonRational] = sympify_pythonrational

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\_compilation\__init__.py ===
""" This sub-module is private, i.e. external code should not depend on it.

These functions are used by tests run as part of continuous integration.
Once the implementation is mature (it should support the major
platforms: Windows, OS X & Linux) it may become official API which
 may be relied upon by downstream libraries. Until then API may break
without prior notice.

TODO:
- (optionally) clean up after tempfile.mkdtemp()
- cross-platform testing
- caching of compiler choice and intermediate files

"""

from .compilation import compile_link_import_strings, compile_run_strings
from .availability import has_fortran, has_c, has_cxx

__all__ = [
    'compile_link_import_strings', 'compile_run_strings',
    'has_fortran', 'has_c', 'has_cxx',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\module_with_deprecations.py ===
regular = "hi"

import sys

from .. import _deprecate

_deprecate.deprecate_attributes(
    __name__,
    {
        "dep1": _deprecate.DeprecatedAttribute("value1", "1.1", issue=1),
        "dep2": _deprecate.DeprecatedAttribute(
            "value2",
            "1.2",
            issue=1,
            instead="instead-string",
        ),
    },
)

this_mod = sys.modules[__name__]
assert this_mod.regular == "hi"
assert "dep1" not in globals()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\contrib\emscripten\request.py ===
from __future__ import annotations

from dataclasses import dataclass, field

from ..._base_connection import _TYPE_BODY


@dataclass
class EmscriptenRequest:
    method: str
    url: str
    params: dict[str, str] | None = None
    body: _TYPE_BODY | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 0
    decode_content: bool = True

    def set_header(self, name: str, value: str) -> None:
        self.headers[name.capitalize()] = value

    def set_body(self, body: _TYPE_BODY | None) -> None:
        self.body = body

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_set_index.py ===
"""
See also: test_reindex.py:TestReindexSetIndex
"""

from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    date_range,
    period_range,
    to_datetime,
)
import pandas._testing as tm


@pytest.fixture
def frame_of_index_cols():
    """
    Fixture for DataFrame of columns that can be used for indexing

    Columns are ['A', 'B', 'C', 'D', 'E', ('tuple', 'as', 'label')];
    'A' & 'B' contain duplicates (but are jointly unique), the rest are unique.

         A      B  C         D         E  (tuple, as, label)
    0  foo    one  a  0.608477 -0.012500           -1.664297
    1  foo    two  b -0.633460  0.249614           -0.364411
    2  foo  three  c  0.615256  2.154968           -0.834666
    3  bar    one  d  0.234246  1.085675            0.718445
    4  bar    two  e  0.533841 -0.005702           -3.533912
    """
    df = DataFrame(
        {
            "A": ["foo", "foo", "foo", "bar", "bar"],
            "B": ["one", "two", "three", "one", "two"],
            "C": ["a", "b", "c", "d", "e"],
            "D": np.random.default_rng(2).standard_normal(5),
            "E": np.random.default_rng(2).standard_normal(5),
            ("tuple", "as", "label"): np.random.default_rng(2).standard_normal(5),
        }
    )
    return df


class TestSetIndex:
    def test_set_index_multiindex(self):
        # segfault in GH#3308
        d = {"t1": [2, 2.5, 3], "t2": [4, 5, 6]}
        df = DataFrame(d)
        tuples = [(0, 1), (0, 2), (1, 2)]
        df["tuples"] = tuples

        index = MultiIndex.from_tuples(df["tuples"])
        # it works!
        df.set_index(index)

    def test_set_index_empty_column(self):
        # GH#1971
        df = DataFrame(
            [
                {"a": 1, "p": 0},
                {"a": 2, "m": 10},
                {"a": 3, "m": 11, "p": 20},
                {"a": 4, "m": 12, "p": 21},
            ],
            columns=["a", "m", "p", "x"],
        )

        result = df.set_index(["a", "x"])

        expected = df[["m", "p"]]
        expected.index = MultiIndex.from_arrays([df["a"], df["x"]], names=["a", "x"])
        tm.assert_frame_equal(result, expected)

    def test_set_index_empty_dataframe(self):
        # GH#38419
        df1 = DataFrame(
            {"a": Series(dtype="datetime64[ns]"), "b": Series(dtype="int64"), "c": []}
        )

        df2 = df1.set_index(["a", "b"])
        result = df2.index.to_frame().dtypes
        expected = df1[["a", "b"]].dtypes
        tm.assert_series_equal(result, expected)

    def test_set_index_multiindexcolumns(self):
        columns = MultiIndex.from_tuples([("foo", 1), ("foo", 2), ("bar", 1)])
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)), columns=columns
        )

        result = df.set_index(df.columns[0])

        expected = df.iloc[:, 1:]
        expected.index = df.iloc[:, 0].values
        expected.index.names = [df.columns[0]]
        tm.assert_frame_equal(result, expected)

    def test_set_index_timezone(self):
        # GH#12358
        # tz-aware Series should retain the tz
        idx = DatetimeIndex(["2014-01-01 10:10:10"], tz="UTC").tz_convert("Europe/Rome")
        df = DataFrame({"A": idx})
        assert df.set_index(idx).index[0].hour == 11
        assert DatetimeIndex(Series(df.A))[0].hour == 11
        assert df.set_index(df.A).index[0].hour == 11

    def test_set_index_cast_datetimeindex(self):
        df = DataFrame(
            {
                "A": [datetime(2000, 1, 1) + timedelta(i) for i in range(1000)],
                "B": np.random.default_rng(2).standard_normal(1000),
            }
        )

        idf = df.set_index("A")
        assert isinstance(idf.index, DatetimeIndex)

    def test_set_index_dst(self):
        di = date_range("2006-10-29 00:00:00", periods=3, freq="h", tz="US/Pacific")

        df = DataFrame(data={"a": [0, 1, 2], "b": [3, 4, 5]}, index=di).reset_index()
        # single level
        res = df.set_index("index")
        exp = DataFrame(
            data={"a": [0, 1, 2], "b": [3, 4, 5]},
            index=Index(di, name="index"),
        )
        exp.index = exp.index._with_freq(None)
        tm.assert_frame_equal(res, exp)

        # GH#12920
        res = df.set_index(["index", "a"])
        exp_index = MultiIndex.from_arrays([di, [0, 1, 2]], names=["index", "a"])
        exp = DataFrame({"b": [3, 4, 5]}, index=exp_index)
        tm.assert_frame_equal(res, exp)

    def test_set_index(self, float_string_frame):
        df = float_string_frame
        idx = Index(np.arange(len(df))[::-1])

        df = df.set_index(idx)
        tm.assert_index_equal(df.index, idx)
        with pytest.raises(ValueError, match="Length mismatch"):
            df.set_index(idx[::2])

    def test_set_index_names(self):
        df = DataFrame(
            np.ones((10, 4)),
            columns=Index(list("ABCD")),
            index=Index([f"i-{i}" for i in range(10)]),
        )
        df.index.name = "name"

        assert df.set_index(df.index).index.names == ["name"]

        mi = MultiIndex.from_arrays(df[["A", "B"]].T.values, names=["A", "B"])
        mi2 = MultiIndex.from_arrays(
            df[["A", "B", "A", "B"]].T.values, names=["A", "B", "C", "D"]
        )

        df = df.set_index(["A", "B"])

        assert df.set_index(df.index).index.names == ["A", "B"]

        # Check that set_index isn't converting a MultiIndex into an Index
        assert isinstance(df.set_index(df.index).index, MultiIndex)

        # Check actual equality
        tm.assert_index_equal(df.set_index(df.index).index, mi)

        idx2 = df.index.rename(["C", "D"])

        # Check that [MultiIndex, MultiIndex] yields a MultiIndex rather
        # than a pair of tuples
        assert isinstance(df.set_index([df.index, idx2]).index, MultiIndex)

        # Check equality
        tm.assert_index_equal(df.set_index([df.index, idx2]).index, mi2)

    # A has duplicate values, C does not
    @pytest.mark.parametrize("keys", ["A", "C", ["A", "B"], ("tuple", "as", "label")])
    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("drop", [True, False])
    def test_set_index_drop_inplace(self, frame_of_index_cols, drop, inplace, keys):
        df = frame_of_index_cols

        if isinstance(keys, list):
            idx = MultiIndex.from_arrays([df[x] for x in keys], names=keys)
        else:
            idx = Index(df[keys], name=keys)
        expected = df.drop(keys, axis=1) if drop else df
        expected.index = idx

        if inplace:
            result = df.copy()
            return_value = result.set_index(keys, drop=drop, inplace=True)
            assert return_value is None
        else:
            result = df.set_index(keys, drop=drop)

        tm.assert_frame_equal(result, expected)

    # A has duplicate values, C does not
    @pytest.mark.parametrize("keys", ["A", "C", ["A", "B"], ("tuple", "as", "label")])
    @pytest.mark.parametrize("drop", [True, False])
    def test_set_index_append(self, frame_of_index_cols, drop, keys):
        df = frame_of_index_cols

        keys = keys if isinstance(keys, list) else [keys]
        idx = MultiIndex.from_arrays(
            [df.index] + [df[x] for x in keys], names=[None] + keys
        )
        expected = df.drop(keys, axis=1) if drop else df.copy()
        expected.index = idx

        result = df.set_index(keys, drop=drop, append=True)

        tm.assert_frame_equal(result, expected)

    # A has duplicate values, C does not
    @pytest.mark.parametrize("keys", ["A", "C", ["A", "B"], ("tuple", "as", "label")])
    @pytest.mark.parametrize("drop", [True, False])
    def test_set_index_append_to_multiindex(self, frame_of_index_cols, drop, keys):
        # append to existing multiindex
        df = frame_of_index_cols.set_index(["D"], drop=drop, append=True)

        keys = keys if isinstance(keys, list) else [keys]
        expected = frame_of_index_cols.set_index(["D"] + keys, drop=drop, append=True)

        result = df.set_index(keys, drop=drop, append=True)

        tm.assert_frame_equal(result, expected)

    def test_set_index_after_mutation(self):
        # GH#1590
        df = DataFrame({"val": [0, 1, 2], "key": ["a", "b", "c"]})
        expected = DataFrame({"val": [1, 2]}, Index(["b", "c"], name="key"))

        df2 = df.loc[df.index.map(lambda indx: indx >= 1)]
        result = df2.set_index("key")
        tm.assert_frame_equal(result, expected)

    # MultiIndex constructor does not work directly on Series -> lambda
    # Add list-of-list constructor because list is ambiguous -> lambda
    # also test index name if append=True (name is duplicate here for B)
    @pytest.mark.parametrize(
        "box",
        [
            Series,
            Index,
            np.array,
            list,
            lambda x: [list(x)],
            lambda x: MultiIndex.from_arrays([x]),
        ],
    )
    @pytest.mark.parametrize(
        "append, index_name", [(True, None), (True, "B"), (True, "test"), (False, None)]
    )
    @pytest.mark.parametrize("drop", [True, False])
    def test_set_index_pass_single_array(
        self, frame_of_index_cols, drop, append, index_name, box
    ):
        df = frame_of_index_cols
        df.index.name = index_name

        key = box(df["B"])
        if box == list:
            # list of strings gets interpreted as list of keys
            msg = "['one', 'two', 'three', 'one', 'two']"
            with pytest.raises(KeyError, match=msg):
                df.set_index(key, drop=drop, append=append)
        else:
            # np.array/list-of-list "forget" the name of B
            name_mi = getattr(key, "names", None)
            name = [getattr(key, "name", None)] if name_mi is None else name_mi

            result = df.set_index(key, drop=drop, append=append)

            # only valid column keys are dropped
            # since B is always passed as array above, nothing is dropped
            expected = df.set_index(["B"], drop=False, append=append)
            expected.index.names = [index_name] + name if append else name

            tm.assert_frame_equal(result, expected)

    # MultiIndex constructor does not work directly on Series -> lambda
    # also test index name if append=True (name is duplicate here for A & B)
    @pytest.mark.parametrize(
        "box", [Series, Index, np.array, list, lambda x: MultiIndex.from_arrays([x])]
    )
    @pytest.mark.parametrize(
        "append, index_name",
        [(True, None), (True, "A"), (True, "B"), (True, "test"), (False, None)],
    )
    @pytest.mark.parametrize("drop", [True, False])
    def test_set_index_pass_arrays(
        self, frame_of_index_cols, drop, append, index_name, box
    ):
        df = frame_of_index_cols
        df.index.name = index_name

        keys = ["A", box(df["B"])]
        # np.array/list "forget" the name of B
        names = ["A", None if box in [np.array, list, tuple, iter] else "B"]

        result = df.set_index(keys, drop=drop, append=append)

        # only valid column keys are dropped
        # since B is always passed as array above, only A is dropped, if at all
        expected = df.set_index(["A", "B"], drop=False, append=append)
        expected = expected.drop("A", axis=1) if drop else expected
        expected.index.names = [index_name] + names if append else names

        tm.assert_frame_equal(result, expected)

    # MultiIndex constructor does not work directly on Series -> lambda
    # We also emulate a "constructor" for the label -> lambda
    # also test index name if append=True (name is duplicate here for A)
    @pytest.mark.parametrize(
        "box2",
        [
            Series,
            Index,
            np.array,
            list,
            iter,
            lambda x: MultiIndex.from_arrays([x]),
            lambda x: x.name,
        ],
    )
    @pytest.mark.parametrize(
        "box1",
        [
            Series,
            Index,
            np.array,
            list,
            iter,
            lambda x: MultiIndex.from_arrays([x]),
            lambda x: x.name,
        ],
    )
    @pytest.mark.parametrize(
        "append, index_name", [(True, None), (True, "A"), (True, "test"), (False, None)]
    )
    @pytest.mark.parametrize("drop", [True, False])
    def test_set_index_pass_arrays_duplicate(
        self, frame_of_index_cols, drop, append, index_name, box1, box2
    ):
        df = frame_of_index_cols
        df.index.name = index_name

        keys = [box1(df["A"]), box2(df["A"])]
        result = df.set_index(keys, drop=drop, append=append)

        # if either box is iter, it has been consumed; re-read
        keys = [box1(df["A"]), box2(df["A"])]

        # need to adapt first drop for case that both keys are 'A' --
        # cannot drop the same column twice;
        # plain == would give ambiguous Boolean error for containers
        first_drop = (
            False
            if (
                isinstance(keys[0], str)
                and keys[0] == "A"
                and isinstance(keys[1], str)
                and keys[1] == "A"
            )
            else drop
        )
        # to test against already-tested behaviour, we add sequentially,
        # hence second append always True; must wrap keys in list, otherwise
        # box = list would be interpreted as keys
        expected = df.set_index([keys[0]], drop=first_drop, append=append)
        expected = expected.set_index([keys[1]], drop=drop, append=True)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("append", [True, False])
    @pytest.mark.parametrize("drop", [True, False])
    def test_set_index_pass_multiindex(self, frame_of_index_cols, drop, append):
        df = frame_of_index_cols
        keys = MultiIndex.from_arrays([df["A"], df["B"]], names=["A", "B"])

        result = df.set_index(keys, drop=drop, append=append)

        # setting with a MultiIndex will never drop columns
        expected = df.set_index(["A", "B"], drop=False, append=append)

        tm.assert_frame_equal(result, expected)

    def test_construction_with_categorical_index(self):
        ci = CategoricalIndex(list("ab") * 5, name="B")

        # with Categorical
        df = DataFrame(
            {"A": np.random.default_rng(2).standard_normal(10), "B": ci.values}
        )
        idf = df.set_index("B")
        tm.assert_index_equal(idf.index, ci)

        # from a CategoricalIndex
        df = DataFrame({"A": np.random.default_rng(2).standard_normal(10), "B": ci})
        idf = df.set_index("B")
        tm.assert_index_equal(idf.index, ci)

        # round-trip
        idf = idf.reset_index().set_index("B")
        tm.assert_index_equal(idf.index, ci)

    def test_set_index_preserve_categorical_dtype(self):
        # GH#13743, GH#13854
        df = DataFrame(
            {
                "A": [1, 2, 1, 1, 2],
                "B": [10, 16, 22, 28, 34],
                "C1": Categorical(list("abaab"), categories=list("bac"), ordered=False),
                "C2": Categorical(list("abaab"), categories=list("bac"), ordered=True),
            }
        )
        for cols in ["C1", "C2", ["A", "C1"], ["A", "C2"], ["C1", "C2"]]:
            result = df.set_index(cols).reset_index()
            result = result.reindex(columns=df.columns)
            tm.assert_frame_equal(result, df)

    def test_set_index_datetime(self):
        # GH#3950
        df = DataFrame(
            {
                "label": ["a", "a", "a", "b", "b", "b"],
                "datetime": [
                    "2011-07-19 07:00:00",
                    "2011-07-19 08:00:00",
                    "2011-07-19 09:00:00",
                    "2011-07-19 07:00:00",
                    "2011-07-19 08:00:00",
                    "2011-07-19 09:00:00",
                ],
                "value": range(6),
            }
        )
        df.index = to_datetime(df.pop("datetime"), utc=True)
        df.index = df.index.tz_convert("US/Pacific")

        expected = DatetimeIndex(
            ["2011-07-19 07:00:00", "2011-07-19 08:00:00", "2011-07-19 09:00:00"],
            name="datetime",
        )
        expected = expected.tz_localize("UTC").tz_convert("US/Pacific")

        df = df.set_index("label", append=True)
        tm.assert_index_equal(df.index.levels[0], expected)
        tm.assert_index_equal(df.index.levels[1], Index(["a", "b"], name="label"))
        assert df.index.names == ["datetime", "label"]

        df = df.swaplevel(0, 1)
        tm.assert_index_equal(df.index.levels[0], Index(["a", "b"], name="label"))
        tm.assert_index_equal(df.index.levels[1], expected)
        assert df.index.names == ["label", "datetime"]

        df = DataFrame(np.random.default_rng(2).random(6))
        idx1 = DatetimeIndex(
            [
                "2011-07-19 07:00:00",
                "2011-07-19 08:00:00",
                "2011-07-19 09:00:00",
                "2011-07-19 07:00:00",
                "2011-07-19 08:00:00",
                "2011-07-19 09:00:00",
            ],
            tz="US/Eastern",
        )
        idx2 = DatetimeIndex(
            [
                "2012-04-01 09:00",
                "2012-04-01 09:00",
                "2012-04-01 09:00",
                "2012-04-02 09:00",
                "2012-04-02 09:00",
                "2012-04-02 09:00",
            ],
            tz="US/Eastern",
        )
        idx3 = date_range("2011-01-01 09:00", periods=6, tz="Asia/Tokyo")
        idx3 = idx3._with_freq(None)

        df = df.set_index(idx1)
        df = df.set_index(idx2, append=True)
        df = df.set_index(idx3, append=True)

        expected1 = DatetimeIndex(
            ["2011-07-19 07:00:00", "2011-07-19 08:00:00", "2011-07-19 09:00:00"],
            tz="US/Eastern",
        )
        expected2 = DatetimeIndex(
            ["2012-04-01 09:00", "2012-04-02 09:00"], tz="US/Eastern"
        )

        tm.assert_index_equal(df.index.levels[0], expected1)
        tm.assert_index_equal(df.index.levels[1], expected2)
        tm.assert_index_equal(df.index.levels[2], idx3)

        # GH#7092
        tm.assert_index_equal(df.index.get_level_values(0), idx1)
        tm.assert_index_equal(df.index.get_level_values(1), idx2)
        tm.assert_index_equal(df.index.get_level_values(2), idx3)

    def test_set_index_period(self):
        # GH#6631
        df = DataFrame(np.random.default_rng(2).random(6))
        idx1 = period_range("2011-01-01", periods=3, freq="M")
        idx1 = idx1.append(idx1)
        idx2 = period_range("2013-01-01 09:00", periods=2, freq="h")
        idx2 = idx2.append(idx2).append(idx2)
        idx3 = period_range("2005", periods=6, freq="Y")

        df = df.set_index(idx1)
        df = df.set_index(idx2, append=True)
        df = df.set_index(idx3, append=True)

        expected1 = period_range("2011-01-01", periods=3, freq="M")
        expected2 = period_range("2013-01-01 09:00", periods=2, freq="h")

        tm.assert_index_equal(df.index.levels[0], expected1)
        tm.assert_index_equal(df.index.levels[1], expected2)
        tm.assert_index_equal(df.index.levels[2], idx3)

        tm.assert_index_equal(df.index.get_level_values(0), idx1)
        tm.assert_index_equal(df.index.get_level_values(1), idx2)
        tm.assert_index_equal(df.index.get_level_values(2), idx3)


class TestSetIndexInvalid:
    def test_set_index_verify_integrity(self, frame_of_index_cols):
        df = frame_of_index_cols

        with pytest.raises(ValueError, match="Index has duplicate keys"):
            df.set_index("A", verify_integrity=True)
        # with MultiIndex
        with pytest.raises(ValueError, match="Index has duplicate keys"):
            df.set_index([df["A"], df["A"]], verify_integrity=True)

    @pytest.mark.parametrize("append", [True, False])
    @pytest.mark.parametrize("drop", [True, False])
    def test_set_index_raise_keys(self, frame_of_index_cols, drop, append):
        df = frame_of_index_cols

        with pytest.raises(KeyError, match="['foo', 'bar', 'baz']"):
            # column names are A-E, as well as one tuple
            df.set_index(["foo", "bar", "baz"], drop=drop, append=append)

        # non-existent key in list with arrays
        with pytest.raises(KeyError, match="X"):
            df.set_index([df["A"], df["B"], "X"], drop=drop, append=append)

        msg = "[('foo', 'foo', 'foo', 'bar', 'bar')]"
        # tuples always raise KeyError
        with pytest.raises(KeyError, match=msg):
            df.set_index(tuple(df["A"]), drop=drop, append=append)

        # also within a list
        with pytest.raises(KeyError, match=msg):
            df.set_index(["A", df["A"], tuple(df["A"])], drop=drop, append=append)

    @pytest.mark.parametrize("append", [True, False])
    @pytest.mark.parametrize("drop", [True, False])
    @pytest.mark.parametrize("box", [set], ids=["set"])
    def test_set_index_raise_on_type(self, frame_of_index_cols, box, drop, append):
        df = frame_of_index_cols

        msg = 'The parameter "keys" may be a column key, .*'
        # forbidden type, e.g. set
        with pytest.raises(TypeError, match=msg):
            df.set_index(box(df["A"]), drop=drop, append=append)

        # forbidden type in list, e.g. set
        with pytest.raises(TypeError, match=msg):
            df.set_index(["A", df["A"], box(df["A"])], drop=drop, append=append)

    # MultiIndex constructor does not work directly on Series -> lambda
    @pytest.mark.parametrize(
        "box",
        [Series, Index, np.array, iter, lambda x: MultiIndex.from_arrays([x])],
        ids=["Series", "Index", "np.array", "iter", "MultiIndex"],
    )
    @pytest.mark.parametrize("length", [4, 6], ids=["too_short", "too_long"])
    @pytest.mark.parametrize("append", [True, False])
    @pytest.mark.parametrize("drop", [True, False])
    def test_set_index_raise_on_len(
        self, frame_of_index_cols, box, length, drop, append
    ):
        # GH 24984
        df = frame_of_index_cols  # has length 5

        values = np.random.default_rng(2).integers(0, 10, (length,))

        msg = "Length mismatch: Expected 5 rows, received array of length.*"

        # wrong length directly
        with pytest.raises(ValueError, match=msg):
            df.set_index(box(values), drop=drop, append=append)

        # wrong length in list
        with pytest.raises(ValueError, match=msg):
            df.set_index(["A", df.A, box(values)], drop=drop, append=append)


class TestSetIndexCustomLabelType:
    def test_set_index_custom_label_type(self):
        # GH#24969

        class Thing:
            def __init__(self, name, color) -> None:
                self.name = name
                self.color = color

            def __str__(self) -> str:
                return f"<Thing {repr(self.name)}>"

            # necessary for pretty KeyError
            __repr__ = __str__

        thing1 = Thing("One", "red")
        thing2 = Thing("Two", "blue")
        df = DataFrame({thing1: [0, 1], thing2: [2, 3]})
        expected = DataFrame({thing1: [0, 1]}, index=Index([2, 3], name=thing2))

        # use custom label directly
        result = df.set_index(thing2)
        tm.assert_frame_equal(result, expected)

        # custom label wrapped in list
        result = df.set_index([thing2])
        tm.assert_frame_equal(result, expected)

        # missing key
        thing3 = Thing("Three", "pink")
        msg = "<Thing 'Three'>"
        with pytest.raises(KeyError, match=msg):
            # missing label directly
            df.set_index(thing3)

        with pytest.raises(KeyError, match=msg):
            # missing label in list
            df.set_index([thing3])

    def test_set_index_custom_label_hashable_iterable(self):
        # GH#24969

        # actual example discussed in GH 24984 was e.g. for shapely.geometry
        # objects (e.g. a collection of Points) that can be both hashable and
        # iterable; using frozenset as a stand-in for testing here

        class Thing(frozenset):
            # need to stabilize repr for KeyError (due to random order in sets)
            def __repr__(self) -> str:
                tmp = sorted(self)
                joined_reprs = ", ".join(map(repr, tmp))
                # double curly brace prints one brace in format string
                return f"frozenset({{{joined_reprs}}})"

        thing1 = Thing(["One", "red"])
        thing2 = Thing(["Two", "blue"])
        df = DataFrame({thing1: [0, 1], thing2: [2, 3]})
        expected = DataFrame({thing1: [0, 1]}, index=Index([2, 3], name=thing2))

        # use custom label directly
        result = df.set_index(thing2)
        tm.assert_frame_equal(result, expected)

        # custom label wrapped in list
        result = df.set_index([thing2])
        tm.assert_frame_equal(result, expected)

        # missing key
        thing3 = Thing(["Three", "pink"])
        msg = r"frozenset\(\{'Three', 'pink'\}\)"
        with pytest.raises(KeyError, match=msg):
            # missing label directly
            df.set_index(thing3)

        with pytest.raises(KeyError, match=msg):
            # missing label in list
            df.set_index([thing3])

    def test_set_index_custom_label_type_raises(self):
        # GH#24969

        # purposefully inherit from something unhashable
        class Thing(set):
            def __init__(self, name, color) -> None:
                self.name = name
                self.color = color

            def __str__(self) -> str:
                return f"<Thing {repr(self.name)}>"

        thing1 = Thing("One", "red")
        thing2 = Thing("Two", "blue")
        df = DataFrame([[0, 2], [1, 3]], columns=[thing1, thing2])

        msg = 'The parameter "keys" may be a column key, .*'

        with pytest.raises(TypeError, match=msg):
            # use custom label directly
            df.set_index(thing2)

        with pytest.raises(TypeError, match=msg):
            # custom label wrapped in list
            df.set_index([thing2])

    def test_set_index_periodindex(self):
        # GH#6631
        df = DataFrame(np.random.default_rng(2).random(6))
        idx1 = period_range("2011/01/01", periods=6, freq="M")
        idx2 = period_range("2013", periods=6, freq="Y")

        df = df.set_index(idx1)
        tm.assert_index_equal(df.index, idx1)
        df = df.set_index(idx2)
        tm.assert_index_equal(df.index, idx2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\strings\test_split_partition.py ===
from datetime import datetime
import re

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    _testing as tm,
)
from pandas.tests.strings import (
    _convert_na_value,
    is_object_or_nan_string_dtype,
)


@pytest.mark.parametrize("method", ["split", "rsplit"])
def test_split(any_string_dtype, method):
    values = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"], dtype=any_string_dtype)

    result = getattr(values.str, method)("_")
    exp = Series([["a", "b", "c"], ["c", "d", "e"], np.nan, ["f", "g", "h"]])
    exp = _convert_na_value(values, exp)
    tm.assert_series_equal(result, exp)


@pytest.mark.parametrize("method", ["split", "rsplit"])
def test_split_more_than_one_char(any_string_dtype, method):
    # more than one char
    values = Series(["a__b__c", "c__d__e", np.nan, "f__g__h"], dtype=any_string_dtype)
    result = getattr(values.str, method)("__")
    exp = Series([["a", "b", "c"], ["c", "d", "e"], np.nan, ["f", "g", "h"]])
    exp = _convert_na_value(values, exp)
    tm.assert_series_equal(result, exp)

    result = getattr(values.str, method)("__", expand=False)
    tm.assert_series_equal(result, exp)


def test_split_more_regex_split(any_string_dtype):
    # regex split
    values = Series(["a,b_c", "c_d,e", np.nan, "f,g,h"], dtype=any_string_dtype)
    result = values.str.split("[,_]")
    exp = Series([["a", "b", "c"], ["c", "d", "e"], np.nan, ["f", "g", "h"]])
    exp = _convert_na_value(values, exp)
    tm.assert_series_equal(result, exp)


def test_split_regex(any_string_dtype):
    # GH 43563
    # explicit regex = True split
    values = Series("xxxjpgzzz.jpg", dtype=any_string_dtype)
    result = values.str.split(r"\.jpg", regex=True)
    exp = Series([["xxxjpgzzz", ""]])
    tm.assert_series_equal(result, exp)


def test_split_regex_explicit(any_string_dtype):
    # explicit regex = True split with compiled regex
    regex_pat = re.compile(r".jpg")
    values = Series("xxxjpgzzz.jpg", dtype=any_string_dtype)
    result = values.str.split(regex_pat)
    exp = Series([["xx", "zzz", ""]])
    tm.assert_series_equal(result, exp)

    # explicit regex = False split
    result = values.str.split(r"\.jpg", regex=False)
    exp = Series([["xxxjpgzzz.jpg"]])
    tm.assert_series_equal(result, exp)

    # non explicit regex split, pattern length == 1
    result = values.str.split(r".")
    exp = Series([["xxxjpgzzz", "jpg"]])
    tm.assert_series_equal(result, exp)

    # non explicit regex split, pattern length != 1
    result = values.str.split(r".jpg")
    exp = Series([["xx", "zzz", ""]])
    tm.assert_series_equal(result, exp)

    # regex=False with pattern compiled regex raises error
    with pytest.raises(
        ValueError,
        match="Cannot use a compiled regex as replacement pattern with regex=False",
    ):
        values.str.split(regex_pat, regex=False)


@pytest.mark.parametrize("expand", [None, False])
@pytest.mark.parametrize("method", ["split", "rsplit"])
def test_split_object_mixed(expand, method):
    mixed = Series(["a_b_c", np.nan, "d_e_f", True, datetime.today(), None, 1, 2.0])
    result = getattr(mixed.str, method)("_", expand=expand)
    exp = Series(
        [
            ["a", "b", "c"],
            np.nan,
            ["d", "e", "f"],
            np.nan,
            np.nan,
            None,
            np.nan,
            np.nan,
        ]
    )
    assert isinstance(result, Series)
    tm.assert_almost_equal(result, exp)


@pytest.mark.parametrize("method", ["split", "rsplit"])
@pytest.mark.parametrize("n", [None, 0])
def test_split_n(any_string_dtype, method, n):
    s = Series(["a b", pd.NA, "b c"], dtype=any_string_dtype)
    expected = Series([["a", "b"], pd.NA, ["b", "c"]])
    result = getattr(s.str, method)(" ", n=n)
    expected = _convert_na_value(s, expected)
    tm.assert_series_equal(result, expected)


def test_rsplit(any_string_dtype):
    # regex split is not supported by rsplit
    values = Series(["a,b_c", "c_d,e", np.nan, "f,g,h"], dtype=any_string_dtype)
    result = values.str.rsplit("[,_]")
    exp = Series([["a,b_c"], ["c_d,e"], np.nan, ["f,g,h"]])
    exp = _convert_na_value(values, exp)
    tm.assert_series_equal(result, exp)


def test_rsplit_max_number(any_string_dtype):
    # setting max number of splits, make sure it's from reverse
    values = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"], dtype=any_string_dtype)
    result = values.str.rsplit("_", n=1)
    exp = Series([["a_b", "c"], ["c_d", "e"], np.nan, ["f_g", "h"]])
    exp = _convert_na_value(values, exp)
    tm.assert_series_equal(result, exp)


def test_split_blank_string(any_string_dtype):
    # expand blank split GH 20067
    values = Series([""], name="test", dtype=any_string_dtype)
    result = values.str.split(expand=True)
    exp = DataFrame([[]], dtype=any_string_dtype)  # NOTE: this is NOT an empty df
    tm.assert_frame_equal(result, exp)


def test_split_blank_string_with_non_empty(any_string_dtype):
    values = Series(["a b c", "a b", "", " "], name="test", dtype=any_string_dtype)
    result = values.str.split(expand=True)
    exp = DataFrame(
        [
            ["a", "b", "c"],
            ["a", "b", None],
            [None, None, None],
            [None, None, None],
        ],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, exp)


@pytest.mark.parametrize("method", ["split", "rsplit"])
def test_split_noargs(any_string_dtype, method):
    # #1859
    s = Series(["Wes McKinney", "Travis  Oliphant"], dtype=any_string_dtype)
    result = getattr(s.str, method)()
    expected = ["Travis", "Oliphant"]
    assert result[1] == expected


@pytest.mark.parametrize(
    "data, pat",
    [
        (["bd asdf jfg", "kjasdflqw asdfnfk"], None),
        (["bd asdf jfg", "kjasdflqw asdfnfk"], "asdf"),
        (["bd_asdf_jfg", "kjasdflqw_asdfnfk"], "_"),
    ],
)
@pytest.mark.parametrize("n", [-1, 0])
def test_split_maxsplit(data, pat, any_string_dtype, n):
    # re.split 0, str.split -1
    s = Series(data, dtype=any_string_dtype)

    result = s.str.split(pat=pat, n=n)
    xp = s.str.split(pat=pat)
    tm.assert_series_equal(result, xp)


@pytest.mark.parametrize(
    "data, pat, expected",
    [
        (
            ["split once", "split once too!"],
            None,
            Series({0: ["split", "once"], 1: ["split", "once too!"]}),
        ),
        (
            ["split_once", "split_once_too!"],
            "_",
            Series({0: ["split", "once"], 1: ["split", "once_too!"]}),
        ),
    ],
)
def test_split_no_pat_with_nonzero_n(data, pat, expected, any_string_dtype):
    s = Series(data, dtype=any_string_dtype)
    result = s.str.split(pat=pat, n=1)
    tm.assert_series_equal(expected, result, check_index_type=False)


def test_split_to_dataframe_no_splits(any_string_dtype):
    s = Series(["nosplit", "alsonosplit"], dtype=any_string_dtype)
    result = s.str.split("_", expand=True)
    exp = DataFrame({0: Series(["nosplit", "alsonosplit"], dtype=any_string_dtype)})
    tm.assert_frame_equal(result, exp)


def test_split_to_dataframe(any_string_dtype):
    s = Series(["some_equal_splits", "with_no_nans"], dtype=any_string_dtype)
    result = s.str.split("_", expand=True)
    exp = DataFrame(
        {0: ["some", "with"], 1: ["equal", "no"], 2: ["splits", "nans"]},
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, exp)


def test_split_to_dataframe_unequal_splits(any_string_dtype):
    s = Series(
        ["some_unequal_splits", "one_of_these_things_is_not"], dtype=any_string_dtype
    )
    result = s.str.split("_", expand=True)
    exp = DataFrame(
        {
            0: ["some", "one"],
            1: ["unequal", "of"],
            2: ["splits", "these"],
            3: [None, "things"],
            4: [None, "is"],
            5: [None, "not"],
        },
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, exp)


def test_split_to_dataframe_with_index(any_string_dtype):
    s = Series(
        ["some_splits", "with_index"], index=["preserve", "me"], dtype=any_string_dtype
    )
    result = s.str.split("_", expand=True)
    exp = DataFrame(
        {0: ["some", "with"], 1: ["splits", "index"]},
        index=["preserve", "me"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, exp)

    with pytest.raises(ValueError, match="expand must be"):
        s.str.split("_", expand="not_a_boolean")


def test_split_to_multiindex_expand_no_splits():
    # https://github.com/pandas-dev/pandas/issues/23677

    idx = Index(["nosplit", "alsonosplit", np.nan])
    result = idx.str.split("_", expand=True)
    exp = idx
    tm.assert_index_equal(result, exp)
    assert result.nlevels == 1


def test_split_to_multiindex_expand():
    idx = Index(["some_equal_splits", "with_no_nans", np.nan, None])
    result = idx.str.split("_", expand=True)
    exp = MultiIndex.from_tuples(
        [
            ("some", "equal", "splits"),
            ("with", "no", "nans"),
            [np.nan, np.nan, np.nan],
            [None, None, None],
        ]
    )
    tm.assert_index_equal(result, exp)
    assert result.nlevels == 3


def test_split_to_multiindex_expand_unequal_splits():
    idx = Index(["some_unequal_splits", "one_of_these_things_is_not", np.nan, None])
    result = idx.str.split("_", expand=True)
    exp = MultiIndex.from_tuples(
        [
            ("some", "unequal", "splits", np.nan, np.nan, np.nan),
            ("one", "of", "these", "things", "is", "not"),
            (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
            (None, None, None, None, None, None),
        ]
    )
    tm.assert_index_equal(result, exp)
    assert result.nlevels == 6

    with pytest.raises(ValueError, match="expand must be"):
        idx.str.split("_", expand="not_a_boolean")


def test_rsplit_to_dataframe_expand_no_splits(any_string_dtype):
    s = Series(["nosplit", "alsonosplit"], dtype=any_string_dtype)
    result = s.str.rsplit("_", expand=True)
    exp = DataFrame({0: Series(["nosplit", "alsonosplit"])}, dtype=any_string_dtype)
    tm.assert_frame_equal(result, exp)


def test_rsplit_to_dataframe_expand(any_string_dtype):
    s = Series(["some_equal_splits", "with_no_nans"], dtype=any_string_dtype)
    result = s.str.rsplit("_", expand=True)
    exp = DataFrame(
        {0: ["some", "with"], 1: ["equal", "no"], 2: ["splits", "nans"]},
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, exp)

    result = s.str.rsplit("_", expand=True, n=2)
    exp = DataFrame(
        {0: ["some", "with"], 1: ["equal", "no"], 2: ["splits", "nans"]},
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, exp)

    result = s.str.rsplit("_", expand=True, n=1)
    exp = DataFrame(
        {0: ["some_equal", "with_no"], 1: ["splits", "nans"]}, dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, exp)


def test_rsplit_to_dataframe_expand_with_index(any_string_dtype):
    s = Series(
        ["some_splits", "with_index"], index=["preserve", "me"], dtype=any_string_dtype
    )
    result = s.str.rsplit("_", expand=True)
    exp = DataFrame(
        {0: ["some", "with"], 1: ["splits", "index"]},
        index=["preserve", "me"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, exp)


def test_rsplit_to_multiindex_expand_no_split():
    idx = Index(["nosplit", "alsonosplit"])
    result = idx.str.rsplit("_", expand=True)
    exp = idx
    tm.assert_index_equal(result, exp)
    assert result.nlevels == 1


def test_rsplit_to_multiindex_expand():
    idx = Index(["some_equal_splits", "with_no_nans"])
    result = idx.str.rsplit("_", expand=True)
    exp = MultiIndex.from_tuples([("some", "equal", "splits"), ("with", "no", "nans")])
    tm.assert_index_equal(result, exp)
    assert result.nlevels == 3


def test_rsplit_to_multiindex_expand_n():
    idx = Index(["some_equal_splits", "with_no_nans"])
    result = idx.str.rsplit("_", expand=True, n=1)
    exp = MultiIndex.from_tuples([("some_equal", "splits"), ("with_no", "nans")])
    tm.assert_index_equal(result, exp)
    assert result.nlevels == 2


def test_split_nan_expand(any_string_dtype):
    # gh-18450
    s = Series(["foo,bar,baz", np.nan], dtype=any_string_dtype)
    result = s.str.split(",", expand=True)
    exp = DataFrame(
        [["foo", "bar", "baz"], [np.nan, np.nan, np.nan]], dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, exp)

    # check that these are actually np.nan/pd.NA and not None
    # TODO see GH 18463
    # tm.assert_frame_equal does not differentiate
    if is_object_or_nan_string_dtype(any_string_dtype):
        assert all(np.isnan(x) for x in result.iloc[1])
    else:
        assert all(x is pd.NA for x in result.iloc[1])


def test_split_with_name_series(any_string_dtype):
    # GH 12617

    # should preserve name
    s = Series(["a,b", "c,d"], name="xxx", dtype=any_string_dtype)
    res = s.str.split(",")
    exp = Series([["a", "b"], ["c", "d"]], name="xxx")
    tm.assert_series_equal(res, exp)

    res = s.str.split(",", expand=True)
    exp = DataFrame([["a", "b"], ["c", "d"]], dtype=any_string_dtype)
    tm.assert_frame_equal(res, exp)


def test_split_with_name_index():
    # GH 12617
    idx = Index(["a,b", "c,d"], name="xxx")
    res = idx.str.split(",")
    exp = Index([["a", "b"], ["c", "d"]], name="xxx")
    assert res.nlevels == 1
    tm.assert_index_equal(res, exp)

    res = idx.str.split(",", expand=True)
    exp = MultiIndex.from_tuples([("a", "b"), ("c", "d")])
    assert res.nlevels == 2
    tm.assert_index_equal(res, exp)


@pytest.mark.parametrize(
    "method, exp",
    [
        [
            "partition",
            [
                ("a", "__", "b__c"),
                ("c", "__", "d__e"),
                np.nan,
                ("f", "__", "g__h"),
                None,
            ],
        ],
        [
            "rpartition",
            [
                ("a__b", "__", "c"),
                ("c__d", "__", "e"),
                np.nan,
                ("f__g", "__", "h"),
                None,
            ],
        ],
    ],
)
def test_partition_series_more_than_one_char(method, exp, any_string_dtype):
    # https://github.com/pandas-dev/pandas/issues/23558
    # more than one char
    s = Series(["a__b__c", "c__d__e", np.nan, "f__g__h", None], dtype=any_string_dtype)
    result = getattr(s.str, method)("__", expand=False)
    expected = Series(exp)
    expected = _convert_na_value(s, expected)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method, exp",
    [
        [
            "partition",
            [("a", " ", "b c"), ("c", " ", "d e"), np.nan, ("f", " ", "g h"), None],
        ],
        [
            "rpartition",
            [("a b", " ", "c"), ("c d", " ", "e"), np.nan, ("f g", " ", "h"), None],
        ],
    ],
)
def test_partition_series_none(any_string_dtype, method, exp):
    # https://github.com/pandas-dev/pandas/issues/23558
    # None
    s = Series(["a b c", "c d e", np.nan, "f g h", None], dtype=any_string_dtype)
    result = getattr(s.str, method)(expand=False)
    expected = Series(exp)
    expected = _convert_na_value(s, expected)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method, exp",
    [
        [
            "partition",
            [("abc", "", ""), ("cde", "", ""), np.nan, ("fgh", "", ""), None],
        ],
        [
            "rpartition",
            [("", "", "abc"), ("", "", "cde"), np.nan, ("", "", "fgh"), None],
        ],
    ],
)
def test_partition_series_not_split(any_string_dtype, method, exp):
    # https://github.com/pandas-dev/pandas/issues/23558
    # Not split
    s = Series(["abc", "cde", np.nan, "fgh", None], dtype=any_string_dtype)
    result = getattr(s.str, method)("_", expand=False)
    expected = Series(exp)
    expected = _convert_na_value(s, expected)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method, exp",
    [
        [
            "partition",
            [("a", "_", "b_c"), ("c", "_", "d_e"), np.nan, ("f", "_", "g_h")],
        ],
        [
            "rpartition",
            [("a_b", "_", "c"), ("c_d", "_", "e"), np.nan, ("f_g", "_", "h")],
        ],
    ],
)
def test_partition_series_unicode(any_string_dtype, method, exp):
    # https://github.com/pandas-dev/pandas/issues/23558
    # unicode
    s = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"], dtype=any_string_dtype)

    result = getattr(s.str, method)("_", expand=False)
    expected = Series(exp)
    expected = _convert_na_value(s, expected)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("method", ["partition", "rpartition"])
def test_partition_series_stdlib(any_string_dtype, method):
    # https://github.com/pandas-dev/pandas/issues/23558
    # compare to standard lib
    s = Series(["A_B_C", "B_C_D", "E_F_G", "EFGHEF"], dtype=any_string_dtype)
    result = getattr(s.str, method)("_", expand=False).tolist()
    assert result == [getattr(v, method)("_") for v in s]


@pytest.mark.parametrize(
    "method, expand, exp, exp_levels",
    [
        [
            "partition",
            False,
            np.array(
                [("a", "_", "b_c"), ("c", "_", "d_e"), ("f", "_", "g_h"), np.nan, None],
                dtype=object,
            ),
            1,
        ],
        [
            "rpartition",
            False,
            np.array(
                [("a_b", "_", "c"), ("c_d", "_", "e"), ("f_g", "_", "h"), np.nan, None],
                dtype=object,
            ),
            1,
        ],
    ],
)
def test_partition_index(method, expand, exp, exp_levels):
    # https://github.com/pandas-dev/pandas/issues/23558

    values = Index(["a_b_c", "c_d_e", "f_g_h", np.nan, None])

    result = getattr(values.str, method)("_", expand=expand)
    exp = Index(exp)
    tm.assert_index_equal(result, exp)
    assert result.nlevels == exp_levels


@pytest.mark.parametrize(
    "method, exp",
    [
        [
            "partition",
            {
                0: ["a", "c", np.nan, "f", None],
                1: ["_", "_", np.nan, "_", None],
                2: ["b_c", "d_e", np.nan, "g_h", None],
            },
        ],
        [
            "rpartition",
            {
                0: ["a_b", "c_d", np.nan, "f_g", None],
                1: ["_", "_", np.nan, "_", None],
                2: ["c", "e", np.nan, "h", None],
            },
        ],
    ],
)
def test_partition_to_dataframe(any_string_dtype, method, exp):
    # https://github.com/pandas-dev/pandas/issues/23558

    s = Series(["a_b_c", "c_d_e", np.nan, "f_g_h", None], dtype=any_string_dtype)
    result = getattr(s.str, method)("_")
    expected = DataFrame(
        exp,
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "method, exp",
    [
        [
            "partition",
            {
                0: ["a", "c", np.nan, "f", None],
                1: ["_", "_", np.nan, "_", None],
                2: ["b_c", "d_e", np.nan, "g_h", None],
            },
        ],
        [
            "rpartition",
            {
                0: ["a_b", "c_d", np.nan, "f_g", None],
                1: ["_", "_", np.nan, "_", None],
                2: ["c", "e", np.nan, "h", None],
            },
        ],
    ],
)
def test_partition_to_dataframe_from_series(any_string_dtype, method, exp):
    # https://github.com/pandas-dev/pandas/issues/23558
    s = Series(["a_b_c", "c_d_e", np.nan, "f_g_h", None], dtype=any_string_dtype)
    result = getattr(s.str, method)("_", expand=True)
    expected = DataFrame(
        exp,
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)


def test_partition_with_name(any_string_dtype):
    # GH 12617

    s = Series(["a,b", "c,d"], name="xxx", dtype=any_string_dtype)
    result = s.str.partition(",")
    expected = DataFrame(
        {0: ["a", "c"], 1: [",", ","], 2: ["b", "d"]}, dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)


def test_partition_with_name_expand(any_string_dtype):
    # GH 12617
    # should preserve name
    s = Series(["a,b", "c,d"], name="xxx", dtype=any_string_dtype)
    result = s.str.partition(",", expand=False)
    expected = Series([("a", ",", "b"), ("c", ",", "d")], name="xxx")
    tm.assert_series_equal(result, expected)


def test_partition_index_with_name():
    idx = Index(["a,b", "c,d"], name="xxx")
    result = idx.str.partition(",")
    expected = MultiIndex.from_tuples([("a", ",", "b"), ("c", ",", "d")])
    assert result.nlevels == 3
    tm.assert_index_equal(result, expected)


def test_partition_index_with_name_expand_false():
    idx = Index(["a,b", "c,d"], name="xxx")
    # should preserve name
    result = idx.str.partition(",", expand=False)
    expected = Index(np.array([("a", ",", "b"), ("c", ",", "d")]), name="xxx")
    assert result.nlevels == 1
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("method", ["partition", "rpartition"])
def test_partition_sep_kwarg(any_string_dtype, method):
    # GH 22676; depr kwarg "pat" in favor of "sep"
    s = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"], dtype=any_string_dtype)

    expected = getattr(s.str, method)(sep="_")
    result = getattr(s.str, method)("_")
    tm.assert_frame_equal(result, expected)


def test_get():
    ser = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"])
    result = ser.str.split("_").str.get(1)
    expected = Series(["b", "d", np.nan, "g"], dtype=object)
    tm.assert_series_equal(result, expected)


def test_get_mixed_object():
    ser = Series(["a_b_c", np.nan, "c_d_e", True, datetime.today(), None, 1, 2.0])
    result = ser.str.split("_").str.get(1)
    expected = Series(
        ["b", np.nan, "d", np.nan, np.nan, None, np.nan, np.nan], dtype=object
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("idx", [2, -3])
def test_get_bounds(idx):
    ser = Series(["1_2_3_4_5", "6_7_8_9_10", "11_12"])
    result = ser.str.split("_").str.get(idx)
    expected = Series(["3", "8", np.nan], dtype=object)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "idx, exp", [[2, [3, 3, np.nan, "b"]], [-1, [3, 3, np.nan, np.nan]]]
)
def test_get_complex(idx, exp):
    # GH 20671, getting value not in dict raising `KeyError`
    ser = Series([(1, 2, 3), [1, 2, 3], {1, 2, 3}, {1: "a", 2: "b", 3: "c"}])

    result = ser.str.get(idx)
    expected = Series(exp)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("to_type", [tuple, list, np.array])
def test_get_complex_nested(to_type):
    ser = Series([to_type([to_type([1, 2])])])

    result = ser.str.get(0)
    expected = Series([to_type([1, 2])])
    tm.assert_series_equal(result, expected)

    result = ser.str.get(1)
    expected = Series([np.nan])
    tm.assert_series_equal(result, expected)


def test_get_strings(any_string_dtype):
    ser = Series(["a", "ab", np.nan, "abc"], dtype=any_string_dtype)
    result = ser.str.get(2)
    expected = Series([np.nan, np.nan, np.nan, "c"], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\test_header.py ===
"""
Tests that the file header is properly handled or inferred
during parsing for all of the parsers defined in parsers.py
"""

from collections import namedtuple
from io import StringIO

import numpy as np
import pytest

from pandas.errors import ParserError

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


@xfail_pyarrow  # TypeError: an integer is required
def test_read_with_bad_header(all_parsers):
    parser = all_parsers
    msg = r"but only \d+ lines in file"

    with pytest.raises(ValueError, match=msg):
        s = StringIO(",,")
        parser.read_csv(s, header=[10])


def test_negative_header(all_parsers):
    # see gh-27779
    parser = all_parsers
    data = """1,2,3,4,5
6,7,8,9,10
11,12,13,14,15
"""
    with pytest.raises(
        ValueError,
        match="Passing negative integer to header is invalid. "
        "For no header, use header=None instead",
    ):
        parser.read_csv(StringIO(data), header=-1)


@pytest.mark.parametrize("header", [([-1, 2, 4]), ([-5, 0])])
def test_negative_multi_index_header(all_parsers, header):
    # see gh-27779
    parser = all_parsers
    data = """1,2,3,4,5
        6,7,8,9,10
        11,12,13,14,15
        """
    with pytest.raises(
        ValueError, match="cannot specify multi-index header with negative integers"
    ):
        parser.read_csv(StringIO(data), header=header)


@pytest.mark.parametrize("header", [True, False])
def test_bool_header_arg(all_parsers, header):
    # see gh-6114
    parser = all_parsers
    data = """\
MyColumn
a
b
a
b"""
    msg = "Passing a bool to header is invalid"
    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), header=header)


@xfail_pyarrow  # AssertionError: DataFrame are different
def test_header_with_index_col(all_parsers):
    parser = all_parsers
    data = """foo,1,2,3
bar,4,5,6
baz,7,8,9
"""
    names = ["A", "B", "C"]
    result = parser.read_csv(StringIO(data), names=names)

    expected = DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        index=["foo", "bar", "baz"],
        columns=["A", "B", "C"],
    )
    tm.assert_frame_equal(result, expected)


def test_header_not_first_line(all_parsers):
    parser = all_parsers
    data = """got,to,ignore,this,line
got,to,ignore,this,line
index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
"""
    data2 = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
"""

    result = parser.read_csv(StringIO(data), header=2, index_col=0)
    expected = parser.read_csv(StringIO(data2), header=0, index_col=0)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
def test_header_multi_index(all_parsers):
    parser = all_parsers

    data = """\
C0,,C_l0_g0,C_l0_g1,C_l0_g2

C1,,C_l1_g0,C_l1_g1,C_l1_g2
C2,,C_l2_g0,C_l2_g1,C_l2_g2
C3,,C_l3_g0,C_l3_g1,C_l3_g2
R0,R1,,,
R_l0_g0,R_l1_g0,R0C0,R0C1,R0C2
R_l0_g1,R_l1_g1,R1C0,R1C1,R1C2
R_l0_g2,R_l1_g2,R2C0,R2C1,R2C2
R_l0_g3,R_l1_g3,R3C0,R3C1,R3C2
R_l0_g4,R_l1_g4,R4C0,R4C1,R4C2
"""
    result = parser.read_csv(StringIO(data), header=[0, 1, 2, 3], index_col=[0, 1])
    data_gen_f = lambda r, c: f"R{r}C{c}"

    data = [[data_gen_f(r, c) for c in range(3)] for r in range(5)]
    index = MultiIndex.from_arrays(
        [[f"R_l0_g{i}" for i in range(5)], [f"R_l1_g{i}" for i in range(5)]],
        names=["R0", "R1"],
    )
    columns = MultiIndex.from_arrays(
        [
            [f"C_l0_g{i}" for i in range(3)],
            [f"C_l1_g{i}" for i in range(3)],
            [f"C_l2_g{i}" for i in range(3)],
            [f"C_l3_g{i}" for i in range(3)],
        ],
        names=["C0", "C1", "C2", "C3"],
    )
    expected = DataFrame(data, columns=columns, index=index)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "kwargs,msg",
    [
        (
            {"index_col": ["foo", "bar"]},
            (
                "index_col must only contain "
                "row numbers when specifying "
                "a multi-index header"
            ),
        ),
        (
            {"index_col": [0, 1], "names": ["foo", "bar"]},
            ("cannot specify names when specifying a multi-index header"),
        ),
        (
            {"index_col": [0, 1], "usecols": ["foo", "bar"]},
            ("cannot specify usecols when specifying a multi-index header"),
        ),
    ],
)
def test_header_multi_index_invalid(all_parsers, kwargs, msg):
    data = """\
C0,,C_l0_g0,C_l0_g1,C_l0_g2

C1,,C_l1_g0,C_l1_g1,C_l1_g2
C2,,C_l2_g0,C_l2_g1,C_l2_g2
C3,,C_l3_g0,C_l3_g1,C_l3_g2
R0,R1,,,
R_l0_g0,R_l1_g0,R0C0,R0C1,R0C2
R_l0_g1,R_l1_g1,R1C0,R1C1,R1C2
R_l0_g2,R_l1_g2,R2C0,R2C1,R2C2
R_l0_g3,R_l1_g3,R3C0,R3C1,R3C2
R_l0_g4,R_l1_g4,R4C0,R4C1,R4C2
"""
    parser = all_parsers

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), header=[0, 1, 2, 3], **kwargs)


_TestTuple = namedtuple("_TestTuple", ["first", "second"])


@xfail_pyarrow  # TypeError: an integer is required
@pytest.mark.parametrize(
    "kwargs",
    [
        {"header": [0, 1]},
        {
            "skiprows": 3,
            "names": [
                ("a", "q"),
                ("a", "r"),
                ("a", "s"),
                ("b", "t"),
                ("c", "u"),
                ("c", "v"),
            ],
        },
        {
            "skiprows": 3,
            "names": [
                _TestTuple("a", "q"),
                _TestTuple("a", "r"),
                _TestTuple("a", "s"),
                _TestTuple("b", "t"),
                _TestTuple("c", "u"),
                _TestTuple("c", "v"),
            ],
        },
    ],
)
def test_header_multi_index_common_format1(all_parsers, kwargs):
    parser = all_parsers
    expected = DataFrame(
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
        index=["one", "two"],
        columns=MultiIndex.from_tuples(
            [("a", "q"), ("a", "r"), ("a", "s"), ("b", "t"), ("c", "u"), ("c", "v")]
        ),
    )
    data = """,a,a,a,b,c,c
,q,r,s,t,u,v
,,,,,,
one,1,2,3,4,5,6
two,7,8,9,10,11,12"""

    result = parser.read_csv(StringIO(data), index_col=0, **kwargs)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
@pytest.mark.parametrize(
    "kwargs",
    [
        {"header": [0, 1]},
        {
            "skiprows": 2,
            "names": [
                ("a", "q"),
                ("a", "r"),
                ("a", "s"),
                ("b", "t"),
                ("c", "u"),
                ("c", "v"),
            ],
        },
        {
            "skiprows": 2,
            "names": [
                _TestTuple("a", "q"),
                _TestTuple("a", "r"),
                _TestTuple("a", "s"),
                _TestTuple("b", "t"),
                _TestTuple("c", "u"),
                _TestTuple("c", "v"),
            ],
        },
    ],
)
def test_header_multi_index_common_format2(all_parsers, kwargs):
    parser = all_parsers
    expected = DataFrame(
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
        index=["one", "two"],
        columns=MultiIndex.from_tuples(
            [("a", "q"), ("a", "r"), ("a", "s"), ("b", "t"), ("c", "u"), ("c", "v")]
        ),
    )
    data = """,a,a,a,b,c,c
,q,r,s,t,u,v
one,1,2,3,4,5,6
two,7,8,9,10,11,12"""

    result = parser.read_csv(StringIO(data), index_col=0, **kwargs)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
@pytest.mark.parametrize(
    "kwargs",
    [
        {"header": [0, 1]},
        {
            "skiprows": 2,
            "names": [
                ("a", "q"),
                ("a", "r"),
                ("a", "s"),
                ("b", "t"),
                ("c", "u"),
                ("c", "v"),
            ],
        },
        {
            "skiprows": 2,
            "names": [
                _TestTuple("a", "q"),
                _TestTuple("a", "r"),
                _TestTuple("a", "s"),
                _TestTuple("b", "t"),
                _TestTuple("c", "u"),
                _TestTuple("c", "v"),
            ],
        },
    ],
)
def test_header_multi_index_common_format3(all_parsers, kwargs):
    parser = all_parsers
    expected = DataFrame(
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
        index=["one", "two"],
        columns=MultiIndex.from_tuples(
            [("a", "q"), ("a", "r"), ("a", "s"), ("b", "t"), ("c", "u"), ("c", "v")]
        ),
    )
    expected = expected.reset_index(drop=True)
    data = """a,a,a,b,c,c
q,r,s,t,u,v
1,2,3,4,5,6
7,8,9,10,11,12"""

    result = parser.read_csv(StringIO(data), index_col=None, **kwargs)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
def test_header_multi_index_common_format_malformed1(all_parsers):
    parser = all_parsers
    expected = DataFrame(
        np.array([[2, 3, 4, 5, 6], [8, 9, 10, 11, 12]], dtype="int64"),
        index=Index([1, 7]),
        columns=MultiIndex(
            levels=[["a", "b", "c"], ["r", "s", "t", "u", "v"]],
            codes=[[0, 0, 1, 2, 2], [0, 1, 2, 3, 4]],
            names=["a", "q"],
        ),
    )
    data = """a,a,a,b,c,c
q,r,s,t,u,v
1,2,3,4,5,6
7,8,9,10,11,12"""

    result = parser.read_csv(StringIO(data), header=[0, 1], index_col=0)
    tm.assert_frame_equal(expected, result)


@xfail_pyarrow  # TypeError: an integer is required
def test_header_multi_index_common_format_malformed2(all_parsers):
    parser = all_parsers
    expected = DataFrame(
        np.array([[2, 3, 4, 5, 6], [8, 9, 10, 11, 12]], dtype="int64"),
        index=Index([1, 7]),
        columns=MultiIndex(
            levels=[["a", "b", "c"], ["r", "s", "t", "u", "v"]],
            codes=[[0, 0, 1, 2, 2], [0, 1, 2, 3, 4]],
            names=[None, "q"],
        ),
    )

    data = """,a,a,b,c,c
q,r,s,t,u,v
1,2,3,4,5,6
7,8,9,10,11,12"""

    result = parser.read_csv(StringIO(data), header=[0, 1], index_col=0)
    tm.assert_frame_equal(expected, result)


@xfail_pyarrow  # TypeError: an integer is required
def test_header_multi_index_common_format_malformed3(all_parsers):
    parser = all_parsers
    expected = DataFrame(
        np.array([[3, 4, 5, 6], [9, 10, 11, 12]], dtype="int64"),
        index=MultiIndex(levels=[[1, 7], [2, 8]], codes=[[0, 1], [0, 1]]),
        columns=MultiIndex(
            levels=[["a", "b", "c"], ["s", "t", "u", "v"]],
            codes=[[0, 1, 2, 2], [0, 1, 2, 3]],
            names=[None, "q"],
        ),
    )
    data = """,a,a,b,c,c
q,r,s,t,u,v
1,2,3,4,5,6
7,8,9,10,11,12"""

    result = parser.read_csv(StringIO(data), header=[0, 1], index_col=[0, 1])
    tm.assert_frame_equal(expected, result)


@xfail_pyarrow  # TypeError: an integer is required
def test_header_multi_index_blank_line(all_parsers):
    # GH 40442
    parser = all_parsers
    data = [[None, None], [1, 2], [3, 4]]
    columns = MultiIndex.from_tuples([("a", "A"), ("b", "B")])
    expected = DataFrame(data, columns=columns)
    data = "a,b\nA,B\n,\n1,2\n3,4"
    result = parser.read_csv(StringIO(data), header=[0, 1])
    tm.assert_frame_equal(expected, result)


@pytest.mark.parametrize(
    "data,header", [("1,2,3\n4,5,6", None), ("foo,bar,baz\n1,2,3\n4,5,6", 0)]
)
def test_header_names_backward_compat(all_parsers, data, header, request):
    # see gh-2539
    parser = all_parsers

    if parser.engine == "pyarrow" and header is not None:
        mark = pytest.mark.xfail(reason="DataFrame.columns are different")
        request.applymarker(mark)

    expected = parser.read_csv(StringIO("1,2,3\n4,5,6"), names=["a", "b", "c"])

    result = parser.read_csv(StringIO(data), names=["a", "b", "c"], header=header)
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block: cannot infer
@pytest.mark.parametrize("kwargs", [{}, {"index_col": False}])
def test_read_only_header_no_rows(all_parsers, kwargs):
    # See gh-7773
    parser = all_parsers
    expected = DataFrame(columns=["a", "b", "c"])

    result = parser.read_csv(StringIO("a,b,c"), **kwargs)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "kwargs,names",
    [
        ({}, [0, 1, 2, 3, 4]),
        (
            {"names": ["foo", "bar", "baz", "quux", "panda"]},
            ["foo", "bar", "baz", "quux", "panda"],
        ),
    ],
)
def test_no_header(all_parsers, kwargs, names):
    parser = all_parsers
    data = """1,2,3,4,5
6,7,8,9,10
11,12,13,14,15
"""
    expected = DataFrame(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]], columns=names
    )
    result = parser.read_csv(StringIO(data), header=None, **kwargs)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("header", [["a", "b"], "string_header"])
def test_non_int_header(all_parsers, header):
    # see gh-16338
    msg = "header must be integer or list of integers"
    data = """1,2\n3,4"""
    parser = all_parsers

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), header=header)


@xfail_pyarrow  # TypeError: an integer is required
def test_singleton_header(all_parsers):
    # see gh-7757
    data = """a,b,c\n0,1,2\n1,2,3"""
    parser = all_parsers

    expected = DataFrame({"a": [0, 1], "b": [1, 2], "c": [2, 3]})
    result = parser.read_csv(StringIO(data), header=[0])
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
@pytest.mark.parametrize(
    "data,expected",
    [
        (
            "A,A,A,B\none,one,one,two\n0,40,34,0.1",
            DataFrame(
                [[0, 40, 34, 0.1]],
                columns=MultiIndex.from_tuples(
                    [("A", "one"), ("A", "one.1"), ("A", "one.2"), ("B", "two")]
                ),
            ),
        ),
        (
            "A,A,A,B\none,one,one.1,two\n0,40,34,0.1",
            DataFrame(
                [[0, 40, 34, 0.1]],
                columns=MultiIndex.from_tuples(
                    [("A", "one"), ("A", "one.1"), ("A", "one.1.1"), ("B", "two")]
                ),
            ),
        ),
        (
            "A,A,A,B,B\none,one,one.1,two,two\n0,40,34,0.1,0.1",
            DataFrame(
                [[0, 40, 34, 0.1, 0.1]],
                columns=MultiIndex.from_tuples(
                    [
                        ("A", "one"),
                        ("A", "one.1"),
                        ("A", "one.1.1"),
                        ("B", "two"),
                        ("B", "two.1"),
                    ]
                ),
            ),
        ),
    ],
)
def test_mangles_multi_index(all_parsers, data, expected):
    # see gh-18062
    parser = all_parsers

    result = parser.read_csv(StringIO(data), header=[0, 1])
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is requireds
@pytest.mark.parametrize("index_col", [None, [0]])
@pytest.mark.parametrize(
    "columns", [None, (["", "Unnamed"]), (["Unnamed", ""]), (["Unnamed", "NotUnnamed"])]
)
def test_multi_index_unnamed(all_parsers, index_col, columns):
    # see gh-23687
    #
    # When specifying a multi-index header, make sure that
    # we don't error just because one of the rows in our header
    # has ALL column names containing the string "Unnamed". The
    # correct condition to check is whether the row contains
    # ALL columns that did not have names (and instead were given
    # placeholder ones).
    parser = all_parsers
    header = [0, 1]

    if index_col is None:
        data = ",".join(columns or ["", ""]) + "\n0,1\n2,3\n4,5\n"
    else:
        data = ",".join([""] + (columns or ["", ""])) + "\n,0,1\n0,2,3\n1,4,5\n"

    result = parser.read_csv(StringIO(data), header=header, index_col=index_col)
    exp_columns = []

    if columns is None:
        columns = ["", "", ""]

    for i, col in enumerate(columns):
        if not col:  # Unnamed.
            col = f"Unnamed: {i if index_col is None else i + 1}_level_0"

        exp_columns.append(col)

    columns = MultiIndex.from_tuples(zip(exp_columns, ["0", "1"]))
    expected = DataFrame([[2, 3], [4, 5]], columns=columns)
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Expected 2 columns, got 3
def test_names_longer_than_header_but_equal_with_data_rows(all_parsers):
    # GH#38453
    parser = all_parsers
    data = """a, b
1,2,3
5,6,4
"""
    result = parser.read_csv(StringIO(data), header=0, names=["A", "B", "C"])
    expected = DataFrame({"A": [1, 5], "B": [2, 6], "C": [3, 4]})
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
def test_read_csv_multiindex_columns(all_parsers):
    # GH#6051
    parser = all_parsers

    s1 = "Male, Male, Male, Female, Female\nR, R, L, R, R\n.86, .67, .88, .78, .81"
    s2 = (
        "Male, Male, Male, Female, Female\n"
        "R, R, L, R, R\n"
        ".86, .67, .88, .78, .81\n"
        ".86, .67, .88, .78, .82"
    )

    mi = MultiIndex.from_tuples(
        [
            ("Male", "R"),
            (" Male", " R"),
            (" Male", " L"),
            (" Female", " R"),
            (" Female", " R.1"),
        ]
    )
    expected = DataFrame(
        [[0.86, 0.67, 0.88, 0.78, 0.81], [0.86, 0.67, 0.88, 0.78, 0.82]], columns=mi
    )

    df1 = parser.read_csv(StringIO(s1), header=[0, 1])
    tm.assert_frame_equal(df1, expected.iloc[:1])
    df2 = parser.read_csv(StringIO(s2), header=[0, 1])
    tm.assert_frame_equal(df2, expected)


@xfail_pyarrow  # TypeError: an integer is required
def test_read_csv_multi_header_length_check(all_parsers):
    # GH#43102
    parser = all_parsers

    case = """row11,row12,row13
row21,row22, row23
row31,row32
"""

    with pytest.raises(
        ParserError, match="Header rows must have an equal number of columns."
    ):
        parser.read_csv(StringIO(case), header=[0, 2])


@skip_pyarrow  # CSV parse error: Expected 3 columns, got 2
def test_header_none_and_implicit_index(all_parsers):
    # GH#22144
    parser = all_parsers
    data = "x,1,5\ny,2\nz,3\n"
    result = parser.read_csv(StringIO(data), names=["a", "b"], header=None)
    expected = DataFrame(
        {"a": [1, 2, 3], "b": [5, np.nan, np.nan]}, index=["x", "y", "z"]
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # regex mismatch "CSV parse error: Expected 2 columns, got "
def test_header_none_and_implicit_index_in_second_row(all_parsers):
    # GH#22144
    parser = all_parsers
    data = "x,1\ny,2,5\nz,3\n"
    with pytest.raises(ParserError, match="Expected 2 fields in line 2, saw 3"):
        parser.read_csv(StringIO(data), names=["a", "b"], header=None)


def test_header_none_and_on_bad_lines_skip(all_parsers):
    # GH#22144
    parser = all_parsers
    data = "x,1\ny,2,5\nz,3\n"
    result = parser.read_csv(
        StringIO(data), names=["a", "b"], header=None, on_bad_lines="skip"
    )
    expected = DataFrame({"a": ["x", "z"], "b": [1, 3]})
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is requireds
def test_header_missing_rows(all_parsers):
    # GH#47400
    parser = all_parsers
    data = """a,b
1,2
"""
    msg = r"Passed header=\[0,1,2\], len of 3, but only 2 lines in file"
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), header=[0, 1, 2])


# ValueError: The 'delim_whitespace' option is not supported with the 'pyarrow' engine
@xfail_pyarrow
def test_header_multiple_whitespaces(all_parsers):
    # GH#54931
    parser = all_parsers
    data = """aa    bb(1,1)   cc(1,1)
                0  2  3.5"""

    result = parser.read_csv(StringIO(data), sep=r"\s+")
    expected = DataFrame({"aa": [0], "bb(1,1)": 2, "cc(1,1)": 3.5})
    tm.assert_frame_equal(result, expected)


# ValueError: The 'delim_whitespace' option is not supported with the 'pyarrow' engine
@xfail_pyarrow
def test_header_delim_whitespace(all_parsers):
    # GH#54918
    parser = all_parsers
    data = """a,b
1,2
3,4
    """

    depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"
    with tm.assert_produces_warning(
        FutureWarning, match=depr_msg, check_stacklevel=False
    ):
        result = parser.read_csv(StringIO(data), delim_whitespace=True)
    expected = DataFrame({"a,b": ["1,2", "3,4"]})
    tm.assert_frame_equal(result, expected)


def test_usecols_no_header_pyarrow(pyarrow_parser_only):
    parser = pyarrow_parser_only
    data = """
a,i,x
b,j,y
"""
    result = parser.read_csv(
        StringIO(data),
        header=None,
        usecols=[0, 1],
        dtype="string[pyarrow]",
        dtype_backend="pyarrow",
        engine="pyarrow",
    )
    expected = DataFrame([["a", "i"], ["b", "j"]], dtype="string[pyarrow]")
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\indexing\test_getitem.py ===
"""
Series.__getitem__ test classes are organized by the type of key passed.
"""
from datetime import (
    date,
    datetime,
    time,
)

import numpy as np
import pytest

from pandas._libs.tslibs import (
    conversion,
    timezones,
)

from pandas.core.dtypes.common import is_scalar

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    Series,
    Timestamp,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.indexing import IndexingError

from pandas.tseries.offsets import BDay


class TestSeriesGetitemScalars:
    def test_getitem_object_index_float_string(self):
        # GH#17286
        ser = Series([1] * 4, index=Index(["a", "b", "c", 1.0]))
        assert ser["a"] == 1
        assert ser[1.0] == 1

    def test_getitem_float_keys_tuple_values(self):
        # see GH#13509

        # unique Index
        ser = Series([(1, 1), (2, 2), (3, 3)], index=[0.0, 0.1, 0.2], name="foo")
        result = ser[0.0]
        assert result == (1, 1)

        # non-unique Index
        expected = Series([(1, 1), (2, 2)], index=[0.0, 0.0], name="foo")
        ser = Series([(1, 1), (2, 2), (3, 3)], index=[0.0, 0.0, 0.2], name="foo")

        result = ser[0.0]
        tm.assert_series_equal(result, expected)

    def test_getitem_unrecognized_scalar(self):
        # GH#32684 a scalar key that is not recognized by lib.is_scalar

        # a series that might be produced via `frame.dtypes`
        ser = Series([1, 2], index=[np.dtype("O"), np.dtype("i8")])

        key = ser.index[1]

        result = ser[key]
        assert result == 2

    def test_getitem_negative_out_of_bounds(self):
        ser = Series(["a"] * 10, index=["a"] * 10)

        msg = "index -11 is out of bounds for axis 0 with size 10|index out of bounds"
        warn_msg = "Series.__getitem__ treating keys as positions is deprecated"
        with pytest.raises(IndexError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=warn_msg):
                ser[-11]

    def test_getitem_out_of_bounds_indexerror(self, datetime_series):
        # don't segfault, GH#495
        msg = r"index \d+ is out of bounds for axis 0 with size \d+"
        warn_msg = "Series.__getitem__ treating keys as positions is deprecated"
        with pytest.raises(IndexError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=warn_msg):
                datetime_series[len(datetime_series)]

    def test_getitem_out_of_bounds_empty_rangeindex_keyerror(self):
        # GH#917
        # With a RangeIndex, an int key gives a KeyError
        ser = Series([], dtype=object)
        with pytest.raises(KeyError, match="-1"):
            ser[-1]

    def test_getitem_keyerror_with_integer_index(self, any_int_numpy_dtype):
        dtype = any_int_numpy_dtype
        ser = Series(
            np.random.default_rng(2).standard_normal(6),
            index=Index([0, 0, 1, 1, 2, 2], dtype=dtype),
        )

        with pytest.raises(KeyError, match=r"^5$"):
            ser[5]

        with pytest.raises(KeyError, match=r"^'c'$"):
            ser["c"]

        # not monotonic
        ser = Series(
            np.random.default_rng(2).standard_normal(6), index=[2, 2, 0, 0, 1, 1]
        )

        with pytest.raises(KeyError, match=r"^5$"):
            ser[5]

        with pytest.raises(KeyError, match=r"^'c'$"):
            ser["c"]

    def test_getitem_int64(self, datetime_series):
        idx = np.int64(5)
        msg = "Series.__getitem__ treating keys as positions is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = datetime_series[idx]
        assert res == datetime_series.iloc[5]

    def test_getitem_full_range(self):
        # github.com/pandas-dev/pandas/commit/4f433773141d2eb384325714a2776bcc5b2e20f7
        ser = Series(range(5), index=list(range(5)))
        result = ser[list(range(5))]
        tm.assert_series_equal(result, ser)

    # ------------------------------------------------------------------
    # Series with DatetimeIndex

    @pytest.mark.parametrize("tzstr", ["Europe/Berlin", "dateutil/Europe/Berlin"])
    def test_getitem_pydatetime_tz(self, tzstr):
        tz = timezones.maybe_get_tz(tzstr)

        index = date_range(
            start="2012-12-24 16:00", end="2012-12-24 18:00", freq="h", tz=tzstr
        )
        ts = Series(index=index, data=index.hour)
        time_pandas = Timestamp("2012-12-24 17:00", tz=tzstr)

        dt = datetime(2012, 12, 24, 17, 0)
        time_datetime = conversion.localize_pydatetime(dt, tz)
        assert ts[time_pandas] == ts[time_datetime]

    @pytest.mark.parametrize("tz", ["US/Eastern", "dateutil/US/Eastern"])
    def test_string_index_alias_tz_aware(self, tz):
        rng = date_range("1/1/2000", periods=10, tz=tz)
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

        result = ser["1/3/2000"]
        tm.assert_almost_equal(result, ser.iloc[2])

    def test_getitem_time_object(self):
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

        mask = (rng.hour == 9) & (rng.minute == 30)
        result = ts[time(9, 30)]
        expected = ts[mask]
        result.index = result.index._with_freq(None)
        tm.assert_series_equal(result, expected)

    # ------------------------------------------------------------------
    # Series with CategoricalIndex

    def test_getitem_scalar_categorical_index(self):
        cats = Categorical([Timestamp("12-31-1999"), Timestamp("12-31-2000")])

        ser = Series([1, 2], index=cats)

        expected = ser.iloc[0]
        result = ser[cats[0]]
        assert result == expected

    def test_getitem_numeric_categorical_listlike_matches_scalar(self):
        # GH#15470
        ser = Series(["a", "b", "c"], index=pd.CategoricalIndex([2, 1, 0]))

        # 0 is treated as a label
        assert ser[0] == "c"

        # the listlike analogue should also be treated as labels
        res = ser[[0]]
        expected = ser.iloc[-1:]
        tm.assert_series_equal(res, expected)

        res2 = ser[[0, 1, 2]]
        tm.assert_series_equal(res2, ser.iloc[::-1])

    def test_getitem_integer_categorical_not_positional(self):
        # GH#14865
        ser = Series(["a", "b", "c"], index=Index([1, 2, 3], dtype="category"))
        assert ser.get(3) == "c"
        assert ser[3] == "c"

    def test_getitem_str_with_timedeltaindex(self):
        rng = timedelta_range("1 day 10:11:12", freq="h", periods=500)
        ser = Series(np.arange(len(rng)), index=rng)

        key = "6 days, 23:11:12"
        indexer = rng.get_loc(key)
        assert indexer == 133

        result = ser[key]
        assert result == ser.iloc[133]

        msg = r"^Timedelta\('50 days 00:00:00'\)$"
        with pytest.raises(KeyError, match=msg):
            rng.get_loc("50 days")
        with pytest.raises(KeyError, match=msg):
            ser["50 days"]

    def test_getitem_bool_index_positional(self):
        # GH#48653
        ser = Series({True: 1, False: 0})
        msg = "Series.__getitem__ treating keys as positions is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = ser[0]
        assert result == 1


class TestSeriesGetitemSlices:
    def test_getitem_partial_str_slice_with_datetimeindex(self):
        # GH#34860
        arr = date_range("1/1/2008", "1/1/2009")
        ser = arr.to_series()
        result = ser["2008"]

        rng = date_range(start="2008-01-01", end="2008-12-31")
        expected = Series(rng, index=rng)

        tm.assert_series_equal(result, expected)

    def test_getitem_slice_strings_with_datetimeindex(self):
        idx = DatetimeIndex(
            ["1/1/2000", "1/2/2000", "1/2/2000", "1/3/2000", "1/4/2000"]
        )

        ts = Series(np.random.default_rng(2).standard_normal(len(idx)), index=idx)

        result = ts["1/2/2000":]
        expected = ts[1:]
        tm.assert_series_equal(result, expected)

        result = ts["1/2/2000":"1/3/2000"]
        expected = ts[1:4]
        tm.assert_series_equal(result, expected)

    def test_getitem_partial_str_slice_with_timedeltaindex(self):
        rng = timedelta_range("1 day 10:11:12", freq="h", periods=500)
        ser = Series(np.arange(len(rng)), index=rng)

        result = ser["5 day":"6 day"]
        expected = ser.iloc[86:134]
        tm.assert_series_equal(result, expected)

        result = ser["5 day":]
        expected = ser.iloc[86:]
        tm.assert_series_equal(result, expected)

        result = ser[:"6 day"]
        expected = ser.iloc[:134]
        tm.assert_series_equal(result, expected)

    def test_getitem_partial_str_slice_high_reso_with_timedeltaindex(self):
        # higher reso
        rng = timedelta_range("1 day 10:11:12", freq="us", periods=2000)
        ser = Series(np.arange(len(rng)), index=rng)

        result = ser["1 day 10:11:12":]
        expected = ser.iloc[0:]
        tm.assert_series_equal(result, expected)

        result = ser["1 day 10:11:12.001":]
        expected = ser.iloc[1000:]
        tm.assert_series_equal(result, expected)

        result = ser["1 days, 10:11:12.001001"]
        assert result == ser.iloc[1001]

    def test_getitem_slice_2d(self, datetime_series):
        # GH#30588 multi-dimensional indexing deprecated
        with pytest.raises(ValueError, match="Multi-dimensional indexing"):
            datetime_series[:, np.newaxis]

    def test_getitem_median_slice_bug(self):
        index = date_range("20090415", "20090519", freq="2B")
        ser = Series(np.random.default_rng(2).standard_normal(13), index=index)

        indexer = [slice(6, 7, None)]
        msg = "Indexing with a single-item list"
        with pytest.raises(ValueError, match=msg):
            # GH#31299
            ser[indexer]
        # but we're OK with a single-element tuple
        result = ser[(indexer[0],)]
        expected = ser[indexer[0]]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "slc, positions",
        [
            [slice(date(2018, 1, 1), None), [0, 1, 2]],
            [slice(date(2019, 1, 2), None), [2]],
            [slice(date(2020, 1, 1), None), []],
            [slice(None, date(2020, 1, 1)), [0, 1, 2]],
            [slice(None, date(2019, 1, 1)), [0]],
        ],
    )
    def test_getitem_slice_date(self, slc, positions):
        # https://github.com/pandas-dev/pandas/issues/31501
        ser = Series(
            [0, 1, 2],
            DatetimeIndex(["2019-01-01", "2019-01-01T06:00:00", "2019-01-02"]),
        )
        result = ser[slc]
        expected = ser.take(positions)
        tm.assert_series_equal(result, expected)

    def test_getitem_slice_float_raises(self, datetime_series):
        msg = (
            "cannot do slice indexing on DatetimeIndex with these indexers "
            r"\[{key}\] of type float"
        )
        with pytest.raises(TypeError, match=msg.format(key=r"4\.0")):
            datetime_series[4.0:10.0]

        with pytest.raises(TypeError, match=msg.format(key=r"4\.5")):
            datetime_series[4.5:10.0]

    def test_getitem_slice_bug(self):
        ser = Series(range(10), index=list(range(10)))
        result = ser[-12:]
        tm.assert_series_equal(result, ser)

        result = ser[-7:]
        tm.assert_series_equal(result, ser[3:])

        result = ser[:-12]
        tm.assert_series_equal(result, ser[:0])

    def test_getitem_slice_integers(self):
        ser = Series(
            np.random.default_rng(2).standard_normal(8),
            index=[2, 4, 6, 8, 10, 12, 14, 16],
        )

        result = ser[:4]
        expected = Series(ser.values[:4], index=[2, 4, 6, 8])
        tm.assert_series_equal(result, expected)


class TestSeriesGetitemListLike:
    @pytest.mark.parametrize("box", [list, np.array, Index, Series])
    def test_getitem_no_matches(self, box):
        # GH#33462 we expect the same behavior for list/ndarray/Index/Series
        ser = Series(["A", "B"])

        key = Series(["C"])
        key = box(key)

        msg = r"None of \[Index\(\['C'\], dtype='object|str'\)\] are in the \[index\]"
        with pytest.raises(KeyError, match=msg):
            ser[key]

    def test_getitem_intlist_intindex_periodvalues(self):
        ser = Series(period_range("2000-01-01", periods=10, freq="D"))

        result = ser[[2, 4]]
        exp = Series(
            [pd.Period("2000-01-03", freq="D"), pd.Period("2000-01-05", freq="D")],
            index=[2, 4],
            dtype="Period[D]",
        )
        tm.assert_series_equal(result, exp)
        assert result.dtype == "Period[D]"

    @pytest.mark.parametrize("box", [list, np.array, Index])
    def test_getitem_intlist_intervalindex_non_int(self, box):
        # GH#33404 fall back to positional since ints are unambiguous
        dti = date_range("2000-01-03", periods=3)._with_freq(None)
        ii = pd.IntervalIndex.from_breaks(dti)
        ser = Series(range(len(ii)), index=ii)

        expected = ser.iloc[:1]
        key = box([0])
        msg = "Series.__getitem__ treating keys as positions is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = ser[key]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("box", [list, np.array, Index])
    @pytest.mark.parametrize("dtype", [np.int64, np.float64, np.uint64])
    def test_getitem_intlist_multiindex_numeric_level(self, dtype, box):
        # GH#33404 do _not_ fall back to positional since ints are ambiguous
        idx = Index(range(4)).astype(dtype)
        dti = date_range("2000-01-03", periods=3)
        mi = pd.MultiIndex.from_product([idx, dti])
        ser = Series(range(len(mi))[::-1], index=mi)

        key = box([5])
        with pytest.raises(KeyError, match="5"):
            ser[key]

    def test_getitem_uint_array_key(self, any_unsigned_int_numpy_dtype):
        # GH #37218
        ser = Series([1, 2, 3])
        key = np.array([4], dtype=any_unsigned_int_numpy_dtype)

        with pytest.raises(KeyError, match="4"):
            ser[key]
        with pytest.raises(KeyError, match="4"):
            ser.loc[key]


class TestGetitemBooleanMask:
    def test_getitem_boolean(self, string_series):
        ser = string_series
        mask = ser > ser.median()

        # passing list is OK
        result = ser[list(mask)]
        expected = ser[mask]
        tm.assert_series_equal(result, expected)
        tm.assert_index_equal(result.index, ser.index[mask])

    def test_getitem_boolean_empty(self):
        ser = Series([], dtype=np.int64)
        ser.index.name = "index_name"
        ser = ser[ser.isna()]
        assert ser.index.name == "index_name"
        assert ser.dtype == np.int64

        # GH#5877
        # indexing with empty series
        ser = Series(["A", "B"], dtype=object)
        expected = Series(dtype=object, index=Index([], dtype="int64"))
        result = ser[Series([], dtype=object)]
        tm.assert_series_equal(result, expected)

        # invalid because of the boolean indexer
        # that's empty or not-aligned
        msg = (
            r"Unalignable boolean Series provided as indexer \(index of "
            r"the boolean Series and of the indexed object do not match"
        )
        with pytest.raises(IndexingError, match=msg):
            ser[Series([], dtype=bool)]

        with pytest.raises(IndexingError, match=msg):
            ser[Series([True], dtype=bool)]

    def test_getitem_boolean_object(self, string_series):
        # using column from DataFrame

        ser = string_series
        mask = ser > ser.median()
        omask = mask.astype(object)

        # getitem
        result = ser[omask]
        expected = ser[mask]
        tm.assert_series_equal(result, expected)

        # setitem
        s2 = ser.copy()
        cop = ser.copy()
        cop[omask] = 5
        s2[mask] = 5
        tm.assert_series_equal(cop, s2)

        # nans raise exception
        omask[5:10] = np.nan
        msg = "Cannot mask with non-boolean array containing NA / NaN values"
        with pytest.raises(ValueError, match=msg):
            ser[omask]
        with pytest.raises(ValueError, match=msg):
            ser[omask] = 5

    def test_getitem_boolean_dt64_copies(self):
        # GH#36210
        dti = date_range("2016-01-01", periods=4, tz="US/Pacific")
        key = np.array([True, True, False, False])

        ser = Series(dti._data)

        res = ser[key]
        assert res._values._ndarray.base is None

        # compare with numeric case for reference
        ser2 = Series(range(4))
        res2 = ser2[key]
        assert res2._values.base is None

    def test_getitem_boolean_corner(self, datetime_series):
        ts = datetime_series
        mask_shifted = ts.shift(1, freq=BDay()) > ts.median()

        msg = (
            r"Unalignable boolean Series provided as indexer \(index of "
            r"the boolean Series and of the indexed object do not match"
        )
        with pytest.raises(IndexingError, match=msg):
            ts[mask_shifted]

        with pytest.raises(IndexingError, match=msg):
            ts.loc[mask_shifted]

    def test_getitem_boolean_different_order(self, string_series):
        ordered = string_series.sort_values()

        sel = string_series[ordered > 0]
        exp = string_series[string_series > 0]
        tm.assert_series_equal(sel, exp)

    def test_getitem_boolean_contiguous_preserve_freq(self):
        rng = date_range("1/1/2000", "3/1/2000", freq="B")

        mask = np.zeros(len(rng), dtype=bool)
        mask[10:20] = True

        masked = rng[mask]
        expected = rng[10:20]
        assert expected.freq == rng.freq
        tm.assert_index_equal(masked, expected)

        mask[22] = True
        masked = rng[mask]
        assert masked.freq is None


class TestGetitemCallable:
    def test_getitem_callable(self):
        # GH#12533
        ser = Series(4, index=list("ABCD"))
        result = ser[lambda x: "A"]
        assert result == ser.loc["A"]

        result = ser[lambda x: ["A", "B"]]
        expected = ser.loc[["A", "B"]]
        tm.assert_series_equal(result, expected)

        result = ser[lambda x: [True, False, True, True]]
        expected = ser.iloc[[0, 2, 3]]
        tm.assert_series_equal(result, expected)


def test_getitem_generator(string_series):
    gen = (x > 0 for x in string_series)
    result = string_series[gen]
    result2 = string_series[iter(string_series > 0)]
    expected = string_series[string_series > 0]
    tm.assert_series_equal(result, expected)
    tm.assert_series_equal(result2, expected)


@pytest.mark.parametrize(
    "series",
    [
        Series([0, 1]),
        Series(date_range("2012-01-01", periods=2)),
        Series(date_range("2012-01-01", periods=2, tz="CET")),
    ],
)
def test_getitem_ndim_deprecated(series):
    with pytest.raises(ValueError, match="Multi-dimensional indexing"):
        series[:, None]


def test_getitem_multilevel_scalar_slice_not_implemented(
    multiindex_year_month_day_dataframe_random_data,
):
    # not implementing this for now
    df = multiindex_year_month_day_dataframe_random_data
    ser = df["A"]

    msg = r"\(2000, slice\(3, 4, None\)\)"
    with pytest.raises(TypeError, match=msg):
        ser[2000, 3:4]


def test_getitem_dataframe_raises():
    rng = list(range(10))
    ser = Series(10, index=rng)
    df = DataFrame(rng, index=rng)
    msg = (
        "Indexing a Series with DataFrame is not supported, "
        "use the appropriate DataFrame column"
    )
    with pytest.raises(TypeError, match=msg):
        ser[df > 5]


def test_getitem_assignment_series_alignment():
    # https://github.com/pandas-dev/pandas/issues/37427
    # with getitem, when assigning with a Series, it is not first aligned
    ser = Series(range(10))
    idx = np.array([2, 4, 9])
    ser[idx] = Series([10, 11, 12])
    expected = Series([0, 1, 10, 3, 11, 5, 6, 7, 8, 12])
    tm.assert_series_equal(ser, expected)


def test_getitem_duplicate_index_mistyped_key_raises_keyerror():
    # GH#29189 float_index.get_loc(None) should raise KeyError, not TypeError
    ser = Series([2, 5, 6, 8], index=[2.0, 4.0, 4.0, 5.0])
    with pytest.raises(KeyError, match="None"):
        ser[None]

    with pytest.raises(KeyError, match="None"):
        ser.index.get_loc(None)

    with pytest.raises(KeyError, match="None"):
        ser.index._engine.get_loc(None)


def test_getitem_1tuple_slice_without_multiindex():
    ser = Series(range(5))
    key = (slice(3),)

    result = ser[key]
    expected = ser[key[0]]
    tm.assert_series_equal(result, expected)


def test_getitem_preserve_name(datetime_series):
    result = datetime_series[datetime_series > 0]
    assert result.name == datetime_series.name

    msg = "Series.__getitem__ treating keys as positions is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = datetime_series[[0, 2, 4]]
    assert result.name == datetime_series.name

    result = datetime_series[5:10]
    assert result.name == datetime_series.name


def test_getitem_with_integer_labels():
    # integer indexes, be careful
    ser = Series(
        np.random.default_rng(2).standard_normal(10), index=list(range(0, 20, 2))
    )
    inds = [0, 2, 5, 7, 8]
    arr_inds = np.array([0, 2, 5, 7, 8])
    with pytest.raises(KeyError, match="not in index"):
        ser[inds]

    with pytest.raises(KeyError, match="not in index"):
        ser[arr_inds]


def test_getitem_missing(datetime_series):
    # missing
    d = datetime_series.index[0] - BDay()
    msg = r"Timestamp\('1999-12-31 00:00:00'\)"
    with pytest.raises(KeyError, match=msg):
        datetime_series[d]


def test_getitem_fancy(string_series, object_series):
    msg = "Series.__getitem__ treating keys as positions is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        slice1 = string_series[[1, 2, 3]]
        slice2 = object_series[[1, 2, 3]]
    assert string_series.index[2] == slice1.index[1]
    assert object_series.index[2] == slice2.index[1]
    assert string_series.iloc[2] == slice1.iloc[1]
    assert object_series.iloc[2] == slice2.iloc[1]


def test_getitem_box_float64(datetime_series):
    msg = "Series.__getitem__ treating keys as positions is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        value = datetime_series[5]
    assert isinstance(value, np.float64)


def test_getitem_unordered_dup():
    obj = Series(range(5), index=["c", "a", "a", "b", "b"])
    assert is_scalar(obj["c"])
    assert obj["c"] == 0


def test_getitem_dups():
    ser = Series(range(5), index=["A", "A", "B", "C", "C"], dtype=np.int64)
    expected = Series([3, 4], index=["C", "C"], dtype=np.int64)
    result = ser["C"]
    tm.assert_series_equal(result, expected)


def test_getitem_categorical_str():
    # GH#31765
    ser = Series(range(5), index=Categorical(["a", "b", "c", "a", "b"]))
    result = ser["a"]
    expected = ser.iloc[[0, 3]]
    tm.assert_series_equal(result, expected)


def test_slice_can_reorder_not_uniquely_indexed():
    ser = Series(1, index=["a", "a", "b", "b", "c"])
    ser[::-1]  # it works!


@pytest.mark.parametrize("index_vals", ["aabcd", "aadcb"])
def test_duplicated_index_getitem_positional_indexer(index_vals):
    # GH 11747
    s = Series(range(5), index=list(index_vals))

    msg = "Series.__getitem__ treating keys as positions is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = s[3]
    assert result == 3


class TestGetitemDeprecatedIndexers:
    @pytest.mark.parametrize("key", [{1}, {1: 1}])
    def test_getitem_dict_and_set_deprecated(self, key):
        # GH#42825 enforced in 2.0
        ser = Series([1, 2, 3])
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            ser[key]

    @pytest.mark.parametrize("key", [{1}, {1: 1}])
    def test_setitem_dict_and_set_disallowed(self, key):
        # GH#42825 enforced in 2.0
        ser = Series([1, 2, 3])
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            ser[key] = 1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_densebasic.py ===
"""Tests for dense recursive polynomials' basic tools. """

from sympy.polys.densebasic import (
    ninf,
    dup_LC, dmp_LC,
    dup_TC, dmp_TC,
    dmp_ground_LC, dmp_ground_TC,
    dmp_true_LT,
    dup_degree, dmp_degree,
    dmp_degree_in, dmp_degree_list,
    dup_strip, dmp_strip,
    dmp_validate,
    dup_reverse,
    dup_copy, dmp_copy,
    dup_normal, dmp_normal,
    dup_convert, dmp_convert,
    dup_from_sympy, dmp_from_sympy,
    dup_nth, dmp_nth, dmp_ground_nth,
    dmp_zero_p, dmp_zero,
    dmp_one_p, dmp_one,
    dmp_ground_p, dmp_ground,
    dmp_negative_p, dmp_positive_p,
    dmp_zeros, dmp_grounds,
    dup_from_dict, dup_from_raw_dict,
    dup_to_dict, dup_to_raw_dict,
    dmp_from_dict, dmp_to_dict,
    dmp_swap, dmp_permute,
    dmp_nest, dmp_raise,
    dup_deflate, dmp_deflate,
    dup_multi_deflate, dmp_multi_deflate,
    dup_inflate, dmp_inflate,
    dmp_exclude, dmp_include,
    dmp_inject, dmp_eject,
    dup_terms_gcd, dmp_terms_gcd,
    dmp_list_terms, dmp_apply_pairs,
    dup_slice,
    dup_random,
)

from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring

from sympy.core.singleton import S
from sympy.testing.pytest import raises

from sympy.core.numbers import oo

f_0, f_1, f_2, f_3, f_4, f_5, f_6 = [ f.to_dense() for f in f_polys() ]

def test_dup_LC():
    assert dup_LC([], ZZ) == 0
    assert dup_LC([2, 3, 4, 5], ZZ) == 2


def test_dup_TC():
    assert dup_TC([], ZZ) == 0
    assert dup_TC([2, 3, 4, 5], ZZ) == 5


def test_dmp_LC():
    assert dmp_LC([[]], ZZ) == []
    assert dmp_LC([[2, 3, 4], [5]], ZZ) == [2, 3, 4]
    assert dmp_LC([[[]]], ZZ) == [[]]
    assert dmp_LC([[[2], [3, 4]], [[5]]], ZZ) == [[2], [3, 4]]


def test_dmp_TC():
    assert dmp_TC([[]], ZZ) == []
    assert dmp_TC([[2, 3, 4], [5]], ZZ) == [5]
    assert dmp_TC([[[]]], ZZ) == [[]]
    assert dmp_TC([[[2], [3, 4]], [[5]]], ZZ) == [[5]]


def test_dmp_ground_LC():
    assert dmp_ground_LC([[]], 1, ZZ) == 0
    assert dmp_ground_LC([[2, 3, 4], [5]], 1, ZZ) == 2
    assert dmp_ground_LC([[[]]], 2, ZZ) == 0
    assert dmp_ground_LC([[[2], [3, 4]], [[5]]], 2, ZZ) == 2


def test_dmp_ground_TC():
    assert dmp_ground_TC([[]], 1, ZZ) == 0
    assert dmp_ground_TC([[2, 3, 4], [5]], 1, ZZ) == 5
    assert dmp_ground_TC([[[]]], 2, ZZ) == 0
    assert dmp_ground_TC([[[2], [3, 4]], [[5]]], 2, ZZ) == 5


def test_dmp_true_LT():
    assert dmp_true_LT([[]], 1, ZZ) == ((0, 0), 0)
    assert dmp_true_LT([[7]], 1, ZZ) == ((0, 0), 7)

    assert dmp_true_LT([[1, 0]], 1, ZZ) == ((0, 1), 1)
    assert dmp_true_LT([[1], []], 1, ZZ) == ((1, 0), 1)
    assert dmp_true_LT([[1, 0], []], 1, ZZ) == ((1, 1), 1)


def test_dup_degree():
    assert ninf == float('-inf')
    assert dup_degree([]) is ninf
    assert dup_degree([1]) == 0
    assert dup_degree([1, 0]) == 1
    assert dup_degree([1, 0, 0, 0, 1]) == 4


def test_dmp_degree():
    assert dmp_degree([[]], 1) is ninf
    assert dmp_degree([[[]]], 2) is ninf

    assert dmp_degree([[1]], 1) == 0
    assert dmp_degree([[2], [1]], 1) == 1


def test_dmp_degree_in():
    assert dmp_degree_in([[[]]], 0, 2) is ninf
    assert dmp_degree_in([[[]]], 1, 2) is ninf
    assert dmp_degree_in([[[]]], 2, 2) is ninf

    assert dmp_degree_in([[[1]]], 0, 2) == 0
    assert dmp_degree_in([[[1]]], 1, 2) == 0
    assert dmp_degree_in([[[1]]], 2, 2) == 0

    assert dmp_degree_in(f_4, 0, 2) == 9
    assert dmp_degree_in(f_4, 1, 2) == 12
    assert dmp_degree_in(f_4, 2, 2) == 8

    assert dmp_degree_in(f_6, 0, 2) == 4
    assert dmp_degree_in(f_6, 1, 2) == 4
    assert dmp_degree_in(f_6, 2, 2) == 6
    assert dmp_degree_in(f_6, 3, 3) == 3

    raises(IndexError, lambda: dmp_degree_in([[1]], -5, 1))


def test_dmp_degree_list():
    assert dmp_degree_list([[[[ ]]]], 3) == (-oo, -oo, -oo, -oo)
    assert dmp_degree_list([[[[1]]]], 3) == ( 0, 0, 0, 0)

    assert dmp_degree_list(f_0, 2) == (2, 2, 2)
    assert dmp_degree_list(f_1, 2) == (3, 3, 3)
    assert dmp_degree_list(f_2, 2) == (5, 3, 3)
    assert dmp_degree_list(f_3, 2) == (5, 4, 7)
    assert dmp_degree_list(f_4, 2) == (9, 12, 8)
    assert dmp_degree_list(f_5, 2) == (3, 3, 3)
    assert dmp_degree_list(f_6, 3) == (4, 4, 6, 3)


def test_dup_strip():
    assert dup_strip([]) == []
    assert dup_strip([0]) == []
    assert dup_strip([0, 0, 0]) == []

    assert dup_strip([1]) == [1]
    assert dup_strip([0, 1]) == [1]
    assert dup_strip([0, 0, 0, 1]) == [1]

    assert dup_strip([1, 2, 0]) == [1, 2, 0]
    assert dup_strip([0, 1, 2, 0]) == [1, 2, 0]
    assert dup_strip([0, 0, 0, 1, 2, 0]) == [1, 2, 0]


def test_dmp_strip():
    assert dmp_strip([0, 1, 0], 0) == [1, 0]

    assert dmp_strip([[]], 1) == [[]]
    assert dmp_strip([[], []], 1) == [[]]
    assert dmp_strip([[], [], []], 1) == [[]]

    assert dmp_strip([[[]]], 2) == [[[]]]
    assert dmp_strip([[[]], [[]]], 2) == [[[]]]
    assert dmp_strip([[[]], [[]], [[]]], 2) == [[[]]]

    assert dmp_strip([[[1]]], 2) == [[[1]]]
    assert dmp_strip([[[]], [[1]]], 2) == [[[1]]]
    assert dmp_strip([[[]], [[1]], [[]]], 2) == [[[1]], [[]]]


def test_dmp_validate():
    assert dmp_validate([]) == ([], 0)
    assert dmp_validate([0, 0, 0, 1, 0]) == ([1, 0], 0)

    assert dmp_validate([[[]]]) == ([[[]]], 2)
    assert dmp_validate([[0], [], [0], [1], [0]]) == ([[1], []], 1)

    raises(ValueError, lambda: dmp_validate([[0], 0, [0], [1], [0]]))


def test_dup_reverse():
    assert dup_reverse([1, 2, 0, 3]) == [3, 0, 2, 1]
    assert dup_reverse([1, 2, 3, 0]) == [3, 2, 1]


def test_dup_copy():
    f = [ZZ(1), ZZ(0), ZZ(2)]
    g = dup_copy(f)

    g[0], g[2] = ZZ(7), ZZ(0)

    assert f != g


def test_dmp_copy():
    f = [[ZZ(1)], [ZZ(2), ZZ(0)]]
    g = dmp_copy(f, 1)

    g[0][0], g[1][1] = ZZ(7), ZZ(1)

    assert f != g


def test_dup_normal():
    assert dup_normal([0, 0, 2, 1, 0, 11, 0], ZZ) == \
        [ZZ(2), ZZ(1), ZZ(0), ZZ(11), ZZ(0)]


def test_dmp_normal():
    assert dmp_normal([[0], [], [0, 2, 1], [0], [11], []], 1, ZZ) == \
        [[ZZ(2), ZZ(1)], [], [ZZ(11)], []]


def test_dup_convert():
    K0, K1 = ZZ['x'], ZZ

    f = [K0(1), K0(2), K0(0), K0(3)]

    assert dup_convert(f, K0, K1) == \
        [ZZ(1), ZZ(2), ZZ(0), ZZ(3)]


def test_dmp_convert():
    K0, K1 = ZZ['x'], ZZ

    f = [[K0(1)], [K0(2)], [], [K0(3)]]

    assert dmp_convert(f, 1, K0, K1) == \
        [[ZZ(1)], [ZZ(2)], [], [ZZ(3)]]


def test_dup_from_sympy():
    assert dup_from_sympy([S.One, S(2)], ZZ) == \
        [ZZ(1), ZZ(2)]
    assert dup_from_sympy([S.Half, S(3)], QQ) == \
        [QQ(1, 2), QQ(3, 1)]


def test_dmp_from_sympy():
    assert dmp_from_sympy([[S.One, S(2)], [S.Zero]], 1, ZZ) == \
        [[ZZ(1), ZZ(2)], []]
    assert dmp_from_sympy([[S.Half, S(2)]], 1, QQ) == \
        [[QQ(1, 2), QQ(2, 1)]]


def test_dup_nth():
    assert dup_nth([1, 2, 3], 0, ZZ) == 3
    assert dup_nth([1, 2, 3], 1, ZZ) == 2
    assert dup_nth([1, 2, 3], 2, ZZ) == 1

    assert dup_nth([1, 2, 3], 9, ZZ) == 0

    raises(IndexError, lambda: dup_nth([3, 4, 5], -1, ZZ))


def test_dmp_nth():
    assert dmp_nth([[1], [2], [3]], 0, 1, ZZ) == [3]
    assert dmp_nth([[1], [2], [3]], 1, 1, ZZ) == [2]
    assert dmp_nth([[1], [2], [3]], 2, 1, ZZ) == [1]

    assert dmp_nth([[1], [2], [3]], 9, 1, ZZ) == []

    raises(IndexError, lambda: dmp_nth([[3], [4], [5]], -1, 1, ZZ))


def test_dmp_ground_nth():
    assert dmp_ground_nth([[]], (0, 0), 1, ZZ) == 0
    assert dmp_ground_nth([[1], [2], [3]], (0, 0), 1, ZZ) == 3
    assert dmp_ground_nth([[1], [2], [3]], (1, 0), 1, ZZ) == 2
    assert dmp_ground_nth([[1], [2], [3]], (2, 0), 1, ZZ) == 1

    assert dmp_ground_nth([[1], [2], [3]], (2, 1), 1, ZZ) == 0
    assert dmp_ground_nth([[1], [2], [3]], (3, 0), 1, ZZ) == 0

    raises(IndexError, lambda: dmp_ground_nth([[3], [4], [5]], (2, -1), 1, ZZ))


def test_dmp_zero_p():
    assert dmp_zero_p([], 0) is True
    assert dmp_zero_p([[]], 1) is True

    assert dmp_zero_p([[[]]], 2) is True
    assert dmp_zero_p([[[1]]], 2) is False


def test_dmp_zero():
    assert dmp_zero(0) == []
    assert dmp_zero(2) == [[[]]]


def test_dmp_one_p():
    assert dmp_one_p([1], 0, ZZ) is True
    assert dmp_one_p([[1]], 1, ZZ) is True
    assert dmp_one_p([[[1]]], 2, ZZ) is True
    assert dmp_one_p([[[12]]], 2, ZZ) is False


def test_dmp_one():
    assert dmp_one(0, ZZ) == [ZZ(1)]
    assert dmp_one(2, ZZ) == [[[ZZ(1)]]]


def test_dmp_ground_p():
    assert dmp_ground_p([], 0, 0) is True
    assert dmp_ground_p([[]], 0, 1) is True
    assert dmp_ground_p([[]], 1, 1) is False

    assert dmp_ground_p([[ZZ(1)]], 1, 1) is True
    assert dmp_ground_p([[[ZZ(2)]]], 2, 2) is True

    assert dmp_ground_p([[[ZZ(2)]]], 3, 2) is False
    assert dmp_ground_p([[[ZZ(3)], []]], 3, 2) is False

    assert dmp_ground_p([], None, 0) is True
    assert dmp_ground_p([[]], None, 1) is True

    assert dmp_ground_p([ZZ(1)], None, 0) is True
    assert dmp_ground_p([[[ZZ(1)]]], None, 2) is True

    assert dmp_ground_p([[[ZZ(3)], []]], None, 2) is False


def test_dmp_ground():
    assert dmp_ground(ZZ(0), 2) == [[[]]]

    assert dmp_ground(ZZ(7), -1) == ZZ(7)
    assert dmp_ground(ZZ(7), 0) == [ZZ(7)]
    assert dmp_ground(ZZ(7), 2) == [[[ZZ(7)]]]


def test_dmp_zeros():
    assert dmp_zeros(4, 0, ZZ) == [[], [], [], []]

    assert dmp_zeros(0, 2, ZZ) == []
    assert dmp_zeros(1, 2, ZZ) == [[[[]]]]
    assert dmp_zeros(2, 2, ZZ) == [[[[]]], [[[]]]]
    assert dmp_zeros(3, 2, ZZ) == [[[[]]], [[[]]], [[[]]]]

    assert dmp_zeros(3, -1, ZZ) == [0, 0, 0]


def test_dmp_grounds():
    assert dmp_grounds(ZZ(7), 0, 2) == []

    assert dmp_grounds(ZZ(7), 1, 2) == [[[[7]]]]
    assert dmp_grounds(ZZ(7), 2, 2) == [[[[7]]], [[[7]]]]
    assert dmp_grounds(ZZ(7), 3, 2) == [[[[7]]], [[[7]]], [[[7]]]]

    assert dmp_grounds(ZZ(7), 3, -1) == [7, 7, 7]


def test_dmp_negative_p():
    assert dmp_negative_p([[[]]], 2, ZZ) is False
    assert dmp_negative_p([[[1], [2]]], 2, ZZ) is False
    assert dmp_negative_p([[[-1], [2]]], 2, ZZ) is True


def test_dmp_positive_p():
    assert dmp_positive_p([[[]]], 2, ZZ) is False
    assert dmp_positive_p([[[1], [2]]], 2, ZZ) is True
    assert dmp_positive_p([[[-1], [2]]], 2, ZZ) is False


def test_dup_from_to_dict():
    assert dup_from_raw_dict({}, ZZ) == []
    assert dup_from_dict({}, ZZ) == []

    assert dup_to_raw_dict([]) == {}
    assert dup_to_dict([]) == {}

    assert dup_to_raw_dict([], ZZ, zero=True) == {0: ZZ(0)}
    assert dup_to_dict([], ZZ, zero=True) == {(0,): ZZ(0)}

    f = [3, 0, 0, 2, 0, 0, 0, 0, 8]
    g = {8: 3, 5: 2, 0: 8}
    h = {(8,): 3, (5,): 2, (0,): 8}

    assert dup_from_raw_dict(g, ZZ) == f
    assert dup_from_dict(h, ZZ) == f

    assert dup_to_raw_dict(f) == g
    assert dup_to_dict(f) == h

    R, x,y = ring("x,y", ZZ)
    K = R.to_domain()

    f = [R(3), R(0), R(2), R(0), R(0), R(8)]
    g = {5: R(3), 3: R(2), 0: R(8)}
    h = {(5,): R(3), (3,): R(2), (0,): R(8)}

    assert dup_from_raw_dict(g, K) == f
    assert dup_from_dict(h, K) == f

    assert dup_to_raw_dict(f) == g
    assert dup_to_dict(f) == h


def test_dmp_from_to_dict():
    assert dmp_from_dict({}, 1, ZZ) == [[]]
    assert dmp_to_dict([[]], 1) == {}

    assert dmp_to_dict([], 0, ZZ, zero=True) == {(0,): ZZ(0)}
    assert dmp_to_dict([[]], 1, ZZ, zero=True) == {(0, 0): ZZ(0)}

    f = [[3], [], [], [2], [], [], [], [], [8]]
    g = {(8, 0): 3, (5, 0): 2, (0, 0): 8}

    assert dmp_from_dict(g, 1, ZZ) == f
    assert dmp_to_dict(f, 1) == g


def test_dmp_swap():
    f = dmp_normal([[1, 0, 0], [], [1, 0], [], [1]], 1, ZZ)
    g = dmp_normal([[1, 0, 0, 0, 0], [1, 0, 0], [1]], 1, ZZ)

    assert dmp_swap(f, 1, 1, 1, ZZ) == f

    assert dmp_swap(f, 0, 1, 1, ZZ) == g
    assert dmp_swap(g, 0, 1, 1, ZZ) == f

    raises(IndexError, lambda: dmp_swap(f, -1, -7, 1, ZZ))


def test_dmp_permute():
    f = dmp_normal([[1, 0, 0], [], [1, 0], [], [1]], 1, ZZ)
    g = dmp_normal([[1, 0, 0, 0, 0], [1, 0, 0], [1]], 1, ZZ)

    assert dmp_permute(f, [0, 1], 1, ZZ) == f
    assert dmp_permute(g, [0, 1], 1, ZZ) == g

    assert dmp_permute(f, [1, 0], 1, ZZ) == g
    assert dmp_permute(g, [1, 0], 1, ZZ) == f


def test_dmp_nest():
    assert dmp_nest(ZZ(1), 2, ZZ) == [[[1]]]

    assert dmp_nest([[1]], 0, ZZ) == [[1]]
    assert dmp_nest([[1]], 1, ZZ) == [[[1]]]
    assert dmp_nest([[1]], 2, ZZ) == [[[[1]]]]


def test_dmp_raise():
    assert dmp_raise([], 2, 0, ZZ) == [[[]]]
    assert dmp_raise([[1]], 0, 1, ZZ) == [[1]]

    assert dmp_raise([[1, 2, 3], [], [2, 3]], 2, 1, ZZ) == \
        [[[[1]], [[2]], [[3]]], [[[]]], [[[2]], [[3]]]]


def test_dup_deflate():
    assert dup_deflate([], ZZ) == (1, [])
    assert dup_deflate([2], ZZ) == (1, [2])
    assert dup_deflate([1, 2, 3], ZZ) == (1, [1, 2, 3])
    assert dup_deflate([1, 0, 2, 0, 3], ZZ) == (2, [1, 2, 3])

    assert dup_deflate(dup_from_raw_dict({7: 1, 1: 1}, ZZ), ZZ) == \
        (1, [1, 0, 0, 0, 0, 0, 1, 0])
    assert dup_deflate(dup_from_raw_dict({7: 1, 0: 1}, ZZ), ZZ) == \
        (7, [1, 1])
    assert dup_deflate(dup_from_raw_dict({7: 1, 3: 1}, ZZ), ZZ) == \
        (1, [1, 0, 0, 0, 1, 0, 0, 0])

    assert dup_deflate(dup_from_raw_dict({7: 1, 4: 1}, ZZ), ZZ) == \
        (1, [1, 0, 0, 1, 0, 0, 0, 0])
    assert dup_deflate(dup_from_raw_dict({8: 1, 4: 1}, ZZ), ZZ) == \
        (4, [1, 1, 0])

    assert dup_deflate(dup_from_raw_dict({8: 1}, ZZ), ZZ) == \
        (8, [1, 0])
    assert dup_deflate(dup_from_raw_dict({7: 1}, ZZ), ZZ) == \
        (7, [1, 0])
    assert dup_deflate(dup_from_raw_dict({1: 1}, ZZ), ZZ) == \
        (1, [1, 0])


def test_dmp_deflate():
    assert dmp_deflate([[]], 1, ZZ) == ((1, 1), [[]])
    assert dmp_deflate([[2]], 1, ZZ) == ((1, 1), [[2]])

    f = [[1, 0, 0], [], [1, 0], [], [1]]

    assert dmp_deflate(f, 1, ZZ) == ((2, 1), [[1, 0, 0], [1, 0], [1]])


def test_dup_multi_deflate():
    assert dup_multi_deflate(([2],), ZZ) == (1, ([2],))
    assert dup_multi_deflate(([], []), ZZ) == (1, ([], []))

    assert dup_multi_deflate(([1, 2, 3],), ZZ) == (1, ([1, 2, 3],))
    assert dup_multi_deflate(([1, 0, 2, 0, 3],), ZZ) == (2, ([1, 2, 3],))

    assert dup_multi_deflate(([1, 0, 2, 0, 3], [2, 0, 0]), ZZ) == \
        (2, ([1, 2, 3], [2, 0]))
    assert dup_multi_deflate(([1, 0, 2, 0, 3], [2, 1, 0]), ZZ) == \
        (1, ([1, 0, 2, 0, 3], [2, 1, 0]))


def test_dmp_multi_deflate():
    assert dmp_multi_deflate(([[]],), 1, ZZ) == \
        ((1, 1), ([[]],))
    assert dmp_multi_deflate(([[]], [[]]), 1, ZZ) == \
        ((1, 1), ([[]], [[]]))

    assert dmp_multi_deflate(([[1]], [[]]), 1, ZZ) == \
        ((1, 1), ([[1]], [[]]))
    assert dmp_multi_deflate(([[1]], [[2]]), 1, ZZ) == \
        ((1, 1), ([[1]], [[2]]))
    assert dmp_multi_deflate(([[1]], [[2, 0]]), 1, ZZ) == \
        ((1, 1), ([[1]], [[2, 0]]))

    assert dmp_multi_deflate(([[2, 0]], [[2, 0]]), 1, ZZ) == \
        ((1, 1), ([[2, 0]], [[2, 0]]))

    assert dmp_multi_deflate(
        ([[2]], [[2, 0, 0]]), 1, ZZ) == ((1, 2), ([[2]], [[2, 0]]))
    assert dmp_multi_deflate(
        ([[2, 0, 0]], [[2, 0, 0]]), 1, ZZ) == ((1, 2), ([[2, 0]], [[2, 0]]))

    assert dmp_multi_deflate(([2, 0, 0], [1, 0, 4, 0, 1]), 0, ZZ) == \
        ((2,), ([2, 0], [1, 4, 1]))

    f = [[1, 0, 0], [], [1, 0], [], [1]]
    g = [[1, 0, 1, 0], [], [1]]

    assert dmp_multi_deflate((f,), 1, ZZ) == \
        ((2, 1), ([[1, 0, 0], [1, 0], [1]],))

    assert dmp_multi_deflate((f, g), 1, ZZ) == \
        ((2, 1), ([[1, 0, 0], [1, 0], [1]],
                  [[1, 0, 1, 0], [1]]))


def test_dup_inflate():
    assert dup_inflate([], 17, ZZ) == []

    assert dup_inflate([1, 2, 3], 1, ZZ) == [1, 2, 3]
    assert dup_inflate([1, 2, 3], 2, ZZ) == [1, 0, 2, 0, 3]
    assert dup_inflate([1, 2, 3], 3, ZZ) == [1, 0, 0, 2, 0, 0, 3]
    assert dup_inflate([1, 2, 3], 4, ZZ) == [1, 0, 0, 0, 2, 0, 0, 0, 3]

    raises(IndexError, lambda: dup_inflate([1, 2, 3], 0, ZZ))


def test_dmp_inflate():
    assert dmp_inflate([1], (3,), 0, ZZ) == [1]

    assert dmp_inflate([[]], (3, 7), 1, ZZ) == [[]]
    assert dmp_inflate([[2]], (1, 2), 1, ZZ) == [[2]]

    assert dmp_inflate([[2, 0]], (1, 1), 1, ZZ) == [[2, 0]]
    assert dmp_inflate([[2, 0]], (1, 2), 1, ZZ) == [[2, 0, 0]]
    assert dmp_inflate([[2, 0]], (1, 3), 1, ZZ) == [[2, 0, 0, 0]]

    assert dmp_inflate([[1, 0, 0], [1], [1, 0]], (2, 1), 1, ZZ) == \
        [[1, 0, 0], [], [1], [], [1, 0]]

    raises(IndexError, lambda: dmp_inflate([[]], (-3, 7), 1, ZZ))


def test_dmp_exclude():
    assert dmp_exclude([[[]]], 2, ZZ) == ([], [[[]]], 2)
    assert dmp_exclude([[[7]]], 2, ZZ) == ([], [[[7]]], 2)

    assert dmp_exclude([1, 2, 3], 0, ZZ) == ([], [1, 2, 3], 0)
    assert dmp_exclude([[1], [2, 3]], 1, ZZ) == ([], [[1], [2, 3]], 1)

    assert dmp_exclude([[1, 2, 3]], 1, ZZ) == ([0], [1, 2, 3], 0)
    assert dmp_exclude([[1], [2], [3]], 1, ZZ) == ([1], [1, 2, 3], 0)

    assert dmp_exclude([[[1, 2, 3]]], 2, ZZ) == ([0, 1], [1, 2, 3], 0)
    assert dmp_exclude([[[1]], [[2]], [[3]]], 2, ZZ) == ([1, 2], [1, 2, 3], 0)


def test_dmp_include():
    assert dmp_include([1, 2, 3], [], 0, ZZ) == [1, 2, 3]

    assert dmp_include([1, 2, 3], [0], 0, ZZ) == [[1, 2, 3]]
    assert dmp_include([1, 2, 3], [1], 0, ZZ) == [[1], [2], [3]]

    assert dmp_include([1, 2, 3], [0, 1], 0, ZZ) == [[[1, 2, 3]]]
    assert dmp_include([1, 2, 3], [1, 2], 0, ZZ) == [[[1]], [[2]], [[3]]]


def test_dmp_inject():
    R, x,y = ring("x,y", ZZ)
    K = R.to_domain()

    assert dmp_inject([], 0, K) == ([[[]]], 2)
    assert dmp_inject([[]], 1, K) == ([[[[]]]], 3)

    assert dmp_inject([R(1)], 0, K) == ([[[1]]], 2)
    assert dmp_inject([[R(1)]], 1, K) == ([[[[1]]]], 3)

    assert dmp_inject([R(1), 2*x + 3*y + 4], 0, K) == ([[[1]], [[2], [3, 4]]], 2)

    f = [3*x**2 + 7*x*y + 5*y**2, 2*x, R(0), x*y**2 + 11]
    g = [[[3], [7, 0], [5, 0, 0]], [[2], []], [[]], [[1, 0, 0], [11]]]

    assert dmp_inject(f, 0, K) == (g, 2)


def test_dmp_eject():
    R, x,y = ring("x,y", ZZ)
    K = R.to_domain()

    assert dmp_eject([[[]]], 2, K) == []
    assert dmp_eject([[[[]]]], 3, K) == [[]]

    assert dmp_eject([[[1]]], 2, K) == [R(1)]
    assert dmp_eject([[[[1]]]], 3, K) == [[R(1)]]

    assert dmp_eject([[[1]], [[2], [3, 4]]], 2, K) == [R(1), 2*x + 3*y + 4]

    f = [3*x**2 + 7*x*y + 5*y**2, 2*x, R(0), x*y**2 + 11]
    g = [[[3], [7, 0], [5, 0, 0]], [[2], []], [[]], [[1, 0, 0], [11]]]

    assert dmp_eject(g, 2, K) == f


def test_dup_terms_gcd():
    assert dup_terms_gcd([], ZZ) == (0, [])
    assert dup_terms_gcd([1, 0, 1], ZZ) == (0, [1, 0, 1])
    assert dup_terms_gcd([1, 0, 1, 0], ZZ) == (1, [1, 0, 1])


def test_dmp_terms_gcd():
    assert dmp_terms_gcd([[]], 1, ZZ) == ((0, 0), [[]])

    assert dmp_terms_gcd([1, 0, 1, 0], 0, ZZ) == ((1,), [1, 0, 1])
    assert dmp_terms_gcd([[1], [], [1], []], 1, ZZ) == ((1, 0), [[1], [], [1]])

    assert dmp_terms_gcd(
        [[1, 0], [], [1]], 1, ZZ) == ((0, 0), [[1, 0], [], [1]])
    assert dmp_terms_gcd(
        [[1, 0], [1, 0, 0], [], []], 1, ZZ) == ((2, 1), [[1], [1, 0]])


def test_dmp_list_terms():
    assert dmp_list_terms([[[]]], 2, ZZ) == [((0, 0, 0), 0)]
    assert dmp_list_terms([[[1]]], 2, ZZ) == [((0, 0, 0), 1)]

    assert dmp_list_terms([1, 2, 4, 3, 5], 0, ZZ) == \
        [((4,), 1), ((3,), 2), ((2,), 4), ((1,), 3), ((0,), 5)]

    assert dmp_list_terms([[1], [2, 4], [3, 5, 0]], 1, ZZ) == \
        [((2, 0), 1), ((1, 1), 2), ((1, 0), 4), ((0, 2), 3), ((0, 1), 5)]

    f = [[2, 0, 0, 0], [1, 0, 0], []]

    assert dmp_list_terms(f, 1, ZZ, order='lex') == [((2, 3), 2), ((1, 2), 1)]
    assert dmp_list_terms(
        f, 1, ZZ, order='grlex') == [((2, 3), 2), ((1, 2), 1)]

    f = [[2, 0, 0, 0], [1, 0, 0, 0, 0, 0], []]

    assert dmp_list_terms(f, 1, ZZ, order='lex') == [((2, 3), 2), ((1, 5), 1)]
    assert dmp_list_terms(
        f, 1, ZZ, order='grlex') == [((1, 5), 1), ((2, 3), 2)]


def test_dmp_apply_pairs():
    h = lambda a, b: a*b

    assert dmp_apply_pairs([1, 2, 3], [4, 5, 6], h, [], 0, ZZ) == [4, 10, 18]

    assert dmp_apply_pairs([2, 3], [4, 5, 6], h, [], 0, ZZ) == [10, 18]
    assert dmp_apply_pairs([1, 2, 3], [5, 6], h, [], 0, ZZ) == [10, 18]

    assert dmp_apply_pairs(
        [[1, 2], [3]], [[4, 5], [6]], h, [], 1, ZZ) == [[4, 10], [18]]

    assert dmp_apply_pairs(
        [[1, 2], [3]], [[4], [5, 6]], h, [], 1, ZZ) == [[8], [18]]
    assert dmp_apply_pairs(
        [[1], [2, 3]], [[4, 5], [6]], h, [], 1, ZZ) == [[5], [18]]


def test_dup_slice():
    f = [1, 2, 3, 4]

    assert dup_slice(f, 0, 0, ZZ) == []
    assert dup_slice(f, 0, 1, ZZ) == [4]
    assert dup_slice(f, 0, 2, ZZ) == [3, 4]
    assert dup_slice(f, 0, 3, ZZ) == [2, 3, 4]
    assert dup_slice(f, 0, 4, ZZ) == [1, 2, 3, 4]

    assert dup_slice(f, 0, 4, ZZ) == f
    assert dup_slice(f, 0, 9, ZZ) == f

    assert dup_slice(f, 1, 0, ZZ) == []
    assert dup_slice(f, 1, 1, ZZ) == []
    assert dup_slice(f, 1, 2, ZZ) == [3, 0]
    assert dup_slice(f, 1, 3, ZZ) == [2, 3, 0]
    assert dup_slice(f, 1, 4, ZZ) == [1, 2, 3, 0]

    assert dup_slice([1, 2], 0, 3, ZZ) == [1, 2]

    g = [1, 0, 0, 2]

    assert dup_slice(g, 0, 3, ZZ) == [2]


def test_dup_random():
    f = dup_random(0, -10, 10, ZZ)

    assert dup_degree(f) == 0
    assert all(-10 <= c <= 10 for c in f)

    f = dup_random(1, -20, 20, ZZ)

    assert dup_degree(f) == 1
    assert all(-20 <= c <= 20 for c in f)

    f = dup_random(2, -30, 30, ZZ)

    assert dup_degree(f) == 2
    assert all(-30 <= c <= 30 for c in f)

    f = dup_random(3, -40, 40, ZZ)

    assert dup_degree(f) == 3
    assert all(-40 <= c <= 40 for c in f)