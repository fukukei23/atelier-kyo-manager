
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\aggregate\test_aggregate.py ===
"""
test .agg behavior / note that .apply is tested generally in test_groupby.py
"""
import datetime
import functools
from functools import partial
import re

import numpy as np
import pytest

from pandas.errors import SpecificationError

from pandas.core.dtypes.common import is_integer_dtype

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    concat,
    to_datetime,
)
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping


def test_groupby_agg_no_extra_calls():
    # GH#31760
    df = DataFrame({"key": ["a", "b", "c", "c"], "value": [1, 2, 3, 4]})
    gb = df.groupby("key")["value"]

    def dummy_func(x):
        assert len(x) != 0
        return x.sum()

    gb.agg(dummy_func)


def test_agg_regression1(tsframe):
    grouped = tsframe.groupby([lambda x: x.year, lambda x: x.month])
    result = grouped.agg("mean")
    expected = grouped.mean()
    tm.assert_frame_equal(result, expected)


def test_agg_must_agg(df):
    grouped = df.groupby("A")["C"]

    msg = "Must produce aggregated value"
    with pytest.raises(Exception, match=msg):
        grouped.agg(lambda x: x.describe())
    with pytest.raises(Exception, match=msg):
        grouped.agg(lambda x: x.index[:2])


def test_agg_ser_multi_key(df):
    f = lambda x: x.sum()
    results = df.C.groupby([df.A, df.B]).aggregate(f)
    expected = df.groupby(["A", "B"]).sum()["C"]
    tm.assert_series_equal(results, expected)


def test_groupby_aggregation_mixed_dtype():
    # GH 6212
    expected = DataFrame(
        {
            "v1": [5, 5, 7, np.nan, 3, 3, 4, 1],
            "v2": [55, 55, 77, np.nan, 33, 33, 44, 11],
        },
        index=MultiIndex.from_tuples(
            [
                (1, 95),
                (1, 99),
                (2, 95),
                (2, 99),
                ("big", "damp"),
                ("blue", "dry"),
                ("red", "red"),
                ("red", "wet"),
            ],
            names=["by1", "by2"],
        ),
    )

    df = DataFrame(
        {
            "v1": [1, 3, 5, 7, 8, 3, 5, np.nan, 4, 5, 7, 9],
            "v2": [11, 33, 55, 77, 88, 33, 55, np.nan, 44, 55, 77, 99],
            "by1": ["red", "blue", 1, 2, np.nan, "big", 1, 2, "red", 1, np.nan, 12],
            "by2": [
                "wet",
                "dry",
                99,
                95,
                np.nan,
                "damp",
                95,
                99,
                "red",
                99,
                np.nan,
                np.nan,
            ],
        }
    )

    g = df.groupby(["by1", "by2"])
    result = g[["v1", "v2"]].mean()
    tm.assert_frame_equal(result, expected)


def test_groupby_aggregation_multi_level_column():
    # GH 29772
    lst = [
        [True, True, True, False],
        [True, False, np.nan, False],
        [True, True, np.nan, False],
        [True, True, np.nan, False],
    ]
    df = DataFrame(
        data=lst,
        columns=MultiIndex.from_tuples([("A", 0), ("A", 1), ("B", 0), ("B", 1)]),
    )

    msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby(level=1, axis=1)
    result = gb.sum(numeric_only=False)
    expected = DataFrame({0: [2.0, True, True, True], 1: [1, 0, 1, 1]})

    tm.assert_frame_equal(result, expected)


def test_agg_apply_corner(ts, tsframe):
    # nothing to group, all NA
    grouped = ts.groupby(ts * np.nan, group_keys=False)
    assert ts.dtype == np.float64

    # groupby float64 values results in a float64 Index
    exp = Series([], dtype=np.float64, index=Index([], dtype=np.float64))
    tm.assert_series_equal(grouped.sum(), exp)
    tm.assert_series_equal(grouped.agg("sum"), exp)
    tm.assert_series_equal(grouped.apply("sum"), exp, check_index_type=False)

    # DataFrame
    grouped = tsframe.groupby(tsframe["A"] * np.nan, group_keys=False)
    exp_df = DataFrame(
        columns=tsframe.columns,
        dtype=float,
        index=Index([], name="A", dtype=np.float64),
    )
    tm.assert_frame_equal(grouped.sum(), exp_df)
    tm.assert_frame_equal(grouped.agg("sum"), exp_df)

    msg = "The behavior of DataFrame.sum with axis=None is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg, check_stacklevel=False):
        res = grouped.apply(np.sum)
    tm.assert_frame_equal(res, exp_df)


def test_agg_grouping_is_list_tuple(ts):
    df = DataFrame(
        np.random.default_rng(2).standard_normal((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=pd.date_range("2000-01-01", periods=30, freq="B"),
    )

    grouped = df.groupby(lambda x: x.year)
    grouper = grouped._grouper.groupings[0].grouping_vector
    grouped._grouper.groupings[0] = Grouping(ts.index, list(grouper))

    result = grouped.agg("mean")
    expected = grouped.mean()
    tm.assert_frame_equal(result, expected)

    grouped._grouper.groupings[0] = Grouping(ts.index, tuple(grouper))

    result = grouped.agg("mean")
    expected = grouped.mean()
    tm.assert_frame_equal(result, expected)


def test_agg_python_multiindex(multiindex_dataframe_random_data):
    grouped = multiindex_dataframe_random_data.groupby(["A", "B"])

    result = grouped.agg("mean")
    expected = grouped.mean()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "groupbyfunc", [lambda x: x.weekday(), [lambda x: x.month, lambda x: x.weekday()]]
)
def test_aggregate_str_func(tsframe, groupbyfunc):
    grouped = tsframe.groupby(groupbyfunc)

    # single series
    result = grouped["A"].agg("std")
    expected = grouped["A"].std()
    tm.assert_series_equal(result, expected)

    # group frame by function name
    result = grouped.aggregate("var")
    expected = grouped.var()
    tm.assert_frame_equal(result, expected)

    # group frame by function dict
    result = grouped.agg({"A": "var", "B": "std", "C": "mean", "D": "sem"})
    expected = DataFrame(
        {
            "A": grouped["A"].var(),
            "B": grouped["B"].std(),
            "C": grouped["C"].mean(),
            "D": grouped["D"].sem(),
        }
    )
    tm.assert_frame_equal(result, expected)


def test_std_masked_dtype(any_numeric_ea_dtype):
    # GH#35516
    df = DataFrame(
        {
            "a": [2, 1, 1, 1, 2, 2, 1],
            "b": Series([pd.NA, 1, 2, 1, 1, 1, 2], dtype="Float64"),
        }
    )
    result = df.groupby("a").std()
    expected = DataFrame(
        {"b": [0.57735, 0]}, index=Index([1, 2], name="a"), dtype="Float64"
    )
    tm.assert_frame_equal(result, expected)


def test_agg_str_with_kwarg_axis_1_raises(df, reduction_func):
    gb = df.groupby(level=0)
    warn_msg = f"DataFrameGroupBy.{reduction_func} with axis=1 is deprecated"
    if reduction_func in ("idxmax", "idxmin"):
        error = TypeError
        msg = "'[<>]' not supported between instances of 'float' and 'str'"
        warn = FutureWarning
    else:
        error = ValueError
        msg = f"Operation {reduction_func} does not support axis=1"
        warn = None
    with pytest.raises(error, match=msg):
        with tm.assert_produces_warning(warn, match=warn_msg):
            gb.agg(reduction_func, axis=1)


@pytest.mark.parametrize(
    "func, expected, dtype, result_dtype_dict",
    [
        ("sum", [5, 7, 9], "int64", {}),
        ("std", [4.5**0.5] * 3, int, {"i": float, "j": float, "k": float}),
        ("var", [4.5] * 3, int, {"i": float, "j": float, "k": float}),
        ("sum", [5, 7, 9], "Int64", {"j": "int64"}),
        ("std", [4.5**0.5] * 3, "Int64", {"i": float, "j": float, "k": float}),
        ("var", [4.5] * 3, "Int64", {"i": "float64", "j": "float64", "k": "float64"}),
    ],
)
def test_multiindex_groupby_mixed_cols_axis1(func, expected, dtype, result_dtype_dict):
    # GH#43209
    df = DataFrame(
        [[1, 2, 3, 4, 5, 6]] * 3,
        columns=MultiIndex.from_product([["a", "b"], ["i", "j", "k"]]),
    ).astype({("a", "j"): dtype, ("b", "j"): dtype})

    msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby(level=1, axis=1)
    result = gb.agg(func)
    expected = DataFrame([expected] * 3, columns=["i", "j", "k"]).astype(
        result_dtype_dict
    )

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "func, expected_data, result_dtype_dict",
    [
        ("sum", [[2, 4], [10, 12], [18, 20]], {10: "int64", 20: "int64"}),
        # std should ideally return Int64 / Float64 #43330
        ("std", [[2**0.5] * 2] * 3, "float64"),
        ("var", [[2] * 2] * 3, {10: "float64", 20: "float64"}),
    ],
)
def test_groupby_mixed_cols_axis1(func, expected_data, result_dtype_dict):
    # GH#43209
    df = DataFrame(
        np.arange(12).reshape(3, 4),
        index=Index([0, 1, 0], name="y"),
        columns=Index([10, 20, 10, 20], name="x"),
        dtype="int64",
    ).astype({10: "Int64"})

    msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby("x", axis=1)
    result = gb.agg(func)
    expected = DataFrame(
        data=expected_data,
        index=Index([0, 1, 0], name="y"),
        columns=Index([10, 20], name="x"),
    ).astype(result_dtype_dict)
    tm.assert_frame_equal(result, expected)


def test_aggregate_item_by_item(df):
    grouped = df.groupby("A")

    aggfun_0 = lambda ser: ser.size
    result = grouped.agg(aggfun_0)
    foosum = (df.A == "foo").sum()
    barsum = (df.A == "bar").sum()
    K = len(result.columns)

    # GH5782
    exp = Series(np.array([foosum] * K), index=list("BCD"), name="foo")
    tm.assert_series_equal(result.xs("foo"), exp)

    exp = Series(np.array([barsum] * K), index=list("BCD"), name="bar")
    tm.assert_almost_equal(result.xs("bar"), exp)

    def aggfun_1(ser):
        return ser.size

    result = DataFrame().groupby(df.A).agg(aggfun_1)
    assert isinstance(result, DataFrame)
    assert len(result) == 0


def test_wrap_agg_out(three_group):
    grouped = three_group.groupby(["A", "B"])

    def func(ser):
        if ser.dtype in (object, "string"):
            raise TypeError("Test error message")
        return ser.sum()

    with pytest.raises(TypeError, match="Test error message"):
        grouped.aggregate(func)
    result = grouped[["D", "E", "F"]].aggregate(func)
    exp_grouped = three_group.loc[:, ["A", "B", "D", "E", "F"]]
    expected = exp_grouped.groupby(["A", "B"]).aggregate(func)
    tm.assert_frame_equal(result, expected)


def test_agg_multiple_functions_maintain_order(df):
    # GH #610
    funcs = [("mean", np.mean), ("max", np.max), ("min", np.min)]
    msg = "is currently using SeriesGroupBy.mean"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("A")["C"].agg(funcs)
    exp_cols = Index(["mean", "max", "min"])

    tm.assert_index_equal(result.columns, exp_cols)


def test_series_index_name(df):
    grouped = df.loc[:, ["C"]].groupby(df["A"])
    result = grouped.agg(lambda x: x.mean())
    assert result.index.name == "A"


def test_agg_multiple_functions_same_name():
    # GH 30880
    df = DataFrame(
        np.random.default_rng(2).standard_normal((1000, 3)),
        index=pd.date_range("1/1/2012", freq="s", periods=1000),
        columns=["A", "B", "C"],
    )
    result = df.resample("3min").agg(
        {"A": [partial(np.quantile, q=0.9999), partial(np.quantile, q=0.1111)]}
    )
    expected_index = pd.date_range("1/1/2012", freq="3min", periods=6)
    expected_columns = MultiIndex.from_tuples([("A", "quantile"), ("A", "quantile")])
    expected_values = np.array(
        [df.resample("3min").A.quantile(q=q).values for q in [0.9999, 0.1111]]
    ).T
    expected = DataFrame(
        expected_values, columns=expected_columns, index=expected_index
    )
    tm.assert_frame_equal(result, expected)


def test_agg_multiple_functions_same_name_with_ohlc_present():
    # GH 30880
    # ohlc expands dimensions, so different test to the above is required.
    df = DataFrame(
        np.random.default_rng(2).standard_normal((1000, 3)),
        index=pd.date_range("1/1/2012", freq="s", periods=1000, name="dti"),
        columns=Index(["A", "B", "C"], name="alpha"),
    )
    result = df.resample("3min").agg(
        {"A": ["ohlc", partial(np.quantile, q=0.9999), partial(np.quantile, q=0.1111)]}
    )
    expected_index = pd.date_range("1/1/2012", freq="3min", periods=6, name="dti")
    expected_columns = MultiIndex.from_tuples(
        [
            ("A", "ohlc", "open"),
            ("A", "ohlc", "high"),
            ("A", "ohlc", "low"),
            ("A", "ohlc", "close"),
            ("A", "quantile", "A"),
            ("A", "quantile", "A"),
        ],
        names=["alpha", None, None],
    )
    non_ohlc_expected_values = np.array(
        [df.resample("3min").A.quantile(q=q).values for q in [0.9999, 0.1111]]
    ).T
    expected_values = np.hstack(
        [df.resample("3min").A.ohlc(), non_ohlc_expected_values]
    )
    expected = DataFrame(
        expected_values, columns=expected_columns, index=expected_index
    )
    tm.assert_frame_equal(result, expected)


def test_multiple_functions_tuples_and_non_tuples(df):
    # #1359
    # Columns B and C would cause partial failure
    df = df.drop(columns=["B", "C"])

    funcs = [("foo", "mean"), "std"]
    ex_funcs = [("foo", "mean"), ("std", "std")]

    result = df.groupby("A")["D"].agg(funcs)
    expected = df.groupby("A")["D"].agg(ex_funcs)
    tm.assert_frame_equal(result, expected)

    result = df.groupby("A").agg(funcs)
    expected = df.groupby("A").agg(ex_funcs)
    tm.assert_frame_equal(result, expected)


def test_more_flexible_frame_multi_function(df):
    grouped = df.groupby("A")

    exmean = grouped.agg({"C": "mean", "D": "mean"})
    exstd = grouped.agg({"C": "std", "D": "std"})

    expected = concat([exmean, exstd], keys=["mean", "std"], axis=1)
    expected = expected.swaplevel(0, 1, axis=1).sort_index(level=0, axis=1)

    d = {"C": ["mean", "std"], "D": ["mean", "std"]}
    result = grouped.aggregate(d)

    tm.assert_frame_equal(result, expected)

    # be careful
    result = grouped.aggregate({"C": "mean", "D": ["mean", "std"]})
    expected = grouped.aggregate({"C": "mean", "D": ["mean", "std"]})
    tm.assert_frame_equal(result, expected)

    def numpymean(x):
        return np.mean(x)

    def numpystd(x):
        return np.std(x, ddof=1)

    # this uses column selection & renaming
    msg = r"nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        d = {"C": "mean", "D": {"foo": "mean", "bar": "std"}}
        grouped.aggregate(d)

    # But without renaming, these functions are OK
    d = {"C": ["mean"], "D": [numpymean, numpystd]}
    grouped.aggregate(d)


def test_multi_function_flexible_mix(df):
    # GH #1268
    grouped = df.groupby("A")

    # Expected
    d = {"C": {"foo": "mean", "bar": "std"}, "D": {"sum": "sum"}}
    # this uses column selection & renaming
    msg = r"nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        grouped.aggregate(d)

    # Test 1
    d = {"C": {"foo": "mean", "bar": "std"}, "D": "sum"}
    # this uses column selection & renaming
    with pytest.raises(SpecificationError, match=msg):
        grouped.aggregate(d)

    # Test 2
    d = {"C": {"foo": "mean", "bar": "std"}, "D": "sum"}
    # this uses column selection & renaming
    with pytest.raises(SpecificationError, match=msg):
        grouped.aggregate(d)


def test_groupby_agg_coercing_bools():
    # issue 14873
    dat = DataFrame({"a": [1, 1, 2, 2], "b": [0, 1, 2, 3], "c": [None, None, 1, 1]})
    gp = dat.groupby("a")

    index = Index([1, 2], name="a")

    result = gp["b"].aggregate(lambda x: (x != 0).all())
    expected = Series([False, True], index=index, name="b")
    tm.assert_series_equal(result, expected)

    result = gp["c"].aggregate(lambda x: x.isnull().all())
    expected = Series([True, False], index=index, name="c")
    tm.assert_series_equal(result, expected)


def test_groupby_agg_dict_with_getitem():
    # issue 25471
    dat = DataFrame({"A": ["A", "A", "B", "B", "B"], "B": [1, 2, 1, 1, 2]})
    result = dat.groupby("A")[["B"]].agg({"B": "sum"})

    expected = DataFrame({"B": [3, 4]}, index=["A", "B"]).rename_axis("A", axis=0)

    tm.assert_frame_equal(result, expected)


def test_groupby_agg_dict_dup_columns():
    # GH#55006
    df = DataFrame(
        [[1, 2, 3, 4], [1, 3, 4, 5], [2, 4, 5, 6]],
        columns=["a", "b", "c", "c"],
    )
    gb = df.groupby("a")
    result = gb.agg({"b": "sum"})
    expected = DataFrame({"b": [5, 4]}, index=Index([1, 2], name="a"))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "op",
    [
        lambda x: x.sum(),
        lambda x: x.cumsum(),
        lambda x: x.transform("sum"),
        lambda x: x.transform("cumsum"),
        lambda x: x.agg("sum"),
        lambda x: x.agg("cumsum"),
    ],
)
def test_bool_agg_dtype(op):
    # GH 7001
    # Bool sum aggregations result in int
    df = DataFrame({"a": [1, 1], "b": [False, True]})
    s = df.set_index("a")["b"]

    result = op(df.groupby("a"))["b"].dtype
    assert is_integer_dtype(result)

    result = op(s.groupby("a")).dtype
    assert is_integer_dtype(result)


@pytest.mark.parametrize(
    "keys, agg_index",
    [
        (["a"], Index([1], name="a")),
        (["a", "b"], MultiIndex([[1], [2]], [[0], [0]], names=["a", "b"])),
    ],
)
@pytest.mark.parametrize(
    "input_dtype", ["bool", "int32", "int64", "float32", "float64"]
)
@pytest.mark.parametrize(
    "result_dtype", ["bool", "int32", "int64", "float32", "float64"]
)
@pytest.mark.parametrize("method", ["apply", "aggregate", "transform"])
def test_callable_result_dtype_frame(
    keys, agg_index, input_dtype, result_dtype, method
):
    # GH 21240
    df = DataFrame({"a": [1], "b": [2], "c": [True]})
    df["c"] = df["c"].astype(input_dtype)
    op = getattr(df.groupby(keys)[["c"]], method)
    result = op(lambda x: x.astype(result_dtype).iloc[0])
    expected_index = pd.RangeIndex(0, 1) if method == "transform" else agg_index
    expected = DataFrame({"c": [df["c"].iloc[0]]}, index=expected_index).astype(
        result_dtype
    )
    if method == "apply":
        expected.columns.names = [0]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "keys, agg_index",
    [
        (["a"], Index([1], name="a")),
        (["a", "b"], MultiIndex([[1], [2]], [[0], [0]], names=["a", "b"])),
    ],
)
@pytest.mark.parametrize("input", [True, 1, 1.0])
@pytest.mark.parametrize("dtype", [bool, int, float])
@pytest.mark.parametrize("method", ["apply", "aggregate", "transform"])
def test_callable_result_dtype_series(keys, agg_index, input, dtype, method):
    # GH 21240
    df = DataFrame({"a": [1], "b": [2], "c": [input]})
    op = getattr(df.groupby(keys)["c"], method)
    result = op(lambda x: x.astype(dtype).iloc[0])
    expected_index = pd.RangeIndex(0, 1) if method == "transform" else agg_index
    expected = Series([df["c"].iloc[0]], index=expected_index, name="c").astype(dtype)
    tm.assert_series_equal(result, expected)


def test_order_aggregate_multiple_funcs():
    # GH 25692
    df = DataFrame({"A": [1, 1, 2, 2], "B": [1, 2, 3, 4]})

    res = df.groupby("A").agg(["sum", "max", "mean", "ohlc", "min"])
    result = res.columns.levels[1]

    expected = Index(["sum", "max", "mean", "ohlc", "min"])

    tm.assert_index_equal(result, expected)


def test_ohlc_ea_dtypes(any_numeric_ea_dtype):
    # GH#37493
    df = DataFrame(
        {"a": [1, 1, 2, 3, 4, 4], "b": [22, 11, pd.NA, 10, 20, pd.NA]},
        dtype=any_numeric_ea_dtype,
    )
    gb = df.groupby("a")
    result = gb.ohlc()
    expected = DataFrame(
        [[22, 22, 11, 11], [pd.NA] * 4, [10] * 4, [20] * 4],
        columns=MultiIndex.from_product([["b"], ["open", "high", "low", "close"]]),
        index=Index([1, 2, 3, 4], dtype=any_numeric_ea_dtype, name="a"),
        dtype=any_numeric_ea_dtype,
    )
    tm.assert_frame_equal(result, expected)

    gb2 = df.groupby("a", as_index=False)
    result2 = gb2.ohlc()
    expected2 = expected.reset_index()
    tm.assert_frame_equal(result2, expected2)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("how", ["first", "last", "min", "max", "mean", "median"])
def test_uint64_type_handling(dtype, how):
    # GH 26310
    df = DataFrame({"x": 6903052872240755750, "y": [1, 2]})
    expected = df.groupby("y").agg({"x": how})
    df.x = df.x.astype(dtype)
    result = df.groupby("y").agg({"x": how})
    if how not in ("mean", "median"):
        # mean and median always result in floats
        result.x = result.x.astype(np.int64)
    tm.assert_frame_equal(result, expected, check_exact=True)


def test_func_duplicates_raises():
    # GH28426
    msg = "Function names"
    df = DataFrame({"A": [0, 0, 1, 1], "B": [1, 2, 3, 4]})
    with pytest.raises(SpecificationError, match=msg):
        df.groupby("A").agg(["min", "min"])


@pytest.mark.parametrize(
    "index",
    [
        pd.CategoricalIndex(list("abc")),
        pd.interval_range(0, 3),
        pd.period_range("2020", periods=3, freq="D"),
        MultiIndex.from_tuples([("a", 0), ("a", 1), ("b", 0)]),
    ],
)
def test_agg_index_has_complex_internals(index):
    # GH 31223
    df = DataFrame({"group": [1, 1, 2], "value": [0, 1, 0]}, index=index)
    result = df.groupby("group").agg({"value": Series.nunique})
    expected = DataFrame({"group": [1, 2], "value": [2, 1]}).set_index("group")
    tm.assert_frame_equal(result, expected)


def test_agg_split_block():
    # https://github.com/pandas-dev/pandas/issues/31522
    df = DataFrame(
        {
            "key1": ["a", "a", "b", "b", "a"],
            "key2": ["one", "two", "one", "two", "one"],
            "key3": ["three", "three", "three", "six", "six"],
        }
    )
    result = df.groupby("key1").min()
    expected = DataFrame(
        {"key2": ["one", "one"], "key3": ["six", "six"]},
        index=Index(["a", "b"], name="key1"),
    )
    tm.assert_frame_equal(result, expected)


def test_agg_split_object_part_datetime():
    # https://github.com/pandas-dev/pandas/pull/31616
    df = DataFrame(
        {
            "A": pd.date_range("2000", periods=4),
            "B": ["a", "b", "c", "d"],
            "C": [1, 2, 3, 4],
            "D": ["b", "c", "d", "e"],
            "E": pd.date_range("2000", periods=4),
            "F": [1, 2, 3, 4],
        }
    ).astype(object)
    result = df.groupby([0, 0, 0, 0]).min()
    expected = DataFrame(
        {
            "A": [pd.Timestamp("2000")],
            "B": ["a"],
            "C": [1],
            "D": ["b"],
            "E": [pd.Timestamp("2000")],
            "F": [1],
        },
        index=np.array([0]),
        dtype=object,
    )
    tm.assert_frame_equal(result, expected)


class TestNamedAggregationSeries:
    def test_series_named_agg(self):
        df = Series([1, 2, 3, 4])
        gr = df.groupby([0, 0, 1, 1])
        result = gr.agg(a="sum", b="min")
        expected = DataFrame(
            {"a": [3, 7], "b": [1, 3]}, columns=["a", "b"], index=np.array([0, 1])
        )
        tm.assert_frame_equal(result, expected)

        result = gr.agg(b="min", a="sum")
        expected = expected[["b", "a"]]
        tm.assert_frame_equal(result, expected)

    def test_no_args_raises(self):
        gr = Series([1, 2]).groupby([0, 1])
        with pytest.raises(TypeError, match="Must provide"):
            gr.agg()

        # but we do allow this
        result = gr.agg([])
        expected = DataFrame(columns=[])
        tm.assert_frame_equal(result, expected)

    def test_series_named_agg_duplicates_no_raises(self):
        # GH28426
        gr = Series([1, 2, 3]).groupby([0, 0, 1])
        grouped = gr.agg(a="sum", b="sum")
        expected = DataFrame({"a": [3, 3], "b": [3, 3]}, index=np.array([0, 1]))
        tm.assert_frame_equal(expected, grouped)

    def test_mangled(self):
        gr = Series([1, 2, 3]).groupby([0, 0, 1])
        result = gr.agg(a=lambda x: 0, b=lambda x: 1)
        expected = DataFrame({"a": [0, 0], "b": [1, 1]}, index=np.array([0, 1]))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "inp",
        [
            pd.NamedAgg(column="anything", aggfunc="min"),
            ("anything", "min"),
            ["anything", "min"],
        ],
    )
    def test_named_agg_nametuple(self, inp):
        # GH34422
        s = Series([1, 1, 2, 2, 3, 3, 4, 5])
        msg = f"func is expected but received {type(inp).__name__}"
        with pytest.raises(TypeError, match=msg):
            s.groupby(s.values).agg(a=inp)


class TestNamedAggregationDataFrame:
    def test_agg_relabel(self):
        df = DataFrame(
            {"group": ["a", "a", "b", "b"], "A": [0, 1, 2, 3], "B": [5, 6, 7, 8]}
        )
        result = df.groupby("group").agg(a_max=("A", "max"), b_max=("B", "max"))
        expected = DataFrame(
            {"a_max": [1, 3], "b_max": [6, 8]},
            index=Index(["a", "b"], name="group"),
            columns=["a_max", "b_max"],
        )
        tm.assert_frame_equal(result, expected)

        # order invariance
        p98 = functools.partial(np.percentile, q=98)
        result = df.groupby("group").agg(
            b_min=("B", "min"),
            a_min=("A", "min"),
            a_mean=("A", "mean"),
            a_max=("A", "max"),
            b_max=("B", "max"),
            a_98=("A", p98),
        )
        expected = DataFrame(
            {
                "b_min": [5, 7],
                "a_min": [0, 2],
                "a_mean": [0.5, 2.5],
                "a_max": [1, 3],
                "b_max": [6, 8],
                "a_98": [0.98, 2.98],
            },
            index=Index(["a", "b"], name="group"),
            columns=["b_min", "a_min", "a_mean", "a_max", "b_max", "a_98"],
        )
        tm.assert_frame_equal(result, expected)

    def test_agg_relabel_non_identifier(self):
        df = DataFrame(
            {"group": ["a", "a", "b", "b"], "A": [0, 1, 2, 3], "B": [5, 6, 7, 8]}
        )

        result = df.groupby("group").agg(**{"my col": ("A", "max")})
        expected = DataFrame({"my col": [1, 3]}, index=Index(["a", "b"], name="group"))
        tm.assert_frame_equal(result, expected)

    def test_duplicate_no_raises(self):
        # GH 28426, if use same input function on same column,
        # no error should raise
        df = DataFrame({"A": [0, 0, 1, 1], "B": [1, 2, 3, 4]})

        grouped = df.groupby("A").agg(a=("B", "min"), b=("B", "min"))
        expected = DataFrame({"a": [1, 3], "b": [1, 3]}, index=Index([0, 1], name="A"))
        tm.assert_frame_equal(grouped, expected)

        quant50 = functools.partial(np.percentile, q=50)
        quant70 = functools.partial(np.percentile, q=70)
        quant50.__name__ = "quant50"
        quant70.__name__ = "quant70"

        test = DataFrame({"col1": ["a", "a", "b", "b", "b"], "col2": [1, 2, 3, 4, 5]})

        grouped = test.groupby("col1").agg(
            quantile_50=("col2", quant50), quantile_70=("col2", quant70)
        )
        expected = DataFrame(
            {"quantile_50": [1.5, 4.0], "quantile_70": [1.7, 4.4]},
            index=Index(["a", "b"], name="col1"),
        )
        tm.assert_frame_equal(grouped, expected)

    def test_agg_relabel_with_level(self):
        df = DataFrame(
            {"A": [0, 0, 1, 1], "B": [1, 2, 3, 4]},
            index=MultiIndex.from_product([["A", "B"], ["a", "b"]]),
        )
        result = df.groupby(level=0).agg(
            aa=("A", "max"), bb=("A", "min"), cc=("B", "mean")
        )
        expected = DataFrame(
            {"aa": [0, 1], "bb": [0, 1], "cc": [1.5, 3.5]}, index=["A", "B"]
        )
        tm.assert_frame_equal(result, expected)

    def test_agg_relabel_other_raises(self):
        df = DataFrame({"A": [0, 0, 1], "B": [1, 2, 3]})
        grouped = df.groupby("A")
        match = "Must provide"
        with pytest.raises(TypeError, match=match):
            grouped.agg(foo=1)

        with pytest.raises(TypeError, match=match):
            grouped.agg()

        with pytest.raises(TypeError, match=match):
            grouped.agg(a=("B", "max"), b=(1, 2, 3))

    def test_missing_raises(self):
        df = DataFrame({"A": [0, 1], "B": [1, 2]})
        match = re.escape("Column(s) ['C'] do not exist")
        with pytest.raises(KeyError, match=match):
            df.groupby("A").agg(c=("C", "sum"))

    def test_agg_namedtuple(self):
        df = DataFrame({"A": [0, 1], "B": [1, 2]})
        result = df.groupby("A").agg(
            b=pd.NamedAgg("B", "sum"), c=pd.NamedAgg(column="B", aggfunc="count")
        )
        expected = df.groupby("A").agg(b=("B", "sum"), c=("B", "count"))
        tm.assert_frame_equal(result, expected)

    def test_mangled(self):
        df = DataFrame({"A": [0, 1], "B": [1, 2], "C": [3, 4]})
        result = df.groupby("A").agg(b=("B", lambda x: 0), c=("C", lambda x: 1))
        expected = DataFrame({"b": [0, 0], "c": [1, 1]}, index=Index([0, 1], name="A"))
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "agg_col1, agg_col2, agg_col3, agg_result1, agg_result2, agg_result3",
    [
        (
            (("y", "A"), "max"),
            (("y", "A"), np.mean),
            (("y", "B"), "mean"),
            [1, 3],
            [0.5, 2.5],
            [5.5, 7.5],
        ),
        (
            (("y", "A"), lambda x: max(x)),
            (("y", "A"), lambda x: 1),
            (("y", "B"), np.mean),
            [1, 3],
            [1, 1],
            [5.5, 7.5],
        ),
        (
            pd.NamedAgg(("y", "A"), "max"),
            pd.NamedAgg(("y", "B"), np.mean),
            pd.NamedAgg(("y", "A"), lambda x: 1),
            [1, 3],
            [5.5, 7.5],
            [1, 1],
        ),
    ],
)
def test_agg_relabel_multiindex_column(
    agg_col1, agg_col2, agg_col3, agg_result1, agg_result2, agg_result3
):
    # GH 29422, add tests for multiindex column cases
    df = DataFrame(
        {"group": ["a", "a", "b", "b"], "A": [0, 1, 2, 3], "B": [5, 6, 7, 8]}
    )
    df.columns = MultiIndex.from_tuples([("x", "group"), ("y", "A"), ("y", "B")])
    idx = Index(["a", "b"], name=("x", "group"))

    result = df.groupby(("x", "group")).agg(a_max=(("y", "A"), "max"))
    expected = DataFrame({"a_max": [1, 3]}, index=idx)
    tm.assert_frame_equal(result, expected)

    msg = "is currently using SeriesGroupBy.mean"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby(("x", "group")).agg(
            col_1=agg_col1, col_2=agg_col2, col_3=agg_col3
        )
    expected = DataFrame(
        {"col_1": agg_result1, "col_2": agg_result2, "col_3": agg_result3}, index=idx
    )
    tm.assert_frame_equal(result, expected)


def test_agg_relabel_multiindex_raises_not_exist():
    # GH 29422, add test for raises scenario when aggregate column does not exist
    df = DataFrame(
        {"group": ["a", "a", "b", "b"], "A": [0, 1, 2, 3], "B": [5, 6, 7, 8]}
    )
    df.columns = MultiIndex.from_tuples([("x", "group"), ("y", "A"), ("y", "B")])

    with pytest.raises(KeyError, match="do not exist"):
        df.groupby(("x", "group")).agg(a=(("Y", "a"), "max"))


def test_agg_relabel_multiindex_duplicates():
    # GH29422, add test for raises scenario when getting duplicates
    # GH28426, after this change, duplicates should also work if the relabelling is
    # different
    df = DataFrame(
        {"group": ["a", "a", "b", "b"], "A": [0, 1, 2, 3], "B": [5, 6, 7, 8]}
    )
    df.columns = MultiIndex.from_tuples([("x", "group"), ("y", "A"), ("y", "B")])

    result = df.groupby(("x", "group")).agg(
        a=(("y", "A"), "min"), b=(("y", "A"), "min")
    )
    idx = Index(["a", "b"], name=("x", "group"))
    expected = DataFrame({"a": [0, 2], "b": [0, 2]}, index=idx)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("kwargs", [{"c": ["min"]}, {"b": [], "c": ["min"]}])
def test_groupby_aggregate_empty_key(kwargs):
    # GH: 32580
    df = DataFrame({"a": [1, 1, 2], "b": [1, 2, 3], "c": [1, 2, 4]})
    result = df.groupby("a").agg(kwargs)
    expected = DataFrame(
        [1, 4],
        index=Index([1, 2], dtype="int64", name="a"),
        columns=MultiIndex.from_tuples([["c", "min"]]),
    )
    tm.assert_frame_equal(result, expected)


def test_groupby_aggregate_empty_key_empty_return():
    # GH: 32580 Check if everything works, when return is empty
    df = DataFrame({"a": [1, 1, 2], "b": [1, 2, 3], "c": [1, 2, 4]})
    result = df.groupby("a").agg({"b": []})
    expected = DataFrame(columns=MultiIndex(levels=[["b"], []], codes=[[], []]))
    tm.assert_frame_equal(result, expected)


def test_groupby_aggregate_empty_with_multiindex_frame():
    # GH 39178
    df = DataFrame(columns=["a", "b", "c"])
    result = df.groupby(["a", "b"], group_keys=False).agg(d=("c", list))
    expected = DataFrame(
        columns=["d"], index=MultiIndex([[], []], [[], []], names=["a", "b"])
    )
    tm.assert_frame_equal(result, expected)


def test_grouby_agg_loses_results_with_as_index_false_relabel():
    # GH 32240: When the aggregate function relabels column names and
    # as_index=False is specified, the results are dropped.

    df = DataFrame(
        {"key": ["x", "y", "z", "x", "y", "z"], "val": [1.0, 0.8, 2.0, 3.0, 3.6, 0.75]}
    )

    grouped = df.groupby("key", as_index=False)
    result = grouped.agg(min_val=pd.NamedAgg(column="val", aggfunc="min"))
    expected = DataFrame({"key": ["x", "y", "z"], "min_val": [1.0, 0.8, 0.75]})
    tm.assert_frame_equal(result, expected)


def test_grouby_agg_loses_results_with_as_index_false_relabel_multiindex():
    # GH 32240: When the aggregate function relabels column names and
    # as_index=False is specified, the results are dropped. Check if
    # multiindex is returned in the right order

    df = DataFrame(
        {
            "key": ["x", "y", "x", "y", "x", "x"],
            "key1": ["a", "b", "c", "b", "a", "c"],
            "val": [1.0, 0.8, 2.0, 3.0, 3.6, 0.75],
        }
    )

    grouped = df.groupby(["key", "key1"], as_index=False)
    result = grouped.agg(min_val=pd.NamedAgg(column="val", aggfunc="min"))
    expected = DataFrame(
        {"key": ["x", "x", "y"], "key1": ["a", "c", "b"], "min_val": [1.0, 0.75, 0.8]}
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "func", [lambda s: s.mean(), lambda s: np.mean(s), lambda s: np.nanmean(s)]
)
def test_multiindex_custom_func(func):
    # GH 31777
    data = [[1, 4, 2], [5, 7, 1]]
    df = DataFrame(
        data,
        columns=MultiIndex.from_arrays(
            [[1, 1, 2], [3, 4, 3]], names=["Sisko", "Janeway"]
        ),
    )
    result = df.groupby(np.array([0, 1])).agg(func)
    expected_dict = {
        (1, 3): {0: 1.0, 1: 5.0},
        (1, 4): {0: 4.0, 1: 7.0},
        (2, 3): {0: 2.0, 1: 1.0},
    }
    expected = DataFrame(expected_dict, index=np.array([0, 1]), columns=df.columns)
    tm.assert_frame_equal(result, expected)


def myfunc(s):
    return np.percentile(s, q=0.90)


@pytest.mark.parametrize("func", [lambda s: np.percentile(s, q=0.90), myfunc])
def test_lambda_named_agg(func):
    # see gh-28467
    animals = DataFrame(
        {
            "kind": ["cat", "dog", "cat", "dog"],
            "height": [9.1, 6.0, 9.5, 34.0],
            "weight": [7.9, 7.5, 9.9, 198.0],
        }
    )

    result = animals.groupby("kind").agg(
        mean_height=("height", "mean"), perc90=("height", func)
    )
    expected = DataFrame(
        [[9.3, 9.1036], [20.0, 6.252]],
        columns=["mean_height", "perc90"],
        index=Index(["cat", "dog"], name="kind"),
    )

    tm.assert_frame_equal(result, expected)


def test_aggregate_mixed_types():
    # GH 16916
    df = DataFrame(
        data=np.array([0] * 9).reshape(3, 3), columns=list("XYZ"), index=list("abc")
    )
    df["grouping"] = ["group 1", "group 1", 2]
    result = df.groupby("grouping").aggregate(lambda x: x.tolist())
    expected_data = [[[0], [0], [0]], [[0, 0], [0, 0], [0, 0]]]
    expected = DataFrame(
        expected_data,
        index=Index([2, "group 1"], dtype="object", name="grouping"),
        columns=Index(["X", "Y", "Z"]),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.xfail(reason="Not implemented;see GH 31256")
def test_aggregate_udf_na_extension_type():
    # https://github.com/pandas-dev/pandas/pull/31359
    # This is currently failing to cast back to Int64Dtype.
    # The presence of the NA causes two problems
    # 1. NA is not an instance of Int64Dtype.type (numpy.int64)
    # 2. The presence of an NA forces object type, so the non-NA values is
    #    a Python int rather than a NumPy int64. Python ints aren't
    #    instances of numpy.int64.
    def aggfunc(x):
        if all(x > 2):
            return 1
        else:
            return pd.NA

    df = DataFrame({"A": pd.array([1, 2, 3])})
    result = df.groupby([1, 1, 2]).agg(aggfunc)
    expected = DataFrame({"A": pd.array([1, pd.NA], dtype="Int64")}, index=[1, 2])
    tm.assert_frame_equal(result, expected)


class TestLambdaMangling:
    def test_basic(self):
        df = DataFrame({"A": [0, 0, 1, 1], "B": [1, 2, 3, 4]})
        result = df.groupby("A").agg({"B": [lambda x: 0, lambda x: 1]})

        expected = DataFrame(
            {("B", "<lambda_0>"): [0, 0], ("B", "<lambda_1>"): [1, 1]},
            index=Index([0, 1], name="A"),
        )
        tm.assert_frame_equal(result, expected)

    def test_mangle_series_groupby(self):
        gr = Series([1, 2, 3, 4]).groupby([0, 0, 1, 1])
        result = gr.agg([lambda x: 0, lambda x: 1])
        exp_data = {"<lambda_0>": [0, 0], "<lambda_1>": [1, 1]}
        expected = DataFrame(exp_data, index=np.array([0, 1]))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.xfail(reason="GH-26611. kwargs for multi-agg.")
    def test_with_kwargs(self):
        f1 = lambda x, y, b=1: x.sum() + y + b
        f2 = lambda x, y, b=2: x.sum() + y * b
        result = Series([1, 2]).groupby([0, 0]).agg([f1, f2], 0)
        expected = DataFrame({"<lambda_0>": [4], "<lambda_1>": [6]})
        tm.assert_frame_equal(result, expected)

        result = Series([1, 2]).groupby([0, 0]).agg([f1, f2], 0, b=10)
        expected = DataFrame({"<lambda_0>": [13], "<lambda_1>": [30]})
        tm.assert_frame_equal(result, expected)

    def test_agg_with_one_lambda(self):
        # GH 25719, write tests for DataFrameGroupby.agg with only one lambda
        df = DataFrame(
            {
                "kind": ["cat", "dog", "cat", "dog"],
                "height": [9.1, 6.0, 9.5, 34.0],
                "weight": [7.9, 7.5, 9.9, 198.0],
            }
        )

        columns = ["height_sqr_min", "height_max", "weight_max"]
        expected = DataFrame(
            {
                "height_sqr_min": [82.81, 36.00],
                "height_max": [9.5, 34.0],
                "weight_max": [9.9, 198.0],
            },
            index=Index(["cat", "dog"], name="kind"),
            columns=columns,
        )

        # check pd.NameAgg case
        result1 = df.groupby(by="kind").agg(
            height_sqr_min=pd.NamedAgg(
                column="height", aggfunc=lambda x: np.min(x**2)
            ),
            height_max=pd.NamedAgg(column="height", aggfunc="max"),
            weight_max=pd.NamedAgg(column="weight", aggfunc="max"),
        )
        tm.assert_frame_equal(result1, expected)

        # check agg(key=(col, aggfunc)) case
        result2 = df.groupby(by="kind").agg(
            height_sqr_min=("height", lambda x: np.min(x**2)),
            height_max=("height", "max"),
            weight_max=("weight", "max"),
        )
        tm.assert_frame_equal(result2, expected)

    def test_agg_multiple_lambda(self):
        # GH25719, test for DataFrameGroupby.agg with multiple lambdas
        # with mixed aggfunc
        df = DataFrame(
            {
                "kind": ["cat", "dog", "cat", "dog"],
                "height": [9.1, 6.0, 9.5, 34.0],
                "weight": [7.9, 7.5, 9.9, 198.0],
            }
        )
        columns = [
            "height_sqr_min",
            "height_max",
            "weight_max",
            "height_max_2",
            "weight_min",
        ]
        expected = DataFrame(
            {
                "height_sqr_min": [82.81, 36.00],
                "height_max": [9.5, 34.0],
                "weight_max": [9.9, 198.0],
                "height_max_2": [9.5, 34.0],
                "weight_min": [7.9, 7.5],
            },
            index=Index(["cat", "dog"], name="kind"),
            columns=columns,
        )

        # check agg(key=(col, aggfunc)) case
        result1 = df.groupby(by="kind").agg(
            height_sqr_min=("height", lambda x: np.min(x**2)),
            height_max=("height", "max"),
            weight_max=("weight", "max"),
            height_max_2=("height", lambda x: np.max(x)),
            weight_min=("weight", lambda x: np.min(x)),
        )
        tm.assert_frame_equal(result1, expected)

        # check pd.NamedAgg case
        result2 = df.groupby(by="kind").agg(
            height_sqr_min=pd.NamedAgg(
                column="height", aggfunc=lambda x: np.min(x**2)
            ),
            height_max=pd.NamedAgg(column="height", aggfunc="max"),
            weight_max=pd.NamedAgg(column="weight", aggfunc="max"),
            height_max_2=pd.NamedAgg(column="height", aggfunc=lambda x: np.max(x)),
            weight_min=pd.NamedAgg(column="weight", aggfunc=lambda x: np.min(x)),
        )
        tm.assert_frame_equal(result2, expected)


def test_groupby_get_by_index():
    # GH 33439
    df = DataFrame({"A": ["S", "W", "W"], "B": [1.0, 1.0, 2.0]})
    res = df.groupby("A").agg({"B": lambda x: x.get(x.index[-1])})
    expected = DataFrame({"A": ["S", "W"], "B": [1.0, 2.0]}).set_index("A")
    tm.assert_frame_equal(res, expected)


@pytest.mark.parametrize(
    "grp_col_dict, exp_data",
    [
        ({"nr": "min", "cat_ord": "min"}, {"nr": [1, 5], "cat_ord": ["a", "c"]}),
        ({"cat_ord": "min"}, {"cat_ord": ["a", "c"]}),
        ({"nr": "min"}, {"nr": [1, 5]}),
    ],
)
def test_groupby_single_agg_cat_cols(grp_col_dict, exp_data):
    # test single aggregations on ordered categorical cols GHGH27800

    # create the result dataframe
    input_df = DataFrame(
        {
            "nr": [1, 2, 3, 4, 5, 6, 7, 8],
            "cat_ord": list("aabbccdd"),
            "cat": list("aaaabbbb"),
        }
    )

    input_df = input_df.astype({"cat": "category", "cat_ord": "category"})
    input_df["cat_ord"] = input_df["cat_ord"].cat.as_ordered()
    result_df = input_df.groupby("cat", observed=False).agg(grp_col_dict)

    # create expected dataframe
    cat_index = pd.CategoricalIndex(
        ["a", "b"], categories=["a", "b"], ordered=False, name="cat", dtype="category"
    )

    expected_df = DataFrame(data=exp_data, index=cat_index)

    if "cat_ord" in expected_df:
        # ordered categorical columns should be preserved
        dtype = input_df["cat_ord"].dtype
        expected_df["cat_ord"] = expected_df["cat_ord"].astype(dtype)

    tm.assert_frame_equal(result_df, expected_df)


@pytest.mark.parametrize(
    "grp_col_dict, exp_data",
    [
        ({"nr": ["min", "max"], "cat_ord": "min"}, [(1, 4, "a"), (5, 8, "c")]),
        ({"nr": "min", "cat_ord": ["min", "max"]}, [(1, "a", "b"), (5, "c", "d")]),
        ({"cat_ord": ["min", "max"]}, [("a", "b"), ("c", "d")]),
    ],
)
def test_groupby_combined_aggs_cat_cols(grp_col_dict, exp_data):
    # test combined aggregations on ordered categorical cols GH27800

    # create the result dataframe
    input_df = DataFrame(
        {
            "nr": [1, 2, 3, 4, 5, 6, 7, 8],
            "cat_ord": list("aabbccdd"),
            "cat": list("aaaabbbb"),
        }
    )

    input_df = input_df.astype({"cat": "category", "cat_ord": "category"})
    input_df["cat_ord"] = input_df["cat_ord"].cat.as_ordered()
    result_df = input_df.groupby("cat", observed=False).agg(grp_col_dict)

    # create expected dataframe
    cat_index = pd.CategoricalIndex(
        ["a", "b"], categories=["a", "b"], ordered=False, name="cat", dtype="category"
    )

    # unpack the grp_col_dict to create the multi-index tuple
    # this tuple will be used to create the expected dataframe index
    multi_index_list = []
    for k, v in grp_col_dict.items():
        if isinstance(v, list):
            multi_index_list.extend([k, value] for value in v)
        else:
            multi_index_list.append([k, v])
    multi_index = MultiIndex.from_tuples(tuple(multi_index_list))

    expected_df = DataFrame(data=exp_data, columns=multi_index, index=cat_index)
    for col in expected_df.columns:
        if isinstance(col, tuple) and "cat_ord" in col:
            # ordered categorical should be preserved
            expected_df[col] = expected_df[col].astype(input_df["cat_ord"].dtype)

    tm.assert_frame_equal(result_df, expected_df)


def test_nonagg_agg():
    # GH 35490 - Single/Multiple agg of non-agg function give same results
    # TODO: agg should raise for functions that don't aggregate
    df = DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 2, 1]})
    g = df.groupby("a")

    result = g.agg(["cumsum"])
    result.columns = result.columns.droplevel(-1)
    expected = g.agg("cumsum")

    tm.assert_frame_equal(result, expected)


def test_aggregate_datetime_objects():
    # https://github.com/pandas-dev/pandas/issues/36003
    # ensure we don't raise an error but keep object dtype for out-of-bounds
    # datetimes
    df = DataFrame(
        {
            "A": ["X", "Y"],
            "B": [
                datetime.datetime(2005, 1, 1, 10, 30, 23, 540000),
                datetime.datetime(3005, 1, 1, 10, 30, 23, 540000),
            ],
        }
    )
    result = df.groupby("A").B.max()
    expected = df.set_index("A")["B"]
    tm.assert_series_equal(result, expected)


def test_groupby_index_object_dtype():
    # GH 40014
    df = DataFrame({"c0": ["x", "x", "x"], "c1": ["x", "x", "y"], "p": [0, 1, 2]})
    df.index = df.index.astype("O")
    grouped = df.groupby(["c0", "c1"])
    res = grouped.p.agg(lambda x: all(x > 0))
    # Check that providing a user-defined function in agg()
    # produces the correct index shape when using an object-typed index.
    expected_index = MultiIndex.from_tuples(
        [("x", "x"), ("x", "y")], names=("c0", "c1")
    )
    expected = Series([False, True], index=expected_index, name="p")
    tm.assert_series_equal(res, expected)


def test_timeseries_groupby_agg():
    # GH#43290

    def func(ser):
        if ser.isna().all():
            return None
        return np.sum(ser)

    df = DataFrame([1.0], index=[pd.Timestamp("2018-01-16 00:00:00+00:00")])
    res = df.groupby(lambda x: 1).agg(func)

    expected = DataFrame([[1.0]], index=[1])
    tm.assert_frame_equal(res, expected)


def test_groupby_agg_precision(any_real_numeric_dtype):
    if any_real_numeric_dtype in tm.ALL_INT_NUMPY_DTYPES:
        max_value = np.iinfo(any_real_numeric_dtype).max
    if any_real_numeric_dtype in tm.FLOAT_NUMPY_DTYPES:
        max_value = np.finfo(any_real_numeric_dtype).max
    if any_real_numeric_dtype in tm.FLOAT_EA_DTYPES:
        max_value = np.finfo(any_real_numeric_dtype.lower()).max
    if any_real_numeric_dtype in tm.ALL_INT_EA_DTYPES:
        max_value = np.iinfo(any_real_numeric_dtype.lower()).max

    df = DataFrame(
        {
            "key1": ["a"],
            "key2": ["b"],
            "key3": pd.array([max_value], dtype=any_real_numeric_dtype),
        }
    )
    arrays = [["a"], ["b"]]
    index = MultiIndex.from_arrays(arrays, names=("key1", "key2"))

    expected = DataFrame(
        {"key3": pd.array([max_value], dtype=any_real_numeric_dtype)}, index=index
    )
    result = df.groupby(["key1", "key2"]).agg(lambda x: x)
    tm.assert_frame_equal(result, expected)


def test_groupby_aggregate_directory(reduction_func):
    # GH#32793
    if reduction_func in ["corrwith", "nth"]:
        return None

    obj = DataFrame([[0, 1], [0, np.nan]])

    result_reduced_series = obj.groupby(0).agg(reduction_func)
    result_reduced_frame = obj.groupby(0).agg({1: reduction_func})

    if reduction_func in ["size", "ngroup"]:
        # names are different: None / 1
        tm.assert_series_equal(
            result_reduced_series, result_reduced_frame[1], check_names=False
        )
    else:
        tm.assert_frame_equal(result_reduced_series, result_reduced_frame)
        tm.assert_series_equal(
            result_reduced_series.dtypes, result_reduced_frame.dtypes
        )


def test_group_mean_timedelta_nat():
    # GH43132
    data = Series(["1 day", "3 days", "NaT"], dtype="timedelta64[ns]")
    expected = Series(["2 days"], dtype="timedelta64[ns]", index=np.array([0]))

    result = data.groupby([0, 0, 0]).mean()

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "input_data, expected_output",
    [
        (  # no timezone
            ["2021-01-01T00:00", "NaT", "2021-01-01T02:00"],
            ["2021-01-01T01:00"],
        ),
        (  # timezone
            ["2021-01-01T00:00-0100", "NaT", "2021-01-01T02:00-0100"],
            ["2021-01-01T01:00-0100"],
        ),
    ],
)
def test_group_mean_datetime64_nat(input_data, expected_output):
    # GH43132
    data = to_datetime(Series(input_data))
    expected = to_datetime(Series(expected_output, index=np.array([0])))

    result = data.groupby([0, 0, 0]).mean()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "func, output", [("mean", [8 + 18j, 10 + 22j]), ("sum", [40 + 90j, 50 + 110j])]
)
def test_groupby_complex(func, output):
    # GH#43701
    data = Series(np.arange(20).reshape(10, 2).dot([1, 2j]))
    result = data.groupby(data.index % 2).agg(func)
    expected = Series(output)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", ["min", "max", "var"])
def test_groupby_complex_raises(func):
    # GH#43701
    data = Series(np.arange(20).reshape(10, 2).dot([1, 2j]))
    msg = "No matching signature found"
    with pytest.raises(TypeError, match=msg):
        data.groupby(data.index % 2).agg(func)


@pytest.mark.parametrize(
    "func", [["min"], ["mean", "max"], {"b": "sum"}, {"b": "prod", "c": "median"}]
)
def test_multi_axis_1_raises(func):
    # GH#46995
    df = DataFrame({"a": [1, 1, 2], "b": [3, 4, 5], "c": [6, 7, 8]})
    msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby("a", axis=1)
    with pytest.raises(NotImplementedError, match="axis other than 0 is not supported"):
        gb.agg(func)


@pytest.mark.parametrize(
    "test, constant",
    [
        ([[20, "A"], [20, "B"], [10, "C"]], {0: [10, 20], 1: ["C", ["A", "B"]]}),
        ([[20, "A"], [20, "B"], [30, "C"]], {0: [20, 30], 1: [["A", "B"], "C"]}),
        ([["a", 1], ["a", 1], ["b", 2], ["b", 3]], {0: ["a", "b"], 1: [1, [2, 3]]}),
        pytest.param(
            [["a", 1], ["a", 2], ["b", 3], ["b", 3]],
            {0: ["a", "b"], 1: [[1, 2], 3]},
            marks=pytest.mark.xfail,
        ),
    ],
)
def test_agg_of_mode_list(test, constant):
    # GH#25581
    df1 = DataFrame(test)
    result = df1.groupby(0).agg(Series.mode)
    # Mode usually only returns 1 value, but can return a list in the case of a tie.

    expected = DataFrame(constant)
    expected = expected.set_index(0)

    tm.assert_frame_equal(result, expected)


def test_dataframe_groupy_agg_list_like_func_with_args():
    # GH#50624
    df = DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    gb = df.groupby("y")

    def foo1(x, a=1, c=0):
        return x.sum() + a + c

    def foo2(x, b=2, c=0):
        return x.sum() + b + c

    msg = r"foo1\(\) got an unexpected keyword argument 'b'"
    with pytest.raises(TypeError, match=msg):
        gb.agg([foo1, foo2], 3, b=3, c=4)

    result = gb.agg([foo1, foo2], 3, c=4)
    expected = DataFrame(
        [[8, 8], [9, 9], [10, 10]],
        index=Index(["a", "b", "c"], name="y"),
        columns=MultiIndex.from_tuples([("x", "foo1"), ("x", "foo2")]),
    )
    tm.assert_frame_equal(result, expected)


def test_series_groupy_agg_list_like_func_with_args():
    # GH#50624
    s = Series([1, 2, 3])
    sgb = s.groupby(s)

    def foo1(x, a=1, c=0):
        return x.sum() + a + c

    def foo2(x, b=2, c=0):
        return x.sum() + b + c

    msg = r"foo1\(\) got an unexpected keyword argument 'b'"
    with pytest.raises(TypeError, match=msg):
        sgb.agg([foo1, foo2], 3, b=3, c=4)

    result = sgb.agg([foo1, foo2], 3, c=4)
    expected = DataFrame(
        [[8, 8], [9, 9], [10, 10]], index=Index([1, 2, 3]), columns=["foo1", "foo2"]
    )
    tm.assert_frame_equal(result, expected)


def test_agg_groupings_selection():
    # GH#51186 - a selected grouping should be in the output of agg
    df = DataFrame({"a": [1, 1, 2], "b": [3, 3, 4], "c": [5, 6, 7]})
    gb = df.groupby(["a", "b"])
    selected_gb = gb[["b", "c"]]
    result = selected_gb.agg(lambda x: x.sum())
    index = MultiIndex(
        levels=[[1, 2], [3, 4]], codes=[[0, 1], [0, 1]], names=["a", "b"]
    )
    expected = DataFrame({"b": [6, 4], "c": [11, 7]}, index=index)
    tm.assert_frame_equal(result, expected)


def test_agg_multiple_with_as_index_false_subset_to_a_single_column():
    # GH#50724
    df = DataFrame({"a": [1, 1, 2], "b": [3, 4, 5]})
    gb = df.groupby("a", as_index=False)["b"]
    result = gb.agg(["sum", "mean"])
    expected = DataFrame({"a": [1, 2], "sum": [7, 5], "mean": [3.5, 5.0]})
    tm.assert_frame_equal(result, expected)


def test_agg_with_as_index_false_with_list():
    # GH#52849
    df = DataFrame({"a1": [0, 0, 1], "a2": [2, 3, 3], "b": [4, 5, 6]})
    gb = df.groupby(by=["a1", "a2"], as_index=False)
    result = gb.agg(["sum"])

    expected = DataFrame(
        data=[[0, 2, 4], [0, 3, 5], [1, 3, 6]],
        columns=MultiIndex.from_tuples([("a1", ""), ("a2", ""), ("b", "sum")]),
    )
    tm.assert_frame_equal(result, expected)


def test_groupby_agg_extension_timedelta_cumsum_with_named_aggregation():
    # GH#41720
    expected = DataFrame(
        {
            "td": {
                0: pd.Timedelta("0 days 01:00:00"),
                1: pd.Timedelta("0 days 01:15:00"),
                2: pd.Timedelta("0 days 01:15:00"),
            }
        }
    )
    df = DataFrame(
        {
            "td": Series(
                ["0 days 01:00:00", "0 days 00:15:00", "0 days 01:15:00"],
                dtype="timedelta64[ns]",
            ),
            "grps": ["a", "a", "b"],
        }
    )
    gb = df.groupby("grps")
    result = gb.agg(td=("td", "cumsum"))
    tm.assert_frame_equal(result, expected)


def test_groupby_aggregation_empty_group():
    # https://github.com/pandas-dev/pandas/issues/18869
    def func(x):
        if len(x) == 0:
            raise ValueError("length must not be 0")
        return len(x)

    df = DataFrame(
        {"A": pd.Categorical(["a", "a"], categories=["a", "b", "c"]), "B": [1, 1]}
    )
    msg = "length must not be 0"
    with pytest.raises(ValueError, match=msg):
        df.groupby("A", observed=False).agg(func)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_fp.py ===
"""
Easy-to-use test-generating code:

cases = '''
exp 2.25
log 2.25
'''

from mpmath import *
mp.dps = 20
for test in cases.splitlines():
    if not test:
        continue
    words = test.split()
    fname = words[0]
    args = words[1:]
    argstr = ", ".join(args)
    testline = "%s(%s)" % (fname, argstr)
    ans = str(eval(testline))
    print "    assert ae(fp.%s, %s)" % (testline, ans)

"""

from mpmath import fp

def ae(x, y, tol=1e-12):
    if x == y:
        return True
    return abs(x-y) <= tol*abs(y)

def test_conj():
    assert fp.conj(4) == 4
    assert fp.conj(3+4j) == 3-4j
    assert fp.fdot([1,2],[3,2+1j], conjugate=True) == 7-2j

def test_fp_number_parts():
    assert ae(fp.arg(3), 0.0)
    assert ae(fp.arg(-3), 3.1415926535897932385)
    assert ae(fp.arg(3j), 1.5707963267948966192)
    assert ae(fp.arg(-3j), -1.5707963267948966192)
    assert ae(fp.arg(2+3j), 0.98279372324732906799)
    assert ae(fp.arg(-1-1j), -2.3561944901923449288)
    assert ae(fp.re(2.5), 2.5)
    assert ae(fp.re(2.5+3j), 2.5)
    assert ae(fp.im(2.5), 0.0)
    assert ae(fp.im(2.5+3j), 3.0)
    assert ae(fp.floor(2.5), 2.0)
    assert ae(fp.floor(2), 2.0)
    assert ae(fp.floor(2.0+0j), (2.0 + 0.0j))
    assert ae(fp.floor(-1.5-0.5j), (-2.0 - 1.0j))
    assert ae(fp.ceil(2.5), 3.0)
    assert ae(fp.ceil(2), 2.0)
    assert ae(fp.ceil(2.0+0j), (2.0 + 0.0j))
    assert ae(fp.ceil(-1.5-0.5j), (-1.0 + 0.0j))

def test_fp_cospi_sinpi():
    assert ae(fp.sinpi(0), 0.0)
    assert ae(fp.sinpi(0.25), 0.7071067811865475244)
    assert ae(fp.sinpi(0.5), 1.0)
    assert ae(fp.sinpi(0.75), 0.7071067811865475244)
    assert ae(fp.sinpi(1), 0.0)
    assert ae(fp.sinpi(1.25), -0.7071067811865475244)
    assert ae(fp.sinpi(1.5), -1.0)
    assert ae(fp.sinpi(1.75), -0.7071067811865475244)
    assert ae(fp.sinpi(2), 0.0)
    assert ae(fp.sinpi(2.25), 0.7071067811865475244)
    assert ae(fp.sinpi(0+3j), (0.0 + 6195.8238636085899556j))
    assert ae(fp.sinpi(0.25+3j), (4381.1091260582448033 + 4381.1090689950686908j))
    assert ae(fp.sinpi(0.5+3j), (6195.8239443081075259 + 0.0j))
    assert ae(fp.sinpi(0.75+3j), (4381.1091260582448033 - 4381.1090689950686908j))
    assert ae(fp.sinpi(1+3j), (0.0 - 6195.8238636085899556j))
    assert ae(fp.sinpi(1.25+3j), (-4381.1091260582448033 - 4381.1090689950686908j))
    assert ae(fp.sinpi(1.5+3j), (-6195.8239443081075259 + 0.0j))
    assert ae(fp.sinpi(1.75+3j), (-4381.1091260582448033 + 4381.1090689950686908j))
    assert ae(fp.sinpi(2+3j), (0.0 + 6195.8238636085899556j))
    assert ae(fp.sinpi(2.25+3j), (4381.1091260582448033 + 4381.1090689950686908j))
    assert ae(fp.sinpi(-0.75), -0.7071067811865475244)
    assert ae(fp.sinpi(-1e-10), -3.1415926535897933529e-10)
    assert ae(fp.sinpi(1e-10), 3.1415926535897933529e-10)
    assert ae(fp.sinpi(1e-10+1e-10j), (3.141592653589793353e-10 + 3.1415926535897933528e-10j))
    assert ae(fp.sinpi(1e-10-1e-10j), (3.141592653589793353e-10 - 3.1415926535897933528e-10j))
    assert ae(fp.sinpi(-1e-10+1e-10j), (-3.141592653589793353e-10 + 3.1415926535897933528e-10j))
    assert ae(fp.sinpi(-1e-10-1e-10j), (-3.141592653589793353e-10 - 3.1415926535897933528e-10j))
    assert ae(fp.cospi(0), 1.0)
    assert ae(fp.cospi(0.25), 0.7071067811865475244)
    assert ae(fp.cospi(0.5), 0.0)
    assert ae(fp.cospi(0.75), -0.7071067811865475244)
    assert ae(fp.cospi(1), -1.0)
    assert ae(fp.cospi(1.25), -0.7071067811865475244)
    assert ae(fp.cospi(1.5), 0.0)
    assert ae(fp.cospi(1.75), 0.7071067811865475244)
    assert ae(fp.cospi(2), 1.0)
    assert ae(fp.cospi(2.25), 0.7071067811865475244)
    assert ae(fp.cospi(0+3j), (6195.8239443081075259 + 0.0j))
    assert ae(fp.cospi(0.25+3j), (4381.1091260582448033 - 4381.1090689950686908j))
    assert ae(fp.cospi(0.5+3j), (0.0 - 6195.8238636085899556j))
    assert ae(fp.cospi(0.75+3j), (-4381.1091260582448033 - 4381.1090689950686908j))
    assert ae(fp.cospi(1+3j), (-6195.8239443081075259 + 0.0j))
    assert ae(fp.cospi(1.25+3j), (-4381.1091260582448033 + 4381.1090689950686908j))
    assert ae(fp.cospi(1.5+3j), (0.0 + 6195.8238636085899556j))
    assert ae(fp.cospi(1.75+3j), (4381.1091260582448033 + 4381.1090689950686908j))
    assert ae(fp.cospi(2+3j), (6195.8239443081075259 + 0.0j))
    assert ae(fp.cospi(2.25+3j), (4381.1091260582448033 - 4381.1090689950686908j))
    assert ae(fp.cospi(-0.75), -0.7071067811865475244)
    assert ae(fp.sinpi(-0.7), -0.80901699437494750611)
    assert ae(fp.cospi(-0.7), -0.5877852522924730163)
    assert ae(fp.cospi(-3+2j), (-267.74676148374822225 + 0.0j))
    assert ae(fp.sinpi(-3+2j), (0.0 - 267.74489404101651426j))
    assert ae(fp.sinpi(-0.7+2j), (-216.6116802292079471 - 157.37650009392034693j))
    assert ae(fp.cospi(-0.7+2j), (-157.37759774921754565 + 216.61016943630197336j))

def test_fp_expj():
    assert ae(fp.expj(0), (1.0 + 0.0j))
    assert ae(fp.expj(1), (0.5403023058681397174 + 0.84147098480789650665j))
    assert ae(fp.expj(2), (-0.416146836547142387 + 0.9092974268256816954j))
    assert ae(fp.expj(0.75), (0.73168886887382088631 + 0.68163876002333416673j))
    assert ae(fp.expj(2+3j), (-0.020718731002242879378 + 0.045271253156092975488j))
    assert ae(fp.expjpi(0), (1.0 + 0.0j))
    assert ae(fp.expjpi(1), (-1.0 + 0.0j))
    assert ae(fp.expjpi(2), (1.0 + 0.0j))
    assert ae(fp.expjpi(0.75), (-0.7071067811865475244 + 0.7071067811865475244j))
    assert ae(fp.expjpi(2+3j), (0.000080699517570304599239 + 0.0j))

def test_fp_bernoulli():
    assert ae(fp.bernoulli(0), 1.0)
    assert ae(fp.bernoulli(1), -0.5)
    assert ae(fp.bernoulli(2), 0.16666666666666666667)
    assert ae(fp.bernoulli(10), 0.075757575757575757576)
    assert ae(fp.bernoulli(11), 0.0)

def test_fp_gamma():
    assert ae(fp.gamma(1), 1.0)
    assert ae(fp.gamma(1.5), 0.88622692545275801365)
    assert ae(fp.gamma(10), 362880.0)
    assert ae(fp.gamma(-0.5), -3.5449077018110320546)
    assert ae(fp.gamma(-7.1), 0.0016478244570263333622)
    assert ae(fp.gamma(12.3), 83385367.899970000963)
    assert ae(fp.gamma(2+0j), (1.0 + 0.0j))
    assert ae(fp.gamma(-2.5+0j), (-0.94530872048294188123 + 0.0j))
    assert ae(fp.gamma(3+4j), (0.0052255384713692141947 - 0.17254707929430018772j))
    assert ae(fp.gamma(-3-4j), (0.00001460997305874775607 - 0.000020760733311509070396j))
    assert ae(fp.fac(0), 1.0)
    assert ae(fp.fac(1), 1.0)
    assert ae(fp.fac(20), 2432902008176640000.0)
    assert ae(fp.fac(-3.5), -0.94530872048294188123)
    assert ae(fp.fac(2+3j), (-0.44011340763700171113 - 0.06363724312631702183j))
    assert ae(fp.loggamma(1.0), 0.0)
    assert ae(fp.loggamma(2.0), 0.0)
    assert ae(fp.loggamma(3.0), 0.69314718055994530942)
    assert ae(fp.loggamma(7.25), 7.0521854507385394449)
    assert ae(fp.loggamma(1000.0), 5905.2204232091812118)
    assert ae(fp.loggamma(1e50), 1.1412925464970229298e+52)
    assert ae(fp.loggamma(1e25+1e25j), (5.6125802751733671621e+26 + 5.7696599078528568383e+26j))
    assert ae(fp.loggamma(3+4j), (-1.7566267846037841105 + 4.7426644380346579282j))
    assert ae(fp.loggamma(-0.5), (1.2655121234846453965 - 3.1415926535897932385j))
    assert ae(fp.loggamma(-1.25), (1.3664317612369762346 - 6.2831853071795864769j))
    assert ae(fp.loggamma(-2.75), (0.0044878975359557733115 - 9.4247779607693797154j))
    assert ae(fp.loggamma(-3.5), (-1.3090066849930420464 - 12.566370614359172954j))
    assert ae(fp.loggamma(-4.5), (-2.8130840817693161197 - 15.707963267948966192j))
    assert ae(fp.loggamma(-2+3j), (-6.776523813485657093 - 4.568791367260286402j))
    assert ae(fp.loggamma(-1000.3), (-5912.8440347785205041 - 3144.7342462433830317j))
    assert ae(fp.loggamma(-100-100j), (-632.35117666833135562 - 158.37641469650352462j))
    assert ae(fp.loggamma(1e-10), 23.025850929882735237)
    assert ae(fp.loggamma(-1e-10), (23.02585092999817837 - 3.1415926535897932385j))
    assert ae(fp.loggamma(1e-10j), (23.025850929940456804 - 1.5707963268526181857j))
    assert ae(fp.loggamma(1e-10j-1e-10), (22.679277339718205716 - 2.3561944902500664954j))

def test_fp_psi():
    assert ae(fp.psi(0, 3.7), 1.1671535393615114409)
    assert ae(fp.psi(0, 0.5), -1.9635100260214234794)
    assert ae(fp.psi(0, 1), -0.57721566490153286061)
    assert ae(fp.psi(0, -2.5), 1.1031566406452431872)
    assert ae(fp.psi(0, 12.9), 2.5179671503279156347)
    assert ae(fp.psi(0, 100), 4.6001618527380874002)
    assert ae(fp.psi(0, 2500.3), 7.8239660143238547877)
    assert ae(fp.psi(0, 1e40), 92.103403719761827391)
    assert ae(fp.psi(0, 1e200), 460.51701859880913677)
    assert ae(fp.psi(0, 3.7+0j), (1.1671535393615114409 + 0.0j))
    assert ae(fp.psi(1, 3), 0.39493406684822643647)
    assert ae(fp.psi(3, 2+3j), (-0.05383196209159972116 + 0.0076890935247364805218j))
    assert ae(fp.psi(4, -0.5+1j), (1.2719531355492328195 - 18.211833410936276774j))
    assert ae(fp.harmonic(0), 0.0)
    assert ae(fp.harmonic(1), 1.0)
    assert ae(fp.harmonic(2), 1.5)
    assert ae(fp.harmonic(100), 5.1873775176396202608)
    assert ae(fp.harmonic(-2.5), 1.2803723055467760478)
    assert ae(fp.harmonic(2+3j), (1.9390425294578375875 + 0.87336044981834544043j))
    assert ae(fp.harmonic(-5-4j), (2.3725754822349437733 - 2.4160904444801621j))

def test_fp_zeta():
    assert ae(fp.zeta(1e100), 1.0)
    assert ae(fp.zeta(3), 1.2020569031595942854)
    assert ae(fp.zeta(2+0j), (1.6449340668482264365 + 0.0j))
    assert ae(fp.zeta(0.93), -13.713619351638164784)
    assert ae(fp.zeta(1.74), 1.9796863545771774095)
    assert ae(fp.zeta(0.0), -0.5)
    assert ae(fp.zeta(-1.0), -0.083333333333333333333)
    assert ae(fp.zeta(-2.0), 0.0)
    assert ae(fp.zeta(-3.0), 0.0083333333333333333333)
    assert ae(fp.zeta(-500.0), 0.0)
    assert ae(fp.zeta(-7.4), 0.0036537321227995882447)
    assert ae(fp.zeta(2.1), 1.5602165335033620158)
    assert ae(fp.zeta(26.9), 1.0000000079854809935)
    assert ae(fp.zeta(26), 1.0000000149015548284)
    assert ae(fp.zeta(27), 1.0000000074507117898)
    assert ae(fp.zeta(28), 1.0000000037253340248)
    assert ae(fp.zeta(27.1), 1.000000006951755045)
    assert ae(fp.zeta(32.7), 1.0000000001433243232)
    assert ae(fp.zeta(100), 1.0)
    assert ae(fp.altzeta(3.5), 0.92755357777394803511)
    assert ae(fp.altzeta(1), 0.69314718055994530942)
    assert ae(fp.altzeta(2), 0.82246703342411321824)
    assert ae(fp.altzeta(0), 0.5)
    assert ae(fp.zeta(-2+3j, 1), (0.13297115587929864827 + 0.12305330040458776494j))
    assert ae(fp.zeta(-2+3j, 5), (18.384866151867576927 - 11.377015110597711009j))
    assert ae(fp.zeta(1.0000000001), 9999999173.1735741337)
    assert ae(fp.zeta(0.9999999999), -9999999172.0191428039)
    assert ae(fp.zeta(1+0.000000001j), (0.57721566490153286061 - 999999999.99999993765j))
    assert ae(fp.primezeta(2.5+4j), (-0.16922458243438033385 - 0.010847965298387727811j))
    assert ae(fp.primezeta(4), 0.076993139764246844943)
    assert ae(fp.riemannr(3.7), 2.3034079839110855717)
    assert ae(fp.riemannr(8), 3.9011860449341499474)
    assert ae(fp.riemannr(3+4j), (2.2369653314259991796 + 1.6339943856990281694j))

def test_fp_hyp2f1():
    assert ae(fp.hyp2f1(1, (3,2), 3.25, 5.0), (-0.46600275923108143059 - 0.74393667908854842325j))
    assert ae(fp.hyp2f1(1+1j, (3,2), 3.25, 5.0), (-5.9208875603806515987 - 2.3813557707889590686j))
    assert ae(fp.hyp2f1(1+1j, (3,2), 3.25, 2+3j), (0.17174552030925080445 + 0.19589781970539389999j))

def test_fp_erf():
    assert fp.erf(2) == fp.erf(2.0) == fp.erf(2.0+0.0j)
    assert fp.erf(fp.inf) == 1.0
    assert fp.erf(fp.ninf) == -1.0
    assert ae(fp.erf(0), 0.0)
    assert ae(fp.erf(-0), -0.0)
    assert ae(fp.erf(0.3), 0.32862675945912741619)
    assert ae(fp.erf(-0.3), -0.32862675945912741619)
    assert ae(fp.erf(0.9), 0.79690821242283213966)
    assert ae(fp.erf(-0.9), -0.79690821242283213966)
    assert ae(fp.erf(1.0), 0.84270079294971486934)
    assert ae(fp.erf(-1.0), -0.84270079294971486934)
    assert ae(fp.erf(1.1), 0.88020506957408172966)
    assert ae(fp.erf(-1.1), -0.88020506957408172966)
    assert ae(fp.erf(8.5), 1.0)
    assert ae(fp.erf(-8.5), -1.0)
    assert ae(fp.erf(9.1), 1.0)
    assert ae(fp.erf(-9.1), -1.0)
    assert ae(fp.erf(20.0), 1.0)
    assert ae(fp.erf(-20.0), -1.0)
    assert ae(fp.erf(10000.0), 1.0)
    assert ae(fp.erf(-10000.0), -1.0)
    assert ae(fp.erf(1e+50), 1.0)
    assert ae(fp.erf(-1e+50), -1.0)
    assert ae(fp.erf(1j), 1.650425758797542876j)
    assert ae(fp.erf(-1j), -1.650425758797542876j)
    assert ae(fp.erf((2+3j)), (-20.829461427614568389 + 8.6873182714701631444j))
    assert ae(fp.erf(-(2+3j)), -(-20.829461427614568389 + 8.6873182714701631444j))
    assert ae(fp.erf((8+9j)), (-1072004.2525062051158 + 364149.91954310255423j))
    assert ae(fp.erf(-(8+9j)), -(-1072004.2525062051158 + 364149.91954310255423j))
    assert fp.erfc(fp.inf) == 0.0
    assert fp.erfc(fp.ninf) == 2.0
    assert fp.erfc(0) == 1
    assert fp.erfc(-0.0) == 1
    assert fp.erfc(0+0j) == 1
    assert ae(fp.erfc(0.3), 0.67137324054087258381)
    assert ae(fp.erfc(-0.3), 1.3286267594591274162)
    assert ae(fp.erfc(0.9), 0.20309178757716786034)
    assert ae(fp.erfc(-0.9), 1.7969082124228321397)
    assert ae(fp.erfc(1.0), 0.15729920705028513066)
    assert ae(fp.erfc(-1.0), 1.8427007929497148693)
    assert ae(fp.erfc(1.1), 0.11979493042591827034)
    assert ae(fp.erfc(-1.1), 1.8802050695740817297)
    assert ae(fp.erfc(8.5), 2.7623240713337714461e-33)
    assert ae(fp.erfc(-8.5), 2.0)
    assert ae(fp.erfc(9.1), 6.6969004279886077452e-38)
    assert ae(fp.erfc(-9.1), 2.0)
    assert ae(fp.erfc(20.0), 5.3958656116079009289e-176)
    assert ae(fp.erfc(-20.0), 2.0)
    assert ae(fp.erfc(10000.0), 0.0)
    assert ae(fp.erfc(-10000.0), 2.0)
    assert ae(fp.erfc(1e+50), 0.0)
    assert ae(fp.erfc(-1e+50), 2.0)
    assert ae(fp.erfc(1j), (1.0 - 1.650425758797542876j))
    assert ae(fp.erfc(-1j), (1.0 + 1.650425758797542876j))
    assert ae(fp.erfc((2+3j)), (21.829461427614568389 - 8.6873182714701631444j), 1e-13)
    assert ae(fp.erfc(-(2+3j)), (-19.829461427614568389 + 8.6873182714701631444j), 1e-13)
    assert ae(fp.erfc((8+9j)), (1072005.2525062051158 - 364149.91954310255423j))
    assert ae(fp.erfc(-(8+9j)), (-1072003.2525062051158 + 364149.91954310255423j))
    assert ae(fp.erfc(20+0j), (5.3958656116079009289e-176 + 0.0j))

def test_fp_lambertw():
    assert ae(fp.lambertw(0.0), 0.0)
    assert ae(fp.lambertw(1.0), 0.567143290409783873)
    assert ae(fp.lambertw(7.5), 1.5662309537823875394)
    assert ae(fp.lambertw(-0.25), -0.35740295618138890307)
    assert ae(fp.lambertw(-10.0), (1.3699809685212708156 + 2.140194527074713196j))
    assert ae(fp.lambertw(0+0j), (0.0 + 0.0j))
    assert ae(fp.lambertw(4+0j), (1.2021678731970429392 + 0.0j))
    assert ae(fp.lambertw(1000.5), 5.2500227450408980127)
    assert ae(fp.lambertw(1e100), 224.84310644511850156)
    assert ae(fp.lambertw(-1000.0), (5.1501630246362515223 + 2.6641981432905204596j))
    assert ae(fp.lambertw(1e-10), 9.9999999990000003645e-11)
    assert ae(fp.lambertw(1e-10j), (1.0000000000000000728e-20 + 1.0000000000000000364e-10j))
    assert ae(fp.lambertw(3+4j), (1.2815618061237758782 + 0.53309522202097107131j))
    assert ae(fp.lambertw(-3-4j), (1.0750730665692549276 - 1.3251023817343588823j))
    assert ae(fp.lambertw(10000+1000j), (7.2361526563371602186 + 0.087567810943839352034j))
    assert ae(fp.lambertw(0.0, -1), -fp.inf)
    assert ae(fp.lambertw(1.0, -1), (-1.5339133197935745079 - 4.3751851530618983855j))
    assert ae(fp.lambertw(7.5, -1), (0.44125668415098614999 - 4.8039842008452390179j))
    assert ae(fp.lambertw(-0.25, -1), -2.1532923641103496492)
    assert ae(fp.lambertw(-10.0, -1), (1.3699809685212708156 - 2.140194527074713196j))
    assert ae(fp.lambertw(0+0j, -1), -fp.inf)
    assert ae(fp.lambertw(4+0j, -1), (-0.15730793189620765317 - 4.6787800704666656212j))
    assert ae(fp.lambertw(1000.5, -1), (4.9153765415404024736 - 5.4465682700815159569j))
    assert ae(fp.lambertw(1e100, -1), (224.84272130101601052 - 6.2553713838167244141j))
    assert ae(fp.lambertw(-1000.0, -1), (5.1501630246362515223 - 2.6641981432905204596j))
    assert ae(fp.lambertw(1e-10, -1), (-26.303186778379041521 - 3.2650939117038283975j))
    assert ae(fp.lambertw(1e-10j, -1), (-26.297238779529035028 - 1.6328071613455765135j))
    assert ae(fp.lambertw(3+4j, -1), (0.25856740686699741676 - 3.8521166861614355895j))
    assert ae(fp.lambertw(-3-4j, -1), (-0.32028750204310768396 - 6.8801677192091972343j))
    assert ae(fp.lambertw(10000+1000j, -1), (7.0255308742285435567 - 5.5177506835734067601j))
    assert ae(fp.lambertw(0.0, 2), -fp.inf)
    assert ae(fp.lambertw(1.0, 2), (-2.4015851048680028842 + 10.776299516115070898j))
    assert ae(fp.lambertw(7.5, 2), (-0.38003357962843791529 + 10.960916473368746184j))
    assert ae(fp.lambertw(-0.25, 2), (-4.0558735269061511898 + 13.852334658567271386j))
    assert ae(fp.lambertw(-10.0, 2), (-0.34479123764318858696 + 14.112740596763592363j))
    assert ae(fp.lambertw(0+0j, 2), -fp.inf)
    assert ae(fp.lambertw(4+0j, 2), (-1.0070343323804262788 + 10.903476551861683082j))
    assert ae(fp.lambertw(1000.5, 2), (4.4076185165459395295 + 11.365524591091402177j))
    assert ae(fp.lambertw(1e100, 2), (224.84156762724875878 + 12.510785262632255672j))
    assert ae(fp.lambertw(-1000.0, 2), (4.1984245610246530756 + 14.420478573754313845j))
    assert ae(fp.lambertw(1e-10, 2), (-26.362258095445866488 + 9.7800247407031482519j))
    assert ae(fp.lambertw(1e-10j, 2), (-26.384250801683084252 + 11.403535950607739763j))
    assert ae(fp.lambertw(3+4j, 2), (-0.86554679943333993562 + 11.849956798331992027j))
    assert ae(fp.lambertw(-3-4j, 2), (-0.55792273874679112639 + 8.7173627024159324811j))
    assert ae(fp.lambertw(10000+1000j, 2), (6.6223802254585662734 + 11.61348646825020766j))

def test_fp_stress_ei_e1():
    # Can be tightened on recent Pythons with more accurate math/cmath
    ATOL = 1e-13
    PTOL = 1e-12
    v = fp.e1(1.1641532182693481445e-10)
    assert ae(v, 22.296641293693077672, tol=ATOL)
    assert type(v) is float
    v = fp.e1(0.25)
    assert ae(v, 1.0442826344437381945, tol=ATOL)
    assert type(v) is float
    v = fp.e1(1.0)
    assert ae(v, 0.21938393439552027368, tol=ATOL)
    assert type(v) is float
    v = fp.e1(2.0)
    assert ae(v, 0.048900510708061119567, tol=ATOL)
    assert type(v) is float
    v = fp.e1(5.0)
    assert ae(v, 0.0011482955912753257973, tol=ATOL)
    assert type(v) is float
    v = fp.e1(20.0)
    assert ae(v, 9.8355252906498816904e-11, tol=ATOL)
    assert type(v) is float
    v = fp.e1(30.0)
    assert ae(v, 3.0215520106888125448e-15, tol=ATOL)
    assert type(v) is float
    v = fp.e1(40.0)
    assert ae(v, 1.0367732614516569722e-19, tol=ATOL)
    assert type(v) is float
    v = fp.e1(50.0)
    assert ae(v, 3.7832640295504590187e-24, tol=ATOL)
    assert type(v) is float
    v = fp.e1(80.0)
    assert ae(v, 2.2285432586884729112e-37, tol=ATOL)
    assert type(v) is float
    v = fp.e1((1.1641532182693481445e-10 + 0.0j))
    assert ae(v, (22.296641293693077672 + 0.0j), tol=ATOL)
    assert ae(v.real, 22.296641293693077672, tol=PTOL)
    assert v.imag == 0
    v = fp.e1((0.25 + 0.0j))
    assert ae(v, (1.0442826344437381945 + 0.0j), tol=ATOL)
    assert ae(v.real, 1.0442826344437381945, tol=PTOL)
    assert v.imag == 0
    v = fp.e1((1.0 + 0.0j))
    assert ae(v, (0.21938393439552027368 + 0.0j), tol=ATOL)
    assert ae(v.real, 0.21938393439552027368, tol=PTOL)
    assert v.imag == 0
    v = fp.e1((2.0 + 0.0j))
    assert ae(v, (0.048900510708061119567 + 0.0j), tol=ATOL)
    assert ae(v.real, 0.048900510708061119567, tol=PTOL)
    assert v.imag == 0
    v = fp.e1((5.0 + 0.0j))
    assert ae(v, (0.0011482955912753257973 + 0.0j), tol=ATOL)
    assert ae(v.real, 0.0011482955912753257973, tol=PTOL)
    assert v.imag == 0
    v = fp.e1((20.0 + 0.0j))
    assert ae(v, (9.8355252906498816904e-11 + 0.0j), tol=ATOL)
    assert ae(v.real, 9.8355252906498816904e-11, tol=PTOL)
    assert v.imag == 0
    v = fp.e1((30.0 + 0.0j))
    assert ae(v, (3.0215520106888125448e-15 + 0.0j), tol=ATOL)
    assert ae(v.real, 3.0215520106888125448e-15, tol=PTOL)
    assert v.imag == 0
    v = fp.e1((40.0 + 0.0j))
    assert ae(v, (1.0367732614516569722e-19 + 0.0j), tol=ATOL)
    assert ae(v.real, 1.0367732614516569722e-19, tol=PTOL)
    assert v.imag == 0
    v = fp.e1((50.0 + 0.0j))
    assert ae(v, (3.7832640295504590187e-24 + 0.0j), tol=ATOL)
    assert ae(v.real, 3.7832640295504590187e-24, tol=PTOL)
    assert v.imag == 0
    v = fp.e1((80.0 + 0.0j))
    assert ae(v, (2.2285432586884729112e-37 + 0.0j), tol=ATOL)
    assert ae(v.real, 2.2285432586884729112e-37, tol=PTOL)
    assert v.imag == 0
    v = fp.e1((4.6566128730773925781e-10 + 1.1641532182693481445e-10j))
    assert ae(v, (20.880034622014215597 - 0.24497866301044883237j), tol=ATOL)
    assert ae(v.real, 20.880034622014215597, tol=PTOL)
    assert ae(v.imag, -0.24497866301044883237, tol=PTOL)
    v = fp.e1((1.0 + 0.25j))
    assert ae(v, (0.19731063945004229095 - 0.087366045774299963672j), tol=ATOL)
    assert ae(v.real, 0.19731063945004229095, tol=PTOL)
    assert ae(v.imag, -0.087366045774299963672, tol=PTOL)
    v = fp.e1((4.0 + 1.0j))
    assert ae(v, (0.0013106173980145506944 - 0.0034542480199350626699j), tol=ATOL)
    assert ae(v.real, 0.0013106173980145506944, tol=PTOL)
    assert ae(v.imag, -0.0034542480199350626699, tol=PTOL)
    v = fp.e1((8.0 + 2.0j))
    assert ae(v, (-0.000022278049065270225945 - 0.000029191940456521555288j), tol=ATOL)
    assert ae(v.real, -0.000022278049065270225945, tol=PTOL)
    assert ae(v.imag, -0.000029191940456521555288, tol=PTOL)
    v = fp.e1((20.0 + 5.0j))
    assert ae(v, (4.7711374515765346894e-11 + 8.2902652405126947359e-11j), tol=ATOL)
    assert ae(v.real, 4.7711374515765346894e-11, tol=PTOL)
    assert ae(v.imag, 8.2902652405126947359e-11, tol=PTOL)
    v = fp.e1((80.0 + 20.0j))
    assert ae(v, (3.8353473865788235787e-38 - 2.129247592349605139e-37j), tol=ATOL)
    assert ae(v.real, 3.8353473865788235787e-38, tol=PTOL)
    assert ae(v.imag, -2.129247592349605139e-37, tol=PTOL)
    v = fp.e1((120.0 + 30.0j))
    assert ae(v, (2.3836002337480334716e-55 + 5.6704043587126198306e-55j), tol=ATOL)
    assert ae(v.real, 2.3836002337480334716e-55, tol=PTOL)
    assert ae(v.imag, 5.6704043587126198306e-55, tol=PTOL)
    v = fp.e1((160.0 + 40.0j))
    assert ae(v, (-1.6238022898654510661e-72 - 1.104172355572287367e-72j), tol=ATOL)
    assert ae(v.real, -1.6238022898654510661e-72, tol=PTOL)
    assert ae(v.imag, -1.104172355572287367e-72, tol=PTOL)
    v = fp.e1((200.0 + 50.0j))
    assert ae(v, (6.6800061461666228487e-90 + 1.4473816083541016115e-91j), tol=ATOL)
    assert ae(v.real, 6.6800061461666228487e-90, tol=PTOL)
    assert ae(v.imag, 1.4473816083541016115e-91, tol=PTOL)
    v = fp.e1((320.0 + 80.0j))
    assert ae(v, (4.2737871527778786157e-143 + 3.1789935525785660314e-142j), tol=ATOL)
    assert ae(v.real, 4.2737871527778786157e-143, tol=PTOL)
    assert ae(v.imag, 3.1789935525785660314e-142, tol=PTOL)
    v = fp.e1((1.1641532182693481445e-10 + 1.1641532182693481445e-10j))
    assert ae(v, (21.950067703413105017 - 0.7853981632810329878j), tol=ATOL)
    assert ae(v.real, 21.950067703413105017, tol=PTOL)
    assert ae(v.imag, -0.7853981632810329878, tol=PTOL)
    v = fp.e1((0.25 + 0.25j))
    assert ae(v, (0.71092525792923287894 - 0.56491812441304194711j), tol=ATOL)
    assert ae(v.real, 0.71092525792923287894, tol=PTOL)
    assert ae(v.imag, -0.56491812441304194711, tol=PTOL)
    v = fp.e1((1.0 + 1.0j))
    assert ae(v, (0.00028162445198141832551 - 0.17932453503935894015j), tol=ATOL)
    assert ae(v.real, 0.00028162445198141832551, tol=PTOL)
    assert ae(v.imag, -0.17932453503935894015, tol=PTOL)
    v = fp.e1((2.0 + 2.0j))
    assert ae(v, (-0.033767089606562004246 - 0.018599414169750541925j), tol=ATOL)
    assert ae(v.real, -0.033767089606562004246, tol=PTOL)
    assert ae(v.imag, -0.018599414169750541925, tol=PTOL)
    v = fp.e1((5.0 + 5.0j))
    assert ae(v, (0.0007266506660356393891 + 0.00047102780163522245054j), tol=ATOL)
    assert ae(v.real, 0.0007266506660356393891, tol=PTOL)
    assert ae(v.imag, 0.00047102780163522245054, tol=PTOL)
    v = fp.e1((20.0 + 20.0j))
    assert ae(v, (-2.3824537449367396579e-11 - 6.6969873156525615158e-11j), tol=ATOL)
    assert ae(v.real, -2.3824537449367396579e-11, tol=PTOL)
    assert ae(v.imag, -6.6969873156525615158e-11, tol=PTOL)
    v = fp.e1((30.0 + 30.0j))
    assert ae(v, (1.7316045841744061617e-15 + 1.3065678019487308689e-15j), tol=ATOL)
    assert ae(v.real, 1.7316045841744061617e-15, tol=PTOL)
    assert ae(v.imag, 1.3065678019487308689e-15, tol=PTOL)
    v = fp.e1((40.0 + 40.0j))
    assert ae(v, (-7.4001043002899232182e-20 - 4.991847855336816304e-21j), tol=ATOL)
    assert ae(v.real, -7.4001043002899232182e-20, tol=PTOL)
    assert ae(v.imag, -4.991847855336816304e-21, tol=PTOL)
    v = fp.e1((50.0 + 50.0j))
    assert ae(v, (2.3566128324644641219e-24 - 1.3188326726201614778e-24j), tol=ATOL)
    assert ae(v.real, 2.3566128324644641219e-24, tol=PTOL)
    assert ae(v.imag, -1.3188326726201614778e-24, tol=PTOL)
    v = fp.e1((80.0 + 80.0j))
    assert ae(v, (9.8279750572186526673e-38 + 1.243952841288868831e-37j), tol=ATOL)
    assert ae(v.real, 9.8279750572186526673e-38, tol=PTOL)
    assert ae(v.imag, 1.243952841288868831e-37, tol=PTOL)
    v = fp.e1((1.1641532182693481445e-10 + 4.6566128730773925781e-10j))
    assert ae(v, (20.880034621664969632 - 1.3258176632023711778j), tol=ATOL)
    assert ae(v.real, 20.880034621664969632, tol=PTOL)
    assert ae(v.imag, -1.3258176632023711778, tol=PTOL)
    v = fp.e1((0.25 + 1.0j))
    assert ae(v, (-0.16868306393667788761 - 0.4858011885947426971j), tol=ATOL)
    assert ae(v.real, -0.16868306393667788761, tol=PTOL)
    assert ae(v.imag, -0.4858011885947426971, tol=PTOL)
    v = fp.e1((1.0 + 4.0j))
    assert ae(v, (0.03373591813926547318 + 0.073523452241083821877j), tol=ATOL)
    assert ae(v.real, 0.03373591813926547318, tol=PTOL)
    assert ae(v.imag, 0.073523452241083821877, tol=PTOL)
    v = fp.e1((2.0 + 8.0j))
    assert ae(v, (-0.015392833434733785143 - 0.0031747121557605415914j), tol=ATOL)
    assert ae(v.real, -0.015392833434733785143, tol=PTOL)
    assert ae(v.imag, -0.0031747121557605415914, tol=PTOL)
    v = fp.e1((5.0 + 20.0j))
    assert ae(v, (-0.00024419662286542966525 - 0.00021008322966152755674j), tol=ATOL)
    assert ae(v.real, -0.00024419662286542966525, tol=PTOL)
    assert ae(v.imag, -0.00021008322966152755674, tol=PTOL)
    v = fp.e1((20.0 + 80.0j))
    assert ae(v, (2.3255552781051330088e-11 + 8.9463918891349438007e-12j), tol=ATOL)
    assert ae(v.real, 2.3255552781051330088e-11, tol=PTOL)
    assert ae(v.imag, 8.9463918891349438007e-12, tol=PTOL)
    v = fp.e1((30.0 + 120.0j))
    assert ae(v, (-2.7068919097124652332e-16 - 7.0477762411705130239e-16j), tol=ATOL)
    assert ae(v.real, -2.7068919097124652332e-16, tol=PTOL)
    assert ae(v.imag, -7.0477762411705130239e-16, tol=PTOL)
    v = fp.e1((40.0 + 160.0j))
    assert ae(v, (-1.1695597827678024687e-20 + 2.2907401455645736661e-20j), tol=ATOL)
    assert ae(v.real, -1.1695597827678024687e-20, tol=PTOL)
    assert ae(v.imag, 2.2907401455645736661e-20, tol=PTOL)
    v = fp.e1((50.0 + 200.0j))
    assert ae(v, (9.0323746914410162531e-25 - 2.3950601790033530935e-25j), tol=ATOL)
    assert ae(v.real, 9.0323746914410162531e-25, tol=PTOL)
    assert ae(v.imag, -2.3950601790033530935e-25, tol=PTOL)
    v = fp.e1((80.0 + 320.0j))
    assert ae(v, (3.4819106748728063576e-38 - 4.215653005615772724e-38j), tol=ATOL)
    assert ae(v.real, 3.4819106748728063576e-38, tol=PTOL)
    assert ae(v.imag, -4.215653005615772724e-38, tol=PTOL)
    v = fp.e1((0.0 + 1.1641532182693481445e-10j))
    assert ae(v, (22.29664129357666235 - 1.5707963266784812974j), tol=ATOL)
    assert ae(v.real, 22.29664129357666235, tol=PTOL)
    assert ae(v.imag, -1.5707963266784812974, tol=PTOL)
    v = fp.e1((0.0 + 0.25j))
    assert ae(v, (0.82466306258094565309 - 1.3216627564751394551j), tol=ATOL)
    assert ae(v.real, 0.82466306258094565309, tol=PTOL)
    assert ae(v.imag, -1.3216627564751394551, tol=PTOL)
    v = fp.e1((0.0 + 1.0j))
    assert ae(v, (-0.33740392290096813466 - 0.62471325642771360429j), tol=ATOL)
    assert ae(v.real, -0.33740392290096813466, tol=PTOL)
    assert ae(v.imag, -0.62471325642771360429, tol=PTOL)
    v = fp.e1((0.0 + 2.0j))
    assert ae(v, (-0.4229808287748649957 + 0.034616650007798229345j), tol=ATOL)
    assert ae(v.real, -0.4229808287748649957, tol=PTOL)
    assert ae(v.imag, 0.034616650007798229345, tol=PTOL)
    v = fp.e1((0.0 + 5.0j))
    assert ae(v, (0.19002974965664387862 - 0.020865081850222481957j), tol=ATOL)
    assert ae(v.real, 0.19002974965664387862, tol=PTOL)
    assert ae(v.imag, -0.020865081850222481957, tol=PTOL)
    v = fp.e1((0.0 + 20.0j))
    assert ae(v, (-0.04441982084535331654 - 0.022554625751456779068j), tol=ATOL)
    assert ae(v.real, -0.04441982084535331654, tol=PTOL)
    assert ae(v.imag, -0.022554625751456779068, tol=PTOL)
    v = fp.e1((0.0 + 30.0j))
    assert ae(v, (0.033032417282071143779 - 0.0040397867645455082476j), tol=ATOL)
    assert ae(v.real, 0.033032417282071143779, tol=PTOL)
    assert ae(v.imag, -0.0040397867645455082476, tol=PTOL)
    v = fp.e1((0.0 + 40.0j))
    assert ae(v, (-0.019020007896208766962 + 0.016188792559887887544j), tol=ATOL)
    assert ae(v.real, -0.019020007896208766962, tol=PTOL)
    assert ae(v.imag, 0.016188792559887887544, tol=PTOL)
    v = fp.e1((0.0 + 50.0j))
    assert ae(v, (0.0056283863241163054402 - 0.019179254308960724503j), tol=ATOL)
    assert ae(v.real, 0.0056283863241163054402, tol=PTOL)
    assert ae(v.imag, -0.019179254308960724503, tol=PTOL)
    v = fp.e1((0.0 + 80.0j))
    assert ae(v, (0.012402501155070958192 + 0.0015345601175906961199j), tol=ATOL)
    assert ae(v.real, 0.012402501155070958192, tol=PTOL)
    assert ae(v.imag, 0.0015345601175906961199, tol=PTOL)
    v = fp.e1((-1.1641532182693481445e-10 + 4.6566128730773925781e-10j))
    assert ae(v, (20.880034621432138988 - 1.8157749894560994861j), tol=ATOL)
    assert ae(v.real, 20.880034621432138988, tol=PTOL)
    assert ae(v.imag, -1.8157749894560994861, tol=PTOL)
    v = fp.e1((-0.25 + 1.0j))
    assert ae(v, (-0.59066621214766308594 - 0.74474454765205036972j), tol=ATOL)
    assert ae(v.real, -0.59066621214766308594, tol=PTOL)
    assert ae(v.imag, -0.74474454765205036972, tol=PTOL)
    v = fp.e1((-1.0 + 4.0j))
    assert ae(v, (0.49739047283060471093 + 0.41543605404038863174j), tol=ATOL)
    assert ae(v.real, 0.49739047283060471093, tol=PTOL)
    assert ae(v.imag, 0.41543605404038863174, tol=PTOL)
    v = fp.e1((-2.0 + 8.0j))
    assert ae(v, (-0.8705211147733730969 + 0.24099328498605539667j), tol=ATOL)
    assert ae(v.real, -0.8705211147733730969, tol=PTOL)
    assert ae(v.imag, 0.24099328498605539667, tol=PTOL)
    v = fp.e1((-5.0 + 20.0j))
    assert ae(v, (-7.0789514293925893007 - 1.6102177171960790536j), tol=ATOL)
    assert ae(v.real, -7.0789514293925893007, tol=PTOL)
    assert ae(v.imag, -1.6102177171960790536, tol=PTOL)
    v = fp.e1((-20.0 + 80.0j))
    assert ae(v, (5855431.4907298084434 - 720920.93315409165707j), tol=ATOL)
    assert ae(v.real, 5855431.4907298084434, tol=PTOL)
    assert ae(v.imag, -720920.93315409165707, tol=PTOL)
    v = fp.e1((-30.0 + 120.0j))
    assert ae(v, (-65402491644.703470747 - 56697658399.657460294j), tol=ATOL)
    assert ae(v.real, -65402491644.703470747, tol=PTOL)
    assert ae(v.imag, -56697658399.657460294, tol=PTOL)
    v = fp.e1((-40.0 + 160.0j))
    assert ae(v, (25504929379604.776769 + 1429035198630573.2463j), tol=ATOL)
    assert ae(v.real, 25504929379604.776769, tol=PTOL)
    assert ae(v.imag, 1429035198630573.2463, tol=PTOL)
    v = fp.e1((-50.0 + 200.0j))
    assert ae(v, (18437746526988116954.0 - 17146362239046152345.0j), tol=ATOL)
    assert ae(v.real, 18437746526988116954.0, tol=PTOL)
    assert ae(v.imag, -17146362239046152345.0, tol=PTOL)
    v = fp.e1((-80.0 + 320.0j))
    assert ae(v, (3.3464697299634526706e+31 - 1.6473152633843023919e+32j), tol=ATOL)
    assert ae(v.real, 3.3464697299634526706e+31, tol=PTOL)
    assert ae(v.imag, -1.6473152633843023919e+32, tol=PTOL)
    v = fp.e1((-4.6566128730773925781e-10 + 1.1641532182693481445e-10j))
    assert ae(v, (20.880034621082893023 - 2.8966139903465137624j), tol=ATOL)
    assert ae(v.real, 20.880034621082893023, tol=PTOL)
    assert ae(v.imag, -2.8966139903465137624, tol=PTOL)
    v = fp.e1((-1.0 + 0.25j))
    assert ae(v, (-1.8942716983721074932 - 2.4689102827070540799j), tol=ATOL)
    assert ae(v.real, -1.8942716983721074932, tol=PTOL)
    assert ae(v.imag, -2.4689102827070540799, tol=PTOL)
    v = fp.e1((-4.0 + 1.0j))
    assert ae(v, (-14.806699492675420438 + 9.1384225230837893776j), tol=ATOL)
    assert ae(v.real, -14.806699492675420438, tol=PTOL)
    assert ae(v.imag, 9.1384225230837893776, tol=PTOL)
    v = fp.e1((-8.0 + 2.0j))
    assert ae(v, (54.633252667426386294 + 413.20318163814670688j), tol=ATOL)
    assert ae(v.real, 54.633252667426386294, tol=PTOL)
    assert ae(v.imag, 413.20318163814670688, tol=PTOL)
    v = fp.e1((-20.0 + 5.0j))
    assert ae(v, (-711836.97165402624643 - 24745250.939695900956j), tol=ATOL)
    assert ae(v.real, -711836.97165402624643, tol=PTOL)
    assert ae(v.imag, -24745250.939695900956, tol=PTOL)
    v = fp.e1((-80.0 + 20.0j))
    assert ae(v, (-4.2139911108612653091e+32 + 5.3367124741918251637e+32j), tol=ATOL)
    assert ae(v.real, -4.2139911108612653091e+32, tol=PTOL)
    assert ae(v.imag, 5.3367124741918251637e+32, tol=PTOL)
    v = fp.e1((-120.0 + 30.0j))
    assert ae(v, (9.7760616203707508892e+48 - 1.058257682317195792e+50j), tol=ATOL)
    assert ae(v.real, 9.7760616203707508892e+48, tol=PTOL)
    assert ae(v.imag, -1.058257682317195792e+50, tol=PTOL)
    v = fp.e1((-160.0 + 40.0j))
    assert ae(v, (8.7065541466623638861e+66 + 1.6577106725141739889e+67j), tol=ATOL)
    assert ae(v.real, 8.7065541466623638861e+66, tol=PTOL)
    assert ae(v.imag, 1.6577106725141739889e+67, tol=PTOL)
    v = fp.e1((-200.0 + 50.0j))
    assert ae(v, (-3.070744996327018106e+84 - 1.7243244846769415903e+84j), tol=ATOL)
    assert ae(v.real, -3.070744996327018106e+84, tol=PTOL)
    assert ae(v.imag, -1.7243244846769415903e+84, tol=PTOL)
    v = fp.e1((-320.0 + 80.0j))
    assert ae(v, (9.9960598637998647276e+135 - 2.6855081527595608863e+136j), tol=ATOL)
    assert ae(v.real, 9.9960598637998647276e+135, tol=PTOL)
    assert ae(v.imag, -2.6855081527595608863e+136, tol=PTOL)
    v = fp.e1(-1.1641532182693481445e-10)
    assert ae(v, (22.296641293460247028 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, 22.296641293460247028, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.e1(-0.25)
    assert ae(v, (0.54254326466191372953 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, 0.54254326466191372953, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.e1(-1.0)
    assert ae(v, (-1.8951178163559367555 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -1.8951178163559367555, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.e1(-2.0)
    assert ae(v, (-4.9542343560018901634 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -4.9542343560018901634, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.e1(-5.0)
    assert ae(v, (-40.185275355803177455 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -40.185275355803177455, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.e1(-20.0)
    assert ae(v, (-25615652.66405658882 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -25615652.66405658882, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.e1(-30.0)
    assert ae(v, (-368973209407.27419706 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -368973209407.27419706, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.e1(-40.0)
    assert ae(v, (-6039718263611241.5784 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -6039718263611241.5784, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.e1(-50.0)
    assert ae(v, (-1.0585636897131690963e+20 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -1.0585636897131690963e+20, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.e1(-80.0)
    assert ae(v, (-7.0146000049047999696e+32 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -7.0146000049047999696e+32, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.e1((-1.1641532182693481445e-10 + 0.0j))
    assert ae(v, (22.296641293460247028 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, 22.296641293460247028, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.e1((-0.25 + 0.0j))
    assert ae(v, (0.54254326466191372953 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, 0.54254326466191372953, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.e1((-1.0 + 0.0j))
    assert ae(v, (-1.8951178163559367555 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -1.8951178163559367555, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.e1((-2.0 + 0.0j))
    assert ae(v, (-4.9542343560018901634 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -4.9542343560018901634, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.e1((-5.0 + 0.0j))
    assert ae(v, (-40.185275355803177455 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -40.185275355803177455, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.e1((-20.0 + 0.0j))
    assert ae(v, (-25615652.66405658882 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -25615652.66405658882, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.e1((-30.0 + 0.0j))
    assert ae(v, (-368973209407.27419706 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -368973209407.27419706, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.e1((-40.0 + 0.0j))
    assert ae(v, (-6039718263611241.5784 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -6039718263611241.5784, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.e1((-50.0 + 0.0j))
    assert ae(v, (-1.0585636897131690963e+20 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -1.0585636897131690963e+20, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.e1((-80.0 + 0.0j))
    assert ae(v, (-7.0146000049047999696e+32 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -7.0146000049047999696e+32, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.e1((-4.6566128730773925781e-10 - 1.1641532182693481445e-10j))
    assert ae(v, (20.880034621082893023 + 2.8966139903465137624j), tol=ATOL)
    assert ae(v.real, 20.880034621082893023, tol=PTOL)
    assert ae(v.imag, 2.8966139903465137624, tol=PTOL)
    v = fp.e1((-1.0 - 0.25j))
    assert ae(v, (-1.8942716983721074932 + 2.4689102827070540799j), tol=ATOL)
    assert ae(v.real, -1.8942716983721074932, tol=PTOL)
    assert ae(v.imag, 2.4689102827070540799, tol=PTOL)
    v = fp.e1((-4.0 - 1.0j))
    assert ae(v, (-14.806699492675420438 - 9.1384225230837893776j), tol=ATOL)
    assert ae(v.real, -14.806699492675420438, tol=PTOL)
    assert ae(v.imag, -9.1384225230837893776, tol=PTOL)
    v = fp.e1((-8.0 - 2.0j))
    assert ae(v, (54.633252667426386294 - 413.20318163814670688j), tol=ATOL)
    assert ae(v.real, 54.633252667426386294, tol=PTOL)
    assert ae(v.imag, -413.20318163814670688, tol=PTOL)
    v = fp.e1((-20.0 - 5.0j))
    assert ae(v, (-711836.97165402624643 + 24745250.939695900956j), tol=ATOL)
    assert ae(v.real, -711836.97165402624643, tol=PTOL)
    assert ae(v.imag, 24745250.939695900956, tol=PTOL)
    v = fp.e1((-80.0 - 20.0j))
    assert ae(v, (-4.2139911108612653091e+32 - 5.3367124741918251637e+32j), tol=ATOL)
    assert ae(v.real, -4.2139911108612653091e+32, tol=PTOL)
    assert ae(v.imag, -5.3367124741918251637e+32, tol=PTOL)
    v = fp.e1((-120.0 - 30.0j))
    assert ae(v, (9.7760616203707508892e+48 + 1.058257682317195792e+50j), tol=ATOL)
    assert ae(v.real, 9.7760616203707508892e+48, tol=PTOL)
    assert ae(v.imag, 1.058257682317195792e+50, tol=PTOL)
    v = fp.e1((-160.0 - 40.0j))
    assert ae(v, (8.7065541466623638861e+66 - 1.6577106725141739889e+67j), tol=ATOL)
    assert ae(v.real, 8.7065541466623638861e+66, tol=PTOL)
    assert ae(v.imag, -1.6577106725141739889e+67, tol=PTOL)
    v = fp.e1((-200.0 - 50.0j))
    assert ae(v, (-3.070744996327018106e+84 + 1.7243244846769415903e+84j), tol=ATOL)
    assert ae(v.real, -3.070744996327018106e+84, tol=PTOL)
    assert ae(v.imag, 1.7243244846769415903e+84, tol=PTOL)
    v = fp.e1((-320.0 - 80.0j))
    assert ae(v, (9.9960598637998647276e+135 + 2.6855081527595608863e+136j), tol=ATOL)
    assert ae(v.real, 9.9960598637998647276e+135, tol=PTOL)
    assert ae(v.imag, 2.6855081527595608863e+136, tol=PTOL)
    v = fp.e1((-1.1641532182693481445e-10 - 1.1641532182693481445e-10j))
    assert ae(v, (21.950067703180274374 + 2.356194490075929607j), tol=ATOL)
    assert ae(v.real, 21.950067703180274374, tol=PTOL)
    assert ae(v.imag, 2.356194490075929607, tol=PTOL)
    v = fp.e1((-0.25 - 0.25j))
    assert ae(v, (0.21441047326710323254 + 2.0732153554307936389j), tol=ATOL)
    assert ae(v.real, 0.21441047326710323254, tol=PTOL)
    assert ae(v.imag, 2.0732153554307936389, tol=PTOL)
    v = fp.e1((-1.0 - 1.0j))
    assert ae(v, (-1.7646259855638540684 + 0.7538228020792708192j), tol=ATOL)
    assert ae(v.real, -1.7646259855638540684, tol=PTOL)
    assert ae(v.imag, 0.7538228020792708192, tol=PTOL)
    v = fp.e1((-2.0 - 2.0j))
    assert ae(v, (-1.8920781621855474089 - 2.1753697842428647236j), tol=ATOL)
    assert ae(v.real, -1.8920781621855474089, tol=PTOL)
    assert ae(v.imag, -2.1753697842428647236, tol=PTOL)
    v = fp.e1((-5.0 - 5.0j))
    assert ae(v, (13.470936071475245856 + 18.464085049321024206j), tol=ATOL)
    assert ae(v.real, 13.470936071475245856, tol=PTOL)
    assert ae(v.imag, 18.464085049321024206, tol=PTOL)
    v = fp.e1((-20.0 - 20.0j))
    assert ae(v, (-16589317.398788971896 - 5831702.3296441771206j), tol=ATOL)
    assert ae(v.real, -16589317.398788971896, tol=PTOL)
    assert ae(v.imag, -5831702.3296441771206, tol=PTOL)
    v = fp.e1((-30.0 - 30.0j))
    assert ae(v, (154596484273.69322527 + 204179357837.41389696j), tol=ATOL)
    assert ae(v.real, 154596484273.69322527, tol=PTOL)
    assert ae(v.imag, 204179357837.41389696, tol=PTOL)
    v = fp.e1((-40.0 - 40.0j))
    assert ae(v, (-287512180321448.45408 - 4203502407932314.974j), tol=ATOL)
    assert ae(v.real, -287512180321448.45408, tol=PTOL)
    assert ae(v.imag, -4203502407932314.974, tol=PTOL)
    v = fp.e1((-50.0 - 50.0j))
    assert ae(v, (-36128528616649268826.0 + 64648801861338741963.0j), tol=ATOL)
    assert ae(v.real, -36128528616649268826.0, tol=PTOL)
    assert ae(v.imag, 64648801861338741963.0, tol=PTOL)
    v = fp.e1((-80.0 - 80.0j))
    assert ae(v, (3.8674816337930010217e+32 + 3.0540709639658071041e+32j), tol=ATOL)
    assert ae(v.real, 3.8674816337930010217e+32, tol=PTOL)
    assert ae(v.imag, 3.0540709639658071041e+32, tol=PTOL)
    v = fp.e1((-1.1641532182693481445e-10 - 4.6566128730773925781e-10j))
    assert ae(v, (20.880034621432138988 + 1.8157749894560994861j), tol=ATOL)
    assert ae(v.real, 20.880034621432138988, tol=PTOL)
    assert ae(v.imag, 1.8157749894560994861, tol=PTOL)
    v = fp.e1((-0.25 - 1.0j))
    assert ae(v, (-0.59066621214766308594 + 0.74474454765205036972j), tol=ATOL)
    assert ae(v.real, -0.59066621214766308594, tol=PTOL)
    assert ae(v.imag, 0.74474454765205036972, tol=PTOL)
    v = fp.e1((-1.0 - 4.0j))
    assert ae(v, (0.49739047283060471093 - 0.41543605404038863174j), tol=ATOL)
    assert ae(v.real, 0.49739047283060471093, tol=PTOL)
    assert ae(v.imag, -0.41543605404038863174, tol=PTOL)
    v = fp.e1((-2.0 - 8.0j))
    assert ae(v, (-0.8705211147733730969 - 0.24099328498605539667j), tol=ATOL)
    assert ae(v.real, -0.8705211147733730969, tol=PTOL)
    assert ae(v.imag, -0.24099328498605539667, tol=PTOL)
    v = fp.e1((-5.0 - 20.0j))
    assert ae(v, (-7.0789514293925893007 + 1.6102177171960790536j), tol=ATOL)
    assert ae(v.real, -7.0789514293925893007, tol=PTOL)
    assert ae(v.imag, 1.6102177171960790536, tol=PTOL)
    v = fp.e1((-20.0 - 80.0j))
    assert ae(v, (5855431.4907298084434 + 720920.93315409165707j), tol=ATOL)
    assert ae(v.real, 5855431.4907298084434, tol=PTOL)
    assert ae(v.imag, 720920.93315409165707, tol=PTOL)
    v = fp.e1((-30.0 - 120.0j))
    assert ae(v, (-65402491644.703470747 + 56697658399.657460294j), tol=ATOL)
    assert ae(v.real, -65402491644.703470747, tol=PTOL)
    assert ae(v.imag, 56697658399.657460294, tol=PTOL)
    v = fp.e1((-40.0 - 160.0j))
    assert ae(v, (25504929379604.776769 - 1429035198630573.2463j), tol=ATOL)
    assert ae(v.real, 25504929379604.776769, tol=PTOL)
    assert ae(v.imag, -1429035198630573.2463, tol=PTOL)
    v = fp.e1((-50.0 - 200.0j))
    assert ae(v, (18437746526988116954.0 + 17146362239046152345.0j), tol=ATOL)
    assert ae(v.real, 18437746526988116954.0, tol=PTOL)
    assert ae(v.imag, 17146362239046152345.0, tol=PTOL)
    v = fp.e1((-80.0 - 320.0j))
    assert ae(v, (3.3464697299634526706e+31 + 1.6473152633843023919e+32j), tol=ATOL)
    assert ae(v.real, 3.3464697299634526706e+31, tol=PTOL)
    assert ae(v.imag, 1.6473152633843023919e+32, tol=PTOL)
    v = fp.e1((0.0 - 1.1641532182693481445e-10j))
    assert ae(v, (22.29664129357666235 + 1.5707963266784812974j), tol=ATOL)
    assert ae(v.real, 22.29664129357666235, tol=PTOL)
    assert ae(v.imag, 1.5707963266784812974, tol=PTOL)
    v = fp.e1((0.0 - 0.25j))
    assert ae(v, (0.82466306258094565309 + 1.3216627564751394551j), tol=ATOL)
    assert ae(v.real, 0.82466306258094565309, tol=PTOL)
    assert ae(v.imag, 1.3216627564751394551, tol=PTOL)
    v = fp.e1((0.0 - 1.0j))
    assert ae(v, (-0.33740392290096813466 + 0.62471325642771360429j), tol=ATOL)
    assert ae(v.real, -0.33740392290096813466, tol=PTOL)
    assert ae(v.imag, 0.62471325642771360429, tol=PTOL)
    v = fp.e1((0.0 - 2.0j))
    assert ae(v, (-0.4229808287748649957 - 0.034616650007798229345j), tol=ATOL)
    assert ae(v.real, -0.4229808287748649957, tol=PTOL)
    assert ae(v.imag, -0.034616650007798229345, tol=PTOL)
    v = fp.e1((0.0 - 5.0j))
    assert ae(v, (0.19002974965664387862 + 0.020865081850222481957j), tol=ATOL)
    assert ae(v.real, 0.19002974965664387862, tol=PTOL)
    assert ae(v.imag, 0.020865081850222481957, tol=PTOL)
    v = fp.e1((0.0 - 20.0j))
    assert ae(v, (-0.04441982084535331654 + 0.022554625751456779068j), tol=ATOL)
    assert ae(v.real, -0.04441982084535331654, tol=PTOL)
    assert ae(v.imag, 0.022554625751456779068, tol=PTOL)
    v = fp.e1((0.0 - 30.0j))
    assert ae(v, (0.033032417282071143779 + 0.0040397867645455082476j), tol=ATOL)
    assert ae(v.real, 0.033032417282071143779, tol=PTOL)
    assert ae(v.imag, 0.0040397867645455082476, tol=PTOL)
    v = fp.e1((0.0 - 40.0j))
    assert ae(v, (-0.019020007896208766962 - 0.016188792559887887544j), tol=ATOL)
    assert ae(v.real, -0.019020007896208766962, tol=PTOL)
    assert ae(v.imag, -0.016188792559887887544, tol=PTOL)
    v = fp.e1((0.0 - 50.0j))
    assert ae(v, (0.0056283863241163054402 + 0.019179254308960724503j), tol=ATOL)
    assert ae(v.real, 0.0056283863241163054402, tol=PTOL)
    assert ae(v.imag, 0.019179254308960724503, tol=PTOL)
    v = fp.e1((0.0 - 80.0j))
    assert ae(v, (0.012402501155070958192 - 0.0015345601175906961199j), tol=ATOL)
    assert ae(v.real, 0.012402501155070958192, tol=PTOL)
    assert ae(v.imag, -0.0015345601175906961199, tol=PTOL)
    v = fp.e1((1.1641532182693481445e-10 - 4.6566128730773925781e-10j))
    assert ae(v, (20.880034621664969632 + 1.3258176632023711778j), tol=ATOL)
    assert ae(v.real, 20.880034621664969632, tol=PTOL)
    assert ae(v.imag, 1.3258176632023711778, tol=PTOL)
    v = fp.e1((0.25 - 1.0j))
    assert ae(v, (-0.16868306393667788761 + 0.4858011885947426971j), tol=ATOL)
    assert ae(v.real, -0.16868306393667788761, tol=PTOL)
    assert ae(v.imag, 0.4858011885947426971, tol=PTOL)
    v = fp.e1((1.0 - 4.0j))
    assert ae(v, (0.03373591813926547318 - 0.073523452241083821877j), tol=ATOL)
    assert ae(v.real, 0.03373591813926547318, tol=PTOL)
    assert ae(v.imag, -0.073523452241083821877, tol=PTOL)
    v = fp.e1((2.0 - 8.0j))
    assert ae(v, (-0.015392833434733785143 + 0.0031747121557605415914j), tol=ATOL)
    assert ae(v.real, -0.015392833434733785143, tol=PTOL)
    assert ae(v.imag, 0.0031747121557605415914, tol=PTOL)
    v = fp.e1((5.0 - 20.0j))
    assert ae(v, (-0.00024419662286542966525 + 0.00021008322966152755674j), tol=ATOL)
    assert ae(v.real, -0.00024419662286542966525, tol=PTOL)
    assert ae(v.imag, 0.00021008322966152755674, tol=PTOL)
    v = fp.e1((20.0 - 80.0j))
    assert ae(v, (2.3255552781051330088e-11 - 8.9463918891349438007e-12j), tol=ATOL)
    assert ae(v.real, 2.3255552781051330088e-11, tol=PTOL)
    assert ae(v.imag, -8.9463918891349438007e-12, tol=PTOL)
    v = fp.e1((30.0 - 120.0j))
    assert ae(v, (-2.7068919097124652332e-16 + 7.0477762411705130239e-16j), tol=ATOL)
    assert ae(v.real, -2.7068919097124652332e-16, tol=PTOL)
    assert ae(v.imag, 7.0477762411705130239e-16, tol=PTOL)
    v = fp.e1((40.0 - 160.0j))
    assert ae(v, (-1.1695597827678024687e-20 - 2.2907401455645736661e-20j), tol=ATOL)
    assert ae(v.real, -1.1695597827678024687e-20, tol=PTOL)
    assert ae(v.imag, -2.2907401455645736661e-20, tol=PTOL)
    v = fp.e1((50.0 - 200.0j))
    assert ae(v, (9.0323746914410162531e-25 + 2.3950601790033530935e-25j), tol=ATOL)
    assert ae(v.real, 9.0323746914410162531e-25, tol=PTOL)
    assert ae(v.imag, 2.3950601790033530935e-25, tol=PTOL)
    v = fp.e1((80.0 - 320.0j))
    assert ae(v, (3.4819106748728063576e-38 + 4.215653005615772724e-38j), tol=ATOL)
    assert ae(v.real, 3.4819106748728063576e-38, tol=PTOL)
    assert ae(v.imag, 4.215653005615772724e-38, tol=PTOL)
    v = fp.e1((1.1641532182693481445e-10 - 1.1641532182693481445e-10j))
    assert ae(v, (21.950067703413105017 + 0.7853981632810329878j), tol=ATOL)
    assert ae(v.real, 21.950067703413105017, tol=PTOL)
    assert ae(v.imag, 0.7853981632810329878, tol=PTOL)
    v = fp.e1((0.25 - 0.25j))
    assert ae(v, (0.71092525792923287894 + 0.56491812441304194711j), tol=ATOL)
    assert ae(v.real, 0.71092525792923287894, tol=PTOL)
    assert ae(v.imag, 0.56491812441304194711, tol=PTOL)
    v = fp.e1((1.0 - 1.0j))
    assert ae(v, (0.00028162445198141832551 + 0.17932453503935894015j), tol=ATOL)
    assert ae(v.real, 0.00028162445198141832551, tol=PTOL)
    assert ae(v.imag, 0.17932453503935894015, tol=PTOL)
    v = fp.e1((2.0 - 2.0j))
    assert ae(v, (-0.033767089606562004246 + 0.018599414169750541925j), tol=ATOL)
    assert ae(v.real, -0.033767089606562004246, tol=PTOL)
    assert ae(v.imag, 0.018599414169750541925, tol=PTOL)
    v = fp.e1((5.0 - 5.0j))
    assert ae(v, (0.0007266506660356393891 - 0.00047102780163522245054j), tol=ATOL)
    assert ae(v.real, 0.0007266506660356393891, tol=PTOL)
    assert ae(v.imag, -0.00047102780163522245054, tol=PTOL)
    v = fp.e1((20.0 - 20.0j))
    assert ae(v, (-2.3824537449367396579e-11 + 6.6969873156525615158e-11j), tol=ATOL)
    assert ae(v.real, -2.3824537449367396579e-11, tol=PTOL)
    assert ae(v.imag, 6.6969873156525615158e-11, tol=PTOL)
    v = fp.e1((30.0 - 30.0j))
    assert ae(v, (1.7316045841744061617e-15 - 1.3065678019487308689e-15j), tol=ATOL)
    assert ae(v.real, 1.7316045841744061617e-15, tol=PTOL)
    assert ae(v.imag, -1.3065678019487308689e-15, tol=PTOL)
    v = fp.e1((40.0 - 40.0j))
    assert ae(v, (-7.4001043002899232182e-20 + 4.991847855336816304e-21j), tol=ATOL)
    assert ae(v.real, -7.4001043002899232182e-20, tol=PTOL)
    assert ae(v.imag, 4.991847855336816304e-21, tol=PTOL)
    v = fp.e1((50.0 - 50.0j))
    assert ae(v, (2.3566128324644641219e-24 + 1.3188326726201614778e-24j), tol=ATOL)
    assert ae(v.real, 2.3566128324644641219e-24, tol=PTOL)
    assert ae(v.imag, 1.3188326726201614778e-24, tol=PTOL)
    v = fp.e1((80.0 - 80.0j))
    assert ae(v, (9.8279750572186526673e-38 - 1.243952841288868831e-37j), tol=ATOL)
    assert ae(v.real, 9.8279750572186526673e-38, tol=PTOL)
    assert ae(v.imag, -1.243952841288868831e-37, tol=PTOL)
    v = fp.e1((4.6566128730773925781e-10 - 1.1641532182693481445e-10j))
    assert ae(v, (20.880034622014215597 + 0.24497866301044883237j), tol=ATOL)
    assert ae(v.real, 20.880034622014215597, tol=PTOL)
    assert ae(v.imag, 0.24497866301044883237, tol=PTOL)
    v = fp.e1((1.0 - 0.25j))
    assert ae(v, (0.19731063945004229095 + 0.087366045774299963672j), tol=ATOL)
    assert ae(v.real, 0.19731063945004229095, tol=PTOL)
    assert ae(v.imag, 0.087366045774299963672, tol=PTOL)
    v = fp.e1((4.0 - 1.0j))
    assert ae(v, (0.0013106173980145506944 + 0.0034542480199350626699j), tol=ATOL)
    assert ae(v.real, 0.0013106173980145506944, tol=PTOL)
    assert ae(v.imag, 0.0034542480199350626699, tol=PTOL)
    v = fp.e1((8.0 - 2.0j))
    assert ae(v, (-0.000022278049065270225945 + 0.000029191940456521555288j), tol=ATOL)
    assert ae(v.real, -0.000022278049065270225945, tol=PTOL)
    assert ae(v.imag, 0.000029191940456521555288, tol=PTOL)
    v = fp.e1((20.0 - 5.0j))
    assert ae(v, (4.7711374515765346894e-11 - 8.2902652405126947359e-11j), tol=ATOL)
    assert ae(v.real, 4.7711374515765346894e-11, tol=PTOL)
    assert ae(v.imag, -8.2902652405126947359e-11, tol=PTOL)
    v = fp.e1((80.0 - 20.0j))
    assert ae(v, (3.8353473865788235787e-38 + 2.129247592349605139e-37j), tol=ATOL)
    assert ae(v.real, 3.8353473865788235787e-38, tol=PTOL)
    assert ae(v.imag, 2.129247592349605139e-37, tol=PTOL)
    v = fp.e1((120.0 - 30.0j))
    assert ae(v, (2.3836002337480334716e-55 - 5.6704043587126198306e-55j), tol=ATOL)
    assert ae(v.real, 2.3836002337480334716e-55, tol=PTOL)
    assert ae(v.imag, -5.6704043587126198306e-55, tol=PTOL)
    v = fp.e1((160.0 - 40.0j))
    assert ae(v, (-1.6238022898654510661e-72 + 1.104172355572287367e-72j), tol=ATOL)
    assert ae(v.real, -1.6238022898654510661e-72, tol=PTOL)
    assert ae(v.imag, 1.104172355572287367e-72, tol=PTOL)
    v = fp.e1((200.0 - 50.0j))
    assert ae(v, (6.6800061461666228487e-90 - 1.4473816083541016115e-91j), tol=ATOL)
    assert ae(v.real, 6.6800061461666228487e-90, tol=PTOL)
    assert ae(v.imag, -1.4473816083541016115e-91, tol=PTOL)
    v = fp.e1((320.0 - 80.0j))
    assert ae(v, (4.2737871527778786157e-143 - 3.1789935525785660314e-142j), tol=ATOL)
    assert ae(v.real, 4.2737871527778786157e-143, tol=PTOL)
    assert ae(v.imag, -3.1789935525785660314e-142, tol=PTOL)
    v = fp.ei(1.1641532182693481445e-10)
    assert ae(v, -22.296641293460247028, tol=ATOL)
    assert type(v) is float
    v = fp.ei(0.25)
    assert ae(v, -0.54254326466191372953, tol=ATOL)
    assert type(v) is float
    v = fp.ei(1.0)
    assert ae(v, 1.8951178163559367555, tol=ATOL)
    assert type(v) is float
    v = fp.ei(2.0)
    assert ae(v, 4.9542343560018901634, tol=ATOL)
    assert type(v) is float
    v = fp.ei(5.0)
    assert ae(v, 40.185275355803177455, tol=ATOL)
    assert type(v) is float
    v = fp.ei(20.0)
    assert ae(v, 25615652.66405658882, tol=ATOL)
    assert type(v) is float
    v = fp.ei(30.0)
    assert ae(v, 368973209407.27419706, tol=ATOL)
    assert type(v) is float
    v = fp.ei(40.0)
    assert ae(v, 6039718263611241.5784, tol=ATOL)
    assert type(v) is float
    v = fp.ei(50.0)
    assert ae(v, 1.0585636897131690963e+20, tol=ATOL)
    assert type(v) is float
    v = fp.ei(80.0)
    assert ae(v, 7.0146000049047999696e+32, tol=ATOL)
    assert type(v) is float
    v = fp.ei((1.1641532182693481445e-10 + 0.0j))
    assert ae(v, (-22.296641293460247028 + 0.0j), tol=ATOL)
    assert ae(v.real, -22.296641293460247028, tol=PTOL)
    assert v.imag == 0
    v = fp.ei((0.25 + 0.0j))
    assert ae(v, (-0.54254326466191372953 + 0.0j), tol=ATOL)
    assert ae(v.real, -0.54254326466191372953, tol=PTOL)
    assert v.imag == 0
    v = fp.ei((1.0 + 0.0j))
    assert ae(v, (1.8951178163559367555 + 0.0j), tol=ATOL)
    assert ae(v.real, 1.8951178163559367555, tol=PTOL)
    assert v.imag == 0
    v = fp.ei((2.0 + 0.0j))
    assert ae(v, (4.9542343560018901634 + 0.0j), tol=ATOL)
    assert ae(v.real, 4.9542343560018901634, tol=PTOL)
    assert v.imag == 0
    v = fp.ei((5.0 + 0.0j))
    assert ae(v, (40.185275355803177455 + 0.0j), tol=ATOL)
    assert ae(v.real, 40.185275355803177455, tol=PTOL)
    assert v.imag == 0
    v = fp.ei((20.0 + 0.0j))
    assert ae(v, (25615652.66405658882 + 0.0j), tol=ATOL)
    assert ae(v.real, 25615652.66405658882, tol=PTOL)
    assert v.imag == 0
    v = fp.ei((30.0 + 0.0j))
    assert ae(v, (368973209407.27419706 + 0.0j), tol=ATOL)
    assert ae(v.real, 368973209407.27419706, tol=PTOL)
    assert v.imag == 0
    v = fp.ei((40.0 + 0.0j))
    assert ae(v, (6039718263611241.5784 + 0.0j), tol=ATOL)
    assert ae(v.real, 6039718263611241.5784, tol=PTOL)
    assert v.imag == 0
    v = fp.ei((50.0 + 0.0j))
    assert ae(v, (1.0585636897131690963e+20 + 0.0j), tol=ATOL)
    assert ae(v.real, 1.0585636897131690963e+20, tol=PTOL)
    assert v.imag == 0
    v = fp.ei((80.0 + 0.0j))
    assert ae(v, (7.0146000049047999696e+32 + 0.0j), tol=ATOL)
    assert ae(v.real, 7.0146000049047999696e+32, tol=PTOL)
    assert v.imag == 0
    v = fp.ei((4.6566128730773925781e-10 + 1.1641532182693481445e-10j))
    assert ae(v, (-20.880034621082893023 + 0.24497866324327947603j), tol=ATOL)
    assert ae(v.real, -20.880034621082893023, tol=PTOL)
    assert ae(v.imag, 0.24497866324327947603, tol=PTOL)
    v = fp.ei((1.0 + 0.25j))
    assert ae(v, (1.8942716983721074932 + 0.67268237088273915854j), tol=ATOL)
    assert ae(v.real, 1.8942716983721074932, tol=PTOL)
    assert ae(v.imag, 0.67268237088273915854, tol=PTOL)
    v = fp.ei((4.0 + 1.0j))
    assert ae(v, (14.806699492675420438 + 12.280015176673582616j), tol=ATOL)
    assert ae(v.real, 14.806699492675420438, tol=PTOL)
    assert ae(v.imag, 12.280015176673582616, tol=PTOL)
    v = fp.ei((8.0 + 2.0j))
    assert ae(v, (-54.633252667426386294 + 416.34477429173650012j), tol=ATOL)
    assert ae(v.real, -54.633252667426386294, tol=PTOL)
    assert ae(v.imag, 416.34477429173650012, tol=PTOL)
    v = fp.ei((20.0 + 5.0j))
    assert ae(v, (711836.97165402624643 - 24745247.798103247366j), tol=ATOL)
    assert ae(v.real, 711836.97165402624643, tol=PTOL)
    assert ae(v.imag, -24745247.798103247366, tol=PTOL)
    v = fp.ei((80.0 + 20.0j))
    assert ae(v, (4.2139911108612653091e+32 + 5.3367124741918251637e+32j), tol=ATOL)
    assert ae(v.real, 4.2139911108612653091e+32, tol=PTOL)
    assert ae(v.imag, 5.3367124741918251637e+32, tol=PTOL)
    v = fp.ei((120.0 + 30.0j))
    assert ae(v, (-9.7760616203707508892e+48 - 1.058257682317195792e+50j), tol=ATOL)
    assert ae(v.real, -9.7760616203707508892e+48, tol=PTOL)
    assert ae(v.imag, -1.058257682317195792e+50, tol=PTOL)
    v = fp.ei((160.0 + 40.0j))
    assert ae(v, (-8.7065541466623638861e+66 + 1.6577106725141739889e+67j), tol=ATOL)
    assert ae(v.real, -8.7065541466623638861e+66, tol=PTOL)
    assert ae(v.imag, 1.6577106725141739889e+67, tol=PTOL)
    v = fp.ei((200.0 + 50.0j))
    assert ae(v, (3.070744996327018106e+84 - 1.7243244846769415903e+84j), tol=ATOL)
    assert ae(v.real, 3.070744996327018106e+84, tol=PTOL)
    assert ae(v.imag, -1.7243244846769415903e+84, tol=PTOL)
    v = fp.ei((320.0 + 80.0j))
    assert ae(v, (-9.9960598637998647276e+135 - 2.6855081527595608863e+136j), tol=ATOL)
    assert ae(v.real, -9.9960598637998647276e+135, tol=PTOL)
    assert ae(v.imag, -2.6855081527595608863e+136, tol=PTOL)
    v = fp.ei((1.1641532182693481445e-10 + 1.1641532182693481445e-10j))
    assert ae(v, (-21.950067703180274374 + 0.78539816351386363145j), tol=ATOL)
    assert ae(v.real, -21.950067703180274374, tol=PTOL)
    assert ae(v.imag, 0.78539816351386363145, tol=PTOL)
    v = fp.ei((0.25 + 0.25j))
    assert ae(v, (-0.21441047326710323254 + 1.0683772981589995996j), tol=ATOL)
    assert ae(v.real, -0.21441047326710323254, tol=PTOL)
    assert ae(v.imag, 1.0683772981589995996, tol=PTOL)
    v = fp.ei((1.0 + 1.0j))
    assert ae(v, (1.7646259855638540684 + 2.3877698515105224193j), tol=ATOL)
    assert ae(v.real, 1.7646259855638540684, tol=PTOL)
    assert ae(v.imag, 2.3877698515105224193, tol=PTOL)
    v = fp.ei((2.0 + 2.0j))
    assert ae(v, (1.8920781621855474089 + 5.3169624378326579621j), tol=ATOL)
    assert ae(v.real, 1.8920781621855474089, tol=PTOL)
    assert ae(v.imag, 5.3169624378326579621, tol=PTOL)
    v = fp.ei((5.0 + 5.0j))
    assert ae(v, (-13.470936071475245856 - 15.322492395731230968j), tol=ATOL)
    assert ae(v.real, -13.470936071475245856, tol=PTOL)
    assert ae(v.imag, -15.322492395731230968, tol=PTOL)
    v = fp.ei((20.0 + 20.0j))
    assert ae(v, (16589317.398788971896 + 5831705.4712368307104j), tol=ATOL)
    assert ae(v.real, 16589317.398788971896, tol=PTOL)
    assert ae(v.imag, 5831705.4712368307104, tol=PTOL)
    v = fp.ei((30.0 + 30.0j))
    assert ae(v, (-154596484273.69322527 - 204179357834.2723043j), tol=ATOL)
    assert ae(v.real, -154596484273.69322527, tol=PTOL)
    assert ae(v.imag, -204179357834.2723043, tol=PTOL)
    v = fp.ei((40.0 + 40.0j))
    assert ae(v, (287512180321448.45408 + 4203502407932318.1156j), tol=ATOL)
    assert ae(v.real, 287512180321448.45408, tol=PTOL)
    assert ae(v.imag, 4203502407932318.1156, tol=PTOL)
    v = fp.ei((50.0 + 50.0j))
    assert ae(v, (36128528616649268826.0 - 64648801861338741960.0j), tol=ATOL)
    assert ae(v.real, 36128528616649268826.0, tol=PTOL)
    assert ae(v.imag, -64648801861338741960.0, tol=PTOL)
    v = fp.ei((80.0 + 80.0j))
    assert ae(v, (-3.8674816337930010217e+32 - 3.0540709639658071041e+32j), tol=ATOL)
    assert ae(v.real, -3.8674816337930010217e+32, tol=PTOL)
    assert ae(v.imag, -3.0540709639658071041e+32, tol=PTOL)
    v = fp.ei((1.1641532182693481445e-10 + 4.6566128730773925781e-10j))
    assert ae(v, (-20.880034621432138988 + 1.3258176641336937524j), tol=ATOL)
    assert ae(v.real, -20.880034621432138988, tol=PTOL)
    assert ae(v.imag, 1.3258176641336937524, tol=PTOL)
    v = fp.ei((0.25 + 1.0j))
    assert ae(v, (0.59066621214766308594 + 2.3968481059377428687j), tol=ATOL)
    assert ae(v.real, 0.59066621214766308594, tol=PTOL)
    assert ae(v.imag, 2.3968481059377428687, tol=PTOL)
    v = fp.ei((1.0 + 4.0j))
    assert ae(v, (-0.49739047283060471093 + 3.5570287076301818702j), tol=ATOL)
    assert ae(v.real, -0.49739047283060471093, tol=PTOL)
    assert ae(v.imag, 3.5570287076301818702, tol=PTOL)
    v = fp.ei((2.0 + 8.0j))
    assert ae(v, (0.8705211147733730969 + 3.3825859385758486351j), tol=ATOL)
    assert ae(v.real, 0.8705211147733730969, tol=PTOL)
    assert ae(v.imag, 3.3825859385758486351, tol=PTOL)
    v = fp.ei((5.0 + 20.0j))
    assert ae(v, (7.0789514293925893007 + 1.5313749363937141849j), tol=ATOL)
    assert ae(v.real, 7.0789514293925893007, tol=PTOL)
    assert ae(v.imag, 1.5313749363937141849, tol=PTOL)
    v = fp.ei((20.0 + 80.0j))
    assert ae(v, (-5855431.4907298084434 - 720917.79156143806727j), tol=ATOL)
    assert ae(v.real, -5855431.4907298084434, tol=PTOL)
    assert ae(v.imag, -720917.79156143806727, tol=PTOL)
    v = fp.ei((30.0 + 120.0j))
    assert ae(v, (65402491644.703470747 - 56697658396.51586764j), tol=ATOL)
    assert ae(v.real, 65402491644.703470747, tol=PTOL)
    assert ae(v.imag, -56697658396.51586764, tol=PTOL)
    v = fp.ei((40.0 + 160.0j))
    assert ae(v, (-25504929379604.776769 + 1429035198630576.3879j), tol=ATOL)
    assert ae(v.real, -25504929379604.776769, tol=PTOL)
    assert ae(v.imag, 1429035198630576.3879, tol=PTOL)
    v = fp.ei((50.0 + 200.0j))
    assert ae(v, (-18437746526988116954.0 - 17146362239046152342.0j), tol=ATOL)
    assert ae(v.real, -18437746526988116954.0, tol=PTOL)
    assert ae(v.imag, -17146362239046152342.0, tol=PTOL)
    v = fp.ei((80.0 + 320.0j))
    assert ae(v, (-3.3464697299634526706e+31 - 1.6473152633843023919e+32j), tol=ATOL)
    assert ae(v.real, -3.3464697299634526706e+31, tol=PTOL)
    assert ae(v.imag, -1.6473152633843023919e+32, tol=PTOL)
    v = fp.ei((0.0 + 1.1641532182693481445e-10j))
    assert ae(v, (-22.29664129357666235 + 1.5707963269113119411j), tol=ATOL)
    assert ae(v.real, -22.29664129357666235, tol=PTOL)
    assert ae(v.imag, 1.5707963269113119411, tol=PTOL)
    v = fp.ei((0.0 + 0.25j))
    assert ae(v, (-0.82466306258094565309 + 1.8199298971146537833j), tol=ATOL)
    assert ae(v.real, -0.82466306258094565309, tol=PTOL)
    assert ae(v.imag, 1.8199298971146537833, tol=PTOL)
    v = fp.ei((0.0 + 1.0j))
    assert ae(v, (0.33740392290096813466 + 2.5168793971620796342j), tol=ATOL)
    assert ae(v.real, 0.33740392290096813466, tol=PTOL)
    assert ae(v.imag, 2.5168793971620796342, tol=PTOL)
    v = fp.ei((0.0 + 2.0j))
    assert ae(v, (0.4229808287748649957 + 3.1762093035975914678j), tol=ATOL)
    assert ae(v.real, 0.4229808287748649957, tol=PTOL)
    assert ae(v.imag, 3.1762093035975914678, tol=PTOL)
    v = fp.ei((0.0 + 5.0j))
    assert ae(v, (-0.19002974965664387862 + 3.1207275717395707565j), tol=ATOL)
    assert ae(v.real, -0.19002974965664387862, tol=PTOL)
    assert ae(v.imag, 3.1207275717395707565, tol=PTOL)
    v = fp.ei((0.0 + 20.0j))
    assert ae(v, (0.04441982084535331654 + 3.1190380278383364594j), tol=ATOL)
    assert ae(v.real, 0.04441982084535331654, tol=PTOL)
    assert ae(v.imag, 3.1190380278383364594, tol=PTOL)
    v = fp.ei((0.0 + 30.0j))
    assert ae(v, (-0.033032417282071143779 + 3.1375528668252477302j), tol=ATOL)
    assert ae(v.real, -0.033032417282071143779, tol=PTOL)
    assert ae(v.imag, 3.1375528668252477302, tol=PTOL)
    v = fp.ei((0.0 + 40.0j))
    assert ae(v, (0.019020007896208766962 + 3.157781446149681126j), tol=ATOL)
    assert ae(v.real, 0.019020007896208766962, tol=PTOL)
    assert ae(v.imag, 3.157781446149681126, tol=PTOL)
    v = fp.ei((0.0 + 50.0j))
    assert ae(v, (-0.0056283863241163054402 + 3.122413399280832514j), tol=ATOL)
    assert ae(v.real, -0.0056283863241163054402, tol=PTOL)
    assert ae(v.imag, 3.122413399280832514, tol=PTOL)
    v = fp.ei((0.0 + 80.0j))
    assert ae(v, (-0.012402501155070958192 + 3.1431272137073839346j), tol=ATOL)
    assert ae(v.real, -0.012402501155070958192, tol=PTOL)
    assert ae(v.imag, 3.1431272137073839346, tol=PTOL)
    v = fp.ei((-1.1641532182693481445e-10 + 4.6566128730773925781e-10j))
    assert ae(v, (-20.880034621664969632 + 1.8157749903874220607j), tol=ATOL)
    assert ae(v.real, -20.880034621664969632, tol=PTOL)
    assert ae(v.imag, 1.8157749903874220607, tol=PTOL)
    v = fp.ei((-0.25 + 1.0j))
    assert ae(v, (0.16868306393667788761 + 2.6557914649950505414j), tol=ATOL)
    assert ae(v.real, 0.16868306393667788761, tol=PTOL)
    assert ae(v.imag, 2.6557914649950505414, tol=PTOL)
    v = fp.ei((-1.0 + 4.0j))
    assert ae(v, (-0.03373591813926547318 + 3.2151161058308770603j), tol=ATOL)
    assert ae(v.real, -0.03373591813926547318, tol=PTOL)
    assert ae(v.imag, 3.2151161058308770603, tol=PTOL)
    v = fp.ei((-2.0 + 8.0j))
    assert ae(v, (0.015392833434733785143 + 3.1384179414340326969j), tol=ATOL)
    assert ae(v.real, 0.015392833434733785143, tol=PTOL)
    assert ae(v.imag, 3.1384179414340326969, tol=PTOL)
    v = fp.ei((-5.0 + 20.0j))
    assert ae(v, (0.00024419662286542966525 + 3.1413825703601317109j), tol=ATOL)
    assert ae(v.real, 0.00024419662286542966525, tol=PTOL)
    assert ae(v.imag, 3.1413825703601317109, tol=PTOL)
    v = fp.ei((-20.0 + 80.0j))
    assert ae(v, (-2.3255552781051330088e-11 + 3.1415926535987396304j), tol=ATOL)
    assert ae(v.real, -2.3255552781051330088e-11, tol=PTOL)
    assert ae(v.imag, 3.1415926535987396304, tol=PTOL)
    v = fp.ei((-30.0 + 120.0j))
    assert ae(v, (2.7068919097124652332e-16 + 3.1415926535897925337j), tol=ATOL)
    assert ae(v.real, 2.7068919097124652332e-16, tol=PTOL)
    assert ae(v.imag, 3.1415926535897925337, tol=PTOL)
    v = fp.ei((-40.0 + 160.0j))
    assert ae(v, (1.1695597827678024687e-20 + 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, 1.1695597827678024687e-20, tol=PTOL)
    assert ae(v.imag, 3.1415926535897932385, tol=PTOL)
    v = fp.ei((-50.0 + 200.0j))
    assert ae(v, (-9.0323746914410162531e-25 + 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -9.0323746914410162531e-25, tol=PTOL)
    assert ae(v.imag, 3.1415926535897932385, tol=PTOL)
    v = fp.ei((-80.0 + 320.0j))
    assert ae(v, (-3.4819106748728063576e-38 + 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -3.4819106748728063576e-38, tol=PTOL)
    assert ae(v.imag, 3.1415926535897932385, tol=PTOL)
    v = fp.ei((-4.6566128730773925781e-10 + 1.1641532182693481445e-10j))
    assert ae(v, (-20.880034622014215597 + 2.8966139905793444061j), tol=ATOL)
    assert ae(v.real, -20.880034622014215597, tol=PTOL)
    assert ae(v.imag, 2.8966139905793444061, tol=PTOL)
    v = fp.ei((-1.0 + 0.25j))
    assert ae(v, (-0.19731063945004229095 + 3.0542266078154932748j), tol=ATOL)
    assert ae(v.real, -0.19731063945004229095, tol=PTOL)
    assert ae(v.imag, 3.0542266078154932748, tol=PTOL)
    v = fp.ei((-4.0 + 1.0j))
    assert ae(v, (-0.0013106173980145506944 + 3.1381384055698581758j), tol=ATOL)
    assert ae(v.real, -0.0013106173980145506944, tol=PTOL)
    assert ae(v.imag, 3.1381384055698581758, tol=PTOL)
    v = fp.ei((-8.0 + 2.0j))
    assert ae(v, (0.000022278049065270225945 + 3.1415634616493367169j), tol=ATOL)
    assert ae(v.real, 0.000022278049065270225945, tol=PTOL)
    assert ae(v.imag, 3.1415634616493367169, tol=PTOL)
    v = fp.ei((-20.0 + 5.0j))
    assert ae(v, (-4.7711374515765346894e-11 + 3.1415926536726958909j), tol=ATOL)
    assert ae(v.real, -4.7711374515765346894e-11, tol=PTOL)
    assert ae(v.imag, 3.1415926536726958909, tol=PTOL)
    v = fp.ei((-80.0 + 20.0j))
    assert ae(v, (-3.8353473865788235787e-38 + 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -3.8353473865788235787e-38, tol=PTOL)
    assert ae(v.imag, 3.1415926535897932385, tol=PTOL)
    v = fp.ei((-120.0 + 30.0j))
    assert ae(v, (-2.3836002337480334716e-55 + 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -2.3836002337480334716e-55, tol=PTOL)
    assert ae(v.imag, 3.1415926535897932385, tol=PTOL)
    v = fp.ei((-160.0 + 40.0j))
    assert ae(v, (1.6238022898654510661e-72 + 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, 1.6238022898654510661e-72, tol=PTOL)
    assert ae(v.imag, 3.1415926535897932385, tol=PTOL)
    v = fp.ei((-200.0 + 50.0j))
    assert ae(v, (-6.6800061461666228487e-90 + 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -6.6800061461666228487e-90, tol=PTOL)
    assert ae(v.imag, 3.1415926535897932385, tol=PTOL)
    v = fp.ei((-320.0 + 80.0j))
    assert ae(v, (-4.2737871527778786157e-143 + 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -4.2737871527778786157e-143, tol=PTOL)
    assert ae(v.imag, 3.1415926535897932385, tol=PTOL)
    v = fp.ei(-1.1641532182693481445e-10)
    assert ae(v, -22.296641293693077672, tol=ATOL)
    assert type(v) is float
    v = fp.ei(-0.25)
    assert ae(v, -1.0442826344437381945, tol=ATOL)
    assert type(v) is float
    v = fp.ei(-1.0)
    assert ae(v, -0.21938393439552027368, tol=ATOL)
    assert type(v) is float
    v = fp.ei(-2.0)
    assert ae(v, -0.048900510708061119567, tol=ATOL)
    assert type(v) is float
    v = fp.ei(-5.0)
    assert ae(v, -0.0011482955912753257973, tol=ATOL)
    assert type(v) is float
    v = fp.ei(-20.0)
    assert ae(v, -9.8355252906498816904e-11, tol=ATOL)
    assert type(v) is float
    v = fp.ei(-30.0)
    assert ae(v, -3.0215520106888125448e-15, tol=ATOL)
    assert type(v) is float
    v = fp.ei(-40.0)
    assert ae(v, -1.0367732614516569722e-19, tol=ATOL)
    assert type(v) is float
    v = fp.ei(-50.0)
    assert ae(v, -3.7832640295504590187e-24, tol=ATOL)
    assert type(v) is float
    v = fp.ei(-80.0)
    assert ae(v, -2.2285432586884729112e-37, tol=ATOL)
    assert type(v) is float
    v = fp.ei((-1.1641532182693481445e-10 + 0.0j))
    assert ae(v, (-22.296641293693077672 + 0.0j), tol=ATOL)
    assert ae(v.real, -22.296641293693077672, tol=PTOL)
    assert v.imag == 0
    v = fp.ei((-0.25 + 0.0j))
    assert ae(v, (-1.0442826344437381945 + 0.0j), tol=ATOL)
    assert ae(v.real, -1.0442826344437381945, tol=PTOL)
    assert v.imag == 0
    v = fp.ei((-1.0 + 0.0j))
    assert ae(v, (-0.21938393439552027368 + 0.0j), tol=ATOL)
    assert ae(v.real, -0.21938393439552027368, tol=PTOL)
    assert v.imag == 0
    v = fp.ei((-2.0 + 0.0j))
    assert ae(v, (-0.048900510708061119567 + 0.0j), tol=ATOL)
    assert ae(v.real, -0.048900510708061119567, tol=PTOL)
    assert v.imag == 0
    v = fp.ei((-5.0 + 0.0j))
    assert ae(v, (-0.0011482955912753257973 + 0.0j), tol=ATOL)
    assert ae(v.real, -0.0011482955912753257973, tol=PTOL)
    assert v.imag == 0
    v = fp.ei((-20.0 + 0.0j))
    assert ae(v, (-9.8355252906498816904e-11 + 0.0j), tol=ATOL)
    assert ae(v.real, -9.8355252906498816904e-11, tol=PTOL)
    assert v.imag == 0
    v = fp.ei((-30.0 + 0.0j))
    assert ae(v, (-3.0215520106888125448e-15 + 0.0j), tol=ATOL)
    assert ae(v.real, -3.0215520106888125448e-15, tol=PTOL)
    assert v.imag == 0
    v = fp.ei((-40.0 + 0.0j))
    assert ae(v, (-1.0367732614516569722e-19 + 0.0j), tol=ATOL)
    assert ae(v.real, -1.0367732614516569722e-19, tol=PTOL)
    assert v.imag == 0
    v = fp.ei((-50.0 + 0.0j))
    assert ae(v, (-3.7832640295504590187e-24 + 0.0j), tol=ATOL)
    assert ae(v.real, -3.7832640295504590187e-24, tol=PTOL)
    assert v.imag == 0
    v = fp.ei((-80.0 + 0.0j))
    assert ae(v, (-2.2285432586884729112e-37 + 0.0j), tol=ATOL)
    assert ae(v.real, -2.2285432586884729112e-37, tol=PTOL)
    assert v.imag == 0
    v = fp.ei((-4.6566128730773925781e-10 - 1.1641532182693481445e-10j))
    assert ae(v, (-20.880034622014215597 - 2.8966139905793444061j), tol=ATOL)
    assert ae(v.real, -20.880034622014215597, tol=PTOL)
    assert ae(v.imag, -2.8966139905793444061, tol=PTOL)
    v = fp.ei((-1.0 - 0.25j))
    assert ae(v, (-0.19731063945004229095 - 3.0542266078154932748j), tol=ATOL)
    assert ae(v.real, -0.19731063945004229095, tol=PTOL)
    assert ae(v.imag, -3.0542266078154932748, tol=PTOL)
    v = fp.ei((-4.0 - 1.0j))
    assert ae(v, (-0.0013106173980145506944 - 3.1381384055698581758j), tol=ATOL)
    assert ae(v.real, -0.0013106173980145506944, tol=PTOL)
    assert ae(v.imag, -3.1381384055698581758, tol=PTOL)
    v = fp.ei((-8.0 - 2.0j))
    assert ae(v, (0.000022278049065270225945 - 3.1415634616493367169j), tol=ATOL)
    assert ae(v.real, 0.000022278049065270225945, tol=PTOL)
    assert ae(v.imag, -3.1415634616493367169, tol=PTOL)
    v = fp.ei((-20.0 - 5.0j))
    assert ae(v, (-4.7711374515765346894e-11 - 3.1415926536726958909j), tol=ATOL)
    assert ae(v.real, -4.7711374515765346894e-11, tol=PTOL)
    assert ae(v.imag, -3.1415926536726958909, tol=PTOL)
    v = fp.ei((-80.0 - 20.0j))
    assert ae(v, (-3.8353473865788235787e-38 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -3.8353473865788235787e-38, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.ei((-120.0 - 30.0j))
    assert ae(v, (-2.3836002337480334716e-55 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -2.3836002337480334716e-55, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.ei((-160.0 - 40.0j))
    assert ae(v, (1.6238022898654510661e-72 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, 1.6238022898654510661e-72, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.ei((-200.0 - 50.0j))
    assert ae(v, (-6.6800061461666228487e-90 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -6.6800061461666228487e-90, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.ei((-320.0 - 80.0j))
    assert ae(v, (-4.2737871527778786157e-143 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -4.2737871527778786157e-143, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.ei((-1.1641532182693481445e-10 - 1.1641532182693481445e-10j))
    assert ae(v, (-21.950067703413105017 - 2.3561944903087602507j), tol=ATOL)
    assert ae(v.real, -21.950067703413105017, tol=PTOL)
    assert ae(v.imag, -2.3561944903087602507, tol=PTOL)
    v = fp.ei((-0.25 - 0.25j))
    assert ae(v, (-0.71092525792923287894 - 2.5766745291767512913j), tol=ATOL)
    assert ae(v.real, -0.71092525792923287894, tol=PTOL)
    assert ae(v.imag, -2.5766745291767512913, tol=PTOL)
    v = fp.ei((-1.0 - 1.0j))
    assert ae(v, (-0.00028162445198141832551 - 2.9622681185504342983j), tol=ATOL)
    assert ae(v.real, -0.00028162445198141832551, tol=PTOL)
    assert ae(v.imag, -2.9622681185504342983, tol=PTOL)
    v = fp.ei((-2.0 - 2.0j))
    assert ae(v, (0.033767089606562004246 - 3.1229932394200426965j), tol=ATOL)
    assert ae(v.real, 0.033767089606562004246, tol=PTOL)
    assert ae(v.imag, -3.1229932394200426965, tol=PTOL)
    v = fp.ei((-5.0 - 5.0j))
    assert ae(v, (-0.0007266506660356393891 - 3.1420636813914284609j), tol=ATOL)
    assert ae(v.real, -0.0007266506660356393891, tol=PTOL)
    assert ae(v.imag, -3.1420636813914284609, tol=PTOL)
    v = fp.ei((-20.0 - 20.0j))
    assert ae(v, (2.3824537449367396579e-11 - 3.1415926535228233653j), tol=ATOL)
    assert ae(v.real, 2.3824537449367396579e-11, tol=PTOL)
    assert ae(v.imag, -3.1415926535228233653, tol=PTOL)
    v = fp.ei((-30.0 - 30.0j))
    assert ae(v, (-1.7316045841744061617e-15 - 3.141592653589794545j), tol=ATOL)
    assert ae(v.real, -1.7316045841744061617e-15, tol=PTOL)
    assert ae(v.imag, -3.141592653589794545, tol=PTOL)
    v = fp.ei((-40.0 - 40.0j))
    assert ae(v, (7.4001043002899232182e-20 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, 7.4001043002899232182e-20, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.ei((-50.0 - 50.0j))
    assert ae(v, (-2.3566128324644641219e-24 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -2.3566128324644641219e-24, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.ei((-80.0 - 80.0j))
    assert ae(v, (-9.8279750572186526673e-38 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -9.8279750572186526673e-38, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.ei((-1.1641532182693481445e-10 - 4.6566128730773925781e-10j))
    assert ae(v, (-20.880034621664969632 - 1.8157749903874220607j), tol=ATOL)
    assert ae(v.real, -20.880034621664969632, tol=PTOL)
    assert ae(v.imag, -1.8157749903874220607, tol=PTOL)
    v = fp.ei((-0.25 - 1.0j))
    assert ae(v, (0.16868306393667788761 - 2.6557914649950505414j), tol=ATOL)
    assert ae(v.real, 0.16868306393667788761, tol=PTOL)
    assert ae(v.imag, -2.6557914649950505414, tol=PTOL)
    v = fp.ei((-1.0 - 4.0j))
    assert ae(v, (-0.03373591813926547318 - 3.2151161058308770603j), tol=ATOL)
    assert ae(v.real, -0.03373591813926547318, tol=PTOL)
    assert ae(v.imag, -3.2151161058308770603, tol=PTOL)
    v = fp.ei((-2.0 - 8.0j))
    assert ae(v, (0.015392833434733785143 - 3.1384179414340326969j), tol=ATOL)
    assert ae(v.real, 0.015392833434733785143, tol=PTOL)
    assert ae(v.imag, -3.1384179414340326969, tol=PTOL)
    v = fp.ei((-5.0 - 20.0j))
    assert ae(v, (0.00024419662286542966525 - 3.1413825703601317109j), tol=ATOL)
    assert ae(v.real, 0.00024419662286542966525, tol=PTOL)
    assert ae(v.imag, -3.1413825703601317109, tol=PTOL)
    v = fp.ei((-20.0 - 80.0j))
    assert ae(v, (-2.3255552781051330088e-11 - 3.1415926535987396304j), tol=ATOL)
    assert ae(v.real, -2.3255552781051330088e-11, tol=PTOL)
    assert ae(v.imag, -3.1415926535987396304, tol=PTOL)
    v = fp.ei((-30.0 - 120.0j))
    assert ae(v, (2.7068919097124652332e-16 - 3.1415926535897925337j), tol=ATOL)
    assert ae(v.real, 2.7068919097124652332e-16, tol=PTOL)
    assert ae(v.imag, -3.1415926535897925337, tol=PTOL)
    v = fp.ei((-40.0 - 160.0j))
    assert ae(v, (1.1695597827678024687e-20 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, 1.1695597827678024687e-20, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.ei((-50.0 - 200.0j))
    assert ae(v, (-9.0323746914410162531e-25 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -9.0323746914410162531e-25, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.ei((-80.0 - 320.0j))
    assert ae(v, (-3.4819106748728063576e-38 - 3.1415926535897932385j), tol=ATOL)
    assert ae(v.real, -3.4819106748728063576e-38, tol=PTOL)
    assert ae(v.imag, -3.1415926535897932385, tol=PTOL)
    v = fp.ei((0.0 - 1.1641532182693481445e-10j))
    assert ae(v, (-22.29664129357666235 - 1.5707963269113119411j), tol=ATOL)
    assert ae(v.real, -22.29664129357666235, tol=PTOL)
    assert ae(v.imag, -1.5707963269113119411, tol=PTOL)
    v = fp.ei((0.0 - 0.25j))
    assert ae(v, (-0.82466306258094565309 - 1.8199298971146537833j), tol=ATOL)
    assert ae(v.real, -0.82466306258094565309, tol=PTOL)
    assert ae(v.imag, -1.8199298971146537833, tol=PTOL)
    v = fp.ei((0.0 - 1.0j))
    assert ae(v, (0.33740392290096813466 - 2.5168793971620796342j), tol=ATOL)
    assert ae(v.real, 0.33740392290096813466, tol=PTOL)
    assert ae(v.imag, -2.5168793971620796342, tol=PTOL)
    v = fp.ei((0.0 - 2.0j))
    assert ae(v, (0.4229808287748649957 - 3.1762093035975914678j), tol=ATOL)
    assert ae(v.real, 0.4229808287748649957, tol=PTOL)
    assert ae(v.imag, -3.1762093035975914678, tol=PTOL)
    v = fp.ei((0.0 - 5.0j))
    assert ae(v, (-0.19002974965664387862 - 3.1207275717395707565j), tol=ATOL)
    assert ae(v.real, -0.19002974965664387862, tol=PTOL)
    assert ae(v.imag, -3.1207275717395707565, tol=PTOL)
    v = fp.ei((0.0 - 20.0j))
    assert ae(v, (0.04441982084535331654 - 3.1190380278383364594j), tol=ATOL)
    assert ae(v.real, 0.04441982084535331654, tol=PTOL)
    assert ae(v.imag, -3.1190380278383364594, tol=PTOL)
    v = fp.ei((0.0 - 30.0j))
    assert ae(v, (-0.033032417282071143779 - 3.1375528668252477302j), tol=ATOL)
    assert ae(v.real, -0.033032417282071143779, tol=PTOL)
    assert ae(v.imag, -3.1375528668252477302, tol=PTOL)
    v = fp.ei((0.0 - 40.0j))
    assert ae(v, (0.019020007896208766962 - 3.157781446149681126j), tol=ATOL)
    assert ae(v.real, 0.019020007896208766962, tol=PTOL)
    assert ae(v.imag, -3.157781446149681126, tol=PTOL)
    v = fp.ei((0.0 - 50.0j))
    assert ae(v, (-0.0056283863241163054402 - 3.122413399280832514j), tol=ATOL)
    assert ae(v.real, -0.0056283863241163054402, tol=PTOL)
    assert ae(v.imag, -3.122413399280832514, tol=PTOL)
    v = fp.ei((0.0 - 80.0j))
    assert ae(v, (-0.012402501155070958192 - 3.1431272137073839346j), tol=ATOL)
    assert ae(v.real, -0.012402501155070958192, tol=PTOL)
    assert ae(v.imag, -3.1431272137073839346, tol=PTOL)
    v = fp.ei((1.1641532182693481445e-10 - 4.6566128730773925781e-10j))
    assert ae(v, (-20.880034621432138988 - 1.3258176641336937524j), tol=ATOL)
    assert ae(v.real, -20.880034621432138988, tol=PTOL)
    assert ae(v.imag, -1.3258176641336937524, tol=PTOL)
    v = fp.ei((0.25 - 1.0j))
    assert ae(v, (0.59066621214766308594 - 2.3968481059377428687j), tol=ATOL)
    assert ae(v.real, 0.59066621214766308594, tol=PTOL)
    assert ae(v.imag, -2.3968481059377428687, tol=PTOL)
    v = fp.ei((1.0 - 4.0j))
    assert ae(v, (-0.49739047283060471093 - 3.5570287076301818702j), tol=ATOL)
    assert ae(v.real, -0.49739047283060471093, tol=PTOL)
    assert ae(v.imag, -3.5570287076301818702, tol=PTOL)
    v = fp.ei((2.0 - 8.0j))
    assert ae(v, (0.8705211147733730969 - 3.3825859385758486351j), tol=ATOL)
    assert ae(v.real, 0.8705211147733730969, tol=PTOL)
    assert ae(v.imag, -3.3825859385758486351, tol=PTOL)
    v = fp.ei((5.0 - 20.0j))
    assert ae(v, (7.0789514293925893007 - 1.5313749363937141849j), tol=ATOL)
    assert ae(v.real, 7.0789514293925893007, tol=PTOL)
    assert ae(v.imag, -1.5313749363937141849, tol=PTOL)
    v = fp.ei((20.0 - 80.0j))
    assert ae(v, (-5855431.4907298084434 + 720917.79156143806727j), tol=ATOL)
    assert ae(v.real, -5855431.4907298084434, tol=PTOL)
    assert ae(v.imag, 720917.79156143806727, tol=PTOL)
    v = fp.ei((30.0 - 120.0j))
    assert ae(v, (65402491644.703470747 + 56697658396.51586764j), tol=ATOL)
    assert ae(v.real, 65402491644.703470747, tol=PTOL)
    assert ae(v.imag, 56697658396.51586764, tol=PTOL)
    v = fp.ei((40.0 - 160.0j))
    assert ae(v, (-25504929379604.776769 - 1429035198630576.3879j), tol=ATOL)
    assert ae(v.real, -25504929379604.776769, tol=PTOL)
    assert ae(v.imag, -1429035198630576.3879, tol=PTOL)
    v = fp.ei((50.0 - 200.0j))
    assert ae(v, (-18437746526988116954.0 + 17146362239046152342.0j), tol=ATOL)
    assert ae(v.real, -18437746526988116954.0, tol=PTOL)
    assert ae(v.imag, 17146362239046152342.0, tol=PTOL)
    v = fp.ei((80.0 - 320.0j))
    assert ae(v, (-3.3464697299634526706e+31 + 1.6473152633843023919e+32j), tol=ATOL)
    assert ae(v.real, -3.3464697299634526706e+31, tol=PTOL)
    assert ae(v.imag, 1.6473152633843023919e+32, tol=PTOL)
    v = fp.ei((1.1641532182693481445e-10 - 1.1641532182693481445e-10j))
    assert ae(v, (-21.950067703180274374 - 0.78539816351386363145j), tol=ATOL)
    assert ae(v.real, -21.950067703180274374, tol=PTOL)
    assert ae(v.imag, -0.78539816351386363145, tol=PTOL)
    v = fp.ei((0.25 - 0.25j))
    assert ae(v, (-0.21441047326710323254 - 1.0683772981589995996j), tol=ATOL)
    assert ae(v.real, -0.21441047326710323254, tol=PTOL)
    assert ae(v.imag, -1.0683772981589995996, tol=PTOL)
    v = fp.ei((1.0 - 1.0j))
    assert ae(v, (1.7646259855638540684 - 2.3877698515105224193j), tol=ATOL)
    assert ae(v.real, 1.7646259855638540684, tol=PTOL)
    assert ae(v.imag, -2.3877698515105224193, tol=PTOL)
    v = fp.ei((2.0 - 2.0j))
    assert ae(v, (1.8920781621855474089 - 5.3169624378326579621j), tol=ATOL)
    assert ae(v.real, 1.8920781621855474089, tol=PTOL)
    assert ae(v.imag, -5.3169624378326579621, tol=PTOL)
    v = fp.ei((5.0 - 5.0j))
    assert ae(v, (-13.470936071475245856 + 15.322492395731230968j), tol=ATOL)
    assert ae(v.real, -13.470936071475245856, tol=PTOL)
    assert ae(v.imag, 15.322492395731230968, tol=PTOL)
    v = fp.ei((20.0 - 20.0j))
    assert ae(v, (16589317.398788971896 - 5831705.4712368307104j), tol=ATOL)
    assert ae(v.real, 16589317.398788971896, tol=PTOL)
    assert ae(v.imag, -5831705.4712368307104, tol=PTOL)
    v = fp.ei((30.0 - 30.0j))
    assert ae(v, (-154596484273.69322527 + 204179357834.2723043j), tol=ATOL)
    assert ae(v.real, -154596484273.69322527, tol=PTOL)
    assert ae(v.imag, 204179357834.2723043, tol=PTOL)
    v = fp.ei((40.0 - 40.0j))
    assert ae(v, (287512180321448.45408 - 4203502407932318.1156j), tol=ATOL)
    assert ae(v.real, 287512180321448.45408, tol=PTOL)
    assert ae(v.imag, -4203502407932318.1156, tol=PTOL)
    v = fp.ei((50.0 - 50.0j))
    assert ae(v, (36128528616649268826.0 + 64648801861338741960.0j), tol=ATOL)
    assert ae(v.real, 36128528616649268826.0, tol=PTOL)
    assert ae(v.imag, 64648801861338741960.0, tol=PTOL)
    v = fp.ei((80.0 - 80.0j))
    assert ae(v, (-3.8674816337930010217e+32 + 3.0540709639658071041e+32j), tol=ATOL)
    assert ae(v.real, -3.8674816337930010217e+32, tol=PTOL)
    assert ae(v.imag, 3.0540709639658071041e+32, tol=PTOL)
    v = fp.ei((4.6566128730773925781e-10 - 1.1641532182693481445e-10j))
    assert ae(v, (-20.880034621082893023 - 0.24497866324327947603j), tol=ATOL)
    assert ae(v.real, -20.880034621082893023, tol=PTOL)
    assert ae(v.imag, -0.24497866324327947603, tol=PTOL)
    v = fp.ei((1.0 - 0.25j))
    assert ae(v, (1.8942716983721074932 - 0.67268237088273915854j), tol=ATOL)
    assert ae(v.real, 1.8942716983721074932, tol=PTOL)
    assert ae(v.imag, -0.67268237088273915854, tol=PTOL)
    v = fp.ei((4.0 - 1.0j))
    assert ae(v, (14.806699492675420438 - 12.280015176673582616j), tol=ATOL)
    assert ae(v.real, 14.806699492675420438, tol=PTOL)
    assert ae(v.imag, -12.280015176673582616, tol=PTOL)
    v = fp.ei((8.0 - 2.0j))
    assert ae(v, (-54.633252667426386294 - 416.34477429173650012j), tol=ATOL)
    assert ae(v.real, -54.633252667426386294, tol=PTOL)
    assert ae(v.imag, -416.34477429173650012, tol=PTOL)
    v = fp.ei((20.0 - 5.0j))
    assert ae(v, (711836.97165402624643 + 24745247.798103247366j), tol=ATOL)
    assert ae(v.real, 711836.97165402624643, tol=PTOL)
    assert ae(v.imag, 24745247.798103247366, tol=PTOL)
    v = fp.ei((80.0 - 20.0j))
    assert ae(v, (4.2139911108612653091e+32 - 5.3367124741918251637e+32j), tol=ATOL)
    assert ae(v.real, 4.2139911108612653091e+32, tol=PTOL)
    assert ae(v.imag, -5.3367124741918251637e+32, tol=PTOL)
    v = fp.ei((120.0 - 30.0j))
    assert ae(v, (-9.7760616203707508892e+48 + 1.058257682317195792e+50j), tol=ATOL)
    assert ae(v.real, -9.7760616203707508892e+48, tol=PTOL)
    assert ae(v.imag, 1.058257682317195792e+50, tol=PTOL)
    v = fp.ei((160.0 - 40.0j))
    assert ae(v, (-8.7065541466623638861e+66 - 1.6577106725141739889e+67j), tol=ATOL)
    assert ae(v.real, -8.7065541466623638861e+66, tol=PTOL)
    assert ae(v.imag, -1.6577106725141739889e+67, tol=PTOL)
    v = fp.ei((200.0 - 50.0j))
    assert ae(v, (3.070744996327018106e+84 + 1.7243244846769415903e+84j), tol=ATOL)
    assert ae(v.real, 3.070744996327018106e+84, tol=PTOL)
    assert ae(v.imag, 1.7243244846769415903e+84, tol=PTOL)
    v = fp.ei((320.0 - 80.0j))
    assert ae(v, (-9.9960598637998647276e+135 + 2.6855081527595608863e+136j), tol=ATOL)
    assert ae(v.real, -9.9960598637998647276e+135, tol=PTOL)
    assert ae(v.imag, 2.6855081527595608863e+136, tol=PTOL)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_replace.py ===
from __future__ import annotations

from datetime import datetime
import re

import numpy as np
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


@pytest.fixture
def mix_ab() -> dict[str, list[int | str]]:
    return {"a": list(range(4)), "b": list("ab..")}


@pytest.fixture
def mix_abc() -> dict[str, list[float | str]]:
    return {"a": list(range(4)), "b": list("ab.."), "c": ["a", "b", np.nan, "d"]}


class TestDataFrameReplace:
    def test_replace_inplace(self, datetime_frame, float_string_frame):
        datetime_frame.loc[datetime_frame.index[:5], "A"] = np.nan
        datetime_frame.loc[datetime_frame.index[-5:], "A"] = np.nan

        tsframe = datetime_frame.copy()
        return_value = tsframe.replace(np.nan, 0, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(tsframe, datetime_frame.fillna(0))

        # mixed type
        mf = float_string_frame
        mf.iloc[5:20, mf.columns.get_loc("foo")] = np.nan
        mf.iloc[-10:, mf.columns.get_loc("A")] = np.nan

        result = float_string_frame.replace(np.nan, 0)
        expected = float_string_frame.copy()
        expected["foo"] = expected["foo"].astype(object)
        expected = expected.fillna(value=0)
        tm.assert_frame_equal(result, expected)

        tsframe = datetime_frame.copy()
        return_value = tsframe.replace([np.nan], [0], inplace=True)
        assert return_value is None
        tm.assert_frame_equal(tsframe, datetime_frame.fillna(0))

    @pytest.mark.parametrize(
        "to_replace,values,expected",
        [
            # lists of regexes and values
            # list of [re1, re2, ..., reN] -> [v1, v2, ..., vN]
            (
                [r"\s*\.\s*", r"e|f|g"],
                [np.nan, "crap"],
                {
                    "a": ["a", "b", np.nan, np.nan],
                    "b": ["crap"] * 3 + ["h"],
                    "c": ["h", "crap", "l", "o"],
                },
            ),
            # list of [re1, re2, ..., reN] -> [re1, re2, .., reN]
            (
                [r"\s*(\.)\s*", r"(e|f|g)"],
                [r"\1\1", r"\1_crap"],
                {
                    "a": ["a", "b", "..", ".."],
                    "b": ["e_crap", "f_crap", "g_crap", "h"],
                    "c": ["h", "e_crap", "l", "o"],
                },
            ),
            # list of [re1, re2, ..., reN] -> [(re1 or v1), (re2 or v2), ..., (reN
            # or vN)]
            (
                [r"\s*(\.)\s*", r"e"],
                [r"\1\1", r"crap"],
                {
                    "a": ["a", "b", "..", ".."],
                    "b": ["crap", "f", "g", "h"],
                    "c": ["h", "crap", "l", "o"],
                },
            ),
        ],
    )
    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("use_value_regex_args", [True, False])
    def test_regex_replace_list_obj(
        self, to_replace, values, expected, inplace, use_value_regex_args
    ):
        df = DataFrame({"a": list("ab.."), "b": list("efgh"), "c": list("helo")})

        if use_value_regex_args:
            result = df.replace(value=values, regex=to_replace, inplace=inplace)
        else:
            result = df.replace(to_replace, values, regex=True, inplace=inplace)

        if inplace:
            assert result is None
            result = df

        expected = DataFrame(expected)
        tm.assert_frame_equal(result, expected)

    def test_regex_replace_list_mixed(self, mix_ab):
        # mixed frame to make sure this doesn't break things
        dfmix = DataFrame(mix_ab)

        # lists of regexes and values
        # list of [re1, re2, ..., reN] -> [v1, v2, ..., vN]
        to_replace_res = [r"\s*\.\s*", r"a"]
        values = [np.nan, "crap"]
        mix2 = {"a": list(range(4)), "b": list("ab.."), "c": list("halo")}
        dfmix2 = DataFrame(mix2)
        res = dfmix2.replace(to_replace_res, values, regex=True)
        expec = DataFrame(
            {
                "a": mix2["a"],
                "b": ["crap", "b", np.nan, np.nan],
                "c": ["h", "crap", "l", "o"],
            }
        )
        tm.assert_frame_equal(res, expec)

        # list of [re1, re2, ..., reN] -> [re1, re2, .., reN]
        to_replace_res = [r"\s*(\.)\s*", r"(a|b)"]
        values = [r"\1\1", r"\1_crap"]
        res = dfmix.replace(to_replace_res, values, regex=True)
        expec = DataFrame({"a": mix_ab["a"], "b": ["a_crap", "b_crap", "..", ".."]})
        tm.assert_frame_equal(res, expec)

        # list of [re1, re2, ..., reN] -> [(re1 or v1), (re2 or v2), ..., (reN
        # or vN)]
        to_replace_res = [r"\s*(\.)\s*", r"a", r"(b)"]
        values = [r"\1\1", r"crap", r"\1_crap"]
        res = dfmix.replace(to_replace_res, values, regex=True)
        expec = DataFrame({"a": mix_ab["a"], "b": ["crap", "b_crap", "..", ".."]})
        tm.assert_frame_equal(res, expec)

        to_replace_res = [r"\s*(\.)\s*", r"a", r"(b)"]
        values = [r"\1\1", r"crap", r"\1_crap"]
        res = dfmix.replace(regex=to_replace_res, value=values)
        expec = DataFrame({"a": mix_ab["a"], "b": ["crap", "b_crap", "..", ".."]})
        tm.assert_frame_equal(res, expec)

    def test_regex_replace_list_mixed_inplace(self, mix_ab):
        dfmix = DataFrame(mix_ab)
        # the same inplace
        # lists of regexes and values
        # list of [re1, re2, ..., reN] -> [v1, v2, ..., vN]
        to_replace_res = [r"\s*\.\s*", r"a"]
        values = [np.nan, "crap"]
        res = dfmix.copy()
        return_value = res.replace(to_replace_res, values, inplace=True, regex=True)
        assert return_value is None
        expec = DataFrame({"a": mix_ab["a"], "b": ["crap", "b", np.nan, np.nan]})
        tm.assert_frame_equal(res, expec)

        # list of [re1, re2, ..., reN] -> [re1, re2, .., reN]
        to_replace_res = [r"\s*(\.)\s*", r"(a|b)"]
        values = [r"\1\1", r"\1_crap"]
        res = dfmix.copy()
        return_value = res.replace(to_replace_res, values, inplace=True, regex=True)
        assert return_value is None
        expec = DataFrame({"a": mix_ab["a"], "b": ["a_crap", "b_crap", "..", ".."]})
        tm.assert_frame_equal(res, expec)

        # list of [re1, re2, ..., reN] -> [(re1 or v1), (re2 or v2), ..., (reN
        # or vN)]
        to_replace_res = [r"\s*(\.)\s*", r"a", r"(b)"]
        values = [r"\1\1", r"crap", r"\1_crap"]
        res = dfmix.copy()
        return_value = res.replace(to_replace_res, values, inplace=True, regex=True)
        assert return_value is None
        expec = DataFrame({"a": mix_ab["a"], "b": ["crap", "b_crap", "..", ".."]})
        tm.assert_frame_equal(res, expec)

        to_replace_res = [r"\s*(\.)\s*", r"a", r"(b)"]
        values = [r"\1\1", r"crap", r"\1_crap"]
        res = dfmix.copy()
        return_value = res.replace(regex=to_replace_res, value=values, inplace=True)
        assert return_value is None
        expec = DataFrame({"a": mix_ab["a"], "b": ["crap", "b_crap", "..", ".."]})
        tm.assert_frame_equal(res, expec)

    def test_regex_replace_dict_mixed(self, mix_abc):
        dfmix = DataFrame(mix_abc)

        # dicts
        # single dict {re1: v1}, search the whole frame
        # need test for this...

        # list of dicts {re1: v1, re2: v2, ..., re3: v3}, search the whole
        # frame
        res = dfmix.replace({"b": r"\s*\.\s*"}, {"b": np.nan}, regex=True)
        res2 = dfmix.copy()
        return_value = res2.replace(
            {"b": r"\s*\.\s*"}, {"b": np.nan}, inplace=True, regex=True
        )
        assert return_value is None
        expec = DataFrame(
            {"a": mix_abc["a"], "b": ["a", "b", np.nan, np.nan], "c": mix_abc["c"]}
        )
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)

        # list of dicts {re1: re11, re2: re12, ..., reN: re1N}, search the
        # whole frame
        res = dfmix.replace({"b": r"\s*(\.)\s*"}, {"b": r"\1ty"}, regex=True)
        res2 = dfmix.copy()
        return_value = res2.replace(
            {"b": r"\s*(\.)\s*"}, {"b": r"\1ty"}, inplace=True, regex=True
        )
        assert return_value is None
        expec = DataFrame(
            {"a": mix_abc["a"], "b": ["a", "b", ".ty", ".ty"], "c": mix_abc["c"]}
        )
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)

        res = dfmix.replace(regex={"b": r"\s*(\.)\s*"}, value={"b": r"\1ty"})
        res2 = dfmix.copy()
        return_value = res2.replace(
            regex={"b": r"\s*(\.)\s*"}, value={"b": r"\1ty"}, inplace=True
        )
        assert return_value is None
        expec = DataFrame(
            {"a": mix_abc["a"], "b": ["a", "b", ".ty", ".ty"], "c": mix_abc["c"]}
        )
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)

        # scalar -> dict
        # to_replace regex, {value: value}
        expec = DataFrame(
            {"a": mix_abc["a"], "b": [np.nan, "b", ".", "."], "c": mix_abc["c"]}
        )
        res = dfmix.replace("a", {"b": np.nan}, regex=True)
        res2 = dfmix.copy()
        return_value = res2.replace("a", {"b": np.nan}, regex=True, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)

        res = dfmix.replace("a", {"b": np.nan}, regex=True)
        res2 = dfmix.copy()
        return_value = res2.replace(regex="a", value={"b": np.nan}, inplace=True)
        assert return_value is None
        expec = DataFrame(
            {"a": mix_abc["a"], "b": [np.nan, "b", ".", "."], "c": mix_abc["c"]}
        )
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)

    def test_regex_replace_dict_nested(self, mix_abc):
        # nested dicts will not work until this is implemented for Series
        dfmix = DataFrame(mix_abc)
        res = dfmix.replace({"b": {r"\s*\.\s*": np.nan}}, regex=True)
        res2 = dfmix.copy()
        res4 = dfmix.copy()
        return_value = res2.replace(
            {"b": {r"\s*\.\s*": np.nan}}, inplace=True, regex=True
        )
        assert return_value is None
        res3 = dfmix.replace(regex={"b": {r"\s*\.\s*": np.nan}})
        return_value = res4.replace(regex={"b": {r"\s*\.\s*": np.nan}}, inplace=True)
        assert return_value is None
        expec = DataFrame(
            {"a": mix_abc["a"], "b": ["a", "b", np.nan, np.nan], "c": mix_abc["c"]}
        )
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)
        tm.assert_frame_equal(res3, expec)
        tm.assert_frame_equal(res4, expec)

    def test_regex_replace_dict_nested_non_first_character(self, any_string_dtype):
        # GH 25259
        dtype = any_string_dtype
        df = DataFrame({"first": ["abc", "bca", "cab"]}, dtype=dtype)
        result = df.replace({"a": "."}, regex=True)
        expected = DataFrame({"first": [".bc", "bc.", "c.b"]}, dtype=dtype)
        tm.assert_frame_equal(result, expected)

    def test_regex_replace_dict_nested_gh4115(self):
        df = DataFrame(
            {"Type": Series(["Q", "T", "Q", "Q", "T"], dtype=object), "tmp": 2}
        )
        expected = DataFrame({"Type": [0, 1, 0, 0, 1], "tmp": 2})
        msg = "Downcasting behavior in `replace`"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.replace({"Type": {"Q": 0, "T": 1}})

        tm.assert_frame_equal(result, expected)

    def test_regex_replace_list_to_scalar(self, mix_abc, using_infer_string):
        df = DataFrame(mix_abc)
        expec = DataFrame(
            {
                "a": mix_abc["a"],
                "b": [np.nan] * 4,
                "c": [np.nan, np.nan, np.nan, "d"],
            }
        )
        if using_infer_string:
            expec["b"] = expec["b"].astype("str")
        msg = "Downcasting behavior in `replace`"
        warn = None if using_infer_string else FutureWarning
        with tm.assert_produces_warning(warn, match=msg):
            res = df.replace([r"\s*\.\s*", "a|b"], np.nan, regex=True)
        res2 = df.copy()
        res3 = df.copy()
        with tm.assert_produces_warning(warn, match=msg):
            return_value = res2.replace(
                [r"\s*\.\s*", "a|b"], np.nan, regex=True, inplace=True
            )
        assert return_value is None
        with tm.assert_produces_warning(warn, match=msg):
            return_value = res3.replace(
                regex=[r"\s*\.\s*", "a|b"], value=np.nan, inplace=True
            )
        assert return_value is None
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)
        tm.assert_frame_equal(res3, expec)

    def test_regex_replace_str_to_numeric(self, mix_abc):
        # what happens when you try to replace a numeric value with a regex?
        df = DataFrame(mix_abc)
        res = df.replace(r"\s*\.\s*", 0, regex=True)
        res2 = df.copy()
        return_value = res2.replace(r"\s*\.\s*", 0, inplace=True, regex=True)
        assert return_value is None
        res3 = df.copy()
        return_value = res3.replace(regex=r"\s*\.\s*", value=0, inplace=True)
        assert return_value is None
        expec = DataFrame({"a": mix_abc["a"], "b": ["a", "b", 0, 0], "c": mix_abc["c"]})
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)
        tm.assert_frame_equal(res3, expec)

    def test_regex_replace_regex_list_to_numeric(self, mix_abc):
        df = DataFrame(mix_abc)
        res = df.replace([r"\s*\.\s*", "b"], 0, regex=True)
        res2 = df.copy()
        return_value = res2.replace([r"\s*\.\s*", "b"], 0, regex=True, inplace=True)
        assert return_value is None
        res3 = df.copy()
        return_value = res3.replace(regex=[r"\s*\.\s*", "b"], value=0, inplace=True)
        assert return_value is None
        expec = DataFrame(
            {"a": mix_abc["a"], "b": ["a", 0, 0, 0], "c": ["a", 0, np.nan, "d"]}
        )
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)
        tm.assert_frame_equal(res3, expec)

    def test_regex_replace_series_of_regexes(self, mix_abc):
        df = DataFrame(mix_abc)
        s1 = Series({"b": r"\s*\.\s*"})
        s2 = Series({"b": np.nan})
        res = df.replace(s1, s2, regex=True)
        res2 = df.copy()
        return_value = res2.replace(s1, s2, inplace=True, regex=True)
        assert return_value is None
        res3 = df.copy()
        return_value = res3.replace(regex=s1, value=s2, inplace=True)
        assert return_value is None
        expec = DataFrame(
            {"a": mix_abc["a"], "b": ["a", "b", np.nan, np.nan], "c": mix_abc["c"]}
        )
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)
        tm.assert_frame_equal(res3, expec)

    def test_regex_replace_numeric_to_object_conversion(self, mix_abc):
        df = DataFrame(mix_abc)
        expec = DataFrame({"a": ["a", 1, 2, 3], "b": mix_abc["b"], "c": mix_abc["c"]})
        res = df.replace(0, "a")
        tm.assert_frame_equal(res, expec)
        assert res.a.dtype == np.object_

    @pytest.mark.parametrize(
        "to_replace", [{"": np.nan, ",": ""}, {",": "", "": np.nan}]
    )
    def test_joint_simple_replace_and_regex_replace(self, to_replace):
        # GH-39338
        df = DataFrame(
            {
                "col1": ["1,000", "a", "3"],
                "col2": ["a", "", "b"],
                "col3": ["a", "b", "c"],
            }
        )
        result = df.replace(regex=to_replace)
        expected = DataFrame(
            {
                "col1": ["1000", "a", "3"],
                "col2": ["a", np.nan, "b"],
                "col3": ["a", "b", "c"],
            }
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("metachar", ["[]", "()", r"\d", r"\w", r"\s"])
    def test_replace_regex_metachar(self, metachar):
        df = DataFrame({"a": [metachar, "else"]})
        result = df.replace({"a": {metachar: "paren"}})
        expected = DataFrame({"a": ["paren", "else"]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "data,to_replace,expected",
        [
            (["xax", "xbx"], {"a": "c", "b": "d"}, ["xcx", "xdx"]),
            (["d", "", ""], {r"^\s*$": pd.NA}, ["d", pd.NA, pd.NA]),
        ],
    )
    def test_regex_replace_string_types(
        self, data, to_replace, expected, frame_or_series, any_string_dtype
    ):
        # GH-41333, GH-35977
        dtype = any_string_dtype
        obj = frame_or_series(data, dtype=dtype)
        result = obj.replace(to_replace, regex=True)
        expected = frame_or_series(expected, dtype=dtype)

        tm.assert_equal(result, expected)

    def test_replace(self, datetime_frame):
        datetime_frame.loc[datetime_frame.index[:5], "A"] = np.nan
        datetime_frame.loc[datetime_frame.index[-5:], "A"] = np.nan

        zero_filled = datetime_frame.replace(np.nan, -1e8)
        tm.assert_frame_equal(zero_filled, datetime_frame.fillna(-1e8))
        tm.assert_frame_equal(zero_filled.replace(-1e8, np.nan), datetime_frame)

        datetime_frame.loc[datetime_frame.index[:5], "A"] = np.nan
        datetime_frame.loc[datetime_frame.index[-5:], "A"] = np.nan
        datetime_frame.loc[datetime_frame.index[:5], "B"] = -1e8

        # empty
        df = DataFrame(index=["a", "b"])
        tm.assert_frame_equal(df, df.replace(5, 7))

        # GH 11698
        # test for mixed data types.
        df = DataFrame(
            [("-", pd.to_datetime("20150101")), ("a", pd.to_datetime("20150102"))]
        )
        df1 = df.replace("-", np.nan)
        expected_df = DataFrame(
            [(np.nan, pd.to_datetime("20150101")), ("a", pd.to_datetime("20150102"))]
        )
        tm.assert_frame_equal(df1, expected_df)

    def test_replace_list(self):
        obj = {"a": list("ab.."), "b": list("efgh"), "c": list("helo")}
        dfobj = DataFrame(obj)

        # lists of regexes and values
        # list of [v1, v2, ..., vN] -> [v1, v2, ..., vN]
        to_replace_res = [r".", r"e"]
        values = [np.nan, "crap"]
        res = dfobj.replace(to_replace_res, values)
        expec = DataFrame(
            {
                "a": ["a", "b", np.nan, np.nan],
                "b": ["crap", "f", "g", "h"],
                "c": ["h", "crap", "l", "o"],
            }
        )
        tm.assert_frame_equal(res, expec)

        # list of [v1, v2, ..., vN] -> [v1, v2, .., vN]
        to_replace_res = [r".", r"f"]
        values = [r"..", r"crap"]
        res = dfobj.replace(to_replace_res, values)
        expec = DataFrame(
            {
                "a": ["a", "b", "..", ".."],
                "b": ["e", "crap", "g", "h"],
                "c": ["h", "e", "l", "o"],
            }
        )
        tm.assert_frame_equal(res, expec)

    def test_replace_with_empty_list(self, frame_or_series):
        # GH 21977
        ser = Series([["a", "b"], [], np.nan, [1]])
        obj = DataFrame({"col": ser})
        obj = tm.get_obj(obj, frame_or_series)
        expected = obj
        result = obj.replace([], np.nan)
        tm.assert_equal(result, expected)

        # GH 19266
        msg = (
            "NumPy boolean array indexing assignment cannot assign {size} "
            "input values to the 1 output values where the mask is true"
        )
        with pytest.raises(ValueError, match=msg.format(size=0)):
            obj.replace({np.nan: []})
        with pytest.raises(ValueError, match=msg.format(size=2)):
            obj.replace({np.nan: ["dummy", "alt"]})

    def test_replace_series_dict(self):
        # from GH 3064
        df = DataFrame({"zero": {"a": 0.0, "b": 1}, "one": {"a": 2.0, "b": 0}})
        result = df.replace(0, {"zero": 0.5, "one": 1.0})
        expected = DataFrame({"zero": {"a": 0.5, "b": 1}, "one": {"a": 2.0, "b": 1.0}})
        tm.assert_frame_equal(result, expected)

        result = df.replace(0, df.mean())
        tm.assert_frame_equal(result, expected)

        # series to series/dict
        df = DataFrame({"zero": {"a": 0.0, "b": 1}, "one": {"a": 2.0, "b": 0}})
        s = Series({"zero": 0.0, "one": 2.0})
        result = df.replace(s, {"zero": 0.5, "one": 1.0})
        expected = DataFrame({"zero": {"a": 0.5, "b": 1}, "one": {"a": 1.0, "b": 0.0}})
        tm.assert_frame_equal(result, expected)

        result = df.replace(s, df.mean())
        tm.assert_frame_equal(result, expected)

    def test_replace_convert(self):
        # gh 3907
        df = DataFrame([["foo", "bar", "bah"], ["bar", "foo", "bah"]])
        m = {"foo": 1, "bar": 2, "bah": 3}
        msg = "Downcasting behavior in `replace` "
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rep = df.replace(m)
        expec = Series([np.int64] * 3)
        res = rep.dtypes
        tm.assert_series_equal(expec, res)

    def test_replace_mixed(self, float_string_frame):
        mf = float_string_frame
        mf.iloc[5:20, mf.columns.get_loc("foo")] = np.nan
        mf.iloc[-10:, mf.columns.get_loc("A")] = np.nan

        result = float_string_frame.replace(np.nan, -18)
        expected = float_string_frame.copy()
        expected["foo"] = expected["foo"].astype(object)
        expected = expected.fillna(value=-18)
        tm.assert_frame_equal(result, expected)
        expected2 = float_string_frame.copy()
        expected2["foo"] = expected2["foo"].astype(object)
        tm.assert_frame_equal(result.replace(-18, np.nan), expected2)

        result = float_string_frame.replace(np.nan, -1e8)
        expected = float_string_frame.copy()
        expected["foo"] = expected["foo"].astype(object)
        expected = expected.fillna(value=-1e8)
        tm.assert_frame_equal(result, expected)
        expected2 = float_string_frame.copy()
        expected2["foo"] = expected2["foo"].astype(object)
        tm.assert_frame_equal(result.replace(-1e8, np.nan), expected2)

    def test_replace_mixed_int_block_upcasting(self):
        # int block upcasting
        df = DataFrame(
            {
                "A": Series([1.0, 2.0], dtype="float64"),
                "B": Series([0, 1], dtype="int64"),
            }
        )
        expected = DataFrame(
            {
                "A": Series([1.0, 2.0], dtype="float64"),
                "B": Series([0.5, 1], dtype="float64"),
            }
        )
        result = df.replace(0, 0.5)
        tm.assert_frame_equal(result, expected)

        return_value = df.replace(0, 0.5, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(df, expected)

    def test_replace_mixed_int_block_splitting(self):
        # int block splitting
        df = DataFrame(
            {
                "A": Series([1.0, 2.0], dtype="float64"),
                "B": Series([0, 1], dtype="int64"),
                "C": Series([1, 2], dtype="int64"),
            }
        )
        expected = DataFrame(
            {
                "A": Series([1.0, 2.0], dtype="float64"),
                "B": Series([0.5, 1], dtype="float64"),
                "C": Series([1, 2], dtype="int64"),
            }
        )
        result = df.replace(0, 0.5)
        tm.assert_frame_equal(result, expected)

    def test_replace_mixed2(self, using_infer_string):
        # to object block upcasting
        df = DataFrame(
            {
                "A": Series([1.0, 2.0], dtype="float64"),
                "B": Series([0, 1], dtype="int64"),
            }
        )
        expected = DataFrame(
            {
                "A": Series([1, "foo"], dtype="object"),
                "B": Series([0, 1], dtype="int64"),
            }
        )
        result = df.replace(2, "foo")
        tm.assert_frame_equal(result, expected)

        expected = DataFrame(
            {
                "A": Series(["foo", "bar"], dtype="object"),
                "B": Series([0, "foo"], dtype="object"),
            }
        )
        result = df.replace([1, 2], ["foo", "bar"])
        tm.assert_frame_equal(result, expected)

    def test_replace_mixed3(self):
        # test case from
        df = DataFrame(
            {"A": Series([3, 0], dtype="int64"), "B": Series([0, 3], dtype="int64")}
        )
        result = df.replace(3, df.mean().to_dict())
        expected = df.copy().astype("float64")
        m = df.mean()
        expected.iloc[0, 0] = m.iloc[0]
        expected.iloc[1, 1] = m.iloc[1]
        tm.assert_frame_equal(result, expected)

    def test_replace_nullable_int_with_string_doesnt_cast(self):
        # GH#25438 don't cast df['a'] to float64
        df = DataFrame({"a": [1, 2, 3, np.nan], "b": ["some", "strings", "here", "he"]})
        df["a"] = df["a"].astype("Int64")

        res = df.replace("", np.nan)
        tm.assert_series_equal(res["a"], df["a"])

    @pytest.mark.parametrize("dtype", ["boolean", "Int64", "Float64"])
    def test_replace_with_nullable_column(self, dtype):
        # GH-44499
        nullable_ser = Series([1, 0, 1], dtype=dtype)
        df = DataFrame({"A": ["A", "B", "x"], "B": nullable_ser})
        result = df.replace("x", "X")
        expected = DataFrame({"A": ["A", "B", "X"], "B": nullable_ser})
        tm.assert_frame_equal(result, expected)

    def test_replace_simple_nested_dict(self):
        df = DataFrame({"col": range(1, 5)})
        expected = DataFrame({"col": ["a", 2, 3, "b"]})

        result = df.replace({"col": {1: "a", 4: "b"}})
        tm.assert_frame_equal(expected, result)

        # in this case, should be the same as the not nested version
        result = df.replace({1: "a", 4: "b"})
        tm.assert_frame_equal(expected, result)

    def test_replace_simple_nested_dict_with_nonexistent_value(self):
        df = DataFrame({"col": range(1, 5)})
        expected = DataFrame({"col": ["a", 2, 3, "b"]})

        result = df.replace({-1: "-", 1: "a", 4: "b"})
        tm.assert_frame_equal(expected, result)

        result = df.replace({"col": {-1: "-", 1: "a", 4: "b"}})
        tm.assert_frame_equal(expected, result)

    def test_replace_NA_with_None(self):
        # gh-45601
        df = DataFrame({"value": [42, None]}).astype({"value": "Int64"})
        result = df.replace({pd.NA: None})
        expected = DataFrame({"value": [42, None]}, dtype=object)
        tm.assert_frame_equal(result, expected)

    def test_replace_NAT_with_None(self):
        # gh-45836
        df = DataFrame([pd.NaT, pd.NaT])
        result = df.replace({pd.NaT: None, np.nan: None})
        expected = DataFrame([None, None])
        tm.assert_frame_equal(result, expected)

    def test_replace_with_None_keeps_categorical(self):
        # gh-46634
        cat_series = Series(["b", "b", "b", "d"], dtype="category")
        df = DataFrame(
            {
                "id": Series([5, 4, 3, 2], dtype="float64"),
                "col": cat_series,
            }
        )
        result = df.replace({3: None})

        expected = DataFrame(
            {
                "id": Series([5.0, 4.0, None, 2.0], dtype="object"),
                "col": cat_series,
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_replace_value_is_none(self, datetime_frame):
        orig_value = datetime_frame.iloc[0, 0]
        orig2 = datetime_frame.iloc[1, 0]

        datetime_frame.iloc[0, 0] = np.nan
        datetime_frame.iloc[1, 0] = 1

        result = datetime_frame.replace(to_replace={np.nan: 0})
        expected = datetime_frame.T.replace(to_replace={np.nan: 0}).T
        tm.assert_frame_equal(result, expected)

        result = datetime_frame.replace(to_replace={np.nan: 0, 1: -1e8})
        tsframe = datetime_frame.copy()
        tsframe.iloc[0, 0] = 0
        tsframe.iloc[1, 0] = -1e8
        expected = tsframe
        tm.assert_frame_equal(expected, result)
        datetime_frame.iloc[0, 0] = orig_value
        datetime_frame.iloc[1, 0] = orig2

    def test_replace_for_new_dtypes(self, datetime_frame):
        # dtypes
        tsframe = datetime_frame.copy().astype(np.float32)
        tsframe.loc[tsframe.index[:5], "A"] = np.nan
        tsframe.loc[tsframe.index[-5:], "A"] = np.nan

        zero_filled = tsframe.replace(np.nan, -1e8)
        tm.assert_frame_equal(zero_filled, tsframe.fillna(-1e8))
        tm.assert_frame_equal(zero_filled.replace(-1e8, np.nan), tsframe)

        tsframe.loc[tsframe.index[:5], "A"] = np.nan
        tsframe.loc[tsframe.index[-5:], "A"] = np.nan
        tsframe.loc[tsframe.index[:5], "B"] = np.nan
        msg = "DataFrame.fillna with 'method' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # TODO: what is this even testing?
            result = tsframe.fillna(method="bfill")
            tm.assert_frame_equal(result, tsframe.fillna(method="bfill"))

    @pytest.mark.parametrize(
        "frame, to_replace, value, expected",
        [
            (DataFrame({"ints": [1, 2, 3]}), 1, 0, DataFrame({"ints": [0, 2, 3]})),
            (
                DataFrame({"ints": [1, 2, 3]}, dtype=np.int32),
                1,
                0,
                DataFrame({"ints": [0, 2, 3]}, dtype=np.int32),
            ),
            (
                DataFrame({"ints": [1, 2, 3]}, dtype=np.int16),
                1,
                0,
                DataFrame({"ints": [0, 2, 3]}, dtype=np.int16),
            ),
            (
                DataFrame({"bools": [True, False, True]}),
                False,
                True,
                DataFrame({"bools": [True, True, True]}),
            ),
            (
                DataFrame({"complex": [1j, 2j, 3j]}),
                1j,
                0,
                DataFrame({"complex": [0j, 2j, 3j]}),
            ),
            (
                DataFrame(
                    {
                        "datetime64": Index(
                            [
                                datetime(2018, 5, 28),
                                datetime(2018, 7, 28),
                                datetime(2018, 5, 28),
                            ]
                        )
                    }
                ),
                datetime(2018, 5, 28),
                datetime(2018, 7, 28),
                DataFrame({"datetime64": Index([datetime(2018, 7, 28)] * 3)}),
            ),
            # GH 20380
            (
                DataFrame({"dt": [datetime(3017, 12, 20)], "str": ["foo"]}),
                "foo",
                "bar",
                DataFrame({"dt": [datetime(3017, 12, 20)], "str": ["bar"]}),
            ),
            # GH 36782
            (
                DataFrame({"dt": [datetime(2920, 10, 1)]}),
                datetime(2920, 10, 1),
                datetime(2020, 10, 1),
                DataFrame({"dt": [datetime(2020, 10, 1)]}),
            ),
            (
                DataFrame(
                    {
                        "A": date_range("20130101", periods=3, tz="US/Eastern"),
                        "B": [0, np.nan, 2],
                    }
                ),
                Timestamp("20130102", tz="US/Eastern"),
                Timestamp("20130104", tz="US/Eastern"),
                DataFrame(
                    {
                        "A": pd.DatetimeIndex(
                            [
                                Timestamp("20130101", tz="US/Eastern"),
                                Timestamp("20130104", tz="US/Eastern"),
                                Timestamp("20130103", tz="US/Eastern"),
                            ]
                        ).as_unit("ns"),
                        "B": [0, np.nan, 2],
                    }
                ),
            ),
            # GH 35376
            (
                DataFrame([[1, 1.0], [2, 2.0]]),
                1.0,
                5,
                DataFrame([[5, 5.0], [2, 2.0]]),
            ),
            (
                DataFrame([[1, 1.0], [2, 2.0]]),
                1,
                5,
                DataFrame([[5, 5.0], [2, 2.0]]),
            ),
            (
                DataFrame([[1, 1.0], [2, 2.0]]),
                1.0,
                5.0,
                DataFrame([[5, 5.0], [2, 2.0]]),
            ),
            (
                DataFrame([[1, 1.0], [2, 2.0]]),
                1,
                5.0,
                DataFrame([[5, 5.0], [2, 2.0]]),
            ),
        ],
    )
    def test_replace_dtypes(self, frame, to_replace, value, expected):
        warn = None
        if isinstance(to_replace, datetime) and to_replace.year == 2920:
            warn = FutureWarning
        msg = "Downcasting behavior in `replace` "
        with tm.assert_produces_warning(warn, match=msg):
            result = frame.replace(to_replace, value)
        tm.assert_frame_equal(result, expected)

    def test_replace_input_formats_listlike(self):
        # both dicts
        to_rep = {"A": np.nan, "B": 0, "C": ""}
        values = {"A": 0, "B": -1, "C": "missing"}
        df = DataFrame(
            {"A": [np.nan, 0, np.inf], "B": [0, 2, 5], "C": ["", "asdf", "fd"]}
        )
        filled = df.replace(to_rep, values)
        expected = {k: v.replace(to_rep[k], values[k]) for k, v in df.items()}
        tm.assert_frame_equal(filled, DataFrame(expected))

        result = df.replace([0, 2, 5], [5, 2, 0])
        expected = DataFrame(
            {"A": [np.nan, 5, np.inf], "B": [5, 2, 0], "C": ["", "asdf", "fd"]}
        )
        tm.assert_frame_equal(result, expected)

        # scalar to dict
        values = {"A": 0, "B": -1, "C": "missing"}
        df = DataFrame(
            {"A": [np.nan, 0, np.nan], "B": [0, 2, 5], "C": ["", "asdf", "fd"]}
        )
        filled = df.replace(np.nan, values)
        expected = {k: v.replace(np.nan, values[k]) for k, v in df.items()}
        tm.assert_frame_equal(filled, DataFrame(expected))

        # list to list
        to_rep = [np.nan, 0, ""]
        values = [-2, -1, "missing"]
        result = df.replace(to_rep, values)
        expected = df.copy()
        for rep, value in zip(to_rep, values):
            return_value = expected.replace(rep, value, inplace=True)
            assert return_value is None
        tm.assert_frame_equal(result, expected)

        msg = r"Replacement lists must match in length\. Expecting 3 got 2"
        with pytest.raises(ValueError, match=msg):
            df.replace(to_rep, values[1:])

    def test_replace_input_formats_scalar(self):
        df = DataFrame(
            {"A": [np.nan, 0, np.inf], "B": [0, 2, 5], "C": ["", "asdf", "fd"]}
        )

        # dict to scalar
        to_rep = {"A": np.nan, "B": 0, "C": ""}
        filled = df.replace(to_rep, 0)
        expected = {k: v.replace(to_rep[k], 0) for k, v in df.items()}
        tm.assert_frame_equal(filled, DataFrame(expected))

        msg = "value argument must be scalar, dict, or Series"
        with pytest.raises(TypeError, match=msg):
            df.replace(to_rep, [np.nan, 0, ""])

        # list to scalar
        to_rep = [np.nan, 0, ""]
        result = df.replace(to_rep, -1)
        expected = df.copy()
        for rep in to_rep:
            return_value = expected.replace(rep, -1, inplace=True)
            assert return_value is None
        tm.assert_frame_equal(result, expected)

    def test_replace_limit(self):
        # TODO
        pass

    def test_replace_dict_no_regex(self, any_string_dtype):
        answer = Series(
            {
                0: "Strongly Agree",
                1: "Agree",
                2: "Neutral",
                3: "Disagree",
                4: "Strongly Disagree",
            },
            dtype=any_string_dtype,
        )
        weights = {
            "Agree": 4,
            "Disagree": 2,
            "Neutral": 3,
            "Strongly Agree": 5,
            "Strongly Disagree": 1,
        }
        expected = Series({0: 5, 1: 4, 2: 3, 3: 2, 4: 1})
        msg = "Downcasting behavior in `replace` "
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = answer.replace(weights)
        tm.assert_series_equal(result, expected)

    def test_replace_series_no_regex(self, any_string_dtype):
        answer = Series(
            {
                0: "Strongly Agree",
                1: "Agree",
                2: "Neutral",
                3: "Disagree",
                4: "Strongly Disagree",
            },
            dtype=any_string_dtype,
        )
        weights = Series(
            {
                "Agree": 4,
                "Disagree": 2,
                "Neutral": 3,
                "Strongly Agree": 5,
                "Strongly Disagree": 1,
            }
        )
        expected = Series({0: 5, 1: 4, 2: 3, 3: 2, 4: 1})
        msg = "Downcasting behavior in `replace` "
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = answer.replace(weights)
        tm.assert_series_equal(result, expected)

    def test_replace_dict_tuple_list_ordering_remains_the_same(self):
        df = DataFrame({"A": [np.nan, 1]})
        res1 = df.replace(to_replace={np.nan: 0, 1: -1e8})
        res2 = df.replace(to_replace=(1, np.nan), value=[-1e8, 0])
        res3 = df.replace(to_replace=[1, np.nan], value=[-1e8, 0])

        expected = DataFrame({"A": [0, -1e8]})
        tm.assert_frame_equal(res1, res2)
        tm.assert_frame_equal(res2, res3)
        tm.assert_frame_equal(res3, expected)

    def test_replace_doesnt_replace_without_regex(self):
        df = DataFrame(
            {
                "fol": [1, 2, 2, 3],
                "T_opp": ["0", "vr", "0", "0"],
                "T_Dir": ["0", "0", "0", "bt"],
                "T_Enh": ["vo", "0", "0", "0"],
            }
        )
        res = df.replace({r"\D": 1})
        tm.assert_frame_equal(df, res)

    def test_replace_bool_with_string(self):
        df = DataFrame({"a": [True, False], "b": list("ab")})
        result = df.replace(True, "a")
        expected = DataFrame({"a": ["a", False], "b": df.b})
        tm.assert_frame_equal(result, expected)

    def test_replace_pure_bool_with_string_no_op(self):
        df = DataFrame(np.random.default_rng(2).random((2, 2)) > 0.5)
        result = df.replace("asdf", "fdsa")
        tm.assert_frame_equal(df, result)

    def test_replace_bool_with_bool(self):
        df = DataFrame(np.random.default_rng(2).random((2, 2)) > 0.5)
        result = df.replace(False, True)
        expected = DataFrame(np.ones((2, 2), dtype=bool))
        tm.assert_frame_equal(result, expected)

    def test_replace_with_dict_with_bool_keys(self):
        df = DataFrame({0: [True, False], 1: [False, True]})
        result = df.replace({"asdf": "asdb", True: "yes"})
        expected = DataFrame({0: ["yes", False], 1: [False, "yes"]})
        tm.assert_frame_equal(result, expected)

    def test_replace_dict_strings_vs_ints(self):
        # GH#34789
        df = DataFrame({"Y0": [1, 2], "Y1": [3, 4]})
        result = df.replace({"replace_string": "test"})

        tm.assert_frame_equal(result, df)

        result = df["Y0"].replace({"replace_string": "test"})
        tm.assert_series_equal(result, df["Y0"])

    def test_replace_truthy(self):
        df = DataFrame({"a": [True, True]})
        r = df.replace([np.inf, -np.inf], np.nan)
        e = df
        tm.assert_frame_equal(r, e)

    def test_nested_dict_overlapping_keys_replace_int(self):
        # GH 27660 keep behaviour consistent for simple dictionary and
        # nested dictionary replacement
        df = DataFrame({"a": list(range(1, 5))})

        result = df.replace({"a": dict(zip(range(1, 5), range(2, 6)))})
        expected = df.replace(dict(zip(range(1, 5), range(2, 6))))
        tm.assert_frame_equal(result, expected)

    def test_nested_dict_overlapping_keys_replace_str(self):
        # GH 27660
        a = np.arange(1, 5)
        astr = a.astype(str)
        bstr = np.arange(2, 6).astype(str)
        df = DataFrame({"a": astr})
        result = df.replace(dict(zip(astr, bstr)))
        expected = df.replace({"a": dict(zip(astr, bstr))})
        tm.assert_frame_equal(result, expected)

    def test_replace_swapping_bug(self):
        df = DataFrame({"a": [True, False, True]})
        res = df.replace({"a": {True: "Y", False: "N"}})
        expect = DataFrame({"a": ["Y", "N", "Y"]}, dtype=object)
        tm.assert_frame_equal(res, expect)

        df = DataFrame({"a": [0, 1, 0]})
        res = df.replace({"a": {0: "Y", 1: "N"}})
        expect = DataFrame({"a": ["Y", "N", "Y"]}, dtype=object)
        tm.assert_frame_equal(res, expect)

    def test_replace_period(self):
        d = {
            "fname": {
                "out_augmented_AUG_2011.json": pd.Period(year=2011, month=8, freq="M"),
                "out_augmented_JAN_2011.json": pd.Period(year=2011, month=1, freq="M"),
                "out_augmented_MAY_2012.json": pd.Period(year=2012, month=5, freq="M"),
                "out_augmented_SUBSIDY_WEEK.json": pd.Period(
                    year=2011, month=4, freq="M"
                ),
                "out_augmented_AUG_2012.json": pd.Period(year=2012, month=8, freq="M"),
                "out_augmented_MAY_2011.json": pd.Period(year=2011, month=5, freq="M"),
                "out_augmented_SEP_2013.json": pd.Period(year=2013, month=9, freq="M"),
            }
        }

        df = DataFrame(
            [
                "out_augmented_AUG_2012.json",
                "out_augmented_SEP_2013.json",
                "out_augmented_SUBSIDY_WEEK.json",
                "out_augmented_MAY_2012.json",
                "out_augmented_MAY_2011.json",
                "out_augmented_AUG_2011.json",
                "out_augmented_JAN_2011.json",
            ],
            columns=["fname"],
        )
        assert set(df.fname.values) == set(d["fname"].keys())

        expected = DataFrame({"fname": [d["fname"][k] for k in df.fname.values]})
        assert expected.dtypes.iloc[0] == "Period[M]"
        msg = "Downcasting behavior in `replace` "
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.replace(d)
        tm.assert_frame_equal(result, expected)

    def test_replace_datetime(self):
        d = {
            "fname": {
                "out_augmented_AUG_2011.json": Timestamp("2011-08"),
                "out_augmented_JAN_2011.json": Timestamp("2011-01"),
                "out_augmented_MAY_2012.json": Timestamp("2012-05"),
                "out_augmented_SUBSIDY_WEEK.json": Timestamp("2011-04"),
                "out_augmented_AUG_2012.json": Timestamp("2012-08"),
                "out_augmented_MAY_2011.json": Timestamp("2011-05"),
                "out_augmented_SEP_2013.json": Timestamp("2013-09"),
            }
        }

        df = DataFrame(
            [
                "out_augmented_AUG_2012.json",
                "out_augmented_SEP_2013.json",
                "out_augmented_SUBSIDY_WEEK.json",
                "out_augmented_MAY_2012.json",
                "out_augmented_MAY_2011.json",
                "out_augmented_AUG_2011.json",
                "out_augmented_JAN_2011.json",
            ],
            columns=["fname"],
        )
        assert set(df.fname.values) == set(d["fname"].keys())
        expected = DataFrame({"fname": [d["fname"][k] for k in df.fname.values]})
        msg = "Downcasting behavior in `replace` "
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.replace(d)
        tm.assert_frame_equal(result, expected)

    def test_replace_datetimetz(self):
        # GH 11326
        # behaving poorly when presented with a datetime64[ns, tz]
        df = DataFrame(
            {
                "A": date_range("20130101", periods=3, tz="US/Eastern"),
                "B": [0, np.nan, 2],
            }
        )
        result = df.replace(np.nan, 1)
        expected = DataFrame(
            {
                "A": date_range("20130101", periods=3, tz="US/Eastern"),
                "B": Series([0, 1, 2], dtype="float64"),
            }
        )
        tm.assert_frame_equal(result, expected)

        result = df.fillna(1)
        tm.assert_frame_equal(result, expected)

        result = df.replace(0, np.nan)
        expected = DataFrame(
            {
                "A": date_range("20130101", periods=3, tz="US/Eastern"),
                "B": [np.nan, np.nan, 2],
            }
        )
        tm.assert_frame_equal(result, expected)

        result = df.replace(
            Timestamp("20130102", tz="US/Eastern"),
            Timestamp("20130104", tz="US/Eastern"),
        )
        expected = DataFrame(
            {
                "A": [
                    Timestamp("20130101", tz="US/Eastern"),
                    Timestamp("20130104", tz="US/Eastern"),
                    Timestamp("20130103", tz="US/Eastern"),
                ],
                "B": [0, np.nan, 2],
            }
        )
        expected["A"] = expected["A"].dt.as_unit("ns")
        tm.assert_frame_equal(result, expected)

        result = df.copy()
        result.iloc[1, 0] = np.nan
        result = result.replace({"A": pd.NaT}, Timestamp("20130104", tz="US/Eastern"))
        tm.assert_frame_equal(result, expected)

        # pre-2.0 this would coerce to object with mismatched tzs
        result = df.copy()
        result.iloc[1, 0] = np.nan
        result = result.replace({"A": pd.NaT}, Timestamp("20130104", tz="US/Pacific"))
        expected = DataFrame(
            {
                "A": [
                    Timestamp("20130101", tz="US/Eastern"),
                    Timestamp("20130104", tz="US/Pacific").tz_convert("US/Eastern"),
                    Timestamp("20130103", tz="US/Eastern"),
                ],
                "B": [0, np.nan, 2],
            }
        )
        expected["A"] = expected["A"].dt.as_unit("ns")
        tm.assert_frame_equal(result, expected)

        result = df.copy()
        result.iloc[1, 0] = np.nan
        result = result.replace({"A": np.nan}, Timestamp("20130104"))
        expected = DataFrame(
            {
                "A": [
                    Timestamp("20130101", tz="US/Eastern"),
                    Timestamp("20130104"),
                    Timestamp("20130103", tz="US/Eastern"),
                ],
                "B": [0, np.nan, 2],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_replace_with_empty_dictlike(self, mix_abc):
        # GH 15289
        df = DataFrame(mix_abc)
        tm.assert_frame_equal(df, df.replace({}))
        tm.assert_frame_equal(df, df.replace(Series([], dtype=object)))

        tm.assert_frame_equal(df, df.replace({"b": {}}))
        tm.assert_frame_equal(df, df.replace(Series({"b": {}})))

    @pytest.mark.parametrize(
        "to_replace, method, expected",
        [
            (0, "bfill", {"A": [1, 1, 2], "B": [5, np.nan, 7], "C": ["a", "b", "c"]}),
            (
                np.nan,
                "bfill",
                {"A": [0, 1, 2], "B": [5.0, 7.0, 7.0], "C": ["a", "b", "c"]},
            ),
            ("d", "ffill", {"A": [0, 1, 2], "B": [5, np.nan, 7], "C": ["a", "b", "c"]}),
            (
                [0, 2],
                "bfill",
                {"A": [1, 1, 2], "B": [5, np.nan, 7], "C": ["a", "b", "c"]},
            ),
            (
                [1, 2],
                "pad",
                {"A": [0, 0, 0], "B": [5, np.nan, 7], "C": ["a", "b", "c"]},
            ),
            (
                (1, 2),
                "bfill",
                {"A": [0, 2, 2], "B": [5, np.nan, 7], "C": ["a", "b", "c"]},
            ),
            (
                ["b", "c"],
                "ffill",
                {"A": [0, 1, 2], "B": [5, np.nan, 7], "C": ["a", "a", "a"]},
            ),
        ],
    )
    def test_replace_method(self, to_replace, method, expected):
        # GH 19632
        df = DataFrame({"A": [0, 1, 2], "B": [5, np.nan, 7], "C": ["a", "b", "c"]})

        msg = "The 'method' keyword in DataFrame.replace is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.replace(to_replace=to_replace, value=None, method=method)
        expected = DataFrame(expected)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "replace_dict, final_data",
        [({"a": 1, "b": 1}, [[3, 3], [2, 2]]), ({"a": 1, "b": 2}, [[3, 1], [2, 3]])],
    )
    def test_categorical_replace_with_dict(self, replace_dict, final_data):
        # GH 26988
        df = DataFrame([[1, 1], [2, 2]], columns=["a", "b"], dtype="category")

        final_data = np.array(final_data)

        a = pd.Categorical(final_data[:, 0], categories=[3, 2])

        ex_cat = [3, 2] if replace_dict["b"] == 1 else [1, 3]
        b = pd.Categorical(final_data[:, 1], categories=ex_cat)

        expected = DataFrame({"a": a, "b": b})
        msg2 = "with CategoricalDtype is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg2):
            result = df.replace(replace_dict, 3)
        tm.assert_frame_equal(result, expected)
        msg = (
            r"Attributes of DataFrame.iloc\[:, 0\] \(column name=\"a\"\) are "
            "different"
        )
        with pytest.raises(AssertionError, match=msg):
            # ensure non-inplace call does not affect original
            tm.assert_frame_equal(df, expected)
        with tm.assert_produces_warning(FutureWarning, match=msg2):
            return_value = df.replace(replace_dict, 3, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "df, to_replace, exp",
        [
            (
                {"col1": [1, 2, 3], "col2": [4, 5, 6]},
                {4: 5, 5: 6, 6: 7},
                {"col1": [1, 2, 3], "col2": [5, 6, 7]},
            ),
            (
                {"col1": [1, 2, 3], "col2": ["4", "5", "6"]},
                {"4": "5", "5": "6", "6": "7"},
                {"col1": [1, 2, 3], "col2": ["5", "6", "7"]},
            ),
        ],
    )
    def test_replace_commutative(self, df, to_replace, exp):
        # GH 16051
        # DataFrame.replace() overwrites when values are non-numeric
        # also added to data frame whilst issue was for series

        df = DataFrame(df)

        expected = DataFrame(exp)
        result = df.replace(to_replace)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "replacer",
        [
            Timestamp("20170827"),
            np.int8(1),
            np.int16(1),
            np.float32(1),
            np.float64(1),
        ],
    )
    def test_replace_replacer_dtype(self, replacer):
        # GH26632
        df = DataFrame(["a"], dtype=object)
        msg = "Downcasting behavior in `replace` "
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.replace({"a": replacer, "b": replacer})
        expected = DataFrame([replacer])
        tm.assert_frame_equal(result, expected)

    def test_replace_after_convert_dtypes(self):
        # GH31517
        df = DataFrame({"grp": [1, 2, 3, 4, 5]}, dtype="Int64")
        result = df.replace(1, 10)
        expected = DataFrame({"grp": [10, 2, 3, 4, 5]}, dtype="Int64")
        tm.assert_frame_equal(result, expected)

    def test_replace_invalid_to_replace(self):
        # GH 18634
        # API: replace() should raise an exception if invalid argument is given
        df = DataFrame({"one": ["a", "b ", "c"], "two": ["d ", "e ", "f "]})
        msg = (
            r"Expecting 'to_replace' to be either a scalar, array-like, "
            r"dict or None, got invalid type.*"
        )
        msg2 = (
            "DataFrame.replace without 'value' and with non-dict-like "
            "'to_replace' is deprecated"
        )
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=msg2):
                df.replace(lambda x: x.strip())

    @pytest.mark.parametrize("dtype", ["float", "float64", "int64", "Int64", "boolean"])
    @pytest.mark.parametrize("value", [np.nan, pd.NA])
    def test_replace_no_replacement_dtypes(self, dtype, value):
        # https://github.com/pandas-dev/pandas/issues/32988
        df = DataFrame(np.eye(2), dtype=dtype)
        result = df.replace(to_replace=[None, -np.inf, np.inf], value=value)
        tm.assert_frame_equal(result, df)

    @pytest.mark.parametrize("replacement", [np.nan, 5])
    def test_replace_with_duplicate_columns(self, replacement):
        # GH 24798
        result = DataFrame({"A": [1, 2, 3], "A1": [4, 5, 6], "B": [7, 8, 9]})
        result.columns = list("AAB")

        expected = DataFrame(
            {"A": [1, 2, 3], "A1": [4, 5, 6], "B": [replacement, 8, 9]}
        )
        expected.columns = list("AAB")

        result["B"] = result["B"].replace(7, replacement)

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("value", [pd.Period("2020-01"), pd.Interval(0, 5)])
    def test_replace_ea_ignore_float(self, frame_or_series, value):
        # GH#34871
        obj = DataFrame({"Per": [value] * 3})
        obj = tm.get_obj(obj, frame_or_series)

        expected = obj.copy()
        result = obj.replace(1.0, 0.0)
        tm.assert_equal(expected, result)

    def test_replace_value_category_type(self):
        """
        Test for #23305: to ensure category dtypes are maintained
        after replace with direct values
        """

        # create input data
        input_dict = {
            "col1": [1, 2, 3, 4],
            "col2": ["a", "b", "c", "d"],
            "col3": [1.5, 2.5, 3.5, 4.5],
            "col4": ["cat1", "cat2", "cat3", "cat4"],
            "col5": ["obj1", "obj2", "obj3", "obj4"],
        }
        # explicitly cast columns as category and order them
        input_df = DataFrame(data=input_dict).astype(
            {"col2": "category", "col4": "category"}
        )
        input_df["col2"] = input_df["col2"].cat.reorder_categories(
            ["a", "b", "c", "d"], ordered=True
        )
        input_df["col4"] = input_df["col4"].cat.reorder_categories(
            ["cat1", "cat2", "cat3", "cat4"], ordered=True
        )

        # create expected dataframe
        expected_dict = {
            "col1": [1, 2, 3, 4],
            "col2": ["a", "b", "c", "z"],
            "col3": [1.5, 2.5, 3.5, 4.5],
            "col4": ["cat1", "catX", "cat3", "cat4"],
            "col5": ["obj9", "obj2", "obj3", "obj4"],
        }
        # explicitly cast columns as category and order them
        expected = DataFrame(data=expected_dict).astype(
            {"col2": "category", "col4": "category"}
        )
        expected["col2"] = expected["col2"].cat.reorder_categories(
            ["a", "b", "c", "z"], ordered=True
        )
        expected["col4"] = expected["col4"].cat.reorder_categories(
            ["cat1", "catX", "cat3", "cat4"], ordered=True
        )

        # replace values in input dataframe
        msg = (
            r"The behavior of Series\.replace \(and DataFrame.replace\) "
            "with CategoricalDtype"
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            input_df = input_df.replace("d", "z")
            input_df = input_df.replace("obj1", "obj9")
            result = input_df.replace("cat2", "catX")

        result = result.astype({"col1": "int64", "col3": "float64", "col5": "str"})
        tm.assert_frame_equal(result, expected)

    def test_replace_dict_category_type(self):
        """
        Test to ensure category dtypes are maintained
        after replace with dict values
        """
        # GH#35268, GH#44940

        # create input dataframe
        input_dict = {"col1": ["a"], "col2": ["obj1"], "col3": ["cat1"]}
        # explicitly cast columns as category
        input_df = DataFrame(data=input_dict).astype(
            {"col1": "category", "col2": "category", "col3": "category"}
        )

        # create expected dataframe
        expected_dict = {"col1": ["z"], "col2": ["obj9"], "col3": ["catX"]}
        # explicitly cast columns as category
        expected = DataFrame(data=expected_dict).astype(
            {"col1": "category", "col2": "category", "col3": "category"}
        )

        # replace values in input dataframe using a dict
        msg = (
            r"The behavior of Series\.replace \(and DataFrame.replace\) "
            "with CategoricalDtype"
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = input_df.replace({"a": "z", "obj1": "obj9", "cat1": "catX"})

        tm.assert_frame_equal(result, expected)

    def test_replace_with_compiled_regex(self):
        # https://github.com/pandas-dev/pandas/issues/35680
        df = DataFrame(["a", "b", "c"])
        regex = re.compile("^a$")
        result = df.replace({regex: "z"}, regex=True)
        expected = DataFrame(["z", "b", "c"])
        tm.assert_frame_equal(result, expected)

    def test_replace_intervals(self):
        # https://github.com/pandas-dev/pandas/issues/35931
        df = DataFrame({"a": [pd.Interval(0, 1), pd.Interval(0, 1)]})
        result = df.replace({"a": {pd.Interval(0, 1): "x"}})
        expected = DataFrame({"a": ["x", "x"]}, dtype=object)
        tm.assert_frame_equal(result, expected)

    def test_replace_unicode(self):
        # GH: 16784
        columns_values_map = {"positive": {"正面": 1, "中立": 1, "负面": 0}}
        df1 = DataFrame({"positive": np.ones(3)})
        result = df1.replace(columns_values_map)
        expected = DataFrame({"positive": np.ones(3)})
        tm.assert_frame_equal(result, expected)

    def test_replace_bytes(self, frame_or_series):
        # GH#38900
        obj = frame_or_series(["o"]).astype("|S")
        expected = obj.copy()
        obj = obj.replace({None: np.nan})
        tm.assert_equal(obj, expected)

    @pytest.mark.parametrize(
        "data, to_replace, value, expected",
        [
            ([1], [1.0], [0], [0]),
            ([1], [1], [0], [0]),
            ([1.0], [1.0], [0], [0.0]),
            ([1.0], [1], [0], [0.0]),
        ],
    )
    @pytest.mark.parametrize("box", [list, tuple, np.array])
    def test_replace_list_with_mixed_type(
        self, data, to_replace, value, expected, box, frame_or_series
    ):
        # GH#40371
        obj = frame_or_series(data)
        expected = frame_or_series(expected)
        result = obj.replace(box(to_replace), value)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("val", [2, np.nan, 2.0])
    def test_replace_value_none_dtype_numeric(self, val):
        # GH#48231
        df = DataFrame({"a": [1, val]})
        result = df.replace(val, None)
        expected = DataFrame({"a": [1, None]}, dtype=object)
        tm.assert_frame_equal(result, expected)

        df = DataFrame({"a": [1, val]})
        result = df.replace({val: None})
        tm.assert_frame_equal(result, expected)

    def test_replace_with_nil_na(self):
        # GH 32075
        ser = DataFrame({"a": ["nil", pd.NA]})
        expected = DataFrame({"a": ["anything else", pd.NA]}, index=[0, 1])
        result = ser.replace("nil", "anything else")
        tm.assert_frame_equal(expected, result)


class TestDataFrameReplaceRegex:
    @pytest.mark.parametrize(
        "data",
        [
            {"a": list("ab.."), "b": list("efgh")},
            {"a": list("ab.."), "b": list(range(4))},
        ],
    )
    @pytest.mark.parametrize(
        "to_replace,value", [(r"\s*\.\s*", np.nan), (r"\s*(\.)\s*", r"\1\1\1")]
    )
    @pytest.mark.parametrize("compile_regex", [True, False])
    @pytest.mark.parametrize("regex_kwarg", [True, False])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_regex_replace_scalar(
        self, data, to_replace, value, compile_regex, regex_kwarg, inplace
    ):
        df = DataFrame(data)
        expected = df.copy()

        if compile_regex:
            to_replace = re.compile(to_replace)

        if regex_kwarg:
            regex = to_replace
            to_replace = None
        else:
            regex = True

        result = df.replace(to_replace, value, inplace=inplace, regex=regex)

        if inplace:
            assert result is None
            result = df

        if value is np.nan:
            expected_replace_val = np.nan
        else:
            expected_replace_val = "..."

        expected.loc[expected["a"] == ".", "a"] = expected_replace_val
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("regex", [False, True])
    def test_replace_regex_dtype_frame(self, regex):
        # GH-48644
        df1 = DataFrame({"A": ["0"], "B": ["0"]})
        expected_df1 = DataFrame({"A": [1], "B": [1]})
        msg = "Downcasting behavior in `replace`"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result_df1 = df1.replace(to_replace="0", value=1, regex=regex)
        tm.assert_frame_equal(result_df1, expected_df1)

        df2 = DataFrame({"A": ["0"], "B": ["1"]})
        expected_df2 = DataFrame({"A": [1], "B": ["1"]})
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result_df2 = df2.replace(to_replace="0", value=1, regex=regex)
        tm.assert_frame_equal(result_df2, expected_df2)

    def test_replace_with_value_also_being_replaced(self):
        # GH46306
        df = DataFrame({"A": [0, 1, 2], "B": [1, 0, 2]})
        result = df.replace({0: 1, 1: np.nan})
        expected = DataFrame({"A": [1, np.nan, 2], "B": [np.nan, 1, 2]})
        tm.assert_frame_equal(result, expected)

    def test_replace_categorical_no_replacement(self):
        # GH#46672
        df = DataFrame(
            {
                "a": ["one", "two", None, "three"],
                "b": ["one", None, "two", "three"],
            },
            dtype="category",
        )
        expected = df.copy()

        result = df.replace(to_replace=[".", "def"], value=["_", None])
        tm.assert_frame_equal(result, expected)

    def test_replace_object_splitting(self, using_infer_string):
        # GH#53977
        df = DataFrame({"a": ["a"], "b": "b"})
        if using_infer_string:
            assert len(df._mgr.blocks) == 2
        else:
            assert len(df._mgr.blocks) == 1
        df.replace(to_replace=r"^\s*$", value="", inplace=True, regex=True)
        if using_infer_string:
            assert len(df._mgr.blocks) == 2
        else:
            assert len(df._mgr.blocks) == 1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\click\_textwrap.py ===
from __future__ import annotations

import collections.abc as cabc
import textwrap
from contextlib import contextmanager


class TextWrapper(textwrap.TextWrapper):
    def _handle_long_word(
        self,
        reversed_chunks: list[str],
        cur_line: list[str],
        cur_len: int,
        width: int,
    ) -> None:
        space_left = max(width - cur_len, 1)

        if self.break_long_words:
            last = reversed_chunks[-1]
            cut = last[:space_left]
            res = last[space_left:]
            cur_line.append(cut)
            reversed_chunks[-1] = res
        elif not cur_line:
            cur_line.append(reversed_chunks.pop())

    @contextmanager
    def extra_indent(self, indent: str) -> cabc.Iterator[None]:
        old_initial_indent = self.initial_indent
        old_subsequent_indent = self.subsequent_indent
        self.initial_indent += indent
        self.subsequent_indent += indent

        try:
            yield
        finally:
            self.initial_indent = old_initial_indent
            self.subsequent_indent = old_subsequent_indent

    def indent_only(self, text: str) -> str:
        rv = []

        for idx, line in enumerate(text.splitlines()):
            indent = self.initial_indent

            if idx > 0:
                indent = self.subsequent_indent

            rv.append(f"{indent}{line}")

        return "\n".join(rv)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask\globals.py ===
from __future__ import annotations

import typing as t
from contextvars import ContextVar

from werkzeug.local import LocalProxy

if t.TYPE_CHECKING:  # pragma: no cover
    from .app import Flask
    from .ctx import _AppCtxGlobals
    from .ctx import AppContext
    from .ctx import RequestContext
    from .sessions import SessionMixin
    from .wrappers import Request


_no_app_msg = """\
Working outside of application context.

This typically means that you attempted to use functionality that needed
the current application. To solve this, set up an application context
with app.app_context(). See the documentation for more information.\
"""
_cv_app: ContextVar[AppContext] = ContextVar("flask.app_ctx")
app_ctx: AppContext = LocalProxy(  # type: ignore[assignment]
    _cv_app, unbound_message=_no_app_msg
)
current_app: Flask = LocalProxy(  # type: ignore[assignment]
    _cv_app, "app", unbound_message=_no_app_msg
)
g: _AppCtxGlobals = LocalProxy(  # type: ignore[assignment]
    _cv_app, "g", unbound_message=_no_app_msg
)

_no_req_msg = """\
Working outside of request context.

This typically means that you attempted to use functionality that needed
an active HTTP request. Consult the documentation on testing for
information about how to avoid this problem.\
"""
_cv_request: ContextVar[RequestContext] = ContextVar("flask.request_ctx")
request_ctx: RequestContext = LocalProxy(  # type: ignore[assignment]
    _cv_request, unbound_message=_no_req_msg
)
request: Request = LocalProxy(  # type: ignore[assignment]
    _cv_request, "request", unbound_message=_no_req_msg
)
session: SessionMixin = LocalProxy(  # type: ignore[assignment]
    _cv_request, "session", unbound_message=_no_req_msg
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chartsheet\views.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors import (
    Bool,
    Integer,
    Typed,
    Sequence
)
from openpyxl.descriptors.excel import ExtensionList
from openpyxl.descriptors.serialisable import Serialisable


class ChartsheetView(Serialisable):
    tagname = "sheetView"

    tabSelected = Bool(allow_none=True)
    zoomScale = Integer(allow_none=True)
    workbookViewId = Integer()
    zoomToFit = Bool(allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ()

    def __init__(self,
                 tabSelected=None,
                 zoomScale=None,
                 workbookViewId=0,
                 zoomToFit=True,
                 extLst=None,
                 ):
        self.tabSelected = tabSelected
        self.zoomScale = zoomScale
        self.workbookViewId = workbookViewId
        self.zoomToFit = zoomToFit


class ChartsheetViewList(Serialisable):
    tagname = "sheetViews"

    sheetView = Sequence(expected_type=ChartsheetView, )
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('sheetView',)

    def __init__(self,
                 sheetView=None,
                 extLst=None,
                 ):
        if sheetView is None:
            sheetView = [ChartsheetView()]
        self.sheetView = sheetView

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\worksheet\formula.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.compat import safe_string

class DataTableFormula:


    t = "dataTable"

    def __init__(self,
                 ref,
                 ca=False,
                 dt2D=False,
                 dtr=False,
                 r1=None,
                 r2=None,
                 del1=False,
                 del2=False,
                 **kw):
        self.ref = ref
        self.ca = ca
        self.dt2D = dt2D
        self.dtr = dtr
        self.r1 = r1
        self.r2 = r2
        self.del1 = del1
        self.del2 = del2


    def __iter__(self):
        for k in ["t", "ref", "dt2D", "dtr", "r1", "r2", "del1", "del2", "ca"]:
            v = getattr(self, k)
            if v:
                yield k, safe_string(v)


class ArrayFormula:

    t = "array"


    def __init__(self, ref, text=None):
        self.ref = ref
        self.text = text


    def __iter__(self):
        for k in ["t", "ref"]:
            v = getattr(self, k)
            if v:
                yield k, safe_string(v)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\packages\backports\makefile.py ===
# -*- coding: utf-8 -*-
"""
backports.makefile
~~~~~~~~~~~~~~~~~~

Backports the Python 3 ``socket.makefile`` method for use with anything that
wants to create a "fake" socket object.
"""
import io
from socket import SocketIO


def backport_makefile(
    self, mode="r", buffering=None, encoding=None, errors=None, newline=None
):
    """
    Backport of ``socket.makefile`` from Python 3.5.
    """
    if not set(mode) <= {"r", "w", "b"}:
        raise ValueError("invalid mode %r (only r, w, b allowed)" % (mode,))
    writing = "w" in mode
    reading = "r" in mode or not writing
    assert reading or writing
    binary = "b" in mode
    rawmode = ""
    if reading:
        rawmode += "r"
    if writing:
        rawmode += "w"
    raw = SocketIO(self, rawmode)
    self._makefile_refs += 1
    if buffering is None:
        buffering = -1
    if buffering < 0:
        buffering = io.DEFAULT_BUFFER_SIZE
    if buffering == 0:
        if not binary:
            raise ValueError("unbuffered streams must be binary")
        return raw
    if reading and writing:
        buffer = io.BufferedRWPair(raw, raw, buffering)
    elif reading:
        buffer = io.BufferedReader(raw, buffering)
    else:
        assert writing
        buffer = io.BufferedWriter(raw, buffering)
    if binary:
        return buffer
    text = io.TextIOWrapper(buffer, encoding, errors, newline)
    text.mode = mode
    return text

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\plugin\bootstrap.py ===
# testing/plugin/bootstrap.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

"""
Bootstrapper for test framework plugins.

The entire rationale for this system is to get the modules in plugin/
imported without importing all of the supporting library, so that we can
set up things for testing before coverage starts.

The rationale for all of plugin/ being *in* the supporting library in the
first place is so that the testing and plugin suite is available to other
libraries, mainly external SQLAlchemy and Alembic dialects, to make use
of the same test environment and standard suites available to
SQLAlchemy/Alembic themselves without the need to ship/install a separate
package outside of SQLAlchemy.


"""

import importlib.util
import os
import sys


bootstrap_file = locals()["bootstrap_file"]
to_bootstrap = locals()["to_bootstrap"]


def load_file_as_module(name):
    path = os.path.join(os.path.dirname(bootstrap_file), "%s.py" % name)

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if to_bootstrap == "pytest":
    sys.modules["sqla_plugin_base"] = load_file_as_module("plugin_base")
    sys.modules["sqla_plugin_base"].bootstrapped_as_sqlalchemy = True
    sys.modules["sqla_pytestplugin"] = load_file_as_module("pytestplugin")
else:
    raise Exception("unknown bootstrap: %s" % to_bootstrap)  # noqa

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\kauers.py ===
def finite_diff(expression, variable, increment=1):
    """
    Takes as input a polynomial expression and the variable used to construct
    it and returns the difference between function's value when the input is
    incremented to 1 and the original function value. If you want an increment
    other than one supply it as a third argument.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.series.kauers import finite_diff
    >>> finite_diff(x**2, x)
    2*x + 1
    >>> finite_diff(y**3 + 2*y**2 + 3*y + 4, y)
    3*y**2 + 7*y + 6
    >>> finite_diff(x**2 + 3*x + 8, x, 2)
    4*x + 10
    >>> finite_diff(z**3 + 8*z, z, 3)
    9*z**2 + 27*z + 51
    """
    expression = expression.expand()
    expression2 = expression.subs(variable, variable + increment)
    expression2 = expression2.expand()
    return expression2 - expression

def finite_diff_kauers(sum):
    """
    Takes as input a Sum instance and returns the difference between the sum
    with the upper index incremented by 1 and the original sum. For example,
    if S(n) is a sum, then finite_diff_kauers will return S(n + 1) - S(n).

    Examples
    ========

    >>> from sympy.series.kauers import finite_diff_kauers
    >>> from sympy import Sum
    >>> from sympy.abc import x, y, m, n, k
    >>> finite_diff_kauers(Sum(k, (k, 1, n)))
    n + 1
    >>> finite_diff_kauers(Sum(1/k, (k, 1, n)))
    1/(n + 1)
    >>> finite_diff_kauers(Sum((x*y**2), (x, 1, n), (y, 1, m)))
    (m + 1)**2*(n + 1)
    >>> finite_diff_kauers(Sum((x*y), (x, 1, m), (y, 1, n)))
    (m + 1)*(n + 1)
    """
    function = sum.function
    for l in sum.limits:
        function = function.subs(l[0], l[- 1] + 1)
    return function

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_tests\type_tests\run.py ===
from __future__ import annotations

from typing import TYPE_CHECKING, overload

import trio
from typing_extensions import assert_type

if TYPE_CHECKING:
    from collections.abc import Sequence


async def sleep_sort(values: Sequence[float]) -> list[float]:
    return [1]


async def has_optional(arg: int | None = None) -> int:
    return 5


@overload
async def foo_overloaded(arg: int) -> str: ...


@overload
async def foo_overloaded(arg: str) -> int: ...


async def foo_overloaded(arg: int | str) -> int | str:
    if isinstance(arg, str):
        return 5
    return "hello"


v = trio.run(
    sleep_sort,
    (1, 3, 5, 2, 4),
    clock=trio.testing.MockClock(autojump_threshold=0),
)
assert_type(v, "list[float]")
trio.run(sleep_sort, ["hi", "there"])  # type: ignore[arg-type]
trio.run(sleep_sort)  # type: ignore[arg-type]

r = trio.run(has_optional)
assert_type(r, int)
r = trio.run(has_optional, 5)
trio.run(has_optional, 7, 8)  # type: ignore[arg-type]
trio.run(has_optional, "hello")  # type: ignore[arg-type]


assert_type(trio.run(foo_overloaded, 5), str)
assert_type(trio.run(foo_overloaded, ""), int)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\webdriver_manager\firefox.py ===
import os
from typing import Optional

from webdriver_manager.core.download_manager import DownloadManager
from webdriver_manager.core.driver_cache import DriverCacheManager
from webdriver_manager.core.manager import DriverManager
from webdriver_manager.core.os_manager import OperationSystemManager
from webdriver_manager.drivers.firefox import GeckoDriver


class GeckoDriverManager(DriverManager):
    def __init__(
            self,
            version: Optional[str] = None,
            name: str = "geckodriver",
            url: str = "https://github.com/mozilla/geckodriver/releases/download",
            latest_release_url: str = "https://api.github.com/repos/mozilla/geckodriver/releases/latest",
            mozila_release_tag: str = "https://api.github.com/repos/mozilla/geckodriver/releases/tags/{0}",
            download_manager: Optional[DownloadManager] = None,
            cache_manager: Optional[DriverCacheManager] = None,
            os_system_manager: Optional[OperationSystemManager] = None
    ):
        super(GeckoDriverManager, self).__init__(
            download_manager=download_manager,
            cache_manager=cache_manager
        )

        self.driver = GeckoDriver(
            driver_version=version,
            name=name,
            url=url,
            latest_release_url=latest_release_url,
            mozila_release_tag=mozila_release_tag,
            http_client=self.http_client,
            os_system_manager=os_system_manager
        )

    def install(self) -> str:
        driver_path = self._get_driver_binary_path(self.driver)
        os.chmod(driver_path, 0o755)
        return driver_path

    def get_os_type(self):
        os_type = super().get_os_type()
        if not self._os_system_manager.is_mac_os(os_type):
            return os_type

        macos = 'macos'
        if self._os_system_manager.is_arch(os_type):
            return f"{macos}-aarch64"
        return macos

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\test_html.py ===
from collections.abc import Iterator
from functools import partial
from io import (
    BytesIO,
    StringIO,
)
import os
from pathlib import Path
import re
import threading
from urllib.error import URLError

import numpy as np
import pytest

from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    NA,
    DataFrame,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
    read_csv,
    read_html,
    to_datetime,
)
import pandas._testing as tm

from pandas.io.common import file_path_to_url


@pytest.fixture(
    params=[
        "chinese_utf-16.html",
        "chinese_utf-32.html",
        "chinese_utf-8.html",
        "letz_latin1.html",
    ]
)
def html_encoding_file(request, datapath):
    """Parametrized fixture for HTML encoding test filenames."""
    return datapath("io", "data", "html_encoding", request.param)


def assert_framelist_equal(list1, list2, *args, **kwargs):
    assert len(list1) == len(list2), (
        "lists are not of equal size "
        f"len(list1) == {len(list1)}, "
        f"len(list2) == {len(list2)}"
    )
    msg = "not all list elements are DataFrames"
    both_frames = all(
        map(
            lambda x, y: isinstance(x, DataFrame) and isinstance(y, DataFrame),
            list1,
            list2,
        )
    )
    assert both_frames, msg
    for frame_i, frame_j in zip(list1, list2):
        tm.assert_frame_equal(frame_i, frame_j, *args, **kwargs)
        assert not frame_i.empty, "frames are both empty"


def test_bs4_version_fails(monkeypatch, datapath):
    bs4 = pytest.importorskip("bs4")
    pytest.importorskip("html5lib")

    monkeypatch.setattr(bs4, "__version__", "4.2")
    with pytest.raises(ImportError, match="Pandas requires version"):
        read_html(datapath("io", "data", "html", "spam.html"), flavor="bs4")


def test_invalid_flavor():
    url = "google.com"
    flavor = "invalid flavor"
    msg = r"\{" + flavor + r"\} is not a valid set of flavors"

    with pytest.raises(ValueError, match=msg):
        read_html(StringIO(url), match="google", flavor=flavor)


def test_same_ordering(datapath):
    pytest.importorskip("bs4")
    pytest.importorskip("lxml")
    pytest.importorskip("html5lib")

    filename = datapath("io", "data", "html", "valid_markup.html")
    dfs_lxml = read_html(filename, index_col=0, flavor=["lxml"])
    dfs_bs4 = read_html(filename, index_col=0, flavor=["bs4"])
    assert_framelist_equal(dfs_lxml, dfs_bs4)


@pytest.fixture(
    params=[
        pytest.param("bs4", marks=[td.skip_if_no("bs4"), td.skip_if_no("html5lib")]),
        pytest.param("lxml", marks=td.skip_if_no("lxml")),
    ],
)
def flavor_read_html(request):
    return partial(read_html, flavor=request.param)


class TestReadHtml:
    def test_literal_html_deprecation(self, flavor_read_html):
        # GH 53785
        msg = (
            "Passing literal html to 'read_html' is deprecated and "
            "will be removed in a future version. To read from a "
            "literal string, wrap it in a 'StringIO' object."
        )

        with tm.assert_produces_warning(FutureWarning, match=msg):
            flavor_read_html(
                """<table>
                <thead>
                    <tr>
                        <th>A</th>
                        <th>B</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>1</td>
                        <td>2</td>
                    </tr>
                </tbody>
                <tbody>
                    <tr>
                        <td>3</td>
                        <td>4</td>
                    </tr>
                </tbody>
            </table>"""
            )

    @pytest.fixture
    def spam_data(self, datapath):
        return datapath("io", "data", "html", "spam.html")

    @pytest.fixture
    def banklist_data(self, datapath):
        return datapath("io", "data", "html", "banklist.html")

    def test_to_html_compat(self, flavor_read_html):
        df = (
            DataFrame(
                np.random.default_rng(2).random((4, 3)),
                columns=pd.Index(list("abc")),
            )
            # pylint: disable-next=consider-using-f-string
            .map("{:.3f}".format).astype(float)
        )
        out = df.to_html()
        res = flavor_read_html(
            StringIO(out), attrs={"class": "dataframe"}, index_col=0
        )[0]
        tm.assert_frame_equal(res, df)

    def test_dtype_backend(self, string_storage, dtype_backend, flavor_read_html):
        # GH#50286
        df = DataFrame(
            {
                "a": Series([1, np.nan, 3], dtype="Int64"),
                "b": Series([1, 2, 3], dtype="Int64"),
                "c": Series([1.5, np.nan, 2.5], dtype="Float64"),
                "d": Series([1.5, 2.0, 2.5], dtype="Float64"),
                "e": [True, False, None],
                "f": [True, False, True],
                "g": ["a", "b", "c"],
                "h": ["a", "b", None],
            }
        )

        out = df.to_html(index=False)
        with pd.option_context("mode.string_storage", string_storage):
            result = flavor_read_html(StringIO(out), dtype_backend=dtype_backend)[0]

        if dtype_backend == "pyarrow":
            pa = pytest.importorskip("pyarrow")
            string_dtype = pd.ArrowDtype(pa.string())
        else:
            string_dtype = pd.StringDtype(string_storage)

        expected = DataFrame(
            {
                "a": Series([1, np.nan, 3], dtype="Int64"),
                "b": Series([1, 2, 3], dtype="Int64"),
                "c": Series([1.5, np.nan, 2.5], dtype="Float64"),
                "d": Series([1.5, 2.0, 2.5], dtype="Float64"),
                "e": Series([True, False, NA], dtype="boolean"),
                "f": Series([True, False, True], dtype="boolean"),
                "g": Series(["a", "b", "c"], dtype=string_dtype),
                "h": Series(["a", "b", None], dtype=string_dtype),
            }
        )

        if dtype_backend == "pyarrow":
            import pyarrow as pa

            from pandas.arrays import ArrowExtensionArray

            expected = DataFrame(
                {
                    col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True))
                    for col in expected.columns
                }
            )

        # the storage of the str columns' Index is also affected by the
        # string_storage setting -> ignore that for checking the result
        tm.assert_frame_equal(result, expected, check_column_type=False)

    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_banklist_url(self, httpserver, banklist_data, flavor_read_html):
        with open(banklist_data, encoding="utf-8") as f:
            httpserver.serve_content(content=f.read())
            df1 = flavor_read_html(
                # lxml cannot find attrs leave out for now
                httpserver.url,
                match="First Federal Bank of Florida",  # attrs={"class": "dataTable"}
            )
            # lxml cannot find attrs leave out for now
            df2 = flavor_read_html(
                httpserver.url,
                match="Metcalf Bank",
            )  # attrs={"class": "dataTable"})

        assert_framelist_equal(df1, df2)

    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_spam_url(self, httpserver, spam_data, flavor_read_html):
        with open(spam_data, encoding="utf-8") as f:
            httpserver.serve_content(content=f.read())
            df1 = flavor_read_html(httpserver.url, match=".*Water.*")
            df2 = flavor_read_html(httpserver.url, match="Unit")

        assert_framelist_equal(df1, df2)

    @pytest.mark.slow
    def test_banklist(self, banklist_data, flavor_read_html):
        df1 = flavor_read_html(
            banklist_data, match=".*Florida.*", attrs={"id": "table"}
        )
        df2 = flavor_read_html(
            banklist_data, match="Metcalf Bank", attrs={"id": "table"}
        )

        assert_framelist_equal(df1, df2)

    def test_spam(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match=".*Water.*")
        df2 = flavor_read_html(spam_data, match="Unit")
        assert_framelist_equal(df1, df2)

        assert df1[0].iloc[0, 0] == "Proximates"
        assert df1[0].columns[0] == "Nutrient"

    def test_spam_no_match(self, spam_data, flavor_read_html):
        dfs = flavor_read_html(spam_data)
        for df in dfs:
            assert isinstance(df, DataFrame)

    def test_banklist_no_match(self, banklist_data, flavor_read_html):
        dfs = flavor_read_html(banklist_data, attrs={"id": "table"})
        for df in dfs:
            assert isinstance(df, DataFrame)

    def test_spam_header(self, spam_data, flavor_read_html):
        df = flavor_read_html(spam_data, match=".*Water.*", header=2)[0]
        assert df.columns[0] == "Proximates"
        assert not df.empty

    def test_skiprows_int(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match=".*Water.*", skiprows=1)
        df2 = flavor_read_html(spam_data, match="Unit", skiprows=1)

        assert_framelist_equal(df1, df2)

    def test_skiprows_range(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match=".*Water.*", skiprows=range(2))
        df2 = flavor_read_html(spam_data, match="Unit", skiprows=range(2))

        assert_framelist_equal(df1, df2)

    def test_skiprows_list(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match=".*Water.*", skiprows=[1, 2])
        df2 = flavor_read_html(spam_data, match="Unit", skiprows=[2, 1])

        assert_framelist_equal(df1, df2)

    def test_skiprows_set(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match=".*Water.*", skiprows={1, 2})
        df2 = flavor_read_html(spam_data, match="Unit", skiprows={2, 1})

        assert_framelist_equal(df1, df2)

    def test_skiprows_slice(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match=".*Water.*", skiprows=1)
        df2 = flavor_read_html(spam_data, match="Unit", skiprows=1)

        assert_framelist_equal(df1, df2)

    def test_skiprows_slice_short(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match=".*Water.*", skiprows=slice(2))
        df2 = flavor_read_html(spam_data, match="Unit", skiprows=slice(2))

        assert_framelist_equal(df1, df2)

    def test_skiprows_slice_long(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match=".*Water.*", skiprows=slice(2, 5))
        df2 = flavor_read_html(spam_data, match="Unit", skiprows=slice(4, 1, -1))

        assert_framelist_equal(df1, df2)

    def test_skiprows_ndarray(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match=".*Water.*", skiprows=np.arange(2))
        df2 = flavor_read_html(spam_data, match="Unit", skiprows=np.arange(2))

        assert_framelist_equal(df1, df2)

    def test_skiprows_invalid(self, spam_data, flavor_read_html):
        with pytest.raises(TypeError, match=("is not a valid type for skipping rows")):
            flavor_read_html(spam_data, match=".*Water.*", skiprows="asdf")

    def test_index(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match=".*Water.*", index_col=0)
        df2 = flavor_read_html(spam_data, match="Unit", index_col=0)
        assert_framelist_equal(df1, df2)

    def test_header_and_index_no_types(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match=".*Water.*", header=1, index_col=0)
        df2 = flavor_read_html(spam_data, match="Unit", header=1, index_col=0)
        assert_framelist_equal(df1, df2)

    def test_header_and_index_with_types(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match=".*Water.*", header=1, index_col=0)
        df2 = flavor_read_html(spam_data, match="Unit", header=1, index_col=0)
        assert_framelist_equal(df1, df2)

    def test_infer_types(self, spam_data, flavor_read_html):
        # 10892 infer_types removed
        df1 = flavor_read_html(spam_data, match=".*Water.*", index_col=0)
        df2 = flavor_read_html(spam_data, match="Unit", index_col=0)
        assert_framelist_equal(df1, df2)

    def test_string_io(self, spam_data, flavor_read_html):
        with open(spam_data, encoding="UTF-8") as f:
            data1 = StringIO(f.read())

        with open(spam_data, encoding="UTF-8") as f:
            data2 = StringIO(f.read())

        df1 = flavor_read_html(data1, match=".*Water.*")
        df2 = flavor_read_html(data2, match="Unit")
        assert_framelist_equal(df1, df2)

    def test_string(self, spam_data, flavor_read_html):
        with open(spam_data, encoding="UTF-8") as f:
            data = f.read()

        df1 = flavor_read_html(StringIO(data), match=".*Water.*")
        df2 = flavor_read_html(StringIO(data), match="Unit")

        assert_framelist_equal(df1, df2)

    def test_file_like(self, spam_data, flavor_read_html):
        with open(spam_data, encoding="UTF-8") as f:
            df1 = flavor_read_html(f, match=".*Water.*")

        with open(spam_data, encoding="UTF-8") as f:
            df2 = flavor_read_html(f, match="Unit")

        assert_framelist_equal(df1, df2)

    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_bad_url_protocol(self, httpserver, flavor_read_html):
        httpserver.serve_content("urlopen error unknown url type: git", code=404)
        with pytest.raises(URLError, match="urlopen error unknown url type: git"):
            flavor_read_html("git://github.com", match=".*Water.*")

    @pytest.mark.slow
    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_invalid_url(self, httpserver, flavor_read_html):
        httpserver.serve_content("Name or service not known", code=404)
        with pytest.raises((URLError, ValueError), match="HTTP Error 404: NOT FOUND"):
            flavor_read_html(httpserver.url, match=".*Water.*")

    @pytest.mark.slow
    def test_file_url(self, banklist_data, flavor_read_html):
        url = banklist_data
        dfs = flavor_read_html(
            file_path_to_url(os.path.abspath(url)), match="First", attrs={"id": "table"}
        )
        assert isinstance(dfs, list)
        for df in dfs:
            assert isinstance(df, DataFrame)

    @pytest.mark.slow
    def test_invalid_table_attrs(self, banklist_data, flavor_read_html):
        url = banklist_data
        with pytest.raises(ValueError, match="No tables found"):
            flavor_read_html(
                url, match="First Federal Bank of Florida", attrs={"id": "tasdfable"}
            )

    @pytest.mark.slow
    def test_multiindex_header(self, banklist_data, flavor_read_html):
        df = flavor_read_html(
            banklist_data, match="Metcalf", attrs={"id": "table"}, header=[0, 1]
        )[0]
        assert isinstance(df.columns, MultiIndex)

    @pytest.mark.slow
    def test_multiindex_index(self, banklist_data, flavor_read_html):
        df = flavor_read_html(
            banklist_data, match="Metcalf", attrs={"id": "table"}, index_col=[0, 1]
        )[0]
        assert isinstance(df.index, MultiIndex)

    @pytest.mark.slow
    def test_multiindex_header_index(self, banklist_data, flavor_read_html):
        df = flavor_read_html(
            banklist_data,
            match="Metcalf",
            attrs={"id": "table"},
            header=[0, 1],
            index_col=[0, 1],
        )[0]
        assert isinstance(df.columns, MultiIndex)
        assert isinstance(df.index, MultiIndex)

    @pytest.mark.slow
    def test_multiindex_header_skiprows_tuples(self, banklist_data, flavor_read_html):
        df = flavor_read_html(
            banklist_data,
            match="Metcalf",
            attrs={"id": "table"},
            header=[0, 1],
            skiprows=1,
        )[0]
        assert isinstance(df.columns, MultiIndex)

    @pytest.mark.slow
    def test_multiindex_header_skiprows(self, banklist_data, flavor_read_html):
        df = flavor_read_html(
            banklist_data,
            match="Metcalf",
            attrs={"id": "table"},
            header=[0, 1],
            skiprows=1,
        )[0]
        assert isinstance(df.columns, MultiIndex)

    @pytest.mark.slow
    def test_multiindex_header_index_skiprows(self, banklist_data, flavor_read_html):
        df = flavor_read_html(
            banklist_data,
            match="Metcalf",
            attrs={"id": "table"},
            header=[0, 1],
            index_col=[0, 1],
            skiprows=1,
        )[0]
        assert isinstance(df.index, MultiIndex)
        assert isinstance(df.columns, MultiIndex)

    @pytest.mark.slow
    def test_regex_idempotency(self, banklist_data, flavor_read_html):
        url = banklist_data
        dfs = flavor_read_html(
            file_path_to_url(os.path.abspath(url)),
            match=re.compile(re.compile("Florida")),
            attrs={"id": "table"},
        )
        assert isinstance(dfs, list)
        for df in dfs:
            assert isinstance(df, DataFrame)

    def test_negative_skiprows(self, spam_data, flavor_read_html):
        msg = r"\(you passed a negative value\)"
        with pytest.raises(ValueError, match=msg):
            flavor_read_html(spam_data, match="Water", skiprows=-1)

    @pytest.fixture
    def python_docs(self):
        return """
          <table class="contentstable" align="center"><tr>
            <td width="50%">
            <p class="biglink"><a class="biglink" href="whatsnew/2.7.html">What's new in Python 2.7?</a><br/>
                <span class="linkdescr">or <a href="whatsnew/index.html">all "What's new" documents</a> since 2.0</span></p>
            <p class="biglink"><a class="biglink" href="tutorial/index.html">Tutorial</a><br/>
                <span class="linkdescr">start here</span></p>
            <p class="biglink"><a class="biglink" href="library/index.html">Library Reference</a><br/>
                <span class="linkdescr">keep this under your pillow</span></p>
            <p class="biglink"><a class="biglink" href="reference/index.html">Language Reference</a><br/>
                <span class="linkdescr">describes syntax and language elements</span></p>
            <p class="biglink"><a class="biglink" href="using/index.html">Python Setup and Usage</a><br/>
                <span class="linkdescr">how to use Python on different platforms</span></p>
            <p class="biglink"><a class="biglink" href="howto/index.html">Python HOWTOs</a><br/>
                <span class="linkdescr">in-depth documents on specific topics</span></p>
            </td><td width="50%">
            <p class="biglink"><a class="biglink" href="installing/index.html">Installing Python Modules</a><br/>
                <span class="linkdescr">installing from the Python Package Index &amp; other sources</span></p>
            <p class="biglink"><a class="biglink" href="distributing/index.html">Distributing Python Modules</a><br/>
                <span class="linkdescr">publishing modules for installation by others</span></p>
            <p class="biglink"><a class="biglink" href="extending/index.html">Extending and Embedding</a><br/>
                <span class="linkdescr">tutorial for C/C++ programmers</span></p>
            <p class="biglink"><a class="biglink" href="c-api/index.html">Python/C API</a><br/>
                <span class="linkdescr">reference for C/C++ programmers</span></p>
            <p class="biglink"><a class="biglink" href="faq/index.html">FAQs</a><br/>
                <span class="linkdescr">frequently asked questions (with answers!)</span></p>
            </td></tr>
        </table>

        <p><strong>Indices and tables:</strong></p>
        <table class="contentstable" align="center"><tr>
            <td width="50%">
            <p class="biglink"><a class="biglink" href="py-modindex.html">Python Global Module Index</a><br/>
                <span class="linkdescr">quick access to all modules</span></p>
            <p class="biglink"><a class="biglink" href="genindex.html">General Index</a><br/>
                <span class="linkdescr">all functions, classes, terms</span></p>
            <p class="biglink"><a class="biglink" href="glossary.html">Glossary</a><br/>
                <span class="linkdescr">the most important terms explained</span></p>
            </td><td width="50%">
            <p class="biglink"><a class="biglink" href="search.html">Search page</a><br/>
                <span class="linkdescr">search this documentation</span></p>
            <p class="biglink"><a class="biglink" href="contents.html">Complete Table of Contents</a><br/>
                <span class="linkdescr">lists all sections and subsections</span></p>
            </td></tr>
        </table>
        """  # noqa: E501

    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_multiple_matches(self, python_docs, httpserver, flavor_read_html):
        httpserver.serve_content(content=python_docs)
        dfs = flavor_read_html(httpserver.url, match="Python")
        assert len(dfs) > 1

    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_python_docs_table(self, python_docs, httpserver, flavor_read_html):
        httpserver.serve_content(content=python_docs)
        dfs = flavor_read_html(httpserver.url, match="Python")
        zz = [df.iloc[0, 0][0:4] for df in dfs]
        assert sorted(zz) == ["Pyth", "What"]

    def test_empty_tables(self, flavor_read_html):
        """
        Make sure that read_html ignores empty tables.
        """
        html = """
            <table>
                <thead>
                    <tr>
                        <th>A</th>
                        <th>B</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>1</td>
                        <td>2</td>
                    </tr>
                </tbody>
            </table>
            <table>
                <tbody>
                </tbody>
            </table>
        """
        result = flavor_read_html(StringIO(html))
        assert len(result) == 1

    def test_multiple_tbody(self, flavor_read_html):
        # GH-20690
        # Read all tbody tags within a single table.
        result = flavor_read_html(
            StringIO(
                """<table>
            <thead>
                <tr>
                    <th>A</th>
                    <th>B</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>1</td>
                    <td>2</td>
                </tr>
            </tbody>
            <tbody>
                <tr>
                    <td>3</td>
                    <td>4</td>
                </tr>
            </tbody>
        </table>"""
            )
        )[0]

        expected = DataFrame(data=[[1, 2], [3, 4]], columns=["A", "B"])

        tm.assert_frame_equal(result, expected)

    def test_header_and_one_column(self, flavor_read_html):
        """
        Don't fail with bs4 when there is a header and only one column
        as described in issue #9178
        """
        result = flavor_read_html(
            StringIO(
                """<table>
                <thead>
                    <tr>
                        <th>Header</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>first</td>
                    </tr>
                </tbody>
            </table>"""
            )
        )[0]

        expected = DataFrame(data={"Header": "first"}, index=[0])

        tm.assert_frame_equal(result, expected)

    def test_thead_without_tr(self, flavor_read_html):
        """
        Ensure parser adds <tr> within <thead> on malformed HTML.
        """
        result = flavor_read_html(
            StringIO(
                """<table>
            <thead>
                <tr>
                    <th>Country</th>
                    <th>Municipality</th>
                    <th>Year</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Ukraine</td>
                    <th>Odessa</th>
                    <td>1944</td>
                </tr>
            </tbody>
        </table>"""
            )
        )[0]

        expected = DataFrame(
            data=[["Ukraine", "Odessa", 1944]],
            columns=["Country", "Municipality", "Year"],
        )

        tm.assert_frame_equal(result, expected)

    def test_tfoot_read(self, flavor_read_html):
        """
        Make sure that read_html reads tfoot, containing td or th.
        Ignores empty tfoot
        """
        data_template = """<table>
            <thead>
                <tr>
                    <th>A</th>
                    <th>B</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>bodyA</td>
                    <td>bodyB</td>
                </tr>
            </tbody>
            <tfoot>
                {footer}
            </tfoot>
        </table>"""

        expected1 = DataFrame(data=[["bodyA", "bodyB"]], columns=["A", "B"])

        expected2 = DataFrame(
            data=[["bodyA", "bodyB"], ["footA", "footB"]], columns=["A", "B"]
        )

        data1 = data_template.format(footer="")
        data2 = data_template.format(footer="<tr><td>footA</td><th>footB</th></tr>")

        result1 = flavor_read_html(StringIO(data1))[0]
        result2 = flavor_read_html(StringIO(data2))[0]

        tm.assert_frame_equal(result1, expected1)
        tm.assert_frame_equal(result2, expected2)

    def test_parse_header_of_non_string_column(self, flavor_read_html):
        # GH5048: if header is specified explicitly, an int column should be
        # parsed as int while its header is parsed as str
        result = flavor_read_html(
            StringIO(
                """
            <table>
                <tr>
                    <td>S</td>
                    <td>I</td>
                </tr>
                <tr>
                    <td>text</td>
                    <td>1944</td>
                </tr>
            </table>
        """
            ),
            header=0,
        )[0]

        expected = DataFrame([["text", 1944]], columns=("S", "I"))

        tm.assert_frame_equal(result, expected)

    @pytest.mark.slow
    def test_banklist_header(self, banklist_data, datapath, flavor_read_html):
        from pandas.io.html import _remove_whitespace

        def try_remove_ws(x):
            try:
                return _remove_whitespace(x)
            except AttributeError:
                return x

        df = flavor_read_html(banklist_data, match="Metcalf", attrs={"id": "table"})[0]
        ground_truth = read_csv(
            datapath("io", "data", "csv", "banklist.csv"),
            converters={"Updated Date": Timestamp, "Closing Date": Timestamp},
        )
        assert df.shape == ground_truth.shape
        old = [
            "First Vietnamese American Bank In Vietnamese",
            "Westernbank Puerto Rico En Espanol",
            "R-G Premier Bank of Puerto Rico En Espanol",
            "Eurobank En Espanol",
            "Sanderson State Bank En Espanol",
            "Washington Mutual Bank (Including its subsidiary Washington "
            "Mutual Bank FSB)",
            "Silver State Bank En Espanol",
            "AmTrade International Bank En Espanol",
            "Hamilton Bank, NA En Espanol",
            "The Citizens Savings Bank Pioneer Community Bank, Inc.",
        ]
        new = [
            "First Vietnamese American Bank",
            "Westernbank Puerto Rico",
            "R-G Premier Bank of Puerto Rico",
            "Eurobank",
            "Sanderson State Bank",
            "Washington Mutual Bank",
            "Silver State Bank",
            "AmTrade International Bank",
            "Hamilton Bank, NA",
            "The Citizens Savings Bank",
        ]
        dfnew = df.map(try_remove_ws).replace(old, new)
        gtnew = ground_truth.map(try_remove_ws)
        converted = dfnew
        date_cols = ["Closing Date", "Updated Date"]
        converted[date_cols] = converted[date_cols].apply(to_datetime)
        tm.assert_frame_equal(converted, gtnew)

    @pytest.mark.slow
    def test_gold_canyon(self, banklist_data, flavor_read_html):
        gc = "Gold Canyon"
        with open(banklist_data, encoding="utf-8") as f:
            raw_text = f.read()

        assert gc in raw_text
        df = flavor_read_html(
            banklist_data, match="Gold Canyon", attrs={"id": "table"}
        )[0]
        assert gc in df.to_string()

    def test_different_number_of_cols(self, flavor_read_html):
        expected = flavor_read_html(
            StringIO(
                """<table>
                        <thead>
                            <tr style="text-align: right;">
                            <th></th>
                            <th>C_l0_g0</th>
                            <th>C_l0_g1</th>
                            <th>C_l0_g2</th>
                            <th>C_l0_g3</th>
                            <th>C_l0_g4</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                            <th>R_l0_g0</th>
                            <td> 0.763</td>
                            <td> 0.233</td>
                            <td> nan</td>
                            <td> nan</td>
                            <td> nan</td>
                            </tr>
                            <tr>
                            <th>R_l0_g1</th>
                            <td> 0.244</td>
                            <td> 0.285</td>
                            <td> 0.392</td>
                            <td> 0.137</td>
                            <td> 0.222</td>
                            </tr>
                        </tbody>
                    </table>"""
            ),
            index_col=0,
        )[0]

        result = flavor_read_html(
            StringIO(
                """<table>
                    <thead>
                        <tr style="text-align: right;">
                        <th></th>
                        <th>C_l0_g0</th>
                        <th>C_l0_g1</th>
                        <th>C_l0_g2</th>
                        <th>C_l0_g3</th>
                        <th>C_l0_g4</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                        <th>R_l0_g0</th>
                        <td> 0.763</td>
                        <td> 0.233</td>
                        </tr>
                        <tr>
                        <th>R_l0_g1</th>
                        <td> 0.244</td>
                        <td> 0.285</td>
                        <td> 0.392</td>
                        <td> 0.137</td>
                        <td> 0.222</td>
                        </tr>
                    </tbody>
                 </table>"""
            ),
            index_col=0,
        )[0]

        tm.assert_frame_equal(result, expected)

    def test_colspan_rowspan_1(self, flavor_read_html):
        # GH17054
        result = flavor_read_html(
            StringIO(
                """
            <table>
                <tr>
                    <th>A</th>
                    <th colspan="1">B</th>
                    <th rowspan="1">C</th>
                </tr>
                <tr>
                    <td>a</td>
                    <td>b</td>
                    <td>c</td>
                </tr>
            </table>
        """
            )
        )[0]

        expected = DataFrame([["a", "b", "c"]], columns=["A", "B", "C"])

        tm.assert_frame_equal(result, expected)

    def test_colspan_rowspan_copy_values(self, flavor_read_html):
        # GH17054

        # In ASCII, with lowercase letters being copies:
        #
        # X x Y Z W
        # A B b z C

        result = flavor_read_html(
            StringIO(
                """
            <table>
                <tr>
                    <td colspan="2">X</td>
                    <td>Y</td>
                    <td rowspan="2">Z</td>
                    <td>W</td>
                </tr>
                <tr>
                    <td>A</td>
                    <td colspan="2">B</td>
                    <td>C</td>
                </tr>
            </table>
        """
            ),
            header=0,
        )[0]

        expected = DataFrame(
            data=[["A", "B", "B", "Z", "C"]], columns=["X", "X.1", "Y", "Z", "W"]
        )

        tm.assert_frame_equal(result, expected)

    def test_colspan_rowspan_both_not_1(self, flavor_read_html):
        # GH17054

        # In ASCII, with lowercase letters being copies:
        #
        # A B b b C
        # a b b b D

        result = flavor_read_html(
            StringIO(
                """
            <table>
                <tr>
                    <td rowspan="2">A</td>
                    <td rowspan="2" colspan="3">B</td>
                    <td>C</td>
                </tr>
                <tr>
                    <td>D</td>
                </tr>
            </table>
        """
            ),
            header=0,
        )[0]

        expected = DataFrame(
            data=[["A", "B", "B", "B", "D"]], columns=["A", "B", "B.1", "B.2", "C"]
        )

        tm.assert_frame_equal(result, expected)

    def test_rowspan_at_end_of_row(self, flavor_read_html):
        # GH17054

        # In ASCII, with lowercase letters being copies:
        #
        # A B
        # C b

        result = flavor_read_html(
            StringIO(
                """
            <table>
                <tr>
                    <td>A</td>
                    <td rowspan="2">B</td>
                </tr>
                <tr>
                    <td>C</td>
                </tr>
            </table>
        """
            ),
            header=0,
        )[0]

        expected = DataFrame(data=[["C", "B"]], columns=["A", "B"])

        tm.assert_frame_equal(result, expected)

    def test_rowspan_only_rows(self, flavor_read_html):
        # GH17054

        result = flavor_read_html(
            StringIO(
                """
            <table>
                <tr>
                    <td rowspan="3">A</td>
                    <td rowspan="3">B</td>
                </tr>
            </table>
        """
            ),
            header=0,
        )[0]

        expected = DataFrame(data=[["A", "B"], ["A", "B"]], columns=["A", "B"])

        tm.assert_frame_equal(result, expected)

    def test_header_inferred_from_rows_with_only_th(self, flavor_read_html):
        # GH17054
        result = flavor_read_html(
            StringIO(
                """
            <table>
                <tr>
                    <th>A</th>
                    <th>B</th>
                </tr>
                <tr>
                    <th>a</th>
                    <th>b</th>
                </tr>
                <tr>
                    <td>1</td>
                    <td>2</td>
                </tr>
            </table>
        """
            )
        )[0]

        columns = MultiIndex(levels=[["A", "B"], ["a", "b"]], codes=[[0, 1], [0, 1]])
        expected = DataFrame(data=[[1, 2]], columns=columns)

        tm.assert_frame_equal(result, expected)

    def test_parse_dates_list(self, flavor_read_html):
        df = DataFrame({"date": date_range("1/1/2001", periods=10)})
        expected = df.to_html()
        res = flavor_read_html(StringIO(expected), parse_dates=[1], index_col=0)
        tm.assert_frame_equal(df, res[0])
        res = flavor_read_html(StringIO(expected), parse_dates=["date"], index_col=0)
        tm.assert_frame_equal(df, res[0])

    def test_parse_dates_combine(self, flavor_read_html):
        raw_dates = Series(date_range("1/1/2001", periods=10))
        df = DataFrame(
            {
                "date": raw_dates.map(lambda x: str(x.date())),
                "time": raw_dates.map(lambda x: str(x.time())),
            }
        )
        res = flavor_read_html(
            StringIO(df.to_html()), parse_dates={"datetime": [1, 2]}, index_col=1
        )
        newdf = DataFrame({"datetime": raw_dates})
        tm.assert_frame_equal(newdf, res[0])

    def test_wikipedia_states_table(self, datapath, flavor_read_html):
        data = datapath("io", "data", "html", "wikipedia_states.html")
        assert os.path.isfile(data), f"{repr(data)} is not a file"
        assert os.path.getsize(data), f"{repr(data)} is an empty file"
        result = flavor_read_html(data, match="Arizona", header=1)[0]
        assert result.shape == (60, 12)
        assert "Unnamed" in result.columns[-1]
        assert result["sq mi"].dtype == np.dtype("float64")
        assert np.allclose(result.loc[0, "sq mi"], 665384.04)

    def test_wikipedia_states_multiindex(self, datapath, flavor_read_html):
        data = datapath("io", "data", "html", "wikipedia_states.html")
        result = flavor_read_html(data, match="Arizona", index_col=0)[0]
        assert result.shape == (60, 11)
        assert "Unnamed" in result.columns[-1][1]
        assert result.columns.nlevels == 2
        assert np.allclose(result.loc["Alaska", ("Total area[2]", "sq mi")], 665384.04)

    def test_parser_error_on_empty_header_row(self, flavor_read_html):
        result = flavor_read_html(
            StringIO(
                """
                <table>
                    <thead>
                        <tr><th></th><th></tr>
                        <tr><th>A</th><th>B</th></tr>
                    </thead>
                    <tbody>
                        <tr><td>a</td><td>b</td></tr>
                    </tbody>
                </table>
            """
            ),
            header=[0, 1],
        )
        expected = DataFrame(
            [["a", "b"]],
            columns=MultiIndex.from_tuples(
                [("Unnamed: 0_level_0", "A"), ("Unnamed: 1_level_0", "B")]
            ),
        )
        tm.assert_frame_equal(result[0], expected)

    def test_decimal_rows(self, flavor_read_html):
        # GH 12907
        result = flavor_read_html(
            StringIO(
                """<html>
            <body>
             <table>
                <thead>
                    <tr>
                        <th>Header</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>1100#101</td>
                    </tr>
                </tbody>
            </table>
            </body>
        </html>"""
            ),
            decimal="#",
        )[0]

        expected = DataFrame(data={"Header": 1100.101}, index=[0])

        assert result["Header"].dtype == np.dtype("float64")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("arg", [True, False])
    def test_bool_header_arg(self, spam_data, arg, flavor_read_html):
        # GH 6114
        msg = re.escape(
            "Passing a bool to header is invalid. Use header=None for no header or "
            "header=int or list-like of ints to specify the row(s) making up the "
            "column names"
        )
        with pytest.raises(TypeError, match=msg):
            flavor_read_html(spam_data, header=arg)

    def test_converters(self, flavor_read_html):
        # GH 13461
        result = flavor_read_html(
            StringIO(
                """<table>
                 <thead>
                   <tr>
                     <th>a</th>
                    </tr>
                 </thead>
                 <tbody>
                   <tr>
                     <td> 0.763</td>
                   </tr>
                   <tr>
                     <td> 0.244</td>
                   </tr>
                 </tbody>
               </table>"""
            ),
            converters={"a": str},
        )[0]

        expected = DataFrame({"a": ["0.763", "0.244"]})

        tm.assert_frame_equal(result, expected)

    def test_na_values(self, flavor_read_html):
        # GH 13461
        result = flavor_read_html(
            StringIO(
                """<table>
                 <thead>
                   <tr>
                     <th>a</th>
                   </tr>
                 </thead>
                 <tbody>
                   <tr>
                     <td> 0.763</td>
                   </tr>
                   <tr>
                     <td> 0.244</td>
                   </tr>
                 </tbody>
               </table>"""
            ),
            na_values=[0.244],
        )[0]

        expected = DataFrame({"a": [0.763, np.nan]})

        tm.assert_frame_equal(result, expected)

    def test_keep_default_na(self, flavor_read_html):
        html_data = """<table>
                        <thead>
                            <tr>
                            <th>a</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                            <td> N/A</td>
                            </tr>
                            <tr>
                            <td> NA</td>
                            </tr>
                        </tbody>
                    </table>"""

        expected_df = DataFrame({"a": ["N/A", "NA"]})
        html_df = flavor_read_html(StringIO(html_data), keep_default_na=False)[0]
        tm.assert_frame_equal(expected_df, html_df)

        expected_df = DataFrame({"a": [np.nan, np.nan]})
        html_df = flavor_read_html(StringIO(html_data), keep_default_na=True)[0]
        tm.assert_frame_equal(expected_df, html_df)

    def test_preserve_empty_rows(self, flavor_read_html):
        result = flavor_read_html(
            StringIO(
                """
            <table>
                <tr>
                    <th>A</th>
                    <th>B</th>
                </tr>
                <tr>
                    <td>a</td>
                    <td>b</td>
                </tr>
                <tr>
                    <td></td>
                    <td></td>
                </tr>
            </table>
        """
            )
        )[0]

        expected = DataFrame(data=[["a", "b"], [np.nan, np.nan]], columns=["A", "B"])

        tm.assert_frame_equal(result, expected)

    def test_ignore_empty_rows_when_inferring_header(self, flavor_read_html):
        result = flavor_read_html(
            StringIO(
                """
            <table>
                <thead>
                    <tr><th></th><th></tr>
                    <tr><th>A</th><th>B</th></tr>
                    <tr><th>a</th><th>b</th></tr>
                </thead>
                <tbody>
                    <tr><td>1</td><td>2</td></tr>
                </tbody>
            </table>
        """
            )
        )[0]

        columns = MultiIndex(levels=[["A", "B"], ["a", "b"]], codes=[[0, 1], [0, 1]])
        expected = DataFrame(data=[[1, 2]], columns=columns)

        tm.assert_frame_equal(result, expected)

    def test_multiple_header_rows(self, flavor_read_html):
        # Issue #13434
        expected_df = DataFrame(
            data=[("Hillary", 68, "D"), ("Bernie", 74, "D"), ("Donald", 69, "R")]
        )
        expected_df.columns = [
            ["Unnamed: 0_level_0", "Age", "Party"],
            ["Name", "Unnamed: 1_level_1", "Unnamed: 2_level_1"],
        ]
        html = expected_df.to_html(index=False)
        html_df = flavor_read_html(StringIO(html))[0]
        tm.assert_frame_equal(expected_df, html_df)

    def test_works_on_valid_markup(self, datapath, flavor_read_html):
        filename = datapath("io", "data", "html", "valid_markup.html")
        dfs = flavor_read_html(filename, index_col=0)
        assert isinstance(dfs, list)
        assert isinstance(dfs[0], DataFrame)

    @pytest.mark.slow
    def test_fallback_success(self, datapath, flavor_read_html):
        banklist_data = datapath("io", "data", "html", "banklist.html")

        flavor_read_html(banklist_data, match=".*Water.*", flavor=["lxml", "html5lib"])

    def test_to_html_timestamp(self):
        rng = date_range("2000-01-01", periods=10)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), index=rng)

        result = df.to_html()
        assert "2000-01-01" in result

    def test_to_html_borderless(self):
        df = DataFrame([{"A": 1, "B": 2}])
        out_border_default = df.to_html()
        out_border_true = df.to_html(border=True)
        out_border_explicit_default = df.to_html(border=1)
        out_border_nondefault = df.to_html(border=2)
        out_border_zero = df.to_html(border=0)

        out_border_false = df.to_html(border=False)

        assert ' border="1"' in out_border_default
        assert out_border_true == out_border_default
        assert out_border_default == out_border_explicit_default
        assert out_border_default != out_border_nondefault
        assert ' border="2"' in out_border_nondefault
        assert ' border="0"' not in out_border_zero
        assert " border" not in out_border_false
        assert out_border_zero == out_border_false

    @pytest.mark.parametrize(
        "displayed_only,exp0,exp1",
        [
            (True, DataFrame(["foo"]), None),
            (False, DataFrame(["foo  bar  baz  qux"]), DataFrame(["foo"])),
        ],
    )
    def test_displayed_only(self, displayed_only, exp0, exp1, flavor_read_html):
        # GH 20027
        data = """<html>
          <body>
            <table>
              <tr>
                <td>
                  foo
                  <span style="display:none;text-align:center">bar</span>
                  <span style="display:none">baz</span>
                  <span style="display: none">qux</span>
                </td>
              </tr>
            </table>
            <table style="display: none">
              <tr>
                <td>foo</td>
              </tr>
            </table>
          </body>
        </html>"""

        dfs = flavor_read_html(StringIO(data), displayed_only=displayed_only)
        tm.assert_frame_equal(dfs[0], exp0)

        if exp1 is not None:
            tm.assert_frame_equal(dfs[1], exp1)
        else:
            assert len(dfs) == 1  # Should not parse hidden table

    @pytest.mark.parametrize("displayed_only", [True, False])
    def test_displayed_only_with_many_elements(self, displayed_only, flavor_read_html):
        html_table = """
        <table>
            <tr>
                <th>A</th>
                <th>B</th>
            </tr>
            <tr>
                <td>1</td>
                <td>2</td>
            </tr>
            <tr>
                <td><span style="display:none"></span>4</td>
                <td>5</td>
            </tr>
        </table>
        """
        result = flavor_read_html(StringIO(html_table), displayed_only=displayed_only)[
            0
        ]
        expected = DataFrame({"A": [1, 4], "B": [2, 5]})
        tm.assert_frame_equal(result, expected)

    @td.skip_if_windows()
    @pytest.mark.filterwarnings(
        "ignore:You provided Unicode markup but also provided a value for "
        "from_encoding.*:UserWarning"
    )
    def test_encode(self, html_encoding_file, flavor_read_html):
        base_path = os.path.basename(html_encoding_file)
        root = os.path.splitext(base_path)[0]
        _, encoding = root.split("_")

        try:
            with open(html_encoding_file, "rb") as fobj:
                from_string = flavor_read_html(
                    fobj.read(), encoding=encoding, index_col=0
                ).pop()

            with open(html_encoding_file, "rb") as fobj:
                from_file_like = flavor_read_html(
                    BytesIO(fobj.read()), encoding=encoding, index_col=0
                ).pop()

            from_filename = flavor_read_html(
                html_encoding_file, encoding=encoding, index_col=0
            ).pop()
            tm.assert_frame_equal(from_string, from_file_like)
            tm.assert_frame_equal(from_string, from_filename)
        except Exception:
            # seems utf-16/32 fail on windows
            if is_platform_windows():
                if "16" in encoding or "32" in encoding:
                    pytest.skip()
            raise

    def test_parse_failure_unseekable(self, flavor_read_html):
        # Issue #17975

        if flavor_read_html.keywords.get("flavor") == "lxml":
            pytest.skip("Not applicable for lxml")

        class UnseekableStringIO(StringIO):
            def seekable(self):
                return False

        bad = UnseekableStringIO(
            """
            <table><tr><td>spam<foobr />eggs</td></tr></table>"""
        )

        assert flavor_read_html(bad)

        with pytest.raises(ValueError, match="passed a non-rewindable file object"):
            flavor_read_html(bad)

    def test_parse_failure_rewinds(self, flavor_read_html):
        # Issue #17975

        class MockFile:
            def __init__(self, data) -> None:
                self.data = data
                self.at_end = False

            def read(self, size=None):
                data = "" if self.at_end else self.data
                self.at_end = True
                return data

            def seek(self, offset):
                self.at_end = False

            def seekable(self):
                return True

            # GH 49036 pylint checks for presence of __next__ for iterators
            def __next__(self):
                ...

            def __iter__(self) -> Iterator:
                # `is_file_like` depends on the presence of
                # the __iter__ attribute.
                return self

        good = MockFile("<table><tr><td>spam<br />eggs</td></tr></table>")
        bad = MockFile("<table><tr><td>spam<foobr />eggs</td></tr></table>")

        assert flavor_read_html(good)
        assert flavor_read_html(bad)

    @pytest.mark.slow
    @pytest.mark.single_cpu
    def test_importcheck_thread_safety(self, datapath, flavor_read_html):
        # see gh-16928

        class ErrorThread(threading.Thread):
            def run(self):
                try:
                    super().run()
                except Exception as err:
                    self.err = err
                else:
                    self.err = None

        filename = datapath("io", "data", "html", "valid_markup.html")
        helper_thread1 = ErrorThread(target=flavor_read_html, args=(filename,))
        helper_thread2 = ErrorThread(target=flavor_read_html, args=(filename,))

        helper_thread1.start()
        helper_thread2.start()

        while helper_thread1.is_alive() or helper_thread2.is_alive():
            pass
        assert None is helper_thread1.err is helper_thread2.err

    def test_parse_path_object(self, datapath, flavor_read_html):
        # GH 37705
        file_path_string = datapath("io", "data", "html", "spam.html")
        file_path = Path(file_path_string)
        df1 = flavor_read_html(file_path_string)[0]
        df2 = flavor_read_html(file_path)[0]
        tm.assert_frame_equal(df1, df2)

    def test_parse_br_as_space(self, flavor_read_html):
        # GH 29528: pd.read_html() convert <br> to space
        result = flavor_read_html(
            StringIO(
                """
            <table>
                <tr>
                    <th>A</th>
                </tr>
                <tr>
                    <td>word1<br>word2</td>
                </tr>
            </table>
        """
            )
        )[0]

        expected = DataFrame(data=[["word1 word2"]], columns=["A"])

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("arg", ["all", "body", "header", "footer"])
    def test_extract_links(self, arg, flavor_read_html):
        gh_13141_data = """
          <table>
            <tr>
              <th>HTTP</th>
              <th>FTP</th>
              <th><a href="https://en.wiktionary.org/wiki/linkless">Linkless</a></th>
            </tr>
            <tr>
              <td><a href="https://en.wikipedia.org/">Wikipedia</a></td>
              <td>SURROUNDING <a href="ftp://ftp.us.debian.org/">Debian</a> TEXT</td>
              <td>Linkless</td>
            </tr>
            <tfoot>
              <tr>
                <td><a href="https://en.wikipedia.org/wiki/Page_footer">Footer</a></td>
                <td>
                  Multiple <a href="1">links:</a> <a href="2">Only first captured.</a>
                </td>
              </tr>
            </tfoot>
          </table>
          """

        gh_13141_expected = {
            "head_ignore": ["HTTP", "FTP", "Linkless"],
            "head_extract": [
                ("HTTP", None),
                ("FTP", None),
                ("Linkless", "https://en.wiktionary.org/wiki/linkless"),
            ],
            "body_ignore": ["Wikipedia", "SURROUNDING Debian TEXT", "Linkless"],
            "body_extract": [
                ("Wikipedia", "https://en.wikipedia.org/"),
                ("SURROUNDING Debian TEXT", "ftp://ftp.us.debian.org/"),
                ("Linkless", None),
            ],
            "footer_ignore": [
                "Footer",
                "Multiple links: Only first captured.",
                None,
            ],
            "footer_extract": [
                ("Footer", "https://en.wikipedia.org/wiki/Page_footer"),
                ("Multiple links: Only first captured.", "1"),
                None,
            ],
        }

        data_exp = gh_13141_expected["body_ignore"]
        foot_exp = gh_13141_expected["footer_ignore"]
        head_exp = gh_13141_expected["head_ignore"]
        if arg == "all":
            data_exp = gh_13141_expected["body_extract"]
            foot_exp = gh_13141_expected["footer_extract"]
            head_exp = gh_13141_expected["head_extract"]
        elif arg == "body":
            data_exp = gh_13141_expected["body_extract"]
        elif arg == "footer":
            foot_exp = gh_13141_expected["footer_extract"]
        elif arg == "header":
            head_exp = gh_13141_expected["head_extract"]

        result = flavor_read_html(StringIO(gh_13141_data), extract_links=arg)[0]
        expected = DataFrame([data_exp, foot_exp], columns=head_exp)
        expected = expected.fillna(np.nan)
        tm.assert_frame_equal(result, expected)

    def test_extract_links_bad(self, spam_data):
        msg = (
            "`extract_links` must be one of "
            '{None, "header", "footer", "body", "all"}, got "incorrect"'
        )
        with pytest.raises(ValueError, match=msg):
            read_html(spam_data, extract_links="incorrect")

    def test_extract_links_all_no_header(self, flavor_read_html):
        # GH 48316
        data = """
        <table>
          <tr>
            <td>
              <a href='https://google.com'>Google.com</a>
            </td>
          </tr>
        </table>
        """
        result = flavor_read_html(StringIO(data), extract_links="all")[0]
        expected = DataFrame([[("Google.com", "https://google.com")]])
        tm.assert_frame_equal(result, expected)

    def test_invalid_dtype_backend(self):
        msg = (
            "dtype_backend numpy is invalid, only 'numpy_nullable' and "
            "'pyarrow' are allowed."
        )
        with pytest.raises(ValueError, match=msg):
            read_html("test", dtype_backend="numpy")

    def test_style_tag(self, flavor_read_html):
        # GH 48316
        data = """
        <table>
            <tr>
                <th>
                    <style>.style</style>
                    A
                    </th>
                <th>B</th>
            </tr>
            <tr>
                <td>A1</td>
                <td>B1</td>
            </tr>
            <tr>
                <td>A2</td>
                <td>B2</td>
            </tr>
        </table>
        """
        result = flavor_read_html(StringIO(data))[0]
        expected = DataFrame(data=[["A1", "B1"], ["A2", "B2"]], columns=["A", "B"])
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\elementary\tests\test_piecewise.py ===
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import unchanged
from sympy.core.function import (Function, diff, expand)
from sympy.core.mul import Mul
from sympy.core.mod import Mod
from sympy.core.numbers import (Float, I, Rational, oo, pi, zoo)
from sympy.core.relational import (Eq, Ge, Gt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (Abs, adjoint, arg, conjugate, im, re, transpose)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.piecewise import (Piecewise,
    piecewise_fold, piecewise_exclusive, Undefined, ExprCondPair)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.integrals.integrals import (Integral, integrate)
from sympy.logic.boolalg import (And, ITE, Not, Or)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.printing import srepr
from sympy.sets.contains import Contains
from sympy.sets.sets import Interval
from sympy.solvers.solvers import solve
from sympy.testing.pytest import raises, slow
from sympy.utilities.lambdify import lambdify

a, b, c, d, x, y = symbols('a:d, x, y')
z = symbols('z', nonzero=True)


def test_piecewise1():

    # Test canonicalization
    assert Piecewise((x, x < 1.)).has(1.0)  # doesn't get changed to x < 1
    assert unchanged(Piecewise, ExprCondPair(x, x < 1), ExprCondPair(0, True))
    assert Piecewise((x, x < 1), (0, True)) == Piecewise(ExprCondPair(x, x < 1),
                                                         ExprCondPair(0, True))
    assert Piecewise((x, x < 1), (0, True), (1, True)) == \
        Piecewise((x, x < 1), (0, True))
    assert Piecewise((x, x < 1), (0, False), (-1, 1 > 2)) == \
        Piecewise((x, x < 1))
    assert Piecewise((x, x < 1), (0, x < 1), (0, True)) == \
        Piecewise((x, x < 1), (0, True))
    assert Piecewise((x, x < 1), (0, x < 2), (0, True)) == \
        Piecewise((x, x < 1), (0, True))
    assert Piecewise((x, x < 1), (x, x < 2), (0, True)) == \
        Piecewise((x, Or(x < 1, x < 2)), (0, True))
    assert Piecewise((x, x < 1), (x, x < 2), (x, True)) == x
    assert Piecewise((x, True)) == x
    # Explicitly constructed empty Piecewise not accepted
    raises(TypeError, lambda: Piecewise())
    # False condition is never retained
    assert Piecewise((2*x, x < 0), (x, False)) == \
        Piecewise((2*x, x < 0), (x, False), evaluate=False) == \
        Piecewise((2*x, x < 0))
    assert Piecewise((x, False)) == Undefined
    raises(TypeError, lambda: Piecewise(x))
    assert Piecewise((x, 1)) == x  # 1 and 0 are accepted as True/False
    raises(TypeError, lambda: Piecewise((x, 2)))
    raises(TypeError, lambda: Piecewise((x, x**2)))
    raises(TypeError, lambda: Piecewise(([1], True)))
    assert Piecewise(((1, 2), True)) == Tuple(1, 2)
    cond = (Piecewise((1, x < 0), (2, True)) < y)
    assert Piecewise((1, cond)
        ) == Piecewise((1, ITE(x < 0, y > 1, y > 2)))

    assert Piecewise((1, x > 0), (2, And(x <= 0, x > -1))
        ) == Piecewise((1, x > 0), (2, x > -1))
    assert Piecewise((1, x <= 0), (2, (x < 0) & (x > -1))
        ) == Piecewise((1, x <= 0))

    # test for supporting Contains in Piecewise
    pwise = Piecewise(
        (1, And(x <= 6, x > 1, Contains(x, S.Integers))),
        (0, True))
    assert pwise.subs(x, pi) == 0
    assert pwise.subs(x, 2) == 1
    assert pwise.subs(x, 7) == 0

    # Test subs
    p = Piecewise((-1, x < -1), (x**2, x < 0), (log(x), x >= 0))
    p_x2 = Piecewise((-1, x**2 < -1), (x**4, x**2 < 0), (log(x**2), x**2 >= 0))
    assert p.subs(x, x**2) == p_x2
    assert p.subs(x, -5) == -1
    assert p.subs(x, -1) == 1
    assert p.subs(x, 1) == log(1)

    # More subs tests
    p2 = Piecewise((1, x < pi), (-1, x < 2*pi), (0, x > 2*pi))
    p3 = Piecewise((1, Eq(x, 0)), (1/x, True))
    p4 = Piecewise((1, Eq(x, 0)), (2, 1/x>2))
    assert p2.subs(x, 2) == 1
    assert p2.subs(x, 4) == -1
    assert p2.subs(x, 10) == 0
    assert p3.subs(x, 0.0) == 1
    assert p4.subs(x, 0.0) == 1


    f, g, h = symbols('f,g,h', cls=Function)
    pf = Piecewise((f(x), x < -1), (f(x) + h(x) + 2, x <= 1))
    pg = Piecewise((g(x), x < -1), (g(x) + h(x) + 2, x <= 1))
    assert pg.subs(g, f) == pf

    assert Piecewise((1, Eq(x, 0)), (0, True)).subs(x, 0) == 1
    assert Piecewise((1, Eq(x, 0)), (0, True)).subs(x, 1) == 0
    assert Piecewise((1, Eq(x, y)), (0, True)).subs(x, y) == 1
    assert Piecewise((1, Eq(x, z)), (0, True)).subs(x, z) == 1
    assert Piecewise((1, Eq(exp(x), cos(z))), (0, True)).subs(x, z) == \
        Piecewise((1, Eq(exp(z), cos(z))), (0, True))

    p5 = Piecewise( (0, Eq(cos(x) + y, 0)), (1, True))
    assert p5.subs(y, 0) == Piecewise( (0, Eq(cos(x), 0)), (1, True))

    assert Piecewise((-1, y < 1), (0, x < 0), (1, Eq(x, 0)), (2, True)
        ).subs(x, 1) == Piecewise((-1, y < 1), (2, True))
    assert Piecewise((1, Eq(x**2, -1)), (2, x < 0)).subs(x, I) == 1

    p6 = Piecewise((x, x > 0))
    n = symbols('n', negative=True)
    assert p6.subs(x, n) == Undefined

    # Test evalf
    assert p.evalf() == Piecewise((-1.0, x < -1), (x**2, x < 0), (log(x), True))
    assert p.evalf(subs={x: -2}) == -1.0
    assert p.evalf(subs={x: -1}) == 1.0
    assert p.evalf(subs={x: 1}) == log(1)
    assert p6.evalf(subs={x: -5}) == Undefined

    # Test doit
    f_int = Piecewise((Integral(x, (x, 0, 1)), x < 1))
    assert f_int.doit() == Piecewise( (S.Half, x < 1) )

    # Test differentiation
    f = x
    fp = x*p
    dp = Piecewise((0, x < -1), (2*x, x < 0), (1/x, x >= 0))
    fp_dx = x*dp + p
    assert diff(p, x) == dp
    assert diff(f*p, x) == fp_dx

    # Test simple arithmetic
    assert x*p == fp
    assert x*p + p == p + x*p
    assert p + f == f + p
    assert p + dp == dp + p
    assert p - dp == -(dp - p)

    # Test power
    dp2 = Piecewise((0, x < -1), (4*x**2, x < 0), (1/x**2, x >= 0))
    assert dp**2 == dp2

    # Test _eval_interval
    f1 = x*y + 2
    f2 = x*y**2 + 3
    peval = Piecewise((f1, x < 0), (f2, x > 0))
    peval_interval = f1.subs(
        x, 0) - f1.subs(x, -1) + f2.subs(x, 1) - f2.subs(x, 0)
    assert peval._eval_interval(x, 0, 0) == 0
    assert peval._eval_interval(x, -1, 1) == peval_interval
    peval2 = Piecewise((f1, x < 0), (f2, True))
    assert peval2._eval_interval(x, 0, 0) == 0
    assert peval2._eval_interval(x, 1, -1) == -peval_interval
    assert peval2._eval_interval(x, -1, -2) == f1.subs(x, -2) - f1.subs(x, -1)
    assert peval2._eval_interval(x, -1, 1) == peval_interval
    assert peval2._eval_interval(x, None, 0) == peval2.subs(x, 0)
    assert peval2._eval_interval(x, -1, None) == -peval2.subs(x, -1)

    # Test integration
    assert p.integrate() == Piecewise(
        (-x, x < -1),
        (x**3/3 + Rational(4, 3), x < 0),
        (x*log(x) - x + Rational(4, 3), True))
    p = Piecewise((x, x < 1), (x**2, -1 <= x), (x, 3 < x))
    assert integrate(p, (x, -2, 2)) == Rational(5, 6)
    assert integrate(p, (x, 2, -2)) == Rational(-5, 6)
    p = Piecewise((0, x < 0), (1, x < 1), (0, x < 2), (1, x < 3), (0, True))
    assert integrate(p, (x, -oo, oo)) == 2
    p = Piecewise((x, x < -10), (x**2, x <= -1), (x, 1 < x))
    assert integrate(p, (x, -2, 2)) == Undefined

    # Test commutativity
    assert isinstance(p, Piecewise) and p.is_commutative is True


def test_piecewise_free_symbols():
    f = Piecewise((x, a < 0), (y, True))
    assert f.free_symbols == {x, y, a}


def test_piecewise_integrate1():
    x, y = symbols('x y', real=True)

    f = Piecewise(((x - 2)**2, x >= 0), (1, True))
    assert integrate(f, (x, -2, 2)) == Rational(14, 3)

    g = Piecewise(((x - 5)**5, x >= 4), (f, True))
    assert integrate(g, (x, -2, 2)) == Rational(14, 3)
    assert integrate(g, (x, -2, 5)) == Rational(43, 6)

    assert g == Piecewise(((x - 5)**5, x >= 4), (f, x < 4))

    g = Piecewise(((x - 5)**5, 2 <= x), (f, x < 2))
    assert integrate(g, (x, -2, 2)) == Rational(14, 3)
    assert integrate(g, (x, -2, 5)) == Rational(-701, 6)

    assert g == Piecewise(((x - 5)**5, 2 <= x), (f, True))

    g = Piecewise(((x - 5)**5, 2 <= x), (2*f, True))
    assert integrate(g, (x, -2, 2)) == Rational(28, 3)
    assert integrate(g, (x, -2, 5)) == Rational(-673, 6)


def test_piecewise_integrate1b():
    g = Piecewise((1, x > 0), (0, Eq(x, 0)), (-1, x < 0))
    assert integrate(g, (x, -1, 1)) == 0

    g = Piecewise((1, x - y < 0), (0, True))
    assert integrate(g, (y, -oo, 0)) == -Min(0, x)
    assert g.subs(x, -3).integrate((y, -oo, 0)) == 3
    assert integrate(g, (y, 0, -oo)) == Min(0, x)
    assert integrate(g, (y, 0, oo)) == -Max(0, x) + oo
    assert integrate(g, (y, -oo, 42)) == -Min(42, x) + 42
    assert integrate(g, (y, -oo, oo)) == -x + oo

    g = Piecewise((0, x < 0), (x, x <= 1), (1, True))
    gy1 = g.integrate((x, y, 1))
    g1y = g.integrate((x, 1, y))
    for yy in (-1, S.Half, 2):
        assert g.integrate((x, yy, 1)) == gy1.subs(y, yy)
        assert g.integrate((x, 1, yy)) == g1y.subs(y, yy)
    assert gy1 == Piecewise(
        (-Min(1, Max(0, y))**2/2 + S.Half, y < 1),
        (-y + 1, True))
    assert g1y == Piecewise(
        (Min(1, Max(0, y))**2/2 - S.Half, y < 1),
        (y - 1, True))


@slow
def test_piecewise_integrate1ca():
    y = symbols('y', real=True)
    g = Piecewise(
        (1 - x, Interval(0, 1).contains(x)),
        (1 + x, Interval(-1, 0).contains(x)),
        (0, True)
        )
    gy1 = g.integrate((x, y, 1))
    g1y = g.integrate((x, 1, y))

    assert g.integrate((x, -2, 1)) == gy1.subs(y, -2)
    assert g.integrate((x, 1, -2)) == g1y.subs(y, -2)
    assert g.integrate((x, 0, 1)) == gy1.subs(y, 0)
    assert g.integrate((x, 1, 0)) == g1y.subs(y, 0)
    assert g.integrate((x, 2, 1)) == gy1.subs(y, 2)
    assert g.integrate((x, 1, 2)) == g1y.subs(y, 2)
    assert piecewise_fold(gy1.rewrite(Piecewise)
        ).simplify() == Piecewise(
            (1, y <= -1),
            (-y**2/2 - y + S.Half, y <= 0),
            (y**2/2 - y + S.Half, y < 1),
            (0, True))
    assert piecewise_fold(g1y.rewrite(Piecewise)
        ).simplify() == Piecewise(
            (-1, y <= -1),
            (y**2/2 + y - S.Half, y <= 0),
            (-y**2/2 + y - S.Half, y < 1),
            (0, True))
    assert gy1 == Piecewise(
        (
            -Min(1, Max(-1, y))**2/2 - Min(1, Max(-1, y)) +
            Min(1, Max(0, y))**2 + S.Half, y < 1),
        (0, True)
        )
    assert g1y == Piecewise(
        (
            Min(1, Max(-1, y))**2/2 + Min(1, Max(-1, y)) -
            Min(1, Max(0, y))**2 - S.Half, y < 1),
        (0, True))


@slow
def test_piecewise_integrate1cb():
    y = symbols('y', real=True)
    g = Piecewise(
        (0, Or(x <= -1, x >= 1)),
        (1 - x, x > 0),
        (1 + x, True)
        )
    gy1 = g.integrate((x, y, 1))
    g1y = g.integrate((x, 1, y))

    assert g.integrate((x, -2, 1)) == gy1.subs(y, -2)
    assert g.integrate((x, 1, -2)) == g1y.subs(y, -2)
    assert g.integrate((x, 0, 1)) == gy1.subs(y, 0)
    assert g.integrate((x, 1, 0)) == g1y.subs(y, 0)
    assert g.integrate((x, 2, 1)) == gy1.subs(y, 2)
    assert g.integrate((x, 1, 2)) == g1y.subs(y, 2)

    assert piecewise_fold(gy1.rewrite(Piecewise)
        ).simplify() == Piecewise(
            (1, y <= -1),
            (-y**2/2 - y + S.Half, y <= 0),
            (y**2/2 - y + S.Half, y < 1),
            (0, True))
    assert piecewise_fold(g1y.rewrite(Piecewise)
        ).simplify() == Piecewise(
            (-1, y <= -1),
            (y**2/2 + y - S.Half, y <= 0),
            (-y**2/2 + y - S.Half, y < 1),
            (0, True))

    # g1y and gy1 should simplify if the condition that y < 1
    # is applied, e.g. Min(1, Max(-1, y)) --> Max(-1, y)
    assert gy1 == Piecewise(
        (
            -Min(1, Max(-1, y))**2/2 - Min(1, Max(-1, y)) +
            Min(1, Max(0, y))**2 + S.Half, y < 1),
        (0, True)
        )
    assert g1y == Piecewise(
        (
            Min(1, Max(-1, y))**2/2 + Min(1, Max(-1, y)) -
            Min(1, Max(0, y))**2 - S.Half, y < 1),
        (0, True))


def test_piecewise_integrate2():
    from itertools import permutations
    lim = Tuple(x, c, d)
    p = Piecewise((1, x < a), (2, x > b), (3, True))
    q = p.integrate(lim)
    assert q == Piecewise(
        (-c + 2*d - 2*Min(d, Max(a, c)) + Min(d, Max(a, b, c)), c < d),
        (-2*c + d + 2*Min(c, Max(a, d)) - Min(c, Max(a, b, d)), True))
    for v in permutations((1, 2, 3, 4)):
        r = dict(zip((a, b, c, d), v))
        assert p.subs(r).integrate(lim.subs(r)) == q.subs(r)


def test_meijer_bypass():
    # totally bypass meijerg machinery when dealing
    # with Piecewise in integrate
    assert Piecewise((1, x < 4), (0, True)).integrate((x, oo, 1)) == -3


def test_piecewise_integrate3_inequality_conditions():
    from sympy.utilities.iterables import cartes
    lim = (x, 0, 5)
    # set below includes two pts below range, 2 pts in range,
    # 2 pts above range, and the boundaries
    N = (-2, -1, 0, 1, 2, 5, 6, 7)

    p = Piecewise((1, x > a), (2, x > b), (0, True))
    ans = p.integrate(lim)
    for i, j in cartes(N, repeat=2):
        reps = dict(zip((a, b), (i, j)))
        assert ans.subs(reps) == p.subs(reps).integrate(lim)
    assert ans.subs(a, 4).subs(b, 1) == 0 + 2*3 + 1

    p = Piecewise((1, x > a), (2, x < b), (0, True))
    ans = p.integrate(lim)
    for i, j in cartes(N, repeat=2):
        reps = dict(zip((a, b), (i, j)))
        assert ans.subs(reps) == p.subs(reps).integrate(lim)

    # delete old tests that involved c1 and c2 since those
    # reduce to the above except that a value of 0 was used
    # for two expressions whereas the above uses 3 different
    # values


@slow
def test_piecewise_integrate4_symbolic_conditions():
    a = Symbol('a', real=True)
    b = Symbol('b', real=True)
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    p0 = Piecewise((0, Or(x < a, x > b)), (1, True))
    p1 = Piecewise((0, x < a), (0, x > b), (1, True))
    p2 = Piecewise((0, x > b), (0, x < a), (1, True))
    p3 = Piecewise((0, x < a), (1, x < b), (0, True))
    p4 = Piecewise((0, x > b), (1, x > a), (0, True))
    p5 = Piecewise((1, And(a < x, x < b)), (0, True))

    # check values of a=1, b=3 (and reversed) with values
    # of y of 0, 1, 2, 3, 4
    lim = Tuple(x, -oo, y)
    for p in (p0, p1, p2, p3, p4, p5):
        ans = p.integrate(lim)
        for i in range(5):
            reps = {a:1, b:3, y:i}
            assert ans.subs(reps) == p.subs(reps).integrate(lim.subs(reps))
            reps = {a: 3, b:1, y:i}
            assert ans.subs(reps) == p.subs(reps).integrate(lim.subs(reps))
    lim = Tuple(x, y, oo)
    for p in (p0, p1, p2, p3, p4, p5):
        ans = p.integrate(lim)
        for i in range(5):
            reps = {a:1, b:3, y:i}
            assert ans.subs(reps) == p.subs(reps).integrate(lim.subs(reps))
            reps = {a:3, b:1, y:i}
            assert ans.subs(reps) == p.subs(reps).integrate(lim.subs(reps))

    ans = Piecewise(
        (0, x <= Min(a, b)),
        (x - Min(a, b), x <= b),
        (b - Min(a, b), True))
    for i in (p0, p1, p2, p4):
        assert i.integrate(x) == ans
    assert p3.integrate(x) == Piecewise(
        (0, x < a),
        (-a + x, x <= Max(a, b)),
        (-a + Max(a, b), True))
    assert p5.integrate(x) == Piecewise(
        (0, x <= a),
        (-a + x, x <= Max(a, b)),
        (-a + Max(a, b), True))

    p1 = Piecewise((0, x < a), (S.Half, x > b), (1, True))
    p2 = Piecewise((S.Half, x > b), (0, x < a), (1, True))
    p3 = Piecewise((0, x < a), (1, x < b), (S.Half, True))
    p4 = Piecewise((S.Half, x > b), (1, x > a), (0, True))
    p5 = Piecewise((1, And(a < x, x < b)), (S.Half, x > b), (0, True))

    # check values of a=1, b=3 (and reversed) with values
    # of y of 0, 1, 2, 3, 4
    lim = Tuple(x, -oo, y)
    for p in (p1, p2, p3, p4, p5):
        ans = p.integrate(lim)
        for i in range(5):
            reps = {a:1, b:3, y:i}
            assert ans.subs(reps) == p.subs(reps).integrate(lim.subs(reps))
            reps = {a: 3, b:1, y:i}
            assert ans.subs(reps) == p.subs(reps).integrate(lim.subs(reps))


def test_piecewise_integrate5_independent_conditions():
    p = Piecewise((0, Eq(y, 0)), (x*y, True))
    assert integrate(p, (x, 1, 3)) == Piecewise((0, Eq(y, 0)), (4*y, True))


def test_issue_22917():
    p = (Piecewise((0, ITE((x - y > 1) | (2 * x - 2 * y > 1), False,
                           ITE(x - y > 1, 2 * y - 2 < -1, 2 * x - 2 * y > 1))),
                   (Piecewise((0, ITE(x - y > 1, True, 2 * x - 2 * y > 1)),
                              (2 * Piecewise((0, x - y > 1), (y, True)), True)), True))
         + 2 * Piecewise((1, ITE((x - y > 1) | (2 * x - 2 * y > 1), False,
                                 ITE(x - y > 1, 2 * y - 2 < -1, 2 * x - 2 * y > 1))),
                         (Piecewise((1, ITE(x - y > 1, True, 2 * x - 2 * y > 1)),
                                    (2 * Piecewise((1, x - y > 1), (x, True)), True)), True)))
    assert piecewise_fold(p) == Piecewise((2, (x - y > S.Half) | (x - y > 1)),
                                          (2*y + 4, x - y > 1),
                                          (4*x + 2*y, True))
    assert piecewise_fold(p > 1).rewrite(ITE) == ITE((x - y > S.Half) | (x - y > 1), True,
                                                     ITE(x - y > 1, 2*y + 4 > 1, 4*x + 2*y > 1))


def test_piecewise_simplify():
    p = Piecewise(((x**2 + 1)/x**2, Eq(x*(1 + x) - x**2, 0)),
                  ((-1)**x*(-1), True))
    assert p.simplify() == \
        Piecewise((zoo, Eq(x, 0)), ((-1)**(x + 1), True))
    # simplify when there are Eq in conditions
    assert Piecewise(
        (a, And(Eq(a, 0), Eq(a + b, 0))), (1, True)).simplify(
        ) == Piecewise(
        (0, And(Eq(a, 0), Eq(b, 0))), (1, True))
    assert Piecewise((2*x*factorial(a)/(factorial(y)*factorial(-y + a)),
        Eq(y, 0) & Eq(-y + a, 0)), (2*factorial(a)/(factorial(y)*factorial(-y
        + a)), Eq(y, 0) & Eq(-y + a, 1)), (0, True)).simplify(
        ) == Piecewise(
            (2*x, And(Eq(a, 0), Eq(y, 0))),
            (2, And(Eq(a, 1), Eq(y, 0))),
            (0, True))
    args = (2, And(Eq(x, 2), Ge(y, 0))), (x, True)
    assert Piecewise(*args).simplify() == Piecewise(*args)
    args = (1, Eq(x, 0)), (sin(x)/x, True)
    assert Piecewise(*args).simplify() == Piecewise(*args)
    assert Piecewise((2 + y, And(Eq(x, 2), Eq(y, 0))), (x, True)
        ).simplify() == x
    # check that x or f(x) are recognized as being Symbol-like for lhs
    args = Tuple((1, Eq(x, 0)), (sin(x) + 1 + x, True))
    ans = x + sin(x) + 1
    f = Function('f')
    assert Piecewise(*args).simplify() == ans
    assert Piecewise(*args.subs(x, f(x))).simplify() == ans.subs(x, f(x))

    # issue 18634
    d = Symbol("d", integer=True)
    n = Symbol("n", integer=True)
    t = Symbol("t", positive=True)
    expr = Piecewise((-d + 2*n, Eq(1/t, 1)), (t**(1 - 4*n)*t**(4*n - 1)*(-d + 2*n), True))
    assert expr.simplify() == -d + 2*n

    # issue 22747
    p = Piecewise((0, (t < -2) & (t < -1) & (t < 0)), ((t/2 + 1)*(t +
        1)*(t + 2), (t < -1) & (t < 0)), ((S.Half - t/2)*(1 - t)*(t + 1),
        (t < -2) & (t < -1) & (t < 1)), ((t + 1)*(-t*(t/2 + 1) + (S.Half
        - t/2)*(1 - t)), (t < -2) & (t < -1) & (t < 0) & (t < 1)), ((t +
        1)*((S.Half - t/2)*(1 - t) + (t/2 + 1)*(t + 2)), (t < -1) & (t <
        1)), ((t + 1)*(-t*(t/2 + 1) + (S.Half - t/2)*(1 - t)), (t < -1) &
        (t < 0) & (t < 1)), (0, (t < -2) & (t < -1)), ((t/2 + 1)*(t +
        1)*(t + 2), t < -1), ((t + 1)*(-t*(t/2 + 1) + (S.Half - t/2)*(t +
        1)), (t < 0) & ((t < -2) | (t < 0))), ((S.Half - t/2)*(1 - t)*(t
        + 1), (t < 1) & ((t < -2) | (t < 1))), (0, True)) + Piecewise((0,
        (t < -1) & (t < 0) & (t < 1)), ((1 - t)*(t/2 + S.Half)*(t + 1),
        (t < 0) & (t < 1)), ((1 - t)*(1 - t/2)*(2 - t), (t < -1) & (t <
        0) & (t < 2)), ((1 - t)*((1 - t)*(t/2 + S.Half) + (1 - t/2)*(2 -
        t)), (t < -1) & (t < 0) & (t < 1) & (t < 2)), ((1 - t)*((1 -
        t/2)*(2 - t) + (t/2 + S.Half)*(t + 1)), (t < 0) & (t < 2)), ((1 -
        t)*((1 - t)*(t/2 + S.Half) + (1 - t/2)*(2 - t)), (t < 0) & (t <
        1) & (t < 2)), (0, (t < -1) & (t < 0)), ((1 - t)*(t/2 +
        S.Half)*(t + 1), t < 0), ((1 - t)*(t*(1 - t/2) + (1 - t)*(t/2 +
        S.Half)), (t < 1) & ((t < -1) | (t < 1))), ((1 - t)*(1 - t/2)*(2
        - t), (t < 2) & ((t < -1) | (t < 2))), (0, True))
    assert p.simplify() == Piecewise(
        (0, t < -2), ((t + 1)*(t + 2)**2/2, t < -1), (-3*t**3/2
        - 5*t**2/2 + 1, t < 0), (3*t**3/2 - 5*t**2/2 + 1, t < 1), ((1 -
        t)*(t - 2)**2/2, t < 2), (0, True))

    # coverage
    nan = Undefined
    assert Piecewise((1, x > 3), (2, x < 2), (3, x > 1)).simplify(
        )  == Piecewise((1, x > 3), (2, x < 2), (3, True))
    assert Piecewise((1, x < 2), (2, x < 1), (3, True)).simplify(
        ) == Piecewise((1, x < 2), (3, True))
    assert Piecewise((1, x > 2)).simplify() == Piecewise((1, x > 2),
        (nan, True))
    assert Piecewise((1, (x >= 2) & (x < oo))
        ).simplify() == Piecewise((1, (x >= 2) & (x < oo)), (nan, True))
    assert Piecewise((1, x < 2), (2, (x > 1) & (x < 3)), (3, True)
        ). simplify() == Piecewise((1, x < 2), (2, x < 3), (3, True))
    assert Piecewise((1, x < 2), (2, (x <= 3) & (x > 1)), (3, True)
        ).simplify() == Piecewise((1, x < 2), (2, x <= 3), (3, True))
    assert Piecewise((1, x < 2), (2, (x > 2) & (x < 3)), (3, True)
        ).simplify() == Piecewise((1, x < 2), (2, (x > 2) & (x < 3)),
        (3, True))
    assert Piecewise((1, x < 2), (2, (x >= 1) & (x <= 3)), (3, True)
        ).simplify() == Piecewise((1, x < 2), (2, x <= 3), (3, True))
    assert Piecewise((1, x < 1), (2, (x >= 2) & (x <= 3)), (3, True)
        ).simplify() == Piecewise((1, x < 1), (2, (x >= 2) & (x <= 3)),
        (3, True))
    # https://github.com/sympy/sympy/issues/25603
    assert Piecewise((log(x), (x <= 5) & (x > 3)), (x, True)
        ).simplify() == Piecewise((log(x), (x <= 5) & (x > 3)), (x, True))

    assert Piecewise((1, (x >= 1) & (x < 3)), (2, (x > 2) & (x < 4))
        ).simplify() == Piecewise((1, (x >= 1) & (x < 3)), (
        2, (x >= 3) & (x < 4)), (nan, True))
    assert Piecewise((1, (x >= 1) & (x <= 3)), (2, (x > 2) & (x < 4))
        ).simplify() == Piecewise((1, (x >= 1) & (x <= 3)), (
        2, (x > 3) & (x < 4)), (nan, True))

    # involves a symbolic range so cset.inf fails
    L = Symbol('L', nonnegative=True)
    p = Piecewise((nan, x <= 0), (0, (x >= 0) & (L > x) & (L - x <= 0)),
        (x - L/2, (L > x) & (L - x <= 0)),
        (L/2 - x, (x >= 0) & (L > x)),
        (0, L > x), (nan, True))
    assert p.simplify() == Piecewise(
        (nan, x <= 0), (L/2 - x, L > x), (nan, True))
    assert p.subs(L, y).simplify() == Piecewise(
        (nan, x <= 0), (-x + y/2, x < Max(0, y)), (0, x < y), (nan, True))


def test_piecewise_solve():
    abs2 = Piecewise((-x, x <= 0), (x, x > 0))
    f = abs2.subs(x, x - 2)
    assert solve(f, x) == [2]
    assert solve(f - 1, x) == [1, 3]

    f = Piecewise(((x - 2)**2, x >= 0), (1, True))
    assert solve(f, x) == [2]

    g = Piecewise(((x - 5)**5, x >= 4), (f, True))
    assert solve(g, x) == [2, 5]

    g = Piecewise(((x - 5)**5, x >= 4), (f, x < 4))
    assert solve(g, x) == [2, 5]

    g = Piecewise(((x - 5)**5, x >= 2), (f, x < 2))
    assert solve(g, x) == [5]

    g = Piecewise(((x - 5)**5, x >= 2), (f, True))
    assert solve(g, x) == [5]

    g = Piecewise(((x - 5)**5, x >= 2), (f, True), (10, False))
    assert solve(g, x) == [5]

    g = Piecewise(((x - 5)**5, x >= 2),
                  (-x + 2, x - 2 <= 0), (x - 2, x - 2 > 0))
    assert solve(g, x) == [5]

    # if no symbol is given the piecewise detection must still work
    assert solve(Piecewise((x - 2, x > 2), (2 - x, True)) - 3) == [-1, 5]

    f = Piecewise(((x - 2)**2, x >= 0), (0, True))
    raises(NotImplementedError, lambda: solve(f, x))

    def nona(ans):
        return list(filter(lambda x: x is not S.NaN, ans))
    p = Piecewise((x**2 - 4, x < y), (x - 2, True))
    ans = solve(p, x)
    assert nona([i.subs(y, -2) for i in ans]) == [2]
    assert nona([i.subs(y, 2) for i in ans]) == [-2, 2]
    assert nona([i.subs(y, 3) for i in ans]) == [-2, 2]
    assert ans == [
        Piecewise((-2, y > -2), (S.NaN, True)),
        Piecewise((2, y <= 2), (S.NaN, True)),
        Piecewise((2, y > 2), (S.NaN, True))]

    # issue 6060
    absxm3 = Piecewise(
        (x - 3, 0 <= x - 3),
        (3 - x, 0 > x - 3)
    )
    assert solve(absxm3 - y, x) == [
        Piecewise((-y + 3, -y < 0), (S.NaN, True)),
        Piecewise((y + 3, y >= 0), (S.NaN, True))]
    p = Symbol('p', positive=True)
    assert solve(absxm3 - p, x) == [-p + 3, p + 3]

    # issue 6989
    f = Function('f')
    assert solve(Eq(-f(x), Piecewise((1, x > 0), (0, True))), f(x)) == \
        [Piecewise((-1, x > 0), (0, True))]

    # issue 8587
    f = Piecewise((2*x**2, And(0 < x, x < 1)), (2, True))
    assert solve(f - 1) == [1/sqrt(2)]


def test_piecewise_fold():
    p = Piecewise((x, x < 1), (1, 1 <= x))

    assert piecewise_fold(x*p) == Piecewise((x**2, x < 1), (x, 1 <= x))
    assert piecewise_fold(p + p) == Piecewise((2*x, x < 1), (2, 1 <= x))
    assert piecewise_fold(Piecewise((1, x < 0), (2, True))
                          + Piecewise((10, x < 0), (-10, True))) == \
        Piecewise((11, x < 0), (-8, True))

    p1 = Piecewise((0, x < 0), (x, x <= 1), (0, True))
    p2 = Piecewise((0, x < 0), (1 - x, x <= 1), (0, True))

    p = 4*p1 + 2*p2
    assert integrate(
        piecewise_fold(p), (x, -oo, oo)) == integrate(2*x + 2, (x, 0, 1))

    assert piecewise_fold(
        Piecewise((1, y <= 0), (-Piecewise((2, y >= 0)), True)
        )) == Piecewise((1, y <= 0), (-2, y >= 0))

    assert piecewise_fold(Piecewise((x, ITE(x > 0, y < 1, y > 1)))
        ) == Piecewise((x, ((x <= 0) | (y < 1)) & ((x > 0) | (y > 1))))

    a, b = (Piecewise((2, Eq(x, 0)), (0, True)),
        Piecewise((x, Eq(-x + y, 0)), (1, Eq(-x + y, 1)), (0, True)))
    assert piecewise_fold(Mul(a, b, evaluate=False)
        ) == piecewise_fold(Mul(b, a, evaluate=False))


def test_piecewise_fold_piecewise_in_cond():
    p1 = Piecewise((cos(x), x < 0), (0, True))
    p2 = Piecewise((0, Eq(p1, 0)), (p1 / Abs(p1), True))
    assert p2.subs(x, -pi/2) == 0
    assert p2.subs(x, 1) == 0
    assert p2.subs(x, -pi/4) == 1
    p4 = Piecewise((0, Eq(p1, 0)), (1,True))
    ans = piecewise_fold(p4)
    for i in range(-1, 1):
        assert ans.subs(x, i) == p4.subs(x, i)

    r1 = 1 < Piecewise((1, x < 1), (3, True))
    ans = piecewise_fold(r1)
    for i in range(2):
        assert ans.subs(x, i) == r1.subs(x, i)

    p5 = Piecewise((1, x < 0), (3, True))
    p6 = Piecewise((1, x < 1), (3, True))
    p7 = Piecewise((1, p5 < p6), (0, True))
    ans = piecewise_fold(p7)
    for i in range(-1, 2):
        assert ans.subs(x, i) == p7.subs(x, i)


def test_piecewise_fold_piecewise_in_cond_2():
    p1 = Piecewise((cos(x), x < 0), (0, True))
    p2 = Piecewise((0, Eq(p1, 0)), (1 / p1, True))
    p3 = Piecewise(
        (0, (x >= 0) | Eq(cos(x), 0)),
        (1/cos(x), x < 0),
        (zoo, True))  # redundant b/c all x are already covered
    assert(piecewise_fold(p2) == p3)


def test_piecewise_fold_expand():
    p1 = Piecewise((1, Interval(0, 1, False, True).contains(x)), (0, True))

    p2 = piecewise_fold(expand((1 - x)*p1))
    cond = ((x >= 0) & (x < 1))
    assert piecewise_fold(expand((1 - x)*p1), evaluate=False
        ) == Piecewise((1 - x, cond), (-x, cond), (1, cond), (0, True), evaluate=False)
    assert piecewise_fold(expand((1 - x)*p1), evaluate=None
        ) == Piecewise((1 - x, cond), (0, True))
    assert p2 == Piecewise((1 - x, cond), (0, True))
    assert p2 == expand(piecewise_fold((1 - x)*p1))


def test_piecewise_duplicate():
    p = Piecewise((x, x < -10), (x**2, x <= -1), (x, 1 < x))
    assert p == Piecewise(*p.args)


def test_doit():
    p1 = Piecewise((x, x < 1), (x**2, -1 <= x), (x, 3 < x))
    p2 = Piecewise((x, x < 1), (Integral(2 * x), -1 <= x), (x, 3 < x))
    assert p2.doit() == p1
    assert p2.doit(deep=False) == p2
    # issue 17165
    p1 = Sum(y**x, (x, -1, oo)).doit()
    assert p1.doit() == p1


def test_piecewise_interval():
    p1 = Piecewise((x, Interval(0, 1).contains(x)), (0, True))
    assert p1.subs(x, -0.5) == 0
    assert p1.subs(x, 0.5) == 0.5
    assert p1.diff(x) == Piecewise((1, Interval(0, 1).contains(x)), (0, True))
    assert integrate(p1, x) == Piecewise(
        (0, x <= 0),
        (x**2/2, x <= 1),
        (S.Half, True))


def test_piecewise_exclusive():
    p = Piecewise((0, x < 0), (S.Half, x <= 0), (1, True))
    assert piecewise_exclusive(p) == Piecewise((0, x < 0), (S.Half, Eq(x, 0)),
                                               (1, x > 0), evaluate=False)
    assert piecewise_exclusive(p + 2) == Piecewise((0, x < 0), (S.Half, Eq(x, 0)),
                                               (1, x > 0), evaluate=False) + 2
    assert piecewise_exclusive(Piecewise((1, y <= 0),
                                         (-Piecewise((2, y >= 0)), True))) == \
        Piecewise((1, y <= 0),
                  (-Piecewise((2, y >= 0),
                              (S.NaN, y < 0), evaluate=False), y > 0), evaluate=False)
    assert piecewise_exclusive(Piecewise((1, x > y))) == Piecewise((1, x > y),
                                                                  (S.NaN, x <= y),
                                                                  evaluate=False)
    assert piecewise_exclusive(Piecewise((1, x > y)),
                               skip_nan=True) == Piecewise((1, x > y))

    xr, yr = symbols('xr, yr', real=True)

    p1 = Piecewise((1, xr < 0), (2, True), evaluate=False)
    p1x = Piecewise((1, xr < 0), (2, xr >= 0), evaluate=False)

    p2 = Piecewise((p1, yr < 0), (3, True), evaluate=False)
    p2x = Piecewise((p1, yr < 0), (3, yr >= 0), evaluate=False)
    p2xx = Piecewise((p1x, yr < 0), (3, yr >= 0), evaluate=False)

    assert piecewise_exclusive(p2) == p2xx
    assert piecewise_exclusive(p2, deep=False) == p2x


def test_piecewise_collapse():
    assert Piecewise((x, True)) == x
    a = x < 1
    assert Piecewise((x, a), (x + 1, a)) == Piecewise((x, a))
    assert Piecewise((x, a), (x + 1, a.reversed)) == Piecewise((x, a))
    b = x < 5
    def canonical(i):
        if isinstance(i, Piecewise):
            return Piecewise(*i.args)
        return i
    for args in [
        ((1, a), (Piecewise((2, a), (3, b)), b)),
        ((1, a), (Piecewise((2, a), (3, b.reversed)), b)),
        ((1, a), (Piecewise((2, a), (3, b)), b), (4, True)),
        ((1, a), (Piecewise((2, a), (3, b), (4, True)), b)),
        ((1, a), (Piecewise((2, a), (3, b), (4, True)), b), (5, True))]:
        for i in (0, 2, 10):
            assert canonical(
                Piecewise(*args, evaluate=False).subs(x, i)
                ) == canonical(Piecewise(*args).subs(x, i))
    r1, r2, r3, r4 = symbols('r1:5')
    a = x < r1
    b = x < r2
    c = x < r3
    d = x < r4
    assert Piecewise((1, a), (Piecewise(
        (2, a), (3, b), (4, c)), b), (5, c)
        ) == Piecewise((1, a), (3, b), (5, c))
    assert Piecewise((1, a), (Piecewise(
        (2, a), (3, b), (4, c), (6, True)), c), (5, d)
        ) == Piecewise((1, a), (Piecewise(
        (3, b), (4, c)), c), (5, d))
    assert Piecewise((1, Or(a, d)), (Piecewise(
        (2, d), (3, b), (4, c)), b), (5, c)
        ) == Piecewise((1, Or(a, d)), (Piecewise(
        (2, d), (3, b)), b), (5, c))
    assert Piecewise((1, c), (2, ~c), (3, S.true)
        ) == Piecewise((1, c), (2, S.true))
    assert Piecewise((1, c), (2, And(~c, b)), (3,True)
        ) == Piecewise((1, c), (2, b), (3, True))
    assert Piecewise((1, c), (2, Or(~c, b)), (3,True)
        ).subs(dict(zip((r1, r2, r3, r4, x), (1, 2, 3, 4, 3.5))))  == 2
    assert Piecewise((1, c), (2, ~c)) == Piecewise((1, c), (2, True))


def test_piecewise_lambdify():
    p = Piecewise(
        (x**2, x < 0),
        (x, Interval(0, 1, False, True).contains(x)),
        (2 - x, x >= 1),
        (0, True)
    )

    f = lambdify(x, p)
    assert f(-2.0) == 4.0
    assert f(0.0) == 0.0
    assert f(0.5) == 0.5
    assert f(2.0) == 0.0


def test_piecewise_series():
    from sympy.series.order import O
    p1 = Piecewise((sin(x), x < 0), (cos(x), x > 0))
    p2 = Piecewise((x + O(x**2), x < 0), (1 + O(x**2), x > 0))
    assert p1.nseries(x, n=2) == p2


def test_piecewise_as_leading_term():
    p1 = Piecewise((1/x, x > 1), (0, True))
    p2 = Piecewise((x, x > 1), (0, True))
    p3 = Piecewise((1/x, x > 1), (x, True))
    p4 = Piecewise((x, x > 1), (1/x, True))
    p5 = Piecewise((1/x, x > 1), (x, True))
    p6 = Piecewise((1/x, x < 1), (x, True))
    p7 = Piecewise((x, x < 1), (1/x, True))
    p8 = Piecewise((x, x > 1), (1/x, True))
    assert p1.as_leading_term(x) == 0
    assert p2.as_leading_term(x) == 0
    assert p3.as_leading_term(x) == x
    assert p4.as_leading_term(x) == 1/x
    assert p5.as_leading_term(x) == x
    assert p6.as_leading_term(x) == 1/x
    assert p7.as_leading_term(x) == x
    assert p8.as_leading_term(x) == 1/x


def test_piecewise_complex():
    p1 = Piecewise((2, x < 0), (1, 0 <= x))
    p2 = Piecewise((2*I, x < 0), (I, 0 <= x))
    p3 = Piecewise((I*x, x > 1), (1 + I, True))
    p4 = Piecewise((-I*conjugate(x), x > 1), (1 - I, True))

    assert conjugate(p1) == p1
    assert conjugate(p2) == piecewise_fold(-p2)
    assert conjugate(p3) == p4

    assert p1.is_imaginary is False
    assert p1.is_real is True
    assert p2.is_imaginary is True
    assert p2.is_real is False
    assert p3.is_imaginary is None
    assert p3.is_real is None

    assert p1.as_real_imag() == (p1, 0)
    assert p2.as_real_imag() == (0, -I*p2)


def test_conjugate_transpose():
    A, B = symbols("A B", commutative=False)
    p = Piecewise((A*B**2, x > 0), (A**2*B, True))
    assert p.adjoint() == \
        Piecewise((adjoint(A*B**2), x > 0), (adjoint(A**2*B), True))
    assert p.conjugate() == \
        Piecewise((conjugate(A*B**2), x > 0), (conjugate(A**2*B), True))
    assert p.transpose() == \
        Piecewise((transpose(A*B**2), x > 0), (transpose(A**2*B), True))


def test_piecewise_evaluate():
    assert Piecewise((x, True)) == x
    assert Piecewise((x, True), evaluate=True) == x
    assert Piecewise((1, Eq(1, x))).args == ((1, Eq(x, 1)),)
    assert Piecewise((1, Eq(1, x)), evaluate=False).args == (
        (1, Eq(1, x)),)
    # like the additive and multiplicative identities that
    # cannot be kept in Add/Mul, we also do not keep a single True
    p = Piecewise((x, True), evaluate=False)
    assert p == x


def test_as_expr_set_pairs():
    assert Piecewise((x, x > 0), (-x, x <= 0)).as_expr_set_pairs() == \
        [(x, Interval(0, oo, True, True)), (-x, Interval(-oo, 0))]

    assert Piecewise(((x - 2)**2, x >= 0), (0, True)).as_expr_set_pairs() == \
        [((x - 2)**2, Interval(0, oo)), (0, Interval(-oo, 0, True, True))]


def test_S_srepr_is_identity():
    p = Piecewise((10, Eq(x, 0)), (12, True))
    q = S(srepr(p))
    assert p == q


def test_issue_12587():
    # sort holes into intervals
    p = Piecewise((1, x > 4), (2, Not((x <= 3) & (x > -1))), (3, True))
    assert p.integrate((x, -5, 5)) == 23
    p = Piecewise((1, x > 1), (2, x < y), (3, True))
    lim = x, -3, 3
    ans = p.integrate(lim)
    for i in range(-1, 3):
        assert ans.subs(y, i) == p.subs(y, i).integrate(lim)


def test_issue_11045():
    assert integrate(1/(x*sqrt(x**2 - 1)), (x, 1, 2)) == pi/3

    # handle And with Or arguments
    assert Piecewise((1, And(Or(x < 1, x > 3), x < 2)), (0, True)
        ).integrate((x, 0, 3)) == 1

    # hidden false
    assert Piecewise((1, x > 1), (2, x > x + 1), (3, True)
        ).integrate((x, 0, 3)) == 5
    # targetcond is Eq
    assert Piecewise((1, x > 1), (2, Eq(1, x)), (3, True)
        ).integrate((x, 0, 4)) == 6
    # And has Relational needing to be solved
    assert Piecewise((1, And(2*x > x + 1, x < 2)), (0, True)
        ).integrate((x, 0, 3)) == 1
    # Or has Relational needing to be solved
    assert Piecewise((1, Or(2*x > x + 2, x < 1)), (0, True)
        ).integrate((x, 0, 3)) == 2
    # ignore hidden false (handled in canonicalization)
    assert Piecewise((1, x > 1), (2, x > x + 1), (3, True)
        ).integrate((x, 0, 3)) == 5
    # watch for hidden True Piecewise
    assert Piecewise((2, Eq(1 - x, x*(1/x - 1))), (0, True)
        ).integrate((x, 0, 3)) == 6

    # overlapping conditions of targetcond are recognized and ignored;
    # the condition x > 3 will be pre-empted by the first condition
    assert Piecewise((1, Or(x < 1, x > 2)), (2, x > 3), (3, True)
        ).integrate((x, 0, 4)) == 6

    # convert Ne to Or
    assert Piecewise((1, Ne(x, 0)), (2, True)
        ).integrate((x, -1, 1)) == 2

    # no default but well defined
    assert Piecewise((x, (x > 1) & (x < 3)), (1, (x < 4))
        ).integrate((x, 1, 4)) == 5

    p = Piecewise((x, (x > 1) & (x < 3)), (1, (x < 4)))
    nan = Undefined
    i = p.integrate((x, 1, y))
    assert i == Piecewise(
        (y - 1, y < 1),
        (Min(3, y)**2/2 - Min(3, y) + Min(4, y) - S.Half,
            y <= Min(4, y)),
        (nan, True))
    assert p.integrate((x, 1, -1)) == i.subs(y, -1)
    assert p.integrate((x, 1, 4)) == 5
    assert p.integrate((x, 1, 5)) is nan

    # handle Not
    p = Piecewise((1, x > 1), (2, Not(And(x > 1, x< 3))), (3, True))
    assert p.integrate((x, 0, 3)) == 4

    # handle updating of int_expr when there is overlap
    p = Piecewise(
        (1, And(5 > x, x > 1)),
        (2, Or(x < 3, x > 7)),
        (4, x < 8))
    assert p.integrate((x, 0, 10)) == 20

    # And with Eq arg handling
    assert Piecewise((1, x < 1), (2, And(Eq(x, 3), x > 1))
        ).integrate((x, 0, 3)) is S.NaN
    assert Piecewise((1, x < 1), (2, And(Eq(x, 3), x > 1)), (3, True)
        ).integrate((x, 0, 3)) == 7
    assert Piecewise((1, x < 0), (2, And(Eq(x, 3), x < 1)), (3, True)
        ).integrate((x, -1, 1)) == 4
    # middle condition doesn't matter: it's a zero width interval
    assert Piecewise((1, x < 1), (2, Eq(x, 3) & (y < x)), (3, True)
        ).integrate((x, 0, 3)) == 7


def test_holes():
    nan = Undefined
    assert Piecewise((1, x < 2)).integrate(x) == Piecewise(
        (x, x < 2), (nan, True))
    assert Piecewise((1, And(x > 1, x < 2))).integrate(x) == Piecewise(
        (nan, x < 1), (x, x < 2), (nan, True))
    assert Piecewise((1, And(x > 1, x < 2))).integrate((x, 0, 3)) is nan
    assert Piecewise((1, And(x > 0, x < 4))).integrate((x, 1, 3)) == 2

    # this also tests that the integrate method is used on non-Piecwise
    # arguments in _eval_integral
    A, B = symbols("A B")
    a, b = symbols('a b', real=True)
    assert Piecewise((A, And(x < 0, a < 1)), (B, Or(x < 1, a > 2))
        ).integrate(x) == Piecewise(
        (B*x, (a > 2)),
        (Piecewise((A*x, x < 0), (B*x, x < 1), (nan, True)), a < 1),
        (Piecewise((B*x, x < 1), (nan, True)), True))


def test_issue_11922():
    def f(x):
        return Piecewise((0, x < -1), (1 - x**2, x < 1), (0, True))
    autocorr = lambda k: (
        f(x) * f(x + k)).integrate((x, -1, 1))
    assert autocorr(1.9) > 0
    k = symbols('k')
    good_autocorr = lambda k: (
        (1 - x**2) * f(x + k)).integrate((x, -1, 1))
    a = good_autocorr(k)
    assert a.subs(k, 3) == 0
    k = symbols('k', positive=True)
    a = good_autocorr(k)
    assert a.subs(k, 3) == 0
    assert Piecewise((0, x < 1), (10, (x >= 1))
        ).integrate() == Piecewise((0, x < 1), (10*x - 10, True))


def test_issue_5227():
    f = 0.0032513612725229*Piecewise((0, x < -80.8461538461539),
        (-0.0160799238820171*x + 1.33215984776403, x < 2),
        (Piecewise((0.3, x > 123), (0.7, True)) +
        Piecewise((0.4, x > 2), (0.6, True)), x <=
        123), (-0.00817409766454352*x + 2.10541401273885, x <
        380.571428571429), (0, True))
    i = integrate(f, (x, -oo, oo))
    assert i == Integral(f, (x, -oo, oo)).doit()
    assert str(i) == '1.00195081676351'
    assert Piecewise((1, x - y < 0), (0, True)
        ).integrate(y) == Piecewise((0, y <= x), (-x + y, True))


def test_issue_10137():
    a = Symbol('a', real=True)
    b = Symbol('b', real=True)
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    p0 = Piecewise((0, Or(x < a, x > b)), (1, True))
    p1 = Piecewise((0, Or(a > x, b < x)), (1, True))
    assert integrate(p0, (x, y, oo)) == integrate(p1, (x, y, oo))
    p3 = Piecewise((1, And(0 < x, x < a)), (0, True))
    p4 = Piecewise((1, And(a > x, x > 0)), (0, True))
    ip3 = integrate(p3, x)
    assert ip3 == Piecewise(
        (0, x <= 0),
        (x, x <= Max(0, a)),
        (Max(0, a), True))
    ip4 = integrate(p4, x)
    assert ip4 == ip3
    assert p3.integrate((x, 2, 4)) == Min(4, Max(2, a)) - 2
    assert p4.integrate((x, 2, 4)) == Min(4, Max(2, a)) - 2


def test_stackoverflow_43852159():
    f = lambda x: Piecewise((1, (x >= -1) & (x <= 1)), (0, True))
    Conv = lambda x: integrate(f(x - y)*f(y), (y, -oo, +oo))
    cx = Conv(x)
    assert cx.subs(x, -1.5) == cx.subs(x, 1.5)
    assert cx.subs(x, 3) == 0
    assert piecewise_fold(f(x - y)*f(y)) == Piecewise(
        (1, (y >= -1) & (y <= 1) & (x - y >= -1) & (x - y <= 1)),
        (0, True))


def test_issue_12557():
    '''
    # 3200 seconds to compute the fourier part of issue
    import sympy as sym
    x,y,z,t = sym.symbols('x y z t')
    k = sym.symbols("k", integer=True)
    fourier = sym.fourier_series(sym.cos(k*x)*sym.sqrt(x**2),
                                 (x, -sym.pi, sym.pi))
    assert fourier == FourierSeries(
    sqrt(x**2)*cos(k*x), (x, -pi, pi), (Piecewise((pi**2,
    Eq(k, 0)), (2*(-1)**k/k**2 - 2/k**2, True))/(2*pi),
    SeqFormula(Piecewise((pi**2, (Eq(_n, 0) & Eq(k, 0)) | (Eq(_n, 0) &
    Eq(_n, k) & Eq(k, 0)) | (Eq(_n, 0) & Eq(k, 0) & Eq(_n, -k)) | (Eq(_n,
    0) & Eq(_n, k) & Eq(k, 0) & Eq(_n, -k))), (pi**2/2, Eq(_n, k) | Eq(_n,
    -k) | (Eq(_n, 0) & Eq(_n, k)) | (Eq(_n, k) & Eq(k, 0)) | (Eq(_n, 0) &
    Eq(_n, -k)) | (Eq(_n, k) & Eq(_n, -k)) | (Eq(k, 0) & Eq(_n, -k)) |
    (Eq(_n, 0) & Eq(_n, k) & Eq(_n, -k)) | (Eq(_n, k) & Eq(k, 0) & Eq(_n,
    -k))), ((-1)**k*pi**2*_n**3*sin(pi*_n)/(pi*_n**4 - 2*pi*_n**2*k**2 +
    pi*k**4) - (-1)**k*pi**2*_n**3*sin(pi*_n)/(-pi*_n**4 + 2*pi*_n**2*k**2
    - pi*k**4) + (-1)**k*pi*_n**2*cos(pi*_n)/(pi*_n**4 - 2*pi*_n**2*k**2 +
    pi*k**4) - (-1)**k*pi*_n**2*cos(pi*_n)/(-pi*_n**4 + 2*pi*_n**2*k**2 -
    pi*k**4) - (-1)**k*pi**2*_n*k**2*sin(pi*_n)/(pi*_n**4 -
    2*pi*_n**2*k**2 + pi*k**4) +
    (-1)**k*pi**2*_n*k**2*sin(pi*_n)/(-pi*_n**4 + 2*pi*_n**2*k**2 -
    pi*k**4) + (-1)**k*pi*k**2*cos(pi*_n)/(pi*_n**4 - 2*pi*_n**2*k**2 +
    pi*k**4) - (-1)**k*pi*k**2*cos(pi*_n)/(-pi*_n**4 + 2*pi*_n**2*k**2 -
    pi*k**4) - (2*_n**2 + 2*k**2)/(_n**4 - 2*_n**2*k**2 + k**4),
    True))*cos(_n*x)/pi, (_n, 1, oo)), SeqFormula(0, (_k, 1, oo))))
    '''
    x = symbols("x", real=True)
    k = symbols('k', integer=True, finite=True)
    abs2 = lambda x: Piecewise((-x, x <= 0), (x, x > 0))
    assert integrate(abs2(x), (x, -pi, pi)) == pi**2
    func = cos(k*x)*sqrt(x**2)
    assert integrate(func, (x, -pi, pi)) == Piecewise(
        (2*(-1)**k/k**2 - 2/k**2, Ne(k, 0)), (pi**2, True))

def test_issue_6900():
    from itertools import permutations
    t0, t1, T, t = symbols('t0, t1 T t')
    f = Piecewise((0, t < t0), (x, And(t0 <= t, t < t1)), (0, t >= t1))
    g = f.integrate(t)
    assert g == Piecewise(
        (0, t <= t0),
        (t*x - t0*x, t <= Max(t0, t1)),
        (-t0*x + x*Max(t0, t1), True))
    for i in permutations(range(2)):
        reps = dict(zip((t0,t1), i))
        for tt in range(-1,3):
            assert (g.xreplace(reps).subs(t,tt) ==
                f.xreplace(reps).integrate(t).subs(t,tt))
    lim = Tuple(t, t0, T)
    g = f.integrate(lim)
    ans = Piecewise(
        (-t0*x + x*Min(T, Max(t0, t1)), T > t0),
        (0, True))
    for i in permutations(range(3)):
        reps = dict(zip((t0,t1,T), i))
        tru = f.xreplace(reps).integrate(lim.xreplace(reps))
        assert tru == ans.xreplace(reps)
    assert g == ans


def test_issue_10122():
    assert solve(abs(x) + abs(x - 1) - 1 > 0, x
        ) == Or(And(-oo < x, x < S.Zero), And(S.One < x, x < oo))


def test_issue_4313():
    u = Piecewise((0, x <= 0), (1, x >= a), (x/a, True))
    e = (u - u.subs(x, y))**2/(x - y)**2
    M = Max(0, a)
    assert integrate(e,  x).expand() == Piecewise(
        (Piecewise(
            (0, x <= 0),
            (-y**2/(a**2*x - a**2*y) + x/a**2 - 2*y*log(-y)/a**2 +
                2*y*log(x - y)/a**2 - y/a**2, x <= M),
            (-y**2/(-a**2*y + a**2*M) + 1/(-y + M) -
                1/(x - y) - 2*y*log(-y)/a**2 + 2*y*log(-y +
                M)/a**2 - y/a**2 + M/a**2, True)),
        ((a <= y) & (y <= 0)) | ((y <= 0) & (y > -oo))),
        (Piecewise(
            (-1/(x - y), x <= 0),
            (-a**2/(a**2*x - a**2*y) + 2*a*y/(a**2*x - a**2*y) -
                y**2/(a**2*x - a**2*y) + 2*log(-y)/a - 2*log(x - y)/a +
                2/a + x/a**2 - 2*y*log(-y)/a**2 + 2*y*log(x - y)/a**2 -
                y/a**2, x <= M),
            (-a**2/(-a**2*y + a**2*M) + 2*a*y/(-a**2*y +
                a**2*M) - y**2/(-a**2*y + a**2*M) +
                2*log(-y)/a - 2*log(-y + M)/a + 2/a -
                2*y*log(-y)/a**2 + 2*y*log(-y + M)/a**2 -
                y/a**2 + M/a**2, True)),
        a <= y),
        (Piecewise(
            (-y**2/(a**2*x - a**2*y), x <= 0),
            (x/a**2 + y/a**2, x <= M),
            (a**2/(-a**2*y + a**2*M) -
                a**2/(a**2*x - a**2*y) - 2*a*y/(-a**2*y + a**2*M) +
                2*a*y/(a**2*x - a**2*y) + y**2/(-a**2*y + a**2*M) -
                y**2/(a**2*x - a**2*y) + y/a**2 + M/a**2, True)),
        True))


def test__intervals():
    assert Piecewise((x + 2, Eq(x, 3)))._intervals(x) == (True, [])
    assert Piecewise(
        (1, x > x + 1),
        (Piecewise((1, x < x + 1)), 2*x < 2*x + 1),
        (1, True))._intervals(x) == (True, [(-oo, oo, 1, 1)])
    assert Piecewise((1, Ne(x, I)), (0, True))._intervals(x) == (True,
        [(-oo, oo, 1, 0)])
    assert Piecewise((-cos(x), sin(x) >= 0), (cos(x), True)
        )._intervals(x) == (True,
        [(0, pi, -cos(x), 0), (-oo, oo, cos(x), 1)])
    # the following tests that duplicates are removed and that non-Eq
    # generated zero-width intervals are removed
    assert Piecewise((1, Abs(x**(-2)) > 1), (0, True)
        )._intervals(x) == (True,
        [(-1, 0, 1, 0), (0, 1, 1, 0), (-oo, oo, 0, 1)])


def test_containment():
    a, b, c, d, e = [1, 2, 3, 4, 5]
    p = (Piecewise((d, x > 1), (e, True))*
        Piecewise((a, Abs(x - 1) < 1), (b, Abs(x - 2) < 2), (c, True)))
    assert p.integrate(x).diff(x) == Piecewise(
        (c*e, x <= 0),
        (a*e, x <= 1),
        (a*d, x < 2),  # this is what we want to get right
        (b*d, x < 4),
        (c*d, True))


def test_piecewise_with_DiracDelta():
    d1 = DiracDelta(x - 1)
    assert integrate(d1, (x, -oo, oo)) == 1
    assert integrate(d1, (x, 0, 2)) == 1
    assert Piecewise((d1, Eq(x, 2)), (0, True)).integrate(x) == 0
    assert Piecewise((d1, x < 2), (0, True)).integrate(x) == Piecewise(
        (Heaviside(x - 1), x < 2), (1, True))
    # TODO raise error if function is discontinuous at limit of
    # integration, e.g. integrate(d1, (x, -2, 1)) or Piecewise(
    # (d1, Eq(x, 1)


def test_issue_10258():
    assert Piecewise((0, x < 1), (1, True)).is_zero is None
    assert Piecewise((-1, x < 1), (1, True)).is_zero is False
    a = Symbol('a', zero=True)
    assert Piecewise((0, x < 1), (a, True)).is_zero
    assert Piecewise((1, x < 1), (a, x < 3)).is_zero is None
    a = Symbol('a')
    assert Piecewise((0, x < 1), (a, True)).is_zero is None
    assert Piecewise((0, x < 1), (1, True)).is_nonzero is None
    assert Piecewise((1, x < 1), (2, True)).is_nonzero
    assert Piecewise((0, x < 1), (oo, True)).is_finite is None
    assert Piecewise((0, x < 1), (1, True)).is_finite
    b = Basic()
    assert Piecewise((b, x < 1)).is_finite is None

    # 10258
    c = Piecewise((1, x < 0), (2, True)) < 3
    assert c != True
    assert piecewise_fold(c) == True


def test_issue_10087():
    a, b = Piecewise((x, x > 1), (2, True)), Piecewise((x, x > 3), (3, True))
    m = a*b
    f = piecewise_fold(m)
    for i in (0, 2, 4):
        assert m.subs(x, i) == f.subs(x, i)
    m = a + b
    f = piecewise_fold(m)
    for i in (0, 2, 4):
        assert m.subs(x, i) == f.subs(x, i)


def test_issue_8919():
    c = symbols('c:5')
    x = symbols("x")
    f1 = Piecewise((c[1], x < 1), (c[2], True))
    f2 = Piecewise((c[3], x < Rational(1, 3)), (c[4], True))
    assert integrate(f1*f2, (x, 0, 2)
        ) == c[1]*c[3]/3 + 2*c[1]*c[4]/3 + c[2]*c[4]
    f1 = Piecewise((0, x < 1), (2, True))
    f2 = Piecewise((3, x < 2), (0, True))
    assert integrate(f1*f2, (x, 0, 3)) == 6

    y = symbols("y", positive=True)
    a, b, c, x, z = symbols("a,b,c,x,z", real=True)
    I = Integral(Piecewise(
        (0, (x >= y) | (x < 0) | (b > c)),
        (a, True)), (x, 0, z))
    ans = I.doit()
    assert ans == Piecewise((0, b > c), (a*Min(y, z) - a*Min(0, z), True))
    for cond in (True, False):
        for yy in range(1, 3):
            for zz in range(-yy, 0, yy):
                reps = [(b > c, cond), (y, yy), (z, zz)]
                assert ans.subs(reps) == I.subs(reps).doit()


def test_unevaluated_integrals():
    f = Function('f')
    p = Piecewise((1, Eq(f(x) - 1, 0)), (2, x - 10 < 0), (0, True))
    assert p.integrate(x) == Integral(p, x)
    assert p.integrate((x, 0, 5)) == Integral(p, (x, 0, 5))
    # test it by replacing f(x) with x%2 which will not
    # affect the answer: the integrand is essentially 2 over
    # the domain of integration
    assert Integral(p, (x, 0, 5)).subs(f(x), x%2).n() == 10.0

    # this is a test of using _solve_inequality when
    # solve_univariate_inequality fails
    assert p.integrate(y) == Piecewise(
        (y, Eq(f(x), 1) | ((x < 10) & Eq(f(x), 1))),
        (2*y, (x > -oo) & (x < 10)), (0, True))


def test_conditions_as_alternate_booleans():
    a, b, c = symbols('a:c')
    assert Piecewise((x, Piecewise((y < 1, x > 0), (y > 1, True)))
        ) == Piecewise((x, ITE(x > 0, y < 1, y > 1)))


def test_Piecewise_rewrite_as_ITE():
    a, b, c, d = symbols('a:d')

    def _ITE(*args):
        return Piecewise(*args).rewrite(ITE)

    assert _ITE((a, x < 1), (b, x >= 1)) == ITE(x < 1, a, b)
    assert _ITE((a, x < 1), (b, x < oo)) == ITE(x < 1, a, b)
    assert _ITE((a, x < 1), (b, Or(y < 1, x < oo)), (c, y > 0)
               ) == ITE(x < 1, a, b)
    assert _ITE((a, x < 1), (b, True)) == ITE(x < 1, a, b)
    assert _ITE((a, x < 1), (b, x < 2), (c, True)
               ) == ITE(x < 1, a, ITE(x < 2, b, c))
    assert _ITE((a, x < 1), (b, y < 2), (c, True)
               ) == ITE(x < 1, a, ITE(y < 2, b, c))
    assert _ITE((a, x < 1), (b, x < oo), (c, y < 1)
               ) == ITE(x < 1, a, b)
    assert _ITE((a, x < 1), (c, y < 1), (b, x < oo), (d, True)
               ) == ITE(x < 1, a, ITE(y < 1, c, b))
    assert _ITE((a, x < 0), (b, Or(x < oo, y < 1))
               ) == ITE(x < 0, a, b)
    raises(TypeError, lambda: _ITE((x + 1, x < 1), (x, True)))
    # if `a` in the following were replaced with y then the coverage
    # is complete but something other than as_set would need to be
    # used to detect this
    raises(NotImplementedError, lambda: _ITE((x, x < y), (y, x >= a)))
    raises(ValueError, lambda: _ITE((a, x < 2), (b, x > 3)))


def test_Piecewise_replace_relational_27538():
    x, y = symbols('x, y')
    p1 = Piecewise(
        (0, Eq(x, True)),
        (1, True),
    )
    p2 = p1.xreplace({x: y < 1})
    assert p2.subs(y, 0) == 0
    assert p2.subs(y, 1) == 1


def test_issue_14052():
    assert integrate(abs(sin(x)), (x, 0, 2*pi)) == 4


def test_issue_14240():
    assert piecewise_fold(
        Piecewise((1, a), (2, b), (4, True)) +
        Piecewise((8, a), (16, True))
        ) == Piecewise((9, a), (18, b), (20, True))
    assert piecewise_fold(
        Piecewise((2, a), (3, b), (5, True)) *
        Piecewise((7, a), (11, True))
        ) == Piecewise((14, a), (33, b), (55, True))
    # these will hang if naive folding is used
    assert piecewise_fold(Add(*[
        Piecewise((i, a), (0, True)) for i in range(40)])
        ) == Piecewise((780, a), (0, True))
    assert piecewise_fold(Mul(*[
        Piecewise((i, a), (0, True)) for i in range(1, 41)])
        ) == Piecewise((factorial(40), a), (0, True))


def test_issue_14787():
    x = Symbol('x')
    f = Piecewise((x, x < 1), ((S(58) / 7), True))
    assert str(f.evalf()) == "Piecewise((x, x < 1), (8.28571428571429, True))"

def test_issue_21481():
    b, e = symbols('b e')
    C = Piecewise(
        (2,
        ((b > 1) & (e > 0)) |
        ((b > 0) & (b < 1) & (e < 0)) |
        ((e >= 2) & (b < -1) & Eq(Mod(e, 2), 0)) |
        ((e <= -2) & (b > -1) & (b < 0) & Eq(Mod(e, 2), 0))),
        (S.Half,
        ((b > 1) & (e < 0)) |
        ((b > 0) & (e > 0) & (b < 1)) |
        ((e <= -2) & (b < -1) & Eq(Mod(e, 2), 0)) |
        ((e >= 2) & (b > -1) & (b < 0) & Eq(Mod(e, 2), 0))),
        (-S.Half,
        Eq(Mod(e, 2), 1) &
        (((e <= -1) & (b < -1)) | ((e >= 1) & (b > -1) & (b < 0)))),
        (-2,
        ((e >= 1) & (b < -1) & Eq(Mod(e, 2), 1)) |
        ((e <= -1) & (b > -1) & (b < 0) & Eq(Mod(e, 2), 1)))
    )
    A = Piecewise(
        (1, Eq(b, 1) | Eq(e, 0) | (Eq(b, -1) & Eq(Mod(e, 2), 0))),
        (0, Eq(b, 0) & (e > 0)),
        (-1, Eq(b, -1) & Eq(Mod(e, 2), 1)),
        (C, Eq(im(b), 0) & Eq(im(e), 0))
    )

    B = piecewise_fold(A)
    sa = A.simplify()
    sb = B.simplify()
    v = (-2, -1, -S.Half, 0, S.Half, 1, 2)
    for i in v:
        for j in v:
            r = {b:i, e:j}
            ok = [k.xreplace(r) for k in (A, B, sa, sb)]
            assert len(set(ok)) == 1


def test_issue_8458():
    x, y = symbols('x y')
    # Original issue
    p1 = Piecewise((0, Eq(x, 0)), (sin(x), True))
    assert p1.simplify() == sin(x)
    # Slightly larger variant
    p2 = Piecewise((x, Eq(x, 0)), (4*x + (y-2)**4, Eq(x, 0) & Eq(x+y, 2)), (sin(x), True))
    assert p2.simplify() == sin(x)
    # Test for problem highlighted during review
    p3 = Piecewise((x+1, Eq(x, -1)), (4*x + (y-2)**4, Eq(x, 0) & Eq(x+y, 2)), (sin(x), True))
    assert p3.simplify() == Piecewise((0, Eq(x, -1)), (sin(x), True))


def test_issue_16417():
    z = Symbol('z')
    assert unchanged(Piecewise, (1, Or(Eq(im(z), 0), Gt(re(z), 0))), (2, True))

    x = Symbol('x')
    assert unchanged(Piecewise, (S.Pi, re(x) < 0),
                 (0, Or(re(x) > 0, Ne(im(x), 0))),
                 (S.NaN, True))
    r = Symbol('r', real=True)
    p = Piecewise((S.Pi, re(r) < 0),
                 (0, Or(re(r) > 0, Ne(im(r), 0))),
                 (S.NaN, True))
    assert p == Piecewise((S.Pi, r < 0),
                 (0, r > 0),
                 (S.NaN, True), evaluate=False)
    # Does not work since imaginary != 0...
    #i = Symbol('i', imaginary=True)
    #p = Piecewise((S.Pi, re(i) < 0),
    #              (0, Or(re(i) > 0, Ne(im(i), 0))),
    #              (S.NaN, True))
    #assert p == Piecewise((0, Ne(im(i), 0)),
    #                      (S.NaN, True), evaluate=False)
    i = I*r
    p = Piecewise((S.Pi, re(i) < 0),
                  (0, Or(re(i) > 0, Ne(im(i), 0))),
                  (S.NaN, True))
    assert p == Piecewise((0, Ne(im(i), 0)),
                          (S.NaN, True), evaluate=False)
    assert p == Piecewise((0, Ne(r, 0)),
                          (S.NaN, True), evaluate=False)


def test_eval_rewrite_as_KroneckerDelta():
    x, y, z, n, t, m = symbols('x y z n t m')
    K = KroneckerDelta
    f = lambda p: expand(p.rewrite(K))

    p1 = Piecewise((0, Eq(x, y)), (1, True))
    assert f(p1) == 1 - K(x, y)

    p2 = Piecewise((x, Eq(y,0)), (z, Eq(t,0)), (n, True))
    assert f(p2) == n*K(0, t)*K(0, y) - n*K(0, t) - n*K(0, y) + n + \
           x*K(0, y) - z*K(0, t)*K(0, y) + z*K(0, t)

    p3 = Piecewise((1, Ne(x, y)), (0, True))
    assert f(p3) == 1 - K(x, y)

    p4 = Piecewise((1, Eq(x, 3)), (4, True), (5, True))
    assert f(p4) == 4 - 3*K(3, x)

    p5 = Piecewise((3, Ne(x, 2)), (4, Eq(y, 2)), (5, True))
    assert f(p5) == -K(2, x)*K(2, y) + 2*K(2, x) + 3

    p6 = Piecewise((0, Ne(x, 1) & Ne(y, 4)), (1, True))
    assert f(p6) == -K(1, x)*K(4, y) + K(1, x) + K(4, y)

    p7 = Piecewise((2, Eq(y, 3) & Ne(x, 2)), (1, True))
    assert f(p7) == -K(2, x)*K(3, y) + K(3, y) + 1

    p8 = Piecewise((4, Eq(x, 3) & Ne(y, 2)), (1, True))
    assert f(p8) == -3*K(2, y)*K(3, x) + 3*K(3, x) + 1

    p9 = Piecewise((6, Eq(x, 4) & Eq(y, 1)), (1, True))
    assert f(p9) == 5 * K(1, y) * K(4, x) + 1

    p10 = Piecewise((4, Ne(x, -4) | Ne(y, 1)), (1, True))
    assert f(p10) == -3 * K(-4, x) * K(1, y) + 4

    p11 = Piecewise((1, Eq(y, 2) | Ne(x, -3)), (2, True))
    assert f(p11) == -K(-3, x)*K(2, y) + K(-3, x) + 1

    p12 = Piecewise((-1, Eq(x, 1) | Ne(y, 3)), (1, True))
    assert f(p12) == -2*K(1, x)*K(3, y) + 2*K(3, y) - 1

    p13 = Piecewise((3, Eq(x, 2) | Eq(y, 4)), (1, True))
    assert f(p13) == -2*K(2, x)*K(4, y) + 2*K(2, x) + 2*K(4, y) + 1

    p14 = Piecewise((1, Ne(x, 0) | Ne(y, 1)), (3, True))
    assert f(p14) == 2 * K(0, x) * K(1, y) + 1

    p15 = Piecewise((2, Eq(x, 3) | Ne(y, 2)), (3, Eq(x, 4) & Eq(y, 5)), (1, True))
    assert f(p15) == -2*K(2, y)*K(3, x)*K(4, x)*K(5, y) + K(2, y)*K(3, x) + \
           2*K(2, y)*K(4, x)*K(5, y) - K(2, y) + 2

    p16 = Piecewise((0, Ne(m, n)), (1, True))*Piecewise((0, Ne(n, t)), (1, True))\
          *Piecewise((0, Ne(n, x)), (1, True)) - Piecewise((0, Ne(t, x)), (1, True))
    assert f(p16) == K(m, n)*K(n, t)*K(n, x) - K(t, x)

    p17 = Piecewise((0, Ne(t, x) & (Ne(m, n) | Ne(n, t) | Ne(n, x))),
                    (1, Ne(t, x)), (-1, Ne(m, n) | Ne(n, t) | Ne(n, x)), (0, True))
    assert f(p17) == K(m, n)*K(n, t)*K(n, x) - K(t, x)

    p18 = Piecewise((-4, Eq(y, 1) | (Eq(x, -5) & Eq(x, z))), (4, True))
    assert f(p18) == 8*K(-5, x)*K(1, y)*K(x, z) - 8*K(-5, x)*K(x, z) - 8*K(1, y) + 4

    p19 = Piecewise((0, x > 2), (1, True))
    assert f(p19) == p19

    p20 = Piecewise((0, And(x < 2, x > -5)), (1, True))
    assert f(p20) == p20

    p21 = Piecewise((0, Or(x > 1, x < 0)), (1, True))
    assert f(p21) == p21

    p22 = Piecewise((0, ~((Eq(y, -1) | Ne(x, 0)) & (Ne(x, 1) | Ne(y, -1)))), (1, True))
    assert f(p22) == K(-1, y)*K(0, x) - K(-1, y)*K(1, x) - K(0, x) + 1


@slow
def test_identical_conds_issue():
    from sympy.stats import Uniform, density
    u1 = Uniform('u1', 0, 1)
    u2 = Uniform('u2', 0, 1)
    # Result is quite big, so not really important here (and should ideally be
    # simpler). Should not give an exception though.
    density(u1 + u2)


def test_issue_7370():
    f = Piecewise((1, x <= 2400))
    v = integrate(f, (x, 0, Float("252.4", 30)))
    assert str(v) == '252.400000000000000000000000000'


def test_issue_14933():
    x = Symbol('x')
    y = Symbol('y')

    inp = MatrixSymbol('inp', 1, 1)
    rep_dict = {y: inp[0, 0], x: inp[0, 0]}

    p = Piecewise((1, ITE(y > 0, x < 0, True)))
    assert p.xreplace(rep_dict) == Piecewise((1, ITE(inp[0, 0] > 0, inp[0, 0] < 0, True)))


def test_issue_16715():
    raises(NotImplementedError, lambda: Piecewise((x, x<0), (0, y>1)).as_expr_set_pairs())


def test_issue_20360():
    t, tau = symbols("t tau", real=True)
    n = symbols("n", integer=True)
    lam = pi * (n - S.Half)
    eq = integrate(exp(lam * tau), (tau, 0, t))
    assert eq.simplify() == (2*exp(pi*t*(2*n - 1)/2) - 2)/(pi*(2*n - 1))


def test_piecewise_eval():
    # XXX these tests might need modification if this
    # simplification is moved out of eval and into
    # boolalg or Piecewise simplification functions
    f = lambda x: x.args[0].cond
    # unsimplified
    assert f(Piecewise((x, (x > -oo) & (x < 3)))
        ) == ((x > -oo) & (x < 3))
    assert f(Piecewise((x, (x > -oo) & (x < oo)))
        ) == ((x > -oo) & (x < oo))
    assert f(Piecewise((x, (x > -3) & (x < 3)))
        ) == ((x > -3) & (x < 3))
    assert f(Piecewise((x, (x > -3) & (x < oo)))
        ) == ((x > -3) & (x < oo))
    assert f(Piecewise((x, (x <= 3) & (x > -oo)))
        ) == ((x <= 3) & (x > -oo))
    assert f(Piecewise((x, (x <= 3) & (x > -3)))
        ) == ((x <= 3) & (x > -3))
    assert f(Piecewise((x, (x >= -3) & (x < 3)))
        ) == ((x >= -3) & (x < 3))
    assert f(Piecewise((x, (x >= -3) & (x < oo)))
        ) == ((x >= -3) & (x < oo))
    assert f(Piecewise((x, (x >= -3) & (x <= 3)))
        ) == ((x >= -3) & (x <= 3))
    # could simplify by keeping only the first
    # arg of result
    assert f(Piecewise((x, (x <= oo) & (x > -oo)))
        ) == (x > -oo) & (x <= oo)
    assert f(Piecewise((x, (x <= oo) & (x > -3)))
        ) == (x > -3) & (x <= oo)
    assert f(Piecewise((x, (x >= -oo) & (x < 3)))
        ) == (x < 3) & (x >= -oo)
    assert f(Piecewise((x, (x >= -oo) & (x < oo)))
        ) == (x < oo) & (x >= -oo)
    assert f(Piecewise((x, (x >= -oo) & (x <= 3)))
        ) == (x <= 3) & (x >= -oo)
    assert f(Piecewise((x, (x >= -oo) & (x <= oo)))
        ) == (x <= oo) & (x >= -oo)  # but cannot be True unless x is real
    assert f(Piecewise((x, (x >= -3) & (x <= oo)))
        ) == (x >= -3) & (x <= oo)
    assert f(Piecewise((x, (Abs(arg(a)) <= 1) | (Abs(arg(a)) < 1)))
        ) == (Abs(arg(a)) <= 1) | (Abs(arg(a)) < 1)


def test_issue_22533():
    x = Symbol('x', real=True)
    f = Piecewise((-1 / x, x <= 0), (1 / x, True))
    assert integrate(f, x) == Piecewise((-log(x), x <= 0), (log(x), True))


def test_issue_24072():
    assert Piecewise((1, x > 1), (2, x <= 1), (3, x <= 1)
        ) == Piecewise((1, x > 1), (2, True))


def test_piecewise__eval_is_meromorphic():
    """ Issue 24127: Tests eval_is_meromorphic auxiliary method """
    x = symbols('x', real=True)
    f = Piecewise((1, x < 0), (sqrt(1 - x), True))
    assert f.is_meromorphic(x, I) is None
    assert f.is_meromorphic(x, -1) == True
    assert f.is_meromorphic(x, 0) == None
    assert f.is_meromorphic(x, 1) == False
    assert f.is_meromorphic(x, 2) == True
    assert f.is_meromorphic(x, Symbol('a')) == None
    assert f.is_meromorphic(x, Symbol('a', real=True)) == None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\tests\test_codegen.py ===
from io import StringIO

from sympy.core import symbols, Eq, pi, Catalan, Lambda, Dummy
from sympy.core.relational import Equality
from sympy.core.symbol import Symbol
from sympy.functions.special.error_functions import erf
from sympy.integrals.integrals import Integral
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import (
    codegen, make_routine, CCodeGen, C89CodeGen, C99CodeGen, InputArgument,
    CodeGenError, FCodeGen, CodeGenArgumentListError, OutputArgument,
    InOutArgument)
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function

#FIXME: Fails due to circular import in with core
# from sympy import codegen


def get_string(dump_fn, routines, prefix="file", header=False, empty=False):
    """Wrapper for dump_fn. dump_fn writes its results to a stream object and
       this wrapper returns the contents of that stream as a string. This
       auxiliary function is used by many tests below.

       The header and the empty lines are not generated to facilitate the
       testing of the output.
    """
    output = StringIO()
    dump_fn(routines, output, prefix, header, empty)
    source = output.getvalue()
    output.close()
    return source


def test_Routine_argument_order():
    a, x, y, z = symbols('a x y z')
    expr = (x + y)*z
    raises(CodeGenArgumentListError, lambda: make_routine("test", expr,
           argument_sequence=[z, x]))
    raises(CodeGenArgumentListError, lambda: make_routine("test", Eq(a,
           expr), argument_sequence=[z, x, y]))
    r = make_routine('test', Eq(a, expr), argument_sequence=[z, x, a, y])
    assert [ arg.name for arg in r.arguments ] == [z, x, a, y]
    assert [ type(arg) for arg in r.arguments ] == [
        InputArgument, InputArgument, OutputArgument, InputArgument  ]
    r = make_routine('test', Eq(z, expr), argument_sequence=[z, x, y])
    assert [ type(arg) for arg in r.arguments ] == [
        InOutArgument, InputArgument, InputArgument ]

    from sympy.tensor import IndexedBase, Idx
    A, B = map(IndexedBase, ['A', 'B'])
    m = symbols('m', integer=True)
    i = Idx('i', m)
    r = make_routine('test', Eq(A[i], B[i]), argument_sequence=[B, A, m])
    assert [ arg.name for arg in r.arguments ] == [B.label, A.label, m]

    expr = Integral(x*y*z, (x, 1, 2), (y, 1, 3))
    r = make_routine('test', Eq(a, expr), argument_sequence=[z, x, a, y])
    assert [ arg.name for arg in r.arguments ] == [z, x, a, y]


def test_empty_c_code():
    code_gen = C89CodeGen()
    source = get_string(code_gen.dump_c, [])
    assert source == "#include \"file.h\"\n#include <math.h>\n"


def test_empty_c_code_with_comment():
    code_gen = C89CodeGen()
    source = get_string(code_gen.dump_c, [], header=True)
    assert source[:82] == (
        "/******************************************************************************\n *"
    )
          #   "                    Code generated with SymPy 0.7.2-git                    "
    assert source[158:] == (                                                              "*\n"
            " *                                                                            *\n"
            " *              See http://www.sympy.org/ for more information.               *\n"
            " *                                                                            *\n"
            " *                       This file is part of 'project'                       *\n"
            " ******************************************************************************/\n"
            "#include \"file.h\"\n"
            "#include <math.h>\n"
            )


def test_empty_c_header():
    code_gen = C99CodeGen()
    source = get_string(code_gen.dump_h, [])
    assert source == "#ifndef PROJECT__FILE__H\n#define PROJECT__FILE__H\n#endif\n"


def test_simple_c_code():
    x, y, z = symbols('x,y,z')
    expr = (x + y)*z
    routine = make_routine("test", expr)
    code_gen = C89CodeGen()
    source = get_string(code_gen.dump_c, [routine])
    expected = (
        "#include \"file.h\"\n"
        "#include <math.h>\n"
        "double test(double x, double y, double z) {\n"
        "   double test_result;\n"
        "   test_result = z*(x + y);\n"
        "   return test_result;\n"
        "}\n"
    )
    assert source == expected


def test_c_code_reserved_words():
    x, y, z = symbols('if, typedef, while')
    expr = (x + y) * z
    routine = make_routine("test", expr)
    code_gen = C99CodeGen()
    source = get_string(code_gen.dump_c, [routine])
    expected = (
        "#include \"file.h\"\n"
        "#include <math.h>\n"
        "double test(double if_, double typedef_, double while_) {\n"
        "   double test_result;\n"
        "   test_result = while_*(if_ + typedef_);\n"
        "   return test_result;\n"
        "}\n"
    )
    assert source == expected


def test_numbersymbol_c_code():
    routine = make_routine("test", pi**Catalan)
    code_gen = C89CodeGen()
    source = get_string(code_gen.dump_c, [routine])
    expected = (
        "#include \"file.h\"\n"
        "#include <math.h>\n"
        "double test() {\n"
        "   double test_result;\n"
        "   double const Catalan = %s;\n"
        "   test_result = pow(M_PI, Catalan);\n"
        "   return test_result;\n"
        "}\n"
    ) % Catalan.evalf(17)
    assert source == expected


def test_c_code_argument_order():
    x, y, z = symbols('x,y,z')
    expr = x + y
    routine = make_routine("test", expr, argument_sequence=[z, x, y])
    code_gen = C89CodeGen()
    source = get_string(code_gen.dump_c, [routine])
    expected = (
        "#include \"file.h\"\n"
        "#include <math.h>\n"
        "double test(double z, double x, double y) {\n"
        "   double test_result;\n"
        "   test_result = x + y;\n"
        "   return test_result;\n"
        "}\n"
    )
    assert source == expected


def test_simple_c_header():
    x, y, z = symbols('x,y,z')
    expr = (x + y)*z
    routine = make_routine("test", expr)
    code_gen = C89CodeGen()
    source = get_string(code_gen.dump_h, [routine])
    expected = (
        "#ifndef PROJECT__FILE__H\n"
        "#define PROJECT__FILE__H\n"
        "double test(double x, double y, double z);\n"
        "#endif\n"
    )
    assert source == expected


def test_simple_c_codegen():
    x, y, z = symbols('x,y,z')
    expr = (x + y)*z
    expected = [
        ("file.c",
        "#include \"file.h\"\n"
        "#include <math.h>\n"
        "double test(double x, double y, double z) {\n"
        "   double test_result;\n"
        "   test_result = z*(x + y);\n"
        "   return test_result;\n"
        "}\n"),
        ("file.h",
        "#ifndef PROJECT__FILE__H\n"
        "#define PROJECT__FILE__H\n"
        "double test(double x, double y, double z);\n"
        "#endif\n")
    ]
    result = codegen(("test", expr), "C", "file", header=False, empty=False)
    assert result == expected


def test_multiple_results_c():
    x, y, z = symbols('x,y,z')
    expr1 = (x + y)*z
    expr2 = (x - y)*z
    routine = make_routine(
        "test",
        [expr1, expr2]
    )
    code_gen = C99CodeGen()
    raises(CodeGenError, lambda: get_string(code_gen.dump_h, [routine]))


def test_no_results_c():
    raises(ValueError, lambda: make_routine("test", []))


def test_ansi_math1_codegen():
    # not included: log10
    from sympy.functions.elementary.complexes import Abs
    from sympy.functions.elementary.exponential import log
    from sympy.functions.elementary.hyperbolic import (cosh, sinh, tanh)
    from sympy.functions.elementary.integers import (ceiling, floor)
    from sympy.functions.elementary.miscellaneous import sqrt
    from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, tan)
    x = symbols('x')
    name_expr = [
        ("test_fabs", Abs(x)),
        ("test_acos", acos(x)),
        ("test_asin", asin(x)),
        ("test_atan", atan(x)),
        ("test_ceil", ceiling(x)),
        ("test_cos", cos(x)),
        ("test_cosh", cosh(x)),
        ("test_floor", floor(x)),
        ("test_log", log(x)),
        ("test_ln", log(x)),
        ("test_sin", sin(x)),
        ("test_sinh", sinh(x)),
        ("test_sqrt", sqrt(x)),
        ("test_tan", tan(x)),
        ("test_tanh", tanh(x)),
    ]
    result = codegen(name_expr, "C89", "file", header=False, empty=False)
    assert result[0][0] == "file.c"
    assert result[0][1] == (
        '#include "file.h"\n#include <math.h>\n'
        'double test_fabs(double x) {\n   double test_fabs_result;\n   test_fabs_result = fabs(x);\n   return test_fabs_result;\n}\n'
        'double test_acos(double x) {\n   double test_acos_result;\n   test_acos_result = acos(x);\n   return test_acos_result;\n}\n'
        'double test_asin(double x) {\n   double test_asin_result;\n   test_asin_result = asin(x);\n   return test_asin_result;\n}\n'
        'double test_atan(double x) {\n   double test_atan_result;\n   test_atan_result = atan(x);\n   return test_atan_result;\n}\n'
        'double test_ceil(double x) {\n   double test_ceil_result;\n   test_ceil_result = ceil(x);\n   return test_ceil_result;\n}\n'
        'double test_cos(double x) {\n   double test_cos_result;\n   test_cos_result = cos(x);\n   return test_cos_result;\n}\n'
        'double test_cosh(double x) {\n   double test_cosh_result;\n   test_cosh_result = cosh(x);\n   return test_cosh_result;\n}\n'
        'double test_floor(double x) {\n   double test_floor_result;\n   test_floor_result = floor(x);\n   return test_floor_result;\n}\n'
        'double test_log(double x) {\n   double test_log_result;\n   test_log_result = log(x);\n   return test_log_result;\n}\n'
        'double test_ln(double x) {\n   double test_ln_result;\n   test_ln_result = log(x);\n   return test_ln_result;\n}\n'
        'double test_sin(double x) {\n   double test_sin_result;\n   test_sin_result = sin(x);\n   return test_sin_result;\n}\n'
        'double test_sinh(double x) {\n   double test_sinh_result;\n   test_sinh_result = sinh(x);\n   return test_sinh_result;\n}\n'
        'double test_sqrt(double x) {\n   double test_sqrt_result;\n   test_sqrt_result = sqrt(x);\n   return test_sqrt_result;\n}\n'
        'double test_tan(double x) {\n   double test_tan_result;\n   test_tan_result = tan(x);\n   return test_tan_result;\n}\n'
        'double test_tanh(double x) {\n   double test_tanh_result;\n   test_tanh_result = tanh(x);\n   return test_tanh_result;\n}\n'
    )
    assert result[1][0] == "file.h"
    assert result[1][1] == (
        '#ifndef PROJECT__FILE__H\n#define PROJECT__FILE__H\n'
        'double test_fabs(double x);\ndouble test_acos(double x);\n'
        'double test_asin(double x);\ndouble test_atan(double x);\n'
        'double test_ceil(double x);\ndouble test_cos(double x);\n'
        'double test_cosh(double x);\ndouble test_floor(double x);\n'
        'double test_log(double x);\ndouble test_ln(double x);\n'
        'double test_sin(double x);\ndouble test_sinh(double x);\n'
        'double test_sqrt(double x);\ndouble test_tan(double x);\n'
        'double test_tanh(double x);\n#endif\n'
    )


def test_ansi_math2_codegen():
    # not included: frexp, ldexp, modf, fmod
    from sympy.functions.elementary.trigonometric import atan2
    x, y = symbols('x,y')
    name_expr = [
        ("test_atan2", atan2(x, y)),
        ("test_pow", x**y),
    ]
    result = codegen(name_expr, "C89", "file", header=False, empty=False)
    assert result[0][0] == "file.c"
    assert result[0][1] == (
        '#include "file.h"\n#include <math.h>\n'
        'double test_atan2(double x, double y) {\n   double test_atan2_result;\n   test_atan2_result = atan2(x, y);\n   return test_atan2_result;\n}\n'
        'double test_pow(double x, double y) {\n   double test_pow_result;\n   test_pow_result = pow(x, y);\n   return test_pow_result;\n}\n'
    )
    assert result[1][0] == "file.h"
    assert result[1][1] == (
        '#ifndef PROJECT__FILE__H\n#define PROJECT__FILE__H\n'
        'double test_atan2(double x, double y);\n'
        'double test_pow(double x, double y);\n'
        '#endif\n'
    )


def test_complicated_codegen():
    from sympy.functions.elementary.trigonometric import (cos, sin, tan)
    x, y, z = symbols('x,y,z')
    name_expr = [
        ("test1", ((sin(x) + cos(y) + tan(z))**7).expand()),
        ("test2", cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))))),
    ]
    result = codegen(name_expr, "C89", "file", header=False, empty=False)
    assert result[0][0] == "file.c"
    assert result[0][1] == (
        '#include "file.h"\n#include <math.h>\n'
        'double test1(double x, double y, double z) {\n'
        '   double test1_result;\n'
        '   test1_result = '
        'pow(sin(x), 7) + '
        '7*pow(sin(x), 6)*cos(y) + '
        '7*pow(sin(x), 6)*tan(z) + '
        '21*pow(sin(x), 5)*pow(cos(y), 2) + '
        '42*pow(sin(x), 5)*cos(y)*tan(z) + '
        '21*pow(sin(x), 5)*pow(tan(z), 2) + '
        '35*pow(sin(x), 4)*pow(cos(y), 3) + '
        '105*pow(sin(x), 4)*pow(cos(y), 2)*tan(z) + '
        '105*pow(sin(x), 4)*cos(y)*pow(tan(z), 2) + '
        '35*pow(sin(x), 4)*pow(tan(z), 3) + '
        '35*pow(sin(x), 3)*pow(cos(y), 4) + '
        '140*pow(sin(x), 3)*pow(cos(y), 3)*tan(z) + '
        '210*pow(sin(x), 3)*pow(cos(y), 2)*pow(tan(z), 2) + '
        '140*pow(sin(x), 3)*cos(y)*pow(tan(z), 3) + '
        '35*pow(sin(x), 3)*pow(tan(z), 4) + '
        '21*pow(sin(x), 2)*pow(cos(y), 5) + '
        '105*pow(sin(x), 2)*pow(cos(y), 4)*tan(z) + '
        '210*pow(sin(x), 2)*pow(cos(y), 3)*pow(tan(z), 2) + '
        '210*pow(sin(x), 2)*pow(cos(y), 2)*pow(tan(z), 3) + '
        '105*pow(sin(x), 2)*cos(y)*pow(tan(z), 4) + '
        '21*pow(sin(x), 2)*pow(tan(z), 5) + '
        '7*sin(x)*pow(cos(y), 6) + '
        '42*sin(x)*pow(cos(y), 5)*tan(z) + '
        '105*sin(x)*pow(cos(y), 4)*pow(tan(z), 2) + '
        '140*sin(x)*pow(cos(y), 3)*pow(tan(z), 3) + '
        '105*sin(x)*pow(cos(y), 2)*pow(tan(z), 4) + '
        '42*sin(x)*cos(y)*pow(tan(z), 5) + '
        '7*sin(x)*pow(tan(z), 6) + '
        'pow(cos(y), 7) + '
        '7*pow(cos(y), 6)*tan(z) + '
        '21*pow(cos(y), 5)*pow(tan(z), 2) + '
        '35*pow(cos(y), 4)*pow(tan(z), 3) + '
        '35*pow(cos(y), 3)*pow(tan(z), 4) + '
        '21*pow(cos(y), 2)*pow(tan(z), 5) + '
        '7*cos(y)*pow(tan(z), 6) + '
        'pow(tan(z), 7);\n'
        '   return test1_result;\n'
        '}\n'
        'double test2(double x, double y, double z) {\n'
        '   double test2_result;\n'
        '   test2_result = cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))));\n'
        '   return test2_result;\n'
        '}\n'
    )
    assert result[1][0] == "file.h"
    assert result[1][1] == (
        '#ifndef PROJECT__FILE__H\n'
        '#define PROJECT__FILE__H\n'
        'double test1(double x, double y, double z);\n'
        'double test2(double x, double y, double z);\n'
        '#endif\n'
    )


def test_loops_c():
    from sympy.tensor import IndexedBase, Idx
    from sympy.core.symbol import symbols
    n, m = symbols('n m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)

    (f1, code), (f2, interface) = codegen(
        ('matrix_vector', Eq(y[i], A[i, j]*x[j])), "C99", "file", header=False, empty=False)

    assert f1 == 'file.c'
    expected = (
        '#include "file.h"\n'
        '#include <math.h>\n'
        'void matrix_vector(double *A, int m, int n, double *x, double *y) {\n'
        '   for (int i=0; i<m; i++){\n'
        '      y[i] = 0;\n'
        '   }\n'
        '   for (int i=0; i<m; i++){\n'
        '      for (int j=0; j<n; j++){\n'
        '         y[i] = %(rhs)s + y[i];\n'
        '      }\n'
        '   }\n'
        '}\n'
    )

    assert (code == expected % {'rhs': 'A[%s]*x[j]' % (i*n + j)} or
            code == expected % {'rhs': 'A[%s]*x[j]' % (j + i*n)} or
            code == expected % {'rhs': 'x[j]*A[%s]' % (i*n + j)} or
            code == expected % {'rhs': 'x[j]*A[%s]' % (j + i*n)})
    assert f2 == 'file.h'
    assert interface == (
        '#ifndef PROJECT__FILE__H\n'
        '#define PROJECT__FILE__H\n'
        'void matrix_vector(double *A, int m, int n, double *x, double *y);\n'
        '#endif\n'
    )


def test_dummy_loops_c():
    from sympy.tensor import IndexedBase, Idx
    i, m = symbols('i m', integer=True, cls=Dummy)
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx(i, m)
    expected = (
        '#include "file.h"\n'
        '#include <math.h>\n'
        'void test_dummies(int m_%(mno)i, double *x, double *y) {\n'
        '   for (int i_%(ino)i=0; i_%(ino)i<m_%(mno)i; i_%(ino)i++){\n'
        '      y[i_%(ino)i] = x[i_%(ino)i];\n'
        '   }\n'
        '}\n'
    ) % {'ino': i.label.dummy_index, 'mno': m.dummy_index}
    r = make_routine('test_dummies', Eq(y[i], x[i]))
    c89 = C89CodeGen()
    c99 = C99CodeGen()
    code = get_string(c99.dump_c, [r])
    assert code == expected
    with raises(NotImplementedError):
        get_string(c89.dump_c, [r])

def test_partial_loops_c():
    # check that loop boundaries are determined by Idx, and array strides
    # determined by shape of IndexedBase object.
    from sympy.tensor import IndexedBase, Idx
    from sympy.core.symbol import symbols
    n, m, o, p = symbols('n m o p', integer=True)
    A = IndexedBase('A', shape=(m, p))
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx('i', (o, m - 5))  # Note: bounds are inclusive
    j = Idx('j', n)          # dimension n corresponds to bounds (0, n - 1)

    (f1, code), (f2, interface) = codegen(
        ('matrix_vector', Eq(y[i], A[i, j]*x[j])), "C99", "file", header=False, empty=False)

    assert f1 == 'file.c'
    expected = (
        '#include "file.h"\n'
        '#include <math.h>\n'
        'void matrix_vector(double *A, int m, int n, int o, int p, double *x, double *y) {\n'
        '   for (int i=o; i<%(upperi)s; i++){\n'
        '      y[i] = 0;\n'
        '   }\n'
        '   for (int i=o; i<%(upperi)s; i++){\n'
        '      for (int j=0; j<n; j++){\n'
        '         y[i] = %(rhs)s + y[i];\n'
        '      }\n'
        '   }\n'
        '}\n'
    ) % {'upperi': m - 4, 'rhs': '%(rhs)s'}

    assert (code == expected % {'rhs': 'A[%s]*x[j]' % (i*p + j)} or
            code == expected % {'rhs': 'A[%s]*x[j]' % (j + i*p)} or
            code == expected % {'rhs': 'x[j]*A[%s]' % (i*p + j)} or
            code == expected % {'rhs': 'x[j]*A[%s]' % (j + i*p)})
    assert f2 == 'file.h'
    assert interface == (
        '#ifndef PROJECT__FILE__H\n'
        '#define PROJECT__FILE__H\n'
        'void matrix_vector(double *A, int m, int n, int o, int p, double *x, double *y);\n'
        '#endif\n'
    )


def test_output_arg_c():
    from sympy.core.relational import Equality
    from sympy.functions.elementary.trigonometric import (cos, sin)
    x, y, z = symbols("x,y,z")
    r = make_routine("foo", [Equality(y, sin(x)), cos(x)])
    c = C89CodeGen()
    result = c.write([r], "test", header=False, empty=False)
    assert result[0][0] == "test.c"
    expected = (
        '#include "test.h"\n'
        '#include <math.h>\n'
        'double foo(double x, double *y) {\n'
        '   (*y) = sin(x);\n'
        '   double foo_result;\n'
        '   foo_result = cos(x);\n'
        '   return foo_result;\n'
        '}\n'
    )
    assert result[0][1] == expected


def test_output_arg_c_reserved_words():
    from sympy.core.relational import Equality
    from sympy.functions.elementary.trigonometric import (cos, sin)
    x, y, z = symbols("if, while, z")
    r = make_routine("foo", [Equality(y, sin(x)), cos(x)])
    c = C89CodeGen()
    result = c.write([r], "test", header=False, empty=False)
    assert result[0][0] == "test.c"
    expected = (
        '#include "test.h"\n'
        '#include <math.h>\n'
        'double foo(double if_, double *while_) {\n'
        '   (*while_) = sin(if_);\n'
        '   double foo_result;\n'
        '   foo_result = cos(if_);\n'
        '   return foo_result;\n'
        '}\n'
    )
    assert result[0][1] == expected


def test_multidim_c_argument_cse():
    A_sym = MatrixSymbol('A', 3, 3)
    b_sym = MatrixSymbol('b', 3, 1)
    A = Matrix(A_sym)
    b = Matrix(b_sym)
    c = A*b
    cgen = CCodeGen(project="test", cse=True)
    r = cgen.routine("c", c)
    r.arguments[-1].result_var = "out"
    r.arguments[-1]._name = "out"
    code = get_string(cgen.dump_c, [r], prefix="test")
    expected = (
        '#include "test.h"\n'
        "#include <math.h>\n"
        "void c(double *A, double *b, double *out) {\n"
        "   out[0] = A[0]*b[0] + A[1]*b[1] + A[2]*b[2];\n"
        "   out[1] = A[3]*b[0] + A[4]*b[1] + A[5]*b[2];\n"
        "   out[2] = A[6]*b[0] + A[7]*b[1] + A[8]*b[2];\n"
        "}\n"
    )
    assert code == expected


def test_ccode_results_named_ordered():
    x, y, z = symbols('x,y,z')
    B, C = symbols('B,C')
    A = MatrixSymbol('A', 1, 3)
    expr1 = Equality(A, Matrix([[1, 2, x]]))
    expr2 = Equality(C, (x + y)*z)
    expr3 = Equality(B, 2*x)
    name_expr = ("test", [expr1, expr2, expr3])
    expected = (
        '#include "test.h"\n'
        '#include <math.h>\n'
        'void test(double x, double *C, double z, double y, double *A, double *B) {\n'
        '   (*C) = z*(x + y);\n'
        '   A[0] = 1;\n'
        '   A[1] = 2;\n'
        '   A[2] = x;\n'
        '   (*B) = 2*x;\n'
        '}\n'
    )

    result = codegen(name_expr, "c", "test", header=False, empty=False,
                     argument_sequence=(x, C, z, y, A, B))
    source = result[0][1]
    assert source == expected


def test_ccode_matrixsymbol_slice():
    A = MatrixSymbol('A', 5, 3)
    B = MatrixSymbol('B', 1, 3)
    C = MatrixSymbol('C', 1, 3)
    D = MatrixSymbol('D', 5, 1)
    name_expr = ("test", [Equality(B, A[0, :]),
                          Equality(C, A[1, :]),
                          Equality(D, A[:, 2])])
    result = codegen(name_expr, "c99", "test", header=False, empty=False)
    source = result[0][1]
    expected = (
        '#include "test.h"\n'
        '#include <math.h>\n'
        'void test(double *A, double *B, double *C, double *D) {\n'
        '   B[0] = A[0];\n'
        '   B[1] = A[1];\n'
        '   B[2] = A[2];\n'
        '   C[0] = A[3];\n'
        '   C[1] = A[4];\n'
        '   C[2] = A[5];\n'
        '   D[0] = A[2];\n'
        '   D[1] = A[5];\n'
        '   D[2] = A[8];\n'
        '   D[3] = A[11];\n'
        '   D[4] = A[14];\n'
        '}\n'
    )
    assert source == expected

def test_ccode_cse():
    a, b, c, d = symbols('a b c d')
    e = MatrixSymbol('e', 3, 1)
    name_expr = ("test", [Equality(e, Matrix([[a*b], [a*b + c*d], [a*b*c*d]]))])
    generator = CCodeGen(cse=True)
    result = codegen(name_expr, code_gen=generator, header=False, empty=False)
    source = result[0][1]
    expected = (
        '#include "test.h"\n'
        '#include <math.h>\n'
        'void test(double a, double b, double c, double d, double *e) {\n'
        '   const double x0 = a*b;\n'
        '   const double x1 = c*d;\n'
        '   e[0] = x0;\n'
        '   e[1] = x0 + x1;\n'
        '   e[2] = x0*x1;\n'
        '}\n'
    )
    assert source == expected

def test_ccode_unused_array_arg():
    x = MatrixSymbol('x', 2, 1)
    # x does not appear in output
    name_expr = ("test", 1.0)
    generator = CCodeGen()
    result = codegen(name_expr, code_gen=generator, header=False, empty=False, argument_sequence=(x,))
    source = result[0][1]
    # note: x should appear as (double *)
    expected = (
        '#include "test.h"\n'
        '#include <math.h>\n'
        'double test(double *x) {\n'
        '   double test_result;\n'
        '   test_result = 1.0;\n'
        '   return test_result;\n'
        '}\n'
    )
    assert source == expected

def test_ccode_unused_array_arg_func():
    # issue 16689
    X = MatrixSymbol('X',3,1)
    Y = MatrixSymbol('Y',3,1)
    z = symbols('z',integer = True)
    name_expr = ('testBug', X[0] + X[1])
    result = codegen(name_expr, language='C', header=False, empty=False, argument_sequence=(X, Y, z))
    source = result[0][1]
    expected = (
        '#include "testBug.h"\n'
        '#include <math.h>\n'
        'double testBug(double *X, double *Y, int z) {\n'
        '   double testBug_result;\n'
        '   testBug_result = X[0] + X[1];\n'
        '   return testBug_result;\n'
        '}\n'
    )
    assert source == expected

def test_empty_f_code():
    code_gen = FCodeGen()
    source = get_string(code_gen.dump_f95, [])
    assert source == ""


def test_empty_f_code_with_header():
    code_gen = FCodeGen()
    source = get_string(code_gen.dump_f95, [], header=True)
    assert source[:82] == (
        "!******************************************************************************\n!*"
    )
          #   "                    Code generated with SymPy 0.7.2-git                    "
    assert source[158:] == (                                                              "*\n"
            "!*                                                                            *\n"
            "!*              See http://www.sympy.org/ for more information.               *\n"
            "!*                                                                            *\n"
            "!*                       This file is part of 'project'                       *\n"
            "!******************************************************************************\n"
            )


def test_empty_f_header():
    code_gen = FCodeGen()
    source = get_string(code_gen.dump_h, [])
    assert source == ""


def test_simple_f_code():
    x, y, z = symbols('x,y,z')
    expr = (x + y)*z
    routine = make_routine("test", expr)
    code_gen = FCodeGen()
    source = get_string(code_gen.dump_f95, [routine])
    expected = (
        "REAL*8 function test(x, y, z)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "REAL*8, intent(in) :: y\n"
        "REAL*8, intent(in) :: z\n"
        "test = z*(x + y)\n"
        "end function\n"
    )
    assert source == expected


def test_numbersymbol_f_code():
    routine = make_routine("test", pi**Catalan)
    code_gen = FCodeGen()
    source = get_string(code_gen.dump_f95, [routine])
    expected = (
        "REAL*8 function test()\n"
        "implicit none\n"
        "REAL*8, parameter :: Catalan = %sd0\n"
        "REAL*8, parameter :: pi = %sd0\n"
        "test = pi**Catalan\n"
        "end function\n"
    ) % (Catalan.evalf(17), pi.evalf(17))
    assert source == expected

def test_erf_f_code():
    x = symbols('x')
    routine = make_routine("test", erf(x) - erf(-2 * x))
    code_gen = FCodeGen()
    source = get_string(code_gen.dump_f95, [routine])
    expected = (
        "REAL*8 function test(x)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "test = erf(x) + erf(2.0d0*x)\n"
        "end function\n"
    )
    assert source == expected, source

def test_f_code_argument_order():
    x, y, z = symbols('x,y,z')
    expr = x + y
    routine = make_routine("test", expr, argument_sequence=[z, x, y])
    code_gen = FCodeGen()
    source = get_string(code_gen.dump_f95, [routine])
    expected = (
        "REAL*8 function test(z, x, y)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: z\n"
        "REAL*8, intent(in) :: x\n"
        "REAL*8, intent(in) :: y\n"
        "test = x + y\n"
        "end function\n"
    )
    assert source == expected


def test_simple_f_header():
    x, y, z = symbols('x,y,z')
    expr = (x + y)*z
    routine = make_routine("test", expr)
    code_gen = FCodeGen()
    source = get_string(code_gen.dump_h, [routine])
    expected = (
        "interface\n"
        "REAL*8 function test(x, y, z)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "REAL*8, intent(in) :: y\n"
        "REAL*8, intent(in) :: z\n"
        "end function\n"
        "end interface\n"
    )
    assert source == expected


def test_simple_f_codegen():
    x, y, z = symbols('x,y,z')
    expr = (x + y)*z
    result = codegen(
        ("test", expr), "F95", "file", header=False, empty=False)
    expected = [
        ("file.f90",
        "REAL*8 function test(x, y, z)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "REAL*8, intent(in) :: y\n"
        "REAL*8, intent(in) :: z\n"
        "test = z*(x + y)\n"
        "end function\n"),
        ("file.h",
        "interface\n"
        "REAL*8 function test(x, y, z)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "REAL*8, intent(in) :: y\n"
        "REAL*8, intent(in) :: z\n"
        "end function\n"
        "end interface\n")
    ]
    assert result == expected


def test_multiple_results_f():
    x, y, z = symbols('x,y,z')
    expr1 = (x + y)*z
    expr2 = (x - y)*z
    routine = make_routine(
        "test",
        [expr1, expr2]
    )
    code_gen = FCodeGen()
    raises(CodeGenError, lambda: get_string(code_gen.dump_h, [routine]))


def test_no_results_f():
    raises(ValueError, lambda: make_routine("test", []))


def test_intrinsic_math_codegen():
    # not included: log10
    from sympy.functions.elementary.complexes import Abs
    from sympy.functions.elementary.exponential import log
    from sympy.functions.elementary.hyperbolic import (cosh, sinh, tanh)
    from sympy.functions.elementary.miscellaneous import sqrt
    from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, tan)
    x = symbols('x')
    name_expr = [
        ("test_abs", Abs(x)),
        ("test_acos", acos(x)),
        ("test_asin", asin(x)),
        ("test_atan", atan(x)),
        ("test_cos", cos(x)),
        ("test_cosh", cosh(x)),
        ("test_log", log(x)),
        ("test_ln", log(x)),
        ("test_sin", sin(x)),
        ("test_sinh", sinh(x)),
        ("test_sqrt", sqrt(x)),
        ("test_tan", tan(x)),
        ("test_tanh", tanh(x)),
    ]
    result = codegen(name_expr, "F95", "file", header=False, empty=False)
    assert result[0][0] == "file.f90"
    expected = (
        'REAL*8 function test_abs(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_abs = abs(x)\n'
        'end function\n'
        'REAL*8 function test_acos(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_acos = acos(x)\n'
        'end function\n'
        'REAL*8 function test_asin(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_asin = asin(x)\n'
        'end function\n'
        'REAL*8 function test_atan(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_atan = atan(x)\n'
        'end function\n'
        'REAL*8 function test_cos(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_cos = cos(x)\n'
        'end function\n'
        'REAL*8 function test_cosh(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_cosh = cosh(x)\n'
        'end function\n'
        'REAL*8 function test_log(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_log = log(x)\n'
        'end function\n'
        'REAL*8 function test_ln(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_ln = log(x)\n'
        'end function\n'
        'REAL*8 function test_sin(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_sin = sin(x)\n'
        'end function\n'
        'REAL*8 function test_sinh(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_sinh = sinh(x)\n'
        'end function\n'
        'REAL*8 function test_sqrt(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_sqrt = sqrt(x)\n'
        'end function\n'
        'REAL*8 function test_tan(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_tan = tan(x)\n'
        'end function\n'
        'REAL*8 function test_tanh(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_tanh = tanh(x)\n'
        'end function\n'
    )
    assert result[0][1] == expected

    assert result[1][0] == "file.h"
    expected = (
        'interface\n'
        'REAL*8 function test_abs(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_acos(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_asin(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_atan(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_cos(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_cosh(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_log(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_ln(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_sin(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_sinh(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_sqrt(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_tan(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_tanh(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
    )
    assert result[1][1] == expected


def test_intrinsic_math2_codegen():
    # not included: frexp, ldexp, modf, fmod
    from sympy.functions.elementary.trigonometric import atan2
    x, y = symbols('x,y')
    name_expr = [
        ("test_atan2", atan2(x, y)),
        ("test_pow", x**y),
    ]
    result = codegen(name_expr, "F95", "file", header=False, empty=False)
    assert result[0][0] == "file.f90"
    expected = (
        'REAL*8 function test_atan2(x, y)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(in) :: y\n'
        'test_atan2 = atan2(x, y)\n'
        'end function\n'
        'REAL*8 function test_pow(x, y)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(in) :: y\n'
        'test_pow = x**y\n'
        'end function\n'
    )
    assert result[0][1] == expected

    assert result[1][0] == "file.h"
    expected = (
        'interface\n'
        'REAL*8 function test_atan2(x, y)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(in) :: y\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_pow(x, y)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(in) :: y\n'
        'end function\n'
        'end interface\n'
    )
    assert result[1][1] == expected


def test_complicated_codegen_f95():
    from sympy.functions.elementary.trigonometric import (cos, sin, tan)
    x, y, z = symbols('x,y,z')
    name_expr = [
        ("test1", ((sin(x) + cos(y) + tan(z))**7).expand()),
        ("test2", cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))))),
    ]
    result = codegen(name_expr, "F95", "file", header=False, empty=False)
    assert result[0][0] == "file.f90"
    expected = (
        'REAL*8 function test1(x, y, z)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(in) :: y\n'
        'REAL*8, intent(in) :: z\n'
        'test1 = sin(x)**7 + 7*sin(x)**6*cos(y) + 7*sin(x)**6*tan(z) + 21*sin(x) &\n'
        '      **5*cos(y)**2 + 42*sin(x)**5*cos(y)*tan(z) + 21*sin(x)**5*tan(z) &\n'
        '      **2 + 35*sin(x)**4*cos(y)**3 + 105*sin(x)**4*cos(y)**2*tan(z) + &\n'
        '      105*sin(x)**4*cos(y)*tan(z)**2 + 35*sin(x)**4*tan(z)**3 + 35*sin( &\n'
        '      x)**3*cos(y)**4 + 140*sin(x)**3*cos(y)**3*tan(z) + 210*sin(x)**3* &\n'
        '      cos(y)**2*tan(z)**2 + 140*sin(x)**3*cos(y)*tan(z)**3 + 35*sin(x) &\n'
        '      **3*tan(z)**4 + 21*sin(x)**2*cos(y)**5 + 105*sin(x)**2*cos(y)**4* &\n'
        '      tan(z) + 210*sin(x)**2*cos(y)**3*tan(z)**2 + 210*sin(x)**2*cos(y) &\n'
        '      **2*tan(z)**3 + 105*sin(x)**2*cos(y)*tan(z)**4 + 21*sin(x)**2*tan &\n'
        '      (z)**5 + 7*sin(x)*cos(y)**6 + 42*sin(x)*cos(y)**5*tan(z) + 105* &\n'
        '      sin(x)*cos(y)**4*tan(z)**2 + 140*sin(x)*cos(y)**3*tan(z)**3 + 105 &\n'
        '      *sin(x)*cos(y)**2*tan(z)**4 + 42*sin(x)*cos(y)*tan(z)**5 + 7*sin( &\n'
        '      x)*tan(z)**6 + cos(y)**7 + 7*cos(y)**6*tan(z) + 21*cos(y)**5*tan( &\n'
        '      z)**2 + 35*cos(y)**4*tan(z)**3 + 35*cos(y)**3*tan(z)**4 + 21*cos( &\n'
        '      y)**2*tan(z)**5 + 7*cos(y)*tan(z)**6 + tan(z)**7\n'
        'end function\n'
        'REAL*8 function test2(x, y, z)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(in) :: y\n'
        'REAL*8, intent(in) :: z\n'
        'test2 = cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))))\n'
        'end function\n'
    )
    assert result[0][1] == expected
    assert result[1][0] == "file.h"
    expected = (
        'interface\n'
        'REAL*8 function test1(x, y, z)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(in) :: y\n'
        'REAL*8, intent(in) :: z\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test2(x, y, z)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(in) :: y\n'
        'REAL*8, intent(in) :: z\n'
        'end function\n'
        'end interface\n'
    )
    assert result[1][1] == expected


def test_loops():
    from sympy.tensor import IndexedBase, Idx
    from sympy.core.symbol import symbols

    n, m = symbols('n,m', integer=True)
    A, x, y = map(IndexedBase, 'Axy')
    i = Idx('i', m)
    j = Idx('j', n)

    (f1, code), (f2, interface) = codegen(
        ('matrix_vector', Eq(y[i], A[i, j]*x[j])), "F95", "file", header=False, empty=False)

    assert f1 == 'file.f90'
    expected = (
        'subroutine matrix_vector(A, m, n, x, y)\n'
        'implicit none\n'
        'INTEGER*4, intent(in) :: m\n'
        'INTEGER*4, intent(in) :: n\n'
        'REAL*8, intent(in), dimension(1:m, 1:n) :: A\n'
        'REAL*8, intent(in), dimension(1:n) :: x\n'
        'REAL*8, intent(out), dimension(1:m) :: y\n'
        'INTEGER*4 :: i\n'
        'INTEGER*4 :: j\n'
        'do i = 1, m\n'
        '   y(i) = 0\n'
        'end do\n'
        'do i = 1, m\n'
        '   do j = 1, n\n'
        '      y(i) = %(rhs)s + y(i)\n'
        '   end do\n'
        'end do\n'
        'end subroutine\n'
    )

    assert code == expected % {'rhs': 'A(i, j)*x(j)'} or\
        code == expected % {'rhs': 'x(j)*A(i, j)'}
    assert f2 == 'file.h'
    assert interface == (
        'interface\n'
        'subroutine matrix_vector(A, m, n, x, y)\n'
        'implicit none\n'
        'INTEGER*4, intent(in) :: m\n'
        'INTEGER*4, intent(in) :: n\n'
        'REAL*8, intent(in), dimension(1:m, 1:n) :: A\n'
        'REAL*8, intent(in), dimension(1:n) :: x\n'
        'REAL*8, intent(out), dimension(1:m) :: y\n'
        'end subroutine\n'
        'end interface\n'
    )


def test_dummy_loops_f95():
    from sympy.tensor import IndexedBase, Idx
    i, m = symbols('i m', integer=True, cls=Dummy)
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx(i, m)
    expected = (
        'subroutine test_dummies(m_%(mcount)i, x, y)\n'
        'implicit none\n'
        'INTEGER*4, intent(in) :: m_%(mcount)i\n'
        'REAL*8, intent(in), dimension(1:m_%(mcount)i) :: x\n'
        'REAL*8, intent(out), dimension(1:m_%(mcount)i) :: y\n'
        'INTEGER*4 :: i_%(icount)i\n'
        'do i_%(icount)i = 1, m_%(mcount)i\n'
        '   y(i_%(icount)i) = x(i_%(icount)i)\n'
        'end do\n'
        'end subroutine\n'
    ) % {'icount': i.label.dummy_index, 'mcount': m.dummy_index}
    r = make_routine('test_dummies', Eq(y[i], x[i]))
    c = FCodeGen()
    code = get_string(c.dump_f95, [r])
    assert code == expected


def test_loops_InOut():
    from sympy.tensor import IndexedBase, Idx
    from sympy.core.symbol import symbols

    i, j, n, m = symbols('i,j,n,m', integer=True)
    A, x, y = symbols('A,x,y')
    A = IndexedBase(A)[Idx(i, m), Idx(j, n)]
    x = IndexedBase(x)[Idx(j, n)]
    y = IndexedBase(y)[Idx(i, m)]

    (f1, code), (f2, interface) = codegen(
        ('matrix_vector', Eq(y, y + A*x)), "F95", "file", header=False, empty=False)

    assert f1 == 'file.f90'
    expected = (
        'subroutine matrix_vector(A, m, n, x, y)\n'
        'implicit none\n'
        'INTEGER*4, intent(in) :: m\n'
        'INTEGER*4, intent(in) :: n\n'
        'REAL*8, intent(in), dimension(1:m, 1:n) :: A\n'
        'REAL*8, intent(in), dimension(1:n) :: x\n'
        'REAL*8, intent(inout), dimension(1:m) :: y\n'
        'INTEGER*4 :: i\n'
        'INTEGER*4 :: j\n'
        'do i = 1, m\n'
        '   do j = 1, n\n'
        '      y(i) = %(rhs)s + y(i)\n'
        '   end do\n'
        'end do\n'
        'end subroutine\n'
    )

    assert (code == expected % {'rhs': 'A(i, j)*x(j)'} or
            code == expected % {'rhs': 'x(j)*A(i, j)'})
    assert f2 == 'file.h'
    assert interface == (
        'interface\n'
        'subroutine matrix_vector(A, m, n, x, y)\n'
        'implicit none\n'
        'INTEGER*4, intent(in) :: m\n'
        'INTEGER*4, intent(in) :: n\n'
        'REAL*8, intent(in), dimension(1:m, 1:n) :: A\n'
        'REAL*8, intent(in), dimension(1:n) :: x\n'
        'REAL*8, intent(inout), dimension(1:m) :: y\n'
        'end subroutine\n'
        'end interface\n'
    )


def test_partial_loops_f():
    # check that loop boundaries are determined by Idx, and array strides
    # determined by shape of IndexedBase object.
    from sympy.tensor import IndexedBase, Idx
    from sympy.core.symbol import symbols
    n, m, o, p = symbols('n m o p', integer=True)
    A = IndexedBase('A', shape=(m, p))
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx('i', (o, m - 5))  # Note: bounds are inclusive
    j = Idx('j', n)          # dimension n corresponds to bounds (0, n - 1)

    (f1, code), (f2, interface) = codegen(
        ('matrix_vector', Eq(y[i], A[i, j]*x[j])), "F95", "file", header=False, empty=False)

    expected = (
        'subroutine matrix_vector(A, m, n, o, p, x, y)\n'
        'implicit none\n'
        'INTEGER*4, intent(in) :: m\n'
        'INTEGER*4, intent(in) :: n\n'
        'INTEGER*4, intent(in) :: o\n'
        'INTEGER*4, intent(in) :: p\n'
        'REAL*8, intent(in), dimension(1:m, 1:p) :: A\n'
        'REAL*8, intent(in), dimension(1:n) :: x\n'
        'REAL*8, intent(out), dimension(1:%(iup-ilow)s) :: y\n'
        'INTEGER*4 :: i\n'
        'INTEGER*4 :: j\n'
        'do i = %(ilow)s, %(iup)s\n'
        '   y(i) = 0\n'
        'end do\n'
        'do i = %(ilow)s, %(iup)s\n'
        '   do j = 1, n\n'
        '      y(i) = %(rhs)s + y(i)\n'
        '   end do\n'
        'end do\n'
        'end subroutine\n'
    ) % {
        'rhs': '%(rhs)s',
        'iup': str(m - 4),
        'ilow': str(1 + o),
        'iup-ilow': str(m - 4 - o)
    }

    assert code == expected % {'rhs': 'A(i, j)*x(j)'} or\
        code == expected % {'rhs': 'x(j)*A(i, j)'}


def test_output_arg_f():
    from sympy.core.relational import Equality
    from sympy.functions.elementary.trigonometric import (cos, sin)
    x, y, z = symbols("x,y,z")
    r = make_routine("foo", [Equality(y, sin(x)), cos(x)])
    c = FCodeGen()
    result = c.write([r], "test", header=False, empty=False)
    assert result[0][0] == "test.f90"
    assert result[0][1] == (
        'REAL*8 function foo(x, y)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(out) :: y\n'
        'y = sin(x)\n'
        'foo = cos(x)\n'
        'end function\n'
    )


def test_inline_function():
    from sympy.tensor import IndexedBase, Idx
    from sympy.core.symbol import symbols
    n, m = symbols('n m', integer=True)
    A, x, y = map(IndexedBase, 'Axy')
    i = Idx('i', m)
    p = FCodeGen()
    func = implemented_function('func', Lambda(n, n*(n + 1)))
    routine = make_routine('test_inline', Eq(y[i], func(x[i])))
    code = get_string(p.dump_f95, [routine])
    expected = (
        'subroutine test_inline(m, x, y)\n'
        'implicit none\n'
        'INTEGER*4, intent(in) :: m\n'
        'REAL*8, intent(in), dimension(1:m) :: x\n'
        'REAL*8, intent(out), dimension(1:m) :: y\n'
        'INTEGER*4 :: i\n'
        'do i = 1, m\n'
        '   y(i) = %s*%s\n'
        'end do\n'
        'end subroutine\n'
    )
    args = ('x(i)', '(x(i) + 1)')
    assert code == expected % args or\
        code == expected % args[::-1]


def test_f_code_call_signature_wrap():
    # Issue #7934
    x = symbols('x:20')
    expr = 0
    for sym in x:
        expr += sym
    routine = make_routine("test", expr)
    code_gen = FCodeGen()
    source = get_string(code_gen.dump_f95, [routine])
    expected = """\
REAL*8 function test(x0, x1, x10, x11, x12, x13, x14, x15, x16, x17, x18, &
      x19, x2, x3, x4, x5, x6, x7, x8, x9)
implicit none
REAL*8, intent(in) :: x0
REAL*8, intent(in) :: x1
REAL*8, intent(in) :: x10
REAL*8, intent(in) :: x11
REAL*8, intent(in) :: x12
REAL*8, intent(in) :: x13
REAL*8, intent(in) :: x14
REAL*8, intent(in) :: x15
REAL*8, intent(in) :: x16
REAL*8, intent(in) :: x17
REAL*8, intent(in) :: x18
REAL*8, intent(in) :: x19
REAL*8, intent(in) :: x2
REAL*8, intent(in) :: x3
REAL*8, intent(in) :: x4
REAL*8, intent(in) :: x5
REAL*8, intent(in) :: x6
REAL*8, intent(in) :: x7
REAL*8, intent(in) :: x8
REAL*8, intent(in) :: x9
test = x0 + x1 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + &
      x19 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9
end function
"""
    assert source == expected


def test_check_case():
    x, X = symbols('x,X')
    raises(CodeGenError, lambda: codegen(('test', x*X), 'f95', 'prefix'))


def test_check_case_false_positive():
    # The upper case/lower case exception should not be triggered by SymPy
    # objects that differ only because of assumptions.  (It may be useful to
    # have a check for that as well, but here we only want to test against
    # false positives with respect to case checking.)
    x1 = symbols('x')
    x2 = symbols('x', my_assumption=True)
    try:
        codegen(('test', x1*x2), 'f95', 'prefix')
    except CodeGenError as e:
        if e.args[0].startswith("Fortran ignores case."):
            raise AssertionError("This exception should not be raised!")


def test_c_fortran_omit_routine_name():
    x, y = symbols("x,y")
    name_expr = [("foo", 2*x)]
    result = codegen(name_expr, "F95", header=False, empty=False)
    expresult = codegen(name_expr, "F95", "foo", header=False, empty=False)
    assert result[0][1] == expresult[0][1]

    name_expr = ("foo", x*y)
    result = codegen(name_expr, "F95", header=False, empty=False)
    expresult = codegen(name_expr, "F95", "foo", header=False, empty=False)
    assert result[0][1] == expresult[0][1]

    name_expr = ("foo", Matrix([[x, y], [x+y, x-y]]))
    result = codegen(name_expr, "C89", header=False, empty=False)
    expresult = codegen(name_expr, "C89", "foo", header=False, empty=False)
    assert result[0][1] == expresult[0][1]


def test_fcode_matrix_output():
    x, y, z = symbols('x,y,z')
    e1 = x + y
    e2 = Matrix([[x, y], [z, 16]])
    name_expr = ("test", (e1, e2))
    result = codegen(name_expr, "f95", "test", header=False, empty=False)
    source = result[0][1]
    expected = (
        "REAL*8 function test(x, y, z, out_%(hash)s)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "REAL*8, intent(in) :: y\n"
        "REAL*8, intent(in) :: z\n"
        "REAL*8, intent(out), dimension(1:2, 1:2) :: out_%(hash)s\n"
        "out_%(hash)s(1, 1) = x\n"
        "out_%(hash)s(2, 1) = z\n"
        "out_%(hash)s(1, 2) = y\n"
        "out_%(hash)s(2, 2) = 16\n"
        "test = x + y\n"
        "end function\n"
    )
    # look for the magic number
    a = source.splitlines()[5]
    b = a.split('_')
    out = b[1]
    expected = expected % {'hash': out}
    assert source == expected


def test_fcode_results_named_ordered():
    x, y, z = symbols('x,y,z')
    B, C = symbols('B,C')
    A = MatrixSymbol('A', 1, 3)
    expr1 = Equality(A, Matrix([[1, 2, x]]))
    expr2 = Equality(C, (x + y)*z)
    expr3 = Equality(B, 2*x)
    name_expr = ("test", [expr1, expr2, expr3])
    result = codegen(name_expr, "f95", "test", header=False, empty=False,
                     argument_sequence=(x, z, y, C, A, B))
    source = result[0][1]
    expected = (
        "subroutine test(x, z, y, C, A, B)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "REAL*8, intent(in) :: z\n"
        "REAL*8, intent(in) :: y\n"
        "REAL*8, intent(out) :: C\n"
        "REAL*8, intent(out) :: B\n"
        "REAL*8, intent(out), dimension(1:1, 1:3) :: A\n"
        "C = z*(x + y)\n"
        "A(1, 1) = 1\n"
        "A(1, 2) = 2\n"
        "A(1, 3) = x\n"
        "B = 2*x\n"
        "end subroutine\n"
    )
    assert source == expected


def test_fcode_matrixsymbol_slice():
    A = MatrixSymbol('A', 2, 3)
    B = MatrixSymbol('B', 1, 3)
    C = MatrixSymbol('C', 1, 3)
    D = MatrixSymbol('D', 2, 1)
    name_expr = ("test", [Equality(B, A[0, :]),
                          Equality(C, A[1, :]),
                          Equality(D, A[:, 2])])
    result = codegen(name_expr, "f95", "test", header=False, empty=False)
    source = result[0][1]
    expected = (
        "subroutine test(A, B, C, D)\n"
        "implicit none\n"
        "REAL*8, intent(in), dimension(1:2, 1:3) :: A\n"
        "REAL*8, intent(out), dimension(1:1, 1:3) :: B\n"
        "REAL*8, intent(out), dimension(1:1, 1:3) :: C\n"
        "REAL*8, intent(out), dimension(1:2, 1:1) :: D\n"
        "B(1, 1) = A(1, 1)\n"
        "B(1, 2) = A(1, 2)\n"
        "B(1, 3) = A(1, 3)\n"
        "C(1, 1) = A(2, 1)\n"
        "C(1, 2) = A(2, 2)\n"
        "C(1, 3) = A(2, 3)\n"
        "D(1, 1) = A(1, 3)\n"
        "D(2, 1) = A(2, 3)\n"
        "end subroutine\n"
    )
    assert source == expected


def test_fcode_matrixsymbol_slice_autoname():
    # see issue #8093
    A = MatrixSymbol('A', 2, 3)
    name_expr = ("test", A[:, 1])
    result = codegen(name_expr, "f95", "test", header=False, empty=False)
    source = result[0][1]
    expected = (
        "subroutine test(A, out_%(hash)s)\n"
        "implicit none\n"
        "REAL*8, intent(in), dimension(1:2, 1:3) :: A\n"
        "REAL*8, intent(out), dimension(1:2, 1:1) :: out_%(hash)s\n"
        "out_%(hash)s(1, 1) = A(1, 2)\n"
        "out_%(hash)s(2, 1) = A(2, 2)\n"
        "end subroutine\n"
    )
    # look for the magic number
    a = source.splitlines()[3]
    b = a.split('_')
    out = b[1]
    expected = expected % {'hash': out}
    assert source == expected


def test_global_vars():
    x, y, z, t = symbols("x y z t")
    result = codegen(('f', x*y), "F95", header=False, empty=False,
                     global_vars=(y,))
    source = result[0][1]
    expected = (
        "REAL*8 function f(x)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "f = x*y\n"
        "end function\n"
        )
    assert source == expected

    expected = (
        '#include "f.h"\n'
        '#include <math.h>\n'
        'double f(double x, double y) {\n'
        '   double f_result;\n'
        '   f_result = x*y + z;\n'
        '   return f_result;\n'
        '}\n'
    )
    result = codegen(('f', x*y+z), "C", header=False, empty=False,
                     global_vars=(z, t))
    source = result[0][1]
    assert source == expected

def test_custom_codegen():
    from sympy.printing.c import C99CodePrinter
    from sympy.functions.elementary.exponential import exp

    printer = C99CodePrinter(settings={'user_functions': {'exp': 'fastexp'}})

    x, y = symbols('x y')
    expr = exp(x + y)

    # replace math.h with a different header
    gen = C99CodeGen(printer=printer,
                     preprocessor_statements=['#include "fastexp.h"'])

    expected = (
        '#include "expr.h"\n'
        '#include "fastexp.h"\n'
        'double expr(double x, double y) {\n'
        '   double expr_result;\n'
        '   expr_result = fastexp(x + y);\n'
        '   return expr_result;\n'
        '}\n'
    )

    result = codegen(('expr', expr), header=False, empty=False, code_gen=gen)
    source = result[0][1]
    assert source == expected

    # use both math.h and an external header
    gen = C99CodeGen(printer=printer)
    gen.preprocessor_statements.append('#include "fastexp.h"')

    expected = (
        '#include "expr.h"\n'
        '#include <math.h>\n'
        '#include "fastexp.h"\n'
        'double expr(double x, double y) {\n'
        '   double expr_result;\n'
        '   expr_result = fastexp(x + y);\n'
        '   return expr_result;\n'
        '}\n'
    )

    result = codegen(('expr', expr), header=False, empty=False, code_gen=gen)
    source = result[0][1]
    assert source == expected

def test_c_with_printer():
    # issue 13586
    from sympy.printing.c import C99CodePrinter
    class CustomPrinter(C99CodePrinter):
        def _print_Pow(self, expr):
            return "fastpow({}, {})".format(self._print(expr.base),
                                            self._print(expr.exp))

    x = symbols('x')
    expr = x**3
    expected =[
        ("file.c",
        "#include \"file.h\"\n"
        "#include <math.h>\n"
        "double test(double x) {\n"
        "   double test_result;\n"
        "   test_result = fastpow(x, 3);\n"
        "   return test_result;\n"
        "}\n"),
        ("file.h",
        "#ifndef PROJECT__FILE__H\n"
        "#define PROJECT__FILE__H\n"
        "double test(double x);\n"
        "#endif\n")
    ]
    result = codegen(("test", expr), "C","file", header=False, empty=False, printer = CustomPrinter())
    assert result == expected


def test_fcode_complex():
    import sympy.utilities.codegen
    sympy.utilities.codegen.COMPLEX_ALLOWED = True
    x = Symbol('x', real=True)
    y = Symbol('y',real=True)
    result = codegen(('test',x+y), 'f95', 'test', header=False, empty=False)
    source = (result[0][1])
    expected = (
        "REAL*8 function test(x, y)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "REAL*8, intent(in) :: y\n"
        "test = x + y\n"
        "end function\n")
    assert source == expected
    x = Symbol('x')
    y = Symbol('y',real=True)
    result = codegen(('test',x+y), 'f95', 'test', header=False, empty=False)
    source = (result[0][1])
    expected = (
        "COMPLEX*16 function test(x, y)\n"
        "implicit none\n"
        "COMPLEX*16, intent(in) :: x\n"
        "REAL*8, intent(in) :: y\n"
        "test = x + y\n"
        "end function\n"
        )
    assert source==expected
    sympy.utilities.codegen.COMPLEX_ALLOWED = False