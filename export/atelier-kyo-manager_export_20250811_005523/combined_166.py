
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\test_groupby.py ===
from datetime import datetime
import decimal
from decimal import Decimal
import re

import numpy as np
import pytest

from pandas.errors import (
    PerformanceWarning,
    SpecificationError,
)
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Grouper,
    Index,
    Interval,
    MultiIndex,
    RangeIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    to_datetime,
)
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com

pytestmark = pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")


def test_repr():
    # GH18203
    result = repr(Grouper(key="A", level="B"))
    expected = "Grouper(key='A', level='B', axis=0, sort=False, dropna=True)"
    assert result == expected


def test_groupby_std_datetimelike(warn_copy_on_write):
    # GH#48481
    tdi = pd.timedelta_range("1 Day", periods=10000)
    ser = Series(tdi)
    ser[::5] *= 2  # get different std for different groups

    df = ser.to_frame("A").copy()

    df["B"] = ser + Timestamp(0)
    df["C"] = ser + Timestamp(0, tz="UTC")
    df.iloc[-1] = pd.NaT  # last group includes NaTs

    gb = df.groupby(list(range(5)) * 2000)

    result = gb.std()

    # Note: this does not _exactly_ match what we would get if we did
    # [gb.get_group(i).std() for i in gb.groups]
    #  but it _does_ match the floating point error we get doing the
    #  same operation on int64 data xref GH#51332
    td1 = Timedelta("2887 days 11:21:02.326710176")
    td4 = Timedelta("2886 days 00:42:34.664668096")
    exp_ser = Series([td1 * 2, td1, td1, td1, td4], index=np.arange(5))
    expected = DataFrame({"A": exp_ser, "B": exp_ser, "C": exp_ser})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", ["int64", "int32", "float64", "float32"])
def test_basic_aggregations(dtype):
    data = Series(np.arange(9) // 3, index=np.arange(9), dtype=dtype)

    index = np.arange(9)
    np.random.default_rng(2).shuffle(index)
    data = data.reindex(index)

    grouped = data.groupby(lambda x: x // 3, group_keys=False)

    for k, v in grouped:
        assert len(v) == 3

    msg = "using SeriesGroupBy.mean"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        agged = grouped.aggregate(np.mean)
    assert agged[1] == 1

    msg = "using SeriesGroupBy.mean"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = grouped.agg(np.mean)
    tm.assert_series_equal(agged, expected)  # shorthand
    tm.assert_series_equal(agged, grouped.mean())
    result = grouped.sum()
    msg = "using SeriesGroupBy.sum"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = grouped.agg(np.sum)
    tm.assert_series_equal(result, expected)

    expected = grouped.apply(lambda x: x * x.sum())
    transformed = grouped.transform(lambda x: x * x.sum())
    assert transformed[7] == 12
    tm.assert_series_equal(transformed, expected)

    value_grouped = data.groupby(data)
    msg = "using SeriesGroupBy.mean"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = value_grouped.aggregate(np.mean)
    tm.assert_series_equal(result, agged, check_index_type=False)

    # complex agg
    msg = "using SeriesGroupBy.[mean|std]"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        agged = grouped.aggregate([np.mean, np.std])

    msg = r"nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        grouped.aggregate({"one": np.mean, "two": np.std})

    group_constants = {0: 10, 1: 20, 2: 30}
    msg = (
        "Pinning the groupby key to each group in SeriesGroupBy.agg is deprecated, "
        "and cases that relied on it will raise in a future version"
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        # GH#41090
        agged = grouped.agg(lambda x: group_constants[x.name] + x.mean())
    assert agged[1] == 21

    # corner cases
    msg = "Must produce aggregated value"
    # exception raised is type Exception
    with pytest.raises(Exception, match=msg):
        grouped.aggregate(lambda x: x * 2)


def test_groupby_nonobject_dtype(multiindex_dataframe_random_data):
    key = multiindex_dataframe_random_data.index.codes[0]
    grouped = multiindex_dataframe_random_data.groupby(key)
    result = grouped.sum()

    expected = multiindex_dataframe_random_data.groupby(key.astype("O")).sum()
    assert result.index.dtype == np.int8
    assert expected.index.dtype == np.int64
    tm.assert_frame_equal(result, expected, check_index_type=False)


def test_groupby_nonobject_dtype_mixed():
    # GH 3911, mixed frame non-conversion
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
            "C": np.random.default_rng(2).standard_normal(8),
            "D": np.array(np.random.default_rng(2).standard_normal(8), dtype="float32"),
        }
    )
    df["value"] = range(len(df))

    def max_value(group):
        return group.loc[group["value"].idxmax()]

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        applied = df.groupby("A").apply(max_value)
    result = applied.dtypes
    expected = df.dtypes
    tm.assert_series_equal(result, expected)


def test_inconsistent_return_type():
    # GH5592
    # inconsistent return type
    df = DataFrame(
        {
            "A": ["Tiger", "Tiger", "Tiger", "Lamb", "Lamb", "Pony", "Pony"],
            "B": Series(np.arange(7), dtype="int64"),
            "C": date_range("20130101", periods=7),
        }
    )

    def f_0(grp):
        return grp.iloc[0]

    expected = df.groupby("A").first()[["B"]]
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("A").apply(f_0)[["B"]]
    tm.assert_frame_equal(result, expected)

    def f_1(grp):
        if grp.name == "Tiger":
            return None
        return grp.iloc[0]

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("A").apply(f_1)[["B"]]
    e = expected.copy()
    e.loc["Tiger"] = np.nan
    tm.assert_frame_equal(result, e)

    def f_2(grp):
        if grp.name == "Pony":
            return None
        return grp.iloc[0]

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("A").apply(f_2)[["B"]]
    e = expected.copy()
    e.loc["Pony"] = np.nan
    tm.assert_frame_equal(result, e)

    # 5592 revisited, with datetimes
    def f_3(grp):
        if grp.name == "Pony":
            return None
        return grp.iloc[0]

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("A").apply(f_3)[["C"]]
    e = df.groupby("A").first()[["C"]]
    e.loc["Pony"] = pd.NaT
    tm.assert_frame_equal(result, e)

    # scalar outputs
    def f_4(grp):
        if grp.name == "Pony":
            return None
        return grp.iloc[0].loc["C"]

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("A").apply(f_4)
    e = df.groupby("A").first()["C"].copy()
    e.loc["Pony"] = np.nan
    e.name = None
    tm.assert_series_equal(result, e)


def test_pass_args_kwargs(ts, tsframe):
    def f(x, q=None, axis=0):
        return np.percentile(x, q, axis=axis)

    g = lambda x: np.percentile(x, 80, axis=0)

    # Series
    ts_grouped = ts.groupby(lambda x: x.month)
    agg_result = ts_grouped.agg(np.percentile, 80, axis=0)
    apply_result = ts_grouped.apply(np.percentile, 80, axis=0)
    trans_result = ts_grouped.transform(np.percentile, 80, axis=0)

    agg_expected = ts_grouped.quantile(0.8)
    trans_expected = ts_grouped.transform(g)

    tm.assert_series_equal(apply_result, agg_expected)
    tm.assert_series_equal(agg_result, agg_expected)
    tm.assert_series_equal(trans_result, trans_expected)

    agg_result = ts_grouped.agg(f, q=80)
    apply_result = ts_grouped.apply(f, q=80)
    trans_result = ts_grouped.transform(f, q=80)
    tm.assert_series_equal(agg_result, agg_expected)
    tm.assert_series_equal(apply_result, agg_expected)
    tm.assert_series_equal(trans_result, trans_expected)

    # DataFrame
    for as_index in [True, False]:
        df_grouped = tsframe.groupby(lambda x: x.month, as_index=as_index)
        warn = None if as_index else FutureWarning
        msg = "A grouping .* was excluded from the result"
        with tm.assert_produces_warning(warn, match=msg):
            agg_result = df_grouped.agg(np.percentile, 80, axis=0)
        with tm.assert_produces_warning(warn, match=msg):
            apply_result = df_grouped.apply(DataFrame.quantile, 0.8)
        with tm.assert_produces_warning(warn, match=msg):
            expected = df_grouped.quantile(0.8)
        tm.assert_frame_equal(apply_result, expected, check_names=False)
        tm.assert_frame_equal(agg_result, expected)

        apply_result = df_grouped.apply(DataFrame.quantile, [0.4, 0.8])
        with tm.assert_produces_warning(warn, match=msg):
            expected_seq = df_grouped.quantile([0.4, 0.8])
        tm.assert_frame_equal(apply_result, expected_seq, check_names=False)

        with tm.assert_produces_warning(warn, match=msg):
            agg_result = df_grouped.agg(f, q=80)
        with tm.assert_produces_warning(warn, match=msg):
            apply_result = df_grouped.apply(DataFrame.quantile, q=0.8)
        tm.assert_frame_equal(agg_result, expected)
        tm.assert_frame_equal(apply_result, expected, check_names=False)


@pytest.mark.parametrize("as_index", [True, False])
def test_pass_args_kwargs_duplicate_columns(tsframe, as_index):
    # go through _aggregate_frame with self.axis == 0 and duplicate columns
    tsframe.columns = ["A", "B", "A", "C"]
    gb = tsframe.groupby(lambda x: x.month, as_index=as_index)

    warn = None if as_index else FutureWarning
    msg = "A grouping .* was excluded from the result"
    with tm.assert_produces_warning(warn, match=msg):
        res = gb.agg(np.percentile, 80, axis=0)

    ex_data = {
        1: tsframe[tsframe.index.month == 1].quantile(0.8),
        2: tsframe[tsframe.index.month == 2].quantile(0.8),
    }
    expected = DataFrame(ex_data).T
    if not as_index:
        # TODO: try to get this more consistent?
        expected.index = Index(range(2))

    tm.assert_frame_equal(res, expected)


def test_len():
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    grouped = df.groupby([lambda x: x.year, lambda x: x.month, lambda x: x.day])
    assert len(grouped) == len(df)

    grouped = df.groupby([lambda x: x.year, lambda x: x.month])
    expected = len({(x.year, x.month) for x in df.index})
    assert len(grouped) == expected


def test_len_nan_group():
    # issue 11016
    df = DataFrame({"a": [np.nan] * 3, "b": [1, 2, 3]})
    assert len(df.groupby("a")) == 0
    assert len(df.groupby("b")) == 3
    assert len(df.groupby(["a", "b"])) == 3


def test_basic_regression():
    # regression
    result = Series([1.0 * x for x in list(range(1, 10)) * 10])

    data = np.random.default_rng(2).random(1100) * 10.0
    groupings = Series(data)

    grouped = result.groupby(groupings)
    grouped.mean()


@pytest.mark.parametrize(
    "dtype", ["float64", "float32", "int64", "int32", "int16", "int8"]
)
def test_with_na_groups(dtype):
    index = Index(np.arange(10))
    values = Series(np.ones(10), index, dtype=dtype)
    labels = Series(
        [np.nan, "foo", "bar", "bar", np.nan, np.nan, "bar", "bar", np.nan, "foo"],
        index=index,
    )

    # this SHOULD be an int
    grouped = values.groupby(labels)
    agged = grouped.agg(len)
    expected = Series([4, 2], index=["bar", "foo"])

    tm.assert_series_equal(agged, expected, check_dtype=False)

    # assert issubclass(agged.dtype.type, np.integer)

    # explicitly return a float from my function
    def f(x):
        return float(len(x))

    agged = grouped.agg(f)
    expected = Series([4.0, 2.0], index=["bar", "foo"])

    tm.assert_series_equal(agged, expected)


def test_indices_concatenation_order():
    # GH 2808

    def f1(x):
        y = x[(x.b % 2) == 1] ** 2
        if y.empty:
            multiindex = MultiIndex(levels=[[]] * 2, codes=[[]] * 2, names=["b", "c"])
            res = DataFrame(columns=["a"], index=multiindex)
            return res
        else:
            y = y.set_index(["b", "c"])
            return y

    def f2(x):
        y = x[(x.b % 2) == 1] ** 2
        if y.empty:
            return DataFrame()
        else:
            y = y.set_index(["b", "c"])
            return y

    def f3(x):
        y = x[(x.b % 2) == 1] ** 2
        if y.empty:
            multiindex = MultiIndex(
                levels=[[]] * 2, codes=[[]] * 2, names=["foo", "bar"]
            )
            res = DataFrame(columns=["a", "b"], index=multiindex)
            return res
        else:
            return y

    df = DataFrame({"a": [1, 2, 2, 2], "b": range(4), "c": range(5, 9)})

    df2 = DataFrame({"a": [3, 2, 2, 2], "b": range(4), "c": range(5, 9)})

    depr_msg = "The behavior of array concatenation with empty entries is deprecated"

    # correct result
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result1 = df.groupby("a").apply(f1)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result2 = df2.groupby("a").apply(f1)
    tm.assert_frame_equal(result1, result2)

    # should fail (not the same number of levels)
    msg = "Cannot concat indices that do not have the same number of levels"
    with pytest.raises(AssertionError, match=msg):
        df.groupby("a").apply(f2)
    with pytest.raises(AssertionError, match=msg):
        df2.groupby("a").apply(f2)

    # should fail (incorrect shape)
    with pytest.raises(AssertionError, match=msg):
        df.groupby("a").apply(f3)
    with pytest.raises(AssertionError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            df2.groupby("a").apply(f3)


def test_attr_wrapper(ts):
    grouped = ts.groupby(lambda x: x.weekday())

    result = grouped.std()
    expected = grouped.agg(lambda x: np.std(x, ddof=1))
    tm.assert_series_equal(result, expected)

    # this is pretty cool
    result = grouped.describe()
    expected = {name: gp.describe() for name, gp in grouped}
    expected = DataFrame(expected).T
    tm.assert_frame_equal(result, expected)

    # get attribute
    result = grouped.dtype
    expected = grouped.agg(lambda x: x.dtype)
    tm.assert_series_equal(result, expected)

    # make sure raises error
    msg = "'SeriesGroupBy' object has no attribute 'foo'"
    with pytest.raises(AttributeError, match=msg):
        getattr(grouped, "foo")


def test_frame_groupby(tsframe):
    grouped = tsframe.groupby(lambda x: x.weekday())

    # aggregate
    aggregated = grouped.aggregate("mean")
    assert len(aggregated) == 5
    assert len(aggregated.columns) == 4

    # by string
    tscopy = tsframe.copy()
    tscopy["weekday"] = [x.weekday() for x in tscopy.index]
    stragged = tscopy.groupby("weekday").aggregate("mean")
    tm.assert_frame_equal(stragged, aggregated, check_names=False)

    # transform
    grouped = tsframe.head(30).groupby(lambda x: x.weekday())
    transformed = grouped.transform(lambda x: x - x.mean())
    assert len(transformed) == 30
    assert len(transformed.columns) == 4

    # transform propagate
    transformed = grouped.transform(lambda x: x.mean())
    for name, group in grouped:
        mean = group.mean()
        for idx in group.index:
            tm.assert_series_equal(transformed.xs(idx), mean, check_names=False)

    # iterate
    for weekday, group in grouped:
        assert group.index[0].weekday() == weekday

    # groups / group_indices
    groups = grouped.groups
    indices = grouped.indices

    for k, v in groups.items():
        samething = tsframe.index.take(indices[k])
        assert (samething == v).all()


def test_frame_groupby_columns(tsframe):
    mapping = {"A": 0, "B": 0, "C": 1, "D": 1}
    msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        grouped = tsframe.groupby(mapping, axis=1)

    # aggregate
    aggregated = grouped.aggregate("mean")
    assert len(aggregated) == len(tsframe)
    assert len(aggregated.columns) == 2

    # transform
    tf = lambda x: x - x.mean()
    msg = "The 'axis' keyword in DataFrame.groupby is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        groupedT = tsframe.T.groupby(mapping, axis=0)
    tm.assert_frame_equal(groupedT.transform(tf).T, grouped.transform(tf))

    # iterate
    for k, v in grouped:
        assert len(v.columns) == 2


def test_frame_set_name_single(df):
    grouped = df.groupby("A")

    result = grouped.mean(numeric_only=True)
    assert result.index.name == "A"

    result = df.groupby("A", as_index=False).mean(numeric_only=True)
    assert result.index.name != "A"

    result = grouped[["C", "D"]].agg("mean")
    assert result.index.name == "A"

    result = grouped.agg({"C": "mean", "D": "std"})
    assert result.index.name == "A"

    result = grouped["C"].mean()
    assert result.index.name == "A"
    result = grouped["C"].agg("mean")
    assert result.index.name == "A"
    result = grouped["C"].agg(["mean", "std"])
    assert result.index.name == "A"

    msg = r"nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        grouped["C"].agg({"foo": "mean", "bar": "std"})


def test_multi_func(df):
    col1 = df["A"]
    col2 = df["B"]

    grouped = df.groupby([col1.get, col2.get])
    agged = grouped.mean(numeric_only=True)
    expected = df.groupby(["A", "B"]).mean()

    # TODO groupby get drops names
    tm.assert_frame_equal(
        agged.loc[:, ["C", "D"]], expected.loc[:, ["C", "D"]], check_names=False
    )

    # some "groups" with no data
    df = DataFrame(
        {
            "v1": np.random.default_rng(2).standard_normal(6),
            "v2": np.random.default_rng(2).standard_normal(6),
            "k1": np.array(["b", "b", "b", "a", "a", "a"]),
            "k2": np.array(["1", "1", "1", "2", "2", "2"]),
        },
        index=["one", "two", "three", "four", "five", "six"],
    )
    # only verify that it works for now
    grouped = df.groupby(["k1", "k2"])
    grouped.agg("sum")


def test_multi_key_multiple_functions(df):
    grouped = df.groupby(["A", "B"])["C"]

    agged = grouped.agg(["mean", "std"])
    expected = DataFrame({"mean": grouped.agg("mean"), "std": grouped.agg("std")})
    tm.assert_frame_equal(agged, expected)


def test_frame_multi_key_function_list():
    data = DataFrame(
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
            "D": np.random.default_rng(2).standard_normal(11),
            "E": np.random.default_rng(2).standard_normal(11),
            "F": np.random.default_rng(2).standard_normal(11),
        }
    )

    grouped = data.groupby(["A", "B"])
    funcs = ["mean", "std"]
    agged = grouped.agg(funcs)
    expected = pd.concat(
        [grouped["D"].agg(funcs), grouped["E"].agg(funcs), grouped["F"].agg(funcs)],
        keys=["D", "E", "F"],
        axis=1,
    )
    assert isinstance(agged.index, MultiIndex)
    assert isinstance(expected.index, MultiIndex)
    tm.assert_frame_equal(agged, expected)


def test_frame_multi_key_function_list_partial_failure(using_infer_string):
    data = DataFrame(
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

    grouped = data.groupby(["A", "B"])
    funcs = ["mean", "std"]
    msg = re.escape("agg function failed [how->mean,dtype->")
    if using_infer_string:
        msg = "dtype 'str' does not support operation 'mean'"
    with pytest.raises(TypeError, match=msg):
        grouped.agg(funcs)


@pytest.mark.parametrize("op", [lambda x: x.sum(), lambda x: x.mean()])
def test_groupby_multiple_columns(df, op):
    data = df
    grouped = data.groupby(["A", "B"])

    result1 = op(grouped)

    keys = []
    values = []
    for n1, gp1 in data.groupby("A"):
        for n2, gp2 in gp1.groupby("B"):
            keys.append((n1, n2))
            values.append(op(gp2.loc[:, ["C", "D"]]))

    mi = MultiIndex.from_tuples(keys, names=["A", "B"])
    expected = pd.concat(values, axis=1).T
    expected.index = mi

    # a little bit crude
    for col in ["C", "D"]:
        result_col = op(grouped[col])
        pivoted = result1[col]
        exp = expected[col]
        tm.assert_series_equal(result_col, exp)
        tm.assert_series_equal(pivoted, exp)

    # test single series works the same
    result = data["C"].groupby([data["A"], data["B"]]).mean()
    expected = data.groupby(["A", "B"]).mean()["C"]

    tm.assert_series_equal(result, expected)


def test_as_index_select_column():
    # GH 5764
    df = DataFrame([[1, 2], [1, 4], [5, 6]], columns=["A", "B"])
    result = df.groupby("A", as_index=False)["B"].get_group(1)
    expected = Series([2, 4], name="B")
    tm.assert_series_equal(result, expected)

    result = df.groupby("A", as_index=False, group_keys=True)["B"].apply(
        lambda x: x.cumsum()
    )
    expected = Series(
        [2, 6, 6], name="B", index=MultiIndex.from_tuples([(0, 0), (0, 1), (1, 2)])
    )
    tm.assert_series_equal(result, expected)


def test_obj_arg_get_group_deprecated():
    depr_msg = "obj is deprecated"

    df = DataFrame({"a": [1, 1, 2], "b": [3, 4, 5]})
    expected = df.iloc[df.groupby("b").indices.get(4)]
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        result = df.groupby("b").get_group(4, obj=df)
        tm.assert_frame_equal(result, expected)


def test_groupby_as_index_select_column_sum_empty_df():
    # GH 35246
    df = DataFrame(columns=Index(["A", "B", "C"], name="alpha"))
    left = df.groupby(by="A", as_index=False)["B"].sum(numeric_only=False)

    expected = DataFrame(columns=df.columns[:2], index=range(0))
    # GH#50744 - Columns after selection shouldn't retain names
    expected.columns.names = [None]
    tm.assert_frame_equal(left, expected)


def test_groupby_as_index_agg(df):
    grouped = df.groupby("A", as_index=False)

    # single-key

    result = grouped[["C", "D"]].agg("mean")
    expected = grouped.mean(numeric_only=True)
    tm.assert_frame_equal(result, expected)

    result2 = grouped.agg({"C": "mean", "D": "sum"})
    expected2 = grouped.mean(numeric_only=True)
    expected2["D"] = grouped.sum()["D"]
    tm.assert_frame_equal(result2, expected2)

    grouped = df.groupby("A", as_index=True)

    msg = r"nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        grouped["C"].agg({"Q": "sum"})

    # multi-key

    grouped = df.groupby(["A", "B"], as_index=False)

    result = grouped.agg("mean")
    expected = grouped.mean()
    tm.assert_frame_equal(result, expected)

    result2 = grouped.agg({"C": "mean", "D": "sum"})
    expected2 = grouped.mean()
    expected2["D"] = grouped.sum()["D"]
    tm.assert_frame_equal(result2, expected2)

    expected3 = grouped["C"].sum()
    expected3 = DataFrame(expected3).rename(columns={"C": "Q"})
    msg = "Passing a dictionary to SeriesGroupBy.agg is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result3 = grouped["C"].agg({"Q": "sum"})
    tm.assert_frame_equal(result3, expected3)

    # GH7115 & GH8112 & GH8582
    df = DataFrame(
        np.random.default_rng(2).integers(0, 100, (50, 3)),
        columns=["jim", "joe", "jolie"],
    )
    ts = Series(np.random.default_rng(2).integers(5, 10, 50), name="jim")

    gr = df.groupby(ts)
    gr.nth(0)  # invokes set_selection_from_grouper internally

    msg = "The behavior of DataFrame.sum with axis=None is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg, check_stacklevel=False):
        res = gr.apply(sum)
    with tm.assert_produces_warning(FutureWarning, match=msg, check_stacklevel=False):
        alt = df.groupby(ts).apply(sum)
    tm.assert_frame_equal(res, alt)

    for attr in ["mean", "max", "count", "idxmax", "cumsum", "all"]:
        gr = df.groupby(ts, as_index=False)
        left = getattr(gr, attr)()

        gr = df.groupby(ts.values, as_index=True)
        right = getattr(gr, attr)().reset_index(drop=True)

        tm.assert_frame_equal(left, right)


def test_ops_not_as_index(reduction_func):
    # GH 10355, 21090
    # Using as_index=False should not modify grouped column

    if reduction_func in ("corrwith", "nth", "ngroup"):
        pytest.skip(f"GH 5755: Test not applicable for {reduction_func}")

    df = DataFrame(
        np.random.default_rng(2).integers(0, 5, size=(100, 2)), columns=["a", "b"]
    )
    expected = getattr(df.groupby("a"), reduction_func)()
    if reduction_func == "size":
        expected = expected.rename("size")
    expected = expected.reset_index()

    if reduction_func != "size":
        # 32 bit compat -> groupby preserves dtype whereas reset_index casts to int64
        expected["a"] = expected["a"].astype(df["a"].dtype)

    g = df.groupby("a", as_index=False)

    result = getattr(g, reduction_func)()
    tm.assert_frame_equal(result, expected)

    result = g.agg(reduction_func)
    tm.assert_frame_equal(result, expected)

    result = getattr(g["b"], reduction_func)()
    tm.assert_frame_equal(result, expected)

    result = g["b"].agg(reduction_func)
    tm.assert_frame_equal(result, expected)


def test_as_index_series_return_frame(df):
    grouped = df.groupby("A", as_index=False)
    grouped2 = df.groupby(["A", "B"], as_index=False)

    result = grouped["C"].agg("sum")
    expected = grouped.agg("sum").loc[:, ["A", "C"]]
    assert isinstance(result, DataFrame)
    tm.assert_frame_equal(result, expected)

    result2 = grouped2["C"].agg("sum")
    expected2 = grouped2.agg("sum").loc[:, ["A", "B", "C"]]
    assert isinstance(result2, DataFrame)
    tm.assert_frame_equal(result2, expected2)

    result = grouped["C"].sum()
    expected = grouped.sum().loc[:, ["A", "C"]]
    assert isinstance(result, DataFrame)
    tm.assert_frame_equal(result, expected)

    result2 = grouped2["C"].sum()
    expected2 = grouped2.sum().loc[:, ["A", "B", "C"]]
    assert isinstance(result2, DataFrame)
    tm.assert_frame_equal(result2, expected2)


def test_as_index_series_column_slice_raises(df):
    # GH15072
    grouped = df.groupby("A", as_index=False)
    msg = r"Column\(s\) C already selected"

    with pytest.raises(IndexError, match=msg):
        grouped["C"].__getitem__("D")


def test_groupby_as_index_cython(df):
    data = df

    # single-key
    grouped = data.groupby("A", as_index=False)
    result = grouped.mean(numeric_only=True)
    expected = data.groupby(["A"]).mean(numeric_only=True)
    expected.insert(0, "A", expected.index)
    expected.index = RangeIndex(len(expected))
    tm.assert_frame_equal(result, expected)

    # multi-key
    grouped = data.groupby(["A", "B"], as_index=False)
    result = grouped.mean()
    expected = data.groupby(["A", "B"]).mean()

    arrays = list(zip(*expected.index.values))
    expected.insert(0, "A", arrays[0])
    expected.insert(1, "B", arrays[1])
    expected.index = RangeIndex(len(expected))
    tm.assert_frame_equal(result, expected)


def test_groupby_as_index_series_scalar(df):
    grouped = df.groupby(["A", "B"], as_index=False)

    # GH #421

    result = grouped["C"].agg(len)
    expected = grouped.agg(len).loc[:, ["A", "B", "C"]]
    tm.assert_frame_equal(result, expected)


def test_groupby_as_index_corner(df, ts):
    msg = "as_index=False only valid with DataFrame"
    with pytest.raises(TypeError, match=msg):
        ts.groupby(lambda x: x.weekday(), as_index=False)

    msg = "as_index=False only valid for axis=0"
    depr_msg = "DataFrame.groupby with axis=1 is deprecated"
    with pytest.raises(ValueError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            df.groupby(lambda x: x.lower(), as_index=False, axis=1)


def test_groupby_multiple_key():
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    grouped = df.groupby([lambda x: x.year, lambda x: x.month, lambda x: x.day])
    agged = grouped.sum()
    tm.assert_almost_equal(df.values, agged.values)

    depr_msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        grouped = df.T.groupby(
            [lambda x: x.year, lambda x: x.month, lambda x: x.day], axis=1
        )

    agged = grouped.agg(lambda x: x.sum())
    tm.assert_index_equal(agged.index, df.columns)
    tm.assert_almost_equal(df.T.values, agged.values)

    agged = grouped.agg(lambda x: x.sum())
    tm.assert_almost_equal(df.T.values, agged.values)


def test_groupby_multi_corner(df):
    # test that having an all-NA column doesn't mess you up
    df = df.copy()
    df["bad"] = np.nan
    agged = df.groupby(["A", "B"]).mean()

    expected = df.groupby(["A", "B"]).mean()
    expected["bad"] = np.nan

    tm.assert_frame_equal(agged, expected)


def test_raises_on_nuisance(df, using_infer_string):
    grouped = df.groupby("A")
    msg = re.escape("agg function failed [how->mean,dtype->")
    if using_infer_string:
        msg = "dtype 'str' does not support operation 'mean'"
    with pytest.raises(TypeError, match=msg):
        grouped.agg("mean")
    with pytest.raises(TypeError, match=msg):
        grouped.mean()

    df = df.loc[:, ["A", "C", "D"]]
    df["E"] = datetime.now()
    grouped = df.groupby("A")
    msg = "datetime64 type does not support sum operations"
    with pytest.raises(TypeError, match=msg):
        grouped.agg("sum")
    with pytest.raises(TypeError, match=msg):
        grouped.sum()

    # won't work with axis = 1
    depr_msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        grouped = df.groupby({"A": 0, "C": 0, "D": 1, "E": 1}, axis=1)
    msg = "does not support reduction 'sum'|Cannot perform reduction 'sum'"
    with pytest.raises(TypeError, match=msg):
        grouped.agg(lambda x: x.sum(0, numeric_only=False))


@pytest.mark.parametrize(
    "agg_function",
    ["max", "min"],
)
def test_keep_nuisance_agg(df, agg_function):
    # GH 38815
    grouped = df.groupby("A")
    result = getattr(grouped, agg_function)()
    expected = result.copy()
    expected.loc["bar", "B"] = getattr(df.loc[df["A"] == "bar", "B"], agg_function)()
    expected.loc["foo", "B"] = getattr(df.loc[df["A"] == "foo", "B"], agg_function)()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "agg_function",
    ["sum", "mean", "prod", "std", "var", "sem", "median"],
)
@pytest.mark.parametrize("numeric_only", [True, False])
def test_omit_nuisance_agg(df, agg_function, numeric_only, using_infer_string):
    # GH 38774, GH 38815
    grouped = df.groupby("A")

    no_drop_nuisance = ("var", "std", "sem", "mean", "prod", "median")
    if agg_function in no_drop_nuisance and not numeric_only:
        # Added numeric_only as part of GH#46560; these do not drop nuisance
        # columns when numeric_only is False
        if using_infer_string:
            msg = f"dtype 'str' does not support operation '{agg_function}'"
            klass = TypeError
        elif agg_function in ("std", "sem"):
            klass = ValueError
            msg = "could not convert string to float: 'one'"
        else:
            klass = TypeError
            msg = re.escape(f"agg function failed [how->{agg_function},dtype->")
        with pytest.raises(klass, match=msg):
            getattr(grouped, agg_function)(numeric_only=numeric_only)
    else:
        result = getattr(grouped, agg_function)(numeric_only=numeric_only)
        if not numeric_only and agg_function == "sum":
            # sum is successful on column B
            columns = ["A", "B", "C", "D"]
        else:
            columns = ["A", "C", "D"]
        expected = getattr(df.loc[:, columns].groupby("A"), agg_function)(
            numeric_only=numeric_only
        )
        tm.assert_frame_equal(result, expected)


def test_raise_on_nuisance_python_single(df, using_infer_string):
    # GH 38815
    grouped = df.groupby("A")

    err = ValueError
    msg = "could not convert"
    if using_infer_string:
        err = TypeError
        msg = "dtype 'str' does not support operation 'skew'"
    with pytest.raises(err, match=msg):
        grouped.skew()


def test_raise_on_nuisance_python_multiple(three_group, using_infer_string):
    grouped = three_group.groupby(["A", "B"])
    msg = re.escape("agg function failed [how->mean,dtype->")
    if using_infer_string:
        msg = "dtype 'str' does not support operation 'mean'"
    with pytest.raises(TypeError, match=msg):
        grouped.agg("mean")
    with pytest.raises(TypeError, match=msg):
        grouped.mean()


def test_empty_groups_corner(multiindex_dataframe_random_data):
    # handle empty groups
    df = DataFrame(
        {
            "k1": np.array(["b", "b", "b", "a", "a", "a"]),
            "k2": np.array(["1", "1", "1", "2", "2", "2"]),
            "k3": ["foo", "bar"] * 3,
            "v1": np.random.default_rng(2).standard_normal(6),
            "v2": np.random.default_rng(2).standard_normal(6),
        }
    )

    grouped = df.groupby(["k1", "k2"])
    result = grouped[["v1", "v2"]].agg("mean")
    expected = grouped.mean(numeric_only=True)
    tm.assert_frame_equal(result, expected)

    grouped = multiindex_dataframe_random_data[3:5].groupby(level=0)
    agged = grouped.apply(lambda x: x.mean())
    agged_A = grouped["A"].apply("mean")
    tm.assert_series_equal(agged["A"], agged_A)
    assert agged.index.name == "first"


def test_nonsense_func():
    df = DataFrame([0])
    msg = r"unsupported operand type\(s\) for \+: 'int' and 'str'"
    with pytest.raises(TypeError, match=msg):
        df.groupby(lambda x: x + "foo")


def test_wrap_aggregated_output_multindex(
    multiindex_dataframe_random_data, using_infer_string
):
    df = multiindex_dataframe_random_data.T
    df["baz", "two"] = "peekaboo"

    keys = [np.array([0, 0, 1]), np.array([0, 0, 1])]
    msg = re.escape("agg function failed [how->mean,dtype->")
    if using_infer_string:
        msg = "dtype 'str' does not support operation 'mean'"
    with pytest.raises(TypeError, match=msg):
        df.groupby(keys).agg("mean")
    agged = df.drop(columns=("baz", "two")).groupby(keys).agg("mean")
    assert isinstance(agged.columns, MultiIndex)

    def aggfun(ser):
        if ser.name == ("foo", "one"):
            raise TypeError("Test error message")
        return ser.sum()

    with pytest.raises(TypeError, match="Test error message"):
        df.groupby(keys).aggregate(aggfun)


def test_groupby_level_apply(multiindex_dataframe_random_data):
    result = multiindex_dataframe_random_data.groupby(level=0).count()
    assert result.index.name == "first"
    result = multiindex_dataframe_random_data.groupby(level=1).count()
    assert result.index.name == "second"

    result = multiindex_dataframe_random_data["A"].groupby(level=0).count()
    assert result.index.name == "first"


def test_groupby_level_mapper(multiindex_dataframe_random_data):
    deleveled = multiindex_dataframe_random_data.reset_index()

    mapper0 = {"foo": 0, "bar": 0, "baz": 1, "qux": 1}
    mapper1 = {"one": 0, "two": 0, "three": 1}

    result0 = multiindex_dataframe_random_data.groupby(mapper0, level=0).sum()
    result1 = multiindex_dataframe_random_data.groupby(mapper1, level=1).sum()

    mapped_level0 = np.array(
        [mapper0.get(x) for x in deleveled["first"]], dtype=np.int64
    )
    mapped_level1 = np.array(
        [mapper1.get(x) for x in deleveled["second"]], dtype=np.int64
    )
    expected0 = multiindex_dataframe_random_data.groupby(mapped_level0).sum()
    expected1 = multiindex_dataframe_random_data.groupby(mapped_level1).sum()
    expected0.index.name, expected1.index.name = "first", "second"

    tm.assert_frame_equal(result0, expected0)
    tm.assert_frame_equal(result1, expected1)


def test_groupby_level_nonmulti():
    # GH 1313, GH 13901
    s = Series([1, 2, 3, 10, 4, 5, 20, 6], Index([1, 2, 3, 1, 4, 5, 2, 6], name="foo"))
    expected = Series([11, 22, 3, 4, 5, 6], Index(range(1, 7), name="foo"))

    result = s.groupby(level=0).sum()
    tm.assert_series_equal(result, expected)
    result = s.groupby(level=[0]).sum()
    tm.assert_series_equal(result, expected)
    result = s.groupby(level=-1).sum()
    tm.assert_series_equal(result, expected)
    result = s.groupby(level=[-1]).sum()
    tm.assert_series_equal(result, expected)

    msg = "level > 0 or level < -1 only valid with MultiIndex"
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=1)
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=-2)
    msg = "No group keys passed!"
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=[])
    msg = "multiple levels only valid with MultiIndex"
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=[0, 0])
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=[0, 1])
    msg = "level > 0 or level < -1 only valid with MultiIndex"
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=[1])


def test_groupby_complex():
    # GH 12902
    a = Series(data=np.arange(4) * (1 + 2j), index=[0, 0, 1, 1])
    expected = Series((1 + 2j, 5 + 10j))

    result = a.groupby(level=0).sum()
    tm.assert_series_equal(result, expected)


def test_groupby_complex_mean():
    # GH 26475
    df = DataFrame(
        [
            {"a": 2, "b": 1 + 2j},
            {"a": 1, "b": 1 + 1j},
            {"a": 1, "b": 1 + 2j},
        ]
    )
    result = df.groupby("b").mean()
    expected = DataFrame(
        [[1.0], [1.5]],
        index=Index([(1 + 1j), (1 + 2j)], name="b"),
        columns=Index(["a"]),
    )
    tm.assert_frame_equal(result, expected)


def test_groupby_complex_numbers():
    # GH 17927
    df = DataFrame(
        [
            {"a": 1, "b": 1 + 1j},
            {"a": 1, "b": 1 + 2j},
            {"a": 4, "b": 1},
        ]
    )
    expected = DataFrame(
        np.array([1, 1, 1], dtype=np.int64),
        index=Index([(1 + 1j), (1 + 2j), (1 + 0j)], name="b"),
        columns=Index(["a"]),
    )
    result = df.groupby("b", sort=False).count()
    tm.assert_frame_equal(result, expected)

    # Sorted by the magnitude of the complex numbers
    expected.index = Index([(1 + 0j), (1 + 1j), (1 + 2j)], name="b")
    result = df.groupby("b", sort=True).count()
    tm.assert_frame_equal(result, expected)


def test_groupby_series_indexed_differently():
    s1 = Series(
        [5.0, -9.0, 4.0, 100.0, -5.0, 55.0, 6.7],
        index=Index(["a", "b", "c", "d", "e", "f", "g"]),
    )
    s2 = Series(
        [1.0, 1.0, 4.0, 5.0, 5.0, 7.0], index=Index(["a", "b", "d", "f", "g", "h"])
    )

    grouped = s1.groupby(s2)
    agged = grouped.mean()
    exp = s1.groupby(s2.reindex(s1.index).get).mean()
    tm.assert_series_equal(agged, exp)


def test_groupby_with_hier_columns():
    tuples = list(
        zip(
            *[
                ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
                ["one", "two", "one", "two", "one", "two", "one", "two"],
            ]
        )
    )
    index = MultiIndex.from_tuples(tuples)
    columns = MultiIndex.from_tuples(
        [("A", "cat"), ("B", "dog"), ("B", "cat"), ("A", "dog")]
    )
    df = DataFrame(
        np.random.default_rng(2).standard_normal((8, 4)), index=index, columns=columns
    )

    result = df.groupby(level=0).mean()
    tm.assert_index_equal(result.columns, columns)

    depr_msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        gb = df.groupby(level=0, axis=1)
    result = gb.mean()
    tm.assert_index_equal(result.index, df.index)

    result = df.groupby(level=0).agg("mean")
    tm.assert_index_equal(result.columns, columns)

    result = df.groupby(level=0).apply(lambda x: x.mean())
    tm.assert_index_equal(result.columns, columns)

    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        gb = df.groupby(level=0, axis=1)
    result = gb.agg(lambda x: x.mean(1))
    tm.assert_index_equal(result.columns, Index(["A", "B"]))
    tm.assert_index_equal(result.index, df.index)

    # add a nuisance column
    sorted_columns, _ = columns.sortlevel(0)
    df["A", "foo"] = "bar"
    result = df.groupby(level=0).mean(numeric_only=True)
    tm.assert_index_equal(result.columns, df.columns[:-1])


def test_grouping_ndarray(df):
    grouped = df.groupby(df["A"].values)
    grouped2 = df.groupby(df["A"].rename(None))

    result = grouped.sum()
    expected = grouped2.sum()
    tm.assert_frame_equal(result, expected)


def test_groupby_wrong_multi_labels():
    index = Index([0, 1, 2, 3, 4], name="index")
    data = DataFrame(
        {
            "foo": ["foo1", "foo1", "foo2", "foo1", "foo3"],
            "bar": ["bar1", "bar2", "bar2", "bar1", "bar1"],
            "baz": ["baz1", "baz1", "baz1", "baz2", "baz2"],
            "spam": ["spam2", "spam3", "spam2", "spam1", "spam1"],
            "data": [20, 30, 40, 50, 60],
        },
        index=index,
    )

    grouped = data.groupby(["foo", "bar", "baz", "spam"])

    result = grouped.agg("mean")
    expected = grouped.mean()
    tm.assert_frame_equal(result, expected)


def test_groupby_series_with_name(df):
    result = df.groupby(df["A"]).mean(numeric_only=True)
    result2 = df.groupby(df["A"], as_index=False).mean(numeric_only=True)
    assert result.index.name == "A"
    assert "A" in result2

    result = df.groupby([df["A"], df["B"]]).mean()
    result2 = df.groupby([df["A"], df["B"]], as_index=False).mean()
    assert result.index.names == ("A", "B")
    assert "A" in result2
    assert "B" in result2


def test_seriesgroupby_name_attr(df):
    # GH 6265
    result = df.groupby("A")["C"]
    assert result.count().name == "C"
    assert result.mean().name == "C"

    testFunc = lambda x: np.sum(x) * 2
    assert result.agg(testFunc).name == "C"


def test_consistency_name():
    # GH 12363

    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": np.random.default_rng(2).standard_normal(8) + 1.0,
            "D": np.arange(8),
        }
    )

    expected = df.groupby(["A"]).B.count()
    result = df.B.groupby(df.A).count()
    tm.assert_series_equal(result, expected)


def test_groupby_name_propagation(df):
    # GH 6124
    def summarize(df, name=None):
        return Series({"count": 1, "mean": 2, "omissions": 3}, name=name)

    def summarize_random_name(df):
        # Provide a different name for each Series.  In this case, groupby
        # should not attempt to propagate the Series name since they are
        # inconsistent.
        return Series({"count": 1, "mean": 2, "omissions": 3}, name=df.iloc[0]["A"])

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        metrics = df.groupby("A").apply(summarize)
    assert metrics.columns.name is None
    with tm.assert_produces_warning(FutureWarning, match=msg):
        metrics = df.groupby("A").apply(summarize, "metrics")
    assert metrics.columns.name == "metrics"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        metrics = df.groupby("A").apply(summarize_random_name)
    assert metrics.columns.name is None


def test_groupby_nonstring_columns():
    df = DataFrame([np.arange(10) for x in range(10)])
    grouped = df.groupby(0)
    result = grouped.mean()
    expected = df.groupby(df[0]).mean()
    tm.assert_frame_equal(result, expected)


def test_groupby_mixed_type_columns():
    # GH 13432, unorderable types in py3
    df = DataFrame([[0, 1, 2]], columns=["A", "B", 0])
    expected = DataFrame([[1, 2]], columns=["B", 0], index=Index([0], name="A"))

    result = df.groupby("A").first()
    tm.assert_frame_equal(result, expected)

    result = df.groupby("A").sum()
    tm.assert_frame_equal(result, expected)


def test_cython_grouper_series_bug_noncontig():
    arr = np.empty((100, 100))
    arr.fill(np.nan)
    obj = Series(arr[:, 0])
    inds = np.tile(range(10), 10)

    result = obj.groupby(inds).agg(Series.median)
    assert result.isna().all()


def test_series_grouper_noncontig_index():
    index = Index(["a" * 10] * 100)

    values = Series(np.random.default_rng(2).standard_normal(50), index=index[::2])
    labels = np.random.default_rng(2).integers(0, 5, 50)

    # it works!
    grouped = values.groupby(labels)

    # accessing the index elements causes segfault
    f = lambda x: len(set(map(id, x.index)))
    grouped.agg(f)


def test_convert_objects_leave_decimal_alone():
    s = Series(range(5))
    labels = np.array(["a", "b", "c", "d", "e"], dtype="O")

    def convert_fast(x):
        return Decimal(str(x.mean()))

    def convert_force_pure(x):
        # base will be length 0
        assert len(x.values.base) > 0
        return Decimal(str(x.mean()))

    grouped = s.groupby(labels)

    result = grouped.agg(convert_fast)
    assert result.dtype == np.object_
    assert isinstance(result.iloc[0], Decimal)

    result = grouped.agg(convert_force_pure)
    assert result.dtype == np.object_
    assert isinstance(result.iloc[0], Decimal)


def test_groupby_dtype_inference_empty():
    # GH 6733
    df = DataFrame({"x": [], "range": np.arange(0, dtype="int64")})
    assert df["x"].dtype == np.float64

    result = df.groupby("x").first()
    exp_index = Index([], name="x", dtype=np.float64)
    expected = DataFrame({"range": Series([], index=exp_index, dtype="int64")})
    tm.assert_frame_equal(result, expected, by_blocks=True)


def test_groupby_unit64_float_conversion():
    # GH: 30859 groupby converts unit64 to floats sometimes
    df = DataFrame({"first": [1], "second": [1], "value": [16148277970000000000]})
    result = df.groupby(["first", "second"])["value"].max()
    expected = Series(
        [16148277970000000000],
        MultiIndex.from_product([[1], [1]], names=["first", "second"]),
        name="value",
    )
    tm.assert_series_equal(result, expected)


def test_groupby_list_infer_array_like(df):
    result = df.groupby(list(df["A"])).mean(numeric_only=True)
    expected = df.groupby(df["A"]).mean(numeric_only=True)
    tm.assert_frame_equal(result, expected, check_names=False)

    with pytest.raises(KeyError, match=r"^'foo'$"):
        df.groupby(list(df["A"][:-1]))

    # pathological case of ambiguity
    df = DataFrame(
        {
            "foo": [0, 1],
            "bar": [3, 4],
            "val": np.random.default_rng(2).standard_normal(2),
        }
    )

    result = df.groupby(["foo", "bar"]).mean()
    expected = df.groupby([df["foo"], df["bar"]]).mean()[["val"]]


def test_groupby_keys_same_size_as_index():
    # GH 11185
    freq = "s"
    index = date_range(
        start=Timestamp("2015-09-29T11:34:44-0700"), periods=2, freq=freq
    )
    df = DataFrame([["A", 10], ["B", 15]], columns=["metric", "values"], index=index)
    result = df.groupby([Grouper(level=0, freq=freq), "metric"]).mean()
    expected = df.set_index([df.index, "metric"]).astype(float)

    tm.assert_frame_equal(result, expected)


def test_groupby_one_row():
    # GH 11741
    msg = r"^'Z'$"
    df1 = DataFrame(
        np.random.default_rng(2).standard_normal((1, 4)), columns=list("ABCD")
    )
    with pytest.raises(KeyError, match=msg):
        df1.groupby("Z")
    df2 = DataFrame(
        np.random.default_rng(2).standard_normal((2, 4)), columns=list("ABCD")
    )
    with pytest.raises(KeyError, match=msg):
        df2.groupby("Z")


def test_groupby_nat_exclude():
    # GH 6992
    df = DataFrame(
        {
            "values": np.random.default_rng(2).standard_normal(8),
            "dt": [
                np.nan,
                Timestamp("2013-01-01"),
                np.nan,
                Timestamp("2013-02-01"),
                np.nan,
                Timestamp("2013-02-01"),
                np.nan,
                Timestamp("2013-01-01"),
            ],
            "str": [np.nan, "a", np.nan, "a", np.nan, "a", np.nan, "b"],
        }
    )
    grouped = df.groupby("dt")

    expected = [Index([1, 7]), Index([3, 5])]
    keys = sorted(grouped.groups.keys())
    assert len(keys) == 2
    for k, e in zip(keys, expected):
        # grouped.groups keys are np.datetime64 with system tz
        # not to be affected by tz, only compare values
        tm.assert_index_equal(grouped.groups[k], e)

    # confirm obj is not filtered
    tm.assert_frame_equal(grouped._grouper.groupings[0].obj, df)
    assert grouped.ngroups == 2

    expected = {
        Timestamp("2013-01-01 00:00:00"): np.array([1, 7], dtype=np.intp),
        Timestamp("2013-02-01 00:00:00"): np.array([3, 5], dtype=np.intp),
    }

    for k in grouped.indices:
        tm.assert_numpy_array_equal(grouped.indices[k], expected[k])

    tm.assert_frame_equal(grouped.get_group(Timestamp("2013-01-01")), df.iloc[[1, 7]])
    tm.assert_frame_equal(grouped.get_group(Timestamp("2013-02-01")), df.iloc[[3, 5]])

    with pytest.raises(KeyError, match=r"^NaT$"):
        grouped.get_group(pd.NaT)

    nan_df = DataFrame(
        {"nan": [np.nan, np.nan, np.nan], "nat": [pd.NaT, pd.NaT, pd.NaT]}
    )
    assert nan_df["nan"].dtype == "float64"
    assert nan_df["nat"].dtype == "datetime64[ns]"

    for key in ["nan", "nat"]:
        grouped = nan_df.groupby(key)
        assert grouped.groups == {}
        assert grouped.ngroups == 0
        assert grouped.indices == {}
        with pytest.raises(KeyError, match=r"^nan$"):
            grouped.get_group(np.nan)
        with pytest.raises(KeyError, match=r"^NaT$"):
            grouped.get_group(pd.NaT)


def test_groupby_two_group_keys_all_nan():
    # GH #36842: Grouping over two group keys shouldn't raise an error
    df = DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan], "c": [1, 2]})
    result = df.groupby(["a", "b"]).indices
    assert result == {}


def test_groupby_2d_malformed():
    d = DataFrame(index=range(2))
    d["group"] = ["g1", "g2"]
    d["zeros"] = [0, 0]
    d["ones"] = [1, 1]
    d["label"] = ["l1", "l2"]
    tmp = d.groupby(["group"]).mean(numeric_only=True)
    res_values = np.array([[0.0, 1.0], [0.0, 1.0]])
    tm.assert_index_equal(tmp.columns, Index(["zeros", "ones"], dtype=object))
    tm.assert_numpy_array_equal(tmp.values, res_values)


def test_int32_overflow():
    B = np.concatenate((np.arange(10000), np.arange(10000), np.arange(5000)))
    A = np.arange(25000)
    df = DataFrame(
        {
            "A": A,
            "B": B,
            "C": A,
            "D": B,
            "E": np.random.default_rng(2).standard_normal(25000),
        }
    )

    left = df.groupby(["A", "B", "C", "D"]).sum()
    right = df.groupby(["D", "C", "B", "A"]).sum()
    assert len(left) == len(right)


def test_groupby_sort_multi():
    df = DataFrame(
        {
            "a": ["foo", "bar", "baz"],
            "b": [3, 2, 1],
            "c": [0, 1, 2],
            "d": np.random.default_rng(2).standard_normal(3),
        }
    )

    tups = [tuple(row) for row in df[["a", "b", "c"]].values]
    tups = com.asarray_tuplesafe(tups)
    result = df.groupby(["a", "b", "c"], sort=True).sum()
    tm.assert_numpy_array_equal(result.index.values, tups[[1, 2, 0]])

    tups = [tuple(row) for row in df[["c", "a", "b"]].values]
    tups = com.asarray_tuplesafe(tups)
    result = df.groupby(["c", "a", "b"], sort=True).sum()
    tm.assert_numpy_array_equal(result.index.values, tups)

    tups = [tuple(x) for x in df[["b", "c", "a"]].values]
    tups = com.asarray_tuplesafe(tups)
    result = df.groupby(["b", "c", "a"], sort=True).sum()
    tm.assert_numpy_array_equal(result.index.values, tups[[2, 1, 0]])

    df = DataFrame(
        {
            "a": [0, 1, 2, 0, 1, 2],
            "b": [0, 0, 0, 1, 1, 1],
            "d": np.random.default_rng(2).standard_normal(6),
        }
    )
    grouped = df.groupby(["a", "b"])["d"]
    result = grouped.sum()

    def _check_groupby(df, result, keys, field, f=lambda x: x.sum()):
        tups = [tuple(row) for row in df[keys].values]
        tups = com.asarray_tuplesafe(tups)
        expected = f(df.groupby(tups)[field])
        for k, v in expected.items():
            assert result[k] == v

    _check_groupby(df, result, ["a", "b"], "d")


def test_dont_clobber_name_column():
    df = DataFrame(
        {"key": ["a", "a", "a", "b", "b", "b"], "name": ["foo", "bar", "baz"] * 2}
    )

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("key", group_keys=False).apply(lambda x: x)
    tm.assert_frame_equal(result, df)


def test_skip_group_keys():
    tsf = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )

    grouped = tsf.groupby(lambda x: x.month, group_keys=False)
    result = grouped.apply(lambda x: x.sort_values(by="A")[:3])

    pieces = [group.sort_values(by="A")[:3] for key, group in grouped]

    expected = pd.concat(pieces)
    tm.assert_frame_equal(result, expected)

    grouped = tsf["A"].groupby(lambda x: x.month, group_keys=False)
    result = grouped.apply(lambda x: x.sort_values()[:3])

    pieces = [group.sort_values()[:3] for key, group in grouped]

    expected = pd.concat(pieces)
    tm.assert_series_equal(result, expected)


def test_no_nonsense_name(float_frame):
    # GH #995
    s = float_frame["C"].copy()
    s.name = None

    result = s.groupby(float_frame["A"]).agg("sum")
    assert result.name is None


def test_multifunc_sum_bug():
    # GH #1065
    x = DataFrame(np.arange(9).reshape(3, 3))
    x["test"] = 0
    x["fl"] = [1.3, 1.5, 1.6]

    grouped = x.groupby("test")
    result = grouped.agg({"fl": "sum", 2: "size"})
    assert result["fl"].dtype == np.float64


def test_handle_dict_return_value(df):
    def f(group):
        return {"max": group.max(), "min": group.min()}

    def g(group):
        return Series({"max": group.max(), "min": group.min()})

    result = df.groupby("A")["C"].apply(f)
    expected = df.groupby("A")["C"].apply(g)

    assert isinstance(result, Series)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("grouper", ["A", ["A", "B"]])
def test_set_group_name(df, grouper):
    def f(group):
        assert group.name is not None
        return group

    def freduce(group):
        assert group.name is not None
        return group.sum()

    def freducex(x):
        return freduce(x)

    grouped = df.groupby(grouper, group_keys=False)

    # make sure all these work
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        grouped.apply(f)
    grouped.aggregate(freduce)
    grouped.aggregate({"C": freduce, "D": freduce})
    grouped.transform(f)

    grouped["C"].apply(f)
    grouped["C"].aggregate(freduce)
    grouped["C"].aggregate([freduce, freducex])
    grouped["C"].transform(f)


def test_group_name_available_in_inference_pass():
    # gh-15062
    df = DataFrame({"a": [0, 0, 1, 1, 2, 2], "b": np.arange(6)})

    names = []

    def f(group):
        names.append(group.name)
        return group.copy()

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.groupby("a", sort=False, group_keys=False).apply(f)

    expected_names = [0, 1, 2]
    assert names == expected_names


def test_no_dummy_key_names(df):
    # see gh-1291
    result = df.groupby(df["A"].values).sum()
    assert result.index.name is None

    result2 = df.groupby([df["A"].values, df["B"].values]).sum()
    assert result2.index.names == (None, None)


def test_groupby_sort_multiindex_series():
    # series multiindex groupby sort argument was not being passed through
    # _compress_group_index
    # GH 9444
    index = MultiIndex(
        levels=[[1, 2], [1, 2]],
        codes=[[0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 0, 0]],
        names=["a", "b"],
    )
    mseries = Series([0, 1, 2, 3, 4, 5], index=index)
    index = MultiIndex(
        levels=[[1, 2], [1, 2]], codes=[[0, 0, 1], [1, 0, 0]], names=["a", "b"]
    )
    mseries_result = Series([0, 2, 4], index=index)

    result = mseries.groupby(level=["a", "b"], sort=False).first()
    tm.assert_series_equal(result, mseries_result)
    result = mseries.groupby(level=["a", "b"], sort=True).first()
    tm.assert_series_equal(result, mseries_result.sort_index())


def test_groupby_reindex_inside_function():
    periods = 1000
    ind = date_range(start="2012/1/1", freq="5min", periods=periods)
    df = DataFrame({"high": np.arange(periods), "low": np.arange(periods)}, index=ind)

    def agg_before(func, fix=False):
        """
        Run an aggregate func on the subset of data.
        """

        def _func(data):
            d = data.loc[data.index.map(lambda x: x.hour < 11)].dropna()
            if fix:
                data[data.index[0]]
            if len(d) == 0:
                return None
            return func(d)

        return _func

    grouped = df.groupby(lambda x: datetime(x.year, x.month, x.day))
    closure_bad = grouped.agg({"high": agg_before(np.max)})
    closure_good = grouped.agg({"high": agg_before(np.max, True)})

    tm.assert_frame_equal(closure_bad, closure_good)


def test_groupby_multiindex_missing_pair():
    # GH9049
    df = DataFrame(
        {
            "group1": ["a", "a", "a", "b"],
            "group2": ["c", "c", "d", "c"],
            "value": [1, 1, 1, 5],
        }
    )
    df = df.set_index(["group1", "group2"])
    df_grouped = df.groupby(level=["group1", "group2"], sort=True)

    res = df_grouped.agg("sum")
    idx = MultiIndex.from_tuples(
        [("a", "c"), ("a", "d"), ("b", "c")], names=["group1", "group2"]
    )
    exp = DataFrame([[2], [1], [5]], index=idx, columns=["value"])

    tm.assert_frame_equal(res, exp)


def test_groupby_multiindex_not_lexsorted():
    # GH 11640

    # define the lexsorted version
    lexsorted_mi = MultiIndex.from_tuples(
        [("a", ""), ("b1", "c1"), ("b2", "c2")], names=["b", "c"]
    )
    lexsorted_df = DataFrame([[1, 3, 4]], columns=lexsorted_mi)
    assert lexsorted_df.columns._is_lexsorted()

    # define the non-lexsorted version
    not_lexsorted_df = DataFrame(
        columns=["a", "b", "c", "d"], data=[[1, "b1", "c1", 3], [1, "b2", "c2", 4]]
    )
    not_lexsorted_df = not_lexsorted_df.pivot_table(
        index="a", columns=["b", "c"], values="d"
    )
    not_lexsorted_df = not_lexsorted_df.reset_index()
    assert not not_lexsorted_df.columns._is_lexsorted()

    expected = lexsorted_df.groupby("a").mean()
    with tm.assert_produces_warning(PerformanceWarning):
        result = not_lexsorted_df.groupby("a").mean()
    tm.assert_frame_equal(expected, result)

    # a transforming function should work regardless of sort
    # GH 14776
    df = DataFrame(
        {"x": ["a", "a", "b", "a"], "y": [1, 1, 2, 2], "z": [1, 2, 3, 4]}
    ).set_index(["x", "y"])
    assert not df.index._is_lexsorted()

    for level in [0, 1, [0, 1]]:
        for sort in [False, True]:
            result = df.groupby(level=level, sort=sort, group_keys=False).apply(
                DataFrame.drop_duplicates
            )
            expected = df
            tm.assert_frame_equal(expected, result)

            result = (
                df.sort_index()
                .groupby(level=level, sort=sort, group_keys=False)
                .apply(DataFrame.drop_duplicates)
            )
            expected = df.sort_index()
            tm.assert_frame_equal(expected, result)


def test_index_label_overlaps_location():
    # checking we don't have any label/location confusion in the
    # wake of GH5375
    df = DataFrame(list("ABCDE"), index=[2, 0, 2, 1, 1])
    g = df.groupby(list("ababb"))
    actual = g.filter(lambda x: len(x) > 2)
    expected = df.iloc[[1, 3, 4]]
    tm.assert_frame_equal(actual, expected)

    ser = df[0]
    g = ser.groupby(list("ababb"))
    actual = g.filter(lambda x: len(x) > 2)
    expected = ser.take([1, 3, 4])
    tm.assert_series_equal(actual, expected)

    #  and again, with a generic Index of floats
    df.index = df.index.astype(float)
    g = df.groupby(list("ababb"))
    actual = g.filter(lambda x: len(x) > 2)
    expected = df.iloc[[1, 3, 4]]
    tm.assert_frame_equal(actual, expected)

    ser = df[0]
    g = ser.groupby(list("ababb"))
    actual = g.filter(lambda x: len(x) > 2)
    expected = ser.take([1, 3, 4])
    tm.assert_series_equal(actual, expected)


def test_transform_doesnt_clobber_ints():
    # GH 7972
    n = 6
    x = np.arange(n)
    df = DataFrame({"a": x // 2, "b": 2.0 * x, "c": 3.0 * x})
    df2 = DataFrame({"a": x // 2 * 1.0, "b": 2.0 * x, "c": 3.0 * x})

    gb = df.groupby("a")
    result = gb.transform("mean")

    gb2 = df2.groupby("a")
    expected = gb2.transform("mean")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "sort_column",
    ["ints", "floats", "strings", ["ints", "floats"], ["ints", "strings"]],
)
@pytest.mark.parametrize(
    "group_column", ["int_groups", "string_groups", ["int_groups", "string_groups"]]
)
def test_groupby_preserves_sort(sort_column, group_column):
    # Test to ensure that groupby always preserves sort order of original
    # object. Issue #8588 and #9651

    df = DataFrame(
        {
            "int_groups": [3, 1, 0, 1, 0, 3, 3, 3],
            "string_groups": ["z", "a", "z", "a", "a", "g", "g", "g"],
            "ints": [8, 7, 4, 5, 2, 9, 1, 1],
            "floats": [2.3, 5.3, 6.2, -2.4, 2.2, 1.1, 1.1, 5],
            "strings": ["z", "d", "a", "e", "word", "word2", "42", "47"],
        }
    )

    # Try sorting on different types and with different group types

    df = df.sort_values(by=sort_column)
    g = df.groupby(group_column)

    def test_sort(x):
        tm.assert_frame_equal(x, x.sort_values(by=sort_column))

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        g.apply(test_sort)


def test_pivot_table_values_key_error():
    # This test is designed to replicate the error in issue #14938
    df = DataFrame(
        {
            "eventDate": date_range(datetime.today(), periods=20, freq="ME").tolist(),
            "thename": range(20),
        }
    )

    df["year"] = df.set_index("eventDate").index.year
    df["month"] = df.set_index("eventDate").index.month

    with pytest.raises(KeyError, match="'badname'"):
        df.reset_index().pivot_table(
            index="year", columns="month", values="badname", aggfunc="count"
        )


@pytest.mark.parametrize("columns", ["C", ["C"]])
@pytest.mark.parametrize("keys", [["A"], ["A", "B"]])
@pytest.mark.parametrize(
    "values",
    [
        [True],
        [0],
        [0.0],
        ["a"],
        Categorical([0]),
        [to_datetime(0)],
        date_range(0, 1, 1, tz="US/Eastern"),
        pd.period_range("2016-01-01", periods=3, freq="D"),
        pd.array([0], dtype="Int64"),
        pd.array([0], dtype="Float64"),
        pd.array([False], dtype="boolean"),
    ],
    ids=[
        "bool",
        "int",
        "float",
        "str",
        "cat",
        "dt64",
        "dt64tz",
        "period",
        "Int64",
        "Float64",
        "boolean",
    ],
)
@pytest.mark.parametrize("method", ["attr", "agg", "apply"])
@pytest.mark.parametrize(
    "op", ["idxmax", "idxmin", "min", "max", "sum", "prod", "skew"]
)
def test_empty_groupby(
    columns, keys, values, method, op, using_array_manager, dropna, using_infer_string
):
    # GH8093 & GH26411
    override_dtype = None

    if isinstance(values, BooleanArray) and op in ["sum", "prod"]:
        # We expect to get Int64 back for these
        override_dtype = "Int64"

    if isinstance(values[0], bool) and op in ("prod", "sum"):
        # sum/product of bools is an integer
        override_dtype = "int64"

    df = DataFrame({"A": values, "B": values, "C": values}, columns=list("ABC"))

    if hasattr(values, "dtype"):
        # check that we did the construction right
        assert (df.dtypes == values.dtype).all()

    df = df.iloc[:0]

    gb = df.groupby(keys, group_keys=False, dropna=dropna, observed=False)[columns]

    def get_result(**kwargs):
        if method == "attr":
            return getattr(gb, op)(**kwargs)
        else:
            return getattr(gb, method)(op, **kwargs)

    def get_categorical_invalid_expected():
        # Categorical is special without 'observed=True', we get an NaN entry
        #  corresponding to the unobserved group. If we passed observed=True
        #  to groupby, expected would just be 'df.set_index(keys)[columns]'
        #  as below
        lev = Categorical([0], dtype=values.dtype)
        if len(keys) != 1:
            idx = MultiIndex.from_product([lev, lev], names=keys)
        else:
            # all columns are dropped, but we end up with one row
            # Categorical is special without 'observed=True'
            idx = Index(lev, name=keys[0])

        if using_infer_string:
            columns = Index([], dtype="str")
        else:
            columns = []
        expected = DataFrame([], columns=columns, index=idx)
        return expected

    is_per = isinstance(df.dtypes.iloc[0], pd.PeriodDtype)
    is_dt64 = df.dtypes.iloc[0].kind == "M"
    is_cat = isinstance(values, Categorical)
    is_str = isinstance(df.dtypes.iloc[0], pd.StringDtype)

    if (
        isinstance(values, Categorical)
        and not values.ordered
        and op in ["min", "max", "idxmin", "idxmax"]
    ):
        if op in ["min", "max"]:
            msg = f"Cannot perform {op} with non-ordered Categorical"
            klass = TypeError
        else:
            msg = f"Can't get {op} of an empty group due to unobserved categories"
            klass = ValueError
        with pytest.raises(klass, match=msg):
            get_result()

        if op in ["min", "max", "idxmin", "idxmax"] and isinstance(columns, list):
            # i.e. DataframeGroupBy, not SeriesGroupBy
            result = get_result(numeric_only=True)
            expected = get_categorical_invalid_expected()
            tm.assert_equal(result, expected)
        return

    if op in ["prod", "sum", "skew"]:
        # ops that require more than just ordered-ness
        if is_dt64 or is_cat or is_per or (is_str and op != "sum"):
            # GH#41291
            # datetime64 -> prod and sum are invalid
            if is_dt64:
                msg = "datetime64 type does not support"
            elif is_per:
                msg = "Period type does not support"
            elif is_str:
                msg = f"dtype 'str' does not support operation '{op}'"
            else:
                msg = "category type does not support"
            if op == "skew":
                msg = "|".join([msg, "does not support reduction 'skew'"])
            with pytest.raises(TypeError, match=msg):
                get_result()

            if not isinstance(columns, list):
                # i.e. SeriesGroupBy
                return
            elif op == "skew":
                # TODO: test the numeric_only=True case
                return
            else:
                # i.e. op in ["prod", "sum"]:
                # i.e. DataFrameGroupBy
                # ops that require more than just ordered-ness
                # GH#41291
                result = get_result(numeric_only=True)

                # with numeric_only=True, these are dropped, and we get
                # an empty DataFrame back
                expected = df.set_index(keys)[[]]
                if is_cat:
                    expected = get_categorical_invalid_expected()
                tm.assert_equal(result, expected)
                return

    result = get_result()
    expected = df.set_index(keys)[columns]
    if op in ["idxmax", "idxmin"]:
        expected = expected.astype(df.index.dtype)
    if override_dtype is not None:
        expected = expected.astype(override_dtype)
    if len(keys) == 1:
        expected.index.name = keys[0]
    tm.assert_equal(result, expected)


def test_empty_groupby_apply_nonunique_columns():
    # GH#44417
    df = DataFrame(np.random.default_rng(2).standard_normal((0, 4)))
    df[3] = df[3].astype(np.int64)
    df.columns = [0, 1, 2, 0]
    gb = df.groupby(df[1], group_keys=False)
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = gb.apply(lambda x: x)
    assert (res.dtypes == df.dtypes).all()


def test_tuple_as_grouping():
    # https://github.com/pandas-dev/pandas/issues/18314
    df = DataFrame(
        {
            ("a", "b"): [1, 1, 1, 1],
            "a": [2, 2, 2, 2],
            "b": [2, 2, 2, 2],
            "c": [1, 1, 1, 1],
        }
    )

    with pytest.raises(KeyError, match=r"('a', 'b')"):
        df[["a", "b", "c"]].groupby(("a", "b"))

    result = df.groupby(("a", "b"))["c"].sum()
    expected = Series([4], name="c", index=Index([1], name=("a", "b")))
    tm.assert_series_equal(result, expected)


def test_tuple_correct_keyerror():
    # https://github.com/pandas-dev/pandas/issues/18798
    df = DataFrame(1, index=range(3), columns=MultiIndex.from_product([[1, 2], [3, 4]]))
    with pytest.raises(KeyError, match=r"^\(7, 8\)$"):
        df.groupby((7, 8)).mean()


def test_groupby_agg_ohlc_non_first():
    # GH 21716
    df = DataFrame(
        [[1], [1]],
        columns=Index(["foo"], name="mycols"),
        index=date_range("2018-01-01", periods=2, freq="D", name="dti"),
    )

    expected = DataFrame(
        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        columns=MultiIndex.from_tuples(
            (
                ("foo", "sum", "foo"),
                ("foo", "ohlc", "open"),
                ("foo", "ohlc", "high"),
                ("foo", "ohlc", "low"),
                ("foo", "ohlc", "close"),
            ),
            names=["mycols", None, None],
        ),
        index=date_range("2018-01-01", periods=2, freq="D", name="dti"),
    )

    result = df.groupby(Grouper(freq="D")).agg(["sum", "ohlc"])

    tm.assert_frame_equal(result, expected)


def test_groupby_multiindex_nat():
    # GH 9236
    values = [
        (pd.NaT, "a"),
        (datetime(2012, 1, 2), "a"),
        (datetime(2012, 1, 2), "b"),
        (datetime(2012, 1, 3), "a"),
    ]
    mi = MultiIndex.from_tuples(values, names=["date", None])
    ser = Series([3, 2, 2.5, 4], index=mi)

    result = ser.groupby(level=1).mean()
    expected = Series([3.0, 2.5], index=["a", "b"])
    tm.assert_series_equal(result, expected)


def test_groupby_empty_list_raises():
    # GH 5289
    values = zip(range(10), range(10))
    df = DataFrame(values, columns=["apple", "b"])
    msg = "Grouper and axis must be same length"
    with pytest.raises(ValueError, match=msg):
        df.groupby([[]])


def test_groupby_multiindex_series_keys_len_equal_group_axis():
    # GH 25704
    index_array = [["x", "x"], ["a", "b"], ["k", "k"]]
    index_names = ["first", "second", "third"]
    ri = MultiIndex.from_arrays(index_array, names=index_names)
    s = Series(data=[1, 2], index=ri)
    result = s.groupby(["first", "third"]).sum()

    index_array = [["x"], ["k"]]
    index_names = ["first", "third"]
    ei = MultiIndex.from_arrays(index_array, names=index_names)
    expected = Series([3], index=ei)

    tm.assert_series_equal(result, expected)


def test_groupby_groups_in_BaseGrouper():
    # GH 26326
    # Test if DataFrame grouped with a pandas.Grouper has correct groups
    mi = MultiIndex.from_product([["A", "B"], ["C", "D"]], names=["alpha", "beta"])
    df = DataFrame({"foo": [1, 2, 1, 2], "bar": [1, 2, 3, 4]}, index=mi)
    result = df.groupby([Grouper(level="alpha"), "beta"])
    expected = df.groupby(["alpha", "beta"])
    assert result.groups == expected.groups

    result = df.groupby(["beta", Grouper(level="alpha")])
    expected = df.groupby(["beta", "alpha"])
    assert result.groups == expected.groups


@pytest.mark.parametrize("group_name", ["x", ["x"]])
def test_groupby_axis_1(group_name):
    # GH 27614
    df = DataFrame(
        np.arange(12).reshape(3, 4), index=[0, 1, 0], columns=[10, 20, 10, 20]
    )
    df.index.name = "y"
    df.columns.name = "x"

    depr_msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        gb = df.groupby(group_name, axis=1)

    results = gb.sum()
    expected = df.T.groupby(group_name).sum().T
    tm.assert_frame_equal(results, expected)

    # test on MI column
    iterables = [["bar", "baz", "foo"], ["one", "two"]]
    mi = MultiIndex.from_product(iterables=iterables, names=["x", "x1"])
    df = DataFrame(np.arange(18).reshape(3, 6), index=[0, 1, 0], columns=mi)
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        gb = df.groupby(group_name, axis=1)
    results = gb.sum()
    expected = df.T.groupby(group_name).sum().T
    tm.assert_frame_equal(results, expected)


@pytest.mark.parametrize(
    "op, expected",
    [
        (
            "shift",
            {
                "time": [
                    None,
                    None,
                    Timestamp("2019-01-01 12:00:00"),
                    Timestamp("2019-01-01 12:30:00"),
                    None,
                    None,
                ]
            },
        ),
        (
            "bfill",
            {
                "time": [
                    Timestamp("2019-01-01 12:00:00"),
                    Timestamp("2019-01-01 12:30:00"),
                    Timestamp("2019-01-01 14:00:00"),
                    Timestamp("2019-01-01 14:30:00"),
                    Timestamp("2019-01-01 14:00:00"),
                    Timestamp("2019-01-01 14:30:00"),
                ]
            },
        ),
        (
            "ffill",
            {
                "time": [
                    Timestamp("2019-01-01 12:00:00"),
                    Timestamp("2019-01-01 12:30:00"),
                    Timestamp("2019-01-01 12:00:00"),
                    Timestamp("2019-01-01 12:30:00"),
                    Timestamp("2019-01-01 14:00:00"),
                    Timestamp("2019-01-01 14:30:00"),
                ]
            },
        ),
    ],
)
def test_shift_bfill_ffill_tz(tz_naive_fixture, op, expected):
    # GH19995, GH27992: Check that timezone does not drop in shift, bfill, and ffill
    tz = tz_naive_fixture
    data = {
        "id": ["A", "B", "A", "B", "A", "B"],
        "time": [
            Timestamp("2019-01-01 12:00:00"),
            Timestamp("2019-01-01 12:30:00"),
            None,
            None,
            Timestamp("2019-01-01 14:00:00"),
            Timestamp("2019-01-01 14:30:00"),
        ],
    }
    df = DataFrame(data).assign(time=lambda x: x.time.dt.tz_localize(tz))

    grouped = df.groupby("id")
    result = getattr(grouped, op)()
    expected = DataFrame(expected).assign(time=lambda x: x.time.dt.tz_localize(tz))
    tm.assert_frame_equal(result, expected)


def test_groupby_only_none_group():
    # see GH21624
    # this was crashing with "ValueError: Length of passed values is 1, index implies 0"
    df = DataFrame({"g": [None], "x": 1})
    actual = df.groupby("g")["x"].transform("sum")
    expected = Series([np.nan], name="x")

    tm.assert_series_equal(actual, expected)


def test_groupby_duplicate_index():
    # GH#29189 the groupby call here used to raise
    ser = Series([2, 5, 6, 8], index=[2.0, 4.0, 4.0, 5.0])
    gb = ser.groupby(level=0)

    result = gb.mean()
    expected = Series([2, 5.5, 8], index=[2.0, 4.0, 5.0])
    tm.assert_series_equal(result, expected)


def test_group_on_empty_multiindex(transformation_func, request):
    # GH 47787
    # With one row, those are transforms so the schema should be the same
    df = DataFrame(
        data=[[1, Timestamp("today"), 3, 4]],
        columns=["col_1", "col_2", "col_3", "col_4"],
    )
    df["col_3"] = df["col_3"].astype(int)
    df["col_4"] = df["col_4"].astype(int)
    df = df.set_index(["col_1", "col_2"])
    if transformation_func == "fillna":
        args = ("ffill",)
    else:
        args = ()
    warn = FutureWarning if transformation_func == "fillna" else None
    warn_msg = "DataFrameGroupBy.fillna is deprecated"
    with tm.assert_produces_warning(warn, match=warn_msg):
        result = df.iloc[:0].groupby(["col_1"]).transform(transformation_func, *args)
    with tm.assert_produces_warning(warn, match=warn_msg):
        expected = df.groupby(["col_1"]).transform(transformation_func, *args).iloc[:0]
    if transformation_func in ("diff", "shift"):
        expected = expected.astype(int)
    tm.assert_equal(result, expected)

    warn_msg = "SeriesGroupBy.fillna is deprecated"
    with tm.assert_produces_warning(warn, match=warn_msg):
        result = (
            df["col_3"]
            .iloc[:0]
            .groupby(["col_1"])
            .transform(transformation_func, *args)
        )
    warn_msg = "SeriesGroupBy.fillna is deprecated"
    with tm.assert_produces_warning(warn, match=warn_msg):
        expected = (
            df["col_3"]
            .groupby(["col_1"])
            .transform(transformation_func, *args)
            .iloc[:0]
        )
    if transformation_func in ("diff", "shift"):
        expected = expected.astype(int)
    tm.assert_equal(result, expected)


def test_groupby_crash_on_nunique(axis):
    # Fix following 30253
    dti = date_range("2016-01-01", periods=2, name="foo")
    df = DataFrame({("A", "B"): [1, 2], ("A", "C"): [1, 3], ("D", "B"): [0, 0]})
    df.columns.names = ("bar", "baz")
    df.index = dti

    axis_number = df._get_axis_number(axis)
    if not axis_number:
        df = df.T
        msg = "The 'axis' keyword in DataFrame.groupby is deprecated"
    else:
        msg = "DataFrame.groupby with axis=1 is deprecated"

    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby(axis=axis_number, level=0)
    result = gb.nunique()

    expected = DataFrame({"A": [1, 2], "D": [1, 1]}, index=dti)
    expected.columns.name = "bar"
    if not axis_number:
        expected = expected.T

    tm.assert_frame_equal(result, expected)

    if axis_number == 0:
        # same thing, but empty columns
        with tm.assert_produces_warning(FutureWarning, match=msg):
            gb2 = df[[]].groupby(axis=axis_number, level=0)
        exp = expected[[]]
    else:
        # same thing, but empty rows
        with tm.assert_produces_warning(FutureWarning, match=msg):
            gb2 = df.loc[[]].groupby(axis=axis_number, level=0)
        # default for empty when we can't infer a dtype is float64
        exp = expected.loc[[]].astype(np.float64)

    res = gb2.nunique()
    tm.assert_frame_equal(res, exp)


def test_groupby_list_level():
    # GH 9790
    expected = DataFrame(np.arange(0, 9).reshape(3, 3), dtype=float)
    result = expected.groupby(level=[0]).mean()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "max_seq_items, expected",
    [
        (5, "{0: [0], 1: [1], 2: [2], 3: [3], 4: [4]}"),
        (4, "{0: [0], 1: [1], 2: [2], 3: [3], ...}"),
        (1, "{0: [0], ...}"),
    ],
)
def test_groups_repr_truncates(max_seq_items, expected):
    # GH 1135
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 1)))
    df["a"] = df.index

    with pd.option_context("display.max_seq_items", max_seq_items):
        result = df.groupby("a").groups.__repr__()
        assert result == expected

        result = df.groupby(np.array(df.a)).groups.__repr__()
        assert result == expected


def test_group_on_two_row_multiindex_returns_one_tuple_key():
    # GH 18451
    df = DataFrame([{"a": 1, "b": 2, "c": 99}, {"a": 1, "b": 2, "c": 88}])
    df = df.set_index(["a", "b"])

    grp = df.groupby(["a", "b"])
    result = grp.indices
    expected = {(1, 2): np.array([0, 1], dtype=np.int64)}

    assert len(result) == 1
    key = (1, 2)
    assert (result[key] == expected[key]).all()


@pytest.mark.parametrize(
    "klass, attr, value",
    [
        (DataFrame, "level", "a"),
        (DataFrame, "as_index", False),
        (DataFrame, "sort", False),
        (DataFrame, "group_keys", False),
        (DataFrame, "observed", True),
        (DataFrame, "dropna", False),
        (Series, "level", "a"),
        (Series, "as_index", False),
        (Series, "sort", False),
        (Series, "group_keys", False),
        (Series, "observed", True),
        (Series, "dropna", False),
    ],
)
def test_subsetting_columns_keeps_attrs(klass, attr, value):
    # GH 9959 - When subsetting columns, don't drop attributes
    df = DataFrame({"a": [1], "b": [2], "c": [3]})
    if attr != "axis":
        df = df.set_index("a")

    expected = df.groupby("a", **{attr: value})
    result = expected[["b"]] if klass is DataFrame else expected["b"]
    assert getattr(result, attr) == getattr(expected, attr)


def test_subsetting_columns_axis_1():
    # GH 37725
    df = DataFrame({"A": [1], "B": [2], "C": [3]})
    msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        g = df.groupby([0, 0, 1], axis=1)
    match = "Cannot subset columns when using axis=1"
    with pytest.raises(ValueError, match=match):
        g[["A", "B"]].sum()


@pytest.mark.parametrize("func", ["sum", "any", "shift"])
def test_groupby_column_index_name_lost(func):
    # GH: 29764 groupby loses index sometimes
    expected = Index(["a"], name="idx")
    df = DataFrame([[1]], columns=expected)
    df_grouped = df.groupby([1])
    result = getattr(df_grouped, func)().columns
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "infer_string",
    [
        False,
        pytest.param(True, marks=td.skip_if_no("pyarrow")),
    ],
)
def test_groupby_duplicate_columns(infer_string):
    # GH: 31735
    if infer_string:
        pytest.importorskip("pyarrow")
    df = DataFrame(
        {"A": ["f", "e", "g", "h"], "B": ["a", "b", "c", "d"], "C": [1, 2, 3, 4]}
    ).astype(object)
    df.columns = ["A", "B", "B"]
    with pd.option_context("future.infer_string", infer_string):
        result = df.groupby([0, 0, 0, 0]).min()
    expected = DataFrame(
        [["e", "a", 1]], index=np.array([0]), columns=["A", "B", "B"], dtype=object
    )
    tm.assert_frame_equal(result, expected)


def test_groupby_series_with_tuple_name():
    # GH 37755
    ser = Series([1, 2, 3, 4], index=[1, 1, 2, 2], name=("a", "a"))
    ser.index.name = ("b", "b")
    result = ser.groupby(level=0).last()
    expected = Series([2, 4], index=[1, 2], name=("a", "a"))
    expected.index.name = ("b", "b")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "func, values", [("sum", [97.0, 98.0]), ("mean", [24.25, 24.5])]
)
def test_groupby_numerical_stability_sum_mean(func, values):
    # GH#38778
    data = [1e16, 1e16, 97, 98, -5e15, -5e15, -5e15, -5e15]
    df = DataFrame({"group": [1, 2] * 4, "a": data, "b": data})
    result = getattr(df.groupby("group"), func)()
    expected = DataFrame({"a": values, "b": values}, index=Index([1, 2], name="group"))
    tm.assert_frame_equal(result, expected)


def test_groupby_numerical_stability_cumsum():
    # GH#38934
    data = [1e16, 1e16, 97, 98, -5e15, -5e15, -5e15, -5e15]
    df = DataFrame({"group": [1, 2] * 4, "a": data, "b": data})
    result = df.groupby("group").cumsum()
    exp_data = (
        [1e16] * 2 + [1e16 + 96, 1e16 + 98] + [5e15 + 97, 5e15 + 98] + [97.0, 98.0]
    )
    expected = DataFrame({"a": exp_data, "b": exp_data})
    tm.assert_frame_equal(result, expected, check_exact=True)


def test_groupby_cumsum_skipna_false():
    # GH#46216 don't propagate np.nan above the diagonal
    arr = np.random.default_rng(2).standard_normal((5, 5))
    df = DataFrame(arr)
    for i in range(5):
        df.iloc[i, i] = np.nan

    df["A"] = 1
    gb = df.groupby("A")

    res = gb.cumsum(skipna=False)

    expected = df[[0, 1, 2, 3, 4]].cumsum(skipna=False)
    tm.assert_frame_equal(res, expected)


def test_groupby_cumsum_timedelta64():
    # GH#46216 don't ignore is_datetimelike in libgroupby.group_cumsum
    dti = date_range("2016-01-01", periods=5)
    ser = Series(dti) - dti[0]
    ser[2] = pd.NaT

    df = DataFrame({"A": 1, "B": ser})
    gb = df.groupby("A")

    res = gb.cumsum(numeric_only=False, skipna=True)
    exp = DataFrame({"B": [ser[0], ser[1], pd.NaT, ser[4], ser[4] * 2]})
    tm.assert_frame_equal(res, exp)

    res = gb.cumsum(numeric_only=False, skipna=False)
    exp = DataFrame({"B": [ser[0], ser[1], pd.NaT, pd.NaT, pd.NaT]})
    tm.assert_frame_equal(res, exp)


def test_groupby_mean_duplicate_index(rand_series_with_duplicate_datetimeindex):
    dups = rand_series_with_duplicate_datetimeindex
    result = dups.groupby(level=0).mean()
    expected = dups.groupby(dups.index).mean()
    tm.assert_series_equal(result, expected)


def test_groupby_all_nan_groups_drop():
    # GH 15036
    s = Series([1, 2, 3], [np.nan, np.nan, np.nan])
    result = s.groupby(s.index).sum()
    expected = Series([], index=Index([], dtype=np.float64), dtype=np.int64)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("numeric_only", [True, False])
def test_groupby_empty_multi_column(as_index, numeric_only):
    # GH 15106 & GH 41998
    df = DataFrame(data=[], columns=["A", "B", "C"])
    gb = df.groupby(["A", "B"], as_index=as_index)
    result = gb.sum(numeric_only=numeric_only)
    if as_index:
        index = MultiIndex([[], []], [[], []], names=["A", "B"])
        columns = ["C"] if not numeric_only else Index([], dtype="str")
    else:
        index = RangeIndex(0)
        columns = ["A", "B", "C"] if not numeric_only else ["A", "B"]
    expected = DataFrame([], columns=columns, index=index)
    tm.assert_frame_equal(result, expected)


def test_groupby_aggregation_non_numeric_dtype():
    # GH #43108
    df = DataFrame(
        [["M", [1]], ["M", [1]], ["W", [10]], ["W", [20]]], columns=["MW", "v"]
    )

    expected = DataFrame(
        {
            "v": [[1, 1], [10, 20]],
        },
        index=Index(["M", "W"], name="MW"),
    )

    gb = df.groupby(by=["MW"])
    result = gb.sum()
    tm.assert_frame_equal(result, expected)


def test_groupby_aggregation_multi_non_numeric_dtype():
    # GH #42395
    df = DataFrame(
        {
            "x": [1, 0, 1, 1, 0],
            "y": [Timedelta(i, "days") for i in range(1, 6)],
            "z": [Timedelta(i * 10, "days") for i in range(1, 6)],
        }
    )

    expected = DataFrame(
        {
            "y": [Timedelta(i, "days") for i in range(7, 9)],
            "z": [Timedelta(i * 10, "days") for i in range(7, 9)],
        },
        index=Index([0, 1], dtype="int64", name="x"),
    )

    gb = df.groupby(by=["x"])
    result = gb.sum()
    tm.assert_frame_equal(result, expected)


def test_groupby_aggregation_numeric_with_non_numeric_dtype():
    # GH #43108
    df = DataFrame(
        {
            "x": [1, 0, 1, 1, 0],
            "y": [Timedelta(i, "days") for i in range(1, 6)],
            "z": list(range(1, 6)),
        }
    )

    expected = DataFrame(
        {"y": [Timedelta(7, "days"), Timedelta(8, "days")], "z": [7, 8]},
        index=Index([0, 1], dtype="int64", name="x"),
    )

    gb = df.groupby(by=["x"])
    result = gb.sum()
    tm.assert_frame_equal(result, expected)


def test_groupby_filtered_df_std():
    # GH 16174
    dicts = [
        {"filter_col": False, "groupby_col": True, "bool_col": True, "float_col": 10.5},
        {"filter_col": True, "groupby_col": True, "bool_col": True, "float_col": 20.5},
        {"filter_col": True, "groupby_col": True, "bool_col": True, "float_col": 30.5},
    ]
    df = DataFrame(dicts)

    df_filter = df[df["filter_col"] == True]  # noqa: E712
    dfgb = df_filter.groupby("groupby_col")
    result = dfgb.std()
    expected = DataFrame(
        [[0.0, 0.0, 7.071068]],
        columns=["filter_col", "bool_col", "float_col"],
        index=Index([True], name="groupby_col"),
    )
    tm.assert_frame_equal(result, expected)


def test_datetime_categorical_multikey_groupby_indices():
    # GH 26859
    df = DataFrame(
        {
            "a": Series(list("abc")),
            "b": Series(
                to_datetime(["2018-01-01", "2018-02-01", "2018-03-01"]),
                dtype="category",
            ),
            "c": Categorical.from_codes([-1, 0, 1], categories=[0, 1]),
        }
    )
    result = df.groupby(["a", "b"], observed=False).indices
    expected = {
        ("a", Timestamp("2018-01-01 00:00:00")): np.array([0]),
        ("b", Timestamp("2018-02-01 00:00:00")): np.array([1]),
        ("c", Timestamp("2018-03-01 00:00:00")): np.array([2]),
    }
    assert result == expected


def test_rolling_wrong_param_min_period():
    # GH34037
    name_l = ["Alice"] * 5 + ["Bob"] * 5
    val_l = [np.nan, np.nan, 1, 2, 3] + [np.nan, 1, 2, 3, 4]
    test_df = DataFrame([name_l, val_l]).T
    test_df.columns = ["name", "val"]

    result_error_msg = (
        r"^[a-zA-Z._]*\(\) got an unexpected keyword argument 'min_period'"
    )
    with pytest.raises(TypeError, match=result_error_msg):
        test_df.groupby("name")["val"].rolling(window=2, min_period=1).sum()


def test_by_column_values_with_same_starting_value(any_string_dtype):
    # GH29635
    df = DataFrame(
        {
            "Name": ["Thomas", "Thomas", "Thomas John"],
            "Credit": [1200, 1300, 900],
            "Mood": Series(["sad", "happy", "happy"], dtype=any_string_dtype),
        }
    )
    aggregate_details = {"Mood": Series.mode, "Credit": "sum"}

    result = df.groupby(["Name"]).agg(aggregate_details)
    expected_result = DataFrame(
        {
            "Mood": [["happy", "sad"], "happy"],
            "Credit": [2500, 900],
            "Name": ["Thomas", "Thomas John"],
        }
    ).set_index("Name")

    tm.assert_frame_equal(result, expected_result)


def test_groupby_none_in_first_mi_level():
    # GH#47348
    arr = [[None, 1, 0, 1], [2, 3, 2, 3]]
    ser = Series(1, index=MultiIndex.from_arrays(arr, names=["a", "b"]))
    result = ser.groupby(level=[0, 1]).sum()
    expected = Series(
        [1, 2], MultiIndex.from_tuples([(0.0, 2), (1.0, 3)], names=["a", "b"])
    )
    tm.assert_series_equal(result, expected)


def test_groupby_none_column_name(using_infer_string):
    # GH#47348
    df = DataFrame({None: [1, 1, 2, 2], "b": [1, 1, 2, 3], "c": [4, 5, 6, 7]})
    by = [np.nan] if using_infer_string else [None]
    gb = df.groupby(by=by)
    result = gb.sum()
    expected = DataFrame({"b": [2, 5], "c": [9, 13]}, index=Index([1, 2], name=by[0]))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("selection", [None, "a", ["a"]])
def test_single_element_list_grouping(selection):
    # GH#42795, GH#53500
    df = DataFrame({"a": [1, 2], "b": [np.nan, 5], "c": [np.nan, 2]}, index=["x", "y"])
    grouped = df.groupby(["a"]) if selection is None else df.groupby(["a"])[selection]
    result = [key for key, _ in grouped]

    expected = [(1,), (2,)]
    assert result == expected


def test_groupby_string_dtype():
    # GH 40148
    df = DataFrame({"str_col": ["a", "b", "c", "a"], "num_col": [1, 2, 3, 2]})
    df["str_col"] = df["str_col"].astype("string")
    expected = DataFrame(
        {
            "str_col": [
                "a",
                "b",
                "c",
            ],
            "num_col": [1.5, 2.0, 3.0],
        }
    )
    expected["str_col"] = expected["str_col"].astype("string")
    grouped = df.groupby("str_col", as_index=False)
    result = grouped.mean()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "level_arg, multiindex", [([0], False), ((0,), False), ([0], True), ((0,), True)]
)
def test_single_element_listlike_level_grouping_deprecation(level_arg, multiindex):
    # GH 51583
    df = DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}, index=["x", "y"])
    if multiindex:
        df = df.set_index(["a", "b"])
    depr_msg = (
        "Creating a Groupby object with a length-1 list-like "
        "level parameter will yield indexes as tuples in a future version. "
        "To keep indexes as scalars, create Groupby objects with "
        "a scalar level parameter instead."
    )
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        [key for key, _ in df.groupby(level=level_arg)]


@pytest.mark.parametrize("func", ["sum", "cumsum", "cumprod", "prod"])
def test_groupby_avoid_casting_to_float(func):
    # GH#37493
    val = 922337203685477580
    df = DataFrame({"a": 1, "b": [val]})
    result = getattr(df.groupby("a"), func)() - val
    expected = DataFrame({"b": [0]}, index=Index([1], name="a"))
    if func in ["cumsum", "cumprod"]:
        expected = expected.reset_index(drop=True)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("func, val", [("sum", 3), ("prod", 2)])
def test_groupby_sum_support_mask(any_numeric_ea_dtype, func, val):
    # GH#37493
    df = DataFrame({"a": 1, "b": [1, 2, pd.NA]}, dtype=any_numeric_ea_dtype)
    result = getattr(df.groupby("a"), func)()
    expected = DataFrame(
        {"b": [val]},
        index=Index([1], name="a", dtype=any_numeric_ea_dtype),
        dtype=any_numeric_ea_dtype,
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("val, dtype", [(111, "int"), (222, "uint")])
def test_groupby_overflow(val, dtype):
    # GH#37493
    df = DataFrame({"a": 1, "b": [val, val]}, dtype=f"{dtype}8")
    result = df.groupby("a").sum()
    expected = DataFrame(
        {"b": [val * 2]},
        index=Index([1], name="a", dtype=f"{dtype}8"),
        dtype=f"{dtype}64",
    )
    tm.assert_frame_equal(result, expected)

    result = df.groupby("a").cumsum()
    expected = DataFrame({"b": [val, val * 2]}, dtype=f"{dtype}64")
    tm.assert_frame_equal(result, expected)

    result = df.groupby("a").prod()
    expected = DataFrame(
        {"b": [val * val]},
        index=Index([1], name="a", dtype=f"{dtype}8"),
        dtype=f"{dtype}64",
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("skipna, val", [(True, 3), (False, pd.NA)])
def test_groupby_cumsum_mask(any_numeric_ea_dtype, skipna, val):
    # GH#37493
    df = DataFrame({"a": 1, "b": [1, pd.NA, 2]}, dtype=any_numeric_ea_dtype)
    result = df.groupby("a").cumsum(skipna=skipna)
    expected = DataFrame(
        {"b": [1, pd.NA, val]},
        dtype=any_numeric_ea_dtype,
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "val_in, index, val_out",
    [
        (
            [1.0, 2.0, 3.0, 4.0, 5.0],
            ["foo", "foo", "bar", "baz", "blah"],
            [3.0, 4.0, 5.0, 3.0],
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ["foo", "foo", "bar", "baz", "blah", "blah"],
            [3.0, 4.0, 11.0, 3.0],
        ),
    ],
)
def test_groupby_index_name_in_index_content(val_in, index, val_out):
    # GH 48567
    series = Series(data=val_in, name="values", index=Index(index, name="blah"))
    result = series.groupby("blah").sum()
    expected = Series(
        data=val_out,
        name="values",
        index=Index(["bar", "baz", "blah", "foo"], name="blah"),
    )
    tm.assert_series_equal(result, expected)

    result = series.to_frame().groupby("blah").sum()
    expected = expected.to_frame()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("n", [1, 10, 32, 100, 1000])
def test_sum_of_booleans(n):
    # GH 50347
    df = DataFrame({"groupby_col": 1, "bool": [True] * n})
    df["bool"] = df["bool"].eq(True)
    result = df.groupby("groupby_col").sum()
    expected = DataFrame({"bool": [n]}, index=Index([1], name="groupby_col"))
    tm.assert_frame_equal(result, expected)


@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in remainder:RuntimeWarning"
)
@pytest.mark.parametrize("method", ["head", "tail", "nth", "first", "last"])
def test_groupby_method_drop_na(method):
    # GH 21755
    df = DataFrame({"A": ["a", np.nan, "b", np.nan, "c"], "B": range(5)})

    if method == "nth":
        result = getattr(df.groupby("A"), method)(n=0)
    else:
        result = getattr(df.groupby("A"), method)()

    if method in ["first", "last"]:
        expected = DataFrame({"B": [0, 2, 4]}).set_index(
            Series(["a", "b", "c"], name="A")
        )
    else:
        expected = DataFrame({"A": ["a", "b", "c"], "B": [0, 2, 4]}, index=[0, 2, 4])
    tm.assert_frame_equal(result, expected)


def test_groupby_reduce_period():
    # GH#51040
    pi = pd.period_range("2016-01-01", periods=100, freq="D")
    grps = list(range(10)) * 10
    ser = pi.to_series()
    gb = ser.groupby(grps)

    with pytest.raises(TypeError, match="Period type does not support sum operations"):
        gb.sum()
    with pytest.raises(
        TypeError, match="Period type does not support cumsum operations"
    ):
        gb.cumsum()
    with pytest.raises(TypeError, match="Period type does not support prod operations"):
        gb.prod()
    with pytest.raises(
        TypeError, match="Period type does not support cumprod operations"
    ):
        gb.cumprod()

    res = gb.max()
    expected = ser[-10:]
    expected.index = Index(range(10), dtype=int)
    tm.assert_series_equal(res, expected)

    res = gb.min()
    expected = ser[:10]
    expected.index = Index(range(10), dtype=int)
    tm.assert_series_equal(res, expected)


def test_obj_with_exclusions_duplicate_columns():
    # GH#50806
    df = DataFrame([[0, 1, 2, 3]])
    df.columns = [0, 1, 2, 0]
    gb = df.groupby(df[1])
    result = gb._obj_with_exclusions
    expected = df.take([0, 2, 3], axis=1)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("numeric_only", [True, False])
def test_groupby_numeric_only_std_no_result(numeric_only):
    # GH 51080
    dicts_non_numeric = [{"a": "foo", "b": "bar"}, {"a": "car", "b": "dar"}]
    df = DataFrame(dicts_non_numeric, dtype=object)
    dfgb = df.groupby("a", as_index=False, sort=False)

    if numeric_only:
        result = dfgb.std(numeric_only=True)
        expected_df = DataFrame(["foo", "car"], columns=["a"])
        tm.assert_frame_equal(result, expected_df)
    else:
        with pytest.raises(
            ValueError, match="could not convert string to float: 'bar'"
        ):
            dfgb.std(numeric_only=numeric_only)


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast:RuntimeWarning")
def test_grouping_with_categorical_interval_columns():
    # GH#34164
    df = DataFrame({"x": [0.1, 0.2, 0.3, -0.4, 0.5], "w": ["a", "b", "a", "c", "a"]})
    qq = pd.qcut(df["x"], q=np.linspace(0, 1, 5))
    result = df.groupby([qq, "w"], observed=False)["x"].agg("mean")
    categorical_index_level_1 = Categorical(
        [
            Interval(-0.401, 0.1, closed="right"),
            Interval(0.1, 0.2, closed="right"),
            Interval(0.2, 0.3, closed="right"),
            Interval(0.3, 0.5, closed="right"),
        ],
        ordered=True,
    )
    index_level_2 = ["a", "b", "c"]
    mi = MultiIndex.from_product(
        [categorical_index_level_1, index_level_2], names=["x", "w"]
    )
    expected = Series(
        np.array(
            [
                0.1,
                np.nan,
                -0.4,
                np.nan,
                0.2,
                np.nan,
                0.3,
                np.nan,
                np.nan,
                0.5,
                np.nan,
                np.nan,
            ]
        ),
        index=mi,
        name="x",
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("bug_var", [1, "a"])
def test_groupby_sum_on_nan_should_return_nan(bug_var):
    # GH 24196
    df = DataFrame({"A": [bug_var, bug_var, bug_var, np.nan]})
    if isinstance(bug_var, str):
        df = df.astype(object)
    dfgb = df.groupby(lambda x: x)
    result = dfgb.sum(min_count=1)

    expected_df = DataFrame(
        [bug_var, bug_var, bug_var, None], columns=["A"], dtype=df["A"].dtype
    )
    tm.assert_frame_equal(result, expected_df)


@pytest.mark.parametrize(
    "method",
    [
        "count",
        "corr",
        "cummax",
        "cummin",
        "cumprod",
        "describe",
        "rank",
        "quantile",
        "diff",
        "shift",
        "all",
        "any",
        "idxmin",
        "idxmax",
        "ffill",
        "bfill",
        "pct_change",
    ],
)
def test_groupby_selection_with_methods(df, method):
    # some methods which require DatetimeIndex
    rng = date_range("2014", periods=len(df))
    df.index = rng

    g = df.groupby(["A"])[["C"]]
    g_exp = df[["C"]].groupby(df["A"])
    # TODO check groupby with > 1 col ?

    res = getattr(g, method)()
    exp = getattr(g_exp, method)()

    # should always be frames!
    tm.assert_frame_equal(res, exp)


def test_groupby_selection_other_methods(df):
    # some methods which require DatetimeIndex
    rng = date_range("2014", periods=len(df))
    df.columns.name = "foo"
    df.index = rng

    g = df.groupby(["A"])[["C"]]
    g_exp = df[["C"]].groupby(df["A"])

    # methods which aren't just .foo()
    warn_msg = "DataFrameGroupBy.fillna is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=warn_msg):
        tm.assert_frame_equal(g.fillna(0), g_exp.fillna(0))
    msg = "DataFrameGroupBy.dtypes is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        tm.assert_frame_equal(g.dtypes, g_exp.dtypes)
    tm.assert_frame_equal(g.apply(lambda x: x.sum()), g_exp.apply(lambda x: x.sum()))

    tm.assert_frame_equal(g.resample("D").mean(), g_exp.resample("D").mean())
    tm.assert_frame_equal(g.resample("D").ohlc(), g_exp.resample("D").ohlc())

    tm.assert_frame_equal(
        g.filter(lambda x: len(x) == 3), g_exp.filter(lambda x: len(x) == 3)
    )


def test_groupby_with_Time_Grouper(unit):
    idx2 = to_datetime(
        [
            "2016-08-31 22:08:12.000",
            "2016-08-31 22:09:12.200",
            "2016-08-31 22:20:12.400",
        ]
    ).as_unit(unit)

    test_data = DataFrame(
        {"quant": [1.0, 1.0, 3.0], "quant2": [1.0, 1.0, 3.0], "time2": idx2}
    )

    time2 = date_range("2016-08-31 22:08:00", periods=13, freq="1min", unit=unit)
    expected_output = DataFrame(
        {
            "time2": time2,
            "quant": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            "quant2": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        }
    )

    gb = test_data.groupby(Grouper(key="time2", freq="1min"))
    result = gb.count().reset_index()

    tm.assert_frame_equal(result, expected_output)


def test_groupby_series_with_datetimeindex_month_name():
    # GH 48509
    s = Series([0, 1, 0], index=date_range("2022-01-01", periods=3), name="jan")
    result = s.groupby(s).count()
    expected = Series([2, 1], name="jan")
    expected.index.name = "jan"
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("test_series", [True, False])
@pytest.mark.parametrize(
    "kwarg, value, name, warn",
    [
        ("by", "a", 1, None),
        ("by", ["a"], 1, FutureWarning),
        ("by", ["a"], (1,), None),
        ("level", 0, 1, None),
        ("level", [0], 1, FutureWarning),
        ("level", [0], (1,), None),
    ],
)
def test_depr_get_group_len_1_list_likes(test_series, kwarg, value, name, warn):
    # GH#25971
    obj = DataFrame({"b": [3, 4, 5]}, index=Index([1, 1, 2], name="a"))
    if test_series:
        obj = obj["b"]
    gb = obj.groupby(**{kwarg: value})
    msg = "you will need to pass a length-1 tuple"
    with tm.assert_produces_warning(warn, match=msg):
        result = gb.get_group(name)
    if test_series:
        expected = Series([3, 4], index=Index([1, 1], name="a"), name="b")
    else:
        expected = DataFrame({"b": [3, 4]}, index=Index([1, 1], name="a"))
    tm.assert_equal(result, expected)


def test_groupby_ngroup_with_nan():
    # GH#50100
    df = DataFrame({"a": Categorical([np.nan]), "b": [1]})
    result = df.groupby(["a", "b"], dropna=False, observed=False).ngroup()
    expected = Series([0])
    tm.assert_series_equal(result, expected)


def test_get_group_axis_1():
    # GH#54858
    df = DataFrame(
        {
            "col1": [0, 3, 2, 3],
            "col2": [4, 1, 6, 7],
            "col3": [3, 8, 2, 10],
            "col4": [1, 13, 6, 15],
            "col5": [-4, 5, 6, -7],
        }
    )
    with tm.assert_produces_warning(FutureWarning, match="deprecated"):
        grouped = df.groupby(axis=1, by=[1, 2, 3, 2, 1])
    result = grouped.get_group(1)
    expected = DataFrame(
        {
            "col1": [0, 3, 2, 3],
            "col5": [-4, 5, 6, -7],
        }
    )
    tm.assert_frame_equal(result, expected)


def test_groupby_ffill_with_duplicated_index():
    # GH#43412
    df = DataFrame({"a": [1, 2, 3, 4, np.nan, np.nan]}, index=[0, 1, 2, 0, 1, 2])

    result = df.groupby(level=0).ffill()
    expected = DataFrame({"a": [1, 2, 3, 4, 2, 3]}, index=[0, 1, 2, 0, 1, 2])
    tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize("test_series", [True, False])
def test_decimal_na_sort(test_series):
    # GH#54847
    # We catch both TypeError and decimal.InvalidOperation exceptions in safe_sort.
    # If this next assert raises, we can just catch TypeError
    assert not isinstance(decimal.InvalidOperation, TypeError)
    df = DataFrame(
        {
            "key": [Decimal(1), Decimal(1), None, None],
            "value": [Decimal(2), Decimal(3), Decimal(4), Decimal(5)],
        }
    )
    gb = df.groupby("key", dropna=False)
    if test_series:
        gb = gb["value"]
    result = gb._grouper.result_index
    expected = Index([Decimal(1), None], name="key")
    tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\attr\__init__.py ===
# SPDX-License-Identifier: MIT

"""
Classes Without Boilerplate
"""

from functools import partial
from typing import Callable, Literal, Protocol

from . import converters, exceptions, filters, setters, validators
from ._cmp import cmp_using
from ._config import get_run_validators, set_run_validators
from ._funcs import asdict, assoc, astuple, has, resolve_types
from ._make import (
    NOTHING,
    Attribute,
    Converter,
    Factory,
    _Nothing,
    attrib,
    attrs,
    evolve,
    fields,
    fields_dict,
    make_class,
    validate,
)
from ._next_gen import define, field, frozen, mutable
from ._version_info import VersionInfo


s = attributes = attrs
ib = attr = attrib
dataclass = partial(attrs, auto_attribs=True)  # happy Easter ;)


class AttrsInstance(Protocol):
    pass


NothingType = Literal[_Nothing.NOTHING]

__all__ = [
    "NOTHING",
    "Attribute",
    "AttrsInstance",
    "Converter",
    "Factory",
    "NothingType",
    "asdict",
    "assoc",
    "astuple",
    "attr",
    "attrib",
    "attributes",
    "attrs",
    "cmp_using",
    "converters",
    "define",
    "evolve",
    "exceptions",
    "field",
    "fields",
    "fields_dict",
    "filters",
    "frozen",
    "get_run_validators",
    "has",
    "ib",
    "make_class",
    "mutable",
    "resolve_types",
    "s",
    "set_run_validators",
    "setters",
    "validate",
    "validators",
]


def _make_getattr(mod_name: str) -> Callable:
    """
    Create a metadata proxy for packaging information that uses *mod_name* in
    its warnings and errors.
    """

    def __getattr__(name: str) -> str:
        if name not in ("__version__", "__version_info__"):
            msg = f"module {mod_name} has no attribute {name}"
            raise AttributeError(msg)

        from importlib.metadata import metadata

        meta = metadata("attrs")

        if name == "__version_info__":
            return VersionInfo._from_version_string(meta["version"])

        return meta["version"]

    return __getattr__


__getattr__ = _make_getattr(__name__)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\utils\virtualenv.py ===
import logging
import os
import re
import site
import sys
from typing import List, Optional

logger = logging.getLogger(__name__)
_INCLUDE_SYSTEM_SITE_PACKAGES_REGEX = re.compile(
    r"include-system-site-packages\s*=\s*(?P<value>true|false)"
)


def _running_under_venv() -> bool:
    """Checks if sys.base_prefix and sys.prefix match.

    This handles PEP 405 compliant virtual environments.
    """
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _running_under_legacy_virtualenv() -> bool:
    """Checks if sys.real_prefix is set.

    This handles virtual environments created with pypa's virtualenv.
    """
    # pypa/virtualenv case
    return hasattr(sys, "real_prefix")


def running_under_virtualenv() -> bool:
    """True if we're running inside a virtual environment, False otherwise."""
    return _running_under_venv() or _running_under_legacy_virtualenv()


def _get_pyvenv_cfg_lines() -> Optional[List[str]]:
    """Reads {sys.prefix}/pyvenv.cfg and returns its contents as list of lines

    Returns None, if it could not read/access the file.
    """
    pyvenv_cfg_file = os.path.join(sys.prefix, "pyvenv.cfg")
    try:
        # Although PEP 405 does not specify, the built-in venv module always
        # writes with UTF-8. (pypa/pip#8717)
        with open(pyvenv_cfg_file, encoding="utf-8") as f:
            return f.read().splitlines()  # avoids trailing newlines
    except OSError:
        return None


def _no_global_under_venv() -> bool:
    """Check `{sys.prefix}/pyvenv.cfg` for system site-packages inclusion

    PEP 405 specifies that when system site-packages are not supposed to be
    visible from a virtual environment, `pyvenv.cfg` must contain the following
    line:

        include-system-site-packages = false

    Additionally, log a warning if accessing the file fails.
    """
    cfg_lines = _get_pyvenv_cfg_lines()
    if cfg_lines is None:
        # We're not in a "sane" venv, so assume there is no system
        # site-packages access (since that's PEP 405's default state).
        logger.warning(
            "Could not access 'pyvenv.cfg' despite a virtual environment "
            "being active. Assuming global site-packages is not accessible "
            "in this environment."
        )
        return True

    for line in cfg_lines:
        match = _INCLUDE_SYSTEM_SITE_PACKAGES_REGEX.match(line)
        if match is not None and match.group("value") == "false":
            return True
    return False


def _no_global_under_legacy_virtualenv() -> bool:
    """Check if "no-global-site-packages.txt" exists beside site.py

    This mirrors logic in pypa/virtualenv for determining whether system
    site-packages are visible in the virtual environment.
    """
    site_mod_dir = os.path.dirname(os.path.abspath(site.__file__))
    no_global_site_packages_file = os.path.join(
        site_mod_dir,
        "no-global-site-packages.txt",
    )
    return os.path.exists(no_global_site_packages_file)


def virtualenv_no_global() -> bool:
    """Returns a boolean, whether running in venv with no system site-packages."""
    # PEP 405 compliance needs to be checked first since virtualenv >=20 would
    # return True for both checks, but is only able to use the PEP 405 config.
    if _running_under_venv():
        return _no_global_under_venv()

    if _running_under_legacy_virtualenv():
        return _no_global_under_legacy_virtualenv()

    return False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\pygments\scanner.py ===
"""
    pygments.scanner
    ~~~~~~~~~~~~~~~~

    This library implements a regex based scanner. Some languages
    like Pascal are easy to parse but have some keywords that
    depend on the context. Because of this it's impossible to lex
    that just by using a regular expression lexer like the
    `RegexLexer`.

    Have a look at the `DelphiLexer` to get an idea of how to use
    this scanner.

    :copyright: Copyright 2006-2025 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""
import re


class EndOfText(RuntimeError):
    """
    Raise if end of text is reached and the user
    tried to call a match function.
    """


class Scanner:
    """
    Simple scanner

    All method patterns are regular expression strings (not
    compiled expressions!)
    """

    def __init__(self, text, flags=0):
        """
        :param text:    The text which should be scanned
        :param flags:   default regular expression flags
        """
        self.data = text
        self.data_length = len(text)
        self.start_pos = 0
        self.pos = 0
        self.flags = flags
        self.last = None
        self.match = None
        self._re_cache = {}

    def eos(self):
        """`True` if the scanner reached the end of text."""
        return self.pos >= self.data_length
    eos = property(eos, eos.__doc__)

    def check(self, pattern):
        """
        Apply `pattern` on the current position and return
        the match object. (Doesn't touch pos). Use this for
        lookahead.
        """
        if self.eos:
            raise EndOfText()
        if pattern not in self._re_cache:
            self._re_cache[pattern] = re.compile(pattern, self.flags)
        return self._re_cache[pattern].match(self.data, self.pos)

    def test(self, pattern):
        """Apply a pattern on the current position and check
        if it patches. Doesn't touch pos.
        """
        return self.check(pattern) is not None

    def scan(self, pattern):
        """
        Scan the text for the given pattern and update pos/match
        and related fields. The return value is a boolean that
        indicates if the pattern matched. The matched value is
        stored on the instance as ``match``, the last value is
        stored as ``last``. ``start_pos`` is the position of the
        pointer before the pattern was matched, ``pos`` is the
        end position.
        """
        if self.eos:
            raise EndOfText()
        if pattern not in self._re_cache:
            self._re_cache[pattern] = re.compile(pattern, self.flags)
        self.last = self.match
        m = self._re_cache[pattern].match(self.data, self.pos)
        if m is None:
            return False
        self.start_pos = m.start()
        self.pos = m.end()
        self.match = m.group()
        return True

    def get_char(self):
        """Scan exactly one char."""
        self.scan('.')

    def __repr__(self):
        return '<%s %d/%d>' % (
            self.__class__.__name__,
            self.pos,
            self.data_length
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\headless_experimental.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: HeadlessExperimental (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

@dataclass
class ScreenshotParams:
    '''
    Encoding options for a screenshot.
    '''
    #: Image compression format (defaults to png).
    format_: typing.Optional[str] = None

    #: Compression quality from range [0..100] (jpeg and webp only).
    quality: typing.Optional[int] = None

    #: Optimize image encoding for speed, not for resulting size (defaults to false)
    optimize_for_speed: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        if self.format_ is not None:
            json['format'] = self.format_
        if self.quality is not None:
            json['quality'] = self.quality
        if self.optimize_for_speed is not None:
            json['optimizeForSpeed'] = self.optimize_for_speed
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            format_=str(json['format']) if 'format' in json else None,
            quality=int(json['quality']) if 'quality' in json else None,
            optimize_for_speed=bool(json['optimizeForSpeed']) if 'optimizeForSpeed' in json else None,
        )


def begin_frame(
        frame_time_ticks: typing.Optional[float] = None,
        interval: typing.Optional[float] = None,
        no_display_updates: typing.Optional[bool] = None,
        screenshot: typing.Optional[ScreenshotParams] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[bool, typing.Optional[str]]]:
    '''
    Sends a BeginFrame to the target and returns when the frame was completed. Optionally captures a
    screenshot from the resulting frame. Requires that the target was created with enabled
    BeginFrameControl. Designed for use with --run-all-compositor-stages-before-draw, see also
    https://goo.gle/chrome-headless-rendering for more background.

    :param frame_time_ticks: *(Optional)* Timestamp of this BeginFrame in Renderer TimeTicks (milliseconds of uptime). If not set, the current time will be used.
    :param interval: *(Optional)* The interval between BeginFrames that is reported to the compositor, in milliseconds. Defaults to a 60 frames/second interval, i.e. about 16.666 milliseconds.
    :param no_display_updates: *(Optional)* Whether updates should not be committed and drawn onto the display. False by default. If true, only side effects of the BeginFrame will be run, such as layout and animations, but any visual updates may not be visible on the display or in screenshots.
    :param screenshot: *(Optional)* If set, a screenshot of the frame will be captured and returned in the response. Otherwise, no screenshot will be captured. Note that capturing a screenshot can fail, for example, during renderer initialization. In such a case, no screenshot data will be returned.
    :returns: A tuple with the following items:

        0. **hasDamage** - Whether the BeginFrame resulted in damage and, thus, a new frame was committed to the display. Reported for diagnostic uses, may be removed in the future.
        1. **screenshotData** - *(Optional)* Base64-encoded image data of the screenshot, if one was requested and successfully taken.
    '''
    params: T_JSON_DICT = dict()
    if frame_time_ticks is not None:
        params['frameTimeTicks'] = frame_time_ticks
    if interval is not None:
        params['interval'] = interval
    if no_display_updates is not None:
        params['noDisplayUpdates'] = no_display_updates
    if screenshot is not None:
        params['screenshot'] = screenshot.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'HeadlessExperimental.beginFrame',
        'params': params,
    }
    json = yield cmd_dict
    return (
        bool(json['hasDamage']),
        str(json['screenshotData']) if 'screenshotData' in json else None
    )


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables headless events for the target.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'HeadlessExperimental.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables headless events for the target.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'HeadlessExperimental.enable',
    }
    json = yield cmd_dict

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\headless_experimental.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: HeadlessExperimental (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

@dataclass
class ScreenshotParams:
    '''
    Encoding options for a screenshot.
    '''
    #: Image compression format (defaults to png).
    format_: typing.Optional[str] = None

    #: Compression quality from range [0..100] (jpeg and webp only).
    quality: typing.Optional[int] = None

    #: Optimize image encoding for speed, not for resulting size (defaults to false)
    optimize_for_speed: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        if self.format_ is not None:
            json['format'] = self.format_
        if self.quality is not None:
            json['quality'] = self.quality
        if self.optimize_for_speed is not None:
            json['optimizeForSpeed'] = self.optimize_for_speed
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            format_=str(json['format']) if 'format' in json else None,
            quality=int(json['quality']) if 'quality' in json else None,
            optimize_for_speed=bool(json['optimizeForSpeed']) if 'optimizeForSpeed' in json else None,
        )


def begin_frame(
        frame_time_ticks: typing.Optional[float] = None,
        interval: typing.Optional[float] = None,
        no_display_updates: typing.Optional[bool] = None,
        screenshot: typing.Optional[ScreenshotParams] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[bool, typing.Optional[str]]]:
    '''
    Sends a BeginFrame to the target and returns when the frame was completed. Optionally captures a
    screenshot from the resulting frame. Requires that the target was created with enabled
    BeginFrameControl. Designed for use with --run-all-compositor-stages-before-draw, see also
    https://goo.gle/chrome-headless-rendering for more background.

    :param frame_time_ticks: *(Optional)* Timestamp of this BeginFrame in Renderer TimeTicks (milliseconds of uptime). If not set, the current time will be used.
    :param interval: *(Optional)* The interval between BeginFrames that is reported to the compositor, in milliseconds. Defaults to a 60 frames/second interval, i.e. about 16.666 milliseconds.
    :param no_display_updates: *(Optional)* Whether updates should not be committed and drawn onto the display. False by default. If true, only side effects of the BeginFrame will be run, such as layout and animations, but any visual updates may not be visible on the display or in screenshots.
    :param screenshot: *(Optional)* If set, a screenshot of the frame will be captured and returned in the response. Otherwise, no screenshot will be captured. Note that capturing a screenshot can fail, for example, during renderer initialization. In such a case, no screenshot data will be returned.
    :returns: A tuple with the following items:

        0. **hasDamage** - Whether the BeginFrame resulted in damage and, thus, a new frame was committed to the display. Reported for diagnostic uses, may be removed in the future.
        1. **screenshotData** - *(Optional)* Base64-encoded image data of the screenshot, if one was requested and successfully taken.
    '''
    params: T_JSON_DICT = dict()
    if frame_time_ticks is not None:
        params['frameTimeTicks'] = frame_time_ticks
    if interval is not None:
        params['interval'] = interval
    if no_display_updates is not None:
        params['noDisplayUpdates'] = no_display_updates
    if screenshot is not None:
        params['screenshot'] = screenshot.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'HeadlessExperimental.beginFrame',
        'params': params,
    }
    json = yield cmd_dict
    return (
        bool(json['hasDamage']),
        str(json['screenshotData']) if 'screenshotData' in json else None
    )


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables headless events for the target.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'HeadlessExperimental.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables headless events for the target.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'HeadlessExperimental.enable',
    }
    json = yield cmd_dict

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\headless_experimental.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: HeadlessExperimental (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

@dataclass
class ScreenshotParams:
    '''
    Encoding options for a screenshot.
    '''
    #: Image compression format (defaults to png).
    format_: typing.Optional[str] = None

    #: Compression quality from range [0..100] (jpeg and webp only).
    quality: typing.Optional[int] = None

    #: Optimize image encoding for speed, not for resulting size (defaults to false)
    optimize_for_speed: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        if self.format_ is not None:
            json['format'] = self.format_
        if self.quality is not None:
            json['quality'] = self.quality
        if self.optimize_for_speed is not None:
            json['optimizeForSpeed'] = self.optimize_for_speed
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            format_=str(json['format']) if 'format' in json else None,
            quality=int(json['quality']) if 'quality' in json else None,
            optimize_for_speed=bool(json['optimizeForSpeed']) if 'optimizeForSpeed' in json else None,
        )


def begin_frame(
        frame_time_ticks: typing.Optional[float] = None,
        interval: typing.Optional[float] = None,
        no_display_updates: typing.Optional[bool] = None,
        screenshot: typing.Optional[ScreenshotParams] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[bool, typing.Optional[str]]]:
    '''
    Sends a BeginFrame to the target and returns when the frame was completed. Optionally captures a
    screenshot from the resulting frame. Requires that the target was created with enabled
    BeginFrameControl. Designed for use with --run-all-compositor-stages-before-draw, see also
    https://goo.gle/chrome-headless-rendering for more background.

    :param frame_time_ticks: *(Optional)* Timestamp of this BeginFrame in Renderer TimeTicks (milliseconds of uptime). If not set, the current time will be used.
    :param interval: *(Optional)* The interval between BeginFrames that is reported to the compositor, in milliseconds. Defaults to a 60 frames/second interval, i.e. about 16.666 milliseconds.
    :param no_display_updates: *(Optional)* Whether updates should not be committed and drawn onto the display. False by default. If true, only side effects of the BeginFrame will be run, such as layout and animations, but any visual updates may not be visible on the display or in screenshots.
    :param screenshot: *(Optional)* If set, a screenshot of the frame will be captured and returned in the response. Otherwise, no screenshot will be captured. Note that capturing a screenshot can fail, for example, during renderer initialization. In such a case, no screenshot data will be returned.
    :returns: A tuple with the following items:

        0. **hasDamage** - Whether the BeginFrame resulted in damage and, thus, a new frame was committed to the display. Reported for diagnostic uses, may be removed in the future.
        1. **screenshotData** - *(Optional)* Base64-encoded image data of the screenshot, if one was requested and successfully taken.
    '''
    params: T_JSON_DICT = dict()
    if frame_time_ticks is not None:
        params['frameTimeTicks'] = frame_time_ticks
    if interval is not None:
        params['interval'] = interval
    if no_display_updates is not None:
        params['noDisplayUpdates'] = no_display_updates
    if screenshot is not None:
        params['screenshot'] = screenshot.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'HeadlessExperimental.beginFrame',
        'params': params,
    }
    json = yield cmd_dict
    return (
        bool(json['hasDamage']),
        str(json['screenshotData']) if 'screenshotData' in json else None
    )


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables headless events for the target.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'HeadlessExperimental.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables headless events for the target.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'HeadlessExperimental.enable',
    }
    json = yield cmd_dict

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\mysql\__init__.py ===
# dialects/mysql/__init__.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


from . import aiomysql  # noqa
from . import asyncmy  # noqa
from . import base  # noqa
from . import cymysql  # noqa
from . import mariadbconnector  # noqa
from . import mysqlconnector  # noqa
from . import mysqldb  # noqa
from . import pymysql  # noqa
from . import pyodbc  # noqa
from .base import BIGINT
from .base import BINARY
from .base import BIT
from .base import BLOB
from .base import BOOLEAN
from .base import CHAR
from .base import DATE
from .base import DATETIME
from .base import DECIMAL
from .base import DOUBLE
from .base import ENUM
from .base import FLOAT
from .base import INTEGER
from .base import JSON
from .base import LONGBLOB
from .base import LONGTEXT
from .base import MEDIUMBLOB
from .base import MEDIUMINT
from .base import MEDIUMTEXT
from .base import NCHAR
from .base import NUMERIC
from .base import NVARCHAR
from .base import REAL
from .base import SET
from .base import SMALLINT
from .base import TEXT
from .base import TIME
from .base import TIMESTAMP
from .base import TINYBLOB
from .base import TINYINT
from .base import TINYTEXT
from .base import VARBINARY
from .base import VARCHAR
from .base import YEAR
from .dml import Insert
from .dml import insert
from .expression import match
from .mariadb import INET4
from .mariadb import INET6

# default dialect
base.dialect = dialect = mysqldb.dialect

__all__ = (
    "BIGINT",
    "BINARY",
    "BIT",
    "BLOB",
    "BOOLEAN",
    "CHAR",
    "DATE",
    "DATETIME",
    "DECIMAL",
    "DOUBLE",
    "ENUM",
    "FLOAT",
    "INET4",
    "INET6",
    "INTEGER",
    "INTEGER",
    "JSON",
    "LONGBLOB",
    "LONGTEXT",
    "MEDIUMBLOB",
    "MEDIUMINT",
    "MEDIUMTEXT",
    "NCHAR",
    "NVARCHAR",
    "NUMERIC",
    "SET",
    "SMALLINT",
    "REAL",
    "TEXT",
    "TIME",
    "TIMESTAMP",
    "TINYBLOB",
    "TINYINT",
    "TINYTEXT",
    "VARBINARY",
    "VARCHAR",
    "YEAR",
    "dialect",
    "insert",
    "Insert",
    "match",
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_local.py ===
from __future__ import annotations

from typing import Generic, TypeVar, cast

# Runvar implementations
import attrs

from .._util import NoPublicConstructor, final
from . import _run

T = TypeVar("T")


@final
class _NoValue: ...


@final
@attrs.define(eq=False)
class RunVarToken(Generic[T], metaclass=NoPublicConstructor):
    _var: RunVar[T]
    previous_value: T | type[_NoValue] = _NoValue
    redeemed: bool = attrs.field(default=False, init=False)

    @classmethod
    def _empty(cls, var: RunVar[T]) -> RunVarToken[T]:
        return cls._create(var)


@final
@attrs.define(eq=False, repr=False)
class RunVar(Generic[T]):
    """The run-local variant of a context variable.

    :class:`RunVar` objects are similar to context variable objects,
    except that they are shared across a single call to :func:`trio.run`
    rather than a single task.

    """

    _name: str = attrs.field(alias="name")
    _default: T | type[_NoValue] = attrs.field(default=_NoValue, alias="default")

    def get(self, default: T | type[_NoValue] = _NoValue) -> T:
        """Gets the value of this :class:`RunVar` for the current run call."""
        try:
            return cast("T", _run.GLOBAL_RUN_CONTEXT.runner._locals[self])
        except AttributeError:
            raise RuntimeError("Cannot be used outside of a run context") from None
        except KeyError:
            # contextvars consistency
            # `type: ignore` awaiting https://github.com/python/mypy/issues/15553 to be fixed & released
            if default is not _NoValue:
                return default  # type: ignore[return-value]

            if self._default is not _NoValue:
                return self._default  # type: ignore[return-value]

            raise LookupError(self) from None

    def set(self, value: T) -> RunVarToken[T]:
        """Sets the value of this :class:`RunVar` for this current run
        call.

        """
        try:
            old_value = self.get()
        except LookupError:
            token = RunVarToken._empty(self)
        else:
            token = RunVarToken[T]._create(self, old_value)

        # This can't fail, because if we weren't in Trio context then the
        # get() above would have failed.
        _run.GLOBAL_RUN_CONTEXT.runner._locals[self] = value
        return token

    def reset(self, token: RunVarToken[T]) -> None:
        """Resets the value of this :class:`RunVar` to what it was
        previously specified by the token.

        """
        if token is None:
            raise TypeError("token must not be none")

        if token.redeemed:
            raise ValueError("token has already been used")

        if token._var is not self:
            raise ValueError("token is not for us")

        previous = token.previous_value
        try:
            if previous is _NoValue:
                _run.GLOBAL_RUN_CONTEXT.runner._locals.pop(self)
            else:
                _run.GLOBAL_RUN_CONTEXT.runner._locals[self] = previous
        except AttributeError:
            raise RuntimeError("Cannot be used outside of a run context") from None

        token.redeemed = True

    def __repr__(self) -> str:
        return f"<RunVar name={self._name!r}>"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\styles\borders.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.compat import safe_string
from openpyxl.descriptors import (
    NoneSet,
    Typed,
    Bool,
    Alias,
    Sequence,
    Integer,
)
from openpyxl.descriptors.serialisable import Serialisable

from .colors import ColorDescriptor


BORDER_NONE = None
BORDER_DASHDOT = 'dashDot'
BORDER_DASHDOTDOT = 'dashDotDot'
BORDER_DASHED = 'dashed'
BORDER_DOTTED = 'dotted'
BORDER_DOUBLE = 'double'
BORDER_HAIR = 'hair'
BORDER_MEDIUM = 'medium'
BORDER_MEDIUMDASHDOT = 'mediumDashDot'
BORDER_MEDIUMDASHDOTDOT = 'mediumDashDotDot'
BORDER_MEDIUMDASHED = 'mediumDashed'
BORDER_SLANTDASHDOT = 'slantDashDot'
BORDER_THICK = 'thick'
BORDER_THIN = 'thin'


class Side(Serialisable):

    """Border options for use in styles.
    Caution: if you do not specify a border_style, other attributes will
    have no effect !"""


    color = ColorDescriptor(allow_none=True)
    style = NoneSet(values=('dashDot','dashDotDot', 'dashed','dotted',
                            'double','hair', 'medium', 'mediumDashDot', 'mediumDashDotDot',
                            'mediumDashed', 'slantDashDot', 'thick', 'thin')
                    )
    border_style = Alias('style')

    def __init__(self, style=None, color=None, border_style=None):
        if border_style is not None:
            style = border_style
        self.style = style
        self.color = color


class Border(Serialisable):
    """Border positioning for use in styles."""

    tagname = "border"

    __elements__ = ('start', 'end', 'left', 'right', 'top', 'bottom',
                    'diagonal', 'vertical', 'horizontal')

    # child elements
    start = Typed(expected_type=Side, allow_none=True)
    end = Typed(expected_type=Side, allow_none=True)
    left = Typed(expected_type=Side, allow_none=True)
    right = Typed(expected_type=Side, allow_none=True)
    top = Typed(expected_type=Side, allow_none=True)
    bottom = Typed(expected_type=Side, allow_none=True)
    diagonal = Typed(expected_type=Side, allow_none=True)
    vertical = Typed(expected_type=Side, allow_none=True)
    horizontal = Typed(expected_type=Side, allow_none=True)
    # attributes
    outline = Bool()
    diagonalUp = Bool()
    diagonalDown = Bool()

    def __init__(self, left=None, right=None, top=None,
                 bottom=None, diagonal=None, diagonal_direction=None,
                 vertical=None, horizontal=None, diagonalUp=False, diagonalDown=False,
                 outline=True, start=None, end=None):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.diagonal = diagonal
        self.vertical = vertical
        self.horizontal = horizontal
        self.diagonal_direction = diagonal_direction
        self.diagonalUp = diagonalUp
        self.diagonalDown = diagonalDown
        self.outline = outline
        self.start = start
        self.end = end

    def __iter__(self):
        for attr in self.__attrs__:
            value = getattr(self, attr)
            if value and attr != "outline":
                yield attr, safe_string(value)
            elif attr == "outline" and not value:
                yield attr, safe_string(value)

DEFAULT_BORDER = Border(left=Side(), right=Side(), top=Side(), bottom=Side(), diagonal=Side())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\util\_exceptions.py ===
from __future__ import annotations

import contextlib
import inspect
import os
import re
from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import FrameType


@contextlib.contextmanager
def rewrite_exception(old_name: str, new_name: str) -> Generator[None, None, None]:
    """
    Rewrite the message of an exception.
    """
    try:
        yield
    except Exception as err:
        if not err.args:
            raise
        msg = str(err.args[0])
        msg = msg.replace(old_name, new_name)
        args: tuple[str, ...] = (msg,)
        if len(err.args) > 1:
            args = args + err.args[1:]
        err.args = args
        raise


def find_stack_level() -> int:
    """
    Find the first place in the stack that is not inside pandas
    (tests notwithstanding).
    """

    import pandas as pd

    pkg_dir = os.path.dirname(pd.__file__)
    test_dir = os.path.join(pkg_dir, "tests")

    # https://stackoverflow.com/questions/17407119/python-inspect-stack-is-slow
    frame: FrameType | None = inspect.currentframe()
    try:
        n = 0
        while frame:
            filename = inspect.getfile(frame)
            if filename.startswith(pkg_dir) and not filename.startswith(test_dir):
                frame = frame.f_back
                n += 1
            else:
                break
    finally:
        # See note in
        # https://docs.python.org/3/library/inspect.html#inspect.Traceback
        del frame
    return n


@contextlib.contextmanager
def rewrite_warning(
    target_message: str,
    target_category: type[Warning],
    new_message: str,
    new_category: type[Warning] | None = None,
) -> Generator[None, None, None]:
    """
    Rewrite the message of a warning.

    Parameters
    ----------
    target_message : str
        Warning message to match.
    target_category : Warning
        Warning type to match.
    new_message : str
        New warning message to emit.
    new_category : Warning or None, default None
        New warning type to emit. When None, will be the same as target_category.
    """
    if new_category is None:
        new_category = target_category
    with warnings.catch_warnings(record=True) as record:
        yield
    if len(record) > 0:
        match = re.compile(target_message)
        for warning in record:
            if warning.category is target_category and re.search(
                match, str(warning.message)
            ):
                category = new_category
                message: Warning | str = new_message
            else:
                category, message = warning.category, warning.message
            warnings.warn_explicit(
                message=message,
                category=category,
                filename=warning.filename,
                lineno=warning.lineno,
            )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\req\__init__.py ===
import collections
import logging
from dataclasses import dataclass
from typing import Generator, List, Optional, Sequence, Tuple

from pip._internal.cli.progress_bars import get_install_progress_renderer
from pip._internal.utils.logging import indent_log

from .req_file import parse_requirements
from .req_install import InstallRequirement
from .req_set import RequirementSet

__all__ = [
    "RequirementSet",
    "InstallRequirement",
    "parse_requirements",
    "install_given_reqs",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InstallationResult:
    name: str


def _validate_requirements(
    requirements: List[InstallRequirement],
) -> Generator[Tuple[str, InstallRequirement], None, None]:
    for req in requirements:
        assert req.name, f"invalid to-be-installed requirement: {req}"
        yield req.name, req


def install_given_reqs(
    requirements: List[InstallRequirement],
    global_options: Sequence[str],
    root: Optional[str],
    home: Optional[str],
    prefix: Optional[str],
    warn_script_location: bool,
    use_user_site: bool,
    pycompile: bool,
    progress_bar: str,
) -> List[InstallationResult]:
    """
    Install everything in the given list.

    (to be called after having downloaded and unpacked the packages)
    """
    to_install = collections.OrderedDict(_validate_requirements(requirements))

    if to_install:
        logger.info(
            "Installing collected packages: %s",
            ", ".join(to_install.keys()),
        )

    installed = []

    show_progress = logger.isEnabledFor(logging.INFO) and len(to_install) > 1

    items = iter(to_install.values())
    if show_progress:
        renderer = get_install_progress_renderer(
            bar_type=progress_bar, total=len(to_install)
        )
        items = renderer(items)

    with indent_log():
        for requirement in items:
            req_name = requirement.name
            assert req_name is not None
            if requirement.should_reinstall:
                logger.info("Attempting uninstall: %s", req_name)
                with indent_log():
                    uninstalled_pathset = requirement.uninstall(auto_confirm=True)
            else:
                uninstalled_pathset = None

            try:
                requirement.install(
                    global_options,
                    root=root,
                    home=home,
                    prefix=prefix,
                    warn_script_location=warn_script_location,
                    use_user_site=use_user_site,
                    pycompile=pycompile,
                )
            except Exception:
                # if install did not succeed, rollback previous uninstall
                if uninstalled_pathset and not requirement.install_succeeded:
                    uninstalled_pathset.rollback()
                raise
            else:
                if uninstalled_pathset and requirement.install_succeeded:
                    uninstalled_pathset.commit()

            installed.append(InstallationResult(req_name))

    return installed

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\__init__.py ===
"""Core module. Provides the basic operations needed in sympy.
"""

from .sympify import sympify, SympifyError
from .cache import cacheit
from .assumptions import assumptions, check_assumptions, failing_assumptions, common_assumptions
from .basic import Basic, Atom
from .singleton import S
from .expr import Expr, AtomicExpr, UnevaluatedExpr
from .symbol import Symbol, Wild, Dummy, symbols, var
from .numbers import Number, Float, Rational, Integer, NumberSymbol, \
    RealNumber, igcd, ilcm, seterr, E, I, nan, oo, pi, zoo, \
    AlgebraicNumber, comp, mod_inverse
from .power import Pow
from .intfunc import integer_nthroot, integer_log, num_digits, trailing
from .mul import Mul, prod
from .add import Add
from .mod import Mod
from .relational import ( Rel, Eq, Ne, Lt, Le, Gt, Ge,
    Equality, GreaterThan, LessThan, Unequality, StrictGreaterThan,
    StrictLessThan )
from .multidimensional import vectorize
from .function import Lambda, WildFunction, Derivative, diff, FunctionClass, \
    Function, Subs, expand, PoleError, count_ops, \
    expand_mul, expand_log, expand_func, \
    expand_trig, expand_complex, expand_multinomial, nfloat, \
    expand_power_base, expand_power_exp, arity
from .evalf import PrecisionExhausted, N
from .containers import Tuple, Dict
from .exprtools import gcd_terms, factor_terms, factor_nc
from .parameters import evaluate
from .kind import UndefinedKind, NumberKind, BooleanKind
from .traversal import preorder_traversal, bottom_up, use, postorder_traversal
from .sorting import default_sort_key, ordered

# expose singletons
Catalan = S.Catalan
EulerGamma = S.EulerGamma
GoldenRatio = S.GoldenRatio
TribonacciConstant = S.TribonacciConstant

__all__ = [
    'sympify', 'SympifyError',

    'cacheit',

    'assumptions', 'check_assumptions', 'failing_assumptions',
    'common_assumptions',

    'Basic', 'Atom',

    'S',

    'Expr', 'AtomicExpr', 'UnevaluatedExpr',

    'Symbol', 'Wild', 'Dummy', 'symbols', 'var',

    'Number', 'Float', 'Rational', 'Integer', 'NumberSymbol', 'RealNumber',
    'igcd', 'ilcm', 'seterr', 'E', 'I', 'nan', 'oo', 'pi', 'zoo',
    'AlgebraicNumber', 'comp', 'mod_inverse',

    'Pow',

    'integer_nthroot', 'integer_log', 'num_digits', 'trailing',

    'Mul', 'prod',

    'Add',

    'Mod',

    'Rel', 'Eq', 'Ne', 'Lt', 'Le', 'Gt', 'Ge', 'Equality', 'GreaterThan',
    'LessThan', 'Unequality', 'StrictGreaterThan', 'StrictLessThan',

    'vectorize',

    'Lambda', 'WildFunction', 'Derivative', 'diff', 'FunctionClass',
    'Function', 'Subs', 'expand', 'PoleError', 'count_ops', 'expand_mul',
    'expand_log', 'expand_func', 'expand_trig', 'expand_complex',
    'expand_multinomial', 'nfloat', 'expand_power_base', 'expand_power_exp',
    'arity',

    'PrecisionExhausted', 'N',

    'evalf', # The module?

    'Tuple', 'Dict',

    'gcd_terms', 'factor_terms', 'factor_nc',

    'evaluate',

    'Catalan',
    'EulerGamma',
    'GoldenRatio',
    'TribonacciConstant',

    'UndefinedKind', 'NumberKind', 'BooleanKind',

    'preorder_traversal', 'bottom_up', 'use', 'postorder_traversal',

    'default_sort_key', 'ordered',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\transpose.py ===
from sympy.core.basic import Basic
from sympy.matrices.expressions.matexpr import MatrixExpr


class Transpose(MatrixExpr):
    """
    The transpose of a matrix expression.

    This is a symbolic object that simply stores its argument without
    evaluating it. To actually compute the transpose, use the ``transpose()``
    function, or the ``.T`` attribute of matrices.

    Examples
    ========

    >>> from sympy import MatrixSymbol, Transpose, transpose
    >>> A = MatrixSymbol('A', 3, 5)
    >>> B = MatrixSymbol('B', 5, 3)
    >>> Transpose(A)
    A.T
    >>> A.T == transpose(A) == Transpose(A)
    True
    >>> Transpose(A*B)
    (A*B).T
    >>> transpose(A*B)
    B.T*A.T

    """
    is_Transpose = True

    def doit(self, **hints):
        arg = self.arg
        if hints.get('deep', True) and isinstance(arg, Basic):
            arg = arg.doit(**hints)
        _eval_transpose = getattr(arg, '_eval_transpose', None)
        if _eval_transpose is not None:
            result = _eval_transpose()
            return result if result is not None else Transpose(arg)
        else:
            return Transpose(arg)

    @property
    def arg(self):
        return self.args[0]

    @property
    def shape(self):
        return self.arg.shape[::-1]

    def _entry(self, i, j, expand=False, **kwargs):
        return self.arg._entry(j, i, expand=expand, **kwargs)

    def _eval_adjoint(self):
        return self.arg.conjugate()

    def _eval_conjugate(self):
        return self.arg.adjoint()

    def _eval_transpose(self):
        return self.arg

    def _eval_trace(self):
        from .trace import Trace
        return Trace(self.arg)  # Trace(X.T) => Trace(X)

    def _eval_determinant(self):
        from sympy.matrices.expressions.determinant import det
        return det(self.arg)

    def _eval_derivative(self, x):
        # x is a scalar:
        return self.arg._eval_derivative(x)

    def _eval_derivative_matrix_lines(self, x):
        lines = self.args[0]._eval_derivative_matrix_lines(x)
        return [i.transpose() for i in lines]


def transpose(expr):
    """Matrix transpose"""
    return Transpose(expr).doit(deep=False)


from sympy.assumptions.ask import ask, Q
from sympy.assumptions.refine import handlers_dict


def refine_Transpose(expr, assumptions):
    """
    >>> from sympy import MatrixSymbol, Q, assuming, refine
    >>> X = MatrixSymbol('X', 2, 2)
    >>> X.T
    X.T
    >>> with assuming(Q.symmetric(X)):
    ...     print(refine(X.T))
    X
    """
    if ask(Q.symmetric(expr), assumptions):
        return expr.arg

    return expr

handlers_dict['Transpose'] = refine_Transpose

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\kind.py ===
"""Kinds for Operators, Bras, and Kets.

This module defines kinds for operators, bras, and kets. These are useful
in various places in ``sympy.physics.quantum`` as you often want to know
what the kind is of a compound expression. For example, if you multiply
an operator, bra, or ket by a number, you get back another operator, bra,
or ket - even though if you did an ``isinstance`` check you would find that
you have a ``Mul`` instead. The kind system is meant to give you a quick
way of determining how a compound expression behaves in terms of lower
level kinds.

The resolution calculation of kinds for compound expressions can be found
either in container classes or in functions that are registered with
kind dispatchers.
"""

from sympy.core.mul import Mul
from sympy.core.kind import Kind, _NumberKind


__all__ = [
    '_KetKind',
    'KetKind',
    '_BraKind',
    'BraKind',
    '_OperatorKind',
    'OperatorKind',
]


class _KetKind(Kind):
    """A kind for quantum kets."""

    def __new__(cls):
        obj = super().__new__(cls)
        return obj

    def __repr__(self):
        return "KetKind"

# Create an instance as many situations need this.
KetKind = _KetKind()


class _BraKind(Kind):
    """A kind for quantum bras."""

    def __new__(cls):
        obj = super().__new__(cls)
        return obj

    def __repr__(self):
        return "BraKind"

# Create an instance as many situations need this.
BraKind = _BraKind()


from sympy.core.kind import Kind

class _OperatorKind(Kind):
    """A kind for quantum operators."""

    def __new__(cls):
        obj = super().__new__(cls)
        return obj

    def __repr__(self):
        return "OperatorKind"

# Create an instance as many situations need this.
OperatorKind = _OperatorKind()


#-----------------------------------------------------------------------------
# Kind resolution.
#-----------------------------------------------------------------------------

# Note: We can't currently add kind dispatchers for the following combinations
#       as the Mul._kind_dispatcher is set to commutative and will also
#       register the opposite order, which isn't correct for these pairs:
#
# 1. (_OperatorKind, _KetKind)
# 2. (_BraKind, _OperatorKind)
# 3. (_BraKind, _KetKind)


@Mul._kind_dispatcher.register(_NumberKind, _KetKind)
def _mul_number_ket_kind(lhs, rhs):
    """Perform the kind calculation of NumberKind*KetKind -> KetKind."""
    return KetKind


@Mul._kind_dispatcher.register(_NumberKind, _BraKind)
def _mul_number_bra_kind(lhs, rhs):
    """Perform the kind calculation of NumberKind*BraKind -> BraKind."""
    return BraKind


@Mul._kind_dispatcher.register(_NumberKind, _OperatorKind)
def _mul_operator_kind(lhs, rhs):
    """Perform the kind calculation of NumberKind*OperatorKind -> OperatorKind."""
    return OperatorKind

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\matrixcache.py ===
"""A cache for storing small matrices in multiple formats."""

from sympy.core.numbers import (I, Rational, pi)
from sympy.core.power import Pow
from sympy.functions.elementary.exponential import exp
from sympy.matrices.dense import Matrix

from sympy.physics.quantum.matrixutils import (
    to_sympy, to_numpy, to_scipy_sparse
)


class MatrixCache:
    """A cache for small matrices in different formats.

    This class takes small matrices in the standard ``sympy.Matrix`` format,
    and then converts these to both ``numpy.matrix`` and
    ``scipy.sparse.csr_matrix`` matrices. These matrices are then stored for
    future recovery.
    """

    def __init__(self, dtype='complex'):
        self._cache = {}
        self.dtype = dtype

    def cache_matrix(self, name, m):
        """Cache a matrix by its name.

        Parameters
        ----------
        name : str
            A descriptive name for the matrix, like "identity2".
        m : list of lists
            The raw matrix data as a SymPy Matrix.
        """
        try:
            self._sympy_matrix(name, m)
        except ImportError:
            pass
        try:
            self._numpy_matrix(name, m)
        except ImportError:
            pass
        try:
            self._scipy_sparse_matrix(name, m)
        except ImportError:
            pass

    def get_matrix(self, name, format):
        """Get a cached matrix by name and format.

        Parameters
        ----------
        name : str
            A descriptive name for the matrix, like "identity2".
        format : str
            The format desired ('sympy', 'numpy', 'scipy.sparse')
        """
        m = self._cache.get((name, format))
        if m is not None:
            return m
        raise NotImplementedError(
            'Matrix with name %s and format %s is not available.' %
            (name, format)
        )

    def _store_matrix(self, name, format, m):
        self._cache[(name, format)] = m

    def _sympy_matrix(self, name, m):
        self._store_matrix(name, 'sympy', to_sympy(m))

    def _numpy_matrix(self, name, m):
        m = to_numpy(m, dtype=self.dtype)
        self._store_matrix(name, 'numpy', m)

    def _scipy_sparse_matrix(self, name, m):
        # TODO: explore different sparse formats. But sparse.kron will use
        # coo in most cases, so we use that here.
        m = to_scipy_sparse(m, dtype=self.dtype)
        self._store_matrix(name, 'scipy.sparse', m)


sqrt2_inv = Pow(2, Rational(-1, 2), evaluate=False)

# Save the common matrices that we will need
matrix_cache = MatrixCache()
matrix_cache.cache_matrix('eye2', Matrix([[1, 0], [0, 1]]))
matrix_cache.cache_matrix('op11', Matrix([[0, 0], [0, 1]]))  # |1><1|
matrix_cache.cache_matrix('op00', Matrix([[1, 0], [0, 0]]))  # |0><0|
matrix_cache.cache_matrix('op10', Matrix([[0, 0], [1, 0]]))  # |1><0|
matrix_cache.cache_matrix('op01', Matrix([[0, 1], [0, 0]]))  # |0><1|
matrix_cache.cache_matrix('X', Matrix([[0, 1], [1, 0]]))
matrix_cache.cache_matrix('Y', Matrix([[0, -I], [I, 0]]))
matrix_cache.cache_matrix('Z', Matrix([[1, 0], [0, -1]]))
matrix_cache.cache_matrix('S', Matrix([[1, 0], [0, I]]))
matrix_cache.cache_matrix('T', Matrix([[1, 0], [0, exp(I*pi/4)]]))
matrix_cache.cache_matrix('H', sqrt2_inv*Matrix([[1, 1], [1, -1]]))
matrix_cache.cache_matrix('Hsqrt2', Matrix([[1, 1], [1, -1]]))
matrix_cache.cache_matrix(
    'SWAP', Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))
matrix_cache.cache_matrix('ZX', sqrt2_inv*Matrix([[1, 1], [1, -1]]))
matrix_cache.cache_matrix('ZY', Matrix([[I, 0], [0, -I]]))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\approximants.py ===
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.polys.polytools import lcm
from sympy.utilities import public

@public
def approximants(l, X=Symbol('x'), simplify=False):
    """
    Return a generator for consecutive Pade approximants for a series.
    It can also be used for computing the rational generating function of a
    series when possible, since the last approximant returned by the generator
    will be the generating function (if any).

    Explanation
    ===========

    The input list can contain more complex expressions than integer or rational
    numbers; symbols may also be involved in the computation. An example below
    show how to compute the generating function of the whole Pascal triangle.

    The generator can be asked to apply the sympy.simplify function on each
    generated term, which will make the computation slower; however it may be
    useful when symbols are involved in the expressions.

    Examples
    ========

    >>> from sympy.series import approximants
    >>> from sympy import lucas, fibonacci, symbols, binomial
    >>> g = [lucas(k) for k in range(16)]
    >>> [e for e in approximants(g)]
    [2, -4/(x - 2), (5*x - 2)/(3*x - 1), (x - 2)/(x**2 + x - 1)]

    >>> h = [fibonacci(k) for k in range(16)]
    >>> [e for e in approximants(h)]
    [x, -x/(x - 1), (x**2 - x)/(2*x - 1), -x/(x**2 + x - 1)]

    >>> x, t = symbols("x,t")
    >>> p=[sum(binomial(k,i)*x**i for i in range(k+1)) for k in range(16)]
    >>> y = approximants(p, t)
    >>> for k in range(3): print(next(y))
    1
    (x + 1)/((-x - 1)*(t*(x + 1) + (x + 1)/(-x - 1)))
    nan

    >>> y = approximants(p, t, simplify=True)
    >>> for k in range(3): print(next(y))
    1
    -1/(t*(x + 1) - 1)
    nan

    See Also
    ========

    sympy.concrete.guess.guess_generating_function_rational
    mpmath.pade
    """
    from sympy.simplify import simplify as simp
    from sympy.simplify.radsimp import denom
    p1, q1 = [S.One], [S.Zero]
    p2, q2 = [S.Zero], [S.One]
    while len(l):
        b = 0
        while l[b]==0:
            b += 1
            if b == len(l):
                return
        m = [S.One/l[b]]
        for k in range(b+1, len(l)):
            s = 0
            for j in range(b, k):
                s -= l[j+1] * m[b-j-1]
            m.append(s/l[b])
        l = m
        a, l[0] = l[0], 0
        p = [0] * max(len(p2), b+len(p1))
        q = [0] * max(len(q2), b+len(q1))
        for k in range(len(p2)):
            p[k] = a*p2[k]
        for k in range(b, b+len(p1)):
            p[k] += p1[k-b]
        for k in range(len(q2)):
            q[k] = a*q2[k]
        for k in range(b, b+len(q1)):
            q[k] += q1[k-b]
        while p[-1]==0: p.pop()
        while q[-1]==0: q.pop()
        p1, p2 = p2, p
        q1, q2 = q2, q

        # yield result
        c = 1
        for x in p:
            c = lcm(c, denom(x))
        for x in q:
            c = lcm(c, denom(x))
        out = ( sum(c*e*X**k for k, e in enumerate(p))
              / sum(c*e*X**k for k, e in enumerate(q)) )
        if simplify:
            yield(simp(out))
        else:
            yield out
    return

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\colorama\ansi.py ===
# Copyright Jonathan Hartley 2013. BSD 3-Clause license, see LICENSE file.
'''
This module generates ANSI character codes to printing colors to terminals.
See: http://en.wikipedia.org/wiki/ANSI_escape_code
'''

CSI = '\033['
OSC = '\033]'
BEL = '\a'


def code_to_chars(code):
    return CSI + str(code) + 'm'

def set_title(title):
    return OSC + '2;' + title + BEL

def clear_screen(mode=2):
    return CSI + str(mode) + 'J'

def clear_line(mode=2):
    return CSI + str(mode) + 'K'


class AnsiCodes(object):
    def __init__(self):
        # the subclasses declare class attributes which are numbers.
        # Upon instantiation we define instance attributes, which are the same
        # as the class attributes but wrapped with the ANSI escape sequence
        for name in dir(self):
            if not name.startswith('_'):
                value = getattr(self, name)
                setattr(self, name, code_to_chars(value))


class AnsiCursor(object):
    def UP(self, n=1):
        return CSI + str(n) + 'A'
    def DOWN(self, n=1):
        return CSI + str(n) + 'B'
    def FORWARD(self, n=1):
        return CSI + str(n) + 'C'
    def BACK(self, n=1):
        return CSI + str(n) + 'D'
    def POS(self, x=1, y=1):
        return CSI + str(y) + ';' + str(x) + 'H'


class AnsiFore(AnsiCodes):
    BLACK           = 30
    RED             = 31
    GREEN           = 32
    YELLOW          = 33
    BLUE            = 34
    MAGENTA         = 35
    CYAN            = 36
    WHITE           = 37
    RESET           = 39

    # These are fairly well supported, but not part of the standard.
    LIGHTBLACK_EX   = 90
    LIGHTRED_EX     = 91
    LIGHTGREEN_EX   = 92
    LIGHTYELLOW_EX  = 93
    LIGHTBLUE_EX    = 94
    LIGHTMAGENTA_EX = 95
    LIGHTCYAN_EX    = 96
    LIGHTWHITE_EX   = 97


class AnsiBack(AnsiCodes):
    BLACK           = 40
    RED             = 41
    GREEN           = 42
    YELLOW          = 43
    BLUE            = 44
    MAGENTA         = 45
    CYAN            = 46
    WHITE           = 47
    RESET           = 49

    # These are fairly well supported, but not part of the standard.
    LIGHTBLACK_EX   = 100
    LIGHTRED_EX     = 101
    LIGHTGREEN_EX   = 102
    LIGHTYELLOW_EX  = 103
    LIGHTBLUE_EX    = 104
    LIGHTMAGENTA_EX = 105
    LIGHTCYAN_EX    = 106
    LIGHTWHITE_EX   = 107


class AnsiStyle(AnsiCodes):
    BRIGHT    = 1
    DIM       = 2
    NORMAL    = 22
    RESET_ALL = 0

Fore   = AnsiFore()
Back   = AnsiBack()
Style  = AnsiStyle()
Cursor = AnsiCursor()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\stable_diffusion\demo_txt2img.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Modified from TensorRT demo diffusion, which has the following license:
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------

import coloredlogs
from cuda import cudart
from demo_utils import (
    add_controlnet_arguments,
    arg_parser,
    get_metadata,
    load_pipelines,
    parse_arguments,
    process_controlnet_arguments,
    repeat_prompt,
)


def main(args):
    controlnet_images, controlnet_scale = process_controlnet_arguments(args)

    pipeline, refiner = load_pipelines(args)
    assert refiner is None

    prompt, negative_prompt = repeat_prompt(args)
    batch_size = len(prompt)
    pipeline.load_resources(args.height, args.width, batch_size)

    def run_inference(warmup=False):
        return pipeline.run(
            prompt,
            negative_prompt,
            args.height,
            args.width,
            denoising_steps=args.denoising_steps,
            guidance=args.guidance,
            seed=args.seed,
            controlnet_images=controlnet_images,
            controlnet_scales=controlnet_scale,
            show_latency=not warmup,
            output_type="pil",
            deterministic=args.deterministic,
        )

    if not args.disable_cuda_graph:
        # inference once to get cuda graph
        _, _ = run_inference(warmup=True)

    print("[I] Warming up ..")
    for _ in range(args.num_warmup_runs):
        _, _ = run_inference(warmup=True)

    print("[I] Running StableDiffusion pipeline")
    if args.nvtx_profile:
        cudart.cudaProfilerStart()
    images, perf_data = run_inference(warmup=False)
    if args.nvtx_profile:
        cudart.cudaProfilerStop()

    metadata = get_metadata(args, False)
    metadata.update(pipeline.metadata())
    if perf_data:
        metadata.update(perf_data)
    metadata["images"] = len(images)
    print(metadata)
    pipeline.save_images(images, prompt, negative_prompt, metadata)

    pipeline.teardown()


if __name__ == "__main__":
    coloredlogs.install(fmt="%(funcName)20s: %(message)s")

    parser = arg_parser("Options for Stable Diffusion Demo")
    add_controlnet_arguments(parser)
    args = parse_arguments(is_xl=False, parser=parser)

    if args.user_compute_stream:
        import torch

        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            main(args)
    else:
        main(args)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\__init__.py ===
"""
Python HTTP library with thread-safe connection pooling, file post support, user friendly, and more
"""
from __future__ import absolute_import

# Set default logging handler to avoid "No handler found" warnings.
import logging
import warnings
from logging import NullHandler

from . import exceptions
from ._version import __version__
from .connectionpool import HTTPConnectionPool, HTTPSConnectionPool, connection_from_url
from .filepost import encode_multipart_formdata
from .poolmanager import PoolManager, ProxyManager, proxy_from_url
from .response import HTTPResponse
from .util.request import make_headers
from .util.retry import Retry
from .util.timeout import Timeout
from .util.url import get_host

# === NOTE TO REPACKAGERS AND VENDORS ===
# Please delete this block, this logic is only
# for urllib3 being distributed via PyPI.
# See: https://github.com/urllib3/urllib3/issues/2680
try:
    import urllib3_secure_extra  # type: ignore # noqa: F401
except ImportError:
    pass
else:
    warnings.warn(
        "'urllib3[secure]' extra is deprecated and will be removed "
        "in a future release of urllib3 2.x. Read more in this issue: "
        "https://github.com/urllib3/urllib3/issues/2680",
        category=DeprecationWarning,
        stacklevel=2,
    )

__author__ = "Andrey Petrov (andrey.petrov@shazow.net)"
__license__ = "MIT"
__version__ = __version__

__all__ = (
    "HTTPConnectionPool",
    "HTTPSConnectionPool",
    "PoolManager",
    "ProxyManager",
    "HTTPResponse",
    "Retry",
    "Timeout",
    "add_stderr_logger",
    "connection_from_url",
    "disable_warnings",
    "encode_multipart_formdata",
    "get_host",
    "make_headers",
    "proxy_from_url",
)

logging.getLogger(__name__).addHandler(NullHandler())


def add_stderr_logger(level=logging.DEBUG):
    """
    Helper for quickly adding a StreamHandler to the logger. Useful for
    debugging.

    Returns the handler after adding it.
    """
    # This method needs to be in this __init__.py to get the __name__ correct
    # even if urllib3 is vendored within another package.
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.debug("Added a stderr logging handler to logger: %s", __name__)
    return handler


# ... Clean up.
del NullHandler


# All warning filters *must* be appended unless you're really certain that they
# shouldn't be: otherwise, it's very hard for users to use most Python
# mechanisms to silence them.
# SecurityWarning's always go off by default.
warnings.simplefilter("always", exceptions.SecurityWarning, append=True)
# SubjectAltNameWarning's should go off once per host
warnings.simplefilter("default", exceptions.SubjectAltNameWarning, append=True)
# InsecurePlatformWarning's don't vary between requests, so we keep it default.
warnings.simplefilter("default", exceptions.InsecurePlatformWarning, append=True)
# SNIMissingWarnings should go off only once.
warnings.simplefilter("default", exceptions.SNIMissingWarning, append=True)


def disable_warnings(category=exceptions.HTTPWarning):
    """
    Helper for quickly disabling all urllib3 warnings.
    """
    warnings.simplefilter("ignore", category)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\_shape.py ===
from sympy.core.relational import Eq
from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.logic.boolalg import Boolean, And
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.exceptions import ShapeError
from typing import Union


def is_matadd_valid(*args: MatrixExpr) -> Boolean:
    """Return the symbolic condition how ``MatAdd``, ``HadamardProduct``
    makes sense.

    Parameters
    ==========

    args
        The list of arguments of matrices to be tested for.

    Examples
    ========

    >>> from sympy import MatrixSymbol, symbols
    >>> from sympy.matrices.expressions._shape import is_matadd_valid

    >>> m, n, p, q = symbols('m n p q')
    >>> A = MatrixSymbol('A', m, n)
    >>> B = MatrixSymbol('B', p, q)
    >>> is_matadd_valid(A, B)
    Eq(m, p) & Eq(n, q)
    """
    rows, cols = zip(*(arg.shape for arg in args))
    return And(
        *(Eq(i, j) for i, j in zip(rows[:-1], rows[1:])),
        *(Eq(i, j) for i, j in zip(cols[:-1], cols[1:])),
    )


def is_matmul_valid(*args: Union[MatrixExpr, Expr]) -> Boolean:
    """Return the symbolic condition how ``MatMul`` makes sense

    Parameters
    ==========

    args
        The list of arguments of matrices and scalar expressions to be tested
        for.

    Examples
    ========

    >>> from sympy import MatrixSymbol, symbols
    >>> from sympy.matrices.expressions._shape import is_matmul_valid

    >>> m, n, p, q = symbols('m n p q')
    >>> A = MatrixSymbol('A', m, n)
    >>> B = MatrixSymbol('B', p, q)
    >>> is_matmul_valid(A, B)
    Eq(n, p)
    """
    rows, cols = zip(*(arg.shape for arg in args if isinstance(arg, MatrixExpr)))
    return And(*(Eq(i, j) for i, j in zip(cols[:-1], rows[1:])))


def is_square(arg: MatrixExpr, /) -> Boolean:
    """Return the symbolic condition how the matrix is assumed to be square

    Parameters
    ==========

    arg
        The matrix to be tested for.

    Examples
    ========

    >>> from sympy import MatrixSymbol, symbols
    >>> from sympy.matrices.expressions._shape import is_square

    >>> m, n = symbols('m n')
    >>> A = MatrixSymbol('A', m, n)
    >>> is_square(A)
    Eq(m, n)
    """
    return Eq(arg.rows, arg.cols)


def validate_matadd_integer(*args: MatrixExpr) -> None:
    """Validate matrix shape for addition only for integer values"""
    rows, cols = zip(*(x.shape for x in args))
    if len(set(filter(lambda x: isinstance(x, (int, Integer)), rows))) > 1:
        raise ShapeError(f"Matrices have mismatching shape: {rows}")
    if len(set(filter(lambda x: isinstance(x, (int, Integer)), cols))) > 1:
        raise ShapeError(f"Matrices have mismatching shape: {cols}")


def validate_matmul_integer(*args: MatrixExpr) -> None:
    """Validate matrix shape for multiplication only for integer values"""
    for A, B in zip(args[:-1], args[1:]):
        i, j = A.cols, B.rows
        if isinstance(i, (int, Integer)) and isinstance(j, (int, Integer)) and i != j:
            raise ShapeError("Matrices are not aligned", i, j)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\pygletplot\plot_surface.py ===
import pyglet.gl as pgl

from sympy.core import S
from sympy.plotting.pygletplot.plot_mode_base import PlotModeBase


class PlotSurface(PlotModeBase):

    default_rot_preset = 'perspective'

    def _on_calculate_verts(self):
        self.u_interval = self.intervals[0]
        self.u_set = list(self.u_interval.frange())
        self.v_interval = self.intervals[1]
        self.v_set = list(self.v_interval.frange())
        self.bounds = [[S.Infinity, S.NegativeInfinity, 0],
                       [S.Infinity, S.NegativeInfinity, 0],
                       [S.Infinity, S.NegativeInfinity, 0]]
        evaluate = self._get_evaluator()

        self._calculating_verts_pos = 0.0
        self._calculating_verts_len = float(
            self.u_interval.v_len*self.v_interval.v_len)

        verts = []
        b = self.bounds
        for u in self.u_set:
            column = []
            for v in self.v_set:
                try:
                    _e = evaluate(u, v)  # calculate vertex
                except ZeroDivisionError:
                    _e = None
                if _e is not None:  # update bounding box
                    for axis in range(3):
                        b[axis][0] = min([b[axis][0], _e[axis]])
                        b[axis][1] = max([b[axis][1], _e[axis]])
                column.append(_e)
                self._calculating_verts_pos += 1.0

            verts.append(column)
        for axis in range(3):
            b[axis][2] = b[axis][1] - b[axis][0]
            if b[axis][2] == 0.0:
                b[axis][2] = 1.0

        self.verts = verts
        self.push_wireframe(self.draw_verts(False, False))
        self.push_solid(self.draw_verts(False, True))

    def _on_calculate_cverts(self):
        if not self.verts or not self.color:
            return

        def set_work_len(n):
            self._calculating_cverts_len = float(n)

        def inc_work_pos():
            self._calculating_cverts_pos += 1.0
        set_work_len(1)
        self._calculating_cverts_pos = 0
        self.cverts = self.color.apply_to_surface(self.verts,
                                                  self.u_set,
                                                  self.v_set,
                                                  set_len=set_work_len,
                                                  inc_pos=inc_work_pos)
        self.push_solid(self.draw_verts(True, True))

    def calculate_one_cvert(self, u, v):
        vert = self.verts[u][v]
        return self.color(vert[0], vert[1], vert[2],
                          self.u_set[u], self.v_set[v])

    def draw_verts(self, use_cverts, use_solid_color):
        def f():
            for u in range(1, len(self.u_set)):
                pgl.glBegin(pgl.GL_QUAD_STRIP)
                for v in range(len(self.v_set)):
                    pa = self.verts[u - 1][v]
                    pb = self.verts[u][v]
                    if pa is None or pb is None:
                        pgl.glEnd()
                        pgl.glBegin(pgl.GL_QUAD_STRIP)
                        continue
                    if use_cverts:
                        ca = self.cverts[u - 1][v]
                        cb = self.cverts[u][v]
                        if ca is None:
                            ca = (0, 0, 0)
                        if cb is None:
                            cb = (0, 0, 0)
                    else:
                        if use_solid_color:
                            ca = cb = self.default_solid_color
                        else:
                            ca = cb = self.default_wireframe_color
                    pgl.glColor3f(*ca)
                    pgl.glVertex3f(*pa)
                    pgl.glColor3f(*cb)
                    pgl.glVertex3f(*pb)
                pgl.glEnd()
        return f

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\testing\quality_unicode.py ===
import re
import fnmatch


message_unicode_B = \
    "File contains a unicode character : %s, line %s. " \
    "But not in the whitelist. " \
    "Add the file to the whitelist in " + __file__
message_unicode_D = \
    "File does not contain a unicode character : %s." \
    "but is in the whitelist. " \
    "Remove the file from the whitelist in " + __file__


encoding_header_re = re.compile(
    r'^[ \t\f]*#.*?coding[:=][ \t]*([-_.a-zA-Z0-9]+)')

# Whitelist pattern for files which can have unicode.
unicode_whitelist = [
    # Author names can include non-ASCII characters
    r'*/bin/authors_update.py',
    r'*/bin/mailmap_check.py',

    # These files have functions and test functions for unicode input and
    # output.
    r'*/sympy/testing/tests/test_code_quality.py',
    r'*/sympy/physics/vector/tests/test_printing.py',
    r'*/physics/quantum/tests/test_printing.py',
    r'*/sympy/vector/tests/test_printing.py',
    r'*/sympy/parsing/tests/test_sympy_parser.py',
    r'*/sympy/printing/pretty/stringpict.py',
    r'*/sympy/printing/pretty/tests/test_pretty.py',
    r'*/sympy/printing/tests/test_conventions.py',
    r'*/sympy/printing/tests/test_preview.py',
    r'*/liealgebras/type_g.py',
    r'*/liealgebras/weyl_group.py',
    r'*/liealgebras/tests/test_type_G.py',

    # wigner.py and polarization.py have unicode doctests. These probably
    # don't need to be there but some of the examples that are there are
    # pretty ugly without use_unicode (matrices need to be wrapped across
    # multiple lines etc)
    r'*/sympy/physics/wigner.py',
    r'*/sympy/physics/optics/polarization.py',

    # joint.py uses some unicode for variable names in the docstrings
    r'*/sympy/physics/mechanics/joint.py',

    # lll method has unicode in docstring references and author name
    r'*/sympy/polys/matrices/domainmatrix.py',
    r'*/sympy/matrices/repmatrix.py',

    # Explanation of symbols uses greek letters
    r'*/sympy/core/symbol.py',
]

unicode_strict_whitelist = [
    r'*/sympy/parsing/latex/_antlr/__init__.py',
    # test_mathematica.py uses some unicode for testing Greek characters are working #24055
    r'*/sympy/parsing/tests/test_mathematica.py',
]


def _test_this_file_encoding(
    fname, test_file,
    unicode_whitelist=unicode_whitelist,
    unicode_strict_whitelist=unicode_strict_whitelist):
    """Test helper function for unicode test

    The test may have to operate on filewise manner, so it had moved
    to a separate process.
    """
    has_unicode = False

    is_in_whitelist = False
    is_in_strict_whitelist = False
    for patt in unicode_whitelist:
        if fnmatch.fnmatch(fname, patt):
            is_in_whitelist = True
            break
    for patt in unicode_strict_whitelist:
        if fnmatch.fnmatch(fname, patt):
            is_in_strict_whitelist = True
            is_in_whitelist = True
            break

    if is_in_whitelist:
        for idx, line in enumerate(test_file):
            try:
                line.encode(encoding='ascii')
            except (UnicodeEncodeError, UnicodeDecodeError):
                has_unicode = True

        if not has_unicode and not is_in_strict_whitelist:
            assert False, message_unicode_D % fname

    else:
        for idx, line in enumerate(test_file):
            try:
                line.encode(encoding='ascii')
            except (UnicodeEncodeError, UnicodeDecodeError):
                assert False, message_unicode_B % (fname, idx + 1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_tests\test_exceptiongroup_gc.py ===
from __future__ import annotations

import gc
import sys
from traceback import extract_tb
from typing import TYPE_CHECKING, Callable, NoReturn

import pytest

from .._concat_tb import concat_tb

if TYPE_CHECKING:
    from types import TracebackType

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup


def raiser1() -> NoReturn:
    raiser1_2()


def raiser1_2() -> NoReturn:
    raiser1_3()


def raiser1_3() -> NoReturn:
    raise ValueError("raiser1_string")


def raiser2() -> NoReturn:
    raiser2_2()


def raiser2_2() -> NoReturn:
    raise KeyError("raiser2_string")


def get_exc(raiser: Callable[[], NoReturn]) -> Exception:
    try:
        raiser()
    except Exception as exc:
        return exc
    raise AssertionError("raiser should always raise")  # pragma: no cover


def get_tb(raiser: Callable[[], NoReturn]) -> TracebackType | None:
    return get_exc(raiser).__traceback__


def test_concat_tb() -> None:
    tb1 = get_tb(raiser1)
    tb2 = get_tb(raiser2)

    # These return a list of (filename, lineno, fn name, text) tuples
    # https://docs.python.org/3/library/traceback.html#traceback.extract_tb
    entries1 = extract_tb(tb1)
    entries2 = extract_tb(tb2)

    tb12 = concat_tb(tb1, tb2)
    assert extract_tb(tb12) == entries1 + entries2

    tb21 = concat_tb(tb2, tb1)
    assert extract_tb(tb21) == entries2 + entries1

    # Check degenerate cases
    assert extract_tb(concat_tb(None, tb1)) == entries1
    assert extract_tb(concat_tb(tb1, None)) == entries1
    assert concat_tb(None, None) is None

    # Make sure the original tracebacks didn't get mutated by mistake
    assert extract_tb(get_tb(raiser1)) == entries1
    assert extract_tb(get_tb(raiser2)) == entries2


# Unclear if this can still fail, removing the `del` from _concat_tb.copy_tb does not seem
# to trigger it (on a platform where the `del` is executed)
@pytest.mark.skipif(
    sys.implementation.name != "cpython",
    reason="Only makes sense with refcounting GC",
)
def test_ExceptionGroup_catch_doesnt_create_cyclic_garbage() -> None:
    # https://github.com/python-trio/trio/pull/2063
    gc.collect()
    old_flags = gc.get_debug()

    def make_multi() -> NoReturn:
        raise ExceptionGroup("", [get_exc(raiser1), get_exc(raiser2)])

    try:
        gc.set_debug(gc.DEBUG_SAVEALL)
        with pytest.raises(ExceptionGroup) as excinfo:
            # covers ~~MultiErrorCatcher.__exit__ and~~ _concat_tb.copy_tb
            # TODO: is the above comment true anymore? as this no longer uses MultiError.catch
            raise make_multi()
        for exc in excinfo.value.exceptions:
            assert isinstance(exc, (ValueError, KeyError))
        gc.collect()
        assert not gc.garbage
    finally:
        gc.set_debug(old_flags)
        gc.garbage.clear()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_ufunc.py ===
import ctypes as ct
import itertools
import pickle
import sys
import warnings

import numpy._core._operand_flag_tests as opflag_tests
import numpy._core._rational_tests as _rational_tests
import numpy._core._umath_tests as umt
import pytest
from pytest import param

import numpy as np
import numpy._core.umath as ncu
import numpy.linalg._umath_linalg as uml
from numpy.exceptions import AxisError
from numpy.testing import (
    HAS_REFCOUNT,
    IS_PYPY,
    IS_WASM,
    assert_,
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_no_warnings,
    assert_raises,
    suppress_warnings,
)
from numpy.testing._private.utils import requires_memory

UNARY_UFUNCS = [obj for obj in np._core.umath.__dict__.values()
                    if isinstance(obj, np.ufunc)]
UNARY_OBJECT_UFUNCS = [uf for uf in UNARY_UFUNCS if "O->O" in uf.types]

# Remove functions that do not support `floats`
UNARY_OBJECT_UFUNCS.remove(np.bitwise_count)


class TestUfuncKwargs:
    def test_kwarg_exact(self):
        assert_raises(TypeError, np.add, 1, 2, castingx='safe')
        assert_raises(TypeError, np.add, 1, 2, dtypex=int)
        assert_raises(TypeError, np.add, 1, 2, extobjx=[4096])
        assert_raises(TypeError, np.add, 1, 2, outx=None)
        assert_raises(TypeError, np.add, 1, 2, sigx='ii->i')
        assert_raises(TypeError, np.add, 1, 2, signaturex='ii->i')
        assert_raises(TypeError, np.add, 1, 2, subokx=False)
        assert_raises(TypeError, np.add, 1, 2, wherex=[True])

    def test_sig_signature(self):
        assert_raises(TypeError, np.add, 1, 2, sig='ii->i',
                      signature='ii->i')

    def test_sig_dtype(self):
        assert_raises(TypeError, np.add, 1, 2, sig='ii->i',
                      dtype=int)
        assert_raises(TypeError, np.add, 1, 2, signature='ii->i',
                      dtype=int)

    def test_extobj_removed(self):
        assert_raises(TypeError, np.add, 1, 2, extobj=[4096])


class TestUfuncGenericLoops:
    """Test generic loops.

    The loops to be tested are:

        PyUFunc_ff_f_As_dd_d
        PyUFunc_ff_f
        PyUFunc_dd_d
        PyUFunc_gg_g
        PyUFunc_FF_F_As_DD_D
        PyUFunc_DD_D
        PyUFunc_FF_F
        PyUFunc_GG_G
        PyUFunc_OO_O
        PyUFunc_OO_O_method
        PyUFunc_f_f_As_d_d
        PyUFunc_d_d
        PyUFunc_f_f
        PyUFunc_g_g
        PyUFunc_F_F_As_D_D
        PyUFunc_F_F
        PyUFunc_D_D
        PyUFunc_G_G
        PyUFunc_O_O
        PyUFunc_O_O_method
        PyUFunc_On_Om

    Where:

        f -- float
        d -- double
        g -- long double
        F -- complex float
        D -- complex double
        G -- complex long double
        O -- python object

    It is difficult to assure that each of these loops is entered from the
    Python level as the special cased loops are a moving target and the
    corresponding types are architecture dependent. We probably need to
    define C level testing ufuncs to get at them. For the time being, I've
    just looked at the signatures registered in the build directory to find
    relevant functions.

    """
    np_dtypes = [
        (np.single, np.single), (np.single, np.double),
        (np.csingle, np.csingle), (np.csingle, np.cdouble),
        (np.double, np.double), (np.longdouble, np.longdouble),
        (np.cdouble, np.cdouble), (np.clongdouble, np.clongdouble)]

    @pytest.mark.parametrize('input_dtype,output_dtype', np_dtypes)
    def test_unary_PyUFunc(self, input_dtype, output_dtype, f=np.exp, x=0, y=1):
        xs = np.full(10, input_dtype(x), dtype=output_dtype)
        ys = f(xs)[::2]
        assert_allclose(ys, y)
        assert_equal(ys.dtype, output_dtype)

    def f2(x, y):
        return x**y

    @pytest.mark.parametrize('input_dtype,output_dtype', np_dtypes)
    def test_binary_PyUFunc(self, input_dtype, output_dtype, f=f2, x=0, y=1):
        xs = np.full(10, input_dtype(x), dtype=output_dtype)
        ys = f(xs, xs)[::2]
        assert_allclose(ys, y)
        assert_equal(ys.dtype, output_dtype)

    # class to use in testing object method loops
    class foo:
        def conjugate(self):
            return np.bool(1)

        def logical_xor(self, obj):
            return np.bool(1)

    def test_unary_PyUFunc_O_O(self):
        x = np.ones(10, dtype=object)
        assert_(np.all(np.abs(x) == 1))

    def test_unary_PyUFunc_O_O_method_simple(self, foo=foo):
        x = np.full(10, foo(), dtype=object)
        assert_(np.all(np.conjugate(x) == True))

    def test_binary_PyUFunc_OO_O(self):
        x = np.ones(10, dtype=object)
        assert_(np.all(np.add(x, x) == 2))

    def test_binary_PyUFunc_OO_O_method(self, foo=foo):
        x = np.full(10, foo(), dtype=object)
        assert_(np.all(np.logical_xor(x, x)))

    def test_binary_PyUFunc_On_Om_method(self, foo=foo):
        x = np.full((10, 2, 3), foo(), dtype=object)
        assert_(np.all(np.logical_xor(x, x)))

    def test_python_complex_conjugate(self):
        # The conjugate ufunc should fall back to calling the method:
        arr = np.array([1 + 2j, 3 - 4j], dtype="O")
        assert isinstance(arr[0], complex)
        res = np.conjugate(arr)
        assert res.dtype == np.dtype("O")
        assert_array_equal(res, np.array([1 - 2j, 3 + 4j], dtype="O"))

    @pytest.mark.parametrize("ufunc", UNARY_OBJECT_UFUNCS)
    def test_unary_PyUFunc_O_O_method_full(self, ufunc):
        """Compare the result of the object loop with non-object one"""
        val = np.float64(np.pi / 4)

        class MyFloat(np.float64):
            def __getattr__(self, attr):
                try:
                    return super().__getattr__(attr)
                except AttributeError:
                    return lambda: getattr(np._core.umath, attr)(val)

        # Use 0-D arrays, to ensure the same element call
        num_arr = np.array(val, dtype=np.float64)
        obj_arr = np.array(MyFloat(val), dtype="O")

        with np.errstate(all="raise"):
            try:
                res_num = ufunc(num_arr)
            except Exception as exc:
                with assert_raises(type(exc)):
                    ufunc(obj_arr)
            else:
                res_obj = ufunc(obj_arr)
                assert_array_almost_equal(res_num.astype("O"), res_obj)


def _pickleable_module_global():
    pass


class TestUfunc:
    def test_pickle(self):
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            assert_(pickle.loads(pickle.dumps(np.sin,
                                              protocol=proto)) is np.sin)

            # Check that ufunc not defined in the top level numpy namespace
            # such as numpy._core._rational_tests.test_add can also be pickled
            res = pickle.loads(pickle.dumps(_rational_tests.test_add,
                                            protocol=proto))
            assert_(res is _rational_tests.test_add)

    def test_pickle_withstring(self):
        astring = (b"cnumpy.core\n_ufunc_reconstruct\np0\n"
                   b"(S'numpy._core.umath'\np1\nS'cos'\np2\ntp3\nRp4\n.")
        assert_(pickle.loads(astring) is np.cos)

    @pytest.mark.skipif(IS_PYPY, reason="'is' check does not work on PyPy")
    def test_pickle_name_is_qualname(self):
        # This tests that a simplification of our ufunc pickle code will
        # lead to allowing qualnames as names.  Future ufuncs should
        # possible add a specific qualname, or a hook into pickling instead
        # (dask+numba may benefit).
        _pickleable_module_global.ufunc = umt._pickleable_module_global_ufunc

        obj = pickle.loads(pickle.dumps(_pickleable_module_global.ufunc))
        assert obj is umt._pickleable_module_global_ufunc

    def test_reduceat_shifting_sum(self):
        L = 6
        x = np.arange(L)
        idx = np.array(list(zip(np.arange(L - 2), np.arange(L - 2) + 2))).ravel()
        assert_array_equal(np.add.reduceat(x, idx)[::2], [1, 3, 5, 7])

    def test_all_ufunc(self):
        """Try to check presence and results of all ufuncs.

        The list of ufuncs comes from generate_umath.py and is as follows:

        =====  ====  =============  ===============  ========================
        done   args   function        types                notes
        =====  ====  =============  ===============  ========================
        n      1     conjugate      nums + O
        n      1     absolute       nums + O         complex -> real
        n      1     negative       nums + O
        n      1     sign           nums + O         -> int
        n      1     invert         bool + ints + O  flts raise an error
        n      1     degrees        real + M         cmplx raise an error
        n      1     radians        real + M         cmplx raise an error
        n      1     arccos         flts + M
        n      1     arccosh        flts + M
        n      1     arcsin         flts + M
        n      1     arcsinh        flts + M
        n      1     arctan         flts + M
        n      1     arctanh        flts + M
        n      1     cos            flts + M
        n      1     sin            flts + M
        n      1     tan            flts + M
        n      1     cosh           flts + M
        n      1     sinh           flts + M
        n      1     tanh           flts + M
        n      1     exp            flts + M
        n      1     expm1          flts + M
        n      1     log            flts + M
        n      1     log10          flts + M
        n      1     log1p          flts + M
        n      1     sqrt           flts + M         real x < 0 raises error
        n      1     ceil           real + M
        n      1     trunc          real + M
        n      1     floor          real + M
        n      1     fabs           real + M
        n      1     rint           flts + M
        n      1     isnan          flts             -> bool
        n      1     isinf          flts             -> bool
        n      1     isfinite       flts             -> bool
        n      1     signbit        real             -> bool
        n      1     modf           real             -> (frac, int)
        n      1     logical_not    bool + nums + M  -> bool
        n      2     left_shift     ints + O         flts raise an error
        n      2     right_shift    ints + O         flts raise an error
        n      2     add            bool + nums + O  boolean + is ||
        n      2     subtract       bool + nums + O  boolean - is ^
        n      2     multiply       bool + nums + O  boolean * is &
        n      2     divide         nums + O
        n      2     floor_divide   nums + O
        n      2     true_divide    nums + O         bBhH -> f, iIlLqQ -> d
        n      2     fmod           nums + M
        n      2     power          nums + O
        n      2     greater        bool + nums + O  -> bool
        n      2     greater_equal  bool + nums + O  -> bool
        n      2     less           bool + nums + O  -> bool
        n      2     less_equal     bool + nums + O  -> bool
        n      2     equal          bool + nums + O  -> bool
        n      2     not_equal      bool + nums + O  -> bool
        n      2     logical_and    bool + nums + M  -> bool
        n      2     logical_or     bool + nums + M  -> bool
        n      2     logical_xor    bool + nums + M  -> bool
        n      2     maximum        bool + nums + O
        n      2     minimum        bool + nums + O
        n      2     bitwise_and    bool + ints + O  flts raise an error
        n      2     bitwise_or     bool + ints + O  flts raise an error
        n      2     bitwise_xor    bool + ints + O  flts raise an error
        n      2     arctan2        real + M
        n      2     remainder      ints + real + O
        n      2     hypot          real + M
        =====  ====  =============  ===============  ========================

        Types other than those listed will be accepted, but they are cast to
        the smallest compatible type for which the function is defined. The
        casting rules are:

        bool -> int8 -> float32
        ints -> double

        """
        pass

    # from include/numpy/ufuncobject.h
    size_inferred = 2
    can_ignore = 4

    def test_signature0(self):
        # the arguments to test_signature are: nin, nout, core_signature
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            2, 1, "(i),(i)->()")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1,  1,  0))
        assert_equal(ixs, (0, 0))
        assert_equal(flags, (self.size_inferred,))
        assert_equal(sizes, (-1,))

    def test_signature1(self):
        # empty core signature; treat as plain ufunc (with trivial core)
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            2, 1, "(),()->()")
        assert_equal(enabled, 0)
        assert_equal(num_dims, (0,  0,  0))
        assert_equal(ixs, ())
        assert_equal(flags, ())
        assert_equal(sizes, ())

    def test_signature2(self):
        # more complicated names for variables
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            2, 1, "(i1,i2),(J_1)->(_kAB)")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (2, 1, 1))
        assert_equal(ixs, (0, 1, 2, 3))
        assert_equal(flags, (self.size_inferred,) * 4)
        assert_equal(sizes, (-1, -1, -1, -1))

    def test_signature3(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            2, 1, "(i1, i12),   (J_1)->(i12, i2)")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (2, 1, 2))
        assert_equal(ixs, (0, 1, 2, 1, 3))
        assert_equal(flags, (self.size_inferred,) * 4)
        assert_equal(sizes, (-1, -1, -1, -1))

    def test_signature4(self):
        # matrix_multiply signature from _umath_tests
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            2, 1, "(n,k),(k,m)->(n,m)")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (2, 2, 2))
        assert_equal(ixs, (0, 1, 1, 2, 0, 2))
        assert_equal(flags, (self.size_inferred,) * 3)
        assert_equal(sizes, (-1, -1, -1))

    def test_signature5(self):
        # matmul signature from _umath_tests
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            2, 1, "(n?,k),(k,m?)->(n?,m?)")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (2, 2, 2))
        assert_equal(ixs, (0, 1, 1, 2, 0, 2))
        assert_equal(flags, (self.size_inferred | self.can_ignore,
                             self.size_inferred,
                             self.size_inferred | self.can_ignore))
        assert_equal(sizes, (-1, -1, -1))

    def test_signature6(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            1, 1, "(3)->()")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 0))
        assert_equal(ixs, (0,))
        assert_equal(flags, (0,))
        assert_equal(sizes, (3,))

    def test_signature7(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            3, 1, "(3),(03,3),(n)->(9)")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 2, 1, 1))
        assert_equal(ixs, (0, 0, 0, 1, 2))
        assert_equal(flags, (0, self.size_inferred, 0))
        assert_equal(sizes, (3, -1, 9))

    def test_signature8(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            3, 1, "(3?),(3?,3?),(n)->(9)")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 2, 1, 1))
        assert_equal(ixs, (0, 0, 0, 1, 2))
        assert_equal(flags, (self.can_ignore, self.size_inferred, 0))
        assert_equal(sizes, (3, -1, 9))

    def test_signature9(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            1, 1, "(  3)  -> ( )")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 0))
        assert_equal(ixs, (0,))
        assert_equal(flags, (0,))
        assert_equal(sizes, (3,))

    def test_signature10(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            3, 1, "( 3? ) , (3? ,  3?) ,(n )-> ( 9)")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 2, 1, 1))
        assert_equal(ixs, (0, 0, 0, 1, 2))
        assert_equal(flags, (self.can_ignore, self.size_inferred, 0))
        assert_equal(sizes, (3, -1, 9))

    def test_signature_failure_extra_parenthesis(self):
        with assert_raises(ValueError):
            umt.test_signature(2, 1, "((i)),(i)->()")

    def test_signature_failure_mismatching_parenthesis(self):
        with assert_raises(ValueError):
            umt.test_signature(2, 1, "(i),)i(->()")

    def test_signature_failure_signature_missing_input_arg(self):
        with assert_raises(ValueError):
            umt.test_signature(2, 1, "(i),->()")

    def test_signature_failure_signature_missing_output_arg(self):
        with assert_raises(ValueError):
            umt.test_signature(2, 2, "(i),(i)->()")

    def test_get_signature(self):
        assert_equal(np.vecdot.signature, "(n),(n)->()")

    def test_forced_sig(self):
        a = 0.5 * np.arange(3, dtype='f8')
        assert_equal(np.add(a, 0.5), [0.5, 1, 1.5])
        with assert_raises(TypeError):
            np.add(a, 0.5, sig='i', casting='unsafe')
        assert_equal(np.add(a, 0.5, sig='ii->i', casting='unsafe'), [0, 0, 1])
        with assert_raises(TypeError):
            np.add(a, 0.5, sig=('i4',), casting='unsafe')
        assert_equal(np.add(a, 0.5, sig=('i4', 'i4', 'i4'),
                                            casting='unsafe'), [0, 0, 1])

        b = np.zeros((3,), dtype='f8')
        np.add(a, 0.5, out=b)
        assert_equal(b, [0.5, 1, 1.5])
        b[:] = 0
        with assert_raises(TypeError):
            np.add(a, 0.5, sig='i', out=b, casting='unsafe')
        assert_equal(b, [0, 0, 0])
        np.add(a, 0.5, sig='ii->i', out=b, casting='unsafe')
        assert_equal(b, [0, 0, 1])
        b[:] = 0
        with assert_raises(TypeError):
            np.add(a, 0.5, sig=('i4',), out=b, casting='unsafe')
        assert_equal(b, [0, 0, 0])
        np.add(a, 0.5, sig=('i4', 'i4', 'i4'), out=b, casting='unsafe')
        assert_equal(b, [0, 0, 1])

    def test_signature_all_None(self):
        # signature all None, is an acceptable alternative (since 1.21)
        # to not providing a signature.
        res1 = np.add([3], [4], sig=(None, None, None))
        res2 = np.add([3], [4])
        assert_array_equal(res1, res2)
        res1 = np.maximum([3], [4], sig=(None, None, None))
        res2 = np.maximum([3], [4])
        assert_array_equal(res1, res2)

        with pytest.raises(TypeError):
            # special case, that would be deprecated anyway, so errors:
            np.add(3, 4, signature=(None,))

    def test_signature_dtype_type(self):
        # Since that will be the normal behaviour (past NumPy 1.21)
        # we do support the types already:
        float_dtype = type(np.dtype(np.float64))
        np.add(3, 4, signature=(float_dtype, float_dtype, None))

    @pytest.mark.parametrize("get_kwarg", [
            param(lambda dt: {"dtype": dt}, id="dtype"),
            param(lambda dt: {"signature": (dt, None, None)}, id="signature")])
    def test_signature_dtype_instances_allowed(self, get_kwarg):
        # We allow certain dtype instances when there is a clear singleton
        # and the given one is equivalent; mainly for backcompat.
        int64 = np.dtype("int64")
        int64_2 = pickle.loads(pickle.dumps(int64))
        # Relies on pickling behavior, if assert fails just remove test...
        assert int64 is not int64_2

        assert np.add(1, 2, **get_kwarg(int64_2)).dtype == int64
        td = np.timedelta64(2, "s")
        assert np.add(td, td, **get_kwarg("m8")).dtype == "m8[s]"

        msg = "The `dtype` and `signature` arguments to ufuncs"

        with pytest.raises(TypeError, match=msg):
            np.add(3, 5, **get_kwarg(np.dtype("int64").newbyteorder()))
        with pytest.raises(TypeError, match=msg):
            np.add(3, 5, **get_kwarg(np.dtype("m8[ns]")))
        with pytest.raises(TypeError, match=msg):
            np.add(3, 5, **get_kwarg("m8[ns]"))

    @pytest.mark.parametrize("casting", ["unsafe", "same_kind", "safe"])
    def test_partial_signature_mismatch(self, casting):
        # If the second argument matches already, no need to specify it:
        res = np.ldexp(np.float32(1.), np.int_(2), dtype="d")
        assert res.dtype == "d"
        res = np.ldexp(np.float32(1.), np.int_(2), signature=(None, None, "d"))
        assert res.dtype == "d"

        # ldexp only has a loop for long input as second argument, overriding
        # the output cannot help with that (no matter the casting)
        with pytest.raises(TypeError):
            np.ldexp(1., np.uint64(3), dtype="d")
        with pytest.raises(TypeError):
            np.ldexp(1., np.uint64(3), signature=(None, None, "d"))

    def test_partial_signature_mismatch_with_cache(self):
        with pytest.raises(TypeError):
            np.add(np.float16(1), np.uint64(2), sig=("e", "d", None))
        # Ensure e,d->None is in the dispatching cache (double loop)
        np.add(np.float16(1), np.float64(2))
        # The error must still be raised:
        with pytest.raises(TypeError):
            np.add(np.float16(1), np.uint64(2), sig=("e", "d", None))

    def test_use_output_signature_for_all_arguments(self):
        # Test that providing only `dtype=` or `signature=(None, None, dtype)`
        # is sufficient if falling back to a homogeneous signature works.
        # In this case, the `intp, intp -> intp` loop is chosen.
        res = np.power(1.5, 2.8, dtype=np.intp, casting="unsafe")
        assert res == 1  # the cast happens first.
        res = np.power(1.5, 2.8, signature=(None, None, np.intp),
                       casting="unsafe")
        assert res == 1
        with pytest.raises(TypeError):
            # the unsafe casting would normally cause errors though:
            np.power(1.5, 2.8, dtype=np.intp)

    def test_signature_errors(self):
        with pytest.raises(TypeError,
                    match="the signature object to ufunc must be a string or"):
            np.add(3, 4, signature=123.)  # neither a string nor a tuple

        with pytest.raises(ValueError):
            # bad symbols that do not translate to dtypes
            np.add(3, 4, signature="%^->#")

        with pytest.raises(ValueError):
            np.add(3, 4, signature=b"ii-i")  # incomplete and byte string

        with pytest.raises(ValueError):
            np.add(3, 4, signature="ii>i")  # incomplete string

        with pytest.raises(ValueError):
            np.add(3, 4, signature=(None, "f8"))  # bad length

        with pytest.raises(UnicodeDecodeError):
            np.add(3, 4, signature=b"\xff\xff->i")

    def test_forced_dtype_times(self):
        # Signatures only set the type numbers (not the actual loop dtypes)
        # so using `M` in a signature/dtype should generally work:
        a = np.array(['2010-01-02', '1999-03-14', '1833-03'], dtype='>M8[D]')
        np.maximum(a, a, dtype="M")
        np.maximum.reduce(a, dtype="M")

        arr = np.arange(10, dtype="m8[s]")
        np.add(arr, arr, dtype="m")
        np.maximum(arr, arr, dtype="m")

    @pytest.mark.parametrize("ufunc", [np.add, np.sqrt])
    def test_cast_safety(self, ufunc):
        """Basic test for the safest casts, because ufuncs inner loops can
        indicate a cast-safety as well (which is normally always "no").
        """
        def call_ufunc(arr, **kwargs):
            return ufunc(*(arr,) * ufunc.nin, **kwargs)

        arr = np.array([1., 2., 3.], dtype=np.float32)
        arr_bs = arr.astype(arr.dtype.newbyteorder())
        expected = call_ufunc(arr)
        # Normally, a "no" cast:
        res = call_ufunc(arr, casting="no")
        assert_array_equal(expected, res)
        # Byte-swapping is not allowed with "no" though:
        with pytest.raises(TypeError):
            call_ufunc(arr_bs, casting="no")

        # But is allowed with "equiv":
        res = call_ufunc(arr_bs, casting="equiv")
        assert_array_equal(expected, res)

        # Casting to float64 is safe, but not equiv:
        with pytest.raises(TypeError):
            call_ufunc(arr_bs, dtype=np.float64, casting="equiv")

        # but it is safe cast:
        res = call_ufunc(arr_bs, dtype=np.float64, casting="safe")
        expected = call_ufunc(arr.astype(np.float64))  # upcast
        assert_array_equal(expected, res)

    @pytest.mark.parametrize("ufunc", [np.add, np.equal])
    def test_cast_safety_scalar(self, ufunc):
        # We test add and equal, because equal has special scalar handling
        # Note that the "equiv" casting behavior should maybe be considered
        # a current implementation detail.
        with pytest.raises(TypeError):
            # this picks an integer loop, which is not safe
            ufunc(3., 4., dtype=int, casting="safe")

        with pytest.raises(TypeError):
            # We accept python float as float64 but not float32 for equiv.
            ufunc(3., 4., dtype="float32", casting="equiv")

        # Special case for object and equal (note that equiv implies safe)
        ufunc(3, 4, dtype=object, casting="equiv")
        # Picks a double loop for both, first is equiv, second safe:
        ufunc(np.array([3.]), 3., casting="equiv")
        ufunc(np.array([3.]), 3, casting="safe")
        ufunc(np.array([3]), 3, casting="equiv")

    def test_cast_safety_scalar_special(self):
        # We allow this (and it succeeds) via object, although the equiv
        # part may not be important.
        np.equal(np.array([3]), 2**300, casting="equiv")

    def test_true_divide(self):
        a = np.array(10)
        b = np.array(20)
        tgt = np.array(0.5)

        for tc in 'bhilqBHILQefdgFDG':
            dt = np.dtype(tc)
            aa = a.astype(dt)
            bb = b.astype(dt)

            # Check result value and dtype.
            for x, y in itertools.product([aa, -aa], [bb, -bb]):

                # Check with no output type specified
                if tc in 'FDG':
                    tgt = complex(x) / complex(y)
                else:
                    tgt = float(x) / float(y)

                res = np.true_divide(x, y)
                rtol = max(np.finfo(res).resolution, 1e-15)
                assert_allclose(res, tgt, rtol=rtol)

                if tc in 'bhilqBHILQ':
                    assert_(res.dtype.name == 'float64')
                else:
                    assert_(res.dtype.name == dt.name)

                # Check with output type specified.  This also checks for the
                # incorrect casts in issue gh-3484 because the unary '-' does
                # not change types, even for unsigned types, Hence casts in the
                # ufunc from signed to unsigned and vice versa will lead to
                # errors in the values.
                for tcout in 'bhilqBHILQ':
                    dtout = np.dtype(tcout)
                    assert_raises(TypeError, np.true_divide, x, y, dtype=dtout)

                for tcout in 'efdg':
                    dtout = np.dtype(tcout)
                    if tc in 'FDG':
                        # Casting complex to float is not allowed
                        assert_raises(TypeError, np.true_divide, x, y, dtype=dtout)
                    else:
                        tgt = float(x) / float(y)
                        rtol = max(np.finfo(dtout).resolution, 1e-15)
                        # The value of tiny for double double is NaN
                        with suppress_warnings() as sup:
                            sup.filter(UserWarning)
                            if not np.isnan(np.finfo(dtout).tiny):
                                atol = max(np.finfo(dtout).tiny, 3e-308)
                            else:
                                atol = 3e-308
                        # Some test values result in invalid for float16
                        # and the cast to it may overflow to inf.
                        with np.errstate(invalid='ignore', over='ignore'):
                            res = np.true_divide(x, y, dtype=dtout)
                        if not np.isfinite(res) and tcout == 'e':
                            continue
                        assert_allclose(res, tgt, rtol=rtol, atol=atol)
                        assert_(res.dtype.name == dtout.name)

                for tcout in 'FDG':
                    dtout = np.dtype(tcout)
                    tgt = complex(x) / complex(y)
                    rtol = max(np.finfo(dtout).resolution, 1e-15)
                    # The value of tiny for double double is NaN
                    with suppress_warnings() as sup:
                        sup.filter(UserWarning)
                        if not np.isnan(np.finfo(dtout).tiny):
                            atol = max(np.finfo(dtout).tiny, 3e-308)
                        else:
                            atol = 3e-308
                    res = np.true_divide(x, y, dtype=dtout)
                    if not np.isfinite(res):
                        continue
                    assert_allclose(res, tgt, rtol=rtol, atol=atol)
                    assert_(res.dtype.name == dtout.name)

        # Check booleans
        a = np.ones((), dtype=np.bool)
        res = np.true_divide(a, a)
        assert_(res == 1.0)
        assert_(res.dtype.name == 'float64')
        res = np.true_divide(~a, a)
        assert_(res == 0.0)
        assert_(res.dtype.name == 'float64')

    def test_sum_stability(self):
        a = np.ones(500, dtype=np.float32)
        assert_almost_equal((a / 10.).sum() - a.size / 10., 0, 4)

        a = np.ones(500, dtype=np.float64)
        assert_almost_equal((a / 10.).sum() - a.size / 10., 0, 13)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_sum(self):
        for dt in (int, np.float16, np.float32, np.float64, np.longdouble):
            for v in (0, 1, 2, 7, 8, 9, 15, 16, 19, 127,
                      128, 1024, 1235):
                # warning if sum overflows, which it does in float16
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", RuntimeWarning)

                    tgt = dt(v * (v + 1) / 2)
                    overflow = not np.isfinite(tgt)
                    assert_equal(len(w), 1 * overflow)

                    d = np.arange(1, v + 1, dtype=dt)

                    assert_almost_equal(np.sum(d), tgt)
                    assert_equal(len(w), 2 * overflow)

                    assert_almost_equal(np.sum(d[::-1]), tgt)
                    assert_equal(len(w), 3 * overflow)

            d = np.ones(500, dtype=dt)
            assert_almost_equal(np.sum(d[::2]), 250.)
            assert_almost_equal(np.sum(d[1::2]), 250.)
            assert_almost_equal(np.sum(d[::3]), 167.)
            assert_almost_equal(np.sum(d[1::3]), 167.)
            assert_almost_equal(np.sum(d[::-2]), 250.)
            assert_almost_equal(np.sum(d[-1::-2]), 250.)
            assert_almost_equal(np.sum(d[::-3]), 167.)
            assert_almost_equal(np.sum(d[-1::-3]), 167.)
            # sum with first reduction entry != 0
            d = np.ones((1,), dtype=dt)
            d += d
            assert_almost_equal(d, 2.)

    def test_sum_complex(self):
        for dt in (np.complex64, np.complex128, np.clongdouble):
            for v in (0, 1, 2, 7, 8, 9, 15, 16, 19, 127,
                      128, 1024, 1235):
                tgt = dt(v * (v + 1) / 2) - dt((v * (v + 1) / 2) * 1j)
                d = np.empty(v, dtype=dt)
                d.real = np.arange(1, v + 1)
                d.imag = -np.arange(1, v + 1)
                assert_almost_equal(np.sum(d), tgt)
                assert_almost_equal(np.sum(d[::-1]), tgt)

            d = np.ones(500, dtype=dt) + 1j
            assert_almost_equal(np.sum(d[::2]), 250. + 250j)
            assert_almost_equal(np.sum(d[1::2]), 250. + 250j)
            assert_almost_equal(np.sum(d[::3]), 167. + 167j)
            assert_almost_equal(np.sum(d[1::3]), 167. + 167j)
            assert_almost_equal(np.sum(d[::-2]), 250. + 250j)
            assert_almost_equal(np.sum(d[-1::-2]), 250. + 250j)
            assert_almost_equal(np.sum(d[::-3]), 167. + 167j)
            assert_almost_equal(np.sum(d[-1::-3]), 167. + 167j)
            # sum with first reduction entry != 0
            d = np.ones((1,), dtype=dt) + 1j
            d += d
            assert_almost_equal(d, 2. + 2j)

    def test_sum_initial(self):
        # Integer, single axis
        assert_equal(np.sum([3], initial=2), 5)

        # Floating point
        assert_almost_equal(np.sum([0.2], initial=0.1), 0.3)

        # Multiple non-adjacent axes
        assert_equal(np.sum(np.ones((2, 3, 5), dtype=np.int64), axis=(0, 2), initial=2),
                     [12, 12, 12])

    def test_sum_where(self):
        # More extensive tests done in test_reduction_with_where.
        assert_equal(np.sum([[1., 2.], [3., 4.]], where=[True, False]), 4.)
        assert_equal(np.sum([[1., 2.], [3., 4.]], axis=0, initial=5.,
                            where=[True, False]), [9., 5.])

    def test_vecdot(self):
        arr1 = np.arange(6).reshape((2, 3))
        arr2 = np.arange(3).reshape((1, 3))

        actual = np.vecdot(arr1, arr2)
        expected = np.array([5, 14])

        assert_array_equal(actual, expected)

        actual2 = np.vecdot(arr1.T, arr2.T, axis=-2)
        assert_array_equal(actual2, expected)

        actual3 = np.vecdot(arr1.astype("object"), arr2)
        assert_array_equal(actual3, expected.astype("object"))

    def test_matvec(self):
        arr1 = np.arange(6).reshape((2, 3))
        arr2 = np.arange(3).reshape((1, 3))

        actual = np.matvec(arr1, arr2)
        expected = np.array([[5, 14]])

        assert_array_equal(actual, expected)

        actual2 = np.matvec(arr1.T, arr2.T, axes=[(-1, -2), -2, -1])
        assert_array_equal(actual2, expected)

        actual3 = np.matvec(arr1.astype("object"), arr2)
        assert_array_equal(actual3, expected.astype("object"))

    @pytest.mark.parametrize("vec", [
        np.array([[1., 2., 3.], [4., 5., 6.]]),
        np.array([[1., 2j, 3.], [4., 5., 6j]]),
        np.array([[1., 2., 3.], [4., 5., 6.]], dtype=object),
        np.array([[1., 2j, 3.], [4., 5., 6j]], dtype=object)])
    @pytest.mark.parametrize("matrix", [
        None,
        np.array([[1. + 1j, 0.5, -0.5j],
                  [0.25, 2j, 0.],
                  [4., 0., -1j]])])
    def test_vecmatvec_identity(self, matrix, vec):
        """Check that (x†A)x equals x†(Ax)."""
        mat = matrix if matrix is not None else np.eye(3)
        matvec = np.matvec(mat, vec)  # Ax
        vecmat = np.vecmat(vec, mat)  # x†A
        if matrix is None:
            assert_array_equal(matvec, vec)
            assert_array_equal(vecmat.conj(), vec)
        assert_array_equal(matvec, (mat @ vec[..., np.newaxis]).squeeze(-1))
        assert_array_equal(vecmat, (vec[..., np.newaxis].mT.conj()
                                    @ mat).squeeze(-2))
        expected = np.einsum('...i,ij,...j', vec.conj(), mat, vec)
        vec_matvec = (vec.conj() * matvec).sum(-1)
        vecmat_vec = (vecmat * vec).sum(-1)
        assert_array_equal(vec_matvec, expected)
        assert_array_equal(vecmat_vec, expected)

    @pytest.mark.parametrize("ufunc, shape1, shape2, conj", [
        (np.vecdot, (3,), (3,), True),
        (np.vecmat, (3,), (3, 1), True),
        (np.matvec, (1, 3), (3,), False),
        (np.matmul, (1, 3), (3, 1), False),
    ])
    def test_vecdot_matvec_vecmat_complex(self, ufunc, shape1, shape2, conj):
        arr1 = np.array([1, 2j, 3])
        arr2 = np.array([1, 2, 3])

        actual1 = ufunc(arr1.reshape(shape1), arr2.reshape(shape2))
        expected1 = np.array(((arr1.conj() if conj else arr1) * arr2).sum(),
                             ndmin=min(len(shape1), len(shape2)))
        assert_array_equal(actual1, expected1)
        # This would fail for conj=True, since matmul omits the conjugate.
        if not conj:
            assert_array_equal(arr1.reshape(shape1) @ arr2.reshape(shape2),
                               expected1)

        actual2 = ufunc(arr2.reshape(shape1), arr1.reshape(shape2))
        expected2 = np.array(((arr2.conj() if conj else arr2) * arr1).sum(),
                             ndmin=min(len(shape1), len(shape2)))
        assert_array_equal(actual2, expected2)

        actual3 = ufunc(arr1.reshape(shape1).astype("object"),
                        arr2.reshape(shape2).astype("object"))
        expected3 = expected1.astype(object)
        assert_array_equal(actual3, expected3)

    def test_vecdot_subclass(self):
        class MySubclass(np.ndarray):
            pass

        arr1 = np.arange(6).reshape((2, 3)).view(MySubclass)
        arr2 = np.arange(3).reshape((1, 3)).view(MySubclass)
        result = np.vecdot(arr1, arr2)
        assert isinstance(result, MySubclass)

    def test_vecdot_object_no_conjugate(self):
        arr = np.array(["1", "2"], dtype=object)
        with pytest.raises(AttributeError, match="conjugate"):
            np.vecdot(arr, arr)

    def test_vecdot_object_breaks_outer_loop_on_error(self):
        arr1 = np.ones((3, 3)).astype(object)
        arr2 = arr1.copy()
        arr2[1, 1] = None
        out = np.zeros(3).astype(object)
        with pytest.raises(TypeError, match=r"\*: 'float' and 'NoneType'"):
            np.vecdot(arr1, arr2, out=out)
        assert out[0] == 3
        assert out[1] == out[2] == 0

    def test_broadcast(self):
        msg = "broadcast"
        a = np.arange(4).reshape((2, 1, 2))
        b = np.arange(4).reshape((1, 2, 2))
        assert_array_equal(np.vecdot(a, b), np.sum(a * b, axis=-1), err_msg=msg)
        msg = "extend & broadcast loop dimensions"
        b = np.arange(4).reshape((2, 2))
        assert_array_equal(np.vecdot(a, b), np.sum(a * b, axis=-1), err_msg=msg)
        # Broadcast in core dimensions should fail
        a = np.arange(8).reshape((4, 2))
        b = np.arange(4).reshape((4, 1))
        assert_raises(ValueError, np.vecdot, a, b)
        # Extend core dimensions should fail
        a = np.arange(8).reshape((4, 2))
        b = np.array(7)
        assert_raises(ValueError, np.vecdot, a, b)
        # Broadcast should fail
        a = np.arange(2).reshape((2, 1, 1))
        b = np.arange(3).reshape((3, 1, 1))
        assert_raises(ValueError, np.vecdot, a, b)

        # Writing to a broadcasted array with overlap should warn, gh-2705
        a = np.arange(2)
        b = np.arange(4).reshape((2, 2))
        u, v = np.broadcast_arrays(a, b)
        assert_equal(u.strides[0], 0)
        x = u + v
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            u += v
            assert_equal(len(w), 1)
            assert_(x[0, 0] != u[0, 0])

        # Output reduction should not be allowed.
        # See gh-15139
        a = np.arange(6).reshape(3, 2)
        b = np.ones(2)
        out = np.empty(())
        assert_raises(ValueError, np.vecdot, a, b, out)
        out2 = np.empty(3)
        c = np.vecdot(a, b, out2)
        assert_(c is out2)

    def test_out_broadcasts(self):
        # For ufuncs and gufuncs (not for reductions), we currently allow
        # the output to cause broadcasting of the input arrays.
        # both along dimensions with shape 1 and dimensions which do not
        # exist at all in the inputs.
        arr = np.arange(3).reshape(1, 3)
        out = np.empty((5, 4, 3))
        np.add(arr, arr, out=out)
        assert (out == np.arange(3) * 2).all()

        # The same holds for gufuncs (gh-16484)
        np.vecdot(arr, arr, out=out)
        # the result would be just a scalar `5`, but is broadcast fully:
        assert (out == 5).all()

    @pytest.mark.parametrize(["arr", "out"], [
                ([2], np.empty(())),
                ([1, 2], np.empty(1)),
                (np.ones((4, 3)), np.empty((4, 1)))],
            ids=["(1,)->()", "(2,)->(1,)", "(4, 3)->(4, 1)"])
    def test_out_broadcast_errors(self, arr, out):
        # Output is (currently) allowed to broadcast inputs, but it cannot be
        # smaller than the actual result.
        with pytest.raises(ValueError, match="non-broadcastable"):
            np.positive(arr, out=out)

        with pytest.raises(ValueError, match="non-broadcastable"):
            np.add(np.ones(()), arr, out=out)

    def test_type_cast(self):
        msg = "type cast"
        a = np.arange(6, dtype='short').reshape((2, 3))
        assert_array_equal(np.vecdot(a, a), np.sum(a * a, axis=-1),
                           err_msg=msg)
        msg = "type cast on one argument"
        a = np.arange(6).reshape((2, 3))
        b = a + 0.1
        assert_array_almost_equal(np.vecdot(a, b), np.sum(a * b, axis=-1),
                                  err_msg=msg)

    def test_endian(self):
        msg = "big endian"
        a = np.arange(6, dtype='>i4').reshape((2, 3))
        assert_array_equal(np.vecdot(a, a), np.sum(a * a, axis=-1),
                           err_msg=msg)
        msg = "little endian"
        a = np.arange(6, dtype='<i4').reshape((2, 3))
        assert_array_equal(np.vecdot(a, a), np.sum(a * a, axis=-1),
                           err_msg=msg)

        # Output should always be native-endian
        Ba = np.arange(1, dtype='>f8')
        La = np.arange(1, dtype='<f8')
        assert_equal((Ba + Ba).dtype, np.dtype('f8'))
        assert_equal((Ba + La).dtype, np.dtype('f8'))
        assert_equal((La + Ba).dtype, np.dtype('f8'))
        assert_equal((La + La).dtype, np.dtype('f8'))

        assert_equal(np.absolute(La).dtype, np.dtype('f8'))
        assert_equal(np.absolute(Ba).dtype, np.dtype('f8'))
        assert_equal(np.negative(La).dtype, np.dtype('f8'))
        assert_equal(np.negative(Ba).dtype, np.dtype('f8'))

    def test_incontiguous_array(self):
        msg = "incontiguous memory layout of array"
        x = np.arange(64).reshape((2, 2, 2, 2, 2, 2))
        a = x[:, 0, :, 0, :, 0]
        b = x[:, 1, :, 1, :, 1]
        a[0, 0, 0] = -1
        msg2 = "make sure it references to the original array"
        assert_equal(x[0, 0, 0, 0, 0, 0], -1, err_msg=msg2)
        assert_array_equal(np.vecdot(a, b), np.sum(a * b, axis=-1), err_msg=msg)
        x = np.arange(24).reshape(2, 3, 4)
        a = x.T
        b = x.T
        a[0, 0, 0] = -1
        assert_equal(x[0, 0, 0], -1, err_msg=msg2)
        assert_array_equal(np.vecdot(a, b), np.sum(a * b, axis=-1), err_msg=msg)

    def test_output_argument(self):
        msg = "output argument"
        a = np.arange(12).reshape((2, 3, 2))
        b = np.arange(4).reshape((2, 1, 2)) + 1
        c = np.zeros((2, 3), dtype='int')
        np.vecdot(a, b, c)
        assert_array_equal(c, np.sum(a * b, axis=-1), err_msg=msg)
        c[:] = -1
        np.vecdot(a, b, out=c)
        assert_array_equal(c, np.sum(a * b, axis=-1), err_msg=msg)

        msg = "output argument with type cast"
        c = np.zeros((2, 3), dtype='int16')
        np.vecdot(a, b, c)
        assert_array_equal(c, np.sum(a * b, axis=-1), err_msg=msg)
        c[:] = -1
        np.vecdot(a, b, out=c)
        assert_array_equal(c, np.sum(a * b, axis=-1), err_msg=msg)

        msg = "output argument with incontiguous layout"
        c = np.zeros((2, 3, 4), dtype='int16')
        np.vecdot(a, b, c[..., 0])
        assert_array_equal(c[..., 0], np.sum(a * b, axis=-1), err_msg=msg)
        c[:] = -1
        np.vecdot(a, b, out=c[..., 0])
        assert_array_equal(c[..., 0], np.sum(a * b, axis=-1), err_msg=msg)

    @pytest.mark.parametrize("arg", ["array", "scalar", "subclass"])
    def test_output_ellipsis(self, arg):
        class subclass(np.ndarray):
            def __array_wrap__(self, obj, context=None, return_value=None):
                return super().__array_wrap__(obj, context, return_value)

        if arg == "scalar":
            one = 1
            expected_type = np.ndarray
        elif arg == "array":
            one = np.array(1)
            expected_type = np.ndarray
        elif arg == "subclass":
            one = np.array(1).view(subclass)
            expected_type = subclass

        assert type(np.add(one, 2, out=...)) is expected_type
        assert type(np.add.reduce(one, out=...)) is expected_type
        res1, res2 = np.divmod(one, 2, out=...)
        assert type(res1) is type(res2) is expected_type

    def test_output_ellipsis_errors(self):
        with pytest.raises(TypeError,
                match=r"out=\.\.\. is only allowed as a keyword argument."):
            np.add(1, 2, ...)

        with pytest.raises(TypeError,
                match=r"out=\.\.\. is only allowed as a keyword argument."):
            np.add.reduce(1, (), None, ...)

        with pytest.raises(TypeError,
                match=r"must use `\.\.\.` as `out=\.\.\.` and not per-operand/in a tuple"):
            np.negative(1, out=(...,))

        with pytest.raises(TypeError,
                match=r"must use `\.\.\.` as `out=\.\.\.` and not per-operand/in a tuple"):
            # We only allow out=... not individual args for now
            np.divmod(1, 2, out=(np.empty(()), ...))

        with pytest.raises(TypeError,
                match=r"must use `\.\.\.` as `out=\.\.\.` and not per-operand/in a tuple"):
            np.add.reduce(1, out=(...,))

    def test_axes_argument(self):
        # vecdot signature: '(n),(n)->()'
        a = np.arange(27.).reshape((3, 3, 3))
        b = np.arange(10., 19.).reshape((3, 1, 3))
        # basic tests on inputs (outputs tested below with matrix_multiply).
        c = np.vecdot(a, b)
        assert_array_equal(c, (a * b).sum(-1))
        # default
        c = np.vecdot(a, b, axes=[(-1,), (-1,), ()])
        assert_array_equal(c, (a * b).sum(-1))
        # integers ok for single axis.
        c = np.vecdot(a, b, axes=[-1, -1, ()])
        assert_array_equal(c, (a * b).sum(-1))
        # mix fine
        c = np.vecdot(a, b, axes=[(-1,), -1, ()])
        assert_array_equal(c, (a * b).sum(-1))
        # can omit last axis.
        c = np.vecdot(a, b, axes=[-1, -1])
        assert_array_equal(c, (a * b).sum(-1))
        # can pass in other types of integer (with __index__ protocol)
        c = np.vecdot(a, b, axes=[np.int8(-1), np.array(-1, dtype=np.int32)])
        assert_array_equal(c, (a * b).sum(-1))
        # swap some axes
        c = np.vecdot(a, b, axes=[0, 0])
        assert_array_equal(c, (a * b).sum(0))
        c = np.vecdot(a, b, axes=[0, 2])
        assert_array_equal(c, (a.transpose(1, 2, 0) * b).sum(-1))
        # Check errors for improperly constructed axes arguments.
        # should have list.
        assert_raises(TypeError, np.vecdot, a, b, axes=-1)
        # needs enough elements
        assert_raises(ValueError, np.vecdot, a, b, axes=[-1])
        # should pass in indices.
        assert_raises(TypeError, np.vecdot, a, b, axes=[-1.0, -1.0])
        assert_raises(TypeError, np.vecdot, a, b, axes=[(-1.0,), -1])
        assert_raises(TypeError, np.vecdot, a, b, axes=[None, 1])
        # cannot pass an index unless there is only one dimension
        # (output is wrong in this case)
        assert_raises(AxisError, np.vecdot, a, b, axes=[-1, -1, -1])
        # or pass in generally the wrong number of axes
        assert_raises(AxisError, np.vecdot, a, b, axes=[-1, -1, (-1,)])
        assert_raises(AxisError, np.vecdot, a, b, axes=[-1, (-2, -1), ()])
        # axes need to have same length.
        assert_raises(ValueError, np.vecdot, a, b, axes=[0, 1])

        # matrix_multiply signature: '(m,n),(n,p)->(m,p)'
        mm = umt.matrix_multiply
        a = np.arange(12).reshape((2, 3, 2))
        b = np.arange(8).reshape((2, 2, 2, 1)) + 1
        # Sanity check.
        c = mm(a, b)
        assert_array_equal(c, np.matmul(a, b))
        # Default axes.
        c = mm(a, b, axes=[(-2, -1), (-2, -1), (-2, -1)])
        assert_array_equal(c, np.matmul(a, b))
        # Default with explicit axes.
        c = mm(a, b, axes=[(1, 2), (2, 3), (2, 3)])
        assert_array_equal(c, np.matmul(a, b))
        # swap some axes.
        c = mm(a, b, axes=[(0, -1), (1, 2), (-2, -1)])
        assert_array_equal(c, np.matmul(a.transpose(1, 0, 2),
                                        b.transpose(0, 3, 1, 2)))
        # Default with output array.
        c = np.empty((2, 2, 3, 1))
        d = mm(a, b, out=c, axes=[(1, 2), (2, 3), (2, 3)])
        assert_(c is d)
        assert_array_equal(c, np.matmul(a, b))
        # Transposed output array
        c = np.empty((1, 2, 2, 3))
        d = mm(a, b, out=c, axes=[(-2, -1), (-2, -1), (3, 0)])
        assert_(c is d)
        assert_array_equal(c, np.matmul(a, b).transpose(3, 0, 1, 2))
        # Check errors for improperly constructed axes arguments.
        # wrong argument
        assert_raises(TypeError, mm, a, b, axis=1)
        # axes should be list
        assert_raises(TypeError, mm, a, b, axes=1)
        assert_raises(TypeError, mm, a, b, axes=((-2, -1), (-2, -1), (-2, -1)))
        # list needs to have right length
        assert_raises(ValueError, mm, a, b, axes=[])
        assert_raises(ValueError, mm, a, b, axes=[(-2, -1)])
        # list should not contain None, or lists
        assert_raises(TypeError, mm, a, b, axes=[None, None, None])
        assert_raises(TypeError,
                      mm, a, b, axes=[[-2, -1], [-2, -1], [-2, -1]])
        assert_raises(TypeError,
                      mm, a, b, axes=[(-2, -1), (-2, -1), [-2, -1]])
        assert_raises(TypeError, mm, a, b, axes=[(-2, -1), (-2, -1), None])
        # single integers are AxisErrors if more are required
        assert_raises(AxisError, mm, a, b, axes=[-1, -1, -1])
        assert_raises(AxisError, mm, a, b, axes=[(-2, -1), (-2, -1), -1])
        # tuples should not have duplicated values
        assert_raises(ValueError, mm, a, b, axes=[(-2, -1), (-2, -1), (-2, -2)])
        # arrays should have enough axes.
        z = np.zeros((2, 2))
        assert_raises(ValueError, mm, z, z[0])
        assert_raises(ValueError, mm, z, z, out=z[:, 0])
        assert_raises(ValueError, mm, z[1], z, axes=[0, 1])
        assert_raises(ValueError, mm, z, z, out=z[0], axes=[0, 1])
        # Regular ufuncs should not accept axes.
        assert_raises(TypeError, np.add, 1., 1., axes=[0])
        # should be able to deal with bad unrelated kwargs.
        assert_raises(TypeError, mm, z, z, axes=[0, 1], parrot=True)

    def test_axis_argument(self):
        # vecdot signature: '(n),(n)->()'
        a = np.arange(27.).reshape((3, 3, 3))
        b = np.arange(10., 19.).reshape((3, 1, 3))
        c = np.vecdot(a, b)
        assert_array_equal(c, (a * b).sum(-1))
        c = np.vecdot(a, b, axis=-1)
        assert_array_equal(c, (a * b).sum(-1))
        out = np.zeros_like(c)
        d = np.vecdot(a, b, axis=-1, out=out)
        assert_(d is out)
        assert_array_equal(d, c)
        c = np.vecdot(a, b, axis=0)
        assert_array_equal(c, (a * b).sum(0))
        # Sanity checks on innerwt and cumsum.
        a = np.arange(6).reshape((2, 3))
        b = np.arange(10, 16).reshape((2, 3))
        w = np.arange(20, 26).reshape((2, 3))
        assert_array_equal(umt.innerwt(a, b, w, axis=0),
                           np.sum(a * b * w, axis=0))
        assert_array_equal(umt.cumsum(a, axis=0), np.cumsum(a, axis=0))
        assert_array_equal(umt.cumsum(a, axis=-1), np.cumsum(a, axis=-1))
        out = np.empty_like(a)
        b = umt.cumsum(a, out=out, axis=0)
        assert_(out is b)
        assert_array_equal(b, np.cumsum(a, axis=0))
        b = umt.cumsum(a, out=out, axis=1)
        assert_(out is b)
        assert_array_equal(b, np.cumsum(a, axis=-1))
        # Check errors.
        # Cannot pass in both axis and axes.
        assert_raises(TypeError, np.vecdot, a, b, axis=0, axes=[0, 0])
        # Not an integer.
        assert_raises(TypeError, np.vecdot, a, b, axis=[0])
        # more than 1 core dimensions.
        mm = umt.matrix_multiply
        assert_raises(TypeError, mm, a, b, axis=1)
        # Output wrong size in axis.
        out = np.empty((1, 2, 3), dtype=a.dtype)
        assert_raises(ValueError, umt.cumsum, a, out=out, axis=0)
        # Regular ufuncs should not accept axis.
        assert_raises(TypeError, np.add, 1., 1., axis=0)

    def test_keepdims_argument(self):
        # vecdot signature: '(n),(n)->()'
        a = np.arange(27.).reshape((3, 3, 3))
        b = np.arange(10., 19.).reshape((3, 1, 3))
        c = np.vecdot(a, b)
        assert_array_equal(c, (a * b).sum(-1))
        c = np.vecdot(a, b, keepdims=False)
        assert_array_equal(c, (a * b).sum(-1))
        c = np.vecdot(a, b, keepdims=True)
        assert_array_equal(c, (a * b).sum(-1, keepdims=True))
        out = np.zeros_like(c)
        d = np.vecdot(a, b, keepdims=True, out=out)
        assert_(d is out)
        assert_array_equal(d, c)
        # Now combined with axis and axes.
        c = np.vecdot(a, b, axis=-1, keepdims=False)
        assert_array_equal(c, (a * b).sum(-1, keepdims=False))
        c = np.vecdot(a, b, axis=-1, keepdims=True)
        assert_array_equal(c, (a * b).sum(-1, keepdims=True))
        c = np.vecdot(a, b, axis=0, keepdims=False)
        assert_array_equal(c, (a * b).sum(0, keepdims=False))
        c = np.vecdot(a, b, axis=0, keepdims=True)
        assert_array_equal(c, (a * b).sum(0, keepdims=True))
        c = np.vecdot(a, b, axes=[(-1,), (-1,), ()], keepdims=False)
        assert_array_equal(c, (a * b).sum(-1))
        c = np.vecdot(a, b, axes=[(-1,), (-1,), (-1,)], keepdims=True)
        assert_array_equal(c, (a * b).sum(-1, keepdims=True))
        c = np.vecdot(a, b, axes=[0, 0], keepdims=False)
        assert_array_equal(c, (a * b).sum(0))
        c = np.vecdot(a, b, axes=[0, 0, 0], keepdims=True)
        assert_array_equal(c, (a * b).sum(0, keepdims=True))
        c = np.vecdot(a, b, axes=[0, 2], keepdims=False)
        assert_array_equal(c, (a.transpose(1, 2, 0) * b).sum(-1))
        c = np.vecdot(a, b, axes=[0, 2], keepdims=True)
        assert_array_equal(c, (a.transpose(1, 2, 0) * b).sum(-1,
                                                             keepdims=True))
        c = np.vecdot(a, b, axes=[0, 2, 2], keepdims=True)
        assert_array_equal(c, (a.transpose(1, 2, 0) * b).sum(-1,
                                                             keepdims=True))
        c = np.vecdot(a, b, axes=[0, 2, 0], keepdims=True)
        assert_array_equal(c, (a * b.transpose(2, 0, 1)).sum(0, keepdims=True))
        # Hardly useful, but should work.
        c = np.vecdot(a, b, axes=[0, 2, 1], keepdims=True)
        assert_array_equal(c, (a.transpose(1, 0, 2) * b.transpose(0, 2, 1))
                           .sum(1, keepdims=True))
        # Check with two core dimensions.
        a = np.eye(3) * np.arange(4.)[:, np.newaxis, np.newaxis]
        expected = uml.det(a)
        c = uml.det(a, keepdims=False)
        assert_array_equal(c, expected)
        c = uml.det(a, keepdims=True)
        assert_array_equal(c, expected[:, np.newaxis, np.newaxis])
        a = np.eye(3) * np.arange(4.)[:, np.newaxis, np.newaxis]
        expected_s, expected_l = uml.slogdet(a)
        cs, cl = uml.slogdet(a, keepdims=False)
        assert_array_equal(cs, expected_s)
        assert_array_equal(cl, expected_l)
        cs, cl = uml.slogdet(a, keepdims=True)
        assert_array_equal(cs, expected_s[:, np.newaxis, np.newaxis])
        assert_array_equal(cl, expected_l[:, np.newaxis, np.newaxis])
        # Sanity check on innerwt.
        a = np.arange(6).reshape((2, 3))
        b = np.arange(10, 16).reshape((2, 3))
        w = np.arange(20, 26).reshape((2, 3))
        assert_array_equal(umt.innerwt(a, b, w, keepdims=True),
                           np.sum(a * b * w, axis=-1, keepdims=True))
        assert_array_equal(umt.innerwt(a, b, w, axis=0, keepdims=True),
                           np.sum(a * b * w, axis=0, keepdims=True))
        # Check errors.
        # Not a boolean
        assert_raises(TypeError, np.vecdot, a, b, keepdims='true')
        # More than 1 core dimension, and core output dimensions.
        mm = umt.matrix_multiply
        assert_raises(TypeError, mm, a, b, keepdims=True)
        assert_raises(TypeError, mm, a, b, keepdims=False)
        # Regular ufuncs should not accept keepdims.
        assert_raises(TypeError, np.add, 1., 1., keepdims=False)

    def test_innerwt(self):
        a = np.arange(6).reshape((2, 3))
        b = np.arange(10, 16).reshape((2, 3))
        w = np.arange(20, 26).reshape((2, 3))
        assert_array_equal(umt.innerwt(a, b, w), np.sum(a * b * w, axis=-1))
        a = np.arange(100, 124).reshape((2, 3, 4))
        b = np.arange(200, 224).reshape((2, 3, 4))
        w = np.arange(300, 324).reshape((2, 3, 4))
        assert_array_equal(umt.innerwt(a, b, w), np.sum(a * b * w, axis=-1))

    def test_innerwt_empty(self):
        """Test generalized ufunc with zero-sized operands"""
        a = np.array([], dtype='f8')
        b = np.array([], dtype='f8')
        w = np.array([], dtype='f8')
        assert_array_equal(umt.innerwt(a, b, w), np.sum(a * b * w, axis=-1))

    def test_cross1d(self):
        """Test with fixed-sized signature."""
        a = np.eye(3)
        assert_array_equal(umt.cross1d(a, a), np.zeros((3, 3)))
        out = np.zeros((3, 3))
        result = umt.cross1d(a[0], a, out)
        assert_(result is out)
        assert_array_equal(result, np.vstack((np.zeros(3), a[2], -a[1])))
        assert_raises(ValueError, umt.cross1d, np.eye(4), np.eye(4))
        assert_raises(ValueError, umt.cross1d, a, np.arange(4.))
        # Wrong output core dimension.
        assert_raises(ValueError, umt.cross1d, a, np.arange(3.), np.zeros((3, 4)))
        # Wrong output broadcast dimension (see gh-15139).
        assert_raises(ValueError, umt.cross1d, a, np.arange(3.), np.zeros(3))

    def test_can_ignore_signature(self):
        # Comparing the effects of ? in signature:
        # matrix_multiply: (m,n),(n,p)->(m,p)    # all must be there.
        # matmul:        (m?,n),(n,p?)->(m?,p?)  # allow missing m, p.
        mat = np.arange(12).reshape((2, 3, 2))
        single_vec = np.arange(2)
        col_vec = single_vec[:, np.newaxis]
        col_vec_array = np.arange(8).reshape((2, 2, 2, 1)) + 1
        # matrix @ single column vector with proper dimension
        mm_col_vec = umt.matrix_multiply(mat, col_vec)
        # matmul does the same thing
        matmul_col_vec = umt.matmul(mat, col_vec)
        assert_array_equal(matmul_col_vec, mm_col_vec)
        # matrix @ vector without dimension making it a column vector.
        # matrix multiply fails -> missing core dim.
        assert_raises(ValueError, umt.matrix_multiply, mat, single_vec)
        # matmul mimicker passes, and returns a vector.
        matmul_col = umt.matmul(mat, single_vec)
        assert_array_equal(matmul_col, mm_col_vec.squeeze())
        # Now with a column array: same as for column vector,
        # broadcasting sensibly.
        mm_col_vec = umt.matrix_multiply(mat, col_vec_array)
        matmul_col_vec = umt.matmul(mat, col_vec_array)
        assert_array_equal(matmul_col_vec, mm_col_vec)
        # As above, but for row vector
        single_vec = np.arange(3)
        row_vec = single_vec[np.newaxis, :]
        row_vec_array = np.arange(24).reshape((4, 2, 1, 1, 3)) + 1
        # row vector @ matrix
        mm_row_vec = umt.matrix_multiply(row_vec, mat)
        matmul_row_vec = umt.matmul(row_vec, mat)
        assert_array_equal(matmul_row_vec, mm_row_vec)
        # single row vector @ matrix
        assert_raises(ValueError, umt.matrix_multiply, single_vec, mat)
        matmul_row = umt.matmul(single_vec, mat)
        assert_array_equal(matmul_row, mm_row_vec.squeeze())
        # row vector array @ matrix
        mm_row_vec = umt.matrix_multiply(row_vec_array, mat)
        matmul_row_vec = umt.matmul(row_vec_array, mat)
        assert_array_equal(matmul_row_vec, mm_row_vec)
        # Now for vector combinations
        # row vector @ column vector
        col_vec = row_vec.T
        col_vec_array = row_vec_array.swapaxes(-2, -1)
        mm_row_col_vec = umt.matrix_multiply(row_vec, col_vec)
        matmul_row_col_vec = umt.matmul(row_vec, col_vec)
        assert_array_equal(matmul_row_col_vec, mm_row_col_vec)
        # single row vector @ single col vector
        assert_raises(ValueError, umt.matrix_multiply, single_vec, single_vec)
        matmul_row_col = umt.matmul(single_vec, single_vec)
        assert_array_equal(matmul_row_col, mm_row_col_vec.squeeze())
        # row vector array @ matrix
        mm_row_col_array = umt.matrix_multiply(row_vec_array, col_vec_array)
        matmul_row_col_array = umt.matmul(row_vec_array, col_vec_array)
        assert_array_equal(matmul_row_col_array, mm_row_col_array)
        # Finally, check that things are *not* squeezed if one gives an
        # output.
        out = np.zeros_like(mm_row_col_array)
        out = umt.matrix_multiply(row_vec_array, col_vec_array, out=out)
        assert_array_equal(out, mm_row_col_array)
        out[:] = 0
        out = umt.matmul(row_vec_array, col_vec_array, out=out)
        assert_array_equal(out, mm_row_col_array)
        # And check one cannot put missing dimensions back.
        out = np.zeros_like(mm_row_col_vec)
        assert_raises(ValueError, umt.matrix_multiply, single_vec, single_vec,
                      out)
        # But fine for matmul, since it is just a broadcast.
        out = umt.matmul(single_vec, single_vec, out)
        assert_array_equal(out, mm_row_col_vec.squeeze())

    def test_matrix_multiply(self):
        self.compare_matrix_multiply_results(np.int64)
        self.compare_matrix_multiply_results(np.double)

    def test_matrix_multiply_umath_empty(self):
        res = umt.matrix_multiply(np.ones((0, 10)), np.ones((10, 0)))
        assert_array_equal(res, np.zeros((0, 0)))
        res = umt.matrix_multiply(np.ones((10, 0)), np.ones((0, 10)))
        assert_array_equal(res, np.zeros((10, 10)))

    def compare_matrix_multiply_results(self, tp):
        d1 = np.array(np.random.rand(2, 3, 4), dtype=tp)
        d2 = np.array(np.random.rand(2, 3, 4), dtype=tp)
        msg = f"matrix multiply on type {d1.dtype.name}"

        def permute_n(n):
            if n == 1:
                return ([0],)
            ret = ()
            base = permute_n(n - 1)
            for perm in base:
                for i in range(n):
                    new = perm + [n - 1]
                    new[n - 1] = new[i]
                    new[i] = n - 1
                    ret += (new,)
            return ret

        def slice_n(n):
            if n == 0:
                return ((),)
            ret = ()
            base = slice_n(n - 1)
            for sl in base:
                ret += (sl + (slice(None),),)
                ret += (sl + (slice(0, 1),),)
            return ret

        def broadcastable(s1, s2):
            return s1 == s2 or 1 in {s1, s2}

        permute_3 = permute_n(3)
        slice_3 = slice_n(3) + ((slice(None, None, -1),) * 3,)

        ref = True
        for p1 in permute_3:
            for p2 in permute_3:
                for s1 in slice_3:
                    for s2 in slice_3:
                        a1 = d1.transpose(p1)[s1]
                        a2 = d2.transpose(p2)[s2]
                        ref = ref and a1.base is not None
                        ref = ref and a2.base is not None
                        if (a1.shape[-1] == a2.shape[-2] and
                                broadcastable(a1.shape[0], a2.shape[0])):
                            assert_array_almost_equal(
                                umt.matrix_multiply(a1, a2),
                                np.sum(a2[..., np.newaxis].swapaxes(-3, -1) *
                                       a1[..., np.newaxis, :], axis=-1),
                                err_msg=msg + f' {str(a1.shape)} {str(a2.shape)}')

        assert_equal(ref, True, err_msg="reference check")

    def test_euclidean_pdist(self):
        a = np.arange(12, dtype=float).reshape(4, 3)
        out = np.empty((a.shape[0] * (a.shape[0] - 1) // 2,), dtype=a.dtype)
        umt.euclidean_pdist(a, out)
        b = np.sqrt(np.sum((a[:, None] - a)**2, axis=-1))
        b = b[~np.tri(a.shape[0], dtype=bool)]
        assert_almost_equal(out, b)
        # An output array is required to determine p with signature (n,d)->(p)
        assert_raises(ValueError, umt.euclidean_pdist, a)

    def test_cumsum(self):
        a = np.arange(10)
        result = umt.cumsum(a)
        assert_array_equal(result, a.cumsum())

    def test_object_logical(self):
        a = np.array([3, None, True, False, "test", ""], dtype=object)
        assert_equal(np.logical_or(a, None),
                        np.array([x or None for x in a], dtype=object))
        assert_equal(np.logical_or(a, True),
                        np.array([x or True for x in a], dtype=object))
        assert_equal(np.logical_or(a, 12),
                        np.array([x or 12 for x in a], dtype=object))
        assert_equal(np.logical_or(a, "blah"),
                        np.array([x or "blah" for x in a], dtype=object))

        assert_equal(np.logical_and(a, None),
                        np.array([x and None for x in a], dtype=object))
        assert_equal(np.logical_and(a, True),
                        np.array([x and True for x in a], dtype=object))
        assert_equal(np.logical_and(a, 12),
                        np.array([x and 12 for x in a], dtype=object))
        assert_equal(np.logical_and(a, "blah"),
                        np.array([x and "blah" for x in a], dtype=object))

        assert_equal(np.logical_not(a),
                        np.array([not x for x in a], dtype=object))

        assert_equal(np.logical_or.reduce(a), 3)
        assert_equal(np.logical_and.reduce(a), None)

    def test_object_comparison(self):
        class HasComparisons:
            def __eq__(self, other):
                return '=='

        arr0d = np.array(HasComparisons())
        assert_equal(arr0d == arr0d, True)
        assert_equal(np.equal(arr0d, arr0d), True)  # normal behavior is a cast

        arr1d = np.array([HasComparisons()])
        assert_equal(arr1d == arr1d, np.array([True]))
        assert_equal(np.equal(arr1d, arr1d), np.array([True]))  # normal behavior is a cast
        assert_equal(np.equal(arr1d, arr1d, dtype=object), np.array(['==']))

    def test_object_array_reduction(self):
        # Reductions on object arrays
        a = np.array(['a', 'b', 'c'], dtype=object)
        assert_equal(np.sum(a), 'abc')
        assert_equal(np.max(a), 'c')
        assert_equal(np.min(a), 'a')
        a = np.array([True, False, True], dtype=object)
        assert_equal(np.sum(a), 2)
        assert_equal(np.prod(a), 0)
        assert_equal(np.any(a), True)
        assert_equal(np.all(a), False)
        assert_equal(np.max(a), True)
        assert_equal(np.min(a), False)
        assert_equal(np.array([[1]], dtype=object).sum(), 1)
        assert_equal(np.array([[[1, 2]]], dtype=object).sum((0, 1)), [1, 2])
        assert_equal(np.array([1], dtype=object).sum(initial=1), 2)
        assert_equal(np.array([[1], [2, 3]], dtype=object)
                     .sum(initial=[0], where=[False, True]), [0, 2, 3])

    def test_object_array_accumulate_inplace(self):
        # Checks that in-place accumulates work, see also gh-7402
        arr = np.ones(4, dtype=object)
        arr[:] = [[1] for i in range(4)]
        # Twice reproduced also for tuples:
        np.add.accumulate(arr, out=arr)
        np.add.accumulate(arr, out=arr)
        assert_array_equal(arr,
                           np.array([[1] * i for i in [1, 3, 6, 10]], dtype=object),
                          )

        # And the same if the axis argument is used
        arr = np.ones((2, 4), dtype=object)
        arr[0, :] = [[2] for i in range(4)]
        np.add.accumulate(arr, out=arr, axis=-1)
        np.add.accumulate(arr, out=arr, axis=-1)
        assert_array_equal(arr[0, :],
                           np.array([[2] * i for i in [1, 3, 6, 10]], dtype=object),
                          )

    def test_object_array_accumulate_failure(self):
        # Typical accumulation on object works as expected:
        res = np.add.accumulate(np.array([1, 0, 2], dtype=object))
        assert_array_equal(res, np.array([1, 1, 3], dtype=object))
        # But errors are propagated from the inner-loop if they occur:
        with pytest.raises(TypeError):
            np.add.accumulate([1, None, 2])

    def test_object_array_reduceat_inplace(self):
        # Checks that in-place reduceats work, see also gh-7465
        arr = np.empty(4, dtype=object)
        arr[:] = [[1] for i in range(4)]
        out = np.empty(4, dtype=object)
        out[:] = [[1] for i in range(4)]
        np.add.reduceat(arr, np.arange(4), out=arr)
        np.add.reduceat(arr, np.arange(4), out=arr)
        assert_array_equal(arr, out)

        # And the same if the axis argument is used
        arr = np.ones((2, 4), dtype=object)
        arr[0, :] = [[2] for i in range(4)]
        out = np.ones((2, 4), dtype=object)
        out[0, :] = [[2] for i in range(4)]
        np.add.reduceat(arr, np.arange(4), out=arr, axis=-1)
        np.add.reduceat(arr, np.arange(4), out=arr, axis=-1)
        assert_array_equal(arr, out)

    def test_object_array_reduceat_failure(self):
        # Reduceat works as expected when no invalid operation occurs (None is
        # not involved in an operation here)
        res = np.add.reduceat(np.array([1, None, 2], dtype=object), [1, 2])
        assert_array_equal(res, np.array([None, 2], dtype=object))
        # But errors when None would be involved in an operation:
        with pytest.raises(TypeError):
            np.add.reduceat([1, None, 2], [0, 2])

    def test_zerosize_reduction(self):
        # Test with default dtype and object dtype
        for a in [[], np.array([], dtype=object)]:
            assert_equal(np.sum(a), 0)
            assert_equal(np.prod(a), 1)
            assert_equal(np.any(a), False)
            assert_equal(np.all(a), True)
            assert_raises(ValueError, np.max, a)
            assert_raises(ValueError, np.min, a)

    def test_axis_out_of_bounds(self):
        a = np.array([False, False])
        assert_raises(AxisError, a.all, axis=1)
        a = np.array([False, False])
        assert_raises(AxisError, a.all, axis=-2)

        a = np.array([False, False])
        assert_raises(AxisError, a.any, axis=1)
        a = np.array([False, False])
        assert_raises(AxisError, a.any, axis=-2)

    def test_scalar_reduction(self):
        # The functions 'sum', 'prod', etc allow specifying axis=0
        # even for scalars
        assert_equal(np.sum(3, axis=0), 3)
        assert_equal(np.prod(3.5, axis=0), 3.5)
        assert_equal(np.any(True, axis=0), True)
        assert_equal(np.all(False, axis=0), False)
        assert_equal(np.max(3, axis=0), 3)
        assert_equal(np.min(2.5, axis=0), 2.5)

        # Check scalar behaviour for ufuncs without an identity
        assert_equal(np.power.reduce(3), 3)

        # Make sure that scalars are coming out from this operation
        assert_(type(np.prod(np.float32(2.5), axis=0)) is np.float32)
        assert_(type(np.sum(np.float32(2.5), axis=0)) is np.float32)
        assert_(type(np.max(np.float32(2.5), axis=0)) is np.float32)
        assert_(type(np.min(np.float32(2.5), axis=0)) is np.float32)

        # check if scalars/0-d arrays get cast
        assert_(type(np.any(0, axis=0)) is np.bool)

        # assert that 0-d arrays get wrapped
        class MyArray(np.ndarray):
            pass
        a = np.array(1).view(MyArray)
        assert_(type(np.any(a)) is MyArray)

    def test_casting_out_param(self):
        # Test that it's possible to do casts on output
        a = np.ones((200, 100), np.int64)
        b = np.ones((200, 100), np.int64)
        c = np.ones((200, 100), np.float64)
        np.add(a, b, out=c)
        assert_equal(c, 2)

        a = np.zeros(65536)
        b = np.zeros(65536, dtype=np.float32)
        np.subtract(a, 0, out=b)
        assert_equal(b, 0)

    def test_where_param(self):
        # Test that the where= ufunc parameter works with regular arrays
        a = np.arange(7)
        b = np.ones(7)
        c = np.zeros(7)
        np.add(a, b, out=c, where=(a % 2 == 1))
        assert_equal(c, [0, 2, 0, 4, 0, 6, 0])

        a = np.arange(4).reshape(2, 2) + 2
        np.power(a, [2, 3], out=a, where=[[0, 1], [1, 0]])
        assert_equal(a, [[2, 27], [16, 5]])
        # Broadcasting the where= parameter
        np.subtract(a, 2, out=a, where=[True, False])
        assert_equal(a, [[0, 27], [14, 5]])

    def test_where_param_buffer_output(self):
        # This test is temporarily skipped because it requires
        # adding masking features to the nditer to work properly

        # With casting on output
        a = np.ones(10, np.int64)
        b = np.ones(10, np.int64)
        c = 1.5 * np.ones(10, np.float64)
        np.add(a, b, out=c, where=[1, 0, 0, 1, 0, 0, 1, 1, 1, 0])
        assert_equal(c, [2, 1.5, 1.5, 2, 1.5, 1.5, 2, 2, 2, 1.5])

    def test_where_param_alloc(self):
        # With casting and allocated output
        a = np.array([1], dtype=np.int64)
        m = np.array([True], dtype=bool)
        assert_equal(np.sqrt(a, where=m), [1])

        # No casting and allocated output
        a = np.array([1], dtype=np.float64)
        m = np.array([True], dtype=bool)
        assert_equal(np.sqrt(a, where=m), [1])

    def test_where_with_broadcasting(self):
        # See gh-17198
        a = np.random.random((5000, 4))
        b = np.random.random((5000, 1))

        where = a > 0.3
        out = np.full_like(a, 0)
        np.less(a, b, where=where, out=out)
        b_where = np.broadcast_to(b, a.shape)[where]
        assert_array_equal((a[where] < b_where), out[where].astype(bool))
        assert not out[~where].any()  # outside mask, out remains all 0

    @staticmethod
    def identityless_reduce_arrs():
        yield np.empty((2, 3, 4), order='C')
        yield np.empty((2, 3, 4), order='F')
        # Mixed order (reduce order differs outer)
        yield np.empty((2, 4, 3), order='C').swapaxes(1, 2)
        # Reversed order
        yield np.empty((2, 3, 4), order='C')[::-1, ::-1, ::-1]
        # Not contiguous
        yield np.empty((3, 5, 4), order='C').swapaxes(1, 2)[1:, 1:, 1:]
        # Not contiguous and not aligned
        a = np.empty((3 * 4 * 5 * 8 + 1,), dtype='i1')
        a = a[1:].view(dtype='f8')
        a.shape = (3, 4, 5)
        a = a[1:, 1:, 1:]
        yield a

    @pytest.mark.parametrize("a", identityless_reduce_arrs())
    @pytest.mark.parametrize("pos", [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    def test_identityless_reduction(self, a, pos):
        # np.minimum.reduce is an identityless reduction
        a[...] = 1
        a[pos] = 0

        for axis in [None, (0, 1), (0, 2), (1, 2), 0, 1, 2, ()]:
            if axis is None:
                axes = np.array([], dtype=np.intp)
            else:
                axes = np.delete(np.arange(a.ndim), axis)

            expected_pos = tuple(np.array(pos)[axes])
            expected = np.ones(np.array(a.shape)[axes])
            expected[expected_pos] = 0

            res = np.minimum.reduce(a, axis=axis)
            assert_equal(res, expected, strict=True)

            res = np.full_like(res, np.nan)
            np.minimum.reduce(a, axis=axis, out=res)
            assert_equal(res, expected, strict=True)

    @requires_memory(6 * 1024**3)
    @pytest.mark.skipif(sys.maxsize < 2**32,
            reason="test array too large for 32bit platform")
    def test_identityless_reduction_huge_array(self):
        # Regression test for gh-20921 (copying identity incorrectly failed)
        arr = np.zeros((2, 2**31), 'uint8')
        arr[:, 0] = [1, 3]
        arr[:, -1] = [4, 1]
        res = np.maximum.reduce(arr, axis=0)
        del arr
        assert res[0] == 3
        assert res[-1] == 4

    def test_reduce_identity_depends_on_loop(self):
        """
        The type of the result should always depend on the selected loop, not
        necessarily the output (only relevant for object arrays).
        """
        # For an object loop, the default value 0 with type int is used:
        assert type(np.add.reduce([], dtype=object)) is int
        out = np.array(None, dtype=object)
        # When the loop is float64 but `out` is object this does not happen,
        # the result is float64 cast to object (which gives Python `float`).
        np.add.reduce([], out=out, dtype=np.float64)
        assert type(out[()]) is float

    def test_initial_reduction(self):
        # np.minimum.reduce is an identityless reduction

        # For cases like np.maximum(np.abs(...), initial=0)
        # More generally, a supremum over non-negative numbers.
        assert_equal(np.maximum.reduce([], initial=0), 0)

        # For cases like reduction of an empty array over the reals.
        assert_equal(np.minimum.reduce([], initial=np.inf), np.inf)
        assert_equal(np.maximum.reduce([], initial=-np.inf), -np.inf)

        # Random tests
        assert_equal(np.minimum.reduce([5], initial=4), 4)
        assert_equal(np.maximum.reduce([4], initial=5), 5)
        assert_equal(np.maximum.reduce([5], initial=4), 5)
        assert_equal(np.minimum.reduce([4], initial=5), 4)

        # Check initial=None raises ValueError for both types of ufunc reductions
        assert_raises(ValueError, np.minimum.reduce, [], initial=None)
        assert_raises(ValueError, np.add.reduce, [], initial=None)
        # Also in the somewhat special object case:
        with pytest.raises(ValueError):
            np.add.reduce([], initial=None, dtype=object)

        # Check that np._NoValue gives default behavior.
        assert_equal(np.add.reduce([], initial=np._NoValue), 0)

        # Check that initial kwarg behaves as intended for dtype=object
        a = np.array([10], dtype=object)
        res = np.add.reduce(a, initial=5)
        assert_equal(res, 15)

    def test_empty_reduction_and_identity(self):
        arr = np.zeros((0, 5))
        # OK, since the reduction itself is *not* empty, the result is
        assert np.true_divide.reduce(arr, axis=1).shape == (0,)
        # Not OK, the reduction itself is empty and we have no identity
        with pytest.raises(ValueError):
            np.true_divide.reduce(arr, axis=0)

        # Test that an empty reduction fails also if the result is empty
        arr = np.zeros((0, 0, 5))
        with pytest.raises(ValueError):
            np.true_divide.reduce(arr, axis=1)

        # Division reduction makes sense with `initial=1` (empty or not):
        res = np.true_divide.reduce(arr, axis=1, initial=1)
        assert_array_equal(res, np.ones((0, 5)))

    @pytest.mark.parametrize('axis', (0, 1, None))
    @pytest.mark.parametrize('where', (np.array([False, True, True]),
                                       np.array([[True], [False], [True]]),
                                       np.array([[True, False, False],
                                                 [False, True, False],
                                                 [False, True, True]])))
    def test_reduction_with_where(self, axis, where):
        a = np.arange(9.).reshape(3, 3)
        a_copy = a.copy()
        a_check = np.zeros_like(a)
        np.positive(a, out=a_check, where=where)

        res = np.add.reduce(a, axis=axis, where=where)
        check = a_check.sum(axis)
        assert_equal(res, check)
        # Check we do not overwrite elements of a internally.
        assert_array_equal(a, a_copy)

    @pytest.mark.parametrize(('axis', 'where'),
                             ((0, np.array([True, False, True])),
                              (1, [True, True, False]),
                              (None, True)))
    @pytest.mark.parametrize('initial', (-np.inf, 5.))
    def test_reduction_with_where_and_initial(self, axis, where, initial):
        a = np.arange(9.).reshape(3, 3)
        a_copy = a.copy()
        a_check = np.full(a.shape, -np.inf)
        np.positive(a, out=a_check, where=where)

        res = np.maximum.reduce(a, axis=axis, where=where, initial=initial)
        check = a_check.max(axis, initial=initial)
        assert_equal(res, check)

    def test_reduction_where_initial_needed(self):
        a = np.arange(9.).reshape(3, 3)
        m = [False, True, False]
        assert_raises(ValueError, np.maximum.reduce, a, where=m)

    def test_identityless_reduction_nonreorderable(self):
        a = np.array([[8.0, 2.0, 2.0], [1.0, 0.5, 0.25]])

        res = np.divide.reduce(a, axis=0)
        assert_equal(res, [8.0, 4.0, 8.0])

        res = np.divide.reduce(a, axis=1)
        assert_equal(res, [2.0, 8.0])

        res = np.divide.reduce(a, axis=())
        assert_equal(res, a)

        assert_raises(ValueError, np.divide.reduce, a, axis=(0, 1))

    def test_reduce_zero_axis(self):
        # If we have a n x m array and do a reduction with axis=1, then we are
        # doing n reductions, and each reduction takes an m-element array. For
        # a reduction operation without an identity, then:
        #   n > 0, m > 0: fine
        #   n = 0, m > 0: fine, doing 0 reductions of m-element arrays
        #   n > 0, m = 0: can't reduce a 0-element array, ValueError
        #   n = 0, m = 0: can't reduce a 0-element array, ValueError (for
        #     consistency with the above case)
        # This test doesn't actually look at return values, it just checks to
        # make sure that error we get an error in exactly those cases where we
        # expect one, and assumes the calculations themselves are done
        # correctly.

        def ok(f, *args, **kwargs):
            f(*args, **kwargs)

        def err(f, *args, **kwargs):
            assert_raises(ValueError, f, *args, **kwargs)

        def t(expect, func, n, m):
            expect(func, np.zeros((n, m)), axis=1)
            expect(func, np.zeros((m, n)), axis=0)
            expect(func, np.zeros((n // 2, n // 2, m)), axis=2)
            expect(func, np.zeros((n // 2, m, n // 2)), axis=1)
            expect(func, np.zeros((n, m // 2, m // 2)), axis=(1, 2))
            expect(func, np.zeros((m // 2, n, m // 2)), axis=(0, 2))
            expect(func, np.zeros((m // 3, m // 3, m // 3,
                                  n // 2, n // 2)),
                                 axis=(0, 1, 2))
            # Check what happens if the inner (resp. outer) dimensions are a
            # mix of zero and non-zero:
            expect(func, np.zeros((10, m, n)), axis=(0, 1))
            expect(func, np.zeros((10, n, m)), axis=(0, 2))
            expect(func, np.zeros((m, 10, n)), axis=0)
            expect(func, np.zeros((10, m, n)), axis=1)
            expect(func, np.zeros((10, n, m)), axis=2)

        # np.maximum is just an arbitrary ufunc with no reduction identity
        assert_equal(np.maximum.identity, None)
        t(ok, np.maximum.reduce, 30, 30)
        t(ok, np.maximum.reduce, 0, 30)
        t(err, np.maximum.reduce, 30, 0)
        t(err, np.maximum.reduce, 0, 0)
        err(np.maximum.reduce, [])
        np.maximum.reduce(np.zeros((0, 0)), axis=())

        # all of the combinations are fine for a reduction that has an
        # identity
        t(ok, np.add.reduce, 30, 30)
        t(ok, np.add.reduce, 0, 30)
        t(ok, np.add.reduce, 30, 0)
        t(ok, np.add.reduce, 0, 0)
        np.add.reduce([])
        np.add.reduce(np.zeros((0, 0)), axis=())

        # OTOH, accumulate always makes sense for any combination of n and m,
        # because it maps an m-element array to an m-element array. These
        # tests are simpler because accumulate doesn't accept multiple axes.
        for uf in (np.maximum, np.add):
            uf.accumulate(np.zeros((30, 0)), axis=0)
            uf.accumulate(np.zeros((0, 30)), axis=0)
            uf.accumulate(np.zeros((30, 30)), axis=0)
            uf.accumulate(np.zeros((0, 0)), axis=0)

    def test_safe_casting(self):
        # In old versions of numpy, in-place operations used the 'unsafe'
        # casting rules. In versions >= 1.10, 'same_kind' is the
        # default and an exception is raised instead of a warning.
        # when 'same_kind' is not satisfied.
        a = np.array([1, 2, 3], dtype=int)
        # Non-in-place addition is fine
        assert_array_equal(assert_no_warnings(np.add, a, 1.1),
                           [2.1, 3.1, 4.1])
        assert_raises(TypeError, np.add, a, 1.1, out=a)

        def add_inplace(a, b):
            a += b

        assert_raises(TypeError, add_inplace, a, 1.1)
        # Make sure that explicitly overriding the exception is allowed:
        assert_no_warnings(np.add, a, 1.1, out=a, casting="unsafe")
        assert_array_equal(a, [2, 3, 4])

    def test_ufunc_custom_out(self):
        # Test ufunc with built in input types and custom output type

        a = np.array([0, 1, 2], dtype='i8')
        b = np.array([0, 1, 2], dtype='i8')
        c = np.empty(3, dtype=_rational_tests.rational)

        # Output must be specified so numpy knows what
        # ufunc signature to look for
        result = _rational_tests.test_add(a, b, c)
        target = np.array([0, 2, 4], dtype=_rational_tests.rational)
        assert_equal(result, target)

        # The new resolution means that we can (usually) find custom loops
        # as long as they match exactly:
        result = _rational_tests.test_add(a, b)
        assert_equal(result, target)

        # This works even more generally, so long the default common-dtype
        # promoter works out:
        result = _rational_tests.test_add(a, b.astype(np.uint16), out=c)
        assert_equal(result, target)

        # This scalar path used to go into legacy promotion, but doesn't now:
        result = _rational_tests.test_add(a, np.uint16(2))
        target = np.array([2, 3, 4], dtype=_rational_tests.rational)
        assert_equal(result, target)

    def test_operand_flags(self):
        a = np.arange(16, dtype=int).reshape(4, 4)
        b = np.arange(9, dtype=int).reshape(3, 3)
        opflag_tests.inplace_add(a[:-1, :-1], b)
        assert_equal(a, np.array([[0, 2, 4, 3], [7, 9, 11, 7],
            [14, 16, 18, 11], [12, 13, 14, 15]]))

        a = np.array(0)
        opflag_tests.inplace_add(a, 3)
        assert_equal(a, 3)
        opflag_tests.inplace_add(a, [3, 4])
        assert_equal(a, 10)

    def test_struct_ufunc(self):
        import numpy._core._struct_ufunc_tests as struct_ufunc

        a = np.array([(1, 2, 3)], dtype='u8,u8,u8')
        b = np.array([(1, 2, 3)], dtype='u8,u8,u8')

        result = struct_ufunc.add_triplet(a, b)
        assert_equal(result, np.array([(2, 4, 6)], dtype='u8,u8,u8'))
        assert_raises(RuntimeError, struct_ufunc.register_fail)

    def test_custom_ufunc(self):
        a = np.array(
            [_rational_tests.rational(1, 2),
             _rational_tests.rational(1, 3),
             _rational_tests.rational(1, 4)],
            dtype=_rational_tests.rational)
        b = np.array(
            [_rational_tests.rational(1, 2),
             _rational_tests.rational(1, 3),
             _rational_tests.rational(1, 4)],
            dtype=_rational_tests.rational)

        result = _rational_tests.test_add_rationals(a, b)
        expected = np.array(
            [_rational_tests.rational(1),
             _rational_tests.rational(2, 3),
             _rational_tests.rational(1, 2)],
            dtype=_rational_tests.rational)
        assert_equal(result, expected)

    def test_custom_ufunc_forced_sig(self):
        # gh-9351 - looking for a non-first userloop would previously hang
        with assert_raises(TypeError):
            np.multiply(_rational_tests.rational(1), 1,
                        signature=(_rational_tests.rational, int, None))

    def test_custom_array_like(self):

        class MyThing:
            __array_priority__ = 1000

            rmul_count = 0
            getitem_count = 0

            def __init__(self, shape):
                self.shape = shape

            def __len__(self):
                return self.shape[0]

            def __getitem__(self, i):
                MyThing.getitem_count += 1
                if not isinstance(i, tuple):
                    i = (i,)
                if len(i) > self.ndim:
                    raise IndexError("boo")

                return MyThing(self.shape[len(i):])

            def __rmul__(self, other):
                MyThing.rmul_count += 1
                return self

        np.float64(5) * MyThing((3, 3))
        assert_(MyThing.rmul_count == 1, MyThing.rmul_count)
        assert_(MyThing.getitem_count <= 2, MyThing.getitem_count)

    def test_array_wrap_array_priority(self):
        class ArrayPriorityBase(np.ndarray):
            @classmethod
            def __array_wrap__(cls, array, context=None, return_scalar=False):
                return cls

        class ArrayPriorityMinus0(ArrayPriorityBase):
            __array_priority__ = 0

        class ArrayPriorityMinus1000(ArrayPriorityBase):
            __array_priority__ = -1000

        class ArrayPriorityMinus1000b(ArrayPriorityBase):
            __array_priority__ = -1000

        class ArrayPriorityMinus2000(ArrayPriorityBase):
            __array_priority__ = -2000

        x = ArrayPriorityMinus1000(2)
        xb = ArrayPriorityMinus1000b(2)
        y = ArrayPriorityMinus2000(2)

        assert np.add(x, y) is ArrayPriorityMinus1000
        assert np.add(y, x) is ArrayPriorityMinus1000
        assert np.add(x, xb) is ArrayPriorityMinus1000
        assert np.add(xb, x) is ArrayPriorityMinus1000b
        assert np.add(np.zeros(2), ArrayPriorityMinus0(2)) is ArrayPriorityMinus0
        assert type(np.add(xb, x, np.zeros(2))) is np.ndarray

    @pytest.mark.parametrize("a", (
                             np.arange(10, dtype=int),
                             np.arange(10, dtype=_rational_tests.rational),
                             ))
    def test_ufunc_at_basic(self, a):

        aa = a.copy()
        np.add.at(aa, [2, 5, 2], 1)
        assert_equal(aa, [0, 1, 4, 3, 4, 6, 6, 7, 8, 9])

        with pytest.raises(ValueError):
            # missing second operand
            np.add.at(aa, [2, 5, 3])

        aa = a.copy()
        np.negative.at(aa, [2, 5, 3])
        assert_equal(aa, [0, 1, -2, -3, 4, -5, 6, 7, 8, 9])

        aa = a.copy()
        b = np.array([100, 100, 100])
        np.add.at(aa, [2, 5, 2], b)
        assert_equal(aa, [0, 1, 202, 3, 4, 105, 6, 7, 8, 9])

        with pytest.raises(ValueError):
            # extraneous second operand
            np.negative.at(a, [2, 5, 3], [1, 2, 3])

        with pytest.raises(ValueError):
            # second operand cannot be converted to an array
            np.add.at(a, [2, 5, 3], [[1, 2], 1])

    # ufuncs with indexed loops for performance in ufunc.at
    indexed_ufuncs = [np.add, np.subtract, np.multiply, np.floor_divide,
                      np.maximum, np.minimum, np.fmax, np.fmin]

    @pytest.mark.parametrize(
                "typecode", np.typecodes['AllInteger'] + np.typecodes['Float'])
    @pytest.mark.parametrize("ufunc", indexed_ufuncs)
    def test_ufunc_at_inner_loops(self, typecode, ufunc):
        if ufunc is np.divide and typecode in np.typecodes['AllInteger']:
            # Avoid divide-by-zero and inf for integer divide
            a = np.ones(100, dtype=typecode)
            indx = np.random.randint(100, size=30, dtype=np.intp)
            vals = np.arange(1, 31, dtype=typecode)
        else:
            a = np.ones(1000, dtype=typecode)
            indx = np.random.randint(1000, size=3000, dtype=np.intp)
            vals = np.arange(3000, dtype=typecode)
        atag = a.copy()
        # Do the calculation twice and compare the answers
        with warnings.catch_warnings(record=True) as w_at:
            warnings.simplefilter('always')
            ufunc.at(a, indx, vals)
        with warnings.catch_warnings(record=True) as w_loop:
            warnings.simplefilter('always')
            for i, v in zip(indx, vals):
                # Make sure all the work happens inside the ufunc
                # in order to duplicate error/warning handling
                ufunc(atag[i], v, out=atag[i:i + 1], casting="unsafe")
        assert_equal(atag, a)
        # If w_loop warned, make sure w_at warned as well
        if len(w_loop) > 0:
            #
            assert len(w_at) > 0
            assert w_at[0].category == w_loop[0].category
            assert str(w_at[0].message)[:10] == str(w_loop[0].message)[:10]

    @pytest.mark.parametrize("typecode", np.typecodes['Complex'])
    @pytest.mark.parametrize("ufunc", [np.add, np.subtract, np.multiply])
    def test_ufunc_at_inner_loops_complex(self, typecode, ufunc):
        a = np.ones(10, dtype=typecode)
        indx = np.concatenate([np.ones(6, dtype=np.intp),
                               np.full(18, 4, dtype=np.intp)])
        value = a.dtype.type(1j)
        ufunc.at(a, indx, value)
        expected = np.ones_like(a)
        if ufunc is np.multiply:
            expected[1] = expected[4] = -1
        else:
            expected[1] += 6 * (value if ufunc is np.add else -value)
            expected[4] += 18 * (value if ufunc is np.add else -value)

        assert_array_equal(a, expected)

    def test_ufunc_at_ellipsis(self):
        # Make sure the indexed loop check does not choke on iters
        # with subspaces
        arr = np.zeros(5)
        np.add.at(arr, slice(None), np.ones(5))
        assert_array_equal(arr, np.ones(5))

    def test_ufunc_at_negative(self):
        arr = np.ones(5, dtype=np.int32)
        indx = np.arange(5)
        umt.indexed_negative.at(arr, indx)
        # If it is [-1, -1, -1, -100, 0] then the regular strided loop was used
        assert np.all(arr == [-1, -1, -1, -200, -1])

    def test_ufunc_at_large(self):
        # issue gh-23457
        indices = np.zeros(8195, dtype=np.int16)
        b = np.zeros(8195, dtype=float)
        b[0] = 10
        b[1] = 5
        b[8192:] = 100
        a = np.zeros(1, dtype=float)
        np.add.at(a, indices, b)
        assert a[0] == b.sum()

    def test_cast_index_fastpath(self):
        arr = np.zeros(10)
        values = np.ones(100000)
        # index must be cast, which may be buffered in chunks:
        index = np.zeros(len(values), dtype=np.uint8)
        np.add.at(arr, index, values)
        assert arr[0] == len(values)

    @pytest.mark.parametrize("value", [
        np.ones(1), np.ones(()), np.float64(1.), 1.])
    def test_ufunc_at_scalar_value_fastpath(self, value):
        arr = np.zeros(1000)
        # index must be cast, which may be buffered in chunks:
        index = np.repeat(np.arange(1000), 2)
        np.add.at(arr, index, value)
        assert_array_equal(arr, np.full_like(arr, 2 * value))

    def test_ufunc_at_multiD(self):
        a = np.arange(9).reshape(3, 3)
        b = np.array([[100, 100, 100], [200, 200, 200], [300, 300, 300]])
        np.add.at(a, (slice(None), [1, 2, 1]), b)
        assert_equal(a, [[0, 201, 102], [3, 404, 205], [6, 607, 308]])

        a = np.arange(27).reshape(3, 3, 3)
        b = np.array([100, 200, 300])
        np.add.at(a, (slice(None), slice(None), [1, 2, 1]), b)
        assert_equal(a,
            [[[0, 401, 202],
              [3, 404, 205],
              [6, 407, 208]],

             [[9, 410, 211],
              [12, 413, 214],
              [15, 416, 217]],

             [[18, 419, 220],
              [21, 422, 223],
              [24, 425, 226]]])

        a = np.arange(9).reshape(3, 3)
        b = np.array([[100, 100, 100], [200, 200, 200], [300, 300, 300]])
        np.add.at(a, ([1, 2, 1], slice(None)), b)
        assert_equal(a, [[0, 1, 2], [403, 404, 405], [206, 207, 208]])

        a = np.arange(27).reshape(3, 3, 3)
        b = np.array([100, 200, 300])
        np.add.at(a, (slice(None), [1, 2, 1], slice(None)), b)
        assert_equal(a,
            [[[0,  1,  2],
              [203, 404, 605],
              [106, 207, 308]],

             [[9,  10, 11],
              [212, 413, 614],
              [115, 216, 317]],

             [[18, 19, 20],
              [221, 422, 623],
              [124, 225, 326]]])

        a = np.arange(9).reshape(3, 3)
        b = np.array([100, 200, 300])
        np.add.at(a, (0, [1, 2, 1]), b)
        assert_equal(a, [[0, 401, 202], [3, 4, 5], [6, 7, 8]])

        a = np.arange(27).reshape(3, 3, 3)
        b = np.array([100, 200, 300])
        np.add.at(a, ([1, 2, 1], 0, slice(None)), b)
        assert_equal(a,
            [[[0,  1,  2],
              [3,  4,  5],
              [6,  7,  8]],

             [[209, 410, 611],
              [12,  13, 14],
              [15,  16, 17]],

             [[118, 219, 320],
              [21,  22, 23],
              [24,  25, 26]]])

        a = np.arange(27).reshape(3, 3, 3)
        b = np.array([100, 200, 300])
        np.add.at(a, (slice(None), slice(None), slice(None)), b)
        assert_equal(a,
            [[[100, 201, 302],
              [103, 204, 305],
              [106, 207, 308]],

             [[109, 210, 311],
              [112, 213, 314],
              [115, 216, 317]],

             [[118, 219, 320],
              [121, 222, 323],
              [124, 225, 326]]])

    def test_ufunc_at_0D(self):
        a = np.array(0)
        np.add.at(a, (), 1)
        assert_equal(a, 1)

        assert_raises(IndexError, np.add.at, a, 0, 1)
        assert_raises(IndexError, np.add.at, a, [], 1)

    def test_ufunc_at_dtypes(self):
        # Test mixed dtypes
        a = np.arange(10)
        np.power.at(a, [1, 2, 3, 2], 3.5)
        assert_equal(a, np.array([0, 1, 4414, 46, 4, 5, 6, 7, 8, 9]))

    def test_ufunc_at_boolean(self):
        # Test boolean indexing and boolean ufuncs
        a = np.arange(10)
        index = a % 2 == 0
        np.equal.at(a, index, [0, 2, 4, 6, 8])
        assert_equal(a, [1, 1, 1, 3, 1, 5, 1, 7, 1, 9])

        # Test unary operator
        a = np.arange(10, dtype='u4')
        np.invert.at(a, [2, 5, 2])
        assert_equal(a, [0, 1, 2, 3, 4, 5 ^ 0xffffffff, 6, 7, 8, 9])

    def test_ufunc_at_advanced(self):
        # Test empty subspace
        orig = np.arange(4)
        a = orig[:, None][:, 0:0]
        np.add.at(a, [0, 1], 3)
        assert_array_equal(orig, np.arange(4))

        # Test with swapped byte order
        index = np.array([1, 2, 1], np.dtype('i').newbyteorder())
        values = np.array([1, 2, 3, 4], np.dtype('f').newbyteorder())
        np.add.at(values, index, 3)
        assert_array_equal(values, [1, 8, 6, 4])

        # Test exception thrown
        values = np.array(['a', 1], dtype=object)
        assert_raises(TypeError, np.add.at, values, [0, 1], 1)
        assert_array_equal(values, np.array(['a', 1], dtype=object))

        # Test multiple output ufuncs raise error, gh-5665
        assert_raises(ValueError, np.modf.at, np.arange(10), [1])

        # Test maximum
        a = np.array([1, 2, 3])
        np.maximum.at(a, [0], 0)
        assert_equal(a, np.array([1, 2, 3]))

    @pytest.mark.parametrize("dtype",
            np.typecodes['AllInteger'] + np.typecodes['Float'])
    @pytest.mark.parametrize("ufunc",
            [np.add, np.subtract, np.divide, np.minimum, np.maximum])
    def test_at_negative_indexes(self, dtype, ufunc):
        a = np.arange(0, 10).astype(dtype)
        indxs = np.array([-1, 1, -1, 2]).astype(np.intp)
        vals = np.array([1, 5, 2, 10], dtype=a.dtype)

        expected = a.copy()
        for i, v in zip(indxs, vals):
            expected[i] = ufunc(expected[i], v)

        ufunc.at(a, indxs, vals)
        assert_array_equal(a, expected)
        assert np.all(indxs == [-1, 1, -1, 2])

    def test_at_not_none_signature(self):
        # Test ufuncs with non-trivial signature raise a TypeError
        a = np.ones((2, 2, 2))
        b = np.ones((1, 2, 2))
        assert_raises(TypeError, np.matmul.at, a, [0], b)

        a = np.array([[[1, 2], [3, 4]]])
        assert_raises(TypeError, np.linalg._umath_linalg.det.at, a, [0])

    def test_at_no_loop_for_op(self):
        # str dtype does not have a ufunc loop for np.add
        arr = np.ones(10, dtype=str)
        with pytest.raises(np._core._exceptions._UFuncNoLoopError):
            np.add.at(arr, [0, 1], [0, 1])

    def test_at_output_casting(self):
        arr = np.array([-1])
        np.equal.at(arr, [0], [0])
        assert arr[0] == 0

    def test_at_broadcast_failure(self):
        arr = np.arange(5)
        with pytest.raises(ValueError):
            np.add.at(arr, [0, 1], [1, 2, 3])

    def test_reduce_arguments(self):
        f = np.add.reduce
        d = np.ones((5, 2), dtype=int)
        o = np.ones((2,), dtype=d.dtype)
        r = o * 5
        assert_equal(f(d), r)
        # a, axis=0, dtype=None, out=None, keepdims=False
        assert_equal(f(d, axis=0), r)
        assert_equal(f(d, 0), r)
        assert_equal(f(d, 0, dtype=None), r)
        assert_equal(f(d, 0, dtype='i'), r)
        assert_equal(f(d, 0, 'i'), r)
        assert_equal(f(d, 0, None), r)
        assert_equal(f(d, 0, None, out=None), r)
        assert_equal(f(d, 0, None, out=o), r)
        assert_equal(f(d, 0, None, o), r)
        assert_equal(f(d, 0, None, None), r)
        assert_equal(f(d, 0, None, None, keepdims=False), r)
        assert_equal(f(d, 0, None, None, True), r.reshape((1,) + r.shape))
        assert_equal(f(d, 0, None, None, False, 0), r)
        assert_equal(f(d, 0, None, None, False, initial=0), r)
        assert_equal(f(d, 0, None, None, False, 0, True), r)
        assert_equal(f(d, 0, None, None, False, 0, where=True), r)
        # multiple keywords
        assert_equal(f(d, axis=0, dtype=None, out=None, keepdims=False), r)
        assert_equal(f(d, 0, dtype=None, out=None, keepdims=False), r)
        assert_equal(f(d, 0, None, out=None, keepdims=False), r)
        assert_equal(f(d, 0, None, out=None, keepdims=False, initial=0,
                       where=True), r)

        # too little
        assert_raises(TypeError, f)
        # too much
        assert_raises(TypeError, f, d, 0, None, None, False, 0, True, 1)
        # invalid axis
        assert_raises(TypeError, f, d, "invalid")
        assert_raises(TypeError, f, d, axis="invalid")
        assert_raises(TypeError, f, d, axis="invalid", dtype=None,
                      keepdims=True)
        # invalid dtype
        assert_raises(TypeError, f, d, 0, "invalid")
        assert_raises(TypeError, f, d, dtype="invalid")
        assert_raises(TypeError, f, d, dtype="invalid", out=None)
        # invalid out
        assert_raises(TypeError, f, d, 0, None, "invalid")
        assert_raises(TypeError, f, d, out="invalid")
        assert_raises(TypeError, f, d, out="invalid", dtype=None)
        # keepdims boolean, no invalid value
        # assert_raises(TypeError, f, d, 0, None, None, "invalid")
        # assert_raises(TypeError, f, d, keepdims="invalid", axis=0, dtype=None)
        # invalid mix
        assert_raises(TypeError, f, d, 0, keepdims="invalid", dtype="invalid",
                     out=None)

        # invalid keyword
        assert_raises(TypeError, f, d, axis=0, dtype=None, invalid=0)
        assert_raises(TypeError, f, d, invalid=0)
        assert_raises(TypeError, f, d, 0, keepdims=True, invalid="invalid",
                      out=None)
        assert_raises(TypeError, f, d, axis=0, dtype=None, keepdims=True,
                      out=None, invalid=0)
        assert_raises(TypeError, f, d, axis=0, dtype=None,
                      out=None, invalid=0)

    def test_structured_equal(self):
        # https://github.com/numpy/numpy/issues/4855

        class MyA(np.ndarray):
            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                return getattr(ufunc, method)(*(input.view(np.ndarray)
                                              for input in inputs), **kwargs)
        a = np.arange(12.).reshape(4, 3)
        ra = a.view(dtype=('f8,f8,f8')).squeeze()
        mra = ra.view(MyA)

        target = np.array([True, False, False, False], dtype=bool)
        assert_equal(np.all(target == (mra == ra[0])), True)

    def test_scalar_equal(self):
        # Scalar comparisons should always work, without deprecation warnings.
        # even when the ufunc fails.
        a = np.array(0.)
        b = np.array('a')
        assert_(a != b)
        assert_(b != a)
        assert_(not (a == b))
        assert_(not (b == a))

    def test_NotImplemented_not_returned(self):
        # See gh-5964 and gh-2091. Some of these functions are not operator
        # related and were fixed for other reasons in the past.
        binary_funcs = [
            np.power, np.add, np.subtract, np.multiply, np.divide,
            np.true_divide, np.floor_divide, np.bitwise_and, np.bitwise_or,
            np.bitwise_xor, np.left_shift, np.right_shift, np.fmax,
            np.fmin, np.fmod, np.hypot, np.logaddexp, np.logaddexp2,
            np.maximum, np.minimum, np.mod,
            np.greater, np.greater_equal, np.less, np.less_equal,
            np.equal, np.not_equal]

        a = np.array('1')
        b = 1
        c = np.array([1., 2.])
        for f in binary_funcs:
            assert_raises(TypeError, f, a, b)
            assert_raises(TypeError, f, c, a)

    @pytest.mark.parametrize("ufunc",
             [np.logical_and, np.logical_or])  # logical_xor object loop is bad
    @pytest.mark.parametrize("signature",
             [(None, None, object), (object, None, None),
              (None, object, None)])
    def test_logical_ufuncs_object_signatures(self, ufunc, signature):
        a = np.array([True, None, False], dtype=object)
        res = ufunc(a, a, signature=signature)
        assert res.dtype == object

    @pytest.mark.parametrize("ufunc",
            [np.logical_and, np.logical_or, np.logical_xor])
    @pytest.mark.parametrize("signature",
                 [(bool, None, object), (object, None, bool),
                  (None, object, bool)])
    def test_logical_ufuncs_mixed_object_signatures(self, ufunc, signature):
        # Most mixed signatures fail (except those with bool out, e.g. `OO->?`)
        a = np.array([True, None, False])
        with pytest.raises(TypeError):
            ufunc(a, a, signature=signature)

    @pytest.mark.parametrize("ufunc",
            [np.logical_and, np.logical_or, np.logical_xor])
    def test_logical_ufuncs_support_anything(self, ufunc):
        # The logical ufuncs support even input that can't be promoted:
        a = np.array(b'1', dtype="V3")
        c = np.array([1., 2.])
        assert_array_equal(ufunc(a, c), ufunc([True, True], True))
        assert ufunc.reduce(a) == True
        # check that the output has no effect:
        out = np.zeros(2, dtype=np.int32)
        expected = ufunc([True, True], True).astype(out.dtype)
        assert_array_equal(ufunc(a, c, out=out), expected)
        out = np.zeros((), dtype=np.int32)
        assert ufunc.reduce(a, out=out) == True
        # Last check, test reduction when out and a match (the complexity here
        # is that the "i,i->?" may seem right, but should not match.
        a = np.array([3], dtype="i")
        out = np.zeros((), dtype=a.dtype)
        assert ufunc.reduce(a, out=out) == 1

    @pytest.mark.parametrize("ufunc",
            [np.logical_and, np.logical_or, np.logical_xor])
    @pytest.mark.parametrize("dtype", ["S", "U"])
    @pytest.mark.parametrize("values", [["1", "hi", "0"], ["", ""]])
    def test_logical_ufuncs_supports_string(self, ufunc, dtype, values):
        # note that values are either all true or all false
        arr = np.array(values, dtype=dtype)
        obj_arr = np.array(values, dtype=object)
        res = ufunc(arr, arr)
        expected = ufunc(obj_arr, obj_arr, dtype=bool)

        assert_array_equal(res, expected)

        res = ufunc.reduce(arr)
        expected = ufunc.reduce(obj_arr, dtype=bool)
        assert_array_equal(res, expected)

    @pytest.mark.parametrize("ufunc",
             [np.logical_and, np.logical_or, np.logical_xor])
    def test_logical_ufuncs_out_cast_check(self, ufunc):
        a = np.array('1')
        c = np.array([1., 2.])
        out = a.copy()
        with pytest.raises(TypeError):
            # It would be safe, but not equiv casting:
            ufunc(a, c, out=out, casting="equiv")

    def test_reducelike_byteorder_resolution(self):
        # See gh-20699, byte-order changes need some extra care in the type
        # resolution to make the following succeed:
        arr_be = np.arange(10, dtype=">i8")
        arr_le = np.arange(10, dtype="<i8")

        assert np.add.reduce(arr_be) == np.add.reduce(arr_le)
        assert_array_equal(np.add.accumulate(arr_be), np.add.accumulate(arr_le))
        assert_array_equal(
            np.add.reduceat(arr_be, [1]), np.add.reduceat(arr_le, [1]))

    def test_reducelike_out_promotes(self):
        # Check that the out argument to reductions is considered for
        # promotion.  See also gh-20455.
        # Note that these paths could prefer `initial=` in the future and
        # do not up-cast to the default integer for add and prod
        arr = np.ones(1000, dtype=np.uint8)
        out = np.zeros((), dtype=np.uint16)
        assert np.add.reduce(arr, out=out) == 1000
        arr[:10] = 2
        assert np.multiply.reduce(arr, out=out) == 2**10

        # For legacy dtypes, the signature currently has to be forced if `out=`
        # is passed.  The two paths below should differ, without `dtype=` the
        # expected result should be: `np.prod(arr.astype("f8")).astype("f4")`!
        arr = np.full(5, 2**25 - 1, dtype=np.int64)

        # float32 and int64 promote to float64:
        res = np.zeros((), dtype=np.float32)
        # If `dtype=` is passed, the calculation is forced to float32:
        single_res = np.zeros((), dtype=np.float32)
        np.multiply.reduce(arr, out=single_res, dtype=np.float32)
        assert single_res != res

    def test_reducelike_output_needs_identical_cast(self):
        # Checks the case where a simple byte-swap works, mainly tests that
        # this is not rejected directly.
        # (interesting because we require descriptor identity in reducelikes).
        arr = np.ones(20, dtype="f8")
        out = np.empty((), dtype=arr.dtype.newbyteorder())
        expected = np.add.reduce(arr)
        np.add.reduce(arr, out=out)
        assert_array_equal(expected, out)
        # Check reduceat:
        out = np.empty(2, dtype=arr.dtype.newbyteorder())
        expected = np.add.reduceat(arr, [0, 1])
        np.add.reduceat(arr, [0, 1], out=out)
        assert_array_equal(expected, out)
        # And accumulate:
        out = np.empty(arr.shape, dtype=arr.dtype.newbyteorder())
        expected = np.add.accumulate(arr)
        np.add.accumulate(arr, out=out)
        assert_array_equal(expected, out)

    def test_reduce_noncontig_output(self):
        # Check that reduction deals with non-contiguous output arrays
        # appropriately.
        #
        # gh-8036

        x = np.arange(7 * 13 * 8, dtype=np.int16).reshape(7, 13, 8)
        x = x[4:6, 1:11:6, 1:5].transpose(1, 2, 0)
        y_base = np.arange(4 * 4, dtype=np.int16).reshape(4, 4)
        y = y_base[::2, :]

        y_base_copy = y_base.copy()

        r0 = np.add.reduce(x, out=y.copy(), axis=2)
        r1 = np.add.reduce(x, out=y, axis=2)

        # The results should match, and y_base shouldn't get clobbered
        assert_equal(r0, r1)
        assert_equal(y_base[1, :], y_base_copy[1, :])
        assert_equal(y_base[3, :], y_base_copy[3, :])

    @pytest.mark.parametrize("with_cast", [True, False])
    def test_reduceat_and_accumulate_out_shape_mismatch(self, with_cast):
        # Should raise an error mentioning "shape" or "size"
        arr = np.arange(5)
        out = np.arange(3)  # definitely wrong shape
        if with_cast:
            # If a cast is necessary on the output, we can be sure to use
            # the generic NpyIter (non-fast) path.
            out = out.astype(np.float64)

        with pytest.raises(ValueError, match="(shape|size)"):
            np.add.reduceat(arr, [0, 3], out=out)

        with pytest.raises(ValueError, match="(shape|size)"):
            np.add.accumulate(arr, out=out)

    @pytest.mark.parametrize('out_shape',
                             [(), (1,), (3,), (1, 1), (1, 3), (4, 3)])
    @pytest.mark.parametrize('keepdims', [True, False])
    @pytest.mark.parametrize('f_reduce', [np.add.reduce, np.minimum.reduce])
    def test_reduce_wrong_dimension_output(self, f_reduce, keepdims, out_shape):
        # Test that we're not incorrectly broadcasting dimensions.
        # See gh-15144 (failed for np.add.reduce previously).
        a = np.arange(12.).reshape(4, 3)
        out = np.empty(out_shape, a.dtype)

        correct_out = f_reduce(a, axis=0, keepdims=keepdims)
        if out_shape != correct_out.shape:
            with assert_raises(ValueError):
                f_reduce(a, axis=0, out=out, keepdims=keepdims)
        else:
            check = f_reduce(a, axis=0, out=out, keepdims=keepdims)
            assert_(check is out)
            assert_array_equal(check, correct_out)

    def test_reduce_output_does_not_broadcast_input(self):
        # Test that the output shape cannot broadcast an input dimension
        # (it never can add dimensions, but it might expand an existing one)
        a = np.ones((1, 10))
        out_correct = (np.empty((1, 1)))
        out_incorrect = np.empty((3, 1))
        np.add.reduce(a, axis=-1, out=out_correct, keepdims=True)
        np.add.reduce(a, axis=-1, out=out_correct[:, 0], keepdims=False)
        with assert_raises(ValueError):
            np.add.reduce(a, axis=-1, out=out_incorrect, keepdims=True)
        with assert_raises(ValueError):
            np.add.reduce(a, axis=-1, out=out_incorrect[:, 0], keepdims=False)

    def test_reduce_output_subclass_ok(self):
        class MyArr(np.ndarray):
            pass

        out = np.empty(())
        np.add.reduce(np.ones(5), out=out)  # no subclass, all fine
        out = out.view(MyArr)
        assert np.add.reduce(np.ones(5), out=out) is out
        assert type(np.add.reduce(out)) is MyArr

    def test_no_doc_string(self):
        # gh-9337
        assert_('\n' not in umt.inner1d_no_doc.__doc__)

    def test_invalid_args(self):
        # gh-7961
        exc = pytest.raises(TypeError, np.sqrt, None)
        # minimally check the exception text
        assert exc.match('loop of ufunc does not support')

    @pytest.mark.parametrize('nat', [np.datetime64('nat'), np.timedelta64('nat')])
    def test_nat_is_not_finite(self, nat):
        try:
            assert not np.isfinite(nat)
        except TypeError:
            pass  # ok, just not implemented

    @pytest.mark.parametrize('nat', [np.datetime64('nat'), np.timedelta64('nat')])
    def test_nat_is_nan(self, nat):
        try:
            assert np.isnan(nat)
        except TypeError:
            pass  # ok, just not implemented

    @pytest.mark.parametrize('nat', [np.datetime64('nat'), np.timedelta64('nat')])
    def test_nat_is_not_inf(self, nat):
        try:
            assert not np.isinf(nat)
        except TypeError:
            pass  # ok, just not implemented


class TestGUFuncProcessCoreDims:

    def test_conv1d_full_without_out(self):
        x = np.arange(5.0)
        y = np.arange(13.0)
        w = umt.conv1d_full(x, y)
        assert_equal(w, np.convolve(x, y, mode='full'))

    def test_conv1d_full_with_out(self):
        x = np.arange(5.0)
        y = np.arange(13.0)
        out = np.zeros(len(x) + len(y) - 1)
        umt.conv1d_full(x, y, out=out)
        assert_equal(out, np.convolve(x, y, mode='full'))

    def test_conv1d_full_basic_broadcast(self):
        # x.shape is (3, 6)
        x = np.array([[1, 3, 0, -10, 2, 2],
                      [0, -1, 2, 2, 10, 4],
                      [8, 9, 10, 2, 23, 3]])
        # y.shape is (2, 1, 7)
        y = np.array([[[3, 4, 5, 20, 30, 40, 29]],
                      [[5, 6, 7, 10, 11, 12, -5]]])
        # result should have shape (2, 3, 12)
        result = umt.conv1d_full(x, y)
        assert result.shape == (2, 3, 12)
        for i in range(2):
            for j in range(3):
                assert_equal(result[i, j], np.convolve(x[j], y[i, 0]))

    def test_bad_out_shape(self):
        x = np.ones((1, 2))
        y = np.ones((2, 3))
        out = np.zeros((2, 3))  # Not the correct shape.
        with pytest.raises(ValueError, match=r'does not equal m \+ n - 1'):
            umt.conv1d_full(x, y, out=out)

    def test_bad_input_both_inputs_length_zero(self):
        with pytest.raises(ValueError,
                           match='both inputs have core dimension 0'):
            umt.conv1d_full([], [])


@pytest.mark.parametrize('ufunc', [getattr(np, x) for x in dir(np)
                                   if isinstance(getattr(np, x), np.ufunc)])
def test_ufunc_types(ufunc):
    '''
    Check all ufuncs that the correct type is returned. Avoid
    object and boolean types since many operations are not defined for
    for them.

    Choose the shape so even dot and matmul will succeed
    '''
    for typ in ufunc.types:
        # types is a list of strings like ii->i
        if 'O' in typ or '?' in typ:
            continue
        inp, out = typ.split('->')
        args = [np.ones((3, 3), t) for t in inp]
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings("always")
            res = ufunc(*args)
        if isinstance(res, tuple):
            outs = tuple(out)
            assert len(res) == len(outs)
            for r, t in zip(res, outs):
                assert r.dtype == np.dtype(t)
        else:
            assert res.dtype == np.dtype(out)

@pytest.mark.parametrize('ufunc', [getattr(np, x) for x in dir(np)
                                if isinstance(getattr(np, x), np.ufunc)])
def test_ufunc_noncontiguous(ufunc):
    '''
    Check that contiguous and non-contiguous calls to ufuncs
    have the same results for values in range(9)
    '''
    for typ in ufunc.types:
        # types is a list of strings like ii->i
        if any(set('O?mM') & set(typ)):
            # bool, object, datetime are too irregular for this simple test
            continue
        inp, out = typ.split('->')
        args_c = [np.empty((6, 6), t) for t in inp]
        # non contiguous (2, 3 step on the two dimensions)
        args_n = [np.empty((12, 18), t)[::2, ::3] for t in inp]
        # alignment != itemsize is possible.  So create an array with such
        # an odd step manually.
        args_o = []
        for t in inp:
            orig_dt = np.dtype(t)
            off_dt = f"S{orig_dt.alignment}"  # offset by alignment
            dtype = np.dtype([("_", off_dt), ("t", orig_dt)], align=False)
            args_o.append(np.empty((6, 6), dtype=dtype)["t"])
        for a in args_c + args_n + args_o:
            a.flat = range(1, 37)

        with warnings.catch_warnings(record=True):
            warnings.filterwarnings("always")
            res_c = ufunc(*args_c)
            res_n = ufunc(*args_n)
            res_o = ufunc(*args_o)
        if len(out) == 1:
            res_c = (res_c,)
            res_n = (res_n,)
            res_o = (res_o,)
        for c_ar, n_ar, o_ar in zip(res_c, res_n, res_o):
            dt = c_ar.dtype
            if np.issubdtype(dt, np.floating):
                # for floating point results allow a small fuss in comparisons
                # since different algorithms (libm vs. intrinsics) can be used
                # for different input strides
                res_eps = np.finfo(dt).eps
                tol = 3 * res_eps
                assert_allclose(res_c, res_n, atol=tol, rtol=tol)
                assert_allclose(res_c, res_o, atol=tol, rtol=tol)
            else:
                assert_equal(c_ar, n_ar)
                assert_equal(c_ar, o_ar)


@pytest.mark.parametrize('ufunc', [np.sign, np.equal])
def test_ufunc_warn_with_nan(ufunc):
    # issue gh-15127
    # test that calling certain ufuncs with a non-standard `nan` value does not
    # emit a warning
    # `b` holds a 64 bit signaling nan: the most significant bit of the
    # significand is zero.
    b = np.array([0x7ff0000000000001], 'i8').view('f8')
    assert np.isnan(b)
    if ufunc.nin == 1:
        ufunc(b)
    elif ufunc.nin == 2:
        ufunc(b, b.copy())
    else:
        raise ValueError('ufunc with more than 2 inputs')


@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
def test_ufunc_out_casterrors():
    # Tests that casting errors are correctly reported and buffers are
    # cleared.
    # The following array can be added to itself as an object array, but
    # the result cannot be cast to an integer output:
    value = 123  # relies on python cache (leak-check will still find it)
    arr = np.array([value] * int(ncu.BUFSIZE * 1.5) +
                   ["string"] +
                   [value] * int(1.5 * ncu.BUFSIZE), dtype=object)
    out = np.ones(len(arr), dtype=np.intp)

    count = sys.getrefcount(value)
    with pytest.raises(ValueError):
        # Output casting failure:
        np.add(arr, arr, out=out, casting="unsafe")

    assert count == sys.getrefcount(value)
    # output is unchanged after the error, this shows that the iteration
    # was aborted (this is not necessarily defined behaviour)
    assert out[-1] == 1

    with pytest.raises(ValueError):
        # Input casting failure:
        np.add(arr, arr, out=out, dtype=np.intp, casting="unsafe")

    assert count == sys.getrefcount(value)
    # output is unchanged after the error, this shows that the iteration
    # was aborted (this is not necessarily defined behaviour)
    assert out[-1] == 1


@pytest.mark.parametrize("bad_offset", [0, int(ncu.BUFSIZE * 1.5)])
def test_ufunc_input_casterrors(bad_offset):
    value = 123
    arr = np.array([value] * bad_offset +
                   ["string"] +
                   [value] * int(1.5 * ncu.BUFSIZE), dtype=object)
    with pytest.raises(ValueError):
        # Force cast inputs, but the buffered cast of `arr` to intp fails:
        np.add(arr, arr, dtype=np.intp, casting="unsafe")


@pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
@pytest.mark.parametrize("bad_offset", [0, int(ncu.BUFSIZE * 1.5)])
def test_ufunc_input_floatingpoint_error(bad_offset):
    value = 123
    arr = np.array([value] * bad_offset +
                   [np.nan] +
                   [value] * int(1.5 * ncu.BUFSIZE))
    with np.errstate(invalid="raise"), pytest.raises(FloatingPointError):
        # Force cast inputs, but the buffered cast of `arr` to intp fails:
        np.add(arr, arr, dtype=np.intp, casting="unsafe")


def test_trivial_loop_invalid_cast():
    # This tests the fast-path "invalid cast", see gh-19904.
    with pytest.raises(TypeError,
            match="cast ufunc 'add' input 0"):
        # the void dtype definitely cannot cast to double:
        np.add(np.array(1, "i,i"), 3, signature="dd->d")


@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
@pytest.mark.parametrize("offset",
        [0, ncu.BUFSIZE // 2, int(1.5 * ncu.BUFSIZE)])
def test_reduce_casterrors(offset):
    # Test reporting of casting errors in reductions, we test various
    # offsets to where the casting error will occur, since these may occur
    # at different places during the reduction procedure. For example
    # the first item may be special.
    value = 123  # relies on python cache (leak-check will still find it)
    arr = np.array([value] * offset +
                   ["string"] +
                   [value] * int(1.5 * ncu.BUFSIZE), dtype=object)
    out = np.array(-1, dtype=np.intp)

    count = sys.getrefcount(value)
    with pytest.raises(ValueError, match="invalid literal"):
        # This is an unsafe cast, but we currently always allow that.
        # Note that the double loop is picked, but the cast fails.
        # `initial=None` disables the use of an identity here to test failures
        # while copying the first values path (not used when identity exists).
        np.add.reduce(arr, dtype=np.intp, out=out, initial=None)
    assert count == sys.getrefcount(value)
    # If an error occurred during casting, the operation is done at most until
    # the error occurs (the result of which would be `value * offset`) and -1
    # if the error happened immediately.
    # This does not define behaviour, the output is invalid and thus undefined
    assert out[()] < value * offset


def test_object_reduce_cleanup_on_failure():
    # Test cleanup, including of the initial value (manually provided or not)
    with pytest.raises(TypeError):
        np.add.reduce([1, 2, None], initial=4)

    with pytest.raises(TypeError):
        np.add.reduce([1, 2, None])


@pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
@pytest.mark.parametrize("method",
        [np.add.accumulate, np.add.reduce,
         pytest.param(lambda x: np.add.reduceat(x, [0]), id="reduceat"),
         pytest.param(lambda x: np.log.at(x, [2]), id="at")])
def test_ufunc_methods_floaterrors(method):
    # adding inf and -inf (or log(-inf) creates an invalid float and warns
    arr = np.array([np.inf, 0, -np.inf])
    with np.errstate(all="warn"):
        with pytest.warns(RuntimeWarning, match="invalid value"):
            method(arr)

    arr = np.array([np.inf, 0, -np.inf])
    with np.errstate(all="raise"):
        with pytest.raises(FloatingPointError):
            method(arr)


def _check_neg_zero(value):
    if value != 0.0:
        return False
    if not np.signbit(value.real):
        return False
    if value.dtype.kind == "c":
        return np.signbit(value.imag)
    return True

@pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
def test_addition_negative_zero(dtype):
    dtype = np.dtype(dtype)
    if dtype.kind == "c":
        neg_zero = dtype.type(complex(-0.0, -0.0))
    else:
        neg_zero = dtype.type(-0.0)

    arr = np.array(neg_zero)
    arr2 = np.array(neg_zero)

    assert _check_neg_zero(arr + arr2)
    # In-place ops may end up on a different path (reduce path) see gh-21211
    arr += arr2
    assert _check_neg_zero(arr)


@pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
@pytest.mark.parametrize("use_initial", [True, False])
def test_addition_reduce_negative_zero(dtype, use_initial):
    dtype = np.dtype(dtype)
    if dtype.kind == "c":
        neg_zero = dtype.type(complex(-0.0, -0.0))
    else:
        neg_zero = dtype.type(-0.0)

    kwargs = {}
    if use_initial:
        kwargs["initial"] = neg_zero
    else:
        pytest.xfail("-0. propagation in sum currently requires initial")

    # Test various length, in case SIMD paths or chunking play a role.
    # 150 extends beyond the pairwise blocksize; probably not important.
    for i in range(150):
        arr = np.array([neg_zero] * i, dtype=dtype)
        res = np.sum(arr, **kwargs)
        if i > 0 or use_initial:
            assert _check_neg_zero(res)
        else:
            # `sum([])` should probably be 0.0 and not -0.0 like `sum([-0.0])`
            assert not np.signbit(res.real)
            assert not np.signbit(res.imag)


@pytest.mark.parametrize(["dt1", "dt2"],
        [("S", "U"), ("U", "S"), ("S", "d"), ("S", "V"), ("U", "l")])
def test_addition_string_types(dt1, dt2):
    arr1 = np.array([1234234], dtype=dt1)
    arr2 = np.array([b"423"], dtype=dt2)
    with pytest.raises(np._core._exceptions.UFuncTypeError) as exc:
        np.add(arr1, arr2)


@pytest.mark.parametrize("order1,order2",
                         [(">", ">"), ("<", "<"), (">", "<"), ("<", ">")])
def test_addition_unicode_inverse_byte_order(order1, order2):
    element = 'abcd'
    arr1 = np.array([element], dtype=f"{order1}U4")
    arr2 = np.array([element], dtype=f"{order2}U4")
    result = arr1 + arr2
    assert result == 2 * element


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_find_non_long_args(dtype):
    element = 'abcd'
    start = dtype(0)
    end = dtype(len(element))
    arr = np.array([element])
    result = np._core.umath.find(arr, "a", start, end)
    assert result.dtype == np.dtype("intp")
    assert result == 0


def test_find_access_past_buffer():
    # This checks that no read past the string buffer occurs in
    # string_fastsearch.h. The buffer class makes sure this is checked.
    # To see it in action, you can remove the checks in the buffer and
    # this test will produce an 'Invalid read' if run under valgrind.
    arr = np.array([b'abcd', b'ebcd'])
    result = np._core.umath.find(arr, b'cde', 0, np.iinfo(np.int64).max)
    assert np.all(result == -1)


class TestLowlevelAPIAccess:
    def test_resolve_dtypes_basic(self):
        # Basic test for dtype resolution:
        i4 = np.dtype("i4")
        f4 = np.dtype("f4")
        f8 = np.dtype("f8")

        r = np.add.resolve_dtypes((i4, f4, None))
        assert r == (f8, f8, f8)

        # Signature uses the same logic to parse as ufunc (less strict)
        # the following is "same-kind" casting so works:
        r = np.add.resolve_dtypes((
                i4, i4, None), signature=(None, None, "f4"))
        assert r == (f4, f4, f4)

        # Check NEP 50 "weak" promotion also:
        r = np.add.resolve_dtypes((f4, int, None))
        assert r == (f4, f4, f4)

        with pytest.raises(TypeError):
            np.add.resolve_dtypes((i4, f4, None), casting="no")

    def test_resolve_dtypes_comparison(self):
        i4 = np.dtype("i4")
        i8 = np.dtype("i8")
        b = np.dtype("?")
        r = np.equal.resolve_dtypes((i4, i8, None))
        assert r == (i8, i8, b)

    def test_weird_dtypes(self):
        S0 = np.dtype("S0")
        # S0 is often converted by NumPy to S1, but not here:
        r = np.equal.resolve_dtypes((S0, S0, None))
        assert r == (S0, S0, np.dtype(bool))

        # Subarray dtypes are weird and may not work fully, we preserve them
        # leading to a TypeError (currently no equal loop for void/structured)
        dts = np.dtype("10i")
        with pytest.raises(TypeError):
            np.equal.resolve_dtypes((dts, dts, None))

    def test_resolve_dtypes_reduction(self):
        i2 = np.dtype("i2")
        default_int_ = np.dtype(np.int_)
        # Check special addition resolution:
        res = np.add.resolve_dtypes((None, i2, None), reduction=True)
        assert res == (default_int_, default_int_, default_int_)

    def test_resolve_dtypes_reduction_no_output(self):
        i4 = np.dtype("i4")
        with pytest.raises(TypeError):
            # May be allowable at some point?
            np.add.resolve_dtypes((i4, i4, i4), reduction=True)

    @pytest.mark.parametrize("dtypes", [
            (np.dtype("i"), np.dtype("i")),
            (None, np.dtype("i"), np.dtype("f")),
            (np.dtype("i"), None, np.dtype("f")),
            ("i4", "i4", None)])
    def test_resolve_dtypes_errors(self, dtypes):
        with pytest.raises(TypeError):
            np.add.resolve_dtypes(dtypes)

    def test_resolve_dtypes_reduction_errors(self):
        i2 = np.dtype("i2")

        with pytest.raises(TypeError):
            np.add.resolve_dtypes((None, i2, i2))

        with pytest.raises(TypeError):
            np.add.signature((None, None, "i4"))

    @pytest.mark.skipif(not hasattr(ct, "pythonapi"),
            reason="`ctypes.pythonapi` required for capsule unpacking.")
    def test_loop_access(self):
        # This is a basic test for the full strided loop access
        data_t = ct.c_char_p * 2
        dim_t = ct.c_ssize_t * 1
        strides_t = ct.c_ssize_t * 2
        strided_loop_t = ct.CFUNCTYPE(
                ct.c_int, ct.c_void_p, data_t, dim_t, strides_t, ct.c_void_p)

        class call_info_t(ct.Structure):
            _fields_ = [
                ("strided_loop", strided_loop_t),
                ("context", ct.c_void_p),
                ("auxdata", ct.c_void_p),
                ("requires_pyapi", ct.c_byte),
                ("no_floatingpoint_errors", ct.c_byte),
            ]

        i4 = np.dtype("i4")
        dt, call_info_obj = np.negative._resolve_dtypes_and_context((i4, i4))
        assert dt == (i4, i4)  # can be used without casting

        # Fill in the rest of the information:
        np.negative._get_strided_loop(call_info_obj)

        ct.pythonapi.PyCapsule_GetPointer.restype = ct.c_void_p
        call_info = ct.pythonapi.PyCapsule_GetPointer(
                ct.py_object(call_info_obj),
                ct.c_char_p(b"numpy_1.24_ufunc_call_info"))

        call_info = ct.cast(call_info, ct.POINTER(call_info_t)).contents

        arr = np.arange(10, dtype=i4)
        call_info.strided_loop(
                call_info.context,
                data_t(arr.ctypes.data, arr.ctypes.data),
                arr.ctypes.shape,  # is a C-array with 10 here
                strides_t(arr.ctypes.strides[0], arr.ctypes.strides[0]),
                call_info.auxdata)

        # We just directly called the negative inner-loop in-place:
        assert_array_equal(arr, -np.arange(10, dtype=i4))

    @pytest.mark.parametrize("strides", [1, (1, 2, 3), (1, "2")])
    def test__get_strided_loop_errors_bad_strides(self, strides):
        i4 = np.dtype("i4")
        dt, call_info = np.negative._resolve_dtypes_and_context((i4, i4))

        with pytest.raises(TypeError, match="fixed_strides.*tuple.*or None"):
            np.negative._get_strided_loop(call_info, fixed_strides=strides)

    def test__get_strided_loop_errors_bad_call_info(self):
        i4 = np.dtype("i4")
        dt, call_info = np.negative._resolve_dtypes_and_context((i4, i4))

        with pytest.raises(ValueError, match="PyCapsule"):
            np.negative._get_strided_loop("not the capsule!")

        with pytest.raises(TypeError, match=".*incompatible context"):
            np.add._get_strided_loop(call_info)

        np.negative._get_strided_loop(call_info)
        with pytest.raises(TypeError):
            # cannot call it a second time:
            np.negative._get_strided_loop(call_info)

    def test_long_arrays(self):
        t = np.zeros((1029, 917), dtype=np.single)
        t[0][0] = 1
        t[28][414] = 1
        tc = np.cos(t)
        assert_equal(tc[0][0], tc[28][414])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\onnx_model_gpt2.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

import onnx
from fusion_gpt_attention import FusionGptAttention
from fusion_gpt_attention_megatron import FusionGptAttentionMegatron
from fusion_gpt_attention_no_past import FusionGptAttentionNoPast
from fusion_rotary_attention import FusionRotaryAttention
from onnx_model_bert import BertOnnxModel

logger = logging.getLogger(__name__)


class Gpt2OnnxModel(BertOnnxModel):
    def __init__(self, model, num_heads, hidden_size):
        super().__init__(model, num_heads, hidden_size)

    def fuse_attention(self):
        if len(self.model.graph.input) == 1 or len(self.model.graph.output) == 1:
            fusion = FusionGptAttentionNoPast(self, self.num_heads)
            fusion.apply()
        else:
            fusion = FusionGptAttention(self, self.num_heads)
            fusion.apply()
            fusion = FusionGptAttentionMegatron(self, self.num_heads)
            fusion.apply()

        fusion = FusionRotaryAttention(self, self.hidden_size, self.num_heads)
        fusion.apply()

    def postprocess(self):
        """
        Remove extra reshape nodes.
        """
        logger.debug("start postprocessing...")

        input_name_to_nodes = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_node()

        reshape_count = 0
        for gemm_node in self.get_nodes_by_op_type("Gemm"):
            reshape_after_gemm = self.find_first_child_by_type(
                gemm_node, "Reshape", input_name_to_nodes, recursive=False
            )

            nodes = self.match_parent_path(gemm_node, ["Reshape", "FastGelu"], [0, 0], output_name_to_node)
            if nodes is None:
                nodes = self.match_parent_path(
                    gemm_node,
                    ["Reshape", "LayerNormalization"],
                    [0, 0],
                    output_name_to_node,
                )

                if nodes is None:
                    nodes = self.match_parent_path(
                        gemm_node,
                        ["Reshape", "SkipLayerNormalization"],
                        [0, 0],
                        output_name_to_node,
                    )

                    if nodes is None:
                        continue

            (reshape_before_gemm, root_node) = nodes

            matmul_node_name = self.create_node_name("MatMul", "FullyConnect_MatMul")
            matmul_node = onnx.helper.make_node(
                "MatMul",
                inputs=[matmul_node_name + "_input", gemm_node.input[1]],
                outputs=[matmul_node_name + "_output"],
                name=matmul_node_name,
            )

            add_node_name = self.create_node_name("Add", "FullyConnect_Add")
            add_node = onnx.helper.make_node(
                "Add",
                inputs=[matmul_node_name + "_output", gemm_node.input[2]],
                outputs=[add_node_name + "_output"],
                name=add_node_name,
            )

            self.replace_input_of_all_nodes(reshape_after_gemm.output[0], add_node_name + "_output")

            # Link root node output with MatMul
            self.replace_input_of_all_nodes(root_node.output[0], matmul_node_name + "_input")
            root_node.output[0] = matmul_node_name + "_input"

            self.replace_input_of_all_nodes(reshape_after_gemm.output[0], add_node_name + "_output")

            self.add_node(matmul_node)
            self.add_node(add_node)

            reshape_count += 2

        self.prune_graph()
        logger.info(f"postprocess: remove Reshape count: {reshape_count}")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\utils\glibc.py ===
import os
import sys
from typing import Optional, Tuple


def glibc_version_string() -> Optional[str]:
    "Returns glibc version string, or None if not using glibc."
    return glibc_version_string_confstr() or glibc_version_string_ctypes()


def glibc_version_string_confstr() -> Optional[str]:
    "Primary implementation of glibc_version_string using os.confstr."
    # os.confstr is quite a bit faster than ctypes.DLL. It's also less likely
    # to be broken or missing. This strategy is used in the standard library
    # platform module:
    # https://github.com/python/cpython/blob/fcf1d003bf4f0100c9d0921ff3d70e1127ca1b71/Lib/platform.py#L175-L183
    if sys.platform == "win32":
        return None
    try:
        gnu_libc_version = os.confstr("CS_GNU_LIBC_VERSION")
        if gnu_libc_version is None:
            return None
        # os.confstr("CS_GNU_LIBC_VERSION") returns a string like "glibc 2.17":
        _, version = gnu_libc_version.split()
    except (AttributeError, OSError, ValueError):
        # os.confstr() or CS_GNU_LIBC_VERSION not available (or a bad value)...
        return None
    return version


def glibc_version_string_ctypes() -> Optional[str]:
    "Fallback implementation of glibc_version_string using ctypes."

    try:
        import ctypes
    except ImportError:
        return None

    # ctypes.CDLL(None) internally calls dlopen(NULL), and as the dlopen
    # manpage says, "If filename is NULL, then the returned handle is for the
    # main program". This way we can let the linker do the work to figure out
    # which libc our process is actually using.
    #
    # We must also handle the special case where the executable is not a
    # dynamically linked executable. This can occur when using musl libc,
    # for example. In this situation, dlopen() will error, leading to an
    # OSError. Interestingly, at least in the case of musl, there is no
    # errno set on the OSError. The single string argument used to construct
    # OSError comes from libc itself and is therefore not portable to
    # hard code here. In any case, failure to call dlopen() means we
    # can't proceed, so we bail on our attempt.
    try:
        process_namespace = ctypes.CDLL(None)
    except OSError:
        return None

    try:
        gnu_get_libc_version = process_namespace.gnu_get_libc_version
    except AttributeError:
        # Symbol doesn't exist -> therefore, we are not linked to
        # glibc.
        return None

    # Call gnu_get_libc_version, which returns a string like "2.5"
    gnu_get_libc_version.restype = ctypes.c_char_p
    version_str: str = gnu_get_libc_version()
    # py2 / py3 compatibility:
    if not isinstance(version_str, str):
        version_str = version_str.decode("ascii")

    return version_str


# platform.libc_ver regularly returns completely nonsensical glibc
# versions. E.g. on my computer, platform says:
#
#   ~$ python2.7 -c 'import platform; print(platform.libc_ver())'
#   ('glibc', '2.7')
#   ~$ python3.5 -c 'import platform; print(platform.libc_ver())'
#   ('glibc', '2.9')
#
# But the truth is:
#
#   ~$ ldd --version
#   ldd (Debian GLIBC 2.22-11) 2.22
#
# This is unfortunate, because it means that the linehaul data on libc
# versions that was generated by pip 8.1.2 and earlier is useless and
# misleading. Solution: instead of using platform, use our code that actually
# works.
def libc_ver() -> Tuple[str, str]:
    """Try to determine the glibc version

    Returns a tuple of strings (lib, version) which default to empty strings
    in case the lookup fails.
    """
    glibc_version = glibc_version_string()
    if glibc_version is None:
        return ("", "")
    else:
        return ("glibc", glibc_version)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\jupyter.py ===
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Sequence

if TYPE_CHECKING:
    from pip._vendor.rich.console import ConsoleRenderable

from . import get_console
from .segment import Segment
from .terminal_theme import DEFAULT_TERMINAL_THEME

if TYPE_CHECKING:
    from pip._vendor.rich.console import ConsoleRenderable

JUPYTER_HTML_FORMAT = """\
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">{code}</pre>
"""


class JupyterRenderable:
    """A shim to write html to Jupyter notebook."""

    def __init__(self, html: str, text: str) -> None:
        self.html = html
        self.text = text

    def _repr_mimebundle_(
        self, include: Sequence[str], exclude: Sequence[str], **kwargs: Any
    ) -> Dict[str, str]:
        data = {"text/plain": self.text, "text/html": self.html}
        if include:
            data = {k: v for (k, v) in data.items() if k in include}
        if exclude:
            data = {k: v for (k, v) in data.items() if k not in exclude}
        return data


class JupyterMixin:
    """Add to an Rich renderable to make it render in Jupyter notebook."""

    __slots__ = ()

    def _repr_mimebundle_(
        self: "ConsoleRenderable",
        include: Sequence[str],
        exclude: Sequence[str],
        **kwargs: Any,
    ) -> Dict[str, str]:
        console = get_console()
        segments = list(console.render(self, console.options))
        html = _render_segments(segments)
        text = console._render_buffer(segments)
        data = {"text/plain": text, "text/html": html}
        if include:
            data = {k: v for (k, v) in data.items() if k in include}
        if exclude:
            data = {k: v for (k, v) in data.items() if k not in exclude}
        return data


def _render_segments(segments: Iterable[Segment]) -> str:
    def escape(text: str) -> str:
        """Escape html."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    fragments: List[str] = []
    append_fragment = fragments.append
    theme = DEFAULT_TERMINAL_THEME
    for text, style, control in Segment.simplify(segments):
        if control:
            continue
        text = escape(text)
        if style:
            rule = style.get_html_style(theme)
            text = f'<span style="{rule}">{text}</span>' if rule else text
            if style.link:
                text = f'<a href="{style.link}" target="_blank">{text}</a>'
        append_fragment(text)

    code = "".join(fragments)
    html = JUPYTER_HTML_FORMAT.format(code=code)

    return html


def display(segments: Iterable[Segment], text: str) -> None:
    """Render segments to Jupyter."""
    html = _render_segments(segments)
    jupyter_renderable = JupyterRenderable(html, text)
    try:
        from IPython.display import display as ipython_display

        ipython_display(jupyter_renderable)
    except ModuleNotFoundError:
        # Handle the case where the Console has force_jupyter=True,
        # but IPython is not installed.
        pass


def print(*args: Any, **kwargs: Any) -> None:
    """Proxy for Console print."""
    console = get_console()
    return console.print(*args, **kwargs)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\acceleration.py ===
"""
Convergence acceleration / extrapolation methods for series and
sequences.

References:
Carl M. Bender & Steven A. Orszag, "Advanced Mathematical Methods for
Scientists and Engineers: Asymptotic Methods and Perturbation Theory",
Springer 1999. (Shanks transformation: pp. 368-375, Richardson
extrapolation: pp. 375-377.)
"""

from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.functions.combinatorial.factorials import factorial


def richardson(A, k, n, N):
    """
    Calculate an approximation for lim k->oo A(k) using Richardson
    extrapolation with the terms A(n), A(n+1), ..., A(n+N+1).
    Choosing N ~= 2*n often gives good results.

    Examples
    ========

    A simple example is to calculate exp(1) using the limit definition.
    This limit converges slowly; n = 100 only produces two accurate
    digits:

        >>> from sympy.abc import n
        >>> e = (1 + 1/n)**n
        >>> print(round(e.subs(n, 100).evalf(), 10))
        2.7048138294

    Richardson extrapolation with 11 appropriately chosen terms gives
    a value that is accurate to the indicated precision:

        >>> from sympy import E
        >>> from sympy.series.acceleration import richardson
        >>> print(round(richardson(e, n, 10, 20).evalf(), 10))
        2.7182818285
        >>> print(round(E.evalf(), 10))
        2.7182818285

    Another useful application is to speed up convergence of series.
    Computing 100 terms of the zeta(2) series 1/k**2 yields only
    two accurate digits:

        >>> from sympy.abc import k, n
        >>> from sympy import Sum
        >>> A = Sum(k**-2, (k, 1, n))
        >>> print(round(A.subs(n, 100).evalf(), 10))
        1.6349839002

    Richardson extrapolation performs much better:

        >>> from sympy import pi
        >>> print(round(richardson(A, n, 10, 20).evalf(), 10))
        1.6449340668
        >>> print(round(((pi**2)/6).evalf(), 10))     # Exact value
        1.6449340668

    """
    s = S.Zero
    for j in range(0, N + 1):
        s += (A.subs(k, Integer(n + j)).doit() * (n + j)**N *
              S.NegativeOne**(j + N) / (factorial(j) * factorial(N - j)))
    return s


def shanks(A, k, n, m=1):
    """
    Calculate an approximation for lim k->oo A(k) using the n-term Shanks
    transformation S(A)(n). With m > 1, calculate the m-fold recursive
    Shanks transformation S(S(...S(A)...))(n).

    The Shanks transformation is useful for summing Taylor series that
    converge slowly near a pole or singularity, e.g. for log(2):

        >>> from sympy.abc import k, n
        >>> from sympy import Sum, Integer
        >>> from sympy.series.acceleration import shanks
        >>> A = Sum(Integer(-1)**(k+1) / k, (k, 1, n))
        >>> print(round(A.subs(n, 100).doit().evalf(), 10))
        0.6881721793
        >>> print(round(shanks(A, n, 25).evalf(), 10))
        0.6931396564
        >>> print(round(shanks(A, n, 25, 5).evalf(), 10))
        0.6931471806

    The correct value is 0.6931471805599453094172321215.
    """
    table = [A.subs(k, Integer(j)).doit() for j in range(n + m + 2)]
    table2 = table.copy()

    for i in range(1, m + 1):
        for j in range(i, n + m + 1):
            x, y, z = table[j - 1], table[j], table[j + 1]
            table2[j] = (z*x - y**2) / (z + x - 2*y)
        table = table2.copy()
    return table[n]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\util\response.py ===
from __future__ import annotations

import http.client as httplib
from email.errors import MultipartInvariantViolationDefect, StartBoundaryNotFoundDefect

from ..exceptions import HeaderParsingError


def is_fp_closed(obj: object) -> bool:
    """
    Checks whether a given file-like object is closed.

    :param obj:
        The file-like object to check.
    """

    try:
        # Check `isclosed()` first, in case Python3 doesn't set `closed`.
        # GH Issue #928
        return obj.isclosed()  # type: ignore[no-any-return, attr-defined]
    except AttributeError:
        pass

    try:
        # Check via the official file-like-object way.
        return obj.closed  # type: ignore[no-any-return, attr-defined]
    except AttributeError:
        pass

    try:
        # Check if the object is a container for another file-like object that
        # gets released on exhaustion (e.g. HTTPResponse).
        return obj.fp is None  # type: ignore[attr-defined]
    except AttributeError:
        pass

    raise ValueError("Unable to determine whether fp is closed.")


def assert_header_parsing(headers: httplib.HTTPMessage) -> None:
    """
    Asserts whether all headers have been successfully parsed.
    Extracts encountered errors from the result of parsing headers.

    Only works on Python 3.

    :param http.client.HTTPMessage headers: Headers to verify.

    :raises urllib3.exceptions.HeaderParsingError:
        If parsing errors are found.
    """

    # This will fail silently if we pass in the wrong kind of parameter.
    # To make debugging easier add an explicit check.
    if not isinstance(headers, httplib.HTTPMessage):
        raise TypeError(f"expected httplib.Message, got {type(headers)}.")

    unparsed_data = None

    # get_payload is actually email.message.Message.get_payload;
    # we're only interested in the result if it's not a multipart message
    if not headers.is_multipart():
        payload = headers.get_payload()

        if isinstance(payload, (bytes, str)):
            unparsed_data = payload

    # httplib is assuming a response body is available
    # when parsing headers even when httplib only sends
    # header data to parse_headers() This results in
    # defects on multipart responses in particular.
    # See: https://github.com/urllib3/urllib3/issues/800

    # So we ignore the following defects:
    # - StartBoundaryNotFoundDefect:
    #     The claimed start boundary was never found.
    # - MultipartInvariantViolationDefect:
    #     A message claimed to be a multipart but no subparts were found.
    defects = [
        defect
        for defect in headers.defects
        if not isinstance(
            defect, (StartBoundaryNotFoundDefect, MultipartInvariantViolationDefect)
        )
    ]

    if defects or unparsed_data:
        raise HeaderParsingError(defects=defects, unparsed_data=unparsed_data)


def is_response_to_head(response: httplib.HTTPResponse) -> bool:
    """
    Checks whether the request of a response has been a HEAD-request.

    :param http.client.HTTPResponse response:
        Response to check if the originating request
        used 'HEAD' as a method.
    """
    # FIXME: Can we do this somehow without accessing private httplib _method?
    method_str = response._method  # type: str  # type: ignore[attr-defined]
    return method_str.upper() == "HEAD"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\duration.py ===
# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Contains the Duration helper APIs."""

import datetime

from google.protobuf.duration_pb2 import Duration


def from_json_string(value: str) -> Duration:
  """Converts a string to Duration.

  Args:
    value: A string to be converted. The string must end with 's'. Any
      fractional digits (or none) are accepted as long as they fit into
      precision. For example: "1s", "1.01s", "1.0000001s", "-3.100s"

  Raises:
    ValueError: On parsing problems.
  """
  duration = Duration()
  duration.FromJsonString(value)
  return duration


def from_microseconds(micros: float) -> Duration:
  """Converts microseconds to Duration."""
  duration = Duration()
  duration.FromMicroseconds(micros)
  return duration


def from_milliseconds(millis: float) -> Duration:
  """Converts milliseconds to Duration."""
  duration = Duration()
  duration.FromMilliseconds(millis)
  return duration


def from_nanoseconds(nanos: float) -> Duration:
  """Converts nanoseconds to Duration."""
  duration = Duration()
  duration.FromNanoseconds(nanos)
  return duration


def from_seconds(seconds: float) -> Duration:
  """Converts seconds to Duration."""
  duration = Duration()
  duration.FromSeconds(seconds)
  return duration


def from_timedelta(td: datetime.timedelta) -> Duration:
  """Converts timedelta to Duration."""
  duration = Duration()
  duration.FromTimedelta(td)
  return duration


def to_json_string(duration: Duration) -> str:
  """Converts Duration to string format.

  Returns:
    A string converted from self. The string format will contains
    3, 6, or 9 fractional digits depending on the precision required to
    represent the exact Duration value. For example: "1s", "1.010s",
    "1.000000100s", "-3.100s"
  """
  return duration.ToJsonString()


def to_microseconds(duration: Duration) -> int:
  """Converts a Duration to microseconds."""
  return duration.ToMicroseconds()


def to_milliseconds(duration: Duration) -> int:
  """Converts a Duration to milliseconds."""
  return duration.ToMilliseconds()


def to_nanoseconds(duration: Duration) -> int:
  """Converts a Duration to nanoseconds."""
  return duration.ToNanoseconds()


def to_seconds(duration: Duration) -> int:
  """Converts a Duration to seconds."""
  return duration.ToSeconds()


def to_timedelta(duration: Duration) -> datetime.timedelta:
  """Converts Duration to timedelta."""
  return duration.ToTimedelta()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\_string_helpers.py ===
"""
String-handling utilities to avoid locale-dependence.

Used primarily to generate type name aliases.
"""
# "import string" is costly to import!
# Construct the translation tables directly
#   "A" = chr(65), "a" = chr(97)
_all_chars = tuple(map(chr, range(256)))
_ascii_upper = _all_chars[65:65 + 26]
_ascii_lower = _all_chars[97:97 + 26]
LOWER_TABLE = _all_chars[:65] + _ascii_lower + _all_chars[65 + 26:]
UPPER_TABLE = _all_chars[:97] + _ascii_upper + _all_chars[97 + 26:]


def english_lower(s):
    """ Apply English case rules to convert ASCII strings to all lower case.

    This is an internal utility function to replace calls to str.lower() such
    that we can avoid changing behavior with changing locales. In particular,
    Turkish has distinct dotted and dotless variants of the Latin letter "I" in
    both lowercase and uppercase. Thus, "I".lower() != "i" in a "tr" locale.

    Parameters
    ----------
    s : str

    Returns
    -------
    lowered : str

    Examples
    --------
    >>> from numpy._core.numerictypes import english_lower
    >>> english_lower('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_')
    'abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz0123456789_'
    >>> english_lower('')
    ''
    """
    lowered = s.translate(LOWER_TABLE)
    return lowered


def english_upper(s):
    """ Apply English case rules to convert ASCII strings to all upper case.

    This is an internal utility function to replace calls to str.upper() such
    that we can avoid changing behavior with changing locales. In particular,
    Turkish has distinct dotted and dotless variants of the Latin letter "I" in
    both lowercase and uppercase. Thus, "i".upper() != "I" in a "tr" locale.

    Parameters
    ----------
    s : str

    Returns
    -------
    uppered : str

    Examples
    --------
    >>> from numpy._core.numerictypes import english_upper
    >>> english_upper('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_')
    'ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'
    >>> english_upper('')
    ''
    """
    uppered = s.translate(UPPER_TABLE)
    return uppered


def english_capitalize(s):
    """ Apply English case rules to convert the first character of an ASCII
    string to upper case.

    This is an internal utility function to replace calls to str.capitalize()
    such that we can avoid changing behavior with changing locales.

    Parameters
    ----------
    s : str

    Returns
    -------
    capitalized : str

    Examples
    --------
    >>> from numpy._core.numerictypes import english_capitalize
    >>> english_capitalize('int8')
    'Int8'
    >>> english_capitalize('Int8')
    'Int8'
    >>> english_capitalize('')
    ''
    """
    if s:
        return english_upper(s[0]) + s[1:]
    else:
        return s

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\palette.py ===
from math import sqrt
from functools import lru_cache
from typing import Sequence, Tuple, TYPE_CHECKING

from .color_triplet import ColorTriplet

if TYPE_CHECKING:
    from pip._vendor.rich.table import Table


class Palette:
    """A palette of available colors."""

    def __init__(self, colors: Sequence[Tuple[int, int, int]]):
        self._colors = colors

    def __getitem__(self, number: int) -> ColorTriplet:
        return ColorTriplet(*self._colors[number])

    def __rich__(self) -> "Table":
        from pip._vendor.rich.color import Color
        from pip._vendor.rich.style import Style
        from pip._vendor.rich.text import Text
        from pip._vendor.rich.table import Table

        table = Table(
            "index",
            "RGB",
            "Color",
            title="Palette",
            caption=f"{len(self._colors)} colors",
            highlight=True,
            caption_justify="right",
        )
        for index, color in enumerate(self._colors):
            table.add_row(
                str(index),
                repr(color),
                Text(" " * 16, style=Style(bgcolor=Color.from_rgb(*color))),
            )
        return table

    # This is somewhat inefficient and needs caching
    @lru_cache(maxsize=1024)
    def match(self, color: Tuple[int, int, int]) -> int:
        """Find a color from a palette that most closely matches a given color.

        Args:
            color (Tuple[int, int, int]): RGB components in range 0 > 255.

        Returns:
            int: Index of closes matching color.
        """
        red1, green1, blue1 = color
        _sqrt = sqrt
        get_color = self._colors.__getitem__

        def get_color_distance(index: int) -> float:
            """Get the distance to a color."""
            red2, green2, blue2 = get_color(index)
            red_mean = (red1 + red2) // 2
            red = red1 - red2
            green = green1 - green2
            blue = blue1 - blue2
            return _sqrt(
                (((512 + red_mean) * red * red) >> 8)
                + 4 * green * green
                + (((767 - red_mean) * blue * blue) >> 8)
            )

        min_index = min(range(len(self._colors)), key=get_color_distance)
        return min_index


if __name__ == "__main__":  # pragma: no cover
    import colorsys
    from typing import Iterable
    from pip._vendor.rich.color import Color
    from pip._vendor.rich.console import Console, ConsoleOptions
    from pip._vendor.rich.segment import Segment
    from pip._vendor.rich.style import Style

    class ColorBox:
        def __rich_console__(
            self, console: Console, options: ConsoleOptions
        ) -> Iterable[Segment]:
            height = console.size.height - 3
            for y in range(0, height):
                for x in range(options.max_width):
                    h = x / options.max_width
                    l = y / (height + 1)
                    r1, g1, b1 = colorsys.hls_to_rgb(h, l, 1.0)
                    r2, g2, b2 = colorsys.hls_to_rgb(h, l + (1 / height / 2), 1.0)
                    bgcolor = Color.from_rgb(r1 * 255, g1 * 255, b1 * 255)
                    color = Color.from_rgb(r2 * 255, g2 * 255, b2 * 255)
                    yield Segment("▄", Style(color=color, bgcolor=bgcolor))
                yield Segment.line()

    console = Console()
    console.print(ColorBox())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\by.py ===
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
"""The By implementation."""

from typing import Dict, Literal, Optional


class By:
    """Set of supported locator strategies.

    ID:
    --
    Select the element by its ID.

    >>> element = driver.find_element(By.ID, "myElement")

    XPATH:
    ------
    Select the element via XPATH.
        - absolute path
        - relative path

    >>> element = driver.find_element(By.XPATH, "//html/body/div")

    LINK_TEXT:
    ----------
    Select the link element having the exact text.

    >>> element = driver.find_element(By.LINK_TEXT, "myLink")

    PARTIAL_LINK_TEXT:
    ------------------
    Select the link element having the partial text.

    >>> element = driver.find_element(By.PARTIAL_LINK_TEXT, "my")

    NAME:
    ----
    Select the element by its name attribute.

    >>> element = driver.find_element(By.NAME, "myElement")

    TAG_NAME:
    --------
    Select the element by its tag name.

    >>> element = driver.find_element(By.TAG_NAME, "div")

    CLASS_NAME:
    -----------
    Select the element by its class name.

    >>> element = driver.find_element(By.CLASS_NAME, "myElement")

    CSS_SELECTOR:
    -------------
    Select the element by its CSS selector.

    >>> element = driver.find_element(By.CSS_SELECTOR, "div.myElement")
    """

    ID = "id"
    XPATH = "xpath"
    LINK_TEXT = "link text"
    PARTIAL_LINK_TEXT = "partial link text"
    NAME = "name"
    TAG_NAME = "tag name"
    CLASS_NAME = "class name"
    CSS_SELECTOR = "css selector"

    _custom_finders: Dict[str, str] = {}

    @classmethod
    def register_custom_finder(cls, name: str, strategy: str) -> None:
        cls._custom_finders[name] = strategy

    @classmethod
    def get_finder(cls, name: str) -> Optional[str]:
        return cls._custom_finders.get(name) or getattr(cls, name.upper(), None)

    @classmethod
    def clear_custom_finders(cls) -> None:
        cls._custom_finders.clear()


ByType = Literal["id", "xpath", "link text", "partial link text", "name", "tag name", "class name", "css selector"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\gmpyrationalfield.py ===
"""Implementation of :class:`GMPYRationalField` class. """


from sympy.polys.domains.groundtypes import (
    GMPYRational, SymPyRational,
    gmpy_numer, gmpy_denom, factorial as gmpy_factorial,
)
from sympy.polys.domains.rationalfield import RationalField
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public

@public
class GMPYRationalField(RationalField):
    """Rational field based on GMPY's ``mpq`` type.

    This will be the implementation of :ref:`QQ` if ``gmpy`` or ``gmpy2`` is
    installed. Elements will be of type ``gmpy.mpq``.
    """

    dtype = GMPYRational
    zero = dtype(0)
    one = dtype(1)
    tp = type(one)
    alias = 'QQ_gmpy'

    def __init__(self):
        pass

    def get_ring(self):
        """Returns ring associated with ``self``. """
        from sympy.polys.domains import GMPYIntegerRing
        return GMPYIntegerRing()

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return SymPyRational(int(gmpy_numer(a)),
                             int(gmpy_denom(a)))

    def from_sympy(self, a):
        """Convert SymPy's Integer to ``dtype``. """
        if a.is_Rational:
            return GMPYRational(a.p, a.q)
        elif a.is_Float:
            from sympy.polys.domains import RR
            return GMPYRational(*map(int, RR.to_rational(a)))
        else:
            raise CoercionFailed("expected ``Rational`` object, got %s" % a)

    def from_ZZ_python(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return GMPYRational(a)

    def from_QQ_python(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return GMPYRational(a.numerator, a.denominator)

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpz`` object to ``dtype``. """
        return GMPYRational(a)

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpq`` object to ``dtype``. """
        return a

    def from_GaussianRationalField(K1, a, K0):
        """Convert a ``GaussianElement`` object to ``dtype``. """
        if a.y == 0:
            return GMPYRational(a.x)

    def from_RealField(K1, a, K0):
        """Convert a mpmath ``mpf`` object to ``dtype``. """
        return GMPYRational(*map(int, K0.to_rational(a)))

    def exquo(self, a, b):
        """Exact quotient of ``a`` and ``b``, implies ``__truediv__``.  """
        return GMPYRational(a) / GMPYRational(b)

    def quo(self, a, b):
        """Quotient of ``a`` and ``b``, implies ``__truediv__``. """
        return GMPYRational(a) / GMPYRational(b)

    def rem(self, a, b):
        """Remainder of ``a`` and ``b``, implies nothing.  """
        return self.zero

    def div(self, a, b):
        """Division of ``a`` and ``b``, implies ``__truediv__``. """
        return GMPYRational(a) / GMPYRational(b), self.zero

    def numer(self, a):
        """Returns numerator of ``a``. """
        return a.numerator

    def denom(self, a):
        """Returns denominator of ``a``. """
        return a.denominator

    def factorial(self, a):
        """Returns factorial of ``a``. """
        return GMPYRational(gmpy_factorial(int(a)))