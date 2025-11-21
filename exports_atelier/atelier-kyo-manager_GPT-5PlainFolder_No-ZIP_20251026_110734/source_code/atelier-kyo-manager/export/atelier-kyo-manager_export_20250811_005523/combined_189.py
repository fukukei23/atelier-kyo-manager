
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\methods\test_value_counts.py ===
"""
these are systematically testing all of the args to value_counts
with different size combinations. This is to ensure stability of the sorting
and proper parameter handling
"""


import numpy as np
import pytest

from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    Grouper,
    Index,
    MultiIndex,
    Series,
    date_range,
    to_datetime,
)
import pandas._testing as tm
from pandas.util.version import Version


def tests_value_counts_index_names_category_column():
    # GH44324 Missing name of index category column
    df = DataFrame(
        {
            "gender": ["female"],
            "country": ["US"],
        }
    )
    df["gender"] = df["gender"].astype("category")
    result = df.groupby("country")["gender"].value_counts()

    # Construct expected, very specific multiindex
    df_mi_expected = DataFrame([["US", "female"]], columns=["country", "gender"])
    df_mi_expected["gender"] = df_mi_expected["gender"].astype("category")
    mi_expected = MultiIndex.from_frame(df_mi_expected)
    expected = Series([1], index=mi_expected, name="count")

    tm.assert_series_equal(result, expected)


def seed_df(seed_nans, n, m):
    days = date_range("2015-08-24", periods=10)

    frame = DataFrame(
        {
            "1st": np.random.default_rng(2).choice(list("abcd"), n),
            "2nd": np.random.default_rng(2).choice(days, n),
            "3rd": np.random.default_rng(2).integers(1, m + 1, n),
        }
    )

    if seed_nans:
        # Explicitly cast to float to avoid implicit cast when setting nan
        frame["3rd"] = frame["3rd"].astype("float")
        frame.loc[1::11, "1st"] = np.nan
        frame.loc[3::17, "2nd"] = np.nan
        frame.loc[7::19, "3rd"] = np.nan
        frame.loc[8::19, "3rd"] = np.nan
        frame.loc[9::19, "3rd"] = np.nan

    return frame


@pytest.mark.slow
@pytest.mark.parametrize("seed_nans", [True, False])
@pytest.mark.parametrize("num_rows", [10, 50])
@pytest.mark.parametrize("max_int", [5, 20])
@pytest.mark.parametrize("keys", ["1st", "2nd", ["1st", "2nd"]], ids=repr)
@pytest.mark.parametrize("bins", [None, [0, 5]], ids=repr)
@pytest.mark.parametrize("isort", [True, False])
@pytest.mark.parametrize("normalize, name", [(True, "proportion"), (False, "count")])
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("dropna", [True, False])
def test_series_groupby_value_counts(
    seed_nans,
    num_rows,
    max_int,
    keys,
    bins,
    isort,
    normalize,
    name,
    sort,
    ascending,
    dropna,
):
    df = seed_df(seed_nans, num_rows, max_int)

    def rebuild_index(df):
        arr = list(map(df.index.get_level_values, range(df.index.nlevels)))
        df.index = MultiIndex.from_arrays(arr, names=df.index.names)
        return df

    kwargs = {
        "normalize": normalize,
        "sort": sort,
        "ascending": ascending,
        "dropna": dropna,
        "bins": bins,
    }

    gr = df.groupby(keys, sort=isort)
    left = gr["3rd"].value_counts(**kwargs)

    gr = df.groupby(keys, sort=isort)
    right = gr["3rd"].apply(Series.value_counts, **kwargs)
    right.index.names = right.index.names[:-1] + ["3rd"]
    # https://github.com/pandas-dev/pandas/issues/49909
    right = right.rename(name)

    # have to sort on index because of unstable sort on values
    left, right = map(rebuild_index, (left, right))  # xref GH9212
    tm.assert_series_equal(left.sort_index(), right.sort_index())


@pytest.mark.parametrize("utc", [True, False])
def test_series_groupby_value_counts_with_grouper(utc):
    # GH28479
    df = DataFrame(
        {
            "Timestamp": [
                1565083561,
                1565083561 + 86400,
                1565083561 + 86500,
                1565083561 + 86400 * 2,
                1565083561 + 86400 * 3,
                1565083561 + 86500 * 3,
                1565083561 + 86400 * 4,
            ],
            "Food": ["apple", "apple", "banana", "banana", "orange", "orange", "pear"],
        }
    ).drop([3])

    df["Datetime"] = to_datetime(df["Timestamp"], utc=utc, unit="s")
    dfg = df.groupby(Grouper(freq="1D", key="Datetime"))

    # have to sort on index because of unstable sort on values xref GH9212
    result = dfg["Food"].value_counts().sort_index()
    expected = dfg["Food"].apply(Series.value_counts).sort_index()
    expected.index.names = result.index.names
    # https://github.com/pandas-dev/pandas/issues/49909
    expected = expected.rename("count")

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("columns", [["A", "B"], ["A", "B", "C"]])
def test_series_groupby_value_counts_empty(columns):
    # GH39172
    df = DataFrame(columns=columns)
    dfg = df.groupby(columns[:-1])

    result = dfg[columns[-1]].value_counts()
    expected = Series([], dtype=result.dtype, name="count")
    expected.index = MultiIndex.from_arrays([[]] * len(columns), names=columns)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("columns", [["A", "B"], ["A", "B", "C"]])
def test_series_groupby_value_counts_one_row(columns):
    # GH42618
    df = DataFrame(data=[range(len(columns))], columns=columns)
    dfg = df.groupby(columns[:-1])

    result = dfg[columns[-1]].value_counts()
    expected = df.value_counts()

    tm.assert_series_equal(result, expected)


def test_series_groupby_value_counts_on_categorical():
    # GH38672

    s = Series(Categorical(["a"], categories=["a", "b"]))
    result = s.groupby([0]).value_counts()

    expected = Series(
        data=[1, 0],
        index=MultiIndex.from_arrays(
            [
                np.array([0, 0]),
                CategoricalIndex(
                    ["a", "b"], categories=["a", "b"], ordered=False, dtype="category"
                ),
            ]
        ),
        name="count",
    )

    # Expected:
    # 0  a    1
    #    b    0
    # dtype: int64

    tm.assert_series_equal(result, expected)


def test_series_groupby_value_counts_no_sort():
    # GH#50482
    df = DataFrame(
        {
            "gender": ["male", "male", "female", "male", "female", "male"],
            "education": ["low", "medium", "high", "low", "high", "low"],
            "country": ["US", "FR", "US", "FR", "FR", "FR"],
        }
    )
    gb = df.groupby(["country", "gender"], sort=False)["education"]
    result = gb.value_counts(sort=False)
    index = MultiIndex(
        levels=[["US", "FR"], ["male", "female"], ["low", "medium", "high"]],
        codes=[[0, 1, 0, 1, 1], [0, 0, 1, 0, 1], [0, 1, 2, 0, 2]],
        names=["country", "gender", "education"],
    )
    expected = Series([1, 1, 1, 2, 1], index=index, name="count")
    tm.assert_series_equal(result, expected)


@pytest.fixture
def education_df():
    return DataFrame(
        {
            "gender": ["male", "male", "female", "male", "female", "male"],
            "education": ["low", "medium", "high", "low", "high", "low"],
            "country": ["US", "FR", "US", "FR", "FR", "FR"],
        }
    )


def test_axis(education_df):
    msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gp = education_df.groupby("country", axis=1)
    with pytest.raises(NotImplementedError, match="axis"):
        gp.value_counts()


def test_bad_subset(education_df):
    gp = education_df.groupby("country")
    with pytest.raises(ValueError, match="subset"):
        gp.value_counts(subset=["country"])


def test_basic(education_df, request):
    # gh43564
    if Version(np.__version__) >= Version("1.25"):
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    "pandas default unstable sorting of duplicates"
                    "issue with numpy>=1.25 with AVX instructions"
                ),
                strict=False,
            )
        )
    result = education_df.groupby("country")[["gender", "education"]].value_counts(
        normalize=True
    )
    expected = Series(
        data=[0.5, 0.25, 0.25, 0.5, 0.5],
        index=MultiIndex.from_tuples(
            [
                ("FR", "male", "low"),
                ("FR", "female", "high"),
                ("FR", "male", "medium"),
                ("US", "female", "high"),
                ("US", "male", "low"),
            ],
            names=["country", "gender", "education"],
        ),
        name="proportion",
    )
    tm.assert_series_equal(result, expected)


def _frame_value_counts(df, keys, normalize, sort, ascending):
    return df[keys].value_counts(normalize=normalize, sort=sort, ascending=ascending)


@pytest.mark.parametrize("groupby", ["column", "array", "function"])
@pytest.mark.parametrize("normalize, name", [(True, "proportion"), (False, "count")])
@pytest.mark.parametrize(
    "sort, ascending",
    [
        (False, None),
        (True, True),
        (True, False),
    ],
)
@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize("frame", [True, False])
def test_against_frame_and_seriesgroupby(
    education_df,
    groupby,
    normalize,
    name,
    sort,
    ascending,
    as_index,
    frame,
    request,
    using_infer_string,
):
    # test all parameters:
    # - Use column, array or function as by= parameter
    # - Whether or not to normalize
    # - Whether or not to sort and how
    # - Whether or not to use the groupby as an index
    # - 3-way compare against:
    #   - apply with :meth:`~DataFrame.value_counts`
    #   - `~SeriesGroupBy.value_counts`
    if Version(np.__version__) >= Version("1.25") and frame and sort and normalize:
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    "pandas default unstable sorting of duplicates"
                    "issue with numpy>=1.25 with AVX instructions"
                ),
                strict=False,
            )
        )
    by = {
        "column": "country",
        "array": education_df["country"].values,
        "function": lambda x: education_df["country"][x] == "US",
    }[groupby]

    gp = education_df.groupby(by=by, as_index=as_index)
    result = gp[["gender", "education"]].value_counts(
        normalize=normalize, sort=sort, ascending=ascending
    )
    if frame:
        # compare against apply with DataFrame value_counts
        warn = FutureWarning if groupby == "column" else None
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(warn, match=msg):
            expected = gp.apply(
                _frame_value_counts, ["gender", "education"], normalize, sort, ascending
            )

        if as_index:
            tm.assert_series_equal(result, expected)
        else:
            name = "proportion" if normalize else "count"
            expected = expected.reset_index().rename({0: name}, axis=1)
            if groupby == "column":
                expected = expected.rename({"level_0": "country"}, axis=1)
                expected["country"] = np.where(expected["country"], "US", "FR")
            elif groupby == "function":
                expected["level_0"] = expected["level_0"] == 1
            else:
                expected["level_0"] = np.where(expected["level_0"], "US", "FR")
            tm.assert_frame_equal(result, expected)
    else:
        # compare against SeriesGroupBy value_counts
        education_df["both"] = education_df["gender"] + "-" + education_df["education"]
        expected = gp["both"].value_counts(
            normalize=normalize, sort=sort, ascending=ascending
        )
        expected.name = name
        if as_index:
            index_frame = expected.index.to_frame(index=False)
            index_frame["gender"] = index_frame["both"].str.split("-").str.get(0)
            index_frame["education"] = index_frame["both"].str.split("-").str.get(1)
            del index_frame["both"]
            index_frame2 = index_frame.rename({0: None}, axis=1)
            expected.index = MultiIndex.from_frame(index_frame2)

            if index_frame2.columns.isna()[0]:
                # with using_infer_string, the columns in index_frame as string
                #  dtype, which makes the rename({0: None}) above use np.nan
                #  instead of None, so we need to set None more explicitly.
                expected.index.names = [None] + expected.index.names[1:]
            tm.assert_series_equal(result, expected)
        else:
            expected.insert(1, "gender", expected["both"].str.split("-").str.get(0))
            expected.insert(2, "education", expected["both"].str.split("-").str.get(1))
            if using_infer_string:
                expected = expected.astype({"gender": "str", "education": "str"})
            del expected["both"]
            tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize(
    "sort, ascending, expected_rows, expected_count, expected_group_size",
    [
        (False, None, [0, 1, 2, 3, 4], [1, 1, 1, 2, 1], [1, 3, 1, 3, 1]),
        (True, False, [3, 0, 1, 2, 4], [2, 1, 1, 1, 1], [3, 1, 3, 1, 1]),
        (True, True, [0, 1, 2, 4, 3], [1, 1, 1, 1, 2], [1, 3, 1, 1, 3]),
    ],
)
def test_compound(
    education_df,
    normalize,
    sort,
    ascending,
    expected_rows,
    expected_count,
    expected_group_size,
    any_string_dtype,
    using_infer_string,
):
    dtype = any_string_dtype
    education_df = education_df.astype(dtype)
    education_df.columns = education_df.columns.astype(dtype)
    # Multiple groupby keys and as_index=False
    gp = education_df.groupby(["country", "gender"], as_index=False, sort=False)
    result = gp["education"].value_counts(
        normalize=normalize, sort=sort, ascending=ascending
    )
    expected = DataFrame()
    for column in ["country", "gender", "education"]:
        expected[column] = [education_df[column][row] for row in expected_rows]
        expected = expected.astype(dtype)
        expected.columns = expected.columns.astype(dtype)
    if normalize:
        expected["proportion"] = expected_count
        expected["proportion"] /= expected_group_size
        if dtype == "string[pyarrow]":
            # TODO(nullable) also string[python] should return nullable dtypes
            expected["proportion"] = expected["proportion"].convert_dtypes()
    else:
        expected["count"] = expected_count
        if dtype == "string[pyarrow]":
            expected["count"] = expected["count"].convert_dtypes()
    if using_infer_string and dtype == object:
        expected = expected.astype(
            {"country": "str", "gender": "str", "education": "str"}
        )

    tm.assert_frame_equal(result, expected)


@pytest.fixture
def animals_df():
    return DataFrame(
        {"key": [1, 1, 1, 1], "num_legs": [2, 4, 4, 6], "num_wings": [2, 0, 0, 0]},
        index=["falcon", "dog", "cat", "ant"],
    )


@pytest.mark.parametrize(
    "sort, ascending, normalize, name, expected_data, expected_index",
    [
        (False, None, False, "count", [1, 2, 1], [(1, 1, 1), (2, 4, 6), (2, 0, 0)]),
        (True, True, False, "count", [1, 1, 2], [(1, 1, 1), (2, 6, 4), (2, 0, 0)]),
        (True, False, False, "count", [2, 1, 1], [(1, 1, 1), (4, 2, 6), (0, 2, 0)]),
        (
            True,
            False,
            True,
            "proportion",
            [0.5, 0.25, 0.25],
            [(1, 1, 1), (4, 2, 6), (0, 2, 0)],
        ),
    ],
)
def test_data_frame_value_counts(
    animals_df, sort, ascending, normalize, name, expected_data, expected_index
):
    # 3-way compare with :meth:`~DataFrame.value_counts`
    # Tests from frame/methods/test_value_counts.py
    result_frame = animals_df.value_counts(
        sort=sort, ascending=ascending, normalize=normalize
    )
    expected = Series(
        data=expected_data,
        index=MultiIndex.from_arrays(
            expected_index, names=["key", "num_legs", "num_wings"]
        ),
        name=name,
    )
    tm.assert_series_equal(result_frame, expected)

    result_frame_groupby = animals_df.groupby("key").value_counts(
        sort=sort, ascending=ascending, normalize=normalize
    )

    tm.assert_series_equal(result_frame_groupby, expected)


@pytest.fixture
def nulls_df():
    n = np.nan
    return DataFrame(
        {
            "A": [1, 1, n, 4, n, 6, 6, 6, 6],
            "B": [1, 1, 3, n, n, 6, 6, 6, 6],
            "C": [1, 2, 3, 4, 5, 6, n, 8, n],
            "D": [1, 2, 3, 4, 5, 6, 7, n, n],
        }
    )


@pytest.mark.parametrize(
    "group_dropna, count_dropna, expected_rows, expected_values",
    [
        (
            False,
            False,
            [0, 1, 3, 5, 7, 6, 8, 2, 4],
            [0.5, 0.5, 1.0, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0],
        ),
        (False, True, [0, 1, 3, 5, 2, 4], [0.5, 0.5, 1.0, 1.0, 1.0, 1.0]),
        (True, False, [0, 1, 5, 7, 6, 8], [0.5, 0.5, 0.25, 0.25, 0.25, 0.25]),
        (True, True, [0, 1, 5], [0.5, 0.5, 1.0]),
    ],
)
def test_dropna_combinations(
    nulls_df, group_dropna, count_dropna, expected_rows, expected_values, request
):
    if Version(np.__version__) >= Version("1.25") and not group_dropna:
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    "pandas default unstable sorting of duplicates"
                    "issue with numpy>=1.25 with AVX instructions"
                ),
                strict=False,
            )
        )
    gp = nulls_df.groupby(["A", "B"], dropna=group_dropna)
    result = gp.value_counts(normalize=True, sort=True, dropna=count_dropna)
    columns = DataFrame()
    for column in nulls_df.columns:
        columns[column] = [nulls_df[column][row] for row in expected_rows]
    index = MultiIndex.from_frame(columns)
    expected = Series(data=expected_values, index=index, name="proportion")
    tm.assert_series_equal(result, expected)


@pytest.fixture
def names_with_nulls_df(nulls_fixture):
    return DataFrame(
        {
            "key": [1, 1, 1, 1],
            "first_name": ["John", "Anne", "John", "Beth"],
            "middle_name": ["Smith", nulls_fixture, nulls_fixture, "Louise"],
        },
    )


@pytest.mark.parametrize(
    "dropna, expected_data, expected_index",
    [
        (
            True,
            [1, 1],
            MultiIndex.from_arrays(
                [(1, 1), ("Beth", "John"), ("Louise", "Smith")],
                names=["key", "first_name", "middle_name"],
            ),
        ),
        (
            False,
            [1, 1, 1, 1],
            MultiIndex(
                levels=[
                    Index([1]),
                    Index(["Anne", "Beth", "John"]),
                    Index(["Louise", "Smith", np.nan]),
                ],
                codes=[[0, 0, 0, 0], [0, 1, 2, 2], [2, 0, 1, 2]],
                names=["key", "first_name", "middle_name"],
            ),
        ),
    ],
)
@pytest.mark.parametrize("normalize, name", [(False, "count"), (True, "proportion")])
def test_data_frame_value_counts_dropna(
    names_with_nulls_df, dropna, normalize, name, expected_data, expected_index
):
    # GH 41334
    # 3-way compare with :meth:`~DataFrame.value_counts`
    # Tests with nulls from frame/methods/test_value_counts.py
    result_frame = names_with_nulls_df.value_counts(dropna=dropna, normalize=normalize)
    expected = Series(
        data=expected_data,
        index=expected_index,
        name=name,
    )
    if normalize:
        expected /= float(len(expected_data))

    tm.assert_series_equal(result_frame, expected)

    result_frame_groupby = names_with_nulls_df.groupby("key").value_counts(
        dropna=dropna, normalize=normalize
    )

    tm.assert_series_equal(result_frame_groupby, expected)


@pytest.mark.parametrize("as_index", [False, True])
@pytest.mark.parametrize("observed", [False, True])
@pytest.mark.parametrize(
    "normalize, name, expected_data",
    [
        (
            False,
            "count",
            np.array([2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64),
        ),
        (
            True,
            "proportion",
            np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
        ),
    ],
)
def test_categorical_single_grouper_with_only_observed_categories(
    education_df, as_index, observed, normalize, name, expected_data, request
):
    # Test single categorical grouper with only observed grouping categories
    # when non-groupers are also categorical
    if Version(np.__version__) >= Version("1.25"):
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    "pandas default unstable sorting of duplicates"
                    "issue with numpy>=1.25 with AVX instructions"
                ),
                strict=False,
            )
        )

    gp = education_df.astype("category").groupby(
        "country", as_index=as_index, observed=observed
    )
    result = gp.value_counts(normalize=normalize)

    expected_index = MultiIndex.from_tuples(
        [
            ("FR", "male", "low"),
            ("FR", "female", "high"),
            ("FR", "male", "medium"),
            ("FR", "female", "low"),
            ("FR", "female", "medium"),
            ("FR", "male", "high"),
            ("US", "female", "high"),
            ("US", "male", "low"),
            ("US", "female", "low"),
            ("US", "female", "medium"),
            ("US", "male", "high"),
            ("US", "male", "medium"),
        ],
        names=["country", "gender", "education"],
    )

    expected_series = Series(
        data=expected_data,
        index=expected_index,
        name=name,
    )
    for i in range(3):
        expected_series.index = expected_series.index.set_levels(
            CategoricalIndex(expected_series.index.levels[i]), level=i
        )

    if as_index:
        tm.assert_series_equal(result, expected_series)
    else:
        expected = expected_series.reset_index(
            name="proportion" if normalize else "count"
        )
        tm.assert_frame_equal(result, expected)


def assert_categorical_single_grouper(
    education_df, as_index, observed, expected_index, normalize, name, expected_data
):
    # Test single categorical grouper when non-groupers are also categorical
    education_df = education_df.copy().astype("category")

    # Add non-observed grouping categories
    education_df["country"] = education_df["country"].cat.add_categories(["ASIA"])

    gp = education_df.groupby("country", as_index=as_index, observed=observed)
    result = gp.value_counts(normalize=normalize)

    expected_series = Series(
        data=expected_data,
        index=MultiIndex.from_tuples(
            expected_index,
            names=["country", "gender", "education"],
        ),
        name=name,
    )
    for i in range(3):
        index_level = CategoricalIndex(expected_series.index.levels[i])
        if i == 0:
            index_level = index_level.set_categories(
                education_df["country"].cat.categories
            )
        expected_series.index = expected_series.index.set_levels(index_level, level=i)

    if as_index:
        tm.assert_series_equal(result, expected_series)
    else:
        expected = expected_series.reset_index(name=name)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize(
    "normalize, name, expected_data",
    [
        (
            False,
            "count",
            np.array([2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64),
        ),
        (
            True,
            "proportion",
            np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
        ),
    ],
)
def test_categorical_single_grouper_observed_true(
    education_df, as_index, normalize, name, expected_data, request
):
    # GH#46357

    if Version(np.__version__) >= Version("1.25"):
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    "pandas default unstable sorting of duplicates"
                    "issue with numpy>=1.25 with AVX instructions"
                ),
                strict=False,
            )
        )

    expected_index = [
        ("FR", "male", "low"),
        ("FR", "female", "high"),
        ("FR", "male", "medium"),
        ("FR", "female", "low"),
        ("FR", "female", "medium"),
        ("FR", "male", "high"),
        ("US", "female", "high"),
        ("US", "male", "low"),
        ("US", "female", "low"),
        ("US", "female", "medium"),
        ("US", "male", "high"),
        ("US", "male", "medium"),
    ]

    assert_categorical_single_grouper(
        education_df=education_df,
        as_index=as_index,
        observed=True,
        expected_index=expected_index,
        normalize=normalize,
        name=name,
        expected_data=expected_data,
    )


@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize(
    "normalize, name, expected_data",
    [
        (
            False,
            "count",
            np.array(
                [2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64
            ),
        ),
        (
            True,
            "proportion",
            np.array(
                [
                    0.5,
                    0.25,
                    0.25,
                    0.0,
                    0.0,
                    0.0,
                    0.5,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ),
    ],
)
def test_categorical_single_grouper_observed_false(
    education_df, as_index, normalize, name, expected_data, request
):
    # GH#46357

    if Version(np.__version__) >= Version("1.25"):
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    "pandas default unstable sorting of duplicates"
                    "issue with numpy>=1.25 with AVX instructions"
                ),
                strict=False,
            )
        )

    expected_index = [
        ("FR", "male", "low"),
        ("FR", "female", "high"),
        ("FR", "male", "medium"),
        ("FR", "female", "low"),
        ("FR", "female", "medium"),
        ("FR", "male", "high"),
        ("US", "female", "high"),
        ("US", "male", "low"),
        ("US", "female", "low"),
        ("US", "female", "medium"),
        ("US", "male", "high"),
        ("US", "male", "medium"),
        ("ASIA", "female", "high"),
        ("ASIA", "female", "low"),
        ("ASIA", "female", "medium"),
        ("ASIA", "male", "high"),
        ("ASIA", "male", "low"),
        ("ASIA", "male", "medium"),
    ]

    assert_categorical_single_grouper(
        education_df=education_df,
        as_index=as_index,
        observed=False,
        expected_index=expected_index,
        normalize=normalize,
        name=name,
        expected_data=expected_data,
    )


@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize(
    "observed, expected_index",
    [
        (
            False,
            [
                ("FR", "high", "female"),
                ("FR", "high", "male"),
                ("FR", "low", "male"),
                ("FR", "low", "female"),
                ("FR", "medium", "male"),
                ("FR", "medium", "female"),
                ("US", "high", "female"),
                ("US", "high", "male"),
                ("US", "low", "male"),
                ("US", "low", "female"),
                ("US", "medium", "female"),
                ("US", "medium", "male"),
            ],
        ),
        (
            True,
            [
                ("FR", "high", "female"),
                ("FR", "low", "male"),
                ("FR", "medium", "male"),
                ("US", "high", "female"),
                ("US", "low", "male"),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "normalize, name, expected_data",
    [
        (
            False,
            "count",
            np.array([1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 0, 0], dtype=np.int64),
        ),
        (
            True,
            "proportion",
            # NaN values corresponds to non-observed groups
            np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ),
    ],
)
def test_categorical_multiple_groupers(
    education_df, as_index, observed, expected_index, normalize, name, expected_data
):
    # GH#46357

    # Test multiple categorical groupers when non-groupers are non-categorical
    education_df = education_df.copy()
    education_df["country"] = education_df["country"].astype("category")
    education_df["education"] = education_df["education"].astype("category")

    gp = education_df.groupby(
        ["country", "education"], as_index=as_index, observed=observed
    )
    result = gp.value_counts(normalize=normalize)

    expected_series = Series(
        data=expected_data[expected_data > 0.0] if observed else expected_data,
        index=MultiIndex.from_tuples(
            expected_index,
            names=["country", "education", "gender"],
        ),
        name=name,
    )
    for i in range(2):
        expected_series.index = expected_series.index.set_levels(
            CategoricalIndex(expected_series.index.levels[i]), level=i
        )

    if as_index:
        tm.assert_series_equal(result, expected_series)
    else:
        expected = expected_series.reset_index(
            name="proportion" if normalize else "count"
        )
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("as_index", [False, True])
@pytest.mark.parametrize("observed", [False, True])
@pytest.mark.parametrize(
    "normalize, name, expected_data",
    [
        (
            False,
            "count",
            np.array([2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64),
        ),
        (
            True,
            "proportion",
            # NaN values corresponds to non-observed groups
            np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
        ),
    ],
)
def test_categorical_non_groupers(
    education_df, as_index, observed, normalize, name, expected_data, request
):
    # GH#46357 Test non-observed categories are included in the result,
    # regardless of `observed`

    if Version(np.__version__) >= Version("1.25"):
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    "pandas default unstable sorting of duplicates"
                    "issue with numpy>=1.25 with AVX instructions"
                ),
                strict=False,
            )
        )

    education_df = education_df.copy()
    education_df["gender"] = education_df["gender"].astype("category")
    education_df["education"] = education_df["education"].astype("category")

    gp = education_df.groupby("country", as_index=as_index, observed=observed)
    result = gp.value_counts(normalize=normalize)

    expected_index = [
        ("FR", "male", "low"),
        ("FR", "female", "high"),
        ("FR", "male", "medium"),
        ("FR", "female", "low"),
        ("FR", "female", "medium"),
        ("FR", "male", "high"),
        ("US", "female", "high"),
        ("US", "male", "low"),
        ("US", "female", "low"),
        ("US", "female", "medium"),
        ("US", "male", "high"),
        ("US", "male", "medium"),
    ]
    expected_series = Series(
        data=expected_data,
        index=MultiIndex.from_tuples(
            expected_index,
            names=["country", "gender", "education"],
        ),
        name=name,
    )
    for i in range(1, 3):
        expected_series.index = expected_series.index.set_levels(
            CategoricalIndex(expected_series.index.levels[i]), level=i
        )

    if as_index:
        tm.assert_series_equal(result, expected_series)
    else:
        expected = expected_series.reset_index(
            name="proportion" if normalize else "count"
        )
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "normalize, expected_label, expected_values",
    [
        (False, "count", [1, 1, 1]),
        (True, "proportion", [0.5, 0.5, 1.0]),
    ],
)
def test_mixed_groupings(normalize, expected_label, expected_values):
    # Test multiple groupings
    df = DataFrame({"A": [1, 2, 1], "B": [1, 2, 3]})
    gp = df.groupby([[4, 5, 4], "A", lambda i: 7 if i == 1 else 8], as_index=False)
    result = gp.value_counts(sort=True, normalize=normalize)
    expected = DataFrame(
        {
            "level_0": np.array([4, 4, 5], dtype=int),
            "A": [1, 1, 2],
            "level_2": [8, 8, 7],
            "B": [1, 3, 2],
            expected_label: expected_values,
        }
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "test, columns, expected_names",
    [
        ("repeat", list("abbde"), ["a", None, "d", "b", "b", "e"]),
        ("level", list("abcd") + ["level_1"], ["a", None, "d", "b", "c", "level_1"]),
    ],
)
@pytest.mark.parametrize("as_index", [False, True])
def test_column_label_duplicates(test, columns, expected_names, as_index):
    # GH 44992
    # Test for duplicate input column labels and generated duplicate labels
    df = DataFrame([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]], columns=columns)
    expected_data = [(1, 0, 7, 3, 5, 9), (2, 1, 8, 4, 6, 10)]
    keys = ["a", np.array([0, 1], dtype=np.int64), "d"]
    result = df.groupby(keys, as_index=as_index).value_counts()
    if as_index:
        expected = Series(
            data=(1, 1),
            index=MultiIndex.from_tuples(
                expected_data,
                names=expected_names,
            ),
            name="count",
        )
        tm.assert_series_equal(result, expected)
    else:
        expected_data = [list(row) + [1] for row in expected_data]
        expected_columns = list(expected_names)
        expected_columns[1] = "level_1"
        expected_columns.append("count")
        expected = DataFrame(expected_data, columns=expected_columns)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "normalize, expected_label",
    [
        (False, "count"),
        (True, "proportion"),
    ],
)
def test_result_label_duplicates(normalize, expected_label):
    # Test for result column label duplicating an input column label
    gb = DataFrame([[1, 2, 3]], columns=["a", "b", expected_label]).groupby(
        "a", as_index=False
    )
    msg = f"Column label '{expected_label}' is duplicate of result column"
    with pytest.raises(ValueError, match=msg):
        gb.value_counts(normalize=normalize)


def test_ambiguous_grouping():
    # Test that groupby is not confused by groupings length equal to row count
    df = DataFrame({"a": [1, 1]})
    gb = df.groupby(np.array([1, 1], dtype=np.int64))
    result = gb.value_counts()
    expected = Series(
        [2], index=MultiIndex.from_tuples([[1, 1]], names=[None, "a"]), name="count"
    )
    tm.assert_series_equal(result, expected)


def test_subset_overlaps_gb_key_raises():
    # GH 46383
    df = DataFrame({"c1": ["a", "b", "c"], "c2": ["x", "y", "y"]}, index=[0, 1, 1])
    msg = "Keys {'c1'} in subset cannot be in the groupby column keys."
    with pytest.raises(ValueError, match=msg):
        df.groupby("c1").value_counts(subset=["c1"])


def test_subset_doesnt_exist_in_frame():
    # GH 46383
    df = DataFrame({"c1": ["a", "b", "c"], "c2": ["x", "y", "y"]}, index=[0, 1, 1])
    msg = "Keys {'c3'} in subset do not exist in the DataFrame."
    with pytest.raises(ValueError, match=msg):
        df.groupby("c1").value_counts(subset=["c3"])


def test_subset():
    # GH 46383
    df = DataFrame({"c1": ["a", "b", "c"], "c2": ["x", "y", "y"]}, index=[0, 1, 1])
    result = df.groupby(level=0).value_counts(subset=["c2"])
    expected = Series(
        [1, 2],
        index=MultiIndex.from_arrays([[0, 1], ["x", "y"]], names=[None, "c2"]),
        name="count",
    )
    tm.assert_series_equal(result, expected)


def test_subset_duplicate_columns():
    # GH 46383
    df = DataFrame(
        [["a", "x", "x"], ["b", "y", "y"], ["b", "y", "y"]],
        index=[0, 1, 1],
        columns=["c1", "c2", "c2"],
    )
    result = df.groupby(level=0).value_counts(subset=["c2"])
    expected = Series(
        [1, 2],
        index=MultiIndex.from_arrays(
            [[0, 1], ["x", "y"], ["x", "y"]], names=[None, "c2", "c2"]
        ),
        name="count",
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("utc", [True, False])
def test_value_counts_time_grouper(utc, unit):
    # GH#50486
    df = DataFrame(
        {
            "Timestamp": [
                1565083561,
                1565083561 + 86400,
                1565083561 + 86500,
                1565083561 + 86400 * 2,
                1565083561 + 86400 * 3,
                1565083561 + 86500 * 3,
                1565083561 + 86400 * 4,
            ],
            "Food": ["apple", "apple", "banana", "banana", "orange", "orange", "pear"],
        }
    ).drop([3])

    df["Datetime"] = to_datetime(df["Timestamp"], utc=utc, unit="s").dt.as_unit(unit)
    gb = df.groupby(Grouper(freq="1D", key="Datetime"))
    result = gb.value_counts()
    dates = to_datetime(
        ["2019-08-06", "2019-08-07", "2019-08-09", "2019-08-10"], utc=utc
    ).as_unit(unit)
    timestamps = df["Timestamp"].unique()
    index = MultiIndex(
        levels=[dates, timestamps, ["apple", "banana", "orange", "pear"]],
        codes=[[0, 1, 1, 2, 2, 3], range(6), [0, 0, 1, 2, 2, 3]],
        names=["Datetime", "Timestamp", "Food"],
    )
    expected = Series(1, index=index, name="count")
    tm.assert_series_equal(result, expected)


def test_value_counts_integer_columns():
    # GH#55627
    df = DataFrame({1: ["a", "a", "a"], 2: ["a", "a", "d"], 3: ["a", "b", "c"]})
    gp = df.groupby([1, 2], as_index=False, sort=False)
    result = gp[3].value_counts()
    expected = DataFrame(
        {1: ["a", "a", "a"], 2: ["a", "a", "d"], 3: ["a", "b", "c"], "count": 1}
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("vc_sort", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
def test_value_counts_sort(sort, vc_sort, normalize):
    # GH#55951
    df = DataFrame({"a": [2, 1, 1, 1], 0: [3, 4, 3, 3]})
    gb = df.groupby("a", sort=sort)
    result = gb.value_counts(sort=vc_sort, normalize=normalize)

    if normalize:
        values = [2 / 3, 1 / 3, 1.0]
    else:
        values = [2, 1, 1]
    index = MultiIndex(
        levels=[[1, 2], [3, 4]], codes=[[0, 0, 1], [0, 1, 0]], names=["a", 0]
    )
    expected = Series(values, index=index, name="proportion" if normalize else "count")
    if sort and vc_sort:
        taker = [0, 1, 2]
    elif sort and not vc_sort:
        taker = [0, 1, 2]
    elif not sort and vc_sort:
        taker = [0, 2, 1]
    else:
        taker = [2, 1, 0]
    expected = expected.take(taker)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("vc_sort", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
def test_value_counts_sort_categorical(sort, vc_sort, normalize):
    # GH#55951
    df = DataFrame({"a": [2, 1, 1, 1], 0: [3, 4, 3, 3]}, dtype="category")
    gb = df.groupby("a", sort=sort, observed=True)
    result = gb.value_counts(sort=vc_sort, normalize=normalize)

    if normalize:
        values = [2 / 3, 1 / 3, 1.0, 0.0]
    else:
        values = [2, 1, 1, 0]
    name = "proportion" if normalize else "count"
    expected = DataFrame(
        {
            "a": Categorical([1, 1, 2, 2]),
            0: Categorical([3, 4, 3, 4]),
            name: values,
        }
    ).set_index(["a", 0])[name]
    if sort and vc_sort:
        taker = [0, 1, 2, 3]
    elif sort and not vc_sort:
        taker = [0, 1, 2, 3]
    elif not sort and vc_sort:
        taker = [0, 2, 1, 3]
    else:
        taker = [2, 3, 0, 1]
    expected = expected.take(taker)

    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\combinatorial\tests\test_comb_numbers.py ===
import string

from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.function import (diff, expand_func)
from sympy.core import (EulerGamma, TribonacciConstant)
from sympy.core.numbers import (Float, I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.numbers import carmichael
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.integers import floor
from sympy.polys.polytools import cancel
from sympy.series.limits import limit, Limit
from sympy.series.order import O
from sympy.functions import (
    bernoulli, harmonic, bell, fibonacci, tribonacci, lucas, euler, catalan,
    genocchi, andre, partition, divisor_sigma, udivisor_sigma, legendre_symbol,
    jacobi_symbol, kronecker_symbol, mobius,
    primenu, primeomega, totient, reduced_totient, primepi,
    motzkin, binomial, gamma, sqrt, cbrt, hyper, log, digamma,
    trigamma, polygamma, factorial, sin, cos, cot, polylog, zeta, dirichlet_eta)
from sympy.functions.combinatorial.numbers import _nT
from sympy.ntheory.factor_ import factorint

from sympy.core.expr import unchanged
from sympy.core.numbers import GoldenRatio, Integer

from sympy.testing.pytest import raises, nocache_fail, warns_deprecated_sympy
from sympy.abc import x


def test_carmichael():
    with warns_deprecated_sympy():
        assert carmichael.is_prime(2821) == False


def test_bernoulli():
    assert bernoulli(0) == 1
    assert bernoulli(1) == Rational(1, 2)
    assert bernoulli(2) == Rational(1, 6)
    assert bernoulli(3) == 0
    assert bernoulli(4) == Rational(-1, 30)
    assert bernoulli(5) == 0
    assert bernoulli(6) == Rational(1, 42)
    assert bernoulli(7) == 0
    assert bernoulli(8) == Rational(-1, 30)
    assert bernoulli(10) == Rational(5, 66)
    assert bernoulli(1000001) == 0

    assert bernoulli(0, x) == 1
    assert bernoulli(1, x) == x - S.Half
    assert bernoulli(2, x) == x**2 - x + Rational(1, 6)
    assert bernoulli(3, x) == x**3 - (3*x**2)/2 + x/2

    # Should be fast; computed with mpmath
    b = bernoulli(1000)
    assert b.p % 10**10 == 7950421099
    assert b.q == 342999030

    b = bernoulli(10**6, evaluate=False).evalf()
    assert str(b) == '-2.23799235765713e+4767529'

    # Issue #8527
    l = Symbol('l', integer=True)
    m = Symbol('m', integer=True, nonnegative=True)
    n = Symbol('n', integer=True, positive=True)
    assert isinstance(bernoulli(2 * l + 1), bernoulli)
    assert isinstance(bernoulli(2 * m + 1), bernoulli)
    assert bernoulli(2 * n + 1) == 0

    assert bernoulli(x, 1) == bernoulli(x)

    assert str(bernoulli(0.0, 2.3).evalf(n=10)) == '1.000000000'
    assert str(bernoulli(1.0).evalf(n=10)) == '0.5000000000'
    assert str(bernoulli(1.2).evalf(n=10)) == '0.4195995367'
    assert str(bernoulli(1.2, 0.8).evalf(n=10)) == '0.2144830348'
    assert str(bernoulli(1.2, -0.8).evalf(n=10)) == '-1.158865646 - 0.6745558744*I'
    assert str(bernoulli(3.0, 1j).evalf(n=10)) == '1.5 - 0.5*I'
    assert str(bernoulli(I).evalf(n=10)) == '0.9268485643 - 0.5821580598*I'
    assert str(bernoulli(I, I).evalf(n=10)) == '0.1267792071 + 0.01947413152*I'
    assert bernoulli(x).evalf() == bernoulli(x)


def test_bernoulli_rewrite():
    from sympy.functions.elementary.piecewise import Piecewise
    n = Symbol('n', integer=True, nonnegative=True)

    assert bernoulli(-1).rewrite(zeta) == pi**2/6
    assert bernoulli(-2).rewrite(zeta) == 2*zeta(3)
    assert not bernoulli(n, -3).rewrite(zeta).has(harmonic)
    assert bernoulli(-4, x).rewrite(zeta) == 4*zeta(5, x)
    assert isinstance(bernoulli(n, x).rewrite(zeta), Piecewise)
    assert bernoulli(n+1, x).rewrite(zeta) == -(n+1) * zeta(-n, x)


def test_fibonacci():
    assert [fibonacci(n) for n in range(-3, 5)] == [2, -1, 1, 0, 1, 1, 2, 3]
    assert fibonacci(100) == 354224848179261915075
    assert [lucas(n) for n in range(-3, 5)] == [-4, 3, -1, 2, 1, 3, 4, 7]
    assert lucas(100) == 792070839848372253127

    assert fibonacci(1, x) == 1
    assert fibonacci(2, x) == x
    assert fibonacci(3, x) == x**2 + 1
    assert fibonacci(4, x) == x**3 + 2*x

    # issue #8800
    n = Dummy('n')
    assert fibonacci(n).limit(n, S.Infinity) is S.Infinity
    assert lucas(n).limit(n, S.Infinity) is S.Infinity

    assert fibonacci(n).rewrite(sqrt) == \
        2**(-n)*sqrt(5)*((1 + sqrt(5))**n - (-sqrt(5) + 1)**n) / 5
    assert fibonacci(n).rewrite(sqrt).subs(n, 10).expand() == fibonacci(10)
    assert fibonacci(n).rewrite(GoldenRatio).subs(n,10).evalf() == \
        Float(fibonacci(10))
    assert lucas(n).rewrite(sqrt) == \
        (fibonacci(n-1).rewrite(sqrt) + fibonacci(n+1).rewrite(sqrt)).simplify()
    assert lucas(n).rewrite(sqrt).subs(n, 10).expand() == lucas(10)
    raises(ValueError, lambda: fibonacci(-3, x))


def test_tribonacci():
    assert [tribonacci(n) for n in range(8)] == [0, 1, 1, 2, 4, 7, 13, 24]
    assert tribonacci(100) == 98079530178586034536500564

    assert tribonacci(0, x) == 0
    assert tribonacci(1, x) == 1
    assert tribonacci(2, x) == x**2
    assert tribonacci(3, x) == x**4 + x
    assert tribonacci(4, x) == x**6 + 2*x**3 + 1
    assert tribonacci(5, x) == x**8 + 3*x**5 + 3*x**2

    n = Dummy('n')
    assert tribonacci(n).limit(n, S.Infinity) is S.Infinity

    w = (-1 + S.ImaginaryUnit * sqrt(3)) / 2
    a = (1 + cbrt(19 + 3*sqrt(33)) + cbrt(19 - 3*sqrt(33))) / 3
    b = (1 + w*cbrt(19 + 3*sqrt(33)) + w**2*cbrt(19 - 3*sqrt(33))) / 3
    c = (1 + w**2*cbrt(19 + 3*sqrt(33)) + w*cbrt(19 - 3*sqrt(33))) / 3
    assert tribonacci(n).rewrite(sqrt) == \
      (a**(n + 1)/((a - b)*(a - c))
      + b**(n + 1)/((b - a)*(b - c))
      + c**(n + 1)/((c - a)*(c - b)))
    assert tribonacci(n).rewrite(sqrt).subs(n, 4).simplify() == tribonacci(4)
    assert tribonacci(n).rewrite(GoldenRatio).subs(n,10).evalf() == \
        Float(tribonacci(10))
    assert tribonacci(n).rewrite(TribonacciConstant) == floor(
            3*TribonacciConstant**n*(102*sqrt(33) + 586)**Rational(1, 3)/
            (-2*(102*sqrt(33) + 586)**Rational(1, 3) + 4 + (102*sqrt(33)
            + 586)**Rational(2, 3)) + S.Half)
    raises(ValueError, lambda: tribonacci(-1, x))


@nocache_fail
def test_bell():
    assert [bell(n) for n in range(8)] == [1, 1, 2, 5, 15, 52, 203, 877]

    assert bell(0, x) == 1
    assert bell(1, x) == x
    assert bell(2, x) == x**2 + x
    assert bell(5, x) == x**5 + 10*x**4 + 25*x**3 + 15*x**2 + x
    assert bell(oo) is S.Infinity
    raises(ValueError, lambda: bell(oo, x))

    raises(ValueError, lambda: bell(-1))
    raises(ValueError, lambda: bell(S.Half))

    X = symbols('x:6')
    # X = (x0, x1, .. x5)
    # at the same time: X[1] = x1, X[2] = x2 for standard readablity.
    # but we must supply zero-based indexed object X[1:] = (x1, .. x5)

    assert bell(6, 2, X[1:]) == 6*X[5]*X[1] + 15*X[4]*X[2] + 10*X[3]**2
    assert bell(
        6, 3, X[1:]) == 15*X[4]*X[1]**2 + 60*X[3]*X[2]*X[1] + 15*X[2]**3

    X = (1, 10, 100, 1000, 10000)
    assert bell(6, 2, X) == (6 + 15 + 10)*10000

    X = (1, 2, 3, 3, 5)
    assert bell(6, 2, X) == 6*5 + 15*3*2 + 10*3**2

    X = (1, 2, 3, 5)
    assert bell(6, 3, X) == 15*5 + 60*3*2 + 15*2**3

    # Dobinski's formula
    n = Symbol('n', integer=True, nonnegative=True)
    # For large numbers, this is too slow
    # For nonintegers, there are significant precision errors
    for i in [0, 2, 3, 7, 13, 42, 55]:
        # Running without the cache this is either very slow or goes into an
        # infinite loop.
        assert bell(i).evalf() == bell(n).rewrite(Sum).evalf(subs={n: i})

    m = Symbol("m")
    assert bell(m).rewrite(Sum) == bell(m)
    assert bell(n, m).rewrite(Sum) == bell(n, m)
    # issue 9184
    n = Dummy('n')
    assert bell(n).limit(n, S.Infinity) is S.Infinity


def test_harmonic():
    n = Symbol("n")
    m = Symbol("m")

    assert harmonic(n, 0) == n
    assert harmonic(n).evalf() == harmonic(n)
    assert harmonic(n, 1) == harmonic(n)
    assert harmonic(1, n) == 1

    assert harmonic(0, 1) == 0
    assert harmonic(1, 1) == 1
    assert harmonic(2, 1) == Rational(3, 2)
    assert harmonic(3, 1) == Rational(11, 6)
    assert harmonic(4, 1) == Rational(25, 12)
    assert harmonic(0, 2) == 0
    assert harmonic(1, 2) == 1
    assert harmonic(2, 2) == Rational(5, 4)
    assert harmonic(3, 2) == Rational(49, 36)
    assert harmonic(4, 2) == Rational(205, 144)
    assert harmonic(0, 3) == 0
    assert harmonic(1, 3) == 1
    assert harmonic(2, 3) == Rational(9, 8)
    assert harmonic(3, 3) == Rational(251, 216)
    assert harmonic(4, 3) == Rational(2035, 1728)

    assert harmonic(oo, -1) is S.NaN
    assert harmonic(oo, 0) is oo
    assert harmonic(oo, S.Half) is oo
    assert harmonic(oo, 1) is oo
    assert harmonic(oo, 2) == (pi**2)/6
    assert harmonic(oo, 3) == zeta(3)
    assert harmonic(oo, Dummy(negative=True)) is S.NaN
    ip = Dummy(integer=True, positive=True)
    if (1/ip <= 1) is True:  #---------------------------------+
        assert None, 'delete this if-block and the next line' #|
    ip = Dummy(even=True, positive=True)  #--------------------+
    assert harmonic(oo, 1/ip) is oo
    assert harmonic(oo, 1 + ip) is zeta(1 + ip)

    assert harmonic(0, m) == 0
    assert harmonic(-1, -1) == 0
    assert harmonic(-1, 0) == -1
    assert harmonic(-1, 1) is S.ComplexInfinity
    assert harmonic(-1, 2) is S.NaN
    assert harmonic(-3, -2) == -5
    assert harmonic(-3, -3) == 9


def test_harmonic_rational():
    ne = S(6)
    no = S(5)
    pe = S(8)
    po = S(9)
    qe = S(10)
    qo = S(13)

    Heee = harmonic(ne + pe/qe)
    Aeee = (-log(10) + 2*(Rational(-1, 4) + sqrt(5)/4)*log(sqrt(-sqrt(5)/8 + Rational(5, 8)))
             + 2*(-sqrt(5)/4 - Rational(1, 4))*log(sqrt(sqrt(5)/8 + Rational(5, 8)))
             + pi*sqrt(2*sqrt(5)/5 + 1)/2 + Rational(13944145, 4720968))

    Heeo = harmonic(ne + pe/qo)
    Aeeo = (-log(26) + 2*log(sin(pi*Rational(3, 13)))*cos(pi*Rational(4, 13)) + 2*log(sin(pi*Rational(2, 13)))*cos(pi*Rational(32, 13))
             + 2*log(sin(pi*Rational(5, 13)))*cos(pi*Rational(80, 13)) - 2*log(sin(pi*Rational(6, 13)))*cos(pi*Rational(5, 13))
             - 2*log(sin(pi*Rational(4, 13)))*cos(pi/13) + pi*cot(pi*Rational(5, 13))/2 - 2*log(sin(pi/13))*cos(pi*Rational(3, 13))
             + Rational(2422020029, 702257080))

    Heoe = harmonic(ne + po/qe)
    Aeoe = (-log(20) + 2*(Rational(1, 4) + sqrt(5)/4)*log(Rational(-1, 4) + sqrt(5)/4)
             + 2*(Rational(-1, 4) + sqrt(5)/4)*log(sqrt(-sqrt(5)/8 + Rational(5, 8)))
             + 2*(-sqrt(5)/4 - Rational(1, 4))*log(sqrt(sqrt(5)/8 + Rational(5, 8)))
             + 2*(-sqrt(5)/4 + Rational(1, 4))*log(Rational(1, 4) + sqrt(5)/4)
             + Rational(11818877030, 4286604231) + pi*sqrt(2*sqrt(5) + 5)/2)

    Heoo = harmonic(ne + po/qo)
    Aeoo = (-log(26) + 2*log(sin(pi*Rational(3, 13)))*cos(pi*Rational(54, 13)) + 2*log(sin(pi*Rational(4, 13)))*cos(pi*Rational(6, 13))
             + 2*log(sin(pi*Rational(6, 13)))*cos(pi*Rational(108, 13)) - 2*log(sin(pi*Rational(5, 13)))*cos(pi/13)
             - 2*log(sin(pi/13))*cos(pi*Rational(5, 13)) + pi*cot(pi*Rational(4, 13))/2
             - 2*log(sin(pi*Rational(2, 13)))*cos(pi*Rational(3, 13)) + Rational(11669332571, 3628714320))

    Hoee = harmonic(no + pe/qe)
    Aoee = (-log(10) + 2*(Rational(-1, 4) + sqrt(5)/4)*log(sqrt(-sqrt(5)/8 + Rational(5, 8)))
             + 2*(-sqrt(5)/4 - Rational(1, 4))*log(sqrt(sqrt(5)/8 + Rational(5, 8)))
             + pi*sqrt(2*sqrt(5)/5 + 1)/2 + Rational(779405, 277704))

    Hoeo = harmonic(no + pe/qo)
    Aoeo = (-log(26) + 2*log(sin(pi*Rational(3, 13)))*cos(pi*Rational(4, 13)) + 2*log(sin(pi*Rational(2, 13)))*cos(pi*Rational(32, 13))
             + 2*log(sin(pi*Rational(5, 13)))*cos(pi*Rational(80, 13)) - 2*log(sin(pi*Rational(6, 13)))*cos(pi*Rational(5, 13))
             - 2*log(sin(pi*Rational(4, 13)))*cos(pi/13) + pi*cot(pi*Rational(5, 13))/2
             - 2*log(sin(pi/13))*cos(pi*Rational(3, 13)) + Rational(53857323, 16331560))

    Hooe = harmonic(no + po/qe)
    Aooe = (-log(20) + 2*(Rational(1, 4) + sqrt(5)/4)*log(Rational(-1, 4) + sqrt(5)/4)
             + 2*(Rational(-1, 4) + sqrt(5)/4)*log(sqrt(-sqrt(5)/8 + Rational(5, 8)))
             + 2*(-sqrt(5)/4 - Rational(1, 4))*log(sqrt(sqrt(5)/8 + Rational(5, 8)))
             + 2*(-sqrt(5)/4 + Rational(1, 4))*log(Rational(1, 4) + sqrt(5)/4)
             + Rational(486853480, 186374097) + pi*sqrt(2*sqrt(5) + 5)/2)

    Hooo = harmonic(no + po/qo)
    Aooo = (-log(26) + 2*log(sin(pi*Rational(3, 13)))*cos(pi*Rational(54, 13)) + 2*log(sin(pi*Rational(4, 13)))*cos(pi*Rational(6, 13))
             + 2*log(sin(pi*Rational(6, 13)))*cos(pi*Rational(108, 13)) - 2*log(sin(pi*Rational(5, 13)))*cos(pi/13)
             - 2*log(sin(pi/13))*cos(pi*Rational(5, 13)) + pi*cot(pi*Rational(4, 13))/2
             - 2*log(sin(pi*Rational(2, 13)))*cos(3*pi/13) + Rational(383693479, 125128080))

    H = [Heee, Heeo, Heoe, Heoo, Hoee, Hoeo, Hooe, Hooo]
    A = [Aeee, Aeeo, Aeoe, Aeoo, Aoee, Aoeo, Aooe, Aooo]
    for h, a in zip(H, A):
        e = expand_func(h).doit()
        assert cancel(e/a) == 1
        assert abs(h.n() - a.n()) < 1e-12


def test_harmonic_evalf():
    assert str(harmonic(1.5).evalf(n=10)) == '1.280372306'
    assert str(harmonic(1.5, 2).evalf(n=10)) == '1.154576311'  # issue 7443
    assert str(harmonic(4.0, -3).evalf(n=10)) == '100.0000000'
    assert str(harmonic(7.0, 1.0).evalf(n=10)) == '2.592857143'
    assert str(harmonic(1, pi).evalf(n=10)) == '1.000000000'
    assert str(harmonic(2, pi).evalf(n=10)) == '1.113314732'
    assert str(harmonic(1000.0, pi).evalf(n=10)) == '1.176241563'
    assert str(harmonic(I).evalf(n=10)) == '0.6718659855 + 1.076674047*I'
    assert str(harmonic(I, I).evalf(n=10)) == '-0.3970915266 + 1.9629689*I'

    assert harmonic(-1.0, 1).evalf() is S.NaN
    assert harmonic(-2.0, 2.0).evalf() is S.NaN

def test_harmonic_rewrite():
    from sympy.functions.elementary.piecewise import Piecewise
    n = Symbol("n")
    m = Symbol("m", integer=True, positive=True)
    x1 = Symbol("x1", positive=True)
    x2 = Symbol("x2", negative=True)

    assert harmonic(n).rewrite(digamma) == polygamma(0, n + 1) + EulerGamma
    assert harmonic(n).rewrite(trigamma) == polygamma(0, n + 1) + EulerGamma
    assert harmonic(n).rewrite(polygamma) == polygamma(0, n + 1) + EulerGamma

    assert harmonic(n,3).rewrite(polygamma) == polygamma(2, n + 1)/2 - polygamma(2, 1)/2
    assert isinstance(harmonic(n,m).rewrite(polygamma), Piecewise)

    assert expand_func(harmonic(n+4)) == harmonic(n) + 1/(n + 4) + 1/(n + 3) + 1/(n + 2) + 1/(n + 1)
    assert expand_func(harmonic(n-4)) == harmonic(n) - 1/(n - 1) - 1/(n - 2) - 1/(n - 3) - 1/n

    assert harmonic(n, m).rewrite("tractable") == harmonic(n, m).rewrite(polygamma)
    assert harmonic(n, x1).rewrite("tractable") == harmonic(n, x1)
    assert harmonic(n, x1 + 1).rewrite("tractable") == zeta(x1 + 1) - zeta(x1 + 1, n + 1)
    assert harmonic(n, x2).rewrite("tractable") == zeta(x2) - zeta(x2, n + 1)

    _k = Dummy("k")
    assert harmonic(n).rewrite(Sum).dummy_eq(Sum(1/_k, (_k, 1, n)))
    assert harmonic(n, m).rewrite(Sum).dummy_eq(Sum(_k**(-m), (_k, 1, n)))


def test_harmonic_calculus():
    y = Symbol("y", positive=True)
    z = Symbol("z", negative=True)
    assert harmonic(x, 1).limit(x, 0) == 0
    assert harmonic(x, y).limit(x, 0) == 0
    assert harmonic(x, 1).series(x, y, 2) == \
            harmonic(y) + (x - y)*zeta(2, y + 1) + O((x - y)**2, (x, y))
    assert limit(harmonic(x, y), x, oo) == harmonic(oo, y)
    assert limit(harmonic(x, y + 1), x, oo) == zeta(y + 1)
    assert limit(harmonic(x, y - 1), x, oo) == harmonic(oo, y - 1)
    assert limit(harmonic(x, z), x, oo) == Limit(harmonic(x, z), x, oo, dir='-')
    assert limit(harmonic(x, z + 1), x, oo) == oo
    assert limit(harmonic(x, z + 2), x, oo) == harmonic(oo, z + 2)
    assert limit(harmonic(x, z - 1), x, oo) == Limit(harmonic(x, z - 1), x, oo, dir='-')


def test_euler():
    assert euler(0) == 1
    assert euler(1) == 0
    assert euler(2) == -1
    assert euler(3) == 0
    assert euler(4) == 5
    assert euler(6) == -61
    assert euler(8) == 1385

    assert euler(20, evaluate=False) != 370371188237525

    n = Symbol('n', integer=True)
    assert euler(n) != -1
    assert euler(n).subs(n, 2) == -1

    assert euler(-1) == S.Pi / 2
    assert euler(-1, 1) == 2*log(2)
    assert euler(-2).evalf() == (2*S.Catalan).evalf()
    assert euler(-3).evalf() == (S.Pi**3 / 16).evalf()
    assert str(euler(2.3).evalf(n=10)) == '-1.052850274'
    assert str(euler(1.2, 3.4).evalf(n=10)) == '3.575613489'
    assert str(euler(I).evalf(n=10)) == '1.248446443 - 0.7675445124*I'
    assert str(euler(I, I).evalf(n=10)) == '0.04812930469 + 0.01052411008*I'

    assert euler(20).evalf() == 370371188237525.0
    assert euler(20, evaluate=False).evalf() == 370371188237525.0

    assert euler(n).rewrite(Sum) == euler(n)
    n = Symbol('n', integer=True, nonnegative=True)
    assert euler(2*n + 1).rewrite(Sum) == 0
    _j = Dummy('j')
    _k = Dummy('k')
    assert euler(2*n).rewrite(Sum).dummy_eq(
            I*Sum((-1)**_j*2**(-_k)*I**(-_k)*(-2*_j + _k)**(2*n + 1)*
                  binomial(_k, _j)/_k, (_j, 0, _k), (_k, 1, 2*n + 1)))


def test_euler_odd():
    n = Symbol('n', odd=True, positive=True)
    assert euler(n) == 0
    n = Symbol('n', odd=True)
    assert euler(n) != 0


def test_euler_polynomials():
    assert euler(0, x) == 1
    assert euler(1, x) == x - S.Half
    assert euler(2, x) == x**2 - x
    assert euler(3, x) == x**3 - (3*x**2)/2 + Rational(1, 4)
    m = Symbol('m')
    assert isinstance(euler(m, x), euler)
    from sympy.core.numbers import Float
    A = Float('-0.46237208575048694923364757452876131e8')  # from Maple
    B = euler(19, S.Pi).evalf(32)
    assert abs((A - B)/A) < 1e-31
    z = Float(0.1) + Float(0.2)*I
    expected = Float(-3126.54721663773 ) + Float(565.736261497056) * I
    assert abs(euler(13, z) - expected) < 1e-10


def test_euler_polynomial_rewrite():
    m = Symbol('m')
    A = euler(m, x).rewrite('Sum')
    assert A.subs({m:3, x:5}).doit() == euler(3, 5)


def test_catalan():
    n = Symbol('n', integer=True)
    m = Symbol('m', integer=True, positive=True)
    k = Symbol('k', integer=True, nonnegative=True)
    p = Symbol('p', nonnegative=True)

    catalans = [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796, 58786]
    for i, c in enumerate(catalans):
        assert catalan(i) == c
        assert catalan(n).rewrite(factorial).subs(n, i) == c
        assert catalan(n).rewrite(Product).subs(n, i).doit() == c

    assert unchanged(catalan, x)
    assert catalan(2*x).rewrite(binomial) == binomial(4*x, 2*x)/(2*x + 1)
    assert catalan(S.Half).rewrite(gamma) == 8/(3*pi)
    assert catalan(S.Half).rewrite(factorial).rewrite(gamma) ==\
        8 / (3 * pi)
    assert catalan(3*x).rewrite(gamma) == 4**(
        3*x)*gamma(3*x + S.Half)/(sqrt(pi)*gamma(3*x + 2))
    assert catalan(x).rewrite(hyper) == hyper((-x + 1, -x), (2,), 1)

    assert catalan(n).rewrite(factorial) == factorial(2*n) / (factorial(n + 1)
                                                              * factorial(n))
    assert isinstance(catalan(n).rewrite(Product), catalan)
    assert isinstance(catalan(m).rewrite(Product), Product)

    assert diff(catalan(x), x) == (polygamma(
        0, x + S.Half) - polygamma(0, x + 2) + log(4))*catalan(x)

    assert catalan(x).evalf() == catalan(x)
    c = catalan(S.Half).evalf()
    assert str(c) == '0.848826363156775'
    c = catalan(I).evalf(3)
    assert str((re(c), im(c))) == '(0.398, -0.0209)'

    # Assumptions
    assert catalan(p).is_positive is True
    assert catalan(k).is_integer is True
    assert catalan(m+3).is_composite is True


def test_genocchi():
    genocchis = [0, -1, -1, 0, 1, 0, -3, 0, 17]
    for n, g in enumerate(genocchis):
        assert genocchi(n) == g

    m = Symbol('m', integer=True)
    n = Symbol('n', integer=True, positive=True)
    assert unchanged(genocchi, m)
    assert genocchi(2*n + 1) == 0
    gn = 2 * (1 - 2**n) * bernoulli(n)
    assert genocchi(n).rewrite(bernoulli).factor() == gn.factor()
    gnx = 2 * (bernoulli(n, x) - 2**n * bernoulli(n, (x+1) / 2))
    assert genocchi(n, x).rewrite(bernoulli).factor() == gnx.factor()
    assert genocchi(2 * n).is_odd
    assert genocchi(2 * n).is_even is False
    assert genocchi(2 * n + 1).is_even
    assert genocchi(n).is_integer
    assert genocchi(4 * n).is_positive
    # these are the only 2 prime Genocchi numbers
    assert genocchi(6, evaluate=False).is_prime == S(-3).is_prime
    assert genocchi(8, evaluate=False).is_prime
    assert genocchi(4 * n + 2).is_negative
    assert genocchi(4 * n + 1).is_negative is False
    assert genocchi(4 * n - 2).is_negative

    g0 = genocchi(0, evaluate=False)
    assert g0.is_positive is False
    assert g0.is_negative is False
    assert g0.is_even is True
    assert g0.is_odd is False

    assert genocchi(0, x) == 0
    assert genocchi(1, x) == -1
    assert genocchi(2, x) == 1 - 2*x
    assert genocchi(3, x) == 3*x - 3*x**2
    assert genocchi(4, x) == -1 + 6*x**2 - 4*x**3
    y = Symbol("y")
    assert genocchi(5, (x+y)**100) == -5*(x+y)**400 + 10*(x+y)**300 - 5*(x+y)**100

    assert str(genocchi(5.0, 4.0).evalf(n=10)) == '-660.0000000'
    assert str(genocchi(Rational(5, 4)).evalf(n=10)) == '-1.104286457'
    assert str(genocchi(-2).evalf(n=10)) == '3.606170709'
    assert str(genocchi(1.3, 3.7).evalf(n=10)) == '-1.847375373'
    assert str(genocchi(I, 1.0).evalf(n=10)) == '-0.3161917278 - 1.45311955*I'

    n = Symbol('n')
    assert genocchi(n, x).rewrite(dirichlet_eta) == -2*n * dirichlet_eta(1-n, x)


def test_andre():
    nums = [1, 1, 1, 2, 5, 16, 61, 272, 1385, 7936, 50521]
    for n, a in enumerate(nums):
        assert andre(n) == a
    assert andre(S.Infinity) == S.Infinity
    assert andre(-1) == -log(2)
    assert andre(-2) == -2*S.Catalan
    assert andre(-3) == 3*zeta(3)/16
    assert andre(-5) == -15*zeta(5)/256
    # In fact andre(-2*n) is related to the Dirichlet *beta* function
    # at 2*n, but SymPy doesn't implement that (or general L-functions)
    assert unchanged(andre, -4)

    n = Symbol('n', integer=True, nonnegative=True)
    assert unchanged(andre, n)
    assert andre(n).is_integer is True
    assert andre(n).is_positive is True

    assert str(andre(10, evaluate=False).evalf(n=10)) == '50521.00000'
    assert str(andre(-1, evaluate=False).evalf(n=10)) == '-0.6931471806'
    assert str(andre(-2, evaluate=False).evalf(n=10)) == '-1.831931188'
    assert str(andre(-4, evaluate=False).evalf(n=10)) == '1.977889103'
    assert str(andre(I, evaluate=False).evalf(n=10)) == '2.378417833 + 0.6343322845*I'

    assert andre(x).rewrite(polylog) == \
            (-I)**(x+1) * polylog(-x, I) + I**(x+1) * polylog(-x, -I)
    assert andre(x).rewrite(zeta) == \
            2 * gamma(x+1) / (2*pi)**(x+1) * \
            (zeta(x+1, Rational(1,4)) - cos(pi*x) * zeta(x+1, Rational(3,4)))


@nocache_fail
def test_partition():
    partition_nums = [1, 1, 2, 3, 5, 7, 11, 15, 22]
    for n, p in enumerate(partition_nums):
        assert partition(n) == p

    x = Symbol('x')
    y = Symbol('y', real=True)
    m = Symbol('m', integer=True)
    n = Symbol('n', integer=True, negative=True)
    p = Symbol('p', integer=True, nonnegative=True)
    assert partition(m).is_integer
    assert not partition(m).is_negative
    assert partition(m).is_nonnegative
    assert partition(n).is_zero
    assert partition(p).is_positive
    assert partition(x).subs(x, 7) == 15
    assert partition(y).subs(y, 8) == 22
    raises(TypeError, lambda: partition(Rational(5, 4)))
    assert partition(9, evaluate=False) % 5 == 0
    assert partition(5*m + 4) % 5 == 0
    assert partition(47, evaluate=False) % 7 == 0
    assert partition(7*m + 5) % 7 == 0
    assert partition(50, evaluate=False) % 11 == 0
    assert partition(11*m + 6) % 11 == 0


def test_divisor_sigma():
    # error
    m = Symbol('m', integer=False)
    raises(TypeError, lambda: divisor_sigma(m))
    raises(TypeError, lambda: divisor_sigma(4.5))
    raises(TypeError, lambda: divisor_sigma(1, m))
    raises(TypeError, lambda: divisor_sigma(1, 4.5))
    m = Symbol('m', positive=False)
    raises(ValueError, lambda: divisor_sigma(m))
    raises(ValueError, lambda: divisor_sigma(0))
    m = Symbol('m', negative=True)
    raises(ValueError, lambda: divisor_sigma(1, m))
    raises(ValueError, lambda: divisor_sigma(1, -1))

    # special case
    p = Symbol('p', prime=True)
    k = Symbol('k', integer=True)
    assert divisor_sigma(p, 1) == p + 1
    assert divisor_sigma(p, k) == p**k + 1

    # property
    n = Symbol('n', integer=True, positive=True)
    assert divisor_sigma(n).is_integer is True
    assert divisor_sigma(n).is_positive is True

    # symbolic
    k = Symbol('k', integer=True, zero=False)
    assert divisor_sigma(4, k) == 2**(2*k) + 2**k + 1
    assert divisor_sigma(6, k) == (2**k + 1) * (3**k + 1)

    # Integer
    assert divisor_sigma(23450) == 50592
    assert divisor_sigma(23450, 0) == 24
    assert divisor_sigma(23450, 1) == 50592
    assert divisor_sigma(23450, 2) == 730747500
    assert divisor_sigma(23450, 3) == 14666785333344


def test_udivisor_sigma():
    # error
    m = Symbol('m', integer=False)
    raises(TypeError, lambda: udivisor_sigma(m))
    raises(TypeError, lambda: udivisor_sigma(4.5))
    raises(TypeError, lambda: udivisor_sigma(1, m))
    raises(TypeError, lambda: udivisor_sigma(1, 4.5))
    m = Symbol('m', positive=False)
    raises(ValueError, lambda: udivisor_sigma(m))
    raises(ValueError, lambda: udivisor_sigma(0))
    m = Symbol('m', negative=True)
    raises(ValueError, lambda: udivisor_sigma(1, m))
    raises(ValueError, lambda: udivisor_sigma(1, -1))

    # special case
    p = Symbol('p', prime=True)
    k = Symbol('k', integer=True)
    assert udivisor_sigma(p, 1) == p + 1
    assert udivisor_sigma(p, k) == p**k + 1

    # property
    n = Symbol('n', integer=True, positive=True)
    assert udivisor_sigma(n).is_integer is True
    assert udivisor_sigma(n).is_positive is True

    # Integer
    A034444 = [1, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 4, 2, 4, 4, 2, 2, 4, 2, 4,
               4, 4, 2, 4, 2, 4, 2, 4, 2, 8, 2, 2, 4, 4, 4, 4, 2, 4, 4, 4,
               2, 8, 2, 4, 4, 4, 2, 4, 2, 4, 4, 4, 2, 4, 4, 4, 4, 4, 2, 8]
    for n, val in enumerate(A034444, 1):
        assert udivisor_sigma(n, 0) == val
    A034448 = [1, 3, 4, 5, 6, 12, 8, 9, 10, 18, 12, 20, 14, 24, 24, 17, 18,
               30, 20, 30, 32, 36, 24, 36, 26, 42, 28, 40, 30, 72, 32, 33,
               48, 54, 48, 50, 38, 60, 56, 54, 42, 96, 44, 60, 60, 72, 48]
    for n, val in enumerate(A034448, 1):
        assert udivisor_sigma(n, 1) == val
    A034676 = [1, 5, 10, 17, 26, 50, 50, 65, 82, 130, 122, 170, 170, 250,
               260, 257, 290, 410, 362, 442, 500, 610, 530, 650, 626, 850,
               730, 850, 842, 1300, 962, 1025, 1220, 1450, 1300, 1394, 1370]
    for n, val in enumerate(A034676, 1):
        assert udivisor_sigma(n, 2) == val


def test_legendre_symbol():
    # error
    m = Symbol('m', integer=False)
    raises(TypeError, lambda: legendre_symbol(m, 3))
    raises(TypeError, lambda: legendre_symbol(4.5, 3))
    raises(TypeError, lambda: legendre_symbol(1, m))
    raises(TypeError, lambda: legendre_symbol(1, 4.5))
    m = Symbol('m', prime=False)
    raises(ValueError, lambda: legendre_symbol(1, m))
    raises(ValueError, lambda: legendre_symbol(1, 6))
    m = Symbol('m', odd=False)
    raises(ValueError, lambda: legendre_symbol(1, m))
    raises(ValueError, lambda: legendre_symbol(1, 2))

    # special case
    p = Symbol('p', prime=True)
    k = Symbol('k', integer=True)
    assert legendre_symbol(p*k, p) == 0
    assert legendre_symbol(1, p) == 1

    # property
    n = Symbol('n')
    m = Symbol('m')
    assert legendre_symbol(m, n).is_integer is True
    assert legendre_symbol(m, n).is_prime is False

    # Integer
    assert legendre_symbol(5, 11) == 1
    assert legendre_symbol(25, 41) == 1
    assert legendre_symbol(67, 101) == -1
    assert legendre_symbol(0, 13) == 0
    assert legendre_symbol(9, 3) == 0


def test_jacobi_symbol():
    # error
    m = Symbol('m', integer=False)
    raises(TypeError, lambda: jacobi_symbol(m, 3))
    raises(TypeError, lambda: jacobi_symbol(4.5, 3))
    raises(TypeError, lambda: jacobi_symbol(1, m))
    raises(TypeError, lambda: jacobi_symbol(1, 4.5))
    m = Symbol('m', positive=False)
    raises(ValueError, lambda: jacobi_symbol(1, m))
    raises(ValueError, lambda: jacobi_symbol(1, -6))
    m = Symbol('m', odd=False)
    raises(ValueError, lambda: jacobi_symbol(1, m))
    raises(ValueError, lambda: jacobi_symbol(1, 2))

    # special case
    p = Symbol('p', integer=True)
    k = Symbol('k', integer=True)
    assert jacobi_symbol(p*k, p) == 0
    assert jacobi_symbol(1, p) == 1
    assert jacobi_symbol(1, 1) == 1
    assert jacobi_symbol(0, 1) == 1

    # property
    n = Symbol('n')
    m = Symbol('m')
    assert jacobi_symbol(m, n).is_integer is True
    assert jacobi_symbol(m, n).is_prime is False

    # Integer
    assert jacobi_symbol(25, 41) == 1
    assert jacobi_symbol(-23, 83) == -1
    assert jacobi_symbol(3, 9) == 0
    assert jacobi_symbol(42, 97) == -1
    assert jacobi_symbol(3, 5) == -1
    assert jacobi_symbol(7, 9) == 1
    assert jacobi_symbol(0, 3) == 0
    assert jacobi_symbol(0, 1) == 1
    assert jacobi_symbol(2, 1) == 1
    assert jacobi_symbol(1, 3) == 1


def test_kronecker_symbol():
    # error
    m = Symbol('m', integer=False)
    raises(TypeError, lambda: kronecker_symbol(m, 3))
    raises(TypeError, lambda: kronecker_symbol(4.5, 3))
    raises(TypeError, lambda: kronecker_symbol(1, m))
    raises(TypeError, lambda: kronecker_symbol(1, 4.5))

    # special case
    p = Symbol('p', integer=True)
    assert kronecker_symbol(1, p) == 1
    assert kronecker_symbol(1, 1) == 1
    assert kronecker_symbol(0, 1) == 1

    # property
    n = Symbol('n')
    m = Symbol('m')
    assert kronecker_symbol(m, n).is_integer is True
    assert kronecker_symbol(m, n).is_prime is False

    # Integer
    for n in range(3, 10, 2):
        for a in range(-n, n):
            val = kronecker_symbol(a, n)
            assert val == jacobi_symbol(a, n)
            minus = kronecker_symbol(a, -n)
            if a < 0:
                assert -minus == val
            else:
                assert minus == val
            even = kronecker_symbol(a, 2 * n)
            if a % 2 == 0:
                assert even == 0
            elif a % 8 in [1, 7]:
                assert even == val
            else:
                assert -even == val
    assert kronecker_symbol(1, 0) == kronecker_symbol(-1, 0) == 1
    assert kronecker_symbol(0, 0) == 0


def test_mobius():
    # error
    m = Symbol('m', integer=False)
    raises(TypeError, lambda: mobius(m))
    raises(TypeError, lambda: mobius(4.5))
    m = Symbol('m', positive=False)
    raises(ValueError, lambda: mobius(m))
    raises(ValueError, lambda: mobius(-3))

    # special case
    p = Symbol('p', prime=True)
    assert mobius(p) == -1

    # property
    n = Symbol('n', integer=True, positive=True)
    assert mobius(n).is_integer is True
    assert mobius(n).is_prime is False

    # symbolic
    n = Symbol('n', integer=True, positive=True)
    k = Symbol('k', integer=True, positive=True)
    assert mobius(n**2) == 0
    assert mobius(4*n) == 0
    assert isinstance(mobius(n**k), mobius)
    assert mobius(n**(k+1)) == 0
    assert isinstance(mobius(3**k), mobius)
    assert mobius(3**(k+1)) == 0
    m = Symbol('m')
    assert isinstance(mobius(4*m), mobius)

    # Integer
    assert mobius(13*7) == 1
    assert mobius(1) == 1
    assert mobius(13*7*5) == -1
    assert mobius(13**2) == 0
    A008683 = [1, -1, -1, 0, -1, 1, -1, 0, 0, 1, -1, 0, -1, 1, 1, 0, -1, 0,
               -1, 0, 1, 1, -1, 0, 0, 1, 0, 0, -1, -1, -1, 0, 1, 1, 1, 0, -1,
               1, 1, 0, -1, -1, -1, 0, 0, 1, -1, 0, 0, 0, 1, 0, -1, 0, 1, 0]
    for n, val in enumerate(A008683, 1):
        assert mobius(n) == val


def test_primenu():
    # error
    m = Symbol('m', integer=False)
    raises(TypeError, lambda: primenu(m))
    raises(TypeError, lambda: primenu(4.5))
    m = Symbol('m', positive=False)
    raises(ValueError, lambda: primenu(m))
    raises(ValueError, lambda: primenu(0))

    # special case
    p = Symbol('p', prime=True)
    assert primenu(p) == 1

    # property
    n = Symbol('n', integer=True, positive=True)
    assert primenu(n).is_integer is True
    assert primenu(n).is_nonnegative is True

    # Integer
    assert primenu(7*13) == 2
    assert primenu(2*17*19) == 3
    assert primenu(2**3 * 17 * 19**2) == 3
    A001221 = [0, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2,
               1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1, 3, 1, 1, 2, 2, 2, 2]
    for n, val in enumerate(A001221, 1):
        assert primenu(n) == val


def test_primeomega():
    # error
    m = Symbol('m', integer=False)
    raises(TypeError, lambda: primeomega(m))
    raises(TypeError, lambda: primeomega(4.5))
    m = Symbol('m', positive=False)
    raises(ValueError, lambda: primeomega(m))
    raises(ValueError, lambda: primeomega(0))

    # special case
    p = Symbol('p', prime=True)
    assert primeomega(p) == 1

    # property
    n = Symbol('n', integer=True, positive=True)
    assert primeomega(n).is_integer is True
    assert primeomega(n).is_nonnegative is True

    # Integer
    assert primeomega(7*13) == 2
    assert primeomega(2*17*19) == 3
    assert primeomega(2**3 * 17 * 19**2) == 6
    A001222 = [0, 1, 1, 2, 1, 2, 1, 3, 2, 2, 1, 3, 1, 2, 2, 4, 1, 3,
               1, 3, 2, 2, 1, 4, 2, 2, 3, 3, 1, 3, 1, 5, 2, 2, 2, 4]
    for n, val in enumerate(A001222, 1):
        assert primeomega(n) == val


def test_totient():
    # error
    m = Symbol('m', integer=False)
    raises(TypeError, lambda: totient(m))
    raises(TypeError, lambda: totient(4.5))
    m = Symbol('m', positive=False)
    raises(ValueError, lambda: totient(m))
    raises(ValueError, lambda: totient(0))

    # special case
    p = Symbol('p', prime=True)
    assert totient(p) == p - 1

    # property
    n = Symbol('n', integer=True, positive=True)
    assert totient(n).is_integer is True
    assert totient(n).is_positive is True

    # Integer
    assert totient(7*13) == totient(factorint(7*13)) == (7-1)*(13-1)
    assert totient(2*17*19) == totient(factorint(2*17*19)) == (17-1)*(19-1)
    assert totient(2**3 * 17 * 19**2) == totient({2: 3, 17: 1, 19: 2}) == 2**2 * (17-1) * 19*(19-1)
    A000010 = [1, 1, 2, 2, 4, 2, 6, 4, 6, 4, 10, 4, 12, 6, 8, 8, 16,
               6, 18, 8, 12, 10, 22, 8, 20, 12, 18, 12, 28, 8, 30, 16,
               20, 16, 24, 12, 36, 18, 24, 16, 40, 12, 42, 20, 24, 22]
    for n, val in enumerate(A000010, 1):
        assert totient(n) == val


def test_reduced_totient():
    # error
    m = Symbol('m', integer=False)
    raises(TypeError, lambda: reduced_totient(m))
    raises(TypeError, lambda: reduced_totient(4.5))
    m = Symbol('m', positive=False)
    raises(ValueError, lambda: reduced_totient(m))
    raises(ValueError, lambda: reduced_totient(0))

    # special case
    p = Symbol('p', prime=True)
    assert reduced_totient(p) == p - 1

    # property
    n = Symbol('n', integer=True, positive=True)
    assert reduced_totient(n).is_integer is True
    assert reduced_totient(n).is_positive is True

    # Integer
    assert reduced_totient(7*13) == reduced_totient(factorint(7*13)) == 12
    assert reduced_totient(2*17*19) == reduced_totient(factorint(2*17*19)) == 144
    assert reduced_totient(2**2 * 11) == reduced_totient({2: 2, 11: 1}) == 10
    assert reduced_totient(2**3 * 17 * 19**2) == reduced_totient({2: 3, 17: 1, 19: 2}) == 2736
    A002322 = [1, 1, 2, 2, 4, 2, 6, 2, 6, 4, 10, 2, 12, 6, 4, 4, 16, 6,
               18, 4, 6, 10, 22, 2, 20, 12, 18, 6, 28, 4, 30, 8, 10, 16,
               12, 6, 36, 18, 12, 4, 40, 6, 42, 10, 12, 22, 46, 4, 42]
    for n, val in enumerate(A002322, 1):
        assert reduced_totient(n) == val


def test_primepi():
    # error
    z = Symbol('z', real=False)
    raises(TypeError, lambda: primepi(z))
    raises(TypeError, lambda: primepi(I))

    # property
    n = Symbol('n', integer=True, positive=True)
    assert primepi(n).is_integer is True
    assert primepi(n).is_nonnegative is True

    # infinity
    assert primepi(oo) == oo
    assert primepi(-oo) == 0

    # symbol
    x = Symbol('x')
    assert isinstance(primepi(x), primepi)

    # Integer
    assert primepi(0) == 0
    A000720 = [0, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 8,
               8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 11, 11, 11, 11, 11, 11,
               12, 12, 12, 12, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15]
    for n, val in enumerate(A000720, 1):
        assert primepi(n) == primepi(n + 0.5) == val


def test__nT():
    assert [_nT(i, j) for i in range(5) for j in range(i + 2)] == [
            1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 2, 1, 1, 0]
    check = [_nT(10, i) for i in range(11)]
    assert check == [0, 1, 5, 8, 9, 7, 5, 3, 2, 1, 1]
    assert all(type(i) is int for i in check)
    assert _nT(10, 5) == 7
    assert _nT(100, 98) == 2
    assert _nT(100, 100) == 1
    assert _nT(10, 3) == 8


def test_nC_nP_nT():
    from sympy.utilities.iterables import (
        multiset_permutations, multiset_combinations, multiset_partitions,
        partitions, subsets, permutations)
    from sympy.functions.combinatorial.numbers import (
        nP, nC, nT, stirling, _stirling1, _stirling2, _multiset_histogram, _AOP_product)

    from sympy.combinatorics.permutations import Permutation
    from sympy.core.random import choice

    c = string.ascii_lowercase
    for i in range(100):
        s = ''.join(choice(c) for i in range(7))
        u = len(s) == len(set(s))
        try:
            tot = 0
            for i in range(8):
                check = nP(s, i)
                tot += check
                assert len(list(multiset_permutations(s, i))) == check
                if u:
                    assert nP(len(s), i) == check
            assert nP(s) == tot
        except AssertionError:
            print(s, i, 'failed perm test')
            raise ValueError()

    for i in range(100):
        s = ''.join(choice(c) for i in range(7))
        u = len(s) == len(set(s))
        try:
            tot = 0
            for i in range(8):
                check = nC(s, i)
                tot += check
                assert len(list(multiset_combinations(s, i))) == check
                if u:
                    assert nC(len(s), i) == check
            assert nC(s) == tot
            if u:
                assert nC(len(s)) == tot
        except AssertionError:
            print(s, i, 'failed combo test')
            raise ValueError()

    for i in range(1, 10):
        tot = 0
        for j in range(1, i + 2):
            check = nT(i, j)
            assert check.is_Integer
            tot += check
            assert sum(1 for p in partitions(i, j, size=True) if p[0] == j) == check
        assert nT(i) == tot

    for i in range(1, 10):
        tot = 0
        for j in range(1, i + 2):
            check = nT(range(i), j)
            tot += check
            assert len(list(multiset_partitions(list(range(i)), j))) == check
        assert nT(range(i)) == tot

    for i in range(100):
        s = ''.join(choice(c) for i in range(7))
        u = len(s) == len(set(s))
        try:
            tot = 0
            for i in range(1, 8):
                check = nT(s, i)
                tot += check
                assert len(list(multiset_partitions(s, i))) == check
                if u:
                    assert nT(range(len(s)), i) == check
            if u:
                assert nT(range(len(s))) == tot
            assert nT(s) == tot
        except AssertionError:
            print(s, i, 'failed partition test')
            raise ValueError()

    # tests for Stirling numbers of the first kind that are not tested in the
    # above
    assert [stirling(9, i, kind=1) for i in range(11)] == [
        0, 40320, 109584, 118124, 67284, 22449, 4536, 546, 36, 1, 0]
    perms = list(permutations(range(4)))
    assert [sum(1 for p in perms if Permutation(p).cycles == i)
            for i in range(5)] == [0, 6, 11, 6, 1] == [
            stirling(4, i, kind=1) for i in range(5)]
    # http://oeis.org/A008275
    assert [stirling(n, k, signed=1)
        for n in range(10) for k in range(1, n + 1)] == [
            1, -1,
            1, 2, -3,
            1, -6, 11, -6,
            1, 24, -50, 35, -10,
            1, -120, 274, -225, 85, -15,
            1, 720, -1764, 1624, -735, 175, -21,
            1, -5040, 13068, -13132, 6769, -1960, 322, -28,
            1, 40320, -109584, 118124, -67284, 22449, -4536, 546, -36, 1]
    # https://en.wikipedia.org/wiki/Stirling_numbers_of_the_first_kind
    assert  [stirling(n, k, kind=1)
        for n in range(10) for k in range(n+1)] == [
            1,
            0, 1,
            0, 1, 1,
            0, 2, 3, 1,
            0, 6, 11, 6, 1,
            0, 24, 50, 35, 10, 1,
            0, 120, 274, 225, 85, 15, 1,
            0, 720, 1764, 1624, 735, 175, 21, 1,
            0, 5040, 13068, 13132, 6769, 1960, 322, 28, 1,
            0, 40320, 109584, 118124, 67284, 22449, 4536, 546, 36, 1]
    # https://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind
    assert [stirling(n, k, kind=2)
        for n in range(10) for k in range(n+1)] == [
            1,
            0, 1,
            0, 1, 1,
            0, 1, 3, 1,
            0, 1, 7, 6, 1,
            0, 1, 15, 25, 10, 1,
            0, 1, 31, 90, 65, 15, 1,
            0, 1, 63, 301, 350, 140, 21, 1,
            0, 1, 127, 966, 1701, 1050, 266, 28, 1,
            0, 1, 255, 3025, 7770, 6951, 2646, 462, 36, 1]
    assert stirling(3, 4, kind=1) == stirling(3, 4, kind=1) == 0
    raises(ValueError, lambda: stirling(-2, 2))

    # Assertion that the return type is SymPy Integer.
    assert isinstance(_stirling1(6, 3), Integer)
    assert isinstance(_stirling2(6, 3), Integer)

    def delta(p):
        if len(p) == 1:
            return oo
        return min(abs(i[0] - i[1]) for i in subsets(p, 2))
    parts = multiset_partitions(range(5), 3)
    d = 2
    assert (sum(1 for p in parts if all(delta(i) >= d for i in p)) ==
            stirling(5, 3, d=d) == 7)

    # other coverage tests
    assert nC('abb', 2) == nC('aab', 2) == 2
    assert nP(3, 3, replacement=True) == nP('aabc', 3, replacement=True) == 27
    assert nP(3, 4) == 0
    assert nP('aabc', 5) == 0
    assert nC(4, 2, replacement=True) == nC('abcdd', 2, replacement=True) == \
        len(list(multiset_combinations('aabbccdd', 2))) == 10
    assert nC('abcdd') == sum(nC('abcdd', i) for i in range(6)) == 24
    assert nC(list('abcdd'), 4) == 4
    assert nT('aaaa') == nT(4) == len(list(partitions(4))) == 5
    assert nT('aaab') == len(list(multiset_partitions('aaab'))) == 7
    assert nC('aabb'*3, 3) == 4  # aaa, bbb, abb, baa
    assert dict(_AOP_product((4,1,1,1))) == {
        0: 1, 1: 4, 2: 7, 3: 8, 4: 8, 5: 7, 6: 4, 7: 1}
    # the following was the first t that showed a problem in a previous form of
    # the function, so it's not as random as it may appear
    t = (3, 9, 4, 6, 6, 5, 5, 2, 10, 4)
    assert sum(_AOP_product(t)[i] for i in range(55)) == 58212000
    raises(ValueError, lambda: _multiset_histogram({1:'a'}))


def test_PR_14617():
    from sympy.functions.combinatorial.numbers import nT
    for n in (0, []):
        for k in (-1, 0, 1):
            if k == 0:
                assert nT(n, k) == 1
            else:
                assert nT(n, k) == 0


def test_issue_8496():
    n = Symbol("n")
    k = Symbol("k")

    raises(TypeError, lambda: catalan(n, k))


def test_issue_8601():
    n = Symbol('n', integer=True, negative=True)

    assert catalan(n - 1) is S.Zero
    assert catalan(Rational(-1, 2)) is S.ComplexInfinity
    assert catalan(-S.One) == Rational(-1, 2)
    c1 = catalan(-5.6).evalf()
    assert str(c1) == '6.93334070531408e-5'
    c2 = catalan(-35.4).evalf()
    assert str(c2) == '-4.14189164517449e-24'


def test_motzkin():
    assert motzkin.is_motzkin(4) == True
    assert motzkin.is_motzkin(9) == True
    assert motzkin.is_motzkin(10) == False
    assert motzkin.find_motzkin_numbers_in_range(10,200) == [21, 51, 127]
    assert motzkin.find_motzkin_numbers_in_range(10,400) == [21, 51, 127, 323]
    assert motzkin.find_motzkin_numbers_in_range(10,1600) == [21, 51, 127, 323, 835]
    assert motzkin.find_first_n_motzkins(5) == [1, 1, 2, 4, 9]
    assert motzkin.find_first_n_motzkins(7) == [1, 1, 2, 4, 9, 21, 51]
    assert motzkin.find_first_n_motzkins(10) == [1, 1, 2, 4, 9, 21, 51, 127, 323, 835]
    raises(ValueError, lambda: motzkin.eval(77.58))
    raises(ValueError, lambda: motzkin.eval(-8))
    raises(ValueError, lambda: motzkin.find_motzkin_numbers_in_range(-2,7))
    raises(ValueError, lambda: motzkin.find_motzkin_numbers_in_range(13,7))
    raises(ValueError, lambda: motzkin.find_first_n_motzkins(112.8))


def test_nD_derangements():
    from sympy.utilities.iterables import (partitions, multiset,
        multiset_derangements, multiset_permutations)
    from sympy.functions.combinatorial.numbers import nD

    got = []
    for i in partitions(8, k=4):
        s = []
        it = 0
        for k, v in i.items():
            for i in range(v):
                s.extend([it]*k)
                it += 1
        ms = multiset(s)
        c1 = sum(1 for i in multiset_permutations(s) if
            all(i != j for i, j in zip(i, s)))
        assert c1 == nD(ms) == nD(ms, 0) == nD(ms, 1)
        v = [tuple(i) for i in multiset_derangements(s)]
        c2 = len(v)
        assert c2 == len(set(v))
        assert c1 == c2
        got.append(c1)
    assert got == [1, 4, 6, 12, 24, 24, 61, 126, 315, 780, 297, 772,
        2033, 5430, 14833]

    assert nD('1112233456', brute=True) == nD('1112233456') == 16356
    assert nD('') == nD([]) == nD({}) == 0
    assert nD({1: 0}) == 0
    raises(ValueError, lambda: nD({1: -1}))
    assert nD('112') == 0
    assert nD(i='112') == 0
    assert [nD(n=i) for i in range(6)] == [0, 0, 1, 2, 9, 44]
    assert nD((i for i in range(4))) == nD('0123') == 9
    assert nD(m=(i for i in range(4))) == 3
    assert nD(m={0: 1, 1: 1, 2: 1, 3: 1}) == 3
    assert nD(m=[0, 1, 2, 3]) == 3
    raises(TypeError, lambda: nD(m=0))
    raises(TypeError, lambda: nD(-1))
    assert nD({-1: 1, -2: 1}) == 1
    assert nD(m={0: 3}) == 0
    raises(ValueError, lambda: nD(i='123', n=3))
    raises(ValueError, lambda: nD(i='123', m=(1,2)))
    raises(ValueError, lambda: nD(n=0, m=(1,2)))
    raises(ValueError, lambda: nD({1: -1}))
    raises(ValueError, lambda: nD(m={-1: 1, 2: 1}))
    raises(ValueError, lambda: nD(m={1: -1, 2: 1}))
    raises(ValueError, lambda: nD(m=[-1, 2]))
    raises(TypeError, lambda: nD({1: x}))
    raises(TypeError, lambda: nD(m={1: x}))
    raises(TypeError, lambda: nD(m={x: 1}))


def test_deprecated_ntheory_symbolic_functions():
    from sympy.testing.pytest import warns_deprecated_sympy

    with warns_deprecated_sympy():
        assert not carmichael.is_carmichael(3)
    with warns_deprecated_sympy():
        assert carmichael.find_carmichael_numbers_in_range(10, 20) == []
    with warns_deprecated_sympy():
        assert carmichael.find_first_n_carmichaels(1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\itsdangerous\__init__.py ===
from __future__ import annotations

import typing as t

from .encoding import base64_decode as base64_decode
from .encoding import base64_encode as base64_encode
from .encoding import want_bytes as want_bytes
from .exc import BadData as BadData
from .exc import BadHeader as BadHeader
from .exc import BadPayload as BadPayload
from .exc import BadSignature as BadSignature
from .exc import BadTimeSignature as BadTimeSignature
from .exc import SignatureExpired as SignatureExpired
from .serializer import Serializer as Serializer
from .signer import HMACAlgorithm as HMACAlgorithm
from .signer import NoneAlgorithm as NoneAlgorithm
from .signer import Signer as Signer
from .timed import TimedSerializer as TimedSerializer
from .timed import TimestampSigner as TimestampSigner
from .url_safe import URLSafeSerializer as URLSafeSerializer
from .url_safe import URLSafeTimedSerializer as URLSafeTimedSerializer


def __getattr__(name: str) -> t.Any:
    if name == "__version__":
        import importlib.metadata
        import warnings

        warnings.warn(
            "The '__version__' attribute is deprecated and will be removed in"
            " ItsDangerous 2.3. Use feature detection or"
            " 'importlib.metadata.version(\"itsdangerous\")' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return importlib.metadata.version("itsdangerous")

    raise AttributeError(name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\jinja2\__init__.py ===
"""Jinja is a template engine written in pure Python. It provides a
non-XML syntax that supports inline expressions and an optional
sandboxed environment.
"""

from .bccache import BytecodeCache as BytecodeCache
from .bccache import FileSystemBytecodeCache as FileSystemBytecodeCache
from .bccache import MemcachedBytecodeCache as MemcachedBytecodeCache
from .environment import Environment as Environment
from .environment import Template as Template
from .exceptions import TemplateAssertionError as TemplateAssertionError
from .exceptions import TemplateError as TemplateError
from .exceptions import TemplateNotFound as TemplateNotFound
from .exceptions import TemplateRuntimeError as TemplateRuntimeError
from .exceptions import TemplatesNotFound as TemplatesNotFound
from .exceptions import TemplateSyntaxError as TemplateSyntaxError
from .exceptions import UndefinedError as UndefinedError
from .loaders import BaseLoader as BaseLoader
from .loaders import ChoiceLoader as ChoiceLoader
from .loaders import DictLoader as DictLoader
from .loaders import FileSystemLoader as FileSystemLoader
from .loaders import FunctionLoader as FunctionLoader
from .loaders import ModuleLoader as ModuleLoader
from .loaders import PackageLoader as PackageLoader
from .loaders import PrefixLoader as PrefixLoader
from .runtime import ChainableUndefined as ChainableUndefined
from .runtime import DebugUndefined as DebugUndefined
from .runtime import make_logging_undefined as make_logging_undefined
from .runtime import StrictUndefined as StrictUndefined
from .runtime import Undefined as Undefined
from .utils import clear_caches as clear_caches
from .utils import is_undefined as is_undefined
from .utils import pass_context as pass_context
from .utils import pass_environment as pass_environment
from .utils import pass_eval_context as pass_eval_context
from .utils import select_autoescape as select_autoescape

__version__ = "3.1.6"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\operations\build\metadata.py ===
"""Metadata generation logic for source distributions."""

import os

from pip._vendor.pyproject_hooks import BuildBackendHookCaller

from pip._internal.build_env import BuildEnvironment
from pip._internal.exceptions import (
    InstallationSubprocessError,
    MetadataGenerationFailed,
)
from pip._internal.utils.subprocess import runner_with_spinner_message
from pip._internal.utils.temp_dir import TempDirectory


def generate_metadata(
    build_env: BuildEnvironment, backend: BuildBackendHookCaller, details: str
) -> str:
    """Generate metadata using mechanisms described in PEP 517.

    Returns the generated metadata directory.
    """
    metadata_tmpdir = TempDirectory(kind="modern-metadata", globally_managed=True)

    metadata_dir = metadata_tmpdir.path

    with build_env:
        # Note that BuildBackendHookCaller implements a fallback for
        # prepare_metadata_for_build_wheel, so we don't have to
        # consider the possibility that this hook doesn't exist.
        runner = runner_with_spinner_message("Preparing metadata (pyproject.toml)")
        with backend.subprocess_runner(runner):
            try:
                distinfo_dir = backend.prepare_metadata_for_build_wheel(metadata_dir)
            except InstallationSubprocessError as error:
                raise MetadataGenerationFailed(package_details=details) from error

    return os.path.join(metadata_dir, distinfo_dir)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\utils\_log.py ===
"""Customize logging

Defines custom logger class for the `logger.verbose(...)` method.

init_logging() must be called before any other modules that call logging.getLogger.
"""

import logging
from typing import Any, cast

# custom log level for `--verbose` output
# between DEBUG and INFO
VERBOSE = 15


class VerboseLogger(logging.Logger):
    """Custom Logger, defining a verbose log-level

    VERBOSE is between INFO and DEBUG.
    """

    def verbose(self, msg: str, *args: Any, **kwargs: Any) -> None:
        return self.log(VERBOSE, msg, *args, **kwargs)


def getLogger(name: str) -> VerboseLogger:
    """logging.getLogger, but ensures our VerboseLogger class is returned"""
    return cast(VerboseLogger, logging.getLogger(name))


def init_logging() -> None:
    """Register our VerboseLogger and VERBOSE log level.

    Should be called before any calls to getLogger(),
    i.e. in pip._internal.__init__
    """
    logging.setLoggerClass(VerboseLogger)
    logging.addLevelName(VERBOSE, "VERBOSE")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\color_triplet.py ===
from typing import NamedTuple, Tuple


class ColorTriplet(NamedTuple):
    """The red, green, and blue components of a color."""

    red: int
    """Red component in 0 to 255 range."""
    green: int
    """Green component in 0 to 255 range."""
    blue: int
    """Blue component in 0 to 255 range."""

    @property
    def hex(self) -> str:
        """get the color triplet in CSS style."""
        red, green, blue = self
        return f"#{red:02x}{green:02x}{blue:02x}"

    @property
    def rgb(self) -> str:
        """The color in RGB format.

        Returns:
            str: An rgb color, e.g. ``"rgb(100,23,255)"``.
        """
        red, green, blue = self
        return f"rgb({red},{green},{blue})"

    @property
    def normalized(self) -> Tuple[float, float, float]:
        """Convert components into floats between 0 and 1.

        Returns:
            Tuple[float, float, float]: A tuple of three normalized colour components.
        """
        red, green, blue = self
        return red / 255.0, green / 255.0, blue / 255.0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\diagnose.py ===
import os
import platform

from pip._vendor.rich import inspect
from pip._vendor.rich.console import Console, get_windows_console_features
from pip._vendor.rich.panel import Panel
from pip._vendor.rich.pretty import Pretty


def report() -> None:  # pragma: no cover
    """Print a report to the terminal with debugging information"""
    console = Console()
    inspect(console)
    features = get_windows_console_features()
    inspect(features)

    env_names = (
        "CLICOLOR",
        "COLORTERM",
        "COLUMNS",
        "JPY_PARENT_PID",
        "JUPYTER_COLUMNS",
        "JUPYTER_LINES",
        "LINES",
        "NO_COLOR",
        "TERM_PROGRAM",
        "TERM",
        "TTY_COMPATIBLE",
        "VSCODE_VERBOSE_LOGGING",
    )
    env = {name: os.getenv(name) for name in env_names}
    console.print(Panel.fit((Pretty(env)), title="[b]Environment Variables"))

    console.print(f'platform="{platform.system()}"')


if __name__ == "__main__":  # pragma: no cover
    report()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\configuration\startup.py ===
# -*- coding: UTF-8 -*-
# Example snippet to use in a PYTHONSTARTUP file


try:
    import atexit

    # pyreadline3.rlmain.config_path=r"c:\xxx\pyreadlineconfig.ini"
    import readline

    import pyreadline3.rlmain
    import pyreadline3.unicode_helper

    #
    #
    # Normally the codepage for pyreadline3 is set to be sys.stdout.encoding
    # if you need to change this uncomment the following line
    # pyreadline3.unicode_helper.pyreadline_codepage="utf8"
except ImportError:
    print("Module readline not available.")
else:
    # import tab completion functionality
    import rlcompleter

    # Override completer from rlcompleter to disable automatic ( on callable
    completer_obj = rlcompleter.Completer()

    def nop(val, word):
        return word

    completer_obj._callable_postfix = nop
    readline.set_completer(completer_obj.complete)

    # activate tab completion
    readline.parse_and_bind("tab: complete")
    readline.read_history_file()
    atexit.register(readline.write_history_file)
    del readline, rlcompleter, atexit

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\bidi\common.py ===
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

from typing import Dict


def command_builder(method: str, params: Dict = None) -> Dict:
    """Build a command iterator to send to the BiDi protocol.

    Parameters:
    -----------
        method: The method to execute.
        params: The parameters to pass to the method. Default is None.

    Returns:
    --------
        The response from the command execution.
    """
    if params is None:
        params = {}

    command = {"method": method, "params": params}
    cmd = yield command
    return cmd

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\meijerint_doc.py ===
""" This module cooks up a docstring when imported. Its only purpose is to
    be displayed in the sphinx documentation. """

from __future__ import annotations
from typing import Any

from sympy.integrals.meijerint import _create_lookup_table
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol
from sympy.printing.latex import latex

t: dict[tuple[type[Basic], ...], list[Any]] = {}
_create_lookup_table(t)


doc = ""
for about, category in t.items():
    if about == ():
        doc += 'Elementary functions:\n\n'
    else:
        doc += 'Functions involving ' + ', '.join('`%s`' % latex(
            list(category[0][0].atoms(func))[0]) for func in about) + ':\n\n'
    for formula, gs, cond, hint in category:
        if not isinstance(gs, list):
            g: Expr = Symbol('\\text{generated}')
        else:
            g = Add(*[fac*f for (fac, f) in gs])
        obj = Eq(formula, g)
        if cond is True:
            cond = ""
        else:
            cond = ',\\text{ if } %s' % latex(cond)
        doc += ".. math::\n  %s%s\n\n" % (latex(obj), cond)

__doc__ = doc

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\autolev\_parse_autolev_antlr.py ===
from importlib.metadata import version
from sympy.external import import_module


autolevparser = import_module('sympy.parsing.autolev._antlr.autolevparser',
                              import_kwargs={'fromlist': ['AutolevParser']})
autolevlexer = import_module('sympy.parsing.autolev._antlr.autolevlexer',
                             import_kwargs={'fromlist': ['AutolevLexer']})
autolevlistener = import_module('sympy.parsing.autolev._antlr.autolevlistener',
                                import_kwargs={'fromlist': ['AutolevListener']})

AutolevParser = getattr(autolevparser, 'AutolevParser', None)
AutolevLexer = getattr(autolevlexer, 'AutolevLexer', None)
AutolevListener = getattr(autolevlistener, 'AutolevListener', None)


def parse_autolev(autolev_code, include_numeric):
    antlr4 = import_module('antlr4')
    if not antlr4 or not version('antlr4-python3-runtime').startswith('4.11'):
        raise ImportError("Autolev parsing requires the antlr4 Python package,"
                          " provided by pip (antlr4-python3-runtime)"
                          " conda (antlr-python-runtime), version 4.11")
    try:
        l = autolev_code.readlines()
        input_stream = antlr4.InputStream("".join(l))
    except Exception:
        input_stream = antlr4.InputStream(autolev_code)

    if AutolevListener:
        from ._listener_autolev_antlr import MyListener
        lexer = AutolevLexer(input_stream)
        token_stream = antlr4.CommonTokenStream(lexer)
        parser = AutolevParser(token_stream)
        tree = parser.prog()
        my_listener = MyListener(include_numeric)
        walker = antlr4.ParseTreeWalker()
        walker.walk(my_listener, tree)
        return "".join(my_listener.output_code)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\optics\__init__.py ===
__all__ = [
    'TWave',

    'RayTransferMatrix', 'FreeSpace', 'FlatRefraction', 'CurvedRefraction',
    'FlatMirror', 'CurvedMirror', 'ThinLens', 'GeometricRay', 'BeamParameter',
    'waist2rayleigh', 'rayleigh2waist', 'geometric_conj_ab',
    'geometric_conj_af', 'geometric_conj_bf', 'gaussian_conj',
    'conjugate_gauss_beams',

    'Medium',

    'refraction_angle', 'deviation', 'fresnel_coefficients', 'brewster_angle',
    'critical_angle', 'lens_makers_formula', 'mirror_formula', 'lens_formula',
    'hyperfocal_distance', 'transverse_magnification',

    'jones_vector', 'stokes_vector', 'jones_2_stokes', 'linear_polarizer',
    'phase_retarder', 'half_wave_retarder', 'quarter_wave_retarder',
    'transmissive_filter', 'reflective_filter', 'mueller_matrix',
    'polarizing_beam_splitter',
]
from .waves import TWave

from .gaussopt import (RayTransferMatrix, FreeSpace, FlatRefraction,
        CurvedRefraction, FlatMirror, CurvedMirror, ThinLens, GeometricRay,
        BeamParameter, waist2rayleigh, rayleigh2waist, geometric_conj_ab,
        geometric_conj_af, geometric_conj_bf, gaussian_conj,
        conjugate_gauss_beams)

from .medium import Medium

from .utils import (refraction_angle, deviation, fresnel_coefficients,
        brewster_angle, critical_angle, lens_makers_formula, mirror_formula,
        lens_formula, hyperfocal_distance, transverse_magnification)

from .polarization import (jones_vector, stokes_vector, jones_2_stokes,
        linear_polarizer, phase_retarder, half_wave_retarder,
        quarter_wave_retarder, transmissive_filter, reflective_filter,
        mueller_matrix, polarizing_beam_splitter)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\domainelement.py ===
"""Trait for implementing domain elements. """


from sympy.utilities import public

@public
class DomainElement:
    """
    Represents an element of a domain.

    Mix in this trait into a class whose instances should be recognized as
    elements of a domain. Method ``parent()`` gives that domain.
    """

    __slots__ = ()

    def parent(self):
        """Get the domain associated with ``self``

        Examples
        ========

        >>> from sympy import ZZ, symbols
        >>> x, y = symbols('x, y')
        >>> K = ZZ[x,y]
        >>> p = K(x)**2 + K(y)**2
        >>> p
        x**2 + y**2
        >>> p.parent()
        ZZ[x,y]

        Notes
        =====

        This is used by :py:meth:`~.Domain.convert` to identify the domain
        associated with a domain element.
        """
        raise NotImplementedError("abstract method")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\webdriver_manager\core\http.py ===
import requests
from requests import Response, exceptions

from webdriver_manager.core.config import ssl_verify


class HttpClient:
    def get(self, url, params=None, **kwargs) -> Response:
        raise NotImplementedError

    @staticmethod
    def validate_response(resp: requests.Response):
        status_code = resp.status_code
        if status_code == 404:
            raise ValueError(f"There is no such driver by url {resp.url}")
        elif status_code == 401:
            raise ValueError(f"API Rate limit exceeded. You have to add GH_TOKEN!!!")
        elif resp.status_code != 200:
            raise ValueError(
                f"response body:\n{resp.text}\n"
                f"request url:\n{resp.request.url}\n"
                f"response headers:\n{dict(resp.headers)}\n"
            )


class WDMHttpClient(HttpClient):
    def __init__(self):
        self._ssl_verify = ssl_verify()

    def get(self, url, **kwargs) -> Response:
        try:
            resp = requests.get(
                url=url, verify=self._ssl_verify, stream=True, **kwargs)
        except exceptions.ConnectionError:
            raise exceptions.ConnectionError(f"Could not reach host. Are you offline?")
        self.validate_response(resp)
        return resp


# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\tests\test_perm_groups.py ===
from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
    DihedralGroup, AlternatingGroup, AbelianGroup, RubikGroup
from sympy.combinatorics.perm_groups import (PermutationGroup,
    _orbit_transversal, Coset, SymmetricPermutationGroup)
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
    _verify_normal_closure
from sympy.testing.pytest import skip, XFAIL, slow

rmul = Permutation.rmul


def test_has():
    a = Permutation([1, 0])
    G = PermutationGroup([a])
    assert G.is_abelian
    a = Permutation([2, 0, 1])
    b = Permutation([2, 1, 0])
    G = PermutationGroup([a, b])
    assert not G.is_abelian

    G = PermutationGroup([a])
    assert G.has(a)
    assert not G.has(b)

    a = Permutation([2, 0, 1, 3, 4, 5])
    b = Permutation([0, 2, 1, 3, 4])
    assert PermutationGroup(a, b).degree == \
        PermutationGroup(a, b).degree == 6

    g = PermutationGroup(Permutation(0, 2, 1))
    assert Tuple(1, g).has(g)


def test_generate():
    a = Permutation([1, 0])
    g = list(PermutationGroup([a]).generate())
    assert g == [Permutation([0, 1]), Permutation([1, 0])]
    assert len(list(PermutationGroup(Permutation((0, 1))).generate())) == 1
    g = PermutationGroup([a]).generate(method='dimino')
    assert list(g) == [Permutation([0, 1]), Permutation([1, 0])]
    a = Permutation([2, 0, 1])
    b = Permutation([2, 1, 0])
    G = PermutationGroup([a, b])
    g = G.generate()
    v1 = [p.array_form for p in list(g)]
    v1.sort()
    assert v1 == [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0,
        1], [2, 1, 0]]
    v2 = list(G.generate(method='dimino', af=True))
    assert v1 == sorted(v2)
    a = Permutation([2, 0, 1, 3, 4, 5])
    b = Permutation([2, 1, 3, 4, 5, 0])
    g = PermutationGroup([a, b]).generate(af=True)
    assert len(list(g)) == 360


def test_order():
    a = Permutation([2, 0, 1, 3, 4, 5, 6, 7, 8, 9])
    b = Permutation([2, 1, 3, 4, 5, 6, 7, 8, 9, 0])
    g = PermutationGroup([a, b])
    assert g.order() == 1814400
    assert PermutationGroup().order() == 1


def test_equality():
    p_1 = Permutation(0, 1, 3)
    p_2 = Permutation(0, 2, 3)
    p_3 = Permutation(0, 1, 2)
    p_4 = Permutation(0, 1, 3)
    g_1 = PermutationGroup(p_1, p_2)
    g_2 = PermutationGroup(p_3, p_4)
    g_3 = PermutationGroup(p_2, p_1)
    g_4 = PermutationGroup(p_1, p_2)

    assert g_1 != g_2
    assert g_1.generators != g_2.generators
    assert g_1.equals(g_2)
    assert g_1 != g_3
    assert g_1.equals(g_3)
    assert g_1 == g_4


def test_stabilizer():
    S = SymmetricGroup(2)
    H = S.stabilizer(0)
    assert H.generators == [Permutation(1)]
    a = Permutation([2, 0, 1, 3, 4, 5])
    b = Permutation([2, 1, 3, 4, 5, 0])
    G = PermutationGroup([a, b])
    G0 = G.stabilizer(0)
    assert G0.order() == 60

    gens_cube = [[1, 3, 5, 7, 0, 2, 4, 6], [1, 3, 0, 2, 5, 7, 4, 6]]
    gens = [Permutation(p) for p in gens_cube]
    G = PermutationGroup(gens)
    G2 = G.stabilizer(2)
    assert G2.order() == 6
    G2_1 = G2.stabilizer(1)
    v = list(G2_1.generate(af=True))
    assert v == [[0, 1, 2, 3, 4, 5, 6, 7], [3, 1, 2, 0, 7, 5, 6, 4]]

    gens = (
        (1, 2, 0, 4, 5, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
        (0, 1, 2, 3, 4, 5, 19, 6, 8, 9, 10, 11, 12, 13, 14,
         15, 16, 7, 17, 18),
        (0, 1, 2, 3, 4, 5, 6, 7, 9, 18, 16, 11, 12, 13, 14, 15, 8, 17, 10, 19))
    gens = [Permutation(p) for p in gens]
    G = PermutationGroup(gens)
    G2 = G.stabilizer(2)
    assert G2.order() == 181440
    S = SymmetricGroup(3)
    assert [G.order() for G in S.basic_stabilizers] == [6, 2]


def test_center():
    # the center of the dihedral group D_n is of order 2 for even n
    for i in (4, 6, 10):
        D = DihedralGroup(i)
        assert (D.center()).order() == 2
    # the center of the dihedral group D_n is of order 1 for odd n>2
    for i in (3, 5, 7):
        D = DihedralGroup(i)
        assert (D.center()).order() == 1
    # the center of an abelian group is the group itself
    for i in (2, 3, 5):
        for j in (1, 5, 7):
            for k in (1, 1, 11):
                G = AbelianGroup(i, j, k)
                assert G.center().is_subgroup(G)
    # the center of a nonabelian simple group is trivial
    for i in(1, 5, 9):
        A = AlternatingGroup(i)
        assert (A.center()).order() == 1
    # brute-force verifications
    D = DihedralGroup(5)
    A = AlternatingGroup(3)
    C = CyclicGroup(4)
    G.is_subgroup(D*A*C)
    assert _verify_centralizer(G, G)


def test_centralizer():
    # the centralizer of the trivial group is the entire group
    S = SymmetricGroup(2)
    assert S.centralizer(Permutation(list(range(2)))).is_subgroup(S)
    A = AlternatingGroup(5)
    assert A.centralizer(Permutation(list(range(5)))).is_subgroup(A)
    # a centralizer in the trivial group is the trivial group itself
    triv = PermutationGroup([Permutation([0, 1, 2, 3])])
    D = DihedralGroup(4)
    assert triv.centralizer(D).is_subgroup(triv)
    # brute-force verifications for centralizers of groups
    for i in (4, 5, 6):
        S = SymmetricGroup(i)
        A = AlternatingGroup(i)
        C = CyclicGroup(i)
        D = DihedralGroup(i)
        for gp in (S, A, C, D):
            for gp2 in (S, A, C, D):
                if not gp2.is_subgroup(gp):
                    assert _verify_centralizer(gp, gp2)
    # verify the centralizer for all elements of several groups
    S = SymmetricGroup(5)
    elements = list(S.generate_dimino())
    for element in elements:
        assert _verify_centralizer(S, element)
    A = AlternatingGroup(5)
    elements = list(A.generate_dimino())
    for element in elements:
        assert _verify_centralizer(A, element)
    D = DihedralGroup(7)
    elements = list(D.generate_dimino())
    for element in elements:
        assert _verify_centralizer(D, element)
    # verify centralizers of small groups within small groups
    small = []
    for i in (1, 2, 3):
        small.append(SymmetricGroup(i))
        small.append(AlternatingGroup(i))
        small.append(DihedralGroup(i))
        small.append(CyclicGroup(i))
    for gp in small:
        for gp2 in small:
            if gp.degree == gp2.degree:
                assert _verify_centralizer(gp, gp2)


def test_coset_rank():
    gens_cube = [[1, 3, 5, 7, 0, 2, 4, 6], [1, 3, 0, 2, 5, 7, 4, 6]]
    gens = [Permutation(p) for p in gens_cube]
    G = PermutationGroup(gens)
    i = 0
    for h in G.generate(af=True):
        rk = G.coset_rank(h)
        assert rk == i
        h1 = G.coset_unrank(rk, af=True)
        assert h == h1
        i += 1
    assert G.coset_unrank(48) is None
    assert G.coset_unrank(G.coset_rank(gens[0])) == gens[0]


def test_coset_factor():
    a = Permutation([0, 2, 1])
    G = PermutationGroup([a])
    c = Permutation([2, 1, 0])
    assert not G.coset_factor(c)
    assert G.coset_rank(c) is None

    a = Permutation([2, 0, 1, 3, 4, 5])
    b = Permutation([2, 1, 3, 4, 5, 0])
    g = PermutationGroup([a, b])
    assert g.order() == 360
    d = Permutation([1, 0, 2, 3, 4, 5])
    assert not g.coset_factor(d.array_form)
    assert not g.contains(d)
    assert Permutation(2) in G
    c = Permutation([1, 0, 2, 3, 5, 4])
    v = g.coset_factor(c, True)
    tr = g.basic_transversals
    p = Permutation.rmul(*[tr[i][v[i]] for i in range(len(g.base))])
    assert p == c
    v = g.coset_factor(c)
    p = Permutation.rmul(*v)
    assert p == c
    assert g.contains(c)
    G = PermutationGroup([Permutation([2, 1, 0])])
    p = Permutation([1, 0, 2])
    assert G.coset_factor(p) == []


def test_orbits():
    a = Permutation([2, 0, 1])
    b = Permutation([2, 1, 0])
    g = PermutationGroup([a, b])
    assert g.orbit(0) == {0, 1, 2}
    assert g.orbits() == [{0, 1, 2}]
    assert g.is_transitive() and g.is_transitive(strict=False)
    assert g.orbit_transversal(0) == \
        [Permutation(
            [0, 1, 2]), Permutation([2, 0, 1]), Permutation([1, 2, 0])]
    assert g.orbit_transversal(0, True) == \
        [(0, Permutation([0, 1, 2])), (2, Permutation([2, 0, 1])),
        (1, Permutation([1, 2, 0]))]

    G = DihedralGroup(6)
    transversal, slps = _orbit_transversal(G.degree, G.generators, 0, True, slp=True)
    for i, t in transversal:
        slp = slps[i]
        w = G.identity
        for s in slp:
            w = G.generators[s]*w
        assert w == t

    a = Permutation(list(range(1, 100)) + [0])
    G = PermutationGroup([a])
    assert [min(o) for o in G.orbits()] == [0]
    G = PermutationGroup(rubik_cube_generators())
    assert [min(o) for o in G.orbits()] == [0, 1]
    assert not G.is_transitive() and not G.is_transitive(strict=False)
    G = PermutationGroup([Permutation(0, 1, 3), Permutation(3)(0, 1)])
    assert not G.is_transitive() and G.is_transitive(strict=False)
    assert PermutationGroup(
        Permutation(3)).is_transitive(strict=False) is False


def test_is_normal():
    gens_s5 = [Permutation(p) for p in [[1, 2, 3, 4, 0], [2, 1, 4, 0, 3]]]
    G1 = PermutationGroup(gens_s5)
    assert G1.order() == 120
    gens_a5 = [Permutation(p) for p in [[1, 0, 3, 2, 4], [2, 1, 4, 3, 0]]]
    G2 = PermutationGroup(gens_a5)
    assert G2.order() == 60
    assert G2.is_normal(G1)
    gens3 = [Permutation(p) for p in [[2, 1, 3, 0, 4], [1, 2, 0, 3, 4]]]
    G3 = PermutationGroup(gens3)
    assert not G3.is_normal(G1)
    assert G3.order() == 12
    G4 = G1.normal_closure(G3.generators)
    assert G4.order() == 60
    gens5 = [Permutation(p) for p in [[1, 2, 3, 0, 4], [1, 2, 0, 3, 4]]]
    G5 = PermutationGroup(gens5)
    assert G5.order() == 24
    G6 = G1.normal_closure(G5.generators)
    assert G6.order() == 120
    assert G1.is_subgroup(G6)
    assert not G1.is_subgroup(G4)
    assert G2.is_subgroup(G4)
    I5 = PermutationGroup(Permutation(4))
    assert I5.is_normal(G5)
    assert I5.is_normal(G6, strict=False)
    p1 = Permutation([1, 0, 2, 3, 4])
    p2 = Permutation([0, 1, 2, 4, 3])
    p3 = Permutation([3, 4, 2, 1, 0])
    id_ = Permutation([0, 1, 2, 3, 4])
    H = PermutationGroup([p1, p3])
    H_n1 = PermutationGroup([p1, p2])
    H_n2_1 = PermutationGroup(p1)
    H_n2_2 = PermutationGroup(p2)
    H_id = PermutationGroup(id_)
    assert H_n1.is_normal(H)
    assert H_n2_1.is_normal(H_n1)
    assert H_n2_2.is_normal(H_n1)
    assert H_id.is_normal(H_n2_1)
    assert H_id.is_normal(H_n1)
    assert H_id.is_normal(H)
    assert not H_n2_1.is_normal(H)
    assert not H_n2_2.is_normal(H)


def test_eq():
    a = [[1, 2, 0, 3, 4, 5], [1, 0, 2, 3, 4, 5], [2, 1, 0, 3, 4, 5], [
        1, 2, 0, 3, 4, 5]]
    a = [Permutation(p) for p in a + [[1, 2, 3, 4, 5, 0]]]
    g = Permutation([1, 2, 3, 4, 5, 0])
    G1, G2, G3 = [PermutationGroup(x) for x in [a[:2], a[2:4], [g, g**2]]]
    assert G1.order() == G2.order() == G3.order() == 6
    assert G1.is_subgroup(G2)
    assert not G1.is_subgroup(G3)
    G4 = PermutationGroup([Permutation([0, 1])])
    assert not G1.is_subgroup(G4)
    assert G4.is_subgroup(G1, 0)
    assert PermutationGroup(g, g).is_subgroup(PermutationGroup(g))
    assert SymmetricGroup(3).is_subgroup(SymmetricGroup(4), 0)
    assert SymmetricGroup(3).is_subgroup(SymmetricGroup(3)*CyclicGroup(5), 0)
    assert not CyclicGroup(5).is_subgroup(SymmetricGroup(3)*CyclicGroup(5), 0)
    assert CyclicGroup(3).is_subgroup(SymmetricGroup(3)*CyclicGroup(5), 0)


def test_derived_subgroup():
    a = Permutation([1, 0, 2, 4, 3])
    b = Permutation([0, 1, 3, 2, 4])
    G = PermutationGroup([a, b])
    C = G.derived_subgroup()
    assert C.order() == 3
    assert C.is_normal(G)
    assert C.is_subgroup(G, 0)
    assert not G.is_subgroup(C, 0)
    gens_cube = [[1, 3, 5, 7, 0, 2, 4, 6], [1, 3, 0, 2, 5, 7, 4, 6]]
    gens = [Permutation(p) for p in gens_cube]
    G = PermutationGroup(gens)
    C = G.derived_subgroup()
    assert C.order() == 12


def test_is_solvable():
    a = Permutation([1, 2, 0])
    b = Permutation([1, 0, 2])
    G = PermutationGroup([a, b])
    assert G.is_solvable
    G = PermutationGroup([a])
    assert G.is_solvable
    a = Permutation([1, 2, 3, 4, 0])
    b = Permutation([1, 0, 2, 3, 4])
    G = PermutationGroup([a, b])
    assert not G.is_solvable
    P = SymmetricGroup(10)
    S = P.sylow_subgroup(3)
    assert S.is_solvable

def test_rubik1():
    gens = rubik_cube_generators()
    gens1 = [gens[-1]] + [p**2 for p in gens[1:]]
    G1 = PermutationGroup(gens1)
    assert G1.order() == 19508428800
    gens2 = [p**2 for p in gens]
    G2 = PermutationGroup(gens2)
    assert G2.order() == 663552
    assert G2.is_subgroup(G1, 0)
    C1 = G1.derived_subgroup()
    assert C1.order() == 4877107200
    assert C1.is_subgroup(G1, 0)
    assert not G2.is_subgroup(C1, 0)

    G = RubikGroup(2)
    assert G.order() == 3674160


@XFAIL
def test_rubik():
    skip('takes too much time')
    G = PermutationGroup(rubik_cube_generators())
    assert G.order() == 43252003274489856000
    G1 = PermutationGroup(G[:3])
    assert G1.order() == 170659735142400
    assert not G1.is_normal(G)
    G2 = G.normal_closure(G1.generators)
    assert G2.is_subgroup(G)


def test_direct_product():
    C = CyclicGroup(4)
    D = DihedralGroup(4)
    G = C*C*C
    assert G.order() == 64
    assert G.degree == 12
    assert len(G.orbits()) == 3
    assert G.is_abelian is True
    H = D*C
    assert H.order() == 32
    assert H.is_abelian is False


def test_orbit_rep():
    G = DihedralGroup(6)
    assert G.orbit_rep(1, 3) in [Permutation([2, 3, 4, 5, 0, 1]),
    Permutation([4, 3, 2, 1, 0, 5])]
    H = CyclicGroup(4)*G
    assert H.orbit_rep(1, 5) is False


def test_schreier_vector():
    G = CyclicGroup(50)
    v = [0]*50
    v[23] = -1
    assert G.schreier_vector(23) == v
    H = DihedralGroup(8)
    assert H.schreier_vector(2) == [0, 1, -1, 0, 0, 1, 0, 0]
    L = SymmetricGroup(4)
    assert L.schreier_vector(1) == [1, -1, 0, 0]


def test_random_pr():
    D = DihedralGroup(6)
    r = 11
    n = 3
    _random_prec_n = {}
    _random_prec_n[0] = {'s': 7, 't': 3, 'x': 2, 'e': -1}
    _random_prec_n[1] = {'s': 5, 't': 5, 'x': 1, 'e': -1}
    _random_prec_n[2] = {'s': 3, 't': 4, 'x': 2, 'e': 1}
    D._random_pr_init(r, n, _random_prec_n=_random_prec_n)
    assert D._random_gens[11] == [0, 1, 2, 3, 4, 5]
    _random_prec = {'s': 2, 't': 9, 'x': 1, 'e': -1}
    assert D.random_pr(_random_prec=_random_prec) == \
        Permutation([0, 5, 4, 3, 2, 1])


def test_is_alt_sym():
    G = DihedralGroup(10)
    assert G.is_alt_sym() is False
    assert G._eval_is_alt_sym_naive() is False
    assert G._eval_is_alt_sym_naive(only_alt=True) is False
    assert G._eval_is_alt_sym_naive(only_sym=True) is False

    S = SymmetricGroup(10)
    assert S._eval_is_alt_sym_naive() is True
    assert S._eval_is_alt_sym_naive(only_alt=True) is False
    assert S._eval_is_alt_sym_naive(only_sym=True) is True

    N_eps = 10
    _random_prec = {'N_eps': N_eps,
        0: Permutation([[2], [1, 4], [0, 6, 7, 8, 9, 3, 5]]),
        1: Permutation([[1, 8, 7, 6, 3, 5, 2, 9], [0, 4]]),
        2: Permutation([[5, 8], [4, 7], [0, 1, 2, 3, 6, 9]]),
        3: Permutation([[3], [0, 8, 2, 7, 4, 1, 6, 9, 5]]),
        4: Permutation([[8], [4, 7, 9], [3, 6], [0, 5, 1, 2]]),
        5: Permutation([[6], [0, 2, 4, 5, 1, 8, 3, 9, 7]]),
        6: Permutation([[6, 9, 8], [4, 5], [1, 3, 7], [0, 2]]),
        7: Permutation([[4], [0, 2, 9, 1, 3, 8, 6, 5, 7]]),
        8: Permutation([[1, 5, 6, 3], [0, 2, 7, 8, 4, 9]]),
        9: Permutation([[8], [6, 7], [2, 3, 4, 5], [0, 1, 9]])}
    assert S.is_alt_sym(_random_prec=_random_prec) is True

    A = AlternatingGroup(10)
    assert A._eval_is_alt_sym_naive() is True
    assert A._eval_is_alt_sym_naive(only_alt=True) is True
    assert A._eval_is_alt_sym_naive(only_sym=True) is False

    _random_prec = {'N_eps': N_eps,
        0: Permutation([[1, 6, 4, 2, 7, 8, 5, 9, 3], [0]]),
        1: Permutation([[1], [0, 5, 8, 4, 9, 2, 3, 6, 7]]),
        2: Permutation([[1, 9, 8, 3, 2, 5], [0, 6, 7, 4]]),
        3: Permutation([[6, 8, 9], [4, 5], [1, 3, 7, 2], [0]]),
        4: Permutation([[8], [5], [4], [2, 6, 9, 3], [1], [0, 7]]),
        5: Permutation([[3, 6], [0, 8, 1, 7, 5, 9, 4, 2]]),
        6: Permutation([[5], [2, 9], [1, 8, 3], [0, 4, 7, 6]]),
        7: Permutation([[1, 8, 4, 7, 2, 3], [0, 6, 9, 5]]),
        8: Permutation([[5, 8, 7], [3], [1, 4, 2, 6], [0, 9]]),
        9: Permutation([[4, 9, 6], [3, 8], [1, 2], [0, 5, 7]])}
    assert A.is_alt_sym(_random_prec=_random_prec) is False

    G = PermutationGroup(
        Permutation(1, 3, size=8)(0, 2, 4, 6),
        Permutation(5, 7, size=8)(0, 2, 4, 6))
    assert G.is_alt_sym() is False

    # Tests for monte-carlo c_n parameter setting, and which guarantees
    # to give False.
    G = DihedralGroup(10)
    assert G._eval_is_alt_sym_monte_carlo() is False
    G = DihedralGroup(20)
    assert G._eval_is_alt_sym_monte_carlo() is False

    # A dry-running test to check if it looks up for the updated cache.
    G = DihedralGroup(6)
    G.is_alt_sym()
    assert G.is_alt_sym() is False


def test_minimal_block():
    D = DihedralGroup(6)
    block_system = D.minimal_block([0, 3])
    for i in range(3):
        assert block_system[i] == block_system[i + 3]
    S = SymmetricGroup(6)
    assert S.minimal_block([0, 1]) == [0, 0, 0, 0, 0, 0]

    assert Tetra.pgroup.minimal_block([0, 1]) == [0, 0, 0, 0]

    P1 = PermutationGroup(Permutation(1, 5)(2, 4), Permutation(0, 1, 2, 3, 4, 5))
    P2 = PermutationGroup(Permutation(0, 1, 2, 3, 4, 5), Permutation(1, 5)(2, 4))
    assert P1.minimal_block([0, 2]) == [0, 1, 0, 1, 0, 1]
    assert P2.minimal_block([0, 2]) == [0, 1, 0, 1, 0, 1]


def test_minimal_blocks():
    P = PermutationGroup(Permutation(1, 5)(2, 4), Permutation(0, 1, 2, 3, 4, 5))
    assert P.minimal_blocks() == [[0, 1, 0, 1, 0, 1], [0, 1, 2, 0, 1, 2]]

    P = SymmetricGroup(5)
    assert P.minimal_blocks() == [[0]*5]

    P = PermutationGroup(Permutation(0, 3))
    assert P.minimal_blocks() is False


def test_max_div():
    S = SymmetricGroup(10)
    assert S.max_div == 5


def test_is_primitive():
    S = SymmetricGroup(5)
    assert S.is_primitive() is True
    C = CyclicGroup(7)
    assert C.is_primitive() is True

    a = Permutation(0, 1, 2, size=6)
    b = Permutation(3, 4, 5, size=6)
    G = PermutationGroup(a, b)
    assert G.is_primitive() is False


def test_random_stab():
    S = SymmetricGroup(5)
    _random_el = Permutation([1, 3, 2, 0, 4])
    _random_prec = {'rand': _random_el}
    g = S.random_stab(2, _random_prec=_random_prec)
    assert g == Permutation([1, 3, 2, 0, 4])
    h = S.random_stab(1)
    assert h(1) == 1


def test_transitivity_degree():
    perm = Permutation([1, 2, 0])
    C = PermutationGroup([perm])
    assert C.transitivity_degree == 1
    gen1 = Permutation([1, 2, 0, 3, 4])
    gen2 = Permutation([1, 2, 3, 4, 0])
    # alternating group of degree 5
    Alt = PermutationGroup([gen1, gen2])
    assert Alt.transitivity_degree == 3


def test_schreier_sims_random():
    assert sorted(Tetra.pgroup.base) == [0, 1]

    S = SymmetricGroup(3)
    base = [0, 1]
    strong_gens = [Permutation([1, 2, 0]), Permutation([1, 0, 2]),
                  Permutation([0, 2, 1])]
    assert S.schreier_sims_random(base, strong_gens, 5) == (base, strong_gens)
    D = DihedralGroup(3)
    _random_prec = {'g': [Permutation([2, 0, 1]), Permutation([1, 2, 0]),
                         Permutation([1, 0, 2])]}
    base = [0, 1]
    strong_gens = [Permutation([1, 2, 0]), Permutation([2, 1, 0]),
                  Permutation([0, 2, 1])]
    assert D.schreier_sims_random([], D.generators, 2,
           _random_prec=_random_prec) == (base, strong_gens)


def test_baseswap():
    S = SymmetricGroup(4)
    S.schreier_sims()
    base = S.base
    strong_gens = S.strong_gens
    assert base == [0, 1, 2]
    deterministic = S.baseswap(base, strong_gens, 1, randomized=False)
    randomized = S.baseswap(base, strong_gens, 1)
    assert deterministic[0] == [0, 2, 1]
    assert _verify_bsgs(S, deterministic[0], deterministic[1]) is True
    assert randomized[0] == [0, 2, 1]
    assert _verify_bsgs(S, randomized[0], randomized[1]) is True


def test_schreier_sims_incremental():
    identity = Permutation([0, 1, 2, 3, 4])
    TrivialGroup = PermutationGroup([identity])
    base, strong_gens = TrivialGroup.schreier_sims_incremental(base=[0, 1, 2])
    assert _verify_bsgs(TrivialGroup, base, strong_gens) is True
    S = SymmetricGroup(5)
    base, strong_gens = S.schreier_sims_incremental(base=[0, 1, 2])
    assert _verify_bsgs(S, base, strong_gens) is True
    D = DihedralGroup(2)
    base, strong_gens = D.schreier_sims_incremental(base=[1])
    assert _verify_bsgs(D, base, strong_gens) is True
    A = AlternatingGroup(7)
    gens = A.generators[:]
    gen0 = gens[0]
    gen1 = gens[1]
    gen1 = rmul(gen1, ~gen0)
    gen0 = rmul(gen0, gen1)
    gen1 = rmul(gen0, gen1)
    base, strong_gens = A.schreier_sims_incremental(base=[0, 1], gens=gens)
    assert _verify_bsgs(A, base, strong_gens) is True
    C = CyclicGroup(11)
    gen = C.generators[0]
    base, strong_gens = C.schreier_sims_incremental(gens=[gen**3])
    assert _verify_bsgs(C, base, strong_gens) is True


def _subgroup_search(i, j, k):
    prop_true = lambda x: True
    prop_fix_points = lambda x: [x(point) for point in points] == points
    prop_comm_g = lambda x: rmul(x, g) == rmul(g, x)
    prop_even = lambda x: x.is_even
    for i in range(i, j, k):
        S = SymmetricGroup(i)
        A = AlternatingGroup(i)
        C = CyclicGroup(i)
        Sym = S.subgroup_search(prop_true)
        assert Sym.is_subgroup(S)
        Alt = S.subgroup_search(prop_even)
        assert Alt.is_subgroup(A)
        Sym = S.subgroup_search(prop_true, init_subgroup=C)
        assert Sym.is_subgroup(S)
        points = [7]
        assert S.stabilizer(7).is_subgroup(S.subgroup_search(prop_fix_points))
        points = [3, 4]
        assert S.stabilizer(3).stabilizer(4).is_subgroup(
            S.subgroup_search(prop_fix_points))
        points = [3, 5]
        fix35 = A.subgroup_search(prop_fix_points)
        points = [5]
        fix5 = A.subgroup_search(prop_fix_points)
        assert A.subgroup_search(prop_fix_points, init_subgroup=fix35
            ).is_subgroup(fix5)
        base, strong_gens = A.schreier_sims_incremental()
        g = A.generators[0]
        comm_g = \
            A.subgroup_search(prop_comm_g, base=base, strong_gens=strong_gens)
        assert _verify_bsgs(comm_g, base, comm_g.generators) is True
        assert [prop_comm_g(gen) is True for gen in comm_g.generators]


def test_subgroup_search():
    _subgroup_search(10, 15, 2)


@XFAIL
def test_subgroup_search2():
    skip('takes too much time')
    _subgroup_search(16, 17, 1)


def test_normal_closure():
    # the normal closure of the trivial group is trivial
    S = SymmetricGroup(3)
    identity = Permutation([0, 1, 2])
    closure = S.normal_closure(identity)
    assert closure.is_trivial
    # the normal closure of the entire group is the entire group
    A = AlternatingGroup(4)
    assert A.normal_closure(A).is_subgroup(A)
    # brute-force verifications for subgroups
    for i in (3, 4, 5):
        S = SymmetricGroup(i)
        A = AlternatingGroup(i)
        D = DihedralGroup(i)
        C = CyclicGroup(i)
        for gp in (A, D, C):
            assert _verify_normal_closure(S, gp)
    # brute-force verifications for all elements of a group
    S = SymmetricGroup(5)
    elements = list(S.generate_dimino())
    for element in elements:
        assert _verify_normal_closure(S, element)
    # small groups
    small = []
    for i in (1, 2, 3):
        small.append(SymmetricGroup(i))
        small.append(AlternatingGroup(i))
        small.append(DihedralGroup(i))
        small.append(CyclicGroup(i))
    for gp in small:
        for gp2 in small:
            if gp2.is_subgroup(gp, 0) and gp2.degree == gp.degree:
                assert _verify_normal_closure(gp, gp2)


def test_derived_series():
    # the derived series of the trivial group consists only of the trivial group
    triv = PermutationGroup([Permutation([0, 1, 2])])
    assert triv.derived_series()[0].is_subgroup(triv)
    # the derived series for a simple group consists only of the group itself
    for i in (5, 6, 7):
        A = AlternatingGroup(i)
        assert A.derived_series()[0].is_subgroup(A)
    # the derived series for S_4 is S_4 > A_4 > K_4 > triv
    S = SymmetricGroup(4)
    series = S.derived_series()
    assert series[1].is_subgroup(AlternatingGroup(4))
    assert series[2].is_subgroup(DihedralGroup(2))
    assert series[3].is_trivial


def test_lower_central_series():
    # the lower central series of the trivial group consists of the trivial
    # group
    triv = PermutationGroup([Permutation([0, 1, 2])])
    assert triv.lower_central_series()[0].is_subgroup(triv)
    # the lower central series of a simple group consists of the group itself
    for i in (5, 6, 7):
        A = AlternatingGroup(i)
        assert A.lower_central_series()[0].is_subgroup(A)
    # GAP-verified example
    S = SymmetricGroup(6)
    series = S.lower_central_series()
    assert len(series) == 2
    assert series[1].is_subgroup(AlternatingGroup(6))


def test_commutator():
    # the commutator of the trivial group and the trivial group is trivial
    S = SymmetricGroup(3)
    triv = PermutationGroup([Permutation([0, 1, 2])])
    assert S.commutator(triv, triv).is_subgroup(triv)
    # the commutator of the trivial group and any other group is again trivial
    A = AlternatingGroup(3)
    assert S.commutator(triv, A).is_subgroup(triv)
    # the commutator is commutative
    for i in (3, 4, 5):
        S = SymmetricGroup(i)
        A = AlternatingGroup(i)
        D = DihedralGroup(i)
        assert S.commutator(A, D).is_subgroup(S.commutator(D, A))
    # the commutator of an abelian group is trivial
    S = SymmetricGroup(7)
    A1 = AbelianGroup(2, 5)
    A2 = AbelianGroup(3, 4)
    triv = PermutationGroup([Permutation([0, 1, 2, 3, 4, 5, 6])])
    assert S.commutator(A1, A1).is_subgroup(triv)
    assert S.commutator(A2, A2).is_subgroup(triv)
    # examples calculated by hand
    S = SymmetricGroup(3)
    A = AlternatingGroup(3)
    assert S.commutator(A, S).is_subgroup(A)


def test_is_nilpotent():
    # every abelian group is nilpotent
    for i in (1, 2, 3):
        C = CyclicGroup(i)
        Ab = AbelianGroup(i, i + 2)
        assert C.is_nilpotent
        assert Ab.is_nilpotent
    Ab = AbelianGroup(5, 7, 10)
    assert Ab.is_nilpotent
    # A_5 is not solvable and thus not nilpotent
    assert AlternatingGroup(5).is_nilpotent is False


def test_is_trivial():
    for i in range(5):
        triv = PermutationGroup([Permutation(list(range(i)))])
        assert triv.is_trivial


def test_pointwise_stabilizer():
    S = SymmetricGroup(2)
    stab = S.pointwise_stabilizer([0])
    assert stab.generators == [Permutation(1)]
    S = SymmetricGroup(5)
    points = []
    stab = S
    for point in (2, 0, 3, 4, 1):
        stab = stab.stabilizer(point)
        points.append(point)
        assert S.pointwise_stabilizer(points).is_subgroup(stab)


def test_make_perm():
    assert cube.pgroup.make_perm(5, seed=list(range(5))) == \
        Permutation([4, 7, 6, 5, 0, 3, 2, 1])
    assert cube.pgroup.make_perm(7, seed=list(range(7))) == \
        Permutation([6, 7, 3, 2, 5, 4, 0, 1])


def test_elements():
    from sympy.sets.sets import FiniteSet

    p = Permutation(2, 3)
    assert set(PermutationGroup(p).elements) == {Permutation(3), Permutation(2, 3)}
    assert FiniteSet(*PermutationGroup(p).elements) \
        == FiniteSet(Permutation(2, 3), Permutation(3))


def test_is_group():
    assert PermutationGroup(Permutation(1,2), Permutation(2,4)).is_group is True
    assert SymmetricGroup(4).is_group is True


def test_PermutationGroup():
    assert PermutationGroup() == PermutationGroup(Permutation())
    assert (PermutationGroup() == 0) is False


def test_coset_transvesal():
    G = AlternatingGroup(5)
    H = PermutationGroup(Permutation(0,1,2),Permutation(1,2)(3,4))
    assert G.coset_transversal(H) == \
        [Permutation(4), Permutation(2, 3, 4), Permutation(2, 4, 3),
         Permutation(1, 2, 4), Permutation(4)(1, 2, 3), Permutation(1, 3)(2, 4),
         Permutation(0, 1, 2, 3, 4), Permutation(0, 1, 2, 4, 3),
         Permutation(0, 1, 3, 2, 4), Permutation(0, 2, 4, 1, 3)]


def test_coset_table():
    G = PermutationGroup(Permutation(0,1,2,3), Permutation(0,1,2),
         Permutation(0,4,2,7), Permutation(5,6), Permutation(0,7))
    H = PermutationGroup(Permutation(0,1,2,3), Permutation(0,7))
    assert G.coset_table(H) == \
        [[0, 0, 0, 0, 1, 2, 3, 3, 0, 0], [4, 5, 2, 5, 6, 0, 7, 7, 1, 1],
         [5, 4, 5, 1, 0, 6, 8, 8, 6, 6], [3, 3, 3, 3, 7, 8, 0, 0, 3, 3],
         [2, 1, 4, 4, 4, 4, 9, 9, 4, 4], [1, 2, 1, 2, 5, 5, 10, 10, 5, 5],
         [6, 6, 6, 6, 2, 1, 11, 11, 2, 2], [9, 10, 8, 10, 11, 3, 1, 1, 7, 7],
         [10, 9, 10, 7, 3, 11, 2, 2, 11, 11], [8, 7, 9, 9, 9, 9, 4, 4, 9, 9],
         [7, 8, 7, 8, 10, 10, 5, 5, 10, 10], [11, 11, 11, 11, 8, 7, 6, 6, 8, 8]]


def test_subgroup():
    G = PermutationGroup(Permutation(0,1,2), Permutation(0,2,3))
    H = G.subgroup([Permutation(0,1,3)])
    assert H.is_subgroup(G)


def test_generator_product():
    G = SymmetricGroup(5)
    p = Permutation(0, 2, 3)(1, 4)
    gens = G.generator_product(p)
    assert all(g in G.strong_gens for g in gens)
    w = G.identity
    for g in gens:
        w = g*w
    assert w == p


def test_sylow_subgroup():
    P = PermutationGroup(Permutation(1, 5)(2, 4), Permutation(0, 1, 2, 3, 4, 5))
    S = P.sylow_subgroup(2)
    assert S.order() == 4

    P = DihedralGroup(12)
    S = P.sylow_subgroup(3)
    assert S.order() == 3

    P = PermutationGroup(
        Permutation(1, 5)(2, 4), Permutation(0, 1, 2, 3, 4, 5), Permutation(0, 2))
    S = P.sylow_subgroup(3)
    assert S.order() == 9
    S = P.sylow_subgroup(2)
    assert S.order() == 8

    P = SymmetricGroup(10)
    S = P.sylow_subgroup(2)
    assert S.order() == 256
    S = P.sylow_subgroup(3)
    assert S.order() == 81
    S = P.sylow_subgroup(5)
    assert S.order() == 25

    # the length of the lower central series
    # of a p-Sylow subgroup of Sym(n) grows with
    # the highest exponent exp of p such
    # that n >= p**exp
    exp = 1
    length = 0
    for i in range(2, 9):
        P = SymmetricGroup(i)
        S = P.sylow_subgroup(2)
        ls = S.lower_central_series()
        if i // 2**exp > 0:
            # length increases with exponent
            assert len(ls) > length
            length = len(ls)
            exp += 1
        else:
            assert len(ls) == length

    G = SymmetricGroup(100)
    S = G.sylow_subgroup(3)
    assert G.order() % S.order() == 0
    assert G.order()/S.order() % 3 > 0

    G = AlternatingGroup(100)
    S = G.sylow_subgroup(2)
    assert G.order() % S.order() == 0
    assert G.order()/S.order() % 2 > 0

    G = DihedralGroup(18)
    S = G.sylow_subgroup(p=2)
    assert S.order() == 4

    G = DihedralGroup(50)
    S = G.sylow_subgroup(p=2)
    assert S.order() == 4


@slow
def test_presentation():
    def _test(P):
        G = P.presentation()
        return G.order() == P.order()

    def _strong_test(P):
        G = P.strong_presentation()
        chk = len(G.generators) == len(P.strong_gens)
        return chk and G.order() == P.order()

    P = PermutationGroup(Permutation(0,1,5,2)(3,7,4,6), Permutation(0,3,5,4)(1,6,2,7))
    assert _test(P)

    P = AlternatingGroup(5)
    assert _test(P)

    P = SymmetricGroup(5)
    assert _test(P)

    P = PermutationGroup(
        [Permutation(0,3,1,2), Permutation(3)(0,1), Permutation(0,1)(2,3)])
    assert _strong_test(P)

    P = DihedralGroup(6)
    assert _strong_test(P)

    a = Permutation(0,1)(2,3)
    b = Permutation(0,2)(3,1)
    c = Permutation(4,5)
    P = PermutationGroup(c, a, b)
    assert _strong_test(P)


def test_polycyclic():
    a = Permutation([0, 1, 2])
    b = Permutation([2, 1, 0])
    G = PermutationGroup([a, b])
    assert G.is_polycyclic is True

    a = Permutation([1, 2, 3, 4, 0])
    b = Permutation([1, 0, 2, 3, 4])
    G = PermutationGroup([a, b])
    assert G.is_polycyclic is False


def test_elementary():
    a = Permutation([1, 5, 2, 0, 3, 6, 4])
    G = PermutationGroup([a])
    assert G.is_elementary(7) is False

    a = Permutation(0, 1)(2, 3)
    b = Permutation(0, 2)(3, 1)
    G = PermutationGroup([a, b])
    assert G.is_elementary(2) is True
    c = Permutation(4, 5, 6)
    G = PermutationGroup([a, b, c])
    assert G.is_elementary(2) is False

    G = SymmetricGroup(4).sylow_subgroup(2)
    assert G.is_elementary(2) is False
    H = AlternatingGroup(4).sylow_subgroup(2)
    assert H.is_elementary(2) is True


def test_perfect():
    G = AlternatingGroup(3)
    assert G.is_perfect is False
    G = AlternatingGroup(5)
    assert G.is_perfect is True


def test_index():
    G = PermutationGroup(Permutation(0,1,2), Permutation(0,2,3))
    H = G.subgroup([Permutation(0,1,3)])
    assert G.index(H) == 4


def test_cyclic():
    G = SymmetricGroup(2)
    assert G.is_cyclic
    G = AbelianGroup(3, 7)
    assert G.is_cyclic
    G = AbelianGroup(7, 7)
    assert not G.is_cyclic
    G = AlternatingGroup(3)
    assert G.is_cyclic
    G = AlternatingGroup(4)
    assert not G.is_cyclic

    # Order less than 6
    G = PermutationGroup(Permutation(0, 1, 2), Permutation(0, 2, 1))
    assert G.is_cyclic
    G = PermutationGroup(
        Permutation(0, 1, 2, 3),
        Permutation(0, 2)(1, 3)
    )
    assert G.is_cyclic
    G = PermutationGroup(
        Permutation(3),
        Permutation(0, 1)(2, 3),
        Permutation(0, 2)(1, 3),
        Permutation(0, 3)(1, 2)
    )
    assert G.is_cyclic is False

    # Order 15
    G = PermutationGroup(
        Permutation(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
        Permutation(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13)
    )
    assert G.is_cyclic

    # Distinct prime orders
    assert PermutationGroup._distinct_primes_lemma([3, 5]) is True
    assert PermutationGroup._distinct_primes_lemma([5, 7]) is True
    assert PermutationGroup._distinct_primes_lemma([2, 3]) is None
    assert PermutationGroup._distinct_primes_lemma([3, 5, 7]) is None
    assert PermutationGroup._distinct_primes_lemma([5, 7, 13]) is True

    G = PermutationGroup(
        Permutation(0, 1, 2, 3),
        Permutation(0, 2)(1, 3))
    assert G.is_cyclic
    assert G._is_abelian

    # Non-abelian and therefore not cyclic
    G = PermutationGroup(*SymmetricGroup(3).generators)
    assert G.is_cyclic is False

    # Abelian and cyclic
    G = PermutationGroup(
        Permutation(0, 1, 2, 3),
        Permutation(4, 5, 6)
    )
    assert G.is_cyclic

    # Abelian but not cyclic
    G = PermutationGroup(
        Permutation(0, 1),
        Permutation(2, 3),
        Permutation(4, 5, 6)
    )
    assert G.is_cyclic is False


def test_dihedral():
    G = SymmetricGroup(2)
    assert G.is_dihedral
    G = SymmetricGroup(3)
    assert G.is_dihedral

    G = AbelianGroup(2, 2)
    assert G.is_dihedral
    G = CyclicGroup(4)
    assert not G.is_dihedral

    G = AbelianGroup(3, 5)
    assert not G.is_dihedral
    G = AbelianGroup(2)
    assert G.is_dihedral
    G = AbelianGroup(6)
    assert not G.is_dihedral

    # D6, generated by two adjacent flips
    G = PermutationGroup(
        Permutation(1, 5)(2, 4),
        Permutation(0, 1)(3, 4)(2, 5))
    assert G.is_dihedral

    # D7, generated by a flip and a rotation
    G = PermutationGroup(
        Permutation(1, 6)(2, 5)(3, 4),
        Permutation(0, 1, 2, 3, 4, 5, 6))
    assert G.is_dihedral

    # S4, presented by three generators, fails due to having exactly 9
    # elements of order 2:
    G = PermutationGroup(
        Permutation(0, 1), Permutation(0, 2),
        Permutation(0, 3))
    assert not G.is_dihedral

    # D7, given by three generators
    G = PermutationGroup(
        Permutation(1, 6)(2, 5)(3, 4),
        Permutation(2, 0)(3, 6)(4, 5),
        Permutation(0, 1, 2, 3, 4, 5, 6))
    assert G.is_dihedral


def test_abelian_invariants():
    G = AbelianGroup(2, 3, 4)
    assert G.abelian_invariants() == [2, 3, 4]
    G=PermutationGroup([Permutation(1, 2, 3, 4), Permutation(1, 2), Permutation(5, 6)])
    assert G.abelian_invariants() == [2, 2]
    G = AlternatingGroup(7)
    assert G.abelian_invariants() == []
    G = AlternatingGroup(4)
    assert G.abelian_invariants() == [3]
    G = DihedralGroup(4)
    assert G.abelian_invariants() == [2, 2]

    G = PermutationGroup([Permutation(1, 2, 3, 4, 5, 6, 7)])
    assert G.abelian_invariants() == [7]
    G = DihedralGroup(12)
    S = G.sylow_subgroup(3)
    assert S.abelian_invariants() == [3]
    G = PermutationGroup(Permutation(0, 1, 2), Permutation(0, 2, 3))
    assert G.abelian_invariants() == [3]
    G = PermutationGroup([Permutation(0, 1), Permutation(0, 2, 4, 6)(1, 3, 5, 7)])
    assert G.abelian_invariants() == [2, 4]
    G = SymmetricGroup(30)
    S = G.sylow_subgroup(2)
    assert S.abelian_invariants() == [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    S = G.sylow_subgroup(3)
    assert S.abelian_invariants() == [3, 3, 3, 3]
    S = G.sylow_subgroup(5)
    assert S.abelian_invariants() == [5, 5, 5]


def test_composition_series():
    a = Permutation(1, 2, 3)
    b = Permutation(1, 2)
    G = PermutationGroup([a, b])
    comp_series = G.composition_series()
    assert comp_series == G.derived_series()
    # The first group in the composition series is always the group itself and
    # the last group in the series is the trivial group.
    S = SymmetricGroup(4)
    assert S.composition_series()[0] == S
    assert len(S.composition_series()) == 5
    A = AlternatingGroup(4)
    assert A.composition_series()[0] == A
    assert len(A.composition_series()) == 4

    # the composition series for C_8 is C_8 > C_4 > C_2 > triv
    G = CyclicGroup(8)
    series = G.composition_series()
    assert is_isomorphic(series[1], CyclicGroup(4))
    assert is_isomorphic(series[2], CyclicGroup(2))
    assert series[3].is_trivial


def test_is_symmetric():
    a = Permutation(0, 1, 2)
    b = Permutation(0, 1, size=3)
    assert PermutationGroup(a, b).is_symmetric is True

    a = Permutation(0, 2, 1)
    b = Permutation(1, 2, size=3)
    assert PermutationGroup(a, b).is_symmetric is True

    a = Permutation(0, 1, 2, 3)
    b = Permutation(0, 3)(1, 2)
    assert PermutationGroup(a, b).is_symmetric is False

def test_conjugacy_class():
    S = SymmetricGroup(4)
    x = Permutation(1, 2, 3)
    C = {Permutation(0, 1, 2, size = 4), Permutation(0, 1, 3),
             Permutation(0, 2, 1, size = 4), Permutation(0, 2, 3),
             Permutation(0, 3, 1), Permutation(0, 3, 2),
             Permutation(1, 2, 3), Permutation(1, 3, 2)}
    assert S.conjugacy_class(x) == C

def test_conjugacy_classes():
    S = SymmetricGroup(3)
    expected = [{Permutation(size = 3)},
         {Permutation(0, 1, size = 3), Permutation(0, 2), Permutation(1, 2)},
         {Permutation(0, 1, 2), Permutation(0, 2, 1)}]
    computed = S.conjugacy_classes()

    assert len(expected) == len(computed)
    assert all(e in computed for e in expected)

def test_coset_class():
    a = Permutation(1, 2)
    b = Permutation(0, 1)
    G = PermutationGroup([a, b])
    #Creating right coset
    rht_coset = G*a
    #Checking whether it is left coset or right coset
    assert rht_coset.is_right_coset
    assert not rht_coset.is_left_coset
    #Creating list representation of coset
    list_repr = rht_coset.as_list()
    expected = [Permutation(0, 2), Permutation(0, 2, 1), Permutation(1, 2),
                Permutation(2), Permutation(2)(0, 1), Permutation(0, 1, 2)]
    for ele in list_repr:
        assert ele in expected
    #Creating left coset
    left_coset = a*G
    #Checking whether it is left coset or right coset
    assert not left_coset.is_right_coset
    assert left_coset.is_left_coset
    #Creating list representation of Coset
    list_repr = left_coset.as_list()
    expected = [Permutation(2)(0, 1), Permutation(0, 1, 2), Permutation(1, 2),
    Permutation(2), Permutation(0, 2), Permutation(0, 2, 1)]
    for ele in list_repr:
        assert ele in expected

    G = PermutationGroup(Permutation(1, 2, 3, 4), Permutation(2, 3, 4))
    H = PermutationGroup(Permutation(1, 2, 3, 4))
    g = Permutation(1, 3)(2, 4)
    rht_coset = Coset(g, H, G, dir='+')
    assert rht_coset.is_right_coset
    list_repr = rht_coset.as_list()
    expected = [Permutation(1, 2, 3, 4), Permutation(4), Permutation(1, 3)(2, 4),
    Permutation(1, 4, 3, 2)]
    for ele in list_repr:
        assert ele in expected

def test_symmetricpermutationgroup():
    a = SymmetricPermutationGroup(5)
    assert a.degree == 5
    assert a.order() == 120
    assert a.identity() == Permutation(4)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\tests\test_joint.py ===
from sympy.core.function import expand_mul
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy import Matrix, simplify, eye, zeros
from sympy.core.symbol import symbols
from sympy.physics.mechanics import (
    dynamicsymbols, RigidBody, Particle, JointsMethod, PinJoint, PrismaticJoint,
    CylindricalJoint, PlanarJoint, SphericalJoint, WeldJoint, Body)
from sympy.physics.mechanics.joint import Joint
from sympy.physics.vector import Vector, ReferenceFrame, Point
from sympy.testing.pytest import raises, warns_deprecated_sympy


t = dynamicsymbols._t # type: ignore


def _generate_body(interframe=False):
    N = ReferenceFrame('N')
    A = ReferenceFrame('A')
    P = RigidBody('P', frame=N)
    C = RigidBody('C', frame=A)
    if interframe:
        Pint, Cint = ReferenceFrame('P_int'), ReferenceFrame('C_int')
        Pint.orient_axis(N, N.x, pi)
        Cint.orient_axis(A, A.y, -pi / 2)
        return N, A, P, C, Pint, Cint
    return N, A, P, C


def test_Joint():
    parent = RigidBody('parent')
    child = RigidBody('child')
    raises(TypeError, lambda: Joint('J', parent, child))


def test_coordinate_generation():
    q, u, qj, uj = dynamicsymbols('q u q_J u_J')
    q0j, q1j, q2j, q3j, u0j, u1j, u2j, u3j = dynamicsymbols('q0:4_J u0:4_J')
    q0, q1, q2, q3, u0, u1, u2, u3 = dynamicsymbols('q0:4 u0:4')
    _, _, P, C = _generate_body()
    # Using PinJoint to access Joint's coordinate generation method
    J = PinJoint('J', P, C)
    # Test single given
    assert J._fill_coordinate_list(q, 1) == Matrix([q])
    assert J._fill_coordinate_list([u], 1) == Matrix([u])
    assert J._fill_coordinate_list([u], 1, offset=2) == Matrix([u])
    # Test None
    assert J._fill_coordinate_list(None, 1) == Matrix([qj])
    assert J._fill_coordinate_list([None], 1) == Matrix([qj])
    assert J._fill_coordinate_list([q0, None, None], 3) == Matrix(
        [q0, q1j, q2j])
    # Test autofill
    assert J._fill_coordinate_list(None, 3) == Matrix([q0j, q1j, q2j])
    assert J._fill_coordinate_list([], 3) == Matrix([q0j, q1j, q2j])
    # Test offset
    assert J._fill_coordinate_list([], 3, offset=1) == Matrix([q1j, q2j, q3j])
    assert J._fill_coordinate_list([q1, None, q3], 3, offset=1) == Matrix(
        [q1, q2j, q3])
    assert J._fill_coordinate_list(None, 2, offset=2) == Matrix([q2j, q3j])
    # Test label
    assert J._fill_coordinate_list(None, 1, 'u') == Matrix([uj])
    assert J._fill_coordinate_list([], 3, 'u') == Matrix([u0j, u1j, u2j])
    # Test single numbering
    assert J._fill_coordinate_list(None, 1, number_single=True) == Matrix([q0j])
    assert J._fill_coordinate_list([], 1, 'u', 2, True) == Matrix([u2j])
    assert J._fill_coordinate_list([], 3, 'q') == Matrix([q0j, q1j, q2j])
    # Test invalid number of coordinates supplied
    raises(ValueError, lambda: J._fill_coordinate_list([q0, q1], 1))
    raises(ValueError, lambda: J._fill_coordinate_list([u0, u1, None], 2, 'u'))
    raises(ValueError, lambda: J._fill_coordinate_list([q0, q1], 3))
    # Test incorrect coordinate type
    raises(TypeError, lambda: J._fill_coordinate_list([q0, symbols('q1')], 2))
    raises(TypeError, lambda: J._fill_coordinate_list([q0 + q1, q1], 2))
    # Test if derivative as generalized speed is allowed
    _, _, P, C = _generate_body()
    PinJoint('J', P, C, q1, q1.diff(t))
    # Test duplicate coordinates
    _, _, P, C = _generate_body()
    raises(ValueError, lambda: SphericalJoint('J', P, C, [q1j, None, None]))
    raises(ValueError, lambda: SphericalJoint('J', P, C, speeds=[u0, u0, u1]))


def test_pin_joint():
    P = RigidBody('P')
    C = RigidBody('C')
    l, m = symbols('l m')
    q, u = dynamicsymbols('q_J, u_J')
    Pj = PinJoint('J', P, C)
    assert Pj.name == 'J'
    assert Pj.parent == P
    assert Pj.child == C
    assert Pj.coordinates == Matrix([q])
    assert Pj.speeds == Matrix([u])
    assert Pj.kdes == Matrix([u - q.diff(t)])
    assert Pj.joint_axis == P.frame.x
    assert Pj.child_point.pos_from(C.masscenter) == Vector(0)
    assert Pj.parent_point.pos_from(P.masscenter) == Vector(0)
    assert Pj.parent_point.pos_from(Pj._child_point) == Vector(0)
    assert C.masscenter.pos_from(P.masscenter) == Vector(0)
    assert Pj.parent_interframe == P.frame
    assert Pj.child_interframe == C.frame
    assert Pj.__str__() == 'PinJoint: J  parent: P  child: C'

    P1 = RigidBody('P1')
    C1 = RigidBody('C1')
    Pint = ReferenceFrame('P_int')
    Pint.orient_axis(P1.frame, P1.y, pi / 2)
    J1 = PinJoint('J1', P1, C1, parent_point=l*P1.frame.x,
                  child_point=m*C1.frame.y, joint_axis=P1.frame.z,
                  parent_interframe=Pint)
    assert J1._joint_axis == P1.frame.z
    assert J1._child_point.pos_from(C1.masscenter) == m * C1.frame.y
    assert J1._parent_point.pos_from(P1.masscenter) == l * P1.frame.x
    assert J1._parent_point.pos_from(J1._child_point) == Vector(0)
    assert (P1.masscenter.pos_from(C1.masscenter) ==
            -l*P1.frame.x + m*C1.frame.y)
    assert J1.parent_interframe == Pint
    assert J1.child_interframe == C1.frame

    q, u = dynamicsymbols('q, u')
    N, A, P, C, Pint, Cint = _generate_body(True)
    parent_point = P.masscenter.locatenew('parent_point', N.x + N.y)
    child_point = C.masscenter.locatenew('child_point', C.y + C.z)
    J = PinJoint('J', P, C, q, u, parent_point=parent_point,
                 child_point=child_point, parent_interframe=Pint,
                 child_interframe=Cint, joint_axis=N.z)
    assert J.joint_axis == N.z
    assert J.parent_point.vel(N) == 0
    assert J.parent_point == parent_point
    assert J.child_point == child_point
    assert J.child_point.pos_from(P.masscenter) == N.x + N.y
    assert J.parent_point.pos_from(C.masscenter) == C.y + C.z
    assert C.masscenter.pos_from(P.masscenter) == N.x + N.y - C.y - C.z
    assert C.masscenter.vel(N).express(N) == (u * sin(q) - u * cos(q)) * N.x + (
            -u * sin(q) - u * cos(q)) * N.y
    assert J.parent_interframe == Pint
    assert J.child_interframe == Cint


def test_particle_compatibility():
    m, l = symbols('m l')
    C_frame = ReferenceFrame('C')
    P = Particle('P')
    C = Particle('C', mass=m)
    q, u = dynamicsymbols('q, u')
    J = PinJoint('J', P, C, q, u, child_interframe=C_frame,
                 child_point=l * C_frame.y)
    assert J.child_interframe == C_frame
    assert J.parent_interframe.name == 'J_P_frame'
    assert C.masscenter.pos_from(P.masscenter) == -l * C_frame.y
    assert C_frame.dcm(J.parent_interframe) == Matrix([[1, 0, 0],
                                                       [0, cos(q), sin(q)],
                                                       [0, -sin(q), cos(q)]])
    assert C.masscenter.vel(J.parent_interframe) == -l * u * C_frame.z
    # Test with specified joint axis
    P_frame = ReferenceFrame('P')
    C_frame = ReferenceFrame('C')
    P = Particle('P')
    C = Particle('C', mass=m)
    q, u = dynamicsymbols('q, u')
    J = PinJoint('J', P, C, q, u, parent_interframe=P_frame,
                 child_interframe=C_frame, child_point=l * C_frame.y,
                 joint_axis=P_frame.z)
    assert J.joint_axis == J.parent_interframe.z
    assert C_frame.dcm(J.parent_interframe) == Matrix([[cos(q), sin(q), 0],
                                                       [-sin(q), cos(q), 0],
                                                       [0, 0, 1]])
    assert P.masscenter.vel(J.parent_interframe) == 0
    assert C.masscenter.vel(J.parent_interframe) == l * u * C_frame.x
    q1, q2, q3, u1, u2, u3 = dynamicsymbols('q1:4 u1:4')
    qdot_to_u = {qi.diff(t): ui for qi, ui in ((q1, u1), (q2, u2), (q3, u3))}
    # Test compatibility for prismatic joint
    P, C = Particle('P'), Particle('C')
    J = PrismaticJoint('J', P, C, q, u)
    assert J.parent_interframe.dcm(J.child_interframe) == eye(3)
    assert C.masscenter.pos_from(P.masscenter) == q * J.parent_interframe.x
    assert P.masscenter.vel(J.parent_interframe) == 0
    assert C.masscenter.vel(J.parent_interframe) == u * J.parent_interframe.x
    # Test compatibility for cylindrical joint
    P, C = Particle('P'), Particle('C')
    P_frame = ReferenceFrame('P_frame')
    J = CylindricalJoint('J', P, C, q1, q2, u1, u2, parent_interframe=P_frame,
                         parent_point=l * P_frame.x, joint_axis=P_frame.y)
    assert J.parent_interframe.dcm(J.child_interframe) == Matrix([
        [cos(q1), 0, sin(q1)], [0, 1, 0], [-sin(q1), 0, cos(q1)]])
    assert C.masscenter.pos_from(P.masscenter) == l * P_frame.x + q2 * P_frame.y
    assert C.masscenter.vel(J.parent_interframe) == u2 * P_frame.y
    assert P.masscenter.vel(J.child_interframe).xreplace(qdot_to_u) == (
        -u2 * P_frame.y - l * u1 * P_frame.z)
    # Test compatibility for planar joint
    P, C = Particle('P'), Particle('C')
    C_frame = ReferenceFrame('C_frame')
    J = PlanarJoint('J', P, C, q1, [q2, q3], u1, [u2, u3],
                    child_interframe=C_frame, child_point=l * C_frame.z)
    P_frame = J.parent_interframe
    assert J.parent_interframe.dcm(J.child_interframe) == Matrix([
        [1, 0, 0], [0, cos(q1), -sin(q1)], [0, sin(q1), cos(q1)]])
    assert C.masscenter.pos_from(P.masscenter) == (
        -l * C_frame.z + q2 * P_frame.y + q3 * P_frame.z)
    assert C.masscenter.vel(J.parent_interframe) == (
        l * u1 * C_frame.y + u2 * P_frame.y + u3 * P_frame.z)
    # Test compatibility for weld joint
    P, C = Particle('P'), Particle('C')
    C_frame, P_frame = ReferenceFrame('C_frame'), ReferenceFrame('P_frame')
    J = WeldJoint('J', P, C, parent_interframe=P_frame,
                  child_interframe=C_frame, parent_point=l * P_frame.x,
                  child_point=l * C_frame.y)
    assert P_frame.dcm(C_frame) == eye(3)
    assert C.masscenter.pos_from(P.masscenter) == l * P_frame.x - l * C_frame.y
    assert C.masscenter.vel(J.parent_interframe) == 0


def test_body_compatibility():
    m, l = symbols('m l')
    C_frame = ReferenceFrame('C')
    with warns_deprecated_sympy():
        P = Body('P')
        C = Body('C', mass=m, frame=C_frame)
    q, u = dynamicsymbols('q, u')
    PinJoint('J', P, C, q, u, child_point=l * C_frame.y)
    assert C.frame == C_frame
    assert P.frame.name == 'P_frame'
    assert C.masscenter.pos_from(P.masscenter) == -l * C.y
    assert C.frame.dcm(P.frame) == Matrix([[1, 0, 0],
                                           [0, cos(q), sin(q)],
                                           [0, -sin(q), cos(q)]])
    assert C.masscenter.vel(P.frame) == -l * u * C.z


def test_pin_joint_double_pendulum():
    q1, q2 = dynamicsymbols('q1 q2')
    u1, u2 = dynamicsymbols('u1 u2')
    m, l = symbols('m l')
    N = ReferenceFrame('N')
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    C = RigidBody('C', frame=N)  # ceiling
    PartP = RigidBody('P', frame=A, mass=m)
    PartR = RigidBody('R', frame=B, mass=m)

    J1 = PinJoint('J1', C, PartP, speeds=u1, coordinates=q1,
                  child_point=-l*A.x, joint_axis=C.frame.z)
    J2 = PinJoint('J2', PartP, PartR, speeds=u2, coordinates=q2,
                  child_point=-l*B.x, joint_axis=PartP.frame.z)

    # Check orientation
    assert N.dcm(A) == Matrix([[cos(q1), -sin(q1), 0],
                               [sin(q1), cos(q1), 0], [0, 0, 1]])
    assert A.dcm(B) == Matrix([[cos(q2), -sin(q2), 0],
                               [sin(q2), cos(q2), 0], [0, 0, 1]])
    assert simplify(N.dcm(B)) == Matrix([[cos(q1 + q2), -sin(q1 + q2), 0],
                                                 [sin(q1 + q2), cos(q1 + q2), 0],
                                                 [0, 0, 1]])

    # Check Angular Velocity
    assert A.ang_vel_in(N) == u1 * N.z
    assert B.ang_vel_in(A) == u2 * A.z
    assert B.ang_vel_in(N) == u1 * N.z + u2 * A.z

    # Check kde
    assert J1.kdes == Matrix([u1 - q1.diff(t)])
    assert J2.kdes == Matrix([u2 - q2.diff(t)])

    # Check Linear Velocity
    assert PartP.masscenter.vel(N) == l*u1*A.y
    assert PartR.masscenter.vel(A) == l*u2*B.y
    assert PartR.masscenter.vel(N) == l*u1*A.y + l*(u1 + u2)*B.y


def test_pin_joint_chaos_pendulum():
    mA, mB, lA, lB, h = symbols('mA, mB, lA, lB, h')
    theta, phi, omega, alpha = dynamicsymbols('theta phi omega alpha')
    N = ReferenceFrame('N')
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    lA = (lB - h / 2) / 2
    lC = (lB/2 + h/4)
    rod = RigidBody('rod', frame=A, mass=mA)
    plate = RigidBody('plate', mass=mB, frame=B)
    C = RigidBody('C', frame=N)
    J1 = PinJoint('J1', C, rod, coordinates=theta, speeds=omega,
                  child_point=lA*A.z, joint_axis=N.y)
    J2 = PinJoint('J2', rod, plate, coordinates=phi, speeds=alpha,
                  parent_point=lC*A.z, joint_axis=A.z)

    # Check orientation
    assert A.dcm(N) == Matrix([[cos(theta), 0, -sin(theta)],
                               [0, 1, 0],
                               [sin(theta), 0, cos(theta)]])
    assert A.dcm(B) == Matrix([[cos(phi), -sin(phi), 0],
                               [sin(phi), cos(phi), 0],
                               [0, 0, 1]])
    assert B.dcm(N) == Matrix([
        [cos(phi)*cos(theta), sin(phi), -sin(theta)*cos(phi)],
        [-sin(phi)*cos(theta), cos(phi), sin(phi)*sin(theta)],
        [sin(theta), 0, cos(theta)]])

    # Check Angular Velocity
    assert A.ang_vel_in(N) == omega*N.y
    assert A.ang_vel_in(B) == -alpha*A.z
    assert N.ang_vel_in(B) == -omega*N.y - alpha*A.z

    # Check kde
    assert J1.kdes == Matrix([omega - theta.diff(t)])
    assert J2.kdes == Matrix([alpha - phi.diff(t)])

    # Check pos of masscenters
    assert C.masscenter.pos_from(rod.masscenter) == lA*A.z
    assert rod.masscenter.pos_from(plate.masscenter) == - lC * A.z

    # Check Linear Velocities
    assert rod.masscenter.vel(N) == (h/4 - lB/2)*omega*A.x
    assert plate.masscenter.vel(N) == ((h/4 - lB/2)*omega +
                                       (h/4 + lB/2)*omega)*A.x


def test_pin_joint_interframe():
    q, u = dynamicsymbols('q, u')
    # Check not connected
    N, A, P, C = _generate_body()
    Pint, Cint = ReferenceFrame('Pint'), ReferenceFrame('Cint')
    raises(ValueError, lambda: PinJoint('J', P, C, parent_interframe=Pint))
    raises(ValueError, lambda: PinJoint('J', P, C, child_interframe=Cint))
    # Check not fixed interframe
    Pint.orient_axis(N, N.z, q)
    Cint.orient_axis(A, A.z, q)
    raises(ValueError, lambda: PinJoint('J', P, C, parent_interframe=Pint))
    raises(ValueError, lambda: PinJoint('J', P, C, child_interframe=Cint))
    # Check only parent_interframe
    N, A, P, C = _generate_body()
    Pint = ReferenceFrame('Pint')
    Pint.orient_body_fixed(N, (pi / 4, pi, pi / 3), 'xyz')
    PinJoint('J', P, C, q, u, parent_point=N.x, child_point=-C.y,
             parent_interframe=Pint, joint_axis=Pint.x)
    assert simplify(N.dcm(A)) - Matrix([
        [-1 / 2, sqrt(3) * cos(q) / 2, -sqrt(3) * sin(q) / 2],
        [sqrt(6) / 4, sqrt(2) * (2 * sin(q) + cos(q)) / 4,
         sqrt(2) * (-sin(q) + 2 * cos(q)) / 4],
        [sqrt(6) / 4, sqrt(2) * (-2 * sin(q) + cos(q)) / 4,
         -sqrt(2) * (sin(q) + 2 * cos(q)) / 4]]) == zeros(3)
    assert A.ang_vel_in(N) == u * Pint.x
    assert C.masscenter.pos_from(P.masscenter) == N.x + A.y
    assert C.masscenter.vel(N) == u * A.z
    assert P.masscenter.vel(Pint) == Vector(0)
    assert C.masscenter.vel(Pint) == u * A.z
    # Check only child_interframe
    N, A, P, C = _generate_body()
    Cint = ReferenceFrame('Cint')
    Cint.orient_body_fixed(A, (2 * pi / 3, -pi, pi / 2), 'xyz')
    PinJoint('J', P, C, q, u, parent_point=-N.z, child_point=C.x,
             child_interframe=Cint, joint_axis=P.x + P.z)
    assert simplify(N.dcm(A)) == Matrix([
        [-sqrt(2) * sin(q) / 2,
         -sqrt(3) * (cos(q) - 1) / 4 - cos(q) / 4 - S(1) / 4,
         sqrt(3) * (cos(q) + 1) / 4 - cos(q) / 4 + S(1) / 4],
        [cos(q), (sqrt(2) + sqrt(6)) * -sin(q) / 4,
         (-sqrt(2) + sqrt(6)) * sin(q) / 4],
        [sqrt(2) * sin(q) / 2,
         sqrt(3) * (cos(q) + 1) / 4 + cos(q) / 4 - S(1) / 4,
         sqrt(3) * (1 - cos(q)) / 4 + cos(q) / 4 + S(1) / 4]])
    assert A.ang_vel_in(N) == sqrt(2) * u / 2 * N.x + sqrt(2) * u / 2 * N.z
    assert C.masscenter.pos_from(P.masscenter) == - N.z - A.x
    assert C.masscenter.vel(N).simplify() == (
        -sqrt(6) - sqrt(2)) * u / 4 * A.y + (
               -sqrt(2) + sqrt(6)) * u / 4 * A.z
    assert C.masscenter.vel(Cint) == Vector(0)
    # Check combination
    N, A, P, C = _generate_body()
    Pint, Cint = ReferenceFrame('Pint'), ReferenceFrame('Cint')
    Pint.orient_body_fixed(N, (-pi / 2, pi, pi / 2), 'xyz')
    Cint.orient_body_fixed(A, (2 * pi / 3, -pi, pi / 2), 'xyz')
    PinJoint('J', P, C, q, u, parent_point=N.x - N.y, child_point=-C.z,
             parent_interframe=Pint, child_interframe=Cint,
             joint_axis=Pint.x + Pint.z)
    assert simplify(N.dcm(A)) == Matrix([
        [cos(q), (sqrt(2) + sqrt(6)) * -sin(q) / 4,
         (-sqrt(2) + sqrt(6)) * sin(q) / 4],
        [-sqrt(2) * sin(q) / 2,
         -sqrt(3) * (cos(q) + 1) / 4 - cos(q) / 4 + S(1) / 4,
         sqrt(3) * (cos(q) - 1) / 4 - cos(q) / 4 - S(1) / 4],
        [sqrt(2) * sin(q) / 2,
         sqrt(3) * (cos(q) - 1) / 4 + cos(q) / 4 + S(1) / 4,
         -sqrt(3) * (cos(q) + 1) / 4 + cos(q) / 4 - S(1) / 4]])
    assert A.ang_vel_in(N) == sqrt(2) * u / 2 * Pint.x + sqrt(
        2) * u / 2 * Pint.z
    assert C.masscenter.pos_from(P.masscenter) == N.x - N.y + A.z
    N_v_C = (-sqrt(2) + sqrt(6)) * u / 4 * A.x
    assert C.masscenter.vel(N).simplify() == N_v_C
    assert C.masscenter.vel(Pint).simplify() == N_v_C
    assert C.masscenter.vel(Cint) == Vector(0)


def test_pin_joint_joint_axis():
    q, u = dynamicsymbols('q, u')
    # Check parent as reference
    N, A, P, C, Pint, Cint = _generate_body(True)
    pin = PinJoint('J', P, C, q, u, parent_interframe=Pint,
                   child_interframe=Cint, joint_axis=P.y)
    assert pin.joint_axis == P.y
    assert N.dcm(A) == Matrix([[sin(q), 0, cos(q)], [0, -1, 0],
                               [cos(q), 0, -sin(q)]])
    # Check parent_interframe as reference
    N, A, P, C, Pint, Cint = _generate_body(True)
    pin = PinJoint('J', P, C, q, u, parent_interframe=Pint,
                   child_interframe=Cint, joint_axis=Pint.y)
    assert pin.joint_axis == Pint.y
    assert N.dcm(A) == Matrix([[-sin(q), 0, cos(q)], [0, -1, 0],
                               [cos(q), 0, sin(q)]])
    # Check combination of joint_axis with interframes supplied as vectors (2x)
    N, A, P, C = _generate_body()
    pin = PinJoint('J', P, C, q, u, parent_interframe=N.z,
                   child_interframe=-C.z, joint_axis=N.z)
    assert pin.joint_axis == N.z
    assert N.dcm(A) == Matrix([[-cos(q), -sin(q), 0], [-sin(q), cos(q), 0],
                               [0, 0, -1]])
    N, A, P, C = _generate_body()
    pin = PinJoint('J', P, C, q, u, parent_interframe=N.z,
                   child_interframe=-C.z, joint_axis=N.x)
    assert pin.joint_axis == N.x
    assert N.dcm(A) == Matrix([[-1, 0, 0], [0, cos(q), sin(q)],
                               [0, sin(q), -cos(q)]])
    # Check time varying axis
    N, A, P, C, Pint, Cint = _generate_body(True)
    raises(ValueError, lambda: PinJoint('J', P, C,
                                        joint_axis=cos(q) * N.x + sin(q) * N.y))
    # Check joint_axis provided in child frame
    raises(ValueError, lambda: PinJoint('J', P, C, joint_axis=C.x))
    # Check some invalid combinations
    raises(ValueError, lambda: PinJoint('J', P, C, joint_axis=P.x + C.y))
    raises(ValueError, lambda: PinJoint(
        'J', P, C, parent_interframe=Pint, child_interframe=Cint,
        joint_axis=Pint.x + C.y))
    raises(ValueError, lambda: PinJoint(
        'J', P, C, parent_interframe=Pint, child_interframe=Cint,
        joint_axis=P.x + Cint.y))
    # Check valid special combination
    N, A, P, C, Pint, Cint = _generate_body(True)
    PinJoint('J', P, C, parent_interframe=Pint, child_interframe=Cint,
             joint_axis=Pint.x + P.y)
    # Check invalid zero vector
    raises(Exception, lambda: PinJoint(
        'J', P, C, parent_interframe=Pint, child_interframe=Cint,
        joint_axis=Vector(0)))
    raises(Exception, lambda: PinJoint(
        'J', P, C, parent_interframe=Pint, child_interframe=Cint,
        joint_axis=P.y + Pint.y))


def test_pin_joint_arbitrary_axis():
    q, u = dynamicsymbols('q_J, u_J')

    # When the bodies are attached though masscenters but axes are opposite.
    N, A, P, C = _generate_body()
    PinJoint('J', P, C, child_interframe=-A.x)

    assert (-A.x).angle_between(N.x) == 0
    assert -A.x.express(N) == N.x
    assert A.dcm(N) == Matrix([[-1, 0, 0],
                               [0, -cos(q), -sin(q)],
                               [0, -sin(q), cos(q)]])
    assert A.ang_vel_in(N) == u*N.x
    assert A.ang_vel_in(N).magnitude() == sqrt(u**2)
    assert C.masscenter.pos_from(P.masscenter) == 0
    assert C.masscenter.pos_from(P.masscenter).express(N).simplify() == 0
    assert C.masscenter.vel(N) == 0

    # When axes are different and parent joint is at masscenter but child joint
    # is at a unit vector from child masscenter.
    N, A, P, C = _generate_body()
    PinJoint('J', P, C, child_interframe=A.y, child_point=A.x)

    assert A.y.angle_between(N.x) == 0  # Axis are aligned
    assert A.y.express(N) == N.x
    assert A.dcm(N) == Matrix([[0, -cos(q), -sin(q)],
                               [1, 0, 0],
                               [0, -sin(q), cos(q)]])
    assert A.ang_vel_in(N) == u*N.x
    assert A.ang_vel_in(N).express(A) == u * A.y
    assert A.ang_vel_in(N).magnitude() == sqrt(u**2)
    assert A.ang_vel_in(N).cross(A.y) == 0
    assert C.masscenter.vel(N) == u*A.z
    assert C.masscenter.pos_from(P.masscenter) == -A.x
    assert (C.masscenter.pos_from(P.masscenter).express(N).simplify() ==
            cos(q)*N.y + sin(q)*N.z)
    assert C.masscenter.vel(N).angle_between(A.x) == pi/2

    # Similar to previous case but wrt parent body
    N, A, P, C = _generate_body()
    PinJoint('J', P, C, parent_interframe=N.y, parent_point=N.x)

    assert N.y.angle_between(A.x) == 0  # Axis are aligned
    assert N.y.express(A) == A.x
    assert A.dcm(N) == Matrix([[0, 1, 0],
                               [-cos(q), 0, sin(q)],
                               [sin(q), 0, cos(q)]])
    assert A.ang_vel_in(N) == u*N.y
    assert A.ang_vel_in(N).express(A) == u*A.x
    assert A.ang_vel_in(N).magnitude() == sqrt(u**2)
    angle = A.ang_vel_in(N).angle_between(A.x)
    assert angle.xreplace({u: 1}) == 0
    assert C.masscenter.vel(N) == 0
    assert C.masscenter.pos_from(P.masscenter) == N.x

    # Both joint pos id defined but different axes
    N, A, P, C = _generate_body()
    PinJoint('J', P, C, parent_point=N.x, child_point=A.x,
             child_interframe=A.x + A.y)
    assert expand_mul(N.x.angle_between(A.x + A.y)) == 0  # Axis are aligned
    assert (A.x + A.y).express(N).simplify() == sqrt(2)*N.x
    assert simplify(A.dcm(N)) == Matrix([
        [sqrt(2)/2, -sqrt(2)*cos(q)/2, -sqrt(2)*sin(q)/2],
        [sqrt(2)/2, sqrt(2)*cos(q)/2, sqrt(2)*sin(q)/2],
        [0, -sin(q), cos(q)]])
    assert A.ang_vel_in(N) == u*N.x
    assert (A.ang_vel_in(N).express(A).simplify() ==
            (u*A.x + u*A.y)/sqrt(2))
    assert A.ang_vel_in(N).magnitude() == sqrt(u**2)
    angle = A.ang_vel_in(N).angle_between(A.x + A.y)
    assert angle.xreplace({u: 1}) == 0
    assert C.masscenter.vel(N).simplify() == (u * A.z)/sqrt(2)
    assert C.masscenter.pos_from(P.masscenter) == N.x - A.x
    assert (C.masscenter.pos_from(P.masscenter).express(N).simplify() ==
            (1 - sqrt(2)/2)*N.x + sqrt(2)*cos(q)/2*N.y +
            sqrt(2)*sin(q)/2*N.z)
    assert (C.masscenter.vel(N).express(N).simplify() ==
            -sqrt(2)*u*sin(q)/2*N.y + sqrt(2)*u*cos(q)/2*N.z)
    assert C.masscenter.vel(N).angle_between(A.x) == pi/2

    N, A, P, C = _generate_body()
    PinJoint('J', P, C, parent_point=N.x, child_point=A.x,
             child_interframe=A.x + A.y - A.z)
    assert expand_mul(N.x.angle_between(A.x + A.y - A.z)) == 0  # Axis aligned
    assert (A.x + A.y - A.z).express(N).simplify() == sqrt(3)*N.x
    assert simplify(A.dcm(N)) == Matrix([
        [sqrt(3)/3, -sqrt(6)*sin(q + pi/4)/3,
         sqrt(6)*cos(q + pi/4)/3],
        [sqrt(3)/3, sqrt(6)*cos(q + pi/12)/3,
         sqrt(6)*sin(q + pi/12)/3],
        [-sqrt(3)/3, sqrt(6)*cos(q + 5*pi/12)/3,
         sqrt(6)*sin(q + 5*pi/12)/3]])
    assert A.ang_vel_in(N) == u*N.x
    assert A.ang_vel_in(N).express(A).simplify() == (u*A.x + u*A.y -
                                                     u*A.z)/sqrt(3)
    assert A.ang_vel_in(N).magnitude() == sqrt(u**2)
    angle = A.ang_vel_in(N).angle_between(A.x + A.y-A.z)
    assert angle.xreplace({u: 1}).simplify() == 0
    assert C.masscenter.vel(N).simplify() == (u*A.y + u*A.z)/sqrt(3)
    assert C.masscenter.pos_from(P.masscenter) == N.x - A.x
    assert (C.masscenter.pos_from(P.masscenter).express(N).simplify() ==
            (1 - sqrt(3)/3)*N.x + sqrt(6)*sin(q + pi/4)/3*N.y -
            sqrt(6)*cos(q + pi/4)/3*N.z)
    assert (C.masscenter.vel(N).express(N).simplify() ==
            sqrt(6)*u*cos(q + pi/4)/3*N.y +
            sqrt(6)*u*sin(q + pi/4)/3*N.z)
    assert C.masscenter.vel(N).angle_between(A.x) == pi/2

    N, A, P, C = _generate_body()
    m, n = symbols('m n')
    PinJoint('J', P, C, parent_point=m * N.x, child_point=n * A.x,
             child_interframe=A.x + A.y - A.z,
             parent_interframe=N.x - N.y + N.z)
    angle = (N.x - N.y + N.z).angle_between(A.x + A.y - A.z)
    assert expand_mul(angle) == 0  # Axis are aligned
    assert ((A.x-A.y+A.z).express(N).simplify() ==
            (-4*cos(q)/3 - S(1)/3)*N.x + (S(1)/3 - 4*sin(q + pi/6)/3)*N.y +
            (4*cos(q + pi/3)/3 - S(1)/3)*N.z)
    assert simplify(A.dcm(N)) == Matrix([
        [S(1)/3 - 2*cos(q)/3, -2*sin(q + pi/6)/3 - S(1)/3,
         2*cos(q + pi/3)/3 + S(1)/3],
        [2*cos(q + pi/3)/3 + S(1)/3, 2*cos(q)/3 - S(1)/3,
         2*sin(q + pi/6)/3 + S(1)/3],
        [-2*sin(q + pi/6)/3 - S(1)/3, 2*cos(q + pi/3)/3 + S(1)/3,
         2*cos(q)/3 - S(1)/3]])
    assert (A.ang_vel_in(N) - (u*N.x - u*N.y + u*N.z)/sqrt(3)).simplify()
    assert A.ang_vel_in(N).express(A).simplify() == (u*A.x + u*A.y -
                                                     u*A.z)/sqrt(3)
    assert A.ang_vel_in(N).magnitude() == sqrt(u**2)
    angle = A.ang_vel_in(N).angle_between(A.x+A.y-A.z)
    assert angle.xreplace({u: 1}).simplify() == 0
    assert (C.masscenter.vel(N).simplify() ==
            sqrt(3)*n*u/3*A.y + sqrt(3)*n*u/3*A.z)
    assert C.masscenter.pos_from(P.masscenter) == m*N.x - n*A.x
    assert (C.masscenter.pos_from(P.masscenter).express(N).simplify() ==
            (m + n*(2*cos(q) - 1)/3)*N.x + n*(2*sin(q + pi/6) +
            1)/3*N.y - n*(2*cos(q + pi/3) + 1)/3*N.z)
    assert (C.masscenter.vel(N).express(N).simplify() ==
            - 2*n*u*sin(q)/3*N.x + 2*n*u*cos(q + pi/6)/3*N.y +
            2*n*u*sin(q + pi/3)/3*N.z)
    assert C.masscenter.vel(N).dot(N.x - N.y + N.z).simplify() == 0


def test_create_aligned_frame_pi():
    N, A, P, C = _generate_body()
    f = Joint._create_aligned_interframe(P, -P.x, P.x)
    assert f.z == P.z
    f = Joint._create_aligned_interframe(P, -P.y, P.y)
    assert f.x == P.x
    f = Joint._create_aligned_interframe(P, -P.z, P.z)
    assert f.y == P.y
    f = Joint._create_aligned_interframe(P, -P.x - P.y, P.x + P.y)
    assert f.z == P.z
    f = Joint._create_aligned_interframe(P, -P.y - P.z, P.y + P.z)
    assert f.x == P.x
    f = Joint._create_aligned_interframe(P, -P.x - P.z, P.x + P.z)
    assert f.y == P.y
    f = Joint._create_aligned_interframe(P, -P.x - P.y - P.z, P.x + P.y + P.z)
    assert f.y - f.z == P.y - P.z


def test_pin_joint_axis():
    q, u = dynamicsymbols('q u')
    # Test default joint axis
    N, A, P, C, Pint, Cint = _generate_body(True)
    J = PinJoint('J', P, C, q, u, parent_interframe=Pint, child_interframe=Cint)
    assert J.joint_axis == Pint.x
    # Test for the same joint axis expressed in different frames
    N_R_A = Matrix([[0, sin(q), cos(q)],
                    [0, -cos(q), sin(q)],
                    [1, 0, 0]])
    N, A, P, C, Pint, Cint = _generate_body(True)
    PinJoint('J', P, C, q, u, parent_interframe=Pint, child_interframe=Cint,
             joint_axis=N.z)
    assert N.dcm(A) == N_R_A
    N, A, P, C, Pint, Cint = _generate_body(True)
    PinJoint('J', P, C, q, u, parent_interframe=Pint, child_interframe=Cint,
             joint_axis=-Pint.z)
    assert N.dcm(A) == N_R_A
    # Test time varying joint axis
    N, A, P, C, Pint, Cint = _generate_body(True)
    raises(ValueError, lambda: PinJoint('J', P, C, joint_axis=q * N.z))


def test_locate_joint_pos():
    # Test Vector and default
    N, A, P, C = _generate_body()
    joint = PinJoint('J', P, C, parent_point=N.y + N.z)
    assert joint.parent_point.name == 'J_P_joint'
    assert joint.parent_point.pos_from(P.masscenter) == N.y + N.z
    assert joint.child_point == C.masscenter
    # Test Point objects
    N, A, P, C = _generate_body()
    parent_point = P.masscenter.locatenew('p', N.y + N.z)
    joint = PinJoint('J', P, C, parent_point=parent_point,
                     child_point=C.masscenter)
    assert joint.parent_point == parent_point
    assert joint.child_point == C.masscenter
    # Check invalid type
    N, A, P, C = _generate_body()
    raises(TypeError,
           lambda: PinJoint('J', P, C, parent_point=N.x.to_matrix(N)))
    # Test time varying positions
    q = dynamicsymbols('q')
    N, A, P, C = _generate_body()
    raises(ValueError, lambda: PinJoint('J', P, C, parent_point=q * N.x))
    N, A, P, C = _generate_body()
    child_point = C.masscenter.locatenew('p', q * A.y)
    raises(ValueError, lambda: PinJoint('J', P, C, child_point=child_point))
    # Test undefined position
    child_point = Point('p')
    raises(ValueError, lambda: PinJoint('J', P, C, child_point=child_point))


def test_locate_joint_frame():
    # Test rotated frame and default
    N, A, P, C = _generate_body()
    parent_interframe = ReferenceFrame('int_frame')
    parent_interframe.orient_axis(N, N.z, 1)
    joint = PinJoint('J', P, C, parent_interframe=parent_interframe)
    assert joint.parent_interframe == parent_interframe
    assert joint.parent_interframe.ang_vel_in(N) == 0
    assert joint.child_interframe == A
    # Test time varying orientations
    q = dynamicsymbols('q')
    N, A, P, C = _generate_body()
    parent_interframe = ReferenceFrame('int_frame')
    parent_interframe.orient_axis(N, N.z, q)
    raises(ValueError,
           lambda: PinJoint('J', P, C, parent_interframe=parent_interframe))
    # Test undefined frame
    N, A, P, C = _generate_body()
    child_interframe = ReferenceFrame('int_frame')
    child_interframe.orient_axis(N, N.z, 1)  # Defined with respect to parent
    raises(ValueError,
           lambda: PinJoint('J', P, C, child_interframe=child_interframe))


def test_prismatic_joint():
    _, _, P, C = _generate_body()
    q, u = dynamicsymbols('q_S, u_S')
    S = PrismaticJoint('S', P, C)
    assert S.name == 'S'
    assert S.parent == P
    assert S.child == C
    assert S.coordinates == Matrix([q])
    assert S.speeds == Matrix([u])
    assert S.kdes == Matrix([u - q.diff(t)])
    assert S.joint_axis == P.frame.x
    assert S.child_point.pos_from(C.masscenter) == Vector(0)
    assert S.parent_point.pos_from(P.masscenter) == Vector(0)
    assert S.parent_point.pos_from(S.child_point) == - q * P.frame.x
    assert P.masscenter.pos_from(C.masscenter) == - q * P.frame.x
    assert C.masscenter.vel(P.frame) == u * P.frame.x
    assert P.frame.ang_vel_in(C.frame) == 0
    assert C.frame.ang_vel_in(P.frame) == 0
    assert S.__str__() == 'PrismaticJoint: S  parent: P  child: C'

    N, A, P, C = _generate_body()
    l, m = symbols('l m')
    Pint = ReferenceFrame('P_int')
    Pint.orient_axis(P.frame, P.y, pi / 2)
    S = PrismaticJoint('S', P, C, parent_point=l * P.frame.x,
                       child_point=m * C.frame.y, joint_axis=P.frame.z,
                       parent_interframe=Pint)

    assert S.joint_axis == P.frame.z
    assert S.child_point.pos_from(C.masscenter) == m * C.frame.y
    assert S.parent_point.pos_from(P.masscenter) == l * P.frame.x
    assert S.parent_point.pos_from(S.child_point) == - q * P.frame.z
    assert P.masscenter.pos_from(C.masscenter) == - l * N.x - q * N.z + m * A.y
    assert C.masscenter.vel(P.frame) == u * P.frame.z
    assert P.masscenter.vel(Pint) == Vector(0)
    assert C.frame.ang_vel_in(P.frame) == 0
    assert P.frame.ang_vel_in(C.frame) == 0

    _, _, P, C = _generate_body()
    Pint = ReferenceFrame('P_int')
    Pint.orient_axis(P.frame, P.y, pi / 2)
    S = PrismaticJoint('S', P, C, parent_point=l * P.frame.z,
                       child_point=m * C.frame.x, joint_axis=P.frame.z,
                       parent_interframe=Pint)
    assert S.joint_axis == P.frame.z
    assert S.child_point.pos_from(C.masscenter) == m * C.frame.x
    assert S.parent_point.pos_from(P.masscenter) == l * P.frame.z
    assert S.parent_point.pos_from(S.child_point) == - q * P.frame.z
    assert P.masscenter.pos_from(C.masscenter) == (-l - q)*P.frame.z + m*C.frame.x
    assert C.masscenter.vel(P.frame) == u * P.frame.z
    assert C.frame.ang_vel_in(P.frame) == 0
    assert P.frame.ang_vel_in(C.frame) == 0


def test_prismatic_joint_arbitrary_axis():
    q, u = dynamicsymbols('q_S, u_S')

    N, A, P, C = _generate_body()
    PrismaticJoint('S', P, C, child_interframe=-A.x)

    assert (-A.x).angle_between(N.x) == 0
    assert -A.x.express(N) == N.x
    assert A.dcm(N) == Matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    assert C.masscenter.pos_from(P.masscenter) == q * N.x
    assert C.masscenter.pos_from(P.masscenter).express(A).simplify() == -q * A.x
    assert C.masscenter.vel(N) == u * N.x
    assert C.masscenter.vel(N).express(A) == -u * A.x
    assert A.ang_vel_in(N) == 0
    assert N.ang_vel_in(A) == 0

    #When axes are different and parent joint is at masscenter but child joint is at a unit vector from
    #child masscenter.
    N, A, P, C = _generate_body()
    PrismaticJoint('S', P, C, child_interframe=A.y, child_point=A.x)

    assert A.y.angle_between(N.x) == 0 #Axis are aligned
    assert A.y.express(N) == N.x
    assert A.dcm(N) == Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert C.masscenter.vel(N) == u * N.x
    assert C.masscenter.vel(N).express(A) == u * A.y
    assert C.masscenter.pos_from(P.masscenter) == q*N.x - A.x
    assert C.masscenter.pos_from(P.masscenter).express(N).simplify() == q*N.x + N.y
    assert A.ang_vel_in(N) == 0
    assert N.ang_vel_in(A) == 0

    #Similar to previous case but wrt parent body
    N, A, P, C = _generate_body()
    PrismaticJoint('S', P, C, parent_interframe=N.y, parent_point=N.x)

    assert N.y.angle_between(A.x) == 0 #Axis are aligned
    assert N.y.express(A) ==  A.x
    assert A.dcm(N) == Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    assert C.masscenter.vel(N) == u * N.y
    assert C.masscenter.vel(N).express(A) == u * A.x
    assert C.masscenter.pos_from(P.masscenter) == N.x + q*N.y
    assert A.ang_vel_in(N) == 0
    assert N.ang_vel_in(A) == 0

    #Both joint pos is defined but different axes
    N, A, P, C = _generate_body()
    PrismaticJoint('S', P, C, parent_point=N.x, child_point=A.x,
                   child_interframe=A.x + A.y)
    assert N.x.angle_between(A.x + A.y) == 0 #Axis are aligned
    assert (A.x + A.y).express(N) == sqrt(2)*N.x
    assert A.dcm(N) == Matrix([[sqrt(2)/2, -sqrt(2)/2, 0], [sqrt(2)/2, sqrt(2)/2, 0], [0, 0, 1]])
    assert C.masscenter.pos_from(P.masscenter) == (q + 1)*N.x - A.x
    assert C.masscenter.pos_from(P.masscenter).express(N) == (q - sqrt(2)/2 + 1)*N.x + sqrt(2)/2*N.y
    assert C.masscenter.vel(N).express(A) == u * (A.x + A.y)/sqrt(2)
    assert C.masscenter.vel(N) == u*N.x
    assert A.ang_vel_in(N) == 0
    assert N.ang_vel_in(A) == 0

    N, A, P, C = _generate_body()
    PrismaticJoint('S', P, C, parent_point=N.x, child_point=A.x,
                   child_interframe=A.x + A.y - A.z)
    assert N.x.angle_between(A.x + A.y - A.z).simplify() == 0 #Axis are aligned
    assert ((A.x + A.y - A.z).express(N) - sqrt(3)*N.x).simplify() == 0
    assert simplify(A.dcm(N)) == Matrix([[sqrt(3)/3, -sqrt(3)/3, sqrt(3)/3],
                                                 [sqrt(3)/3, sqrt(3)/6 + S(1)/2, S(1)/2 - sqrt(3)/6],
                                                 [-sqrt(3)/3, S(1)/2 - sqrt(3)/6, sqrt(3)/6 + S(1)/2]])
    assert C.masscenter.pos_from(P.masscenter) == (q + 1)*N.x - A.x
    assert (C.masscenter.pos_from(P.masscenter).express(N) -
        ((q - sqrt(3)/3 + 1)*N.x + sqrt(3)/3*N.y - sqrt(3)/3*N.z)).simplify() == 0
    assert C.masscenter.vel(N) == u*N.x
    assert (C.masscenter.vel(N).express(A) - (
        sqrt(3)*u/3*A.x + sqrt(3)*u/3*A.y - sqrt(3)*u/3*A.z)).simplify()
    assert A.ang_vel_in(N) == 0
    assert N.ang_vel_in(A) == 0

    N, A, P, C = _generate_body()
    m, n = symbols('m n')
    PrismaticJoint('S', P, C, parent_point=m*N.x, child_point=n*A.x,
                   child_interframe=A.x + A.y - A.z,
                   parent_interframe=N.x - N.y + N.z)
    # 0 angle means that the axis are aligned
    assert (N.x-N.y+N.z).angle_between(A.x+A.y-A.z).simplify() == 0
    assert ((A.x+A.y-A.z).express(N) - (N.x - N.y + N.z)).simplify() == 0
    assert simplify(A.dcm(N)) == Matrix([[-S(1)/3, -S(2)/3, S(2)/3],
                                                 [S(2)/3, S(1)/3, S(2)/3],
                                                 [-S(2)/3, S(2)/3, S(1)/3]])
    assert (C.masscenter.pos_from(P.masscenter) - (
        (m + sqrt(3)*q/3)*N.x - sqrt(3)*q/3*N.y + sqrt(3)*q/3*N.z - n*A.x)
            ).express(N).simplify() == 0
    assert (C.masscenter.pos_from(P.masscenter).express(N) - (
        (m + n/3 + sqrt(3)*q/3)*N.x + (2*n/3 - sqrt(3)*q/3)*N.y +
        (-2*n/3 + sqrt(3)*q/3)*N.z)).simplify() == 0
    assert (C.masscenter.vel(N).express(N) - (
        sqrt(3)*u/3*N.x - sqrt(3)*u/3*N.y + sqrt(3)*u/3*N.z)).simplify() == 0
    assert (C.masscenter.vel(N).express(A) -
            (sqrt(3)*u/3*A.x + sqrt(3)*u/3*A.y - sqrt(3)*u/3*A.z)).simplify() == 0
    assert A.ang_vel_in(N) == 0
    assert N.ang_vel_in(A) == 0


def test_cylindrical_joint():
    N, A, P, C = _generate_body()
    q0_def, q1_def, u0_def, u1_def = dynamicsymbols('q0:2_J, u0:2_J')
    Cj = CylindricalJoint('J', P, C)
    assert Cj.name == 'J'
    assert Cj.parent == P
    assert Cj.child == C
    assert Cj.coordinates == Matrix([q0_def, q1_def])
    assert Cj.speeds == Matrix([u0_def, u1_def])
    assert Cj.rotation_coordinate == q0_def
    assert Cj.translation_coordinate == q1_def
    assert Cj.rotation_speed == u0_def
    assert Cj.translation_speed == u1_def
    assert Cj.kdes == Matrix([u0_def - q0_def.diff(t), u1_def - q1_def.diff(t)])
    assert Cj.joint_axis == N.x
    assert Cj.child_point.pos_from(C.masscenter) == Vector(0)
    assert Cj.parent_point.pos_from(P.masscenter) == Vector(0)
    assert Cj.parent_point.pos_from(Cj._child_point) == -q1_def * N.x
    assert C.masscenter.pos_from(P.masscenter) == q1_def * N.x
    assert Cj.child_point.vel(N) == u1_def * N.x
    assert A.ang_vel_in(N) == u0_def * N.x
    assert Cj.parent_interframe == N
    assert Cj.child_interframe == A
    assert Cj.__str__() == 'CylindricalJoint: J  parent: P  child: C'

    q0, q1, u0, u1 = dynamicsymbols('q0:2, u0:2')
    l, m = symbols('l, m')
    N, A, P, C, Pint, Cint = _generate_body(True)
    Cj = CylindricalJoint('J', P, C, rotation_coordinate=q0, rotation_speed=u0,
                          translation_speed=u1, parent_point=m * N.x,
                          child_point=l * A.y, parent_interframe=Pint,
                          child_interframe=Cint, joint_axis=2 * N.z)
    assert Cj.coordinates == Matrix([q0, q1_def])
    assert Cj.speeds == Matrix([u0, u1])
    assert Cj.rotation_coordinate == q0
    assert Cj.translation_coordinate == q1_def
    assert Cj.rotation_speed == u0
    assert Cj.translation_speed == u1
    assert Cj.kdes == Matrix([u0 - q0.diff(t), u1 - q1_def.diff(t)])
    assert Cj.joint_axis == 2 * N.z
    assert Cj.child_point.pos_from(C.masscenter) == l * A.y
    assert Cj.parent_point.pos_from(P.masscenter) == m * N.x
    assert Cj.parent_point.pos_from(Cj._child_point) == -q1_def * N.z
    assert C.masscenter.pos_from(
        P.masscenter) == m * N.x + q1_def * N.z - l * A.y
    assert C.masscenter.vel(N) == u1 * N.z - u0 * l * A.z
    assert A.ang_vel_in(N) == u0 * N.z


def test_planar_joint():
    N, A, P, C = _generate_body()
    q0_def, q1_def, q2_def = dynamicsymbols('q0:3_J')
    u0_def, u1_def, u2_def = dynamicsymbols('u0:3_J')
    Cj = PlanarJoint('J', P, C)
    assert Cj.name == 'J'
    assert Cj.parent == P
    assert Cj.child == C
    assert Cj.coordinates == Matrix([q0_def, q1_def, q2_def])
    assert Cj.speeds == Matrix([u0_def, u1_def, u2_def])
    assert Cj.rotation_coordinate == q0_def
    assert Cj.planar_coordinates == Matrix([q1_def, q2_def])
    assert Cj.rotation_speed == u0_def
    assert Cj.planar_speeds == Matrix([u1_def, u2_def])
    assert Cj.kdes == Matrix([u0_def - q0_def.diff(t), u1_def - q1_def.diff(t),
                              u2_def - q2_def.diff(t)])
    assert Cj.rotation_axis == N.x
    assert Cj.planar_vectors == [N.y, N.z]
    assert Cj.child_point.pos_from(C.masscenter) == Vector(0)
    assert Cj.parent_point.pos_from(P.masscenter) == Vector(0)
    r_P_C = q1_def * N.y + q2_def * N.z
    assert Cj.parent_point.pos_from(Cj.child_point) == -r_P_C
    assert C.masscenter.pos_from(P.masscenter) == r_P_C
    assert Cj.child_point.vel(N) == u1_def * N.y + u2_def * N.z
    assert A.ang_vel_in(N) == u0_def * N.x
    assert Cj.parent_interframe == N
    assert Cj.child_interframe == A
    assert Cj.__str__() == 'PlanarJoint: J  parent: P  child: C'

    q0, q1, q2, u0, u1, u2 = dynamicsymbols('q0:3, u0:3')
    l, m = symbols('l, m')
    N, A, P, C, Pint, Cint = _generate_body(True)
    Cj = PlanarJoint('J', P, C, rotation_coordinate=q0,
                     planar_coordinates=[q1, q2], planar_speeds=[u1, u2],
                     parent_point=m * N.x, child_point=l * A.y,
                     parent_interframe=Pint, child_interframe=Cint)
    assert Cj.coordinates == Matrix([q0, q1, q2])
    assert Cj.speeds == Matrix([u0_def, u1, u2])
    assert Cj.rotation_coordinate == q0
    assert Cj.planar_coordinates == Matrix([q1, q2])
    assert Cj.rotation_speed == u0_def
    assert Cj.planar_speeds == Matrix([u1, u2])
    assert Cj.kdes == Matrix([u0_def - q0.diff(t), u1 - q1.diff(t),
                              u2 - q2.diff(t)])
    assert Cj.rotation_axis == Pint.x
    assert Cj.planar_vectors == [Pint.y, Pint.z]
    assert Cj.child_point.pos_from(C.masscenter) == l * A.y
    assert Cj.parent_point.pos_from(P.masscenter) == m * N.x
    assert Cj.parent_point.pos_from(Cj.child_point) == q1 * N.y + q2 * N.z
    assert C.masscenter.pos_from(
        P.masscenter) == m * N.x - q1 * N.y - q2 * N.z - l * A.y
    assert C.masscenter.vel(N) == -u1 * N.y - u2 * N.z + u0_def * l * A.x
    assert A.ang_vel_in(N) == u0_def * N.x


def test_planar_joint_advanced():
    # Tests whether someone is able to just specify two normals, which will form
    # the rotation axis seen from the parent and child body.
    # This specific example is a block on a slope, which has that same slope of
    # 30 degrees, so in the zero configuration the frames of the parent and
    # child are actually aligned.
    q0, q1, q2, u0, u1, u2 = dynamicsymbols('q0:3, u0:3')
    l1, l2 = symbols('l1:3')
    N, A, P, C = _generate_body()
    J = PlanarJoint('J', P, C, q0, [q1, q2], u0, [u1, u2],
                    parent_point=l1 * N.z,
                    child_point=-l2 * C.z,
                    parent_interframe=N.z + N.y / sqrt(3),
                    child_interframe=A.z + A.y / sqrt(3))
    assert J.rotation_axis.express(N) == (N.z + N.y / sqrt(3)).normalize()
    assert J.rotation_axis.express(A) == (A.z + A.y / sqrt(3)).normalize()
    assert J.rotation_axis.angle_between(N.z) == pi / 6
    assert N.dcm(A).xreplace({q0: 0, q1: 0, q2: 0}) == eye(3)
    N_R_A = Matrix([
        [cos(q0), -sqrt(3) * sin(q0) / 2, sin(q0) / 2],
        [sqrt(3) * sin(q0) / 2, 3 * cos(q0) / 4 + 1 / 4,
         sqrt(3) * (1 - cos(q0)) / 4],
        [-sin(q0) / 2, sqrt(3) * (1 - cos(q0)) / 4, cos(q0) / 4 + 3 / 4]])
    # N.dcm(A) == N_R_A did not work
    assert simplify(N.dcm(A) - N_R_A) == zeros(3)


def test_spherical_joint():
    N, A, P, C = _generate_body()
    q0, q1, q2, u0, u1, u2 = dynamicsymbols('q0:3_S, u0:3_S')
    S = SphericalJoint('S', P, C)
    assert S.name == 'S'
    assert S.parent == P
    assert S.child == C
    assert S.coordinates == Matrix([q0, q1, q2])
    assert S.speeds == Matrix([u0, u1, u2])
    assert S.kdes == Matrix([u0 - q0.diff(t), u1 - q1.diff(t), u2 - q2.diff(t)])
    assert S.child_point.pos_from(C.masscenter) == Vector(0)
    assert S.parent_point.pos_from(P.masscenter) == Vector(0)
    assert S.parent_point.pos_from(S.child_point) == Vector(0)
    assert P.masscenter.pos_from(C.masscenter) == Vector(0)
    assert C.masscenter.vel(N) == Vector(0)
    assert N.ang_vel_in(A) == (-u0 * cos(q1) * cos(q2) - u1 * sin(q2)) * A.x + (
            u0 * sin(q2) * cos(q1) - u1 * cos(q2)) * A.y + (
                   -u0 * sin(q1) - u2) * A.z
    assert A.ang_vel_in(N) == (u0 * cos(q1) * cos(q2) + u1 * sin(q2)) * A.x + (
            -u0 * sin(q2) * cos(q1) + u1 * cos(q2)) * A.y + (
                   u0 * sin(q1) + u2) * A.z
    assert S.__str__() == 'SphericalJoint: S  parent: P  child: C'
    assert S._rot_type == 'BODY'
    assert S._rot_order == 123
    assert S._amounts is None


def test_spherical_joint_speeds_as_derivative_terms():
    # This tests checks whether the system remains valid if the user chooses to
    # pass the derivative of the generalized coordinates as generalized speeds
    q0, q1, q2 = dynamicsymbols('q0:3')
    u0, u1, u2 = dynamicsymbols('q0:3', 1)
    N, A, P, C = _generate_body()
    S = SphericalJoint('S', P, C, coordinates=[q0, q1, q2], speeds=[u0, u1, u2])
    assert S.coordinates == Matrix([q0, q1, q2])
    assert S.speeds == Matrix([u0, u1, u2])
    assert S.kdes == Matrix([0, 0, 0])
    assert N.ang_vel_in(A) == (-u0 * cos(q1) * cos(q2) - u1 * sin(q2)) * A.x + (
        u0 * sin(q2) * cos(q1) - u1 * cos(q2)) * A.y + (
               -u0 * sin(q1) - u2) * A.z


def test_spherical_joint_coords():
    q0s, q1s, q2s, u0s, u1s, u2s = dynamicsymbols('q0:3_S, u0:3_S')
    q0, q1, q2, q3, u0, u1, u2, u4 = dynamicsymbols('q0:4, u0:4')
    # Test coordinates as list
    N, A, P, C = _generate_body()
    S = SphericalJoint('S', P, C, [q0, q1, q2], [u0, u1, u2])
    assert S.coordinates == Matrix([q0, q1, q2])
    assert S.speeds == Matrix([u0, u1, u2])
    # Test coordinates as Matrix
    N, A, P, C = _generate_body()
    S = SphericalJoint('S', P, C, Matrix([q0, q1, q2]),
                       Matrix([u0, u1, u2]))
    assert S.coordinates == Matrix([q0, q1, q2])
    assert S.speeds == Matrix([u0, u1, u2])
    # Test too few generalized coordinates
    N, A, P, C = _generate_body()
    raises(ValueError,
           lambda: SphericalJoint('S', P, C, Matrix([q0, q1]), Matrix([u0])))
    # Test too many generalized coordinates
    raises(ValueError, lambda: SphericalJoint(
        'S', P, C, Matrix([q0, q1, q2, q3]), Matrix([u0, u1, u2])))
    raises(ValueError, lambda: SphericalJoint(
        'S', P, C, Matrix([q0, q1, q2]), Matrix([u0, u1, u2, u4])))


def test_spherical_joint_orient_body():
    q0, q1, q2, u0, u1, u2 = dynamicsymbols('q0:3, u0:3')
    N_R_A = Matrix([
        [-sin(q1), -sin(q2) * cos(q1), cos(q1) * cos(q2)],
        [-sin(q0) * cos(q1), sin(q0) * sin(q1) * sin(q2) - cos(q0) * cos(q2),
         -sin(q0) * sin(q1) * cos(q2) - sin(q2) * cos(q0)],
        [cos(q0) * cos(q1), -sin(q0) * cos(q2) - sin(q1) * sin(q2) * cos(q0),
         -sin(q0) * sin(q2) + sin(q1) * cos(q0) * cos(q2)]])
    N_w_A = Matrix([[-u0 * sin(q1) - u2],
                    [-u0 * sin(q2) * cos(q1) + u1 * cos(q2)],
                    [u0 * cos(q1) * cos(q2) + u1 * sin(q2)]])
    N_v_Co = Matrix([
        [-sqrt(2) * (u0 * cos(q2 + pi / 4) * cos(q1) + u1 * sin(q2 + pi / 4))],
        [-u0 * sin(q1) - u2], [-u0 * sin(q1) - u2]])
    # Test default rot_type='BODY', rot_order=123
    N, A, P, C, Pint, Cint = _generate_body(True)
    S = SphericalJoint('S', P, C, coordinates=[q0, q1, q2], speeds=[u0, u1, u2],
                       parent_point=N.x + N.y, child_point=-A.y + A.z,
                       parent_interframe=Pint, child_interframe=Cint,
                       rot_type='body', rot_order=123)
    assert S._rot_type.upper() == 'BODY'
    assert S._rot_order == 123
    assert simplify(N.dcm(A) - N_R_A) == zeros(3)
    assert simplify(A.ang_vel_in(N).to_matrix(A) - N_w_A) == zeros(3, 1)
    assert simplify(C.masscenter.vel(N).to_matrix(A)) == N_v_Co
    # Test change of amounts
    N, A, P, C, Pint, Cint = _generate_body(True)
    S = SphericalJoint('S', P, C, coordinates=[q0, q1, q2], speeds=[u0, u1, u2],
                       parent_point=N.x + N.y, child_point=-A.y + A.z,
                       parent_interframe=Pint, child_interframe=Cint,
                       rot_type='BODY', amounts=(q1, q0, q2), rot_order=123)
    switch_order = lambda expr: expr.xreplace(
        {q0: q1, q1: q0, q2: q2, u0: u1, u1: u0, u2: u2})
    assert S._rot_type.upper() == 'BODY'
    assert S._rot_order == 123
    assert simplify(N.dcm(A) - switch_order(N_R_A)) == zeros(3)
    assert simplify(A.ang_vel_in(N).to_matrix(A) - switch_order(N_w_A)
                            ) == zeros(3, 1)
    assert simplify(C.masscenter.vel(N).to_matrix(A)) == switch_order(N_v_Co)
    # Test different rot_order
    N, A, P, C, Pint, Cint = _generate_body(True)
    S = SphericalJoint('S', P, C, coordinates=[q0, q1, q2], speeds=[u0, u1, u2],
                       parent_point=N.x + N.y, child_point=-A.y + A.z,
                       parent_interframe=Pint, child_interframe=Cint,
                       rot_type='BodY', rot_order='yxz')
    assert S._rot_type.upper() == 'BODY'
    assert S._rot_order == 'yxz'
    assert simplify(N.dcm(A) - Matrix([
        [-sin(q0) * cos(q1), sin(q0) * sin(q1) * cos(q2) - sin(q2) * cos(q0),
         sin(q0) * sin(q1) * sin(q2) + cos(q0) * cos(q2)],
        [-sin(q1), -cos(q1) * cos(q2), -sin(q2) * cos(q1)],
        [cos(q0) * cos(q1), -sin(q0) * sin(q2) - sin(q1) * cos(q0) * cos(q2),
         sin(q0) * cos(q2) - sin(q1) * sin(q2) * cos(q0)]])) == zeros(3)
    assert simplify(A.ang_vel_in(N).to_matrix(A) - Matrix([
        [u0 * sin(q1) - u2], [u0 * cos(q1) * cos(q2) - u1 * sin(q2)],
        [u0 * sin(q2) * cos(q1) + u1 * cos(q2)]])) == zeros(3, 1)
    assert simplify(C.masscenter.vel(N).to_matrix(A)) == Matrix([
        [-sqrt(2) * (u0 * sin(q2 + pi / 4) * cos(q1) + u1 * cos(q2 + pi / 4))],
        [u0 * sin(q1) - u2], [u0 * sin(q1) - u2]])


def test_spherical_joint_orient_space():
    q0, q1, q2, u0, u1, u2 = dynamicsymbols('q0:3, u0:3')
    N_R_A = Matrix([
        [-sin(q0) * sin(q2) - sin(q1) * cos(q0) * cos(q2),
         sin(q0) * sin(q1) * cos(q2) - sin(q2) * cos(q0), cos(q1) * cos(q2)],
        [-sin(q0) * cos(q2) + sin(q1) * sin(q2) * cos(q0),
         -sin(q0) * sin(q1) * sin(q2) - cos(q0) * cos(q2), -sin(q2) * cos(q1)],
        [cos(q0) * cos(q1), -sin(q0) * cos(q1), sin(q1)]])
    N_w_A = Matrix([
        [u1 * sin(q0) - u2 * cos(q0) * cos(q1)],
        [u1 * cos(q0) + u2 * sin(q0) * cos(q1)], [u0 - u2 * sin(q1)]])
    N_v_Co = Matrix([
        [u0 - u2 * sin(q1)], [u0 - u2 * sin(q1)],
        [sqrt(2) * (-u1 * sin(q0 + pi / 4) + u2 * cos(q0 + pi / 4) * cos(q1))]])
    # Test default rot_type='BODY', rot_order=123
    N, A, P, C, Pint, Cint = _generate_body(True)
    S = SphericalJoint('S', P, C, coordinates=[q0, q1, q2], speeds=[u0, u1, u2],
                       parent_point=N.x + N.z, child_point=-A.x + A.y,
                       parent_interframe=Pint, child_interframe=Cint,
                       rot_type='space', rot_order=123)
    assert S._rot_type.upper() == 'SPACE'
    assert S._rot_order == 123
    assert simplify(N.dcm(A) - N_R_A) == zeros(3)
    assert simplify(A.ang_vel_in(N).to_matrix(A)) == N_w_A
    assert simplify(C.masscenter.vel(N).to_matrix(A)) == N_v_Co
    # Test change of amounts
    switch_order = lambda expr: expr.xreplace(
        {q0: q1, q1: q0, q2: q2, u0: u1, u1: u0, u2: u2})
    N, A, P, C, Pint, Cint = _generate_body(True)
    S = SphericalJoint('S', P, C, coordinates=[q0, q1, q2], speeds=[u0, u1, u2],
                       parent_point=N.x + N.z, child_point=-A.x + A.y,
                       parent_interframe=Pint, child_interframe=Cint,
                       rot_type='SPACE', amounts=(q1, q0, q2), rot_order=123)
    assert S._rot_type.upper() == 'SPACE'
    assert S._rot_order == 123
    assert simplify(N.dcm(A) - switch_order(N_R_A)) == zeros(3)
    assert simplify(A.ang_vel_in(N).to_matrix(A)) == switch_order(N_w_A)
    assert simplify(C.masscenter.vel(N).to_matrix(A)) == switch_order(N_v_Co)
    # Test different rot_order
    N, A, P, C, Pint, Cint = _generate_body(True)
    S = SphericalJoint('S', P, C, coordinates=[q0, q1, q2], speeds=[u0, u1, u2],
                       parent_point=N.x + N.z, child_point=-A.x + A.y,
                       parent_interframe=Pint, child_interframe=Cint,
                       rot_type='SPaCe', rot_order='zxy')
    assert S._rot_type.upper() == 'SPACE'
    assert S._rot_order == 'zxy'
    assert simplify(N.dcm(A) - Matrix([
        [-sin(q2) * cos(q1), -sin(q0) * cos(q2) + sin(q1) * sin(q2) * cos(q0),
         sin(q0) * sin(q1) * sin(q2) + cos(q0) * cos(q2)],
        [-sin(q1), -cos(q0) * cos(q1), -sin(q0) * cos(q1)],
        [cos(q1) * cos(q2), -sin(q0) * sin(q2) - sin(q1) * cos(q0) * cos(q2),
         -sin(q0) * sin(q1) * cos(q2) + sin(q2) * cos(q0)]]))
    assert simplify(A.ang_vel_in(N).to_matrix(A) - Matrix([
        [-u0 + u2 * sin(q1)], [-u1 * sin(q0) + u2 * cos(q0) * cos(q1)],
        [u1 * cos(q0) + u2 * sin(q0) * cos(q1)]])) == zeros(3, 1)
    assert simplify(C.masscenter.vel(N).to_matrix(A) - Matrix([
        [u1 * cos(q0) + u2 * sin(q0) * cos(q1)],
        [u1 * cos(q0) + u2 * sin(q0) * cos(q1)],
        [u0 + u1 * sin(q0) - u2 * sin(q1) -
         u2 * cos(q0) * cos(q1)]])) == zeros(3, 1)


def test_weld_joint():
    _, _, P, C = _generate_body()
    W = WeldJoint('W', P, C)
    assert W.name == 'W'
    assert W.parent == P
    assert W.child == C
    assert W.coordinates == Matrix()
    assert W.speeds == Matrix()
    assert W.kdes == Matrix(1, 0, []).T
    assert P.frame.dcm(C.frame) == eye(3)
    assert W.child_point.pos_from(C.masscenter) == Vector(0)
    assert W.parent_point.pos_from(P.masscenter) == Vector(0)
    assert W.parent_point.pos_from(W.child_point) == Vector(0)
    assert P.masscenter.pos_from(C.masscenter) == Vector(0)
    assert C.masscenter.vel(P.frame) == Vector(0)
    assert P.frame.ang_vel_in(C.frame) == 0
    assert C.frame.ang_vel_in(P.frame) == 0
    assert W.__str__() == 'WeldJoint: W  parent: P  child: C'

    N, A, P, C = _generate_body()
    l, m = symbols('l m')
    Pint = ReferenceFrame('P_int')
    Pint.orient_axis(P.frame, P.y, pi / 2)
    W = WeldJoint('W', P, C, parent_point=l * P.frame.x,
                  child_point=m * C.frame.y, parent_interframe=Pint)

    assert W.child_point.pos_from(C.masscenter) == m * C.frame.y
    assert W.parent_point.pos_from(P.masscenter) == l * P.frame.x
    assert W.parent_point.pos_from(W.child_point) == Vector(0)
    assert P.masscenter.pos_from(C.masscenter) == - l * N.x + m * A.y
    assert C.masscenter.vel(P.frame) == Vector(0)
    assert P.masscenter.vel(Pint) == Vector(0)
    assert C.frame.ang_vel_in(P.frame) == 0
    assert P.frame.ang_vel_in(C.frame) == 0
    assert P.x == A.z

    with warns_deprecated_sympy():
        JointsMethod(P, W)  # Tests #10770


def test_deprecated_parent_child_axis():
    q, u = dynamicsymbols('q_J, u_J')
    N, A, P, C = _generate_body()
    with warns_deprecated_sympy():
        PinJoint('J', P, C, child_axis=-A.x)
    assert (-A.x).angle_between(N.x) == 0
    assert -A.x.express(N) == N.x
    assert A.dcm(N) == Matrix([[-1, 0, 0],
                               [0, -cos(q), -sin(q)],
                               [0, -sin(q), cos(q)]])
    assert A.ang_vel_in(N) == u * N.x
    assert A.ang_vel_in(N).magnitude() == sqrt(u ** 2)

    N, A, P, C = _generate_body()
    with warns_deprecated_sympy():
        PrismaticJoint('J', P, C, parent_axis=P.x + P.y)
    assert (A.x).angle_between(N.x + N.y) == 0
    assert A.x.express(N) == (N.x + N.y) / sqrt(2)
    assert A.dcm(N) == Matrix([[sqrt(2) / 2, sqrt(2) / 2, 0],
                               [-sqrt(2) / 2, sqrt(2) / 2, 0], [0, 0, 1]])
    assert A.ang_vel_in(N) == Vector(0)


def test_deprecated_joint_pos():
    N, A, P, C = _generate_body()
    with warns_deprecated_sympy():
        pin = PinJoint('J', P, C, parent_joint_pos=N.x + N.y,
                       child_joint_pos=C.y - C.z)
    assert pin.parent_point.pos_from(P.masscenter) == N.x + N.y
    assert pin.child_point.pos_from(C.masscenter) == C.y - C.z

    N, A, P, C = _generate_body()
    with warns_deprecated_sympy():
        slider = PrismaticJoint('J', P, C, parent_joint_pos=N.z + N.y,
                                child_joint_pos=C.y - C.x)
    assert slider.parent_point.pos_from(P.masscenter) == N.z + N.y
    assert slider.child_point.pos_from(C.masscenter) == C.y - C.x

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\test_grouping.py ===
"""
test where we are determining what we are grouping, or getting groups
"""
from datetime import (
    date,
    timedelta,
)

import numpy as np
import pytest

import pandas as pd
from pandas import (
    CategoricalIndex,
    DataFrame,
    Grouper,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
    period_range,
)
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping

# selection
# --------------------------------


class TestSelection:
    def test_select_bad_cols(self):
        df = DataFrame([[1, 2]], columns=["A", "B"])
        g = df.groupby("A")
        with pytest.raises(KeyError, match="\"Columns not found: 'C'\""):
            g[["C"]]

        with pytest.raises(KeyError, match="^[^A]+$"):
            # A should not be referenced as a bad column...
            # will have to rethink regex if you change message!
            g[["A", "C"]]

    def test_groupby_duplicated_column_errormsg(self):
        # GH7511
        df = DataFrame(
            columns=["A", "B", "A", "C"], data=[range(4), range(2, 6), range(0, 8, 2)]
        )

        msg = "Grouper for 'A' not 1-dimensional"
        with pytest.raises(ValueError, match=msg):
            df.groupby("A")
        with pytest.raises(ValueError, match=msg):
            df.groupby(["A", "B"])

        grouped = df.groupby("B")
        c = grouped.count()
        assert c.columns.nlevels == 1
        assert c.columns.size == 3

    def test_column_select_via_attr(self, df):
        result = df.groupby("A").C.sum()
        expected = df.groupby("A")["C"].sum()
        tm.assert_series_equal(result, expected)

        df["mean"] = 1.5
        result = df.groupby("A").mean(numeric_only=True)
        expected = df.groupby("A")[["C", "D", "mean"]].agg("mean")
        tm.assert_frame_equal(result, expected)

    def test_getitem_list_of_columns(self):
        df = DataFrame(
            {
                "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
                "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
                "C": np.random.default_rng(2).standard_normal(8),
                "D": np.random.default_rng(2).standard_normal(8),
                "E": np.random.default_rng(2).standard_normal(8),
            }
        )

        result = df.groupby("A")[["C", "D"]].mean()
        result2 = df.groupby("A")[df.columns[2:4]].mean()

        expected = df.loc[:, ["A", "C", "D"]].groupby("A").mean()

        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

    def test_getitem_numeric_column_names(self):
        # GH #13731
        df = DataFrame(
            {
                0: list("abcd") * 2,
                2: np.random.default_rng(2).standard_normal(8),
                4: np.random.default_rng(2).standard_normal(8),
                6: np.random.default_rng(2).standard_normal(8),
            }
        )
        result = df.groupby(0)[df.columns[1:3]].mean()
        result2 = df.groupby(0)[[2, 4]].mean()

        expected = df.loc[:, [0, 2, 4]].groupby(0).mean()

        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

        # per GH 23566 enforced deprecation raises a ValueError
        with pytest.raises(ValueError, match="Cannot subset columns with a tuple"):
            df.groupby(0)[2, 4].mean()

    def test_getitem_single_tuple_of_columns_raises(self, df):
        # per GH 23566 enforced deprecation raises a ValueError
        with pytest.raises(ValueError, match="Cannot subset columns with a tuple"):
            df.groupby("A")["C", "D"].mean()

    def test_getitem_single_column(self):
        df = DataFrame(
            {
                "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
                "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
                "C": np.random.default_rng(2).standard_normal(8),
                "D": np.random.default_rng(2).standard_normal(8),
                "E": np.random.default_rng(2).standard_normal(8),
            }
        )

        result = df.groupby("A")["C"].mean()

        as_frame = df.loc[:, ["A", "C"]].groupby("A").mean()
        as_series = as_frame.iloc[:, 0]
        expected = as_series

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "func", [lambda x: x.sum(), lambda x: x.agg(lambda y: y.sum())]
    )
    def test_getitem_from_grouper(self, func):
        # GH 50383
        df = DataFrame({"a": [1, 1, 2], "b": 3, "c": 4, "d": 5})
        gb = df.groupby(["a", "b"])[["a", "c"]]

        idx = MultiIndex.from_tuples([(1, 3), (2, 3)], names=["a", "b"])
        expected = DataFrame({"a": [2, 2], "c": [8, 4]}, index=idx)
        result = func(gb)

        tm.assert_frame_equal(result, expected)

    def test_indices_grouped_by_tuple_with_lambda(self):
        # GH 36158
        df = DataFrame(
            {
                "Tuples": (
                    (x, y)
                    for x in [0, 1]
                    for y in np.random.default_rng(2).integers(3, 5, 5)
                )
            }
        )

        gb = df.groupby("Tuples")
        gb_lambda = df.groupby(lambda x: df.iloc[x, 0])

        expected = gb.indices
        result = gb_lambda.indices

        tm.assert_dict_equal(result, expected)


# grouping
# --------------------------------


class TestGrouping:
    @pytest.mark.parametrize(
        "index",
        [
            Index(list("abcde")),
            Index(np.arange(5)),
            Index(np.arange(5, dtype=float)),
            date_range("2020-01-01", periods=5),
            period_range("2020-01-01", periods=5),
        ],
    )
    def test_grouper_index_types(self, index):
        # related GH5375
        # groupby misbehaving when using a Floatlike index
        df = DataFrame(np.arange(10).reshape(5, 2), columns=list("AB"), index=index)

        df.groupby(list("abcde"), group_keys=False).apply(lambda x: x)

        df.index = df.index[::-1]
        df.groupby(list("abcde"), group_keys=False).apply(lambda x: x)

    def test_grouper_multilevel_freq(self):
        # GH 7885
        # with level and freq specified in a Grouper
        d0 = date.today() - timedelta(days=14)
        dates = date_range(d0, date.today())
        date_index = MultiIndex.from_product([dates, dates], names=["foo", "bar"])
        df = DataFrame(np.random.default_rng(2).integers(0, 100, 225), index=date_index)

        # Check string level
        expected = (
            df.reset_index()
            .groupby([Grouper(key="foo", freq="W"), Grouper(key="bar", freq="W")])
            .sum()
        )
        # reset index changes columns dtype to object
        expected.columns = Index([0], dtype="int64")

        result = df.groupby(
            [Grouper(level="foo", freq="W"), Grouper(level="bar", freq="W")]
        ).sum()
        tm.assert_frame_equal(result, expected)

        # Check integer level
        result = df.groupby(
            [Grouper(level=0, freq="W"), Grouper(level=1, freq="W")]
        ).sum()
        tm.assert_frame_equal(result, expected)

    def test_grouper_creation_bug(self):
        # GH 8795
        df = DataFrame({"A": [0, 0, 1, 1, 2, 2], "B": [1, 2, 3, 4, 5, 6]})
        g = df.groupby("A")
        expected = g.sum()

        g = df.groupby(Grouper(key="A"))
        result = g.sum()
        tm.assert_frame_equal(result, expected)

        msg = "Grouper axis keyword is deprecated and will be removed"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            gpr = Grouper(key="A", axis=0)
        g = df.groupby(gpr)
        result = g.sum()
        tm.assert_frame_equal(result, expected)

        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = g.apply(lambda x: x.sum())
        expected["A"] = [0, 2, 4]
        expected = expected.loc[:, ["A", "B"]]
        tm.assert_frame_equal(result, expected)

    def test_grouper_creation_bug2(self):
        # GH14334
        # Grouper(key=...) may be passed in a list
        df = DataFrame(
            {"A": [0, 0, 0, 1, 1, 1], "B": [1, 1, 2, 2, 3, 3], "C": [1, 2, 3, 4, 5, 6]}
        )
        # Group by single column
        expected = df.groupby("A").sum()
        g = df.groupby([Grouper(key="A")])
        result = g.sum()
        tm.assert_frame_equal(result, expected)

        # Group by two columns
        # using a combination of strings and Grouper objects
        expected = df.groupby(["A", "B"]).sum()

        # Group with two Grouper objects
        g = df.groupby([Grouper(key="A"), Grouper(key="B")])
        result = g.sum()
        tm.assert_frame_equal(result, expected)

        # Group with a string and a Grouper object
        g = df.groupby(["A", Grouper(key="B")])
        result = g.sum()
        tm.assert_frame_equal(result, expected)

        # Group with a Grouper object and a string
        g = df.groupby([Grouper(key="A"), "B"])
        result = g.sum()
        tm.assert_frame_equal(result, expected)

    def test_grouper_creation_bug3(self, unit):
        # GH8866
        dti = date_range("20130101", periods=2, unit=unit)
        mi = MultiIndex.from_product(
            [list("ab"), range(2), dti],
            names=["one", "two", "three"],
        )
        ser = Series(
            np.arange(8, dtype="int64"),
            index=mi,
        )
        result = ser.groupby(Grouper(level="three", freq="ME")).sum()
        exp_dti = pd.DatetimeIndex(
            [Timestamp("2013-01-31")], freq="ME", name="three"
        ).as_unit(unit)
        expected = Series(
            [28],
            index=exp_dti,
        )
        tm.assert_series_equal(result, expected)

        # just specifying a level breaks
        result = ser.groupby(Grouper(level="one")).sum()
        expected = ser.groupby(level="one").sum()
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("func", [False, True])
    def test_grouper_returning_tuples(self, func):
        # GH 22257 , both with dict and with callable
        df = DataFrame({"X": ["A", "B", "A", "B"], "Y": [1, 4, 3, 2]})
        mapping = dict(zip(range(4), [("C", 5), ("D", 6)] * 2))

        if func:
            gb = df.groupby(by=lambda idx: mapping[idx], sort=False)
        else:
            gb = df.groupby(by=mapping, sort=False)

        name, expected = next(iter(gb))
        assert name == ("C", 5)
        result = gb.get_group(name)

        tm.assert_frame_equal(result, expected)

    def test_grouper_column_and_index(self):
        # GH 14327

        # Grouping a multi-index frame by a column and an index level should
        # be equivalent to resetting the index and grouping by two columns
        idx = MultiIndex.from_tuples(
            [("a", 1), ("a", 2), ("a", 3), ("b", 1), ("b", 2), ("b", 3)]
        )
        idx.names = ["outer", "inner"]
        df_multi = DataFrame(
            {"A": np.arange(6), "B": ["one", "one", "two", "two", "one", "one"]},
            index=idx,
        )
        result = df_multi.groupby(["B", Grouper(level="inner")]).mean(numeric_only=True)
        expected = (
            df_multi.reset_index().groupby(["B", "inner"]).mean(numeric_only=True)
        )
        tm.assert_frame_equal(result, expected)

        # Test the reverse grouping order
        result = df_multi.groupby([Grouper(level="inner"), "B"]).mean(numeric_only=True)
        expected = (
            df_multi.reset_index().groupby(["inner", "B"]).mean(numeric_only=True)
        )
        tm.assert_frame_equal(result, expected)

        # Grouping a single-index frame by a column and the index should
        # be equivalent to resetting the index and grouping by two columns
        df_single = df_multi.reset_index("outer")
        result = df_single.groupby(["B", Grouper(level="inner")]).mean(
            numeric_only=True
        )
        expected = (
            df_single.reset_index().groupby(["B", "inner"]).mean(numeric_only=True)
        )
        tm.assert_frame_equal(result, expected)

        # Test the reverse grouping order
        result = df_single.groupby([Grouper(level="inner"), "B"]).mean(
            numeric_only=True
        )
        expected = (
            df_single.reset_index().groupby(["inner", "B"]).mean(numeric_only=True)
        )
        tm.assert_frame_equal(result, expected)

    def test_groupby_levels_and_columns(self):
        # GH9344, GH9049
        idx_names = ["x", "y"]
        idx = MultiIndex.from_tuples([(1, 1), (1, 2), (3, 4), (5, 6)], names=idx_names)
        df = DataFrame(np.arange(12).reshape(-1, 3), index=idx)

        by_levels = df.groupby(level=idx_names).mean()
        # reset_index changes columns dtype to object
        by_columns = df.reset_index().groupby(idx_names).mean()

        # without casting, by_columns.columns is object-dtype
        by_columns.columns = by_columns.columns.astype(np.int64)
        tm.assert_frame_equal(by_levels, by_columns)

    def test_groupby_categorical_index_and_columns(self, observed):
        # GH18432, adapted for GH25871
        columns = ["A", "B", "A", "B"]
        categories = ["B", "A"]
        data = np.array(
            [[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 1, 2]], int
        )
        cat_columns = CategoricalIndex(columns, categories=categories, ordered=True)
        df = DataFrame(data=data, columns=cat_columns)
        depr_msg = "DataFrame.groupby with axis=1 is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            result = df.groupby(axis=1, level=0, observed=observed).sum()
        expected_data = np.array([[4, 2], [4, 2], [4, 2], [4, 2], [4, 2]], int)
        expected_columns = CategoricalIndex(
            categories, categories=categories, ordered=True
        )
        expected = DataFrame(data=expected_data, columns=expected_columns)
        tm.assert_frame_equal(result, expected)

        # test transposed version
        df = DataFrame(data.T, index=cat_columns)
        msg = "The 'axis' keyword in DataFrame.groupby is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.groupby(axis=0, level=0, observed=observed).sum()
        expected = DataFrame(data=expected_data.T, index=expected_columns)
        tm.assert_frame_equal(result, expected)

    def test_grouper_getting_correct_binner(self):
        # GH 10063
        # using a non-time-based grouper and a time-based grouper
        # and specifying levels
        df = DataFrame(
            {"A": 1},
            index=MultiIndex.from_product(
                [list("ab"), date_range("20130101", periods=80)], names=["one", "two"]
            ),
        )
        result = df.groupby(
            [Grouper(level="one"), Grouper(level="two", freq="ME")]
        ).sum()
        expected = DataFrame(
            {"A": [31, 28, 21, 31, 28, 21]},
            index=MultiIndex.from_product(
                [list("ab"), date_range("20130101", freq="ME", periods=3)],
                names=["one", "two"],
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_grouper_iter(self, df):
        gb = df.groupby("A")
        msg = "DataFrameGroupBy.grouper is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            grouper = gb.grouper
        result = sorted(grouper)
        expected = ["bar", "foo"]
        assert result == expected

    def test_empty_groups(self, df):
        # see gh-1048
        with pytest.raises(ValueError, match="No group keys passed!"):
            df.groupby([])

    def test_groupby_grouper(self, df):
        grouped = df.groupby("A")
        msg = "DataFrameGroupBy.grouper is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            grouper = grouped.grouper
        result = df.groupby(grouper).mean(numeric_only=True)
        expected = grouped.mean(numeric_only=True)
        tm.assert_frame_equal(result, expected)

    def test_groupby_dict_mapping(self):
        # GH #679
        s = Series({"T1": 5})
        result = s.groupby({"T1": "T2"}).agg("sum")
        expected = s.groupby(["T2"]).agg("sum")
        tm.assert_series_equal(result, expected)

        s = Series([1.0, 2.0, 3.0, 4.0], index=list("abcd"))
        mapping = {"a": 0, "b": 0, "c": 1, "d": 1}

        result = s.groupby(mapping).mean()
        result2 = s.groupby(mapping).agg("mean")
        exp_key = np.array([0, 0, 1, 1], dtype=np.int64)
        expected = s.groupby(exp_key).mean()
        expected2 = s.groupby(exp_key).mean()
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result, result2)
        tm.assert_series_equal(result, expected2)

    @pytest.mark.parametrize(
        "index",
        [
            [0, 1, 2, 3],
            ["a", "b", "c", "d"],
            [Timestamp(2021, 7, 28 + i) for i in range(4)],
        ],
    )
    def test_groupby_series_named_with_tuple(self, frame_or_series, index):
        # GH 42731
        obj = frame_or_series([1, 2, 3, 4], index=index)
        groups = Series([1, 0, 1, 0], index=index, name=("a", "a"))
        result = obj.groupby(groups).last()
        expected = frame_or_series([4, 3])
        expected.index.name = ("a", "a")
        tm.assert_equal(result, expected)

    def test_groupby_grouper_f_sanity_checked(self):
        dates = date_range("01-Jan-2013", periods=12, freq="MS")
        ts = Series(np.random.default_rng(2).standard_normal(12), index=dates)

        # GH51979
        # simple check that the passed function doesn't operates on the whole index
        msg = "'Timestamp' object is not subscriptable"
        with pytest.raises(TypeError, match=msg):
            ts.groupby(lambda key: key[0:6])

        result = ts.groupby(lambda x: x).sum()
        expected = ts.groupby(ts.index).sum()
        expected.index.freq = None
        tm.assert_series_equal(result, expected)

    def test_groupby_with_datetime_key(self):
        # GH 51158
        df = DataFrame(
            {
                "id": ["a", "b"] * 3,
                "b": date_range("2000-01-01", "2000-01-03", freq="9h"),
            }
        )
        grouper = Grouper(key="b", freq="D")
        gb = df.groupby([grouper, "id"])

        # test number of groups
        expected = {
            (Timestamp("2000-01-01"), "a"): [0, 2],
            (Timestamp("2000-01-01"), "b"): [1],
            (Timestamp("2000-01-02"), "a"): [4],
            (Timestamp("2000-01-02"), "b"): [3, 5],
        }
        tm.assert_dict_equal(gb.groups, expected)

        # test number of group keys
        assert len(gb.groups.keys()) == 4

    def test_grouping_error_on_multidim_input(self, df):
        msg = "Grouper for '<class 'pandas.core.frame.DataFrame'>' not 1-dimensional"
        with pytest.raises(ValueError, match=msg):
            Grouping(df.index, df[["A", "A"]])

    def test_multiindex_passthru(self):
        # GH 7997
        # regression from 0.14.1
        df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        df.columns = MultiIndex.from_tuples([(0, 1), (1, 1), (2, 1)])

        depr_msg = "DataFrame.groupby with axis=1 is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            gb = df.groupby(axis=1, level=[0, 1])
        result = gb.first()
        tm.assert_frame_equal(result, df)

    def test_multiindex_negative_level(self, multiindex_dataframe_random_data):
        # GH 13901
        result = multiindex_dataframe_random_data.groupby(level=-1).sum()
        expected = multiindex_dataframe_random_data.groupby(level="second").sum()
        tm.assert_frame_equal(result, expected)

        result = multiindex_dataframe_random_data.groupby(level=-2).sum()
        expected = multiindex_dataframe_random_data.groupby(level="first").sum()
        tm.assert_frame_equal(result, expected)

        result = multiindex_dataframe_random_data.groupby(level=[-2, -1]).sum()
        expected = multiindex_dataframe_random_data.sort_index()
        tm.assert_frame_equal(result, expected)

        result = multiindex_dataframe_random_data.groupby(level=[-1, "first"]).sum()
        expected = multiindex_dataframe_random_data.groupby(
            level=["second", "first"]
        ).sum()
        tm.assert_frame_equal(result, expected)

    def test_multifunc_select_col_integer_cols(self, df):
        df.columns = np.arange(len(df.columns))

        # it works!
        msg = "Passing a dictionary to SeriesGroupBy.agg is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            df.groupby(1, as_index=False)[2].agg({"Q": np.mean})

    def test_multiindex_columns_empty_level(self):
        lst = [["count", "values"], ["to filter", ""]]
        midx = MultiIndex.from_tuples(lst)

        df = DataFrame([[1, "A"]], columns=midx)

        grouped = df.groupby("to filter").groups
        assert grouped["A"] == [0]

        grouped = df.groupby([("to filter", "")]).groups
        assert grouped["A"] == [0]

        df = DataFrame([[1, "A"], [2, "B"]], columns=midx)

        expected = df.groupby("to filter").groups
        result = df.groupby([("to filter", "")]).groups
        assert result == expected

        df = DataFrame([[1, "A"], [2, "A"]], columns=midx)

        expected = df.groupby("to filter").groups
        result = df.groupby([("to filter", "")]).groups
        tm.assert_dict_equal(result, expected)

    def test_groupby_multiindex_tuple(self):
        # GH 17979
        df = DataFrame(
            [[1, 2, 3, 4], [3, 4, 5, 6], [1, 4, 2, 3]],
            columns=MultiIndex.from_arrays([["a", "b", "b", "c"], [1, 1, 2, 2]]),
        )
        expected = df.groupby([("b", 1)]).groups
        result = df.groupby(("b", 1)).groups
        tm.assert_dict_equal(expected, result)

        df2 = DataFrame(
            df.values,
            columns=MultiIndex.from_arrays(
                [["a", "b", "b", "c"], ["d", "d", "e", "e"]]
            ),
        )
        expected = df2.groupby([("b", "d")]).groups
        result = df.groupby(("b", 1)).groups
        tm.assert_dict_equal(expected, result)

        df3 = DataFrame(df.values, columns=[("a", "d"), ("b", "d"), ("b", "e"), "c"])
        expected = df3.groupby([("b", "d")]).groups
        result = df.groupby(("b", 1)).groups
        tm.assert_dict_equal(expected, result)

    def test_groupby_multiindex_partial_indexing_equivalence(self):
        # GH 17977
        df = DataFrame(
            [[1, 2, 3, 4], [3, 4, 5, 6], [1, 4, 2, 3]],
            columns=MultiIndex.from_arrays([["a", "b", "b", "c"], [1, 1, 2, 2]]),
        )

        expected_mean = df.groupby([("a", 1)])[[("b", 1), ("b", 2)]].mean()
        result_mean = df.groupby([("a", 1)])["b"].mean()
        tm.assert_frame_equal(expected_mean, result_mean)

        expected_sum = df.groupby([("a", 1)])[[("b", 1), ("b", 2)]].sum()
        result_sum = df.groupby([("a", 1)])["b"].sum()
        tm.assert_frame_equal(expected_sum, result_sum)

        expected_count = df.groupby([("a", 1)])[[("b", 1), ("b", 2)]].count()
        result_count = df.groupby([("a", 1)])["b"].count()
        tm.assert_frame_equal(expected_count, result_count)

        expected_min = df.groupby([("a", 1)])[[("b", 1), ("b", 2)]].min()
        result_min = df.groupby([("a", 1)])["b"].min()
        tm.assert_frame_equal(expected_min, result_min)

        expected_max = df.groupby([("a", 1)])[[("b", 1), ("b", 2)]].max()
        result_max = df.groupby([("a", 1)])["b"].max()
        tm.assert_frame_equal(expected_max, result_max)

        expected_groups = df.groupby([("a", 1)])[[("b", 1), ("b", 2)]].groups
        result_groups = df.groupby([("a", 1)])["b"].groups
        tm.assert_dict_equal(expected_groups, result_groups)

    @pytest.mark.parametrize("sort", [True, False])
    def test_groupby_level(self, sort, multiindex_dataframe_random_data, df):
        # GH 17537
        frame = multiindex_dataframe_random_data
        deleveled = frame.reset_index()

        result0 = frame.groupby(level=0, sort=sort).sum()
        result1 = frame.groupby(level=1, sort=sort).sum()

        expected0 = frame.groupby(deleveled["first"].values, sort=sort).sum()
        expected1 = frame.groupby(deleveled["second"].values, sort=sort).sum()

        expected0.index.name = "first"
        expected1.index.name = "second"

        assert result0.index.name == "first"
        assert result1.index.name == "second"

        tm.assert_frame_equal(result0, expected0)
        tm.assert_frame_equal(result1, expected1)
        assert result0.index.name == frame.index.names[0]
        assert result1.index.name == frame.index.names[1]

        # groupby level name
        result0 = frame.groupby(level="first", sort=sort).sum()
        result1 = frame.groupby(level="second", sort=sort).sum()
        tm.assert_frame_equal(result0, expected0)
        tm.assert_frame_equal(result1, expected1)

        # axis=1
        msg = "DataFrame.groupby with axis=1 is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result0 = frame.T.groupby(level=0, axis=1, sort=sort).sum()
            result1 = frame.T.groupby(level=1, axis=1, sort=sort).sum()
        tm.assert_frame_equal(result0, expected0.T)
        tm.assert_frame_equal(result1, expected1.T)

        # raise exception for non-MultiIndex
        msg = "level > 0 or level < -1 only valid with MultiIndex"
        with pytest.raises(ValueError, match=msg):
            df.groupby(level=1)

    def test_groupby_level_index_names(self, axis):
        # GH4014 this used to raise ValueError since 'exp'>1 (in py2)
        df = DataFrame({"exp": ["A"] * 3 + ["B"] * 3, "var1": range(6)}).set_index(
            "exp"
        )
        if axis in (1, "columns"):
            df = df.T
            depr_msg = "DataFrame.groupby with axis=1 is deprecated"
        else:
            depr_msg = "The 'axis' keyword in DataFrame.groupby is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            df.groupby(level="exp", axis=axis)
        msg = f"level name foo is not the name of the {df._get_axis_name(axis)}"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg):
                df.groupby(level="foo", axis=axis)

    @pytest.mark.parametrize("sort", [True, False])
    def test_groupby_level_with_nas(self, sort):
        # GH 17537
        index = MultiIndex(
            levels=[[1, 0], [0, 1, 2, 3]],
            codes=[[1, 1, 1, 1, 0, 0, 0, 0], [0, 1, 2, 3, 0, 1, 2, 3]],
        )

        # factorizing doesn't confuse things
        s = Series(np.arange(8.0), index=index)
        result = s.groupby(level=0, sort=sort).sum()
        expected = Series([6.0, 22.0], index=[0, 1])
        tm.assert_series_equal(result, expected)

        index = MultiIndex(
            levels=[[1, 0], [0, 1, 2, 3]],
            codes=[[1, 1, 1, 1, -1, 0, 0, 0], [0, 1, 2, 3, 0, 1, 2, 3]],
        )

        # factorizing doesn't confuse things
        s = Series(np.arange(8.0), index=index)
        result = s.groupby(level=0, sort=sort).sum()
        expected = Series([6.0, 18.0], index=[0.0, 1.0])
        tm.assert_series_equal(result, expected)

    def test_groupby_args(self, multiindex_dataframe_random_data):
        # PR8618 and issue 8015
        frame = multiindex_dataframe_random_data

        msg = "You have to supply one of 'by' and 'level'"
        with pytest.raises(TypeError, match=msg):
            frame.groupby()

        msg = "You have to supply one of 'by' and 'level'"
        with pytest.raises(TypeError, match=msg):
            frame.groupby(by=None, level=None)

    @pytest.mark.parametrize(
        "sort,labels",
        [
            [True, [2, 2, 2, 0, 0, 1, 1, 3, 3, 3]],
            [False, [0, 0, 0, 1, 1, 2, 2, 3, 3, 3]],
        ],
    )
    def test_level_preserve_order(self, sort, labels, multiindex_dataframe_random_data):
        # GH 17537
        grouped = multiindex_dataframe_random_data.groupby(level=0, sort=sort)
        exp_labels = np.array(labels, np.intp)
        tm.assert_almost_equal(grouped._grouper.codes[0], exp_labels)

    def test_grouping_labels(self, multiindex_dataframe_random_data):
        grouped = multiindex_dataframe_random_data.groupby(
            multiindex_dataframe_random_data.index.get_level_values(0)
        )
        exp_labels = np.array([2, 2, 2, 0, 0, 1, 1, 3, 3, 3], dtype=np.intp)
        tm.assert_almost_equal(grouped._grouper.codes[0], exp_labels)

    def test_list_grouper_with_nat(self):
        # GH 14715
        df = DataFrame({"date": date_range("1/1/2011", periods=365, freq="D")})
        df.iloc[-1] = pd.NaT
        grouper = Grouper(key="date", freq="YS")

        # Grouper in a list grouping
        result = df.groupby([grouper])
        expected = {Timestamp("2011-01-01"): Index(list(range(364)))}
        tm.assert_dict_equal(result.groups, expected)

        # Test case without a list
        result = df.groupby(grouper)
        expected = {Timestamp("2011-01-01"): 365}
        tm.assert_dict_equal(result.groups, expected)

    @pytest.mark.parametrize(
        "func,expected",
        [
            (
                "transform",
                Series(name=2, dtype=np.float64),
            ),
            (
                "agg",
                Series(
                    name=2, dtype=np.float64, index=Index([], dtype=np.float64, name=1)
                ),
            ),
            (
                "apply",
                Series(
                    name=2, dtype=np.float64, index=Index([], dtype=np.float64, name=1)
                ),
            ),
        ],
    )
    def test_evaluate_with_empty_groups(self, func, expected):
        # 26208
        # test transform'ing empty groups
        # (not testing other agg fns, because they return
        # different index objects.
        df = DataFrame({1: [], 2: []})
        g = df.groupby(1, group_keys=False)
        result = getattr(g[2], func)(lambda x: x)
        tm.assert_series_equal(result, expected)

    def test_groupby_empty(self):
        # https://github.com/pandas-dev/pandas/issues/27190
        s = Series([], name="name", dtype="float64")
        gr = s.groupby([])

        result = gr.mean()
        expected = s.set_axis(Index([], dtype=np.intp))
        tm.assert_series_equal(result, expected)

        # check group properties
        assert len(gr._grouper.groupings) == 1
        tm.assert_numpy_array_equal(
            gr._grouper.group_info[0], np.array([], dtype=np.dtype(np.intp))
        )

        tm.assert_numpy_array_equal(
            gr._grouper.group_info[1], np.array([], dtype=np.dtype(np.intp))
        )

        assert gr._grouper.group_info[2] == 0

        # check name
        gb = s.groupby(s)
        msg = "SeriesGroupBy.grouper is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            grouper = gb.grouper
        result = grouper.names
        expected = ["name"]
        assert result == expected

    def test_groupby_level_index_value_all_na(self):
        # issue 20519
        df = DataFrame(
            [["x", np.nan, 10], [None, np.nan, 20]], columns=["A", "B", "C"]
        ).set_index(["A", "B"])
        result = df.groupby(level=["A", "B"]).sum()
        expected = DataFrame(
            data=[],
            index=MultiIndex(
                levels=[Index(["x"], dtype="str"), Index([], dtype="float64")],
                codes=[[], []],
                names=["A", "B"],
            ),
            columns=["C"],
            dtype="int64",
        )
        tm.assert_frame_equal(result, expected)

    def test_groupby_multiindex_level_empty(self):
        # https://github.com/pandas-dev/pandas/issues/31670
        df = DataFrame(
            [[123, "a", 1.0], [123, "b", 2.0]], columns=["id", "category", "value"]
        )
        df = df.set_index(["id", "category"])
        empty = df[df.value < 0]
        result = empty.groupby("id").sum()
        expected = DataFrame(
            dtype="float64",
            columns=["value"],
            index=Index([], dtype=np.int64, name="id"),
        )
        tm.assert_frame_equal(result, expected)


# get_group
# --------------------------------


class TestGetGroup:
    def test_get_group(self):
        # GH 5267
        # be datelike friendly
        df = DataFrame(
            {
                "DATE": pd.to_datetime(
                    [
                        "10-Oct-2013",
                        "10-Oct-2013",
                        "10-Oct-2013",
                        "11-Oct-2013",
                        "11-Oct-2013",
                        "11-Oct-2013",
                    ]
                ),
                "label": ["foo", "foo", "bar", "foo", "foo", "bar"],
                "VAL": [1, 2, 3, 4, 5, 6],
            }
        )

        g = df.groupby("DATE")
        key = next(iter(g.groups))
        result1 = g.get_group(key)
        result2 = g.get_group(Timestamp(key).to_pydatetime())
        result3 = g.get_group(str(Timestamp(key)))
        tm.assert_frame_equal(result1, result2)
        tm.assert_frame_equal(result1, result3)

        g = df.groupby(["DATE", "label"])

        key = next(iter(g.groups))
        result1 = g.get_group(key)
        result2 = g.get_group((Timestamp(key[0]).to_pydatetime(), key[1]))
        result3 = g.get_group((str(Timestamp(key[0])), key[1]))
        tm.assert_frame_equal(result1, result2)
        tm.assert_frame_equal(result1, result3)

        # must pass a same-length tuple with multiple keys
        msg = "must supply a tuple to get_group with multiple grouping keys"
        with pytest.raises(ValueError, match=msg):
            g.get_group("foo")
        with pytest.raises(ValueError, match=msg):
            g.get_group("foo")
        msg = "must supply a same-length tuple to get_group with multiple grouping keys"
        with pytest.raises(ValueError, match=msg):
            g.get_group(("foo", "bar", "baz"))

    def test_get_group_empty_bins(self, observed):
        d = DataFrame([3, 1, 7, 6])
        bins = [0, 5, 10, 15]
        g = d.groupby(pd.cut(d[0], bins), observed=observed)

        # TODO: should prob allow a str of Interval work as well
        # IOW '(0, 5]'
        result = g.get_group(pd.Interval(0, 5))
        expected = DataFrame([3, 1], index=[0, 1])
        tm.assert_frame_equal(result, expected)

        msg = r"Interval\(10, 15, closed='right'\)"
        with pytest.raises(KeyError, match=msg):
            g.get_group(pd.Interval(10, 15))

    def test_get_group_grouped_by_tuple(self):
        # GH 8121
        df = DataFrame([[(1,), (1, 2), (1,), (1, 2)]], index=["ids"]).T
        gr = df.groupby("ids")
        expected = DataFrame({"ids": [(1,), (1,)]}, index=[0, 2])
        result = gr.get_group((1,))
        tm.assert_frame_equal(result, expected)

        dt = pd.to_datetime(["2010-01-01", "2010-01-02", "2010-01-01", "2010-01-02"])
        df = DataFrame({"ids": [(x,) for x in dt]})
        gr = df.groupby("ids")
        result = gr.get_group(("2010-01-01",))
        expected = DataFrame({"ids": [(dt[0],), (dt[0],)]}, index=[0, 2])
        tm.assert_frame_equal(result, expected)

    def test_get_group_grouped_by_tuple_with_lambda(self):
        # GH 36158
        df = DataFrame(
            {
                "Tuples": (
                    (x, y)
                    for x in [0, 1]
                    for y in np.random.default_rng(2).integers(3, 5, 5)
                )
            }
        )

        gb = df.groupby("Tuples")
        gb_lambda = df.groupby(lambda x: df.iloc[x, 0])

        expected = gb.get_group(next(iter(gb.groups.keys())))
        result = gb_lambda.get_group(next(iter(gb_lambda.groups.keys())))

        tm.assert_frame_equal(result, expected)

    def test_groupby_with_empty(self):
        index = pd.DatetimeIndex(())
        data = ()
        series = Series(data, index, dtype=object)
        grouper = Grouper(freq="D")
        grouped = series.groupby(grouper)
        assert next(iter(grouped), None) is None

    def test_groupby_with_single_column(self):
        df = DataFrame({"a": list("abssbab")})
        tm.assert_frame_equal(df.groupby("a").get_group("a"), df.iloc[[0, 5]])
        # GH 13530
        exp = DataFrame(
            index=Index(["a", "b", "s"], name="a"), columns=Index([], dtype="str")
        )
        tm.assert_frame_equal(df.groupby("a").count(), exp)
        tm.assert_frame_equal(df.groupby("a").sum(), exp)

        exp = df.iloc[[3, 4, 5]]
        tm.assert_frame_equal(df.groupby("a").nth(1), exp)

    def test_gb_key_len_equal_axis_len(self):
        # GH16843
        # test ensures that index and column keys are recognized correctly
        # when number of keys equals axis length of groupby
        df = DataFrame(
            [["foo", "bar", "B", 1], ["foo", "bar", "B", 2], ["foo", "baz", "C", 3]],
            columns=["first", "second", "third", "one"],
        )
        df = df.set_index(["first", "second"])
        df = df.groupby(["first", "second", "third"]).size()
        assert df.loc[("foo", "bar", "B")] == 2
        assert df.loc[("foo", "baz", "C")] == 1


# groups & iteration
# --------------------------------


class TestIteration:
    def test_groups(self, df):
        grouped = df.groupby(["A"])
        groups = grouped.groups
        assert groups is grouped.groups  # caching works

        for k, v in grouped.groups.items():
            assert (df.loc[v]["A"] == k).all()

        grouped = df.groupby(["A", "B"])
        groups = grouped.groups
        assert groups is grouped.groups  # caching works

        for k, v in grouped.groups.items():
            assert (df.loc[v]["A"] == k[0]).all()
            assert (df.loc[v]["B"] == k[1]).all()

    def test_grouping_is_iterable(self, tsframe):
        # this code path isn't used anywhere else
        # not sure it's useful
        grouped = tsframe.groupby([lambda x: x.weekday(), lambda x: x.year])

        # test it works
        for g in grouped._grouper.groupings[0]:
            pass

    def test_multi_iter(self):
        s = Series(np.arange(6))
        k1 = np.array(["a", "a", "a", "b", "b", "b"])
        k2 = np.array(["1", "2", "1", "2", "1", "2"])

        grouped = s.groupby([k1, k2])

        iterated = list(grouped)
        expected = [
            ("a", "1", s[[0, 2]]),
            ("a", "2", s[[1]]),
            ("b", "1", s[[4]]),
            ("b", "2", s[[3, 5]]),
        ]
        for i, ((one, two), three) in enumerate(iterated):
            e1, e2, e3 = expected[i]
            assert e1 == one
            assert e2 == two
            tm.assert_series_equal(three, e3)

    def test_multi_iter_frame(self, three_group):
        k1 = np.array(["b", "b", "b", "a", "a", "a"])
        k2 = np.array(["1", "2", "1", "2", "1", "2"])
        df = DataFrame(
            {
                "v1": np.random.default_rng(2).standard_normal(6),
                "v2": np.random.default_rng(2).standard_normal(6),
                "k1": k1,
                "k2": k2,
            },
            index=["one", "two", "three", "four", "five", "six"],
        )

        grouped = df.groupby(["k1", "k2"])

        # things get sorted!
        iterated = list(grouped)
        idx = df.index
        expected = [
            ("a", "1", df.loc[idx[[4]]]),
            ("a", "2", df.loc[idx[[3, 5]]]),
            ("b", "1", df.loc[idx[[0, 2]]]),
            ("b", "2", df.loc[idx[[1]]]),
        ]
        for i, ((one, two), three) in enumerate(iterated):
            e1, e2, e3 = expected[i]
            assert e1 == one
            assert e2 == two
            tm.assert_frame_equal(three, e3)

        # don't iterate through groups with no data
        df["k1"] = np.array(["b", "b", "b", "a", "a", "a"])
        df["k2"] = np.array(["1", "1", "1", "2", "2", "2"])
        grouped = df.groupby(["k1", "k2"])
        # calling `dict` on a DataFrameGroupBy leads to a TypeError,
        # we need to use a dictionary comprehension here
        # pylint: disable-next=unnecessary-comprehension
        groups = {key: gp for key, gp in grouped}  # noqa: C416
        assert len(groups) == 2

        # axis = 1
        three_levels = three_group.groupby(["A", "B", "C"]).mean()
        depr_msg = "DataFrame.groupby with axis=1 is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            grouped = three_levels.T.groupby(axis=1, level=(1, 2))
        for key, group in grouped:
            pass

    def test_dictify(self, df):
        dict(iter(df.groupby("A")))
        dict(iter(df.groupby(["A", "B"])))
        dict(iter(df["C"].groupby(df["A"])))
        dict(iter(df["C"].groupby([df["A"], df["B"]])))
        dict(iter(df.groupby("A")["C"]))
        dict(iter(df.groupby(["A", "B"])["C"]))

    def test_groupby_with_small_elem(self):
        # GH 8542
        # length=2
        df = DataFrame(
            {"event": ["start", "start"], "change": [1234, 5678]},
            index=pd.DatetimeIndex(["2014-09-10", "2013-10-10"]),
        )
        grouped = df.groupby([Grouper(freq="ME"), "event"])
        assert len(grouped.groups) == 2
        assert grouped.ngroups == 2
        assert (Timestamp("2014-09-30"), "start") in grouped.groups
        assert (Timestamp("2013-10-31"), "start") in grouped.groups

        res = grouped.get_group((Timestamp("2014-09-30"), "start"))
        tm.assert_frame_equal(res, df.iloc[[0], :])
        res = grouped.get_group((Timestamp("2013-10-31"), "start"))
        tm.assert_frame_equal(res, df.iloc[[1], :])

        df = DataFrame(
            {"event": ["start", "start", "start"], "change": [1234, 5678, 9123]},
            index=pd.DatetimeIndex(["2014-09-10", "2013-10-10", "2014-09-15"]),
        )
        grouped = df.groupby([Grouper(freq="ME"), "event"])
        assert len(grouped.groups) == 2
        assert grouped.ngroups == 2
        assert (Timestamp("2014-09-30"), "start") in grouped.groups
        assert (Timestamp("2013-10-31"), "start") in grouped.groups

        res = grouped.get_group((Timestamp("2014-09-30"), "start"))
        tm.assert_frame_equal(res, df.iloc[[0, 2], :])
        res = grouped.get_group((Timestamp("2013-10-31"), "start"))
        tm.assert_frame_equal(res, df.iloc[[1], :])

        # length=3
        df = DataFrame(
            {"event": ["start", "start", "start"], "change": [1234, 5678, 9123]},
            index=pd.DatetimeIndex(["2014-09-10", "2013-10-10", "2014-08-05"]),
        )
        grouped = df.groupby([Grouper(freq="ME"), "event"])
        assert len(grouped.groups) == 3
        assert grouped.ngroups == 3
        assert (Timestamp("2014-09-30"), "start") in grouped.groups
        assert (Timestamp("2013-10-31"), "start") in grouped.groups
        assert (Timestamp("2014-08-31"), "start") in grouped.groups

        res = grouped.get_group((Timestamp("2014-09-30"), "start"))
        tm.assert_frame_equal(res, df.iloc[[0], :])
        res = grouped.get_group((Timestamp("2013-10-31"), "start"))
        tm.assert_frame_equal(res, df.iloc[[1], :])
        res = grouped.get_group((Timestamp("2014-08-31"), "start"))
        tm.assert_frame_equal(res, df.iloc[[2], :])

    def test_grouping_string_repr(self):
        # GH 13394
        mi = MultiIndex.from_arrays([list("AAB"), list("aba")])
        df = DataFrame([[1, 2, 3]], columns=mi)
        gr = df.groupby(df[("A", "a")])

        result = gr._grouper.groupings[0].__repr__()
        expected = "Grouping(('A', 'a'))"
        assert result == expected


def test_grouping_by_key_is_in_axis():
    # GH#50413 - Groupers specified by key are in-axis
    df = DataFrame({"a": [1, 1, 2], "b": [1, 1, 2], "c": [3, 4, 5]}).set_index("a")
    gb = df.groupby([Grouper(level="a"), Grouper(key="b")], as_index=False)
    assert not gb._grouper.groupings[0].in_axis
    assert gb._grouper.groupings[1].in_axis

    # Currently only in-axis groupings are including in the result when as_index=False;
    # This is likely to change in the future.
    msg = "A grouping .* was excluded from the result"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = gb.sum()
    expected = DataFrame({"b": [1, 2], "c": [7, 5]})
    tm.assert_frame_equal(result, expected)


def test_grouper_groups():
    # GH#51182 check Grouper.groups does not raise AttributeError
    df = DataFrame({"a": [1, 2, 3], "b": 1})
    grper = Grouper(key="a")
    gb = df.groupby(grper)

    msg = "Use GroupBy.groups instead"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = grper.groups
    assert res is gb.groups

    msg = "Use GroupBy.grouper instead"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = grper.grouper
    assert res is gb._grouper

    msg = "Grouper.obj is deprecated and will be removed"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = grper.obj
    assert res is gb.obj

    msg = "Use Resampler.ax instead"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        grper.ax

    msg = "Grouper.indexer is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        grper.indexer


@pytest.mark.parametrize("attr", ["group_index", "result_index", "group_arraylike"])
def test_depr_grouping_attrs(attr):
    # GH#56148
    df = DataFrame({"a": [1, 1, 2], "b": [3, 4, 5]})
    gb = df.groupby("a")
    msg = f"{attr} is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        getattr(gb._grouper.groupings[0], attr)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\dtypes\test_dtypes.py ===
import re
import weakref

import numpy as np
import pytest
import pytz

from pandas._libs.tslibs.dtypes import NpyDatetimeUnit

from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_datetime64_dtype,
    is_datetime64_ns_dtype,
    is_datetime64tz_dtype,
    is_dtype_equal,
    is_interval_dtype,
    is_period_dtype,
    is_string_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    IntervalDtype,
    PeriodDtype,
)

import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DatetimeIndex,
    IntervalIndex,
    Series,
    SparseDtype,
    date_range,
)
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray


class Base:
    def test_hash(self, dtype):
        hash(dtype)

    def test_equality_invalid(self, dtype):
        assert not dtype == "foo"
        assert not is_dtype_equal(dtype, np.int64)

    def test_numpy_informed(self, dtype):
        # npdev 2020-02-02 changed from "data type not understood" to
        #  "Cannot interpret 'foo' as a data type"
        msg = "|".join(
            ["data type not understood", "Cannot interpret '.*' as a data type"]
        )
        with pytest.raises(TypeError, match=msg):
            np.dtype(dtype)

        assert not dtype == np.str_
        assert not np.str_ == dtype

    def test_pickle(self, dtype):
        # make sure our cache is NOT pickled

        # clear the cache
        type(dtype).reset_cache()
        assert not len(dtype._cache_dtypes)

        # force back to the cache
        result = tm.round_trip_pickle(dtype)
        if not isinstance(dtype, PeriodDtype):
            # Because PeriodDtype has a cython class as a base class,
            #  it has different pickle semantics, and its cache is re-populated
            #  on un-pickling.
            assert not len(dtype._cache_dtypes)
        assert result == dtype


class TestCategoricalDtype(Base):
    @pytest.fixture
    def dtype(self):
        """
        Class level fixture of dtype for TestCategoricalDtype
        """
        return CategoricalDtype()

    def test_hash_vs_equality(self, dtype):
        dtype2 = CategoricalDtype()
        assert dtype == dtype2
        assert dtype2 == dtype
        assert hash(dtype) == hash(dtype2)

    def test_equality(self, dtype):
        assert dtype == "category"
        assert is_dtype_equal(dtype, "category")
        assert "category" == dtype
        assert is_dtype_equal("category", dtype)

        assert dtype == CategoricalDtype()
        assert is_dtype_equal(dtype, CategoricalDtype())
        assert CategoricalDtype() == dtype
        assert is_dtype_equal(CategoricalDtype(), dtype)

        assert dtype != "foo"
        assert not is_dtype_equal(dtype, "foo")
        assert "foo" != dtype
        assert not is_dtype_equal("foo", dtype)

    def test_construction_from_string(self, dtype):
        result = CategoricalDtype.construct_from_string("category")
        assert is_dtype_equal(dtype, result)
        msg = "Cannot construct a 'CategoricalDtype' from 'foo'"
        with pytest.raises(TypeError, match=msg):
            CategoricalDtype.construct_from_string("foo")

    def test_constructor_invalid(self):
        msg = "Parameter 'categories' must be list-like"
        with pytest.raises(TypeError, match=msg):
            CategoricalDtype("category")

    dtype1 = CategoricalDtype(["a", "b"], ordered=True)
    dtype2 = CategoricalDtype(["x", "y"], ordered=False)
    c = Categorical([0, 1], dtype=dtype1)

    @pytest.mark.parametrize(
        "values, categories, ordered, dtype, expected",
        [
            [None, None, None, None, CategoricalDtype()],
            [None, ["a", "b"], True, None, dtype1],
            [c, None, None, dtype2, dtype2],
            [c, ["x", "y"], False, None, dtype2],
        ],
    )
    def test_from_values_or_dtype(self, values, categories, ordered, dtype, expected):
        result = CategoricalDtype._from_values_or_dtype(
            values, categories, ordered, dtype
        )
        assert result == expected

    @pytest.mark.parametrize(
        "values, categories, ordered, dtype",
        [
            [None, ["a", "b"], True, dtype2],
            [None, ["a", "b"], None, dtype2],
            [None, None, True, dtype2],
        ],
    )
    def test_from_values_or_dtype_raises(self, values, categories, ordered, dtype):
        msg = "Cannot specify `categories` or `ordered` together with `dtype`."
        with pytest.raises(ValueError, match=msg):
            CategoricalDtype._from_values_or_dtype(values, categories, ordered, dtype)

    def test_from_values_or_dtype_invalid_dtype(self):
        msg = "Cannot not construct CategoricalDtype from <class 'object'>"
        with pytest.raises(ValueError, match=msg):
            CategoricalDtype._from_values_or_dtype(None, None, None, object)

    def test_is_dtype(self, dtype):
        assert CategoricalDtype.is_dtype(dtype)
        assert CategoricalDtype.is_dtype("category")
        assert CategoricalDtype.is_dtype(CategoricalDtype())
        assert not CategoricalDtype.is_dtype("foo")
        assert not CategoricalDtype.is_dtype(np.float64)

    def test_basic(self, dtype):
        msg = "is_categorical_dtype is deprecated"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_categorical_dtype(dtype)

            factor = Categorical(["a", "b", "b", "a", "a", "c", "c", "c"])

            s = Series(factor, name="A")

            # dtypes
            assert is_categorical_dtype(s.dtype)
            assert is_categorical_dtype(s)
            assert not is_categorical_dtype(np.dtype("float64"))

    def test_tuple_categories(self):
        categories = [(1, "a"), (2, "b"), (3, "c")]
        result = CategoricalDtype(categories)
        assert all(result.categories == categories)

    @pytest.mark.parametrize(
        "categories, expected",
        [
            ([True, False], True),
            ([True, False, None], True),
            ([True, False, "a", "b'"], False),
            ([0, 1], False),
        ],
    )
    def test_is_boolean(self, categories, expected):
        cat = Categorical(categories)
        assert cat.dtype._is_boolean is expected
        assert is_bool_dtype(cat) is expected
        assert is_bool_dtype(cat.dtype) is expected

    def test_dtype_specific_categorical_dtype(self):
        expected = "datetime64[ns]"
        dti = DatetimeIndex([], dtype=expected)
        result = str(Categorical(dti).categories.dtype)
        assert result == expected

    def test_not_string(self):
        # though CategoricalDtype has object kind, it cannot be string
        assert not is_string_dtype(CategoricalDtype())

    def test_repr_range_categories(self):
        rng = pd.Index(range(3))
        dtype = CategoricalDtype(categories=rng, ordered=False)
        result = repr(dtype)

        expected = (
            "CategoricalDtype(categories=range(0, 3), ordered=False, "
            "categories_dtype=int64)"
        )
        assert result == expected

    def test_update_dtype(self):
        # GH 27338
        result = CategoricalDtype(["a"]).update_dtype(Categorical(["b"], ordered=True))
        expected = CategoricalDtype(["b"], ordered=True)
        assert result == expected

    def test_repr(self):
        cat = Categorical(pd.Index([1, 2, 3], dtype="int32"))
        result = cat.dtype.__repr__()
        expected = (
            "CategoricalDtype(categories=[1, 2, 3], ordered=False, "
            "categories_dtype=int32)"
        )
        assert result == expected


class TestDatetimeTZDtype(Base):
    @pytest.fixture
    def dtype(self):
        """
        Class level fixture of dtype for TestDatetimeTZDtype
        """
        return DatetimeTZDtype("ns", "US/Eastern")

    def test_alias_to_unit_raises(self):
        # 23990
        with pytest.raises(ValueError, match="Passing a dtype alias"):
            DatetimeTZDtype("datetime64[ns, US/Central]")

    def test_alias_to_unit_bad_alias_raises(self):
        # 23990
        with pytest.raises(TypeError, match=""):
            DatetimeTZDtype("this is a bad string")

        with pytest.raises(TypeError, match=""):
            DatetimeTZDtype("datetime64[ns, US/NotATZ]")

    def test_hash_vs_equality(self, dtype):
        # make sure that we satisfy is semantics
        dtype2 = DatetimeTZDtype("ns", "US/Eastern")
        dtype3 = DatetimeTZDtype(dtype2)
        assert dtype == dtype2
        assert dtype2 == dtype
        assert dtype3 == dtype
        assert hash(dtype) == hash(dtype2)
        assert hash(dtype) == hash(dtype3)

        dtype4 = DatetimeTZDtype("ns", "US/Central")
        assert dtype2 != dtype4
        assert hash(dtype2) != hash(dtype4)

    def test_construction_non_nanosecond(self):
        res = DatetimeTZDtype("ms", "US/Eastern")
        assert res.unit == "ms"
        assert res._creso == NpyDatetimeUnit.NPY_FR_ms.value
        assert res.str == "|M8[ms]"
        assert str(res) == "datetime64[ms, US/Eastern]"
        assert res.base == np.dtype("M8[ms]")

    def test_day_not_supported(self):
        msg = "DatetimeTZDtype only supports s, ms, us, ns units"
        with pytest.raises(ValueError, match=msg):
            DatetimeTZDtype("D", "US/Eastern")

    def test_subclass(self):
        a = DatetimeTZDtype.construct_from_string("datetime64[ns, US/Eastern]")
        b = DatetimeTZDtype.construct_from_string("datetime64[ns, CET]")

        assert issubclass(type(a), type(a))
        assert issubclass(type(a), type(b))

    def test_compat(self, dtype):
        msg = "is_datetime64tz_dtype is deprecated"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_datetime64tz_dtype(dtype)
            assert is_datetime64tz_dtype("datetime64[ns, US/Eastern]")
        assert is_datetime64_any_dtype(dtype)
        assert is_datetime64_any_dtype("datetime64[ns, US/Eastern]")
        assert is_datetime64_ns_dtype(dtype)
        assert is_datetime64_ns_dtype("datetime64[ns, US/Eastern]")
        assert not is_datetime64_dtype(dtype)
        assert not is_datetime64_dtype("datetime64[ns, US/Eastern]")

    def test_construction_from_string(self, dtype):
        result = DatetimeTZDtype.construct_from_string("datetime64[ns, US/Eastern]")
        assert is_dtype_equal(dtype, result)

    @pytest.mark.parametrize(
        "string",
        [
            "foo",
            "datetime64[ns, notatz]",
            # non-nano unit
            "datetime64[ps, UTC]",
            # dateutil str that returns None from gettz
            "datetime64[ns, dateutil/invalid]",
        ],
    )
    def test_construct_from_string_invalid_raises(self, string):
        msg = f"Cannot construct a 'DatetimeTZDtype' from '{string}'"
        with pytest.raises(TypeError, match=re.escape(msg)):
            DatetimeTZDtype.construct_from_string(string)

    def test_construct_from_string_wrong_type_raises(self):
        msg = "'construct_from_string' expects a string, got <class 'list'>"
        with pytest.raises(TypeError, match=msg):
            DatetimeTZDtype.construct_from_string(["datetime64[ns, notatz]"])

    def test_is_dtype(self, dtype):
        assert not DatetimeTZDtype.is_dtype(None)
        assert DatetimeTZDtype.is_dtype(dtype)
        assert DatetimeTZDtype.is_dtype("datetime64[ns, US/Eastern]")
        assert DatetimeTZDtype.is_dtype("M8[ns, US/Eastern]")
        assert not DatetimeTZDtype.is_dtype("foo")
        assert DatetimeTZDtype.is_dtype(DatetimeTZDtype("ns", "US/Pacific"))
        assert not DatetimeTZDtype.is_dtype(np.float64)

    def test_equality(self, dtype):
        assert is_dtype_equal(dtype, "datetime64[ns, US/Eastern]")
        assert is_dtype_equal(dtype, "M8[ns, US/Eastern]")
        assert is_dtype_equal(dtype, DatetimeTZDtype("ns", "US/Eastern"))
        assert not is_dtype_equal(dtype, "foo")
        assert not is_dtype_equal(dtype, DatetimeTZDtype("ns", "CET"))
        assert not is_dtype_equal(
            DatetimeTZDtype("ns", "US/Eastern"), DatetimeTZDtype("ns", "US/Pacific")
        )

        # numpy compat
        assert is_dtype_equal(np.dtype("M8[ns]"), "datetime64[ns]")

        assert dtype == "M8[ns, US/Eastern]"

    def test_basic(self, dtype):
        msg = "is_datetime64tz_dtype is deprecated"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_datetime64tz_dtype(dtype)

        dr = date_range("20130101", periods=3, tz="US/Eastern")
        s = Series(dr, name="A")

        # dtypes
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_datetime64tz_dtype(s.dtype)
            assert is_datetime64tz_dtype(s)
            assert not is_datetime64tz_dtype(np.dtype("float64"))
            assert not is_datetime64tz_dtype(1.0)

    def test_dst(self):
        dr1 = date_range("2013-01-01", periods=3, tz="US/Eastern")
        s1 = Series(dr1, name="A")
        assert isinstance(s1.dtype, DatetimeTZDtype)

        dr2 = date_range("2013-08-01", periods=3, tz="US/Eastern")
        s2 = Series(dr2, name="A")
        assert isinstance(s2.dtype, DatetimeTZDtype)
        assert s1.dtype == s2.dtype

    @pytest.mark.parametrize("tz", ["UTC", "US/Eastern"])
    @pytest.mark.parametrize("constructor", ["M8", "datetime64"])
    def test_parser(self, tz, constructor):
        # pr #11245
        dtz_str = f"{constructor}[ns, {tz}]"
        result = DatetimeTZDtype.construct_from_string(dtz_str)
        expected = DatetimeTZDtype("ns", tz)
        assert result == expected

    def test_empty(self):
        with pytest.raises(TypeError, match="A 'tz' is required."):
            DatetimeTZDtype()

    def test_tz_standardize(self):
        # GH 24713
        tz = pytz.timezone("US/Eastern")
        dr = date_range("2013-01-01", periods=3, tz="US/Eastern")
        dtype = DatetimeTZDtype("ns", dr.tz)
        assert dtype.tz == tz
        dtype = DatetimeTZDtype("ns", dr[0].tz)
        assert dtype.tz == tz


class TestPeriodDtype(Base):
    @pytest.fixture
    def dtype(self):
        """
        Class level fixture of dtype for TestPeriodDtype
        """
        return PeriodDtype("D")

    def test_hash_vs_equality(self, dtype):
        # make sure that we satisfy is semantics
        dtype2 = PeriodDtype("D")
        dtype3 = PeriodDtype(dtype2)
        assert dtype == dtype2
        assert dtype2 == dtype
        assert dtype3 == dtype
        assert dtype is not dtype2
        assert dtype2 is not dtype
        assert dtype3 is not dtype
        assert hash(dtype) == hash(dtype2)
        assert hash(dtype) == hash(dtype3)

    def test_construction(self):
        with pytest.raises(ValueError, match="Invalid frequency: xx"):
            PeriodDtype("xx")

        for s in ["period[D]", "Period[D]", "D"]:
            dt = PeriodDtype(s)
            assert dt.freq == pd.tseries.offsets.Day()

        for s in ["period[3D]", "Period[3D]", "3D"]:
            dt = PeriodDtype(s)
            assert dt.freq == pd.tseries.offsets.Day(3)

        for s in [
            "period[26h]",
            "Period[26h]",
            "26h",
            "period[1D2h]",
            "Period[1D2h]",
            "1D2h",
        ]:
            dt = PeriodDtype(s)
            assert dt.freq == pd.tseries.offsets.Hour(26)

    def test_cannot_use_custom_businessday(self):
        # GH#52534
        msg = "C is not supported as period frequency"
        msg1 = "<CustomBusinessDay> is not supported as period frequency"
        msg2 = r"PeriodDtype\[B\] is deprecated"
        with pytest.raises(ValueError, match=msg):
            PeriodDtype("C")
        with pytest.raises(ValueError, match=msg1):
            with tm.assert_produces_warning(FutureWarning, match=msg2):
                PeriodDtype(pd.offsets.CustomBusinessDay())

    def test_subclass(self):
        a = PeriodDtype("period[D]")
        b = PeriodDtype("period[3D]")

        assert issubclass(type(a), type(a))
        assert issubclass(type(a), type(b))

    def test_identity(self):
        assert PeriodDtype("period[D]") == PeriodDtype("period[D]")
        assert PeriodDtype("period[D]") is not PeriodDtype("period[D]")

        assert PeriodDtype("period[3D]") == PeriodDtype("period[3D]")
        assert PeriodDtype("period[3D]") is not PeriodDtype("period[3D]")

        assert PeriodDtype("period[1s1us]") == PeriodDtype("period[1000001us]")
        assert PeriodDtype("period[1s1us]") is not PeriodDtype("period[1000001us]")

    def test_compat(self, dtype):
        assert not is_datetime64_ns_dtype(dtype)
        assert not is_datetime64_ns_dtype("period[D]")
        assert not is_datetime64_dtype(dtype)
        assert not is_datetime64_dtype("period[D]")

    def test_construction_from_string(self, dtype):
        result = PeriodDtype("period[D]")
        assert is_dtype_equal(dtype, result)
        result = PeriodDtype.construct_from_string("period[D]")
        assert is_dtype_equal(dtype, result)

        with pytest.raises(TypeError, match="list"):
            PeriodDtype.construct_from_string([1, 2, 3])

    @pytest.mark.parametrize(
        "string",
        [
            "foo",
            "period[foo]",
            "foo[D]",
            "datetime64[ns]",
            "datetime64[ns, US/Eastern]",
        ],
    )
    def test_construct_dtype_from_string_invalid_raises(self, string):
        msg = f"Cannot construct a 'PeriodDtype' from '{string}'"
        with pytest.raises(TypeError, match=re.escape(msg)):
            PeriodDtype.construct_from_string(string)

    def test_is_dtype(self, dtype):
        assert PeriodDtype.is_dtype(dtype)
        assert PeriodDtype.is_dtype("period[D]")
        assert PeriodDtype.is_dtype("period[3D]")
        assert PeriodDtype.is_dtype(PeriodDtype("3D"))
        assert PeriodDtype.is_dtype("period[us]")
        assert PeriodDtype.is_dtype("period[s]")
        assert PeriodDtype.is_dtype(PeriodDtype("us"))
        assert PeriodDtype.is_dtype(PeriodDtype("s"))

        assert not PeriodDtype.is_dtype("D")
        assert not PeriodDtype.is_dtype("3D")
        assert not PeriodDtype.is_dtype("U")
        assert not PeriodDtype.is_dtype("s")
        assert not PeriodDtype.is_dtype("foo")
        assert not PeriodDtype.is_dtype(np.object_)
        assert not PeriodDtype.is_dtype(np.int64)
        assert not PeriodDtype.is_dtype(np.float64)

    def test_equality(self, dtype):
        assert is_dtype_equal(dtype, "period[D]")
        assert is_dtype_equal(dtype, PeriodDtype("D"))
        assert is_dtype_equal(dtype, PeriodDtype("D"))
        assert is_dtype_equal(PeriodDtype("D"), PeriodDtype("D"))

        assert not is_dtype_equal(dtype, "D")
        assert not is_dtype_equal(PeriodDtype("D"), PeriodDtype("2D"))

    def test_basic(self, dtype):
        msg = "is_period_dtype is deprecated"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_period_dtype(dtype)

            pidx = pd.period_range("2013-01-01 09:00", periods=5, freq="h")

            assert is_period_dtype(pidx.dtype)
            assert is_period_dtype(pidx)

            s = Series(pidx, name="A")

            assert is_period_dtype(s.dtype)
            assert is_period_dtype(s)

            assert not is_period_dtype(np.dtype("float64"))
            assert not is_period_dtype(1.0)

    def test_freq_argument_required(self):
        # GH#27388
        msg = "missing 1 required positional argument: 'freq'"
        with pytest.raises(TypeError, match=msg):
            PeriodDtype()

        msg = "PeriodDtype argument should be string or BaseOffset, got NoneType"
        with pytest.raises(TypeError, match=msg):
            # GH#51790
            PeriodDtype(None)

    def test_not_string(self):
        # though PeriodDtype has object kind, it cannot be string
        assert not is_string_dtype(PeriodDtype("D"))

    def test_perioddtype_caching_dateoffset_normalize(self):
        # GH 24121
        per_d = PeriodDtype(pd.offsets.YearEnd(normalize=True))
        assert per_d.freq.normalize

        per_d2 = PeriodDtype(pd.offsets.YearEnd(normalize=False))
        assert not per_d2.freq.normalize

    def test_dont_keep_ref_after_del(self):
        # GH 54184
        dtype = PeriodDtype("D")
        ref = weakref.ref(dtype)
        del dtype
        assert ref() is None


class TestIntervalDtype(Base):
    @pytest.fixture
    def dtype(self):
        """
        Class level fixture of dtype for TestIntervalDtype
        """
        return IntervalDtype("int64", "right")

    def test_hash_vs_equality(self, dtype):
        # make sure that we satisfy is semantics
        dtype2 = IntervalDtype("int64", "right")
        dtype3 = IntervalDtype(dtype2)
        assert dtype == dtype2
        assert dtype2 == dtype
        assert dtype3 == dtype
        assert dtype is not dtype2
        assert dtype2 is not dtype3
        assert dtype3 is not dtype
        assert hash(dtype) == hash(dtype2)
        assert hash(dtype) == hash(dtype3)

        dtype1 = IntervalDtype("interval")
        dtype2 = IntervalDtype(dtype1)
        dtype3 = IntervalDtype("interval")
        assert dtype2 == dtype1
        assert dtype2 == dtype2
        assert dtype2 == dtype3
        assert dtype2 is not dtype1
        assert dtype2 is dtype2
        assert dtype2 is not dtype3
        assert hash(dtype2) == hash(dtype1)
        assert hash(dtype2) == hash(dtype2)
        assert hash(dtype2) == hash(dtype3)

    @pytest.mark.parametrize(
        "subtype", ["interval[int64]", "Interval[int64]", "int64", np.dtype("int64")]
    )
    def test_construction(self, subtype):
        i = IntervalDtype(subtype, closed="right")
        assert i.subtype == np.dtype("int64")
        msg = "is_interval_dtype is deprecated"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_interval_dtype(i)

    @pytest.mark.parametrize(
        "subtype", ["interval[int64]", "Interval[int64]", "int64", np.dtype("int64")]
    )
    def test_construction_allows_closed_none(self, subtype):
        # GH#38394
        dtype = IntervalDtype(subtype)

        assert dtype.closed is None

    def test_closed_mismatch(self):
        msg = "'closed' keyword does not match value specified in dtype string"
        with pytest.raises(ValueError, match=msg):
            IntervalDtype("interval[int64, left]", "right")

    @pytest.mark.parametrize("subtype", [None, "interval", "Interval"])
    def test_construction_generic(self, subtype):
        # generic
        i = IntervalDtype(subtype)
        assert i.subtype is None
        msg = "is_interval_dtype is deprecated"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_interval_dtype(i)

    @pytest.mark.parametrize(
        "subtype",
        [
            CategoricalDtype(list("abc"), False),
            CategoricalDtype(list("wxyz"), True),
            object,
            str,
            "<U10",
            "interval[category]",
            "interval[object]",
        ],
    )
    def test_construction_not_supported(self, subtype):
        # GH 19016
        msg = (
            "category, object, and string subtypes are not supported "
            "for IntervalDtype"
        )
        with pytest.raises(TypeError, match=msg):
            IntervalDtype(subtype)

    @pytest.mark.parametrize("subtype", ["xx", "IntervalA", "Interval[foo]"])
    def test_construction_errors(self, subtype):
        msg = "could not construct IntervalDtype"
        with pytest.raises(TypeError, match=msg):
            IntervalDtype(subtype)

    def test_closed_must_match(self):
        # GH#37933
        dtype = IntervalDtype(np.float64, "left")

        msg = "dtype.closed and 'closed' do not match"
        with pytest.raises(ValueError, match=msg):
            IntervalDtype(dtype, closed="both")

    def test_closed_invalid(self):
        with pytest.raises(ValueError, match="closed must be one of"):
            IntervalDtype(np.float64, "foo")

    def test_construction_from_string(self, dtype):
        result = IntervalDtype("interval[int64, right]")
        assert is_dtype_equal(dtype, result)
        result = IntervalDtype.construct_from_string("interval[int64, right]")
        assert is_dtype_equal(dtype, result)

    @pytest.mark.parametrize("string", [0, 3.14, ("a", "b"), None])
    def test_construction_from_string_errors(self, string):
        # these are invalid entirely
        msg = f"'construct_from_string' expects a string, got {type(string)}"

        with pytest.raises(TypeError, match=re.escape(msg)):
            IntervalDtype.construct_from_string(string)

    @pytest.mark.parametrize("string", ["foo", "foo[int64]", "IntervalA"])
    def test_construction_from_string_error_subtype(self, string):
        # this is an invalid subtype
        msg = (
            "Incorrectly formatted string passed to constructor. "
            r"Valid formats include Interval or Interval\[dtype\] "
            "where dtype is numeric, datetime, or timedelta"
        )

        with pytest.raises(TypeError, match=msg):
            IntervalDtype.construct_from_string(string)

    def test_subclass(self):
        a = IntervalDtype("interval[int64, right]")
        b = IntervalDtype("interval[int64, right]")

        assert issubclass(type(a), type(a))
        assert issubclass(type(a), type(b))

    def test_is_dtype(self, dtype):
        assert IntervalDtype.is_dtype(dtype)
        assert IntervalDtype.is_dtype("interval")
        assert IntervalDtype.is_dtype(IntervalDtype("float64"))
        assert IntervalDtype.is_dtype(IntervalDtype("int64"))
        assert IntervalDtype.is_dtype(IntervalDtype(np.int64))
        assert IntervalDtype.is_dtype(IntervalDtype("float64", "left"))
        assert IntervalDtype.is_dtype(IntervalDtype("int64", "right"))
        assert IntervalDtype.is_dtype(IntervalDtype(np.int64, "both"))

        assert not IntervalDtype.is_dtype("D")
        assert not IntervalDtype.is_dtype("3D")
        assert not IntervalDtype.is_dtype("us")
        assert not IntervalDtype.is_dtype("S")
        assert not IntervalDtype.is_dtype("foo")
        assert not IntervalDtype.is_dtype("IntervalA")
        assert not IntervalDtype.is_dtype(np.object_)
        assert not IntervalDtype.is_dtype(np.int64)
        assert not IntervalDtype.is_dtype(np.float64)

    def test_equality(self, dtype):
        assert is_dtype_equal(dtype, "interval[int64, right]")
        assert is_dtype_equal(dtype, IntervalDtype("int64", "right"))
        assert is_dtype_equal(
            IntervalDtype("int64", "right"), IntervalDtype("int64", "right")
        )

        assert not is_dtype_equal(dtype, "interval[int64]")
        assert not is_dtype_equal(dtype, IntervalDtype("int64"))
        assert not is_dtype_equal(
            IntervalDtype("int64", "right"), IntervalDtype("int64")
        )

        assert not is_dtype_equal(dtype, "int64")
        assert not is_dtype_equal(
            IntervalDtype("int64", "neither"), IntervalDtype("float64", "right")
        )
        assert not is_dtype_equal(
            IntervalDtype("int64", "both"), IntervalDtype("int64", "left")
        )

        # invalid subtype comparisons do not raise when directly compared
        dtype1 = IntervalDtype("float64", "left")
        dtype2 = IntervalDtype("datetime64[ns, US/Eastern]", "left")
        assert dtype1 != dtype2
        assert dtype2 != dtype1

    @pytest.mark.parametrize(
        "subtype",
        [
            None,
            "interval",
            "Interval",
            "int64",
            "uint64",
            "float64",
            "complex128",
            "datetime64",
            "timedelta64",
            PeriodDtype("Q"),
        ],
    )
    def test_equality_generic(self, subtype):
        # GH 18980
        closed = "right" if subtype is not None else None
        dtype = IntervalDtype(subtype, closed=closed)
        assert is_dtype_equal(dtype, "interval")
        assert is_dtype_equal(dtype, IntervalDtype())

    @pytest.mark.parametrize(
        "subtype",
        [
            "int64",
            "uint64",
            "float64",
            "complex128",
            "datetime64",
            "timedelta64",
            PeriodDtype("Q"),
        ],
    )
    def test_name_repr(self, subtype):
        # GH 18980
        closed = "right" if subtype is not None else None
        dtype = IntervalDtype(subtype, closed=closed)
        expected = f"interval[{subtype}, {closed}]"
        assert str(dtype) == expected
        assert dtype.name == "interval"

    @pytest.mark.parametrize("subtype", [None, "interval", "Interval"])
    def test_name_repr_generic(self, subtype):
        # GH 18980
        dtype = IntervalDtype(subtype)
        assert str(dtype) == "interval"
        assert dtype.name == "interval"

    def test_basic(self, dtype):
        msg = "is_interval_dtype is deprecated"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_interval_dtype(dtype)

            ii = IntervalIndex.from_breaks(range(3))

            assert is_interval_dtype(ii.dtype)
            assert is_interval_dtype(ii)

            s = Series(ii, name="A")

            assert is_interval_dtype(s.dtype)
            assert is_interval_dtype(s)

    def test_basic_dtype(self):
        msg = "is_interval_dtype is deprecated"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_interval_dtype("interval[int64, both]")
            assert is_interval_dtype(IntervalIndex.from_tuples([(0, 1)]))
            assert is_interval_dtype(IntervalIndex.from_breaks(np.arange(4)))
            assert is_interval_dtype(
                IntervalIndex.from_breaks(date_range("20130101", periods=3))
            )
            assert not is_interval_dtype("U")
            assert not is_interval_dtype("S")
            assert not is_interval_dtype("foo")
            assert not is_interval_dtype(np.object_)
            assert not is_interval_dtype(np.int64)
            assert not is_interval_dtype(np.float64)

    def test_caching(self):
        # GH 54184: Caching not shown to improve performance
        IntervalDtype.reset_cache()
        dtype = IntervalDtype("int64", "right")
        assert len(IntervalDtype._cache_dtypes) == 0

        IntervalDtype("interval")
        assert len(IntervalDtype._cache_dtypes) == 0

        IntervalDtype.reset_cache()
        tm.round_trip_pickle(dtype)
        assert len(IntervalDtype._cache_dtypes) == 0

    def test_not_string(self):
        # GH30568: though IntervalDtype has object kind, it cannot be string
        assert not is_string_dtype(IntervalDtype())

    def test_unpickling_without_closed(self):
        # GH#38394
        dtype = IntervalDtype("interval")

        assert dtype._closed is None

        tm.round_trip_pickle(dtype)

    def test_dont_keep_ref_after_del(self):
        # GH 54184
        dtype = IntervalDtype("int64", "right")
        ref = weakref.ref(dtype)
        del dtype
        assert ref() is None


class TestCategoricalDtypeParametrized:
    @pytest.mark.parametrize(
        "categories",
        [
            list("abcd"),
            np.arange(1000),
            ["a", "b", 10, 2, 1.3, True],
            [True, False],
            date_range("2017", periods=4),
        ],
    )
    def test_basic(self, categories, ordered):
        c1 = CategoricalDtype(categories, ordered=ordered)
        tm.assert_index_equal(c1.categories, pd.Index(categories))
        assert c1.ordered is ordered

    def test_order_matters(self):
        categories = ["a", "b"]
        c1 = CategoricalDtype(categories, ordered=True)
        c2 = CategoricalDtype(categories, ordered=False)
        c3 = CategoricalDtype(categories, ordered=None)
        assert c1 is not c2
        assert c1 is not c3

    @pytest.mark.parametrize("ordered", [False, None])
    def test_unordered_same(self, ordered):
        c1 = CategoricalDtype(["a", "b"], ordered=ordered)
        c2 = CategoricalDtype(["b", "a"], ordered=ordered)
        assert hash(c1) == hash(c2)

    def test_categories(self):
        result = CategoricalDtype(["a", "b", "c"])
        tm.assert_index_equal(result.categories, pd.Index(["a", "b", "c"]))
        assert result.ordered is False

    def test_equal_but_different(self):
        c1 = CategoricalDtype([1, 2, 3])
        c2 = CategoricalDtype([1.0, 2.0, 3.0])
        assert c1 is not c2
        assert c1 != c2

    def test_equal_but_different_mixed_dtypes(self):
        c1 = CategoricalDtype([1, 2, "3"])
        c2 = CategoricalDtype(["3", 1, 2])
        assert c1 is not c2
        assert c1 == c2

    def test_equal_empty_ordered(self):
        c1 = CategoricalDtype([], ordered=True)
        c2 = CategoricalDtype([], ordered=True)
        assert c1 is not c2
        assert c1 == c2

    def test_equal_empty_unordered(self):
        c1 = CategoricalDtype([])
        c2 = CategoricalDtype([])
        assert c1 is not c2
        assert c1 == c2

    @pytest.mark.parametrize("v1, v2", [([1, 2, 3], [1, 2, 3]), ([1, 2, 3], [3, 2, 1])])
    def test_order_hashes_different(self, v1, v2):
        c1 = CategoricalDtype(v1, ordered=False)
        c2 = CategoricalDtype(v2, ordered=True)
        c3 = CategoricalDtype(v1, ordered=None)
        assert c1 is not c2
        assert c1 is not c3

    def test_nan_invalid(self):
        msg = "Categorical categories cannot be null"
        with pytest.raises(ValueError, match=msg):
            CategoricalDtype([1, 2, np.nan])

    def test_non_unique_invalid(self):
        msg = "Categorical categories must be unique"
        with pytest.raises(ValueError, match=msg):
            CategoricalDtype([1, 2, 1])

    def test_same_categories_different_order(self):
        c1 = CategoricalDtype(["a", "b"], ordered=True)
        c2 = CategoricalDtype(["b", "a"], ordered=True)
        assert c1 is not c2

    @pytest.mark.parametrize("ordered1", [True, False, None])
    @pytest.mark.parametrize("ordered2", [True, False, None])
    def test_categorical_equality(self, ordered1, ordered2):
        # same categories, same order
        # any combination of None/False are equal
        # True/True is the only combination with True that are equal
        c1 = CategoricalDtype(list("abc"), ordered1)
        c2 = CategoricalDtype(list("abc"), ordered2)
        result = c1 == c2
        expected = bool(ordered1) is bool(ordered2)
        assert result is expected

        # same categories, different order
        # any combination of None/False are equal (order doesn't matter)
        # any combination with True are not equal (different order of cats)
        c1 = CategoricalDtype(list("abc"), ordered1)
        c2 = CategoricalDtype(list("cab"), ordered2)
        result = c1 == c2
        expected = (bool(ordered1) is False) and (bool(ordered2) is False)
        assert result is expected

        # different categories
        c2 = CategoricalDtype([1, 2, 3], ordered2)
        assert c1 != c2

        # none categories
        c1 = CategoricalDtype(list("abc"), ordered1)
        c2 = CategoricalDtype(None, ordered2)
        c3 = CategoricalDtype(None, ordered1)
        assert c1 != c2
        assert c2 != c1
        assert c2 == c3

    def test_categorical_dtype_equality_requires_categories(self):
        # CategoricalDtype with categories=None is *not* equal to
        #  any fully-initialized CategoricalDtype
        first = CategoricalDtype(["a", "b"])
        second = CategoricalDtype()
        third = CategoricalDtype(ordered=True)

        assert second == second
        assert third == third

        assert first != second
        assert second != first
        assert first != third
        assert third != first
        assert second == third
        assert third == second

    @pytest.mark.parametrize("categories", [list("abc"), None])
    @pytest.mark.parametrize("other", ["category", "not a category"])
    def test_categorical_equality_strings(self, categories, ordered, other):
        c1 = CategoricalDtype(categories, ordered)
        result = c1 == other
        expected = other == "category"
        assert result is expected

    def test_invalid_raises(self):
        with pytest.raises(TypeError, match="ordered"):
            CategoricalDtype(["a", "b"], ordered="foo")

        with pytest.raises(TypeError, match="'categories' must be list-like"):
            CategoricalDtype("category")

    def test_mixed(self):
        a = CategoricalDtype(["a", "b", 1, 2])
        b = CategoricalDtype(["a", "b", "1", "2"])
        assert hash(a) != hash(b)

    def test_from_categorical_dtype_identity(self):
        c1 = Categorical([1, 2], categories=[1, 2, 3], ordered=True)
        # Identity test for no changes
        c2 = CategoricalDtype._from_categorical_dtype(c1)
        assert c2 is c1

    def test_from_categorical_dtype_categories(self):
        c1 = Categorical([1, 2], categories=[1, 2, 3], ordered=True)
        # override categories
        result = CategoricalDtype._from_categorical_dtype(c1, categories=[2, 3])
        assert result == CategoricalDtype([2, 3], ordered=True)

    def test_from_categorical_dtype_ordered(self):
        c1 = Categorical([1, 2], categories=[1, 2, 3], ordered=True)
        # override ordered
        result = CategoricalDtype._from_categorical_dtype(c1, ordered=False)
        assert result == CategoricalDtype([1, 2, 3], ordered=False)

    def test_from_categorical_dtype_both(self):
        c1 = Categorical([1, 2], categories=[1, 2, 3], ordered=True)
        # override ordered
        result = CategoricalDtype._from_categorical_dtype(
            c1, categories=[1, 2], ordered=False
        )
        assert result == CategoricalDtype([1, 2], ordered=False)

    def test_str_vs_repr(self, ordered, using_infer_string):
        c1 = CategoricalDtype(["a", "b"], ordered=ordered)
        assert str(c1) == "category"
        # Py2 will have unicode prefixes
        dtype = "str" if using_infer_string else "object"
        pat = (
            r"CategoricalDtype\(categories=\[.*\], ordered={ordered}, "
            rf"categories_dtype={dtype}\)"
        )
        assert re.match(pat.format(ordered=ordered), repr(c1))

    def test_categorical_categories(self):
        # GH17884
        c1 = CategoricalDtype(Categorical(["a", "b"]))
        tm.assert_index_equal(c1.categories, pd.Index(["a", "b"]))
        c1 = CategoricalDtype(CategoricalIndex(["a", "b"]))
        tm.assert_index_equal(c1.categories, pd.Index(["a", "b"]))

    @pytest.mark.parametrize(
        "new_categories", [list("abc"), list("cba"), list("wxyz"), None]
    )
    @pytest.mark.parametrize("new_ordered", [True, False, None])
    def test_update_dtype(self, ordered, new_categories, new_ordered):
        original_categories = list("abc")
        dtype = CategoricalDtype(original_categories, ordered)
        new_dtype = CategoricalDtype(new_categories, new_ordered)

        result = dtype.update_dtype(new_dtype)
        expected_categories = pd.Index(new_categories or original_categories)
        expected_ordered = new_ordered if new_ordered is not None else dtype.ordered

        tm.assert_index_equal(result.categories, expected_categories)
        assert result.ordered is expected_ordered

    def test_update_dtype_string(self, ordered):
        dtype = CategoricalDtype(list("abc"), ordered)
        expected_categories = dtype.categories
        expected_ordered = dtype.ordered
        result = dtype.update_dtype("category")
        tm.assert_index_equal(result.categories, expected_categories)
        assert result.ordered is expected_ordered

    @pytest.mark.parametrize("bad_dtype", ["foo", object, np.int64, PeriodDtype("Q")])
    def test_update_dtype_errors(self, bad_dtype):
        dtype = CategoricalDtype(list("abc"), False)
        msg = "a CategoricalDtype must be passed to perform an update, "
        with pytest.raises(ValueError, match=msg):
            dtype.update_dtype(bad_dtype)


@pytest.mark.parametrize(
    "dtype", [CategoricalDtype, IntervalDtype, DatetimeTZDtype, PeriodDtype]
)
def test_registry(dtype):
    assert dtype in registry.dtypes


@pytest.mark.parametrize(
    "dtype, expected",
    [
        ("int64", None),
        ("interval", IntervalDtype()),
        ("interval[int64, neither]", IntervalDtype()),
        ("interval[datetime64[ns], left]", IntervalDtype("datetime64[ns]", "left")),
        ("period[D]", PeriodDtype("D")),
        ("category", CategoricalDtype()),
        ("datetime64[ns, US/Eastern]", DatetimeTZDtype("ns", "US/Eastern")),
    ],
)
def test_registry_find(dtype, expected):
    assert registry.find(dtype) == expected


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (str, False),
        (int, False),
        (bool, True),
        (np.bool_, True),
        (np.array(["a", "b"]), False),
        (Series([1, 2]), False),
        (np.array([True, False]), True),
        (Series([True, False]), True),
        (SparseArray([True, False]), True),
        (SparseDtype(bool), True),
    ],
)
def test_is_bool_dtype(dtype, expected):
    result = is_bool_dtype(dtype)
    assert result is expected


def test_is_bool_dtype_sparse():
    result = is_bool_dtype(Series(SparseArray([True, False])))
    assert result is True


@pytest.mark.parametrize(
    "check",
    [
        is_categorical_dtype,
        is_datetime64tz_dtype,
        is_period_dtype,
        is_datetime64_ns_dtype,
        is_datetime64_dtype,
        is_interval_dtype,
        is_datetime64_any_dtype,
        is_string_dtype,
        is_bool_dtype,
    ],
)
def test_is_dtype_no_warning(check):
    data = pd.DataFrame({"A": [1, 2]})

    warn = None
    msg = f"{check.__name__} is deprecated"
    if (
        check is is_categorical_dtype
        or check is is_interval_dtype
        or check is is_datetime64tz_dtype
        or check is is_period_dtype
    ):
        warn = DeprecationWarning

    with tm.assert_produces_warning(warn, match=msg):
        check(data)

    with tm.assert_produces_warning(warn, match=msg):
        check(data["A"])


def test_period_dtype_compare_to_string():
    # https://github.com/pandas-dev/pandas/issues/37265
    dtype = PeriodDtype(freq="M")
    assert (dtype == "period[M]") is True
    assert (dtype != "period[M]") is False


def test_compare_complex_dtypes():
    # GH 28050
    df = pd.DataFrame(np.arange(5).astype(np.complex128))
    msg = "'<' not supported between instances of 'complex' and 'complex'"

    with pytest.raises(TypeError, match=msg):
        df < df.astype(object)

    with pytest.raises(TypeError, match=msg):
        df.lt(df.astype(object))


def test_cast_string_to_complex():
    # GH 4895
    expected = pd.DataFrame(["1.0+5j", "1.5-3j"], dtype=complex)
    result = pd.DataFrame(["1.0+5j", "1.5-3j"]).astype(complex)
    tm.assert_frame_equal(result, expected)


def test_categorical_complex():
    result = Categorical([1, 2 + 2j])
    expected = Categorical([1.0 + 0.0j, 2.0 + 2.0j])
    tm.assert_categorical_equal(result, expected)
    result = Categorical([1, 2, 2 + 2j])
    expected = Categorical([1.0 + 0.0j, 2.0 + 0.0j, 2.0 + 2.0j])
    tm.assert_categorical_equal(result, expected)


def test_multi_column_dtype_assignment():
    # GH #27583
    df = pd.DataFrame({"a": [0.0], "b": 0.0})
    expected = pd.DataFrame({"a": [0], "b": 0})

    df[["a", "b"]] = 0
    tm.assert_frame_equal(df, expected)

    df["b"] = 0
    tm.assert_frame_equal(df, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\formats\test_to_string.py ===
from datetime import (
    datetime,
    timedelta,
)
from io import StringIO
import re
import sys
from textwrap import dedent

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas import (
    CategoricalIndex,
    DataFrame,
    Index,
    NaT,
    Series,
    Timestamp,
    concat,
    date_range,
    get_option,
    option_context,
    read_csv,
    timedelta_range,
    to_datetime,
)
import pandas._testing as tm


def _three_digit_exp():
    return f"{1.7e8:.4g}" == "1.7e+008"


class TestDataFrameToStringFormatters:
    def test_to_string_masked_ea_with_formatter(self):
        # GH#39336
        df = DataFrame(
            {
                "a": Series([0.123456789, 1.123456789], dtype="Float64"),
                "b": Series([1, 2], dtype="Int64"),
            }
        )
        result = df.to_string(formatters=["{:.2f}".format, "{:.2f}".format])
        expected = dedent(
            """\
                  a     b
            0  0.12  1.00
            1  1.12  2.00"""
        )
        assert result == expected

    def test_to_string_with_formatters(self):
        df = DataFrame(
            {
                "int": [1, 2, 3],
                "float": [1.0, 2.0, 3.0],
                "object": [(1, 2), True, False],
            },
            columns=["int", "float", "object"],
        )

        formatters = [
            ("int", lambda x: f"0x{x:x}"),
            ("float", lambda x: f"[{x: 4.1f}]"),
            ("object", lambda x: f"-{x!s}-"),
        ]
        result = df.to_string(formatters=dict(formatters))
        result2 = df.to_string(formatters=list(zip(*formatters))[1])
        assert result == (
            "  int  float    object\n"
            "0 0x1 [ 1.0]  -(1, 2)-\n"
            "1 0x2 [ 2.0]    -True-\n"
            "2 0x3 [ 3.0]   -False-"
        )
        assert result == result2

    def test_to_string_with_datetime64_monthformatter(self):
        months = [datetime(2016, 1, 1), datetime(2016, 2, 2)]
        x = DataFrame({"months": months})

        def format_func(x):
            return x.strftime("%Y-%m")

        result = x.to_string(formatters={"months": format_func})
        expected = dedent(
            """\
            months
            0 2016-01
            1 2016-02"""
        )
        assert result.strip() == expected

    def test_to_string_with_datetime64_hourformatter(self):
        x = DataFrame(
            {"hod": to_datetime(["10:10:10.100", "12:12:12.120"], format="%H:%M:%S.%f")}
        )

        def format_func(x):
            return x.strftime("%H:%M")

        result = x.to_string(formatters={"hod": format_func})
        expected = dedent(
            """\
            hod
            0 10:10
            1 12:12"""
        )
        assert result.strip() == expected

    def test_to_string_with_formatters_unicode(self):
        df = DataFrame({"c/\u03c3": [1, 2, 3]})
        result = df.to_string(formatters={"c/\u03c3": str})
        expected = dedent(
            """\
              c/\u03c3
            0   1
            1   2
            2   3"""
        )
        assert result == expected

        def test_to_string_index_formatter(self):
            df = DataFrame([range(5), range(5, 10), range(10, 15)])

            rs = df.to_string(formatters={"__index__": lambda x: "abc"[x]})

            xp = dedent(
                """\
                0   1   2   3   4
            a   0   1   2   3   4
            b   5   6   7   8   9
            c  10  11  12  13  14\
            """
            )
            assert rs == xp

    def test_no_extra_space(self):
        # GH#52690: Check that no extra space is given
        col1 = "TEST"
        col2 = "PANDAS"
        col3 = "to_string"
        expected = f"{col1:<6s} {col2:<7s} {col3:<10s}"
        df = DataFrame([{"col1": "TEST", "col2": "PANDAS", "col3": "to_string"}])
        d = {"col1": "{:<6s}".format, "col2": "{:<7s}".format, "col3": "{:<10s}".format}
        result = df.to_string(index=False, header=False, formatters=d)
        assert result == expected


class TestDataFrameToStringColSpace:
    def test_to_string_with_column_specific_col_space_raises(self):
        df = DataFrame(
            np.random.default_rng(2).random(size=(3, 3)), columns=["a", "b", "c"]
        )

        msg = (
            "Col_space length\\(\\d+\\) should match "
            "DataFrame number of columns\\(\\d+\\)"
        )
        with pytest.raises(ValueError, match=msg):
            df.to_string(col_space=[30, 40])

        with pytest.raises(ValueError, match=msg):
            df.to_string(col_space=[30, 40, 50, 60])

        msg = "unknown column"
        with pytest.raises(ValueError, match=msg):
            df.to_string(col_space={"a": "foo", "b": 23, "d": 34})

    def test_to_string_with_column_specific_col_space(self):
        df = DataFrame(
            np.random.default_rng(2).random(size=(3, 3)), columns=["a", "b", "c"]
        )

        result = df.to_string(col_space={"a": 10, "b": 11, "c": 12})
        # 3 separating space + each col_space for (id, a, b, c)
        assert len(result.split("\n")[1]) == (3 + 1 + 10 + 11 + 12)

        result = df.to_string(col_space=[10, 11, 12])
        assert len(result.split("\n")[1]) == (3 + 1 + 10 + 11 + 12)

    def test_to_string_with_col_space(self):
        df = DataFrame(np.random.default_rng(2).random(size=(1, 3)))
        c10 = len(df.to_string(col_space=10).split("\n")[1])
        c20 = len(df.to_string(col_space=20).split("\n")[1])
        c30 = len(df.to_string(col_space=30).split("\n")[1])
        assert c10 < c20 < c30

        # GH#8230
        # col_space wasn't being applied with header=False
        with_header = df.to_string(col_space=20)
        with_header_row1 = with_header.splitlines()[1]
        no_header = df.to_string(col_space=20, header=False)
        assert len(with_header_row1) == len(no_header)

    def test_to_string_repr_tuples(self):
        buf = StringIO()

        df = DataFrame({"tups": list(zip(range(10), range(10)))})
        repr(df)
        df.to_string(col_space=10, buf=buf)


class TestDataFrameToStringHeader:
    def test_to_string_header_false(self):
        # GH#49230
        df = DataFrame([1, 2])
        df.index.name = "a"
        s = df.to_string(header=False)
        expected = "a   \n0  1\n1  2"
        assert s == expected

        df = DataFrame([[1, 2], [3, 4]])
        df.index.name = "a"
        s = df.to_string(header=False)
        expected = "a      \n0  1  2\n1  3  4"
        assert s == expected

    def test_to_string_multindex_header(self):
        # GH#16718
        df = DataFrame({"a": [0], "b": [1], "c": [2], "d": [3]}).set_index(["a", "b"])
        res = df.to_string(header=["r1", "r2"])
        exp = "    r1 r2\na b      \n0 1  2  3"
        assert res == exp

    def test_to_string_no_header(self):
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        df_s = df.to_string(header=False)
        expected = "0  1  4\n1  2  5\n2  3  6"

        assert df_s == expected

    def test_to_string_specified_header(self):
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        df_s = df.to_string(header=["X", "Y"])
        expected = "   X  Y\n0  1  4\n1  2  5\n2  3  6"

        assert df_s == expected

        msg = "Writing 2 cols but got 1 aliases"
        with pytest.raises(ValueError, match=msg):
            df.to_string(header=["X"])


class TestDataFrameToStringLineWidth:
    def test_to_string_line_width(self):
        df = DataFrame(123, index=range(10, 15), columns=range(30))
        lines = df.to_string(line_width=80)
        assert max(len(line) for line in lines.split("\n")) == 80

    def test_to_string_line_width_no_index(self):
        # GH#13998, GH#22505
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        df_s = df.to_string(line_width=1, index=False)
        expected = " x  \\\n 1   \n 2   \n 3   \n\n y  \n 4  \n 5  \n 6  "

        assert df_s == expected

        df = DataFrame({"x": [11, 22, 33], "y": [4, 5, 6]})

        df_s = df.to_string(line_width=1, index=False)
        expected = " x  \\\n11   \n22   \n33   \n\n y  \n 4  \n 5  \n 6  "

        assert df_s == expected

        df = DataFrame({"x": [11, 22, -33], "y": [4, 5, -6]})

        df_s = df.to_string(line_width=1, index=False)
        expected = "  x  \\\n 11   \n 22   \n-33   \n\n y  \n 4  \n 5  \n-6  "

        assert df_s == expected

    def test_to_string_line_width_no_header(self):
        # GH#53054
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        df_s = df.to_string(line_width=1, header=False)
        expected = "0  1  \\\n1  2   \n2  3   \n\n0  4  \n1  5  \n2  6  "

        assert df_s == expected

        df = DataFrame({"x": [11, 22, 33], "y": [4, 5, 6]})

        df_s = df.to_string(line_width=1, header=False)
        expected = "0  11  \\\n1  22   \n2  33   \n\n0  4  \n1  5  \n2  6  "

        assert df_s == expected

        df = DataFrame({"x": [11, 22, -33], "y": [4, 5, -6]})

        df_s = df.to_string(line_width=1, header=False)
        expected = "0  11  \\\n1  22   \n2 -33   \n\n0  4  \n1  5  \n2 -6  "

        assert df_s == expected

    def test_to_string_line_width_with_both_index_and_header(self):
        # GH#53054
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        df_s = df.to_string(line_width=1)
        expected = (
            "   x  \\\n0  1   \n1  2   \n2  3   \n\n   y  \n0  4  \n1  5  \n2  6  "
        )

        assert df_s == expected

        df = DataFrame({"x": [11, 22, 33], "y": [4, 5, 6]})

        df_s = df.to_string(line_width=1)
        expected = (
            "    x  \\\n0  11   \n1  22   \n2  33   \n\n   y  \n0  4  \n1  5  \n2  6  "
        )

        assert df_s == expected

        df = DataFrame({"x": [11, 22, -33], "y": [4, 5, -6]})

        df_s = df.to_string(line_width=1)
        expected = (
            "    x  \\\n0  11   \n1  22   \n2 -33   \n\n   y  \n0  4  \n1  5  \n2 -6  "
        )

        assert df_s == expected

    def test_to_string_line_width_no_index_no_header(self):
        # GH#53054
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        df_s = df.to_string(line_width=1, index=False, header=False)
        expected = "1  \\\n2   \n3   \n\n4  \n5  \n6  "

        assert df_s == expected

        df = DataFrame({"x": [11, 22, 33], "y": [4, 5, 6]})

        df_s = df.to_string(line_width=1, index=False, header=False)
        expected = "11  \\\n22   \n33   \n\n4  \n5  \n6  "

        assert df_s == expected

        df = DataFrame({"x": [11, 22, -33], "y": [4, 5, -6]})

        df_s = df.to_string(line_width=1, index=False, header=False)
        expected = " 11  \\\n 22   \n-33   \n\n 4  \n 5  \n-6  "

        assert df_s == expected


class TestToStringNumericFormatting:
    def test_to_string_float_format_no_fixed_width(self):
        # GH#21625
        df = DataFrame({"x": [0.19999]})
        expected = "      x\n0 0.200"
        assert df.to_string(float_format="%.3f") == expected

        # GH#22270
        df = DataFrame({"x": [100.0]})
        expected = "    x\n0 100"
        assert df.to_string(float_format="%.0f") == expected

    def test_to_string_small_float_values(self):
        df = DataFrame({"a": [1.5, 1e-17, -5.5e-7]})

        result = df.to_string()
        # sadness per above
        if _three_digit_exp():
            expected = (
                "               a\n"
                "0  1.500000e+000\n"
                "1  1.000000e-017\n"
                "2 -5.500000e-007"
            )
        else:
            expected = (
                "              a\n"
                "0  1.500000e+00\n"
                "1  1.000000e-17\n"
                "2 -5.500000e-07"
            )
        assert result == expected

        # but not all exactly zero
        df = df * 0
        result = df.to_string()
        expected = "   0\n0  0\n1  0\n2 -0"
        # TODO: assert that these match??

    def test_to_string_complex_float_formatting(self):
        # GH #25514, 25745
        with option_context("display.precision", 5):
            df = DataFrame(
                {
                    "x": [
                        (0.4467846931321966 + 0.0715185102060818j),
                        (0.2739442392974528 + 0.23515228785438969j),
                        (0.26974928742135185 + 0.3250604054898979j),
                        (-1j),
                    ]
                }
            )
            result = df.to_string()
            expected = (
                "                  x\n0  0.44678+0.07152j\n"
                "1  0.27394+0.23515j\n"
                "2  0.26975+0.32506j\n"
                "3 -0.00000-1.00000j"
            )
            assert result == expected

    def test_to_string_format_inf(self):
        # GH#24861
        df = DataFrame(
            {
                "A": [-np.inf, np.inf, -1, -2.1234, 3, 4],
                "B": [-np.inf, np.inf, "foo", "foooo", "fooooo", "bar"],
            }
        )
        result = df.to_string()

        expected = (
            "        A       B\n"
            "0    -inf    -inf\n"
            "1     inf     inf\n"
            "2 -1.0000     foo\n"
            "3 -2.1234   foooo\n"
            "4  3.0000  fooooo\n"
            "5  4.0000     bar"
        )
        assert result == expected

        df = DataFrame(
            {
                "A": [-np.inf, np.inf, -1.0, -2.0, 3.0, 4.0],
                "B": [-np.inf, np.inf, "foo", "foooo", "fooooo", "bar"],
            }
        )
        result = df.to_string()

        expected = (
            "     A       B\n"
            "0 -inf    -inf\n"
            "1  inf     inf\n"
            "2 -1.0     foo\n"
            "3 -2.0   foooo\n"
            "4  3.0  fooooo\n"
            "5  4.0     bar"
        )
        assert result == expected

    def test_to_string_int_formatting(self):
        df = DataFrame({"x": [-15, 20, 25, -35]})
        assert issubclass(df["x"].dtype.type, np.integer)

        output = df.to_string()
        expected = "    x\n0 -15\n1  20\n2  25\n3 -35"
        assert output == expected

    def test_to_string_float_formatting(self):
        with option_context(
            "display.precision",
            5,
            "display.notebook_repr_html",
            False,
        ):
            df = DataFrame(
                {"x": [0, 0.25, 3456.000, 12e45, 1.64e6, 1.7e8, 1.253456, np.pi, -1e6]}
            )

            df_s = df.to_string()

            if _three_digit_exp():
                expected = (
                    "              x\n0  0.00000e+000\n1  2.50000e-001\n"
                    "2  3.45600e+003\n3  1.20000e+046\n4  1.64000e+006\n"
                    "5  1.70000e+008\n6  1.25346e+000\n7  3.14159e+000\n"
                    "8 -1.00000e+006"
                )
            else:
                expected = (
                    "             x\n0  0.00000e+00\n1  2.50000e-01\n"
                    "2  3.45600e+03\n3  1.20000e+46\n4  1.64000e+06\n"
                    "5  1.70000e+08\n6  1.25346e+00\n7  3.14159e+00\n"
                    "8 -1.00000e+06"
                )
            assert df_s == expected

            df = DataFrame({"x": [3234, 0.253]})
            df_s = df.to_string()

            expected = "          x\n0  3234.000\n1     0.253"
            assert df_s == expected

        assert get_option("display.precision") == 6

        df = DataFrame({"x": [1e9, 0.2512]})
        df_s = df.to_string()

        if _three_digit_exp():
            expected = "               x\n0  1.000000e+009\n1  2.512000e-001"
        else:
            expected = "              x\n0  1.000000e+09\n1  2.512000e-01"
        assert df_s == expected


class TestDataFrameToString:
    def test_to_string_decimal(self):
        # GH#23614
        df = DataFrame({"A": [6.0, 3.1, 2.2]})
        expected = "     A\n0  6,0\n1  3,1\n2  2,2"
        assert df.to_string(decimal=",") == expected

    def test_to_string_left_justify_cols(self):
        df = DataFrame({"x": [3234, 0.253]})
        df_s = df.to_string(justify="left")
        expected = "   x       \n0  3234.000\n1     0.253"
        assert df_s == expected

    def test_to_string_format_na(self):
        df = DataFrame(
            {
                "A": [np.nan, -1, -2.1234, 3, 4],
                "B": [np.nan, "foo", "foooo", "fooooo", "bar"],
            }
        )
        result = df.to_string()

        expected = (
            "        A       B\n"
            "0     NaN     NaN\n"
            "1 -1.0000     foo\n"
            "2 -2.1234   foooo\n"
            "3  3.0000  fooooo\n"
            "4  4.0000     bar"
        )
        assert result == expected

        df = DataFrame(
            {
                "A": [np.nan, -1.0, -2.0, 3.0, 4.0],
                "B": [np.nan, "foo", "foooo", "fooooo", "bar"],
            }
        )
        result = df.to_string()

        expected = (
            "     A       B\n"
            "0  NaN     NaN\n"
            "1 -1.0     foo\n"
            "2 -2.0   foooo\n"
            "3  3.0  fooooo\n"
            "4  4.0     bar"
        )
        assert result == expected

    def test_to_string_with_dict_entries(self):
        df = DataFrame({"A": [{"a": 1, "b": 2}]})

        val = df.to_string()
        assert "'a': 1" in val
        assert "'b': 2" in val

    def test_to_string_with_categorical_columns(self):
        # GH#35439
        data = [[4, 2], [3, 2], [4, 3]]
        cols = ["aaaaaaaaa", "b"]
        df = DataFrame(data, columns=cols)
        df_cat_cols = DataFrame(data, columns=CategoricalIndex(cols))

        assert df.to_string() == df_cat_cols.to_string()

    def test_repr_embedded_ndarray(self):
        arr = np.empty(10, dtype=[("err", object)])
        for i in range(len(arr)):
            arr["err"][i] = np.random.default_rng(2).standard_normal(i)

        df = DataFrame(arr)
        repr(df["err"])
        repr(df)
        df.to_string()

    def test_to_string_truncate(self):
        # GH 9784 - dont truncate when calling DataFrame.to_string
        df = DataFrame(
            [
                {
                    "a": "foo",
                    "b": "bar",
                    "c": "let's make this a very VERY long line that is longer "
                    "than the default 50 character limit",
                    "d": 1,
                },
                {"a": "foo", "b": "bar", "c": "stuff", "d": 1},
            ]
        )
        df.set_index(["a", "b", "c"])
        assert df.to_string() == (
            "     a    b                                         "
            "                                                c  d\n"
            "0  foo  bar  let's make this a very VERY long line t"
            "hat is longer than the default 50 character limit  1\n"
            "1  foo  bar                                         "
            "                                            stuff  1"
        )
        with option_context("max_colwidth", 20):
            # the display option has no effect on the to_string method
            assert df.to_string() == (
                "     a    b                                         "
                "                                                c  d\n"
                "0  foo  bar  let's make this a very VERY long line t"
                "hat is longer than the default 50 character limit  1\n"
                "1  foo  bar                                         "
                "                                            stuff  1"
            )
        assert df.to_string(max_colwidth=20) == (
            "     a    b                    c  d\n"
            "0  foo  bar  let's make this ...  1\n"
            "1  foo  bar                stuff  1"
        )

    @pytest.mark.parametrize(
        "input_array, expected",
        [
            ({"A": ["a"]}, "A\na"),
            ({"A": ["a", "b"], "B": ["c", "dd"]}, "A  B\na  c\nb dd"),
            ({"A": ["a", 1], "B": ["aa", 1]}, "A  B\na aa\n1  1"),
        ],
    )
    def test_format_remove_leading_space_dataframe(self, input_array, expected):
        # GH#24980
        df = DataFrame(input_array).to_string(index=False)
        assert df == expected

    @pytest.mark.parametrize(
        "data,expected",
        [
            (
                {"col1": [1, 2], "col2": [3, 4]},
                "   col1  col2\n0     1     3\n1     2     4",
            ),
            (
                {"col1": ["Abc", 0.756], "col2": [np.nan, 4.5435]},
                "    col1    col2\n0    Abc     NaN\n1  0.756  4.5435",
            ),
            (
                {"col1": [np.nan, "a"], "col2": [0.009, 3.543], "col3": ["Abc", 23]},
                "  col1   col2 col3\n0  NaN  0.009  Abc\n1    a  3.543   23",
            ),
        ],
    )
    def test_to_string_max_rows_zero(self, data, expected):
        # GH#35394
        result = DataFrame(data=data).to_string(max_rows=0)
        assert result == expected

    @pytest.mark.parametrize(
        "max_cols, max_rows, expected",
        [
            (
                10,
                None,
                " 0   1   2   3   4   ...  6   7   8   9   10\n"
                "  0   0   0   0   0  ...   0   0   0   0   0\n"
                "  0   0   0   0   0  ...   0   0   0   0   0\n"
                "  0   0   0   0   0  ...   0   0   0   0   0\n"
                "  0   0   0   0   0  ...   0   0   0   0   0",
            ),
            (
                None,
                2,
                " 0   1   2   3   4   5   6   7   8   9   10\n"
                "  0   0   0   0   0   0   0   0   0   0   0\n"
                " ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..\n"
                "  0   0   0   0   0   0   0   0   0   0   0",
            ),
            (
                10,
                2,
                " 0   1   2   3   4   ...  6   7   8   9   10\n"
                "  0   0   0   0   0  ...   0   0   0   0   0\n"
                " ..  ..  ..  ..  ..  ...  ..  ..  ..  ..  ..\n"
                "  0   0   0   0   0  ...   0   0   0   0   0",
            ),
            (
                9,
                2,
                " 0   1   2   3   ...  7   8   9   10\n"
                "  0   0   0   0  ...   0   0   0   0\n"
                " ..  ..  ..  ..  ...  ..  ..  ..  ..\n"
                "  0   0   0   0  ...   0   0   0   0",
            ),
            (
                1,
                1,
                " 0  ...\n 0  ...\n..  ...",
            ),
        ],
    )
    def test_truncation_no_index(self, max_cols, max_rows, expected):
        df = DataFrame([[0] * 11] * 4)
        assert (
            df.to_string(index=False, max_cols=max_cols, max_rows=max_rows) == expected
        )

    def test_to_string_no_index(self):
        # GH#16839, GH#13032
        df = DataFrame({"x": [11, 22], "y": [33, -44], "z": ["AAA", "   "]})

        df_s = df.to_string(index=False)
        # Leading space is expected for positive numbers.
        expected = " x   y   z\n11  33 AAA\n22 -44    "
        assert df_s == expected

        df_s = df[["y", "x", "z"]].to_string(index=False)
        expected = "  y  x   z\n 33 11 AAA\n-44 22    "
        assert df_s == expected

    def test_to_string_unicode_columns(self, float_frame):
        df = DataFrame({"\u03c3": np.arange(10.0)})

        buf = StringIO()
        df.to_string(buf=buf)
        buf.getvalue()

        buf = StringIO()
        df.info(buf=buf)
        buf.getvalue()

        result = float_frame.to_string()
        assert isinstance(result, str)

    @pytest.mark.parametrize("na_rep", ["NaN", "Ted"])
    def test_to_string_na_rep_and_float_format(self, na_rep):
        # GH#13828
        df = DataFrame([["A", 1.2225], ["A", None]], columns=["Group", "Data"])
        result = df.to_string(na_rep=na_rep, float_format="{:.2f}".format)
        expected = dedent(
            f"""\
               Group  Data
             0     A  1.22
             1     A   {na_rep}"""
        )
        assert result == expected

    def test_to_string_string_dtype(self):
        # GH#50099
        pytest.importorskip("pyarrow")
        df = DataFrame(
            {"x": ["foo", "bar", "baz"], "y": ["a", "b", "c"], "z": [1, 2, 3]}
        )
        df = df.astype(
            {"x": "string[pyarrow]", "y": "string[python]", "z": "int64[pyarrow]"}
        )
        result = df.dtypes.to_string()
        expected = dedent(
            """\
            x    string[pyarrow]
            y     string[python]
            z     int64[pyarrow]"""
        )
        assert result == expected

    def test_to_string_pos_args_deprecation(self):
        # GH#54229
        df = DataFrame({"a": [1, 2, 3]})
        msg = (
            "Starting with pandas version 3.0 all arguments of to_string "
            "except for the "
            "argument 'buf' will be keyword-only."
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            buf = StringIO()
            df.to_string(buf, None, None, True, True)

    def test_to_string_utf8_columns(self):
        n = "\u05d0".encode()
        df = DataFrame([1, 2], columns=[n])

        with option_context("display.max_rows", 1):
            repr(df)

    def test_to_string_unicode_two(self):
        dm = DataFrame({"c/\u03c3": []})
        buf = StringIO()
        dm.to_string(buf)

    def test_to_string_unicode_three(self):
        dm = DataFrame(["\xc2"])
        buf = StringIO()
        dm.to_string(buf)

    def test_to_string_with_float_index(self):
        index = Index([1.5, 2, 3, 4, 5])
        df = DataFrame(np.arange(5), index=index)

        result = df.to_string()
        expected = "     0\n1.5  0\n2.0  1\n3.0  2\n4.0  3\n5.0  4"
        assert result == expected

    def test_to_string(self):
        # big mixed
        biggie = DataFrame(
            {
                "A": np.random.default_rng(2).standard_normal(200),
                "B": Index([f"{i}?!" for i in range(200)]),
            },
        )

        biggie.loc[:20, "A"] = np.nan
        biggie.loc[:20, "B"] = np.nan
        s = biggie.to_string()

        buf = StringIO()
        retval = biggie.to_string(buf=buf)
        assert retval is None
        assert buf.getvalue() == s

        assert isinstance(s, str)

        # print in right order
        result = biggie.to_string(
            columns=["B", "A"], col_space=17, float_format="%.5f".__mod__
        )
        lines = result.split("\n")
        header = lines[0].strip().split()
        joined = "\n".join([re.sub(r"\s+", " ", x).strip() for x in lines[1:]])
        recons = read_csv(StringIO(joined), names=header, header=None, sep=" ")
        tm.assert_series_equal(recons["B"], biggie["B"])
        assert recons["A"].count() == biggie["A"].count()
        assert (np.abs(recons["A"].dropna() - biggie["A"].dropna()) < 0.1).all()

        # FIXME: don't leave commented-out
        # expected = ['B', 'A']
        # assert header == expected

        result = biggie.to_string(columns=["A"], col_space=17)
        header = result.split("\n")[0].strip().split()
        expected = ["A"]
        assert header == expected

        biggie.to_string(columns=["B", "A"], formatters={"A": lambda x: f"{x:.1f}"})

        biggie.to_string(columns=["B", "A"], float_format=str)
        biggie.to_string(columns=["B", "A"], col_space=12, float_format=str)

        frame = DataFrame(index=np.arange(200))
        frame.to_string()

    # TODO: split or simplify this test?
    @pytest.mark.xfail(using_string_dtype(), reason="fix when arrow is default")
    def test_to_string_index_with_nan(self):
        # GH#2850
        df = DataFrame(
            {
                "id1": {0: "1a3", 1: "9h4"},
                "id2": {0: np.nan, 1: "d67"},
                "id3": {0: "78d", 1: "79d"},
                "value": {0: 123, 1: 64},
            }
        )

        # multi-index
        y = df.set_index(["id1", "id2", "id3"])
        result = y.to_string()
        expected = (
            "             value\nid1 id2 id3       \n"
            "1a3 NaN 78d    123\n9h4 d67 79d     64"
        )
        assert result == expected

        # index
        y = df.set_index("id2")
        result = y.to_string()
        expected = (
            "     id1  id3  value\nid2                 \n"
            "NaN  1a3  78d    123\nd67  9h4  79d     64"
        )
        assert result == expected

        # with append (this failed in 0.12)
        y = df.set_index(["id1", "id2"]).set_index("id3", append=True)
        result = y.to_string()
        expected = (
            "             value\nid1 id2 id3       \n"
            "1a3 NaN 78d    123\n9h4 d67 79d     64"
        )
        assert result == expected

        # all-nan in mi
        df2 = df.copy()
        df2.loc[:, "id2"] = np.nan
        y = df2.set_index("id2")
        result = y.to_string()
        expected = (
            "     id1  id3  value\nid2                 \n"
            "NaN  1a3  78d    123\nNaN  9h4  79d     64"
        )
        assert result == expected

        # partial nan in mi
        df2 = df.copy()
        df2.loc[:, "id2"] = np.nan
        y = df2.set_index(["id2", "id3"])
        result = y.to_string()
        expected = (
            "         id1  value\nid2 id3            \n"
            "NaN 78d  1a3    123\n    79d  9h4     64"
        )
        assert result == expected

        df = DataFrame(
            {
                "id1": {0: np.nan, 1: "9h4"},
                "id2": {0: np.nan, 1: "d67"},
                "id3": {0: np.nan, 1: "79d"},
                "value": {0: 123, 1: 64},
            }
        )

        y = df.set_index(["id1", "id2", "id3"])
        result = y.to_string()
        expected = (
            "             value\nid1 id2 id3       \n"
            "NaN NaN NaN    123\n9h4 d67 79d     64"
        )
        assert result == expected

    def test_to_string_nonunicode_nonascii_alignment(self):
        df = DataFrame([["aa\xc3\xa4\xc3\xa4", 1], ["bbbb", 2]])
        rep_str = df.to_string()
        lines = rep_str.split("\n")
        assert len(lines[1]) == len(lines[2])

    def test_unicode_problem_decoding_as_ascii(self):
        df = DataFrame({"c/\u03c3": Series({"test": np.nan})})
        str(df.to_string())

    def test_to_string_repr_unicode(self):
        buf = StringIO()

        unicode_values = ["\u03c3"] * 10
        unicode_values = np.array(unicode_values, dtype=object)
        df = DataFrame({"unicode": unicode_values})
        df.to_string(col_space=10, buf=buf)

        # it works!
        repr(df)
        # it works even if sys.stdin in None
        _stdin = sys.stdin
        try:
            sys.stdin = None
            repr(df)
        finally:
            sys.stdin = _stdin


class TestSeriesToString:
    def test_to_string_without_index(self):
        # GH#11729 Test index=False option
        ser = Series([1, 2, 3, 4])
        result = ser.to_string(index=False)
        expected = "\n".join(["1", "2", "3", "4"])
        assert result == expected

    def test_to_string_name(self):
        ser = Series(range(100), dtype="int64")
        ser.name = "myser"
        res = ser.to_string(max_rows=2, name=True)
        exp = "0      0\n      ..\n99    99\nName: myser"
        assert res == exp
        res = ser.to_string(max_rows=2, name=False)
        exp = "0      0\n      ..\n99    99"
        assert res == exp

    def test_to_string_dtype(self):
        ser = Series(range(100), dtype="int64")
        res = ser.to_string(max_rows=2, dtype=True)
        exp = "0      0\n      ..\n99    99\ndtype: int64"
        assert res == exp
        res = ser.to_string(max_rows=2, dtype=False)
        exp = "0      0\n      ..\n99    99"
        assert res == exp

    def test_to_string_length(self):
        ser = Series(range(100), dtype="int64")
        res = ser.to_string(max_rows=2, length=True)
        exp = "0      0\n      ..\n99    99\nLength: 100"
        assert res == exp

    def test_to_string_na_rep(self):
        ser = Series(index=range(100), dtype=np.float64)
        res = ser.to_string(na_rep="foo", max_rows=2)
        exp = "0    foo\n      ..\n99   foo"
        assert res == exp

    def test_to_string_float_format(self):
        ser = Series(range(10), dtype="float64")
        res = ser.to_string(float_format=lambda x: f"{x:2.1f}", max_rows=2)
        exp = "0   0.0\n     ..\n9   9.0"
        assert res == exp

    def test_to_string_header(self):
        ser = Series(range(10), dtype="int64")
        ser.index.name = "foo"
        res = ser.to_string(header=True, max_rows=2)
        exp = "foo\n0    0\n    ..\n9    9"
        assert res == exp
        res = ser.to_string(header=False, max_rows=2)
        exp = "0    0\n    ..\n9    9"
        assert res == exp

    def test_to_string_empty_col(self):
        # GH#13653
        ser = Series(["", "Hello", "World", "", "", "Mooooo", "", ""])
        res = ser.to_string(index=False)
        exp = "      \n Hello\n World\n      \n      \nMooooo\n      \n      "
        assert re.match(exp, res)

    def test_to_string_timedelta64(self):
        Series(np.array([1100, 20], dtype="timedelta64[ns]")).to_string()

        ser = Series(date_range("2012-1-1", periods=3, freq="D"))

        # GH#2146

        # adding NaTs
        y = ser - ser.shift(1)
        result = y.to_string()
        assert "1 days" in result
        assert "00:00:00" not in result
        assert "NaT" in result

        # with frac seconds
        o = Series([datetime(2012, 1, 1, microsecond=150)] * 3)
        y = ser - o
        result = y.to_string()
        assert "-1 days +23:59:59.999850" in result

        # rounding?
        o = Series([datetime(2012, 1, 1, 1)] * 3)
        y = ser - o
        result = y.to_string()
        assert "-1 days +23:00:00" in result
        assert "1 days 23:00:00" in result

        o = Series([datetime(2012, 1, 1, 1, 1)] * 3)
        y = ser - o
        result = y.to_string()
        assert "-1 days +22:59:00" in result
        assert "1 days 22:59:00" in result

        o = Series([datetime(2012, 1, 1, 1, 1, microsecond=150)] * 3)
        y = ser - o
        result = y.to_string()
        assert "-1 days +22:58:59.999850" in result
        assert "0 days 22:58:59.999850" in result

        # neg time
        td = timedelta(minutes=5, seconds=3)
        s2 = Series(date_range("2012-1-1", periods=3, freq="D")) + td
        y = ser - s2
        result = y.to_string()
        assert "-1 days +23:54:57" in result

        td = timedelta(microseconds=550)
        s2 = Series(date_range("2012-1-1", periods=3, freq="D")) + td
        y = ser - td
        result = y.to_string()
        assert "2012-01-01 23:59:59.999450" in result

        # no boxing of the actual elements
        td = Series(timedelta_range("1 days", periods=3))
        result = td.to_string()
        assert result == "0   1 days\n1   2 days\n2   3 days"

    def test_to_string(self):
        ts = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10, freq="B"),
        )
        buf = StringIO()

        s = ts.to_string()

        retval = ts.to_string(buf=buf)
        assert retval is None
        assert buf.getvalue().strip() == s

        # pass float_format
        format = "%.4f".__mod__
        result = ts.to_string(float_format=format)
        result = [x.split()[1] for x in result.split("\n")[:-1]]
        expected = [format(x) for x in ts]
        assert result == expected

        # empty string
        result = ts[:0].to_string()
        assert result == "Series([], Freq: B)"

        result = ts[:0].to_string(length=0)
        assert result == "Series([], Freq: B)"

        # name and length
        cp = ts.copy()
        cp.name = "foo"
        result = cp.to_string(length=True, name=True, dtype=True)
        last_line = result.split("\n")[-1].strip()
        assert last_line == (f"Freq: B, Name: foo, Length: {len(cp)}, dtype: float64")

    @pytest.mark.parametrize(
        "input_array, expected",
        [
            ("a", "a"),
            (["a", "b"], "a\nb"),
            ([1, "a"], "1\na"),
            (1, "1"),
            ([0, -1], " 0\n-1"),
            (1.0, "1.0"),
            ([" a", " b"], " a\n b"),
            ([".1", "1"], ".1\n 1"),
            (["10", "-10"], " 10\n-10"),
        ],
    )
    def test_format_remove_leading_space_series(self, input_array, expected):
        # GH: 24980
        ser = Series(input_array)
        result = ser.to_string(index=False)
        assert result == expected

    def test_to_string_complex_number_trims_zeros(self):
        ser = Series([1.000000 + 1.000000j, 1.0 + 1.0j, 1.05 + 1.0j])
        result = ser.to_string()
        expected = dedent(
            """\
            0    1.00+1.00j
            1    1.00+1.00j
            2    1.05+1.00j"""
        )
        assert result == expected

    def test_nullable_float_to_string(self, float_ea_dtype):
        # https://github.com/pandas-dev/pandas/issues/36775
        dtype = float_ea_dtype
        ser = Series([0.0, 1.0, None], dtype=dtype)
        result = ser.to_string()
        expected = dedent(
            """\
            0     0.0
            1     1.0
            2    <NA>"""
        )
        assert result == expected

    def test_nullable_int_to_string(self, any_int_ea_dtype):
        # https://github.com/pandas-dev/pandas/issues/36775
        dtype = any_int_ea_dtype
        ser = Series([0, 1, None], dtype=dtype)
        result = ser.to_string()
        expected = dedent(
            """\
            0       0
            1       1
            2    <NA>"""
        )
        assert result == expected

    def test_to_string_mixed(self):
        ser = Series(["foo", np.nan, -1.23, 4.56])
        result = ser.to_string()
        expected = "".join(["0     foo\n", "1     NaN\n", "2   -1.23\n", "3    4.56"])
        assert result == expected

        # but don't count NAs as floats
        ser = Series(["foo", np.nan, "bar", "baz"])
        result = ser.to_string()
        expected = "".join(["0    foo\n", "1    NaN\n", "2    bar\n", "3    baz"])
        assert result == expected

        ser = Series(["foo", 5, "bar", "baz"])
        result = ser.to_string()
        expected = "".join(["0    foo\n", "1      5\n", "2    bar\n", "3    baz"])
        assert result == expected

    def test_to_string_float_na_spacing(self):
        ser = Series([0.0, 1.5678, 2.0, -3.0, 4.0])
        ser[::2] = np.nan

        result = ser.to_string()
        expected = (
            "0       NaN\n"
            "1    1.5678\n"
            "2       NaN\n"
            "3   -3.0000\n"
            "4       NaN"
        )
        assert result == expected

    def test_to_string_with_datetimeindex(self):
        index = date_range("20130102", periods=6)
        ser = Series(1, index=index)
        result = ser.to_string()
        assert "2013-01-02" in result

        # nat in index
        s2 = Series(2, index=[Timestamp("20130111"), NaT])
        ser = concat([s2, ser])
        result = ser.to_string()
        assert "NaT" in result

        # nat in summary
        result = str(s2.index)
        assert "NaT" in result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\any_pb2.py ===
# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: google/protobuf/any.proto
# Protobuf Python Version: 6.31.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    6,
    31,
    1,
    '',
    'google/protobuf/any.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x19google/protobuf/any.proto\x12\x0fgoogle.protobuf\"6\n\x03\x41ny\x12\x19\n\x08type_url\x18\x01 \x01(\tR\x07typeUrl\x12\x14\n\x05value\x18\x02 \x01(\x0cR\x05valueBv\n\x13\x63om.google.protobufB\x08\x41nyProtoP\x01Z,google.golang.org/protobuf/types/known/anypb\xa2\x02\x03GPB\xaa\x02\x1eGoogle.Protobuf.WellKnownTypesb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'google.protobuf.any_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\023com.google.protobufB\010AnyProtoP\001Z,google.golang.org/protobuf/types/known/anypb\242\002\003GPB\252\002\036Google.Protobuf.WellKnownTypes'
  _globals['_ANY']._serialized_start=46
  _globals['_ANY']._serialized_end=100
# @@protoc_insertion_point(module_scope)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\duration_pb2.py ===
# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: google/protobuf/duration.proto
# Protobuf Python Version: 6.31.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    6,
    31,
    1,
    '',
    'google/protobuf/duration.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1egoogle/protobuf/duration.proto\x12\x0fgoogle.protobuf\":\n\x08\x44uration\x12\x18\n\x07seconds\x18\x01 \x01(\x03R\x07seconds\x12\x14\n\x05nanos\x18\x02 \x01(\x05R\x05nanosB\x83\x01\n\x13\x63om.google.protobufB\rDurationProtoP\x01Z1google.golang.org/protobuf/types/known/durationpb\xf8\x01\x01\xa2\x02\x03GPB\xaa\x02\x1eGoogle.Protobuf.WellKnownTypesb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'google.protobuf.duration_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\023com.google.protobufB\rDurationProtoP\001Z1google.golang.org/protobuf/types/known/durationpb\370\001\001\242\002\003GPB\252\002\036Google.Protobuf.WellKnownTypes'
  _globals['_DURATION']._serialized_start=51
  _globals['_DURATION']._serialized_end=109
# @@protoc_insertion_point(module_scope)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\empty_pb2.py ===
# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: google/protobuf/empty.proto
# Protobuf Python Version: 6.31.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    6,
    31,
    1,
    '',
    'google/protobuf/empty.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1bgoogle/protobuf/empty.proto\x12\x0fgoogle.protobuf\"\x07\n\x05\x45mptyB}\n\x13\x63om.google.protobufB\nEmptyProtoP\x01Z.google.golang.org/protobuf/types/known/emptypb\xf8\x01\x01\xa2\x02\x03GPB\xaa\x02\x1eGoogle.Protobuf.WellKnownTypesb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'google.protobuf.empty_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\023com.google.protobufB\nEmptyProtoP\001Z.google.golang.org/protobuf/types/known/emptypb\370\001\001\242\002\003GPB\252\002\036Google.Protobuf.WellKnownTypes'
  _globals['_EMPTY']._serialized_start=48
  _globals['_EMPTY']._serialized_end=55
# @@protoc_insertion_point(module_scope)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\field_mask_pb2.py ===
# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: google/protobuf/field_mask.proto
# Protobuf Python Version: 6.31.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    6,
    31,
    1,
    '',
    'google/protobuf/field_mask.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n google/protobuf/field_mask.proto\x12\x0fgoogle.protobuf\"!\n\tFieldMask\x12\x14\n\x05paths\x18\x01 \x03(\tR\x05pathsB\x85\x01\n\x13\x63om.google.protobufB\x0e\x46ieldMaskProtoP\x01Z2google.golang.org/protobuf/types/known/fieldmaskpb\xf8\x01\x01\xa2\x02\x03GPB\xaa\x02\x1eGoogle.Protobuf.WellKnownTypesb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'google.protobuf.field_mask_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\023com.google.protobufB\016FieldMaskProtoP\001Z2google.golang.org/protobuf/types/known/fieldmaskpb\370\001\001\242\002\003GPB\252\002\036Google.Protobuf.WellKnownTypes'
  _globals['_FIELDMASK']._serialized_start=53
  _globals['_FIELDMASK']._serialized_end=86
# @@protoc_insertion_point(module_scope)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\source_context_pb2.py ===
# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: google/protobuf/source_context.proto
# Protobuf Python Version: 6.31.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    6,
    31,
    1,
    '',
    'google/protobuf/source_context.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n$google/protobuf/source_context.proto\x12\x0fgoogle.protobuf\",\n\rSourceContext\x12\x1b\n\tfile_name\x18\x01 \x01(\tR\x08\x66ileNameB\x8a\x01\n\x13\x63om.google.protobufB\x12SourceContextProtoP\x01Z6google.golang.org/protobuf/types/known/sourcecontextpb\xa2\x02\x03GPB\xaa\x02\x1eGoogle.Protobuf.WellKnownTypesb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'google.protobuf.source_context_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\023com.google.protobufB\022SourceContextProtoP\001Z6google.golang.org/protobuf/types/known/sourcecontextpb\242\002\003GPB\252\002\036Google.Protobuf.WellKnownTypes'
  _globals['_SOURCECONTEXT']._serialized_start=57
  _globals['_SOURCECONTEXT']._serialized_end=101
# @@protoc_insertion_point(module_scope)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\timestamp_pb2.py ===
# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: google/protobuf/timestamp.proto
# Protobuf Python Version: 6.31.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    6,
    31,
    1,
    '',
    'google/protobuf/timestamp.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1fgoogle/protobuf/timestamp.proto\x12\x0fgoogle.protobuf\";\n\tTimestamp\x12\x18\n\x07seconds\x18\x01 \x01(\x03R\x07seconds\x12\x14\n\x05nanos\x18\x02 \x01(\x05R\x05nanosB\x85\x01\n\x13\x63om.google.protobufB\x0eTimestampProtoP\x01Z2google.golang.org/protobuf/types/known/timestamppb\xf8\x01\x01\xa2\x02\x03GPB\xaa\x02\x1eGoogle.Protobuf.WellKnownTypesb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'google.protobuf.timestamp_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\023com.google.protobufB\016TimestampProtoP\001Z2google.golang.org/protobuf/types/known/timestamppb\370\001\001\242\002\003GPB\252\002\036Google.Protobuf.WellKnownTypes'
  _globals['_TIMESTAMP']._serialized_start=52
  _globals['_TIMESTAMP']._serialized_end=111
# @@protoc_insertion_point(module_scope)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\qdq_helpers\optimize_qdq_model.py ===
#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import pathlib

import onnx


def optimize_qdq_model():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="Update a QDQ format ONNX model to ensure optimal performance when executed using ONNX Runtime.",
    )

    parser.add_argument("input_model", type=pathlib.Path, help="Provide path to ONNX model to update.")
    parser.add_argument("output_model", type=pathlib.Path, help="Provide path to write updated ONNX model to.")

    args = parser.parse_args()

    model = onnx.load(str(args.input_model.resolve(strict=True)))

    # run QDQ model optimizations here

    # Originally, the fixing up of DQ nodes with multiple consumers was implemented as one such optimization.
    # That was moved to an ORT graph transformer.
    print("As of ORT 1.15, the fixing up of DQ nodes with multiple consumers is done by an ORT graph transformer.")

    # There are no optimizations being run currently but we expect that there may be in the future.

    onnx.save(model, str(args.output_model.resolve()))


if __name__ == "__main__":
    optimize_qdq_model()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\compat\pyarrow.py ===
""" support pyarrow compatibility across versions """

from __future__ import annotations

from pandas.util.version import Version

try:
    import pyarrow as pa

    _palv = Version(Version(pa.__version__).base_version)
    pa_version_under10p1 = _palv < Version("10.0.1")
    pa_version_under11p0 = _palv < Version("11.0.0")
    pa_version_under12p0 = _palv < Version("12.0.0")
    pa_version_under13p0 = _palv < Version("13.0.0")
    pa_version_under14p0 = _palv < Version("14.0.0")
    pa_version_under14p1 = _palv < Version("14.0.1")
    pa_version_under15p0 = _palv < Version("15.0.0")
    pa_version_under16p0 = _palv < Version("16.0.0")
    pa_version_under17p0 = _palv < Version("17.0.0")
    pa_version_under18p0 = _palv < Version("18.0.0")
    pa_version_under19p0 = _palv < Version("19.0.0")
    pa_version_under20p0 = _palv < Version("20.0.0")
    HAS_PYARROW = True
except ImportError:
    pa_version_under10p1 = True
    pa_version_under11p0 = True
    pa_version_under12p0 = True
    pa_version_under13p0 = True
    pa_version_under14p0 = True
    pa_version_under14p1 = True
    pa_version_under15p0 = True
    pa_version_under16p0 = True
    pa_version_under17p0 = True
    pa_version_under18p0 = True
    pa_version_under19p0 = True
    pa_version_under20p0 = True
    HAS_PYARROW = False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\operations\build\wheel.py ===
import logging
import os
from typing import Optional

from pip._vendor.pyproject_hooks import BuildBackendHookCaller

from pip._internal.utils.subprocess import runner_with_spinner_message

logger = logging.getLogger(__name__)


def build_wheel_pep517(
    name: str,
    backend: BuildBackendHookCaller,
    metadata_directory: str,
    tempd: str,
) -> Optional[str]:
    """Build one InstallRequirement using the PEP 517 build process.

    Returns path to wheel if successfully built. Otherwise, returns None.
    """
    assert metadata_directory is not None
    try:
        logger.debug("Destination directory: %s", tempd)

        runner = runner_with_spinner_message(
            f"Building wheel for {name} (pyproject.toml)"
        )
        with backend.subprocess_runner(runner):
            wheel_name = backend.build_wheel(
                tempd,
                metadata_directory=metadata_directory,
            )
    except Exception:
        logger.error("Failed building wheel for %s", name)
        return None
    return os.path.join(tempd, wheel_name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\constrain.py ===
from typing import Optional, TYPE_CHECKING

from .jupyter import JupyterMixin
from .measure import Measurement

if TYPE_CHECKING:
    from .console import Console, ConsoleOptions, RenderableType, RenderResult


class Constrain(JupyterMixin):
    """Constrain the width of a renderable to a given number of characters.

    Args:
        renderable (RenderableType): A renderable object.
        width (int, optional): The maximum width (in characters) to render. Defaults to 80.
    """

    def __init__(self, renderable: "RenderableType", width: Optional[int] = 80) -> None:
        self.renderable = renderable
        self.width = width

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "RenderResult":
        if self.width is None:
            yield self.renderable
        else:
            child_options = options.update_width(min(self.width, options.max_width))
            yield from console.render(self.renderable, child_options)

    def __rich_measure__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "Measurement":
        if self.width is not None:
            options = options.update_width(self.width)
        measurement = Measurement.get(console, options, self.renderable)
        return measurement

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\__init__.py ===
# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2003-2006 Gary Bishop.
#       Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#       Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************


from platform import system
from importlib.metadata import version, PackageNotFoundError

from . import (
    clipboard,
    console,
    lineeditor,
    logger,
    modes,
    rlmain,
    unicode_helper,
)
from .rlmain import *

_S = system()

if _S.lower() != "windows":
    raise RuntimeError("pyreadline3 is for Windows only, not {}.".format(_S))

del system, _S

try:
    __version__ = version("pyreadline3")
except PackageNotFoundError:
    # package is not installed
    pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\console\event.py ===
class Event(object):
    """Represent events from the console."""

    def __init__(self, console, input):
        pass

    def __repr__(self):
        """Display an event for debugging."""
        if self.type in ["KeyPress", "KeyRelease"]:
            chr = self.char
            if ord(chr) < ord("A"):
                chr = "?"
            s = "%s char='%s'%d keysym='%s' keycode=%d:%x state=%x keyinfo=%s" % (
                self.type,
                chr,
                ord(self.char),
                self.keysym,
                self.keycode,
                self.keycode,
                self.state,
                self.keyinfo,
            )
        elif self.type in ["Motion", "Button"]:
            s = "%s x=%d y=%d state=%x" % (self.type, self.x, self.y, self.state)
        elif self.type == "Configure":
            s = "%s w=%d h=%d" % (self.type, self.width, self.height)
        elif self.type in ["FocusIn", "FocusOut"]:
            s = self.type
        elif self.type == "Menu":
            s = "%s state=%x" % (self.type, self.state)
        else:
            s = "unknown event type"
        return s


#    def __str__(self):
#        return "('%s',%s,%s,%s)"%(self.char,self.key,self.state,self.keyinfo)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\actions\input_device.py ===
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

import uuid
from typing import Any, List, Optional


class InputDevice:
    """Describes the input device being used for the action."""

    def __init__(self, name: Optional[str] = None):
        self.name = name or uuid.uuid4()
        self.actions: List[Any] = []

    def add_action(self, action: Any) -> None:
        """"""
        self.actions.append(action)

    def clear_actions(self) -> None:
        self.actions = []

    def create_pause(self, duration: float = 0) -> None:
        pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\autolev\test-examples\ruletest3.py ===
import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

frame_a = _me.ReferenceFrame('a')
frame_b = _me.ReferenceFrame('b')
frame_n = _me.ReferenceFrame('n')
x1, x2, x3 = _me.dynamicsymbols('x1 x2 x3')
l = _sm.symbols('l', real=True)
v1 = x1*frame_a.x+x2*frame_a.y+x3*frame_a.z
v2 = x1*frame_b.x+x2*frame_b.y+x3*frame_b.z
v3 = x1*frame_n.x+x2*frame_n.y+x3*frame_n.z
v = v1+v2+v3
point_c = _me.Point('c')
point_d = _me.Point('d')
point_po1 = _me.Point('po1')
point_po2 = _me.Point('po2')
point_po3 = _me.Point('po3')
particle_l = _me.Particle('l', _me.Point('l_pt'), _sm.Symbol('m'))
particle_p1 = _me.Particle('p1', _me.Point('p1_pt'), _sm.Symbol('m'))
particle_p2 = _me.Particle('p2', _me.Point('p2_pt'), _sm.Symbol('m'))
particle_p3 = _me.Particle('p3', _me.Point('p3_pt'), _sm.Symbol('m'))
body_s_cm = _me.Point('s_cm')
body_s_cm.set_vel(frame_n, 0)
body_s_f = _me.ReferenceFrame('s_f')
body_s = _me.RigidBody('s', body_s_cm, body_s_f, _sm.symbols('m'), (_me.outer(body_s_f.x,body_s_f.x),body_s_cm))
body_r1_cm = _me.Point('r1_cm')
body_r1_cm.set_vel(frame_n, 0)
body_r1_f = _me.ReferenceFrame('r1_f')
body_r1 = _me.RigidBody('r1', body_r1_cm, body_r1_f, _sm.symbols('m'), (_me.outer(body_r1_f.x,body_r1_f.x),body_r1_cm))
body_r2_cm = _me.Point('r2_cm')
body_r2_cm.set_vel(frame_n, 0)
body_r2_f = _me.ReferenceFrame('r2_f')
body_r2 = _me.RigidBody('r2', body_r2_cm, body_r2_f, _sm.symbols('m'), (_me.outer(body_r2_f.x,body_r2_f.x),body_r2_cm))
v4 = x1*body_s_f.x+x2*body_s_f.y+x3*body_s_f.z
body_s_cm.set_pos(point_c, l*frame_n.x)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\strategies\traverse.py ===
"""Strategies to Traverse a Tree."""
from sympy.strategies.util import basic_fns
from sympy.strategies.core import chain, do_one


def top_down(rule, fns=basic_fns):
    """Apply a rule down a tree running it on the top nodes first."""
    return chain(rule, lambda expr: sall(top_down(rule, fns), fns)(expr))


def bottom_up(rule, fns=basic_fns):
    """Apply a rule down a tree running it on the bottom nodes first."""
    return chain(lambda expr: sall(bottom_up(rule, fns), fns)(expr), rule)


def top_down_once(rule, fns=basic_fns):
    """Apply a rule down a tree - stop on success."""
    return do_one(rule, lambda expr: sall(top_down(rule, fns), fns)(expr))


def bottom_up_once(rule, fns=basic_fns):
    """Apply a rule up a tree - stop on success."""
    return do_one(lambda expr: sall(bottom_up(rule, fns), fns)(expr), rule)


def sall(rule, fns=basic_fns):
    """Strategic all - apply rule to args."""
    op, new, children, leaf = map(fns.get, ('op', 'new', 'children', 'leaf'))

    def all_rl(expr):
        if leaf(expr):
            return expr
        else:
            args = map(rule, children(expr))
            return new(op(expr), *args)

    return all_rl

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_str.py ===
from sympy import MatAdd
from sympy.algebras.quaternion import Quaternion
from sympy.assumptions.ask import Q
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.combinatorics.partitions import Partition
from sympy.concrete.summations import (Sum, summation)
from sympy.core.add import Add
from sympy.core.containers import (Dict, Tuple)
from sympy.core.expr import UnevaluatedExpr, Expr
from sympy.core.function import (Derivative, Function, Lambda, Subs, WildFunction)
from sympy.core.mul import Mul
from sympy.core import (Catalan, EulerGamma, GoldenRatio, TribonacciConstant)
from sympy.core.numbers import (E, Float, I, Integer, Rational, nan, oo, pi, zoo)
from sympy.core.parameters import _exp_is_pow
from sympy.core.power import Pow
from sympy.core.relational import (Eq, Rel, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, Wild, symbols)
from sympy.functions.combinatorial.factorials import (factorial, factorial2, subfactorial)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.delta_functions import Heaviside
from sympy.functions.special.zeta_functions import zeta
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (Equivalent, false, true, Xor)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions import Identity
from sympy.matrices.expressions.slice import MatrixSlice
from sympy.matrices import SparseMatrix
from sympy.polys.polytools import factor
from sympy.series.limits import Limit
from sympy.series.order import O
from sympy.sets.sets import (Complement, FiniteSet, Interval, SymmetricDifference)
from sympy.stats import (Covariance, Expectation, Probability, Variance)
from sympy.stats.rv import RandomSymbol
from sympy.external import import_module
from sympy.physics.control.lti import TransferFunction, Series, Parallel, \
    Feedback, TransferFunctionMatrix, MIMOSeries, MIMOParallel, MIMOFeedback
from sympy.physics.units import second, joule
from sympy.polys import (Poly, rootof, RootSum, groebner, ring, field, ZZ, QQ,
    ZZ_I, QQ_I, lex, grlex)
from sympy.geometry import Point, Circle, Polygon, Ellipse, Triangle
from sympy.tensor import NDimArray
from sympy.tensor.array.expressions.array_expressions import ArraySymbol, ArrayElement

from sympy.testing.pytest import raises, warns_deprecated_sympy

from sympy.printing import sstr, sstrrepr, StrPrinter
from sympy.physics.quantum.trace import Tr

x, y, z, w, t = symbols('x,y,z,w,t')
d = Dummy('d')


def test_printmethod():
    class R(Abs):
        def _sympystr(self, printer):
            return "foo(%s)" % printer._print(self.args[0])
    assert sstr(R(x)) == "foo(x)"

    class R(Abs):
        def _sympystr(self, printer):
            return "foo"
    assert sstr(R(x)) == "foo"


def test_Abs():
    assert str(Abs(x)) == "Abs(x)"
    assert str(Abs(Rational(1, 6))) == "1/6"
    assert str(Abs(Rational(-1, 6))) == "1/6"


def test_Add():
    assert str(x + y) == "x + y"
    assert str(x + 1) == "x + 1"
    assert str(x + x**2) == "x**2 + x"
    assert str(Add(0, 1, evaluate=False)) == "0 + 1"
    assert str(Add(0, 0, 1, evaluate=False)) == "0 + 0 + 1"
    assert str(1.0*x) == "1.0*x"
    assert str(5 + x + y + x*y + x**2 + y**2) == "x**2 + x*y + x + y**2 + y + 5"
    assert str(1 + x + x**2/2 + x**3/3) == "x**3/3 + x**2/2 + x + 1"
    assert str(2*x - 7*x**2 + 2 + 3*y) == "-7*x**2 + 2*x + 3*y + 2"
    assert str(x - y) == "x - y"
    assert str(2 - x) == "2 - x"
    assert str(x - 2) == "x - 2"
    assert str(x - y - z - w) == "-w + x - y - z"
    assert str(x - z*y**2*z*w) == "-w*y**2*z**2 + x"
    assert str(x - 1*y*x*y) == "-x*y**2 + x"
    assert str(sin(x).series(x, 0, 15)) == "x - x**3/6 + x**5/120 - x**7/5040 + x**9/362880 - x**11/39916800 + x**13/6227020800 + O(x**15)"
    assert str(Add(Add(-w, x, evaluate=False), Add(-y, z,  evaluate=False),  evaluate=False)) == "(-w + x) + (-y + z)"
    assert str(Add(Add(-x, -y, evaluate=False), -z, evaluate=False)) == "-z + (-x - y)"
    assert str(Add(Add(Add(-x, -y, evaluate=False), -z, evaluate=False), -t, evaluate=False)) == "-t + (-z + (-x - y))"


def test_Catalan():
    assert str(Catalan) == "Catalan"


def test_ComplexInfinity():
    assert str(zoo) == "zoo"


def test_Derivative():
    assert str(Derivative(x, y)) == "Derivative(x, y)"
    assert str(Derivative(x**2, x, evaluate=False)) == "Derivative(x**2, x)"
    assert str(Derivative(
        x**2/y, x, y, evaluate=False)) == "Derivative(x**2/y, x, y)"


def test_dict():
    assert str({1: 1 + x}) == sstr({1: 1 + x}) == "{1: x + 1}"
    assert str({1: x**2, 2: y*x}) in ("{1: x**2, 2: x*y}", "{2: x*y, 1: x**2}")
    assert sstr({1: x**2, 2: y*x}) == "{1: x**2, 2: x*y}"


def test_Dict():
    assert str(Dict({1: 1 + x})) == sstr({1: 1 + x}) == "{1: x + 1}"
    assert str(Dict({1: x**2, 2: y*x})) in (
        "{1: x**2, 2: x*y}", "{2: x*y, 1: x**2}")
    assert sstr(Dict({1: x**2, 2: y*x})) == "{1: x**2, 2: x*y}"


def test_Dummy():
    assert str(d) == "_d"
    assert str(d + x) == "_d + x"


def test_EulerGamma():
    assert str(EulerGamma) == "EulerGamma"


def test_Exp():
    assert str(E) == "E"
    with _exp_is_pow(True):
        assert str(exp(x)) == "E**x"


def test_factorial():
    n = Symbol('n', integer=True)
    assert str(factorial(-2)) == "zoo"
    assert str(factorial(0)) == "1"
    assert str(factorial(7)) == "5040"
    assert str(factorial(n)) == "factorial(n)"
    assert str(factorial(2*n)) == "factorial(2*n)"
    assert str(factorial(factorial(n))) == 'factorial(factorial(n))'
    assert str(factorial(factorial2(n))) == 'factorial(factorial2(n))'
    assert str(factorial2(factorial(n))) == 'factorial2(factorial(n))'
    assert str(factorial2(factorial2(n))) == 'factorial2(factorial2(n))'
    assert str(subfactorial(3)) == "2"
    assert str(subfactorial(n)) == "subfactorial(n)"
    assert str(subfactorial(2*n)) == "subfactorial(2*n)"


def test_Function():
    f = Function('f')
    fx = f(x)
    w = WildFunction('w')
    assert str(f) == "f"
    assert str(fx) == "f(x)"
    assert str(w) == "w_"


def test_Geometry():
    assert sstr(Point(0, 0)) == 'Point2D(0, 0)'
    assert sstr(Circle(Point(0, 0), 3)) == 'Circle(Point2D(0, 0), 3)'
    assert sstr(Ellipse(Point(1, 2), 3, 4)) == 'Ellipse(Point2D(1, 2), 3, 4)'
    assert sstr(Triangle(Point(1, 1), Point(7, 8), Point(0, -1))) == \
        'Triangle(Point2D(1, 1), Point2D(7, 8), Point2D(0, -1))'
    assert sstr(Polygon(Point(5, 6), Point(-2, -3), Point(0, 0), Point(4, 7))) == \
        'Polygon(Point2D(5, 6), Point2D(-2, -3), Point2D(0, 0), Point2D(4, 7))'
    assert sstr(Triangle(Point(0, 0), Point(1, 0), Point(0, 1)), sympy_integers=True) == \
        'Triangle(Point2D(S(0), S(0)), Point2D(S(1), S(0)), Point2D(S(0), S(1)))'
    assert sstr(Ellipse(Point(1, 2), 3, 4), sympy_integers=True) == \
        'Ellipse(Point2D(S(1), S(2)), S(3), S(4))'


def test_GoldenRatio():
    assert str(GoldenRatio) == "GoldenRatio"


def test_Heaviside():
    assert str(Heaviside(x)) == str(Heaviside(x, S.Half)) == "Heaviside(x)"
    assert str(Heaviside(x, 1)) == "Heaviside(x, 1)"


def test_TribonacciConstant():
    assert str(TribonacciConstant) == "TribonacciConstant"


def test_ImaginaryUnit():
    assert str(I) == "I"


def test_Infinity():
    assert str(oo) == "oo"
    assert str(oo*I) == "oo*I"


def test_Integer():
    assert str(Integer(-1)) == "-1"
    assert str(Integer(1)) == "1"
    assert str(Integer(-3)) == "-3"
    assert str(Integer(0)) == "0"
    assert str(Integer(25)) == "25"


def test_Integral():
    assert str(Integral(sin(x), y)) == "Integral(sin(x), y)"
    assert str(Integral(sin(x), (y, 0, 1))) == "Integral(sin(x), (y, 0, 1))"


def test_Interval():
    n = (S.NegativeInfinity, 1, 2, S.Infinity)
    for i in range(len(n)):
        for j in range(i + 1, len(n)):
            for l in (True, False):
                for r in (True, False):
                    ival = Interval(n[i], n[j], l, r)
                    assert S(str(ival)) == ival


def test_AccumBounds():
    a = Symbol('a', real=True)
    assert str(AccumBounds(0, a)) == "AccumBounds(0, a)"
    assert str(AccumBounds(0, 1)) == "AccumBounds(0, 1)"


def test_Lambda():
    assert str(Lambda(d, d**2)) == "Lambda(_d, _d**2)"
    # issue 2908
    assert str(Lambda((), 1)) == "Lambda((), 1)"
    assert str(Lambda((), x)) == "Lambda((), x)"
    assert str(Lambda((x, y), x+y)) == "Lambda((x, y), x + y)"
    assert str(Lambda(((x, y),), x+y)) == "Lambda(((x, y),), x + y)"


def test_Limit():
    assert str(Limit(sin(x)/x, x, y)) == "Limit(sin(x)/x, x, y, dir='+')"
    assert str(Limit(1/x, x, 0)) == "Limit(1/x, x, 0, dir='+')"
    assert str(
        Limit(sin(x)/x, x, y, dir="-")) == "Limit(sin(x)/x, x, y, dir='-')"


def test_list():
    assert str([x]) == sstr([x]) == "[x]"
    assert str([x**2, x*y + 1]) == sstr([x**2, x*y + 1]) == "[x**2, x*y + 1]"
    assert str([x**2, [y + x]]) == sstr([x**2, [y + x]]) == "[x**2, [x + y]]"


def test_Matrix_str():
    M = Matrix([[x**+1, 1], [y, x + y]])
    assert str(M) == "Matrix([[x, 1], [y, x + y]])"
    assert sstr(M) == "Matrix([\n[x,     1],\n[y, x + y]])"
    M = Matrix([[1]])
    assert str(M) == sstr(M) == "Matrix([[1]])"
    M = Matrix([[1, 2]])
    assert str(M) == sstr(M) ==  "Matrix([[1, 2]])"
    M = Matrix()
    assert str(M) == sstr(M) == "Matrix(0, 0, [])"
    M = Matrix(0, 1, lambda i, j: 0)
    assert str(M) == sstr(M) == "Matrix(0, 1, [])"


def test_Mul():
    assert str(x/y) == "x/y"
    assert str(y/x) == "y/x"
    assert str(x/y/z) == "x/(y*z)"
    assert str((x + 1)/(y + 2)) == "(x + 1)/(y + 2)"
    assert str(2*x/3) == '2*x/3'
    assert str(-2*x/3) == '-2*x/3'
    assert str(-1.0*x) == '-1.0*x'
    assert str(1.0*x) == '1.0*x'
    assert str(Mul(0, 1, evaluate=False)) == '0*1'
    assert str(Mul(1, 0, evaluate=False)) == '1*0'
    assert str(Mul(1, 1, evaluate=False)) == '1*1'
    assert str(Mul(1, 1, 1, evaluate=False)) == '1*1*1'
    assert str(Mul(1, 2, evaluate=False)) == '1*2'
    assert str(Mul(1, S.Half, evaluate=False)) == '1*(1/2)'
    assert str(Mul(1, 1, S.Half, evaluate=False)) == '1*1*(1/2)'
    assert str(Mul(1, 1, 2, 3, x, evaluate=False)) == '1*1*2*3*x'
    assert str(Mul(1, -1, evaluate=False)) == '1*(-1)'
    assert str(Mul(-1, 1, evaluate=False)) == '-1*1'
    assert str(Mul(4, 3, 2, 1, 0, y, x, evaluate=False)) == '4*3*2*1*0*y*x'
    assert str(Mul(4, 3, 2, 1+z, 0, y, x, evaluate=False)) == '4*3*2*(z + 1)*0*y*x'
    assert str(Mul(Rational(2, 3), Rational(5, 7), evaluate=False)) == '(2/3)*(5/7)'
    # For issue 14160
    assert str(Mul(-2, x, Pow(Mul(y,y,evaluate=False), -1, evaluate=False),
                                                evaluate=False)) == '-2*x/(y*y)'
    # issue 21537
    assert str(Mul(x, Pow(1/y, -1, evaluate=False), evaluate=False)) == 'x/(1/y)'

    # Issue 24108
    from sympy.core.parameters import evaluate
    with evaluate(False):
        assert str(Mul(Pow(Integer(2), Integer(-1)), Add(Integer(-1), Mul(Integer(-1), Integer(1))))) == "(-1 - 1*1)/2"

    class CustomClass1(Expr):
        is_commutative = True

    class CustomClass2(Expr):
        is_commutative = True
    cc1 = CustomClass1()
    cc2 = CustomClass2()
    assert str(Rational(2)*cc1) == '2*CustomClass1()'
    assert str(cc1*Rational(2)) == '2*CustomClass1()'
    assert str(cc1*Float("1.5")) == '1.5*CustomClass1()'
    assert str(cc2*Rational(2)) == '2*CustomClass2()'
    assert str(cc2*Rational(2)*cc1) == '2*CustomClass1()*CustomClass2()'
    assert str(cc1*Rational(2)*cc2) == '2*CustomClass1()*CustomClass2()'


def test_NaN():
    assert str(nan) == "nan"


def test_NegativeInfinity():
    assert str(-oo) == "-oo"

def test_Order():
    assert str(O(x)) == "O(x)"
    assert str(O(x**2)) == "O(x**2)"
    assert str(O(x*y)) == "O(x*y, x, y)"
    assert str(O(x, x)) == "O(x)"
    assert str(O(x, (x, 0))) == "O(x)"
    assert str(O(x, (x, oo))) == "O(x, (x, oo))"
    assert str(O(x, x, y)) == "O(x, x, y)"
    assert str(O(x, x, y)) == "O(x, x, y)"
    assert str(O(x, (x, oo), (y, oo))) == "O(x, (x, oo), (y, oo))"


def test_Permutation_Cycle():
    from sympy.combinatorics import Permutation, Cycle

    # general principle: economically, canonically show all moved elements
    # and the size of the permutation.

    for p, s in [
        (Cycle(),
        '()'),
        (Cycle(2),
        '(2)'),
        (Cycle(2, 1),
        '(1 2)'),
        (Cycle(1, 2)(5)(6, 7)(10),
        '(1 2)(6 7)(10)'),
        (Cycle(3, 4)(1, 2)(3, 4),
        '(1 2)(4)'),
    ]:
        assert sstr(p) == s

    for p, s in [
        (Permutation([]),
        'Permutation([])'),
        (Permutation([], size=1),
        'Permutation([0])'),
        (Permutation([], size=2),
        'Permutation([0, 1])'),
        (Permutation([], size=10),
        'Permutation([], size=10)'),
        (Permutation([1, 0, 2]),
        'Permutation([1, 0, 2])'),
        (Permutation([1, 0, 2, 3, 4, 5]),
        'Permutation([1, 0], size=6)'),
        (Permutation([1, 0, 2, 3, 4, 5], size=10),
        'Permutation([1, 0], size=10)'),
    ]:
        assert sstr(p, perm_cyclic=False) == s

    for p, s in [
        (Permutation([]),
        '()'),
        (Permutation([], size=1),
        '(0)'),
        (Permutation([], size=2),
        '(1)'),
        (Permutation([], size=10),
        '(9)'),
        (Permutation([1, 0, 2]),
        '(2)(0 1)'),
        (Permutation([1, 0, 2, 3, 4, 5]),
        '(5)(0 1)'),
        (Permutation([1, 0, 2, 3, 4, 5], size=10),
        '(9)(0 1)'),
        (Permutation([0, 1, 3, 2, 4, 5], size=10),
        '(9)(2 3)'),
    ]:
        assert sstr(p) == s


    with warns_deprecated_sympy():
        old_print_cyclic = Permutation.print_cyclic
        Permutation.print_cyclic = False
        assert sstr(Permutation([1, 0, 2])) == 'Permutation([1, 0, 2])'
        Permutation.print_cyclic = old_print_cyclic

def test_Pi():
    assert str(pi) == "pi"


def test_Poly():
    assert str(Poly(0, x)) == "Poly(0, x, domain='ZZ')"
    assert str(Poly(1, x)) == "Poly(1, x, domain='ZZ')"
    assert str(Poly(x, x)) == "Poly(x, x, domain='ZZ')"

    assert str(Poly(2*x + 1, x)) == "Poly(2*x + 1, x, domain='ZZ')"
    assert str(Poly(2*x - 1, x)) == "Poly(2*x - 1, x, domain='ZZ')"

    assert str(Poly(-1, x)) == "Poly(-1, x, domain='ZZ')"
    assert str(Poly(-x, x)) == "Poly(-x, x, domain='ZZ')"

    assert str(Poly(-2*x + 1, x)) == "Poly(-2*x + 1, x, domain='ZZ')"
    assert str(Poly(-2*x - 1, x)) == "Poly(-2*x - 1, x, domain='ZZ')"

    assert str(Poly(x - 1, x)) == "Poly(x - 1, x, domain='ZZ')"
    assert str(Poly(2*x + x**5, x)) == "Poly(x**5 + 2*x, x, domain='ZZ')"

    assert str(Poly(3**(2*x), 3**x)) == "Poly((3**x)**2, 3**x, domain='ZZ')"
    assert str(Poly((x**2)**x)) == "Poly(((x**2)**x), (x**2)**x, domain='ZZ')"

    assert str(Poly((x + y)**3, (x + y), expand=False)
                ) == "Poly((x + y)**3, x + y, domain='ZZ')"
    assert str(Poly((x - 1)**2, (x - 1), expand=False)
                ) == "Poly((x - 1)**2, x - 1, domain='ZZ')"

    assert str(
        Poly(x**2 + 1 + y, x)) == "Poly(x**2 + y + 1, x, domain='ZZ[y]')"
    assert str(
        Poly(x**2 - 1 + y, x)) == "Poly(x**2 + y - 1, x, domain='ZZ[y]')"

    assert str(Poly(x**2 + I*x, x)) == "Poly(x**2 + I*x, x, domain='ZZ_I')"
    assert str(Poly(x**2 - I*x, x)) == "Poly(x**2 - I*x, x, domain='ZZ_I')"

    assert str(Poly(-x*y*z + x*y - 1, x, y, z)
               ) == "Poly(-x*y*z + x*y - 1, x, y, z, domain='ZZ')"
    assert str(Poly(-w*x**21*y**7*z + (1 + w)*z**3 - 2*x*z + 1, x, y, z)) == \
        "Poly(-w*x**21*y**7*z - 2*x*z + (w + 1)*z**3 + 1, x, y, z, domain='ZZ[w]')"

    assert str(Poly(x**2 + 1, x, modulus=2)) == "Poly(x**2 + 1, x, modulus=2)"
    assert str(Poly(2*x**2 + 3*x + 4, x, modulus=17)) == "Poly(2*x**2 + 3*x + 4, x, modulus=17)"


def test_PolyRing():
    assert str(ring("x", ZZ, lex)[0]) == "Polynomial ring in x over ZZ with lex order"
    assert str(ring("x,y", QQ, grlex)[0]) == "Polynomial ring in x, y over QQ with grlex order"
    assert str(ring("x,y,z", ZZ["t"], lex)[0]) == "Polynomial ring in x, y, z over ZZ[t] with lex order"


def test_FracField():
    assert str(field("x", ZZ, lex)[0]) == "Rational function field in x over ZZ with lex order"
    assert str(field("x,y", QQ, grlex)[0]) == "Rational function field in x, y over QQ with grlex order"
    assert str(field("x,y,z", ZZ["t"], lex)[0]) == "Rational function field in x, y, z over ZZ[t] with lex order"


def test_PolyElement():
    Ruv, u,v = ring("u,v", ZZ)
    Rxyz, x,y,z = ring("x,y,z", Ruv)
    Rx_zzi, xz = ring("x", ZZ_I)

    assert str(x - x) == "0"
    assert str(x - 1) == "x - 1"
    assert str(x + 1) == "x + 1"
    assert str(x**2) == "x**2"

    assert str((u**2 + 3*u*v + 1)*x**2*y + u + 1) == "(u**2 + 3*u*v + 1)*x**2*y + u + 1"
    assert str((u**2 + 3*u*v + 1)*x**2*y + (u + 1)*x) == "(u**2 + 3*u*v + 1)*x**2*y + (u + 1)*x"
    assert str((u**2 + 3*u*v + 1)*x**2*y + (u + 1)*x + 1) == "(u**2 + 3*u*v + 1)*x**2*y + (u + 1)*x + 1"
    assert str((-u**2 + 3*u*v - 1)*x**2*y - (u + 1)*x - 1) == "-(u**2 - 3*u*v + 1)*x**2*y - (u + 1)*x - 1"

    assert str(-(v**2 + v + 1)*x + 3*u*v + 1) == "-(v**2 + v + 1)*x + 3*u*v + 1"
    assert str(-(v**2 + v + 1)*x - 3*u*v + 1) == "-(v**2 + v + 1)*x - 3*u*v + 1"

    assert str((1+I)*xz + 2) == "(1 + 1*I)*x + (2 + 0*I)"


def test_FracElement():
    Fuv, u,v = field("u,v", ZZ)
    Fxyzt, x,y,z,t = field("x,y,z,t", Fuv)
    Rx_zzi, xz = field("x", QQ_I)
    i = QQ_I(0, 1)

    assert str(x - x) == "0"
    assert str(x - 1) == "x - 1"
    assert str(x + 1) == "x + 1"

    assert str(x/3) == "x/3"
    assert str(x/z) == "x/z"
    assert str(x*y/z) == "x*y/z"
    assert str(x/(z*t)) == "x/(z*t)"
    assert str(x*y/(z*t)) == "x*y/(z*t)"

    assert str((x - 1)/y) == "(x - 1)/y"
    assert str((x + 1)/y) == "(x + 1)/y"
    assert str((-x - 1)/y) == "(-x - 1)/y"
    assert str((x + 1)/(y*z)) == "(x + 1)/(y*z)"
    assert str(-y/(x + 1)) == "-y/(x + 1)"
    assert str(y*z/(x + 1)) == "y*z/(x + 1)"

    assert str(((u + 1)*x*y + 1)/((v - 1)*z - 1)) == "((u + 1)*x*y + 1)/((v - 1)*z - 1)"
    assert str(((u + 1)*x*y + 1)/((v - 1)*z - t*u*v - 1)) == "((u + 1)*x*y + 1)/((v - 1)*z - u*v*t - 1)"

    assert str((1+i)/xz) == "(1 + 1*I)/x"
    assert str(((1+i)*xz - i)/xz) == "((1 + 1*I)*x + (0 + -1*I))/x"


def test_GaussianInteger():
    assert str(ZZ_I(1, 0)) == "1"
    assert str(ZZ_I(-1, 0)) == "-1"
    assert str(ZZ_I(0, 1)) == "I"
    assert str(ZZ_I(0, -1)) == "-I"
    assert str(ZZ_I(0, 2)) == "2*I"
    assert str(ZZ_I(0, -2)) == "-2*I"
    assert str(ZZ_I(1, 1)) == "1 + I"
    assert str(ZZ_I(-1, -1)) == "-1 - I"
    assert str(ZZ_I(-1, -2)) == "-1 - 2*I"


def test_GaussianRational():
    assert str(QQ_I(1, 0)) == "1"
    assert str(QQ_I(QQ(2, 3), 0)) == "2/3"
    assert str(QQ_I(0, QQ(2, 3))) == "2*I/3"
    assert str(QQ_I(QQ(1, 2), QQ(-2, 3))) == "1/2 - 2*I/3"


def test_Pow():
    assert str(x**-1) == "1/x"
    assert str(x**-2) == "x**(-2)"
    assert str(x**2) == "x**2"
    assert str((x + y)**-1) == "1/(x + y)"
    assert str((x + y)**-2) == "(x + y)**(-2)"
    assert str((x + y)**2) == "(x + y)**2"
    assert str((x + y)**(1 + x)) == "(x + y)**(x + 1)"
    assert str(x**Rational(1, 3)) == "x**(1/3)"
    assert str(1/x**Rational(1, 3)) == "x**(-1/3)"
    assert str(sqrt(sqrt(x))) == "x**(1/4)"
    # not the same as x**-1
    assert str(x**-1.0) == 'x**(-1.0)'
    # see issue #2860
    assert str(Pow(S(2), -1.0, evaluate=False)) == '2**(-1.0)'


def test_sqrt():
    assert str(sqrt(x)) == "sqrt(x)"
    assert str(sqrt(x**2)) == "sqrt(x**2)"
    assert str(1/sqrt(x)) == "1/sqrt(x)"
    assert str(1/sqrt(x**2)) == "1/sqrt(x**2)"
    assert str(y/sqrt(x)) == "y/sqrt(x)"
    assert str(x**0.5) == "x**0.5"
    assert str(1/x**0.5) == "x**(-0.5)"


def test_Rational():
    n1 = Rational(1, 4)
    n2 = Rational(1, 3)
    n3 = Rational(2, 4)
    n4 = Rational(2, -4)
    n5 = Rational(0)
    n7 = Rational(3)
    n8 = Rational(-3)
    assert str(n1*n2) == "1/12"
    assert str(n1*n2) == "1/12"
    assert str(n3) == "1/2"
    assert str(n1*n3) == "1/8"
    assert str(n1 + n3) == "3/4"
    assert str(n1 + n2) == "7/12"
    assert str(n1 + n4) == "-1/4"
    assert str(n4*n4) == "1/4"
    assert str(n4 + n2) == "-1/6"
    assert str(n4 + n5) == "-1/2"
    assert str(n4*n5) == "0"
    assert str(n3 + n4) == "0"
    assert str(n1**n7) == "1/64"
    assert str(n2**n7) == "1/27"
    assert str(n2**n8) == "27"
    assert str(n7**n8) == "1/27"
    assert str(Rational("-25")) == "-25"
    assert str(Rational("1.25")) == "5/4"
    assert str(Rational("-2.6e-2")) == "-13/500"
    assert str(S("25/7")) == "25/7"
    assert str(S("-123/569")) == "-123/569"
    assert str(S("0.1[23]", rational=1)) == "61/495"
    assert str(S("5.1[666]", rational=1)) == "31/6"
    assert str(S("-5.1[666]", rational=1)) == "-31/6"
    assert str(S("0.[9]", rational=1)) == "1"
    assert str(S("-0.[9]", rational=1)) == "-1"

    assert str(sqrt(Rational(1, 4))) == "1/2"
    assert str(sqrt(Rational(1, 36))) == "1/6"

    assert str((123**25) ** Rational(1, 25)) == "123"
    assert str((123**25 + 1)**Rational(1, 25)) != "123"
    assert str((123**25 - 1)**Rational(1, 25)) != "123"
    assert str((123**25 - 1)**Rational(1, 25)) != "122"

    assert str(sqrt(Rational(81, 36))**3) == "27/8"
    assert str(1/sqrt(Rational(81, 36))**3) == "8/27"

    assert str(sqrt(-4)) == str(2*I)
    assert str(2**Rational(1, 10**10)) == "2**(1/10000000000)"

    assert sstr(Rational(2, 3), sympy_integers=True) == "S(2)/3"
    x = Symbol("x")
    assert sstr(x**Rational(2, 3), sympy_integers=True) == "x**(S(2)/3)"
    assert sstr(Eq(x, Rational(2, 3)), sympy_integers=True) == "Eq(x, S(2)/3)"
    assert sstr(Limit(x, x, Rational(7, 2)), sympy_integers=True) == \
        "Limit(x, x, S(7)/2, dir='+')"


def test_Float():
    # NOTE dps is the whole number of decimal digits
    assert str(Float('1.23', dps=1 + 2)) == '1.23'
    assert str(Float('1.23456789', dps=1 + 8)) == '1.23456789'
    assert str(
        Float('1.234567890123456789', dps=1 + 18)) == '1.234567890123456789'
    assert str(pi.evalf(1 + 2)) == '3.14'
    assert str(pi.evalf(1 + 14)) == '3.14159265358979'
    assert str(pi.evalf(1 + 64)) == ('3.141592653589793238462643383279'
                                     '5028841971693993751058209749445923')
    assert str(pi.round(-1)) == '0.0'
    assert str((pi**400 - (pi**400).round(1)).n(2)) == '-0.e+88'
    assert sstr(Float("100"), full_prec=False, min=-2, max=2) == '1.0e+2'
    assert sstr(Float("100"), full_prec=False, min=-2, max=3) == '100.0'
    assert sstr(Float("0.1"), full_prec=False, min=-2, max=3) == '0.1'
    assert sstr(Float("0.099"), min=-2, max=3) == '9.90000000000000e-2'


def test_Relational():
    assert str(Rel(x, y, "<")) == "x < y"
    assert str(Rel(x + y, y, "==")) == "Eq(x + y, y)"
    assert str(Rel(x, y, "!=")) == "Ne(x, y)"
    assert str(Eq(x, 1) | Eq(x, 2)) == "Eq(x, 1) | Eq(x, 2)"
    assert str(Ne(x, 1) & Ne(x, 2)) == "Ne(x, 1) & Ne(x, 2)"


def test_AppliedBinaryRelation():
    assert str(Q.eq(x, y)) == "Q.eq(x, y)"
    assert str(Q.ne(x, y)) == "Q.ne(x, y)"


def test_CRootOf():
    assert str(rootof(x**5 + 2*x - 1, 0)) == "CRootOf(x**5 + 2*x - 1, 0)"


def test_RootSum():
    f = x**5 + 2*x - 1

    assert str(
        RootSum(f, Lambda(z, z), auto=False)) == "RootSum(x**5 + 2*x - 1)"
    assert str(RootSum(f, Lambda(
        z, z**2), auto=False)) == "RootSum(x**5 + 2*x - 1, Lambda(z, z**2))"


def test_GroebnerBasis():
    assert str(groebner(
        [], x, y)) == "GroebnerBasis([], x, y, domain='ZZ', order='lex')"

    F = [x**2 - 3*y - x + 1, y**2 - 2*x + y - 1]

    assert str(groebner(F, order='grlex')) == \
        "GroebnerBasis([x**2 - x - 3*y + 1, y**2 - 2*x + y - 1], x, y, domain='ZZ', order='grlex')"
    assert str(groebner(F, order='lex')) == \
        "GroebnerBasis([2*x - y**2 - y + 1, y**4 + 2*y**3 - 3*y**2 - 16*y + 7], x, y, domain='ZZ', order='lex')"

def test_set():
    assert sstr(set()) == 'set()'
    assert sstr(frozenset()) == 'frozenset()'

    assert sstr({1}) == '{1}'
    assert sstr(frozenset([1])) == 'frozenset({1})'
    assert sstr({1, 2, 3}) == '{1, 2, 3}'
    assert sstr(frozenset([1, 2, 3])) == 'frozenset({1, 2, 3})'

    assert sstr(
        {1, x, x**2, x**3, x**4}) == '{1, x, x**2, x**3, x**4}'
    assert sstr(
        frozenset([1, x, x**2, x**3, x**4])) == 'frozenset({1, x, x**2, x**3, x**4})'


def test_SparseMatrix():
    M = SparseMatrix([[x**+1, 1], [y, x + y]])
    assert str(M) == "Matrix([[x, 1], [y, x + y]])"
    assert sstr(M) == "Matrix([\n[x,     1],\n[y, x + y]])"


def test_Sum():
    assert str(summation(cos(3*z), (z, x, y))) == "Sum(cos(3*z), (z, x, y))"
    assert str(Sum(x*y**2, (x, -2, 2), (y, -5, 5))) == \
        "Sum(x*y**2, (x, -2, 2), (y, -5, 5))"


def test_Symbol():
    assert str(y) == "y"
    assert str(x) == "x"
    e = x
    assert str(e) == "x"


def test_tuple():
    assert str((x,)) == sstr((x,)) == "(x,)"
    assert str((x + y, 1 + x)) == sstr((x + y, 1 + x)) == "(x + y, x + 1)"
    assert str((x + y, (
        1 + x, x**2))) == sstr((x + y, (1 + x, x**2))) == "(x + y, (x + 1, x**2))"


def test_Series_str():
    tf1 = TransferFunction(x*y**2 - z, y**3 - t**3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    tf3 = TransferFunction(t*x**2 - t**w*x + w, t - y, y)
    assert str(Series(tf1, tf2)) == \
        "Series(TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y))"
    assert str(Series(tf1, tf2, tf3)) == \
        "Series(TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y), TransferFunction(t*x**2 - t**w*x + w, t - y, y))"
    assert str(Series(-tf2, tf1)) == \
        "Series(TransferFunction(-x + y, x + y, y), TransferFunction(x*y**2 - z, -t**3 + y**3, y))"


def test_MIMOSeries_str():
    tf1 = TransferFunction(x*y**2 - z, y**3 - t**3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    tfm_1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
    tfm_2 = TransferFunctionMatrix([[tf2, tf1], [tf1, tf2]])
    assert str(MIMOSeries(tfm_1, tfm_2)) == \
        "MIMOSeries(TransferFunctionMatrix(((TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y)), "\
            "(TransferFunction(x - y, x + y, y), TransferFunction(x*y**2 - z, -t**3 + y**3, y)))), "\
                "TransferFunctionMatrix(((TransferFunction(x - y, x + y, y), TransferFunction(x*y**2 - z, -t**3 + y**3, y)), "\
                    "(TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y)))))"


def test_TransferFunction_str():
    tf1 = TransferFunction(x - 1, x + 1, x)
    assert str(tf1) == "TransferFunction(x - 1, x + 1, x)"
    tf2 = TransferFunction(x + 1, 2 - y, x)
    assert str(tf2) == "TransferFunction(x + 1, 2 - y, x)"
    tf3 = TransferFunction(y, y**2 + 2*y + 3, y)
    assert str(tf3) == "TransferFunction(y, y**2 + 2*y + 3, y)"


def test_Parallel_str():
    tf1 = TransferFunction(x*y**2 - z, y**3 - t**3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    tf3 = TransferFunction(t*x**2 - t**w*x + w, t - y, y)
    assert str(Parallel(tf1, tf2)) == \
        "Parallel(TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y))"
    assert str(Parallel(tf1, tf2, tf3)) == \
        "Parallel(TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y), TransferFunction(t*x**2 - t**w*x + w, t - y, y))"
    assert str(Parallel(-tf2, tf1)) == \
        "Parallel(TransferFunction(-x + y, x + y, y), TransferFunction(x*y**2 - z, -t**3 + y**3, y))"


def test_MIMOParallel_str():
    tf1 = TransferFunction(x*y**2 - z, y**3 - t**3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    tfm_1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
    tfm_2 = TransferFunctionMatrix([[tf2, tf1], [tf1, tf2]])
    assert str(MIMOParallel(tfm_1, tfm_2)) == \
        "MIMOParallel(TransferFunctionMatrix(((TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y)), "\
            "(TransferFunction(x - y, x + y, y), TransferFunction(x*y**2 - z, -t**3 + y**3, y)))), "\
                "TransferFunctionMatrix(((TransferFunction(x - y, x + y, y), TransferFunction(x*y**2 - z, -t**3 + y**3, y)), "\
                    "(TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y)))))"


def test_Feedback_str():
    tf1 = TransferFunction(x*y**2 - z, y**3 - t**3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    tf3 = TransferFunction(t*x**2 - t**w*x + w, t - y, y)
    assert str(Feedback(tf1*tf2, tf3)) == \
        "Feedback(Series(TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y)), " \
        "TransferFunction(t*x**2 - t**w*x + w, t - y, y), -1)"
    assert str(Feedback(tf1, TransferFunction(1, 1, y), 1)) == \
        "Feedback(TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(1, 1, y), 1)"


def test_MIMOFeedback_str():
    tf1 = TransferFunction(x**2 - y**3, y - z, x)
    tf2 = TransferFunction(y - x, z + y, x)
    tfm_1 = TransferFunctionMatrix([[tf2, tf1], [tf1, tf2]])
    tfm_2 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
    assert (str(MIMOFeedback(tfm_1, tfm_2)) \
            == "MIMOFeedback(TransferFunctionMatrix(((TransferFunction(-x + y, y + z, x), TransferFunction(x**2 - y**3, y - z, x))," \
            " (TransferFunction(x**2 - y**3, y - z, x), TransferFunction(-x + y, y + z, x)))), " \
            "TransferFunctionMatrix(((TransferFunction(x**2 - y**3, y - z, x), " \
            "TransferFunction(-x + y, y + z, x)), (TransferFunction(-x + y, y + z, x), TransferFunction(x**2 - y**3, y - z, x)))), -1)")
    assert (str(MIMOFeedback(tfm_1, tfm_2, 1)) \
            == "MIMOFeedback(TransferFunctionMatrix(((TransferFunction(-x + y, y + z, x), TransferFunction(x**2 - y**3, y - z, x)), " \
            "(TransferFunction(x**2 - y**3, y - z, x), TransferFunction(-x + y, y + z, x)))), " \
            "TransferFunctionMatrix(((TransferFunction(x**2 - y**3, y - z, x), TransferFunction(-x + y, y + z, x)), "\
            "(TransferFunction(-x + y, y + z, x), TransferFunction(x**2 - y**3, y - z, x)))), 1)")


def test_TransferFunctionMatrix_str():
    tf1 = TransferFunction(x*y**2 - z, y**3 - t**3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    tf3 = TransferFunction(t*x**2 - t**w*x + w, t - y, y)
    assert str(TransferFunctionMatrix([[tf1], [tf2]])) == \
        "TransferFunctionMatrix(((TransferFunction(x*y**2 - z, -t**3 + y**3, y),), (TransferFunction(x - y, x + y, y),)))"
    assert str(TransferFunctionMatrix([[tf1, tf2], [tf3, tf2]])) == \
        "TransferFunctionMatrix(((TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y)), (TransferFunction(t*x**2 - t**w*x + w, t - y, y), TransferFunction(x - y, x + y, y))))"


def test_Quaternion_str_printer():
    q = Quaternion(x, y, z, t)
    assert str(q) == "x + y*i + z*j + t*k"
    q = Quaternion(x,y,z,x*t)
    assert str(q) == "x + y*i + z*j + t*x*k"
    q = Quaternion(x,y,z,x+t)
    assert str(q) == "x + y*i + z*j + (t + x)*k"


def test_Quantity_str():
    assert sstr(second, abbrev=True) == "s"
    assert sstr(joule, abbrev=True) == "J"
    assert str(second) == "second"
    assert str(joule) == "joule"


def test_wild_str():
    # Check expressions containing Wild not causing infinite recursion
    w = Wild('x')
    assert str(w + 1) == 'x_ + 1'
    assert str(exp(2**w) + 5) == 'exp(2**x_) + 5'
    assert str(3*w + 1) == '3*x_ + 1'
    assert str(1/w + 1) == '1 + 1/x_'
    assert str(w**2 + 1) == 'x_**2 + 1'
    assert str(1/(1 - w)) == '1/(1 - x_)'


def test_wild_matchpy():
    from sympy.utilities.matchpy_connector import WildDot, WildPlus, WildStar

    matchpy = import_module("matchpy")

    if matchpy is None:
        return

    wd = WildDot('w_')
    wp = WildPlus('w__')
    ws = WildStar('w___')

    assert str(wd) == 'w_'
    assert str(wp) == 'w__'
    assert str(ws) == 'w___'

    assert str(wp/ws + 2**wd) == '2**w_ + w__/w___'
    assert str(sin(wd)*cos(wp)*sqrt(ws)) == 'sqrt(w___)*sin(w_)*cos(w__)'


def test_zeta():
    assert str(zeta(3)) == "zeta(3)"


def test_issue_3101():
    e = x - y
    a = str(e)
    b = str(e)
    assert a == b


def test_issue_3103():
    e = -2*sqrt(x) - y/sqrt(x)/2
    assert str(e) not in ["(-2)*x**1/2(-1/2)*x**(-1/2)*y",
            "-2*x**1/2(-1/2)*x**(-1/2)*y", "-2*x**1/2-1/2*x**-1/2*w"]
    assert str(e) == "-2*sqrt(x) - y/(2*sqrt(x))"


def test_issue_4021():
    e = Integral(x, x) + 1
    assert str(e) == 'Integral(x, x) + 1'


def test_sstrrepr():
    assert sstr('abc') == 'abc'
    assert sstrrepr('abc') == "'abc'"

    e = ['a', 'b', 'c', x]
    assert sstr(e) == "[a, b, c, x]"
    assert sstrrepr(e) == "['a', 'b', 'c', x]"


def test_infinity():
    assert sstr(oo*I) == "oo*I"


def test_full_prec():
    assert sstr(S("0.3"), full_prec=True) == "0.300000000000000"
    assert sstr(S("0.3"), full_prec="auto") == "0.300000000000000"
    assert sstr(S("0.3"), full_prec=False) == "0.3"
    assert sstr(S("0.3")*x, full_prec=True) in [
        "0.300000000000000*x",
        "x*0.300000000000000"
    ]
    assert sstr(S("0.3")*x, full_prec="auto") in [
        "0.3*x",
        "x*0.3"
    ]
    assert sstr(S("0.3")*x, full_prec=False) in [
        "0.3*x",
        "x*0.3"
    ]


def test_noncommutative():
    A, B, C = symbols('A,B,C', commutative=False)

    assert sstr(A*B*C**-1) == "A*B*C**(-1)"
    assert sstr(C**-1*A*B) == "C**(-1)*A*B"
    assert sstr(A*C**-1*B) == "A*C**(-1)*B"
    assert sstr(sqrt(A)) == "sqrt(A)"
    assert sstr(1/sqrt(A)) == "A**(-1/2)"


def test_empty_printer():
    str_printer = StrPrinter()
    assert str_printer.emptyPrinter("foo") == "foo"
    assert str_printer.emptyPrinter(x*y) == "x*y"
    assert str_printer.emptyPrinter(32) == "32"

def test_decimal_printer():
    dec_printer = StrPrinter(settings={"dps":3})
    f = Function('f')
    assert dec_printer.doprint(f(1.329294)) == "f(1.33)"


def test_settings():
    raises(TypeError, lambda: sstr(S(4), method="garbage"))


def test_RandomDomain():
    from sympy.stats import Normal, Die, Exponential, pspace, where
    X = Normal('x1', 0, 1)
    assert str(where(X > 0)) == "Domain: (0 < x1) & (x1 < oo)"

    D = Die('d1', 6)
    assert str(where(D > 4)) == "Domain: Eq(d1, 5) | Eq(d1, 6)"

    A = Exponential('a', 1)
    B = Exponential('b', 1)
    assert str(pspace(Tuple(A, B)).domain) == "Domain: (0 <= a) & (0 <= b) & (a < oo) & (b < oo)"


def test_FiniteSet():
    assert str(FiniteSet(*range(1, 51))) == (
        '{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,'
        ' 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,'
        ' 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50}'
    )
    assert str(FiniteSet(*range(1, 6))) == '{1, 2, 3, 4, 5}'
    assert str(FiniteSet(*[x*y, x**2])) == '{x**2, x*y}'
    assert str(FiniteSet(FiniteSet(FiniteSet(x, y), 5), FiniteSet(x,y), 5)
               ) == 'FiniteSet(5, FiniteSet(5, {x, y}), {x, y})'


def test_Partition():
    assert str(Partition(FiniteSet(x, y), {z})) == 'Partition({z}, {x, y})'

def test_UniversalSet():
    assert str(S.UniversalSet) == 'UniversalSet'


def test_PrettyPoly():
    F = QQ.frac_field(x, y)
    R = QQ[x, y]
    assert sstr(F.convert(x/(x + y))) == sstr(x/(x + y))
    assert sstr(R.convert(x + y)) == sstr(x + y)


def test_categories():
    from sympy.categories import (Object, NamedMorphism,
        IdentityMorphism, Category)

    A = Object("A")
    B = Object("B")

    f = NamedMorphism(A, B, "f")
    id_A = IdentityMorphism(A)

    K = Category("K")

    assert str(A) == 'Object("A")'
    assert str(f) == 'NamedMorphism(Object("A"), Object("B"), "f")'
    assert str(id_A) == 'IdentityMorphism(Object("A"))'

    assert str(K) == 'Category("K")'


def test_Tr():
    A, B = symbols('A B', commutative=False)
    t = Tr(A*B)
    assert str(t) == 'Tr(A*B)'


def test_issue_6387():
    assert str(factor(-3.0*z + 3)) == '-3.0*(1.0*z - 1.0)'


def test_MatMul_MatAdd():
    X, Y = MatrixSymbol("X", 2, 2), MatrixSymbol("Y", 2, 2)
    assert str(2*(X + Y)) == "2*X + 2*Y"

    assert str(I*X) == "I*X"
    assert str(-I*X) == "-I*X"
    assert str((1 + I)*X) == '(1 + I)*X'
    assert str(-(1 + I)*X) == '(-1 - I)*X'
    assert str(MatAdd(MatAdd(X, Y), MatAdd(X, Y))) == '(X + Y) + (X + Y)'


def test_MatrixSlice():
    n = Symbol('n', integer=True)
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', 10, 10)
    Z = MatrixSymbol('Z', 10, 10)

    assert str(MatrixSlice(X, (None, None, None), (None, None, None))) == 'X[:, :]'
    assert str(X[x:x + 1, y:y + 1]) == 'X[x:x + 1, y:y + 1]'
    assert str(X[x:x + 1:2, y:y + 1:2]) == 'X[x:x + 1:2, y:y + 1:2]'
    assert str(X[:x, y:]) == 'X[:x, y:]'
    assert str(X[:x, y:]) == 'X[:x, y:]'
    assert str(X[x:, :y]) == 'X[x:, :y]'
    assert str(X[x:y, z:w]) == 'X[x:y, z:w]'
    assert str(X[x:y:t, w:t:x]) == 'X[x:y:t, w:t:x]'
    assert str(X[x::y, t::w]) == 'X[x::y, t::w]'
    assert str(X[:x:y, :t:w]) == 'X[:x:y, :t:w]'
    assert str(X[::x, ::y]) == 'X[::x, ::y]'
    assert str(MatrixSlice(X, (0, None, None), (0, None, None))) == 'X[:, :]'
    assert str(MatrixSlice(X, (None, n, None), (None, n, None))) == 'X[:, :]'
    assert str(MatrixSlice(X, (0, n, None), (0, n, None))) == 'X[:, :]'
    assert str(MatrixSlice(X, (0, n, 2), (0, n, 2))) == 'X[::2, ::2]'
    assert str(X[1:2:3, 4:5:6]) == 'X[1:2:3, 4:5:6]'
    assert str(X[1:3:5, 4:6:8]) == 'X[1:3:5, 4:6:8]'
    assert str(X[1:10:2]) == 'X[1:10:2, :]'
    assert str(Y[:5, 1:9:2]) == 'Y[:5, 1:9:2]'
    assert str(Y[:5, 1:10:2]) == 'Y[:5, 1::2]'
    assert str(Y[5, :5:2]) == 'Y[5:6, :5:2]'
    assert str(X[0:1, 0:1]) == 'X[:1, :1]'
    assert str(X[0:1:2, 0:1:2]) == 'X[:1:2, :1:2]'
    assert str((Y + Z)[2:, 2:]) == '(Y + Z)[2:, 2:]'

def test_true_false():
    assert str(true) == repr(true) == sstr(true) == "True"
    assert str(false) == repr(false) == sstr(false) == "False"

def test_Equivalent():
    assert str(Equivalent(y, x)) == "Equivalent(x, y)"

def test_Xor():
    assert str(Xor(y, x, evaluate=False)) == "x ^ y"

def test_Complement():
    assert str(Complement(S.Reals, S.Naturals)) == 'Complement(Reals, Naturals)'

def test_SymmetricDifference():
    assert str(SymmetricDifference(Interval(2, 3), Interval(3, 4),evaluate=False)) == \
           'SymmetricDifference(Interval(2, 3), Interval(3, 4))'


def test_UnevaluatedExpr():
    a, b = symbols("a b")
    expr1 = 2*UnevaluatedExpr(a+b)
    assert str(expr1) == "2*(a + b)"


def test_MatrixElement_printing():
    # test cases for issue #11821
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)
    C = MatrixSymbol("C", 1, 3)

    assert(str(A[0, 0]) == "A[0, 0]")
    assert(str(3 * A[0, 0]) == "3*A[0, 0]")

    F = C[0, 0].subs(C, A - B)
    assert str(F) == "(A - B)[0, 0]"


def test_MatrixSymbol_printing():
    A = MatrixSymbol("A", 3, 3)
    B = MatrixSymbol("B", 3, 3)

    assert str(A - A*B - B) == "A - A*B - B"
    assert str(A*B - (A+B)) == "-A + A*B - B"
    assert str(A**(-1)) == "A**(-1)"
    assert str(A**3) == "A**3"


def test_MatrixExpressions():
    n = Symbol('n', integer=True)
    X = MatrixSymbol('X', n, n)

    assert str(X) == "X"

    # Apply function elementwise (`ElementwiseApplyFunc`):

    expr = (X.T*X).applyfunc(sin)
    assert str(expr) == 'Lambda(_d, sin(_d)).(X.T*X)'

    lamda = Lambda(x, 1/x)
    expr = (n*X).applyfunc(lamda)
    assert str(expr) == 'Lambda(x, 1/x).(n*X)'


def test_Subs_printing():
    assert str(Subs(x, (x,), (1,))) == 'Subs(x, x, 1)'
    assert str(Subs(x + y, (x, y), (1, 2))) == 'Subs(x + y, (x, y), (1, 2))'


def test_issue_15716():
    e = Integral(factorial(x), (x, -oo, oo))
    assert e.as_terms() == ([(e, ((1.0, 0.0), (1,), ()))], [e])


def test_str_special_matrices():
    from sympy.matrices import Identity, ZeroMatrix, OneMatrix
    assert str(Identity(4)) == 'I'
    assert str(ZeroMatrix(2, 2)) == '0'
    assert str(OneMatrix(2, 2)) == '1'


def test_issue_14567():
    assert factorial(Sum(-1, (x, 0, 0))) + y  # doesn't raise an error


def test_issue_21823():
    assert str(Partition([1, 2])) == 'Partition({1, 2})'
    assert str(Partition({1, 2})) == 'Partition({1, 2})'


def test_issue_22689():
    assert str(Mul(Pow(x,-2, evaluate=False), Pow(3,-1,evaluate=False), evaluate=False)) == "1/(x**2*3)"


def test_issue_21119_21460():
    ss = lambda x: str(S(x, evaluate=False))
    assert ss('4/2') == '4/2'
    assert ss('4/-2') == '4/(-2)'
    assert ss('-4/2') == '-4/2'
    assert ss('-4/-2') == '-4/(-2)'
    assert ss('-2*3/-1') == '-2*3/(-1)'
    assert ss('-2*3/-1/2') == '-2*3/(-1*2)'
    assert ss('4/2/1') == '4/(2*1)'
    assert ss('-2/-1/2') == '-2/(-1*2)'
    assert ss('2*3*4**(-2*3)') == '2*3/4**(2*3)'
    assert ss('2*3*1*4**(-2*3)') == '2*3*1/4**(2*3)'


def test_Str():
    from sympy.core.symbol import Str
    assert str(Str('x')) == 'x'
    assert sstrrepr(Str('x')) == "Str('x')"


def test_diffgeom():
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseScalarField
    x,y = symbols('x y', real=True)
    m = Manifold('M', 2)
    assert str(m) == "M"
    p = Patch('P', m)
    assert str(p) == "P"
    rect = CoordSystem('rect', p, [x, y])
    assert str(rect) == "rect"
    b = BaseScalarField(rect, 0)
    assert str(b) == "x"

def test_NDimArray():
    assert sstr(NDimArray(1.0), full_prec=True) == '1.00000000000000'
    assert sstr(NDimArray(1.0), full_prec=False) == '1.0'
    assert sstr(NDimArray([1.0, 2.0]), full_prec=True) == '[1.00000000000000, 2.00000000000000]'
    assert sstr(NDimArray([1.0, 2.0]), full_prec=False) == '[1.0, 2.0]'
    assert sstr(NDimArray([], (0,))) == 'ImmutableDenseNDimArray([], (0,))'
    assert sstr(NDimArray([], (0, 0))) == 'ImmutableDenseNDimArray([], (0, 0))'
    assert sstr(NDimArray([], (0, 1))) == 'ImmutableDenseNDimArray([], (0, 1))'
    assert sstr(NDimArray([], (1, 0))) == 'ImmutableDenseNDimArray([], (1, 0))'

def test_Predicate():
    assert sstr(Q.even) == 'Q.even'

def test_AppliedPredicate():
    assert sstr(Q.even(x)) == 'Q.even(x)'

def test_printing_str_array_expressions():
    assert sstr(ArraySymbol("A", (2, 3, 4))) == "A"
    assert sstr(ArrayElement("A", (2, 1/(1-x), 0))) == "A[2, 1/(1 - x), 0]"
    M = MatrixSymbol("M", 3, 3)
    N = MatrixSymbol("N", 3, 3)
    assert sstr(ArrayElement(M*N, [x, 0])) == "(M*N)[x, 0]"

def test_printing_stats():
    # issue 24132
    x = RandomSymbol("x")
    y = RandomSymbol("y")
    z1 = Probability(x > 0)*Identity(2)
    z2 = Expectation(x)*Identity(2)
    z3 = Variance(x)*Identity(2)
    z4 = Covariance(x, y) * Identity(2)

    assert str(z1) == "Probability(x > 0)*I"
    assert str(z2) == "Expectation(x)*I"
    assert str(z3) == "Variance(x)*I"
    assert str(z4) ==  "Covariance(x, y)*I"
    assert z1.is_commutative == False
    assert z2.is_commutative == False
    assert z3.is_commutative == False
    assert z4.is_commutative == False
    assert z2._eval_is_commutative() == False
    assert z3._eval_is_commutative() == False
    assert z4._eval_is_commutative() == False