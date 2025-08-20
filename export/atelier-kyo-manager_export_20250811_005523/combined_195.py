
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_sort_values.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    NaT,
    Timestamp,
    date_range,
)
import pandas._testing as tm
from pandas.util.version import Version


class TestDataFrameSortValues:
    @pytest.mark.parametrize("dtype", [np.uint8, bool])
    def test_sort_values_sparse_no_warning(self, dtype):
        # GH#45618
        ser = pd.Series(Categorical(["a", "b", "a"], categories=["a", "b", "c"]))
        df = pd.get_dummies(ser, dtype=dtype, sparse=True)

        with tm.assert_produces_warning(None):
            # No warnings about constructing Index from SparseArray
            df.sort_values(by=df.columns.tolist())

    def test_sort_values(self):
        frame = DataFrame(
            [[1, 1, 2], [3, 1, 0], [4, 5, 6]], index=[1, 2, 3], columns=list("ABC")
        )

        # by column (axis=0)
        sorted_df = frame.sort_values(by="A")
        indexer = frame["A"].argsort().values
        expected = frame.loc[frame.index[indexer]]
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by="A", ascending=False)
        indexer = indexer[::-1]
        expected = frame.loc[frame.index[indexer]]
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by="A", ascending=False)
        tm.assert_frame_equal(sorted_df, expected)

        # GH4839
        sorted_df = frame.sort_values(by=["A"], ascending=[False])
        tm.assert_frame_equal(sorted_df, expected)

        # multiple bys
        sorted_df = frame.sort_values(by=["B", "C"])
        expected = frame.loc[[2, 1, 3]]
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by=["B", "C"], ascending=False)
        tm.assert_frame_equal(sorted_df, expected[::-1])

        sorted_df = frame.sort_values(by=["B", "A"], ascending=[True, False])
        tm.assert_frame_equal(sorted_df, expected)

        msg = "No axis named 2 for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            frame.sort_values(by=["A", "B"], axis=2, inplace=True)

        # by row (axis=1): GH#10806
        sorted_df = frame.sort_values(by=3, axis=1)
        expected = frame
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by=3, axis=1, ascending=False)
        expected = frame.reindex(columns=["C", "B", "A"])
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by=[1, 2], axis="columns")
        expected = frame.reindex(columns=["B", "A", "C"])
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by=[1, 3], axis=1, ascending=[True, False])
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by=[1, 3], axis=1, ascending=False)
        expected = frame.reindex(columns=["C", "B", "A"])
        tm.assert_frame_equal(sorted_df, expected)

        msg = r"Length of ascending \(5\) != length of by \(2\)"
        with pytest.raises(ValueError, match=msg):
            frame.sort_values(by=["A", "B"], axis=0, ascending=[True] * 5)

    def test_sort_values_by_empty_list(self):
        # https://github.com/pandas-dev/pandas/issues/40258
        expected = DataFrame({"a": [1, 4, 2, 5, 3, 6]})
        result = expected.sort_values(by=[])
        tm.assert_frame_equal(result, expected)
        assert result is not expected

    def test_sort_values_inplace(self):
        frame = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=[1, 2, 3, 4],
            columns=["A", "B", "C", "D"],
        )

        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(by="A", inplace=True)
        assert return_value is None
        expected = frame.sort_values(by="A")
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(by=1, axis=1, inplace=True)
        assert return_value is None
        expected = frame.sort_values(by=1, axis=1)
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(by="A", ascending=False, inplace=True)
        assert return_value is None
        expected = frame.sort_values(by="A", ascending=False)
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(
            by=["A", "B"], ascending=False, inplace=True
        )
        assert return_value is None
        expected = frame.sort_values(by=["A", "B"], ascending=False)
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_multicolumn(self):
        A = np.arange(5).repeat(20)
        B = np.tile(np.arange(5), 20)
        np.random.default_rng(2).shuffle(A)
        np.random.default_rng(2).shuffle(B)
        frame = DataFrame(
            {"A": A, "B": B, "C": np.random.default_rng(2).standard_normal(100)}
        )

        result = frame.sort_values(by=["A", "B"])
        indexer = np.lexsort((frame["B"], frame["A"]))
        expected = frame.take(indexer)
        tm.assert_frame_equal(result, expected)

        result = frame.sort_values(by=["A", "B"], ascending=False)
        indexer = np.lexsort(
            (frame["B"].rank(ascending=False), frame["A"].rank(ascending=False))
        )
        expected = frame.take(indexer)
        tm.assert_frame_equal(result, expected)

        result = frame.sort_values(by=["B", "A"])
        indexer = np.lexsort((frame["A"], frame["B"]))
        expected = frame.take(indexer)
        tm.assert_frame_equal(result, expected)

    def test_sort_values_multicolumn_uint64(self):
        # GH#9918
        # uint64 multicolumn sort

        df = DataFrame(
            {
                "a": pd.Series([18446637057563306014, 1162265347240853609]),
                "b": pd.Series([1, 2]),
            }
        )
        df["a"] = df["a"].astype(np.uint64)
        result = df.sort_values(["a", "b"])

        expected = DataFrame(
            {
                "a": pd.Series([18446637057563306014, 1162265347240853609]),
                "b": pd.Series([1, 2]),
            },
            index=pd.Index([1, 0]),
        )

        tm.assert_frame_equal(result, expected)

    def test_sort_values_nan(self):
        # GH#3917
        df = DataFrame(
            {"A": [1, 2, np.nan, 1, 6, 8, 4], "B": [9, np.nan, 5, 2, 5, 4, 5]}
        )

        # sort one column only
        expected = DataFrame(
            {"A": [np.nan, 1, 1, 2, 4, 6, 8], "B": [5, 9, 2, np.nan, 5, 5, 4]},
            index=[2, 0, 3, 1, 6, 4, 5],
        )
        sorted_df = df.sort_values(["A"], na_position="first")
        tm.assert_frame_equal(sorted_df, expected)

        expected = DataFrame(
            {"A": [np.nan, 8, 6, 4, 2, 1, 1], "B": [5, 4, 5, 5, np.nan, 9, 2]},
            index=[2, 5, 4, 6, 1, 0, 3],
        )
        sorted_df = df.sort_values(["A"], na_position="first", ascending=False)
        tm.assert_frame_equal(sorted_df, expected)

        expected = df.reindex(columns=["B", "A"])
        sorted_df = df.sort_values(by=1, axis=1, na_position="first")
        tm.assert_frame_equal(sorted_df, expected)

        # na_position='last', order
        expected = DataFrame(
            {"A": [1, 1, 2, 4, 6, 8, np.nan], "B": [2, 9, np.nan, 5, 5, 4, 5]},
            index=[3, 0, 1, 6, 4, 5, 2],
        )
        sorted_df = df.sort_values(["A", "B"])
        tm.assert_frame_equal(sorted_df, expected)

        # na_position='first', order
        expected = DataFrame(
            {"A": [np.nan, 1, 1, 2, 4, 6, 8], "B": [5, 2, 9, np.nan, 5, 5, 4]},
            index=[2, 3, 0, 1, 6, 4, 5],
        )
        sorted_df = df.sort_values(["A", "B"], na_position="first")
        tm.assert_frame_equal(sorted_df, expected)

        # na_position='first', not order
        expected = DataFrame(
            {"A": [np.nan, 1, 1, 2, 4, 6, 8], "B": [5, 9, 2, np.nan, 5, 5, 4]},
            index=[2, 0, 3, 1, 6, 4, 5],
        )
        sorted_df = df.sort_values(["A", "B"], ascending=[1, 0], na_position="first")
        tm.assert_frame_equal(sorted_df, expected)

        # na_position='last', not order
        expected = DataFrame(
            {"A": [8, 6, 4, 2, 1, 1, np.nan], "B": [4, 5, 5, np.nan, 2, 9, 5]},
            index=[5, 4, 6, 1, 3, 0, 2],
        )
        sorted_df = df.sort_values(["A", "B"], ascending=[0, 1], na_position="last")
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_stable_descending_sort(self):
        # GH#6399
        df = DataFrame(
            [[2, "first"], [2, "second"], [1, "a"], [1, "b"]],
            columns=["sort_col", "order"],
        )
        sorted_df = df.sort_values(by="sort_col", kind="mergesort", ascending=False)
        tm.assert_frame_equal(df, sorted_df)

    @pytest.mark.parametrize(
        "expected_idx_non_na, ascending",
        [
            [
                [3, 4, 5, 0, 1, 8, 6, 9, 7, 10, 13, 14],
                [True, True],
            ],
            [
                [0, 3, 4, 5, 1, 8, 6, 7, 10, 13, 14, 9],
                [True, False],
            ],
            [
                [9, 7, 10, 13, 14, 6, 8, 1, 3, 4, 5, 0],
                [False, True],
            ],
            [
                [7, 10, 13, 14, 9, 6, 8, 1, 0, 3, 4, 5],
                [False, False],
            ],
        ],
    )
    @pytest.mark.parametrize("na_position", ["first", "last"])
    def test_sort_values_stable_multicolumn_sort(
        self, expected_idx_non_na, ascending, na_position
    ):
        # GH#38426 Clarify sort_values with mult. columns / labels is stable
        df = DataFrame(
            {
                "A": [1, 2, np.nan, 1, 1, 1, 6, 8, 4, 8, 8, np.nan, np.nan, 8, 8],
                "B": [9, np.nan, 5, 2, 2, 2, 5, 4, 5, 3, 4, np.nan, np.nan, 4, 4],
            }
        )
        # All rows with NaN in col "B" only have unique values in "A", therefore,
        # only the rows with NaNs in "A" have to be treated individually:
        expected_idx = (
            [11, 12, 2] + expected_idx_non_na
            if na_position == "first"
            else expected_idx_non_na + [2, 11, 12]
        )
        expected = df.take(expected_idx)
        sorted_df = df.sort_values(
            ["A", "B"], ascending=ascending, na_position=na_position
        )
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_stable_categorial(self):
        # GH#16793
        df = DataFrame({"x": Categorical(np.repeat([1, 2, 3, 4], 5), ordered=True)})
        expected = df.copy()
        sorted_df = df.sort_values("x", kind="mergesort")
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_datetimes(self):
        # GH#3461, argsort / lexsort differences for a datetime column
        df = DataFrame(
            ["a", "a", "a", "b", "c", "d", "e", "f", "g"],
            columns=["A"],
            index=date_range("20130101", periods=9),
        )
        dts = [
            Timestamp(x)
            for x in [
                "2004-02-11",
                "2004-01-21",
                "2004-01-26",
                "2005-09-20",
                "2010-10-04",
                "2009-05-12",
                "2008-11-12",
                "2010-09-28",
                "2010-09-28",
            ]
        ]
        df["B"] = dts[::2] + dts[1::2]
        df["C"] = 2.0
        df["A1"] = 3.0

        df1 = df.sort_values(by="A")
        df2 = df.sort_values(by=["A"])
        tm.assert_frame_equal(df1, df2)

        df1 = df.sort_values(by="B")
        df2 = df.sort_values(by=["B"])
        tm.assert_frame_equal(df1, df2)

        df1 = df.sort_values(by="B")

        df2 = df.sort_values(by=["C", "B"])
        tm.assert_frame_equal(df1, df2)

    def test_sort_values_frame_column_inplace_sort_exception(
        self, float_frame, using_copy_on_write
    ):
        s = float_frame["A"]
        float_frame_orig = float_frame.copy()
        if using_copy_on_write:
            # INFO(CoW) Series is a new object, so can be changed inplace
            # without modifying original datafame
            s.sort_values(inplace=True)
            tm.assert_series_equal(s, float_frame_orig["A"].sort_values())
            # column in dataframe is not changed
            tm.assert_frame_equal(float_frame, float_frame_orig)
        else:
            with pytest.raises(ValueError, match="This Series is a view"):
                s.sort_values(inplace=True)

        cp = s.copy()
        cp.sort_values()  # it works!

    def test_sort_values_nat_values_in_int_column(self):
        # GH#14922: "sorting with large float and multiple columns incorrect"

        # cause was that the int64 value NaT was considered as "na". Which is
        # only correct for datetime64 columns.

        int_values = (2, int(NaT._value))
        float_values = (2.0, -1.797693e308)

        df = DataFrame(
            {"int": int_values, "float": float_values}, columns=["int", "float"]
        )

        df_reversed = DataFrame(
            {"int": int_values[::-1], "float": float_values[::-1]},
            columns=["int", "float"],
            index=[1, 0],
        )

        # NaT is not a "na" for int64 columns, so na_position must not
        # influence the result:
        df_sorted = df.sort_values(["int", "float"], na_position="last")
        tm.assert_frame_equal(df_sorted, df_reversed)

        df_sorted = df.sort_values(["int", "float"], na_position="first")
        tm.assert_frame_equal(df_sorted, df_reversed)

        # reverse sorting order
        df_sorted = df.sort_values(["int", "float"], ascending=False)
        tm.assert_frame_equal(df_sorted, df)

        # and now check if NaT is still considered as "na" for datetime64
        # columns:
        df = DataFrame(
            {"datetime": [Timestamp("2016-01-01"), NaT], "float": float_values},
            columns=["datetime", "float"],
        )

        df_reversed = DataFrame(
            {"datetime": [NaT, Timestamp("2016-01-01")], "float": float_values[::-1]},
            columns=["datetime", "float"],
            index=[1, 0],
        )

        df_sorted = df.sort_values(["datetime", "float"], na_position="first")
        tm.assert_frame_equal(df_sorted, df_reversed)

        df_sorted = df.sort_values(["datetime", "float"], na_position="last")
        tm.assert_frame_equal(df_sorted, df)

        # Ascending should not affect the results.
        df_sorted = df.sort_values(["datetime", "float"], ascending=False)
        tm.assert_frame_equal(df_sorted, df)

    def test_sort_nat(self):
        # GH 16836

        d1 = [Timestamp(x) for x in ["2016-01-01", "2015-01-01", np.nan, "2016-01-01"]]
        d2 = [
            Timestamp(x)
            for x in ["2017-01-01", "2014-01-01", "2016-01-01", "2015-01-01"]
        ]
        df = DataFrame({"a": d1, "b": d2}, index=[0, 1, 2, 3])

        d3 = [Timestamp(x) for x in ["2015-01-01", "2016-01-01", "2016-01-01", np.nan]]
        d4 = [
            Timestamp(x)
            for x in ["2014-01-01", "2015-01-01", "2017-01-01", "2016-01-01"]
        ]
        expected = DataFrame({"a": d3, "b": d4}, index=[1, 3, 0, 2])
        sorted_df = df.sort_values(by=["a", "b"])
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_na_position_with_categories(self):
        # GH#22556
        # Positioning missing value properly when column is Categorical.
        categories = ["A", "B", "C"]
        category_indices = [0, 2, 4]
        list_of_nans = [np.nan, np.nan]
        na_indices = [1, 3]
        na_position_first = "first"
        na_position_last = "last"
        column_name = "c"

        reversed_categories = sorted(categories, reverse=True)
        reversed_category_indices = sorted(category_indices, reverse=True)
        reversed_na_indices = sorted(na_indices)

        df = DataFrame(
            {
                column_name: Categorical(
                    ["A", np.nan, "B", np.nan, "C"], categories=categories, ordered=True
                )
            }
        )
        # sort ascending with na first
        result = df.sort_values(
            by=column_name, ascending=True, na_position=na_position_first
        )
        expected = DataFrame(
            {
                column_name: Categorical(
                    list_of_nans + categories, categories=categories, ordered=True
                )
            },
            index=na_indices + category_indices,
        )

        tm.assert_frame_equal(result, expected)

        # sort ascending with na last
        result = df.sort_values(
            by=column_name, ascending=True, na_position=na_position_last
        )
        expected = DataFrame(
            {
                column_name: Categorical(
                    categories + list_of_nans, categories=categories, ordered=True
                )
            },
            index=category_indices + na_indices,
        )

        tm.assert_frame_equal(result, expected)

        # sort descending with na first
        result = df.sort_values(
            by=column_name, ascending=False, na_position=na_position_first
        )
        expected = DataFrame(
            {
                column_name: Categorical(
                    list_of_nans + reversed_categories,
                    categories=categories,
                    ordered=True,
                )
            },
            index=reversed_na_indices + reversed_category_indices,
        )

        tm.assert_frame_equal(result, expected)

        # sort descending with na last
        result = df.sort_values(
            by=column_name, ascending=False, na_position=na_position_last
        )
        expected = DataFrame(
            {
                column_name: Categorical(
                    reversed_categories + list_of_nans,
                    categories=categories,
                    ordered=True,
                )
            },
            index=reversed_category_indices + reversed_na_indices,
        )

        tm.assert_frame_equal(result, expected)

    def test_sort_values_nat(self):
        # GH#16836

        d1 = [Timestamp(x) for x in ["2016-01-01", "2015-01-01", np.nan, "2016-01-01"]]
        d2 = [
            Timestamp(x)
            for x in ["2017-01-01", "2014-01-01", "2016-01-01", "2015-01-01"]
        ]
        df = DataFrame({"a": d1, "b": d2}, index=[0, 1, 2, 3])

        d3 = [Timestamp(x) for x in ["2015-01-01", "2016-01-01", "2016-01-01", np.nan]]
        d4 = [
            Timestamp(x)
            for x in ["2014-01-01", "2015-01-01", "2017-01-01", "2016-01-01"]
        ]
        expected = DataFrame({"a": d3, "b": d4}, index=[1, 3, 0, 2])
        sorted_df = df.sort_values(by=["a", "b"])
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_na_position_with_categories_raises(self):
        df = DataFrame(
            {
                "c": Categorical(
                    ["A", np.nan, "B", np.nan, "C"],
                    categories=["A", "B", "C"],
                    ordered=True,
                )
            }
        )

        with pytest.raises(ValueError, match="invalid na_position: bad_position"):
            df.sort_values(by="c", ascending=False, na_position="bad_position")

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize(
        "original_dict, sorted_dict, ignore_index, output_index",
        [
            ({"A": [1, 2, 3]}, {"A": [3, 2, 1]}, True, [0, 1, 2]),
            ({"A": [1, 2, 3]}, {"A": [3, 2, 1]}, False, [2, 1, 0]),
            (
                {"A": [1, 2, 3], "B": [2, 3, 4]},
                {"A": [3, 2, 1], "B": [4, 3, 2]},
                True,
                [0, 1, 2],
            ),
            (
                {"A": [1, 2, 3], "B": [2, 3, 4]},
                {"A": [3, 2, 1], "B": [4, 3, 2]},
                False,
                [2, 1, 0],
            ),
        ],
    )
    def test_sort_values_ignore_index(
        self, inplace, original_dict, sorted_dict, ignore_index, output_index
    ):
        # GH 30114
        df = DataFrame(original_dict)
        expected = DataFrame(sorted_dict, index=output_index)
        kwargs = {"ignore_index": ignore_index, "inplace": inplace}

        if inplace:
            result_df = df.copy()
            result_df.sort_values("A", ascending=False, **kwargs)
        else:
            result_df = df.sort_values("A", ascending=False, **kwargs)

        tm.assert_frame_equal(result_df, expected)
        tm.assert_frame_equal(df, DataFrame(original_dict))

    def test_sort_values_nat_na_position_default(self):
        # GH 13230
        expected = DataFrame(
            {
                "A": [1, 2, 3, 4, 4],
                "date": pd.DatetimeIndex(
                    [
                        "2010-01-01 09:00:00",
                        "2010-01-01 09:00:01",
                        "2010-01-01 09:00:02",
                        "2010-01-01 09:00:03",
                        "NaT",
                    ]
                ),
            }
        )
        result = expected.sort_values(["A", "date"])
        tm.assert_frame_equal(result, expected)

    def test_sort_values_item_cache(self, using_array_manager, using_copy_on_write):
        # previous behavior incorrect retained an invalid _item_cache entry
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 3)), columns=["A", "B", "C"]
        )
        df["D"] = df["A"] * 2
        ser = df["A"]
        if not using_array_manager:
            assert len(df._mgr.blocks) == 2

        df.sort_values(by="A")

        if using_copy_on_write:
            ser.iloc[0] = 99
            assert df.iloc[0, 0] == df["A"][0]
            assert df.iloc[0, 0] != 99
        else:
            ser.values[0] = 99
            assert df.iloc[0, 0] == df["A"][0]
            assert df.iloc[0, 0] == 99

    def test_sort_values_reshaping(self):
        # GH 39426
        values = list(range(21))
        expected = DataFrame([values], columns=values)
        df = expected.sort_values(expected.index[0], axis=1, ignore_index=True)

        tm.assert_frame_equal(df, expected)

    def test_sort_values_no_by_inplace(self):
        # GH#50643
        df = DataFrame({"a": [1, 2, 3]})
        expected = df.copy()
        result = df.sort_values(by=[], inplace=True)
        tm.assert_frame_equal(df, expected)
        assert result is None

    def test_sort_values_no_op_reset_index(self):
        # GH#52553
        df = DataFrame({"A": [10, 20], "B": [1, 5]}, index=[2, 3])
        result = df.sort_values(by="A", ignore_index=True)
        expected = DataFrame({"A": [10, 20], "B": [1, 5]})
        tm.assert_frame_equal(result, expected)


class TestDataFrameSortKey:  # test key sorting (issue 27237)
    def test_sort_values_inplace_key(self, sort_by_key):
        frame = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=[1, 2, 3, 4],
            columns=["A", "B", "C", "D"],
        )

        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(by="A", inplace=True, key=sort_by_key)
        assert return_value is None
        expected = frame.sort_values(by="A", key=sort_by_key)
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(
            by=1, axis=1, inplace=True, key=sort_by_key
        )
        assert return_value is None
        expected = frame.sort_values(by=1, axis=1, key=sort_by_key)
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(
            by="A", ascending=False, inplace=True, key=sort_by_key
        )
        assert return_value is None
        expected = frame.sort_values(by="A", ascending=False, key=sort_by_key)
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.copy()
        sorted_df.sort_values(
            by=["A", "B"], ascending=False, inplace=True, key=sort_by_key
        )
        expected = frame.sort_values(by=["A", "B"], ascending=False, key=sort_by_key)
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_key(self):
        df = DataFrame(np.array([0, 5, np.nan, 3, 2, np.nan]))

        result = df.sort_values(0)
        expected = df.iloc[[0, 4, 3, 1, 2, 5]]
        tm.assert_frame_equal(result, expected)

        result = df.sort_values(0, key=lambda x: x + 5)
        expected = df.iloc[[0, 4, 3, 1, 2, 5]]
        tm.assert_frame_equal(result, expected)

        result = df.sort_values(0, key=lambda x: -x, ascending=False)
        expected = df.iloc[[0, 4, 3, 1, 2, 5]]
        tm.assert_frame_equal(result, expected)

    def test_sort_values_by_key(self):
        df = DataFrame(
            {
                "a": np.array([0, 3, np.nan, 3, 2, np.nan]),
                "b": np.array([0, 2, np.nan, 5, 2, np.nan]),
            }
        )

        result = df.sort_values("a", key=lambda x: -x)
        expected = df.iloc[[1, 3, 4, 0, 2, 5]]
        tm.assert_frame_equal(result, expected)

        result = df.sort_values(by=["a", "b"], key=lambda x: -x)
        expected = df.iloc[[3, 1, 4, 0, 2, 5]]
        tm.assert_frame_equal(result, expected)

        result = df.sort_values(by=["a", "b"], key=lambda x: -x, ascending=False)
        expected = df.iloc[[0, 4, 1, 3, 2, 5]]
        tm.assert_frame_equal(result, expected)

    def test_sort_values_by_key_by_name(self):
        df = DataFrame(
            {
                "a": np.array([0, 3, np.nan, 3, 2, np.nan]),
                "b": np.array([0, 2, np.nan, 5, 2, np.nan]),
            }
        )

        def key(col):
            if col.name == "a":
                return -col
            else:
                return col

        result = df.sort_values(by="a", key=key)
        expected = df.iloc[[1, 3, 4, 0, 2, 5]]
        tm.assert_frame_equal(result, expected)

        result = df.sort_values(by=["a"], key=key)
        expected = df.iloc[[1, 3, 4, 0, 2, 5]]
        tm.assert_frame_equal(result, expected)

        result = df.sort_values(by="b", key=key)
        expected = df.iloc[[0, 1, 4, 3, 2, 5]]
        tm.assert_frame_equal(result, expected)

        result = df.sort_values(by=["a", "b"], key=key)
        expected = df.iloc[[1, 3, 4, 0, 2, 5]]
        tm.assert_frame_equal(result, expected)

    def test_sort_values_key_string(self):
        df = DataFrame(np.array([["hello", "goodbye"], ["hello", "Hello"]]))

        result = df.sort_values(1)
        expected = df[::-1]
        tm.assert_frame_equal(result, expected)

        result = df.sort_values([0, 1], key=lambda col: col.str.lower())
        tm.assert_frame_equal(result, df)

        result = df.sort_values(
            [0, 1], key=lambda col: col.str.lower(), ascending=False
        )
        expected = df.sort_values(1, key=lambda col: col.str.lower(), ascending=False)
        tm.assert_frame_equal(result, expected)

    def test_sort_values_key_empty(self, sort_by_key):
        df = DataFrame(np.array([]))

        df.sort_values(0, key=sort_by_key)
        df.sort_index(key=sort_by_key)

    def test_changes_length_raises(self):
        df = DataFrame({"A": [1, 2, 3]})
        with pytest.raises(ValueError, match="change the shape"):
            df.sort_values("A", key=lambda x: x[:1])

    def test_sort_values_key_axes(self):
        df = DataFrame({0: ["Hello", "goodbye"], 1: [0, 1]})

        result = df.sort_values(0, key=lambda col: col.str.lower())
        expected = df[::-1]
        tm.assert_frame_equal(result, expected)

        result = df.sort_values(1, key=lambda col: -col)
        expected = df[::-1]
        tm.assert_frame_equal(result, expected)

    def test_sort_values_key_dict_axis(self):
        df = DataFrame({0: ["Hello", 0], 1: ["goodbye", 1]})

        result = df.sort_values(0, key=lambda col: col.str.lower(), axis=1)
        expected = df.loc[:, ::-1]
        tm.assert_frame_equal(result, expected)

        result = df.sort_values(1, key=lambda col: -col, axis=1)
        expected = df.loc[:, ::-1]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("ordered", [True, False])
    def test_sort_values_key_casts_to_categorical(self, ordered):
        # https://github.com/pandas-dev/pandas/issues/36383
        categories = ["c", "b", "a"]
        df = DataFrame({"x": [1, 1, 1], "y": ["a", "b", "c"]})

        def sorter(key):
            if key.name == "y":
                return pd.Series(
                    Categorical(key, categories=categories, ordered=ordered)
                )
            return key

        result = df.sort_values(by=["x", "y"], key=sorter)
        expected = DataFrame(
            {"x": [1, 1, 1], "y": ["c", "b", "a"]}, index=pd.Index([2, 1, 0])
        )

        tm.assert_frame_equal(result, expected)


@pytest.fixture
def df_none():
    return DataFrame(
        {
            "outer": ["a", "a", "a", "b", "b", "b"],
            "inner": [1, 2, 2, 2, 1, 1],
            "A": np.arange(6, 0, -1),
            ("B", 5): ["one", "one", "two", "two", "one", "one"],
        }
    )


@pytest.fixture(params=[["outer"], ["outer", "inner"]])
def df_idx(request, df_none):
    levels = request.param
    return df_none.set_index(levels)


@pytest.fixture(
    params=[
        "inner",  # index level
        ["outer"],  # list of index level
        "A",  # column
        [("B", 5)],  # list of column
        ["inner", "outer"],  # two index levels
        [("B", 5), "outer"],  # index level and column
        ["A", ("B", 5)],  # Two columns
        ["inner", "outer"],  # two index levels and column
    ]
)
def sort_names(request):
    return request.param


@pytest.fixture(params=[True, False])
def ascending(request):
    return request.param


class TestSortValuesLevelAsStr:
    def test_sort_index_level_and_column_label(
        self, df_none, df_idx, sort_names, ascending, request
    ):
        # GH#14353
        if (
            Version(np.__version__) >= Version("1.25")
            and request.node.callspec.id == "df_idx0-inner-True"
        ):
            request.applymarker(
                pytest.mark.xfail(
                    reason=(
                        "pandas default unstable sorting of duplicates"
                        "issue with numpy>=1.25 with AVX instructions"
                    ),
                    strict=False,
                )
            )

        # Get index levels from df_idx
        levels = df_idx.index.names

        # Compute expected by sorting on columns and the setting index
        expected = df_none.sort_values(
            by=sort_names, ascending=ascending, axis=0
        ).set_index(levels)

        # Compute result sorting on mix on columns and index levels
        result = df_idx.sort_values(by=sort_names, ascending=ascending, axis=0)

        tm.assert_frame_equal(result, expected)

    def test_sort_column_level_and_index_label(
        self, df_none, df_idx, sort_names, ascending, request
    ):
        # GH#14353

        # Get levels from df_idx
        levels = df_idx.index.names

        # Compute expected by sorting on axis=0, setting index levels, and then
        # transposing. For some cases this will result in a frame with
        # multiple column levels
        expected = (
            df_none.sort_values(by=sort_names, ascending=ascending, axis=0)
            .set_index(levels)
            .T
        )

        # Compute result by transposing and sorting on axis=1.
        result = df_idx.T.sort_values(by=sort_names, ascending=ascending, axis=1)

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

        tm.assert_frame_equal(result, expected)

    def test_sort_values_validate_ascending_for_value_error(self):
        # GH41634
        df = DataFrame({"D": [23, 7, 21]})

        msg = 'For argument "ascending" expected type bool, received type str.'
        with pytest.raises(ValueError, match=msg):
            df.sort_values(by="D", ascending="False")

    @pytest.mark.parametrize("ascending", [False, 0, 1, True])
    def test_sort_values_validate_ascending_functional(self, ascending):
        df = DataFrame({"D": [23, 7, 21]})
        indexer = df["D"].argsort().values

        if not ascending:
            indexer = indexer[::-1]

        expected = df.loc[df.index[indexer]]
        result = df.sort_values(by="D", ascending=ascending)
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\merge\test_multi.py ===
import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    RangeIndex,
    Series,
    Timestamp,
    option_context,
)
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import merge


@pytest.fixture
def left():
    """left dataframe (not multi-indexed) for multi-index join tests"""
    # a little relevant example with NAs
    key1 = ["bar", "bar", "bar", "foo", "foo", "baz", "baz", "qux", "qux", "snap"]
    key2 = ["two", "one", "three", "one", "two", "one", "two", "two", "three", "one"]

    data = np.random.default_rng(2).standard_normal(len(key1))
    return DataFrame({"key1": key1, "key2": key2, "data": data})


@pytest.fixture
def right(multiindex_dataframe_random_data):
    """right dataframe (multi-indexed) for multi-index join tests"""
    df = multiindex_dataframe_random_data
    df.index.names = ["key1", "key2"]

    df.columns = ["j_one", "j_two", "j_three"]
    return df


@pytest.fixture
def left_multi():
    return DataFrame(
        {
            "Origin": ["A", "A", "B", "B", "C"],
            "Destination": ["A", "B", "A", "C", "A"],
            "Period": ["AM", "AM", "IP", "AM", "OP"],
            "TripPurp": ["hbw", "nhb", "hbo", "nhb", "hbw"],
            "Trips": [1987, 3647, 2470, 4296, 4444],
        },
        columns=["Origin", "Destination", "Period", "TripPurp", "Trips"],
    ).set_index(["Origin", "Destination", "Period", "TripPurp"])


@pytest.fixture
def right_multi():
    return DataFrame(
        {
            "Origin": ["A", "A", "B", "B", "C", "C", "E"],
            "Destination": ["A", "B", "A", "B", "A", "B", "F"],
            "Period": ["AM", "AM", "IP", "AM", "OP", "IP", "AM"],
            "LinkType": ["a", "b", "c", "b", "a", "b", "a"],
            "Distance": [100, 80, 90, 80, 75, 35, 55],
        },
        columns=["Origin", "Destination", "Period", "LinkType", "Distance"],
    ).set_index(["Origin", "Destination", "Period", "LinkType"])


@pytest.fixture
def on_cols_multi():
    return ["Origin", "Destination", "Period"]


class TestMergeMulti:
    def test_merge_on_multikey(self, left, right, join_type):
        on_cols = ["key1", "key2"]
        result = left.join(right, on=on_cols, how=join_type).reset_index(drop=True)

        expected = merge(left, right.reset_index(), on=on_cols, how=join_type)

        tm.assert_frame_equal(result, expected)

        result = left.join(right, on=on_cols, how=join_type, sort=True).reset_index(
            drop=True
        )

        expected = merge(
            left, right.reset_index(), on=on_cols, how=join_type, sort=True
        )

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "infer_string", [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))]
    )
    @pytest.mark.parametrize("sort", [True, False])
    def test_left_join_multi_index(self, sort, infer_string):
        with option_context("future.infer_string", infer_string):
            icols = ["1st", "2nd", "3rd"]

            def bind_cols(df):
                iord = lambda a: 0 if a != a else ord(a)
                f = lambda ts: ts.map(iord) - ord("a")
                return f(df["1st"]) + f(df["3rd"]) * 1e2 + df["2nd"].fillna(0) * 10

            def run_asserts(left, right, sort):
                res = left.join(right, on=icols, how="left", sort=sort)

                assert len(left) < len(res) + 1
                assert not res["4th"].isna().any()
                assert not res["5th"].isna().any()

                tm.assert_series_equal(res["4th"], -res["5th"], check_names=False)
                result = bind_cols(res.iloc[:, :-2])
                tm.assert_series_equal(res["4th"], result, check_names=False)
                assert result.name is None

                if sort:
                    tm.assert_frame_equal(res, res.sort_values(icols, kind="mergesort"))

                out = merge(left, right.reset_index(), on=icols, sort=sort, how="left")

                res.index = RangeIndex(len(res))
                tm.assert_frame_equal(out, res)

            lc = list(map(chr, np.arange(ord("a"), ord("z") + 1)))
            left = DataFrame(
                np.random.default_rng(2).choice(lc, (50, 2)), columns=["1st", "3rd"]
            )
            # Explicit cast to float to avoid implicit cast when setting nan
            left.insert(
                1,
                "2nd",
                np.random.default_rng(2).integers(0, 10, len(left)).astype("float"),
            )

            i = np.random.default_rng(2).permutation(len(left))
            right = left.iloc[i].copy()

            left["4th"] = bind_cols(left)
            right["5th"] = -bind_cols(right)
            right.set_index(icols, inplace=True)

            run_asserts(left, right, sort)

            # inject some nulls
            left.loc[1::4, "1st"] = np.nan
            left.loc[2::5, "2nd"] = np.nan
            left.loc[3::6, "3rd"] = np.nan
            left["4th"] = bind_cols(left)

            i = np.random.default_rng(2).permutation(len(left))
            right = left.iloc[i, :-1]
            right["5th"] = -bind_cols(right)
            right.set_index(icols, inplace=True)

            run_asserts(left, right, sort)

    @pytest.mark.parametrize("sort", [False, True])
    def test_merge_right_vs_left(self, left, right, sort):
        # compare left vs right merge with multikey
        on_cols = ["key1", "key2"]
        merged_left_right = left.merge(
            right, left_on=on_cols, right_index=True, how="left", sort=sort
        )

        merge_right_left = right.merge(
            left, right_on=on_cols, left_index=True, how="right", sort=sort
        )

        # Reorder columns
        merge_right_left = merge_right_left[merged_left_right.columns]

        tm.assert_frame_equal(merged_left_right, merge_right_left)

    def test_merge_multiple_cols_with_mixed_cols_index(self):
        # GH29522
        s = Series(
            range(6),
            MultiIndex.from_product([["A", "B"], [1, 2, 3]], names=["lev1", "lev2"]),
            name="Amount",
        )
        df = DataFrame({"lev1": list("AAABBB"), "lev2": [1, 2, 3, 1, 2, 3], "col": 0})
        result = merge(df, s.reset_index(), on=["lev1", "lev2"])
        expected = DataFrame(
            {
                "lev1": list("AAABBB"),
                "lev2": [1, 2, 3, 1, 2, 3],
                "col": [0] * 6,
                "Amount": range(6),
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_compress_group_combinations(self):
        # ~ 40000000 possible unique groups
        key1 = [str(i) for i in range(10000)]
        key1 = np.tile(key1, 2)
        key2 = key1[::-1]

        df = DataFrame(
            {
                "key1": key1,
                "key2": key2,
                "value1": np.random.default_rng(2).standard_normal(20000),
            }
        )

        df2 = DataFrame(
            {
                "key1": key1[::2],
                "key2": key2[::2],
                "value2": np.random.default_rng(2).standard_normal(10000),
            }
        )

        # just to hit the label compression code path
        merge(df, df2, how="outer")

    def test_left_join_index_preserve_order(self):
        on_cols = ["k1", "k2"]
        left = DataFrame(
            {
                "k1": [0, 1, 2] * 8,
                "k2": ["foo", "bar"] * 12,
                "v": np.array(np.arange(24), dtype=np.int64),
            }
        )

        index = MultiIndex.from_tuples([(2, "bar"), (1, "foo")])
        right = DataFrame({"v2": [5, 7]}, index=index)

        result = left.join(right, on=on_cols)

        expected = left.copy()
        expected["v2"] = np.nan
        expected.loc[(expected.k1 == 2) & (expected.k2 == "bar"), "v2"] = 5
        expected.loc[(expected.k1 == 1) & (expected.k2 == "foo"), "v2"] = 7

        tm.assert_frame_equal(result, expected)

        result.sort_values(on_cols, kind="mergesort", inplace=True)
        expected = left.join(right, on=on_cols, sort=True)

        tm.assert_frame_equal(result, expected)

        # test join with multi dtypes blocks
        left = DataFrame(
            {
                "k1": [0, 1, 2] * 8,
                "k2": ["foo", "bar"] * 12,
                "k3": np.array([0, 1, 2] * 8, dtype=np.float32),
                "v": np.array(np.arange(24), dtype=np.int32),
            }
        )

        index = MultiIndex.from_tuples([(2, "bar"), (1, "foo")])
        right = DataFrame({"v2": [5, 7]}, index=index)

        result = left.join(right, on=on_cols)

        expected = left.copy()
        expected["v2"] = np.nan
        expected.loc[(expected.k1 == 2) & (expected.k2 == "bar"), "v2"] = 5
        expected.loc[(expected.k1 == 1) & (expected.k2 == "foo"), "v2"] = 7

        tm.assert_frame_equal(result, expected)

        result = result.sort_values(on_cols, kind="mergesort")
        expected = left.join(right, on=on_cols, sort=True)

        tm.assert_frame_equal(result, expected)

    def test_left_join_index_multi_match_multiindex(self):
        left = DataFrame(
            [
                ["X", "Y", "C", "a"],
                ["W", "Y", "C", "e"],
                ["V", "Q", "A", "h"],
                ["V", "R", "D", "i"],
                ["X", "Y", "D", "b"],
                ["X", "Y", "A", "c"],
                ["W", "Q", "B", "f"],
                ["W", "R", "C", "g"],
                ["V", "Y", "C", "j"],
                ["X", "Y", "B", "d"],
            ],
            columns=["cola", "colb", "colc", "tag"],
            index=[3, 2, 0, 1, 7, 6, 4, 5, 9, 8],
        )

        right = DataFrame(
            [
                ["W", "R", "C", 0],
                ["W", "Q", "B", 3],
                ["W", "Q", "B", 8],
                ["X", "Y", "A", 1],
                ["X", "Y", "A", 4],
                ["X", "Y", "B", 5],
                ["X", "Y", "C", 6],
                ["X", "Y", "C", 9],
                ["X", "Q", "C", -6],
                ["X", "R", "C", -9],
                ["V", "Y", "C", 7],
                ["V", "R", "D", 2],
                ["V", "R", "D", -1],
                ["V", "Q", "A", -3],
            ],
            columns=["col1", "col2", "col3", "val"],
        ).set_index(["col1", "col2", "col3"])

        result = left.join(right, on=["cola", "colb", "colc"], how="left")

        expected = DataFrame(
            [
                ["X", "Y", "C", "a", 6],
                ["X", "Y", "C", "a", 9],
                ["W", "Y", "C", "e", np.nan],
                ["V", "Q", "A", "h", -3],
                ["V", "R", "D", "i", 2],
                ["V", "R", "D", "i", -1],
                ["X", "Y", "D", "b", np.nan],
                ["X", "Y", "A", "c", 1],
                ["X", "Y", "A", "c", 4],
                ["W", "Q", "B", "f", 3],
                ["W", "Q", "B", "f", 8],
                ["W", "R", "C", "g", 0],
                ["V", "Y", "C", "j", 7],
                ["X", "Y", "B", "d", 5],
            ],
            columns=["cola", "colb", "colc", "tag", "val"],
            index=[3, 3, 2, 0, 1, 1, 7, 6, 6, 4, 4, 5, 9, 8],
        )

        tm.assert_frame_equal(result, expected)

        result = left.join(right, on=["cola", "colb", "colc"], how="left", sort=True)

        expected = expected.sort_values(["cola", "colb", "colc"], kind="mergesort")

        tm.assert_frame_equal(result, expected)

    def test_left_join_index_multi_match(self):
        left = DataFrame(
            [["c", 0], ["b", 1], ["a", 2], ["b", 3]],
            columns=["tag", "val"],
            index=[2, 0, 1, 3],
        )

        right = DataFrame(
            [
                ["a", "v"],
                ["c", "w"],
                ["c", "x"],
                ["d", "y"],
                ["a", "z"],
                ["c", "r"],
                ["e", "q"],
                ["c", "s"],
            ],
            columns=["tag", "char"],
        ).set_index("tag")

        result = left.join(right, on="tag", how="left")

        expected = DataFrame(
            [
                ["c", 0, "w"],
                ["c", 0, "x"],
                ["c", 0, "r"],
                ["c", 0, "s"],
                ["b", 1, np.nan],
                ["a", 2, "v"],
                ["a", 2, "z"],
                ["b", 3, np.nan],
            ],
            columns=["tag", "val", "char"],
            index=[2, 2, 2, 2, 0, 1, 1, 3],
        )

        tm.assert_frame_equal(result, expected)

        result = left.join(right, on="tag", how="left", sort=True)
        expected2 = expected.sort_values("tag", kind="mergesort")

        tm.assert_frame_equal(result, expected2)

        # GH7331 - maintain left frame order in left merge
        result = merge(left, right.reset_index(), how="left", on="tag")
        expected.index = RangeIndex(len(expected))
        tm.assert_frame_equal(result, expected)

    def test_left_merge_na_buglet(self):
        left = DataFrame(
            {
                "id": list("abcde"),
                "v1": np.random.default_rng(2).standard_normal(5),
                "v2": np.random.default_rng(2).standard_normal(5),
                "dummy": list("abcde"),
                "v3": np.random.default_rng(2).standard_normal(5),
            },
            columns=["id", "v1", "v2", "dummy", "v3"],
        )
        right = DataFrame(
            {
                "id": ["a", "b", np.nan, np.nan, np.nan],
                "sv3": [1.234, 5.678, np.nan, np.nan, np.nan],
            }
        )

        result = merge(left, right, on="id", how="left")

        rdf = right.drop(["id"], axis=1)
        expected = left.join(rdf)
        tm.assert_frame_equal(result, expected)

    def test_merge_na_keys(self):
        data = [
            [1950, "A", 1.5],
            [1950, "B", 1.5],
            [1955, "B", 1.5],
            [1960, "B", np.nan],
            [1970, "B", 4.0],
            [1950, "C", 4.0],
            [1960, "C", np.nan],
            [1965, "C", 3.0],
            [1970, "C", 4.0],
        ]

        frame = DataFrame(data, columns=["year", "panel", "data"])

        other_data = [
            [1960, "A", np.nan],
            [1970, "A", np.nan],
            [1955, "A", np.nan],
            [1965, "A", np.nan],
            [1965, "B", np.nan],
            [1955, "C", np.nan],
        ]
        other = DataFrame(other_data, columns=["year", "panel", "data"])

        result = frame.merge(other, how="outer")

        expected = frame.fillna(-999).merge(other.fillna(-999), how="outer")
        expected = expected.replace(-999, np.nan)

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("klass", [None, np.asarray, Series, Index])
    def test_merge_datetime_index(self, klass):
        # see gh-19038
        df = DataFrame(
            [1, 2, 3], ["2016-01-01", "2017-01-01", "2018-01-01"], columns=["a"]
        )
        df.index = pd.to_datetime(df.index)
        on_vector = df.index.year

        if klass is not None:
            on_vector = klass(on_vector)

        exp_years = np.array([2016, 2017, 2018], dtype=np.int32)
        expected = DataFrame({"a": [1, 2, 3], "key_1": exp_years})

        result = df.merge(df, on=["a", on_vector], how="inner")
        tm.assert_frame_equal(result, expected)

        expected = DataFrame({"key_0": exp_years, "a_x": [1, 2, 3], "a_y": [1, 2, 3]})

        result = df.merge(df, on=[df.index.year], how="inner")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("merge_type", ["left", "right"])
    def test_merge_datetime_multi_index_empty_df(self, merge_type):
        # see gh-36895

        left = DataFrame(
            data={
                "data": [1.5, 1.5],
            },
            index=MultiIndex.from_tuples(
                [[Timestamp("1950-01-01"), "A"], [Timestamp("1950-01-02"), "B"]],
                names=["date", "panel"],
            ),
        )

        right = DataFrame(
            index=MultiIndex.from_tuples([], names=["date", "panel"]), columns=["state"]
        )

        expected_index = MultiIndex.from_tuples(
            [[Timestamp("1950-01-01"), "A"], [Timestamp("1950-01-02"), "B"]],
            names=["date", "panel"],
        )

        if merge_type == "left":
            expected = DataFrame(
                data={
                    "data": [1.5, 1.5],
                    "state": np.array([np.nan, np.nan], dtype=object),
                },
                index=expected_index,
            )
            results_merge = left.merge(right, how="left", on=["date", "panel"])
            results_join = left.join(right, how="left")
        else:
            expected = DataFrame(
                data={
                    "state": np.array([np.nan, np.nan], dtype=object),
                    "data": [1.5, 1.5],
                },
                index=expected_index,
            )
            results_merge = right.merge(left, how="right", on=["date", "panel"])
            results_join = right.join(left, how="right")

        tm.assert_frame_equal(results_merge, expected)
        tm.assert_frame_equal(results_join, expected)

    @pytest.fixture
    def household(self):
        household = DataFrame(
            {
                "household_id": [1, 2, 3],
                "male": [0, 1, 0],
                "wealth": [196087.3, 316478.7, 294750],
            },
            columns=["household_id", "male", "wealth"],
        ).set_index("household_id")
        return household

    @pytest.fixture
    def portfolio(self):
        portfolio = DataFrame(
            {
                "household_id": [1, 2, 2, 3, 3, 3, 4],
                "asset_id": [
                    "nl0000301109",
                    "nl0000289783",
                    "gb00b03mlx29",
                    "gb00b03mlx29",
                    "lu0197800237",
                    "nl0000289965",
                    np.nan,
                ],
                "name": [
                    "ABN Amro",
                    "Robeco",
                    "Royal Dutch Shell",
                    "Royal Dutch Shell",
                    "AAB Eastern Europe Equity Fund",
                    "Postbank BioTech Fonds",
                    np.nan,
                ],
                "share": [1.0, 0.4, 0.6, 0.15, 0.6, 0.25, 1.0],
            },
            columns=["household_id", "asset_id", "name", "share"],
        ).set_index(["household_id", "asset_id"])
        return portfolio

    @pytest.fixture
    def expected(self):
        expected = (
            DataFrame(
                {
                    "male": [0, 1, 1, 0, 0, 0],
                    "wealth": [
                        196087.3,
                        316478.7,
                        316478.7,
                        294750.0,
                        294750.0,
                        294750.0,
                    ],
                    "name": [
                        "ABN Amro",
                        "Robeco",
                        "Royal Dutch Shell",
                        "Royal Dutch Shell",
                        "AAB Eastern Europe Equity Fund",
                        "Postbank BioTech Fonds",
                    ],
                    "share": [1.00, 0.40, 0.60, 0.15, 0.60, 0.25],
                    "household_id": [1, 2, 2, 3, 3, 3],
                    "asset_id": [
                        "nl0000301109",
                        "nl0000289783",
                        "gb00b03mlx29",
                        "gb00b03mlx29",
                        "lu0197800237",
                        "nl0000289965",
                    ],
                }
            )
            .set_index(["household_id", "asset_id"])
            .reindex(columns=["male", "wealth", "name", "share"])
        )
        return expected

    def test_join_multi_levels(self, portfolio, household, expected):
        portfolio = portfolio.copy()
        household = household.copy()

        # GH 3662
        # merge multi-levels
        result = household.join(portfolio, how="inner")
        tm.assert_frame_equal(result, expected)

    def test_join_multi_levels_merge_equivalence(self, portfolio, household, expected):
        portfolio = portfolio.copy()
        household = household.copy()

        # equivalency
        result = merge(
            household.reset_index(),
            portfolio.reset_index(),
            on=["household_id"],
            how="inner",
        ).set_index(["household_id", "asset_id"])
        tm.assert_frame_equal(result, expected)

    def test_join_multi_levels_outer(self, portfolio, household, expected):
        portfolio = portfolio.copy()
        household = household.copy()

        result = household.join(portfolio, how="outer")
        expected = concat(
            [
                expected,
                (
                    DataFrame(
                        {"share": [1.00]},
                        index=MultiIndex.from_tuples(
                            [(4, np.nan)], names=["household_id", "asset_id"]
                        ),
                    )
                ),
            ],
            axis=0,
            sort=True,
        ).reindex(columns=expected.columns)
        tm.assert_frame_equal(result, expected, check_index_type=False)

    def test_join_multi_levels_invalid(self, portfolio, household):
        portfolio = portfolio.copy()
        household = household.copy()

        # invalid cases
        household.index.name = "foo"

        with pytest.raises(
            ValueError, match="cannot join with no overlapping index names"
        ):
            household.join(portfolio, how="inner")

        portfolio2 = portfolio.copy()
        portfolio2.index.set_names(["household_id", "foo"])

        with pytest.raises(ValueError, match="columns overlap but no suffix specified"):
            portfolio2.join(portfolio, how="inner")

    def test_join_multi_levels2(self):
        # some more advanced merges
        # GH6360
        household = DataFrame(
            {
                "household_id": [1, 2, 2, 3, 3, 3, 4],
                "asset_id": [
                    "nl0000301109",
                    "nl0000301109",
                    "gb00b03mlx29",
                    "gb00b03mlx29",
                    "lu0197800237",
                    "nl0000289965",
                    np.nan,
                ],
                "share": [1.0, 0.4, 0.6, 0.15, 0.6, 0.25, 1.0],
            },
            columns=["household_id", "asset_id", "share"],
        ).set_index(["household_id", "asset_id"])

        log_return = DataFrame(
            {
                "asset_id": [
                    "gb00b03mlx29",
                    "gb00b03mlx29",
                    "gb00b03mlx29",
                    "lu0197800237",
                    "lu0197800237",
                ],
                "t": [233, 234, 235, 180, 181],
                "log_return": [
                    0.09604978,
                    -0.06524096,
                    0.03532373,
                    0.03025441,
                    0.036997,
                ],
            }
        ).set_index(["asset_id", "t"])

        expected = (
            DataFrame(
                {
                    "household_id": [2, 2, 2, 3, 3, 3, 3, 3],
                    "asset_id": [
                        "gb00b03mlx29",
                        "gb00b03mlx29",
                        "gb00b03mlx29",
                        "gb00b03mlx29",
                        "gb00b03mlx29",
                        "gb00b03mlx29",
                        "lu0197800237",
                        "lu0197800237",
                    ],
                    "t": [233, 234, 235, 233, 234, 235, 180, 181],
                    "share": [0.6, 0.6, 0.6, 0.15, 0.15, 0.15, 0.6, 0.6],
                    "log_return": [
                        0.09604978,
                        -0.06524096,
                        0.03532373,
                        0.09604978,
                        -0.06524096,
                        0.03532373,
                        0.03025441,
                        0.036997,
                    ],
                }
            )
            .set_index(["household_id", "asset_id", "t"])
            .reindex(columns=["share", "log_return"])
        )

        # this is the equivalency
        result = merge(
            household.reset_index(),
            log_return.reset_index(),
            on=["asset_id"],
            how="inner",
        ).set_index(["household_id", "asset_id", "t"])
        tm.assert_frame_equal(result, expected)

        expected = (
            DataFrame(
                {
                    "household_id": [2, 2, 2, 3, 3, 3, 3, 3, 3, 1, 2, 4],
                    "asset_id": [
                        "gb00b03mlx29",
                        "gb00b03mlx29",
                        "gb00b03mlx29",
                        "gb00b03mlx29",
                        "gb00b03mlx29",
                        "gb00b03mlx29",
                        "lu0197800237",
                        "lu0197800237",
                        "nl0000289965",
                        "nl0000301109",
                        "nl0000301109",
                        None,
                    ],
                    "t": [
                        233,
                        234,
                        235,
                        233,
                        234,
                        235,
                        180,
                        181,
                        None,
                        None,
                        None,
                        None,
                    ],
                    "share": [
                        0.6,
                        0.6,
                        0.6,
                        0.15,
                        0.15,
                        0.15,
                        0.6,
                        0.6,
                        0.25,
                        1.0,
                        0.4,
                        1.0,
                    ],
                    "log_return": [
                        0.09604978,
                        -0.06524096,
                        0.03532373,
                        0.09604978,
                        -0.06524096,
                        0.03532373,
                        0.03025441,
                        0.036997,
                        None,
                        None,
                        None,
                        None,
                    ],
                }
            )
            .set_index(["household_id", "asset_id", "t"])
            .reindex(columns=["share", "log_return"])
        )

        result = merge(
            household.reset_index(),
            log_return.reset_index(),
            on=["asset_id"],
            how="outer",
        ).set_index(["household_id", "asset_id", "t"])

        tm.assert_frame_equal(result, expected)


class TestJoinMultiMulti:
    def test_join_multi_multi(self, left_multi, right_multi, join_type, on_cols_multi):
        left_names = left_multi.index.names
        right_names = right_multi.index.names
        if join_type == "right":
            level_order = right_names + left_names.difference(right_names)
        else:
            level_order = left_names + right_names.difference(left_names)
        # Multi-index join tests
        expected = (
            merge(
                left_multi.reset_index(),
                right_multi.reset_index(),
                how=join_type,
                on=on_cols_multi,
            )
            .set_index(level_order)
            .sort_index()
        )

        result = left_multi.join(right_multi, how=join_type).sort_index()
        tm.assert_frame_equal(result, expected)

    def test_join_multi_empty_frames(
        self, left_multi, right_multi, join_type, on_cols_multi
    ):
        left_multi = left_multi.drop(columns=left_multi.columns)
        right_multi = right_multi.drop(columns=right_multi.columns)

        left_names = left_multi.index.names
        right_names = right_multi.index.names
        if join_type == "right":
            level_order = right_names + left_names.difference(right_names)
        else:
            level_order = left_names + right_names.difference(left_names)

        expected = (
            merge(
                left_multi.reset_index(),
                right_multi.reset_index(),
                how=join_type,
                on=on_cols_multi,
            )
            .set_index(level_order)
            .sort_index()
        )

        result = left_multi.join(right_multi, how=join_type).sort_index()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("box", [None, np.asarray, Series, Index])
    def test_merge_datetime_index(self, box):
        # see gh-19038
        df = DataFrame(
            [1, 2, 3], ["2016-01-01", "2017-01-01", "2018-01-01"], columns=["a"]
        )
        df.index = pd.to_datetime(df.index)
        on_vector = df.index.year

        if box is not None:
            on_vector = box(on_vector)

        exp_years = np.array([2016, 2017, 2018], dtype=np.int32)
        expected = DataFrame({"a": [1, 2, 3], "key_1": exp_years})

        result = df.merge(df, on=["a", on_vector], how="inner")
        tm.assert_frame_equal(result, expected)

        expected = DataFrame({"key_0": exp_years, "a_x": [1, 2, 3], "a_y": [1, 2, 3]})

        result = df.merge(df, on=[df.index.year], how="inner")
        tm.assert_frame_equal(result, expected)

    def test_single_common_level(self):
        index_left = MultiIndex.from_tuples(
            [("K0", "X0"), ("K0", "X1"), ("K1", "X2")], names=["key", "X"]
        )

        left = DataFrame(
            {"A": ["A0", "A1", "A2"], "B": ["B0", "B1", "B2"]}, index=index_left
        )

        index_right = MultiIndex.from_tuples(
            [("K0", "Y0"), ("K1", "Y1"), ("K2", "Y2"), ("K2", "Y3")], names=["key", "Y"]
        )

        right = DataFrame(
            {"C": ["C0", "C1", "C2", "C3"], "D": ["D0", "D1", "D2", "D3"]},
            index=index_right,
        )

        result = left.join(right)
        expected = merge(
            left.reset_index(), right.reset_index(), on=["key"], how="inner"
        ).set_index(["key", "X", "Y"])

        tm.assert_frame_equal(result, expected)

    def test_join_multi_wrong_order(self):
        # GH 25760
        # GH 28956

        midx1 = MultiIndex.from_product([[1, 2], [3, 4]], names=["a", "b"])
        midx3 = MultiIndex.from_tuples([(4, 1), (3, 2), (3, 1)], names=["b", "a"])

        left = DataFrame(index=midx1, data={"x": [10, 20, 30, 40]})
        right = DataFrame(index=midx3, data={"y": ["foo", "bar", "fing"]})

        result = left.join(right)

        expected = DataFrame(
            index=midx1,
            data={"x": [10, 20, 30, 40], "y": ["fing", "foo", "bar", np.nan]},
        )

        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_mem_overlap.py ===
import itertools

import pytest
from numpy._core._multiarray_tests import internal_overlap, solve_diophantine

import numpy as np
from numpy._core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_, assert_array_equal, assert_equal, assert_raises

ndims = 2
size = 10
shape = tuple([size] * ndims)

MAY_SHARE_BOUNDS = 0
MAY_SHARE_EXACT = -1


def _indices_for_nelems(nelems):
    """Returns slices of length nelems, from start onwards, in direction sign."""

    if nelems == 0:
        return [size // 2]  # int index

    res = []
    for step in (1, 2):
        for sign in (-1, 1):
            start = size // 2 - nelems * step * sign // 2
            stop = start + nelems * step * sign
            res.append(slice(start, stop, step * sign))

    return res


def _indices_for_axis():
    """Returns (src, dst) pairs of indices."""

    res = []
    for nelems in (0, 2, 3):
        ind = _indices_for_nelems(nelems)
        res.extend(itertools.product(ind, ind))  # all assignments of size "nelems"

    return res


def _indices(ndims):
    """Returns ((axis0_src, axis0_dst), (axis1_src, axis1_dst), ... ) index pairs."""

    ind = _indices_for_axis()
    return itertools.product(ind, repeat=ndims)


def _check_assignment(srcidx, dstidx):
    """Check assignment arr[dstidx] = arr[srcidx] works."""

    arr = np.arange(np.prod(shape)).reshape(shape)

    cpy = arr.copy()

    cpy[dstidx] = arr[srcidx]
    arr[dstidx] = arr[srcidx]

    assert_(np.all(arr == cpy),
            f'assigning arr[{dstidx}] = arr[{srcidx}]')


def test_overlapping_assignments():
    # Test automatically generated assignments which overlap in memory.

    inds = _indices(ndims)

    for ind in inds:
        srcidx = tuple(a[0] for a in ind)
        dstidx = tuple(a[1] for a in ind)

        _check_assignment(srcidx, dstidx)


@pytest.mark.slow
def test_diophantine_fuzz():
    # Fuzz test the diophantine solver
    rng = np.random.RandomState(1234)

    max_int = np.iinfo(np.intp).max

    for ndim in range(10):
        feasible_count = 0
        infeasible_count = 0

        min_count = 500 // (ndim + 1)

        while min(feasible_count, infeasible_count) < min_count:
            # Ensure big and small integer problems
            A_max = 1 + rng.randint(0, 11, dtype=np.intp)**6
            U_max = rng.randint(0, 11, dtype=np.intp)**6

            A_max = min(max_int, A_max)
            U_max = min(max_int - 1, U_max)

            A = tuple(int(rng.randint(1, A_max + 1, dtype=np.intp))
                      for j in range(ndim))
            U = tuple(int(rng.randint(0, U_max + 2, dtype=np.intp))
                      for j in range(ndim))

            b_ub = min(max_int - 2, sum(a * ub for a, ub in zip(A, U)))
            b = int(rng.randint(-1, b_ub + 2, dtype=np.intp))

            if ndim == 0 and feasible_count < min_count:
                b = 0

            X = solve_diophantine(A, U, b)

            if X is None:
                # Check the simplified decision problem agrees
                X_simplified = solve_diophantine(A, U, b, simplify=1)
                assert_(X_simplified is None, (A, U, b, X_simplified))

                # Check no solution exists (provided the problem is
                # small enough so that brute force checking doesn't
                # take too long)
                ranges = tuple(range(0, a * ub + 1, a) for a, ub in zip(A, U))

                size = 1
                for r in ranges:
                    size *= len(r)
                if size < 100000:
                    assert_(not any(sum(w) == b for w in itertools.product(*ranges)))
                    infeasible_count += 1
            else:
                # Check the simplified decision problem agrees
                X_simplified = solve_diophantine(A, U, b, simplify=1)
                assert_(X_simplified is not None, (A, U, b, X_simplified))

                # Check validity
                assert_(sum(a * x for a, x in zip(A, X)) == b)
                assert_(all(0 <= x <= ub for x, ub in zip(X, U)))
                feasible_count += 1


def test_diophantine_overflow():
    # Smoke test integer overflow detection
    max_intp = np.iinfo(np.intp).max
    max_int64 = np.iinfo(np.int64).max

    if max_int64 <= max_intp:
        # Check that the algorithm works internally in 128-bit;
        # solving this problem requires large intermediate numbers
        A = (max_int64 // 2, max_int64 // 2 - 10)
        U = (max_int64 // 2, max_int64 // 2 - 10)
        b = 2 * (max_int64 // 2) - 10

        assert_equal(solve_diophantine(A, U, b), (1, 1))


def check_may_share_memory_exact(a, b):
    got = np.may_share_memory(a, b, max_work=MAY_SHARE_EXACT)

    assert_equal(np.may_share_memory(a, b),
                 np.may_share_memory(a, b, max_work=MAY_SHARE_BOUNDS))

    a.fill(0)
    b.fill(0)
    a.fill(1)
    exact = b.any()

    err_msg = ""
    if got != exact:
        err_msg = "    " + "\n    ".join([
            f"base_a - base_b = {a.__array_interface__['data'][0] - b.__array_interface__['data'][0]!r}",
            f"shape_a = {a.shape!r}",
            f"shape_b = {b.shape!r}",
            f"strides_a = {a.strides!r}",
            f"strides_b = {b.strides!r}",
            f"size_a = {a.size!r}",
            f"size_b = {b.size!r}"
        ])

    assert_equal(got, exact, err_msg=err_msg)


def test_may_share_memory_manual():
    # Manual test cases for may_share_memory

    # Base arrays
    xs0 = [
        np.zeros([13, 21, 23, 22], dtype=np.int8),
        np.zeros([13, 21, 23 * 2, 22], dtype=np.int8)[:, :, ::2, :]
    ]

    # Generate all negative stride combinations
    xs = []
    for x in xs0:
        for ss in itertools.product(*(([slice(None), slice(None, None, -1)],) * 4)):
            xp = x[ss]
            xs.append(xp)

    for x in xs:
        # The default is a simple extent check
        assert_(np.may_share_memory(x[:, 0, :], x[:, 1, :]))
        assert_(np.may_share_memory(x[:, 0, :], x[:, 1, :], max_work=None))

        # Exact checks
        check_may_share_memory_exact(x[:, 0, :], x[:, 1, :])
        check_may_share_memory_exact(x[:, ::7], x[:, 3::3])

        try:
            xp = x.ravel()
            if xp.flags.owndata:
                continue
            xp = xp.view(np.int16)
        except ValueError:
            continue

        # 0-size arrays cannot overlap
        check_may_share_memory_exact(x.ravel()[6:6],
                                     xp.reshape(13, 21, 23, 11)[:, ::7])

        # Test itemsize is dealt with
        check_may_share_memory_exact(x[:, ::7],
                                     xp.reshape(13, 21, 23, 11))
        check_may_share_memory_exact(x[:, ::7],
                                     xp.reshape(13, 21, 23, 11)[:, 3::3])
        check_may_share_memory_exact(x.ravel()[6:7],
                                     xp.reshape(13, 21, 23, 11)[:, ::7])

    # Check unit size
    x = np.zeros([1], dtype=np.int8)
    check_may_share_memory_exact(x, x)
    check_may_share_memory_exact(x, x.copy())


def iter_random_view_pairs(x, same_steps=True, equal_size=False):
    rng = np.random.RandomState(1234)

    if equal_size and same_steps:
        raise ValueError

    def random_slice(n, step):
        start = rng.randint(0, n + 1, dtype=np.intp)
        stop = rng.randint(start, n + 1, dtype=np.intp)
        if rng.randint(0, 2, dtype=np.intp) == 0:
            stop, start = start, stop
            step *= -1
        return slice(start, stop, step)

    def random_slice_fixed_size(n, step, size):
        start = rng.randint(0, n + 1 - size * step)
        stop = start + (size - 1) * step + 1
        if rng.randint(0, 2) == 0:
            stop, start = start - 1, stop - 1
            if stop < 0:
                stop = None
            step *= -1
        return slice(start, stop, step)

    # First a few regular views
    yield x, x
    for j in range(1, 7, 3):
        yield x[j:], x[:-j]
        yield x[..., j:], x[..., :-j]

    # An array with zero stride internal overlap
    strides = list(x.strides)
    strides[0] = 0
    xp = as_strided(x, shape=x.shape, strides=strides)
    yield x, xp
    yield xp, xp

    # An array with non-zero stride internal overlap
    strides = list(x.strides)
    if strides[0] > 1:
        strides[0] = 1
    xp = as_strided(x, shape=x.shape, strides=strides)
    yield x, xp
    yield xp, xp

    # Then discontiguous views
    while True:
        steps = tuple(rng.randint(1, 11, dtype=np.intp)
                      if rng.randint(0, 5, dtype=np.intp) == 0 else 1
                      for j in range(x.ndim))
        s1 = tuple(random_slice(p, s) for p, s in zip(x.shape, steps))

        t1 = np.arange(x.ndim)
        rng.shuffle(t1)

        if equal_size:
            t2 = t1
        else:
            t2 = np.arange(x.ndim)
            rng.shuffle(t2)

        a = x[s1]

        if equal_size:
            if a.size == 0:
                continue

            steps2 = tuple(rng.randint(1, max(2, p // (1 + pa)))
                           if rng.randint(0, 5) == 0 else 1
                           for p, s, pa in zip(x.shape, s1, a.shape))
            s2 = tuple(random_slice_fixed_size(p, s, pa)
                       for p, s, pa in zip(x.shape, steps2, a.shape))
        elif same_steps:
            steps2 = steps
        else:
            steps2 = tuple(rng.randint(1, 11, dtype=np.intp)
                           if rng.randint(0, 5, dtype=np.intp) == 0 else 1
                           for j in range(x.ndim))

        if not equal_size:
            s2 = tuple(random_slice(p, s) for p, s in zip(x.shape, steps2))

        a = a.transpose(t1)
        b = x[s2].transpose(t2)

        yield a, b


def check_may_share_memory_easy_fuzz(get_max_work, same_steps, min_count):
    # Check that overlap problems with common strides are solved with
    # little work.
    x = np.zeros([17, 34, 71, 97], dtype=np.int16)

    feasible = 0
    infeasible = 0

    pair_iter = iter_random_view_pairs(x, same_steps)

    while min(feasible, infeasible) < min_count:
        a, b = next(pair_iter)

        bounds_overlap = np.may_share_memory(a, b)
        may_share_answer = np.may_share_memory(a, b)
        easy_answer = np.may_share_memory(a, b, max_work=get_max_work(a, b))
        exact_answer = np.may_share_memory(a, b, max_work=MAY_SHARE_EXACT)

        if easy_answer != exact_answer:
            # assert_equal is slow...
            assert_equal(easy_answer, exact_answer)

        if may_share_answer != bounds_overlap:
            assert_equal(may_share_answer, bounds_overlap)

        if bounds_overlap:
            if exact_answer:
                feasible += 1
            else:
                infeasible += 1


@pytest.mark.slow
def test_may_share_memory_easy_fuzz():
    # Check that overlap problems with common strides are always
    # solved with little work.

    check_may_share_memory_easy_fuzz(get_max_work=lambda a, b: 1,
                                     same_steps=True,
                                     min_count=2000)


@pytest.mark.slow
def test_may_share_memory_harder_fuzz():
    # Overlap problems with not necessarily common strides take more
    # work.
    #
    # The work bound below can't be reduced much. Harder problems can
    # also exist but not be detected here, as the set of problems
    # comes from RNG.

    check_may_share_memory_easy_fuzz(get_max_work=lambda a, b: max(a.size, b.size) // 2,
                                     same_steps=False,
                                     min_count=2000)


def test_shares_memory_api():
    x = np.zeros([4, 5, 6], dtype=np.int8)

    assert_equal(np.shares_memory(x, x), True)
    assert_equal(np.shares_memory(x, x.copy()), False)

    a = x[:, ::2, ::3]
    b = x[:, ::3, ::2]
    assert_equal(np.shares_memory(a, b), True)
    assert_equal(np.shares_memory(a, b, max_work=None), True)
    assert_raises(
        np.exceptions.TooHardError, np.shares_memory, a, b, max_work=1
    )


def test_may_share_memory_bad_max_work():
    x = np.zeros([1])
    assert_raises(OverflowError, np.may_share_memory, x, x, max_work=10**100)
    assert_raises(OverflowError, np.shares_memory, x, x, max_work=10**100)


def test_internal_overlap_diophantine():
    def check(A, U, exists=None):
        X = solve_diophantine(A, U, 0, require_ub_nontrivial=1)

        if exists is None:
            exists = (X is not None)

        if X is not None:
            assert_(sum(a * x for a, x in zip(A, X)) == sum(a * u // 2 for a, u in zip(A, U)))
            assert_(all(0 <= x <= u for x, u in zip(X, U)))
            assert_(any(x != u // 2 for x, u in zip(X, U)))

        if exists:
            assert_(X is not None, repr(X))
        else:
            assert_(X is None, repr(X))

    # Smoke tests
    check((3, 2), (2 * 2, 3 * 2), exists=True)
    check((3 * 2, 2), (15 * 2, (3 - 1) * 2), exists=False)


def test_internal_overlap_slices():
    # Slicing an array never generates internal overlap

    x = np.zeros([17, 34, 71, 97], dtype=np.int16)

    rng = np.random.RandomState(1234)

    def random_slice(n, step):
        start = rng.randint(0, n + 1, dtype=np.intp)
        stop = rng.randint(start, n + 1, dtype=np.intp)
        if rng.randint(0, 2, dtype=np.intp) == 0:
            stop, start = start, stop
            step *= -1
        return slice(start, stop, step)

    cases = 0
    min_count = 5000

    while cases < min_count:
        steps = tuple(rng.randint(1, 11, dtype=np.intp)
                      if rng.randint(0, 5, dtype=np.intp) == 0 else 1
                      for j in range(x.ndim))
        t1 = np.arange(x.ndim)
        rng.shuffle(t1)
        s1 = tuple(random_slice(p, s) for p, s in zip(x.shape, steps))
        a = x[s1].transpose(t1)

        assert_(not internal_overlap(a))
        cases += 1


def check_internal_overlap(a, manual_expected=None):
    got = internal_overlap(a)

    # Brute-force check
    m = set()
    ranges = tuple(range(n) for n in a.shape)
    for v in itertools.product(*ranges):
        offset = sum(s * w for s, w in zip(a.strides, v))
        if offset in m:
            expected = True
            break
        else:
            m.add(offset)
    else:
        expected = False

    # Compare
    if got != expected:
        assert_equal(got, expected, err_msg=repr((a.strides, a.shape)))
    if manual_expected is not None and expected != manual_expected:
        assert_equal(expected, manual_expected)
    return got


def test_internal_overlap_manual():
    # Stride tricks can construct arrays with internal overlap

    # We don't care about memory bounds, the array is not
    # read/write accessed
    x = np.arange(1).astype(np.int8)

    # Check low-dimensional special cases

    check_internal_overlap(x, False)  # 1-dim
    check_internal_overlap(x.reshape([]), False)  # 0-dim

    a = as_strided(x, strides=(3, 4), shape=(4, 4))
    check_internal_overlap(a, False)

    a = as_strided(x, strides=(3, 4), shape=(5, 4))
    check_internal_overlap(a, True)

    a = as_strided(x, strides=(0,), shape=(0,))
    check_internal_overlap(a, False)

    a = as_strided(x, strides=(0,), shape=(1,))
    check_internal_overlap(a, False)

    a = as_strided(x, strides=(0,), shape=(2,))
    check_internal_overlap(a, True)

    a = as_strided(x, strides=(0, -9993), shape=(87, 22))
    check_internal_overlap(a, True)

    a = as_strided(x, strides=(0, -9993), shape=(1, 22))
    check_internal_overlap(a, False)

    a = as_strided(x, strides=(0, -9993), shape=(0, 22))
    check_internal_overlap(a, False)


def test_internal_overlap_fuzz():
    # Fuzz check; the brute-force check is fairly slow

    x = np.arange(1).astype(np.int8)

    overlap = 0
    no_overlap = 0
    min_count = 100

    rng = np.random.RandomState(1234)

    while min(overlap, no_overlap) < min_count:
        ndim = rng.randint(1, 4, dtype=np.intp)

        strides = tuple(rng.randint(-1000, 1000, dtype=np.intp)
                        for j in range(ndim))
        shape = tuple(rng.randint(1, 30, dtype=np.intp)
                      for j in range(ndim))

        a = as_strided(x, strides=strides, shape=shape)
        result = check_internal_overlap(a)

        if result:
            overlap += 1
        else:
            no_overlap += 1


def test_non_ndarray_inputs():
    # Regression check for gh-5604

    class MyArray:
        def __init__(self, data):
            self.data = data

        @property
        def __array_interface__(self):
            return self.data.__array_interface__

    class MyArray2:
        def __init__(self, data):
            self.data = data

        def __array__(self, dtype=None, copy=None):
            return self.data

    for cls in [MyArray, MyArray2]:
        x = np.arange(5)

        assert_(np.may_share_memory(cls(x[::2]), x[1::2]))
        assert_(not np.shares_memory(cls(x[::2]), x[1::2]))

        assert_(np.shares_memory(cls(x[1::3]), x[::2]))
        assert_(np.may_share_memory(cls(x[1::3]), x[::2]))


def view_element_first_byte(x):
    """Construct an array viewing the first byte of each element of `x`"""
    from numpy.lib._stride_tricks_impl import DummyArray
    interface = dict(x.__array_interface__)
    interface['typestr'] = '|b1'
    interface['descr'] = [('', '|b1')]
    return np.asarray(DummyArray(interface, x))


def assert_copy_equivalent(operation, args, out, **kwargs):
    """
    Check that operation(*args, out=out) produces results
    equivalent to out[...] = operation(*args, out=out.copy())
    """

    kwargs['out'] = out
    kwargs2 = dict(kwargs)
    kwargs2['out'] = out.copy()

    out_orig = out.copy()
    out[...] = operation(*args, **kwargs2)
    expected = out.copy()
    out[...] = out_orig

    got = operation(*args, **kwargs).copy()

    if (got != expected).any():
        assert_equal(got, expected)


class TestUFunc:
    """
    Test ufunc call memory overlap handling
    """

    def check_unary_fuzz(self, operation, get_out_axis_size, dtype=np.int16,
                             count=5000):
        shapes = [7, 13, 8, 21, 29, 32]

        rng = np.random.RandomState(1234)

        for ndim in range(1, 6):
            x = rng.randint(0, 2**16, size=shapes[:ndim]).astype(dtype)

            it = iter_random_view_pairs(x, same_steps=False, equal_size=True)

            min_count = count // (ndim + 1)**2

            overlapping = 0
            while overlapping < min_count:
                a, b = next(it)

                a_orig = a.copy()
                b_orig = b.copy()

                if get_out_axis_size is None:
                    assert_copy_equivalent(operation, [a], out=b)

                    if np.shares_memory(a, b):
                        overlapping += 1
                else:
                    for axis in itertools.chain(range(ndim), [None]):
                        a[...] = a_orig
                        b[...] = b_orig

                        # Determine size for reduction axis (None if scalar)
                        outsize, scalarize = get_out_axis_size(a, b, axis)
                        if outsize == 'skip':
                            continue

                        # Slice b to get an output array of the correct size
                        sl = [slice(None)] * ndim
                        if axis is None:
                            if outsize is None:
                                sl = [slice(0, 1)] + [0] * (ndim - 1)
                            else:
                                sl = [slice(0, outsize)] + [0] * (ndim - 1)
                        elif outsize is None:
                            k = b.shape[axis] // 2
                            if ndim == 1:
                                sl[axis] = slice(k, k + 1)
                            else:
                                sl[axis] = k
                        else:
                            assert b.shape[axis] >= outsize
                            sl[axis] = slice(0, outsize)
                        b_out = b[tuple(sl)]

                        if scalarize:
                            b_out = b_out.reshape([])

                        if np.shares_memory(a, b_out):
                            overlapping += 1

                        # Check result
                        assert_copy_equivalent(operation, [a], out=b_out, axis=axis)

    @pytest.mark.slow
    def test_unary_ufunc_call_fuzz(self):
        self.check_unary_fuzz(np.invert, None, np.int16)

    @pytest.mark.slow
    def test_unary_ufunc_call_complex_fuzz(self):
        # Complex typically has a smaller alignment than itemsize
        self.check_unary_fuzz(np.negative, None, np.complex128, count=500)

    def test_binary_ufunc_accumulate_fuzz(self):
        def get_out_axis_size(a, b, axis):
            if axis is None:
                if a.ndim == 1:
                    return a.size, False
                else:
                    return 'skip', False  # accumulate doesn't support this
            else:
                return a.shape[axis], False

        self.check_unary_fuzz(np.add.accumulate, get_out_axis_size,
                              dtype=np.int16, count=500)

    def test_binary_ufunc_reduce_fuzz(self):
        def get_out_axis_size(a, b, axis):
            return None, (axis is None or a.ndim == 1)

        self.check_unary_fuzz(np.add.reduce, get_out_axis_size,
                              dtype=np.int16, count=500)

    def test_binary_ufunc_reduceat_fuzz(self):
        def get_out_axis_size(a, b, axis):
            if axis is None:
                if a.ndim == 1:
                    return a.size, False
                else:
                    return 'skip', False  # reduceat doesn't support this
            else:
                return a.shape[axis], False

        def do_reduceat(a, out, axis):
            if axis is None:
                size = len(a)
                step = size // len(out)
            else:
                size = a.shape[axis]
                step = a.shape[axis] // out.shape[axis]
            idx = np.arange(0, size, step)
            return np.add.reduceat(a, idx, out=out, axis=axis)

        self.check_unary_fuzz(do_reduceat, get_out_axis_size,
                              dtype=np.int16, count=500)

    def test_binary_ufunc_reduceat_manual(self):
        def check(ufunc, a, ind, out):
            c1 = ufunc.reduceat(a.copy(), ind.copy(), out=out.copy())
            c2 = ufunc.reduceat(a, ind, out=out)
            assert_array_equal(c1, c2)

        # Exactly same input/output arrays
        a = np.arange(10000, dtype=np.int16)
        check(np.add, a, a[::-1].copy(), a)

        # Overlap with index
        a = np.arange(10000, dtype=np.int16)
        check(np.add, a, a[::-1], a)

    @pytest.mark.slow
    def test_unary_gufunc_fuzz(self):
        shapes = [7, 13, 8, 21, 29, 32]
        gufunc = _umath_tests.euclidean_pdist

        rng = np.random.RandomState(1234)

        for ndim in range(2, 6):
            x = rng.rand(*shapes[:ndim])

            it = iter_random_view_pairs(x, same_steps=False, equal_size=True)

            min_count = 500 // (ndim + 1)**2

            overlapping = 0
            while overlapping < min_count:
                a, b = next(it)

                if min(a.shape[-2:]) < 2 or min(b.shape[-2:]) < 2 or a.shape[-1] < 2:
                    continue

                # Ensure the shapes are so that euclidean_pdist is happy
                if b.shape[-1] > b.shape[-2]:
                    b = b[..., 0, :]
                else:
                    b = b[..., :, 0]

                n = a.shape[-2]
                p = n * (n - 1) // 2
                if p <= b.shape[-1] and p > 0:
                    b = b[..., :p]
                else:
                    n = max(2, int(np.sqrt(b.shape[-1])) // 2)
                    p = n * (n - 1) // 2
                    a = a[..., :n, :]
                    b = b[..., :p]

                # Call
                if np.shares_memory(a, b):
                    overlapping += 1

                with np.errstate(over='ignore', invalid='ignore'):
                    assert_copy_equivalent(gufunc, [a], out=b)

    def test_ufunc_at_manual(self):
        def check(ufunc, a, ind, b=None):
            a0 = a.copy()
            if b is None:
                ufunc.at(a0, ind.copy())
                c1 = a0.copy()
                ufunc.at(a, ind)
                c2 = a.copy()
            else:
                ufunc.at(a0, ind.copy(), b.copy())
                c1 = a0.copy()
                ufunc.at(a, ind, b)
                c2 = a.copy()
            assert_array_equal(c1, c2)

        # Overlap with index
        a = np.arange(10000, dtype=np.int16)
        check(np.invert, a[::-1], a)

        # Overlap with second data array
        a = np.arange(100, dtype=np.int16)
        ind = np.arange(0, 100, 2, dtype=np.int16)
        check(np.add, a, ind, a[25:75])

    def test_unary_ufunc_1d_manual(self):
        # Exercise ufunc fast-paths (that avoid creation of an `np.nditer`)

        def check(a, b):
            a_orig = a.copy()
            b_orig = b.copy()

            b0 = b.copy()
            c1 = ufunc(a, out=b0)
            c2 = ufunc(a, out=b)
            assert_array_equal(c1, c2)

            # Trigger "fancy ufunc loop" code path
            mask = view_element_first_byte(b).view(np.bool)

            a[...] = a_orig
            b[...] = b_orig
            c1 = ufunc(a, out=b.copy(), where=mask.copy()).copy()

            a[...] = a_orig
            b[...] = b_orig
            c2 = ufunc(a, out=b, where=mask.copy()).copy()

            # Also, mask overlapping with output
            a[...] = a_orig
            b[...] = b_orig
            c3 = ufunc(a, out=b, where=mask).copy()

            assert_array_equal(c1, c2)
            assert_array_equal(c1, c3)

        dtypes = [np.int8, np.int16, np.int32, np.int64, np.float32,
                  np.float64, np.complex64, np.complex128]
        dtypes = [np.dtype(x) for x in dtypes]

        for dtype in dtypes:
            if np.issubdtype(dtype, np.integer):
                ufunc = np.invert
            else:
                ufunc = np.reciprocal

            n = 1000
            k = 10
            indices = [
                np.index_exp[:n],
                np.index_exp[k:k + n],
                np.index_exp[n - 1::-1],
                np.index_exp[k + n - 1:k - 1:-1],
                np.index_exp[:2 * n:2],
                np.index_exp[k:k + 2 * n:2],
                np.index_exp[2 * n - 1::-2],
                np.index_exp[k + 2 * n - 1:k - 1:-2],
            ]

            for xi, yi in itertools.product(indices, indices):
                v = np.arange(1, 1 + n * 2 + k, dtype=dtype)
                x = v[xi]
                y = v[yi]

                with np.errstate(all='ignore'):
                    check(x, y)

                    # Scalar cases
                    check(x[:1], y)
                    check(x[-1:], y)
                    check(x[:1].reshape([]), y)
                    check(x[-1:].reshape([]), y)

    def test_unary_ufunc_where_same(self):
        # Check behavior at wheremask overlap
        ufunc = np.invert

        def check(a, out, mask):
            c1 = ufunc(a, out=out.copy(), where=mask.copy())
            c2 = ufunc(a, out=out, where=mask)
            assert_array_equal(c1, c2)

        # Check behavior with same input and output arrays
        x = np.arange(100).astype(np.bool)
        check(x, x, x)
        check(x, x.copy(), x)
        check(x, x, x.copy())

    @pytest.mark.slow
    def test_binary_ufunc_1d_manual(self):
        ufunc = np.add

        def check(a, b, c):
            c0 = c.copy()
            c1 = ufunc(a, b, out=c0)
            c2 = ufunc(a, b, out=c)
            assert_array_equal(c1, c2)

        for dtype in [np.int8, np.int16, np.int32, np.int64,
                      np.float32, np.float64, np.complex64, np.complex128]:
            # Check different data dependency orders

            n = 1000
            k = 10

            indices = []
            for p in [1, 2]:
                indices.extend([
                    np.index_exp[:p * n:p],
                    np.index_exp[k:k + p * n:p],
                    np.index_exp[p * n - 1::-p],
                    np.index_exp[k + p * n - 1:k - 1:-p],
                ])

            for x, y, z in itertools.product(indices, indices, indices):
                v = np.arange(6 * n).astype(dtype)
                x = v[x]
                y = v[y]
                z = v[z]

                check(x, y, z)

                # Scalar cases
                check(x[:1], y, z)
                check(x[-1:], y, z)
                check(x[:1].reshape([]), y, z)
                check(x[-1:].reshape([]), y, z)
                check(x, y[:1], z)
                check(x, y[-1:], z)
                check(x, y[:1].reshape([]), z)
                check(x, y[-1:].reshape([]), z)

    def test_inplace_op_simple_manual(self):
        rng = np.random.RandomState(1234)
        x = rng.rand(200, 200)  # bigger than bufsize

        x += x.T
        assert_array_equal(x - x.T, 0)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\timestamp\test_timestamp.py ===
""" test the scalar Timestamp """

import calendar
from datetime import (
    datetime,
    timedelta,
    timezone,
)
import locale
import time
import unicodedata

from dateutil.tz import (
    tzlocal,
    tzutc,
)
from hypothesis import (
    given,
    strategies as st,
)
import numpy as np
import pytest
import pytz
from pytz import utc

from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.timezones import (
    dateutil_gettz as gettz,
    get_timezone,
    maybe_get_tz,
    tz_compare,
)
from pandas.compat import IS64

from pandas import (
    NaT,
    Timedelta,
    Timestamp,
)
import pandas._testing as tm

from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TestTimestampProperties:
    def test_properties_business(self):
        freq = to_offset("B")

        ts = Timestamp("2017-10-01")
        assert ts.dayofweek == 6
        assert ts.day_of_week == 6
        assert ts.is_month_start  # not a weekday
        assert not freq.is_month_start(ts)
        assert freq.is_month_start(ts + Timedelta(days=1))
        assert not freq.is_quarter_start(ts)
        assert freq.is_quarter_start(ts + Timedelta(days=1))

        ts = Timestamp("2017-09-30")
        assert ts.dayofweek == 5
        assert ts.day_of_week == 5
        assert ts.is_month_end
        assert not freq.is_month_end(ts)
        assert freq.is_month_end(ts - Timedelta(days=1))
        assert ts.is_quarter_end
        assert not freq.is_quarter_end(ts)
        assert freq.is_quarter_end(ts - Timedelta(days=1))

    @pytest.mark.parametrize(
        "attr, expected",
        [
            ["year", 2014],
            ["month", 12],
            ["day", 31],
            ["hour", 23],
            ["minute", 59],
            ["second", 0],
            ["microsecond", 0],
            ["nanosecond", 0],
            ["dayofweek", 2],
            ["day_of_week", 2],
            ["quarter", 4],
            ["dayofyear", 365],
            ["day_of_year", 365],
            ["week", 1],
            ["daysinmonth", 31],
        ],
    )
    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    def test_fields(self, attr, expected, tz):
        # GH 10050
        # GH 13303
        ts = Timestamp("2014-12-31 23:59:00", tz=tz)
        result = getattr(ts, attr)
        # that we are int like
        assert isinstance(result, int)
        assert result == expected

    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    def test_millisecond_raises(self, tz):
        ts = Timestamp("2014-12-31 23:59:00", tz=tz)
        msg = "'Timestamp' object has no attribute 'millisecond'"
        with pytest.raises(AttributeError, match=msg):
            ts.millisecond

    @pytest.mark.parametrize(
        "start", ["is_month_start", "is_quarter_start", "is_year_start"]
    )
    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    def test_is_start(self, start, tz):
        ts = Timestamp("2014-01-01 00:00:00", tz=tz)
        assert getattr(ts, start)

    @pytest.mark.parametrize("end", ["is_month_end", "is_year_end", "is_quarter_end"])
    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    def test_is_end(self, end, tz):
        ts = Timestamp("2014-12-31 23:59:59", tz=tz)
        assert getattr(ts, end)

    # GH 12806
    @pytest.mark.parametrize(
        "data",
        [Timestamp("2017-08-28 23:00:00"), Timestamp("2017-08-28 23:00:00", tz="EST")],
    )
    # error: Unsupported operand types for + ("List[None]" and "List[str]")
    @pytest.mark.parametrize(
        "time_locale", [None] + tm.get_locales()  # type: ignore[operator]
    )
    def test_names(self, data, time_locale):
        # GH 17354
        # Test .day_name(), .month_name
        if time_locale is None:
            expected_day = "Monday"
            expected_month = "August"
        else:
            with tm.set_locale(time_locale, locale.LC_TIME):
                expected_day = calendar.day_name[0].capitalize()
                expected_month = calendar.month_name[8].capitalize()

        result_day = data.day_name(time_locale)
        result_month = data.month_name(time_locale)

        # Work around https://github.com/pandas-dev/pandas/issues/22342
        # different normalizations
        expected_day = unicodedata.normalize("NFD", expected_day)
        expected_month = unicodedata.normalize("NFD", expected_month)

        result_day = unicodedata.normalize("NFD", result_day)
        result_month = unicodedata.normalize("NFD", result_month)

        assert result_day == expected_day
        assert result_month == expected_month

        # Test NaT
        nan_ts = Timestamp(NaT)
        assert np.isnan(nan_ts.day_name(time_locale))
        assert np.isnan(nan_ts.month_name(time_locale))

    def test_is_leap_year(self, tz_naive_fixture):
        tz = tz_naive_fixture
        if not IS64 and tz == tzlocal():
            # https://github.com/dateutil/dateutil/issues/197
            pytest.skip(
                "tzlocal() on a 32 bit platform causes internal overflow errors"
            )
        # GH 13727
        dt = Timestamp("2000-01-01 00:00:00", tz=tz)
        assert dt.is_leap_year
        assert isinstance(dt.is_leap_year, bool)

        dt = Timestamp("1999-01-01 00:00:00", tz=tz)
        assert not dt.is_leap_year

        dt = Timestamp("2004-01-01 00:00:00", tz=tz)
        assert dt.is_leap_year

        dt = Timestamp("2100-01-01 00:00:00", tz=tz)
        assert not dt.is_leap_year

    def test_woy_boundary(self):
        # make sure weeks at year boundaries are correct
        d = datetime(2013, 12, 31)
        result = Timestamp(d).week
        expected = 1  # ISO standard
        assert result == expected

        d = datetime(2008, 12, 28)
        result = Timestamp(d).week
        expected = 52  # ISO standard
        assert result == expected

        d = datetime(2009, 12, 31)
        result = Timestamp(d).week
        expected = 53  # ISO standard
        assert result == expected

        d = datetime(2010, 1, 1)
        result = Timestamp(d).week
        expected = 53  # ISO standard
        assert result == expected

        d = datetime(2010, 1, 3)
        result = Timestamp(d).week
        expected = 53  # ISO standard
        assert result == expected

        result = np.array(
            [
                Timestamp(datetime(*args)).week
                for args in [(2000, 1, 1), (2000, 1, 2), (2005, 1, 1), (2005, 1, 2)]
            ]
        )
        assert (result == [52, 52, 53, 53]).all()

    def test_resolution(self):
        # GH#21336, GH#21365
        dt = Timestamp("2100-01-01 00:00:00.000000000")
        assert dt.resolution == Timedelta(nanoseconds=1)

        # Check that the attribute is available on the class, mirroring
        #  the stdlib datetime behavior
        assert Timestamp.resolution == Timedelta(nanoseconds=1)

        assert dt.as_unit("us").resolution == Timedelta(microseconds=1)
        assert dt.as_unit("ms").resolution == Timedelta(milliseconds=1)
        assert dt.as_unit("s").resolution == Timedelta(seconds=1)

    @pytest.mark.parametrize(
        "date_string, expected",
        [
            ("0000-2-29", 1),
            ("0000-3-1", 2),
            ("1582-10-14", 3),
            ("-0040-1-1", 4),
            ("2023-06-18", 6),
        ],
    )
    def test_dow_historic(self, date_string, expected):
        # GH 53738
        ts = Timestamp(date_string)
        dow = ts.weekday()
        assert dow == expected

    @given(
        ts=st.datetimes(),
        sign=st.sampled_from(["-", ""]),
    )
    def test_dow_parametric(self, ts, sign):
        # GH 53738
        ts = (
            f"{sign}{str(ts.year).zfill(4)}"
            f"-{str(ts.month).zfill(2)}"
            f"-{str(ts.day).zfill(2)}"
        )
        result = Timestamp(ts).weekday()
        expected = (
            (np.datetime64(ts) - np.datetime64("1970-01-01")).astype("int64") - 4
        ) % 7
        assert result == expected


class TestTimestamp:
    @pytest.mark.parametrize("tz", [None, pytz.timezone("US/Pacific")])
    def test_disallow_setting_tz(self, tz):
        # GH#3746
        ts = Timestamp("2010")
        msg = "Cannot directly set timezone"
        with pytest.raises(AttributeError, match=msg):
            ts.tz = tz

    def test_default_to_stdlib_utc(self):
        assert Timestamp.utcnow().tz is timezone.utc
        assert Timestamp.now("UTC").tz is timezone.utc
        assert Timestamp("2016-01-01", tz="UTC").tz is timezone.utc

    def test_tz(self):
        tstr = "2014-02-01 09:00"
        ts = Timestamp(tstr)
        local = ts.tz_localize("Asia/Tokyo")
        assert local.hour == 9
        assert local == Timestamp(tstr, tz="Asia/Tokyo")
        conv = local.tz_convert("US/Eastern")
        assert conv == Timestamp("2014-01-31 19:00", tz="US/Eastern")
        assert conv.hour == 19

        # preserves nanosecond
        ts = Timestamp(tstr) + offsets.Nano(5)
        local = ts.tz_localize("Asia/Tokyo")
        assert local.hour == 9
        assert local.nanosecond == 5
        conv = local.tz_convert("US/Eastern")
        assert conv.nanosecond == 5
        assert conv.hour == 19

    def test_utc_z_designator(self):
        assert get_timezone(Timestamp("2014-11-02 01:00Z").tzinfo) is timezone.utc

    def test_asm8(self):
        ns = [Timestamp.min._value, Timestamp.max._value, 1000]

        for n in ns:
            assert (
                Timestamp(n).asm8.view("i8") == np.datetime64(n, "ns").view("i8") == n
            )

        assert Timestamp("nat").asm8.view("i8") == np.datetime64("nat", "ns").view("i8")

    def test_class_ops(self):
        def compare(x, y):
            assert int((Timestamp(x)._value - Timestamp(y)._value) / 1e9) == 0

        compare(Timestamp.now(), datetime.now())
        compare(Timestamp.now("UTC"), datetime.now(pytz.timezone("UTC")))
        compare(Timestamp.now("UTC"), datetime.now(tzutc()))
        compare(Timestamp.utcnow(), datetime.now(timezone.utc))
        compare(Timestamp.today(), datetime.today())
        current_time = calendar.timegm(datetime.now().utctimetuple())

        ts_utc = Timestamp.utcfromtimestamp(current_time)
        assert ts_utc.timestamp() == current_time
        compare(
            Timestamp.fromtimestamp(current_time), datetime.fromtimestamp(current_time)
        )
        compare(
            # Support tz kwarg in Timestamp.fromtimestamp
            Timestamp.fromtimestamp(current_time, "UTC"),
            datetime.fromtimestamp(current_time, utc),
        )
        compare(
            # Support tz kwarg in Timestamp.fromtimestamp
            Timestamp.fromtimestamp(current_time, tz="UTC"),
            datetime.fromtimestamp(current_time, utc),
        )

        date_component = datetime.now(timezone.utc)
        time_component = (date_component + timedelta(minutes=10)).time()
        compare(
            Timestamp.combine(date_component, time_component),
            datetime.combine(date_component, time_component),
        )

    def test_basics_nanos(self):
        val = np.int64(946_684_800_000_000_000).view("M8[ns]")
        stamp = Timestamp(val.view("i8") + 500)
        assert stamp.year == 2000
        assert stamp.month == 1
        assert stamp.microsecond == 0
        assert stamp.nanosecond == 500

        # GH 14415
        val = np.iinfo(np.int64).min + 80_000_000_000_000
        stamp = Timestamp(val)
        assert stamp.year == 1677
        assert stamp.month == 9
        assert stamp.day == 21
        assert stamp.microsecond == 145224
        assert stamp.nanosecond == 192

    def test_roundtrip(self):
        # test value to string and back conversions
        # further test accessors
        base = Timestamp("20140101 00:00:00").as_unit("ns")

        result = Timestamp(base._value + Timedelta("5ms")._value)
        assert result == Timestamp(f"{base}.005000")
        assert result.microsecond == 5000

        result = Timestamp(base._value + Timedelta("5us")._value)
        assert result == Timestamp(f"{base}.000005")
        assert result.microsecond == 5

        result = Timestamp(base._value + Timedelta("5ns")._value)
        assert result == Timestamp(f"{base}.000000005")
        assert result.nanosecond == 5
        assert result.microsecond == 0

        result = Timestamp(base._value + Timedelta("6ms 5us")._value)
        assert result == Timestamp(f"{base}.006005")
        assert result.microsecond == 5 + 6 * 1000

        result = Timestamp(base._value + Timedelta("200ms 5us")._value)
        assert result == Timestamp(f"{base}.200005")
        assert result.microsecond == 5 + 200 * 1000

    def test_hash_equivalent(self):
        d = {datetime(2011, 1, 1): 5}
        stamp = Timestamp(datetime(2011, 1, 1))
        assert d[stamp] == 5

    @pytest.mark.parametrize(
        "timezone, year, month, day, hour",
        [["America/Chicago", 2013, 11, 3, 1], ["America/Santiago", 2021, 4, 3, 23]],
    )
    def test_hash_timestamp_with_fold(self, timezone, year, month, day, hour):
        # see gh-33931
        test_timezone = gettz(timezone)
        transition_1 = Timestamp(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=0,
            fold=0,
            tzinfo=test_timezone,
        )
        transition_2 = Timestamp(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=0,
            fold=1,
            tzinfo=test_timezone,
        )
        assert hash(transition_1) == hash(transition_2)


class TestTimestampNsOperations:
    def test_nanosecond_string_parsing(self):
        ts = Timestamp("2013-05-01 07:15:45.123456789")
        # GH 7878
        expected_repr = "2013-05-01 07:15:45.123456789"
        expected_value = 1_367_392_545_123_456_789
        assert ts._value == expected_value
        assert expected_repr in repr(ts)

        ts = Timestamp("2013-05-01 07:15:45.123456789+09:00", tz="Asia/Tokyo")
        assert ts._value == expected_value - 9 * 3600 * 1_000_000_000
        assert expected_repr in repr(ts)

        ts = Timestamp("2013-05-01 07:15:45.123456789", tz="UTC")
        assert ts._value == expected_value
        assert expected_repr in repr(ts)

        ts = Timestamp("2013-05-01 07:15:45.123456789", tz="US/Eastern")
        assert ts._value == expected_value + 4 * 3600 * 1_000_000_000
        assert expected_repr in repr(ts)

        # GH 10041
        ts = Timestamp("20130501T071545.123456789")
        assert ts._value == expected_value
        assert expected_repr in repr(ts)

    def test_nanosecond_timestamp(self):
        # GH 7610
        expected = 1_293_840_000_000_000_005
        t = Timestamp("2011-01-01") + offsets.Nano(5)
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000005')"
        assert t._value == expected
        assert t.nanosecond == 5

        t = Timestamp(t)
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000005')"
        assert t._value == expected
        assert t.nanosecond == 5

        t = Timestamp("2011-01-01 00:00:00.000000005")
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000005')"
        assert t._value == expected
        assert t.nanosecond == 5

        expected = 1_293_840_000_000_000_010
        t = t + offsets.Nano(5)
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000010')"
        assert t._value == expected
        assert t.nanosecond == 10

        t = Timestamp(t)
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000010')"
        assert t._value == expected
        assert t.nanosecond == 10

        t = Timestamp("2011-01-01 00:00:00.000000010")
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000010')"
        assert t._value == expected
        assert t.nanosecond == 10


class TestTimestampConversion:
    def test_conversion(self):
        # GH#9255
        ts = Timestamp("2000-01-01").as_unit("ns")

        result = ts.to_pydatetime()
        expected = datetime(2000, 1, 1)
        assert result == expected
        assert type(result) == type(expected)

        result = ts.to_datetime64()
        expected = np.datetime64(ts._value, "ns")
        assert result == expected
        assert type(result) == type(expected)
        assert result.dtype == expected.dtype

    def test_to_period_tz_warning(self):
        # GH#21333 make sure a warning is issued when timezone
        # info is lost
        ts = Timestamp("2009-04-15 16:17:18", tz="US/Eastern")
        with tm.assert_produces_warning(UserWarning):
            # warning that timezone info will be lost
            ts.to_period("D")

    def test_to_numpy_alias(self):
        # GH 24653: alias .to_numpy() for scalars
        ts = Timestamp(datetime.now())
        assert ts.to_datetime64() == ts.to_numpy()

        # GH#44460
        msg = "dtype and copy arguments are ignored"
        with pytest.raises(ValueError, match=msg):
            ts.to_numpy("M8[s]")
        with pytest.raises(ValueError, match=msg):
            ts.to_numpy(copy=True)


class TestNonNano:
    @pytest.fixture(params=["s", "ms", "us"])
    def reso(self, request):
        return request.param

    @pytest.fixture
    def dt64(self, reso):
        # cases that are in-bounds for nanosecond, so we can compare against
        #  the existing implementation.
        return np.datetime64("2016-01-01", reso)

    @pytest.fixture
    def ts(self, dt64):
        return Timestamp._from_dt64(dt64)

    @pytest.fixture
    def ts_tz(self, ts, tz_aware_fixture):
        tz = maybe_get_tz(tz_aware_fixture)
        return Timestamp._from_value_and_reso(ts._value, ts._creso, tz)

    def test_non_nano_construction(self, dt64, ts, reso):
        assert ts._value == dt64.view("i8")

        if reso == "s":
            assert ts._creso == NpyDatetimeUnit.NPY_FR_s.value
        elif reso == "ms":
            assert ts._creso == NpyDatetimeUnit.NPY_FR_ms.value
        elif reso == "us":
            assert ts._creso == NpyDatetimeUnit.NPY_FR_us.value

    def test_non_nano_fields(self, dt64, ts):
        alt = Timestamp(dt64)

        assert ts.year == alt.year
        assert ts.month == alt.month
        assert ts.day == alt.day
        assert ts.hour == ts.minute == ts.second == ts.microsecond == 0
        assert ts.nanosecond == 0

        assert ts.to_julian_date() == alt.to_julian_date()
        assert ts.weekday() == alt.weekday()
        assert ts.isoweekday() == alt.isoweekday()

    def test_start_end_fields(self, ts):
        assert ts.is_year_start
        assert ts.is_quarter_start
        assert ts.is_month_start
        assert not ts.is_year_end
        assert not ts.is_month_end
        assert not ts.is_month_end

        # 2016-01-01 is a Friday, so is year/quarter/month start with this freq
        assert ts.is_year_start
        assert ts.is_quarter_start
        assert ts.is_month_start
        assert not ts.is_year_end
        assert not ts.is_month_end
        assert not ts.is_month_end

    def test_day_name(self, dt64, ts):
        alt = Timestamp(dt64)
        assert ts.day_name() == alt.day_name()

    def test_month_name(self, dt64, ts):
        alt = Timestamp(dt64)
        assert ts.month_name() == alt.month_name()

    def test_tz_convert(self, ts):
        ts = Timestamp._from_value_and_reso(ts._value, ts._creso, utc)

        tz = pytz.timezone("US/Pacific")
        result = ts.tz_convert(tz)

        assert isinstance(result, Timestamp)
        assert result._creso == ts._creso
        assert tz_compare(result.tz, tz)

    def test_repr(self, dt64, ts):
        alt = Timestamp(dt64)

        assert str(ts) == str(alt)
        assert repr(ts) == repr(alt)

    def test_comparison(self, dt64, ts):
        alt = Timestamp(dt64)

        assert ts == dt64
        assert dt64 == ts
        assert ts == alt
        assert alt == ts

        assert not ts != dt64
        assert not dt64 != ts
        assert not ts != alt
        assert not alt != ts

        assert not ts < dt64
        assert not dt64 < ts
        assert not ts < alt
        assert not alt < ts

        assert not ts > dt64
        assert not dt64 > ts
        assert not ts > alt
        assert not alt > ts

        assert ts >= dt64
        assert dt64 >= ts
        assert ts >= alt
        assert alt >= ts

        assert ts <= dt64
        assert dt64 <= ts
        assert ts <= alt
        assert alt <= ts

    def test_cmp_cross_reso(self):
        # numpy gets this wrong because of silent overflow
        dt64 = np.datetime64(9223372800, "s")  # won't fit in M8[ns]
        ts = Timestamp._from_dt64(dt64)

        # subtracting 3600*24 gives a datetime64 that _can_ fit inside the
        #  nanosecond implementation bounds.
        other = Timestamp(dt64 - 3600 * 24).as_unit("ns")
        assert other < ts
        assert other.asm8 > ts.asm8  # <- numpy gets this wrong
        assert ts > other
        assert ts.asm8 < other.asm8  # <- numpy gets this wrong
        assert not other == ts
        assert ts != other

    @pytest.mark.xfail(reason="Dispatches to np.datetime64 which is wrong")
    def test_cmp_cross_reso_reversed_dt64(self):
        dt64 = np.datetime64(106752, "D")  # won't fit in M8[ns]
        ts = Timestamp._from_dt64(dt64)
        other = Timestamp(dt64 - 1)

        assert other.asm8 < ts

    def test_pickle(self, ts, tz_aware_fixture):
        tz = tz_aware_fixture
        tz = maybe_get_tz(tz)
        ts = Timestamp._from_value_and_reso(ts._value, ts._creso, tz)
        rt = tm.round_trip_pickle(ts)
        assert rt._creso == ts._creso
        assert rt == ts

    def test_normalize(self, dt64, ts):
        alt = Timestamp(dt64)
        result = ts.normalize()
        assert result._creso == ts._creso
        assert result == alt.normalize()

    def test_asm8(self, dt64, ts):
        rt = ts.asm8
        assert rt == dt64
        assert rt.dtype == dt64.dtype

    def test_to_numpy(self, dt64, ts):
        res = ts.to_numpy()
        assert res == dt64
        assert res.dtype == dt64.dtype

    def test_to_datetime64(self, dt64, ts):
        res = ts.to_datetime64()
        assert res == dt64
        assert res.dtype == dt64.dtype

    def test_timestamp(self, dt64, ts):
        alt = Timestamp(dt64)
        assert ts.timestamp() == alt.timestamp()

    def test_to_period(self, dt64, ts):
        alt = Timestamp(dt64)
        assert ts.to_period("D") == alt.to_period("D")

    @pytest.mark.parametrize(
        "td", [timedelta(days=4), Timedelta(days=4), np.timedelta64(4, "D")]
    )
    def test_addsub_timedeltalike_non_nano(self, dt64, ts, td):
        exp_reso = max(ts._creso, Timedelta(td)._creso)

        result = ts - td
        expected = Timestamp(dt64) - td
        assert isinstance(result, Timestamp)
        assert result._creso == exp_reso
        assert result == expected

        result = ts + td
        expected = Timestamp(dt64) + td
        assert isinstance(result, Timestamp)
        assert result._creso == exp_reso
        assert result == expected

        result = td + ts
        expected = td + Timestamp(dt64)
        assert isinstance(result, Timestamp)
        assert result._creso == exp_reso
        assert result == expected

    def test_addsub_offset(self, ts_tz):
        # specifically non-Tick offset
        off = offsets.YearEnd(1)
        result = ts_tz + off

        assert isinstance(result, Timestamp)
        assert result._creso == ts_tz._creso
        if ts_tz.month == 12 and ts_tz.day == 31:
            assert result.year == ts_tz.year + 1
        else:
            assert result.year == ts_tz.year
        assert result.day == 31
        assert result.month == 12
        assert tz_compare(result.tz, ts_tz.tz)

        result = ts_tz - off

        assert isinstance(result, Timestamp)
        assert result._creso == ts_tz._creso
        assert result.year == ts_tz.year - 1
        assert result.day == 31
        assert result.month == 12
        assert tz_compare(result.tz, ts_tz.tz)

    def test_sub_datetimelike_mismatched_reso(self, ts_tz):
        # case with non-lossy rounding
        ts = ts_tz

        # choose a unit for `other` that doesn't match ts_tz's;
        #  this construction ensures we get cases with other._creso < ts._creso
        #  and cases with other._creso > ts._creso
        unit = {
            NpyDatetimeUnit.NPY_FR_us.value: "ms",
            NpyDatetimeUnit.NPY_FR_ms.value: "s",
            NpyDatetimeUnit.NPY_FR_s.value: "us",
        }[ts._creso]
        other = ts.as_unit(unit)
        assert other._creso != ts._creso

        result = ts - other
        assert isinstance(result, Timedelta)
        assert result._value == 0
        assert result._creso == max(ts._creso, other._creso)

        result = other - ts
        assert isinstance(result, Timedelta)
        assert result._value == 0
        assert result._creso == max(ts._creso, other._creso)

        if ts._creso < other._creso:
            # Case where rounding is lossy
            other2 = other + Timedelta._from_value_and_reso(1, other._creso)
            exp = ts.as_unit(other.unit) - other2

            res = ts - other2
            assert res == exp
            assert res._creso == max(ts._creso, other._creso)

            res = other2 - ts
            assert res == -exp
            assert res._creso == max(ts._creso, other._creso)
        else:
            ts2 = ts + Timedelta._from_value_and_reso(1, ts._creso)
            exp = ts2 - other.as_unit(ts2.unit)

            res = ts2 - other
            assert res == exp
            assert res._creso == max(ts._creso, other._creso)
            res = other - ts2
            assert res == -exp
            assert res._creso == max(ts._creso, other._creso)

    def test_sub_timedeltalike_mismatched_reso(self, ts_tz):
        # case with non-lossy rounding
        ts = ts_tz

        # choose a unit for `other` that doesn't match ts_tz's;
        #  this construction ensures we get cases with other._creso < ts._creso
        #  and cases with other._creso > ts._creso
        unit = {
            NpyDatetimeUnit.NPY_FR_us.value: "ms",
            NpyDatetimeUnit.NPY_FR_ms.value: "s",
            NpyDatetimeUnit.NPY_FR_s.value: "us",
        }[ts._creso]
        other = Timedelta(0).as_unit(unit)
        assert other._creso != ts._creso

        result = ts + other
        assert isinstance(result, Timestamp)
        assert result == ts
        assert result._creso == max(ts._creso, other._creso)

        result = other + ts
        assert isinstance(result, Timestamp)
        assert result == ts
        assert result._creso == max(ts._creso, other._creso)

        if ts._creso < other._creso:
            # Case where rounding is lossy
            other2 = other + Timedelta._from_value_and_reso(1, other._creso)
            exp = ts.as_unit(other.unit) + other2
            res = ts + other2
            assert res == exp
            assert res._creso == max(ts._creso, other._creso)
            res = other2 + ts
            assert res == exp
            assert res._creso == max(ts._creso, other._creso)
        else:
            ts2 = ts + Timedelta._from_value_and_reso(1, ts._creso)
            exp = ts2 + other.as_unit(ts2.unit)

            res = ts2 + other
            assert res == exp
            assert res._creso == max(ts._creso, other._creso)
            res = other + ts2
            assert res == exp
            assert res._creso == max(ts._creso, other._creso)

    def test_addition_doesnt_downcast_reso(self):
        # https://github.com/pandas-dev/pandas/pull/48748#pullrequestreview-1122635413
        ts = Timestamp(year=2022, month=1, day=1, microsecond=999999).as_unit("us")
        td = Timedelta(microseconds=1).as_unit("us")
        res = ts + td
        assert res._creso == ts._creso

    def test_sub_timedelta64_mismatched_reso(self, ts_tz):
        ts = ts_tz

        res = ts + np.timedelta64(1, "ns")
        exp = ts.as_unit("ns") + np.timedelta64(1, "ns")
        assert exp == res
        assert exp._creso == NpyDatetimeUnit.NPY_FR_ns.value

    def test_min(self, ts):
        assert ts.min <= ts
        assert ts.min._creso == ts._creso
        assert ts.min._value == NaT._value + 1

    def test_max(self, ts):
        assert ts.max >= ts
        assert ts.max._creso == ts._creso
        assert ts.max._value == np.iinfo(np.int64).max

    def test_resolution(self, ts):
        expected = Timedelta._from_value_and_reso(1, ts._creso)
        result = ts.resolution
        assert result == expected
        assert result._creso == expected._creso

    def test_out_of_ns_bounds(self):
        # https://github.com/pandas-dev/pandas/issues/51060
        result = Timestamp(-52700112000, unit="s")
        assert result == Timestamp("0300-01-01")
        assert result.to_numpy() == np.datetime64("0300-01-01T00:00:00", "s")


def test_timestamp_class_min_max_resolution():
    # when accessed on the class (as opposed to an instance), we default
    #  to nanoseconds
    assert Timestamp.min == Timestamp(NaT._value + 1)
    assert Timestamp.min._creso == NpyDatetimeUnit.NPY_FR_ns.value

    assert Timestamp.max == Timestamp(np.iinfo(np.int64).max)
    assert Timestamp.max._creso == NpyDatetimeUnit.NPY_FR_ns.value

    assert Timestamp.resolution == Timedelta(1)
    assert Timestamp.resolution._creso == NpyDatetimeUnit.NPY_FR_ns.value


def test_delimited_date():
    # https://github.com/pandas-dev/pandas/issues/50231
    with tm.assert_produces_warning(None):
        result = Timestamp("13-01-2000")
    expected = Timestamp(2000, 1, 13)
    assert result == expected


def test_utctimetuple():
    # GH 32174
    ts = Timestamp("2000-01-01", tz="UTC")
    result = ts.utctimetuple()
    expected = time.struct_time((2000, 1, 1, 0, 0, 0, 5, 1, 0))
    assert result == expected


def test_negative_dates():
    # https://github.com/pandas-dev/pandas/issues/50787
    ts = Timestamp("-2000-01-01")
    msg = (
        " not yet supported on Timestamps which are outside the range of "
        "Python's standard library. For now, please call the components you need "
        r"\(such as `.year` and `.month`\) and construct your string from there.$"
    )
    func = "^strftime"
    with pytest.raises(NotImplementedError, match=func + msg):
        ts.strftime("%Y")

    msg = (
        " not yet supported on Timestamps which "
        "are outside the range of Python's standard library. "
    )
    func = "^date"
    with pytest.raises(NotImplementedError, match=func + msg):
        ts.date()
    func = "^isocalendar"
    with pytest.raises(NotImplementedError, match=func + msg):
        ts.isocalendar()
    func = "^timetuple"
    with pytest.raises(NotImplementedError, match=func + msg):
        ts.timetuple()
    func = "^toordinal"
    with pytest.raises(NotImplementedError, match=func + msg):
        ts.toordinal()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\exceptions.py ===
"""Exceptions defined by Beautiful Soup itself."""

from typing import Union


class StopParsing(Exception):
    """Exception raised by a TreeBuilder if it's unable to continue parsing."""


class FeatureNotFound(ValueError):
    """Exception raised by the BeautifulSoup constructor if no parser with the
    requested features is found.
    """


class ParserRejectedMarkup(Exception):
    """An Exception to be raised when the underlying parser simply
    refuses to parse the given markup.
    """

    def __init__(self, message_or_exception: Union[str, Exception]):
        """Explain why the parser rejected the given markup, either
        with a textual explanation or another exception.
        """
        if isinstance(message_or_exception, Exception):
            e = message_or_exception
            message_or_exception = "%s: %s" % (e.__class__.__name__, str(e))
        super(ParserRejectedMarkup, self).__init__(message_or_exception)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chartsheet\properties.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors import (
    Bool,
    String,
    Typed
)
from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.styles import Color


class ChartsheetProperties(Serialisable):
    tagname = "sheetPr"

    published = Bool(allow_none=True)
    codeName = String(allow_none=True)
    tabColor = Typed(expected_type=Color, allow_none=True)

    __elements__ = ('tabColor',)

    def __init__(self,
                 published=None,
                 codeName=None,
                 tabColor=None,
                 ):
        self.published = published
        self.codeName = codeName
        self.tabColor = tabColor

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\strings\__init__.py ===
"""
Implementation of pandas.Series.str and its interface.

* strings.accessor.StringMethods : Accessor for Series.str
* strings.base.BaseStringArrayMethods: Mixin ABC for EAs to implement str methods

Most methods on the StringMethods accessor follow the pattern:

    1. extract the array from the series (or index)
    2. Call that array's implementation of the string method
    3. Wrap the result (in a Series, index, or DataFrame)

Pandas extension arrays implementing string methods should inherit from
pandas.core.strings.base.BaseStringArrayMethods. This is an ABC defining
the various string methods. To avoid namespace clashes and pollution,
these are prefixed with `_str_`. So ``Series.str.upper()`` calls
``Series.array._str_upper()``. The interface isn't currently public
to other string extension arrays.
"""
# Pandas current implementation is in ObjectStringArrayMixin. This is designed
# to work on object-dtype ndarrays.
#
# BaseStringArrayMethods
#  - ObjectStringArrayMixin
#     - StringArray
#     - NumpyExtensionArray
#     - Categorical
#     - ArrowStringArray

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\models\index.py ===
import urllib.parse


class PackageIndex:
    """Represents a Package Index and provides easier access to endpoints"""

    __slots__ = ["url", "netloc", "simple_url", "pypi_url", "file_storage_domain"]

    def __init__(self, url: str, file_storage_domain: str) -> None:
        super().__init__()
        self.url = url
        self.netloc = urllib.parse.urlsplit(url).netloc
        self.simple_url = self._url_for_path("simple")
        self.pypi_url = self._url_for_path("pypi")

        # This is part of a temporary hack used to block installs of PyPI
        # packages which depend on external urls only necessary until PyPI can
        # block such packages themselves
        self.file_storage_domain = file_storage_domain

    def _url_for_path(self, path: str) -> str:
        return urllib.parse.urljoin(self.url, path)


PyPI = PackageIndex("https://pypi.org/", file_storage_domain="files.pythonhosted.org")
TestPyPI = PackageIndex(
    "https://test.pypi.org/", file_storage_domain="test-files.pythonhosted.org"
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\fixtures\__init__.py ===
# testing/fixtures/__init__.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors
from .base import FutureEngineMixin as FutureEngineMixin
from .base import TestBase as TestBase
from .mypy import MypyTest as MypyTest
from .orm import after_test as after_test
from .orm import close_all_sessions as close_all_sessions
from .orm import DeclarativeMappedTest as DeclarativeMappedTest
from .orm import fixture_session as fixture_session
from .orm import MappedTest as MappedTest
from .orm import ORMTest as ORMTest
from .orm import RemoveORMEventsGlobally as RemoveORMEventsGlobally
from .orm import (
    stop_test_class_inside_fixtures as stop_test_class_inside_fixtures,
)
from .sql import CacheKeyFixture as CacheKeyFixture
from .sql import (
    ComputedReflectionFixtureTest as ComputedReflectionFixtureTest,
)
from .sql import insertmanyvalues_fixture as insertmanyvalues_fixture
from .sql import NoCache as NoCache
from .sql import RemovesEvents as RemovesEvents
from .sql import TablesTest as TablesTest

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_astype.py ===
import re

import numpy as np
import pytest

from pandas._config import using_string_dtype

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    Categorical,
    CategoricalDtype,
    DataFrame,
    DatetimeTZDtype,
    Index,
    Interval,
    IntervalDtype,
    NaT,
    Series,
    Timedelta,
    Timestamp,
    concat,
    date_range,
    option_context,
)
import pandas._testing as tm


def _check_cast(df, v):
    """
    Check if all dtypes of df are equal to v
    """
    assert all(s.dtype.name == v for _, s in df.items())


class TestAstype:
    def test_astype_float(self, float_frame):
        casted = float_frame.astype(int)
        expected = DataFrame(
            float_frame.values.astype(int),
            index=float_frame.index,
            columns=float_frame.columns,
        )
        tm.assert_frame_equal(casted, expected)

        casted = float_frame.astype(np.int32)
        expected = DataFrame(
            float_frame.values.astype(np.int32),
            index=float_frame.index,
            columns=float_frame.columns,
        )
        tm.assert_frame_equal(casted, expected)

        float_frame["foo"] = "5"
        casted = float_frame.astype(int)
        expected = DataFrame(
            float_frame.values.astype(int),
            index=float_frame.index,
            columns=float_frame.columns,
        )
        tm.assert_frame_equal(casted, expected)

    def test_astype_mixed_float(self, mixed_float_frame):
        # mixed casting
        casted = mixed_float_frame.reindex(columns=["A", "B"]).astype("float32")
        _check_cast(casted, "float32")

        casted = mixed_float_frame.reindex(columns=["A", "B"]).astype("float16")
        _check_cast(casted, "float16")

    def test_astype_mixed_type(self):
        # mixed casting
        df = DataFrame(
            {
                "a": 1.0,
                "b": 2,
                "c": "foo",
                "float32": np.array([1.0] * 10, dtype="float32"),
                "int32": np.array([1] * 10, dtype="int32"),
            },
            index=np.arange(10),
        )
        mn = df._get_numeric_data().copy()
        mn["little_float"] = np.array(12345.0, dtype="float16")
        mn["big_float"] = np.array(123456789101112.0, dtype="float64")

        casted = mn.astype("float64")
        _check_cast(casted, "float64")

        casted = mn.astype("int64")
        _check_cast(casted, "int64")

        casted = mn.reindex(columns=["little_float"]).astype("float16")
        _check_cast(casted, "float16")

        casted = mn.astype("float32")
        _check_cast(casted, "float32")

        casted = mn.astype("int32")
        _check_cast(casted, "int32")

        # to object
        casted = mn.astype("O")
        _check_cast(casted, "object")

    def test_astype_with_exclude_string(self, float_frame):
        df = float_frame.copy()
        expected = float_frame.astype(int)
        df["string"] = "foo"
        casted = df.astype(int, errors="ignore")

        expected["string"] = "foo"
        tm.assert_frame_equal(casted, expected)

        df = float_frame.copy()
        expected = float_frame.astype(np.int32)
        df["string"] = "foo"
        casted = df.astype(np.int32, errors="ignore")

        expected["string"] = "foo"
        tm.assert_frame_equal(casted, expected)

    def test_astype_with_view_float(self, float_frame):
        # this is the only real reason to do it this way
        tf = np.round(float_frame).astype(np.int32)
        tf.astype(np.float32, copy=False)

        # TODO(wesm): verification?
        tf = float_frame.astype(np.float64)
        tf.astype(np.int64, copy=False)

    def test_astype_with_view_mixed_float(self, mixed_float_frame):
        tf = mixed_float_frame.reindex(columns=["A", "B", "C"])

        tf.astype(np.int64)
        tf.astype(np.float32)

    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    @pytest.mark.parametrize("val", [np.nan, np.inf])
    def test_astype_cast_nan_inf_int(self, val, dtype):
        # see GH#14265
        #
        # Check NaN and inf --> raise error when converting to int.
        msg = "Cannot convert non-finite values \\(NA or inf\\) to integer"
        df = DataFrame([val])

        with pytest.raises(ValueError, match=msg):
            df.astype(dtype)

    def test_astype_str(self):
        # see GH#9757
        a = Series(date_range("2010-01-04", periods=5))
        b = Series(date_range("3/6/2012 00:00", periods=5, tz="US/Eastern"))
        c = Series([Timedelta(x, unit="d") for x in range(5)])
        d = Series(range(5))
        e = Series([0.0, 0.2, 0.4, 0.6, 0.8])

        df = DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})

        # Datetime-like
        result = df.astype(str)

        expected = DataFrame(
            {
                "a": list(map(str, (Timestamp(x)._date_repr for x in a._values))),
                "b": list(map(str, map(Timestamp, b._values))),
                "c": [Timedelta(x)._repr_base() for x in c._values],
                "d": list(map(str, d._values)),
                "e": list(map(str, e._values)),
            },
            dtype="str",
        )

        tm.assert_frame_equal(result, expected)

    def test_astype_str_float(self, using_infer_string):
        # see GH#11302
        result = DataFrame([np.nan]).astype(str)
        expected = DataFrame([np.nan if using_infer_string else "nan"], dtype="str")

        tm.assert_frame_equal(result, expected)
        result = DataFrame([1.12345678901234567890]).astype(str)

        val = "1.1234567890123457"
        expected = DataFrame([val], dtype="str")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dtype_class", [dict, Series])
    def test_astype_dict_like(self, dtype_class):
        # GH7271 & GH16717
        a = Series(date_range("2010-01-04", periods=5))
        b = Series(range(5))
        c = Series([0.0, 0.2, 0.4, 0.6, 0.8])
        d = Series(["1.0", "2", "3.14", "4", "5.4"])
        df = DataFrame({"a": a, "b": b, "c": c, "d": d})
        original = df.copy(deep=True)

        # change type of a subset of columns
        dt1 = dtype_class({"b": "str", "d": "float32"})
        result = df.astype(dt1)
        expected = DataFrame(
            {
                "a": a,
                "b": Series(["0", "1", "2", "3", "4"], dtype="str"),
                "c": c,
                "d": Series([1.0, 2.0, 3.14, 4.0, 5.4], dtype="float32"),
            }
        )
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(df, original)

        dt2 = dtype_class({"b": np.float32, "c": "float32", "d": np.float64})
        result = df.astype(dt2)
        expected = DataFrame(
            {
                "a": a,
                "b": Series([0.0, 1.0, 2.0, 3.0, 4.0], dtype="float32"),
                "c": Series([0.0, 0.2, 0.4, 0.6, 0.8], dtype="float32"),
                "d": Series([1.0, 2.0, 3.14, 4.0, 5.4], dtype="float64"),
            }
        )
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(df, original)

        # change all columns
        dt3 = dtype_class({"a": str, "b": str, "c": str, "d": str})
        tm.assert_frame_equal(df.astype(dt3), df.astype(str))
        tm.assert_frame_equal(df, original)

        # error should be raised when using something other than column labels
        # in the keys of the dtype dict
        dt4 = dtype_class({"b": str, 2: str})
        dt5 = dtype_class({"e": str})
        msg_frame = (
            "Only a column name can be used for the key in a dtype mappings argument. "
            "'{}' not found in columns."
        )
        with pytest.raises(KeyError, match=msg_frame.format(2)):
            df.astype(dt4)
        with pytest.raises(KeyError, match=msg_frame.format("e")):
            df.astype(dt5)
        tm.assert_frame_equal(df, original)

        # if the dtypes provided are the same as the original dtypes, the
        # resulting DataFrame should be the same as the original DataFrame
        dt6 = dtype_class({col: df[col].dtype for col in df.columns})
        equiv = df.astype(dt6)
        tm.assert_frame_equal(df, equiv)
        tm.assert_frame_equal(df, original)

        # GH#16717
        # if dtypes provided is empty, the resulting DataFrame
        # should be the same as the original DataFrame
        dt7 = dtype_class({}) if dtype_class is dict else dtype_class({}, dtype=object)
        equiv = df.astype(dt7)
        tm.assert_frame_equal(df, equiv)
        tm.assert_frame_equal(df, original)

    def test_astype_duplicate_col(self):
        a1 = Series([1, 2, 3, 4, 5], name="a")
        b = Series([0.1, 0.2, 0.4, 0.6, 0.8], name="b")
        a2 = Series([0, 1, 2, 3, 4], name="a")
        df = concat([a1, b, a2], axis=1)

        result = df.astype("str")
        a1_str = Series(["1", "2", "3", "4", "5"], dtype="str", name="a")
        b_str = Series(["0.1", "0.2", "0.4", "0.6", "0.8"], dtype="str", name="b")
        a2_str = Series(["0", "1", "2", "3", "4"], dtype="str", name="a")
        expected = concat([a1_str, b_str, a2_str], axis=1)
        tm.assert_frame_equal(result, expected)

        result = df.astype({"a": "str"})
        expected = concat([a1_str, b, a2_str], axis=1)
        tm.assert_frame_equal(result, expected)

    def test_astype_duplicate_col_series_arg(self):
        # GH#44417
        vals = np.random.default_rng(2).standard_normal((3, 4))
        df = DataFrame(vals, columns=["A", "B", "C", "A"])
        dtypes = df.dtypes
        dtypes.iloc[0] = str
        dtypes.iloc[2] = "Float64"

        result = df.astype(dtypes)
        expected = DataFrame(
            {
                0: Series(vals[:, 0].astype(str), dtype="str"),
                1: vals[:, 1],
                2: pd.array(vals[:, 2], dtype="Float64"),
                3: vals[:, 3],
            }
        )
        expected.columns = df.columns
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            "category",
            CategoricalDtype(),
            CategoricalDtype(ordered=True),
            CategoricalDtype(ordered=False),
            CategoricalDtype(categories=list("abcdef")),
            CategoricalDtype(categories=list("edba"), ordered=False),
            CategoricalDtype(categories=list("edcb"), ordered=True),
        ],
        ids=repr,
    )
    def test_astype_categorical(self, dtype):
        # GH#18099
        d = {"A": list("abbc"), "B": list("bccd"), "C": list("cdde")}
        df = DataFrame(d)
        result = df.astype(dtype)
        expected = DataFrame({k: Categorical(v, dtype=dtype) for k, v in d.items()})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("cls", [CategoricalDtype, DatetimeTZDtype, IntervalDtype])
    def test_astype_categoricaldtype_class_raises(self, cls):
        df = DataFrame({"A": ["a", "a", "b", "c"]})
        xpr = f"Expected an instance of {cls.__name__}"
        with pytest.raises(TypeError, match=xpr):
            df.astype({"A": cls})

        with pytest.raises(TypeError, match=xpr):
            df["A"].astype(cls)

    @pytest.mark.parametrize("dtype", ["Int64", "Int32", "Int16"])
    def test_astype_extension_dtypes(self, dtype):
        # GH#22578
        df = DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], columns=["a", "b"])

        expected1 = DataFrame(
            {
                "a": pd.array([1, 3, 5], dtype=dtype),
                "b": pd.array([2, 4, 6], dtype=dtype),
            }
        )
        tm.assert_frame_equal(df.astype(dtype), expected1)
        tm.assert_frame_equal(df.astype("int64").astype(dtype), expected1)
        tm.assert_frame_equal(df.astype(dtype).astype("float64"), df)

        df = DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], columns=["a", "b"])
        df["b"] = df["b"].astype(dtype)
        expected2 = DataFrame(
            {"a": [1.0, 3.0, 5.0], "b": pd.array([2, 4, 6], dtype=dtype)}
        )
        tm.assert_frame_equal(df, expected2)

        tm.assert_frame_equal(df.astype(dtype), expected1)
        tm.assert_frame_equal(df.astype("int64").astype(dtype), expected1)

    @pytest.mark.parametrize("dtype", ["Int64", "Int32", "Int16"])
    def test_astype_extension_dtypes_1d(self, dtype):
        # GH#22578
        df = DataFrame({"a": [1.0, 2.0, 3.0]})

        expected1 = DataFrame({"a": pd.array([1, 2, 3], dtype=dtype)})
        tm.assert_frame_equal(df.astype(dtype), expected1)
        tm.assert_frame_equal(df.astype("int64").astype(dtype), expected1)

        df = DataFrame({"a": [1.0, 2.0, 3.0]})
        df["a"] = df["a"].astype(dtype)
        expected2 = DataFrame({"a": pd.array([1, 2, 3], dtype=dtype)})
        tm.assert_frame_equal(df, expected2)

        tm.assert_frame_equal(df.astype(dtype), expected1)
        tm.assert_frame_equal(df.astype("int64").astype(dtype), expected1)

    @pytest.mark.parametrize("dtype", ["category", "Int64"])
    def test_astype_extension_dtypes_duplicate_col(self, dtype):
        # GH#24704
        a1 = Series([0, np.nan, 4], name="a")
        a2 = Series([np.nan, 3, 5], name="a")
        df = concat([a1, a2], axis=1)

        result = df.astype(dtype)
        expected = concat([a1.astype(dtype), a2.astype(dtype)], axis=1)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype", [{100: "float64", 200: "uint64"}, "category", "float64"]
    )
    def test_astype_column_metadata(self, dtype):
        # GH#19920
        columns = Index([100, 200, 300], dtype=np.uint64, name="foo")
        df = DataFrame(np.arange(15).reshape(5, 3), columns=columns)
        df = df.astype(dtype)
        tm.assert_index_equal(df.columns, columns)

    @pytest.mark.parametrize("unit", ["Y", "M", "W", "D", "h", "m"])
    def test_astype_from_object_to_datetime_unit(self, unit):
        vals = [
            ["2015-01-01", "2015-01-02", "2015-01-03"],
            ["2017-01-01", "2017-01-02", "2017-02-03"],
        ]
        df = DataFrame(vals, dtype=object)
        msg = (
            rf"Unexpected value for 'dtype': 'datetime64\[{unit}\]'. "
            r"Must be 'datetime64\[s\]', 'datetime64\[ms\]', 'datetime64\[us\]', "
            r"'datetime64\[ns\]' or DatetimeTZDtype"
        )
        with pytest.raises(ValueError, match=msg):
            df.astype(f"M8[{unit}]")

    @pytest.mark.parametrize("unit", ["Y", "M", "W", "D", "h", "m"])
    def test_astype_from_object_to_timedelta_unit(self, unit):
        vals = [
            ["1 Day", "2 Days", "3 Days"],
            ["4 Days", "5 Days", "6 Days"],
        ]
        df = DataFrame(vals, dtype=object)
        msg = (
            r"Cannot convert from timedelta64\[ns\] to timedelta64\[.*\]. "
            "Supported resolutions are 's', 'ms', 'us', 'ns'"
        )
        with pytest.raises(ValueError, match=msg):
            # TODO: this is ValueError while for DatetimeArray it is TypeError;
            #  get these consistent
            df.astype(f"m8[{unit}]")

    @pytest.mark.parametrize("dtype", ["M8", "m8"])
    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s", "h", "m", "D"])
    def test_astype_from_datetimelike_to_object(self, dtype, unit):
        # tests astype to object dtype
        # GH#19223 / GH#12425
        dtype = f"{dtype}[{unit}]"
        arr = np.array([[1, 2, 3]], dtype=dtype)
        df = DataFrame(arr)
        result = df.astype(object)
        assert (result.dtypes == object).all()

        if dtype.startswith("M8"):
            assert result.iloc[0, 0] == Timestamp(1, unit=unit)
        else:
            assert result.iloc[0, 0] == Timedelta(1, unit=unit)

    @pytest.mark.parametrize("arr_dtype", [np.int64, np.float64])
    @pytest.mark.parametrize("dtype", ["M8", "m8"])
    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s", "h", "m", "D"])
    def test_astype_to_datetimelike_unit(self, arr_dtype, dtype, unit):
        # tests all units from numeric origination
        # GH#19223 / GH#12425
        dtype = f"{dtype}[{unit}]"
        arr = np.array([[1, 2, 3]], dtype=arr_dtype)
        df = DataFrame(arr)
        result = df.astype(dtype)
        expected = DataFrame(arr.astype(dtype))

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s", "h", "m", "D"])
    def test_astype_to_datetime_unit(self, unit):
        # tests all units from datetime origination
        # GH#19223
        dtype = f"M8[{unit}]"
        arr = np.array([[1, 2, 3]], dtype=dtype)
        df = DataFrame(arr)
        ser = df.iloc[:, 0]
        idx = Index(ser)
        dta = ser._values

        if unit in ["ns", "us", "ms", "s"]:
            # GH#48928
            result = df.astype(dtype)
        else:
            # we use the nearest supported dtype (i.e. M8[s])
            msg = rf"Cannot cast DatetimeArray to dtype datetime64\[{unit}\]"
            with pytest.raises(TypeError, match=msg):
                df.astype(dtype)

            with pytest.raises(TypeError, match=msg):
                ser.astype(dtype)

            with pytest.raises(TypeError, match=msg.replace("Array", "Index")):
                idx.astype(dtype)

            with pytest.raises(TypeError, match=msg):
                dta.astype(dtype)

            return

        exp_df = DataFrame(arr.astype(dtype))
        assert (exp_df.dtypes == dtype).all()
        tm.assert_frame_equal(result, exp_df)

        res_ser = ser.astype(dtype)
        exp_ser = exp_df.iloc[:, 0]
        assert exp_ser.dtype == dtype
        tm.assert_series_equal(res_ser, exp_ser)

        exp_dta = exp_ser._values

        res_index = idx.astype(dtype)
        exp_index = Index(exp_ser)
        assert exp_index.dtype == dtype
        tm.assert_index_equal(res_index, exp_index)

        res_dta = dta.astype(dtype)
        assert exp_dta.dtype == dtype
        tm.assert_extension_array_equal(res_dta, exp_dta)

    @pytest.mark.parametrize("unit", ["ns"])
    def test_astype_to_timedelta_unit_ns(self, unit):
        # preserver the timedelta conversion
        # GH#19223
        dtype = f"m8[{unit}]"
        arr = np.array([[1, 2, 3]], dtype=dtype)
        df = DataFrame(arr)
        result = df.astype(dtype)
        expected = DataFrame(arr.astype(dtype))

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("unit", ["us", "ms", "s", "h", "m", "D"])
    def test_astype_to_timedelta_unit(self, unit):
        # coerce to float
        # GH#19223 until 2.0 used to coerce to float
        dtype = f"m8[{unit}]"
        arr = np.array([[1, 2, 3]], dtype=dtype)
        df = DataFrame(arr)
        ser = df.iloc[:, 0]
        tdi = Index(ser)
        tda = tdi._values

        if unit in ["us", "ms", "s"]:
            assert (df.dtypes == dtype).all()
            result = df.astype(dtype)
        else:
            # We get the nearest supported unit, i.e. "s"
            assert (df.dtypes == "m8[s]").all()

            msg = (
                rf"Cannot convert from timedelta64\[s\] to timedelta64\[{unit}\]. "
                "Supported resolutions are 's', 'ms', 'us', 'ns'"
            )
            with pytest.raises(ValueError, match=msg):
                df.astype(dtype)
            with pytest.raises(ValueError, match=msg):
                ser.astype(dtype)
            with pytest.raises(ValueError, match=msg):
                tdi.astype(dtype)
            with pytest.raises(ValueError, match=msg):
                tda.astype(dtype)

            return

        result = df.astype(dtype)
        # The conversion is a no-op, so we just get a copy
        expected = df
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s", "h", "m", "D"])
    def test_astype_to_incorrect_datetimelike(self, unit):
        # trying to astype a m to a M, or vice-versa
        # GH#19224
        dtype = f"M8[{unit}]"
        other = f"m8[{unit}]"

        df = DataFrame(np.array([[1, 2, 3]], dtype=dtype))
        msg = "|".join(
            [
                # BlockManager path
                rf"Cannot cast DatetimeArray to dtype timedelta64\[{unit}\]",
                # ArrayManager path
                "cannot astype a datetimelike from "
                rf"\[datetime64\[ns\]\] to \[timedelta64\[{unit}\]\]",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            df.astype(other)

        msg = "|".join(
            [
                # BlockManager path
                rf"Cannot cast TimedeltaArray to dtype datetime64\[{unit}\]",
                # ArrayManager path
                "cannot astype a timedelta from "
                rf"\[timedelta64\[ns\]\] to \[datetime64\[{unit}\]\]",
            ]
        )
        df = DataFrame(np.array([[1, 2, 3]], dtype=other))
        with pytest.raises(TypeError, match=msg):
            df.astype(dtype)

    def test_astype_arg_for_errors(self):
        # GH#14878

        df = DataFrame([1, 2, 3])

        msg = (
            "Expected value of kwarg 'errors' to be one of "
            "['raise', 'ignore']. Supplied value is 'True'"
        )
        with pytest.raises(ValueError, match=re.escape(msg)):
            df.astype(np.float64, errors=True)

        df.astype(np.int8, errors="ignore")

    def test_astype_invalid_conversion(self):
        # GH#47571
        df = DataFrame({"a": [1, 2, "text"], "b": [1, 2, 3]})

        msg = (
            "invalid literal for int() with base 10: 'text': "
            "Error while type casting for column 'a'"
        )

        with pytest.raises(ValueError, match=re.escape(msg)):
            df.astype({"a": int})

    def test_astype_arg_for_errors_dictlist(self):
        # GH#25905
        df = DataFrame(
            [
                {"a": "1", "b": "16.5%", "c": "test"},
                {"a": "2.2", "b": "15.3", "c": "another_test"},
            ]
        )
        expected = DataFrame(
            [
                {"a": 1.0, "b": "16.5%", "c": "test"},
                {"a": 2.2, "b": "15.3", "c": "another_test"},
            ]
        )
        expected["c"] = expected["c"].astype("object")
        type_dict = {"a": "float64", "b": "float64", "c": "object"}

        result = df.astype(dtype=type_dict, errors="ignore")

        tm.assert_frame_equal(result, expected)

    def test_astype_dt64tz(self, timezone_frame):
        # astype
        expected = np.array(
            [
                [
                    Timestamp("2013-01-01 00:00:00"),
                    Timestamp("2013-01-02 00:00:00"),
                    Timestamp("2013-01-03 00:00:00"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00-0500", tz="US/Eastern"),
                    NaT,
                    Timestamp("2013-01-03 00:00:00-0500", tz="US/Eastern"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00+0100", tz="CET"),
                    NaT,
                    Timestamp("2013-01-03 00:00:00+0100", tz="CET"),
                ],
            ],
            dtype=object,
        ).T
        expected = DataFrame(
            expected,
            index=timezone_frame.index,
            columns=timezone_frame.columns,
            dtype=object,
        )
        result = timezone_frame.astype(object)
        tm.assert_frame_equal(result, expected)

        msg = "Cannot use .astype to convert from timezone-aware dtype to timezone-"
        with pytest.raises(TypeError, match=msg):
            # dt64tz->dt64 deprecated
            timezone_frame.astype("datetime64[ns]")

    def test_astype_dt64tz_to_str(self, timezone_frame, using_infer_string):
        # str formatting
        result = timezone_frame.astype(str)
        na_value = np.nan if using_infer_string else "NaT"
        expected = DataFrame(
            [
                [
                    "2013-01-01",
                    "2013-01-01 00:00:00-05:00",
                    "2013-01-01 00:00:00+01:00",
                ],
                ["2013-01-02", na_value, na_value],
                [
                    "2013-01-03",
                    "2013-01-03 00:00:00-05:00",
                    "2013-01-03 00:00:00+01:00",
                ],
            ],
            columns=timezone_frame.columns,
            dtype="str",
        )
        tm.assert_frame_equal(result, expected)

        with option_context("display.max_columns", 20):
            result = str(timezone_frame)
            assert (
                "0 2013-01-01 2013-01-01 00:00:00-05:00 2013-01-01 00:00:00+01:00"
            ) in result
            assert (
                "1 2013-01-02                       NaT                       NaT"
            ) in result
            assert (
                "2 2013-01-03 2013-01-03 00:00:00-05:00 2013-01-03 00:00:00+01:00"
            ) in result

    def test_astype_empty_dtype_dict(self):
        # issue mentioned further down in the following issue's thread
        # https://github.com/pandas-dev/pandas/issues/33113
        df = DataFrame()
        result = df.astype({})
        tm.assert_frame_equal(result, df)
        assert result is not df

    @pytest.mark.parametrize(
        "data, dtype",
        [
            (["x", "y", "z"], "string[python]"),
            pytest.param(
                ["x", "y", "z"],
                "string[pyarrow]",
                marks=td.skip_if_no("pyarrow"),
            ),
            (["x", "y", "z"], "category"),
            (3 * [Timestamp("2020-01-01", tz="UTC")], None),
            (3 * [Interval(0, 1)], None),
        ],
    )
    @pytest.mark.parametrize("errors", ["raise", "ignore"])
    def test_astype_ignores_errors_for_extension_dtypes(self, data, dtype, errors):
        # https://github.com/pandas-dev/pandas/issues/35471
        df = DataFrame(Series(data, dtype=dtype))
        if errors == "ignore":
            expected = df
            result = df.astype(float, errors=errors)
            tm.assert_frame_equal(result, expected)
        else:
            msg = "(Cannot cast)|(could not convert)"
            with pytest.raises((ValueError, TypeError), match=msg):
                df.astype(float, errors=errors)

    def test_astype_tz_conversion(self):
        # GH 35973
        val = {"tz": date_range("2020-08-30", freq="d", periods=2, tz="Europe/London")}
        df = DataFrame(val)
        result = df.astype({"tz": "datetime64[ns, Europe/Berlin]"})

        expected = df
        expected["tz"] = expected["tz"].dt.tz_convert("Europe/Berlin")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("tz", ["UTC", "Europe/Berlin"])
    def test_astype_tz_object_conversion(self, tz):
        # GH 35973
        val = {"tz": date_range("2020-08-30", freq="d", periods=2, tz="Europe/London")}
        expected = DataFrame(val)

        # convert expected to object dtype from other tz str (independently tested)
        result = expected.astype({"tz": f"datetime64[ns, {tz}]"})
        result = result.astype({"tz": "object"})

        # do real test: object dtype to a specified tz, different from construction tz.
        result = result.astype({"tz": "datetime64[ns, Europe/London]"})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.xfail(using_string_dtype(), reason="TODO(infer_string) GH#60639")
    def test_astype_dt64_to_string(
        self, frame_or_series, tz_naive_fixture, using_infer_string
    ):
        # GH#41409
        tz = tz_naive_fixture

        dti = date_range("2016-01-01", periods=3, tz=tz)
        dta = dti._data
        dta[0] = NaT

        obj = frame_or_series(dta)
        result = obj.astype("string")

        # Check that Series/DataFrame.astype matches DatetimeArray.astype
        expected = frame_or_series(dta.astype("string"))
        tm.assert_equal(result, expected)

        item = result.iloc[0]
        if frame_or_series is DataFrame:
            item = item.iloc[0]
        if using_infer_string:
            assert item is np.nan
        else:
            assert item is pd.NA

        # For non-NA values, we should match what we get for non-EA str
        alt = obj.astype(str)
        assert np.all(alt.iloc[1:] == result.iloc[1:])

    def test_astype_td64_to_string(self, frame_or_series):
        # GH#41409
        tdi = pd.timedelta_range("1 Day", periods=3)
        obj = frame_or_series(tdi)

        expected = frame_or_series(["1 days", "2 days", "3 days"], dtype="string")
        result = obj.astype("string")
        tm.assert_equal(result, expected)

    def test_astype_bytes(self):
        # GH#39474
        result = DataFrame(["foo", "bar", "baz"]).astype(bytes)
        assert result.dtypes[0] == np.dtype("S3")

    @pytest.mark.parametrize(
        "index_slice",
        [
            np.s_[:2, :2],
            np.s_[:1, :2],
            np.s_[:2, :1],
            np.s_[::2, ::2],
            np.s_[::1, ::2],
            np.s_[::2, ::1],
        ],
    )
    def test_astype_noncontiguous(self, index_slice):
        # GH#42396
        data = np.arange(16).reshape(4, 4)
        df = DataFrame(data)

        result = df.iloc[index_slice].astype("int16")
        expected = df.iloc[index_slice]
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_astype_retain_attrs(self, any_numpy_dtype):
        # GH#44414
        df = DataFrame({"a": [0, 1, 2], "b": [3, 4, 5]})
        df.attrs["Location"] = "Michigan"

        result = df.astype({"a": any_numpy_dtype}).attrs
        expected = df.attrs

        tm.assert_dict_equal(expected, result)


class TestAstypeCategorical:
    def test_astype_from_categorical3(self):
        df = DataFrame({"cats": [1, 2, 3, 4, 5, 6], "vals": [1, 2, 3, 4, 5, 6]})
        cats = Categorical([1, 2, 3, 4, 5, 6])
        exp_df = DataFrame({"cats": cats, "vals": [1, 2, 3, 4, 5, 6]})
        df["cats"] = df["cats"].astype("category")
        tm.assert_frame_equal(exp_df, df)

    def test_astype_from_categorical4(self):
        df = DataFrame(
            {"cats": ["a", "b", "b", "a", "a", "d"], "vals": [1, 2, 3, 4, 5, 6]}
        )
        cats = Categorical(["a", "b", "b", "a", "a", "d"])
        exp_df = DataFrame({"cats": cats, "vals": [1, 2, 3, 4, 5, 6]})
        df["cats"] = df["cats"].astype("category")
        tm.assert_frame_equal(exp_df, df)

    def test_categorical_astype_to_int(self, any_int_dtype):
        # GH#39402

        df = DataFrame(data={"col1": pd.array([2.0, 1.0, 3.0])})
        df.col1 = df.col1.astype("category")
        df.col1 = df.col1.astype(any_int_dtype)
        expected = DataFrame({"col1": pd.array([2, 1, 3], dtype=any_int_dtype)})
        tm.assert_frame_equal(df, expected)

    def test_astype_categorical_to_string_missing(self):
        # https://github.com/pandas-dev/pandas/issues/41797
        df = DataFrame(["a", "b", np.nan])
        expected = df.astype(str)
        cat = df.astype("category")
        result = cat.astype(str)
        tm.assert_frame_equal(result, expected)


class IntegerArrayNoCopy(pd.core.arrays.IntegerArray):
    # GH 42501

    def copy(self):
        assert False


class Int16DtypeNoCopy(pd.Int16Dtype):
    # GH 42501

    @classmethod
    def construct_array_type(cls):
        return IntegerArrayNoCopy


def test_frame_astype_no_copy():
    # GH 42501
    df = DataFrame({"a": [1, 4, None, 5], "b": [6, 7, 8, 9]}, dtype=object)
    result = df.astype({"a": Int16DtypeNoCopy()}, copy=False)

    assert result.a.dtype == pd.Int16Dtype()
    assert np.shares_memory(df.b.values, result.b.values)


@pytest.mark.parametrize("dtype", ["int64", "Int64"])
def test_astype_copies(dtype):
    # GH#50984
    pytest.importorskip("pyarrow")
    df = DataFrame({"a": [1, 2, 3]}, dtype=dtype)
    result = df.astype("int64[pyarrow]", copy=True)
    df.iloc[0, 0] = 100
    expected = DataFrame({"a": [1, 2, 3]}, dtype="int64[pyarrow]")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("val", [None, 1, 1.5, np.nan, NaT])
def test_astype_to_string_not_modifying_input(string_storage, val):
    # GH#51073
    df = DataFrame({"a": ["a", "b", val]})
    expected = df.copy()
    with option_context("mode.string_storage", string_storage):
        df.astype("string", copy=False)
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize("val", [None, 1, 1.5, np.nan, NaT])
def test_astype_to_string_dtype_not_modifying_input(any_string_dtype, val):
    # GH#51073 - variant of the above test with explicit dtype instances
    df = DataFrame({"a": ["a", "b", val]})
    expected = df.copy()
    df.astype(any_string_dtype)
    tm.assert_frame_equal(df, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\dtypes\test_missing.py ===
from contextlib import nullcontext
from datetime import datetime
from decimal import Decimal

import numpy as np
import pytest

from pandas._config import config as cf

from pandas._libs import missing as libmissing
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p25

from pandas.core.dtypes.common import (
    is_float,
    is_scalar,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    IntervalDtype,
    PeriodDtype,
)
from pandas.core.dtypes.missing import (
    array_equivalent,
    is_valid_na_for_dtype,
    isna,
    isnull,
    na_value_for_dtype,
    notna,
    notnull,
)

import pandas as pd
from pandas import (
    DatetimeIndex,
    Index,
    NaT,
    Series,
    TimedeltaIndex,
    date_range,
    period_range,
)
import pandas._testing as tm

fix_now = pd.Timestamp("2021-01-01")
fix_utcnow = pd.Timestamp("2021-01-01", tz="UTC")


@pytest.mark.parametrize("notna_f", [notna, notnull])
def test_notna_notnull(notna_f):
    assert notna_f(1.0)
    assert not notna_f(None)
    assert not notna_f(np.nan)

    msg = "use_inf_as_na option is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with cf.option_context("mode.use_inf_as_na", False):
            assert notna_f(np.inf)
            assert notna_f(-np.inf)

            arr = np.array([1.5, np.inf, 3.5, -np.inf])
            result = notna_f(arr)
            assert result.all()

    with tm.assert_produces_warning(FutureWarning, match=msg):
        with cf.option_context("mode.use_inf_as_na", True):
            assert not notna_f(np.inf)
            assert not notna_f(-np.inf)

            arr = np.array([1.5, np.inf, 3.5, -np.inf])
            result = notna_f(arr)
            assert result.sum() == 2


@pytest.mark.parametrize("null_func", [notna, notnull, isna, isnull])
@pytest.mark.parametrize(
    "ser",
    [
        Series(
            [str(i) for i in range(5)],
            index=Index([str(i) for i in range(5)], dtype=object),
            dtype=object,
        ),
        Series(range(5), date_range("2020-01-01", periods=5)),
        Series(range(5), period_range("2020-01-01", periods=5)),
    ],
)
def test_null_check_is_series(null_func, ser):
    msg = "use_inf_as_na option is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with cf.option_context("mode.use_inf_as_na", False):
            assert isinstance(null_func(ser), Series)


class TestIsNA:
    def test_0d_array(self):
        assert isna(np.array(np.nan))
        assert not isna(np.array(0.0))
        assert not isna(np.array(0))
        # test object dtype
        assert isna(np.array(np.nan, dtype=object))
        assert not isna(np.array(0.0, dtype=object))
        assert not isna(np.array(0, dtype=object))

    @pytest.mark.parametrize("shape", [(4, 0), (4,)])
    def test_empty_object(self, shape):
        arr = np.empty(shape=shape, dtype=object)
        result = isna(arr)
        expected = np.ones(shape=shape, dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("isna_f", [isna, isnull])
    def test_isna_isnull(self, isna_f):
        assert not isna_f(1.0)
        assert isna_f(None)
        assert isna_f(np.nan)
        assert float("nan")
        assert not isna_f(np.inf)
        assert not isna_f(-np.inf)

        # type
        assert not isna_f(type(Series(dtype=object)))
        assert not isna_f(type(Series(dtype=np.float64)))
        assert not isna_f(type(pd.DataFrame()))

    @pytest.mark.parametrize("isna_f", [isna, isnull])
    @pytest.mark.parametrize(
        "data",
        [
            np.arange(4, dtype=float),
            [0.0, 1.0, 0.0, 1.0],
            Series(list("abcd")),
            date_range("2020-01-01", periods=4),
        ],
    )
    @pytest.mark.parametrize(
        "index",
        [
            date_range("2020-01-01", periods=4),
            range(4),
            period_range("2020-01-01", periods=4),
        ],
    )
    def test_isna_isnull_frame(self, isna_f, data, index):
        # frame
        df = pd.DataFrame(data, index=index)
        result = isna_f(df)
        expected = df.apply(isna_f)
        tm.assert_frame_equal(result, expected)

    def test_isna_lists(self):
        result = isna([[False]])
        exp = np.array([[False]])
        tm.assert_numpy_array_equal(result, exp)

        result = isna([[1], [2]])
        exp = np.array([[False], [False]])
        tm.assert_numpy_array_equal(result, exp)

        # list of strings / unicode
        result = isna(["foo", "bar"])
        exp = np.array([False, False])
        tm.assert_numpy_array_equal(result, exp)

        result = isna(["foo", "bar"])
        exp = np.array([False, False])
        tm.assert_numpy_array_equal(result, exp)

        # GH20675
        result = isna([np.nan, "world"])
        exp = np.array([True, False])
        tm.assert_numpy_array_equal(result, exp)

    def test_isna_nat(self):
        result = isna([NaT])
        exp = np.array([True])
        tm.assert_numpy_array_equal(result, exp)

        result = isna(np.array([NaT], dtype=object))
        exp = np.array([True])
        tm.assert_numpy_array_equal(result, exp)

    def test_isna_numpy_nat(self):
        arr = np.array(
            [
                NaT,
                np.datetime64("NaT"),
                np.timedelta64("NaT"),
                np.datetime64("NaT", "s"),
            ]
        )
        result = isna(arr)
        expected = np.array([True] * 4)
        tm.assert_numpy_array_equal(result, expected)

    def test_isna_datetime(self):
        assert not isna(datetime.now())
        assert notna(datetime.now())

        idx = date_range("1/1/1990", periods=20)
        exp = np.ones(len(idx), dtype=bool)
        tm.assert_numpy_array_equal(notna(idx), exp)

        idx = np.asarray(idx)
        idx[0] = iNaT
        idx = DatetimeIndex(idx)
        mask = isna(idx)
        assert mask[0]
        exp = np.array([True] + [False] * (len(idx) - 1), dtype=bool)
        tm.assert_numpy_array_equal(mask, exp)

        # GH 9129
        pidx = idx.to_period(freq="M")
        mask = isna(pidx)
        assert mask[0]
        exp = np.array([True] + [False] * (len(idx) - 1), dtype=bool)
        tm.assert_numpy_array_equal(mask, exp)

        mask = isna(pidx[1:])
        exp = np.zeros(len(mask), dtype=bool)
        tm.assert_numpy_array_equal(mask, exp)

    def test_isna_old_datetimelike(self):
        # isna_old should work for dt64tz, td64, and period, not just tznaive
        dti = date_range("2016-01-01", periods=3)
        dta = dti._data
        dta[-1] = NaT
        expected = np.array([False, False, True], dtype=bool)

        objs = [dta, dta.tz_localize("US/Eastern"), dta - dta, dta.to_period("D")]

        for obj in objs:
            msg = "use_inf_as_na option is deprecated"
            with tm.assert_produces_warning(FutureWarning, match=msg):
                with cf.option_context("mode.use_inf_as_na", True):
                    result = isna(obj)

            tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "value, expected",
        [
            (np.complex128(np.nan), True),
            (np.float64(1), False),
            (np.array([1, 1 + 0j, np.nan, 3]), np.array([False, False, True, False])),
            (
                np.array([1, 1 + 0j, np.nan, 3], dtype=object),
                np.array([False, False, True, False]),
            ),
            (
                np.array([1, 1 + 0j, np.nan, 3]).astype(object),
                np.array([False, False, True, False]),
            ),
        ],
    )
    def test_complex(self, value, expected):
        result = isna(value)
        if is_scalar(result):
            assert result is expected
        else:
            tm.assert_numpy_array_equal(result, expected)

    def test_datetime_other_units(self):
        idx = DatetimeIndex(["2011-01-01", "NaT", "2011-01-02"])
        exp = np.array([False, True, False])
        tm.assert_numpy_array_equal(isna(idx), exp)
        tm.assert_numpy_array_equal(notna(idx), ~exp)
        tm.assert_numpy_array_equal(isna(idx.values), exp)
        tm.assert_numpy_array_equal(notna(idx.values), ~exp)

    @pytest.mark.parametrize(
        "dtype",
        [
            "datetime64[D]",
            "datetime64[h]",
            "datetime64[m]",
            "datetime64[s]",
            "datetime64[ms]",
            "datetime64[us]",
            "datetime64[ns]",
        ],
    )
    def test_datetime_other_units_astype(self, dtype):
        idx = DatetimeIndex(["2011-01-01", "NaT", "2011-01-02"])
        values = idx.values.astype(dtype)

        exp = np.array([False, True, False])
        tm.assert_numpy_array_equal(isna(values), exp)
        tm.assert_numpy_array_equal(notna(values), ~exp)

        exp = Series([False, True, False])
        s = Series(values)
        tm.assert_series_equal(isna(s), exp)
        tm.assert_series_equal(notna(s), ~exp)
        s = Series(values, dtype=object)
        tm.assert_series_equal(isna(s), exp)
        tm.assert_series_equal(notna(s), ~exp)

    def test_timedelta_other_units(self):
        idx = TimedeltaIndex(["1 days", "NaT", "2 days"])
        exp = np.array([False, True, False])
        tm.assert_numpy_array_equal(isna(idx), exp)
        tm.assert_numpy_array_equal(notna(idx), ~exp)
        tm.assert_numpy_array_equal(isna(idx.values), exp)
        tm.assert_numpy_array_equal(notna(idx.values), ~exp)

    @pytest.mark.parametrize(
        "dtype",
        [
            "timedelta64[D]",
            "timedelta64[h]",
            "timedelta64[m]",
            "timedelta64[s]",
            "timedelta64[ms]",
            "timedelta64[us]",
            "timedelta64[ns]",
        ],
    )
    def test_timedelta_other_units_dtype(self, dtype):
        idx = TimedeltaIndex(["1 days", "NaT", "2 days"])
        values = idx.values.astype(dtype)

        exp = np.array([False, True, False])
        tm.assert_numpy_array_equal(isna(values), exp)
        tm.assert_numpy_array_equal(notna(values), ~exp)

        exp = Series([False, True, False])
        s = Series(values)
        tm.assert_series_equal(isna(s), exp)
        tm.assert_series_equal(notna(s), ~exp)
        s = Series(values, dtype=object)
        tm.assert_series_equal(isna(s), exp)
        tm.assert_series_equal(notna(s), ~exp)

    def test_period(self):
        idx = pd.PeriodIndex(["2011-01", "NaT", "2012-01"], freq="M")
        exp = np.array([False, True, False])
        tm.assert_numpy_array_equal(isna(idx), exp)
        tm.assert_numpy_array_equal(notna(idx), ~exp)

        exp = Series([False, True, False])
        s = Series(idx)
        tm.assert_series_equal(isna(s), exp)
        tm.assert_series_equal(notna(s), ~exp)
        s = Series(idx, dtype=object)
        tm.assert_series_equal(isna(s), exp)
        tm.assert_series_equal(notna(s), ~exp)

    def test_decimal(self):
        # scalars GH#23530
        a = Decimal(1.0)
        assert isna(a) is False
        assert notna(a) is True

        b = Decimal("NaN")
        assert isna(b) is True
        assert notna(b) is False

        # array
        arr = np.array([a, b])
        expected = np.array([False, True])
        result = isna(arr)
        tm.assert_numpy_array_equal(result, expected)

        result = notna(arr)
        tm.assert_numpy_array_equal(result, ~expected)

        # series
        ser = Series(arr)
        expected = Series(expected)
        result = isna(ser)
        tm.assert_series_equal(result, expected)

        result = notna(ser)
        tm.assert_series_equal(result, ~expected)

        # index
        idx = Index(arr)
        expected = np.array([False, True])
        result = isna(idx)
        tm.assert_numpy_array_equal(result, expected)

        result = notna(idx)
        tm.assert_numpy_array_equal(result, ~expected)


@pytest.mark.parametrize("dtype_equal", [True, False])
def test_array_equivalent(dtype_equal):
    assert array_equivalent(
        np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), dtype_equal=dtype_equal
    )
    assert array_equivalent(
        np.array([np.nan, 1, np.nan]),
        np.array([np.nan, 1, np.nan]),
        dtype_equal=dtype_equal,
    )
    assert array_equivalent(
        np.array([np.nan, None], dtype="object"),
        np.array([np.nan, None], dtype="object"),
        dtype_equal=dtype_equal,
    )
    # Check the handling of nested arrays in array_equivalent_object
    assert array_equivalent(
        np.array([np.array([np.nan, None], dtype="object"), None], dtype="object"),
        np.array([np.array([np.nan, None], dtype="object"), None], dtype="object"),
        dtype_equal=dtype_equal,
    )
    assert array_equivalent(
        np.array([np.nan, 1 + 1j], dtype="complex"),
        np.array([np.nan, 1 + 1j], dtype="complex"),
        dtype_equal=dtype_equal,
    )
    assert not array_equivalent(
        np.array([np.nan, 1 + 1j], dtype="complex"),
        np.array([np.nan, 1 + 2j], dtype="complex"),
        dtype_equal=dtype_equal,
    )
    assert not array_equivalent(
        np.array([np.nan, 1, np.nan]),
        np.array([np.nan, 2, np.nan]),
        dtype_equal=dtype_equal,
    )
    assert not array_equivalent(
        np.array(["a", "b", "c", "d"]), np.array(["e", "e"]), dtype_equal=dtype_equal
    )
    assert array_equivalent(
        Index([0, np.nan]), Index([0, np.nan]), dtype_equal=dtype_equal
    )
    assert not array_equivalent(
        Index([0, np.nan]), Index([1, np.nan]), dtype_equal=dtype_equal
    )


@pytest.mark.parametrize("dtype_equal", [True, False])
def test_array_equivalent_tdi(dtype_equal):
    assert array_equivalent(
        TimedeltaIndex([0, np.nan]),
        TimedeltaIndex([0, np.nan]),
        dtype_equal=dtype_equal,
    )
    assert not array_equivalent(
        TimedeltaIndex([0, np.nan]),
        TimedeltaIndex([1, np.nan]),
        dtype_equal=dtype_equal,
    )


@pytest.mark.parametrize("dtype_equal", [True, False])
def test_array_equivalent_dti(dtype_equal):
    assert array_equivalent(
        DatetimeIndex([0, np.nan]), DatetimeIndex([0, np.nan]), dtype_equal=dtype_equal
    )
    assert not array_equivalent(
        DatetimeIndex([0, np.nan]), DatetimeIndex([1, np.nan]), dtype_equal=dtype_equal
    )

    dti1 = DatetimeIndex([0, np.nan], tz="US/Eastern")
    dti2 = DatetimeIndex([0, np.nan], tz="CET")
    dti3 = DatetimeIndex([1, np.nan], tz="US/Eastern")

    assert array_equivalent(
        dti1,
        dti1,
        dtype_equal=dtype_equal,
    )
    assert not array_equivalent(
        dti1,
        dti3,
        dtype_equal=dtype_equal,
    )
    # The rest are not dtype_equal
    assert not array_equivalent(DatetimeIndex([0, np.nan]), dti1)
    assert array_equivalent(
        dti2,
        dti1,
    )

    assert not array_equivalent(DatetimeIndex([0, np.nan]), TimedeltaIndex([0, np.nan]))


@pytest.mark.parametrize(
    "val", [1, 1.1, 1 + 1j, True, "abc", [1, 2], (1, 2), {1, 2}, {"a": 1}, None]
)
def test_array_equivalent_series(val):
    arr = np.array([1, 2])
    msg = "elementwise comparison failed"
    cm = (
        # stacklevel is chosen to make sense when called from .equals
        tm.assert_produces_warning(FutureWarning, match=msg, check_stacklevel=False)
        if isinstance(val, str) and not np_version_gte1p25
        else nullcontext()
    )
    with cm:
        assert not array_equivalent(Series([arr, arr]), Series([arr, val]))


def test_array_equivalent_array_mismatched_shape():
    # to trigger the motivating bug, the first N elements of the arrays need
    #  to match
    first = np.array([1, 2, 3])
    second = np.array([1, 2])

    left = Series([first, "a"], dtype=object)
    right = Series([second, "a"], dtype=object)
    assert not array_equivalent(left, right)


def test_array_equivalent_array_mismatched_dtype():
    # same shape, different dtype can still be equivalent
    first = np.array([1, 2], dtype=np.float64)
    second = np.array([1, 2])

    left = Series([first, "a"], dtype=object)
    right = Series([second, "a"], dtype=object)
    assert array_equivalent(left, right)


def test_array_equivalent_different_dtype_but_equal():
    # Unclear if this is exposed anywhere in the public-facing API
    assert array_equivalent(np.array([1, 2]), np.array([1.0, 2.0]))


@pytest.mark.parametrize(
    "lvalue, rvalue",
    [
        # There are 3 variants for each of lvalue and rvalue. We include all
        #  three for the tz-naive `now` and exclude the datetim64 variant
        #  for utcnow because it drops tzinfo.
        (fix_now, fix_utcnow),
        (fix_now.to_datetime64(), fix_utcnow),
        (fix_now.to_pydatetime(), fix_utcnow),
        (fix_now, fix_utcnow),
        (fix_now.to_datetime64(), fix_utcnow.to_pydatetime()),
        (fix_now.to_pydatetime(), fix_utcnow.to_pydatetime()),
    ],
)
def test_array_equivalent_tzawareness(lvalue, rvalue):
    # we shouldn't raise if comparing tzaware and tznaive datetimes
    left = np.array([lvalue], dtype=object)
    right = np.array([rvalue], dtype=object)

    assert not array_equivalent(left, right, strict_nan=True)
    assert not array_equivalent(left, right, strict_nan=False)


def test_array_equivalent_compat():
    # see gh-13388
    m = np.array([(1, 2), (3, 4)], dtype=[("a", int), ("b", float)])
    n = np.array([(1, 2), (3, 4)], dtype=[("a", int), ("b", float)])
    assert array_equivalent(m, n, strict_nan=True)
    assert array_equivalent(m, n, strict_nan=False)

    m = np.array([(1, 2), (3, 4)], dtype=[("a", int), ("b", float)])
    n = np.array([(1, 2), (4, 3)], dtype=[("a", int), ("b", float)])
    assert not array_equivalent(m, n, strict_nan=True)
    assert not array_equivalent(m, n, strict_nan=False)

    m = np.array([(1, 2), (3, 4)], dtype=[("a", int), ("b", float)])
    n = np.array([(1, 2), (3, 4)], dtype=[("b", int), ("a", float)])
    assert not array_equivalent(m, n, strict_nan=True)
    assert not array_equivalent(m, n, strict_nan=False)


@pytest.mark.parametrize("dtype", ["O", "S", "U"])
def test_array_equivalent_str(dtype):
    assert array_equivalent(
        np.array(["A", "B"], dtype=dtype), np.array(["A", "B"], dtype=dtype)
    )
    assert not array_equivalent(
        np.array(["A", "B"], dtype=dtype), np.array(["A", "X"], dtype=dtype)
    )


@pytest.mark.parametrize("strict_nan", [True, False])
def test_array_equivalent_nested(strict_nan):
    # reached in groupby aggregations, make sure we use np.any when checking
    #  if the comparison is truthy
    left = np.array([np.array([50, 70, 90]), np.array([20, 30])], dtype=object)
    right = np.array([np.array([50, 70, 90]), np.array([20, 30])], dtype=object)

    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)

    left = np.empty(2, dtype=object)
    left[:] = [np.array([50, 70, 90]), np.array([20, 30, 40])]
    right = np.empty(2, dtype=object)
    right[:] = [np.array([50, 70, 90]), np.array([20, 30, 40])]
    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)

    left = np.array([np.array([50, 50, 50]), np.array([40, 40])], dtype=object)
    right = np.array([50, 40])
    assert not array_equivalent(left, right, strict_nan=strict_nan)


@pytest.mark.filterwarnings("ignore:elementwise comparison failed:DeprecationWarning")
@pytest.mark.parametrize("strict_nan", [True, False])
def test_array_equivalent_nested2(strict_nan):
    # more than one level of nesting
    left = np.array(
        [
            np.array([np.array([50, 70]), np.array([90])], dtype=object),
            np.array([np.array([20, 30])], dtype=object),
        ],
        dtype=object,
    )
    right = np.array(
        [
            np.array([np.array([50, 70]), np.array([90])], dtype=object),
            np.array([np.array([20, 30])], dtype=object),
        ],
        dtype=object,
    )
    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)

    left = np.array([np.array([np.array([50, 50, 50])], dtype=object)], dtype=object)
    right = np.array([50])
    assert not array_equivalent(left, right, strict_nan=strict_nan)


@pytest.mark.parametrize("strict_nan", [True, False])
def test_array_equivalent_nested_list(strict_nan):
    left = np.array([[50, 70, 90], [20, 30]], dtype=object)
    right = np.array([[50, 70, 90], [20, 30]], dtype=object)

    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)

    left = np.array([[50, 50, 50], [40, 40]], dtype=object)
    right = np.array([50, 40])
    assert not array_equivalent(left, right, strict_nan=strict_nan)


@pytest.mark.filterwarnings("ignore:elementwise comparison failed:DeprecationWarning")
@pytest.mark.xfail(reason="failing")
@pytest.mark.parametrize("strict_nan", [True, False])
def test_array_equivalent_nested_mixed_list(strict_nan):
    # mixed arrays / lists in left and right
    # https://github.com/pandas-dev/pandas/issues/50360
    left = np.array([np.array([1, 2, 3]), np.array([4, 5])], dtype=object)
    right = np.array([[1, 2, 3], [4, 5]], dtype=object)

    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)

    # multiple levels of nesting
    left = np.array(
        [
            np.array([np.array([1, 2, 3]), np.array([4, 5])], dtype=object),
            np.array([np.array([6]), np.array([7, 8]), np.array([9])], dtype=object),
        ],
        dtype=object,
    )
    right = np.array([[[1, 2, 3], [4, 5]], [[6], [7, 8], [9]]], dtype=object)
    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)

    # same-length lists
    subarr = np.empty(2, dtype=object)
    subarr[:] = [
        np.array([None, "b"], dtype=object),
        np.array(["c", "d"], dtype=object),
    ]
    left = np.array([subarr, None], dtype=object)
    right = np.array([[[None, "b"], ["c", "d"]], None], dtype=object)
    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)


@pytest.mark.xfail(reason="failing")
@pytest.mark.parametrize("strict_nan", [True, False])
def test_array_equivalent_nested_dicts(strict_nan):
    left = np.array([{"f1": 1, "f2": np.array(["a", "b"], dtype=object)}], dtype=object)
    right = np.array(
        [{"f1": 1, "f2": np.array(["a", "b"], dtype=object)}], dtype=object
    )
    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)

    right2 = np.array([{"f1": 1, "f2": ["a", "b"]}], dtype=object)
    assert array_equivalent(left, right2, strict_nan=strict_nan)
    assert not array_equivalent(left, right2[::-1], strict_nan=strict_nan)


def test_array_equivalent_index_with_tuples():
    # GH#48446
    idx1 = Index(np.array([(pd.NA, 4), (1, 1)], dtype="object"))
    idx2 = Index(np.array([(1, 1), (pd.NA, 4)], dtype="object"))
    assert not array_equivalent(idx1, idx2)
    assert not idx1.equals(idx2)
    assert not array_equivalent(idx2, idx1)
    assert not idx2.equals(idx1)

    idx1 = Index(np.array([(4, pd.NA), (1, 1)], dtype="object"))
    idx2 = Index(np.array([(1, 1), (4, pd.NA)], dtype="object"))
    assert not array_equivalent(idx1, idx2)
    assert not idx1.equals(idx2)
    assert not array_equivalent(idx2, idx1)
    assert not idx2.equals(idx1)


@pytest.mark.parametrize(
    "dtype, na_value",
    [
        # Datetime-like
        (np.dtype("M8[ns]"), np.datetime64("NaT", "ns")),
        (np.dtype("m8[ns]"), np.timedelta64("NaT", "ns")),
        (DatetimeTZDtype.construct_from_string("datetime64[ns, US/Eastern]"), NaT),
        (PeriodDtype("M"), NaT),
        # Integer
        ("u1", 0),
        ("u2", 0),
        ("u4", 0),
        ("u8", 0),
        ("i1", 0),
        ("i2", 0),
        ("i4", 0),
        ("i8", 0),
        # Bool
        ("bool", False),
        # Float
        ("f2", np.nan),
        ("f4", np.nan),
        ("f8", np.nan),
        # Object
        ("O", np.nan),
        # Interval
        (IntervalDtype(), np.nan),
    ],
)
def test_na_value_for_dtype(dtype, na_value):
    result = na_value_for_dtype(pandas_dtype(dtype))
    # identify check doesn't work for datetime64/timedelta64("NaT") bc they
    #  are not singletons
    assert result is na_value or (
        isna(result) and isna(na_value) and type(result) is type(na_value)
    )


class TestNAObj:
    def _check_behavior(self, arr, expected):
        result = libmissing.isnaobj(arr)
        tm.assert_numpy_array_equal(result, expected)
        result = libmissing.isnaobj(arr, inf_as_na=True)
        tm.assert_numpy_array_equal(result, expected)

        arr = np.atleast_2d(arr)
        expected = np.atleast_2d(expected)

        result = libmissing.isnaobj(arr)
        tm.assert_numpy_array_equal(result, expected)
        result = libmissing.isnaobj(arr, inf_as_na=True)
        tm.assert_numpy_array_equal(result, expected)

        # Test fortran order
        arr = arr.copy(order="F")
        result = libmissing.isnaobj(arr)
        tm.assert_numpy_array_equal(result, expected)
        result = libmissing.isnaobj(arr, inf_as_na=True)
        tm.assert_numpy_array_equal(result, expected)

    def test_basic(self):
        arr = np.array([1, None, "foo", -5.1, NaT, np.nan])
        expected = np.array([False, True, False, False, True, True])

        self._check_behavior(arr, expected)

    def test_non_obj_dtype(self):
        arr = np.array([1, 3, np.nan, 5], dtype=float)
        expected = np.array([False, False, True, False])

        self._check_behavior(arr, expected)

    def test_empty_arr(self):
        arr = np.array([])
        expected = np.array([], dtype=bool)

        self._check_behavior(arr, expected)

    def test_empty_str_inp(self):
        arr = np.array([""])  # empty but not na
        expected = np.array([False])

        self._check_behavior(arr, expected)

    def test_empty_like(self):
        # see gh-13717: no segfaults!
        arr = np.empty_like([None])
        expected = np.array([True])

        self._check_behavior(arr, expected)


m8_units = ["as", "ps", "ns", "us", "ms", "s", "m", "h", "D", "W", "M", "Y"]

na_vals = (
    [
        None,
        NaT,
        float("NaN"),
        complex("NaN"),
        np.nan,
        np.float64("NaN"),
        np.float32("NaN"),
        np.complex64(np.nan),
        np.complex128(np.nan),
        np.datetime64("NaT"),
        np.timedelta64("NaT"),
    ]
    + [np.datetime64("NaT", unit) for unit in m8_units]
    + [np.timedelta64("NaT", unit) for unit in m8_units]
)

inf_vals = [
    float("inf"),
    float("-inf"),
    complex("inf"),
    complex("-inf"),
    np.inf,
    -np.inf,
]

int_na_vals = [
    # Values that match iNaT, which we treat as null in specific cases
    np.int64(NaT._value),
    int(NaT._value),
]

sometimes_na_vals = [Decimal("NaN")]

never_na_vals = [
    # float/complex values that when viewed as int64 match iNaT
    -0.0,
    np.float64("-0.0"),
    -0j,
    np.complex64(-0j),
]


class TestLibMissing:
    @pytest.mark.parametrize("func", [libmissing.checknull, isna])
    @pytest.mark.parametrize(
        "value", na_vals + sometimes_na_vals  # type: ignore[operator]
    )
    def test_checknull_na_vals(self, func, value):
        assert func(value)

    @pytest.mark.parametrize("func", [libmissing.checknull, isna])
    @pytest.mark.parametrize("value", inf_vals)
    def test_checknull_inf_vals(self, func, value):
        assert not func(value)

    @pytest.mark.parametrize("func", [libmissing.checknull, isna])
    @pytest.mark.parametrize("value", int_na_vals)
    def test_checknull_intna_vals(self, func, value):
        assert not func(value)

    @pytest.mark.parametrize("func", [libmissing.checknull, isna])
    @pytest.mark.parametrize("value", never_na_vals)
    def test_checknull_never_na_vals(self, func, value):
        assert not func(value)

    @pytest.mark.parametrize(
        "value", na_vals + sometimes_na_vals  # type: ignore[operator]
    )
    def test_checknull_old_na_vals(self, value):
        assert libmissing.checknull(value, inf_as_na=True)

    @pytest.mark.parametrize("value", inf_vals)
    def test_checknull_old_inf_vals(self, value):
        assert libmissing.checknull(value, inf_as_na=True)

    @pytest.mark.parametrize("value", int_na_vals)
    def test_checknull_old_intna_vals(self, value):
        assert not libmissing.checknull(value, inf_as_na=True)

    @pytest.mark.parametrize("value", int_na_vals)
    def test_checknull_old_never_na_vals(self, value):
        assert not libmissing.checknull(value, inf_as_na=True)

    def test_is_matching_na(self, nulls_fixture, nulls_fixture2):
        left = nulls_fixture
        right = nulls_fixture2

        assert libmissing.is_matching_na(left, left)

        if left is right:
            assert libmissing.is_matching_na(left, right)
        elif is_float(left) and is_float(right):
            # np.nan vs float("NaN") we consider as matching
            assert libmissing.is_matching_na(left, right)
        elif type(left) is type(right):
            # e.g. both Decimal("NaN")
            assert libmissing.is_matching_na(left, right)
        else:
            assert not libmissing.is_matching_na(left, right)

    def test_is_matching_na_nan_matches_none(self):
        assert not libmissing.is_matching_na(None, np.nan)
        assert not libmissing.is_matching_na(np.nan, None)

        assert libmissing.is_matching_na(None, np.nan, nan_matches_none=True)
        assert libmissing.is_matching_na(np.nan, None, nan_matches_none=True)


class TestIsValidNAForDtype:
    def test_is_valid_na_for_dtype_interval(self):
        dtype = IntervalDtype("int64", "left")
        assert not is_valid_na_for_dtype(NaT, dtype)

        dtype = IntervalDtype("datetime64[ns]", "both")
        assert not is_valid_na_for_dtype(NaT, dtype)

    def test_is_valid_na_for_dtype_categorical(self):
        dtype = CategoricalDtype(categories=[0, 1, 2])
        assert is_valid_na_for_dtype(np.nan, dtype)

        assert not is_valid_na_for_dtype(NaT, dtype)
        assert not is_valid_na_for_dtype(np.datetime64("NaT", "ns"), dtype)
        assert not is_valid_na_for_dtype(np.timedelta64("NaT", "ns"), dtype)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\methods\test_nth.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    isna,
)
import pandas._testing as tm


def test_first_last_nth(df):
    # tests for first / last / nth
    grouped = df.groupby("A")
    first = grouped.first()
    expected = df.loc[[1, 0], ["B", "C", "D"]]
    expected.index = Index(["bar", "foo"], name="A")
    expected = expected.sort_index()
    tm.assert_frame_equal(first, expected)

    nth = grouped.nth(0)
    expected = df.loc[[0, 1]]
    tm.assert_frame_equal(nth, expected)

    last = grouped.last()
    expected = df.loc[[5, 7], ["B", "C", "D"]]
    expected.index = Index(["bar", "foo"], name="A")
    tm.assert_frame_equal(last, expected)

    nth = grouped.nth(-1)
    expected = df.iloc[[5, 7]]
    tm.assert_frame_equal(nth, expected)

    nth = grouped.nth(1)
    expected = df.iloc[[2, 3]]
    tm.assert_frame_equal(nth, expected)

    # it works!
    grouped["B"].first()
    grouped["B"].last()
    grouped["B"].nth(0)

    df = df.copy()
    df.loc[df["A"] == "foo", "B"] = np.nan
    grouped = df.groupby("A")
    assert isna(grouped["B"].first()["foo"])
    assert isna(grouped["B"].last()["foo"])
    assert isna(grouped["B"].nth(0).iloc[0])

    # v0.14.0 whatsnew
    df = DataFrame([[1, np.nan], [1, 4], [5, 6]], columns=["A", "B"])
    g = df.groupby("A")
    result = g.first()
    expected = df.iloc[[1, 2]].set_index("A")
    tm.assert_frame_equal(result, expected)

    expected = df.iloc[[1, 2]]
    result = g.nth(0, dropna="any")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("method", ["first", "last"])
def test_first_last_with_na_object(method, nulls_fixture):
    # https://github.com/pandas-dev/pandas/issues/32123
    groups = DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 3, nulls_fixture]}).groupby("a")
    result = getattr(groups, method)()

    if method == "first":
        values = [1, 3]
    else:
        values = [2, 3]

    values = np.array(values, dtype=result["b"].dtype)
    idx = Index([1, 2], name="a")
    expected = DataFrame({"b": values}, index=idx)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("index", [0, -1])
def test_nth_with_na_object(index, nulls_fixture):
    # https://github.com/pandas-dev/pandas/issues/32123
    df = DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 3, nulls_fixture]})
    groups = df.groupby("a")
    result = groups.nth(index)
    expected = df.iloc[[0, 2]] if index == 0 else df.iloc[[1, 3]]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("method", ["first", "last"])
def test_first_last_with_None(method):
    # https://github.com/pandas-dev/pandas/issues/32800
    # None should be preserved as object dtype
    df = DataFrame.from_dict({"id": ["a"], "value": [None]})
    groups = df.groupby("id", as_index=False)
    result = getattr(groups, method)()

    tm.assert_frame_equal(result, df)


@pytest.mark.parametrize("method", ["first", "last"])
@pytest.mark.parametrize(
    "df, expected",
    [
        (
            DataFrame({"id": "a", "value": [None, "foo", np.nan]}),
            DataFrame({"value": ["foo"]}, index=Index(["a"], name="id")),
        ),
        (
            DataFrame({"id": "a", "value": [np.nan]}, dtype=object),
            DataFrame({"value": [None]}, index=Index(["a"], name="id")),
        ),
    ],
)
def test_first_last_with_None_expanded(method, df, expected):
    # GH 32800, 38286
    result = getattr(df.groupby("id"), method)()
    tm.assert_frame_equal(result, expected)


def test_first_last_nth_dtypes():
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
            "C": np.random.default_rng(2).standard_normal(8),
            "D": np.array(np.random.default_rng(2).standard_normal(8), dtype="float32"),
        }
    )
    df["E"] = True
    df["F"] = 1

    # tests for first / last / nth
    grouped = df.groupby("A")
    first = grouped.first()
    expected = df.loc[[1, 0], ["B", "C", "D", "E", "F"]]
    expected.index = Index(["bar", "foo"], name="A")
    expected = expected.sort_index()
    tm.assert_frame_equal(first, expected)

    last = grouped.last()
    expected = df.loc[[5, 7], ["B", "C", "D", "E", "F"]]
    expected.index = Index(["bar", "foo"], name="A")
    expected = expected.sort_index()
    tm.assert_frame_equal(last, expected)

    nth = grouped.nth(1)
    expected = df.iloc[[2, 3]]
    tm.assert_frame_equal(nth, expected)


def test_first_last_nth_dtypes2():
    # GH 2763, first/last shifting dtypes
    idx = list(range(10))
    idx.append(9)
    ser = Series(data=range(11), index=idx, name="IntCol")
    assert ser.dtype == "int64"
    f = ser.groupby(level=0).first()
    assert f.dtype == "int64"


def test_first_last_nth_nan_dtype():
    # GH 33591
    df = DataFrame({"data": ["A"], "nans": Series([None], dtype=object)})
    grouped = df.groupby("data")

    expected = df.set_index("data").nans
    tm.assert_series_equal(grouped.nans.first(), expected)
    tm.assert_series_equal(grouped.nans.last(), expected)

    expected = df.nans
    tm.assert_series_equal(grouped.nans.nth(-1), expected)
    tm.assert_series_equal(grouped.nans.nth(0), expected)


def test_first_strings_timestamps():
    # GH 11244
    test = DataFrame(
        {
            Timestamp("2012-01-01 00:00:00"): ["a", "b"],
            Timestamp("2012-01-02 00:00:00"): ["c", "d"],
            "name": ["e", "e"],
            "aaaa": ["f", "g"],
        }
    )
    result = test.groupby("name").first()
    expected = DataFrame(
        [["a", "c", "f"]],
        columns=Index([Timestamp("2012-01-01"), Timestamp("2012-01-02"), "aaaa"]),
        index=Index(["e"], name="name"),
    )
    tm.assert_frame_equal(result, expected)


def test_nth():
    df = DataFrame([[1, np.nan], [1, 4], [5, 6]], columns=["A", "B"])
    gb = df.groupby("A")

    tm.assert_frame_equal(gb.nth(0), df.iloc[[0, 2]])
    tm.assert_frame_equal(gb.nth(1), df.iloc[[1]])
    tm.assert_frame_equal(gb.nth(2), df.loc[[]])
    tm.assert_frame_equal(gb.nth(-1), df.iloc[[1, 2]])
    tm.assert_frame_equal(gb.nth(-2), df.iloc[[0]])
    tm.assert_frame_equal(gb.nth(-3), df.loc[[]])
    tm.assert_series_equal(gb.B.nth(0), df.B.iloc[[0, 2]])
    tm.assert_series_equal(gb.B.nth(1), df.B.iloc[[1]])
    tm.assert_frame_equal(gb[["B"]].nth(0), df[["B"]].iloc[[0, 2]])

    tm.assert_frame_equal(gb.nth(0, dropna="any"), df.iloc[[1, 2]])
    tm.assert_frame_equal(gb.nth(-1, dropna="any"), df.iloc[[1, 2]])

    tm.assert_frame_equal(gb.nth(7, dropna="any"), df.iloc[:0])
    tm.assert_frame_equal(gb.nth(2, dropna="any"), df.iloc[:0])


def test_nth2():
    # out of bounds, regression from 0.13.1
    # GH 6621
    df = DataFrame(
        {
            "color": {0: "green", 1: "green", 2: "red", 3: "red", 4: "red"},
            "food": {0: "ham", 1: "eggs", 2: "eggs", 3: "ham", 4: "pork"},
            "two": {
                0: 1.5456590000000001,
                1: -0.070345000000000005,
                2: -2.4004539999999999,
                3: 0.46206000000000003,
                4: 0.52350799999999997,
            },
            "one": {
                0: 0.56573799999999996,
                1: -0.9742360000000001,
                2: 1.033801,
                3: -0.78543499999999999,
                4: 0.70422799999999997,
            },
        }
    ).set_index(["color", "food"])

    result = df.groupby(level=0, as_index=False).nth(2)
    expected = df.iloc[[-1]]
    tm.assert_frame_equal(result, expected)

    result = df.groupby(level=0, as_index=False).nth(3)
    expected = df.loc[[]]
    tm.assert_frame_equal(result, expected)


def test_nth3():
    # GH 7559
    # from the vbench
    df = DataFrame(np.random.default_rng(2).integers(1, 10, (100, 2)), dtype="int64")
    ser = df[1]
    gb = df[0]
    expected = ser.groupby(gb).first()
    expected2 = ser.groupby(gb).apply(lambda x: x.iloc[0])
    tm.assert_series_equal(expected2, expected, check_names=False)
    assert expected.name == 1
    assert expected2.name == 1

    # validate first
    v = ser[gb == 1].iloc[0]
    assert expected.iloc[0] == v
    assert expected2.iloc[0] == v

    with pytest.raises(ValueError, match="For a DataFrame"):
        ser.groupby(gb, sort=False).nth(0, dropna=True)


def test_nth4():
    # doc example
    df = DataFrame([[1, np.nan], [1, 4], [5, 6]], columns=["A", "B"])
    gb = df.groupby("A")
    result = gb.B.nth(0, dropna="all")
    expected = df.B.iloc[[1, 2]]
    tm.assert_series_equal(result, expected)


def test_nth5():
    # test multiple nth values
    df = DataFrame([[1, np.nan], [1, 3], [1, 4], [5, 6], [5, 7]], columns=["A", "B"])
    gb = df.groupby("A")

    tm.assert_frame_equal(gb.nth(0), df.iloc[[0, 3]])
    tm.assert_frame_equal(gb.nth([0]), df.iloc[[0, 3]])
    tm.assert_frame_equal(gb.nth([0, 1]), df.iloc[[0, 1, 3, 4]])
    tm.assert_frame_equal(gb.nth([0, -1]), df.iloc[[0, 2, 3, 4]])
    tm.assert_frame_equal(gb.nth([0, 1, 2]), df.iloc[[0, 1, 2, 3, 4]])
    tm.assert_frame_equal(gb.nth([0, 1, -1]), df.iloc[[0, 1, 2, 3, 4]])
    tm.assert_frame_equal(gb.nth([2]), df.iloc[[2]])
    tm.assert_frame_equal(gb.nth([3, 4]), df.loc[[]])


def test_nth_bdays(unit):
    business_dates = pd.date_range(
        start="4/1/2014", end="6/30/2014", freq="B", unit=unit
    )
    df = DataFrame(1, index=business_dates, columns=["a", "b"])
    # get the first, fourth and last two business days for each month
    key = [df.index.year, df.index.month]
    result = df.groupby(key, as_index=False).nth([0, 3, -2, -1])
    expected_dates = pd.to_datetime(
        [
            "2014/4/1",
            "2014/4/4",
            "2014/4/29",
            "2014/4/30",
            "2014/5/1",
            "2014/5/6",
            "2014/5/29",
            "2014/5/30",
            "2014/6/2",
            "2014/6/5",
            "2014/6/27",
            "2014/6/30",
        ]
    ).as_unit(unit)
    expected = DataFrame(1, columns=["a", "b"], index=expected_dates)
    tm.assert_frame_equal(result, expected)


def test_nth_multi_grouper(three_group):
    # PR 9090, related to issue 8979
    # test nth on multiple groupers
    grouped = three_group.groupby(["A", "B"])
    result = grouped.nth(0)
    expected = three_group.iloc[[0, 3, 4, 7]]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data, expected_first, expected_last",
    [
        (
            {
                "id": ["A"],
                "time": Timestamp("2012-02-01 14:00:00", tz="US/Central"),
                "foo": [1],
            },
            {
                "id": ["A"],
                "time": Timestamp("2012-02-01 14:00:00", tz="US/Central"),
                "foo": [1],
            },
            {
                "id": ["A"],
                "time": Timestamp("2012-02-01 14:00:00", tz="US/Central"),
                "foo": [1],
            },
        ),
        (
            {
                "id": ["A", "B", "A"],
                "time": [
                    Timestamp("2012-01-01 13:00:00", tz="America/New_York"),
                    Timestamp("2012-02-01 14:00:00", tz="US/Central"),
                    Timestamp("2012-03-01 12:00:00", tz="Europe/London"),
                ],
                "foo": [1, 2, 3],
            },
            {
                "id": ["A", "B"],
                "time": [
                    Timestamp("2012-01-01 13:00:00", tz="America/New_York"),
                    Timestamp("2012-02-01 14:00:00", tz="US/Central"),
                ],
                "foo": [1, 2],
            },
            {
                "id": ["A", "B"],
                "time": [
                    Timestamp("2012-03-01 12:00:00", tz="Europe/London"),
                    Timestamp("2012-02-01 14:00:00", tz="US/Central"),
                ],
                "foo": [3, 2],
            },
        ),
    ],
)
def test_first_last_tz(data, expected_first, expected_last):
    # GH15884
    # Test that the timezone is retained when calling first
    # or last on groupby with as_index=False

    df = DataFrame(data)

    result = df.groupby("id", as_index=False).first()
    expected = DataFrame(expected_first)
    cols = ["id", "time", "foo"]
    tm.assert_frame_equal(result[cols], expected[cols])

    result = df.groupby("id", as_index=False)["time"].first()
    tm.assert_frame_equal(result, expected[["id", "time"]])

    result = df.groupby("id", as_index=False).last()
    expected = DataFrame(expected_last)
    cols = ["id", "time", "foo"]
    tm.assert_frame_equal(result[cols], expected[cols])

    result = df.groupby("id", as_index=False)["time"].last()
    tm.assert_frame_equal(result, expected[["id", "time"]])


@pytest.mark.parametrize(
    "method, ts, alpha",
    [
        ["first", Timestamp("2013-01-01", tz="US/Eastern"), "a"],
        ["last", Timestamp("2013-01-02", tz="US/Eastern"), "b"],
    ],
)
def test_first_last_tz_multi_column(method, ts, alpha, unit):
    # GH 21603
    category_string = Series(list("abc")).astype("category")
    dti = pd.date_range("20130101", periods=3, tz="US/Eastern", unit=unit)
    df = DataFrame(
        {
            "group": [1, 1, 2],
            "category_string": category_string,
            "datetimetz": dti,
        }
    )
    result = getattr(df.groupby("group"), method)()
    expected = DataFrame(
        {
            "category_string": pd.Categorical(
                [alpha, "c"], dtype=category_string.dtype
            ),
            "datetimetz": [ts, Timestamp("2013-01-03", tz="US/Eastern")],
        },
        index=Index([1, 2], name="group"),
    )
    expected["datetimetz"] = expected["datetimetz"].dt.as_unit(unit)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "values",
    [
        pd.array([True, False], dtype="boolean"),
        pd.array([1, 2], dtype="Int64"),
        pd.to_datetime(["2020-01-01", "2020-02-01"]),
        pd.to_timedelta([1, 2], unit="D"),
    ],
)
@pytest.mark.parametrize("function", ["first", "last", "min", "max"])
def test_first_last_extension_array_keeps_dtype(values, function):
    # https://github.com/pandas-dev/pandas/issues/33071
    # https://github.com/pandas-dev/pandas/issues/32194
    df = DataFrame({"a": [1, 2], "b": values})
    grouped = df.groupby("a")
    idx = Index([1, 2], name="a")
    expected_series = Series(values, name="b", index=idx)
    expected_frame = DataFrame({"b": values}, index=idx)

    result_series = getattr(grouped["b"], function)()
    tm.assert_series_equal(result_series, expected_series)

    result_frame = grouped.agg({"b": function})
    tm.assert_frame_equal(result_frame, expected_frame)


def test_nth_multi_index_as_expected():
    # PR 9090, related to issue 8979
    # test nth on MultiIndex
    three_group = DataFrame(
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
        }
    )
    grouped = three_group.groupby(["A", "B"])
    result = grouped.nth(0)
    expected = three_group.iloc[[0, 3, 4, 7]]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "op, n, expected_rows",
    [
        ("head", -1, [0]),
        ("head", 0, []),
        ("head", 1, [0, 2]),
        ("head", 7, [0, 1, 2]),
        ("tail", -1, [1]),
        ("tail", 0, []),
        ("tail", 1, [1, 2]),
        ("tail", 7, [0, 1, 2]),
    ],
)
@pytest.mark.parametrize("columns", [None, [], ["A"], ["B"], ["A", "B"]])
@pytest.mark.parametrize("as_index", [True, False])
def test_groupby_head_tail(op, n, expected_rows, columns, as_index):
    df = DataFrame([[1, 2], [1, 4], [5, 6]], columns=["A", "B"])
    g = df.groupby("A", as_index=as_index)
    expected = df.iloc[expected_rows]
    if columns is not None:
        g = g[columns]
        expected = expected[columns]
    result = getattr(g, op)(n)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "op, n, expected_cols",
    [
        ("head", -1, [0]),
        ("head", 0, []),
        ("head", 1, [0, 2]),
        ("head", 7, [0, 1, 2]),
        ("tail", -1, [1]),
        ("tail", 0, []),
        ("tail", 1, [1, 2]),
        ("tail", 7, [0, 1, 2]),
    ],
)
def test_groupby_head_tail_axis_1(op, n, expected_cols):
    # GH 9772
    df = DataFrame(
        [[1, 2, 3], [1, 4, 5], [2, 6, 7], [3, 8, 9]], columns=["A", "B", "C"]
    )
    msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        g = df.groupby([0, 0, 1], axis=1)
    expected = df.iloc[:, expected_cols]
    result = getattr(g, op)(n)
    tm.assert_frame_equal(result, expected)


def test_group_selection_cache():
    # GH 12839 nth, head, and tail should return same result consistently
    df = DataFrame([[1, 2], [1, 4], [5, 6]], columns=["A", "B"])
    expected = df.iloc[[0, 2]]

    g = df.groupby("A")
    result1 = g.head(n=2)
    result2 = g.nth(0)
    tm.assert_frame_equal(result1, df)
    tm.assert_frame_equal(result2, expected)

    g = df.groupby("A")
    result1 = g.tail(n=2)
    result2 = g.nth(0)
    tm.assert_frame_equal(result1, df)
    tm.assert_frame_equal(result2, expected)

    g = df.groupby("A")
    result1 = g.nth(0)
    result2 = g.head(n=2)
    tm.assert_frame_equal(result1, expected)
    tm.assert_frame_equal(result2, df)

    g = df.groupby("A")
    result1 = g.nth(0)
    result2 = g.tail(n=2)
    tm.assert_frame_equal(result1, expected)
    tm.assert_frame_equal(result2, df)


def test_nth_empty():
    # GH 16064
    df = DataFrame(index=[0], columns=["a", "b", "c"])
    result = df.groupby("a").nth(10)
    expected = df.iloc[:0]
    tm.assert_frame_equal(result, expected)

    result = df.groupby(["a", "b"]).nth(10)
    expected = df.iloc[:0]
    tm.assert_frame_equal(result, expected)


def test_nth_column_order():
    # GH 20760
    # Check that nth preserves column order
    df = DataFrame(
        [[1, "b", 100], [1, "a", 50], [1, "a", np.nan], [2, "c", 200], [2, "d", 150]],
        columns=["A", "C", "B"],
    )
    result = df.groupby("A").nth(0)
    expected = df.iloc[[0, 3]]
    tm.assert_frame_equal(result, expected)

    result = df.groupby("A").nth(-1, dropna="any")
    expected = df.iloc[[1, 4]]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dropna", [None, "any", "all"])
def test_nth_nan_in_grouper(dropna):
    # GH 26011
    df = DataFrame(
        {
            "a": [np.nan, "a", np.nan, "b", np.nan],
            "b": [0, 2, 4, 6, 8],
            "c": [1, 3, 5, 7, 9],
        }
    )
    result = df.groupby("a").nth(0, dropna=dropna)
    expected = df.iloc[[1, 3]]

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dropna", [None, "any", "all"])
def test_nth_nan_in_grouper_series(dropna):
    # GH 26454
    df = DataFrame(
        {
            "a": [np.nan, "a", np.nan, "b", np.nan],
            "b": [0, 2, 4, 6, 8],
        }
    )
    result = df.groupby("a")["b"].nth(0, dropna=dropna)
    expected = df["b"].iloc[[1, 3]]

    tm.assert_series_equal(result, expected)


def test_first_categorical_and_datetime_data_nat():
    # GH 20520
    df = DataFrame(
        {
            "group": ["first", "first", "second", "third", "third"],
            "time": 5 * [np.datetime64("NaT")],
            "categories": Series(["a", "b", "c", "a", "b"], dtype="category"),
        }
    )
    result = df.groupby("group").first()
    expected = DataFrame(
        {
            "time": 3 * [np.datetime64("NaT")],
            "categories": Series(["a", "c", "a"]).astype(
                pd.CategoricalDtype(["a", "b", "c"])
            ),
        }
    )
    expected.index = Index(["first", "second", "third"], name="group")
    tm.assert_frame_equal(result, expected)


def test_first_multi_key_groupby_categorical():
    # GH 22512
    df = DataFrame(
        {
            "A": [1, 1, 1, 2, 2],
            "B": [100, 100, 200, 100, 100],
            "C": ["apple", "orange", "mango", "mango", "orange"],
            "D": ["jupiter", "mercury", "mars", "venus", "venus"],
        }
    )
    df = df.astype({"D": "category"})
    result = df.groupby(by=["A", "B"]).first()
    expected = DataFrame(
        {
            "C": ["apple", "mango", "mango"],
            "D": Series(["jupiter", "mars", "venus"]).astype(
                pd.CategoricalDtype(["jupiter", "mars", "mercury", "venus"])
            ),
        }
    )
    expected.index = MultiIndex.from_tuples(
        [(1, 100), (1, 200), (2, 100)], names=["A", "B"]
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("method", ["first", "last", "nth"])
def test_groupby_last_first_nth_with_none(method, nulls_fixture):
    # GH29645
    expected = Series(["y"], dtype=object)
    data = Series(
        [nulls_fixture, nulls_fixture, nulls_fixture, "y", nulls_fixture],
        index=[0, 0, 0, 0, 0],
        dtype=object,
    ).groupby(level=0)

    if method == "nth":
        result = getattr(data, method)(3)
    else:
        result = getattr(data, method)()

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "arg, expected_rows",
    [
        [slice(None, 3, 2), [0, 1, 4, 5]],
        [slice(None, -2), [0, 2, 5]],
        [[slice(None, 2), slice(-2, None)], [0, 1, 2, 3, 4, 6, 7]],
        [[0, 1, slice(-2, None)], [0, 1, 2, 3, 4, 6, 7]],
    ],
)
def test_slice(slice_test_df, slice_test_grouped, arg, expected_rows):
    # Test slices     GH #42947

    result = slice_test_grouped.nth[arg]
    equivalent = slice_test_grouped.nth(arg)
    expected = slice_test_df.iloc[expected_rows]

    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(equivalent, expected)


def test_nth_indexed(slice_test_df, slice_test_grouped):
    # Test index notation     GH #44688

    result = slice_test_grouped.nth[0, 1, -2:]
    equivalent = slice_test_grouped.nth([0, 1, slice(-2, None)])
    expected = slice_test_df.iloc[[0, 1, 2, 3, 4, 6, 7]]

    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(equivalent, expected)


def test_invalid_argument(slice_test_grouped):
    # Test for error on invalid argument

    with pytest.raises(TypeError, match="Invalid index"):
        slice_test_grouped.nth(3.14)


def test_negative_step(slice_test_grouped):
    # Test for error on negative slice step

    with pytest.raises(ValueError, match="Invalid step"):
        slice_test_grouped.nth(slice(None, None, -1))


def test_np_ints(slice_test_df, slice_test_grouped):
    # Test np ints work

    result = slice_test_grouped.nth(np.array([0, 1]))
    expected = slice_test_df.iloc[[0, 1, 2, 3, 4]]
    tm.assert_frame_equal(result, expected)


def test_groupby_nth_with_column_axis():
    # GH43926
    df = DataFrame(
        [
            [4, 5, 6],
            [8, 8, 7],
        ],
        index=["z", "y"],
        columns=["C", "B", "A"],
    )
    msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby(df.iloc[1], axis=1)
    result = gb.nth(0)
    expected = df.iloc[:, [0, 2]]
    tm.assert_frame_equal(result, expected)


def test_groupby_nth_interval():
    # GH#24205
    idx_result = MultiIndex(
        [
            pd.CategoricalIndex([pd.Interval(0, 1), pd.Interval(1, 2)]),
            pd.CategoricalIndex([pd.Interval(0, 10), pd.Interval(10, 20)]),
        ],
        [[0, 0, 0, 1, 1], [0, 1, 1, 0, -1]],
    )
    df_result = DataFrame({"col": range(len(idx_result))}, index=idx_result)
    result = df_result.groupby(level=[0, 1], observed=False).nth(0)
    val_expected = [0, 1, 3]
    idx_expected = MultiIndex(
        [
            pd.CategoricalIndex([pd.Interval(0, 1), pd.Interval(1, 2)]),
            pd.CategoricalIndex([pd.Interval(0, 10), pd.Interval(10, 20)]),
        ],
        [[0, 0, 1], [0, 1, 0]],
    )
    expected = DataFrame(val_expected, index=idx_expected, columns=["col"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "start, stop, expected_values, expected_columns",
    [
        (None, None, [0, 1, 2, 3, 4], list("ABCDE")),
        (None, 1, [0, 3], list("AD")),
        (None, 9, [0, 1, 2, 3, 4], list("ABCDE")),
        (None, -1, [0, 1, 3], list("ABD")),
        (1, None, [1, 2, 4], list("BCE")),
        (1, -1, [1], list("B")),
        (-1, None, [2, 4], list("CE")),
        (-1, 2, [4], list("E")),
    ],
)
@pytest.mark.parametrize("method", ["call", "index"])
def test_nth_slices_with_column_axis(
    start, stop, expected_values, expected_columns, method
):
    df = DataFrame([range(5)], columns=[list("ABCDE")])
    msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby([5, 5, 5, 6, 6], axis=1)
    result = {
        "call": lambda start, stop: gb.nth(slice(start, stop)),
        "index": lambda start, stop: gb.nth[start:stop],
    }[method](start, stop)
    expected = DataFrame([expected_values], columns=[expected_columns])
    tm.assert_frame_equal(result, expected)


@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in remainder:RuntimeWarning"
)
def test_head_tail_dropna_true():
    # GH#45089
    df = DataFrame(
        [["a", "z"], ["b", np.nan], ["c", np.nan], ["c", np.nan]], columns=["X", "Y"]
    )
    expected = DataFrame([["a", "z"]], columns=["X", "Y"])

    result = df.groupby(["X", "Y"]).head(n=1)
    tm.assert_frame_equal(result, expected)

    result = df.groupby(["X", "Y"]).tail(n=1)
    tm.assert_frame_equal(result, expected)

    result = df.groupby(["X", "Y"]).nth(n=0)
    tm.assert_frame_equal(result, expected)


def test_head_tail_dropna_false():
    # GH#45089
    df = DataFrame([["a", "z"], ["b", np.nan], ["c", np.nan]], columns=["X", "Y"])
    expected = DataFrame([["a", "z"], ["b", np.nan], ["c", np.nan]], columns=["X", "Y"])

    result = df.groupby(["X", "Y"], dropna=False).head(n=1)
    tm.assert_frame_equal(result, expected)

    result = df.groupby(["X", "Y"], dropna=False).tail(n=1)
    tm.assert_frame_equal(result, expected)

    result = df.groupby(["X", "Y"], dropna=False).nth(n=0)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("selection", ("b", ["b"], ["b", "c"]))
@pytest.mark.parametrize("dropna", ["any", "all", None])
def test_nth_after_selection(selection, dropna):
    # GH#11038, GH#53518
    df = DataFrame(
        {
            "a": [1, 1, 2],
            "b": [np.nan, 3, 4],
            "c": [5, 6, 7],
        }
    )
    gb = df.groupby("a")[selection]
    result = gb.nth(0, dropna=dropna)
    if dropna == "any" or (dropna == "all" and selection != ["b", "c"]):
        locs = [1, 2]
    else:
        locs = [0, 2]
    expected = df.loc[locs, selection]
    tm.assert_equal(result, expected)


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
def test_groupby_nth_int_like_precision(data):
    # GH#6620, GH#9311
    df = DataFrame({"a": [1, 1], "b": data})

    grouped = df.groupby("a")
    result = grouped.nth(0)
    expected = DataFrame({"a": 1, "b": [data[0]]})

    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_functions.py ===
from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath

def mpc_ae(a, b, eps=eps):
    res = True
    res = res and a.real.ae(b.real, eps)
    res = res and a.imag.ae(b.imag, eps)
    return res

#----------------------------------------------------------------------------
# Constants and functions
#

tpi = "3.1415926535897932384626433832795028841971693993751058209749445923078\
1640628620899862803482534211706798"
te = "2.71828182845904523536028747135266249775724709369995957496696762772407\
663035354759457138217852516642743"
tdegree = "0.017453292519943295769236907684886127134428718885417254560971914\
4017100911460344944368224156963450948221"
teuler = "0.5772156649015328606065120900824024310421593359399235988057672348\
84867726777664670936947063291746749516"
tln2 = "0.693147180559945309417232121458176568075500134360255254120680009493\
393621969694715605863326996418687542"
tln10 = "2.30258509299404568401799145468436420760110148862877297603332790096\
757260967735248023599720508959829834"
tcatalan = "0.91596559417721901505460351493238411077414937428167213426649811\
9621763019776254769479356512926115106249"
tkhinchin = "2.6854520010653064453097148354817956938203822939944629530511523\
4555721885953715200280114117493184769800"
tglaisher = "1.2824271291006226368753425688697917277676889273250011920637400\
2174040630885882646112973649195820237439420646"
tapery = "1.2020569031595942853997381615114499907649862923404988817922715553\
4183820578631309018645587360933525815"
tphi = "1.618033988749894848204586834365638117720309179805762862135448622705\
26046281890244970720720418939113748475"
tmertens = "0.26149721284764278375542683860869585905156664826119920619206421\
3924924510897368209714142631434246651052"
ttwinprime = "0.660161815846869573927812110014555778432623360284733413319448\
423335405642304495277143760031413839867912"

def test_constants():
    for prec in [3, 7, 10, 15, 20, 37, 80, 100, 29]:
        mp.dps = prec
        assert pi == mpf(tpi)
        assert e == mpf(te)
        assert degree == mpf(tdegree)
        assert euler == mpf(teuler)
        assert ln2 == mpf(tln2)
        assert ln10 == mpf(tln10)
        assert catalan == mpf(tcatalan)
        assert khinchin == mpf(tkhinchin)
        assert glaisher == mpf(tglaisher)
        assert phi == mpf(tphi)
        if prec < 50:
            assert mertens == mpf(tmertens)
            assert twinprime == mpf(ttwinprime)
    mp.dps = 15
    assert pi >= -1
    assert pi > 2
    assert pi > 3
    assert pi < 4

def test_exact_sqrts():
    for i in range(20000):
        assert sqrt(mpf(i*i)) == i
    random.seed(1)
    for prec in [100, 300, 1000, 10000]:
        mp.dps = prec
        for i in range(20):
            A = random.randint(10**(prec//2-2), 10**(prec//2-1))
            assert sqrt(mpf(A*A)) == A
    mp.dps = 15
    for i in range(100):
        for a in [1, 8, 25, 112307]:
            assert sqrt(mpf((a*a, 2*i))) == mpf((a, i))
            assert sqrt(mpf((a*a, -2*i))) == mpf((a, -i))

def test_sqrt_rounding():
    for i in [2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]:
        i = from_int(i)
        for dps in [7, 15, 83, 106, 2000]:
            mp.dps = dps
            a = mpf_pow_int(mpf_sqrt(i, mp.prec, round_down), 2, mp.prec, round_down)
            b = mpf_pow_int(mpf_sqrt(i, mp.prec, round_up), 2, mp.prec, round_up)
            assert mpf_lt(a, i)
            assert mpf_gt(b, i)
    random.seed(1234)
    prec = 100
    for rnd in [round_down, round_nearest, round_ceiling]:
        for i in range(100):
            a = mpf_rand(prec)
            b = mpf_mul(a, a)
            assert mpf_sqrt(b, prec, rnd) == a
    # Test some extreme cases
    mp.dps = 100
    a = mpf(9) + 1e-90
    b = mpf(9) - 1e-90
    mp.dps = 15
    assert sqrt(a, rounding='d') == 3
    assert sqrt(a, rounding='n') == 3
    assert sqrt(a, rounding='u') > 3
    assert sqrt(b, rounding='d') < 3
    assert sqrt(b, rounding='n') == 3
    assert sqrt(b, rounding='u') == 3
    # A worst case, from the MPFR test suite
    assert sqrt(mpf('7.0503726185518891')) == mpf('2.655253776675949')

def test_float_sqrt():
    mp.dps = 15
    # These should round identically
    for x in [0, 1e-7, 0.1, 0.5, 1, 2, 3, 4, 5, 0.333, 76.19]:
        assert sqrt(mpf(x)) == float(x)**0.5
    assert sqrt(-1) == 1j
    assert sqrt(-2).ae(cmath.sqrt(-2))
    assert sqrt(-3).ae(cmath.sqrt(-3))
    assert sqrt(-100).ae(cmath.sqrt(-100))
    assert sqrt(1j).ae(cmath.sqrt(1j))
    assert sqrt(-1j).ae(cmath.sqrt(-1j))
    assert sqrt(math.pi + math.e*1j).ae(cmath.sqrt(math.pi + math.e*1j))
    assert sqrt(math.pi - math.e*1j).ae(cmath.sqrt(math.pi - math.e*1j))

def test_hypot():
    assert hypot(0, 0) == 0
    assert hypot(0, 0.33) == mpf(0.33)
    assert hypot(0.33, 0) == mpf(0.33)
    assert hypot(-0.33, 0) == mpf(0.33)
    assert hypot(3, 4) == mpf(5)

def test_exact_cbrt():
    for i in range(0, 20000, 200):
        assert cbrt(mpf(i*i*i)) == i
    random.seed(1)
    for prec in [100, 300, 1000, 10000]:
        mp.dps = prec
        A = random.randint(10**(prec//2-2), 10**(prec//2-1))
        assert cbrt(mpf(A*A*A)) == A
    mp.dps = 15

def test_exp():
    assert exp(0) == 1
    assert exp(10000).ae(mpf('8.8068182256629215873e4342'))
    assert exp(-10000).ae(mpf('1.1354838653147360985e-4343'))
    a = exp(mpf((1, 8198646019315405, -53, 53)))
    assert(a.bc == bitcount(a.man))
    mp.prec = 67
    a = exp(mpf((1, 1781864658064754565, -60, 61)))
    assert(a.bc == bitcount(a.man))
    mp.prec = 53
    assert exp(ln2 * 10).ae(1024)
    assert exp(2+2j).ae(cmath.exp(2+2j))

def test_issue_73():
    mp.dps = 512
    a = exp(-1)
    b = exp(1)
    mp.dps = 15
    assert (+a).ae(0.36787944117144233)
    assert (+b).ae(2.7182818284590451)

def test_log():
    mp.dps = 15
    assert log(1) == 0
    for x in [0.5, 1.5, 2.0, 3.0, 100, 10**50, 1e-50]:
        assert log(x).ae(math.log(x))
        assert log(x, x) == 1
    assert log(1024, 2) == 10
    assert log(10**1234, 10) == 1234
    assert log(2+2j).ae(cmath.log(2+2j))
    # Accuracy near 1
    assert (log(0.6+0.8j).real*10**17).ae(2.2204460492503131)
    assert (log(0.6-0.8j).real*10**17).ae(2.2204460492503131)
    assert (log(0.8-0.6j).real*10**17).ae(2.2204460492503131)
    assert (log(1+1e-8j).real*10**16).ae(0.5)
    assert (log(1-1e-8j).real*10**16).ae(0.5)
    assert (log(-1+1e-8j).real*10**16).ae(0.5)
    assert (log(-1-1e-8j).real*10**16).ae(0.5)
    assert (log(1j+1e-8).real*10**16).ae(0.5)
    assert (log(1j-1e-8).real*10**16).ae(0.5)
    assert (log(-1j+1e-8).real*10**16).ae(0.5)
    assert (log(-1j-1e-8).real*10**16).ae(0.5)
    assert (log(1+1e-40j).real*10**80).ae(0.5)
    assert (log(1j+1e-40).real*10**80).ae(0.5)
    # Huge
    assert log(ldexp(1.234,10**20)).ae(log(2)*1e20)
    assert log(ldexp(1.234,10**200)).ae(log(2)*1e200)
    # Some special values
    assert log(mpc(0,0)) == mpc(-inf,0)
    assert isnan(log(mpc(nan,0)).real)
    assert isnan(log(mpc(nan,0)).imag)
    assert isnan(log(mpc(0,nan)).real)
    assert isnan(log(mpc(0,nan)).imag)
    assert isnan(log(mpc(nan,1)).real)
    assert isnan(log(mpc(nan,1)).imag)
    assert isnan(log(mpc(1,nan)).real)
    assert isnan(log(mpc(1,nan)).imag)

def test_trig_hyperb_basic():
    for x in (list(range(100)) + list(range(-100,0))):
        t = x / 4.1
        assert cos(mpf(t)).ae(math.cos(t))
        assert sin(mpf(t)).ae(math.sin(t))
        assert tan(mpf(t)).ae(math.tan(t))
        assert cosh(mpf(t)).ae(math.cosh(t))
        assert sinh(mpf(t)).ae(math.sinh(t))
        assert tanh(mpf(t)).ae(math.tanh(t))
    assert sin(1+1j).ae(cmath.sin(1+1j))
    assert sin(-4-3.6j).ae(cmath.sin(-4-3.6j))
    assert cos(1+1j).ae(cmath.cos(1+1j))
    assert cos(-4-3.6j).ae(cmath.cos(-4-3.6j))

def test_degrees():
    assert cos(0*degree) == 1
    assert cos(90*degree).ae(0)
    assert cos(180*degree).ae(-1)
    assert cos(270*degree).ae(0)
    assert cos(360*degree).ae(1)
    assert sin(0*degree) == 0
    assert sin(90*degree).ae(1)
    assert sin(180*degree).ae(0)
    assert sin(270*degree).ae(-1)
    assert sin(360*degree).ae(0)

def random_complexes(N):
    random.seed(1)
    a = []
    for i in range(N):
        x1 = random.uniform(-10, 10)
        y1 = random.uniform(-10, 10)
        x2 = random.uniform(-10, 10)
        y2 = random.uniform(-10, 10)
        z1 = complex(x1, y1)
        z2 = complex(x2, y2)
        a.append((z1, z2))
    return a

def test_complex_powers():
    for dps in [15, 30, 100]:
        # Check accuracy for complex square root
        mp.dps = dps
        a = mpc(1j)**0.5
        assert a.real == a.imag == mpf(2)**0.5 / 2
    mp.dps = 15
    random.seed(1)
    for (z1, z2) in random_complexes(100):
        assert (mpc(z1)**mpc(z2)).ae(z1**z2, 1e-12)
    assert (e**(-pi*1j)).ae(-1)
    mp.dps = 50
    assert (e**(-pi*1j)).ae(-1)
    mp.dps = 15

def test_complex_sqrt_accuracy():
    def test_mpc_sqrt(lst):
        for a, b in lst:
            z = mpc(a + j*b)
            assert mpc_ae(sqrt(z*z), z)
            z = mpc(-a + j*b)
            assert mpc_ae(sqrt(z*z), -z)
            z = mpc(a - j*b)
            assert mpc_ae(sqrt(z*z), z)
            z = mpc(-a - j*b)
            assert mpc_ae(sqrt(z*z), -z)
    random.seed(2)
    N = 10
    mp.dps = 30
    dps = mp.dps
    test_mpc_sqrt([(random.uniform(0, 10),random.uniform(0, 10)) for i in range(N)])
    test_mpc_sqrt([(i + 0.1, (i + 0.2)*10**i) for i in range(N)])
    mp.dps = 15

def test_atan():
    mp.dps = 15
    assert atan(-2.3).ae(math.atan(-2.3))
    assert atan(1e-50) == 1e-50
    assert atan(1e50).ae(pi/2)
    assert atan(-1e-50) == -1e-50
    assert atan(-1e50).ae(-pi/2)
    assert atan(10**1000).ae(pi/2)
    for dps in [25, 70, 100, 300, 1000]:
        mp.dps = dps
        assert (4*atan(1)).ae(pi)
    mp.dps = 15
    pi2 = pi/2
    assert atan(mpc(inf,-1)).ae(pi2)
    assert atan(mpc(inf,0)).ae(pi2)
    assert atan(mpc(inf,1)).ae(pi2)
    assert atan(mpc(1,inf)).ae(pi2)
    assert atan(mpc(0,inf)).ae(pi2)
    assert atan(mpc(-1,inf)).ae(-pi2)
    assert atan(mpc(-inf,1)).ae(-pi2)
    assert atan(mpc(-inf,0)).ae(-pi2)
    assert atan(mpc(-inf,-1)).ae(-pi2)
    assert atan(mpc(-1,-inf)).ae(-pi2)
    assert atan(mpc(0,-inf)).ae(-pi2)
    assert atan(mpc(1,-inf)).ae(pi2)

def test_atan2():
    mp.dps = 15
    assert atan2(1,1).ae(pi/4)
    assert atan2(1,-1).ae(3*pi/4)
    assert atan2(-1,-1).ae(-3*pi/4)
    assert atan2(-1,1).ae(-pi/4)
    assert atan2(-1,0).ae(-pi/2)
    assert atan2(1,0).ae(pi/2)
    assert atan2(0,0) == 0
    assert atan2(inf,0).ae(pi/2)
    assert atan2(-inf,0).ae(-pi/2)
    assert isnan(atan2(inf,inf))
    assert isnan(atan2(-inf,inf))
    assert isnan(atan2(inf,-inf))
    assert isnan(atan2(3,nan))
    assert isnan(atan2(nan,3))
    assert isnan(atan2(0,nan))
    assert isnan(atan2(nan,0))
    assert atan2(0,inf) == 0
    assert atan2(0,-inf).ae(pi)
    assert atan2(10,inf) == 0
    assert atan2(-10,inf) == 0
    assert atan2(-10,-inf).ae(-pi)
    assert atan2(10,-inf).ae(pi)
    assert atan2(inf,10).ae(pi/2)
    assert atan2(inf,-10).ae(pi/2)
    assert atan2(-inf,10).ae(-pi/2)
    assert atan2(-inf,-10).ae(-pi/2)

def test_areal_inverses():
    assert asin(mpf(0)) == 0
    assert asinh(mpf(0)) == 0
    assert acosh(mpf(1)) == 0
    assert isinstance(asin(mpf(0.5)), mpf)
    assert isinstance(asin(mpf(2.0)), mpc)
    assert isinstance(acos(mpf(0.5)), mpf)
    assert isinstance(acos(mpf(2.0)), mpc)
    assert isinstance(atanh(mpf(0.1)), mpf)
    assert isinstance(atanh(mpf(1.1)), mpc)

    random.seed(1)
    for i in range(50):
        x = random.uniform(0, 1)
        assert asin(mpf(x)).ae(math.asin(x))
        assert acos(mpf(x)).ae(math.acos(x))

        x = random.uniform(-10, 10)
        assert asinh(mpf(x)).ae(cmath.asinh(x).real)
        assert isinstance(asinh(mpf(x)), mpf)
        x = random.uniform(1, 10)
        assert acosh(mpf(x)).ae(cmath.acosh(x).real)
        assert isinstance(acosh(mpf(x)), mpf)
        x = random.uniform(-10, 0.999)
        assert isinstance(acosh(mpf(x)), mpc)

        x = random.uniform(-1, 1)
        assert atanh(mpf(x)).ae(cmath.atanh(x).real)
        assert isinstance(atanh(mpf(x)), mpf)

    dps = mp.dps
    mp.dps = 300
    assert isinstance(asin(0.5), mpf)
    mp.dps = 1000
    assert asin(1).ae(pi/2)
    assert asin(-1).ae(-pi/2)
    mp.dps = dps

def test_invhyperb_inaccuracy():
    mp.dps = 15
    assert (asinh(1e-5)*10**5).ae(0.99999999998333333)
    assert (asinh(1e-10)*10**10).ae(1)
    assert (asinh(1e-50)*10**50).ae(1)
    assert (asinh(-1e-5)*10**5).ae(-0.99999999998333333)
    assert (asinh(-1e-10)*10**10).ae(-1)
    assert (asinh(-1e-50)*10**50).ae(-1)
    assert asinh(10**20).ae(46.744849040440862)
    assert asinh(-10**20).ae(-46.744849040440862)
    assert (tanh(1e-10)*10**10).ae(1)
    assert (tanh(-1e-10)*10**10).ae(-1)
    assert (atanh(1e-10)*10**10).ae(1)
    assert (atanh(-1e-10)*10**10).ae(-1)

def test_complex_functions():
    for x in (list(range(10)) + list(range(-10,0))):
        for y in (list(range(10)) + list(range(-10,0))):
            z = complex(x, y)/4.3 + 0.01j
            assert exp(mpc(z)).ae(cmath.exp(z))
            assert log(mpc(z)).ae(cmath.log(z))
            assert cos(mpc(z)).ae(cmath.cos(z))
            assert sin(mpc(z)).ae(cmath.sin(z))
            assert tan(mpc(z)).ae(cmath.tan(z))
            assert sinh(mpc(z)).ae(cmath.sinh(z))
            assert cosh(mpc(z)).ae(cmath.cosh(z))
            assert tanh(mpc(z)).ae(cmath.tanh(z))

def test_complex_inverse_functions():
    mp.dps = 15
    iv.dps = 15
    for (z1, z2) in random_complexes(30):
        # apparently cmath uses a different branch, so we
        # can't use it for comparison
        assert sinh(asinh(z1)).ae(z1)
        #
        assert acosh(z1).ae(cmath.acosh(z1))
        assert atanh(z1).ae(cmath.atanh(z1))
        assert atan(z1).ae(cmath.atan(z1))
        # the reason we set a big eps here is that the cmath
        # functions are inaccurate
        assert asin(z1).ae(cmath.asin(z1), rel_eps=1e-12)
        assert acos(z1).ae(cmath.acos(z1), rel_eps=1e-12)
        one = mpf(1)
    for i in range(-9, 10, 3):
        for k in range(-9, 10, 3):
            a = 0.9*j*10**k + 0.8*one*10**i
            b = cos(acos(a))
            assert b.ae(a)
            b = sin(asin(a))
            assert b.ae(a)
    one = mpf(1)
    err = 2*10**-15
    for i in range(-9, 9, 3):
        for k in range(-9, 9, 3):
            a = -0.9*10**k + j*0.8*one*10**i
            b = cosh(acosh(a))
            assert b.ae(a, err)
            b = sinh(asinh(a))
            assert b.ae(a, err)

def test_reciprocal_functions():
    assert sec(3).ae(-1.01010866590799375)
    assert csc(3).ae(7.08616739573718592)
    assert cot(3).ae(-7.01525255143453347)
    assert sech(3).ae(0.0993279274194332078)
    assert csch(3).ae(0.0998215696688227329)
    assert coth(3).ae(1.00496982331368917)
    assert asec(3).ae(1.23095941734077468)
    assert acsc(3).ae(0.339836909454121937)
    assert acot(3).ae(0.321750554396642193)
    assert asech(0.5).ae(1.31695789692481671)
    assert acsch(3).ae(0.327450150237258443)
    assert acoth(3).ae(0.346573590279972655)
    assert acot(0).ae(1.5707963267948966192)
    assert acoth(0).ae(1.5707963267948966192j)

def test_ldexp():
    mp.dps = 15
    assert ldexp(mpf(2.5), 0) == 2.5
    assert ldexp(mpf(2.5), -1) == 1.25
    assert ldexp(mpf(2.5), 2) == 10
    assert ldexp(mpf('inf'), 3) == mpf('inf')

def test_frexp():
    mp.dps = 15
    assert frexp(0) == (0.0, 0)
    assert frexp(9) == (0.5625, 4)
    assert frexp(1) == (0.5, 1)
    assert frexp(0.2) == (0.8, -2)
    assert frexp(1000) == (0.9765625, 10)

def test_aliases():
    assert ln(7) == log(7)
    assert log10(3.75) == log(3.75,10)
    assert degrees(5.6) == 5.6 / degree
    assert radians(5.6) == 5.6 * degree
    assert power(-1,0.5) == j
    assert fmod(25,7) == 4.0 and isinstance(fmod(25,7), mpf)

def test_arg_sign():
    assert arg(3) == 0
    assert arg(-3).ae(pi)
    assert arg(j).ae(pi/2)
    assert arg(-j).ae(-pi/2)
    assert arg(0) == 0
    assert isnan(atan2(3,nan))
    assert isnan(atan2(nan,3))
    assert isnan(atan2(0,nan))
    assert isnan(atan2(nan,0))
    assert isnan(atan2(nan,nan))
    assert arg(inf) == 0
    assert arg(-inf).ae(pi)
    assert isnan(arg(nan))
    #assert arg(inf*j).ae(pi/2)
    assert sign(0) == 0
    assert sign(3) == 1
    assert sign(-3) == -1
    assert sign(inf) == 1
    assert sign(-inf) == -1
    assert isnan(sign(nan))
    assert sign(j) == j
    assert sign(-3*j) == -j
    assert sign(1+j).ae((1+j)/sqrt(2))

def test_misc_bugs():
    # test that this doesn't raise an exception
    mp.dps = 1000
    log(1302)
    mp.dps = 15

def test_arange():
    assert arange(10) == [mpf('0.0'), mpf('1.0'), mpf('2.0'), mpf('3.0'),
                          mpf('4.0'), mpf('5.0'), mpf('6.0'), mpf('7.0'),
                          mpf('8.0'), mpf('9.0')]
    assert arange(-5, 5) == [mpf('-5.0'), mpf('-4.0'), mpf('-3.0'),
                             mpf('-2.0'), mpf('-1.0'), mpf('0.0'),
                             mpf('1.0'), mpf('2.0'), mpf('3.0'), mpf('4.0')]
    assert arange(0, 1, 0.1) == [mpf('0.0'), mpf('0.10000000000000001'),
                                 mpf('0.20000000000000001'),
                                 mpf('0.30000000000000004'),
                                 mpf('0.40000000000000002'),
                                 mpf('0.5'), mpf('0.60000000000000009'),
                                 mpf('0.70000000000000007'),
                                 mpf('0.80000000000000004'),
                                 mpf('0.90000000000000002')]
    assert arange(17, -9, -3) == [mpf('17.0'), mpf('14.0'), mpf('11.0'),
                                  mpf('8.0'), mpf('5.0'), mpf('2.0'),
                                  mpf('-1.0'), mpf('-4.0'), mpf('-7.0')]
    assert arange(0.2, 0.1, -0.1) == [mpf('0.20000000000000001')]
    assert arange(0) == []
    assert arange(1000, -1) == []
    assert arange(-1.23, 3.21, -0.0000001) == []

def test_linspace():
    assert linspace(2, 9, 7) == [mpf('2.0'), mpf('3.166666666666667'),
        mpf('4.3333333333333339'), mpf('5.5'), mpf('6.666666666666667'),
        mpf('7.8333333333333339'), mpf('9.0')]
    assert linspace(2, 9, 7, endpoint=0) == [mpf('2.0'), mpf('3.0'), mpf('4.0'),
        mpf('5.0'), mpf('6.0'), mpf('7.0'), mpf('8.0')]
    assert linspace(2, 7, 1) == [mpf(2)]

def test_float_cbrt():
    mp.dps = 30
    for a in arange(0,10,0.1):
        assert cbrt(a*a*a).ae(a, eps)
    assert cbrt(-1).ae(0.5 + j*sqrt(3)/2)
    one_third = mpf(1)/3
    for a in arange(0,10,2.7) + [0.1 + 10**5]:
        a = mpc(a + 1.1j)
        r1 = cbrt(a)
        mp.dps += 10
        r2 = pow(a, one_third)
        mp.dps -= 10
        assert r1.ae(r2, eps)
    mp.dps = 100
    for n in range(100, 301, 100):
        w = 10**n + j*10**-3
        z = w*w*w
        r = cbrt(z)
        assert mpc_ae(r, w, eps)
    mp.dps = 15

def test_root():
    mp.dps = 30
    random.seed(1)
    a = random.randint(0, 10000)
    p = a*a*a
    r = nthroot(mpf(p), 3)
    assert r == a
    for n in range(4, 10):
        p = p*a
        assert nthroot(mpf(p), n) == a
    mp.dps = 40
    for n in range(10, 5000, 100):
        for a in [random.random()*10000, random.random()*10**100]:
            r = nthroot(a, n)
            r1 = pow(a, mpf(1)/n)
            assert r.ae(r1)
            r = nthroot(a, -n)
            r1 = pow(a, -mpf(1)/n)
            assert r.ae(r1)
    # XXX: this is broken right now
    # tests for nthroot rounding
    for rnd in ['nearest', 'up', 'down']:
        mp.rounding = rnd
        for n in [-5, -3, 3, 5]:
            prec = 50
            for i in range(10):
                mp.prec = prec
                a = rand()
                mp.prec = 2*prec
                b = a**n
                mp.prec = prec
                r = nthroot(b, n)
                assert r == a
    mp.dps = 30
    for n in range(3, 21):
        a = (random.random() + j*random.random())
        assert nthroot(a, n).ae(pow(a, mpf(1)/n))
        assert mpc_ae(nthroot(a, n), pow(a, mpf(1)/n))
        a = (random.random()*10**100 + j*random.random())
        r = nthroot(a, n)
        mp.dps += 4
        r1 = pow(a, mpf(1)/n)
        mp.dps -= 4
        assert r.ae(r1)
        assert mpc_ae(r, r1, eps)
        r = nthroot(a, -n)
        mp.dps += 4
        r1 = pow(a, -mpf(1)/n)
        mp.dps -= 4
        assert r.ae(r1)
        assert mpc_ae(r, r1, eps)
    mp.dps = 15
    assert nthroot(4, 1) == 4
    assert nthroot(4, 0) == 1
    assert nthroot(4, -1) == 0.25
    assert nthroot(inf, 1) == inf
    assert nthroot(inf, 2) == inf
    assert nthroot(inf, 3) == inf
    assert nthroot(inf, -1) == 0
    assert nthroot(inf, -2) == 0
    assert nthroot(inf, -3) == 0
    assert nthroot(j, 1) == j
    assert nthroot(j, 0) == 1
    assert nthroot(j, -1) == -j
    assert isnan(nthroot(nan, 1))
    assert isnan(nthroot(nan, 0))
    assert isnan(nthroot(nan, -1))
    assert isnan(nthroot(inf, 0))
    assert root(2,3) == nthroot(2,3)
    assert root(16,4,0) == 2
    assert root(16,4,1) == 2j
    assert root(16,4,2) == -2
    assert root(16,4,3) == -2j
    assert root(16,4,4) == 2
    assert root(-125,3,1) == -5

def test_issue_136():
    for dps in [20, 80]:
        mp.dps = dps
        r = nthroot(mpf('-1e-20'), 4)
        assert r.ae(mpf(10)**(-5) * (1 + j) * mpf(2)**(-0.5))
    mp.dps = 80
    assert nthroot('-1e-3', 4).ae(mpf(10)**(-3./4) * (1 + j)/sqrt(2))
    assert nthroot('-1e-6', 4).ae((1 + j)/(10 * sqrt(20)))
    # Check that this doesn't take eternity to compute
    mp.dps = 20
    assert nthroot('-1e100000000', 4).ae((1+j)*mpf('1e25000000')/sqrt(2))
    mp.dps = 15

def test_mpcfun_real_imag():
    mp.dps = 15
    x = mpf(0.3)
    y = mpf(0.4)
    assert exp(mpc(x,0)) == exp(x)
    assert exp(mpc(0,y)) == mpc(cos(y),sin(y))
    assert cos(mpc(x,0)) == cos(x)
    assert sin(mpc(x,0)) == sin(x)
    assert cos(mpc(0,y)) == cosh(y)
    assert sin(mpc(0,y)) == mpc(0,sinh(y))
    assert cospi(mpc(x,0)) == cospi(x)
    assert sinpi(mpc(x,0)) == sinpi(x)
    assert cospi(mpc(0,y)).ae(cosh(pi*y))
    assert sinpi(mpc(0,y)).ae(mpc(0,sinh(pi*y)))
    c, s = cospi_sinpi(mpc(x,0))
    assert c == cospi(x)
    assert s == sinpi(x)
    c, s = cospi_sinpi(mpc(0,y))
    assert c.ae(cosh(pi*y))
    assert s.ae(mpc(0,sinh(pi*y)))
    c, s = cos_sin(mpc(x,0))
    assert c == cos(x)
    assert s == sin(x)
    c, s = cos_sin(mpc(0,y))
    assert c == cosh(y)
    assert s == mpc(0,sinh(y))

def test_perturbation_rounding():
    mp.dps = 100
    a = pi/10**50
    b = -pi/10**50
    c = 1 + a
    d = 1 + b
    mp.dps = 15
    assert exp(a) == 1
    assert exp(a, rounding='c') > 1
    assert exp(b, rounding='c') == 1
    assert exp(a, rounding='f') == 1
    assert exp(b, rounding='f') < 1
    assert cos(a) == 1
    assert cos(a, rounding='c') == 1
    assert cos(b, rounding='c') == 1
    assert cos(a, rounding='f') < 1
    assert cos(b, rounding='f') < 1
    for f in [sin, atan, asinh, tanh]:
        assert f(a) == +a
        assert f(a, rounding='c') > a
        assert f(a, rounding='f') < a
        assert f(b) == +b
        assert f(b, rounding='c') > b
        assert f(b, rounding='f') < b
    for f in [asin, tan, sinh, atanh]:
        assert f(a) == +a
        assert f(b) == +b
        assert f(a, rounding='c') > a
        assert f(b, rounding='c') > b
        assert f(a, rounding='f') < a
        assert f(b, rounding='f') < b
    assert ln(c) == +a
    assert ln(d) == +b
    assert ln(c, rounding='c') > a
    assert ln(c, rounding='f') < a
    assert ln(d, rounding='c') > b
    assert ln(d, rounding='f') < b
    assert cosh(a) == 1
    assert cosh(b) == 1
    assert cosh(a, rounding='c') > 1
    assert cosh(b, rounding='c') > 1
    assert cosh(a, rounding='f') == 1
    assert cosh(b, rounding='f') == 1

def test_integer_parts():
    assert floor(3.2) == 3
    assert ceil(3.2) == 4
    assert floor(3.2+5j) == 3+5j
    assert ceil(3.2+5j) == 4+5j

def test_complex_parts():
    assert fabs('3') == 3
    assert fabs(3+4j) == 5
    assert re(3) == 3
    assert re(1+4j) == 1
    assert im(3) == 0
    assert im(1+4j) == 4
    assert conj(3) == 3
    assert conj(3+4j) == 3-4j
    assert mpf(3).conjugate() == 3

def test_cospi_sinpi():
    assert sinpi(0) == 0
    assert sinpi(0.5) == 1
    assert sinpi(1) == 0
    assert sinpi(1.5) == -1
    assert sinpi(2) == 0
    assert sinpi(2.5) == 1
    assert sinpi(-0.5) == -1
    assert cospi(0) == 1
    assert cospi(0.5) == 0
    assert cospi(1) == -1
    assert cospi(1.5) == 0
    assert cospi(2) == 1
    assert cospi(2.5) == 0
    assert cospi(-0.5) == 0
    assert cospi(100000000000.25).ae(sqrt(2)/2)
    a = cospi(2+3j)
    assert a.real.ae(cos((2+3j)*pi).real)
    assert a.imag == 0
    b = sinpi(2+3j)
    assert b.imag.ae(sin((2+3j)*pi).imag)
    assert b.real == 0
    mp.dps = 35
    x1 = mpf(10000) - mpf('1e-15')
    x2 = mpf(10000) + mpf('1e-15')
    x3 = mpf(10000.5) - mpf('1e-15')
    x4 = mpf(10000.5) + mpf('1e-15')
    x5 = mpf(10001) - mpf('1e-15')
    x6 = mpf(10001) + mpf('1e-15')
    x7 = mpf(10001.5) - mpf('1e-15')
    x8 = mpf(10001.5) + mpf('1e-15')
    mp.dps = 15
    M = 10**15
    assert (sinpi(x1)*M).ae(-pi)
    assert (sinpi(x2)*M).ae(pi)
    assert (cospi(x3)*M).ae(pi)
    assert (cospi(x4)*M).ae(-pi)
    assert (sinpi(x5)*M).ae(pi)
    assert (sinpi(x6)*M).ae(-pi)
    assert (cospi(x7)*M).ae(-pi)
    assert (cospi(x8)*M).ae(pi)
    assert 0.999 < cospi(x1, rounding='d') < 1
    assert 0.999 < cospi(x2, rounding='d') < 1
    assert 0.999 < sinpi(x3, rounding='d') < 1
    assert 0.999 < sinpi(x4, rounding='d') < 1
    assert -1 < cospi(x5, rounding='d') < -0.999
    assert -1 < cospi(x6, rounding='d') < -0.999
    assert -1 < sinpi(x7, rounding='d') < -0.999
    assert -1 < sinpi(x8, rounding='d') < -0.999
    assert (sinpi(1e-15)*M).ae(pi)
    assert (sinpi(-1e-15)*M).ae(-pi)
    assert cospi(1e-15) == 1
    assert cospi(1e-15, rounding='d') < 1

def test_expj():
    assert expj(0) == 1
    assert expj(1).ae(exp(j))
    assert expj(j).ae(exp(-1))
    assert expj(1+j).ae(exp(j*(1+j)))
    assert expjpi(0) == 1
    assert expjpi(1).ae(exp(j*pi))
    assert expjpi(j).ae(exp(-pi))
    assert expjpi(1+j).ae(exp(j*pi*(1+j)))
    assert expjpi(-10**15 * j).ae('2.22579818340535731e+1364376353841841')

def test_sinc():
    assert sinc(0) == sincpi(0) == 1
    assert sinc(inf) == sincpi(inf) == 0
    assert sinc(-inf) == sincpi(-inf) == 0
    assert sinc(2).ae(0.45464871341284084770)
    assert sinc(2+3j).ae(0.4463290318402435457-2.7539470277436474940j)
    assert sincpi(2) == 0
    assert sincpi(1.5).ae(-0.212206590789193781)

def test_fibonacci():
    mp.dps = 15
    assert [fibonacci(n) for n in range(-5, 10)] == \
        [5, -3, 2, -1, 1, 0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    assert fib(2.5).ae(1.4893065462657091)
    assert fib(3+4j).ae(-5248.51130728372 - 14195.962288353j)
    assert fib(1000).ae(4.3466557686937455e+208)
    assert str(fib(10**100)) == '6.24499112864607e+2089876402499787337692720892375554168224592399182109535392875613974104853496745963277658556235103534'
    mp.dps = 2100
    a = fib(10000)
    assert a % 10**10 == 9947366875
    mp.dps = 15
    assert fibonacci(inf) == inf
    assert fib(3+0j) == 2

def test_call_with_dps():
    mp.dps = 15
    assert abs(exp(1, dps=30)-e(dps=35)) < 1e-29

def test_tanh():
    mp.dps = 15
    assert tanh(0) == 0
    assert tanh(inf) == 1
    assert tanh(-inf) == -1
    assert isnan(tanh(nan))
    assert tanh(mpc('inf', '0')) == 1

def test_atanh():
    mp.dps = 15
    assert atanh(0) == 0
    assert atanh(0.5).ae(0.54930614433405484570)
    assert atanh(-0.5).ae(-0.54930614433405484570)
    assert atanh(1) == inf
    assert atanh(-1) == -inf
    assert isnan(atanh(nan))
    assert isinstance(atanh(1), mpf)
    assert isinstance(atanh(-1), mpf)
    # Limits at infinity
    jpi2 = j*pi/2
    assert atanh(inf).ae(-jpi2)
    assert atanh(-inf).ae(jpi2)
    assert atanh(mpc(inf,-1)).ae(-jpi2)
    assert atanh(mpc(inf,0)).ae(-jpi2)
    assert atanh(mpc(inf,1)).ae(jpi2)
    assert atanh(mpc(1,inf)).ae(jpi2)
    assert atanh(mpc(0,inf)).ae(jpi2)
    assert atanh(mpc(-1,inf)).ae(jpi2)
    assert atanh(mpc(-inf,1)).ae(jpi2)
    assert atanh(mpc(-inf,0)).ae(jpi2)
    assert atanh(mpc(-inf,-1)).ae(-jpi2)
    assert atanh(mpc(-1,-inf)).ae(-jpi2)
    assert atanh(mpc(0,-inf)).ae(-jpi2)
    assert atanh(mpc(1,-inf)).ae(-jpi2)

def test_expm1():
    mp.dps = 15
    assert expm1(0) == 0
    assert expm1(3).ae(exp(3)-1)
    assert expm1(inf) == inf
    assert expm1(1e-50).ae(1e-50)
    assert (expm1(1e-10)*1e10).ae(1.00000000005)

def test_log1p():
    mp.dps = 15
    assert log1p(0) == 0
    assert log1p(3).ae(log(1+3))
    assert log1p(inf) == inf
    assert log1p(1e-50).ae(1e-50)
    assert (log1p(1e-10)*1e10).ae(0.99999999995)

def test_powm1():
    mp.dps = 15
    assert powm1(2,3) == 7
    assert powm1(-1,2) == 0
    assert powm1(-1,0) == 0
    assert powm1(-2,0) == 0
    assert powm1(3+4j,0) == 0
    assert powm1(0,1) == -1
    assert powm1(0,0) == 0
    assert powm1(1,0) == 0
    assert powm1(1,2) == 0
    assert powm1(1,3+4j) == 0
    assert powm1(1,5) == 0
    assert powm1(j,4) == 0
    assert powm1(-j,4) == 0
    assert (powm1(2,1e-100)*1e100).ae(ln2)
    assert powm1(2,'1e-100000000000') != 0
    assert (powm1(fadd(1,1e-100,exact=True), 5)*1e100).ae(5)

def test_unitroots():
    assert unitroots(1) == [1]
    assert unitroots(2) == [1, -1]
    a, b, c = unitroots(3)
    assert a == 1
    assert b.ae(-0.5 + 0.86602540378443864676j)
    assert c.ae(-0.5 - 0.86602540378443864676j)
    assert unitroots(1, primitive=True) == [1]
    assert unitroots(2, primitive=True) == [-1]
    assert unitroots(3, primitive=True) == unitroots(3)[1:]
    assert unitroots(4, primitive=True) == [j, -j]
    assert len(unitroots(17, primitive=True)) == 16
    assert len(unitroots(16, primitive=True)) == 8

def test_cyclotomic():
    mp.dps = 15
    assert [cyclotomic(n,1) for n in range(31)] == [1,0,2,3,2,5,1,7,2,3,1,11,1,13,1,1,2,17,1,19,1,1,1,23,1,5,1,3,1,29,1]
    assert [cyclotomic(n,-1) for n in range(31)] == [1,-2,0,1,2,1,3,1,2,1,5,1,1,1,7,1,2,1,3,1,1,1,11,1,1,1,13,1,1,1,1]
    assert [cyclotomic(n,j) for n in range(21)] == [1,-1+j,1+j,j,0,1,-j,j,2,-j,1,j,3,1,-j,1,2,1,j,j,5]
    assert [cyclotomic(n,-j) for n in range(21)] == [1,-1-j,1-j,-j,0,1,j,-j,2,j,1,-j,3,1,j,1,2,1,-j,-j,5]
    assert cyclotomic(1624,j) == 1
    assert cyclotomic(33600,j) == 1
    u = sqrt(j, prec=500)
    assert cyclotomic(8, u).ae(0)
    assert cyclotomic(30, u).ae(5.8284271247461900976)
    assert cyclotomic(2040, u).ae(1)
    assert cyclotomic(0,2.5) == 1
    assert cyclotomic(1,2.5) == 2.5-1
    assert cyclotomic(2,2.5) == 2.5+1
    assert cyclotomic(3,2.5) == 2.5**2 + 2.5 + 1
    assert cyclotomic(7,2.5) == 406.234375

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\categories\tests\test_drawing.py ===
from sympy.categories.diagram_drawing import _GrowableGrid, ArrowStringDescription
from sympy.categories import (DiagramGrid, Object, NamedMorphism,
                              Diagram, XypicDiagramDrawer, xypic_draw_diagram)
from sympy.sets.sets import FiniteSet


def test_GrowableGrid():
    grid = _GrowableGrid(1, 2)

    # Check dimensions.
    assert grid.width == 1
    assert grid.height == 2

    # Check initialization of elements.
    assert grid[0, 0] is None
    assert grid[1, 0] is None

    # Check assignment to elements.
    grid[0, 0] = 1
    grid[1, 0] = "two"

    assert grid[0, 0] == 1
    assert grid[1, 0] == "two"

    # Check appending a row.
    grid.append_row()

    assert grid.width == 1
    assert grid.height == 3

    assert grid[0, 0] == 1
    assert grid[1, 0] == "two"
    assert grid[2, 0] is None

    # Check appending a column.
    grid.append_column()
    assert grid.width == 2
    assert grid.height == 3

    assert grid[0, 0] == 1
    assert grid[1, 0] == "two"
    assert grid[2, 0] is None

    assert grid[0, 1] is None
    assert grid[1, 1] is None
    assert grid[2, 1] is None

    grid = _GrowableGrid(1, 2)
    grid[0, 0] = 1
    grid[1, 0] = "two"

    # Check prepending a row.
    grid.prepend_row()
    assert grid.width == 1
    assert grid.height == 3

    assert grid[0, 0] is None
    assert grid[1, 0] == 1
    assert grid[2, 0] == "two"

    # Check prepending a column.
    grid.prepend_column()
    assert grid.width == 2
    assert grid.height == 3

    assert grid[0, 0] is None
    assert grid[1, 0] is None
    assert grid[2, 0] is None

    assert grid[0, 1] is None
    assert grid[1, 1] == 1
    assert grid[2, 1] == "two"


def test_DiagramGrid():
    # Set up some objects and morphisms.
    A = Object("A")
    B = Object("B")
    C = Object("C")
    D = Object("D")
    E = Object("E")

    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    h = NamedMorphism(D, A, "h")
    k = NamedMorphism(D, B, "k")

    # A one-morphism diagram.
    d = Diagram([f])
    grid = DiagramGrid(d)

    assert grid.width == 2
    assert grid.height == 1
    assert grid[0, 0] == A
    assert grid[0, 1] == B
    assert grid.morphisms == {f: FiniteSet()}

    # A triangle.
    d = Diagram([f, g], {g * f: "unique"})
    grid = DiagramGrid(d)

    assert grid.width == 2
    assert grid.height == 2
    assert grid[0, 0] == A
    assert grid[0, 1] == B
    assert grid[1, 0] == C
    assert grid[1, 1] is None
    assert grid.morphisms == {f: FiniteSet(), g: FiniteSet(),
                              g * f: FiniteSet("unique")}

    # A triangle with a "loop" morphism.
    l_A = NamedMorphism(A, A, "l_A")
    d = Diagram([f, g, l_A])
    grid = DiagramGrid(d)

    assert grid.width == 2
    assert grid.height == 2
    assert grid[0, 0] == A
    assert grid[0, 1] == B
    assert grid[1, 0] is None
    assert grid[1, 1] == C
    assert grid.morphisms == {f: FiniteSet(), g: FiniteSet(), l_A: FiniteSet()}

    # A simple diagram.
    d = Diagram([f, g, h, k])
    grid = DiagramGrid(d)

    assert grid.width == 3
    assert grid.height == 2
    assert grid[0, 0] == A
    assert grid[0, 1] == B
    assert grid[0, 2] == D
    assert grid[1, 0] is None
    assert grid[1, 1] == C
    assert grid[1, 2] is None
    assert grid.morphisms == {f: FiniteSet(), g: FiniteSet(), h: FiniteSet(),
                              k: FiniteSet()}

    assert str(grid) == '[[Object("A"), Object("B"), Object("D")], ' \
        '[None, Object("C"), None]]'

    # A chain of morphisms.
    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    h = NamedMorphism(C, D, "h")
    k = NamedMorphism(D, E, "k")
    d = Diagram([f, g, h, k])
    grid = DiagramGrid(d)

    assert grid.width == 3
    assert grid.height == 3
    assert grid[0, 0] == A
    assert grid[0, 1] == B
    assert grid[0, 2] is None
    assert grid[1, 0] is None
    assert grid[1, 1] == C
    assert grid[1, 2] == D
    assert grid[2, 0] is None
    assert grid[2, 1] is None
    assert grid[2, 2] == E
    assert grid.morphisms == {f: FiniteSet(), g: FiniteSet(), h: FiniteSet(),
                              k: FiniteSet()}

    # A square.
    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, D, "g")
    h = NamedMorphism(A, C, "h")
    k = NamedMorphism(C, D, "k")
    d = Diagram([f, g, h, k])
    grid = DiagramGrid(d)

    assert grid.width == 2
    assert grid.height == 2
    assert grid[0, 0] == A
    assert grid[0, 1] == B
    assert grid[1, 0] == C
    assert grid[1, 1] == D
    assert grid.morphisms == {f: FiniteSet(), g: FiniteSet(), h: FiniteSet(),
                              k: FiniteSet()}

    # A strange diagram which resulted from a typo when creating a
    # test for five lemma, but which allowed to stop one extra problem
    # in the algorithm.
    A = Object("A")
    B = Object("B")
    C = Object("C")
    D = Object("D")
    E = Object("E")
    A_ = Object("A'")
    B_ = Object("B'")
    C_ = Object("C'")
    D_ = Object("D'")
    E_ = Object("E'")

    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    h = NamedMorphism(C, D, "h")
    i = NamedMorphism(D, E, "i")

    # These 4 morphisms should be between primed objects.
    j = NamedMorphism(A, B, "j")
    k = NamedMorphism(B, C, "k")
    l = NamedMorphism(C, D, "l")
    m = NamedMorphism(D, E, "m")

    o = NamedMorphism(A, A_, "o")
    p = NamedMorphism(B, B_, "p")
    q = NamedMorphism(C, C_, "q")
    r = NamedMorphism(D, D_, "r")
    s = NamedMorphism(E, E_, "s")

    d = Diagram([f, g, h, i, j, k, l, m, o, p, q, r, s])
    grid = DiagramGrid(d)

    assert grid.width == 3
    assert grid.height == 4
    assert grid[0, 0] is None
    assert grid[0, 1] == A
    assert grid[0, 2] == A_
    assert grid[1, 0] == C
    assert grid[1, 1] == B
    assert grid[1, 2] == B_
    assert grid[2, 0] == C_
    assert grid[2, 1] == D
    assert grid[2, 2] == D_
    assert grid[3, 0] is None
    assert grid[3, 1] == E
    assert grid[3, 2] == E_

    morphisms = {}
    for m in [f, g, h, i, j, k, l, m, o, p, q, r, s]:
        morphisms[m] = FiniteSet()
    assert grid.morphisms == morphisms

    # A cube.
    A1 = Object("A1")
    A2 = Object("A2")
    A3 = Object("A3")
    A4 = Object("A4")
    A5 = Object("A5")
    A6 = Object("A6")
    A7 = Object("A7")
    A8 = Object("A8")

    # The top face of the cube.
    f1 = NamedMorphism(A1, A2, "f1")
    f2 = NamedMorphism(A1, A3, "f2")
    f3 = NamedMorphism(A2, A4, "f3")
    f4 = NamedMorphism(A3, A4, "f3")

    # The bottom face of the cube.
    f5 = NamedMorphism(A5, A6, "f5")
    f6 = NamedMorphism(A5, A7, "f6")
    f7 = NamedMorphism(A6, A8, "f7")
    f8 = NamedMorphism(A7, A8, "f8")

    # The remaining morphisms.
    f9 = NamedMorphism(A1, A5, "f9")
    f10 = NamedMorphism(A2, A6, "f10")
    f11 = NamedMorphism(A3, A7, "f11")
    f12 = NamedMorphism(A4, A8, "f11")

    d = Diagram([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12])
    grid = DiagramGrid(d)

    assert grid.width == 4
    assert grid.height == 3
    assert grid[0, 0] is None
    assert grid[0, 1] == A5
    assert grid[0, 2] == A6
    assert grid[0, 3] is None
    assert grid[1, 0] is None
    assert grid[1, 1] == A1
    assert grid[1, 2] == A2
    assert grid[1, 3] is None
    assert grid[2, 0] == A7
    assert grid[2, 1] == A3
    assert grid[2, 2] == A4
    assert grid[2, 3] == A8

    morphisms = {}
    for m in [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]:
        morphisms[m] = FiniteSet()
    assert grid.morphisms == morphisms

    # A line diagram.
    A = Object("A")
    B = Object("B")
    C = Object("C")
    D = Object("D")
    E = Object("E")

    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    h = NamedMorphism(C, D, "h")
    i = NamedMorphism(D, E, "i")
    d = Diagram([f, g, h, i])
    grid = DiagramGrid(d, layout="sequential")

    assert grid.width == 5
    assert grid.height == 1
    assert grid[0, 0] == A
    assert grid[0, 1] == B
    assert grid[0, 2] == C
    assert grid[0, 3] == D
    assert grid[0, 4] == E
    assert grid.morphisms == {f: FiniteSet(), g: FiniteSet(), h: FiniteSet(),
                              i: FiniteSet()}

    # Test the transposed version.
    grid = DiagramGrid(d, layout="sequential", transpose=True)

    assert grid.width == 1
    assert grid.height == 5
    assert grid[0, 0] == A
    assert grid[1, 0] == B
    assert grid[2, 0] == C
    assert grid[3, 0] == D
    assert grid[4, 0] == E
    assert grid.morphisms == {f: FiniteSet(), g: FiniteSet(), h: FiniteSet(),
                              i: FiniteSet()}

    # A pullback.
    m1 = NamedMorphism(A, B, "m1")
    m2 = NamedMorphism(A, C, "m2")
    s1 = NamedMorphism(B, D, "s1")
    s2 = NamedMorphism(C, D, "s2")
    f1 = NamedMorphism(E, B, "f1")
    f2 = NamedMorphism(E, C, "f2")
    g = NamedMorphism(E, A, "g")

    d = Diagram([m1, m2, s1, s2, f1, f2], {g: "unique"})
    grid = DiagramGrid(d)

    assert grid.width == 3
    assert grid.height == 2
    assert grid[0, 0] == A
    assert grid[0, 1] == B
    assert grid[0, 2] == E
    assert grid[1, 0] == C
    assert grid[1, 1] == D
    assert grid[1, 2] is None

    morphisms = {g: FiniteSet("unique")}
    for m in [m1, m2, s1, s2, f1, f2]:
        morphisms[m] = FiniteSet()
    assert grid.morphisms == morphisms

    # Test the pullback with sequential layout, just for stress
    # testing.
    grid = DiagramGrid(d, layout="sequential")

    assert grid.width == 5
    assert grid.height == 1
    assert grid[0, 0] == D
    assert grid[0, 1] == B
    assert grid[0, 2] == A
    assert grid[0, 3] == C
    assert grid[0, 4] == E
    assert grid.morphisms == morphisms

    # Test a pullback with object grouping.
    grid = DiagramGrid(d, groups=FiniteSet(E, FiniteSet(A, B, C, D)))

    assert grid.width == 3
    assert grid.height == 2
    assert grid[0, 0] == E
    assert grid[0, 1] == A
    assert grid[0, 2] == B
    assert grid[1, 0] is None
    assert grid[1, 1] == C
    assert grid[1, 2] == D
    assert grid.morphisms == morphisms

    # Five lemma, actually.
    A = Object("A")
    B = Object("B")
    C = Object("C")
    D = Object("D")
    E = Object("E")
    A_ = Object("A'")
    B_ = Object("B'")
    C_ = Object("C'")
    D_ = Object("D'")
    E_ = Object("E'")

    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    h = NamedMorphism(C, D, "h")
    i = NamedMorphism(D, E, "i")

    j = NamedMorphism(A_, B_, "j")
    k = NamedMorphism(B_, C_, "k")
    l = NamedMorphism(C_, D_, "l")
    m = NamedMorphism(D_, E_, "m")

    o = NamedMorphism(A, A_, "o")
    p = NamedMorphism(B, B_, "p")
    q = NamedMorphism(C, C_, "q")
    r = NamedMorphism(D, D_, "r")
    s = NamedMorphism(E, E_, "s")

    d = Diagram([f, g, h, i, j, k, l, m, o, p, q, r, s])
    grid = DiagramGrid(d)

    assert grid.width == 5
    assert grid.height == 3
    assert grid[0, 0] is None
    assert grid[0, 1] == A
    assert grid[0, 2] == A_
    assert grid[0, 3] is None
    assert grid[0, 4] is None
    assert grid[1, 0] == C
    assert grid[1, 1] == B
    assert grid[1, 2] == B_
    assert grid[1, 3] == C_
    assert grid[1, 4] is None
    assert grid[2, 0] == D
    assert grid[2, 1] == E
    assert grid[2, 2] is None
    assert grid[2, 3] == D_
    assert grid[2, 4] == E_

    morphisms = {}
    for m in [f, g, h, i, j, k, l, m, o, p, q, r, s]:
        morphisms[m] = FiniteSet()
    assert grid.morphisms == morphisms

    # Test the five lemma with object grouping.
    grid = DiagramGrid(d, FiniteSet(
        FiniteSet(A, B, C, D, E), FiniteSet(A_, B_, C_, D_, E_)))

    assert grid.width == 6
    assert grid.height == 3
    assert grid[0, 0] == A
    assert grid[0, 1] == B
    assert grid[0, 2] is None
    assert grid[0, 3] == A_
    assert grid[0, 4] == B_
    assert grid[0, 5] is None
    assert grid[1, 0] is None
    assert grid[1, 1] == C
    assert grid[1, 2] == D
    assert grid[1, 3] is None
    assert grid[1, 4] == C_
    assert grid[1, 5] == D_
    assert grid[2, 0] is None
    assert grid[2, 1] is None
    assert grid[2, 2] == E
    assert grid[2, 3] is None
    assert grid[2, 4] is None
    assert grid[2, 5] == E_
    assert grid.morphisms == morphisms

    # Test the five lemma with object grouping, but mixing containers
    # to represent groups.
    grid = DiagramGrid(d, [(A, B, C, D, E), {A_, B_, C_, D_, E_}])

    assert grid.width == 6
    assert grid.height == 3
    assert grid[0, 0] == A
    assert grid[0, 1] == B
    assert grid[0, 2] is None
    assert grid[0, 3] == A_
    assert grid[0, 4] == B_
    assert grid[0, 5] is None
    assert grid[1, 0] is None
    assert grid[1, 1] == C
    assert grid[1, 2] == D
    assert grid[1, 3] is None
    assert grid[1, 4] == C_
    assert grid[1, 5] == D_
    assert grid[2, 0] is None
    assert grid[2, 1] is None
    assert grid[2, 2] == E
    assert grid[2, 3] is None
    assert grid[2, 4] is None
    assert grid[2, 5] == E_
    assert grid.morphisms == morphisms

    # Test the five lemma with object grouping and hints.
    grid = DiagramGrid(d, {
        FiniteSet(A, B, C, D, E): {"layout": "sequential",
                                   "transpose": True},
        FiniteSet(A_, B_, C_, D_, E_): {"layout": "sequential",
                                        "transpose": True}},
        transpose=True)

    assert grid.width == 5
    assert grid.height == 2
    assert grid[0, 0] == A
    assert grid[0, 1] == B
    assert grid[0, 2] == C
    assert grid[0, 3] == D
    assert grid[0, 4] == E
    assert grid[1, 0] == A_
    assert grid[1, 1] == B_
    assert grid[1, 2] == C_
    assert grid[1, 3] == D_
    assert grid[1, 4] == E_
    assert grid.morphisms == morphisms

    # A two-triangle disconnected diagram.
    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    f_ = NamedMorphism(A_, B_, "f")
    g_ = NamedMorphism(B_, C_, "g")
    d = Diagram([f, g, f_, g_], {g * f: "unique", g_ * f_: "unique"})
    grid = DiagramGrid(d)

    assert grid.width == 4
    assert grid.height == 2
    assert grid[0, 0] == A
    assert grid[0, 1] == B
    assert grid[0, 2] == A_
    assert grid[0, 3] == B_
    assert grid[1, 0] == C
    assert grid[1, 1] is None
    assert grid[1, 2] == C_
    assert grid[1, 3] is None
    assert grid.morphisms == {f: FiniteSet(), g: FiniteSet(), f_: FiniteSet(),
                              g_: FiniteSet(), g * f: FiniteSet("unique"),
                              g_ * f_: FiniteSet("unique")}

    # A two-morphism disconnected diagram.
    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(C, D, "g")
    d = Diagram([f, g])
    grid = DiagramGrid(d)

    assert grid.width == 4
    assert grid.height == 1
    assert grid[0, 0] == A
    assert grid[0, 1] == B
    assert grid[0, 2] == C
    assert grid[0, 3] == D
    assert grid.morphisms == {f: FiniteSet(), g: FiniteSet()}

    # Test a one-object diagram.
    f = NamedMorphism(A, A, "f")
    d = Diagram([f])
    grid = DiagramGrid(d)

    assert grid.width == 1
    assert grid.height == 1
    assert grid[0, 0] == A

    # Test a two-object disconnected diagram.
    g = NamedMorphism(B, B, "g")
    d = Diagram([f, g])
    grid = DiagramGrid(d)

    assert grid.width == 2
    assert grid.height == 1
    assert grid[0, 0] == A
    assert grid[0, 1] == B


def test_DiagramGrid_pseudopod():
    # Test a diagram in which even growing a pseudopod does not
    # eventually help.
    A = Object("A")
    B = Object("B")
    C = Object("C")
    D = Object("D")
    E = Object("E")
    F = Object("F")
    A_ = Object("A'")
    B_ = Object("B'")
    C_ = Object("C'")
    D_ = Object("D'")
    E_ = Object("E'")

    f1 = NamedMorphism(A, B, "f1")
    f2 = NamedMorphism(A, C, "f2")
    f3 = NamedMorphism(A, D, "f3")
    f4 = NamedMorphism(A, E, "f4")
    f5 = NamedMorphism(A, A_, "f5")
    f6 = NamedMorphism(A, B_, "f6")
    f7 = NamedMorphism(A, C_, "f7")
    f8 = NamedMorphism(A, D_, "f8")
    f9 = NamedMorphism(A, E_, "f9")
    f10 = NamedMorphism(A, F, "f10")
    d = Diagram([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10])
    grid = DiagramGrid(d)

    assert grid.width == 5
    assert grid.height == 3
    assert grid[0, 0] == E
    assert grid[0, 1] == C
    assert grid[0, 2] == C_
    assert grid[0, 3] == E_
    assert grid[0, 4] == F
    assert grid[1, 0] == D
    assert grid[1, 1] == A
    assert grid[1, 2] == A_
    assert grid[1, 3] is None
    assert grid[1, 4] is None
    assert grid[2, 0] == D_
    assert grid[2, 1] == B
    assert grid[2, 2] == B_
    assert grid[2, 3] is None
    assert grid[2, 4] is None

    morphisms = {}
    for f in [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]:
        morphisms[f] = FiniteSet()
    assert grid.morphisms == morphisms


def test_ArrowStringDescription():
    astr = ArrowStringDescription("cm", "", None, "", "", "d", "r", "_", "f")
    assert str(astr) == "\\ar[dr]_{f}"

    astr = ArrowStringDescription("cm", "", 12, "", "", "d", "r", "_", "f")
    assert str(astr) == "\\ar[dr]_{f}"

    astr = ArrowStringDescription("cm", "^", 12, "", "", "d", "r", "_", "f")
    assert str(astr) == "\\ar@/^12cm/[dr]_{f}"

    astr = ArrowStringDescription("cm", "", 12, "r", "", "d", "r", "_", "f")
    assert str(astr) == "\\ar[dr]_{f}"

    astr = ArrowStringDescription("cm", "", 12, "r", "u", "d", "r", "_", "f")
    assert str(astr) == "\\ar@(r,u)[dr]_{f}"

    astr = ArrowStringDescription("cm", "", 12, "r", "u", "d", "r", "_", "f")
    assert str(astr) == "\\ar@(r,u)[dr]_{f}"

    astr = ArrowStringDescription("cm", "", 12, "r", "u", "d", "r", "_", "f")
    astr.arrow_style = "{-->}"
    assert str(astr) == "\\ar@(r,u)@{-->}[dr]_{f}"

    astr = ArrowStringDescription("cm", "_", 12, "", "", "d", "r", "_", "f")
    astr.arrow_style = "{-->}"
    assert str(astr) == "\\ar@/_12cm/@{-->}[dr]_{f}"


def test_XypicDiagramDrawer_line():
    # A linear diagram.
    A = Object("A")
    B = Object("B")
    C = Object("C")
    D = Object("D")
    E = Object("E")

    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    h = NamedMorphism(C, D, "h")
    i = NamedMorphism(D, E, "i")
    d = Diagram([f, g, h, i])
    grid = DiagramGrid(d, layout="sequential")
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "A \\ar[r]^{f} & B \\ar[r]^{g} & C \\ar[r]^{h} & D \\ar[r]^{i} & E \n" \
        "}\n"

    # The same diagram, transposed.
    grid = DiagramGrid(d, layout="sequential", transpose=True)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "A \\ar[d]^{f} \\\\\n" \
        "B \\ar[d]^{g} \\\\\n" \
        "C \\ar[d]^{h} \\\\\n" \
        "D \\ar[d]^{i} \\\\\n" \
        "E \n" \
        "}\n"


def test_XypicDiagramDrawer_triangle():
    # A triangle diagram.
    A = Object("A")
    B = Object("B")
    C = Object("C")
    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")

    d = Diagram([f, g], {g * f: "unique"})
    grid = DiagramGrid(d)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "A \\ar[d]_{g\\circ f} \\ar[r]^{f} & B \\ar[ld]^{g} \\\\\n" \
        "C & \n" \
        "}\n"

    # The same diagram, transposed.
    grid = DiagramGrid(d, transpose=True)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "A \\ar[r]^{g\\circ f} \\ar[d]_{f} & C \\\\\n" \
        "B \\ar[ru]_{g} & \n" \
        "}\n"

    # The same diagram, with a masked morphism.
    assert drawer.draw(d, grid, masked=[g]) == "\\xymatrix{\n" \
        "A \\ar[r]^{g\\circ f} \\ar[d]_{f} & C \\\\\n" \
        "B & \n" \
        "}\n"

    # The same diagram with a formatter for "unique".
    def formatter(astr):
        astr.label = "\\exists !" + astr.label
        astr.arrow_style = "{-->}"

    drawer.arrow_formatters["unique"] = formatter
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "A \\ar@{-->}[r]^{\\exists !g\\circ f} \\ar[d]_{f} & C \\\\\n" \
        "B \\ar[ru]_{g} & \n" \
        "}\n"

    # The same diagram with a default formatter.
    def default_formatter(astr):
        astr.label_displacement = "(0.45)"

    drawer.default_arrow_formatter = default_formatter
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "A \\ar@{-->}[r]^(0.45){\\exists !g\\circ f} \\ar[d]_(0.45){f} & C \\\\\n" \
        "B \\ar[ru]_(0.45){g} & \n" \
        "}\n"

    # A triangle diagram with a lot of morphisms between the same
    # objects.
    f1 = NamedMorphism(B, A, "f1")
    f2 = NamedMorphism(A, B, "f2")
    g1 = NamedMorphism(C, B, "g1")
    g2 = NamedMorphism(B, C, "g2")
    d = Diagram([f, f1, f2, g, g1, g2], {f1 * g1: "unique", g2 * f2: "unique"})

    grid = DiagramGrid(d, transpose=True)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid, masked=[f1*g1*g2*f2, g2*f2*f1*g1]) == \
        "\\xymatrix{\n" \
        "A \\ar[r]^{g_{2}\\circ f_{2}} \\ar[d]_{f} \\ar@/^3mm/[d]^{f_{2}} " \
        "& C \\ar@/^3mm/[l]^{f_{1}\\circ g_{1}} \\ar@/^3mm/[ld]^{g_{1}} \\\\\n" \
        "B \\ar@/^3mm/[u]^{f_{1}} \\ar[ru]_{g} \\ar@/^3mm/[ru]^{g_{2}} & \n" \
        "}\n"


def test_XypicDiagramDrawer_cube():
    # A cube diagram.
    A1 = Object("A1")
    A2 = Object("A2")
    A3 = Object("A3")
    A4 = Object("A4")
    A5 = Object("A5")
    A6 = Object("A6")
    A7 = Object("A7")
    A8 = Object("A8")

    # The top face of the cube.
    f1 = NamedMorphism(A1, A2, "f1")
    f2 = NamedMorphism(A1, A3, "f2")
    f3 = NamedMorphism(A2, A4, "f3")
    f4 = NamedMorphism(A3, A4, "f3")

    # The bottom face of the cube.
    f5 = NamedMorphism(A5, A6, "f5")
    f6 = NamedMorphism(A5, A7, "f6")
    f7 = NamedMorphism(A6, A8, "f7")
    f8 = NamedMorphism(A7, A8, "f8")

    # The remaining morphisms.
    f9 = NamedMorphism(A1, A5, "f9")
    f10 = NamedMorphism(A2, A6, "f10")
    f11 = NamedMorphism(A3, A7, "f11")
    f12 = NamedMorphism(A4, A8, "f11")

    d = Diagram([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12])
    grid = DiagramGrid(d)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "& A_{5} \\ar[r]^{f_{5}} \\ar[ldd]_{f_{6}} & A_{6} \\ar[rdd]^{f_{7}} " \
        "& \\\\\n" \
        "& A_{1} \\ar[r]^{f_{1}} \\ar[d]^{f_{2}} \\ar[u]^{f_{9}} & A_{2} " \
        "\\ar[d]^{f_{3}} \\ar[u]_{f_{10}} & \\\\\n" \
        "A_{7} \\ar@/_3mm/[rrr]_{f_{8}} & A_{3} \\ar[r]^{f_{3}} \\ar[l]_{f_{11}} " \
        "& A_{4} \\ar[r]^{f_{11}} & A_{8} \n" \
        "}\n"

    # The same diagram, transposed.
    grid = DiagramGrid(d, transpose=True)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "& & A_{7} \\ar@/^3mm/[ddd]^{f_{8}} \\\\\n" \
        "A_{5} \\ar[d]_{f_{5}} \\ar[rru]^{f_{6}} & A_{1} \\ar[d]^{f_{1}} " \
        "\\ar[r]^{f_{2}} \\ar[l]^{f_{9}} & A_{3} \\ar[d]_{f_{3}} " \
        "\\ar[u]^{f_{11}} \\\\\n" \
        "A_{6} \\ar[rrd]_{f_{7}} & A_{2} \\ar[r]^{f_{3}} \\ar[l]^{f_{10}} " \
        "& A_{4} \\ar[d]_{f_{11}} \\\\\n" \
        "& & A_{8} \n" \
        "}\n"


def test_XypicDiagramDrawer_curved_and_loops():
    # A simple diagram, with a curved arrow.
    A = Object("A")
    B = Object("B")
    C = Object("C")
    D = Object("D")

    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    h = NamedMorphism(D, A, "h")
    k = NamedMorphism(D, B, "k")
    d = Diagram([f, g, h, k])
    grid = DiagramGrid(d)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "A \\ar[r]_{f} & B \\ar[d]^{g} & D \\ar[l]^{k} \\ar@/_3mm/[ll]_{h} \\\\\n" \
        "& C & \n" \
        "}\n"

    # The same diagram, transposed.
    grid = DiagramGrid(d, transpose=True)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "A \\ar[d]^{f} & \\\\\n" \
        "B \\ar[r]^{g} & C \\\\\n" \
        "D \\ar[u]_{k} \\ar@/^3mm/[uu]^{h} & \n" \
        "}\n"

    # The same diagram, larger and rotated.
    assert drawer.draw(d, grid, diagram_format="@+1cm@dr") == \
        "\\xymatrix@+1cm@dr{\n" \
        "A \\ar[d]^{f} & \\\\\n" \
        "B \\ar[r]^{g} & C \\\\\n" \
        "D \\ar[u]_{k} \\ar@/^3mm/[uu]^{h} & \n" \
        "}\n"

    # A simple diagram with three curved arrows.
    h1 = NamedMorphism(D, A, "h1")
    h2 = NamedMorphism(A, D, "h2")
    k = NamedMorphism(D, B, "k")
    d = Diagram([f, g, h, k, h1, h2])
    grid = DiagramGrid(d)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "A \\ar[r]_{f} \\ar@/^3mm/[rr]^{h_{2}} & B \\ar[d]^{g} & D \\ar[l]^{k} " \
        "\\ar@/_7mm/[ll]_{h} \\ar@/_11mm/[ll]_{h_{1}} \\\\\n" \
        "& C & \n" \
        "}\n"

    # The same diagram, transposed.
    grid = DiagramGrid(d, transpose=True)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "A \\ar[d]^{f} \\ar@/_3mm/[dd]_{h_{2}} & \\\\\n" \
        "B \\ar[r]^{g} & C \\\\\n" \
        "D \\ar[u]_{k} \\ar@/^7mm/[uu]^{h} \\ar@/^11mm/[uu]^{h_{1}} & \n" \
        "}\n"

    # The same diagram, with "loop" morphisms.
    l_A = NamedMorphism(A, A, "l_A")
    l_D = NamedMorphism(D, D, "l_D")
    l_C = NamedMorphism(C, C, "l_C")
    d = Diagram([f, g, h, k, h1, h2, l_A, l_D, l_C])
    grid = DiagramGrid(d)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "A \\ar[r]_{f} \\ar@/^3mm/[rr]^{h_{2}} \\ar@(u,l)[]^{l_{A}} " \
        "& B \\ar[d]^{g} & D \\ar[l]^{k} \\ar@/_7mm/[ll]_{h} " \
        "\\ar@/_11mm/[ll]_{h_{1}} \\ar@(r,u)[]^{l_{D}} \\\\\n" \
        "& C \\ar@(l,d)[]^{l_{C}} & \n" \
        "}\n"

    # The same diagram with "loop" morphisms, transposed.
    grid = DiagramGrid(d, transpose=True)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "A \\ar[d]^{f} \\ar@/_3mm/[dd]_{h_{2}} \\ar@(r,u)[]^{l_{A}} & \\\\\n" \
        "B \\ar[r]^{g} & C \\ar@(r,u)[]^{l_{C}} \\\\\n" \
        "D \\ar[u]_{k} \\ar@/^7mm/[uu]^{h} \\ar@/^11mm/[uu]^{h_{1}} " \
        "\\ar@(l,d)[]^{l_{D}} & \n" \
        "}\n"

    # The same diagram with two "loop" morphisms per object.
    l_A_ = NamedMorphism(A, A, "n_A")
    l_D_ = NamedMorphism(D, D, "n_D")
    l_C_ = NamedMorphism(C, C, "n_C")
    d = Diagram([f, g, h, k, h1, h2, l_A, l_D, l_C, l_A_, l_D_, l_C_])
    grid = DiagramGrid(d)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "A \\ar[r]_{f} \\ar@/^3mm/[rr]^{h_{2}} \\ar@(u,l)[]^{l_{A}} " \
        "\\ar@/^3mm/@(l,d)[]^{n_{A}} & B \\ar[d]^{g} & D \\ar[l]^{k} " \
        "\\ar@/_7mm/[ll]_{h} \\ar@/_11mm/[ll]_{h_{1}} \\ar@(r,u)[]^{l_{D}} " \
        "\\ar@/^3mm/@(d,r)[]^{n_{D}} \\\\\n" \
        "& C \\ar@(l,d)[]^{l_{C}} \\ar@/^3mm/@(d,r)[]^{n_{C}} & \n" \
        "}\n"

    # The same diagram with two "loop" morphisms per object, transposed.
    grid = DiagramGrid(d, transpose=True)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "A \\ar[d]^{f} \\ar@/_3mm/[dd]_{h_{2}} \\ar@(r,u)[]^{l_{A}} " \
        "\\ar@/^3mm/@(u,l)[]^{n_{A}} & \\\\\n" \
        "B \\ar[r]^{g} & C \\ar@(r,u)[]^{l_{C}} \\ar@/^3mm/@(d,r)[]^{n_{C}} \\\\\n" \
        "D \\ar[u]_{k} \\ar@/^7mm/[uu]^{h} \\ar@/^11mm/[uu]^{h_{1}} " \
        "\\ar@(l,d)[]^{l_{D}} \\ar@/^3mm/@(d,r)[]^{n_{D}} & \n" \
        "}\n"


def test_xypic_draw_diagram():
    # A linear diagram.
    A = Object("A")
    B = Object("B")
    C = Object("C")
    D = Object("D")
    E = Object("E")

    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    h = NamedMorphism(C, D, "h")
    i = NamedMorphism(D, E, "i")
    d = Diagram([f, g, h, i])

    grid = DiagramGrid(d, layout="sequential")
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == xypic_draw_diagram(d, layout="sequential")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\interval\test_interval.py ===
from itertools import permutations
import re

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Index,
    Interval,
    IntervalIndex,
    Timedelta,
    Timestamp,
    date_range,
    interval_range,
    isna,
    notna,
    timedelta_range,
)
import pandas._testing as tm
import pandas.core.common as com


@pytest.fixture(params=[None, "foo"])
def name(request):
    return request.param


class TestIntervalIndex:
    index = IntervalIndex.from_arrays([0, 1], [1, 2])

    def create_index(self, closed="right"):
        return IntervalIndex.from_breaks(range(11), closed=closed)

    def create_index_with_nan(self, closed="right"):
        mask = [True, False] + [True] * 8
        return IntervalIndex.from_arrays(
            np.where(mask, np.arange(10), np.nan),
            np.where(mask, np.arange(1, 11), np.nan),
            closed=closed,
        )

    def test_properties(self, closed):
        index = self.create_index(closed=closed)
        assert len(index) == 10
        assert index.size == 10
        assert index.shape == (10,)

        tm.assert_index_equal(index.left, Index(np.arange(10, dtype=np.int64)))
        tm.assert_index_equal(index.right, Index(np.arange(1, 11, dtype=np.int64)))
        tm.assert_index_equal(index.mid, Index(np.arange(0.5, 10.5, dtype=np.float64)))

        assert index.closed == closed

        ivs = [
            Interval(left, right, closed)
            for left, right in zip(range(10), range(1, 11))
        ]
        expected = np.array(ivs, dtype=object)
        tm.assert_numpy_array_equal(np.asarray(index), expected)

        # with nans
        index = self.create_index_with_nan(closed=closed)
        assert len(index) == 10
        assert index.size == 10
        assert index.shape == (10,)

        expected_left = Index([0, np.nan, 2, 3, 4, 5, 6, 7, 8, 9])
        expected_right = expected_left + 1
        expected_mid = expected_left + 0.5
        tm.assert_index_equal(index.left, expected_left)
        tm.assert_index_equal(index.right, expected_right)
        tm.assert_index_equal(index.mid, expected_mid)

        assert index.closed == closed

        ivs = [
            Interval(left, right, closed) if notna(left) else np.nan
            for left, right in zip(expected_left, expected_right)
        ]
        expected = np.array(ivs, dtype=object)
        tm.assert_numpy_array_equal(np.asarray(index), expected)

    @pytest.mark.parametrize(
        "breaks",
        [
            [1, 1, 2, 5, 15, 53, 217, 1014, 5335, 31240, 201608],
            [-np.inf, -100, -10, 0.5, 1, 1.5, 3.8, 101, 202, np.inf],
            date_range("2017-01-01", "2017-01-04"),
            pytest.param(
                date_range("2017-01-01", "2017-01-04", unit="s"),
                marks=pytest.mark.xfail(reason="mismatched result unit"),
            ),
            pd.to_timedelta(["1ns", "2ms", "3s", "4min", "5h", "6D"]),
        ],
    )
    def test_length(self, closed, breaks):
        # GH 18789
        index = IntervalIndex.from_breaks(breaks, closed=closed)
        result = index.length
        expected = Index(iv.length for iv in index)
        tm.assert_index_equal(result, expected)

        # with NA
        index = index.insert(1, np.nan)
        result = index.length
        expected = Index(iv.length if notna(iv) else iv for iv in index)
        tm.assert_index_equal(result, expected)

    def test_with_nans(self, closed):
        index = self.create_index(closed=closed)
        assert index.hasnans is False

        result = index.isna()
        expected = np.zeros(len(index), dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

        result = index.notna()
        expected = np.ones(len(index), dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

        index = self.create_index_with_nan(closed=closed)
        assert index.hasnans is True

        result = index.isna()
        expected = np.array([False, True] + [False] * (len(index) - 2))
        tm.assert_numpy_array_equal(result, expected)

        result = index.notna()
        expected = np.array([True, False] + [True] * (len(index) - 2))
        tm.assert_numpy_array_equal(result, expected)

    def test_copy(self, closed):
        expected = self.create_index(closed=closed)

        result = expected.copy()
        assert result.equals(expected)

        result = expected.copy(deep=True)
        assert result.equals(expected)
        assert result.left is not expected.left

    def test_ensure_copied_data(self, closed):
        # exercise the copy flag in the constructor

        # not copying
        index = self.create_index(closed=closed)
        result = IntervalIndex(index, copy=False)
        tm.assert_numpy_array_equal(
            index.left.values, result.left.values, check_same="same"
        )
        tm.assert_numpy_array_equal(
            index.right.values, result.right.values, check_same="same"
        )

        # by-definition make a copy
        result = IntervalIndex(np.array(index), copy=False)
        tm.assert_numpy_array_equal(
            index.left.values, result.left.values, check_same="copy"
        )
        tm.assert_numpy_array_equal(
            index.right.values, result.right.values, check_same="copy"
        )

    def test_delete(self, closed):
        breaks = np.arange(1, 11, dtype=np.int64)
        expected = IntervalIndex.from_breaks(breaks, closed=closed)
        result = self.create_index(closed=closed).delete(0)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "data",
        [
            interval_range(0, periods=10, closed="neither"),
            interval_range(1.7, periods=8, freq=2.5, closed="both"),
            interval_range(Timestamp("20170101"), periods=12, closed="left"),
            interval_range(Timedelta("1 day"), periods=6, closed="right"),
        ],
    )
    def test_insert(self, data):
        item = data[0]
        idx_item = IntervalIndex([item])

        # start
        expected = idx_item.append(data)
        result = data.insert(0, item)
        tm.assert_index_equal(result, expected)

        # end
        expected = data.append(idx_item)
        result = data.insert(len(data), item)
        tm.assert_index_equal(result, expected)

        # mid
        expected = data[:3].append(idx_item).append(data[3:])
        result = data.insert(3, item)
        tm.assert_index_equal(result, expected)

        # invalid type
        res = data.insert(1, "foo")
        expected = data.astype(object).insert(1, "foo")
        tm.assert_index_equal(res, expected)

        msg = "can only insert Interval objects and NA into an IntervalArray"
        with pytest.raises(TypeError, match=msg):
            data._data.insert(1, "foo")

        # invalid closed
        msg = "'value.closed' is 'left', expected 'right'."
        for closed in {"left", "right", "both", "neither"} - {item.closed}:
            msg = f"'value.closed' is '{closed}', expected '{item.closed}'."
            bad_item = Interval(item.left, item.right, closed=closed)
            res = data.insert(1, bad_item)
            expected = data.astype(object).insert(1, bad_item)
            tm.assert_index_equal(res, expected)
            with pytest.raises(ValueError, match=msg):
                data._data.insert(1, bad_item)

        # GH 18295 (test missing)
        na_idx = IntervalIndex([np.nan], closed=data.closed)
        for na in [np.nan, None, pd.NA]:
            expected = data[:1].append(na_idx).append(data[1:])
            result = data.insert(1, na)
            tm.assert_index_equal(result, expected)

        if data.left.dtype.kind not in ["m", "M"]:
            # trying to insert pd.NaT into a numeric-dtyped Index should cast
            expected = data.astype(object).insert(1, pd.NaT)

            msg = "can only insert Interval objects and NA into an IntervalArray"
            with pytest.raises(TypeError, match=msg):
                data._data.insert(1, pd.NaT)

        result = data.insert(1, pd.NaT)
        tm.assert_index_equal(result, expected)

    def test_is_unique_interval(self, closed):
        """
        Interval specific tests for is_unique in addition to base class tests
        """
        # unique overlapping - distinct endpoints
        idx = IntervalIndex.from_tuples([(0, 1), (0.5, 1.5)], closed=closed)
        assert idx.is_unique is True

        # unique overlapping - shared endpoints
        idx = IntervalIndex.from_tuples([(1, 2), (1, 3), (2, 3)], closed=closed)
        assert idx.is_unique is True

        # unique nested
        idx = IntervalIndex.from_tuples([(-1, 1), (-2, 2)], closed=closed)
        assert idx.is_unique is True

        # unique NaN
        idx = IntervalIndex.from_tuples([(np.nan, np.nan)], closed=closed)
        assert idx.is_unique is True

        # non-unique NaN
        idx = IntervalIndex.from_tuples(
            [(np.nan, np.nan), (np.nan, np.nan)], closed=closed
        )
        assert idx.is_unique is False

    def test_monotonic(self, closed):
        # increasing non-overlapping
        idx = IntervalIndex.from_tuples([(0, 1), (2, 3), (4, 5)], closed=closed)
        assert idx.is_monotonic_increasing is True
        assert idx._is_strictly_monotonic_increasing is True
        assert idx.is_monotonic_decreasing is False
        assert idx._is_strictly_monotonic_decreasing is False

        # decreasing non-overlapping
        idx = IntervalIndex.from_tuples([(4, 5), (2, 3), (1, 2)], closed=closed)
        assert idx.is_monotonic_increasing is False
        assert idx._is_strictly_monotonic_increasing is False
        assert idx.is_monotonic_decreasing is True
        assert idx._is_strictly_monotonic_decreasing is True

        # unordered non-overlapping
        idx = IntervalIndex.from_tuples([(0, 1), (4, 5), (2, 3)], closed=closed)
        assert idx.is_monotonic_increasing is False
        assert idx._is_strictly_monotonic_increasing is False
        assert idx.is_monotonic_decreasing is False
        assert idx._is_strictly_monotonic_decreasing is False

        # increasing overlapping
        idx = IntervalIndex.from_tuples([(0, 2), (0.5, 2.5), (1, 3)], closed=closed)
        assert idx.is_monotonic_increasing is True
        assert idx._is_strictly_monotonic_increasing is True
        assert idx.is_monotonic_decreasing is False
        assert idx._is_strictly_monotonic_decreasing is False

        # decreasing overlapping
        idx = IntervalIndex.from_tuples([(1, 3), (0.5, 2.5), (0, 2)], closed=closed)
        assert idx.is_monotonic_increasing is False
        assert idx._is_strictly_monotonic_increasing is False
        assert idx.is_monotonic_decreasing is True
        assert idx._is_strictly_monotonic_decreasing is True

        # unordered overlapping
        idx = IntervalIndex.from_tuples([(0.5, 2.5), (0, 2), (1, 3)], closed=closed)
        assert idx.is_monotonic_increasing is False
        assert idx._is_strictly_monotonic_increasing is False
        assert idx.is_monotonic_decreasing is False
        assert idx._is_strictly_monotonic_decreasing is False

        # increasing overlapping shared endpoints
        idx = IntervalIndex.from_tuples([(1, 2), (1, 3), (2, 3)], closed=closed)
        assert idx.is_monotonic_increasing is True
        assert idx._is_strictly_monotonic_increasing is True
        assert idx.is_monotonic_decreasing is False
        assert idx._is_strictly_monotonic_decreasing is False

        # decreasing overlapping shared endpoints
        idx = IntervalIndex.from_tuples([(2, 3), (1, 3), (1, 2)], closed=closed)
        assert idx.is_monotonic_increasing is False
        assert idx._is_strictly_monotonic_increasing is False
        assert idx.is_monotonic_decreasing is True
        assert idx._is_strictly_monotonic_decreasing is True

        # stationary
        idx = IntervalIndex.from_tuples([(0, 1), (0, 1)], closed=closed)
        assert idx.is_monotonic_increasing is True
        assert idx._is_strictly_monotonic_increasing is False
        assert idx.is_monotonic_decreasing is True
        assert idx._is_strictly_monotonic_decreasing is False

        # empty
        idx = IntervalIndex([], closed=closed)
        assert idx.is_monotonic_increasing is True
        assert idx._is_strictly_monotonic_increasing is True
        assert idx.is_monotonic_decreasing is True
        assert idx._is_strictly_monotonic_decreasing is True

    def test_is_monotonic_with_nans(self):
        # GH#41831
        index = IntervalIndex([np.nan, np.nan])

        assert not index.is_monotonic_increasing
        assert not index._is_strictly_monotonic_increasing
        assert not index.is_monotonic_increasing
        assert not index._is_strictly_monotonic_decreasing
        assert not index.is_monotonic_decreasing

    @pytest.mark.parametrize(
        "breaks",
        [
            date_range("20180101", periods=4),
            date_range("20180101", periods=4, tz="US/Eastern"),
            timedelta_range("0 days", periods=4),
        ],
        ids=lambda x: str(x.dtype),
    )
    def test_maybe_convert_i8(self, breaks):
        # GH 20636
        index = IntervalIndex.from_breaks(breaks)

        # intervalindex
        result = index._maybe_convert_i8(index)
        expected = IntervalIndex.from_breaks(breaks.asi8)
        tm.assert_index_equal(result, expected)

        # interval
        interval = Interval(breaks[0], breaks[1])
        result = index._maybe_convert_i8(interval)
        expected = Interval(breaks[0]._value, breaks[1]._value)
        assert result == expected

        # datetimelike index
        result = index._maybe_convert_i8(breaks)
        expected = Index(breaks.asi8)
        tm.assert_index_equal(result, expected)

        # datetimelike scalar
        result = index._maybe_convert_i8(breaks[0])
        expected = breaks[0]._value
        assert result == expected

        # list-like of datetimelike scalars
        result = index._maybe_convert_i8(list(breaks))
        expected = Index(breaks.asi8)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "breaks",
        [date_range("2018-01-01", periods=5), timedelta_range("0 days", periods=5)],
    )
    def test_maybe_convert_i8_nat(self, breaks):
        # GH 20636
        index = IntervalIndex.from_breaks(breaks)

        to_convert = breaks._constructor([pd.NaT] * 3).as_unit("ns")
        expected = Index([np.nan] * 3, dtype=np.float64)
        result = index._maybe_convert_i8(to_convert)
        tm.assert_index_equal(result, expected)

        to_convert = to_convert.insert(0, breaks[0])
        expected = expected.insert(0, float(breaks[0]._value))
        result = index._maybe_convert_i8(to_convert)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "make_key",
        [lambda breaks: breaks, list],
        ids=["lambda", "list"],
    )
    def test_maybe_convert_i8_numeric(self, make_key, any_real_numpy_dtype):
        # GH 20636
        breaks = np.arange(5, dtype=any_real_numpy_dtype)
        index = IntervalIndex.from_breaks(breaks)
        key = make_key(breaks)

        result = index._maybe_convert_i8(key)
        kind = breaks.dtype.kind
        expected_dtype = {"i": np.int64, "u": np.uint64, "f": np.float64}[kind]
        expected = Index(key, dtype=expected_dtype)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "make_key",
        [
            IntervalIndex.from_breaks,
            lambda breaks: Interval(breaks[0], breaks[1]),
            lambda breaks: breaks[0],
        ],
        ids=["IntervalIndex", "Interval", "scalar"],
    )
    def test_maybe_convert_i8_numeric_identical(self, make_key, any_real_numpy_dtype):
        # GH 20636
        breaks = np.arange(5, dtype=any_real_numpy_dtype)
        index = IntervalIndex.from_breaks(breaks)
        key = make_key(breaks)

        # test if _maybe_convert_i8 won't change key if an Interval or IntervalIndex
        result = index._maybe_convert_i8(key)
        assert result is key

    @pytest.mark.parametrize(
        "breaks1, breaks2",
        permutations(
            [
                date_range("20180101", periods=4),
                date_range("20180101", periods=4, tz="US/Eastern"),
                timedelta_range("0 days", periods=4),
            ],
            2,
        ),
        ids=lambda x: str(x.dtype),
    )
    @pytest.mark.parametrize(
        "make_key",
        [
            IntervalIndex.from_breaks,
            lambda breaks: Interval(breaks[0], breaks[1]),
            lambda breaks: breaks,
            lambda breaks: breaks[0],
            list,
        ],
        ids=["IntervalIndex", "Interval", "Index", "scalar", "list"],
    )
    def test_maybe_convert_i8_errors(self, breaks1, breaks2, make_key):
        # GH 20636
        index = IntervalIndex.from_breaks(breaks1)
        key = make_key(breaks2)

        msg = (
            f"Cannot index an IntervalIndex of subtype {breaks1.dtype} with "
            f"values of dtype {breaks2.dtype}"
        )
        msg = re.escape(msg)
        with pytest.raises(ValueError, match=msg):
            index._maybe_convert_i8(key)

    def test_contains_method(self):
        # can select values that are IN the range of a value
        i = IntervalIndex.from_arrays([0, 1], [1, 2])

        expected = np.array([False, False], dtype="bool")
        actual = i.contains(0)
        tm.assert_numpy_array_equal(actual, expected)
        actual = i.contains(3)
        tm.assert_numpy_array_equal(actual, expected)

        expected = np.array([True, False], dtype="bool")
        actual = i.contains(0.5)
        tm.assert_numpy_array_equal(actual, expected)
        actual = i.contains(1)
        tm.assert_numpy_array_equal(actual, expected)

        # __contains__ not implemented for "interval in interval", follow
        # that for the contains method for now
        with pytest.raises(
            NotImplementedError, match="contains not implemented for two"
        ):
            i.contains(Interval(0, 1))

    def test_dropna(self, closed):
        expected = IntervalIndex.from_tuples([(0.0, 1.0), (1.0, 2.0)], closed=closed)

        ii = IntervalIndex.from_tuples([(0, 1), (1, 2), np.nan], closed=closed)
        result = ii.dropna()
        tm.assert_index_equal(result, expected)

        ii = IntervalIndex.from_arrays([0, 1, np.nan], [1, 2, np.nan], closed=closed)
        result = ii.dropna()
        tm.assert_index_equal(result, expected)

    def test_non_contiguous(self, closed):
        index = IntervalIndex.from_tuples([(0, 1), (2, 3)], closed=closed)
        target = [0.5, 1.5, 2.5]
        actual = index.get_indexer(target)
        expected = np.array([0, -1, 1], dtype="intp")
        tm.assert_numpy_array_equal(actual, expected)

        assert 1.5 not in index

    def test_isin(self, closed):
        index = self.create_index(closed=closed)

        expected = np.array([True] + [False] * (len(index) - 1))
        result = index.isin(index[:1])
        tm.assert_numpy_array_equal(result, expected)

        result = index.isin([index[0]])
        tm.assert_numpy_array_equal(result, expected)

        other = IntervalIndex.from_breaks(np.arange(-2, 10), closed=closed)
        expected = np.array([True] * (len(index) - 1) + [False])
        result = index.isin(other)
        tm.assert_numpy_array_equal(result, expected)

        result = index.isin(other.tolist())
        tm.assert_numpy_array_equal(result, expected)

        for other_closed in ["right", "left", "both", "neither"]:
            other = self.create_index(closed=other_closed)
            expected = np.repeat(closed == other_closed, len(index))
            result = index.isin(other)
            tm.assert_numpy_array_equal(result, expected)

            result = index.isin(other.tolist())
            tm.assert_numpy_array_equal(result, expected)

    def test_comparison(self):
        actual = Interval(0, 1) < self.index
        expected = np.array([False, True])
        tm.assert_numpy_array_equal(actual, expected)

        actual = Interval(0.5, 1.5) < self.index
        expected = np.array([False, True])
        tm.assert_numpy_array_equal(actual, expected)
        actual = self.index > Interval(0.5, 1.5)
        tm.assert_numpy_array_equal(actual, expected)

        actual = self.index == self.index
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(actual, expected)
        actual = self.index <= self.index
        tm.assert_numpy_array_equal(actual, expected)
        actual = self.index >= self.index
        tm.assert_numpy_array_equal(actual, expected)

        actual = self.index < self.index
        expected = np.array([False, False])
        tm.assert_numpy_array_equal(actual, expected)
        actual = self.index > self.index
        tm.assert_numpy_array_equal(actual, expected)

        actual = self.index == IntervalIndex.from_breaks([0, 1, 2], "left")
        tm.assert_numpy_array_equal(actual, expected)

        actual = self.index == self.index.values
        tm.assert_numpy_array_equal(actual, np.array([True, True]))
        actual = self.index.values == self.index
        tm.assert_numpy_array_equal(actual, np.array([True, True]))
        actual = self.index <= self.index.values
        tm.assert_numpy_array_equal(actual, np.array([True, True]))
        actual = self.index != self.index.values
        tm.assert_numpy_array_equal(actual, np.array([False, False]))
        actual = self.index > self.index.values
        tm.assert_numpy_array_equal(actual, np.array([False, False]))
        actual = self.index.values > self.index
        tm.assert_numpy_array_equal(actual, np.array([False, False]))

        # invalid comparisons
        actual = self.index == 0
        tm.assert_numpy_array_equal(actual, np.array([False, False]))
        actual = self.index == self.index.left
        tm.assert_numpy_array_equal(actual, np.array([False, False]))

        msg = "|".join(
            [
                "not supported between instances of 'int' and '.*.Interval'",
                r"Invalid comparison between dtype=interval\[int64, right\] and ",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            self.index > 0
        with pytest.raises(TypeError, match=msg):
            self.index <= 0
        with pytest.raises(TypeError, match=msg):
            self.index > np.arange(2)

        msg = "Lengths must match to compare"
        with pytest.raises(ValueError, match=msg):
            self.index > np.arange(3)

    def test_missing_values(self, closed):
        idx = Index(
            [np.nan, Interval(0, 1, closed=closed), Interval(1, 2, closed=closed)]
        )
        idx2 = IntervalIndex.from_arrays([np.nan, 0, 1], [np.nan, 1, 2], closed=closed)
        assert idx.equals(idx2)

        msg = (
            "missing values must be missing in the same location both left "
            "and right sides"
        )
        with pytest.raises(ValueError, match=msg):
            IntervalIndex.from_arrays(
                [np.nan, 0, 1], np.array([0, 1, 2]), closed=closed
            )

        tm.assert_numpy_array_equal(isna(idx), np.array([True, False, False]))

    def test_sort_values(self, closed):
        index = self.create_index(closed=closed)

        result = index.sort_values()
        tm.assert_index_equal(result, index)

        result = index.sort_values(ascending=False)
        tm.assert_index_equal(result, index[::-1])

        # with nan
        index = IntervalIndex([Interval(1, 2), np.nan, Interval(0, 1)])

        result = index.sort_values()
        expected = IntervalIndex([Interval(0, 1), Interval(1, 2), np.nan])
        tm.assert_index_equal(result, expected)

        result = index.sort_values(ascending=False, na_position="first")
        expected = IntervalIndex([np.nan, Interval(1, 2), Interval(0, 1)])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    def test_datetime(self, tz):
        start = Timestamp("2000-01-01", tz=tz)
        dates = date_range(start=start, periods=10)
        index = IntervalIndex.from_breaks(dates)

        # test mid
        start = Timestamp("2000-01-01T12:00", tz=tz)
        expected = date_range(start=start, periods=9)
        tm.assert_index_equal(index.mid, expected)

        # __contains__ doesn't check individual points
        assert Timestamp("2000-01-01", tz=tz) not in index
        assert Timestamp("2000-01-01T12", tz=tz) not in index
        assert Timestamp("2000-01-02", tz=tz) not in index
        iv_true = Interval(
            Timestamp("2000-01-02", tz=tz), Timestamp("2000-01-03", tz=tz)
        )
        iv_false = Interval(
            Timestamp("1999-12-31", tz=tz), Timestamp("2000-01-01", tz=tz)
        )
        assert iv_true in index
        assert iv_false not in index

        # .contains does check individual points
        assert not index.contains(Timestamp("2000-01-01", tz=tz)).any()
        assert index.contains(Timestamp("2000-01-01T12", tz=tz)).any()
        assert index.contains(Timestamp("2000-01-02", tz=tz)).any()

        # test get_indexer
        start = Timestamp("1999-12-31T12:00", tz=tz)
        target = date_range(start=start, periods=7, freq="12h")
        actual = index.get_indexer(target)
        expected = np.array([-1, -1, 0, 0, 1, 1, 2], dtype="intp")
        tm.assert_numpy_array_equal(actual, expected)

        start = Timestamp("2000-01-08T18:00", tz=tz)
        target = date_range(start=start, periods=7, freq="6h")
        actual = index.get_indexer(target)
        expected = np.array([7, 7, 8, 8, 8, 8, -1], dtype="intp")
        tm.assert_numpy_array_equal(actual, expected)

    def test_append(self, closed):
        index1 = IntervalIndex.from_arrays([0, 1], [1, 2], closed=closed)
        index2 = IntervalIndex.from_arrays([1, 2], [2, 3], closed=closed)

        result = index1.append(index2)
        expected = IntervalIndex.from_arrays([0, 1, 1, 2], [1, 2, 2, 3], closed=closed)
        tm.assert_index_equal(result, expected)

        result = index1.append([index1, index2])
        expected = IntervalIndex.from_arrays(
            [0, 1, 0, 1, 1, 2], [1, 2, 1, 2, 2, 3], closed=closed
        )
        tm.assert_index_equal(result, expected)

        for other_closed in {"left", "right", "both", "neither"} - {closed}:
            index_other_closed = IntervalIndex.from_arrays(
                [0, 1], [1, 2], closed=other_closed
            )
            result = index1.append(index_other_closed)
            expected = index1.astype(object).append(index_other_closed.astype(object))
            tm.assert_index_equal(result, expected)

    def test_is_non_overlapping_monotonic(self, closed):
        # Should be True in all cases
        tpls = [(0, 1), (2, 3), (4, 5), (6, 7)]
        idx = IntervalIndex.from_tuples(tpls, closed=closed)
        assert idx.is_non_overlapping_monotonic is True

        idx = IntervalIndex.from_tuples(tpls[::-1], closed=closed)
        assert idx.is_non_overlapping_monotonic is True

        # Should be False in all cases (overlapping)
        tpls = [(0, 2), (1, 3), (4, 5), (6, 7)]
        idx = IntervalIndex.from_tuples(tpls, closed=closed)
        assert idx.is_non_overlapping_monotonic is False

        idx = IntervalIndex.from_tuples(tpls[::-1], closed=closed)
        assert idx.is_non_overlapping_monotonic is False

        # Should be False in all cases (non-monotonic)
        tpls = [(0, 1), (2, 3), (6, 7), (4, 5)]
        idx = IntervalIndex.from_tuples(tpls, closed=closed)
        assert idx.is_non_overlapping_monotonic is False

        idx = IntervalIndex.from_tuples(tpls[::-1], closed=closed)
        assert idx.is_non_overlapping_monotonic is False

        # Should be False for closed='both', otherwise True (GH16560)
        if closed == "both":
            idx = IntervalIndex.from_breaks(range(4), closed=closed)
            assert idx.is_non_overlapping_monotonic is False
        else:
            idx = IntervalIndex.from_breaks(range(4), closed=closed)
            assert idx.is_non_overlapping_monotonic is True

    @pytest.mark.parametrize(
        "start, shift, na_value",
        [
            (0, 1, np.nan),
            (Timestamp("2018-01-01"), Timedelta("1 day"), pd.NaT),
            (Timedelta("0 days"), Timedelta("1 day"), pd.NaT),
        ],
    )
    def test_is_overlapping(self, start, shift, na_value, closed):
        # GH 23309
        # see test_interval_tree.py for extensive tests; interface tests here

        # non-overlapping
        tuples = [(start + n * shift, start + (n + 1) * shift) for n in (0, 2, 4)]
        index = IntervalIndex.from_tuples(tuples, closed=closed)
        assert index.is_overlapping is False

        # non-overlapping with NA
        tuples = [(na_value, na_value)] + tuples + [(na_value, na_value)]
        index = IntervalIndex.from_tuples(tuples, closed=closed)
        assert index.is_overlapping is False

        # overlapping
        tuples = [(start + n * shift, start + (n + 2) * shift) for n in range(3)]
        index = IntervalIndex.from_tuples(tuples, closed=closed)
        assert index.is_overlapping is True

        # overlapping with NA
        tuples = [(na_value, na_value)] + tuples + [(na_value, na_value)]
        index = IntervalIndex.from_tuples(tuples, closed=closed)
        assert index.is_overlapping is True

        # common endpoints
        tuples = [(start + n * shift, start + (n + 1) * shift) for n in range(3)]
        index = IntervalIndex.from_tuples(tuples, closed=closed)
        result = index.is_overlapping
        expected = closed == "both"
        assert result is expected

        # common endpoints with NA
        tuples = [(na_value, na_value)] + tuples + [(na_value, na_value)]
        index = IntervalIndex.from_tuples(tuples, closed=closed)
        result = index.is_overlapping
        assert result is expected

        # intervals with duplicate left values
        a = [10, 15, 20, 25, 30, 35, 40, 45, 45, 50, 55, 60, 65, 70, 75, 80, 85]
        b = [15, 20, 25, 30, 35, 40, 45, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
        index = IntervalIndex.from_arrays(a, b, closed="right")
        result = index.is_overlapping
        assert result is False

    @pytest.mark.parametrize(
        "tuples",
        [
            list(zip(range(10), range(1, 11))),
            list(
                zip(
                    date_range("20170101", periods=10),
                    date_range("20170101", periods=10),
                )
            ),
            list(
                zip(
                    timedelta_range("0 days", periods=10),
                    timedelta_range("1 day", periods=10),
                )
            ),
        ],
    )
    def test_to_tuples(self, tuples):
        # GH 18756
        idx = IntervalIndex.from_tuples(tuples)
        result = idx.to_tuples()
        expected = Index(com.asarray_tuplesafe(tuples))
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "tuples",
        [
            list(zip(range(10), range(1, 11))) + [np.nan],
            list(
                zip(
                    date_range("20170101", periods=10),
                    date_range("20170101", periods=10),
                )
            )
            + [np.nan],
            list(
                zip(
                    timedelta_range("0 days", periods=10),
                    timedelta_range("1 day", periods=10),
                )
            )
            + [np.nan],
        ],
    )
    @pytest.mark.parametrize("na_tuple", [True, False])
    def test_to_tuples_na(self, tuples, na_tuple):
        # GH 18756
        idx = IntervalIndex.from_tuples(tuples)
        result = idx.to_tuples(na_tuple=na_tuple)

        # check the non-NA portion
        expected_notna = Index(com.asarray_tuplesafe(tuples[:-1]))
        result_notna = result[:-1]
        tm.assert_index_equal(result_notna, expected_notna)

        # check the NA portion
        result_na = result[-1]
        if na_tuple:
            assert isinstance(result_na, tuple)
            assert len(result_na) == 2
            assert all(isna(x) for x in result_na)
        else:
            assert isna(result_na)

    def test_nbytes(self):
        # GH 19209
        left = np.arange(0, 4, dtype="i8")
        right = np.arange(1, 5, dtype="i8")

        result = IntervalIndex.from_arrays(left, right).nbytes
        expected = 64  # 4 * 8 * 2
        assert result == expected

    @pytest.mark.parametrize("new_closed", ["left", "right", "both", "neither"])
    def test_set_closed(self, name, closed, new_closed):
        # GH 21670
        index = interval_range(0, 5, closed=closed, name=name)
        result = index.set_closed(new_closed)
        expected = interval_range(0, 5, closed=new_closed, name=name)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("bad_closed", ["foo", 10, "LEFT", True, False])
    def test_set_closed_errors(self, bad_closed):
        # GH 21670
        index = interval_range(0, 5)
        msg = f"invalid option for 'closed': {bad_closed}"
        with pytest.raises(ValueError, match=msg):
            index.set_closed(bad_closed)

    def test_is_all_dates(self):
        # GH 23576
        year_2017 = Interval(
            Timestamp("2017-01-01 00:00:00"), Timestamp("2018-01-01 00:00:00")
        )
        year_2017_index = IntervalIndex([year_2017])
        assert not year_2017_index._is_all_dates


def test_dir():
    # GH#27571 dir(interval_index) should not raise
    index = IntervalIndex.from_arrays([0, 1], [1, 2])
    result = dir(index)
    assert "str" not in result


def test_searchsorted_different_argument_classes(listlike_box):
    # https://github.com/pandas-dev/pandas/issues/32762
    values = IntervalIndex([Interval(0, 1), Interval(1, 2)])
    result = values.searchsorted(listlike_box(values))
    expected = np.array([0, 1], dtype=result.dtype)
    tm.assert_numpy_array_equal(result, expected)

    result = values._data.searchsorted(listlike_box(values))
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "arg", [[1, 2], ["a", "b"], [Timestamp("2020-01-01", tz="Europe/London")] * 2]
)
def test_searchsorted_invalid_argument(arg):
    values = IntervalIndex([Interval(0, 1), Interval(1, 2)])
    msg = "'<' not supported between instances of 'pandas._libs.interval.Interval' and "
    with pytest.raises(TypeError, match=msg):
        values.searchsorted(arg)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\concat\test_concat.py ===
from collections import (
    abc,
    deque,
)
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    PeriodIndex,
    Series,
    concat,
    date_range,
)
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tests.extension.decimal import to_decimal


class TestConcatenate:
    def test_append_concat(self):
        # GH#1815
        d1 = date_range("12/31/1990", "12/31/1999", freq="YE-DEC")
        d2 = date_range("12/31/2000", "12/31/2009", freq="YE-DEC")

        s1 = Series(np.random.default_rng(2).standard_normal(10), d1)
        s2 = Series(np.random.default_rng(2).standard_normal(10), d2)

        s1 = s1.to_period()
        s2 = s2.to_period()

        # drops index
        result = concat([s1, s2])
        assert isinstance(result.index, PeriodIndex)
        assert result.index[0] == s1.index[0]

    # test is not written to work with string dtype (checks .base)
    @pytest.mark.xfail(using_string_dtype(), reason="TODO(infer_string)")
    def test_concat_copy(self, using_array_manager, using_copy_on_write):
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 3)))
        df2 = DataFrame(np.random.default_rng(2).integers(0, 10, size=4).reshape(4, 1))
        df3 = DataFrame({5: "foo"}, index=range(4))

        # These are actual copies.
        result = concat([df, df2, df3], axis=1, copy=True)

        if not using_copy_on_write:
            for arr in result._mgr.arrays:
                assert not any(
                    np.shares_memory(arr, y)
                    for x in [df, df2, df3]
                    for y in x._mgr.arrays
                )
        else:
            for arr in result._mgr.arrays:
                assert arr.base is not None

        # These are the same.
        result = concat([df, df2, df3], axis=1, copy=False)

        for arr in result._mgr.arrays:
            if arr.dtype.kind == "f":
                assert arr.base is df._mgr.arrays[0].base
            elif arr.dtype.kind in ["i", "u"]:
                assert arr.base is df2._mgr.arrays[0].base
            elif arr.dtype == object:
                if using_array_manager:
                    # we get the same array object, which has no base
                    assert arr is df3._mgr.arrays[0]
                else:
                    assert arr.base is not None
                assert arr.base is not None

        # Float block was consolidated.
        df4 = DataFrame(np.random.default_rng(2).standard_normal((4, 1)))
        result = concat([df, df2, df3, df4], axis=1, copy=False)
        for arr in result._mgr.arrays:
            if arr.dtype.kind == "f":
                if using_array_manager or using_copy_on_write:
                    # this is a view on some array in either df or df4
                    assert any(
                        np.shares_memory(arr, other)
                        for other in df._mgr.arrays + df4._mgr.arrays
                    )
                else:
                    # the block was consolidated, so we got a copy anyway
                    assert arr.base is None
            elif arr.dtype.kind in ["i", "u"]:
                assert arr.base is df2._mgr.arrays[0].base
            elif arr.dtype == object:
                # this is a view on df3
                assert any(np.shares_memory(arr, other) for other in df3._mgr.arrays)

    def test_concat_with_group_keys(self):
        # axis=0
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 4)))
        df2 = DataFrame(np.random.default_rng(2).standard_normal((4, 4)))

        result = concat([df, df2], keys=[0, 1])
        exp_index = MultiIndex.from_arrays(
            [[0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 0, 1, 2, 3]]
        )
        expected = DataFrame(np.r_[df.values, df2.values], index=exp_index)
        tm.assert_frame_equal(result, expected)

        result = concat([df, df], keys=[0, 1])
        exp_index2 = MultiIndex.from_arrays([[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]])
        expected = DataFrame(np.r_[df.values, df.values], index=exp_index2)
        tm.assert_frame_equal(result, expected)

        # axis=1
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 3)))
        df2 = DataFrame(np.random.default_rng(2).standard_normal((4, 4)))

        result = concat([df, df2], keys=[0, 1], axis=1)
        expected = DataFrame(np.c_[df.values, df2.values], columns=exp_index)
        tm.assert_frame_equal(result, expected)

        result = concat([df, df], keys=[0, 1], axis=1)
        expected = DataFrame(np.c_[df.values, df.values], columns=exp_index2)
        tm.assert_frame_equal(result, expected)

    def test_concat_keys_specific_levels(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        pieces = [df.iloc[:, [0, 1]], df.iloc[:, [2]], df.iloc[:, [3]]]
        level = ["three", "two", "one", "zero"]
        result = concat(
            pieces,
            axis=1,
            keys=["one", "two", "three"],
            levels=[level],
            names=["group_key"],
        )

        tm.assert_index_equal(result.columns.levels[0], Index(level, name="group_key"))
        tm.assert_index_equal(result.columns.levels[1], Index([0, 1, 2, 3]))

        assert result.columns.names == ["group_key", None]

    @pytest.mark.parametrize("mapping", ["mapping", "dict"])
    def test_concat_mapping(self, mapping, non_dict_mapping_subclass):
        constructor = dict if mapping == "dict" else non_dict_mapping_subclass
        frames = constructor(
            {
                "foo": DataFrame(np.random.default_rng(2).standard_normal((4, 3))),
                "bar": DataFrame(np.random.default_rng(2).standard_normal((4, 3))),
                "baz": DataFrame(np.random.default_rng(2).standard_normal((4, 3))),
                "qux": DataFrame(np.random.default_rng(2).standard_normal((4, 3))),
            }
        )

        sorted_keys = list(frames.keys())

        result = concat(frames)
        expected = concat([frames[k] for k in sorted_keys], keys=sorted_keys)
        tm.assert_frame_equal(result, expected)

        result = concat(frames, axis=1)
        expected = concat([frames[k] for k in sorted_keys], keys=sorted_keys, axis=1)
        tm.assert_frame_equal(result, expected)

        keys = ["baz", "foo", "bar"]
        result = concat(frames, keys=keys)
        expected = concat([frames[k] for k in keys], keys=keys)
        tm.assert_frame_equal(result, expected)

    def test_concat_keys_and_levels(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((1, 3)))
        df2 = DataFrame(np.random.default_rng(2).standard_normal((1, 4)))

        levels = [["foo", "baz"], ["one", "two"]]
        names = ["first", "second"]
        result = concat(
            [df, df2, df, df2],
            keys=[("foo", "one"), ("foo", "two"), ("baz", "one"), ("baz", "two")],
            levels=levels,
            names=names,
        )
        expected = concat([df, df2, df, df2])
        exp_index = MultiIndex(
            levels=levels + [[0]],
            codes=[[0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 0, 0]],
            names=names + [None],
        )
        expected.index = exp_index

        tm.assert_frame_equal(result, expected)

        # no names
        result = concat(
            [df, df2, df, df2],
            keys=[("foo", "one"), ("foo", "two"), ("baz", "one"), ("baz", "two")],
            levels=levels,
        )
        assert result.index.names == (None,) * 3

        # no levels
        result = concat(
            [df, df2, df, df2],
            keys=[("foo", "one"), ("foo", "two"), ("baz", "one"), ("baz", "two")],
            names=["first", "second"],
        )
        assert result.index.names == ("first", "second", None)
        tm.assert_index_equal(
            result.index.levels[0], Index(["baz", "foo"], name="first")
        )

    def test_concat_keys_levels_no_overlap(self):
        # GH #1406
        df = DataFrame(np.random.default_rng(2).standard_normal((1, 3)), index=["a"])
        df2 = DataFrame(np.random.default_rng(2).standard_normal((1, 4)), index=["b"])

        msg = "Values not found in passed level"
        with pytest.raises(ValueError, match=msg):
            concat([df, df], keys=["one", "two"], levels=[["foo", "bar", "baz"]])

        msg = "Key one not in level"
        with pytest.raises(ValueError, match=msg):
            concat([df, df2], keys=["one", "two"], levels=[["foo", "bar", "baz"]])

    def test_crossed_dtypes_weird_corner(self):
        columns = ["A", "B", "C", "D"]
        df1 = DataFrame(
            {
                "A": np.array([1, 2, 3, 4], dtype="f8"),
                "B": np.array([1, 2, 3, 4], dtype="i8"),
                "C": np.array([1, 2, 3, 4], dtype="f8"),
                "D": np.array([1, 2, 3, 4], dtype="i8"),
            },
            columns=columns,
        )

        df2 = DataFrame(
            {
                "A": np.array([1, 2, 3, 4], dtype="i8"),
                "B": np.array([1, 2, 3, 4], dtype="f8"),
                "C": np.array([1, 2, 3, 4], dtype="i8"),
                "D": np.array([1, 2, 3, 4], dtype="f8"),
            },
            columns=columns,
        )

        appended = concat([df1, df2], ignore_index=True)
        expected = DataFrame(
            np.concatenate([df1.values, df2.values], axis=0), columns=columns
        )
        tm.assert_frame_equal(appended, expected)

        df = DataFrame(np.random.default_rng(2).standard_normal((1, 3)), index=["a"])
        df2 = DataFrame(np.random.default_rng(2).standard_normal((1, 4)), index=["b"])
        result = concat([df, df2], keys=["one", "two"], names=["first", "second"])
        assert result.index.names == ("first", "second")

    def test_with_mixed_tuples(self, sort):
        # 10697
        # columns have mixed tuples, so handle properly
        df1 = DataFrame({"A": "foo", ("B", 1): "bar"}, index=range(2))
        df2 = DataFrame({"B": "foo", ("B", 1): "bar"}, index=range(2))

        # it works
        concat([df1, df2], sort=sort)

    def test_concat_mixed_objs_columns(self):
        # Test column-wise concat for mixed series/frames (axis=1)
        # G2385

        index = date_range("01-Jan-2013", periods=10, freq="h")
        arr = np.arange(10, dtype="int64")
        s1 = Series(arr, index=index)
        s2 = Series(arr, index=index)
        df = DataFrame(arr.reshape(-1, 1), index=index)

        expected = DataFrame(
            np.repeat(arr, 2).reshape(-1, 2), index=index, columns=[0, 0]
        )
        result = concat([df, df], axis=1)
        tm.assert_frame_equal(result, expected)

        expected = DataFrame(
            np.repeat(arr, 2).reshape(-1, 2), index=index, columns=[0, 1]
        )
        result = concat([s1, s2], axis=1)
        tm.assert_frame_equal(result, expected)

        expected = DataFrame(
            np.repeat(arr, 3).reshape(-1, 3), index=index, columns=[0, 1, 2]
        )
        result = concat([s1, s2, s1], axis=1)
        tm.assert_frame_equal(result, expected)

        expected = DataFrame(
            np.repeat(arr, 5).reshape(-1, 5), index=index, columns=[0, 0, 1, 2, 3]
        )
        result = concat([s1, df, s2, s2, s1], axis=1)
        tm.assert_frame_equal(result, expected)

        # with names
        s1.name = "foo"
        expected = DataFrame(
            np.repeat(arr, 3).reshape(-1, 3), index=index, columns=["foo", 0, 0]
        )
        result = concat([s1, df, s2], axis=1)
        tm.assert_frame_equal(result, expected)

        s2.name = "bar"
        expected = DataFrame(
            np.repeat(arr, 3).reshape(-1, 3), index=index, columns=["foo", 0, "bar"]
        )
        result = concat([s1, df, s2], axis=1)
        tm.assert_frame_equal(result, expected)

        # ignore index
        expected = DataFrame(
            np.repeat(arr, 3).reshape(-1, 3), index=index, columns=[0, 1, 2]
        )
        result = concat([s1, df, s2], axis=1, ignore_index=True)
        tm.assert_frame_equal(result, expected)

    def test_concat_mixed_objs_index(self):
        # Test row-wise concat for mixed series/frames with a common name
        # GH2385, GH15047

        index = date_range("01-Jan-2013", periods=10, freq="h")
        arr = np.arange(10, dtype="int64")
        s1 = Series(arr, index=index)
        s2 = Series(arr, index=index)
        df = DataFrame(arr.reshape(-1, 1), index=index)

        expected = DataFrame(
            np.tile(arr, 3).reshape(-1, 1), index=index.tolist() * 3, columns=[0]
        )
        result = concat([s1, df, s2])
        tm.assert_frame_equal(result, expected)

    def test_concat_mixed_objs_index_names(self):
        # Test row-wise concat for mixed series/frames with distinct names
        # GH2385, GH15047

        index = date_range("01-Jan-2013", periods=10, freq="h")
        arr = np.arange(10, dtype="int64")
        s1 = Series(arr, index=index, name="foo")
        s2 = Series(arr, index=index, name="bar")
        df = DataFrame(arr.reshape(-1, 1), index=index)

        expected = DataFrame(
            np.kron(np.where(np.identity(3) == 1, 1, np.nan), arr).T,
            index=index.tolist() * 3,
            columns=["foo", 0, "bar"],
        )
        result = concat([s1, df, s2])
        tm.assert_frame_equal(result, expected)

        # Rename all series to 0 when ignore_index=True
        expected = DataFrame(np.tile(arr, 3).reshape(-1, 1), columns=[0])
        result = concat([s1, df, s2], ignore_index=True)
        tm.assert_frame_equal(result, expected)

    def test_dtype_coercion(self):
        # 12411
        df = DataFrame({"date": [pd.Timestamp("20130101").tz_localize("UTC"), pd.NaT]})

        result = concat([df.iloc[[0]], df.iloc[[1]]])
        tm.assert_series_equal(result.dtypes, df.dtypes)

        # 12045
        df = DataFrame({"date": [datetime(2012, 1, 1), datetime(1012, 1, 2)]})
        result = concat([df.iloc[[0]], df.iloc[[1]]])
        tm.assert_series_equal(result.dtypes, df.dtypes)

        # 11594
        df = DataFrame({"text": ["some words"] + [None] * 9})
        result = concat([df.iloc[[0]], df.iloc[[1]]])
        tm.assert_series_equal(result.dtypes, df.dtypes)

    def test_concat_single_with_key(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))

        result = concat([df], keys=["foo"])
        expected = concat([df, df], keys=["foo", "bar"])
        tm.assert_frame_equal(result, expected[:10])

    def test_concat_no_items_raises(self):
        with pytest.raises(ValueError, match="No objects to concatenate"):
            concat([])

    def test_concat_exclude_none(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))

        pieces = [df[:5], None, None, df[5:]]
        result = concat(pieces)
        tm.assert_frame_equal(result, df)
        with pytest.raises(ValueError, match="All objects passed were None"):
            concat([None, None])

    def test_concat_keys_with_none(self):
        # #1649
        df0 = DataFrame([[10, 20, 30], [10, 20, 30], [10, 20, 30]])

        result = concat({"a": None, "b": df0, "c": df0[:2], "d": df0[:1], "e": df0})
        expected = concat({"b": df0, "c": df0[:2], "d": df0[:1], "e": df0})
        tm.assert_frame_equal(result, expected)

        result = concat(
            [None, df0, df0[:2], df0[:1], df0], keys=["a", "b", "c", "d", "e"]
        )
        expected = concat([df0, df0[:2], df0[:1], df0], keys=["b", "c", "d", "e"])
        tm.assert_frame_equal(result, expected)

    def test_concat_bug_1719(self):
        ts1 = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        ts2 = ts1.copy()[::2]

        # to join with union
        # these two are of different length!
        left = concat([ts1, ts2], join="outer", axis=1)
        right = concat([ts2, ts1], join="outer", axis=1)

        assert len(left) == len(right)

    def test_concat_bug_2972(self):
        ts0 = Series(np.zeros(5))
        ts1 = Series(np.ones(5))
        ts0.name = ts1.name = "same name"
        result = concat([ts0, ts1], axis=1)

        expected = DataFrame({0: ts0, 1: ts1})
        expected.columns = ["same name", "same name"]
        tm.assert_frame_equal(result, expected)

    def test_concat_bug_3602(self):
        # GH 3602, duplicate columns
        df1 = DataFrame(
            {
                "firmNo": [0, 0, 0, 0],
                "prc": [6, 6, 6, 6],
                "stringvar": ["rrr", "rrr", "rrr", "rrr"],
            }
        )
        df2 = DataFrame(
            {"C": [9, 10, 11, 12], "misc": [1, 2, 3, 4], "prc": [6, 6, 6, 6]}
        )
        expected = DataFrame(
            [
                [0, 6, "rrr", 9, 1, 6],
                [0, 6, "rrr", 10, 2, 6],
                [0, 6, "rrr", 11, 3, 6],
                [0, 6, "rrr", 12, 4, 6],
            ]
        )
        expected.columns = ["firmNo", "prc", "stringvar", "C", "misc", "prc"]

        result = concat([df1, df2], axis=1)
        tm.assert_frame_equal(result, expected)

    def test_concat_iterables(self):
        # GH8645 check concat works with tuples, list, generators, and weird
        # stuff like deque and custom iterables
        df1 = DataFrame([1, 2, 3])
        df2 = DataFrame([4, 5, 6])
        expected = DataFrame([1, 2, 3, 4, 5, 6])
        tm.assert_frame_equal(concat((df1, df2), ignore_index=True), expected)
        tm.assert_frame_equal(concat([df1, df2], ignore_index=True), expected)
        tm.assert_frame_equal(
            concat((df for df in (df1, df2)), ignore_index=True), expected
        )
        tm.assert_frame_equal(concat(deque((df1, df2)), ignore_index=True), expected)

        class CustomIterator1:
            def __len__(self) -> int:
                return 2

            def __getitem__(self, index):
                try:
                    return {0: df1, 1: df2}[index]
                except KeyError as err:
                    raise IndexError from err

        tm.assert_frame_equal(concat(CustomIterator1(), ignore_index=True), expected)

        class CustomIterator2(abc.Iterable):
            def __iter__(self) -> Iterator:
                yield df1
                yield df2

        tm.assert_frame_equal(concat(CustomIterator2(), ignore_index=True), expected)

    def test_concat_order(self):
        # GH 17344, GH#47331
        dfs = [DataFrame(index=range(3), columns=["a", 1, None])]
        dfs += [DataFrame(index=range(3), columns=[None, 1, "a"]) for _ in range(100)]

        result = concat(dfs, sort=True).columns
        expected = Index([1, "a", None])
        tm.assert_index_equal(result, expected)

    def test_concat_different_extension_dtypes_upcasts(self):
        a = Series(pd.array([1, 2], dtype="Int64"))
        b = Series(to_decimal([1, 2]))

        result = concat([a, b], ignore_index=True)
        expected = Series([1, 2, Decimal(1), Decimal(2)], dtype=object)
        tm.assert_series_equal(result, expected)

    def test_concat_ordered_dict(self):
        # GH 21510
        expected = concat(
            [Series(range(3)), Series(range(4))], keys=["First", "Another"]
        )
        result = concat({"First": Series(range(3)), "Another": Series(range(4))})
        tm.assert_series_equal(result, expected)

    def test_concat_duplicate_indices_raise(self):
        # GH 45888: test raise for concat DataFrames with duplicate indices
        # https://github.com/pandas-dev/pandas/issues/36263
        df1 = DataFrame(
            np.random.default_rng(2).standard_normal(5),
            index=[0, 1, 2, 3, 3],
            columns=["a"],
        )
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal(5),
            index=[0, 1, 2, 2, 4],
            columns=["b"],
        )
        msg = "Reindexing only valid with uniquely valued Index objects"
        with pytest.raises(InvalidIndexError, match=msg):
            concat([df1, df2], axis=1)


def test_concat_no_unnecessary_upcast(float_numpy_dtype, frame_or_series):
    # GH 13247
    dims = frame_or_series(dtype=object).ndim
    dt = float_numpy_dtype

    dfs = [
        frame_or_series(np.array([1], dtype=dt, ndmin=dims)),
        frame_or_series(np.array([np.nan], dtype=dt, ndmin=dims)),
        frame_or_series(np.array([5], dtype=dt, ndmin=dims)),
    ]
    x = concat(dfs)
    assert x.values.dtype == dt


@pytest.mark.parametrize("pdt", [Series, DataFrame])
def test_concat_will_upcast(pdt, any_signed_int_numpy_dtype):
    dt = any_signed_int_numpy_dtype
    dims = pdt().ndim
    dfs = [
        pdt(np.array([1], dtype=dt, ndmin=dims)),
        pdt(np.array([np.nan], ndmin=dims)),
        pdt(np.array([5], dtype=dt, ndmin=dims)),
    ]
    x = concat(dfs)
    assert x.values.dtype == "float64"


def test_concat_empty_and_non_empty_frame_regression():
    # GH 18178 regression test
    df1 = DataFrame({"foo": [1]})
    df2 = DataFrame({"foo": []})
    expected = DataFrame({"foo": [1.0]})
    result = concat([df1, df2])
    tm.assert_frame_equal(result, expected)


def test_concat_sparse():
    # GH 23557
    a = Series(SparseArray([0, 1, 2]))
    expected = DataFrame(data=[[0, 0], [1, 1], [2, 2]]).astype(
        pd.SparseDtype(np.int64, 0)
    )
    result = concat([a, a], axis=1)
    tm.assert_frame_equal(result, expected)


def test_concat_dense_sparse():
    # GH 30668
    dtype = pd.SparseDtype(np.float64, None)
    a = Series(pd.arrays.SparseArray([1, None]), dtype=dtype)
    b = Series([1], dtype=float)
    expected = Series(data=[1, None, 1], index=[0, 1, 0]).astype(dtype)
    result = concat([a, b], axis=0)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("keys", [["e", "f", "f"], ["f", "e", "f"]])
def test_duplicate_keys(keys):
    # GH 33654
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    s1 = Series([7, 8, 9], name="c")
    s2 = Series([10, 11, 12], name="d")
    result = concat([df, s1, s2], axis=1, keys=keys)
    expected_values = [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]]
    expected_columns = MultiIndex.from_tuples(
        [(keys[0], "a"), (keys[0], "b"), (keys[1], "c"), (keys[2], "d")]
    )
    expected = DataFrame(expected_values, columns=expected_columns)
    tm.assert_frame_equal(result, expected)


def test_duplicate_keys_same_frame():
    # GH 43595
    keys = ["e", "e"]
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = concat([df, df], axis=1, keys=keys)
    expected_values = [[1, 4, 1, 4], [2, 5, 2, 5], [3, 6, 3, 6]]
    expected_columns = MultiIndex.from_tuples(
        [(keys[0], "a"), (keys[0], "b"), (keys[1], "a"), (keys[1], "b")]
    )
    expected = DataFrame(expected_values, columns=expected_columns)
    tm.assert_frame_equal(result, expected)


@pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning"
)
@pytest.mark.parametrize(
    "obj",
    [
        tm.SubclassedDataFrame({"A": np.arange(0, 10)}),
        tm.SubclassedSeries(np.arange(0, 10), name="A"),
    ],
)
def test_concat_preserves_subclass(obj):
    # GH28330 -- preserve subclass

    result = concat([obj, obj])
    assert isinstance(result, type(obj))


def test_concat_frame_axis0_extension_dtypes():
    # preserve extension dtype (through common_dtype mechanism)
    df1 = DataFrame({"a": pd.array([1, 2, 3], dtype="Int64")})
    df2 = DataFrame({"a": np.array([4, 5, 6])})

    result = concat([df1, df2], ignore_index=True)
    expected = DataFrame({"a": [1, 2, 3, 4, 5, 6]}, dtype="Int64")
    tm.assert_frame_equal(result, expected)

    result = concat([df2, df1], ignore_index=True)
    expected = DataFrame({"a": [4, 5, 6, 1, 2, 3]}, dtype="Int64")
    tm.assert_frame_equal(result, expected)


def test_concat_preserves_extension_int64_dtype():
    # GH 24768
    df_a = DataFrame({"a": [-1]}, dtype="Int64")
    df_b = DataFrame({"b": [1]}, dtype="Int64")
    result = concat([df_a, df_b], ignore_index=True)
    expected = DataFrame({"a": [-1, None], "b": [None, 1]}, dtype="Int64")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "dtype1,dtype2,expected_dtype",
    [
        ("bool", "bool", "bool"),
        ("boolean", "bool", "boolean"),
        ("bool", "boolean", "boolean"),
        ("boolean", "boolean", "boolean"),
    ],
)
def test_concat_bool_types(dtype1, dtype2, expected_dtype):
    # GH 42800
    ser1 = Series([True, False], dtype=dtype1)
    ser2 = Series([False, True], dtype=dtype2)
    result = concat([ser1, ser2], ignore_index=True)
    expected = Series([True, False, False, True], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("keys", "integrity"),
    [
        (["red"] * 3, True),
        (["red"] * 3, False),
        (["red", "blue", "red"], False),
        (["red", "blue", "red"], True),
    ],
)
def test_concat_repeated_keys(keys, integrity):
    # GH: 20816
    series_list = [Series({"a": 1}), Series({"b": 2}), Series({"c": 3})]
    result = concat(series_list, keys=keys, verify_integrity=integrity)
    tuples = list(zip(keys, ["a", "b", "c"]))
    expected = Series([1, 2, 3], index=MultiIndex.from_tuples(tuples))
    tm.assert_series_equal(result, expected)


def test_concat_null_object_with_dti():
    # GH#40841
    dti = pd.DatetimeIndex(
        ["2021-04-08 21:21:14+00:00"], dtype="datetime64[ns, UTC]", name="Time (UTC)"
    )
    right = DataFrame(data={"C": [0.5274]}, index=dti)

    idx = Index([None], dtype="object", name="Maybe Time (UTC)")
    left = DataFrame(data={"A": [None], "B": [np.nan]}, index=idx)

    result = concat([left, right], axis="columns")

    exp_index = Index([None, dti[0]], dtype=object)
    expected = DataFrame(
        {
            "A": np.array([None, np.nan], dtype=object),
            "B": [np.nan, np.nan],
            "C": [np.nan, 0.5274],
        },
        index=exp_index,
    )
    tm.assert_frame_equal(result, expected)


def test_concat_multiindex_with_empty_rangeindex():
    # GH#41234
    mi = MultiIndex.from_tuples([("B", 1), ("C", 1)])
    df1 = DataFrame([[1, 2]], columns=mi)
    df2 = DataFrame(index=[1], columns=pd.RangeIndex(0))

    result = concat([df1, df2])
    expected = DataFrame([[1, 2], [np.nan, np.nan]], columns=mi)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        Series(data=[1, 2]),
        DataFrame(
            data={
                "col1": [1, 2],
            }
        ),
        DataFrame(dtype=float),
        Series(dtype=float),
    ],
)
def test_concat_drop_attrs(data):
    # GH#41828
    df1 = data.copy()
    df1.attrs = {1: 1}
    df2 = data.copy()
    df2.attrs = {1: 2}
    df = concat([df1, df2])
    assert len(df.attrs) == 0


@pytest.mark.parametrize(
    "data",
    [
        Series(data=[1, 2]),
        DataFrame(
            data={
                "col1": [1, 2],
            }
        ),
        DataFrame(dtype=float),
        Series(dtype=float),
    ],
)
def test_concat_retain_attrs(data):
    # GH#41828
    df1 = data.copy()
    df1.attrs = {1: 1}
    df2 = data.copy()
    df2.attrs = {1: 1}
    df = concat([df1, df2])
    assert df.attrs[1] == 1


@td.skip_array_manager_invalid_test
@pytest.mark.parametrize("df_dtype", ["float64", "int64", "datetime64[ns]"])
@pytest.mark.parametrize("empty_dtype", [None, "float64", "object"])
def test_concat_ignore_empty_object_float(empty_dtype, df_dtype):
    # https://github.com/pandas-dev/pandas/issues/45637
    df = DataFrame({"foo": [1, 2], "bar": [1, 2]}, dtype=df_dtype)
    empty = DataFrame(columns=["foo", "bar"], dtype=empty_dtype)

    msg = "The behavior of DataFrame concatenation with empty or all-NA entries"
    warn = None
    if df_dtype == "datetime64[ns]" or (
        df_dtype == "float64" and empty_dtype != "float64"
    ):
        warn = FutureWarning
    with tm.assert_produces_warning(warn, match=msg):
        result = concat([empty, df])
    expected = df
    if df_dtype == "int64":
        # TODO what exact behaviour do we want for integer eventually?
        if empty_dtype == "float64":
            expected = df.astype("float64")
        else:
            expected = df.astype("object")
    tm.assert_frame_equal(result, expected)


@td.skip_array_manager_invalid_test
@pytest.mark.parametrize("df_dtype", ["float64", "int64", "datetime64[ns]"])
@pytest.mark.parametrize("empty_dtype", [None, "float64", "object"])
def test_concat_ignore_all_na_object_float(empty_dtype, df_dtype):
    df = DataFrame({"foo": [1, 2], "bar": [1, 2]}, dtype=df_dtype)
    empty = DataFrame({"foo": [np.nan], "bar": [np.nan]}, dtype=empty_dtype)

    if df_dtype == "int64":
        # TODO what exact behaviour do we want for integer eventually?
        if empty_dtype == "object":
            df_dtype = "object"
        else:
            df_dtype = "float64"

    msg = "The behavior of DataFrame concatenation with empty or all-NA entries"
    warn = None
    if empty_dtype != df_dtype and empty_dtype is not None:
        warn = FutureWarning
    elif df_dtype == "datetime64[ns]":
        warn = FutureWarning

    with tm.assert_produces_warning(warn, match=msg):
        result = concat([empty, df], ignore_index=True)

    expected = DataFrame({"foo": [np.nan, 1, 2], "bar": [np.nan, 1, 2]}, dtype=df_dtype)
    tm.assert_frame_equal(result, expected)


@td.skip_array_manager_invalid_test
def test_concat_ignore_empty_from_reindex():
    # https://github.com/pandas-dev/pandas/pull/43507#issuecomment-920375856
    df1 = DataFrame({"a": [1], "b": [pd.Timestamp("2012-01-01")]})
    df2 = DataFrame({"a": [2]})

    aligned = df2.reindex(columns=df1.columns)

    msg = "The behavior of DataFrame concatenation with empty or all-NA entries"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = concat([df1, aligned], ignore_index=True)
    expected = df1 = DataFrame({"a": [1, 2], "b": [pd.Timestamp("2012-01-01"), pd.NaT]})
    tm.assert_frame_equal(result, expected)


def test_concat_mismatched_keys_length():
    # GH#43485
    ser = Series(range(5))
    sers = [ser + n for n in range(4)]
    keys = ["A", "B", "C"]

    msg = r"The behavior of pd.concat with len\(keys\) != len\(objs\) is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        concat(sers, keys=keys, axis=1)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        concat(sers, keys=keys, axis=0)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        concat((x for x in sers), keys=(y for y in keys), axis=1)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        concat((x for x in sers), keys=(y for y in keys), axis=0)


def test_concat_multiindex_with_category():
    df1 = DataFrame(
        {
            "c1": Series(list("abc"), dtype="category"),
            "c2": Series(list("eee"), dtype="category"),
            "i2": Series([1, 2, 3]),
        }
    )
    df1 = df1.set_index(["c1", "c2"])
    df2 = DataFrame(
        {
            "c1": Series(list("abc"), dtype="category"),
            "c2": Series(list("eee"), dtype="category"),
            "i2": Series([4, 5, 6]),
        }
    )
    df2 = df2.set_index(["c1", "c2"])
    result = concat([df1, df2])
    expected = DataFrame(
        {
            "c1": Series(list("abcabc"), dtype="category"),
            "c2": Series(list("eeeeee"), dtype="category"),
            "i2": Series([1, 2, 3, 4, 5, 6]),
        }
    )
    expected = expected.set_index(["c1", "c2"])
    tm.assert_frame_equal(result, expected)


def test_concat_ea_upcast():
    # GH#54848
    df1 = DataFrame(["a"], dtype="string")
    df2 = DataFrame([1], dtype="Int64")
    result = concat([df1, df2])
    expected = DataFrame(["a", 1], index=[0, 0])
    tm.assert_frame_equal(result, expected)


def test_concat_none_with_timezone_timestamp():
    # GH#52093
    df1 = DataFrame([{"A": None}])
    df2 = DataFrame([{"A": pd.Timestamp("1990-12-20 00:00:00+00:00")}])
    msg = "The behavior of DataFrame concatenation with empty or all-NA entries"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = concat([df1, df2], ignore_index=True)
    expected = DataFrame({"A": [None, pd.Timestamp("1990-12-20 00:00:00+00:00")]})
    tm.assert_frame_equal(result, expected)