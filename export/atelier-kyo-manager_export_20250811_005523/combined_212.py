
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_align.py ===
from datetime import timezone

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    date_range,
)
import pandas._testing as tm


class TestDataFrameAlign:
    def test_align_asfreq_method_raises(self):
        df = DataFrame({"A": [1, np.nan, 2]})
        msg = "Invalid fill method"
        msg2 = "The 'method', 'limit', and 'fill_axis' keywords"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=msg2):
                df.align(df.iloc[::-1], method="asfreq")

    def test_frame_align_aware(self):
        idx1 = date_range("2001", periods=5, freq="h", tz="US/Eastern")
        idx2 = date_range("2001", periods=5, freq="2h", tz="US/Eastern")
        df1 = DataFrame(np.random.default_rng(2).standard_normal((len(idx1), 3)), idx1)
        df2 = DataFrame(np.random.default_rng(2).standard_normal((len(idx2), 3)), idx2)
        new1, new2 = df1.align(df2)
        assert df1.index.tz == new1.index.tz
        assert df2.index.tz == new2.index.tz

        # different timezones convert to UTC

        # frame with frame
        df1_central = df1.tz_convert("US/Central")
        new1, new2 = df1.align(df1_central)
        assert new1.index.tz is timezone.utc
        assert new2.index.tz is timezone.utc

        # frame with Series
        new1, new2 = df1.align(df1_central[0], axis=0)
        assert new1.index.tz is timezone.utc
        assert new2.index.tz is timezone.utc

        df1[0].align(df1_central, axis=0)
        assert new1.index.tz is timezone.utc
        assert new2.index.tz is timezone.utc

    def test_align_float(self, float_frame, using_copy_on_write):
        af, bf = float_frame.align(float_frame)
        assert af._mgr is not float_frame._mgr

        af, bf = float_frame.align(float_frame, copy=False)
        if not using_copy_on_write:
            assert af._mgr is float_frame._mgr
        else:
            assert af._mgr is not float_frame._mgr

        # axis = 0
        other = float_frame.iloc[:-5, :3]
        af, bf = float_frame.align(other, axis=0, fill_value=-1)

        tm.assert_index_equal(bf.columns, other.columns)

        # test fill value
        join_idx = float_frame.index.join(other.index)
        diff_a = float_frame.index.difference(join_idx)
        diff_a_vals = af.reindex(diff_a).values
        assert (diff_a_vals == -1).all()

        af, bf = float_frame.align(other, join="right", axis=0)
        tm.assert_index_equal(bf.columns, other.columns)
        tm.assert_index_equal(bf.index, other.index)
        tm.assert_index_equal(af.index, other.index)

        # axis = 1
        other = float_frame.iloc[:-5, :3].copy()
        af, bf = float_frame.align(other, axis=1)
        tm.assert_index_equal(bf.columns, float_frame.columns)
        tm.assert_index_equal(bf.index, other.index)

        # test fill value
        join_idx = float_frame.index.join(other.index)
        diff_a = float_frame.index.difference(join_idx)
        diff_a_vals = af.reindex(diff_a).values

        assert (diff_a_vals == -1).all()

        af, bf = float_frame.align(other, join="inner", axis=1)
        tm.assert_index_equal(bf.columns, other.columns)

        msg = (
            "The 'method', 'limit', and 'fill_axis' keywords in DataFrame.align "
            "are deprecated"
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            af, bf = float_frame.align(other, join="inner", axis=1, method="pad")
        tm.assert_index_equal(bf.columns, other.columns)

        msg = (
            "The 'method', 'limit', and 'fill_axis' keywords in DataFrame.align "
            "are deprecated"
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            af, bf = float_frame.align(
                other.iloc[:, 0], join="inner", axis=1, method=None, fill_value=None
            )
        tm.assert_index_equal(bf.index, Index([]).astype(bf.index.dtype))

        msg = (
            "The 'method', 'limit', and 'fill_axis' keywords in DataFrame.align "
            "are deprecated"
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            af, bf = float_frame.align(
                other.iloc[:, 0], join="inner", axis=1, method=None, fill_value=0
            )
        tm.assert_index_equal(bf.index, Index([]).astype(bf.index.dtype))

        # Try to align DataFrame to Series along bad axis
        msg = "No axis named 2 for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            float_frame.align(af.iloc[0, :3], join="inner", axis=2)

    def test_align_frame_with_series(self, float_frame):
        # align dataframe to series with broadcast or not
        idx = float_frame.index
        s = Series(range(len(idx)), index=idx)

        left, right = float_frame.align(s, axis=0)
        tm.assert_index_equal(left.index, float_frame.index)
        tm.assert_index_equal(right.index, float_frame.index)
        assert isinstance(right, Series)

        msg = "The 'broadcast_axis' keyword in DataFrame.align is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            left, right = float_frame.align(s, broadcast_axis=1)
        tm.assert_index_equal(left.index, float_frame.index)
        expected = {c: s for c in float_frame.columns}
        expected = DataFrame(
            expected, index=float_frame.index, columns=float_frame.columns
        )
        tm.assert_frame_equal(right, expected)

    def test_align_series_condition(self):
        # see gh-9558
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df[df["a"] == 2]
        expected = DataFrame([[2, 5]], index=[1], columns=["a", "b"])
        tm.assert_frame_equal(result, expected)

        result = df.where(df["a"] == 2, 0)
        expected = DataFrame({"a": [0, 2, 0], "b": [0, 5, 0]})
        tm.assert_frame_equal(result, expected)

    def test_align_int(self, int_frame):
        # test other non-float types
        other = DataFrame(index=range(5), columns=["A", "B", "C"])

        msg = (
            "The 'method', 'limit', and 'fill_axis' keywords in DataFrame.align "
            "are deprecated"
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            af, bf = int_frame.align(other, join="inner", axis=1, method="pad")
        tm.assert_index_equal(bf.columns, other.columns)

    def test_align_mixed_type(self, float_string_frame):
        msg = (
            "The 'method', 'limit', and 'fill_axis' keywords in DataFrame.align "
            "are deprecated"
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            af, bf = float_string_frame.align(
                float_string_frame, join="inner", axis=1, method="pad"
            )
        tm.assert_index_equal(bf.columns, float_string_frame.columns)

    def test_align_mixed_float(self, mixed_float_frame):
        # mixed floats/ints
        other = DataFrame(index=range(5), columns=["A", "B", "C"])

        msg = (
            "The 'method', 'limit', and 'fill_axis' keywords in DataFrame.align "
            "are deprecated"
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            af, bf = mixed_float_frame.align(
                other.iloc[:, 0], join="inner", axis=1, method=None, fill_value=0
            )
        tm.assert_index_equal(bf.index, Index([]))

    def test_align_mixed_int(self, mixed_int_frame):
        other = DataFrame(index=range(5), columns=["A", "B", "C"])

        msg = (
            "The 'method', 'limit', and 'fill_axis' keywords in DataFrame.align "
            "are deprecated"
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            af, bf = mixed_int_frame.align(
                other.iloc[:, 0], join="inner", axis=1, method=None, fill_value=0
            )
        tm.assert_index_equal(bf.index, Index([]))

    @pytest.mark.parametrize(
        "l_ordered,r_ordered,expected",
        [
            [True, True, pd.CategoricalIndex],
            [True, False, Index],
            [False, True, Index],
            [False, False, pd.CategoricalIndex],
        ],
    )
    def test_align_categorical(self, l_ordered, r_ordered, expected):
        # GH-28397
        df_1 = DataFrame(
            {
                "A": np.arange(6, dtype="int64"),
                "B": Series(list("aabbca")).astype(
                    pd.CategoricalDtype(list("cab"), ordered=l_ordered)
                ),
            }
        ).set_index("B")
        df_2 = DataFrame(
            {
                "A": np.arange(5, dtype="int64"),
                "B": Series(list("babca")).astype(
                    pd.CategoricalDtype(list("cab"), ordered=r_ordered)
                ),
            }
        ).set_index("B")

        aligned_1, aligned_2 = df_1.align(df_2)
        assert isinstance(aligned_1.index, expected)
        assert isinstance(aligned_2.index, expected)
        tm.assert_index_equal(aligned_1.index, aligned_2.index)

    def test_align_multiindex(self):
        # GH#10665
        # same test cases as test_align_multiindex in test_series.py

        midx = pd.MultiIndex.from_product(
            [range(2), range(3), range(2)], names=("a", "b", "c")
        )
        idx = Index(range(2), name="b")
        df1 = DataFrame(np.arange(12, dtype="int64"), index=midx)
        df2 = DataFrame(np.arange(2, dtype="int64"), index=idx)

        # these must be the same results (but flipped)
        res1l, res1r = df1.align(df2, join="left")
        res2l, res2r = df2.align(df1, join="right")

        expl = df1
        tm.assert_frame_equal(expl, res1l)
        tm.assert_frame_equal(expl, res2r)
        expr = DataFrame([0, 0, 1, 1, np.nan, np.nan] * 2, index=midx)
        tm.assert_frame_equal(expr, res1r)
        tm.assert_frame_equal(expr, res2l)

        res1l, res1r = df1.align(df2, join="right")
        res2l, res2r = df2.align(df1, join="left")

        exp_idx = pd.MultiIndex.from_product(
            [range(2), range(2), range(2)], names=("a", "b", "c")
        )
        expl = DataFrame([0, 1, 2, 3, 6, 7, 8, 9], index=exp_idx)
        tm.assert_frame_equal(expl, res1l)
        tm.assert_frame_equal(expl, res2r)
        expr = DataFrame([0, 0, 1, 1] * 2, index=exp_idx)
        tm.assert_frame_equal(expr, res1r)
        tm.assert_frame_equal(expr, res2l)

    def test_align_series_combinations(self):
        df = DataFrame({"a": [1, 3, 5], "b": [1, 3, 5]}, index=list("ACE"))
        s = Series([1, 2, 4], index=list("ABD"), name="x")

        # frame + series
        res1, res2 = df.align(s, axis=0)
        exp1 = DataFrame(
            {"a": [1, np.nan, 3, np.nan, 5], "b": [1, np.nan, 3, np.nan, 5]},
            index=list("ABCDE"),
        )
        exp2 = Series([1, 2, np.nan, 4, np.nan], index=list("ABCDE"), name="x")

        tm.assert_frame_equal(res1, exp1)
        tm.assert_series_equal(res2, exp2)

        # series + frame
        res1, res2 = s.align(df)
        tm.assert_series_equal(res1, exp2)
        tm.assert_frame_equal(res2, exp1)

    def test_multiindex_align_to_series_with_common_index_level(self):
        #  GH-46001
        foo_index = Index([1, 2, 3], name="foo")
        bar_index = Index([1, 2], name="bar")

        series = Series([1, 2], index=bar_index, name="foo_series")
        df = DataFrame(
            {"col": np.arange(6)},
            index=pd.MultiIndex.from_product([foo_index, bar_index]),
        )

        expected_r = Series([1, 2] * 3, index=df.index, name="foo_series")
        result_l, result_r = df.align(series, axis=0)

        tm.assert_frame_equal(result_l, df)
        tm.assert_series_equal(result_r, expected_r)

    def test_multiindex_align_to_series_with_common_index_level_missing_in_left(self):
        #  GH-46001
        foo_index = Index([1, 2, 3], name="foo")
        bar_index = Index([1, 2], name="bar")

        series = Series(
            [1, 2, 3, 4], index=Index([1, 2, 3, 4], name="bar"), name="foo_series"
        )
        df = DataFrame(
            {"col": np.arange(6)},
            index=pd.MultiIndex.from_product([foo_index, bar_index]),
        )

        expected_r = Series([1, 2] * 3, index=df.index, name="foo_series")
        result_l, result_r = df.align(series, axis=0)

        tm.assert_frame_equal(result_l, df)
        tm.assert_series_equal(result_r, expected_r)

    def test_multiindex_align_to_series_with_common_index_level_missing_in_right(self):
        #  GH-46001
        foo_index = Index([1, 2, 3], name="foo")
        bar_index = Index([1, 2, 3, 4], name="bar")

        series = Series([1, 2], index=Index([1, 2], name="bar"), name="foo_series")
        df = DataFrame(
            {"col": np.arange(12)},
            index=pd.MultiIndex.from_product([foo_index, bar_index]),
        )

        expected_r = Series(
            [1, 2, np.nan, np.nan] * 3, index=df.index, name="foo_series"
        )
        result_l, result_r = df.align(series, axis=0)

        tm.assert_frame_equal(result_l, df)
        tm.assert_series_equal(result_r, expected_r)

    def test_multiindex_align_to_series_with_common_index_level_missing_in_both(self):
        #  GH-46001
        foo_index = Index([1, 2, 3], name="foo")
        bar_index = Index([1, 3, 4], name="bar")

        series = Series(
            [1, 2, 3], index=Index([1, 2, 4], name="bar"), name="foo_series"
        )
        df = DataFrame(
            {"col": np.arange(9)},
            index=pd.MultiIndex.from_product([foo_index, bar_index]),
        )

        expected_r = Series([1, np.nan, 3] * 3, index=df.index, name="foo_series")
        result_l, result_r = df.align(series, axis=0)

        tm.assert_frame_equal(result_l, df)
        tm.assert_series_equal(result_r, expected_r)

    def test_multiindex_align_to_series_with_common_index_level_non_unique_cols(self):
        #  GH-46001
        foo_index = Index([1, 2, 3], name="foo")
        bar_index = Index([1, 2], name="bar")

        series = Series([1, 2], index=bar_index, name="foo_series")
        df = DataFrame(
            np.arange(18).reshape(6, 3),
            index=pd.MultiIndex.from_product([foo_index, bar_index]),
        )
        df.columns = ["cfoo", "cbar", "cfoo"]

        expected = Series([1, 2] * 3, index=df.index, name="foo_series")
        result_left, result_right = df.align(series, axis=0)

        tm.assert_series_equal(result_right, expected)
        tm.assert_index_equal(result_left.columns, df.columns)

    def test_missing_axis_specification_exception(self):
        df = DataFrame(np.arange(50).reshape((10, 5)))
        series = Series(np.arange(5))

        with pytest.raises(ValueError, match=r"axis=0 or 1"):
            df.align(series)

    @pytest.mark.parametrize("method", ["pad", "bfill"])
    @pytest.mark.parametrize("axis", [0, 1, None])
    @pytest.mark.parametrize("fill_axis", [0, 1])
    @pytest.mark.parametrize("how", ["inner", "outer", "left", "right"])
    @pytest.mark.parametrize(
        "left_slice",
        [
            [slice(4), slice(10)],
            [slice(0), slice(0)],
        ],
    )
    @pytest.mark.parametrize(
        "right_slice",
        [
            [slice(2, None), slice(6, None)],
            [slice(0), slice(0)],
        ],
    )
    @pytest.mark.parametrize("limit", [1, None])
    def test_align_fill_method(
        self, how, method, axis, fill_axis, float_frame, left_slice, right_slice, limit
    ):
        frame = float_frame
        left = frame.iloc[left_slice[0], left_slice[1]]
        right = frame.iloc[right_slice[0], right_slice[1]]

        msg = (
            "The 'method', 'limit', and 'fill_axis' keywords in DataFrame.align "
            "are deprecated"
        )

        with tm.assert_produces_warning(FutureWarning, match=msg):
            aa, ab = left.align(
                right,
                axis=axis,
                join=how,
                method=method,
                limit=limit,
                fill_axis=fill_axis,
            )

        join_index, join_columns = None, None

        ea, eb = left, right
        if axis is None or axis == 0:
            join_index = left.index.join(right.index, how=how)
            ea = ea.reindex(index=join_index)
            eb = eb.reindex(index=join_index)

        if axis is None or axis == 1:
            join_columns = left.columns.join(right.columns, how=how)
            ea = ea.reindex(columns=join_columns)
            eb = eb.reindex(columns=join_columns)

        msg = "DataFrame.fillna with 'method' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            ea = ea.fillna(axis=fill_axis, method=method, limit=limit)
            eb = eb.fillna(axis=fill_axis, method=method, limit=limit)

        tm.assert_frame_equal(aa, ea)
        tm.assert_frame_equal(ab, eb)

    def test_align_series_check_copy(self):
        # GH#
        df = DataFrame({0: [1, 2]})
        ser = Series([1], name=0)
        expected = ser.copy()
        result, other = df.align(ser, axis=1)
        ser.iloc[0] = 100
        tm.assert_series_equal(other, expected)

    def test_align_identical_different_object(self):
        # GH#51032
        df = DataFrame({"a": [1, 2]})
        ser = Series([3, 4])
        result, result2 = df.align(ser, axis=0)
        tm.assert_frame_equal(result, df)
        tm.assert_series_equal(result2, ser)
        assert df is not result
        assert ser is not result2

    def test_align_identical_different_object_columns(self):
        # GH#51032
        df = DataFrame({"a": [1, 2]})
        ser = Series([1], index=["a"])
        result, result2 = df.align(ser, axis=1)
        tm.assert_frame_equal(result, df)
        tm.assert_series_equal(result2, ser)
        assert df is not result
        assert ser is not result2

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\test_assert_series_equal.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Series,
)
import pandas._testing as tm


def _assert_series_equal_both(a, b, **kwargs):
    """
    Check that two Series equal.

    This check is performed commutatively.

    Parameters
    ----------
    a : Series
        The first Series to compare.
    b : Series
        The second Series to compare.
    kwargs : dict
        The arguments passed to `tm.assert_series_equal`.
    """
    tm.assert_series_equal(a, b, **kwargs)
    tm.assert_series_equal(b, a, **kwargs)


def _assert_not_series_equal(a, b, **kwargs):
    """
    Check that two Series are not equal.

    Parameters
    ----------
    a : Series
        The first Series to compare.
    b : Series
        The second Series to compare.
    kwargs : dict
        The arguments passed to `tm.assert_series_equal`.
    """
    try:
        tm.assert_series_equal(a, b, **kwargs)
        msg = "The two Series were equal when they shouldn't have been"

        pytest.fail(msg=msg)
    except AssertionError:
        pass


def _assert_not_series_equal_both(a, b, **kwargs):
    """
    Check that two Series are not equal.

    This check is performed commutatively.

    Parameters
    ----------
    a : Series
        The first Series to compare.
    b : Series
        The second Series to compare.
    kwargs : dict
        The arguments passed to `tm.assert_series_equal`.
    """
    _assert_not_series_equal(a, b, **kwargs)
    _assert_not_series_equal(b, a, **kwargs)


@pytest.mark.parametrize("data", [range(3), list("abc"), list("áàä")])
def test_series_equal(data):
    _assert_series_equal_both(Series(data), Series(data))


@pytest.mark.parametrize(
    "data1,data2",
    [
        (range(3), range(1, 4)),
        (list("abc"), list("xyz")),
        (list("áàä"), list("éèë")),
        (list("áàä"), list(b"aaa")),
        (range(3), range(4)),
    ],
)
def test_series_not_equal_value_mismatch(data1, data2):
    _assert_not_series_equal_both(Series(data1), Series(data2))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dtype": "float64"},  # dtype mismatch
        {"index": [1, 2, 4]},  # index mismatch
        {"name": "foo"},  # name mismatch
    ],
)
def test_series_not_equal_metadata_mismatch(kwargs):
    data = range(3)
    s1 = Series(data)

    s2 = Series(data, **kwargs)
    _assert_not_series_equal_both(s1, s2)


@pytest.mark.parametrize("data1,data2", [(0.12345, 0.12346), (0.1235, 0.1236)])
@pytest.mark.parametrize("dtype", ["float32", "float64", "Float32"])
@pytest.mark.parametrize("decimals", [0, 1, 2, 3, 5, 10])
def test_less_precise(data1, data2, dtype, decimals):
    rtol = 10**-decimals
    s1 = Series([data1], dtype=dtype)
    s2 = Series([data2], dtype=dtype)

    if decimals in (5, 10) or (decimals >= 3 and abs(data1 - data2) >= 0.0005):
        msg = "Series values are different"
        with pytest.raises(AssertionError, match=msg):
            tm.assert_series_equal(s1, s2, rtol=rtol)
    else:
        _assert_series_equal_both(s1, s2, rtol=rtol)


@pytest.mark.parametrize(
    "s1,s2,msg",
    [
        # Index
        (
            Series(["l1", "l2"], index=[1, 2]),
            Series(["l1", "l2"], index=[1.0, 2.0]),
            "Series\\.index are different",
        ),
        # MultiIndex
        (
            DataFrame.from_records(
                {"a": [1, 2], "b": [2.1, 1.5], "c": ["l1", "l2"]}, index=["a", "b"]
            ).c,
            DataFrame.from_records(
                {"a": [1.0, 2.0], "b": [2.1, 1.5], "c": ["l1", "l2"]}, index=["a", "b"]
            ).c,
            "MultiIndex level \\[0\\] are different",
        ),
    ],
)
def test_series_equal_index_dtype(s1, s2, msg, check_index_type):
    kwargs = {"check_index_type": check_index_type}

    if check_index_type:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_series_equal(s1, s2, **kwargs)
    else:
        tm.assert_series_equal(s1, s2, **kwargs)


@pytest.mark.parametrize("check_like", [True, False])
def test_series_equal_order_mismatch(check_like):
    s1 = Series([1, 2, 3], index=["a", "b", "c"])
    s2 = Series([3, 2, 1], index=["c", "b", "a"])

    if not check_like:  # Do not ignore index ordering.
        with pytest.raises(AssertionError, match="Series.index are different"):
            tm.assert_series_equal(s1, s2, check_like=check_like)
    else:
        _assert_series_equal_both(s1, s2, check_like=check_like)


@pytest.mark.parametrize("check_index", [True, False])
def test_series_equal_index_mismatch(check_index):
    s1 = Series([1, 2, 3], index=["a", "b", "c"])
    s2 = Series([1, 2, 3], index=["c", "b", "a"])

    if check_index:  # Do not ignore index.
        with pytest.raises(AssertionError, match="Series.index are different"):
            tm.assert_series_equal(s1, s2, check_index=check_index)
    else:
        _assert_series_equal_both(s1, s2, check_index=check_index)


def test_series_invalid_param_combination():
    left = Series(dtype=object)
    right = Series(dtype=object)
    with pytest.raises(
        ValueError, match="check_like must be False if check_index is False"
    ):
        tm.assert_series_equal(left, right, check_index=False, check_like=True)


def test_series_equal_length_mismatch(rtol):
    msg = """Series are different

Series length are different
\\[left\\]:  3, RangeIndex\\(start=0, stop=3, step=1\\)
\\[right\\]: 4, RangeIndex\\(start=0, stop=4, step=1\\)"""

    s1 = Series([1, 2, 3])
    s2 = Series([1, 2, 3, 4])

    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s1, s2, rtol=rtol)


def test_series_equal_numeric_values_mismatch(rtol):
    msg = """Series are different

Series values are different \\(33\\.33333 %\\)
\\[index\\]: \\[0, 1, 2\\]
\\[left\\]:  \\[1, 2, 3\\]
\\[right\\]: \\[1, 2, 4\\]"""

    s1 = Series([1, 2, 3])
    s2 = Series([1, 2, 4])

    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s1, s2, rtol=rtol)


def test_series_equal_categorical_values_mismatch(rtol, using_infer_string):
    if using_infer_string:
        msg = """Series are different

Series values are different \\(66\\.66667 %\\)
\\[index\\]: \\[0, 1, 2\\]
\\[left\\]:  \\['a', 'b', 'c'\\]
Categories \\(3, str\\): \\[a, b, c\\]
\\[right\\]: \\['a', 'c', 'b'\\]
Categories \\(3, str\\): \\[a, b, c\\]"""
    else:
        msg = """Series are different

Series values are different \\(66\\.66667 %\\)
\\[index\\]: \\[0, 1, 2\\]
\\[left\\]:  \\['a', 'b', 'c'\\]
Categories \\(3, object\\): \\['a', 'b', 'c'\\]
\\[right\\]: \\['a', 'c', 'b'\\]
Categories \\(3, object\\): \\['a', 'b', 'c'\\]"""

    s1 = Series(Categorical(["a", "b", "c"]))
    s2 = Series(Categorical(["a", "c", "b"]))

    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s1, s2, rtol=rtol)


def test_series_equal_datetime_values_mismatch(rtol):
    msg = """Series are different

Series values are different \\(100.0 %\\)
\\[index\\]: \\[0, 1, 2\\]
\\[left\\]:  \\[1514764800000000000, 1514851200000000000, 1514937600000000000\\]
\\[right\\]: \\[1549065600000000000, 1549152000000000000, 1549238400000000000\\]"""

    s1 = Series(pd.date_range("2018-01-01", periods=3, freq="D"))
    s2 = Series(pd.date_range("2019-02-02", periods=3, freq="D"))

    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s1, s2, rtol=rtol)


def test_series_equal_categorical_mismatch(check_categorical, using_infer_string):
    if using_infer_string:
        dtype = "str"
    else:
        dtype = "object"
    msg = f"""Attributes of Series are different

Attribute "dtype" are different
\\[left\\]:  CategoricalDtype\\(categories=\\['a', 'b'\\], ordered=False, \
categories_dtype={dtype}\\)
\\[right\\]: CategoricalDtype\\(categories=\\['a', 'b', 'c'\\], \
ordered=False, categories_dtype={dtype}\\)"""

    s1 = Series(Categorical(["a", "b"]))
    s2 = Series(Categorical(["a", "b"], categories=list("abc")))

    if check_categorical:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_series_equal(s1, s2, check_categorical=check_categorical)
    else:
        _assert_series_equal_both(s1, s2, check_categorical=check_categorical)


def test_assert_series_equal_extension_dtype_mismatch():
    # https://github.com/pandas-dev/pandas/issues/32747
    left = Series(pd.array([1, 2, 3], dtype="Int64"))
    right = left.astype(int)

    msg = """Attributes of Series are different

Attribute "dtype" are different
\\[left\\]:  Int64
\\[right\\]: int[32|64]"""

    tm.assert_series_equal(left, right, check_dtype=False)

    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(left, right, check_dtype=True)


def test_assert_series_equal_interval_dtype_mismatch():
    # https://github.com/pandas-dev/pandas/issues/32747
    left = Series([pd.Interval(0, 1)], dtype="interval")
    right = left.astype(object)

    msg = """Attributes of Series are different

Attribute "dtype" are different
\\[left\\]:  interval\\[int64, right\\]
\\[right\\]: object"""

    tm.assert_series_equal(left, right, check_dtype=False)

    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(left, right, check_dtype=True)


def test_series_equal_series_type():
    class MySeries(Series):
        pass

    s1 = Series([1, 2])
    s2 = Series([1, 2])
    s3 = MySeries([1, 2])

    tm.assert_series_equal(s1, s2, check_series_type=False)
    tm.assert_series_equal(s1, s2, check_series_type=True)

    tm.assert_series_equal(s1, s3, check_series_type=False)
    tm.assert_series_equal(s3, s1, check_series_type=False)

    with pytest.raises(AssertionError, match="Series classes are different"):
        tm.assert_series_equal(s1, s3, check_series_type=True)

    with pytest.raises(AssertionError, match="Series classes are different"):
        tm.assert_series_equal(s3, s1, check_series_type=True)


def test_series_equal_exact_for_nonnumeric():
    # https://github.com/pandas-dev/pandas/issues/35446
    s1 = Series(["a", "b"])
    s2 = Series(["a", "b"])
    s3 = Series(["b", "a"])

    tm.assert_series_equal(s1, s2, check_exact=True)
    tm.assert_series_equal(s2, s1, check_exact=True)

    msg = """Series are different

Series values are different \\(100\\.0 %\\)
\\[index\\]: \\[0, 1\\]
\\[left\\]:  \\[a, b\\]
\\[right\\]: \\[b, a\\]"""
    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s1, s3, check_exact=True)

    msg = """Series are different

Series values are different \\(100\\.0 %\\)
\\[index\\]: \\[0, 1\\]
\\[left\\]:  \\[b, a\\]
\\[right\\]: \\[a, b\\]"""
    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s3, s1, check_exact=True)


def test_assert_series_equal_ignore_extension_dtype_mismatch():
    # https://github.com/pandas-dev/pandas/issues/35715
    left = Series([1, 2, 3], dtype="Int64")
    right = Series([1, 2, 3], dtype="Int32")
    tm.assert_series_equal(left, right, check_dtype=False)


def test_assert_series_equal_ignore_extension_dtype_mismatch_cross_class():
    # https://github.com/pandas-dev/pandas/issues/35715
    left = Series([1, 2, 3], dtype="Int64")
    right = Series([1, 2, 3], dtype="int64")
    tm.assert_series_equal(left, right, check_dtype=False)


def test_allows_duplicate_labels():
    left = Series([1])
    right = Series([1]).set_flags(allows_duplicate_labels=False)
    tm.assert_series_equal(left, left)
    tm.assert_series_equal(right, right)
    tm.assert_series_equal(left, right, check_flags=False)
    tm.assert_series_equal(right, left, check_flags=False)

    with pytest.raises(AssertionError, match="<Flags"):
        tm.assert_series_equal(left, right)

    with pytest.raises(AssertionError, match="<Flags"):
        tm.assert_series_equal(left, right)


def test_assert_series_equal_identical_na(nulls_fixture):
    ser = Series([nulls_fixture])

    tm.assert_series_equal(ser, ser.copy())

    # while we're here do Index too
    idx = pd.Index(ser)
    tm.assert_index_equal(idx, idx.copy(deep=True))


def test_identical_nested_series_is_equal():
    # GH#22400
    x = Series(
        [
            0,
            0.0131142231938,
            1.77774652865e-05,
            np.array([0.4722720840328748, 0.4216929783681722]),
        ]
    )
    y = Series(
        [
            0,
            0.0131142231938,
            1.77774652865e-05,
            np.array([0.4722720840328748, 0.4216929783681722]),
        ]
    )
    # These two arrays should be equal, nesting could cause issue

    tm.assert_series_equal(x, x)
    tm.assert_series_equal(x, x, check_exact=True)
    tm.assert_series_equal(x, y)
    tm.assert_series_equal(x, y, check_exact=True)


@pytest.mark.parametrize("dtype", ["datetime64", "timedelta64"])
def test_check_dtype_false_different_reso(dtype):
    # GH 52449
    ser_s = Series([1000213, 2131232, 21312331]).astype(f"{dtype}[s]")
    ser_ms = ser_s.astype(f"{dtype}[ms]")
    with pytest.raises(AssertionError, match="Attributes of Series are different"):
        tm.assert_series_equal(ser_s, ser_ms)
    tm.assert_series_equal(ser_ms, ser_s, check_dtype=False)

    ser_ms -= Series([1, 1, 1]).astype(f"{dtype}[ms]")

    with pytest.raises(AssertionError, match="Series are different"):
        tm.assert_series_equal(ser_s, ser_ms)

    with pytest.raises(AssertionError, match="Series are different"):
        tm.assert_series_equal(ser_s, ser_ms, check_dtype=False)


@pytest.mark.parametrize("dtype", ["Int64", "int64"])
def test_large_unequal_ints(dtype):
    # https://github.com/pandas-dev/pandas/issues/55882
    left = Series([1577840521123000], dtype=dtype)
    right = Series([1577840521123543], dtype=dtype)
    with pytest.raises(AssertionError, match="Series are different"):
        tm.assert_series_equal(left, right)


@pytest.mark.parametrize("dtype", [None, object])
@pytest.mark.parametrize("check_exact", [True, False])
@pytest.mark.parametrize("val", [3, 3.5])
def test_ea_and_numpy_no_dtype_check(val, check_exact, dtype):
    # GH#56651
    left = Series([1, 2, val], dtype=dtype)
    right = Series(pd.array([1, 2, val]))
    tm.assert_series_equal(left, right, check_dtype=False, check_exact=check_exact)


def test_assert_series_equal_int_tol():
    # GH#56646
    left = Series([81, 18, 121, 38, 74, 72, 81, 81, 146, 81, 81, 170, 74, 74])
    right = Series([72, 9, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72])
    tm.assert_series_equal(left, right, rtol=1.5)

    tm.assert_frame_equal(left.to_frame(), right.to_frame(), rtol=1.5)
    tm.assert_extension_array_equal(
        left.astype("Int64").values, right.astype("Int64").values, rtol=1.5
    )


def test_assert_series_equal_index_exact_default():
    # GH#57067
    ser1 = Series(np.zeros(6, dtype=int), [0, 0.2, 0.4, 0.6, 0.8, 1])
    ser2 = Series(np.zeros(6, dtype=int), np.linspace(0, 1, 6))
    tm.assert_series_equal(ser1, ser2)
    tm.assert_frame_equal(ser1.to_frame(), ser2.to_frame())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\geometry\tests\test_point.py ===
from sympy.core.basic import Basic
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.parameters import evaluate
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.geometry import Line, Point, Point2D, Point3D, Line3D, Plane
from sympy.geometry.entity import rotate, scale, translate, GeometryEntity
from sympy.matrices import Matrix
from sympy.utilities.iterables import subsets, permutations, cartes
from sympy.utilities.misc import Undecidable
from sympy.testing.pytest import raises, warns


def test_point():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    x1 = Symbol('x1', real=True)
    x2 = Symbol('x2', real=True)
    y1 = Symbol('y1', real=True)
    y2 = Symbol('y2', real=True)
    half = S.Half
    p1 = Point(x1, x2)
    p2 = Point(y1, y2)
    p3 = Point(0, 0)
    p4 = Point(1, 1)
    p5 = Point(0, 1)
    line = Line(Point(1, 0), slope=1)

    assert p1 in p1
    assert p1 not in p2
    assert p2.y == y2
    assert (p3 + p4) == p4
    assert (p2 - p1) == Point(y1 - x1, y2 - x2)
    assert -p2 == Point(-y1, -y2)
    raises(TypeError, lambda: Point(1))
    raises(ValueError, lambda: Point([1]))
    raises(ValueError, lambda: Point(3, I))
    raises(ValueError, lambda: Point(2*I, I))
    raises(ValueError, lambda: Point(3 + I, I))

    assert Point(34.05, sqrt(3)) == Point(Rational(681, 20), sqrt(3))
    assert Point.midpoint(p3, p4) == Point(half, half)
    assert Point.midpoint(p1, p4) == Point(half + half*x1, half + half*x2)
    assert Point.midpoint(p2, p2) == p2
    assert p2.midpoint(p2) == p2
    assert p1.origin == Point(0, 0)

    assert Point.distance(p3, p4) == sqrt(2)
    assert Point.distance(p1, p1) == 0
    assert Point.distance(p3, p2) == sqrt(p2.x**2 + p2.y**2)
    raises(TypeError, lambda: Point.distance(p1, 0))
    raises(TypeError, lambda: Point.distance(p1, GeometryEntity()))

    # distance should be symmetric
    assert p1.distance(line) == line.distance(p1)
    assert p4.distance(line) == line.distance(p4)

    assert Point.taxicab_distance(p4, p3) == 2

    assert Point.canberra_distance(p4, p5) == 1
    raises(ValueError, lambda: Point.canberra_distance(p3, p3))

    p1_1 = Point(x1, x1)
    p1_2 = Point(y2, y2)
    p1_3 = Point(x1 + 1, x1)
    assert Point.is_collinear(p3)

    with warns(UserWarning, test_stacklevel=False):
        assert Point.is_collinear(p3, Point(p3, dim=4))
    assert p3.is_collinear()
    assert Point.is_collinear(p3, p4)
    assert Point.is_collinear(p3, p4, p1_1, p1_2)
    assert Point.is_collinear(p3, p4, p1_1, p1_3) is False
    assert Point.is_collinear(p3, p3, p4, p5) is False

    raises(TypeError, lambda: Point.is_collinear(line))
    raises(TypeError, lambda: p1_1.is_collinear(line))

    assert p3.intersection(Point(0, 0)) == [p3]
    assert p3.intersection(p4) == []
    assert p3.intersection(line) == []
    with warns(UserWarning, test_stacklevel=False):
        assert Point.intersection(Point(0, 0, 0), Point(0, 0)) == [Point(0, 0, 0)]

    x_pos = Symbol('x', positive=True)
    p2_1 = Point(x_pos, 0)
    p2_2 = Point(0, x_pos)
    p2_3 = Point(-x_pos, 0)
    p2_4 = Point(0, -x_pos)
    p2_5 = Point(x_pos, 5)
    assert Point.is_concyclic(p2_1)
    assert Point.is_concyclic(p2_1, p2_2)
    assert Point.is_concyclic(p2_1, p2_2, p2_3, p2_4)
    for pts in permutations((p2_1, p2_2, p2_3, p2_5)):
        assert Point.is_concyclic(*pts) is False
    assert Point.is_concyclic(p4, p4 * 2, p4 * 3) is False
    assert Point(0, 0).is_concyclic((1, 1), (2, 2), (2, 1)) is False
    assert Point.is_concyclic(Point(0, 0, 0, 0), Point(1, 0, 0, 0), Point(1, 1, 0, 0), Point(1, 1, 1, 0)) is False

    assert p1.is_scalar_multiple(p1)
    assert p1.is_scalar_multiple(2*p1)
    assert not p1.is_scalar_multiple(p2)
    assert Point.is_scalar_multiple(Point(1, 1), (-1, -1))
    assert Point.is_scalar_multiple(Point(0, 0), (0, -1))
    # test when is_scalar_multiple can't be determined
    raises(Undecidable, lambda: Point.is_scalar_multiple(Point(sympify("x1%y1"), sympify("x2%y2")), Point(0, 1)))

    assert Point(0, 1).orthogonal_direction == Point(1, 0)
    assert Point(1, 0).orthogonal_direction == Point(0, 1)

    assert p1.is_zero is None
    assert p3.is_zero
    assert p4.is_zero is False
    assert p1.is_nonzero is None
    assert p3.is_nonzero is False
    assert p4.is_nonzero

    assert p4.scale(2, 3) == Point(2, 3)
    assert p3.scale(2, 3) == p3

    assert p4.rotate(pi, Point(0.5, 0.5)) == p3
    assert p1.__radd__(p2) == p1.midpoint(p2).scale(2, 2)
    assert (-p3).__rsub__(p4) == p3.midpoint(p4).scale(2, 2)

    assert p4 * 5 == Point(5, 5)
    assert p4 / 5 == Point(0.2, 0.2)
    assert 5 * p4 == Point(5, 5)

    raises(ValueError, lambda: Point(0, 0) + 10)

    # Point differences should be simplified
    assert Point(x*(x - 1), y) - Point(x**2 - x, y + 1) == Point(0, -1)

    a, b = S.Half, Rational(1, 3)
    assert Point(a, b).evalf(2) == \
        Point(a.n(2), b.n(2), evaluate=False)
    raises(ValueError, lambda: Point(1, 2) + 1)

    # test project
    assert Point.project((0, 1), (1, 0)) == Point(0, 0)
    assert Point.project((1, 1), (1, 0)) == Point(1, 0)
    raises(ValueError, lambda: Point.project(p1, Point(0, 0)))

    # test transformations
    p = Point(1, 0)
    assert p.rotate(pi/2) == Point(0, 1)
    assert p.rotate(pi/2, p) == p
    p = Point(1, 1)
    assert p.scale(2, 3) == Point(2, 3)
    assert p.translate(1, 2) == Point(2, 3)
    assert p.translate(1) == Point(2, 1)
    assert p.translate(y=1) == Point(1, 2)
    assert p.translate(*p.args) == Point(2, 2)

    # Check invalid input for transform
    raises(ValueError, lambda: p3.transform(p3))
    raises(ValueError, lambda: p.transform(Matrix([[1, 0], [0, 1]])))

    # test __contains__
    assert 0 in Point(0, 0, 0, 0)
    assert 1 not in Point(0, 0, 0, 0)

    # test affine_rank
    assert Point.affine_rank() == -1


def test_point3D():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    x1 = Symbol('x1', real=True)
    x2 = Symbol('x2', real=True)
    x3 = Symbol('x3', real=True)
    y1 = Symbol('y1', real=True)
    y2 = Symbol('y2', real=True)
    y3 = Symbol('y3', real=True)
    half = S.Half
    p1 = Point3D(x1, x2, x3)
    p2 = Point3D(y1, y2, y3)
    p3 = Point3D(0, 0, 0)
    p4 = Point3D(1, 1, 1)
    p5 = Point3D(0, 1, 2)

    assert p1 in p1
    assert p1 not in p2
    assert p2.y == y2
    assert (p3 + p4) == p4
    assert (p2 - p1) == Point3D(y1 - x1, y2 - x2, y3 - x3)
    assert -p2 == Point3D(-y1, -y2, -y3)

    assert Point(34.05, sqrt(3)) == Point(Rational(681, 20), sqrt(3))
    assert Point3D.midpoint(p3, p4) == Point3D(half, half, half)
    assert Point3D.midpoint(p1, p4) == Point3D(half + half*x1, half + half*x2,
                                         half + half*x3)
    assert Point3D.midpoint(p2, p2) == p2
    assert p2.midpoint(p2) == p2

    assert Point3D.distance(p3, p4) == sqrt(3)
    assert Point3D.distance(p1, p1) == 0
    assert Point3D.distance(p3, p2) == sqrt(p2.x**2 + p2.y**2 + p2.z**2)

    p1_1 = Point3D(x1, x1, x1)
    p1_2 = Point3D(y2, y2, y2)
    p1_3 = Point3D(x1 + 1, x1, x1)
    Point3D.are_collinear(p3)
    assert Point3D.are_collinear(p3, p4)
    assert Point3D.are_collinear(p3, p4, p1_1, p1_2)
    assert Point3D.are_collinear(p3, p4, p1_1, p1_3) is False
    assert Point3D.are_collinear(p3, p3, p4, p5) is False

    assert p3.intersection(Point3D(0, 0, 0)) == [p3]
    assert p3.intersection(p4) == []


    assert p4 * 5 == Point3D(5, 5, 5)
    assert p4 / 5 == Point3D(0.2, 0.2, 0.2)
    assert 5 * p4 == Point3D(5, 5, 5)

    raises(ValueError, lambda: Point3D(0, 0, 0) + 10)

    # Test coordinate properties
    assert p1.coordinates == (x1, x2, x3)
    assert p2.coordinates == (y1, y2, y3)
    assert p3.coordinates == (0, 0, 0)
    assert p4.coordinates == (1, 1, 1)
    assert p5.coordinates == (0, 1, 2)
    assert p5.x == 0
    assert p5.y == 1
    assert p5.z == 2

    # Point differences should be simplified
    assert Point3D(x*(x - 1), y, 2) - Point3D(x**2 - x, y + 1, 1) == \
        Point3D(0, -1, 1)

    a, b, c = S.Half, Rational(1, 3), Rational(1, 4)
    assert Point3D(a, b, c).evalf(2) == \
        Point(a.n(2), b.n(2), c.n(2), evaluate=False)
    raises(ValueError, lambda: Point3D(1, 2, 3) + 1)

    # test transformations
    p = Point3D(1, 1, 1)
    assert p.scale(2, 3) == Point3D(2, 3, 1)
    assert p.translate(1, 2) == Point3D(2, 3, 1)
    assert p.translate(1) == Point3D(2, 1, 1)
    assert p.translate(z=1) == Point3D(1, 1, 2)
    assert p.translate(*p.args) == Point3D(2, 2, 2)

    # Test __new__
    assert Point3D(0.1, 0.2, evaluate=False, on_morph='ignore').args[0].is_Float

    # Test length property returns correctly
    assert p.length == 0
    assert p1_1.length == 0
    assert p1_2.length == 0

    # Test are_colinear type error
    raises(TypeError, lambda: Point3D.are_collinear(p, x))

    # Test are_coplanar
    assert Point.are_coplanar()
    assert Point.are_coplanar((1, 2, 0), (1, 2, 0), (1, 3, 0))
    assert Point.are_coplanar((1, 2, 0), (1, 2, 3))
    with warns(UserWarning, test_stacklevel=False):
        raises(ValueError, lambda: Point2D.are_coplanar((1, 2), (1, 2, 3)))
    assert Point3D.are_coplanar((1, 2, 0), (1, 2, 3))
    assert Point.are_coplanar((0, 0, 0), (1, 1, 0), (1, 1, 1), (1, 2, 1)) is False
    planar2 = Point3D(1, -1, 1)
    planar3 = Point3D(-1, 1, 1)
    assert Point3D.are_coplanar(p, planar2, planar3) == True
    assert Point3D.are_coplanar(p, planar2, planar3, p3) == False
    assert Point.are_coplanar(p, planar2)
    planar2 = Point3D(1, 1, 2)
    planar3 = Point3D(1, 1, 3)
    assert Point3D.are_coplanar(p, planar2, planar3)  # line, not plane
    plane = Plane((1, 2, 1), (2, 1, 0), (3, 1, 2))
    assert Point.are_coplanar(*[plane.projection(((-1)**i, i)) for i in range(4)])

    # all 2D points are coplanar
    assert Point.are_coplanar(Point(x, y), Point(x, x + y), Point(y, x + 2)) is True

    # Test Intersection
    assert planar2.intersection(Line3D(p, planar3)) == [Point3D(1, 1, 2)]

    # Test Scale
    assert planar2.scale(1, 1, 1) == planar2
    assert planar2.scale(2, 2, 2, planar3) == Point3D(1, 1, 1)
    assert planar2.scale(1, 1, 1, p3) == planar2

    # Test Transform
    identity = Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert p.transform(identity) == p
    trans = Matrix([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]])
    assert p.transform(trans) == Point3D(2, 2, 2)
    raises(ValueError, lambda: p.transform(p))
    raises(ValueError, lambda: p.transform(Matrix([[1, 0], [0, 1]])))

    # Test Equals
    assert p.equals(x1) == False

    # Test __sub__
    p_4d = Point(0, 0, 0, 1)
    with warns(UserWarning, test_stacklevel=False):
        assert p - p_4d == Point(1, 1, 1, -1)
    p_4d3d = Point(0, 0, 1, 0)
    with warns(UserWarning, test_stacklevel=False):
        assert p - p_4d3d == Point(1, 1, 0, 0)


def test_Point2D():

    # Test Distance
    p1 = Point2D(1, 5)
    p2 = Point2D(4, 2.5)
    p3 = (6, 3)
    assert p1.distance(p2) == sqrt(61)/2
    assert p2.distance(p3) == sqrt(17)/2

    # Test coordinates
    assert p1.x == 1
    assert p1.y == 5
    assert p2.x == 4
    assert p2.y == S(5)/2
    assert p1.coordinates == (1, 5)
    assert p2.coordinates == (4, S(5)/2)

    # test bounds
    assert p1.bounds == (1, 5, 1, 5)

def test_issue_9214():
    p1 = Point3D(4, -2, 6)
    p2 = Point3D(1, 2, 3)
    p3 = Point3D(7, 2, 3)

    assert Point3D.are_collinear(p1, p2, p3) is False


def test_issue_11617():
    p1 = Point3D(1,0,2)
    p2 = Point2D(2,0)

    with warns(UserWarning, test_stacklevel=False):
        assert p1.distance(p2) == sqrt(5)


def test_transform():
    p = Point(1, 1)
    assert p.transform(rotate(pi/2)) == Point(-1, 1)
    assert p.transform(scale(3, 2)) == Point(3, 2)
    assert p.transform(translate(1, 2)) == Point(2, 3)
    assert Point(1, 1).scale(2, 3, (4, 5)) == \
        Point(-2, -7)
    assert Point(1, 1).translate(4, 5) == \
        Point(5, 6)


def test_concyclic_doctest_bug():
    p1, p2 = Point(-1, 0), Point(1, 0)
    p3, p4 = Point(0, 1), Point(-1, 2)
    assert Point.is_concyclic(p1, p2, p3)
    assert not Point.is_concyclic(p1, p2, p3, p4)


def test_arguments():
    """Functions accepting `Point` objects in `geometry`
    should also accept tuples and lists and
    automatically convert them to points."""

    singles2d = ((1,2), [1,2], Point(1,2))
    singles2d2 = ((1,3), [1,3], Point(1,3))
    doubles2d = cartes(singles2d, singles2d2)
    p2d = Point2D(1,2)
    singles3d = ((1,2,3), [1,2,3], Point(1,2,3))
    doubles3d = subsets(singles3d, 2)
    p3d = Point3D(1,2,3)
    singles4d = ((1,2,3,4), [1,2,3,4], Point(1,2,3,4))
    doubles4d = subsets(singles4d, 2)
    p4d = Point(1,2,3,4)

    # test 2D
    test_single = ['distance', 'is_scalar_multiple', 'taxicab_distance', 'midpoint', 'intersection', 'dot', 'equals', '__add__', '__sub__']
    test_double = ['is_concyclic', 'is_collinear']
    for p in singles2d:
        Point2D(p)
    for func in test_single:
        for p in singles2d:
            getattr(p2d, func)(p)
    for func in test_double:
        for p in doubles2d:
            getattr(p2d, func)(*p)

    # test 3D
    test_double = ['is_collinear']
    for p in singles3d:
        Point3D(p)
    for func in test_single:
        for p in singles3d:
            getattr(p3d, func)(p)
    for func in test_double:
        for p in doubles3d:
            getattr(p3d, func)(*p)

    # test 4D
    test_double = ['is_collinear']
    for p in singles4d:
        Point(p)
    for func in test_single:
        for p in singles4d:
            getattr(p4d, func)(p)
    for func in test_double:
        for p in doubles4d:
            getattr(p4d, func)(*p)

    # test evaluate=False for ops
    x = Symbol('x')
    a = Point(0, 1)
    assert a + (0.1, x) == Point(0.1, 1 + x, evaluate=False)
    a = Point(0, 1)
    assert a/10.0 == Point(0, 0.1, evaluate=False)
    a = Point(0, 1)
    assert a*10.0 == Point(0, 10.0, evaluate=False)

    # test evaluate=False when changing dimensions
    u = Point(.1, .2, evaluate=False)
    u4 = Point(u, dim=4, on_morph='ignore')
    assert u4.args == (.1, .2, 0, 0)
    assert all(i.is_Float for i in u4.args[:2])
    # and even when *not* changing dimensions
    assert all(i.is_Float for i in Point(u).args)

    # never raise error if creating an origin
    assert Point(dim=3, on_morph='error')

    # raise error with unmatched dimension
    raises(ValueError, lambda: Point(1, 1, dim=3, on_morph='error'))
    # test unknown on_morph
    raises(ValueError, lambda: Point(1, 1, dim=3, on_morph='unknown'))
    # test invalid expressions
    raises(TypeError, lambda: Point(Basic(), Basic()))

def test_unit():
    assert Point(1, 1).unit == Point(sqrt(2)/2, sqrt(2)/2)


def test_dot():
    raises(TypeError, lambda: Point(1, 2).dot(Line((0, 0), (1, 1))))


def test__normalize_dimension():
    assert Point._normalize_dimension(Point(1, 2), Point(3, 4)) == [
        Point(1, 2), Point(3, 4)]
    assert Point._normalize_dimension(
        Point(1, 2), Point(3, 4, 0), on_morph='ignore') == [
        Point(1, 2, 0), Point(3, 4, 0)]


def test_issue_22684():
    # Used to give an error
    with evaluate(False):
        Point(1, 2)


def test_direction_cosine():
    p1 = Point3D(0, 0, 0)
    p2 = Point3D(1, 1, 1)

    assert p1.direction_cosine(Point3D(1, 0, 0)) == [1, 0, 0]
    assert p1.direction_cosine(Point3D(0, 1, 0)) == [0, 1, 0]
    assert p1.direction_cosine(Point3D(0, 0, pi)) == [0, 0, 1]

    assert p1.direction_cosine(Point3D(5, 0, 0)) == [1, 0, 0]
    assert p1.direction_cosine(Point3D(0, sqrt(3), 0)) == [0, 1, 0]
    assert p1.direction_cosine(Point3D(0, 0, 5)) == [0, 0, 1]

    assert p1.direction_cosine(Point3D(2.4, 2.4, 0)) == [sqrt(2)/2, sqrt(2)/2, 0]
    assert p1.direction_cosine(Point3D(1, 1, 1)) == [sqrt(3) / 3, sqrt(3) / 3, sqrt(3) / 3]
    assert p1.direction_cosine(Point3D(-12, 0 -15)) == [-4*sqrt(41)/41, -5*sqrt(41)/41, 0]

    assert p2.direction_cosine(Point3D(0, 0, 0)) == [-sqrt(3) / 3, -sqrt(3) / 3, -sqrt(3) / 3]
    assert p2.direction_cosine(Point3D(1, 1, 12)) == [0, 0, 1]
    assert p2.direction_cosine(Point3D(12, 1, 12)) == [sqrt(2) / 2, 0, sqrt(2) / 2]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\cffi\__init__.py ===
__all__ = ['FFI', 'VerificationError', 'VerificationMissing', 'CDefError',
           'FFIError']

from .api import FFI
from .error import CDefError, FFIError, VerificationError, VerificationMissing
from .error import PkgConfigError

__version__ = "1.17.1"
__version_info__ = (1, 17, 1)

# The verifier module file names are based on the CRC32 of a string that
# contains the following version number.  It may be older than __version__
# if nothing is clearly incompatible.
__version_verifier_modules__ = "0.8.6"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\functions\__init__.py ===
from . import functions
# Hack to update methods
from . import factorials
from . import hypergeometric
from . import expintegrals
from . import bessel
from . import orthogonal
from . import theta
from . import elliptic
from . import signals
from . import zeta
from . import rszeta
from . import zetazeros
from . import qfunctions

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\worksheet\drawing.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors.excel import Relation


class Drawing(Serialisable):

    tagname = "drawing"

    id = Relation()

    def __init__(self, id=None):
        self.id = id

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\requests\__version__.py ===
# .-. .-. .-. . . .-. .-. .-. .-.
# |(  |-  |.| | | |-  `-.  |  `-.
# ' ' `-' `-`.`-' `-' `-'  '  `-'

__title__ = "requests"
__description__ = "Python HTTP for Humans."
__url__ = "https://requests.readthedocs.io"
__version__ = "2.32.3"
__build__ = 0x023203
__author__ = "Kenneth Reitz"
__author_email__ = "me@kennethreitz.org"
__license__ = "Apache-2.0"
__copyright__ = "Copyright Kenneth Reitz"
__cake__ = "\u2728 \U0001f370 \u2728"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\keysyms\__init__.py ===
from pyreadline3.py3k_compat import is_ironpython

from . import winconstants

if is_ironpython:
    try:
        from .ironpython_keysyms import *
    except ImportError as x:
        raise ImportError("Could not import keysym for local ironpython version") from x
else:
    try:
        from .keysyms import *
    except ImportError as x:
        raise ImportError("Could not import keysym for local python version") from x

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\requests\__version__.py ===
# .-. .-. .-. . . .-. .-. .-. .-.
# |(  |-  |.| | | |-  `-.  |  `-.
# ' ' `-' `-`.`-' `-' `-'  '  `-'

__title__ = "requests"
__description__ = "Python HTTP for Humans."
__url__ = "https://requests.readthedocs.io"
__version__ = "2.32.4"
__build__ = 0x023204
__author__ = "Kenneth Reitz"
__author_email__ = "me@kennethreitz.org"
__license__ = "Apache-2.0"
__copyright__ = "Copyright Kenneth Reitz"
__cake__ = "\u2728 \U0001f370 \u2728"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\cxxnodes.py ===
"""
AST nodes specific to C++.
"""

from sympy.codegen.ast import Attribute, String, Token, Type, none

class using(Token):
    """ Represents a 'using' statement in C++ """
    __slots__ = _fields = ('type', 'alias')
    defaults = {'alias': none}
    _construct_type = Type
    _construct_alias = String

constexpr = Attribute('constexpr')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\tests\test_rewriting.py ===
import tempfile
from sympy.core.numbers import pi, Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import (cos, sin, sinc)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.assumptions import assuming, Q
from sympy.external import import_module
from sympy.printing.codeprinter import ccode
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.codegen.cfunctions import log2, exp2, expm1, log1p
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2
from sympy.codegen.scipy_nodes import cosm1, powm1
from sympy.codegen.rewriting import (
    optimize, cosm1_opt, log2_opt, exp2_opt, expm1_opt, log1p_opt, powm1_opt, optims_c99,
    create_expand_pow_optimization, matinv_opt, logaddexp_opt, logaddexp2_opt,
    optims_numpy, optims_scipy, sinc_opts, FuncMinusOneOptim
)
from sympy.testing.pytest import XFAIL, skip
from sympy.utilities import lambdify
from sympy.utilities._compilation import compile_link_import_strings, has_c
from sympy.utilities._compilation.util import may_xfail

cython = import_module('cython')
numpy = import_module('numpy')
scipy = import_module('scipy')


def test_log2_opt():
    x = Symbol('x')
    expr1 = 7*log(3*x + 5)/(log(2))
    opt1 = optimize(expr1, [log2_opt])
    assert opt1 == 7*log2(3*x + 5)
    assert opt1.rewrite(log) == expr1

    expr2 = 3*log(5*x + 7)/(13*log(2))
    opt2 = optimize(expr2, [log2_opt])
    assert opt2 == 3*log2(5*x + 7)/13
    assert opt2.rewrite(log) == expr2

    expr3 = log(x)/log(2)
    opt3 = optimize(expr3, [log2_opt])
    assert opt3 == log2(x)
    assert opt3.rewrite(log) == expr3

    expr4 = log(x)/log(2) + log(x+1)
    opt4 = optimize(expr4, [log2_opt])
    assert opt4 == log2(x) + log(2)*log2(x+1)
    assert opt4.rewrite(log) == expr4

    expr5 = log(17)
    opt5 = optimize(expr5, [log2_opt])
    assert opt5 == expr5

    expr6 = log(x + 3)/log(2)
    opt6 = optimize(expr6, [log2_opt])
    assert str(opt6) == 'log2(x + 3)'
    assert opt6.rewrite(log) == expr6


def test_exp2_opt():
    x = Symbol('x')
    expr1 = 1 + 2**x
    opt1 = optimize(expr1, [exp2_opt])
    assert opt1 == 1 + exp2(x)
    assert opt1.rewrite(Pow) == expr1

    expr2 = 1 + 3**x
    assert expr2 == optimize(expr2, [exp2_opt])


def test_expm1_opt():
    x = Symbol('x')

    expr1 = exp(x) - 1
    opt1 = optimize(expr1, [expm1_opt])
    assert expm1(x) - opt1 == 0
    assert opt1.rewrite(exp) == expr1

    expr2 = 3*exp(x) - 3
    opt2 = optimize(expr2, [expm1_opt])
    assert 3*expm1(x) == opt2
    assert opt2.rewrite(exp) == expr2

    expr3 = 3*exp(x) - 5
    opt3 = optimize(expr3, [expm1_opt])
    assert 3*expm1(x) - 2 == opt3
    assert opt3.rewrite(exp) == expr3
    expm1_opt_non_opportunistic = FuncMinusOneOptim(exp, expm1, opportunistic=False)
    assert expr3 == optimize(expr3, [expm1_opt_non_opportunistic])
    assert opt1 == optimize(expr1, [expm1_opt_non_opportunistic])
    assert opt2 == optimize(expr2, [expm1_opt_non_opportunistic])

    expr4 = 3*exp(x) + log(x) - 3
    opt4 = optimize(expr4, [expm1_opt])
    assert 3*expm1(x) + log(x) == opt4
    assert opt4.rewrite(exp) == expr4

    expr5 = 3*exp(2*x) - 3
    opt5 = optimize(expr5, [expm1_opt])
    assert 3*expm1(2*x) == opt5
    assert opt5.rewrite(exp) == expr5

    expr6 = (2*exp(x) + 1)/(exp(x) + 1) + 1
    opt6 = optimize(expr6, [expm1_opt])
    assert opt6.count_ops() <= expr6.count_ops()

    def ev(e):
        return e.subs(x, 3).evalf()
    assert abs(ev(expr6) - ev(opt6)) < 1e-15

    y = Symbol('y')
    expr7 = (2*exp(x) - 1)/(1 - exp(y)) - 1/(1-exp(y))
    opt7 = optimize(expr7, [expm1_opt])
    assert -2*expm1(x)/expm1(y) == opt7
    assert (opt7.rewrite(exp) - expr7).factor() == 0

    expr8 = (1+exp(x))**2 - 4
    opt8 = optimize(expr8, [expm1_opt])
    tgt8a = (exp(x) + 3)*expm1(x)
    tgt8b = 2*expm1(x) + expm1(2*x)
    # Both tgt8a & tgt8b seem to give full precision (~16 digits for double)
    # for x=1e-7 (compare with expr8 which only achieves ~8 significant digits).
    # If we can show that either tgt8a or tgt8b is preferable, we can
    # change this test to ensure the preferable version is returned.
    assert (tgt8a - tgt8b).rewrite(exp).factor() == 0
    assert opt8 in (tgt8a, tgt8b)
    assert (opt8.rewrite(exp) - expr8).factor() == 0

    expr9 = sin(expr8)
    opt9 = optimize(expr9, [expm1_opt])
    tgt9a = sin(tgt8a)
    tgt9b = sin(tgt8b)
    assert opt9 in (tgt9a, tgt9b)
    assert (opt9.rewrite(exp) - expr9.rewrite(exp)).factor().is_zero


def test_expm1_two_exp_terms():
    x, y = map(Symbol, 'x y'.split())
    expr1 = exp(x) + exp(y) - 2
    opt1 = optimize(expr1, [expm1_opt])
    assert opt1 == expm1(x) + expm1(y)


def test_cosm1_opt():
    x = Symbol('x')

    expr1 = cos(x) - 1
    opt1 = optimize(expr1, [cosm1_opt])
    assert cosm1(x) - opt1 == 0
    assert opt1.rewrite(cos) == expr1

    expr2 = 3*cos(x) - 3
    opt2 = optimize(expr2, [cosm1_opt])
    assert 3*cosm1(x) == opt2
    assert opt2.rewrite(cos) == expr2

    expr3 = 3*cos(x) - 5
    opt3 = optimize(expr3, [cosm1_opt])
    assert 3*cosm1(x) - 2 == opt3
    assert opt3.rewrite(cos) == expr3
    cosm1_opt_non_opportunistic = FuncMinusOneOptim(cos, cosm1, opportunistic=False)
    assert expr3 == optimize(expr3, [cosm1_opt_non_opportunistic])
    assert opt1 == optimize(expr1, [cosm1_opt_non_opportunistic])
    assert opt2 == optimize(expr2, [cosm1_opt_non_opportunistic])

    expr4 = 3*cos(x) + log(x) - 3
    opt4 = optimize(expr4, [cosm1_opt])
    assert 3*cosm1(x) + log(x) == opt4
    assert opt4.rewrite(cos) == expr4

    expr5 = 3*cos(2*x) - 3
    opt5 = optimize(expr5, [cosm1_opt])
    assert 3*cosm1(2*x) == opt5
    assert opt5.rewrite(cos) == expr5

    expr6 = 2 - 2*cos(x)
    opt6 = optimize(expr6, [cosm1_opt])
    assert -2*cosm1(x) == opt6
    assert opt6.rewrite(cos) == expr6


def test_cosm1_two_cos_terms():
    x, y = map(Symbol, 'x y'.split())
    expr1 = cos(x) + cos(y) - 2
    opt1 = optimize(expr1, [cosm1_opt])
    assert opt1 == cosm1(x) + cosm1(y)


def test_expm1_cosm1_mixed():
    x = Symbol('x')
    expr1 = exp(x) + cos(x) - 2
    opt1 = optimize(expr1, [expm1_opt, cosm1_opt])
    assert opt1 == cosm1(x) + expm1(x)


def _check_num_lambdify(expr, opt, val_subs, approx_ref, lambdify_kw=None, poorness=1e10):
    """ poorness=1e10 signifies that `expr` loses precision of at least ten decimal digits. """
    num_ref = expr.subs(val_subs).evalf()
    eps = numpy.finfo(numpy.float64).eps
    assert abs(num_ref - approx_ref) < approx_ref*eps
    f1 = lambdify(list(val_subs.keys()), opt, **(lambdify_kw or {}))
    args_float = tuple(map(float, val_subs.values()))
    num_err1 = abs(f1(*args_float) - approx_ref)
    assert num_err1 < abs(num_ref*eps)
    f2 = lambdify(list(val_subs.keys()), expr, **(lambdify_kw or {}))
    num_err2 = abs(f2(*args_float) - approx_ref)
    assert num_err2 > abs(num_ref*eps*poorness)   # this only ensures that the *test* works as intended


def test_cosm1_apart():
    x = Symbol('x')

    expr1 = 1/cos(x) - 1
    opt1 = optimize(expr1, [cosm1_opt])
    assert opt1 == -cosm1(x)/cos(x)
    if scipy:
        _check_num_lambdify(expr1, opt1, {x: S(10)**-30}, 5e-61, lambdify_kw={"modules": 'scipy'})

    expr2 = 2/cos(x) - 2
    opt2 = optimize(expr2, optims_scipy)
    assert opt2 == -2*cosm1(x)/cos(x)
    if scipy:
        _check_num_lambdify(expr2, opt2, {x: S(10)**-30}, 1e-60, lambdify_kw={"modules": 'scipy'})

    expr3 = pi/cos(3*x) - pi
    opt3 = optimize(expr3, [cosm1_opt])
    assert opt3 == -pi*cosm1(3*x)/cos(3*x)
    if scipy:
        _check_num_lambdify(expr3, opt3, {x: S(10)**-30/3}, float(5e-61*pi), lambdify_kw={"modules": 'scipy'})


def test_powm1():
    args = x, y = map(Symbol, "xy")

    expr1 = x**y - 1
    opt1 = optimize(expr1, [powm1_opt])
    assert opt1 == powm1(x, y)
    for arg in args:
        assert expr1.diff(arg) == opt1.diff(arg)
    if scipy and tuple(map(int, scipy.version.version.split('.')[:3])) >= (1, 10, 0):
        subs1_a = {x: Rational(*(1.0+1e-13).as_integer_ratio()), y: pi}
        ref1_f64_a = 3.139081648208105e-13
        _check_num_lambdify(expr1, opt1, subs1_a, ref1_f64_a, lambdify_kw={"modules": 'scipy'}, poorness=10**11)

        subs1_b = {x: pi, y: Rational(*(1e-10).as_integer_ratio())}
        ref1_f64_b = 1.1447298859149205e-10
        _check_num_lambdify(expr1, opt1, subs1_b, ref1_f64_b, lambdify_kw={"modules": 'scipy'}, poorness=10**9)


def test_log1p_opt():
    x = Symbol('x')
    expr1 = log(x + 1)
    opt1 = optimize(expr1, [log1p_opt])
    assert log1p(x) - opt1 == 0
    assert opt1.rewrite(log) == expr1

    expr2 = log(3*x + 3)
    opt2 = optimize(expr2, [log1p_opt])
    assert log1p(x) + log(3) == opt2
    assert (opt2.rewrite(log) - expr2).simplify() == 0

    expr3 = log(2*x + 1)
    opt3 = optimize(expr3, [log1p_opt])
    assert log1p(2*x) - opt3 == 0
    assert opt3.rewrite(log) == expr3

    expr4 = log(x+3)
    opt4 = optimize(expr4, [log1p_opt])
    assert str(opt4) == 'log(x + 3)'


def test_optims_c99():
    x = Symbol('x')

    expr1 = 2**x + log(x)/log(2) + log(x + 1) + exp(x) - 1
    opt1 = optimize(expr1, optims_c99).simplify()
    assert opt1 == exp2(x) + log2(x) + log1p(x) + expm1(x)
    assert opt1.rewrite(exp).rewrite(log).rewrite(Pow) == expr1

    expr2 = log(x)/log(2) + log(x + 1)
    opt2 = optimize(expr2, optims_c99)
    assert opt2 == log2(x) + log1p(x)
    assert opt2.rewrite(log) == expr2

    expr3 = log(x)/log(2) + log(17*x + 17)
    opt3 = optimize(expr3, optims_c99)
    delta3 = opt3 - (log2(x) + log(17) + log1p(x))
    assert delta3 == 0
    assert (opt3.rewrite(log) - expr3).simplify() == 0

    expr4 = 2**x + 3*log(5*x + 7)/(13*log(2)) + 11*exp(x) - 11 + log(17*x + 17)
    opt4 = optimize(expr4, optims_c99).simplify()
    delta4 = opt4 - (exp2(x) + 3*log2(5*x + 7)/13 + 11*expm1(x) + log(17) + log1p(x))
    assert delta4 == 0
    assert (opt4.rewrite(exp).rewrite(log).rewrite(Pow) - expr4).simplify() == 0

    expr5 = 3*exp(2*x) - 3
    opt5 = optimize(expr5, optims_c99)
    delta5 = opt5 - 3*expm1(2*x)
    assert delta5 == 0
    assert opt5.rewrite(exp) == expr5

    expr6 = exp(2*x) - 3
    opt6 = optimize(expr6, optims_c99)
    assert opt6 in (expm1(2*x) - 2, expr6)  # expm1(2*x) - 2 is not better or worse

    expr7 = log(3*x + 3)
    opt7 = optimize(expr7, optims_c99)
    delta7 = opt7 - (log(3) + log1p(x))
    assert delta7 == 0
    assert (opt7.rewrite(log) - expr7).simplify() == 0

    expr8 = log(2*x + 3)
    opt8 = optimize(expr8, optims_c99)
    assert opt8 == expr8


def test_create_expand_pow_optimization():
    cc = lambda x: ccode(
        optimize(x, [create_expand_pow_optimization(4)]))
    x = Symbol('x')
    assert cc(x**4) == 'x*x*x*x'
    assert cc(x**4 + x**2) == 'x*x + x*x*x*x'
    assert cc(x**5 + x**4) == 'pow(x, 5) + x*x*x*x'
    assert cc(sin(x)**4) == 'pow(sin(x), 4)'
    # gh issue 15335
    assert cc(x**(-4)) == '1.0/(x*x*x*x)'
    assert cc(x**(-5)) == 'pow(x, -5)'
    assert cc(-x**4) == '-(x*x*x*x)'
    assert cc(x**4 - x**2) == '-(x*x) + x*x*x*x'
    i = Symbol('i', integer=True)
    assert cc(x**i - x**2) == 'pow(x, i) - (x*x)'
    y = Symbol('y', real=True)
    assert cc(Abs(exp(y**4))) == "exp(y*y*y*y)"

    # gh issue 20753
    cc2 = lambda x: ccode(optimize(x, [create_expand_pow_optimization(
        4, base_req=lambda b: b.is_Function)]))
    assert cc2(x**3 + sin(x)**3) == "pow(x, 3) + sin(x)*sin(x)*sin(x)"


def test_matsolve():
    n = Symbol('n', integer=True)
    A = MatrixSymbol('A', n, n)
    x = MatrixSymbol('x', n, 1)

    with assuming(Q.fullrank(A)):
        assert optimize(A**(-1) * x, [matinv_opt]) == MatrixSolve(A, x)
        assert optimize(A**(-1) * x + x, [matinv_opt]) == MatrixSolve(A, x) + x


def test_logaddexp_opt():
    x, y = map(Symbol, 'x y'.split())
    expr1 = log(exp(x) + exp(y))
    opt1 = optimize(expr1, [logaddexp_opt])
    assert logaddexp(x, y) - opt1 == 0
    assert logaddexp(y, x) - opt1 == 0
    assert opt1.rewrite(log) == expr1


def test_logaddexp2_opt():
    x, y = map(Symbol, 'x y'.split())
    expr1 = log(2**x + 2**y)/log(2)
    opt1 = optimize(expr1, [logaddexp2_opt])
    assert logaddexp2(x, y) - opt1 == 0
    assert logaddexp2(y, x) - opt1 == 0
    assert opt1.rewrite(log) == expr1


def test_sinc_opts():
    def check(d):
        for k, v in d.items():
            assert optimize(k, sinc_opts) == v

    x = Symbol('x')
    check({
        sin(x)/x       : sinc(x),
        sin(2*x)/(2*x) : sinc(2*x),
        sin(3*x)/x     : 3*sinc(3*x),
        x*sin(x)       : x*sin(x)
    })

    y = Symbol('y')
    check({
        sin(x*y)/(x*y)       : sinc(x*y),
        y*sin(x/y)/x         : sinc(x/y),
        sin(sin(x))/sin(x)   : sinc(sin(x)),
        sin(3*sin(x))/sin(x) : 3*sinc(3*sin(x)),
        sin(x)/y             : sin(x)/y
    })


def test_optims_numpy():
    def check(d):
        for k, v in d.items():
            assert optimize(k, optims_numpy) == v

    x = Symbol('x')
    check({
        sin(2*x)/(2*x) + exp(2*x) - 1: sinc(2*x) + expm1(2*x),
        log(x+3)/log(2) + log(x**2 + 1): log1p(x**2) + log2(x+3)
    })


@XFAIL  # room for improvement, ideally this test case should pass.
def test_optims_numpy_TODO():
    def check(d):
        for k, v in d.items():
            assert optimize(k, optims_numpy) == v

    x, y = map(Symbol, 'x y'.split())
    check({
        log(x*y)*sin(x*y)*log(x*y+1)/(log(2)*x*y): log2(x*y)*sinc(x*y)*log1p(x*y),
        exp(x*sin(y)/y) - 1: expm1(x*sinc(y))
    })


@may_xfail
def test_compiled_ccode_with_rewriting():
    if not cython:
        skip("cython not installed.")
    if not has_c():
        skip("No C compiler found.")

    x = Symbol('x')
    about_two = 2**(58/S(117))*3**(97/S(117))*5**(4/S(39))*7**(92/S(117))/S(30)*pi
    # about_two: 1.999999999999581826
    unchanged = 2*exp(x) - about_two
    xval = S(10)**-11
    ref = unchanged.subs(x, xval).n(19) # 2.0418173913673213e-11

    rewritten = optimize(2*exp(x) - about_two, [expm1_opt])

    # Unfortunately, we need to call ``.n()`` on our expressions before we hand them
    # to ``ccode``, and we need to request a large number of significant digits.
    # In this test, results converged for double precision when the following number
    # of significant digits were chosen:
    NUMBER_OF_DIGITS = 25   # TODO: this should ideally be automatically handled.

    func_c = '''
#include <math.h>

double func_unchanged(double x) {
    return %(unchanged)s;
}
double func_rewritten(double x) {
    return %(rewritten)s;
}
''' % {"unchanged": ccode(unchanged.n(NUMBER_OF_DIGITS)),
           "rewritten": ccode(rewritten.n(NUMBER_OF_DIGITS))}

    func_pyx = '''
#cython: language_level=3
cdef extern double func_unchanged(double)
cdef extern double func_rewritten(double)
def py_unchanged(x):
    return func_unchanged(x)
def py_rewritten(x):
    return func_rewritten(x)
'''
    with tempfile.TemporaryDirectory() as folder:
        mod, info = compile_link_import_strings(
            [('func.c', func_c), ('_func.pyx', func_pyx)],
            build_dir=folder, compile_kwargs={"std": 'c99'}
        )
        err_rewritten = abs(mod.py_rewritten(1e-11) - ref)
        err_unchanged = abs(mod.py_unchanged(1e-11) - ref)
        assert 1e-27 < err_rewritten < 1e-25  # highly accurate.
        assert 1e-19 < err_unchanged < 1e-16  # quite poor.

    # Tolerances used above were determined as follows:
    # >>> no_opt = unchanged.subs(x, xval.evalf()).evalf()
    # >>> with_opt = rewritten.n(25).subs(x, 1e-11).evalf()
    # >>> with_opt - ref, no_opt - ref
    # (1.1536301877952077e-26, 1.6547074214222335e-18)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\autolev\test-examples\ruletest11.py ===
import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

x, y = _me.dynamicsymbols('x y')
a11, a12, a21, a22, b1, b2 = _sm.symbols('a11 a12 a21 a22 b1 b2', real=True)
eqn = _sm.Matrix([[0]])
eqn[0] = a11*x+a12*y-b1
eqn = eqn.row_insert(eqn.shape[0], _sm.Matrix([[0]]))
eqn[eqn.shape[0]-1] = a21*x+a22*y-b2
eqn_list = []
for i in eqn:  eqn_list.append(i.subs({a11:2, a12:5, a21:3, a22:4, b1:7, b2:6}))
print(_sm.linsolve(eqn_list, x,y))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\autolev\test-examples\ruletest12.py ===
import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

x, y = _me.dynamicsymbols('x y')
a, b, r = _sm.symbols('a b r', real=True)
eqn = _sm.Matrix([[0]])
eqn[0] = a*x**3+b*y**2-r
eqn = eqn.row_insert(eqn.shape[0], _sm.Matrix([[0]]))
eqn[eqn.shape[0]-1] = a*_sm.sin(x)**2+b*_sm.cos(2*y)-r**2
matrix_list = []
for i in eqn:matrix_list.append(i.subs({a:2.0, b:3.0, r:1.0}))
print(_sm.nsolve(matrix_list,(x,y),(_np.deg2rad(30),3.14)))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\strategies\branch\__init__.py ===
from . import traverse
from .core import (
    condition, debug, multiplex, exhaust, notempty,
    chain, onaction, sfilter, yieldify, do_one, identity)
from .tools import canon

__all__ = [
    'traverse',

    'condition', 'debug', 'multiplex', 'exhaust', 'notempty', 'chain',
    'onaction', 'sfilter', 'yieldify', 'do_one', 'identity',

    'canon',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\common\test_file_buffer_url.py ===
"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""
from io import (
    BytesIO,
    StringIO,
)
import os
import platform
from urllib.error import URLError
import uuid

import numpy as np
import pytest

from pandas.errors import (
    EmptyDataError,
    ParserError,
)
import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    Index,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


@pytest.mark.network
@pytest.mark.single_cpu
def test_url(all_parsers, csv_dir_path, httpserver):
    parser = all_parsers
    kwargs = {"sep": "\t"}

    local_path = os.path.join(csv_dir_path, "salaries.csv")
    with open(local_path, encoding="utf-8") as f:
        httpserver.serve_content(content=f.read())

    url_result = parser.read_csv(httpserver.url, **kwargs)

    local_result = parser.read_csv(local_path, **kwargs)
    tm.assert_frame_equal(url_result, local_result)


@pytest.mark.slow
def test_local_file(all_parsers, csv_dir_path):
    parser = all_parsers
    kwargs = {"sep": "\t"}

    local_path = os.path.join(csv_dir_path, "salaries.csv")
    local_result = parser.read_csv(local_path, **kwargs)
    url = "file://localhost/" + local_path

    try:
        url_result = parser.read_csv(url, **kwargs)
        tm.assert_frame_equal(url_result, local_result)
    except URLError:
        # Fails on some systems.
        pytest.skip("Failing on: " + " ".join(platform.uname()))


@xfail_pyarrow  # AssertionError: DataFrame.index are different
def test_path_path_lib(all_parsers):
    parser = all_parsers
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )
    result = tm.round_trip_pathlib(df.to_csv, lambda p: parser.read_csv(p, index_col=0))
    tm.assert_frame_equal(df, result)


@xfail_pyarrow  # AssertionError: DataFrame.index are different
def test_path_local_path(all_parsers):
    parser = all_parsers
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )
    result = tm.round_trip_localpath(
        df.to_csv, lambda p: parser.read_csv(p, index_col=0)
    )
    tm.assert_frame_equal(df, result)


def test_nonexistent_path(all_parsers):
    # gh-2428: pls no segfault
    # gh-14086: raise more helpful FileNotFoundError
    # GH#29233 "File foo" instead of "File b'foo'"
    parser = all_parsers
    path = f"{uuid.uuid4()}.csv"

    msg = r"\[Errno 2\]"
    with pytest.raises(FileNotFoundError, match=msg) as e:
        parser.read_csv(path)
    assert path == e.value.filename


@td.skip_if_windows  # os.chmod does not work in windows
def test_no_permission(all_parsers):
    # GH 23784
    parser = all_parsers

    msg = r"\[Errno 13\]"
    with tm.ensure_clean() as path:
        os.chmod(path, 0)  # make file unreadable

        # verify that this process cannot open the file (not running as sudo)
        try:
            with open(path, encoding="utf-8"):
                pass
            pytest.skip("Running as sudo.")
        except PermissionError:
            pass

        with pytest.raises(PermissionError, match=msg) as e:
            parser.read_csv(path)
        assert path == e.value.filename


@pytest.mark.parametrize(
    "data,kwargs,expected,msg",
    [
        # gh-10728: WHITESPACE_LINE
        (
            "a,b,c\n4,5,6\n ",
            {},
            DataFrame([[4, 5, 6]], columns=["a", "b", "c"]),
            None,
        ),
        # gh-10548: EAT_LINE_COMMENT
        (
            "a,b,c\n4,5,6\n#comment",
            {"comment": "#"},
            DataFrame([[4, 5, 6]], columns=["a", "b", "c"]),
            None,
        ),
        # EAT_CRNL_NOP
        (
            "a,b,c\n4,5,6\n\r",
            {},
            DataFrame([[4, 5, 6]], columns=["a", "b", "c"]),
            None,
        ),
        # EAT_COMMENT
        (
            "a,b,c\n4,5,6#comment",
            {"comment": "#"},
            DataFrame([[4, 5, 6]], columns=["a", "b", "c"]),
            None,
        ),
        # SKIP_LINE
        (
            "a,b,c\n4,5,6\nskipme",
            {"skiprows": [2]},
            DataFrame([[4, 5, 6]], columns=["a", "b", "c"]),
            None,
        ),
        # EAT_LINE_COMMENT
        (
            "a,b,c\n4,5,6\n#comment",
            {"comment": "#", "skip_blank_lines": False},
            DataFrame([[4, 5, 6]], columns=["a", "b", "c"]),
            None,
        ),
        # IN_FIELD
        (
            "a,b,c\n4,5,6\n ",
            {"skip_blank_lines": False},
            DataFrame([["4", 5, 6], [" ", None, None]], columns=["a", "b", "c"]),
            None,
        ),
        # EAT_CRNL
        (
            "a,b,c\n4,5,6\n\r",
            {"skip_blank_lines": False},
            DataFrame([[4, 5, 6], [None, None, None]], columns=["a", "b", "c"]),
            None,
        ),
        # ESCAPED_CHAR
        (
            "a,b,c\n4,5,6\n\\",
            {"escapechar": "\\"},
            None,
            "(EOF following escape character)|(unexpected end of data)",
        ),
        # ESCAPE_IN_QUOTED_FIELD
        (
            'a,b,c\n4,5,6\n"\\',
            {"escapechar": "\\"},
            None,
            "(EOF inside string starting at row 2)|(unexpected end of data)",
        ),
        # IN_QUOTED_FIELD
        (
            'a,b,c\n4,5,6\n"',
            {"escapechar": "\\"},
            None,
            "(EOF inside string starting at row 2)|(unexpected end of data)",
        ),
    ],
    ids=[
        "whitespace-line",
        "eat-line-comment",
        "eat-crnl-nop",
        "eat-comment",
        "skip-line",
        "eat-line-comment",
        "in-field",
        "eat-crnl",
        "escaped-char",
        "escape-in-quoted-field",
        "in-quoted-field",
    ],
)
def test_eof_states(all_parsers, data, kwargs, expected, msg, request):
    # see gh-10728, gh-10548
    parser = all_parsers

    if parser.engine == "pyarrow" and "comment" in kwargs:
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
        return

    if parser.engine == "pyarrow" and "\r" not in data:
        # pandas.errors.ParserError: CSV parse error: Expected 3 columns, got 1:
        # ValueError: skiprows argument must be an integer when using engine='pyarrow'
        # AssertionError: Regex pattern did not match.
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

    if expected is None:
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
    else:
        result = parser.read_csv(StringIO(data), **kwargs)
        tm.assert_frame_equal(result, expected)


def test_temporary_file(all_parsers):
    # see gh-13398
    parser = all_parsers
    data = "0 0"

    with tm.ensure_clean(mode="w+", return_filelike=True) as new_file:
        new_file.write(data)
        new_file.flush()
        new_file.seek(0)

        if parser.engine == "pyarrow":
            msg = "the 'pyarrow' engine does not support regex separators"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(new_file, sep=r"\s+", header=None)
            return

        result = parser.read_csv(new_file, sep=r"\s+", header=None)

        expected = DataFrame([[0, 0]])
        tm.assert_frame_equal(result, expected)


def test_internal_eof_byte(all_parsers):
    # see gh-5500
    parser = all_parsers
    data = "a,b\n1\x1a,2"

    expected = DataFrame([["1\x1a", 2]], columns=["a", "b"])
    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)


def test_internal_eof_byte_to_file(all_parsers):
    # see gh-16559
    parser = all_parsers
    data = b'c1,c2\r\n"test \x1a    test", test\r\n'
    expected = DataFrame([["test \x1a    test", " test"]], columns=["c1", "c2"])
    path = f"__{uuid.uuid4()}__.csv"

    with tm.ensure_clean(path) as path:
        with open(path, "wb") as f:
            f.write(data)

        result = parser.read_csv(path)
        tm.assert_frame_equal(result, expected)


def test_file_handle_string_io(all_parsers):
    # gh-14418
    #
    # Don't close user provided file handles.
    parser = all_parsers
    data = "a,b\n1,2"

    fh = StringIO(data)
    parser.read_csv(fh)
    assert not fh.closed


def test_file_handles_with_open(all_parsers, csv1):
    # gh-14418
    #
    # Don't close user provided file handles.
    parser = all_parsers

    for mode in ["r", "rb"]:
        with open(csv1, mode, encoding="utf-8" if mode == "r" else None) as f:
            parser.read_csv(f)
            assert not f.closed


def test_invalid_file_buffer_class(all_parsers):
    # see gh-15337
    class InvalidBuffer:
        pass

    parser = all_parsers
    msg = "Invalid file path or buffer object type"

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(InvalidBuffer())


def test_invalid_file_buffer_mock(all_parsers):
    # see gh-15337
    parser = all_parsers
    msg = "Invalid file path or buffer object type"

    class Foo:
        pass

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(Foo())


def test_valid_file_buffer_seems_invalid(all_parsers):
    # gh-16135: we want to ensure that "tell" and "seek"
    # aren't actually being used when we call `read_csv`
    #
    # Thus, while the object may look "invalid" (these
    # methods are attributes of the `StringIO` class),
    # it is still a valid file-object for our purposes.
    class NoSeekTellBuffer(StringIO):
        def tell(self):
            raise AttributeError("No tell method")

        def seek(self, pos, whence=0):
            raise AttributeError("No seek method")

    data = "a\n1"
    parser = all_parsers
    expected = DataFrame({"a": [1]})

    result = parser.read_csv(NoSeekTellBuffer(data))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("io_class", [StringIO, BytesIO])
@pytest.mark.parametrize("encoding", [None, "utf-8"])
def test_read_csv_file_handle(all_parsers, io_class, encoding):
    """
    Test whether read_csv does not close user-provided file handles.

    GH 36980
    """
    parser = all_parsers
    expected = DataFrame({"a": [1], "b": [2]})

    content = "a,b\n1,2"
    handle = io_class(content.encode("utf-8") if io_class == BytesIO else content)

    tm.assert_frame_equal(parser.read_csv(handle, encoding=encoding), expected)
    assert not handle.closed


def test_memory_map_compression(all_parsers, compression):
    """
    Support memory map for compressed files.

    GH 37621
    """
    parser = all_parsers
    expected = DataFrame({"a": [1], "b": [2]})

    with tm.ensure_clean() as path:
        expected.to_csv(path, index=False, compression=compression)

        if parser.engine == "pyarrow":
            msg = "The 'memory_map' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(path, memory_map=True, compression=compression)
            return

        result = parser.read_csv(path, memory_map=True, compression=compression)

    tm.assert_frame_equal(
        result,
        expected,
    )


def test_context_manager(all_parsers, datapath):
    # make sure that opened files are closed
    parser = all_parsers

    path = datapath("io", "data", "csv", "iris.csv")

    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(path, chunksize=1)
        return

    reader = parser.read_csv(path, chunksize=1)
    assert not reader.handles.handle.closed
    try:
        with reader:
            next(reader)
            assert False
    except AssertionError:
        assert reader.handles.handle.closed


def test_context_manageri_user_provided(all_parsers, datapath):
    # make sure that user-provided handles are not closed
    parser = all_parsers

    with open(datapath("io", "data", "csv", "iris.csv"), encoding="utf-8") as path:
        if parser.engine == "pyarrow":
            msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(path, chunksize=1)
            return

        reader = parser.read_csv(path, chunksize=1)
        assert not reader.handles.handle.closed
        try:
            with reader:
                next(reader)
                assert False
        except AssertionError:
            assert not reader.handles.handle.closed


@skip_pyarrow  # ParserError: Empty CSV file
def test_file_descriptor_leak(all_parsers, using_copy_on_write):
    # GH 31488
    parser = all_parsers
    with tm.ensure_clean() as path:
        with pytest.raises(EmptyDataError, match="No columns to parse from file"):
            parser.read_csv(path)


def test_memory_map(all_parsers, csv_dir_path):
    mmap_file = os.path.join(csv_dir_path, "test_mmap.csv")
    parser = all_parsers

    expected = DataFrame(
        {"a": [1, 2, 3], "b": ["one", "two", "three"], "c": ["I", "II", "III"]}
    )

    if parser.engine == "pyarrow":
        msg = "The 'memory_map' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(mmap_file, memory_map=True)
        return

    result = parser.read_csv(mmap_file, memory_map=True)
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\indexing\test_where.py ===
import numpy as np
import pytest

from pandas.core.dtypes.common import is_integer

import pandas as pd
from pandas import (
    Series,
    Timestamp,
    date_range,
    isna,
)
import pandas._testing as tm


def test_where_unsafe_int(any_signed_int_numpy_dtype):
    s = Series(np.arange(10), dtype=any_signed_int_numpy_dtype)
    mask = s < 5

    s[mask] = range(2, 7)
    expected = Series(
        list(range(2, 7)) + list(range(5, 10)),
        dtype=any_signed_int_numpy_dtype,
    )

    tm.assert_series_equal(s, expected)


def test_where_unsafe_float(float_numpy_dtype):
    s = Series(np.arange(10), dtype=float_numpy_dtype)
    mask = s < 5

    s[mask] = range(2, 7)
    data = list(range(2, 7)) + list(range(5, 10))
    expected = Series(data, dtype=float_numpy_dtype)

    tm.assert_series_equal(s, expected)


@pytest.mark.parametrize(
    "dtype,expected_dtype",
    [
        (np.int8, np.float64),
        (np.int16, np.float64),
        (np.int32, np.float64),
        (np.int64, np.float64),
        (np.float32, np.float32),
        (np.float64, np.float64),
    ],
)
def test_where_unsafe_upcast(dtype, expected_dtype):
    # see gh-9743
    s = Series(np.arange(10), dtype=dtype)
    values = [2.5, 3.5, 4.5, 5.5, 6.5]
    mask = s < 5
    expected = Series(values + list(range(5, 10)), dtype=expected_dtype)
    warn = (
        None
        if np.dtype(dtype).kind == np.dtype(expected_dtype).kind == "f"
        else FutureWarning
    )
    with tm.assert_produces_warning(warn, match="incompatible dtype"):
        s[mask] = values
    tm.assert_series_equal(s, expected)


def test_where_unsafe():
    # see gh-9731
    s = Series(np.arange(10), dtype="int64")
    values = [2.5, 3.5, 4.5, 5.5]

    mask = s > 5
    expected = Series(list(range(6)) + values, dtype="float64")

    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        s[mask] = values
    tm.assert_series_equal(s, expected)

    # see gh-3235
    s = Series(np.arange(10), dtype="int64")
    mask = s < 5
    s[mask] = range(2, 7)
    expected = Series(list(range(2, 7)) + list(range(5, 10)), dtype="int64")
    tm.assert_series_equal(s, expected)
    assert s.dtype == expected.dtype

    s = Series(np.arange(10), dtype="int64")
    mask = s > 5
    s[mask] = [0] * 4
    expected = Series([0, 1, 2, 3, 4, 5] + [0] * 4, dtype="int64")
    tm.assert_series_equal(s, expected)

    s = Series(np.arange(10))
    mask = s > 5

    msg = "cannot set using a list-like indexer with a different length than the value"
    with pytest.raises(ValueError, match=msg):
        s[mask] = [5, 4, 3, 2, 1]

    with pytest.raises(ValueError, match=msg):
        s[mask] = [0] * 5

    # dtype changes
    s = Series([1, 2, 3, 4])
    result = s.where(s > 2, np.nan)
    expected = Series([np.nan, np.nan, 3, 4])
    tm.assert_series_equal(result, expected)

    # GH 4667
    # setting with None changes dtype
    s = Series(range(10)).astype(float)
    s[8] = None
    result = s[8]
    assert isna(result)

    s = Series(range(10)).astype(float)
    s[s > 8] = None
    result = s[isna(s)]
    expected = Series(np.nan, index=[9])
    tm.assert_series_equal(result, expected)


def test_where():
    s = Series(np.random.default_rng(2).standard_normal(5))
    cond = s > 0

    rs = s.where(cond).dropna()
    rs2 = s[cond]
    tm.assert_series_equal(rs, rs2)

    rs = s.where(cond, -s)
    tm.assert_series_equal(rs, s.abs())

    rs = s.where(cond)
    assert s.shape == rs.shape
    assert rs is not s

    # test alignment
    cond = Series([True, False, False, True, False], index=s.index)
    s2 = -(s.abs())

    expected = s2[cond].reindex(s2.index[:3]).reindex(s2.index)
    rs = s2.where(cond[:3])
    tm.assert_series_equal(rs, expected)

    expected = s2.abs()
    expected.iloc[0] = s2[0]
    rs = s2.where(cond[:3], -s2)
    tm.assert_series_equal(rs, expected)


def test_where_error():
    s = Series(np.random.default_rng(2).standard_normal(5))
    cond = s > 0

    msg = "Array conditional must be same shape as self"
    with pytest.raises(ValueError, match=msg):
        s.where(1)
    with pytest.raises(ValueError, match=msg):
        s.where(cond[:3].values, -s)

    # GH 2745
    s = Series([1, 2])
    s[[True, False]] = [0, 1]
    expected = Series([0, 2])
    tm.assert_series_equal(s, expected)

    # failures
    msg = "cannot set using a list-like indexer with a different length than the value"
    with pytest.raises(ValueError, match=msg):
        s[[True, False]] = [0, 2, 3]

    with pytest.raises(ValueError, match=msg):
        s[[True, False]] = []


@pytest.mark.parametrize("klass", [list, tuple, np.array, Series])
def test_where_array_like(klass):
    # see gh-15414
    s = Series([1, 2, 3])
    cond = [False, True, True]
    expected = Series([np.nan, 2, 3])

    result = s.where(klass(cond))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "cond",
    [
        [1, 0, 1],
        Series([2, 5, 7]),
        ["True", "False", "True"],
        [Timestamp("2017-01-01"), pd.NaT, Timestamp("2017-01-02")],
    ],
)
def test_where_invalid_input(cond):
    # see gh-15414: only boolean arrays accepted
    s = Series([1, 2, 3])
    msg = "Boolean array expected for the condition"

    with pytest.raises(ValueError, match=msg):
        s.where(cond)

    msg = "Array conditional must be same shape as self"
    with pytest.raises(ValueError, match=msg):
        s.where([True])


def test_where_ndframe_align():
    msg = "Array conditional must be same shape as self"
    s = Series([1, 2, 3])

    cond = [True]
    with pytest.raises(ValueError, match=msg):
        s.where(cond)

    expected = Series([1, np.nan, np.nan])

    out = s.where(Series(cond))
    tm.assert_series_equal(out, expected)

    cond = np.array([False, True, False, True])
    with pytest.raises(ValueError, match=msg):
        s.where(cond)

    expected = Series([np.nan, 2, np.nan])

    out = s.where(Series(cond))
    tm.assert_series_equal(out, expected)


def test_where_setitem_invalid():
    # GH 2702
    # make sure correct exceptions are raised on invalid list assignment

    msg = (
        lambda x: f"cannot set using a {x} indexer with a "
        "different length than the value"
    )
    # slice
    s = Series(list("abc"), dtype=object)

    with pytest.raises(ValueError, match=msg("slice")):
        s[0:3] = list(range(27))

    s[0:3] = list(range(3))
    expected = Series([0, 1, 2])
    tm.assert_series_equal(s.astype(np.int64), expected)

    # slice with step
    s = Series(list("abcdef"), dtype=object)

    with pytest.raises(ValueError, match=msg("slice")):
        s[0:4:2] = list(range(27))

    s = Series(list("abcdef"), dtype=object)
    s[0:4:2] = list(range(2))
    expected = Series([0, "b", 1, "d", "e", "f"])
    tm.assert_series_equal(s, expected)

    # neg slices
    s = Series(list("abcdef"), dtype=object)

    with pytest.raises(ValueError, match=msg("slice")):
        s[:-1] = list(range(27))

    s[-3:-1] = list(range(2))
    expected = Series(["a", "b", "c", 0, 1, "f"])
    tm.assert_series_equal(s, expected)

    # list
    s = Series(list("abc"), dtype=object)

    with pytest.raises(ValueError, match=msg("list-like")):
        s[[0, 1, 2]] = list(range(27))

    s = Series(list("abc"), dtype=object)

    with pytest.raises(ValueError, match=msg("list-like")):
        s[[0, 1, 2]] = list(range(2))

    # scalar
    s = Series(list("abc"), dtype=object)
    s[0] = list(range(10))
    expected = Series([list(range(10)), "b", "c"])
    tm.assert_series_equal(s, expected)


@pytest.mark.parametrize("size", range(2, 6))
@pytest.mark.parametrize(
    "mask", [[True, False, False, False, False], [True, False], [False]]
)
@pytest.mark.parametrize(
    "item", [2.0, np.nan, np.finfo(float).max, np.finfo(float).min]
)
# Test numpy arrays, lists and tuples as the input to be
# broadcast
@pytest.mark.parametrize(
    "box", [lambda x: np.array([x]), lambda x: [x], lambda x: (x,)]
)
def test_broadcast(size, mask, item, box):
    # GH#8801, GH#4195
    selection = np.resize(mask, size)

    data = np.arange(size, dtype=float)

    # Construct the expected series by taking the source
    # data or item based on the selection
    expected = Series(
        [item if use_item else data[i] for i, use_item in enumerate(selection)]
    )

    s = Series(data)

    s[selection] = item
    tm.assert_series_equal(s, expected)

    s = Series(data)
    result = s.where(~selection, box(item))
    tm.assert_series_equal(result, expected)

    s = Series(data)
    result = s.mask(selection, box(item))
    tm.assert_series_equal(result, expected)


def test_where_inplace():
    s = Series(np.random.default_rng(2).standard_normal(5))
    cond = s > 0

    rs = s.copy()

    rs.where(cond, inplace=True)
    tm.assert_series_equal(rs.dropna(), s[cond])
    tm.assert_series_equal(rs, s.where(cond))

    rs = s.copy()
    rs.where(cond, -s, inplace=True)
    tm.assert_series_equal(rs, s.where(cond, -s))


def test_where_dups():
    # GH 4550
    # where crashes with dups in index
    s1 = Series(list(range(3)))
    s2 = Series(list(range(3)))
    comb = pd.concat([s1, s2])
    result = comb.where(comb < 2)
    expected = Series([0, 1, np.nan, 0, 1, np.nan], index=[0, 1, 2, 0, 1, 2])
    tm.assert_series_equal(result, expected)

    # GH 4548
    # inplace updating not working with dups
    comb[comb < 1] = 5
    expected = Series([5, 1, 2, 5, 1, 2], index=[0, 1, 2, 0, 1, 2])
    tm.assert_series_equal(comb, expected)

    comb[comb < 2] += 10
    expected = Series([5, 11, 2, 5, 11, 2], index=[0, 1, 2, 0, 1, 2])
    tm.assert_series_equal(comb, expected)


def test_where_numeric_with_string():
    # GH 9280
    s = Series([1, 2, 3])
    w = s.where(s > 1, "X")

    assert not is_integer(w[0])
    assert is_integer(w[1])
    assert is_integer(w[2])
    assert isinstance(w[0], str)
    assert w.dtype == "object"

    w = s.where(s > 1, ["X", "Y", "Z"])
    assert not is_integer(w[0])
    assert is_integer(w[1])
    assert is_integer(w[2])
    assert isinstance(w[0], str)
    assert w.dtype == "object"

    w = s.where(s > 1, np.array(["X", "Y", "Z"]))
    assert not is_integer(w[0])
    assert is_integer(w[1])
    assert is_integer(w[2])
    assert isinstance(w[0], str)
    assert w.dtype == "object"


@pytest.mark.parametrize("dtype", ["timedelta64[ns]", "datetime64[ns]"])
def test_where_datetimelike_coerce(dtype):
    ser = Series([1, 2], dtype=dtype)
    expected = Series([10, 10])
    mask = np.array([False, False])

    msg = "Downcasting behavior in Series and DataFrame methods 'where'"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rs = ser.where(mask, [10, 10])
    tm.assert_series_equal(rs, expected)

    with tm.assert_produces_warning(FutureWarning, match=msg):
        rs = ser.where(mask, 10)
    tm.assert_series_equal(rs, expected)

    with tm.assert_produces_warning(FutureWarning, match=msg):
        rs = ser.where(mask, 10.0)
    tm.assert_series_equal(rs, expected)

    with tm.assert_produces_warning(FutureWarning, match=msg):
        rs = ser.where(mask, [10.0, 10.0])
    tm.assert_series_equal(rs, expected)

    rs = ser.where(mask, [10.0, np.nan])
    expected = Series([10, np.nan], dtype="object")
    tm.assert_series_equal(rs, expected)


def test_where_datetimetz():
    # GH 15701
    timestamps = ["2016-12-31 12:00:04+00:00", "2016-12-31 12:00:04.010000+00:00"]
    ser = Series([Timestamp(t) for t in timestamps], dtype="datetime64[ns, UTC]")
    rs = ser.where(Series([False, True]))
    expected = Series([pd.NaT, ser[1]], dtype="datetime64[ns, UTC]")
    tm.assert_series_equal(rs, expected)


def test_where_sparse():
    # GH#17198 make sure we dont get an AttributeError for sp_index
    ser = Series(pd.arrays.SparseArray([1, 2]))
    result = ser.where(ser >= 2, 0)
    expected = Series(pd.arrays.SparseArray([0, 2]))
    tm.assert_series_equal(result, expected)


def test_where_empty_series_and_empty_cond_having_non_bool_dtypes():
    # https://github.com/pandas-dev/pandas/issues/34592
    ser = Series([], dtype=float)
    result = ser.where([])
    tm.assert_series_equal(result, ser)


def test_where_categorical(frame_or_series):
    # https://github.com/pandas-dev/pandas/issues/18888
    exp = frame_or_series(
        pd.Categorical(["A", "A", "B", "B", np.nan], categories=["A", "B", "C"]),
        dtype="category",
    )
    df = frame_or_series(["A", "A", "B", "B", "C"], dtype="category")
    res = df.where(df != "C")
    tm.assert_equal(exp, res)


def test_where_datetimelike_categorical(tz_naive_fixture):
    # GH#37682
    tz = tz_naive_fixture

    dr = date_range("2001-01-01", periods=3, tz=tz)._with_freq(None)
    lvals = pd.DatetimeIndex([dr[0], dr[1], pd.NaT])
    rvals = pd.Categorical([dr[0], pd.NaT, dr[2]])

    mask = np.array([True, True, False])

    # DatetimeIndex.where
    res = lvals.where(mask, rvals)
    tm.assert_index_equal(res, dr)

    # DatetimeArray.where
    res = lvals._data._where(mask, rvals)
    tm.assert_datetime_array_equal(res, dr._data)

    # Series.where
    res = Series(lvals).where(mask, rvals)
    tm.assert_series_equal(res, Series(dr))

    # DataFrame.where
    res = pd.DataFrame(lvals).where(mask[:, None], pd.DataFrame(rvals))

    tm.assert_frame_equal(res, pd.DataFrame(dr))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\tests\test_derivatives.py ===
"""
Some examples have been taken from:

http://www.math.uwaterloo.ca/~hwolkowi//matrixcookbook.pdf
"""
from sympy import KroneckerProduct
from sympy.combinatorics import Permutation
from sympy.concrete.summations import Sum
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.expressions.determinant import Determinant
from sympy.matrices.expressions.diagonal import DiagMatrix
from sympy.matrices.expressions.hadamard import (HadamardPower, HadamardProduct, hadamard_product)
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import OneMatrix
from sympy.matrices.expressions.trace import Trace
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.special import (Identity, ZeroMatrix)
from sympy.tensor.array.array_derivatives import ArrayDerivative
from sympy.matrices.expressions import hadamard_power
from sympy.tensor.array.expressions.array_expressions import ArrayAdd, ArrayTensorProduct, PermuteDims

i, j, k = symbols("i j k")
m, n = symbols("m n")

X = MatrixSymbol("X", k, k)
x = MatrixSymbol("x", k, 1)
y = MatrixSymbol("y", k, 1)

A = MatrixSymbol("A", k, k)
B = MatrixSymbol("B", k, k)
C = MatrixSymbol("C", k, k)
D = MatrixSymbol("D", k, k)

a = MatrixSymbol("a", k, 1)
b = MatrixSymbol("b", k, 1)
c = MatrixSymbol("c", k, 1)
d = MatrixSymbol("d", k, 1)


KDelta = lambda i, j: KroneckerDelta(i, j, (0, k-1))


def _check_derivative_with_explicit_matrix(expr, x, diffexpr, dim=2):
    # TODO: this is commented because it slows down the tests.
    return

    expr = expr.xreplace({k: dim})
    x = x.xreplace({k: dim})
    diffexpr = diffexpr.xreplace({k: dim})

    expr = expr.as_explicit()
    x = x.as_explicit()
    diffexpr = diffexpr.as_explicit()

    assert expr.diff(x).reshape(*diffexpr.shape).tomatrix() == diffexpr


def test_matrix_derivative_by_scalar():
    assert A.diff(i) == ZeroMatrix(k, k)
    assert (A*(X + B)*c).diff(i) == ZeroMatrix(k, 1)
    assert x.diff(i) == ZeroMatrix(k, 1)
    assert (x.T*y).diff(i) == ZeroMatrix(1, 1)
    assert (x*x.T).diff(i) == ZeroMatrix(k, k)
    assert (x + y).diff(i) == ZeroMatrix(k, 1)
    assert hadamard_power(x, 2).diff(i) == ZeroMatrix(k, 1)
    assert hadamard_power(x, i).diff(i).dummy_eq(
        HadamardProduct(x.applyfunc(log), HadamardPower(x, i)))
    assert hadamard_product(x, y).diff(i) == ZeroMatrix(k, 1)
    assert hadamard_product(i*OneMatrix(k, 1), x, y).diff(i) == hadamard_product(x, y)
    assert (i*x).diff(i) == x
    assert (sin(i)*A*B*x).diff(i) == cos(i)*A*B*x
    assert x.applyfunc(sin).diff(i) == ZeroMatrix(k, 1)
    assert Trace(i**2*X).diff(i) == 2*i*Trace(X)

    mu = symbols("mu")
    expr = (2*mu*x)
    assert expr.diff(x) == 2*mu*Identity(k)


def test_one_matrix():
    assert MatMul(x.T, OneMatrix(k, 1)).diff(x) == OneMatrix(k, 1)


def test_matrix_derivative_non_matrix_result():
    # This is a 4-dimensional array:
    I = Identity(k)
    AdA = PermuteDims(ArrayTensorProduct(I, I), Permutation(3)(1, 2))
    assert A.diff(A) == AdA
    assert A.T.diff(A) == PermuteDims(ArrayTensorProduct(I, I), Permutation(3)(1, 2, 3))
    assert (2*A).diff(A) == PermuteDims(ArrayTensorProduct(2*I, I), Permutation(3)(1, 2))
    assert MatAdd(A, A).diff(A) == ArrayAdd(AdA, AdA)
    assert (A + B).diff(A) == AdA


def test_matrix_derivative_trivial_cases():
    # Cookbook example 33:
    # TODO: find a way to represent a four-dimensional zero-array:
    assert X.diff(A) == ArrayDerivative(X, A)


def test_matrix_derivative_with_inverse():

    # Cookbook example 61:
    expr = a.T*Inverse(X)*b
    assert expr.diff(X) == -Inverse(X).T*a*b.T*Inverse(X).T

    # Cookbook example 62:
    expr = Determinant(Inverse(X))
    # Not implemented yet:
    # assert expr.diff(X) == -Determinant(X.inv())*(X.inv()).T

    # Cookbook example 63:
    expr = Trace(A*Inverse(X)*B)
    assert expr.diff(X) == -(X**(-1)*B*A*X**(-1)).T

    # Cookbook example 64:
    expr = Trace(Inverse(X + A))
    assert expr.diff(X) == -(Inverse(X + A)).T**2


def test_matrix_derivative_vectors_and_scalars():

    assert x.diff(x) == Identity(k)
    assert x[i, 0].diff(x[m, 0]).doit() == KDelta(m, i)

    assert x.T.diff(x) == Identity(k)

    # Cookbook example 69:
    expr = x.T*a
    assert expr.diff(x) == a
    assert expr[0, 0].diff(x[m, 0]).doit() == a[m, 0]
    expr = a.T*x
    assert expr.diff(x) == a

    # Cookbook example 70:
    expr = a.T*X*b
    assert expr.diff(X) == a*b.T

    # Cookbook example 71:
    expr = a.T*X.T*b
    assert expr.diff(X) == b*a.T

    # Cookbook example 72:
    expr = a.T*X*a
    assert expr.diff(X) == a*a.T
    expr = a.T*X.T*a
    assert expr.diff(X) == a*a.T

    # Cookbook example 77:
    expr = b.T*X.T*X*c
    assert expr.diff(X) == X*b*c.T + X*c*b.T

    # Cookbook example 78:
    expr = (B*x + b).T*C*(D*x + d)
    assert expr.diff(x) == B.T*C*(D*x + d) + D.T*C.T*(B*x + b)

    # Cookbook example 81:
    expr = x.T*B*x
    assert expr.diff(x) == B*x + B.T*x

    # Cookbook example 82:
    expr = b.T*X.T*D*X*c
    assert expr.diff(X) == D.T*X*b*c.T + D*X*c*b.T

    # Cookbook example 83:
    expr = (X*b + c).T*D*(X*b + c)
    assert expr.diff(X) == D*(X*b + c)*b.T + D.T*(X*b + c)*b.T
    assert str(expr[0, 0].diff(X[m, n]).doit()) == \
        'b[n, 0]*Sum((c[_i_1, 0] + Sum(X[_i_1, _i_3]*b[_i_3, 0], (_i_3, 0, k - 1)))*D[_i_1, m], (_i_1, 0, k - 1)) + Sum((c[_i_2, 0] + Sum(X[_i_2, _i_4]*b[_i_4, 0], (_i_4, 0, k - 1)))*D[m, _i_2]*b[n, 0], (_i_2, 0, k - 1))'

    # See https://github.com/sympy/sympy/issues/16504#issuecomment-1018339957
    expr = x*x.T*x
    I = Identity(k)
    assert expr.diff(x) == KroneckerProduct(I, x.T*x) + 2*x*x.T


def test_matrix_derivatives_of_traces():

    expr = Trace(A)*A
    I = Identity(k)
    assert expr.diff(A) == ArrayAdd(ArrayTensorProduct(I, A), PermuteDims(ArrayTensorProduct(Trace(A)*I, I), Permutation(3)(1, 2)))
    assert expr[i, j].diff(A[m, n]).doit() == (
        KDelta(i, m)*KDelta(j, n)*Trace(A) +
        KDelta(m, n)*A[i, j]
    )

    ## First order:

    # Cookbook example 99:
    expr = Trace(X)
    assert expr.diff(X) == Identity(k)
    assert expr.rewrite(Sum).diff(X[m, n]).doit() == KDelta(m, n)

    # Cookbook example 100:
    expr = Trace(X*A)
    assert expr.diff(X) == A.T
    assert expr.rewrite(Sum).diff(X[m, n]).doit() == A[n, m]

    # Cookbook example 101:
    expr = Trace(A*X*B)
    assert expr.diff(X) == A.T*B.T
    assert expr.rewrite(Sum).diff(X[m, n]).doit().dummy_eq((A.T*B.T)[m, n])

    # Cookbook example 102:
    expr = Trace(A*X.T*B)
    assert expr.diff(X) == B*A

    # Cookbook example 103:
    expr = Trace(X.T*A)
    assert expr.diff(X) == A

    # Cookbook example 104:
    expr = Trace(A*X.T)
    assert expr.diff(X) == A

    # Cookbook example 105:
    # TODO: TensorProduct is not supported
    #expr = Trace(TensorProduct(A, X))
    #assert expr.diff(X) == Trace(A)*Identity(k)

    ## Second order:

    # Cookbook example 106:
    expr = Trace(X**2)
    assert expr.diff(X) == 2*X.T

    # Cookbook example 107:
    expr = Trace(X**2*B)
    assert expr.diff(X) == (X*B + B*X).T
    expr = Trace(MatMul(X, X, B))
    assert expr.diff(X) == (X*B + B*X).T

    # Cookbook example 108:
    expr = Trace(X.T*B*X)
    assert expr.diff(X) == B*X + B.T*X

    # Cookbook example 109:
    expr = Trace(B*X*X.T)
    assert expr.diff(X) == B*X + B.T*X

    # Cookbook example 110:
    expr = Trace(X*X.T*B)
    assert expr.diff(X) == B*X + B.T*X

    # Cookbook example 111:
    expr = Trace(X*B*X.T)
    assert expr.diff(X) == X*B.T + X*B

    # Cookbook example 112:
    expr = Trace(B*X.T*X)
    assert expr.diff(X) == X*B.T + X*B

    # Cookbook example 113:
    expr = Trace(X.T*X*B)
    assert expr.diff(X) == X*B.T + X*B

    # Cookbook example 114:
    expr = Trace(A*X*B*X)
    assert expr.diff(X) == A.T*X.T*B.T + B.T*X.T*A.T

    # Cookbook example 115:
    expr = Trace(X.T*X)
    assert expr.diff(X) == 2*X
    expr = Trace(X*X.T)
    assert expr.diff(X) == 2*X

    # Cookbook example 116:
    expr = Trace(B.T*X.T*C*X*B)
    assert expr.diff(X) == C.T*X*B*B.T + C*X*B*B.T

    # Cookbook example 117:
    expr = Trace(X.T*B*X*C)
    assert expr.diff(X) == B*X*C + B.T*X*C.T

    # Cookbook example 118:
    expr = Trace(A*X*B*X.T*C)
    assert expr.diff(X) == A.T*C.T*X*B.T + C*A*X*B

    # Cookbook example 119:
    expr = Trace((A*X*B + C)*(A*X*B + C).T)
    assert expr.diff(X) == 2*A.T*(A*X*B + C)*B.T

    # Cookbook example 120:
    # TODO: no support for TensorProduct.
    # expr = Trace(TensorProduct(X, X))
    # expr = Trace(X)*Trace(X)
    # expr.diff(X) == 2*Trace(X)*Identity(k)

    # Higher Order

    # Cookbook example 121:
    expr = Trace(X**k)
    #assert expr.diff(X) == k*(X**(k-1)).T

    # Cookbook example 122:
    expr = Trace(A*X**k)
    #assert expr.diff(X) == # Needs indices

    # Cookbook example 123:
    expr = Trace(B.T*X.T*C*X*X.T*C*X*B)
    assert expr.diff(X) == C*X*X.T*C*X*B*B.T + C.T*X*B*B.T*X.T*C.T*X + C*X*B*B.T*X.T*C*X + C.T*X*X.T*C.T*X*B*B.T

    # Other

    # Cookbook example 124:
    expr = Trace(A*X**(-1)*B)
    assert expr.diff(X) == -Inverse(X).T*A.T*B.T*Inverse(X).T

    # Cookbook example 125:
    expr = Trace(Inverse(X.T*C*X)*A)
    # Warning: result in the cookbook is equivalent if B and C are symmetric:
    assert expr.diff(X) == - X.inv().T*A.T*X.inv()*C.inv().T*X.inv().T - X.inv().T*A*X.inv()*C.inv()*X.inv().T

    # Cookbook example 126:
    expr = Trace((X.T*C*X).inv()*(X.T*B*X))
    assert expr.diff(X) == -2*C*X*(X.T*C*X).inv()*X.T*B*X*(X.T*C*X).inv() + 2*B*X*(X.T*C*X).inv()

    # Cookbook example 127:
    expr = Trace((A + X.T*C*X).inv()*(X.T*B*X))
    # Warning: result in the cookbook is equivalent if B and C are symmetric:
    assert expr.diff(X) == B*X*Inverse(A + X.T*C*X) - C*X*Inverse(A + X.T*C*X)*X.T*B*X*Inverse(A + X.T*C*X) - C.T*X*Inverse(A.T + (C*X).T*X)*X.T*B.T*X*Inverse(A.T + (C*X).T*X) + B.T*X*Inverse(A.T + (C*X).T*X)


def test_derivatives_of_complicated_matrix_expr():
    expr = a.T*(A*X*(X.T*B + X*A) + B.T*X.T*(a*b.T*(X*D*X.T + X*(X.T*B + A*X)*D*B - X.T*C.T*A)*B + B*(X*D.T + B*A*X*A.T - 3*X*D))*B + 42*X*B*X.T*A.T*(X + X.T))*b
    result = (B*(B*A*X*A.T - 3*X*D + X*D.T) + a*b.T*(X*(A*X + X.T*B)*D*B + X*D*X.T - X.T*C.T*A)*B)*B*b*a.T*B.T + B**2*b*a.T*B.T*X.T*a*b.T*X*D + 42*A*X*B.T*X.T*a*b.T + B*D*B**3*b*a.T*B.T*X.T*a*b.T*X + B*b*a.T*A*X + a*b.T*(42*X + 42*X.T)*A*X*B.T + b*a.T*X*B*a*b.T*B.T**2*X*D.T + b*a.T*X*B*a*b.T*B.T**3*D.T*(B.T*X + X.T*A.T) + 42*b*a.T*X*B*X.T*A.T + A.T*(42*X + 42*X.T)*b*a.T*X*B + A.T*B.T**2*X*B*a*b.T*B.T*A + A.T*a*b.T*(A.T*X.T + B.T*X) + A.T*X.T*b*a.T*X*B*a*b.T*B.T**3*D.T + B.T*X*B*a*b.T*B.T*D - 3*B.T*X*B*a*b.T*B.T*D.T - C.T*A*B**2*b*a.T*B.T*X.T*a*b.T + X.T*A.T*a*b.T*A.T
    assert expr.diff(X) == result


def test_mixed_deriv_mixed_expressions():

    expr = 3*Trace(A)
    assert expr.diff(A) == 3*Identity(k)

    expr = k
    deriv = expr.diff(A)
    assert isinstance(deriv, ZeroMatrix)
    assert deriv == ZeroMatrix(k, k)

    expr = Trace(A)**2
    assert expr.diff(A) == (2*Trace(A))*Identity(k)

    expr = Trace(A)*A
    I = Identity(k)
    assert expr.diff(A) == ArrayAdd(ArrayTensorProduct(I, A), PermuteDims(ArrayTensorProduct(Trace(A)*I, I), Permutation(3)(1, 2)))

    expr = Trace(Trace(A)*A)
    assert expr.diff(A) == (2*Trace(A))*Identity(k)

    expr = Trace(Trace(Trace(A)*A)*A)
    assert expr.diff(A) == (3*Trace(A)**2)*Identity(k)


def test_derivatives_matrix_norms():

    expr = x.T*y
    assert expr.diff(x) == y
    assert expr[0, 0].diff(x[m, 0]).doit() == y[m, 0]

    expr = (x.T*y)**S.Half
    assert expr.diff(x) == y/(2*sqrt(x.T*y))

    expr = (x.T*x)**S.Half
    assert expr.diff(x) == x*(x.T*x)**Rational(-1, 2)

    expr = (c.T*a*x.T*b)**S.Half
    assert expr.diff(x) == b*a.T*c/sqrt(c.T*a*x.T*b)/2

    expr = (c.T*a*x.T*b)**Rational(1, 3)
    assert expr.diff(x) == b*a.T*c*(c.T*a*x.T*b)**Rational(-2, 3)/3

    expr = (a.T*X*b)**S.Half
    assert expr.diff(X) == a/(2*sqrt(a.T*X*b))*b.T

    expr = d.T*x*(a.T*X*b)**S.Half*y.T*c
    assert expr.diff(X) == a/(2*sqrt(a.T*X*b))*x.T*d*y.T*c*b.T


def test_derivatives_elementwise_applyfunc():

    expr = x.applyfunc(tan)
    assert expr.diff(x).dummy_eq(
        DiagMatrix(x.applyfunc(lambda x: tan(x)**2 + 1)))
    assert expr[i, 0].diff(x[m, 0]).doit() == (tan(x[i, 0])**2 + 1)*KDelta(i, m)
    _check_derivative_with_explicit_matrix(expr, x, expr.diff(x))

    expr = (i**2*x).applyfunc(sin)
    assert expr.diff(i).dummy_eq(
        HadamardProduct((2*i)*x, (i**2*x).applyfunc(cos)))
    assert expr[i, 0].diff(i).doit() == 2*i*x[i, 0]*cos(i**2*x[i, 0])
    _check_derivative_with_explicit_matrix(expr, i, expr.diff(i))

    expr = (log(i)*A*B).applyfunc(sin)
    assert expr.diff(i).dummy_eq(
        HadamardProduct(A*B/i, (log(i)*A*B).applyfunc(cos)))
    _check_derivative_with_explicit_matrix(expr, i, expr.diff(i))

    expr = A*x.applyfunc(exp)
    # TODO: restore this result (currently returning the transpose):
    #  assert expr.diff(x).dummy_eq(DiagMatrix(x.applyfunc(exp))*A.T)
    _check_derivative_with_explicit_matrix(expr, x, expr.diff(x))

    expr = x.T*A*x + k*y.applyfunc(sin).T*x
    assert expr.diff(x).dummy_eq(A.T*x + A*x + k*y.applyfunc(sin))
    _check_derivative_with_explicit_matrix(expr, x, expr.diff(x))

    expr = x.applyfunc(sin).T*y
    # TODO: restore (currently returning the transpose):
    #  assert expr.diff(x).dummy_eq(DiagMatrix(x.applyfunc(cos))*y)
    _check_derivative_with_explicit_matrix(expr, x, expr.diff(x))

    expr = (a.T * X * b).applyfunc(sin)
    assert expr.diff(X).dummy_eq(a*(a.T*X*b).applyfunc(cos)*b.T)
    _check_derivative_with_explicit_matrix(expr, X, expr.diff(X))

    expr = a.T * X.applyfunc(sin) * b
    assert expr.diff(X).dummy_eq(
        DiagMatrix(a)*X.applyfunc(cos)*DiagMatrix(b))
    _check_derivative_with_explicit_matrix(expr, X, expr.diff(X))

    expr = a.T * (A*X*B).applyfunc(sin) * b
    assert expr.diff(X).dummy_eq(
        A.T*DiagMatrix(a)*(A*X*B).applyfunc(cos)*DiagMatrix(b)*B.T)
    _check_derivative_with_explicit_matrix(expr, X, expr.diff(X))

    expr = a.T * (A*X*b).applyfunc(sin) * b.T
    # TODO: not implemented
    #assert expr.diff(X) == ...
    #_check_derivative_with_explicit_matrix(expr, X, expr.diff(X))

    expr = a.T*A*X.applyfunc(sin)*B*b
    assert expr.diff(X).dummy_eq(
        HadamardProduct(A.T * a * b.T * B.T, X.applyfunc(cos)))

    expr = a.T * (A*X.applyfunc(sin)*B).applyfunc(log) * b
    # TODO: wrong
    # assert expr.diff(X) == A.T*DiagMatrix(a)*(A*X.applyfunc(sin)*B).applyfunc(Lambda(k, 1/k))*DiagMatrix(b)*B.T

    expr = a.T * (X.applyfunc(sin)).applyfunc(log) * b
    # TODO: wrong
    # assert expr.diff(X) == DiagMatrix(a)*X.applyfunc(sin).applyfunc(Lambda(k, 1/k))*DiagMatrix(b)


def test_derivatives_of_hadamard_expressions():

    # Hadamard Product

    expr = hadamard_product(a, x, b)
    assert expr.diff(x) == DiagMatrix(hadamard_product(b, a))

    expr = a.T*hadamard_product(A, X, B)*b
    assert expr.diff(X) == HadamardProduct(a*b.T, A, B)

    # Hadamard Power

    expr = hadamard_power(x, 2)
    assert expr.diff(x).doit() == 2*DiagMatrix(x)

    expr = hadamard_power(x.T, 2)
    assert expr.diff(x).doit() == 2*DiagMatrix(x)

    expr = hadamard_power(x, S.Half)
    assert expr.diff(x) == S.Half*DiagMatrix(hadamard_power(x, Rational(-1, 2)))

    expr = hadamard_power(a.T*X*b, 2)
    assert expr.diff(X) == 2*a*a.T*X*b*b.T

    expr = hadamard_power(a.T*X*b, S.Half)
    assert expr.diff(X) == a/(2*sqrt(a.T*X*b))*b.T

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_rcode.py ===
from sympy.core import (S, pi, oo, Symbol, symbols, Rational, Integer,
                        GoldenRatio, EulerGamma, Catalan, Lambda, Dummy)
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
                             gamma, sign, Max, Min, factorial, beta)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.sets import Range
from sympy.logic import ITE
from sympy.codegen import For, aug_assign, Assignment
from sympy.testing.pytest import raises
from sympy.printing.rcode import RCodePrinter
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol

from sympy.printing.rcode import rcode

x, y, z = symbols('x,y,z')


def test_printmethod():
    class fabs(Abs):
        def _rcode(self, printer):
            return "abs(%s)" % printer._print(self.args[0])

    assert rcode(fabs(x)) == "abs(x)"


def test_rcode_sqrt():
    assert rcode(sqrt(x)) == "sqrt(x)"
    assert rcode(x**0.5) == "sqrt(x)"
    assert rcode(sqrt(x)) == "sqrt(x)"


def test_rcode_Pow():
    assert rcode(x**3) == "x^3"
    assert rcode(x**(y**3)) == "x^(y^3)"
    g = implemented_function('g', Lambda(x, 2*x))
    assert rcode(1/(g(x)*3.5)**(x - y**x)/(x**2 + y)) == \
        "(3.5*2*x)^(-x + y^x)/(x^2 + y)"
    assert rcode(x**-1.0) == '1.0/x'
    assert rcode(x**Rational(2, 3)) == 'x^(2.0/3.0)'
    _cond_cfunc = [(lambda base, exp: exp.is_integer, "dpowi"),
                   (lambda base, exp: not exp.is_integer, "pow")]
    assert rcode(x**3, user_functions={'Pow': _cond_cfunc}) == 'dpowi(x, 3)'
    assert rcode(x**3.2, user_functions={'Pow': _cond_cfunc}) == 'pow(x, 3.2)'


def test_rcode_Max():
    # Test for gh-11926
    assert rcode(Max(x,x*x),user_functions={"Max":"my_max", "Pow":"my_pow"}) == 'my_max(x, my_pow(x, 2))'


def test_rcode_constants_mathh():
    assert rcode(exp(1)) == "exp(1)"
    assert rcode(pi) == "pi"
    assert rcode(oo) == "Inf"
    assert rcode(-oo) == "-Inf"


def test_rcode_constants_other():
    assert rcode(2*GoldenRatio) == "GoldenRatio = 1.61803398874989;\n2*GoldenRatio"
    assert rcode(
        2*Catalan) == "Catalan = 0.915965594177219;\n2*Catalan"
    assert rcode(2*EulerGamma) == "EulerGamma = 0.577215664901533;\n2*EulerGamma"


def test_rcode_Rational():
    assert rcode(Rational(3, 7)) == "3.0/7.0"
    assert rcode(Rational(18, 9)) == "2"
    assert rcode(Rational(3, -7)) == "-3.0/7.0"
    assert rcode(Rational(-3, -7)) == "3.0/7.0"
    assert rcode(x + Rational(3, 7)) == "x + 3.0/7.0"
    assert rcode(Rational(3, 7)*x) == "(3.0/7.0)*x"


def test_rcode_Integer():
    assert rcode(Integer(67)) == "67"
    assert rcode(Integer(-1)) == "-1"


def test_rcode_functions():
    assert rcode(sin(x) ** cos(x)) == "sin(x)^cos(x)"
    assert rcode(factorial(x) + gamma(y)) == "factorial(x) + gamma(y)"
    assert rcode(beta(Min(x, y), Max(x, y))) == "beta(min(x, y), max(x, y))"


def test_rcode_inline_function():
    x = symbols('x')
    g = implemented_function('g', Lambda(x, 2*x))
    assert rcode(g(x)) == "2*x"
    g = implemented_function('g', Lambda(x, 2*x/Catalan))
    assert rcode(
        g(x)) == "Catalan = %s;\n2*x/Catalan" % Catalan.n()
    A = IndexedBase('A')
    i = Idx('i', symbols('n', integer=True))
    g = implemented_function('g', Lambda(x, x*(1 + x)*(2 + x)))
    res=rcode(g(A[i]), assign_to=A[i])
    ref=(
        "for (i in 1:n){\n"
        "   A[i] = (A[i] + 1)*(A[i] + 2)*A[i];\n"
        "}"
    )
    assert res == ref


def test_rcode_exceptions():
    assert rcode(ceiling(x)) == "ceiling(x)"
    assert rcode(Abs(x)) == "abs(x)"
    assert rcode(gamma(x)) == "gamma(x)"


def test_rcode_user_functions():
    x = symbols('x', integer=False)
    n = symbols('n', integer=True)
    custom_functions = {
        "ceiling": "myceil",
        "Abs": [(lambda x: not x.is_integer, "fabs"), (lambda x: x.is_integer, "abs")],
    }
    assert rcode(ceiling(x), user_functions=custom_functions) == "myceil(x)"
    assert rcode(Abs(x), user_functions=custom_functions) == "fabs(x)"
    assert rcode(Abs(n), user_functions=custom_functions) == "abs(n)"


def test_rcode_boolean():
    assert rcode(True) == "True"
    assert rcode(S.true) == "True"
    assert rcode(False) == "False"
    assert rcode(S.false) == "False"
    assert rcode(x & y) == "x & y"
    assert rcode(x | y) == "x | y"
    assert rcode(~x) == "!x"
    assert rcode(x & y & z) == "x & y & z"
    assert rcode(x | y | z) == "x | y | z"
    assert rcode((x & y) | z) == "z | x & y"
    assert rcode((x | y) & z) == "z & (x | y)"

def test_rcode_Relational():
    assert rcode(Eq(x, y)) == "x == y"
    assert rcode(Ne(x, y)) == "x != y"
    assert rcode(Le(x, y)) == "x <= y"
    assert rcode(Lt(x, y)) == "x < y"
    assert rcode(Gt(x, y)) == "x > y"
    assert rcode(Ge(x, y)) == "x >= y"


def test_rcode_Piecewise():
    expr = Piecewise((x, x < 1), (x**2, True))
    res=rcode(expr)
    ref="ifelse(x < 1,x,x^2)"
    assert res == ref
    tau=Symbol("tau")
    res=rcode(expr,tau)
    ref="tau = ifelse(x < 1,x,x^2);"
    assert res == ref

    expr = 2*Piecewise((x, x < 1), (x**2, x<2), (x**3,True))
    assert rcode(expr) == "2*ifelse(x < 1,x,ifelse(x < 2,x^2,x^3))"
    res = rcode(expr, assign_to='c')
    assert res == "c = 2*ifelse(x < 1,x,ifelse(x < 2,x^2,x^3));"

    # Check that Piecewise without a True (default) condition error
    #expr = Piecewise((x, x < 1), (x**2, x > 1), (sin(x), x > 0))
    #raises(ValueError, lambda: rcode(expr))
    expr = 2*Piecewise((x, x < 1), (x**2, x<2))
    assert(rcode(expr))== "2*ifelse(x < 1,x,ifelse(x < 2,x^2,NA))"


def test_rcode_sinc():
    from sympy.functions.elementary.trigonometric import sinc
    expr = sinc(x)
    res = rcode(expr)
    ref = "(ifelse(x != 0,sin(x)/x,1))"
    assert res == ref


def test_rcode_Piecewise_deep():
    p = rcode(2*Piecewise((x, x < 1), (x + 1, x < 2), (x**2, True)))
    assert p == "2*ifelse(x < 1,x,ifelse(x < 2,x + 1,x^2))"
    expr = x*y*z + x**2 + y**2 + Piecewise((0, x < 0.5), (1, True)) + cos(z) - 1
    p = rcode(expr)
    ref="x^2 + x*y*z + y^2 + ifelse(x < 0.5,0,1) + cos(z) - 1"
    assert p == ref

    ref="c = x^2 + x*y*z + y^2 + ifelse(x < 0.5,0,1) + cos(z) - 1;"
    p = rcode(expr, assign_to='c')
    assert p == ref


def test_rcode_ITE():
    expr = ITE(x < 1, y, z)
    p = rcode(expr)
    ref="ifelse(x < 1,y,z)"
    assert p == ref


def test_rcode_settings():
    raises(TypeError, lambda: rcode(sin(x), method="garbage"))


def test_rcode_Indexed():
    n, m, o = symbols('n m o', integer=True)
    i, j, k = Idx('i', n), Idx('j', m), Idx('k', o)
    p = RCodePrinter()
    p._not_r = set()

    x = IndexedBase('x')[j]
    assert p._print_Indexed(x) == 'x[j]'
    A = IndexedBase('A')[i, j]
    assert p._print_Indexed(A) == 'A[i, j]'
    B = IndexedBase('B')[i, j, k]
    assert p._print_Indexed(B) == 'B[i, j, k]'

    assert p._not_r == set()

def test_rcode_Indexed_without_looking_for_contraction():
    len_y = 5
    y = IndexedBase('y', shape=(len_y,))
    x = IndexedBase('x', shape=(len_y,))
    Dy = IndexedBase('Dy', shape=(len_y-1,))
    i = Idx('i', len_y-1)
    e=Eq(Dy[i], (y[i+1]-y[i])/(x[i+1]-x[i]))
    code0 = rcode(e.rhs, assign_to=e.lhs, contract=False)
    assert code0 == 'Dy[i] = (y[%s] - y[i])/(x[%s] - x[i]);' % (i + 1, i + 1)


def test_rcode_loops_matrix_vector():
    n, m = symbols('n m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)

    s = (
        'for (i in 1:m){\n'
        '   y[i] = 0;\n'
        '}\n'
        'for (i in 1:m){\n'
        '   for (j in 1:n){\n'
        '      y[i] = A[i, j]*x[j] + y[i];\n'
        '   }\n'
        '}'
    )
    c = rcode(A[i, j]*x[j], assign_to=y[i])
    assert c == s


def test_dummy_loops():
    # the following line could also be
    # [Dummy(s, integer=True) for s in 'im']
    # or [Dummy(integer=True) for s in 'im']
    i, m = symbols('i m', integer=True, cls=Dummy)
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx(i, m)

    expected = (
            'for (i_%(icount)i in 1:m_%(mcount)i){\n'
        '   y[i_%(icount)i] = x[i_%(icount)i];\n'
        '}'
    ) % {'icount': i.label.dummy_index, 'mcount': m.dummy_index}
    code = rcode(x[i], assign_to=y[i])
    assert code == expected


def test_rcode_loops_add():
    n, m = symbols('n m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    z = IndexedBase('z')
    i = Idx('i', m)
    j = Idx('j', n)

    s = (
        'for (i in 1:m){\n'
        '   y[i] = x[i] + z[i];\n'
        '}\n'
        'for (i in 1:m){\n'
        '   for (j in 1:n){\n'
        '      y[i] = A[i, j]*x[j] + y[i];\n'
        '   }\n'
        '}'
    )
    c = rcode(A[i, j]*x[j] + x[i] + z[i], assign_to=y[i])
    assert c == s


def test_rcode_loops_multiple_contractions():
    n, m, o, p = symbols('n m o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)

    s = (
        'for (i in 1:m){\n'
        '   y[i] = 0;\n'
        '}\n'
        'for (i in 1:m){\n'
        '   for (j in 1:n){\n'
        '      for (k in 1:o){\n'
        '         for (l in 1:p){\n'
        '            y[i] = a[i, j, k, l]*b[j, k, l] + y[i];\n'
        '         }\n'
        '      }\n'
        '   }\n'
        '}'
    )
    c = rcode(b[j, k, l]*a[i, j, k, l], assign_to=y[i])
    assert c == s


def test_rcode_loops_addfactor():
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
        'for (i in 1:m){\n'
        '   y[i] = 0;\n'
        '}\n'
        'for (i in 1:m){\n'
        '   for (j in 1:n){\n'
        '      for (k in 1:o){\n'
        '         for (l in 1:p){\n'
        '            y[i] = (a[i, j, k, l] + b[i, j, k, l])*c[j, k, l] + y[i];\n'
        '         }\n'
        '      }\n'
        '   }\n'
        '}'
    )
    c = rcode((a[i, j, k, l] + b[i, j, k, l])*c[j, k, l], assign_to=y[i])
    assert c == s


def test_rcode_loops_multiple_terms():
    n, m, o, p = symbols('n m o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)

    s0 = (
        'for (i in 1:m){\n'
        '   y[i] = 0;\n'
        '}\n'
    )
    s1 = (
        'for (i in 1:m){\n'
        '   for (j in 1:n){\n'
        '      for (k in 1:o){\n'
        '         y[i] = b[j]*b[k]*c[i, j, k] + y[i];\n'
        '      }\n'
        '   }\n'
        '}\n'
    )
    s2 = (
        'for (i in 1:m){\n'
        '   for (k in 1:o){\n'
        '      y[i] = a[i, k]*b[k] + y[i];\n'
        '   }\n'
        '}\n'
    )
    s3 = (
        'for (i in 1:m){\n'
        '   for (j in 1:n){\n'
        '      y[i] = a[i, j]*b[j] + y[i];\n'
        '   }\n'
        '}\n'
    )
    c = rcode(
        b[j]*a[i, j] + b[k]*a[i, k] + b[j]*b[k]*c[i, j, k], assign_to=y[i])

    ref={}
    ref[0] = s0 + s1 + s2 + s3[:-1]
    ref[1] = s0 + s1 + s3 + s2[:-1]
    ref[2] = s0 + s2 + s1 + s3[:-1]
    ref[3] = s0 + s2 + s3 + s1[:-1]
    ref[4] = s0 + s3 + s1 + s2[:-1]
    ref[5] = s0 + s3 + s2 + s1[:-1]

    assert (c == ref[0] or
            c == ref[1] or
            c == ref[2] or
            c == ref[3] or
            c == ref[4] or
            c == ref[5])


def test_dereference_printing():
    expr = x + y + sin(z) + z
    assert rcode(expr, dereference=[z]) == "x + y + (*z) + sin((*z))"


def test_Matrix_printing():
    # Test returning a Matrix
    mat = Matrix([x*y, Piecewise((2 + x, y>0), (y, True)), sin(z)])
    A = MatrixSymbol('A', 3, 1)
    p = rcode(mat, A)
    assert p == (
        "A[0] = x*y;\n"
        "A[1] = ifelse(y > 0,x + 2,y);\n"
        "A[2] = sin(z);")
    # Test using MatrixElements in expressions
    expr = Piecewise((2*A[2, 0], x > 0), (A[2, 0], True)) + sin(A[1, 0]) + A[0, 0]
    p = rcode(expr)
    assert p  == ("ifelse(x > 0,2*A[2],A[2]) + sin(A[1]) + A[0]")
    # Test using MatrixElements in a Matrix
    q = MatrixSymbol('q', 5, 1)
    M = MatrixSymbol('M', 3, 3)
    m = Matrix([[sin(q[1,0]), 0, cos(q[2,0])],
        [q[1,0] + q[2,0], q[3, 0], 5],
        [2*q[4, 0]/q[1,0], sqrt(q[0,0]) + 4, 0]])
    assert rcode(m, M) == (
        "M[0] = sin(q[1]);\n"
        "M[1] = 0;\n"
        "M[2] = cos(q[2]);\n"
        "M[3] = q[1] + q[2];\n"
        "M[4] = q[3];\n"
        "M[5] = 5;\n"
        "M[6] = 2*q[4]/q[1];\n"
        "M[7] = sqrt(q[0]) + 4;\n"
        "M[8] = 0;")


def test_rcode_sgn():

    expr = sign(x) * y
    assert rcode(expr) == 'y*sign(x)'
    p = rcode(expr, 'z')
    assert p  == 'z = y*sign(x);'

    p = rcode(sign(2 * x + x**2) * x + x**2)
    assert p  == "x^2 + x*sign(x^2 + 2*x)"

    expr = sign(cos(x))
    p = rcode(expr)
    assert p == 'sign(cos(x))'

def test_rcode_Assignment():
    assert rcode(Assignment(x, y + z)) == 'x = y + z;'
    assert rcode(aug_assign(x, '+', y + z)) == 'x += y + z;'


def test_rcode_For():
    f = For(x, Range(0, 10, 2), [aug_assign(y, '*', x)])
    sol = rcode(f)
    assert sol == ("for(x in seq(from=0, to=9, by=2){\n"
                   "   y *= x;\n"
                   "}")


def test_MatrixElement_printing():
    # test cases for issue #11821
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)
    C = MatrixSymbol("C", 1, 3)

    assert(rcode(A[0, 0]) == "A[0]")
    assert(rcode(3 * A[0, 0]) == "3*A[0]")

    F = C[0, 0].subs(C, A - B)
    assert(rcode(F) == "(A - B)[0]")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\tests\test_spec_polynomials.py ===
from sympy.concrete.summations import Sum
from sympy.core.function import (Derivative, diff)
from sympy.core.numbers import (Rational, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.combinatorial.factorials import (RisingFactorial, binomial, factorial)
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper
from sympy.functions.special.polynomials import (assoc_laguerre, assoc_legendre, chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root, gegenbauer, hermite, hermite_prob, jacobi, jacobi_normalized, laguerre, legendre)
from sympy.polys.orthopolys import laguerre_poly
from sympy.polys.polyroots import roots

from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.testing.pytest import raises


x = Symbol('x')


def test_jacobi():
    n = Symbol("n")
    a = Symbol("a")
    b = Symbol("b")

    assert jacobi(0, a, b, x) == 1
    assert jacobi(1, a, b, x) == a/2 - b/2 + x*(a/2 + b/2 + 1)

    assert jacobi(n, a, a, x) == RisingFactorial(
        a + 1, n)*gegenbauer(n, a + S.Half, x)/RisingFactorial(2*a + 1, n)
    assert jacobi(n, a, -a, x) == ((-1)**a*(-x + 1)**(-a/2)*(x + 1)**(a/2)*assoc_legendre(n, a, x)*
                                   factorial(-a + n)*gamma(a + n + 1)/(factorial(a + n)*gamma(n + 1)))
    assert jacobi(n, -b, b, x) == ((-x + 1)**(b/2)*(x + 1)**(-b/2)*assoc_legendre(n, b, x)*
                                   gamma(-b + n + 1)/gamma(n + 1))
    assert jacobi(n, 0, 0, x) == legendre(n, x)
    assert jacobi(n, S.Half, S.Half, x) == RisingFactorial(
        Rational(3, 2), n)*chebyshevu(n, x)/factorial(n + 1)
    assert jacobi(n, Rational(-1, 2), Rational(-1, 2), x) == RisingFactorial(
        S.Half, n)*chebyshevt(n, x)/factorial(n)

    X = jacobi(n, a, b, x)
    assert isinstance(X, jacobi)

    assert jacobi(n, a, b, -x) == (-1)**n*jacobi(n, b, a, x)
    assert jacobi(n, a, b, 0) == 2**(-n)*gamma(a + n + 1)*hyper(
        (-b - n, -n), (a + 1,), -1)/(factorial(n)*gamma(a + 1))
    assert jacobi(n, a, b, 1) == RisingFactorial(a + 1, n)/factorial(n)

    m = Symbol("m", positive=True)
    assert jacobi(m, a, b, oo) == oo*RisingFactorial(a + b + m + 1, m)
    assert unchanged(jacobi, n, a, b, oo)

    assert conjugate(jacobi(m, a, b, x)) == \
        jacobi(m, conjugate(a), conjugate(b), conjugate(x))

    _k = Dummy('k')
    assert diff(jacobi(n, a, b, x), n) == Derivative(jacobi(n, a, b, x), n)
    assert diff(jacobi(n, a, b, x), a).dummy_eq(Sum((jacobi(n, a, b, x) +
        (2*_k + a + b + 1)*RisingFactorial(_k + b + 1, -_k + n)*jacobi(_k, a,
        b, x)/((-_k + n)*RisingFactorial(_k + a + b + 1, -_k + n)))/(_k + a
        + b + n + 1), (_k, 0, n - 1)))
    assert diff(jacobi(n, a, b, x), b).dummy_eq(Sum(((-1)**(-_k + n)*(2*_k +
        a + b + 1)*RisingFactorial(_k + a + 1, -_k + n)*jacobi(_k, a, b, x)/
        ((-_k + n)*RisingFactorial(_k + a + b + 1, -_k + n)) + jacobi(n, a,
        b, x))/(_k + a + b + n + 1), (_k, 0, n - 1)))
    assert diff(jacobi(n, a, b, x), x) == \
        (a/2 + b/2 + n/2 + S.Half)*jacobi(n - 1, a + 1, b + 1, x)

    assert jacobi_normalized(n, a, b, x) == \
           (jacobi(n, a, b, x)/sqrt(2**(a + b + 1)*gamma(a + n + 1)*gamma(b + n + 1)
                                    /((a + b + 2*n + 1)*factorial(n)*gamma(a + b + n + 1))))

    raises(ValueError, lambda: jacobi(-2.1, a, b, x))
    raises(ValueError, lambda: jacobi(Dummy(positive=True, integer=True), 1, 2, oo))

    assert jacobi(n, a, b, x).rewrite(Sum).dummy_eq(Sum((S.Half - x/2)
        **_k*RisingFactorial(-n, _k)*RisingFactorial(_k + a + 1, -_k + n)*
        RisingFactorial(a + b + n + 1, _k)/factorial(_k), (_k, 0, n))/factorial(n))
    assert jacobi(n, a, b, x).rewrite("polynomial").dummy_eq(Sum((S.Half - x/2)
        **_k*RisingFactorial(-n, _k)*RisingFactorial(_k + a + 1, -_k + n)*
        RisingFactorial(a + b + n + 1, _k)/factorial(_k), (_k, 0, n))/factorial(n))
    raises(ArgumentIndexError, lambda: jacobi(n, a, b, x).fdiff(5))


def test_gegenbauer():
    n = Symbol("n")
    a = Symbol("a")

    assert gegenbauer(0, a, x) == 1
    assert gegenbauer(1, a, x) == 2*a*x
    assert gegenbauer(2, a, x) == -a + x**2*(2*a**2 + 2*a)
    assert gegenbauer(3, a, x) == \
        x**3*(4*a**3/3 + 4*a**2 + a*Rational(8, 3)) + x*(-2*a**2 - 2*a)

    assert gegenbauer(-1, a, x) == 0
    assert gegenbauer(n, S.Half, x) == legendre(n, x)
    assert gegenbauer(n, 1, x) == chebyshevu(n, x)
    assert gegenbauer(n, -1, x) == 0

    X = gegenbauer(n, a, x)
    assert isinstance(X, gegenbauer)

    assert gegenbauer(n, a, -x) == (-1)**n*gegenbauer(n, a, x)
    assert gegenbauer(n, a, 0) == 2**n*sqrt(pi) * \
        gamma(a + n/2)/(gamma(a)*gamma(-n/2 + S.Half)*gamma(n + 1))
    assert gegenbauer(n, a, 1) == gamma(2*a + n)/(gamma(2*a)*gamma(n + 1))

    assert gegenbauer(n, Rational(3, 4), -1) is zoo
    assert gegenbauer(n, Rational(1, 4), -1) == (sqrt(2)*cos(pi*(n + S.One/4))*
                      gamma(n + S.Half)/(sqrt(pi)*gamma(n + 1)))

    m = Symbol("m", positive=True)
    assert gegenbauer(m, a, oo) == oo*RisingFactorial(a, m)
    assert unchanged(gegenbauer, n, a, oo)

    assert conjugate(gegenbauer(n, a, x)) == gegenbauer(n, conjugate(a), conjugate(x))

    _k = Dummy('k')

    assert diff(gegenbauer(n, a, x), n) == Derivative(gegenbauer(n, a, x), n)
    assert diff(gegenbauer(n, a, x), a).dummy_eq(Sum((2*(-1)**(-_k + n) + 2)*
        (_k + a)*gegenbauer(_k, a, x)/((-_k + n)*(_k + 2*a + n)) + ((2*_k +
        2)/((_k + 2*a)*(2*_k + 2*a + 1)) + 2/(_k + 2*a + n))*gegenbauer(n, a
        , x), (_k, 0, n - 1)))
    assert diff(gegenbauer(n, a, x), x) == 2*a*gegenbauer(n - 1, a + 1, x)

    assert gegenbauer(n, a, x).rewrite(Sum).dummy_eq(
        Sum((-1)**_k*(2*x)**(-2*_k + n)*RisingFactorial(a, -_k + n)
        /(factorial(_k)*factorial(-2*_k + n)), (_k, 0, floor(n/2))))
    assert gegenbauer(n, a, x).rewrite("polynomial").dummy_eq(
        Sum((-1)**_k*(2*x)**(-2*_k + n)*RisingFactorial(a, -_k + n)
        /(factorial(_k)*factorial(-2*_k + n)), (_k, 0, floor(n/2))))

    raises(ArgumentIndexError, lambda: gegenbauer(n, a, x).fdiff(4))


def test_legendre():
    assert legendre(0, x) == 1
    assert legendre(1, x) == x
    assert legendre(2, x) == ((3*x**2 - 1)/2).expand()
    assert legendre(3, x) == ((5*x**3 - 3*x)/2).expand()
    assert legendre(4, x) == ((35*x**4 - 30*x**2 + 3)/8).expand()
    assert legendre(5, x) == ((63*x**5 - 70*x**3 + 15*x)/8).expand()
    assert legendre(6, x) == ((231*x**6 - 315*x**4 + 105*x**2 - 5)/16).expand()

    assert legendre(10, -1) == 1
    assert legendre(11, -1) == -1
    assert legendre(10, 1) == 1
    assert legendre(11, 1) == 1
    assert legendre(10, 0) != 0
    assert legendre(11, 0) == 0

    assert legendre(-1, x) == 1
    k = Symbol('k')
    assert legendre(5 - k, x).subs(k, 2) == ((5*x**3 - 3*x)/2).expand()

    assert roots(legendre(4, x), x) == {
        sqrt(Rational(3, 7) - Rational(2, 35)*sqrt(30)): 1,
        -sqrt(Rational(3, 7) - Rational(2, 35)*sqrt(30)): 1,
        sqrt(Rational(3, 7) + Rational(2, 35)*sqrt(30)): 1,
        -sqrt(Rational(3, 7) + Rational(2, 35)*sqrt(30)): 1,
    }

    n = Symbol("n")

    X = legendre(n, x)
    assert isinstance(X, legendre)
    assert unchanged(legendre, n, x)

    assert legendre(n, 0) == sqrt(pi)/(gamma(S.Half - n/2)*gamma(n/2 + 1))
    assert legendre(n, 1) == 1
    assert legendre(n, oo) is oo
    assert legendre(-n, x) == legendre(n - 1, x)
    assert legendre(n, -x) == (-1)**n*legendre(n, x)
    assert unchanged(legendre, -n + k, x)

    assert conjugate(legendre(n, x)) == legendre(n, conjugate(x))

    assert diff(legendre(n, x), x) == \
        n*(x*legendre(n, x) - legendre(n - 1, x))/(x**2 - 1)
    assert diff(legendre(n, x), n) == Derivative(legendre(n, x), n)

    _k = Dummy('k')
    assert legendre(n, x).rewrite(Sum).dummy_eq(Sum((-1)**_k*(S.Half -
            x/2)**_k*(x/2 + S.Half)**(-_k + n)*binomial(n, _k)**2, (_k, 0, n)))
    assert legendre(n, x).rewrite("polynomial").dummy_eq(Sum((-1)**_k*(S.Half -
            x/2)**_k*(x/2 + S.Half)**(-_k + n)*binomial(n, _k)**2, (_k, 0, n)))
    raises(ArgumentIndexError, lambda: legendre(n, x).fdiff(1))
    raises(ArgumentIndexError, lambda: legendre(n, x).fdiff(3))


def test_assoc_legendre():
    Plm = assoc_legendre
    Q = sqrt(1 - x**2)

    assert Plm(0, 0, x) == 1
    assert Plm(1, 0, x) == x
    assert Plm(1, 1, x) == -Q
    assert Plm(2, 0, x) == (3*x**2 - 1)/2
    assert Plm(2, 1, x) == -3*x*Q
    assert Plm(2, 2, x) == 3*Q**2
    assert Plm(3, 0, x) == (5*x**3 - 3*x)/2
    assert Plm(3, 1, x).expand() == (( 3*(1 - 5*x**2)/2 ).expand() * Q).expand()
    assert Plm(3, 2, x) == 15*x * Q**2
    assert Plm(3, 3, x) == -15 * Q**3

    # negative m
    assert Plm(1, -1, x) == -Plm(1, 1, x)/2
    assert Plm(2, -2, x) == Plm(2, 2, x)/24
    assert Plm(2, -1, x) == -Plm(2, 1, x)/6
    assert Plm(3, -3, x) == -Plm(3, 3, x)/720
    assert Plm(3, -2, x) == Plm(3, 2, x)/120
    assert Plm(3, -1, x) == -Plm(3, 1, x)/12

    n = Symbol("n")
    m = Symbol("m")
    X = Plm(n, m, x)
    assert isinstance(X, assoc_legendre)

    assert Plm(n, 0, x) == legendre(n, x)
    assert Plm(n, m, 0) == 2**m*sqrt(pi)/(gamma(-m/2 - n/2 +
                           S.Half)*gamma(-m/2 + n/2 + 1))

    assert diff(Plm(m, n, x), x) == (m*x*assoc_legendre(m, n, x) -
                (m + n)*assoc_legendre(m - 1, n, x))/(x**2 - 1)

    _k = Dummy('k')
    assert Plm(m, n, x).rewrite(Sum).dummy_eq(
            (1 - x**2)**(n/2)*Sum((-1)**_k*2**(-m)*x**(-2*_k + m - n)*factorial
             (-2*_k + 2*m)/(factorial(_k)*factorial(-_k + m)*factorial(-2*_k + m
              - n)), (_k, 0, floor(m/2 - n/2))))
    assert Plm(m, n, x).rewrite("polynomial").dummy_eq(
            (1 - x**2)**(n/2)*Sum((-1)**_k*2**(-m)*x**(-2*_k + m - n)*factorial
             (-2*_k + 2*m)/(factorial(_k)*factorial(-_k + m)*factorial(-2*_k + m
              - n)), (_k, 0, floor(m/2 - n/2))))
    assert conjugate(assoc_legendre(n, m, x)) == \
        assoc_legendre(n, conjugate(m), conjugate(x))
    raises(ValueError, lambda: Plm(0, 1, x))
    raises(ValueError, lambda: Plm(-1, 1, x))
    raises(ArgumentIndexError, lambda: Plm(n, m, x).fdiff(1))
    raises(ArgumentIndexError, lambda: Plm(n, m, x).fdiff(2))
    raises(ArgumentIndexError, lambda: Plm(n, m, x).fdiff(4))


def test_chebyshev():
    assert chebyshevt(0, x) == 1
    assert chebyshevt(1, x) == x
    assert chebyshevt(2, x) == 2*x**2 - 1
    assert chebyshevt(3, x) == 4*x**3 - 3*x

    for n in range(1, 4):
        for k in range(n):
            z = chebyshevt_root(n, k)
            assert chebyshevt(n, z) == 0
        raises(ValueError, lambda: chebyshevt_root(n, n))

    for n in range(1, 4):
        for k in range(n):
            z = chebyshevu_root(n, k)
            assert chebyshevu(n, z) == 0
        raises(ValueError, lambda: chebyshevu_root(n, n))

    n = Symbol("n")
    X = chebyshevt(n, x)
    assert isinstance(X, chebyshevt)
    assert unchanged(chebyshevt, n, x)
    assert chebyshevt(n, -x) == (-1)**n*chebyshevt(n, x)
    assert chebyshevt(-n, x) == chebyshevt(n, x)

    assert chebyshevt(n, 0) == cos(pi*n/2)
    assert chebyshevt(n, 1) == 1
    assert chebyshevt(n, oo) is oo

    assert conjugate(chebyshevt(n, x)) == chebyshevt(n, conjugate(x))

    assert diff(chebyshevt(n, x), x) == n*chebyshevu(n - 1, x)

    X = chebyshevu(n, x)
    assert isinstance(X, chebyshevu)

    y = Symbol('y')
    assert chebyshevu(n, -x) == (-1)**n*chebyshevu(n, x)
    assert chebyshevu(-n, x) == -chebyshevu(n - 2, x)
    assert unchanged(chebyshevu, -n + y, x)

    assert chebyshevu(n, 0) == cos(pi*n/2)
    assert chebyshevu(n, 1) == n + 1
    assert chebyshevu(n, oo) is oo

    assert conjugate(chebyshevu(n, x)) == chebyshevu(n, conjugate(x))

    assert diff(chebyshevu(n, x), x) == \
        (-x*chebyshevu(n, x) + (n + 1)*chebyshevt(n + 1, x))/(x**2 - 1)

    _k = Dummy('k')
    assert chebyshevt(n, x).rewrite(Sum).dummy_eq(Sum(x**(-2*_k + n)
                    *(x**2 - 1)**_k*binomial(n, 2*_k), (_k, 0, floor(n/2))))
    assert chebyshevt(n, x).rewrite("polynomial").dummy_eq(Sum(x**(-2*_k + n)
                    *(x**2 - 1)**_k*binomial(n, 2*_k), (_k, 0, floor(n/2))))
    assert chebyshevu(n, x).rewrite(Sum).dummy_eq(Sum((-1)**_k*(2*x)
                    **(-2*_k + n)*factorial(-_k + n)/(factorial(_k)*
                       factorial(-2*_k + n)), (_k, 0, floor(n/2))))
    assert chebyshevu(n, x).rewrite("polynomial").dummy_eq(Sum((-1)**_k*(2*x)
                    **(-2*_k + n)*factorial(-_k + n)/(factorial(_k)*
                       factorial(-2*_k + n)), (_k, 0, floor(n/2))))
    raises(ArgumentIndexError, lambda: chebyshevt(n, x).fdiff(1))
    raises(ArgumentIndexError, lambda: chebyshevt(n, x).fdiff(3))
    raises(ArgumentIndexError, lambda: chebyshevu(n, x).fdiff(1))
    raises(ArgumentIndexError, lambda: chebyshevu(n, x).fdiff(3))


def test_hermite():
    assert hermite(0, x) == 1
    assert hermite(1, x) == 2*x
    assert hermite(2, x) == 4*x**2 - 2
    assert hermite(3, x) == 8*x**3 - 12*x
    assert hermite(4, x) == 16*x**4 - 48*x**2 + 12
    assert hermite(6, x) == 64*x**6 - 480*x**4 + 720*x**2 - 120

    n = Symbol("n")
    assert unchanged(hermite, n, x)
    assert hermite(n, -x) == (-1)**n*hermite(n, x)
    assert unchanged(hermite, -n, x)

    assert hermite(n, 0) == 2**n*sqrt(pi)/gamma(S.Half - n/2)
    assert hermite(n, oo) is oo

    assert conjugate(hermite(n, x)) == hermite(n, conjugate(x))

    _k = Dummy('k')
    assert hermite(n, x).rewrite(Sum).dummy_eq(factorial(n)*Sum((-1)
        **_k*(2*x)**(-2*_k + n)/(factorial(_k)*factorial(-2*_k + n)), (_k,
        0, floor(n/2))))
    assert hermite(n, x).rewrite("polynomial").dummy_eq(factorial(n)*Sum((-1)
        **_k*(2*x)**(-2*_k + n)/(factorial(_k)*factorial(-2*_k + n)), (_k,
        0, floor(n/2))))

    assert diff(hermite(n, x), x) == 2*n*hermite(n - 1, x)
    assert diff(hermite(n, x), n) == Derivative(hermite(n, x), n)
    raises(ArgumentIndexError, lambda: hermite(n, x).fdiff(3))

    assert hermite(n, x).rewrite(hermite_prob) == \
            sqrt(2)**n * hermite_prob(n, x*sqrt(2))


def test_hermite_prob():
    assert hermite_prob(0, x) == 1
    assert hermite_prob(1, x) == x
    assert hermite_prob(2, x) == x**2 - 1
    assert hermite_prob(3, x) == x**3 - 3*x
    assert hermite_prob(4, x) == x**4 - 6*x**2 + 3
    assert hermite_prob(6, x) == x**6 - 15*x**4 + 45*x**2 - 15

    n = Symbol("n")
    assert unchanged(hermite_prob, n, x)
    assert hermite_prob(n, -x) == (-1)**n*hermite_prob(n, x)
    assert unchanged(hermite_prob, -n, x)

    assert hermite_prob(n, 0) == sqrt(pi)/gamma(S.Half - n/2)
    assert hermite_prob(n, oo) is oo

    assert conjugate(hermite_prob(n, x)) == hermite_prob(n, conjugate(x))

    _k = Dummy('k')
    assert hermite_prob(n, x).rewrite(Sum).dummy_eq(factorial(n) *
        Sum((-S.Half)**_k * x**(n-2*_k) / (factorial(_k) * factorial(n-2*_k)),
        (_k, 0, floor(n/2))))
    assert hermite_prob(n, x).rewrite("polynomial").dummy_eq(factorial(n) *
        Sum((-S.Half)**_k * x**(n-2*_k) / (factorial(_k) * factorial(n-2*_k)),
        (_k, 0, floor(n/2))))

    assert diff(hermite_prob(n, x), x) == n*hermite_prob(n-1, x)
    assert diff(hermite_prob(n, x), n) == Derivative(hermite_prob(n, x), n)
    raises(ArgumentIndexError, lambda: hermite_prob(n, x).fdiff(3))

    assert hermite_prob(n, x).rewrite(hermite) == \
            sqrt(2)**(-n) * hermite(n, x/sqrt(2))


def test_laguerre():
    n = Symbol("n")
    m = Symbol("m", negative=True)

    # Laguerre polynomials:
    assert laguerre(0, x) == 1
    assert laguerre(1, x) == -x + 1
    assert laguerre(2, x) == x**2/2 - 2*x + 1
    assert laguerre(3, x) == -x**3/6 + 3*x**2/2 - 3*x + 1
    assert laguerre(-2, x) == (x + 1)*exp(x)

    X = laguerre(n, x)
    assert isinstance(X, laguerre)

    assert laguerre(n, 0) == 1
    assert laguerre(n, oo) == (-1)**n*oo
    assert laguerre(n, -oo) is oo

    assert conjugate(laguerre(n, x)) == laguerre(n, conjugate(x))

    _k = Dummy('k')

    assert laguerre(n, x).rewrite(Sum).dummy_eq(
        Sum(x**_k*RisingFactorial(-n, _k)/factorial(_k)**2, (_k, 0, n)))
    assert laguerre(n, x).rewrite("polynomial").dummy_eq(
        Sum(x**_k*RisingFactorial(-n, _k)/factorial(_k)**2, (_k, 0, n)))
    assert laguerre(m, x).rewrite(Sum).dummy_eq(
        exp(x)*Sum((-x)**_k*RisingFactorial(m + 1, _k)/factorial(_k)**2,
            (_k, 0, -m - 1)))
    assert laguerre(m, x).rewrite("polynomial").dummy_eq(
        exp(x)*Sum((-x)**_k*RisingFactorial(m + 1, _k)/factorial(_k)**2,
            (_k, 0, -m - 1)))

    assert diff(laguerre(n, x), x) == -assoc_laguerre(n - 1, 1, x)

    k = Symbol('k')
    assert laguerre(-n, x) == exp(x)*laguerre(n - 1, -x)
    assert laguerre(-3, x) == exp(x)*laguerre(2, -x)
    assert unchanged(laguerre, -n + k, x)

    raises(ValueError, lambda: laguerre(-2.1, x))
    raises(ValueError, lambda: laguerre(Rational(5, 2), x))
    raises(ArgumentIndexError, lambda: laguerre(n, x).fdiff(1))
    raises(ArgumentIndexError, lambda: laguerre(n, x).fdiff(3))


def test_assoc_laguerre():
    n = Symbol("n")
    m = Symbol("m")
    alpha = Symbol("alpha")

    # generalized Laguerre polynomials:
    assert assoc_laguerre(0, alpha, x) == 1
    assert assoc_laguerre(1, alpha, x) == -x + alpha + 1
    assert assoc_laguerre(2, alpha, x).expand() == \
        (x**2/2 - (alpha + 2)*x + (alpha + 2)*(alpha + 1)/2).expand()
    assert assoc_laguerre(3, alpha, x).expand() == \
        (-x**3/6 + (alpha + 3)*x**2/2 - (alpha + 2)*(alpha + 3)*x/2 +
        (alpha + 1)*(alpha + 2)*(alpha + 3)/6).expand()

    # Test the lowest 10 polynomials with laguerre_poly, to make sure it works:
    for i in range(10):
        assert assoc_laguerre(i, 0, x).expand() == laguerre_poly(i, x)

    X = assoc_laguerre(n, m, x)
    assert isinstance(X, assoc_laguerre)

    assert assoc_laguerre(n, 0, x) == laguerre(n, x)
    assert assoc_laguerre(n, alpha, 0) == binomial(alpha + n, alpha)
    p = Symbol("p", positive=True)
    assert assoc_laguerre(p, alpha, oo) == (-1)**p*oo
    assert assoc_laguerre(p, alpha, -oo) is oo

    assert diff(assoc_laguerre(n, alpha, x), x) == \
        -assoc_laguerre(n - 1, alpha + 1, x)
    _k = Dummy('k')
    assert diff(assoc_laguerre(n, alpha, x), alpha).dummy_eq(
        Sum(assoc_laguerre(_k, alpha, x)/(-alpha + n), (_k, 0, n - 1)))

    assert conjugate(assoc_laguerre(n, alpha, x)) == \
        assoc_laguerre(n, conjugate(alpha), conjugate(x))

    assert assoc_laguerre(n, alpha, x).rewrite(Sum).dummy_eq(
            gamma(alpha + n + 1)*Sum(x**_k*RisingFactorial(-n, _k)/
            (factorial(_k)*gamma(_k + alpha + 1)), (_k, 0, n))/factorial(n))
    assert assoc_laguerre(n, alpha, x).rewrite("polynomial").dummy_eq(
            gamma(alpha + n + 1)*Sum(x**_k*RisingFactorial(-n, _k)/
            (factorial(_k)*gamma(_k + alpha + 1)), (_k, 0, n))/factorial(n))
    raises(ValueError, lambda: assoc_laguerre(-2.1, alpha, x))
    raises(ArgumentIndexError, lambda: assoc_laguerre(n, alpha, x).fdiff(1))
    raises(ArgumentIndexError, lambda: assoc_laguerre(n, alpha, x).fdiff(4))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\tests\test_decompositions.py ===
from sympy.core.function import expand_mul
from sympy.core.numbers import I, Rational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.complexes import Abs
from sympy.simplify.simplify import simplify
from sympy.matrices.exceptions import NonSquareMatrixError
from sympy.matrices import Matrix, zeros, eye, SparseMatrix
from sympy.abc import x, y, z
from sympy.testing.pytest import raises, slow
from sympy.testing.matrices import allclose


def test_LUdecomp():
    testmat = Matrix([[0, 2, 5, 3],
                      [3, 3, 7, 4],
                      [8, 4, 0, 2],
                      [-2, 6, 3, 4]])
    L, U, p = testmat.LUdecomposition()
    assert L.is_lower
    assert U.is_upper
    assert (L*U).permute_rows(p, 'backward') - testmat == zeros(4)

    testmat = Matrix([[6, -2, 7, 4],
                      [0, 3, 6, 7],
                      [1, -2, 7, 4],
                      [-9, 2, 6, 3]])
    L, U, p = testmat.LUdecomposition()
    assert L.is_lower
    assert U.is_upper
    assert (L*U).permute_rows(p, 'backward') - testmat == zeros(4)

    # non-square
    testmat = Matrix([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9],
                      [10, 11, 12]])
    L, U, p = testmat.LUdecomposition(rankcheck=False)
    assert L.is_lower
    assert U.is_upper
    assert (L*U).permute_rows(p, 'backward') - testmat == zeros(4, 3)

    # square and singular
    testmat = Matrix([[1, 2, 3],
                      [2, 4, 6],
                      [4, 5, 6]])
    L, U, p = testmat.LUdecomposition(rankcheck=False)
    assert L.is_lower
    assert U.is_upper
    assert (L*U).permute_rows(p, 'backward') - testmat == zeros(3)

    M = Matrix(((1, x, 1), (2, y, 0), (y, 0, z)))
    L, U, p = M.LUdecomposition()
    assert L.is_lower
    assert U.is_upper
    assert (L*U).permute_rows(p, 'backward') - M == zeros(3)

    mL = Matrix((
        (1, 0, 0),
        (2, 3, 0),
    ))
    assert mL.is_lower is True
    assert mL.is_upper is False
    mU = Matrix((
        (1, 2, 3),
        (0, 4, 5),
    ))
    assert mU.is_lower is False
    assert mU.is_upper is True

    # test FF LUdecomp
    M = Matrix([[1, 3, 3],
                [3, 2, 6],
                [3, 2, 2]])
    P, L, Dee, U = M.LUdecompositionFF()
    assert P*M == L*Dee.inv()*U

    M = Matrix([[1,  2, 3,  4],
                [3, -1, 2,  3],
                [3,  1, 3, -2],
                [6, -1, 0,  2]])
    P, L, Dee, U = M.LUdecompositionFF()
    assert P*M == L*Dee.inv()*U

    M = Matrix([[0, 0, 1],
                [2, 3, 0],
                [3, 1, 4]])
    P, L, Dee, U = M.LUdecompositionFF()
    assert P*M == L*Dee.inv()*U

    # issue 15794
    M = Matrix(
        [[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]
    )
    raises(ValueError, lambda : M.LUdecomposition_Simple(rankcheck=True))

def test_singular_value_decompositionD():
    A = Matrix([[1, 2], [2, 1]])
    U, S, V = A.singular_value_decomposition()
    assert U * S * V.T == A
    assert U.T * U == eye(U.cols)
    assert V.T * V == eye(V.cols)

    B = Matrix([[1, 2]])
    U, S, V = B.singular_value_decomposition()

    assert U * S * V.T == B
    assert U.T * U == eye(U.cols)
    assert V.T * V == eye(V.cols)

    C = Matrix([
        [1, 0, 0, 0, 2],
        [0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0],
    ])

    U, S, V = C.singular_value_decomposition()

    assert U * S * V.T == C
    assert U.T * U == eye(U.cols)
    assert V.T * V == eye(V.cols)

    D = Matrix([[Rational(1, 3), sqrt(2)], [0, Rational(1, 4)]])
    U, S, V = D.singular_value_decomposition()
    assert simplify(U.T * U) == eye(U.cols)
    assert simplify(V.T * V) == eye(V.cols)
    assert simplify(U * S * V.T) == D


def test_QR():
    A = Matrix([[1, 2], [2, 3]])
    Q, S = A.QRdecomposition()
    R = Rational
    assert Q == Matrix([
        [  5**R(-1, 2),  (R(2)/5)*(R(1)/5)**R(-1, 2)],
        [2*5**R(-1, 2), (-R(1)/5)*(R(1)/5)**R(-1, 2)]])
    assert S == Matrix([[5**R(1, 2), 8*5**R(-1, 2)], [0, (R(1)/5)**R(1, 2)]])
    assert Q*S == A
    assert Q.T * Q == eye(2)

    A = Matrix([[1, 1, 1], [1, 1, 3], [2, 3, 4]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[12, 0, -51], [6, 0, 167], [-4, 0, 24]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    x = Symbol('x')
    A = Matrix([x])
    Q, R = A.QRdecomposition()
    assert Q == Matrix([x / Abs(x)])
    assert R == Matrix([Abs(x)])

    A = Matrix([[x, 0], [0, x]])
    Q, R = A.QRdecomposition()
    assert Q == x / Abs(x) * Matrix([[1, 0], [0, 1]])
    assert R == Abs(x) * Matrix([[1, 0], [0, 1]])


def test_QR_non_square():
    # Narrow (cols < rows) matrices
    A = Matrix([[9, 0, 26], [12, 0, -7], [0, 4, 4], [0, -3, -3]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[1, -1, 4], [1, 4, -2], [1, 4, 2], [1, -1, 0]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix(2, 1, [1, 2])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    # Wide (cols > rows) matrices
    A = Matrix([[1, 2, 3], [4, 5, 6]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[1, 2, 3, 4], [1, 4, 9, 16], [1, 8, 27, 64]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix(1, 2, [1, 2])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

def test_QR_trivial():
    # Rank deficient matrices
    A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]).T
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    # Zero rank matrices
    A = Matrix([[0, 0, 0]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[0, 0, 0]]).T
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[0, 0, 0], [0, 0, 0]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[0, 0, 0], [0, 0, 0]]).T
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    # Rank deficient matrices with zero norm from beginning columns
    A = Matrix([[0, 0, 0], [1, 2, 3]]).T
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[0, 0, 0, 0], [1, 2, 3, 4], [0, 0, 0, 0]]).T
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[0, 0, 0, 0], [1, 2, 3, 4], [0, 0, 0, 0], [2, 4, 6, 8]]).T
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R

    A = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 2, 3]]).T
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q*R


def test_QR_float():
    A = Matrix([[1, 1], [1, 1.01]])
    Q, R = A.QRdecomposition()
    assert allclose(Q * R, A)
    assert allclose(Q * Q.T, Matrix.eye(2))
    assert allclose(Q.T * Q, Matrix.eye(2))

    A = Matrix([[1, 1], [1, 1.001]])
    Q, R = A.QRdecomposition()
    assert allclose(Q * R, A)
    assert allclose(Q * Q.T, Matrix.eye(2))
    assert allclose(Q.T * Q, Matrix.eye(2))


def test_LUdecomposition_Simple_iszerofunc():
    # Test if callable passed to matrices.LUdecomposition_Simple() as iszerofunc keyword argument is used inside
    # matrices.LUdecomposition_Simple()
    magic_string = "I got passed in!"
    def goofyiszero(value):
        raise ValueError(magic_string)

    try:
        lu, p = Matrix([[1, 0], [0, 1]]).LUdecomposition_Simple(iszerofunc=goofyiszero)
    except ValueError as err:
        assert magic_string == err.args[0]
        return

    assert False

def test_LUdecomposition_iszerofunc():
    # Test if callable passed to matrices.LUdecomposition() as iszerofunc keyword argument is used inside
    # matrices.LUdecomposition_Simple()
    magic_string = "I got passed in!"
    def goofyiszero(value):
        raise ValueError(magic_string)

    try:
        l, u, p = Matrix([[1, 0], [0, 1]]).LUdecomposition(iszerofunc=goofyiszero)
    except ValueError as err:
        assert magic_string == err.args[0]
        return

    assert False

def test_LDLdecomposition():
    raises(NonSquareMatrixError, lambda: Matrix((1, 2)).LDLdecomposition())
    raises(ValueError, lambda: Matrix(((1, 2), (3, 4))).LDLdecomposition())
    raises(ValueError, lambda: Matrix(((5 + I, 0), (0, 1))).LDLdecomposition())
    raises(ValueError, lambda: Matrix(((1, 5), (5, 1))).LDLdecomposition())
    raises(ValueError, lambda: Matrix(((1, 2), (3, 4))).LDLdecomposition(hermitian=False))
    A = Matrix(((1, 5), (5, 1)))
    L, D = A.LDLdecomposition(hermitian=False)
    assert L * D * L.T == A
    A = Matrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))
    L, D = A.LDLdecomposition()
    assert L * D * L.T == A
    assert L.is_lower
    assert L == Matrix([[1, 0, 0], [ Rational(3, 5), 1, 0], [Rational(-1, 5), Rational(1, 3), 1]])
    assert D.is_diagonal()
    assert D == Matrix([[25, 0, 0], [0, 9, 0], [0, 0, 9]])
    A = Matrix(((4, -2*I, 2 + 2*I), (2*I, 2, -1 + I), (2 - 2*I, -1 - I, 11)))
    L, D = A.LDLdecomposition()
    assert expand_mul(L * D * L.H) == A
    assert L.expand() == Matrix([[1, 0, 0], [I/2, 1, 0], [S.Half - I/2, 0, 1]])
    assert D.expand() == Matrix(((4, 0, 0), (0, 1, 0), (0, 0, 9)))

    raises(NonSquareMatrixError, lambda: SparseMatrix((1, 2)).LDLdecomposition())
    raises(ValueError, lambda: SparseMatrix(((1, 2), (3, 4))).LDLdecomposition())
    raises(ValueError, lambda: SparseMatrix(((5 + I, 0), (0, 1))).LDLdecomposition())
    raises(ValueError, lambda: SparseMatrix(((1, 5), (5, 1))).LDLdecomposition())
    raises(ValueError, lambda: SparseMatrix(((1, 2), (3, 4))).LDLdecomposition(hermitian=False))
    A = SparseMatrix(((1, 5), (5, 1)))
    L, D = A.LDLdecomposition(hermitian=False)
    assert L * D * L.T == A
    A = SparseMatrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))
    L, D = A.LDLdecomposition()
    assert L * D * L.T == A
    assert L.is_lower
    assert L == Matrix([[1, 0, 0], [ Rational(3, 5), 1, 0], [Rational(-1, 5), Rational(1, 3), 1]])
    assert D.is_diagonal()
    assert D == Matrix([[25, 0, 0], [0, 9, 0], [0, 0, 9]])
    A = SparseMatrix(((4, -2*I, 2 + 2*I), (2*I, 2, -1 + I), (2 - 2*I, -1 - I, 11)))
    L, D = A.LDLdecomposition()
    assert expand_mul(L * D * L.H) == A
    assert L == Matrix(((1, 0, 0), (I/2, 1, 0), (S.Half - I/2, 0, 1)))
    assert D == Matrix(((4, 0, 0), (0, 1, 0), (0, 0, 9)))

def test_pinv_succeeds_with_rank_decomposition_method():
    # Test rank decomposition method of pseudoinverse succeeding
    As = [Matrix([
        [61, 89, 55, 20, 71, 0],
        [62, 96, 85, 85, 16, 0],
        [69, 56, 17,  4, 54, 0],
        [10, 54, 91, 41, 71, 0],
        [ 7, 30, 10, 48, 90, 0],
        [0,0,0,0,0,0]])]
    for A in As:
        A_pinv = A.pinv(method="RD")
        AAp = A * A_pinv
        ApA = A_pinv * A
        assert simplify(AAp * A) == A
        assert simplify(ApA * A_pinv) == A_pinv
        assert AAp.H == AAp
        assert ApA.H == ApA

def test_rank_decomposition():
    a = Matrix(0, 0, [])
    c, f = a.rank_decomposition()
    assert f.is_echelon
    assert c.cols == f.rows == a.rank()
    assert c * f == a

    a = Matrix(1, 1, [5])
    c, f = a.rank_decomposition()
    assert f.is_echelon
    assert c.cols == f.rows == a.rank()
    assert c * f == a

    a = Matrix(3, 3, [1, 2, 3, 1, 2, 3, 1, 2, 3])
    c, f = a.rank_decomposition()
    assert f.is_echelon
    assert c.cols == f.rows == a.rank()
    assert c * f == a

    a = Matrix([
        [0, 0, 1, 2, 2, -5, 3],
        [-1, 5, 2, 2, 1, -7, 5],
        [0, 0, -2, -3, -3, 8, -5],
        [-1, 5, 0, -1, -2, 1, 0]])
    c, f = a.rank_decomposition()
    assert f.is_echelon
    assert c.cols == f.rows == a.rank()
    assert c * f == a


@slow
def test_upper_hessenberg_decomposition():
    A = Matrix([
        [1, 0, sqrt(3)],
        [sqrt(2), Rational(1, 2), 2],
        [1, Rational(1, 4), 3],
    ])
    H, P = A.upper_hessenberg_decomposition()
    assert simplify(P * P.H) == eye(P.cols)
    assert simplify(P.H * P) == eye(P.cols)
    assert H.is_upper_hessenberg
    assert (simplify(P * H * P.H)) == A


    B = Matrix([
        [1, 2, 10],
        [8, 2, 5],
        [3, 12, 34],
    ])
    H, P = B.upper_hessenberg_decomposition()
    assert simplify(P * P.H) == eye(P.cols)
    assert simplify(P.H * P) == eye(P.cols)
    assert H.is_upper_hessenberg
    assert simplify(P * H * P.H) == B

    C = Matrix([
        [1, sqrt(2), 2, 3],
        [0, 5, 3, 4],
        [1, 1, 4, sqrt(5)],
        [0, 2, 2, 3]
    ])

    H, P = C.upper_hessenberg_decomposition()
    assert simplify(P * P.H) == eye(P.cols)
    assert simplify(P.H * P) == eye(P.cols)
    assert H.is_upper_hessenberg
    assert simplify(P * H * P.H) == C

    D = Matrix([
        [1, 2, 3],
        [-3, 5, 6],
        [4, -8, 9],
    ])
    H, P = D.upper_hessenberg_decomposition()
    assert simplify(P * P.H) == eye(P.cols)
    assert simplify(P.H * P) == eye(P.cols)
    assert H.is_upper_hessenberg
    assert simplify(P * H * P.H) == D

    E = Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0]
    ])

    H, P = E.upper_hessenberg_decomposition()
    assert simplify(P * P.H) == eye(P.cols)
    assert simplify(P.H * P) == eye(P.cols)
    assert H.is_upper_hessenberg
    assert simplify(P * H * P.H) == E

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_drop_duplicates.py ===
from datetime import datetime
import re

import numpy as np
import pytest

from pandas import (
    DataFrame,
    NaT,
    concat,
)
import pandas._testing as tm


@pytest.mark.parametrize("subset", ["a", ["a"], ["a", "B"]])
def test_drop_duplicates_with_misspelled_column_name(subset):
    # GH 19730
    df = DataFrame({"A": [0, 0, 1], "B": [0, 0, 1], "C": [0, 0, 1]})
    msg = re.escape("Index(['a'], dtype=")

    with pytest.raises(KeyError, match=msg):
        df.drop_duplicates(subset)


def test_drop_duplicates():
    df = DataFrame(
        {
            "AAA": ["foo", "bar", "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": range(8),
        }
    )
    # single column
    result = df.drop_duplicates("AAA")
    expected = df[:2]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("AAA", keep="last")
    expected = df.loc[[6, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("AAA", keep=False)
    expected = df.loc[[]]
    tm.assert_frame_equal(result, expected)
    assert len(result) == 0

    # multi column
    expected = df.loc[[0, 1, 2, 3]]
    result = df.drop_duplicates(np.array(["AAA", "B"]))
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates(["AAA", "B"])
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates(("AAA", "B"), keep="last")
    expected = df.loc[[0, 5, 6, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates(("AAA", "B"), keep=False)
    expected = df.loc[[0]]
    tm.assert_frame_equal(result, expected)

    # consider everything
    df2 = df.loc[:, ["AAA", "B", "C"]]

    result = df2.drop_duplicates()
    # in this case only
    expected = df2.drop_duplicates(["AAA", "B"])
    tm.assert_frame_equal(result, expected)

    result = df2.drop_duplicates(keep="last")
    expected = df2.drop_duplicates(["AAA", "B"], keep="last")
    tm.assert_frame_equal(result, expected)

    result = df2.drop_duplicates(keep=False)
    expected = df2.drop_duplicates(["AAA", "B"], keep=False)
    tm.assert_frame_equal(result, expected)

    # integers
    result = df.drop_duplicates("C")
    expected = df.iloc[[0, 2]]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates("C", keep="last")
    expected = df.iloc[[-2, -1]]
    tm.assert_frame_equal(result, expected)

    df["E"] = df["C"].astype("int8")
    result = df.drop_duplicates("E")
    expected = df.iloc[[0, 2]]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates("E", keep="last")
    expected = df.iloc[[-2, -1]]
    tm.assert_frame_equal(result, expected)

    # GH 11376
    df = DataFrame({"x": [7, 6, 3, 3, 4, 8, 0], "y": [0, 6, 5, 5, 9, 1, 2]})
    expected = df.loc[df.index != 3]
    tm.assert_frame_equal(df.drop_duplicates(), expected)

    df = DataFrame([[1, 0], [0, 2]])
    tm.assert_frame_equal(df.drop_duplicates(), df)

    df = DataFrame([[-2, 0], [0, -4]])
    tm.assert_frame_equal(df.drop_duplicates(), df)

    x = np.iinfo(np.int64).max / 3 * 2
    df = DataFrame([[-x, x], [0, x + 4]])
    tm.assert_frame_equal(df.drop_duplicates(), df)

    df = DataFrame([[-x, x], [x, x + 4]])
    tm.assert_frame_equal(df.drop_duplicates(), df)

    # GH 11864
    df = DataFrame([i] * 9 for i in range(16))
    df = concat([df, DataFrame([[1] + [0] * 8])], ignore_index=True)

    for keep in ["first", "last", False]:
        assert df.duplicated(keep=keep).sum() == 0


def test_drop_duplicates_with_duplicate_column_names():
    # GH17836
    df = DataFrame([[1, 2, 5], [3, 4, 6], [3, 4, 7]], columns=["a", "a", "b"])

    result0 = df.drop_duplicates()
    tm.assert_frame_equal(result0, df)

    result1 = df.drop_duplicates("a")
    expected1 = df[:2]
    tm.assert_frame_equal(result1, expected1)


def test_drop_duplicates_for_take_all():
    df = DataFrame(
        {
            "AAA": ["foo", "bar", "baz", "bar", "foo", "bar", "qux", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": range(8),
        }
    )
    # single column
    result = df.drop_duplicates("AAA")
    expected = df.iloc[[0, 1, 2, 6]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("AAA", keep="last")
    expected = df.iloc[[2, 5, 6, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("AAA", keep=False)
    expected = df.iloc[[2, 6]]
    tm.assert_frame_equal(result, expected)

    # multiple columns
    result = df.drop_duplicates(["AAA", "B"])
    expected = df.iloc[[0, 1, 2, 3, 4, 6]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates(["AAA", "B"], keep="last")
    expected = df.iloc[[0, 1, 2, 5, 6, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates(["AAA", "B"], keep=False)
    expected = df.iloc[[0, 1, 2, 6]]
    tm.assert_frame_equal(result, expected)


def test_drop_duplicates_tuple():
    df = DataFrame(
        {
            ("AA", "AB"): ["foo", "bar", "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": range(8),
        }
    )
    # single column
    result = df.drop_duplicates(("AA", "AB"))
    expected = df[:2]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates(("AA", "AB"), keep="last")
    expected = df.loc[[6, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates(("AA", "AB"), keep=False)
    expected = df.loc[[]]  # empty df
    assert len(result) == 0
    tm.assert_frame_equal(result, expected)

    # multi column
    expected = df.loc[[0, 1, 2, 3]]
    result = df.drop_duplicates((("AA", "AB"), "B"))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "df",
    [
        DataFrame(),
        DataFrame(columns=[]),
        DataFrame(columns=["A", "B", "C"]),
        DataFrame(index=[]),
        DataFrame(index=["A", "B", "C"]),
    ],
)
def test_drop_duplicates_empty(df):
    # GH 20516
    result = df.drop_duplicates()
    tm.assert_frame_equal(result, df)

    result = df.copy()
    result.drop_duplicates(inplace=True)
    tm.assert_frame_equal(result, df)


def test_drop_duplicates_NA():
    # none
    df = DataFrame(
        {
            "A": [None, None, "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1.0, np.nan, np.nan, np.nan, 1.0, 1.0, 1, 1.0],
            "D": range(8),
        }
    )
    # single column
    result = df.drop_duplicates("A")
    expected = df.loc[[0, 2, 3]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("A", keep="last")
    expected = df.loc[[1, 6, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("A", keep=False)
    expected = df.loc[[]]  # empty df
    tm.assert_frame_equal(result, expected)
    assert len(result) == 0

    # multi column
    result = df.drop_duplicates(["A", "B"])
    expected = df.loc[[0, 2, 3, 6]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates(["A", "B"], keep="last")
    expected = df.loc[[1, 5, 6, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates(["A", "B"], keep=False)
    expected = df.loc[[6]]
    tm.assert_frame_equal(result, expected)

    # nan
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1.0, np.nan, np.nan, np.nan, 1.0, 1.0, 1, 1.0],
            "D": range(8),
        }
    )
    # single column
    result = df.drop_duplicates("C")
    expected = df[:2]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("C", keep="last")
    expected = df.loc[[3, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("C", keep=False)
    expected = df.loc[[]]  # empty df
    tm.assert_frame_equal(result, expected)
    assert len(result) == 0

    # multi column
    result = df.drop_duplicates(["C", "B"])
    expected = df.loc[[0, 1, 2, 4]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates(["C", "B"], keep="last")
    expected = df.loc[[1, 3, 6, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates(["C", "B"], keep=False)
    expected = df.loc[[1]]
    tm.assert_frame_equal(result, expected)


def test_drop_duplicates_NA_for_take_all():
    # none
    df = DataFrame(
        {
            "A": [None, None, "foo", "bar", "foo", "baz", "bar", "qux"],
            "C": [1.0, np.nan, np.nan, np.nan, 1.0, 2.0, 3, 1.0],
        }
    )

    # single column
    result = df.drop_duplicates("A")
    expected = df.iloc[[0, 2, 3, 5, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("A", keep="last")
    expected = df.iloc[[1, 4, 5, 6, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("A", keep=False)
    expected = df.iloc[[5, 7]]
    tm.assert_frame_equal(result, expected)

    # nan

    # single column
    result = df.drop_duplicates("C")
    expected = df.iloc[[0, 1, 5, 6]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("C", keep="last")
    expected = df.iloc[[3, 5, 6, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("C", keep=False)
    expected = df.iloc[[5, 6]]
    tm.assert_frame_equal(result, expected)


def test_drop_duplicates_inplace():
    orig = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": range(8),
        }
    )
    # single column
    df = orig.copy()
    return_value = df.drop_duplicates("A", inplace=True)
    expected = orig[:2]
    result = df
    tm.assert_frame_equal(result, expected)
    assert return_value is None

    df = orig.copy()
    return_value = df.drop_duplicates("A", keep="last", inplace=True)
    expected = orig.loc[[6, 7]]
    result = df
    tm.assert_frame_equal(result, expected)
    assert return_value is None

    df = orig.copy()
    return_value = df.drop_duplicates("A", keep=False, inplace=True)
    expected = orig.loc[[]]
    result = df
    tm.assert_frame_equal(result, expected)
    assert len(df) == 0
    assert return_value is None

    # multi column
    df = orig.copy()
    return_value = df.drop_duplicates(["A", "B"], inplace=True)
    expected = orig.loc[[0, 1, 2, 3]]
    result = df
    tm.assert_frame_equal(result, expected)
    assert return_value is None

    df = orig.copy()
    return_value = df.drop_duplicates(["A", "B"], keep="last", inplace=True)
    expected = orig.loc[[0, 5, 6, 7]]
    result = df
    tm.assert_frame_equal(result, expected)
    assert return_value is None

    df = orig.copy()
    return_value = df.drop_duplicates(["A", "B"], keep=False, inplace=True)
    expected = orig.loc[[0]]
    result = df
    tm.assert_frame_equal(result, expected)
    assert return_value is None

    # consider everything
    orig2 = orig.loc[:, ["A", "B", "C"]].copy()

    df2 = orig2.copy()
    return_value = df2.drop_duplicates(inplace=True)
    # in this case only
    expected = orig2.drop_duplicates(["A", "B"])
    result = df2
    tm.assert_frame_equal(result, expected)
    assert return_value is None

    df2 = orig2.copy()
    return_value = df2.drop_duplicates(keep="last", inplace=True)
    expected = orig2.drop_duplicates(["A", "B"], keep="last")
    result = df2
    tm.assert_frame_equal(result, expected)
    assert return_value is None

    df2 = orig2.copy()
    return_value = df2.drop_duplicates(keep=False, inplace=True)
    expected = orig2.drop_duplicates(["A", "B"], keep=False)
    result = df2
    tm.assert_frame_equal(result, expected)
    assert return_value is None


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize(
    "origin_dict, output_dict, ignore_index, output_index",
    [
        ({"A": [2, 2, 3]}, {"A": [2, 3]}, True, [0, 1]),
        ({"A": [2, 2, 3]}, {"A": [2, 3]}, False, [0, 2]),
        ({"A": [2, 2, 3], "B": [2, 2, 4]}, {"A": [2, 3], "B": [2, 4]}, True, [0, 1]),
        ({"A": [2, 2, 3], "B": [2, 2, 4]}, {"A": [2, 3], "B": [2, 4]}, False, [0, 2]),
    ],
)
def test_drop_duplicates_ignore_index(
    inplace, origin_dict, output_dict, ignore_index, output_index
):
    # GH 30114
    df = DataFrame(origin_dict)
    expected = DataFrame(output_dict, index=output_index)

    if inplace:
        result_df = df.copy()
        result_df.drop_duplicates(ignore_index=ignore_index, inplace=inplace)
    else:
        result_df = df.drop_duplicates(ignore_index=ignore_index, inplace=inplace)

    tm.assert_frame_equal(result_df, expected)
    tm.assert_frame_equal(df, DataFrame(origin_dict))


def test_drop_duplicates_null_in_object_column(nulls_fixture):
    # https://github.com/pandas-dev/pandas/issues/32992
    df = DataFrame([[1, nulls_fixture], [2, "a"]], dtype=object)
    result = df.drop_duplicates()
    tm.assert_frame_equal(result, df)


def test_drop_duplicates_series_vs_dataframe(keep):
    # GH#14192
    df = DataFrame(
        {
            "a": [1, 1, 1, "one", "one"],
            "b": [2, 2, np.nan, np.nan, np.nan],
            "c": [3, 3, np.nan, np.nan, "three"],
            "d": [1, 2, 3, 4, 4],
            "e": [
                datetime(2015, 1, 1),
                datetime(2015, 1, 1),
                datetime(2015, 2, 1),
                NaT,
                NaT,
            ],
        }
    )
    for column in df.columns:
        dropped_frame = df[[column]].drop_duplicates(keep=keep)
        dropped_series = df[column].drop_duplicates(keep=keep)
        tm.assert_frame_equal(dropped_frame, dropped_series.to_frame())


@pytest.mark.parametrize("arg", [[1], 1, "True", [], 0])
def test_drop_duplicates_non_boolean_ignore_index(arg):
    # GH#38274
    df = DataFrame({"a": [1, 2, 1, 3]})
    msg = '^For argument "ignore_index" expected type bool, received type .*.$'
    with pytest.raises(ValueError, match=msg):
        df.drop_duplicates(ignore_index=arg)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\indexing\test_getitem.py ===
import re

import numpy as np
import pytest

from pandas import (
    Categorical,
    CategoricalDtype,
    CategoricalIndex,
    DataFrame,
    DateOffset,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    concat,
    date_range,
    get_dummies,
    period_range,
)
import pandas._testing as tm
from pandas.core.arrays import SparseArray


class TestGetitem:
    def test_getitem_unused_level_raises(self):
        # GH#20410
        mi = MultiIndex(
            levels=[["a_lot", "onlyone", "notevenone"], [1970, ""]],
            codes=[[1, 0], [1, 0]],
        )
        df = DataFrame(-1, index=range(3), columns=mi)

        with pytest.raises(KeyError, match="notevenone"):
            df["notevenone"]

    def test_getitem_periodindex(self):
        rng = period_range("1/1/2000", periods=5)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)), columns=rng)

        ts = df[rng[0]]
        tm.assert_series_equal(ts, df.iloc[:, 0])

        ts = df["1/1/2000"]
        tm.assert_series_equal(ts, df.iloc[:, 0])

    def test_getitem_list_of_labels_categoricalindex_cols(self):
        # GH#16115
        cats = Categorical([Timestamp("12-31-1999"), Timestamp("12-31-2000")])

        expected = DataFrame([[1, 0], [0, 1]], dtype="bool", index=[0, 1], columns=cats)
        dummies = get_dummies(cats)
        result = dummies[list(dummies.columns)]
        tm.assert_frame_equal(result, expected)

    def test_getitem_sparse_column_return_type_and_dtype(self):
        # https://github.com/pandas-dev/pandas/issues/23559
        data = SparseArray([0, 1])
        df = DataFrame({"A": data})
        expected = Series(data, name="A")
        result = df["A"]
        tm.assert_series_equal(result, expected)

        # Also check iloc and loc while we're here
        result = df.iloc[:, 0]
        tm.assert_series_equal(result, expected)

        result = df.loc[:, "A"]
        tm.assert_series_equal(result, expected)

    def test_getitem_string_columns(self):
        # GH#46185
        df = DataFrame([[1, 2]], columns=Index(["A", "B"], dtype="string"))
        result = df.A
        expected = df["A"]
        tm.assert_series_equal(result, expected)


class TestGetitemListLike:
    def test_getitem_list_missing_key(self):
        # GH#13822, incorrect error string with non-unique columns when missing
        # column is accessed
        df = DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]})
        df.columns = ["x", "x", "z"]

        # Check that we get the correct value in the KeyError
        with pytest.raises(KeyError, match=r"\['y'\] not in index"):
            df[["x", "y", "z"]]

    def test_getitem_list_duplicates(self):
        # GH#1943
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)), columns=list("AABC")
        )
        df.columns.name = "foo"

        result = df[["B", "C"]]
        assert result.columns.name == "foo"

        expected = df.iloc[:, 2:]
        tm.assert_frame_equal(result, expected)

    def test_getitem_dupe_cols(self):
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "a", "b"])
        msg = "\"None of [Index(['baf'], dtype="
        with pytest.raises(KeyError, match=re.escape(msg)):
            df[["baf"]]

    @pytest.mark.parametrize(
        "idx_type",
        [
            list,
            iter,
            Index,
            set,
            lambda keys: dict(zip(keys, range(len(keys)))),
            lambda keys: dict(zip(keys, range(len(keys)))).keys(),
        ],
        ids=["list", "iter", "Index", "set", "dict", "dict_keys"],
    )
    @pytest.mark.parametrize("levels", [1, 2])
    def test_getitem_listlike(self, idx_type, levels, float_frame):
        # GH#21294

        if levels == 1:
            frame, missing = float_frame, "food"
        else:
            # MultiIndex columns
            frame = DataFrame(
                np.random.default_rng(2).standard_normal((8, 3)),
                columns=Index(
                    [("foo", "bar"), ("baz", "qux"), ("peek", "aboo")],
                    name=("sth", "sth2"),
                ),
            )
            missing = ("good", "food")

        keys = [frame.columns[1], frame.columns[0]]
        idx = idx_type(keys)
        idx_check = list(idx_type(keys))

        if isinstance(idx, (set, dict)):
            with pytest.raises(TypeError, match="as an indexer is not supported"):
                frame[idx]

            return
        else:
            result = frame[idx]

        expected = frame.loc[:, idx_check]
        expected.columns.names = frame.columns.names

        tm.assert_frame_equal(result, expected)

        idx = idx_type(keys + [missing])
        with pytest.raises(KeyError, match="not in index"):
            frame[idx]

    def test_getitem_iloc_generator(self):
        # GH#39614
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        indexer = (x for x in [1, 2])
        result = df.iloc[indexer]
        expected = DataFrame({"a": [2, 3], "b": [5, 6]}, index=[1, 2])
        tm.assert_frame_equal(result, expected)

    def test_getitem_iloc_two_dimensional_generator(self):
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        indexer = (x for x in [1, 2])
        result = df.iloc[indexer, 1]
        expected = Series([5, 6], name="b", index=[1, 2])
        tm.assert_series_equal(result, expected)

    def test_getitem_iloc_dateoffset_days(self):
        # GH 46671
        df = DataFrame(
            list(range(10)),
            index=date_range("01-01-2022", periods=10, freq=DateOffset(days=1)),
        )
        result = df.loc["2022-01-01":"2022-01-03"]
        expected = DataFrame(
            [0, 1, 2],
            index=DatetimeIndex(
                ["2022-01-01", "2022-01-02", "2022-01-03"],
                dtype="datetime64[ns]",
                freq=DateOffset(days=1),
            ),
        )
        tm.assert_frame_equal(result, expected)

        df = DataFrame(
            list(range(10)),
            index=date_range(
                "01-01-2022", periods=10, freq=DateOffset(days=1, hours=2)
            ),
        )
        result = df.loc["2022-01-01":"2022-01-03"]
        expected = DataFrame(
            [0, 1, 2],
            index=DatetimeIndex(
                ["2022-01-01 00:00:00", "2022-01-02 02:00:00", "2022-01-03 04:00:00"],
                dtype="datetime64[ns]",
                freq=DateOffset(days=1, hours=2),
            ),
        )
        tm.assert_frame_equal(result, expected)

        df = DataFrame(
            list(range(10)),
            index=date_range("01-01-2022", periods=10, freq=DateOffset(minutes=3)),
        )
        result = df.loc["2022-01-01":"2022-01-03"]
        tm.assert_frame_equal(result, df)


class TestGetitemCallable:
    def test_getitem_callable(self, float_frame):
        # GH#12533
        result = float_frame[lambda x: "A"]
        expected = float_frame.loc[:, "A"]
        tm.assert_series_equal(result, expected)

        result = float_frame[lambda x: ["A", "B"]]
        expected = float_frame.loc[:, ["A", "B"]]
        tm.assert_frame_equal(result, float_frame.loc[:, ["A", "B"]])

        df = float_frame[:3]
        result = df[lambda x: [True, False, True]]
        expected = float_frame.iloc[[0, 2], :]
        tm.assert_frame_equal(result, expected)

    def test_loc_multiindex_columns_one_level(self):
        # GH#29749
        df = DataFrame([[1, 2]], columns=[["a", "b"]])
        expected = DataFrame([1], columns=[["a"]])

        result = df["a"]
        tm.assert_frame_equal(result, expected)

        result = df.loc[:, "a"]
        tm.assert_frame_equal(result, expected)


class TestGetitemBooleanMask:
    def test_getitem_bool_mask_categorical_index(self):
        df3 = DataFrame(
            {
                "A": np.arange(6, dtype="int64"),
            },
            index=CategoricalIndex(
                [1, 1, 2, 1, 3, 2],
                dtype=CategoricalDtype([3, 2, 1], ordered=True),
                name="B",
            ),
        )
        df4 = DataFrame(
            {
                "A": np.arange(6, dtype="int64"),
            },
            index=CategoricalIndex(
                [1, 1, 2, 1, 3, 2],
                dtype=CategoricalDtype([3, 2, 1], ordered=False),
                name="B",
            ),
        )

        result = df3[df3.index == "a"]
        expected = df3.iloc[[]]
        tm.assert_frame_equal(result, expected)

        result = df4[df4.index == "a"]
        expected = df4.iloc[[]]
        tm.assert_frame_equal(result, expected)

        result = df3[df3.index == 1]
        expected = df3.iloc[[0, 1, 3]]
        tm.assert_frame_equal(result, expected)

        result = df4[df4.index == 1]
        expected = df4.iloc[[0, 1, 3]]
        tm.assert_frame_equal(result, expected)

        # since we have an ordered categorical

        # CategoricalIndex([1, 1, 2, 1, 3, 2],
        #         categories=[3, 2, 1],
        #         ordered=True,
        #         name='B')
        result = df3[df3.index < 2]
        expected = df3.iloc[[4]]
        tm.assert_frame_equal(result, expected)

        result = df3[df3.index > 1]
        expected = df3.iloc[[]]
        tm.assert_frame_equal(result, expected)

        # unordered
        # cannot be compared

        # CategoricalIndex([1, 1, 2, 1, 3, 2],
        #         categories=[3, 2, 1],
        #         ordered=False,
        #         name='B')
        msg = "Unordered Categoricals can only compare equality or not"
        with pytest.raises(TypeError, match=msg):
            df4[df4.index < 2]
        with pytest.raises(TypeError, match=msg):
            df4[df4.index > 1]

    @pytest.mark.parametrize(
        "data1,data2,expected_data",
        (
            (
                [[1, 2], [3, 4]],
                [[0.5, 6], [7, 8]],
                [[np.nan, 3.0], [np.nan, 4.0], [np.nan, 7.0], [6.0, 8.0]],
            ),
            (
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
                [[np.nan, 3.0], [np.nan, 4.0], [5, 7], [6, 8]],
            ),
        ),
    )
    def test_getitem_bool_mask_duplicate_columns_mixed_dtypes(
        self,
        data1,
        data2,
        expected_data,
    ):
        # GH#31954

        df1 = DataFrame(np.array(data1))
        df2 = DataFrame(np.array(data2))
        df = concat([df1, df2], axis=1)

        result = df[df > 2]

        exdict = {i: np.array(col) for i, col in enumerate(expected_data)}
        expected = DataFrame(exdict).rename(columns={2: 0, 3: 1})
        tm.assert_frame_equal(result, expected)

    @pytest.fixture
    def df_dup_cols(self):
        dups = ["A", "A", "C", "D"]
        df = DataFrame(np.arange(12).reshape(3, 4), columns=dups, dtype="float64")
        return df

    def test_getitem_boolean_frame_unaligned_with_duplicate_columns(self, df_dup_cols):
        # `df.A > 6` is a DataFrame with a different shape from df

        # boolean with the duplicate raises
        df = df_dup_cols
        msg = "cannot reindex on an axis with duplicate labels"
        with pytest.raises(ValueError, match=msg):
            df[df.A > 6]

    def test_getitem_boolean_series_with_duplicate_columns(self, df_dup_cols):
        # boolean indexing
        # GH#4879
        df = DataFrame(
            np.arange(12).reshape(3, 4), columns=["A", "B", "C", "D"], dtype="float64"
        )
        expected = df[df.C > 6]
        expected.columns = df_dup_cols.columns

        df = df_dup_cols
        result = df[df.C > 6]

        tm.assert_frame_equal(result, expected)

    def test_getitem_boolean_frame_with_duplicate_columns(self, df_dup_cols):
        # where
        df = DataFrame(
            np.arange(12).reshape(3, 4), columns=["A", "B", "C", "D"], dtype="float64"
        )
        # `df > 6` is a DataFrame with the same shape+alignment as df
        expected = df[df > 6]
        expected.columns = df_dup_cols.columns

        df = df_dup_cols
        result = df[df > 6]

        tm.assert_frame_equal(result, expected)

    def test_getitem_empty_frame_with_boolean(self):
        # Test for issue GH#11859

        df = DataFrame()
        df2 = df[df > 0]
        tm.assert_frame_equal(df, df2)

    def test_getitem_returns_view_when_column_is_unique_in_df(
        self, using_copy_on_write, warn_copy_on_write
    ):
        # GH#45316
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "a", "b"])
        df_orig = df.copy()
        view = df["b"]
        with tm.assert_cow_warning(warn_copy_on_write):
            view.loc[:] = 100
        if using_copy_on_write:
            expected = df_orig
        else:
            expected = DataFrame([[1, 2, 100], [4, 5, 100]], columns=["a", "a", "b"])
        tm.assert_frame_equal(df, expected)

    def test_getitem_frozenset_unique_in_column(self):
        # GH#41062
        df = DataFrame([[1, 2, 3, 4]], columns=[frozenset(["KEY"]), "B", "C", "C"])
        result = df[frozenset(["KEY"])]
        expected = Series([1], name=frozenset(["KEY"]))
        tm.assert_series_equal(result, expected)


class TestGetitemSlice:
    def test_getitem_slice_float64(self, frame_or_series):
        values = np.arange(10.0, 50.0, 2)
        index = Index(values)

        start, end = values[[5, 15]]

        data = np.random.default_rng(2).standard_normal((20, 3))
        if frame_or_series is not DataFrame:
            data = data[:, 0]

        obj = frame_or_series(data, index=index)

        result = obj[start:end]
        expected = obj.iloc[5:16]
        tm.assert_equal(result, expected)

        result = obj.loc[start:end]
        tm.assert_equal(result, expected)

    def test_getitem_datetime_slice(self):
        # GH#43223
        df = DataFrame(
            {"a": 0},
            index=DatetimeIndex(
                [
                    "11.01.2011 22:00",
                    "11.01.2011 23:00",
                    "12.01.2011 00:00",
                    "2011-01-13 00:00",
                ]
            ),
        )
        with pytest.raises(
            KeyError, match="Value based partial slicing on non-monotonic"
        ):
            df["2011-01-01":"2011-11-01"]

    def test_getitem_slice_same_dim_only_one_axis(self):
        # GH#54622
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 8)))
        result = df.iloc[(slice(None, None, 2),)]
        assert result.shape == (5, 8)
        expected = df.iloc[slice(None, None, 2), slice(None)]
        tm.assert_frame_equal(result, expected)


class TestGetitemDeprecatedIndexers:
    @pytest.mark.parametrize("key", [{"a", "b"}, {"a": "a"}])
    def test_getitem_dict_and_set_deprecated(self, key):
        # GH#42825 enforced in 2.0
        df = DataFrame(
            [[1, 2], [3, 4]], columns=MultiIndex.from_tuples([("a", 1), ("b", 2)])
        )
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            df[key]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_cov_corr.py ===
import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    date_range,
    isna,
)
import pandas._testing as tm


class TestDataFrameCov:
    def test_cov(self, float_frame, float_string_frame):
        # min_periods no NAs (corner case)
        expected = float_frame.cov()
        result = float_frame.cov(min_periods=len(float_frame))

        tm.assert_frame_equal(expected, result)

        result = float_frame.cov(min_periods=len(float_frame) + 1)
        assert isna(result.values).all()

        # with NAs
        frame = float_frame.copy()
        frame.iloc[:5, frame.columns.get_loc("A")] = np.nan
        frame.iloc[5:10, frame.columns.get_loc("B")] = np.nan
        result = frame.cov(min_periods=len(frame) - 8)
        expected = frame.cov()
        expected.loc["A", "B"] = np.nan
        expected.loc["B", "A"] = np.nan
        tm.assert_frame_equal(result, expected)

        # regular
        result = frame.cov()
        expected = frame["A"].cov(frame["C"])
        tm.assert_almost_equal(result["A"]["C"], expected)

        # fails on non-numeric types
        with pytest.raises(ValueError, match="could not convert string to float"):
            float_string_frame.cov()
        result = float_string_frame.cov(numeric_only=True)
        expected = float_string_frame.loc[:, ["A", "B", "C", "D"]].cov()
        tm.assert_frame_equal(result, expected)

        # Single column frame
        df = DataFrame(np.linspace(0.0, 1.0, 10))
        result = df.cov()
        expected = DataFrame(
            np.cov(df.values.T).reshape((1, 1)), index=df.columns, columns=df.columns
        )
        tm.assert_frame_equal(result, expected)
        df.loc[0] = np.nan
        result = df.cov()
        expected = DataFrame(
            np.cov(df.values[1:].T).reshape((1, 1)),
            index=df.columns,
            columns=df.columns,
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("test_ddof", [None, 0, 1, 2, 3])
    def test_cov_ddof(self, test_ddof):
        # GH#34611
        np_array1 = np.random.default_rng(2).random(10)
        np_array2 = np.random.default_rng(2).random(10)
        df = DataFrame({0: np_array1, 1: np_array2})
        result = df.cov(ddof=test_ddof)
        expected_np = np.cov(np_array1, np_array2, ddof=test_ddof)
        expected = DataFrame(expected_np)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "other_column", [pd.array([1, 2, 3]), np.array([1.0, 2.0, 3.0])]
    )
    def test_cov_nullable_integer(self, other_column):
        # https://github.com/pandas-dev/pandas/issues/33803
        data = DataFrame({"a": pd.array([1, 2, None]), "b": other_column})
        result = data.cov()
        arr = np.array([[0.5, 0.5], [0.5, 1.0]])
        expected = DataFrame(arr, columns=["a", "b"], index=["a", "b"])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("numeric_only", [True, False])
    def test_cov_numeric_only(self, numeric_only):
        # when dtypes of pandas series are different
        # then ndarray will have dtype=object,
        # so it need to be properly handled
        df = DataFrame({"a": [1, 0], "c": ["x", "y"]})
        expected = DataFrame(0.5, index=["a"], columns=["a"])
        if numeric_only:
            result = df.cov(numeric_only=numeric_only)
            tm.assert_frame_equal(result, expected)
        else:
            with pytest.raises(ValueError, match="could not convert string to float"):
                df.cov(numeric_only=numeric_only)


class TestDataFrameCorr:
    # DataFrame.corr(), as opposed to DataFrame.corrwith

    @pytest.mark.parametrize("method", ["pearson", "kendall", "spearman"])
    def test_corr_scipy_method(self, float_frame, method):
        pytest.importorskip("scipy")
        float_frame.loc[float_frame.index[:5], "A"] = np.nan
        float_frame.loc[float_frame.index[5:10], "B"] = np.nan
        float_frame.loc[float_frame.index[:10], "A"] = float_frame["A"][10:20].copy()

        correls = float_frame.corr(method=method)
        expected = float_frame["A"].corr(float_frame["C"], method=method)
        tm.assert_almost_equal(correls["A"]["C"], expected)

    # ---------------------------------------------------------------------

    def test_corr_non_numeric(self, float_string_frame):
        with pytest.raises(ValueError, match="could not convert string to float"):
            float_string_frame.corr()
        result = float_string_frame.corr(numeric_only=True)
        expected = float_string_frame.loc[:, ["A", "B", "C", "D"]].corr()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("meth", ["pearson", "kendall", "spearman"])
    def test_corr_nooverlap(self, meth):
        # nothing in common
        pytest.importorskip("scipy")
        df = DataFrame(
            {
                "A": [1, 1.5, 1, np.nan, np.nan, np.nan],
                "B": [np.nan, np.nan, np.nan, 1, 1.5, 1],
                "C": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        rs = df.corr(meth)
        assert isna(rs.loc["A", "B"])
        assert isna(rs.loc["B", "A"])
        assert rs.loc["A", "A"] == 1
        assert rs.loc["B", "B"] == 1
        assert isna(rs.loc["C", "C"])

    @pytest.mark.parametrize("meth", ["pearson", "spearman"])
    def test_corr_constant(self, meth):
        # constant --> all NA
        df = DataFrame(
            {
                "A": [1, 1, 1, np.nan, np.nan, np.nan],
                "B": [np.nan, np.nan, np.nan, 1, 1, 1],
            }
        )
        rs = df.corr(meth)
        assert isna(rs.values).all()

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize("meth", ["pearson", "kendall", "spearman"])
    def test_corr_int_and_boolean(self, meth):
        # when dtypes of pandas series are different
        # then ndarray will have dtype=object,
        # so it need to be properly handled
        pytest.importorskip("scipy")
        df = DataFrame({"a": [True, False], "b": [1, 0]})

        expected = DataFrame(np.ones((2, 2)), index=["a", "b"], columns=["a", "b"])
        result = df.corr(meth)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("method", ["cov", "corr"])
    def test_corr_cov_independent_index_column(self, method):
        # GH#14617
        df = DataFrame(
            np.random.default_rng(2).standard_normal(4 * 10).reshape(10, 4),
            columns=list("abcd"),
        )
        result = getattr(df, method)()
        assert result.index is not result.columns
        assert result.index.equals(result.columns)

    def test_corr_invalid_method(self):
        # GH#22298
        df = DataFrame(np.random.default_rng(2).normal(size=(10, 2)))
        msg = "method must be either 'pearson', 'spearman', 'kendall', or a callable, "
        with pytest.raises(ValueError, match=msg):
            df.corr(method="____")

    def test_corr_int(self):
        # dtypes other than float64 GH#1761
        df = DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]})

        df.cov()
        df.corr()

    @pytest.mark.parametrize(
        "nullable_column", [pd.array([1, 2, 3]), pd.array([1, 2, None])]
    )
    @pytest.mark.parametrize(
        "other_column",
        [pd.array([1, 2, 3]), np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, np.nan])],
    )
    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_corr_nullable_integer(self, nullable_column, other_column, method):
        # https://github.com/pandas-dev/pandas/issues/33803
        pytest.importorskip("scipy")
        data = DataFrame({"a": nullable_column, "b": other_column})
        result = data.corr(method=method)
        expected = DataFrame(np.ones((2, 2)), columns=["a", "b"], index=["a", "b"])
        tm.assert_frame_equal(result, expected)

    def test_corr_item_cache(self, using_copy_on_write, warn_copy_on_write):
        # Check that corr does not lead to incorrect entries in item_cache

        df = DataFrame({"A": range(10)})
        df["B"] = range(10)[::-1]

        ser = df["A"]  # populate item_cache
        assert len(df._mgr.arrays) == 2  # i.e. 2 blocks

        _ = df.corr(numeric_only=True)

        if using_copy_on_write:
            ser.iloc[0] = 99
            assert df.loc[0, "A"] == 0
        else:
            # Check that the corr didn't break link between ser and df
            ser.values[0] = 99
            assert df.loc[0, "A"] == 99
            if not warn_copy_on_write:
                assert df["A"] is ser
            assert df.values[0, 0] == 99

    @pytest.mark.parametrize("length", [2, 20, 200, 2000])
    def test_corr_for_constant_columns(self, length):
        # GH: 37448
        df = DataFrame(length * [[0.4, 0.1]], columns=["A", "B"])
        result = df.corr()
        expected = DataFrame(
            {"A": [np.nan, np.nan], "B": [np.nan, np.nan]}, index=["A", "B"]
        )
        tm.assert_frame_equal(result, expected)

    def test_calc_corr_small_numbers(self):
        # GH: 37452
        df = DataFrame(
            {"A": [1.0e-20, 2.0e-20, 3.0e-20], "B": [1.0e-20, 2.0e-20, 3.0e-20]}
        )
        result = df.corr()
        expected = DataFrame({"A": [1.0, 1.0], "B": [1.0, 1.0]}, index=["A", "B"])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_corr_min_periods_greater_than_length(self, method):
        pytest.importorskip("scipy")
        df = DataFrame({"A": [1, 2], "B": [1, 2]})
        result = df.corr(method=method, min_periods=3)
        expected = DataFrame(
            {"A": [np.nan, np.nan], "B": [np.nan, np.nan]}, index=["A", "B"]
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("meth", ["pearson", "kendall", "spearman"])
    @pytest.mark.parametrize("numeric_only", [True, False])
    def test_corr_numeric_only(self, meth, numeric_only):
        # when dtypes of pandas series are different
        # then ndarray will have dtype=object,
        # so it need to be properly handled
        pytest.importorskip("scipy")
        df = DataFrame({"a": [1, 0], "b": [1, 0], "c": ["x", "y"]})
        expected = DataFrame(np.ones((2, 2)), index=["a", "b"], columns=["a", "b"])
        if numeric_only:
            result = df.corr(meth, numeric_only=numeric_only)
            tm.assert_frame_equal(result, expected)
        else:
            with pytest.raises(ValueError, match="could not convert string to float"):
                df.corr(meth, numeric_only=numeric_only)


class TestDataFrameCorrWith:
    @pytest.mark.parametrize(
        "dtype",
        [
            "float64",
            "Float64",
            pytest.param("float64[pyarrow]", marks=td.skip_if_no("pyarrow")),
        ],
    )
    def test_corrwith(self, datetime_frame, dtype):
        datetime_frame = datetime_frame.astype(dtype)

        a = datetime_frame
        noise = Series(np.random.default_rng(2).standard_normal(len(a)), index=a.index)

        b = datetime_frame.add(noise, axis=0)

        # make sure order does not matter
        b = b.reindex(columns=b.columns[::-1], index=b.index[::-1][10:])
        del b["B"]

        colcorr = a.corrwith(b, axis=0)
        tm.assert_almost_equal(colcorr["A"], a["A"].corr(b["A"]))

        rowcorr = a.corrwith(b, axis=1)
        tm.assert_series_equal(rowcorr, a.T.corrwith(b.T, axis=0))

        dropped = a.corrwith(b, axis=0, drop=True)
        tm.assert_almost_equal(dropped["A"], a["A"].corr(b["A"]))
        assert "B" not in dropped

        dropped = a.corrwith(b, axis=1, drop=True)
        assert a.index[-1] not in dropped.index

        # non time-series data
        index = ["a", "b", "c", "d", "e"]
        columns = ["one", "two", "three", "four"]
        df1 = DataFrame(
            np.random.default_rng(2).standard_normal((5, 4)),
            index=index,
            columns=columns,
        )
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=index[:4],
            columns=columns,
        )
        correls = df1.corrwith(df2, axis=1)
        for row in index[:4]:
            tm.assert_almost_equal(correls[row], df1.loc[row].corr(df2.loc[row]))

    def test_corrwith_with_objects(self, using_infer_string):
        df1 = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        df2 = df1.copy()
        cols = ["A", "B", "C", "D"]

        df1["obj"] = "foo"
        df2["obj"] = "bar"

        if using_infer_string:
            msg = "Cannot perform reduction 'mean' with string dtype"
            with pytest.raises(TypeError, match=msg):
                df1.corrwith(df2)
        else:
            with pytest.raises(TypeError, match="Could not convert"):
                df1.corrwith(df2)
        result = df1.corrwith(df2, numeric_only=True)
        expected = df1.loc[:, cols].corrwith(df2.loc[:, cols])
        tm.assert_series_equal(result, expected)

        with pytest.raises(TypeError, match="unsupported operand type"):
            df1.corrwith(df2, axis=1)
        result = df1.corrwith(df2, axis=1, numeric_only=True)
        expected = df1.loc[:, cols].corrwith(df2.loc[:, cols], axis=1)
        tm.assert_series_equal(result, expected)

    def test_corrwith_series(self, datetime_frame):
        result = datetime_frame.corrwith(datetime_frame["A"])
        expected = datetime_frame.apply(datetime_frame["A"].corr)

        tm.assert_series_equal(result, expected)

    def test_corrwith_matches_corrcoef(self):
        df1 = DataFrame(np.arange(10000), columns=["a"])
        df2 = DataFrame(np.arange(10000) ** 2, columns=["a"])
        c1 = df1.corrwith(df2)["a"]
        c2 = np.corrcoef(df1["a"], df2["a"])[0][1]

        tm.assert_almost_equal(c1, c2)
        assert c1 < 1

    @pytest.mark.parametrize("numeric_only", [True, False])
    def test_corrwith_mixed_dtypes(self, numeric_only):
        # GH#18570
        df = DataFrame(
            {"a": [1, 4, 3, 2], "b": [4, 6, 7, 3], "c": ["a", "b", "c", "d"]}
        )
        s = Series([0, 6, 7, 3])
        if numeric_only:
            result = df.corrwith(s, numeric_only=numeric_only)
            corrs = [df["a"].corr(s), df["b"].corr(s)]
            expected = Series(data=corrs, index=["a", "b"])
            tm.assert_series_equal(result, expected)
        else:
            with pytest.raises(
                ValueError,
                match="could not convert string to float",
            ):
                df.corrwith(s, numeric_only=numeric_only)

    def test_corrwith_index_intersection(self):
        df1 = DataFrame(
            np.random.default_rng(2).random(size=(10, 2)), columns=["a", "b"]
        )
        df2 = DataFrame(
            np.random.default_rng(2).random(size=(10, 3)), columns=["a", "b", "c"]
        )

        result = df1.corrwith(df2, drop=True).index.sort_values()
        expected = df1.columns.intersection(df2.columns).sort_values()
        tm.assert_index_equal(result, expected)

    def test_corrwith_index_union(self):
        df1 = DataFrame(
            np.random.default_rng(2).random(size=(10, 2)), columns=["a", "b"]
        )
        df2 = DataFrame(
            np.random.default_rng(2).random(size=(10, 3)), columns=["a", "b", "c"]
        )

        result = df1.corrwith(df2, drop=False).index.sort_values()
        expected = df1.columns.union(df2.columns).sort_values()
        tm.assert_index_equal(result, expected)

    def test_corrwith_dup_cols(self):
        # GH#21925
        df1 = DataFrame(np.vstack([np.arange(10)] * 3).T)
        df2 = df1.copy()
        df2 = pd.concat((df2, df2[0]), axis=1)

        result = df1.corrwith(df2)
        expected = Series(np.ones(4), index=[0, 0, 1, 2])
        tm.assert_series_equal(result, expected)

    def test_corr_numerical_instabilities(self):
        # GH#45640
        df = DataFrame([[0.2, 0.4], [0.4, 0.2]])
        result = df.corr()
        expected = DataFrame({0: [1.0, -1.0], 1: [-1.0, 1.0]})
        tm.assert_frame_equal(result - 1, expected - 1, atol=1e-17)

    def test_corrwith_spearman(self):
        # GH#21925
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).random(size=(100, 3)))
        result = df.corrwith(df**2, method="spearman")
        expected = Series(np.ones(len(result)))
        tm.assert_series_equal(result, expected)

    def test_corrwith_kendall(self):
        # GH#21925
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).random(size=(100, 3)))
        result = df.corrwith(df**2, method="kendall")
        expected = Series(np.ones(len(result)))
        tm.assert_series_equal(result, expected)

    def test_corrwith_spearman_with_tied_data(self):
        # GH#48826
        pytest.importorskip("scipy")
        df1 = DataFrame(
            {
                "A": [1, np.nan, 7, 8],
                "B": [False, True, True, False],
                "C": [10, 4, 9, 3],
            }
        )
        df2 = df1[["B", "C"]]
        result = (df1 + 1).corrwith(df2.B, method="spearman")
        expected = Series([0.0, 1.0, 0.0], index=["A", "B", "C"])
        tm.assert_series_equal(result, expected)

        df_bool = DataFrame(
            {"A": [True, True, False, False], "B": [True, False, False, True]}
        )
        ser_bool = Series([True, True, False, True])
        result = df_bool.corrwith(ser_bool)
        expected = Series([0.57735, 0.57735], index=["A", "B"])
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\concat\test_index.py ===
from copy import deepcopy

import numpy as np
import pytest

from pandas.errors import PerformanceWarning

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    concat,
)
import pandas._testing as tm


class TestIndexConcat:
    def test_concat_ignore_index(self, sort):
        frame1 = DataFrame(
            {"test1": ["a", "b", "c"], "test2": [1, 2, 3], "test3": [4.5, 3.2, 1.2]}
        )
        frame2 = DataFrame({"test3": [5.2, 2.2, 4.3]})
        frame1.index = Index(["x", "y", "z"])
        frame2.index = Index(["x", "y", "q"])

        v1 = concat([frame1, frame2], axis=1, ignore_index=True, sort=sort)

        nan = np.nan
        expected = DataFrame(
            [
                [nan, nan, nan, 4.3],
                ["a", 1, 4.5, 5.2],
                ["b", 2, 3.2, 2.2],
                ["c", 3, 1.2, nan],
            ],
            index=Index(["q", "x", "y", "z"]),
        )
        if not sort:
            expected = expected.loc[["x", "y", "z", "q"]]

        tm.assert_frame_equal(v1, expected)

    @pytest.mark.parametrize(
        "name_in1,name_in2,name_in3,name_out",
        [
            ("idx", "idx", "idx", "idx"),
            ("idx", "idx", None, None),
            ("idx", None, None, None),
            ("idx1", "idx2", None, None),
            ("idx1", "idx1", "idx2", None),
            ("idx1", "idx2", "idx3", None),
            (None, None, None, None),
        ],
    )
    def test_concat_same_index_names(self, name_in1, name_in2, name_in3, name_out):
        # GH13475
        indices = [
            Index(["a", "b", "c"], name=name_in1),
            Index(["b", "c", "d"], name=name_in2),
            Index(["c", "d", "e"], name=name_in3),
        ]
        frames = [
            DataFrame({c: [0, 1, 2]}, index=i) for i, c in zip(indices, ["x", "y", "z"])
        ]
        result = concat(frames, axis=1)

        exp_ind = Index(["a", "b", "c", "d", "e"], name=name_out)
        expected = DataFrame(
            {
                "x": [0, 1, 2, np.nan, np.nan],
                "y": [np.nan, 0, 1, 2, np.nan],
                "z": [np.nan, np.nan, 0, 1, 2],
            },
            index=exp_ind,
        )

        tm.assert_frame_equal(result, expected)

    def test_concat_rename_index(self):
        a = DataFrame(
            np.random.default_rng(2).random((3, 3)),
            columns=list("ABC"),
            index=Index(list("abc"), name="index_a"),
        )
        b = DataFrame(
            np.random.default_rng(2).random((3, 3)),
            columns=list("ABC"),
            index=Index(list("abc"), name="index_b"),
        )

        result = concat([a, b], keys=["key0", "key1"], names=["lvl0", "lvl1"])

        exp = concat([a, b], keys=["key0", "key1"], names=["lvl0"])
        names = list(exp.index.names)
        names[1] = "lvl1"
        exp.index.set_names(names, inplace=True)

        tm.assert_frame_equal(result, exp)
        assert result.index.names == exp.index.names

    def test_concat_copy_index_series(self, axis, using_copy_on_write):
        # GH 29879
        ser = Series([1, 2])
        comb = concat([ser, ser], axis=axis, copy=True)
        if not using_copy_on_write or axis in [0, "index"]:
            assert comb.index is not ser.index
        else:
            assert comb.index is ser.index

    def test_concat_copy_index_frame(self, axis, using_copy_on_write):
        # GH 29879
        df = DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
        comb = concat([df, df], axis=axis, copy=True)
        if not using_copy_on_write:
            assert not comb.index.is_(df.index)
            assert not comb.columns.is_(df.columns)
        elif axis in [0, "index"]:
            assert not comb.index.is_(df.index)
            assert comb.columns.is_(df.columns)
        elif axis in [1, "columns"]:
            assert comb.index.is_(df.index)
            assert not comb.columns.is_(df.columns)

    def test_default_index(self):
        # is_series and ignore_index
        s1 = Series([1, 2, 3], name="x")
        s2 = Series([4, 5, 6], name="y")
        res = concat([s1, s2], axis=1, ignore_index=True)
        assert isinstance(res.columns, pd.RangeIndex)
        exp = DataFrame([[1, 4], [2, 5], [3, 6]])
        # use check_index_type=True to check the result have
        # RangeIndex (default index)
        tm.assert_frame_equal(res, exp, check_index_type=True, check_column_type=True)

        # is_series and all inputs have no names
        s1 = Series([1, 2, 3])
        s2 = Series([4, 5, 6])
        res = concat([s1, s2], axis=1, ignore_index=False)
        assert isinstance(res.columns, pd.RangeIndex)
        exp = DataFrame([[1, 4], [2, 5], [3, 6]])
        exp.columns = pd.RangeIndex(2)
        tm.assert_frame_equal(res, exp, check_index_type=True, check_column_type=True)

        # is_dataframe and ignore_index
        df1 = DataFrame({"A": [1, 2], "B": [5, 6]})
        df2 = DataFrame({"A": [3, 4], "B": [7, 8]})

        res = concat([df1, df2], axis=0, ignore_index=True)
        exp = DataFrame([[1, 5], [2, 6], [3, 7], [4, 8]], columns=["A", "B"])
        tm.assert_frame_equal(res, exp, check_index_type=True, check_column_type=True)

        res = concat([df1, df2], axis=1, ignore_index=True)
        exp = DataFrame([[1, 5, 3, 7], [2, 6, 4, 8]])
        tm.assert_frame_equal(res, exp, check_index_type=True, check_column_type=True)

    def test_dups_index(self):
        # GH 4771

        # single dtypes
        df = DataFrame(
            np.random.default_rng(2).integers(0, 10, size=40).reshape(10, 4),
            columns=["A", "A", "C", "C"],
        )

        result = concat([df, df], axis=1)
        tm.assert_frame_equal(result.iloc[:, :4], df)
        tm.assert_frame_equal(result.iloc[:, 4:], df)

        result = concat([df, df], axis=0)
        tm.assert_frame_equal(result.iloc[:10], df)
        tm.assert_frame_equal(result.iloc[10:], df)

        # multi dtypes
        df = concat(
            [
                DataFrame(
                    np.random.default_rng(2).standard_normal((10, 4)),
                    columns=["A", "A", "B", "B"],
                ),
                DataFrame(
                    np.random.default_rng(2).integers(0, 10, size=20).reshape(10, 2),
                    columns=["A", "C"],
                ),
            ],
            axis=1,
        )

        result = concat([df, df], axis=1)
        tm.assert_frame_equal(result.iloc[:, :6], df)
        tm.assert_frame_equal(result.iloc[:, 6:], df)

        result = concat([df, df], axis=0)
        tm.assert_frame_equal(result.iloc[:10], df)
        tm.assert_frame_equal(result.iloc[10:], df)

        # append
        result = df.iloc[0:8, :]._append(df.iloc[8:])
        tm.assert_frame_equal(result, df)

        result = df.iloc[0:8, :]._append(df.iloc[8:9])._append(df.iloc[9:10])
        tm.assert_frame_equal(result, df)

        expected = concat([df, df], axis=0)
        result = df._append(df)
        tm.assert_frame_equal(result, expected)


class TestMultiIndexConcat:
    def test_concat_multiindex_with_keys(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data
        index = frame.index
        result = concat([frame, frame], keys=[0, 1], names=["iteration"])

        assert result.index.names == ("iteration",) + index.names
        tm.assert_frame_equal(result.loc[0], frame)
        tm.assert_frame_equal(result.loc[1], frame)
        assert result.index.nlevels == 3

    def test_concat_multiindex_with_none_in_index_names(self):
        # GH 15787
        index = MultiIndex.from_product([[1], range(5)], names=["level1", None])
        df = DataFrame({"col": range(5)}, index=index, dtype=np.int32)

        result = concat([df, df], keys=[1, 2], names=["level2"])
        index = MultiIndex.from_product(
            [[1, 2], [1], range(5)], names=["level2", "level1", None]
        )
        expected = DataFrame({"col": list(range(5)) * 2}, index=index, dtype=np.int32)
        tm.assert_frame_equal(result, expected)

        result = concat([df, df[:2]], keys=[1, 2], names=["level2"])
        level2 = [1] * 5 + [2] * 2
        level1 = [1] * 7
        no_name = list(range(5)) + list(range(2))
        tuples = list(zip(level2, level1, no_name))
        index = MultiIndex.from_tuples(tuples, names=["level2", "level1", None])
        expected = DataFrame({"col": no_name}, index=index, dtype=np.int32)
        tm.assert_frame_equal(result, expected)

    def test_concat_multiindex_rangeindex(self):
        # GH13542
        # when multi-index levels are RangeIndex objects
        # there is a bug in concat with objects of len 1

        df = DataFrame(np.random.default_rng(2).standard_normal((9, 2)))
        df.index = MultiIndex(
            levels=[pd.RangeIndex(3), pd.RangeIndex(3)],
            codes=[np.repeat(np.arange(3), 3), np.tile(np.arange(3), 3)],
        )

        res = concat([df.iloc[[2, 3, 4], :], df.iloc[[5], :]])
        exp = df.iloc[[2, 3, 4, 5], :]
        tm.assert_frame_equal(res, exp)

    def test_concat_multiindex_dfs_with_deepcopy(self):
        # GH 9967
        example_multiindex1 = MultiIndex.from_product([["a"], ["b"]])
        example_dataframe1 = DataFrame([0], index=example_multiindex1)

        example_multiindex2 = MultiIndex.from_product([["a"], ["c"]])
        example_dataframe2 = DataFrame([1], index=example_multiindex2)

        example_dict = {"s1": example_dataframe1, "s2": example_dataframe2}
        expected_index = MultiIndex(
            levels=[["s1", "s2"], ["a"], ["b", "c"]],
            codes=[[0, 1], [0, 0], [0, 1]],
            names=["testname", None, None],
        )
        expected = DataFrame([[0], [1]], index=expected_index)
        result_copy = concat(deepcopy(example_dict), names=["testname"])
        tm.assert_frame_equal(result_copy, expected)
        result_no_copy = concat(example_dict, names=["testname"])
        tm.assert_frame_equal(result_no_copy, expected)

    @pytest.mark.parametrize(
        "mi1_list",
        [
            [["a"], range(2)],
            [["b"], np.arange(2.0, 4.0)],
            [["c"], ["A", "B"]],
            [["d"], pd.date_range(start="2017", end="2018", periods=2)],
        ],
    )
    @pytest.mark.parametrize(
        "mi2_list",
        [
            [["a"], range(2)],
            [["b"], np.arange(2.0, 4.0)],
            [["c"], ["A", "B"]],
            [["d"], pd.date_range(start="2017", end="2018", periods=2)],
        ],
    )
    def test_concat_with_various_multiindex_dtypes(
        self, mi1_list: list, mi2_list: list
    ):
        # GitHub #23478
        mi1 = MultiIndex.from_product(mi1_list)
        mi2 = MultiIndex.from_product(mi2_list)

        df1 = DataFrame(np.zeros((1, len(mi1))), columns=mi1)
        df2 = DataFrame(np.zeros((1, len(mi2))), columns=mi2)

        if mi1_list[0] == mi2_list[0]:
            expected_mi = MultiIndex(
                levels=[mi1_list[0], list(mi1_list[1])],
                codes=[[0, 0, 0, 0], [0, 1, 0, 1]],
            )
        else:
            expected_mi = MultiIndex(
                levels=[
                    mi1_list[0] + mi2_list[0],
                    list(mi1_list[1]) + list(mi2_list[1]),
                ],
                codes=[[0, 0, 1, 1], [0, 1, 2, 3]],
            )

        expected_df = DataFrame(np.zeros((1, len(expected_mi))), columns=expected_mi)

        with tm.assert_produces_warning(None):
            result_df = concat((df1, df2), axis=1)

        tm.assert_frame_equal(expected_df, result_df)

    def test_concat_multiindex_(self):
        # GitHub #44786
        df = DataFrame({"col": ["a", "b", "c"]}, index=["1", "2", "2"])
        df = concat([df], keys=["X"])

        iterables = [["X"], ["1", "2", "2"]]
        result_index = df.index
        expected_index = MultiIndex.from_product(iterables)

        tm.assert_index_equal(result_index, expected_index)

        result_df = df
        expected_df = DataFrame(
            {"col": ["a", "b", "c"]}, index=MultiIndex.from_product(iterables)
        )
        tm.assert_frame_equal(result_df, expected_df)

    def test_concat_with_key_not_unique(self):
        # GitHub #46519
        df1 = DataFrame({"name": [1]})
        df2 = DataFrame({"name": [2]})
        df3 = DataFrame({"name": [3]})
        df_a = concat([df1, df2, df3], keys=["x", "y", "x"])
        # the warning is caused by indexing unsorted multi-index
        with tm.assert_produces_warning(
            PerformanceWarning, match="indexing past lexsort depth"
        ):
            out_a = df_a.loc[("x", 0), :]

        df_b = DataFrame(
            {"name": [1, 2, 3]}, index=Index([("x", 0), ("y", 0), ("x", 0)])
        )
        with tm.assert_produces_warning(
            PerformanceWarning, match="indexing past lexsort depth"
        ):
            out_b = df_b.loc[("x", 0)]

        tm.assert_frame_equal(out_a, out_b)

        df1 = DataFrame({"name": ["a", "a", "b"]})
        df2 = DataFrame({"name": ["a", "b"]})
        df3 = DataFrame({"name": ["c", "d"]})
        df_a = concat([df1, df2, df3], keys=["x", "y", "x"])
        with tm.assert_produces_warning(
            PerformanceWarning, match="indexing past lexsort depth"
        ):
            out_a = df_a.loc[("x", 0), :]

        df_b = DataFrame(
            {
                "a": ["x", "x", "x", "y", "y", "x", "x"],
                "b": [0, 1, 2, 0, 1, 0, 1],
                "name": list("aababcd"),
            }
        ).set_index(["a", "b"])
        df_b.index.names = [None, None]
        with tm.assert_produces_warning(
            PerformanceWarning, match="indexing past lexsort depth"
        ):
            out_b = df_b.loc[("x", 0), :]

        tm.assert_frame_equal(out_a, out_b)

    def test_concat_with_duplicated_levels(self):
        # keyword levels should be unique
        df1 = DataFrame({"A": [1]}, index=["x"])
        df2 = DataFrame({"A": [1]}, index=["y"])
        msg = r"Level values not unique: \['x', 'y', 'y'\]"
        with pytest.raises(ValueError, match=msg):
            concat([df1, df2], keys=["x", "y"], levels=[["x", "y", "y"]])

    @pytest.mark.parametrize("levels", [[["x", "y"]], [["x", "y", "y"]]])
    def test_concat_with_levels_with_none_keys(self, levels):
        df1 = DataFrame({"A": [1]}, index=["x"])
        df2 = DataFrame({"A": [1]}, index=["y"])
        msg = "levels supported only when keys is not None"
        with pytest.raises(ValueError, match=msg):
            concat([df1, df2], levels=levels)

    def test_concat_range_index_result(self):
        # GH#47501
        df1 = DataFrame({"a": [1, 2]})
        df2 = DataFrame({"b": [1, 2]})

        result = concat([df1, df2], sort=True, axis=1)
        expected = DataFrame({"a": [1, 2], "b": [1, 2]})
        tm.assert_frame_equal(result, expected)
        expected_index = pd.RangeIndex(0, 2)
        tm.assert_index_equal(result.index, expected_index, exact=True)

    def test_concat_index_keep_dtype(self):
        # GH#47329
        df1 = DataFrame([[0, 1, 1]], columns=Index([1, 2, 3], dtype="object"))
        df2 = DataFrame([[0, 1]], columns=Index([1, 2], dtype="object"))
        result = concat([df1, df2], ignore_index=True, join="outer", sort=True)
        expected = DataFrame(
            [[0, 1, 1.0], [0, 1, np.nan]], columns=Index([1, 2, 3], dtype="object")
        )
        tm.assert_frame_equal(result, expected)

    def test_concat_index_keep_dtype_ea_numeric(self, any_numeric_ea_dtype):
        # GH#47329
        df1 = DataFrame(
            [[0, 1, 1]], columns=Index([1, 2, 3], dtype=any_numeric_ea_dtype)
        )
        df2 = DataFrame([[0, 1]], columns=Index([1, 2], dtype=any_numeric_ea_dtype))
        result = concat([df1, df2], ignore_index=True, join="outer", sort=True)
        expected = DataFrame(
            [[0, 1, 1.0], [0, 1, np.nan]],
            columns=Index([1, 2, 3], dtype=any_numeric_ea_dtype),
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["Int8", "Int16", "Int32"])
    def test_concat_index_find_common(self, dtype):
        # GH#47329
        df1 = DataFrame([[0, 1, 1]], columns=Index([1, 2, 3], dtype=dtype))
        df2 = DataFrame([[0, 1]], columns=Index([1, 2], dtype="Int32"))
        result = concat([df1, df2], ignore_index=True, join="outer", sort=True)
        expected = DataFrame(
            [[0, 1, 1.0], [0, 1, np.nan]], columns=Index([1, 2, 3], dtype="Int32")
        )
        tm.assert_frame_equal(result, expected)

    def test_concat_axis_1_sort_false_rangeindex(self, using_infer_string):
        # GH 46675
        s1 = Series(["a", "b", "c"])
        s2 = Series(["a", "b"])
        s3 = Series(["a", "b", "c", "d"])
        s4 = Series([], dtype=object if not using_infer_string else "str")
        result = concat(
            [s1, s2, s3, s4], sort=False, join="outer", ignore_index=False, axis=1
        )
        expected = DataFrame(
            [
                ["a"] * 3 + [np.nan],
                ["b"] * 3 + [np.nan],
                ["c", np.nan] * 2,
                [np.nan] * 2 + ["d"] + [np.nan],
            ],
            dtype=object if not using_infer_string else "str",
        )
        tm.assert_frame_equal(
            result, expected, check_index_type=True, check_column_type=True
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\ma\tests\test_subclassing.py ===
"""Tests suite for MaskedArray & subclassing.

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu

"""
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.ma.core import (
    MaskedArray,
    add,
    arange,
    array,
    asanyarray,
    asarray,
    divide,
    hypot,
    log,
    masked,
    masked_array,
    nomask,
)
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises

# from numpy.ma.core import (

def assert_startswith(a, b):
    # produces a better error message than assert_(a.startswith(b))
    assert_equal(a[:len(b)], b)

class SubArray(np.ndarray):
    # Defines a generic np.ndarray subclass, that stores some metadata
    # in the  dictionary `info`.
    def __new__(cls, arr, info={}):
        x = np.asanyarray(arr).view(cls)
        x.info = info.copy()
        return x

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        self.info = getattr(obj, 'info', {}).copy()

    def __add__(self, other):
        result = super().__add__(other)
        result.info['added'] = result.info.get('added', 0) + 1
        return result

    def __iadd__(self, other):
        result = super().__iadd__(other)
        result.info['iadded'] = result.info.get('iadded', 0) + 1
        return result


subarray = SubArray


class SubMaskedArray(MaskedArray):
    """Pure subclass of MaskedArray, keeping some info on subclass."""
    def __new__(cls, info=None, **kwargs):
        obj = super().__new__(cls, **kwargs)
        obj._optinfo['info'] = info
        return obj


class MSubArray(SubArray, MaskedArray):

    def __new__(cls, data, info={}, mask=nomask):
        subarr = SubArray(data, info)
        _data = MaskedArray.__new__(cls, data=subarr, mask=mask)
        _data.info = subarr.info
        return _data

    @property
    def _series(self):
        _view = self.view(MaskedArray)
        _view._sharedmask = False
        return _view


msubarray = MSubArray


# Also a subclass that overrides __str__, __repr__ and __setitem__, disallowing
# setting to non-class values (and thus np.ma.core.masked_print_option)
# and overrides __array_wrap__, updating the info dict, to check that this
# doesn't get destroyed by MaskedArray._update_from.  But this one also needs
# its own iterator...
class CSAIterator:
    """
    Flat iterator object that uses its own setter/getter
    (works around ndarray.flat not propagating subclass setters/getters
    see https://github.com/numpy/numpy/issues/4564)
    roughly following MaskedIterator
    """
    def __init__(self, a):
        self._original = a
        self._dataiter = a.view(np.ndarray).flat

    def __iter__(self):
        return self

    def __getitem__(self, indx):
        out = self._dataiter.__getitem__(indx)
        if not isinstance(out, np.ndarray):
            out = out.__array__()
        out = out.view(type(self._original))
        return out

    def __setitem__(self, index, value):
        self._dataiter[index] = self._original._validate_input(value)

    def __next__(self):
        return next(self._dataiter).__array__().view(type(self._original))


class ComplicatedSubArray(SubArray):

    def __str__(self):
        return f'myprefix {self.view(SubArray)} mypostfix'

    def __repr__(self):
        # Return a repr that does not start with 'name('
        return f'<{self.__class__.__name__} {self}>'

    def _validate_input(self, value):
        if not isinstance(value, ComplicatedSubArray):
            raise ValueError("Can only set to MySubArray values")
        return value

    def __setitem__(self, item, value):
        # validation ensures direct assignment with ndarray or
        # masked_print_option will fail
        super().__setitem__(item, self._validate_input(value))

    def __getitem__(self, item):
        # ensure getter returns our own class also for scalars
        value = super().__getitem__(item)
        if not isinstance(value, np.ndarray):  # scalar
            value = value.__array__().view(ComplicatedSubArray)
        return value

    @property
    def flat(self):
        return CSAIterator(self)

    @flat.setter
    def flat(self, value):
        y = self.ravel()
        y[:] = value

    def __array_wrap__(self, obj, context=None, return_scalar=False):
        obj = super().__array_wrap__(obj, context, return_scalar)
        if context is not None and context[0] is np.multiply:
            obj.info['multiplied'] = obj.info.get('multiplied', 0) + 1

        return obj


class WrappedArray(NDArrayOperatorsMixin):
    """
    Wrapping a MaskedArray rather than subclassing to test that
    ufunc deferrals are commutative.
    See: https://github.com/numpy/numpy/issues/15200)
    """
    __slots__ = ('_array', 'attrs')
    __array_priority__ = 20

    def __init__(self, array, **attrs):
        self._array = array
        self.attrs = attrs

    def __repr__(self):
        return f"{self.__class__.__name__}(\n{self._array}\n{self.attrs}\n)"

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self._array)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            inputs = [arg._array if isinstance(arg, self.__class__) else arg
                      for arg in inputs]
            return self.__class__(ufunc(*inputs, **kwargs), **self.attrs)
        else:
            return NotImplemented


class TestSubclassing:
    # Test suite for masked subclasses of ndarray.

    def setup_method(self):
        x = np.arange(5, dtype='float')
        mx = msubarray(x, mask=[0, 1, 0, 0, 0])
        self.data = (x, mx)

    def test_data_subclassing(self):
        # Tests whether the subclass is kept.
        x = np.arange(5)
        m = [0, 0, 1, 0, 0]
        xsub = SubArray(x)
        xmsub = masked_array(xsub, mask=m)
        assert_(isinstance(xmsub, MaskedArray))
        assert_equal(xmsub._data, xsub)
        assert_(isinstance(xmsub._data, SubArray))

    def test_maskedarray_subclassing(self):
        # Tests subclassing MaskedArray
        (x, mx) = self.data
        assert_(isinstance(mx._data, subarray))

    def test_masked_unary_operations(self):
        # Tests masked_unary_operation
        (x, mx) = self.data
        with np.errstate(divide='ignore'):
            assert_(isinstance(log(mx), msubarray))
            assert_equal(log(x), np.log(x))

    def test_masked_binary_operations(self):
        # Tests masked_binary_operation
        (x, mx) = self.data
        # Result should be a msubarray
        assert_(isinstance(add(mx, mx), msubarray))
        assert_(isinstance(add(mx, x), msubarray))
        # Result should work
        assert_equal(add(mx, x), mx + x)
        assert_(isinstance(add(mx, mx)._data, subarray))
        assert_(isinstance(add.outer(mx, mx), msubarray))
        assert_(isinstance(hypot(mx, mx), msubarray))
        assert_(isinstance(hypot(mx, x), msubarray))

    def test_masked_binary_operations2(self):
        # Tests domained_masked_binary_operation
        (x, mx) = self.data
        xmx = masked_array(mx.data.__array__(), mask=mx.mask)
        assert_(isinstance(divide(mx, mx), msubarray))
        assert_(isinstance(divide(mx, x), msubarray))
        assert_equal(divide(mx, mx), divide(xmx, xmx))

    def test_attributepropagation(self):
        x = array(arange(5), mask=[0] + [1] * 4)
        my = masked_array(subarray(x))
        ym = msubarray(x)
        #
        z = (my + 1)
        assert_(isinstance(z, MaskedArray))
        assert_(not isinstance(z, MSubArray))
        assert_(isinstance(z._data, SubArray))
        assert_equal(z._data.info, {})
        #
        z = (ym + 1)
        assert_(isinstance(z, MaskedArray))
        assert_(isinstance(z, MSubArray))
        assert_(isinstance(z._data, SubArray))
        assert_(z._data.info['added'] > 0)
        # Test that inplace methods from data get used (gh-4617)
        ym += 1
        assert_(isinstance(ym, MaskedArray))
        assert_(isinstance(ym, MSubArray))
        assert_(isinstance(ym._data, SubArray))
        assert_(ym._data.info['iadded'] > 0)
        #
        ym._set_mask([1, 0, 0, 0, 1])
        assert_equal(ym._mask, [1, 0, 0, 0, 1])
        ym._series._set_mask([0, 0, 0, 0, 1])
        assert_equal(ym._mask, [0, 0, 0, 0, 1])
        #
        xsub = subarray(x, info={'name': 'x'})
        mxsub = masked_array(xsub)
        assert_(hasattr(mxsub, 'info'))
        assert_equal(mxsub.info, xsub.info)

    def test_subclasspreservation(self):
        # Checks that masked_array(...,subok=True) preserves the class.
        x = np.arange(5)
        m = [0, 0, 1, 0, 0]
        xinfo = list(zip(x, m))
        xsub = MSubArray(x, mask=m, info={'xsub': xinfo})
        #
        mxsub = masked_array(xsub, subok=False)
        assert_(not isinstance(mxsub, MSubArray))
        assert_(isinstance(mxsub, MaskedArray))
        assert_equal(mxsub._mask, m)
        #
        mxsub = asarray(xsub)
        assert_(not isinstance(mxsub, MSubArray))
        assert_(isinstance(mxsub, MaskedArray))
        assert_equal(mxsub._mask, m)
        #
        mxsub = masked_array(xsub, subok=True)
        assert_(isinstance(mxsub, MSubArray))
        assert_equal(mxsub.info, xsub.info)
        assert_equal(mxsub._mask, xsub._mask)
        #
        mxsub = asanyarray(xsub)
        assert_(isinstance(mxsub, MSubArray))
        assert_equal(mxsub.info, xsub.info)
        assert_equal(mxsub._mask, m)

    def test_subclass_items(self):
        """test that getter and setter go via baseclass"""
        x = np.arange(5)
        xcsub = ComplicatedSubArray(x)
        mxcsub = masked_array(xcsub, mask=[True, False, True, False, False])
        # getter should  return a ComplicatedSubArray, even for single item
        # first check we wrote ComplicatedSubArray correctly
        assert_(isinstance(xcsub[1], ComplicatedSubArray))
        assert_(isinstance(xcsub[1, ...], ComplicatedSubArray))
        assert_(isinstance(xcsub[1:4], ComplicatedSubArray))

        # now that it propagates inside the MaskedArray
        assert_(isinstance(mxcsub[1], ComplicatedSubArray))
        assert_(isinstance(mxcsub[1, ...].data, ComplicatedSubArray))
        assert_(mxcsub[0] is masked)
        assert_(isinstance(mxcsub[0, ...].data, ComplicatedSubArray))
        assert_(isinstance(mxcsub[1:4].data, ComplicatedSubArray))

        # also for flattened version (which goes via MaskedIterator)
        assert_(isinstance(mxcsub.flat[1].data, ComplicatedSubArray))
        assert_(mxcsub.flat[0] is masked)
        assert_(isinstance(mxcsub.flat[1:4].base, ComplicatedSubArray))

        # setter should only work with ComplicatedSubArray input
        # first check we wrote ComplicatedSubArray correctly
        assert_raises(ValueError, xcsub.__setitem__, 1, x[4])
        # now that it propagates inside the MaskedArray
        assert_raises(ValueError, mxcsub.__setitem__, 1, x[4])
        assert_raises(ValueError, mxcsub.__setitem__, slice(1, 4), x[1:4])
        mxcsub[1] = xcsub[4]
        mxcsub[1:4] = xcsub[1:4]
        # also for flattened version (which goes via MaskedIterator)
        assert_raises(ValueError, mxcsub.flat.__setitem__, 1, x[4])
        assert_raises(ValueError, mxcsub.flat.__setitem__, slice(1, 4), x[1:4])
        mxcsub.flat[1] = xcsub[4]
        mxcsub.flat[1:4] = xcsub[1:4]

    def test_subclass_nomask_items(self):
        x = np.arange(5)
        xcsub = ComplicatedSubArray(x)
        mxcsub_nomask = masked_array(xcsub)

        assert_(isinstance(mxcsub_nomask[1, ...].data, ComplicatedSubArray))
        assert_(isinstance(mxcsub_nomask[0, ...].data, ComplicatedSubArray))

        assert_(isinstance(mxcsub_nomask[1], ComplicatedSubArray))
        assert_(isinstance(mxcsub_nomask[0], ComplicatedSubArray))

    def test_subclass_repr(self):
        """test that repr uses the name of the subclass
        and 'array' for np.ndarray"""
        x = np.arange(5)
        mx = masked_array(x, mask=[True, False, True, False, False])
        assert_startswith(repr(mx), 'masked_array')
        xsub = SubArray(x)
        mxsub = masked_array(xsub, mask=[True, False, True, False, False])
        assert_startswith(repr(mxsub),
            f'masked_{SubArray.__name__}(data=[--, 1, --, 3, 4]')

    def test_subclass_str(self):
        """test str with subclass that has overridden str, setitem"""
        # first without override
        x = np.arange(5)
        xsub = SubArray(x)
        mxsub = masked_array(xsub, mask=[True, False, True, False, False])
        assert_equal(str(mxsub), '[-- 1 -- 3 4]')

        xcsub = ComplicatedSubArray(x)
        assert_raises(ValueError, xcsub.__setitem__, 0,
                      np.ma.core.masked_print_option)
        mxcsub = masked_array(xcsub, mask=[True, False, True, False, False])
        assert_equal(str(mxcsub), 'myprefix [-- 1 -- 3 4] mypostfix')

    def test_pure_subclass_info_preservation(self):
        # Test that ufuncs and methods conserve extra information consistently;
        # see gh-7122.
        arr1 = SubMaskedArray('test', data=[1, 2, 3, 4, 5, 6])
        arr2 = SubMaskedArray(data=[0, 1, 2, 3, 4, 5])
        diff1 = np.subtract(arr1, arr2)
        assert_('info' in diff1._optinfo)
        assert_(diff1._optinfo['info'] == 'test')
        diff2 = arr1 - arr2
        assert_('info' in diff2._optinfo)
        assert_(diff2._optinfo['info'] == 'test')


class ArrayNoInheritance:
    """Quantity-like class that does not inherit from ndarray"""
    def __init__(self, data, units):
        self.magnitude = data
        self.units = units

    def __getattr__(self, attr):
        return getattr(self.magnitude, attr)


def test_array_no_inheritance():
    data_masked = np.ma.array([1, 2, 3], mask=[True, False, True])
    data_masked_units = ArrayNoInheritance(data_masked, 'meters')

    # Get the masked representation of the Quantity-like class
    new_array = np.ma.array(data_masked_units)
    assert_equal(data_masked.data, new_array.data)
    assert_equal(data_masked.mask, new_array.mask)
    # Test sharing the mask
    data_masked.mask = [True, False, False]
    assert_equal(data_masked.mask, new_array.mask)
    assert_(new_array.sharedmask)

    # Get the masked representation of the Quantity-like class
    new_array = np.ma.array(data_masked_units, copy=True)
    assert_equal(data_masked.data, new_array.data)
    assert_equal(data_masked.mask, new_array.mask)
    # Test that the mask is not shared when copy=True
    data_masked.mask = [True, False, True]
    assert_equal([True, False, False], new_array.mask)
    assert_(not new_array.sharedmask)

    # Get the masked representation of the Quantity-like class
    new_array = np.ma.array(data_masked_units, keep_mask=False)
    assert_equal(data_masked.data, new_array.data)
    # The change did not affect the original mask
    assert_equal(data_masked.mask, [True, False, True])
    # Test that the mask is False and not shared when keep_mask=False
    assert_(not new_array.mask)
    assert_(not new_array.sharedmask)


class TestClassWrapping:
    # Test suite for classes that wrap MaskedArrays

    def setup_method(self):
        m = np.ma.masked_array([1, 3, 5], mask=[False, True, False])
        wm = WrappedArray(m)
        self.data = (m, wm)

    def test_masked_unary_operations(self):
        # Tests masked_unary_operation
        (m, wm) = self.data
        with np.errstate(divide='ignore'):
            assert_(isinstance(np.log(wm), WrappedArray))

    def test_masked_binary_operations(self):
        # Tests masked_binary_operation
        (m, wm) = self.data
        # Result should be a WrappedArray
        assert_(isinstance(np.add(wm, wm), WrappedArray))
        assert_(isinstance(np.add(m, wm), WrappedArray))
        assert_(isinstance(np.add(wm, m), WrappedArray))
        # add and '+' should call the same ufunc
        assert_equal(np.add(m, wm), m + wm)
        assert_(isinstance(np.hypot(m, wm), WrappedArray))
        assert_(isinstance(np.hypot(wm, m), WrappedArray))
        # Test domained binary operations
        assert_(isinstance(np.divide(wm, m), WrappedArray))
        assert_(isinstance(np.divide(m, wm), WrappedArray))
        assert_equal(np.divide(wm, m) * m, np.divide(m, m) * wm)
        # Test broadcasting
        m2 = np.stack([m, m])
        assert_(isinstance(np.divide(wm, m2), WrappedArray))
        assert_(isinstance(np.divide(m2, wm), WrappedArray))
        assert_equal(np.divide(m2, wm), np.divide(wm, m2))

    def test_mixins_have_slots(self):
        mixin = NDArrayOperatorsMixin()
        # Should raise an error
        assert_raises(AttributeError, mixin.__setattr__, "not_a_real_attr", 1)

        m = np.ma.masked_array([1, 3, 5], mask=[False, True, False])
        wm = WrappedArray(m)
        assert_raises(AttributeError, wm.__setattr__, "not_an_attr", 2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\base\getitem.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


class BaseGetitemTests:
    """Tests for ExtensionArray.__getitem__."""

    def test_iloc_series(self, data):
        ser = pd.Series(data)
        result = ser.iloc[:4]
        expected = pd.Series(data[:4])
        tm.assert_series_equal(result, expected)

        result = ser.iloc[[0, 1, 2, 3]]
        tm.assert_series_equal(result, expected)

    def test_iloc_frame(self, data):
        df = pd.DataFrame({"A": data, "B": np.arange(len(data), dtype="int64")})
        expected = pd.DataFrame({"A": data[:4]})

        # slice -> frame
        result = df.iloc[:4, [0]]
        tm.assert_frame_equal(result, expected)

        # sequence -> frame
        result = df.iloc[[0, 1, 2, 3], [0]]
        tm.assert_frame_equal(result, expected)

        expected = pd.Series(data[:4], name="A")

        # slice -> series
        result = df.iloc[:4, 0]
        tm.assert_series_equal(result, expected)

        # sequence -> series
        result = df.iloc[:4, 0]
        tm.assert_series_equal(result, expected)

        # GH#32959 slice columns with step
        result = df.iloc[:, ::2]
        tm.assert_frame_equal(result, df[["A"]])
        result = df[["B", "A"]].iloc[:, ::2]
        tm.assert_frame_equal(result, df[["B"]])

    def test_iloc_frame_single_block(self, data):
        # GH#32959 null slice along index, slice along columns with single-block
        df = pd.DataFrame({"A": data})

        result = df.iloc[:, :]
        tm.assert_frame_equal(result, df)

        result = df.iloc[:, :1]
        tm.assert_frame_equal(result, df)

        result = df.iloc[:, :2]
        tm.assert_frame_equal(result, df)

        result = df.iloc[:, ::2]
        tm.assert_frame_equal(result, df)

        result = df.iloc[:, 1:2]
        tm.assert_frame_equal(result, df.iloc[:, :0])

        result = df.iloc[:, -1:]
        tm.assert_frame_equal(result, df)

    def test_loc_series(self, data):
        ser = pd.Series(data)
        result = ser.loc[:3]
        expected = pd.Series(data[:4])
        tm.assert_series_equal(result, expected)

        result = ser.loc[[0, 1, 2, 3]]
        tm.assert_series_equal(result, expected)

    def test_loc_frame(self, data):
        df = pd.DataFrame({"A": data, "B": np.arange(len(data), dtype="int64")})
        expected = pd.DataFrame({"A": data[:4]})

        # slice -> frame
        result = df.loc[:3, ["A"]]
        tm.assert_frame_equal(result, expected)

        # sequence -> frame
        result = df.loc[[0, 1, 2, 3], ["A"]]
        tm.assert_frame_equal(result, expected)

        expected = pd.Series(data[:4], name="A")

        # slice -> series
        result = df.loc[:3, "A"]
        tm.assert_series_equal(result, expected)

        # sequence -> series
        result = df.loc[:3, "A"]
        tm.assert_series_equal(result, expected)

    def test_loc_iloc_frame_single_dtype(self, data):
        # GH#27110 bug in ExtensionBlock.iget caused df.iloc[n] to incorrectly
        #  return a scalar
        df = pd.DataFrame({"A": data})
        expected = pd.Series([data[2]], index=["A"], name=2, dtype=data.dtype)

        result = df.loc[2]
        tm.assert_series_equal(result, expected)

        expected = pd.Series(
            [data[-1]], index=["A"], name=len(data) - 1, dtype=data.dtype
        )
        result = df.iloc[-1]
        tm.assert_series_equal(result, expected)

    def test_getitem_scalar(self, data):
        result = data[0]
        assert isinstance(result, data.dtype.type)

        result = pd.Series(data)[0]
        assert isinstance(result, data.dtype.type)

    def test_getitem_invalid(self, data):
        # TODO: box over scalar, [scalar], (scalar,)?

        msg = (
            r"only integers, slices \(`:`\), ellipsis \(`...`\), numpy.newaxis "
            r"\(`None`\) and integer or boolean arrays are valid indices"
        )
        with pytest.raises(IndexError, match=msg):
            data["foo"]
        with pytest.raises(IndexError, match=msg):
            data[2.5]

        ub = len(data)
        msg = "|".join(
            [
                "list index out of range",  # json
                "index out of bounds",  # pyarrow
                "Out of bounds access",  # Sparse
                f"loc must be an integer between -{ub} and {ub}",  # Sparse
                f"index {ub+1} is out of bounds for axis 0 with size {ub}",
                f"index -{ub+1} is out of bounds for axis 0 with size {ub}",
            ]
        )
        with pytest.raises(IndexError, match=msg):
            data[ub + 1]
        with pytest.raises(IndexError, match=msg):
            data[-ub - 1]

    def test_getitem_scalar_na(self, data_missing, na_cmp, na_value):
        result = data_missing[0]
        assert na_cmp(result, na_value)

    def test_getitem_empty(self, data):
        # Indexing with empty list
        result = data[[]]
        assert len(result) == 0
        assert isinstance(result, type(data))

        expected = data[np.array([], dtype="int64")]
        tm.assert_extension_array_equal(result, expected)

    def test_getitem_mask(self, data):
        # Empty mask, raw array
        mask = np.zeros(len(data), dtype=bool)
        result = data[mask]
        assert len(result) == 0
        assert isinstance(result, type(data))

        # Empty mask, in series
        mask = np.zeros(len(data), dtype=bool)
        result = pd.Series(data)[mask]
        assert len(result) == 0
        assert result.dtype == data.dtype

        # non-empty mask, raw array
        mask[0] = True
        result = data[mask]
        assert len(result) == 1
        assert isinstance(result, type(data))

        # non-empty mask, in series
        result = pd.Series(data)[mask]
        assert len(result) == 1
        assert result.dtype == data.dtype

    def test_getitem_mask_raises(self, data):
        mask = np.array([True, False])
        msg = f"Boolean index has wrong length: 2 instead of {len(data)}"
        with pytest.raises(IndexError, match=msg):
            data[mask]

        mask = pd.array(mask, dtype="boolean")
        with pytest.raises(IndexError, match=msg):
            data[mask]

    def test_getitem_boolean_array_mask(self, data):
        mask = pd.array(np.zeros(data.shape, dtype="bool"), dtype="boolean")
        result = data[mask]
        assert len(result) == 0
        assert isinstance(result, type(data))

        result = pd.Series(data)[mask]
        assert len(result) == 0
        assert result.dtype == data.dtype

        mask[:5] = True
        expected = data.take([0, 1, 2, 3, 4])
        result = data[mask]
        tm.assert_extension_array_equal(result, expected)

        expected = pd.Series(expected)
        result = pd.Series(data)[mask]
        tm.assert_series_equal(result, expected)

    def test_getitem_boolean_na_treated_as_false(self, data):
        # https://github.com/pandas-dev/pandas/issues/31503
        mask = pd.array(np.zeros(data.shape, dtype="bool"), dtype="boolean")
        mask[:2] = pd.NA
        mask[2:4] = True

        result = data[mask]
        expected = data[mask.fillna(False)]

        tm.assert_extension_array_equal(result, expected)

        s = pd.Series(data)

        result = s[mask]
        expected = s[mask.fillna(False)]

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "idx",
        [[0, 1, 2], pd.array([0, 1, 2], dtype="Int64"), np.array([0, 1, 2])],
        ids=["list", "integer-array", "numpy-array"],
    )
    def test_getitem_integer_array(self, data, idx):
        result = data[idx]
        assert len(result) == 3
        assert isinstance(result, type(data))
        expected = data.take([0, 1, 2])
        tm.assert_extension_array_equal(result, expected)

        expected = pd.Series(expected)
        result = pd.Series(data)[idx]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "idx",
        [[0, 1, 2, pd.NA], pd.array([0, 1, 2, pd.NA], dtype="Int64")],
        ids=["list", "integer-array"],
    )
    def test_getitem_integer_with_missing_raises(self, data, idx):
        msg = "Cannot index with an integer indexer containing NA values"
        with pytest.raises(ValueError, match=msg):
            data[idx]

    @pytest.mark.xfail(
        reason="Tries label-based and raises KeyError; "
        "in some cases raises when calling np.asarray"
    )
    @pytest.mark.parametrize(
        "idx",
        [[0, 1, 2, pd.NA], pd.array([0, 1, 2, pd.NA], dtype="Int64")],
        ids=["list", "integer-array"],
    )
    def test_getitem_series_integer_with_missing_raises(self, data, idx):
        msg = "Cannot index with an integer indexer containing NA values"
        # TODO: this raises KeyError about labels not found (it tries label-based)

        ser = pd.Series(data, index=[chr(100 + i) for i in range(len(data))])
        with pytest.raises(ValueError, match=msg):
            ser[idx]

    def test_getitem_slice(self, data):
        # getitem[slice] should return an array
        result = data[slice(0)]  # empty
        assert isinstance(result, type(data))

        result = data[slice(1)]  # scalar
        assert isinstance(result, type(data))

    def test_getitem_ellipsis_and_slice(self, data):
        # GH#40353 this is called from slice_block_rows
        result = data[..., :]
        tm.assert_extension_array_equal(result, data)

        result = data[:, ...]
        tm.assert_extension_array_equal(result, data)

        result = data[..., :3]
        tm.assert_extension_array_equal(result, data[:3])

        result = data[:3, ...]
        tm.assert_extension_array_equal(result, data[:3])

        result = data[..., ::2]
        tm.assert_extension_array_equal(result, data[::2])

        result = data[::2, ...]
        tm.assert_extension_array_equal(result, data[::2])

    def test_get(self, data):
        # GH 20882
        s = pd.Series(data, index=[2 * i for i in range(len(data))])
        assert s.get(4) == s.iloc[2]

        result = s.get([4, 6])
        expected = s.iloc[[2, 3]]
        tm.assert_series_equal(result, expected)

        result = s.get(slice(2))
        expected = s.iloc[[0, 1]]
        tm.assert_series_equal(result, expected)

        assert s.get(-1) is None
        assert s.get(s.index.max() + 1) is None

        s = pd.Series(data[:6], index=list("abcdef"))
        assert s.get("c") == s.iloc[2]

        result = s.get(slice("b", "d"))
        expected = s.iloc[[1, 2, 3]]
        tm.assert_series_equal(result, expected)

        result = s.get("Z")
        assert result is None

        msg = "Series.__getitem__ treating keys as positions is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert s.get(4) == s.iloc[4]
            assert s.get(-1) == s.iloc[-1]
            assert s.get(len(s)) is None

        # GH 21257
        s = pd.Series(data)
        with tm.assert_produces_warning(None):
            # GH#45324 make sure we aren't giving a spurious FutureWarning
            s2 = s[::2]
        assert s2.get(1) is None

    def test_take_sequence(self, data):
        result = pd.Series(data)[[0, 1, 3]]
        assert result.iloc[0] == data[0]
        assert result.iloc[1] == data[1]
        assert result.iloc[2] == data[3]

    def test_take(self, data, na_value, na_cmp):
        result = data.take([0, -1])
        assert result.dtype == data.dtype
        assert result[0] == data[0]
        assert result[1] == data[-1]

        result = data.take([0, -1], allow_fill=True, fill_value=na_value)
        assert result[0] == data[0]
        assert na_cmp(result[1], na_value)

        with pytest.raises(IndexError, match="out of bounds"):
            data.take([len(data) + 1])

    def test_take_empty(self, data, na_value, na_cmp):
        empty = data[:0]

        result = empty.take([-1], allow_fill=True)
        assert na_cmp(result[0], na_value)

        msg = "cannot do a non-empty take from an empty axes|out of bounds"

        with pytest.raises(IndexError, match=msg):
            empty.take([-1])

        with pytest.raises(IndexError, match="cannot do a non-empty take"):
            empty.take([0, 1])

    def test_take_negative(self, data):
        # https://github.com/pandas-dev/pandas/issues/20640
        n = len(data)
        result = data.take([0, -n, n - 1, -1])
        expected = data.take([0, 0, n - 1, n - 1])
        tm.assert_extension_array_equal(result, expected)

    def test_take_non_na_fill_value(self, data_missing):
        fill_value = data_missing[1]  # valid
        na = data_missing[0]

        arr = data_missing._from_sequence(
            [na, fill_value, na], dtype=data_missing.dtype
        )
        result = arr.take([-1, 1], fill_value=fill_value, allow_fill=True)
        expected = arr.take([1, 1])
        tm.assert_extension_array_equal(result, expected)

    def test_take_pandas_style_negative_raises(self, data, na_value):
        with pytest.raises(ValueError, match=""):
            data.take([0, -2], fill_value=na_value, allow_fill=True)

    @pytest.mark.parametrize("allow_fill", [True, False])
    def test_take_out_of_bounds_raises(self, data, allow_fill):
        arr = data[:3]

        with pytest.raises(IndexError, match="out of bounds|out-of-bounds"):
            arr.take(np.asarray([0, 3]), allow_fill=allow_fill)

    def test_take_series(self, data):
        s = pd.Series(data)
        result = s.take([0, -1])
        expected = pd.Series(
            data._from_sequence([data[0], data[len(data) - 1]], dtype=s.dtype),
            index=[0, len(data) - 1],
        )
        tm.assert_series_equal(result, expected)

    def test_reindex(self, data, na_value):
        s = pd.Series(data)
        result = s.reindex([0, 1, 3])
        expected = pd.Series(data.take([0, 1, 3]), index=[0, 1, 3])
        tm.assert_series_equal(result, expected)

        n = len(data)
        result = s.reindex([-1, 0, n])
        expected = pd.Series(
            data._from_sequence([na_value, data[0], na_value], dtype=s.dtype),
            index=[-1, 0, n],
        )
        tm.assert_series_equal(result, expected)

        result = s.reindex([n, n + 1])
        expected = pd.Series(
            data._from_sequence([na_value, na_value], dtype=s.dtype), index=[n, n + 1]
        )
        tm.assert_series_equal(result, expected)

    def test_reindex_non_na_fill_value(self, data_missing):
        valid = data_missing[1]
        na = data_missing[0]

        arr = data_missing._from_sequence([na, valid], dtype=data_missing.dtype)
        ser = pd.Series(arr)
        result = ser.reindex([0, 1, 2], fill_value=valid)
        expected = pd.Series(
            data_missing._from_sequence([na, valid, valid], dtype=data_missing.dtype)
        )

        tm.assert_series_equal(result, expected)

    def test_loc_len1(self, data):
        # see GH-27785 take_nd with indexer of len 1 resulting in wrong ndim
        df = pd.DataFrame({"A": data})
        res = df.loc[[0], "A"]
        assert res.ndim == 1
        assert res._mgr.arrays[0].ndim == 1
        if hasattr(res._mgr, "blocks"):
            assert res._mgr._block.ndim == 1

    def test_item(self, data):
        # https://github.com/pandas-dev/pandas/pull/30175
        s = pd.Series(data)
        result = s[:1].item()
        assert result == data[0]

        msg = "can only convert an array of size 1 to a Python scalar"
        with pytest.raises(ValueError, match=msg):
            s[:0].item()

        with pytest.raises(ValueError, match=msg):
            s.item()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\tests\test_blockmatrix.py ===
from sympy.matrices.expressions.trace import Trace
from sympy.testing.pytest import raises, slow
from sympy.matrices.expressions.blockmatrix import (
    block_collapse, bc_matmul, bc_block_plus_ident, BlockDiagMatrix,
    BlockMatrix, bc_dist, bc_matadd, bc_transpose, bc_inverse,
    blockcut, reblock_2x2, deblock)
from sympy.matrices.expressions import (
    MatrixSymbol, Identity, trace, det, ZeroMatrix, OneMatrix)
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matpow import MatPow
from sympy.matrices.expressions.transpose import Transpose
from sympy.matrices.exceptions import NonInvertibleMatrixError
from sympy.matrices import (
    Matrix, ImmutableMatrix, ImmutableSparseMatrix, zeros)
from sympy.core import Tuple, Expr, S, Function
from sympy.core.symbol import Symbol, symbols
from sympy.functions import transpose, im, re

i, j, k, l, m, n, p = symbols('i:n, p', integer=True)
A = MatrixSymbol('A', n, n)
B = MatrixSymbol('B', n, n)
C = MatrixSymbol('C', n, n)
D = MatrixSymbol('D', n, n)
G = MatrixSymbol('G', n, n)
H = MatrixSymbol('H', n, n)
b1 = BlockMatrix([[G, H]])
b2 = BlockMatrix([[G], [H]])

def test_bc_matmul():
    assert bc_matmul(H*b1*b2*G) == BlockMatrix([[(H*G*G + H*H*H)*G]])

def test_bc_matadd():
    assert bc_matadd(BlockMatrix([[G, H]]) + BlockMatrix([[H, H]])) == \
            BlockMatrix([[G+H, H+H]])

def test_bc_transpose():
    assert bc_transpose(Transpose(BlockMatrix([[A, B], [C, D]]))) == \
            BlockMatrix([[A.T, C.T], [B.T, D.T]])

def test_bc_dist_diag():
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', m, m)
    C = MatrixSymbol('C', l, l)
    X = BlockDiagMatrix(A, B, C)

    assert bc_dist(X+X).equals(BlockDiagMatrix(2*A, 2*B, 2*C))

def test_block_plus_ident():
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, m)
    C = MatrixSymbol('C', m, n)
    D = MatrixSymbol('D', m, m)
    X = BlockMatrix([[A, B], [C, D]])
    Z = MatrixSymbol('Z', n + m, n + m)
    assert bc_block_plus_ident(X + Identity(m + n) + Z) == \
            BlockDiagMatrix(Identity(n), Identity(m)) + X + Z

def test_BlockMatrix():
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', n, k)
    C = MatrixSymbol('C', l, m)
    D = MatrixSymbol('D', l, k)
    M = MatrixSymbol('M', m + k, p)
    N = MatrixSymbol('N', l + n, k + m)
    X = BlockMatrix(Matrix([[A, B], [C, D]]))

    assert X.__class__(*X.args) == X

    # block_collapse does nothing on normal inputs
    E = MatrixSymbol('E', n, m)
    assert block_collapse(A + 2*E) == A + 2*E
    F = MatrixSymbol('F', m, m)
    assert block_collapse(E.T*A*F) == E.T*A*F

    assert X.shape == (l + n, k + m)
    assert X.blockshape == (2, 2)
    assert transpose(X) == BlockMatrix(Matrix([[A.T, C.T], [B.T, D.T]]))
    assert transpose(X).shape == X.shape[::-1]

    # Test that BlockMatrices and MatrixSymbols can still mix
    assert (X*M).is_MatMul
    assert X._blockmul(M).is_MatMul
    assert (X*M).shape == (n + l, p)
    assert (X + N).is_MatAdd
    assert X._blockadd(N).is_MatAdd
    assert (X + N).shape == X.shape

    E = MatrixSymbol('E', m, 1)
    F = MatrixSymbol('F', k, 1)

    Y = BlockMatrix(Matrix([[E], [F]]))

    assert (X*Y).shape == (l + n, 1)
    assert block_collapse(X*Y).blocks[0, 0] == A*E + B*F
    assert block_collapse(X*Y).blocks[1, 0] == C*E + D*F

    # block_collapse passes down into container objects, transposes, and inverse
    assert block_collapse(transpose(X*Y)) == transpose(block_collapse(X*Y))
    assert block_collapse(Tuple(X*Y, 2*X)) == (
        block_collapse(X*Y), block_collapse(2*X))

    # Make sure that MatrixSymbols will enter 1x1 BlockMatrix if it simplifies
    Ab = BlockMatrix([[A]])
    Z = MatrixSymbol('Z', *A.shape)
    assert block_collapse(Ab + Z) == A + Z

def test_block_collapse_explicit_matrices():
    A = Matrix([[1, 2], [3, 4]])
    assert block_collapse(BlockMatrix([[A]])) == A

    A = ImmutableSparseMatrix([[1, 2], [3, 4]])
    assert block_collapse(BlockMatrix([[A]])) == A

def test_issue_17624():
    a = MatrixSymbol("a", 2, 2)
    z = ZeroMatrix(2, 2)
    b = BlockMatrix([[a, z], [z, z]])
    assert block_collapse(b * b) == BlockMatrix([[a**2, z], [z, z]])
    assert block_collapse(b * b * b) == BlockMatrix([[a**3, z], [z, z]])

def test_issue_18618():
    A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert A == Matrix(BlockDiagMatrix(A))

def test_BlockMatrix_trace():
    A, B, C, D = [MatrixSymbol(s, 3, 3) for s in 'ABCD']
    X = BlockMatrix([[A, B], [C, D]])
    assert trace(X) == trace(A) + trace(D)
    assert trace(BlockMatrix([ZeroMatrix(n, n)])) == 0

def test_BlockMatrix_Determinant():
    A, B, C, D = [MatrixSymbol(s, 3, 3) for s in 'ABCD']
    X = BlockMatrix([[A, B], [C, D]])
    from sympy.assumptions.ask import Q
    from sympy.assumptions.assume import assuming
    with assuming(Q.invertible(A)):
        assert det(X) == det(A) * det(X.schur('A'))

    assert isinstance(det(X), Expr)
    assert det(BlockMatrix([A])) == det(A)
    assert det(BlockMatrix([ZeroMatrix(n, n)])) == 0

def test_squareBlockMatrix():
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, m)
    C = MatrixSymbol('C', m, n)
    D = MatrixSymbol('D', m, m)
    X = BlockMatrix([[A, B], [C, D]])
    Y = BlockMatrix([[A]])

    assert X.is_square

    Q = X + Identity(m + n)
    assert (block_collapse(Q) ==
        BlockMatrix([[A + Identity(n), B], [C, D + Identity(m)]]))

    assert (X + MatrixSymbol('Q', n + m, n + m)).is_MatAdd
    assert (X * MatrixSymbol('Q', n + m, n + m)).is_MatMul

    assert block_collapse(Y.I) == A.I

    assert isinstance(X.inverse(), Inverse)

    assert not X.is_Identity

    Z = BlockMatrix([[Identity(n), B], [C, D]])
    assert not Z.is_Identity


def test_BlockMatrix_2x2_inverse_symbolic():
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', n, k - m)
    C = MatrixSymbol('C', k - n, m)
    D = MatrixSymbol('D', k - n, k - m)
    X = BlockMatrix([[A, B], [C, D]])
    assert X.is_square and X.shape == (k, k)
    assert isinstance(block_collapse(X.I), Inverse)  # Can't invert when none of the blocks is square

    # test code path where only A is invertible
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, m)
    C = MatrixSymbol('C', m, n)
    D = ZeroMatrix(m, m)
    X = BlockMatrix([[A, B], [C, D]])
    assert block_collapse(X.inverse()) == BlockMatrix([
        [A.I + A.I * B * X.schur('A').I * C * A.I, -A.I * B * X.schur('A').I],
        [-X.schur('A').I * C * A.I, X.schur('A').I],
    ])

    # test code path where only B is invertible
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', n, n)
    C = ZeroMatrix(m, m)
    D = MatrixSymbol('D', m, n)
    X = BlockMatrix([[A, B], [C, D]])
    assert block_collapse(X.inverse()) == BlockMatrix([
        [-X.schur('B').I * D * B.I, X.schur('B').I],
        [B.I + B.I * A * X.schur('B').I * D * B.I, -B.I * A * X.schur('B').I],
    ])

    # test code path where only C is invertible
    A = MatrixSymbol('A', n, m)
    B = ZeroMatrix(n, n)
    C = MatrixSymbol('C', m, m)
    D = MatrixSymbol('D', m, n)
    X = BlockMatrix([[A, B], [C, D]])
    assert block_collapse(X.inverse()) == BlockMatrix([
        [-C.I * D * X.schur('C').I, C.I + C.I * D * X.schur('C').I * A * C.I],
        [X.schur('C').I, -X.schur('C').I * A * C.I],
    ])

    # test code path where only D is invertible
    A = ZeroMatrix(n, n)
    B = MatrixSymbol('B', n, m)
    C = MatrixSymbol('C', m, n)
    D = MatrixSymbol('D', m, m)
    X = BlockMatrix([[A, B], [C, D]])
    assert block_collapse(X.inverse()) == BlockMatrix([
        [X.schur('D').I, -X.schur('D').I * B * D.I],
        [-D.I * C * X.schur('D').I, D.I + D.I * C * X.schur('D').I * B * D.I],
    ])


def test_BlockMatrix_2x2_inverse_numeric():
    """Test 2x2 block matrix inversion numerically for all 4 formulas"""
    M = Matrix([[1, 2], [3, 4]])
    # rank deficient matrices that have full rank when two of them combined
    D1 = Matrix([[1, 2], [2, 4]])
    D2 = Matrix([[1, 3], [3, 9]])
    D3 = Matrix([[1, 4], [4, 16]])
    assert D1.rank() == D2.rank() == D3.rank() == 1
    assert (D1 + D2).rank() == (D2 + D3).rank() == (D3 + D1).rank() == 2

    # Only A is invertible
    K = BlockMatrix([[M, D1], [D2, D3]])
    assert block_collapse(K.inv()).as_explicit() == K.as_explicit().inv()
    # Only B is invertible
    K = BlockMatrix([[D1, M], [D2, D3]])
    assert block_collapse(K.inv()).as_explicit() == K.as_explicit().inv()
    # Only C is invertible
    K = BlockMatrix([[D1, D2], [M, D3]])
    assert block_collapse(K.inv()).as_explicit() == K.as_explicit().inv()
    # Only D is invertible
    K = BlockMatrix([[D1, D2], [D3, M]])
    assert block_collapse(K.inv()).as_explicit() == K.as_explicit().inv()


@slow
def test_BlockMatrix_3x3_symbolic():
    # Only test one of these, instead of all permutations, because it's slow
    rowblocksizes = (n, m, k)
    colblocksizes = (m, k, n)
    K = BlockMatrix([
        [MatrixSymbol('M%s%s' % (rows, cols), rows, cols) for cols in colblocksizes]
        for rows in rowblocksizes
    ])
    collapse = block_collapse(K.I)
    assert isinstance(collapse, BlockMatrix)


def test_BlockDiagMatrix():
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', m, m)
    C = MatrixSymbol('C', l, l)
    M = MatrixSymbol('M', n + m + l, n + m + l)

    X = BlockDiagMatrix(A, B, C)
    Y = BlockDiagMatrix(A, 2*B, 3*C)

    assert X.blocks[1, 1] == B
    assert X.shape == (n + m + l, n + m + l)
    assert all(X.blocks[i, j].is_ZeroMatrix if i != j else X.blocks[i, j] in [A, B, C]
            for i in range(3) for j in range(3))
    assert X.__class__(*X.args) == X
    assert X.get_diag_blocks() == (A, B, C)

    assert isinstance(block_collapse(X.I * X), Identity)

    assert bc_matmul(X*X) == BlockDiagMatrix(A*A, B*B, C*C)
    assert block_collapse(X*X) == BlockDiagMatrix(A*A, B*B, C*C)
    #XXX: should be == ??
    assert block_collapse(X + X).equals(BlockDiagMatrix(2*A, 2*B, 2*C))
    assert block_collapse(X*Y) == BlockDiagMatrix(A*A, 2*B*B, 3*C*C)
    assert block_collapse(X + Y) == BlockDiagMatrix(2*A, 3*B, 4*C)

    # Ensure that BlockDiagMatrices can still interact with normal MatrixExprs
    assert (X*(2*M)).is_MatMul
    assert (X + (2*M)).is_MatAdd

    assert (X._blockmul(M)).is_MatMul
    assert (X._blockadd(M)).is_MatAdd

def test_BlockDiagMatrix_nonsquare():
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', k, l)
    X = BlockDiagMatrix(A, B)
    assert X.shape == (n + k, m + l)
    assert X.shape == (n + k, m + l)
    assert X.rowblocksizes == [n, k]
    assert X.colblocksizes == [m, l]
    C = MatrixSymbol('C', n, m)
    D = MatrixSymbol('D', k, l)
    Y = BlockDiagMatrix(C, D)
    assert block_collapse(X + Y) == BlockDiagMatrix(A + C, B + D)
    assert block_collapse(X * Y.T) == BlockDiagMatrix(A * C.T, B * D.T)
    raises(NonInvertibleMatrixError, lambda: BlockDiagMatrix(A, C.T).inverse())

def test_BlockDiagMatrix_determinant():
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', m, m)
    assert det(BlockDiagMatrix()) == 1
    assert det(BlockDiagMatrix(A)) == det(A)
    assert det(BlockDiagMatrix(A, B)) == det(A) * det(B)

    # non-square blocks
    C = MatrixSymbol('C', m, n)
    D = MatrixSymbol('D', n, m)
    assert det(BlockDiagMatrix(C, D)) == 0

def test_BlockDiagMatrix_trace():
    assert trace(BlockDiagMatrix()) == 0
    assert trace(BlockDiagMatrix(ZeroMatrix(n, n))) == 0
    A = MatrixSymbol('A', n, n)
    assert trace(BlockDiagMatrix(A)) == trace(A)
    B = MatrixSymbol('B', m, m)
    assert trace(BlockDiagMatrix(A, B)) == trace(A) + trace(B)

    # non-square blocks
    C = MatrixSymbol('C', m, n)
    D = MatrixSymbol('D', n, m)
    assert isinstance(trace(BlockDiagMatrix(C, D)), Trace)

def test_BlockDiagMatrix_transpose():
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', k, l)
    assert transpose(BlockDiagMatrix()) == BlockDiagMatrix()
    assert transpose(BlockDiagMatrix(A)) == BlockDiagMatrix(A.T)
    assert transpose(BlockDiagMatrix(A, B)) == BlockDiagMatrix(A.T, B.T)

def test_issue_2460():
    bdm1 = BlockDiagMatrix(Matrix([i]), Matrix([j]))
    bdm2 = BlockDiagMatrix(Matrix([k]), Matrix([l]))
    assert block_collapse(bdm1 + bdm2) == BlockDiagMatrix(Matrix([i + k]), Matrix([j + l]))

def test_blockcut():
    A = MatrixSymbol('A', n, m)
    B = blockcut(A, (n/2, n/2), (m/2, m/2))
    assert B == BlockMatrix([[A[:n/2, :m/2], A[:n/2, m/2:]],
                             [A[n/2:, :m/2], A[n/2:, m/2:]]])

    M = ImmutableMatrix(4, 4, range(16))
    B = blockcut(M, (2, 2), (2, 2))
    assert M == ImmutableMatrix(B)

    B = blockcut(M, (1, 3), (2, 2))
    assert ImmutableMatrix(B.blocks[0, 1]) == ImmutableMatrix([[2, 3]])

def test_reblock_2x2():
    B = BlockMatrix([[MatrixSymbol('A_%d%d'%(i,j), 2, 2)
                            for j in range(3)]
                            for i in range(3)])
    assert B.blocks.shape == (3, 3)

    BB = reblock_2x2(B)
    assert BB.blocks.shape == (2, 2)

    assert B.shape == BB.shape
    assert B.as_explicit() == BB.as_explicit()

def test_deblock():
    B = BlockMatrix([[MatrixSymbol('A_%d%d'%(i,j), n, n)
                    for j in range(4)]
                    for i in range(4)])

    assert deblock(reblock_2x2(B)) == B

def test_block_collapse_type():
    bm1 = BlockDiagMatrix(ImmutableMatrix([1]), ImmutableMatrix([2]))
    bm2 = BlockDiagMatrix(ImmutableMatrix([3]), ImmutableMatrix([4]))

    assert bm1.T.__class__ == BlockDiagMatrix
    assert block_collapse(bm1 - bm2).__class__ == BlockDiagMatrix
    assert block_collapse(Inverse(bm1)).__class__ == BlockDiagMatrix
    assert block_collapse(Transpose(bm1)).__class__ == BlockDiagMatrix
    assert bc_transpose(Transpose(bm1)).__class__ == BlockDiagMatrix
    assert bc_inverse(Inverse(bm1)).__class__ == BlockDiagMatrix

def test_invalid_block_matrix():
    raises(ValueError, lambda: BlockMatrix([
        [Identity(2), Identity(5)],
    ]))
    raises(ValueError, lambda: BlockMatrix([
        [Identity(n), Identity(m)],
    ]))
    raises(ValueError, lambda: BlockMatrix([
        [ZeroMatrix(n, n), ZeroMatrix(n, n)],
        [ZeroMatrix(n, n - 1), ZeroMatrix(n, n + 1)],
    ]))
    raises(ValueError, lambda: BlockMatrix([
        [ZeroMatrix(n - 1, n), ZeroMatrix(n, n)],
        [ZeroMatrix(n + 1, n), ZeroMatrix(n, n)],
    ]))

def test_block_lu_decomposition():
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, m)
    C = MatrixSymbol('C', m, n)
    D = MatrixSymbol('D', m, m)
    X = BlockMatrix([[A, B], [C, D]])

    #LDU decomposition
    L, D, U = X.LDUdecomposition()
    assert block_collapse(L*D*U) == X

    #UDL decomposition
    U, D, L = X.UDLdecomposition()
    assert block_collapse(U*D*L) == X

    #LU decomposition
    L, U = X.LUdecomposition()
    assert block_collapse(L*U) == X

def test_issue_21866():
    n  = 10
    I  = Identity(n)
    O  = ZeroMatrix(n, n)
    A  = BlockMatrix([[  I,  O,  O,  O ],
                      [  O,  I,  O,  O ],
                      [  O,  O,  I,  O ],
                      [  I,  O,  O,  I ]])
    Ainv = block_collapse(A.inv())
    AinvT = BlockMatrix([[  I,  O,  O,  O ],
                      [  O,  I,  O,  O ],
                      [  O,  O,  I,  O ],
                      [  -I,  O,  O,  I ]])
    assert Ainv == AinvT


def test_adjoint_and_special_matrices():
    A = Identity(3)
    B = OneMatrix(3, 2)
    C = ZeroMatrix(2, 3)
    D = Identity(2)
    X = BlockMatrix([[A, B], [C, D]])
    X2 = BlockMatrix([[A, S.ImaginaryUnit*B], [C, D]])
    assert X.adjoint() == BlockMatrix([[A, ZeroMatrix(3, 2)], [OneMatrix(2, 3), D]])
    assert re(X) == X
    assert X2.adjoint() == BlockMatrix([[A, ZeroMatrix(3, 2)], [-S.ImaginaryUnit*OneMatrix(2, 3), D]])
    assert im(X2) == BlockMatrix([[ZeroMatrix(3, 3), OneMatrix(3, 2)], [ZeroMatrix(2, 3), ZeroMatrix(2, 2)]])


def test_block_matrix_derivative():
    x = symbols('x')
    A = Matrix(3, 3, [Function(f'a{i}')(x) for i in range(9)])
    bc = BlockMatrix([[A[:2, :2], A[:2, 2]], [A[2, :2], A[2:, 2]]])
    assert Matrix(bc.diff(x)) - A.diff(x) == zeros(3, 3)


def test_transpose_inverse_commute():
    n = Symbol('n')
    I = Identity(n)
    Z = ZeroMatrix(n, n)
    A = BlockMatrix([[I, Z], [Z, I]])

    assert block_collapse(A.transpose().inverse()) == A
    assert block_collapse(A.inverse().transpose()) == A

    assert block_collapse(MatPow(A.transpose(), -2)) == MatPow(A, -2)
    assert block_collapse(MatPow(A, -2).transpose()) == MatPow(A, -2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\tests\test_autowrap.py ===
# Tests that require installed backends go into
# sympy/test_external/test_autowrap

import os
import tempfile
import shutil
from io import StringIO
from pathlib import Path

from sympy.core import symbols, Eq
from sympy.utilities.autowrap import (autowrap, binary_function,
            CythonCodeWrapper, UfuncifyCodeWrapper, CodeWrapper)
from sympy.utilities.codegen import (
    CCodeGen, C99CodeGen, CodeGenArgumentListError, make_routine
)
from sympy.testing.pytest import raises
from sympy.testing.tmpfiles import TmpFileManager


def get_string(dump_fn, routines, prefix="file", **kwargs):
    """Wrapper for dump_fn. dump_fn writes its results to a stream object and
       this wrapper returns the contents of that stream as a string. This
       auxiliary function is used by many tests below.

       The header and the empty lines are not generator to facilitate the
       testing of the output.
    """
    output = StringIO()
    dump_fn(routines, output, prefix, **kwargs)
    source = output.getvalue()
    output.close()
    return source


def test_cython_wrapper_scalar_function():
    x, y, z = symbols('x,y,z')
    expr = (x + y)*z
    routine = make_routine("test", expr)
    code_gen = CythonCodeWrapper(CCodeGen())
    source = get_string(code_gen.dump_pyx, [routine])

    expected = (
        "cdef extern from 'file.h':\n"
        "    double test(double x, double y, double z)\n"
        "\n"
        "def test_c(double x, double y, double z):\n"
        "\n"
        "    return test(x, y, z)")
    assert source == expected


def test_cython_wrapper_outarg():
    from sympy.core.relational import Equality
    x, y, z = symbols('x,y,z')
    code_gen = CythonCodeWrapper(C99CodeGen())

    routine = make_routine("test", Equality(z, x + y))
    source = get_string(code_gen.dump_pyx, [routine])
    expected = (
        "cdef extern from 'file.h':\n"
        "    void test(double x, double y, double *z)\n"
        "\n"
        "def test_c(double x, double y):\n"
        "\n"
        "    cdef double z = 0\n"
        "    test(x, y, &z)\n"
        "    return z")
    assert source == expected


def test_cython_wrapper_inoutarg():
    from sympy.core.relational import Equality
    x, y, z = symbols('x,y,z')
    code_gen = CythonCodeWrapper(C99CodeGen())
    routine = make_routine("test", Equality(z, x + y + z))
    source = get_string(code_gen.dump_pyx, [routine])
    expected = (
        "cdef extern from 'file.h':\n"
        "    void test(double x, double y, double *z)\n"
        "\n"
        "def test_c(double x, double y, double z):\n"
        "\n"
        "    test(x, y, &z)\n"
        "    return z")
    assert source == expected


def test_cython_wrapper_compile_flags():
    from sympy.core.relational import Equality
    x, y, z = symbols('x,y,z')
    routine = make_routine("test", Equality(z, x + y))

    code_gen = CythonCodeWrapper(CCodeGen())

    expected = """\
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
cy_opts = {'compiler_directives': {'language_level': '3'}}

ext_mods = [Extension(
    'wrapper_module_%(num)s', ['wrapper_module_%(num)s.pyx', 'wrapped_code_%(num)s.c'],
    include_dirs=[],
    library_dirs=[],
    libraries=[],
    extra_compile_args=['-std=c99'],
    extra_link_args=[]
)]
setup(ext_modules=cythonize(ext_mods, **cy_opts))
""" % {'num': CodeWrapper._module_counter}

    temp_dir = tempfile.mkdtemp()
    TmpFileManager.tmp_folder(temp_dir)
    setup_file_path = os.path.join(temp_dir, 'setup.py')

    code_gen._prepare_files(routine, build_dir=temp_dir)
    setup_text = Path(setup_file_path).read_text()
    assert setup_text == expected

    code_gen = CythonCodeWrapper(CCodeGen(),
                                 include_dirs=['/usr/local/include', '/opt/booger/include'],
                                 library_dirs=['/user/local/lib'],
                                 libraries=['thelib', 'nilib'],
                                 extra_compile_args=['-slow-math'],
                                 extra_link_args=['-lswamp', '-ltrident'],
                                 cythonize_options={'compiler_directives': {'boundscheck': False}}
                                 )
    expected = """\
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
cy_opts = {'compiler_directives': {'boundscheck': False}}

ext_mods = [Extension(
    'wrapper_module_%(num)s', ['wrapper_module_%(num)s.pyx', 'wrapped_code_%(num)s.c'],
    include_dirs=['/usr/local/include', '/opt/booger/include'],
    library_dirs=['/user/local/lib'],
    libraries=['thelib', 'nilib'],
    extra_compile_args=['-slow-math', '-std=c99'],
    extra_link_args=['-lswamp', '-ltrident']
)]
setup(ext_modules=cythonize(ext_mods, **cy_opts))
""" % {'num': CodeWrapper._module_counter}

    code_gen._prepare_files(routine, build_dir=temp_dir)
    setup_text = Path(setup_file_path).read_text()
    assert setup_text == expected

    expected = """\
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
cy_opts = {'compiler_directives': {'boundscheck': False}}
import numpy as np

ext_mods = [Extension(
    'wrapper_module_%(num)s', ['wrapper_module_%(num)s.pyx', 'wrapped_code_%(num)s.c'],
    include_dirs=['/usr/local/include', '/opt/booger/include', np.get_include()],
    library_dirs=['/user/local/lib'],
    libraries=['thelib', 'nilib'],
    extra_compile_args=['-slow-math', '-std=c99'],
    extra_link_args=['-lswamp', '-ltrident']
)]
setup(ext_modules=cythonize(ext_mods, **cy_opts))
""" % {'num': CodeWrapper._module_counter}

    code_gen._need_numpy = True
    code_gen._prepare_files(routine, build_dir=temp_dir)
    setup_text = Path(setup_file_path).read_text()
    assert setup_text == expected

    TmpFileManager.cleanup()

def test_cython_wrapper_unique_dummyvars():
    from sympy.core.relational import Equality
    from sympy.core.symbol import Dummy
    x, y, z = Dummy('x'), Dummy('y'), Dummy('z')
    x_id, y_id, z_id = [str(d.dummy_index) for d in [x, y, z]]
    expr = Equality(z, x + y)
    routine = make_routine("test", expr)
    code_gen = CythonCodeWrapper(CCodeGen())
    source = get_string(code_gen.dump_pyx, [routine])
    expected_template = (
        "cdef extern from 'file.h':\n"
        "    void test(double x_{x_id}, double y_{y_id}, double *z_{z_id})\n"
        "\n"
        "def test_c(double x_{x_id}, double y_{y_id}):\n"
        "\n"
        "    cdef double z_{z_id} = 0\n"
        "    test(x_{x_id}, y_{y_id}, &z_{z_id})\n"
        "    return z_{z_id}")
    expected = expected_template.format(x_id=x_id, y_id=y_id, z_id=z_id)
    assert source == expected

def test_autowrap_dummy():
    x, y, z = symbols('x y z')

    # Uses DummyWrapper to test that codegen works as expected

    f = autowrap(x + y, backend='dummy')
    assert f() == str(x + y)
    assert f.args == "x, y"
    assert f.returns == "nameless"
    f = autowrap(Eq(z, x + y), backend='dummy')
    assert f() == str(x + y)
    assert f.args == "x, y"
    assert f.returns == "z"
    f = autowrap(Eq(z, x + y + z), backend='dummy')
    assert f() == str(x + y + z)
    assert f.args == "x, y, z"
    assert f.returns == "z"


def test_autowrap_args():
    x, y, z = symbols('x y z')

    raises(CodeGenArgumentListError, lambda: autowrap(Eq(z, x + y),
           backend='dummy', args=[x]))
    f = autowrap(Eq(z, x + y), backend='dummy', args=[y, x])
    assert f() == str(x + y)
    assert f.args == "y, x"
    assert f.returns == "z"

    raises(CodeGenArgumentListError, lambda: autowrap(Eq(z, x + y + z),
           backend='dummy', args=[x, y]))
    f = autowrap(Eq(z, x + y + z), backend='dummy', args=[y, x, z])
    assert f() == str(x + y + z)
    assert f.args == "y, x, z"
    assert f.returns == "z"

    f = autowrap(Eq(z, x + y + z), backend='dummy', args=(y, x, z))
    assert f() == str(x + y + z)
    assert f.args == "y, x, z"
    assert f.returns == "z"

def test_autowrap_store_files():
    x, y = symbols('x y')
    tmp = tempfile.mkdtemp()
    TmpFileManager.tmp_folder(tmp)

    f = autowrap(x + y, backend='dummy', tempdir=tmp)
    assert f() == str(x + y)
    assert os.access(tmp, os.F_OK)

    TmpFileManager.cleanup()

def test_autowrap_store_files_issue_gh12939():
    x, y = symbols('x y')
    tmp = './tmp'
    saved_cwd = os.getcwd()
    temp_cwd = tempfile.mkdtemp()
    try:
        os.chdir(temp_cwd)
        f = autowrap(x + y, backend='dummy', tempdir=tmp)
        assert f() == str(x + y)
        assert os.access(tmp, os.F_OK)
    finally:
        os.chdir(saved_cwd)
        shutil.rmtree(temp_cwd)


def test_binary_function():
    x, y = symbols('x y')
    f = binary_function('f', x + y, backend='dummy')
    assert f._imp_() == str(x + y)


def test_ufuncify_source():
    x, y, z = symbols('x,y,z')
    code_wrapper = UfuncifyCodeWrapper(C99CodeGen("ufuncify"))
    routine = make_routine("test", x + y + z)
    source = get_string(code_wrapper.dump_c, [routine])
    expected = """\
#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"
#include "file.h"

static PyMethodDef wrapper_module_%(num)sMethods[] = {
        {NULL, NULL, 0, NULL}
};

#ifdef NPY_1_19_API_VERSION
static void test_ufunc(char **args, const npy_intp *dimensions, const npy_intp* steps, void* data)
#else
static void test_ufunc(char **args, npy_intp *dimensions, npy_intp* steps, void* data)
#endif
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in0 = args[0];
    char *in1 = args[1];
    char *in2 = args[2];
    char *out0 = args[3];
    npy_intp in0_step = steps[0];
    npy_intp in1_step = steps[1];
    npy_intp in2_step = steps[2];
    npy_intp out0_step = steps[3];
    for (i = 0; i < n; i++) {
        *((double *)out0) = test(*(double *)in0, *(double *)in1, *(double *)in2);
        in0 += in0_step;
        in1 += in1_step;
        in2 += in2_step;
        out0 += out0_step;
    }
}
PyUFuncGenericFunction test_funcs[1] = {&test_ufunc};
static char test_types[4] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};
static void *test_data[1] = {NULL};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "wrapper_module_%(num)s",
    NULL,
    -1,
    wrapper_module_%(num)sMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_wrapper_module_%(num)s(void)
{
    PyObject *m, *d;
    PyObject *ufunc0;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    import_array();
    import_umath();
    d = PyModule_GetDict(m);
    ufunc0 = PyUFunc_FromFuncAndData(test_funcs, test_data, test_types, 1, 3, 1,
            PyUFunc_None, "wrapper_module_%(num)s", "Created in SymPy with Ufuncify", 0);
    PyDict_SetItemString(d, "test", ufunc0);
    Py_DECREF(ufunc0);
    return m;
}
#else
PyMODINIT_FUNC initwrapper_module_%(num)s(void)
{
    PyObject *m, *d;
    PyObject *ufunc0;
    m = Py_InitModule("wrapper_module_%(num)s", wrapper_module_%(num)sMethods);
    if (m == NULL) {
        return;
    }
    import_array();
    import_umath();
    d = PyModule_GetDict(m);
    ufunc0 = PyUFunc_FromFuncAndData(test_funcs, test_data, test_types, 1, 3, 1,
            PyUFunc_None, "wrapper_module_%(num)s", "Created in SymPy with Ufuncify", 0);
    PyDict_SetItemString(d, "test", ufunc0);
    Py_DECREF(ufunc0);
}
#endif""" % {'num': CodeWrapper._module_counter}
    assert source == expected


def test_ufuncify_source_multioutput():
    x, y, z = symbols('x,y,z')
    var_symbols = (x, y, z)
    expr = x + y**3 + 10*z**2
    code_wrapper = UfuncifyCodeWrapper(C99CodeGen("ufuncify"))
    routines = [make_routine("func{}".format(i), expr.diff(var_symbols[i]), var_symbols) for i in range(len(var_symbols))]
    source = get_string(code_wrapper.dump_c, routines, funcname='multitest')
    expected = """\
#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"
#include "file.h"

static PyMethodDef wrapper_module_%(num)sMethods[] = {
        {NULL, NULL, 0, NULL}
};

#ifdef NPY_1_19_API_VERSION
static void multitest_ufunc(char **args, const npy_intp *dimensions, const npy_intp* steps, void* data)
#else
static void multitest_ufunc(char **args, npy_intp *dimensions, npy_intp* steps, void* data)
#endif
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in0 = args[0];
    char *in1 = args[1];
    char *in2 = args[2];
    char *out0 = args[3];
    char *out1 = args[4];
    char *out2 = args[5];
    npy_intp in0_step = steps[0];
    npy_intp in1_step = steps[1];
    npy_intp in2_step = steps[2];
    npy_intp out0_step = steps[3];
    npy_intp out1_step = steps[4];
    npy_intp out2_step = steps[5];
    for (i = 0; i < n; i++) {
        *((double *)out0) = func0(*(double *)in0, *(double *)in1, *(double *)in2);
        *((double *)out1) = func1(*(double *)in0, *(double *)in1, *(double *)in2);
        *((double *)out2) = func2(*(double *)in0, *(double *)in1, *(double *)in2);
        in0 += in0_step;
        in1 += in1_step;
        in2 += in2_step;
        out0 += out0_step;
        out1 += out1_step;
        out2 += out2_step;
    }
}
PyUFuncGenericFunction multitest_funcs[1] = {&multitest_ufunc};
static char multitest_types[6] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};
static void *multitest_data[1] = {NULL};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "wrapper_module_%(num)s",
    NULL,
    -1,
    wrapper_module_%(num)sMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_wrapper_module_%(num)s(void)
{
    PyObject *m, *d;
    PyObject *ufunc0;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    import_array();
    import_umath();
    d = PyModule_GetDict(m);
    ufunc0 = PyUFunc_FromFuncAndData(multitest_funcs, multitest_data, multitest_types, 1, 3, 3,
            PyUFunc_None, "wrapper_module_%(num)s", "Created in SymPy with Ufuncify", 0);
    PyDict_SetItemString(d, "multitest", ufunc0);
    Py_DECREF(ufunc0);
    return m;
}
#else
PyMODINIT_FUNC initwrapper_module_%(num)s(void)
{
    PyObject *m, *d;
    PyObject *ufunc0;
    m = Py_InitModule("wrapper_module_%(num)s", wrapper_module_%(num)sMethods);
    if (m == NULL) {
        return;
    }
    import_array();
    import_umath();
    d = PyModule_GetDict(m);
    ufunc0 = PyUFunc_FromFuncAndData(multitest_funcs, multitest_data, multitest_types, 1, 3, 3,
            PyUFunc_None, "wrapper_module_%(num)s", "Created in SymPy with Ufuncify", 0);
    PyDict_SetItemString(d, "multitest", ufunc0);
    Py_DECREF(ufunc0);
}
#endif""" % {'num': CodeWrapper._module_counter}
    assert source == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\test_expressions.py ===
import operator
import re

import numpy as np
import pytest

from pandas import option_context
import pandas._testing as tm
from pandas.core.api import (
    DataFrame,
    Index,
    Series,
)
from pandas.core.computation import expressions as expr


@pytest.fixture
def _frame():
    return DataFrame(
        np.random.default_rng(2).standard_normal((10001, 4)),
        columns=list("ABCD"),
        dtype="float64",
    )


@pytest.fixture
def _frame2():
    return DataFrame(
        np.random.default_rng(2).standard_normal((100, 4)),
        columns=list("ABCD"),
        dtype="float64",
    )


@pytest.fixture
def _mixed(_frame):
    return DataFrame(
        {
            "A": _frame["A"].copy(),
            "B": _frame["B"].astype("float32"),
            "C": _frame["C"].astype("int64"),
            "D": _frame["D"].astype("int32"),
        }
    )


@pytest.fixture
def _mixed2(_frame2):
    return DataFrame(
        {
            "A": _frame2["A"].copy(),
            "B": _frame2["B"].astype("float32"),
            "C": _frame2["C"].astype("int64"),
            "D": _frame2["D"].astype("int32"),
        }
    )


@pytest.fixture
def _integer():
    return DataFrame(
        np.random.default_rng(2).integers(1, 100, size=(10001, 4)),
        columns=list("ABCD"),
        dtype="int64",
    )


@pytest.fixture
def _integer_integers(_integer):
    # integers to get a case with zeros
    return _integer * np.random.default_rng(2).integers(0, 2, size=np.shape(_integer))


@pytest.fixture
def _integer2():
    return DataFrame(
        np.random.default_rng(2).integers(1, 100, size=(101, 4)),
        columns=list("ABCD"),
        dtype="int64",
    )


@pytest.fixture
def _array(_frame):
    return _frame["A"].values.copy()


@pytest.fixture
def _array2(_frame2):
    return _frame2["A"].values.copy()


@pytest.fixture
def _array_mixed(_mixed):
    return _mixed["D"].values.copy()


@pytest.fixture
def _array_mixed2(_mixed2):
    return _mixed2["D"].values.copy()


@pytest.mark.skipif(not expr.USE_NUMEXPR, reason="not using numexpr")
class TestExpressions:
    @staticmethod
    def call_op(df, other, flex: bool, opname: str):
        if flex:
            op = lambda x, y: getattr(x, opname)(y)
            op.__name__ = opname
        else:
            op = getattr(operator, opname)

        with option_context("compute.use_numexpr", False):
            expected = op(df, other)

        expr.get_test_result()

        result = op(df, other)
        return result, expected

    @pytest.mark.parametrize(
        "fixture",
        [
            "_integer",
            "_integer2",
            "_integer_integers",
            "_frame",
            "_frame2",
            "_mixed",
            "_mixed2",
        ],
    )
    @pytest.mark.parametrize("flex", [True, False])
    @pytest.mark.parametrize(
        "arith", ["add", "sub", "mul", "mod", "truediv", "floordiv"]
    )
    def test_run_arithmetic(self, request, fixture, flex, arith, monkeypatch):
        df = request.getfixturevalue(fixture)
        with monkeypatch.context() as m:
            m.setattr(expr, "_MIN_ELEMENTS", 0)
            result, expected = self.call_op(df, df, flex, arith)

            if arith == "truediv":
                assert all(x.kind == "f" for x in expected.dtypes.values)
            tm.assert_equal(expected, result)

            for i in range(len(df.columns)):
                result, expected = self.call_op(
                    df.iloc[:, i], df.iloc[:, i], flex, arith
                )
                if arith == "truediv":
                    assert expected.dtype.kind == "f"
                tm.assert_equal(expected, result)

    @pytest.mark.parametrize(
        "fixture",
        [
            "_integer",
            "_integer2",
            "_integer_integers",
            "_frame",
            "_frame2",
            "_mixed",
            "_mixed2",
        ],
    )
    @pytest.mark.parametrize("flex", [True, False])
    def test_run_binary(self, request, fixture, flex, comparison_op, monkeypatch):
        """
        tests solely that the result is the same whether or not numexpr is
        enabled.  Need to test whether the function does the correct thing
        elsewhere.
        """
        df = request.getfixturevalue(fixture)
        arith = comparison_op.__name__
        with option_context("compute.use_numexpr", False):
            other = df.copy() + 1

        with monkeypatch.context() as m:
            m.setattr(expr, "_MIN_ELEMENTS", 0)
            expr.set_test_mode(True)

            result, expected = self.call_op(df, other, flex, arith)

            used_numexpr = expr.get_test_result()
            assert used_numexpr, "Did not use numexpr as expected."
            tm.assert_equal(expected, result)

            for i in range(len(df.columns)):
                binary_comp = other.iloc[:, i] + 1
                self.call_op(df.iloc[:, i], binary_comp, flex, "add")

    def test_invalid(self):
        array = np.random.default_rng(2).standard_normal(1_000_001)
        array2 = np.random.default_rng(2).standard_normal(100)

        # no op
        result = expr._can_use_numexpr(operator.add, None, array, array, "evaluate")
        assert not result

        # min elements
        result = expr._can_use_numexpr(operator.add, "+", array2, array2, "evaluate")
        assert not result

        # ok, we only check on first part of expression
        result = expr._can_use_numexpr(operator.add, "+", array, array2, "evaluate")
        assert result

    @pytest.mark.filterwarnings("ignore:invalid value encountered in:RuntimeWarning")
    @pytest.mark.parametrize(
        "opname,op_str",
        [("add", "+"), ("sub", "-"), ("mul", "*"), ("truediv", "/"), ("pow", "**")],
    )
    @pytest.mark.parametrize(
        "left_fix,right_fix", [("_array", "_array2"), ("_array_mixed", "_array_mixed2")]
    )
    def test_binary_ops(self, request, opname, op_str, left_fix, right_fix):
        left = request.getfixturevalue(left_fix)
        right = request.getfixturevalue(right_fix)

        def testit(left, right, opname, op_str):
            if opname == "pow":
                left = np.abs(left)

            op = getattr(operator, opname)

            # array has 0s
            result = expr.evaluate(op, left, left, use_numexpr=True)
            expected = expr.evaluate(op, left, left, use_numexpr=False)
            tm.assert_numpy_array_equal(result, expected)

            result = expr._can_use_numexpr(op, op_str, right, right, "evaluate")
            assert not result

        with option_context("compute.use_numexpr", False):
            testit(left, right, opname, op_str)

        expr.set_numexpr_threads(1)
        testit(left, right, opname, op_str)
        expr.set_numexpr_threads()
        testit(left, right, opname, op_str)

    @pytest.mark.parametrize(
        "left_fix,right_fix", [("_array", "_array2"), ("_array_mixed", "_array_mixed2")]
    )
    def test_comparison_ops(self, request, comparison_op, left_fix, right_fix):
        left = request.getfixturevalue(left_fix)
        right = request.getfixturevalue(right_fix)

        def testit():
            f12 = left + 1
            f22 = right + 1

            op = comparison_op

            result = expr.evaluate(op, left, f12, use_numexpr=True)
            expected = expr.evaluate(op, left, f12, use_numexpr=False)
            tm.assert_numpy_array_equal(result, expected)

            result = expr._can_use_numexpr(op, op, right, f22, "evaluate")
            assert not result

        with option_context("compute.use_numexpr", False):
            testit()

        expr.set_numexpr_threads(1)
        testit()
        expr.set_numexpr_threads()
        testit()

    @pytest.mark.parametrize("cond", [True, False])
    @pytest.mark.parametrize("fixture", ["_frame", "_frame2", "_mixed", "_mixed2"])
    def test_where(self, request, cond, fixture):
        df = request.getfixturevalue(fixture)

        def testit():
            c = np.empty(df.shape, dtype=np.bool_)
            c.fill(cond)
            result = expr.where(c, df.values, df.values + 1)
            expected = np.where(c, df.values, df.values + 1)
            tm.assert_numpy_array_equal(result, expected)

        with option_context("compute.use_numexpr", False):
            testit()

        expr.set_numexpr_threads(1)
        testit()
        expr.set_numexpr_threads()
        testit()

    @pytest.mark.parametrize(
        "op_str,opname", [("/", "truediv"), ("//", "floordiv"), ("**", "pow")]
    )
    def test_bool_ops_raise_on_arithmetic(self, op_str, opname):
        df = DataFrame(
            {
                "a": np.random.default_rng(2).random(10) > 0.5,
                "b": np.random.default_rng(2).random(10) > 0.5,
            }
        )

        msg = f"operator '{opname}' not implemented for bool dtypes"
        f = getattr(operator, opname)
        err_msg = re.escape(msg)

        with pytest.raises(NotImplementedError, match=err_msg):
            f(df, df)

        with pytest.raises(NotImplementedError, match=err_msg):
            f(df.a, df.b)

        with pytest.raises(NotImplementedError, match=err_msg):
            f(df.a, True)

        with pytest.raises(NotImplementedError, match=err_msg):
            f(False, df.a)

        with pytest.raises(NotImplementedError, match=err_msg):
            f(False, df)

        with pytest.raises(NotImplementedError, match=err_msg):
            f(df, True)

    @pytest.mark.parametrize(
        "op_str,opname", [("+", "add"), ("*", "mul"), ("-", "sub")]
    )
    def test_bool_ops_warn_on_arithmetic(self, op_str, opname):
        n = 10
        df = DataFrame(
            {
                "a": np.random.default_rng(2).random(n) > 0.5,
                "b": np.random.default_rng(2).random(n) > 0.5,
            }
        )

        subs = {"+": "|", "*": "&", "-": "^"}
        sub_funcs = {"|": "or_", "&": "and_", "^": "xor"}

        f = getattr(operator, opname)
        fe = getattr(operator, sub_funcs[subs[op_str]])

        if op_str == "-":
            # raises TypeError
            return

        with tm.use_numexpr(True, min_elements=5):
            with tm.assert_produces_warning():
                r = f(df, df)
                e = fe(df, df)
                tm.assert_frame_equal(r, e)

            with tm.assert_produces_warning():
                r = f(df.a, df.b)
                e = fe(df.a, df.b)
                tm.assert_series_equal(r, e)

            with tm.assert_produces_warning():
                r = f(df.a, True)
                e = fe(df.a, True)
                tm.assert_series_equal(r, e)

            with tm.assert_produces_warning():
                r = f(False, df.a)
                e = fe(False, df.a)
                tm.assert_series_equal(r, e)

            with tm.assert_produces_warning():
                r = f(False, df)
                e = fe(False, df)
                tm.assert_frame_equal(r, e)

            with tm.assert_produces_warning():
                r = f(df, True)
                e = fe(df, True)
                tm.assert_frame_equal(r, e)

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (
                DataFrame(
                    [[0, 1, 2, "aa"], [0, 1, 2, "aa"]], columns=["a", "b", "c", "dtype"]
                ),
                DataFrame([[False, False], [False, False]], columns=["a", "dtype"]),
            ),
            (
                DataFrame(
                    [[0, 3, 2, "aa"], [0, 4, 2, "aa"], [0, 1, 1, "bb"]],
                    columns=["a", "b", "c", "dtype"],
                ),
                DataFrame(
                    [[False, False], [False, False], [False, False]],
                    columns=["a", "dtype"],
                ),
            ),
        ],
    )
    def test_bool_ops_column_name_dtype(self, test_input, expected):
        # GH 22383 - .ne fails if columns containing column name 'dtype'
        result = test_input.loc[:, ["a", "dtype"]].ne(test_input.loc[:, ["a", "dtype"]])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "arith", ("add", "sub", "mul", "mod", "truediv", "floordiv")
    )
    @pytest.mark.parametrize("axis", (0, 1))
    def test_frame_series_axis(self, axis, arith, _frame, monkeypatch):
        # GH#26736 Dataframe.floordiv(Series, axis=1) fails

        df = _frame
        if axis == 1:
            other = df.iloc[0, :]
        else:
            other = df.iloc[:, 0]

        with monkeypatch.context() as m:
            m.setattr(expr, "_MIN_ELEMENTS", 0)

            op_func = getattr(df, arith)

            with option_context("compute.use_numexpr", False):
                expected = op_func(other, axis=axis)

            result = op_func(other, axis=axis)
            tm.assert_frame_equal(expected, result)

    @pytest.mark.parametrize(
        "op",
        [
            "__mod__",
            "__rmod__",
            "__floordiv__",
            "__rfloordiv__",
        ],
    )
    @pytest.mark.parametrize("box", [DataFrame, Series, Index])
    @pytest.mark.parametrize("scalar", [-5, 5])
    def test_python_semantics_with_numexpr_installed(
        self, op, box, scalar, monkeypatch
    ):
        # https://github.com/pandas-dev/pandas/issues/36047
        with monkeypatch.context() as m:
            m.setattr(expr, "_MIN_ELEMENTS", 0)
            data = np.arange(-50, 50)
            obj = box(data)
            method = getattr(obj, op)
            result = method(scalar)

            # compare result with numpy
            with option_context("compute.use_numexpr", False):
                expected = method(scalar)

            tm.assert_equal(result, expected)

            # compare result element-wise with Python
            for i, elem in enumerate(data):
                if box == DataFrame:
                    scalar_result = result.iloc[i, 0]
                else:
                    scalar_result = result[i]
                try:
                    expected = getattr(int(elem), op)(scalar)
                except ZeroDivisionError:
                    pass
                else:
                    assert scalar_result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\test_partial_slicing.py ===
""" test partial slicing on Series/Frame """

from datetime import datetime

import numpy as np
import pytest

from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestSlicing:
    def test_string_index_series_name_converted(self):
        # GH#1644
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            index=date_range("1/1/2000", periods=10),
        )

        result = df.loc["1/3/2000"]
        assert result.name == df.index[2]

        result = df.T["1/3/2000"]
        assert result.name == df.index[2]

    def test_stringified_slice_with_tz(self):
        # GH#2658
        start = "2013-01-07"
        idx = date_range(start=start, freq="1d", periods=10, tz="US/Eastern")
        df = DataFrame(np.arange(10), index=idx)
        df["2013-01-14 23:44:34.437768-05:00":]  # no exception here

    def test_return_type_doesnt_depend_on_monotonicity(self):
        # GH#24892 we get Series back regardless of whether our DTI is monotonic
        dti = date_range(start="2015-5-13 23:59:00", freq="min", periods=3)
        ser = Series(range(3), index=dti)

        # non-monotonic index
        ser2 = Series(range(3), index=[dti[1], dti[0], dti[2]])

        # key with resolution strictly lower than "min"
        key = "2015-5-14 00"

        # monotonic increasing index
        result = ser.loc[key]
        expected = ser.iloc[1:]
        tm.assert_series_equal(result, expected)

        # monotonic decreasing index
        result = ser.iloc[::-1].loc[key]
        expected = ser.iloc[::-1][:-1]
        tm.assert_series_equal(result, expected)

        # non-monotonic index
        result2 = ser2.loc[key]
        expected2 = ser2.iloc[::2]
        tm.assert_series_equal(result2, expected2)

    def test_return_type_doesnt_depend_on_monotonicity_higher_reso(self):
        # GH#24892 we get Series back regardless of whether our DTI is monotonic
        dti = date_range(start="2015-5-13 23:59:00", freq="min", periods=3)
        ser = Series(range(3), index=dti)

        # non-monotonic index
        ser2 = Series(range(3), index=[dti[1], dti[0], dti[2]])

        # key with resolution strictly *higher) than "min"
        key = "2015-5-14 00:00:00"

        # monotonic increasing index
        result = ser.loc[key]
        assert result == 1

        # monotonic decreasing index
        result = ser.iloc[::-1].loc[key]
        assert result == 1

        # non-monotonic index
        result2 = ser2.loc[key]
        assert result2 == 0

    def test_monotone_DTI_indexing_bug(self):
        # GH 19362
        # Testing accessing the first element in a monotonic descending
        # partial string indexing.

        df = DataFrame(list(range(5)))
        date_list = [
            "2018-01-02",
            "2017-02-10",
            "2016-03-10",
            "2015-03-15",
            "2014-03-16",
        ]
        date_index = DatetimeIndex(date_list)
        df["date"] = date_index
        expected = DataFrame({0: list(range(5)), "date": date_index})
        tm.assert_frame_equal(df, expected)

        # We get a slice because df.index's resolution is hourly and we
        #  are slicing with a daily-resolution string.  If both were daily,
        #  we would get a single item back
        dti = date_range("20170101 01:00:00", periods=3)
        df = DataFrame({"A": [1, 2, 3]}, index=dti[::-1])

        expected = DataFrame({"A": 1}, index=dti[-1:][::-1])
        result = df.loc["2017-01-03"]
        tm.assert_frame_equal(result, expected)

        result2 = df.iloc[::-1].loc["2017-01-03"]
        expected2 = expected.iloc[::-1]
        tm.assert_frame_equal(result2, expected2)

    def test_slice_year(self):
        dti = date_range(freq="B", start=datetime(2005, 1, 1), periods=500)

        s = Series(np.arange(len(dti)), index=dti)
        result = s["2005"]
        expected = s[s.index.year == 2005]
        tm.assert_series_equal(result, expected)

        df = DataFrame(np.random.default_rng(2).random((len(dti), 5)), index=dti)
        result = df.loc["2005"]
        expected = df[df.index.year == 2005]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "partial_dtime",
        [
            "2019",
            "2019Q4",
            "Dec 2019",
            "2019-12-31",
            "2019-12-31 23",
            "2019-12-31 23:59",
        ],
    )
    def test_slice_end_of_period_resolution(self, partial_dtime):
        # GH#31064
        dti = date_range("2019-12-31 23:59:55.999999999", periods=10, freq="s")

        ser = Series(range(10), index=dti)
        result = ser[partial_dtime]
        expected = ser.iloc[:5]
        tm.assert_series_equal(result, expected)

    def test_slice_quarter(self):
        dti = date_range(freq="D", start=datetime(2000, 6, 1), periods=500)

        s = Series(np.arange(len(dti)), index=dti)
        assert len(s["2001Q1"]) == 90

        df = DataFrame(np.random.default_rng(2).random((len(dti), 5)), index=dti)
        assert len(df.loc["1Q01"]) == 90

    def test_slice_month(self):
        dti = date_range(freq="D", start=datetime(2005, 1, 1), periods=500)
        s = Series(np.arange(len(dti)), index=dti)
        assert len(s["2005-11"]) == 30

        df = DataFrame(np.random.default_rng(2).random((len(dti), 5)), index=dti)
        assert len(df.loc["2005-11"]) == 30

        tm.assert_series_equal(s["2005-11"], s["11-2005"])

    def test_partial_slice(self):
        rng = date_range(freq="D", start=datetime(2005, 1, 1), periods=500)
        s = Series(np.arange(len(rng)), index=rng)

        result = s["2005-05":"2006-02"]
        expected = s["20050501":"20060228"]
        tm.assert_series_equal(result, expected)

        result = s["2005-05":]
        expected = s["20050501":]
        tm.assert_series_equal(result, expected)

        result = s[:"2006-02"]
        expected = s[:"20060228"]
        tm.assert_series_equal(result, expected)

        result = s["2005-1-1"]
        assert result == s.iloc[0]

        with pytest.raises(KeyError, match=r"^'2004-12-31'$"):
            s["2004-12-31"]

    def test_partial_slice_daily(self):
        rng = date_range(freq="h", start=datetime(2005, 1, 31), periods=500)
        s = Series(np.arange(len(rng)), index=rng)

        result = s["2005-1-31"]
        tm.assert_series_equal(result, s.iloc[:24])

        with pytest.raises(KeyError, match=r"^'2004-12-31 00'$"):
            s["2004-12-31 00"]

    def test_partial_slice_hourly(self):
        rng = date_range(freq="min", start=datetime(2005, 1, 1, 20, 0, 0), periods=500)
        s = Series(np.arange(len(rng)), index=rng)

        result = s["2005-1-1"]
        tm.assert_series_equal(result, s.iloc[: 60 * 4])

        result = s["2005-1-1 20"]
        tm.assert_series_equal(result, s.iloc[:60])

        assert s["2005-1-1 20:00"] == s.iloc[0]
        with pytest.raises(KeyError, match=r"^'2004-12-31 00:15'$"):
            s["2004-12-31 00:15"]

    def test_partial_slice_minutely(self):
        rng = date_range(freq="s", start=datetime(2005, 1, 1, 23, 59, 0), periods=500)
        s = Series(np.arange(len(rng)), index=rng)

        result = s["2005-1-1 23:59"]
        tm.assert_series_equal(result, s.iloc[:60])

        result = s["2005-1-1"]
        tm.assert_series_equal(result, s.iloc[:60])

        assert s[Timestamp("2005-1-1 23:59:00")] == s.iloc[0]
        with pytest.raises(KeyError, match=r"^'2004-12-31 00:00:00'$"):
            s["2004-12-31 00:00:00"]

    def test_partial_slice_second_precision(self):
        rng = date_range(
            start=datetime(2005, 1, 1, 0, 0, 59, microsecond=999990),
            periods=20,
            freq="us",
        )
        s = Series(np.arange(20), rng)

        tm.assert_series_equal(s["2005-1-1 00:00"], s.iloc[:10])
        tm.assert_series_equal(s["2005-1-1 00:00:59"], s.iloc[:10])

        tm.assert_series_equal(s["2005-1-1 00:01"], s.iloc[10:])
        tm.assert_series_equal(s["2005-1-1 00:01:00"], s.iloc[10:])

        assert s[Timestamp("2005-1-1 00:00:59.999990")] == s.iloc[0]
        with pytest.raises(KeyError, match="2005-1-1 00:00:00"):
            s["2005-1-1 00:00:00"]

    def test_partial_slicing_dataframe(self):
        # GH14856
        # Test various combinations of string slicing resolution vs.
        # index resolution
        # - If string resolution is less precise than index resolution,
        # string is considered a slice
        # - If string resolution is equal to or more precise than index
        # resolution, string is considered an exact match
        formats = [
            "%Y",
            "%Y-%m",
            "%Y-%m-%d",
            "%Y-%m-%d %H",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
        ]
        resolutions = ["year", "month", "day", "hour", "minute", "second"]
        for rnum, resolution in enumerate(resolutions[2:], 2):
            # we check only 'day', 'hour', 'minute' and 'second'
            unit = Timedelta("1 " + resolution)
            middate = datetime(2012, 1, 1, 0, 0, 0)
            index = DatetimeIndex([middate - unit, middate, middate + unit])
            values = [1, 2, 3]
            df = DataFrame({"a": values}, index, dtype=np.int64)
            assert df.index.resolution == resolution

            # Timestamp with the same resolution as index
            # Should be exact match for Series (return scalar)
            # and raise KeyError for Frame
            for timestamp, expected in zip(index, values):
                ts_string = timestamp.strftime(formats[rnum])
                # make ts_string as precise as index
                result = df["a"][ts_string]
                assert isinstance(result, np.int64)
                assert result == expected
                msg = rf"^'{ts_string}'$"
                with pytest.raises(KeyError, match=msg):
                    df[ts_string]

            # Timestamp with resolution less precise than index
            for fmt in formats[:rnum]:
                for element, theslice in [[0, slice(None, 1)], [1, slice(1, None)]]:
                    ts_string = index[element].strftime(fmt)

                    # Series should return slice
                    result = df["a"][ts_string]
                    expected = df["a"][theslice]
                    tm.assert_series_equal(result, expected)

                    # pre-2.0 df[ts_string] was overloaded to interpret this
                    #  as slicing along index
                    with pytest.raises(KeyError, match=ts_string):
                        df[ts_string]

            # Timestamp with resolution more precise than index
            # Compatible with existing key
            # Should return scalar for Series
            # and raise KeyError for Frame
            for fmt in formats[rnum + 1 :]:
                ts_string = index[1].strftime(fmt)
                result = df["a"][ts_string]
                assert isinstance(result, np.int64)
                assert result == 2
                msg = rf"^'{ts_string}'$"
                with pytest.raises(KeyError, match=msg):
                    df[ts_string]

            # Not compatible with existing key
            # Should raise KeyError
            for fmt, res in list(zip(formats, resolutions))[rnum + 1 :]:
                ts = index[1] + Timedelta("1 " + res)
                ts_string = ts.strftime(fmt)
                msg = rf"^'{ts_string}'$"
                with pytest.raises(KeyError, match=msg):
                    df["a"][ts_string]
                with pytest.raises(KeyError, match=msg):
                    df[ts_string]

    def test_partial_slicing_with_multiindex(self):
        # GH 4758
        # partial string indexing with a multi-index buggy
        df = DataFrame(
            {
                "ACCOUNT": ["ACCT1", "ACCT1", "ACCT1", "ACCT2"],
                "TICKER": ["ABC", "MNP", "XYZ", "XYZ"],
                "val": [1, 2, 3, 4],
            },
            index=date_range("2013-06-19 09:30:00", periods=4, freq="5min"),
        )
        df_multi = df.set_index(["ACCOUNT", "TICKER"], append=True)

        expected = DataFrame(
            [[1]], index=Index(["ABC"], name="TICKER"), columns=["val"]
        )
        result = df_multi.loc[("2013-06-19 09:30:00", "ACCT1")]
        tm.assert_frame_equal(result, expected)

        expected = df_multi.loc[
            (Timestamp("2013-06-19 09:30:00", tz=None), "ACCT1", "ABC")
        ]
        result = df_multi.loc[("2013-06-19 09:30:00", "ACCT1", "ABC")]
        tm.assert_series_equal(result, expected)

        # partial string indexing on first level, scalar indexing on the other two
        result = df_multi.loc[("2013-06-19", "ACCT1", "ABC")]
        expected = df_multi.iloc[:1].droplevel([1, 2])
        tm.assert_frame_equal(result, expected)

    def test_partial_slicing_with_multiindex_series(self):
        # GH 4294
        # partial slice on a series mi
        ser = Series(
            range(250),
            index=MultiIndex.from_product(
                [date_range("2000-1-1", periods=50), range(5)]
            ),
        )

        s2 = ser[:-1].copy()
        expected = s2["2000-1-4"]
        result = s2[Timestamp("2000-1-4")]
        tm.assert_series_equal(result, expected)

        result = ser[Timestamp("2000-1-4")]
        expected = ser["2000-1-4"]
        tm.assert_series_equal(result, expected)

        df2 = DataFrame(ser)
        expected = df2.xs("2000-1-4")
        result = df2.loc[Timestamp("2000-1-4")]
        tm.assert_frame_equal(result, expected)

    def test_partial_slice_requires_monotonicity(self):
        # Disallowed since 2.0 (GH 37819)
        ser = Series(np.arange(10), date_range("2014-01-01", periods=10))

        nonmonotonic = ser.iloc[[3, 5, 4]]
        timestamp = Timestamp("2014-01-10")
        with pytest.raises(
            KeyError, match="Value based partial slicing on non-monotonic"
        ):
            nonmonotonic["2014-01-10":]

        with pytest.raises(KeyError, match=r"Timestamp\('2014-01-10 00:00:00'\)"):
            nonmonotonic[timestamp:]

        with pytest.raises(
            KeyError, match="Value based partial slicing on non-monotonic"
        ):
            nonmonotonic.loc["2014-01-10":]

        with pytest.raises(KeyError, match=r"Timestamp\('2014-01-10 00:00:00'\)"):
            nonmonotonic.loc[timestamp:]

    def test_loc_datetime_length_one(self):
        # GH16071
        df = DataFrame(
            columns=["1"],
            index=date_range("2016-10-01T00:00:00", "2016-10-01T23:59:59"),
        )
        result = df.loc[datetime(2016, 10, 1) :]
        tm.assert_frame_equal(result, df)

        result = df.loc["2016-10-01T00:00:00":]
        tm.assert_frame_equal(result, df)

    @pytest.mark.parametrize(
        "start",
        [
            "2018-12-02 21:50:00+00:00",
            Timestamp("2018-12-02 21:50:00+00:00"),
            Timestamp("2018-12-02 21:50:00+00:00").to_pydatetime(),
        ],
    )
    @pytest.mark.parametrize(
        "end",
        [
            "2018-12-02 21:52:00+00:00",
            Timestamp("2018-12-02 21:52:00+00:00"),
            Timestamp("2018-12-02 21:52:00+00:00").to_pydatetime(),
        ],
    )
    def test_getitem_with_datestring_with_UTC_offset(self, start, end):
        # GH 24076
        idx = date_range(
            start="2018-12-02 14:50:00-07:00",
            end="2018-12-02 14:50:00-07:00",
            freq="1min",
        )
        df = DataFrame(1, index=idx, columns=["A"])
        result = df[start:end]
        expected = df.iloc[0:3, :]
        tm.assert_frame_equal(result, expected)

        # GH 16785
        start = str(start)
        end = str(end)
        with pytest.raises(ValueError, match="Both dates must"):
            df[start : end[:-4] + "1:00"]

        with pytest.raises(ValueError, match="The index must be timezone"):
            df = df.tz_localize(None)
            df[start:end]

    def test_slice_reduce_to_series(self):
        # GH 27516
        df = DataFrame(
            {"A": range(24)}, index=date_range("2000", periods=24, freq="ME")
        )
        expected = Series(
            range(12), index=date_range("2000", periods=12, freq="ME"), name="A"
        )
        result = df.loc["2000", "A"]
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\window\test_numba.py ===
import numpy as np
import pytest

from pandas.compat import is_platform_arm
from pandas.errors import NumbaUtilError
import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    Series,
    option_context,
    to_datetime,
)
import pandas._testing as tm
from pandas.util.version import Version

pytestmark = [pytest.mark.single_cpu]

numba = pytest.importorskip("numba")
pytestmark.append(
    pytest.mark.skipif(
        Version(numba.__version__) == Version("0.61") and is_platform_arm(),
        reason=f"Segfaults on ARM platforms with numba {numba.__version__}",
    )
)


@pytest.fixture(params=["single", "table"])
def method(request):
    """method keyword in rolling/expanding/ewm constructor"""
    return request.param


@pytest.fixture(
    params=[
        ["sum", {}],
        ["mean", {}],
        ["median", {}],
        ["max", {}],
        ["min", {}],
        ["var", {}],
        ["var", {"ddof": 0}],
        ["std", {}],
        ["std", {"ddof": 0}],
    ]
)
def arithmetic_numba_supported_operators(request):
    return request.param


@td.skip_if_no("numba")
@pytest.mark.filterwarnings("ignore")
# Filter warnings when parallel=True and the function can't be parallelized by Numba
class TestEngine:
    @pytest.mark.parametrize("jit", [True, False])
    def test_numba_vs_cython_apply(self, jit, nogil, parallel, nopython, center, step):
        def f(x, *args):
            arg_sum = 0
            for arg in args:
                arg_sum += arg
            return np.mean(x) + arg_sum

        if jit:
            import numba

            f = numba.jit(f)

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
        args = (2,)

        s = Series(range(10))
        result = s.rolling(2, center=center, step=step).apply(
            f, args=args, engine="numba", engine_kwargs=engine_kwargs, raw=True
        )
        expected = s.rolling(2, center=center, step=step).apply(
            f, engine="cython", args=args, raw=True
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "data",
        [
            DataFrame(np.eye(5)),
            DataFrame(
                [
                    [5, 7, 7, 7, np.nan, np.inf, 4, 3, 3, 3],
                    [5, 7, 7, 7, np.nan, np.inf, 7, 3, 3, 3],
                    [np.nan, np.nan, 5, 6, 7, 5, 5, 5, 5, 5],
                ]
            ).T,
            Series(range(5), name="foo"),
            Series([20, 10, 10, np.inf, 1, 1, 2, 3]),
            Series([20, 10, 10, np.nan, 10, 1, 2, 3]),
        ],
    )
    def test_numba_vs_cython_rolling_methods(
        self,
        data,
        nogil,
        parallel,
        nopython,
        arithmetic_numba_supported_operators,
        step,
    ):
        method, kwargs = arithmetic_numba_supported_operators

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        roll = data.rolling(3, step=step)
        result = getattr(roll, method)(
            engine="numba", engine_kwargs=engine_kwargs, **kwargs
        )
        expected = getattr(roll, method)(engine="cython", **kwargs)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "data", [DataFrame(np.eye(5)), Series(range(5), name="foo")]
    )
    def test_numba_vs_cython_expanding_methods(
        self, data, nogil, parallel, nopython, arithmetic_numba_supported_operators
    ):
        method, kwargs = arithmetic_numba_supported_operators

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        data = DataFrame(np.eye(5))
        expand = data.expanding()
        result = getattr(expand, method)(
            engine="numba", engine_kwargs=engine_kwargs, **kwargs
        )
        expected = getattr(expand, method)(engine="cython", **kwargs)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("jit", [True, False])
    def test_cache_apply(self, jit, nogil, parallel, nopython, step):
        # Test that the functions are cached correctly if we switch functions
        def func_1(x):
            return np.mean(x) + 4

        def func_2(x):
            return np.std(x) * 5

        if jit:
            import numba

            func_1 = numba.jit(func_1)
            func_2 = numba.jit(func_2)

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        roll = Series(range(10)).rolling(2, step=step)
        result = roll.apply(
            func_1, engine="numba", engine_kwargs=engine_kwargs, raw=True
        )
        expected = roll.apply(func_1, engine="cython", raw=True)
        tm.assert_series_equal(result, expected)

        result = roll.apply(
            func_2, engine="numba", engine_kwargs=engine_kwargs, raw=True
        )
        expected = roll.apply(func_2, engine="cython", raw=True)
        tm.assert_series_equal(result, expected)
        # This run should use the cached func_1
        result = roll.apply(
            func_1, engine="numba", engine_kwargs=engine_kwargs, raw=True
        )
        expected = roll.apply(func_1, engine="cython", raw=True)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "window,window_kwargs",
        [
            ["rolling", {"window": 3, "min_periods": 0}],
            ["expanding", {}],
        ],
    )
    def test_dont_cache_args(
        self, window, window_kwargs, nogil, parallel, nopython, method
    ):
        # GH 42287

        def add(values, x):
            return np.sum(values) + x

        engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}
        df = DataFrame({"value": [0, 0, 0]})
        result = getattr(df, window)(method=method, **window_kwargs).apply(
            add, raw=True, engine="numba", engine_kwargs=engine_kwargs, args=(1,)
        )
        expected = DataFrame({"value": [1.0, 1.0, 1.0]})
        tm.assert_frame_equal(result, expected)

        result = getattr(df, window)(method=method, **window_kwargs).apply(
            add, raw=True, engine="numba", engine_kwargs=engine_kwargs, args=(2,)
        )
        expected = DataFrame({"value": [2.0, 2.0, 2.0]})
        tm.assert_frame_equal(result, expected)

    def test_dont_cache_engine_kwargs(self):
        # If the user passes a different set of engine_kwargs don't return the same
        # jitted function
        nogil = False
        parallel = True
        nopython = True

        def func(x):
            return nogil + parallel + nopython

        engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}
        df = DataFrame({"value": [0, 0, 0]})
        result = df.rolling(1).apply(
            func, raw=True, engine="numba", engine_kwargs=engine_kwargs
        )
        expected = DataFrame({"value": [2.0, 2.0, 2.0]})
        tm.assert_frame_equal(result, expected)

        parallel = False
        engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}
        result = df.rolling(1).apply(
            func, raw=True, engine="numba", engine_kwargs=engine_kwargs
        )
        expected = DataFrame({"value": [1.0, 1.0, 1.0]})
        tm.assert_frame_equal(result, expected)


@td.skip_if_no("numba")
class TestEWM:
    @pytest.mark.parametrize(
        "grouper", [lambda x: x, lambda x: x.groupby("A")], ids=["None", "groupby"]
    )
    @pytest.mark.parametrize("method", ["mean", "sum"])
    def test_invalid_engine(self, grouper, method):
        df = DataFrame({"A": ["a", "b", "a", "b"], "B": range(4)})
        with pytest.raises(ValueError, match="engine must be either"):
            getattr(grouper(df).ewm(com=1.0), method)(engine="foo")

    @pytest.mark.parametrize(
        "grouper", [lambda x: x, lambda x: x.groupby("A")], ids=["None", "groupby"]
    )
    @pytest.mark.parametrize("method", ["mean", "sum"])
    def test_invalid_engine_kwargs(self, grouper, method):
        df = DataFrame({"A": ["a", "b", "a", "b"], "B": range(4)})
        with pytest.raises(ValueError, match="cython engine does not"):
            getattr(grouper(df).ewm(com=1.0), method)(
                engine="cython", engine_kwargs={"nopython": True}
            )

    @pytest.mark.parametrize("grouper", ["None", "groupby"])
    @pytest.mark.parametrize("method", ["mean", "sum"])
    def test_cython_vs_numba(
        self, grouper, method, nogil, parallel, nopython, ignore_na, adjust
    ):
        df = DataFrame({"B": range(4)})
        if grouper == "None":
            grouper = lambda x: x
        else:
            df["A"] = ["a", "b", "a", "b"]
            grouper = lambda x: x.groupby("A")
        if method == "sum":
            adjust = True
        ewm = grouper(df).ewm(com=1.0, adjust=adjust, ignore_na=ignore_na)

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
        result = getattr(ewm, method)(engine="numba", engine_kwargs=engine_kwargs)
        expected = getattr(ewm, method)(engine="cython")

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("grouper", ["None", "groupby"])
    def test_cython_vs_numba_times(self, grouper, nogil, parallel, nopython, ignore_na):
        # GH 40951

        df = DataFrame({"B": [0, 0, 1, 1, 2, 2]})
        if grouper == "None":
            grouper = lambda x: x
        else:
            grouper = lambda x: x.groupby("A")
            df["A"] = ["a", "b", "a", "b", "b", "a"]

        halflife = "23 days"
        times = to_datetime(
            [
                "2020-01-01",
                "2020-01-01",
                "2020-01-02",
                "2020-01-10",
                "2020-02-23",
                "2020-01-03",
            ]
        )
        ewm = grouper(df).ewm(
            halflife=halflife, adjust=True, ignore_na=ignore_na, times=times
        )

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        result = ewm.mean(engine="numba", engine_kwargs=engine_kwargs)
        expected = ewm.mean(engine="cython")

        tm.assert_frame_equal(result, expected)


@td.skip_if_no("numba")
def test_use_global_config():
    def f(x):
        return np.mean(x) + 2

    s = Series(range(10))
    with option_context("compute.use_numba", True):
        result = s.rolling(2).apply(f, engine=None, raw=True)
    expected = s.rolling(2).apply(f, engine="numba", raw=True)
    tm.assert_series_equal(expected, result)


@td.skip_if_no("numba")
def test_invalid_kwargs_nopython():
    with pytest.raises(NumbaUtilError, match="numba does not support kwargs with"):
        Series(range(1)).rolling(1).apply(
            lambda x: x, kwargs={"a": 1}, engine="numba", raw=True
        )


@td.skip_if_no("numba")
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore")
# Filter warnings when parallel=True and the function can't be parallelized by Numba
class TestTableMethod:
    def test_table_series_valueerror(self):
        def f(x):
            return np.sum(x, axis=0) + 1

        with pytest.raises(
            ValueError, match="method='table' not applicable for Series objects."
        ):
            Series(range(1)).rolling(1, method="table").apply(
                f, engine="numba", raw=True
            )

    def test_table_method_rolling_methods(
        self,
        axis,
        nogil,
        parallel,
        nopython,
        arithmetic_numba_supported_operators,
        step,
    ):
        method, kwargs = arithmetic_numba_supported_operators

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        df = DataFrame(np.eye(3))
        roll_table = df.rolling(2, method="table", axis=axis, min_periods=0, step=step)
        if method in ("var", "std"):
            with pytest.raises(NotImplementedError, match=f"{method} not supported"):
                getattr(roll_table, method)(
                    engine_kwargs=engine_kwargs, engine="numba", **kwargs
                )
        else:
            roll_single = df.rolling(
                2, method="single", axis=axis, min_periods=0, step=step
            )
            result = getattr(roll_table, method)(
                engine_kwargs=engine_kwargs, engine="numba", **kwargs
            )
            expected = getattr(roll_single, method)(
                engine_kwargs=engine_kwargs, engine="numba", **kwargs
            )
            tm.assert_frame_equal(result, expected)

    def test_table_method_rolling_apply(self, axis, nogil, parallel, nopython, step):
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        def f(x):
            return np.sum(x, axis=0) + 1

        df = DataFrame(np.eye(3))
        result = df.rolling(
            2, method="table", axis=axis, min_periods=0, step=step
        ).apply(f, raw=True, engine_kwargs=engine_kwargs, engine="numba")
        expected = df.rolling(
            2, method="single", axis=axis, min_periods=0, step=step
        ).apply(f, raw=True, engine_kwargs=engine_kwargs, engine="numba")
        tm.assert_frame_equal(result, expected)

    def test_table_method_rolling_weighted_mean(self, step):
        def weighted_mean(x):
            arr = np.ones((1, x.shape[1]))
            arr[:, :2] = (x[:, :2] * x[:, 2]).sum(axis=0) / x[:, 2].sum()
            return arr

        df = DataFrame([[1, 2, 0.6], [2, 3, 0.4], [3, 4, 0.2], [4, 5, 0.7]])
        result = df.rolling(2, method="table", min_periods=0, step=step).apply(
            weighted_mean, raw=True, engine="numba"
        )
        expected = DataFrame(
            [
                [1.0, 2.0, 1.0],
                [1.8, 2.0, 1.0],
                [3.333333, 2.333333, 1.0],
                [1.555556, 7, 1.0],
            ]
        )[::step]
        tm.assert_frame_equal(result, expected)

    def test_table_method_expanding_apply(self, axis, nogil, parallel, nopython):
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        def f(x):
            return np.sum(x, axis=0) + 1

        df = DataFrame(np.eye(3))
        result = df.expanding(method="table", axis=axis).apply(
            f, raw=True, engine_kwargs=engine_kwargs, engine="numba"
        )
        expected = df.expanding(method="single", axis=axis).apply(
            f, raw=True, engine_kwargs=engine_kwargs, engine="numba"
        )
        tm.assert_frame_equal(result, expected)

    def test_table_method_expanding_methods(
        self, axis, nogil, parallel, nopython, arithmetic_numba_supported_operators
    ):
        method, kwargs = arithmetic_numba_supported_operators

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        df = DataFrame(np.eye(3))
        expand_table = df.expanding(method="table", axis=axis)
        if method in ("var", "std"):
            with pytest.raises(NotImplementedError, match=f"{method} not supported"):
                getattr(expand_table, method)(
                    engine_kwargs=engine_kwargs, engine="numba", **kwargs
                )
        else:
            expand_single = df.expanding(method="single", axis=axis)
            result = getattr(expand_table, method)(
                engine_kwargs=engine_kwargs, engine="numba", **kwargs
            )
            expected = getattr(expand_single, method)(
                engine_kwargs=engine_kwargs, engine="numba", **kwargs
            )
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("data", [np.eye(3), np.ones((2, 3)), np.ones((3, 2))])
    @pytest.mark.parametrize("method", ["mean", "sum"])
    def test_table_method_ewm(self, data, method, axis, nogil, parallel, nopython):
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        df = DataFrame(data)

        result = getattr(df.ewm(com=1, method="table", axis=axis), method)(
            engine_kwargs=engine_kwargs, engine="numba"
        )
        expected = getattr(df.ewm(com=1, method="single", axis=axis), method)(
            engine_kwargs=engine_kwargs, engine="numba"
        )
        tm.assert_frame_equal(result, expected)


@td.skip_if_no("numba")
def test_npfunc_no_warnings():
    df = DataFrame({"col1": [1, 2, 3, 4, 5]})
    with tm.assert_produces_warning(False):
        df.col1.rolling(2).apply(np.prod, raw=True, engine="numba")