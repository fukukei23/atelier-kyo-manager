
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\apply\test_str.py ===
from itertools import chain
import operator

import numpy as np
import pytest

from pandas.core.dtypes.common import is_number

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm
from pandas.tests.apply.common import (
    frame_transform_kernels,
    series_transform_kernels,
)


@pytest.mark.parametrize("func", ["sum", "mean", "min", "max", "std"])
@pytest.mark.parametrize(
    "args,kwds",
    [
        pytest.param([], {}, id="no_args_or_kwds"),
        pytest.param([1], {}, id="axis_from_args"),
        pytest.param([], {"axis": 1}, id="axis_from_kwds"),
        pytest.param([], {"numeric_only": True}, id="optional_kwds"),
        pytest.param([1, True], {"numeric_only": True}, id="args_and_kwds"),
    ],
)
@pytest.mark.parametrize("how", ["agg", "apply"])
def test_apply_with_string_funcs(request, float_frame, func, args, kwds, how):
    if len(args) > 1 and how == "agg":
        request.applymarker(
            pytest.mark.xfail(
                raises=TypeError,
                reason="agg/apply signature mismatch - agg passes 2nd "
                "argument to func",
            )
        )
    result = getattr(float_frame, how)(func, *args, **kwds)
    expected = getattr(float_frame, func)(*args, **kwds)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("arg", ["sum", "mean", "min", "max", "std"])
def test_with_string_args(datetime_series, arg):
    result = datetime_series.apply(arg)
    expected = getattr(datetime_series, arg)()
    assert result == expected


@pytest.mark.parametrize("op", ["mean", "median", "std", "var"])
@pytest.mark.parametrize("how", ["agg", "apply"])
def test_apply_np_reducer(op, how):
    # GH 39116
    float_frame = DataFrame({"a": [1, 2], "b": [3, 4]})
    result = getattr(float_frame, how)(op)
    # pandas ddof defaults to 1, numpy to 0
    kwargs = {"ddof": 1} if op in ("std", "var") else {}
    expected = Series(
        getattr(np, op)(float_frame, axis=0, **kwargs), index=float_frame.columns
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "op", ["abs", "ceil", "cos", "cumsum", "exp", "log", "sqrt", "square"]
)
@pytest.mark.parametrize("how", ["transform", "apply"])
def test_apply_np_transformer(float_frame, op, how):
    # GH 39116

    # float_frame will _usually_ have negative values, which will
    #  trigger the warning here, but let's put one in just to be sure
    float_frame.iloc[0, 0] = -1.0
    warn = None
    if op in ["log", "sqrt"]:
        warn = RuntimeWarning

    with tm.assert_produces_warning(warn, check_stacklevel=False):
        # float_frame fixture is defined in conftest.py, so we don't check the
        # stacklevel as otherwise the test would fail.
        result = getattr(float_frame, how)(op)
        expected = getattr(np, op)(float_frame)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "series, func, expected",
    chain(
        tm.get_cython_table_params(
            Series(dtype=np.float64),
            [
                ("sum", 0),
                ("max", np.nan),
                ("min", np.nan),
                ("all", True),
                ("any", False),
                ("mean", np.nan),
                ("prod", 1),
                ("std", np.nan),
                ("var", np.nan),
                ("median", np.nan),
            ],
        ),
        tm.get_cython_table_params(
            Series([np.nan, 1, 2, 3]),
            [
                ("sum", 6),
                ("max", 3),
                ("min", 1),
                ("all", True),
                ("any", True),
                ("mean", 2),
                ("prod", 6),
                ("std", 1),
                ("var", 1),
                ("median", 2),
            ],
        ),
        tm.get_cython_table_params(
            Series("a b c".split()),
            [
                ("sum", "abc"),
                ("max", "c"),
                ("min", "a"),
                ("all", True),
                ("any", True),
            ],
        ),
    ),
)
def test_agg_cython_table_series(series, func, expected):
    # GH21224
    # test reducing functions in
    # pandas.core.base.SelectionMixin._cython_table
    warn = None if isinstance(func, str) else FutureWarning
    with tm.assert_produces_warning(warn, match="is currently using Series.*"):
        result = series.agg(func)
    if is_number(expected):
        assert np.isclose(result, expected, equal_nan=True)
    else:
        assert result == expected


@pytest.mark.parametrize(
    "series, func, expected",
    chain(
        tm.get_cython_table_params(
            Series(dtype=np.float64),
            [
                ("cumprod", Series([], dtype=np.float64)),
                ("cumsum", Series([], dtype=np.float64)),
            ],
        ),
        tm.get_cython_table_params(
            Series([np.nan, 1, 2, 3]),
            [
                ("cumprod", Series([np.nan, 1, 2, 6])),
                ("cumsum", Series([np.nan, 1, 3, 6])),
            ],
        ),
        tm.get_cython_table_params(
            Series("a b c".split()), [("cumsum", Series(["a", "ab", "abc"]))]
        ),
    ),
)
def test_agg_cython_table_transform_series(series, func, expected):
    # GH21224
    # test transforming functions in
    # pandas.core.base.SelectionMixin._cython_table (cumprod, cumsum)
    warn = None if isinstance(func, str) else FutureWarning
    with tm.assert_produces_warning(warn, match="is currently using Series.*"):
        result = series.agg(func)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "df, func, expected",
    chain(
        tm.get_cython_table_params(
            DataFrame(),
            [
                ("sum", Series(dtype="float64")),
                ("max", Series(dtype="float64")),
                ("min", Series(dtype="float64")),
                ("all", Series(dtype=bool)),
                ("any", Series(dtype=bool)),
                ("mean", Series(dtype="float64")),
                ("prod", Series(dtype="float64")),
                ("std", Series(dtype="float64")),
                ("var", Series(dtype="float64")),
                ("median", Series(dtype="float64")),
            ],
        ),
        tm.get_cython_table_params(
            DataFrame([[np.nan, 1], [1, 2]]),
            [
                ("sum", Series([1.0, 3])),
                ("max", Series([1.0, 2])),
                ("min", Series([1.0, 1])),
                ("all", Series([True, True])),
                ("any", Series([True, True])),
                ("mean", Series([1, 1.5])),
                ("prod", Series([1.0, 2])),
                ("std", Series([np.nan, 0.707107])),
                ("var", Series([np.nan, 0.5])),
                ("median", Series([1, 1.5])),
            ],
        ),
    ),
)
def test_agg_cython_table_frame(df, func, expected, axis):
    # GH 21224
    # test reducing functions in
    # pandas.core.base.SelectionMixin._cython_table
    warn = None if isinstance(func, str) else FutureWarning
    with tm.assert_produces_warning(warn, match="is currently using DataFrame.*"):
        # GH#53425
        result = df.agg(func, axis=axis)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "df, func, expected",
    chain(
        tm.get_cython_table_params(
            DataFrame(), [("cumprod", DataFrame()), ("cumsum", DataFrame())]
        ),
        tm.get_cython_table_params(
            DataFrame([[np.nan, 1], [1, 2]]),
            [
                ("cumprod", DataFrame([[np.nan, 1], [1, 2]])),
                ("cumsum", DataFrame([[np.nan, 1], [1, 3]])),
            ],
        ),
    ),
)
def test_agg_cython_table_transform_frame(df, func, expected, axis):
    # GH 21224
    # test transforming functions in
    # pandas.core.base.SelectionMixin._cython_table (cumprod, cumsum)
    if axis in ("columns", 1):
        # operating blockwise doesn't let us preserve dtypes
        expected = expected.astype("float64")

    warn = None if isinstance(func, str) else FutureWarning
    with tm.assert_produces_warning(warn, match="is currently using DataFrame.*"):
        # GH#53425
        result = df.agg(func, axis=axis)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("op", series_transform_kernels)
def test_transform_groupby_kernel_series(request, string_series, op):
    # GH 35964
    if op == "ngroup":
        request.applymarker(
            pytest.mark.xfail(raises=ValueError, reason="ngroup not valid for NDFrame")
        )
    args = [0.0] if op == "fillna" else []
    ones = np.ones(string_series.shape[0])

    warn = FutureWarning if op == "fillna" else None
    msg = "SeriesGroupBy.fillna is deprecated"
    with tm.assert_produces_warning(warn, match=msg):
        expected = string_series.groupby(ones).transform(op, *args)
    result = string_series.transform(op, 0, *args)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("op", frame_transform_kernels)
def test_transform_groupby_kernel_frame(request, axis, float_frame, op):
    if op == "ngroup":
        request.applymarker(
            pytest.mark.xfail(raises=ValueError, reason="ngroup not valid for NDFrame")
        )

    # GH 35964

    args = [0.0] if op == "fillna" else []
    if axis in (0, "index"):
        ones = np.ones(float_frame.shape[0])
        msg = "The 'axis' keyword in DataFrame.groupby is deprecated"
    else:
        ones = np.ones(float_frame.shape[1])
        msg = "DataFrame.groupby with axis=1 is deprecated"

    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = float_frame.groupby(ones, axis=axis)

    warn = FutureWarning if op == "fillna" else None
    op_msg = "DataFrameGroupBy.fillna is deprecated"
    with tm.assert_produces_warning(warn, match=op_msg):
        expected = gb.transform(op, *args)

    result = float_frame.transform(op, axis, *args)
    tm.assert_frame_equal(result, expected)

    # same thing, but ensuring we have multiple blocks
    assert "E" not in float_frame.columns
    float_frame["E"] = float_frame["A"].copy()
    assert len(float_frame._mgr.arrays) > 1

    if axis in (0, "index"):
        ones = np.ones(float_frame.shape[0])
    else:
        ones = np.ones(float_frame.shape[1])
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb2 = float_frame.groupby(ones, axis=axis)
    warn = FutureWarning if op == "fillna" else None
    op_msg = "DataFrameGroupBy.fillna is deprecated"
    with tm.assert_produces_warning(warn, match=op_msg):
        expected2 = gb2.transform(op, *args)
    result2 = float_frame.transform(op, axis, *args)
    tm.assert_frame_equal(result2, expected2)


@pytest.mark.parametrize("method", ["abs", "shift", "pct_change", "cumsum", "rank"])
def test_transform_method_name(method):
    # GH 19760
    df = DataFrame({"A": [-1, 2]})
    result = df.transform(method)
    expected = operator.methodcaller(method)(df)
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\boolean\test_construction.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.core.arrays.boolean import coerce_to_array


def test_boolean_array_constructor():
    values = np.array([True, False, True, False], dtype="bool")
    mask = np.array([False, False, False, True], dtype="bool")

    result = BooleanArray(values, mask)
    expected = pd.array([True, False, True, None], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)

    with pytest.raises(TypeError, match="values should be boolean numpy array"):
        BooleanArray(values.tolist(), mask)

    with pytest.raises(TypeError, match="mask should be boolean numpy array"):
        BooleanArray(values, mask.tolist())

    with pytest.raises(TypeError, match="values should be boolean numpy array"):
        BooleanArray(values.astype(int), mask)

    with pytest.raises(TypeError, match="mask should be boolean numpy array"):
        BooleanArray(values, None)

    with pytest.raises(ValueError, match="values.shape must match mask.shape"):
        BooleanArray(values.reshape(1, -1), mask)

    with pytest.raises(ValueError, match="values.shape must match mask.shape"):
        BooleanArray(values, mask.reshape(1, -1))


def test_boolean_array_constructor_copy():
    values = np.array([True, False, True, False], dtype="bool")
    mask = np.array([False, False, False, True], dtype="bool")

    result = BooleanArray(values, mask)
    assert result._data is values
    assert result._mask is mask

    result = BooleanArray(values, mask, copy=True)
    assert result._data is not values
    assert result._mask is not mask


def test_to_boolean_array():
    expected = BooleanArray(
        np.array([True, False, True]), np.array([False, False, False])
    )

    result = pd.array([True, False, True], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)
    result = pd.array(np.array([True, False, True]), dtype="boolean")
    tm.assert_extension_array_equal(result, expected)
    result = pd.array(np.array([True, False, True], dtype=object), dtype="boolean")
    tm.assert_extension_array_equal(result, expected)

    # with missing values
    expected = BooleanArray(
        np.array([True, False, True]), np.array([False, False, True])
    )

    result = pd.array([True, False, None], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)
    result = pd.array(np.array([True, False, None], dtype=object), dtype="boolean")
    tm.assert_extension_array_equal(result, expected)


def test_to_boolean_array_all_none():
    expected = BooleanArray(np.array([True, True, True]), np.array([True, True, True]))

    result = pd.array([None, None, None], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)
    result = pd.array(np.array([None, None, None], dtype=object), dtype="boolean")
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    "a, b",
    [
        ([True, False, None, np.nan, pd.NA], [True, False, None, None, None]),
        ([True, np.nan], [True, None]),
        ([True, pd.NA], [True, None]),
        ([np.nan, np.nan], [None, None]),
        (np.array([np.nan, np.nan], dtype=float), [None, None]),
    ],
)
def test_to_boolean_array_missing_indicators(a, b):
    result = pd.array(a, dtype="boolean")
    expected = pd.array(b, dtype="boolean")
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    "values",
    [
        ["foo", "bar"],
        ["1", "2"],
        # "foo",
        [1, 2],
        [1.0, 2.0],
        pd.date_range("20130101", periods=2),
        np.array(["foo"]),
        np.array([1, 2]),
        np.array([1.0, 2.0]),
        [np.nan, {"a": 1}],
    ],
)
def test_to_boolean_array_error(values):
    # error in converting existing arrays to BooleanArray
    msg = "Need to pass bool-like value"
    with pytest.raises(TypeError, match=msg):
        pd.array(values, dtype="boolean")


def test_to_boolean_array_from_integer_array():
    result = pd.array(np.array([1, 0, 1, 0]), dtype="boolean")
    expected = pd.array([True, False, True, False], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)

    # with missing values
    result = pd.array(np.array([1, 0, 1, None]), dtype="boolean")
    expected = pd.array([True, False, True, None], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)


def test_to_boolean_array_from_float_array():
    result = pd.array(np.array([1.0, 0.0, 1.0, 0.0]), dtype="boolean")
    expected = pd.array([True, False, True, False], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)

    # with missing values
    result = pd.array(np.array([1.0, 0.0, 1.0, np.nan]), dtype="boolean")
    expected = pd.array([True, False, True, None], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)


def test_to_boolean_array_integer_like():
    # integers of 0's and 1's
    result = pd.array([1, 0, 1, 0], dtype="boolean")
    expected = pd.array([True, False, True, False], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)

    # with missing values
    result = pd.array([1, 0, 1, None], dtype="boolean")
    expected = pd.array([True, False, True, None], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)


def test_coerce_to_array():
    # TODO this is currently not public API
    values = np.array([True, False, True, False], dtype="bool")
    mask = np.array([False, False, False, True], dtype="bool")
    result = BooleanArray(*coerce_to_array(values, mask=mask))
    expected = BooleanArray(values, mask)
    tm.assert_extension_array_equal(result, expected)
    assert result._data is values
    assert result._mask is mask
    result = BooleanArray(*coerce_to_array(values, mask=mask, copy=True))
    expected = BooleanArray(values, mask)
    tm.assert_extension_array_equal(result, expected)
    assert result._data is not values
    assert result._mask is not mask

    # mixed missing from values and mask
    values = [True, False, None, False]
    mask = np.array([False, False, False, True], dtype="bool")
    result = BooleanArray(*coerce_to_array(values, mask=mask))
    expected = BooleanArray(
        np.array([True, False, True, True]), np.array([False, False, True, True])
    )
    tm.assert_extension_array_equal(result, expected)
    result = BooleanArray(*coerce_to_array(np.array(values, dtype=object), mask=mask))
    tm.assert_extension_array_equal(result, expected)
    result = BooleanArray(*coerce_to_array(values, mask=mask.tolist()))
    tm.assert_extension_array_equal(result, expected)

    # raise errors for wrong dimension
    values = np.array([True, False, True, False], dtype="bool")
    mask = np.array([False, False, False, True], dtype="bool")

    # passing 2D values is OK as long as no mask
    coerce_to_array(values.reshape(1, -1))

    with pytest.raises(ValueError, match="values.shape and mask.shape must match"):
        coerce_to_array(values.reshape(1, -1), mask=mask)

    with pytest.raises(ValueError, match="values.shape and mask.shape must match"):
        coerce_to_array(values, mask=mask.reshape(1, -1))


def test_coerce_to_array_from_boolean_array():
    # passing BooleanArray to coerce_to_array
    values = np.array([True, False, True, False], dtype="bool")
    mask = np.array([False, False, False, True], dtype="bool")
    arr = BooleanArray(values, mask)
    result = BooleanArray(*coerce_to_array(arr))
    tm.assert_extension_array_equal(result, arr)
    # no copy
    assert result._data is arr._data
    assert result._mask is arr._mask

    result = BooleanArray(*coerce_to_array(arr), copy=True)
    tm.assert_extension_array_equal(result, arr)
    assert result._data is not arr._data
    assert result._mask is not arr._mask

    with pytest.raises(ValueError, match="cannot pass mask for BooleanArray input"):
        coerce_to_array(arr, mask=mask)


def test_coerce_to_numpy_array():
    # with missing values -> object dtype
    arr = pd.array([True, False, None], dtype="boolean")
    result = np.array(arr)
    expected = np.array([True, False, pd.NA], dtype="object")
    tm.assert_numpy_array_equal(result, expected)

    # also with no missing values -> object dtype
    arr = pd.array([True, False, True], dtype="boolean")
    result = np.array(arr)
    expected = np.array([True, False, True], dtype="bool")
    tm.assert_numpy_array_equal(result, expected)

    # force bool dtype
    result = np.array(arr, dtype="bool")
    expected = np.array([True, False, True], dtype="bool")
    tm.assert_numpy_array_equal(result, expected)
    # with missing values will raise error
    arr = pd.array([True, False, None], dtype="boolean")
    msg = (
        "cannot convert to 'bool'-dtype NumPy array with missing values. "
        "Specify an appropriate 'na_value' for this dtype."
    )
    with pytest.raises(ValueError, match=msg):
        np.array(arr, dtype="bool")


def test_to_boolean_array_from_strings():
    result = BooleanArray._from_sequence_of_strings(
        np.array(["True", "False", "1", "1.0", "0", "0.0", np.nan], dtype=object),
        dtype="boolean",
    )
    expected = BooleanArray(
        np.array([True, False, True, True, False, False, False]),
        np.array([False, False, False, False, False, False, True]),
    )

    tm.assert_extension_array_equal(result, expected)


def test_to_boolean_array_from_strings_invalid_string():
    with pytest.raises(ValueError, match="cannot be cast"):
        BooleanArray._from_sequence_of_strings(["donkey"], dtype="boolean")


@pytest.mark.parametrize("box", [True, False], ids=["series", "array"])
def test_to_numpy(box):
    con = pd.Series if box else pd.array
    # default (with or without missing values) -> object dtype
    arr = con([True, False, True], dtype="boolean")
    result = arr.to_numpy()
    expected = np.array([True, False, True], dtype="bool")
    tm.assert_numpy_array_equal(result, expected)

    arr = con([True, False, None], dtype="boolean")
    result = arr.to_numpy()
    expected = np.array([True, False, pd.NA], dtype="object")
    tm.assert_numpy_array_equal(result, expected)

    arr = con([True, False, None], dtype="boolean")
    result = arr.to_numpy(dtype="str")
    expected = np.array([True, False, pd.NA], dtype=f"{tm.ENDIAN}U5")
    tm.assert_numpy_array_equal(result, expected)

    # no missing values -> can convert to bool, otherwise raises
    arr = con([True, False, True], dtype="boolean")
    result = arr.to_numpy(dtype="bool")
    expected = np.array([True, False, True], dtype="bool")
    tm.assert_numpy_array_equal(result, expected)

    arr = con([True, False, None], dtype="boolean")
    with pytest.raises(ValueError, match="cannot convert to 'bool'-dtype"):
        result = arr.to_numpy(dtype="bool")

    # specify dtype and na_value
    arr = con([True, False, None], dtype="boolean")
    result = arr.to_numpy(dtype=object, na_value=None)
    expected = np.array([True, False, None], dtype="object")
    tm.assert_numpy_array_equal(result, expected)

    result = arr.to_numpy(dtype=bool, na_value=False)
    expected = np.array([True, False, False], dtype="bool")
    tm.assert_numpy_array_equal(result, expected)

    result = arr.to_numpy(dtype="int64", na_value=-99)
    expected = np.array([1, 0, -99], dtype="int64")
    tm.assert_numpy_array_equal(result, expected)

    result = arr.to_numpy(dtype="float64", na_value=np.nan)
    expected = np.array([1, 0, np.nan], dtype="float64")
    tm.assert_numpy_array_equal(result, expected)

    # converting to int or float without specifying na_value raises
    with pytest.raises(ValueError, match="cannot convert to 'int64'-dtype"):
        arr.to_numpy(dtype="int64")


def test_to_numpy_copy():
    # to_numpy can be zero-copy if no missing values
    arr = pd.array([True, False, True], dtype="boolean")
    result = arr.to_numpy(dtype=bool)
    result[0] = False
    tm.assert_extension_array_equal(
        arr, pd.array([False, False, True], dtype="boolean")
    )

    arr = pd.array([True, False, True], dtype="boolean")
    result = arr.to_numpy(dtype=bool, copy=True)
    result[0] = False
    tm.assert_extension_array_equal(arr, pd.array([True, False, True], dtype="boolean"))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_modulargcd.py ===
from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, AlgebraicField
from sympy.polys.modulargcd import (
    modgcd_univariate,
    modgcd_bivariate,
    _chinese_remainder_reconstruction_multivariate,
    modgcd_multivariate,
    _to_ZZ_poly,
    _to_ANP_poly,
    func_field_modgcd,
    _func_field_modgcd_m)
from sympy.functions.elementary.miscellaneous import sqrt


def test_modgcd_univariate_integers():
    R, x = ring("x", ZZ)

    f, g = R.zero, R.zero
    assert modgcd_univariate(f, g) == (0, 0, 0)

    f, g = R.zero, x
    assert modgcd_univariate(f, g) == (x, 0, 1)
    assert modgcd_univariate(g, f) == (x, 1, 0)

    f, g = R.zero, -x
    assert modgcd_univariate(f, g) == (x, 0, -1)
    assert modgcd_univariate(g, f) == (x, -1, 0)

    f, g = 2*x, R(2)
    assert modgcd_univariate(f, g) == (2, x, 1)

    f, g = 2*x + 2, 6*x**2 - 6
    assert modgcd_univariate(f, g) == (2*x + 2, 1, 3*x - 3)

    f = x**4 + 8*x**3 + 21*x**2 + 22*x + 8
    g = x**3 + 6*x**2 + 11*x + 6

    h = x**2 + 3*x + 2

    cff = x**2 + 5*x + 4
    cfg = x + 3

    assert modgcd_univariate(f, g) == (h, cff, cfg)

    f = x**4 - 4
    g = x**4 + 4*x**2 + 4

    h = x**2 + 2

    cff = x**2 - 2
    cfg = x**2 + 2

    assert modgcd_univariate(f, g) == (h, cff, cfg)

    f = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    g = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    h = 1

    cff = f
    cfg = g

    assert modgcd_univariate(f, g) == (h, cff, cfg)

    f = - 352518131239247345597970242177235495263669787845475025293906825864749649589178600387510272*x**49 \
        + 46818041807522713962450042363465092040687472354933295397472942006618953623327997952*x**42 \
        + 378182690892293941192071663536490788434899030680411695933646320291525827756032*x**35 \
        + 112806468807371824947796775491032386836656074179286744191026149539708928*x**28 \
        - 12278371209708240950316872681744825481125965781519138077173235712*x**21 \
        + 289127344604779611146960547954288113529690984687482920704*x**14 \
        + 19007977035740498977629742919480623972236450681*x**7 \
        + 311973482284542371301330321821976049

    g =   365431878023781158602430064717380211405897160759702125019136*x**21 \
        + 197599133478719444145775798221171663643171734081650688*x**14 \
        - 9504116979659010018253915765478924103928886144*x**7 \
        - 311973482284542371301330321821976049

    assert modgcd_univariate(f, f.diff(x))[0] == g

    f = 1317378933230047068160*x + 2945748836994210856960
    g = 120352542776360960*x + 269116466014453760

    h = 120352542776360960*x + 269116466014453760
    cff = 10946
    cfg = 1

    assert modgcd_univariate(f, g) == (h, cff, cfg)


def test_modgcd_bivariate_integers():
    R, x, y = ring("x,y", ZZ)

    f, g = R.zero, R.zero
    assert modgcd_bivariate(f, g) == (0, 0, 0)

    f, g = 2*x, R(2)
    assert modgcd_bivariate(f, g) == (2, x, 1)

    f, g = x + 2*y, x + y
    assert modgcd_bivariate(f, g) == (1, f, g)

    f, g = x**2 + 2*x*y + y**2, x**3 + y**3
    assert modgcd_bivariate(f, g) == (x + y, x + y, x**2 - x*y + y**2)

    f, g = x*y**2 + 2*x*y + x, x*y**3 + x
    assert modgcd_bivariate(f, g) == (x*y + x, y + 1, y**2 - y + 1)

    f, g = x**2*y**2 + x**2*y + 1, x*y**2 + x*y + 1
    assert modgcd_bivariate(f, g) == (1, f, g)

    f = 2*x*y**2 + 4*x*y + 2*x + y**2 + 2*y + 1
    g = 2*x*y**3 + 2*x + y**3 + 1
    assert modgcd_bivariate(f, g) == (2*x*y + 2*x + y + 1, y + 1, y**2 - y + 1)

    f, g = 2*x**2 + 4*x + 2, x + 1
    assert modgcd_bivariate(f, g) == (x + 1, 2*x + 2, 1)

    f, g = x + 1, 2*x**2 + 4*x + 2
    assert modgcd_bivariate(f, g) == (x + 1, 1, 2*x + 2)

    f = 2*x**2 + 4*x*y - 2*x - 4*y
    g = x**2 + x - 2
    assert modgcd_bivariate(f, g) == (x - 1, 2*x + 4*y, x + 2)

    f = 2*x**2 + 2*x*y - 3*x - 3*y
    g = 4*x*y - 2*x + 4*y**2 - 2*y
    assert modgcd_bivariate(f, g) == (x + y, 2*x - 3, 4*y - 2)


def test_chinese_remainder():
    R, x, y = ring("x, y", ZZ)
    p, q = 3, 5

    hp = x**3*y - x**2 - 1
    hq = -x**3*y - 2*x*y**2 + 2

    hpq = _chinese_remainder_reconstruction_multivariate(hp, hq, p, q)

    assert hpq.trunc_ground(p) == hp
    assert hpq.trunc_ground(q) == hq

    T, z = ring("z", R)
    p, q = 3, 7

    hp = (x*y + 1)*z**2 + x
    hq = (x**2 - 3*y)*z + 2

    hpq = _chinese_remainder_reconstruction_multivariate(hp, hq, p, q)

    assert hpq.trunc_ground(p) == hp
    assert hpq.trunc_ground(q) == hq


def test_modgcd_multivariate_integers():
    R, x, y = ring("x,y", ZZ)

    f, g = R.zero, R.zero
    assert modgcd_multivariate(f, g) == (0, 0, 0)

    f, g = 2*x**2 + 4*x + 2, x + 1
    assert modgcd_multivariate(f, g) == (x + 1, 2*x + 2, 1)

    f, g = x + 1, 2*x**2 + 4*x + 2
    assert modgcd_multivariate(f, g) == (x + 1, 1, 2*x + 2)

    f = 2*x**2 + 2*x*y - 3*x - 3*y
    g = 4*x*y - 2*x + 4*y**2 - 2*y
    assert modgcd_multivariate(f, g) == (x + y, 2*x - 3, 4*y - 2)

    f, g = x*y**2 + 2*x*y + x, x*y**3 + x
    assert modgcd_multivariate(f, g) == (x*y + x, y + 1, y**2 - y + 1)

    f, g = x**2*y**2 + x**2*y + 1, x*y**2 + x*y + 1
    assert modgcd_multivariate(f, g) == (1, f, g)

    f = x**4 + 8*x**3 + 21*x**2 + 22*x + 8
    g = x**3 + 6*x**2 + 11*x + 6

    h = x**2 + 3*x + 2

    cff = x**2 + 5*x + 4
    cfg = x + 3

    assert modgcd_multivariate(f, g) == (h, cff, cfg)

    R, x, y, z, u = ring("x,y,z,u", ZZ)

    f, g = x + y + z, -x - y - z - u
    assert modgcd_multivariate(f, g) == (1, f, g)

    f, g = u**2 + 2*u + 1, 2*u + 2
    assert modgcd_multivariate(f, g) == (u + 1, u + 1, 2)

    f, g = z**2*u**2 + 2*z**2*u + z**2 + z*u + z, u**2 + 2*u + 1
    h, cff, cfg = u + 1, z**2*u + z**2 + z, u + 1

    assert modgcd_multivariate(f, g) == (h, cff, cfg)
    assert modgcd_multivariate(g, f) == (h, cfg, cff)

    R, x, y, z = ring("x,y,z", ZZ)

    f, g = x - y*z, x - y*z
    assert modgcd_multivariate(f, g) == (x - y*z, 1, 1)

    f, g, h = R.fateman_poly_F_1()
    H, cff, cfg = modgcd_multivariate(f, g)

    assert H == h and H*cff == f and H*cfg == g

    R, x, y, z, u, v = ring("x,y,z,u,v", ZZ)

    f, g, h = R.fateman_poly_F_1()
    H, cff, cfg = modgcd_multivariate(f, g)

    assert H == h and H*cff == f and H*cfg == g

    R, x, y, z, u, v, a, b = ring("x,y,z,u,v,a,b", ZZ)

    f, g, h = R.fateman_poly_F_1()
    H, cff, cfg = modgcd_multivariate(f, g)

    assert H == h and H*cff == f and H*cfg == g

    R, x, y, z, u, v, a, b, c, d = ring("x,y,z,u,v,a,b,c,d", ZZ)

    f, g, h = R.fateman_poly_F_1()
    H, cff, cfg = modgcd_multivariate(f, g)

    assert H == h and H*cff == f and H*cfg == g

    R, x, y, z = ring("x,y,z", ZZ)

    f, g, h = R.fateman_poly_F_2()
    H, cff, cfg = modgcd_multivariate(f, g)

    assert H == h and H*cff == f and H*cfg == g

    f, g, h = R.fateman_poly_F_3()
    H, cff, cfg = modgcd_multivariate(f, g)

    assert H == h and H*cff == f and H*cfg == g

    R, x, y, z, t = ring("x,y,z,t", ZZ)

    f, g, h = R.fateman_poly_F_3()
    H, cff, cfg = modgcd_multivariate(f, g)

    assert H == h and H*cff == f and H*cfg == g


def test_to_ZZ_ANP_poly():
    A = AlgebraicField(QQ, sqrt(2))
    R, x = ring("x", A)
    f = x*(sqrt(2) + 1)

    T, x_, z_ = ring("x_, z_", ZZ)
    f_ = x_*z_ + x_

    assert _to_ZZ_poly(f, T) == f_
    assert _to_ANP_poly(f_, R) == f

    R, x, t, s = ring("x, t, s", A)
    f = x*t**2 + x*s + sqrt(2)

    D, t_, s_ = ring("t_, s_", ZZ)
    T, x_, z_ = ring("x_, z_", D)
    f_ = (t_**2 + s_)*x_ + z_

    assert _to_ZZ_poly(f, T) == f_
    assert _to_ANP_poly(f_, R) == f


def test_modgcd_algebraic_field():
    A = AlgebraicField(QQ, sqrt(2))
    R, x = ring("x", A)
    one = A.one

    f, g = 2*x, R(2)
    assert func_field_modgcd(f, g) == (one, f, g)

    f, g = 2*x, R(sqrt(2))
    assert func_field_modgcd(f, g) == (one, f, g)

    f, g = 2*x + 2, 6*x**2 - 6
    assert func_field_modgcd(f, g) == (x + 1, R(2), 6*x - 6)

    R, x, y = ring("x, y", A)

    f, g = x + sqrt(2)*y, x + y
    assert func_field_modgcd(f, g) == (one, f, g)

    f, g = x*y + sqrt(2)*y**2, R(sqrt(2))*y
    assert func_field_modgcd(f, g) == (y, x + sqrt(2)*y, R(sqrt(2)))

    f, g = x**2 + 2*sqrt(2)*x*y + 2*y**2, x + sqrt(2)*y
    assert func_field_modgcd(f, g) == (g, g, one)

    A = AlgebraicField(QQ, sqrt(2), sqrt(3))
    R, x, y, z = ring("x, y, z", A)

    h = x**2*y**7 + sqrt(6)/21*z
    f, g = h*(27*y**3 + 1), h*(y + x)
    assert func_field_modgcd(f, g) == (h, 27*y**3+1, y+x)

    h = x**13*y**3 + 1/2*x**10 + 1/sqrt(2)
    f, g = h*(x + 1), h*sqrt(2)/sqrt(3)
    assert func_field_modgcd(f, g) == (h, x + 1, R(sqrt(2)/sqrt(3)))

    A = AlgebraicField(QQ, sqrt(2)**(-1)*sqrt(3))
    R, x = ring("x", A)

    f, g = x + 1, x - 1
    assert func_field_modgcd(f, g) == (A.one, f, g)


# when func_field_modgcd supports function fields, this test can be changed
def test_modgcd_func_field():
    D, t = ring("t", ZZ)
    R, x, z = ring("x, z", D)

    minpoly = (z**2*t**2 + z**2*t - 1).drop(0)
    f, g = x + 1, x - 1

    assert _func_field_modgcd_m(f, g, minpoly) == R.one

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\tests\test_prde.py ===
"""Most of these tests come from the examples in Bronstein's book."""
from sympy.integrals.risch import DifferentialExtension, derivation
from sympy.integrals.prde import (prde_normal_denom, prde_special_denom,
    prde_linear_constraints, constant_system, prde_spde, prde_no_cancel_b_large,
    prde_no_cancel_b_small, limited_integrate_reduce, limited_integrate,
    is_deriv_k, is_log_deriv_k_t_radical, parametric_log_deriv_heu,
    is_log_deriv_k_t_radical_in_field, param_poly_rischDE, param_rischDE,
    prde_cancel_liouvillian)

from sympy.polys.polymatrix import PolyMatrix as Matrix

from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.polytools import Poly
from sympy.abc import x, t, n

t0, t1, t2, t3, k = symbols('t:4 k')


def test_prde_normal_denom():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t**2, t)]})
    fa = Poly(1, t)
    fd = Poly(x, t)
    G = [(Poly(t, t), Poly(1 + t**2, t)), (Poly(1, t), Poly(x + x*t**2, t))]
    assert prde_normal_denom(fa, fd, G, DE) == \
        (Poly(x, t, domain='ZZ(x)'), (Poly(1, t, domain='ZZ(x)'), Poly(1, t,
            domain='ZZ(x)')), [(Poly(x*t, t, domain='ZZ(x)'),
         Poly(t**2 + 1, t, domain='ZZ(x)')), (Poly(1, t, domain='ZZ(x)'),
             Poly(t**2 + 1, t, domain='ZZ(x)'))], Poly(1, t, domain='ZZ(x)'))
    G = [(Poly(t, t), Poly(t**2 + 2*t + 1, t)), (Poly(x*t, t),
        Poly(t**2 + 2*t + 1, t)), (Poly(x*t**2, t), Poly(t**2 + 2*t + 1, t))]
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    assert prde_normal_denom(Poly(x, t), Poly(1, t), G, DE) == \
        (Poly(t + 1, t), (Poly((-1 + x)*t + x, t), Poly(1, t, domain='ZZ[x]')), [(Poly(t, t),
        Poly(1, t)), (Poly(x*t, t), Poly(1, t, domain='ZZ[x]')), (Poly(x*t**2, t),
        Poly(1, t, domain='ZZ[x]'))], Poly(t + 1, t))


def test_prde_special_denom():
    a = Poly(t + 1, t)
    ba = Poly(t**2, t)
    bd = Poly(1, t)
    G = [(Poly(t, t), Poly(1, t)), (Poly(t**2, t), Poly(1, t)), (Poly(t**3, t), Poly(1, t))]
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    assert prde_special_denom(a, ba, bd, G, DE) == \
        (Poly(t + 1, t), Poly(t**2, t), [(Poly(t, t), Poly(1, t)),
        (Poly(t**2, t), Poly(1, t)), (Poly(t**3, t), Poly(1, t))], Poly(1, t))
    G = [(Poly(t, t), Poly(1, t)), (Poly(1, t), Poly(t, t))]
    assert prde_special_denom(Poly(1, t), Poly(t**2, t), Poly(1, t), G, DE) == \
        (Poly(1, t), Poly(t**2 - 1, t), [(Poly(t**2, t), Poly(1, t)),
        (Poly(1, t), Poly(1, t))], Poly(t, t))
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-2*x*t0, t0)]})
    DE.decrement_level()
    G = [(Poly(t, t), Poly(t**2, t)), (Poly(2*t, t), Poly(t, t))]
    assert prde_special_denom(Poly(5*x*t + 1, t), Poly(t**2 + 2*x**3*t, t), Poly(t**3 + 2, t), G, DE) == \
        (Poly(5*x*t + 1, t), Poly(0, t, domain='ZZ[x]'), [(Poly(t, t), Poly(t**2, t)),
        (Poly(2*t, t), Poly(t, t))], Poly(1, x))
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly((t**2 + 1)*2*x, t)]})
    G = [(Poly(t + x, t), Poly(t*x, t)), (Poly(2*t, t), Poly(x**2, x))]
    assert prde_special_denom(Poly(5*x*t + 1, t), Poly(t**2 + 2*x**3*t, t), Poly(t**3, t), G, DE) == \
        (Poly(5*x*t + 1, t), Poly(0, t, domain='ZZ[x]'), [(Poly(t + x, t), Poly(x*t, t)),
        (Poly(2*t, t, x), Poly(x**2, t, x))], Poly(1, t))
    assert prde_special_denom(Poly(t + 1, t), Poly(t**2, t), Poly(t**3, t), G, DE) == \
        (Poly(t + 1, t), Poly(0, t, domain='ZZ[x]'), [(Poly(t + x, t), Poly(x*t, t)), (Poly(2*t, t, x),
        Poly(x**2, t, x))], Poly(1, t))


def test_prde_linear_constraints():
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    G = [(Poly(2*x**3 + 3*x + 1, x), Poly(x**2 - 1, x)), (Poly(1, x), Poly(x - 1, x)),
        (Poly(1, x), Poly(x + 1, x))]
    assert prde_linear_constraints(Poly(1, x), Poly(0, x), G, DE) == \
        ((Poly(2*x, x, domain='QQ'), Poly(0, x, domain='QQ'), Poly(0, x, domain='QQ')),
            Matrix([[1, 1, -1], [5, 1, 1]], x))
    G = [(Poly(t, t), Poly(1, t)), (Poly(t**2, t), Poly(1, t)), (Poly(t**3, t), Poly(1, t))]
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    assert prde_linear_constraints(Poly(t + 1, t), Poly(t**2, t), G, DE) == \
        ((Poly(t, t, domain='QQ'), Poly(t**2, t, domain='QQ'), Poly(t**3, t, domain='QQ')),
            Matrix(0, 3, [], t))
    G = [(Poly(2*x, t), Poly(t, t)), (Poly(-x, t), Poly(t, t))]
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})
    assert prde_linear_constraints(Poly(1, t), Poly(0, t), G, DE) == \
        ((Poly(0, t, domain='QQ[x]'), Poly(0, t, domain='QQ[x]')), Matrix([[2*x, -x]], t))


def test_constant_system():
    A = Matrix([[-(x + 3)/(x - 1), (x + 1)/(x - 1), 1],
                [-x - 3, x + 1, x - 1],
                [2*(x + 3)/(x - 1), 0, 0]], t)
    u = Matrix([[(x + 1)/(x - 1)], [x + 1], [0]], t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    R = QQ.frac_field(x)[t]
    assert constant_system(A, u, DE) == \
        (Matrix([[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 0],
                 [0, 0, 1]], ring=R), Matrix([0, 1, 0, 0], ring=R))


def test_prde_spde():
    D = [Poly(x, t), Poly(-x*t, t)]
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})
    # TODO: when bound_degree() can handle this, test degree bound from that too
    assert prde_spde(Poly(t, t), Poly(-1/x, t), D, n, DE) == \
        (Poly(t, t), Poly(0, t, domain='ZZ(x)'),
        [Poly(2*x, t, domain='ZZ(x)'), Poly(-x, t, domain='ZZ(x)')],
        [Poly(-x**2, t, domain='ZZ(x)'), Poly(0, t, domain='ZZ(x)')], n - 1)


def test_prde_no_cancel():
    # b large
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    assert prde_no_cancel_b_large(Poly(1, x), [Poly(x**2, x), Poly(1, x)], 2, DE) == \
        ([Poly(x**2 - 2*x + 2, x), Poly(1, x)], Matrix([[1, 0, -1, 0],
                                                        [0, 1, 0, -1]], x))
    assert prde_no_cancel_b_large(Poly(1, x), [Poly(x**3, x), Poly(1, x)], 3, DE) == \
        ([Poly(x**3 - 3*x**2 + 6*x - 6, x), Poly(1, x)], Matrix([[1, 0, -1, 0],
                                                                 [0, 1, 0, -1]], x))
    assert prde_no_cancel_b_large(Poly(x, x), [Poly(x**2, x), Poly(1, x)], 1, DE) == \
        ([Poly(x, x, domain='ZZ'), Poly(0, x, domain='ZZ')], Matrix([[1, -1,  0,  0],
                                                                    [1,  0, -1,  0],
                                                                    [0,  1,  0, -1]], x))
    # b small
    # XXX: Is there a better example of a monomial with D.degree() > 2?
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**3 + 1, t)]})

    # My original q was t**4 + t + 1, but this solution implies q == t**4
    # (c1 = 4), with some of the ci for the original q equal to 0.
    G = [Poly(t**6, t), Poly(x*t**5, t), Poly(t**3, t), Poly(x*t**2, t), Poly(1 + x, t)]
    R = QQ.frac_field(x)[t]
    assert prde_no_cancel_b_small(Poly(x*t, t), G, 4, DE) == \
        ([Poly(t**4/4 - x/12*t**3 + x**2/24*t**2 + (Rational(-11, 12) - x**3/24)*t + x/24, t),
        Poly(x/3*t**3 - x**2/6*t**2 + (Rational(-1, 3) + x**3/6)*t - x/6, t), Poly(t, t),
        Poly(0, t), Poly(0, t)], Matrix([[1, 0,              -1, 0, 0,  0,  0,  0,  0,  0],
                                         [0, 1, Rational(-1, 4), 0, 0,  0,  0,  0,  0,  0],
                                         [0, 0,               0, 0, 0,  0,  0,  0,  0,  0],
                                         [0, 0,               0, 1, 0,  0,  0,  0,  0,  0],
                                         [0, 0,               0, 0, 1,  0,  0,  0,  0,  0],
                                         [1, 0,               0, 0, 0, -1,  0,  0,  0,  0],
                                         [0, 1,               0, 0, 0,  0, -1,  0,  0,  0],
                                         [0, 0,               1, 0, 0,  0,  0, -1,  0,  0],
                                         [0, 0,               0, 1, 0,  0,  0,  0, -1,  0],
                                         [0, 0,               0, 0, 1,  0,  0,  0,  0, -1]], ring=R))

    # TODO: Add test for deg(b) <= 0 with b small
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t**2, t)]})
    b = Poly(-1/x**2, t, field=True)  # deg(b) == 0
    q = [Poly(x**i*t**j, t, field=True) for i in range(2) for j in range(3)]
    h, A = prde_no_cancel_b_small(b, q, 3, DE)
    V = A.nullspace()
    R = QQ.frac_field(x)[t]
    assert len(V) == 1
    assert V[0] == Matrix([Rational(-1, 2), 0, 0, 1, 0, 0]*3, ring=R)
    assert (Matrix([h])*V[0][6:, :])[0] == Poly(x**2/2, t, domain='QQ(x)')
    assert (Matrix([q])*V[0][:6, :])[0] == Poly(x - S.Half, t, domain='QQ(x)')


def test_prde_cancel_liouvillian():
    ### 1. case == 'primitive'
    # used when integrating f = log(x) - log(x - 1)
    # Not taken from 'the' book
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})
    p0 = Poly(0, t, field=True)
    p1 = Poly((x - 1)*t, t, domain='ZZ(x)')
    p2 = Poly(x - 1, t, domain='ZZ(x)')
    p3 = Poly(-x**2 + x, t, domain='ZZ(x)')
    h, A = prde_cancel_liouvillian(Poly(-1/(x - 1), t), [Poly(-x + 1, t), Poly(1, t)], 1, DE)
    V = A.nullspace()
    assert h == [p0, p0, p1, p0, p0, p0, p0, p0, p0, p0, p2, p3, p0, p0, p0, p0]
    assert A.rank() == 16
    assert (Matrix([h])*V[0][:16, :]) == Matrix([[Poly(0, t, domain='QQ(x)')]])

    ### 2. case == 'exp'
    # used when integrating log(x/exp(x) + 1)
    # Not taken from book
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-t, t)]})
    assert prde_cancel_liouvillian(Poly(0, t, domain='QQ[x]'), [Poly(1, t, domain='QQ(x)')], 0, DE) == \
            ([Poly(1, t, domain='QQ'), Poly(x, t, domain='ZZ(x)')], Matrix([[-1, 0, 1]], DE.t))


def test_param_poly_rischDE():
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    a = Poly(x**2 - x, x, field=True)
    b = Poly(1, x, field=True)
    q = [Poly(x, x, field=True), Poly(x**2, x, field=True)]
    h, A = param_poly_rischDE(a, b, q, 3, DE)

    assert A.nullspace() == [Matrix([0, 1, 1, 1], DE.t)]  # c1, c2, d1, d2
    # Solution of a*Dp + b*p = c1*q1 + c2*q2 = q2 = x**2
    # is d1*h1 + d2*h2 = h1 + h2 = x.
    assert h[0] + h[1] == Poly(x, x, domain='QQ')
    # a*Dp + b*p = q1 = x has no solution.

    a = Poly(x**2 - x, x, field=True)
    b = Poly(x**2 - 5*x + 3, x, field=True)
    q = [Poly(1, x, field=True), Poly(x, x, field=True),
         Poly(x**2, x, field=True)]
    h, A = param_poly_rischDE(a, b, q, 3, DE)

    assert A.nullspace() == [Matrix([3, -5, 1, -5, 1, 1], DE.t)]
    p = -Poly(5, DE.t)*h[0] + h[1] + h[2]  # Poly(1, x)
    assert a*derivation(p, DE) + b*p == Poly(x**2 - 5*x + 3, x, domain='QQ')


def test_param_rischDE():
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    p1, px = Poly(1, x, field=True), Poly(x, x, field=True)
    G = [(p1, px), (p1, p1), (px, p1)]  # [1/x, 1, x]
    h, A = param_rischDE(-p1, Poly(x**2, x, field=True), G, DE)
    assert len(h) == 3
    p = [hi[0].as_expr()/hi[1].as_expr() for hi in h]
    V = A.nullspace()
    assert len(V) == 2
    assert V[0] == Matrix([-1, 1, 0, -1, 1, 0], DE.t)
    y = -p[0] + p[1] + 0*p[2]  # x
    assert y.diff(x) - y/x**2 == 1 - 1/x  # Dy + f*y == -G0 + G1 + 0*G2

    # the below test computation takes place while computing the integral
    # of 'f = log(log(x + exp(x)))'
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    G = [(Poly(t + x, t, domain='ZZ(x)'), Poly(1, t, domain='QQ')), (Poly(0, t, domain='QQ'), Poly(1, t, domain='QQ'))]
    h, A = param_rischDE(Poly(-t - 1, t, field=True), Poly(t + x, t, field=True), G, DE)
    assert len(h) == 5
    p = [hi[0].as_expr()/hi[1].as_expr() for hi in h]
    V = A.nullspace()
    assert len(V) == 3
    assert V[0] == Matrix([0, 0, 0, 0, 1, 0, 0], DE.t)
    y = 0*p[0] + 0*p[1] + 1*p[2] + 0*p[3] + 0*p[4]
    assert y.diff(t) - y/(t + x) == 0   # Dy + f*y = 0*G0 + 0*G1


def test_limited_integrate_reduce():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})
    assert limited_integrate_reduce(Poly(x, t), Poly(t**2, t), [(Poly(x, t),
    Poly(t, t))], DE) == \
        (Poly(t, t), Poly(-1/x, t), Poly(t, t), 1, (Poly(x, t), Poly(1, t, domain='ZZ[x]')),
        [(Poly(-x*t, t), Poly(1, t, domain='ZZ[x]'))])


def test_limited_integrate():
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    G = [(Poly(x, x), Poly(x + 1, x))]
    assert limited_integrate(Poly(-(1 + x + 5*x**2 - 3*x**3), x),
    Poly(1 - x - x**2 + x**3, x), G, DE) == \
        ((Poly(x**2 - x + 2, x), Poly(x - 1, x, domain='QQ')), [2])
    G = [(Poly(1, x), Poly(x, x))]
    assert limited_integrate(Poly(5*x**2, x), Poly(3, x), G, DE) == \
        ((Poly(5*x**3/9, x), Poly(1, x, domain='QQ')), [0])


def test_is_log_deriv_k_t_radical():
    DE = DifferentialExtension(extension={'D': [Poly(1, x)], 'exts': [None],
        'extargs': [None]})
    assert is_log_deriv_k_t_radical(Poly(2*x, x), Poly(1, x), DE) is None

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(2*t1, t1), Poly(1/x, t2)],
        'exts': [None, 'exp', 'log'], 'extargs': [None, 2*x, x]})
    assert is_log_deriv_k_t_radical(Poly(x + t2/2, t2), Poly(1, t2), DE) == \
        ([(t1, 1), (x, 1)], t1*x, 2, 0)
    # TODO: Add more tests

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t0, t0), Poly(1/x, t)],
        'exts': [None, 'exp', 'log'], 'extargs': [None, x, x]})
    assert is_log_deriv_k_t_radical(Poly(x + t/2 + 3, t), Poly(1, t), DE) == \
        ([(t0, 2), (x, 1)], x*t0**2, 2, 3)


def test_is_deriv_k():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t1), Poly(1/(x + 1), t2)],
        'exts': [None, 'log', 'log'], 'extargs': [None, x, x + 1]})
    assert is_deriv_k(Poly(2*x**2 + 2*x, t2), Poly(1, t2), DE) == \
        ([(t1, 1), (t2, 1)], t1 + t2, 2)

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t1), Poly(t2, t2)],
        'exts': [None, 'log', 'exp'], 'extargs': [None, x, x]})
    assert is_deriv_k(Poly(x**2*t2**3, t2), Poly(1, t2), DE) == \
        ([(x, 3), (t1, 2)], 2*t1 + 3*x, 1)
    # TODO: Add more tests, including ones with exponentials

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(2/x, t1)],
        'exts': [None, 'log'], 'extargs': [None, x**2]})
    assert is_deriv_k(Poly(x, t1), Poly(1, t1), DE) == \
        ([(t1, S.Half)], t1/2, 1)

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(2/(1 + x), t0)],
        'exts': [None, 'log'], 'extargs': [None, x**2 + 2*x + 1]})
    assert is_deriv_k(Poly(1 + x, t0), Poly(1, t0), DE) == \
        ([(t0, S.Half)], t0/2, 1)

    # Issue 10798
    # DE = DifferentialExtension(log(1/x), x)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-1/x, t)],
        'exts': [None, 'log'], 'extargs': [None, 1/x]})
    assert is_deriv_k(Poly(1, t), Poly(x, t), DE) == ([(t, 1)], t, 1)


def test_is_log_deriv_k_t_radical_in_field():
    # NOTE: any potential constant factor in the second element of the result
    # doesn't matter, because it cancels in Da/a.
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})
    assert is_log_deriv_k_t_radical_in_field(Poly(5*t + 1, t), Poly(2*t*x, t), DE) == \
        (2, t*x**5)
    assert is_log_deriv_k_t_radical_in_field(Poly(2 + 3*t, t), Poly(5*x*t, t), DE) == \
        (5, x**3*t**2)

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-t/x**2, t)]})
    assert is_log_deriv_k_t_radical_in_field(Poly(-(1 + 2*t), t),
    Poly(2*x**2 + 2*x**2*t, t), DE) == \
        (2, t + t**2)
    assert is_log_deriv_k_t_radical_in_field(Poly(-1, t), Poly(x**2, t), DE) == \
        (1, t)
    assert is_log_deriv_k_t_radical_in_field(Poly(1, t), Poly(2*x**2, t), DE) == \
        (2, 1/t)


def test_parametric_log_deriv():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})
    assert parametric_log_deriv_heu(Poly(5*t**2 + t - 6, t), Poly(2*x*t**2, t),
    Poly(-1, t), Poly(x*t**2, t), DE) == \
        (2, 6, t*x**5)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\vector\tests\test_field_functions.py ===
from sympy.core.function import Derivative
from sympy.vector.vector import Vector
from sympy.vector.coordsysrect import CoordSys3D
from sympy.simplify import simplify
from sympy.core.symbol import symbols
from sympy.core import S
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.vector.vector import Dot
from sympy.vector.operators import curl, divergence, gradient, Gradient, Divergence, Cross
from sympy.vector.deloperator import Del
from sympy.vector.functions import (is_conservative, is_solenoidal,
                                    scalar_potential, directional_derivative,
                                    laplacian, scalar_potential_difference)
from sympy.testing.pytest import raises

C = CoordSys3D('C')
i, j, k = C.base_vectors()
x, y, z = C.base_scalars()
delop = Del()
a, b, c, q = symbols('a b c q')


def test_del_operator():
    # Tests for curl

    assert delop ^ Vector.zero == Vector.zero
    assert ((delop ^ Vector.zero).doit() == Vector.zero ==
            curl(Vector.zero))
    assert delop.cross(Vector.zero) == delop ^ Vector.zero
    assert (delop ^ i).doit() == Vector.zero
    assert delop.cross(2*y**2*j, doit=True) == Vector.zero
    assert delop.cross(2*y**2*j) == delop ^ 2*y**2*j
    v = x*y*z * (i + j + k)
    assert ((delop ^ v).doit() ==
            (-x*y + x*z)*i + (x*y - y*z)*j + (-x*z + y*z)*k ==
            curl(v))
    assert delop ^ v == delop.cross(v)
    assert (delop.cross(2*x**2*j) ==
            (Derivative(0, C.y) - Derivative(2*C.x**2, C.z))*C.i +
            (-Derivative(0, C.x) + Derivative(0, C.z))*C.j +
            (-Derivative(0, C.y) + Derivative(2*C.x**2, C.x))*C.k)
    assert (delop.cross(2*x**2*j, doit=True) == 4*x*k ==
            curl(2*x**2*j))

    #Tests for divergence
    assert delop & Vector.zero is S.Zero == divergence(Vector.zero)
    assert (delop & Vector.zero).doit() is S.Zero
    assert delop.dot(Vector.zero) == delop & Vector.zero
    assert (delop & i).doit() is S.Zero
    assert (delop & x**2*i).doit() == 2*x == divergence(x**2*i)
    assert (delop.dot(v, doit=True) == x*y + y*z + z*x ==
            divergence(v))
    assert delop & v == delop.dot(v)
    assert delop.dot(1/(x*y*z) * (i + j + k), doit=True) == \
           - 1 / (x*y*z**2) - 1 / (x*y**2*z) - 1 / (x**2*y*z)
    v = x*i + y*j + z*k
    assert (delop & v == Derivative(C.x, C.x) +
            Derivative(C.y, C.y) + Derivative(C.z, C.z))
    assert delop.dot(v, doit=True) == 3 == divergence(v)
    assert delop & v == delop.dot(v)
    assert simplify((delop & v).doit()) == 3

    #Tests for gradient
    assert (delop.gradient(0, doit=True) == Vector.zero ==
            gradient(0))
    assert delop.gradient(0) == delop(0)
    assert (delop(S.Zero)).doit() == Vector.zero
    assert (delop(x) == (Derivative(C.x, C.x))*C.i +
            (Derivative(C.x, C.y))*C.j + (Derivative(C.x, C.z))*C.k)
    assert (delop(x)).doit() == i == gradient(x)
    assert (delop(x*y*z) ==
            (Derivative(C.x*C.y*C.z, C.x))*C.i +
            (Derivative(C.x*C.y*C.z, C.y))*C.j +
            (Derivative(C.x*C.y*C.z, C.z))*C.k)
    assert (delop.gradient(x*y*z, doit=True) ==
            y*z*i + z*x*j + x*y*k ==
            gradient(x*y*z))
    assert delop(x*y*z) == delop.gradient(x*y*z)
    assert (delop(2*x**2)).doit() == 4*x*i
    assert ((delop(a*sin(y) / x)).doit() ==
            -a*sin(y)/x**2 * i + a*cos(y)/x * j)

    #Tests for directional derivative
    assert (Vector.zero & delop)(a) is S.Zero
    assert ((Vector.zero & delop)(a)).doit() is S.Zero
    assert ((v & delop)(Vector.zero)).doit() == Vector.zero
    assert ((v & delop)(S.Zero)).doit() is S.Zero
    assert ((i & delop)(x)).doit() == 1
    assert ((j & delop)(y)).doit() == 1
    assert ((k & delop)(z)).doit() == 1
    assert ((i & delop)(x*y*z)).doit() == y*z
    assert ((v & delop)(x)).doit() == x
    assert ((v & delop)(x*y*z)).doit() == 3*x*y*z
    assert (v & delop)(x + y + z) == C.x + C.y + C.z
    assert ((v & delop)(x + y + z)).doit() == x + y + z
    assert ((v & delop)(v)).doit() == v
    assert ((i & delop)(v)).doit() == i
    assert ((j & delop)(v)).doit() == j
    assert ((k & delop)(v)).doit() == k
    assert ((v & delop)(Vector.zero)).doit() == Vector.zero

    # Tests for laplacian on scalar fields
    assert laplacian(x*y*z) is S.Zero
    assert laplacian(x**2) == S(2)
    assert laplacian(x**2*y**2*z**2) == \
                    2*y**2*z**2 + 2*x**2*z**2 + 2*x**2*y**2
    A = CoordSys3D('A', transformation="spherical", variable_names=["r", "theta", "phi"])
    B = CoordSys3D('B', transformation='cylindrical', variable_names=["r", "theta", "z"])
    assert laplacian(A.r + A.theta + A.phi) == 2/A.r + cos(A.theta)/(A.r**2*sin(A.theta))
    assert laplacian(B.r + B.theta + B.z) == 1/B.r

    # Tests for laplacian on vector fields
    assert laplacian(x*y*z*(i + j + k)) == Vector.zero
    assert laplacian(x*y**2*z*(i + j + k)) == \
                            2*x*z*i + 2*x*z*j + 2*x*z*k


def test_product_rules():
    """
    Tests the six product rules defined with respect to the Del
    operator

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Del

    """

    #Define the scalar and vector functions
    f = 2*x*y*z
    g = x*y + y*z + z*x
    u = x**2*i + 4*j - y**2*z*k
    v = 4*i + x*y*z*k

    # First product rule
    lhs = delop(f * g, doit=True)
    rhs = (f * delop(g) + g * delop(f)).doit()
    assert simplify(lhs) == simplify(rhs)

    # Second product rule
    lhs = delop(u & v).doit()
    rhs = ((u ^ (delop ^ v)) + (v ^ (delop ^ u)) + \
          ((u & delop)(v)) + ((v & delop)(u))).doit()
    assert simplify(lhs) == simplify(rhs)

    # Third product rule
    lhs = (delop & (f*v)).doit()
    rhs = ((f * (delop & v)) + (v & (delop(f)))).doit()
    assert simplify(lhs) == simplify(rhs)

    # Fourth product rule
    lhs = (delop & (u ^ v)).doit()
    rhs = ((v & (delop ^ u)) - (u & (delop ^ v))).doit()
    assert simplify(lhs) == simplify(rhs)

    # Fifth product rule
    lhs = (delop ^ (f * v)).doit()
    rhs = (((delop(f)) ^ v) + (f * (delop ^ v))).doit()
    assert simplify(lhs) == simplify(rhs)

    # Sixth product rule
    lhs = (delop ^ (u ^ v)).doit()
    rhs = (u * (delop & v) - v * (delop & u) +
           (v & delop)(u) - (u & delop)(v)).doit()
    assert simplify(lhs) == simplify(rhs)


P = C.orient_new_axis('P', q, C.k)  # type: ignore
scalar_field = 2*x**2*y*z
grad_field = gradient(scalar_field)
vector_field = y**2*i + 3*x*j + 5*y*z*k
curl_field = curl(vector_field)


def test_conservative():
    assert is_conservative(Vector.zero) is True
    assert is_conservative(i) is True
    assert is_conservative(2 * i + 3 * j + 4 * k) is True
    assert (is_conservative(y*z*i + x*z*j + x*y*k) is
            True)
    assert is_conservative(x * j) is False
    assert is_conservative(grad_field) is True
    assert is_conservative(curl_field) is False
    assert (is_conservative(4*x*y*z*i + 2*x**2*z*j) is
            False)
    assert is_conservative(z*P.i + P.x*k) is True


def test_solenoidal():
    assert is_solenoidal(Vector.zero) is True
    assert is_solenoidal(i) is True
    assert is_solenoidal(2 * i + 3 * j + 4 * k) is True
    assert (is_solenoidal(y*z*i + x*z*j + x*y*k) is
            True)
    assert is_solenoidal(y * j) is False
    assert is_solenoidal(grad_field) is False
    assert is_solenoidal(curl_field) is True
    assert is_solenoidal((-2*y + 3)*k) is True
    assert is_solenoidal(cos(q)*i + sin(q)*j + cos(q)*P.k) is True
    assert is_solenoidal(z*P.i + P.x*k) is True


def test_directional_derivative():
    assert directional_derivative(C.x*C.y*C.z, 3*C.i + 4*C.j + C.k) == C.x*C.y + 4*C.x*C.z + 3*C.y*C.z
    assert directional_derivative(5*C.x**2*C.z, 3*C.i + 4*C.j + C.k) == 5*C.x**2 + 30*C.x*C.z
    assert directional_derivative(5*C.x**2*C.z, 4*C.j) is S.Zero

    D = CoordSys3D("D", "spherical", variable_names=["r", "theta", "phi"],
                   vector_names=["e_r", "e_theta", "e_phi"])
    r, theta, phi = D.base_scalars()
    e_r, e_theta, e_phi = D.base_vectors()
    assert directional_derivative(r**2*e_r, e_r) == 2*r*e_r
    assert directional_derivative(5*r**2*phi, 3*e_r + 4*e_theta + e_phi) == 5*r**2 + 30*r*phi


def test_scalar_potential():
    assert scalar_potential(Vector.zero, C) == 0
    assert scalar_potential(i, C) == x
    assert scalar_potential(j, C) == y
    assert scalar_potential(k, C) == z
    assert scalar_potential(y*z*i + x*z*j + x*y*k, C) == x*y*z
    assert scalar_potential(grad_field, C) == scalar_field
    assert scalar_potential(z*P.i + P.x*k, C) == x*z*cos(q) + y*z*sin(q)
    assert scalar_potential(z*P.i + P.x*k, P) == P.x*P.z
    raises(ValueError, lambda: scalar_potential(x*j, C))


def test_scalar_potential_difference():
    point1 = C.origin.locate_new('P1', 1*i + 2*j + 3*k)
    point2 = C.origin.locate_new('P2', 4*i + 5*j + 6*k)
    genericpointC = C.origin.locate_new('RP', x*i + y*j + z*k)
    genericpointP = P.origin.locate_new('PP', P.x*P.i + P.y*P.j + P.z*P.k)
    assert scalar_potential_difference(S.Zero, C, point1, point2) == 0
    assert (scalar_potential_difference(scalar_field, C, C.origin,
                                        genericpointC) ==
            scalar_field)
    assert (scalar_potential_difference(grad_field, C, C.origin,
                                        genericpointC) ==
            scalar_field)
    assert scalar_potential_difference(grad_field, C, point1, point2) == 948
    assert (scalar_potential_difference(y*z*i + x*z*j +
                                        x*y*k, C, point1,
                                        genericpointC) ==
            x*y*z - 6)
    potential_diff_P = (2*P.z*(P.x*sin(q) + P.y*cos(q))*
                        (P.x*cos(q) - P.y*sin(q))**2)
    assert (scalar_potential_difference(grad_field, P, P.origin,
                                        genericpointP).simplify() ==
            potential_diff_P.simplify())


def test_differential_operators_curvilinear_system():
    A = CoordSys3D('A', transformation="spherical", variable_names=["r", "theta", "phi"])
    B = CoordSys3D('B', transformation='cylindrical', variable_names=["r", "theta", "z"])
    # Test for spherical coordinate system and gradient
    assert gradient(3*A.r + 4*A.theta) == 3*A.i + 4/A.r*A.j
    assert gradient(3*A.r*A.phi + 4*A.theta) == 3*A.phi*A.i + 4/A.r*A.j + (3/sin(A.theta))*A.k
    assert gradient(0*A.r + 0*A.theta+0*A.phi) == Vector.zero
    assert gradient(A.r*A.theta*A.phi) == A.theta*A.phi*A.i + A.phi*A.j + (A.theta/sin(A.theta))*A.k
    # Test for spherical coordinate system and divergence
    assert divergence(A.r * A.i + A.theta * A.j + A.phi * A.k) == \
           (sin(A.theta)*A.r + cos(A.theta)*A.r*A.theta)/(sin(A.theta)*A.r**2) + 3 + 1/(sin(A.theta)*A.r)
    assert divergence(3*A.r*A.phi*A.i + A.theta*A.j + A.r*A.theta*A.phi*A.k) == \
           (sin(A.theta)*A.r + cos(A.theta)*A.r*A.theta)/(sin(A.theta)*A.r**2) + 9*A.phi + A.theta/sin(A.theta)
    assert divergence(Vector.zero) == 0
    assert divergence(0*A.i + 0*A.j + 0*A.k) == 0
    # Test for spherical coordinate system and curl
    assert curl(A.r*A.i + A.theta*A.j + A.phi*A.k) == \
           (cos(A.theta)*A.phi/(sin(A.theta)*A.r))*A.i + (-A.phi/A.r)*A.j + A.theta/A.r*A.k
    assert curl(A.r*A.j + A.phi*A.k) == (cos(A.theta)*A.phi/(sin(A.theta)*A.r))*A.i + (-A.phi/A.r)*A.j + 2*A.k

    # Test for cylindrical coordinate system and gradient
    assert gradient(0*B.r + 0*B.theta+0*B.z) == Vector.zero
    assert gradient(B.r*B.theta*B.z) == B.theta*B.z*B.i + B.z*B.j + B.r*B.theta*B.k
    assert gradient(3*B.r) == 3*B.i
    assert gradient(2*B.theta) == 2/B.r * B.j
    assert gradient(4*B.z) == 4*B.k
    # Test for cylindrical coordinate system and divergence
    assert divergence(B.r*B.i + B.theta*B.j + B.z*B.k) == 3 + 1/B.r
    assert divergence(B.r*B.j + B.z*B.k) == 1
    # Test for cylindrical coordinate system and curl
    assert curl(B.r*B.j + B.z*B.k) == 2*B.k
    assert curl(3*B.i + 2/B.r*B.j + 4*B.k) == Vector.zero

def test_mixed_coordinates():
    # gradient
    a = CoordSys3D('a')
    b = CoordSys3D('b')
    c = CoordSys3D('c')
    assert gradient(a.x*b.y) == b.y*a.i + a.x*b.j
    assert gradient(3*cos(q)*a.x*b.x+a.y*(a.x+(cos(q)+b.x))) ==\
           (a.y + 3*b.x*cos(q))*a.i + (a.x + b.x + cos(q))*a.j + (3*a.x*cos(q) + a.y)*b.i
    # Some tests need further work:
    # assert gradient(a.x*(cos(a.x+b.x))) == (cos(a.x + b.x))*a.i + a.x*Gradient(cos(a.x + b.x))
    # assert gradient(cos(a.x + b.x)*cos(a.x + b.z)) == Gradient(cos(a.x + b.x)*cos(a.x + b.z))
    assert gradient(a.x**b.y) == Gradient(a.x**b.y)
    # assert gradient(cos(a.x+b.y)*a.z) == None
    assert gradient(cos(a.x*b.y)) == Gradient(cos(a.x*b.y))
    assert gradient(3*cos(q)*a.x*b.x*a.z*a.y+ b.y*b.z + cos(a.x+a.y)*b.z) == \
           (3*a.y*a.z*b.x*cos(q) - b.z*sin(a.x + a.y))*a.i + \
           (3*a.x*a.z*b.x*cos(q) - b.z*sin(a.x + a.y))*a.j + (3*a.x*a.y*b.x*cos(q))*a.k + \
           (3*a.x*a.y*a.z*cos(q))*b.i + b.z*b.j + (b.y + cos(a.x + a.y))*b.k
    # divergence
    assert divergence(a.i*a.x+a.j*a.y+a.z*a.k + b.i*b.x+b.j*b.y+b.z*b.k + c.i*c.x+c.j*c.y+c.z*c.k) == S(9)
    # assert divergence(3*a.i*a.x*cos(a.x+b.z) + a.j*b.x*c.z) == None
    assert divergence(3*a.i*a.x*a.z + b.j*b.x*c.z + 3*a.j*a.z*a.y) == \
            6*a.z + b.x*Dot(b.j, c.k)
    assert divergence(3*cos(q)*a.x*b.x*b.i*c.x) == \
        3*a.x*b.x*cos(q)*Dot(b.i, c.i) + 3*a.x*c.x*cos(q) + 3*b.x*c.x*cos(q)*Dot(b.i, a.i)
    assert divergence(a.x*b.x*c.x*Cross(a.x*a.i, a.y*b.j)) ==\
           a.x*b.x*c.x*Divergence(Cross(a.x*a.i, a.y*b.j)) + \
           b.x*c.x*Dot(Cross(a.x*a.i, a.y*b.j), a.i) + \
           a.x*c.x*Dot(Cross(a.x*a.i, a.y*b.j), b.i) + \
           a.x*b.x*Dot(Cross(a.x*a.i, a.y*b.j), c.i)
    assert divergence(a.x*b.x*c.x*(a.x*a.i + b.x*b.i)) == \
                4*a.x*b.x*c.x +\
                a.x**2*c.x*Dot(a.i, b.i) +\
                a.x**2*b.x*Dot(a.i, c.i) +\
                b.x**2*c.x*Dot(b.i, a.i) +\
                a.x*b.x**2*Dot(b.i, c.i)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\common\test_read_errors.py ===
"""
Tests that work on the Python, C and PyArrow engines but do not have a
specific classification into the other test modules.
"""
import codecs
import csv
from io import StringIO
import os
from pathlib import Path

import numpy as np
import pytest

from pandas.compat import PY311
from pandas.errors import (
    EmptyDataError,
    ParserError,
    ParserWarning,
)

from pandas import DataFrame
import pandas._testing as tm

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


def test_empty_decimal_marker(all_parsers):
    data = """A|B|C
1|2,334|5
10|13|10.
"""
    # Parsers support only length-1 decimals
    msg = "Only length-1 decimal markers supported"
    parser = all_parsers

    if parser.engine == "pyarrow":
        msg = (
            "only single character unicode strings can be "
            "converted to Py_UCS4, got length 0"
        )

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), decimal="")


def test_bad_stream_exception(all_parsers, csv_dir_path):
    # see gh-13652
    #
    # This test validates that both the Python engine and C engine will
    # raise UnicodeDecodeError instead of C engine raising ParserError
    # and swallowing the exception that caused read to fail.
    path = os.path.join(csv_dir_path, "sauron.SHIFT_JIS.csv")
    codec = codecs.lookup("utf-8")
    utf8 = codecs.lookup("utf-8")
    parser = all_parsers
    msg = "'utf-8' codec can't decode byte"

    # Stream must be binary UTF8.
    with open(path, "rb") as handle, codecs.StreamRecoder(
        handle, utf8.encode, utf8.decode, codec.streamreader, codec.streamwriter
    ) as stream:
        with pytest.raises(UnicodeDecodeError, match=msg):
            parser.read_csv(stream)


def test_malformed(all_parsers):
    # see gh-6607
    parser = all_parsers
    data = """ignore
A,B,C
1,2,3 # comment
1,2,3,4,5
2,3,4
"""
    msg = "Expected 3 fields in line 4, saw 5"
    err = ParserError
    if parser.engine == "pyarrow":
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        err = ValueError
    with pytest.raises(err, match=msg):
        parser.read_csv(StringIO(data), header=1, comment="#")


@pytest.mark.parametrize("nrows", [5, 3, None])
def test_malformed_chunks(all_parsers, nrows):
    data = """ignore
A,B,C
skip
1,2,3
3,5,10 # comment
1,2,3,4,5
2,3,4
"""
    parser = all_parsers

    if parser.engine == "pyarrow":
        msg = "The 'iterator' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data),
                header=1,
                comment="#",
                iterator=True,
                chunksize=1,
                skiprows=[2],
            )
        return

    msg = "Expected 3 fields in line 6, saw 5"
    with parser.read_csv(
        StringIO(data), header=1, comment="#", iterator=True, chunksize=1, skiprows=[2]
    ) as reader:
        with pytest.raises(ParserError, match=msg):
            reader.read(nrows)


@xfail_pyarrow  # does not raise
def test_catch_too_many_names(all_parsers):
    # see gh-5156
    data = """\
1,2,3
4,,6
7,8,9
10,11,12\n"""
    parser = all_parsers
    msg = (
        "Too many columns specified: expected 4 and found 3"
        if parser.engine == "c"
        else "Number of passed names did not match "
        "number of header fields in the file"
    )

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), header=0, names=["a", "b", "c", "d"])


@skip_pyarrow  # CSV parse error: Empty CSV file or block
@pytest.mark.parametrize("nrows", [0, 1, 2, 3, 4, 5])
def test_raise_on_no_columns(all_parsers, nrows):
    parser = all_parsers
    data = "\n" * nrows

    msg = "No columns to parse from file"
    with pytest.raises(EmptyDataError, match=msg):
        parser.read_csv(StringIO(data))


def test_unexpected_keyword_parameter_exception(all_parsers):
    # GH-34976
    parser = all_parsers

    msg = "{}\\(\\) got an unexpected keyword argument 'foo'"
    with pytest.raises(TypeError, match=msg.format("read_csv")):
        parser.read_csv("foo.csv", foo=1)
    with pytest.raises(TypeError, match=msg.format("read_table")):
        parser.read_table("foo.tsv", foo=1)


def test_suppress_error_output(all_parsers):
    # see gh-15925
    parser = all_parsers
    data = "a\n1\n1,2,3\n4\n5,6,7"
    expected = DataFrame({"a": [1, 4]})

    result = parser.read_csv(StringIO(data), on_bad_lines="skip")
    tm.assert_frame_equal(result, expected)


def test_error_bad_lines(all_parsers):
    # see gh-15925
    parser = all_parsers
    data = "a\n1\n1,2,3\n4\n5,6,7"

    msg = "Expected 1 fields in line 3, saw 3"

    if parser.engine == "pyarrow":
        # "CSV parse error: Expected 1 columns, got 3: 1,2,3"
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

    with pytest.raises(ParserError, match=msg):
        parser.read_csv(StringIO(data), on_bad_lines="error")


def test_warn_bad_lines(all_parsers):
    # see gh-15925
    parser = all_parsers
    data = "a\n1\n1,2,3\n4\n5,6,7"
    expected = DataFrame({"a": [1, 4]})
    match_msg = "Skipping line"

    expected_warning = ParserWarning
    if parser.engine == "pyarrow":
        match_msg = "Expected 1 columns, but found 3: 1,2,3"
        expected_warning = (ParserWarning, DeprecationWarning)

    with tm.assert_produces_warning(
        expected_warning, match=match_msg, check_stacklevel=False
    ):
        result = parser.read_csv(StringIO(data), on_bad_lines="warn")
    tm.assert_frame_equal(result, expected)


def test_read_csv_wrong_num_columns(all_parsers):
    # Too few columns.
    data = """A,B,C,D,E,F
1,2,3,4,5,6
6,7,8,9,10,11,12
11,12,13,14,15,16
"""
    parser = all_parsers
    msg = "Expected 6 fields in line 3, saw 7"

    if parser.engine == "pyarrow":
        # Expected 6 columns, got 7: 6,7,8,9,10,11,12
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

    with pytest.raises(ParserError, match=msg):
        parser.read_csv(StringIO(data))


def test_null_byte_char(request, all_parsers):
    # see gh-2741
    data = "\x00,foo"
    names = ["a", "b"]
    parser = all_parsers

    if parser.engine == "c" or (parser.engine == "python" and PY311):
        if parser.engine == "python" and PY311:
            request.applymarker(
                pytest.mark.xfail(
                    reason="In Python 3.11, this is read as an empty character not null"
                )
            )
        expected = DataFrame([[np.nan, "foo"]], columns=names)
        out = parser.read_csv(StringIO(data), names=names)
        tm.assert_frame_equal(out, expected)
    else:
        if parser.engine == "pyarrow":
            # CSV parse error: Empty CSV file or block: "
            # cannot infer number of columns"
            pytest.skip(reason="https://github.com/apache/arrow/issues/38676")
        else:
            msg = "NULL byte detected"
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), names=names)


@pytest.mark.filterwarnings("always::ResourceWarning")
def test_open_file(request, all_parsers):
    # GH 39024
    parser = all_parsers

    msg = "Could not determine delimiter"
    err = csv.Error
    if parser.engine == "c":
        msg = "the 'c' engine does not support sep=None with delim_whitespace=False"
        err = ValueError
    elif parser.engine == "pyarrow":
        msg = (
            "the 'pyarrow' engine does not support sep=None with delim_whitespace=False"
        )
        err = ValueError

    with tm.ensure_clean() as path:
        file = Path(path)
        file.write_bytes(b"\xe4\na\n1")

        with tm.assert_produces_warning(None):
            # should not trigger a ResourceWarning
            with pytest.raises(err, match=msg):
                parser.read_csv(file, sep=None, encoding_errors="replace")


def test_invalid_on_bad_line(all_parsers):
    parser = all_parsers
    data = "a\n1\n1,2,3\n4\n5,6,7"
    with pytest.raises(ValueError, match="Argument abc is invalid for on_bad_lines"):
        parser.read_csv(StringIO(data), on_bad_lines="abc")


def test_bad_header_uniform_error(all_parsers):
    parser = all_parsers
    data = "+++123456789...\ncol1,col2,col3,col4\n1,2,3,4\n"
    msg = "Expected 2 fields in line 2, saw 4"
    if parser.engine == "c":
        msg = (
            "Could not construct index. Requested to use 1 "
            "number of columns, but 3 left to parse."
        )
    elif parser.engine == "pyarrow":
        # "CSV parse error: Expected 1 columns, got 4: col1,col2,col3,col4"
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

    with pytest.raises(ParserError, match=msg):
        parser.read_csv(StringIO(data), index_col=0, on_bad_lines="error")


def test_on_bad_lines_warn_correct_formatting(all_parsers):
    # see gh-15925
    parser = all_parsers
    data = """1,2
a,b
a,b,c
a,b,d
a,b
"""
    expected = DataFrame({"1": "a", "2": ["b"] * 2})
    match_msg = "Skipping line"

    expected_warning = ParserWarning
    if parser.engine == "pyarrow":
        match_msg = "Expected 2 columns, but found 3: a,b,c"
        expected_warning = (ParserWarning, DeprecationWarning)

    with tm.assert_produces_warning(
        expected_warning, match=match_msg, check_stacklevel=False
    ):
        result = parser.read_csv(StringIO(data), on_bad_lines="warn")
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\greenlet\tests\leakcheck.py ===
# Copyright (c) 2018 gevent community
# Copyright (c) 2021 greenlet community
#
# This was originally part of gevent's test suite. The main author
# (Jason Madden) vendored a copy of it into greenlet.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from __future__ import print_function

import os
import sys
import gc

from functools import wraps
import unittest


import objgraph

# graphviz 0.18 (Nov 7 2021), available only on Python 3.6 and newer,
# has added type hints (sigh). It wants to use ``typing.Literal`` for
# some stuff, but that's only available on Python 3.9+. If that's not
# found, it creates a ``unittest.mock.MagicMock`` object and annotates
# with that. These are GC'able objects, and doing almost *anything*
# with them results in an explosion of objects. For example, trying to
# compare them for equality creates new objects. This causes our
# leakchecks to fail, with reports like:
#
# greenlet.tests.leakcheck.LeakCheckError: refcount increased by [337, 1333, 343, 430, 530, 643, 769]
# _Call          1820      +546
# dict           4094       +76
# MagicProxy      585       +73
# tuple          2693       +66
# _CallList        24        +3
# weakref        1441        +1
# function       5996        +1
# type            736        +1
# cell            592        +1
# MagicMock         8        +1
#
# To avoid this, we *could* filter this type of object out early. In
# principle it could leak, but we don't use mocks in greenlet, so it
# doesn't leak from us. However, a further issue is that ``MagicMock``
# objects have subobjects that are also GC'able, like ``_Call``, and
# those create new mocks of their own too. So we'd have to filter them
# as well, and they're not public. That's OK, we can workaround the
# problem by being very careful to never compare by equality or other
# user-defined operators, only using object identity or other builtin
# functions.

RUNNING_ON_GITHUB_ACTIONS = os.environ.get('GITHUB_ACTIONS')
RUNNING_ON_TRAVIS = os.environ.get('TRAVIS') or RUNNING_ON_GITHUB_ACTIONS
RUNNING_ON_APPVEYOR = os.environ.get('APPVEYOR')
RUNNING_ON_CI = RUNNING_ON_TRAVIS or RUNNING_ON_APPVEYOR
RUNNING_ON_MANYLINUX = os.environ.get('GREENLET_MANYLINUX')
SKIP_LEAKCHECKS = RUNNING_ON_MANYLINUX or os.environ.get('GREENLET_SKIP_LEAKCHECKS')
SKIP_FAILING_LEAKCHECKS = os.environ.get('GREENLET_SKIP_FAILING_LEAKCHECKS')
ONLY_FAILING_LEAKCHECKS = os.environ.get('GREENLET_ONLY_FAILING_LEAKCHECKS')

def ignores_leakcheck(func):
    """
    Ignore the given object during leakchecks.

    Can be applied to a method, in which case the method will run, but
    will not be subject to leak checks.

    If applied to a class, the entire class will be skipped during leakchecks. This
    is intended to be used for classes that are very slow and cause problems such as
    test timeouts; typically it will be used for classes that are subclasses of a base
    class and specify variants of behaviour (such as pool sizes).
    """
    func.ignore_leakcheck = True
    return func

def fails_leakcheck(func):
    """
    Mark that the function is known to leak.
    """
    func.fails_leakcheck = True
    if SKIP_FAILING_LEAKCHECKS:
        func = unittest.skip("Skipping known failures")(func)
    return func

class LeakCheckError(AssertionError):
    pass

if hasattr(sys, 'getobjects'):
    # In a Python build with ``--with-trace-refs``, make objgraph
    # trace *all* the objects, not just those that are tracked by the
    # GC
    class _MockGC(object):
        def get_objects(self):
            return sys.getobjects(0) # pylint:disable=no-member
        def __getattr__(self, name):
            return getattr(gc, name)
    objgraph.gc = _MockGC()
    fails_strict_leakcheck = fails_leakcheck
else:
    def fails_strict_leakcheck(func):
        """
        Decorator for a function that is known to fail when running
        strict (``sys.getobjects()``) leakchecks.

        This type of leakcheck finds all objects, even those, such as
        strings, which are not tracked by the garbage collector.
        """
        return func

class ignores_types_in_strict_leakcheck(object):
    def __init__(self, types):
        self.types = types
    def __call__(self, func):
        func.leakcheck_ignore_types = self.types
        return func

class _RefCountChecker(object):

    # Some builtin things that we ignore
    # XXX: Those things were ignored by gevent, but they're important here,
    # presumably.
    IGNORED_TYPES = () #(tuple, dict, types.FrameType, types.TracebackType)

    def __init__(self, testcase, function):
        self.testcase = testcase
        self.function = function
        self.deltas = []
        self.peak_stats = {}
        self.ignored_types = ()

        # The very first time we are called, we have already been
        # self.setUp() by the test runner, so we don't need to do it again.
        self.needs_setUp = False

    def _include_object_p(self, obj):
        # pylint:disable=too-many-return-statements
        #
        # See the comment block at the top. We must be careful to
        # avoid invoking user-defined operations.
        if obj is self:
            return False
        kind = type(obj)
        # ``self._include_object_p == obj`` returns NotImplemented
        # for non-function objects, which causes the interpreter
        # to try to reverse the order of arguments...which leads
        # to the explosion of mock objects. We don't want that, so we implement
        # the check manually.
        if kind == type(self._include_object_p):
            try:
                # pylint:disable=not-callable
                exact_method_equals = self._include_object_p.__eq__(obj)
            except AttributeError:
                # Python 2.7 methods may only have __cmp__, and that raises a
                # TypeError for non-method arguments
                # pylint:disable=no-member
                exact_method_equals = self._include_object_p.__cmp__(obj) == 0

            if exact_method_equals is not NotImplemented and exact_method_equals:
                return False

        # Similarly, we need to check identity in our __dict__ to avoid mock explosions.
        for x in self.__dict__.values():
            if obj is x:
                return False


        if kind in self.ignored_types or kind in self.IGNORED_TYPES:
            return False

        return True

    def _growth(self):
        return objgraph.growth(limit=None, peak_stats=self.peak_stats,
                               filter=self._include_object_p)

    def _report_diff(self, growth):
        if not growth:
            return "<Unable to calculate growth>"

        lines = []
        width = max(len(name) for name, _, _ in growth)
        for name, count, delta in growth:
            lines.append('%-*s%9d %+9d' % (width, name, count, delta))

        diff = '\n'.join(lines)
        return diff


    def _run_test(self, args, kwargs):
        gc_enabled = gc.isenabled()
        gc.disable()

        if self.needs_setUp:
            self.testcase.setUp()
            self.testcase.skipTearDown = False
        try:
            self.function(self.testcase, *args, **kwargs)
        finally:
            self.testcase.tearDown()
            self.testcase.doCleanups()
            self.testcase.skipTearDown = True
            self.needs_setUp = True
            if gc_enabled:
                gc.enable()

    def _growth_after(self):
        # Grab post snapshot
        # pylint:disable=no-member
        if 'urlparse' in sys.modules:
            sys.modules['urlparse'].clear_cache()
        if 'urllib.parse' in sys.modules:
            sys.modules['urllib.parse'].clear_cache()

        return self._growth()

    def _check_deltas(self, growth):
        # Return false when we have decided there is no leak,
        # true if we should keep looping, raises an assertion
        # if we have decided there is a leak.

        deltas = self.deltas
        if not deltas:
            # We haven't run yet, no data, keep looping
            return True

        if gc.garbage:
            raise LeakCheckError("Generated uncollectable garbage %r" % (gc.garbage,))


        # the following configurations are classified as "no leak"
        # [0, 0]
        # [x, 0, 0]
        # [... a, b, c, d]  where a+b+c+d = 0
        #
        # the following configurations are classified as "leak"
        # [... z, z, z]  where z > 0

        if deltas[-2:] == [0, 0] and len(deltas) in (2, 3):
            return False

        if deltas[-3:] == [0, 0, 0]:
            return False

        if len(deltas) >= 4 and sum(deltas[-4:]) == 0:
            return False

        if len(deltas) >= 3 and deltas[-1] > 0 and deltas[-1] == deltas[-2] and deltas[-2] == deltas[-3]:
            diff = self._report_diff(growth)
            raise LeakCheckError('refcount increased by %r\n%s' % (deltas, diff))

        # OK, we don't know for sure yet. Let's search for more
        if sum(deltas[-3:]) <= 0 or sum(deltas[-4:]) <= 0 or deltas[-4:].count(0) >= 2:
            # this is suspicious, so give a few more runs
            limit = 11
        else:
            limit = 7
        if len(deltas) >= limit:
            raise LeakCheckError('refcount increased by %r\n%s'
                                 % (deltas,
                                    self._report_diff(growth)))

        # We couldn't decide yet, keep going
        return True

    def __call__(self, args, kwargs):
        for _ in range(3):
            gc.collect()

        expect_failure = getattr(self.function, 'fails_leakcheck', False)
        if expect_failure:
            self.testcase.expect_greenlet_leak = True
        self.ignored_types = getattr(self.function, "leakcheck_ignore_types", ())

        # Capture state before; the incremental will be
        # updated by each call to _growth_after
        growth = self._growth()

        try:
            while self._check_deltas(growth):
                self._run_test(args, kwargs)

                growth = self._growth_after()

                self.deltas.append(sum((stat[2] for stat in growth)))
        except LeakCheckError:
            if not expect_failure:
                raise
        else:
            if expect_failure:
                raise LeakCheckError("Expected %s to leak but it did not." % (self.function,))

def wrap_refcount(method):
    if getattr(method, 'ignore_leakcheck', False) or SKIP_LEAKCHECKS:
        return method

    @wraps(method)
    def wrapper(self, *args, **kwargs): # pylint:disable=too-many-branches
        if getattr(self, 'ignore_leakcheck', False):
            raise unittest.SkipTest("This class ignored during leakchecks")
        if ONLY_FAILING_LEAKCHECKS and not getattr(method, 'fails_leakcheck', False):
            raise unittest.SkipTest("Only running tests that fail leakchecks.")
        return _RefCountChecker(self, method)(args, kwargs)

    return wrapper

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\_backends\__init__.py ===
def f2py_build_generator(name):
    if name == "meson":
        from ._meson import MesonBackend
        return MesonBackend
    elif name == "distutils":
        from ._distutils import DistutilsBackend
        return DistutilsBackend
    else:
        raise ValueError(f"Unknown backend: {name}")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\TypeInfoValue.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

class TypeInfoValue(object):
    NONE = 0
    tensor_type = 1
    sequence_type = 2
    map_type = 3

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\array_algos\__init__.py ===
"""
core.array_algos is for algorithms that operate on ndarray and ExtensionArray.
These should:

- Assume that any Index, Series, or DataFrame objects have already been unwrapped.
- Assume that any list arguments have already been cast to ndarray/EA.
- Not depend on Index, Series, or DataFrame, nor import any of these.
- May dispatch to ExtensionArray methods, but should not import from core.arrays.
"""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\formats\__init__.py ===
# ruff: noqa: TCH004
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # import modules that have public classes/functions
    from pandas.io.formats import style

    # and mark only those modules as public
    __all__ = ["style"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\parsers\__init__.py ===
from pandas.io.parsers.readers import (
    TextFileReader,
    TextParser,
    read_csv,
    read_fwf,
    read_table,
)

__all__ = ["TextFileReader", "TextParser", "read_csv", "read_fwf", "read_table"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\test_cumulative.py ===
import numpy as np
import pytest

from pandas.errors import UnsupportedFunctionCall
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm


@pytest.fixture(
    params=[np.int32, np.int64, np.float32, np.float64, "Int64", "Float64"],
    ids=["np.int32", "np.int64", "np.float32", "np.float64", "Int64", "Float64"],
)
def dtypes_for_minmax(request):
    """
    Fixture of dtypes with min and max values used for testing
    cummin and cummax
    """
    dtype = request.param

    np_type = dtype
    if dtype == "Int64":
        np_type = np.int64
    elif dtype == "Float64":
        np_type = np.float64

    min_val = (
        np.iinfo(np_type).min
        if np.dtype(np_type).kind == "i"
        else np.finfo(np_type).min
    )
    max_val = (
        np.iinfo(np_type).max
        if np.dtype(np_type).kind == "i"
        else np.finfo(np_type).max
    )

    return (dtype, min_val, max_val)


def test_groupby_cumprod():
    # GH 4095
    df = DataFrame({"key": ["b"] * 10, "value": 2})

    actual = df.groupby("key")["value"].cumprod()
    expected = df.groupby("key", group_keys=False)["value"].apply(lambda x: x.cumprod())
    expected.name = "value"
    tm.assert_series_equal(actual, expected)

    df = DataFrame({"key": ["b"] * 100, "value": 2})
    df["value"] = df["value"].astype(float)
    actual = df.groupby("key")["value"].cumprod()
    expected = df.groupby("key", group_keys=False)["value"].apply(lambda x: x.cumprod())
    expected.name = "value"
    tm.assert_series_equal(actual, expected)


@pytest.mark.skip_ubsan
def test_groupby_cumprod_overflow():
    # GH#37493 if we overflow we return garbage consistent with numpy
    df = DataFrame({"key": ["b"] * 4, "value": 100_000})
    actual = df.groupby("key")["value"].cumprod()
    expected = Series(
        [100_000, 10_000_000_000, 1_000_000_000_000_000, 7766279631452241920],
        name="value",
    )
    tm.assert_series_equal(actual, expected)

    numpy_result = df.groupby("key", group_keys=False)["value"].apply(
        lambda x: x.cumprod()
    )
    numpy_result.name = "value"
    tm.assert_series_equal(actual, numpy_result)


def test_groupby_cumprod_nan_influences_other_columns():
    # GH#48064
    df = DataFrame(
        {
            "a": 1,
            "b": [1, np.nan, 2],
            "c": [1, 2, 3.0],
        }
    )
    result = df.groupby("a").cumprod(numeric_only=True, skipna=False)
    expected = DataFrame({"b": [1, np.nan, np.nan], "c": [1, 2, 6.0]})
    tm.assert_frame_equal(result, expected)


def test_cummin(dtypes_for_minmax):
    dtype = dtypes_for_minmax[0]
    min_val = dtypes_for_minmax[1]

    # GH 15048
    base_df = DataFrame({"A": [1, 1, 1, 1, 2, 2, 2, 2], "B": [3, 4, 3, 2, 2, 3, 2, 1]})
    expected_mins = [3, 3, 3, 2, 2, 2, 2, 1]

    df = base_df.astype(dtype)

    expected = DataFrame({"B": expected_mins}).astype(dtype)
    result = df.groupby("A").cummin()
    tm.assert_frame_equal(result, expected)
    result = df.groupby("A", group_keys=False).B.apply(lambda x: x.cummin()).to_frame()
    tm.assert_frame_equal(result, expected)

    # Test w/ min value for dtype
    df.loc[[2, 6], "B"] = min_val
    df.loc[[1, 5], "B"] = min_val + 1
    expected.loc[[2, 3, 6, 7], "B"] = min_val
    expected.loc[[1, 5], "B"] = min_val + 1  # should not be rounded to min_val
    result = df.groupby("A").cummin()
    tm.assert_frame_equal(result, expected, check_exact=True)
    expected = (
        df.groupby("A", group_keys=False).B.apply(lambda x: x.cummin()).to_frame()
    )
    tm.assert_frame_equal(result, expected, check_exact=True)

    # Test nan in some values
    # Explicit cast to float to avoid implicit cast when setting nan
    base_df = base_df.astype({"B": "float"})
    base_df.loc[[0, 2, 4, 6], "B"] = np.nan
    expected = DataFrame({"B": [np.nan, 4, np.nan, 2, np.nan, 3, np.nan, 1]})
    result = base_df.groupby("A").cummin()
    tm.assert_frame_equal(result, expected)
    expected = (
        base_df.groupby("A", group_keys=False).B.apply(lambda x: x.cummin()).to_frame()
    )
    tm.assert_frame_equal(result, expected)

    # GH 15561
    df = DataFrame({"a": [1], "b": pd.to_datetime(["2001"])})
    expected = Series(pd.to_datetime("2001"), index=[0], name="b")

    result = df.groupby("a")["b"].cummin()
    tm.assert_series_equal(expected, result)

    # GH 15635
    df = DataFrame({"a": [1, 2, 1], "b": [1, 2, 2]})
    result = df.groupby("a").b.cummin()
    expected = Series([1, 2, 1], name="b")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("method", ["cummin", "cummax"])
@pytest.mark.parametrize("dtype", ["UInt64", "Int64", "Float64", "float", "boolean"])
def test_cummin_max_all_nan_column(method, dtype):
    base_df = DataFrame({"A": [1, 1, 1, 1, 2, 2, 2, 2], "B": [np.nan] * 8})
    base_df["B"] = base_df["B"].astype(dtype)
    grouped = base_df.groupby("A")

    expected = DataFrame({"B": [np.nan] * 8}, dtype=dtype)
    result = getattr(grouped, method)()
    tm.assert_frame_equal(expected, result)

    result = getattr(grouped["B"], method)().to_frame()
    tm.assert_frame_equal(expected, result)


def test_cummax(dtypes_for_minmax):
    dtype = dtypes_for_minmax[0]
    max_val = dtypes_for_minmax[2]

    # GH 15048
    base_df = DataFrame({"A": [1, 1, 1, 1, 2, 2, 2, 2], "B": [3, 4, 3, 2, 2, 3, 2, 1]})
    expected_maxs = [3, 4, 4, 4, 2, 3, 3, 3]

    df = base_df.astype(dtype)

    expected = DataFrame({"B": expected_maxs}).astype(dtype)
    result = df.groupby("A").cummax()
    tm.assert_frame_equal(result, expected)
    result = df.groupby("A", group_keys=False).B.apply(lambda x: x.cummax()).to_frame()
    tm.assert_frame_equal(result, expected)

    # Test w/ max value for dtype
    df.loc[[2, 6], "B"] = max_val
    expected.loc[[2, 3, 6, 7], "B"] = max_val
    result = df.groupby("A").cummax()
    tm.assert_frame_equal(result, expected)
    expected = (
        df.groupby("A", group_keys=False).B.apply(lambda x: x.cummax()).to_frame()
    )
    tm.assert_frame_equal(result, expected)

    # Test nan in some values
    # Explicit cast to float to avoid implicit cast when setting nan
    base_df = base_df.astype({"B": "float"})
    base_df.loc[[0, 2, 4, 6], "B"] = np.nan
    expected = DataFrame({"B": [np.nan, 4, np.nan, 4, np.nan, 3, np.nan, 3]})
    result = base_df.groupby("A").cummax()
    tm.assert_frame_equal(result, expected)
    expected = (
        base_df.groupby("A", group_keys=False).B.apply(lambda x: x.cummax()).to_frame()
    )
    tm.assert_frame_equal(result, expected)

    # GH 15561
    df = DataFrame({"a": [1], "b": pd.to_datetime(["2001"])})
    expected = Series(pd.to_datetime("2001"), index=[0], name="b")

    result = df.groupby("a")["b"].cummax()
    tm.assert_series_equal(expected, result)

    # GH 15635
    df = DataFrame({"a": [1, 2, 1], "b": [2, 1, 1]})
    result = df.groupby("a").b.cummax()
    expected = Series([2, 1, 2], name="b")
    tm.assert_series_equal(result, expected)


def test_cummax_i8_at_implementation_bound():
    # the minimum value used to be treated as NPY_NAT+1 instead of NPY_NAT
    #  for int64 dtype GH#46382
    ser = Series([pd.NaT._value + n for n in range(5)])
    df = DataFrame({"A": 1, "B": ser, "C": ser._values.view("M8[ns]")})
    gb = df.groupby("A")

    res = gb.cummax()
    exp = df[["B", "C"]]
    tm.assert_frame_equal(res, exp)


@pytest.mark.parametrize("method", ["cummin", "cummax"])
@pytest.mark.parametrize("dtype", ["float", "Int64", "Float64"])
@pytest.mark.parametrize(
    "groups,expected_data",
    [
        ([1, 1, 1], [1, None, None]),
        ([1, 2, 3], [1, None, 2]),
        ([1, 3, 3], [1, None, None]),
    ],
)
def test_cummin_max_skipna(method, dtype, groups, expected_data):
    # GH-34047
    df = DataFrame({"a": Series([1, None, 2], dtype=dtype)})
    orig = df.copy()
    gb = df.groupby(groups)["a"]

    result = getattr(gb, method)(skipna=False)
    expected = Series(expected_data, dtype=dtype, name="a")

    # check we didn't accidentally alter df
    tm.assert_frame_equal(df, orig)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("method", ["cummin", "cummax"])
def test_cummin_max_skipna_multiple_cols(method):
    # Ensure missing value in "a" doesn't cause "b" to be nan-filled
    df = DataFrame({"a": [np.nan, 2.0, 2.0], "b": [2.0, 2.0, 2.0]})
    gb = df.groupby([1, 1, 1])[["a", "b"]]

    result = getattr(gb, method)(skipna=False)
    expected = DataFrame({"a": [np.nan, np.nan, np.nan], "b": [2.0, 2.0, 2.0]})

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("func", ["cumprod", "cumsum"])
def test_numpy_compat(func):
    # see gh-12811
    df = DataFrame({"A": [1, 2, 1], "B": [1, 2, 3]})
    g = df.groupby("A")

    msg = "numpy operations are not valid with groupby"

    with pytest.raises(UnsupportedFunctionCall, match=msg):
        getattr(g, func)(1, 2, 3)
    with pytest.raises(UnsupportedFunctionCall, match=msg):
        getattr(g, func)(foo=1)


@td.skip_if_32bit
@pytest.mark.parametrize("method", ["cummin", "cummax"])
@pytest.mark.parametrize(
    "dtype,val", [("UInt64", np.iinfo("uint64").max), ("Int64", 2**53 + 1)]
)
def test_nullable_int_not_cast_as_float(method, dtype, val):
    data = [val, pd.NA]
    df = DataFrame({"grp": [1, 1], "b": data}, dtype=dtype)
    grouped = df.groupby("grp")

    result = grouped.transform(method)
    expected = DataFrame({"b": data}, dtype=dtype)

    tm.assert_frame_equal(result, expected)


def test_cython_api2():
    # this takes the fast apply path

    # cumsum (GH5614)
    df = DataFrame([[1, 2, np.nan], [1, np.nan, 9], [3, 4, 9]], columns=["A", "B", "C"])
    expected = DataFrame([[2, np.nan], [np.nan, 9], [4, 9]], columns=["B", "C"])
    result = df.groupby("A").cumsum()
    tm.assert_frame_equal(result, expected)

    # GH 5755 - cumsum is a transformer and should ignore as_index
    result = df.groupby("A", as_index=False).cumsum()
    tm.assert_frame_equal(result, expected)

    # GH 13994
    msg = "DataFrameGroupBy.cumsum with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("A").cumsum(axis=1)
    expected = df.cumsum(axis=1)
    tm.assert_frame_equal(result, expected)

    msg = "DataFrameGroupBy.cumprod with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("A").cumprod(axis=1)
    expected = df.cumprod(axis=1)
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\test_assert_index_equal.py ===
import numpy as np
import pytest

from pandas import (
    NA,
    Categorical,
    CategoricalIndex,
    Index,
    MultiIndex,
    NaT,
    RangeIndex,
)
import pandas._testing as tm


def test_index_equal_levels_mismatch():
    msg = """Index are different

Index levels are different
\\[left\\]:  1, Index\\(\\[1, 2, 3\\], dtype='int64'\\)
\\[right\\]: 2, MultiIndex\\(\\[\\('A', 1\\),
            \\('A', 2\\),
            \\('B', 3\\),
            \\('B', 4\\)\\],
           \\)"""

    idx1 = Index([1, 2, 3])
    idx2 = MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 3), ("B", 4)])

    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, exact=False)


def test_index_equal_values_mismatch(check_exact):
    msg = """MultiIndex level \\[1\\] are different

MultiIndex level \\[1\\] values are different \\(25\\.0 %\\)
\\[left\\]:  Index\\(\\[2, 2, 3, 4\\], dtype='int64'\\)
\\[right\\]: Index\\(\\[1, 2, 3, 4\\], dtype='int64'\\)"""

    idx1 = MultiIndex.from_tuples([("A", 2), ("A", 2), ("B", 3), ("B", 4)])
    idx2 = MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 3), ("B", 4)])

    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, check_exact=check_exact)


def test_index_equal_length_mismatch(check_exact):
    msg = """Index are different

Index length are different
\\[left\\]:  3, Index\\(\\[1, 2, 3\\], dtype='int64'\\)
\\[right\\]: 4, Index\\(\\[1, 2, 3, 4\\], dtype='int64'\\)"""

    idx1 = Index([1, 2, 3])
    idx2 = Index([1, 2, 3, 4])

    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, check_exact=check_exact)


@pytest.mark.parametrize("exact", [False, "equiv"])
def test_index_equal_class(exact):
    idx1 = Index([0, 1, 2])
    idx2 = RangeIndex(3)

    tm.assert_index_equal(idx1, idx2, exact=exact)


def test_int_float_index_equal_class_mismatch(check_exact):
    msg = """Index are different

Attribute "inferred_type" are different
\\[left\\]:  integer
\\[right\\]: floating"""

    idx1 = Index([1, 2, 3])
    idx2 = Index([1, 2, 3], dtype=np.float64)

    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, exact=True, check_exact=check_exact)


def test_range_index_equal_class_mismatch(check_exact):
    msg = """Index are different

Index classes are different
\\[left\\]:  Index\\(\\[1, 2, 3\\], dtype='int64'\\)
\\[right\\]: """

    idx1 = Index([1, 2, 3])
    idx2 = RangeIndex(range(3))

    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, exact=True, check_exact=check_exact)


def test_index_equal_values_close(check_exact):
    idx1 = Index([1, 2, 3.0])
    idx2 = Index([1, 2, 3.0000000001])

    if check_exact:
        msg = """Index are different

Index values are different \\(33\\.33333 %\\)
\\[left\\]:  Index\\(\\[1.0, 2.0, 3.0], dtype='float64'\\)
\\[right\\]: Index\\(\\[1.0, 2.0, 3.0000000001\\], dtype='float64'\\)"""

        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(idx1, idx2, check_exact=check_exact)
    else:
        tm.assert_index_equal(idx1, idx2, check_exact=check_exact)


def test_index_equal_values_less_close(check_exact, rtol):
    idx1 = Index([1, 2, 3.0])
    idx2 = Index([1, 2, 3.0001])
    kwargs = {"check_exact": check_exact, "rtol": rtol}

    if check_exact or rtol < 0.5e-3:
        msg = """Index are different

Index values are different \\(33\\.33333 %\\)
\\[left\\]:  Index\\(\\[1.0, 2.0, 3.0], dtype='float64'\\)
\\[right\\]: Index\\(\\[1.0, 2.0, 3.0001\\], dtype='float64'\\)"""

        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(idx1, idx2, **kwargs)
    else:
        tm.assert_index_equal(idx1, idx2, **kwargs)


def test_index_equal_values_too_far(check_exact, rtol):
    idx1 = Index([1, 2, 3])
    idx2 = Index([1, 2, 4])
    kwargs = {"check_exact": check_exact, "rtol": rtol}

    msg = """Index are different

Index values are different \\(33\\.33333 %\\)
\\[left\\]:  Index\\(\\[1, 2, 3\\], dtype='int64'\\)
\\[right\\]: Index\\(\\[1, 2, 4\\], dtype='int64'\\)"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, **kwargs)


@pytest.mark.parametrize("check_order", [True, False])
def test_index_equal_value_order_mismatch(check_exact, rtol, check_order):
    idx1 = Index([1, 2, 3])
    idx2 = Index([3, 2, 1])

    msg = """Index are different

Index values are different \\(66\\.66667 %\\)
\\[left\\]:  Index\\(\\[1, 2, 3\\], dtype='int64'\\)
\\[right\\]: Index\\(\\[3, 2, 1\\], dtype='int64'\\)"""

    if check_order:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(
                idx1, idx2, check_exact=check_exact, rtol=rtol, check_order=True
            )
    else:
        tm.assert_index_equal(
            idx1, idx2, check_exact=check_exact, rtol=rtol, check_order=False
        )


def test_index_equal_level_values_mismatch(check_exact, rtol):
    idx1 = MultiIndex.from_tuples([("A", 2), ("A", 2), ("B", 3), ("B", 4)])
    idx2 = MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 3), ("B", 4)])
    kwargs = {"check_exact": check_exact, "rtol": rtol}

    msg = """MultiIndex level \\[1\\] are different

MultiIndex level \\[1\\] values are different \\(25\\.0 %\\)
\\[left\\]:  Index\\(\\[2, 2, 3, 4\\], dtype='int64'\\)
\\[right\\]: Index\\(\\[1, 2, 3, 4\\], dtype='int64'\\)"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, **kwargs)


@pytest.mark.parametrize(
    "name1,name2",
    [(None, "x"), ("x", "x"), (np.nan, np.nan), (NaT, NaT), (np.nan, NaT)],
)
def test_index_equal_names(name1, name2):
    idx1 = Index([1, 2, 3], name=name1)
    idx2 = Index([1, 2, 3], name=name2)

    if name1 == name2 or name1 is name2:
        tm.assert_index_equal(idx1, idx2)
    else:
        name1 = "'x'" if name1 == "x" else name1
        name2 = "'x'" if name2 == "x" else name2
        msg = f"""Index are different

Attribute "names" are different
\\[left\\]:  \\[{name1}\\]
\\[right\\]: \\[{name2}\\]"""

        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(idx1, idx2)


def test_index_equal_category_mismatch(check_categorical, using_infer_string):
    if using_infer_string:
        dtype = "str"
    else:
        dtype = "object"
    msg = f"""Index are different

Attribute "dtype" are different
\\[left\\]:  CategoricalDtype\\(categories=\\['a', 'b'\\], ordered=False, \
categories_dtype={dtype}\\)
\\[right\\]: CategoricalDtype\\(categories=\\['a', 'b', 'c'\\], \
ordered=False, categories_dtype={dtype}\\)"""

    idx1 = Index(Categorical(["a", "b"]))
    idx2 = Index(Categorical(["a", "b"], categories=["a", "b", "c"]))

    if check_categorical:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(idx1, idx2, check_categorical=check_categorical)
    else:
        tm.assert_index_equal(idx1, idx2, check_categorical=check_categorical)


@pytest.mark.parametrize("exact", [False, True])
def test_index_equal_range_categories(check_categorical, exact):
    # GH41263
    msg = """\
Index are different

Index classes are different
\\[left\\]:  RangeIndex\\(start=0, stop=10, step=1\\)
\\[right\\]: Index\\(\\[0, 1, 2, 3, 4, 5, 6, 7, 8, 9\\], dtype='int64'\\)"""

    rcat = CategoricalIndex(RangeIndex(10))
    icat = CategoricalIndex(list(range(10)))

    if check_categorical and exact:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(rcat, icat, check_categorical=True, exact=True)
    else:
        tm.assert_index_equal(
            rcat, icat, check_categorical=check_categorical, exact=exact
        )


def test_assert_index_equal_different_inferred_types():
    # GH#31884
    msg = """\
Index are different

Attribute "inferred_type" are different
\\[left\\]:  mixed
\\[right\\]: datetime"""

    idx1 = Index([NA, np.datetime64("nat")])
    idx2 = Index([NA, NaT])
    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2)


def test_assert_index_equal_different_names_check_order_false():
    # GH#47328
    idx1 = Index([1, 3], name="a")
    idx2 = Index([3, 1], name="b")
    with pytest.raises(AssertionError, match='"names" are different'):
        tm.assert_index_equal(idx1, idx2, check_order=False, check_names=True)


def test_assert_index_equal_mixed_dtype():
    # GH#39168
    idx = Index(["foo", "bar", 42])
    tm.assert_index_equal(idx, idx, check_order=False)


def test_assert_index_equal_ea_dtype_order_false(any_numeric_ea_dtype):
    # GH#47207
    idx1 = Index([1, 3], dtype=any_numeric_ea_dtype)
    idx2 = Index([3, 1], dtype=any_numeric_ea_dtype)
    tm.assert_index_equal(idx1, idx2, check_order=False)


def test_assert_index_equal_object_ints_order_false():
    # GH#47207
    idx1 = Index([1, 3], dtype="object")
    idx2 = Index([3, 1], dtype="object")
    tm.assert_index_equal(idx1, idx2, check_order=False)


@pytest.mark.parametrize("check_categorical", [True, False])
@pytest.mark.parametrize("check_names", [True, False])
def test_assert_ea_index_equal_non_matching_na(check_names, check_categorical):
    # GH#48608
    idx1 = Index([1, 2], dtype="Int64")
    idx2 = Index([1, NA], dtype="Int64")
    with pytest.raises(AssertionError, match="50.0 %"):
        tm.assert_index_equal(
            idx1, idx2, check_names=check_names, check_categorical=check_categorical
        )


@pytest.mark.parametrize("check_categorical", [True, False])
def test_assert_multi_index_dtype_check_categorical(check_categorical):
    # GH#52126
    idx1 = MultiIndex.from_arrays([Categorical(np.array([1, 2], dtype=np.uint64))])
    idx2 = MultiIndex.from_arrays([Categorical(np.array([1, 2], dtype=np.int64))])
    if check_categorical:
        with pytest.raises(
            AssertionError, match=r"^MultiIndex level \[0\] are different"
        ):
            tm.assert_index_equal(idx1, idx2, check_categorical=check_categorical)
    else:
        tm.assert_index_equal(idx1, idx2, check_categorical=check_categorical)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\dependency_groups\_toml_compat.py ===
try:
    import tomllib
except ImportError:
    try:
        from pip._vendor import tomli as tomllib  # type: ignore[no-redef, unused-ignore]
    except ModuleNotFoundError:  # pragma: no cover
        tomllib = None  # type: ignore[assignment, unused-ignore]

__all__ = ("tomllib",)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\latex\_antlr\__init__.py ===
# *** GENERATED BY `setup.py antlr`, DO NOT EDIT BY HAND ***
#
# Generated from ../LaTeX.g4, derived from latex2sympy
#     latex2sympy is licensed under the MIT license
#     https://github.com/augustt198/latex2sympy/blob/master/LICENSE.txt
#
# Generated with antlr4
#    antlr4 is licensed under the BSD-3-Clause License
#    https://github.com/antlr/antlr4/blob/master/LICENSE.txt

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\benchmarks\bench_limit.py ===
from sympy.core.numbers import oo
from sympy.core.symbol import Symbol
from sympy.series.limits import limit

x = Symbol('x')


def timeit_limit_1x():
    limit(1/x, x, oo)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\type_tests\check_wraps.py ===
# https://github.com/python-trio/trio/issues/2775#issuecomment-1702267589
# (except platform independent...)
import trio
from typing_extensions import assert_type


async def fn(s: trio.SocketStream) -> None:
    result = await s.socket.sendto(b"a", "h")
    assert_type(result, int)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\json\test_json_table_schema_ext_dtype.py ===
"""Tests for ExtensionDtype Table Schema integration."""

from collections import OrderedDict
import datetime as dt
import decimal
from io import StringIO
import json

import pytest

from pandas import (
    NA,
    DataFrame,
    Index,
    array,
    read_json,
)
import pandas._testing as tm
from pandas.core.arrays.integer import Int64Dtype
from pandas.core.arrays.string_ import StringDtype
from pandas.core.series import Series
from pandas.tests.extension.date import (
    DateArray,
    DateDtype,
)
from pandas.tests.extension.decimal.array import (
    DecimalArray,
    DecimalDtype,
)

from pandas.io.json._table_schema import (
    as_json_table_type,
    build_table_schema,
)


class TestBuildSchema:
    def test_build_table_schema(self):
        df = DataFrame(
            {
                "A": DateArray([dt.date(2021, 10, 10)]),
                "B": DecimalArray([decimal.Decimal(10)]),
                "C": array(["pandas"], dtype="string"),
                "D": array([10], dtype="Int64"),
            }
        )
        result = build_table_schema(df, version=False)
        expected = {
            "fields": [
                {"name": "index", "type": "integer"},
                {"name": "A", "type": "any", "extDtype": "DateDtype"},
                {"name": "B", "type": "number", "extDtype": "decimal"},
                {"name": "C", "type": "any", "extDtype": "string"},
                {"name": "D", "type": "integer", "extDtype": "Int64"},
            ],
            "primaryKey": ["index"],
        }
        assert result == expected
        result = build_table_schema(df)
        assert "pandas_version" in result


class TestTableSchemaType:
    @pytest.mark.parametrize(
        "date_data",
        [
            DateArray([dt.date(2021, 10, 10)]),
            DateArray(dt.date(2021, 10, 10)),
            Series(DateArray(dt.date(2021, 10, 10))),
        ],
    )
    def test_as_json_table_type_ext_date_array_dtype(self, date_data):
        assert as_json_table_type(date_data.dtype) == "any"

    def test_as_json_table_type_ext_date_dtype(self):
        assert as_json_table_type(DateDtype()) == "any"

    @pytest.mark.parametrize(
        "decimal_data",
        [
            DecimalArray([decimal.Decimal(10)]),
            Series(DecimalArray([decimal.Decimal(10)])),
        ],
    )
    def test_as_json_table_type_ext_decimal_array_dtype(self, decimal_data):
        assert as_json_table_type(decimal_data.dtype) == "number"

    def test_as_json_table_type_ext_decimal_dtype(self):
        assert as_json_table_type(DecimalDtype()) == "number"

    @pytest.mark.parametrize(
        "string_data",
        [
            array(["pandas"], dtype="string"),
            Series(array(["pandas"], dtype="string")),
        ],
    )
    def test_as_json_table_type_ext_string_array_dtype(self, string_data):
        assert as_json_table_type(string_data.dtype) == "any"

    def test_as_json_table_type_ext_string_dtype(self):
        assert as_json_table_type(StringDtype()) == "any"

    @pytest.mark.parametrize(
        "integer_data",
        [
            array([10], dtype="Int64"),
            Series(array([10], dtype="Int64")),
        ],
    )
    def test_as_json_table_type_ext_integer_array_dtype(self, integer_data):
        assert as_json_table_type(integer_data.dtype) == "integer"

    def test_as_json_table_type_ext_integer_dtype(self):
        assert as_json_table_type(Int64Dtype()) == "integer"


class TestTableOrient:
    @pytest.fixture
    def da(self):
        return DateArray([dt.date(2021, 10, 10)])

    @pytest.fixture
    def dc(self):
        return DecimalArray([decimal.Decimal(10)])

    @pytest.fixture
    def sa(self):
        return array(["pandas"], dtype="string")

    @pytest.fixture
    def ia(self):
        return array([10], dtype="Int64")

    @pytest.fixture
    def df(self, da, dc, sa, ia):
        return DataFrame(
            {
                "A": da,
                "B": dc,
                "C": sa,
                "D": ia,
            }
        )

    def test_build_date_series(self, da):
        s = Series(da, name="a")
        s.index.name = "id"
        result = s.to_json(orient="table", date_format="iso")
        result = json.loads(result, object_pairs_hook=OrderedDict)

        assert "pandas_version" in result["schema"]
        result["schema"].pop("pandas_version")

        fields = [
            {"name": "id", "type": "integer"},
            {"name": "a", "type": "any", "extDtype": "DateDtype"},
        ]

        schema = {"fields": fields, "primaryKey": ["id"]}

        expected = OrderedDict(
            [
                ("schema", schema),
                ("data", [OrderedDict([("id", 0), ("a", "2021-10-10T00:00:00.000")])]),
            ]
        )

        assert result == expected

    def test_build_decimal_series(self, dc):
        s = Series(dc, name="a")
        s.index.name = "id"
        result = s.to_json(orient="table", date_format="iso")
        result = json.loads(result, object_pairs_hook=OrderedDict)

        assert "pandas_version" in result["schema"]
        result["schema"].pop("pandas_version")

        fields = [
            {"name": "id", "type": "integer"},
            {"name": "a", "type": "number", "extDtype": "decimal"},
        ]

        schema = {"fields": fields, "primaryKey": ["id"]}

        expected = OrderedDict(
            [
                ("schema", schema),
                ("data", [OrderedDict([("id", 0), ("a", 10.0)])]),
            ]
        )

        assert result == expected

    def test_build_string_series(self, sa):
        s = Series(sa, name="a")
        s.index.name = "id"
        result = s.to_json(orient="table", date_format="iso")
        result = json.loads(result, object_pairs_hook=OrderedDict)

        assert "pandas_version" in result["schema"]
        result["schema"].pop("pandas_version")

        fields = [
            {"name": "id", "type": "integer"},
            {"name": "a", "type": "any", "extDtype": "string"},
        ]

        schema = {"fields": fields, "primaryKey": ["id"]}

        expected = OrderedDict(
            [
                ("schema", schema),
                ("data", [OrderedDict([("id", 0), ("a", "pandas")])]),
            ]
        )

        assert result == expected

    def test_build_int64_series(self, ia):
        s = Series(ia, name="a")
        s.index.name = "id"
        result = s.to_json(orient="table", date_format="iso")
        result = json.loads(result, object_pairs_hook=OrderedDict)

        assert "pandas_version" in result["schema"]
        result["schema"].pop("pandas_version")

        fields = [
            {"name": "id", "type": "integer"},
            {"name": "a", "type": "integer", "extDtype": "Int64"},
        ]

        schema = {"fields": fields, "primaryKey": ["id"]}

        expected = OrderedDict(
            [
                ("schema", schema),
                ("data", [OrderedDict([("id", 0), ("a", 10)])]),
            ]
        )

        assert result == expected

    def test_to_json(self, df):
        df = df.copy()
        df.index.name = "idx"
        result = df.to_json(orient="table", date_format="iso")
        result = json.loads(result, object_pairs_hook=OrderedDict)

        assert "pandas_version" in result["schema"]
        result["schema"].pop("pandas_version")

        fields = [
            OrderedDict({"name": "idx", "type": "integer"}),
            OrderedDict({"name": "A", "type": "any", "extDtype": "DateDtype"}),
            OrderedDict({"name": "B", "type": "number", "extDtype": "decimal"}),
            OrderedDict({"name": "C", "type": "any", "extDtype": "string"}),
            OrderedDict({"name": "D", "type": "integer", "extDtype": "Int64"}),
        ]

        schema = OrderedDict({"fields": fields, "primaryKey": ["idx"]})
        data = [
            OrderedDict(
                [
                    ("idx", 0),
                    ("A", "2021-10-10T00:00:00.000"),
                    ("B", 10.0),
                    ("C", "pandas"),
                    ("D", 10),
                ]
            )
        ]
        expected = OrderedDict([("schema", schema), ("data", data)])

        assert result == expected

    def test_json_ext_dtype_reading_roundtrip(self):
        # GH#40255
        df = DataFrame(
            {
                "a": Series([2, NA], dtype="Int64"),
                "b": Series([1.5, NA], dtype="Float64"),
                "c": Series([True, NA], dtype="boolean"),
            },
            index=Index([1, NA], dtype="Int64"),
        )
        expected = df.copy()
        data_json = df.to_json(orient="table", indent=4)
        result = read_json(StringIO(data_json), orient="table")
        tm.assert_frame_equal(result, expected)

    def test_json_ext_dtype_reading(self):
        # GH#40255
        data_json = """{
            "schema":{
                "fields":[
                    {
                        "name":"a",
                        "type":"integer",
                        "extDtype":"Int64"
                    }
                ],
            },
            "data":[
                {
                    "a":2
                },
                {
                    "a":null
                }
            ]
        }"""
        result = read_json(StringIO(data_json), orient="table")
        expected = DataFrame({"a": Series([2, NA], dtype="Int64")})
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\numberfields\tests\test_subfield.py ===
"""Tests for the subfield problem and allied problems. """

from sympy.core.numbers import (AlgebraicNumber, I, pi, Rational)
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.external.gmpy import MPQ
from sympy.polys.numberfields.subfield import (
    is_isomorphism_possible,
    field_isomorphism_pslq,
    field_isomorphism,
    primitive_element,
    to_number_field,
)
from sympy.polys.domains import QQ
from sympy.polys.polyerrors import IsomorphismFailed
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.testing.pytest import raises

from sympy.abc import x

Q = Rational


def test_field_isomorphism_pslq():
    a = AlgebraicNumber(I)
    b = AlgebraicNumber(I*sqrt(3))

    raises(NotImplementedError, lambda: field_isomorphism_pslq(a, b))

    a = AlgebraicNumber(sqrt(2))
    b = AlgebraicNumber(sqrt(3))
    c = AlgebraicNumber(sqrt(7))
    d = AlgebraicNumber(sqrt(2) + sqrt(3))
    e = AlgebraicNumber(sqrt(2) + sqrt(3) + sqrt(7))

    assert field_isomorphism_pslq(a, a) == [1, 0]
    assert field_isomorphism_pslq(a, b) is None
    assert field_isomorphism_pslq(a, c) is None
    assert field_isomorphism_pslq(a, d) == [Q(1, 2), 0, -Q(9, 2), 0]
    assert field_isomorphism_pslq(
        a, e) == [Q(1, 80), 0, -Q(1, 2), 0, Q(59, 20), 0]

    assert field_isomorphism_pslq(b, a) is None
    assert field_isomorphism_pslq(b, b) == [1, 0]
    assert field_isomorphism_pslq(b, c) is None
    assert field_isomorphism_pslq(b, d) == [-Q(1, 2), 0, Q(11, 2), 0]
    assert field_isomorphism_pslq(b, e) == [-Q(
        3, 640), 0, Q(67, 320), 0, -Q(297, 160), 0, Q(313, 80), 0]

    assert field_isomorphism_pslq(c, a) is None
    assert field_isomorphism_pslq(c, b) is None
    assert field_isomorphism_pslq(c, c) == [1, 0]
    assert field_isomorphism_pslq(c, d) is None
    assert field_isomorphism_pslq(c, e) == [Q(
        3, 640), 0, -Q(71, 320), 0, Q(377, 160), 0, -Q(469, 80), 0]

    assert field_isomorphism_pslq(d, a) is None
    assert field_isomorphism_pslq(d, b) is None
    assert field_isomorphism_pslq(d, c) is None
    assert field_isomorphism_pslq(d, d) == [1, 0]
    assert field_isomorphism_pslq(d, e) == [-Q(
        3, 640), 0, Q(71, 320), 0, -Q(377, 160), 0, Q(549, 80), 0]

    assert field_isomorphism_pslq(e, a) is None
    assert field_isomorphism_pslq(e, b) is None
    assert field_isomorphism_pslq(e, c) is None
    assert field_isomorphism_pslq(e, d) is None
    assert field_isomorphism_pslq(e, e) == [1, 0]

    f = AlgebraicNumber(3*sqrt(2) + 8*sqrt(7) - 5)

    assert field_isomorphism_pslq(
        f, e) == [Q(3, 80), 0, -Q(139, 80), 0, Q(347, 20), 0, -Q(761, 20), -5]


def test_field_isomorphism():
    assert field_isomorphism(3, sqrt(2)) == [3]

    assert field_isomorphism( I*sqrt(3), I*sqrt(3)/2) == [ 2, 0]
    assert field_isomorphism(-I*sqrt(3), I*sqrt(3)/2) == [-2, 0]

    assert field_isomorphism( I*sqrt(3), -I*sqrt(3)/2) == [-2, 0]
    assert field_isomorphism(-I*sqrt(3), -I*sqrt(3)/2) == [ 2, 0]

    assert field_isomorphism( 2*I*sqrt(3)/7, 5*I*sqrt(3)/3) == [ Rational(6, 35), 0]
    assert field_isomorphism(-2*I*sqrt(3)/7, 5*I*sqrt(3)/3) == [Rational(-6, 35), 0]

    assert field_isomorphism( 2*I*sqrt(3)/7, -5*I*sqrt(3)/3) == [Rational(-6, 35), 0]
    assert field_isomorphism(-2*I*sqrt(3)/7, -5*I*sqrt(3)/3) == [ Rational(6, 35), 0]

    assert field_isomorphism(
        2*I*sqrt(3)/7 + 27, 5*I*sqrt(3)/3) == [ Rational(6, 35), 27]
    assert field_isomorphism(
        -2*I*sqrt(3)/7 + 27, 5*I*sqrt(3)/3) == [Rational(-6, 35), 27]

    assert field_isomorphism(
        2*I*sqrt(3)/7 + 27, -5*I*sqrt(3)/3) == [Rational(-6, 35), 27]
    assert field_isomorphism(
        -2*I*sqrt(3)/7 + 27, -5*I*sqrt(3)/3) == [ Rational(6, 35), 27]

    p = AlgebraicNumber( sqrt(2) + sqrt(3))
    q = AlgebraicNumber(-sqrt(2) + sqrt(3))
    r = AlgebraicNumber( sqrt(2) - sqrt(3))
    s = AlgebraicNumber(-sqrt(2) - sqrt(3))

    pos_coeffs = [ S.Half, S.Zero, Rational(-9, 2), S.Zero]
    neg_coeffs = [Rational(-1, 2), S.Zero, Rational(9, 2), S.Zero]

    a = AlgebraicNumber(sqrt(2))

    assert is_isomorphism_possible(a, p) is True
    assert is_isomorphism_possible(a, q) is True
    assert is_isomorphism_possible(a, r) is True
    assert is_isomorphism_possible(a, s) is True

    assert field_isomorphism(a, p, fast=True) == pos_coeffs
    assert field_isomorphism(a, q, fast=True) == neg_coeffs
    assert field_isomorphism(a, r, fast=True) == pos_coeffs
    assert field_isomorphism(a, s, fast=True) == neg_coeffs

    assert field_isomorphism(a, p, fast=False) == pos_coeffs
    assert field_isomorphism(a, q, fast=False) == neg_coeffs
    assert field_isomorphism(a, r, fast=False) == pos_coeffs
    assert field_isomorphism(a, s, fast=False) == neg_coeffs

    a = AlgebraicNumber(-sqrt(2))

    assert is_isomorphism_possible(a, p) is True
    assert is_isomorphism_possible(a, q) is True
    assert is_isomorphism_possible(a, r) is True
    assert is_isomorphism_possible(a, s) is True

    assert field_isomorphism(a, p, fast=True) == neg_coeffs
    assert field_isomorphism(a, q, fast=True) == pos_coeffs
    assert field_isomorphism(a, r, fast=True) == neg_coeffs
    assert field_isomorphism(a, s, fast=True) == pos_coeffs

    assert field_isomorphism(a, p, fast=False) == neg_coeffs
    assert field_isomorphism(a, q, fast=False) == pos_coeffs
    assert field_isomorphism(a, r, fast=False) == neg_coeffs
    assert field_isomorphism(a, s, fast=False) == pos_coeffs

    pos_coeffs = [ S.Half, S.Zero, Rational(-11, 2), S.Zero]
    neg_coeffs = [Rational(-1, 2), S.Zero, Rational(11, 2), S.Zero]

    a = AlgebraicNumber(sqrt(3))

    assert is_isomorphism_possible(a, p) is True
    assert is_isomorphism_possible(a, q) is True
    assert is_isomorphism_possible(a, r) is True
    assert is_isomorphism_possible(a, s) is True

    assert field_isomorphism(a, p, fast=True) == neg_coeffs
    assert field_isomorphism(a, q, fast=True) == neg_coeffs
    assert field_isomorphism(a, r, fast=True) == pos_coeffs
    assert field_isomorphism(a, s, fast=True) == pos_coeffs

    assert field_isomorphism(a, p, fast=False) == neg_coeffs
    assert field_isomorphism(a, q, fast=False) == neg_coeffs
    assert field_isomorphism(a, r, fast=False) == pos_coeffs
    assert field_isomorphism(a, s, fast=False) == pos_coeffs

    a = AlgebraicNumber(-sqrt(3))

    assert is_isomorphism_possible(a, p) is True
    assert is_isomorphism_possible(a, q) is True
    assert is_isomorphism_possible(a, r) is True
    assert is_isomorphism_possible(a, s) is True

    assert field_isomorphism(a, p, fast=True) == pos_coeffs
    assert field_isomorphism(a, q, fast=True) == pos_coeffs
    assert field_isomorphism(a, r, fast=True) == neg_coeffs
    assert field_isomorphism(a, s, fast=True) == neg_coeffs

    assert field_isomorphism(a, p, fast=False) == pos_coeffs
    assert field_isomorphism(a, q, fast=False) == pos_coeffs
    assert field_isomorphism(a, r, fast=False) == neg_coeffs
    assert field_isomorphism(a, s, fast=False) == neg_coeffs

    pos_coeffs = [ Rational(3, 2), S.Zero, Rational(-33, 2), -S(8)]
    neg_coeffs = [Rational(-3, 2), S.Zero, Rational(33, 2), -S(8)]

    a = AlgebraicNumber(3*sqrt(3) - 8)

    assert is_isomorphism_possible(a, p) is True
    assert is_isomorphism_possible(a, q) is True
    assert is_isomorphism_possible(a, r) is True
    assert is_isomorphism_possible(a, s) is True

    assert field_isomorphism(a, p, fast=True) == neg_coeffs
    assert field_isomorphism(a, q, fast=True) == neg_coeffs
    assert field_isomorphism(a, r, fast=True) == pos_coeffs
    assert field_isomorphism(a, s, fast=True) == pos_coeffs

    assert field_isomorphism(a, p, fast=False) == neg_coeffs
    assert field_isomorphism(a, q, fast=False) == neg_coeffs
    assert field_isomorphism(a, r, fast=False) == pos_coeffs
    assert field_isomorphism(a, s, fast=False) == pos_coeffs

    a = AlgebraicNumber(3*sqrt(2) + 2*sqrt(3) + 1)

    pos_1_coeffs = [ S.Half, S.Zero, Rational(-5, 2), S.One]
    neg_5_coeffs = [Rational(-5, 2), S.Zero, Rational(49, 2), S.One]
    pos_5_coeffs = [ Rational(5, 2), S.Zero, Rational(-49, 2), S.One]
    neg_1_coeffs = [Rational(-1, 2), S.Zero, Rational(5, 2), S.One]

    assert is_isomorphism_possible(a, p) is True
    assert is_isomorphism_possible(a, q) is True
    assert is_isomorphism_possible(a, r) is True
    assert is_isomorphism_possible(a, s) is True

    assert field_isomorphism(a, p, fast=True) == pos_1_coeffs
    assert field_isomorphism(a, q, fast=True) == neg_5_coeffs
    assert field_isomorphism(a, r, fast=True) == pos_5_coeffs
    assert field_isomorphism(a, s, fast=True) == neg_1_coeffs

    assert field_isomorphism(a, p, fast=False) == pos_1_coeffs
    assert field_isomorphism(a, q, fast=False) == neg_5_coeffs
    assert field_isomorphism(a, r, fast=False) == pos_5_coeffs
    assert field_isomorphism(a, s, fast=False) == neg_1_coeffs

    a = AlgebraicNumber(sqrt(2))
    b = AlgebraicNumber(sqrt(3))
    c = AlgebraicNumber(sqrt(7))

    assert is_isomorphism_possible(a, b) is True
    assert is_isomorphism_possible(b, a) is True

    assert is_isomorphism_possible(c, p) is False

    assert field_isomorphism(sqrt(2), sqrt(3), fast=True) is None
    assert field_isomorphism(sqrt(3), sqrt(2), fast=True) is None

    assert field_isomorphism(sqrt(2), sqrt(3), fast=False) is None
    assert field_isomorphism(sqrt(3), sqrt(2), fast=False) is None

    a = AlgebraicNumber(sqrt(2))
    b = AlgebraicNumber(2 ** (S(1) / 3))

    assert is_isomorphism_possible(a, b) is False
    assert field_isomorphism(a, b) is None


def test_primitive_element():
    assert primitive_element([sqrt(2)], x) == (x**2 - 2, [1])
    assert primitive_element(
        [sqrt(2), sqrt(3)], x) == (x**4 - 10*x**2 + 1, [1, 1])

    assert primitive_element([sqrt(2)], x, polys=True) == (Poly(x**2 - 2, domain='QQ'), [1])
    assert primitive_element([sqrt(
        2), sqrt(3)], x, polys=True) == (Poly(x**4 - 10*x**2 + 1, domain='QQ'), [1, 1])

    assert primitive_element(
        [sqrt(2)], x, ex=True) == (x**2 - 2, [1], [[1, 0]])
    assert primitive_element([sqrt(2), sqrt(3)], x, ex=True) == \
        (x**4 - 10*x**2 + 1, [1, 1], [[Q(1, 2), 0, -Q(9, 2), 0], [-
         Q(1, 2), 0, Q(11, 2), 0]])

    assert primitive_element(
        [sqrt(2)], x, ex=True, polys=True) == (Poly(x**2 - 2, domain='QQ'), [1], [[1, 0]])
    assert primitive_element([sqrt(2), sqrt(3)], x, ex=True, polys=True) == \
        (Poly(x**4 - 10*x**2 + 1, domain='QQ'), [1, 1], [[Q(1, 2), 0, -Q(9, 2),
         0], [-Q(1, 2), 0, Q(11, 2), 0]])

    assert primitive_element([sqrt(2)], polys=True) == (Poly(x**2 - 2), [1])

    raises(ValueError, lambda: primitive_element([], x, ex=False))
    raises(ValueError, lambda: primitive_element([], x, ex=True))

    # Issue 14117
    a, b = I*sqrt(2*sqrt(2) + 3), I*sqrt(-2*sqrt(2) + 3)
    assert primitive_element([a, b, I], x) == (x**4 + 6*x**2 + 1, [1, 0, 0])

    assert primitive_element([sqrt(2), 0], x) == (x**2 - 2, [1, 0])
    assert primitive_element([0, sqrt(2)], x) == (x**2 - 2, [1, 1])
    assert primitive_element([sqrt(2), 0], x, ex=True) == (x**2 - 2, [1, 0], [[MPQ(1,1), MPQ(0,1)], []])
    assert primitive_element([0, sqrt(2)], x, ex=True) == (x**2 - 2, [1, 1], [[], [MPQ(1,1), MPQ(0,1)]])


def test_to_number_field():
    assert to_number_field(sqrt(2)) == AlgebraicNumber(sqrt(2))
    assert to_number_field(
        [sqrt(2), sqrt(3)]) == AlgebraicNumber(sqrt(2) + sqrt(3))

    a = AlgebraicNumber(sqrt(2) + sqrt(3), [S.Half, S.Zero, Rational(-9, 2), S.Zero])

    assert to_number_field(sqrt(2), sqrt(2) + sqrt(3)) == a
    assert to_number_field(sqrt(2), AlgebraicNumber(sqrt(2) + sqrt(3))) == a

    raises(IsomorphismFailed, lambda: to_number_field(sqrt(2), sqrt(3)))


def test_issue_22561():
    a = to_number_field(sqrt(2), sqrt(2) + sqrt(3))
    b = to_number_field(sqrt(2), sqrt(2) + sqrt(5))
    assert field_isomorphism(a, b) == [1, 0]


def test_issue_22736():
    a = CRootOf(x**4 + x**3 + x**2 + x + 1, -1)
    a._reset()
    b = exp(2*I*pi/5)
    assert field_isomorphism(a, b) == [1, 0]


def test_issue_27798():
    # https://github.com/sympy/sympy/issues/27798
    a, b = CRootOf(49*x**3 - 49*x**2 + 14*x - 1, 2), CRootOf(49*x**3 - 49*x**2 + 14*x - 1, 0)
    assert primitive_element([a, b], polys=True)[0].primitive()[0] == 1
    assert primitive_element([a, b], polys=True, ex=True)[0].primitive()[0] == 1

    f1, f2 = QQ.algebraic_field(a), QQ.algebraic_field(b)
    f3 = f1.unify(f2)
    assert f3.mod.primitive()[0] == 1
    assert Poly(x, x, domain=f1) + Poly(x, x, domain=f2) == Poly(2*x, x, domain=f3)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\sets\tests\test_setexpr.py ===
from sympy.sets.setexpr import SetExpr
from sympy.sets import Interval, FiniteSet, Intersection, ImageSet, Union

from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.trigonometric import cos
from sympy.sets.sets import Set


a, x = symbols("a, x")
_d = Dummy("d")


def test_setexpr():
    se = SetExpr(Interval(0, 1))
    assert isinstance(se.set, Set)
    assert isinstance(se, Expr)


def test_scalar_funcs():
    assert SetExpr(Interval(0, 1)).set == Interval(0, 1)
    a, b = Symbol('a', real=True), Symbol('b', real=True)
    a, b = 1, 2
    # TODO: add support for more functions in the future:
    for f in [exp, log]:
        input_se = f(SetExpr(Interval(a, b)))
        output = input_se.set
        expected = Interval(Min(f(a), f(b)), Max(f(a), f(b)))
        assert output == expected


def test_Add_Mul():
    assert (SetExpr(Interval(0, 1)) + 1).set == Interval(1, 2)
    assert (SetExpr(Interval(0, 1))*2).set == Interval(0, 2)


def test_Pow():
    assert (SetExpr(Interval(0, 2))**2).set == Interval(0, 4)


def test_compound():
    assert (exp(SetExpr(Interval(0, 1))*2 + 1)).set == \
           Interval(exp(1), exp(3))


def test_Interval_Interval():
    assert (SetExpr(Interval(1, 2)) + SetExpr(Interval(10, 20))).set == \
           Interval(11, 22)
    assert (SetExpr(Interval(1, 2))*SetExpr(Interval(10, 20))).set == \
           Interval(10, 40)


def test_FiniteSet_FiniteSet():
    assert (SetExpr(FiniteSet(1, 2, 3)) + SetExpr(FiniteSet(1, 2))).set == \
           FiniteSet(2, 3, 4, 5)
    assert (SetExpr(FiniteSet(1, 2, 3))*SetExpr(FiniteSet(1, 2))).set == \
           FiniteSet(1, 2, 3, 4, 6)


def test_Interval_FiniteSet():
    assert (SetExpr(FiniteSet(1, 2)) + SetExpr(Interval(0, 10))).set == \
           Interval(1, 12)


def test_Many_Sets():
    assert (SetExpr(Interval(0, 1)) +
            SetExpr(Interval(2, 3)) +
            SetExpr(FiniteSet(10, 11, 12))).set == Interval(12, 16)


def test_same_setexprs_are_not_identical():
    a = SetExpr(FiniteSet(0, 1))
    b = SetExpr(FiniteSet(0, 1))
    assert (a + b).set == FiniteSet(0, 1, 2)

    # Cannot detect the set being the same:
    # assert (a + a).set == FiniteSet(0, 2)


def test_Interval_arithmetic():
    i12cc = SetExpr(Interval(1, 2))
    i12lo = SetExpr(Interval.Lopen(1, 2))
    i12ro = SetExpr(Interval.Ropen(1, 2))
    i12o = SetExpr(Interval.open(1, 2))

    n23cc = SetExpr(Interval(-2, 3))
    n23lo = SetExpr(Interval.Lopen(-2, 3))
    n23ro = SetExpr(Interval.Ropen(-2, 3))
    n23o = SetExpr(Interval.open(-2, 3))

    n3n2cc = SetExpr(Interval(-3, -2))

    assert i12cc + i12cc == SetExpr(Interval(2, 4))
    assert i12cc - i12cc == SetExpr(Interval(-1, 1))
    assert i12cc*i12cc == SetExpr(Interval(1, 4))
    assert i12cc/i12cc == SetExpr(Interval(S.Half, 2))
    assert i12cc**2 == SetExpr(Interval(1, 4))
    assert i12cc**3 == SetExpr(Interval(1, 8))

    assert i12lo + i12ro == SetExpr(Interval.open(2, 4))
    assert i12lo - i12ro == SetExpr(Interval.Lopen(-1, 1))
    assert i12lo*i12ro == SetExpr(Interval.open(1, 4))
    assert i12lo/i12ro == SetExpr(Interval.Lopen(S.Half, 2))
    assert i12lo + i12lo == SetExpr(Interval.Lopen(2, 4))
    assert i12lo - i12lo == SetExpr(Interval.open(-1, 1))
    assert i12lo*i12lo == SetExpr(Interval.Lopen(1, 4))
    assert i12lo/i12lo == SetExpr(Interval.open(S.Half, 2))
    assert i12lo + i12cc == SetExpr(Interval.Lopen(2, 4))
    assert i12lo - i12cc == SetExpr(Interval.Lopen(-1, 1))
    assert i12lo*i12cc == SetExpr(Interval.Lopen(1, 4))
    assert i12lo/i12cc == SetExpr(Interval.Lopen(S.Half, 2))
    assert i12lo + i12o == SetExpr(Interval.open(2, 4))
    assert i12lo - i12o == SetExpr(Interval.open(-1, 1))
    assert i12lo*i12o == SetExpr(Interval.open(1, 4))
    assert i12lo/i12o == SetExpr(Interval.open(S.Half, 2))
    assert i12lo**2 == SetExpr(Interval.Lopen(1, 4))
    assert i12lo**3 == SetExpr(Interval.Lopen(1, 8))

    assert i12ro + i12ro == SetExpr(Interval.Ropen(2, 4))
    assert i12ro - i12ro == SetExpr(Interval.open(-1, 1))
    assert i12ro*i12ro == SetExpr(Interval.Ropen(1, 4))
    assert i12ro/i12ro == SetExpr(Interval.open(S.Half, 2))
    assert i12ro + i12cc == SetExpr(Interval.Ropen(2, 4))
    assert i12ro - i12cc == SetExpr(Interval.Ropen(-1, 1))
    assert i12ro*i12cc == SetExpr(Interval.Ropen(1, 4))
    assert i12ro/i12cc == SetExpr(Interval.Ropen(S.Half, 2))
    assert i12ro + i12o == SetExpr(Interval.open(2, 4))
    assert i12ro - i12o == SetExpr(Interval.open(-1, 1))
    assert i12ro*i12o == SetExpr(Interval.open(1, 4))
    assert i12ro/i12o == SetExpr(Interval.open(S.Half, 2))
    assert i12ro**2 == SetExpr(Interval.Ropen(1, 4))
    assert i12ro**3 == SetExpr(Interval.Ropen(1, 8))

    assert i12o + i12lo == SetExpr(Interval.open(2, 4))
    assert i12o - i12lo == SetExpr(Interval.open(-1, 1))
    assert i12o*i12lo == SetExpr(Interval.open(1, 4))
    assert i12o/i12lo == SetExpr(Interval.open(S.Half, 2))
    assert i12o + i12ro == SetExpr(Interval.open(2, 4))
    assert i12o - i12ro == SetExpr(Interval.open(-1, 1))
    assert i12o*i12ro == SetExpr(Interval.open(1, 4))
    assert i12o/i12ro == SetExpr(Interval.open(S.Half, 2))
    assert i12o + i12cc == SetExpr(Interval.open(2, 4))
    assert i12o - i12cc == SetExpr(Interval.open(-1, 1))
    assert i12o*i12cc == SetExpr(Interval.open(1, 4))
    assert i12o/i12cc == SetExpr(Interval.open(S.Half, 2))
    assert i12o**2 == SetExpr(Interval.open(1, 4))
    assert i12o**3 == SetExpr(Interval.open(1, 8))

    assert n23cc + n23cc == SetExpr(Interval(-4, 6))
    assert n23cc - n23cc == SetExpr(Interval(-5, 5))
    assert n23cc*n23cc == SetExpr(Interval(-6, 9))
    assert n23cc/n23cc == SetExpr(Interval.open(-oo, oo))
    assert n23cc + n23ro == SetExpr(Interval.Ropen(-4, 6))
    assert n23cc - n23ro == SetExpr(Interval.Lopen(-5, 5))
    assert n23cc*n23ro == SetExpr(Interval.Ropen(-6, 9))
    assert n23cc/n23ro == SetExpr(Interval.Lopen(-oo, oo))
    assert n23cc + n23lo == SetExpr(Interval.Lopen(-4, 6))
    assert n23cc - n23lo == SetExpr(Interval.Ropen(-5, 5))
    assert n23cc*n23lo == SetExpr(Interval(-6, 9))
    assert n23cc/n23lo == SetExpr(Interval.open(-oo, oo))
    assert n23cc + n23o == SetExpr(Interval.open(-4, 6))
    assert n23cc - n23o == SetExpr(Interval.open(-5, 5))
    assert n23cc*n23o == SetExpr(Interval.open(-6, 9))
    assert n23cc/n23o == SetExpr(Interval.open(-oo, oo))
    assert n23cc**2 == SetExpr(Interval(0, 9))
    assert n23cc**3 == SetExpr(Interval(-8, 27))

    n32cc = SetExpr(Interval(-3, 2))
    n32lo = SetExpr(Interval.Lopen(-3, 2))
    n32ro = SetExpr(Interval.Ropen(-3, 2))
    assert n32cc*n32lo == SetExpr(Interval.Ropen(-6, 9))
    assert n32cc*n32cc == SetExpr(Interval(-6, 9))
    assert n32lo*n32cc == SetExpr(Interval.Ropen(-6, 9))
    assert n32cc*n32ro == SetExpr(Interval(-6, 9))
    assert n32lo*n32ro == SetExpr(Interval.Ropen(-6, 9))
    assert n32cc/n32lo == SetExpr(Interval.Ropen(-oo, oo))
    assert i12cc/n32lo == SetExpr(Interval.Ropen(-oo, oo))

    assert n3n2cc**2 == SetExpr(Interval(4, 9))
    assert n3n2cc**3 == SetExpr(Interval(-27, -8))

    assert n23cc + i12cc == SetExpr(Interval(-1, 5))
    assert n23cc - i12cc == SetExpr(Interval(-4, 2))
    assert n23cc*i12cc == SetExpr(Interval(-4, 6))
    assert n23cc/i12cc == SetExpr(Interval(-2, 3))


def test_SetExpr_Intersection():
    x, y, z, w = symbols("x y z w")
    set1 = Interval(x, y)
    set2 = Interval(w, z)
    inter = Intersection(set1, set2)
    se = SetExpr(inter)
    assert exp(se).set == Intersection(
        ImageSet(Lambda(x, exp(x)), set1),
        ImageSet(Lambda(x, exp(x)), set2))
    assert cos(se).set == ImageSet(Lambda(x, cos(x)), inter)


def test_SetExpr_Interval_div():
    # TODO: some expressions cannot be calculated due to bugs (currently
    # commented):
    assert SetExpr(Interval(-3, -2))/SetExpr(Interval(-2, 1)) == SetExpr(Interval(-oo, oo))
    assert SetExpr(Interval(2, 3))/SetExpr(Interval(-2, 2)) == SetExpr(Interval(-oo, oo))

    assert SetExpr(Interval(-3, -2))/SetExpr(Interval(0, 4)) == SetExpr(Interval(-oo, Rational(-1, 2)))
    assert SetExpr(Interval(2, 4))/SetExpr(Interval(-3, 0)) == SetExpr(Interval(-oo, Rational(-2, 3)))
    assert SetExpr(Interval(2, 4))/SetExpr(Interval(0, 3)) == SetExpr(Interval(Rational(2, 3), oo))

    # assert SetExpr(Interval(0, 1))/SetExpr(Interval(0, 1)) == SetExpr(Interval(0, oo))
    # assert SetExpr(Interval(-1, 0))/SetExpr(Interval(0, 1)) == SetExpr(Interval(-oo, 0))
    assert SetExpr(Interval(-1, 2))/SetExpr(Interval(-2, 2)) == SetExpr(Interval(-oo, oo))

    assert 1/SetExpr(Interval(-1, 2)) == SetExpr(Union(Interval(-oo, -1), Interval(S.Half, oo)))

    assert 1/SetExpr(Interval(0, 2)) == SetExpr(Interval(S.Half, oo))
    assert (-1)/SetExpr(Interval(0, 2)) == SetExpr(Interval(-oo, Rational(-1, 2)))
    assert 1/SetExpr(Interval(-oo, 0)) == SetExpr(Interval.open(-oo, 0))
    assert 1/SetExpr(Interval(-1, 0)) == SetExpr(Interval(-oo, -1))
    # assert (-2)/SetExpr(Interval(-oo, 0)) == SetExpr(Interval(0, oo))
    # assert 1/SetExpr(Interval(-oo, -1)) == SetExpr(Interval(-1, 0))

    # assert SetExpr(Interval(1, 2))/a == Mul(SetExpr(Interval(1, 2)), 1/a, evaluate=False)

    # assert SetExpr(Interval(1, 2))/0 == SetExpr(Interval(1, 2))*zoo
    # assert SetExpr(Interval(1, oo))/oo == SetExpr(Interval(0, oo))
    # assert SetExpr(Interval(1, oo))/(-oo) == SetExpr(Interval(-oo, 0))
    # assert SetExpr(Interval(-oo, -1))/oo == SetExpr(Interval(-oo, 0))
    # assert SetExpr(Interval(-oo, -1))/(-oo) == SetExpr(Interval(0, oo))
    # assert SetExpr(Interval(-oo, oo))/oo == SetExpr(Interval(-oo, oo))
    # assert SetExpr(Interval(-oo, oo))/(-oo) == SetExpr(Interval(-oo, oo))
    # assert SetExpr(Interval(-1, oo))/oo == SetExpr(Interval(0, oo))
    # assert SetExpr(Interval(-1, oo))/(-oo) == SetExpr(Interval(-oo, 0))
    # assert SetExpr(Interval(-oo, 1))/oo == SetExpr(Interval(-oo, 0))
    # assert SetExpr(Interval(-oo, 1))/(-oo) == SetExpr(Interval(0, oo))


def test_SetExpr_Interval_pow():
    assert SetExpr(Interval(0, 2))**2 == SetExpr(Interval(0, 4))
    assert SetExpr(Interval(-1, 1))**2 == SetExpr(Interval(0, 1))
    assert SetExpr(Interval(1, 2))**2 == SetExpr(Interval(1, 4))
    assert SetExpr(Interval(-1, 2))**3 == SetExpr(Interval(-1, 8))
    assert SetExpr(Interval(-1, 1))**0 == SetExpr(FiniteSet(1))


    assert SetExpr(Interval(1, 2))**Rational(5, 2) == SetExpr(Interval(1, 4*sqrt(2)))
    #assert SetExpr(Interval(-1, 2))**Rational(1, 3) == SetExpr(Interval(-1, 2**Rational(1, 3)))
    #assert SetExpr(Interval(0, 2))**S.Half == SetExpr(Interval(0, sqrt(2)))

    #assert SetExpr(Interval(-4, 2))**Rational(2, 3) == SetExpr(Interval(0, 2*2**Rational(1, 3)))

    #assert SetExpr(Interval(-1, 5))**S.Half == SetExpr(Interval(0, sqrt(5)))
    #assert SetExpr(Interval(-oo, 2))**S.Half == SetExpr(Interval(0, sqrt(2)))
    #assert SetExpr(Interval(-2, 3))**(Rational(-1, 4)) == SetExpr(Interval(0, oo))

    assert SetExpr(Interval(1, 5))**(-2) == SetExpr(Interval(Rational(1, 25), 1))
    assert SetExpr(Interval(-1, 3))**(-2) == SetExpr(Interval(0, oo))

    assert SetExpr(Interval(0, 2))**(-2) == SetExpr(Interval(Rational(1, 4), oo))
    assert SetExpr(Interval(-1, 2))**(-3) == SetExpr(Union(Interval(-oo, -1), Interval(Rational(1, 8), oo)))
    assert SetExpr(Interval(-3, -2))**(-3) == SetExpr(Interval(Rational(-1, 8), Rational(-1, 27)))
    assert SetExpr(Interval(-3, -2))**(-2) == SetExpr(Interval(Rational(1, 9), Rational(1, 4)))
    #assert SetExpr(Interval(0, oo))**S.Half == SetExpr(Interval(0, oo))
    #assert SetExpr(Interval(-oo, -1))**Rational(1, 3) == SetExpr(Interval(-oo, -1))
    #assert SetExpr(Interval(-2, 3))**(Rational(-1, 3)) == SetExpr(Interval(-oo, oo))

    assert SetExpr(Interval(-oo, 0))**(-2) == SetExpr(Interval.open(0, oo))
    assert SetExpr(Interval(-2, 0))**(-2) == SetExpr(Interval(Rational(1, 4), oo))

    assert SetExpr(Interval(Rational(1, 3), S.Half))**oo == SetExpr(FiniteSet(0))
    assert SetExpr(Interval(0, S.Half))**oo == SetExpr(FiniteSet(0))
    assert SetExpr(Interval(S.Half, 1))**oo == SetExpr(Interval(0, oo))
    assert SetExpr(Interval(0, 1))**oo == SetExpr(Interval(0, oo))
    assert SetExpr(Interval(2, 3))**oo == SetExpr(FiniteSet(oo))
    assert SetExpr(Interval(1, 2))**oo == SetExpr(Interval(0, oo))
    assert SetExpr(Interval(S.Half, 3))**oo == SetExpr(Interval(0, oo))
    assert SetExpr(Interval(Rational(-1, 3), Rational(-1, 4)))**oo == SetExpr(FiniteSet(0))
    assert SetExpr(Interval(-1, Rational(-1, 2)))**oo == SetExpr(Interval(-oo, oo))
    assert SetExpr(Interval(-3, -2))**oo == SetExpr(FiniteSet(-oo, oo))
    assert SetExpr(Interval(-2, -1))**oo == SetExpr(Interval(-oo, oo))
    assert SetExpr(Interval(-2, Rational(-1, 2)))**oo == SetExpr(Interval(-oo, oo))
    assert SetExpr(Interval(Rational(-1, 2), S.Half))**oo == SetExpr(FiniteSet(0))
    assert SetExpr(Interval(Rational(-1, 2), 1))**oo == SetExpr(Interval(0, oo))
    assert SetExpr(Interval(Rational(-2, 3), 2))**oo == SetExpr(Interval(0, oo))
    assert SetExpr(Interval(-1, 1))**oo == SetExpr(Interval(-oo, oo))
    assert SetExpr(Interval(-1, S.Half))**oo == SetExpr(Interval(-oo, oo))
    assert SetExpr(Interval(-1, 2))**oo == SetExpr(Interval(-oo, oo))
    assert SetExpr(Interval(-2, S.Half))**oo == SetExpr(Interval(-oo, oo))

    assert (SetExpr(Interval(1, 2))**x).dummy_eq(SetExpr(ImageSet(Lambda(_d, _d**x), Interval(1, 2))))

    assert SetExpr(Interval(2, 3))**(-oo) == SetExpr(FiniteSet(0))
    assert SetExpr(Interval(0, 2))**(-oo) == SetExpr(Interval(0, oo))
    assert (SetExpr(Interval(-1, 2))**(-oo)).dummy_eq(SetExpr(ImageSet(Lambda(_d, _d**(-oo)), Interval(-1, 2))))


def test_SetExpr_Integers():
    assert SetExpr(S.Integers) + 1 == SetExpr(S.Integers)
    assert (SetExpr(S.Integers) + I).dummy_eq(
        SetExpr(ImageSet(Lambda(_d, _d + I), S.Integers)))
    assert SetExpr(S.Integers)*(-1) == SetExpr(S.Integers)
    assert (SetExpr(S.Integers)*2).dummy_eq(
        SetExpr(ImageSet(Lambda(_d, 2*_d), S.Integers)))
    assert (SetExpr(S.Integers)*I).dummy_eq(
        SetExpr(ImageSet(Lambda(_d, I*_d), S.Integers)))
    # issue #18050:
    assert SetExpr(S.Integers)._eval_func(Lambda(x, I*x + 1)).dummy_eq(
        SetExpr(ImageSet(Lambda(_d, I*_d + 1), S.Integers)))
    # needs improvement:
    assert (SetExpr(S.Integers)*I + 1).dummy_eq(
        SetExpr(ImageSet(Lambda(x, x + 1),
        ImageSet(Lambda(_d, _d*I), S.Integers))))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\test_na_scalar.py ===
from datetime import (
    date,
    time,
    timedelta,
)
import pickle

import numpy as np
import pytest

from pandas._libs.missing import NA

from pandas.core.dtypes.common import is_scalar

import pandas as pd
import pandas._testing as tm


def test_singleton():
    assert NA is NA
    new_NA = type(NA)()
    assert new_NA is NA


def test_repr():
    assert repr(NA) == "<NA>"
    assert str(NA) == "<NA>"


def test_format():
    # GH-34740
    assert format(NA) == "<NA>"
    assert format(NA, ">10") == "      <NA>"
    assert format(NA, "xxx") == "<NA>"  # NA is flexible, accept any format spec

    assert f"{NA}" == "<NA>"
    assert f"{NA:>10}" == "      <NA>"
    assert f"{NA:xxx}" == "<NA>"


def test_truthiness():
    msg = "boolean value of NA is ambiguous"

    with pytest.raises(TypeError, match=msg):
        bool(NA)

    with pytest.raises(TypeError, match=msg):
        not NA


def test_hashable():
    assert hash(NA) == hash(NA)
    d = {NA: "test"}
    assert d[NA] == "test"


@pytest.mark.parametrize(
    "other", [NA, 1, 1.0, "a", b"a", np.int64(1), np.nan], ids=repr
)
def test_arithmetic_ops(all_arithmetic_functions, other):
    op = all_arithmetic_functions

    if op.__name__ in ("pow", "rpow", "rmod") and isinstance(other, (str, bytes)):
        pytest.skip(reason=f"{op.__name__} with NA and {other} not defined.")
    if op.__name__ in ("divmod", "rdivmod"):
        assert op(NA, other) is (NA, NA)
    else:
        if op.__name__ == "rpow":
            # avoid special case
            other += 1
        assert op(NA, other) is NA


@pytest.mark.parametrize(
    "other",
    [
        NA,
        1,
        1.0,
        "a",
        b"a",
        np.int64(1),
        np.nan,
        np.bool_(True),
        time(0),
        date(1, 2, 3),
        timedelta(1),
        pd.NaT,
    ],
)
def test_comparison_ops(comparison_op, other):
    assert comparison_op(NA, other) is NA
    assert comparison_op(other, NA) is NA


@pytest.mark.parametrize(
    "value",
    [
        0,
        0.0,
        -0,
        -0.0,
        False,
        np.bool_(False),
        np.int_(0),
        np.float64(0),
        np.int_(-0),
        np.float64(-0),
    ],
)
@pytest.mark.parametrize("asarray", [True, False])
def test_pow_special(value, asarray):
    if asarray:
        value = np.array([value])
    result = NA**value

    if asarray:
        result = result[0]
    else:
        # this assertion isn't possible for ndarray.
        assert isinstance(result, type(value))
    assert result == 1


@pytest.mark.parametrize(
    "value", [1, 1.0, True, np.bool_(True), np.int_(1), np.float64(1)]
)
@pytest.mark.parametrize("asarray", [True, False])
def test_rpow_special(value, asarray):
    if asarray:
        value = np.array([value])
    result = value**NA

    if asarray:
        result = result[0]
    elif not isinstance(value, (np.float64, np.bool_, np.int_)):
        # this assertion isn't possible with asarray=True
        assert isinstance(result, type(value))

    assert result == value


@pytest.mark.parametrize("value", [-1, -1.0, np.int_(-1), np.float64(-1)])
@pytest.mark.parametrize("asarray", [True, False])
def test_rpow_minus_one(value, asarray):
    if asarray:
        value = np.array([value])
    result = value**NA

    if asarray:
        result = result[0]

    assert pd.isna(result)


def test_unary_ops():
    assert +NA is NA
    assert -NA is NA
    assert abs(NA) is NA
    assert ~NA is NA


def test_logical_and():
    assert NA & True is NA
    assert True & NA is NA
    assert NA & False is False
    assert False & NA is False
    assert NA & NA is NA

    msg = "unsupported operand type"
    with pytest.raises(TypeError, match=msg):
        NA & 5


def test_logical_or():
    assert NA | True is True
    assert True | NA is True
    assert NA | False is NA
    assert False | NA is NA
    assert NA | NA is NA

    msg = "unsupported operand type"
    with pytest.raises(TypeError, match=msg):
        NA | 5


def test_logical_xor():
    assert NA ^ True is NA
    assert True ^ NA is NA
    assert NA ^ False is NA
    assert False ^ NA is NA
    assert NA ^ NA is NA

    msg = "unsupported operand type"
    with pytest.raises(TypeError, match=msg):
        NA ^ 5


def test_logical_not():
    assert ~NA is NA


@pytest.mark.parametrize("shape", [(3,), (3, 3), (1, 2, 3)])
def test_arithmetic_ndarray(shape, all_arithmetic_functions):
    op = all_arithmetic_functions
    a = np.zeros(shape)
    if op.__name__ == "pow":
        a += 5
    result = op(NA, a)
    expected = np.full(a.shape, NA, dtype=object)
    tm.assert_numpy_array_equal(result, expected)


def test_is_scalar():
    assert is_scalar(NA) is True


def test_isna():
    assert pd.isna(NA) is True
    assert pd.notna(NA) is False


def test_series_isna():
    s = pd.Series([1, NA], dtype=object)
    expected = pd.Series([False, True])
    tm.assert_series_equal(s.isna(), expected)


def test_ufunc():
    assert np.log(NA) is NA
    assert np.add(NA, 1) is NA
    result = np.divmod(NA, 1)
    assert result[0] is NA and result[1] is NA

    result = np.frexp(NA)
    assert result[0] is NA and result[1] is NA


def test_ufunc_raises():
    msg = "ufunc method 'at'"
    with pytest.raises(ValueError, match=msg):
        np.log.at(NA, 0)


def test_binary_input_not_dunder():
    a = np.array([1, 2, 3])
    expected = np.array([NA, NA, NA], dtype=object)
    result = np.logaddexp(a, NA)
    tm.assert_numpy_array_equal(result, expected)

    result = np.logaddexp(NA, a)
    tm.assert_numpy_array_equal(result, expected)

    # all NA, multiple inputs
    assert np.logaddexp(NA, NA) is NA

    result = np.modf(NA, NA)
    assert len(result) == 2
    assert all(x is NA for x in result)


def test_divmod_ufunc():
    # binary in, binary out.
    a = np.array([1, 2, 3])
    expected = np.array([NA, NA, NA], dtype=object)

    result = np.divmod(a, NA)
    assert isinstance(result, tuple)
    for arr in result:
        tm.assert_numpy_array_equal(arr, expected)
        tm.assert_numpy_array_equal(arr, expected)

    result = np.divmod(NA, a)
    for arr in result:
        tm.assert_numpy_array_equal(arr, expected)
        tm.assert_numpy_array_equal(arr, expected)


def test_integer_hash_collision_dict():
    # GH 30013
    result = {NA: "foo", hash(NA): "bar"}

    assert result[NA] == "foo"
    assert result[hash(NA)] == "bar"


def test_integer_hash_collision_set():
    # GH 30013
    result = {NA, hash(NA)}

    assert len(result) == 2
    assert NA in result
    assert hash(NA) in result


def test_pickle_roundtrip():
    # https://github.com/pandas-dev/pandas/issues/31847
    result = pickle.loads(pickle.dumps(NA))
    assert result is NA


def test_pickle_roundtrip_pandas():
    result = tm.round_trip_pickle(NA)
    assert result is NA


@pytest.mark.parametrize(
    "values, dtype", [([1, 2, NA], "Int64"), (["A", "B", NA], "string")]
)
@pytest.mark.parametrize("as_frame", [True, False])
def test_pickle_roundtrip_containers(as_frame, values, dtype):
    s = pd.Series(pd.array(values, dtype=dtype))
    if as_frame:
        s = s.to_frame(name="A")
    result = tm.round_trip_pickle(s)
    tm.assert_equal(result, s)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\comparisons.py ===
from __future__ import annotations

from typing import cast, Any
import numpy as np

c16 = np.complex128()
f8 = np.float64()
i8 = np.int64()
u8 = np.uint64()

c8 = np.complex64()
f4 = np.float32()
i4 = np.int32()
u4 = np.uint32()

dt = np.datetime64(0, "D")
td = np.timedelta64(0, "D")

b_ = np.bool()

b = bool()
c = complex()
f = float()
i = int()

SEQ = (0, 1, 2, 3, 4)

AR_b: np.ndarray[Any, np.dtype[np.bool]] = np.array([True])
AR_u: np.ndarray[Any, np.dtype[np.uint32]] = np.array([1], dtype=np.uint32)
AR_i: np.ndarray[Any, np.dtype[np.int_]] = np.array([1])
AR_f: np.ndarray[Any, np.dtype[np.float64]] = np.array([1.0])
AR_c: np.ndarray[Any, np.dtype[np.complex128]] = np.array([1.0j])
AR_S: np.ndarray[Any, np.dtype[np.bytes_]] = np.array([b"a"], "S")
AR_T = cast(np.ndarray[Any, np.dtypes.StringDType], np.array(["a"], "T"))
AR_U: np.ndarray[Any, np.dtype[np.str_]] = np.array(["a"], "U")
AR_m: np.ndarray[Any, np.dtype[np.timedelta64]] = np.array([np.timedelta64("1")])
AR_M: np.ndarray[Any, np.dtype[np.datetime64]] = np.array([np.datetime64("1")])
AR_O: np.ndarray[Any, np.dtype[np.object_]] = np.array([1], dtype=object)

# Arrays

AR_b > AR_b
AR_b > AR_u
AR_b > AR_i
AR_b > AR_f
AR_b > AR_c

AR_u > AR_b
AR_u > AR_u
AR_u > AR_i
AR_u > AR_f
AR_u > AR_c

AR_i > AR_b
AR_i > AR_u
AR_i > AR_i
AR_i > AR_f
AR_i > AR_c

AR_f > AR_b
AR_f > AR_u
AR_f > AR_i
AR_f > AR_f
AR_f > AR_c

AR_c > AR_b
AR_c > AR_u
AR_c > AR_i
AR_c > AR_f
AR_c > AR_c

AR_S > AR_S
AR_S > b""

AR_T > AR_T
AR_T > AR_U
AR_T > ""

AR_U > AR_U
AR_U > AR_T
AR_U > ""

AR_m > AR_b
AR_m > AR_u
AR_m > AR_i
AR_b > AR_m
AR_u > AR_m
AR_i > AR_m

AR_M > AR_M

AR_O > AR_O
1 > AR_O
AR_O > 1

# Time structures

dt > dt

td > td
td > i
td > i4
td > i8
td > AR_i
td > SEQ

# boolean

b_ > b
b_ > b_
b_ > i
b_ > i8
b_ > i4
b_ > u8
b_ > u4
b_ > f
b_ > f8
b_ > f4
b_ > c
b_ > c16
b_ > c8
b_ > AR_i
b_ > SEQ

# Complex

c16 > c16
c16 > f8
c16 > i8
c16 > c8
c16 > f4
c16 > i4
c16 > b_
c16 > b
c16 > c
c16 > f
c16 > i
c16 > AR_i
c16 > SEQ

c16 > c16
f8 > c16
i8 > c16
c8 > c16
f4 > c16
i4 > c16
b_ > c16
b > c16
c > c16
f > c16
i > c16
AR_i > c16
SEQ > c16

c8 > c16
c8 > f8
c8 > i8
c8 > c8
c8 > f4
c8 > i4
c8 > b_
c8 > b
c8 > c
c8 > f
c8 > i
c8 > AR_i
c8 > SEQ

c16 > c8
f8 > c8
i8 > c8
c8 > c8
f4 > c8
i4 > c8
b_ > c8
b > c8
c > c8
f > c8
i > c8
AR_i > c8
SEQ > c8

# Float

f8 > f8
f8 > i8
f8 > f4
f8 > i4
f8 > b_
f8 > b
f8 > c
f8 > f
f8 > i
f8 > AR_i
f8 > SEQ

f8 > f8
i8 > f8
f4 > f8
i4 > f8
b_ > f8
b > f8
c > f8
f > f8
i > f8
AR_i > f8
SEQ > f8

f4 > f8
f4 > i8
f4 > f4
f4 > i4
f4 > b_
f4 > b
f4 > c
f4 > f
f4 > i
f4 > AR_i
f4 > SEQ

f8 > f4
i8 > f4
f4 > f4
i4 > f4
b_ > f4
b > f4
c > f4
f > f4
i > f4
AR_i > f4
SEQ > f4

# Int

i8 > i8
i8 > u8
i8 > i4
i8 > u4
i8 > b_
i8 > b
i8 > c
i8 > f
i8 > i
i8 > AR_i
i8 > SEQ

u8 > u8
u8 > i4
u8 > u4
u8 > b_
u8 > b
u8 > c
u8 > f
u8 > i
u8 > AR_i
u8 > SEQ

i8 > i8
u8 > i8
i4 > i8
u4 > i8
b_ > i8
b > i8
c > i8
f > i8
i > i8
AR_i > i8
SEQ > i8

u8 > u8
i4 > u8
u4 > u8
b_ > u8
b > u8
c > u8
f > u8
i > u8
AR_i > u8
SEQ > u8

i4 > i8
i4 > i4
i4 > i
i4 > b_
i4 > b
i4 > AR_i
i4 > SEQ

u4 > i8
u4 > i4
u4 > u8
u4 > u4
u4 > i
u4 > b_
u4 > b
u4 > AR_i
u4 > SEQ

i8 > i4
i4 > i4
i > i4
b_ > i4
b > i4
AR_i > i4
SEQ > i4

i8 > u4
i4 > u4
u8 > u4
u4 > u4
b_ > u4
b > u4
i > u4
AR_i > u4
SEQ > u4

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_custom_dtypes.py ===
from tempfile import NamedTemporaryFile

import pytest
from numpy._core._multiarray_umath import (
    _discover_array_parameters as discover_array_params,
)
from numpy._core._multiarray_umath import _get_sfloat_dtype

import numpy as np
from numpy.testing import assert_array_equal

SF = _get_sfloat_dtype()


class TestSFloat:
    def _get_array(self, scaling, aligned=True):
        if not aligned:
            a = np.empty(3 * 8 + 1, dtype=np.uint8)[1:]
            a = a.view(np.float64)
            a[:] = [1., 2., 3.]
        else:
            a = np.array([1., 2., 3.])

        a *= 1. / scaling  # the casting code also uses the reciprocal.
        return a.view(SF(scaling))

    def test_sfloat_rescaled(self):
        sf = SF(1.)
        sf2 = sf.scaled_by(2.)
        assert sf2.get_scaling() == 2.
        sf6 = sf2.scaled_by(3.)
        assert sf6.get_scaling() == 6.

    def test_class_discovery(self):
        # This does not test much, since we always discover the scaling as 1.
        # But most of NumPy (when writing) does not understand DType classes
        dt, _ = discover_array_params([1., 2., 3.], dtype=SF)
        assert dt == SF(1.)

    @pytest.mark.parametrize("scaling", [1., -1., 2.])
    def test_scaled_float_from_floats(self, scaling):
        a = np.array([1., 2., 3.], dtype=SF(scaling))

        assert a.dtype.get_scaling() == scaling
        assert_array_equal(scaling * a.view(np.float64), [1., 2., 3.])

    def test_repr(self):
        # Check the repr, mainly to cover the code paths:
        assert repr(SF(scaling=1.)) == "_ScaledFloatTestDType(scaling=1.0)"

    def test_dtype_str(self):
        assert SF(1.).str == "_ScaledFloatTestDType(scaling=1.0)"

    def test_dtype_name(self):
        assert SF(1.).name == "_ScaledFloatTestDType64"

    def test_sfloat_structured_dtype_printing(self):
        dt = np.dtype([("id", int), ("value", SF(0.5))])
        # repr of structured dtypes need special handling because the
        # implementation bypasses the object repr
        assert "('value', '_ScaledFloatTestDType64')" in repr(dt)

    @pytest.mark.parametrize("scaling", [1., -1., 2.])
    def test_sfloat_from_float(self, scaling):
        a = np.array([1., 2., 3.]).astype(dtype=SF(scaling))

        assert a.dtype.get_scaling() == scaling
        assert_array_equal(scaling * a.view(np.float64), [1., 2., 3.])

    @pytest.mark.parametrize("aligned", [True, False])
    @pytest.mark.parametrize("scaling", [1., -1., 2.])
    def test_sfloat_getitem(self, aligned, scaling):
        a = self._get_array(1., aligned)
        assert a.tolist() == [1., 2., 3.]

    @pytest.mark.parametrize("aligned", [True, False])
    def test_sfloat_casts(self, aligned):
        a = self._get_array(1., aligned)

        assert np.can_cast(a, SF(-1.), casting="equiv")
        assert not np.can_cast(a, SF(-1.), casting="no")
        na = a.astype(SF(-1.))
        assert_array_equal(-1 * na.view(np.float64), a.view(np.float64))

        assert np.can_cast(a, SF(2.), casting="same_kind")
        assert not np.can_cast(a, SF(2.), casting="safe")
        a2 = a.astype(SF(2.))
        assert_array_equal(2 * a2.view(np.float64), a.view(np.float64))

    @pytest.mark.parametrize("aligned", [True, False])
    def test_sfloat_cast_internal_errors(self, aligned):
        a = self._get_array(2e300, aligned)

        with pytest.raises(TypeError,
                match="error raised inside the core-loop: non-finite factor!"):
            a.astype(SF(2e-300))

    def test_sfloat_promotion(self):
        assert np.result_type(SF(2.), SF(3.)) == SF(3.)
        assert np.result_type(SF(3.), SF(2.)) == SF(3.)
        # Float64 -> SF(1.) and then promotes normally, so both of this work:
        assert np.result_type(SF(3.), np.float64) == SF(3.)
        assert np.result_type(np.float64, SF(0.5)) == SF(1.)

        # Test an undefined promotion:
        with pytest.raises(TypeError):
            np.result_type(SF(1.), np.int64)

    def test_basic_multiply(self):
        a = self._get_array(2.)
        b = self._get_array(4.)

        res = a * b
        # multiplies dtype scaling and content separately:
        assert res.dtype.get_scaling() == 8.
        expected_view = a.view(np.float64) * b.view(np.float64)
        assert_array_equal(res.view(np.float64), expected_view)

    def test_possible_and_impossible_reduce(self):
        # For reductions to work, the first and last operand must have the
        # same dtype.  For this parametric DType that is not necessarily true.
        a = self._get_array(2.)
        # Addition reduction works (as of writing requires to pass initial
        # because setting a scaled-float from the default `0` fails).
        res = np.add.reduce(a, initial=0.)
        assert res == a.astype(np.float64).sum()

        # But each multiplication changes the factor, so a reduction is not
        # possible (the relaxed version of the old refusal to handle any
        # flexible dtype).
        with pytest.raises(TypeError,
                match="the resolved dtypes are not compatible"):
            np.multiply.reduce(a)

    def test_basic_ufunc_at(self):
        float_a = np.array([1., 2., 3.])
        b = self._get_array(2.)

        float_b = b.view(np.float64).copy()
        np.multiply.at(float_b, [1, 1, 1], float_a)
        np.multiply.at(b, [1, 1, 1], float_a)

        assert_array_equal(b.view(np.float64), float_b)

    def test_basic_multiply_promotion(self):
        float_a = np.array([1., 2., 3.])
        b = self._get_array(2.)

        res1 = float_a * b
        res2 = b * float_a

        # one factor is one, so we get the factor of b:
        assert res1.dtype == res2.dtype == b.dtype
        expected_view = float_a * b.view(np.float64)
        assert_array_equal(res1.view(np.float64), expected_view)
        assert_array_equal(res2.view(np.float64), expected_view)

        # Check that promotion works when `out` is used:
        np.multiply(b, float_a, out=res2)
        with pytest.raises(TypeError):
            # The promoter accepts this (maybe it should not), but the SFloat
            # result cannot be cast to integer:
            np.multiply(b, float_a, out=np.arange(3))

    def test_basic_addition(self):
        a = self._get_array(2.)
        b = self._get_array(4.)

        res = a + b
        # addition uses the type promotion rules for the result:
        assert res.dtype == np.result_type(a.dtype, b.dtype)
        expected_view = (a.astype(res.dtype).view(np.float64) +
                         b.astype(res.dtype).view(np.float64))
        assert_array_equal(res.view(np.float64), expected_view)

    def test_addition_cast_safety(self):
        """The addition method is special for the scaled float, because it
        includes the "cast" between different factors, thus cast-safety
        is influenced by the implementation.
        """
        a = self._get_array(2.)
        b = self._get_array(-2.)
        c = self._get_array(3.)

        # sign change is "equiv":
        np.add(a, b, casting="equiv")
        with pytest.raises(TypeError):
            np.add(a, b, casting="no")

        # Different factor is "same_kind" (default) so check that "safe" fails
        with pytest.raises(TypeError):
            np.add(a, c, casting="safe")

        # Check that casting the output fails also (done by the ufunc here)
        with pytest.raises(TypeError):
            np.add(a, a, out=c, casting="safe")

    @pytest.mark.parametrize("ufunc",
            [np.logical_and, np.logical_or, np.logical_xor])
    def test_logical_ufuncs_casts_to_bool(self, ufunc):
        a = self._get_array(2.)
        a[0] = 0.  # make sure first element is considered False.

        float_equiv = a.astype(float)
        expected = ufunc(float_equiv, float_equiv)
        res = ufunc(a, a)
        assert_array_equal(res, expected)

        # also check that the same works for reductions:
        expected = ufunc.reduce(float_equiv)
        res = ufunc.reduce(a)
        assert_array_equal(res, expected)

        # The output casting does not match the bool, bool -> bool loop:
        with pytest.raises(TypeError):
            ufunc(a, a, out=np.empty(a.shape, dtype=int), casting="equiv")

    def test_wrapped_and_wrapped_reductions(self):
        a = self._get_array(2.)
        float_equiv = a.astype(float)

        expected = np.hypot(float_equiv, float_equiv)
        res = np.hypot(a, a)
        assert res.dtype == a.dtype
        res_float = res.view(np.float64) * 2
        assert_array_equal(res_float, expected)

        # Also check reduction (keepdims, due to incorrect getitem)
        res = np.hypot.reduce(a, keepdims=True)
        assert res.dtype == a.dtype
        expected = np.hypot.reduce(float_equiv, keepdims=True)
        assert res.view(np.float64) * 2 == expected

    def test_astype_class(self):
        # Very simple test that we accept `.astype()` also on the class.
        # ScaledFloat always returns the default descriptor, but it does
        # check the relevant code paths.
        arr = np.array([1., 2., 3.], dtype=object)

        res = arr.astype(SF)  # passing the class class
        expected = arr.astype(SF(1.))  # above will have discovered 1. scaling
        assert_array_equal(res.view(np.float64), expected.view(np.float64))

    def test_creation_class(self):
        # passing in a dtype class should return
        # the default descriptor
        arr1 = np.array([1., 2., 3.], dtype=SF)
        assert arr1.dtype == SF(1.)
        arr2 = np.array([1., 2., 3.], dtype=SF(1.))
        assert_array_equal(arr1.view(np.float64), arr2.view(np.float64))
        assert arr1.dtype == arr2.dtype

        assert np.empty(3, dtype=SF).dtype == SF(1.)
        assert np.empty_like(arr1, dtype=SF).dtype == SF(1.)
        assert np.zeros(3, dtype=SF).dtype == SF(1.)
        assert np.zeros_like(arr1, dtype=SF).dtype == SF(1.)

    def test_np_save_load(self):
        # this monkeypatch is needed because pickle
        # uses the repr of a type to reconstruct it
        np._ScaledFloatTestDType = SF

        arr = np.array([1.0, 2.0, 3.0], dtype=SF(1.0))

        # adapted from RoundtripTest.roundtrip in np.save tests
        with NamedTemporaryFile("wb", delete=False, suffix=".npz") as f:
            with pytest.warns(UserWarning) as record:
                np.savez(f.name, arr)

        assert len(record) == 1

        with np.load(f.name, allow_pickle=True) as data:
            larr = data["arr_0"]
        assert_array_equal(arr.view(np.float64), larr.view(np.float64))
        assert larr.dtype == arr.dtype == SF(1.0)

        del np._ScaledFloatTestDType

    def test_flatiter(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=SF(1.0))

        for i, val in enumerate(arr.flat):
            assert arr[i] == val

    @pytest.mark.parametrize(
        "index", [
            [1, 2], ..., slice(None, 2, None),
            np.array([True, True, False]), np.array([0, 1])
        ], ids=["int_list", "ellipsis", "slice", "bool_array", "int_array"])
    def test_flatiter_index(self, index):
        arr = np.array([1.0, 2.0, 3.0], dtype=SF(1.0))
        np.testing.assert_array_equal(
            arr[index].view(np.float64), arr.flat[index].view(np.float64))

        arr2 = arr.copy()
        arr[index] = 5.0
        arr2.flat[index] = 5.0
        np.testing.assert_array_equal(
            arr.view(np.float64), arr2.view(np.float64))

def test_type_pickle():
    # can't actually unpickle, but we can pickle (if in namespace)
    import pickle

    np._ScaledFloatTestDType = SF

    s = pickle.dumps(SF)
    res = pickle.loads(s)
    assert res is SF

    del np._ScaledFloatTestDType


def test_is_numeric():
    assert SF._is_numeric

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimelike_\test_sort_values.py ===
import numpy as np
import pytest

from pandas import (
    DatetimeIndex,
    Index,
    NaT,
    PeriodIndex,
    TimedeltaIndex,
    timedelta_range,
)
import pandas._testing as tm


def check_freq_ascending(ordered, orig, ascending):
    """
    Check the expected freq on a PeriodIndex/DatetimeIndex/TimedeltaIndex
    when the original index is generated (or generate-able) with
    period_range/date_range/timedelta_range.
    """
    if isinstance(ordered, PeriodIndex):
        assert ordered.freq == orig.freq
    elif isinstance(ordered, (DatetimeIndex, TimedeltaIndex)):
        if ascending:
            assert ordered.freq.n == orig.freq.n
        else:
            assert ordered.freq.n == -1 * orig.freq.n


def check_freq_nonmonotonic(ordered, orig):
    """
    Check the expected freq on a PeriodIndex/DatetimeIndex/TimedeltaIndex
    when the original index is _not_ generated (or generate-able) with
    period_range/date_range//timedelta_range.
    """
    if isinstance(ordered, PeriodIndex):
        assert ordered.freq == orig.freq
    elif isinstance(ordered, (DatetimeIndex, TimedeltaIndex)):
        assert ordered.freq is None


class TestSortValues:
    @pytest.fixture(params=[DatetimeIndex, TimedeltaIndex, PeriodIndex])
    def non_monotonic_idx(self, request):
        if request.param is DatetimeIndex:
            return DatetimeIndex(["2000-01-04", "2000-01-01", "2000-01-02"])
        elif request.param is PeriodIndex:
            dti = DatetimeIndex(["2000-01-04", "2000-01-01", "2000-01-02"])
            return dti.to_period("D")
        else:
            return TimedeltaIndex(
                ["1 day 00:00:05", "1 day 00:00:01", "1 day 00:00:02"]
            )

    def test_argmin_argmax(self, non_monotonic_idx):
        assert non_monotonic_idx.argmin() == 1
        assert non_monotonic_idx.argmax() == 0

    def test_sort_values(self, non_monotonic_idx):
        idx = non_monotonic_idx
        ordered = idx.sort_values()
        assert ordered.is_monotonic_increasing
        ordered = idx.sort_values(ascending=False)
        assert ordered[::-1].is_monotonic_increasing

        ordered, dexer = idx.sort_values(return_indexer=True)
        assert ordered.is_monotonic_increasing
        tm.assert_numpy_array_equal(dexer, np.array([1, 2, 0], dtype=np.intp))

        ordered, dexer = idx.sort_values(return_indexer=True, ascending=False)
        assert ordered[::-1].is_monotonic_increasing
        tm.assert_numpy_array_equal(dexer, np.array([0, 2, 1], dtype=np.intp))

    def check_sort_values_with_freq(self, idx):
        ordered = idx.sort_values()
        tm.assert_index_equal(ordered, idx)
        check_freq_ascending(ordered, idx, True)

        ordered = idx.sort_values(ascending=False)
        expected = idx[::-1]
        tm.assert_index_equal(ordered, expected)
        check_freq_ascending(ordered, idx, False)

        ordered, indexer = idx.sort_values(return_indexer=True)
        tm.assert_index_equal(ordered, idx)
        tm.assert_numpy_array_equal(indexer, np.array([0, 1, 2], dtype=np.intp))
        check_freq_ascending(ordered, idx, True)

        ordered, indexer = idx.sort_values(return_indexer=True, ascending=False)
        expected = idx[::-1]
        tm.assert_index_equal(ordered, expected)
        tm.assert_numpy_array_equal(indexer, np.array([2, 1, 0], dtype=np.intp))
        check_freq_ascending(ordered, idx, False)

    @pytest.mark.parametrize("freq", ["D", "h"])
    def test_sort_values_with_freq_timedeltaindex(self, freq):
        # GH#10295
        idx = timedelta_range(start=f"1{freq}", periods=3, freq=freq).rename("idx")

        self.check_sort_values_with_freq(idx)

    @pytest.mark.parametrize(
        "idx",
        [
            DatetimeIndex(
                ["2011-01-01", "2011-01-02", "2011-01-03"], freq="D", name="idx"
            ),
            DatetimeIndex(
                ["2011-01-01 09:00", "2011-01-01 10:00", "2011-01-01 11:00"],
                freq="h",
                name="tzidx",
                tz="Asia/Tokyo",
            ),
        ],
    )
    def test_sort_values_with_freq_datetimeindex(self, idx):
        self.check_sort_values_with_freq(idx)

    @pytest.mark.parametrize("freq", ["D", "2D", "4D"])
    def test_sort_values_with_freq_periodindex(self, freq):
        # here with_freq refers to being period_range-like
        idx = PeriodIndex(
            ["2011-01-01", "2011-01-02", "2011-01-03"], freq=freq, name="idx"
        )
        self.check_sort_values_with_freq(idx)

    @pytest.mark.parametrize(
        "idx",
        [
            PeriodIndex(["2011", "2012", "2013"], name="pidx", freq="Y"),
            Index([2011, 2012, 2013], name="idx"),  # for compatibility check
        ],
    )
    def test_sort_values_with_freq_periodindex2(self, idx):
        # here with_freq indicates this is period_range-like
        self.check_sort_values_with_freq(idx)

    def check_sort_values_without_freq(self, idx, expected):
        ordered = idx.sort_values(na_position="first")
        tm.assert_index_equal(ordered, expected)
        check_freq_nonmonotonic(ordered, idx)

        if not idx.isna().any():
            ordered = idx.sort_values()
            tm.assert_index_equal(ordered, expected)
            check_freq_nonmonotonic(ordered, idx)

        ordered = idx.sort_values(ascending=False)
        tm.assert_index_equal(ordered, expected[::-1])
        check_freq_nonmonotonic(ordered, idx)

        ordered, indexer = idx.sort_values(return_indexer=True, na_position="first")
        tm.assert_index_equal(ordered, expected)

        exp = np.array([0, 4, 3, 1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, exp)
        check_freq_nonmonotonic(ordered, idx)

        if not idx.isna().any():
            ordered, indexer = idx.sort_values(return_indexer=True)
            tm.assert_index_equal(ordered, expected)

            exp = np.array([0, 4, 3, 1, 2], dtype=np.intp)
            tm.assert_numpy_array_equal(indexer, exp)
            check_freq_nonmonotonic(ordered, idx)

        ordered, indexer = idx.sort_values(return_indexer=True, ascending=False)
        tm.assert_index_equal(ordered, expected[::-1])

        exp = np.array([2, 1, 3, 0, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, exp)
        check_freq_nonmonotonic(ordered, idx)

    def test_sort_values_without_freq_timedeltaindex(self):
        # GH#10295

        idx = TimedeltaIndex(
            ["1 hour", "3 hour", "5 hour", "2 hour ", "1 hour"], name="idx1"
        )
        expected = TimedeltaIndex(
            ["1 hour", "1 hour", "2 hour", "3 hour", "5 hour"], name="idx1"
        )
        self.check_sort_values_without_freq(idx, expected)

    @pytest.mark.parametrize(
        "index_dates,expected_dates",
        [
            (
                ["2011-01-01", "2011-01-03", "2011-01-05", "2011-01-02", "2011-01-01"],
                ["2011-01-01", "2011-01-01", "2011-01-02", "2011-01-03", "2011-01-05"],
            ),
            (
                ["2011-01-01", "2011-01-03", "2011-01-05", "2011-01-02", "2011-01-01"],
                ["2011-01-01", "2011-01-01", "2011-01-02", "2011-01-03", "2011-01-05"],
            ),
            (
                [NaT, "2011-01-03", "2011-01-05", "2011-01-02", NaT],
                [NaT, NaT, "2011-01-02", "2011-01-03", "2011-01-05"],
            ),
        ],
    )
    def test_sort_values_without_freq_datetimeindex(
        self, index_dates, expected_dates, tz_naive_fixture
    ):
        tz = tz_naive_fixture

        # without freq
        idx = DatetimeIndex(index_dates, tz=tz, name="idx")
        expected = DatetimeIndex(expected_dates, tz=tz, name="idx")

        self.check_sort_values_without_freq(idx, expected)

    @pytest.mark.parametrize(
        "idx,expected",
        [
            (
                PeriodIndex(
                    [
                        "2011-01-01",
                        "2011-01-03",
                        "2011-01-05",
                        "2011-01-02",
                        "2011-01-01",
                    ],
                    freq="D",
                    name="idx1",
                ),
                PeriodIndex(
                    [
                        "2011-01-01",
                        "2011-01-01",
                        "2011-01-02",
                        "2011-01-03",
                        "2011-01-05",
                    ],
                    freq="D",
                    name="idx1",
                ),
            ),
            (
                PeriodIndex(
                    [
                        "2011-01-01",
                        "2011-01-03",
                        "2011-01-05",
                        "2011-01-02",
                        "2011-01-01",
                    ],
                    freq="D",
                    name="idx2",
                ),
                PeriodIndex(
                    [
                        "2011-01-01",
                        "2011-01-01",
                        "2011-01-02",
                        "2011-01-03",
                        "2011-01-05",
                    ],
                    freq="D",
                    name="idx2",
                ),
            ),
            (
                PeriodIndex(
                    [NaT, "2011-01-03", "2011-01-05", "2011-01-02", NaT],
                    freq="D",
                    name="idx3",
                ),
                PeriodIndex(
                    [NaT, NaT, "2011-01-02", "2011-01-03", "2011-01-05"],
                    freq="D",
                    name="idx3",
                ),
            ),
            (
                PeriodIndex(
                    ["2011", "2013", "2015", "2012", "2011"], name="pidx", freq="Y"
                ),
                PeriodIndex(
                    ["2011", "2011", "2012", "2013", "2015"], name="pidx", freq="Y"
                ),
            ),
            (
                # For compatibility check
                Index([2011, 2013, 2015, 2012, 2011], name="idx"),
                Index([2011, 2011, 2012, 2013, 2015], name="idx"),
            ),
        ],
    )
    def test_sort_values_without_freq_periodindex(self, idx, expected):
        # here without_freq means not generateable by period_range
        self.check_sort_values_without_freq(idx, expected)

    def test_sort_values_without_freq_periodindex_nat(self):
        # doesn't quite fit into check_sort_values_without_freq
        idx = PeriodIndex(["2011", "2013", "NaT", "2011"], name="pidx", freq="D")
        expected = PeriodIndex(["NaT", "2011", "2011", "2013"], name="pidx", freq="D")

        ordered = idx.sort_values(na_position="first")
        tm.assert_index_equal(ordered, expected)
        check_freq_nonmonotonic(ordered, idx)

        ordered = idx.sort_values(ascending=False)
        tm.assert_index_equal(ordered, expected[::-1])
        check_freq_nonmonotonic(ordered, idx)


def test_order_stability_compat():
    # GH#35922. sort_values is stable both for normal and datetime-like Index
    pidx = PeriodIndex(["2011", "2013", "2015", "2012", "2011"], name="pidx", freq="Y")
    iidx = Index([2011, 2013, 2015, 2012, 2011], name="idx")
    ordered1, indexer1 = pidx.sort_values(return_indexer=True, ascending=False)
    ordered2, indexer2 = iidx.sort_values(return_indexer=True, ascending=False)
    tm.assert_numpy_array_equal(indexer1, indexer2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\offsets\test_business_quarter.py ===
"""
Tests for the following offsets:
- BQuarterBegin
- BQuarterEnd
"""
from __future__ import annotations

from datetime import datetime

import pytest

import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
    assert_is_on_offset,
    assert_offset_equal,
)

from pandas.tseries.offsets import (
    BQuarterBegin,
    BQuarterEnd,
)


def test_quarterly_dont_normalize():
    date = datetime(2012, 3, 31, 5, 30)

    offsets = (BQuarterEnd, BQuarterBegin)

    for klass in offsets:
        result = date + klass()
        assert result.time() == date.time()


@pytest.mark.parametrize("offset", [BQuarterBegin(), BQuarterEnd()])
def test_on_offset(offset):
    dates = [
        datetime(2016, m, d)
        for m in [10, 11, 12]
        for d in [1, 2, 3, 28, 29, 30, 31]
        if not (m == 11 and d == 31)
    ]
    for date in dates:
        res = offset.is_on_offset(date)
        slow_version = date == (date + offset) - offset
        assert res == slow_version


class TestBQuarterBegin:
    def test_repr(self):
        expected = "<BusinessQuarterBegin: startingMonth=3>"
        assert repr(BQuarterBegin()) == expected
        expected = "<BusinessQuarterBegin: startingMonth=3>"
        assert repr(BQuarterBegin(startingMonth=3)) == expected
        expected = "<BusinessQuarterBegin: startingMonth=1>"
        assert repr(BQuarterBegin(startingMonth=1)) == expected

    def test_is_anchored(self):
        msg = "BQuarterBegin.is_anchored is deprecated "

        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert BQuarterBegin(startingMonth=1).is_anchored()
            assert BQuarterBegin().is_anchored()
            assert not BQuarterBegin(2, startingMonth=1).is_anchored()

    def test_offset_corner_case(self):
        # corner
        offset = BQuarterBegin(n=-1, startingMonth=1)
        assert datetime(2007, 4, 3) + offset == datetime(2007, 4, 2)

    offset_cases = []
    offset_cases.append(
        (
            BQuarterBegin(startingMonth=1),
            {
                datetime(2008, 1, 1): datetime(2008, 4, 1),
                datetime(2008, 1, 31): datetime(2008, 4, 1),
                datetime(2008, 2, 15): datetime(2008, 4, 1),
                datetime(2008, 2, 29): datetime(2008, 4, 1),
                datetime(2008, 3, 15): datetime(2008, 4, 1),
                datetime(2008, 3, 31): datetime(2008, 4, 1),
                datetime(2008, 4, 15): datetime(2008, 7, 1),
                datetime(2007, 3, 15): datetime(2007, 4, 2),
                datetime(2007, 2, 28): datetime(2007, 4, 2),
                datetime(2007, 1, 1): datetime(2007, 4, 2),
                datetime(2007, 4, 15): datetime(2007, 7, 2),
                datetime(2007, 7, 1): datetime(2007, 7, 2),
                datetime(2007, 4, 1): datetime(2007, 4, 2),
                datetime(2007, 4, 2): datetime(2007, 7, 2),
                datetime(2008, 4, 30): datetime(2008, 7, 1),
            },
        )
    )

    offset_cases.append(
        (
            BQuarterBegin(startingMonth=2),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 1),
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2008, 1, 15): datetime(2008, 2, 1),
                datetime(2008, 2, 29): datetime(2008, 5, 1),
                datetime(2008, 3, 15): datetime(2008, 5, 1),
                datetime(2008, 3, 31): datetime(2008, 5, 1),
                datetime(2008, 4, 15): datetime(2008, 5, 1),
                datetime(2008, 8, 15): datetime(2008, 11, 3),
                datetime(2008, 9, 15): datetime(2008, 11, 3),
                datetime(2008, 11, 1): datetime(2008, 11, 3),
                datetime(2008, 4, 30): datetime(2008, 5, 1),
            },
        )
    )

    offset_cases.append(
        (
            BQuarterBegin(startingMonth=1, n=0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2007, 12, 31): datetime(2008, 1, 1),
                datetime(2008, 2, 15): datetime(2008, 4, 1),
                datetime(2008, 2, 29): datetime(2008, 4, 1),
                datetime(2008, 1, 15): datetime(2008, 4, 1),
                datetime(2008, 2, 27): datetime(2008, 4, 1),
                datetime(2008, 3, 15): datetime(2008, 4, 1),
                datetime(2007, 4, 1): datetime(2007, 4, 2),
                datetime(2007, 4, 2): datetime(2007, 4, 2),
                datetime(2007, 7, 1): datetime(2007, 7, 2),
                datetime(2007, 4, 15): datetime(2007, 7, 2),
                datetime(2007, 7, 2): datetime(2007, 7, 2),
            },
        )
    )

    offset_cases.append(
        (
            BQuarterBegin(startingMonth=1, n=-1),
            {
                datetime(2008, 1, 1): datetime(2007, 10, 1),
                datetime(2008, 1, 31): datetime(2008, 1, 1),
                datetime(2008, 2, 15): datetime(2008, 1, 1),
                datetime(2008, 2, 29): datetime(2008, 1, 1),
                datetime(2008, 3, 15): datetime(2008, 1, 1),
                datetime(2008, 3, 31): datetime(2008, 1, 1),
                datetime(2008, 4, 15): datetime(2008, 4, 1),
                datetime(2007, 7, 3): datetime(2007, 7, 2),
                datetime(2007, 4, 3): datetime(2007, 4, 2),
                datetime(2007, 7, 2): datetime(2007, 4, 2),
                datetime(2008, 4, 1): datetime(2008, 1, 1),
            },
        )
    )

    offset_cases.append(
        (
            BQuarterBegin(startingMonth=1, n=2),
            {
                datetime(2008, 1, 1): datetime(2008, 7, 1),
                datetime(2008, 1, 15): datetime(2008, 7, 1),
                datetime(2008, 2, 29): datetime(2008, 7, 1),
                datetime(2008, 3, 15): datetime(2008, 7, 1),
                datetime(2007, 3, 31): datetime(2007, 7, 2),
                datetime(2007, 4, 15): datetime(2007, 10, 1),
                datetime(2008, 4, 30): datetime(2008, 10, 1),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)


class TestBQuarterEnd:
    def test_repr(self):
        expected = "<BusinessQuarterEnd: startingMonth=3>"
        assert repr(BQuarterEnd()) == expected
        expected = "<BusinessQuarterEnd: startingMonth=3>"
        assert repr(BQuarterEnd(startingMonth=3)) == expected
        expected = "<BusinessQuarterEnd: startingMonth=1>"
        assert repr(BQuarterEnd(startingMonth=1)) == expected

    def test_is_anchored(self):
        msg = "BQuarterEnd.is_anchored is deprecated "

        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert BQuarterEnd(startingMonth=1).is_anchored()
            assert BQuarterEnd().is_anchored()
            assert not BQuarterEnd(2, startingMonth=1).is_anchored()

    def test_offset_corner_case(self):
        # corner
        offset = BQuarterEnd(n=-1, startingMonth=1)
        assert datetime(2010, 1, 31) + offset == datetime(2010, 1, 29)

    offset_cases = []
    offset_cases.append(
        (
            BQuarterEnd(startingMonth=1),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 4, 30),
                datetime(2008, 2, 15): datetime(2008, 4, 30),
                datetime(2008, 2, 29): datetime(2008, 4, 30),
                datetime(2008, 3, 15): datetime(2008, 4, 30),
                datetime(2008, 3, 31): datetime(2008, 4, 30),
                datetime(2008, 4, 15): datetime(2008, 4, 30),
                datetime(2008, 4, 30): datetime(2008, 7, 31),
            },
        )
    )

    offset_cases.append(
        (
            BQuarterEnd(startingMonth=2),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 29),
                datetime(2008, 1, 31): datetime(2008, 2, 29),
                datetime(2008, 2, 15): datetime(2008, 2, 29),
                datetime(2008, 2, 29): datetime(2008, 5, 30),
                datetime(2008, 3, 15): datetime(2008, 5, 30),
                datetime(2008, 3, 31): datetime(2008, 5, 30),
                datetime(2008, 4, 15): datetime(2008, 5, 30),
                datetime(2008, 4, 30): datetime(2008, 5, 30),
            },
        )
    )

    offset_cases.append(
        (
            BQuarterEnd(startingMonth=1, n=0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 1, 31),
                datetime(2008, 2, 15): datetime(2008, 4, 30),
                datetime(2008, 2, 29): datetime(2008, 4, 30),
                datetime(2008, 3, 15): datetime(2008, 4, 30),
                datetime(2008, 3, 31): datetime(2008, 4, 30),
                datetime(2008, 4, 15): datetime(2008, 4, 30),
                datetime(2008, 4, 30): datetime(2008, 4, 30),
            },
        )
    )

    offset_cases.append(
        (
            BQuarterEnd(startingMonth=1, n=-1),
            {
                datetime(2008, 1, 1): datetime(2007, 10, 31),
                datetime(2008, 1, 31): datetime(2007, 10, 31),
                datetime(2008, 2, 15): datetime(2008, 1, 31),
                datetime(2008, 2, 29): datetime(2008, 1, 31),
                datetime(2008, 3, 15): datetime(2008, 1, 31),
                datetime(2008, 3, 31): datetime(2008, 1, 31),
                datetime(2008, 4, 15): datetime(2008, 1, 31),
                datetime(2008, 4, 30): datetime(2008, 1, 31),
            },
        )
    )

    offset_cases.append(
        (
            BQuarterEnd(startingMonth=1, n=2),
            {
                datetime(2008, 1, 31): datetime(2008, 7, 31),
                datetime(2008, 2, 15): datetime(2008, 7, 31),
                datetime(2008, 2, 29): datetime(2008, 7, 31),
                datetime(2008, 3, 15): datetime(2008, 7, 31),
                datetime(2008, 3, 31): datetime(2008, 7, 31),
                datetime(2008, 4, 15): datetime(2008, 7, 31),
                datetime(2008, 4, 30): datetime(2008, 10, 31),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (BQuarterEnd(1, startingMonth=1), datetime(2008, 1, 31), True),
        (BQuarterEnd(1, startingMonth=1), datetime(2007, 12, 31), False),
        (BQuarterEnd(1, startingMonth=1), datetime(2008, 2, 29), False),
        (BQuarterEnd(1, startingMonth=1), datetime(2007, 3, 30), False),
        (BQuarterEnd(1, startingMonth=1), datetime(2007, 3, 31), False),
        (BQuarterEnd(1, startingMonth=1), datetime(2008, 4, 30), True),
        (BQuarterEnd(1, startingMonth=1), datetime(2008, 5, 30), False),
        (BQuarterEnd(1, startingMonth=1), datetime(2007, 6, 29), False),
        (BQuarterEnd(1, startingMonth=1), datetime(2007, 6, 30), False),
        (BQuarterEnd(1, startingMonth=2), datetime(2008, 1, 31), False),
        (BQuarterEnd(1, startingMonth=2), datetime(2007, 12, 31), False),
        (BQuarterEnd(1, startingMonth=2), datetime(2008, 2, 29), True),
        (BQuarterEnd(1, startingMonth=2), datetime(2007, 3, 30), False),
        (BQuarterEnd(1, startingMonth=2), datetime(2007, 3, 31), False),
        (BQuarterEnd(1, startingMonth=2), datetime(2008, 4, 30), False),
        (BQuarterEnd(1, startingMonth=2), datetime(2008, 5, 30), True),
        (BQuarterEnd(1, startingMonth=2), datetime(2007, 6, 29), False),
        (BQuarterEnd(1, startingMonth=2), datetime(2007, 6, 30), False),
        (BQuarterEnd(1, startingMonth=3), datetime(2008, 1, 31), False),
        (BQuarterEnd(1, startingMonth=3), datetime(2007, 12, 31), True),
        (BQuarterEnd(1, startingMonth=3), datetime(2008, 2, 29), False),
        (BQuarterEnd(1, startingMonth=3), datetime(2007, 3, 30), True),
        (BQuarterEnd(1, startingMonth=3), datetime(2007, 3, 31), False),
        (BQuarterEnd(1, startingMonth=3), datetime(2008, 4, 30), False),
        (BQuarterEnd(1, startingMonth=3), datetime(2008, 5, 30), False),
        (BQuarterEnd(1, startingMonth=3), datetime(2007, 6, 29), True),
        (BQuarterEnd(1, startingMonth=3), datetime(2007, 6, 30), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\tests\test_kane3.py ===
from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import acos, sin, cos
from sympy.matrices.dense import Matrix
from sympy.physics.mechanics import (ReferenceFrame, dynamicsymbols,
                                     KanesMethod, inertia, Point, RigidBody,
                                     dot)
from sympy.testing.pytest import slow


@slow
def test_bicycle():
    # Code to get equations of motion for a bicycle modeled as in:
    # J.P Meijaard, Jim M Papadopoulos, Andy Ruina and A.L Schwab. Linearized
    # dynamics equations for the balance and steer of a bicycle: a benchmark
    # and review. Proceedings of The Royal Society (2007) 463, 1955-1982
    # doi: 10.1098/rspa.2007.1857

    # Note that this code has been crudely ported from Autolev, which is the
    # reason for some of the unusual naming conventions. It was purposefully as
    # similar as possible in order to aide debugging.

    # Declare Coordinates & Speeds
    # Simple definitions for qdots - qd = u
    # Speeds are:
    # - u1: yaw frame ang. rate
    # - u2: roll frame ang. rate
    # - u3: rear wheel frame ang. rate (spinning motion)
    # - u4: frame ang. rate (pitching motion)
    # - u5: steering frame ang. rate
    # - u6: front wheel ang. rate (spinning motion)
    # Wheel positions are ignorable coordinates, so they are not introduced.
    q1, q2, q4, q5 = dynamicsymbols('q1 q2 q4 q5')
    q1d, q2d, q4d, q5d = dynamicsymbols('q1 q2 q4 q5', 1)
    u1, u2, u3, u4, u5, u6 = dynamicsymbols('u1 u2 u3 u4 u5 u6')
    u1d, u2d, u3d, u4d, u5d, u6d = dynamicsymbols('u1 u2 u3 u4 u5 u6', 1)

    # Declare System's Parameters
    WFrad, WRrad, htangle, forkoffset = symbols('WFrad WRrad htangle forkoffset')
    forklength, framelength, forkcg1 = symbols('forklength framelength forkcg1')
    forkcg3, framecg1, framecg3, Iwr11 = symbols('forkcg3 framecg1 framecg3 Iwr11')
    Iwr22, Iwf11, Iwf22, Iframe11 = symbols('Iwr22 Iwf11 Iwf22 Iframe11')
    Iframe22, Iframe33, Iframe31, Ifork11 = symbols('Iframe22 Iframe33 Iframe31 Ifork11')
    Ifork22, Ifork33, Ifork31, g = symbols('Ifork22 Ifork33 Ifork31 g')
    mframe, mfork, mwf, mwr = symbols('mframe mfork mwf mwr')

    # Set up reference frames for the system
    # N - inertial
    # Y - yaw
    # R - roll
    # WR - rear wheel, rotation angle is ignorable coordinate so not oriented
    # Frame - bicycle frame
    # TempFrame - statically rotated frame for easier reference inertia definition
    # Fork - bicycle fork
    # TempFork - statically rotated frame for easier reference inertia definition
    # WF - front wheel, again posses a ignorable coordinate
    N = ReferenceFrame('N')
    Y = N.orientnew('Y', 'Axis', [q1, N.z])
    R = Y.orientnew('R', 'Axis', [q2, Y.x])
    Frame = R.orientnew('Frame', 'Axis', [q4 + htangle, R.y])
    WR = ReferenceFrame('WR')
    TempFrame = Frame.orientnew('TempFrame', 'Axis', [-htangle, Frame.y])
    Fork = Frame.orientnew('Fork', 'Axis', [q5, Frame.x])
    TempFork = Fork.orientnew('TempFork', 'Axis', [-htangle, Fork.y])
    WF = ReferenceFrame('WF')

    # Kinematics of the Bicycle First block of code is forming the positions of
    # the relevant points
    # rear wheel contact -> rear wheel mass center -> frame mass center +
    # frame/fork connection -> fork mass center + front wheel mass center ->
    # front wheel contact point
    WR_cont = Point('WR_cont')
    WR_mc = WR_cont.locatenew('WR_mc', WRrad * R.z)
    Steer = WR_mc.locatenew('Steer', framelength * Frame.z)
    Frame_mc = WR_mc.locatenew('Frame_mc', - framecg1 * Frame.x
                                           + framecg3 * Frame.z)
    Fork_mc = Steer.locatenew('Fork_mc', - forkcg1 * Fork.x
                                         + forkcg3 * Fork.z)
    WF_mc = Steer.locatenew('WF_mc', forklength * Fork.x + forkoffset * Fork.z)
    WF_cont = WF_mc.locatenew('WF_cont', WFrad * (dot(Fork.y, Y.z) * Fork.y -
                                                  Y.z).normalize())

    # Set the angular velocity of each frame.
    # Angular accelerations end up being calculated automatically by
    # differentiating the angular velocities when first needed.
    # u1 is yaw rate
    # u2 is roll rate
    # u3 is rear wheel rate
    # u4 is frame pitch rate
    # u5 is fork steer rate
    # u6 is front wheel rate
    Y.set_ang_vel(N, u1 * Y.z)
    R.set_ang_vel(Y, u2 * R.x)
    WR.set_ang_vel(Frame, u3 * Frame.y)
    Frame.set_ang_vel(R, u4 * Frame.y)
    Fork.set_ang_vel(Frame, u5 * Fork.x)
    WF.set_ang_vel(Fork, u6 * Fork.y)

    # Form the velocities of the previously defined points, using the 2 - point
    # theorem (written out by hand here).  Accelerations again are calculated
    # automatically when first needed.
    WR_cont.set_vel(N, 0)
    WR_mc.v2pt_theory(WR_cont, N, WR)
    Steer.v2pt_theory(WR_mc, N, Frame)
    Frame_mc.v2pt_theory(WR_mc, N, Frame)
    Fork_mc.v2pt_theory(Steer, N, Fork)
    WF_mc.v2pt_theory(Steer, N, Fork)
    WF_cont.v2pt_theory(WF_mc, N, WF)

    # Sets the inertias of each body. Uses the inertia frame to construct the
    # inertia dyadics. Wheel inertias are only defined by principle moments of
    # inertia, and are in fact constant in the frame and fork reference frames;
    # it is for this reason that the orientations of the wheels does not need
    # to be defined. The frame and fork inertias are defined in the 'Temp'
    # frames which are fixed to the appropriate body frames; this is to allow
    # easier input of the reference values of the benchmark paper. Note that
    # due to slightly different orientations, the products of inertia need to
    # have their signs flipped; this is done later when entering the numerical
    # value.

    Frame_I = (inertia(TempFrame, Iframe11, Iframe22, Iframe33, 0, 0, Iframe31), Frame_mc)
    Fork_I = (inertia(TempFork, Ifork11, Ifork22, Ifork33, 0, 0, Ifork31), Fork_mc)
    WR_I = (inertia(Frame, Iwr11, Iwr22, Iwr11), WR_mc)
    WF_I = (inertia(Fork, Iwf11, Iwf22, Iwf11), WF_mc)

    # Declaration of the RigidBody containers. ::

    BodyFrame = RigidBody('BodyFrame', Frame_mc, Frame, mframe, Frame_I)
    BodyFork = RigidBody('BodyFork', Fork_mc, Fork, mfork, Fork_I)
    BodyWR = RigidBody('BodyWR', WR_mc, WR, mwr, WR_I)
    BodyWF = RigidBody('BodyWF', WF_mc, WF, mwf, WF_I)

    # The kinematic differential equations; they are defined quite simply. Each
    # entry in this list is equal to zero.
    kd = [q1d - u1, q2d - u2, q4d - u4, q5d - u5]

    # The nonholonomic constraints are the velocity of the front wheel contact
    # point dotted into the X, Y, and Z directions; the yaw frame is used as it
    # is "closer" to the front wheel (1 less DCM connecting them). These
    # constraints force the velocity of the front wheel contact point to be 0
    # in the inertial frame; the X and Y direction constraints enforce a
    # "no-slip" condition, and the Z direction constraint forces the front
    # wheel contact point to not move away from the ground frame, essentially
    # replicating the holonomic constraint which does not allow the frame pitch
    # to change in an invalid fashion.

    conlist_speed = [WF_cont.vel(N) & Y.x, WF_cont.vel(N) & Y.y, WF_cont.vel(N) & Y.z]

    # The holonomic constraint is that the position from the rear wheel contact
    # point to the front wheel contact point when dotted into the
    # normal-to-ground plane direction must be zero; effectively that the front
    # and rear wheel contact points are always touching the ground plane. This
    # is actually not part of the dynamic equations, but instead is necessary
    # for the lineraization process.

    conlist_coord = [WF_cont.pos_from(WR_cont) & Y.z]

    # The force list; each body has the appropriate gravitational force applied
    # at its mass center.
    FL = [(Frame_mc, -mframe * g * Y.z),
        (Fork_mc, -mfork * g * Y.z),
        (WF_mc, -mwf * g * Y.z),
        (WR_mc, -mwr * g * Y.z)]
    BL = [BodyFrame, BodyFork, BodyWR, BodyWF]


    # The N frame is the inertial frame, coordinates are supplied in the order
    # of independent, dependent coordinates, as are the speeds. The kinematic
    # differential equation are also entered here.  Here the dependent speeds
    # are specified, in the same order they were provided in earlier, along
    # with the non-holonomic constraints.  The dependent coordinate is also
    # provided, with the holonomic constraint.  Again, this is only provided
    # for the linearization process.

    KM = KanesMethod(N, q_ind=[q1, q2, q5],
            q_dependent=[q4], configuration_constraints=conlist_coord,
            u_ind=[u2, u3, u5],
            u_dependent=[u1, u4, u6], velocity_constraints=conlist_speed,
            kd_eqs=kd,
            constraint_solver="CRAMER")
    (fr, frstar) = KM.kanes_equations(BL, FL)

    # This is the start of entering in the numerical values from the benchmark
    # paper to validate the eigen values of the linearized equations from this
    # model to the reference eigen values. Look at the aforementioned paper for
    # more information. Some of these are intermediate values, used to
    # transform values from the paper into the coordinate systems used in this
    # model.
    PaperRadRear                    =  0.3
    PaperRadFront                   =  0.35
    HTA                             =  (pi / 2 - pi / 10).evalf()
    TrailPaper                      =  0.08
    rake                            =  (-(TrailPaper*sin(HTA)-(PaperRadFront*cos(HTA)))).evalf()
    PaperWb                         =  1.02
    PaperFrameCgX                   =  0.3
    PaperFrameCgZ                   =  0.9
    PaperForkCgX                    =  0.9
    PaperForkCgZ                    =  0.7
    FrameLength                     =  (PaperWb*sin(HTA)-(rake-(PaperRadFront-PaperRadRear)*cos(HTA))).evalf()
    FrameCGNorm                     =  ((PaperFrameCgZ - PaperRadRear-(PaperFrameCgX/sin(HTA))*cos(HTA))*sin(HTA)).evalf()
    FrameCGPar                      =  (PaperFrameCgX / sin(HTA) + (PaperFrameCgZ - PaperRadRear - PaperFrameCgX / sin(HTA) * cos(HTA)) * cos(HTA)).evalf()
    tempa                           =  (PaperForkCgZ - PaperRadFront)
    tempb                           =  (PaperWb-PaperForkCgX)
    tempc                           =  (sqrt(tempa**2+tempb**2)).evalf()
    PaperForkL                      =  (PaperWb*cos(HTA)-(PaperRadFront-PaperRadRear)*sin(HTA)).evalf()
    ForkCGNorm                      =  (rake+(tempc * sin(pi/2-HTA-acos(tempa/tempc)))).evalf()
    ForkCGPar                       =  (tempc * cos((pi/2-HTA)-acos(tempa/tempc))-PaperForkL).evalf()

    # Here is the final assembly of the numerical values. The symbol 'v' is the
    # forward speed of the bicycle (a concept which only makes sense in the
    # upright, static equilibrium case?). These are in a dictionary which will
    # later be substituted in. Again the sign on the *product* of inertia
    # values is flipped here, due to different orientations of coordinate
    # systems.
    v = symbols('v')
    val_dict = {WFrad: PaperRadFront,
                WRrad: PaperRadRear,
                htangle: HTA,
                forkoffset: rake,
                forklength: PaperForkL,
                framelength: FrameLength,
                forkcg1: ForkCGPar,
                forkcg3: ForkCGNorm,
                framecg1: FrameCGNorm,
                framecg3: FrameCGPar,
                Iwr11: 0.0603,
                Iwr22: 0.12,
                Iwf11: 0.1405,
                Iwf22: 0.28,
                Ifork11: 0.05892,
                Ifork22: 0.06,
                Ifork33: 0.00708,
                Ifork31: 0.00756,
                Iframe11: 9.2,
                Iframe22: 11,
                Iframe33: 2.8,
                Iframe31: -2.4,
                mfork: 4,
                mframe: 85,
                mwf: 3,
                mwr: 2,
                g: 9.81,
                q1: 0,
                q2: 0,
                q4: 0,
                q5: 0,
                u1: 0,
                u2: 0,
                u3: v / PaperRadRear,
                u4: 0,
                u5: 0,
                u6: v / PaperRadFront}

    # Linearizes the forcing vector; the equations are set up as MM udot =
    # forcing, where MM is the mass matrix, udot is the vector representing the
    # time derivatives of the generalized speeds, and forcing is a vector which
    # contains both external forcing terms and internal forcing terms, such as
    # centripital or coriolis forces.  This actually returns a matrix with as
    # many rows as *total* coordinates and speeds, but only as many columns as
    # independent coordinates and speeds.

    A, B, _ = KM.linearize(
        A_and_B=True,
        op_point={
            # Operating points for the accelerations are required for the
            # linearizer to eliminate u' terms showing up in the coefficient
            # matrices.
            u1.diff(): 0,
            u2.diff(): 0,
            u3.diff(): 0,
            u4.diff(): 0,
            u5.diff(): 0,
            u6.diff(): 0,
            u1: 0,
            u2: 0,
            u3: v / PaperRadRear,
            u4: 0,
            u5: 0,
            u6: v / PaperRadFront,
            q1: 0,
            q2: 0,
            q4: 0,
            q5: 0,
        },
        linear_solver="CRAMER",
    )
    # As mentioned above, the size of the linearized forcing terms is expanded
    # to include both q's and u's, so the mass matrix must have this done as
    # well.  This will likely be changed to be part of the linearized process,
    # for future reference.
    A_s = A.xreplace(val_dict)
    B_s = B.xreplace(val_dict)

    A_s = A_s.evalf()
    B_s = B_s.evalf()

    # Finally, we construct an "A" matrix for the form xdot = A x (x being the
    # state vector, although in this case, the sizes are a little off). The
    # following line extracts only the minimum entries required for eigenvalue
    # analysis, which correspond to rows and columns for lean, steer, lean
    # rate, and steer rate.
    A = A_s.extract([1, 2, 3, 5], [1, 2, 3, 5])

    # Precomputed for comparison
    Res = Matrix([[               0,                                           0,                  1.0,                    0],
                  [               0,                                           0,                    0,                  1.0],
                  [9.48977444677355, -0.891197738059089*v**2 - 0.571523173729245, -0.105522449805691*v, -0.330515398992311*v],
                  [11.7194768719633,   -1.97171508499972*v**2 + 30.9087533932407,   3.67680523332152*v,  -3.08486552743311*v]])

    # Actual eigenvalue comparison
    eps = 1.e-12
    for i in range(6):
        error = Res.subs(v, i) - A.subs(v, i)
        assert all(abs(x) < eps for x in error)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\test_timedeltas.py ===
from datetime import timedelta

import numpy as np
import pytest

import pandas as pd
from pandas import Timedelta
import pandas._testing as tm
from pandas.core.arrays import (
    DatetimeArray,
    TimedeltaArray,
)


class TestNonNano:
    @pytest.fixture(params=["s", "ms", "us"])
    def unit(self, request):
        return request.param

    @pytest.fixture
    def tda(self, unit):
        arr = np.arange(5, dtype=np.int64).view(f"m8[{unit}]")
        return TimedeltaArray._simple_new(arr, dtype=arr.dtype)

    def test_non_nano(self, unit):
        arr = np.arange(5, dtype=np.int64).view(f"m8[{unit}]")
        tda = TimedeltaArray._simple_new(arr, dtype=arr.dtype)

        assert tda.dtype == arr.dtype
        assert tda[0].unit == unit

    def test_as_unit_raises(self, tda):
        # GH#50616
        with pytest.raises(ValueError, match="Supported units"):
            tda.as_unit("D")

        tdi = pd.Index(tda)
        with pytest.raises(ValueError, match="Supported units"):
            tdi.as_unit("D")

    @pytest.mark.parametrize("field", TimedeltaArray._field_ops)
    def test_fields(self, tda, field):
        as_nano = tda._ndarray.astype("m8[ns]")
        tda_nano = TimedeltaArray._simple_new(as_nano, dtype=as_nano.dtype)

        result = getattr(tda, field)
        expected = getattr(tda_nano, field)
        tm.assert_numpy_array_equal(result, expected)

    def test_to_pytimedelta(self, tda):
        as_nano = tda._ndarray.astype("m8[ns]")
        tda_nano = TimedeltaArray._simple_new(as_nano, dtype=as_nano.dtype)

        result = tda.to_pytimedelta()
        expected = tda_nano.to_pytimedelta()
        tm.assert_numpy_array_equal(result, expected)

    def test_total_seconds(self, unit, tda):
        as_nano = tda._ndarray.astype("m8[ns]")
        tda_nano = TimedeltaArray._simple_new(as_nano, dtype=as_nano.dtype)

        result = tda.total_seconds()
        expected = tda_nano.total_seconds()
        tm.assert_numpy_array_equal(result, expected)

    def test_timedelta_array_total_seconds(self):
        # GH34290
        expected = Timedelta("2 min").total_seconds()

        result = pd.array([Timedelta("2 min")]).total_seconds()[0]
        assert result == expected

    def test_total_seconds_nanoseconds(self):
        # issue #48521
        start_time = pd.Series(["2145-11-02 06:00:00"]).astype("datetime64[ns]")
        end_time = pd.Series(["2145-11-02 07:06:00"]).astype("datetime64[ns]")
        expected = (end_time - start_time).values / np.timedelta64(1, "s")
        result = (end_time - start_time).dt.total_seconds().values
        assert result == expected

    @pytest.mark.parametrize(
        "nat", [np.datetime64("NaT", "ns"), np.datetime64("NaT", "us")]
    )
    def test_add_nat_datetimelike_scalar(self, nat, tda):
        result = tda + nat
        assert isinstance(result, DatetimeArray)
        assert result._creso == tda._creso
        assert result.isna().all()

        result = nat + tda
        assert isinstance(result, DatetimeArray)
        assert result._creso == tda._creso
        assert result.isna().all()

    def test_add_pdnat(self, tda):
        result = tda + pd.NaT
        assert isinstance(result, TimedeltaArray)
        assert result._creso == tda._creso
        assert result.isna().all()

        result = pd.NaT + tda
        assert isinstance(result, TimedeltaArray)
        assert result._creso == tda._creso
        assert result.isna().all()

    # TODO: 2022-07-11 this is the only test that gets to DTA.tz_convert
    #  or tz_localize with non-nano; implement tests specific to that.
    def test_add_datetimelike_scalar(self, tda, tz_naive_fixture):
        ts = pd.Timestamp("2016-01-01", tz=tz_naive_fixture).as_unit("ns")

        expected = tda.as_unit("ns") + ts
        res = tda + ts
        tm.assert_extension_array_equal(res, expected)
        res = ts + tda
        tm.assert_extension_array_equal(res, expected)

        ts += Timedelta(1)  # case where we can't cast losslessly

        exp_values = tda._ndarray + ts.asm8
        expected = (
            DatetimeArray._simple_new(exp_values, dtype=exp_values.dtype)
            .tz_localize("UTC")
            .tz_convert(ts.tz)
        )

        result = tda + ts
        tm.assert_extension_array_equal(result, expected)

        result = ts + tda
        tm.assert_extension_array_equal(result, expected)

    def test_mul_scalar(self, tda):
        other = 2
        result = tda * other
        expected = TimedeltaArray._simple_new(tda._ndarray * other, dtype=tda.dtype)
        tm.assert_extension_array_equal(result, expected)
        assert result._creso == tda._creso

    def test_mul_listlike(self, tda):
        other = np.arange(len(tda))
        result = tda * other
        expected = TimedeltaArray._simple_new(tda._ndarray * other, dtype=tda.dtype)
        tm.assert_extension_array_equal(result, expected)
        assert result._creso == tda._creso

    def test_mul_listlike_object(self, tda):
        other = np.arange(len(tda))
        result = tda * other.astype(object)
        expected = TimedeltaArray._simple_new(tda._ndarray * other, dtype=tda.dtype)
        tm.assert_extension_array_equal(result, expected)
        assert result._creso == tda._creso

    def test_div_numeric_scalar(self, tda):
        other = 2
        result = tda / other
        expected = TimedeltaArray._simple_new(tda._ndarray / other, dtype=tda.dtype)
        tm.assert_extension_array_equal(result, expected)
        assert result._creso == tda._creso

    def test_div_td_scalar(self, tda):
        other = timedelta(seconds=1)
        result = tda / other
        expected = tda._ndarray / np.timedelta64(1, "s")
        tm.assert_numpy_array_equal(result, expected)

    def test_div_numeric_array(self, tda):
        other = np.arange(len(tda))
        result = tda / other
        expected = TimedeltaArray._simple_new(tda._ndarray / other, dtype=tda.dtype)
        tm.assert_extension_array_equal(result, expected)
        assert result._creso == tda._creso

    def test_div_td_array(self, tda):
        other = tda._ndarray + tda._ndarray[-1]
        result = tda / other
        expected = tda._ndarray / other
        tm.assert_numpy_array_equal(result, expected)

    def test_add_timedeltaarraylike(self, tda):
        tda_nano = tda.astype("m8[ns]")

        expected = tda_nano * 2
        res = tda_nano + tda
        tm.assert_extension_array_equal(res, expected)
        res = tda + tda_nano
        tm.assert_extension_array_equal(res, expected)

        expected = tda_nano * 0
        res = tda - tda_nano
        tm.assert_extension_array_equal(res, expected)

        res = tda_nano - tda
        tm.assert_extension_array_equal(res, expected)


class TestTimedeltaArray:
    @pytest.mark.parametrize("dtype", [int, np.int32, np.int64, "uint32", "uint64"])
    def test_astype_int(self, dtype):
        arr = TimedeltaArray._from_sequence(
            [Timedelta("1h"), Timedelta("2h")], dtype="m8[ns]"
        )

        if np.dtype(dtype) != np.int64:
            with pytest.raises(TypeError, match=r"Do obj.astype\('int64'\)"):
                arr.astype(dtype)
            return

        result = arr.astype(dtype)
        expected = arr._ndarray.view("i8")
        tm.assert_numpy_array_equal(result, expected)

    def test_setitem_clears_freq(self):
        a = pd.timedelta_range("1h", periods=2, freq="h")._data
        a[0] = Timedelta("1h")
        assert a.freq is None

    @pytest.mark.parametrize(
        "obj",
        [
            Timedelta(seconds=1),
            Timedelta(seconds=1).to_timedelta64(),
            Timedelta(seconds=1).to_pytimedelta(),
        ],
    )
    def test_setitem_objects(self, obj):
        # make sure we accept timedelta64 and timedelta in addition to Timedelta
        tdi = pd.timedelta_range("2 Days", periods=4, freq="h")
        arr = tdi._data

        arr[0] = obj
        assert arr[0] == Timedelta(seconds=1)

    @pytest.mark.parametrize(
        "other",
        [
            1,
            np.int64(1),
            1.0,
            np.datetime64("NaT"),
            pd.Timestamp("2021-01-01"),
            "invalid",
            np.arange(10, dtype="i8") * 24 * 3600 * 10**9,
            (np.arange(10) * 24 * 3600 * 10**9).view("datetime64[ns]"),
            pd.Timestamp("2021-01-01").to_period("D"),
        ],
    )
    @pytest.mark.parametrize("index", [True, False])
    def test_searchsorted_invalid_types(self, other, index):
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        arr = pd.TimedeltaIndex(data, freq="D")._data
        if index:
            arr = pd.Index(arr)

        msg = "|".join(
            [
                "searchsorted requires compatible dtype or scalar",
                "value should be a 'Timedelta', 'NaT', or array of those. Got",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            arr.searchsorted(other)


class TestUnaryOps:
    def test_abs(self):
        vals = np.array([-3600 * 10**9, "NaT", 7200 * 10**9], dtype="m8[ns]")
        arr = TimedeltaArray._from_sequence(vals)

        evals = np.array([3600 * 10**9, "NaT", 7200 * 10**9], dtype="m8[ns]")
        expected = TimedeltaArray._from_sequence(evals)

        result = abs(arr)
        tm.assert_timedelta_array_equal(result, expected)

        result2 = np.abs(arr)
        tm.assert_timedelta_array_equal(result2, expected)

    def test_pos(self):
        vals = np.array([-3600 * 10**9, "NaT", 7200 * 10**9], dtype="m8[ns]")
        arr = TimedeltaArray._from_sequence(vals)

        result = +arr
        tm.assert_timedelta_array_equal(result, arr)
        assert not tm.shares_memory(result, arr)

        result2 = np.positive(arr)
        tm.assert_timedelta_array_equal(result2, arr)
        assert not tm.shares_memory(result2, arr)

    def test_neg(self):
        vals = np.array([-3600 * 10**9, "NaT", 7200 * 10**9], dtype="m8[ns]")
        arr = TimedeltaArray._from_sequence(vals)

        evals = np.array([3600 * 10**9, "NaT", -7200 * 10**9], dtype="m8[ns]")
        expected = TimedeltaArray._from_sequence(evals)

        result = -arr
        tm.assert_timedelta_array_equal(result, expected)

        result2 = np.negative(arr)
        tm.assert_timedelta_array_equal(result2, expected)

    def test_neg_freq(self):
        tdi = pd.timedelta_range("2 Days", periods=4, freq="h")
        arr = tdi._data

        expected = -tdi._data

        result = -arr
        tm.assert_timedelta_array_equal(result, expected)

        result2 = np.negative(arr)
        tm.assert_timedelta_array_equal(result2, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\timestamp\test_comparisons.py ===
from datetime import (
    datetime,
    timedelta,
)
import operator

import numpy as np
import pytest

from pandas import Timestamp
import pandas._testing as tm


class TestTimestampComparison:
    def test_compare_non_nano_dt64(self):
        # don't raise when converting dt64 to Timestamp in __richcmp__
        dt = np.datetime64("1066-10-14")
        ts = Timestamp(dt)

        assert dt == ts

    def test_comparison_dt64_ndarray(self):
        ts = Timestamp("2021-01-01")
        ts2 = Timestamp("2019-04-05")
        arr = np.array([[ts.asm8, ts2.asm8]], dtype="M8[ns]")

        result = ts == arr
        expected = np.array([[True, False]], dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

        result = arr == ts
        tm.assert_numpy_array_equal(result, expected)

        result = ts != arr
        tm.assert_numpy_array_equal(result, ~expected)

        result = arr != ts
        tm.assert_numpy_array_equal(result, ~expected)

        result = ts2 < arr
        tm.assert_numpy_array_equal(result, expected)

        result = arr < ts2
        tm.assert_numpy_array_equal(result, np.array([[False, False]], dtype=bool))

        result = ts2 <= arr
        tm.assert_numpy_array_equal(result, np.array([[True, True]], dtype=bool))

        result = arr <= ts2
        tm.assert_numpy_array_equal(result, ~expected)

        result = ts >= arr
        tm.assert_numpy_array_equal(result, np.array([[True, True]], dtype=bool))

        result = arr >= ts
        tm.assert_numpy_array_equal(result, np.array([[True, False]], dtype=bool))

    @pytest.mark.parametrize("reverse", [True, False])
    def test_comparison_dt64_ndarray_tzaware(self, reverse, comparison_op):
        ts = Timestamp("2021-01-01 00:00:00.00000", tz="UTC")
        arr = np.array([ts.asm8, ts.asm8], dtype="M8[ns]")

        left, right = ts, arr
        if reverse:
            left, right = arr, ts

        if comparison_op is operator.eq:
            expected = np.array([False, False], dtype=bool)
            result = comparison_op(left, right)
            tm.assert_numpy_array_equal(result, expected)
        elif comparison_op is operator.ne:
            expected = np.array([True, True], dtype=bool)
            result = comparison_op(left, right)
            tm.assert_numpy_array_equal(result, expected)
        else:
            msg = "Cannot compare tz-naive and tz-aware timestamps"
            with pytest.raises(TypeError, match=msg):
                comparison_op(left, right)

    def test_comparison_object_array(self):
        # GH#15183
        ts = Timestamp("2011-01-03 00:00:00-0500", tz="US/Eastern")
        other = Timestamp("2011-01-01 00:00:00-0500", tz="US/Eastern")
        naive = Timestamp("2011-01-01 00:00:00")

        arr = np.array([other, ts], dtype=object)
        res = arr == ts
        expected = np.array([False, True], dtype=bool)
        assert (res == expected).all()

        # 2D case
        arr = np.array([[other, ts], [ts, other]], dtype=object)
        res = arr != ts
        expected = np.array([[True, False], [False, True]], dtype=bool)
        assert res.shape == expected.shape
        assert (res == expected).all()

        # tzaware mismatch
        arr = np.array([naive], dtype=object)
        msg = "Cannot compare tz-naive and tz-aware timestamps"
        with pytest.raises(TypeError, match=msg):
            arr < ts

    def test_comparison(self):
        # 5-18-2012 00:00:00.000
        stamp = 1337299200000000000

        val = Timestamp(stamp)

        assert val == val
        assert not val != val
        assert not val < val
        assert val <= val
        assert not val > val
        assert val >= val

        other = datetime(2012, 5, 18)
        assert val == other
        assert not val != other
        assert not val < other
        assert val <= other
        assert not val > other
        assert val >= other

        other = Timestamp(stamp + 100)

        assert val != other
        assert val != other
        assert val < other
        assert val <= other
        assert other > val
        assert other >= val

    def test_compare_invalid(self):
        # GH#8058
        val = Timestamp("20130101 12:01:02")
        assert not val == "foo"
        assert not val == 10.0
        assert not val == 1
        assert not val == []
        assert not val == {"foo": 1}
        assert not val == np.float64(1)
        assert not val == np.int64(1)

        assert val != "foo"
        assert val != 10.0
        assert val != 1
        assert val != []
        assert val != {"foo": 1}
        assert val != np.float64(1)
        assert val != np.int64(1)

    @pytest.mark.parametrize("tz", [None, "US/Pacific"])
    def test_compare_date(self, tz):
        # GH#36131 comparing Timestamp with date object is deprecated
        ts = Timestamp("2021-01-01 00:00:00.00000", tz=tz)
        dt = ts.to_pydatetime().date()
        # in 2.0 we disallow comparing pydate objects with Timestamps,
        #  following the stdlib datetime behavior.

        msg = "Cannot compare Timestamp with datetime.date"
        for left, right in [(ts, dt), (dt, ts)]:
            assert not left == right
            assert left != right

            with pytest.raises(TypeError, match=msg):
                left < right
            with pytest.raises(TypeError, match=msg):
                left <= right
            with pytest.raises(TypeError, match=msg):
                left > right
            with pytest.raises(TypeError, match=msg):
                left >= right

    def test_cant_compare_tz_naive_w_aware(self, utc_fixture):
        # see GH#1404
        a = Timestamp("3/12/2012")
        b = Timestamp("3/12/2012", tz=utc_fixture)

        msg = "Cannot compare tz-naive and tz-aware timestamps"
        assert not a == b
        assert a != b
        with pytest.raises(TypeError, match=msg):
            a < b
        with pytest.raises(TypeError, match=msg):
            a <= b
        with pytest.raises(TypeError, match=msg):
            a > b
        with pytest.raises(TypeError, match=msg):
            a >= b

        assert not b == a
        assert b != a
        with pytest.raises(TypeError, match=msg):
            b < a
        with pytest.raises(TypeError, match=msg):
            b <= a
        with pytest.raises(TypeError, match=msg):
            b > a
        with pytest.raises(TypeError, match=msg):
            b >= a

        assert not a == b.to_pydatetime()
        assert not a.to_pydatetime() == b

    def test_timestamp_compare_scalars(self):
        # case where ndim == 0
        lhs = np.datetime64(datetime(2013, 12, 6))
        rhs = Timestamp("now")
        nat = Timestamp("nat")

        ops = {"gt": "lt", "lt": "gt", "ge": "le", "le": "ge", "eq": "eq", "ne": "ne"}

        for left, right in ops.items():
            left_f = getattr(operator, left)
            right_f = getattr(operator, right)
            expected = left_f(lhs, rhs)

            result = right_f(rhs, lhs)
            assert result == expected

            expected = left_f(rhs, nat)
            result = right_f(nat, rhs)
            assert result == expected

    def test_timestamp_compare_with_early_datetime(self):
        # e.g. datetime.min
        stamp = Timestamp("2012-01-01")

        assert not stamp == datetime.min
        assert not stamp == datetime(1600, 1, 1)
        assert not stamp == datetime(2700, 1, 1)
        assert stamp != datetime.min
        assert stamp != datetime(1600, 1, 1)
        assert stamp != datetime(2700, 1, 1)
        assert stamp > datetime(1600, 1, 1)
        assert stamp >= datetime(1600, 1, 1)
        assert stamp < datetime(2700, 1, 1)
        assert stamp <= datetime(2700, 1, 1)

        other = Timestamp.min.to_pydatetime(warn=False)
        assert other - timedelta(microseconds=1) < Timestamp.min

    def test_timestamp_compare_oob_dt64(self):
        us = np.timedelta64(1, "us")
        other = np.datetime64(Timestamp.min).astype("M8[us]")

        # This may change if the implementation bound is dropped to match
        #  DatetimeArray/DatetimeIndex GH#24124
        assert Timestamp.min > other
        # Note: numpy gets the reversed comparison wrong

        other = np.datetime64(Timestamp.max).astype("M8[us]")
        assert Timestamp.max > other  # not actually OOB
        assert other < Timestamp.max

        assert Timestamp.max < other + us
        # Note: numpy gets the reversed comparison wrong

        # GH-42794
        other = datetime(9999, 9, 9)
        assert Timestamp.min < other
        assert other > Timestamp.min
        assert Timestamp.max < other
        assert other > Timestamp.max

        other = datetime(1, 1, 1)
        assert Timestamp.max > other
        assert other < Timestamp.max
        assert Timestamp.min > other
        assert other < Timestamp.min

    def test_compare_zerodim_array(self, fixed_now_ts):
        # GH#26916
        ts = fixed_now_ts
        dt64 = np.datetime64("2016-01-01", "ns")
        arr = np.array(dt64)
        assert arr.ndim == 0

        result = arr < ts
        assert result is np.bool_(True)
        result = arr > ts
        assert result is np.bool_(False)


def test_rich_comparison_with_unsupported_type():
    # Comparisons with unsupported objects should return NotImplemented
    # (it previously raised TypeError, see #24011)

    class Inf:
        def __lt__(self, o):
            return False

        def __le__(self, o):
            return isinstance(o, Inf)

        def __gt__(self, o):
            return not isinstance(o, Inf)

        def __ge__(self, o):
            return True

        def __eq__(self, other) -> bool:
            return isinstance(other, Inf)

    inf = Inf()
    timestamp = Timestamp("2018-11-30")

    for left, right in [(inf, timestamp), (timestamp, inf)]:
        assert left > right or left < right
        assert left >= right or left <= right
        assert not left == right  # pylint: disable=unneeded-not
        assert left != right

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\external\tests\test_autowrap.py ===
import sympy
import tempfile
import os
from pathlib import Path
from sympy.core.mod import Mod
from sympy.core.relational import Eq
from sympy.core.symbol import symbols
from sympy.external import import_module
from sympy.tensor import IndexedBase, Idx
from sympy.utilities.autowrap import autowrap, ufuncify, CodeWrapError
from sympy.testing.pytest import skip

numpy = import_module('numpy', min_module_version='1.6.1')
Cython = import_module('Cython', min_module_version='0.15.1')
f2py = import_module('numpy.f2py', import_kwargs={'fromlist': ['f2py']})

f2pyworks = False
if f2py:
    try:
        autowrap(symbols('x'), 'f95', 'f2py')
    except (CodeWrapError, ImportError, OSError):
        f2pyworks = False
    else:
        f2pyworks = True

a, b, c = symbols('a b c')
n, m, d = symbols('n m d', integer=True)
A, B, C = symbols('A B C', cls=IndexedBase)
i = Idx('i', m)
j = Idx('j', n)
k = Idx('k', d)


def has_module(module):
    """
    Return True if module exists, otherwise run skip().

    module should be a string.
    """
    # To give a string of the module name to skip(), this function takes a
    # string.  So we don't waste time running import_module() more than once,
    # just map the three modules tested here in this dict.
    modnames = {'numpy': numpy, 'Cython': Cython, 'f2py': f2py}

    if modnames[module]:
        if module == 'f2py' and not f2pyworks:
            skip("Couldn't run f2py.")
        return True
    skip("Couldn't import %s." % module)

#
# test runners used by several language-backend combinations
#

def runtest_autowrap_twice(language, backend):
    f = autowrap((((a + b)/c)**5).expand(), language, backend)
    g = autowrap((((a + b)/c)**4).expand(), language, backend)

    # check that autowrap updates the module name.  Else, g gives the same as f
    assert f(1, -2, 1) == -1.0
    assert g(1, -2, 1) == 1.0


def runtest_autowrap_trace(language, backend):
    has_module('numpy')
    trace = autowrap(A[i, i], language, backend)
    assert trace(numpy.eye(100)) == 100


def runtest_autowrap_matrix_vector(language, backend):
    has_module('numpy')
    x, y = symbols('x y', cls=IndexedBase)
    expr = Eq(y[i], A[i, j]*x[j])
    mv = autowrap(expr, language, backend)

    # compare with numpy's dot product
    M = numpy.random.rand(10, 20)
    x = numpy.random.rand(20)
    y = numpy.dot(M, x)
    assert numpy.sum(numpy.abs(y - mv(M, x))) < 1e-13


def runtest_autowrap_matrix_matrix(language, backend):
    has_module('numpy')
    expr = Eq(C[i, j], A[i, k]*B[k, j])
    matmat = autowrap(expr, language, backend)

    # compare with numpy's dot product
    M1 = numpy.random.rand(10, 20)
    M2 = numpy.random.rand(20, 15)
    M3 = numpy.dot(M1, M2)
    assert numpy.sum(numpy.abs(M3 - matmat(M1, M2))) < 1e-13


def runtest_ufuncify(language, backend):
    has_module('numpy')
    a, b, c = symbols('a b c')
    fabc = ufuncify([a, b, c], a*b + c, backend=backend)
    facb = ufuncify([a, c, b], a*b + c, backend=backend)
    grid = numpy.linspace(-2, 2, 50)
    b = numpy.linspace(-5, 4, 50)
    c = numpy.linspace(-1, 1, 50)
    expected = grid*b + c
    numpy.testing.assert_allclose(fabc(grid, b, c), expected)
    numpy.testing.assert_allclose(facb(grid, c, b), expected)


def runtest_issue_10274(language, backend):
    expr = (a - b + c)**(13)
    tmp = tempfile.mkdtemp()
    f = autowrap(expr, language, backend, tempdir=tmp,
                 helpers=('helper', a - b + c, (a, b, c)))
    assert f(1, 1, 1) == 1

    for file in os.listdir(tmp):
        if not (file.startswith("wrapped_code_") and file.endswith(".c")):
            continue

        with open(tmp + '/' + file) as fil:
            lines = fil.readlines()
            assert lines[0] == "/******************************************************************************\n"
            assert "Code generated with SymPy " + sympy.__version__ in lines[1]
            assert lines[2:] == [
                " *                                                                            *\n",
                " *              See http://www.sympy.org/ for more information.               *\n",
                " *                                                                            *\n",
                " *                      This file is part of 'autowrap'                       *\n",
                " ******************************************************************************/\n",
                "#include " + '"' + file[:-1]+ 'h"' + "\n",
                "#include <math.h>\n",
                "\n",
                "double helper(double a, double b, double c) {\n",
                "\n",
                "   double helper_result;\n",
                "   helper_result = a - b + c;\n",
                "   return helper_result;\n",
                "\n",
                "}\n",
                "\n",
                "double autofunc(double a, double b, double c) {\n",
                "\n",
                "   double autofunc_result;\n",
                "   autofunc_result = pow(helper(a, b, c), 13);\n",
                "   return autofunc_result;\n",
                "\n",
                "}\n",
                ]


def runtest_issue_15337(language, backend):
    has_module('numpy')
    # NOTE : autowrap was originally designed to only accept an iterable for
    # the kwarg "helpers", but in issue 10274 the user mistakenly thought that
    # if there was only a single helper it did not need to be passed via an
    # iterable that wrapped the helper tuple. There were no tests for this
    # behavior so when the code was changed to accept a single tuple it broke
    # the original behavior. These tests below ensure that both now work.
    a, b, c, d, e = symbols('a, b, c, d, e')
    expr = (a - b + c - d + e)**13
    exp_res = (1. - 2. + 3. - 4. + 5.)**13

    f = autowrap(expr, language, backend, args=(a, b, c, d, e),
                 helpers=('f1', a - b + c, (a, b, c)))
    numpy.testing.assert_allclose(f(1, 2, 3, 4, 5), exp_res)

    f = autowrap(expr, language, backend, args=(a, b, c, d, e),
                 helpers=(('f1', a - b, (a, b)), ('f2', c - d, (c, d))))
    numpy.testing.assert_allclose(f(1, 2, 3, 4, 5), exp_res)


def test_issue_15230():
    has_module('f2py')

    x, y = symbols('x, y')
    expr = Mod(x, 3.0) - Mod(y, -2.0)
    f = autowrap(expr, args=[x, y], language='F95')
    exp_res = float(expr.xreplace({x: 3.5, y: 2.7}).evalf())
    assert abs(f(3.5, 2.7) - exp_res) < 1e-14

    x, y = symbols('x, y', integer=True)
    expr = Mod(x, 3) - Mod(y, -2)
    f = autowrap(expr, args=[x, y], language='F95')
    assert f(3, 2) == expr.xreplace({x: 3, y: 2})

#
# tests of language-backend combinations
#

# f2py


def test_wrap_twice_f95_f2py():
    has_module('f2py')
    runtest_autowrap_twice('f95', 'f2py')


def test_autowrap_trace_f95_f2py():
    has_module('f2py')
    runtest_autowrap_trace('f95', 'f2py')


def test_autowrap_matrix_vector_f95_f2py():
    has_module('f2py')
    runtest_autowrap_matrix_vector('f95', 'f2py')


def test_autowrap_matrix_matrix_f95_f2py():
    has_module('f2py')
    runtest_autowrap_matrix_matrix('f95', 'f2py')


def test_ufuncify_f95_f2py():
    has_module('f2py')
    runtest_ufuncify('f95', 'f2py')


def test_issue_15337_f95_f2py():
    has_module('f2py')
    runtest_issue_15337('f95', 'f2py')

# Cython


def test_wrap_twice_c_cython():
    has_module('Cython')
    runtest_autowrap_twice('C', 'cython')


def test_autowrap_trace_C_Cython():
    has_module('Cython')
    runtest_autowrap_trace('C99', 'cython')


def test_autowrap_matrix_vector_C_cython():
    has_module('Cython')
    runtest_autowrap_matrix_vector('C99', 'cython')


def test_autowrap_matrix_matrix_C_cython():
    has_module('Cython')
    runtest_autowrap_matrix_matrix('C99', 'cython')


def test_ufuncify_C_Cython():
    has_module('Cython')
    runtest_ufuncify('C99', 'cython')


def test_issue_10274_C_cython():
    has_module('Cython')
    runtest_issue_10274('C89', 'cython')


def test_issue_15337_C_cython():
    has_module('Cython')
    runtest_issue_15337('C89', 'cython')


def test_autowrap_custom_printer():
    has_module('Cython')

    from sympy.core.numbers import pi
    from sympy.utilities.codegen import C99CodeGen
    from sympy.printing.c import C99CodePrinter

    class PiPrinter(C99CodePrinter):
        def _print_Pi(self, expr):
            return "S_PI"

    printer = PiPrinter()
    gen = C99CodeGen(printer=printer)
    gen.preprocessor_statements.append('#include "shortpi.h"')

    expr = pi * a

    expected = (
        '#include "%s"\n'
        '#include <math.h>\n'
        '#include "shortpi.h"\n'
        '\n'
        'double autofunc(double a) {\n'
        '\n'
        '   double autofunc_result;\n'
        '   autofunc_result = S_PI*a;\n'
        '   return autofunc_result;\n'
        '\n'
        '}\n'
    )

    tmpdir = tempfile.mkdtemp()
    # write a trivial header file to use in the generated code
    Path(os.path.join(tmpdir, 'shortpi.h')).write_text('#define S_PI 3.14')

    func = autowrap(expr, backend='cython', tempdir=tmpdir, code_gen=gen)

    assert func(4.2) == 3.14 * 4.2

    # check that the generated code is correct
    for filename in os.listdir(tmpdir):
        if filename.startswith('wrapped_code') and filename.endswith('.c'):
            with open(os.path.join(tmpdir, filename)) as f:
                lines = f.readlines()
                expected = expected % filename.replace('.c', '.h')
                assert ''.join(lines[7:]) == expected


# Numpy

def test_ufuncify_numpy():
    # This test doesn't use Cython, but if Cython works, then there is a valid
    # C compiler, which is needed.
    has_module('Cython')
    runtest_ufuncify('C99', 'numpy')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\greenlet\tests\test_contextvars.py ===
from __future__ import print_function

import gc
import sys
import unittest

from functools import partial
from unittest import skipUnless
from unittest import skipIf

from greenlet import greenlet
from greenlet import getcurrent
from . import TestCase
from . import PY314

try:
    from contextvars import Context
    from contextvars import ContextVar
    from contextvars import copy_context
    # From the documentation:
    #
    # Important: Context Variables should be created at the top module
    # level and never in closures. Context objects hold strong
    # references to context variables which prevents context variables
    # from being properly garbage collected.
    ID_VAR = ContextVar("id", default=None)
    VAR_VAR = ContextVar("var", default=None)
    ContextVar = None
except ImportError:
    Context = ContextVar = copy_context = None

# We don't support testing if greenlet's built-in context var support is disabled.
@skipUnless(Context is not None, "ContextVar not supported")
class ContextVarsTests(TestCase):
    def _new_ctx_run(self, *args, **kwargs):
        return copy_context().run(*args, **kwargs)

    def _increment(self, greenlet_id, callback, counts, expect):
        ctx_var = ID_VAR
        if expect is None:
            self.assertIsNone(ctx_var.get())
        else:
            self.assertEqual(ctx_var.get(), expect)
        ctx_var.set(greenlet_id)
        for _ in range(2):
            counts[ctx_var.get()] += 1
            callback()

    def _test_context(self, propagate_by):
        # pylint:disable=too-many-branches
        ID_VAR.set(0)

        callback = getcurrent().switch
        counts = dict((i, 0) for i in range(5))

        lets = [
            greenlet(partial(
                partial(
                    copy_context().run,
                    self._increment
                ) if propagate_by == "run" else self._increment,
                greenlet_id=i,
                callback=callback,
                counts=counts,
                expect=(
                    i - 1 if propagate_by == "share" else
                    0 if propagate_by in ("set", "run") else None
                )
            ))
            for i in range(1, 5)
        ]

        for let in lets:
            if propagate_by == "set":
                let.gr_context = copy_context()
            elif propagate_by == "share":
                let.gr_context = getcurrent().gr_context

        for i in range(2):
            counts[ID_VAR.get()] += 1
            for let in lets:
                let.switch()

        if propagate_by == "run":
            # Must leave each context.run() in reverse order of entry
            for let in reversed(lets):
                let.switch()
        else:
            # No context.run(), so fine to exit in any order.
            for let in lets:
                let.switch()

        for let in lets:
            self.assertTrue(let.dead)
            # When using run(), we leave the run() as the greenlet dies,
            # and there's no context "underneath". When not using run(),
            # gr_context still reflects the context the greenlet was
            # running in.
            if propagate_by == 'run':
                self.assertIsNone(let.gr_context)
            else:
                self.assertIsNotNone(let.gr_context)


        if propagate_by == "share":
            self.assertEqual(counts, {0: 1, 1: 1, 2: 1, 3: 1, 4: 6})
        else:
            self.assertEqual(set(counts.values()), set([2]))

    def test_context_propagated_by_context_run(self):
        self._new_ctx_run(self._test_context, "run")

    def test_context_propagated_by_setting_attribute(self):
        self._new_ctx_run(self._test_context, "set")

    def test_context_not_propagated(self):
        self._new_ctx_run(self._test_context, None)

    def test_context_shared(self):
        self._new_ctx_run(self._test_context, "share")

    def test_break_ctxvars(self):
        let1 = greenlet(copy_context().run)
        let2 = greenlet(copy_context().run)
        let1.switch(getcurrent().switch)
        let2.switch(getcurrent().switch)
        # Since let2 entered the current context and let1 exits its own, the
        # interpreter emits:
        # RuntimeError: cannot exit context: thread state references a different context object
        let1.switch()

    def test_not_broken_if_using_attribute_instead_of_context_run(self):
        let1 = greenlet(getcurrent().switch)
        let2 = greenlet(getcurrent().switch)
        let1.gr_context = copy_context()
        let2.gr_context = copy_context()
        let1.switch()
        let2.switch()
        let1.switch()
        let2.switch()

    def test_context_assignment_while_running(self):
        # pylint:disable=too-many-statements
        ID_VAR.set(None)

        def target():
            self.assertIsNone(ID_VAR.get())
            self.assertIsNone(gr.gr_context)

            # Context is created on first use
            ID_VAR.set(1)
            self.assertIsInstance(gr.gr_context, Context)
            self.assertEqual(ID_VAR.get(), 1)
            self.assertEqual(gr.gr_context[ID_VAR], 1)

            # Clearing the context makes it get re-created as another
            # empty context when next used
            old_context = gr.gr_context
            gr.gr_context = None  # assign None while running
            self.assertIsNone(ID_VAR.get())
            self.assertIsNone(gr.gr_context)
            ID_VAR.set(2)
            self.assertIsInstance(gr.gr_context, Context)
            self.assertEqual(ID_VAR.get(), 2)
            self.assertEqual(gr.gr_context[ID_VAR], 2)

            new_context = gr.gr_context
            getcurrent().parent.switch((old_context, new_context))
            # parent switches us back to old_context

            self.assertEqual(ID_VAR.get(), 1)
            gr.gr_context = new_context  # assign non-None while running
            self.assertEqual(ID_VAR.get(), 2)

            getcurrent().parent.switch()
            # parent switches us back to no context
            self.assertIsNone(ID_VAR.get())
            self.assertIsNone(gr.gr_context)
            gr.gr_context = old_context
            self.assertEqual(ID_VAR.get(), 1)

            getcurrent().parent.switch()
            # parent switches us back to no context
            self.assertIsNone(ID_VAR.get())
            self.assertIsNone(gr.gr_context)

        gr = greenlet(target)

        with self.assertRaisesRegex(AttributeError, "can't delete context attribute"):
            del gr.gr_context

        self.assertIsNone(gr.gr_context)
        old_context, new_context = gr.switch()
        self.assertIs(new_context, gr.gr_context)
        self.assertEqual(old_context[ID_VAR], 1)
        self.assertEqual(new_context[ID_VAR], 2)
        self.assertEqual(new_context.run(ID_VAR.get), 2)
        gr.gr_context = old_context  # assign non-None while suspended
        gr.switch()
        self.assertIs(gr.gr_context, new_context)
        gr.gr_context = None  # assign None while suspended
        gr.switch()
        self.assertIs(gr.gr_context, old_context)
        gr.gr_context = None
        gr.switch()
        self.assertIsNone(gr.gr_context)

        # Make sure there are no reference leaks
        gr = None
        gc.collect()
        # Python 3.14 elides reference counting operations
        # in some cases. See https://github.com/python/cpython/pull/130708
        self.assertEqual(sys.getrefcount(old_context), 2 if not PY314 else 1)
        self.assertEqual(sys.getrefcount(new_context), 2 if not PY314 else 1)

    def test_context_assignment_different_thread(self):
        import threading
        VAR_VAR.set(None)
        ctx = Context()

        is_running = threading.Event()
        should_suspend = threading.Event()
        did_suspend = threading.Event()
        should_exit = threading.Event()
        holder = []

        def greenlet_in_thread_fn():
            VAR_VAR.set(1)
            is_running.set()
            should_suspend.wait(10)
            VAR_VAR.set(2)
            getcurrent().parent.switch()
            holder.append(VAR_VAR.get())

        def thread_fn():
            gr = greenlet(greenlet_in_thread_fn)
            gr.gr_context = ctx
            holder.append(gr)
            gr.switch()
            did_suspend.set()
            should_exit.wait(10)
            gr.switch()
            del gr
            greenlet() # trigger cleanup

        thread = threading.Thread(target=thread_fn, daemon=True)
        thread.start()
        is_running.wait(10)
        gr = holder[0]

        # Can't access or modify context if the greenlet is running
        # in a different thread
        with self.assertRaisesRegex(ValueError, "running in a different"):
            getattr(gr, 'gr_context')
        with self.assertRaisesRegex(ValueError, "running in a different"):
            gr.gr_context = None

        should_suspend.set()
        did_suspend.wait(10)

        # OK to access and modify context if greenlet is suspended
        self.assertIs(gr.gr_context, ctx)
        self.assertEqual(gr.gr_context[VAR_VAR], 2)
        gr.gr_context = None

        should_exit.set()
        thread.join(10)

        self.assertEqual(holder, [gr, None])

        # Context can still be accessed/modified when greenlet is dead:
        self.assertIsNone(gr.gr_context)
        gr.gr_context = ctx
        self.assertIs(gr.gr_context, ctx)

        # Otherwise we leak greenlets on some platforms.
        # XXX: Should be able to do this automatically
        del holder[:]
        gr = None
        thread = None

    def test_context_assignment_wrong_type(self):
        g = greenlet()
        with self.assertRaisesRegex(TypeError,
                                    "greenlet context must be a contextvars.Context or None"):
            g.gr_context = self


@skipIf(Context is not None, "ContextVar supported")
class NoContextVarsTests(TestCase):
    def test_contextvars_errors(self):
        let1 = greenlet(getcurrent().switch)
        self.assertFalse(hasattr(let1, 'gr_context'))
        with self.assertRaises(AttributeError):
            getattr(let1, 'gr_context')

        with self.assertRaises(AttributeError):
            let1.gr_context = None

        let1.switch()

        with self.assertRaises(AttributeError):
            getattr(let1, 'gr_context')

        with self.assertRaises(AttributeError):
            let1.gr_context = None

        del let1


if __name__ == '__main__':
    unittest.main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_facts.py ===
from sympy.core.facts import (deduce_alpha_implications,
        apply_beta_to_alpha_route, rules_2prereq, FactRules, FactKB)
from sympy.core.logic import And, Not
from sympy.testing.pytest import raises

T = True
F = False
U = None


def test_deduce_alpha_implications():
    def D(i):
        I = deduce_alpha_implications(i)
        P = rules_2prereq({
            (k, True): {(v, True) for v in S} for k, S in I.items()})
        return I, P

    # transitivity
    I, P = D([('a', 'b'), ('b', 'c')])
    assert I == {'a': {'b', 'c'}, 'b': {'c'}, Not('b'):
        {Not('a')}, Not('c'): {Not('a'), Not('b')}}
    assert P == {'a': {'b', 'c'}, 'b': {'a', 'c'}, 'c': {'a', 'b'}}

    # Duplicate entry
    I, P = D([('a', 'b'), ('b', 'c'), ('b', 'c')])
    assert I == {'a': {'b', 'c'}, 'b': {'c'}, Not('b'): {Not('a')}, Not('c'): {Not('a'), Not('b')}}
    assert P == {'a': {'b', 'c'}, 'b': {'a', 'c'}, 'c': {'a', 'b'}}

    # see if it is tolerant to cycles
    assert D([('a', 'a'), ('a', 'a')]) == ({}, {})
    assert D([('a', 'b'), ('b', 'a')]) == (
        {'a': {'b'}, 'b': {'a'}, Not('a'): {Not('b')}, Not('b'): {Not('a')}},
        {'a': {'b'}, 'b': {'a'}})

    # see if it catches inconsistency
    raises(ValueError, lambda: D([('a', Not('a'))]))
    raises(ValueError, lambda: D([('a', 'b'), ('b', Not('a'))]))
    raises(ValueError, lambda: D([('a', 'b'), ('b', 'c'), ('b', 'na'),
           ('na', Not('a'))]))

    # see if it handles implications with negations
    I, P = D([('a', Not('b')), ('c', 'b')])
    assert I == {'a': {Not('b'), Not('c')}, 'b': {Not('a')}, 'c': {'b', Not('a')}, Not('b'): {Not('c')}}
    assert P == {'a': {'b', 'c'}, 'b': {'a', 'c'}, 'c': {'a', 'b'}}
    I, P = D([(Not('a'), 'b'), ('a', 'c')])
    assert I == {'a': {'c'}, Not('a'): {'b'}, Not('b'): {'a',
    'c'}, Not('c'): {Not('a'), 'b'},}
    assert P == {'a': {'b', 'c'}, 'b': {'a', 'c'}, 'c': {'a', 'b'}}


    # Long deductions
    I, P = D([('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e')])
    assert I == {'a': {'b', 'c', 'd', 'e'}, 'b': {'c', 'd', 'e'},
        'c': {'d', 'e'}, 'd': {'e'}, Not('b'): {Not('a')},
        Not('c'): {Not('a'), Not('b')}, Not('d'): {Not('a'), Not('b'),
            Not('c')}, Not('e'): {Not('a'), Not('b'), Not('c'), Not('d')}}
    assert P == {'a': {'b', 'c', 'd', 'e'}, 'b': {'a', 'c', 'd',
        'e'}, 'c': {'a', 'b', 'd', 'e'}, 'd': {'a', 'b', 'c', 'e'},
        'e': {'a', 'b', 'c', 'd'}}

    # something related to real-world
    I, P = D([('rat', 'real'), ('int', 'rat')])

    assert I == {'int': {'rat', 'real'}, 'rat': {'real'},
        Not('real'): {Not('rat'), Not('int')}, Not('rat'): {Not('int')}}
    assert P == {'rat': {'int', 'real'}, 'real': {'int', 'rat'},
        'int': {'rat', 'real'}}


# TODO move me to appropriate place
def test_apply_beta_to_alpha_route():
    APPLY = apply_beta_to_alpha_route

    # indicates empty alpha-chain with attached beta-rule #bidx
    def Q(bidx):
        return (set(), [bidx])

    # x -> a        &(a,b) -> x     --  x -> a
    A = {'x': {'a'}}
    B = [(And('a', 'b'), 'x')]
    assert APPLY(A, B) == {'x': ({'a'}, []), 'a': Q(0), 'b': Q(0)}

    # x -> a        &(a,!x) -> b    --  x -> a
    A = {'x': {'a'}}
    B = [(And('a', Not('x')), 'b')]
    assert APPLY(A, B) == {'x': ({'a'}, []), Not('x'): Q(0), 'a': Q(0)}

    # x -> a b      &(a,b) -> c     --  x -> a b c
    A = {'x': {'a', 'b'}}
    B = [(And('a', 'b'), 'c')]
    assert APPLY(A, B) == \
        {'x': ({'a', 'b', 'c'}, []), 'a': Q(0), 'b': Q(0)}

    # x -> a        &(a,b) -> y     --  x -> a [#0]
    A = {'x': {'a'}}
    B = [(And('a', 'b'), 'y')]
    assert APPLY(A, B) == {'x': ({'a'}, [0]), 'a': Q(0), 'b': Q(0)}

    # x -> a b c    &(a,b) -> c     --  x -> a b c
    A = {'x': {'a', 'b', 'c'}}
    B = [(And('a', 'b'), 'c')]
    assert APPLY(A, B) == \
        {'x': ({'a', 'b', 'c'}, []), 'a': Q(0), 'b': Q(0)}

    # x -> a b      &(a,b,c) -> y   --  x -> a b [#0]
    A = {'x': {'a', 'b'}}
    B = [(And('a', 'b', 'c'), 'y')]
    assert APPLY(A, B) == \
        {'x': ({'a', 'b'}, [0]), 'a': Q(0), 'b': Q(0), 'c': Q(0)}

    # x -> a b      &(a,b) -> c     --  x -> a b c d
    # c -> d                            c -> d
    A = {'x': {'a', 'b'}, 'c': {'d'}}
    B = [(And('a', 'b'), 'c')]
    assert APPLY(A, B) == {'x': ({'a', 'b', 'c', 'd'}, []),
        'c': ({'d'}, []), 'a': Q(0), 'b': Q(0)}

    # x -> a b      &(a,b) -> c     --  x -> a b c d e
    # c -> d        &(c,d) -> e         c -> d e
    A = {'x': {'a', 'b'}, 'c': {'d'}}
    B = [(And('a', 'b'), 'c'), (And('c', 'd'), 'e')]
    assert APPLY(A, B) == {'x': ({'a', 'b', 'c', 'd', 'e'}, []),
        'c': ({'d', 'e'}, []), 'a': Q(0), 'b': Q(0), 'd': Q(1)}

    # x -> a b      &(a,y) -> z     --  x -> a b y z
    #               &(a,b) -> y
    A = {'x': {'a', 'b'}}
    B = [(And('a', 'y'), 'z'), (And('a', 'b'), 'y')]
    assert APPLY(A, B) == {'x': ({'a', 'b', 'y', 'z'}, []),
        'a': (set(), [0, 1]), 'y': Q(0), 'b': Q(1)}

    # x -> a b      &(a,!b) -> c    --  x -> a b
    A = {'x': {'a', 'b'}}
    B = [(And('a', Not('b')), 'c')]
    assert APPLY(A, B) == \
        {'x': ({'a', 'b'}, []), 'a': Q(0), Not('b'): Q(0)}

    # !x -> !a !b   &(!a,b) -> c    --  !x -> !a !b
    A = {Not('x'): {Not('a'), Not('b')}}
    B = [(And(Not('a'), 'b'), 'c')]
    assert APPLY(A, B) == \
        {Not('x'): ({Not('a'), Not('b')}, []), Not('a'): Q(0), 'b': Q(0)}

    # x -> a b      &(b,c) -> !a    --  x -> a b
    A = {'x': {'a', 'b'}}
    B = [(And('b', 'c'), Not('a'))]
    assert APPLY(A, B) == {'x': ({'a', 'b'}, []), 'b': Q(0), 'c': Q(0)}

    # x -> a b      &(a, b) -> c    --  x -> a b c p
    # c -> p a
    A = {'x': {'a', 'b'}, 'c': {'p', 'a'}}
    B = [(And('a', 'b'), 'c')]
    assert APPLY(A, B) == {'x': ({'a', 'b', 'c', 'p'}, []),
        'c': ({'p', 'a'}, []), 'a': Q(0), 'b': Q(0)}


def test_FactRules_parse():
    f = FactRules('a -> b')
    assert f.prereq == {'b': {'a'}, 'a': {'b'}}

    f = FactRules('a -> !b')
    assert f.prereq == {'b': {'a'}, 'a': {'b'}}

    f = FactRules('!a -> b')
    assert f.prereq == {'b': {'a'}, 'a': {'b'}}

    f = FactRules('!a -> !b')
    assert f.prereq == {'b': {'a'}, 'a': {'b'}}

    f = FactRules('!z == nz')
    assert f.prereq == {'z': {'nz'}, 'nz': {'z'}}


def test_FactRules_parse2():
    raises(ValueError, lambda: FactRules('a -> !a'))


def test_FactRules_deduce():
    f = FactRules(['a -> b', 'b -> c', 'b -> d', 'c -> e'])

    def D(facts):
        kb = FactKB(f)
        kb.deduce_all_facts(facts)
        return kb

    assert D({'a': T}) == {'a': T, 'b': T, 'c': T, 'd': T, 'e': T}
    assert D({'b': T}) == {        'b': T, 'c': T, 'd': T, 'e': T}
    assert D({'c': T}) == {                'c': T,         'e': T}
    assert D({'d': T}) == {                        'd': T        }
    assert D({'e': T}) == {                                'e': T}

    assert D({'a': F}) == {'a': F                                }
    assert D({'b': F}) == {'a': F, 'b': F                        }
    assert D({'c': F}) == {'a': F, 'b': F, 'c': F                }
    assert D({'d': F}) == {'a': F, 'b': F,         'd': F        }

    assert D({'a': U}) == {'a': U}  # XXX ok?


def test_FactRules_deduce2():
    # pos/neg/zero, but the rules are not sufficient to derive all relations
    f = FactRules(['pos -> !neg', 'pos -> !z'])

    def D(facts):
        kb = FactKB(f)
        kb.deduce_all_facts(facts)
        return kb

    assert D({'pos': T}) == {'pos': T, 'neg': F, 'z': F}
    assert D({'pos': F}) == {'pos': F                  }
    assert D({'neg': T}) == {'pos': F, 'neg': T        }
    assert D({'neg': F}) == {          'neg': F        }
    assert D({'z': T}) == {'pos': F,           'z': T}
    assert D({'z': F}) == {                    'z': F}

    # pos/neg/zero. rules are sufficient to derive all relations
    f = FactRules(['pos -> !neg', 'neg -> !pos', 'pos -> !z', 'neg -> !z'])

    assert D({'pos': T}) == {'pos': T, 'neg': F, 'z': F}
    assert D({'pos': F}) == {'pos': F                  }
    assert D({'neg': T}) == {'pos': F, 'neg': T, 'z': F}
    assert D({'neg': F}) == {          'neg': F        }
    assert D({'z': T}) == {'pos': F, 'neg': F, 'z': T}
    assert D({'z': F}) == {                    'z': F}


def test_FactRules_deduce_multiple():
    # deduction that involves _several_ starting points
    f = FactRules(['real == pos | npos'])

    def D(facts):
        kb = FactKB(f)
        kb.deduce_all_facts(facts)
        return kb

    assert D({'real': T}) == {'real': T}
    assert D({'real': F}) == {'real': F, 'pos': F, 'npos': F}
    assert D({'pos': T}) == {'real': T, 'pos': T}
    assert D({'npos': T}) == {'real': T, 'npos': T}

    # --- key tests below ---
    assert D({'pos': F, 'npos': F}) == {'real': F, 'pos': F, 'npos': F}
    assert D({'real': T, 'pos': F}) == {'real': T, 'pos': F, 'npos': T}
    assert D({'real': T, 'npos': F}) == {'real': T, 'pos': T, 'npos': F}

    assert D({'pos': T, 'npos': F}) == {'real': T, 'pos': T, 'npos': F}
    assert D({'pos': F, 'npos': T}) == {'real': T, 'pos': F, 'npos': T}


def test_FactRules_deduce_multiple2():
    f = FactRules(['real == neg | zero | pos'])

    def D(facts):
        kb = FactKB(f)
        kb.deduce_all_facts(facts)
        return kb

    assert D({'real': T}) == {'real': T}
    assert D({'real': F}) == {'real': F, 'neg': F, 'zero': F, 'pos': F}
    assert D({'neg': T}) == {'real': T, 'neg': T}
    assert D({'zero': T}) == {'real': T, 'zero': T}
    assert D({'pos': T}) == {'real': T, 'pos': T}

    # --- key tests below ---
    assert D({'neg': F, 'zero': F, 'pos': F}) == {'real': F, 'neg': F,
             'zero': F, 'pos': F}
    assert D({'real': T, 'neg': F}) == {'real': T, 'neg': F}
    assert D({'real': T, 'zero': F}) == {'real': T, 'zero': F}
    assert D({'real': T, 'pos': F}) == {'real': T, 'pos': F}

    assert D({'real': T,           'zero': F, 'pos': F}) == {'real': T,
             'neg': T, 'zero': F, 'pos': F}
    assert D({'real': T, 'neg': F,            'pos': F}) == {'real': T,
             'neg': F, 'zero': T, 'pos': F}
    assert D({'real': T, 'neg': F, 'zero': F          }) == {'real': T,
             'neg': F, 'zero': F, 'pos': T}

    assert D({'neg': T, 'zero': F, 'pos': F}) == {'real': T, 'neg': T,
             'zero': F, 'pos': F}
    assert D({'neg': F, 'zero': T, 'pos': F}) == {'real': T, 'neg': F,
             'zero': T, 'pos': F}
    assert D({'neg': F, 'zero': F, 'pos': T}) == {'real': T, 'neg': F,
             'zero': F, 'pos': T}


def test_FactRules_deduce_base():
    # deduction that starts from base

    f = FactRules(['real  == neg | zero | pos',
                   'neg   -> real & !zero & !pos',
                   'pos   -> real & !zero & !neg'])
    base = FactKB(f)

    base.deduce_all_facts({'real': T, 'neg': F})
    assert base == {'real': T, 'neg': F}

    base.deduce_all_facts({'zero': F})
    assert base == {'real': T, 'neg': F, 'zero': F, 'pos': T}


def test_FactRules_deduce_staticext():
    # verify that static beta-extensions deduction takes place
    f = FactRules(['real  == neg | zero | pos',
                   'neg   -> real & !zero & !pos',
                   'pos   -> real & !zero & !neg',
                   'nneg  == real & !neg',
                   'npos  == real & !pos'])

    assert ('npos', True) in f.full_implications[('neg', True)]
    assert ('nneg', True) in f.full_implications[('pos', True)]
    assert ('nneg', True) in f.full_implications[('zero', True)]
    assert ('npos', True) in f.full_implications[('zero', True)]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\tests\test_sequences.py ===
from sympy.core.containers import Tuple
from sympy.core.function import Function
from sympy.core.numbers import oo, Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols, Symbol
from sympy.functions.combinatorial.numbers import tribonacci, fibonacci
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.series import EmptySequence
from sympy.series.sequences import (SeqMul, SeqAdd, SeqPer, SeqFormula,
    sequence)
from sympy.sets.sets import Interval
from sympy.tensor.indexed import Indexed, Idx
from sympy.series.sequences import SeqExpr, SeqExprOp, RecursiveSeq
from sympy.testing.pytest import raises, slow

x, y, z = symbols('x y z')
n, m = symbols('n m')


def test_EmptySequence():
    assert S.EmptySequence is EmptySequence

    assert S.EmptySequence.interval is S.EmptySet
    assert S.EmptySequence.length is S.Zero

    assert list(S.EmptySequence) == []


def test_SeqExpr():
    #SeqExpr is a baseclass and does not take care of
    #ensuring all arguments are Basics hence the use of
    #Tuple(...) here.
    s = SeqExpr(Tuple(1, n, y), Tuple(x, 0, 10))

    assert isinstance(s, SeqExpr)
    assert s.gen == (1, n, y)
    assert s.interval == Interval(0, 10)
    assert s.start == 0
    assert s.stop == 10
    assert s.length == 11
    assert s.variables == (x,)

    assert SeqExpr(Tuple(1, 2, 3), Tuple(x, 0, oo)).length is oo


def test_SeqPer():
    s = SeqPer((1, n, 3), (x, 0, 5))

    assert isinstance(s, SeqPer)
    assert s.periodical == Tuple(1, n, 3)
    assert s.period == 3
    assert s.coeff(3) == 1
    assert s.free_symbols == {n}

    assert list(s) == [1, n, 3, 1, n, 3]
    assert s[:] == [1, n, 3, 1, n, 3]
    assert SeqPer((1, n, 3), (x, -oo, 0))[0:6] == [1, n, 3, 1, n, 3]

    raises(ValueError, lambda: SeqPer((1, 2, 3), (0, 1, 2)))
    raises(ValueError, lambda: SeqPer((1, 2, 3), (x, -oo, oo)))
    raises(ValueError, lambda: SeqPer(n**2, (0, oo)))

    assert SeqPer((n, n**2, n**3), (m, 0, oo))[:6] == \
        [n, n**2, n**3, n, n**2, n**3]
    assert SeqPer((n, n**2, n**3), (n, 0, oo))[:6] == [0, 1, 8, 3, 16, 125]
    assert SeqPer((n, m), (n, 0, oo))[:6] == [0, m, 2, m, 4, m]


def test_SeqFormula():
    s = SeqFormula(n**2, (n, 0, 5))

    assert isinstance(s, SeqFormula)
    assert s.formula == n**2
    assert s.coeff(3) == 9

    assert list(s) == [i**2 for i in range(6)]
    assert s[:] == [i**2 for i in range(6)]
    assert SeqFormula(n**2, (n, -oo, 0))[0:6] == [i**2 for i in range(6)]

    assert SeqFormula(n**2, (0, oo)) == SeqFormula(n**2, (n, 0, oo))

    assert SeqFormula(n**2, (0, m)).subs(m, x) == SeqFormula(n**2, (0, x))
    assert SeqFormula(m*n**2, (n, 0, oo)).subs(m, x) == \
        SeqFormula(x*n**2, (n, 0, oo))

    raises(ValueError, lambda: SeqFormula(n**2, (0, 1, 2)))
    raises(ValueError, lambda: SeqFormula(n**2, (n, -oo, oo)))
    raises(ValueError, lambda: SeqFormula(m*n**2, (0, oo)))

    seq = SeqFormula(x*(y**2 + z), (z, 1, 100))
    assert seq.expand() == SeqFormula(x*y**2 + x*z, (z, 1, 100))
    seq = SeqFormula(sin(x*(y**2 + z)),(z, 1, 100))
    assert seq.expand(trig=True) == SeqFormula(sin(x*y**2)*cos(x*z) + sin(x*z)*cos(x*y**2), (z, 1, 100))
    assert seq.expand() == SeqFormula(sin(x*y**2 + x*z), (z, 1, 100))
    assert seq.expand(trig=False) == SeqFormula(sin(x*y**2 + x*z), (z, 1, 100))
    seq = SeqFormula(exp(x*(y**2 + z)), (z, 1, 100))
    assert seq.expand() == SeqFormula(exp(x*y**2)*exp(x*z), (z, 1, 100))
    assert seq.expand(power_exp=False) == SeqFormula(exp(x*y**2 + x*z), (z, 1, 100))
    assert seq.expand(mul=False, power_exp=False) == SeqFormula(exp(x*(y**2 + z)), (z, 1, 100))

def test_sequence():
    form = SeqFormula(n**2, (n, 0, 5))
    per = SeqPer((1, 2, 3), (n, 0, 5))
    inter = SeqFormula(n**2)

    assert sequence(n**2, (n, 0, 5)) == form
    assert sequence((1, 2, 3), (n, 0, 5)) == per
    assert sequence(n**2) == inter


def test_SeqExprOp():
    form = SeqFormula(n**2, (n, 0, 10))
    per = SeqPer((1, 2, 3), (m, 5, 10))

    s = SeqExprOp(form, per)
    assert s.gen == (n**2, (1, 2, 3))
    assert s.interval == Interval(5, 10)
    assert s.start == 5
    assert s.stop == 10
    assert s.length == 6
    assert s.variables == (n, m)


def test_SeqAdd():
    per = SeqPer((1, 2, 3), (n, 0, oo))
    form = SeqFormula(n**2)

    per_bou = SeqPer((1, 2), (n, 1, 5))
    form_bou = SeqFormula(n**2, (6, 10))
    form_bou2 = SeqFormula(n**2, (1, 5))

    assert SeqAdd() == S.EmptySequence
    assert SeqAdd(S.EmptySequence) == S.EmptySequence
    assert SeqAdd(per) == per
    assert SeqAdd(per, S.EmptySequence) == per
    assert SeqAdd(per_bou, form_bou) == S.EmptySequence

    s = SeqAdd(per_bou, form_bou2, evaluate=False)
    assert s.args == (form_bou2, per_bou)
    assert s[:] == [2, 6, 10, 18, 26]
    assert list(s) == [2, 6, 10, 18, 26]

    assert isinstance(SeqAdd(per, per_bou, evaluate=False), SeqAdd)

    s1 = SeqAdd(per, per_bou)
    assert isinstance(s1, SeqPer)
    assert s1 == SeqPer((2, 4, 4, 3, 3, 5), (n, 1, 5))
    s2 = SeqAdd(form, form_bou)
    assert isinstance(s2, SeqFormula)
    assert s2 == SeqFormula(2*n**2, (6, 10))

    assert SeqAdd(form, form_bou, per) == \
        SeqAdd(per, SeqFormula(2*n**2, (6, 10)))
    assert SeqAdd(form, SeqAdd(form_bou, per)) == \
        SeqAdd(per, SeqFormula(2*n**2, (6, 10)))
    assert SeqAdd(per, SeqAdd(form, form_bou), evaluate=False) == \
        SeqAdd(per, SeqFormula(2*n**2, (6, 10)))

    assert SeqAdd(SeqPer((1, 2), (n, 0, oo)), SeqPer((1, 2), (m, 0, oo))) == \
        SeqPer((2, 4), (n, 0, oo))


def test_SeqMul():
    per = SeqPer((1, 2, 3), (n, 0, oo))
    form = SeqFormula(n**2)

    per_bou = SeqPer((1, 2), (n, 1, 5))
    form_bou = SeqFormula(n**2, (n, 6, 10))
    form_bou2 = SeqFormula(n**2, (1, 5))

    assert SeqMul() == S.EmptySequence
    assert SeqMul(S.EmptySequence) == S.EmptySequence
    assert SeqMul(per) == per
    assert SeqMul(per, S.EmptySequence) == S.EmptySequence
    assert SeqMul(per_bou, form_bou) == S.EmptySequence

    s = SeqMul(per_bou, form_bou2, evaluate=False)
    assert s.args == (form_bou2, per_bou)
    assert s[:] == [1, 8, 9, 32, 25]
    assert list(s) == [1, 8, 9, 32, 25]

    assert isinstance(SeqMul(per, per_bou, evaluate=False), SeqMul)

    s1 = SeqMul(per, per_bou)
    assert isinstance(s1, SeqPer)
    assert s1 == SeqPer((1, 4, 3, 2, 2, 6), (n, 1, 5))
    s2 = SeqMul(form, form_bou)
    assert isinstance(s2, SeqFormula)
    assert s2 == SeqFormula(n**4, (6, 10))

    assert SeqMul(form, form_bou, per) == \
        SeqMul(per, SeqFormula(n**4, (6, 10)))
    assert SeqMul(form, SeqMul(form_bou, per)) == \
        SeqMul(per, SeqFormula(n**4, (6, 10)))
    assert SeqMul(per, SeqMul(form, form_bou2,
                              evaluate=False), evaluate=False) == \
        SeqMul(form, per, form_bou2, evaluate=False)

    assert SeqMul(SeqPer((1, 2), (n, 0, oo)), SeqPer((1, 2), (n, 0, oo))) == \
        SeqPer((1, 4), (n, 0, oo))


def test_add():
    per = SeqPer((1, 2), (n, 0, oo))
    form = SeqFormula(n**2)

    assert per + (SeqPer((2, 3))) == SeqPer((3, 5), (n, 0, oo))
    assert form + SeqFormula(n**3) == SeqFormula(n**2 + n**3)

    assert per + form == SeqAdd(per, form)

    raises(TypeError, lambda: per + n)
    raises(TypeError, lambda: n + per)


def test_sub():
    per = SeqPer((1, 2), (n, 0, oo))
    form = SeqFormula(n**2)

    assert per - (SeqPer((2, 3))) == SeqPer((-1, -1), (n, 0, oo))
    assert form - (SeqFormula(n**3)) == SeqFormula(n**2 - n**3)

    assert per - form == SeqAdd(per, -form)

    raises(TypeError, lambda: per - n)
    raises(TypeError, lambda: n - per)


def test_mul__coeff_mul():
    assert SeqPer((1, 2), (n, 0, oo)).coeff_mul(2) == SeqPer((2, 4), (n, 0, oo))
    assert SeqFormula(n**2).coeff_mul(2) == SeqFormula(2*n**2)
    assert S.EmptySequence.coeff_mul(100) == S.EmptySequence

    assert SeqPer((1, 2), (n, 0, oo)) * (SeqPer((2, 3))) == \
        SeqPer((2, 6), (n, 0, oo))
    assert SeqFormula(n**2) * SeqFormula(n**3) == SeqFormula(n**5)

    assert S.EmptySequence * SeqFormula(n**2) == S.EmptySequence
    assert SeqFormula(n**2) * S.EmptySequence == S.EmptySequence

    raises(TypeError, lambda: sequence(n**2) * n)
    raises(TypeError, lambda: n * sequence(n**2))


def test_neg():
    assert -SeqPer((1, -2), (n, 0, oo)) == SeqPer((-1, 2), (n, 0, oo))
    assert -SeqFormula(n**2) == SeqFormula(-n**2)


def test_operations():
    per = SeqPer((1, 2), (n, 0, oo))
    per2 = SeqPer((2, 4), (n, 0, oo))
    form = SeqFormula(n**2)
    form2 = SeqFormula(n**3)

    assert per + form + form2 == SeqAdd(per, form, form2)
    assert per + form - form2 == SeqAdd(per, form, -form2)
    assert per + form - S.EmptySequence == SeqAdd(per, form)
    assert per + per2 + form == SeqAdd(SeqPer((3, 6), (n, 0, oo)), form)
    assert S.EmptySequence - per == -per
    assert form + form == SeqFormula(2*n**2)

    assert per * form * form2 == SeqMul(per, form, form2)
    assert form * form == SeqFormula(n**4)
    assert form * -form == SeqFormula(-n**4)

    assert form * (per + form2) == SeqMul(form, SeqAdd(per, form2))
    assert form * (per + per) == SeqMul(form, per2)

    assert form.coeff_mul(m) == SeqFormula(m*n**2, (n, 0, oo))
    assert per.coeff_mul(m) == SeqPer((m, 2*m), (n, 0, oo))


def test_Idx_limits():
    i = symbols('i', cls=Idx)
    r = Indexed('r', i)

    assert SeqFormula(r, (i, 0, 5))[:] == [r.subs(i, j) for j in range(6)]
    assert SeqPer((1, 2), (i, 0, 5))[:] == [1, 2, 1, 2, 1, 2]


@slow
def test_find_linear_recurrence():
    assert sequence((0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55), \
    (n, 0, 10)).find_linear_recurrence(11) == [1, 1]
    assert sequence((1, 2, 4, 7, 28, 128, 582, 2745, 13021, 61699, 292521, \
    1387138), (n, 0, 11)).find_linear_recurrence(12) == [5, -2, 6, -11]
    assert sequence(x*n**3+y*n, (n, 0, oo)).find_linear_recurrence(10) \
    == [4, -6, 4, -1]
    assert sequence(x**n, (n,0,20)).find_linear_recurrence(21) == [x]
    assert sequence((1,2,3)).find_linear_recurrence(10, 5) == [0, 0, 1]
    assert sequence(((1 + sqrt(5))/2)**n + \
    (-(1 + sqrt(5))/2)**(-n)).find_linear_recurrence(10) == [1, 1]
    assert sequence(x*((1 + sqrt(5))/2)**n + y*(-(1 + sqrt(5))/2)**(-n), \
    (n,0,oo)).find_linear_recurrence(10) == [1, 1]
    assert sequence((1,2,3,4,6),(n, 0, 4)).find_linear_recurrence(5) == []
    assert sequence((2,3,4,5,6,79),(n, 0, 5)).find_linear_recurrence(6,gfvar=x) \
    == ([], None)
    assert sequence((2,3,4,5,8,30),(n, 0, 5)).find_linear_recurrence(6,gfvar=x) \
    == ([Rational(19, 2), -20, Rational(27, 2)], (-31*x**2 + 32*x - 4)/(27*x**3 - 40*x**2 + 19*x -2))
    assert sequence(fibonacci(n)).find_linear_recurrence(30,gfvar=x) \
    == ([1, 1], -x/(x**2 + x - 1))
    assert sequence(tribonacci(n)).find_linear_recurrence(30,gfvar=x) \
    ==  ([1, 1, 1], -x/(x**3 + x**2 + x - 1))

def test_RecursiveSeq():
    y = Function('y')
    n = Symbol('n')
    fib = RecursiveSeq(y(n - 1) + y(n - 2), y(n), n, [0, 1])
    assert fib.coeff(3) == 2

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\tests\test_discrete_rv.py ===
from sympy.concrete.summations import Sum
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.bessel import besseli
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.zeta_functions import zeta
from sympy.sets.sets import FiniteSet
from sympy.simplify.simplify import simplify
from sympy.utilities.lambdify import lambdify
from sympy.core.relational import Eq, Ne
from sympy.functions.elementary.exponential import exp
from sympy.logic.boolalg import Or
from sympy.sets.fancysets import Range
from sympy.stats import (P, E, variance, density, characteristic_function,
                         where, moment_generating_function, skewness, cdf,
                         kurtosis, coskewness)
from sympy.stats.drv_types import (PoissonDistribution, GeometricDistribution,
                                   FlorySchulz, Poisson, Geometric, Hermite, Logarithmic,
                                    NegativeBinomial, Skellam, YuleSimon, Zeta,
                                    DiscreteRV)
from sympy.testing.pytest import slow, nocache_fail, raises, skip
from sympy.stats.symbolic_probability import Expectation
from sympy.functions.combinatorial.factorials import FallingFactorial

x = Symbol('x')


def test_PoissonDistribution():
    l = 3
    p = PoissonDistribution(l)
    assert abs(p.cdf(10).evalf() - 1) < .001
    assert abs(p.cdf(10.4).evalf() - 1) < .001
    assert p.expectation(x, x) == l
    assert p.expectation(x**2, x) - p.expectation(x, x)**2 == l


def test_Poisson():
    l = 3
    x = Poisson('x', l)
    assert E(x) == l
    assert E(2*x) == 2*l
    assert variance(x) == l
    assert density(x) == PoissonDistribution(l)
    assert isinstance(E(x, evaluate=False), Expectation)
    assert isinstance(E(2*x, evaluate=False), Expectation)
    # issue 8248
    assert x.pspace.compute_expectation(1) == 1
    # issue 27344
    try:
        import numpy as np
    except ImportError:
        skip("numpy not installed")
    y = Poisson('y', np.float64(4.72544290380919e-11))
    assert E(y) == 4.72544290380919e-11
    y = Poisson('y', np.float64(4.725442903809197e-11))
    assert E(y) == 4.725442903809197e-11
    l2 = 5
    z = Poisson('z', l2)
    assert E(z) == l2
    assert E(FallingFactorial(z, 3)) == l2**3
    assert E(z**2) == l2 + l2**2


def test_FlorySchulz():
    a = Symbol("a")
    z = Symbol("z")
    x = FlorySchulz('x', a)
    assert E(x) == (2 - a)/a
    assert (variance(x) - 2*(1 - a)/a**2).simplify() == S(0)
    assert density(x)(z) == a**2*z*(1 - a)**(z - 1)


@slow
def test_GeometricDistribution():
    p = S.One / 5
    d = GeometricDistribution(p)
    assert d.expectation(x, x) == 1/p
    assert d.expectation(x**2, x) - d.expectation(x, x)**2 == (1-p)/p**2
    assert abs(d.cdf(20000).evalf() - 1) < .001
    assert abs(d.cdf(20000.8).evalf() - 1) < .001
    G = Geometric('G', p=S(1)/4)
    assert cdf(G)(S(7)/2) == P(G <= S(7)/2)

    X = Geometric('X', Rational(1, 5))
    Y = Geometric('Y', Rational(3, 10))
    assert coskewness(X, X + Y, X + 2*Y).simplify() == sqrt(230)*Rational(81, 1150)


def test_Hermite():
    a1 = Symbol("a1", positive=True)
    a2 = Symbol("a2", negative=True)
    raises(ValueError, lambda: Hermite("H", a1, a2))

    a1 = Symbol("a1", negative=True)
    a2 = Symbol("a2", positive=True)
    raises(ValueError, lambda: Hermite("H", a1, a2))

    a1 = Symbol("a1", positive=True)
    x = Symbol("x")
    H = Hermite("H", a1, a2)
    assert moment_generating_function(H)(x) == exp(a1*(exp(x) - 1)
                                            + a2*(exp(2*x) - 1))
    assert characteristic_function(H)(x) == exp(a1*(exp(I*x) - 1)
                                            + a2*(exp(2*I*x) - 1))
    assert E(H) == a1 + 2*a2

    H = Hermite("H", a1=5, a2=4)
    assert density(H)(2) == 33*exp(-9)/2
    assert E(H) == 13
    assert variance(H) == 21
    assert kurtosis(H) == Rational(464,147)
    assert skewness(H) == 37*sqrt(21)/441

def test_Logarithmic():
    p = S.Half
    x = Logarithmic('x', p)
    assert E(x) == -p / ((1 - p) * log(1 - p))
    assert variance(x) == -1/log(2)**2 + 2/log(2)
    assert E(2*x**2 + 3*x + 4) == 4 + 7 / log(2)
    assert isinstance(E(x, evaluate=False), Expectation)


@nocache_fail
def test_negative_binomial():
    r = 5
    p = S.One / 3
    x = NegativeBinomial('x', r, p)
    assert E(x) == r * (1 - p) / p
    # This hangs when run with the cache disabled:
    assert variance(x) == r * (1 - p) / p**2
    assert E(x**5 + 2*x + 3) == E(x**5) + 2*E(x) + 3 == Rational(796473, 1)
    assert isinstance(E(x, evaluate=False), Expectation)


def test_skellam():
    mu1 = Symbol('mu1')
    mu2 = Symbol('mu2')
    z = Symbol('z')
    X = Skellam('x', mu1, mu2)

    assert density(X)(z) == (mu1/mu2)**(z/2) * \
        exp(-mu1 - mu2)*besseli(z, 2*sqrt(mu1*mu2))
    assert skewness(X).expand() == mu1/(mu1*sqrt(mu1 + mu2) + mu2 *
                sqrt(mu1 + mu2)) - mu2/(mu1*sqrt(mu1 + mu2) + mu2*sqrt(mu1 + mu2))
    assert variance(X).expand() == mu1 + mu2
    assert E(X) == mu1 - mu2
    assert characteristic_function(X)(z) == exp(
        mu1*exp(I*z) - mu1 - mu2 + mu2*exp(-I*z))
    assert moment_generating_function(X)(z) == exp(
        mu1*exp(z) - mu1 - mu2 + mu2*exp(-z))


def test_yule_simon():
    from sympy.core.singleton import S
    rho = S(3)
    x = YuleSimon('x', rho)
    assert simplify(E(x)) == rho / (rho - 1)
    assert simplify(variance(x)) == rho**2 / ((rho - 1)**2 * (rho - 2))
    assert isinstance(E(x, evaluate=False), Expectation)
    # To test the cdf function
    assert cdf(x)(x) == Piecewise((-beta(floor(x), 4)*floor(x) + 1, x >= 1), (0, True))


def test_zeta():
    s = S(5)
    x = Zeta('x', s)
    assert E(x) == zeta(s-1) / zeta(s)
    assert simplify(variance(x)) == (
        zeta(s) * zeta(s-2) - zeta(s-1)**2) / zeta(s)**2


def test_discrete_probability():
    X = Geometric('X', Rational(1, 5))
    Y = Poisson('Y', 4)
    G = Geometric('e', x)
    assert P(Eq(X, 3)) == Rational(16, 125)
    assert P(X < 3) == Rational(9, 25)
    assert P(X > 3) == Rational(64, 125)
    assert P(X >= 3) == Rational(16, 25)
    assert P(X <= 3) == Rational(61, 125)
    assert P(Ne(X, 3)) == Rational(109, 125)
    assert P(Eq(Y, 3)) == 32*exp(-4)/3
    assert P(Y < 3) == 13*exp(-4)
    assert P(Y > 3).equals(32*(Rational(-71, 32) + 3*exp(4)/32)*exp(-4)/3)
    assert P(Y >= 3).equals(32*(Rational(-39, 32) + 3*exp(4)/32)*exp(-4)/3)
    assert P(Y <= 3) == 71*exp(-4)/3
    assert P(Ne(Y, 3)).equals(
        13*exp(-4) + 32*(Rational(-71, 32) + 3*exp(4)/32)*exp(-4)/3)
    assert P(X < S.Infinity) is S.One
    assert P(X > S.Infinity) is S.Zero
    assert P(G < 3) == x*(2-x)
    assert P(Eq(G, 3)) == x*(-x + 1)**2


def test_DiscreteRV():
    p = S(1)/2
    x = Symbol('x', integer=True, positive=True)
    pdf = p*(1 - p)**(x - 1) # pdf of Geometric Distribution
    D = DiscreteRV(x, pdf, set=S.Naturals, check=True)
    assert E(D) == E(Geometric('G', S(1)/2)) == 2
    assert P(D > 3) == S(1)/8
    assert D.pspace.domain.set == S.Naturals
    raises(ValueError, lambda: DiscreteRV(x, x, FiniteSet(*range(4)), check=True))

    # purposeful invalid pmf but it should not raise since check=False
    # see test_drv_types.test_ContinuousRV for explanation
    X = DiscreteRV(x, 1/x, S.Naturals)
    assert P(X < 2) == 1
    assert E(X) == oo

def test_precomputed_characteristic_functions():
    import mpmath

    def test_cf(dist, support_lower_limit, support_upper_limit):
        pdf = density(dist)
        t = S('t')
        x = S('x')

        # first function is the hardcoded CF of the distribution
        cf1 = lambdify([t], characteristic_function(dist)(t), 'mpmath')

        # second function is the Fourier transform of the density function
        f = lambdify([x, t], pdf(x)*exp(I*x*t), 'mpmath')
        cf2 = lambda t: mpmath.nsum(lambda x: f(x, t), [
            support_lower_limit, support_upper_limit], maxdegree=10)

        # compare the two functions at various points
        for test_point in [2, 5, 8, 11]:
            n1 = cf1(test_point)
            n2 = cf2(test_point)

            assert abs(re(n1) - re(n2)) < 1e-12
            assert abs(im(n1) - im(n2)) < 1e-12

    test_cf(Geometric('g', Rational(1, 3)), 1, mpmath.inf)
    test_cf(Logarithmic('l', Rational(1, 5)), 1, mpmath.inf)
    test_cf(NegativeBinomial('n', 5, Rational(1, 7)), 0, mpmath.inf)
    test_cf(Poisson('p', 5), 0, mpmath.inf)
    test_cf(YuleSimon('y', 5), 1, mpmath.inf)
    test_cf(Zeta('z', 5), 1, mpmath.inf)


def test_moment_generating_functions():
    t = S('t')

    geometric_mgf = moment_generating_function(Geometric('g', S.Half))(t)
    assert geometric_mgf.diff(t).subs(t, 0) == 2

    logarithmic_mgf = moment_generating_function(Logarithmic('l', S.Half))(t)
    assert logarithmic_mgf.diff(t).subs(t, 0) == 1/log(2)

    negative_binomial_mgf = moment_generating_function(
        NegativeBinomial('n', 5, Rational(1, 3)))(t)
    assert negative_binomial_mgf.diff(t).subs(t, 0) == Rational(10, 1)

    poisson_mgf = moment_generating_function(Poisson('p', 5))(t)
    assert poisson_mgf.diff(t).subs(t, 0) == 5

    skellam_mgf = moment_generating_function(Skellam('s', 1, 1))(t)
    assert skellam_mgf.diff(t).subs(
        t, 2) == (-exp(-2) + exp(2))*exp(-2 + exp(-2) + exp(2))

    yule_simon_mgf = moment_generating_function(YuleSimon('y', 3))(t)
    assert simplify(yule_simon_mgf.diff(t).subs(t, 0)) == Rational(3, 2)

    zeta_mgf = moment_generating_function(Zeta('z', 5))(t)
    assert zeta_mgf.diff(t).subs(t, 0) == pi**4/(90*zeta(5))


def test_Or():
    X = Geometric('X', S.Half)
    assert P(Or(X < 3, X > 4)) == Rational(13, 16)
    assert P(Or(X > 2, X > 1)) == P(X > 1)
    assert P(Or(X >= 3, X < 3)) == 1


def test_where():
    X = Geometric('X', Rational(1, 5))
    Y = Poisson('Y', 4)
    assert where(X**2 > 4).set == Range(3, S.Infinity, 1)
    assert where(X**2 >= 4).set == Range(2, S.Infinity, 1)
    assert where(Y**2 < 9).set == Range(0, 3, 1)
    assert where(Y**2 <= 9).set == Range(0, 4, 1)


def test_conditional():
    X = Geometric('X', Rational(2, 3))
    Y = Poisson('Y', 3)
    assert P(X > 2, X > 3) == 1
    assert P(X > 3, X > 2) == Rational(1, 3)
    assert P(Y > 2, Y < 2) == 0
    assert P(Eq(Y, 3), Y >= 0) == 9*exp(-3)/2
    assert P(Eq(Y, 3), Eq(Y, 2)) == 0
    assert P(X < 2, Eq(X, 2)) == 0
    assert P(X > 2, Eq(X, 3)) == 1


def test_product_spaces():
    X1 = Geometric('X1', S.Half)
    X2 = Geometric('X2', Rational(1, 3))
    assert str(P(X1 + X2 < 3).rewrite(Sum)) == (
        "Sum(Piecewise((1/(4*2**n), n >= -1), (0, True)), (n, -oo, -1))/3")
    assert str(P(X1 + X2 > 3).rewrite(Sum)) == (
        'Sum(Piecewise((2**(X2 - n - 2)*(2/3)**(X2 - 1)/6, '
        'X2 - n <= 2), (0, True)), (X2, 1, oo), (n, 1, oo))')
    assert P(Eq(X1 + X2, 3)) == Rational(1, 12)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\decimal\array.py ===
from __future__ import annotations

import decimal
import numbers
import sys
from typing import TYPE_CHECKING

import numpy as np

from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
    is_dtype_equal,
    is_float,
    is_integer,
    pandas_dtype,
)

import pandas as pd
from pandas.api.extensions import (
    no_default,
    register_extension_dtype,
)
from pandas.api.types import (
    is_list_like,
    is_scalar,
)
from pandas.core import arraylike
from pandas.core.algorithms import value_counts_internal as value_counts
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import (
    ExtensionArray,
    ExtensionScalarOpsMixin,
)
from pandas.core.indexers import check_array_indexer

if TYPE_CHECKING:
    from pandas._typing import type_t


@register_extension_dtype
class DecimalDtype(ExtensionDtype):
    type = decimal.Decimal
    name = "decimal"
    na_value = decimal.Decimal("NaN")
    _metadata = ("context",)

    def __init__(self, context=None) -> None:
        self.context = context or decimal.getcontext()

    def __repr__(self) -> str:
        return f"DecimalDtype(context={self.context})"

    @classmethod
    def construct_array_type(cls) -> type_t[DecimalArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return DecimalArray

    @property
    def _is_numeric(self) -> bool:
        return True


class DecimalArray(OpsMixin, ExtensionScalarOpsMixin, ExtensionArray):
    __array_priority__ = 1000

    def __init__(self, values, dtype=None, copy=False, context=None) -> None:
        for i, val in enumerate(values):
            if is_float(val) or is_integer(val):
                if np.isnan(val):
                    values[i] = DecimalDtype.na_value
                else:
                    # error: Argument 1 has incompatible type "float | int |
                    # integer[Any]"; expected "Decimal | float | str | tuple[int,
                    # Sequence[int], int]"
                    values[i] = DecimalDtype.type(val)  # type: ignore[arg-type]
            elif not isinstance(val, decimal.Decimal):
                raise TypeError("All values must be of type " + str(decimal.Decimal))
        values = np.asarray(values, dtype=object)

        self._data = values
        # Some aliases for common attribute names to ensure pandas supports
        # these
        self._items = self.data = self._data
        # those aliases are currently not working due to assumptions
        # in internal code (GH-20735)
        # self._values = self.values = self.data
        self._dtype = DecimalDtype(context)

    @property
    def dtype(self):
        return self._dtype

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        return cls(scalars)

    @classmethod
    def _from_sequence_of_strings(cls, strings, dtype=None, copy=False):
        return cls._from_sequence(
            [decimal.Decimal(x) for x in strings], dtype=dtype, copy=copy
        )

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values)

    _HANDLED_TYPES = (decimal.Decimal, numbers.Number, np.ndarray)

    def to_numpy(
        self,
        dtype=None,
        copy: bool = False,
        na_value: object = no_default,
        decimals=None,
    ) -> np.ndarray:
        result = np.asarray(self, dtype=dtype)
        if decimals is not None:
            result = np.asarray([round(x, decimals) for x in result])
        return result

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        #
        if not all(
            isinstance(t, self._HANDLED_TYPES + (DecimalArray,)) for t in inputs
        ):
            return NotImplemented

        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is not NotImplemented:
            # e.g. test_array_ufunc_series_scalar_other
            return result

        if "out" in kwargs:
            return arraylike.dispatch_ufunc_with_out(
                self, ufunc, method, *inputs, **kwargs
            )

        inputs = tuple(x._data if isinstance(x, DecimalArray) else x for x in inputs)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if method == "reduce":
            result = arraylike.dispatch_reduction_ufunc(
                self, ufunc, method, *inputs, **kwargs
            )
            if result is not NotImplemented:
                return result

        def reconstruct(x):
            if isinstance(x, (decimal.Decimal, numbers.Number)):
                return x
            else:
                return type(self)._from_sequence(x, dtype=self.dtype)

        if ufunc.nout > 1:
            return tuple(reconstruct(x) for x in result)
        else:
            return reconstruct(result)

    def __getitem__(self, item):
        if isinstance(item, numbers.Integral):
            return self._data[item]
        else:
            # array, slice.
            item = pd.api.indexers.check_array_indexer(self, item)
            return type(self)(self._data[item])

    def take(self, indexer, allow_fill=False, fill_value=None):
        from pandas.api.extensions import take

        data = self._data
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value

        result = take(data, indexer, fill_value=fill_value, allow_fill=allow_fill)
        return self._from_sequence(result, dtype=self.dtype)

    def copy(self):
        return type(self)(self._data.copy(), dtype=self.dtype)

    def astype(self, dtype, copy=True):
        if is_dtype_equal(dtype, self._dtype):
            if not copy:
                return self
        dtype = pandas_dtype(dtype)
        if isinstance(dtype, type(self.dtype)):
            return type(self)(self._data, copy=copy, context=dtype.context)

        return super().astype(dtype, copy=copy)

    def __setitem__(self, key, value) -> None:
        if is_list_like(value):
            if is_scalar(key):
                raise ValueError("setting an array element with a sequence.")
            value = [decimal.Decimal(v) for v in value]
        else:
            value = decimal.Decimal(value)

        key = check_array_indexer(self, key)
        self._data[key] = value

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, item) -> bool | np.bool_:
        if not isinstance(item, decimal.Decimal):
            return False
        elif item.is_nan():
            return self.isna().any()
        else:
            return super().__contains__(item)

    @property
    def nbytes(self) -> int:
        n = len(self)
        if n:
            return n * sys.getsizeof(self[0])
        return 0

    def isna(self):
        return np.array([x.is_nan() for x in self._data], dtype=bool)

    @property
    def _na_value(self):
        return decimal.Decimal("NaN")

    def _formatter(self, boxed=False):
        if boxed:
            return "Decimal: {}".format
        return repr

    @classmethod
    def _concat_same_type(cls, to_concat):
        return cls(np.concatenate([x._data for x in to_concat]))

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ):
        if skipna and self.isna().any():
            # If we don't have any NAs, we can ignore skipna
            other = self[~self.isna()]
            result = other._reduce(name, **kwargs)
        elif name == "sum" and len(self) == 0:
            # GH#29630 avoid returning int 0 or np.bool_(False) on old numpy
            result = decimal.Decimal(0)
        else:
            try:
                op = getattr(self.data, name)
            except AttributeError as err:
                raise NotImplementedError(
                    f"decimal does not support the {name} operation"
                ) from err
            result = op(axis=0)

        if keepdims:
            return type(self)([result])
        else:
            return result

    def _cmp_method(self, other, op):
        # For use with OpsMixin
        def convert_values(param):
            if isinstance(param, ExtensionArray) or is_list_like(param):
                ovalues = param
            else:
                # Assume it's an object
                ovalues = [param] * len(self)
            return ovalues

        lvalues = self
        rvalues = convert_values(other)

        # If the operator is not defined for the underlying objects,
        # a TypeError should be raised
        res = [op(a, b) for (a, b) in zip(lvalues, rvalues)]

        return np.asarray(res, dtype=bool)

    def value_counts(self, dropna: bool = True):
        return value_counts(self.to_numpy(), dropna=dropna)

    # We override fillna here to simulate a 3rd party EA that has done so. This
    #  lets us test the deprecation telling authors to implement _pad_or_backfill
    # Simulate a 3rd-party EA that has not yet updated to include a "copy"
    #  keyword in its fillna method.
    # error: Signature of "fillna" incompatible with supertype "ExtensionArray"
    def fillna(  # type: ignore[override]
        self,
        value=None,
        method=None,
        limit: int | None = None,
    ):
        return super().fillna(value=value, method=method, limit=limit, copy=True)


def to_decimal(values, context=None):
    return DecimalArray([decimal.Decimal(x) for x in values], context=context)


def make_data():
    return [decimal.Decimal(val) for val in np.random.default_rng(2).random(100)]


DecimalArray._add_arithmetic_ops()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\test_ufunc.py ===
from functools import partial
import re

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_extension_array_dtype

dtypes = [
    "int64",
    "Int64",
    {"A": "int64", "B": "Int64"},
]


@pytest.mark.parametrize("dtype", dtypes)
def test_unary_unary(dtype):
    # unary input, unary output
    values = np.array([[-1, -1], [1, 1]], dtype="int64")
    df = pd.DataFrame(values, columns=["A", "B"], index=["a", "b"]).astype(dtype=dtype)
    result = np.positive(df)
    expected = pd.DataFrame(
        np.positive(values), index=df.index, columns=df.columns
    ).astype(dtype)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", dtypes)
def test_unary_binary(request, dtype):
    # unary input, binary output
    if is_extension_array_dtype(dtype) or isinstance(dtype, dict):
        request.applymarker(
            pytest.mark.xfail(
                reason="Extension / mixed with multiple outputs not implemented."
            )
        )

    values = np.array([[-1, -1], [1, 1]], dtype="int64")
    df = pd.DataFrame(values, columns=["A", "B"], index=["a", "b"]).astype(dtype=dtype)
    result_pandas = np.modf(df)
    assert isinstance(result_pandas, tuple)
    assert len(result_pandas) == 2
    expected_numpy = np.modf(values)

    for result, b in zip(result_pandas, expected_numpy):
        expected = pd.DataFrame(b, index=df.index, columns=df.columns)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", dtypes)
def test_binary_input_dispatch_binop(dtype):
    # binop ufuncs are dispatched to our dunder methods.
    values = np.array([[-1, -1], [1, 1]], dtype="int64")
    df = pd.DataFrame(values, columns=["A", "B"], index=["a", "b"]).astype(dtype=dtype)
    result = np.add(df, df)
    expected = pd.DataFrame(
        np.add(values, values), index=df.index, columns=df.columns
    ).astype(dtype)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "func,arg,expected",
    [
        (np.add, 1, [2, 3, 4, 5]),
        (
            partial(np.add, where=[[False, True], [True, False]]),
            np.array([[1, 1], [1, 1]]),
            [0, 3, 4, 0],
        ),
        (np.power, np.array([[1, 1], [2, 2]]), [1, 2, 9, 16]),
        (np.subtract, 2, [-1, 0, 1, 2]),
        (
            partial(np.negative, where=np.array([[False, True], [True, False]])),
            None,
            [0, -2, -3, 0],
        ),
    ],
)
def test_ufunc_passes_args(func, arg, expected):
    # GH#40662
    arr = np.array([[1, 2], [3, 4]])
    df = pd.DataFrame(arr)
    result_inplace = np.zeros_like(arr)
    # 1-argument ufunc
    if arg is None:
        result = func(df, out=result_inplace)
    else:
        result = func(df, arg, out=result_inplace)

    expected = np.array(expected).reshape(2, 2)
    tm.assert_numpy_array_equal(result_inplace, expected)

    expected = pd.DataFrame(expected)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype_a", dtypes)
@pytest.mark.parametrize("dtype_b", dtypes)
def test_binary_input_aligns_columns(request, dtype_a, dtype_b):
    if (
        is_extension_array_dtype(dtype_a)
        or isinstance(dtype_a, dict)
        or is_extension_array_dtype(dtype_b)
        or isinstance(dtype_b, dict)
    ):
        request.applymarker(
            pytest.mark.xfail(
                reason="Extension / mixed with multiple inputs not implemented."
            )
        )

    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]}).astype(dtype_a)

    if isinstance(dtype_a, dict) and isinstance(dtype_b, dict):
        dtype_b = dtype_b.copy()
        dtype_b["C"] = dtype_b.pop("B")
    df2 = pd.DataFrame({"A": [1, 2], "C": [3, 4]}).astype(dtype_b)
    # As of 2.0, align first before applying the ufunc
    result = np.heaviside(df1, df2)
    expected = np.heaviside(
        np.array([[1, 3, np.nan], [2, 4, np.nan]]),
        np.array([[1, np.nan, 3], [2, np.nan, 4]]),
    )
    expected = pd.DataFrame(expected, index=[0, 1], columns=["A", "B", "C"])
    tm.assert_frame_equal(result, expected)

    result = np.heaviside(df1, df2.values)
    expected = pd.DataFrame([[1.0, 1.0], [1.0, 1.0]], columns=["A", "B"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", dtypes)
def test_binary_input_aligns_index(request, dtype):
    if is_extension_array_dtype(dtype) or isinstance(dtype, dict):
        request.applymarker(
            pytest.mark.xfail(
                reason="Extension / mixed with multiple inputs not implemented."
            )
        )
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["a", "b"]).astype(dtype)
    df2 = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["a", "c"]).astype(dtype)
    result = np.heaviside(df1, df2)
    expected = np.heaviside(
        np.array([[1, 3], [3, 4], [np.nan, np.nan]]),
        np.array([[1, 3], [np.nan, np.nan], [3, 4]]),
    )
    # TODO(FloatArray): this will be Float64Dtype.
    expected = pd.DataFrame(expected, index=["a", "b", "c"], columns=["A", "B"])
    tm.assert_frame_equal(result, expected)

    result = np.heaviside(df1, df2.values)
    expected = pd.DataFrame(
        [[1.0, 1.0], [1.0, 1.0]], columns=["A", "B"], index=["a", "b"]
    )
    tm.assert_frame_equal(result, expected)


def test_binary_frame_series_raises():
    # We don't currently implement
    df = pd.DataFrame({"A": [1, 2]})
    with pytest.raises(NotImplementedError, match="logaddexp"):
        np.logaddexp(df, df["A"])

    with pytest.raises(NotImplementedError, match="logaddexp"):
        np.logaddexp(df["A"], df)


def test_unary_accumulate_axis():
    # https://github.com/pandas-dev/pandas/issues/39259
    df = pd.DataFrame({"a": [1, 3, 2, 4]})
    result = np.maximum.accumulate(df)
    expected = pd.DataFrame({"a": [1, 3, 3, 4]})
    tm.assert_frame_equal(result, expected)

    df = pd.DataFrame({"a": [1, 3, 2, 4], "b": [0.1, 4.0, 3.0, 2.0]})
    result = np.maximum.accumulate(df)
    # in theory could preserve int dtype for default axis=0
    expected = pd.DataFrame({"a": [1.0, 3.0, 3.0, 4.0], "b": [0.1, 4.0, 4.0, 4.0]})
    tm.assert_frame_equal(result, expected)

    result = np.maximum.accumulate(df, axis=0)
    tm.assert_frame_equal(result, expected)

    result = np.maximum.accumulate(df, axis=1)
    expected = pd.DataFrame({"a": [1.0, 3.0, 2.0, 4.0], "b": [1.0, 4.0, 3.0, 4.0]})
    tm.assert_frame_equal(result, expected)


def test_frame_outer_disallowed():
    df = pd.DataFrame({"A": [1, 2]})
    with pytest.raises(NotImplementedError, match=""):
        # deprecation enforced in 2.0
        np.subtract.outer(df, df)


def test_alignment_deprecation_enforced():
    # Enforced in 2.0
    # https://github.com/pandas-dev/pandas/issues/39184
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = pd.DataFrame({"b": [1, 2, 3], "c": [4, 5, 6]})
    s1 = pd.Series([1, 2], index=["a", "b"])
    s2 = pd.Series([1, 2], index=["b", "c"])

    # binary dataframe / dataframe
    expected = pd.DataFrame({"a": [2, 4, 6], "b": [8, 10, 12]})

    with tm.assert_produces_warning(None):
        # aligned -> no warning!
        result = np.add(df1, df1)
    tm.assert_frame_equal(result, expected)

    result = np.add(df1, df2.values)
    tm.assert_frame_equal(result, expected)

    result = np.add(df1, df2)
    expected = pd.DataFrame({"a": [np.nan] * 3, "b": [5, 7, 9], "c": [np.nan] * 3})
    tm.assert_frame_equal(result, expected)

    result = np.add(df1.values, df2)
    expected = pd.DataFrame({"b": [2, 4, 6], "c": [8, 10, 12]})
    tm.assert_frame_equal(result, expected)

    # binary dataframe / series
    expected = pd.DataFrame({"a": [2, 3, 4], "b": [6, 7, 8]})

    with tm.assert_produces_warning(None):
        # aligned -> no warning!
        result = np.add(df1, s1)
    tm.assert_frame_equal(result, expected)

    result = np.add(df1, s2.values)
    tm.assert_frame_equal(result, expected)

    expected = pd.DataFrame(
        {"a": [np.nan] * 3, "b": [5.0, 6.0, 7.0], "c": [np.nan] * 3}
    )
    result = np.add(df1, s2)
    tm.assert_frame_equal(result, expected)

    msg = "Cannot apply ufunc <ufunc 'add'> to mixed DataFrame and Series inputs."
    with pytest.raises(NotImplementedError, match=msg):
        np.add(s2, df1)


def test_alignment_deprecation_many_inputs_enforced():
    # Enforced in 2.0
    # https://github.com/pandas-dev/pandas/issues/39184
    # test that the deprecation also works with > 2 inputs -> using a numba
    # written ufunc for this because numpy itself doesn't have such ufuncs
    numba = pytest.importorskip("numba")

    @numba.vectorize([numba.float64(numba.float64, numba.float64, numba.float64)])
    def my_ufunc(x, y, z):
        return x + y + z

    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = pd.DataFrame({"b": [1, 2, 3], "c": [4, 5, 6]})
    df3 = pd.DataFrame({"a": [1, 2, 3], "c": [4, 5, 6]})

    result = my_ufunc(df1, df2, df3)
    expected = pd.DataFrame(np.full((3, 3), np.nan), columns=["a", "b", "c"])
    tm.assert_frame_equal(result, expected)

    # all aligned -> no warning
    with tm.assert_produces_warning(None):
        result = my_ufunc(df1, df1, df1)
    expected = pd.DataFrame([[3.0, 12.0], [6.0, 15.0], [9.0, 18.0]], columns=["a", "b"])
    tm.assert_frame_equal(result, expected)

    # mixed frame / arrays
    msg = (
        r"operands could not be broadcast together with shapes \(3,3\) \(3,3\) \(3,2\)"
    )
    with pytest.raises(ValueError, match=msg):
        my_ufunc(df1, df2, df3.values)

    # single frame -> no warning
    with tm.assert_produces_warning(None):
        result = my_ufunc(df1, df2.values, df3.values)
    tm.assert_frame_equal(result, expected)

    # takes indices of first frame
    msg = (
        r"operands could not be broadcast together with shapes \(3,2\) \(3,3\) \(3,3\)"
    )
    with pytest.raises(ValueError, match=msg):
        my_ufunc(df1.values, df2, df3)


def test_array_ufuncs_for_many_arguments():
    # GH39853
    def add3(x, y, z):
        return x + y + z

    ufunc = np.frompyfunc(add3, 3, 1)
    df = pd.DataFrame([[1, 2], [3, 4]])

    result = ufunc(df, df, 1)
    expected = pd.DataFrame([[3, 5], [7, 9]], dtype=object)
    tm.assert_frame_equal(result, expected)

    ser = pd.Series([1, 2])
    msg = (
        "Cannot apply ufunc <ufunc 'add3 (vectorized)'> "
        "to mixed DataFrame and Series inputs."
    )
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        ufunc(df, df, ser)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_convert_dtypes.py ===
from itertools import product

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas._libs import lib

import pandas as pd
import pandas._testing as tm

# Each test case consists of a tuple with the data and dtype to create the
# test Series, the default dtype for the expected result (which is valid
# for most cases), and the specific cases where the result deviates from
# this default. Those overrides are defined as a dict with (keyword, val) as
# dictionary key. In case of multiple items, the last override takes precedence.


@pytest.fixture(
    params=[
        (
            # data
            [1, 2, 3],
            # original dtype
            np.dtype("int32"),
            # default expected dtype
            "Int32",
            # exceptions on expected dtype
            {("convert_integer", False): np.dtype("int32")},
        ),
        (
            [1, 2, 3],
            np.dtype("int64"),
            "Int64",
            {("convert_integer", False): np.dtype("int64")},
        ),
        (
            ["x", "y", "z"],
            np.dtype("O"),
            pd.StringDtype(),
            {("convert_string", False): np.dtype("O")},
        ),
        (
            [True, False, np.nan],
            np.dtype("O"),
            pd.BooleanDtype(),
            {("convert_boolean", False): np.dtype("O")},
        ),
        (
            ["h", "i", np.nan],
            np.dtype("O"),
            pd.StringDtype(),
            {("convert_string", False): np.dtype("O")},
        ),
        (  # GH32117
            ["h", "i", 1],
            np.dtype("O"),
            np.dtype("O"),
            {},
        ),
        (
            [10, np.nan, 20],
            np.dtype("float"),
            "Int64",
            {
                ("convert_integer", False, "convert_floating", True): "Float64",
                ("convert_integer", False, "convert_floating", False): np.dtype(
                    "float"
                ),
            },
        ),
        (
            [np.nan, 100.5, 200],
            np.dtype("float"),
            "Float64",
            {("convert_floating", False): np.dtype("float")},
        ),
        (
            [3, 4, 5],
            "Int8",
            "Int8",
            {},
        ),
        (
            [[1, 2], [3, 4], [5]],
            None,
            np.dtype("O"),
            {},
        ),
        (
            [4, 5, 6],
            np.dtype("uint32"),
            "UInt32",
            {("convert_integer", False): np.dtype("uint32")},
        ),
        (
            [-10, 12, 13],
            np.dtype("i1"),
            "Int8",
            {("convert_integer", False): np.dtype("i1")},
        ),
        (
            [1.2, 1.3],
            np.dtype("float32"),
            "Float32",
            {("convert_floating", False): np.dtype("float32")},
        ),
        (
            [1, 2.0],
            object,
            "Int64",
            {
                ("convert_integer", False): "Float64",
                ("convert_integer", False, "convert_floating", False): np.dtype(
                    "float"
                ),
                ("infer_objects", False): np.dtype("object"),
            },
        ),
        (
            [1, 2.5],
            object,
            "Float64",
            {
                ("convert_floating", False): np.dtype("float"),
                ("infer_objects", False): np.dtype("object"),
            },
        ),
        (["a", "b"], pd.CategoricalDtype(), pd.CategoricalDtype(), {}),
        (
            pd.to_datetime(["2020-01-14 10:00", "2020-01-15 11:11"]).as_unit("s"),
            pd.DatetimeTZDtype(tz="UTC"),
            pd.DatetimeTZDtype(tz="UTC"),
            {},
        ),
        (
            pd.to_datetime(["2020-01-14 10:00", "2020-01-15 11:11"]).as_unit("ms"),
            pd.DatetimeTZDtype(tz="UTC"),
            pd.DatetimeTZDtype(tz="UTC"),
            {},
        ),
        (
            pd.to_datetime(["2020-01-14 10:00", "2020-01-15 11:11"]).as_unit("us"),
            pd.DatetimeTZDtype(tz="UTC"),
            pd.DatetimeTZDtype(tz="UTC"),
            {},
        ),
        (
            pd.to_datetime(["2020-01-14 10:00", "2020-01-15 11:11"]).as_unit("ns"),
            pd.DatetimeTZDtype(tz="UTC"),
            pd.DatetimeTZDtype(tz="UTC"),
            {},
        ),
        (
            pd.to_datetime(["2020-01-14 10:00", "2020-01-15 11:11"]).as_unit("ns"),
            "datetime64[ns]",
            np.dtype("datetime64[ns]"),
            {},
        ),
        (
            pd.to_datetime(["2020-01-14 10:00", "2020-01-15 11:11"]).as_unit("ns"),
            object,
            np.dtype("datetime64[ns]"),
            {("infer_objects", False): np.dtype("object")},
        ),
        (
            pd.period_range("1/1/2011", freq="M", periods=3),
            None,
            pd.PeriodDtype("M"),
            {},
        ),
        (
            pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)]),
            None,
            pd.IntervalDtype("int64", "right"),
            {},
        ),
    ]
)
def test_cases(request):
    return request.param


class TestSeriesConvertDtypes:
    @pytest.mark.xfail(using_string_dtype(), reason="TODO(infer_string)", strict=False)
    @pytest.mark.parametrize("params", product(*[(True, False)] * 5))
    def test_convert_dtypes(
        self,
        test_cases,
        params,
        using_infer_string,
    ):
        data, maindtype, expected_default, expected_other = test_cases
        if (
            hasattr(data, "dtype")
            and lib.is_np_dtype(data.dtype, "M")
            and isinstance(maindtype, pd.DatetimeTZDtype)
        ):
            # this astype is deprecated in favor of tz_localize
            msg = "Cannot use .astype to convert from timezone-naive dtype"
            with pytest.raises(TypeError, match=msg):
                pd.Series(data, dtype=maindtype)
            return

        if maindtype is not None:
            series = pd.Series(data, dtype=maindtype)
        else:
            series = pd.Series(data)

        result = series.convert_dtypes(*params)

        param_names = [
            "infer_objects",
            "convert_string",
            "convert_integer",
            "convert_boolean",
            "convert_floating",
        ]
        params_dict = dict(zip(param_names, params))

        expected_dtype = expected_default
        for spec, dtype in expected_other.items():
            if all(params_dict[key] is val for key, val in zip(spec[::2], spec[1::2])):
                expected_dtype = dtype
        if (
            using_infer_string
            and expected_default == "string"
            and expected_dtype == object
            and params[0]
            and not params[1]
        ):
            # If convert_string=False and infer_objects=True, we end up with the
            # default string dtype instead of preserving object for string data
            expected_dtype = pd.StringDtype(na_value=np.nan)

        expected = pd.Series(data, dtype=expected_dtype)
        tm.assert_series_equal(result, expected)

        # Test that it is a copy
        copy = series.copy(deep=True)

        if result.notna().sum() > 0 and result.dtype in ["interval[int64, right]"]:
            with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
                result[result.notna()] = np.nan
        else:
            result[result.notna()] = np.nan

        # Make sure original not changed
        tm.assert_series_equal(series, copy)

    def test_convert_string_dtype(self, nullable_string_dtype):
        # https://github.com/pandas-dev/pandas/issues/31731 -> converting columns
        # that are already string dtype
        df = pd.DataFrame(
            {"A": ["a", "b", pd.NA], "B": ["ä", "ö", "ü"]}, dtype=nullable_string_dtype
        )
        result = df.convert_dtypes()
        tm.assert_frame_equal(df, result)

    def test_convert_bool_dtype(self):
        # GH32287
        df = pd.DataFrame({"A": pd.array([True])})
        tm.assert_frame_equal(df, df.convert_dtypes())

    def test_convert_byte_string_dtype(self):
        # GH-43183
        byte_str = b"binary-string"

        df = pd.DataFrame(data={"A": byte_str}, index=[0])
        result = df.convert_dtypes()
        expected = df
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "infer_objects, dtype", [(True, "Int64"), (False, "object")]
    )
    def test_convert_dtype_object_with_na(self, infer_objects, dtype):
        # GH#48791
        ser = pd.Series([1, pd.NA])
        result = ser.convert_dtypes(infer_objects=infer_objects)
        expected = pd.Series([1, pd.NA], dtype=dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "infer_objects, dtype", [(True, "Float64"), (False, "object")]
    )
    def test_convert_dtype_object_with_na_float(self, infer_objects, dtype):
        # GH#48791
        ser = pd.Series([1.5, pd.NA])
        result = ser.convert_dtypes(infer_objects=infer_objects)
        expected = pd.Series([1.5, pd.NA], dtype=dtype)
        tm.assert_series_equal(result, expected)

    def test_convert_dtypes_pyarrow_to_np_nullable(self):
        # GH 53648
        pytest.importorskip("pyarrow")
        ser = pd.Series(range(2), dtype="int32[pyarrow]")
        result = ser.convert_dtypes(dtype_backend="numpy_nullable")
        expected = pd.Series(range(2), dtype="Int32")
        tm.assert_series_equal(result, expected)

    def test_convert_dtypes_pyarrow_null(self):
        # GH#55346
        pa = pytest.importorskip("pyarrow")
        ser = pd.Series([None, None])
        result = ser.convert_dtypes(dtype_backend="pyarrow")
        expected = pd.Series([None, None], dtype=pd.ArrowDtype(pa.null()))
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_diff.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestDataFrameDiff:
    def test_diff_requires_integer(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((2, 2)))
        with pytest.raises(ValueError, match="periods must be an integer"):
            df.diff(1.5)

    # GH#44572 np.int64 is accepted
    @pytest.mark.parametrize("num", [1, np.int64(1)])
    def test_diff(self, datetime_frame, num):
        df = datetime_frame
        the_diff = df.diff(num)

        expected = df["A"] - df["A"].shift(num)
        tm.assert_series_equal(the_diff["A"], expected)

    def test_diff_int_dtype(self):
        # int dtype
        a = 10_000_000_000_000_000
        b = a + 1
        ser = Series([a, b])

        rs = DataFrame({"s": ser}).diff()
        assert rs.s[1] == 1

    def test_diff_mixed_numeric(self, datetime_frame):
        # mixed numeric
        tf = datetime_frame.astype("float32")
        the_diff = tf.diff(1)
        tm.assert_series_equal(the_diff["A"], tf["A"] - tf["A"].shift(1))

    def test_diff_axis1_nonconsolidated(self):
        # GH#10907
        df = DataFrame({"y": Series([2]), "z": Series([3])})
        df.insert(0, "x", 1)
        result = df.diff(axis=1)
        expected = DataFrame({"x": np.nan, "y": Series(1), "z": Series(1)})
        tm.assert_frame_equal(result, expected)

    def test_diff_timedelta64_with_nat(self):
        # GH#32441
        arr = np.arange(6).reshape(3, 2).astype("timedelta64[ns]")
        arr[:, 0] = np.timedelta64("NaT", "ns")

        df = DataFrame(arr)
        result = df.diff(1, axis=0)

        expected = DataFrame({0: df[0], 1: [pd.NaT, pd.Timedelta(2), pd.Timedelta(2)]})
        tm.assert_equal(result, expected)

        result = df.diff(0)
        expected = df - df
        assert expected[0].isna().all()
        tm.assert_equal(result, expected)

        result = df.diff(-1, axis=1)
        expected = df * np.nan
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_diff_datetime_axis0_with_nat(self, tz, unit):
        # GH#32441
        dti = pd.DatetimeIndex(["NaT", "2019-01-01", "2019-01-02"], tz=tz).as_unit(unit)
        ser = Series(dti)

        df = ser.to_frame()

        result = df.diff()
        ex_index = pd.TimedeltaIndex([pd.NaT, pd.NaT, pd.Timedelta(days=1)]).as_unit(
            unit
        )
        expected = Series(ex_index).to_frame()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_diff_datetime_with_nat_zero_periods(self, tz):
        # diff on NaT values should give NaT, not timedelta64(0)
        dti = date_range("2016-01-01", periods=4, tz=tz)
        ser = Series(dti)
        df = ser.to_frame().copy()

        df[1] = ser.copy()

        df.iloc[:, 0] = pd.NaT

        expected = df - df
        assert expected[0].isna().all()

        result = df.diff(0, axis=0)
        tm.assert_frame_equal(result, expected)

        result = df.diff(0, axis=1)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_diff_datetime_axis0(self, tz):
        # GH#18578
        df = DataFrame(
            {
                0: date_range("2010", freq="D", periods=2, tz=tz),
                1: date_range("2010", freq="D", periods=2, tz=tz),
            }
        )

        result = df.diff(axis=0)
        expected = DataFrame(
            {
                0: pd.TimedeltaIndex(["NaT", "1 days"]),
                1: pd.TimedeltaIndex(["NaT", "1 days"]),
            }
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_diff_datetime_axis1(self, tz):
        # GH#18578
        df = DataFrame(
            {
                0: date_range("2010", freq="D", periods=2, tz=tz),
                1: date_range("2010", freq="D", periods=2, tz=tz),
            }
        )

        result = df.diff(axis=1)
        expected = DataFrame(
            {
                0: pd.TimedeltaIndex(["NaT", "NaT"]),
                1: pd.TimedeltaIndex(["0 days", "0 days"]),
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_diff_timedelta(self, unit):
        # GH#4533
        df = DataFrame(
            {
                "time": [Timestamp("20130101 9:01"), Timestamp("20130101 9:02")],
                "value": [1.0, 2.0],
            }
        )
        df["time"] = df["time"].dt.as_unit(unit)

        res = df.diff()
        exp = DataFrame(
            [[pd.NaT, np.nan], [pd.Timedelta("00:01:00"), 1]], columns=["time", "value"]
        )
        exp["time"] = exp["time"].dt.as_unit(unit)
        tm.assert_frame_equal(res, exp)

    def test_diff_mixed_dtype(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df["A"] = np.array([1, 2, 3, 4, 5], dtype=object)

        result = df.diff()
        assert result[0].dtype == np.float64

    def test_diff_neg_n(self, datetime_frame):
        rs = datetime_frame.diff(-1)
        xp = datetime_frame - datetime_frame.shift(-1)
        tm.assert_frame_equal(rs, xp)

    def test_diff_float_n(self, datetime_frame):
        rs = datetime_frame.diff(1.0)
        xp = datetime_frame.diff(1)
        tm.assert_frame_equal(rs, xp)

    def test_diff_axis(self):
        # GH#9727
        df = DataFrame([[1.0, 2.0], [3.0, 4.0]])
        tm.assert_frame_equal(
            df.diff(axis=1), DataFrame([[np.nan, 1.0], [np.nan, 1.0]])
        )
        tm.assert_frame_equal(
            df.diff(axis=0), DataFrame([[np.nan, np.nan], [2.0, 2.0]])
        )

    def test_diff_period(self):
        # GH#32995 Don't pass an incorrect axis
        pi = date_range("2016-01-01", periods=3).to_period("D")
        df = DataFrame({"A": pi})

        result = df.diff(1, axis=1)

        expected = (df - pd.NaT).astype(object)
        tm.assert_frame_equal(result, expected)

    def test_diff_axis1_mixed_dtypes(self):
        # GH#32995 operate column-wise when we have mixed dtypes and axis=1
        df = DataFrame({"A": range(3), "B": 2 * np.arange(3, dtype=np.float64)})

        expected = DataFrame({"A": [np.nan, np.nan, np.nan], "B": df["B"] / 2})

        result = df.diff(axis=1)
        tm.assert_frame_equal(result, expected)

        # GH#21437 mixed-float-dtypes
        df = DataFrame(
            {"a": np.arange(3, dtype="float32"), "b": np.arange(3, dtype="float64")}
        )
        result = df.diff(axis=1)
        expected = DataFrame({"a": df["a"] * np.nan, "b": df["b"] * 0})
        tm.assert_frame_equal(result, expected)

    def test_diff_axis1_mixed_dtypes_large_periods(self):
        # GH#32995 operate column-wise when we have mixed dtypes and axis=1
        df = DataFrame({"A": range(3), "B": 2 * np.arange(3, dtype=np.float64)})

        expected = df * np.nan

        result = df.diff(axis=1, periods=3)
        tm.assert_frame_equal(result, expected)

    def test_diff_axis1_mixed_dtypes_negative_periods(self):
        # GH#32995 operate column-wise when we have mixed dtypes and axis=1
        df = DataFrame({"A": range(3), "B": 2 * np.arange(3, dtype=np.float64)})

        expected = DataFrame({"A": -1.0 * df["A"], "B": df["B"] * np.nan})

        result = df.diff(axis=1, periods=-1)
        tm.assert_frame_equal(result, expected)

    def test_diff_sparse(self):
        # GH#28813 .diff() should work for sparse dataframes as well
        sparse_df = DataFrame([[0, 1], [1, 0]], dtype="Sparse[int]")

        result = sparse_df.diff()
        expected = DataFrame(
            [[np.nan, np.nan], [1.0, -1.0]], dtype=pd.SparseDtype("float", 0.0)
        )

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "axis,expected",
        [
            (
                0,
                DataFrame(
                    {
                        "a": [np.nan, 0, 1, 0, np.nan, np.nan, np.nan, 0],
                        "b": [np.nan, 1, np.nan, np.nan, -2, 1, np.nan, np.nan],
                        "c": np.repeat(np.nan, 8),
                        "d": [np.nan, 3, 5, 7, 9, 11, 13, 15],
                    },
                    dtype="Int64",
                ),
            ),
            (
                1,
                DataFrame(
                    {
                        "a": np.repeat(np.nan, 8),
                        "b": [0, 1, np.nan, 1, np.nan, np.nan, np.nan, 0],
                        "c": np.repeat(np.nan, 8),
                        "d": np.repeat(np.nan, 8),
                    },
                    dtype="Int64",
                ),
            ),
        ],
    )
    def test_diff_integer_na(self, axis, expected):
        # GH#24171 IntegerNA Support for DataFrame.diff()
        df = DataFrame(
            {
                "a": np.repeat([0, 1, np.nan, 2], 2),
                "b": np.tile([0, 1, np.nan, 2], 2),
                "c": np.repeat(np.nan, 8),
                "d": np.arange(1, 9) ** 2,
            },
            dtype="Int64",
        )

        # Test case for default behaviour of diff
        result = df.diff(axis=axis)
        tm.assert_frame_equal(result, expected)

    def test_diff_readonly(self):
        # https://github.com/pandas-dev/pandas/issues/35559
        arr = np.random.default_rng(2).standard_normal((5, 2))
        arr.flags.writeable = False
        df = DataFrame(arr)
        result = df.diff()
        expected = DataFrame(np.array(df)).diff()
        tm.assert_frame_equal(result, expected)

    def test_diff_all_int_dtype(self, any_int_numpy_dtype):
        # GH 14773
        df = DataFrame(range(5))
        df = df.astype(any_int_numpy_dtype)
        result = df.diff()
        expected_dtype = (
            "float32" if any_int_numpy_dtype in ("int8", "int16") else "float64"
        )
        expected = DataFrame([np.nan, 1.0, 1.0, 1.0, 1.0], dtype=expected_dtype)
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\test_take.py ===
from datetime import datetime

import numpy as np
import pytest

from pandas._libs import iNaT

import pandas._testing as tm
import pandas.core.algorithms as algos


@pytest.fixture(
    params=[
        (np.int8, np.int16(127), np.int8),
        (np.int8, np.int16(128), np.int16),
        (np.int32, 1, np.int32),
        (np.int32, 2.0, np.float64),
        (np.int32, 3.0 + 4.0j, np.complex128),
        (np.int32, True, np.object_),
        (np.int32, "", np.object_),
        (np.float64, 1, np.float64),
        (np.float64, 2.0, np.float64),
        (np.float64, 3.0 + 4.0j, np.complex128),
        (np.float64, True, np.object_),
        (np.float64, "", np.object_),
        (np.complex128, 1, np.complex128),
        (np.complex128, 2.0, np.complex128),
        (np.complex128, 3.0 + 4.0j, np.complex128),
        (np.complex128, True, np.object_),
        (np.complex128, "", np.object_),
        (np.bool_, 1, np.object_),
        (np.bool_, 2.0, np.object_),
        (np.bool_, 3.0 + 4.0j, np.object_),
        (np.bool_, True, np.bool_),
        (np.bool_, "", np.object_),
    ]
)
def dtype_fill_out_dtype(request):
    return request.param


class TestTake:
    def test_1d_fill_nonna(self, dtype_fill_out_dtype):
        dtype, fill_value, out_dtype = dtype_fill_out_dtype
        data = np.random.default_rng(2).integers(0, 2, 4).astype(dtype)
        indexer = [2, 1, 0, -1]

        result = algos.take_nd(data, indexer, fill_value=fill_value)
        assert (result[[0, 1, 2]] == data[[2, 1, 0]]).all()
        assert result[3] == fill_value
        assert result.dtype == out_dtype

        indexer = [2, 1, 0, 1]

        result = algos.take_nd(data, indexer, fill_value=fill_value)
        assert (result[[0, 1, 2, 3]] == data[indexer]).all()
        assert result.dtype == dtype

    def test_2d_fill_nonna(self, dtype_fill_out_dtype):
        dtype, fill_value, out_dtype = dtype_fill_out_dtype
        data = np.random.default_rng(2).integers(0, 2, (5, 3)).astype(dtype)
        indexer = [2, 1, 0, -1]

        result = algos.take_nd(data, indexer, axis=0, fill_value=fill_value)
        assert (result[[0, 1, 2], :] == data[[2, 1, 0], :]).all()
        assert (result[3, :] == fill_value).all()
        assert result.dtype == out_dtype

        result = algos.take_nd(data, indexer, axis=1, fill_value=fill_value)
        assert (result[:, [0, 1, 2]] == data[:, [2, 1, 0]]).all()
        assert (result[:, 3] == fill_value).all()
        assert result.dtype == out_dtype

        indexer = [2, 1, 0, 1]
        result = algos.take_nd(data, indexer, axis=0, fill_value=fill_value)
        assert (result[[0, 1, 2, 3], :] == data[indexer, :]).all()
        assert result.dtype == dtype

        result = algos.take_nd(data, indexer, axis=1, fill_value=fill_value)
        assert (result[:, [0, 1, 2, 3]] == data[:, indexer]).all()
        assert result.dtype == dtype

    def test_3d_fill_nonna(self, dtype_fill_out_dtype):
        dtype, fill_value, out_dtype = dtype_fill_out_dtype

        data = np.random.default_rng(2).integers(0, 2, (5, 4, 3)).astype(dtype)
        indexer = [2, 1, 0, -1]

        result = algos.take_nd(data, indexer, axis=0, fill_value=fill_value)
        assert (result[[0, 1, 2], :, :] == data[[2, 1, 0], :, :]).all()
        assert (result[3, :, :] == fill_value).all()
        assert result.dtype == out_dtype

        result = algos.take_nd(data, indexer, axis=1, fill_value=fill_value)
        assert (result[:, [0, 1, 2], :] == data[:, [2, 1, 0], :]).all()
        assert (result[:, 3, :] == fill_value).all()
        assert result.dtype == out_dtype

        result = algos.take_nd(data, indexer, axis=2, fill_value=fill_value)
        assert (result[:, :, [0, 1, 2]] == data[:, :, [2, 1, 0]]).all()
        assert (result[:, :, 3] == fill_value).all()
        assert result.dtype == out_dtype

        indexer = [2, 1, 0, 1]
        result = algos.take_nd(data, indexer, axis=0, fill_value=fill_value)
        assert (result[[0, 1, 2, 3], :, :] == data[indexer, :, :]).all()
        assert result.dtype == dtype

        result = algos.take_nd(data, indexer, axis=1, fill_value=fill_value)
        assert (result[:, [0, 1, 2, 3], :] == data[:, indexer, :]).all()
        assert result.dtype == dtype

        result = algos.take_nd(data, indexer, axis=2, fill_value=fill_value)
        assert (result[:, :, [0, 1, 2, 3]] == data[:, :, indexer]).all()
        assert result.dtype == dtype

    def test_1d_other_dtypes(self):
        arr = np.random.default_rng(2).standard_normal(10).astype(np.float32)

        indexer = [1, 2, 3, -1]
        result = algos.take_nd(arr, indexer)
        expected = arr.take(indexer)
        expected[-1] = np.nan
        tm.assert_almost_equal(result, expected)

    def test_2d_other_dtypes(self):
        arr = np.random.default_rng(2).standard_normal((10, 5)).astype(np.float32)

        indexer = [1, 2, 3, -1]

        # axis=0
        result = algos.take_nd(arr, indexer, axis=0)
        expected = arr.take(indexer, axis=0)
        expected[-1] = np.nan
        tm.assert_almost_equal(result, expected)

        # axis=1
        result = algos.take_nd(arr, indexer, axis=1)
        expected = arr.take(indexer, axis=1)
        expected[:, -1] = np.nan
        tm.assert_almost_equal(result, expected)

    def test_1d_bool(self):
        arr = np.array([0, 1, 0], dtype=bool)

        result = algos.take_nd(arr, [0, 2, 2, 1])
        expected = arr.take([0, 2, 2, 1])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.take_nd(arr, [0, 2, -1])
        assert result.dtype == np.object_

    def test_2d_bool(self):
        arr = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=bool)

        result = algos.take_nd(arr, [0, 2, 2, 1])
        expected = arr.take([0, 2, 2, 1], axis=0)
        tm.assert_numpy_array_equal(result, expected)

        result = algos.take_nd(arr, [0, 2, 2, 1], axis=1)
        expected = arr.take([0, 2, 2, 1], axis=1)
        tm.assert_numpy_array_equal(result, expected)

        result = algos.take_nd(arr, [0, 2, -1])
        assert result.dtype == np.object_

    def test_2d_float32(self):
        arr = np.random.default_rng(2).standard_normal((4, 3)).astype(np.float32)
        indexer = [0, 2, -1, 1, -1]

        # axis=0
        result = algos.take_nd(arr, indexer, axis=0)

        expected = arr.take(indexer, axis=0)
        expected[[2, 4], :] = np.nan
        tm.assert_almost_equal(result, expected)

        # axis=1
        result = algos.take_nd(arr, indexer, axis=1)
        expected = arr.take(indexer, axis=1)
        expected[:, [2, 4]] = np.nan
        tm.assert_almost_equal(result, expected)

    def test_2d_datetime64(self):
        # 2005/01/01 - 2006/01/01
        arr = (
            np.random.default_rng(2).integers(11_045_376, 11_360_736, (5, 3))
            * 100_000_000_000
        )
        arr = arr.view(dtype="datetime64[ns]")
        indexer = [0, 2, -1, 1, -1]

        # axis=0
        result = algos.take_nd(arr, indexer, axis=0)
        expected = arr.take(indexer, axis=0)
        expected.view(np.int64)[[2, 4], :] = iNaT
        tm.assert_almost_equal(result, expected)

        result = algos.take_nd(arr, indexer, axis=0, fill_value=datetime(2007, 1, 1))
        expected = arr.take(indexer, axis=0)
        expected[[2, 4], :] = datetime(2007, 1, 1)
        tm.assert_almost_equal(result, expected)

        # axis=1
        result = algos.take_nd(arr, indexer, axis=1)
        expected = arr.take(indexer, axis=1)
        expected.view(np.int64)[:, [2, 4]] = iNaT
        tm.assert_almost_equal(result, expected)

        result = algos.take_nd(arr, indexer, axis=1, fill_value=datetime(2007, 1, 1))
        expected = arr.take(indexer, axis=1)
        expected[:, [2, 4]] = datetime(2007, 1, 1)
        tm.assert_almost_equal(result, expected)

    def test_take_axis_0(self):
        arr = np.arange(12).reshape(4, 3)
        result = algos.take(arr, [0, -1])
        expected = np.array([[0, 1, 2], [9, 10, 11]])
        tm.assert_numpy_array_equal(result, expected)

        # allow_fill=True
        result = algos.take(arr, [0, -1], allow_fill=True, fill_value=0)
        expected = np.array([[0, 1, 2], [0, 0, 0]])
        tm.assert_numpy_array_equal(result, expected)

    def test_take_axis_1(self):
        arr = np.arange(12).reshape(4, 3)
        result = algos.take(arr, [0, -1], axis=1)
        expected = np.array([[0, 2], [3, 5], [6, 8], [9, 11]])
        tm.assert_numpy_array_equal(result, expected)

        # allow_fill=True
        result = algos.take(arr, [0, -1], axis=1, allow_fill=True, fill_value=0)
        expected = np.array([[0, 0], [3, 0], [6, 0], [9, 0]])
        tm.assert_numpy_array_equal(result, expected)

        # GH#26976 make sure we validate along the correct axis
        with pytest.raises(IndexError, match="indices are out-of-bounds"):
            algos.take(arr, [0, 3], axis=1, allow_fill=True, fill_value=0)

    def test_take_non_hashable_fill_value(self):
        arr = np.array([1, 2, 3])
        indexer = np.array([1, -1])
        with pytest.raises(ValueError, match="fill_value must be a scalar"):
            algos.take(arr, indexer, allow_fill=True, fill_value=[1])

        # with object dtype it is allowed
        arr = np.array([1, 2, 3], dtype=object)
        result = algos.take(arr, indexer, allow_fill=True, fill_value=[1])
        expected = np.array([2, [1]], dtype=object)
        tm.assert_numpy_array_equal(result, expected)


class TestExtensionTake:
    # The take method found in pd.api.extensions

    def test_bounds_check_large(self):
        arr = np.array([1, 2])

        msg = "indices are out-of-bounds"
        with pytest.raises(IndexError, match=msg):
            algos.take(arr, [2, 3], allow_fill=True)

        msg = "index 2 is out of bounds for( axis 0 with)? size 2"
        with pytest.raises(IndexError, match=msg):
            algos.take(arr, [2, 3], allow_fill=False)

    def test_bounds_check_small(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        indexer = [0, -1, -2]

        msg = r"'indices' contains values less than allowed \(-2 < -1\)"
        with pytest.raises(ValueError, match=msg):
            algos.take(arr, indexer, allow_fill=True)

        result = algos.take(arr, indexer)
        expected = np.array([1, 3, 2], dtype=np.int64)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("allow_fill", [True, False])
    def test_take_empty(self, allow_fill):
        arr = np.array([], dtype=np.int64)
        # empty take is ok
        result = algos.take(arr, [], allow_fill=allow_fill)
        tm.assert_numpy_array_equal(arr, result)

        msg = "|".join(
            [
                "cannot do a non-empty take from an empty axes.",
                "indices are out-of-bounds",
            ]
        )
        with pytest.raises(IndexError, match=msg):
            algos.take(arr, [0], allow_fill=allow_fill)

    def test_take_na_empty(self):
        result = algos.take(np.array([]), [-1, -1], allow_fill=True, fill_value=0.0)
        expected = np.array([0.0, 0.0])
        tm.assert_numpy_array_equal(result, expected)

    def test_take_coerces_list(self):
        arr = [1, 2, 3]
        msg = "take accepting non-standard inputs is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = algos.take(arr, [0, 0])
        expected = np.array([1, 1])
        tm.assert_numpy_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\external\tests\test_ntheory.py ===
from itertools import permutations

from sympy.external.ntheory import (bit_scan1, remove, bit_scan0, is_fermat_prp,
                                    is_euler_prp, is_strong_prp, gcdext, _lucas_sequence,
                                    is_fibonacci_prp, is_lucas_prp, is_selfridge_prp,
                                    is_strong_lucas_prp, is_strong_selfridge_prp,
                                    is_bpsw_prp, is_strong_bpsw_prp)
from sympy.testing.pytest import raises


def test_bit_scan1():
    assert bit_scan1(0) is None
    assert bit_scan1(1) == 0
    assert bit_scan1(-1) == 0
    assert bit_scan1(2) == 1
    assert bit_scan1(7) == 0
    assert bit_scan1(-7) == 0
    for i in range(100):
        assert bit_scan1(1 << i) == i
        assert bit_scan1((1 << i) * 31337) == i
    for i in range(500):
        n = (1 << 500) + (1 << i)
        assert bit_scan1(n) == i
    assert bit_scan1(1 << 1000001) == 1000001
    assert bit_scan1((1 << 273956)*7**37) == 273956
    # issue 12709
    for i in range(1, 10):
        big = 1 << i
        assert bit_scan1(-big) == bit_scan1(big)


def test_bit_scan0():
    assert bit_scan0(-1) is None
    assert bit_scan0(0) == 0
    assert bit_scan0(1) == 1
    assert bit_scan0(-2) == 0


def test_remove():
    raises(ValueError, lambda: remove(1, 1))
    assert remove(0, 3) == (0, 0)
    for f in range(2, 10):
        for y in range(2, 1000):
            for z in [1, 17, 101, 1009]:
                assert remove(z*f**y, f) == (z, y)


def test_gcdext():
    assert gcdext(0, 0) == (0, 0, 0)
    assert gcdext(3, 0) == (3, 1, 0)
    assert gcdext(0, 4) == (4, 0, 1)

    for n in range(1, 10):
        assert gcdext(n, 1) == gcdext(-n, 1) == (1, 0, 1)
        assert gcdext(n, -1) == gcdext(-n, -1) == (1, 0, -1)
        assert gcdext(n, n) == gcdext(-n, n) == (n, 0, 1)
        assert gcdext(n, -n) == gcdext(-n, -n) == (n, 0, -1)

    for n in range(2, 10):
        assert gcdext(1, n) == gcdext(1, -n) == (1, 1, 0)
        assert gcdext(-1, n) == gcdext(-1, -n) == (1, -1, 0)

    for a, b in permutations([2**5, 3, 5, 7**2, 11], 2):
        g, x, y = gcdext(a, b)
        assert g == a*x + b*y == 1


def test_is_fermat_prp():
    # invalid input
    raises(ValueError, lambda: is_fermat_prp(0, 10))
    raises(ValueError, lambda: is_fermat_prp(5, 1))

    # n = 1
    assert not is_fermat_prp(1, 3)

    # n is prime
    assert is_fermat_prp(2, 4)
    assert is_fermat_prp(3, 2)
    assert is_fermat_prp(11, 3)
    assert is_fermat_prp(2**31-1, 5)

    # A001567
    pseudorpime = [341, 561, 645, 1105, 1387, 1729, 1905, 2047,
                   2465, 2701, 2821, 3277, 4033, 4369, 4371, 4681]
    for n in pseudorpime:
        assert is_fermat_prp(n, 2)

    # A020136
    pseudorpime = [15, 85, 91, 341, 435, 451, 561, 645, 703, 1105,
                   1247, 1271, 1387, 1581, 1695, 1729, 1891, 1905]
    for n in pseudorpime:
        assert is_fermat_prp(n, 4)


def test_is_euler_prp():
    # invalid input
    raises(ValueError, lambda: is_euler_prp(0, 10))
    raises(ValueError, lambda: is_euler_prp(5, 1))

    # n = 1
    assert not is_euler_prp(1, 3)

    # n is prime
    assert is_euler_prp(2, 4)
    assert is_euler_prp(3, 2)
    assert is_euler_prp(11, 3)
    assert is_euler_prp(2**31-1, 5)

    # A047713
    pseudorpime = [561, 1105, 1729, 1905, 2047, 2465, 3277, 4033,
                   4681, 6601, 8321, 8481, 10585, 12801, 15841]
    for n in pseudorpime:
        assert is_euler_prp(n, 2)

    # A048950
    pseudorpime = [121, 703, 1729, 1891, 2821, 3281, 7381, 8401,
                   8911, 10585, 12403, 15457, 15841, 16531, 18721]
    for n in pseudorpime:
        assert is_euler_prp(n, 3)


def test_is_strong_prp():
    # invalid input
    raises(ValueError, lambda: is_strong_prp(0, 10))
    raises(ValueError, lambda: is_strong_prp(5, 1))

    # n = 1
    assert not is_strong_prp(1, 3)

    # n is prime
    assert is_strong_prp(2, 4)
    assert is_strong_prp(3, 2)
    assert is_strong_prp(11, 3)
    assert is_strong_prp(2**31-1, 5)

    # A001262
    pseudorpime = [2047, 3277, 4033, 4681, 8321, 15841, 29341,
                   42799, 49141, 52633, 65281, 74665, 80581]
    for n in pseudorpime:
        assert is_strong_prp(n, 2)

    # A020229
    pseudorpime = [121, 703, 1891, 3281, 8401, 8911, 10585, 12403,
                   16531, 18721, 19345, 23521, 31621, 44287, 47197]
    for n in pseudorpime:
        assert is_strong_prp(n, 3)


def test_lucas_sequence():
    def lucas_u(P, Q, length):
        array = [0] * length
        array[1] = 1
        for k in range(2, length):
            array[k] = P * array[k - 1] - Q * array[k - 2]
        return array

    def lucas_v(P, Q, length):
        array = [0] * length
        array[0] = 2
        array[1] = P
        for k in range(2, length):
            array[k] = P * array[k - 1] - Q * array[k - 2]
        return array

    length = 20
    for P in range(-10, 10):
        for Q in range(-10, 10):
            D = P**2 - 4*Q
            if D == 0:
                continue
            us = lucas_u(P, Q, length)
            vs = lucas_v(P, Q, length)
            for n in range(3, 100, 2):
                for k in range(length):
                    U, V, Qk = _lucas_sequence(n, P, Q, k)
                    assert U == us[k] % n
                    assert V == vs[k] % n
                    assert pow(Q, k, n) == Qk


def test_is_fibonacci_prp():
    # invalid input
    raises(ValueError, lambda: is_fibonacci_prp(3, 2, 1))
    raises(ValueError, lambda: is_fibonacci_prp(3, -5, 1))
    raises(ValueError, lambda: is_fibonacci_prp(3, 5, 2))
    raises(ValueError, lambda: is_fibonacci_prp(0, 5, -1))

    # n = 1
    assert not is_fibonacci_prp(1, 3, 1)

    # n is prime
    assert is_fibonacci_prp(2, 5, 1)
    assert is_fibonacci_prp(3, 6, -1)
    assert is_fibonacci_prp(11, 7, 1)
    assert is_fibonacci_prp(2**31-1, 8, -1)

    # A005845
    pseudorpime = [705, 2465, 2737, 3745, 4181, 5777, 6721,
                   10877, 13201, 15251, 24465, 29281, 34561]
    for n in pseudorpime:
        assert is_fibonacci_prp(n, 1, -1)


def test_is_lucas_prp():
    # invalid input
    raises(ValueError, lambda: is_lucas_prp(3, 2, 1))
    raises(ValueError, lambda: is_lucas_prp(0, 5, -1))
    raises(ValueError, lambda: is_lucas_prp(15, 3, 1))

    # n = 1
    assert not is_lucas_prp(1, 3, 1)

    # n is prime
    assert is_lucas_prp(2, 5, 2)
    assert is_lucas_prp(3, 6, -1)
    assert is_lucas_prp(11, 7, 5)
    assert is_lucas_prp(2**31-1, 8, -3)

    # A081264
    pseudorpime = [323, 377, 1891, 3827, 4181, 5777, 6601, 6721,
                   8149, 10877, 11663, 13201, 13981, 15251, 17119]
    for n in pseudorpime:
        assert is_lucas_prp(n, 1, -1)


def test_is_selfridge_prp():
    # invalid input
    raises(ValueError, lambda: is_selfridge_prp(0))

    # n = 1
    assert not is_selfridge_prp(1)

    # n is prime
    assert is_selfridge_prp(2)
    assert is_selfridge_prp(3)
    assert is_selfridge_prp(11)
    assert is_selfridge_prp(2**31-1)

    # A217120
    pseudorpime = [323, 377, 1159, 1829, 3827, 5459, 5777, 9071,
                   9179, 10877, 11419, 11663, 13919, 14839, 16109]
    for n in pseudorpime:
        assert is_selfridge_prp(n)


def test_is_strong_lucas_prp():
    # invalid input
    raises(ValueError, lambda: is_strong_lucas_prp(3, 2, 1))
    raises(ValueError, lambda: is_strong_lucas_prp(0, 5, -1))
    raises(ValueError, lambda: is_strong_lucas_prp(15, 3, 1))

    # n = 1
    assert not is_strong_lucas_prp(1, 3, 1)

    # n is prime
    assert is_strong_lucas_prp(2, 5, 2)
    assert is_strong_lucas_prp(3, 6, -1)
    assert is_strong_lucas_prp(11, 7, 5)
    assert is_strong_lucas_prp(2**31-1, 8, -3)


def test_is_strong_selfridge_prp():
    # invalid input
    raises(ValueError, lambda: is_strong_selfridge_prp(0))

    # n = 1
    assert not is_strong_selfridge_prp(1)

    # n is prime
    assert is_strong_selfridge_prp(2)
    assert is_strong_selfridge_prp(3)
    assert is_strong_selfridge_prp(11)
    assert is_strong_selfridge_prp(2**31-1)

    # A217255
    pseudorpime = [5459, 5777, 10877, 16109, 18971, 22499, 24569,
                   25199, 40309, 58519, 75077, 97439, 100127, 113573]
    for n in pseudorpime:
        assert is_strong_selfridge_prp(n)


def test_is_bpsw_prp():
    # invalid input
    raises(ValueError, lambda: is_bpsw_prp(0))

    # n = 1
    assert not is_bpsw_prp(1)

    # n is prime
    assert is_bpsw_prp(2)
    assert is_bpsw_prp(3)
    assert is_bpsw_prp(11)
    assert is_bpsw_prp(2**31-1)


def test_is_strong_bpsw_prp():
    # invalid input
    raises(ValueError, lambda: is_strong_bpsw_prp(0))

    # n = 1
    assert not is_strong_bpsw_prp(1)

    # n is prime
    assert is_strong_bpsw_prp(2)
    assert is_strong_bpsw_prp(3)
    assert is_strong_bpsw_prp(11)
    assert is_strong_bpsw_prp(2**31-1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arithmetic\test_interval.py ===
import operator

import numpy as np
import pytest

from pandas.core.dtypes.common import is_list_like

import pandas as pd
from pandas import (
    Categorical,
    Index,
    Interval,
    IntervalIndex,
    Period,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.arrays import (
    BooleanArray,
    IntervalArray,
)
from pandas.tests.arithmetic.common import get_upcast_box


@pytest.fixture(
    params=[
        (Index([0, 2, 4, 4]), Index([1, 3, 5, 8])),
        (Index([0.0, 1.0, 2.0, np.nan]), Index([1.0, 2.0, 3.0, np.nan])),
        (
            timedelta_range("0 days", periods=3).insert(3, pd.NaT),
            timedelta_range("1 day", periods=3).insert(3, pd.NaT),
        ),
        (
            date_range("20170101", periods=3).insert(3, pd.NaT),
            date_range("20170102", periods=3).insert(3, pd.NaT),
        ),
        (
            date_range("20170101", periods=3, tz="US/Eastern").insert(3, pd.NaT),
            date_range("20170102", periods=3, tz="US/Eastern").insert(3, pd.NaT),
        ),
    ],
    ids=lambda x: str(x[0].dtype),
)
def left_right_dtypes(request):
    """
    Fixture for building an IntervalArray from various dtypes
    """
    return request.param


@pytest.fixture
def interval_array(left_right_dtypes):
    """
    Fixture to generate an IntervalArray of various dtypes containing NA if possible
    """
    left, right = left_right_dtypes
    return IntervalArray.from_arrays(left, right)


def create_categorical_intervals(left, right, closed="right"):
    return Categorical(IntervalIndex.from_arrays(left, right, closed))


def create_series_intervals(left, right, closed="right"):
    return Series(IntervalArray.from_arrays(left, right, closed))


def create_series_categorical_intervals(left, right, closed="right"):
    return Series(Categorical(IntervalIndex.from_arrays(left, right, closed)))


class TestComparison:
    @pytest.fixture(params=[operator.eq, operator.ne])
    def op(self, request):
        return request.param

    @pytest.fixture(
        params=[
            IntervalArray.from_arrays,
            IntervalIndex.from_arrays,
            create_categorical_intervals,
            create_series_intervals,
            create_series_categorical_intervals,
        ],
        ids=[
            "IntervalArray",
            "IntervalIndex",
            "Categorical[Interval]",
            "Series[Interval]",
            "Series[Categorical[Interval]]",
        ],
    )
    def interval_constructor(self, request):
        """
        Fixture for all pandas native interval constructors.
        To be used as the LHS of IntervalArray comparisons.
        """
        return request.param

    def elementwise_comparison(self, op, interval_array, other):
        """
        Helper that performs elementwise comparisons between `array` and `other`
        """
        other = other if is_list_like(other) else [other] * len(interval_array)
        expected = np.array([op(x, y) for x, y in zip(interval_array, other)])
        if isinstance(other, Series):
            return Series(expected, index=other.index)
        return expected

    def test_compare_scalar_interval(self, op, interval_array):
        # matches first interval
        other = interval_array[0]
        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        tm.assert_numpy_array_equal(result, expected)

        # matches on a single endpoint but not both
        other = Interval(interval_array.left[0], interval_array.right[1])
        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        tm.assert_numpy_array_equal(result, expected)

    def test_compare_scalar_interval_mixed_closed(self, op, closed, other_closed):
        interval_array = IntervalArray.from_arrays(range(2), range(1, 3), closed=closed)
        other = Interval(0, 1, closed=other_closed)

        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        tm.assert_numpy_array_equal(result, expected)

    def test_compare_scalar_na(self, op, interval_array, nulls_fixture, box_with_array):
        box = box_with_array
        obj = tm.box_expected(interval_array, box)
        result = op(obj, nulls_fixture)

        if nulls_fixture is pd.NA:
            # GH#31882
            exp = np.ones(interval_array.shape, dtype=bool)
            expected = BooleanArray(exp, exp)
        else:
            expected = self.elementwise_comparison(op, interval_array, nulls_fixture)

        if not (box is Index and nulls_fixture is pd.NA):
            # don't cast expected from BooleanArray to ndarray[object]
            xbox = get_upcast_box(obj, nulls_fixture, True)
            expected = tm.box_expected(expected, xbox)

        tm.assert_equal(result, expected)

        rev = op(nulls_fixture, obj)
        tm.assert_equal(rev, expected)

    @pytest.mark.parametrize(
        "other",
        [
            0,
            1.0,
            True,
            "foo",
            Timestamp("2017-01-01"),
            Timestamp("2017-01-01", tz="US/Eastern"),
            Timedelta("0 days"),
            Period("2017-01-01", "D"),
        ],
    )
    def test_compare_scalar_other(self, op, interval_array, other):
        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        tm.assert_numpy_array_equal(result, expected)

    def test_compare_list_like_interval(self, op, interval_array, interval_constructor):
        # same endpoints
        other = interval_constructor(interval_array.left, interval_array.right)
        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        tm.assert_equal(result, expected)

        # different endpoints
        other = interval_constructor(
            interval_array.left[::-1], interval_array.right[::-1]
        )
        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        tm.assert_equal(result, expected)

        # all nan endpoints
        other = interval_constructor([np.nan] * 4, [np.nan] * 4)
        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        tm.assert_equal(result, expected)

    def test_compare_list_like_interval_mixed_closed(
        self, op, interval_constructor, closed, other_closed
    ):
        interval_array = IntervalArray.from_arrays(range(2), range(1, 3), closed=closed)
        other = interval_constructor(range(2), range(1, 3), closed=other_closed)

        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "other",
        [
            (
                Interval(0, 1),
                Interval(Timedelta("1 day"), Timedelta("2 days")),
                Interval(4, 5, "both"),
                Interval(10, 20, "neither"),
            ),
            (0, 1.5, Timestamp("20170103"), np.nan),
            (
                Timestamp("20170102", tz="US/Eastern"),
                Timedelta("2 days"),
                "baz",
                pd.NaT,
            ),
        ],
    )
    def test_compare_list_like_object(self, op, interval_array, other):
        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        tm.assert_numpy_array_equal(result, expected)

    def test_compare_list_like_nan(self, op, interval_array, nulls_fixture):
        other = [nulls_fixture] * 4
        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)

        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "other",
        [
            np.arange(4, dtype="int64"),
            np.arange(4, dtype="float64"),
            date_range("2017-01-01", periods=4),
            date_range("2017-01-01", periods=4, tz="US/Eastern"),
            timedelta_range("0 days", periods=4),
            period_range("2017-01-01", periods=4, freq="D"),
            Categorical(list("abab")),
            Categorical(date_range("2017-01-01", periods=4)),
            pd.array(list("abcd")),
            pd.array(["foo", 3.14, None, object()], dtype=object),
        ],
        ids=lambda x: str(x.dtype),
    )
    def test_compare_list_like_other(self, op, interval_array, other):
        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("length", [1, 3, 5])
    @pytest.mark.parametrize("other_constructor", [IntervalArray, list])
    def test_compare_length_mismatch_errors(self, op, other_constructor, length):
        interval_array = IntervalArray.from_arrays(range(4), range(1, 5))
        other = other_constructor([Interval(0, 1)] * length)
        with pytest.raises(ValueError, match="Lengths must match to compare"):
            op(interval_array, other)

    @pytest.mark.parametrize(
        "constructor, expected_type, assert_func",
        [
            (IntervalIndex, np.array, tm.assert_numpy_array_equal),
            (Series, Series, tm.assert_series_equal),
        ],
    )
    def test_index_series_compat(self, op, constructor, expected_type, assert_func):
        # IntervalIndex/Series that rely on IntervalArray for comparisons
        breaks = range(4)
        index = constructor(IntervalIndex.from_breaks(breaks))

        # scalar comparisons
        other = index[0]
        result = op(index, other)
        expected = expected_type(self.elementwise_comparison(op, index, other))
        assert_func(result, expected)

        other = breaks[0]
        result = op(index, other)
        expected = expected_type(self.elementwise_comparison(op, index, other))
        assert_func(result, expected)

        # list-like comparisons
        other = IntervalArray.from_breaks(breaks)
        result = op(index, other)
        expected = expected_type(self.elementwise_comparison(op, index, other))
        assert_func(result, expected)

        other = [index[0], breaks[0], "foo"]
        result = op(index, other)
        expected = expected_type(self.elementwise_comparison(op, index, other))
        assert_func(result, expected)

    @pytest.mark.parametrize("scalars", ["a", False, 1, 1.0, None])
    def test_comparison_operations(self, scalars):
        # GH #28981
        expected = Series([False, False])
        s = Series([Interval(0, 1), Interval(1, 2)], dtype="interval")
        result = s == scalars
        tm.assert_series_equal(result, expected)