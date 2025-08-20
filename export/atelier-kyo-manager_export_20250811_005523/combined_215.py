
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\pytables\test_put.py ===
import re

import numpy as np
import pytest

from pandas._libs.tslibs import Timestamp

import pandas as pd
from pandas import (
    DataFrame,
    HDFStore,
    Index,
    MultiIndex,
    Series,
    _testing as tm,
    concat,
    date_range,
)
from pandas.tests.io.pytables.common import (
    _maybe_remove,
    ensure_clean_store,
)
from pandas.util import _test_decorators as td

pytestmark = [pytest.mark.single_cpu]


def test_format_type(tmp_path, setup_path):
    df = DataFrame({"A": [1, 2]})
    with HDFStore(tmp_path / setup_path) as store:
        store.put("a", df, format="fixed")
        store.put("b", df, format="table")

        assert store.get_storer("a").format_type == "fixed"
        assert store.get_storer("b").format_type == "table"


def test_format_kwarg_in_constructor(tmp_path, setup_path):
    # GH 13291

    msg = "format is not a defined argument for HDFStore"

    with pytest.raises(ValueError, match=msg):
        HDFStore(tmp_path / setup_path, format="table")


def test_api_default_format(tmp_path, setup_path):
    # default_format option
    with ensure_clean_store(setup_path) as store:
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD")),
            index=Index([f"i-{i}" for i in range(30)]),
        )

        with pd.option_context("io.hdf.default_format", "fixed"):
            _maybe_remove(store, "df")
            store.put("df", df)
            assert not store.get_storer("df").is_table

            msg = "Can only append to Tables"
            with pytest.raises(ValueError, match=msg):
                store.append("df2", df)

        with pd.option_context("io.hdf.default_format", "table"):
            _maybe_remove(store, "df")
            store.put("df", df)
            assert store.get_storer("df").is_table

            _maybe_remove(store, "df2")
            store.append("df2", df)
            assert store.get_storer("df").is_table

    path = tmp_path / setup_path
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )

    with pd.option_context("io.hdf.default_format", "fixed"):
        df.to_hdf(path, key="df")
        with HDFStore(path) as store:
            assert not store.get_storer("df").is_table
        with pytest.raises(ValueError, match=msg):
            df.to_hdf(path, key="df2", append=True)

    with pd.option_context("io.hdf.default_format", "table"):
        df.to_hdf(path, key="df3")
        with HDFStore(path) as store:
            assert store.get_storer("df3").is_table
        df.to_hdf(path, key="df4", append=True)
        with HDFStore(path) as store:
            assert store.get_storer("df4").is_table


def test_put(setup_path):
    with ensure_clean_store(setup_path) as store:
        ts = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        df = DataFrame(
            np.random.default_rng(2).standard_normal((20, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=20, freq="B"),
        )
        store["a"] = ts
        store["b"] = df[:10]
        store["foo/bar/bah"] = df[:10]
        store["foo"] = df[:10]
        store["/foo"] = df[:10]
        store.put("c", df[:10], format="table")

        # not OK, not a table
        msg = "Can only append to Tables"
        with pytest.raises(ValueError, match=msg):
            store.put("b", df[10:], append=True)

        # node does not currently exist, test _is_table_type returns False
        # in this case
        _maybe_remove(store, "f")
        with pytest.raises(ValueError, match=msg):
            store.put("f", df[10:], append=True)

        # can't put to a table (use append instead)
        with pytest.raises(ValueError, match=msg):
            store.put("c", df[10:], append=True)

        # overwrite table
        store.put("c", df[:10], format="table", append=False)
        tm.assert_frame_equal(df[:10], store["c"])


def test_put_string_index(setup_path):
    with ensure_clean_store(setup_path) as store:
        index = Index([f"I am a very long string index: {i}" for i in range(20)])
        s = Series(np.arange(20), index=index)
        df = DataFrame({"A": s, "B": s})

        store["a"] = s
        tm.assert_series_equal(store["a"], s)

        store["b"] = df
        tm.assert_frame_equal(store["b"], df)

        # mixed length
        index = Index(
            ["abcdefghijklmnopqrstuvwxyz1234567890"]
            + [f"I am a very long string index: {i}" for i in range(20)]
        )
        s = Series(np.arange(21), index=index)
        df = DataFrame({"A": s, "B": s})
        store["a"] = s
        tm.assert_series_equal(store["a"], s)

        store["b"] = df
        tm.assert_frame_equal(store["b"], df)


def test_put_compression(setup_path):
    with ensure_clean_store(setup_path) as store:
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )

        store.put("c", df, format="table", complib="zlib")
        tm.assert_frame_equal(store["c"], df)

        # can't compress if format='fixed'
        msg = "Compression not supported on Fixed format stores"
        with pytest.raises(ValueError, match=msg):
            store.put("b", df, format="fixed", complib="zlib")


@td.skip_if_windows
def test_put_compression_blosc(setup_path):
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )

    with ensure_clean_store(setup_path) as store:
        # can't compress if format='fixed'
        msg = "Compression not supported on Fixed format stores"
        with pytest.raises(ValueError, match=msg):
            store.put("b", df, format="fixed", complib="blosc")

        store.put("c", df, format="table", complib="blosc")
        tm.assert_frame_equal(store["c"], df)


def test_put_datetime_ser(setup_path):
    # https://github.com/pandas-dev/pandas/pull/60663
    ser = Series(3 * [Timestamp("20010102").as_unit("ns")])
    with ensure_clean_store(setup_path) as store:
        store.put("ser", ser)
        expected = ser.copy()
        result = store.get("ser")
        tm.assert_series_equal(result, expected)


def test_put_mixed_type(setup_path, using_infer_string):
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    df["obj1"] = "foo"
    df["obj2"] = "bar"
    df["bool1"] = df["A"] > 0
    df["bool2"] = df["B"] > 0
    df["bool3"] = True
    df["int1"] = 1
    df["int2"] = 2
    df["timestamp1"] = Timestamp("20010102").as_unit("ns")
    df["timestamp2"] = Timestamp("20010103").as_unit("ns")
    df["datetime1"] = Timestamp("20010102").as_unit("ns")
    df["datetime2"] = Timestamp("20010103").as_unit("ns")
    df.loc[df.index[3:6], ["obj1"]] = np.nan
    df = df._consolidate()

    with ensure_clean_store(setup_path) as store:
        _maybe_remove(store, "df")

        warning = None if using_infer_string else pd.errors.PerformanceWarning
        with tm.assert_produces_warning(warning):
            store.put("df", df)

        expected = store.get("df")
        tm.assert_frame_equal(expected, df)


def test_put_str_frame(setup_path, string_dtype_arguments):
    # https://github.com/pandas-dev/pandas/pull/60663
    dtype = pd.StringDtype(*string_dtype_arguments)
    df = DataFrame({"a": pd.array(["x", pd.NA, "y"], dtype=dtype)})
    with ensure_clean_store(setup_path) as store:
        _maybe_remove(store, "df")

        store.put("df", df)
        expected_dtype = "str" if dtype.na_value is np.nan else "string"
        expected = df.astype(expected_dtype)
        result = store.get("df")
        tm.assert_frame_equal(result, expected)


def test_put_str_series(setup_path, string_dtype_arguments):
    # https://github.com/pandas-dev/pandas/pull/60663
    dtype = pd.StringDtype(*string_dtype_arguments)
    ser = Series(["x", pd.NA, "y"], dtype=dtype)
    with ensure_clean_store(setup_path) as store:
        _maybe_remove(store, "df")

        store.put("ser", ser)
        expected_dtype = "str" if dtype.na_value is np.nan else "string"
        expected = ser.astype(expected_dtype)
        result = store.get("ser")
        tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("format", ["table", "fixed"])
@pytest.mark.parametrize(
    "index",
    [
        Index([str(i) for i in range(10)]),
        Index(np.arange(10, dtype=float)),
        Index(np.arange(10)),
        date_range("2020-01-01", periods=10),
        pd.period_range("2020-01-01", periods=10),
    ],
)
def test_store_index_types(setup_path, format, index):
    # GH5386
    # test storing various index types

    with ensure_clean_store(setup_path) as store:
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)),
            columns=list("AB"),
            index=index,
        )
        _maybe_remove(store, "df")
        store.put("df", df, format=format)
        tm.assert_frame_equal(df, store["df"])


def test_column_multiindex(setup_path, using_infer_string):
    # GH 4710
    # recreate multi-indexes properly

    index = MultiIndex.from_tuples(
        [("A", "a"), ("A", "b"), ("B", "a"), ("B", "b")], names=["first", "second"]
    )
    df = DataFrame(np.arange(12).reshape(3, 4), columns=index)
    expected = df.set_axis(df.index.to_numpy())

    with ensure_clean_store(setup_path) as store:
        if using_infer_string:
            # TODO(infer_string) make this work for string dtype
            msg = "Saving a MultiIndex with an extension dtype is not supported."
            with pytest.raises(NotImplementedError, match=msg):
                store.put("df", df)
            return
        store.put("df", df)
        tm.assert_frame_equal(
            store["df"], expected, check_index_type=True, check_column_type=True
        )

        store.put("df1", df, format="table")
        tm.assert_frame_equal(
            store["df1"], expected, check_index_type=True, check_column_type=True
        )

        msg = re.escape("cannot use a multi-index on axis [1] with data_columns ['A']")
        with pytest.raises(ValueError, match=msg):
            store.put("df2", df, format="table", data_columns=["A"])
        msg = re.escape("cannot use a multi-index on axis [1] with data_columns True")
        with pytest.raises(ValueError, match=msg):
            store.put("df3", df, format="table", data_columns=True)

    # appending multi-column on existing table (see GH 6167)
    with ensure_clean_store(setup_path) as store:
        store.append("df2", df)
        store.append("df2", df)

        tm.assert_frame_equal(store["df2"], concat((df, df)))

    # non_index_axes name
    df = DataFrame(np.arange(12).reshape(3, 4), columns=Index(list("ABCD"), name="foo"))
    expected = df.set_axis(df.index.to_numpy())

    with ensure_clean_store(setup_path) as store:
        store.put("df1", df, format="table")
        tm.assert_frame_equal(
            store["df1"], expected, check_index_type=True, check_column_type=True
        )


def test_store_multiindex(setup_path):
    # validate multi-index names
    # GH 5527
    with ensure_clean_store(setup_path) as store:

        def make_index(names=None):
            dti = date_range("2013-12-01", "2013-12-02")
            mi = MultiIndex.from_product([dti, range(2), range(3)], names=names)
            return mi

        # no names
        _maybe_remove(store, "df")
        df = DataFrame(np.zeros((12, 2)), columns=["a", "b"], index=make_index())
        store.append("df", df)
        tm.assert_frame_equal(store.select("df"), df)

        # partial names
        _maybe_remove(store, "df")
        df = DataFrame(
            np.zeros((12, 2)),
            columns=["a", "b"],
            index=make_index(["date", None, None]),
        )
        store.append("df", df)
        tm.assert_frame_equal(store.select("df"), df)

        # series
        _maybe_remove(store, "ser")
        ser = Series(np.zeros(12), index=make_index(["date", None, None]))
        store.append("ser", ser)
        xp = Series(np.zeros(12), index=make_index(["date", "level_1", "level_2"]))
        tm.assert_series_equal(store.select("ser"), xp)

        # dup with column
        _maybe_remove(store, "df")
        df = DataFrame(
            np.zeros((12, 2)),
            columns=["a", "b"],
            index=make_index(["date", "a", "t"]),
        )
        msg = "duplicate names/columns in the multi-index when storing as a table"
        with pytest.raises(ValueError, match=msg):
            store.append("df", df)

        # dup within level
        _maybe_remove(store, "df")
        df = DataFrame(
            np.zeros((12, 2)),
            columns=["a", "b"],
            index=make_index(["date", "date", "date"]),
        )
        with pytest.raises(ValueError, match=msg):
            store.append("df", df)

        # fully names
        _maybe_remove(store, "df")
        df = DataFrame(
            np.zeros((12, 2)),
            columns=["a", "b"],
            index=make_index(["date", "s", "t"]),
        )
        store.append("df", df)
        tm.assert_frame_equal(store.select("df"), df)


@pytest.mark.parametrize("format", ["fixed", "table"])
def test_store_periodindex(tmp_path, setup_path, format):
    # GH 7796
    # test of PeriodIndex in HDFStore
    df = DataFrame(
        np.random.default_rng(2).standard_normal((5, 1)),
        index=pd.period_range("20220101", freq="M", periods=5),
    )

    path = tmp_path / setup_path
    df.to_hdf(path, key="df", mode="w", format=format)
    expected = pd.read_hdf(path, "df")
    tm.assert_frame_equal(df, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\test_masked.py ===
"""
This file contains a minimal set of tests for compliance with the extension
array interface test suite, and should contain no other tests.
The test suite for the full functionality of the array is located in
`pandas/tests/arrays/`.

The tests in this file are inherited from the BaseExtensionTests, and only
minimal tweaks should be applied to get the tests passing (by overwriting a
parent method).

Additional tests should either be added to one of the BaseExtensionTests
classes (if they are relevant for the extension interface for all dtypes), or
be added to the array-specific tests in `pandas/tests/arrays/`.

"""
import warnings

import numpy as np
import pytest

from pandas.compat import (
    IS64,
    is_platform_windows,
)
from pandas.compat.numpy import np_version_gt2

from pandas.core.dtypes.common import (
    is_float_dtype,
    is_signed_integer_dtype,
    is_unsigned_integer_dtype,
)

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.arrays.floating import (
    Float32Dtype,
    Float64Dtype,
)
from pandas.core.arrays.integer import (
    Int8Dtype,
    Int16Dtype,
    Int32Dtype,
    Int64Dtype,
    UInt8Dtype,
    UInt16Dtype,
    UInt32Dtype,
    UInt64Dtype,
)
from pandas.tests.extension import base

is_windows_or_32bit = (is_platform_windows() and not np_version_gt2) or not IS64

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:invalid value encountered in divide:RuntimeWarning"
    ),
    pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning"),
    # overflow only relevant for Floating dtype cases cases
    pytest.mark.filterwarnings("ignore:overflow encountered in reduce:RuntimeWarning"),
]


def make_data():
    return list(range(1, 9)) + [pd.NA] + list(range(10, 98)) + [pd.NA] + [99, 100]


def make_float_data():
    return (
        list(np.arange(0.1, 0.9, 0.1))
        + [pd.NA]
        + list(np.arange(1, 9.8, 0.1))
        + [pd.NA]
        + [9.9, 10.0]
    )


def make_bool_data():
    return [True, False] * 4 + [np.nan] + [True, False] * 44 + [np.nan] + [True, False]


@pytest.fixture(
    params=[
        Int8Dtype,
        Int16Dtype,
        Int32Dtype,
        Int64Dtype,
        UInt8Dtype,
        UInt16Dtype,
        UInt32Dtype,
        UInt64Dtype,
        Float32Dtype,
        Float64Dtype,
        BooleanDtype,
    ]
)
def dtype(request):
    return request.param()


@pytest.fixture
def data(dtype):
    if dtype.kind == "f":
        data = make_float_data()
    elif dtype.kind == "b":
        data = make_bool_data()
    else:
        data = make_data()
    return pd.array(data, dtype=dtype)


@pytest.fixture
def data_for_twos(dtype):
    if dtype.kind == "b":
        return pd.array(np.ones(100), dtype=dtype)
    return pd.array(np.ones(100) * 2, dtype=dtype)


@pytest.fixture
def data_missing(dtype):
    if dtype.kind == "f":
        return pd.array([pd.NA, 0.1], dtype=dtype)
    elif dtype.kind == "b":
        return pd.array([np.nan, True], dtype=dtype)
    return pd.array([pd.NA, 1], dtype=dtype)


@pytest.fixture
def data_for_sorting(dtype):
    if dtype.kind == "f":
        return pd.array([0.1, 0.2, 0.0], dtype=dtype)
    elif dtype.kind == "b":
        return pd.array([True, True, False], dtype=dtype)
    return pd.array([1, 2, 0], dtype=dtype)


@pytest.fixture
def data_missing_for_sorting(dtype):
    if dtype.kind == "f":
        return pd.array([0.1, pd.NA, 0.0], dtype=dtype)
    elif dtype.kind == "b":
        return pd.array([True, np.nan, False], dtype=dtype)
    return pd.array([1, pd.NA, 0], dtype=dtype)


@pytest.fixture
def na_cmp():
    # we are pd.NA
    return lambda x, y: x is pd.NA and y is pd.NA


@pytest.fixture
def data_for_grouping(dtype):
    if dtype.kind == "f":
        b = 0.1
        a = 0.0
        c = 0.2
    elif dtype.kind == "b":
        b = True
        a = False
        c = b
    else:
        b = 1
        a = 0
        c = 2

    na = pd.NA
    return pd.array([b, b, na, na, a, a, b, c], dtype=dtype)


class TestMaskedArrays(base.ExtensionTests):
    @pytest.mark.parametrize("na_action", [None, "ignore"])
    def test_map(self, data_missing, na_action):
        result = data_missing.map(lambda x: x, na_action=na_action)
        if data_missing.dtype == Float32Dtype():
            # map roundtrips through objects, which converts to float64
            expected = data_missing.to_numpy(dtype="float64", na_value=np.nan)
        else:
            expected = data_missing.to_numpy()
        tm.assert_numpy_array_equal(result, expected)

    def test_map_na_action_ignore(self, data_missing_for_sorting):
        zero = data_missing_for_sorting[2]
        result = data_missing_for_sorting.map(lambda x: zero, na_action="ignore")
        if data_missing_for_sorting.dtype.kind == "b":
            expected = np.array([False, pd.NA, False], dtype=object)
        else:
            expected = np.array([zero, np.nan, zero])
        tm.assert_numpy_array_equal(result, expected)

    def _get_expected_exception(self, op_name, obj, other):
        try:
            dtype = tm.get_dtype(obj)
        except AttributeError:
            # passed arguments reversed
            dtype = tm.get_dtype(other)

        if dtype.kind == "b":
            if op_name.strip("_").lstrip("r") in ["pow", "truediv", "floordiv"]:
                # match behavior with non-masked bool dtype
                return NotImplementedError
            elif op_name in ["__sub__", "__rsub__"]:
                # exception message would include "numpy boolean subtract""
                return TypeError
            return None
        return None

    def _cast_pointwise_result(self, op_name: str, obj, other, pointwise_result):
        sdtype = tm.get_dtype(obj)
        expected = pointwise_result

        if op_name in ("eq", "ne", "le", "ge", "lt", "gt"):
            return expected.astype("boolean")

        if sdtype.kind in "iu":
            if op_name in ("__rtruediv__", "__truediv__", "__div__"):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        "Downcasting object dtype arrays",
                        category=FutureWarning,
                    )
                    filled = expected.fillna(np.nan)
                expected = filled.astype("Float64")
            else:
                # combine method result in 'biggest' (int64) dtype
                expected = expected.astype(sdtype)
        elif sdtype.kind == "b":
            if op_name in (
                "__floordiv__",
                "__rfloordiv__",
                "__pow__",
                "__rpow__",
                "__mod__",
                "__rmod__",
            ):
                # combine keeps boolean type
                expected = expected.astype("Int8")

            elif op_name in ("__truediv__", "__rtruediv__"):
                # combine with bools does not generate the correct result
                #  (numpy behaviour for div is to regard the bools as numeric)
                op = self.get_op_from_name(op_name)
                expected = self._combine(obj.astype(float), other, op)
                expected = expected.astype("Float64")

            if op_name == "__rpow__":
                # for rpow, combine does not propagate NaN
                result = getattr(obj, op_name)(other)
                expected[result.isna()] = np.nan
        else:
            # combine method result in 'biggest' (float64) dtype
            expected = expected.astype(sdtype)
        return expected

    def test_divmod_series_array(self, data, data_for_twos, request):
        if data.dtype.kind == "b":
            mark = pytest.mark.xfail(
                reason="Inconsistency between floordiv and divmod; we raise for "
                "floordiv but not for divmod. This matches what we do for "
                "non-masked bool dtype."
            )
            request.applymarker(mark)
        super().test_divmod_series_array(data, data_for_twos)

    def test_combine_le(self, data_repeated):
        # TODO: patching self is a bad pattern here
        orig_data1, orig_data2 = data_repeated(2)
        if orig_data1.dtype.kind == "b":
            self._combine_le_expected_dtype = "boolean"
        else:
            # TODO: can we make this boolean?
            self._combine_le_expected_dtype = object
        super().test_combine_le(data_repeated)

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        if op_name in ["any", "all"] and ser.dtype.kind != "b":
            pytest.skip(reason="Tested in tests/reductions/test_reductions.py")
        return True

    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        # overwrite to ensure pd.NA is tested instead of np.nan
        # https://github.com/pandas-dev/pandas/issues/30958

        cmp_dtype = "int64"
        if ser.dtype.kind == "f":
            # Item "dtype[Any]" of "Union[dtype[Any], ExtensionDtype]" has
            # no attribute "numpy_dtype"
            cmp_dtype = ser.dtype.numpy_dtype  # type: ignore[union-attr]
        elif ser.dtype.kind == "b":
            if op_name in ["min", "max"]:
                cmp_dtype = "bool"

        # TODO: prod with integer dtypes does *not* match the result we would
        #  get if we used object for cmp_dtype. In that cae the object result
        #  is a large integer while the non-object case overflows and returns 0
        alt = ser.dropna().astype(cmp_dtype)
        if op_name == "count":
            result = getattr(ser, op_name)()
            expected = getattr(alt, op_name)()
        else:
            result = getattr(ser, op_name)(skipna=skipna)
            expected = getattr(alt, op_name)(skipna=skipna)
            if not skipna and ser.isna().any() and op_name not in ["any", "all"]:
                expected = pd.NA
        tm.assert_almost_equal(result, expected)

    def _get_expected_reduction_dtype(self, arr, op_name: str, skipna: bool):
        if is_float_dtype(arr.dtype):
            cmp_dtype = arr.dtype.name
        elif op_name in ["mean", "median", "var", "std", "skew"]:
            cmp_dtype = "Float64"
        elif op_name in ["max", "min"]:
            cmp_dtype = arr.dtype.name
        elif arr.dtype in ["Int64", "UInt64"]:
            cmp_dtype = arr.dtype.name
        elif is_signed_integer_dtype(arr.dtype):
            # TODO: Why does Window Numpy 2.0 dtype depend on skipna?
            cmp_dtype = (
                "Int32"
                if (is_platform_windows() and (not np_version_gt2 or not skipna))
                or not IS64
                else "Int64"
            )
        elif is_unsigned_integer_dtype(arr.dtype):
            cmp_dtype = (
                "UInt32"
                if (is_platform_windows() and (not np_version_gt2 or not skipna))
                or not IS64
                else "UInt64"
            )
        elif arr.dtype.kind == "b":
            if op_name in ["mean", "median", "var", "std", "skew"]:
                cmp_dtype = "Float64"
            elif op_name in ["min", "max"]:
                cmp_dtype = "boolean"
            elif op_name in ["sum", "prod"]:
                cmp_dtype = (
                    "Int32"
                    if (is_platform_windows() and (not np_version_gt2 or not skipna))
                    or not IS64
                    else "Int64"
                )
            else:
                raise TypeError("not supposed to reach this")
        else:
            raise TypeError("not supposed to reach this")
        return cmp_dtype

    def _supports_accumulation(self, ser: pd.Series, op_name: str) -> bool:
        return True

    def check_accumulate(self, ser: pd.Series, op_name: str, skipna: bool):
        # overwrite to ensure pd.NA is tested instead of np.nan
        # https://github.com/pandas-dev/pandas/issues/30958
        length = 64
        if is_windows_or_32bit:
            # Item "ExtensionDtype" of "Union[dtype[Any], ExtensionDtype]" has
            # no attribute "itemsize"
            if not ser.dtype.itemsize == 8:  # type: ignore[union-attr]
                length = 32

        if ser.dtype.name.startswith("U"):
            expected_dtype = f"UInt{length}"
        elif ser.dtype.name.startswith("I"):
            expected_dtype = f"Int{length}"
        elif ser.dtype.name.startswith("F"):
            # Incompatible types in assignment (expression has type
            # "Union[dtype[Any], ExtensionDtype]", variable has type "str")
            expected_dtype = ser.dtype  # type: ignore[assignment]
        elif ser.dtype.kind == "b":
            if op_name in ("cummin", "cummax"):
                expected_dtype = "boolean"
            else:
                expected_dtype = f"Int{length}"

        if expected_dtype == "Float32" and op_name == "cumprod" and skipna:
            # TODO: xfail?
            pytest.skip(
                f"Float32 precision lead to large differences with op {op_name} "
                f"and skipna={skipna}"
            )

        if op_name == "cumsum":
            result = getattr(ser, op_name)(skipna=skipna)
            expected = pd.Series(
                pd.array(
                    getattr(ser.astype("float64"), op_name)(skipna=skipna),
                    dtype=expected_dtype,
                )
            )
            tm.assert_series_equal(result, expected)
        elif op_name in ["cummax", "cummin"]:
            result = getattr(ser, op_name)(skipna=skipna)
            expected = pd.Series(
                pd.array(
                    getattr(ser.astype("float64"), op_name)(skipna=skipna),
                    dtype=ser.dtype,
                )
            )
            tm.assert_series_equal(result, expected)
        elif op_name == "cumprod":
            result = getattr(ser[:12], op_name)(skipna=skipna)
            expected = pd.Series(
                pd.array(
                    getattr(ser[:12].astype("float64"), op_name)(skipna=skipna),
                    dtype=expected_dtype,
                )
            )
            tm.assert_series_equal(result, expected)

        else:
            raise NotImplementedError(f"{op_name} not supported")


class Test2DCompat(base.Dim2CompatTests):
    pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_describe.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestDataFrameDescribe:
    def test_describe_bool_in_mixed_frame(self):
        df = DataFrame(
            {
                "string_data": ["a", "b", "c", "d", "e"],
                "bool_data": [True, True, False, False, False],
                "int_data": [10, 20, 30, 40, 50],
            }
        )

        # Integer data are included in .describe() output,
        # Boolean and string data are not.
        result = df.describe()
        expected = DataFrame(
            {"int_data": [5, 30, df.int_data.std(), 10, 20, 30, 40, 50]},
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        )
        tm.assert_frame_equal(result, expected)

        # Top value is a boolean value that is False
        result = df.describe(include=["bool"])

        expected = DataFrame(
            {"bool_data": [5, 2, False, 3]}, index=["count", "unique", "top", "freq"]
        )
        tm.assert_frame_equal(result, expected)

    def test_describe_empty_object(self):
        # GH#27183
        df = DataFrame({"A": [None, None]}, dtype=object)
        result = df.describe()
        expected = DataFrame(
            {"A": [0, 0, np.nan, np.nan]},
            dtype=object,
            index=["count", "unique", "top", "freq"],
        )
        tm.assert_frame_equal(result, expected)

        result = df.iloc[:0].describe()
        tm.assert_frame_equal(result, expected)

    def test_describe_bool_frame(self):
        # GH#13891
        df = DataFrame(
            {
                "bool_data_1": [False, False, True, True],
                "bool_data_2": [False, True, True, True],
            }
        )
        result = df.describe()
        expected = DataFrame(
            {"bool_data_1": [4, 2, False, 2], "bool_data_2": [4, 2, True, 3]},
            index=["count", "unique", "top", "freq"],
        )
        tm.assert_frame_equal(result, expected)

        df = DataFrame(
            {
                "bool_data": [False, False, True, True, False],
                "int_data": [0, 1, 2, 3, 4],
            }
        )
        result = df.describe()
        expected = DataFrame(
            {"int_data": [5, 2, df.int_data.std(), 0, 1, 2, 3, 4]},
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        )
        tm.assert_frame_equal(result, expected)

        df = DataFrame(
            {"bool_data": [False, False, True, True], "str_data": ["a", "b", "c", "a"]}
        )
        result = df.describe()
        expected = DataFrame(
            {"bool_data": [4, 2, False, 2], "str_data": [4, 3, "a", 2]},
            index=["count", "unique", "top", "freq"],
        )
        tm.assert_frame_equal(result, expected)

    def test_describe_categorical(self):
        df = DataFrame({"value": np.random.default_rng(2).integers(0, 10000, 100)})
        labels = [f"{i} - {i + 499}" for i in range(0, 10000, 500)]
        cat_labels = Categorical(labels, labels)

        df = df.sort_values(by=["value"], ascending=True)
        df["value_group"] = pd.cut(
            df.value, range(0, 10500, 500), right=False, labels=cat_labels
        )
        cat = df

        # Categoricals should not show up together with numerical columns
        result = cat.describe()
        assert len(result.columns) == 1

        # In a frame, describe() for the cat should be the same as for string
        # arrays (count, unique, top, freq)

        cat = Categorical(
            ["a", "b", "b", "b"], categories=["a", "b", "c"], ordered=True
        )
        s = Series(cat)
        result = s.describe()
        expected = Series([4, 2, "b", 3], index=["count", "unique", "top", "freq"])
        tm.assert_series_equal(result, expected)

        cat = Series(Categorical(["a", "b", "c", "c"]))
        df3 = DataFrame({"cat": cat, "s": ["a", "b", "c", "c"]})
        result = df3.describe()
        tm.assert_numpy_array_equal(result["cat"].values, result["s"].values)

    def test_describe_empty_categorical_column(self):
        # GH#26397
        # Ensure the index of an empty categorical DataFrame column
        # also contains (count, unique, top, freq)
        df = DataFrame({"empty_col": Categorical([])})
        result = df.describe()
        expected = DataFrame(
            {"empty_col": [0, 0, np.nan, np.nan]},
            index=["count", "unique", "top", "freq"],
            dtype="object",
        )
        tm.assert_frame_equal(result, expected)
        # ensure NaN, not None
        assert np.isnan(result.iloc[2, 0])
        assert np.isnan(result.iloc[3, 0])

    def test_describe_categorical_columns(self):
        # GH#11558
        columns = pd.CategoricalIndex(["int1", "int2", "obj"], ordered=True, name="XXX")
        df = DataFrame(
            {
                "int1": [10, 20, 30, 40, 50],
                "int2": [10, 20, 30, 40, 50],
                "obj": ["A", 0, None, "X", 1],
            },
            columns=columns,
        )
        result = df.describe()

        exp_columns = pd.CategoricalIndex(
            ["int1", "int2"],
            categories=["int1", "int2", "obj"],
            ordered=True,
            name="XXX",
        )
        expected = DataFrame(
            {
                "int1": [5, 30, df.int1.std(), 10, 20, 30, 40, 50],
                "int2": [5, 30, df.int2.std(), 10, 20, 30, 40, 50],
            },
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
            columns=exp_columns,
        )

        tm.assert_frame_equal(result, expected)
        tm.assert_categorical_equal(result.columns.values, expected.columns.values)

    def test_describe_datetime_columns(self):
        columns = pd.DatetimeIndex(
            ["2011-01-01", "2011-02-01", "2011-03-01"],
            freq="MS",
            tz="US/Eastern",
            name="XXX",
        )
        df = DataFrame(
            {
                0: [10, 20, 30, 40, 50],
                1: [10, 20, 30, 40, 50],
                2: ["A", 0, None, "X", 1],
            }
        )
        df.columns = columns
        result = df.describe()

        exp_columns = pd.DatetimeIndex(
            ["2011-01-01", "2011-02-01"], freq="MS", tz="US/Eastern", name="XXX"
        )
        expected = DataFrame(
            {
                0: [5, 30, df.iloc[:, 0].std(), 10, 20, 30, 40, 50],
                1: [5, 30, df.iloc[:, 1].std(), 10, 20, 30, 40, 50],
            },
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        )
        expected.columns = exp_columns
        tm.assert_frame_equal(result, expected)
        assert result.columns.freq == "MS"
        assert result.columns.tz == expected.columns.tz

    def test_describe_timedelta_values(self):
        # GH#6145
        t1 = pd.timedelta_range("1 days", freq="D", periods=5)
        t2 = pd.timedelta_range("1 hours", freq="h", periods=5)
        df = DataFrame({"t1": t1, "t2": t2})

        expected = DataFrame(
            {
                "t1": [
                    5,
                    pd.Timedelta("3 days"),
                    df.iloc[:, 0].std(),
                    pd.Timedelta("1 days"),
                    pd.Timedelta("2 days"),
                    pd.Timedelta("3 days"),
                    pd.Timedelta("4 days"),
                    pd.Timedelta("5 days"),
                ],
                "t2": [
                    5,
                    pd.Timedelta("3 hours"),
                    df.iloc[:, 1].std(),
                    pd.Timedelta("1 hours"),
                    pd.Timedelta("2 hours"),
                    pd.Timedelta("3 hours"),
                    pd.Timedelta("4 hours"),
                    pd.Timedelta("5 hours"),
                ],
            },
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        )

        result = df.describe()
        tm.assert_frame_equal(result, expected)

        exp_repr = (
            "                              t1                         t2\n"
            "count                          5                          5\n"
            "mean             3 days 00:00:00            0 days 03:00:00\n"
            "std    1 days 13:56:50.394919273  0 days 01:34:52.099788303\n"
            "min              1 days 00:00:00            0 days 01:00:00\n"
            "25%              2 days 00:00:00            0 days 02:00:00\n"
            "50%              3 days 00:00:00            0 days 03:00:00\n"
            "75%              4 days 00:00:00            0 days 04:00:00\n"
            "max              5 days 00:00:00            0 days 05:00:00"
        )
        assert repr(result) == exp_repr

    def test_describe_tz_values(self, tz_naive_fixture):
        # GH#21332
        tz = tz_naive_fixture
        s1 = Series(range(5))
        start = Timestamp(2018, 1, 1)
        end = Timestamp(2018, 1, 5)
        s2 = Series(date_range(start, end, tz=tz))
        df = DataFrame({"s1": s1, "s2": s2})

        expected = DataFrame(
            {
                "s1": [5, 2, 0, 1, 2, 3, 4, 1.581139],
                "s2": [
                    5,
                    Timestamp(2018, 1, 3).tz_localize(tz),
                    start.tz_localize(tz),
                    s2[1],
                    s2[2],
                    s2[3],
                    end.tz_localize(tz),
                    np.nan,
                ],
            },
            index=["count", "mean", "min", "25%", "50%", "75%", "max", "std"],
        )
        result = df.describe(include="all")
        tm.assert_frame_equal(result, expected)

    def test_datetime_is_numeric_includes_datetime(self):
        df = DataFrame({"a": date_range("2012", periods=3), "b": [1, 2, 3]})
        result = df.describe()
        expected = DataFrame(
            {
                "a": [
                    3,
                    Timestamp("2012-01-02"),
                    Timestamp("2012-01-01"),
                    Timestamp("2012-01-01T12:00:00"),
                    Timestamp("2012-01-02"),
                    Timestamp("2012-01-02T12:00:00"),
                    Timestamp("2012-01-03"),
                    np.nan,
                ],
                "b": [3, 2, 1, 1.5, 2, 2.5, 3, 1],
            },
            index=["count", "mean", "min", "25%", "50%", "75%", "max", "std"],
        )
        tm.assert_frame_equal(result, expected)

    def test_describe_tz_values2(self):
        tz = "CET"
        s1 = Series(range(5))
        start = Timestamp(2018, 1, 1)
        end = Timestamp(2018, 1, 5)
        s2 = Series(date_range(start, end, tz=tz))
        df = DataFrame({"s1": s1, "s2": s2})

        s1_ = s1.describe()
        s2_ = s2.describe()
        idx = [
            "count",
            "mean",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
            "std",
        ]
        expected = pd.concat([s1_, s2_], axis=1, keys=["s1", "s2"]).reindex(
            idx, copy=False
        )

        result = df.describe(include="all")
        tm.assert_frame_equal(result, expected)

    def test_describe_percentiles_integer_idx(self):
        # GH#26660
        df = DataFrame({"x": [1]})
        pct = np.linspace(0, 1, 10 + 1)
        result = df.describe(percentiles=pct)

        expected = DataFrame(
            {"x": [1.0, 1.0, np.nan, 1.0, *(1.0 for _ in pct), 1.0]},
            index=[
                "count",
                "mean",
                "std",
                "min",
                "0%",
                "10%",
                "20%",
                "30%",
                "40%",
                "50%",
                "60%",
                "70%",
                "80%",
                "90%",
                "100%",
                "max",
            ],
        )
        tm.assert_frame_equal(result, expected)

    def test_describe_does_not_raise_error_for_dictlike_elements(self):
        # GH#32409
        df = DataFrame([{"test": {"a": "1"}}, {"test": {"a": "2"}}])
        expected = DataFrame(
            {"test": [2, 2, {"a": "1"}, 1]}, index=["count", "unique", "top", "freq"]
        )
        result = df.describe()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("exclude", ["x", "y", ["x", "y"], ["x", "z"]])
    def test_describe_when_include_all_exclude_not_allowed(self, exclude):
        """
        When include is 'all', then setting exclude != None is not allowed.
        """
        df = DataFrame({"x": [1], "y": [2], "z": [3]})
        msg = "exclude must be None when include is 'all'"
        with pytest.raises(ValueError, match=msg):
            df.describe(include="all", exclude=exclude)

    def test_describe_with_duplicate_columns(self):
        df = DataFrame(
            [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
            columns=["bar", "a", "a"],
            dtype="float64",
        )
        result = df.describe()
        ser = df.iloc[:, 0].describe()
        expected = pd.concat([ser, ser, ser], keys=df.columns, axis=1)
        tm.assert_frame_equal(result, expected)

    def test_ea_with_na(self, any_numeric_ea_dtype):
        # GH#48778

        df = DataFrame({"a": [1, pd.NA, pd.NA], "b": pd.NA}, dtype=any_numeric_ea_dtype)
        result = df.describe()
        expected = DataFrame(
            {"a": [1.0, 1.0, pd.NA] + [1.0] * 5, "b": [0.0] + [pd.NA] * 7},
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
            dtype="Float64",
        )
        tm.assert_frame_equal(result, expected)

    def test_describe_exclude_pa_dtype(self):
        # GH#52570
        pa = pytest.importorskip("pyarrow")
        df = DataFrame(
            {
                "a": Series([1, 2, 3], dtype=pd.ArrowDtype(pa.int8())),
                "b": Series([1, 2, 3], dtype=pd.ArrowDtype(pa.int16())),
                "c": Series([1, 2, 3], dtype=pd.ArrowDtype(pa.int32())),
            }
        )
        result = df.describe(
            include=pd.ArrowDtype(pa.int8()), exclude=pd.ArrowDtype(pa.int32())
        )
        expected = DataFrame(
            {"a": [3, 2, 1, 1, 1.5, 2, 2.5, 3]},
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
            dtype=pd.ArrowDtype(pa.float64()),
        )
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\pytables\test_read.py ===
from contextlib import closing
from pathlib import Path
import re

import numpy as np
import pytest

from pandas._libs.tslibs import Timestamp
from pandas.compat import is_platform_windows

import pandas as pd
from pandas import (
    DataFrame,
    HDFStore,
    Index,
    Series,
    _testing as tm,
    date_range,
    read_hdf,
)
from pandas.tests.io.pytables.common import (
    _maybe_remove,
    ensure_clean_store,
)
from pandas.util import _test_decorators as td

from pandas.io.pytables import TableIterator

pytestmark = [pytest.mark.single_cpu]


def test_read_missing_key_close_store(tmp_path, setup_path):
    # GH 25766
    path = tmp_path / setup_path
    df = DataFrame({"a": range(2), "b": range(2)})
    df.to_hdf(path, key="k1")

    with pytest.raises(KeyError, match="'No object named k2 in the file'"):
        read_hdf(path, "k2")

    # smoke test to test that file is properly closed after
    # read with KeyError before another write
    df.to_hdf(path, key="k2")


def test_read_index_error_close_store(tmp_path, setup_path):
    # GH 25766
    path = tmp_path / setup_path
    df = DataFrame({"A": [], "B": []}, index=[])
    df.to_hdf(path, key="k1")

    with pytest.raises(IndexError, match=r"list index out of range"):
        read_hdf(path, "k1", stop=0)

    # smoke test to test that file is properly closed after
    # read with IndexError before another write
    df.to_hdf(path, key="k1")


def test_read_missing_key_opened_store(tmp_path, setup_path):
    # GH 28699
    path = tmp_path / setup_path
    df = DataFrame({"a": range(2), "b": range(2)})
    df.to_hdf(path, key="k1")

    with HDFStore(path, "r") as store:
        with pytest.raises(KeyError, match="'No object named k2 in the file'"):
            read_hdf(store, "k2")

        # Test that the file is still open after a KeyError and that we can
        # still read from it.
        read_hdf(store, "k1")


def test_read_column(setup_path):
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )

    with ensure_clean_store(setup_path) as store:
        _maybe_remove(store, "df")

        # GH 17912
        # HDFStore.select_column should raise a KeyError
        # exception if the key is not a valid store
        with pytest.raises(KeyError, match="No object named df in the file"):
            store.select_column("df", "index")

        store.append("df", df)
        # error
        with pytest.raises(
            KeyError, match=re.escape("'column [foo] not found in the table'")
        ):
            store.select_column("df", "foo")

        msg = re.escape("select_column() got an unexpected keyword argument 'where'")
        with pytest.raises(TypeError, match=msg):
            store.select_column("df", "index", where=["index>5"])

        # valid
        result = store.select_column("df", "index")
        tm.assert_almost_equal(result.values, Series(df.index).values)
        assert isinstance(result, Series)

        # not a data indexable column
        msg = re.escape(
            "column [values_block_0] can not be extracted individually; "
            "it is not data indexable"
        )
        with pytest.raises(ValueError, match=msg):
            store.select_column("df", "values_block_0")

        # a data column
        df2 = df.copy()
        df2["string"] = "foo"
        store.append("df2", df2, data_columns=["string"])
        result = store.select_column("df2", "string")
        tm.assert_almost_equal(result.values, df2["string"].values)

        # a data column with NaNs, result excludes the NaNs
        df3 = df.copy()
        df3["string"] = "foo"
        df3.loc[df3.index[4:6], "string"] = np.nan
        store.append("df3", df3, data_columns=["string"])
        result = store.select_column("df3", "string")
        tm.assert_almost_equal(result.values, df3["string"].values)

        # start/stop
        result = store.select_column("df3", "string", start=2)
        tm.assert_almost_equal(result.values, df3["string"].values[2:])

        result = store.select_column("df3", "string", start=-2)
        tm.assert_almost_equal(result.values, df3["string"].values[-2:])

        result = store.select_column("df3", "string", stop=2)
        tm.assert_almost_equal(result.values, df3["string"].values[:2])

        result = store.select_column("df3", "string", stop=-2)
        tm.assert_almost_equal(result.values, df3["string"].values[:-2])

        result = store.select_column("df3", "string", start=2, stop=-2)
        tm.assert_almost_equal(result.values, df3["string"].values[2:-2])

        result = store.select_column("df3", "string", start=-2, stop=2)
        tm.assert_almost_equal(result.values, df3["string"].values[-2:2])

        # GH 10392 - make sure column name is preserved
        df4 = DataFrame({"A": np.random.default_rng(2).standard_normal(10), "B": "foo"})
        store.append("df4", df4, data_columns=True)
        expected = df4["B"]
        result = store.select_column("df4", "B")
        tm.assert_series_equal(result, expected)


def test_pytables_native_read(datapath):
    with ensure_clean_store(
        datapath("io", "data", "legacy_hdf/pytables_native.h5"), mode="r"
    ) as store:
        d2 = store["detector/readout"]
    assert isinstance(d2, DataFrame)


@pytest.mark.skipif(is_platform_windows(), reason="native2 read fails oddly on windows")
def test_pytables_native2_read(datapath):
    with ensure_clean_store(
        datapath("io", "data", "legacy_hdf", "pytables_native2.h5"), mode="r"
    ) as store:
        str(store)
        d1 = store["detector"]
    assert isinstance(d1, DataFrame)


def test_legacy_table_fixed_format_read_py2(datapath):
    # GH 24510
    # legacy table with fixed format written in Python 2
    with ensure_clean_store(
        datapath("io", "data", "legacy_hdf", "legacy_table_fixed_py2.h5"), mode="r"
    ) as store:
        result = store.select("df")
    expected = DataFrame(
        [[1, 2, 3, "D"]],
        columns=["A", "B", "C", "D"],
        index=Index(["ABC"], name="INDEX_NAME"),
    )
    tm.assert_frame_equal(expected, result)


def test_legacy_table_fixed_format_read_datetime_py2(datapath):
    # GH 31750
    # legacy table with fixed format and datetime64 column written in Python 2
    expected = DataFrame(
        [[Timestamp("2020-02-06T18:00")]],
        columns=["A"],
        index=Index(["date"]),
        dtype="M8[ns]",
    )
    with ensure_clean_store(
        datapath("io", "data", "legacy_hdf", "legacy_table_fixed_datetime_py2.h5"),
        mode="r",
    ) as store:
        result = store.select("df")
    tm.assert_frame_equal(expected, result)


def test_legacy_table_read_py2(datapath):
    # issue: 24925
    # legacy table written in Python 2
    with ensure_clean_store(
        datapath("io", "data", "legacy_hdf", "legacy_table_py2.h5"), mode="r"
    ) as store:
        result = store.select("table")

    expected = DataFrame({"a": ["a", "b"], "b": [2, 3]})
    tm.assert_frame_equal(expected, result)


def test_read_hdf_open_store(tmp_path, setup_path, using_infer_string):
    # GH10330
    # No check for non-string path_or-buf, and no test of open store
    df = DataFrame(
        np.random.default_rng(2).random((4, 5)),
        index=list("abcd"),
        columns=list("ABCDE"),
    )
    df.index.name = "letters"
    df = df.set_index(keys="E", append=True)

    path = tmp_path / setup_path
    if using_infer_string:
        # TODO(infer_string) make this work for string dtype
        msg = "Saving a MultiIndex with an extension dtype is not supported."
        with pytest.raises(NotImplementedError, match=msg):
            df.to_hdf(path, key="df", mode="w")
        return
    df.to_hdf(path, key="df", mode="w")
    direct = read_hdf(path, "df")
    with HDFStore(path, mode="r") as store:
        indirect = read_hdf(store, "df")
        tm.assert_frame_equal(direct, indirect)
        assert store.is_open


def test_read_hdf_index_not_view(tmp_path, setup_path):
    # GH 37441
    # Ensure that the index of the DataFrame is not a view
    # into the original recarray that pytables reads in
    df = DataFrame(
        np.random.default_rng(2).random((4, 5)),
        index=[0, 1, 2, 3],
        columns=list("ABCDE"),
    )

    path = tmp_path / setup_path
    df.to_hdf(path, key="df", mode="w", format="table")

    df2 = read_hdf(path, "df")
    assert df2.index._data.base is None
    tm.assert_frame_equal(df, df2)


def test_read_hdf_iterator(tmp_path, setup_path):
    df = DataFrame(
        np.random.default_rng(2).random((4, 5)),
        index=list("abcd"),
        columns=list("ABCDE"),
    )
    df.index.name = "letters"
    df = df.set_index(keys="E", append=True)

    path = tmp_path / setup_path
    df.to_hdf(path, key="df", mode="w", format="t")
    direct = read_hdf(path, "df")
    iterator = read_hdf(path, "df", iterator=True)
    with closing(iterator.store):
        assert isinstance(iterator, TableIterator)
        indirect = next(iterator.__iter__())
    tm.assert_frame_equal(direct, indirect)


def test_read_nokey(tmp_path, setup_path):
    # GH10443
    df = DataFrame(
        np.random.default_rng(2).random((4, 5)),
        index=list("abcd"),
        columns=list("ABCDE"),
    )

    # Categorical dtype not supported for "fixed" format. So no need
    # to test with that dtype in the dataframe here.
    path = tmp_path / setup_path
    df.to_hdf(path, key="df", mode="a")
    reread = read_hdf(path)
    tm.assert_frame_equal(df, reread)
    df.to_hdf(path, key="df2", mode="a")

    msg = "key must be provided when HDF5 file contains multiple datasets."
    with pytest.raises(ValueError, match=msg):
        read_hdf(path)


def test_read_nokey_table(tmp_path, setup_path):
    # GH13231
    df = DataFrame({"i": range(5), "c": Series(list("abacd"), dtype="category")})

    path = tmp_path / setup_path
    df.to_hdf(path, key="df", mode="a", format="table")
    reread = read_hdf(path)
    tm.assert_frame_equal(df, reread)
    df.to_hdf(path, key="df2", mode="a", format="table")

    msg = "key must be provided when HDF5 file contains multiple datasets."
    with pytest.raises(ValueError, match=msg):
        read_hdf(path)


def test_read_nokey_empty(tmp_path, setup_path):
    path = tmp_path / setup_path
    store = HDFStore(path)
    store.close()
    msg = re.escape(
        "Dataset(s) incompatible with Pandas data types, not table, or no "
        "datasets found in HDF5 file."
    )
    with pytest.raises(ValueError, match=msg):
        read_hdf(path)


def test_read_from_pathlib_path(tmp_path, setup_path):
    # GH11773
    expected = DataFrame(
        np.random.default_rng(2).random((4, 5)),
        index=list("abcd"),
        columns=list("ABCDE"),
    )
    filename = tmp_path / setup_path
    path_obj = Path(filename)

    expected.to_hdf(path_obj, key="df", mode="a")
    actual = read_hdf(path_obj, key="df")

    tm.assert_frame_equal(expected, actual)


@td.skip_if_no("py.path")
def test_read_from_py_localpath(tmp_path, setup_path):
    # GH11773
    from py.path import local as LocalPath

    expected = DataFrame(
        np.random.default_rng(2).random((4, 5)),
        index=list("abcd"),
        columns=list("ABCDE"),
    )
    filename = tmp_path / setup_path
    path_obj = LocalPath(filename)

    expected.to_hdf(path_obj, key="df", mode="a")
    actual = read_hdf(path_obj, key="df")

    tm.assert_frame_equal(expected, actual)


@pytest.mark.parametrize("format", ["fixed", "table"])
def test_read_hdf_series_mode_r(tmp_path, format, setup_path):
    # GH 16583
    # Tests that reading a Series saved to an HDF file
    # still works if a mode='r' argument is supplied
    series = Series(range(10), dtype=np.float64)
    path = tmp_path / setup_path
    series.to_hdf(path, key="data", format=format)
    result = read_hdf(path, key="data", mode="r")
    tm.assert_series_equal(result, series)


@pytest.mark.filterwarnings(r"ignore:Period with BDay freq is deprecated:FutureWarning")
@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
def test_read_py2_hdf_file_in_py3(datapath):
    # GH 16781

    # tests reading a PeriodIndex DataFrame written in Python2 in Python3

    # the file was generated in Python 2.7 like so:
    #
    # df = DataFrame([1.,2,3], index=pd.PeriodIndex(
    #              ['2015-01-01', '2015-01-02', '2015-01-05'], freq='B'))
    # df.to_hdf('periodindex_0.20.1_x86_64_darwin_2.7.13.h5', 'p')

    expected = DataFrame(
        [1.0, 2, 3],
        index=pd.PeriodIndex(["2015-01-01", "2015-01-02", "2015-01-05"], freq="B"),
    )

    with ensure_clean_store(
        datapath(
            "io", "data", "legacy_hdf", "periodindex_0.20.1_x86_64_darwin_2.7.13.h5"
        ),
        mode="r",
    ) as store:
        result = store["p"]
    tm.assert_frame_equal(result, expected)


def test_read_infer_string(tmp_path, setup_path):
    # GH#54431
    df = DataFrame({"a": ["a", "b", None]})
    path = tmp_path / setup_path
    df.to_hdf(path, key="data", format="table")
    with pd.option_context("future.infer_string", True):
        result = read_hdf(path, key="data", mode="r")
    expected = DataFrame(
        {"a": ["a", "b", None]},
        dtype=pd.StringDtype(na_value=np.nan),
        columns=Index(["a"], dtype=pd.StringDtype(na_value=np.nan)),
    )
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\test_hashing.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.util.hashing import hash_tuples
from pandas.util import (
    hash_array,
    hash_pandas_object,
)


@pytest.fixture(
    params=[
        Series([1, 2, 3] * 3, dtype="int32"),
        Series([None, 2.5, 3.5] * 3, dtype="float32"),
        Series(["a", "b", "c"] * 3, dtype="category"),
        Series(["d", "e", "f"] * 3),
        Series([True, False, True] * 3),
        Series(pd.date_range("20130101", periods=9)),
        Series(pd.date_range("20130101", periods=9, tz="US/Eastern")),
        Series(timedelta_range("2000", periods=9)),
    ]
)
def series(request):
    return request.param


@pytest.fixture(params=[True, False])
def index(request):
    return request.param


def test_consistency():
    # Check that our hash doesn't change because of a mistake
    # in the actual code; this is the ground truth.
    result = hash_pandas_object(Index(["foo", "bar", "baz"]))
    expected = Series(
        np.array(
            [3600424527151052760, 1374399572096150070, 477881037637427054],
            dtype="uint64",
        ),
        index=["foo", "bar", "baz"],
    )
    tm.assert_series_equal(result, expected)


def test_hash_array(series):
    arr = series.values
    tm.assert_numpy_array_equal(hash_array(arr), hash_array(arr))


@pytest.mark.parametrize("dtype", ["U", object])
def test_hash_array_mixed(dtype):
    result1 = hash_array(np.array(["3", "4", "All"]))
    result2 = hash_array(np.array([3, 4, "All"], dtype=dtype))

    tm.assert_numpy_array_equal(result1, result2)


@pytest.mark.parametrize("val", [5, "foo", pd.Timestamp("20130101")])
def test_hash_array_errors(val):
    msg = "must pass a ndarray-like"
    with pytest.raises(TypeError, match=msg):
        hash_array(val)


def test_hash_array_index_exception():
    # GH42003 TypeError instead of AttributeError
    obj = pd.DatetimeIndex(["2018-10-28 01:20:00"], tz="Europe/Berlin")

    msg = "Use hash_pandas_object instead"
    with pytest.raises(TypeError, match=msg):
        hash_array(obj)


def test_hash_tuples():
    tuples = [(1, "one"), (1, "two"), (2, "one")]
    result = hash_tuples(tuples)

    expected = hash_pandas_object(MultiIndex.from_tuples(tuples)).values
    tm.assert_numpy_array_equal(result, expected)

    # We only need to support MultiIndex and list-of-tuples
    msg = "|".join(["object is not iterable", "zip argument #1 must support iteration"])
    with pytest.raises(TypeError, match=msg):
        hash_tuples(tuples[0])


@pytest.mark.parametrize("val", [5, "foo", pd.Timestamp("20130101")])
def test_hash_tuples_err(val):
    msg = "must be convertible to a list-of-tuples"
    with pytest.raises(TypeError, match=msg):
        hash_tuples(val)


def test_multiindex_unique():
    mi = MultiIndex.from_tuples([(118, 472), (236, 118), (51, 204), (102, 51)])
    assert mi.is_unique is True

    result = hash_pandas_object(mi)
    assert result.is_unique is True


def test_multiindex_objects():
    mi = MultiIndex(
        levels=[["b", "d", "a"], [1, 2, 3]],
        codes=[[0, 1, 0, 2], [2, 0, 0, 1]],
        names=["col1", "col2"],
    )
    recons = mi._sort_levels_monotonic()

    # These are equal.
    assert mi.equals(recons)
    assert Index(mi.values).equals(Index(recons.values))


@pytest.mark.parametrize(
    "obj",
    [
        Series([1, 2, 3]),
        Series([1.0, 1.5, 3.2]),
        Series([1.0, 1.5, np.nan]),
        Series([1.0, 1.5, 3.2], index=[1.5, 1.1, 3.3]),
        Series(["a", "b", "c"]),
        Series(["a", np.nan, "c"]),
        Series(["a", None, "c"]),
        Series([True, False, True]),
        Series(dtype=object),
        DataFrame({"x": ["a", "b", "c"], "y": [1, 2, 3]}),
        DataFrame(),
        DataFrame(np.full((10, 4), np.nan)),
        DataFrame(
            {
                "A": [0.0, 1.0, 2.0, 3.0, 4.0],
                "B": [0.0, 1.0, 0.0, 1.0, 0.0],
                "C": Index(["foo1", "foo2", "foo3", "foo4", "foo5"], dtype=object),
                "D": pd.date_range("20130101", periods=5),
            }
        ),
        DataFrame(range(5), index=pd.date_range("2020-01-01", periods=5)),
        Series(range(5), index=pd.date_range("2020-01-01", periods=5)),
        Series(period_range("2020-01-01", periods=10, freq="D")),
        Series(pd.date_range("20130101", periods=3, tz="US/Eastern")),
    ],
)
def test_hash_pandas_object(obj, index):
    a = hash_pandas_object(obj, index=index)
    b = hash_pandas_object(obj, index=index)
    tm.assert_series_equal(a, b)


@pytest.mark.parametrize(
    "obj",
    [
        Series([1, 2, 3]),
        Series([1.0, 1.5, 3.2]),
        Series([1.0, 1.5, np.nan]),
        Series([1.0, 1.5, 3.2], index=[1.5, 1.1, 3.3]),
        Series(["a", "b", "c"]),
        Series(["a", np.nan, "c"]),
        Series(["a", None, "c"]),
        Series([True, False, True]),
        DataFrame({"x": ["a", "b", "c"], "y": [1, 2, 3]}),
        DataFrame(np.full((10, 4), np.nan)),
        DataFrame(
            {
                "A": [0.0, 1.0, 2.0, 3.0, 4.0],
                "B": [0.0, 1.0, 0.0, 1.0, 0.0],
                "C": Index(["foo1", "foo2", "foo3", "foo4", "foo5"], dtype=object),
                "D": pd.date_range("20130101", periods=5),
            }
        ),
        DataFrame(range(5), index=pd.date_range("2020-01-01", periods=5)),
        Series(range(5), index=pd.date_range("2020-01-01", periods=5)),
        Series(period_range("2020-01-01", periods=10, freq="D")),
        Series(pd.date_range("20130101", periods=3, tz="US/Eastern")),
    ],
)
def test_hash_pandas_object_diff_index_non_empty(obj):
    a = hash_pandas_object(obj, index=True)
    b = hash_pandas_object(obj, index=False)
    assert not (a == b).all()


@pytest.mark.parametrize(
    "obj",
    [
        Index([1, 2, 3]),
        Index([True, False, True]),
        timedelta_range("1 day", periods=2),
        period_range("2020-01-01", freq="D", periods=2),
        MultiIndex.from_product(
            [range(5), ["foo", "bar", "baz"], pd.date_range("20130101", periods=2)]
        ),
        MultiIndex.from_product([pd.CategoricalIndex(list("aabc")), range(3)]),
    ],
)
def test_hash_pandas_index(obj, index):
    a = hash_pandas_object(obj, index=index)
    b = hash_pandas_object(obj, index=index)
    tm.assert_series_equal(a, b)


def test_hash_pandas_series(series, index):
    a = hash_pandas_object(series, index=index)
    b = hash_pandas_object(series, index=index)
    tm.assert_series_equal(a, b)


def test_hash_pandas_series_diff_index(series):
    a = hash_pandas_object(series, index=True)
    b = hash_pandas_object(series, index=False)
    assert not (a == b).all()


@pytest.mark.parametrize(
    "obj", [Series([], dtype="float64"), Series([], dtype="object"), Index([])]
)
def test_hash_pandas_empty_object(obj, index):
    # These are by-definition the same with
    # or without the index as the data is empty.
    a = hash_pandas_object(obj, index=index)
    b = hash_pandas_object(obj, index=index)
    tm.assert_series_equal(a, b)


@pytest.mark.parametrize(
    "s1",
    [
        Series(["a", "b", "c", "d"]),
        Series([1000, 2000, 3000, 4000]),
        Series(pd.date_range(0, periods=4)),
    ],
)
@pytest.mark.parametrize("categorize", [True, False])
def test_categorical_consistency(s1, categorize):
    # see gh-15143
    #
    # Check that categoricals hash consistent with their values,
    # not codes. This should work for categoricals of any dtype.
    s2 = s1.astype("category").cat.set_categories(s1)
    s3 = s2.cat.set_categories(list(reversed(s1)))

    # These should all hash identically.
    h1 = hash_pandas_object(s1, categorize=categorize)
    h2 = hash_pandas_object(s2, categorize=categorize)
    h3 = hash_pandas_object(s3, categorize=categorize)

    tm.assert_series_equal(h1, h2)
    tm.assert_series_equal(h1, h3)


def test_categorical_with_nan_consistency():
    c = pd.Categorical.from_codes(
        [-1, 0, 1, 2, 3, 4], categories=pd.date_range("2012-01-01", periods=5, name="B")
    )
    expected = hash_array(c, categorize=False)

    c = pd.Categorical.from_codes([-1, 0], categories=[pd.Timestamp("2012-01-01")])
    result = hash_array(c, categorize=False)

    assert result[0] in expected
    assert result[1] in expected


def test_pandas_errors():
    msg = "Unexpected type for hashing"
    with pytest.raises(TypeError, match=msg):
        hash_pandas_object(pd.Timestamp("20130101"))


def test_hash_keys():
    # Using different hash keys, should have
    # different hashes for the same data.
    #
    # This only matters for object dtypes.
    obj = Series(list("abc"))

    a = hash_pandas_object(obj, hash_key="9876543210123456")
    b = hash_pandas_object(obj, hash_key="9876543210123465")

    assert (a != b).all()


def test_df_hash_keys():
    # DataFrame version of the test_hash_keys.
    # https://github.com/pandas-dev/pandas/issues/41404
    obj = DataFrame({"x": np.arange(3), "y": list("abc")})

    a = hash_pandas_object(obj, hash_key="9876543210123456")
    b = hash_pandas_object(obj, hash_key="9876543210123465")

    assert (a != b).all()


def test_df_encoding():
    # Check that DataFrame recognizes optional encoding.
    # https://github.com/pandas-dev/pandas/issues/41404
    # https://github.com/pandas-dev/pandas/pull/42049
    obj = DataFrame({"x": np.arange(3), "y": list("a+c")})

    a = hash_pandas_object(obj, encoding="utf8")
    b = hash_pandas_object(obj, encoding="utf7")

    # Note that the "+" is encoded as "+-" in utf-7.
    assert a[0] == b[0]
    assert a[1] != b[1]
    assert a[2] == b[2]


def test_invalid_key():
    # This only matters for object dtypes.
    msg = "key should be a 16-byte string encoded"

    with pytest.raises(ValueError, match=msg):
        hash_pandas_object(Series(list("abc")), hash_key="foo")


def test_already_encoded(index):
    # If already encoded, then ok.
    obj = Series(list("abc")).str.encode("utf8")
    a = hash_pandas_object(obj, index=index)
    b = hash_pandas_object(obj, index=index)
    tm.assert_series_equal(a, b)


def test_alternate_encoding(index):
    obj = Series(list("abc"))
    a = hash_pandas_object(obj, index=index)
    b = hash_pandas_object(obj, index=index)
    tm.assert_series_equal(a, b)


@pytest.mark.parametrize("l_exp", range(8))
@pytest.mark.parametrize("l_add", [0, 1])
def test_same_len_hash_collisions(l_exp, l_add):
    length = 2 ** (l_exp + 8) + l_add
    idx = np.array([str(i) for i in range(length)], dtype=object)

    result = hash_array(idx, "utf8")
    assert not result[0] == result[1]


def test_hash_collisions():
    # Hash collisions are bad.
    #
    # https://github.com/pandas-dev/pandas/issues/14711#issuecomment-264885726
    hashes = [
        "Ingrid-9Z9fKIZmkO7i7Cn51Li34pJm44fgX6DYGBNj3VPlOH50m7HnBlPxfIwFMrcNJNMP6PSgLmwWnInciMWrCSAlLEvt7JkJl4IxiMrVbXSa8ZQoVaq5xoQPjltuJEfwdNlO6jo8qRRHvD8sBEBMQASrRa6TsdaPTPCBo3nwIBpE7YzzmyH0vMBhjQZLx1aCT7faSEx7PgFxQhHdKFWROcysamgy9iVj8DO2Fmwg1NNl93rIAqC3mdqfrCxrzfvIY8aJdzin2cHVzy3QUJxZgHvtUtOLxoqnUHsYbNTeq0xcLXpTZEZCxD4PGubIuCNf32c33M7HFsnjWSEjE2yVdWKhmSVodyF8hFYVmhYnMCztQnJrt3O8ZvVRXd5IKwlLexiSp4h888w7SzAIcKgc3g5XQJf6MlSMftDXm9lIsE1mJNiJEv6uY6pgvC3fUPhatlR5JPpVAHNSbSEE73MBzJrhCAbOLXQumyOXigZuPoME7QgJcBalliQol7YZ9",
        "Tim-b9MddTxOWW2AT1Py6vtVbZwGAmYCjbp89p8mxsiFoVX4FyDOF3wFiAkyQTUgwg9sVqVYOZo09Dh1AzhFHbgij52ylF0SEwgzjzHH8TGY8Lypart4p4onnDoDvVMBa0kdthVGKl6K0BDVGzyOXPXKpmnMF1H6rJzqHJ0HywfwS4XYpVwlAkoeNsiicHkJUFdUAhG229INzvIAiJuAHeJDUoyO4DCBqtoZ5TDend6TK7Y914yHlfH3g1WZu5LksKv68VQHJriWFYusW5e6ZZ6dKaMjTwEGuRgdT66iU5nqWTHRH8WSzpXoCFwGcTOwyuqPSe0fTe21DVtJn1FKj9F9nEnR9xOvJUO7E0piCIF4Ad9yAIDY4DBimpsTfKXCu1vdHpKYerzbndfuFe5AhfMduLYZJi5iAw8qKSwR5h86ttXV0Mc0QmXz8dsRvDgxjXSmupPxBggdlqUlC828hXiTPD7am0yETBV0F3bEtvPiNJfremszcV8NcqAoARMe",
    ]

    # These should be different.
    result1 = hash_array(np.asarray(hashes[0:1], dtype=object), "utf8")
    expected1 = np.array([14963968704024874985], dtype=np.uint64)
    tm.assert_numpy_array_equal(result1, expected1)

    result2 = hash_array(np.asarray(hashes[1:2], dtype=object), "utf8")
    expected2 = np.array([16428432627716348016], dtype=np.uint64)
    tm.assert_numpy_array_equal(result2, expected2)

    result = hash_array(np.asarray(hashes, dtype=object), "utf8")
    tm.assert_numpy_array_equal(result, np.concatenate([expected1, expected2], axis=0))


@pytest.mark.parametrize(
    "data, result_data",
    [
        [[tuple("1"), tuple("2")], [10345501319357378243, 8331063931016360761]],
        [[(1,), (2,)], [9408946347443669104, 3278256261030523334]],
    ],
)
def test_hash_with_tuple(data, result_data):
    # GH#28969 array containing a tuple raises on call to arr.astype(str)
    #  apparently a numpy bug github.com/numpy/numpy/issues/9441

    df = DataFrame({"data": data})
    result = hash_pandas_object(df)
    expected = Series(result_data, dtype=np.uint64)
    tm.assert_series_equal(result, expected)


def test_hashable_tuple_args():
    # require that the elements of such tuples are themselves hashable

    df3 = DataFrame(
        {
            "data": [
                (
                    1,
                    [],
                ),
                (
                    2,
                    {},
                ),
            ]
        }
    )
    with pytest.raises(TypeError, match="unhashable type: 'list'"):
        hash_pandas_object(df3)


def test_hash_object_none_key():
    # https://github.com/pandas-dev/pandas/issues/30887
    result = pd.util.hash_pandas_object(Series(["a", "b"]), hash_key=None)
    expected = Series([4578374827886788867, 17338122309987883691], dtype="uint64")
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\tests\test_heurisch.py ===
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Eq, Ne
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (asinh, cosh, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, tan)
from sympy.functions.special.bessel import (besselj, besselk, bessely, jn)
from sympy.functions.special.error_functions import erf
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And
from sympy.matrices import Matrix
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import simplify
from sympy.integrals.heurisch import components, heurisch, heurisch_wrapper
from sympy.testing.pytest import XFAIL, slow
from sympy.integrals.integrals import integrate
from sympy import S

x, y, z, nu = symbols('x,y,z,nu')
f = Function('f')


def test_components():
    assert components(x*y, x) == {x}
    assert components(1/(x + y), x) == {x}
    assert components(sin(x), x) == {sin(x), x}
    assert components(sin(x)*sqrt(log(x)), x) == \
        {log(x), sin(x), sqrt(log(x)), x}
    assert components(x*sin(exp(x)*y), x) == \
        {sin(y*exp(x)), x, exp(x)}
    assert components(x**Rational(17, 54)/sqrt(sin(x)), x) == \
        {sin(x), x**Rational(1, 54), sqrt(sin(x)), x}

    assert components(f(x), x) == \
        {x, f(x)}
    assert components(Derivative(f(x), x), x) == \
        {x, f(x), Derivative(f(x), x)}
    assert components(f(x)*diff(f(x), x), x) == \
        {x, f(x), Derivative(f(x), x), Derivative(f(x), x)}


def test_issue_10680():
    assert isinstance(integrate(x**log(x**log(x**log(x))),x), Integral)


def test_issue_21166():
    assert integrate(sin(x/sqrt(abs(x))), (x, -1, 1)) == 0


def test_heurisch_polynomials():
    assert heurisch(1, x) == x
    assert heurisch(x, x) == x**2/2
    assert heurisch(x**17, x) == x**18/18
    # For coverage
    assert heurisch_wrapper(y, x) == y*x


def test_heurisch_fractions():
    assert heurisch(1/x, x) == log(x)
    assert heurisch(1/(2 + x), x) == log(x + 2)
    assert heurisch(1/(x + sin(y)), x) == log(x + sin(y))

    # Up to a constant, where C = pi*I*Rational(5, 12), Mathematica gives identical
    # result in the first case. The difference is because SymPy changes
    # signs of expressions without any care.
    # XXX ^ ^ ^ is this still correct?
    assert heurisch(5*x**5/(
        2*x**6 - 5), x) in [5*log(2*x**6 - 5) / 12, 5*log(-2*x**6 + 5) / 12]
    assert heurisch(5*x**5/(2*x**6 + 5), x) == 5*log(2*x**6 + 5) / 12

    assert heurisch(1/x**2, x) == -1/x
    assert heurisch(-1/x**5, x) == 1/(4*x**4)


def test_heurisch_log():
    assert heurisch(log(x), x) == x*log(x) - x
    assert heurisch(log(3*x), x) == -x + x*log(3) + x*log(x)
    assert heurisch(log(x**2), x) in [x*log(x**2) - 2*x, 2*x*log(x) - 2*x]


def test_heurisch_exp():
    assert heurisch(exp(x), x) == exp(x)
    assert heurisch(exp(-x), x) == -exp(-x)
    assert heurisch(exp(17*x), x) == exp(17*x) / 17
    assert heurisch(x*exp(x), x) == x*exp(x) - exp(x)
    assert heurisch(x*exp(x**2), x) == exp(x**2) / 2

    assert heurisch(exp(-x**2), x) is None

    assert heurisch(2**x, x) == 2**x/log(2)
    assert heurisch(x*2**x, x) == x*2**x/log(2) - 2**x*log(2)**(-2)

    assert heurisch(Integral(x**z*y, (y, 1, 2), (z, 2, 3)).function, x) == (x*x**z*y)/(z+1)
    assert heurisch(Sum(x**z, (z, 1, 2)).function, z) == x**z/log(x)

    # https://github.com/sympy/sympy/issues/23707
    anti = -exp(z)/(sqrt(x - y)*exp(z*sqrt(x - y)) - exp(z*sqrt(x - y)))
    assert heurisch(exp(z)*exp(-z*sqrt(x - y)), z) == anti


def test_heurisch_trigonometric():
    assert heurisch(sin(x), x) == -cos(x)
    assert heurisch(pi*sin(x) + 1, x) == x - pi*cos(x)

    assert heurisch(cos(x), x) == sin(x)
    assert heurisch(tan(x), x) in [
        log(1 + tan(x)**2)/2,
        log(tan(x) + I) + I*x,
        log(tan(x) - I) - I*x,
    ]

    assert heurisch(sin(x)*sin(y), x) == -cos(x)*sin(y)
    assert heurisch(sin(x)*sin(y), y) == -cos(y)*sin(x)

    # gives sin(x) in answer when run via setup.py and cos(x) when run via py.test
    assert heurisch(sin(x)*cos(x), x) in [sin(x)**2 / 2, -cos(x)**2 / 2]
    assert heurisch(cos(x)/sin(x), x) == log(sin(x))

    assert heurisch(x*sin(7*x), x) == sin(7*x) / 49 - x*cos(7*x) / 7
    assert heurisch(1/pi/4 * x**2*cos(x), x) == 1/pi/4*(x**2*sin(x) -
                    2*sin(x) + 2*x*cos(x))

    assert heurisch(acos(x/4) * asin(x/4), x) == 2*x - (sqrt(16 - x**2))*asin(x/4) \
        + (sqrt(16 - x**2))*acos(x/4) + x*asin(x/4)*acos(x/4)

    assert heurisch(sin(x)/(cos(x)**2+1), x) == -atan(cos(x)) #fixes issue 13723
    assert heurisch(1/(cos(x)+2), x) == 2*sqrt(3)*atan(sqrt(3)*tan(x/2)/3)/3
    assert heurisch(2*sin(x)*cos(x)/(sin(x)**4 + 1), x) == atan(sqrt(2)*sin(x)
        - 1) - atan(sqrt(2)*sin(x) + 1)

    assert heurisch(1/cosh(x), x) == 2*atan(tanh(x/2))


def test_heurisch_hyperbolic():
    assert heurisch(sinh(x), x) == cosh(x)
    assert heurisch(cosh(x), x) == sinh(x)

    assert heurisch(x*sinh(x), x) == x*cosh(x) - sinh(x)
    assert heurisch(x*cosh(x), x) == x*sinh(x) - cosh(x)

    assert heurisch(
        x*asinh(x/2), x) == x**2*asinh(x/2)/2 + asinh(x/2) - x*sqrt(4 + x**2)/4


def test_heurisch_mixed():
    assert heurisch(sin(x)*exp(x), x) == exp(x)*sin(x)/2 - exp(x)*cos(x)/2
    assert heurisch(sin(x/sqrt(-x)), x) == 2*x*cos(x/sqrt(-x))/sqrt(-x) - 2*sin(x/sqrt(-x))


def test_heurisch_radicals():
    assert heurisch(1/sqrt(x), x) == 2*sqrt(x)
    assert heurisch(1/sqrt(x)**3, x) == -2/sqrt(x)
    assert heurisch(sqrt(x)**3, x) == 2*sqrt(x)**5/5

    assert heurisch(sin(x)*sqrt(cos(x)), x) == -2*sqrt(cos(x))**3/3
    y = Symbol('y')
    assert heurisch(sin(y*sqrt(x)), x) == 2/y**2*sin(y*sqrt(x)) - \
        2*sqrt(x)*cos(y*sqrt(x))/y
    assert heurisch_wrapper(sin(y*sqrt(x)), x) == Piecewise(
        (-2*sqrt(x)*cos(sqrt(x)*y)/y + 2*sin(sqrt(x)*y)/y**2, Ne(y, 0)),
        (0, True))
    y = Symbol('y', positive=True)
    assert heurisch_wrapper(sin(y*sqrt(x)), x) == 2/y**2*sin(y*sqrt(x)) - \
        2*sqrt(x)*cos(y*sqrt(x))/y


def test_heurisch_special():
    assert heurisch(erf(x), x) == x*erf(x) + exp(-x**2)/sqrt(pi)
    assert heurisch(exp(-x**2)*erf(x), x) == sqrt(pi)*erf(x)**2 / 4


def test_heurisch_symbolic_coeffs():
    assert heurisch(1/(x + y), x) == log(x + y)
    assert heurisch(1/(x + sqrt(2)), x) == log(x + sqrt(2))
    assert simplify(diff(heurisch(log(x + y + z), y), y)) == log(x + y + z)


def test_heurisch_symbolic_coeffs_1130():
    y = Symbol('y')
    assert heurisch_wrapper(1/(x**2 + y), x) == Piecewise(
    (log(x - sqrt(-y))/(2*sqrt(-y)) - log(x + sqrt(-y))/(2*sqrt(-y)),
    Ne(y, 0)), (-1/x, True))
    y = Symbol('y', positive=True)
    assert heurisch_wrapper(1/(x**2 + y), x) == (atan(x/sqrt(y))/sqrt(y))


def test_heurisch_hacking():
    assert heurisch(sqrt(1 + 7*x**2), x, hints=[]) == \
        x*sqrt(1 + 7*x**2)/2 + sqrt(7)*asinh(sqrt(7)*x)/14
    assert heurisch(sqrt(1 - 7*x**2), x, hints=[]) == \
        x*sqrt(1 - 7*x**2)/2 + sqrt(7)*asin(sqrt(7)*x)/14

    assert heurisch(1/sqrt(1 + 7*x**2), x, hints=[]) == \
        sqrt(7)*asinh(sqrt(7)*x)/7
    assert heurisch(1/sqrt(1 - 7*x**2), x, hints=[]) == \
        sqrt(7)*asin(sqrt(7)*x)/7

    assert heurisch(exp(-7*x**2), x, hints=[]) == \
        sqrt(7*pi)*erf(sqrt(7)*x)/14

    assert heurisch(1/sqrt(9 - 4*x**2), x, hints=[]) == \
        asin(x*Rational(2, 3))/2

    assert heurisch(1/sqrt(9 + 4*x**2), x, hints=[]) == \
        asinh(x*Rational(2, 3))/2

    assert heurisch(1/sqrt(3*x**2-4), x, hints=[]) == \
           sqrt(3)*log(3*x + sqrt(3)*sqrt(3*x**2 - 4))/3


def test_heurisch_function():
    assert heurisch(f(x), x) is None

@XFAIL
def test_heurisch_function_derivative():
    # TODO: it looks like this used to work just by coincindence and
    # thanks to sloppy implementation. Investigate why this used to
    # work at all and if support for this can be restored.

    df = diff(f(x), x)

    assert heurisch(f(x)*df, x) == f(x)**2/2
    assert heurisch(f(x)**2*df, x) == f(x)**3/3
    assert heurisch(df/f(x), x) == log(f(x))


def test_heurisch_wrapper():
    f = 1/(y + x)
    assert heurisch_wrapper(f, x) == log(x + y)
    f = 1/(y - x)
    assert heurisch_wrapper(f, x) == -log(x - y)
    f = 1/((y - x)*(y + x))
    assert heurisch_wrapper(f, x) == Piecewise(
        (-log(x - y)/(2*y) + log(x + y)/(2*y), Ne(y, 0)), (1/x, True))
    # issue 6926
    f = sqrt(x**2/((y - x)*(y + x)))
    assert heurisch_wrapper(f, x) == x*sqrt(-x**2/(x**2 - y**2)) \
    - y**2*sqrt(-x**2/(x**2 - y**2))/x


def test_issue_3609():
    assert heurisch(1/(x * (1 + log(x)**2)), x) == atan(log(x))

### These are examples from the Poor Man's Integrator
### http://www-sop.inria.fr/cafe/Manuel.Bronstein/pmint/examples/


def test_pmint_rat():
    # TODO: heurisch() is off by a constant: -3/4. Possibly different permutation
    # would give the optimal result?

    def drop_const(expr, x):
        if expr.is_Add:
            return Add(*[ arg for arg in expr.args if arg.has(x) ])
        else:
            return expr

    f = (x**7 - 24*x**4 - 4*x**2 + 8*x - 8)/(x**8 + 6*x**6 + 12*x**4 + 8*x**2)
    g = (4 + 8*x**2 + 6*x + 3*x**3)/(x**5 + 4*x**3 + 4*x) + log(x)

    assert drop_const(ratsimp(heurisch(f, x)), x) == g


def test_pmint_trig():
    f = (x - tan(x)) / tan(x)**2 + tan(x)
    g = -x**2/2 - x/tan(x) + log(tan(x)**2 + 1)/2

    assert heurisch(f, x) == g


def test_pmint_logexp():
    f = (1 + x + x*exp(x))*(x + log(x) + exp(x) - 1)/(x + log(x) + exp(x))**2/x
    g = log(x + exp(x) + log(x)) + 1/(x + exp(x) + log(x))

    assert ratsimp(heurisch(f, x)) == g


def test_pmint_erf():
    f = exp(-x**2)*erf(x)/(erf(x)**3 - erf(x)**2 - erf(x) + 1)
    g = sqrt(pi)*log(erf(x) - 1)/8 - sqrt(pi)*log(erf(x) + 1)/8 - sqrt(pi)/(4*erf(x) - 4)

    assert ratsimp(heurisch(f, x)) == g


def test_pmint_LambertW():
    f = LambertW(x)
    g = x*LambertW(x) - x + x/LambertW(x)

    assert heurisch(f, x) == g


def test_pmint_besselj():
    f = besselj(nu + 1, x)/besselj(nu, x)
    g = nu*log(x) - log(besselj(nu, x))

    assert heurisch(f, x) == g

    f = (nu*besselj(nu, x) - x*besselj(nu + 1, x))/x
    g = besselj(nu, x)

    assert heurisch(f, x) == g

    f = jn(nu + 1, x)/jn(nu, x)
    g = nu*log(x) - log(jn(nu, x))

    assert heurisch(f, x) == g


@slow
def test_pmint_bessel_products():
    f = x*besselj(nu, x)*bessely(nu, 2*x)
    g = -2*x*besselj(nu, x)*bessely(nu - 1, 2*x)/3 + x*besselj(nu - 1, x)*bessely(nu, 2*x)/3

    assert heurisch(f, x) == g

    f = x*besselj(nu, x)*besselk(nu, 2*x)
    g = -2*x*besselj(nu, x)*besselk(nu - 1, 2*x)/5 - x*besselj(nu - 1, x)*besselk(nu, 2*x)/5

    assert heurisch(f, x) == g


def test_pmint_WrightOmega():
    def omega(x):
        return LambertW(exp(x))

    f = (1 + omega(x) * (2 + cos(omega(x)) * (x + omega(x))))/(1 + omega(x))/(x + omega(x))
    g = log(x + LambertW(exp(x))) + sin(LambertW(exp(x)))

    assert heurisch(f, x) == g


def test_RR():
    # Make sure the algorithm does the right thing if the ring is RR. See
    # issue 8685.
    assert heurisch(sqrt(1 + 0.25*x**2), x, hints=[]) == \
        0.5*x*sqrt(0.25*x**2 + 1) + 1.0*asinh(0.5*x)

# TODO: convert the rest of PMINT tests:
# Airy functions
# f = (x - AiryAi(x)*AiryAi(1, x)) / (x**2 - AiryAi(x)**2)
# g = Rational(1,2)*ln(x + AiryAi(x)) + Rational(1,2)*ln(x - AiryAi(x))
# f = x**2 * AiryAi(x)
# g = -AiryAi(x) + AiryAi(1, x)*x
# Whittaker functions
# f = WhittakerW(mu + 1, nu, x) / (WhittakerW(mu, nu, x) * x)
# g = x/2 - mu*ln(x) - ln(WhittakerW(mu, nu, x))


def test_issue_22527():
    t, R = symbols(r't R')
    z = Function('z')(t)
    def f(x):
        return x/sqrt(R**2 - x**2)
    Uz = integrate(f(z), z)
    Ut = integrate(f(t), t)
    assert Ut == Uz.subs(z, t)


def test_heurisch_complex_erf_issue_26338():
    r = symbols('r', real=True)
    a = sqrt(pi)*erf((1 + I)/2)/2
    assert integrate(exp(-I*r**2/2), (r, 0, 1)) == a - I*a

    a = exp(-x**2/(2*(2 - I)**2))
    assert heurisch(a, x, hints=[]) is None  # None, not a wrong soln
    a = exp(-r**2/(2*(2 - I)**2))
    assert heurisch(a, r, hints=[]) is None
    a = sqrt(pi)*erf((1 + I)/2)/2
    assert integrate(exp(-I*x**2/2), (x, 0, 1)) == a - I*a


def test_issue_15498():
    Z0 = Function('Z0')
    k01, k10, t, s= symbols('k01 k10 t s', real=True, positive=True)
    m = Matrix([[exp(-k10*t)]])
    _83 = Rational(83, 100)  # 0.83 works, too
    [a, b, c, d, e, f, g] = [100, 0.5, _83, 50, 0.6, 2, 120]
    AIF_btf = a*(d*e*(1 - exp(-(t - b)/e)) + f*g*(1 - exp(-(t - b)/g)))
    AIF_atf = a*(d*e*exp(-(t - b)/e)*(exp((c - b)/e) - 1
        ) + f*g*exp(-(t - b)/g)*(exp((c - b)/g) - 1))
    AIF_sym = Piecewise((0, t < b), (AIF_btf, And(b <= t, t < c)), (AIF_atf, c <= t))
    aif_eq = Eq(Z0(t), AIF_sym)
    f_vec = Matrix([[k01*Z0(t)]])
    integrand = m*m.subs(t, s)**-1*f_vec.subs(aif_eq.lhs, aif_eq.rhs).subs(t, s)
    solution = integrate(integrand[0], (s, 0, t))
    assert solution is not None  # does not hang and takes less than 10 s


@slow
def test_heurisch_issue_26930():
    integrand = x**Rational(4, 3)*log(x)
    anti = 3*x**(S(7)/3)*log(x)/7 - 9*x**(S(7)/3)/49
    assert heurisch(integrand, x) == anti
    assert integrate(integrand, x) == anti
    assert integrate(integrand, (x, 0, 1)) == -S(9)/49


def test_heurisch_issue_26922():

    a, b, x = symbols("a, b, x", real=True, positive=True)
    C = symbols("C", real=True)
    i1 = -C*x*exp(-a*x**2 - sqrt(b)*x)
    i2 = C*x*exp(-a*x**2 + sqrt(b)*x)
    i = Integral(i1, x) + Integral(i2, x)
    res = (
        -C*exp(-a*x**2)*exp(sqrt(b)*x)/(2*a)
        + C*exp(-a*x**2)*exp(-sqrt(b)*x)/(2*a)
        + sqrt(pi)*C*sqrt(b)*exp(b/(4*a))*erf(sqrt(a)*x - sqrt(b)/(2*sqrt(a)))/(4*a**(S(3)/2))
        + sqrt(pi)*C*sqrt(b)*exp(b/(4*a))*erf(sqrt(a)*x + sqrt(b)/(2*sqrt(a)))/(4*a**(S(3)/2))
    )

    assert i.doit(heurisch=False).expand() == res

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\test_clipboard.py ===
from textwrap import dedent

import numpy as np
import pytest

from pandas.errors import (
    PyperclipException,
    PyperclipWindowsException,
)

import pandas as pd
from pandas import (
    NA,
    DataFrame,
    Series,
    get_option,
    read_clipboard,
)
import pandas._testing as tm

from pandas.io.clipboard import (
    CheckedCall,
    _stringifyText,
    init_qt_clipboard,
)


def build_kwargs(sep, excel):
    kwargs = {}
    if excel != "default":
        kwargs["excel"] = excel
    if sep != "default":
        kwargs["sep"] = sep
    return kwargs


@pytest.fixture(
    params=[
        "delims",
        "utf8",
        "utf16",
        "string",
        "long",
        "nonascii",
        "colwidth",
        "mixed",
        "float",
        "int",
    ]
)
def df(request):
    data_type = request.param

    if data_type == "delims":
        return DataFrame({"a": ['"a,\t"b|c', "d\tef`"], "b": ["hi'j", "k''lm"]})
    elif data_type == "utf8":
        return DataFrame({"a": ["µasd", "Ωœ∑`"], "b": ["øπ∆˚¬", "œ∑`®"]})
    elif data_type == "utf16":
        return DataFrame(
            {"a": ["\U0001f44d\U0001f44d", "\U0001f44d\U0001f44d"], "b": ["abc", "def"]}
        )
    elif data_type == "string":
        return DataFrame(
            np.array([f"i-{i}" for i in range(15)]).reshape(5, 3), columns=list("abc")
        )
    elif data_type == "long":
        max_rows = get_option("display.max_rows")
        return DataFrame(
            np.random.default_rng(2).integers(0, 10, size=(max_rows + 1, 3)),
            columns=list("abc"),
        )
    elif data_type == "nonascii":
        return DataFrame({"en": "in English".split(), "es": "en español".split()})
    elif data_type == "colwidth":
        _cw = get_option("display.max_colwidth") + 1
        return DataFrame(
            np.array(["x" * _cw for _ in range(15)]).reshape(5, 3), columns=list("abc")
        )
    elif data_type == "mixed":
        return DataFrame(
            {
                "a": np.arange(1.0, 6.0) + 0.01,
                "b": np.arange(1, 6).astype(np.int64),
                "c": list("abcde"),
            }
        )
    elif data_type == "float":
        return DataFrame(np.random.default_rng(2).random((5, 3)), columns=list("abc"))
    elif data_type == "int":
        return DataFrame(
            np.random.default_rng(2).integers(0, 10, (5, 3)), columns=list("abc")
        )
    else:
        raise ValueError


@pytest.fixture
def mock_ctypes(monkeypatch):
    """
    Mocks WinError to help with testing the clipboard.
    """

    def _mock_win_error():
        return "Window Error"

    # Set raising to False because WinError won't exist on non-windows platforms
    with monkeypatch.context() as m:
        m.setattr("ctypes.WinError", _mock_win_error, raising=False)
        yield


@pytest.mark.usefixtures("mock_ctypes")
def test_checked_call_with_bad_call(monkeypatch):
    """
    Give CheckCall a function that returns a falsey value and
    mock get_errno so it returns false so an exception is raised.
    """

    def _return_false():
        return False

    monkeypatch.setattr("pandas.io.clipboard.get_errno", lambda: True)
    msg = f"Error calling {_return_false.__name__} \\(Window Error\\)"

    with pytest.raises(PyperclipWindowsException, match=msg):
        CheckedCall(_return_false)()


@pytest.mark.usefixtures("mock_ctypes")
def test_checked_call_with_valid_call(monkeypatch):
    """
    Give CheckCall a function that returns a truthy value and
    mock get_errno so it returns true so an exception is not raised.
    The function should return the results from _return_true.
    """

    def _return_true():
        return True

    monkeypatch.setattr("pandas.io.clipboard.get_errno", lambda: False)

    # Give CheckedCall a callable that returns a truthy value s
    checked_call = CheckedCall(_return_true)
    assert checked_call() is True


@pytest.mark.parametrize(
    "text",
    [
        "String_test",
        True,
        1,
        1.0,
        1j,
    ],
)
def test_stringify_text(text):
    valid_types = (str, int, float, bool)

    if isinstance(text, valid_types):
        result = _stringifyText(text)
        assert result == str(text)
    else:
        msg = (
            "only str, int, float, and bool values "
            f"can be copied to the clipboard, not {type(text).__name__}"
        )
        with pytest.raises(PyperclipException, match=msg):
            _stringifyText(text)


@pytest.fixture
def set_pyqt_clipboard(monkeypatch):
    qt_cut, qt_paste = init_qt_clipboard()
    with monkeypatch.context() as m:
        m.setattr(pd.io.clipboard, "clipboard_set", qt_cut)
        m.setattr(pd.io.clipboard, "clipboard_get", qt_paste)
        yield


@pytest.fixture
def clipboard(qapp):
    clip = qapp.clipboard()
    yield clip
    clip.clear()


@pytest.mark.single_cpu
@pytest.mark.clipboard
@pytest.mark.usefixtures("set_pyqt_clipboard")
@pytest.mark.usefixtures("clipboard")
class TestClipboard:
    # Test that default arguments copy as tab delimited
    # Test that explicit delimiters are respected
    @pytest.mark.parametrize("sep", [None, "\t", ",", "|"])
    @pytest.mark.parametrize("encoding", [None, "UTF-8", "utf-8", "utf8"])
    def test_round_trip_frame_sep(self, df, sep, encoding):
        df.to_clipboard(excel=None, sep=sep, encoding=encoding)
        result = read_clipboard(sep=sep or "\t", index_col=0, encoding=encoding)
        tm.assert_frame_equal(df, result)

    # Test white space separator
    def test_round_trip_frame_string(self, df):
        df.to_clipboard(excel=False, sep=None)
        result = read_clipboard()
        assert df.to_string() == result.to_string()
        assert df.shape == result.shape

    # Two character separator is not supported in to_clipboard
    # Test that multi-character separators are not silently passed
    def test_excel_sep_warning(self, df):
        with tm.assert_produces_warning(
            UserWarning,
            match="to_clipboard in excel mode requires a single character separator.",
            check_stacklevel=False,
        ):
            df.to_clipboard(excel=True, sep=r"\t")

    # Separator is ignored when excel=False and should produce a warning
    def test_copy_delim_warning(self, df):
        with tm.assert_produces_warning():
            df.to_clipboard(excel=False, sep="\t")

    # Tests that the default behavior of to_clipboard is tab
    # delimited and excel="True"
    @pytest.mark.parametrize("sep", ["\t", None, "default"])
    @pytest.mark.parametrize("excel", [True, None, "default"])
    def test_clipboard_copy_tabs_default(self, sep, excel, df, clipboard):
        kwargs = build_kwargs(sep, excel)
        df.to_clipboard(**kwargs)
        assert clipboard.text() == df.to_csv(sep="\t")

    # Tests reading of white space separated tables
    @pytest.mark.parametrize("sep", [None, "default"])
    def test_clipboard_copy_strings(self, sep, df):
        kwargs = build_kwargs(sep, False)
        df.to_clipboard(**kwargs)
        result = read_clipboard(sep=r"\s+")
        assert result.to_string() == df.to_string()
        assert df.shape == result.shape

    def test_read_clipboard_infer_excel(self, clipboard):
        # gh-19010: avoid warnings
        clip_kwargs = {"engine": "python"}

        text = dedent(
            """
            John James\tCharlie Mingus
            1\t2
            4\tHarry Carney
            """.strip()
        )
        clipboard.setText(text)
        df = read_clipboard(**clip_kwargs)

        # excel data is parsed correctly
        assert df.iloc[1, 1] == "Harry Carney"

        # having diff tab counts doesn't trigger it
        text = dedent(
            """
            a\t b
            1  2
            3  4
            """.strip()
        )
        clipboard.setText(text)
        res = read_clipboard(**clip_kwargs)

        text = dedent(
            """
            a  b
            1  2
            3  4
            """.strip()
        )
        clipboard.setText(text)
        exp = read_clipboard(**clip_kwargs)

        tm.assert_frame_equal(res, exp)

    def test_infer_excel_with_nulls(self, clipboard):
        # GH41108
        text = "col1\tcol2\n1\tred\n\tblue\n2\tgreen"

        clipboard.setText(text)
        df = read_clipboard()
        df_expected = DataFrame(
            data={"col1": [1, None, 2], "col2": ["red", "blue", "green"]}
        )

        # excel data is parsed correctly
        tm.assert_frame_equal(df, df_expected)

    @pytest.mark.parametrize(
        "multiindex",
        [
            (  # Can't use `dedent` here as it will remove the leading `\t`
                "\n".join(
                    [
                        "\t\t\tcol1\tcol2",
                        "A\t0\tTrue\t1\tred",
                        "A\t1\tTrue\t\tblue",
                        "B\t0\tFalse\t2\tgreen",
                    ]
                ),
                [["A", "A", "B"], [0, 1, 0], [True, True, False]],
            ),
            (
                "\n".join(
                    ["\t\tcol1\tcol2", "A\t0\t1\tred", "A\t1\t\tblue", "B\t0\t2\tgreen"]
                ),
                [["A", "A", "B"], [0, 1, 0]],
            ),
        ],
    )
    def test_infer_excel_with_multiindex(self, clipboard, multiindex):
        # GH41108

        clipboard.setText(multiindex[0])
        df = read_clipboard()
        df_expected = DataFrame(
            data={"col1": [1, None, 2], "col2": ["red", "blue", "green"]},
            index=multiindex[1],
        )

        # excel data is parsed correctly
        tm.assert_frame_equal(df, df_expected)

    def test_invalid_encoding(self, df):
        msg = "clipboard only supports utf-8 encoding"
        # test case for testing invalid encoding
        with pytest.raises(ValueError, match=msg):
            df.to_clipboard(encoding="ascii")
        with pytest.raises(NotImplementedError, match=msg):
            read_clipboard(encoding="ascii")

    @pytest.mark.parametrize("data", ["\U0001f44d...", "Ωœ∑`...", "abcd..."])
    def test_raw_roundtrip(self, data):
        # PR #25040 wide unicode wasn't copied correctly on PY3 on windows
        df = DataFrame({"data": [data]})
        df.to_clipboard()
        result = read_clipboard()
        tm.assert_frame_equal(df, result)

    @pytest.mark.parametrize("engine", ["c", "python"])
    def test_read_clipboard_dtype_backend(
        self, clipboard, string_storage, dtype_backend, engine, using_infer_string
    ):
        # GH#50502
        if dtype_backend == "pyarrow":
            pa = pytest.importorskip("pyarrow")
            if engine == "c" and string_storage == "pyarrow":
                # TODO avoid this exception?
                string_dtype = pd.ArrowDtype(pa.large_string())
            else:
                string_dtype = pd.ArrowDtype(pa.string())
        else:
            string_dtype = pd.StringDtype(string_storage)

        text = """a,b,c,d,e,f,g,h,i
x,1,4.0,x,2,4.0,,True,False
y,2,5.0,,,,,False,"""
        clipboard.setText(text)

        with pd.option_context("mode.string_storage", string_storage):
            result = read_clipboard(sep=",", dtype_backend=dtype_backend, engine=engine)

        expected = DataFrame(
            {
                "a": Series(["x", "y"], dtype=string_dtype),
                "b": Series([1, 2], dtype="Int64"),
                "c": Series([4.0, 5.0], dtype="Float64"),
                "d": Series(["x", None], dtype=string_dtype),
                "e": Series([2, NA], dtype="Int64"),
                "f": Series([4.0, NA], dtype="Float64"),
                "g": Series([NA, NA], dtype="Int64"),
                "h": Series([True, False], dtype="boolean"),
                "i": Series([False, NA], dtype="boolean"),
            }
        )
        if dtype_backend == "pyarrow":
            from pandas.arrays import ArrowExtensionArray

            expected = DataFrame(
                {
                    col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True))
                    for col in expected.columns
                }
            )
            expected["g"] = ArrowExtensionArray(pa.array([None, None]))

        if using_infer_string:
            expected.columns = expected.columns.astype(
                pd.StringDtype(string_storage, na_value=np.nan)
            )

        tm.assert_frame_equal(result, expected)

    def test_invalid_dtype_backend(self):
        msg = (
            "dtype_backend numpy is invalid, only 'numpy_nullable' and "
            "'pyarrow' are allowed."
        )
        with pytest.raises(ValueError, match=msg):
            read_clipboard(dtype_backend="numpy")

    def test_to_clipboard_pos_args_deprecation(self):
        # GH-54229
        df = DataFrame({"a": [1, 2, 3]})
        msg = (
            r"Starting with pandas version 3.0 all arguments of to_clipboard "
            r"will be keyword-only."
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            df.to_clipboard(True, None)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_rename.py ===
from collections import ChainMap
import inspect

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    merge,
)
import pandas._testing as tm


class TestRename:
    def test_rename_signature(self):
        sig = inspect.signature(DataFrame.rename)
        parameters = set(sig.parameters)
        assert parameters == {
            "self",
            "mapper",
            "index",
            "columns",
            "axis",
            "inplace",
            "copy",
            "level",
            "errors",
        }

    def test_rename_mi(self, frame_or_series):
        obj = frame_or_series(
            [11, 21, 31],
            index=MultiIndex.from_tuples([("A", x) for x in ["a", "B", "c"]]),
        )
        obj.rename(str.lower)

    def test_rename(self, float_frame):
        mapping = {"A": "a", "B": "b", "C": "c", "D": "d"}

        renamed = float_frame.rename(columns=mapping)
        renamed2 = float_frame.rename(columns=str.lower)

        tm.assert_frame_equal(renamed, renamed2)
        tm.assert_frame_equal(
            renamed2.rename(columns=str.upper), float_frame, check_names=False
        )

        # index
        data = {"A": {"foo": 0, "bar": 1}}

        df = DataFrame(data)
        renamed = df.rename(index={"foo": "bar", "bar": "foo"})
        tm.assert_index_equal(renamed.index, Index(["bar", "foo"]))

        renamed = df.rename(index=str.upper)
        tm.assert_index_equal(renamed.index, Index(["FOO", "BAR"]))

        # have to pass something
        with pytest.raises(TypeError, match="must pass an index to rename"):
            float_frame.rename()

        # partial columns
        renamed = float_frame.rename(columns={"C": "foo", "D": "bar"})
        tm.assert_index_equal(renamed.columns, Index(["A", "B", "foo", "bar"]))

        # other axis
        renamed = float_frame.T.rename(index={"C": "foo", "D": "bar"})
        tm.assert_index_equal(renamed.index, Index(["A", "B", "foo", "bar"]))

        # index with name
        index = Index(["foo", "bar"], name="name")
        renamer = DataFrame(data, index=index)
        renamed = renamer.rename(index={"foo": "bar", "bar": "foo"})
        tm.assert_index_equal(renamed.index, Index(["bar", "foo"], name="name"))
        assert renamed.index.name == renamer.index.name

    @pytest.mark.parametrize(
        "args,kwargs",
        [
            ((ChainMap({"A": "a"}, {"B": "b"}),), {"axis": "columns"}),
            ((), {"columns": ChainMap({"A": "a"}, {"B": "b"})}),
        ],
    )
    def test_rename_chainmap(self, args, kwargs):
        # see gh-23859
        colAData = range(1, 11)
        colBdata = np.random.default_rng(2).standard_normal(10)

        df = DataFrame({"A": colAData, "B": colBdata})
        result = df.rename(*args, **kwargs)

        expected = DataFrame({"a": colAData, "b": colBdata})
        tm.assert_frame_equal(result, expected)

    def test_rename_multiindex(self):
        tuples_index = [("foo1", "bar1"), ("foo2", "bar2")]
        tuples_columns = [("fizz1", "buzz1"), ("fizz2", "buzz2")]
        index = MultiIndex.from_tuples(tuples_index, names=["foo", "bar"])
        columns = MultiIndex.from_tuples(tuples_columns, names=["fizz", "buzz"])
        df = DataFrame([(0, 0), (1, 1)], index=index, columns=columns)

        #
        # without specifying level -> across all levels

        renamed = df.rename(
            index={"foo1": "foo3", "bar2": "bar3"},
            columns={"fizz1": "fizz3", "buzz2": "buzz3"},
        )
        new_index = MultiIndex.from_tuples(
            [("foo3", "bar1"), ("foo2", "bar3")], names=["foo", "bar"]
        )
        new_columns = MultiIndex.from_tuples(
            [("fizz3", "buzz1"), ("fizz2", "buzz3")], names=["fizz", "buzz"]
        )
        tm.assert_index_equal(renamed.index, new_index)
        tm.assert_index_equal(renamed.columns, new_columns)
        assert renamed.index.names == df.index.names
        assert renamed.columns.names == df.columns.names

        #
        # with specifying a level (GH13766)

        # dict
        new_columns = MultiIndex.from_tuples(
            [("fizz3", "buzz1"), ("fizz2", "buzz2")], names=["fizz", "buzz"]
        )
        renamed = df.rename(columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level=0)
        tm.assert_index_equal(renamed.columns, new_columns)
        renamed = df.rename(columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level="fizz")
        tm.assert_index_equal(renamed.columns, new_columns)

        new_columns = MultiIndex.from_tuples(
            [("fizz1", "buzz1"), ("fizz2", "buzz3")], names=["fizz", "buzz"]
        )
        renamed = df.rename(columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level=1)
        tm.assert_index_equal(renamed.columns, new_columns)
        renamed = df.rename(columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level="buzz")
        tm.assert_index_equal(renamed.columns, new_columns)

        # function
        func = str.upper
        new_columns = MultiIndex.from_tuples(
            [("FIZZ1", "buzz1"), ("FIZZ2", "buzz2")], names=["fizz", "buzz"]
        )
        renamed = df.rename(columns=func, level=0)
        tm.assert_index_equal(renamed.columns, new_columns)
        renamed = df.rename(columns=func, level="fizz")
        tm.assert_index_equal(renamed.columns, new_columns)

        new_columns = MultiIndex.from_tuples(
            [("fizz1", "BUZZ1"), ("fizz2", "BUZZ2")], names=["fizz", "buzz"]
        )
        renamed = df.rename(columns=func, level=1)
        tm.assert_index_equal(renamed.columns, new_columns)
        renamed = df.rename(columns=func, level="buzz")
        tm.assert_index_equal(renamed.columns, new_columns)

        # index
        new_index = MultiIndex.from_tuples(
            [("foo3", "bar1"), ("foo2", "bar2")], names=["foo", "bar"]
        )
        renamed = df.rename(index={"foo1": "foo3", "bar2": "bar3"}, level=0)
        tm.assert_index_equal(renamed.index, new_index)

    def test_rename_nocopy(self, float_frame, using_copy_on_write, warn_copy_on_write):
        renamed = float_frame.rename(columns={"C": "foo"}, copy=False)

        assert np.shares_memory(renamed["foo"]._values, float_frame["C"]._values)

        with tm.assert_cow_warning(warn_copy_on_write):
            renamed.loc[:, "foo"] = 1.0
        if using_copy_on_write:
            assert not (float_frame["C"] == 1.0).all()
        else:
            assert (float_frame["C"] == 1.0).all()

    def test_rename_inplace(self, float_frame):
        float_frame.rename(columns={"C": "foo"})
        assert "C" in float_frame
        assert "foo" not in float_frame

        c_values = float_frame["C"]
        float_frame = float_frame.copy()
        return_value = float_frame.rename(columns={"C": "foo"}, inplace=True)
        assert return_value is None

        assert "C" not in float_frame
        assert "foo" in float_frame
        # GH 44153
        # Used to be id(float_frame["foo"]) != c_id, but flaky in the CI
        assert float_frame["foo"] is not c_values

    def test_rename_bug(self):
        # GH 5344
        # rename set ref_locs, and set_index was not resetting
        df = DataFrame({0: ["foo", "bar"], 1: ["bah", "bas"], 2: [1, 2]})
        df = df.rename(columns={0: "a"})
        df = df.rename(columns={1: "b"})
        df = df.set_index(["a", "b"])
        df.columns = ["2001-01-01"]
        expected = DataFrame(
            [[1], [2]],
            index=MultiIndex.from_tuples(
                [("foo", "bah"), ("bar", "bas")], names=["a", "b"]
            ),
            columns=["2001-01-01"],
        )
        tm.assert_frame_equal(df, expected)

    def test_rename_bug2(self):
        # GH 19497
        # rename was changing Index to MultiIndex if Index contained tuples

        df = DataFrame(data=np.arange(3), index=[(0, 0), (1, 1), (2, 2)], columns=["a"])
        df = df.rename({(1, 1): (5, 4)}, axis="index")
        expected = DataFrame(
            data=np.arange(3), index=[(0, 0), (5, 4), (2, 2)], columns=["a"]
        )
        tm.assert_frame_equal(df, expected)

    def test_rename_errors_raises(self):
        df = DataFrame(columns=["A", "B", "C", "D"])
        with pytest.raises(KeyError, match="'E'] not found in axis"):
            df.rename(columns={"A": "a", "E": "e"}, errors="raise")

    @pytest.mark.parametrize(
        "mapper, errors, expected_columns",
        [
            ({"A": "a", "E": "e"}, "ignore", ["a", "B", "C", "D"]),
            ({"A": "a"}, "raise", ["a", "B", "C", "D"]),
            (str.lower, "raise", ["a", "b", "c", "d"]),
        ],
    )
    def test_rename_errors(self, mapper, errors, expected_columns):
        # GH 13473
        # rename now works with errors parameter
        df = DataFrame(columns=["A", "B", "C", "D"])
        result = df.rename(columns=mapper, errors=errors)
        expected = DataFrame(columns=expected_columns)
        tm.assert_frame_equal(result, expected)

    def test_rename_objects(self, float_string_frame):
        renamed = float_string_frame.rename(columns=str.upper)

        assert "FOO" in renamed
        assert "foo" not in renamed

    def test_rename_axis_style(self):
        # https://github.com/pandas-dev/pandas/issues/12392
        df = DataFrame({"A": [1, 2], "B": [1, 2]}, index=["X", "Y"])
        expected = DataFrame({"a": [1, 2], "b": [1, 2]}, index=["X", "Y"])

        result = df.rename(str.lower, axis=1)
        tm.assert_frame_equal(result, expected)

        result = df.rename(str.lower, axis="columns")
        tm.assert_frame_equal(result, expected)

        result = df.rename({"A": "a", "B": "b"}, axis=1)
        tm.assert_frame_equal(result, expected)

        result = df.rename({"A": "a", "B": "b"}, axis="columns")
        tm.assert_frame_equal(result, expected)

        # Index
        expected = DataFrame({"A": [1, 2], "B": [1, 2]}, index=["x", "y"])
        result = df.rename(str.lower, axis=0)
        tm.assert_frame_equal(result, expected)

        result = df.rename(str.lower, axis="index")
        tm.assert_frame_equal(result, expected)

        result = df.rename({"X": "x", "Y": "y"}, axis=0)
        tm.assert_frame_equal(result, expected)

        result = df.rename({"X": "x", "Y": "y"}, axis="index")
        tm.assert_frame_equal(result, expected)

        result = df.rename(mapper=str.lower, axis="index")
        tm.assert_frame_equal(result, expected)

    def test_rename_mapper_multi(self):
        df = DataFrame({"A": ["a", "b"], "B": ["c", "d"], "C": [1, 2]}).set_index(
            ["A", "B"]
        )
        result = df.rename(str.upper)
        expected = df.rename(index=str.upper)
        tm.assert_frame_equal(result, expected)

    def test_rename_positional_named(self):
        # https://github.com/pandas-dev/pandas/issues/12392
        df = DataFrame({"a": [1, 2], "b": [1, 2]}, index=["X", "Y"])
        result = df.rename(index=str.lower, columns=str.upper)
        expected = DataFrame({"A": [1, 2], "B": [1, 2]}, index=["x", "y"])
        tm.assert_frame_equal(result, expected)

    def test_rename_axis_style_raises(self):
        # see gh-12392
        df = DataFrame({"A": [1, 2], "B": [1, 2]}, index=["0", "1"])

        # Named target and axis
        over_spec_msg = "Cannot specify both 'axis' and any of 'index' or 'columns'"
        with pytest.raises(TypeError, match=over_spec_msg):
            df.rename(index=str.lower, axis=1)

        with pytest.raises(TypeError, match=over_spec_msg):
            df.rename(index=str.lower, axis="columns")

        with pytest.raises(TypeError, match=over_spec_msg):
            df.rename(columns=str.lower, axis="columns")

        with pytest.raises(TypeError, match=over_spec_msg):
            df.rename(index=str.lower, axis=0)

        # Multiple targets and axis
        with pytest.raises(TypeError, match=over_spec_msg):
            df.rename(str.lower, index=str.lower, axis="columns")

        # Too many targets
        over_spec_msg = "Cannot specify both 'mapper' and any of 'index' or 'columns'"
        with pytest.raises(TypeError, match=over_spec_msg):
            df.rename(str.lower, index=str.lower, columns=str.lower)

        # Duplicates
        with pytest.raises(TypeError, match="multiple values"):
            df.rename(id, mapper=id)

    def test_rename_positional_raises(self):
        # GH 29136
        df = DataFrame(columns=["A", "B"])
        msg = r"rename\(\) takes from 1 to 2 positional arguments"

        with pytest.raises(TypeError, match=msg):
            df.rename(None, str.lower)

    def test_rename_no_mappings_raises(self):
        # GH 29136
        df = DataFrame([[1]])
        msg = "must pass an index to rename"
        with pytest.raises(TypeError, match=msg):
            df.rename()

        with pytest.raises(TypeError, match=msg):
            df.rename(None, index=None)

        with pytest.raises(TypeError, match=msg):
            df.rename(None, columns=None)

        with pytest.raises(TypeError, match=msg):
            df.rename(None, columns=None, index=None)

    def test_rename_mapper_and_positional_arguments_raises(self):
        # GH 29136
        df = DataFrame([[1]])
        msg = "Cannot specify both 'mapper' and any of 'index' or 'columns'"
        with pytest.raises(TypeError, match=msg):
            df.rename({}, index={})

        with pytest.raises(TypeError, match=msg):
            df.rename({}, columns={})

        with pytest.raises(TypeError, match=msg):
            df.rename({}, columns={}, index={})

    def test_rename_with_duplicate_columns(self):
        # GH#4403
        df4 = DataFrame(
            {"RT": [0.0454], "TClose": [22.02], "TExg": [0.0422]},
            index=MultiIndex.from_tuples(
                [(600809, 20130331)], names=["STK_ID", "RPT_Date"]
            ),
        )

        df5 = DataFrame(
            {
                "RPT_Date": [20120930, 20121231, 20130331],
                "STK_ID": [600809] * 3,
                "STK_Name": ["饡驦", "饡驦", "饡驦"],
                "TClose": [38.05, 41.66, 30.01],
            },
            index=MultiIndex.from_tuples(
                [(600809, 20120930), (600809, 20121231), (600809, 20130331)],
                names=["STK_ID", "RPT_Date"],
            ),
        )
        # TODO: can we construct this without merge?
        k = merge(df4, df5, how="inner", left_index=True, right_index=True)
        result = k.rename(columns={"TClose_x": "TClose", "TClose_y": "QT_Close"})

        expected = DataFrame(
            [[0.0454, 22.02, 0.0422, 20130331, 600809, "饡驦", 30.01]],
            columns=[
                "RT",
                "TClose",
                "TExg",
                "RPT_Date",
                "STK_ID",
                "STK_Name",
                "QT_Close",
            ],
        ).set_index(["STK_ID", "RPT_Date"], drop=False)
        tm.assert_frame_equal(result, expected)

    def test_rename_boolean_index(self):
        df = DataFrame(np.arange(15).reshape(3, 5), columns=[False, True, 2, 3, 4])
        mapper = {0: "foo", 1: "bar", 2: "bah"}
        res = df.rename(index=mapper)
        exp = DataFrame(
            np.arange(15).reshape(3, 5),
            columns=[False, True, 2, 3, 4],
            index=["foo", "bar", "bah"],
        )
        tm.assert_frame_equal(res, exp)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tslibs\test_parsing.py ===
"""
Tests for Timestamp parsing, aimed at pandas/_libs/tslibs/parsing.pyx
"""
from datetime import datetime
import re

from dateutil.parser import parse as du_parse
from dateutil.tz import tzlocal
from hypothesis import given
import numpy as np
import pytest

from pandas._libs.tslibs import (
    parsing,
    strptime,
)
from pandas._libs.tslibs.parsing import parse_datetime_string_with_reso
from pandas.compat import (
    ISMUSL,
    is_platform_arm,
    is_platform_windows,
)
import pandas.util._test_decorators as td

import pandas._testing as tm
from pandas._testing._hypothesis import DATETIME_NO_TZ


@pytest.mark.skipif(
    is_platform_windows() or ISMUSL or is_platform_arm(),
    reason="TZ setting incorrect on Windows and MUSL Linux",
)
def test_parsing_tzlocal_deprecated():
    # GH#50791
    msg = (
        "Parsing 'EST' as tzlocal.*"
        "Pass the 'tz' keyword or call tz_localize after construction instead"
    )
    dtstr = "Jan 15 2004 03:00 EST"

    with tm.set_timezone("US/Eastern"):
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res, _ = parse_datetime_string_with_reso(dtstr)

        assert isinstance(res.tzinfo, tzlocal)

        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = parsing.py_parse_datetime_string(dtstr)
        assert isinstance(res.tzinfo, tzlocal)


def test_parse_datetime_string_with_reso():
    (parsed, reso) = parse_datetime_string_with_reso("4Q1984")
    (parsed_lower, reso_lower) = parse_datetime_string_with_reso("4q1984")

    assert reso == reso_lower
    assert parsed == parsed_lower


def test_parse_datetime_string_with_reso_nanosecond_reso():
    # GH#46811
    parsed, reso = parse_datetime_string_with_reso("2022-04-20 09:19:19.123456789")
    assert reso == "nanosecond"


def test_parse_datetime_string_with_reso_invalid_type():
    # Raise on invalid input, don't just return it
    msg = "Argument 'date_string' has incorrect type (expected str, got tuple)"
    with pytest.raises(TypeError, match=re.escape(msg)):
        parse_datetime_string_with_reso((4, 5))


@pytest.mark.parametrize(
    "dashed,normal", [("1988-Q2", "1988Q2"), ("2Q-1988", "2Q1988")]
)
def test_parse_time_quarter_with_dash(dashed, normal):
    # see gh-9688
    (parsed_dash, reso_dash) = parse_datetime_string_with_reso(dashed)
    (parsed, reso) = parse_datetime_string_with_reso(normal)

    assert parsed_dash == parsed
    assert reso_dash == reso


@pytest.mark.parametrize("dashed", ["-2Q1992", "2-Q1992", "4-4Q1992"])
def test_parse_time_quarter_with_dash_error(dashed):
    msg = f"Unknown datetime string format, unable to parse: {dashed}"

    with pytest.raises(parsing.DateParseError, match=msg):
        parse_datetime_string_with_reso(dashed)


@pytest.mark.parametrize(
    "date_string,expected",
    [
        ("123.1234", False),
        ("-50000", False),
        ("999", False),
        ("m", False),
        ("T", False),
        ("Mon Sep 16, 2013", True),
        ("2012-01-01", True),
        ("01/01/2012", True),
        ("01012012", True),
        ("0101", True),
        ("1-1", True),
    ],
)
def test_does_not_convert_mixed_integer(date_string, expected):
    assert parsing._does_string_look_like_datetime(date_string) is expected


@pytest.mark.parametrize(
    "date_str,kwargs,msg",
    [
        (
            "2013Q5",
            {},
            (
                "Incorrect quarterly string is given, "
                "quarter must be between 1 and 4: 2013Q5"
            ),
        ),
        # see gh-5418
        (
            "2013Q1",
            {"freq": "INVLD-L-DEC-SAT"},
            (
                "Unable to retrieve month information "
                "from given freq: INVLD-L-DEC-SAT"
            ),
        ),
    ],
)
def test_parsers_quarterly_with_freq_error(date_str, kwargs, msg):
    with pytest.raises(parsing.DateParseError, match=msg):
        parsing.parse_datetime_string_with_reso(date_str, **kwargs)


@pytest.mark.parametrize(
    "date_str,freq,expected",
    [
        ("2013Q2", None, datetime(2013, 4, 1)),
        ("2013Q2", "Y-APR", datetime(2012, 8, 1)),
        ("2013-Q2", "Y-DEC", datetime(2013, 4, 1)),
    ],
)
def test_parsers_quarterly_with_freq(date_str, freq, expected):
    result, _ = parsing.parse_datetime_string_with_reso(date_str, freq=freq)
    assert result == expected


@pytest.mark.parametrize(
    "date_str", ["2Q 2005", "2Q-200Y", "2Q-200", "22Q2005", "2Q200.", "6Q-20"]
)
def test_parsers_quarter_invalid(date_str):
    if date_str == "6Q-20":
        msg = (
            "Incorrect quarterly string is given, quarter "
            f"must be between 1 and 4: {date_str}"
        )
    else:
        msg = f"Unknown datetime string format, unable to parse: {date_str}"

    with pytest.raises(ValueError, match=msg):
        parsing.parse_datetime_string_with_reso(date_str)


@pytest.mark.parametrize(
    "date_str,expected",
    [("201101", datetime(2011, 1, 1, 0, 0)), ("200005", datetime(2000, 5, 1, 0, 0))],
)
def test_parsers_month_freq(date_str, expected):
    result, _ = parsing.parse_datetime_string_with_reso(date_str, freq="ME")
    assert result == expected


@td.skip_if_not_us_locale
@pytest.mark.parametrize(
    "string,fmt",
    [
        ("20111230", "%Y%m%d"),
        ("201112300000", "%Y%m%d%H%M"),
        ("20111230000000", "%Y%m%d%H%M%S"),
        ("20111230T00", "%Y%m%dT%H"),
        ("20111230T0000", "%Y%m%dT%H%M"),
        ("20111230T000000", "%Y%m%dT%H%M%S"),
        ("2011-12-30", "%Y-%m-%d"),
        ("2011", "%Y"),
        ("2011-01", "%Y-%m"),
        ("30-12-2011", "%d-%m-%Y"),
        ("2011-12-30 00:00:00", "%Y-%m-%d %H:%M:%S"),
        ("2011-12-30T00:00:00", "%Y-%m-%dT%H:%M:%S"),
        ("2011-12-30T00:00:00UTC", "%Y-%m-%dT%H:%M:%S%Z"),
        ("2011-12-30T00:00:00Z", "%Y-%m-%dT%H:%M:%S%z"),
        ("2011-12-30T00:00:00+9", "%Y-%m-%dT%H:%M:%S%z"),
        ("2011-12-30T00:00:00+09", "%Y-%m-%dT%H:%M:%S%z"),
        ("2011-12-30T00:00:00+090", None),
        ("2011-12-30T00:00:00+0900", "%Y-%m-%dT%H:%M:%S%z"),
        ("2011-12-30T00:00:00-0900", "%Y-%m-%dT%H:%M:%S%z"),
        ("2011-12-30T00:00:00+09:00", "%Y-%m-%dT%H:%M:%S%z"),
        ("2011-12-30T00:00:00+09:000", None),
        ("2011-12-30T00:00:00+9:0", "%Y-%m-%dT%H:%M:%S%z"),
        ("2011-12-30T00:00:00+09:", None),
        ("2011-12-30T00:00:00.000000UTC", "%Y-%m-%dT%H:%M:%S.%f%Z"),
        ("2011-12-30T00:00:00.000000Z", "%Y-%m-%dT%H:%M:%S.%f%z"),
        ("2011-12-30T00:00:00.000000+9", "%Y-%m-%dT%H:%M:%S.%f%z"),
        ("2011-12-30T00:00:00.000000+09", "%Y-%m-%dT%H:%M:%S.%f%z"),
        ("2011-12-30T00:00:00.000000+090", None),
        ("2011-12-30T00:00:00.000000+0900", "%Y-%m-%dT%H:%M:%S.%f%z"),
        ("2011-12-30T00:00:00.000000-0900", "%Y-%m-%dT%H:%M:%S.%f%z"),
        ("2011-12-30T00:00:00.000000+09:00", "%Y-%m-%dT%H:%M:%S.%f%z"),
        ("2011-12-30T00:00:00.000000+09:000", None),
        ("2011-12-30T00:00:00.000000+9:0", "%Y-%m-%dT%H:%M:%S.%f%z"),
        ("2011-12-30T00:00:00.000000+09:", None),
        ("2011-12-30 00:00:00.000000", "%Y-%m-%d %H:%M:%S.%f"),
        ("Tue 24 Aug 2021 01:30:48", "%a %d %b %Y %H:%M:%S"),
        ("Tuesday 24 Aug 2021 01:30:48", "%A %d %b %Y %H:%M:%S"),
        ("Tue 24 Aug 2021 01:30:48 AM", "%a %d %b %Y %I:%M:%S %p"),
        ("Tuesday 24 Aug 2021 01:30:48 AM", "%A %d %b %Y %I:%M:%S %p"),
        ("27.03.2003 14:55:00.000", "%d.%m.%Y %H:%M:%S.%f"),  # GH50317
    ],
)
def test_guess_datetime_format_with_parseable_formats(string, fmt):
    with tm.maybe_produces_warning(
        UserWarning, fmt is not None and re.search(r"%d.*%m", fmt)
    ):
        result = parsing.guess_datetime_format(string)
    assert result == fmt


@pytest.mark.parametrize("dayfirst,expected", [(True, "%d/%m/%Y"), (False, "%m/%d/%Y")])
def test_guess_datetime_format_with_dayfirst(dayfirst, expected):
    ambiguous_string = "01/01/2011"
    result = parsing.guess_datetime_format(ambiguous_string, dayfirst=dayfirst)
    assert result == expected


@td.skip_if_not_us_locale
@pytest.mark.parametrize(
    "string,fmt",
    [
        ("30/Dec/2011", "%d/%b/%Y"),
        ("30/December/2011", "%d/%B/%Y"),
        ("30/Dec/2011 00:00:00", "%d/%b/%Y %H:%M:%S"),
    ],
)
def test_guess_datetime_format_with_locale_specific_formats(string, fmt):
    result = parsing.guess_datetime_format(string)
    assert result == fmt


@pytest.mark.parametrize(
    "invalid_dt",
    [
        "01/2013",
        "12:00:00",
        "1/1/1/1",
        "this_is_not_a_datetime",
        "51a",
        "13/2019",
        "202001",  # YYYYMM isn't ISO8601
        "2020/01",  # YYYY/MM isn't ISO8601 either
        "87156549591102612381000001219H5",
    ],
)
def test_guess_datetime_format_invalid_inputs(invalid_dt):
    # A datetime string must include a year, month and a day for it to be
    # guessable, in addition to being a string that looks like a datetime.
    assert parsing.guess_datetime_format(invalid_dt) is None


@pytest.mark.parametrize("invalid_type_dt", [9, datetime(2011, 1, 1)])
def test_guess_datetime_format_wrong_type_inputs(invalid_type_dt):
    # A datetime string must include a year, month and a day for it to be
    # guessable, in addition to being a string that looks like a datetime.
    with pytest.raises(
        TypeError,
        match=r"^Argument 'dt_str' has incorrect type \(expected str, got .*\)$",
    ):
        parsing.guess_datetime_format(invalid_type_dt)


@pytest.mark.parametrize(
    "string,fmt,dayfirst,warning",
    [
        ("2011-1-1", "%Y-%m-%d", False, None),
        ("2011-1-1", "%Y-%d-%m", True, None),
        ("1/1/2011", "%m/%d/%Y", False, None),
        ("1/1/2011", "%d/%m/%Y", True, None),
        ("30-1-2011", "%d-%m-%Y", False, UserWarning),
        ("30-1-2011", "%d-%m-%Y", True, None),
        ("2011-1-1 0:0:0", "%Y-%m-%d %H:%M:%S", False, None),
        ("2011-1-1 0:0:0", "%Y-%d-%m %H:%M:%S", True, None),
        ("2011-1-3T00:00:0", "%Y-%m-%dT%H:%M:%S", False, None),
        ("2011-1-3T00:00:0", "%Y-%d-%mT%H:%M:%S", True, None),
        ("2011-1-1 00:00:00", "%Y-%m-%d %H:%M:%S", False, None),
        ("2011-1-1 00:00:00", "%Y-%d-%m %H:%M:%S", True, None),
    ],
)
def test_guess_datetime_format_no_padding(string, fmt, dayfirst, warning):
    # see gh-11142
    msg = (
        rf"Parsing dates in {fmt} format when dayfirst=False \(the default\) "
        "was specified. "
        "Pass `dayfirst=True` or specify a format to silence this warning."
    )
    with tm.assert_produces_warning(warning, match=msg):
        result = parsing.guess_datetime_format(string, dayfirst=dayfirst)
    assert result == fmt


def test_try_parse_dates():
    arr = np.array(["5/1/2000", "6/1/2000", "7/1/2000"], dtype=object)
    result = parsing.try_parse_dates(arr, parser=lambda x: du_parse(x, dayfirst=True))

    expected = np.array([du_parse(d, dayfirst=True) for d in arr])
    tm.assert_numpy_array_equal(result, expected)


def test_parse_datetime_string_with_reso_check_instance_type_raise_exception():
    # issue 20684
    msg = "Argument 'date_string' has incorrect type (expected str, got tuple)"
    with pytest.raises(TypeError, match=re.escape(msg)):
        parse_datetime_string_with_reso((1, 2, 3))

    result = parse_datetime_string_with_reso("2019")
    expected = (datetime(2019, 1, 1), "year")
    assert result == expected


@pytest.mark.parametrize(
    "fmt,expected",
    [
        ("%Y %m %d %H:%M:%S", True),
        ("%Y/%m/%d %H:%M:%S", True),
        (r"%Y\%m\%d %H:%M:%S", True),
        ("%Y-%m-%d %H:%M:%S", True),
        ("%Y.%m.%d %H:%M:%S", True),
        ("%Y%m%d %H:%M:%S", True),
        ("%Y-%m-%dT%H:%M:%S", True),
        ("%Y-%m-%dT%H:%M:%S%z", True),
        ("%Y-%m-%dT%H:%M:%S%Z", False),
        ("%Y-%m-%dT%H:%M:%S.%f", True),
        ("%Y-%m-%dT%H:%M:%S.%f%z", True),
        ("%Y-%m-%dT%H:%M:%S.%f%Z", False),
        ("%Y%m%d", True),
        ("%Y%m", False),
        ("%Y", True),
        ("%Y-%m-%d", True),
        ("%Y-%m", True),
    ],
)
def test_is_iso_format(fmt, expected):
    # see gh-41047
    result = strptime._test_format_is_iso(fmt)
    assert result == expected


@pytest.mark.parametrize(
    "input",
    [
        "2018-01-01T00:00:00.123456789",
        "2018-01-01T00:00:00.123456",
        "2018-01-01T00:00:00.123",
    ],
)
def test_guess_datetime_format_f(input):
    # https://github.com/pandas-dev/pandas/issues/49043
    result = parsing.guess_datetime_format(input)
    expected = "%Y-%m-%dT%H:%M:%S.%f"
    assert result == expected


def _helper_hypothesis_delimited_date(call, date_string, **kwargs):
    msg, result = None, None
    try:
        result = call(date_string, **kwargs)
    except ValueError as err:
        msg = str(err)
    return msg, result


@given(DATETIME_NO_TZ)
@pytest.mark.parametrize("delimiter", list(" -./"))
@pytest.mark.parametrize("dayfirst", [True, False])
@pytest.mark.parametrize(
    "date_format",
    ["%d %m %Y", "%m %d %Y", "%m %Y", "%Y %m %d", "%y %m %d", "%Y%m%d", "%y%m%d"],
)
def test_hypothesis_delimited_date(
    request, date_format, dayfirst, delimiter, test_datetime
):
    if date_format == "%m %Y" and delimiter == ".":
        request.applymarker(
            pytest.mark.xfail(
                reason="parse_datetime_string cannot reliably tell whether "
                "e.g. %m.%Y is a float or a date"
            )
        )
    date_string = test_datetime.strftime(date_format.replace(" ", delimiter))

    except_out_dateutil, result = _helper_hypothesis_delimited_date(
        parsing.py_parse_datetime_string, date_string, dayfirst=dayfirst
    )
    except_in_dateutil, expected = _helper_hypothesis_delimited_date(
        du_parse,
        date_string,
        default=datetime(1, 1, 1),
        dayfirst=dayfirst,
        yearfirst=False,
    )

    assert except_out_dateutil == except_in_dateutil
    assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\intervalmath\tests\test_interval_functions.py ===
from sympy.external import import_module
from sympy.plotting.intervalmath import (
    Abs, acos, acosh, And, asin, asinh, atan, atanh, ceil, cos, cosh,
    exp, floor, imax, imin, interval, log, log10, Or, sin, sinh, sqrt,
    tan, tanh,
)

np = import_module('numpy')
if not np:
    disabled = True


#requires Numpy. Hence included in interval_functions


def test_interval_pow():
    a = 2**interval(1, 2) == interval(2, 4)
    assert a == (True, True)
    a = interval(1, 2)**interval(1, 2) == interval(1, 4)
    assert a == (True, True)
    a = interval(-1, 1)**interval(0.5, 2)
    assert a.is_valid is None
    a = interval(-2, -1) ** interval(1, 2)
    assert a.is_valid is False
    a = interval(-2, -1) ** (1.0 / 2)
    assert a.is_valid is False
    a = interval(-1, 1)**(1.0 / 2)
    assert a.is_valid is None
    a = interval(-1, 1)**(1.0 / 3) == interval(-1, 1)
    assert a == (True, True)
    a = interval(-1, 1)**2 == interval(0, 1)
    assert a == (True, True)
    a = interval(-1, 1) ** (1.0 / 29) == interval(-1, 1)
    assert a == (True, True)
    a = -2**interval(1, 1) == interval(-2, -2)
    assert a == (True, True)

    a = interval(1, 2, is_valid=False)**2
    assert a.is_valid is False

    a = (-3)**interval(1, 2)
    assert a.is_valid is False
    a = (-4)**interval(0.5, 0.5)
    assert a.is_valid is False
    assert ((-3)**interval(1, 1) == interval(-3, -3)) == (True, True)

    a = interval(8, 64)**(2.0 / 3)
    assert abs(a.start - 4) < 1e-10  # eps
    assert abs(a.end - 16) < 1e-10
    a = interval(-8, 64)**(2.0 / 3)
    assert abs(a.start - 4) < 1e-10  # eps
    assert abs(a.end - 16) < 1e-10


def test_exp():
    a = exp(interval(-np.inf, 0))
    assert a.start == np.exp(-np.inf)
    assert a.end == np.exp(0)
    a = exp(interval(1, 2))
    assert a.start == np.exp(1)
    assert a.end == np.exp(2)
    a = exp(1)
    assert a.start == np.exp(1)
    assert a.end == np.exp(1)


def test_log():
    a = log(interval(1, 2))
    assert a.start == 0
    assert a.end == np.log(2)
    a = log(interval(-1, 1))
    assert a.is_valid is None
    a = log(interval(-3, -1))
    assert a.is_valid is False
    a = log(-3)
    assert a.is_valid is False
    a = log(2)
    assert a.start == np.log(2)
    assert a.end == np.log(2)


def test_log10():
    a = log10(interval(1, 2))
    assert a.start == 0
    assert a.end == np.log10(2)
    a = log10(interval(-1, 1))
    assert a.is_valid is None
    a = log10(interval(-3, -1))
    assert a.is_valid is False
    a = log10(-3)
    assert a.is_valid is False
    a = log10(2)
    assert a.start == np.log10(2)
    assert a.end == np.log10(2)


def test_atan():
    a = atan(interval(0, 1))
    assert a.start == np.arctan(0)
    assert a.end == np.arctan(1)
    a = atan(1)
    assert a.start == np.arctan(1)
    assert a.end == np.arctan(1)


def test_sin():
    a = sin(interval(0, np.pi / 4))
    assert a.start == np.sin(0)
    assert a.end == np.sin(np.pi / 4)

    a = sin(interval(-np.pi / 4, np.pi / 4))
    assert a.start == np.sin(-np.pi / 4)
    assert a.end == np.sin(np.pi / 4)

    a = sin(interval(np.pi / 4, 3 * np.pi / 4))
    assert a.start == np.sin(np.pi / 4)
    assert a.end == 1

    a = sin(interval(7 * np.pi / 6, 7 * np.pi / 4))
    assert a.start == -1
    assert a.end == np.sin(7 * np.pi / 6)

    a = sin(interval(0, 3 * np.pi))
    assert a.start == -1
    assert a.end == 1

    a = sin(interval(np.pi / 3, 7 * np.pi / 4))
    assert a.start == -1
    assert a.end == 1

    a = sin(np.pi / 4)
    assert a.start == np.sin(np.pi / 4)
    assert a.end == np.sin(np.pi / 4)

    a = sin(interval(1, 2, is_valid=False))
    assert a.is_valid is False


def test_cos():
    a = cos(interval(0, np.pi / 4))
    assert a.start == np.cos(np.pi / 4)
    assert a.end == 1

    a = cos(interval(-np.pi / 4, np.pi / 4))
    assert a.start == np.cos(-np.pi / 4)
    assert a.end == 1

    a = cos(interval(np.pi / 4, 3 * np.pi / 4))
    assert a.start == np.cos(3 * np.pi / 4)
    assert a.end == np.cos(np.pi / 4)

    a = cos(interval(3 * np.pi / 4, 5 * np.pi / 4))
    assert a.start == -1
    assert a.end == np.cos(3 * np.pi / 4)

    a = cos(interval(0, 3 * np.pi))
    assert a.start == -1
    assert a.end == 1

    a = cos(interval(- np.pi / 3, 5 * np.pi / 4))
    assert a.start == -1
    assert a.end == 1

    a = cos(interval(1, 2, is_valid=False))
    assert a.is_valid is False


def test_tan():
    a = tan(interval(0, np.pi / 4))
    assert a.start == 0
    # must match lib_interval definition of tan:
    assert a.end == np.sin(np.pi / 4)/np.cos(np.pi / 4)

    a = tan(interval(np.pi / 4, 3 * np.pi / 4))
    #discontinuity
    assert a.is_valid is None


def test_sqrt():
    a = sqrt(interval(1, 4))
    assert a.start == 1
    assert a.end == 2

    a = sqrt(interval(0.01, 1))
    assert a.start == np.sqrt(0.01)
    assert a.end == 1

    a = sqrt(interval(-1, 1))
    assert a.is_valid is None

    a = sqrt(interval(-3, -1))
    assert a.is_valid is False

    a = sqrt(4)
    assert (a == interval(2, 2)) == (True, True)

    a = sqrt(-3)
    assert a.is_valid is False


def test_imin():
    a = imin(interval(1, 3), interval(2, 5), interval(-1, 3))
    assert a.start == -1
    assert a.end == 3

    a = imin(-2, interval(1, 4))
    assert a.start == -2
    assert a.end == -2

    a = imin(5, interval(3, 4), interval(-2, 2, is_valid=False))
    assert a.start == 3
    assert a.end == 4


def test_imax():
    a = imax(interval(-2, 2), interval(2, 7), interval(-3, 9))
    assert a.start == 2
    assert a.end == 9

    a = imax(8, interval(1, 4))
    assert a.start == 8
    assert a.end == 8

    a = imax(interval(1, 2), interval(3, 4), interval(-2, 2, is_valid=False))
    assert a.start == 3
    assert a.end == 4


def test_sinh():
    a = sinh(interval(-1, 1))
    assert a.start == np.sinh(-1)
    assert a.end == np.sinh(1)

    a = sinh(1)
    assert a.start == np.sinh(1)
    assert a.end == np.sinh(1)


def test_cosh():
    a = cosh(interval(1, 2))
    assert a.start == np.cosh(1)
    assert a.end == np.cosh(2)
    a = cosh(interval(-2, -1))
    assert a.start == np.cosh(-1)
    assert a.end == np.cosh(-2)

    a = cosh(interval(-2, 1))
    assert a.start == 1
    assert a.end == np.cosh(-2)

    a = cosh(1)
    assert a.start == np.cosh(1)
    assert a.end == np.cosh(1)


def test_tanh():
    a = tanh(interval(-3, 3))
    assert a.start == np.tanh(-3)
    assert a.end == np.tanh(3)

    a = tanh(3)
    assert a.start == np.tanh(3)
    assert a.end == np.tanh(3)


def test_asin():
    a = asin(interval(-0.5, 0.5))
    assert a.start == np.arcsin(-0.5)
    assert a.end == np.arcsin(0.5)

    a = asin(interval(-1.5, 1.5))
    assert a.is_valid is None
    a = asin(interval(-2, -1.5))
    assert a.is_valid is False

    a = asin(interval(0, 2))
    assert a.is_valid is None

    a = asin(interval(2, 5))
    assert a.is_valid is False

    a = asin(0.5)
    assert a.start == np.arcsin(0.5)
    assert a.end == np.arcsin(0.5)

    a = asin(1.5)
    assert a.is_valid is False


def test_acos():
    a = acos(interval(-0.5, 0.5))
    assert a.start == np.arccos(0.5)
    assert a.end == np.arccos(-0.5)

    a = acos(interval(-1.5, 1.5))
    assert a.is_valid is None
    a = acos(interval(-2, -1.5))
    assert a.is_valid is False

    a = acos(interval(0, 2))
    assert a.is_valid is None

    a = acos(interval(2, 5))
    assert a.is_valid is False

    a = acos(0.5)
    assert a.start == np.arccos(0.5)
    assert a.end == np.arccos(0.5)

    a = acos(1.5)
    assert a.is_valid is False


def test_ceil():
    a = ceil(interval(0.2, 0.5))
    assert a.start == 1
    assert a.end == 1

    a = ceil(interval(0.5, 1.5))
    assert a.start == 1
    assert a.end == 2
    assert a.is_valid is None

    a = ceil(interval(-5, 5))
    assert a.is_valid is None

    a = ceil(5.4)
    assert a.start == 6
    assert a.end == 6


def test_floor():
    a = floor(interval(0.2, 0.5))
    assert a.start == 0
    assert a.end == 0

    a = floor(interval(0.5, 1.5))
    assert a.start == 0
    assert a.end == 1
    assert a.is_valid is None

    a = floor(interval(-5, 5))
    assert a.is_valid is None

    a = floor(5.4)
    assert a.start == 5
    assert a.end == 5


def test_asinh():
    a = asinh(interval(1, 2))
    assert a.start == np.arcsinh(1)
    assert a.end == np.arcsinh(2)

    a = asinh(0.5)
    assert a.start == np.arcsinh(0.5)
    assert a.end == np.arcsinh(0.5)


def test_acosh():
    a = acosh(interval(3, 5))
    assert a.start == np.arccosh(3)
    assert a.end == np.arccosh(5)

    a = acosh(interval(0, 3))
    assert a.is_valid is None
    a = acosh(interval(-3, 0.5))
    assert a.is_valid is False

    a = acosh(0.5)
    assert a.is_valid is False

    a = acosh(2)
    assert a.start == np.arccosh(2)
    assert a.end == np.arccosh(2)


def test_atanh():
    a = atanh(interval(-0.5, 0.5))
    assert a.start == np.arctanh(-0.5)
    assert a.end == np.arctanh(0.5)

    a = atanh(interval(0, 3))
    assert a.is_valid is None

    a = atanh(interval(-3, -2))
    assert a.is_valid is False

    a = atanh(0.5)
    assert a.start == np.arctanh(0.5)
    assert a.end == np.arctanh(0.5)

    a = atanh(1.5)
    assert a.is_valid is False


def test_Abs():
    assert (Abs(interval(-0.5, 0.5)) == interval(0, 0.5)) == (True, True)
    assert (Abs(interval(-3, -2)) == interval(2, 3)) == (True, True)
    assert (Abs(-3) == interval(3, 3)) == (True, True)


def test_And():
    args = [(True, True), (True, False), (True, None)]
    assert And(*args) == (True, False)

    args = [(False, True), (None, None), (True, True)]
    assert And(*args) == (False, None)


def test_Or():
    args = [(True, True), (True, False), (False, None)]
    assert Or(*args) == (True, True)
    args = [(None, None), (False, None), (False, False)]
    assert Or(*args) == (None, None)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\certifi\__main__.py ===
import argparse

from certifi import contents, where

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--contents", action="store_true")
args = parser.parse_args()

if args.contents:
    print(contents())
else:
    print(where())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\dateutil\tz\__init__.py ===
# -*- coding: utf-8 -*-
from .tz import *
from .tz import __doc__

__all__ = ["tzutc", "tzoffset", "tzlocal", "tzfile", "tzrange",
           "tzstr", "tzical", "tzwin", "tzwinlocal", "gettz",
           "enfold", "datetime_ambiguous", "datetime_exists",
           "resolve_imaginary", "UTC", "DeprecatedTzFormatWarning"]


class DeprecatedTzFormatWarning(Warning):
    """Warning raised when time zones are parsed from deprecated formats."""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\core\numeric.py ===
def __getattr__(attr_name):
    from numpy._core import numeric

    from ._utils import _raise_warning

    sentinel = object()
    ret = getattr(numeric, attr_name, sentinel)
    if ret is sentinel:
        raise AttributeError(
            f"module 'numpy.core.numeric' has no attribute {attr_name}")
    _raise_warning(attr_name, "numeric")
    return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\matrixlib\__init__.py ===
"""Sub-package containing the matrix class and related functions.

"""
from . import defmatrix
from .defmatrix import *

__all__ = defmatrix.__all__

from numpy._pytesttester import PytestTester

test = PytestTester(__name__)
del PytestTester

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\bart\__init__.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os.path
import sys

sys.path.append(os.path.dirname(__file__))

transformers_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if transformers_dir not in sys.path:
    sys.path.append(transformers_dir)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\bert\__init__.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os.path
import sys

sys.path.append(os.path.dirname(__file__))

transformers_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if transformers_dir not in sys.path:
    sys.path.append(transformers_dir)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\gpt2\__init__.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os.path
import sys

sys.path.append(os.path.dirname(__file__))

transformers_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if transformers_dir not in sys.path:
    sys.path.append(transformers_dir)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\llama\__init__.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import sys

sys.path.append(os.path.dirname(__file__))

transformers_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if transformers_dir not in sys.path:
    sys.path.append(transformers_dir)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\longformer\__init__.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os.path
import sys

sys.path.append(os.path.dirname(__file__))

transformers_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if transformers_dir not in sys.path:
    sys.path.append(transformers_dir)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\phi2\__init__.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import sys

sys.path.append(os.path.dirname(__file__))

transformers_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if transformers_dir not in sys.path:
    sys.path.append(transformers_dir)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\sam2\__init__.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os.path
import sys

sys.path.append(os.path.dirname(__file__))

transformers_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if transformers_dir not in sys.path:
    sys.path.append(transformers_dir)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\stable_diffusion\trt_utilities.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


def init_trt_plugins():
    # Register TensorRT plugins
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\stable_diffusion\__init__.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os.path
import sys

sys.path.append(os.path.dirname(__file__))

transformers_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if transformers_dir not in sys.path:
    sys.path.append(transformers_dir)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\t5\__init__.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os.path
import sys

sys.path.append(os.path.dirname(__file__))

transformers_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if transformers_dir not in sys.path:
    sys.path.append(transformers_dir)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\whisper\__init__.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import sys

sys.path.append(os.path.dirname(__file__))

transformers_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if transformers_dir not in sys.path:
    sys.path.append(transformers_dir)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\descriptors\namespace.py ===
# Copyright (c) 2010-2024 openpyxl


def namespaced(obj, tagname, namespace=None):
    """
    Utility to create a namespaced tag for an object
    """

    namespace = getattr(obj, "namespace", None) or namespace
    if namespace is not None:
        tagname = "{%s}%s" % (namespace, tagname)
    return tagname

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tseries\__init__.py ===
# ruff: noqa: TCH004
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # import modules that have public classes/functions:
    from pandas.tseries import (
        frequencies,
        offsets,
    )

    # and mark only those modules as public
    __all__ = ["frequencies", "offsets"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\main.py ===
from typing import List, Optional


def main(args: Optional[List[str]] = None) -> int:
    """This is preserved for old console scripts that may still be referencing
    it.

    For additional details, see https://github.com/pypa/pip/issues/7498.
    """
    from pip._internal.utils.entrypoints import _wrapper

    return _wrapper(args)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\certifi\__main__.py ===
import argparse

from pip._vendor.certifi import contents, where

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--contents", action="store_true")
args = parser.parse_args()

if args.contents:
    print(contents())
else:
    print(where())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\trace.py ===
from sympy.utilities.exceptions import sympy_deprecation_warning

sympy_deprecation_warning(
    """
    sympy.core.trace is deprecated. Use sympy.physics.quantum.trace
    instead.
    """,
    deprecated_since_version="1.10",
    active_deprecations_target="sympy-core-trace-deprecated",
)

from sympy.physics.quantum.trace import Tr # noqa:F401

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\benchmarks\bench_assumptions.py ===
from sympy.core import Symbol, Integer

x = Symbol('x')
i3 = Integer(3)


def timeit_x_is_integer():
    x.is_integer


def timeit_Integer_is_irrational():
    i3.is_irrational

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\logic\__init__.py ===
from .boolalg import (to_cnf, to_dnf, to_nnf, And, Or, Not, Xor, Nand, Nor, Implies,
    Equivalent, ITE, POSform, SOPform, simplify_logic, bool_map, true, false,
    gateinputcount)
from .inference import satisfiable

__all__ = [
    'to_cnf', 'to_dnf', 'to_nnf', 'And', 'Or', 'Not', 'Xor', 'Nand', 'Nor',
    'Implies', 'Equivalent', 'ITE', 'POSform', 'SOPform', 'simplify_logic',
    'bool_map', 'true', 'false', 'gateinputcount',

    'satisfiable',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\__init__.py ===
"""
A module that helps solving problems in physics.
"""

from . import units
from .matrices import mgamma, msigma, minkowski_tensor, mdft

__all__ = [
    'units',

    'mgamma', 'msigma', 'minkowski_tensor', 'mdft',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\intervalmath\__init__.py ===
from .interval_arithmetic import interval
from .lib_interval import (Abs, exp, log, log10, sin, cos, tan, sqrt,
                          imin, imax, sinh, cosh, tanh, acosh, asinh, atanh,
                          asin, acos, atan, ceil, floor, And, Or)

__all__ = [
    'interval',

    'Abs', 'exp', 'log', 'log10', 'sin', 'cos', 'tan', 'sqrt', 'imin', 'imax',
    'sinh', 'cosh', 'tanh', 'acosh', 'asinh', 'atanh', 'asin', 'acos', 'atan',
    'ceil', 'floor', 'And', 'Or',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domainmatrix.py ===
"""
Stub module to expose DomainMatrix which has now moved to
sympy.polys.matrices package. It should now be imported as:

    >>> from sympy.polys.matrices import DomainMatrix

This module might be removed in future.
"""

from sympy.polys.matrices.domainmatrix import DomainMatrix

__all__ = ['DomainMatrix']

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\pretty\__init__.py ===
"""ASCII-ART 2D pretty-printer"""

from .pretty import (pretty, pretty_print, pprint, pprint_use_unicode,
    pprint_try_use_unicode, pager_print)

# if unicode output is available -- let's use it
pprint_try_use_unicode()

__all__ = [
    'pretty', 'pretty_print', 'pprint', 'pprint_use_unicode',
    'pprint_try_use_unicode', 'pager_print',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\benchmarks\bench_solvers.py ===
from sympy.core.symbol import Symbol
from sympy.matrices.dense import (eye, zeros)
from sympy.solvers.solvers import solve_linear_system

N = 8
M = zeros(N, N + 1)
M[:, :N] = eye(N)
S = [Symbol('A%i' % i) for i in range(N)]


def timeit_linsolve_trivial():
    solve_linear_system(M, *S)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\strategies\branch\tools.py ===
from .core import exhaust, multiplex
from .traverse import top_down


def canon(*rules):
    """ Strategy for canonicalization

    Apply each branching rule in a top-down fashion through the tree.
    Multiplex through all branching rule traversals
    Keep doing this until there is no change.
    """
    return exhaust(multiplex(*map(top_down, rules)))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\expressions\conv_array_to_indexed.py ===
from sympy.tensor.array.expressions import from_array_to_indexed
from sympy.utilities.decorator import deprecated


_conv_to_from_decorator = deprecated(
    "module has been renamed by replacing 'conv_' with 'from_' in its name",
    deprecated_since_version="1.11",
    active_deprecations_target="deprecated-conv-array-expr-module-names",
)


convert_array_to_indexed = _conv_to_from_decorator(from_array_to_indexed.convert_array_to_indexed)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\magic.py ===
"""Functions that involve magic. """

def pollute(names, objects):
    """Pollute the global namespace with symbols -> objects mapping. """
    from inspect import currentframe
    frame = currentframe().f_back.f_back

    try:
        for name, obj in zip(names, objects):
            frame.f_globals[name] = obj
    finally:
        del frame  # break cyclic dependencies as stated in inspect docs

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\pytest.py ===
"""
.. deprecated:: 1.6

   sympy.utilities.pytest has been renamed to sympy.testing.pytest.
"""
from sympy.utilities.exceptions import sympy_deprecation_warning

sympy_deprecation_warning("The sympy.utilities.pytest submodule is deprecated. Use sympy.testing.pytest instead.",
    deprecated_since_version="1.6",
    active_deprecations_target="deprecated-sympy-utilities-submodules")

from sympy.testing.pytest import *  # noqa:F401,F403

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\randtest.py ===
"""
.. deprecated:: 1.6

   sympy.utilities.randtest has been renamed to sympy.core.random.
"""
from sympy.utilities.exceptions import sympy_deprecation_warning

sympy_deprecation_warning("The sympy.utilities.randtest submodule is deprecated. Use sympy.core.random instead.",
    deprecated_since_version="1.6",
    active_deprecations_target="deprecated-sympy-utilities-submodules")

from sympy.core.random import *  # noqa:F401,F403

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\tmpfiles.py ===
"""
.. deprecated:: 1.6

   sympy.utilities.tmpfiles has been renamed to sympy.testing.tmpfiles.
"""
from sympy.utilities.exceptions import sympy_deprecation_warning

sympy_deprecation_warning("The sympy.utilities.tmpfiles submodule is deprecated. Use sympy.testing.tmpfiles instead.",
    deprecated_since_version="1.6",
    active_deprecations_target="deprecated-sympy-utilities-submodules")

from sympy.testing.tmpfiles import *  # noqa:F401,F403

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arithmetic\test_object.py ===
# Arithmetic tests for DataFrame/Series/Index/Array classes that should
# behave identically.
# Specifically for object dtype
import datetime
from decimal import Decimal
import operator

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    Series,
    Timestamp,
    option_context,
)
import pandas._testing as tm
from pandas.core import ops

# ------------------------------------------------------------------
# Comparisons


class TestObjectComparisons:
    def test_comparison_object_numeric_nas(self, comparison_op):
        ser = Series(np.random.default_rng(2).standard_normal(10), dtype=object)
        shifted = ser.shift(2)

        func = comparison_op

        result = func(ser, shifted)
        expected = func(ser.astype(float), shifted.astype(float))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "infer_string", [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))]
    )
    def test_object_comparisons(self, infer_string):
        with option_context("future.infer_string", infer_string):
            ser = Series(["a", "b", np.nan, "c", "a"])

            result = ser == "a"
            expected = Series([True, False, False, False, True])
            tm.assert_series_equal(result, expected)

            result = ser < "a"
            expected = Series([False, False, False, False, False])
            tm.assert_series_equal(result, expected)

            result = ser != "a"
            expected = -(ser == "a")
            tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", [None, object])
    def test_more_na_comparisons(self, dtype):
        left = Series(["a", np.nan, "c"], dtype=dtype)
        right = Series(["a", np.nan, "d"], dtype=dtype)

        result = left == right
        expected = Series([True, False, False])
        tm.assert_series_equal(result, expected)

        result = left != right
        expected = Series([False, True, True])
        tm.assert_series_equal(result, expected)

        result = left == np.nan
        expected = Series([False, False, False])
        tm.assert_series_equal(result, expected)

        result = left != np.nan
        expected = Series([True, True, True])
        tm.assert_series_equal(result, expected)


# ------------------------------------------------------------------
# Arithmetic


class TestArithmetic:
    def test_add_period_to_array_of_offset(self):
        # GH#50162
        per = pd.Period("2012-1-1", freq="D")
        pi = pd.period_range("2012-1-1", periods=10, freq="D")
        idx = per - pi

        expected = pd.Index([x + per for x in idx], dtype=object)
        result = idx + per
        tm.assert_index_equal(result, expected)

        result = per + idx
        tm.assert_index_equal(result, expected)

    # TODO: parametrize
    def test_pow_ops_object(self):
        # GH#22922
        # pow is weird with masking & 1, so testing here
        a = Series([1, np.nan, 1, np.nan], dtype=object)
        b = Series([1, np.nan, np.nan, 1], dtype=object)
        result = a**b
        expected = Series(a.values**b.values, dtype=object)
        tm.assert_series_equal(result, expected)

        result = b**a
        expected = Series(b.values**a.values, dtype=object)

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("op", [operator.add, ops.radd])
    @pytest.mark.parametrize("other", ["category", "Int64"])
    def test_add_extension_scalar(self, other, box_with_array, op):
        # GH#22378
        # Check that scalars satisfying is_extension_array_dtype(obj)
        # do not incorrectly try to dispatch to an ExtensionArray operation

        arr = Series(["a", "b", "c"])
        expected = Series([op(x, other) for x in arr])

        arr = tm.box_expected(arr, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = op(arr, other)
        tm.assert_equal(result, expected)

    def test_objarr_add_str(self, box_with_array):
        ser = Series(["x", np.nan, "x"])
        expected = Series(["xa", np.nan, "xa"])

        ser = tm.box_expected(ser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = ser + "a"
        tm.assert_equal(result, expected)

    def test_objarr_radd_str(self, box_with_array):
        ser = Series(["x", np.nan, "x"])
        expected = Series(["ax", np.nan, "ax"])

        ser = tm.box_expected(ser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = "a" + ser
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "data",
        [
            [1, 2, 3],
            [1.1, 2.2, 3.3],
            [Timestamp("2011-01-01"), Timestamp("2011-01-02"), pd.NaT],
            ["x", "y", 1],
        ],
    )
    @pytest.mark.parametrize("dtype", [None, object])
    def test_objarr_radd_str_invalid(self, dtype, data, box_with_array):
        ser = Series(data, dtype=dtype)

        ser = tm.box_expected(ser, box_with_array)
        msg = "|".join(
            [
                "can only concatenate str",
                "did not contain a loop with signature matching types",
                "unsupported operand type",
                "must be str",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            "foo_" + ser

    @pytest.mark.parametrize("op", [operator.add, ops.radd, operator.sub, ops.rsub])
    def test_objarr_add_invalid(self, op, box_with_array):
        # invalid ops
        box = box_with_array

        obj_ser = Series(list("abc"), dtype=object, name="objects")

        obj_ser = tm.box_expected(obj_ser, box)
        msg = "|".join(
            [
                "can only concatenate str",
                "unsupported operand type",
                "must be str",
                "has no kernel",
                "operation 'add' not supported",
                "operation 'radd' not supported",
                "operation 'sub' not supported",
                "operation 'rsub' not supported",
            ]
        )
        with pytest.raises(Exception, match=msg):
            op(obj_ser, 1)
        with pytest.raises(Exception, match=msg):
            op(obj_ser, np.array(1, dtype=np.int64))

    # TODO: Moved from tests.series.test_operators; needs cleanup
    def test_operators_na_handling(self):
        ser = Series(["foo", "bar", "baz", np.nan])
        result = "prefix_" + ser
        expected = Series(["prefix_foo", "prefix_bar", "prefix_baz", np.nan])
        tm.assert_series_equal(result, expected)

        result = ser + "_suffix"
        expected = Series(["foo_suffix", "bar_suffix", "baz_suffix", np.nan])
        tm.assert_series_equal(result, expected)

    # TODO: parametrize over box
    @pytest.mark.parametrize("dtype", [None, object])
    def test_series_with_dtype_radd_timedelta(self, dtype):
        # note this test is _not_ aimed at timedelta64-dtyped Series
        # as of 2.0 we retain object dtype when ser.dtype == object
        ser = Series(
            [pd.Timedelta("1 days"), pd.Timedelta("2 days"), pd.Timedelta("3 days")],
            dtype=dtype,
        )
        expected = Series(
            [pd.Timedelta("4 days"), pd.Timedelta("5 days"), pd.Timedelta("6 days")],
            dtype=dtype,
        )

        result = pd.Timedelta("3 days") + ser
        tm.assert_series_equal(result, expected)

        result = ser + pd.Timedelta("3 days")
        tm.assert_series_equal(result, expected)

    # TODO: cleanup & parametrize over box
    def test_mixed_timezone_series_ops_object(self):
        # GH#13043
        ser = Series(
            [
                Timestamp("2015-01-01", tz="US/Eastern"),
                Timestamp("2015-01-01", tz="Asia/Tokyo"),
            ],
            name="xxx",
        )
        assert ser.dtype == object

        exp = Series(
            [
                Timestamp("2015-01-02", tz="US/Eastern"),
                Timestamp("2015-01-02", tz="Asia/Tokyo"),
            ],
            name="xxx",
        )
        tm.assert_series_equal(ser + pd.Timedelta("1 days"), exp)
        tm.assert_series_equal(pd.Timedelta("1 days") + ser, exp)

        # object series & object series
        ser2 = Series(
            [
                Timestamp("2015-01-03", tz="US/Eastern"),
                Timestamp("2015-01-05", tz="Asia/Tokyo"),
            ],
            name="xxx",
        )
        assert ser2.dtype == object
        exp = Series(
            [pd.Timedelta("2 days"), pd.Timedelta("4 days")], name="xxx", dtype=object
        )
        tm.assert_series_equal(ser2 - ser, exp)
        tm.assert_series_equal(ser - ser2, -exp)

        ser = Series(
            [pd.Timedelta("01:00:00"), pd.Timedelta("02:00:00")],
            name="xxx",
            dtype=object,
        )
        assert ser.dtype == object

        exp = Series(
            [pd.Timedelta("01:30:00"), pd.Timedelta("02:30:00")],
            name="xxx",
            dtype=object,
        )
        tm.assert_series_equal(ser + pd.Timedelta("00:30:00"), exp)
        tm.assert_series_equal(pd.Timedelta("00:30:00") + ser, exp)

    # TODO: cleanup & parametrize over box
    def test_iadd_preserves_name(self):
        # GH#17067, GH#19723 __iadd__ and __isub__ should preserve index name
        ser = Series([1, 2, 3])
        ser.index.name = "foo"

        ser.index += 1
        assert ser.index.name == "foo"

        ser.index -= 1
        assert ser.index.name == "foo"

    def test_add_string(self):
        # from bug report
        index = pd.Index(["a", "b", "c"])
        index2 = index + "foo"

        assert "a" not in index2
        assert "afoo" in index2

    def test_iadd_string(self):
        index = pd.Index(["a", "b", "c"])
        # doesn't fail test unless there is a check before `+=`
        assert "a" in index

        index += "_x"
        assert "a_x" in index

    def test_add(self):
        index = pd.Index([str(i) for i in range(10)])
        expected = pd.Index(index.values * 2)
        tm.assert_index_equal(index + index, expected)
        tm.assert_index_equal(index + index.tolist(), expected)
        tm.assert_index_equal(index.tolist() + index, expected)

        # test add and radd
        index = pd.Index(list("abc"))
        expected = pd.Index(["a1", "b1", "c1"])
        tm.assert_index_equal(index + "1", expected)
        expected = pd.Index(["1a", "1b", "1c"])
        tm.assert_index_equal("1" + index, expected)

    def test_sub_fail(self):
        index = pd.Index([str(i) for i in range(10)])

        msg = "unsupported operand type|Cannot broadcast|sub' not supported"
        with pytest.raises(TypeError, match=msg):
            index - "a"
        with pytest.raises(TypeError, match=msg):
            index - index
        with pytest.raises(TypeError, match=msg):
            index - index.tolist()
        with pytest.raises(TypeError, match=msg):
            index.tolist() - index

    def test_sub_object(self):
        # GH#19369
        index = pd.Index([Decimal(1), Decimal(2)])
        expected = pd.Index([Decimal(0), Decimal(1)])

        result = index - Decimal(1)
        tm.assert_index_equal(result, expected)

        result = index - pd.Index([Decimal(1), Decimal(1)])
        tm.assert_index_equal(result, expected)

        msg = "unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            index - "foo"

        with pytest.raises(TypeError, match=msg):
            index - np.array([2, "foo"], dtype=object)

    def test_rsub_object(self, fixed_now_ts):
        # GH#19369
        index = pd.Index([Decimal(1), Decimal(2)])
        expected = pd.Index([Decimal(1), Decimal(0)])

        result = Decimal(2) - index
        tm.assert_index_equal(result, expected)

        result = np.array([Decimal(2), Decimal(2)]) - index
        tm.assert_index_equal(result, expected)

        msg = "unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            "foo" - index

        with pytest.raises(TypeError, match=msg):
            np.array([True, fixed_now_ts]) - index


class MyIndex(pd.Index):
    # Simple index subclass that tracks ops calls.

    _calls: int

    @classmethod
    def _simple_new(cls, values, name=None, dtype=None):
        result = object.__new__(cls)
        result._data = values
        result._name = name
        result._calls = 0
        result._reset_identity()

        return result

    def __add__(self, other):
        self._calls += 1
        return self._simple_new(self._data)

    def __radd__(self, other):
        return self.__add__(other)


@pytest.mark.parametrize(
    "other",
    [
        [datetime.timedelta(1), datetime.timedelta(2)],
        [datetime.datetime(2000, 1, 1), datetime.datetime(2000, 1, 2)],
        [pd.Period("2000"), pd.Period("2001")],
        ["a", "b"],
    ],
    ids=["timedelta", "datetime", "period", "object"],
)
def test_index_ops_defer_to_unknown_subclasses(other):
    # https://github.com/pandas-dev/pandas/issues/31109
    values = np.array(
        [datetime.date(2000, 1, 1), datetime.date(2000, 1, 2)], dtype=object
    )
    a = MyIndex._simple_new(values)
    other = pd.Index(other)
    result = other + a
    assert isinstance(result, MyIndex)
    assert a._calls == 1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\categorical\test_operators.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestCategoricalOpsWithFactor:
    def test_categories_none_comparisons(self):
        factor = Categorical(["a", "b", "b", "a", "a", "c", "c", "c"], ordered=True)
        tm.assert_categorical_equal(factor, factor)

    def test_comparisons(self):
        factor = Categorical(["a", "b", "b", "a", "a", "c", "c", "c"], ordered=True)
        result = factor[factor == "a"]
        expected = factor[np.asarray(factor) == "a"]
        tm.assert_categorical_equal(result, expected)

        result = factor[factor != "a"]
        expected = factor[np.asarray(factor) != "a"]
        tm.assert_categorical_equal(result, expected)

        result = factor[factor < "c"]
        expected = factor[np.asarray(factor) < "c"]
        tm.assert_categorical_equal(result, expected)

        result = factor[factor > "a"]
        expected = factor[np.asarray(factor) > "a"]
        tm.assert_categorical_equal(result, expected)

        result = factor[factor >= "b"]
        expected = factor[np.asarray(factor) >= "b"]
        tm.assert_categorical_equal(result, expected)

        result = factor[factor <= "b"]
        expected = factor[np.asarray(factor) <= "b"]
        tm.assert_categorical_equal(result, expected)

        n = len(factor)

        other = factor[np.random.default_rng(2).permutation(n)]
        result = factor == other
        expected = np.asarray(factor) == np.asarray(other)
        tm.assert_numpy_array_equal(result, expected)

        result = factor == "d"
        expected = np.zeros(len(factor), dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

        # comparisons with categoricals
        cat_rev = Categorical(["a", "b", "c"], categories=["c", "b", "a"], ordered=True)
        cat_rev_base = Categorical(
            ["b", "b", "b"], categories=["c", "b", "a"], ordered=True
        )
        cat = Categorical(["a", "b", "c"], ordered=True)
        cat_base = Categorical(["b", "b", "b"], categories=cat.categories, ordered=True)

        # comparisons need to take categories ordering into account
        res_rev = cat_rev > cat_rev_base
        exp_rev = np.array([True, False, False])
        tm.assert_numpy_array_equal(res_rev, exp_rev)

        res_rev = cat_rev < cat_rev_base
        exp_rev = np.array([False, False, True])
        tm.assert_numpy_array_equal(res_rev, exp_rev)

        res = cat > cat_base
        exp = np.array([False, False, True])
        tm.assert_numpy_array_equal(res, exp)

        # Only categories with same categories can be compared
        msg = "Categoricals can only be compared if 'categories' are the same"
        with pytest.raises(TypeError, match=msg):
            cat > cat_rev

        cat_rev_base2 = Categorical(["b", "b", "b"], categories=["c", "b", "a", "d"])

        with pytest.raises(TypeError, match=msg):
            cat_rev > cat_rev_base2

        # Only categories with same ordering information can be compared
        cat_unordered = cat.set_ordered(False)
        assert not (cat > cat).any()

        with pytest.raises(TypeError, match=msg):
            cat > cat_unordered

        # comparison (in both directions) with Series will raise
        s = Series(["b", "b", "b"], dtype=object)
        msg = (
            "Cannot compare a Categorical for op __gt__ with type "
            r"<class 'numpy\.ndarray'>"
        )
        with pytest.raises(TypeError, match=msg):
            cat > s
        with pytest.raises(TypeError, match=msg):
            cat_rev > s
        with pytest.raises(TypeError, match=msg):
            s < cat
        with pytest.raises(TypeError, match=msg):
            s < cat_rev

        # comparison with numpy.array will raise in both direction, but only on
        # newer numpy versions
        a = np.array(["b", "b", "b"], dtype=object)
        with pytest.raises(TypeError, match=msg):
            cat > a
        with pytest.raises(TypeError, match=msg):
            cat_rev > a

        # Make sure that unequal comparison take the categories order in
        # account
        cat_rev = Categorical(list("abc"), categories=list("cba"), ordered=True)
        exp = np.array([True, False, False])
        res = cat_rev > "b"
        tm.assert_numpy_array_equal(res, exp)

        # check that zero-dim array gets unboxed
        res = cat_rev > np.array("b")
        tm.assert_numpy_array_equal(res, exp)


class TestCategoricalOps:
    @pytest.mark.parametrize(
        "categories",
        [["a", "b"], [0, 1], [Timestamp("2019"), Timestamp("2020")]],
    )
    def test_not_equal_with_na(self, categories):
        # https://github.com/pandas-dev/pandas/issues/32276
        c1 = Categorical.from_codes([-1, 0], categories=categories)
        c2 = Categorical.from_codes([0, 1], categories=categories)

        result = c1 != c2

        assert result.all()

    def test_compare_frame(self):
        # GH#24282 check that Categorical.__cmp__(DataFrame) defers to frame
        data = ["a", "b", 2, "a"]
        cat = Categorical(data)

        df = DataFrame(cat)

        result = cat == df.T
        expected = DataFrame([[True, True, True, True]])
        tm.assert_frame_equal(result, expected)

        result = cat[::-1] != df.T
        expected = DataFrame([[False, True, True, False]])
        tm.assert_frame_equal(result, expected)

    def test_compare_frame_raises(self, comparison_op):
        # alignment raises unless we transpose
        op = comparison_op
        cat = Categorical(["a", "b", 2, "a"])
        df = DataFrame(cat)
        msg = "Unable to coerce to Series, length must be 1: given 4"
        with pytest.raises(ValueError, match=msg):
            op(cat, df)

    def test_datetime_categorical_comparison(self):
        dt_cat = Categorical(date_range("2014-01-01", periods=3), ordered=True)
        tm.assert_numpy_array_equal(dt_cat > dt_cat[0], np.array([False, True, True]))
        tm.assert_numpy_array_equal(dt_cat[0] < dt_cat, np.array([False, True, True]))

    def test_reflected_comparison_with_scalars(self):
        # GH8658
        cat = Categorical([1, 2, 3], ordered=True)
        tm.assert_numpy_array_equal(cat > cat[0], np.array([False, True, True]))
        tm.assert_numpy_array_equal(cat[0] < cat, np.array([False, True, True]))

    def test_comparison_with_unknown_scalars(self):
        # https://github.com/pandas-dev/pandas/issues/9836#issuecomment-92123057
        # and following comparisons with scalars not in categories should raise
        # for unequal comps, but not for equal/not equal
        cat = Categorical([1, 2, 3], ordered=True)

        msg = "Invalid comparison between dtype=category and int"
        with pytest.raises(TypeError, match=msg):
            cat < 4
        with pytest.raises(TypeError, match=msg):
            cat > 4
        with pytest.raises(TypeError, match=msg):
            4 < cat
        with pytest.raises(TypeError, match=msg):
            4 > cat

        tm.assert_numpy_array_equal(cat == 4, np.array([False, False, False]))
        tm.assert_numpy_array_equal(cat != 4, np.array([True, True, True]))

    def test_comparison_with_tuple(self):
        cat = Categorical(np.array(["foo", (0, 1), 3, (0, 1)], dtype=object))

        result = cat == "foo"
        expected = np.array([True, False, False, False], dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

        result = cat == (0, 1)
        expected = np.array([False, True, False, True], dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

        result = cat != (0, 1)
        tm.assert_numpy_array_equal(result, ~expected)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_comparison_of_ordered_categorical_with_nan_to_scalar(
        self, compare_operators_no_eq_ne
    ):
        # https://github.com/pandas-dev/pandas/issues/26504
        # BUG: fix ordered categorical comparison with missing values (#26504 )
        # and following comparisons with scalars in categories with missing
        # values should be evaluated as False

        cat = Categorical([1, 2, 3, None], categories=[1, 2, 3], ordered=True)
        scalar = 2
        expected = getattr(np.array(cat), compare_operators_no_eq_ne)(scalar)
        actual = getattr(cat, compare_operators_no_eq_ne)(scalar)
        tm.assert_numpy_array_equal(actual, expected)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_comparison_of_ordered_categorical_with_nan_to_listlike(
        self, compare_operators_no_eq_ne
    ):
        # https://github.com/pandas-dev/pandas/issues/26504
        # and following comparisons of missing values in ordered Categorical
        # with listlike should be evaluated as False

        cat = Categorical([1, 2, 3, None], categories=[1, 2, 3], ordered=True)
        other = Categorical([2, 2, 2, 2], categories=[1, 2, 3], ordered=True)
        expected = getattr(np.array(cat), compare_operators_no_eq_ne)(2)
        actual = getattr(cat, compare_operators_no_eq_ne)(other)
        tm.assert_numpy_array_equal(actual, expected)

    @pytest.mark.parametrize(
        "data,reverse,base",
        [(list("abc"), list("cba"), list("bbb")), ([1, 2, 3], [3, 2, 1], [2, 2, 2])],
    )
    def test_comparisons(self, data, reverse, base):
        cat_rev = Series(Categorical(data, categories=reverse, ordered=True))
        cat_rev_base = Series(Categorical(base, categories=reverse, ordered=True))
        cat = Series(Categorical(data, ordered=True))
        cat_base = Series(
            Categorical(base, categories=cat.cat.categories, ordered=True)
        )
        s = Series(base, dtype=object if base == list("bbb") else None)
        a = np.array(base)

        # comparisons need to take categories ordering into account
        res_rev = cat_rev > cat_rev_base
        exp_rev = Series([True, False, False])
        tm.assert_series_equal(res_rev, exp_rev)

        res_rev = cat_rev < cat_rev_base
        exp_rev = Series([False, False, True])
        tm.assert_series_equal(res_rev, exp_rev)

        res = cat > cat_base
        exp = Series([False, False, True])
        tm.assert_series_equal(res, exp)

        scalar = base[1]
        res = cat > scalar
        exp = Series([False, False, True])
        exp2 = cat.values > scalar
        tm.assert_series_equal(res, exp)
        tm.assert_numpy_array_equal(res.values, exp2)
        res_rev = cat_rev > scalar
        exp_rev = Series([True, False, False])
        exp_rev2 = cat_rev.values > scalar
        tm.assert_series_equal(res_rev, exp_rev)
        tm.assert_numpy_array_equal(res_rev.values, exp_rev2)

        # Only categories with same categories can be compared
        msg = "Categoricals can only be compared if 'categories' are the same"
        with pytest.raises(TypeError, match=msg):
            cat > cat_rev

        # categorical cannot be compared to Series or numpy array, and also
        # not the other way around
        msg = (
            "Cannot compare a Categorical for op __gt__ with type "
            r"<class 'numpy\.ndarray'>"
        )
        with pytest.raises(TypeError, match=msg):
            cat > s
        with pytest.raises(TypeError, match=msg):
            cat_rev > s
        with pytest.raises(TypeError, match=msg):
            cat > a
        with pytest.raises(TypeError, match=msg):
            cat_rev > a

        with pytest.raises(TypeError, match=msg):
            s < cat
        with pytest.raises(TypeError, match=msg):
            s < cat_rev

        with pytest.raises(TypeError, match=msg):
            a < cat
        with pytest.raises(TypeError, match=msg):
            a < cat_rev

    @pytest.mark.parametrize(
        "ctor",
        [
            lambda *args, **kwargs: Categorical(*args, **kwargs),
            lambda *args, **kwargs: Series(Categorical(*args, **kwargs)),
        ],
    )
    def test_unordered_different_order_equal(self, ctor):
        # https://github.com/pandas-dev/pandas/issues/16014
        c1 = ctor(["a", "b"], categories=["a", "b"], ordered=False)
        c2 = ctor(["a", "b"], categories=["b", "a"], ordered=False)
        assert (c1 == c2).all()

        c1 = ctor(["a", "b"], categories=["a", "b"], ordered=False)
        c2 = ctor(["b", "a"], categories=["b", "a"], ordered=False)
        assert (c1 != c2).all()

        c1 = ctor(["a", "a"], categories=["a", "b"], ordered=False)
        c2 = ctor(["b", "b"], categories=["b", "a"], ordered=False)
        assert (c1 != c2).all()

        c1 = ctor(["a", "a"], categories=["a", "b"], ordered=False)
        c2 = ctor(["a", "b"], categories=["b", "a"], ordered=False)
        result = c1 == c2
        tm.assert_numpy_array_equal(np.array(result), np.array([True, False]))

    def test_unordered_different_categories_raises(self):
        c1 = Categorical(["a", "b"], categories=["a", "b"], ordered=False)
        c2 = Categorical(["a", "c"], categories=["c", "a"], ordered=False)

        with pytest.raises(TypeError, match=("Categoricals can only be compared")):
            c1 == c2

    def test_compare_different_lengths(self):
        c1 = Categorical([], categories=["a", "b"])
        c2 = Categorical([], categories=["a"])

        msg = "Categoricals can only be compared if 'categories' are the same."
        with pytest.raises(TypeError, match=msg):
            c1 == c2

    def test_compare_unordered_different_order(self):
        # https://github.com/pandas-dev/pandas/issues/16603#issuecomment-
        # 349290078
        a = Categorical(["a"], categories=["a", "b"])
        b = Categorical(["b"], categories=["b", "a"])
        assert not a.equals(b)

    def test_numeric_like_ops(self):
        df = DataFrame({"value": np.random.default_rng(2).integers(0, 10000, 100)})
        labels = [f"{i} - {i + 499}" for i in range(0, 10000, 500)]
        cat_labels = Categorical(labels, labels)

        df = df.sort_values(by=["value"], ascending=True)
        df["value_group"] = pd.cut(
            df.value, range(0, 10500, 500), right=False, labels=cat_labels
        )

        # numeric ops should not succeed
        for op, str_rep in [
            ("__add__", r"\+"),
            ("__sub__", "-"),
            ("__mul__", r"\*"),
            ("__truediv__", "/"),
        ]:
            msg = f"Series cannot perform the operation {str_rep}|unsupported operand"
            with pytest.raises(TypeError, match=msg):
                getattr(df, op)(df)

        # reduction ops should not succeed (unless specifically defined, e.g.
        # min/max)
        s = df["value_group"]
        for op in ["kurt", "skew", "var", "std", "mean", "sum", "median"]:
            msg = f"does not support reduction '{op}'"
            with pytest.raises(TypeError, match=msg):
                getattr(s, op)(numeric_only=False)

    def test_numeric_like_ops_series(self):
        # numpy ops
        s = Series(Categorical([1, 2, 3, 4]))
        with pytest.raises(TypeError, match="does not support reduction 'sum'"):
            np.sum(s)

    @pytest.mark.parametrize(
        "op, str_rep",
        [
            ("__add__", r"\+"),
            ("__sub__", "-"),
            ("__mul__", r"\*"),
            ("__truediv__", "/"),
        ],
    )
    def test_numeric_like_ops_series_arith(self, op, str_rep):
        # numeric ops on a Series
        s = Series(Categorical([1, 2, 3, 4]))
        msg = f"Series cannot perform the operation {str_rep}|unsupported operand"
        with pytest.raises(TypeError, match=msg):
            getattr(s, op)(2)

    def test_numeric_like_ops_series_invalid(self):
        # invalid ufunc
        s = Series(Categorical([1, 2, 3, 4]))
        msg = "Object with dtype category cannot perform the numpy op log"
        with pytest.raises(TypeError, match=msg):
            np.log(s)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\generic\test_duplicate_labels.py ===
"""Tests dealing with the NDFrame.allows_duplicates."""
import operator

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm

not_implemented = pytest.mark.xfail(reason="Not implemented.")

# ----------------------------------------------------------------------------
# Preservation


class TestPreserves:
    @pytest.mark.parametrize(
        "cls, data",
        [
            (pd.Series, np.array([])),
            (pd.Series, [1, 2]),
            (pd.DataFrame, {}),
            (pd.DataFrame, {"A": [1, 2]}),
        ],
    )
    def test_construction_ok(self, cls, data):
        result = cls(data)
        assert result.flags.allows_duplicate_labels is True

        result = cls(data).set_flags(allows_duplicate_labels=False)
        assert result.flags.allows_duplicate_labels is False

    @pytest.mark.parametrize(
        "func",
        [
            operator.itemgetter(["a"]),
            operator.methodcaller("add", 1),
            operator.methodcaller("rename", str.upper),
            operator.methodcaller("rename", "name"),
            operator.methodcaller("abs"),
            np.abs,
        ],
    )
    def test_preserved_series(self, func):
        s = pd.Series([0, 1], index=["a", "b"]).set_flags(allows_duplicate_labels=False)
        assert func(s).flags.allows_duplicate_labels is False

    @pytest.mark.parametrize(
        "other", [pd.Series(0, index=["a", "b", "c"]), pd.Series(0, index=["a", "b"])]
    )
    # TODO: frame
    @not_implemented
    def test_align(self, other):
        s = pd.Series([0, 1], index=["a", "b"]).set_flags(allows_duplicate_labels=False)
        a, b = s.align(other)
        assert a.flags.allows_duplicate_labels is False
        assert b.flags.allows_duplicate_labels is False

    def test_preserved_frame(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["a", "b"]).set_flags(
            allows_duplicate_labels=False
        )
        assert df.loc[["a"]].flags.allows_duplicate_labels is False
        assert df.loc[:, ["A", "B"]].flags.allows_duplicate_labels is False

    def test_to_frame(self):
        ser = pd.Series(dtype=float).set_flags(allows_duplicate_labels=False)
        assert ser.to_frame().flags.allows_duplicate_labels is False

    @pytest.mark.parametrize("func", ["add", "sub"])
    @pytest.mark.parametrize("frame", [False, True])
    @pytest.mark.parametrize("other", [1, pd.Series([1, 2], name="A")])
    def test_binops(self, func, other, frame):
        df = pd.Series([1, 2], name="A", index=["a", "b"]).set_flags(
            allows_duplicate_labels=False
        )
        if frame:
            df = df.to_frame()
        if isinstance(other, pd.Series) and frame:
            other = other.to_frame()
        func = operator.methodcaller(func, other)
        assert df.flags.allows_duplicate_labels is False
        assert func(df).flags.allows_duplicate_labels is False

    def test_preserve_getitem(self):
        df = pd.DataFrame({"A": [1, 2]}).set_flags(allows_duplicate_labels=False)
        assert df[["A"]].flags.allows_duplicate_labels is False
        assert df["A"].flags.allows_duplicate_labels is False
        assert df.loc[0].flags.allows_duplicate_labels is False
        assert df.loc[[0]].flags.allows_duplicate_labels is False
        assert df.loc[0, ["A"]].flags.allows_duplicate_labels is False

    def test_ndframe_getitem_caching_issue(
        self, request, using_copy_on_write, warn_copy_on_write
    ):
        if not (using_copy_on_write or warn_copy_on_write):
            request.applymarker(pytest.mark.xfail(reason="Unclear behavior."))
        # NDFrame.__getitem__ will cache the first df['A']. May need to
        # invalidate that cache? Update the cached entries?
        df = pd.DataFrame({"A": [0]}).set_flags(allows_duplicate_labels=False)
        assert df["A"].flags.allows_duplicate_labels is False
        df.flags.allows_duplicate_labels = True
        assert df["A"].flags.allows_duplicate_labels is True

    @pytest.mark.parametrize(
        "objs, kwargs",
        [
            # Series
            (
                [
                    pd.Series(1, index=["a", "b"]),
                    pd.Series(2, index=["c", "d"]),
                ],
                {},
            ),
            (
                [
                    pd.Series(1, index=["a", "b"]),
                    pd.Series(2, index=["a", "b"]),
                ],
                {"ignore_index": True},
            ),
            (
                [
                    pd.Series(1, index=["a", "b"]),
                    pd.Series(2, index=["a", "b"]),
                ],
                {"axis": 1},
            ),
            # Frame
            (
                [
                    pd.DataFrame({"A": [1, 2]}, index=["a", "b"]),
                    pd.DataFrame({"A": [1, 2]}, index=["c", "d"]),
                ],
                {},
            ),
            (
                [
                    pd.DataFrame({"A": [1, 2]}, index=["a", "b"]),
                    pd.DataFrame({"A": [1, 2]}, index=["a", "b"]),
                ],
                {"ignore_index": True},
            ),
            (
                [
                    pd.DataFrame({"A": [1, 2]}, index=["a", "b"]),
                    pd.DataFrame({"B": [1, 2]}, index=["a", "b"]),
                ],
                {"axis": 1},
            ),
            # Series / Frame
            (
                [
                    pd.DataFrame({"A": [1, 2]}, index=["a", "b"]),
                    pd.Series([1, 2], index=["a", "b"], name="B"),
                ],
                {"axis": 1},
            ),
        ],
    )
    def test_concat(self, objs, kwargs):
        objs = [x.set_flags(allows_duplicate_labels=False) for x in objs]
        result = pd.concat(objs, **kwargs)
        assert result.flags.allows_duplicate_labels is False

    @pytest.mark.parametrize(
        "left, right, expected",
        [
            # false false false
            pytest.param(
                pd.DataFrame({"A": [0, 1]}, index=["a", "b"]).set_flags(
                    allows_duplicate_labels=False
                ),
                pd.DataFrame({"B": [0, 1]}, index=["a", "d"]).set_flags(
                    allows_duplicate_labels=False
                ),
                False,
                marks=not_implemented,
            ),
            # false true false
            pytest.param(
                pd.DataFrame({"A": [0, 1]}, index=["a", "b"]).set_flags(
                    allows_duplicate_labels=False
                ),
                pd.DataFrame({"B": [0, 1]}, index=["a", "d"]),
                False,
                marks=not_implemented,
            ),
            # true true true
            (
                pd.DataFrame({"A": [0, 1]}, index=["a", "b"]),
                pd.DataFrame({"B": [0, 1]}, index=["a", "d"]),
                True,
            ),
        ],
    )
    def test_merge(self, left, right, expected):
        result = pd.merge(left, right, left_index=True, right_index=True)
        assert result.flags.allows_duplicate_labels is expected

    @not_implemented
    def test_groupby(self):
        # XXX: This is under tested
        # TODO:
        #  - apply
        #  - transform
        #  - Should passing a grouper that disallows duplicates propagate?
        df = pd.DataFrame({"A": [1, 2, 3]}).set_flags(allows_duplicate_labels=False)
        result = df.groupby([0, 0, 1]).agg("count")
        assert result.flags.allows_duplicate_labels is False

    @pytest.mark.parametrize("frame", [True, False])
    @not_implemented
    def test_window(self, frame):
        df = pd.Series(
            1,
            index=pd.date_range("2000", periods=12),
            name="A",
            allows_duplicate_labels=False,
        )
        if frame:
            df = df.to_frame()
        assert df.rolling(3).mean().flags.allows_duplicate_labels is False
        assert df.ewm(3).mean().flags.allows_duplicate_labels is False
        assert df.expanding(3).mean().flags.allows_duplicate_labels is False


# ----------------------------------------------------------------------------
# Raises


class TestRaises:
    @pytest.mark.parametrize(
        "cls, axes",
        [
            (pd.Series, {"index": ["a", "a"], "dtype": float}),
            (pd.DataFrame, {"index": ["a", "a"]}),
            (pd.DataFrame, {"index": ["a", "a"], "columns": ["b", "b"]}),
            (pd.DataFrame, {"columns": ["b", "b"]}),
        ],
    )
    def test_set_flags_with_duplicates(self, cls, axes):
        result = cls(**axes)
        assert result.flags.allows_duplicate_labels is True

        msg = "Index has duplicates."
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            cls(**axes).set_flags(allows_duplicate_labels=False)

    @pytest.mark.parametrize(
        "data",
        [
            pd.Series(index=[0, 0], dtype=float),
            pd.DataFrame(index=[0, 0]),
            pd.DataFrame(columns=[0, 0]),
        ],
    )
    def test_setting_allows_duplicate_labels_raises(self, data):
        msg = "Index has duplicates."
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            data.flags.allows_duplicate_labels = False

        assert data.flags.allows_duplicate_labels is True

    def test_series_raises(self):
        a = pd.Series(0, index=["a", "b"])
        b = pd.Series([0, 1], index=["a", "b"]).set_flags(allows_duplicate_labels=False)
        msg = "Index has duplicates."
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            pd.concat([a, b])

    @pytest.mark.parametrize(
        "getter, target",
        [
            (operator.itemgetter(["A", "A"]), None),
            # loc
            (operator.itemgetter(["a", "a"]), "loc"),
            pytest.param(operator.itemgetter(("a", ["A", "A"])), "loc"),
            (operator.itemgetter((["a", "a"], "A")), "loc"),
            # iloc
            (operator.itemgetter([0, 0]), "iloc"),
            pytest.param(operator.itemgetter((0, [0, 0])), "iloc"),
            pytest.param(operator.itemgetter(([0, 0], 0)), "iloc"),
        ],
    )
    def test_getitem_raises(self, getter, target):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["a", "b"]).set_flags(
            allows_duplicate_labels=False
        )
        if target:
            # df, df.loc, or df.iloc
            target = getattr(df, target)
        else:
            target = df

        msg = "Index has duplicates."
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            getter(target)

    @pytest.mark.parametrize(
        "objs, kwargs",
        [
            (
                [
                    pd.Series(1, index=[0, 1], name="a"),
                    pd.Series(2, index=[0, 1], name="a"),
                ],
                {"axis": 1},
            )
        ],
    )
    def test_concat_raises(self, objs, kwargs):
        objs = [x.set_flags(allows_duplicate_labels=False) for x in objs]
        msg = "Index has duplicates."
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            pd.concat(objs, **kwargs)

    @not_implemented
    def test_merge_raises(self):
        a = pd.DataFrame({"A": [0, 1, 2]}, index=["a", "b", "c"]).set_flags(
            allows_duplicate_labels=False
        )
        b = pd.DataFrame({"B": [0, 1, 2]}, index=["a", "b", "b"])
        msg = "Index has duplicates."
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            pd.merge(a, b, left_index=True, right_index=True)


@pytest.mark.parametrize(
    "idx",
    [
        pd.Index([1, 1]),
        pd.Index(["a", "a"]),
        pd.Index([1.1, 1.1]),
        pd.PeriodIndex([pd.Period("2000", "D")] * 2),
        pd.DatetimeIndex([pd.Timestamp("2000")] * 2),
        pd.TimedeltaIndex([pd.Timedelta("1D")] * 2),
        pd.CategoricalIndex(["a", "a"]),
        pd.IntervalIndex([pd.Interval(0, 1)] * 2),
        pd.MultiIndex.from_tuples([("a", 1), ("a", 1)]),
    ],
    ids=lambda x: type(x).__name__,
)
def test_raises_basic(idx):
    msg = "Index has duplicates."
    with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
        pd.Series(1, index=idx).set_flags(allows_duplicate_labels=False)

    with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
        pd.DataFrame({"A": [1, 1]}, index=idx).set_flags(allows_duplicate_labels=False)

    with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
        pd.DataFrame([[1, 2]], columns=idx).set_flags(allows_duplicate_labels=False)


def test_format_duplicate_labels_message():
    idx = pd.Index(["a", "b", "a", "b", "c"])
    result = idx._format_duplicate_message()
    expected = pd.DataFrame(
        {"positions": [[0, 2], [1, 3]]}, index=pd.Index(["a", "b"], name="label")
    )
    tm.assert_frame_equal(result, expected)


def test_format_duplicate_labels_message_multi():
    idx = pd.MultiIndex.from_product([["A"], ["a", "b", "a", "b", "c"]])
    result = idx._format_duplicate_message()
    expected = pd.DataFrame(
        {"positions": [[0, 2], [1, 3]]},
        index=pd.MultiIndex.from_product([["A"], ["a", "b"]]),
    )
    tm.assert_frame_equal(result, expected)


def test_dataframe_insert_raises():
    df = pd.DataFrame({"A": [1, 2]}).set_flags(allows_duplicate_labels=False)
    msg = "Cannot specify"
    with pytest.raises(ValueError, match=msg):
        df.insert(0, "A", [3, 4], allow_duplicates=True)


@pytest.mark.parametrize(
    "method, frame_only",
    [
        (operator.methodcaller("set_index", "A", inplace=True), True),
        (operator.methodcaller("reset_index", inplace=True), True),
        (operator.methodcaller("rename", lambda x: x, inplace=True), False),
    ],
)
def test_inplace_raises(method, frame_only):
    df = pd.DataFrame({"A": [0, 0], "B": [1, 2]}).set_flags(
        allows_duplicate_labels=False
    )
    s = df["A"]
    s.flags.allows_duplicate_labels = False
    msg = "Cannot specify"

    with pytest.raises(ValueError, match=msg):
        method(df)
    if not frame_only:
        with pytest.raises(ValueError, match=msg):
            method(s)


def test_pickle():
    a = pd.Series([1, 2]).set_flags(allows_duplicate_labels=False)
    b = tm.round_trip_pickle(a)
    tm.assert_series_equal(a, b)

    a = pd.DataFrame({"A": []}).set_flags(allows_duplicate_labels=False)
    b = tm.round_trip_pickle(a)
    tm.assert_frame_equal(a, b)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\multiindex\test_getitem.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
)
import pandas._testing as tm
from pandas.core.indexing import IndexingError

# ----------------------------------------------------------------------------
# test indexing of Series with multi-level Index
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "access_method",
    [lambda s, x: s[:, x], lambda s, x: s.loc[:, x], lambda s, x: s.xs(x, level=1)],
)
@pytest.mark.parametrize(
    "level1_value, expected",
    [(0, Series([1], index=[0])), (1, Series([2, 3], index=[1, 2]))],
)
def test_series_getitem_multiindex(access_method, level1_value, expected):
    # GH 6018
    # series regression getitem with a multi-index

    mi = MultiIndex.from_tuples([(0, 0), (1, 1), (2, 1)], names=["A", "B"])
    ser = Series([1, 2, 3], index=mi)
    expected.index.name = "A"

    result = access_method(ser, level1_value)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("level0_value", ["D", "A"])
def test_series_getitem_duplicates_multiindex(level0_value):
    # GH 5725 the 'A' happens to be a valid Timestamp so the doesn't raise
    # the appropriate error, only in PY3 of course!

    index = MultiIndex(
        levels=[[level0_value, "B", "C"], [0, 26, 27, 37, 57, 67, 75, 82]],
        codes=[[0, 0, 0, 1, 2, 2, 2, 2, 2, 2], [1, 3, 4, 6, 0, 2, 2, 3, 5, 7]],
        names=["tag", "day"],
    )
    arr = np.random.default_rng(2).standard_normal((len(index), 1))
    df = DataFrame(arr, index=index, columns=["val"])

    # confirm indexing on missing value raises KeyError
    if level0_value != "A":
        with pytest.raises(KeyError, match=r"^'A'$"):
            df.val["A"]

    with pytest.raises(KeyError, match=r"^'X'$"):
        df.val["X"]

    result = df.val[level0_value]
    expected = Series(
        arr.ravel()[0:3], name="val", index=Index([26, 37, 57], name="day")
    )
    tm.assert_series_equal(result, expected)


def test_series_getitem(multiindex_year_month_day_dataframe_random_data, indexer_sl):
    s = multiindex_year_month_day_dataframe_random_data["A"]
    expected = s.reindex(s.index[42:65])
    expected.index = expected.index.droplevel(0).droplevel(0)

    result = indexer_sl(s)[2000, 3]
    tm.assert_series_equal(result, expected)


def test_series_getitem_returns_scalar(
    multiindex_year_month_day_dataframe_random_data, indexer_sl
):
    s = multiindex_year_month_day_dataframe_random_data["A"]
    expected = s.iloc[49]

    result = indexer_sl(s)[2000, 3, 10]
    assert result == expected


@pytest.mark.parametrize(
    "indexer,expected_error,expected_error_msg",
    [
        (lambda s: s.__getitem__((2000, 3, 4)), KeyError, r"^\(2000, 3, 4\)$"),
        (lambda s: s[(2000, 3, 4)], KeyError, r"^\(2000, 3, 4\)$"),
        (lambda s: s.loc[(2000, 3, 4)], KeyError, r"^\(2000, 3, 4\)$"),
        (lambda s: s.loc[(2000, 3, 4, 5)], IndexingError, "Too many indexers"),
        (lambda s: s.__getitem__(len(s)), KeyError, ""),  # match should include len(s)
        (lambda s: s[len(s)], KeyError, ""),  # match should include len(s)
        (
            lambda s: s.iloc[len(s)],
            IndexError,
            "single positional indexer is out-of-bounds",
        ),
    ],
)
def test_series_getitem_indexing_errors(
    multiindex_year_month_day_dataframe_random_data,
    indexer,
    expected_error,
    expected_error_msg,
):
    s = multiindex_year_month_day_dataframe_random_data["A"]
    with pytest.raises(expected_error, match=expected_error_msg):
        indexer(s)


def test_series_getitem_corner_generator(
    multiindex_year_month_day_dataframe_random_data,
):
    s = multiindex_year_month_day_dataframe_random_data["A"]
    result = s[(x > 0 for x in s)]
    expected = s[s > 0]
    tm.assert_series_equal(result, expected)


# ----------------------------------------------------------------------------
# test indexing of DataFrame with multi-level Index
# ----------------------------------------------------------------------------


def test_getitem_simple(multiindex_dataframe_random_data):
    df = multiindex_dataframe_random_data.T
    expected = df.values[:, 0]
    result = df["foo", "one"].values
    tm.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "indexer,expected_error_msg",
    [
        (lambda df: df[("foo", "four")], r"^\('foo', 'four'\)$"),
        (lambda df: df["foobar"], r"^'foobar'$"),
    ],
)
def test_frame_getitem_simple_key_error(
    multiindex_dataframe_random_data, indexer, expected_error_msg
):
    df = multiindex_dataframe_random_data.T
    with pytest.raises(KeyError, match=expected_error_msg):
        indexer(df)


def test_tuple_string_column_names():
    # GH#50372
    mi = MultiIndex.from_tuples([("a", "aa"), ("a", "ab"), ("b", "ba"), ("b", "bb")])
    df = DataFrame([range(4), range(1, 5), range(2, 6)], columns=mi)
    df["single_index"] = 0

    df_flat = df.copy()
    df_flat.columns = df_flat.columns.to_flat_index()
    df_flat["new_single_index"] = 0

    result = df_flat[[("a", "aa"), "new_single_index"]]
    expected = DataFrame(
        [[0, 0], [1, 0], [2, 0]], columns=Index([("a", "aa"), "new_single_index"])
    )
    tm.assert_frame_equal(result, expected)


def test_frame_getitem_multicolumn_empty_level():
    df = DataFrame({"a": ["1", "2", "3"], "b": ["2", "3", "4"]})
    df.columns = [
        ["level1 item1", "level1 item2"],
        ["", "level2 item2"],
        ["level3 item1", "level3 item2"],
    ]

    result = df["level1 item1"]
    expected = DataFrame(
        [["1"], ["2"], ["3"]], index=df.index, columns=["level3 item1"]
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "indexer,expected_slice",
    [
        (lambda df: df["foo"], slice(3)),
        (lambda df: df["bar"], slice(3, 5)),
        (lambda df: df.loc[:, "bar"], slice(3, 5)),
    ],
)
def test_frame_getitem_toplevel(
    multiindex_dataframe_random_data, indexer, expected_slice
):
    df = multiindex_dataframe_random_data.T
    expected = df.reindex(columns=df.columns[expected_slice])
    expected.columns = expected.columns.droplevel(0)
    result = indexer(df)
    tm.assert_frame_equal(result, expected)


def test_frame_mixed_depth_get():
    arrays = [
        ["a", "top", "top", "routine1", "routine1", "routine2"],
        ["", "OD", "OD", "result1", "result2", "result1"],
        ["", "wx", "wy", "", "", ""],
    ]

    tuples = sorted(zip(*arrays))
    index = MultiIndex.from_tuples(tuples)
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 6)), columns=index)

    result = df["a"]
    expected = df["a", "", ""].rename("a")
    tm.assert_series_equal(result, expected)

    result = df["routine1", "result1"]
    expected = df["routine1", "result1", ""]
    expected = expected.rename(("routine1", "result1"))
    tm.assert_series_equal(result, expected)


def test_frame_getitem_nan_multiindex(nulls_fixture):
    # GH#29751
    # loc on a multiindex containing nan values
    n = nulls_fixture  # for code readability
    cols = ["a", "b", "c"]
    df = DataFrame(
        [[11, n, 13], [21, n, 23], [31, n, 33], [41, n, 43]],
        columns=cols,
    ).set_index(["a", "b"])
    df["c"] = df["c"].astype("int64")

    idx = (21, n)
    result = df.loc[:idx]
    expected = DataFrame([[11, n, 13], [21, n, 23]], columns=cols).set_index(["a", "b"])
    expected["c"] = expected["c"].astype("int64")
    tm.assert_frame_equal(result, expected)

    result = df.loc[idx:]
    expected = DataFrame(
        [[21, n, 23], [31, n, 33], [41, n, 43]], columns=cols
    ).set_index(["a", "b"])
    expected["c"] = expected["c"].astype("int64")
    tm.assert_frame_equal(result, expected)

    idx1, idx2 = (21, n), (31, n)
    result = df.loc[idx1:idx2]
    expected = DataFrame([[21, n, 23], [31, n, 33]], columns=cols).set_index(["a", "b"])
    expected["c"] = expected["c"].astype("int64")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "indexer,expected",
    [
        (
            (["b"], ["bar", np.nan]),
            (
                DataFrame(
                    [[2, 3], [5, 6]],
                    columns=MultiIndex.from_tuples([("b", "bar"), ("b", np.nan)]),
                    dtype="int64",
                )
            ),
        ),
        (
            (["a", "b"]),
            (
                DataFrame(
                    [[1, 2, 3], [4, 5, 6]],
                    columns=MultiIndex.from_tuples(
                        [("a", "foo"), ("b", "bar"), ("b", np.nan)]
                    ),
                    dtype="int64",
                )
            ),
        ),
        (
            (["b"]),
            (
                DataFrame(
                    [[2, 3], [5, 6]],
                    columns=MultiIndex.from_tuples([("b", "bar"), ("b", np.nan)]),
                    dtype="int64",
                )
            ),
        ),
        (
            (["b"], ["bar"]),
            (
                DataFrame(
                    [[2], [5]],
                    columns=MultiIndex.from_tuples([("b", "bar")]),
                    dtype="int64",
                )
            ),
        ),
        (
            (["b"], [np.nan]),
            (
                DataFrame(
                    [[3], [6]],
                    columns=MultiIndex(
                        codes=[[1], [-1]], levels=[["a", "b"], ["bar", "foo"]]
                    ),
                    dtype="int64",
                )
            ),
        ),
        (("b", np.nan), Series([3, 6], dtype="int64", name=("b", np.nan))),
    ],
)
def test_frame_getitem_nan_cols_multiindex(
    indexer,
    expected,
    nulls_fixture,
):
    # Slicing MultiIndex including levels with nan values, for more information
    # see GH#25154
    df = DataFrame(
        [[1, 2, 3], [4, 5, 6]],
        columns=MultiIndex.from_tuples(
            [("a", "foo"), ("b", "bar"), ("b", nulls_fixture)]
        ),
        dtype="int64",
    )

    result = df.loc[:, indexer]
    tm.assert_equal(result, expected)


# ----------------------------------------------------------------------------
# test indexing of DataFrame with multi-level Index with duplicates
# ----------------------------------------------------------------------------


@pytest.fixture
def dataframe_with_duplicate_index():
    """Fixture for DataFrame used in tests for gh-4145 and gh-4146"""
    data = [["a", "d", "e", "c", "f", "b"], [1, 4, 5, 3, 6, 2], [1, 4, 5, 3, 6, 2]]
    index = ["h1", "h3", "h5"]
    columns = MultiIndex(
        levels=[["A", "B"], ["A1", "A2", "B1", "B2"]],
        codes=[[0, 0, 0, 1, 1, 1], [0, 3, 3, 0, 1, 2]],
        names=["main", "sub"],
    )
    return DataFrame(data, index=index, columns=columns)


@pytest.mark.parametrize(
    "indexer", [lambda df: df[("A", "A1")], lambda df: df.loc[:, ("A", "A1")]]
)
def test_frame_mi_access(dataframe_with_duplicate_index, indexer):
    # GH 4145
    df = dataframe_with_duplicate_index
    index = Index(["h1", "h3", "h5"])
    columns = MultiIndex.from_tuples([("A", "A1")], names=["main", "sub"])
    expected = DataFrame([["a", 1, 1]], index=columns, columns=index).T

    result = indexer(df)
    tm.assert_frame_equal(result, expected)


def test_frame_mi_access_returns_series(dataframe_with_duplicate_index):
    # GH 4146, not returning a block manager when selecting a unique index
    # from a duplicate index
    # as of 4879, this returns a Series (which is similar to what happens
    # with a non-unique)
    df = dataframe_with_duplicate_index
    expected = Series(["a", 1, 1], index=["h1", "h3", "h5"], name="A1")
    result = df["A"]["A1"]
    tm.assert_series_equal(result, expected)


def test_frame_mi_access_returns_frame(dataframe_with_duplicate_index):
    # selecting a non_unique from the 2nd level
    df = dataframe_with_duplicate_index
    expected = DataFrame(
        [["d", 4, 4], ["e", 5, 5]],
        index=Index(["B2", "B2"], name="sub"),
        columns=["h1", "h3", "h5"],
    ).T
    result = df["A"]["B2"]
    tm.assert_frame_equal(result, expected)


def test_frame_mi_empty_slice():
    # GH 15454
    df = DataFrame(0, index=range(2), columns=MultiIndex.from_product([[1], [2]]))
    result = df[[]]
    expected = DataFrame(
        index=[0, 1], columns=MultiIndex(levels=[[1], [2]], codes=[[], []])
    )
    tm.assert_frame_equal(result, expected)


def test_loc_empty_multiindex():
    # GH#36936
    arrays = [["a", "a", "b", "a"], ["a", "a", "b", "b"]]
    index = MultiIndex.from_arrays(arrays, names=("idx1", "idx2"))
    df = DataFrame([1, 2, 3, 4], index=index, columns=["value"])

    # loc on empty multiindex == loc with False mask
    empty_multiindex = df.loc[df.loc[:, "value"] == 0, :].index
    result = df.loc[empty_multiindex, :]
    expected = df.loc[[False] * len(df.index), :]
    tm.assert_frame_equal(result, expected)

    # replacing value with loc on empty multiindex
    df.loc[df.loc[df.loc[:, "value"] == 0].index, "value"] = 5
    result = df
    expected = DataFrame([1, 2, 3, 4], index=index, columns=["value"])
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\plotting\test_converter.py ===
from datetime import (
    date,
    datetime,
)
import subprocess
import sys

import numpy as np
import pytest

import pandas._config.config as cf

from pandas._libs.tslibs import to_offset

from pandas import (
    Index,
    Period,
    PeriodIndex,
    Series,
    Timestamp,
    arrays,
    date_range,
)
import pandas._testing as tm

from pandas.plotting import (
    deregister_matplotlib_converters,
    register_matplotlib_converters,
)
from pandas.tseries.offsets import (
    Day,
    Micro,
    Milli,
    Second,
)

try:
    from pandas.plotting._matplotlib import converter
except ImportError:
    # try / except, rather than skip, to avoid internal refactoring
    # causing an improper skip
    pass

pytest.importorskip("matplotlib.pyplot")
dates = pytest.importorskip("matplotlib.dates")


@pytest.mark.single_cpu
def test_registry_mpl_resets():
    # Check that Matplotlib converters are properly reset (see issue #27481)
    code = (
        "import matplotlib.units as units; "
        "import matplotlib.dates as mdates; "
        "n_conv = len(units.registry); "
        "import pandas as pd; "
        "pd.plotting.register_matplotlib_converters(); "
        "pd.plotting.deregister_matplotlib_converters(); "
        "assert len(units.registry) == n_conv"
    )
    call = [sys.executable, "-c", code]
    subprocess.check_output(call)


def test_timtetonum_accepts_unicode():
    assert converter.time2num("00:01") == converter.time2num("00:01")


class TestRegistration:
    @pytest.mark.single_cpu
    def test_dont_register_by_default(self):
        # Run in subprocess to ensure a clean state
        code = (
            "import matplotlib.units; "
            "import pandas as pd; "
            "units = dict(matplotlib.units.registry); "
            "assert pd.Timestamp not in units"
        )
        call = [sys.executable, "-c", code]
        assert subprocess.check_call(call) == 0

    def test_registering_no_warning(self):
        plt = pytest.importorskip("matplotlib.pyplot")
        s = Series(range(12), index=date_range("2017", periods=12))
        _, ax = plt.subplots()

        # Set to the "warn" state, in case this isn't the first test run
        register_matplotlib_converters()
        ax.plot(s.index, s.values)
        plt.close()

    def test_pandas_plots_register(self):
        plt = pytest.importorskip("matplotlib.pyplot")
        s = Series(range(12), index=date_range("2017", periods=12))
        # Set to the "warn" state, in case this isn't the first test run
        with tm.assert_produces_warning(None) as w:
            s.plot()

        try:
            assert len(w) == 0
        finally:
            plt.close()

    def test_matplotlib_formatters(self):
        units = pytest.importorskip("matplotlib.units")

        # Can't make any assertion about the start state.
        # We we check that toggling converters off removes it, and toggling it
        # on restores it.

        with cf.option_context("plotting.matplotlib.register_converters", True):
            with cf.option_context("plotting.matplotlib.register_converters", False):
                assert Timestamp not in units.registry
            assert Timestamp in units.registry

    def test_option_no_warning(self):
        pytest.importorskip("matplotlib.pyplot")
        ctx = cf.option_context("plotting.matplotlib.register_converters", False)
        plt = pytest.importorskip("matplotlib.pyplot")
        s = Series(range(12), index=date_range("2017", periods=12))
        _, ax = plt.subplots()

        # Test without registering first, no warning
        with ctx:
            ax.plot(s.index, s.values)

        # Now test with registering
        register_matplotlib_converters()
        with ctx:
            ax.plot(s.index, s.values)
        plt.close()

    def test_registry_resets(self):
        units = pytest.importorskip("matplotlib.units")
        dates = pytest.importorskip("matplotlib.dates")

        # make a copy, to reset to
        original = dict(units.registry)

        try:
            # get to a known state
            units.registry.clear()
            date_converter = dates.DateConverter()
            units.registry[datetime] = date_converter
            units.registry[date] = date_converter

            register_matplotlib_converters()
            assert units.registry[date] is not date_converter
            deregister_matplotlib_converters()
            assert units.registry[date] is date_converter

        finally:
            # restore original stater
            units.registry.clear()
            for k, v in original.items():
                units.registry[k] = v


class TestDateTimeConverter:
    @pytest.fixture
    def dtc(self):
        return converter.DatetimeConverter()

    def test_convert_accepts_unicode(self, dtc):
        r1 = dtc.convert("2000-01-01 12:22", None, None)
        r2 = dtc.convert("2000-01-01 12:22", None, None)
        assert r1 == r2, "DatetimeConverter.convert should accept unicode"

    def test_conversion(self, dtc):
        rs = dtc.convert(["2012-1-1"], None, None)[0]
        xp = dates.date2num(datetime(2012, 1, 1))
        assert rs == xp

        rs = dtc.convert("2012-1-1", None, None)
        assert rs == xp

        rs = dtc.convert(date(2012, 1, 1), None, None)
        assert rs == xp

        rs = dtc.convert("2012-1-1", None, None)
        assert rs == xp

        rs = dtc.convert(Timestamp("2012-1-1"), None, None)
        assert rs == xp

        # also testing datetime64 dtype (GH8614)
        rs = dtc.convert("2012-01-01", None, None)
        assert rs == xp

        rs = dtc.convert("2012-01-01 00:00:00+0000", None, None)
        assert rs == xp

        rs = dtc.convert(
            np.array(["2012-01-01 00:00:00+0000", "2012-01-02 00:00:00+0000"]),
            None,
            None,
        )
        assert rs[0] == xp

        # we have a tz-aware date (constructed to that when we turn to utc it
        # is the same as our sample)
        ts = Timestamp("2012-01-01").tz_localize("UTC").tz_convert("US/Eastern")
        rs = dtc.convert(ts, None, None)
        assert rs == xp

        rs = dtc.convert(ts.to_pydatetime(), None, None)
        assert rs == xp

        rs = dtc.convert(Index([ts - Day(1), ts]), None, None)
        assert rs[1] == xp

        rs = dtc.convert(Index([ts - Day(1), ts]).to_pydatetime(), None, None)
        assert rs[1] == xp

    def test_conversion_float(self, dtc):
        rtol = 0.5 * 10**-9

        rs = dtc.convert(Timestamp("2012-1-1 01:02:03", tz="UTC"), None, None)
        xp = converter.mdates.date2num(Timestamp("2012-1-1 01:02:03", tz="UTC"))
        tm.assert_almost_equal(rs, xp, rtol=rtol)

        rs = dtc.convert(
            Timestamp("2012-1-1 09:02:03", tz="Asia/Hong_Kong"), None, None
        )
        tm.assert_almost_equal(rs, xp, rtol=rtol)

        rs = dtc.convert(datetime(2012, 1, 1, 1, 2, 3), None, None)
        tm.assert_almost_equal(rs, xp, rtol=rtol)

    @pytest.mark.parametrize(
        "values",
        [
            [date(1677, 1, 1), date(1677, 1, 2)],
            [datetime(1677, 1, 1, 12), datetime(1677, 1, 2, 12)],
        ],
    )
    def test_conversion_outofbounds_datetime(self, dtc, values):
        # 2579
        rs = dtc.convert(values, None, None)
        xp = converter.mdates.date2num(values)
        tm.assert_numpy_array_equal(rs, xp)
        rs = dtc.convert(values[0], None, None)
        xp = converter.mdates.date2num(values[0])
        assert rs == xp

    @pytest.mark.parametrize(
        "time,format_expected",
        [
            (0, "00:00"),  # time2num(datetime.time.min)
            (86399.999999, "23:59:59.999999"),  # time2num(datetime.time.max)
            (90000, "01:00"),
            (3723, "01:02:03"),
            (39723.2, "11:02:03.200"),
        ],
    )
    def test_time_formatter(self, time, format_expected):
        # issue 18478
        result = converter.TimeFormatter(None)(time)
        assert result == format_expected

    @pytest.mark.parametrize("freq", ("B", "ms", "s"))
    def test_dateindex_conversion(self, freq, dtc):
        rtol = 10**-9
        dateindex = date_range("2020-01-01", periods=10, freq=freq)
        rs = dtc.convert(dateindex, None, None)
        xp = converter.mdates.date2num(dateindex._mpl_repr())
        tm.assert_almost_equal(rs, xp, rtol=rtol)

    @pytest.mark.parametrize("offset", [Second(), Milli(), Micro(50)])
    def test_resolution(self, offset, dtc):
        # Matplotlib's time representation using floats cannot distinguish
        # intervals smaller than ~10 microsecond in the common range of years.
        ts1 = Timestamp("2012-1-1")
        ts2 = ts1 + offset
        val1 = dtc.convert(ts1, None, None)
        val2 = dtc.convert(ts2, None, None)
        if not val1 < val2:
            raise AssertionError(f"{val1} is not less than {val2}.")

    def test_convert_nested(self, dtc):
        inner = [Timestamp("2017-01-01"), Timestamp("2017-01-02")]
        data = [inner, inner]
        result = dtc.convert(data, None, None)
        expected = [dtc.convert(x, None, None) for x in data]
        assert (np.array(result) == expected).all()


class TestPeriodConverter:
    @pytest.fixture
    def pc(self):
        return converter.PeriodConverter()

    @pytest.fixture
    def axis(self):
        class Axis:
            pass

        axis = Axis()
        axis.freq = "D"
        return axis

    def test_convert_accepts_unicode(self, pc, axis):
        r1 = pc.convert("2012-1-1", None, axis)
        r2 = pc.convert("2012-1-1", None, axis)
        assert r1 == r2

    def test_conversion(self, pc, axis):
        rs = pc.convert(["2012-1-1"], None, axis)[0]
        xp = Period("2012-1-1").ordinal
        assert rs == xp

        rs = pc.convert("2012-1-1", None, axis)
        assert rs == xp

        rs = pc.convert([date(2012, 1, 1)], None, axis)[0]
        assert rs == xp

        rs = pc.convert(date(2012, 1, 1), None, axis)
        assert rs == xp

        rs = pc.convert([Timestamp("2012-1-1")], None, axis)[0]
        assert rs == xp

        rs = pc.convert(Timestamp("2012-1-1"), None, axis)
        assert rs == xp

        rs = pc.convert("2012-01-01", None, axis)
        assert rs == xp

        rs = pc.convert("2012-01-01 00:00:00+0000", None, axis)
        assert rs == xp

        rs = pc.convert(
            np.array(
                ["2012-01-01 00:00:00", "2012-01-02 00:00:00"],
                dtype="datetime64[ns]",
            ),
            None,
            axis,
        )
        assert rs[0] == xp

    def test_integer_passthrough(self, pc, axis):
        # GH9012
        rs = pc.convert([0, 1], None, axis)
        xp = [0, 1]
        assert rs == xp

    def test_convert_nested(self, pc, axis):
        data = ["2012-1-1", "2012-1-2"]
        r1 = pc.convert([data, data], None, axis)
        r2 = [pc.convert(data, None, axis) for _ in range(2)]
        assert r1 == r2


class TestTimeDeltaConverter:
    """Test timedelta converter"""

    @pytest.mark.parametrize(
        "x, decimal, format_expected",
        [
            (0.0, 0, "00:00:00"),
            (3972320000000, 1, "01:06:12.3"),
            (713233432000000, 2, "8 days 06:07:13.43"),
            (32423432000000, 4, "09:00:23.4320"),
        ],
    )
    def test_format_timedelta_ticks(self, x, decimal, format_expected):
        tdc = converter.TimeSeries_TimedeltaFormatter
        result = tdc.format_timedelta_ticks(x, pos=None, n_decimals=decimal)
        assert result == format_expected

    @pytest.mark.parametrize("view_interval", [(1, 2), (2, 1)])
    def test_call_w_different_view_intervals(self, view_interval, monkeypatch):
        # previously broke on reversed xlmits; see GH37454
        class mock_axis:
            def get_view_interval(self):
                return view_interval

        tdc = converter.TimeSeries_TimedeltaFormatter()
        monkeypatch.setattr(tdc, "axis", mock_axis())
        tdc(0.0, 0)


@pytest.mark.parametrize("year_span", [11.25, 30, 80, 150, 400, 800, 1500, 2500, 3500])
# The range is limited to 11.25 at the bottom by if statements in
# the _quarterly_finder() function
def test_quarterly_finder(year_span):
    vmin = -1000
    vmax = vmin + year_span * 4
    span = vmax - vmin + 1
    if span < 45:
        pytest.skip("the quarterly finder is only invoked if the span is >= 45")
    nyears = span / 4
    (min_anndef, maj_anndef) = converter._get_default_annual_spacing(nyears)
    result = converter._quarterly_finder(vmin, vmax, to_offset("QE"))
    quarters = PeriodIndex(
        arrays.PeriodArray(np.array([x[0] for x in result]), dtype="period[Q]")
    )
    majors = np.array([x[1] for x in result])
    minors = np.array([x[2] for x in result])
    major_quarters = quarters[majors]
    minor_quarters = quarters[minors]
    check_major_years = major_quarters.year % maj_anndef == 0
    check_minor_years = minor_quarters.year % min_anndef == 0
    check_major_quarters = major_quarters.quarter == 1
    check_minor_quarters = minor_quarters.quarter == 1
    assert np.all(check_major_years)
    assert np.all(check_minor_years)
    assert np.all(check_major_quarters)
    assert np.all(check_minor_quarters)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\concrete\tests\test_products.py ===
from sympy.concrete.products import (Product, product)
from sympy.concrete.summations import Sum
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.factorials import (rf, factorial)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.simplify.combsimp import combsimp
from sympy.simplify.simplify import simplify
from sympy.testing.pytest import raises

a, k, n, m, x = symbols('a,k,n,m,x', integer=True)
f = Function('f')


def test_karr_convention():
    # Test the Karr product convention that we want to hold.
    # See his paper "Summation in Finite Terms" for a detailed
    # reasoning why we really want exactly this definition.
    # The convention is described for sums on page 309 and
    # essentially in section 1.4, definition 3. For products
    # we can find in analogy:
    #
    # \prod_{m <= i < n} f(i) 'has the obvious meaning'      for m < n
    # \prod_{m <= i < n} f(i) = 0                            for m = n
    # \prod_{m <= i < n} f(i) = 1 / \prod_{n <= i < m} f(i)  for m > n
    #
    # It is important to note that he defines all products with
    # the upper limit being *exclusive*.
    # In contrast, SymPy and the usual mathematical notation has:
    #
    # prod_{i = a}^b f(i) = f(a) * f(a+1) * ... * f(b-1) * f(b)
    #
    # with the upper limit *inclusive*. So translating between
    # the two we find that:
    #
    # \prod_{m <= i < n} f(i) = \prod_{i = m}^{n-1} f(i)
    #
    # where we intentionally used two different ways to typeset the
    # products and its limits.

    i = Symbol("i", integer=True)
    k = Symbol("k", integer=True)
    j = Symbol("j", integer=True, positive=True)

    # A simple example with a concrete factors and symbolic limits.

    # The normal product: m = k and n = k + j and therefore m < n:
    m = k
    n = k + j

    a = m
    b = n - 1
    S1 = Product(i**2, (i, a, b)).doit()

    # The reversed product: m = k + j and n = k and therefore m > n:
    m = k + j
    n = k

    a = m
    b = n - 1
    S2 = Product(i**2, (i, a, b)).doit()

    assert S1 * S2 == 1

    # Test the empty product: m = k and n = k and therefore m = n:
    m = k
    n = k

    a = m
    b = n - 1
    Sz = Product(i**2, (i, a, b)).doit()

    assert Sz == 1

    # Another example this time with an unspecified factor and
    # numeric limits. (We can not do both tests in the same example.)
    f = Function("f")

    # The normal product with m < n:
    m = 2
    n = 11

    a = m
    b = n - 1
    S1 = Product(f(i), (i, a, b)).doit()

    # The reversed product with m > n:
    m = 11
    n = 2

    a = m
    b = n - 1
    S2 = Product(f(i), (i, a, b)).doit()

    assert simplify(S1 * S2) == 1

    # Test the empty product with m = n:
    m = 5
    n = 5

    a = m
    b = n - 1
    Sz = Product(f(i), (i, a, b)).doit()

    assert Sz == 1


def test_karr_proposition_2a():
    # Test Karr, page 309, proposition 2, part a
    i, u, v = symbols('i u v', integer=True)

    def test_the_product(m, n):
        # g
        g = i**3 + 2*i**2 - 3*i
        # f = Delta g
        f = simplify(g.subs(i, i+1) / g)
        # The product
        a = m
        b = n - 1
        P = Product(f, (i, a, b)).doit()
        # Test if Product_{m <= i < n} f(i) = g(n) / g(m)
        assert combsimp(P / (g.subs(i, n) / g.subs(i, m))) == 1

    # m < n
    test_the_product(u, u + v)
    # m = n
    test_the_product(u, u)
    # m > n
    test_the_product(u + v, u)


def test_karr_proposition_2b():
    # Test Karr, page 309, proposition 2, part b
    i, u, v, w = symbols('i u v w', integer=True)

    def test_the_product(l, n, m):
        # Productmand
        s = i**3
        # First product
        a = l
        b = n - 1
        S1 = Product(s, (i, a, b)).doit()
        # Second product
        a = l
        b = m - 1
        S2 = Product(s, (i, a, b)).doit()
        # Third product
        a = m
        b = n - 1
        S3 = Product(s, (i, a, b)).doit()
        # Test if S1 = S2 * S3 as required
        assert combsimp(S1 / (S2 * S3)) == 1

    # l < m < n
    test_the_product(u, u + v, u + v + w)
    # l < m = n
    test_the_product(u, u + v, u + v)
    # l < m > n
    test_the_product(u, u + v + w, v)
    # l = m < n
    test_the_product(u, u, u + v)
    # l = m = n
    test_the_product(u, u, u)
    # l = m > n
    test_the_product(u + v, u + v, u)
    # l > m < n
    test_the_product(u + v, u, u + w)
    # l > m = n
    test_the_product(u + v, u, u)
    # l > m > n
    test_the_product(u + v + w, u + v, u)


def test_simple_products():
    assert product(2, (k, a, n)) == 2**(n - a + 1)
    assert product(k, (k, 1, n)) == factorial(n)
    assert product(k**3, (k, 1, n)) == factorial(n)**3

    assert product(k + 1, (k, 0, n - 1)) == factorial(n)
    assert product(k + 1, (k, a, n - 1)) == rf(1 + a, n - a)

    assert product(cos(k), (k, 0, 5)) == cos(1)*cos(2)*cos(3)*cos(4)*cos(5)
    assert product(cos(k), (k, 3, 5)) == cos(3)*cos(4)*cos(5)
    assert product(cos(k), (k, 1, Rational(5, 2))) != cos(1)*cos(2)

    assert isinstance(product(k**k, (k, 1, n)), Product)

    assert Product(x**k, (k, 1, n)).variables == [k]

    raises(ValueError, lambda: Product(n))
    raises(ValueError, lambda: Product(n, k))
    raises(ValueError, lambda: Product(n, k, 1))
    raises(ValueError, lambda: Product(n, k, 1, 10))
    raises(ValueError, lambda: Product(n, (k, 1)))

    assert product(1, (n, 1, oo)) == 1  # issue 8301
    assert product(2, (n, 1, oo)) is oo
    assert product(-1, (n, 1, oo)).func is Product


def test_multiple_products():
    assert product(x, (n, 1, k), (k, 1, m)) == x**(m**2/2 + m/2)
    assert product(f(n), (
        n, 1, m), (m, 1, k)) == Product(f(n), (n, 1, m), (m, 1, k)).doit()
    assert Product(f(n), (m, 1, k), (n, 1, k)).doit() == \
        Product(Product(f(n), (m, 1, k)), (n, 1, k)).doit() == \
        product(f(n), (m, 1, k), (n, 1, k)) == \
        product(product(f(n), (m, 1, k)), (n, 1, k)) == \
        Product(f(n)**k, (n, 1, k))
    assert Product(
        x, (x, 1, k), (k, 1, n)).doit() == Product(factorial(k), (k, 1, n))

    assert Product(x**k, (n, 1, k), (k, 1, m)).variables == [n, k]


def test_rational_products():
    assert product(1 + 1/k, (k, 1, n)) == rf(2, n)/factorial(n)


def test_special_products():
    # Wallis product
    assert product((4*k)**2 / (4*k**2 - 1), (k, 1, n)) == \
        4**n*factorial(n)**2/rf(S.Half, n)/rf(Rational(3, 2), n)

    # Euler's product formula for sin
    assert product(1 + a/k**2, (k, 1, n)) == \
        rf(1 - sqrt(-a), n)*rf(1 + sqrt(-a), n)/factorial(n)**2


def test__eval_product():
    from sympy.abc import i, n
    # issue 4809
    a = Function('a')
    assert product(2*a(i), (i, 1, n)) == 2**n * Product(a(i), (i, 1, n))
    # issue 4810
    assert product(2**i, (i, 1, n)) == 2**(n*(n + 1)/2)
    k, m = symbols('k m', integer=True)
    assert product(2**i, (i, k, m)) == 2**(-k**2/2 + k/2 + m**2/2 + m/2)
    n = Symbol('n', negative=True, integer=True)
    p = Symbol('p', positive=True, integer=True)
    assert product(2**i, (i, n, p)) == 2**(-n**2/2 + n/2 + p**2/2 + p/2)
    assert product(2**i, (i, p, n)) == 2**(n**2/2 + n/2 - p**2/2 + p/2)


def test_product_pow():
    # issue 4817
    assert product(2**f(k), (k, 1, n)) == 2**Sum(f(k), (k, 1, n))
    assert product(2**(2*f(k)), (k, 1, n)) == 2**Sum(2*f(k), (k, 1, n))


def test_infinite_product():
    # issue 5737
    assert isinstance(Product(2**(1/factorial(n)), (n, 0, oo)), Product)


def test_conjugate_transpose():
    p = Product(x**k, (k, 1, 3))
    assert p.adjoint().doit() == p.doit().adjoint()
    assert p.conjugate().doit() == p.doit().conjugate()
    assert p.transpose().doit() == p.doit().transpose()

    A, B = symbols("A B", commutative=False)
    p = Product(A*B**k, (k, 1, 3))
    assert p.adjoint().doit() == p.doit().adjoint()
    assert p.conjugate().doit() == p.doit().conjugate()
    assert p.transpose().doit() == p.doit().transpose()

    p = Product(B**k*A, (k, 1, 3))
    assert p.adjoint().doit() == p.doit().adjoint()
    assert p.conjugate().doit() == p.doit().conjugate()
    assert p.transpose().doit() == p.doit().transpose()


def test_simplify_prod():
    y, t, b, c, v, d = symbols('y, t, b, c, v, d', integer = True)

    _simplify = lambda e: simplify(e, doit=False)
    assert _simplify(Product(x*y, (x, n, m), (y, a, k)) * \
        Product(y, (x, n, m), (y, a, k))) == \
            Product(x*y**2, (x, n, m), (y, a, k))
    assert _simplify(3 * y* Product(x, (x, n, m)) * Product(x, (x, m + 1, a))) \
        == 3 * y * Product(x, (x, n, a))
    assert _simplify(Product(x, (x, k + 1, a)) * Product(x, (x, n, k))) == \
        Product(x, (x, n, a))
    assert _simplify(Product(x, (x, k + 1, a)) * Product(x + 1, (x, n, k))) == \
        Product(x, (x, k + 1, a)) * Product(x + 1, (x, n, k))
    assert _simplify(Product(x, (t, a, b)) * Product(y, (t, a, b)) * \
        Product(x, (t, b+1, c))) == Product(x*y, (t, a, b)) * \
            Product(x, (t, b+1, c))
    assert _simplify(Product(x, (t, a, b)) * Product(x, (t, b+1, c)) * \
        Product(y, (t, a, b))) == Product(x*y, (t, a, b)) * \
            Product(x, (t, b+1, c))
    assert _simplify(Product(sin(t)**2 + cos(t)**2 + 1, (t, a, b))) == \
        Product(2, (t, a, b))
    assert _simplify(Product(sin(t)**2 + cos(t)**2 - 1, (t, a, b))) == \
           Product(0, (t, a, b))
    assert _simplify(Product(v*Product(sin(t)**2 + cos(t)**2, (t, a, b)),
                             (v, c, d))) == Product(v*Product(1, (t, a, b)), (v, c, d))


def test_change_index():
    b, y, c, d, z = symbols('b, y, c, d, z', integer = True)

    assert Product(x, (x, a, b)).change_index(x, x + 1, y) == \
        Product(y - 1, (y, a + 1, b + 1))
    assert Product(x**2, (x, a, b)).change_index(x, x - 1) == \
        Product((x + 1)**2, (x, a - 1, b - 1))
    assert Product(x**2, (x, a, b)).change_index(x, -x, y) == \
        Product((-y)**2, (y, -b, -a))
    assert Product(x, (x, a, b)).change_index(x, -x - 1) == \
        Product(-x - 1, (x, - b - 1, -a - 1))
    assert Product(x*y, (x, a, b), (y, c, d)).change_index(x, x - 1, z) == \
        Product((z + 1)*y, (z, a - 1, b - 1), (y, c, d))


def test_reorder():
    b, y, c, d, z = symbols('b, y, c, d, z', integer = True)

    assert Product(x*y, (x, a, b), (y, c, d)).reorder((0, 1)) == \
        Product(x*y, (y, c, d), (x, a, b))
    assert Product(x, (x, a, b), (x, c, d)).reorder((0, 1)) == \
        Product(x, (x, c, d), (x, a, b))
    assert Product(x*y + z, (x, a, b), (z, m, n), (y, c, d)).reorder(\
        (2, 0), (0, 1)) == Product(x*y + z, (z, m, n), (y, c, d), (x, a, b))
    assert Product(x*y*z, (x, a, b), (y, c, d), (z, m, n)).reorder(\
        (0, 1), (1, 2), (0, 2)) == \
        Product(x*y*z, (x, a, b), (z, m, n), (y, c, d))
    assert Product(x*y*z, (x, a, b), (y, c, d), (z, m, n)).reorder(\
        (x, y), (y, z), (x, z)) == \
        Product(x*y*z, (x, a, b), (z, m, n), (y, c, d))
    assert Product(x*y, (x, a, b), (y, c, d)).reorder((x, 1)) == \
        Product(x*y, (y, c, d), (x, a, b))
    assert Product(x*y, (x, a, b), (y, c, d)).reorder((y, x)) == \
        Product(x*y, (y, c, d), (x, a, b))


def test_Product_is_convergent():
    assert Product(1/n**2, (n, 1, oo)).is_convergent() is S.false
    assert Product(exp(1/n**2), (n, 1, oo)).is_convergent() is S.true
    assert Product(1/n, (n, 1, oo)).is_convergent() is S.false
    assert Product(1 + 1/n, (n, 1, oo)).is_convergent() is S.false
    assert Product(1 + 1/n**2, (n, 1, oo)).is_convergent() is S.true


def test_reverse_order():
    x, y, a, b, c, d= symbols('x, y, a, b, c, d', integer = True)

    assert Product(x, (x, 0, 3)).reverse_order(0) == Product(1/x, (x, 4, -1))
    assert Product(x*y, (x, 1, 5), (y, 0, 6)).reverse_order(0, 1) == \
           Product(x*y, (x, 6, 0), (y, 7, -1))
    assert Product(x, (x, 1, 2)).reverse_order(0) == Product(1/x, (x, 3, 0))
    assert Product(x, (x, 1, 3)).reverse_order(0) == Product(1/x, (x, 4, 0))
    assert Product(x, (x, 1, a)).reverse_order(0) == Product(1/x, (x, a + 1, 0))
    assert Product(x, (x, a, 5)).reverse_order(0) == Product(1/x, (x, 6, a - 1))
    assert Product(x, (x, a + 1, a + 5)).reverse_order(0) == \
           Product(1/x, (x, a + 6, a))
    assert Product(x, (x, a + 1, a + 2)).reverse_order(0) == \
           Product(1/x, (x, a + 3, a))
    assert Product(x, (x, a + 1, a + 1)).reverse_order(0) == \
           Product(1/x, (x, a + 2, a))
    assert Product(x, (x, a, b)).reverse_order(0) == Product(1/x, (x, b + 1, a - 1))
    assert Product(x, (x, a, b)).reverse_order(x) == Product(1/x, (x, b + 1, a - 1))
    assert Product(x*y, (x, a, b), (y, 2, 5)).reverse_order(x, 1) == \
           Product(x*y, (x, b + 1, a - 1), (y, 6, 1))
    assert Product(x*y, (x, a, b), (y, 2, 5)).reverse_order(y, x) == \
           Product(x*y, (x, b + 1, a - 1), (y, 6, 1))


def test_issue_9983():
    n = Symbol('n', integer=True, positive=True)
    p = Product(1 + 1/n**Rational(2, 3), (n, 1, oo))
    assert p.is_convergent() is S.false
    assert product(1 + 1/n**Rational(2, 3), (n, 1, oo)) == p.doit()


def test_issue_13546():
    n = Symbol('n')
    k = Symbol('k')
    p = Product(n + 1 / 2**k, (k, 0, n-1)).doit()
    assert p.subs(n, 2).doit() == Rational(15, 2)


def test_issue_14036():
    a, n = symbols('a n')
    assert product(1 - a**2 / (n*pi)**2, [n, 1, oo]) != 0


def test_rewrite_Sum():
    assert Product(1 - S.Half**2/k**2, (k, 1, oo)).rewrite(Sum) == \
        exp(Sum(log(1 - 1/(4*k**2)), (k, 1, oo)))


def test_KroneckerDelta_Product():
    y = Symbol('y')
    assert Product(x*KroneckerDelta(x, y), (x, 0, 1)).doit() == 0


def test_issue_20848():
    _i = Dummy('i')
    t, y, z = symbols('t y z')
    assert diff(Product(x, (y, 1, z)), x).as_dummy() == Sum(Product(x, (y, 1, _i - 1))*Product(x, (y, _i + 1, z)), (_i, 1, z)).as_dummy()
    assert diff(Product(x, (y, 1, z)), x).doit() == x**(z - 1)*z
    assert diff(Product(x, (y, x, z)), x) == Derivative(Product(x, (y, x, z)), x)
    assert diff(Product(t, (x, 1, z)), x) == S(0)
    assert Product(sin(n*x), (n, -1, 1)).diff(x).doit() == S(0)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\agca\tests\test_modules.py ===
"""Test modules.py code."""

from sympy.polys.agca.modules import FreeModule, ModuleOrder, FreeModulePolyRing
from sympy.polys import CoercionFailed, QQ, lex, grlex, ilex, ZZ
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
from sympy.core.numbers import Rational


def test_FreeModuleElement():
    M = QQ.old_poly_ring(x).free_module(3)
    e = M.convert([1, x, x**2])
    f = [QQ.old_poly_ring(x).convert(1), QQ.old_poly_ring(x).convert(x), QQ.old_poly_ring(x).convert(x**2)]
    assert list(e) == f
    assert f[0] == e[0]
    assert f[1] == e[1]
    assert f[2] == e[2]
    raises(IndexError, lambda: e[3])

    g = M.convert([x, 0, 0])
    assert e + g == M.convert([x + 1, x, x**2])
    assert f + g == M.convert([x + 1, x, x**2])
    assert -e == M.convert([-1, -x, -x**2])
    assert e - g == M.convert([1 - x, x, x**2])
    assert e != g

    assert M.convert([x, x, x]) / QQ.old_poly_ring(x).convert(x) == [1, 1, 1]
    R = QQ.old_poly_ring(x, order="ilex")
    assert R.free_module(1).convert([x]) / R.convert(x) == [1]


def test_FreeModule():
    M1 = FreeModule(QQ.old_poly_ring(x), 2)
    assert M1 == FreeModule(QQ.old_poly_ring(x), 2)
    assert M1 != FreeModule(QQ.old_poly_ring(y), 2)
    assert M1 != FreeModule(QQ.old_poly_ring(x), 3)
    M2 = FreeModule(QQ.old_poly_ring(x, order="ilex"), 2)

    assert [x, 1] in M1
    assert [x] not in M1
    assert [2, y] not in M1
    assert [1/(x + 1), 2] not in M1

    e = M1.convert([x, x**2 + 1])
    X = QQ.old_poly_ring(x).convert(x)
    assert e == [X, X**2 + 1]
    assert e == [x, x**2 + 1]
    assert 2*e == [2*x, 2*x**2 + 2]
    assert e*2 == [2*x, 2*x**2 + 2]
    assert e/2 == [x/2, (x**2 + 1)/2]
    assert x*e == [x**2, x**3 + x]
    assert e*x == [x**2, x**3 + x]
    assert X*e == [x**2, x**3 + x]
    assert e*X == [x**2, x**3 + x]

    assert [x, 1] in M2
    assert [x] not in M2
    assert [2, y] not in M2
    assert [1/(x + 1), 2] in M2

    e = M2.convert([x, x**2 + 1])
    X = QQ.old_poly_ring(x, order="ilex").convert(x)
    assert e == [X, X**2 + 1]
    assert e == [x, x**2 + 1]
    assert 2*e == [2*x, 2*x**2 + 2]
    assert e*2 == [2*x, 2*x**2 + 2]
    assert e/2 == [x/2, (x**2 + 1)/2]
    assert x*e == [x**2, x**3 + x]
    assert e*x == [x**2, x**3 + x]
    assert e/(1 + x) == [x/(1 + x), (x**2 + 1)/(1 + x)]
    assert X*e == [x**2, x**3 + x]
    assert e*X == [x**2, x**3 + x]

    M3 = FreeModule(QQ.old_poly_ring(x, y), 2)
    assert M3.convert(e) == M3.convert([x, x**2 + 1])

    assert not M3.is_submodule(0)
    assert not M3.is_zero()

    raises(NotImplementedError, lambda: ZZ.old_poly_ring(x).free_module(2))
    raises(NotImplementedError, lambda: FreeModulePolyRing(ZZ, 2))
    raises(CoercionFailed, lambda: M1.convert(QQ.old_poly_ring(x).free_module(3)
           .convert([1, 2, 3])))
    raises(CoercionFailed, lambda: M3.convert(1))


def test_ModuleOrder():
    o1 = ModuleOrder(lex, grlex, False)
    o2 = ModuleOrder(ilex, lex, False)

    assert o1 == ModuleOrder(lex, grlex, False)
    assert (o1 != ModuleOrder(lex, grlex, False)) is False
    assert o1 != o2

    assert o1((1, 2, 3)) == (1, (5, (2, 3)))
    assert o2((1, 2, 3)) == (-1, (2, 3))


def test_SubModulePolyRing_global():
    R = QQ.old_poly_ring(x, y)
    F = R.free_module(3)
    Fd = F.submodule([1, 0, 0], [1, 2, 0], [1, 2, 3])
    M = F.submodule([x**2 + y**2, 1, 0], [x, y, 1])

    assert F == Fd
    assert Fd == F
    assert F != M
    assert M != F
    assert Fd != M
    assert M != Fd
    assert Fd == F.submodule(*F.basis())

    assert Fd.is_full_module()
    assert not M.is_full_module()
    assert not Fd.is_zero()
    assert not M.is_zero()
    assert Fd.submodule().is_zero()

    assert M.contains([x**2 + y**2 + x, 1 + y, 1])
    assert not M.contains([x**2 + y**2 + x, 1 + y, 2])
    assert M.contains([y**2, 1 - x*y, -x])

    assert not F.submodule([1 + x, 0, 0]) == F.submodule([1, 0, 0])
    assert F.submodule([1, 0, 0], [0, 1, 0]).union(F.submodule([0, 0, 1])) == F
    assert not M.is_submodule(0)

    m = F.convert([x**2 + y**2, 1, 0])
    n = M.convert(m)
    assert m.module is F
    assert n.module is M

    raises(ValueError, lambda: M.submodule([1, 0, 0]))
    raises(TypeError, lambda: M.union(1))
    raises(ValueError, lambda: M.union(R.free_module(1).submodule([x])))

    assert F.submodule([x, x, x]) != F.submodule([x, x, x], order="ilex")


def test_SubModulePolyRing_local():
    R = QQ.old_poly_ring(x, y, order=ilex)
    F = R.free_module(3)
    Fd = F.submodule([1 + x, 0, 0], [1 + y, 2 + 2*y, 0], [1, 2, 3])
    M = F.submodule([x**2 + y**2, 1, 0], [x, y, 1])

    assert F == Fd
    assert Fd == F
    assert F != M
    assert M != F
    assert Fd != M
    assert M != Fd
    assert Fd == F.submodule(*F.basis())

    assert Fd.is_full_module()
    assert not M.is_full_module()
    assert not Fd.is_zero()
    assert not M.is_zero()
    assert Fd.submodule().is_zero()

    assert M.contains([x**2 + y**2 + x, 1 + y, 1])
    assert not M.contains([x**2 + y**2 + x, 1 + y, 2])
    assert M.contains([y**2, 1 - x*y, -x])

    assert F.submodule([1 + x, 0, 0]) == F.submodule([1, 0, 0])
    assert F.submodule(
        [1, 0, 0], [0, 1, 0]).union(F.submodule([0, 0, 1 + x*y])) == F

    raises(ValueError, lambda: M.submodule([1, 0, 0]))


def test_SubModulePolyRing_nontriv_global():
    R = QQ.old_poly_ring(x, y, z)
    F = R.free_module(1)

    def contains(I, f):
        return F.submodule(*[[g] for g in I]).contains([f])

    assert contains([x, y], x)
    assert contains([x, y], x + y)
    assert not contains([x, y], 1)
    assert not contains([x, y], z)
    assert contains([x**2 + y, x**2 + x], x - y)
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x**2)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x**3)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x**4)
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x*y**2)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x**4 + y**3 + 2*z*y*x)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x*y*z)
    assert contains([x, 1 + x + y, 5 - 7*y], 1)
    assert contains(
        [x**3 + y**3, y**3 + z**3, z**3 + x**3, x**2*y + x**2*z + y**2*z],
        x**3)
    assert not contains(
        [x**3 + y**3, y**3 + z**3, z**3 + x**3, x**2*y + x**2*z + y**2*z],
        x**2 + y**2)

    # compare local order
    assert not contains([x*(1 + x + y), y*(1 + z)], x)
    assert not contains([x*(1 + x + y), y*(1 + z)], x + y)


def test_SubModulePolyRing_nontriv_local():
    R = QQ.old_poly_ring(x, y, z, order=ilex)
    F = R.free_module(1)

    def contains(I, f):
        return F.submodule(*[[g] for g in I]).contains([f])

    assert contains([x, y], x)
    assert contains([x, y], x + y)
    assert not contains([x, y], 1)
    assert not contains([x, y], z)
    assert contains([x**2 + y, x**2 + x], x - y)
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x**2)
    assert contains([x*(1 + x + y), y*(1 + z)], x)
    assert contains([x*(1 + x + y), y*(1 + z)], x + y)


def test_syzygy():
    R = QQ.old_poly_ring(x, y, z)
    M = R.free_module(1).submodule([x*y], [y*z], [x*z])
    S = R.free_module(3).submodule([0, x, -y], [z, -x, 0])
    assert M.syzygy_module() == S

    M2 = M / ([x*y*z],)
    S2 = R.free_module(3).submodule([z, 0, 0], [0, x, 0], [0, 0, y])
    assert M2.syzygy_module() == S2

    F = R.free_module(3)
    assert F.submodule(*F.basis()).syzygy_module() == F.submodule()

    R2 = QQ.old_poly_ring(x, y, z) / [x*y*z]
    M3 = R2.free_module(1).submodule([x*y], [y*z], [x*z])
    S3 = R2.free_module(3).submodule([z, 0, 0], [0, x, 0], [0, 0, y])
    assert M3.syzygy_module() == S3


def test_in_terms_of_generators():
    R = QQ.old_poly_ring(x, order="ilex")
    M = R.free_module(2).submodule([2*x, 0], [1, 2])
    assert M.in_terms_of_generators(
        [x, x]) == [R.convert(Rational(1, 4)), R.convert(x/2)]
    raises(ValueError, lambda: M.in_terms_of_generators([1, 0]))

    M = R.free_module(2) / ([x, 0], [1, 1])
    SM = M.submodule([1, x])
    assert SM.in_terms_of_generators([2, 0]) == [R.convert(-2/(x - 1))]

    R = QQ.old_poly_ring(x, y) / [x**2 - y**2]
    M = R.free_module(2)
    SM = M.submodule([x, 0], [0, y])
    assert SM.in_terms_of_generators(
        [x**2, x**2]) == [R.convert(x), R.convert(y)]


def test_QuotientModuleElement():
    R = QQ.old_poly_ring(x)
    F = R.free_module(3)
    N = F.submodule([1, x, x**2])
    M = F/N
    e = M.convert([x**2, 2, 0])

    assert M.convert([x + 1, x**2 + x, x**3 + x**2]) == 0
    assert e == [x**2, 2, 0] + N == F.convert([x**2, 2, 0]) + N == \
        M.convert(F.convert([x**2, 2, 0]))

    assert M.convert([x**2 + 1, 2*x + 2, x**2]) == e + [0, x, 0] == \
        e + M.convert([0, x, 0]) == e + F.convert([0, x, 0])
    assert M.convert([x**2 + 1, 2, x**2]) == e - [0, x, 0] == \
        e - M.convert([0, x, 0]) == e - F.convert([0, x, 0])
    assert M.convert([0, 2, 0]) == M.convert([x**2, 4, 0]) - e == \
        [x**2, 4, 0] - e == F.convert([x**2, 4, 0]) - e
    assert M.convert([x**3 + x**2, 2*x + 2, 0]) == (1 + x)*e == \
        R.convert(1 + x)*e == e*(1 + x) == e*R.convert(1 + x)
    assert -e == [-x**2, -2, 0]

    f = [x, x, 0] + N
    assert M.convert([1, 1, 0]) == f / x == f / R.convert(x)

    M2 = F/[(2, 2*x, 2*x**2), (0, 0, 1)]
    G = R.free_module(2)
    M3 = G/[[1, x]]
    M4 = F.submodule([1, x, x**2], [1, 0, 0]) / N
    raises(CoercionFailed, lambda: M.convert(G.convert([1, x])))
    raises(CoercionFailed, lambda: M.convert(M3.convert([1, x])))
    raises(CoercionFailed, lambda: M.convert(M2.convert([1, x, x])))
    assert M2.convert(M.convert([2, x, x**2])) == [2, x, 0]
    assert M.convert(M4.convert([2, 0, 0])) == [2, 0, 0]


def test_QuotientModule():
    R = QQ.old_poly_ring(x)
    F = R.free_module(3)
    N = F.submodule([1, x, x**2])
    M = F/N

    assert M != F
    assert M != N
    assert M == F / [(1, x, x**2)]
    assert not M.is_zero()
    assert (F / F.basis()).is_zero()

    SQ = F.submodule([1, x, x**2], [2, 0, 0]) / N
    assert SQ == M.submodule([2, x, x**2])
    assert SQ != M.submodule([2, 1, 0])
    assert SQ != M
    assert M.is_submodule(SQ)
    assert not SQ.is_full_module()

    raises(ValueError, lambda: N/F)
    raises(ValueError, lambda: F.submodule([2, 0, 0]) / N)
    raises(ValueError, lambda: R.free_module(2)/F)
    raises(CoercionFailed, lambda: F.convert(M.convert([1, x, x**2])))

    M1 = F / [[1, 1, 1]]
    M2 = M1.submodule([1, 0, 0], [0, 1, 0])
    assert M1 == M2


def test_ModulesQuotientRing():
    R = QQ.old_poly_ring(x, y, order=(("lex", x), ("ilex", y))) / [x**2 + 1]
    M1 = R.free_module(2)
    assert M1 == R.free_module(2)
    assert M1 != QQ.old_poly_ring(x).free_module(2)
    assert M1 != R.free_module(3)

    assert [x, 1] in M1
    assert [x] not in M1
    assert [1/(R.convert(x) + 1), 2] in M1
    assert [1, 2/(1 + y)] in M1
    assert [1, 2/y] not in M1

    assert M1.convert([x**2, y]) == [-1, y]

    F = R.free_module(3)
    Fd = F.submodule([x**2, 0, 0], [1, 2, 0], [1, 2, 3])
    M = F.submodule([x**2 + y**2, 1, 0], [x, y, 1])

    assert F == Fd
    assert Fd == F
    assert F != M
    assert M != F
    assert Fd != M
    assert M != Fd
    assert Fd == F.submodule(*F.basis())

    assert Fd.is_full_module()
    assert not M.is_full_module()
    assert not Fd.is_zero()
    assert not M.is_zero()
    assert Fd.submodule().is_zero()

    assert M.contains([x**2 + y**2 + x, -x**2 + y, 1])
    assert not M.contains([x**2 + y**2 + x, 1 + y, 2])
    assert M.contains([y**2, 1 - x*y, -x])

    assert F.submodule([x, 0, 0]) == F.submodule([1, 0, 0])
    assert not F.submodule([y, 0, 0]) == F.submodule([1, 0, 0])
    assert F.submodule([1, 0, 0], [0, 1, 0]).union(F.submodule([0, 0, 1])) == F
    assert not M.is_submodule(0)


def test_module_mul():
    R = QQ.old_poly_ring(x)
    M = R.free_module(2)
    S1 = M.submodule([x, 0], [0, x])
    S2 = M.submodule([x**2, 0], [0, x**2])
    I = R.ideal(x)

    assert I*M == M*I == S1 == x*M == M*x
    assert I*S1 == S2 == x*S1


def test_intersection():
    # SCA, example 2.8.5
    F = QQ.old_poly_ring(x, y).free_module(2)
    M1 = F.submodule([x, y], [y, 1])
    M2 = F.submodule([0, y - 1], [x, 1], [y, x])
    I = F.submodule([x, y], [y**2 - y, y - 1], [x*y + y, x + 1])
    I1, rel1, rel2 = M1.intersect(M2, relations=True)
    assert I1 == M2.intersect(M1) == I
    for i, g in enumerate(I1.gens):
        assert g == sum(c*x for c, x in zip(rel1[i], M1.gens)) \
                 == sum(d*y for d, y in zip(rel2[i], M2.gens))

    assert F.submodule([x, y]).intersect(F.submodule([y, x])).is_zero()


def test_quotient():
    # SCA, example 2.8.6
    R = QQ.old_poly_ring(x, y, z)
    F = R.free_module(2)
    assert F.submodule([x*y, x*z], [y*z, x*y]).module_quotient(
        F.submodule([y, z], [z, y])) == QQ.old_poly_ring(x, y, z).ideal(x**2*y**2 - x*y*z**2)
    assert F.submodule([x, y]).module_quotient(F.submodule()).is_whole_ring()

    M = F.submodule([x**2, x**2], [y**2, y**2])
    N = F.submodule([x + y, x + y])
    q, rel = M.module_quotient(N, relations=True)
    assert q == R.ideal(y**2, x - y)
    for i, g in enumerate(q.gens):
        assert g*N.gens[0] == sum(c*x for c, x in zip(rel[i], M.gens))


def test_groebner_extendend():
    M = QQ.old_poly_ring(x, y, z).free_module(3).submodule([x + 1, y, 1], [x*y, z, z**2])
    G, R = M._groebner_vec(extended=True)
    for i, g in enumerate(G):
        assert g == sum(c*gen for c, gen in zip(R[i], M.gens))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\tests\test_fortran_parser.py ===
from sympy.testing.pytest import raises
from sympy.parsing.sym_expr import SymPyExpression
from sympy.external import import_module

lfortran = import_module('lfortran')

if lfortran:
    from sympy.codegen.ast import (Variable, IntBaseType, FloatBaseType, String,
                                   Return, FunctionDefinition, Assignment,
                                   Declaration, CodeBlock)
    from sympy.core import Integer, Float, Add
    from sympy.core.symbol import Symbol


    expr1 = SymPyExpression()
    expr2 = SymPyExpression()
    src = """\
    integer :: a, b, c, d
    real :: p, q, r, s
    """


    def test_sym_expr():
        src1 = (
            src +
            """\
            d = a + b -c
            """
        )
        expr3 = SymPyExpression(src,'f')
        expr4 = SymPyExpression(src1,'f')
        ls1 = expr3.return_expr()
        ls2 = expr4.return_expr()
        for i in range(0, 7):
            assert isinstance(ls1[i], Declaration)
            assert isinstance(ls2[i], Declaration)
        assert isinstance(ls2[8], Assignment)
        assert ls1[0] == Declaration(
            Variable(
                Symbol('a'),
                type = IntBaseType(String('integer')),
                value = Integer(0)
            )
        )
        assert ls1[1] == Declaration(
            Variable(
                Symbol('b'),
                type = IntBaseType(String('integer')),
                value = Integer(0)
            )
        )
        assert ls1[2] == Declaration(
            Variable(
                Symbol('c'),
                type = IntBaseType(String('integer')),
                value = Integer(0)
            )
        )
        assert ls1[3] == Declaration(
            Variable(
                Symbol('d'),
                type = IntBaseType(String('integer')),
                value = Integer(0)
            )
        )
        assert ls1[4] == Declaration(
            Variable(
                Symbol('p'),
                type = FloatBaseType(String('real')),
                value = Float(0.0)
            )
        )
        assert ls1[5] == Declaration(
            Variable(
                Symbol('q'),
                type = FloatBaseType(String('real')),
                value = Float(0.0)
            )
        )
        assert ls1[6] == Declaration(
            Variable(
                Symbol('r'),
                type = FloatBaseType(String('real')),
                value = Float(0.0)
            )
        )
        assert ls1[7] == Declaration(
            Variable(
                Symbol('s'),
                type = FloatBaseType(String('real')),
                value = Float(0.0)
            )
        )
        assert ls2[8] == Assignment(
            Variable(Symbol('d')),
            Symbol('a') + Symbol('b') - Symbol('c')
        )

    def test_assignment():
        src1 = (
            src +
            """\
            a = b
            c = d
            p = q
            r = s
            """
        )
        expr1.convert_to_expr(src1, 'f')
        ls1 = expr1.return_expr()
        for iter in range(0, 12):
            if iter < 8:
                assert isinstance(ls1[iter], Declaration)
            else:
                assert isinstance(ls1[iter], Assignment)
        assert ls1[8] == Assignment(
            Variable(Symbol('a')),
            Variable(Symbol('b'))
        )
        assert ls1[9] == Assignment(
            Variable(Symbol('c')),
            Variable(Symbol('d'))
        )
        assert ls1[10] == Assignment(
            Variable(Symbol('p')),
            Variable(Symbol('q'))
        )
        assert ls1[11] == Assignment(
            Variable(Symbol('r')),
            Variable(Symbol('s'))
        )


    def test_binop_add():
        src1 = (
            src +
            """\
            c = a + b
            d = a + c
            s = p + q + r
            """
        )
        expr1.convert_to_expr(src1, 'f')
        ls1 = expr1.return_expr()
        for iter in range(8, 11):
            assert isinstance(ls1[iter], Assignment)
        assert ls1[8] == Assignment(
            Variable(Symbol('c')),
            Symbol('a') + Symbol('b')
        )
        assert ls1[9] == Assignment(
            Variable(Symbol('d')),
            Symbol('a') + Symbol('c')
        )
        assert ls1[10] == Assignment(
            Variable(Symbol('s')),
            Symbol('p') + Symbol('q') + Symbol('r')
        )


    def test_binop_sub():
        src1 = (
            src +
            """\
            c = a - b
            d = a - c
            s = p - q - r
            """
        )
        expr1.convert_to_expr(src1, 'f')
        ls1 = expr1.return_expr()
        for iter in range(8, 11):
            assert isinstance(ls1[iter], Assignment)
        assert ls1[8] == Assignment(
            Variable(Symbol('c')),
            Symbol('a') - Symbol('b')
        )
        assert ls1[9] == Assignment(
            Variable(Symbol('d')),
            Symbol('a') - Symbol('c')
        )
        assert ls1[10] == Assignment(
            Variable(Symbol('s')),
            Symbol('p') - Symbol('q') - Symbol('r')
        )


    def test_binop_mul():
        src1 = (
            src +
            """\
            c = a * b
            d = a * c
            s = p * q * r
            """
        )
        expr1.convert_to_expr(src1, 'f')
        ls1 = expr1.return_expr()
        for iter in range(8, 11):
            assert isinstance(ls1[iter], Assignment)
        assert ls1[8] == Assignment(
            Variable(Symbol('c')),
            Symbol('a') * Symbol('b')
        )
        assert ls1[9] == Assignment(
            Variable(Symbol('d')),
            Symbol('a') * Symbol('c')
        )
        assert ls1[10] == Assignment(
            Variable(Symbol('s')),
            Symbol('p') * Symbol('q') * Symbol('r')
        )


    def test_binop_div():
        src1 = (
            src +
            """\
            c = a / b
            d = a / c
            s = p / q
            r = q / p
            """
        )
        expr1.convert_to_expr(src1, 'f')
        ls1 = expr1.return_expr()
        for iter in range(8, 12):
            assert isinstance(ls1[iter], Assignment)
        assert ls1[8] == Assignment(
            Variable(Symbol('c')),
            Symbol('a') / Symbol('b')
        )
        assert ls1[9] == Assignment(
            Variable(Symbol('d')),
            Symbol('a') / Symbol('c')
        )
        assert ls1[10] == Assignment(
            Variable(Symbol('s')),
            Symbol('p') / Symbol('q')
        )
        assert ls1[11] == Assignment(
            Variable(Symbol('r')),
            Symbol('q') / Symbol('p')
        )

    def test_mul_binop():
        src1 = (
            src +
            """\
            d = a + b - c
            c = a * b + d
            s = p * q / r
            r = p * s + q / p
            """
        )
        expr1.convert_to_expr(src1, 'f')
        ls1 = expr1.return_expr()
        for iter in range(8, 12):
            assert isinstance(ls1[iter], Assignment)
        assert ls1[8] == Assignment(
            Variable(Symbol('d')),
            Symbol('a') + Symbol('b') - Symbol('c')
        )
        assert ls1[9] == Assignment(
            Variable(Symbol('c')),
            Symbol('a') * Symbol('b') + Symbol('d')
        )
        assert ls1[10] == Assignment(
            Variable(Symbol('s')),
            Symbol('p') * Symbol('q') / Symbol('r')
        )
        assert ls1[11] == Assignment(
            Variable(Symbol('r')),
            Symbol('p') * Symbol('s') + Symbol('q') / Symbol('p')
        )


    def test_function():
        src1 = """\
        integer function f(a,b)
        integer :: x, y
        f = x + y
        end function
        """
        expr1.convert_to_expr(src1, 'f')
        for iter in expr1.return_expr():
            assert isinstance(iter, FunctionDefinition)
            assert iter == FunctionDefinition(
                IntBaseType(String('integer')),
                name=String('f'),
                parameters=(
                    Variable(Symbol('a')),
                    Variable(Symbol('b'))
                ),
                body=CodeBlock(
                    Declaration(
                        Variable(
                            Symbol('a'),
                            type=IntBaseType(String('integer')),
                            value=Integer(0)
                        )
                    ),
                    Declaration(
                        Variable(
                            Symbol('b'),
                            type=IntBaseType(String('integer')),
                            value=Integer(0)
                        )
                    ),
                    Declaration(
                        Variable(
                            Symbol('f'),
                            type=IntBaseType(String('integer')),
                            value=Integer(0)
                        )
                    ),
                    Declaration(
                        Variable(
                            Symbol('x'),
                            type=IntBaseType(String('integer')),
                            value=Integer(0)
                        )
                    ),
                    Declaration(
                        Variable(
                            Symbol('y'),
                            type=IntBaseType(String('integer')),
                            value=Integer(0)
                        )
                    ),
                    Assignment(
                        Variable(Symbol('f')),
                        Add(Symbol('x'), Symbol('y'))
                    ),
                    Return(Variable(Symbol('f')))
                )
            )


    def test_var():
        expr1.convert_to_expr(src, 'f')
        ls = expr1.return_expr()
        for iter in expr1.return_expr():
            assert isinstance(iter, Declaration)
        assert ls[0] == Declaration(
            Variable(
                Symbol('a'),
                type = IntBaseType(String('integer')),
                value = Integer(0)
            )
        )
        assert ls[1] == Declaration(
            Variable(
                Symbol('b'),
                type = IntBaseType(String('integer')),
                value = Integer(0)
            )
        )
        assert ls[2] == Declaration(
            Variable(
                Symbol('c'),
                type = IntBaseType(String('integer')),
                value = Integer(0)
            )
        )
        assert ls[3] == Declaration(
            Variable(
                Symbol('d'),
                type = IntBaseType(String('integer')),
                value = Integer(0)
            )
        )
        assert ls[4] == Declaration(
            Variable(
                Symbol('p'),
                type = FloatBaseType(String('real')),
                value = Float(0.0)
            )
        )
        assert ls[5] == Declaration(
            Variable(
                Symbol('q'),
                type = FloatBaseType(String('real')),
                value = Float(0.0)
            )
        )
        assert ls[6] == Declaration(
            Variable(
                Symbol('r'),
                type = FloatBaseType(String('real')),
                value = Float(0.0)
            )
        )
        assert ls[7] == Declaration(
            Variable(
                Symbol('s'),
                type = FloatBaseType(String('real')),
                value = Float(0.0)
            )
        )

else:
    def test_raise():
        from sympy.parsing.fortran.fortran_parser import ASR2PyVisitor
        raises(ImportError, lambda: ASR2PyVisitor())
        raises(ImportError, lambda: SymPyExpression(' ', mode = 'f'))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\copy_view\test_functions.py ===
import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas.compat import HAS_PYARROW

from pandas import (
    DataFrame,
    Index,
    Series,
    concat,
    merge,
)
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array


def test_concat_frames(using_copy_on_write):
    df = DataFrame({"b": ["a"] * 3}, dtype=object)
    df2 = DataFrame({"a": ["a"] * 3}, dtype=object)
    df_orig = df.copy()
    result = concat([df, df2], axis=1)

    if using_copy_on_write:
        assert np.shares_memory(get_array(result, "b"), get_array(df, "b"))
        assert np.shares_memory(get_array(result, "a"), get_array(df2, "a"))
    else:
        assert not np.shares_memory(get_array(result, "b"), get_array(df, "b"))
        assert not np.shares_memory(get_array(result, "a"), get_array(df2, "a"))

    result.iloc[0, 0] = "d"
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, "b"), get_array(df, "b"))
        assert np.shares_memory(get_array(result, "a"), get_array(df2, "a"))

    result.iloc[0, 1] = "d"
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, "a"), get_array(df2, "a"))
    tm.assert_frame_equal(df, df_orig)


def test_concat_frames_updating_input(using_copy_on_write):
    df = DataFrame({"b": ["a"] * 3}, dtype=object)
    df2 = DataFrame({"a": ["a"] * 3}, dtype=object)
    result = concat([df, df2], axis=1)

    if using_copy_on_write:
        assert np.shares_memory(get_array(result, "b"), get_array(df, "b"))
        assert np.shares_memory(get_array(result, "a"), get_array(df2, "a"))
    else:
        assert not np.shares_memory(get_array(result, "b"), get_array(df, "b"))
        assert not np.shares_memory(get_array(result, "a"), get_array(df2, "a"))

    expected = result.copy()
    df.iloc[0, 0] = "d"
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, "b"), get_array(df, "b"))
        assert np.shares_memory(get_array(result, "a"), get_array(df2, "a"))

    df2.iloc[0, 0] = "d"
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, "a"), get_array(df2, "a"))
    tm.assert_frame_equal(result, expected)


def test_concat_series(using_copy_on_write):
    ser = Series([1, 2], name="a")
    ser2 = Series([3, 4], name="b")
    ser_orig = ser.copy()
    ser2_orig = ser2.copy()
    result = concat([ser, ser2], axis=1)

    if using_copy_on_write:
        assert np.shares_memory(get_array(result, "a"), ser.values)
        assert np.shares_memory(get_array(result, "b"), ser2.values)
    else:
        assert not np.shares_memory(get_array(result, "a"), ser.values)
        assert not np.shares_memory(get_array(result, "b"), ser2.values)

    result.iloc[0, 0] = 100
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, "a"), ser.values)
        assert np.shares_memory(get_array(result, "b"), ser2.values)

    result.iloc[0, 1] = 1000
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, "b"), ser2.values)
    tm.assert_series_equal(ser, ser_orig)
    tm.assert_series_equal(ser2, ser2_orig)


def test_concat_frames_chained(using_copy_on_write):
    df1 = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    df2 = DataFrame({"c": [4, 5, 6]})
    df3 = DataFrame({"d": [4, 5, 6]})
    result = concat([concat([df1, df2], axis=1), df3], axis=1)
    expected = result.copy()

    if using_copy_on_write:
        assert np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
        assert np.shares_memory(get_array(result, "c"), get_array(df2, "c"))
        assert np.shares_memory(get_array(result, "d"), get_array(df3, "d"))
    else:
        assert not np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
        assert not np.shares_memory(get_array(result, "c"), get_array(df2, "c"))
        assert not np.shares_memory(get_array(result, "d"), get_array(df3, "d"))

    df1.iloc[0, 0] = 100
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, "a"), get_array(df1, "a"))

    tm.assert_frame_equal(result, expected)


def test_concat_series_chained(using_copy_on_write):
    ser1 = Series([1, 2, 3], name="a")
    ser2 = Series([4, 5, 6], name="c")
    ser3 = Series([4, 5, 6], name="d")
    result = concat([concat([ser1, ser2], axis=1), ser3], axis=1)
    expected = result.copy()

    if using_copy_on_write:
        assert np.shares_memory(get_array(result, "a"), get_array(ser1, "a"))
        assert np.shares_memory(get_array(result, "c"), get_array(ser2, "c"))
        assert np.shares_memory(get_array(result, "d"), get_array(ser3, "d"))
    else:
        assert not np.shares_memory(get_array(result, "a"), get_array(ser1, "a"))
        assert not np.shares_memory(get_array(result, "c"), get_array(ser2, "c"))
        assert not np.shares_memory(get_array(result, "d"), get_array(ser3, "d"))

    ser1.iloc[0] = 100
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, "a"), get_array(ser1, "a"))

    tm.assert_frame_equal(result, expected)


def test_concat_series_updating_input(using_copy_on_write):
    ser = Series([1, 2], name="a")
    ser2 = Series([3, 4], name="b")
    expected = DataFrame({"a": [1, 2], "b": [3, 4]})
    result = concat([ser, ser2], axis=1)

    if using_copy_on_write:
        assert np.shares_memory(get_array(result, "a"), get_array(ser, "a"))
        assert np.shares_memory(get_array(result, "b"), get_array(ser2, "b"))
    else:
        assert not np.shares_memory(get_array(result, "a"), get_array(ser, "a"))
        assert not np.shares_memory(get_array(result, "b"), get_array(ser2, "b"))

    ser.iloc[0] = 100
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, "a"), get_array(ser, "a"))
        assert np.shares_memory(get_array(result, "b"), get_array(ser2, "b"))
    tm.assert_frame_equal(result, expected)

    ser2.iloc[0] = 1000
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, "b"), get_array(ser2, "b"))
    tm.assert_frame_equal(result, expected)


def test_concat_mixed_series_frame(using_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "c": 1})
    ser = Series([4, 5, 6], name="d")
    result = concat([df, ser], axis=1)
    expected = result.copy()

    if using_copy_on_write:
        assert np.shares_memory(get_array(result, "a"), get_array(df, "a"))
        assert np.shares_memory(get_array(result, "c"), get_array(df, "c"))
        assert np.shares_memory(get_array(result, "d"), get_array(ser, "d"))
    else:
        assert not np.shares_memory(get_array(result, "a"), get_array(df, "a"))
        assert not np.shares_memory(get_array(result, "c"), get_array(df, "c"))
        assert not np.shares_memory(get_array(result, "d"), get_array(ser, "d"))

    ser.iloc[0] = 100
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, "d"), get_array(ser, "d"))

    df.iloc[0, 0] = 100
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, "a"), get_array(df, "a"))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("copy", [True, None, False])
def test_concat_copy_keyword(using_copy_on_write, copy):
    df = DataFrame({"a": [1, 2]})
    df2 = DataFrame({"b": [1.5, 2.5]})

    result = concat([df, df2], axis=1, copy=copy)

    if using_copy_on_write or copy is False:
        assert np.shares_memory(get_array(df, "a"), get_array(result, "a"))
        assert np.shares_memory(get_array(df2, "b"), get_array(result, "b"))
    else:
        assert not np.shares_memory(get_array(df, "a"), get_array(result, "a"))
        assert not np.shares_memory(get_array(df2, "b"), get_array(result, "b"))


@pytest.mark.parametrize(
    "func",
    [
        lambda df1, df2, **kwargs: df1.merge(df2, **kwargs),
        lambda df1, df2, **kwargs: merge(df1, df2, **kwargs),
    ],
)
def test_merge_on_key(using_copy_on_write, func):
    df1 = DataFrame({"key": Series(["a", "b", "c"], dtype=object), "a": [1, 2, 3]})
    df2 = DataFrame({"key": Series(["a", "b", "c"], dtype=object), "b": [4, 5, 6]})
    df1_orig = df1.copy()
    df2_orig = df2.copy()

    result = func(df1, df2, on="key")

    if using_copy_on_write:
        assert np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
        assert np.shares_memory(get_array(result, "b"), get_array(df2, "b"))
        assert np.shares_memory(get_array(result, "key"), get_array(df1, "key"))
        assert not np.shares_memory(get_array(result, "key"), get_array(df2, "key"))
    else:
        assert not np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
        assert not np.shares_memory(get_array(result, "b"), get_array(df2, "b"))

    result.iloc[0, 1] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
        assert np.shares_memory(get_array(result, "b"), get_array(df2, "b"))

    result.iloc[0, 2] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, "b"), get_array(df2, "b"))
    tm.assert_frame_equal(df1, df1_orig)
    tm.assert_frame_equal(df2, df2_orig)


def test_merge_on_index(using_copy_on_write):
    df1 = DataFrame({"a": [1, 2, 3]})
    df2 = DataFrame({"b": [4, 5, 6]})
    df1_orig = df1.copy()
    df2_orig = df2.copy()

    result = merge(df1, df2, left_index=True, right_index=True)

    if using_copy_on_write:
        assert np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
        assert np.shares_memory(get_array(result, "b"), get_array(df2, "b"))
    else:
        assert not np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
        assert not np.shares_memory(get_array(result, "b"), get_array(df2, "b"))

    result.iloc[0, 0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
        assert np.shares_memory(get_array(result, "b"), get_array(df2, "b"))

    result.iloc[0, 1] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, "b"), get_array(df2, "b"))
    tm.assert_frame_equal(df1, df1_orig)
    tm.assert_frame_equal(df2, df2_orig)


@pytest.mark.parametrize(
    "func, how",
    [
        (lambda df1, df2, **kwargs: merge(df2, df1, on="key", **kwargs), "right"),
        (lambda df1, df2, **kwargs: merge(df1, df2, on="key", **kwargs), "left"),
    ],
)
def test_merge_on_key_enlarging_one(using_copy_on_write, func, how):
    df1 = DataFrame({"key": Series(["a", "b", "c"], dtype=object), "a": [1, 2, 3]})
    df2 = DataFrame({"key": Series(["a", "b"], dtype=object), "b": [4, 5]})
    df1_orig = df1.copy()
    df2_orig = df2.copy()

    result = func(df1, df2, how=how)

    if using_copy_on_write:
        assert np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
        assert not np.shares_memory(get_array(result, "b"), get_array(df2, "b"))
        assert df2._mgr._has_no_reference(1)
        assert df2._mgr._has_no_reference(0)
        assert np.shares_memory(get_array(result, "key"), get_array(df1, "key")) is (
            how == "left"
        )
        assert not np.shares_memory(get_array(result, "key"), get_array(df2, "key"))
    else:
        assert not np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
        assert not np.shares_memory(get_array(result, "b"), get_array(df2, "b"))

    if how == "left":
        result.iloc[0, 1] = 0
    else:
        result.iloc[0, 2] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
    tm.assert_frame_equal(df1, df1_orig)
    tm.assert_frame_equal(df2, df2_orig)


@pytest.mark.parametrize("copy", [True, None, False])
def test_merge_copy_keyword(using_copy_on_write, copy):
    df = DataFrame({"a": [1, 2]})
    df2 = DataFrame({"b": [3, 4.5]})

    result = df.merge(df2, copy=copy, left_index=True, right_index=True)

    if using_copy_on_write or copy is False:
        assert np.shares_memory(get_array(df, "a"), get_array(result, "a"))
        assert np.shares_memory(get_array(df2, "b"), get_array(result, "b"))
    else:
        assert not np.shares_memory(get_array(df, "a"), get_array(result, "a"))
        assert not np.shares_memory(get_array(df2, "b"), get_array(result, "b"))


@pytest.mark.xfail(
    using_string_dtype() and HAS_PYARROW,
    reason="TODO(infer_string); result.index infers str dtype while both "
    "df1 and df2 index are object.",
)
def test_join_on_key(using_copy_on_write):
    df_index = Index(["a", "b", "c"], name="key", dtype=object)

    df1 = DataFrame({"a": [1, 2, 3]}, index=df_index.copy(deep=True))
    df2 = DataFrame({"b": [4, 5, 6]}, index=df_index.copy(deep=True))

    df1_orig = df1.copy()
    df2_orig = df2.copy()

    result = df1.join(df2, on="key")

    if using_copy_on_write:
        assert np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
        assert np.shares_memory(get_array(result, "b"), get_array(df2, "b"))
        assert np.shares_memory(get_array(result.index), get_array(df1.index))
        assert not np.shares_memory(get_array(result.index), get_array(df2.index))
    else:
        assert not np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
        assert not np.shares_memory(get_array(result, "b"), get_array(df2, "b"))

    result.iloc[0, 0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
        assert np.shares_memory(get_array(result, "b"), get_array(df2, "b"))

    result.iloc[0, 1] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, "b"), get_array(df2, "b"))

    tm.assert_frame_equal(df1, df1_orig)
    tm.assert_frame_equal(df2, df2_orig)


def test_join_multiple_dataframes_on_key(using_copy_on_write):
    df_index = Index(["a", "b", "c"], name="key", dtype=object)

    df1 = DataFrame({"a": [1, 2, 3]}, index=df_index.copy(deep=True))
    dfs_list = [
        DataFrame({"b": [4, 5, 6]}, index=df_index.copy(deep=True)),
        DataFrame({"c": [7, 8, 9]}, index=df_index.copy(deep=True)),
    ]

    df1_orig = df1.copy()
    dfs_list_orig = [df.copy() for df in dfs_list]

    result = df1.join(dfs_list)

    if using_copy_on_write:
        assert np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
        assert np.shares_memory(get_array(result, "b"), get_array(dfs_list[0], "b"))
        assert np.shares_memory(get_array(result, "c"), get_array(dfs_list[1], "c"))
        assert np.shares_memory(get_array(result.index), get_array(df1.index))
        assert not np.shares_memory(
            get_array(result.index), get_array(dfs_list[0].index)
        )
        assert not np.shares_memory(
            get_array(result.index), get_array(dfs_list[1].index)
        )
    else:
        assert not np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
        assert not np.shares_memory(get_array(result, "b"), get_array(dfs_list[0], "b"))
        assert not np.shares_memory(get_array(result, "c"), get_array(dfs_list[1], "c"))

    result.iloc[0, 0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
        assert np.shares_memory(get_array(result, "b"), get_array(dfs_list[0], "b"))
        assert np.shares_memory(get_array(result, "c"), get_array(dfs_list[1], "c"))

    result.iloc[0, 1] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, "b"), get_array(dfs_list[0], "b"))
        assert np.shares_memory(get_array(result, "c"), get_array(dfs_list[1], "c"))

    result.iloc[0, 2] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, "c"), get_array(dfs_list[1], "c"))

    tm.assert_frame_equal(df1, df1_orig)
    for df, df_orig in zip(dfs_list, dfs_list_orig):
        tm.assert_frame_equal(df, df_orig)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\offsets\test_ticks.py ===
"""
Tests for offsets.Tick and subclasses
"""
from datetime import (
    datetime,
    timedelta,
)

from hypothesis import (
    assume,
    example,
    given,
)
import numpy as np
import pytest

from pandas._libs.tslibs.offsets import delta_to_tick
from pandas.errors import OutOfBoundsTimedelta

from pandas import (
    Timedelta,
    Timestamp,
)
import pandas._testing as tm
from pandas._testing._hypothesis import INT_NEG_999_TO_POS_999
from pandas.tests.tseries.offsets.common import assert_offset_equal

from pandas.tseries import offsets
from pandas.tseries.offsets import (
    Hour,
    Micro,
    Milli,
    Minute,
    Nano,
    Second,
)

# ---------------------------------------------------------------------
# Test Helpers

tick_classes = [Hour, Minute, Second, Milli, Micro, Nano]


# ---------------------------------------------------------------------


def test_apply_ticks():
    result = offsets.Hour(3) + offsets.Hour(4)
    exp = offsets.Hour(7)
    assert result == exp


def test_delta_to_tick():
    delta = timedelta(3)

    tick = delta_to_tick(delta)
    assert tick == offsets.Day(3)

    td = Timedelta(nanoseconds=5)
    tick = delta_to_tick(td)
    assert tick == Nano(5)


@pytest.mark.parametrize("cls", tick_classes)
@example(n=2, m=3)
@example(n=800, m=300)
@example(n=1000, m=5)
@given(n=INT_NEG_999_TO_POS_999, m=INT_NEG_999_TO_POS_999)
def test_tick_add_sub(cls, n, m):
    # For all Tick subclasses and all integers n, m, we should have
    # tick(n) + tick(m) == tick(n+m)
    # tick(n) - tick(m) == tick(n-m)
    left = cls(n)
    right = cls(m)
    expected = cls(n + m)

    assert left + right == expected

    expected = cls(n - m)
    assert left - right == expected


@pytest.mark.arm_slow
@pytest.mark.parametrize("cls", tick_classes)
@example(n=2, m=3)
@given(n=INT_NEG_999_TO_POS_999, m=INT_NEG_999_TO_POS_999)
def test_tick_equality(cls, n, m):
    assume(m != n)
    # tick == tock iff tick.n == tock.n
    left = cls(n)
    right = cls(m)
    assert left != right

    right = cls(n)
    assert left == right
    assert not left != right

    if n != 0:
        assert cls(n) != cls(-n)


# ---------------------------------------------------------------------


def test_Hour():
    assert_offset_equal(Hour(), datetime(2010, 1, 1), datetime(2010, 1, 1, 1))
    assert_offset_equal(Hour(-1), datetime(2010, 1, 1, 1), datetime(2010, 1, 1))
    assert_offset_equal(2 * Hour(), datetime(2010, 1, 1), datetime(2010, 1, 1, 2))
    assert_offset_equal(-1 * Hour(), datetime(2010, 1, 1, 1), datetime(2010, 1, 1))

    assert Hour(3) + Hour(2) == Hour(5)
    assert Hour(3) - Hour(2) == Hour()

    assert Hour(4) != Hour(1)


def test_Minute():
    assert_offset_equal(Minute(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 1))
    assert_offset_equal(Minute(-1), datetime(2010, 1, 1, 0, 1), datetime(2010, 1, 1))
    assert_offset_equal(2 * Minute(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 2))
    assert_offset_equal(-1 * Minute(), datetime(2010, 1, 1, 0, 1), datetime(2010, 1, 1))

    assert Minute(3) + Minute(2) == Minute(5)
    assert Minute(3) - Minute(2) == Minute()
    assert Minute(5) != Minute()


def test_Second():
    assert_offset_equal(Second(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 1))
    assert_offset_equal(Second(-1), datetime(2010, 1, 1, 0, 0, 1), datetime(2010, 1, 1))
    assert_offset_equal(
        2 * Second(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 2)
    )
    assert_offset_equal(
        -1 * Second(), datetime(2010, 1, 1, 0, 0, 1), datetime(2010, 1, 1)
    )

    assert Second(3) + Second(2) == Second(5)
    assert Second(3) - Second(2) == Second()


def test_Millisecond():
    assert_offset_equal(
        Milli(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 0, 1000)
    )
    assert_offset_equal(
        Milli(-1), datetime(2010, 1, 1, 0, 0, 0, 1000), datetime(2010, 1, 1)
    )
    assert_offset_equal(
        Milli(2), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 0, 2000)
    )
    assert_offset_equal(
        2 * Milli(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 0, 2000)
    )
    assert_offset_equal(
        -1 * Milli(), datetime(2010, 1, 1, 0, 0, 0, 1000), datetime(2010, 1, 1)
    )

    assert Milli(3) + Milli(2) == Milli(5)
    assert Milli(3) - Milli(2) == Milli()


def test_MillisecondTimestampArithmetic():
    assert_offset_equal(
        Milli(), Timestamp("2010-01-01"), Timestamp("2010-01-01 00:00:00.001")
    )
    assert_offset_equal(
        Milli(-1), Timestamp("2010-01-01 00:00:00.001"), Timestamp("2010-01-01")
    )


def test_Microsecond():
    assert_offset_equal(Micro(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 0, 1))
    assert_offset_equal(
        Micro(-1), datetime(2010, 1, 1, 0, 0, 0, 1), datetime(2010, 1, 1)
    )

    assert_offset_equal(
        2 * Micro(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 0, 2)
    )
    assert_offset_equal(
        -1 * Micro(), datetime(2010, 1, 1, 0, 0, 0, 1), datetime(2010, 1, 1)
    )

    assert Micro(3) + Micro(2) == Micro(5)
    assert Micro(3) - Micro(2) == Micro()


def test_NanosecondGeneric():
    timestamp = Timestamp(datetime(2010, 1, 1))
    assert timestamp.nanosecond == 0

    result = timestamp + Nano(10)
    assert result.nanosecond == 10

    reverse_result = Nano(10) + timestamp
    assert reverse_result.nanosecond == 10


def test_Nanosecond():
    timestamp = Timestamp(datetime(2010, 1, 1))
    assert_offset_equal(Nano(), timestamp, timestamp + np.timedelta64(1, "ns"))
    assert_offset_equal(Nano(-1), timestamp + np.timedelta64(1, "ns"), timestamp)
    assert_offset_equal(2 * Nano(), timestamp, timestamp + np.timedelta64(2, "ns"))
    assert_offset_equal(-1 * Nano(), timestamp + np.timedelta64(1, "ns"), timestamp)

    assert Nano(3) + Nano(2) == Nano(5)
    assert Nano(3) - Nano(2) == Nano()

    # GH9284
    assert Nano(1) + Nano(10) == Nano(11)
    assert Nano(5) + Micro(1) == Nano(1005)
    assert Micro(5) + Nano(1) == Nano(5001)


@pytest.mark.parametrize(
    "kls, expected",
    [
        (Hour, Timedelta(hours=5)),
        (Minute, Timedelta(hours=2, minutes=3)),
        (Second, Timedelta(hours=2, seconds=3)),
        (Milli, Timedelta(hours=2, milliseconds=3)),
        (Micro, Timedelta(hours=2, microseconds=3)),
        (Nano, Timedelta(hours=2, nanoseconds=3)),
    ],
)
def test_tick_addition(kls, expected):
    offset = kls(3)
    td = Timedelta(hours=2)

    for other in [td, td.to_pytimedelta(), td.to_timedelta64()]:
        result = offset + other
        assert isinstance(result, Timedelta)
        assert result == expected

        result = other + offset
        assert isinstance(result, Timedelta)
        assert result == expected


def test_tick_delta_overflow():
    # GH#55503 raise OutOfBoundsTimedelta, not OverflowError
    tick = offsets.Day(10**9)
    msg = "Cannot cast 1000000000 days 00:00:00 to unit='ns' without overflow"
    depr_msg = "Day.delta is deprecated"
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            tick.delta


@pytest.mark.parametrize("cls", tick_classes)
def test_tick_division(cls):
    off = cls(10)

    assert off / cls(5) == 2
    assert off / 2 == cls(5)
    assert off / 2.0 == cls(5)

    assert off / off._as_pd_timedelta == 1
    assert off / off._as_pd_timedelta.to_timedelta64() == 1

    assert off / Nano(1) == off._as_pd_timedelta / Nano(1)._as_pd_timedelta

    if cls is not Nano:
        # A case where we end up with a smaller class
        result = off / 1000
        assert isinstance(result, offsets.Tick)
        assert not isinstance(result, cls)
        assert result._as_pd_timedelta == off._as_pd_timedelta / 1000

    if cls._nanos_inc < Timedelta(seconds=1)._value:
        # Case where we end up with a bigger class
        result = off / 0.001
        assert isinstance(result, offsets.Tick)
        assert not isinstance(result, cls)
        assert result._as_pd_timedelta == off._as_pd_timedelta / 0.001


def test_tick_mul_float():
    off = Micro(2)

    # Case where we retain type
    result = off * 1.5
    expected = Micro(3)
    assert result == expected
    assert isinstance(result, Micro)

    # Case where we bump up to the next type
    result = off * 1.25
    expected = Nano(2500)
    assert result == expected
    assert isinstance(result, Nano)


@pytest.mark.parametrize("cls", tick_classes)
def test_tick_rdiv(cls):
    off = cls(10)
    delta = off._as_pd_timedelta
    td64 = delta.to_timedelta64()
    instance__type = ".".join([cls.__module__, cls.__name__])
    msg = (
        "unsupported operand type\\(s\\) for \\/: 'int'|'float' and "
        f"'{instance__type}'"
    )

    with pytest.raises(TypeError, match=msg):
        2 / off
    with pytest.raises(TypeError, match=msg):
        2.0 / off

    assert (td64 * 2.5) / off == 2.5

    if cls is not Nano:
        # skip pytimedelta for Nano since it gets dropped
        assert (delta.to_pytimedelta() * 2) / off == 2

    result = np.array([2 * td64, td64]) / off
    expected = np.array([2.0, 1.0])
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("cls1", tick_classes)
@pytest.mark.parametrize("cls2", tick_classes)
def test_tick_zero(cls1, cls2):
    assert cls1(0) == cls2(0)
    assert cls1(0) + cls2(0) == cls1(0)

    if cls1 is not Nano:
        assert cls1(2) + cls2(0) == cls1(2)

    if cls1 is Nano:
        assert cls1(2) + Nano(0) == cls1(2)


@pytest.mark.parametrize("cls", tick_classes)
def test_tick_equalities(cls):
    assert cls() == cls(1)


@pytest.mark.parametrize("cls", tick_classes)
def test_tick_offset(cls):
    msg = f"{cls.__name__}.is_anchored is deprecated "

    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert not cls().is_anchored()


@pytest.mark.parametrize("cls", tick_classes)
def test_compare_ticks(cls):
    three = cls(3)
    four = cls(4)

    assert three < cls(4)
    assert cls(3) < four
    assert four > cls(3)
    assert cls(4) > three
    assert cls(3) == cls(3)
    assert cls(3) != cls(4)


@pytest.mark.parametrize("cls", tick_classes)
def test_compare_ticks_to_strs(cls):
    # GH#23524
    off = cls(19)

    # These tests should work with any strings, but we particularly are
    #  interested in "infer" as that comparison is convenient to make in
    #  Datetime/Timedelta Array/Index constructors
    assert not off == "infer"
    assert not "foo" == off

    instance_type = ".".join([cls.__module__, cls.__name__])
    msg = (
        "'<'|'<='|'>'|'>=' not supported between instances of "
        f"'str' and '{instance_type}'|'{instance_type}' and 'str'"
    )

    for left, right in [("infer", off), (off, "infer")]:
        with pytest.raises(TypeError, match=msg):
            left < right
        with pytest.raises(TypeError, match=msg):
            left <= right
        with pytest.raises(TypeError, match=msg):
            left > right
        with pytest.raises(TypeError, match=msg):
            left >= right


@pytest.mark.parametrize("cls", tick_classes)
def test_compare_ticks_to_timedeltalike(cls):
    off = cls(19)

    td = off._as_pd_timedelta

    others = [td, td.to_timedelta64()]
    if cls is not Nano:
        others.append(td.to_pytimedelta())

    for other in others:
        assert off == other
        assert not off != other
        assert not off < other
        assert not off > other
        assert off <= other
        assert off >= other

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_scalarprint.py ===
""" Test printing of scalar types.

"""
import platform

import pytest

import numpy as np
from numpy.testing import IS_MUSL, assert_, assert_equal, assert_raises


class TestRealScalars:
    def test_str(self):
        svals = [0.0, -0.0, 1, -1, np.inf, -np.inf, np.nan]
        styps = [np.float16, np.float32, np.float64, np.longdouble]
        wanted = [
             ['0.0',  '0.0',  '0.0',  '0.0' ],  # noqa: E202
             ['-0.0', '-0.0', '-0.0', '-0.0'],
             ['1.0',  '1.0',  '1.0',  '1.0' ],  # noqa: E202
             ['-1.0', '-1.0', '-1.0', '-1.0'],
             ['inf',  'inf',  'inf',  'inf' ],  # noqa: E202
             ['-inf', '-inf', '-inf', '-inf'],
             ['nan',  'nan',  'nan',  'nan' ]]  # noqa: E202

        for wants, val in zip(wanted, svals):
            for want, styp in zip(wants, styps):
                msg = f'for str({np.dtype(styp).name}({val!r}))'
                assert_equal(str(styp(val)), want, err_msg=msg)

    def test_scalar_cutoffs(self):
        # test that both the str and repr of np.float64 behaves
        # like python floats in python3.
        def check(v):
            assert_equal(str(np.float64(v)), str(v))
            assert_equal(str(np.float64(v)), repr(v))
            assert_equal(repr(np.float64(v)), f"np.float64({v!r})")
            assert_equal(repr(np.float64(v)), f"np.float64({v})")

        # check we use the same number of significant digits
        check(1.12345678901234567890)
        check(0.0112345678901234567890)

        # check switch from scientific output to positional and back
        check(1e-5)
        check(1e-4)
        check(1e15)
        check(1e16)

    test_cases_gh_28679 = [
        (np.half, -0.000099, "-9.9e-05"),
        (np.half, 0.0001, "0.0001"),
        (np.half, 999, "999.0"),
        (np.half, -1000, "-1e+03"),
        (np.single, 0.000099, "9.9e-05"),
        (np.single, -0.000100001, "-0.000100001"),
        (np.single, 999999, "999999.0"),
        (np.single, -1000000, "-1e+06")
    ]

    @pytest.mark.parametrize("dtype, input_val, expected_str", test_cases_gh_28679)
    def test_gh_28679(self, dtype, input_val, expected_str):
        # test cutoff to exponent notation for half and single
        assert_equal(str(dtype(input_val)), expected_str)

    test_cases_legacy_2_2 = [
        (np.half(65504), "65500.0"),
        (np.single(1.e15), "1000000000000000.0"),
        (np.single(1.e16), "1e+16"),
    ]

    @pytest.mark.parametrize("input_val, expected_str", test_cases_legacy_2_2)
    def test_legacy_2_2_mode(self, input_val, expected_str):
        # test legacy cutoff to exponent notation for half and single
        with np.printoptions(legacy='2.2'):
            assert_equal(str(input_val), expected_str)

    def test_dragon4(self):
        # these tests are adapted from Ryan Juckett's dragon4 implementation,
        # see dragon4.c for details.

        fpos32 = lambda x, **k: np.format_float_positional(np.float32(x), **k)
        fsci32 = lambda x, **k: np.format_float_scientific(np.float32(x), **k)
        fpos64 = lambda x, **k: np.format_float_positional(np.float64(x), **k)
        fsci64 = lambda x, **k: np.format_float_scientific(np.float64(x), **k)

        preckwd = lambda prec: {'unique': False, 'precision': prec}

        assert_equal(fpos32('1.0'), "1.")
        assert_equal(fsci32('1.0'), "1.e+00")
        assert_equal(fpos32('10.234'), "10.234")
        assert_equal(fpos32('-10.234'), "-10.234")
        assert_equal(fsci32('10.234'), "1.0234e+01")
        assert_equal(fsci32('-10.234'), "-1.0234e+01")
        assert_equal(fpos32('1000.0'), "1000.")
        assert_equal(fpos32('1.0', precision=0), "1.")
        assert_equal(fsci32('1.0', precision=0), "1.e+00")
        assert_equal(fpos32('10.234', precision=0), "10.")
        assert_equal(fpos32('-10.234', precision=0), "-10.")
        assert_equal(fsci32('10.234', precision=0), "1.e+01")
        assert_equal(fsci32('-10.234', precision=0), "-1.e+01")
        assert_equal(fpos32('10.234', precision=2), "10.23")
        assert_equal(fsci32('-10.234', precision=2), "-1.02e+01")
        assert_equal(fsci64('9.9999999999999995e-08', **preckwd(16)),
                            '9.9999999999999995e-08')
        assert_equal(fsci64('9.8813129168249309e-324', **preckwd(16)),
                            '9.8813129168249309e-324')
        assert_equal(fsci64('9.9999999999999694e-311', **preckwd(16)),
                            '9.9999999999999694e-311')

        # test rounding
        # 3.1415927410 is closest float32 to np.pi
        assert_equal(fpos32('3.14159265358979323846', **preckwd(10)),
                            "3.1415927410")
        assert_equal(fsci32('3.14159265358979323846', **preckwd(10)),
                            "3.1415927410e+00")
        assert_equal(fpos64('3.14159265358979323846', **preckwd(10)),
                            "3.1415926536")
        assert_equal(fsci64('3.14159265358979323846', **preckwd(10)),
                            "3.1415926536e+00")
        # 299792448 is closest float32 to 299792458
        assert_equal(fpos32('299792458.0', **preckwd(5)), "299792448.00000")
        assert_equal(fsci32('299792458.0', **preckwd(5)), "2.99792e+08")
        assert_equal(fpos64('299792458.0', **preckwd(5)), "299792458.00000")
        assert_equal(fsci64('299792458.0', **preckwd(5)), "2.99792e+08")

        assert_equal(fpos32('3.14159265358979323846', **preckwd(25)),
                            "3.1415927410125732421875000")
        assert_equal(fpos64('3.14159265358979323846', **preckwd(50)),
                         "3.14159265358979311599796346854418516159057617187500")
        assert_equal(fpos64('3.14159265358979323846'), "3.141592653589793")

        # smallest numbers
        assert_equal(fpos32(0.5**(126 + 23), unique=False, precision=149),
                    "0.00000000000000000000000000000000000000000000140129846432"
                    "4817070923729583289916131280261941876515771757068283889791"
                    "08268586060148663818836212158203125")

        assert_equal(fpos64(5e-324, unique=False, precision=1074),
                    "0.00000000000000000000000000000000000000000000000000000000"
                    "0000000000000000000000000000000000000000000000000000000000"
                    "0000000000000000000000000000000000000000000000000000000000"
                    "0000000000000000000000000000000000000000000000000000000000"
                    "0000000000000000000000000000000000000000000000000000000000"
                    "0000000000000000000000000000000000049406564584124654417656"
                    "8792868221372365059802614324764425585682500675507270208751"
                    "8652998363616359923797965646954457177309266567103559397963"
                    "9877479601078187812630071319031140452784581716784898210368"
                    "8718636056998730723050006387409153564984387312473397273169"
                    "6151400317153853980741262385655911710266585566867681870395"
                    "6031062493194527159149245532930545654440112748012970999954"
                    "1931989409080416563324524757147869014726780159355238611550"
                    "1348035264934720193790268107107491703332226844753335720832"
                    "4319360923828934583680601060115061698097530783422773183292"
                    "4790498252473077637592724787465608477820373446969953364701"
                    "7972677717585125660551199131504891101451037862738167250955"
                    "8373897335989936648099411642057026370902792427675445652290"
                    "87538682506419718265533447265625")

        # largest numbers
        f32x = np.finfo(np.float32).max
        assert_equal(fpos32(f32x, **preckwd(0)),
                    "340282346638528859811704183484516925440.")
        assert_equal(fpos64(np.finfo(np.float64).max, **preckwd(0)),
                    "1797693134862315708145274237317043567980705675258449965989"
                    "1747680315726078002853876058955863276687817154045895351438"
                    "2464234321326889464182768467546703537516986049910576551282"
                    "0762454900903893289440758685084551339423045832369032229481"
                    "6580855933212334827479782620414472316873817718091929988125"
                    "0404026184124858368.")
        # Warning: In unique mode only the integer digits necessary for
        # uniqueness are computed, the rest are 0.
        assert_equal(fpos32(f32x),
                    "340282350000000000000000000000000000000.")

        # Further tests of zero-padding vs rounding in different combinations
        # of unique, fractional, precision, min_digits
        # precision can only reduce digits, not add them.
        # min_digits can only extend digits, not reduce them.
        assert_equal(fpos32(f32x, unique=True, fractional=True, precision=0),
                    "340282350000000000000000000000000000000.")
        assert_equal(fpos32(f32x, unique=True, fractional=True, precision=4),
                    "340282350000000000000000000000000000000.")
        assert_equal(fpos32(f32x, unique=True, fractional=True, min_digits=0),
                    "340282346638528859811704183484516925440.")
        assert_equal(fpos32(f32x, unique=True, fractional=True, min_digits=4),
                    "340282346638528859811704183484516925440.0000")
        assert_equal(fpos32(f32x, unique=True, fractional=True,
                                    min_digits=4, precision=4),
                    "340282346638528859811704183484516925440.0000")
        assert_raises(ValueError, fpos32, f32x, unique=True, fractional=False,
                                          precision=0)
        assert_equal(fpos32(f32x, unique=True, fractional=False, precision=4),
                    "340300000000000000000000000000000000000.")
        assert_equal(fpos32(f32x, unique=True, fractional=False, precision=20),
                    "340282350000000000000000000000000000000.")
        assert_equal(fpos32(f32x, unique=True, fractional=False, min_digits=4),
                    "340282350000000000000000000000000000000.")
        assert_equal(fpos32(f32x, unique=True, fractional=False,
                                  min_digits=20),
                    "340282346638528859810000000000000000000.")
        assert_equal(fpos32(f32x, unique=True, fractional=False,
                                  min_digits=15),
                    "340282346638529000000000000000000000000.")
        assert_equal(fpos32(f32x, unique=False, fractional=False, precision=4),
                    "340300000000000000000000000000000000000.")
        # test that unique rounding is preserved when precision is supplied
        # but no extra digits need to be printed (gh-18609)
        a = np.float64.fromhex('-1p-97')
        assert_equal(fsci64(a, unique=True), '-6.310887241768095e-30')
        assert_equal(fsci64(a, unique=False, precision=15),
                     '-6.310887241768094e-30')
        assert_equal(fsci64(a, unique=True, precision=15),
                     '-6.310887241768095e-30')
        assert_equal(fsci64(a, unique=True, min_digits=15),
                     '-6.310887241768095e-30')
        assert_equal(fsci64(a, unique=True, precision=15, min_digits=15),
                     '-6.310887241768095e-30')
        # adds/remove digits in unique mode with unbiased rnding
        assert_equal(fsci64(a, unique=True, precision=14),
                     '-6.31088724176809e-30')
        assert_equal(fsci64(a, unique=True, min_digits=16),
                     '-6.3108872417680944e-30')
        assert_equal(fsci64(a, unique=True, precision=16),
                     '-6.310887241768095e-30')
        assert_equal(fsci64(a, unique=True, min_digits=14),
                     '-6.310887241768095e-30')
        # test min_digits in unique mode with different rounding cases
        assert_equal(fsci64('1e120', min_digits=3), '1.000e+120')
        assert_equal(fsci64('1e100', min_digits=3), '1.000e+100')

        # test trailing zeros
        assert_equal(fpos32('1.0', unique=False, precision=3), "1.000")
        assert_equal(fpos64('1.0', unique=False, precision=3), "1.000")
        assert_equal(fsci32('1.0', unique=False, precision=3), "1.000e+00")
        assert_equal(fsci64('1.0', unique=False, precision=3), "1.000e+00")
        assert_equal(fpos32('1.5', unique=False, precision=3), "1.500")
        assert_equal(fpos64('1.5', unique=False, precision=3), "1.500")
        assert_equal(fsci32('1.5', unique=False, precision=3), "1.500e+00")
        assert_equal(fsci64('1.5', unique=False, precision=3), "1.500e+00")
        # gh-10713
        assert_equal(fpos64('324', unique=False, precision=5,
                                   fractional=False), "324.00")

    available_float_dtypes = [np.float16, np.float32, np.float64, np.float128]\
        if hasattr(np, 'float128') else [np.float16, np.float32, np.float64]

    @pytest.mark.parametrize("tp", available_float_dtypes)
    def test_dragon4_positional_interface(self, tp):
        # test is flaky for musllinux on np.float128
        if IS_MUSL and tp == np.float128:
            pytest.skip("Skipping flaky test of float128 on musllinux")

        fpos = np.format_float_positional

        # test padding
        assert_equal(fpos(tp('1.0'), pad_left=4, pad_right=4), "   1.    ")
        assert_equal(fpos(tp('-1.0'), pad_left=4, pad_right=4), "  -1.    ")
        assert_equal(fpos(tp('-10.2'),
                        pad_left=4, pad_right=4), " -10.2   ")

        # test fixed (non-unique) mode
        assert_equal(fpos(tp('1.0'), unique=False, precision=4), "1.0000")

    @pytest.mark.parametrize("tp", available_float_dtypes)
    def test_dragon4_positional_interface_trim(self, tp):
        # test is flaky for musllinux on np.float128
        if IS_MUSL and tp == np.float128:
            pytest.skip("Skipping flaky test of float128 on musllinux")

        fpos = np.format_float_positional
        # test trimming
        # trim of 'k' or '.' only affects non-unique mode, since unique
        # mode will not output trailing 0s.
        assert_equal(fpos(tp('1.'), unique=False, precision=4, trim='k'),
                        "1.0000")

        assert_equal(fpos(tp('1.'), unique=False, precision=4, trim='.'),
                        "1.")
        assert_equal(fpos(tp('1.2'), unique=False, precision=4, trim='.'),
                        "1.2" if tp != np.float16 else "1.2002")

        assert_equal(fpos(tp('1.'), unique=False, precision=4, trim='0'),
                        "1.0")
        assert_equal(fpos(tp('1.2'), unique=False, precision=4, trim='0'),
                        "1.2" if tp != np.float16 else "1.2002")
        assert_equal(fpos(tp('1.'), trim='0'), "1.0")

        assert_equal(fpos(tp('1.'), unique=False, precision=4, trim='-'),
                        "1")
        assert_equal(fpos(tp('1.2'), unique=False, precision=4, trim='-'),
                        "1.2" if tp != np.float16 else "1.2002")
        assert_equal(fpos(tp('1.'), trim='-'), "1")
        assert_equal(fpos(tp('1.001'), precision=1, trim='-'), "1")

    @pytest.mark.parametrize("tp", available_float_dtypes)
    @pytest.mark.parametrize("pad_val", [10**5, np.iinfo("int32").max])
    def test_dragon4_positional_interface_overflow(self, tp, pad_val):
        # test is flaky for musllinux on np.float128
        if IS_MUSL and tp == np.float128:
            pytest.skip("Skipping flaky test of float128 on musllinux")

        fpos = np.format_float_positional

        # gh-28068
        with pytest.raises(RuntimeError,
                           match="Float formatting result too large"):
            fpos(tp('1.047'), unique=False, precision=pad_val)

        with pytest.raises(RuntimeError,
                           match="Float formatting result too large"):
            fpos(tp('1.047'), precision=2, pad_left=pad_val)

        with pytest.raises(RuntimeError,
                           match="Float formatting result too large"):
            fpos(tp('1.047'), precision=2, pad_right=pad_val)

    @pytest.mark.parametrize("tp", available_float_dtypes)
    def test_dragon4_scientific_interface(self, tp):
        # test is flaky for musllinux on np.float128
        if IS_MUSL and tp == np.float128:
            pytest.skip("Skipping flaky test of float128 on musllinux")

        fsci = np.format_float_scientific

        # test exp_digits
        assert_equal(fsci(tp('1.23e1'), exp_digits=5), "1.23e+00001")

        # test fixed (non-unique) mode
        assert_equal(fsci(tp('1.0'), unique=False, precision=4),
                        "1.0000e+00")

    @pytest.mark.skipif(not platform.machine().startswith("ppc64"),
                        reason="only applies to ppc float128 values")
    def test_ppc64_ibm_double_double128(self):
        # check that the precision decreases once we get into the subnormal
        # range. Unlike float64, this starts around 1e-292 instead of 1e-308,
        # which happens when the first double is normal and the second is
        # subnormal.
        x = np.float128('2.123123123123123123123123123123123e-286')
        got = [str(x / np.float128('2e' + str(i))) for i in range(40)]
        expected = [
            "1.06156156156156156156156156156157e-286",
            "1.06156156156156156156156156156158e-287",
            "1.06156156156156156156156156156159e-288",
            "1.0615615615615615615615615615616e-289",
            "1.06156156156156156156156156156157e-290",
            "1.06156156156156156156156156156156e-291",
            "1.0615615615615615615615615615616e-292",
            "1.0615615615615615615615615615615e-293",
            "1.061561561561561561561561561562e-294",
            "1.06156156156156156156156156155e-295",
            "1.0615615615615615615615615616e-296",
            "1.06156156156156156156156156e-297",
            "1.06156156156156156156156157e-298",
            "1.0615615615615615615615616e-299",
            "1.06156156156156156156156e-300",
            "1.06156156156156156156155e-301",
            "1.0615615615615615615616e-302",
            "1.061561561561561561562e-303",
            "1.06156156156156156156e-304",
            "1.0615615615615615618e-305",
            "1.06156156156156156e-306",
            "1.06156156156156157e-307",
            "1.0615615615615616e-308",
            "1.06156156156156e-309",
            "1.06156156156157e-310",
            "1.0615615615616e-311",
            "1.06156156156e-312",
            "1.06156156154e-313",
            "1.0615615616e-314",
            "1.06156156e-315",
            "1.06156155e-316",
            "1.061562e-317",
            "1.06156e-318",
            "1.06155e-319",
            "1.0617e-320",
            "1.06e-321",
            "1.04e-322",
            "1e-323",
            "0.0",
            "0.0"]
        assert_equal(got, expected)

        # Note: we follow glibc behavior, but it (or gcc) might not be right.
        # In particular we can get two values that print the same but are not
        # equal:
        a = np.float128('2') / np.float128('3')
        b = np.float128(str(a))
        assert_equal(str(a), str(b))
        assert_(a != b)

    def float32_roundtrip(self):
        # gh-9360
        x = np.float32(1024 - 2**-14)
        y = np.float32(1024 - 2**-13)
        assert_(repr(x) != repr(y))
        assert_equal(np.float32(repr(x)), x)
        assert_equal(np.float32(repr(y)), y)

    def float64_vs_python(self):
        # gh-2643, gh-6136, gh-6908
        assert_equal(repr(np.float64(0.1)), repr(0.1))
        assert_(repr(np.float64(0.20000000000000004)) != repr(0.2))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\tests\test_hyper.py ===
from sympy.core.containers import Tuple
from sympy.core.function import Derivative
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import (appellf1, hyper, meijerg)
from sympy.series.order import O
from sympy.abc import x, z, k
from sympy.series.limits import limit
from sympy.testing.pytest import raises, slow
from sympy.core.random import (
    random_complex_number as randcplx,
    verify_numerically as tn,
    test_derivative_numerically as td)


def test_TupleParametersBase():
    # test that our implementation of the chain rule works
    p = hyper((), (), z**2)
    assert p.diff(z) == p*2*z


def test_hyper():
    raises(TypeError, lambda: hyper(1, 2, z))

    assert hyper((2, 1), (1,), z) == hyper(Tuple(1, 2), Tuple(1), z)
    assert hyper((2, 1, 2), (1, 2, 1, 3), z) == hyper((2,), (1, 3), z)
    u = hyper((2, 1, 2), (1, 2, 1, 3), z, evaluate=False)
    assert u.ap == Tuple(1, 2, 2)
    assert u.bq == Tuple(1, 1, 2, 3)

    h = hyper((1, 2), (3, 4, 5), z)
    assert h.ap == Tuple(1, 2)
    assert h.bq == Tuple(3, 4, 5)
    assert h.argument == z
    assert h.is_commutative is True
    h = hyper((2, 1), (4, 3, 5), z)
    assert h.ap == Tuple(1, 2)
    assert h.bq == Tuple(3, 4, 5)
    assert h.argument == z
    assert h.is_commutative is True

    # just a few checks to make sure that all arguments go where they should
    assert tn(hyper(Tuple(), Tuple(), z), exp(z), z)
    assert tn(z*hyper((1, 1), Tuple(2), -z), log(1 + z), z)

    # differentiation
    h = hyper(
        (randcplx(), randcplx(), randcplx()), (randcplx(), randcplx()), z)
    assert td(h, z)

    a1, a2, b1, b2, b3 = symbols('a1:3, b1:4')
    assert hyper((a1, a2), (b1, b2, b3), z).diff(z) == \
        a1*a2/(b1*b2*b3) * hyper((a1 + 1, a2 + 1), (b1 + 1, b2 + 1, b3 + 1), z)

    # differentiation wrt parameters is not supported
    assert hyper([z], [], z).diff(z) == Derivative(hyper([z], [], z), z)

    # hyper is unbranched wrt parameters
    from sympy.functions.elementary.complexes import polar_lift
    assert hyper([polar_lift(z)], [polar_lift(k)], polar_lift(x)) == \
        hyper([z], [k], polar_lift(x))

    # hyper does not automatically evaluate anyway, but the test is to make
    # sure that the evaluate keyword is accepted
    assert hyper((1, 2), (1,), z, evaluate=False).func is hyper


def test_expand_func():
    # evaluation at 1 of Gauss' hypergeometric function:
    from sympy.abc import a, b, c
    from sympy.core.function import expand_func
    a1, b1, c1 = randcplx(), randcplx(), randcplx() + 5
    assert expand_func(hyper([a, b], [c], 1)) == \
        gamma(c)*gamma(-a - b + c)/(gamma(-a + c)*gamma(-b + c))
    assert abs(expand_func(hyper([a1, b1], [c1], 1)).n()
               - hyper([a1, b1], [c1], 1).n()) < 1e-10

    # hyperexpand wrapper for hyper:
    assert expand_func(hyper([], [], z)) == exp(z)
    assert expand_func(hyper([1, 2, 3], [], z)) == hyper([1, 2, 3], [], z)
    assert expand_func(meijerg([[1, 1], []], [[1], [0]], z)) == log(z + 1)
    assert expand_func(meijerg([[1, 1], []], [[], []], z)) == \
        meijerg([[1, 1], []], [[], []], z)


def replace_dummy(expr, sym):
    from sympy.core.symbol import Dummy
    dum = expr.atoms(Dummy)
    if not dum:
        return expr
    assert len(dum) == 1
    return expr.xreplace({dum.pop(): sym})


def test_hyper_rewrite_sum():
    from sympy.concrete.summations import Sum
    from sympy.core.symbol import Dummy
    from sympy.functions.combinatorial.factorials import (RisingFactorial, factorial)
    _k = Dummy("k")
    assert replace_dummy(hyper((1, 2), (1, 3), x).rewrite(Sum), _k) == \
        Sum(x**_k / factorial(_k) * RisingFactorial(2, _k) /
            RisingFactorial(3, _k), (_k, 0, oo))

    assert hyper((1, 2, 3), (-1, 3), z).rewrite(Sum) == \
        hyper((1, 2, 3), (-1, 3), z)


def test_radius_of_convergence():
    assert hyper((1, 2), [3], z).radius_of_convergence == 1
    assert hyper((1, 2), [3, 4], z).radius_of_convergence is oo
    assert hyper((1, 2, 3), [4], z).radius_of_convergence == 0
    assert hyper((0, 1, 2), [4], z).radius_of_convergence is oo
    assert hyper((-1, 1, 2), [-4], z).radius_of_convergence == 0
    assert hyper((-1, -2, 2), [-1], z).radius_of_convergence is oo
    assert hyper((-1, 2), [-1, -2], z).radius_of_convergence == 0
    assert hyper([-1, 1, 3], [-2, 2], z).radius_of_convergence == 1
    assert hyper([-1, 1], [-2, 2], z).radius_of_convergence is oo
    assert hyper([-1, 1, 3], [-2], z).radius_of_convergence == 0
    assert hyper((-1, 2, 3, 4), [], z).radius_of_convergence is oo

    assert hyper([1, 1], [3], 1).convergence_statement == True
    assert hyper([1, 1], [2], 1).convergence_statement == False
    assert hyper([1, 1], [2], -1).convergence_statement == True
    assert hyper([1, 1], [1], -1).convergence_statement == False


def test_meijer():
    raises(TypeError, lambda: meijerg(1, z))
    raises(TypeError, lambda: meijerg(((1,), (2,)), (3,), (4,), z))

    assert meijerg(((1, 2), (3,)), ((4,), (5,)), z) == \
        meijerg(Tuple(1, 2), Tuple(3), Tuple(4), Tuple(5), z)

    g = meijerg((1, 2), (3, 4, 5), (6, 7, 8, 9), (10, 11, 12, 13, 14), z)
    assert g.an == Tuple(1, 2)
    assert g.ap == Tuple(1, 2, 3, 4, 5)
    assert g.aother == Tuple(3, 4, 5)
    assert g.bm == Tuple(6, 7, 8, 9)
    assert g.bq == Tuple(6, 7, 8, 9, 10, 11, 12, 13, 14)
    assert g.bother == Tuple(10, 11, 12, 13, 14)
    assert g.argument == z
    assert g.nu == 75
    assert g.delta == -1
    assert g.is_commutative is True
    assert g.is_number is False
    #issue 13071
    assert meijerg([[],[]], [[S.Half],[0]], 1).is_number is True

    assert meijerg([1, 2], [3], [4], [5], z).delta == S.Half

    # just a few checks to make sure that all arguments go where they should
    assert tn(meijerg(Tuple(), Tuple(), Tuple(0), Tuple(), -z), exp(z), z)
    assert tn(sqrt(pi)*meijerg(Tuple(), Tuple(),
                               Tuple(0), Tuple(S.Half), z**2/4), cos(z), z)
    assert tn(meijerg(Tuple(1, 1), Tuple(), Tuple(1), Tuple(0), z),
              log(1 + z), z)

    # test exceptions
    raises(ValueError, lambda: meijerg(((3, 1), (2,)), ((oo,), (2, 0)), x))
    raises(ValueError, lambda: meijerg(((3, 1), (2,)), ((1,), (2, 0)), x))

    # differentiation
    g = meijerg((randcplx(),), (randcplx() + 2*I,), Tuple(),
                (randcplx(), randcplx()), z)
    assert td(g, z)

    g = meijerg(Tuple(), (randcplx(),), Tuple(),
                (randcplx(), randcplx()), z)
    assert td(g, z)

    g = meijerg(Tuple(), Tuple(), Tuple(randcplx()),
                Tuple(randcplx(), randcplx()), z)
    assert td(g, z)

    a1, a2, b1, b2, c1, c2, d1, d2 = symbols('a1:3, b1:3, c1:3, d1:3')
    assert meijerg((a1, a2), (b1, b2), (c1, c2), (d1, d2), z).diff(z) == \
        (meijerg((a1 - 1, a2), (b1, b2), (c1, c2), (d1, d2), z)
         + (a1 - 1)*meijerg((a1, a2), (b1, b2), (c1, c2), (d1, d2), z))/z

    assert meijerg([z, z], [], [], [], z).diff(z) == \
        Derivative(meijerg([z, z], [], [], [], z), z)

    # meijerg is unbranched wrt parameters
    from sympy.functions.elementary.complexes import polar_lift as pl
    assert meijerg([pl(a1)], [pl(a2)], [pl(b1)], [pl(b2)], pl(z)) == \
        meijerg([a1], [a2], [b1], [b2], pl(z))

    # integrand
    from sympy.abc import a, b, c, d, s
    assert meijerg([a], [b], [c], [d], z).integrand(s) == \
        z**s*gamma(c - s)*gamma(-a + s + 1)/(gamma(b - s)*gamma(-d + s + 1))


def test_meijerg_derivative():
    assert meijerg([], [1, 1], [0, 0, x], [], z).diff(x) == \
        log(z)*meijerg([], [1, 1], [0, 0, x], [], z) \
        + 2*meijerg([], [1, 1, 1], [0, 0, x, 0], [], z)

    y = randcplx()
    a = 5  # mpmath chokes with non-real numbers, and Mod1 with floats
    assert td(meijerg([x], [], [], [], y), x)
    assert td(meijerg([x**2], [], [], [], y), x)
    assert td(meijerg([], [x], [], [], y), x)
    assert td(meijerg([], [], [x], [], y), x)
    assert td(meijerg([], [], [], [x], y), x)
    assert td(meijerg([x], [a], [a + 1], [], y), x)
    assert td(meijerg([x], [a + 1], [a], [], y), x)
    assert td(meijerg([x, a], [], [], [a + 1], y), x)
    assert td(meijerg([x, a + 1], [], [], [a], y), x)
    b = Rational(3, 2)
    assert td(meijerg([a + 2], [b], [b - 3, x], [a], y), x)


def test_meijerg_period():
    assert meijerg([], [1], [0], [], x).get_period() == 2*pi
    assert meijerg([1], [], [], [0], x).get_period() == 2*pi
    assert meijerg([], [], [0], [], x).get_period() == 2*pi  # exp(x)
    assert meijerg(
        [], [], [0], [S.Half], x).get_period() == 2*pi  # cos(sqrt(x))
    assert meijerg(
        [], [], [S.Half], [0], x).get_period() == 4*pi  # sin(sqrt(x))
    assert meijerg([1, 1], [], [1], [0], x).get_period() is oo  # log(1 + x)


def test_hyper_unpolarify():
    from sympy.functions.elementary.exponential import exp_polar
    a = exp_polar(2*pi*I)*x
    b = x
    assert hyper([], [], a).argument == b
    assert hyper([0], [], a).argument == a
    assert hyper([0], [0], a).argument == b
    assert hyper([0, 1], [0], a).argument == a
    assert hyper([0, 1], [0], exp_polar(2*pi*I)).argument == 1


@slow
def test_hyperrep():
    from sympy.functions.special.hyper import (HyperRep, HyperRep_atanh,
        HyperRep_power1, HyperRep_power2, HyperRep_log1, HyperRep_asin1,
        HyperRep_asin2, HyperRep_sqrts1, HyperRep_sqrts2, HyperRep_log2,
        HyperRep_cosasin, HyperRep_sinasin)
    # First test the base class works.
    from sympy.functions.elementary.exponential import exp_polar
    from sympy.functions.elementary.piecewise import Piecewise
    a, b, c, d, z = symbols('a b c d z')

    class myrep(HyperRep):
        @classmethod
        def _expr_small(cls, x):
            return a

        @classmethod
        def _expr_small_minus(cls, x):
            return b

        @classmethod
        def _expr_big(cls, x, n):
            return c*n

        @classmethod
        def _expr_big_minus(cls, x, n):
            return d*n
    assert myrep(z).rewrite('nonrep') == Piecewise((0, abs(z) > 1), (a, True))
    assert myrep(exp_polar(I*pi)*z).rewrite('nonrep') == \
        Piecewise((0, abs(z) > 1), (b, True))
    assert myrep(exp_polar(2*I*pi)*z).rewrite('nonrep') == \
        Piecewise((c, abs(z) > 1), (a, True))
    assert myrep(exp_polar(3*I*pi)*z).rewrite('nonrep') == \
        Piecewise((d, abs(z) > 1), (b, True))
    assert myrep(exp_polar(4*I*pi)*z).rewrite('nonrep') == \
        Piecewise((2*c, abs(z) > 1), (a, True))
    assert myrep(exp_polar(5*I*pi)*z).rewrite('nonrep') == \
        Piecewise((2*d, abs(z) > 1), (b, True))
    assert myrep(z).rewrite('nonrepsmall') == a
    assert myrep(exp_polar(I*pi)*z).rewrite('nonrepsmall') == b

    def t(func, hyp, z):
        """ Test that func is a valid representation of hyp. """
        # First test that func agrees with hyp for small z
        if not tn(func.rewrite('nonrepsmall'), hyp, z,
                  a=Rational(-1, 2), b=Rational(-1, 2), c=S.Half, d=S.Half):
            return False
        # Next check that the two small representations agree.
        if not tn(
            func.rewrite('nonrepsmall').subs(
                z, exp_polar(I*pi)*z).replace(exp_polar, exp),
            func.subs(z, exp_polar(I*pi)*z).rewrite('nonrepsmall'),
                z, a=Rational(-1, 2), b=Rational(-1, 2), c=S.Half, d=S.Half):
            return False
        # Next check continuity along exp_polar(I*pi)*t
        expr = func.subs(z, exp_polar(I*pi)*z).rewrite('nonrep')
        if abs(expr.subs(z, 1 + 1e-15).n() - expr.subs(z, 1 - 1e-15).n()) > 1e-10:
            return False
        # Finally check continuity of the big reps.

        def dosubs(func, a, b):
            rv = func.subs(z, exp_polar(a)*z).rewrite('nonrep')
            return rv.subs(z, exp_polar(b)*z).replace(exp_polar, exp)
        for n in [0, 1, 2, 3, 4, -1, -2, -3, -4]:
            expr1 = dosubs(func, 2*I*pi*n, I*pi/2)
            expr2 = dosubs(func, 2*I*pi*n + I*pi, -I*pi/2)
            if not tn(expr1, expr2, z):
                return False
            expr1 = dosubs(func, 2*I*pi*(n + 1), -I*pi/2)
            expr2 = dosubs(func, 2*I*pi*n + I*pi, I*pi/2)
            if not tn(expr1, expr2, z):
                return False
        return True

    # Now test the various representatives.
    a = Rational(1, 3)
    assert t(HyperRep_atanh(z), hyper([S.Half, 1], [Rational(3, 2)], z), z)
    assert t(HyperRep_power1(a, z), hyper([-a], [], z), z)
    assert t(HyperRep_power2(a, z), hyper([a, a - S.Half], [2*a], z), z)
    assert t(HyperRep_log1(z), -z*hyper([1, 1], [2], z), z)
    assert t(HyperRep_asin1(z), hyper([S.Half, S.Half], [Rational(3, 2)], z), z)
    assert t(HyperRep_asin2(z), hyper([1, 1], [Rational(3, 2)], z), z)
    assert t(HyperRep_sqrts1(a, z), hyper([-a, S.Half - a], [S.Half], z), z)
    assert t(HyperRep_sqrts2(a, z),
             -2*z/(2*a + 1)*hyper([-a - S.Half, -a], [S.Half], z).diff(z), z)
    assert t(HyperRep_log2(z), -z/4*hyper([Rational(3, 2), 1, 1], [2, 2], z), z)
    assert t(HyperRep_cosasin(a, z), hyper([-a, a], [S.Half], z), z)
    assert t(HyperRep_sinasin(a, z), 2*a*z*hyper([1 - a, 1 + a], [Rational(3, 2)], z), z)


@slow
def test_meijerg_eval():
    from sympy.functions.elementary.exponential import exp_polar
    from sympy.functions.special.bessel import besseli
    from sympy.abc import l
    a = randcplx()
    arg = x*exp_polar(k*pi*I)
    expr1 = pi*meijerg([[], [(a + 1)/2]], [[a/2], [-a/2, (a + 1)/2]], arg**2/4)
    expr2 = besseli(a, arg)

    # Test that the two expressions agree for all arguments.
    for x_ in [0.5, 1.5]:
        for k_ in [0.0, 0.1, 0.3, 0.5, 0.8, 1, 5.751, 15.3]:
            assert abs((expr1 - expr2).n(subs={x: x_, k: k_})) < 1e-10
            assert abs((expr1 - expr2).n(subs={x: x_, k: -k_})) < 1e-10

    # Test continuity independently
    eps = 1e-13
    expr2 = expr1.subs(k, l)
    for x_ in [0.5, 1.5]:
        for k_ in [0.5, Rational(1, 3), 0.25, 0.75, Rational(2, 3), 1.0, 1.5]:
            assert abs((expr1 - expr2).n(
                       subs={x: x_, k: k_ + eps, l: k_ - eps})) < 1e-10
            assert abs((expr1 - expr2).n(
                       subs={x: x_, k: -k_ + eps, l: -k_ - eps})) < 1e-10

    expr = (meijerg(((0.5,), ()), ((0.5, 0, 0.5), ()), exp_polar(-I*pi)/4)
            + meijerg(((0.5,), ()), ((0.5, 0, 0.5), ()), exp_polar(I*pi)/4)) \
        /(2*sqrt(pi))
    assert (expr - pi/exp(1)).n(chop=True) == 0


def test_limits():
    k, x = symbols('k, x')
    assert hyper((1,), (Rational(4, 3), Rational(5, 3)), k**2).series(k) == \
           1 + 9*k**2/20 + 81*k**4/1120 + O(k**6) # issue 6350

    # https://github.com/sympy/sympy/issues/11465
    assert limit(1/hyper((1, ), (1, ), x), x, 0) == 1


def test_appellf1():
    a, b1, b2, c, x, y = symbols('a b1 b2 c x y')
    assert appellf1(a, b2, b1, c, y, x) == appellf1(a, b1, b2, c, x, y)
    assert appellf1(a, b1, b1, c, y, x) == appellf1(a, b1, b1, c, x, y)
    assert appellf1(a, b1, b2, c, S.Zero, S.Zero) is S.One

    f = appellf1(a, b1, b2, c, S.Zero, S.Zero, evaluate=False)
    assert f.func is appellf1
    assert f.doit() is S.One


def test_derivative_appellf1():
    from sympy.core.function import diff
    a, b1, b2, c, x, y, z = symbols('a b1 b2 c x y z')
    assert diff(appellf1(a, b1, b2, c, x, y), x) == a*b1*appellf1(a + 1, b2, b1 + 1, c + 1, y, x)/c
    assert diff(appellf1(a, b1, b2, c, x, y), y) == a*b2*appellf1(a + 1, b1, b2 + 1, c + 1, x, y)/c
    assert diff(appellf1(a, b1, b2, c, x, y), z) == 0
    assert diff(appellf1(a, b1, b2, c, x, y), a) ==  Derivative(appellf1(a, b1, b2, c, x, y), a)


def test_eval_nseries():
    a1, b1, a2, b2 = symbols('a1 b1 a2 b2')
    assert hyper((1,2), (1,2,3), x**2)._eval_nseries(x, 7, None) == \
        1 + x**2/3 + x**4/24 + x**6/360 + O(x**7)
    assert exp(x)._eval_nseries(x,7,None) == \
        hyper((a1, b1), (a1, b1), x)._eval_nseries(x, 7, None)
    assert hyper((a1, a2), (b1, b2), x)._eval_nseries(z, 7, None) ==\
        hyper((a1, a2), (b1, b2), x) + O(z**7)
    assert hyper((-S(1)/2, S(1)/2), (1,), 4*x/(x + 1)).nseries(x) == \
        1 - x + x**2/4 - 3*x**3/4 - 15*x**4/64 - 93*x**5/64 + O(x**6)
    assert (pi/2*hyper((-S(1)/2, S(1)/2), (1,), 4*x/(x + 1))).nseries(x) == \
        pi/2 - pi*x/2 + pi*x**2/8 - 3*pi*x**3/8 - 15*pi*x**4/128 - 93*pi*x**5/128 + O(x**6)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\aggregate\test_numba.py ===
import numpy as np
import pytest

from pandas.compat import is_platform_arm
from pandas.errors import NumbaUtilError

from pandas import (
    DataFrame,
    Index,
    NamedAgg,
    Series,
    option_context,
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


def test_correct_function_signature():
    pytest.importorskip("numba")

    def incorrect_function(x):
        return sum(x) * 2.7

    data = DataFrame(
        {"key": ["a", "a", "b", "b", "a"], "data": [1.0, 2.0, 3.0, 4.0, 5.0]},
        columns=["key", "data"],
    )
    with pytest.raises(NumbaUtilError, match="The first 2"):
        data.groupby("key").agg(incorrect_function, engine="numba")

    with pytest.raises(NumbaUtilError, match="The first 2"):
        data.groupby("key")["data"].agg(incorrect_function, engine="numba")


def test_check_nopython_kwargs():
    pytest.importorskip("numba")

    def incorrect_function(values, index):
        return sum(values) * 2.7

    data = DataFrame(
        {"key": ["a", "a", "b", "b", "a"], "data": [1.0, 2.0, 3.0, 4.0, 5.0]},
        columns=["key", "data"],
    )
    with pytest.raises(NumbaUtilError, match="numba does not support"):
        data.groupby("key").agg(incorrect_function, engine="numba", a=1)

    with pytest.raises(NumbaUtilError, match="numba does not support"):
        data.groupby("key")["data"].agg(incorrect_function, engine="numba", a=1)


@pytest.mark.filterwarnings("ignore")
# Filter warnings when parallel=True and the function can't be parallelized by Numba
@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("pandas_obj", ["Series", "DataFrame"])
@pytest.mark.parametrize("as_index", [True, False])
def test_numba_vs_cython(jit, pandas_obj, nogil, parallel, nopython, as_index):
    pytest.importorskip("numba")

    def func_numba(values, index):
        return np.mean(values) * 2.7

    if jit:
        # Test accepted jitted functions
        import numba

        func_numba = numba.jit(func_numba)

    data = DataFrame(
        {0: ["a", "a", "b", "b", "a"], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1]
    )
    engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
    grouped = data.groupby(0, as_index=as_index)
    if pandas_obj == "Series":
        grouped = grouped[1]

    result = grouped.agg(func_numba, engine="numba", engine_kwargs=engine_kwargs)
    expected = grouped.agg(lambda x: np.mean(x) * 2.7, engine="cython")

    tm.assert_equal(result, expected)


@pytest.mark.filterwarnings("ignore")
# Filter warnings when parallel=True and the function can't be parallelized by Numba
@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("pandas_obj", ["Series", "DataFrame"])
def test_cache(jit, pandas_obj, nogil, parallel, nopython):
    # Test that the functions are cached correctly if we switch functions
    pytest.importorskip("numba")

    def func_1(values, index):
        return np.mean(values) - 3.4

    def func_2(values, index):
        return np.mean(values) * 2.7

    if jit:
        import numba

        func_1 = numba.jit(func_1)
        func_2 = numba.jit(func_2)

    data = DataFrame(
        {0: ["a", "a", "b", "b", "a"], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1]
    )
    engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
    grouped = data.groupby(0)
    if pandas_obj == "Series":
        grouped = grouped[1]

    result = grouped.agg(func_1, engine="numba", engine_kwargs=engine_kwargs)
    expected = grouped.agg(lambda x: np.mean(x) - 3.4, engine="cython")
    tm.assert_equal(result, expected)

    # Add func_2 to the cache
    result = grouped.agg(func_2, engine="numba", engine_kwargs=engine_kwargs)
    expected = grouped.agg(lambda x: np.mean(x) * 2.7, engine="cython")
    tm.assert_equal(result, expected)

    # Retest func_1 which should use the cache
    result = grouped.agg(func_1, engine="numba", engine_kwargs=engine_kwargs)
    expected = grouped.agg(lambda x: np.mean(x) - 3.4, engine="cython")
    tm.assert_equal(result, expected)


def test_use_global_config():
    pytest.importorskip("numba")

    def func_1(values, index):
        return np.mean(values) - 3.4

    data = DataFrame(
        {0: ["a", "a", "b", "b", "a"], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1]
    )
    grouped = data.groupby(0)
    expected = grouped.agg(func_1, engine="numba")
    with option_context("compute.use_numba", True):
        result = grouped.agg(func_1, engine=None)
    tm.assert_frame_equal(expected, result)


@pytest.mark.parametrize(
    "agg_kwargs",
    [
        {"func": ["min", "max"]},
        {"func": "min"},
        {"func": {1: ["min", "max"], 2: "sum"}},
        {"bmin": NamedAgg(column=1, aggfunc="min")},
    ],
)
def test_multifunc_numba_vs_cython_frame(agg_kwargs):
    pytest.importorskip("numba")
    data = DataFrame(
        {
            0: ["a", "a", "b", "b", "a"],
            1: [1.0, 2.0, 3.0, 4.0, 5.0],
            2: [1, 2, 3, 4, 5],
        },
        columns=[0, 1, 2],
    )
    grouped = data.groupby(0)
    result = grouped.agg(**agg_kwargs, engine="numba")
    expected = grouped.agg(**agg_kwargs, engine="cython")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "agg_kwargs,expected_func",
    [
        ({"func": lambda values, index: values.sum()}, "sum"),
        # FIXME
        pytest.param(
            {
                "func": [
                    lambda values, index: values.sum(),
                    lambda values, index: values.min(),
                ]
            },
            ["sum", "min"],
            marks=pytest.mark.xfail(
                reason="This doesn't work yet! Fails in nopython pipeline!"
            ),
        ),
    ],
)
def test_multifunc_numba_udf_frame(agg_kwargs, expected_func):
    pytest.importorskip("numba")
    data = DataFrame(
        {
            0: ["a", "a", "b", "b", "a"],
            1: [1.0, 2.0, 3.0, 4.0, 5.0],
            2: [1, 2, 3, 4, 5],
        },
        columns=[0, 1, 2],
    )
    grouped = data.groupby(0)
    result = grouped.agg(**agg_kwargs, engine="numba")
    expected = grouped.agg(expected_func, engine="cython")
    # check_dtype can be removed if GH 44952 is addressed
    # Currently, UDFs still always return float64 while reductions can preserve dtype
    tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize(
    "agg_kwargs",
    [{"func": ["min", "max"]}, {"func": "min"}, {"min_val": "min", "max_val": "max"}],
)
def test_multifunc_numba_vs_cython_series(agg_kwargs):
    pytest.importorskip("numba")
    labels = ["a", "a", "b", "b", "a"]
    data = Series([1.0, 2.0, 3.0, 4.0, 5.0])
    grouped = data.groupby(labels)
    agg_kwargs["engine"] = "numba"
    result = grouped.agg(**agg_kwargs)
    agg_kwargs["engine"] = "cython"
    expected = grouped.agg(**agg_kwargs)
    if isinstance(expected, DataFrame):
        tm.assert_frame_equal(result, expected)
    else:
        tm.assert_series_equal(result, expected)


@pytest.mark.single_cpu
@pytest.mark.parametrize(
    "data,agg_kwargs",
    [
        (Series([1.0, 2.0, 3.0, 4.0, 5.0]), {"func": ["min", "max"]}),
        (Series([1.0, 2.0, 3.0, 4.0, 5.0]), {"func": "min"}),
        (
            DataFrame(
                {1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[1, 2]
            ),
            {"func": ["min", "max"]},
        ),
        (
            DataFrame(
                {1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[1, 2]
            ),
            {"func": "min"},
        ),
        (
            DataFrame(
                {1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[1, 2]
            ),
            {"func": {1: ["min", "max"], 2: "sum"}},
        ),
        (
            DataFrame(
                {1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[1, 2]
            ),
            {"min_col": NamedAgg(column=1, aggfunc="min")},
        ),
    ],
)
def test_multifunc_numba_kwarg_propagation(data, agg_kwargs):
    pytest.importorskip("numba")
    labels = ["a", "a", "b", "b", "a"]
    grouped = data.groupby(labels)
    result = grouped.agg(**agg_kwargs, engine="numba", engine_kwargs={"parallel": True})
    expected = grouped.agg(**agg_kwargs, engine="numba")
    if isinstance(expected, DataFrame):
        tm.assert_frame_equal(result, expected)
    else:
        tm.assert_series_equal(result, expected)


def test_args_not_cached():
    # GH 41647
    pytest.importorskip("numba")

    def sum_last(values, index, n):
        return values[-n:].sum()

    df = DataFrame({"id": [0, 0, 1, 1], "x": [1, 1, 1, 1]})
    grouped_x = df.groupby("id")["x"]
    result = grouped_x.agg(sum_last, 1, engine="numba")
    expected = Series([1.0] * 2, name="x", index=Index([0, 1], name="id"))
    tm.assert_series_equal(result, expected)

    result = grouped_x.agg(sum_last, 2, engine="numba")
    expected = Series([2.0] * 2, name="x", index=Index([0, 1], name="id"))
    tm.assert_series_equal(result, expected)


def test_index_data_correctly_passed():
    # GH 43133
    pytest.importorskip("numba")

    def f(values, index):
        return np.mean(index)

    df = DataFrame({"group": ["A", "A", "B"], "v": [4, 5, 6]}, index=[-1, -2, -3])
    result = df.groupby("group").aggregate(f, engine="numba")
    expected = DataFrame(
        [-1.5, -3.0], columns=["v"], index=Index(["A", "B"], name="group")
    )
    tm.assert_frame_equal(result, expected)


def test_engine_kwargs_not_cached():
    # If the user passes a different set of engine_kwargs don't return the same
    # jitted function
    pytest.importorskip("numba")
    nogil = True
    parallel = False
    nopython = True

    def func_kwargs(values, index):
        return nogil + parallel + nopython

    engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}
    df = DataFrame({"value": [0, 0, 0]})
    result = df.groupby(level=0).aggregate(
        func_kwargs, engine="numba", engine_kwargs=engine_kwargs
    )
    expected = DataFrame({"value": [2.0, 2.0, 2.0]})
    tm.assert_frame_equal(result, expected)

    nogil = False
    engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}
    result = df.groupby(level=0).aggregate(
        func_kwargs, engine="numba", engine_kwargs=engine_kwargs
    )
    expected = DataFrame({"value": [1.0, 1.0, 1.0]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.filterwarnings("ignore")
def test_multiindex_one_key(nogil, parallel, nopython):
    pytest.importorskip("numba")

    def numba_func(values, index):
        return 1

    df = DataFrame([{"A": 1, "B": 2, "C": 3}]).set_index(["A", "B"])
    engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}
    result = df.groupby("A").agg(
        numba_func, engine="numba", engine_kwargs=engine_kwargs
    )
    expected = DataFrame([1.0], index=Index([1], name="A"), columns=["C"])
    tm.assert_frame_equal(result, expected)


def test_multiindex_multi_key_not_supported(nogil, parallel, nopython):
    pytest.importorskip("numba")

    def numba_func(values, index):
        return 1

    df = DataFrame([{"A": 1, "B": 2, "C": 3}]).set_index(["A", "B"])
    engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}
    with pytest.raises(NotImplementedError, match="more than 1 grouping labels"):
        df.groupby(["A", "B"]).agg(
            numba_func, engine="numba", engine_kwargs=engine_kwargs
        )


def test_multilabel_numba_vs_cython(numba_supported_reductions):
    pytest.importorskip("numba")
    reduction, kwargs = numba_supported_reductions
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
            "C": np.random.default_rng(2).standard_normal(8),
            "D": np.random.default_rng(2).standard_normal(8),
        }
    )
    gb = df.groupby(["A", "B"])
    res_agg = gb.agg(reduction, engine="numba", **kwargs)
    expected_agg = gb.agg(reduction, engine="cython", **kwargs)
    tm.assert_frame_equal(res_agg, expected_agg)
    # Test that calling the aggregation directly also works
    direct_res = getattr(gb, reduction)(engine="numba", **kwargs)
    direct_expected = getattr(gb, reduction)(engine="cython", **kwargs)
    tm.assert_frame_equal(direct_res, direct_expected)


def test_multilabel_udf_numba_vs_cython():
    pytest.importorskip("numba")
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
            "C": np.random.default_rng(2).standard_normal(8),
            "D": np.random.default_rng(2).standard_normal(8),
        }
    )
    gb = df.groupby(["A", "B"])
    result = gb.agg(lambda values, index: values.min(), engine="numba")
    expected = gb.agg(lambda x: x.min(), engine="cython")
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\methods\test_tz_localize.py ===
from datetime import (
    datetime,
    timedelta,
)

import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz

from pandas import (
    DatetimeIndex,
    Timestamp,
    bdate_range,
    date_range,
    offsets,
    to_datetime,
)
import pandas._testing as tm

try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Cannot assign to a type  [misc]
    ZoneInfo = None  # type: ignore[misc, assignment]


easts = [pytz.timezone("US/Eastern"), gettz("US/Eastern")]
if ZoneInfo is not None:
    try:
        tz = ZoneInfo("US/Eastern")
    except KeyError:
        # no tzdata
        pass
    else:
        easts.append(tz)


class TestTZLocalize:
    def test_tz_localize_invalidates_freq(self):
        # we only preserve freq in unambiguous cases

        # if localized to US/Eastern, this crosses a DST transition
        dti = date_range("2014-03-08 23:00", "2014-03-09 09:00", freq="h")
        assert dti.freq == "h"

        result = dti.tz_localize(None)  # no-op
        assert result.freq == "h"

        result = dti.tz_localize("UTC")  # unambiguous freq preservation
        assert result.freq == "h"

        result = dti.tz_localize("US/Eastern", nonexistent="shift_forward")
        assert result.freq is None
        assert result.inferred_freq is None  # i.e. we are not _too_ strict here

        # Case where we _can_ keep freq because we're length==1
        dti2 = dti[:1]
        result = dti2.tz_localize("US/Eastern")
        assert result.freq == "h"

    def test_tz_localize_utc_copies(self, utc_fixture):
        # GH#46460
        times = ["2015-03-08 01:00", "2015-03-08 02:00", "2015-03-08 03:00"]
        index = DatetimeIndex(times)

        res = index.tz_localize(utc_fixture)
        assert not tm.shares_memory(res, index)

        res2 = index._data.tz_localize(utc_fixture)
        assert not tm.shares_memory(index._data, res2)

    def test_dti_tz_localize_nonexistent_raise_coerce(self):
        # GH#13057
        times = ["2015-03-08 01:00", "2015-03-08 02:00", "2015-03-08 03:00"]
        index = DatetimeIndex(times)
        tz = "US/Eastern"
        with pytest.raises(pytz.NonExistentTimeError, match="|".join(times)):
            index.tz_localize(tz=tz)

        with pytest.raises(pytz.NonExistentTimeError, match="|".join(times)):
            index.tz_localize(tz=tz, nonexistent="raise")

        result = index.tz_localize(tz=tz, nonexistent="NaT")
        test_times = ["2015-03-08 01:00-05:00", "NaT", "2015-03-08 03:00-04:00"]
        dti = to_datetime(test_times, utc=True)
        expected = dti.tz_convert("US/Eastern")
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("tz", easts)
    def test_dti_tz_localize_ambiguous_infer(self, tz):
        # November 6, 2011, fall back, repeat 2 AM hour
        # With no repeated hours, we cannot infer the transition
        dr = date_range(datetime(2011, 11, 6, 0), periods=5, freq=offsets.Hour())
        with pytest.raises(pytz.AmbiguousTimeError, match="Cannot infer dst time"):
            dr.tz_localize(tz)

    @pytest.mark.parametrize("tz", easts)
    def test_dti_tz_localize_ambiguous_infer2(self, tz, unit):
        # With repeated hours, we can infer the transition
        dr = date_range(
            datetime(2011, 11, 6, 0), periods=5, freq=offsets.Hour(), tz=tz, unit=unit
        )
        times = [
            "11/06/2011 00:00",
            "11/06/2011 01:00",
            "11/06/2011 01:00",
            "11/06/2011 02:00",
            "11/06/2011 03:00",
        ]
        di = DatetimeIndex(times).as_unit(unit)
        result = di.tz_localize(tz, ambiguous="infer")
        expected = dr._with_freq(None)
        tm.assert_index_equal(result, expected)
        result2 = DatetimeIndex(times, tz=tz, ambiguous="infer").as_unit(unit)
        tm.assert_index_equal(result2, expected)

    @pytest.mark.parametrize("tz", easts)
    def test_dti_tz_localize_ambiguous_infer3(self, tz):
        # When there is no dst transition, nothing special happens
        dr = date_range(datetime(2011, 6, 1, 0), periods=10, freq=offsets.Hour())
        localized = dr.tz_localize(tz)
        localized_infer = dr.tz_localize(tz, ambiguous="infer")
        tm.assert_index_equal(localized, localized_infer)

    @pytest.mark.parametrize("tz", easts)
    def test_dti_tz_localize_ambiguous_times(self, tz):
        # March 13, 2011, spring forward, skip from 2 AM to 3 AM
        dr = date_range(datetime(2011, 3, 13, 1, 30), periods=3, freq=offsets.Hour())
        with pytest.raises(pytz.NonExistentTimeError, match="2011-03-13 02:30:00"):
            dr.tz_localize(tz)

        # after dst transition, it works
        dr = date_range(
            datetime(2011, 3, 13, 3, 30), periods=3, freq=offsets.Hour(), tz=tz
        )

        # November 6, 2011, fall back, repeat 2 AM hour
        dr = date_range(datetime(2011, 11, 6, 1, 30), periods=3, freq=offsets.Hour())
        with pytest.raises(pytz.AmbiguousTimeError, match="Cannot infer dst time"):
            dr.tz_localize(tz)

        # UTC is OK
        dr = date_range(
            datetime(2011, 3, 13), periods=48, freq=offsets.Minute(30), tz=pytz.utc
        )

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_dti_tz_localize_pass_dates_to_utc(self, tzstr):
        strdates = ["1/1/2012", "3/1/2012", "4/1/2012"]

        idx = DatetimeIndex(strdates)
        conv = idx.tz_localize(tzstr)

        fromdates = DatetimeIndex(strdates, tz=tzstr)

        assert conv.tz == fromdates.tz
        tm.assert_numpy_array_equal(conv.values, fromdates.values)

    @pytest.mark.parametrize("prefix", ["", "dateutil/"])
    def test_dti_tz_localize(self, prefix):
        tzstr = prefix + "US/Eastern"
        dti = date_range(start="1/1/2005", end="1/1/2005 0:00:30.256", freq="ms")
        dti2 = dti.tz_localize(tzstr)

        dti_utc = date_range(
            start="1/1/2005 05:00", end="1/1/2005 5:00:30.256", freq="ms", tz="utc"
        )

        tm.assert_numpy_array_equal(dti2.values, dti_utc.values)

        dti3 = dti2.tz_convert(prefix + "US/Pacific")
        tm.assert_numpy_array_equal(dti3.values, dti_utc.values)

        dti = date_range(start="11/6/2011 1:59", end="11/6/2011 2:00", freq="ms")
        with pytest.raises(pytz.AmbiguousTimeError, match="Cannot infer dst time"):
            dti.tz_localize(tzstr)

        dti = date_range(start="3/13/2011 1:59", end="3/13/2011 2:00", freq="ms")
        with pytest.raises(pytz.NonExistentTimeError, match="2011-03-13 02:00:00"):
            dti.tz_localize(tzstr)

    @pytest.mark.parametrize(
        "tz",
        [
            "US/Eastern",
            "dateutil/US/Eastern",
            pytz.timezone("US/Eastern"),
            gettz("US/Eastern"),
        ],
    )
    def test_dti_tz_localize_utc_conversion(self, tz):
        # Localizing to time zone should:
        #  1) check for DST ambiguities
        #  2) convert to UTC

        rng = date_range("3/10/2012", "3/11/2012", freq="30min")

        converted = rng.tz_localize(tz)
        expected_naive = rng + offsets.Hour(5)
        tm.assert_numpy_array_equal(converted.asi8, expected_naive.asi8)

        # DST ambiguity, this should fail
        rng = date_range("3/11/2012", "3/12/2012", freq="30min")
        # Is this really how it should fail??
        with pytest.raises(pytz.NonExistentTimeError, match="2012-03-11 02:00:00"):
            rng.tz_localize(tz)

    def test_dti_tz_localize_roundtrip(self, tz_aware_fixture):
        # note: this tz tests that a tz-naive index can be localized
        # and de-localized successfully, when there are no DST transitions
        # in the range.
        idx = date_range(start="2014-06-01", end="2014-08-30", freq="15min")
        tz = tz_aware_fixture
        localized = idx.tz_localize(tz)
        # can't localize a tz-aware object
        with pytest.raises(
            TypeError, match="Already tz-aware, use tz_convert to convert"
        ):
            localized.tz_localize(tz)
        reset = localized.tz_localize(None)
        assert reset.tzinfo is None
        expected = idx._with_freq(None)
        tm.assert_index_equal(reset, expected)

    def test_dti_tz_localize_naive(self):
        rng = date_range("1/1/2011", periods=100, freq="h")

        conv = rng.tz_localize("US/Pacific")
        exp = date_range("1/1/2011", periods=100, freq="h", tz="US/Pacific")

        tm.assert_index_equal(conv, exp._with_freq(None))

    def test_dti_tz_localize_tzlocal(self):
        # GH#13583
        offset = dateutil.tz.tzlocal().utcoffset(datetime(2011, 1, 1))
        offset = int(offset.total_seconds() * 1000000000)

        dti = date_range(start="2001-01-01", end="2001-03-01")
        dti2 = dti.tz_localize(dateutil.tz.tzlocal())
        tm.assert_numpy_array_equal(dti2.asi8 + offset, dti.asi8)

        dti = date_range(start="2001-01-01", end="2001-03-01", tz=dateutil.tz.tzlocal())
        dti2 = dti.tz_localize(None)
        tm.assert_numpy_array_equal(dti2.asi8 - offset, dti.asi8)

    @pytest.mark.parametrize("tz", easts)
    def test_dti_tz_localize_ambiguous_nat(self, tz):
        times = [
            "11/06/2011 00:00",
            "11/06/2011 01:00",
            "11/06/2011 01:00",
            "11/06/2011 02:00",
            "11/06/2011 03:00",
        ]
        di = DatetimeIndex(times)
        localized = di.tz_localize(tz, ambiguous="NaT")

        times = [
            "11/06/2011 00:00",
            np.nan,
            np.nan,
            "11/06/2011 02:00",
            "11/06/2011 03:00",
        ]
        di_test = DatetimeIndex(times, tz="US/Eastern")

        # left dtype is datetime64[ns, US/Eastern]
        # right is datetime64[ns, tzfile('/usr/share/zoneinfo/US/Eastern')]
        tm.assert_numpy_array_equal(di_test.values, localized.values)

    @pytest.mark.parametrize("tz", easts)
    def test_dti_tz_localize_ambiguous_flags(self, tz, unit):
        # November 6, 2011, fall back, repeat 2 AM hour

        # Pass in flags to determine right dst transition
        dr = date_range(
            datetime(2011, 11, 6, 0), periods=5, freq=offsets.Hour(), tz=tz, unit=unit
        )
        times = [
            "11/06/2011 00:00",
            "11/06/2011 01:00",
            "11/06/2011 01:00",
            "11/06/2011 02:00",
            "11/06/2011 03:00",
        ]

        # Test tz_localize
        di = DatetimeIndex(times).as_unit(unit)
        is_dst = [1, 1, 0, 0, 0]
        localized = di.tz_localize(tz, ambiguous=is_dst)
        expected = dr._with_freq(None)
        tm.assert_index_equal(expected, localized)

        result = DatetimeIndex(times, tz=tz, ambiguous=is_dst).as_unit(unit)
        tm.assert_index_equal(result, expected)

        localized = di.tz_localize(tz, ambiguous=np.array(is_dst))
        tm.assert_index_equal(dr, localized)

        localized = di.tz_localize(tz, ambiguous=np.array(is_dst).astype("bool"))
        tm.assert_index_equal(dr, localized)

        # Test constructor
        localized = DatetimeIndex(times, tz=tz, ambiguous=is_dst).as_unit(unit)
        tm.assert_index_equal(dr, localized)

        # Test duplicate times where inferring the dst fails
        times += times
        di = DatetimeIndex(times).as_unit(unit)

        # When the sizes are incompatible, make sure error is raised
        msg = "Length of ambiguous bool-array must be the same size as vals"
        with pytest.raises(Exception, match=msg):
            di.tz_localize(tz, ambiguous=is_dst)

        # When sizes are compatible and there are repeats ('infer' won't work)
        is_dst = np.hstack((is_dst, is_dst))
        localized = di.tz_localize(tz, ambiguous=is_dst)
        dr = dr.append(dr)
        tm.assert_index_equal(dr, localized)

    @pytest.mark.parametrize("tz", easts)
    def test_dti_tz_localize_ambiguous_flags2(self, tz, unit):
        # When there is no dst transition, nothing special happens
        dr = date_range(datetime(2011, 6, 1, 0), periods=10, freq=offsets.Hour())
        is_dst = np.array([1] * 10)
        localized = dr.tz_localize(tz)
        localized_is_dst = dr.tz_localize(tz, ambiguous=is_dst)
        tm.assert_index_equal(localized, localized_is_dst)

    def test_dti_tz_localize_bdate_range(self):
        dr = bdate_range("1/1/2009", "1/1/2010")
        dr_utc = bdate_range("1/1/2009", "1/1/2010", tz=pytz.utc)
        localized = dr.tz_localize(pytz.utc)
        tm.assert_index_equal(dr_utc, localized)

    @pytest.mark.parametrize(
        "start_ts, tz, end_ts, shift",
        [
            ["2015-03-29 02:20:00", "Europe/Warsaw", "2015-03-29 03:00:00", "forward"],
            [
                "2015-03-29 02:20:00",
                "Europe/Warsaw",
                "2015-03-29 01:59:59.999999999",
                "backward",
            ],
            [
                "2015-03-29 02:20:00",
                "Europe/Warsaw",
                "2015-03-29 03:20:00",
                timedelta(hours=1),
            ],
            [
                "2015-03-29 02:20:00",
                "Europe/Warsaw",
                "2015-03-29 01:20:00",
                timedelta(hours=-1),
            ],
            ["2018-03-11 02:33:00", "US/Pacific", "2018-03-11 03:00:00", "forward"],
            [
                "2018-03-11 02:33:00",
                "US/Pacific",
                "2018-03-11 01:59:59.999999999",
                "backward",
            ],
            [
                "2018-03-11 02:33:00",
                "US/Pacific",
                "2018-03-11 03:33:00",
                timedelta(hours=1),
            ],
            [
                "2018-03-11 02:33:00",
                "US/Pacific",
                "2018-03-11 01:33:00",
                timedelta(hours=-1),
            ],
        ],
    )
    @pytest.mark.parametrize("tz_type", ["", "dateutil/"])
    def test_dti_tz_localize_nonexistent_shift(
        self, start_ts, tz, end_ts, shift, tz_type, unit
    ):
        # GH#8917
        tz = tz_type + tz
        if isinstance(shift, str):
            shift = "shift_" + shift
        dti = DatetimeIndex([Timestamp(start_ts)]).as_unit(unit)
        result = dti.tz_localize(tz, nonexistent=shift)
        expected = DatetimeIndex([Timestamp(end_ts)]).tz_localize(tz).as_unit(unit)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("offset", [-1, 1])
    def test_dti_tz_localize_nonexistent_shift_invalid(self, offset, warsaw):
        # GH#8917
        tz = warsaw
        dti = DatetimeIndex([Timestamp("2015-03-29 02:20:00")])
        msg = "The provided timedelta will relocalize on a nonexistent time"
        with pytest.raises(ValueError, match=msg):
            dti.tz_localize(tz, nonexistent=timedelta(seconds=offset))