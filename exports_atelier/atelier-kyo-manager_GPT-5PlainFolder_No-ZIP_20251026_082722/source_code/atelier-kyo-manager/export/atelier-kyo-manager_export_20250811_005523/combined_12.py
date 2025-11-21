
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\tests\test_mixins.py ===
import numbers
import operator

import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises

# NOTE: This class should be kept as an exact copy of the example from the
# docstring for NDArrayOperatorsMixin.

class ArrayLike(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, value):
        self.value = np.asarray(value)

    # One might also consider adding the built-in list type to this
    # list, to support operations like np.add(array_like, list)
    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get('out', ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use ArrayLike instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle ArrayLike objects.
            if not isinstance(x, self._HANDLED_TYPES + (ArrayLike,)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x.value if isinstance(x, ArrayLike) else x
                       for x in inputs)
        if out:
            kwargs['out'] = tuple(
                x.value if isinstance(x, ArrayLike) else x
                for x in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(type(self)(x) for x in result)
        elif method == 'at':
            # no return value
            return None
        else:
            # one return value
            return type(self)(result)

    def __repr__(self):
        return f'{type(self).__name__}({self.value!r})'


def wrap_array_like(result):
    if type(result) is tuple:
        return tuple(ArrayLike(r) for r in result)
    else:
        return ArrayLike(result)


def _assert_equal_type_and_value(result, expected, err_msg=None):
    assert_equal(type(result), type(expected), err_msg=err_msg)
    if isinstance(result, tuple):
        assert_equal(len(result), len(expected), err_msg=err_msg)
        for result_item, expected_item in zip(result, expected):
            _assert_equal_type_and_value(result_item, expected_item, err_msg)
    else:
        assert_equal(result.value, expected.value, err_msg=err_msg)
        assert_equal(getattr(result.value, 'dtype', None),
                     getattr(expected.value, 'dtype', None), err_msg=err_msg)


_ALL_BINARY_OPERATORS = [
    operator.lt,
    operator.le,
    operator.eq,
    operator.ne,
    operator.gt,
    operator.ge,
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.mod,
    divmod,
    pow,
    operator.lshift,
    operator.rshift,
    operator.and_,
    operator.xor,
    operator.or_,
]


class TestNDArrayOperatorsMixin:

    def test_array_like_add(self):

        def check(result):
            _assert_equal_type_and_value(result, ArrayLike(0))

        check(ArrayLike(0) + 0)
        check(0 + ArrayLike(0))

        check(ArrayLike(0) + np.array(0))
        check(np.array(0) + ArrayLike(0))

        check(ArrayLike(np.array(0)) + 0)
        check(0 + ArrayLike(np.array(0)))

        check(ArrayLike(np.array(0)) + np.array(0))
        check(np.array(0) + ArrayLike(np.array(0)))

    def test_inplace(self):
        array_like = ArrayLike(np.array([0]))
        array_like += 1
        _assert_equal_type_and_value(array_like, ArrayLike(np.array([1])))

        array = np.array([0])
        array += ArrayLike(1)
        _assert_equal_type_and_value(array, ArrayLike(np.array([1])))

    def test_opt_out(self):

        class OptOut:
            """Object that opts out of __array_ufunc__."""
            __array_ufunc__ = None

            def __add__(self, other):
                return self

            def __radd__(self, other):
                return self

        array_like = ArrayLike(1)
        opt_out = OptOut()

        # supported operations
        assert_(array_like + opt_out is opt_out)
        assert_(opt_out + array_like is opt_out)

        # not supported
        with assert_raises(TypeError):
            # don't use the Python default, array_like = array_like + opt_out
            array_like += opt_out
        with assert_raises(TypeError):
            array_like - opt_out
        with assert_raises(TypeError):
            opt_out - array_like

    def test_subclass(self):

        class SubArrayLike(ArrayLike):
            """Should take precedence over ArrayLike."""

        x = ArrayLike(0)
        y = SubArrayLike(1)
        _assert_equal_type_and_value(x + y, y)
        _assert_equal_type_and_value(y + x, y)

    def test_object(self):
        x = ArrayLike(0)
        obj = object()
        with assert_raises(TypeError):
            x + obj
        with assert_raises(TypeError):
            obj + x
        with assert_raises(TypeError):
            x += obj

    def test_unary_methods(self):
        array = np.array([-1, 0, 1, 2])
        array_like = ArrayLike(array)
        for op in [operator.neg,
                   operator.pos,
                   abs,
                   operator.invert]:
            _assert_equal_type_and_value(op(array_like), ArrayLike(op(array)))

    def test_forward_binary_methods(self):
        array = np.array([-1, 0, 1, 2])
        array_like = ArrayLike(array)
        for op in _ALL_BINARY_OPERATORS:
            expected = wrap_array_like(op(array, 1))
            actual = op(array_like, 1)
            err_msg = f'failed for operator {op}'
            _assert_equal_type_and_value(expected, actual, err_msg=err_msg)

    def test_reflected_binary_methods(self):
        for op in _ALL_BINARY_OPERATORS:
            expected = wrap_array_like(op(2, 1))
            actual = op(2, ArrayLike(1))
            err_msg = f'failed for operator {op}'
            _assert_equal_type_and_value(expected, actual, err_msg=err_msg)

    def test_matmul(self):
        array = np.array([1, 2], dtype=np.float64)
        array_like = ArrayLike(array)
        expected = ArrayLike(np.float64(5))
        _assert_equal_type_and_value(expected, np.matmul(array_like, array))
        _assert_equal_type_and_value(
            expected, operator.matmul(array_like, array))
        _assert_equal_type_and_value(
            expected, operator.matmul(array, array_like))

    def test_ufunc_at(self):
        array = ArrayLike(np.array([1, 2, 3, 4]))
        assert_(np.negative.at(array, np.array([0, 1])) is None)
        _assert_equal_type_and_value(array, ArrayLike([-1, -2, 3, 4]))

    def test_ufunc_two_outputs(self):
        mantissa, exponent = np.frexp(2 ** -3)
        expected = (ArrayLike(mantissa), ArrayLike(exponent))
        _assert_equal_type_and_value(
            np.frexp(ArrayLike(2 ** -3)), expected)
        _assert_equal_type_and_value(
            np.frexp(ArrayLike(np.array(2 ** -3))), expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\_ufunclike_impl.py ===
"""
Module of functions that are like ufuncs in acting on arrays and optionally
storing results in an output array.

"""
__all__ = ['fix', 'isneginf', 'isposinf']

import numpy._core.numeric as nx
from numpy._core.overrides import array_function_dispatch


def _dispatcher(x, out=None):
    return (x, out)


@array_function_dispatch(_dispatcher, verify=False, module='numpy')
def fix(x, out=None):
    """
    Round to nearest integer towards zero.

    Round an array of floats element-wise to nearest integer towards zero.
    The rounded values have the same data-type as the input.

    Parameters
    ----------
    x : array_like
        An array to be rounded
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have
        a shape that the input broadcasts to. If not provided or None, a
        freshly-allocated array is returned.

    Returns
    -------
    out : ndarray of floats
        An array with the same dimensions and data-type as the input.
        If second argument is not supplied then a new array is returned
        with the rounded values.

        If a second argument is supplied the result is stored there.
        The return value ``out`` is then a reference to that array.

    See Also
    --------
    rint, trunc, floor, ceil
    around : Round to given number of decimals

    Examples
    --------
    >>> import numpy as np
    >>> np.fix(3.14)
    3.0
    >>> np.fix(3)
    3
    >>> np.fix([2.1, 2.9, -2.1, -2.9])
    array([ 2.,  2., -2., -2.])

    """
    # promote back to an array if flattened
    res = nx.asanyarray(nx.ceil(x, out=out))
    res = nx.floor(x, out=res, where=nx.greater_equal(x, 0))

    # when no out argument is passed and no subclasses are involved, flatten
    # scalars
    if out is None and type(res) is nx.ndarray:
        res = res[()]
    return res


@array_function_dispatch(_dispatcher, verify=False, module='numpy')
def isposinf(x, out=None):
    """
    Test element-wise for positive infinity, return result as bool array.

    Parameters
    ----------
    x : array_like
        The input array.
    out : array_like, optional
        A location into which the result is stored. If provided, it must have a
        shape that the input broadcasts to. If not provided or None, a
        freshly-allocated boolean array is returned.

    Returns
    -------
    out : ndarray
        A boolean array with the same dimensions as the input.
        If second argument is not supplied then a boolean array is returned
        with values True where the corresponding element of the input is
        positive infinity and values False where the element of the input is
        not positive infinity.

        If a second argument is supplied the result is stored there. If the
        type of that array is a numeric type the result is represented as zeros
        and ones, if the type is boolean then as False and True.
        The return value `out` is then a reference to that array.

    See Also
    --------
    isinf, isneginf, isfinite, isnan

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754).

    Errors result if the second argument is also supplied when x is a scalar
    input, if first and second arguments have different shapes, or if the
    first argument has complex values

    Examples
    --------
    >>> import numpy as np
    >>> np.isposinf(np.inf)
    True
    >>> np.isposinf(-np.inf)
    False
    >>> np.isposinf([-np.inf, 0., np.inf])
    array([False, False,  True])

    >>> x = np.array([-np.inf, 0., np.inf])
    >>> y = np.array([2, 2, 2])
    >>> np.isposinf(x, y)
    array([0, 0, 1])
    >>> y
    array([0, 0, 1])

    """
    is_inf = nx.isinf(x)
    try:
        signbit = ~nx.signbit(x)
    except TypeError as e:
        dtype = nx.asanyarray(x).dtype
        raise TypeError(f'This operation is not supported for {dtype} values '
                        'because it would be ambiguous.') from e
    else:
        return nx.logical_and(is_inf, signbit, out)


@array_function_dispatch(_dispatcher, verify=False, module='numpy')
def isneginf(x, out=None):
    """
    Test element-wise for negative infinity, return result as bool array.

    Parameters
    ----------
    x : array_like
        The input array.
    out : array_like, optional
        A location into which the result is stored. If provided, it must have a
        shape that the input broadcasts to. If not provided or None, a
        freshly-allocated boolean array is returned.

    Returns
    -------
    out : ndarray
        A boolean array with the same dimensions as the input.
        If second argument is not supplied then a numpy boolean array is
        returned with values True where the corresponding element of the
        input is negative infinity and values False where the element of
        the input is not negative infinity.

        If a second argument is supplied the result is stored there. If the
        type of that array is a numeric type the result is represented as
        zeros and ones, if the type is boolean then as False and True. The
        return value `out` is then a reference to that array.

    See Also
    --------
    isinf, isposinf, isnan, isfinite

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754).

    Errors result if the second argument is also supplied when x is a scalar
    input, if first and second arguments have different shapes, or if the
    first argument has complex values.

    Examples
    --------
    >>> import numpy as np
    >>> np.isneginf(-np.inf)
    True
    >>> np.isneginf(np.inf)
    False
    >>> np.isneginf([-np.inf, 0., np.inf])
    array([ True, False, False])

    >>> x = np.array([-np.inf, 0., np.inf])
    >>> y = np.array([2, 2, 2])
    >>> np.isneginf(x, y)
    array([1, 0, 0])
    >>> y
    array([1, 0, 0])

    """
    is_inf = nx.isinf(x)
    try:
        signbit = nx.signbit(x)
    except TypeError as e:
        dtype = nx.asanyarray(x).dtype
        raise TypeError(f'This operation is not supported for {dtype} values '
                        'because it would be ambiguous.') from e
    else:
        return nx.logical_and(is_inf, signbit, out)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\series.py ===
"""
Data structure for 1-dimensional cross-sectional and time series data
"""
from __future__ import annotations

from collections.abc import (
    Hashable,
    Iterable,
    Mapping,
    Sequence,
)
import operator
import sys
from textwrap import dedent
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    cast,
    overload,
)
import warnings
import weakref

import numpy as np

from pandas._config import (
    using_copy_on_write,
    warn_copy_on_write,
)
from pandas._config.config import _get_option

from pandas._libs import (
    lib,
    properties,
    reshape,
)
from pandas._libs.lib import is_range_indexer
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
    ChainedAssignmentError,
    InvalidIndexError,
    _chained_assignment_method_msg,
    _chained_assignment_msg,
    _chained_assignment_warning_method_msg,
    _chained_assignment_warning_msg,
    _check_cacher,
)
from pandas.util._decorators import (
    Appender,
    Substitution,
    deprecate_nonkeyword_arguments,
    doc,
)
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
    validate_ascending,
    validate_bool_kwarg,
    validate_percentile,
)

from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import (
    LossySetitemError,
    construct_1d_arraylike_from_scalar,
    find_common_type,
    infer_dtype_from,
    maybe_box_native,
    maybe_cast_pointwise_result,
)
from pandas.core.dtypes.common import (
    is_dict_like,
    is_integer,
    is_iterator,
    is_list_like,
    is_object_dtype,
    is_scalar,
    pandas_dtype,
    validate_all_hashable,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    ExtensionDtype,
    SparseDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
from pandas.core.dtypes.inference import is_hashable
from pandas.core.dtypes.missing import (
    isna,
    na_value_for_dtype,
    notna,
    remove_na_arraylike,
)

from pandas.core import (
    algorithms,
    base,
    common as com,
    missing,
    nanops,
    ops,
    roperator,
)
from pandas.core.accessor import CachedAccessor
from pandas.core.apply import SeriesApply
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.arrow import (
    ListAccessor,
    StructAccessor,
)
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.arrays.sparse import SparseAccessor
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import (
    array as pd_array,
    extract_array,
    sanitize_array,
)
from pandas.core.generic import (
    NDFrame,
    make_doc,
)
from pandas.core.indexers import (
    disallow_ndim_indexing,
    unpack_1tuple,
)
from pandas.core.indexes.accessors import CombinedDatetimelikeProperties
from pandas.core.indexes.api import (
    DatetimeIndex,
    Index,
    MultiIndex,
    PeriodIndex,
    default_index,
    ensure_index,
)
import pandas.core.indexes.base as ibase
from pandas.core.indexes.multi import maybe_droplevels
from pandas.core.indexing import (
    check_bool_indexer,
    check_dict_or_set_indexers,
)
from pandas.core.internals import (
    SingleArrayManager,
    SingleBlockManager,
)
from pandas.core.methods import selectn
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import (
    ensure_key_mapped,
    nargsort,
)
from pandas.core.strings.accessor import StringMethods
from pandas.core.tools.datetimes import to_datetime

import pandas.io.formats.format as fmt
from pandas.io.formats.info import (
    INFO_DOCSTRING,
    SeriesInfo,
    series_sub_kwargs,
)
import pandas.plotting

if TYPE_CHECKING:
    from pandas._libs.internals import BlockValuesRefs
    from pandas._typing import (
        AggFuncType,
        AnyAll,
        AnyArrayLike,
        ArrayLike,
        Axis,
        AxisInt,
        CorrelationMethod,
        DropKeep,
        Dtype,
        DtypeObj,
        FilePath,
        Frequency,
        IgnoreRaise,
        IndexKeyFunc,
        IndexLabel,
        Level,
        MutableMappingT,
        NaPosition,
        NumpySorter,
        NumpyValueArrayLike,
        QuantileInterpolation,
        ReindexMethod,
        Renamer,
        Scalar,
        Self,
        SingleManager,
        SortKind,
        StorageOptions,
        Suffixes,
        ValueKeyFunc,
        WriteBuffer,
        npt,
    )

    from pandas.core.frame import DataFrame
    from pandas.core.groupby.generic import SeriesGroupBy

__all__ = ["Series"]

_shared_doc_kwargs = {
    "axes": "index",
    "klass": "Series",
    "axes_single_arg": "{0 or 'index'}",
    "axis": """axis : {0 or 'index'}
        Unused. Parameter needed for compatibility with DataFrame.""",
    "inplace": """inplace : bool, default False
        If True, performs operation inplace and returns None.""",
    "unique": "np.ndarray",
    "duplicated": "Series",
    "optional_by": "",
    "optional_reindex": """
index : array-like, optional
    New labels for the index. Preferably an Index object to avoid
    duplicating data.
axis : int or str, optional
    Unused.""",
}


def _coerce_method(converter):
    """
    Install the scalar coercion methods.
    """

    def wrapper(self):
        if len(self) == 1:
            warnings.warn(
                f"Calling {converter.__name__} on a single element Series is "
                "deprecated and will raise a TypeError in the future. "
                f"Use {converter.__name__}(ser.iloc[0]) instead",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
            return converter(self.iloc[0])
        raise TypeError(f"cannot convert the series to {converter}")

    wrapper.__name__ = f"__{converter.__name__}__"
    return wrapper


# ----------------------------------------------------------------------
# Series class


# error: Cannot override final attribute "ndim" (previously declared in base
# class "NDFrame")
# error: Cannot override final attribute "size" (previously declared in base
# class "NDFrame")
# definition in base class "NDFrame"
class Series(base.IndexOpsMixin, NDFrame):  # type: ignore[misc]
    """
    One-dimensional ndarray with axis labels (including time series).

    Labels need not be unique but must be a hashable type. The object
    supports both integer- and label-based indexing and provides a host of
    methods for performing operations involving the index. Statistical
    methods from ndarray have been overridden to automatically exclude
    missing data (currently represented as NaN).

    Operations between Series (+, -, /, \\*, \\*\\*) align values based on their
    associated index values-- they need not be the same length. The result
    index will be the sorted union of the two indexes.

    Parameters
    ----------
    data : array-like, Iterable, dict, or scalar value
        Contains data stored in Series. If data is a dict, argument order is
        maintained.
    index : array-like or Index (1d)
        Values must be hashable and have the same length as `data`.
        Non-unique index values are allowed. Will default to
        RangeIndex (0, 1, 2, ..., n) if not provided. If data is dict-like
        and index is None, then the keys in the data are used as the index. If the
        index is not None, the resulting Series is reindexed with the index values.
    dtype : str, numpy.dtype, or ExtensionDtype, optional
        Data type for the output Series. If not specified, this will be
        inferred from `data`.
        See the :ref:`user guide <basics.dtypes>` for more usages.
    name : Hashable, default None
        The name to give to the Series.
    copy : bool, default False
        Copy input data. Only affects Series or 1d ndarray input. See examples.

    Notes
    -----
    Please reference the :ref:`User Guide <basics.series>` for more information.

    Examples
    --------
    Constructing Series from a dictionary with an Index specified

    >>> d = {'a': 1, 'b': 2, 'c': 3}
    >>> ser = pd.Series(data=d, index=['a', 'b', 'c'])
    >>> ser
    a   1
    b   2
    c   3
    dtype: int64

    The keys of the dictionary match with the Index values, hence the Index
    values have no effect.

    >>> d = {'a': 1, 'b': 2, 'c': 3}
    >>> ser = pd.Series(data=d, index=['x', 'y', 'z'])
    >>> ser
    x   NaN
    y   NaN
    z   NaN
    dtype: float64

    Note that the Index is first build with the keys from the dictionary.
    After this the Series is reindexed with the given Index values, hence we
    get all NaN as a result.

    Constructing Series from a list with `copy=False`.

    >>> r = [1, 2]
    >>> ser = pd.Series(r, copy=False)
    >>> ser.iloc[0] = 999
    >>> r
    [1, 2]
    >>> ser
    0    999
    1      2
    dtype: int64

    Due to input data type the Series has a `copy` of
    the original data even though `copy=False`, so
    the data is unchanged.

    Constructing Series from a 1d ndarray with `copy=False`.

    >>> r = np.array([1, 2])
    >>> ser = pd.Series(r, copy=False)
    >>> ser.iloc[0] = 999
    >>> r
    array([999,   2])
    >>> ser
    0    999
    1      2
    dtype: int64

    Due to input data type the Series has a `view` on
    the original data, so
    the data is changed as well.
    """

    _typ = "series"
    _HANDLED_TYPES = (Index, ExtensionArray, np.ndarray)

    _name: Hashable
    _metadata: list[str] = ["_name"]
    _internal_names_set = {"index", "name"} | NDFrame._internal_names_set
    _accessors = {"dt", "cat", "str", "sparse"}
    _hidden_attrs = (
        base.IndexOpsMixin._hidden_attrs | NDFrame._hidden_attrs | frozenset([])
    )

    # similar to __array_priority__, positions Series after DataFrame
    #  but before Index and ExtensionArray.  Should NOT be overridden by subclasses.
    __pandas_priority__ = 3000

    # Override cache_readonly bc Series is mutable
    # error: Incompatible types in assignment (expression has type "property",
    # base class "IndexOpsMixin" defined the type as "Callable[[IndexOpsMixin], bool]")
    hasnans = property(  # type: ignore[assignment]
        # error: "Callable[[IndexOpsMixin], bool]" has no attribute "fget"
        base.IndexOpsMixin.hasnans.fget,  # type: ignore[attr-defined]
        doc=base.IndexOpsMixin.hasnans.__doc__,
    )
    _mgr: SingleManager

    # ----------------------------------------------------------------------
    # Constructors

    def __init__(
        self,
        data=None,
        index=None,
        dtype: Dtype | None = None,
        name=None,
        copy: bool | None = None,
        fastpath: bool | lib.NoDefault = lib.no_default,
    ) -> None:
        if fastpath is not lib.no_default:
            warnings.warn(
                "The 'fastpath' keyword in pd.Series is deprecated and will "
                "be removed in a future version.",
                DeprecationWarning,
                stacklevel=find_stack_level(),
            )
        else:
            fastpath = False

        allow_mgr = False
        if (
            isinstance(data, (SingleBlockManager, SingleArrayManager))
            and index is None
            and dtype is None
            and (copy is False or copy is None)
        ):
            if not allow_mgr:
                # GH#52419
                warnings.warn(
                    f"Passing a {type(data).__name__} to {type(self).__name__} "
                    "is deprecated and will raise in a future version. "
                    "Use public APIs instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            if using_copy_on_write():
                data = data.copy(deep=False)
            # GH#33357 called with just the SingleBlockManager
            NDFrame.__init__(self, data)
            if fastpath:
                # e.g. from _box_col_values, skip validation of name
                object.__setattr__(self, "_name", name)
            else:
                self.name = name
            return

        is_pandas_object = isinstance(data, (Series, Index, ExtensionArray))
        data_dtype = getattr(data, "dtype", None)
        original_dtype = dtype

        if isinstance(data, (ExtensionArray, np.ndarray)):
            if copy is not False and using_copy_on_write():
                if dtype is None or astype_is_view(data.dtype, pandas_dtype(dtype)):
                    data = data.copy()
        if copy is None:
            copy = False

        # we are called internally, so short-circuit
        if fastpath:
            # data is a ndarray, index is defined
            if not isinstance(data, (SingleBlockManager, SingleArrayManager)):
                manager = _get_option("mode.data_manager", silent=True)
                if manager == "block":
                    data = SingleBlockManager.from_array(data, index)
                elif manager == "array":
                    data = SingleArrayManager.from_array(data, index)
                allow_mgr = True
            elif using_copy_on_write() and not copy:
                data = data.copy(deep=False)

            if not allow_mgr:
                warnings.warn(
                    f"Passing a {type(data).__name__} to {type(self).__name__} "
                    "is deprecated and will raise in a future version. "
                    "Use public APIs instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            if copy:
                data = data.copy()
            # skips validation of the name
            object.__setattr__(self, "_name", name)
            NDFrame.__init__(self, data)
            return

        if isinstance(data, SingleBlockManager) and using_copy_on_write() and not copy:
            data = data.copy(deep=False)

            if not allow_mgr:
                warnings.warn(
                    f"Passing a {type(data).__name__} to {type(self).__name__} "
                    "is deprecated and will raise in a future version. "
                    "Use public APIs instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        name = ibase.maybe_extract_name(name, data, type(self))

        if index is not None:
            index = ensure_index(index)

        if dtype is not None:
            dtype = self._validate_dtype(dtype)

        if data is None:
            index = index if index is not None else default_index(0)
            if len(index) or dtype is not None:
                data = na_value_for_dtype(pandas_dtype(dtype), compat=False)
            else:
                data = []

        if isinstance(data, MultiIndex):
            raise NotImplementedError(
                "initializing a Series from a MultiIndex is not supported"
            )

        refs = None
        if isinstance(data, Index):
            if dtype is not None:
                data = data.astype(dtype, copy=False)

            if using_copy_on_write():
                refs = data._references
                data = data._values
            else:
                # GH#24096 we need to ensure the index remains immutable
                data = data._values.copy()
            copy = False

        elif isinstance(data, np.ndarray):
            if len(data.dtype):
                # GH#13296 we are dealing with a compound dtype, which
                #  should be treated as 2D
                raise ValueError(
                    "Cannot construct a Series from an ndarray with "
                    "compound dtype.  Use DataFrame instead."
                )
        elif isinstance(data, Series):
            if index is None:
                index = data.index
                data = data._mgr.copy(deep=False)
            else:
                data = data.reindex(index, copy=copy)
                copy = False
                data = data._mgr
        elif isinstance(data, Mapping):
            data, index = self._init_dict(data, index, dtype)
            dtype = None
            copy = False
        elif isinstance(data, (SingleBlockManager, SingleArrayManager)):
            if index is None:
                index = data.index
            elif not data.index.equals(index) or copy:
                # GH#19275 SingleBlockManager input should only be called
                # internally
                raise AssertionError(
                    "Cannot pass both SingleBlockManager "
                    "`data` argument and a different "
                    "`index` argument. `copy` must be False."
                )

            if not allow_mgr:
                warnings.warn(
                    f"Passing a {type(data).__name__} to {type(self).__name__} "
                    "is deprecated and will raise in a future version. "
                    "Use public APIs instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                allow_mgr = True

        elif isinstance(data, ExtensionArray):
            pass
        else:
            data = com.maybe_iterable_to_list(data)
            if is_list_like(data) and not len(data) and dtype is None:
                # GH 29405: Pre-2.0, this defaulted to float.
                dtype = np.dtype(object)

        if index is None:
            if not is_list_like(data):
                data = [data]
            index = default_index(len(data))
        elif is_list_like(data):
            com.require_length_match(data, index)

        # create/copy the manager
        if isinstance(data, (SingleBlockManager, SingleArrayManager)):
            if dtype is not None:
                data = data.astype(dtype=dtype, errors="ignore", copy=copy)
            elif copy:
                data = data.copy()
        else:
            data = sanitize_array(data, index, dtype, copy)

            manager = _get_option("mode.data_manager", silent=True)
            if manager == "block":
                data = SingleBlockManager.from_array(data, index, refs=refs)
            elif manager == "array":
                data = SingleArrayManager.from_array(data, index)

        NDFrame.__init__(self, data)
        self.name = name
        self._set_axis(0, index)

        if original_dtype is None and is_pandas_object and data_dtype == np.object_:
            if self.dtype != data_dtype:
                warnings.warn(
                    "Dtype inference on a pandas object "
                    "(Series, Index, ExtensionArray) is deprecated. The Series "
                    "constructor will keep the original dtype in the future. "
                    "Call `infer_objects` on the result to get the old behavior.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )

    def _init_dict(
        self, data: Mapping, index: Index | None = None, dtype: DtypeObj | None = None
    ):
        """
        Derive the "_mgr" and "index" attributes of a new Series from a
        dictionary input.

        Parameters
        ----------
        data : dict or dict-like
            Data used to populate the new Series.
        index : Index or None, default None
            Index for the new Series: if None, use dict keys.
        dtype : np.dtype, ExtensionDtype, or None, default None
            The dtype for the new Series: if None, infer from data.

        Returns
        -------
        _data : BlockManager for the new Series
        index : index for the new Series
        """
        keys: Index | tuple

        # Looking for NaN in dict doesn't work ({np.nan : 1}[float('nan')]
        # raises KeyError), so we iterate the entire dict, and align
        if data:
            # GH:34717, issue was using zip to extract key and values from data.
            # using generators in effects the performance.
            # Below is the new way of extracting the keys and values

            keys = tuple(data.keys())
            values = list(data.values())  # Generating list of values- faster way
        elif index is not None:
            # fastpath for Series(data=None). Just use broadcasting a scalar
            # instead of reindexing.
            if len(index) or dtype is not None:
                values = na_value_for_dtype(pandas_dtype(dtype), compat=False)
            else:
                values = []
            keys = index
        else:
            keys, values = default_index(0), []

        # Input is now list-like, so rely on "standard" construction:
        s = Series(values, index=keys, dtype=dtype)

        # Now we just make sure the order is respected, if any
        if data and index is not None:
            s = s.reindex(index, copy=False)
        return s._mgr, s.index

    # ----------------------------------------------------------------------

    @property
    def _constructor(self) -> Callable[..., Series]:
        return Series

    def _constructor_from_mgr(self, mgr, axes):
        ser = Series._from_mgr(mgr, axes=axes)
        ser._name = None  # caller is responsible for setting real name

        if type(self) is Series:
            # This would also work `if self._constructor is Series`, but
            #  this check is slightly faster, benefiting the most-common case.
            return ser

        # We assume that the subclass __init__ knows how to handle a
        #  pd.Series object.
        return self._constructor(ser)

    @property
    def _constructor_expanddim(self) -> Callable[..., DataFrame]:
        """
        Used when a manipulation result has one higher dimension as the
        original, such as Series.to_frame()
        """
        from pandas.core.frame import DataFrame

        return DataFrame

    def _constructor_expanddim_from_mgr(self, mgr, axes):
        from pandas.core.frame import DataFrame

        df = DataFrame._from_mgr(mgr, axes=mgr.axes)

        if type(self) is Series:
            # This would also work `if self._constructor_expanddim is DataFrame`,
            #  but this check is slightly faster, benefiting the most-common case.
            return df

        # We assume that the subclass __init__ knows how to handle a
        #  pd.DataFrame object.
        return self._constructor_expanddim(df)

    # types
    @property
    def _can_hold_na(self) -> bool:
        return self._mgr._can_hold_na

    # ndarray compatibility
    @property
    def dtype(self) -> DtypeObj:
        """
        Return the dtype object of the underlying data.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.dtype
        dtype('int64')
        """
        return self._mgr.dtype

    @property
    def dtypes(self) -> DtypeObj:
        """
        Return the dtype object of the underlying data.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.dtypes
        dtype('int64')
        """
        # DataFrame compatibility
        return self.dtype

    @property
    def name(self) -> Hashable:
        """
        Return the name of the Series.

        The name of a Series becomes its index or column name if it is used
        to form a DataFrame. It is also used whenever displaying the Series
        using the interpreter.

        Returns
        -------
        label (hashable object)
            The name of the Series, also the column name if part of a DataFrame.

        See Also
        --------
        Series.rename : Sets the Series name when given a scalar input.
        Index.name : Corresponding Index property.

        Examples
        --------
        The Series name can be set initially when calling the constructor.

        >>> s = pd.Series([1, 2, 3], dtype=np.int64, name='Numbers')
        >>> s
        0    1
        1    2
        2    3
        Name: Numbers, dtype: int64
        >>> s.name = "Integers"
        >>> s
        0    1
        1    2
        2    3
        Name: Integers, dtype: int64

        The name of a Series within a DataFrame is its column name.

        >>> df = pd.DataFrame([[1, 2], [3, 4], [5, 6]],
        ...                   columns=["Odd Numbers", "Even Numbers"])
        >>> df
           Odd Numbers  Even Numbers
        0            1             2
        1            3             4
        2            5             6
        >>> df["Even Numbers"].name
        'Even Numbers'
        """
        return self._name

    @name.setter
    def name(self, value: Hashable) -> None:
        validate_all_hashable(value, error_name=f"{type(self).__name__}.name")
        object.__setattr__(self, "_name", value)

    @property
    def values(self):
        """
        Return Series as ndarray or ndarray-like depending on the dtype.

        .. warning::

           We recommend using :attr:`Series.array` or
           :meth:`Series.to_numpy`, depending on whether you need
           a reference to the underlying data or a NumPy array.

        Returns
        -------
        numpy.ndarray or ndarray-like

        See Also
        --------
        Series.array : Reference to the underlying data.
        Series.to_numpy : A NumPy array representing the underlying data.

        Examples
        --------
        >>> pd.Series([1, 2, 3]).values
        array([1, 2, 3])

        >>> pd.Series(list('aabc')).values
        array(['a', 'a', 'b', 'c'], dtype=object)

        >>> pd.Series(list('aabc')).astype('category').values
        ['a', 'a', 'b', 'c']
        Categories (3, object): ['a', 'b', 'c']

        Timezone aware datetime data is converted to UTC:

        >>> pd.Series(pd.date_range('20130101', periods=3,
        ...                         tz='US/Eastern')).values
        array(['2013-01-01T05:00:00.000000000',
               '2013-01-02T05:00:00.000000000',
               '2013-01-03T05:00:00.000000000'], dtype='datetime64[ns]')
        """
        return self._mgr.external_values()

    @property
    def _values(self):
        """
        Return the internal repr of this data (defined by Block.interval_values).
        This are the values as stored in the Block (ndarray or ExtensionArray
        depending on the Block class), with datetime64[ns] and timedelta64[ns]
        wrapped in ExtensionArrays to match Index._values behavior.

        Differs from the public ``.values`` for certain data types, because of
        historical backwards compatibility of the public attribute (e.g. period
        returns object ndarray and datetimetz a datetime64[ns] ndarray for
        ``.values`` while it returns an ExtensionArray for ``._values`` in those
        cases).

        Differs from ``.array`` in that this still returns the numpy array if
        the Block is backed by a numpy array (except for datetime64 and
        timedelta64 dtypes), while ``.array`` ensures to always return an
        ExtensionArray.

        Overview:

        dtype       | values        | _values       | array                 |
        ----------- | ------------- | ------------- | --------------------- |
        Numeric     | ndarray       | ndarray       | NumpyExtensionArray   |
        Category    | Categorical   | Categorical   | Categorical           |
        dt64[ns]    | ndarray[M8ns] | DatetimeArray | DatetimeArray         |
        dt64[ns tz] | ndarray[M8ns] | DatetimeArray | DatetimeArray         |
        td64[ns]    | ndarray[m8ns] | TimedeltaArray| TimedeltaArray        |
        Period      | ndarray[obj]  | PeriodArray   | PeriodArray           |
        Nullable    | EA            | EA            | EA                    |

        """
        return self._mgr.internal_values()

    @property
    def _references(self) -> BlockValuesRefs | None:
        if isinstance(self._mgr, SingleArrayManager):
            return None
        return self._mgr._block.refs

    # error: Decorated property not supported
    @Appender(base.IndexOpsMixin.array.__doc__)  # type: ignore[misc]
    @property
    def array(self) -> ExtensionArray:
        return self._mgr.array_values()

    # ops
    def ravel(self, order: str = "C") -> ArrayLike:
        """
        Return the flattened underlying data as an ndarray or ExtensionArray.

        .. deprecated:: 2.2.0
            Series.ravel is deprecated. The underlying array is already 1D, so
            ravel is not necessary.  Use :meth:`to_numpy` for conversion to a numpy
            array instead.

        Returns
        -------
        numpy.ndarray or ExtensionArray
            Flattened data of the Series.

        See Also
        --------
        numpy.ndarray.ravel : Return a flattened array.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.ravel()  # doctest: +SKIP
        array([1, 2, 3])
        """
        warnings.warn(
            "Series.ravel is deprecated. The underlying array is already 1D, so "
            "ravel is not necessary.  Use `to_numpy()` for conversion to a numpy "
            "array instead.",
            FutureWarning,
            stacklevel=2,
        )
        arr = self._values.ravel(order=order)
        if isinstance(arr, np.ndarray) and using_copy_on_write():
            arr.flags.writeable = False
        return arr

    def __len__(self) -> int:
        """
        Return the length of the Series.
        """
        return len(self._mgr)

    def view(self, dtype: Dtype | None = None) -> Series:
        """
        Create a new view of the Series.

        .. deprecated:: 2.2.0
            ``Series.view`` is deprecated and will be removed in a future version.
            Use :meth:`Series.astype` as an alternative to change the dtype.

        This function will return a new Series with a view of the same
        underlying values in memory, optionally reinterpreted with a new data
        type. The new data type must preserve the same size in bytes as to not
        cause index misalignment.

        Parameters
        ----------
        dtype : data type
            Data type object or one of their string representations.

        Returns
        -------
        Series
            A new Series object as a view of the same data in memory.

        See Also
        --------
        numpy.ndarray.view : Equivalent numpy function to create a new view of
            the same data in memory.

        Notes
        -----
        Series are instantiated with ``dtype=float64`` by default. While
        ``numpy.ndarray.view()`` will return a view with the same data type as
        the original array, ``Series.view()`` (without specified dtype)
        will try using ``float64`` and may fail if the original data type size
        in bytes is not the same.

        Examples
        --------
        Use ``astype`` to change the dtype instead.
        """
        warnings.warn(
            "Series.view is deprecated and will be removed in a future version. "
            "Use ``astype`` as an alternative to change the dtype.",
            FutureWarning,
            stacklevel=2,
        )
        # self.array instead of self._values so we piggyback on NumpyExtensionArray
        #  implementation
        res_values = self.array.view(dtype)
        res_ser = self._constructor(res_values, index=self.index, copy=False)
        if isinstance(res_ser._mgr, SingleBlockManager):
            blk = res_ser._mgr._block
            blk.refs = cast("BlockValuesRefs", self._references)
            blk.refs.add_reference(blk)
        return res_ser.__finalize__(self, method="view")

    # ----------------------------------------------------------------------
    # NDArray Compat
    def __array__(
        self, dtype: npt.DTypeLike | None = None, copy: bool | None = None
    ) -> np.ndarray:
        """
        Return the values as a NumPy array.

        Users should not call this directly. Rather, it is invoked by
        :func:`numpy.array` and :func:`numpy.asarray`.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to use for the resulting NumPy array. By default,
            the dtype is inferred from the data.

        copy : bool or None, optional
            See :func:`numpy.asarray`.

        Returns
        -------
        numpy.ndarray
            The values in the series converted to a :class:`numpy.ndarray`
            with the specified `dtype`.

        See Also
        --------
        array : Create a new array from data.
        Series.array : Zero-copy view to the array backing the Series.
        Series.to_numpy : Series method for similar behavior.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3])
        >>> np.asarray(ser)
        array([1, 2, 3])

        For timezone-aware data, the timezones may be retained with
        ``dtype='object'``

        >>> tzser = pd.Series(pd.date_range('2000', periods=2, tz="CET"))
        >>> np.asarray(tzser, dtype="object")
        array([Timestamp('2000-01-01 00:00:00+0100', tz='CET'),
               Timestamp('2000-01-02 00:00:00+0100', tz='CET')],
              dtype=object)

        Or the values may be localized to UTC and the tzinfo discarded with
        ``dtype='datetime64[ns]'``

        >>> np.asarray(tzser, dtype="datetime64[ns]")  # doctest: +ELLIPSIS
        array(['1999-12-31T23:00:00.000000000', ...],
              dtype='datetime64[ns]')
        """
        values = self._values
        if copy is None:
            # Note: branch avoids `copy=None` for NumPy 1.x support
            arr = np.asarray(values, dtype=dtype)
        else:
            arr = np.array(values, dtype=dtype, copy=copy)

        if copy is True:
            return arr
        if using_copy_on_write() and (
            copy is False or astype_is_view(values.dtype, arr.dtype)
        ):
            arr = arr.view()
            arr.flags.writeable = False
        return arr

    # ----------------------------------------------------------------------

    def __column_consortium_standard__(self, *, api_version: str | None = None) -> Any:
        """
        Provide entry point to the Consortium DataFrame Standard API.

        This is developed and maintained outside of pandas.
        Please report any issues to https://github.com/data-apis/dataframe-api-compat.
        """
        dataframe_api_compat = import_optional_dependency("dataframe_api_compat")
        return (
            dataframe_api_compat.pandas_standard.convert_to_standard_compliant_column(
                self, api_version=api_version
            )
        )

    # ----------------------------------------------------------------------
    # Unary Methods

    # coercion
    __float__ = _coerce_method(float)
    __int__ = _coerce_method(int)

    # ----------------------------------------------------------------------

    # indexers
    @property
    def axes(self) -> list[Index]:
        """
        Return a list of the row axis labels.
        """
        return [self.index]

    # ----------------------------------------------------------------------
    # Indexing Methods

    def _ixs(self, i: int, axis: AxisInt = 0) -> Any:
        """
        Return the i-th value or values in the Series by location.

        Parameters
        ----------
        i : int

        Returns
        -------
        scalar
        """
        return self._values[i]

    def _slice(self, slobj: slice, axis: AxisInt = 0) -> Series:
        # axis kwarg is retained for compat with NDFrame method
        #  _slice is *always* positional
        mgr = self._mgr.get_slice(slobj, axis=axis)
        out = self._constructor_from_mgr(mgr, axes=mgr.axes)
        out._name = self._name
        return out.__finalize__(self)

    def __getitem__(self, key):
        check_dict_or_set_indexers(key)
        key = com.apply_if_callable(key, self)

        if key is Ellipsis:
            if using_copy_on_write() or warn_copy_on_write():
                return self.copy(deep=False)
            return self

        key_is_scalar = is_scalar(key)
        if isinstance(key, (list, tuple)):
            key = unpack_1tuple(key)

        if is_integer(key) and self.index._should_fallback_to_positional:
            warnings.warn(
                # GH#50617
                "Series.__getitem__ treating keys as positions is deprecated. "
                "In a future version, integer keys will always be treated "
                "as labels (consistent with DataFrame behavior). To access "
                "a value by position, use `ser.iloc[pos]`",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
            return self._values[key]

        elif key_is_scalar:
            return self._get_value(key)

        # Convert generator to list before going through hashable part
        # (We will iterate through the generator there to check for slices)
        if is_iterator(key):
            key = list(key)

        if is_hashable(key) and not isinstance(key, slice):
            # Otherwise index.get_value will raise InvalidIndexError
            try:
                # For labels that don't resolve as scalars like tuples and frozensets
                result = self._get_value(key)

                return result

            except (KeyError, TypeError, InvalidIndexError):
                # InvalidIndexError for e.g. generator
                #  see test_series_getitem_corner_generator
                if isinstance(key, tuple) and isinstance(self.index, MultiIndex):
                    # We still have the corner case where a tuple is a key
                    # in the first level of our MultiIndex
                    return self._get_values_tuple(key)

        if isinstance(key, slice):
            # Do slice check before somewhat-costly is_bool_indexer
            return self._getitem_slice(key)

        if com.is_bool_indexer(key):
            key = check_bool_indexer(self.index, key)
            key = np.asarray(key, dtype=bool)
            return self._get_rows_with_mask(key)

        return self._get_with(key)

    def _get_with(self, key):
        # other: fancy integer or otherwise
        if isinstance(key, ABCDataFrame):
            raise TypeError(
                "Indexing a Series with DataFrame is not "
                "supported, use the appropriate DataFrame column"
            )
        elif isinstance(key, tuple):
            return self._get_values_tuple(key)

        elif not is_list_like(key):
            # e.g. scalars that aren't recognized by lib.is_scalar, GH#32684
            return self.loc[key]

        if not isinstance(key, (list, np.ndarray, ExtensionArray, Series, Index)):
            key = list(key)

        key_type = lib.infer_dtype(key, skipna=False)

        # Note: The key_type == "boolean" case should be caught by the
        #  com.is_bool_indexer check in __getitem__
        if key_type == "integer":
            # We need to decide whether to treat this as a positional indexer
            #  (i.e. self.iloc) or label-based (i.e. self.loc)
            if not self.index._should_fallback_to_positional:
                return self.loc[key]
            else:
                warnings.warn(
                    # GH#50617
                    "Series.__getitem__ treating keys as positions is deprecated. "
                    "In a future version, integer keys will always be treated "
                    "as labels (consistent with DataFrame behavior). To access "
                    "a value by position, use `ser.iloc[pos]`",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
                return self.iloc[key]

        # handle the dup indexing case GH#4246
        return self.loc[key]

    def _get_values_tuple(self, key: tuple):
        # mpl hackaround
        if com.any_none(*key):
            # mpl compat if we look up e.g. ser[:, np.newaxis];
            #  see tests.series.timeseries.test_mpl_compat_hack
            # the asarray is needed to avoid returning a 2D DatetimeArray
            result = np.asarray(self._values[key])
            disallow_ndim_indexing(result)
            return result

        if not isinstance(self.index, MultiIndex):
            raise KeyError("key of type tuple not found and not a MultiIndex")

        # If key is contained, would have returned by now
        indexer, new_index = self.index.get_loc_level(key)
        new_ser = self._constructor(self._values[indexer], index=new_index, copy=False)
        if isinstance(indexer, slice):
            new_ser._mgr.add_references(self._mgr)  # type: ignore[arg-type]
        return new_ser.__finalize__(self)

    def _get_rows_with_mask(self, indexer: npt.NDArray[np.bool_]) -> Series:
        new_mgr = self._mgr.get_rows_with_mask(indexer)
        return self._constructor_from_mgr(new_mgr, axes=new_mgr.axes).__finalize__(self)

    def _get_value(self, label, takeable: bool = False):
        """
        Quickly retrieve single value at passed index label.

        Parameters
        ----------
        label : object
        takeable : interpret the index as indexers, default False

        Returns
        -------
        scalar value
        """
        if takeable:
            return self._values[label]

        # Similar to Index.get_value, but we do not fall back to positional
        loc = self.index.get_loc(label)

        if is_integer(loc):
            return self._values[loc]

        if isinstance(self.index, MultiIndex):
            mi = self.index
            new_values = self._values[loc]
            if len(new_values) == 1 and mi.nlevels == 1:
                # If more than one level left, we can not return a scalar
                return new_values[0]

            new_index = mi[loc]
            new_index = maybe_droplevels(new_index, label)
            new_ser = self._constructor(
                new_values, index=new_index, name=self.name, copy=False
            )
            if isinstance(loc, slice):
                new_ser._mgr.add_references(self._mgr)  # type: ignore[arg-type]
            return new_ser.__finalize__(self)

        else:
            return self.iloc[loc]

    def __setitem__(self, key, value) -> None:
        warn = True
        if not PYPY and using_copy_on_write():
            if sys.getrefcount(self) <= 3:
                warnings.warn(
                    _chained_assignment_msg, ChainedAssignmentError, stacklevel=2
                )
        elif not PYPY and not using_copy_on_write():
            ctr = sys.getrefcount(self)
            ref_count = 3
            if not warn_copy_on_write() and _check_cacher(self):
                # see https://github.com/pandas-dev/pandas/pull/56060#discussion_r1399245221
                ref_count += 1
            if ctr <= ref_count and (
                warn_copy_on_write()
                or (
                    not warn_copy_on_write()
                    and self._mgr.blocks[0].refs.has_reference()  # type: ignore[union-attr]
                )
            ):
                warn = False
                warnings.warn(
                    _chained_assignment_warning_msg, FutureWarning, stacklevel=2
                )

        check_dict_or_set_indexers(key)
        key = com.apply_if_callable(key, self)
        cacher_needs_updating = self._check_is_chained_assignment_possible()

        if key is Ellipsis:
            key = slice(None)

        if isinstance(key, slice):
            indexer = self.index._convert_slice_indexer(key, kind="getitem")
            return self._set_values(indexer, value, warn=warn)

        try:
            self._set_with_engine(key, value, warn=warn)
        except KeyError:
            # We have a scalar (or for MultiIndex or object-dtype, scalar-like)
            #  key that is not present in self.index.
            if is_integer(key):
                if not self.index._should_fallback_to_positional:
                    # GH#33469
                    self.loc[key] = value
                else:
                    # positional setter
                    # can't use _mgr.setitem_inplace yet bc could have *both*
                    #  KeyError and then ValueError, xref GH#45070
                    warnings.warn(
                        # GH#50617
                        "Series.__setitem__ treating keys as positions is deprecated. "
                        "In a future version, integer keys will always be treated "
                        "as labels (consistent with DataFrame behavior). To set "
                        "a value by position, use `ser.iloc[pos] = value`",
                        FutureWarning,
                        stacklevel=find_stack_level(),
                    )
                    self._set_values(key, value)
            else:
                # GH#12862 adding a new key to the Series
                self.loc[key] = value

        except (TypeError, ValueError, LossySetitemError):
            # The key was OK, but we cannot set the value losslessly
            indexer = self.index.get_loc(key)
            self._set_values(indexer, value)

        except InvalidIndexError as err:
            if isinstance(key, tuple) and not isinstance(self.index, MultiIndex):
                # cases with MultiIndex don't get here bc they raise KeyError
                # e.g. test_basic_getitem_setitem_corner
                raise KeyError(
                    "key of type tuple not found and not a MultiIndex"
                ) from err

            if com.is_bool_indexer(key):
                key = check_bool_indexer(self.index, key)
                key = np.asarray(key, dtype=bool)

                if (
                    is_list_like(value)
                    and len(value) != len(self)
                    and not isinstance(value, Series)
                    and not is_object_dtype(self.dtype)
                ):
                    # Series will be reindexed to have matching length inside
                    #  _where call below
                    # GH#44265
                    indexer = key.nonzero()[0]
                    self._set_values(indexer, value)
                    return

                # otherwise with listlike other we interpret series[mask] = other
                #  as series[mask] = other[mask]
                try:
                    self._where(~key, value, inplace=True, warn=warn)
                except InvalidIndexError:
                    # test_where_dups
                    self.iloc[key] = value
                return

            else:
                self._set_with(key, value, warn=warn)

        if cacher_needs_updating:
            self._maybe_update_cacher(inplace=True)

    def _set_with_engine(self, key, value, warn: bool = True) -> None:
        loc = self.index.get_loc(key)

        # this is equivalent to self._values[key] = value
        self._mgr.setitem_inplace(loc, value, warn=warn)

    def _set_with(self, key, value, warn: bool = True) -> None:
        # We got here via exception-handling off of InvalidIndexError, so
        #  key should always be listlike at this point.
        assert not isinstance(key, tuple)

        if is_iterator(key):
            # Without this, the call to infer_dtype will consume the generator
            key = list(key)

        if not self.index._should_fallback_to_positional:
            # Regardless of the key type, we're treating it as labels
            self._set_labels(key, value, warn=warn)

        else:
            # Note: key_type == "boolean" should not occur because that
            #  should be caught by the is_bool_indexer check in __setitem__
            key_type = lib.infer_dtype(key, skipna=False)

            if key_type == "integer":
                warnings.warn(
                    # GH#50617
                    "Series.__setitem__ treating keys as positions is deprecated. "
                    "In a future version, integer keys will always be treated "
                    "as labels (consistent with DataFrame behavior). To set "
                    "a value by position, use `ser.iloc[pos] = value`",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
                self._set_values(key, value, warn=warn)
            else:
                self._set_labels(key, value, warn=warn)

    def _set_labels(self, key, value, warn: bool = True) -> None:
        key = com.asarray_tuplesafe(key)
        indexer: np.ndarray = self.index.get_indexer(key)
        mask = indexer == -1
        if mask.any():
            raise KeyError(f"{key[mask]} not in index")
        self._set_values(indexer, value, warn=warn)

    def _set_values(self, key, value, warn: bool = True) -> None:
        if isinstance(key, (Index, Series)):
            key = key._values

        self._mgr = self._mgr.setitem(indexer=key, value=value, warn=warn)
        self._maybe_update_cacher()

    def _set_value(self, label, value, takeable: bool = False) -> None:
        """
        Quickly set single value at passed label.

        If label is not contained, a new object is created with the label
        placed at the end of the result index.

        Parameters
        ----------
        label : object
            Partial indexing with MultiIndex not allowed.
        value : object
            Scalar value.
        takeable : interpret the index as indexers, default False
        """
        if not takeable:
            try:
                loc = self.index.get_loc(label)
            except KeyError:
                # set using a non-recursive method
                self.loc[label] = value
                return
        else:
            loc = label

        self._set_values(loc, value)

    # ----------------------------------------------------------------------
    # Lookup Caching

    @property
    def _is_cached(self) -> bool:
        """Return boolean indicating if self is cached or not."""
        return getattr(self, "_cacher", None) is not None

    def _get_cacher(self):
        """return my cacher or None"""
        cacher = getattr(self, "_cacher", None)
        if cacher is not None:
            cacher = cacher[1]()
        return cacher

    def _reset_cacher(self) -> None:
        """
        Reset the cacher.
        """
        if hasattr(self, "_cacher"):
            del self._cacher

    def _set_as_cached(self, item, cacher) -> None:
        """
        Set the _cacher attribute on the calling object with a weakref to
        cacher.
        """
        if using_copy_on_write():
            return
        self._cacher = (item, weakref.ref(cacher))

    def _clear_item_cache(self) -> None:
        # no-op for Series
        pass

    def _check_is_chained_assignment_possible(self) -> bool:
        """
        See NDFrame._check_is_chained_assignment_possible.__doc__
        """
        if self._is_view and self._is_cached:
            ref = self._get_cacher()
            if ref is not None and ref._is_mixed_type:
                self._check_setitem_copy(t="referent", force=True)
            return True
        return super()._check_is_chained_assignment_possible()

    def _maybe_update_cacher(
        self, clear: bool = False, verify_is_copy: bool = True, inplace: bool = False
    ) -> None:
        """
        See NDFrame._maybe_update_cacher.__doc__
        """
        # for CoW, we never want to update the parent DataFrame cache
        # if the Series changed, but don't keep track of any cacher
        if using_copy_on_write():
            return
        cacher = getattr(self, "_cacher", None)
        if cacher is not None:
            ref: DataFrame = cacher[1]()

            # we are trying to reference a dead referent, hence
            # a copy
            if ref is None:
                del self._cacher
            elif len(self) == len(ref) and self.name in ref.columns:
                # GH#42530 self.name must be in ref.columns
                # to ensure column still in dataframe
                # otherwise, either self or ref has swapped in new arrays
                ref._maybe_cache_changed(cacher[0], self, inplace=inplace)
            else:
                # GH#33675 we have swapped in a new array, so parent
                #  reference to self is now invalid
                ref._item_cache.pop(cacher[0], None)

        super()._maybe_update_cacher(
            clear=clear, verify_is_copy=verify_is_copy, inplace=inplace
        )

    # ----------------------------------------------------------------------
    # Unsorted

    def repeat(self, repeats: int | Sequence[int], axis: None = None) -> Series:
        """
        Repeat elements of a Series.

        Returns a new Series where each element of the current Series
        is repeated consecutively a given number of times.

        Parameters
        ----------
        repeats : int or array of ints
            The number of repetitions for each element. This should be a
            non-negative integer. Repeating 0 times will return an empty
            Series.
        axis : None
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            Newly created Series with repeated elements.

        See Also
        --------
        Index.repeat : Equivalent function for Index.
        numpy.repeat : Similar method for :class:`numpy.ndarray`.

        Examples
        --------
        >>> s = pd.Series(['a', 'b', 'c'])
        >>> s
        0    a
        1    b
        2    c
        dtype: object
        >>> s.repeat(2)
        0    a
        0    a
        1    b
        1    b
        2    c
        2    c
        dtype: object
        >>> s.repeat([1, 2, 3])
        0    a
        1    b
        1    b
        2    c
        2    c
        2    c
        dtype: object
        """
        nv.validate_repeat((), {"axis": axis})
        new_index = self.index.repeat(repeats)
        new_values = self._values.repeat(repeats)
        return self._constructor(new_values, index=new_index, copy=False).__finalize__(
            self, method="repeat"
        )

    @overload
    def reset_index(
        self,
        level: IndexLabel = ...,
        *,
        drop: Literal[False] = ...,
        name: Level = ...,
        inplace: Literal[False] = ...,
        allow_duplicates: bool = ...,
    ) -> DataFrame:
        ...

    @overload
    def reset_index(
        self,
        level: IndexLabel = ...,
        *,
        drop: Literal[True],
        name: Level = ...,
        inplace: Literal[False] = ...,
        allow_duplicates: bool = ...,
    ) -> Series:
        ...

    @overload
    def reset_index(
        self,
        level: IndexLabel = ...,
        *,
        drop: bool = ...,
        name: Level = ...,
        inplace: Literal[True],
        allow_duplicates: bool = ...,
    ) -> None:
        ...

    def reset_index(
        self,
        level: IndexLabel | None = None,
        *,
        drop: bool = False,
        name: Level = lib.no_default,
        inplace: bool = False,
        allow_duplicates: bool = False,
    ) -> DataFrame | Series | None:
        """
        Generate a new DataFrame or Series with the index reset.

        This is useful when the index needs to be treated as a column, or
        when the index is meaningless and needs to be reset to the default
        before another operation.

        Parameters
        ----------
        level : int, str, tuple, or list, default optional
            For a Series with a MultiIndex, only remove the specified levels
            from the index. Removes all levels by default.
        drop : bool, default False
            Just reset the index, without inserting it as a column in
            the new DataFrame.
        name : object, optional
            The name to use for the column containing the original Series
            values. Uses ``self.name`` by default. This argument is ignored
            when `drop` is True.
        inplace : bool, default False
            Modify the Series in place (do not create a new object).
        allow_duplicates : bool, default False
            Allow duplicate column labels to be created.

            .. versionadded:: 1.5.0

        Returns
        -------
        Series or DataFrame or None
            When `drop` is False (the default), a DataFrame is returned.
            The newly created columns will come first in the DataFrame,
            followed by the original Series values.
            When `drop` is True, a `Series` is returned.
            In either case, if ``inplace=True``, no value is returned.

        See Also
        --------
        DataFrame.reset_index: Analogous function for DataFrame.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4], name='foo',
        ...               index=pd.Index(['a', 'b', 'c', 'd'], name='idx'))

        Generate a DataFrame with default index.

        >>> s.reset_index()
          idx  foo
        0   a    1
        1   b    2
        2   c    3
        3   d    4

        To specify the name of the new column use `name`.

        >>> s.reset_index(name='values')
          idx  values
        0   a       1
        1   b       2
        2   c       3
        3   d       4

        To generate a new Series with the default set `drop` to True.

        >>> s.reset_index(drop=True)
        0    1
        1    2
        2    3
        3    4
        Name: foo, dtype: int64

        The `level` parameter is interesting for Series with a multi-level
        index.

        >>> arrays = [np.array(['bar', 'bar', 'baz', 'baz']),
        ...           np.array(['one', 'two', 'one', 'two'])]
        >>> s2 = pd.Series(
        ...     range(4), name='foo',
        ...     index=pd.MultiIndex.from_arrays(arrays,
        ...                                     names=['a', 'b']))

        To remove a specific level from the Index, use `level`.

        >>> s2.reset_index(level='a')
               a  foo
        b
        one  bar    0
        two  bar    1
        one  baz    2
        two  baz    3

        If `level` is not set, all levels are removed from the Index.

        >>> s2.reset_index()
             a    b  foo
        0  bar  one    0
        1  bar  two    1
        2  baz  one    2
        3  baz  two    3
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        if drop:
            new_index = default_index(len(self))
            if level is not None:
                level_list: Sequence[Hashable]
                if not isinstance(level, (tuple, list)):
                    level_list = [level]
                else:
                    level_list = level
                level_list = [self.index._get_level_number(lev) for lev in level_list]
                if len(level_list) < self.index.nlevels:
                    new_index = self.index.droplevel(level_list)

            if inplace:
                self.index = new_index
            elif using_copy_on_write():
                new_ser = self.copy(deep=False)
                new_ser.index = new_index
                return new_ser.__finalize__(self, method="reset_index")
            else:
                return self._constructor(
                    self._values.copy(), index=new_index, copy=False, dtype=self.dtype
                ).__finalize__(self, method="reset_index")
        elif inplace:
            raise TypeError(
                "Cannot reset_index inplace on a Series to create a DataFrame"
            )
        else:
            if name is lib.no_default:
                # For backwards compatibility, keep columns as [0] instead of
                #  [None] when self.name is None
                if self.name is None:
                    name = 0
                else:
                    name = self.name

            df = self.to_frame(name)
            return df.reset_index(
                level=level, drop=drop, allow_duplicates=allow_duplicates
            )
        return None

    # ----------------------------------------------------------------------
    # Rendering Methods

    def __repr__(self) -> str:
        """
        Return a string representation for a particular Series.
        """
        # pylint: disable=invalid-repr-returned
        repr_params = fmt.get_series_repr_params()
        return self.to_string(**repr_params)

    @overload
    def to_string(
        self,
        buf: None = ...,
        na_rep: str = ...,
        float_format: str | None = ...,
        header: bool = ...,
        index: bool = ...,
        length: bool = ...,
        dtype=...,
        name=...,
        max_rows: int | None = ...,
        min_rows: int | None = ...,
    ) -> str:
        ...

    @overload
    def to_string(
        self,
        buf: FilePath | WriteBuffer[str],
        na_rep: str = ...,
        float_format: str | None = ...,
        header: bool = ...,
        index: bool = ...,
        length: bool = ...,
        dtype=...,
        name=...,
        max_rows: int | None = ...,
        min_rows: int | None = ...,
    ) -> None:
        ...

    def to_string(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        na_rep: str = "NaN",
        float_format: str | None = None,
        header: bool = True,
        index: bool = True,
        length: bool = False,
        dtype: bool = False,
        name: bool = False,
        max_rows: int | None = None,
        min_rows: int | None = None,
    ) -> str | None:
        """
        Render a string representation of the Series.

        Parameters
        ----------
        buf : StringIO-like, optional
            Buffer to write to.
        na_rep : str, optional
            String representation of NaN to use, default 'NaN'.
        float_format : one-parameter function, optional
            Formatter function to apply to columns' elements if they are
            floats, default None.
        header : bool, default True
            Add the Series header (index name).
        index : bool, optional
            Add index (row) labels, default True.
        length : bool, default False
            Add the Series length.
        dtype : bool, default False
            Add the Series dtype.
        name : bool, default False
            Add the Series name if not None.
        max_rows : int, optional
            Maximum number of rows to show before truncating. If None, show
            all.
        min_rows : int, optional
            The number of rows to display in a truncated repr (when number
            of rows is above `max_rows`).

        Returns
        -------
        str or None
            String representation of Series if ``buf=None``, otherwise None.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3]).to_string()
        >>> ser
        '0    1\\n1    2\\n2    3'
        """
        formatter = fmt.SeriesFormatter(
            self,
            name=name,
            length=length,
            header=header,
            index=index,
            dtype=dtype,
            na_rep=na_rep,
            float_format=float_format,
            min_rows=min_rows,
            max_rows=max_rows,
        )
        result = formatter.to_string()

        # catch contract violations
        if not isinstance(result, str):
            raise AssertionError(
                "result must be of type str, type "
                f"of result is {repr(type(result).__name__)}"
            )

        if buf is None:
            return result
        else:
            if hasattr(buf, "write"):
                buf.write(result)
            else:
                with open(buf, "w", encoding="utf-8") as f:
                    f.write(result)
        return None

    @doc(
        klass=_shared_doc_kwargs["klass"],
        storage_options=_shared_docs["storage_options"],
        examples=dedent(
            """Examples
            --------
            >>> s = pd.Series(["elk", "pig", "dog", "quetzal"], name="animal")
            >>> print(s.to_markdown())
            |    | animal   |
            |---:|:---------|
            |  0 | elk      |
            |  1 | pig      |
            |  2 | dog      |
            |  3 | quetzal  |

            Output markdown with a tabulate option.

            >>> print(s.to_markdown(tablefmt="grid"))
            +----+----------+
            |    | animal   |
            +====+==========+
            |  0 | elk      |
            +----+----------+
            |  1 | pig      |
            +----+----------+
            |  2 | dog      |
            +----+----------+
            |  3 | quetzal  |
            +----+----------+"""
        ),
    )
    def to_markdown(
        self,
        buf: IO[str] | None = None,
        mode: str = "wt",
        index: bool = True,
        storage_options: StorageOptions | None = None,
        **kwargs,
    ) -> str | None:
        """
        Print {klass} in Markdown-friendly format.

        Parameters
        ----------
        buf : str, Path or StringIO-like, optional, default None
            Buffer to write to. If None, the output is returned as a string.
        mode : str, optional
            Mode in which file is opened, "wt" by default.
        index : bool, optional, default True
            Add index (row) labels.

        {storage_options}

        **kwargs
            These parameters will be passed to `tabulate \
                <https://pypi.org/project/tabulate>`_.

        Returns
        -------
        str
            {klass} in Markdown-friendly format.

        Notes
        -----
        Requires the `tabulate <https://pypi.org/project/tabulate>`_ package.

        {examples}
        """
        return self.to_frame().to_markdown(
            buf, mode=mode, index=index, storage_options=storage_options, **kwargs
        )

    # ----------------------------------------------------------------------

    def items(self) -> Iterable[tuple[Hashable, Any]]:
        """
        Lazily iterate over (index, value) tuples.

        This method returns an iterable tuple (index, value). This is
        convenient if you want to create a lazy iterator.

        Returns
        -------
        iterable
            Iterable of tuples containing the (index, value) pairs from a
            Series.

        See Also
        --------
        DataFrame.items : Iterate over (column name, Series) pairs.
        DataFrame.iterrows : Iterate over DataFrame rows as (index, Series) pairs.

        Examples
        --------
        >>> s = pd.Series(['A', 'B', 'C'])
        >>> for index, value in s.items():
        ...     print(f"Index : {index}, Value : {value}")
        Index : 0, Value : A
        Index : 1, Value : B
        Index : 2, Value : C
        """
        return zip(iter(self.index), iter(self))

    # ----------------------------------------------------------------------
    # Misc public methods

    def keys(self) -> Index:
        """
        Return alias for index.

        Returns
        -------
        Index
            Index of the Series.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3], index=[0, 1, 2])
        >>> s.keys()
        Index([0, 1, 2], dtype='int64')
        """
        return self.index

    @overload
    def to_dict(
        self, *, into: type[MutableMappingT] | MutableMappingT
    ) -> MutableMappingT:
        ...

    @overload
    def to_dict(self, *, into: type[dict] = ...) -> dict:
        ...

    # error: Incompatible default for argument "into" (default has type "type[
    # dict[Any, Any]]", argument has type "type[MutableMappingT] | MutableMappingT")
    @deprecate_nonkeyword_arguments(
        version="3.0", allowed_args=["self"], name="to_dict"
    )
    def to_dict(
        self,
        into: type[MutableMappingT]
        | MutableMappingT = dict,  # type: ignore[assignment]
    ) -> MutableMappingT:
        """
        Convert Series to {label -> value} dict or dict-like object.

        Parameters
        ----------
        into : class, default dict
            The collections.abc.MutableMapping subclass to use as the return
            object. Can be the actual class or an empty instance of the mapping
            type you want.  If you want a collections.defaultdict, you must
            pass it initialized.

        Returns
        -------
        collections.abc.MutableMapping
            Key-value representation of Series.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.to_dict()
        {0: 1, 1: 2, 2: 3, 3: 4}
        >>> from collections import OrderedDict, defaultdict
        >>> s.to_dict(into=OrderedDict)
        OrderedDict([(0, 1), (1, 2), (2, 3), (3, 4)])
        >>> dd = defaultdict(list)
        >>> s.to_dict(into=dd)
        defaultdict(<class 'list'>, {0: 1, 1: 2, 2: 3, 3: 4})
        """
        # GH16122
        into_c = com.standardize_mapping(into)

        if is_object_dtype(self.dtype) or isinstance(self.dtype, ExtensionDtype):
            return into_c((k, maybe_box_native(v)) for k, v in self.items())
        else:
            # Not an object dtype => all types will be the same so let the default
            # indexer return native python type
            return into_c(self.items())

    def to_frame(self, name: Hashable = lib.no_default) -> DataFrame:
        """
        Convert Series to DataFrame.

        Parameters
        ----------
        name : object, optional
            The passed name should substitute for the series name (if it has
            one).

        Returns
        -------
        DataFrame
            DataFrame representation of Series.

        Examples
        --------
        >>> s = pd.Series(["a", "b", "c"],
        ...               name="vals")
        >>> s.to_frame()
          vals
        0    a
        1    b
        2    c
        """
        columns: Index
        if name is lib.no_default:
            name = self.name
            if name is None:
                # default to [0], same as we would get with DataFrame(self)
                columns = default_index(1)
            else:
                columns = Index([name])
        else:
            columns = Index([name])

        mgr = self._mgr.to_2d_mgr(columns)
        df = self._constructor_expanddim_from_mgr(mgr, axes=mgr.axes)
        return df.__finalize__(self, method="to_frame")

    def _set_name(
        self, name, inplace: bool = False, deep: bool | None = None
    ) -> Series:
        """
        Set the Series name.

        Parameters
        ----------
        name : str
        inplace : bool
            Whether to modify `self` directly or return a copy.
        deep : bool|None, default None
            Whether to do a deep copy, a shallow copy, or Copy on Write(None)
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        ser = self if inplace else self.copy(deep and not using_copy_on_write())
        ser.name = name
        return ser

    @Appender(
        dedent(
            """
        Examples
        --------
        >>> ser = pd.Series([390., 350., 30., 20.],
        ...                 index=['Falcon', 'Falcon', 'Parrot', 'Parrot'],
        ...                 name="Max Speed")
        >>> ser
        Falcon    390.0
        Falcon    350.0
        Parrot     30.0
        Parrot     20.0
        Name: Max Speed, dtype: float64
        >>> ser.groupby(["a", "b", "a", "b"]).mean()
        a    210.0
        b    185.0
        Name: Max Speed, dtype: float64
        >>> ser.groupby(level=0).mean()
        Falcon    370.0
        Parrot     25.0
        Name: Max Speed, dtype: float64
        >>> ser.groupby(ser > 100).mean()
        Max Speed
        False     25.0
        True     370.0
        Name: Max Speed, dtype: float64

        **Grouping by Indexes**

        We can groupby different levels of a hierarchical index
        using the `level` parameter:

        >>> arrays = [['Falcon', 'Falcon', 'Parrot', 'Parrot'],
        ...           ['Captive', 'Wild', 'Captive', 'Wild']]
        >>> index = pd.MultiIndex.from_arrays(arrays, names=('Animal', 'Type'))
        >>> ser = pd.Series([390., 350., 30., 20.], index=index, name="Max Speed")
        >>> ser
        Animal  Type
        Falcon  Captive    390.0
                Wild       350.0
        Parrot  Captive     30.0
                Wild        20.0
        Name: Max Speed, dtype: float64
        >>> ser.groupby(level=0).mean()
        Animal
        Falcon    370.0
        Parrot     25.0
        Name: Max Speed, dtype: float64
        >>> ser.groupby(level="Type").mean()
        Type
        Captive    210.0
        Wild       185.0
        Name: Max Speed, dtype: float64

        We can also choose to include `NA` in group keys or not by defining
        `dropna` parameter, the default setting is `True`.

        >>> ser = pd.Series([1, 2, 3, 3], index=["a", 'a', 'b', np.nan])
        >>> ser.groupby(level=0).sum()
        a    3
        b    3
        dtype: int64

        >>> ser.groupby(level=0, dropna=False).sum()
        a    3
        b    3
        NaN  3
        dtype: int64

        >>> arrays = ['Falcon', 'Falcon', 'Parrot', 'Parrot']
        >>> ser = pd.Series([390., 350., 30., 20.], index=arrays, name="Max Speed")
        >>> ser.groupby(["a", "b", "a", np.nan]).mean()
        a    210.0
        b    350.0
        Name: Max Speed, dtype: float64

        >>> ser.groupby(["a", "b", "a", np.nan], dropna=False).mean()
        a    210.0
        b    350.0
        NaN   20.0
        Name: Max Speed, dtype: float64
        """
        )
    )
    @Appender(_shared_docs["groupby"] % _shared_doc_kwargs)
    def groupby(
        self,
        by=None,
        axis: Axis = 0,
        level: IndexLabel | None = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: bool | lib.NoDefault = lib.no_default,
        dropna: bool = True,
    ) -> SeriesGroupBy:
        from pandas.core.groupby.generic import SeriesGroupBy

        if level is None and by is None:
            raise TypeError("You have to supply one of 'by' and 'level'")
        if not as_index:
            raise TypeError("as_index=False only valid with DataFrame")
        axis = self._get_axis_number(axis)

        return SeriesGroupBy(
            obj=self,
            keys=by,
            axis=axis,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            observed=observed,
            dropna=dropna,
        )

    # ----------------------------------------------------------------------
    # Statistics, overridden ndarray methods

    # TODO: integrate bottleneck
    def count(self) -> int:
        """
        Return number of non-NA/null observations in the Series.

        Returns
        -------
        int
            Number of non-null values in the Series.

        See Also
        --------
        DataFrame.count : Count non-NA cells for each column or row.

        Examples
        --------
        >>> s = pd.Series([0.0, 1.0, np.nan])
        >>> s.count()
        2
        """
        return notna(self._values).sum().astype("int64")

    def mode(self, dropna: bool = True) -> Series:
        """
        Return the mode(s) of the Series.

        The mode is the value that appears most often. There can be multiple modes.

        Always returns Series even if only one value is returned.

        Parameters
        ----------
        dropna : bool, default True
            Don't consider counts of NaN/NaT.

        Returns
        -------
        Series
            Modes of the Series in sorted order.

        Examples
        --------
        >>> s = pd.Series([2, 4, 2, 2, 4, None])
        >>> s.mode()
        0    2.0
        dtype: float64

        More than one mode:

        >>> s = pd.Series([2, 4, 8, 2, 4, None])
        >>> s.mode()
        0    2.0
        1    4.0
        dtype: float64

        With and without considering null value:

        >>> s = pd.Series([2, 4, None, None, 4, None])
        >>> s.mode(dropna=False)
        0   NaN
        dtype: float64
        >>> s = pd.Series([2, 4, None, None, 4, None])
        >>> s.mode()
        0    4.0
        dtype: float64
        """
        # TODO: Add option for bins like value_counts()
        values = self._values
        if isinstance(values, np.ndarray):
            res_values = algorithms.mode(values, dropna=dropna)
        else:
            res_values = values._mode(dropna=dropna)

        # Ensure index is type stable (should always use int index)
        return self._constructor(
            res_values,
            index=range(len(res_values)),
            name=self.name,
            copy=False,
            dtype=self.dtype,
        ).__finalize__(self, method="mode")

    def unique(self) -> ArrayLike:  # pylint: disable=useless-parent-delegation
        """
        Return unique values of Series object.

        Uniques are returned in order of appearance. Hash table-based unique,
        therefore does NOT sort.

        Returns
        -------
        ndarray or ExtensionArray
            The unique values returned as a NumPy array. See Notes.

        See Also
        --------
        Series.drop_duplicates : Return Series with duplicate values removed.
        unique : Top-level unique method for any 1-d array-like object.
        Index.unique : Return Index with unique values from an Index object.

        Notes
        -----
        Returns the unique values as a NumPy array. In case of an
        extension-array backed Series, a new
        :class:`~api.extensions.ExtensionArray` of that type with just
        the unique values is returned. This includes

            * Categorical
            * Period
            * Datetime with Timezone
            * Datetime without Timezone
            * Timedelta
            * Interval
            * Sparse
            * IntegerNA

        See Examples section.

        Examples
        --------
        >>> pd.Series([2, 1, 3, 3], name='A').unique()
        array([2, 1, 3])

        >>> pd.Series([pd.Timestamp('2016-01-01') for _ in range(3)]).unique()
        <DatetimeArray>
        ['2016-01-01 00:00:00']
        Length: 1, dtype: datetime64[ns]

        >>> pd.Series([pd.Timestamp('2016-01-01', tz='US/Eastern')
        ...            for _ in range(3)]).unique()
        <DatetimeArray>
        ['2016-01-01 00:00:00-05:00']
        Length: 1, dtype: datetime64[ns, US/Eastern]

        An Categorical will return categories in the order of
        appearance and with the same dtype.

        >>> pd.Series(pd.Categorical(list('baabc'))).unique()
        ['b', 'a', 'c']
        Categories (3, object): ['a', 'b', 'c']
        >>> pd.Series(pd.Categorical(list('baabc'), categories=list('abc'),
        ...                          ordered=True)).unique()
        ['b', 'a', 'c']
        Categories (3, object): ['a' < 'b' < 'c']
        """
        return super().unique()

    @overload
    def drop_duplicates(
        self,
        *,
        keep: DropKeep = ...,
        inplace: Literal[False] = ...,
        ignore_index: bool = ...,
    ) -> Series:
        ...

    @overload
    def drop_duplicates(
        self, *, keep: DropKeep = ..., inplace: Literal[True], ignore_index: bool = ...
    ) -> None:
        ...

    @overload
    def drop_duplicates(
        self, *, keep: DropKeep = ..., inplace: bool = ..., ignore_index: bool = ...
    ) -> Series | None:
        ...

    def drop_duplicates(
        self,
        *,
        keep: DropKeep = "first",
        inplace: bool = False,
        ignore_index: bool = False,
    ) -> Series | None:
        """
        Return Series with duplicate values removed.

        Parameters
        ----------
        keep : {'first', 'last', ``False``}, default 'first'
            Method to handle dropping duplicates:

            - 'first' : Drop duplicates except for the first occurrence.
            - 'last' : Drop duplicates except for the last occurrence.
            - ``False`` : Drop all duplicates.

        inplace : bool, default ``False``
            If ``True``, performs operation inplace and returns None.

        ignore_index : bool, default ``False``
            If ``True``, the resulting axis will be labeled 0, 1, …, n - 1.

            .. versionadded:: 2.0.0

        Returns
        -------
        Series or None
            Series with duplicates dropped or None if ``inplace=True``.

        See Also
        --------
        Index.drop_duplicates : Equivalent method on Index.
        DataFrame.drop_duplicates : Equivalent method on DataFrame.
        Series.duplicated : Related method on Series, indicating duplicate
            Series values.
        Series.unique : Return unique values as an array.

        Examples
        --------
        Generate a Series with duplicated entries.

        >>> s = pd.Series(['llama', 'cow', 'llama', 'beetle', 'llama', 'hippo'],
        ...               name='animal')
        >>> s
        0     llama
        1       cow
        2     llama
        3    beetle
        4     llama
        5     hippo
        Name: animal, dtype: object

        With the 'keep' parameter, the selection behaviour of duplicated values
        can be changed. The value 'first' keeps the first occurrence for each
        set of duplicated entries. The default value of keep is 'first'.

        >>> s.drop_duplicates()
        0     llama
        1       cow
        3    beetle
        5     hippo
        Name: animal, dtype: object

        The value 'last' for parameter 'keep' keeps the last occurrence for
        each set of duplicated entries.

        >>> s.drop_duplicates(keep='last')
        1       cow
        3    beetle
        4     llama
        5     hippo
        Name: animal, dtype: object

        The value ``False`` for parameter 'keep' discards all sets of
        duplicated entries.

        >>> s.drop_duplicates(keep=False)
        1       cow
        3    beetle
        5     hippo
        Name: animal, dtype: object
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        result = super().drop_duplicates(keep=keep)

        if ignore_index:
            result.index = default_index(len(result))

        if inplace:
            self._update_inplace(result)
            return None
        else:
            return result

    def duplicated(self, keep: DropKeep = "first") -> Series:
        """
        Indicate duplicate Series values.

        Duplicated values are indicated as ``True`` values in the resulting
        Series. Either all duplicates, all except the first or all except the
        last occurrence of duplicates can be indicated.

        Parameters
        ----------
        keep : {'first', 'last', False}, default 'first'
            Method to handle dropping duplicates:

            - 'first' : Mark duplicates as ``True`` except for the first
              occurrence.
            - 'last' : Mark duplicates as ``True`` except for the last
              occurrence.
            - ``False`` : Mark all duplicates as ``True``.

        Returns
        -------
        Series[bool]
            Series indicating whether each value has occurred in the
            preceding values.

        See Also
        --------
        Index.duplicated : Equivalent method on pandas.Index.
        DataFrame.duplicated : Equivalent method on pandas.DataFrame.
        Series.drop_duplicates : Remove duplicate values from Series.

        Examples
        --------
        By default, for each set of duplicated values, the first occurrence is
        set on False and all others on True:

        >>> animals = pd.Series(['llama', 'cow', 'llama', 'beetle', 'llama'])
        >>> animals.duplicated()
        0    False
        1    False
        2     True
        3    False
        4     True
        dtype: bool

        which is equivalent to

        >>> animals.duplicated(keep='first')
        0    False
        1    False
        2     True
        3    False
        4     True
        dtype: bool

        By using 'last', the last occurrence of each set of duplicated values
        is set on False and all others on True:

        >>> animals.duplicated(keep='last')
        0     True
        1    False
        2     True
        3    False
        4    False
        dtype: bool

        By setting keep on ``False``, all duplicates are True:

        >>> animals.duplicated(keep=False)
        0     True
        1    False
        2     True
        3    False
        4     True
        dtype: bool
        """
        res = self._duplicated(keep=keep)
        result = self._constructor(res, index=self.index, copy=False)
        return result.__finalize__(self, method="duplicated")

    def idxmin(self, axis: Axis = 0, skipna: bool = True, *args, **kwargs) -> Hashable:
        """
        Return the row label of the minimum value.

        If multiple values equal the minimum, the first row label with that
        value is returned.

        Parameters
        ----------
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        skipna : bool, default True
            Exclude NA/null values. If the entire Series is NA, the result
            will be NA.
        *args, **kwargs
            Additional arguments and keywords have no effect but might be
            accepted for compatibility with NumPy.

        Returns
        -------
        Index
            Label of the minimum value.

        Raises
        ------
        ValueError
            If the Series is empty.

        See Also
        --------
        numpy.argmin : Return indices of the minimum values
            along the given axis.
        DataFrame.idxmin : Return index of first occurrence of minimum
            over requested axis.
        Series.idxmax : Return index *label* of the first occurrence
            of maximum of values.

        Notes
        -----
        This method is the Series version of ``ndarray.argmin``. This method
        returns the label of the minimum, while ``ndarray.argmin`` returns
        the position. To get the position, use ``series.values.argmin()``.

        Examples
        --------
        >>> s = pd.Series(data=[1, None, 4, 1],
        ...               index=['A', 'B', 'C', 'D'])
        >>> s
        A    1.0
        B    NaN
        C    4.0
        D    1.0
        dtype: float64

        >>> s.idxmin()
        'A'

        If `skipna` is False and there is an NA value in the data,
        the function returns ``nan``.

        >>> s.idxmin(skipna=False)
        nan
        """
        axis = self._get_axis_number(axis)
        with warnings.catch_warnings():
            # TODO(3.0): this catching/filtering can be removed
            # ignore warning produced by argmin since we will issue a different
            #  warning for idxmin
            warnings.simplefilter("ignore")
            i = self.argmin(axis, skipna, *args, **kwargs)

        if i == -1:
            # GH#43587 give correct NA value for Index.
            warnings.warn(
                f"The behavior of {type(self).__name__}.idxmin with all-NA "
                "values, or any-NA and skipna=False, is deprecated. In a future "
                "version this will raise ValueError",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
            return self.index._na_value
        return self.index[i]

    def idxmax(self, axis: Axis = 0, skipna: bool = True, *args, **kwargs) -> Hashable:
        """
        Return the row label of the maximum value.

        If multiple values equal the maximum, the first row label with that
        value is returned.

        Parameters
        ----------
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        skipna : bool, default True
            Exclude NA/null values. If the entire Series is NA, the result
            will be NA.
        *args, **kwargs
            Additional arguments and keywords have no effect but might be
            accepted for compatibility with NumPy.

        Returns
        -------
        Index
            Label of the maximum value.

        Raises
        ------
        ValueError
            If the Series is empty.

        See Also
        --------
        numpy.argmax : Return indices of the maximum values
            along the given axis.
        DataFrame.idxmax : Return index of first occurrence of maximum
            over requested axis.
        Series.idxmin : Return index *label* of the first occurrence
            of minimum of values.

        Notes
        -----
        This method is the Series version of ``ndarray.argmax``. This method
        returns the label of the maximum, while ``ndarray.argmax`` returns
        the position. To get the position, use ``series.values.argmax()``.

        Examples
        --------
        >>> s = pd.Series(data=[1, None, 4, 3, 4],
        ...               index=['A', 'B', 'C', 'D', 'E'])
        >>> s
        A    1.0
        B    NaN
        C    4.0
        D    3.0
        E    4.0
        dtype: float64

        >>> s.idxmax()
        'C'

        If `skipna` is False and there is an NA value in the data,
        the function returns ``nan``.

        >>> s.idxmax(skipna=False)
        nan
        """
        axis = self._get_axis_number(axis)
        with warnings.catch_warnings():
            # TODO(3.0): this catching/filtering can be removed
            # ignore warning produced by argmax since we will issue a different
            #  warning for argmax
            warnings.simplefilter("ignore")
            i = self.argmax(axis, skipna, *args, **kwargs)

        if i == -1:
            # GH#43587 give correct NA value for Index.
            warnings.warn(
                f"The behavior of {type(self).__name__}.idxmax with all-NA "
                "values, or any-NA and skipna=False, is deprecated. In a future "
                "version this will raise ValueError",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
            return self.index._na_value
        return self.index[i]

    def round(self, decimals: int = 0, *args, **kwargs) -> Series:
        """
        Round each value in a Series to the given number of decimals.

        Parameters
        ----------
        decimals : int, default 0
            Number of decimal places to round to. If decimals is negative,
            it specifies the number of positions to the left of the decimal point.
        *args, **kwargs
            Additional arguments and keywords have no effect but might be
            accepted for compatibility with NumPy.

        Returns
        -------
        Series
            Rounded values of the Series.

        See Also
        --------
        numpy.around : Round values of an np.array.
        DataFrame.round : Round values of a DataFrame.

        Examples
        --------
        >>> s = pd.Series([0.1, 1.3, 2.7])
        >>> s.round()
        0    0.0
        1    1.0
        2    3.0
        dtype: float64
        """
        nv.validate_round(args, kwargs)
        if self.dtype == "object":
            raise TypeError("Expected numeric dtype, got object instead.")
        new_mgr = self._mgr.round(decimals=decimals, using_cow=using_copy_on_write())
        return self._constructor_from_mgr(new_mgr, axes=new_mgr.axes).__finalize__(
            self, method="round"
        )

    @overload
    def quantile(
        self, q: float = ..., interpolation: QuantileInterpolation = ...
    ) -> float:
        ...

    @overload
    def quantile(
        self,
        q: Sequence[float] | AnyArrayLike,
        interpolation: QuantileInterpolation = ...,
    ) -> Series:
        ...

    @overload
    def quantile(
        self,
        q: float | Sequence[float] | AnyArrayLike = ...,
        interpolation: QuantileInterpolation = ...,
    ) -> float | Series:
        ...

    def quantile(
        self,
        q: float | Sequence[float] | AnyArrayLike = 0.5,
        interpolation: QuantileInterpolation = "linear",
    ) -> float | Series:
        """
        Return value at the given quantile.

        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)
            The quantile(s) to compute, which can lie in range: 0 <= q <= 1.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            This optional parameter specifies the interpolation method to use,
            when the desired quantile lies between two data points `i` and `j`:

                * linear: `i + (j - i) * (x-i)/(j-i)`, where `(x-i)/(j-i)` is
                  the fractional part of the index surrounded by `i > j`.
                * lower: `i`.
                * higher: `j`.
                * nearest: `i` or `j` whichever is nearest.
                * midpoint: (`i` + `j`) / 2.

        Returns
        -------
        float or Series
            If ``q`` is an array, a Series will be returned where the
            index is ``q`` and the values are the quantiles, otherwise
            a float will be returned.

        See Also
        --------
        core.window.Rolling.quantile : Calculate the rolling quantile.
        numpy.percentile : Returns the q-th percentile(s) of the array elements.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.quantile(.5)
        2.5
        >>> s.quantile([.25, .5, .75])
        0.25    1.75
        0.50    2.50
        0.75    3.25
        dtype: float64
        """
        validate_percentile(q)

        # We dispatch to DataFrame so that core.internals only has to worry
        #  about 2D cases.
        df = self.to_frame()

        result = df.quantile(q=q, interpolation=interpolation, numeric_only=False)
        if result.ndim == 2:
            result = result.iloc[:, 0]

        if is_list_like(q):
            result.name = self.name
            idx = Index(q, dtype=np.float64)
            return self._constructor(result, index=idx, name=self.name)
        else:
            # scalar
            return result.iloc[0]

    def corr(
        self,
        other: Series,
        method: CorrelationMethod = "pearson",
        min_periods: int | None = None,
    ) -> float:
        """
        Compute correlation with `other` Series, excluding missing values.

        The two `Series` objects are not required to be the same length and will be
        aligned internally before the correlation function is applied.

        Parameters
        ----------
        other : Series
            Series with which to compute the correlation.
        method : {'pearson', 'kendall', 'spearman'} or callable
            Method used to compute correlation:

            - pearson : Standard correlation coefficient
            - kendall : Kendall Tau correlation coefficient
            - spearman : Spearman rank correlation
            - callable: Callable with input two 1d ndarrays and returning a float.

            .. warning::
                Note that the returned matrix from corr will have 1 along the
                diagonals and will be symmetric regardless of the callable's
                behavior.
        min_periods : int, optional
            Minimum number of observations needed to have a valid result.

        Returns
        -------
        float
            Correlation with other.

        See Also
        --------
        DataFrame.corr : Compute pairwise correlation between columns.
        DataFrame.corrwith : Compute pairwise correlation with another
            DataFrame or Series.

        Notes
        -----
        Pearson, Kendall and Spearman correlation are currently computed using pairwise complete observations.

        * `Pearson correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_
        * `Kendall rank correlation coefficient <https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient>`_
        * `Spearman's rank correlation coefficient <https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>`_

        Automatic data alignment: as with all pandas operations, automatic data alignment is performed for this method.
        ``corr()`` automatically considers values with matching indices.

        Examples
        --------
        >>> def histogram_intersection(a, b):
        ...     v = np.minimum(a, b).sum().round(decimals=1)
        ...     return v
        >>> s1 = pd.Series([.2, .0, .6, .2])
        >>> s2 = pd.Series([.3, .6, .0, .1])
        >>> s1.corr(s2, method=histogram_intersection)
        0.3

        Pandas auto-aligns the values with matching indices

        >>> s1 = pd.Series([1, 2, 3], index=[0, 1, 2])
        >>> s2 = pd.Series([1, 2, 3], index=[2, 1, 0])
        >>> s1.corr(s2)
        -1.0
        """  # noqa: E501
        this, other = self.align(other, join="inner", copy=False)
        if len(this) == 0:
            return np.nan

        this_values = this.to_numpy(dtype=float, na_value=np.nan, copy=False)
        other_values = other.to_numpy(dtype=float, na_value=np.nan, copy=False)

        if method in ["pearson", "spearman", "kendall"] or callable(method):
            return nanops.nancorr(
                this_values, other_values, method=method, min_periods=min_periods
            )

        raise ValueError(
            "method must be either 'pearson', "
            "'spearman', 'kendall', or a callable, "
            f"'{method}' was supplied"
        )

    def cov(
        self,
        other: Series,
        min_periods: int | None = None,
        ddof: int | None = 1,
    ) -> float:
        """
        Compute covariance with Series, excluding missing values.

        The two `Series` objects are not required to be the same length and
        will be aligned internally before the covariance is calculated.

        Parameters
        ----------
        other : Series
            Series with which to compute the covariance.
        min_periods : int, optional
            Minimum number of observations needed to have a valid result.
        ddof : int, default 1
            Delta degrees of freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.

        Returns
        -------
        float
            Covariance between Series and other normalized by N-1
            (unbiased estimator).

        See Also
        --------
        DataFrame.cov : Compute pairwise covariance of columns.

        Examples
        --------
        >>> s1 = pd.Series([0.90010907, 0.13484424, 0.62036035])
        >>> s2 = pd.Series([0.12528585, 0.26962463, 0.51111198])
        >>> s1.cov(s2)
        -0.01685762652715874
        """
        this, other = self.align(other, join="inner", copy=False)
        if len(this) == 0:
            return np.nan
        this_values = this.to_numpy(dtype=float, na_value=np.nan, copy=False)
        other_values = other.to_numpy(dtype=float, na_value=np.nan, copy=False)
        return nanops.nancov(
            this_values, other_values, min_periods=min_periods, ddof=ddof
        )

    @doc(
        klass="Series",
        extra_params="",
        other_klass="DataFrame",
        examples=dedent(
            """
        Difference with previous row

        >>> s = pd.Series([1, 1, 2, 3, 5, 8])
        >>> s.diff()
        0    NaN
        1    0.0
        2    1.0
        3    1.0
        4    2.0
        5    3.0
        dtype: float64

        Difference with 3rd previous row

        >>> s.diff(periods=3)
        0    NaN
        1    NaN
        2    NaN
        3    2.0
        4    4.0
        5    6.0
        dtype: float64

        Difference with following row

        >>> s.diff(periods=-1)
        0    0.0
        1   -1.0
        2   -1.0
        3   -2.0
        4   -3.0
        5    NaN
        dtype: float64

        Overflow in input dtype

        >>> s = pd.Series([1, 0], dtype=np.uint8)
        >>> s.diff()
        0      NaN
        1    255.0
        dtype: float64"""
        ),
    )
    def diff(self, periods: int = 1) -> Series:
        """
        First discrete difference of element.

        Calculates the difference of a {klass} element compared with another
        element in the {klass} (default is element in previous row).

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for calculating difference, accepts negative
            values.
        {extra_params}
        Returns
        -------
        {klass}
            First differences of the Series.

        See Also
        --------
        {klass}.pct_change: Percent change over given number of periods.
        {klass}.shift: Shift index by desired number of periods with an
            optional time freq.
        {other_klass}.diff: First discrete difference of object.

        Notes
        -----
        For boolean dtypes, this uses :meth:`operator.xor` rather than
        :meth:`operator.sub`.
        The result is calculated according to current dtype in {klass},
        however dtype of the result is always float64.

        Examples
        --------
        {examples}
        """
        result = algorithms.diff(self._values, periods)
        return self._constructor(result, index=self.index, copy=False).__finalize__(
            self, method="diff"
        )

    def autocorr(self, lag: int = 1) -> float:
        """
        Compute the lag-N autocorrelation.

        This method computes the Pearson correlation between
        the Series and its shifted self.

        Parameters
        ----------
        lag : int, default 1
            Number of lags to apply before performing autocorrelation.

        Returns
        -------
        float
            The Pearson correlation between self and self.shift(lag).

        See Also
        --------
        Series.corr : Compute the correlation between two Series.
        Series.shift : Shift index by desired number of periods.
        DataFrame.corr : Compute pairwise correlation of columns.
        DataFrame.corrwith : Compute pairwise correlation between rows or
            columns of two DataFrame objects.

        Notes
        -----
        If the Pearson correlation is not well defined return 'NaN'.

        Examples
        --------
        >>> s = pd.Series([0.25, 0.5, 0.2, -0.05])
        >>> s.autocorr()  # doctest: +ELLIPSIS
        0.10355...
        >>> s.autocorr(lag=2)  # doctest: +ELLIPSIS
        -0.99999...

        If the Pearson correlation is not well defined, then 'NaN' is returned.

        >>> s = pd.Series([1, 0, 0, 0])
        >>> s.autocorr()
        nan
        """
        return self.corr(cast(Series, self.shift(lag)))

    def dot(self, other: AnyArrayLike) -> Series | np.ndarray:
        """
        Compute the dot product between the Series and the columns of other.

        This method computes the dot product between the Series and another
        one, or the Series and each columns of a DataFrame, or the Series and
        each columns of an array.

        It can also be called using `self @ other`.

        Parameters
        ----------
        other : Series, DataFrame or array-like
            The other object to compute the dot product with its columns.

        Returns
        -------
        scalar, Series or numpy.ndarray
            Return the dot product of the Series and other if other is a
            Series, the Series of the dot product of Series and each rows of
            other if other is a DataFrame or a numpy.ndarray between the Series
            and each columns of the numpy array.

        See Also
        --------
        DataFrame.dot: Compute the matrix product with the DataFrame.
        Series.mul: Multiplication of series and other, element-wise.

        Notes
        -----
        The Series and other has to share the same index if other is a Series
        or a DataFrame.

        Examples
        --------
        >>> s = pd.Series([0, 1, 2, 3])
        >>> other = pd.Series([-1, 2, -3, 4])
        >>> s.dot(other)
        8
        >>> s @ other
        8
        >>> df = pd.DataFrame([[0, 1], [-2, 3], [4, -5], [6, 7]])
        >>> s.dot(df)
        0    24
        1    14
        dtype: int64
        >>> arr = np.array([[0, 1], [-2, 3], [4, -5], [6, 7]])
        >>> s.dot(arr)
        array([24, 14])
        """
        if isinstance(other, (Series, ABCDataFrame)):
            common = self.index.union(other.index)
            if len(common) > len(self.index) or len(common) > len(other.index):
                raise ValueError("matrices are not aligned")

            left = self.reindex(index=common, copy=False)
            right = other.reindex(index=common, copy=False)
            lvals = left.values
            rvals = right.values
        else:
            lvals = self.values
            rvals = np.asarray(other)
            if lvals.shape[0] != rvals.shape[0]:
                raise Exception(
                    f"Dot product shape mismatch, {lvals.shape} vs {rvals.shape}"
                )

        if isinstance(other, ABCDataFrame):
            return self._constructor(
                np.dot(lvals, rvals), index=other.columns, copy=False
            ).__finalize__(self, method="dot")
        elif isinstance(other, Series):
            return np.dot(lvals, rvals)
        elif isinstance(rvals, np.ndarray):
            return np.dot(lvals, rvals)
        else:  # pragma: no cover
            raise TypeError(f"unsupported type: {type(other)}")

    def __matmul__(self, other):
        """
        Matrix multiplication using binary `@` operator.
        """
        return self.dot(other)

    def __rmatmul__(self, other):
        """
        Matrix multiplication using binary `@` operator.
        """
        return self.dot(np.transpose(other))

    @doc(base.IndexOpsMixin.searchsorted, klass="Series")
    # Signature of "searchsorted" incompatible with supertype "IndexOpsMixin"
    def searchsorted(  # type: ignore[override]
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        return base.IndexOpsMixin.searchsorted(self, value, side=side, sorter=sorter)

    # -------------------------------------------------------------------
    # Combination

    def _append(
        self, to_append, ignore_index: bool = False, verify_integrity: bool = False
    ):
        from pandas.core.reshape.concat import concat

        if isinstance(to_append, (list, tuple)):
            to_concat = [self]
            to_concat.extend(to_append)
        else:
            to_concat = [self, to_append]
        if any(isinstance(x, (ABCDataFrame,)) for x in to_concat[1:]):
            msg = "to_append should be a Series or list/tuple of Series, got DataFrame"
            raise TypeError(msg)
        return concat(
            to_concat, ignore_index=ignore_index, verify_integrity=verify_integrity
        )

    @doc(
        _shared_docs["compare"],
        dedent(
            """
        Returns
        -------
        Series or DataFrame
            If axis is 0 or 'index' the result will be a Series.
            The resulting index will be a MultiIndex with 'self' and 'other'
            stacked alternately at the inner level.

            If axis is 1 or 'columns' the result will be a DataFrame.
            It will have two columns namely 'self' and 'other'.

        See Also
        --------
        DataFrame.compare : Compare with another DataFrame and show differences.

        Notes
        -----
        Matching NaNs will not appear as a difference.

        Examples
        --------
        >>> s1 = pd.Series(["a", "b", "c", "d", "e"])
        >>> s2 = pd.Series(["a", "a", "c", "b", "e"])

        Align the differences on columns

        >>> s1.compare(s2)
          self other
        1    b     a
        3    d     b

        Stack the differences on indices

        >>> s1.compare(s2, align_axis=0)
        1  self     b
           other    a
        3  self     d
           other    b
        dtype: object

        Keep all original rows

        >>> s1.compare(s2, keep_shape=True)
          self other
        0  NaN   NaN
        1    b     a
        2  NaN   NaN
        3    d     b
        4  NaN   NaN

        Keep all original rows and also all original values

        >>> s1.compare(s2, keep_shape=True, keep_equal=True)
          self other
        0    a     a
        1    b     a
        2    c     c
        3    d     b
        4    e     e
        """
        ),
        klass=_shared_doc_kwargs["klass"],
    )
    def compare(
        self,
        other: Series,
        align_axis: Axis = 1,
        keep_shape: bool = False,
        keep_equal: bool = False,
        result_names: Suffixes = ("self", "other"),
    ) -> DataFrame | Series:
        return super().compare(
            other=other,
            align_axis=align_axis,
            keep_shape=keep_shape,
            keep_equal=keep_equal,
            result_names=result_names,
        )

    def combine(
        self,
        other: Series | Hashable,
        func: Callable[[Hashable, Hashable], Hashable],
        fill_value: Hashable | None = None,
    ) -> Series:
        """
        Combine the Series with a Series or scalar according to `func`.

        Combine the Series and `other` using `func` to perform elementwise
        selection for combined Series.
        `fill_value` is assumed when value is missing at some index
        from one of the two objects being combined.

        Parameters
        ----------
        other : Series or scalar
            The value(s) to be combined with the `Series`.
        func : function
            Function that takes two scalars as inputs and returns an element.
        fill_value : scalar, optional
            The value to assume when an index is missing from
            one Series or the other. The default specifies to use the
            appropriate NaN value for the underlying dtype of the Series.

        Returns
        -------
        Series
            The result of combining the Series with the other object.

        See Also
        --------
        Series.combine_first : Combine Series values, choosing the calling
            Series' values first.

        Examples
        --------
        Consider 2 Datasets ``s1`` and ``s2`` containing
        highest clocked speeds of different birds.

        >>> s1 = pd.Series({'falcon': 330.0, 'eagle': 160.0})
        >>> s1
        falcon    330.0
        eagle     160.0
        dtype: float64
        >>> s2 = pd.Series({'falcon': 345.0, 'eagle': 200.0, 'duck': 30.0})
        >>> s2
        falcon    345.0
        eagle     200.0
        duck       30.0
        dtype: float64

        Now, to combine the two datasets and view the highest speeds
        of the birds across the two datasets

        >>> s1.combine(s2, max)
        duck        NaN
        eagle     200.0
        falcon    345.0
        dtype: float64

        In the previous example, the resulting value for duck is missing,
        because the maximum of a NaN and a float is a NaN.
        So, in the example, we set ``fill_value=0``,
        so the maximum value returned will be the value from some dataset.

        >>> s1.combine(s2, max, fill_value=0)
        duck       30.0
        eagle     200.0
        falcon    345.0
        dtype: float64
        """
        if fill_value is None:
            fill_value = na_value_for_dtype(self.dtype, compat=False)

        if isinstance(other, Series):
            # If other is a Series, result is based on union of Series,
            # so do this element by element
            new_index = self.index.union(other.index)
            new_name = ops.get_op_result_name(self, other)
            new_values = np.empty(len(new_index), dtype=object)
            with np.errstate(all="ignore"):
                for i, idx in enumerate(new_index):
                    lv = self.get(idx, fill_value)
                    rv = other.get(idx, fill_value)
                    new_values[i] = func(lv, rv)
        else:
            # Assume that other is a scalar, so apply the function for
            # each element in the Series
            new_index = self.index
            new_values = np.empty(len(new_index), dtype=object)
            with np.errstate(all="ignore"):
                new_values[:] = [func(lv, other) for lv in self._values]
            new_name = self.name

        # try_float=False is to match agg_series
        npvalues = lib.maybe_convert_objects(new_values, try_float=False)
        # same_dtype here is a kludge to avoid casting e.g. [True, False] to
        #  ["True", "False"]
        same_dtype = isinstance(self.dtype, (StringDtype, CategoricalDtype))
        res_values = maybe_cast_pointwise_result(
            npvalues, self.dtype, same_dtype=same_dtype
        )
        return self._constructor(res_values, index=new_index, name=new_name, copy=False)

    def combine_first(self, other) -> Series:
        """
        Update null elements with value in the same location in 'other'.

        Combine two Series objects by filling null values in one Series with
        non-null values from the other Series. Result index will be the union
        of the two indexes.

        Parameters
        ----------
        other : Series
            The value(s) to be used for filling null values.

        Returns
        -------
        Series
            The result of combining the provided Series with the other object.

        See Also
        --------
        Series.combine : Perform element-wise operation on two Series
            using a given function.

        Examples
        --------
        >>> s1 = pd.Series([1, np.nan])
        >>> s2 = pd.Series([3, 4, 5])
        >>> s1.combine_first(s2)
        0    1.0
        1    4.0
        2    5.0
        dtype: float64

        Null values still persist if the location of that null value
        does not exist in `other`

        >>> s1 = pd.Series({'falcon': np.nan, 'eagle': 160.0})
        >>> s2 = pd.Series({'eagle': 200.0, 'duck': 30.0})
        >>> s1.combine_first(s2)
        duck       30.0
        eagle     160.0
        falcon      NaN
        dtype: float64
        """
        from pandas.core.reshape.concat import concat

        if self.dtype == other.dtype:
            if self.index.equals(other.index):
                return self.mask(self.isna(), other)
            elif self._can_hold_na and not isinstance(self.dtype, SparseDtype):
                this, other = self.align(other, join="outer")
                return this.mask(this.isna(), other)

        new_index = self.index.union(other.index)

        this = self
        # identify the index subset to keep for each series
        keep_other = other.index.difference(this.index[notna(this)])
        keep_this = this.index.difference(keep_other)

        this = this.reindex(keep_this, copy=False)
        other = other.reindex(keep_other, copy=False)

        if this.dtype.kind == "M" and other.dtype.kind != "M":
            other = to_datetime(other)
        combined = concat([this, other])
        combined = combined.reindex(new_index, copy=False)
        return combined.__finalize__(self, method="combine_first")

    def update(self, other: Series | Sequence | Mapping) -> None:
        """
        Modify Series in place using values from passed Series.

        Uses non-NA values from passed Series to make updates. Aligns
        on index.

        Parameters
        ----------
        other : Series, or object coercible into Series

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.update(pd.Series([4, 5, 6]))
        >>> s
        0    4
        1    5
        2    6
        dtype: int64

        >>> s = pd.Series(['a', 'b', 'c'])
        >>> s.update(pd.Series(['d', 'e'], index=[0, 2]))
        >>> s
        0    d
        1    b
        2    e
        dtype: object

        >>> s = pd.Series([1, 2, 3])
        >>> s.update(pd.Series([4, 5, 6, 7, 8]))
        >>> s
        0    4
        1    5
        2    6
        dtype: int64

        If ``other`` contains NaNs the corresponding values are not updated
        in the original Series.

        >>> s = pd.Series([1, 2, 3])
        >>> s.update(pd.Series([4, np.nan, 6]))
        >>> s
        0    4
        1    2
        2    6
        dtype: int64

        ``other`` can also be a non-Series object type
        that is coercible into a Series

        >>> s = pd.Series([1, 2, 3])
        >>> s.update([4, np.nan, 6])
        >>> s
        0    4
        1    2
        2    6
        dtype: int64

        >>> s = pd.Series([1, 2, 3])
        >>> s.update({1: 9})
        >>> s
        0    1
        1    9
        2    3
        dtype: int64
        """
        if not PYPY and using_copy_on_write():
            if sys.getrefcount(self) <= REF_COUNT:
                warnings.warn(
                    _chained_assignment_method_msg,
                    ChainedAssignmentError,
                    stacklevel=2,
                )
        elif not PYPY and not using_copy_on_write() and self._is_view_after_cow_rules():
            ctr = sys.getrefcount(self)
            ref_count = REF_COUNT
            if _check_cacher(self):
                # see https://github.com/pandas-dev/pandas/pull/56060#discussion_r1399245221
                ref_count += 1
            if ctr <= ref_count:
                warnings.warn(
                    _chained_assignment_warning_method_msg,
                    FutureWarning,
                    stacklevel=2,
                )

        if not isinstance(other, Series):
            other = Series(other)

        other = other.reindex_like(self)
        mask = notna(other)

        self._mgr = self._mgr.putmask(mask=mask, new=other)
        self._maybe_update_cacher()

    # ----------------------------------------------------------------------
    # Reindexing, sorting

    @overload
    def sort_values(
        self,
        *,
        axis: Axis = ...,
        ascending: bool | Sequence[bool] = ...,
        inplace: Literal[False] = ...,
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        ignore_index: bool = ...,
        key: ValueKeyFunc = ...,
    ) -> Series:
        ...

    @overload
    def sort_values(
        self,
        *,
        axis: Axis = ...,
        ascending: bool | Sequence[bool] = ...,
        inplace: Literal[True],
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        ignore_index: bool = ...,
        key: ValueKeyFunc = ...,
    ) -> None:
        ...

    @overload
    def sort_values(
        self,
        *,
        axis: Axis = ...,
        ascending: bool | Sequence[bool] = ...,
        inplace: bool = ...,
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        ignore_index: bool = ...,
        key: ValueKeyFunc = ...,
    ) -> Series | None:
        ...

    def sort_values(
        self,
        *,
        axis: Axis = 0,
        ascending: bool | Sequence[bool] = True,
        inplace: bool = False,
        kind: SortKind = "quicksort",
        na_position: NaPosition = "last",
        ignore_index: bool = False,
        key: ValueKeyFunc | None = None,
    ) -> Series | None:
        """
        Sort by the values.

        Sort a Series in ascending or descending order by some
        criterion.

        Parameters
        ----------
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        ascending : bool or list of bools, default True
            If True, sort values in ascending order, otherwise descending.
        inplace : bool, default False
            If True, perform operation in-place.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, default 'quicksort'
            Choice of sorting algorithm. See also :func:`numpy.sort` for more
            information. 'mergesort' and 'stable' are the only stable  algorithms.
        na_position : {'first' or 'last'}, default 'last'
            Argument 'first' puts NaNs at the beginning, 'last' puts NaNs at
            the end.
        ignore_index : bool, default False
            If True, the resulting axis will be labeled 0, 1, …, n - 1.
        key : callable, optional
            If not None, apply the key function to the series values
            before sorting. This is similar to the `key` argument in the
            builtin :meth:`sorted` function, with the notable difference that
            this `key` function should be *vectorized*. It should expect a
            ``Series`` and return an array-like.

        Returns
        -------
        Series or None
            Series ordered by values or None if ``inplace=True``.

        See Also
        --------
        Series.sort_index : Sort by the Series indices.
        DataFrame.sort_values : Sort DataFrame by the values along either axis.
        DataFrame.sort_index : Sort DataFrame by indices.

        Examples
        --------
        >>> s = pd.Series([np.nan, 1, 3, 10, 5])
        >>> s
        0     NaN
        1     1.0
        2     3.0
        3     10.0
        4     5.0
        dtype: float64

        Sort values ascending order (default behaviour)

        >>> s.sort_values(ascending=True)
        1     1.0
        2     3.0
        4     5.0
        3    10.0
        0     NaN
        dtype: float64

        Sort values descending order

        >>> s.sort_values(ascending=False)
        3    10.0
        4     5.0
        2     3.0
        1     1.0
        0     NaN
        dtype: float64

        Sort values putting NAs first

        >>> s.sort_values(na_position='first')
        0     NaN
        1     1.0
        2     3.0
        4     5.0
        3    10.0
        dtype: float64

        Sort a series of strings

        >>> s = pd.Series(['z', 'b', 'd', 'a', 'c'])
        >>> s
        0    z
        1    b
        2    d
        3    a
        4    c
        dtype: object

        >>> s.sort_values()
        3    a
        1    b
        4    c
        2    d
        0    z
        dtype: object

        Sort using a key function. Your `key` function will be
        given the ``Series`` of values and should return an array-like.

        >>> s = pd.Series(['a', 'B', 'c', 'D', 'e'])
        >>> s.sort_values()
        1    B
        3    D
        0    a
        2    c
        4    e
        dtype: object
        >>> s.sort_values(key=lambda x: x.str.lower())
        0    a
        1    B
        2    c
        3    D
        4    e
        dtype: object

        NumPy ufuncs work well here. For example, we can
        sort by the ``sin`` of the value

        >>> s = pd.Series([-4, -2, 0, 2, 4])
        >>> s.sort_values(key=np.sin)
        1   -2
        4    4
        2    0
        0   -4
        3    2
        dtype: int64

        More complicated user-defined functions can be used,
        as long as they expect a Series and return an array-like

        >>> s.sort_values(key=lambda x: (np.tan(x.cumsum())))
        0   -4
        3    2
        4    4
        1   -2
        2    0
        dtype: int64
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        # Validate the axis parameter
        self._get_axis_number(axis)

        # GH 5856/5853
        if inplace and self._is_cached:
            raise ValueError(
                "This Series is a view of some other array, to "
                "sort in-place you must create a copy"
            )

        if is_list_like(ascending):
            ascending = cast(Sequence[bool], ascending)
            if len(ascending) != 1:
                raise ValueError(
                    f"Length of ascending ({len(ascending)}) must be 1 for Series"
                )
            ascending = ascending[0]

        ascending = validate_ascending(ascending)

        if na_position not in ["first", "last"]:
            raise ValueError(f"invalid na_position: {na_position}")

        # GH 35922. Make sorting stable by leveraging nargsort
        if key:
            values_to_sort = cast(Series, ensure_key_mapped(self, key))._values
        else:
            values_to_sort = self._values
        sorted_index = nargsort(values_to_sort, kind, bool(ascending), na_position)

        if is_range_indexer(sorted_index, len(sorted_index)):
            if inplace:
                return self._update_inplace(self)
            return self.copy(deep=None)

        result = self._constructor(
            self._values[sorted_index], index=self.index[sorted_index], copy=False
        )

        if ignore_index:
            result.index = default_index(len(sorted_index))

        if not inplace:
            return result.__finalize__(self, method="sort_values")
        self._update_inplace(result)
        return None

    @overload
    def sort_index(
        self,
        *,
        axis: Axis = ...,
        level: IndexLabel = ...,
        ascending: bool | Sequence[bool] = ...,
        inplace: Literal[True],
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        sort_remaining: bool = ...,
        ignore_index: bool = ...,
        key: IndexKeyFunc = ...,
    ) -> None:
        ...

    @overload
    def sort_index(
        self,
        *,
        axis: Axis = ...,
        level: IndexLabel = ...,
        ascending: bool | Sequence[bool] = ...,
        inplace: Literal[False] = ...,
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        sort_remaining: bool = ...,
        ignore_index: bool = ...,
        key: IndexKeyFunc = ...,
    ) -> Series:
        ...

    @overload
    def sort_index(
        self,
        *,
        axis: Axis = ...,
        level: IndexLabel = ...,
        ascending: bool | Sequence[bool] = ...,
        inplace: bool = ...,
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        sort_remaining: bool = ...,
        ignore_index: bool = ...,
        key: IndexKeyFunc = ...,
    ) -> Series | None:
        ...

    def sort_index(
        self,
        *,
        axis: Axis = 0,
        level: IndexLabel | None = None,
        ascending: bool | Sequence[bool] = True,
        inplace: bool = False,
        kind: SortKind = "quicksort",
        na_position: NaPosition = "last",
        sort_remaining: bool = True,
        ignore_index: bool = False,
        key: IndexKeyFunc | None = None,
    ) -> Series | None:
        """
        Sort Series by index labels.

        Returns a new Series sorted by label if `inplace` argument is
        ``False``, otherwise updates the original series and returns None.

        Parameters
        ----------
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        level : int, optional
            If not None, sort on values in specified index level(s).
        ascending : bool or list-like of bools, default True
            Sort ascending vs. descending. When the index is a MultiIndex the
            sort direction can be controlled for each level individually.
        inplace : bool, default False
            If True, perform operation in-place.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, default 'quicksort'
            Choice of sorting algorithm. See also :func:`numpy.sort` for more
            information. 'mergesort' and 'stable' are the only stable algorithms. For
            DataFrames, this option is only applied when sorting on a single
            column or label.
        na_position : {'first', 'last'}, default 'last'
            If 'first' puts NaNs at the beginning, 'last' puts NaNs at the end.
            Not implemented for MultiIndex.
        sort_remaining : bool, default True
            If True and sorting by level and index is multilevel, sort by other
            levels too (in order) after sorting by specified level.
        ignore_index : bool, default False
            If True, the resulting axis will be labeled 0, 1, …, n - 1.
        key : callable, optional
            If not None, apply the key function to the index values
            before sorting. This is similar to the `key` argument in the
            builtin :meth:`sorted` function, with the notable difference that
            this `key` function should be *vectorized*. It should expect an
            ``Index`` and return an ``Index`` of the same shape.

        Returns
        -------
        Series or None
            The original Series sorted by the labels or None if ``inplace=True``.

        See Also
        --------
        DataFrame.sort_index: Sort DataFrame by the index.
        DataFrame.sort_values: Sort DataFrame by the value.
        Series.sort_values : Sort Series by the value.

        Examples
        --------
        >>> s = pd.Series(['a', 'b', 'c', 'd'], index=[3, 2, 1, 4])
        >>> s.sort_index()
        1    c
        2    b
        3    a
        4    d
        dtype: object

        Sort Descending

        >>> s.sort_index(ascending=False)
        4    d
        3    a
        2    b
        1    c
        dtype: object

        By default NaNs are put at the end, but use `na_position` to place
        them at the beginning

        >>> s = pd.Series(['a', 'b', 'c', 'd'], index=[3, 2, 1, np.nan])
        >>> s.sort_index(na_position='first')
        NaN     d
         1.0    c
         2.0    b
         3.0    a
        dtype: object

        Specify index level to sort

        >>> arrays = [np.array(['qux', 'qux', 'foo', 'foo',
        ...                     'baz', 'baz', 'bar', 'bar']),
        ...           np.array(['two', 'one', 'two', 'one',
        ...                     'two', 'one', 'two', 'one'])]
        >>> s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=arrays)
        >>> s.sort_index(level=1)
        bar  one    8
        baz  one    6
        foo  one    4
        qux  one    2
        bar  two    7
        baz  two    5
        foo  two    3
        qux  two    1
        dtype: int64

        Does not sort by remaining levels when sorting by levels

        >>> s.sort_index(level=1, sort_remaining=False)
        qux  one    2
        foo  one    4
        baz  one    6
        bar  one    8
        qux  two    1
        foo  two    3
        baz  two    5
        bar  two    7
        dtype: int64

        Apply a key function before sorting

        >>> s = pd.Series([1, 2, 3, 4], index=['A', 'b', 'C', 'd'])
        >>> s.sort_index(key=lambda x : x.str.lower())
        A    1
        b    2
        C    3
        d    4
        dtype: int64
        """

        return super().sort_index(
            axis=axis,
            level=level,
            ascending=ascending,
            inplace=inplace,
            kind=kind,
            na_position=na_position,
            sort_remaining=sort_remaining,
            ignore_index=ignore_index,
            key=key,
        )

    def argsort(
        self,
        axis: Axis = 0,
        kind: SortKind = "quicksort",
        order: None = None,
        stable: None = None,
    ) -> Series:
        """
        Return the integer indices that would sort the Series values.

        Override ndarray.argsort. Argsorts the value, omitting NA/null values,
        and places the result in the same locations as the non-NA values.

        Parameters
        ----------
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        kind : {'mergesort', 'quicksort', 'heapsort', 'stable'}, default 'quicksort'
            Choice of sorting algorithm. See :func:`numpy.sort` for more
            information. 'mergesort' and 'stable' are the only stable algorithms.
        order : None
            Has no effect but is accepted for compatibility with numpy.
        stable : None
            Has no effect but is accepted for compatibility with numpy.

        Returns
        -------
        Series[np.intp]
            Positions of values within the sort order with -1 indicating
            nan values.

        See Also
        --------
        numpy.ndarray.argsort : Returns the indices that would sort this array.

        Examples
        --------
        >>> s = pd.Series([3, 2, 1])
        >>> s.argsort()
        0    2
        1    1
        2    0
        dtype: int64
        """
        if axis != -1:
            # GH#54257 We allow -1 here so that np.argsort(series) works
            self._get_axis_number(axis)

        values = self._values
        mask = isna(values)

        if mask.any():
            # TODO(3.0): once this deprecation is enforced we can call
            #  self.array.argsort directly, which will close GH#43840 and
            #  GH#12694
            warnings.warn(
                "The behavior of Series.argsort in the presence of NA values is "
                "deprecated. In a future version, NA values will be ordered "
                "last instead of set to -1.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
            result = np.full(len(self), -1, dtype=np.intp)
            notmask = ~mask
            result[notmask] = np.argsort(values[notmask], kind=kind)
        else:
            result = np.argsort(values, kind=kind)

        res = self._constructor(
            result, index=self.index, name=self.name, dtype=np.intp, copy=False
        )
        return res.__finalize__(self, method="argsort")

    def nlargest(
        self, n: int = 5, keep: Literal["first", "last", "all"] = "first"
    ) -> Series:
        """
        Return the largest `n` elements.

        Parameters
        ----------
        n : int, default 5
            Return this many descending sorted values.
        keep : {'first', 'last', 'all'}, default 'first'
            When there are duplicate values that cannot all fit in a
            Series of `n` elements:

            - ``first`` : return the first `n` occurrences in order
              of appearance.
            - ``last`` : return the last `n` occurrences in reverse
              order of appearance.
            - ``all`` : keep all occurrences. This can result in a Series of
              size larger than `n`.

        Returns
        -------
        Series
            The `n` largest values in the Series, sorted in decreasing order.

        See Also
        --------
        Series.nsmallest: Get the `n` smallest elements.
        Series.sort_values: Sort Series by values.
        Series.head: Return the first `n` rows.

        Notes
        -----
        Faster than ``.sort_values(ascending=False).head(n)`` for small `n`
        relative to the size of the ``Series`` object.

        Examples
        --------
        >>> countries_population = {"Italy": 59000000, "France": 65000000,
        ...                         "Malta": 434000, "Maldives": 434000,
        ...                         "Brunei": 434000, "Iceland": 337000,
        ...                         "Nauru": 11300, "Tuvalu": 11300,
        ...                         "Anguilla": 11300, "Montserrat": 5200}
        >>> s = pd.Series(countries_population)
        >>> s
        Italy       59000000
        France      65000000
        Malta         434000
        Maldives      434000
        Brunei        434000
        Iceland       337000
        Nauru          11300
        Tuvalu         11300
        Anguilla       11300
        Montserrat      5200
        dtype: int64

        The `n` largest elements where ``n=5`` by default.

        >>> s.nlargest()
        France      65000000
        Italy       59000000
        Malta         434000
        Maldives      434000
        Brunei        434000
        dtype: int64

        The `n` largest elements where ``n=3``. Default `keep` value is 'first'
        so Malta will be kept.

        >>> s.nlargest(3)
        France    65000000
        Italy     59000000
        Malta       434000
        dtype: int64

        The `n` largest elements where ``n=3`` and keeping the last duplicates.
        Brunei will be kept since it is the last with value 434000 based on
        the index order.

        >>> s.nlargest(3, keep='last')
        France      65000000
        Italy       59000000
        Brunei        434000
        dtype: int64

        The `n` largest elements where ``n=3`` with all duplicates kept. Note
        that the returned Series has five elements due to the three duplicates.

        >>> s.nlargest(3, keep='all')
        France      65000000
        Italy       59000000
        Malta         434000
        Maldives      434000
        Brunei        434000
        dtype: int64
        """
        return selectn.SelectNSeries(self, n=n, keep=keep).nlargest()

    def nsmallest(
        self, n: int = 5, keep: Literal["first", "last", "all"] = "first"
    ) -> Series:
        """
        Return the smallest `n` elements.

        Parameters
        ----------
        n : int, default 5
            Return this many ascending sorted values.
        keep : {'first', 'last', 'all'}, default 'first'
            When there are duplicate values that cannot all fit in a
            Series of `n` elements:

            - ``first`` : return the first `n` occurrences in order
              of appearance.
            - ``last`` : return the last `n` occurrences in reverse
              order of appearance.
            - ``all`` : keep all occurrences. This can result in a Series of
              size larger than `n`.

        Returns
        -------
        Series
            The `n` smallest values in the Series, sorted in increasing order.

        See Also
        --------
        Series.nlargest: Get the `n` largest elements.
        Series.sort_values: Sort Series by values.
        Series.head: Return the first `n` rows.

        Notes
        -----
        Faster than ``.sort_values().head(n)`` for small `n` relative to
        the size of the ``Series`` object.

        Examples
        --------
        >>> countries_population = {"Italy": 59000000, "France": 65000000,
        ...                         "Brunei": 434000, "Malta": 434000,
        ...                         "Maldives": 434000, "Iceland": 337000,
        ...                         "Nauru": 11300, "Tuvalu": 11300,
        ...                         "Anguilla": 11300, "Montserrat": 5200}
        >>> s = pd.Series(countries_population)
        >>> s
        Italy       59000000
        France      65000000
        Brunei        434000
        Malta         434000
        Maldives      434000
        Iceland       337000
        Nauru          11300
        Tuvalu         11300
        Anguilla       11300
        Montserrat      5200
        dtype: int64

        The `n` smallest elements where ``n=5`` by default.

        >>> s.nsmallest()
        Montserrat    5200
        Nauru        11300
        Tuvalu       11300
        Anguilla     11300
        Iceland     337000
        dtype: int64

        The `n` smallest elements where ``n=3``. Default `keep` value is
        'first' so Nauru and Tuvalu will be kept.

        >>> s.nsmallest(3)
        Montserrat   5200
        Nauru       11300
        Tuvalu      11300
        dtype: int64

        The `n` smallest elements where ``n=3`` and keeping the last
        duplicates. Anguilla and Tuvalu will be kept since they are the last
        with value 11300 based on the index order.

        >>> s.nsmallest(3, keep='last')
        Montserrat   5200
        Anguilla    11300
        Tuvalu      11300
        dtype: int64

        The `n` smallest elements where ``n=3`` with all duplicates kept. Note
        that the returned Series has four elements due to the three duplicates.

        >>> s.nsmallest(3, keep='all')
        Montserrat   5200
        Nauru       11300
        Tuvalu      11300
        Anguilla    11300
        dtype: int64
        """
        return selectn.SelectNSeries(self, n=n, keep=keep).nsmallest()

    @doc(
        klass=_shared_doc_kwargs["klass"],
        extra_params=dedent(
            """copy : bool, default True
            Whether to copy underlying data.

            .. note::
                The `copy` keyword will change behavior in pandas 3.0.
                `Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                will be enabled by default, which means that all methods with a
                `copy` keyword will use a lazy copy mechanism to defer the copy and
                ignore the `copy` keyword. The `copy` keyword will be removed in a
                future version of pandas.

                You can already get the future behavior and improvements through
                enabling copy on write ``pd.options.mode.copy_on_write = True``"""
        ),
        examples=dedent(
            """\
        Examples
        --------
        >>> s = pd.Series(
        ...     ["A", "B", "A", "C"],
        ...     index=[
        ...         ["Final exam", "Final exam", "Coursework", "Coursework"],
        ...         ["History", "Geography", "History", "Geography"],
        ...         ["January", "February", "March", "April"],
        ...     ],
        ... )
        >>> s
        Final exam  History     January      A
                    Geography   February     B
        Coursework  History     March        A
                    Geography   April        C
        dtype: object

        In the following example, we will swap the levels of the indices.
        Here, we will swap the levels column-wise, but levels can be swapped row-wise
        in a similar manner. Note that column-wise is the default behaviour.
        By not supplying any arguments for i and j, we swap the last and second to
        last indices.

        >>> s.swaplevel()
        Final exam  January     History         A
                    February    Geography       B
        Coursework  March       History         A
                    April       Geography       C
        dtype: object

        By supplying one argument, we can choose which index to swap the last
        index with. We can for example swap the first index with the last one as
        follows.

        >>> s.swaplevel(0)
        January     History     Final exam      A
        February    Geography   Final exam      B
        March       History     Coursework      A
        April       Geography   Coursework      C
        dtype: object

        We can also define explicitly which indices we want to swap by supplying values
        for both i and j. Here, we for example swap the first and second indices.

        >>> s.swaplevel(0, 1)
        History     Final exam  January         A
        Geography   Final exam  February        B
        History     Coursework  March           A
        Geography   Coursework  April           C
        dtype: object"""
        ),
    )
    def swaplevel(
        self, i: Level = -2, j: Level = -1, copy: bool | None = None
    ) -> Series:
        """
        Swap levels i and j in a :class:`MultiIndex`.

        Default is to swap the two innermost levels of the index.

        Parameters
        ----------
        i, j : int or str
            Levels of the indices to be swapped. Can pass level name as string.
        {extra_params}

        Returns
        -------
        {klass}
            {klass} with levels swapped in MultiIndex.

        {examples}
        """
        assert isinstance(self.index, MultiIndex)
        result = self.copy(deep=copy and not using_copy_on_write())
        result.index = self.index.swaplevel(i, j)
        return result

    def reorder_levels(self, order: Sequence[Level]) -> Series:
        """
        Rearrange index levels using input order.

        May not drop or duplicate levels.

        Parameters
        ----------
        order : list of int representing new level order
            Reference level by number or key.

        Returns
        -------
        type of caller (new object)

        Examples
        --------
        >>> arrays = [np.array(["dog", "dog", "cat", "cat", "bird", "bird"]),
        ...           np.array(["white", "black", "white", "black", "white", "black"])]
        >>> s = pd.Series([1, 2, 3, 3, 5, 2], index=arrays)
        >>> s
        dog   white    1
              black    2
        cat   white    3
              black    3
        bird  white    5
              black    2
        dtype: int64
        >>> s.reorder_levels([1, 0])
        white  dog     1
        black  dog     2
        white  cat     3
        black  cat     3
        white  bird    5
        black  bird    2
        dtype: int64
        """
        if not isinstance(self.index, MultiIndex):  # pragma: no cover
            raise Exception("Can only reorder levels on a hierarchical axis.")

        result = self.copy(deep=None)
        assert isinstance(result.index, MultiIndex)
        result.index = result.index.reorder_levels(order)
        return result

    def explode(self, ignore_index: bool = False) -> Series:
        """
        Transform each element of a list-like to a row.

        Parameters
        ----------
        ignore_index : bool, default False
            If True, the resulting index will be labeled 0, 1, …, n - 1.

        Returns
        -------
        Series
            Exploded lists to rows; index will be duplicated for these rows.

        See Also
        --------
        Series.str.split : Split string values on specified separator.
        Series.unstack : Unstack, a.k.a. pivot, Series with MultiIndex
            to produce DataFrame.
        DataFrame.melt : Unpivot a DataFrame from wide format to long format.
        DataFrame.explode : Explode a DataFrame from list-like
            columns to long format.

        Notes
        -----
        This routine will explode list-likes including lists, tuples, sets,
        Series, and np.ndarray. The result dtype of the subset rows will
        be object. Scalars will be returned unchanged, and empty list-likes will
        result in a np.nan for that row. In addition, the ordering of elements in
        the output will be non-deterministic when exploding sets.

        Reference :ref:`the user guide <reshaping.explode>` for more examples.

        Examples
        --------
        >>> s = pd.Series([[1, 2, 3], 'foo', [], [3, 4]])
        >>> s
        0    [1, 2, 3]
        1          foo
        2           []
        3       [3, 4]
        dtype: object

        >>> s.explode()
        0      1
        0      2
        0      3
        1    foo
        2    NaN
        3      3
        3      4
        dtype: object
        """
        if isinstance(self.dtype, ExtensionDtype):
            values, counts = self._values._explode()
        elif len(self) and is_object_dtype(self.dtype):
            values, counts = reshape.explode(np.asarray(self._values))
        else:
            result = self.copy()
            return result.reset_index(drop=True) if ignore_index else result

        if ignore_index:
            index: Index = default_index(len(values))
        else:
            index = self.index.repeat(counts)

        return self._constructor(values, index=index, name=self.name, copy=False)

    def unstack(
        self,
        level: IndexLabel = -1,
        fill_value: Hashable | None = None,
        sort: bool = True,
    ) -> DataFrame:
        """
        Unstack, also known as pivot, Series with MultiIndex to produce DataFrame.

        Parameters
        ----------
        level : int, str, or list of these, default last level
            Level(s) to unstack, can pass level name.
        fill_value : scalar value, default None
            Value to use when replacing NaN values.
        sort : bool, default True
            Sort the level(s) in the resulting MultiIndex columns.

        Returns
        -------
        DataFrame
            Unstacked Series.

        Notes
        -----
        Reference :ref:`the user guide <reshaping.stacking>` for more examples.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4],
        ...               index=pd.MultiIndex.from_product([['one', 'two'],
        ...                                                 ['a', 'b']]))
        >>> s
        one  a    1
             b    2
        two  a    3
             b    4
        dtype: int64

        >>> s.unstack(level=-1)
             a  b
        one  1  2
        two  3  4

        >>> s.unstack(level=0)
           one  two
        a    1    3
        b    2    4
        """
        from pandas.core.reshape.reshape import unstack

        return unstack(self, level, fill_value, sort)

    # ----------------------------------------------------------------------
    # function application

    def map(
        self,
        arg: Callable | Mapping | Series,
        na_action: Literal["ignore"] | None = None,
    ) -> Series:
        """
        Map values of Series according to an input mapping or function.

        Used for substituting each value in a Series with another value,
        that may be derived from a function, a ``dict`` or
        a :class:`Series`.

        Parameters
        ----------
        arg : function, collections.abc.Mapping subclass or Series
            Mapping correspondence.
        na_action : {None, 'ignore'}, default None
            If 'ignore', propagate NaN values, without passing them to the
            mapping correspondence.

        Returns
        -------
        Series
            Same index as caller.

        See Also
        --------
        Series.apply : For applying more complex functions on a Series.
        Series.replace: Replace values given in `to_replace` with `value`.
        DataFrame.apply : Apply a function row-/column-wise.
        DataFrame.map : Apply a function elementwise on a whole DataFrame.

        Notes
        -----
        When ``arg`` is a dictionary, values in Series that are not in the
        dictionary (as keys) are converted to ``NaN``. However, if the
        dictionary is a ``dict`` subclass that defines ``__missing__`` (i.e.
        provides a method for default values), then this default is used
        rather than ``NaN``.

        Examples
        --------
        >>> s = pd.Series(['cat', 'dog', np.nan, 'rabbit'])
        >>> s
        0      cat
        1      dog
        2      NaN
        3   rabbit
        dtype: object

        ``map`` accepts a ``dict`` or a ``Series``. Values that are not found
        in the ``dict`` are converted to ``NaN``, unless the dict has a default
        value (e.g. ``defaultdict``):

        >>> s.map({'cat': 'kitten', 'dog': 'puppy'})
        0   kitten
        1    puppy
        2      NaN
        3      NaN
        dtype: object

        It also accepts a function:

        >>> s.map('I am a {}'.format)
        0       I am a cat
        1       I am a dog
        2       I am a nan
        3    I am a rabbit
        dtype: object

        To avoid applying the function to missing values (and keep them as
        ``NaN``) ``na_action='ignore'`` can be used:

        >>> s.map('I am a {}'.format, na_action='ignore')
        0     I am a cat
        1     I am a dog
        2            NaN
        3  I am a rabbit
        dtype: object
        """
        new_values = self._map_values(arg, na_action=na_action)
        return self._constructor(new_values, index=self.index, copy=False).__finalize__(
            self, method="map"
        )

    def _gotitem(self, key, ndim, subset=None) -> Self:
        """
        Sub-classes to define. Return a sliced object.

        Parameters
        ----------
        key : string / list of selections
        ndim : {1, 2}
            Requested ndim of result.
        subset : object, default None
            Subset to act on.
        """
        return self

    _agg_see_also_doc = dedent(
        """
    See Also
    --------
    Series.apply : Invoke function on a Series.
    Series.transform : Transform function producing a Series with like indexes.
    """
    )

    _agg_examples_doc = dedent(
        """
    Examples
    --------
    >>> s = pd.Series([1, 2, 3, 4])
    >>> s
    0    1
    1    2
    2    3
    3    4
    dtype: int64

    >>> s.agg('min')
    1

    >>> s.agg(['min', 'max'])
    min   1
    max   4
    dtype: int64
    """
    )

    @doc(
        _shared_docs["aggregate"],
        klass=_shared_doc_kwargs["klass"],
        axis=_shared_doc_kwargs["axis"],
        see_also=_agg_see_also_doc,
        examples=_agg_examples_doc,
    )
    def aggregate(self, func=None, axis: Axis = 0, *args, **kwargs):
        # Validate the axis parameter
        self._get_axis_number(axis)

        # if func is None, will switch to user-provided "named aggregation" kwargs
        if func is None:
            func = dict(kwargs.items())

        op = SeriesApply(self, func, args=args, kwargs=kwargs)
        result = op.agg()
        return result

    agg = aggregate

    @doc(
        _shared_docs["transform"],
        klass=_shared_doc_kwargs["klass"],
        axis=_shared_doc_kwargs["axis"],
    )
    def transform(
        self, func: AggFuncType, axis: Axis = 0, *args, **kwargs
    ) -> DataFrame | Series:
        # Validate axis argument
        self._get_axis_number(axis)
        ser = (
            self.copy(deep=False)
            if using_copy_on_write() or warn_copy_on_write()
            else self
        )
        result = SeriesApply(ser, func=func, args=args, kwargs=kwargs).transform()
        return result

    def apply(
        self,
        func: AggFuncType,
        convert_dtype: bool | lib.NoDefault = lib.no_default,
        args: tuple[Any, ...] = (),
        *,
        by_row: Literal[False, "compat"] = "compat",
        **kwargs,
    ) -> DataFrame | Series:
        """
        Invoke function on values of Series.

        Can be ufunc (a NumPy function that applies to the entire Series)
        or a Python function that only works on single values.

        Parameters
        ----------
        func : function
            Python function or NumPy ufunc to apply.
        convert_dtype : bool, default True
            Try to find better dtype for elementwise function results. If
            False, leave as dtype=object. Note that the dtype is always
            preserved for some extension array dtypes, such as Categorical.

            .. deprecated:: 2.1.0
                ``convert_dtype`` has been deprecated. Do ``ser.astype(object).apply()``
                instead if you want ``convert_dtype=False``.
        args : tuple
            Positional arguments passed to func after the series value.
        by_row : False or "compat", default "compat"
            If ``"compat"`` and func is a callable, func will be passed each element of
            the Series, like ``Series.map``. If func is a list or dict of
            callables, will first try to translate each func into pandas methods. If
            that doesn't work, will try call to apply again with ``by_row="compat"``
            and if that fails, will call apply again with ``by_row=False``
            (backward compatible).
            If False, the func will be passed the whole Series at once.

            ``by_row`` has no effect when ``func`` is a string.

            .. versionadded:: 2.1.0
        **kwargs
            Additional keyword arguments passed to func.

        Returns
        -------
        Series or DataFrame
            If func returns a Series object the result will be a DataFrame.

        See Also
        --------
        Series.map: For element-wise operations.
        Series.agg: Only perform aggregating type operations.
        Series.transform: Only perform transforming type operations.

        Notes
        -----
        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        Examples
        --------
        Create a series with typical summer temperatures for each city.

        >>> s = pd.Series([20, 21, 12],
        ...               index=['London', 'New York', 'Helsinki'])
        >>> s
        London      20
        New York    21
        Helsinki    12
        dtype: int64

        Square the values by defining a function and passing it as an
        argument to ``apply()``.

        >>> def square(x):
        ...     return x ** 2
        >>> s.apply(square)
        London      400
        New York    441
        Helsinki    144
        dtype: int64

        Square the values by passing an anonymous function as an
        argument to ``apply()``.

        >>> s.apply(lambda x: x ** 2)
        London      400
        New York    441
        Helsinki    144
        dtype: int64

        Define a custom function that needs additional positional
        arguments and pass these additional arguments using the
        ``args`` keyword.

        >>> def subtract_custom_value(x, custom_value):
        ...     return x - custom_value

        >>> s.apply(subtract_custom_value, args=(5,))
        London      15
        New York    16
        Helsinki     7
        dtype: int64

        Define a custom function that takes keyword arguments
        and pass these arguments to ``apply``.

        >>> def add_custom_values(x, **kwargs):
        ...     for month in kwargs:
        ...         x += kwargs[month]
        ...     return x

        >>> s.apply(add_custom_values, june=30, july=20, august=25)
        London      95
        New York    96
        Helsinki    87
        dtype: int64

        Use a function from the Numpy library.

        >>> s.apply(np.log)
        London      2.995732
        New York    3.044522
        Helsinki    2.484907
        dtype: float64
        """
        return SeriesApply(
            self,
            func,
            convert_dtype=convert_dtype,
            by_row=by_row,
            args=args,
            kwargs=kwargs,
        ).apply()

    def _reindex_indexer(
        self,
        new_index: Index | None,
        indexer: npt.NDArray[np.intp] | None,
        copy: bool | None,
    ) -> Series:
        # Note: new_index is None iff indexer is None
        # if not None, indexer is np.intp
        if indexer is None and (
            new_index is None or new_index.names == self.index.names
        ):
            if using_copy_on_write():
                return self.copy(deep=copy)
            if copy or copy is None:
                return self.copy(deep=copy)
            return self

        new_values = algorithms.take_nd(
            self._values, indexer, allow_fill=True, fill_value=None
        )
        return self._constructor(new_values, index=new_index, copy=False)

    def _needs_reindex_multi(self, axes, method, level) -> bool:
        """
        Check if we do need a multi reindex; this is for compat with
        higher dims.
        """
        return False

    @overload
    def rename(
        self,
        index: Renamer | Hashable | None = ...,
        *,
        axis: Axis | None = ...,
        copy: bool = ...,
        inplace: Literal[True],
        level: Level | None = ...,
        errors: IgnoreRaise = ...,
    ) -> None:
        ...

    @overload
    def rename(
        self,
        index: Renamer | Hashable | None = ...,
        *,
        axis: Axis | None = ...,
        copy: bool = ...,
        inplace: Literal[False] = ...,
        level: Level | None = ...,
        errors: IgnoreRaise = ...,
    ) -> Series:
        ...

    @overload
    def rename(
        self,
        index: Renamer | Hashable | None = ...,
        *,
        axis: Axis | None = ...,
        copy: bool = ...,
        inplace: bool = ...,
        level: Level | None = ...,
        errors: IgnoreRaise = ...,
    ) -> Series | None:
        ...

    def rename(
        self,
        index: Renamer | Hashable | None = None,
        *,
        axis: Axis | None = None,
        copy: bool | None = None,
        inplace: bool = False,
        level: Level | None = None,
        errors: IgnoreRaise = "ignore",
    ) -> Series | None:
        """
        Alter Series index labels or name.

        Function / dict values must be unique (1-to-1). Labels not contained in
        a dict / Series will be left as-is. Extra labels listed don't throw an
        error.

        Alternatively, change ``Series.name`` with a scalar value.

        See the :ref:`user guide <basics.rename>` for more.

        Parameters
        ----------
        index : scalar, hashable sequence, dict-like or function optional
            Functions or dict-like are transformations to apply to
            the index.
            Scalar or hashable sequence-like will alter the ``Series.name``
            attribute.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        copy : bool, default True
            Also copy underlying data.

            .. note::
                The `copy` keyword will change behavior in pandas 3.0.
                `Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                will be enabled by default, which means that all methods with a
                `copy` keyword will use a lazy copy mechanism to defer the copy and
                ignore the `copy` keyword. The `copy` keyword will be removed in a
                future version of pandas.

                You can already get the future behavior and improvements through
                enabling copy on write ``pd.options.mode.copy_on_write = True``
        inplace : bool, default False
            Whether to return a new Series. If True the value of copy is ignored.
        level : int or level name, default None
            In case of MultiIndex, only rename labels in the specified level.
        errors : {'ignore', 'raise'}, default 'ignore'
            If 'raise', raise `KeyError` when a `dict-like mapper` or
            `index` contains labels that are not present in the index being transformed.
            If 'ignore', existing keys will be renamed and extra keys will be ignored.

        Returns
        -------
        Series or None
            Series with index labels or name altered or None if ``inplace=True``.

        See Also
        --------
        DataFrame.rename : Corresponding DataFrame method.
        Series.rename_axis : Set the name of the axis.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s
        0    1
        1    2
        2    3
        dtype: int64
        >>> s.rename("my_name")  # scalar, changes Series.name
        0    1
        1    2
        2    3
        Name: my_name, dtype: int64
        >>> s.rename(lambda x: x ** 2)  # function, changes labels
        0    1
        1    2
        4    3
        dtype: int64
        >>> s.rename({1: 3, 2: 5})  # mapping, changes labels
        0    1
        3    2
        5    3
        dtype: int64
        """
        if axis is not None:
            # Make sure we raise if an invalid 'axis' is passed.
            axis = self._get_axis_number(axis)

        if callable(index) or is_dict_like(index):
            # error: Argument 1 to "_rename" of "NDFrame" has incompatible
            # type "Union[Union[Mapping[Any, Hashable], Callable[[Any],
            # Hashable]], Hashable, None]"; expected "Union[Mapping[Any,
            # Hashable], Callable[[Any], Hashable], None]"
            return super()._rename(
                index,  # type: ignore[arg-type]
                copy=copy,
                inplace=inplace,
                level=level,
                errors=errors,
            )
        else:
            return self._set_name(index, inplace=inplace, deep=copy)

    @Appender(
        """
        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s
        0    1
        1    2
        2    3
        dtype: int64

        >>> s.set_axis(['a', 'b', 'c'], axis=0)
        a    1
        b    2
        c    3
        dtype: int64
    """
    )
    @Substitution(
        klass=_shared_doc_kwargs["klass"],
        axes_single_arg=_shared_doc_kwargs["axes_single_arg"],
        extended_summary_sub="",
        axis_description_sub="",
        see_also_sub="",
    )
    @Appender(NDFrame.set_axis.__doc__)
    def set_axis(
        self,
        labels,
        *,
        axis: Axis = 0,
        copy: bool | None = None,
    ) -> Series:
        return super().set_axis(labels, axis=axis, copy=copy)

    # error: Cannot determine type of 'reindex'
    @doc(
        NDFrame.reindex,  # type: ignore[has-type]
        klass=_shared_doc_kwargs["klass"],
        optional_reindex=_shared_doc_kwargs["optional_reindex"],
    )
    def reindex(  # type: ignore[override]
        self,
        index=None,
        *,
        axis: Axis | None = None,
        method: ReindexMethod | None = None,
        copy: bool | None = None,
        level: Level | None = None,
        fill_value: Scalar | None = None,
        limit: int | None = None,
        tolerance=None,
    ) -> Series:
        return super().reindex(
            index=index,
            method=method,
            copy=copy,
            level=level,
            fill_value=fill_value,
            limit=limit,
            tolerance=tolerance,
        )

    @overload  # type: ignore[override]
    def rename_axis(
        self,
        mapper: IndexLabel | lib.NoDefault = ...,
        *,
        index=...,
        axis: Axis = ...,
        copy: bool = ...,
        inplace: Literal[True],
    ) -> None:
        ...

    @overload
    def rename_axis(
        self,
        mapper: IndexLabel | lib.NoDefault = ...,
        *,
        index=...,
        axis: Axis = ...,
        copy: bool = ...,
        inplace: Literal[False] = ...,
    ) -> Self:
        ...

    @overload
    def rename_axis(
        self,
        mapper: IndexLabel | lib.NoDefault = ...,
        *,
        index=...,
        axis: Axis = ...,
        copy: bool = ...,
        inplace: bool = ...,
    ) -> Self | None:
        ...

    @doc(NDFrame.rename_axis)
    def rename_axis(
        self,
        mapper: IndexLabel | lib.NoDefault = lib.no_default,
        *,
        index=lib.no_default,
        axis: Axis = 0,
        copy: bool = True,
        inplace: bool = False,
    ) -> Self | None:
        return super().rename_axis(
            mapper=mapper,
            index=index,
            axis=axis,
            copy=copy,
            inplace=inplace,
        )

    @overload
    def drop(
        self,
        labels: IndexLabel = ...,
        *,
        axis: Axis = ...,
        index: IndexLabel = ...,
        columns: IndexLabel = ...,
        level: Level | None = ...,
        inplace: Literal[True],
        errors: IgnoreRaise = ...,
    ) -> None:
        ...

    @overload
    def drop(
        self,
        labels: IndexLabel = ...,
        *,
        axis: Axis = ...,
        index: IndexLabel = ...,
        columns: IndexLabel = ...,
        level: Level | None = ...,
        inplace: Literal[False] = ...,
        errors: IgnoreRaise = ...,
    ) -> Series:
        ...

    @overload
    def drop(
        self,
        labels: IndexLabel = ...,
        *,
        axis: Axis = ...,
        index: IndexLabel = ...,
        columns: IndexLabel = ...,
        level: Level | None = ...,
        inplace: bool = ...,
        errors: IgnoreRaise = ...,
    ) -> Series | None:
        ...

    def drop(
        self,
        labels: IndexLabel | None = None,
        *,
        axis: Axis = 0,
        index: IndexLabel | None = None,
        columns: IndexLabel | None = None,
        level: Level | None = None,
        inplace: bool = False,
        errors: IgnoreRaise = "raise",
    ) -> Series | None:
        """
        Return Series with specified index labels removed.

        Remove elements of a Series based on specifying the index labels.
        When using a multi-index, labels on different levels can be removed
        by specifying the level.

        Parameters
        ----------
        labels : single label or list-like
            Index labels to drop.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        index : single label or list-like
            Redundant for application on Series, but 'index' can be used instead
            of 'labels'.
        columns : single label or list-like
            No change is made to the Series; use 'index' or 'labels' instead.
        level : int or level name, optional
            For MultiIndex, level for which the labels will be removed.
        inplace : bool, default False
            If True, do operation inplace and return None.
        errors : {'ignore', 'raise'}, default 'raise'
            If 'ignore', suppress error and only existing labels are dropped.

        Returns
        -------
        Series or None
            Series with specified index labels removed or None if ``inplace=True``.

        Raises
        ------
        KeyError
            If none of the labels are found in the index.

        See Also
        --------
        Series.reindex : Return only specified index labels of Series.
        Series.dropna : Return series without null values.
        Series.drop_duplicates : Return Series with duplicate values removed.
        DataFrame.drop : Drop specified labels from rows or columns.

        Examples
        --------
        >>> s = pd.Series(data=np.arange(3), index=['A', 'B', 'C'])
        >>> s
        A  0
        B  1
        C  2
        dtype: int64

        Drop labels B en C

        >>> s.drop(labels=['B', 'C'])
        A  0
        dtype: int64

        Drop 2nd level label in MultiIndex Series

        >>> midx = pd.MultiIndex(levels=[['llama', 'cow', 'falcon'],
        ...                              ['speed', 'weight', 'length']],
        ...                      codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                             [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        >>> s = pd.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3],
        ...               index=midx)
        >>> s
        llama   speed      45.0
                weight    200.0
                length      1.2
        cow     speed      30.0
                weight    250.0
                length      1.5
        falcon  speed     320.0
                weight      1.0
                length      0.3
        dtype: float64

        >>> s.drop(labels='weight', level=1)
        llama   speed      45.0
                length      1.2
        cow     speed      30.0
                length      1.5
        falcon  speed     320.0
                length      0.3
        dtype: float64
        """
        return super().drop(
            labels=labels,
            axis=axis,
            index=index,
            columns=columns,
            level=level,
            inplace=inplace,
            errors=errors,
        )

    def pop(self, item: Hashable) -> Any:
        """
        Return item and drops from series. Raise KeyError if not found.

        Parameters
        ----------
        item : label
            Index of the element that needs to be removed.

        Returns
        -------
        Value that is popped from series.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3])

        >>> ser.pop(0)
        1

        >>> ser
        1    2
        2    3
        dtype: int64
        """
        return super().pop(item=item)

    @doc(INFO_DOCSTRING, **series_sub_kwargs)
    def info(
        self,
        verbose: bool | None = None,
        buf: IO[str] | None = None,
        max_cols: int | None = None,
        memory_usage: bool | str | None = None,
        show_counts: bool = True,
    ) -> None:
        return SeriesInfo(self, memory_usage).render(
            buf=buf,
            max_cols=max_cols,
            verbose=verbose,
            show_counts=show_counts,
        )

    # TODO(3.0): this can be removed once GH#33302 deprecation is enforced
    def _replace_single(self, to_replace, method: str, inplace: bool, limit):
        """
        Replaces values in a Series using the fill method specified when no
        replacement value is given in the replace method
        """

        result = self if inplace else self.copy()

        values = result._values
        mask = missing.mask_missing(values, to_replace)

        if isinstance(values, ExtensionArray):
            # dispatch to the EA's _pad_mask_inplace method
            values._fill_mask_inplace(method, limit, mask)
        else:
            fill_f = missing.get_fill_func(method)
            fill_f(values, limit=limit, mask=mask)

        if inplace:
            return
        return result

    def memory_usage(self, index: bool = True, deep: bool = False) -> int:
        """
        Return the memory usage of the Series.

        The memory usage can optionally include the contribution of
        the index and of elements of `object` dtype.

        Parameters
        ----------
        index : bool, default True
            Specifies whether to include the memory usage of the Series index.
        deep : bool, default False
            If True, introspect the data deeply by interrogating
            `object` dtypes for system-level memory consumption, and include
            it in the returned value.

        Returns
        -------
        int
            Bytes of memory consumed.

        See Also
        --------
        numpy.ndarray.nbytes : Total bytes consumed by the elements of the
            array.
        DataFrame.memory_usage : Bytes consumed by a DataFrame.

        Examples
        --------
        >>> s = pd.Series(range(3))
        >>> s.memory_usage()
        152

        Not including the index gives the size of the rest of the data, which
        is necessarily smaller:

        >>> s.memory_usage(index=False)
        24

        The memory footprint of `object` values is ignored by default:

        >>> s = pd.Series(["a", "b"])
        >>> s.values
        array(['a', 'b'], dtype=object)
        >>> s.memory_usage()
        144
        >>> s.memory_usage(deep=True)
        244
        """
        v = self._memory_usage(deep=deep)
        if index:
            v += self.index.memory_usage(deep=deep)
        return v

    def isin(self, values) -> Series:
        """
        Whether elements in Series are contained in `values`.

        Return a boolean Series showing whether each element in the Series
        matches an element in the passed sequence of `values` exactly.

        Parameters
        ----------
        values : set or list-like
            The sequence of values to test. Passing in a single string will
            raise a ``TypeError``. Instead, turn a single string into a
            list of one element.

        Returns
        -------
        Series
            Series of booleans indicating if each element is in values.

        Raises
        ------
        TypeError
          * If `values` is a string

        See Also
        --------
        DataFrame.isin : Equivalent method on DataFrame.

        Examples
        --------
        >>> s = pd.Series(['llama', 'cow', 'llama', 'beetle', 'llama',
        ...                'hippo'], name='animal')
        >>> s.isin(['cow', 'llama'])
        0     True
        1     True
        2     True
        3    False
        4     True
        5    False
        Name: animal, dtype: bool

        To invert the boolean values, use the ``~`` operator:

        >>> ~s.isin(['cow', 'llama'])
        0    False
        1    False
        2    False
        3     True
        4    False
        5     True
        Name: animal, dtype: bool

        Passing a single string as ``s.isin('llama')`` will raise an error. Use
        a list of one element instead:

        >>> s.isin(['llama'])
        0     True
        1    False
        2     True
        3    False
        4     True
        5    False
        Name: animal, dtype: bool

        Strings and integers are distinct and are therefore not comparable:

        >>> pd.Series([1]).isin(['1'])
        0    False
        dtype: bool
        >>> pd.Series([1.1]).isin(['1.1'])
        0    False
        dtype: bool
        """
        result = algorithms.isin(self._values, values)
        return self._constructor(result, index=self.index, copy=False).__finalize__(
            self, method="isin"
        )

    def between(
        self,
        left,
        right,
        inclusive: Literal["both", "neither", "left", "right"] = "both",
    ) -> Series:
        """
        Return boolean Series equivalent to left <= series <= right.

        This function returns a boolean vector containing `True` wherever the
        corresponding Series element is between the boundary values `left` and
        `right`. NA values are treated as `False`.

        Parameters
        ----------
        left : scalar or list-like
            Left boundary.
        right : scalar or list-like
            Right boundary.
        inclusive : {"both", "neither", "left", "right"}
            Include boundaries. Whether to set each bound as closed or open.

            .. versionchanged:: 1.3.0

        Returns
        -------
        Series
            Series representing whether each element is between left and
            right (inclusive).

        See Also
        --------
        Series.gt : Greater than of series and other.
        Series.lt : Less than of series and other.

        Notes
        -----
        This function is equivalent to ``(left <= ser) & (ser <= right)``

        Examples
        --------
        >>> s = pd.Series([2, 0, 4, 8, np.nan])

        Boundary values are included by default:

        >>> s.between(1, 4)
        0     True
        1    False
        2     True
        3    False
        4    False
        dtype: bool

        With `inclusive` set to ``"neither"`` boundary values are excluded:

        >>> s.between(1, 4, inclusive="neither")
        0     True
        1    False
        2    False
        3    False
        4    False
        dtype: bool

        `left` and `right` can be any scalar value:

        >>> s = pd.Series(['Alice', 'Bob', 'Carol', 'Eve'])
        >>> s.between('Anna', 'Daniel')
        0    False
        1     True
        2     True
        3    False
        dtype: bool
        """
        if inclusive == "both":
            lmask = self >= left
            rmask = self <= right
        elif inclusive == "left":
            lmask = self >= left
            rmask = self < right
        elif inclusive == "right":
            lmask = self > left
            rmask = self <= right
        elif inclusive == "neither":
            lmask = self > left
            rmask = self < right
        else:
            raise ValueError(
                "Inclusive has to be either string of 'both',"
                "'left', 'right', or 'neither'."
            )

        return lmask & rmask

    def case_when(
        self,
        caselist: list[
            tuple[
                ArrayLike | Callable[[Series], Series | np.ndarray | Sequence[bool]],
                ArrayLike | Scalar | Callable[[Series], Series | np.ndarray],
            ],
        ],
    ) -> Series:
        """
        Replace values where the conditions are True.

        Parameters
        ----------
        caselist : A list of tuples of conditions and expected replacements
            Takes the form:  ``(condition0, replacement0)``,
            ``(condition1, replacement1)``, ... .
            ``condition`` should be a 1-D boolean array-like object
            or a callable. If ``condition`` is a callable,
            it is computed on the Series
            and should return a boolean Series or array.
            The callable must not change the input Series
            (though pandas doesn`t check it). ``replacement`` should be a
            1-D array-like object, a scalar or a callable.
            If ``replacement`` is a callable, it is computed on the Series
            and should return a scalar or Series. The callable
            must not change the input Series
            (though pandas doesn`t check it).

            .. versionadded:: 2.2.0

        Returns
        -------
        Series

        See Also
        --------
        Series.mask : Replace values where the condition is True.

        Examples
        --------
        >>> c = pd.Series([6, 7, 8, 9], name='c')
        >>> a = pd.Series([0, 0, 1, 2])
        >>> b = pd.Series([0, 3, 4, 5])

        >>> c.case_when(caselist=[(a.gt(0), a),  # condition, replacement
        ...                       (b.gt(0), b)])
        0    6
        1    3
        2    1
        3    2
        Name: c, dtype: int64
        """
        if not isinstance(caselist, list):
            raise TypeError(
                f"The caselist argument should be a list; instead got {type(caselist)}"
            )

        if not caselist:
            raise ValueError(
                "provide at least one boolean condition, "
                "with a corresponding replacement."
            )

        for num, entry in enumerate(caselist):
            if not isinstance(entry, tuple):
                raise TypeError(
                    f"Argument {num} must be a tuple; instead got {type(entry)}."
                )
            if len(entry) != 2:
                raise ValueError(
                    f"Argument {num} must have length 2; "
                    "a condition and replacement; "
                    f"instead got length {len(entry)}."
                )
        caselist = [
            (
                com.apply_if_callable(condition, self),
                com.apply_if_callable(replacement, self),
            )
            for condition, replacement in caselist
        ]
        default = self.copy()
        conditions, replacements = zip(*caselist)
        common_dtypes = [infer_dtype_from(arg)[0] for arg in [*replacements, default]]
        if len(set(common_dtypes)) > 1:
            common_dtype = find_common_type(common_dtypes)
            updated_replacements = []
            for condition, replacement in zip(conditions, replacements):
                if is_scalar(replacement):
                    replacement = construct_1d_arraylike_from_scalar(
                        value=replacement, length=len(condition), dtype=common_dtype
                    )
                elif isinstance(replacement, ABCSeries):
                    replacement = replacement.astype(common_dtype)
                else:
                    replacement = pd_array(replacement, dtype=common_dtype)
                updated_replacements.append(replacement)
            replacements = updated_replacements
            default = default.astype(common_dtype)

        counter = reversed(range(len(conditions)))
        for position, condition, replacement in zip(
            counter, conditions[::-1], replacements[::-1]
        ):
            try:
                default = default.mask(
                    condition, other=replacement, axis=0, inplace=False, level=None
                )
            except Exception as error:
                raise ValueError(
                    f"Failed to apply condition{position} and replacement{position}."
                ) from error
        return default

    # error: Cannot determine type of 'isna'
    @doc(NDFrame.isna, klass=_shared_doc_kwargs["klass"])  # type: ignore[has-type]
    def isna(self) -> Series:
        return NDFrame.isna(self)

    # error: Cannot determine type of 'isna'
    @doc(NDFrame.isna, klass=_shared_doc_kwargs["klass"])  # type: ignore[has-type]
    def isnull(self) -> Series:
        """
        Series.isnull is an alias for Series.isna.
        """
        return super().isnull()

    # error: Cannot determine type of 'notna'
    @doc(NDFrame.notna, klass=_shared_doc_kwargs["klass"])  # type: ignore[has-type]
    def notna(self) -> Series:
        return super().notna()

    # error: Cannot determine type of 'notna'
    @doc(NDFrame.notna, klass=_shared_doc_kwargs["klass"])  # type: ignore[has-type]
    def notnull(self) -> Series:
        """
        Series.notnull is an alias for Series.notna.
        """
        return super().notnull()

    @overload
    def dropna(
        self,
        *,
        axis: Axis = ...,
        inplace: Literal[False] = ...,
        how: AnyAll | None = ...,
        ignore_index: bool = ...,
    ) -> Series:
        ...

    @overload
    def dropna(
        self,
        *,
        axis: Axis = ...,
        inplace: Literal[True],
        how: AnyAll | None = ...,
        ignore_index: bool = ...,
    ) -> None:
        ...

    def dropna(
        self,
        *,
        axis: Axis = 0,
        inplace: bool = False,
        how: AnyAll | None = None,
        ignore_index: bool = False,
    ) -> Series | None:
        """
        Return a new Series with missing values removed.

        See the :ref:`User Guide <missing_data>` for more on which values are
        considered missing, and how to work with missing data.

        Parameters
        ----------
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        inplace : bool, default False
            If True, do operation inplace and return None.
        how : str, optional
            Not in use. Kept for compatibility.
        ignore_index : bool, default ``False``
            If ``True``, the resulting axis will be labeled 0, 1, …, n - 1.

            .. versionadded:: 2.0.0

        Returns
        -------
        Series or None
            Series with NA entries dropped from it or None if ``inplace=True``.

        See Also
        --------
        Series.isna: Indicate missing values.
        Series.notna : Indicate existing (non-missing) values.
        Series.fillna : Replace missing values.
        DataFrame.dropna : Drop rows or columns which contain NA values.
        Index.dropna : Drop missing indices.

        Examples
        --------
        >>> ser = pd.Series([1., 2., np.nan])
        >>> ser
        0    1.0
        1    2.0
        2    NaN
        dtype: float64

        Drop NA values from a Series.

        >>> ser.dropna()
        0    1.0
        1    2.0
        dtype: float64

        Empty strings are not considered NA values. ``None`` is considered an
        NA value.

        >>> ser = pd.Series([np.nan, 2, pd.NaT, '', None, 'I stay'])
        >>> ser
        0       NaN
        1         2
        2       NaT
        3
        4      None
        5    I stay
        dtype: object
        >>> ser.dropna()
        1         2
        3
        5    I stay
        dtype: object
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        ignore_index = validate_bool_kwarg(ignore_index, "ignore_index")
        # Validate the axis parameter
        self._get_axis_number(axis or 0)

        if self._can_hold_na:
            result = remove_na_arraylike(self)
        else:
            if not inplace:
                result = self.copy(deep=None)
            else:
                result = self

        if ignore_index:
            result.index = default_index(len(result))

        if inplace:
            return self._update_inplace(result)
        else:
            return result

    # ----------------------------------------------------------------------
    # Time series-oriented methods

    def to_timestamp(
        self,
        freq: Frequency | None = None,
        how: Literal["s", "e", "start", "end"] = "start",
        copy: bool | None = None,
    ) -> Series:
        """
        Cast to DatetimeIndex of Timestamps, at *beginning* of period.

        Parameters
        ----------
        freq : str, default frequency of PeriodIndex
            Desired frequency.
        how : {'s', 'e', 'start', 'end'}
            Convention for converting period to timestamp; start of period
            vs. end.
        copy : bool, default True
            Whether or not to return a copy.

            .. note::
                The `copy` keyword will change behavior in pandas 3.0.
                `Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                will be enabled by default, which means that all methods with a
                `copy` keyword will use a lazy copy mechanism to defer the copy and
                ignore the `copy` keyword. The `copy` keyword will be removed in a
                future version of pandas.

                You can already get the future behavior and improvements through
                enabling copy on write ``pd.options.mode.copy_on_write = True``

        Returns
        -------
        Series with DatetimeIndex

        Examples
        --------
        >>> idx = pd.PeriodIndex(['2023', '2024', '2025'], freq='Y')
        >>> s1 = pd.Series([1, 2, 3], index=idx)
        >>> s1
        2023    1
        2024    2
        2025    3
        Freq: Y-DEC, dtype: int64

        The resulting frequency of the Timestamps is `YearBegin`

        >>> s1 = s1.to_timestamp()
        >>> s1
        2023-01-01    1
        2024-01-01    2
        2025-01-01    3
        Freq: YS-JAN, dtype: int64

        Using `freq` which is the offset that the Timestamps will have

        >>> s2 = pd.Series([1, 2, 3], index=idx)
        >>> s2 = s2.to_timestamp(freq='M')
        >>> s2
        2023-01-31    1
        2024-01-31    2
        2025-01-31    3
        Freq: YE-JAN, dtype: int64
        """
        if not isinstance(self.index, PeriodIndex):
            raise TypeError(f"unsupported Type {type(self.index).__name__}")

        new_obj = self.copy(deep=copy and not using_copy_on_write())
        new_index = self.index.to_timestamp(freq=freq, how=how)
        setattr(new_obj, "index", new_index)
        return new_obj

    def to_period(self, freq: str | None = None, copy: bool | None = None) -> Series:
        """
        Convert Series from DatetimeIndex to PeriodIndex.

        Parameters
        ----------
        freq : str, default None
            Frequency associated with the PeriodIndex.
        copy : bool, default True
            Whether or not to return a copy.

            .. note::
                The `copy` keyword will change behavior in pandas 3.0.
                `Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                will be enabled by default, which means that all methods with a
                `copy` keyword will use a lazy copy mechanism to defer the copy and
                ignore the `copy` keyword. The `copy` keyword will be removed in a
                future version of pandas.

                You can already get the future behavior and improvements through
                enabling copy on write ``pd.options.mode.copy_on_write = True``

        Returns
        -------
        Series
            Series with index converted to PeriodIndex.

        Examples
        --------
        >>> idx = pd.DatetimeIndex(['2023', '2024', '2025'])
        >>> s = pd.Series([1, 2, 3], index=idx)
        >>> s = s.to_period()
        >>> s
        2023    1
        2024    2
        2025    3
        Freq: Y-DEC, dtype: int64

        Viewing the index

        >>> s.index
        PeriodIndex(['2023', '2024', '2025'], dtype='period[Y-DEC]')
        """
        if not isinstance(self.index, DatetimeIndex):
            raise TypeError(f"unsupported Type {type(self.index).__name__}")

        new_obj = self.copy(deep=copy and not using_copy_on_write())
        new_index = self.index.to_period(freq=freq)
        setattr(new_obj, "index", new_index)
        return new_obj

    # ----------------------------------------------------------------------
    # Add index
    _AXIS_ORDERS: list[Literal["index", "columns"]] = ["index"]
    _AXIS_LEN = len(_AXIS_ORDERS)
    _info_axis_number: Literal[0] = 0
    _info_axis_name: Literal["index"] = "index"

    index = properties.AxisProperty(
        axis=0,
        doc="""
        The index (axis labels) of the Series.

        The index of a Series is used to label and identify each element of the
        underlying data. The index can be thought of as an immutable ordered set
        (technically a multi-set, as it may contain duplicate labels), and is
        used to index and align data in pandas.

        Returns
        -------
        Index
            The index labels of the Series.

        See Also
        --------
        Series.reindex : Conform Series to new index.
        Index : The base pandas index type.

        Notes
        -----
        For more information on pandas indexing, see the `indexing user guide
        <https://pandas.pydata.org/docs/user_guide/indexing.html>`__.

        Examples
        --------
        To create a Series with a custom index and view the index labels:

        >>> cities = ['Kolkata', 'Chicago', 'Toronto', 'Lisbon']
        >>> populations = [14.85, 2.71, 2.93, 0.51]
        >>> city_series = pd.Series(populations, index=cities)
        >>> city_series.index
        Index(['Kolkata', 'Chicago', 'Toronto', 'Lisbon'], dtype='object')

        To change the index labels of an existing Series:

        >>> city_series.index = ['KOL', 'CHI', 'TOR', 'LIS']
        >>> city_series.index
        Index(['KOL', 'CHI', 'TOR', 'LIS'], dtype='object')
        """,
    )

    # ----------------------------------------------------------------------
    # Accessor Methods
    # ----------------------------------------------------------------------
    str = CachedAccessor("str", StringMethods)
    dt = CachedAccessor("dt", CombinedDatetimelikeProperties)
    cat = CachedAccessor("cat", CategoricalAccessor)
    plot = CachedAccessor("plot", pandas.plotting.PlotAccessor)
    sparse = CachedAccessor("sparse", SparseAccessor)
    struct = CachedAccessor("struct", StructAccessor)
    list = CachedAccessor("list", ListAccessor)

    # ----------------------------------------------------------------------
    # Add plotting methods to Series
    hist = pandas.plotting.hist_series

    # ----------------------------------------------------------------------
    # Template-Based Arithmetic/Comparison Methods

    def _cmp_method(self, other, op):
        res_name = ops.get_op_result_name(self, other)

        if isinstance(other, Series) and not self._indexed_same(other):
            raise ValueError("Can only compare identically-labeled Series objects")

        lvalues = self._values
        rvalues = extract_array(other, extract_numpy=True, extract_range=True)

        res_values = ops.comparison_op(lvalues, rvalues, op)

        return self._construct_result(res_values, name=res_name)

    def _logical_method(self, other, op):
        res_name = ops.get_op_result_name(self, other)
        self, other = self._align_for_op(other, align_asobject=True)

        lvalues = self._values
        rvalues = extract_array(other, extract_numpy=True, extract_range=True)

        res_values = ops.logical_op(lvalues, rvalues, op)
        return self._construct_result(res_values, name=res_name)

    def _arith_method(self, other, op):
        self, other = self._align_for_op(other)
        return base.IndexOpsMixin._arith_method(self, other, op)

    def _align_for_op(self, right, align_asobject: bool = False):
        """align lhs and rhs Series"""
        # TODO: Different from DataFrame._align_for_op, list, tuple and ndarray
        # are not coerced here
        # because Series has inconsistencies described in GH#13637
        left = self

        if isinstance(right, Series):
            # avoid repeated alignment
            if not left.index.equals(right.index):
                if align_asobject:
                    if left.dtype not in (object, np.bool_) or right.dtype not in (
                        object,
                        np.bool_,
                    ):
                        warnings.warn(
                            "Operation between non boolean Series with different "
                            "indexes will no longer return a boolean result in "
                            "a future version. Cast both Series to object type "
                            "to maintain the prior behavior.",
                            FutureWarning,
                            stacklevel=find_stack_level(),
                        )
                    # to keep original value's dtype for bool ops
                    left = left.astype(object)
                    right = right.astype(object)

                left, right = left.align(right, copy=False)

        return left, right

    def _binop(self, other: Series, func, level=None, fill_value=None) -> Series:
        """
        Perform generic binary operation with optional fill value.

        Parameters
        ----------
        other : Series
        func : binary operator
        fill_value : float or object
            Value to substitute for NA/null values. If both Series are NA in a
            location, the result will be NA regardless of the passed fill value.
        level : int or level name, default None
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.

        Returns
        -------
        Series
        """
        this = self

        if not self.index.equals(other.index):
            this, other = self.align(other, level=level, join="outer", copy=False)

        this_vals, other_vals = ops.fill_binop(this._values, other._values, fill_value)

        with np.errstate(all="ignore"):
            result = func(this_vals, other_vals)

        name = ops.get_op_result_name(self, other)
        out = this._construct_result(result, name)
        return cast(Series, out)

    def _construct_result(
        self, result: ArrayLike | tuple[ArrayLike, ArrayLike], name: Hashable
    ) -> Series | tuple[Series, Series]:
        """
        Construct an appropriately-labelled Series from the result of an op.

        Parameters
        ----------
        result : ndarray or ExtensionArray
        name : Label

        Returns
        -------
        Series
            In the case of __divmod__ or __rdivmod__, a 2-tuple of Series.
        """
        if isinstance(result, tuple):
            # produced by divmod or rdivmod

            res1 = self._construct_result(result[0], name=name)
            res2 = self._construct_result(result[1], name=name)

            # GH#33427 assertions to keep mypy happy
            assert isinstance(res1, Series)
            assert isinstance(res2, Series)
            return (res1, res2)

        # TODO: result should always be ArrayLike, but this fails for some
        #  JSONArray tests
        dtype = getattr(result, "dtype", None)
        out = self._constructor(result, index=self.index, dtype=dtype, copy=False)
        out = out.__finalize__(self)

        # Set the result's name after __finalize__ is called because __finalize__
        #  would set it back to self.name
        out.name = name
        return out

    def _flex_method(self, other, op, *, level=None, fill_value=None, axis: Axis = 0):
        if axis is not None:
            self._get_axis_number(axis)

        res_name = ops.get_op_result_name(self, other)

        if isinstance(other, Series):
            return self._binop(other, op, level=level, fill_value=fill_value)
        elif isinstance(other, (np.ndarray, list, tuple)):
            if len(other) != len(self):
                raise ValueError("Lengths must be equal")
            other = self._constructor(other, self.index, copy=False)
            result = self._binop(other, op, level=level, fill_value=fill_value)
            result._name = res_name
            return result
        else:
            if fill_value is not None:
                if isna(other):
                    return op(self, fill_value)
                self = self.fillna(fill_value)

            return op(self, other)

    @Appender(ops.make_flex_doc("eq", "series"))
    def eq(
        self,
        other,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: Axis = 0,
    ) -> Series:
        return self._flex_method(
            other, operator.eq, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("ne", "series"))
    def ne(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        return self._flex_method(
            other, operator.ne, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("le", "series"))
    def le(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        return self._flex_method(
            other, operator.le, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("lt", "series"))
    def lt(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        return self._flex_method(
            other, operator.lt, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("ge", "series"))
    def ge(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        return self._flex_method(
            other, operator.ge, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("gt", "series"))
    def gt(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        return self._flex_method(
            other, operator.gt, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("add", "series"))
    def add(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        return self._flex_method(
            other, operator.add, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("radd", "series"))
    def radd(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        return self._flex_method(
            other, roperator.radd, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("sub", "series"))
    def sub(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        return self._flex_method(
            other, operator.sub, level=level, fill_value=fill_value, axis=axis
        )

    subtract = sub

    @Appender(ops.make_flex_doc("rsub", "series"))
    def rsub(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        return self._flex_method(
            other, roperator.rsub, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("mul", "series"))
    def mul(
        self,
        other,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: Axis = 0,
    ) -> Series:
        return self._flex_method(
            other, operator.mul, level=level, fill_value=fill_value, axis=axis
        )

    multiply = mul

    @Appender(ops.make_flex_doc("rmul", "series"))
    def rmul(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        return self._flex_method(
            other, roperator.rmul, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("truediv", "series"))
    def truediv(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        return self._flex_method(
            other, operator.truediv, level=level, fill_value=fill_value, axis=axis
        )

    div = truediv
    divide = truediv

    @Appender(ops.make_flex_doc("rtruediv", "series"))
    def rtruediv(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        return self._flex_method(
            other, roperator.rtruediv, level=level, fill_value=fill_value, axis=axis
        )

    rdiv = rtruediv

    @Appender(ops.make_flex_doc("floordiv", "series"))
    def floordiv(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        return self._flex_method(
            other, operator.floordiv, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("rfloordiv", "series"))
    def rfloordiv(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        return self._flex_method(
            other, roperator.rfloordiv, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("mod", "series"))
    def mod(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        return self._flex_method(
            other, operator.mod, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("rmod", "series"))
    def rmod(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        return self._flex_method(
            other, roperator.rmod, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("pow", "series"))
    def pow(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        return self._flex_method(
            other, operator.pow, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("rpow", "series"))
    def rpow(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        return self._flex_method(
            other, roperator.rpow, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("divmod", "series"))
    def divmod(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        return self._flex_method(
            other, divmod, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("rdivmod", "series"))
    def rdivmod(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        return self._flex_method(
            other, roperator.rdivmod, level=level, fill_value=fill_value, axis=axis
        )

    # ----------------------------------------------------------------------
    # Reductions

    def _reduce(
        self,
        op,
        # error: Variable "pandas.core.series.Series.str" is not valid as a type
        name: str,  # type: ignore[valid-type]
        *,
        axis: Axis = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        filter_type=None,
        **kwds,
    ):
        """
        Perform a reduction operation.

        If we have an ndarray as a value, then simply perform the operation,
        otherwise delegate to the object.
        """
        delegate = self._values

        if axis is not None:
            self._get_axis_number(axis)

        if isinstance(delegate, ExtensionArray):
            # dispatch to ExtensionArray interface
            return delegate._reduce(name, skipna=skipna, **kwds)

        else:
            # dispatch to numpy arrays
            if numeric_only and self.dtype.kind not in "iufcb":
                # i.e. not is_numeric_dtype(self.dtype)
                kwd_name = "numeric_only"
                if name in ["any", "all"]:
                    kwd_name = "bool_only"
                # GH#47500 - change to TypeError to match other methods
                raise TypeError(
                    f"Series.{name} does not allow {kwd_name}={numeric_only} "
                    "with non-numeric dtypes."
                )
            return op(delegate, skipna=skipna, **kwds)

    @Appender(make_doc("any", ndim=1))
    # error: Signature of "any" incompatible with supertype "NDFrame"
    def any(  # type: ignore[override]
        self,
        *,
        axis: Axis = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs,
    ) -> bool:
        nv.validate_logical_func((), kwargs, fname="any")
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        return self._reduce(
            nanops.nanany,
            name="any",
            axis=axis,
            numeric_only=bool_only,
            skipna=skipna,
            filter_type="bool",
        )

    @Appender(make_doc("all", ndim=1))
    def all(
        self,
        axis: Axis = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs,
    ) -> bool:
        nv.validate_logical_func((), kwargs, fname="all")
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        return self._reduce(
            nanops.nanall,
            name="all",
            axis=axis,
            numeric_only=bool_only,
            skipna=skipna,
            filter_type="bool",
        )

    @doc(make_doc("min", ndim=1))
    def min(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ):
        return NDFrame.min(self, axis, skipna, numeric_only, **kwargs)

    @doc(make_doc("max", ndim=1))
    def max(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ):
        return NDFrame.max(self, axis, skipna, numeric_only, **kwargs)

    @doc(make_doc("sum", ndim=1))
    def sum(
        self,
        axis: Axis | None = None,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count: int = 0,
        **kwargs,
    ):
        return NDFrame.sum(self, axis, skipna, numeric_only, min_count, **kwargs)

    @doc(make_doc("prod", ndim=1))
    def prod(
        self,
        axis: Axis | None = None,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count: int = 0,
        **kwargs,
    ):
        return NDFrame.prod(self, axis, skipna, numeric_only, min_count, **kwargs)

    @doc(make_doc("mean", ndim=1))
    def mean(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ):
        return NDFrame.mean(self, axis, skipna, numeric_only, **kwargs)

    @doc(make_doc("median", ndim=1))
    def median(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ):
        return NDFrame.median(self, axis, skipna, numeric_only, **kwargs)

    @doc(make_doc("sem", ndim=1))
    def sem(
        self,
        axis: Axis | None = None,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs,
    ):
        return NDFrame.sem(self, axis, skipna, ddof, numeric_only, **kwargs)

    @doc(make_doc("var", ndim=1))
    def var(
        self,
        axis: Axis | None = None,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs,
    ):
        return NDFrame.var(self, axis, skipna, ddof, numeric_only, **kwargs)

    @doc(make_doc("std", ndim=1))
    def std(
        self,
        axis: Axis | None = None,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs,
    ):
        return NDFrame.std(self, axis, skipna, ddof, numeric_only, **kwargs)

    @doc(make_doc("skew", ndim=1))
    def skew(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ):
        return NDFrame.skew(self, axis, skipna, numeric_only, **kwargs)

    @doc(make_doc("kurt", ndim=1))
    def kurt(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ):
        return NDFrame.kurt(self, axis, skipna, numeric_only, **kwargs)

    kurtosis = kurt
    product = prod

    @doc(make_doc("cummin", ndim=1))
    def cummin(self, axis: Axis | None = None, skipna: bool = True, *args, **kwargs):
        return NDFrame.cummin(self, axis, skipna, *args, **kwargs)

    @doc(make_doc("cummax", ndim=1))
    def cummax(self, axis: Axis | None = None, skipna: bool = True, *args, **kwargs):
        return NDFrame.cummax(self, axis, skipna, *args, **kwargs)

    @doc(make_doc("cumsum", ndim=1))
    def cumsum(self, axis: Axis | None = None, skipna: bool = True, *args, **kwargs):
        return NDFrame.cumsum(self, axis, skipna, *args, **kwargs)

    @doc(make_doc("cumprod", 1))
    def cumprod(self, axis: Axis | None = None, skipna: bool = True, *args, **kwargs):
        return NDFrame.cumprod(self, axis, skipna, *args, **kwargs)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\sql\schema.py ===
# sql/schema.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""The schema module provides the building blocks for database metadata.

Each element within this module describes a database entity which can be
created and dropped, or is otherwise part of such an entity.  Examples include
tables, columns, sequences, and indexes.

All entities are subclasses of :class:`~sqlalchemy.schema.SchemaItem`, and as
defined in this module they are intended to be agnostic of any vendor-specific
constructs.

A collection of entities are grouped into a unit called
:class:`~sqlalchemy.schema.MetaData`. MetaData serves as a logical grouping of
schema elements, and can also be associated with an actual database connection
such that operations involving the contained elements can contact the database
as needed.

Two of the elements here also build upon their "syntactic" counterparts, which
are defined in :class:`~sqlalchemy.sql.expression.`, specifically
:class:`~sqlalchemy.schema.Table` and :class:`~sqlalchemy.schema.Column`.
Since these objects are part of the SQL expression language, they are usable
as components in SQL expressions.

"""
from __future__ import annotations

from abc import ABC
import collections
from enum import Enum
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence as _typing_Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from . import coercions
from . import ddl
from . import roles
from . import type_api
from . import visitors
from .base import _DefaultDescriptionTuple
from .base import _NoArg
from .base import _NoneName
from .base import _SentinelColumnCharacterization
from .base import _SentinelDefaultCharacterization
from .base import DedupeColumnCollection
from .base import DialectKWArgs
from .base import Executable
from .base import SchemaEventTarget as SchemaEventTarget
from .base import SchemaVisitable as SchemaVisitable
from .coercions import _document_text_coercion
from .elements import ClauseElement
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import quoted_name
from .elements import TextClause
from .selectable import TableClause
from .type_api import to_instance
from .visitors import ExternallyTraversible
from .. import event
from .. import exc
from .. import inspection
from .. import util
from ..util import HasMemoized
from ..util.typing import Final
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypedDict
from ..util.typing import TypeGuard

if typing.TYPE_CHECKING:
    from ._typing import _AutoIncrementType
    from ._typing import _CreateDropBind
    from ._typing import _DDLColumnArgument
    from ._typing import _InfoType
    from ._typing import _TextCoercedExpressionArgument
    from ._typing import _TypeEngineArgument
    from .base import ColumnSet
    from .base import ReadOnlyColumnCollection
    from .compiler import DDLCompiler
    from .elements import BindParameter
    from .elements import KeyedColumnElement
    from .functions import Function
    from .type_api import TypeEngine
    from .visitors import anon_map
    from ..engine import Connection
    from ..engine import Engine
    from ..engine.interfaces import _CoreMultiExecuteParams
    from ..engine.interfaces import CoreExecuteOptionsParameter
    from ..engine.interfaces import ExecutionContext
    from ..engine.reflection import _ReflectionInfo
    from ..sql.selectable import FromClause

_T = TypeVar("_T", bound="Any")
_SI = TypeVar("_SI", bound="SchemaItem")
_TAB = TypeVar("_TAB", bound="Table")


_ConstraintNameArgument = Optional[Union[str, _NoneName]]

_ServerDefaultArgument = Union[
    "FetchedValue", str, TextClause, ColumnElement[Any]
]

_ServerOnUpdateArgument = _ServerDefaultArgument


class SchemaConst(Enum):
    RETAIN_SCHEMA = 1
    """Symbol indicating that a :class:`_schema.Table`, :class:`.Sequence`
    or in some cases a :class:`_schema.ForeignKey` object, in situations
    where the object is being copied for a :meth:`.Table.to_metadata`
    operation, should retain the schema name that it already has.

    """

    BLANK_SCHEMA = 2
    """Symbol indicating that a :class:`_schema.Table` or :class:`.Sequence`
    should have 'None' for its schema, even if the parent
    :class:`_schema.MetaData` has specified a schema.

    .. seealso::

        :paramref:`_schema.MetaData.schema`

        :paramref:`_schema.Table.schema`

        :paramref:`.Sequence.schema`

    """

    NULL_UNSPECIFIED = 3
    """Symbol indicating the "nullable" keyword was not passed to a Column.

    This is used to distinguish between the use case of passing
    ``nullable=None`` to a :class:`.Column`, which has special meaning
    on some backends such as SQL Server.

    """


RETAIN_SCHEMA: Final[Literal[SchemaConst.RETAIN_SCHEMA]] = (
    SchemaConst.RETAIN_SCHEMA
)
BLANK_SCHEMA: Final[Literal[SchemaConst.BLANK_SCHEMA]] = (
    SchemaConst.BLANK_SCHEMA
)
NULL_UNSPECIFIED: Final[Literal[SchemaConst.NULL_UNSPECIFIED]] = (
    SchemaConst.NULL_UNSPECIFIED
)


def _get_table_key(name: str, schema: Optional[str]) -> str:
    if schema is None:
        return name
    else:
        return schema + "." + name


# this should really be in sql/util.py but we'd have to
# break an import cycle
def _copy_expression(
    expression: ColumnElement[Any],
    source_table: Optional[Table],
    target_table: Optional[Table],
) -> ColumnElement[Any]:
    if source_table is None or target_table is None:
        return expression

    fixed_source_table = source_table
    fixed_target_table = target_table

    def replace(
        element: ExternallyTraversible, **kw: Any
    ) -> Optional[ExternallyTraversible]:
        if (
            isinstance(element, Column)
            and element.table is fixed_source_table
            and element.key in fixed_source_table.c
        ):
            return fixed_target_table.c[element.key]
        else:
            return None

    return cast(
        ColumnElement[Any],
        visitors.replacement_traverse(expression, {}, replace),
    )


@inspection._self_inspects
class SchemaItem(SchemaVisitable):
    """Base class for items that define a database schema."""

    __visit_name__ = "schema_item"

    create_drop_stringify_dialect = "default"

    def _init_items(self, *args: SchemaItem, **kw: Any) -> None:
        """Initialize the list of child items for this SchemaItem."""
        for item in args:
            if item is not None:
                try:
                    spwd = item._set_parent_with_dispatch
                except AttributeError as err:
                    raise exc.ArgumentError(
                        "'SchemaItem' object, such as a 'Column' or a "
                        f"'Constraint' expected, got {item!r}"
                    ) from err
                else:
                    spwd(self, **kw)

    def __repr__(self) -> str:
        return util.generic_repr(self, omit_kwarg=["info"])

    @util.memoized_property
    def info(self) -> _InfoType:
        """Info dictionary associated with the object, allowing user-defined
        data to be associated with this :class:`.SchemaItem`.

        The dictionary is automatically generated when first accessed.
        It can also be specified in the constructor of some objects,
        such as :class:`_schema.Table` and :class:`_schema.Column`.

        """
        return {}

    def _schema_item_copy(self, schema_item: _SI) -> _SI:
        if "info" in self.__dict__:
            schema_item.info = self.info.copy()
        schema_item.dispatch._update(self.dispatch)
        return schema_item

    _use_schema_map = True


class HasConditionalDDL:
    """define a class that includes the :meth:`.HasConditionalDDL.ddl_if`
    method, allowing for conditional rendering of DDL.

    Currently applies to constraints and indexes.

    .. versionadded:: 2.0


    """

    _ddl_if: Optional[ddl.DDLIf] = None

    def ddl_if(
        self,
        dialect: Optional[str] = None,
        callable_: Optional[ddl.DDLIfCallable] = None,
        state: Optional[Any] = None,
    ) -> Self:
        r"""apply a conditional DDL rule to this schema item.

        These rules work in a similar manner to the
        :meth:`.ExecutableDDLElement.execute_if` callable, with the added
        feature that the criteria may be checked within the DDL compilation
        phase for a construct such as :class:`.CreateTable`.
        :meth:`.HasConditionalDDL.ddl_if` currently applies towards the
        :class:`.Index` construct as well as all :class:`.Constraint`
        constructs.

        :param dialect: string name of a dialect, or a tuple of string names
         to indicate multiple dialect types.

        :param callable\_: a callable that is constructed using the same form
         as that described in
         :paramref:`.ExecutableDDLElement.execute_if.callable_`.

        :param state: any arbitrary object that will be passed to the
         callable, if present.

        .. versionadded:: 2.0

        .. seealso::

            :ref:`schema_ddl_ddl_if` - background and usage examples


        """
        self._ddl_if = ddl.DDLIf(dialect, callable_, state)
        return self


class HasSchemaAttr(SchemaItem):
    """schema item that includes a top-level schema name"""

    schema: Optional[str]


class Table(
    DialectKWArgs, HasSchemaAttr, TableClause, inspection.Inspectable["Table"]
):
    r"""Represent a table in a database.

    e.g.::

        mytable = Table(
            "mytable",
            metadata,
            Column("mytable_id", Integer, primary_key=True),
            Column("value", String(50)),
        )

    The :class:`_schema.Table`
    object constructs a unique instance of itself based
    on its name and optional schema name within the given
    :class:`_schema.MetaData` object. Calling the :class:`_schema.Table`
    constructor with the same name and same :class:`_schema.MetaData` argument
    a second time will return the *same* :class:`_schema.Table`
    object - in this way
    the :class:`_schema.Table` constructor acts as a registry function.

    .. seealso::

        :ref:`metadata_describing` - Introduction to database metadata

    """

    __visit_name__ = "table"

    if TYPE_CHECKING:

        @util.ro_non_memoized_property
        def primary_key(self) -> PrimaryKeyConstraint: ...

        @util.ro_non_memoized_property
        def foreign_keys(self) -> Set[ForeignKey]: ...

    _columns: DedupeColumnCollection[Column[Any]]

    _sentinel_column: Optional[Column[Any]]

    constraints: Set[Constraint]
    """A collection of all :class:`_schema.Constraint` objects associated with
      this :class:`_schema.Table`.

      Includes :class:`_schema.PrimaryKeyConstraint`,
      :class:`_schema.ForeignKeyConstraint`, :class:`_schema.UniqueConstraint`,
      :class:`_schema.CheckConstraint`.  A separate collection
      :attr:`_schema.Table.foreign_key_constraints` refers to the collection
      of all :class:`_schema.ForeignKeyConstraint` objects, and the
      :attr:`_schema.Table.primary_key` attribute refers to the single
      :class:`_schema.PrimaryKeyConstraint` associated with the
      :class:`_schema.Table`.

      .. seealso::

            :attr:`_schema.Table.constraints`

            :attr:`_schema.Table.primary_key`

            :attr:`_schema.Table.foreign_key_constraints`

            :attr:`_schema.Table.indexes`

            :class:`_reflection.Inspector`


    """

    indexes: Set[Index]
    """A collection of all :class:`_schema.Index` objects associated with this
      :class:`_schema.Table`.

      .. seealso::

            :meth:`_reflection.Inspector.get_indexes`

    """

    if TYPE_CHECKING:

        @util.ro_non_memoized_property
        def columns(self) -> ReadOnlyColumnCollection[str, Column[Any]]: ...

        @util.ro_non_memoized_property
        def exported_columns(
            self,
        ) -> ReadOnlyColumnCollection[str, Column[Any]]: ...

        @util.ro_non_memoized_property
        def c(self) -> ReadOnlyColumnCollection[str, Column[Any]]: ...

    def _gen_cache_key(
        self, anon_map: anon_map, bindparams: List[BindParameter[Any]]
    ) -> Tuple[Any, ...]:
        if self._annotations:
            return (self,) + self._annotations_cache_key
        else:
            return (self,)

    if not typing.TYPE_CHECKING:
        # typing tools seem to be inconsistent in how they handle
        # __new__, so suggest this pattern for classes that use
        # __new__.  apply typing to the __init__ method normally
        @util.deprecated_params(
            mustexist=(
                "1.4",
                "Deprecated alias of :paramref:`_schema.Table.must_exist`",
            ),
        )
        def __new__(cls, *args: Any, **kw: Any) -> Any:
            return cls._new(*args, **kw)

    @classmethod
    def _new(cls, *args: Any, **kw: Any) -> Any:
        if not args and not kw:
            # python3k pickle seems to call this
            return object.__new__(cls)

        try:
            name, metadata, args = args[0], args[1], args[2:]
        except IndexError:
            raise TypeError(
                "Table() takes at least two positional-only "
                "arguments 'name' and 'metadata'"
            )

        schema = kw.get("schema", None)
        if schema is None:
            schema = metadata.schema
        elif schema is BLANK_SCHEMA:
            schema = None
        keep_existing = kw.get("keep_existing", False)
        extend_existing = kw.get("extend_existing", False)

        if keep_existing and extend_existing:
            msg = "keep_existing and extend_existing are mutually exclusive."
            raise exc.ArgumentError(msg)

        must_exist = kw.pop("must_exist", kw.pop("mustexist", False))
        key = _get_table_key(name, schema)
        if key in metadata.tables:
            if not keep_existing and not extend_existing and bool(args):
                raise exc.InvalidRequestError(
                    f"Table '{key}' is already defined for this MetaData "
                    "instance.  Specify 'extend_existing=True' "
                    "to redefine "
                    "options and columns on an "
                    "existing Table object."
                )
            table = metadata.tables[key]
            if extend_existing:
                table._init_existing(*args, **kw)
            return table
        else:
            if must_exist:
                raise exc.InvalidRequestError(f"Table '{key}' not defined")
            table = object.__new__(cls)
            table.dispatch.before_parent_attach(table, metadata)
            metadata._add_table(name, schema, table)
            try:
                table.__init__(name, metadata, *args, _no_init=False, **kw)
                table.dispatch.after_parent_attach(table, metadata)
                return table
            except Exception:
                with util.safe_reraise():
                    metadata._remove_table(name, schema)

    def __init__(
        self,
        name: str,
        metadata: MetaData,
        *args: SchemaItem,
        schema: Optional[Union[str, Literal[SchemaConst.BLANK_SCHEMA]]] = None,
        quote: Optional[bool] = None,
        quote_schema: Optional[bool] = None,
        autoload_with: Optional[Union[Engine, Connection]] = None,
        autoload_replace: bool = True,
        keep_existing: bool = False,
        extend_existing: bool = False,
        resolve_fks: bool = True,
        include_columns: Optional[Collection[str]] = None,
        implicit_returning: bool = True,
        comment: Optional[str] = None,
        info: Optional[Dict[Any, Any]] = None,
        listeners: Optional[
            _typing_Sequence[Tuple[str, Callable[..., Any]]]
        ] = None,
        prefixes: Optional[_typing_Sequence[str]] = None,
        # used internally in the metadata.reflect() process
        _extend_on: Optional[Set[Table]] = None,
        # used by __new__ to bypass __init__
        _no_init: bool = True,
        # dialect-specific keyword args
        **kw: Any,
    ) -> None:
        r"""Constructor for :class:`_schema.Table`.


        :param name: The name of this table as represented in the database.

            The table name, along with the value of the ``schema`` parameter,
            forms a key which uniquely identifies this :class:`_schema.Table`
            within
            the owning :class:`_schema.MetaData` collection.
            Additional calls to :class:`_schema.Table` with the same name,
            metadata,
            and schema name will return the same :class:`_schema.Table` object.

            Names which contain no upper case characters
            will be treated as case insensitive names, and will not be quoted
            unless they are a reserved word or contain special characters.
            A name with any number of upper case characters is considered
            to be case sensitive, and will be sent as quoted.

            To enable unconditional quoting for the table name, specify the flag
            ``quote=True`` to the constructor, or use the :class:`.quoted_name`
            construct to specify the name.

        :param metadata: a :class:`_schema.MetaData`
            object which will contain this
            table.  The metadata is used as a point of association of this table
            with other tables which are referenced via foreign key.  It also
            may be used to associate this table with a particular
            :class:`.Connection` or :class:`.Engine`.

        :param \*args: Additional positional arguments are used primarily
            to add the list of :class:`_schema.Column`
            objects contained within this
            table. Similar to the style of a CREATE TABLE statement, other
            :class:`.SchemaItem` constructs may be added here, including
            :class:`.PrimaryKeyConstraint`, and
            :class:`_schema.ForeignKeyConstraint`.

        :param autoload_replace: Defaults to ``True``; when using
            :paramref:`_schema.Table.autoload_with`
            in conjunction with :paramref:`_schema.Table.extend_existing`,
            indicates
            that :class:`_schema.Column` objects present in the already-existing
            :class:`_schema.Table`
            object should be replaced with columns of the same
            name retrieved from the autoload process.   When ``False``, columns
            already present under existing names will be omitted from the
            reflection process.

            Note that this setting does not impact :class:`_schema.Column` objects
            specified programmatically within the call to :class:`_schema.Table`
            that
            also is autoloading; those :class:`_schema.Column` objects will always
            replace existing columns of the same name when
            :paramref:`_schema.Table.extend_existing` is ``True``.

            .. seealso::

                :paramref:`_schema.Table.autoload_with`

                :paramref:`_schema.Table.extend_existing`

        :param autoload_with: An :class:`_engine.Engine` or
            :class:`_engine.Connection` object,
            or a :class:`_reflection.Inspector` object as returned by
            :func:`_sa.inspect`
            against one, with which this :class:`_schema.Table`
            object will be reflected.
            When set to a non-None value, the autoload process will take place
            for this table against the given engine or connection.

            .. seealso::

                :ref:`metadata_reflection_toplevel`

                :meth:`_events.DDLEvents.column_reflect`

                :ref:`metadata_reflection_dbagnostic_types`

        :param extend_existing: When ``True``, indicates that if this
            :class:`_schema.Table` is already present in the given
            :class:`_schema.MetaData`,
            apply further arguments within the constructor to the existing
            :class:`_schema.Table`.

            If :paramref:`_schema.Table.extend_existing` or
            :paramref:`_schema.Table.keep_existing` are not set,
            and the given name
            of the new :class:`_schema.Table` refers to a :class:`_schema.Table`
            that is
            already present in the target :class:`_schema.MetaData` collection,
            and
            this :class:`_schema.Table`
            specifies additional columns or other constructs
            or flags that modify the table's state, an
            error is raised.  The purpose of these two mutually-exclusive flags
            is to specify what action should be taken when a
            :class:`_schema.Table`
            is specified that matches an existing :class:`_schema.Table`,
            yet specifies
            additional constructs.

            :paramref:`_schema.Table.extend_existing`
            will also work in conjunction
            with :paramref:`_schema.Table.autoload_with` to run a new reflection
            operation against the database, even if a :class:`_schema.Table`
            of the same name is already present in the target
            :class:`_schema.MetaData`; newly reflected :class:`_schema.Column`
            objects
            and other options will be added into the state of the
            :class:`_schema.Table`, potentially overwriting existing columns
            and options of the same name.

            As is always the case with :paramref:`_schema.Table.autoload_with`,
            :class:`_schema.Column` objects can be specified in the same
            :class:`_schema.Table`
            constructor, which will take precedence.  Below, the existing
            table ``mytable`` will be augmented with :class:`_schema.Column`
            objects
            both reflected from the database, as well as the given
            :class:`_schema.Column`
            named "y"::

                Table(
                    "mytable",
                    metadata,
                    Column("y", Integer),
                    extend_existing=True,
                    autoload_with=engine,
                )

            .. seealso::

                :paramref:`_schema.Table.autoload_with`

                :paramref:`_schema.Table.autoload_replace`

                :paramref:`_schema.Table.keep_existing`


        :param implicit_returning: True by default - indicates that
            RETURNING can be used, typically by the ORM, in order to fetch
            server-generated values such as primary key values and
            server side defaults, on those backends which support RETURNING.

            In modern SQLAlchemy there is generally no reason to alter this
            setting, except for some backend specific cases
            (see :ref:`mssql_triggers` in the SQL Server dialect documentation
            for one such example).

        :param include_columns: A list of strings indicating a subset of
            columns to be loaded via the ``autoload`` operation; table columns who
            aren't present in this list will not be represented on the resulting
            ``Table`` object. Defaults to ``None`` which indicates all columns
            should be reflected.

        :param resolve_fks: Whether or not to reflect :class:`_schema.Table`
            objects
            related to this one via :class:`_schema.ForeignKey` objects, when
            :paramref:`_schema.Table.autoload_with` is
            specified.   Defaults to True.  Set to False to disable reflection of
            related tables as :class:`_schema.ForeignKey`
            objects are encountered; may be
            used either to save on SQL calls or to avoid issues with related tables
            that can't be accessed. Note that if a related table is already present
            in the :class:`_schema.MetaData` collection, or becomes present later,
            a
            :class:`_schema.ForeignKey` object associated with this
            :class:`_schema.Table` will
            resolve to that table normally.

            .. versionadded:: 1.3

            .. seealso::

                :paramref:`.MetaData.reflect.resolve_fks`


        :param info: Optional data dictionary which will be populated into the
            :attr:`.SchemaItem.info` attribute of this object.

        :param keep_existing: When ``True``, indicates that if this Table
            is already present in the given :class:`_schema.MetaData`, ignore
            further arguments within the constructor to the existing
            :class:`_schema.Table`, and return the :class:`_schema.Table`
            object as
            originally created. This is to allow a function that wishes
            to define a new :class:`_schema.Table` on first call, but on
            subsequent calls will return the same :class:`_schema.Table`,
            without any of the declarations (particularly constraints)
            being applied a second time.

            If :paramref:`_schema.Table.extend_existing` or
            :paramref:`_schema.Table.keep_existing` are not set,
            and the given name
            of the new :class:`_schema.Table` refers to a :class:`_schema.Table`
            that is
            already present in the target :class:`_schema.MetaData` collection,
            and
            this :class:`_schema.Table`
            specifies additional columns or other constructs
            or flags that modify the table's state, an
            error is raised.  The purpose of these two mutually-exclusive flags
            is to specify what action should be taken when a
            :class:`_schema.Table`
            is specified that matches an existing :class:`_schema.Table`,
            yet specifies
            additional constructs.

            .. seealso::

                :paramref:`_schema.Table.extend_existing`

        :param listeners: A list of tuples of the form ``(<eventname>, <fn>)``
            which will be passed to :func:`.event.listen` upon construction.
            This alternate hook to :func:`.event.listen` allows the establishment
            of a listener function specific to this :class:`_schema.Table` before
            the "autoload" process begins.  Historically this has been intended
            for use with the :meth:`.DDLEvents.column_reflect` event, however
            note that this event hook may now be associated with the
            :class:`_schema.MetaData` object directly::

                def listen_for_reflect(table, column_info):
                    "handle the column reflection event"
                    # ...


                t = Table(
                    "sometable",
                    autoload_with=engine,
                    listeners=[("column_reflect", listen_for_reflect)],
                )

            .. seealso::

                :meth:`_events.DDLEvents.column_reflect`

        :param must_exist: When ``True``, indicates that this Table must already
            be present in the given :class:`_schema.MetaData` collection, else
            an exception is raised.

        :param prefixes:
            A list of strings to insert after CREATE in the CREATE TABLE
            statement.  They will be separated by spaces.

        :param quote: Force quoting of this table's name on or off, corresponding
            to ``True`` or ``False``.  When left at its default of ``None``,
            the column identifier will be quoted according to whether the name is
            case sensitive (identifiers with at least one upper case character are
            treated as case sensitive), or if it's a reserved word.  This flag
            is only needed to force quoting of a reserved word which is not known
            by the SQLAlchemy dialect.

            .. note:: setting this flag to ``False`` will not provide
              case-insensitive behavior for table reflection; table reflection
              will always search for a mixed-case name in a case sensitive
              fashion.  Case insensitive names are specified in SQLAlchemy only
              by stating the name with all lower case characters.

        :param quote_schema: same as 'quote' but applies to the schema identifier.

        :param schema: The schema name for this table, which is required if
            the table resides in a schema other than the default selected schema
            for the engine's database connection.  Defaults to ``None``.

            If the owning :class:`_schema.MetaData` of this :class:`_schema.Table`
            specifies its
            own :paramref:`_schema.MetaData.schema` parameter,
            then that schema name will
            be applied to this :class:`_schema.Table`
            if the schema parameter here is set
            to ``None``.  To set a blank schema name on a :class:`_schema.Table`
            that
            would otherwise use the schema set on the owning
            :class:`_schema.MetaData`,
            specify the special symbol :attr:`.BLANK_SCHEMA`.

            The quoting rules for the schema name are the same as those for the
            ``name`` parameter, in that quoting is applied for reserved words or
            case-sensitive names; to enable unconditional quoting for the schema
            name, specify the flag ``quote_schema=True`` to the constructor, or use
            the :class:`.quoted_name` construct to specify the name.

        :param comment: Optional string that will render an SQL comment on table
            creation.

            .. versionadded:: 1.2 Added the :paramref:`_schema.Table.comment`
                parameter
                to :class:`_schema.Table`.

        :param \**kw: Additional keyword arguments not mentioned above are
            dialect specific, and passed in the form ``<dialectname>_<argname>``.
            See the documentation regarding an individual dialect at
            :ref:`dialect_toplevel` for detail on documented arguments.

        """  # noqa: E501
        if _no_init:
            # don't run __init__ from __new__ by default;
            # __new__ has a specific place that __init__ is called
            return

        super().__init__(quoted_name(name, quote))
        self.metadata = metadata

        if schema is None:
            self.schema = metadata.schema
        elif schema is BLANK_SCHEMA:
            self.schema = None
        else:
            quote_schema = quote_schema
            assert isinstance(schema, str)
            self.schema = quoted_name(schema, quote_schema)

        self._sentinel_column = None

        self.indexes = set()
        self.constraints = set()
        PrimaryKeyConstraint(
            _implicit_generated=True
        )._set_parent_with_dispatch(self)
        self.foreign_keys = set()  # type: ignore
        self._extra_dependencies: Set[Table] = set()
        if self.schema is not None:
            self.fullname = "%s.%s" % (self.schema, self.name)
        else:
            self.fullname = self.name

        self.implicit_returning = implicit_returning
        _reflect_info = kw.pop("_reflect_info", None)

        self.comment = comment

        if info is not None:
            self.info = info

        if listeners is not None:
            for evt, fn in listeners:
                event.listen(self, evt, fn)

        self._prefixes = prefixes if prefixes else []

        self._extra_kwargs(**kw)

        # load column definitions from the database if 'autoload' is defined
        # we do it after the table is in the singleton dictionary to support
        # circular foreign keys
        if autoload_with is not None:
            self._autoload(
                metadata,
                autoload_with,
                include_columns,
                _extend_on=_extend_on,
                _reflect_info=_reflect_info,
                resolve_fks=resolve_fks,
            )

        # initialize all the column, etc. objects.  done after reflection to
        # allow user-overrides

        self._init_items(
            *args,
            allow_replacements=extend_existing
            or keep_existing
            or autoload_with,
            all_names={},
        )

    def _autoload(
        self,
        metadata: MetaData,
        autoload_with: Union[Engine, Connection],
        include_columns: Optional[Collection[str]],
        exclude_columns: Collection[str] = (),
        resolve_fks: bool = True,
        _extend_on: Optional[Set[Table]] = None,
        _reflect_info: _ReflectionInfo | None = None,
    ) -> None:
        insp = inspection.inspect(autoload_with)
        with insp._inspection_context() as conn_insp:
            conn_insp.reflect_table(
                self,
                include_columns,
                exclude_columns,
                resolve_fks,
                _extend_on=_extend_on,
                _reflect_info=_reflect_info,
            )

    @property
    def _sorted_constraints(self) -> List[Constraint]:
        """Return the set of constraints as a list, sorted by creation
        order.

        """

        return sorted(self.constraints, key=lambda c: c._creation_order)

    @property
    def foreign_key_constraints(self) -> Set[ForeignKeyConstraint]:
        """:class:`_schema.ForeignKeyConstraint` objects referred to by this
        :class:`_schema.Table`.

        This list is produced from the collection of
        :class:`_schema.ForeignKey`
        objects currently associated.


        .. seealso::

            :attr:`_schema.Table.constraints`

            :attr:`_schema.Table.foreign_keys`

            :attr:`_schema.Table.indexes`

        """
        return {
            fkc.constraint
            for fkc in self.foreign_keys
            if fkc.constraint is not None
        }

    def _init_existing(self, *args: Any, **kwargs: Any) -> None:
        autoload_with = kwargs.pop("autoload_with", None)
        autoload = kwargs.pop("autoload", autoload_with is not None)
        autoload_replace = kwargs.pop("autoload_replace", True)
        schema = kwargs.pop("schema", None)
        _extend_on = kwargs.pop("_extend_on", None)
        _reflect_info = kwargs.pop("_reflect_info", None)

        # these arguments are only used with _init()
        extend_existing = kwargs.pop("extend_existing", False)
        keep_existing = kwargs.pop("keep_existing", False)

        assert extend_existing
        assert not keep_existing

        if schema and schema != self.schema:
            raise exc.ArgumentError(
                f"Can't change schema of existing table "
                f"from '{self.schema}' to '{schema}'",
            )

        include_columns = kwargs.pop("include_columns", None)
        if include_columns is not None:
            for c in self.c:
                if c.name not in include_columns:
                    self._columns.remove(c)

        resolve_fks = kwargs.pop("resolve_fks", True)

        for key in ("quote", "quote_schema"):
            if key in kwargs:
                raise exc.ArgumentError(
                    "Can't redefine 'quote' or 'quote_schema' arguments"
                )

        # update `self` with these kwargs, if provided
        self.comment = kwargs.pop("comment", self.comment)
        self.implicit_returning = kwargs.pop(
            "implicit_returning", self.implicit_returning
        )
        self.info = kwargs.pop("info", self.info)

        exclude_columns: _typing_Sequence[str]

        if autoload:
            if not autoload_replace:
                # don't replace columns already present.
                # we'd like to do this for constraints also however we don't
                # have simple de-duping for unnamed constraints.
                exclude_columns = [c.name for c in self.c]
            else:
                exclude_columns = ()
            self._autoload(
                self.metadata,
                autoload_with,
                include_columns,
                exclude_columns,
                resolve_fks,
                _extend_on=_extend_on,
                _reflect_info=_reflect_info,
            )

        all_names = {c.name: c for c in self.c}
        self._extra_kwargs(**kwargs)
        self._init_items(*args, allow_replacements=True, all_names=all_names)

    def _extra_kwargs(self, **kwargs: Any) -> None:
        self._validate_dialect_kwargs(kwargs)

    def _init_collections(self) -> None:
        pass

    def _reset_exported(self) -> None:
        pass

    @util.ro_non_memoized_property
    def _autoincrement_column(self) -> Optional[Column[int]]:
        return self.primary_key._autoincrement_column

    @util.ro_memoized_property
    def _sentinel_column_characteristics(
        self,
    ) -> _SentinelColumnCharacterization:
        """determine a candidate column (or columns, in case of a client
        generated composite primary key) which can be used as an
        "insert sentinel" for an INSERT statement.

        The returned structure, :class:`_SentinelColumnCharacterization`,
        includes all the details needed by :class:`.Dialect` and
        :class:`.SQLCompiler` to determine if these column(s) can be used
        as an INSERT..RETURNING sentinel for a particular database
        dialect.

        .. versionadded:: 2.0.10

        """

        sentinel_is_explicit = False
        sentinel_is_autoinc = False
        the_sentinel: Optional[_typing_Sequence[Column[Any]]] = None

        # see if a column was explicitly marked "insert_sentinel=True".
        explicit_sentinel_col = self._sentinel_column

        if explicit_sentinel_col is not None:
            the_sentinel = (explicit_sentinel_col,)
            sentinel_is_explicit = True

        autoinc_col = self._autoincrement_column
        if sentinel_is_explicit and explicit_sentinel_col is autoinc_col:
            assert autoinc_col is not None
            sentinel_is_autoinc = True
        elif explicit_sentinel_col is None and autoinc_col is not None:
            the_sentinel = (autoinc_col,)
            sentinel_is_autoinc = True

        default_characterization = _SentinelDefaultCharacterization.UNKNOWN

        if the_sentinel:
            the_sentinel_zero = the_sentinel[0]
            if the_sentinel_zero.identity:
                if the_sentinel_zero.identity._increment_is_negative:
                    if sentinel_is_explicit:
                        raise exc.InvalidRequestError(
                            "Can't use IDENTITY default with negative "
                            "increment as an explicit sentinel column"
                        )
                    else:
                        if sentinel_is_autoinc:
                            autoinc_col = None
                            sentinel_is_autoinc = False
                        the_sentinel = None
                else:
                    default_characterization = (
                        _SentinelDefaultCharacterization.IDENTITY
                    )
            elif (
                the_sentinel_zero.default is None
                and the_sentinel_zero.server_default is None
            ):
                if the_sentinel_zero.nullable:
                    raise exc.InvalidRequestError(
                        f"Column {the_sentinel_zero} has been marked as a "
                        "sentinel "
                        "column with no default generation function; it "
                        "at least needs to be marked nullable=False assuming "
                        "user-populated sentinel values will be used."
                    )
                default_characterization = (
                    _SentinelDefaultCharacterization.NONE
                )
            elif the_sentinel_zero.default is not None:
                if the_sentinel_zero.default.is_sentinel:
                    default_characterization = (
                        _SentinelDefaultCharacterization.SENTINEL_DEFAULT
                    )
                elif default_is_sequence(the_sentinel_zero.default):
                    if the_sentinel_zero.default._increment_is_negative:
                        if sentinel_is_explicit:
                            raise exc.InvalidRequestError(
                                "Can't use SEQUENCE default with negative "
                                "increment as an explicit sentinel column"
                            )
                        else:
                            if sentinel_is_autoinc:
                                autoinc_col = None
                                sentinel_is_autoinc = False
                            the_sentinel = None

                    default_characterization = (
                        _SentinelDefaultCharacterization.SEQUENCE
                    )
                elif the_sentinel_zero.default.is_callable:
                    default_characterization = (
                        _SentinelDefaultCharacterization.CLIENTSIDE
                    )
            elif the_sentinel_zero.server_default is not None:
                if sentinel_is_explicit:
                    raise exc.InvalidRequestError(
                        f"Column {the_sentinel[0]} can't be a sentinel column "
                        "because it uses an explicit server side default "
                        "that's not the Identity() default."
                    )

                default_characterization = (
                    _SentinelDefaultCharacterization.SERVERSIDE
                )

        if the_sentinel is None and self.primary_key:
            assert autoinc_col is None

            # determine for non-autoincrement pk if all elements are
            # client side
            for _pkc in self.primary_key:
                if _pkc.server_default is not None or (
                    _pkc.default and not _pkc.default.is_callable
                ):
                    break
            else:
                the_sentinel = tuple(self.primary_key)
                default_characterization = (
                    _SentinelDefaultCharacterization.CLIENTSIDE
                )

        return _SentinelColumnCharacterization(
            the_sentinel,
            sentinel_is_explicit,
            sentinel_is_autoinc,
            default_characterization,
        )

    @property
    def autoincrement_column(self) -> Optional[Column[int]]:
        """Returns the :class:`.Column` object which currently represents
        the "auto increment" column, if any, else returns None.

        This is based on the rules for :class:`.Column` as defined by the
        :paramref:`.Column.autoincrement` parameter, which generally means the
        column within a single integer column primary key constraint that is
        not constrained by a foreign key.   If the table does not have such
        a primary key constraint, then there's no "autoincrement" column.
        A :class:`.Table` may have only one column defined as the
        "autoincrement" column.

        .. versionadded:: 2.0.4

        .. seealso::

            :paramref:`.Column.autoincrement`

        """
        return self._autoincrement_column

    @property
    def key(self) -> str:
        """Return the 'key' for this :class:`_schema.Table`.

        This value is used as the dictionary key within the
        :attr:`_schema.MetaData.tables` collection.   It is typically the same
        as that of :attr:`_schema.Table.name` for a table with no
        :attr:`_schema.Table.schema`
        set; otherwise it is typically of the form
        ``schemaname.tablename``.

        """
        return _get_table_key(self.name, self.schema)

    def __repr__(self) -> str:
        return "Table(%s)" % ", ".join(
            [repr(self.name)]
            + [repr(self.metadata)]
            + [repr(x) for x in self.columns]
            + ["%s=%s" % (k, repr(getattr(self, k))) for k in ["schema"]]
        )

    def __str__(self) -> str:
        return _get_table_key(self.description, self.schema)

    def add_is_dependent_on(self, table: Table) -> None:
        """Add a 'dependency' for this Table.

        This is another Table object which must be created
        first before this one can, or dropped after this one.

        Usually, dependencies between tables are determined via
        ForeignKey objects.   However, for other situations that
        create dependencies outside of foreign keys (rules, inheriting),
        this method can manually establish such a link.

        """
        self._extra_dependencies.add(table)

    def append_column(
        self, column: ColumnClause[Any], replace_existing: bool = False
    ) -> None:
        """Append a :class:`_schema.Column` to this :class:`_schema.Table`.

        The "key" of the newly added :class:`_schema.Column`, i.e. the
        value of its ``.key`` attribute, will then be available
        in the ``.c`` collection of this :class:`_schema.Table`, and the
        column definition will be included in any CREATE TABLE, SELECT,
        UPDATE, etc. statements generated from this :class:`_schema.Table`
        construct.

        Note that this does **not** change the definition of the table
        as it exists within any underlying database, assuming that
        table has already been created in the database.   Relational
        databases support the addition of columns to existing tables
        using the SQL ALTER command, which would need to be
        emitted for an already-existing table that doesn't contain
        the newly added column.

        :param replace_existing: When ``True``, allows replacing existing
            columns. When ``False``, the default, an warning will be raised
            if a column with the same ``.key`` already exists. A future
            version of sqlalchemy will instead rise a warning.

            .. versionadded:: 1.4.0
        """

        try:
            column._set_parent_with_dispatch(
                self,
                allow_replacements=replace_existing,
                all_names={c.name: c for c in self.c},
            )
        except exc.DuplicateColumnError as de:
            raise exc.DuplicateColumnError(
                f"{de.args[0]} Specify replace_existing=True to "
                "Table.append_column() to replace an "
                "existing column."
            ) from de

    def append_constraint(self, constraint: Union[Index, Constraint]) -> None:
        """Append a :class:`_schema.Constraint` to this
        :class:`_schema.Table`.

        This has the effect of the constraint being included in any
        future CREATE TABLE statement, assuming specific DDL creation
        events have not been associated with the given
        :class:`_schema.Constraint` object.

        Note that this does **not** produce the constraint within the
        relational database automatically, for a table that already exists
        in the database.   To add a constraint to an
        existing relational database table, the SQL ALTER command must
        be used.  SQLAlchemy also provides the
        :class:`.AddConstraint` construct which can produce this SQL when
        invoked as an executable clause.

        """

        constraint._set_parent_with_dispatch(self)

    def _set_parent(self, parent: SchemaEventTarget, **kw: Any) -> None:
        metadata = parent
        assert isinstance(metadata, MetaData)
        metadata._add_table(self.name, self.schema, self)
        self.metadata = metadata

    def create(self, bind: _CreateDropBind, checkfirst: bool = False) -> None:
        """Issue a ``CREATE`` statement for this
        :class:`_schema.Table`, using the given
        :class:`.Connection` or :class:`.Engine`
        for connectivity.

        .. seealso::

            :meth:`_schema.MetaData.create_all`.

        """

        bind._run_ddl_visitor(ddl.SchemaGenerator, self, checkfirst=checkfirst)

    def drop(self, bind: _CreateDropBind, checkfirst: bool = False) -> None:
        """Issue a ``DROP`` statement for this
        :class:`_schema.Table`, using the given
        :class:`.Connection` or :class:`.Engine` for connectivity.

        .. seealso::

            :meth:`_schema.MetaData.drop_all`.

        """
        bind._run_ddl_visitor(ddl.SchemaDropper, self, checkfirst=checkfirst)

    @util.deprecated(
        "1.4",
        ":meth:`_schema.Table.tometadata` is renamed to "
        ":meth:`_schema.Table.to_metadata`",
    )
    def tometadata(
        self,
        metadata: MetaData,
        schema: Union[str, Literal[SchemaConst.RETAIN_SCHEMA]] = RETAIN_SCHEMA,
        referred_schema_fn: Optional[
            Callable[
                [Table, Optional[str], ForeignKeyConstraint, Optional[str]],
                Optional[str],
            ]
        ] = None,
        name: Optional[str] = None,
    ) -> Table:
        """Return a copy of this :class:`_schema.Table`
        associated with a different
        :class:`_schema.MetaData`.

        See :meth:`_schema.Table.to_metadata` for a full description.

        """
        return self.to_metadata(
            metadata,
            schema=schema,
            referred_schema_fn=referred_schema_fn,
            name=name,
        )

    def to_metadata(
        self,
        metadata: MetaData,
        schema: Union[str, Literal[SchemaConst.RETAIN_SCHEMA]] = RETAIN_SCHEMA,
        referred_schema_fn: Optional[
            Callable[
                [Table, Optional[str], ForeignKeyConstraint, Optional[str]],
                Optional[str],
            ]
        ] = None,
        name: Optional[str] = None,
    ) -> Table:
        """Return a copy of this :class:`_schema.Table` associated with a
        different :class:`_schema.MetaData`.

        E.g.::

            m1 = MetaData()

            user = Table("user", m1, Column("id", Integer, primary_key=True))

            m2 = MetaData()
            user_copy = user.to_metadata(m2)

        .. versionchanged:: 1.4  The :meth:`_schema.Table.to_metadata` function
           was renamed from :meth:`_schema.Table.tometadata`.


        :param metadata: Target :class:`_schema.MetaData` object,
         into which the
         new :class:`_schema.Table` object will be created.

        :param schema: optional string name indicating the target schema.
         Defaults to the special symbol :attr:`.RETAIN_SCHEMA` which indicates
         that no change to the schema name should be made in the new
         :class:`_schema.Table`.  If set to a string name, the new
         :class:`_schema.Table`
         will have this new name as the ``.schema``.  If set to ``None``, the
         schema will be set to that of the schema set on the target
         :class:`_schema.MetaData`, which is typically ``None`` as well,
         unless
         set explicitly::

            m2 = MetaData(schema="newschema")

            # user_copy_one will have "newschema" as the schema name
            user_copy_one = user.to_metadata(m2, schema=None)

            m3 = MetaData()  # schema defaults to None

            # user_copy_two will have None as the schema name
            user_copy_two = user.to_metadata(m3, schema=None)

        :param referred_schema_fn: optional callable which can be supplied
         in order to provide for the schema name that should be assigned
         to the referenced table of a :class:`_schema.ForeignKeyConstraint`.
         The callable accepts this parent :class:`_schema.Table`, the
         target schema that we are changing to, the
         :class:`_schema.ForeignKeyConstraint` object, and the existing
         "target schema" of that constraint.  The function should return the
         string schema name that should be applied.    To reset the schema
         to "none", return the symbol :data:`.BLANK_SCHEMA`.  To effect no
         change, return ``None`` or :data:`.RETAIN_SCHEMA`.

         .. versionchanged:: 1.4.33  The ``referred_schema_fn`` function
            may return the :data:`.BLANK_SCHEMA` or :data:`.RETAIN_SCHEMA`
            symbols.

         E.g.::

                def referred_schema_fn(table, to_schema, constraint, referred_schema):
                    if referred_schema == "base_tables":
                        return referred_schema
                    else:
                        return to_schema


                new_table = table.to_metadata(
                    m2, schema="alt_schema", referred_schema_fn=referred_schema_fn
                )

        :param name: optional string name indicating the target table name.
         If not specified or None, the table name is retained.  This allows
         a :class:`_schema.Table` to be copied to the same
         :class:`_schema.MetaData` target
         with a new name.

        """  # noqa: E501
        if name is None:
            name = self.name

        actual_schema: Optional[str]

        if schema is RETAIN_SCHEMA:
            actual_schema = self.schema
        elif schema is None:
            actual_schema = metadata.schema
        else:
            actual_schema = schema
        key = _get_table_key(name, actual_schema)
        if key in metadata.tables:
            util.warn(
                f"Table '{self.description}' already exists within the given "
                "MetaData - not copying."
            )
            return metadata.tables[key]

        args = []
        for col in self.columns:
            args.append(col._copy(schema=actual_schema, _to_metadata=metadata))
        table = Table(
            name,
            metadata,
            schema=actual_schema,
            comment=self.comment,
            *args,
            **self.kwargs,
        )
        for const in self.constraints:
            if isinstance(const, ForeignKeyConstraint):
                referred_schema = const._referred_schema
                if referred_schema_fn:
                    fk_constraint_schema = referred_schema_fn(
                        self, actual_schema, const, referred_schema
                    )
                else:
                    fk_constraint_schema = (
                        actual_schema
                        if referred_schema == self.schema
                        else None
                    )
                table.append_constraint(
                    const._copy(
                        schema=fk_constraint_schema, target_table=table
                    )
                )
            elif not const._type_bound:
                # skip unique constraints that would be generated
                # by the 'unique' flag on Column
                if const._column_flag:
                    continue

                table.append_constraint(
                    const._copy(schema=actual_schema, target_table=table)
                )
        for index in self.indexes:
            # skip indexes that would be generated
            # by the 'index' flag on Column
            if index._column_flag:
                continue
            Index(
                index.name,
                unique=index.unique,
                *[
                    _copy_expression(expr, self, table)
                    for expr in index._table_bound_expressions
                ],
                _table=table,
                **index.kwargs,
            )
        return self._schema_item_copy(table)


class Column(DialectKWArgs, SchemaItem, ColumnClause[_T]):
    """Represents a column in a database table."""

    __visit_name__ = "column"

    inherit_cache = True
    key: str

    server_default: Optional[FetchedValue]

    def __init__(
        self,
        __name_pos: Optional[
            Union[str, _TypeEngineArgument[_T], SchemaEventTarget]
        ] = None,
        __type_pos: Optional[
            Union[_TypeEngineArgument[_T], SchemaEventTarget]
        ] = None,
        *args: SchemaEventTarget,
        name: Optional[str] = None,
        type_: Optional[_TypeEngineArgument[_T]] = None,
        autoincrement: _AutoIncrementType = "auto",
        default: Optional[Any] = _NoArg.NO_ARG,
        insert_default: Optional[Any] = _NoArg.NO_ARG,
        doc: Optional[str] = None,
        key: Optional[str] = None,
        index: Optional[bool] = None,
        unique: Optional[bool] = None,
        info: Optional[_InfoType] = None,
        nullable: Optional[
            Union[bool, Literal[SchemaConst.NULL_UNSPECIFIED]]
        ] = SchemaConst.NULL_UNSPECIFIED,
        onupdate: Optional[Any] = None,
        primary_key: bool = False,
        server_default: Optional[_ServerDefaultArgument] = None,
        server_onupdate: Optional[_ServerOnUpdateArgument] = None,
        quote: Optional[bool] = None,
        system: bool = False,
        comment: Optional[str] = None,
        insert_sentinel: bool = False,
        _omit_from_statements: bool = False,
        _proxies: Optional[Any] = None,
        **dialect_kwargs: Any,
    ):
        r"""
        Construct a new ``Column`` object.

        :param name: The name of this column as represented in the database.
          This argument may be the first positional argument, or specified
          via keyword.

          Names which contain no upper case characters
          will be treated as case insensitive names, and will not be quoted
          unless they are a reserved word.  Names with any number of upper
          case characters will be quoted and sent exactly.  Note that this
          behavior applies even for databases which standardize upper
          case names as case insensitive such as Oracle Database.

          The name field may be omitted at construction time and applied
          later, at any time before the Column is associated with a
          :class:`_schema.Table`.  This is to support convenient
          usage within the :mod:`~sqlalchemy.ext.declarative` extension.

        :param type\_: The column's type, indicated using an instance which
          subclasses :class:`~sqlalchemy.types.TypeEngine`.  If no arguments
          are required for the type, the class of the type can be sent
          as well, e.g.::

            # use a type with arguments
            Column("data", String(50))

            # use no arguments
            Column("level", Integer)

          The ``type`` argument may be the second positional argument
          or specified by keyword.

          If the ``type`` is ``None`` or is omitted, it will first default to
          the special type :class:`.NullType`.  If and when this
          :class:`_schema.Column` is made to refer to another column using
          :class:`_schema.ForeignKey` and/or
          :class:`_schema.ForeignKeyConstraint`, the type
          of the remote-referenced column will be copied to this column as
          well, at the moment that the foreign key is resolved against that
          remote :class:`_schema.Column` object.

        :param \*args: Additional positional arguments include various
          :class:`.SchemaItem` derived constructs which will be applied
          as options to the column.  These include instances of
          :class:`.Constraint`, :class:`_schema.ForeignKey`,
          :class:`.ColumnDefault`, :class:`.Sequence`, :class:`.Computed`
          :class:`.Identity`.  In some cases an
          equivalent keyword argument is available such as ``server_default``,
          ``default`` and ``unique``.

        :param autoincrement: Set up "auto increment" semantics for an
          **integer primary key column with no foreign key dependencies**
          (see later in this docstring for a more specific definition).
          This may influence the :term:`DDL` that will be emitted for
          this column during a table create, as well as how the column
          will be considered when INSERT statements are compiled and
          executed.

          The default value is the string ``"auto"``,
          which indicates that a single-column (i.e. non-composite) primary key
          that is of an INTEGER type with no other client-side or server-side
          default constructs indicated should receive auto increment semantics
          automatically. Other values include ``True`` (force this column to
          have auto-increment semantics for a :term:`composite primary key` as
          well), ``False`` (this column should never have auto-increment
          semantics), and the string ``"ignore_fk"`` (special-case for foreign
          key columns, see below).

          The term "auto increment semantics" refers both to the kind of DDL
          that will be emitted for the column within a CREATE TABLE statement,
          when methods such as :meth:`.MetaData.create_all` and
          :meth:`.Table.create` are invoked, as well as how the column will be
          considered when an INSERT statement is compiled and emitted to the
          database:

          * **DDL rendering** (i.e. :meth:`.MetaData.create_all`,
            :meth:`.Table.create`): When used on a :class:`.Column` that has
            no other
            default-generating construct associated with it (such as a
            :class:`.Sequence` or :class:`.Identity` construct), the parameter
            will imply that database-specific keywords such as PostgreSQL
            ``SERIAL``, MySQL ``AUTO_INCREMENT``, or ``IDENTITY`` on SQL Server
            should also be rendered.  Not every database backend has an
            "implied" default generator available; for example the Oracle Database
            backends alway needs an explicit construct such as
            :class:`.Identity` to be included with a :class:`.Column` in order
            for the DDL rendered to include auto-generating constructs to also
            be produced in the database.

          * **INSERT semantics** (i.e. when a :func:`_sql.insert` construct is
            compiled into a SQL string and is then executed on a database using
            :meth:`_engine.Connection.execute` or equivalent): A single-row
            INSERT statement will be known to produce a new integer primary key
            value automatically for this column, which will be accessible
            after the statement is invoked via the
            :attr:`.CursorResult.inserted_primary_key` attribute upon the
            :class:`_result.Result` object.   This also applies towards use of the
            ORM when ORM-mapped objects are persisted to the database,
            indicating that a new integer primary key will be available to
            become part of the :term:`identity key` for that object.  This
            behavior takes place regardless of what DDL constructs are
            associated with the :class:`_schema.Column` and is independent
            of the "DDL Rendering" behavior discussed in the previous note
            above.

          The parameter may be set to ``True`` to indicate that a column which
          is part of a composite (i.e. multi-column) primary key should
          have autoincrement semantics, though note that only one column
          within a primary key may have this setting.    It can also
          be set to ``True`` to indicate autoincrement semantics on a
          column that has a client-side or server-side default configured,
          however note that not all dialects can accommodate all styles
          of default as an "autoincrement".  It can also be
          set to ``False`` on a single-column primary key that has a
          datatype of INTEGER in order to disable auto increment semantics
          for that column.

          The setting *only* has an effect for columns which are:

          * Integer derived (i.e. INT, SMALLINT, BIGINT).

          * Part of the primary key

          * Not referring to another column via :class:`_schema.ForeignKey`,
            unless
            the value is specified as ``'ignore_fk'``::

                # turn on autoincrement for this column despite
                # the ForeignKey()
                Column(
                    "id",
                    ForeignKey("other.id"),
                    primary_key=True,
                    autoincrement="ignore_fk",
                )

          It is typically not desirable to have "autoincrement" enabled on a
          column that refers to another via foreign key, as such a column is
          required to refer to a value that originates from elsewhere.

          The setting has these effects on columns that meet the
          above criteria:

          * DDL issued for the column, if the column does not already include
            a default generating construct supported by the backend such as
            :class:`.Identity`, will include database-specific
            keywords intended to signify this column as an
            "autoincrement" column for specific backends.   Behavior for
            primary SQLAlchemy dialects includes:

            * AUTO INCREMENT on MySQL and MariaDB
            * SERIAL on PostgreSQL
            * IDENTITY on MS-SQL - this occurs even without the
              :class:`.Identity` construct as the
              :paramref:`.Column.autoincrement` parameter pre-dates this
              construct.
            * SQLite - SQLite integer primary key columns are implicitly
              "auto incrementing" and no additional keywords are rendered;
              to render the special SQLite keyword ``AUTOINCREMENT``
              is not included as this is unnecessary and not recommended
              by the database vendor.  See the section
              :ref:`sqlite_autoincrement` for more background.
            * Oracle Database - The Oracle Database dialects have no default "autoincrement"
              feature available at this time, instead the :class:`.Identity`
              construct is recommended to achieve this (the :class:`.Sequence`
              construct may also be used).
            * Third-party dialects - consult those dialects' documentation
              for details on their specific behaviors.

          * When a single-row :func:`_sql.insert` construct is compiled and
            executed, which does not set the :meth:`_sql.Insert.inline`
            modifier, newly generated primary key values for this column
            will be automatically retrieved upon statement execution
            using a method specific to the database driver in use:

            * MySQL, SQLite - calling upon ``cursor.lastrowid()``
              (see
              `https://www.python.org/dev/peps/pep-0249/#lastrowid
              <https://www.python.org/dev/peps/pep-0249/#lastrowid>`_)
            * PostgreSQL, SQL Server, Oracle Database - use RETURNING or an equivalent
              construct when rendering an INSERT statement, and then retrieving
              the newly generated primary key values after execution
            * PostgreSQL, Oracle Database for :class:`_schema.Table` objects that
              set :paramref:`_schema.Table.implicit_returning` to False -
              for a :class:`.Sequence` only, the :class:`.Sequence` is invoked
              explicitly before the INSERT statement takes place so that the
              newly generated primary key value is available to the client
            * SQL Server for :class:`_schema.Table` objects that
              set :paramref:`_schema.Table.implicit_returning` to False -
              the ``SELECT scope_identity()`` construct is used after the
              INSERT statement is invoked to retrieve the newly generated
              primary key value.
            * Third-party dialects - consult those dialects' documentation
              for details on their specific behaviors.

          * For multiple-row :func:`_sql.insert` constructs invoked with
            a list of parameters (i.e. "executemany" semantics), primary-key
            retrieving behaviors are generally disabled, however there may
            be special APIs that may be used to retrieve lists of new
            primary key values for an "executemany", such as the psycopg2
            "fast insertmany" feature.  Such features are very new and
            may not yet be well covered in documentation.

        :param default: A scalar, Python callable, or
            :class:`_expression.ColumnElement` expression representing the
            *default value* for this column, which will be invoked upon insert
            if this column is otherwise not specified in the VALUES clause of
            the insert. This is a shortcut to using :class:`.ColumnDefault` as
            a positional argument; see that class for full detail on the
            structure of the argument.

            Contrast this argument to
            :paramref:`_schema.Column.server_default`
            which creates a default generator on the database side.

            .. seealso::

                :ref:`metadata_defaults_toplevel`

        :param insert_default: An alias of :paramref:`.Column.default`
            for compatibility with :func:`_orm.mapped_column`.

            .. versionadded: 2.0.31

        :param doc: optional String that can be used by the ORM or similar
            to document attributes on the Python side.   This attribute does
            **not** render SQL comments; use the
            :paramref:`_schema.Column.comment`
            parameter for this purpose.

        :param key: An optional string identifier which will identify this
            ``Column`` object on the :class:`_schema.Table`.
            When a key is provided,
            this is the only identifier referencing the ``Column`` within the
            application, including ORM attribute mapping; the ``name`` field
            is used only when rendering SQL.

        :param index: When ``True``, indicates that a :class:`_schema.Index`
            construct will be automatically generated for this
            :class:`_schema.Column`, which will result in a "CREATE INDEX"
            statement being emitted for the :class:`_schema.Table` when the DDL
            create operation is invoked.

            Using this flag is equivalent to making use of the
            :class:`_schema.Index` construct explicitly at the level of the
            :class:`_schema.Table` construct itself::

                Table(
                    "some_table",
                    metadata,
                    Column("x", Integer),
                    Index("ix_some_table_x", "x"),
                )

            To add the :paramref:`_schema.Index.unique` flag to the
            :class:`_schema.Index`, set both the
            :paramref:`_schema.Column.unique` and
            :paramref:`_schema.Column.index` flags to True simultaneously,
            which will have the effect of rendering the "CREATE UNIQUE INDEX"
            DDL instruction instead of "CREATE INDEX".

            The name of the index is generated using the
            :ref:`default naming convention <constraint_default_naming_convention>`
            which for the :class:`_schema.Index` construct is of the form
            ``ix_<tablename>_<columnname>``.

            As this flag is intended only as a convenience for the common case
            of adding a single-column, default configured index to a table
            definition, explicit use of the :class:`_schema.Index` construct
            should be preferred for most use cases, including composite indexes
            that encompass more than one column, indexes with SQL expressions
            or ordering, backend-specific index configuration options, and
            indexes that use a specific name.

            .. note:: the :attr:`_schema.Column.index` attribute on
               :class:`_schema.Column`
               **does not indicate** if this column is indexed or not, only
               if this flag was explicitly set here.  To view indexes on
               a column, view the :attr:`_schema.Table.indexes` collection
               or use :meth:`_reflection.Inspector.get_indexes`.

            .. seealso::

                :ref:`schema_indexes`

                :ref:`constraint_naming_conventions`

                :paramref:`_schema.Column.unique`

        :param info: Optional data dictionary which will be populated into the
            :attr:`.SchemaItem.info` attribute of this object.

        :param nullable: When set to ``False``, will cause the "NOT NULL"
            phrase to be added when generating DDL for the column.   When
            ``True``, will normally generate nothing (in SQL this defaults to
            "NULL"), except in some very specific backend-specific edge cases
            where "NULL" may render explicitly.
            Defaults to ``True`` unless :paramref:`_schema.Column.primary_key`
            is also ``True`` or the column specifies a :class:`_sql.Identity`,
            in which case it defaults to ``False``.
            This parameter is only used when issuing CREATE TABLE statements.

            .. note::

                When the column specifies a :class:`_sql.Identity` this
                parameter is in general ignored by the DDL compiler. The
                PostgreSQL database allows nullable identity column by
                setting this parameter to ``True`` explicitly.

        :param onupdate: A scalar, Python callable, or
            :class:`~sqlalchemy.sql.expression.ClauseElement` representing a
            default value to be applied to the column within UPDATE
            statements, which will be invoked upon update if this column is not
            present in the SET clause of the update. This is a shortcut to
            using :class:`.ColumnDefault` as a positional argument with
            ``for_update=True``.

            .. seealso::

                :ref:`metadata_defaults` - complete discussion of onupdate

        :param primary_key: If ``True``, marks this column as a primary key
            column. Multiple columns can have this flag set to specify
            composite primary keys. As an alternative, the primary key of a
            :class:`_schema.Table` can be specified via an explicit
            :class:`.PrimaryKeyConstraint` object.

        :param server_default: A :class:`.FetchedValue` instance, str, Unicode
            or :func:`~sqlalchemy.sql.expression.text` construct representing
            the DDL DEFAULT value for the column.

            String types will be emitted as-is, surrounded by single quotes::

                Column("x", Text, server_default="val")

            will render:

            .. sourcecode:: sql

                x TEXT DEFAULT 'val'

            A :func:`~sqlalchemy.sql.expression.text` expression will be
            rendered as-is, without quotes::

                Column("y", DateTime, server_default=text("NOW()"))

            will render:

            .. sourcecode:: sql

                y DATETIME DEFAULT NOW()

            Strings and text() will be converted into a
            :class:`.DefaultClause` object upon initialization.

            This parameter can also accept complex combinations of contextually
            valid SQLAlchemy expressions or constructs::

                from sqlalchemy import create_engine
                from sqlalchemy import Table, Column, MetaData, ARRAY, Text
                from sqlalchemy.dialects.postgresql import array

                engine = create_engine(
                    "postgresql+psycopg2://scott:tiger@localhost/mydatabase"
                )
                metadata_obj = MetaData()
                tbl = Table(
                    "foo",
                    metadata_obj,
                    Column(
                        "bar", ARRAY(Text), server_default=array(["biz", "bang", "bash"])
                    ),
                )
                metadata_obj.create_all(engine)

            The above results in a table created with the following SQL:

            .. sourcecode:: sql

                CREATE TABLE foo (
                    bar TEXT[] DEFAULT ARRAY['biz', 'bang', 'bash']
                )

            Use :class:`.FetchedValue` to indicate that an already-existing
            column will generate a default value on the database side which
            will be available to SQLAlchemy for post-fetch after inserts. This
            construct does not specify any DDL and the implementation is left
            to the database, such as via a trigger.

            .. seealso::

                :ref:`server_defaults` - complete discussion of server side
                defaults

        :param server_onupdate: A :class:`.FetchedValue` instance
            representing a database-side default generation function,
            such as a trigger. This
            indicates to SQLAlchemy that a newly generated value will be
            available after updates. This construct does not actually
            implement any kind of generation function within the database,
            which instead must be specified separately.


            .. warning:: This directive **does not** currently produce MySQL's
               "ON UPDATE CURRENT_TIMESTAMP()" clause.  See
               :ref:`mysql_timestamp_onupdate` for background on how to
               produce this clause.

            .. seealso::

                :ref:`triggered_columns`

        :param quote: Force quoting of this column's name on or off,
             corresponding to ``True`` or ``False``. When left at its default
             of ``None``, the column identifier will be quoted according to
             whether the name is case sensitive (identifiers with at least one
             upper case character are treated as case sensitive), or if it's a
             reserved word. This flag is only needed to force quoting of a
             reserved word which is not known by the SQLAlchemy dialect.

        :param unique: When ``True``, and the :paramref:`_schema.Column.index`
            parameter is left at its default value of ``False``,
            indicates that a :class:`_schema.UniqueConstraint`
            construct will be automatically generated for this
            :class:`_schema.Column`,
            which will result in a "UNIQUE CONSTRAINT" clause referring
            to this column being included
            in the ``CREATE TABLE`` statement emitted, when the DDL create
            operation for the :class:`_schema.Table` object is invoked.

            When this flag is ``True`` while the
            :paramref:`_schema.Column.index` parameter is simultaneously
            set to ``True``, the effect instead is that a
            :class:`_schema.Index` construct which includes the
            :paramref:`_schema.Index.unique` parameter set to ``True``
            is generated.  See the documentation for
            :paramref:`_schema.Column.index` for additional detail.

            Using this flag is equivalent to making use of the
            :class:`_schema.UniqueConstraint` construct explicitly at the
            level of the :class:`_schema.Table` construct itself::

                Table("some_table", metadata, Column("x", Integer), UniqueConstraint("x"))

            The :paramref:`_schema.UniqueConstraint.name` parameter
            of the unique constraint object is left at its default value
            of ``None``; in the absence of a :ref:`naming convention <constraint_naming_conventions>`
            for the enclosing :class:`_schema.MetaData`, the UNIQUE CONSTRAINT
            construct will be emitted as unnamed, which typically invokes
            a database-specific naming convention to take place.

            As this flag is intended only as a convenience for the common case
            of adding a single-column, default configured unique constraint to a table
            definition, explicit use of the :class:`_schema.UniqueConstraint` construct
            should be preferred for most use cases, including composite constraints
            that encompass more than one column, backend-specific index configuration options, and
            constraints that use a specific name.

            .. note:: the :attr:`_schema.Column.unique` attribute on
                :class:`_schema.Column`
                **does not indicate** if this column has a unique constraint or
                not, only if this flag was explicitly set here.  To view
                indexes and unique constraints that may involve this column,
                view the
                :attr:`_schema.Table.indexes` and/or
                :attr:`_schema.Table.constraints` collections or use
                :meth:`_reflection.Inspector.get_indexes` and/or
                :meth:`_reflection.Inspector.get_unique_constraints`

            .. seealso::

                :ref:`schema_unique_constraint`

                :ref:`constraint_naming_conventions`

                :paramref:`_schema.Column.index`

        :param system: When ``True``, indicates this is a "system" column,
             that is a column which is automatically made available by the
             database, and should not be included in the columns list for a
             ``CREATE TABLE`` statement.

             For more elaborate scenarios where columns should be
             conditionally rendered differently on different backends,
             consider custom compilation rules for :class:`.CreateColumn`.

        :param comment: Optional string that will render an SQL comment on
             table creation.

             .. versionadded:: 1.2 Added the
                :paramref:`_schema.Column.comment`
                parameter to :class:`_schema.Column`.

        :param insert_sentinel: Marks this :class:`_schema.Column` as an
         :term:`insert sentinel` used for optimizing the performance of the
         :term:`insertmanyvalues` feature for tables that don't
         otherwise have qualifying primary key configurations.

         .. versionadded:: 2.0.10

         .. seealso::

            :func:`_schema.insert_sentinel` - all in one helper for declaring
            sentinel columns

            :ref:`engine_insertmanyvalues`

            :ref:`engine_insertmanyvalues_sentinel_columns`


        """  # noqa: E501, RST201, RST202

        l_args = [__name_pos, __type_pos] + list(args)
        del args

        if l_args:
            if isinstance(l_args[0], str):
                if name is not None:
                    raise exc.ArgumentError(
                        "May not pass name positionally and as a keyword."
                    )
                name = l_args.pop(0)  # type: ignore
            elif l_args[0] is None:
                l_args.pop(0)
        if l_args:
            coltype = l_args[0]

            if hasattr(coltype, "_sqla_type"):
                if type_ is not None:
                    raise exc.ArgumentError(
                        "May not pass type_ positionally and as a keyword."
                    )
                type_ = l_args.pop(0)  # type: ignore
            elif l_args[0] is None:
                l_args.pop(0)

        if name is not None:
            name = quoted_name(name, quote)
        elif quote is not None:
            raise exc.ArgumentError(
                "Explicit 'name' is required when sending 'quote' argument"
            )

        # name = None is expected to be an interim state
        # note this use case is legacy now that ORM declarative has a
        # dedicated "column" construct local to the ORM
        super().__init__(name, type_)  # type: ignore

        self.key = key if key is not None else name  # type: ignore
        self.primary_key = primary_key
        self._insert_sentinel = insert_sentinel
        self._omit_from_statements = _omit_from_statements
        self._user_defined_nullable = udn = nullable
        if udn is not NULL_UNSPECIFIED:
            self.nullable = udn
        else:
            self.nullable = not primary_key

        # these default to None because .index and .unique is *not*
        # an informational flag about Column - there can still be an
        # Index or UniqueConstraint referring to this Column.
        self.index = index
        self.unique = unique

        self.system = system
        self.doc = doc
        self.autoincrement: _AutoIncrementType = autoincrement
        self.constraints = set()
        self.foreign_keys = set()
        self.comment = comment
        self.computed = None
        self.identity = None

        # check if this Column is proxying another column

        if _proxies is not None:
            self._proxies = _proxies
        else:
            # otherwise, add DDL-related events
            self._set_type(self.type)

        if insert_default is not _NoArg.NO_ARG:
            resolved_default = insert_default
        elif default is not _NoArg.NO_ARG:
            resolved_default = default
        else:
            resolved_default = None

        if resolved_default is not None:
            if not isinstance(resolved_default, (ColumnDefault, Sequence)):
                resolved_default = ColumnDefault(resolved_default)

            self.default = resolved_default
            l_args.append(resolved_default)
        else:
            self.default = None

        if onupdate is not None:
            if not isinstance(onupdate, (ColumnDefault, Sequence)):
                onupdate = ColumnDefault(onupdate, for_update=True)

            self.onupdate = onupdate
            l_args.append(onupdate)
        else:
            self.onupdate = None

        if server_default is not None:
            if isinstance(server_default, FetchedValue):
                server_default = server_default._as_for_update(False)
                l_args.append(server_default)
            else:
                server_default = DefaultClause(server_default)
                l_args.append(server_default)
        self.server_default = server_default

        if server_onupdate is not None:
            if isinstance(server_onupdate, FetchedValue):
                server_onupdate = server_onupdate._as_for_update(True)
                l_args.append(server_onupdate)
            else:
                server_onupdate = DefaultClause(
                    server_onupdate, for_update=True
                )
                l_args.append(server_onupdate)
        self.server_onupdate = server_onupdate

        self._init_items(*cast(_typing_Sequence[SchemaItem], l_args))

        util.set_creation_order(self)

        if info is not None:
            self.info = info

        self._extra_kwargs(**dialect_kwargs)

    table: Table

    constraints: Set[Constraint]

    foreign_keys: Set[ForeignKey]
    """A collection of all :class:`_schema.ForeignKey` marker objects
       associated with this :class:`_schema.Column`.

       Each object is a member of a :class:`_schema.Table`-wide
       :class:`_schema.ForeignKeyConstraint`.

       .. seealso::

           :attr:`_schema.Table.foreign_keys`

    """

    index: Optional[bool]
    """The value of the :paramref:`_schema.Column.index` parameter.

       Does not indicate if this :class:`_schema.Column` is actually indexed
       or not; use :attr:`_schema.Table.indexes`.

       .. seealso::

           :attr:`_schema.Table.indexes`
    """

    unique: Optional[bool]
    """The value of the :paramref:`_schema.Column.unique` parameter.

       Does not indicate if this :class:`_schema.Column` is actually subject to
       a unique constraint or not; use :attr:`_schema.Table.indexes` and
       :attr:`_schema.Table.constraints`.

       .. seealso::

           :attr:`_schema.Table.indexes`

           :attr:`_schema.Table.constraints`.

    """

    computed: Optional[Computed]

    identity: Optional[Identity]

    def _set_type(self, type_: TypeEngine[Any]) -> None:
        assert self.type._isnull or type_ is self.type

        self.type = type_
        if isinstance(self.type, SchemaEventTarget):
            self.type._set_parent_with_dispatch(self)
        for impl in self.type._variant_mapping.values():
            if isinstance(impl, SchemaEventTarget):
                impl._set_parent_with_dispatch(self)

    @HasMemoized.memoized_attribute
    def _default_description_tuple(self) -> _DefaultDescriptionTuple:
        """used by default.py -> _process_execute_defaults()"""

        return _DefaultDescriptionTuple._from_column_default(self.default)

    @HasMemoized.memoized_attribute
    def _onupdate_description_tuple(self) -> _DefaultDescriptionTuple:
        """used by default.py -> _process_execute_defaults()"""
        return _DefaultDescriptionTuple._from_column_default(self.onupdate)

    @util.memoized_property
    def _gen_static_annotations_cache_key(self) -> bool:  # type: ignore
        """special attribute used by cache key gen, if true, we will
        use a static cache key for the annotations dictionary, else we
        will generate a new cache key for annotations each time.

        Added for #8790

        """
        return self.table is not None and self.table._is_table

    def _extra_kwargs(self, **kwargs: Any) -> None:
        self._validate_dialect_kwargs(kwargs)

    def __str__(self) -> str:
        if self.name is None:
            return "(no name)"
        elif self.table is not None:
            if self.table.named_with_column:
                return self.table.description + "." + self.description
            else:
                return self.description
        else:
            return self.description

    def references(self, column: Column[Any]) -> bool:
        """Return True if this Column references the given column via foreign
        key."""

        for fk in self.foreign_keys:
            if fk.column.proxy_set.intersection(column.proxy_set):
                return True
        else:
            return False

    def append_foreign_key(self, fk: ForeignKey) -> None:
        fk._set_parent_with_dispatch(self)

    def __repr__(self) -> str:
        kwarg = []
        if self.key != self.name:
            kwarg.append("key")
        if self.primary_key:
            kwarg.append("primary_key")
        if not self.nullable:
            kwarg.append("nullable")
        if self.onupdate:
            kwarg.append("onupdate")
        if self.default:
            kwarg.append("default")
        if self.server_default:
            kwarg.append("server_default")
        if self.comment:
            kwarg.append("comment")
        return "Column(%s)" % ", ".join(
            [repr(self.name)]
            + [repr(self.type)]
            + [repr(x) for x in self.foreign_keys if x is not None]
            + [repr(x) for x in self.constraints]
            + [
                (
                    self.table is not None
                    and "table=<%s>" % self.table.description
                    or "table=None"
                )
            ]
            + ["%s=%s" % (k, repr(getattr(self, k))) for k in kwarg]
        )

    def _set_parent(  # type: ignore[override]
        self,
        parent: SchemaEventTarget,
        *,
        all_names: Dict[str, Column[Any]],
        allow_replacements: bool,
        **kw: Any,
    ) -> None:
        table = parent
        assert isinstance(table, Table)
        if not self.name:
            raise exc.ArgumentError(
                "Column must be constructed with a non-blank name or "
                "assign a non-blank .name before adding to a Table."
            )

        self._reset_memoizations()

        if self.key is None:
            self.key = self.name

        existing = getattr(self, "table", None)
        if existing is not None and existing is not table:
            raise exc.ArgumentError(
                f"Column object '{self.key}' already "
                f"assigned to Table '{existing.description}'"
            )

        extra_remove = None
        existing_col = None
        conflicts_on = ""

        if self.key in table._columns:
            existing_col = table._columns[self.key]
            if self.key == self.name:
                conflicts_on = "name"
            else:
                conflicts_on = "key"
        elif self.name in all_names:
            existing_col = all_names[self.name]
            extra_remove = {existing_col}
            conflicts_on = "name"

        if existing_col is not None:
            if existing_col is not self:
                if not allow_replacements:
                    raise exc.DuplicateColumnError(
                        f"A column with {conflicts_on} "
                        f"""'{
                            self.key if conflicts_on == 'key' else self.name
                        }' """
                        f"is already present in table '{table.name}'."
                    )
                for fk in existing_col.foreign_keys:
                    table.foreign_keys.remove(fk)
                    if fk.constraint in table.constraints:
                        # this might have been removed
                        # already, if it's a composite constraint
                        # and more than one col being replaced
                        table.constraints.remove(fk.constraint)

        if extra_remove and existing_col is not None and self.key == self.name:
            util.warn(
                f'Column with user-specified key "{existing_col.key}" is '
                "being replaced with "
                f'plain named column "{self.name}", '
                f'key "{existing_col.key}" is being removed.  If this is a '
                "reflection operation, specify autoload_replace=False to "
                "prevent this replacement."
            )
        table._columns.replace(self, extra_remove=extra_remove)
        all_names[self.name] = self
        self.table = table

        if self._insert_sentinel:
            if self.table._sentinel_column is not None:
                raise exc.ArgumentError(
                    "a Table may have only one explicit sentinel column"
                )
            self.table._sentinel_column = self

        if self.primary_key:
            table.primary_key._replace(self)
        elif self.key in table.primary_key:
            raise exc.ArgumentError(
                f"Trying to redefine primary-key column '{self.key}' as a "
                f"non-primary-key column on table '{table.fullname}'"
            )

        if self.index:
            if isinstance(self.index, str):
                raise exc.ArgumentError(
                    "The 'index' keyword argument on Column is boolean only. "
                    "To create indexes with a specific name, create an "
                    "explicit Index object external to the Table."
                )
            table.append_constraint(
                Index(
                    None, self.key, unique=bool(self.unique), _column_flag=True
                )
            )

        elif self.unique:
            if isinstance(self.unique, str):
                raise exc.ArgumentError(
                    "The 'unique' keyword argument on Column is boolean "
                    "only. To create unique constraints or indexes with a "
                    "specific name, append an explicit UniqueConstraint to "
                    "the Table's list of elements, or create an explicit "
                    "Index object external to the Table."
                )
            table.append_constraint(
                UniqueConstraint(self.key, _column_flag=True)
            )

        self._setup_on_memoized_fks(lambda fk: fk._set_remote_table(table))

        if self.identity and (
            isinstance(self.default, Sequence)
            or isinstance(self.onupdate, Sequence)
        ):
            raise exc.ArgumentError(
                "An column cannot specify both Identity and Sequence."
            )

    def _setup_on_memoized_fks(self, fn: Callable[..., Any]) -> None:
        fk_keys = [
            ((self.table.key, self.key), False),
            ((self.table.key, self.name), True),
        ]
        for fk_key, link_to_name in fk_keys:
            if fk_key in self.table.metadata._fk_memos:
                for fk in self.table.metadata._fk_memos[fk_key]:
                    if fk.link_to_name is link_to_name:
                        fn(fk)

    def _on_table_attach(self, fn: Callable[..., Any]) -> None:
        if self.table is not None:
            fn(self, self.table)
        else:
            event.listen(self, "after_parent_attach", fn)

    @util.deprecated(
        "1.4",
        "The :meth:`_schema.Column.copy` method is deprecated "
        "and will be removed in a future release.",
    )
    def copy(self, **kw: Any) -> Column[Any]:
        return self._copy(**kw)

    def _copy(self, **kw: Any) -> Column[Any]:
        """Create a copy of this ``Column``, uninitialized.

        This is used in :meth:`_schema.Table.to_metadata` and by the ORM.

        """

        # Constraint objects plus non-constraint-bound ForeignKey objects
        args: List[SchemaItem] = [
            c._copy(**kw) for c in self.constraints if not c._type_bound
        ] + [c._copy(**kw) for c in self.foreign_keys if not c.constraint]

        # ticket #5276
        column_kwargs = {}
        for dialect_name in self.dialect_options:
            dialect_options = self.dialect_options[dialect_name]._non_defaults
            for (
                dialect_option_key,
                dialect_option_value,
            ) in dialect_options.items():
                column_kwargs[dialect_name + "_" + dialect_option_key] = (
                    dialect_option_value
                )

        server_default = self.server_default
        server_onupdate = self.server_onupdate
        if isinstance(server_default, (Computed, Identity)):
            # TODO: likely should be copied in all cases
            # TODO: if a Sequence, we would need to transfer the Sequence
            # .metadata as well
            args.append(server_default._copy(**kw))
            server_default = server_onupdate = None

        type_ = self.type
        if isinstance(type_, SchemaEventTarget):
            type_ = type_.copy(**kw)

        # TODO: DefaultGenerator is not copied here!  it's just used again
        # with _set_parent() pointing to the old column.  see the new
        # use of _copy() in the new _merge() method

        c = self._constructor(
            name=self.name,
            type_=type_,
            key=self.key,
            primary_key=self.primary_key,
            unique=self.unique,
            system=self.system,
            # quote=self.quote,  # disabled 2013-08-27 (commit 031ef080)
            index=self.index,
            autoincrement=self.autoincrement,
            default=self.default,
            server_default=server_default,
            onupdate=self.onupdate,
            server_onupdate=server_onupdate,
            doc=self.doc,
            comment=self.comment,
            _omit_from_statements=self._omit_from_statements,
            insert_sentinel=self._insert_sentinel,
            *args,
            **column_kwargs,
        )

        # copy the state of "nullable" exactly, to accommodate for
        # ORM flipping the .nullable flag directly
        c.nullable = self.nullable
        c._user_defined_nullable = self._user_defined_nullable

        return self._schema_item_copy(c)

    def _merge(self, other: Column[Any]) -> None:
        """merge the elements of another column into this one.

        this is used by ORM pep-593 merge and will likely need a lot
        of fixes.


        """

        if self.primary_key:
            other.primary_key = True

        if self.autoincrement != "auto" and other.autoincrement == "auto":
            other.autoincrement = self.autoincrement

        if self.system:
            other.system = self.system

        if self.info:
            other.info.update(self.info)

        type_ = self.type
        if not type_._isnull and other.type._isnull:
            if isinstance(type_, SchemaEventTarget):
                type_ = type_.copy()

            other.type = type_

            if isinstance(type_, SchemaEventTarget):
                type_._set_parent_with_dispatch(other)

            for impl in type_._variant_mapping.values():
                if isinstance(impl, SchemaEventTarget):
                    impl._set_parent_with_dispatch(other)

        if (
            self._user_defined_nullable is not NULL_UNSPECIFIED
            and other._user_defined_nullable is NULL_UNSPECIFIED
        ):
            other.nullable = self.nullable
            other._user_defined_nullable = self._user_defined_nullable

        if self.default is not None and other.default is None:
            new_default = self.default._copy()
            new_default._set_parent(other)

        if self.server_default and other.server_default is None:
            new_server_default = self.server_default
            if isinstance(new_server_default, FetchedValue):
                new_server_default = new_server_default._copy()
                new_server_default._set_parent(other)
            else:
                other.server_default = new_server_default

        if self.server_onupdate and other.server_onupdate is None:
            new_server_onupdate = self.server_onupdate
            new_server_onupdate = new_server_onupdate._copy()
            new_server_onupdate._set_parent(other)

        if self.onupdate and other.onupdate is None:
            new_onupdate = self.onupdate._copy()
            new_onupdate._set_parent(other)

        if self.index in (True, False) and other.index is None:
            other.index = self.index

        if self.unique in (True, False) and other.unique is None:
            other.unique = self.unique

        if self.doc and other.doc is None:
            other.doc = self.doc

        if self.comment and other.comment is None:
            other.comment = self.comment

        for const in self.constraints:
            if not const._type_bound:
                new_const = const._copy()
                new_const._set_parent(other)

        for fk in self.foreign_keys:
            if not fk.constraint:
                new_fk = fk._copy()
                new_fk._set_parent(other)

    def _make_proxy(
        self,
        selectable: FromClause,
        primary_key: ColumnSet,
        foreign_keys: Set[KeyedColumnElement[Any]],
        name: Optional[str] = None,
        key: Optional[str] = None,
        name_is_truncatable: bool = False,
        compound_select_cols: Optional[
            _typing_Sequence[ColumnElement[Any]]
        ] = None,
        **kw: Any,
    ) -> Tuple[str, ColumnClause[_T]]:
        """Create a *proxy* for this column.

        This is a copy of this ``Column`` referenced by a different parent
        (such as an alias or select statement).  The column should
        be used only in select scenarios, as its full DDL/default
        information is not transferred.

        """

        fk = [
            ForeignKey(
                col if col is not None else f._colspec,
                _unresolvable=col is None,
                _constraint=f.constraint,
            )
            for f, col in [
                (fk, fk._resolve_column(raiseerr=False))
                for fk in self.foreign_keys
            ]
        ]

        if name is None and self.name is None:
            raise exc.InvalidRequestError(
                "Cannot initialize a sub-selectable"
                " with this Column object until its 'name' has "
                "been assigned."
            )
        try:
            c = self._constructor(
                (
                    coercions.expect(
                        roles.TruncatedLabelRole, name if name else self.name
                    )
                    if name_is_truncatable
                    else (name or self.name)
                ),
                self.type,
                # this may actually be ._proxy_key when the key is incoming
                key=key if key else name if name else self.key,
                primary_key=self.primary_key,
                nullable=self.nullable,
                _proxies=(
                    list(compound_select_cols)
                    if compound_select_cols
                    else [self]
                ),
                *fk,
            )
        except TypeError as err:
            raise TypeError(
                "Could not create a copy of this %r object.  "
                "Ensure the class includes a _constructor() "
                "attribute or method which accepts the "
                "standard Column constructor arguments, or "
                "references the Column class itself." % self.__class__
            ) from err

        c.table = selectable
        c._propagate_attrs = selectable._propagate_attrs
        if selectable._is_clone_of is not None:
            c._is_clone_of = selectable._is_clone_of.columns.get(c.key)

        if self.primary_key:
            primary_key.add(c)

        if fk:
            foreign_keys.update(fk)  # type: ignore

        return c.key, c


def insert_sentinel(
    name: Optional[str] = None,
    type_: Optional[_TypeEngineArgument[_T]] = None,
    *,
    default: Optional[Any] = None,
    omit_from_statements: bool = True,
) -> Column[Any]:
    """Provides a surrogate :class:`_schema.Column` that will act as a
    dedicated insert :term:`sentinel` column, allowing efficient bulk
    inserts with deterministic RETURNING sorting for tables that
    don't otherwise have qualifying primary key configurations.

    Adding this column to a :class:`.Table` object requires that a
    corresponding database table actually has this column present, so if adding
    it to an existing model, existing database tables would need to be migrated
    (e.g. using ALTER TABLE or similar) to include this column.

    For background on how this object is used, see the section
    :ref:`engine_insertmanyvalues_sentinel_columns` as part of the
    section :ref:`engine_insertmanyvalues`.

    The :class:`_schema.Column` returned will be a nullable integer column by
    default and make use of a sentinel-specific default generator used only in
    "insertmanyvalues" operations.

    .. seealso::

        :func:`_orm.orm_insert_sentinel`

        :paramref:`_schema.Column.insert_sentinel`

        :ref:`engine_insertmanyvalues`

        :ref:`engine_insertmanyvalues_sentinel_columns`


    .. versionadded:: 2.0.10

    """
    return Column(
        name=name,
        type_=type_api.INTEGERTYPE if type_ is None else type_,
        default=(
            default if default is not None else _InsertSentinelColumnDefault()
        ),
        _omit_from_statements=omit_from_statements,
        insert_sentinel=True,
    )


class ForeignKey(DialectKWArgs, SchemaItem):
    """Defines a dependency between two columns.

    ``ForeignKey`` is specified as an argument to a :class:`_schema.Column`
    object,
    e.g.::

        t = Table(
            "remote_table",
            metadata,
            Column("remote_id", ForeignKey("main_table.id")),
        )

    Note that ``ForeignKey`` is only a marker object that defines
    a dependency between two columns.   The actual constraint
    is in all cases represented by the :class:`_schema.ForeignKeyConstraint`
    object.   This object will be generated automatically when
    a ``ForeignKey`` is associated with a :class:`_schema.Column` which
    in turn is associated with a :class:`_schema.Table`.   Conversely,
    when :class:`_schema.ForeignKeyConstraint` is applied to a
    :class:`_schema.Table`,
    ``ForeignKey`` markers are automatically generated to be
    present on each associated :class:`_schema.Column`, which are also
    associated with the constraint object.

    Note that you cannot define a "composite" foreign key constraint,
    that is a constraint between a grouping of multiple parent/child
    columns, using ``ForeignKey`` objects.   To define this grouping,
    the :class:`_schema.ForeignKeyConstraint` object must be used, and applied
    to the :class:`_schema.Table`.   The associated ``ForeignKey`` objects
    are created automatically.

    The ``ForeignKey`` objects associated with an individual
    :class:`_schema.Column`
    object are available in the `foreign_keys` collection
    of that column.

    Further examples of foreign key configuration are in
    :ref:`metadata_foreignkeys`.

    """

    __visit_name__ = "foreign_key"

    parent: Column[Any]

    _table_column: Optional[Column[Any]]

    def __init__(
        self,
        column: _DDLColumnArgument,
        _constraint: Optional[ForeignKeyConstraint] = None,
        use_alter: bool = False,
        name: _ConstraintNameArgument = None,
        onupdate: Optional[str] = None,
        ondelete: Optional[str] = None,
        deferrable: Optional[bool] = None,
        initially: Optional[str] = None,
        link_to_name: bool = False,
        match: Optional[str] = None,
        info: Optional[_InfoType] = None,
        comment: Optional[str] = None,
        _unresolvable: bool = False,
        **dialect_kw: Any,
    ):
        r"""
        Construct a column-level FOREIGN KEY.

        The :class:`_schema.ForeignKey` object when constructed generates a
        :class:`_schema.ForeignKeyConstraint`
        which is associated with the parent
        :class:`_schema.Table` object's collection of constraints.

        :param column: A single target column for the key relationship. A
            :class:`_schema.Column` object or a column name as a string:
            ``tablename.columnkey`` or ``schema.tablename.columnkey``.
            ``columnkey`` is the ``key`` which has been assigned to the column
            (defaults to the column name itself), unless ``link_to_name`` is
            ``True`` in which case the rendered name of the column is used.

        :param name: Optional string. An in-database name for the key if
            `constraint` is not provided.

        :param onupdate: Optional string. If set, emit ON UPDATE <value> when
            issuing DDL for this constraint. Typical values include CASCADE,
            DELETE and RESTRICT.

            .. seealso::

                :ref:`on_update_on_delete`

        :param ondelete: Optional string. If set, emit ON DELETE <value> when
            issuing DDL for this constraint. Typical values include CASCADE,
            SET NULL and RESTRICT.  Some dialects may allow for additional
            syntaxes.

            .. seealso::

                :ref:`on_update_on_delete`

        :param deferrable: Optional bool. If set, emit DEFERRABLE or NOT
            DEFERRABLE when issuing DDL for this constraint.

        :param initially: Optional string. If set, emit INITIALLY <value> when
            issuing DDL for this constraint.

        :param link_to_name: if True, the string name given in ``column`` is
            the rendered name of the referenced column, not its locally
            assigned ``key``.

        :param use_alter: passed to the underlying
            :class:`_schema.ForeignKeyConstraint`
            to indicate the constraint should
            be generated/dropped externally from the CREATE TABLE/ DROP TABLE
            statement.  See :paramref:`_schema.ForeignKeyConstraint.use_alter`
            for further description.

            .. seealso::

                :paramref:`_schema.ForeignKeyConstraint.use_alter`

                :ref:`use_alter`

        :param match: Optional string. If set, emit MATCH <value> when issuing
            DDL for this constraint. Typical values include SIMPLE, PARTIAL
            and FULL.

        :param info: Optional data dictionary which will be populated into the
            :attr:`.SchemaItem.info` attribute of this object.

        :param comment: Optional string that will render an SQL comment on
          foreign key constraint creation.

            .. versionadded:: 2.0

        :param \**dialect_kw:  Additional keyword arguments are dialect
            specific, and passed in the form ``<dialectname>_<argname>``.  The
            arguments are ultimately handled by a corresponding
            :class:`_schema.ForeignKeyConstraint`.
            See the documentation regarding
            an individual dialect at :ref:`dialect_toplevel` for detail on
            documented arguments.

        """

        self._colspec = coercions.expect(roles.DDLReferredColumnRole, column)
        self._unresolvable = _unresolvable

        if isinstance(self._colspec, str):
            self._table_column = None
        else:
            self._table_column = self._colspec

            if not isinstance(
                self._table_column.table, (type(None), TableClause)
            ):
                raise exc.ArgumentError(
                    "ForeignKey received Column not bound "
                    "to a Table, got: %r" % self._table_column.table
                )

        # the linked ForeignKeyConstraint.
        # ForeignKey will create this when parent Column
        # is attached to a Table, *or* ForeignKeyConstraint
        # object passes itself in when creating ForeignKey
        # markers.
        self.constraint = _constraint

        # .parent is not Optional under normal use
        self.parent = None  # type: ignore

        self.use_alter = use_alter
        self.name = name
        self.onupdate = onupdate
        self.ondelete = ondelete
        self.deferrable = deferrable
        self.initially = initially
        self.link_to_name = link_to_name
        self.match = match
        self.comment = comment
        if info:
            self.info = info
        self._unvalidated_dialect_kw = dialect_kw

    def __repr__(self) -> str:
        return "ForeignKey(%r)" % self._get_colspec()

    @util.deprecated(
        "1.4",
        "The :meth:`_schema.ForeignKey.copy` method is deprecated "
        "and will be removed in a future release.",
    )
    def copy(self, *, schema: Optional[str] = None, **kw: Any) -> ForeignKey:
        return self._copy(schema=schema, **kw)

    def _copy(self, *, schema: Optional[str] = None, **kw: Any) -> ForeignKey:
        """Produce a copy of this :class:`_schema.ForeignKey` object.

        The new :class:`_schema.ForeignKey` will not be bound
        to any :class:`_schema.Column`.

        This method is usually used by the internal
        copy procedures of :class:`_schema.Column`, :class:`_schema.Table`,
        and :class:`_schema.MetaData`.

        :param schema: The returned :class:`_schema.ForeignKey` will
          reference the original table and column name, qualified
          by the given string schema name.

        """
        fk = ForeignKey(
            self._get_colspec(schema=schema),
            use_alter=self.use_alter,
            name=self.name,
            onupdate=self.onupdate,
            ondelete=self.ondelete,
            deferrable=self.deferrable,
            initially=self.initially,
            link_to_name=self.link_to_name,
            match=self.match,
            comment=self.comment,
            **self._unvalidated_dialect_kw,
        )
        return self._schema_item_copy(fk)

    def _get_colspec(
        self,
        schema: Optional[
            Union[
                str,
                Literal[SchemaConst.RETAIN_SCHEMA, SchemaConst.BLANK_SCHEMA],
            ]
        ] = None,
        table_name: Optional[str] = None,
        _is_copy: bool = False,
    ) -> str:
        """Return a string based 'column specification' for this
        :class:`_schema.ForeignKey`.

        This is usually the equivalent of the string-based "tablename.colname"
        argument first passed to the object's constructor.

        """
        if schema not in (None, RETAIN_SCHEMA):
            _schema, tname, colname = self._column_tokens
            if table_name is not None:
                tname = table_name
            if schema is BLANK_SCHEMA:
                return "%s.%s" % (tname, colname)
            else:
                return "%s.%s.%s" % (schema, tname, colname)
        elif table_name:
            schema, tname, colname = self._column_tokens
            if schema:
                return "%s.%s.%s" % (schema, table_name, colname)
            else:
                return "%s.%s" % (table_name, colname)
        elif self._table_column is not None:
            if self._table_column.table is None:
                if _is_copy:
                    raise exc.InvalidRequestError(
                        f"Can't copy ForeignKey object which refers to "
                        f"non-table bound Column {self._table_column!r}"
                    )
                else:
                    return self._table_column.key
            return "%s.%s" % (
                self._table_column.table.fullname,
                self._table_column.key,
            )
        else:
            assert isinstance(self._colspec, str)
            return self._colspec

    @property
    def _referred_schema(self) -> Optional[str]:
        return self._column_tokens[0]

    def _table_key(self) -> Any:
        if self._table_column is not None:
            if self._table_column.table is None:
                return None
            else:
                return self._table_column.table.key
        else:
            schema, tname, colname = self._column_tokens
            return _get_table_key(tname, schema)

    target_fullname = property(_get_colspec)

    def references(self, table: Table) -> bool:
        """Return True if the given :class:`_schema.Table`
        is referenced by this
        :class:`_schema.ForeignKey`."""

        return table.corresponding_column(self.column) is not None

    def get_referent(self, table: FromClause) -> Optional[Column[Any]]:
        """Return the :class:`_schema.Column` in the given
        :class:`_schema.Table` (or any :class:`.FromClause`)
        referenced by this :class:`_schema.ForeignKey`.

        Returns None if this :class:`_schema.ForeignKey`
        does not reference the given
        :class:`_schema.Table`.

        """
        # our column is a Column, and any subquery etc. proxying us
        # would be doing so via another Column, so that's what would
        # be returned here
        return table.columns.corresponding_column(self.column)  # type: ignore

    @util.memoized_property
    def _column_tokens(self) -> Tuple[Optional[str], str, Optional[str]]:
        """parse a string-based _colspec into its component parts."""

        m = self._get_colspec().split(".")
        if m is None:
            raise exc.ArgumentError(
                f"Invalid foreign key column specification: {self._colspec}"
            )
        if len(m) == 1:
            tname = m.pop()
            colname = None
        else:
            colname = m.pop()
            tname = m.pop()

        # A FK between column 'bar' and table 'foo' can be
        # specified as 'foo', 'foo.bar', 'dbo.foo.bar',
        # 'otherdb.dbo.foo.bar'. Once we have the column name and
        # the table name, treat everything else as the schema
        # name. Some databases (e.g. Sybase) support
        # inter-database foreign keys. See tickets#1341 and --
        # indirectly related -- Ticket #594. This assumes that '.'
        # will never appear *within* any component of the FK.

        if len(m) > 0:
            schema = ".".join(m)
        else:
            schema = None
        return schema, tname, colname

    def _resolve_col_tokens(self) -> Tuple[Table, str, Optional[str]]:
        if self.parent is None:
            raise exc.InvalidRequestError(
                "this ForeignKey object does not yet have a "
                "parent Column associated with it."
            )

        elif self.parent.table is None:
            raise exc.InvalidRequestError(
                "this ForeignKey's parent column is not yet associated "
                "with a Table."
            )

        parenttable = self.parent.table

        if self._unresolvable:
            schema, tname, colname = self._column_tokens
            tablekey = _get_table_key(tname, schema)
            return parenttable, tablekey, colname

        # assertion
        # basically Column._make_proxy() sends the actual
        # target Column to the ForeignKey object, so the
        # string resolution here is never called.
        for c in self.parent.base_columns:
            if isinstance(c, Column):
                assert c.table is parenttable
                break
        else:
            assert False
        ######################

        schema, tname, colname = self._column_tokens

        if schema is None and parenttable.metadata.schema is not None:
            schema = parenttable.metadata.schema

        tablekey = _get_table_key(tname, schema)
        return parenttable, tablekey, colname

    def _link_to_col_by_colstring(
        self, parenttable: Table, table: Table, colname: Optional[str]
    ) -> Column[Any]:
        _column = None
        if colname is None:
            # colname is None in the case that ForeignKey argument
            # was specified as table name only, in which case we
            # match the column name to the same column on the
            # parent.
            # this use case wasn't working in later 1.x series
            # as it had no test coverage; fixed in 2.0
            parent = self.parent
            assert parent is not None
            key = parent.key
            _column = table.c.get(key, None)
        elif self.link_to_name:
            key = colname
            for c in table.c:
                if c.name == colname:
                    _column = c
        else:
            key = colname
            _column = table.c.get(colname, None)

        if _column is None:
            raise exc.NoReferencedColumnError(
                "Could not initialize target column "
                f"for ForeignKey '{self._colspec}' "
                f"on table '{parenttable.name}': "
                f"table '{table.name}' has no column named '{key}'",
                table.name,
                key,
            )

        return _column

    def _set_target_column(self, column: Column[Any]) -> None:
        assert self.parent is not None

        # propagate TypeEngine to parent if it didn't have one
        if self.parent.type._isnull:
            self.parent.type = column.type

        # super-edgy case, if other FKs point to our column,
        # they'd get the type propagated out also.

        def set_type(fk: ForeignKey) -> None:
            if fk.parent.type._isnull:
                fk.parent.type = column.type

        self.parent._setup_on_memoized_fks(set_type)

        self.column = column  # type: ignore

    @util.ro_memoized_property
    def column(self) -> Column[Any]:
        """Return the target :class:`_schema.Column` referenced by this
        :class:`_schema.ForeignKey`.

        If no target column has been established, an exception
        is raised.

        """

        return self._resolve_column()

    @overload
    def _resolve_column(
        self, *, raiseerr: Literal[True] = ...
    ) -> Column[Any]: ...

    @overload
    def _resolve_column(
        self, *, raiseerr: bool = ...
    ) -> Optional[Column[Any]]: ...

    def _resolve_column(
        self, *, raiseerr: bool = True
    ) -> Optional[Column[Any]]:
        _column: Column[Any]

        if isinstance(self._colspec, str):
            parenttable, tablekey, colname = self._resolve_col_tokens()

            if self._unresolvable or tablekey not in parenttable.metadata:
                if not raiseerr:
                    return None
                raise exc.NoReferencedTableError(
                    f"Foreign key associated with column "
                    f"'{self.parent}' could not find "
                    f"table '{tablekey}' with which to generate a "
                    f"foreign key to target column '{colname}'",
                    tablekey,
                )
            elif parenttable.key not in parenttable.metadata:
                if not raiseerr:
                    return None
                raise exc.InvalidRequestError(
                    f"Table {parenttable} is no longer associated with its "
                    "parent MetaData"
                )
            else:
                table = parenttable.metadata.tables[tablekey]
                return self._link_to_col_by_colstring(
                    parenttable, table, colname
                )

        elif hasattr(self._colspec, "__clause_element__"):
            _column = self._colspec.__clause_element__()
            return _column
        else:
            _column = self._colspec
            return _column

    def _set_parent(self, parent: SchemaEventTarget, **kw: Any) -> None:
        assert isinstance(parent, Column)

        if self.parent is not None and self.parent is not parent:
            raise exc.InvalidRequestError(
                "This ForeignKey already has a parent !"
            )
        self.parent = parent
        self.parent.foreign_keys.add(self)
        self.parent._on_table_attach(self._set_table)

    def _set_remote_table(self, table: Table) -> None:
        parenttable, _, colname = self._resolve_col_tokens()
        _column = self._link_to_col_by_colstring(parenttable, table, colname)
        self._set_target_column(_column)
        assert self.constraint is not None
        self.constraint._validate_dest_table(table)

    def _remove_from_metadata(self, metadata: MetaData) -> None:
        parenttable, table_key, colname = self._resolve_col_tokens()
        fk_key = (table_key, colname)

        if self in metadata._fk_memos[fk_key]:
            # TODO: no test coverage for self not in memos
            metadata._fk_memos[fk_key].remove(self)

    def _set_table(self, column: Column[Any], table: Table) -> None:
        # standalone ForeignKey - create ForeignKeyConstraint
        # on the hosting Table when attached to the Table.
        assert isinstance(table, Table)
        if self.constraint is None:
            self.constraint = ForeignKeyConstraint(
                [],
                [],
                use_alter=self.use_alter,
                name=self.name,
                onupdate=self.onupdate,
                ondelete=self.ondelete,
                deferrable=self.deferrable,
                initially=self.initially,
                match=self.match,
                comment=self.comment,
                **self._unvalidated_dialect_kw,
            )
            self.constraint._append_element(column, self)
            self.constraint._set_parent_with_dispatch(table)
        table.foreign_keys.add(self)
        # set up remote ".column" attribute, or a note to pick it
        # up when the other Table/Column shows up
        if isinstance(self._colspec, str):
            parenttable, table_key, colname = self._resolve_col_tokens()
            fk_key = (table_key, colname)
            if table_key in parenttable.metadata.tables:
                table = parenttable.metadata.tables[table_key]
                try:
                    _column = self._link_to_col_by_colstring(
                        parenttable, table, colname
                    )
                except exc.NoReferencedColumnError:
                    # this is OK, we'll try later
                    pass
                else:
                    self._set_target_column(_column)

            parenttable.metadata._fk_memos[fk_key].append(self)
        elif hasattr(self._colspec, "__clause_element__"):
            _column = self._colspec.__clause_element__()
            self._set_target_column(_column)
        else:
            _column = self._colspec
            self._set_target_column(_column)


if TYPE_CHECKING:

    def default_is_sequence(
        obj: Optional[DefaultGenerator],
    ) -> TypeGuard[Sequence]: ...

    def default_is_clause_element(
        obj: Optional[DefaultGenerator],
    ) -> TypeGuard[ColumnElementColumnDefault]: ...

    def default_is_scalar(
        obj: Optional[DefaultGenerator],
    ) -> TypeGuard[ScalarElementColumnDefault]: ...

else:
    default_is_sequence = operator.attrgetter("is_sequence")

    default_is_clause_element = operator.attrgetter("is_clause_element")

    default_is_scalar = operator.attrgetter("is_scalar")


class DefaultGenerator(Executable, SchemaItem):
    """Base class for column *default* values.

    This object is only present on column.default or column.onupdate.
    It's not valid as a server default.

    """

    __visit_name__ = "default_generator"

    _is_default_generator = True
    is_sequence = False
    is_identity = False
    is_server_default = False
    is_clause_element = False
    is_callable = False
    is_scalar = False
    has_arg = False
    is_sentinel = False
    column: Optional[Column[Any]]

    def __init__(self, for_update: bool = False) -> None:
        self.for_update = for_update

    def _set_parent(self, parent: SchemaEventTarget, **kw: Any) -> None:
        if TYPE_CHECKING:
            assert isinstance(parent, Column)
        self.column = parent
        if self.for_update:
            self.column.onupdate = self
        else:
            self.column.default = self

    def _copy(self) -> DefaultGenerator:
        raise NotImplementedError()

    def _execute_on_connection(
        self,
        connection: Connection,
        distilled_params: _CoreMultiExecuteParams,
        execution_options: CoreExecuteOptionsParameter,
    ) -> Any:
        util.warn_deprecated(
            "Using the .execute() method to invoke a "
            "DefaultGenerator object is deprecated; please use "
            "the .scalar() method.",
            "2.0",
        )
        return self._execute_on_scalar(
            connection, distilled_params, execution_options
        )

    def _execute_on_scalar(
        self,
        connection: Connection,
        distilled_params: _CoreMultiExecuteParams,
        execution_options: CoreExecuteOptionsParameter,
    ) -> Any:
        return connection._execute_default(
            self, distilled_params, execution_options
        )


class ColumnDefault(DefaultGenerator, ABC):
    """A plain default value on a column.

    This could correspond to a constant, a callable function,
    or a SQL clause.

    :class:`.ColumnDefault` is generated automatically
    whenever the ``default``, ``onupdate`` arguments of
    :class:`_schema.Column` are used.  A :class:`.ColumnDefault`
    can be passed positionally as well.

    For example, the following::

        Column("foo", Integer, default=50)

    Is equivalent to::

        Column("foo", Integer, ColumnDefault(50))

    """

    arg: Any

    @overload
    def __new__(
        cls, arg: Callable[..., Any], for_update: bool = ...
    ) -> CallableColumnDefault: ...

    @overload
    def __new__(
        cls, arg: ColumnElement[Any], for_update: bool = ...
    ) -> ColumnElementColumnDefault: ...

    # if I return ScalarElementColumnDefault here, which is what's actually
    # returned, mypy complains that
    # overloads overlap w/ incompatible return types.
    @overload
    def __new__(cls, arg: object, for_update: bool = ...) -> ColumnDefault: ...

    def __new__(
        cls, arg: Any = None, for_update: bool = False
    ) -> ColumnDefault:
        """Construct a new :class:`.ColumnDefault`.


        :param arg: argument representing the default value.
         May be one of the following:

         * a plain non-callable Python value, such as a
           string, integer, boolean, or other simple type.
           The default value will be used as is each time.
         * a SQL expression, that is one which derives from
           :class:`_expression.ColumnElement`.  The SQL expression will
           be rendered into the INSERT or UPDATE statement,
           or in the case of a primary key column when
           RETURNING is not used may be
           pre-executed before an INSERT within a SELECT.
         * A Python callable.  The function will be invoked for each
           new row subject to an INSERT or UPDATE.
           The callable must accept exactly
           zero or one positional arguments.  The one-argument form
           will receive an instance of the :class:`.ExecutionContext`,
           which provides contextual information as to the current
           :class:`_engine.Connection` in use as well as the current
           statement and parameters.

        """

        if isinstance(arg, FetchedValue):
            raise exc.ArgumentError(
                "ColumnDefault may not be a server-side default type."
            )
        elif callable(arg):
            cls = CallableColumnDefault
        elif isinstance(arg, ClauseElement):
            cls = ColumnElementColumnDefault
        elif arg is not None:
            cls = ScalarElementColumnDefault

        return object.__new__(cls)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.arg!r})"


class ScalarElementColumnDefault(ColumnDefault):
    """default generator for a fixed scalar Python value

    .. versionadded: 2.0

    """

    is_scalar = True
    has_arg = True

    def __init__(self, arg: Any, for_update: bool = False) -> None:
        self.for_update = for_update
        self.arg = arg

    def _copy(self) -> ScalarElementColumnDefault:
        return ScalarElementColumnDefault(
            arg=self.arg, for_update=self.for_update
        )


class _InsertSentinelColumnDefault(ColumnDefault):
    """Default generator that's specific to the use of a "sentinel" column
    when using the insertmanyvalues feature.

    This default is used as part of the :func:`_schema.insert_sentinel`
    construct.

    """

    is_sentinel = True
    for_update = False
    arg = None

    def __new__(cls) -> _InsertSentinelColumnDefault:
        return object.__new__(cls)

    def __init__(self) -> None:
        pass

    def _set_parent(self, parent: SchemaEventTarget, **kw: Any) -> None:
        col = cast("Column[Any]", parent)
        if not col._insert_sentinel:
            raise exc.ArgumentError(
                "The _InsertSentinelColumnDefault may only be applied to a "
                "Column marked as insert_sentinel=True"
            )
        elif not col.nullable:
            raise exc.ArgumentError(
                "The _InsertSentinelColumnDefault may only be applied to a "
                "Column that is nullable"
            )

        super()._set_parent(parent, **kw)

    def _copy(self) -> _InsertSentinelColumnDefault:
        return _InsertSentinelColumnDefault()


_SQLExprDefault = Union["ColumnElement[Any]", "TextClause"]


class ColumnElementColumnDefault(ColumnDefault):
    """default generator for a SQL expression

    .. versionadded:: 2.0

    """

    is_clause_element = True
    has_arg = True
    arg: _SQLExprDefault

    def __init__(
        self,
        arg: _SQLExprDefault,
        for_update: bool = False,
    ) -> None:
        self.for_update = for_update
        self.arg = arg

    def _copy(self) -> ColumnElementColumnDefault:
        return ColumnElementColumnDefault(
            arg=self.arg, for_update=self.for_update
        )

    @util.memoized_property
    @util.preload_module("sqlalchemy.sql.sqltypes")
    def _arg_is_typed(self) -> bool:
        sqltypes = util.preloaded.sql_sqltypes

        return not isinstance(self.arg.type, sqltypes.NullType)


class _CallableColumnDefaultProtocol(Protocol):
    def __call__(self, context: ExecutionContext) -> Any: ...


class CallableColumnDefault(ColumnDefault):
    """default generator for a callable Python function

    .. versionadded:: 2.0

    """

    is_callable = True
    arg: _CallableColumnDefaultProtocol
    has_arg = True

    def __init__(
        self,
        arg: Union[_CallableColumnDefaultProtocol, Callable[[], Any]],
        for_update: bool = False,
    ) -> None:
        self.for_update = for_update
        self.arg = self._maybe_wrap_callable(arg)

    def _copy(self) -> CallableColumnDefault:
        return CallableColumnDefault(arg=self.arg, for_update=self.for_update)

    def _maybe_wrap_callable(
        self, fn: Union[_CallableColumnDefaultProtocol, Callable[[], Any]]
    ) -> _CallableColumnDefaultProtocol:
        """Wrap callables that don't accept a context.

        This is to allow easy compatibility with default callables
        that aren't specific to accepting of a context.

        """

        try:
            argspec = util.get_callable_argspec(fn, no_self=True)
        except TypeError:
            return util.wrap_callable(lambda ctx: fn(), fn)  # type: ignore

        defaulted = argspec[3] is not None and len(argspec[3]) or 0
        positionals = len(argspec[0]) - defaulted

        if positionals == 0:
            return util.wrap_callable(lambda ctx: fn(), fn)  # type: ignore

        elif positionals == 1:
            return fn  # type: ignore
        else:
            raise exc.ArgumentError(
                "ColumnDefault Python function takes zero or one "
                "positional arguments"
            )


class IdentityOptions:
    """Defines options for a named database sequence or an identity column.

    .. versionadded:: 1.3.18

    .. seealso::

        :class:`.Sequence`

    """

    def __init__(
        self,
        start: Optional[int] = None,
        increment: Optional[int] = None,
        minvalue: Optional[int] = None,
        maxvalue: Optional[int] = None,
        nominvalue: Optional[bool] = None,
        nomaxvalue: Optional[bool] = None,
        cycle: Optional[bool] = None,
        cache: Optional[int] = None,
        order: Optional[bool] = None,
    ) -> None:
        """Construct a :class:`.IdentityOptions` object.

        See the :class:`.Sequence` documentation for a complete description
        of the parameters.

        :param start: the starting index of the sequence.
        :param increment: the increment value of the sequence.
        :param minvalue: the minimum value of the sequence.
        :param maxvalue: the maximum value of the sequence.
        :param nominvalue: no minimum value of the sequence.
        :param nomaxvalue: no maximum value of the sequence.
        :param cycle: allows the sequence to wrap around when the maxvalue
         or minvalue has been reached.
        :param cache: optional integer value; number of future values in the
         sequence which are calculated in advance.
        :param order: optional boolean value; if ``True``, renders the
         ORDER keyword.

        """
        self.start = start
        self.increment = increment
        self.minvalue = minvalue
        self.maxvalue = maxvalue
        self.nominvalue = nominvalue
        self.nomaxvalue = nomaxvalue
        self.cycle = cycle
        self.cache = cache
        self.order = order

    @property
    def _increment_is_negative(self) -> bool:
        return self.increment is not None and self.increment < 0


class Sequence(HasSchemaAttr, IdentityOptions, DefaultGenerator):
    """Represents a named database sequence.

    The :class:`.Sequence` object represents the name and configurational
    parameters of a database sequence.   It also represents
    a construct that can be "executed" by a SQLAlchemy :class:`_engine.Engine`
    or :class:`_engine.Connection`,
    rendering the appropriate "next value" function
    for the target database and returning a result.

    The :class:`.Sequence` is typically associated with a primary key column::

        some_table = Table(
            "some_table",
            metadata,
            Column(
                "id",
                Integer,
                Sequence("some_table_seq", start=1),
                primary_key=True,
            ),
        )

    When CREATE TABLE is emitted for the above :class:`_schema.Table`, if the
    target platform supports sequences, a CREATE SEQUENCE statement will
    be emitted as well.   For platforms that don't support sequences,
    the :class:`.Sequence` construct is ignored.

    .. seealso::

        :ref:`defaults_sequences`

        :class:`.CreateSequence`

        :class:`.DropSequence`

    """

    __visit_name__ = "sequence"

    is_sequence = True

    column: Optional[Column[Any]]
    data_type: Optional[TypeEngine[int]]

    def __init__(
        self,
        name: str,
        start: Optional[int] = None,
        increment: Optional[int] = None,
        minvalue: Optional[int] = None,
        maxvalue: Optional[int] = None,
        nominvalue: Optional[bool] = None,
        nomaxvalue: Optional[bool] = None,
        cycle: Optional[bool] = None,
        schema: Optional[Union[str, Literal[SchemaConst.BLANK_SCHEMA]]] = None,
        cache: Optional[int] = None,
        order: Optional[bool] = None,
        data_type: Optional[_TypeEngineArgument[int]] = None,
        optional: bool = False,
        quote: Optional[bool] = None,
        metadata: Optional[MetaData] = None,
        quote_schema: Optional[bool] = None,
        for_update: bool = False,
    ) -> None:
        """Construct a :class:`.Sequence` object.

        :param name: the name of the sequence.

        :param start: the starting index of the sequence.  This value is
         used when the CREATE SEQUENCE command is emitted to the database
         as the value of the "START WITH" clause. If ``None``, the
         clause is omitted, which on most platforms indicates a starting
         value of 1.

         .. versionchanged:: 2.0 The :paramref:`.Sequence.start` parameter
            is required in order to have DDL emit "START WITH".  This is a
            reversal of a change made in version 1.4 which would implicitly
            render "START WITH 1" if the :paramref:`.Sequence.start` were
            not included.  See :ref:`change_7211` for more detail.

        :param increment: the increment value of the sequence.  This
         value is used when the CREATE SEQUENCE command is emitted to
         the database as the value of the "INCREMENT BY" clause.  If ``None``,
         the clause is omitted, which on most platforms indicates an
         increment of 1.
        :param minvalue: the minimum value of the sequence.  This
         value is used when the CREATE SEQUENCE command is emitted to
         the database as the value of the "MINVALUE" clause.  If ``None``,
         the clause is omitted, which on most platforms indicates a
         minvalue of 1 and -2^63-1 for ascending and descending sequences,
         respectively.

        :param maxvalue: the maximum value of the sequence.  This
         value is used when the CREATE SEQUENCE command is emitted to
         the database as the value of the "MAXVALUE" clause.  If ``None``,
         the clause is omitted, which on most platforms indicates a
         maxvalue of 2^63-1 and -1 for ascending and descending sequences,
         respectively.

        :param nominvalue: no minimum value of the sequence.  This
         value is used when the CREATE SEQUENCE command is emitted to
         the database as the value of the "NO MINVALUE" clause.  If ``None``,
         the clause is omitted, which on most platforms indicates a
         minvalue of 1 and -2^63-1 for ascending and descending sequences,
         respectively.

        :param nomaxvalue: no maximum value of the sequence.  This
         value is used when the CREATE SEQUENCE command is emitted to
         the database as the value of the "NO MAXVALUE" clause.  If ``None``,
         the clause is omitted, which on most platforms indicates a
         maxvalue of 2^63-1 and -1 for ascending and descending sequences,
         respectively.

        :param cycle: allows the sequence to wrap around when the maxvalue
         or minvalue has been reached by an ascending or descending sequence
         respectively.  This value is used when the CREATE SEQUENCE command
         is emitted to the database as the "CYCLE" clause.  If the limit is
         reached, the next number generated will be the minvalue or maxvalue,
         respectively.  If cycle=False (the default) any calls to nextval
         after the sequence has reached its maximum value will return an
         error.

        :param schema: optional schema name for the sequence, if located
         in a schema other than the default.  The rules for selecting the
         schema name when a :class:`_schema.MetaData`
         is also present are the same
         as that of :paramref:`_schema.Table.schema`.

        :param cache: optional integer value; number of future values in the
         sequence which are calculated in advance.  Renders the CACHE keyword
         understood by Oracle Database and PostgreSQL.

        :param order: optional boolean value; if ``True``, renders the
         ORDER keyword, understood by Oracle Database, indicating the sequence
         is definitively ordered.   May be necessary to provide deterministic
         ordering using Oracle RAC.

        :param data_type: The type to be returned by the sequence, for
         dialects that allow us to choose between INTEGER, BIGINT, etc.
         (e.g., mssql).

         .. versionadded:: 1.4.0

        :param optional: boolean value, when ``True``, indicates that this
         :class:`.Sequence` object only needs to be explicitly generated
         on backends that don't provide another way to generate primary
         key identifiers.  Currently, it essentially means, "don't create
         this sequence on the PostgreSQL backend, where the SERIAL keyword
         creates a sequence for us automatically".
        :param quote: boolean value, when ``True`` or ``False``, explicitly
         forces quoting of the :paramref:`_schema.Sequence.name` on or off.
         When left at its default of ``None``, normal quoting rules based
         on casing and reserved words take place.
        :param quote_schema: Set the quoting preferences for the ``schema``
         name.

        :param metadata: optional :class:`_schema.MetaData` object which this
         :class:`.Sequence` will be associated with.  A :class:`.Sequence`
         that is associated with a :class:`_schema.MetaData`
         gains the following
         capabilities:

         * The :class:`.Sequence` will inherit the
           :paramref:`_schema.MetaData.schema`
           parameter specified to the target :class:`_schema.MetaData`, which
           affects the production of CREATE / DROP DDL, if any.

         * The :meth:`.Sequence.create` and :meth:`.Sequence.drop` methods
           automatically use the engine bound to the :class:`_schema.MetaData`
           object, if any.

         * The :meth:`_schema.MetaData.create_all` and
           :meth:`_schema.MetaData.drop_all`
           methods will emit CREATE / DROP for this :class:`.Sequence`,
           even if the :class:`.Sequence` is not associated with any
           :class:`_schema.Table` / :class:`_schema.Column`
           that's a member of this
           :class:`_schema.MetaData`.

         The above behaviors can only occur if the :class:`.Sequence` is
         explicitly associated with the :class:`_schema.MetaData`
         via this parameter.

         .. seealso::

            :ref:`sequence_metadata` - full discussion of the
            :paramref:`.Sequence.metadata` parameter.

        :param for_update: Indicates this :class:`.Sequence`, when associated
         with a :class:`_schema.Column`,
         should be invoked for UPDATE statements
         on that column's table, rather than for INSERT statements, when
         no value is otherwise present for that column in the statement.

        """
        DefaultGenerator.__init__(self, for_update=for_update)
        IdentityOptions.__init__(
            self,
            start=start,
            increment=increment,
            minvalue=minvalue,
            maxvalue=maxvalue,
            nominvalue=nominvalue,
            nomaxvalue=nomaxvalue,
            cycle=cycle,
            cache=cache,
            order=order,
        )
        self.column = None
        self.name = quoted_name(name, quote)
        self.optional = optional
        if schema is BLANK_SCHEMA:
            self.schema = schema = None
        elif metadata is not None and schema is None and metadata.schema:
            self.schema = schema = metadata.schema
        else:
            self.schema = quoted_name.construct(schema, quote_schema)
        self.metadata = metadata
        self._key = _get_table_key(name, schema)
        if metadata:
            self._set_metadata(metadata)
        if data_type is not None:
            self.data_type = to_instance(data_type)
        else:
            self.data_type = None

    @util.preload_module("sqlalchemy.sql.functions")
    def next_value(self) -> Function[int]:
        """Return a :class:`.next_value` function element
        which will render the appropriate increment function
        for this :class:`.Sequence` within any SQL expression.

        """
        return util.preloaded.sql_functions.func.next_value(self)

    def _set_parent(self, parent: SchemaEventTarget, **kw: Any) -> None:
        column = parent
        assert isinstance(column, Column)
        super()._set_parent(column)
        column._on_table_attach(self._set_table)

    def _copy(self) -> Sequence:
        return Sequence(
            name=self.name,
            start=self.start,
            increment=self.increment,
            minvalue=self.minvalue,
            maxvalue=self.maxvalue,
            nominvalue=self.nominvalue,
            nomaxvalue=self.nomaxvalue,
            cycle=self.cycle,
            schema=self.schema,
            cache=self.cache,
            order=self.order,
            data_type=self.data_type,
            optional=self.optional,
            metadata=self.metadata,
            for_update=self.for_update,
        )

    def _set_table(self, column: Column[Any], table: Table) -> None:
        self._set_metadata(table.metadata)

    def _set_metadata(self, metadata: MetaData) -> None:
        self.metadata = metadata
        self.metadata._sequences[self._key] = self

    def create(self, bind: _CreateDropBind, checkfirst: bool = True) -> None:
        """Creates this sequence in the database."""

        bind._run_ddl_visitor(ddl.SchemaGenerator, self, checkfirst=checkfirst)

    def drop(self, bind: _CreateDropBind, checkfirst: bool = True) -> None:
        """Drops this sequence from the database."""

        bind._run_ddl_visitor(ddl.SchemaDropper, self, checkfirst=checkfirst)

    def _not_a_column_expr(self) -> NoReturn:
        raise exc.InvalidRequestError(
            f"This {self.__class__.__name__} cannot be used directly "
            "as a column expression.  Use func.next_value(sequence) "
            "to produce a 'next value' function that's usable "
            "as a column element."
        )


@inspection._self_inspects
class FetchedValue(SchemaEventTarget):
    """A marker for a transparent database-side default.

    Use :class:`.FetchedValue` when the database is configured
    to provide some automatic default for a column.

    E.g.::

        Column("foo", Integer, FetchedValue())

    Would indicate that some trigger or default generator
    will create a new value for the ``foo`` column during an
    INSERT.

    .. seealso::

        :ref:`triggered_columns`

    """

    is_server_default = True
    reflected = False
    has_argument = False
    is_clause_element = False
    is_identity = False

    column: Optional[Column[Any]]

    def __init__(self, for_update: bool = False) -> None:
        self.for_update = for_update

    def _as_for_update(self, for_update: bool) -> FetchedValue:
        if for_update == self.for_update:
            return self
        else:
            return self._clone(for_update)

    def _copy(self) -> FetchedValue:
        return FetchedValue(self.for_update)

    def _clone(self, for_update: bool) -> Self:
        n = self.__class__.__new__(self.__class__)
        n.__dict__.update(self.__dict__)
        n.__dict__.pop("column", None)
        n.for_update = for_update
        return n

    def _set_parent(self, parent: SchemaEventTarget, **kw: Any) -> None:
        column = parent
        assert isinstance(column, Column)
        self.column = column
        if self.for_update:
            self.column.server_onupdate = self
        else:
            self.column.server_default = self

    def __repr__(self) -> str:
        return util.generic_repr(self)


class DefaultClause(FetchedValue):
    """A DDL-specified DEFAULT column value.

    :class:`.DefaultClause` is a :class:`.FetchedValue`
    that also generates a "DEFAULT" clause when
    "CREATE TABLE" is emitted.

    :class:`.DefaultClause` is generated automatically
    whenever the ``server_default``, ``server_onupdate`` arguments of
    :class:`_schema.Column` are used.  A :class:`.DefaultClause`
    can be passed positionally as well.

    For example, the following::

        Column("foo", Integer, server_default="50")

    Is equivalent to::

        Column("foo", Integer, DefaultClause("50"))

    """

    has_argument = True

    def __init__(
        self,
        arg: Union[str, ClauseElement, TextClause],
        for_update: bool = False,
        _reflected: bool = False,
    ) -> None:
        util.assert_arg_type(arg, (str, ClauseElement, TextClause), "arg")
        super().__init__(for_update)
        self.arg = arg
        self.reflected = _reflected

    def _copy(self) -> DefaultClause:
        return DefaultClause(
            arg=self.arg, for_update=self.for_update, _reflected=self.reflected
        )

    def __repr__(self) -> str:
        return "DefaultClause(%r, for_update=%r)" % (self.arg, self.for_update)


class Constraint(DialectKWArgs, HasConditionalDDL, SchemaItem):
    """A table-level SQL constraint.

    :class:`_schema.Constraint` serves as the base class for the series of
    constraint objects that can be associated with :class:`_schema.Table`
    objects, including :class:`_schema.PrimaryKeyConstraint`,
    :class:`_schema.ForeignKeyConstraint`
    :class:`_schema.UniqueConstraint`, and
    :class:`_schema.CheckConstraint`.

    """

    __visit_name__ = "constraint"

    _creation_order: int
    _column_flag: bool

    def __init__(
        self,
        name: _ConstraintNameArgument = None,
        deferrable: Optional[bool] = None,
        initially: Optional[str] = None,
        info: Optional[_InfoType] = None,
        comment: Optional[str] = None,
        _create_rule: Optional[Any] = None,
        _type_bound: bool = False,
        **dialect_kw: Any,
    ) -> None:
        r"""Create a SQL constraint.

        :param name:
          Optional, the in-database name of this ``Constraint``.

        :param deferrable:
          Optional bool.  If set, emit DEFERRABLE or NOT DEFERRABLE when
          issuing DDL for this constraint.

        :param initially:
          Optional string.  If set, emit INITIALLY <value> when issuing DDL
          for this constraint.

        :param info: Optional data dictionary which will be populated into the
            :attr:`.SchemaItem.info` attribute of this object.

        :param comment: Optional string that will render an SQL comment on
          foreign key constraint creation.

            .. versionadded:: 2.0

        :param \**dialect_kw:  Additional keyword arguments are dialect
            specific, and passed in the form ``<dialectname>_<argname>``.  See
            the documentation regarding an individual dialect at
            :ref:`dialect_toplevel` for detail on documented arguments.

        :param _create_rule:
          used internally by some datatypes that also create constraints.

        :param _type_bound:
          used internally to indicate that this constraint is associated with
          a specific datatype.

        """

        self.name = name
        self.deferrable = deferrable
        self.initially = initially
        if info:
            self.info = info
        self._create_rule = _create_rule
        self._type_bound = _type_bound
        util.set_creation_order(self)
        self._validate_dialect_kwargs(dialect_kw)
        self.comment = comment

    def _should_create_for_compiler(
        self, compiler: DDLCompiler, **kw: Any
    ) -> bool:
        if self._create_rule is not None and not self._create_rule(compiler):
            return False
        elif self._ddl_if is not None:
            return self._ddl_if._should_execute(
                ddl.CreateConstraint(self), self, None, compiler=compiler, **kw
            )
        else:
            return True

    @property
    def table(self) -> Table:
        try:
            if isinstance(self.parent, Table):
                return self.parent
        except AttributeError:
            pass
        raise exc.InvalidRequestError(
            "This constraint is not bound to a table.  Did you "
            "mean to call table.append_constraint(constraint) ?"
        )

    def _set_parent(self, parent: SchemaEventTarget, **kw: Any) -> None:
        assert isinstance(parent, (Table, Column))
        self.parent = parent
        parent.constraints.add(self)

    @util.deprecated(
        "1.4",
        "The :meth:`_schema.Constraint.copy` method is deprecated "
        "and will be removed in a future release.",
    )
    def copy(self, **kw: Any) -> Self:
        return self._copy(**kw)

    def _copy(self, **kw: Any) -> Self:
        raise NotImplementedError()


class ColumnCollectionMixin:
    """A :class:`_expression.ColumnCollection` of :class:`_schema.Column`
    objects.

    This collection represents the columns which are referred to by
    this object.

    """

    _columns: DedupeColumnCollection[Column[Any]]

    _allow_multiple_tables = False

    _pending_colargs: List[Optional[Union[str, Column[Any]]]]

    if TYPE_CHECKING:

        def _set_parent_with_dispatch(
            self, parent: SchemaEventTarget, **kw: Any
        ) -> None: ...

    def __init__(
        self,
        *columns: _DDLColumnArgument,
        _autoattach: bool = True,
        _column_flag: bool = False,
        _gather_expressions: Optional[
            List[Union[str, ColumnElement[Any]]]
        ] = None,
    ) -> None:
        self._column_flag = _column_flag
        self._columns = DedupeColumnCollection()

        processed_expressions: Optional[
            List[Union[ColumnElement[Any], str]]
        ] = _gather_expressions

        if processed_expressions is not None:

            # this is expected to be an empty list
            assert not processed_expressions

            self._pending_colargs = []
            for (
                expr,
                _,
                _,
                add_element,
            ) in coercions.expect_col_expression_collection(
                roles.DDLConstraintColumnRole, columns
            ):
                self._pending_colargs.append(add_element)
                processed_expressions.append(expr)
        else:
            self._pending_colargs = [
                coercions.expect(roles.DDLConstraintColumnRole, column)
                for column in columns
            ]

        if _autoattach and self._pending_colargs:
            self._check_attach()

    def _check_attach(self, evt: bool = False) -> None:
        col_objs = [c for c in self._pending_colargs if isinstance(c, Column)]

        cols_w_table = [c for c in col_objs if isinstance(c.table, Table)]

        cols_wo_table = set(col_objs).difference(cols_w_table)
        if cols_wo_table:
            # feature #3341 - place event listeners for Column objects
            # such that when all those cols are attached, we autoattach.
            assert not evt, "Should not reach here on event call"

            # issue #3411 - don't do the per-column auto-attach if some of the
            # columns are specified as strings.
            has_string_cols = {
                c for c in self._pending_colargs if c is not None
            }.difference(col_objs)
            if not has_string_cols:

                def _col_attached(column: Column[Any], table: Table) -> None:
                    # this isinstance() corresponds with the
                    # isinstance() above; only want to count Table-bound
                    # columns
                    if isinstance(table, Table):
                        cols_wo_table.discard(column)
                        if not cols_wo_table:
                            self._check_attach(evt=True)

                self._cols_wo_table = cols_wo_table
                for col in cols_wo_table:
                    col._on_table_attach(_col_attached)
                return

        columns = cols_w_table

        tables = {c.table for c in columns}
        if len(tables) == 1:
            self._set_parent_with_dispatch(tables.pop())
        elif len(tables) > 1 and not self._allow_multiple_tables:
            table = columns[0].table
            others = [c for c in columns[1:] if c.table is not table]
            if others:
                # black could not format this inline
                other_str = ", ".join("'%s'" % c for c in others)
                raise exc.ArgumentError(
                    f"Column(s) {other_str} "
                    f"are not part of table '{table.description}'."
                )

    @util.ro_memoized_property
    def columns(self) -> ReadOnlyColumnCollection[str, Column[Any]]:
        return self._columns.as_readonly()

    @util.ro_memoized_property
    def c(self) -> ReadOnlyColumnCollection[str, Column[Any]]:
        return self._columns.as_readonly()

    def _col_expressions(
        self, parent: Union[Table, Column[Any]]
    ) -> List[Optional[Column[Any]]]:
        if isinstance(parent, Column):
            result: List[Optional[Column[Any]]] = [
                c for c in self._pending_colargs if isinstance(c, Column)
            ]
            assert len(result) == len(self._pending_colargs)
            return result
        else:
            try:
                return [
                    parent.c[col] if isinstance(col, str) else col
                    for col in self._pending_colargs
                ]
            except KeyError as ke:
                raise exc.ConstraintColumnNotFoundError(
                    f"Can't create {self.__class__.__name__} "
                    f"on table '{parent.description}': no column "
                    f"named '{ke.args[0]}' is present."
                ) from ke

    def _set_parent(self, parent: SchemaEventTarget, **kw: Any) -> None:
        assert isinstance(parent, (Table, Column))

        for col in self._col_expressions(parent):
            if col is not None:
                self._columns.add(col)


class ColumnCollectionConstraint(ColumnCollectionMixin, Constraint):
    """A constraint that proxies a ColumnCollection."""

    def __init__(
        self,
        *columns: _DDLColumnArgument,
        name: _ConstraintNameArgument = None,
        deferrable: Optional[bool] = None,
        initially: Optional[str] = None,
        info: Optional[_InfoType] = None,
        _autoattach: bool = True,
        _column_flag: bool = False,
        _gather_expressions: Optional[List[_DDLColumnArgument]] = None,
        **dialect_kw: Any,
    ) -> None:
        r"""
        :param \*columns:
          A sequence of column names or Column objects.

        :param name:
          Optional, the in-database name of this constraint.

        :param deferrable:
          Optional bool.  If set, emit DEFERRABLE or NOT DEFERRABLE when
          issuing DDL for this constraint.

        :param initially:
          Optional string.  If set, emit INITIALLY <value> when issuing DDL
          for this constraint.

        :param \**dialect_kw: other keyword arguments including
          dialect-specific arguments are propagated to the :class:`.Constraint`
          superclass.

        """
        Constraint.__init__(
            self,
            name=name,
            deferrable=deferrable,
            initially=initially,
            info=info,
            **dialect_kw,
        )
        ColumnCollectionMixin.__init__(
            self, *columns, _autoattach=_autoattach, _column_flag=_column_flag
        )

    columns: ReadOnlyColumnCollection[str, Column[Any]]
    """A :class:`_expression.ColumnCollection` representing the set of columns
    for this constraint.

    """

    def _set_parent(self, parent: SchemaEventTarget, **kw: Any) -> None:
        assert isinstance(parent, (Column, Table))
        Constraint._set_parent(self, parent)
        ColumnCollectionMixin._set_parent(self, parent)

    def __contains__(self, x: Any) -> bool:
        return x in self._columns

    @util.deprecated(
        "1.4",
        "The :meth:`_schema.ColumnCollectionConstraint.copy` method "
        "is deprecated and will be removed in a future release.",
    )
    def copy(
        self,
        *,
        target_table: Optional[Table] = None,
        **kw: Any,
    ) -> ColumnCollectionConstraint:
        return self._copy(target_table=target_table, **kw)

    def _copy(
        self,
        *,
        target_table: Optional[Table] = None,
        **kw: Any,
    ) -> ColumnCollectionConstraint:
        # ticket #5276
        constraint_kwargs = {}
        for dialect_name in self.dialect_options:
            dialect_options = self.dialect_options[dialect_name]._non_defaults
            for (
                dialect_option_key,
                dialect_option_value,
            ) in dialect_options.items():
                constraint_kwargs[dialect_name + "_" + dialect_option_key] = (
                    dialect_option_value
                )

        assert isinstance(self.parent, Table)
        c = self.__class__(
            name=self.name,
            deferrable=self.deferrable,
            initially=self.initially,
            *[
                _copy_expression(expr, self.parent, target_table)
                for expr in self._columns
            ],
            comment=self.comment,
            **constraint_kwargs,
        )
        return self._schema_item_copy(c)

    def contains_column(self, col: Column[Any]) -> bool:
        """Return True if this constraint contains the given column.

        Note that this object also contains an attribute ``.columns``
        which is a :class:`_expression.ColumnCollection` of
        :class:`_schema.Column` objects.

        """

        return self._columns.contains_column(col)

    def __iter__(self) -> Iterator[Column[Any]]:
        return iter(self._columns)

    def __len__(self) -> int:
        return len(self._columns)


class CheckConstraint(ColumnCollectionConstraint):
    """A table- or column-level CHECK constraint.

    Can be included in the definition of a Table or Column.
    """

    _allow_multiple_tables = True

    __visit_name__ = "table_or_column_check_constraint"

    @_document_text_coercion(
        "sqltext",
        ":class:`.CheckConstraint`",
        ":paramref:`.CheckConstraint.sqltext`",
    )
    def __init__(
        self,
        sqltext: _TextCoercedExpressionArgument[Any],
        name: _ConstraintNameArgument = None,
        deferrable: Optional[bool] = None,
        initially: Optional[str] = None,
        table: Optional[Table] = None,
        info: Optional[_InfoType] = None,
        _create_rule: Optional[Any] = None,
        _autoattach: bool = True,
        _type_bound: bool = False,
        **dialect_kw: Any,
    ) -> None:
        r"""Construct a CHECK constraint.

        :param sqltext:
         A string containing the constraint definition, which will be used
         verbatim, or a SQL expression construct.   If given as a string,
         the object is converted to a :func:`_expression.text` object.
         If the textual
         string includes a colon character, escape this using a backslash::

           CheckConstraint(r"foo ~ E'a(?\:b|c)d")

        :param name:
          Optional, the in-database name of the constraint.

        :param deferrable:
          Optional bool.  If set, emit DEFERRABLE or NOT DEFERRABLE when
          issuing DDL for this constraint.

        :param initially:
          Optional string.  If set, emit INITIALLY <value> when issuing DDL
          for this constraint.

        :param info: Optional data dictionary which will be populated into the
            :attr:`.SchemaItem.info` attribute of this object.

        """

        self.sqltext = coercions.expect(roles.DDLExpressionRole, sqltext)
        columns: List[Column[Any]] = []
        visitors.traverse(self.sqltext, {}, {"column": columns.append})

        super().__init__(
            name=name,
            deferrable=deferrable,
            initially=initially,
            _create_rule=_create_rule,
            info=info,
            _type_bound=_type_bound,
            _autoattach=_autoattach,
            *columns,
            **dialect_kw,
        )
        if table is not None:
            self._set_parent_with_dispatch(table)

    @property
    def is_column_level(self) -> bool:
        return not isinstance(self.parent, Table)

    @util.deprecated(
        "1.4",
        "The :meth:`_schema.CheckConstraint.copy` method is deprecated "
        "and will be removed in a future release.",
    )
    def copy(
        self, *, target_table: Optional[Table] = None, **kw: Any
    ) -> CheckConstraint:
        return self._copy(target_table=target_table, **kw)

    def _copy(
        self, *, target_table: Optional[Table] = None, **kw: Any
    ) -> CheckConstraint:
        if target_table is not None:
            # note that target_table is None for the copy process of
            # a column-bound CheckConstraint, so this path is not reached
            # in that case.
            sqltext = _copy_expression(self.sqltext, self.table, target_table)
        else:
            sqltext = self.sqltext
        c = CheckConstraint(
            sqltext,
            name=self.name,
            initially=self.initially,
            deferrable=self.deferrable,
            _create_rule=self._create_rule,
            table=target_table,
            comment=self.comment,
            _autoattach=False,
            _type_bound=self._type_bound,
        )
        return self._schema_item_copy(c)


class ForeignKeyConstraint(ColumnCollectionConstraint):
    """A table-level FOREIGN KEY constraint.

    Defines a single column or composite FOREIGN KEY ... REFERENCES
    constraint. For a no-frills, single column foreign key, adding a
    :class:`_schema.ForeignKey` to the definition of a :class:`_schema.Column`
    is a
    shorthand equivalent for an unnamed, single column
    :class:`_schema.ForeignKeyConstraint`.

    Examples of foreign key configuration are in :ref:`metadata_foreignkeys`.

    """

    __visit_name__ = "foreign_key_constraint"

    def __init__(
        self,
        columns: _typing_Sequence[_DDLColumnArgument],
        refcolumns: _typing_Sequence[_DDLColumnArgument],
        name: _ConstraintNameArgument = None,
        onupdate: Optional[str] = None,
        ondelete: Optional[str] = None,
        deferrable: Optional[bool] = None,
        initially: Optional[str] = None,
        use_alter: bool = False,
        link_to_name: bool = False,
        match: Optional[str] = None,
        table: Optional[Table] = None,
        info: Optional[_InfoType] = None,
        comment: Optional[str] = None,
        **dialect_kw: Any,
    ) -> None:
        r"""Construct a composite-capable FOREIGN KEY.

        :param columns: A sequence of local column names. The named columns
          must be defined and present in the parent Table. The names should
          match the ``key`` given to each column (defaults to the name) unless
          ``link_to_name`` is True.

        :param refcolumns: A sequence of foreign column names or Column
          objects. The columns must all be located within the same Table.

        :param name: Optional, the in-database name of the key.

        :param onupdate: Optional string. If set, emit ON UPDATE <value> when
            issuing DDL for this constraint. Typical values include CASCADE,
            DELETE and RESTRICT.

            .. seealso::

                :ref:`on_update_on_delete`

        :param ondelete: Optional string. If set, emit ON DELETE <value> when
            issuing DDL for this constraint. Typical values include CASCADE,
            SET NULL and RESTRICT.  Some dialects may allow for additional
            syntaxes.

            .. seealso::

                :ref:`on_update_on_delete`

        :param deferrable: Optional bool. If set, emit DEFERRABLE or NOT
          DEFERRABLE when issuing DDL for this constraint.

        :param initially: Optional string. If set, emit INITIALLY <value> when
          issuing DDL for this constraint.

        :param link_to_name: if True, the string name given in ``column`` is
          the rendered name of the referenced column, not its locally assigned
          ``key``.

        :param use_alter: If True, do not emit the DDL for this constraint as
          part of the CREATE TABLE definition. Instead, generate it via an
          ALTER TABLE statement issued after the full collection of tables
          have been created, and drop it via an ALTER TABLE statement before
          the full collection of tables are dropped.

          The use of :paramref:`_schema.ForeignKeyConstraint.use_alter` is
          particularly geared towards the case where two or more tables
          are established within a mutually-dependent foreign key constraint
          relationship; however, the :meth:`_schema.MetaData.create_all` and
          :meth:`_schema.MetaData.drop_all`
          methods will perform this resolution
          automatically, so the flag is normally not needed.

          .. seealso::

                :ref:`use_alter`

        :param match: Optional string. If set, emit MATCH <value> when issuing
          DDL for this constraint. Typical values include SIMPLE, PARTIAL
          and FULL.

        :param info: Optional data dictionary which will be populated into the
            :attr:`.SchemaItem.info` attribute of this object.

        :param comment: Optional string that will render an SQL comment on
          foreign key constraint creation.

            .. versionadded:: 2.0

        :param \**dialect_kw:  Additional keyword arguments are dialect
          specific, and passed in the form ``<dialectname>_<argname>``.  See
          the documentation regarding an individual dialect at
          :ref:`dialect_toplevel` for detail on documented arguments.

        """

        Constraint.__init__(
            self,
            name=name,
            deferrable=deferrable,
            initially=initially,
            info=info,
            comment=comment,
            **dialect_kw,
        )
        self.onupdate = onupdate
        self.ondelete = ondelete
        self.link_to_name = link_to_name
        self.use_alter = use_alter
        self.match = match

        if len(set(columns)) != len(refcolumns):
            if len(set(columns)) != len(columns):
                # e.g. FOREIGN KEY (a, a) REFERENCES r (b, c)
                raise exc.ArgumentError(
                    "ForeignKeyConstraint with duplicate source column "
                    "references are not supported."
                )
            else:
                # e.g. FOREIGN KEY (a) REFERENCES r (b, c)
                # paraphrasing
                # https://www.postgresql.org/docs/current/static/ddl-constraints.html
                raise exc.ArgumentError(
                    "ForeignKeyConstraint number "
                    "of constrained columns must match the number of "
                    "referenced columns."
                )

        # standalone ForeignKeyConstraint - create
        # associated ForeignKey objects which will be applied to hosted
        # Column objects (in col.foreign_keys), either now or when attached
        # to the Table for string-specified names
        self.elements = [
            ForeignKey(
                refcol,
                _constraint=self,
                name=self.name,
                onupdate=self.onupdate,
                ondelete=self.ondelete,
                use_alter=self.use_alter,
                link_to_name=self.link_to_name,
                match=self.match,
                deferrable=self.deferrable,
                initially=self.initially,
                **self.dialect_kwargs,
            )
            for refcol in refcolumns
        ]

        ColumnCollectionMixin.__init__(self, *columns)
        if table is not None:
            if hasattr(self, "parent"):
                assert table is self.parent
            self._set_parent_with_dispatch(table)

    def _append_element(self, column: Column[Any], fk: ForeignKey) -> None:
        self._columns.add(column)
        self.elements.append(fk)

    columns: ReadOnlyColumnCollection[str, Column[Any]]
    """A :class:`_expression.ColumnCollection` representing the set of columns
    for this constraint.

    """

    elements: List[ForeignKey]
    """A sequence of :class:`_schema.ForeignKey` objects.

    Each :class:`_schema.ForeignKey`
    represents a single referring column/referred
    column pair.

    This collection is intended to be read-only.

    """

    @property
    def _elements(self) -> util.OrderedDict[str, ForeignKey]:
        # legacy - provide a dictionary view of (column_key, fk)
        return util.OrderedDict(zip(self.column_keys, self.elements))

    @property
    def _referred_schema(self) -> Optional[str]:
        for elem in self.elements:
            return elem._referred_schema
        else:
            return None

    @property
    def referred_table(self) -> Table:
        """The :class:`_schema.Table` object to which this
        :class:`_schema.ForeignKeyConstraint` references.

        This is a dynamically calculated attribute which may not be available
        if the constraint and/or parent table is not yet associated with
        a metadata collection that contains the referred table.

        """
        return self.elements[0].column.table

    def _validate_dest_table(self, table: Table) -> None:
        table_keys = {elem._table_key() for elem in self.elements}
        if None not in table_keys and len(table_keys) > 1:
            elem0, elem1 = sorted(table_keys)[0:2]
            raise exc.ArgumentError(
                f"ForeignKeyConstraint on "
                f"{table.fullname}({self._col_description}) refers to "
                f"multiple remote tables: {elem0} and {elem1}"
            )

    @property
    def column_keys(self) -> _typing_Sequence[str]:
        """Return a list of string keys representing the local
        columns in this :class:`_schema.ForeignKeyConstraint`.

        This list is either the original string arguments sent
        to the constructor of the :class:`_schema.ForeignKeyConstraint`,
        or if the constraint has been initialized with :class:`_schema.Column`
        objects, is the string ``.key`` of each element.

        """
        if hasattr(self, "parent"):
            return self._columns.keys()
        else:
            return [
                col.key if isinstance(col, ColumnElement) else str(col)
                for col in self._pending_colargs
            ]

    @property
    def _col_description(self) -> str:
        return ", ".join(self.column_keys)

    def _set_parent(self, parent: SchemaEventTarget, **kw: Any) -> None:
        table = parent
        assert isinstance(table, Table)
        Constraint._set_parent(self, table)

        ColumnCollectionConstraint._set_parent(self, table)

        for col, fk in zip(self._columns, self.elements):
            if not hasattr(fk, "parent") or fk.parent is not col:
                fk._set_parent_with_dispatch(col)

        self._validate_dest_table(table)

    @util.deprecated(
        "1.4",
        "The :meth:`_schema.ForeignKeyConstraint.copy` method is deprecated "
        "and will be removed in a future release.",
    )
    def copy(
        self,
        *,
        schema: Optional[str] = None,
        target_table: Optional[Table] = None,
        **kw: Any,
    ) -> ForeignKeyConstraint:
        return self._copy(schema=schema, target_table=target_table, **kw)

    def _copy(
        self,
        *,
        schema: Optional[str] = None,
        target_table: Optional[Table] = None,
        **kw: Any,
    ) -> ForeignKeyConstraint:
        fkc = ForeignKeyConstraint(
            [x.parent.key for x in self.elements],
            [
                x._get_colspec(
                    schema=schema,
                    table_name=(
                        target_table.name
                        if target_table is not None
                        and x._table_key() == x.parent.table.key
                        else None
                    ),
                    _is_copy=True,
                )
                for x in self.elements
            ],
            name=self.name,
            onupdate=self.onupdate,
            ondelete=self.ondelete,
            use_alter=self.use_alter,
            deferrable=self.deferrable,
            initially=self.initially,
            link_to_name=self.link_to_name,
            match=self.match,
            comment=self.comment,
        )
        for self_fk, other_fk in zip(self.elements, fkc.elements):
            self_fk._schema_item_copy(other_fk)
        return self._schema_item_copy(fkc)


class PrimaryKeyConstraint(ColumnCollectionConstraint):
    """A table-level PRIMARY KEY constraint.

    The :class:`.PrimaryKeyConstraint` object is present automatically
    on any :class:`_schema.Table` object; it is assigned a set of
    :class:`_schema.Column` objects corresponding to those marked with
    the :paramref:`_schema.Column.primary_key` flag::

        >>> my_table = Table(
        ...     "mytable",
        ...     metadata,
        ...     Column("id", Integer, primary_key=True),
        ...     Column("version_id", Integer, primary_key=True),
        ...     Column("data", String(50)),
        ... )
        >>> my_table.primary_key
        PrimaryKeyConstraint(
            Column('id', Integer(), table=<mytable>,
                   primary_key=True, nullable=False),
            Column('version_id', Integer(), table=<mytable>,
                   primary_key=True, nullable=False)
        )

    The primary key of a :class:`_schema.Table` can also be specified by using
    a :class:`.PrimaryKeyConstraint` object explicitly; in this mode of usage,
    the "name" of the constraint can also be specified, as well as other
    options which may be recognized by dialects::

        my_table = Table(
            "mytable",
            metadata,
            Column("id", Integer),
            Column("version_id", Integer),
            Column("data", String(50)),
            PrimaryKeyConstraint("id", "version_id", name="mytable_pk"),
        )

    The two styles of column-specification should generally not be mixed.
    An warning is emitted if the columns present in the
    :class:`.PrimaryKeyConstraint`
    don't match the columns that were marked as ``primary_key=True``, if both
    are present; in this case, the columns are taken strictly from the
    :class:`.PrimaryKeyConstraint` declaration, and those columns otherwise
    marked as ``primary_key=True`` are ignored.  This behavior is intended to
    be backwards compatible with previous behavior.

    For the use case where specific options are to be specified on the
    :class:`.PrimaryKeyConstraint`, but the usual style of using
    ``primary_key=True`` flags is still desirable, an empty
    :class:`.PrimaryKeyConstraint` may be specified, which will take on the
    primary key column collection from the :class:`_schema.Table` based on the
    flags::

        my_table = Table(
            "mytable",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("version_id", Integer, primary_key=True),
            Column("data", String(50)),
            PrimaryKeyConstraint(name="mytable_pk", mssql_clustered=True),
        )

    """

    __visit_name__ = "primary_key_constraint"

    def __init__(
        self,
        *columns: _DDLColumnArgument,
        name: Optional[str] = None,
        deferrable: Optional[bool] = None,
        initially: Optional[str] = None,
        info: Optional[_InfoType] = None,
        _implicit_generated: bool = False,
        **dialect_kw: Any,
    ) -> None:
        self._implicit_generated = _implicit_generated
        super().__init__(
            *columns,
            name=name,
            deferrable=deferrable,
            initially=initially,
            info=info,
            **dialect_kw,
        )

    def _set_parent(self, parent: SchemaEventTarget, **kw: Any) -> None:
        table = parent
        assert isinstance(table, Table)
        super()._set_parent(table)

        if table.primary_key is not self:
            table.constraints.discard(table.primary_key)
            table.primary_key = self  # type: ignore
            table.constraints.add(self)

        table_pks = [c for c in table.c if c.primary_key]
        if (
            self._columns
            and table_pks
            and set(table_pks) != set(self._columns)
        ):
            # black could not format these inline
            table_pk_str = ", ".join("'%s'" % c.name for c in table_pks)
            col_str = ", ".join("'%s'" % c.name for c in self._columns)

            util.warn(
                f"Table '{table.name}' specifies columns "
                f"{table_pk_str} as "
                f"primary_key=True, "
                f"not matching locally specified columns {col_str}; "
                f"setting the "
                f"current primary key columns to "
                f"{col_str}. "
                f"This warning "
                f"may become an exception in a future release"
            )
            table_pks[:] = []

        for c in self._columns:
            c.primary_key = True
            if c._user_defined_nullable is NULL_UNSPECIFIED:
                c.nullable = False
        if table_pks:
            self._columns.extend(table_pks)

    def _reload(self, columns: Iterable[Column[Any]]) -> None:
        """repopulate this :class:`.PrimaryKeyConstraint` given
        a set of columns.

        Existing columns in the table that are marked as primary_key=True
        are maintained.

        Also fires a new event.

        This is basically like putting a whole new
        :class:`.PrimaryKeyConstraint` object on the parent
        :class:`_schema.Table` object without actually replacing the object.

        The ordering of the given list of columns is also maintained; these
        columns will be appended to the list of columns after any which
        are already present.

        """
        # set the primary key flag on new columns.
        # note any existing PK cols on the table also have their
        # flag still set.
        for col in columns:
            col.primary_key = True

        self._columns.extend(columns)

        PrimaryKeyConstraint._autoincrement_column._reset(self)  # type: ignore
        self._set_parent_with_dispatch(self.table)

    def _replace(self, col: Column[Any]) -> None:
        PrimaryKeyConstraint._autoincrement_column._reset(self)  # type: ignore
        self._columns.replace(col)

        self.dispatch._sa_event_column_added_to_pk_constraint(self, col)

    @property
    def columns_autoinc_first(self) -> List[Column[Any]]:
        autoinc = self._autoincrement_column

        if autoinc is not None:
            return [autoinc] + [c for c in self._columns if c is not autoinc]
        else:
            return list(self._columns)

    @util.ro_memoized_property
    def _autoincrement_column(self) -> Optional[Column[int]]:
        def _validate_autoinc(col: Column[Any], autoinc_true: bool) -> bool:
            if col.type._type_affinity is None or not issubclass(
                col.type._type_affinity,
                (
                    type_api.INTEGERTYPE._type_affinity,
                    type_api.NUMERICTYPE._type_affinity,
                ),
            ):
                if autoinc_true:
                    raise exc.ArgumentError(
                        f"Column type {col.type} on column '{col}' is not "
                        f"compatible with autoincrement=True"
                    )
                else:
                    return False
            elif (
                not isinstance(col.default, (type(None), Sequence))
                and not autoinc_true
            ):
                return False
            elif (
                col.server_default is not None
                and not isinstance(col.server_default, Identity)
                and not autoinc_true
            ):
                return False
            elif col.foreign_keys and col.autoincrement not in (
                True,
                "ignore_fk",
            ):
                return False
            return True

        if len(self._columns) == 1:
            col = list(self._columns)[0]

            if col.autoincrement is True:
                _validate_autoinc(col, True)
                return col
            elif col.autoincrement in (
                "auto",
                "ignore_fk",
            ) and _validate_autoinc(col, False):
                return col
            else:
                return None

        else:
            autoinc = None
            for col in self._columns:
                if col.autoincrement is True:
                    _validate_autoinc(col, True)
                    if autoinc is not None:
                        raise exc.ArgumentError(
                            f"Only one Column may be marked "
                            f"autoincrement=True, found both "
                            f"{col.name} and {autoinc.name}."
                        )
                    else:
                        autoinc = col

            return autoinc


class UniqueConstraint(ColumnCollectionConstraint):
    """A table-level UNIQUE constraint.

    Defines a single column or composite UNIQUE constraint. For a no-frills,
    single column constraint, adding ``unique=True`` to the ``Column``
    definition is a shorthand equivalent for an unnamed, single column
    UniqueConstraint.
    """

    __visit_name__ = "unique_constraint"


class Index(
    DialectKWArgs, ColumnCollectionMixin, HasConditionalDDL, SchemaItem
):
    """A table-level INDEX.

    Defines a composite (one or more column) INDEX.

    E.g.::

        sometable = Table(
            "sometable",
            metadata,
            Column("name", String(50)),
            Column("address", String(100)),
        )

        Index("some_index", sometable.c.name)

    For a no-frills, single column index, adding
    :class:`_schema.Column` also supports ``index=True``::

        sometable = Table(
            "sometable", metadata, Column("name", String(50), index=True)
        )

    For a composite index, multiple columns can be specified::

        Index("some_index", sometable.c.name, sometable.c.address)

    Functional indexes are supported as well, typically by using the
    :data:`.func` construct in conjunction with table-bound
    :class:`_schema.Column` objects::

        Index("some_index", func.lower(sometable.c.name))

    An :class:`.Index` can also be manually associated with a
    :class:`_schema.Table`,
    either through inline declaration or using
    :meth:`_schema.Table.append_constraint`.  When this approach is used,
    the names
    of the indexed columns can be specified as strings::

        Table(
            "sometable",
            metadata,
            Column("name", String(50)),
            Column("address", String(100)),
            Index("some_index", "name", "address"),
        )

    To support functional or expression-based indexes in this form, the
    :func:`_expression.text` construct may be used::

        from sqlalchemy import text

        Table(
            "sometable",
            metadata,
            Column("name", String(50)),
            Column("address", String(100)),
            Index("some_index", text("lower(name)")),
        )

    .. seealso::

        :ref:`schema_indexes` - General information on :class:`.Index`.

        :ref:`postgresql_indexes` - PostgreSQL-specific options available for
        the :class:`.Index` construct.

        :ref:`mysql_indexes` - MySQL-specific options available for the
        :class:`.Index` construct.

        :ref:`mssql_indexes` - MSSQL-specific options available for the
        :class:`.Index` construct.

    """

    __visit_name__ = "index"

    table: Optional[Table]
    expressions: _typing_Sequence[Union[str, ColumnElement[Any]]]
    _table_bound_expressions: _typing_Sequence[ColumnElement[Any]]

    def __init__(
        self,
        name: Optional[str],
        *expressions: _DDLColumnArgument,
        unique: bool = False,
        quote: Optional[bool] = None,
        info: Optional[_InfoType] = None,
        _table: Optional[Table] = None,
        _column_flag: bool = False,
        **dialect_kw: Any,
    ) -> None:
        r"""Construct an index object.

        :param name:
          The name of the index

        :param \*expressions:
          Column expressions to include in the index.   The expressions
          are normally instances of :class:`_schema.Column`, but may also
          be arbitrary SQL expressions which ultimately refer to a
          :class:`_schema.Column`.

        :param unique=False:
            Keyword only argument; if True, create a unique index.

        :param quote=None:
            Keyword only argument; whether to apply quoting to the name of
            the index.  Works in the same manner as that of
            :paramref:`_schema.Column.quote`.

        :param info=None: Optional data dictionary which will be populated
            into the :attr:`.SchemaItem.info` attribute of this object.

        :param \**dialect_kw: Additional keyword arguments not mentioned above
            are dialect specific, and passed in the form
            ``<dialectname>_<argname>``. See the documentation regarding an
            individual dialect at :ref:`dialect_toplevel` for detail on
            documented arguments.

        """
        self.table = table = None

        self.name = quoted_name.construct(name, quote)
        self.unique = unique
        if info is not None:
            self.info = info

        # TODO: consider "table" argument being public, but for
        # the purpose of the fix here, it starts as private.
        if _table is not None:
            table = _table

        self._validate_dialect_kwargs(dialect_kw)

        self.expressions = []
        # will call _set_parent() if table-bound column
        # objects are present
        ColumnCollectionMixin.__init__(
            self,
            *expressions,
            _column_flag=_column_flag,
            _gather_expressions=self.expressions,
        )
        if table is not None:
            self._set_parent(table)

    def _set_parent(self, parent: SchemaEventTarget, **kw: Any) -> None:
        table = parent
        assert isinstance(table, Table)
        ColumnCollectionMixin._set_parent(self, table)

        if self.table is not None and table is not self.table:
            raise exc.ArgumentError(
                f"Index '{self.name}' is against table "
                f"'{self.table.description}', and "
                f"cannot be associated with table '{table.description}'."
            )
        self.table = table
        table.indexes.add(self)

        expressions = self.expressions
        col_expressions = self._col_expressions(table)
        assert len(expressions) == len(col_expressions)

        exprs = []
        for expr, colexpr in zip(expressions, col_expressions):
            if isinstance(expr, ClauseElement):
                exprs.append(expr)
            elif colexpr is not None:
                exprs.append(colexpr)
            else:
                assert False
        self.expressions = self._table_bound_expressions = exprs

    def create(self, bind: _CreateDropBind, checkfirst: bool = False) -> None:
        """Issue a ``CREATE`` statement for this
        :class:`.Index`, using the given
        :class:`.Connection` or :class:`.Engine`` for connectivity.

        .. seealso::

            :meth:`_schema.MetaData.create_all`.

        """
        bind._run_ddl_visitor(ddl.SchemaGenerator, self, checkfirst=checkfirst)

    def drop(self, bind: _CreateDropBind, checkfirst: bool = False) -> None:
        """Issue a ``DROP`` statement for this
        :class:`.Index`, using the given
        :class:`.Connection` or :class:`.Engine` for connectivity.

        .. seealso::

            :meth:`_schema.MetaData.drop_all`.

        """
        bind._run_ddl_visitor(ddl.SchemaDropper, self, checkfirst=checkfirst)

    def __repr__(self) -> str:
        exprs: _typing_Sequence[Any]  # noqa: F842

        return "Index(%s)" % (
            ", ".join(
                [repr(self.name)]
                + [repr(e) for e in self.expressions]
                + (self.unique and ["unique=True"] or [])
            )
        )


_NamingSchemaCallable = Callable[[Constraint, Table], str]
_NamingSchemaDirective = Union[str, _NamingSchemaCallable]


class _NamingSchemaTD(TypedDict, total=False):
    fk: _NamingSchemaDirective
    pk: _NamingSchemaDirective
    ix: _NamingSchemaDirective
    ck: _NamingSchemaDirective
    uq: _NamingSchemaDirective


_NamingSchemaParameter = Union[
    # it seems like the TypedDict here is useful for pylance typeahead,
    # and not much else
    _NamingSchemaTD,
    # there is no form that allows Union[Type[Any], str] to work in all
    # cases, including breaking out Mapping[] entries for each combination
    # even, therefore keys must be `Any` (see #10264)
    Mapping[Any, _NamingSchemaDirective],
]


DEFAULT_NAMING_CONVENTION: _NamingSchemaParameter = util.immutabledict(
    {"ix": "ix_%(column_0_label)s"}
)


class MetaData(HasSchemaAttr):
    """A collection of :class:`_schema.Table`
    objects and their associated schema
    constructs.

    Holds a collection of :class:`_schema.Table` objects as well as
    an optional binding to an :class:`_engine.Engine` or
    :class:`_engine.Connection`.  If bound, the :class:`_schema.Table` objects
    in the collection and their columns may participate in implicit SQL
    execution.

    The :class:`_schema.Table` objects themselves are stored in the
    :attr:`_schema.MetaData.tables` dictionary.

    :class:`_schema.MetaData` is a thread-safe object for read operations.
    Construction of new tables within a single :class:`_schema.MetaData`
    object,
    either explicitly or via reflection, may not be completely thread-safe.

    .. seealso::

        :ref:`metadata_describing` - Introduction to database metadata

    """

    __visit_name__ = "metadata"

    def __init__(
        self,
        schema: Optional[str] = None,
        quote_schema: Optional[bool] = None,
        naming_convention: Optional[_NamingSchemaParameter] = None,
        info: Optional[_InfoType] = None,
    ) -> None:
        """Create a new MetaData object.

        :param schema:
           The default schema to use for the :class:`_schema.Table`,
           :class:`.Sequence`, and potentially other objects associated with
           this :class:`_schema.MetaData`. Defaults to ``None``.

           .. seealso::

                :ref:`schema_metadata_schema_name` - details on how the
                :paramref:`_schema.MetaData.schema` parameter is used.

                :paramref:`_schema.Table.schema`

                :paramref:`.Sequence.schema`

        :param quote_schema:
            Sets the ``quote_schema`` flag for those :class:`_schema.Table`,
            :class:`.Sequence`, and other objects which make usage of the
            local ``schema`` name.

        :param info: Optional data dictionary which will be populated into the
            :attr:`.SchemaItem.info` attribute of this object.

        :param naming_convention: a dictionary referring to values which
          will establish default naming conventions for :class:`.Constraint`
          and :class:`.Index` objects, for those objects which are not given
          a name explicitly.

          The keys of this dictionary may be:

          * a constraint or Index class, e.g. the :class:`.UniqueConstraint`,
            :class:`_schema.ForeignKeyConstraint` class, the :class:`.Index`
            class

          * a string mnemonic for one of the known constraint classes;
            ``"fk"``, ``"pk"``, ``"ix"``, ``"ck"``, ``"uq"`` for foreign key,
            primary key, index, check, and unique constraint, respectively.

          * the string name of a user-defined "token" that can be used
            to define new naming tokens.

          The values associated with each "constraint class" or "constraint
          mnemonic" key are string naming templates, such as
          ``"uq_%(table_name)s_%(column_0_name)s"``,
          which describe how the name should be composed.  The values
          associated with user-defined "token" keys should be callables of the
          form ``fn(constraint, table)``, which accepts the constraint/index
          object and :class:`_schema.Table` as arguments, returning a string
          result.

          The built-in names are as follows, some of which may only be
          available for certain types of constraint:

            * ``%(table_name)s`` - the name of the :class:`_schema.Table`
              object
              associated with the constraint.

            * ``%(referred_table_name)s`` - the name of the
              :class:`_schema.Table`
              object associated with the referencing target of a
              :class:`_schema.ForeignKeyConstraint`.

            * ``%(column_0_name)s`` - the name of the :class:`_schema.Column`
              at
              index position "0" within the constraint.

            * ``%(column_0N_name)s`` - the name of all :class:`_schema.Column`
              objects in order within the constraint, joined without a
              separator.

            * ``%(column_0_N_name)s`` - the name of all
              :class:`_schema.Column`
              objects in order within the constraint, joined with an
              underscore as a separator.

            * ``%(column_0_label)s``, ``%(column_0N_label)s``,
              ``%(column_0_N_label)s`` - the label of either the zeroth
              :class:`_schema.Column` or all :class:`.Columns`, separated with
              or without an underscore

            * ``%(column_0_key)s``, ``%(column_0N_key)s``,
              ``%(column_0_N_key)s`` - the key of either the zeroth
              :class:`_schema.Column` or all :class:`.Columns`, separated with
              or without an underscore

            * ``%(referred_column_0_name)s``, ``%(referred_column_0N_name)s``
              ``%(referred_column_0_N_name)s``,  ``%(referred_column_0_key)s``,
              ``%(referred_column_0N_key)s``, ...  column tokens which
              render the names/keys/labels of columns that are referenced
              by a  :class:`_schema.ForeignKeyConstraint`.

            * ``%(constraint_name)s`` - a special key that refers to the
              existing name given to the constraint.  When this key is
              present, the :class:`.Constraint` object's existing name will be
              replaced with one that is composed from template string that
              uses this token. When this token is present, it is required that
              the :class:`.Constraint` is given an explicit name ahead of time.

            * user-defined: any additional token may be implemented by passing
              it along with a ``fn(constraint, table)`` callable to the
              naming_convention dictionary.

          .. versionadded:: 1.3.0 - added new ``%(column_0N_name)s``,
             ``%(column_0_N_name)s``, and related tokens that produce
             concatenations of names, keys, or labels for all columns referred
             to by a given constraint.

          .. seealso::

                :ref:`constraint_naming_conventions` - for detailed usage
                examples.

        """
        if schema is not None and not isinstance(schema, str):
            raise exc.ArgumentError(
                "expected schema argument to be a string, "
                f"got {type(schema)}."
            )
        self.tables = util.FacadeDict()
        self.schema = quoted_name.construct(schema, quote_schema)
        self.naming_convention = (
            naming_convention
            if naming_convention
            else DEFAULT_NAMING_CONVENTION
        )
        if info:
            self.info = info
        self._schemas: Set[str] = set()
        self._sequences: Dict[str, Sequence] = {}
        self._fk_memos: Dict[Tuple[str, Optional[str]], List[ForeignKey]] = (
            collections.defaultdict(list)
        )

    tables: util.FacadeDict[str, Table]
    """A dictionary of :class:`_schema.Table`
    objects keyed to their name or "table key".

    The exact key is that determined by the :attr:`_schema.Table.key`
    attribute;
    for a table with no :attr:`_schema.Table.schema` attribute,
    this is the same
    as :attr:`_schema.Table.name`.  For a table with a schema,
    it is typically of the
    form ``schemaname.tablename``.

    .. seealso::

        :attr:`_schema.MetaData.sorted_tables`

    """

    def __repr__(self) -> str:
        return "MetaData()"

    def __contains__(self, table_or_key: Union[str, Table]) -> bool:
        if not isinstance(table_or_key, str):
            table_or_key = table_or_key.key
        return table_or_key in self.tables

    def _add_table(
        self, name: str, schema: Optional[str], table: Table
    ) -> None:
        key = _get_table_key(name, schema)
        self.tables._insert_item(key, table)
        if schema:
            self._schemas.add(schema)

    def _remove_table(self, name: str, schema: Optional[str]) -> None:
        key = _get_table_key(name, schema)
        removed = dict.pop(self.tables, key, None)
        if removed is not None:
            for fk in removed.foreign_keys:
                fk._remove_from_metadata(self)
        if self._schemas:
            self._schemas = {
                t.schema for t in self.tables.values() if t.schema is not None
            }

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "tables": self.tables,
            "schema": self.schema,
            "schemas": self._schemas,
            "sequences": self._sequences,
            "fk_memos": self._fk_memos,
            "naming_convention": self.naming_convention,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.tables = state["tables"]
        self.schema = state["schema"]
        self.naming_convention = state["naming_convention"]
        self._sequences = state["sequences"]
        self._schemas = state["schemas"]
        self._fk_memos = state["fk_memos"]

    def clear(self) -> None:
        """Clear all Table objects from this MetaData."""

        dict.clear(self.tables)  # type: ignore
        self._schemas.clear()
        self._fk_memos.clear()

    def remove(self, table: Table) -> None:
        """Remove the given Table object from this MetaData."""

        self._remove_table(table.name, table.schema)

    @property
    def sorted_tables(self) -> List[Table]:
        """Returns a list of :class:`_schema.Table` objects sorted in order of
        foreign key dependency.

        The sorting will place :class:`_schema.Table`
        objects that have dependencies
        first, before the dependencies themselves, representing the
        order in which they can be created.   To get the order in which
        the tables would be dropped, use the ``reversed()`` Python built-in.

        .. warning::

            The :attr:`.MetaData.sorted_tables` attribute cannot by itself
            accommodate automatic resolution of dependency cycles between
            tables, which are usually caused by mutually dependent foreign key
            constraints. When these cycles are detected, the foreign keys
            of these tables are omitted from consideration in the sort.
            A warning is emitted when this condition occurs, which will be an
            exception raise in a future release.   Tables which are not part
            of the cycle will still be returned in dependency order.

            To resolve these cycles, the
            :paramref:`_schema.ForeignKeyConstraint.use_alter` parameter may be
            applied to those constraints which create a cycle.  Alternatively,
            the :func:`_schema.sort_tables_and_constraints` function will
            automatically return foreign key constraints in a separate
            collection when cycles are detected so that they may be applied
            to a schema separately.

            .. versionchanged:: 1.3.17 - a warning is emitted when
               :attr:`.MetaData.sorted_tables` cannot perform a proper sort
               due to cyclical dependencies.  This will be an exception in a
               future release.  Additionally, the sort will continue to return
               other tables not involved in the cycle in dependency order which
               was not the case previously.

        .. seealso::

            :func:`_schema.sort_tables`

            :func:`_schema.sort_tables_and_constraints`

            :attr:`_schema.MetaData.tables`

            :meth:`_reflection.Inspector.get_table_names`

            :meth:`_reflection.Inspector.get_sorted_table_and_fkc_names`


        """
        return ddl.sort_tables(
            sorted(self.tables.values(), key=lambda t: t.key)  # type: ignore
        )

    # overload needed to work around mypy this mypy
    # https://github.com/python/mypy/issues/17093
    @overload
    def reflect(
        self,
        bind: Engine,
        schema: Optional[str] = ...,
        views: bool = ...,
        only: Union[
            _typing_Sequence[str], Callable[[str, MetaData], bool], None
        ] = ...,
        extend_existing: bool = ...,
        autoload_replace: bool = ...,
        resolve_fks: bool = ...,
        **dialect_kwargs: Any,
    ) -> None: ...

    @overload
    def reflect(
        self,
        bind: Connection,
        schema: Optional[str] = ...,
        views: bool = ...,
        only: Union[
            _typing_Sequence[str], Callable[[str, MetaData], bool], None
        ] = ...,
        extend_existing: bool = ...,
        autoload_replace: bool = ...,
        resolve_fks: bool = ...,
        **dialect_kwargs: Any,
    ) -> None: ...

    @util.preload_module("sqlalchemy.engine.reflection")
    def reflect(
        self,
        bind: Union[Engine, Connection],
        schema: Optional[str] = None,
        views: bool = False,
        only: Union[
            _typing_Sequence[str], Callable[[str, MetaData], bool], None
        ] = None,
        extend_existing: bool = False,
        autoload_replace: bool = True,
        resolve_fks: bool = True,
        **dialect_kwargs: Any,
    ) -> None:
        r"""Load all available table definitions from the database.

        Automatically creates ``Table`` entries in this ``MetaData`` for any
        table available in the database but not yet present in the
        ``MetaData``.  May be called multiple times to pick up tables recently
        added to the database, however no special action is taken if a table
        in this ``MetaData`` no longer exists in the database.

        :param bind:
          A :class:`.Connection` or :class:`.Engine` used to access the
          database.

        :param schema:
          Optional, query and reflect tables from an alternate schema.
          If None, the schema associated with this :class:`_schema.MetaData`
          is used, if any.

        :param views:
          If True, also reflect views (materialized and plain).

        :param only:
          Optional.  Load only a sub-set of available named tables.  May be
          specified as a sequence of names or a callable.

          If a sequence of names is provided, only those tables will be
          reflected.  An error is raised if a table is requested but not
          available.  Named tables already present in this ``MetaData`` are
          ignored.

          If a callable is provided, it will be used as a boolean predicate to
          filter the list of potential table names.  The callable is called
          with a table name and this ``MetaData`` instance as positional
          arguments and should return a true value for any table to reflect.

        :param extend_existing: Passed along to each :class:`_schema.Table` as
          :paramref:`_schema.Table.extend_existing`.

        :param autoload_replace: Passed along to each :class:`_schema.Table`
          as
          :paramref:`_schema.Table.autoload_replace`.

        :param resolve_fks: if True, reflect :class:`_schema.Table`
         objects linked
         to :class:`_schema.ForeignKey` objects located in each
         :class:`_schema.Table`.
         For :meth:`_schema.MetaData.reflect`,
         this has the effect of reflecting
         related tables that might otherwise not be in the list of tables
         being reflected, for example if the referenced table is in a
         different schema or is omitted via the
         :paramref:`.MetaData.reflect.only` parameter.  When False,
         :class:`_schema.ForeignKey` objects are not followed to the
         :class:`_schema.Table`
         in which they link, however if the related table is also part of the
         list of tables that would be reflected in any case, the
         :class:`_schema.ForeignKey` object will still resolve to its related
         :class:`_schema.Table` after the :meth:`_schema.MetaData.reflect`
         operation is
         complete.   Defaults to True.

         .. versionadded:: 1.3.0

         .. seealso::

            :paramref:`_schema.Table.resolve_fks`

        :param \**dialect_kwargs: Additional keyword arguments not mentioned
         above are dialect specific, and passed in the form
         ``<dialectname>_<argname>``.  See the documentation regarding an
         individual dialect at :ref:`dialect_toplevel` for detail on
         documented arguments.

        .. seealso::

            :ref:`metadata_reflection_toplevel`

            :meth:`_events.DDLEvents.column_reflect` - Event used to customize
            the reflected columns. Usually used to generalize the types using
            :meth:`_types.TypeEngine.as_generic`

            :ref:`metadata_reflection_dbagnostic_types` - describes how to
            reflect tables using general types.

        """

        with inspection.inspect(bind)._inspection_context() as insp:
            reflect_opts: Any = {
                "autoload_with": insp,
                "extend_existing": extend_existing,
                "autoload_replace": autoload_replace,
                "resolve_fks": resolve_fks,
                "_extend_on": set(),
            }

            reflect_opts.update(dialect_kwargs)

            if schema is None:
                schema = self.schema

            if schema is not None:
                reflect_opts["schema"] = schema

            kind = util.preloaded.engine_reflection.ObjectKind.TABLE
            available: util.OrderedSet[str] = util.OrderedSet(
                insp.get_table_names(schema)
            )
            if views:
                kind = util.preloaded.engine_reflection.ObjectKind.ANY
                available.update(insp.get_view_names(schema))
                try:
                    available.update(insp.get_materialized_view_names(schema))
                except NotImplementedError:
                    pass

            if schema is not None:
                available_w_schema: util.OrderedSet[str] = util.OrderedSet(
                    [f"{schema}.{name}" for name in available]
                )
            else:
                available_w_schema = available

            current = set(self.tables)

            if only is None:
                load = [
                    name
                    for name, schname in zip(available, available_w_schema)
                    if extend_existing or schname not in current
                ]
            elif callable(only):
                load = [
                    name
                    for name, schname in zip(available, available_w_schema)
                    if (extend_existing or schname not in current)
                    and only(name, self)
                ]
            else:
                missing = [name for name in only if name not in available]
                if missing:
                    s = schema and (" schema '%s'" % schema) or ""
                    missing_str = ", ".join(missing)
                    raise exc.InvalidRequestError(
                        f"Could not reflect: requested table(s) not available "
                        f"in {bind.engine!r}{s}: ({missing_str})"
                    )
                load = [
                    name
                    for name in only
                    if extend_existing or name not in current
                ]
            # pass the available tables so the inspector can
            # choose to ignore the filter_names
            _reflect_info = insp._get_reflection_info(
                schema=schema,
                filter_names=load,
                available=available,
                kind=kind,
                scope=util.preloaded.engine_reflection.ObjectScope.ANY,
                **dialect_kwargs,
            )
            reflect_opts["_reflect_info"] = _reflect_info

            for name in load:
                try:
                    Table(name, self, **reflect_opts)
                except exc.UnreflectableTableError as uerr:
                    util.warn(f"Skipping table {name}: {uerr}")

    def create_all(
        self,
        bind: _CreateDropBind,
        tables: Optional[_typing_Sequence[Table]] = None,
        checkfirst: bool = True,
    ) -> None:
        """Create all tables stored in this metadata.

        Conditional by default, will not attempt to recreate tables already
        present in the target database.

        :param bind:
          A :class:`.Connection` or :class:`.Engine` used to access the
          database.

        :param tables:
          Optional list of ``Table`` objects, which is a subset of the total
          tables in the ``MetaData`` (others are ignored).

        :param checkfirst:
          Defaults to True, don't issue CREATEs for tables already present
          in the target database.

        """
        bind._run_ddl_visitor(
            ddl.SchemaGenerator, self, checkfirst=checkfirst, tables=tables
        )

    def drop_all(
        self,
        bind: _CreateDropBind,
        tables: Optional[_typing_Sequence[Table]] = None,
        checkfirst: bool = True,
    ) -> None:
        """Drop all tables stored in this metadata.

        Conditional by default, will not attempt to drop tables not present in
        the target database.

        :param bind:
          A :class:`.Connection` or :class:`.Engine` used to access the
          database.

        :param tables:
          Optional list of ``Table`` objects, which is a subset of the
          total tables in the ``MetaData`` (others are ignored).

        :param checkfirst:
          Defaults to True, only issue DROPs for tables confirmed to be
          present in the target database.

        """
        bind._run_ddl_visitor(
            ddl.SchemaDropper, self, checkfirst=checkfirst, tables=tables
        )


class Computed(FetchedValue, SchemaItem):
    """Defines a generated column, i.e. "GENERATED ALWAYS AS" syntax.

    The :class:`.Computed` construct is an inline construct added to the
    argument list of a :class:`_schema.Column` object::

        from sqlalchemy import Computed

        Table(
            "square",
            metadata_obj,
            Column("side", Float, nullable=False),
            Column("area", Float, Computed("side * side")),
        )

    See the linked documentation below for complete details.

    .. versionadded:: 1.3.11

    .. seealso::

        :ref:`computed_ddl`

    """

    __visit_name__ = "computed_column"

    column: Optional[Column[Any]]

    @_document_text_coercion(
        "sqltext", ":class:`.Computed`", ":paramref:`.Computed.sqltext`"
    )
    def __init__(
        self, sqltext: _DDLColumnArgument, persisted: Optional[bool] = None
    ) -> None:
        """Construct a GENERATED ALWAYS AS DDL construct to accompany a
        :class:`_schema.Column`.

        :param sqltext:
          A string containing the column generation expression, which will be
          used verbatim, or a SQL expression construct, such as a
          :func:`_expression.text`
          object.   If given as a string, the object is converted to a
          :func:`_expression.text` object.

        :param persisted:
          Optional, controls how this column should be persisted by the
          database.   Possible values are:

          * ``None``, the default, it will use the default persistence
            defined by the database.
          * ``True``, will render ``GENERATED ALWAYS AS ... STORED``, or the
            equivalent for the target database if supported.
          * ``False``, will render ``GENERATED ALWAYS AS ... VIRTUAL``, or
            the equivalent for the target database if supported.

          Specifying ``True`` or ``False`` may raise an error when the DDL
          is emitted to the target database if the database does not support
          that persistence option.   Leaving this parameter at its default
          of ``None`` is guaranteed to succeed for all databases that support
          ``GENERATED ALWAYS AS``.

        """
        self.sqltext = coercions.expect(roles.DDLExpressionRole, sqltext)
        self.persisted = persisted
        self.column = None

    def _set_parent(self, parent: SchemaEventTarget, **kw: Any) -> None:
        assert isinstance(parent, Column)

        if not isinstance(
            parent.server_default, (type(None), Computed)
        ) or not isinstance(parent.server_onupdate, (type(None), Computed)):
            raise exc.ArgumentError(
                "A generated column cannot specify a server_default or a "
                "server_onupdate argument"
            )
        self.column = parent
        parent.computed = self
        self.column.server_onupdate = self
        self.column.server_default = self

    def _as_for_update(self, for_update: bool) -> FetchedValue:
        return self

    @util.deprecated(
        "1.4",
        "The :meth:`_schema.Computed.copy` method is deprecated "
        "and will be removed in a future release.",
    )
    def copy(
        self, *, target_table: Optional[Table] = None, **kw: Any
    ) -> Computed:
        return self._copy(target_table=target_table, **kw)

    def _copy(
        self, *, target_table: Optional[Table] = None, **kw: Any
    ) -> Computed:
        sqltext = _copy_expression(
            self.sqltext,
            self.column.table if self.column is not None else None,
            target_table,
        )
        g = Computed(sqltext, persisted=self.persisted)

        return self._schema_item_copy(g)


class Identity(IdentityOptions, FetchedValue, SchemaItem):
    """Defines an identity column, i.e. "GENERATED { ALWAYS | BY DEFAULT }
    AS IDENTITY" syntax.

    The :class:`.Identity` construct is an inline construct added to the
    argument list of a :class:`_schema.Column` object::

        from sqlalchemy import Identity

        Table(
            "foo",
            metadata_obj,
            Column("id", Integer, Identity()),
            Column("description", Text),
        )

    See the linked documentation below for complete details.

    .. versionadded:: 1.4

    .. seealso::

        :ref:`identity_ddl`

    """

    __visit_name__ = "identity_column"

    is_identity = True

    def __init__(
        self,
        always: bool = False,
        on_null: Optional[bool] = None,
        start: Optional[int] = None,
        increment: Optional[int] = None,
        minvalue: Optional[int] = None,
        maxvalue: Optional[int] = None,
        nominvalue: Optional[bool] = None,
        nomaxvalue: Optional[bool] = None,
        cycle: Optional[bool] = None,
        cache: Optional[int] = None,
        order: Optional[bool] = None,
    ) -> None:
        """Construct a GENERATED { ALWAYS | BY DEFAULT } AS IDENTITY DDL
        construct to accompany a :class:`_schema.Column`.

        See the :class:`.Sequence` documentation for a complete description
        of most parameters.

        .. note::
            MSSQL supports this construct as the preferred alternative to
            generate an IDENTITY on a column, but it uses non standard
            syntax that only support :paramref:`_schema.Identity.start`
            and :paramref:`_schema.Identity.increment`.
            All other parameters are ignored.

        :param always:
          A boolean, that indicates the type of identity column.
          If ``False`` is specified, the default, then the user-specified
          value takes precedence.
          If ``True`` is specified, a user-specified value is not accepted (
          on some backends, like PostgreSQL, OVERRIDING SYSTEM VALUE, or
          similar, may be specified in an INSERT to override the sequence
          value).
          Some backends also have a default value for this parameter,
          ``None`` can be used to omit rendering this part in the DDL. It
          will be treated as ``False`` if a backend does not have a default
          value.

        :param on_null:
          Set to ``True`` to specify ON NULL in conjunction with a
          ``always=False`` identity column. This option is only supported on
          some backends, like Oracle Database.

        :param start: the starting index of the sequence.
        :param increment: the increment value of the sequence.
        :param minvalue: the minimum value of the sequence.
        :param maxvalue: the maximum value of the sequence.
        :param nominvalue: no minimum value of the sequence.
        :param nomaxvalue: no maximum value of the sequence.
        :param cycle: allows the sequence to wrap around when the maxvalue
         or minvalue has been reached.
        :param cache: optional integer value; number of future values in the
         sequence which are calculated in advance.
        :param order: optional boolean value; if true, renders the
         ORDER keyword.

        """
        IdentityOptions.__init__(
            self,
            start=start,
            increment=increment,
            minvalue=minvalue,
            maxvalue=maxvalue,
            nominvalue=nominvalue,
            nomaxvalue=nomaxvalue,
            cycle=cycle,
            cache=cache,
            order=order,
        )
        self.always = always
        self.on_null = on_null
        self.column = None

    def _set_parent(self, parent: SchemaEventTarget, **kw: Any) -> None:
        assert isinstance(parent, Column)
        if not isinstance(
            parent.server_default, (type(None), Identity)
        ) or not isinstance(parent.server_onupdate, type(None)):
            raise exc.ArgumentError(
                "A column with an Identity object cannot specify a "
                "server_default or a server_onupdate argument"
            )
        if parent.autoincrement is False:
            raise exc.ArgumentError(
                "A column with an Identity object cannot specify "
                "autoincrement=False"
            )
        self.column = parent

        parent.identity = self
        if parent._user_defined_nullable is NULL_UNSPECIFIED:
            parent.nullable = False

        parent.server_default = self

    def _as_for_update(self, for_update: bool) -> FetchedValue:
        return self

    @util.deprecated(
        "1.4",
        "The :meth:`_schema.Identity.copy` method is deprecated "
        "and will be removed in a future release.",
    )
    def copy(self, **kw: Any) -> Identity:
        return self._copy(**kw)

    def _copy(self, **kw: Any) -> Identity:
        i = Identity(
            always=self.always,
            on_null=self.on_null,
            start=self.start,
            increment=self.increment,
            minvalue=self.minvalue,
            maxvalue=self.maxvalue,
            nominvalue=self.nominvalue,
            nomaxvalue=self.nomaxvalue,
            cycle=self.cycle,
            cache=self.cache,
            order=self.order,
        )

        return self._schema_item_copy(i)