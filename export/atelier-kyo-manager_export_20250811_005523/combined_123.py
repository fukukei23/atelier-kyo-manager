
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\_mixins.py ===
from __future__ import annotations

from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    overload,
)

import numpy as np

from pandas._libs import lib
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import is_supported_dtype
from pandas._typing import (
    ArrayLike,
    AxisInt,
    Dtype,
    F,
    FillnaOptions,
    PositionalIndexer2D,
    PositionalIndexerTuple,
    ScalarIndexer,
    Self,
    SequenceIndexer,
    Shape,
    TakeIndexer,
    npt,
)
from pandas.errors import AbstractMethodError
from pandas.util._decorators import doc
from pandas.util._validators import (
    validate_bool_kwarg,
    validate_fillna_kwargs,
    validate_insert_loc,
)

from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    ExtensionDtype,
    PeriodDtype,
)
from pandas.core.dtypes.missing import array_equivalent

from pandas.core import missing
from pandas.core.algorithms import (
    take,
    unique,
    value_counts_internal as value_counts,
)
from pandas.core.array_algos.quantile import quantile_with_mask
from pandas.core.array_algos.transforms import shift
from pandas.core.arrays.base import ExtensionArray
from pandas.core.construction import extract_array
from pandas.core.indexers import check_array_indexer
from pandas.core.sorting import nargminmax

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pandas._typing import (
        NumpySorter,
        NumpyValueArrayLike,
    )

    from pandas import Series


def ravel_compat(meth: F) -> F:
    """
    Decorator to ravel a 2D array before passing it to a cython operation,
    then reshape the result to our own shape.
    """

    @wraps(meth)
    def method(self, *args, **kwargs):
        if self.ndim == 1:
            return meth(self, *args, **kwargs)

        flags = self._ndarray.flags
        flat = self.ravel("K")
        result = meth(flat, *args, **kwargs)
        order = "F" if flags.f_contiguous else "C"
        return result.reshape(self.shape, order=order)

    return cast(F, method)


class NDArrayBackedExtensionArray(NDArrayBacked, ExtensionArray):
    """
    ExtensionArray that is backed by a single NumPy ndarray.
    """

    _ndarray: np.ndarray

    # scalar used to denote NA value inside our self._ndarray, e.g. -1
    #  for Categorical, iNaT for Period. Outside of object dtype,
    #  self.isna() should be exactly locations in self._ndarray with
    #  _internal_fill_value.
    _internal_fill_value: Any

    def _box_func(self, x):
        """
        Wrap numpy type in our dtype.type if necessary.
        """
        return x

    def _validate_scalar(self, value):
        # used by NDArrayBackedExtensionIndex.insert
        raise AbstractMethodError(self)

    # ------------------------------------------------------------------------

    def view(self, dtype: Dtype | None = None) -> ArrayLike:
        # We handle datetime64, datetime64tz, timedelta64, and period
        #  dtypes here. Everything else we pass through to the underlying
        #  ndarray.
        if dtype is None or dtype is self.dtype:
            return self._from_backing_data(self._ndarray)

        if isinstance(dtype, type):
            # we sometimes pass non-dtype objects, e.g np.ndarray;
            #  pass those through to the underlying ndarray
            return self._ndarray.view(dtype)

        dtype = pandas_dtype(dtype)
        arr = self._ndarray

        if isinstance(dtype, PeriodDtype):
            cls = dtype.construct_array_type()
            return cls(arr.view("i8"), dtype=dtype)
        elif isinstance(dtype, DatetimeTZDtype):
            dt_cls = dtype.construct_array_type()
            dt64_values = arr.view(f"M8[{dtype.unit}]")
            return dt_cls._simple_new(dt64_values, dtype=dtype)
        elif lib.is_np_dtype(dtype, "M") and is_supported_dtype(dtype):
            from pandas.core.arrays import DatetimeArray

            dt64_values = arr.view(dtype)
            return DatetimeArray._simple_new(dt64_values, dtype=dtype)

        elif lib.is_np_dtype(dtype, "m") and is_supported_dtype(dtype):
            from pandas.core.arrays import TimedeltaArray

            td64_values = arr.view(dtype)
            return TimedeltaArray._simple_new(td64_values, dtype=dtype)

        # error: Argument "dtype" to "view" of "_ArrayOrScalarCommon" has incompatible
        # type "Union[ExtensionDtype, dtype[Any]]"; expected "Union[dtype[Any], None,
        # type, _SupportsDType, str, Union[Tuple[Any, int], Tuple[Any, Union[int,
        # Sequence[int]]], List[Any], _DTypeDict, Tuple[Any, Any]]]"
        return arr.view(dtype=dtype)  # type: ignore[arg-type]

    def take(
        self,
        indices: TakeIndexer,
        *,
        allow_fill: bool = False,
        fill_value: Any = None,
        axis: AxisInt = 0,
    ) -> Self:
        if allow_fill:
            fill_value = self._validate_scalar(fill_value)

        new_data = take(
            self._ndarray,
            indices,
            allow_fill=allow_fill,
            fill_value=fill_value,
            axis=axis,
        )
        return self._from_backing_data(new_data)

    # ------------------------------------------------------------------------

    def equals(self, other) -> bool:
        if type(self) is not type(other):
            return False
        if self.dtype != other.dtype:
            return False
        return bool(array_equivalent(self._ndarray, other._ndarray, dtype_equal=True))

    @classmethod
    def _from_factorized(cls, values, original):
        assert values.dtype == original._ndarray.dtype
        return original._from_backing_data(values)

    def _values_for_argsort(self) -> np.ndarray:
        return self._ndarray

    def _values_for_factorize(self):
        return self._ndarray, self._internal_fill_value

    def _hash_pandas_object(
        self, *, encoding: str, hash_key: str, categorize: bool
    ) -> npt.NDArray[np.uint64]:
        from pandas.core.util.hashing import hash_array

        values = self._ndarray
        return hash_array(
            values, encoding=encoding, hash_key=hash_key, categorize=categorize
        )

    # Signature of "argmin" incompatible with supertype "ExtensionArray"
    def argmin(self, axis: AxisInt = 0, skipna: bool = True):  # type: ignore[override]
        # override base class by adding axis keyword
        validate_bool_kwarg(skipna, "skipna")
        if not skipna and self._hasna:
            raise NotImplementedError
        return nargminmax(self, "argmin", axis=axis)

    # Signature of "argmax" incompatible with supertype "ExtensionArray"
    def argmax(self, axis: AxisInt = 0, skipna: bool = True):  # type: ignore[override]
        # override base class by adding axis keyword
        validate_bool_kwarg(skipna, "skipna")
        if not skipna and self._hasna:
            raise NotImplementedError
        return nargminmax(self, "argmax", axis=axis)

    def unique(self) -> Self:
        new_data = unique(self._ndarray)
        return self._from_backing_data(new_data)

    @classmethod
    @doc(ExtensionArray._concat_same_type)
    def _concat_same_type(
        cls,
        to_concat: Sequence[Self],
        axis: AxisInt = 0,
    ) -> Self:
        if not lib.dtypes_all_equal([x.dtype for x in to_concat]):
            dtypes = {str(x.dtype) for x in to_concat}
            raise ValueError("to_concat must have the same dtype", dtypes)

        return super()._concat_same_type(to_concat, axis=axis)

    @doc(ExtensionArray.searchsorted)
    def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        npvalue = self._validate_setitem_value(value)
        return self._ndarray.searchsorted(npvalue, side=side, sorter=sorter)

    @doc(ExtensionArray.shift)
    def shift(self, periods: int = 1, fill_value=None):
        # NB: shift is always along axis=0
        axis = 0
        fill_value = self._validate_scalar(fill_value)
        new_values = shift(self._ndarray, periods, axis, fill_value)

        return self._from_backing_data(new_values)

    def __setitem__(self, key, value) -> None:
        key = check_array_indexer(self, key)
        value = self._validate_setitem_value(value)
        self._ndarray[key] = value

    def _validate_setitem_value(self, value):
        return value

    @overload
    def __getitem__(self, key: ScalarIndexer) -> Any:
        ...

    @overload
    def __getitem__(
        self,
        key: SequenceIndexer | PositionalIndexerTuple,
    ) -> Self:
        ...

    def __getitem__(
        self,
        key: PositionalIndexer2D,
    ) -> Self | Any:
        if lib.is_integer(key):
            # fast-path
            result = self._ndarray[key]
            if self.ndim == 1:
                return self._box_func(result)
            return self._from_backing_data(result)

        # error: Incompatible types in assignment (expression has type "ExtensionArray",
        # variable has type "Union[int, slice, ndarray]")
        key = extract_array(key, extract_numpy=True)  # type: ignore[assignment]
        key = check_array_indexer(self, key)
        result = self._ndarray[key]
        if lib.is_scalar(result):
            return self._box_func(result)

        result = self._from_backing_data(result)
        return result

    def _fill_mask_inplace(
        self, method: str, limit: int | None, mask: npt.NDArray[np.bool_]
    ) -> None:
        # (for now) when self.ndim == 2, we assume axis=0
        func = missing.get_fill_func(method, ndim=self.ndim)
        func(self._ndarray.T, limit=limit, mask=mask.T)

    def _pad_or_backfill(
        self,
        *,
        method: FillnaOptions,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        copy: bool = True,
    ) -> Self:
        mask = self.isna()
        if mask.any():
            # (for now) when self.ndim == 2, we assume axis=0
            func = missing.get_fill_func(method, ndim=self.ndim)

            npvalues = self._ndarray.T
            if copy:
                npvalues = npvalues.copy()
            func(npvalues, limit=limit, limit_area=limit_area, mask=mask.T)
            npvalues = npvalues.T

            if copy:
                new_values = self._from_backing_data(npvalues)
            else:
                new_values = self

        else:
            if copy:
                new_values = self.copy()
            else:
                new_values = self
        return new_values

    @doc(ExtensionArray.fillna)
    def fillna(
        self, value=None, method=None, limit: int | None = None, copy: bool = True
    ) -> Self:
        value, method = validate_fillna_kwargs(
            value, method, validate_scalar_dict_value=False
        )

        mask = self.isna()
        # error: Argument 2 to "check_value_size" has incompatible type
        # "ExtensionArray"; expected "ndarray"
        value = missing.check_value_size(
            value, mask, len(self)  # type: ignore[arg-type]
        )

        if mask.any():
            if method is not None:
                # (for now) when self.ndim == 2, we assume axis=0
                func = missing.get_fill_func(method, ndim=self.ndim)
                npvalues = self._ndarray.T
                if copy:
                    npvalues = npvalues.copy()
                func(npvalues, limit=limit, mask=mask.T)
                npvalues = npvalues.T

                # TODO: NumpyExtensionArray didn't used to copy, need tests
                #  for this
                new_values = self._from_backing_data(npvalues)
            else:
                # fill with value
                if copy:
                    new_values = self.copy()
                else:
                    new_values = self[:]
                new_values[mask] = value
        else:
            # We validate the fill_value even if there is nothing to fill
            if value is not None:
                self._validate_setitem_value(value)

            if not copy:
                new_values = self[:]
            else:
                new_values = self.copy()
        return new_values

    # ------------------------------------------------------------------------
    # Reductions

    def _wrap_reduction_result(self, axis: AxisInt | None, result):
        if axis is None or self.ndim == 1:
            return self._box_func(result)
        return self._from_backing_data(result)

    # ------------------------------------------------------------------------
    # __array_function__ methods

    def _putmask(self, mask: npt.NDArray[np.bool_], value) -> None:
        """
        Analogue to np.putmask(self, mask, value)

        Parameters
        ----------
        mask : np.ndarray[bool]
        value : scalar or listlike

        Raises
        ------
        TypeError
            If value cannot be cast to self.dtype.
        """
        value = self._validate_setitem_value(value)

        np.putmask(self._ndarray, mask, value)

    def _where(self: Self, mask: npt.NDArray[np.bool_], value) -> Self:
        """
        Analogue to np.where(mask, self, value)

        Parameters
        ----------
        mask : np.ndarray[bool]
        value : scalar or listlike

        Raises
        ------
        TypeError
            If value cannot be cast to self.dtype.
        """
        value = self._validate_setitem_value(value)

        res_values = np.where(mask, self._ndarray, value)
        if res_values.dtype != self._ndarray.dtype:
            raise AssertionError(
                # GH#56410
                "Something has gone wrong, please report a bug at "
                "github.com/pandas-dev/pandas/"
            )
        return self._from_backing_data(res_values)

    # ------------------------------------------------------------------------
    # Index compat methods

    def insert(self, loc: int, item) -> Self:
        """
        Make new ExtensionArray inserting new item at location. Follows
        Python list.append semantics for negative values.

        Parameters
        ----------
        loc : int
        item : object

        Returns
        -------
        type(self)
        """
        loc = validate_insert_loc(loc, len(self))

        code = self._validate_scalar(item)

        new_vals = np.concatenate(
            (
                self._ndarray[:loc],
                np.asarray([code], dtype=self._ndarray.dtype),
                self._ndarray[loc:],
            )
        )
        return self._from_backing_data(new_vals)

    # ------------------------------------------------------------------------
    # Additional array methods
    #  These are not part of the EA API, but we implement them because
    #  pandas assumes they're there.

    def value_counts(self, dropna: bool = True) -> Series:
        """
        Return a Series containing counts of unique values.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NA values.

        Returns
        -------
        Series
        """
        if self.ndim != 1:
            raise NotImplementedError

        from pandas import (
            Index,
            Series,
        )

        if dropna:
            # error: Unsupported operand type for ~ ("ExtensionArray")
            values = self[~self.isna()]._ndarray  # type: ignore[operator]
        else:
            values = self._ndarray

        result = value_counts(values, sort=False, dropna=dropna)

        index_arr = self._from_backing_data(np.asarray(result.index._data))
        index = Index(index_arr, name=result.index.name)
        return Series(result._values, index=index, name=result.name, copy=False)

    def _quantile(
        self,
        qs: npt.NDArray[np.float64],
        interpolation: str,
    ) -> Self:
        # TODO: disable for Categorical if not ordered?

        mask = np.asarray(self.isna())
        arr = self._ndarray
        fill_value = self._internal_fill_value

        res_values = quantile_with_mask(arr, mask, fill_value, qs, interpolation)
        if res_values.dtype == self._ndarray.dtype:
            return self._from_backing_data(res_values)
        else:
            # e.g. test_quantile_empty we are empty integer dtype and res_values
            #  has floating dtype
            # TODO: technically __init__ isn't defined here.
            #  Should we raise NotImplementedError and handle this on NumpyEA?
            return type(self)(res_values)  # type: ignore[call-arg]

    # ------------------------------------------------------------------------
    # numpy-like methods

    @classmethod
    def _empty(cls, shape: Shape, dtype: ExtensionDtype) -> Self:
        """
        Analogous to np.empty(shape, dtype=dtype)

        Parameters
        ----------
        shape : tuple[int]
        dtype : ExtensionDtype
        """
        # The base implementation uses a naive approach to find the dtype
        #  for the backing ndarray
        arr = cls._from_sequence([], dtype=dtype)
        backing = np.empty(shape, dtype=arr._ndarray.dtype)
        return arr._from_backing_data(backing)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\json\_normalize.py ===
# ---------------------------------------------------------------------
# JSON normalization routines
from __future__ import annotations

from collections import (
    abc,
    defaultdict,
)
import copy
from typing import (
    TYPE_CHECKING,
    Any,
    DefaultDict,
)

import numpy as np

from pandas._libs.writers import convert_json_to_lines

import pandas as pd
from pandas import DataFrame

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pandas._typing import (
        IgnoreRaise,
        Scalar,
    )


def convert_to_line_delimits(s: str) -> str:
    """
    Helper function that converts JSON lists to line delimited JSON.
    """
    # Determine we have a JSON list to turn to lines otherwise just return the
    # json object, only lists can
    if not s[0] == "[" and s[-1] == "]":
        return s
    s = s[1:-1]

    return convert_json_to_lines(s)


def nested_to_record(
    ds,
    prefix: str = "",
    sep: str = ".",
    level: int = 0,
    max_level: int | None = None,
):
    """
    A simplified json_normalize

    Converts a nested dict into a flat dict ("record"), unlike json_normalize,
    it does not attempt to extract a subset of the data.

    Parameters
    ----------
    ds : dict or list of dicts
    prefix: the prefix, optional, default: ""
    sep : str, default '.'
        Nested records will generate names separated by sep,
        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar
    level: int, optional, default: 0
        The number of levels in the json string.

    max_level: int, optional, default: None
        The max depth to normalize.

    Returns
    -------
    d - dict or list of dicts, matching `ds`

    Examples
    --------
    >>> nested_to_record(
    ...     dict(flat1=1, dict1=dict(c=1, d=2), nested=dict(e=dict(c=1, d=2), d=2))
    ... )
    {\
'flat1': 1, \
'dict1.c': 1, \
'dict1.d': 2, \
'nested.e.c': 1, \
'nested.e.d': 2, \
'nested.d': 2\
}
    """
    singleton = False
    if isinstance(ds, dict):
        ds = [ds]
        singleton = True
    new_ds = []
    for d in ds:
        new_d = copy.deepcopy(d)
        for k, v in d.items():
            # each key gets renamed with prefix
            if not isinstance(k, str):
                k = str(k)
            if level == 0:
                newkey = k
            else:
                newkey = prefix + sep + k

            # flatten if type is dict and
            # current dict level  < maximum level provided and
            # only dicts gets recurse-flattened
            # only at level>1 do we rename the rest of the keys
            if not isinstance(v, dict) or (
                max_level is not None and level >= max_level
            ):
                if level != 0:  # so we skip copying for top level, common case
                    v = new_d.pop(k)
                    new_d[newkey] = v
                continue

            v = new_d.pop(k)
            new_d.update(nested_to_record(v, newkey, sep, level + 1, max_level))
        new_ds.append(new_d)

    if singleton:
        return new_ds[0]
    return new_ds


def _normalise_json(
    data: Any,
    key_string: str,
    normalized_dict: dict[str, Any],
    separator: str,
) -> dict[str, Any]:
    """
    Main recursive function
    Designed for the most basic use case of pd.json_normalize(data)
    intended as a performance improvement, see #15621

    Parameters
    ----------
    data : Any
        Type dependent on types contained within nested Json
    key_string : str
        New key (with separator(s) in) for data
    normalized_dict : dict
        The new normalized/flattened Json dict
    separator : str, default '.'
        Nested records will generate names separated by sep,
        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar
    """
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{key_string}{separator}{key}"

            if not key_string:
                new_key = new_key.removeprefix(separator)

            _normalise_json(
                data=value,
                key_string=new_key,
                normalized_dict=normalized_dict,
                separator=separator,
            )
    else:
        normalized_dict[key_string] = data
    return normalized_dict


def _normalise_json_ordered(data: dict[str, Any], separator: str) -> dict[str, Any]:
    """
    Order the top level keys and then recursively go to depth

    Parameters
    ----------
    data : dict or list of dicts
    separator : str, default '.'
        Nested records will generate names separated by sep,
        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar

    Returns
    -------
    dict or list of dicts, matching `normalised_json_object`
    """
    top_dict_ = {k: v for k, v in data.items() if not isinstance(v, dict)}
    nested_dict_ = _normalise_json(
        data={k: v for k, v in data.items() if isinstance(v, dict)},
        key_string="",
        normalized_dict={},
        separator=separator,
    )
    return {**top_dict_, **nested_dict_}


def _simple_json_normalize(
    ds: dict | list[dict],
    sep: str = ".",
) -> dict | list[dict] | Any:
    """
    A optimized basic json_normalize

    Converts a nested dict into a flat dict ("record"), unlike
    json_normalize and nested_to_record it doesn't do anything clever.
    But for the most basic use cases it enhances performance.
    E.g. pd.json_normalize(data)

    Parameters
    ----------
    ds : dict or list of dicts
    sep : str, default '.'
        Nested records will generate names separated by sep,
        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar

    Returns
    -------
    frame : DataFrame
    d - dict or list of dicts, matching `normalised_json_object`

    Examples
    --------
    >>> _simple_json_normalize(
    ...     {
    ...         "flat1": 1,
    ...         "dict1": {"c": 1, "d": 2},
    ...         "nested": {"e": {"c": 1, "d": 2}, "d": 2},
    ...     }
    ... )
    {\
'flat1': 1, \
'dict1.c': 1, \
'dict1.d': 2, \
'nested.e.c': 1, \
'nested.e.d': 2, \
'nested.d': 2\
}

    """
    normalised_json_object = {}
    # expect a dictionary, as most jsons are. However, lists are perfectly valid
    if isinstance(ds, dict):
        normalised_json_object = _normalise_json_ordered(data=ds, separator=sep)
    elif isinstance(ds, list):
        normalised_json_list = [_simple_json_normalize(row, sep=sep) for row in ds]
        return normalised_json_list
    return normalised_json_object


def json_normalize(
    data: dict | list[dict],
    record_path: str | list | None = None,
    meta: str | list[str | list[str]] | None = None,
    meta_prefix: str | None = None,
    record_prefix: str | None = None,
    errors: IgnoreRaise = "raise",
    sep: str = ".",
    max_level: int | None = None,
) -> DataFrame:
    """
    Normalize semi-structured JSON data into a flat table.

    Parameters
    ----------
    data : dict or list of dicts
        Unserialized JSON objects.
    record_path : str or list of str, default None
        Path in each object to list of records. If not passed, data will be
        assumed to be an array of records.
    meta : list of paths (str or list of str), default None
        Fields to use as metadata for each record in resulting table.
    meta_prefix : str, default None
        If True, prefix records with dotted (?) path, e.g. foo.bar.field if
        meta is ['foo', 'bar'].
    record_prefix : str, default None
        If True, prefix records with dotted (?) path, e.g. foo.bar.field if
        path to records is ['foo', 'bar'].
    errors : {'raise', 'ignore'}, default 'raise'
        Configures error handling.

        * 'ignore' : will ignore KeyError if keys listed in meta are not
          always present.
        * 'raise' : will raise KeyError if keys listed in meta are not
          always present.
    sep : str, default '.'
        Nested records will generate names separated by sep.
        e.g., for sep='.', {'foo': {'bar': 0}} -> foo.bar.
    max_level : int, default None
        Max number of levels(depth of dict) to normalize.
        if None, normalizes all levels.

    Returns
    -------
    frame : DataFrame
    Normalize semi-structured JSON data into a flat table.

    Examples
    --------
    >>> data = [
    ...     {"id": 1, "name": {"first": "Coleen", "last": "Volk"}},
    ...     {"name": {"given": "Mark", "family": "Regner"}},
    ...     {"id": 2, "name": "Faye Raker"},
    ... ]
    >>> pd.json_normalize(data)
        id name.first name.last name.given name.family        name
    0  1.0     Coleen      Volk        NaN         NaN         NaN
    1  NaN        NaN       NaN       Mark      Regner         NaN
    2  2.0        NaN       NaN        NaN         NaN  Faye Raker

    >>> data = [
    ...     {
    ...         "id": 1,
    ...         "name": "Cole Volk",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    ...     {"name": "Mark Reg", "fitness": {"height": 130, "weight": 60}},
    ...     {
    ...         "id": 2,
    ...         "name": "Faye Raker",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    ... ]
    >>> pd.json_normalize(data, max_level=0)
        id        name                        fitness
    0  1.0   Cole Volk  {'height': 130, 'weight': 60}
    1  NaN    Mark Reg  {'height': 130, 'weight': 60}
    2  2.0  Faye Raker  {'height': 130, 'weight': 60}

    Normalizes nested data up to level 1.

    >>> data = [
    ...     {
    ...         "id": 1,
    ...         "name": "Cole Volk",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    ...     {"name": "Mark Reg", "fitness": {"height": 130, "weight": 60}},
    ...     {
    ...         "id": 2,
    ...         "name": "Faye Raker",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    ... ]
    >>> pd.json_normalize(data, max_level=1)
        id        name  fitness.height  fitness.weight
    0  1.0   Cole Volk             130              60
    1  NaN    Mark Reg             130              60
    2  2.0  Faye Raker             130              60

    >>> data = [
    ...     {
    ...         "state": "Florida",
    ...         "shortname": "FL",
    ...         "info": {"governor": "Rick Scott"},
    ...         "counties": [
    ...             {"name": "Dade", "population": 12345},
    ...             {"name": "Broward", "population": 40000},
    ...             {"name": "Palm Beach", "population": 60000},
    ...         ],
    ...     },
    ...     {
    ...         "state": "Ohio",
    ...         "shortname": "OH",
    ...         "info": {"governor": "John Kasich"},
    ...         "counties": [
    ...             {"name": "Summit", "population": 1234},
    ...             {"name": "Cuyahoga", "population": 1337},
    ...         ],
    ...     },
    ... ]
    >>> result = pd.json_normalize(
    ...     data, "counties", ["state", "shortname", ["info", "governor"]]
    ... )
    >>> result
             name  population    state shortname info.governor
    0        Dade       12345   Florida    FL    Rick Scott
    1     Broward       40000   Florida    FL    Rick Scott
    2  Palm Beach       60000   Florida    FL    Rick Scott
    3      Summit        1234   Ohio       OH    John Kasich
    4    Cuyahoga        1337   Ohio       OH    John Kasich

    >>> data = {"A": [1, 2]}
    >>> pd.json_normalize(data, "A", record_prefix="Prefix.")
        Prefix.0
    0          1
    1          2

    Returns normalized data with columns prefixed with the given string.
    """

    def _pull_field(
        js: dict[str, Any], spec: list | str, extract_record: bool = False
    ) -> Scalar | Iterable:
        """Internal function to pull field"""
        result = js
        try:
            if isinstance(spec, list):
                for field in spec:
                    if result is None:
                        raise KeyError(field)
                    result = result[field]
            else:
                result = result[spec]
        except KeyError as e:
            if extract_record:
                raise KeyError(
                    f"Key {e} not found. If specifying a record_path, all elements of "
                    f"data should have the path."
                ) from e
            if errors == "ignore":
                return np.nan
            else:
                raise KeyError(
                    f"Key {e} not found. To replace missing values of {e} with "
                    f"np.nan, pass in errors='ignore'"
                ) from e

        return result

    def _pull_records(js: dict[str, Any], spec: list | str) -> list:
        """
        Internal function to pull field for records, and similar to
        _pull_field, but require to return list. And will raise error
        if has non iterable value.
        """
        result = _pull_field(js, spec, extract_record=True)

        # GH 31507 GH 30145, GH 26284 if result is not list, raise TypeError if not
        # null, otherwise return an empty list
        if not isinstance(result, list):
            if pd.isnull(result):
                result = []
            else:
                raise TypeError(
                    f"{js} has non list value {result} for path {spec}. "
                    "Must be list or null."
                )
        return result

    if isinstance(data, list) and not data:
        return DataFrame()
    elif isinstance(data, dict):
        # A bit of a hackjob
        data = [data]
    elif isinstance(data, abc.Iterable) and not isinstance(data, str):
        # GH35923 Fix pd.json_normalize to not skip the first element of a
        # generator input
        data = list(data)
    else:
        raise NotImplementedError

    # check to see if a simple recursive function is possible to
    # improve performance (see #15621) but only for cases such
    # as pd.Dataframe(data) or pd.Dataframe(data, sep)
    if (
        record_path is None
        and meta is None
        and meta_prefix is None
        and record_prefix is None
        and max_level is None
    ):
        return DataFrame(_simple_json_normalize(data, sep=sep))

    if record_path is None:
        if any([isinstance(x, dict) for x in y.values()] for y in data):
            # naive normalization, this is idempotent for flat records
            # and potentially will inflate the data considerably for
            # deeply nested structures:
            #  {VeryLong: { b: 1,c:2}} -> {VeryLong.b:1 ,VeryLong.c:@}
            #
            # TODO: handle record value which are lists, at least error
            #       reasonably
            data = nested_to_record(data, sep=sep, max_level=max_level)
        return DataFrame(data)
    elif not isinstance(record_path, list):
        record_path = [record_path]

    if meta is None:
        meta = []
    elif not isinstance(meta, list):
        meta = [meta]

    _meta = [m if isinstance(m, list) else [m] for m in meta]

    # Disastrously inefficient for now
    records: list = []
    lengths = []

    meta_vals: DefaultDict = defaultdict(list)
    meta_keys = [sep.join(val) for val in _meta]

    def _recursive_extract(data, path, seen_meta, level: int = 0) -> None:
        if isinstance(data, dict):
            data = [data]
        if len(path) > 1:
            for obj in data:
                for val, key in zip(_meta, meta_keys):
                    if level + 1 == len(val):
                        seen_meta[key] = _pull_field(obj, val[-1])

                _recursive_extract(obj[path[0]], path[1:], seen_meta, level=level + 1)
        else:
            for obj in data:
                recs = _pull_records(obj, path[0])
                recs = [
                    nested_to_record(r, sep=sep, max_level=max_level)
                    if isinstance(r, dict)
                    else r
                    for r in recs
                ]

                # For repeating the metadata later
                lengths.append(len(recs))
                for val, key in zip(_meta, meta_keys):
                    if level + 1 > len(val):
                        meta_val = seen_meta[key]
                    else:
                        meta_val = _pull_field(obj, val[level:])
                    meta_vals[key].append(meta_val)
                records.extend(recs)

    _recursive_extract(data, record_path, {}, level=0)

    result = DataFrame(records)

    if record_prefix is not None:
        result = result.rename(columns=lambda x: f"{record_prefix}{x}")

    # Data types, a problem
    for k, v in meta_vals.items():
        if meta_prefix is not None:
            k = meta_prefix + k

        if k in result:
            raise ValueError(
                f"Conflicting metadata name {k}, need distinguishing prefix "
            )
        # GH 37782

        values = np.array(v, dtype=object)

        if values.ndim > 1:
            # GH 37782
            values = np.empty((len(v),), dtype=object)
            for i, v in enumerate(v):
                values[i] = v

        result[k] = values.repeat(lengths)
    return result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\whisper\convert_to_onnx.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import argparse
import logging
import os

import torch
from benchmark_helper import Precision, create_onnxruntime_session, prepare_environment, setup_logger
from whisper_chain import chain_model
from whisper_encoder import WhisperEncoder
from whisper_helper import PRETRAINED_WHISPER_MODELS, WhisperHelper

from onnxruntime import quantization

logger = logging.getLogger("")

PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "rocm": "ROCMExecutionProvider",
}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()

    conversion_args = parser.add_argument_group("Conversion Process Args")
    optional_inputs = parser.add_argument_group("Optional Inputs (for WhisperBeamSearch op)")
    optional_outputs = parser.add_argument_group("Optional Outputs (for WhisperBeamSearch op)")
    quant_args = parser.add_argument_group("INT8 Quantization Args")

    #################################
    # Conversion options for Whisper
    #################################

    conversion_args.add_argument(
        "-m",
        "--model_name_or_path",
        required=False,
        default=PRETRAINED_WHISPER_MODELS[0],
        type=str,
        help="Model path, or pretrained model name in the list: " + ", ".join(PRETRAINED_WHISPER_MODELS),
    )

    conversion_args.add_argument(
        "--model_impl",
        required=False,
        default="hf",
        choices=["hf", "openai"],
        type=str,
        help="Select implementation for export of encoder and decoder subgraphs",
    )

    conversion_args.add_argument(
        "--cache_dir",
        required=False,
        type=str,
        default=os.path.join(".", "cache_models"),
        help="Directory to cache pre-trained models",
    )

    conversion_args.add_argument(
        "--output",
        required=False,
        type=str,
        default=os.path.join(".", "onnx_models"),
        help="Output directory",
    )

    conversion_args.add_argument(
        "-o",
        "--optimize_onnx",
        required=False,
        action="store_true",
        help="Use optimizer.py to optimize onnx model",
    )
    conversion_args.set_defaults(optimize_onnx=False)

    conversion_args.add_argument(
        "--use_gpu",
        required=False,
        action="store_true",
        help="Use GPU for model inference",
    )
    conversion_args.set_defaults(use_gpu=False)

    conversion_args.add_argument(
        "-p",
        "--precision",
        required=False,
        type=Precision,
        default=Precision.FLOAT32,
        choices=[Precision.FLOAT32, Precision.FLOAT16, Precision.INT8],
        help="Precision of model to run. fp32 for full precision, fp16 for half precision, int8 for quantization",
    )

    conversion_args.add_argument(
        "--use_int64_inputs",
        required=False,
        action="store_true",
        help="Use int64 instead of int32 for input_ids and attention_mask.",
    )
    conversion_args.set_defaults(use_int64_inputs=False)

    conversion_args.add_argument(
        "-r",
        "--provider",
        required=False,
        type=str,
        default="cpu",
        choices=list(PROVIDERS.keys()),
        help="Provider to benchmark. Default is CPUExecutionProvider.",
    )

    conversion_args.add_argument(
        "--verbose",
        required=False,
        action="store_true",
        help="Enable verbose logging",
    )
    conversion_args.set_defaults(verbose=False)

    conversion_args.add_argument(
        "-e",
        "--use_external_data_format",
        required=False,
        action="store_true",
        help="Save weights in external file. Necessary for 'small', 'medium', and 'large' models. Optional for 'tiny' and 'base' models.",
    )
    conversion_args.set_defaults(use_external_data_format=False)

    conversion_args.add_argument(
        "-w",
        "--overwrite",
        required=False,
        action="store_true",
        help="Overwrite existing ONNX model",
    )
    conversion_args.set_defaults(overwrite=False)

    conversion_args.add_argument(
        "--separate_encoder_and_decoder_init",
        required=False,
        action="store_true",
        help="Do not merge encoder and decoder init to initialize past KV caches. Output 3 instead of 2 ONNX models.",
    )
    conversion_args.set_defaults(separate_encoder_and_decoder_init=False)

    conversion_args.add_argument(
        "--no_beam_search_op",
        required=False,
        action="store_true",
        help="Do not produce model with WhisperBeamSearch op, which chains encdecinit and decoder models into one op.",
    )
    conversion_args.set_defaults(no_beam_search_op=False)

    #############################################################
    # Optional inputs for Whisper
    # (listed below in the order that WhisperBeamSearch expects)
    #############################################################

    optional_inputs.add_argument(
        "-v",
        "--use_vocab_mask",
        required=False,
        action="store_true",
        help="Use vocab_mask as an extra graph input to enable specific logits processing",
    )
    optional_inputs.set_defaults(use_vocab_mask=False)

    optional_inputs.add_argument(
        "-u",
        "--use_prefix_vocab_mask",
        required=False,
        action="store_true",
        help="Use prefix_vocab_mask as an extra graph input to enable specific logits processing",
    )
    optional_inputs.set_defaults(use_prefix_vocab_mask=False)

    optional_inputs.add_argument(
        "-f",
        "--use_forced_decoder_ids",
        required=False,
        action="store_true",
        help="Use decoder_input_ids as an extra graph input to the beam search op",
    )
    optional_inputs.set_defaults(use_forced_decoder_ids=False)

    optional_inputs.add_argument(
        "-l",
        "--use_logits_processor",
        required=False,
        action="store_true",
        help="Use logits_processor as an extra graph input to enable specific logits processing",
    )
    optional_inputs.set_defaults(use_specific_logits_processor=False)

    optional_inputs.add_argument(
        "--collect_cross_qk",
        required=False,
        action="store_true",
        help="Beam search model collect stacked cross QK.",
    )
    optional_inputs.set_defaults(collect_cross_qk=False)

    optional_inputs.add_argument(
        "--extra_decoding_ids",
        required=False,
        action="store_true",
        help="Need extra starting decoding ids for some feature like cross qk. Default if false.",
    )
    optional_inputs.set_defaults(extra_decoding_ids=False)

    optional_inputs.add_argument(
        "-t",
        "--use_temperature",
        required=False,
        action="store_true",
        help="Use temperature as an extra graph input for the WhisperBeamSearch op",
    )
    optional_inputs.set_defaults(use_temperature=False)

    optional_inputs.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=0,
        help="default to 0",
    )

    #############################################################
    # Optional outputs for Whisper
    # (listed below in the order that WhisperBeamSearch expects)
    #############################################################

    optional_outputs.add_argument(
        "--output_sequence_scores",
        required=False,
        action="store_true",
        help="Beam search model output scores for each generated sequence.",
    )
    optional_outputs.set_defaults(output_sequence_scores=False)

    optional_outputs.add_argument(
        "--output_scores",
        required=False,
        action="store_true",
        help="Beam search model output scores over vocab per generated token.",
    )
    optional_outputs.set_defaults(output_scores=False)

    optional_outputs.add_argument(
        "--output_cross_qk",
        required=False,
        action="store_true",
        help="Beam search model output collected qk as output. Also hint collect_cross_qk",
    )
    optional_outputs.set_defaults(output_cross_qk=False)

    optional_outputs.add_argument(
        "--cross_qk_onnx_model",
        required=False,
        type=str,
        default=None,
        help="The model which consumes cross_qk outputs.",
    )

    optional_outputs.add_argument(
        "--output_no_speech_probs",
        required=False,
        action="store_true",
        help="Beam search model output no speech probs which is computed from the encoder/context-decoder graph.",
    )
    optional_outputs.set_defaults(output_no_speech_probs=False)

    ###################################
    # Quantization options for Whisper
    ###################################

    quant_args.add_argument(
        "--quantize_embedding_layer",
        required=False,
        action="store_true",
        help="Quantize MatMul, GEMM, and Gather.",
    )
    quant_args.set_defaults(quantize_embedding_layer=False)

    quant_args.add_argument(
        "--quantize_per_channel",
        required=False,
        action="store_true",
        help="Quantize weights per each channel.",
    )
    quant_args.set_defaults(quantize_per_channel=False)

    quant_args.add_argument(
        "--quantize_reduce_range",
        required=False,
        action="store_true",
        help="Quantize weights with 7 bits.",
    )
    quant_args.set_defaults(quantize_reduce_range=False)

    args = parser.parse_args(argv)
    args.collect_cross_qk = args.collect_cross_qk or args.output_cross_qk

    return args


def export_onnx_models(
    model_name_or_path,
    model_impl,
    cache_dir,
    output_dir,
    use_gpu,
    use_external_data_format,
    optimize_onnx,
    precision,
    verbose,
    use_forced_decoder_ids: bool = False,
    merge_encoder_and_decoder_init: bool = True,
    no_beam_search_op: bool = False,
    output_qk: bool = False,
    overwrite: bool = False,
    use_int32_inputs: bool = True,
    quantize_embedding_layer: bool = False,
    quantize_per_channel: bool = False,
    quantize_reduce_range: bool = False,
    provider: str = "cpu",
):
    device = torch.device("cuda" if use_gpu else "cpu")

    models = WhisperHelper.load_model(
        model_name_or_path,
        model_impl,
        cache_dir,
        device,
        torch.float16 if precision == Precision.FLOAT16 else torch.float32,
        merge_encoder_and_decoder_init,
        no_beam_search_op,
        output_qk,
    )
    config = models["decoder"].config

    if (not use_external_data_format) and (config.num_hidden_layers > 24):
        logger.warning("You MUST pass `--use_external_data_format` because model size > 2GB")
        raise Exception("Please pass `--use_external_data_format` for this model.")

    output_paths = []
    for name, model in models.items():
        print(f"========> Handling {name} model......")
        filename_suffix = "_" + name

        onnx_path = WhisperHelper.get_onnx_path(
            output_dir,
            model_name_or_path,
            suffix=filename_suffix,
            new_folder=False,
        )

        # Export to ONNX
        if overwrite or not os.path.exists(onnx_path):
            logger.info(f"Exporting ONNX model to {onnx_path}")
            WhisperHelper.export_onnx(
                model,
                onnx_path,
                PROVIDERS[provider],
                verbose,
                use_external_data_format,
                use_fp16_inputs=(precision == Precision.FLOAT16),
                use_int32_inputs=use_int32_inputs,
                use_encoder_hidden_states=(name == "decoder_init"),
                use_kv_cache_inputs=(name == "decoder"),
            )
        else:
            logger.info(f"Skip exporting: existing ONNX model {onnx_path}")

        # Optimize ONNX model
        if optimize_onnx or precision != Precision.FLOAT32:
            output_path = WhisperHelper.get_onnx_path(
                output_dir,
                model_name_or_path,
                suffix=filename_suffix + "_" + str(precision),
                new_folder=False,
            )

            if overwrite or not os.path.exists(output_path):
                if optimize_onnx:
                    logger.info(f"Optimizing model to {output_path}")
                    WhisperHelper.optimize_onnx(
                        onnx_path,
                        output_path,
                        precision == Precision.FLOAT16,
                        model.config.encoder_attention_heads,
                        model.config.d_model,
                        model.config.num_hidden_layers,
                        use_external_data_format,
                        use_gpu=use_gpu,
                        provider=provider,
                        is_decoder=(name == "decoder"),
                        no_beam_search_op=no_beam_search_op,
                        output_qk=output_qk,
                    )
                    # Remove old ONNX model and old data file
                    if os.path.exists(onnx_path):
                        os.remove(onnx_path)
                    if os.path.exists(onnx_path + ".data"):
                        os.remove(onnx_path + ".data")
                    onnx_path = output_path

                    if isinstance(model, WhisperEncoder):
                        model.verify_onnx(
                            onnx_path,
                            PROVIDERS[provider],
                            use_fp16_inputs=(precision == Precision.FLOAT16),
                        )
                    else:
                        model.verify_onnx(
                            onnx_path,
                            PROVIDERS[provider],
                            use_fp16_inputs=(precision == Precision.FLOAT16),
                            use_int32_inputs=use_int32_inputs,
                        )

                if precision == Precision.INT8:
                    quantization.quantize_dynamic(
                        onnx_path,
                        output_path,
                        op_types_to_quantize=(
                            ["MatMul", "Gemm", "Gather"] if quantize_embedding_layer else ["MatMul", "Gemm"]
                        ),
                        use_external_data_format=use_external_data_format,
                        per_channel=quantize_per_channel,
                        reduce_range=quantize_reduce_range,
                        extra_options={"MatMulConstBOnly": True},
                    )
            else:
                logger.info(f"Skip optimizing: existing ONNX model {onnx_path}")
        else:
            output_path = onnx_path

        output_paths.append(output_path)

    return output_paths


def main(argv=None):
    args = parse_arguments(argv)

    setup_logger(args.verbose)

    logger.info(f"Arguments:{args}")

    cache_dir = args.cache_dir
    output_dir = args.output if not args.output.endswith(".onnx") else os.path.dirname(args.output)
    prepare_environment(cache_dir, output_dir, args.use_gpu)

    if args.precision == Precision.FLOAT16:
        assert args.use_gpu, "fp16 requires --use_gpu"

    output_paths = export_onnx_models(
        args.model_name_or_path,
        args.model_impl,
        cache_dir,
        output_dir,
        args.use_gpu,
        args.use_external_data_format,
        args.optimize_onnx,
        args.precision,
        args.verbose,
        args.use_forced_decoder_ids,
        not args.separate_encoder_and_decoder_init,
        args.no_beam_search_op,
        args.output_cross_qk,
        args.overwrite,
        not args.use_int64_inputs,
        args.quantize_embedding_layer,
        args.quantize_per_channel,
        args.quantize_reduce_range,
        args.provider,
    )

    max_diff = 0
    if not args.no_beam_search_op:
        logger.info("Chaining model ... :")
        args.beam_model_output_dir = WhisperHelper.get_onnx_path(
            output_dir,
            args.model_name_or_path,
            suffix="_beamsearch",
            new_folder=False,
        )
        for path in output_paths:
            if "encoder_decoder" in path or "encoder" in path:
                args.encoder_path = path
            elif "decoder" in path:
                args.decoder_path = path
        chain_model(args)
        output_paths.append(args.beam_model_output_dir)

        # Check chained model
        ort_session = create_onnxruntime_session(
            args.beam_model_output_dir,
            use_gpu=args.use_gpu,
            provider=args.provider,
        )
        device = torch.device("cuda" if args.use_gpu else "cpu")

        # Wrap parity check in try-except to allow export to continue in case this produces an error
        try:
            with torch.no_grad():
                # Verify batched decoding with prompts for OpenAI implementation
                if args.model_impl == "openai" and args.use_forced_decoder_ids:
                    max_diff = WhisperHelper.verify_onnx(
                        args.model_name_or_path, cache_dir, ort_session, device, batch_size=2, prompt_mode=True
                    )
                else:
                    max_diff = WhisperHelper.verify_onnx(args.model_name_or_path, cache_dir, ort_session, device)
            if max_diff > 1e-4:
                logger.warning("PyTorch and ONNX Runtime results are NOT close")
            else:
                logger.info("PyTorch and ONNX Runtime results are close")
        except Exception as e:
            logger.warning(
                f"An error occurred while trying to verify parity between PyTorch and ONNX Runtime: {e}", exc_info=True
            )

        # Remove extra ONNX models saved in output directory
        for _file in os.listdir(output_dir):
            if "_beamsearch" not in _file and "_jump_times" not in _file:
                path = os.path.join(output_dir, _file)
                os.remove(path)
                if path in output_paths:
                    output_paths.remove(path)

    logger.info(f"Done! Outputs: {output_paths}")
    return max_diff


if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\benchmarks\bench_solvers.py ===
from sympy.polys.rings import ring
from sympy.polys.fields import field
from sympy.polys.domains import ZZ, QQ
from sympy.polys.solvers import solve_lin_sys

# Expected times on 3.4 GHz i7:

# In [1]: %timeit time_solve_lin_sys_189x49()
# 1 loops, best of 3: 864 ms per loop
# In [2]: %timeit time_solve_lin_sys_165x165()
# 1 loops, best of 3: 1.83 s per loop
# In [3]: %timeit time_solve_lin_sys_10x8()
# 1 loops, best of 3: 2.31 s per loop

# Benchmark R_165: shows how fast are arithmetics in QQ.

R_165, uk_0, uk_1, uk_2, uk_3, uk_4, uk_5, uk_6, uk_7, uk_8, uk_9, uk_10, uk_11, uk_12, uk_13, uk_14, uk_15, uk_16, uk_17, uk_18, uk_19, uk_20, uk_21, uk_22, uk_23, uk_24, uk_25, uk_26, uk_27, uk_28, uk_29, uk_30, uk_31, uk_32, uk_33, uk_34, uk_35, uk_36, uk_37, uk_38, uk_39, uk_40, uk_41, uk_42, uk_43, uk_44, uk_45, uk_46, uk_47, uk_48, uk_49, uk_50, uk_51, uk_52, uk_53, uk_54, uk_55, uk_56, uk_57, uk_58, uk_59, uk_60, uk_61, uk_62, uk_63, uk_64, uk_65, uk_66, uk_67, uk_68, uk_69, uk_70, uk_71, uk_72, uk_73, uk_74, uk_75, uk_76, uk_77, uk_78, uk_79, uk_80, uk_81, uk_82, uk_83, uk_84, uk_85, uk_86, uk_87, uk_88, uk_89, uk_90, uk_91, uk_92, uk_93, uk_94, uk_95, uk_96, uk_97, uk_98, uk_99, uk_100, uk_101, uk_102, uk_103, uk_104, uk_105, uk_106, uk_107, uk_108, uk_109, uk_110, uk_111, uk_112, uk_113, uk_114, uk_115, uk_116, uk_117, uk_118, uk_119, uk_120, uk_121, uk_122, uk_123, uk_124, uk_125, uk_126, uk_127, uk_128, uk_129, uk_130, uk_131, uk_132, uk_133, uk_134, uk_135, uk_136, uk_137, uk_138, uk_139, uk_140, uk_141, uk_142, uk_143, uk_144, uk_145, uk_146, uk_147, uk_148, uk_149, uk_150, uk_151, uk_152, uk_153, uk_154, uk_155, uk_156, uk_157, uk_158, uk_159, uk_160, uk_161, uk_162, uk_163, uk_164 = ring("uk_:165", QQ)

def eqs_165x165():
    return [
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 411400*uk_100 + 1683000*uk_101 + 166375*uk_103 + 680625*uk_104 + 2784375*uk_106 + 729*uk_109 + 456471*uk_11 + 4131*uk_110 + 11016*uk_111 + 4455*uk_112 + 18225*uk_113 + 23409*uk_115 + 62424*uk_116 + 25245*uk_117 + 103275*uk_118 + 2586669*uk_12 + 166464*uk_120 + 67320*uk_121 + 275400*uk_122 + 27225*uk_124 + 111375*uk_125 + 455625*uk_127 + 6897784*uk_13 + 132651*uk_130 + 353736*uk_131 + 143055*uk_132 + 585225*uk_133 + 943296*uk_135 + 381480*uk_136 + 1560600*uk_137 + 154275*uk_139 + 2789545*uk_14 + 631125*uk_140 + 2581875*uk_142 + 2515456*uk_145 + 1017280*uk_146 + 4161600*uk_147 + 411400*uk_149 + 11411775*uk_15 + 1683000*uk_150 + 6885000*uk_152 + 166375*uk_155 + 680625*uk_156 + 2784375*uk_158 + 11390625*uk_161 + 3025*uk_17 + 495*uk_18 + 2805*uk_19 + 55*uk_2 + 7480*uk_20 + 3025*uk_21 + 12375*uk_22 + 81*uk_24 + 459*uk_25 + 1224*uk_26 + 495*uk_27 + 2025*uk_28 + 9*uk_3 + 2601*uk_30 + 6936*uk_31 + 2805*uk_32 + 11475*uk_33 + 18496*uk_35 + 7480*uk_36 + 30600*uk_37 + 3025*uk_39 + 51*uk_4 + 12375*uk_40 + 50625*uk_42 + 130470415844959*uk_45 + 141482932855*uk_46 + 23151752649*uk_47 + 131193265011*uk_48 + 349848706696*uk_49 + 136*uk_5 + 141482932855*uk_50 + 578793816225*uk_51 + 153424975*uk_53 + 25105905*uk_54 + 142266795*uk_55 + 379378120*uk_56 + 153424975*uk_57 + 627647625*uk_58 + 55*uk_6 + 4108239*uk_60 + 23280021*uk_61 + 62080056*uk_62 + 25105905*uk_63 + 102705975*uk_64 + 131920119*uk_66 + 351786984*uk_67 + 142266795*uk_68 + 582000525*uk_69 + 225*uk_7 + 938098624*uk_71 + 379378120*uk_72 + 1552001400*uk_73 + 153424975*uk_75 + 627647625*uk_76 + 2567649375*uk_78 + 166375*uk_81 + 27225*uk_82 + 154275*uk_83 + 411400*uk_84 + 166375*uk_85 + 680625*uk_86 + 4455*uk_88 + 25245*uk_89 + 2572416961*uk_9 + 67320*uk_90 + 27225*uk_91 + 111375*uk_92 + 143055*uk_94 + 381480*uk_95 + 154275*uk_96 + 631125*uk_97 + 1017280*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 413820*uk_100 + 1633500*uk_101 + 65340*uk_102 + 178695*uk_103 + 705375*uk_104 + 28215*uk_105 + 2784375*uk_106 + 111375*uk_107 + 4455*uk_108 + 97336*uk_109 + 2333074*uk_11 + 19044*uk_110 + 279312*uk_111 + 120612*uk_112 + 476100*uk_113 + 19044*uk_114 + 3726*uk_115 + 54648*uk_116 + 23598*uk_117 + 93150*uk_118 + 3726*uk_119 + 456471*uk_12 + 801504*uk_120 + 346104*uk_121 + 1366200*uk_122 + 54648*uk_123 + 149454*uk_124 + 589950*uk_125 + 23598*uk_126 + 2328750*uk_127 + 93150*uk_128 + 3726*uk_129 + 6694908*uk_13 + 729*uk_130 + 10692*uk_131 + 4617*uk_132 + 18225*uk_133 + 729*uk_134 + 156816*uk_135 + 67716*uk_136 + 267300*uk_137 + 10692*uk_138 + 29241*uk_139 + 2890983*uk_14 + 115425*uk_140 + 4617*uk_141 + 455625*uk_142 + 18225*uk_143 + 729*uk_144 + 2299968*uk_145 + 993168*uk_146 + 3920400*uk_147 + 156816*uk_148 + 428868*uk_149 + 11411775*uk_15 + 1692900*uk_150 + 67716*uk_151 + 6682500*uk_152 + 267300*uk_153 + 10692*uk_154 + 185193*uk_155 + 731025*uk_156 + 29241*uk_157 + 2885625*uk_158 + 115425*uk_159 + 456471*uk_16 + 4617*uk_160 + 11390625*uk_161 + 455625*uk_162 + 18225*uk_163 + 729*uk_164 + 3025*uk_17 + 2530*uk_18 + 495*uk_19 + 55*uk_2 + 7260*uk_20 + 3135*uk_21 + 12375*uk_22 + 495*uk_23 + 2116*uk_24 + 414*uk_25 + 6072*uk_26 + 2622*uk_27 + 10350*uk_28 + 414*uk_29 + 46*uk_3 + 81*uk_30 + 1188*uk_31 + 513*uk_32 + 2025*uk_33 + 81*uk_34 + 17424*uk_35 + 7524*uk_36 + 29700*uk_37 + 1188*uk_38 + 3249*uk_39 + 9*uk_4 + 12825*uk_40 + 513*uk_41 + 50625*uk_42 + 2025*uk_43 + 81*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 118331180206*uk_47 + 23151752649*uk_48 + 339559038852*uk_49 + 132*uk_5 + 146627766777*uk_50 + 578793816225*uk_51 + 23151752649*uk_52 + 153424975*uk_53 + 128319070*uk_54 + 25105905*uk_55 + 368219940*uk_56 + 159004065*uk_57 + 627647625*uk_58 + 25105905*uk_59 + 57*uk_6 + 107321404*uk_60 + 20997666*uk_61 + 307965768*uk_62 + 132985218*uk_63 + 524941650*uk_64 + 20997666*uk_65 + 4108239*uk_66 + 60254172*uk_67 + 26018847*uk_68 + 102705975*uk_69 + 225*uk_7 + 4108239*uk_70 + 883727856*uk_71 + 381609756*uk_72 + 1506354300*uk_73 + 60254172*uk_74 + 164786031*uk_75 + 650471175*uk_76 + 26018847*uk_77 + 2567649375*uk_78 + 102705975*uk_79 + 9*uk_8 + 4108239*uk_80 + 166375*uk_81 + 139150*uk_82 + 27225*uk_83 + 399300*uk_84 + 172425*uk_85 + 680625*uk_86 + 27225*uk_87 + 116380*uk_88 + 22770*uk_89 + 2572416961*uk_9 + 333960*uk_90 + 144210*uk_91 + 569250*uk_92 + 22770*uk_93 + 4455*uk_94 + 65340*uk_95 + 28215*uk_96 + 111375*uk_97 + 4455*uk_98 + 958320*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 402380*uk_100 + 1534500*uk_101 + 313720*uk_102 + 191455*uk_103 + 730125*uk_104 + 149270*uk_105 + 2784375*uk_106 + 569250*uk_107 + 116380*uk_108 + 912673*uk_109 + 4919743*uk_11 + 432814*uk_110 + 1166716*uk_111 + 555131*uk_112 + 2117025*uk_113 + 432814*uk_114 + 205252*uk_115 + 553288*uk_116 + 263258*uk_117 + 1003950*uk_118 + 205252*uk_119 + 2333074*uk_12 + 1491472*uk_120 + 709652*uk_121 + 2706300*uk_122 + 553288*uk_123 + 337657*uk_124 + 1287675*uk_125 + 263258*uk_126 + 4910625*uk_127 + 1003950*uk_128 + 205252*uk_129 + 6289156*uk_13 + 97336*uk_130 + 262384*uk_131 + 124844*uk_132 + 476100*uk_133 + 97336*uk_134 + 707296*uk_135 + 336536*uk_136 + 1283400*uk_137 + 262384*uk_138 + 160126*uk_139 + 2992421*uk_14 + 610650*uk_140 + 124844*uk_141 + 2328750*uk_142 + 476100*uk_143 + 97336*uk_144 + 1906624*uk_145 + 907184*uk_146 + 3459600*uk_147 + 707296*uk_148 + 431644*uk_149 + 11411775*uk_15 + 1646100*uk_150 + 336536*uk_151 + 6277500*uk_152 + 1283400*uk_153 + 262384*uk_154 + 205379*uk_155 + 783225*uk_156 + 160126*uk_157 + 2986875*uk_158 + 610650*uk_159 + 2333074*uk_16 + 124844*uk_160 + 11390625*uk_161 + 2328750*uk_162 + 476100*uk_163 + 97336*uk_164 + 3025*uk_17 + 5335*uk_18 + 2530*uk_19 + 55*uk_2 + 6820*uk_20 + 3245*uk_21 + 12375*uk_22 + 2530*uk_23 + 9409*uk_24 + 4462*uk_25 + 12028*uk_26 + 5723*uk_27 + 21825*uk_28 + 4462*uk_29 + 97*uk_3 + 2116*uk_30 + 5704*uk_31 + 2714*uk_32 + 10350*uk_33 + 2116*uk_34 + 15376*uk_35 + 7316*uk_36 + 27900*uk_37 + 5704*uk_38 + 3481*uk_39 + 46*uk_4 + 13275*uk_40 + 2714*uk_41 + 50625*uk_42 + 10350*uk_43 + 2116*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 249524445217*uk_47 + 118331180206*uk_48 + 318979703164*uk_49 + 124*uk_5 + 151772600699*uk_50 + 578793816225*uk_51 + 118331180206*uk_52 + 153424975*uk_53 + 270585865*uk_54 + 128319070*uk_55 + 345903580*uk_56 + 164583155*uk_57 + 627647625*uk_58 + 128319070*uk_59 + 59*uk_6 + 477215071*uk_60 + 226308178*uk_61 + 610048132*uk_62 + 290264837*uk_63 + 1106942175*uk_64 + 226308178*uk_65 + 107321404*uk_66 + 289301176*uk_67 + 137651366*uk_68 + 524941650*uk_69 + 225*uk_7 + 107321404*uk_70 + 779855344*uk_71 + 371060204*uk_72 + 1415060100*uk_73 + 289301176*uk_74 + 176552839*uk_75 + 673294725*uk_76 + 137651366*uk_77 + 2567649375*uk_78 + 524941650*uk_79 + 46*uk_8 + 107321404*uk_80 + 166375*uk_81 + 293425*uk_82 + 139150*uk_83 + 375100*uk_84 + 178475*uk_85 + 680625*uk_86 + 139150*uk_87 + 517495*uk_88 + 245410*uk_89 + 2572416961*uk_9 + 661540*uk_90 + 314765*uk_91 + 1200375*uk_92 + 245410*uk_93 + 116380*uk_94 + 313720*uk_95 + 149270*uk_96 + 569250*uk_97 + 116380*uk_98 + 845680*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 389180*uk_100 + 1435500*uk_101 + 618860*uk_102 + 204655*uk_103 + 754875*uk_104 + 325435*uk_105 + 2784375*uk_106 + 1200375*uk_107 + 517495*uk_108 + 3375000*uk_109 + 7607850*uk_11 + 2182500*uk_110 + 2610000*uk_111 + 1372500*uk_112 + 5062500*uk_113 + 2182500*uk_114 + 1411350*uk_115 + 1687800*uk_116 + 887550*uk_117 + 3273750*uk_118 + 1411350*uk_119 + 4919743*uk_12 + 2018400*uk_120 + 1061400*uk_121 + 3915000*uk_122 + 1687800*uk_123 + 558150*uk_124 + 2058750*uk_125 + 887550*uk_126 + 7593750*uk_127 + 3273750*uk_128 + 1411350*uk_129 + 5883404*uk_13 + 912673*uk_130 + 1091444*uk_131 + 573949*uk_132 + 2117025*uk_133 + 912673*uk_134 + 1305232*uk_135 + 686372*uk_136 + 2531700*uk_137 + 1091444*uk_138 + 360937*uk_139 + 3093859*uk_14 + 1331325*uk_140 + 573949*uk_141 + 4910625*uk_142 + 2117025*uk_143 + 912673*uk_144 + 1560896*uk_145 + 820816*uk_146 + 3027600*uk_147 + 1305232*uk_148 + 431636*uk_149 + 11411775*uk_15 + 1592100*uk_150 + 686372*uk_151 + 5872500*uk_152 + 2531700*uk_153 + 1091444*uk_154 + 226981*uk_155 + 837225*uk_156 + 360937*uk_157 + 3088125*uk_158 + 1331325*uk_159 + 4919743*uk_16 + 573949*uk_160 + 11390625*uk_161 + 4910625*uk_162 + 2117025*uk_163 + 912673*uk_164 + 3025*uk_17 + 8250*uk_18 + 5335*uk_19 + 55*uk_2 + 6380*uk_20 + 3355*uk_21 + 12375*uk_22 + 5335*uk_23 + 22500*uk_24 + 14550*uk_25 + 17400*uk_26 + 9150*uk_27 + 33750*uk_28 + 14550*uk_29 + 150*uk_3 + 9409*uk_30 + 11252*uk_31 + 5917*uk_32 + 21825*uk_33 + 9409*uk_34 + 13456*uk_35 + 7076*uk_36 + 26100*uk_37 + 11252*uk_38 + 3721*uk_39 + 97*uk_4 + 13725*uk_40 + 5917*uk_41 + 50625*uk_42 + 21825*uk_43 + 9409*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 385862544150*uk_47 + 249524445217*uk_48 + 298400367476*uk_49 + 116*uk_5 + 156917434621*uk_50 + 578793816225*uk_51 + 249524445217*uk_52 + 153424975*uk_53 + 418431750*uk_54 + 270585865*uk_55 + 323587220*uk_56 + 170162245*uk_57 + 627647625*uk_58 + 270585865*uk_59 + 61*uk_6 + 1141177500*uk_60 + 737961450*uk_61 + 882510600*uk_62 + 464078850*uk_63 + 1711766250*uk_64 + 737961450*uk_65 + 477215071*uk_66 + 570690188*uk_67 + 300104323*uk_68 + 1106942175*uk_69 + 225*uk_7 + 477215071*uk_70 + 682474864*uk_71 + 358887644*uk_72 + 1323765900*uk_73 + 570690188*uk_74 + 188725399*uk_75 + 696118275*uk_76 + 300104323*uk_77 + 2567649375*uk_78 + 1106942175*uk_79 + 97*uk_8 + 477215071*uk_80 + 166375*uk_81 + 453750*uk_82 + 293425*uk_83 + 350900*uk_84 + 184525*uk_85 + 680625*uk_86 + 293425*uk_87 + 1237500*uk_88 + 800250*uk_89 + 2572416961*uk_9 + 957000*uk_90 + 503250*uk_91 + 1856250*uk_92 + 800250*uk_93 + 517495*uk_94 + 618860*uk_95 + 325435*uk_96 + 1200375*uk_97 + 517495*uk_98 + 740080*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 374220*uk_100 + 1336500*uk_101 + 891000*uk_102 + 218295*uk_103 + 779625*uk_104 + 519750*uk_105 + 2784375*uk_106 + 1856250*uk_107 + 1237500*uk_108 + 7189057*uk_109 + 9788767*uk_11 + 5587350*uk_110 + 4022892*uk_111 + 2346687*uk_112 + 8381025*uk_113 + 5587350*uk_114 + 4342500*uk_115 + 3126600*uk_116 + 1823850*uk_117 + 6513750*uk_118 + 4342500*uk_119 + 7607850*uk_12 + 2251152*uk_120 + 1313172*uk_121 + 4689900*uk_122 + 3126600*uk_123 + 766017*uk_124 + 2735775*uk_125 + 1823850*uk_126 + 9770625*uk_127 + 6513750*uk_128 + 4342500*uk_129 + 5477652*uk_13 + 3375000*uk_130 + 2430000*uk_131 + 1417500*uk_132 + 5062500*uk_133 + 3375000*uk_134 + 1749600*uk_135 + 1020600*uk_136 + 3645000*uk_137 + 2430000*uk_138 + 595350*uk_139 + 3195297*uk_14 + 2126250*uk_140 + 1417500*uk_141 + 7593750*uk_142 + 5062500*uk_143 + 3375000*uk_144 + 1259712*uk_145 + 734832*uk_146 + 2624400*uk_147 + 1749600*uk_148 + 428652*uk_149 + 11411775*uk_15 + 1530900*uk_150 + 1020600*uk_151 + 5467500*uk_152 + 3645000*uk_153 + 2430000*uk_154 + 250047*uk_155 + 893025*uk_156 + 595350*uk_157 + 3189375*uk_158 + 2126250*uk_159 + 7607850*uk_16 + 1417500*uk_160 + 11390625*uk_161 + 7593750*uk_162 + 5062500*uk_163 + 3375000*uk_164 + 3025*uk_17 + 10615*uk_18 + 8250*uk_19 + 55*uk_2 + 5940*uk_20 + 3465*uk_21 + 12375*uk_22 + 8250*uk_23 + 37249*uk_24 + 28950*uk_25 + 20844*uk_26 + 12159*uk_27 + 43425*uk_28 + 28950*uk_29 + 193*uk_3 + 22500*uk_30 + 16200*uk_31 + 9450*uk_32 + 33750*uk_33 + 22500*uk_34 + 11664*uk_35 + 6804*uk_36 + 24300*uk_37 + 16200*uk_38 + 3969*uk_39 + 150*uk_4 + 14175*uk_40 + 9450*uk_41 + 50625*uk_42 + 33750*uk_43 + 22500*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 496476473473*uk_47 + 385862544150*uk_48 + 277821031788*uk_49 + 108*uk_5 + 162062268543*uk_50 + 578793816225*uk_51 + 385862544150*uk_52 + 153424975*uk_53 + 538382185*uk_54 + 418431750*uk_55 + 301270860*uk_56 + 175741335*uk_57 + 627647625*uk_58 + 418431750*uk_59 + 63*uk_6 + 1889232031*uk_60 + 1468315050*uk_61 + 1057186836*uk_62 + 616692321*uk_63 + 2202472575*uk_64 + 1468315050*uk_65 + 1141177500*uk_66 + 821647800*uk_67 + 479294550*uk_68 + 1711766250*uk_69 + 225*uk_7 + 1141177500*uk_70 + 591586416*uk_71 + 345092076*uk_72 + 1232471700*uk_73 + 821647800*uk_74 + 201303711*uk_75 + 718941825*uk_76 + 479294550*uk_77 + 2567649375*uk_78 + 1711766250*uk_79 + 150*uk_8 + 1141177500*uk_80 + 166375*uk_81 + 583825*uk_82 + 453750*uk_83 + 326700*uk_84 + 190575*uk_85 + 680625*uk_86 + 453750*uk_87 + 2048695*uk_88 + 1592250*uk_89 + 2572416961*uk_9 + 1146420*uk_90 + 668745*uk_91 + 2388375*uk_92 + 1592250*uk_93 + 1237500*uk_94 + 891000*uk_95 + 519750*uk_96 + 1856250*uk_97 + 1237500*uk_98 + 641520*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 357500*uk_100 + 1237500*uk_101 + 1061500*uk_102 + 232375*uk_103 + 804375*uk_104 + 689975*uk_105 + 2784375*uk_106 + 2388375*uk_107 + 2048695*uk_108 + 9800344*uk_109 + 10853866*uk_11 + 8838628*uk_110 + 4579600*uk_111 + 2976740*uk_112 + 10304100*uk_113 + 8838628*uk_114 + 7971286*uk_115 + 4130200*uk_116 + 2684630*uk_117 + 9292950*uk_118 + 7971286*uk_119 + 9788767*uk_12 + 2140000*uk_120 + 1391000*uk_121 + 4815000*uk_122 + 4130200*uk_123 + 904150*uk_124 + 3129750*uk_125 + 2684630*uk_126 + 10833750*uk_127 + 9292950*uk_128 + 7971286*uk_129 + 5071900*uk_13 + 7189057*uk_130 + 3724900*uk_131 + 2421185*uk_132 + 8381025*uk_133 + 7189057*uk_134 + 1930000*uk_135 + 1254500*uk_136 + 4342500*uk_137 + 3724900*uk_138 + 815425*uk_139 + 3296735*uk_14 + 2822625*uk_140 + 2421185*uk_141 + 9770625*uk_142 + 8381025*uk_143 + 7189057*uk_144 + 1000000*uk_145 + 650000*uk_146 + 2250000*uk_147 + 1930000*uk_148 + 422500*uk_149 + 11411775*uk_15 + 1462500*uk_150 + 1254500*uk_151 + 5062500*uk_152 + 4342500*uk_153 + 3724900*uk_154 + 274625*uk_155 + 950625*uk_156 + 815425*uk_157 + 3290625*uk_158 + 2822625*uk_159 + 9788767*uk_16 + 2421185*uk_160 + 11390625*uk_161 + 9770625*uk_162 + 8381025*uk_163 + 7189057*uk_164 + 3025*uk_17 + 11770*uk_18 + 10615*uk_19 + 55*uk_2 + 5500*uk_20 + 3575*uk_21 + 12375*uk_22 + 10615*uk_23 + 45796*uk_24 + 41302*uk_25 + 21400*uk_26 + 13910*uk_27 + 48150*uk_28 + 41302*uk_29 + 214*uk_3 + 37249*uk_30 + 19300*uk_31 + 12545*uk_32 + 43425*uk_33 + 37249*uk_34 + 10000*uk_35 + 6500*uk_36 + 22500*uk_37 + 19300*uk_38 + 4225*uk_39 + 193*uk_4 + 14625*uk_40 + 12545*uk_41 + 50625*uk_42 + 43425*uk_43 + 37249*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 550497229654*uk_47 + 496476473473*uk_48 + 257241696100*uk_49 + 100*uk_5 + 167207102465*uk_50 + 578793816225*uk_51 + 496476473473*uk_52 + 153424975*uk_53 + 596962630*uk_54 + 538382185*uk_55 + 278954500*uk_56 + 181320425*uk_57 + 627647625*uk_58 + 538382185*uk_59 + 65*uk_6 + 2322727324*uk_60 + 2094796138*uk_61 + 1085386600*uk_62 + 705501290*uk_63 + 2442119850*uk_64 + 2094796138*uk_65 + 1889232031*uk_66 + 978876700*uk_67 + 636269855*uk_68 + 2202472575*uk_69 + 225*uk_7 + 1889232031*uk_70 + 507190000*uk_71 + 329673500*uk_72 + 1141177500*uk_73 + 978876700*uk_74 + 214287775*uk_75 + 741765375*uk_76 + 636269855*uk_77 + 2567649375*uk_78 + 2202472575*uk_79 + 193*uk_8 + 1889232031*uk_80 + 166375*uk_81 + 647350*uk_82 + 583825*uk_83 + 302500*uk_84 + 196625*uk_85 + 680625*uk_86 + 583825*uk_87 + 2518780*uk_88 + 2271610*uk_89 + 2572416961*uk_9 + 1177000*uk_90 + 765050*uk_91 + 2648250*uk_92 + 2271610*uk_93 + 2048695*uk_94 + 1061500*uk_95 + 689975*uk_96 + 2388375*uk_97 + 2048695*uk_98 + 550000*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 339020*uk_100 + 1138500*uk_101 + 1082840*uk_102 + 246895*uk_103 + 829125*uk_104 + 788590*uk_105 + 2784375*uk_106 + 2648250*uk_107 + 2518780*uk_108 + 8120601*uk_109 + 10194519*uk_11 + 8645814*uk_110 + 3716892*uk_111 + 2706867*uk_112 + 9090225*uk_113 + 8645814*uk_114 + 9204996*uk_115 + 3957288*uk_116 + 2881938*uk_117 + 9678150*uk_118 + 9204996*uk_119 + 10853866*uk_12 + 1701264*uk_120 + 1238964*uk_121 + 4160700*uk_122 + 3957288*uk_123 + 902289*uk_124 + 3030075*uk_125 + 2881938*uk_126 + 10175625*uk_127 + 9678150*uk_128 + 9204996*uk_129 + 4666148*uk_13 + 9800344*uk_130 + 4213232*uk_131 + 3068332*uk_132 + 10304100*uk_133 + 9800344*uk_134 + 1811296*uk_135 + 1319096*uk_136 + 4429800*uk_137 + 4213232*uk_138 + 960646*uk_139 + 3398173*uk_14 + 3226050*uk_140 + 3068332*uk_141 + 10833750*uk_142 + 10304100*uk_143 + 9800344*uk_144 + 778688*uk_145 + 567088*uk_146 + 1904400*uk_147 + 1811296*uk_148 + 412988*uk_149 + 11411775*uk_15 + 1386900*uk_150 + 1319096*uk_151 + 4657500*uk_152 + 4429800*uk_153 + 4213232*uk_154 + 300763*uk_155 + 1010025*uk_156 + 960646*uk_157 + 3391875*uk_158 + 3226050*uk_159 + 10853866*uk_16 + 3068332*uk_160 + 11390625*uk_161 + 10833750*uk_162 + 10304100*uk_163 + 9800344*uk_164 + 3025*uk_17 + 11055*uk_18 + 11770*uk_19 + 55*uk_2 + 5060*uk_20 + 3685*uk_21 + 12375*uk_22 + 11770*uk_23 + 40401*uk_24 + 43014*uk_25 + 18492*uk_26 + 13467*uk_27 + 45225*uk_28 + 43014*uk_29 + 201*uk_3 + 45796*uk_30 + 19688*uk_31 + 14338*uk_32 + 48150*uk_33 + 45796*uk_34 + 8464*uk_35 + 6164*uk_36 + 20700*uk_37 + 19688*uk_38 + 4489*uk_39 + 214*uk_4 + 15075*uk_40 + 14338*uk_41 + 50625*uk_42 + 48150*uk_43 + 45796*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 517055809161*uk_47 + 550497229654*uk_48 + 236662360412*uk_49 + 92*uk_5 + 172351936387*uk_50 + 578793816225*uk_51 + 550497229654*uk_52 + 153424975*uk_53 + 560698545*uk_54 + 596962630*uk_55 + 256638140*uk_56 + 186899515*uk_57 + 627647625*uk_58 + 596962630*uk_59 + 67*uk_6 + 2049098319*uk_60 + 2181627066*uk_61 + 937895748*uk_62 + 683032773*uk_63 + 2293766775*uk_64 + 2181627066*uk_65 + 2322727324*uk_66 + 998555672*uk_67 + 727209022*uk_68 + 2442119850*uk_69 + 225*uk_7 + 2322727324*uk_70 + 429285616*uk_71 + 312631916*uk_72 + 1049883300*uk_73 + 998555672*uk_74 + 227677591*uk_75 + 764588925*uk_76 + 727209022*uk_77 + 2567649375*uk_78 + 2442119850*uk_79 + 214*uk_8 + 2322727324*uk_80 + 166375*uk_81 + 608025*uk_82 + 647350*uk_83 + 278300*uk_84 + 202675*uk_85 + 680625*uk_86 + 647350*uk_87 + 2222055*uk_88 + 2365770*uk_89 + 2572416961*uk_9 + 1017060*uk_90 + 740685*uk_91 + 2487375*uk_92 + 2365770*uk_93 + 2518780*uk_94 + 1082840*uk_95 + 788590*uk_96 + 2648250*uk_97 + 2518780*uk_98 + 465520*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 318780*uk_100 + 1039500*uk_101 + 928620*uk_102 + 261855*uk_103 + 853875*uk_104 + 762795*uk_105 + 2784375*uk_106 + 2487375*uk_107 + 2222055*uk_108 + 2863288*uk_109 + 7202098*uk_11 + 4052964*uk_110 + 1693776*uk_111 + 1391316*uk_112 + 4536900*uk_113 + 4052964*uk_114 + 5736942*uk_115 + 2397528*uk_116 + 1969398*uk_117 + 6421950*uk_118 + 5736942*uk_119 + 10194519*uk_12 + 1001952*uk_120 + 823032*uk_121 + 2683800*uk_122 + 2397528*uk_123 + 676062*uk_124 + 2204550*uk_125 + 1969398*uk_126 + 7188750*uk_127 + 6421950*uk_128 + 5736942*uk_129 + 4260396*uk_13 + 8120601*uk_130 + 3393684*uk_131 + 2787669*uk_132 + 9090225*uk_133 + 8120601*uk_134 + 1418256*uk_135 + 1164996*uk_136 + 3798900*uk_137 + 3393684*uk_138 + 956961*uk_139 + 3499611*uk_14 + 3120525*uk_140 + 2787669*uk_141 + 10175625*uk_142 + 9090225*uk_143 + 8120601*uk_144 + 592704*uk_145 + 486864*uk_146 + 1587600*uk_147 + 1418256*uk_148 + 399924*uk_149 + 11411775*uk_15 + 1304100*uk_150 + 1164996*uk_151 + 4252500*uk_152 + 3798900*uk_153 + 3393684*uk_154 + 328509*uk_155 + 1071225*uk_156 + 956961*uk_157 + 3493125*uk_158 + 3120525*uk_159 + 10194519*uk_16 + 2787669*uk_160 + 11390625*uk_161 + 10175625*uk_162 + 9090225*uk_163 + 8120601*uk_164 + 3025*uk_17 + 7810*uk_18 + 11055*uk_19 + 55*uk_2 + 4620*uk_20 + 3795*uk_21 + 12375*uk_22 + 11055*uk_23 + 20164*uk_24 + 28542*uk_25 + 11928*uk_26 + 9798*uk_27 + 31950*uk_28 + 28542*uk_29 + 142*uk_3 + 40401*uk_30 + 16884*uk_31 + 13869*uk_32 + 45225*uk_33 + 40401*uk_34 + 7056*uk_35 + 5796*uk_36 + 18900*uk_37 + 16884*uk_38 + 4761*uk_39 + 201*uk_4 + 15525*uk_40 + 13869*uk_41 + 50625*uk_42 + 45225*uk_43 + 40401*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 365283208462*uk_47 + 517055809161*uk_48 + 216083024724*uk_49 + 84*uk_5 + 177496770309*uk_50 + 578793816225*uk_51 + 517055809161*uk_52 + 153424975*uk_53 + 396115390*uk_54 + 560698545*uk_55 + 234321780*uk_56 + 192478605*uk_57 + 627647625*uk_58 + 560698545*uk_59 + 69*uk_6 + 1022697916*uk_60 + 1447621698*uk_61 + 604976232*uk_62 + 496944762*uk_63 + 1620472050*uk_64 + 1447621698*uk_65 + 2049098319*uk_66 + 856339596*uk_67 + 703421811*uk_68 + 2293766775*uk_69 + 225*uk_7 + 2049098319*uk_70 + 357873264*uk_71 + 293967324*uk_72 + 958589100*uk_73 + 856339596*uk_74 + 241473159*uk_75 + 787412475*uk_76 + 703421811*uk_77 + 2567649375*uk_78 + 2293766775*uk_79 + 201*uk_8 + 2049098319*uk_80 + 166375*uk_81 + 429550*uk_82 + 608025*uk_83 + 254100*uk_84 + 208725*uk_85 + 680625*uk_86 + 608025*uk_87 + 1109020*uk_88 + 1569810*uk_89 + 2572416961*uk_9 + 656040*uk_90 + 538890*uk_91 + 1757250*uk_92 + 1569810*uk_93 + 2222055*uk_94 + 928620*uk_95 + 762795*uk_96 + 2487375*uk_97 + 2222055*uk_98 + 388080*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 296780*uk_100 + 940500*uk_101 + 593560*uk_102 + 277255*uk_103 + 878625*uk_104 + 554510*uk_105 + 2784375*uk_106 + 1757250*uk_107 + 1109020*uk_108 + 15625*uk_109 + 1267975*uk_11 + 88750*uk_110 + 47500*uk_111 + 44375*uk_112 + 140625*uk_113 + 88750*uk_114 + 504100*uk_115 + 269800*uk_116 + 252050*uk_117 + 798750*uk_118 + 504100*uk_119 + 7202098*uk_12 + 144400*uk_120 + 134900*uk_121 + 427500*uk_122 + 269800*uk_123 + 126025*uk_124 + 399375*uk_125 + 252050*uk_126 + 1265625*uk_127 + 798750*uk_128 + 504100*uk_129 + 3854644*uk_13 + 2863288*uk_130 + 1532464*uk_131 + 1431644*uk_132 + 4536900*uk_133 + 2863288*uk_134 + 820192*uk_135 + 766232*uk_136 + 2428200*uk_137 + 1532464*uk_138 + 715822*uk_139 + 3601049*uk_14 + 2268450*uk_140 + 1431644*uk_141 + 7188750*uk_142 + 4536900*uk_143 + 2863288*uk_144 + 438976*uk_145 + 410096*uk_146 + 1299600*uk_147 + 820192*uk_148 + 383116*uk_149 + 11411775*uk_15 + 1214100*uk_150 + 766232*uk_151 + 3847500*uk_152 + 2428200*uk_153 + 1532464*uk_154 + 357911*uk_155 + 1134225*uk_156 + 715822*uk_157 + 3594375*uk_158 + 2268450*uk_159 + 7202098*uk_16 + 1431644*uk_160 + 11390625*uk_161 + 7188750*uk_162 + 4536900*uk_163 + 2863288*uk_164 + 3025*uk_17 + 1375*uk_18 + 7810*uk_19 + 55*uk_2 + 4180*uk_20 + 3905*uk_21 + 12375*uk_22 + 7810*uk_23 + 625*uk_24 + 3550*uk_25 + 1900*uk_26 + 1775*uk_27 + 5625*uk_28 + 3550*uk_29 + 25*uk_3 + 20164*uk_30 + 10792*uk_31 + 10082*uk_32 + 31950*uk_33 + 20164*uk_34 + 5776*uk_35 + 5396*uk_36 + 17100*uk_37 + 10792*uk_38 + 5041*uk_39 + 142*uk_4 + 15975*uk_40 + 10082*uk_41 + 50625*uk_42 + 31950*uk_43 + 20164*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 64310424025*uk_47 + 365283208462*uk_48 + 195503689036*uk_49 + 76*uk_5 + 182641604231*uk_50 + 578793816225*uk_51 + 365283208462*uk_52 + 153424975*uk_53 + 69738625*uk_54 + 396115390*uk_55 + 212005420*uk_56 + 198057695*uk_57 + 627647625*uk_58 + 396115390*uk_59 + 71*uk_6 + 31699375*uk_60 + 180052450*uk_61 + 96366100*uk_62 + 90026225*uk_63 + 285294375*uk_64 + 180052450*uk_65 + 1022697916*uk_66 + 547359448*uk_67 + 511348958*uk_68 + 1620472050*uk_69 + 225*uk_7 + 1022697916*uk_70 + 292952944*uk_71 + 273679724*uk_72 + 867294900*uk_73 + 547359448*uk_74 + 255674479*uk_75 + 810236025*uk_76 + 511348958*uk_77 + 2567649375*uk_78 + 1620472050*uk_79 + 142*uk_8 + 1022697916*uk_80 + 166375*uk_81 + 75625*uk_82 + 429550*uk_83 + 229900*uk_84 + 214775*uk_85 + 680625*uk_86 + 429550*uk_87 + 34375*uk_88 + 195250*uk_89 + 2572416961*uk_9 + 104500*uk_90 + 97625*uk_91 + 309375*uk_92 + 195250*uk_93 + 1109020*uk_94 + 593560*uk_95 + 554510*uk_96 + 1757250*uk_97 + 1109020*uk_98 + 317680*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 321200*uk_100 + 990000*uk_101 + 110000*uk_102 + 293095*uk_103 + 903375*uk_104 + 100375*uk_105 + 2784375*uk_106 + 309375*uk_107 + 34375*uk_108 + 185193*uk_109 + 2890983*uk_11 + 81225*uk_110 + 259920*uk_111 + 237177*uk_112 + 731025*uk_113 + 81225*uk_114 + 35625*uk_115 + 114000*uk_116 + 104025*uk_117 + 320625*uk_118 + 35625*uk_119 + 1267975*uk_12 + 364800*uk_120 + 332880*uk_121 + 1026000*uk_122 + 114000*uk_123 + 303753*uk_124 + 936225*uk_125 + 104025*uk_126 + 2885625*uk_127 + 320625*uk_128 + 35625*uk_129 + 4057520*uk_13 + 15625*uk_130 + 50000*uk_131 + 45625*uk_132 + 140625*uk_133 + 15625*uk_134 + 160000*uk_135 + 146000*uk_136 + 450000*uk_137 + 50000*uk_138 + 133225*uk_139 + 3702487*uk_14 + 410625*uk_140 + 45625*uk_141 + 1265625*uk_142 + 140625*uk_143 + 15625*uk_144 + 512000*uk_145 + 467200*uk_146 + 1440000*uk_147 + 160000*uk_148 + 426320*uk_149 + 11411775*uk_15 + 1314000*uk_150 + 146000*uk_151 + 4050000*uk_152 + 450000*uk_153 + 50000*uk_154 + 389017*uk_155 + 1199025*uk_156 + 133225*uk_157 + 3695625*uk_158 + 410625*uk_159 + 1267975*uk_16 + 45625*uk_160 + 11390625*uk_161 + 1265625*uk_162 + 140625*uk_163 + 15625*uk_164 + 3025*uk_17 + 3135*uk_18 + 1375*uk_19 + 55*uk_2 + 4400*uk_20 + 4015*uk_21 + 12375*uk_22 + 1375*uk_23 + 3249*uk_24 + 1425*uk_25 + 4560*uk_26 + 4161*uk_27 + 12825*uk_28 + 1425*uk_29 + 57*uk_3 + 625*uk_30 + 2000*uk_31 + 1825*uk_32 + 5625*uk_33 + 625*uk_34 + 6400*uk_35 + 5840*uk_36 + 18000*uk_37 + 2000*uk_38 + 5329*uk_39 + 25*uk_4 + 16425*uk_40 + 1825*uk_41 + 50625*uk_42 + 5625*uk_43 + 625*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 146627766777*uk_47 + 64310424025*uk_48 + 205793356880*uk_49 + 80*uk_5 + 187786438153*uk_50 + 578793816225*uk_51 + 64310424025*uk_52 + 153424975*uk_53 + 159004065*uk_54 + 69738625*uk_55 + 223163600*uk_56 + 203636785*uk_57 + 627647625*uk_58 + 69738625*uk_59 + 73*uk_6 + 164786031*uk_60 + 72274575*uk_61 + 231278640*uk_62 + 211041759*uk_63 + 650471175*uk_64 + 72274575*uk_65 + 31699375*uk_66 + 101438000*uk_67 + 92562175*uk_68 + 285294375*uk_69 + 225*uk_7 + 31699375*uk_70 + 324601600*uk_71 + 296198960*uk_72 + 912942000*uk_73 + 101438000*uk_74 + 270281551*uk_75 + 833059575*uk_76 + 92562175*uk_77 + 2567649375*uk_78 + 285294375*uk_79 + 25*uk_8 + 31699375*uk_80 + 166375*uk_81 + 172425*uk_82 + 75625*uk_83 + 242000*uk_84 + 220825*uk_85 + 680625*uk_86 + 75625*uk_87 + 178695*uk_88 + 78375*uk_89 + 2572416961*uk_9 + 250800*uk_90 + 228855*uk_91 + 705375*uk_92 + 78375*uk_93 + 34375*uk_94 + 110000*uk_95 + 100375*uk_96 + 309375*uk_97 + 34375*uk_98 + 352000*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 297000*uk_100 + 891000*uk_101 + 225720*uk_102 + 309375*uk_103 + 928125*uk_104 + 235125*uk_105 + 2784375*uk_106 + 705375*uk_107 + 178695*uk_108 + 6859*uk_109 + 963661*uk_11 + 20577*uk_110 + 25992*uk_111 + 27075*uk_112 + 81225*uk_113 + 20577*uk_114 + 61731*uk_115 + 77976*uk_116 + 81225*uk_117 + 243675*uk_118 + 61731*uk_119 + 2890983*uk_12 + 98496*uk_120 + 102600*uk_121 + 307800*uk_122 + 77976*uk_123 + 106875*uk_124 + 320625*uk_125 + 81225*uk_126 + 961875*uk_127 + 243675*uk_128 + 61731*uk_129 + 3651768*uk_13 + 185193*uk_130 + 233928*uk_131 + 243675*uk_132 + 731025*uk_133 + 185193*uk_134 + 295488*uk_135 + 307800*uk_136 + 923400*uk_137 + 233928*uk_138 + 320625*uk_139 + 3803925*uk_14 + 961875*uk_140 + 243675*uk_141 + 2885625*uk_142 + 731025*uk_143 + 185193*uk_144 + 373248*uk_145 + 388800*uk_146 + 1166400*uk_147 + 295488*uk_148 + 405000*uk_149 + 11411775*uk_15 + 1215000*uk_150 + 307800*uk_151 + 3645000*uk_152 + 923400*uk_153 + 233928*uk_154 + 421875*uk_155 + 1265625*uk_156 + 320625*uk_157 + 3796875*uk_158 + 961875*uk_159 + 2890983*uk_16 + 243675*uk_160 + 11390625*uk_161 + 2885625*uk_162 + 731025*uk_163 + 185193*uk_164 + 3025*uk_17 + 1045*uk_18 + 3135*uk_19 + 55*uk_2 + 3960*uk_20 + 4125*uk_21 + 12375*uk_22 + 3135*uk_23 + 361*uk_24 + 1083*uk_25 + 1368*uk_26 + 1425*uk_27 + 4275*uk_28 + 1083*uk_29 + 19*uk_3 + 3249*uk_30 + 4104*uk_31 + 4275*uk_32 + 12825*uk_33 + 3249*uk_34 + 5184*uk_35 + 5400*uk_36 + 16200*uk_37 + 4104*uk_38 + 5625*uk_39 + 57*uk_4 + 16875*uk_40 + 4275*uk_41 + 50625*uk_42 + 12825*uk_43 + 3249*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 48875922259*uk_47 + 146627766777*uk_48 + 185214021192*uk_49 + 72*uk_5 + 192931272075*uk_50 + 578793816225*uk_51 + 146627766777*uk_52 + 153424975*uk_53 + 53001355*uk_54 + 159004065*uk_55 + 200847240*uk_56 + 209215875*uk_57 + 627647625*uk_58 + 159004065*uk_59 + 75*uk_6 + 18309559*uk_60 + 54928677*uk_61 + 69383592*uk_62 + 72274575*uk_63 + 216823725*uk_64 + 54928677*uk_65 + 164786031*uk_66 + 208150776*uk_67 + 216823725*uk_68 + 650471175*uk_69 + 225*uk_7 + 164786031*uk_70 + 262927296*uk_71 + 273882600*uk_72 + 821647800*uk_73 + 208150776*uk_74 + 285294375*uk_75 + 855883125*uk_76 + 216823725*uk_77 + 2567649375*uk_78 + 650471175*uk_79 + 57*uk_8 + 164786031*uk_80 + 166375*uk_81 + 57475*uk_82 + 172425*uk_83 + 217800*uk_84 + 226875*uk_85 + 680625*uk_86 + 172425*uk_87 + 19855*uk_88 + 59565*uk_89 + 2572416961*uk_9 + 75240*uk_90 + 78375*uk_91 + 235125*uk_92 + 59565*uk_93 + 178695*uk_94 + 225720*uk_95 + 235125*uk_96 + 705375*uk_97 + 178695*uk_98 + 285120*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 304920*uk_100 + 891000*uk_101 + 75240*uk_102 + 326095*uk_103 + 952875*uk_104 + 80465*uk_105 + 2784375*uk_106 + 235125*uk_107 + 19855*uk_108 + 148877*uk_109 + 2688107*uk_11 + 53371*uk_110 + 202248*uk_111 + 216293*uk_112 + 632025*uk_113 + 53371*uk_114 + 19133*uk_115 + 72504*uk_116 + 77539*uk_117 + 226575*uk_118 + 19133*uk_119 + 963661*uk_12 + 274752*uk_120 + 293832*uk_121 + 858600*uk_122 + 72504*uk_123 + 314237*uk_124 + 918225*uk_125 + 77539*uk_126 + 2683125*uk_127 + 226575*uk_128 + 19133*uk_129 + 3651768*uk_13 + 6859*uk_130 + 25992*uk_131 + 27797*uk_132 + 81225*uk_133 + 6859*uk_134 + 98496*uk_135 + 105336*uk_136 + 307800*uk_137 + 25992*uk_138 + 112651*uk_139 + 3905363*uk_14 + 329175*uk_140 + 27797*uk_141 + 961875*uk_142 + 81225*uk_143 + 6859*uk_144 + 373248*uk_145 + 399168*uk_146 + 1166400*uk_147 + 98496*uk_148 + 426888*uk_149 + 11411775*uk_15 + 1247400*uk_150 + 105336*uk_151 + 3645000*uk_152 + 307800*uk_153 + 25992*uk_154 + 456533*uk_155 + 1334025*uk_156 + 112651*uk_157 + 3898125*uk_158 + 329175*uk_159 + 963661*uk_16 + 27797*uk_160 + 11390625*uk_161 + 961875*uk_162 + 81225*uk_163 + 6859*uk_164 + 3025*uk_17 + 2915*uk_18 + 1045*uk_19 + 55*uk_2 + 3960*uk_20 + 4235*uk_21 + 12375*uk_22 + 1045*uk_23 + 2809*uk_24 + 1007*uk_25 + 3816*uk_26 + 4081*uk_27 + 11925*uk_28 + 1007*uk_29 + 53*uk_3 + 361*uk_30 + 1368*uk_31 + 1463*uk_32 + 4275*uk_33 + 361*uk_34 + 5184*uk_35 + 5544*uk_36 + 16200*uk_37 + 1368*uk_38 + 5929*uk_39 + 19*uk_4 + 17325*uk_40 + 1463*uk_41 + 50625*uk_42 + 4275*uk_43 + 361*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 136338098933*uk_47 + 48875922259*uk_48 + 185214021192*uk_49 + 72*uk_5 + 198076105997*uk_50 + 578793816225*uk_51 + 48875922259*uk_52 + 153424975*uk_53 + 147845885*uk_54 + 53001355*uk_55 + 200847240*uk_56 + 214794965*uk_57 + 627647625*uk_58 + 53001355*uk_59 + 77*uk_6 + 142469671*uk_60 + 51074033*uk_61 + 193543704*uk_62 + 206984239*uk_63 + 604824075*uk_64 + 51074033*uk_65 + 18309559*uk_66 + 69383592*uk_67 + 74201897*uk_68 + 216823725*uk_69 + 225*uk_7 + 18309559*uk_70 + 262927296*uk_71 + 281186136*uk_72 + 821647800*uk_73 + 69383592*uk_74 + 300712951*uk_75 + 878706675*uk_76 + 74201897*uk_77 + 2567649375*uk_78 + 216823725*uk_79 + 19*uk_8 + 18309559*uk_80 + 166375*uk_81 + 160325*uk_82 + 57475*uk_83 + 217800*uk_84 + 232925*uk_85 + 680625*uk_86 + 57475*uk_87 + 154495*uk_88 + 55385*uk_89 + 2572416961*uk_9 + 209880*uk_90 + 224455*uk_91 + 655875*uk_92 + 55385*uk_93 + 19855*uk_94 + 75240*uk_95 + 80465*uk_96 + 235125*uk_97 + 19855*uk_98 + 285120*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 278080*uk_100 + 792000*uk_101 + 186560*uk_102 + 343255*uk_103 + 977625*uk_104 + 230285*uk_105 + 2784375*uk_106 + 655875*uk_107 + 154495*uk_108 + uk_109 + 50719*uk_11 + 53*uk_110 + 64*uk_111 + 79*uk_112 + 225*uk_113 + 53*uk_114 + 2809*uk_115 + 3392*uk_116 + 4187*uk_117 + 11925*uk_118 + 2809*uk_119 + 2688107*uk_12 + 4096*uk_120 + 5056*uk_121 + 14400*uk_122 + 3392*uk_123 + 6241*uk_124 + 17775*uk_125 + 4187*uk_126 + 50625*uk_127 + 11925*uk_128 + 2809*uk_129 + 3246016*uk_13 + 148877*uk_130 + 179776*uk_131 + 221911*uk_132 + 632025*uk_133 + 148877*uk_134 + 217088*uk_135 + 267968*uk_136 + 763200*uk_137 + 179776*uk_138 + 330773*uk_139 + 4006801*uk_14 + 942075*uk_140 + 221911*uk_141 + 2683125*uk_142 + 632025*uk_143 + 148877*uk_144 + 262144*uk_145 + 323584*uk_146 + 921600*uk_147 + 217088*uk_148 + 399424*uk_149 + 11411775*uk_15 + 1137600*uk_150 + 267968*uk_151 + 3240000*uk_152 + 763200*uk_153 + 179776*uk_154 + 493039*uk_155 + 1404225*uk_156 + 330773*uk_157 + 3999375*uk_158 + 942075*uk_159 + 2688107*uk_16 + 221911*uk_160 + 11390625*uk_161 + 2683125*uk_162 + 632025*uk_163 + 148877*uk_164 + 3025*uk_17 + 55*uk_18 + 2915*uk_19 + 55*uk_2 + 3520*uk_20 + 4345*uk_21 + 12375*uk_22 + 2915*uk_23 + uk_24 + 53*uk_25 + 64*uk_26 + 79*uk_27 + 225*uk_28 + 53*uk_29 + uk_3 + 2809*uk_30 + 3392*uk_31 + 4187*uk_32 + 11925*uk_33 + 2809*uk_34 + 4096*uk_35 + 5056*uk_36 + 14400*uk_37 + 3392*uk_38 + 6241*uk_39 + 53*uk_4 + 17775*uk_40 + 4187*uk_41 + 50625*uk_42 + 11925*uk_43 + 2809*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 2572416961*uk_47 + 136338098933*uk_48 + 164634685504*uk_49 + 64*uk_5 + 203220939919*uk_50 + 578793816225*uk_51 + 136338098933*uk_52 + 153424975*uk_53 + 2789545*uk_54 + 147845885*uk_55 + 178530880*uk_56 + 220374055*uk_57 + 627647625*uk_58 + 147845885*uk_59 + 79*uk_6 + 50719*uk_60 + 2688107*uk_61 + 3246016*uk_62 + 4006801*uk_63 + 11411775*uk_64 + 2688107*uk_65 + 142469671*uk_66 + 172038848*uk_67 + 212360453*uk_68 + 604824075*uk_69 + 225*uk_7 + 142469671*uk_70 + 207745024*uk_71 + 256435264*uk_72 + 730353600*uk_73 + 172038848*uk_74 + 316537279*uk_75 + 901530225*uk_76 + 212360453*uk_77 + 2567649375*uk_78 + 604824075*uk_79 + 53*uk_8 + 142469671*uk_80 + 166375*uk_81 + 3025*uk_82 + 160325*uk_83 + 193600*uk_84 + 238975*uk_85 + 680625*uk_86 + 160325*uk_87 + 55*uk_88 + 2915*uk_89 + 2572416961*uk_9 + 3520*uk_90 + 4345*uk_91 + 12375*uk_92 + 2915*uk_93 + 154495*uk_94 + 186560*uk_95 + 230285*uk_96 + 655875*uk_97 + 154495*uk_98 + 225280*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 285120*uk_100 + 792000*uk_101 + 3520*uk_102 + 360855*uk_103 + 1002375*uk_104 + 4455*uk_105 + 2784375*uk_106 + 12375*uk_107 + 55*uk_108 + 2197*uk_109 + 659347*uk_11 + 169*uk_110 + 10816*uk_111 + 13689*uk_112 + 38025*uk_113 + 169*uk_114 + 13*uk_115 + 832*uk_116 + 1053*uk_117 + 2925*uk_118 + 13*uk_119 + 50719*uk_12 + 53248*uk_120 + 67392*uk_121 + 187200*uk_122 + 832*uk_123 + 85293*uk_124 + 236925*uk_125 + 1053*uk_126 + 658125*uk_127 + 2925*uk_128 + 13*uk_129 + 3246016*uk_13 + uk_130 + 64*uk_131 + 81*uk_132 + 225*uk_133 + uk_134 + 4096*uk_135 + 5184*uk_136 + 14400*uk_137 + 64*uk_138 + 6561*uk_139 + 4108239*uk_14 + 18225*uk_140 + 81*uk_141 + 50625*uk_142 + 225*uk_143 + uk_144 + 262144*uk_145 + 331776*uk_146 + 921600*uk_147 + 4096*uk_148 + 419904*uk_149 + 11411775*uk_15 + 1166400*uk_150 + 5184*uk_151 + 3240000*uk_152 + 14400*uk_153 + 64*uk_154 + 531441*uk_155 + 1476225*uk_156 + 6561*uk_157 + 4100625*uk_158 + 18225*uk_159 + 50719*uk_16 + 81*uk_160 + 11390625*uk_161 + 50625*uk_162 + 225*uk_163 + uk_164 + 3025*uk_17 + 715*uk_18 + 55*uk_19 + 55*uk_2 + 3520*uk_20 + 4455*uk_21 + 12375*uk_22 + 55*uk_23 + 169*uk_24 + 13*uk_25 + 832*uk_26 + 1053*uk_27 + 2925*uk_28 + 13*uk_29 + 13*uk_3 + uk_30 + 64*uk_31 + 81*uk_32 + 225*uk_33 + uk_34 + 4096*uk_35 + 5184*uk_36 + 14400*uk_37 + 64*uk_38 + 6561*uk_39 + uk_4 + 18225*uk_40 + 81*uk_41 + 50625*uk_42 + 225*uk_43 + uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 33441420493*uk_47 + 2572416961*uk_48 + 164634685504*uk_49 + 64*uk_5 + 208365773841*uk_50 + 578793816225*uk_51 + 2572416961*uk_52 + 153424975*uk_53 + 36264085*uk_54 + 2789545*uk_55 + 178530880*uk_56 + 225953145*uk_57 + 627647625*uk_58 + 2789545*uk_59 + 81*uk_6 + 8571511*uk_60 + 659347*uk_61 + 42198208*uk_62 + 53407107*uk_63 + 148353075*uk_64 + 659347*uk_65 + 50719*uk_66 + 3246016*uk_67 + 4108239*uk_68 + 11411775*uk_69 + 225*uk_7 + 50719*uk_70 + 207745024*uk_71 + 262927296*uk_72 + 730353600*uk_73 + 3246016*uk_74 + 332767359*uk_75 + 924353775*uk_76 + 4108239*uk_77 + 2567649375*uk_78 + 11411775*uk_79 + uk_8 + 50719*uk_80 + 166375*uk_81 + 39325*uk_82 + 3025*uk_83 + 193600*uk_84 + 245025*uk_85 + 680625*uk_86 + 3025*uk_87 + 9295*uk_88 + 715*uk_89 + 2572416961*uk_9 + 45760*uk_90 + 57915*uk_91 + 160875*uk_92 + 715*uk_93 + 55*uk_94 + 3520*uk_95 + 4455*uk_96 + 12375*uk_97 + 55*uk_98 + 225280*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 273900*uk_100 + 742500*uk_101 + 42900*uk_102 + 378895*uk_103 + 1027125*uk_104 + 59345*uk_105 + 2784375*uk_106 + 160875*uk_107 + 9295*uk_108 + 216*uk_109 + 304314*uk_11 + 468*uk_110 + 2160*uk_111 + 2988*uk_112 + 8100*uk_113 + 468*uk_114 + 1014*uk_115 + 4680*uk_116 + 6474*uk_117 + 17550*uk_118 + 1014*uk_119 + 659347*uk_12 + 21600*uk_120 + 29880*uk_121 + 81000*uk_122 + 4680*uk_123 + 41334*uk_124 + 112050*uk_125 + 6474*uk_126 + 303750*uk_127 + 17550*uk_128 + 1014*uk_129 + 3043140*uk_13 + 2197*uk_130 + 10140*uk_131 + 14027*uk_132 + 38025*uk_133 + 2197*uk_134 + 46800*uk_135 + 64740*uk_136 + 175500*uk_137 + 10140*uk_138 + 89557*uk_139 + 4209677*uk_14 + 242775*uk_140 + 14027*uk_141 + 658125*uk_142 + 38025*uk_143 + 2197*uk_144 + 216000*uk_145 + 298800*uk_146 + 810000*uk_147 + 46800*uk_148 + 413340*uk_149 + 11411775*uk_15 + 1120500*uk_150 + 64740*uk_151 + 3037500*uk_152 + 175500*uk_153 + 10140*uk_154 + 571787*uk_155 + 1550025*uk_156 + 89557*uk_157 + 4201875*uk_158 + 242775*uk_159 + 659347*uk_16 + 14027*uk_160 + 11390625*uk_161 + 658125*uk_162 + 38025*uk_163 + 2197*uk_164 + 3025*uk_17 + 330*uk_18 + 715*uk_19 + 55*uk_2 + 3300*uk_20 + 4565*uk_21 + 12375*uk_22 + 715*uk_23 + 36*uk_24 + 78*uk_25 + 360*uk_26 + 498*uk_27 + 1350*uk_28 + 78*uk_29 + 6*uk_3 + 169*uk_30 + 780*uk_31 + 1079*uk_32 + 2925*uk_33 + 169*uk_34 + 3600*uk_35 + 4980*uk_36 + 13500*uk_37 + 780*uk_38 + 6889*uk_39 + 13*uk_4 + 18675*uk_40 + 1079*uk_41 + 50625*uk_42 + 2925*uk_43 + 169*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 15434501766*uk_47 + 33441420493*uk_48 + 154345017660*uk_49 + 60*uk_5 + 213510607763*uk_50 + 578793816225*uk_51 + 33441420493*uk_52 + 153424975*uk_53 + 16737270*uk_54 + 36264085*uk_55 + 167372700*uk_56 + 231532235*uk_57 + 627647625*uk_58 + 36264085*uk_59 + 83*uk_6 + 1825884*uk_60 + 3956082*uk_61 + 18258840*uk_62 + 25258062*uk_63 + 68470650*uk_64 + 3956082*uk_65 + 8571511*uk_66 + 39560820*uk_67 + 54725801*uk_68 + 148353075*uk_69 + 225*uk_7 + 8571511*uk_70 + 182588400*uk_71 + 252580620*uk_72 + 684706500*uk_73 + 39560820*uk_74 + 349403191*uk_75 + 947177325*uk_76 + 54725801*uk_77 + 2567649375*uk_78 + 148353075*uk_79 + 13*uk_8 + 8571511*uk_80 + 166375*uk_81 + 18150*uk_82 + 39325*uk_83 + 181500*uk_84 + 251075*uk_85 + 680625*uk_86 + 39325*uk_87 + 1980*uk_88 + 4290*uk_89 + 2572416961*uk_9 + 19800*uk_90 + 27390*uk_91 + 74250*uk_92 + 4290*uk_93 + 9295*uk_94 + 42900*uk_95 + 59345*uk_96 + 160875*uk_97 + 9295*uk_98 + 198000*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 280500*uk_100 + 742500*uk_101 + 19800*uk_102 + 397375*uk_103 + 1051875*uk_104 + 28050*uk_105 + 2784375*uk_106 + 74250*uk_107 + 1980*uk_108 + 205379*uk_109 + 2992421*uk_11 + 20886*uk_110 + 208860*uk_111 + 295885*uk_112 + 783225*uk_113 + 20886*uk_114 + 2124*uk_115 + 21240*uk_116 + 30090*uk_117 + 79650*uk_118 + 2124*uk_119 + 304314*uk_12 + 212400*uk_120 + 300900*uk_121 + 796500*uk_122 + 21240*uk_123 + 426275*uk_124 + 1128375*uk_125 + 30090*uk_126 + 2986875*uk_127 + 79650*uk_128 + 2124*uk_129 + 3043140*uk_13 + 216*uk_130 + 2160*uk_131 + 3060*uk_132 + 8100*uk_133 + 216*uk_134 + 21600*uk_135 + 30600*uk_136 + 81000*uk_137 + 2160*uk_138 + 43350*uk_139 + 4311115*uk_14 + 114750*uk_140 + 3060*uk_141 + 303750*uk_142 + 8100*uk_143 + 216*uk_144 + 216000*uk_145 + 306000*uk_146 + 810000*uk_147 + 21600*uk_148 + 433500*uk_149 + 11411775*uk_15 + 1147500*uk_150 + 30600*uk_151 + 3037500*uk_152 + 81000*uk_153 + 2160*uk_154 + 614125*uk_155 + 1625625*uk_156 + 43350*uk_157 + 4303125*uk_158 + 114750*uk_159 + 304314*uk_16 + 3060*uk_160 + 11390625*uk_161 + 303750*uk_162 + 8100*uk_163 + 216*uk_164 + 3025*uk_17 + 3245*uk_18 + 330*uk_19 + 55*uk_2 + 3300*uk_20 + 4675*uk_21 + 12375*uk_22 + 330*uk_23 + 3481*uk_24 + 354*uk_25 + 3540*uk_26 + 5015*uk_27 + 13275*uk_28 + 354*uk_29 + 59*uk_3 + 36*uk_30 + 360*uk_31 + 510*uk_32 + 1350*uk_33 + 36*uk_34 + 3600*uk_35 + 5100*uk_36 + 13500*uk_37 + 360*uk_38 + 7225*uk_39 + 6*uk_4 + 19125*uk_40 + 510*uk_41 + 50625*uk_42 + 1350*uk_43 + 36*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 151772600699*uk_47 + 15434501766*uk_48 + 154345017660*uk_49 + 60*uk_5 + 218655441685*uk_50 + 578793816225*uk_51 + 15434501766*uk_52 + 153424975*uk_53 + 164583155*uk_54 + 16737270*uk_55 + 167372700*uk_56 + 237111325*uk_57 + 627647625*uk_58 + 16737270*uk_59 + 85*uk_6 + 176552839*uk_60 + 17954526*uk_61 + 179545260*uk_62 + 254355785*uk_63 + 673294725*uk_64 + 17954526*uk_65 + 1825884*uk_66 + 18258840*uk_67 + 25866690*uk_68 + 68470650*uk_69 + 225*uk_7 + 1825884*uk_70 + 182588400*uk_71 + 258666900*uk_72 + 684706500*uk_73 + 18258840*uk_74 + 366444775*uk_75 + 970000875*uk_76 + 25866690*uk_77 + 2567649375*uk_78 + 68470650*uk_79 + 6*uk_8 + 1825884*uk_80 + 166375*uk_81 + 178475*uk_82 + 18150*uk_83 + 181500*uk_84 + 257125*uk_85 + 680625*uk_86 + 18150*uk_87 + 191455*uk_88 + 19470*uk_89 + 2572416961*uk_9 + 194700*uk_90 + 275825*uk_91 + 730125*uk_92 + 19470*uk_93 + 1980*uk_94 + 19800*uk_95 + 28050*uk_96 + 74250*uk_97 + 1980*uk_98 + 198000*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 267960*uk_100 + 693000*uk_101 + 181720*uk_102 + 416295*uk_103 + 1076625*uk_104 + 282315*uk_105 + 2784375*uk_106 + 730125*uk_107 + 191455*uk_108 + 614125*uk_109 + 4311115*uk_11 + 426275*uk_110 + 404600*uk_111 + 628575*uk_112 + 1625625*uk_113 + 426275*uk_114 + 295885*uk_115 + 280840*uk_116 + 436305*uk_117 + 1128375*uk_118 + 295885*uk_119 + 2992421*uk_12 + 266560*uk_120 + 414120*uk_121 + 1071000*uk_122 + 280840*uk_123 + 643365*uk_124 + 1663875*uk_125 + 436305*uk_126 + 4303125*uk_127 + 1128375*uk_128 + 295885*uk_129 + 2840264*uk_13 + 205379*uk_130 + 194936*uk_131 + 302847*uk_132 + 783225*uk_133 + 205379*uk_134 + 185024*uk_135 + 287448*uk_136 + 743400*uk_137 + 194936*uk_138 + 446571*uk_139 + 4412553*uk_14 + 1154925*uk_140 + 302847*uk_141 + 2986875*uk_142 + 783225*uk_143 + 205379*uk_144 + 175616*uk_145 + 272832*uk_146 + 705600*uk_147 + 185024*uk_148 + 423864*uk_149 + 11411775*uk_15 + 1096200*uk_150 + 287448*uk_151 + 2835000*uk_152 + 743400*uk_153 + 194936*uk_154 + 658503*uk_155 + 1703025*uk_156 + 446571*uk_157 + 4404375*uk_158 + 1154925*uk_159 + 2992421*uk_16 + 302847*uk_160 + 11390625*uk_161 + 2986875*uk_162 + 783225*uk_163 + 205379*uk_164 + 3025*uk_17 + 4675*uk_18 + 3245*uk_19 + 55*uk_2 + 3080*uk_20 + 4785*uk_21 + 12375*uk_22 + 3245*uk_23 + 7225*uk_24 + 5015*uk_25 + 4760*uk_26 + 7395*uk_27 + 19125*uk_28 + 5015*uk_29 + 85*uk_3 + 3481*uk_30 + 3304*uk_31 + 5133*uk_32 + 13275*uk_33 + 3481*uk_34 + 3136*uk_35 + 4872*uk_36 + 12600*uk_37 + 3304*uk_38 + 7569*uk_39 + 59*uk_4 + 19575*uk_40 + 5133*uk_41 + 50625*uk_42 + 13275*uk_43 + 3481*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 218655441685*uk_47 + 151772600699*uk_48 + 144055349816*uk_49 + 56*uk_5 + 223800275607*uk_50 + 578793816225*uk_51 + 151772600699*uk_52 + 153424975*uk_53 + 237111325*uk_54 + 164583155*uk_55 + 156214520*uk_56 + 242690415*uk_57 + 627647625*uk_58 + 164583155*uk_59 + 87*uk_6 + 366444775*uk_60 + 254355785*uk_61 + 241422440*uk_62 + 375067005*uk_63 + 970000875*uk_64 + 254355785*uk_65 + 176552839*uk_66 + 167575576*uk_67 + 260340627*uk_68 + 673294725*uk_69 + 225*uk_7 + 176552839*uk_70 + 159054784*uk_71 + 247102968*uk_72 + 639059400*uk_73 + 167575576*uk_74 + 383892111*uk_75 + 992824425*uk_76 + 260340627*uk_77 + 2567649375*uk_78 + 673294725*uk_79 + 59*uk_8 + 176552839*uk_80 + 166375*uk_81 + 257125*uk_82 + 178475*uk_83 + 169400*uk_84 + 263175*uk_85 + 680625*uk_86 + 178475*uk_87 + 397375*uk_88 + 275825*uk_89 + 2572416961*uk_9 + 261800*uk_90 + 406725*uk_91 + 1051875*uk_92 + 275825*uk_93 + 191455*uk_94 + 181720*uk_95 + 282315*uk_96 + 730125*uk_97 + 191455*uk_98 + 172480*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 254540*uk_100 + 643500*uk_101 + 243100*uk_102 + 435655*uk_103 + 1101375*uk_104 + 416075*uk_105 + 2784375*uk_106 + 1051875*uk_107 + 397375*uk_108 + 474552*uk_109 + 3956082*uk_11 + 517140*uk_110 + 316368*uk_111 + 541476*uk_112 + 1368900*uk_113 + 517140*uk_114 + 563550*uk_115 + 344760*uk_116 + 590070*uk_117 + 1491750*uk_118 + 563550*uk_119 + 4311115*uk_12 + 210912*uk_120 + 360984*uk_121 + 912600*uk_122 + 344760*uk_123 + 617838*uk_124 + 1561950*uk_125 + 590070*uk_126 + 3948750*uk_127 + 1491750*uk_128 + 563550*uk_129 + 2637388*uk_13 + 614125*uk_130 + 375700*uk_131 + 643025*uk_132 + 1625625*uk_133 + 614125*uk_134 + 229840*uk_135 + 393380*uk_136 + 994500*uk_137 + 375700*uk_138 + 673285*uk_139 + 4513991*uk_14 + 1702125*uk_140 + 643025*uk_141 + 4303125*uk_142 + 1625625*uk_143 + 614125*uk_144 + 140608*uk_145 + 240656*uk_146 + 608400*uk_147 + 229840*uk_148 + 411892*uk_149 + 11411775*uk_15 + 1041300*uk_150 + 393380*uk_151 + 2632500*uk_152 + 994500*uk_153 + 375700*uk_154 + 704969*uk_155 + 1782225*uk_156 + 673285*uk_157 + 4505625*uk_158 + 1702125*uk_159 + 4311115*uk_16 + 643025*uk_160 + 11390625*uk_161 + 4303125*uk_162 + 1625625*uk_163 + 614125*uk_164 + 3025*uk_17 + 4290*uk_18 + 4675*uk_19 + 55*uk_2 + 2860*uk_20 + 4895*uk_21 + 12375*uk_22 + 4675*uk_23 + 6084*uk_24 + 6630*uk_25 + 4056*uk_26 + 6942*uk_27 + 17550*uk_28 + 6630*uk_29 + 78*uk_3 + 7225*uk_30 + 4420*uk_31 + 7565*uk_32 + 19125*uk_33 + 7225*uk_34 + 2704*uk_35 + 4628*uk_36 + 11700*uk_37 + 4420*uk_38 + 7921*uk_39 + 85*uk_4 + 20025*uk_40 + 7565*uk_41 + 50625*uk_42 + 19125*uk_43 + 7225*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 200648522958*uk_47 + 218655441685*uk_48 + 133765681972*uk_49 + 52*uk_5 + 228945109529*uk_50 + 578793816225*uk_51 + 218655441685*uk_52 + 153424975*uk_53 + 217584510*uk_54 + 237111325*uk_55 + 145056340*uk_56 + 248269505*uk_57 + 627647625*uk_58 + 237111325*uk_59 + 89*uk_6 + 308574396*uk_60 + 336266970*uk_61 + 205716264*uk_62 + 352091298*uk_63 + 890118450*uk_64 + 336266970*uk_65 + 366444775*uk_66 + 224177980*uk_67 + 383689235*uk_68 + 970000875*uk_69 + 225*uk_7 + 366444775*uk_70 + 137144176*uk_71 + 234727532*uk_72 + 593412300*uk_73 + 224177980*uk_74 + 401745199*uk_75 + 1015647975*uk_76 + 383689235*uk_77 + 2567649375*uk_78 + 970000875*uk_79 + 85*uk_8 + 366444775*uk_80 + 166375*uk_81 + 235950*uk_82 + 257125*uk_83 + 157300*uk_84 + 269225*uk_85 + 680625*uk_86 + 257125*uk_87 + 334620*uk_88 + 364650*uk_89 + 2572416961*uk_9 + 223080*uk_90 + 381810*uk_91 + 965250*uk_92 + 364650*uk_93 + 397375*uk_94 + 243100*uk_95 + 416075*uk_96 + 1051875*uk_97 + 397375*uk_98 + 148720*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 240240*uk_100 + 594000*uk_101 + 205920*uk_102 + 455455*uk_103 + 1126125*uk_104 + 390390*uk_105 + 2784375*uk_106 + 965250*uk_107 + 334620*uk_108 + 32768*uk_109 + 1623008*uk_11 + 79872*uk_110 + 49152*uk_111 + 93184*uk_112 + 230400*uk_113 + 79872*uk_114 + 194688*uk_115 + 119808*uk_116 + 227136*uk_117 + 561600*uk_118 + 194688*uk_119 + 3956082*uk_12 + 73728*uk_120 + 139776*uk_121 + 345600*uk_122 + 119808*uk_123 + 264992*uk_124 + 655200*uk_125 + 227136*uk_126 + 1620000*uk_127 + 561600*uk_128 + 194688*uk_129 + 2434512*uk_13 + 474552*uk_130 + 292032*uk_131 + 553644*uk_132 + 1368900*uk_133 + 474552*uk_134 + 179712*uk_135 + 340704*uk_136 + 842400*uk_137 + 292032*uk_138 + 645918*uk_139 + 4615429*uk_14 + 1597050*uk_140 + 553644*uk_141 + 3948750*uk_142 + 1368900*uk_143 + 474552*uk_144 + 110592*uk_145 + 209664*uk_146 + 518400*uk_147 + 179712*uk_148 + 397488*uk_149 + 11411775*uk_15 + 982800*uk_150 + 340704*uk_151 + 2430000*uk_152 + 842400*uk_153 + 292032*uk_154 + 753571*uk_155 + 1863225*uk_156 + 645918*uk_157 + 4606875*uk_158 + 1597050*uk_159 + 3956082*uk_16 + 553644*uk_160 + 11390625*uk_161 + 3948750*uk_162 + 1368900*uk_163 + 474552*uk_164 + 3025*uk_17 + 1760*uk_18 + 4290*uk_19 + 55*uk_2 + 2640*uk_20 + 5005*uk_21 + 12375*uk_22 + 4290*uk_23 + 1024*uk_24 + 2496*uk_25 + 1536*uk_26 + 2912*uk_27 + 7200*uk_28 + 2496*uk_29 + 32*uk_3 + 6084*uk_30 + 3744*uk_31 + 7098*uk_32 + 17550*uk_33 + 6084*uk_34 + 2304*uk_35 + 4368*uk_36 + 10800*uk_37 + 3744*uk_38 + 8281*uk_39 + 78*uk_4 + 20475*uk_40 + 7098*uk_41 + 50625*uk_42 + 17550*uk_43 + 6084*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 82317342752*uk_47 + 200648522958*uk_48 + 123476014128*uk_49 + 48*uk_5 + 234089943451*uk_50 + 578793816225*uk_51 + 200648522958*uk_52 + 153424975*uk_53 + 89265440*uk_54 + 217584510*uk_55 + 133898160*uk_56 + 253848595*uk_57 + 627647625*uk_58 + 217584510*uk_59 + 91*uk_6 + 51936256*uk_60 + 126594624*uk_61 + 77904384*uk_62 + 147693728*uk_63 + 365176800*uk_64 + 126594624*uk_65 + 308574396*uk_66 + 189891936*uk_67 + 360003462*uk_68 + 890118450*uk_69 + 225*uk_7 + 308574396*uk_70 + 116856576*uk_71 + 221540592*uk_72 + 547765200*uk_73 + 189891936*uk_74 + 420004039*uk_75 + 1038471525*uk_76 + 360003462*uk_77 + 2567649375*uk_78 + 890118450*uk_79 + 78*uk_8 + 308574396*uk_80 + 166375*uk_81 + 96800*uk_82 + 235950*uk_83 + 145200*uk_84 + 275275*uk_85 + 680625*uk_86 + 235950*uk_87 + 56320*uk_88 + 137280*uk_89 + 2572416961*uk_9 + 84480*uk_90 + 160160*uk_91 + 396000*uk_92 + 137280*uk_93 + 334620*uk_94 + 205920*uk_95 + 390390*uk_96 + 965250*uk_97 + 334620*uk_98 + 126720*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 245520*uk_100 + 594000*uk_101 + 84480*uk_102 + 475695*uk_103 + 1150875*uk_104 + 163680*uk_105 + 2784375*uk_106 + 396000*uk_107 + 56320*uk_108 + 39304*uk_109 + 1724446*uk_11 + 36992*uk_110 + 55488*uk_111 + 107508*uk_112 + 260100*uk_113 + 36992*uk_114 + 34816*uk_115 + 52224*uk_116 + 101184*uk_117 + 244800*uk_118 + 34816*uk_119 + 1623008*uk_12 + 78336*uk_120 + 151776*uk_121 + 367200*uk_122 + 52224*uk_123 + 294066*uk_124 + 711450*uk_125 + 101184*uk_126 + 1721250*uk_127 + 244800*uk_128 + 34816*uk_129 + 2434512*uk_13 + 32768*uk_130 + 49152*uk_131 + 95232*uk_132 + 230400*uk_133 + 32768*uk_134 + 73728*uk_135 + 142848*uk_136 + 345600*uk_137 + 49152*uk_138 + 276768*uk_139 + 4716867*uk_14 + 669600*uk_140 + 95232*uk_141 + 1620000*uk_142 + 230400*uk_143 + 32768*uk_144 + 110592*uk_145 + 214272*uk_146 + 518400*uk_147 + 73728*uk_148 + 415152*uk_149 + 11411775*uk_15 + 1004400*uk_150 + 142848*uk_151 + 2430000*uk_152 + 345600*uk_153 + 49152*uk_154 + 804357*uk_155 + 1946025*uk_156 + 276768*uk_157 + 4708125*uk_158 + 669600*uk_159 + 1623008*uk_16 + 95232*uk_160 + 11390625*uk_161 + 1620000*uk_162 + 230400*uk_163 + 32768*uk_164 + 3025*uk_17 + 1870*uk_18 + 1760*uk_19 + 55*uk_2 + 2640*uk_20 + 5115*uk_21 + 12375*uk_22 + 1760*uk_23 + 1156*uk_24 + 1088*uk_25 + 1632*uk_26 + 3162*uk_27 + 7650*uk_28 + 1088*uk_29 + 34*uk_3 + 1024*uk_30 + 1536*uk_31 + 2976*uk_32 + 7200*uk_33 + 1024*uk_34 + 2304*uk_35 + 4464*uk_36 + 10800*uk_37 + 1536*uk_38 + 8649*uk_39 + 32*uk_4 + 20925*uk_40 + 2976*uk_41 + 50625*uk_42 + 7200*uk_43 + 1024*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 87462176674*uk_47 + 82317342752*uk_48 + 123476014128*uk_49 + 48*uk_5 + 239234777373*uk_50 + 578793816225*uk_51 + 82317342752*uk_52 + 153424975*uk_53 + 94844530*uk_54 + 89265440*uk_55 + 133898160*uk_56 + 259427685*uk_57 + 627647625*uk_58 + 89265440*uk_59 + 93*uk_6 + 58631164*uk_60 + 55182272*uk_61 + 82773408*uk_62 + 160373478*uk_63 + 388000350*uk_64 + 55182272*uk_65 + 51936256*uk_66 + 77904384*uk_67 + 150939744*uk_68 + 365176800*uk_69 + 225*uk_7 + 51936256*uk_70 + 116856576*uk_71 + 226409616*uk_72 + 547765200*uk_73 + 77904384*uk_74 + 438668631*uk_75 + 1061295075*uk_76 + 150939744*uk_77 + 2567649375*uk_78 + 365176800*uk_79 + 32*uk_8 + 51936256*uk_80 + 166375*uk_81 + 102850*uk_82 + 96800*uk_83 + 145200*uk_84 + 281325*uk_85 + 680625*uk_86 + 96800*uk_87 + 63580*uk_88 + 59840*uk_89 + 2572416961*uk_9 + 89760*uk_90 + 173910*uk_91 + 420750*uk_92 + 59840*uk_93 + 56320*uk_94 + 84480*uk_95 + 163680*uk_96 + 396000*uk_97 + 56320*uk_98 + 126720*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 250800*uk_100 + 594000*uk_101 + 89760*uk_102 + 496375*uk_103 + 1175625*uk_104 + 177650*uk_105 + 2784375*uk_106 + 420750*uk_107 + 63580*uk_108 + 592704*uk_109 + 4260396*uk_11 + 239904*uk_110 + 338688*uk_111 + 670320*uk_112 + 1587600*uk_113 + 239904*uk_114 + 97104*uk_115 + 137088*uk_116 + 271320*uk_117 + 642600*uk_118 + 97104*uk_119 + 1724446*uk_12 + 193536*uk_120 + 383040*uk_121 + 907200*uk_122 + 137088*uk_123 + 758100*uk_124 + 1795500*uk_125 + 271320*uk_126 + 4252500*uk_127 + 642600*uk_128 + 97104*uk_129 + 2434512*uk_13 + 39304*uk_130 + 55488*uk_131 + 109820*uk_132 + 260100*uk_133 + 39304*uk_134 + 78336*uk_135 + 155040*uk_136 + 367200*uk_137 + 55488*uk_138 + 306850*uk_139 + 4818305*uk_14 + 726750*uk_140 + 109820*uk_141 + 1721250*uk_142 + 260100*uk_143 + 39304*uk_144 + 110592*uk_145 + 218880*uk_146 + 518400*uk_147 + 78336*uk_148 + 433200*uk_149 + 11411775*uk_15 + 1026000*uk_150 + 155040*uk_151 + 2430000*uk_152 + 367200*uk_153 + 55488*uk_154 + 857375*uk_155 + 2030625*uk_156 + 306850*uk_157 + 4809375*uk_158 + 726750*uk_159 + 1724446*uk_16 + 109820*uk_160 + 11390625*uk_161 + 1721250*uk_162 + 260100*uk_163 + 39304*uk_164 + 3025*uk_17 + 4620*uk_18 + 1870*uk_19 + 55*uk_2 + 2640*uk_20 + 5225*uk_21 + 12375*uk_22 + 1870*uk_23 + 7056*uk_24 + 2856*uk_25 + 4032*uk_26 + 7980*uk_27 + 18900*uk_28 + 2856*uk_29 + 84*uk_3 + 1156*uk_30 + 1632*uk_31 + 3230*uk_32 + 7650*uk_33 + 1156*uk_34 + 2304*uk_35 + 4560*uk_36 + 10800*uk_37 + 1632*uk_38 + 9025*uk_39 + 34*uk_4 + 21375*uk_40 + 3230*uk_41 + 50625*uk_42 + 7650*uk_43 + 1156*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 216083024724*uk_47 + 87462176674*uk_48 + 123476014128*uk_49 + 48*uk_5 + 244379611295*uk_50 + 578793816225*uk_51 + 87462176674*uk_52 + 153424975*uk_53 + 234321780*uk_54 + 94844530*uk_55 + 133898160*uk_56 + 265006775*uk_57 + 627647625*uk_58 + 94844530*uk_59 + 95*uk_6 + 357873264*uk_60 + 144853464*uk_61 + 204499008*uk_62 + 404737620*uk_63 + 958589100*uk_64 + 144853464*uk_65 + 58631164*uk_66 + 82773408*uk_67 + 163822370*uk_68 + 388000350*uk_69 + 225*uk_7 + 58631164*uk_70 + 116856576*uk_71 + 231278640*uk_72 + 547765200*uk_73 + 82773408*uk_74 + 457738975*uk_75 + 1084118625*uk_76 + 163822370*uk_77 + 2567649375*uk_78 + 388000350*uk_79 + 34*uk_8 + 58631164*uk_80 + 166375*uk_81 + 254100*uk_82 + 102850*uk_83 + 145200*uk_84 + 287375*uk_85 + 680625*uk_86 + 102850*uk_87 + 388080*uk_88 + 157080*uk_89 + 2572416961*uk_9 + 221760*uk_90 + 438900*uk_91 + 1039500*uk_92 + 157080*uk_93 + 63580*uk_94 + 89760*uk_95 + 177650*uk_96 + 420750*uk_97 + 63580*uk_98 + 126720*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 234740*uk_100 + 544500*uk_101 + 203280*uk_102 + 517495*uk_103 + 1200375*uk_104 + 448140*uk_105 + 2784375*uk_106 + 1039500*uk_107 + 388080*uk_108 + 614125*uk_109 + 4311115*uk_11 + 606900*uk_110 + 317900*uk_111 + 700825*uk_112 + 1625625*uk_113 + 606900*uk_114 + 599760*uk_115 + 314160*uk_116 + 692580*uk_117 + 1606500*uk_118 + 599760*uk_119 + 4260396*uk_12 + 164560*uk_120 + 362780*uk_121 + 841500*uk_122 + 314160*uk_123 + 799765*uk_124 + 1855125*uk_125 + 692580*uk_126 + 4303125*uk_127 + 1606500*uk_128 + 599760*uk_129 + 2231636*uk_13 + 592704*uk_130 + 310464*uk_131 + 684432*uk_132 + 1587600*uk_133 + 592704*uk_134 + 162624*uk_135 + 358512*uk_136 + 831600*uk_137 + 310464*uk_138 + 790356*uk_139 + 4919743*uk_14 + 1833300*uk_140 + 684432*uk_141 + 4252500*uk_142 + 1587600*uk_143 + 592704*uk_144 + 85184*uk_145 + 187792*uk_146 + 435600*uk_147 + 162624*uk_148 + 413996*uk_149 + 11411775*uk_15 + 960300*uk_150 + 358512*uk_151 + 2227500*uk_152 + 831600*uk_153 + 310464*uk_154 + 912673*uk_155 + 2117025*uk_156 + 790356*uk_157 + 4910625*uk_158 + 1833300*uk_159 + 4260396*uk_16 + 684432*uk_160 + 11390625*uk_161 + 4252500*uk_162 + 1587600*uk_163 + 592704*uk_164 + 3025*uk_17 + 4675*uk_18 + 4620*uk_19 + 55*uk_2 + 2420*uk_20 + 5335*uk_21 + 12375*uk_22 + 4620*uk_23 + 7225*uk_24 + 7140*uk_25 + 3740*uk_26 + 8245*uk_27 + 19125*uk_28 + 7140*uk_29 + 85*uk_3 + 7056*uk_30 + 3696*uk_31 + 8148*uk_32 + 18900*uk_33 + 7056*uk_34 + 1936*uk_35 + 4268*uk_36 + 9900*uk_37 + 3696*uk_38 + 9409*uk_39 + 84*uk_4 + 21825*uk_40 + 8148*uk_41 + 50625*uk_42 + 18900*uk_43 + 7056*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 218655441685*uk_47 + 216083024724*uk_48 + 113186346284*uk_49 + 44*uk_5 + 249524445217*uk_50 + 578793816225*uk_51 + 216083024724*uk_52 + 153424975*uk_53 + 237111325*uk_54 + 234321780*uk_55 + 122739980*uk_56 + 270585865*uk_57 + 627647625*uk_58 + 234321780*uk_59 + 97*uk_6 + 366444775*uk_60 + 362133660*uk_61 + 189689060*uk_62 + 418178155*uk_63 + 970000875*uk_64 + 362133660*uk_65 + 357873264*uk_66 + 187457424*uk_67 + 413258412*uk_68 + 958589100*uk_69 + 225*uk_7 + 357873264*uk_70 + 98191984*uk_71 + 216468692*uk_72 + 502118100*uk_73 + 187457424*uk_74 + 477215071*uk_75 + 1106942175*uk_76 + 413258412*uk_77 + 2567649375*uk_78 + 958589100*uk_79 + 84*uk_8 + 357873264*uk_80 + 166375*uk_81 + 257125*uk_82 + 254100*uk_83 + 133100*uk_84 + 293425*uk_85 + 680625*uk_86 + 254100*uk_87 + 397375*uk_88 + 392700*uk_89 + 2572416961*uk_9 + 205700*uk_90 + 453475*uk_91 + 1051875*uk_92 + 392700*uk_93 + 388080*uk_94 + 203280*uk_95 + 448140*uk_96 + 1039500*uk_97 + 388080*uk_98 + 106480*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 217800*uk_100 + 495000*uk_101 + 187000*uk_102 + 539055*uk_103 + 1225125*uk_104 + 462825*uk_105 + 2784375*uk_106 + 1051875*uk_107 + 397375*uk_108 + 29791*uk_109 + 1572289*uk_11 + 81685*uk_110 + 38440*uk_111 + 95139*uk_112 + 216225*uk_113 + 81685*uk_114 + 223975*uk_115 + 105400*uk_116 + 260865*uk_117 + 592875*uk_118 + 223975*uk_119 + 4311115*uk_12 + 49600*uk_120 + 122760*uk_121 + 279000*uk_122 + 105400*uk_123 + 303831*uk_124 + 690525*uk_125 + 260865*uk_126 + 1569375*uk_127 + 592875*uk_128 + 223975*uk_129 + 2028760*uk_13 + 614125*uk_130 + 289000*uk_131 + 715275*uk_132 + 1625625*uk_133 + 614125*uk_134 + 136000*uk_135 + 336600*uk_136 + 765000*uk_137 + 289000*uk_138 + 833085*uk_139 + 5021181*uk_14 + 1893375*uk_140 + 715275*uk_141 + 4303125*uk_142 + 1625625*uk_143 + 614125*uk_144 + 64000*uk_145 + 158400*uk_146 + 360000*uk_147 + 136000*uk_148 + 392040*uk_149 + 11411775*uk_15 + 891000*uk_150 + 336600*uk_151 + 2025000*uk_152 + 765000*uk_153 + 289000*uk_154 + 970299*uk_155 + 2205225*uk_156 + 833085*uk_157 + 5011875*uk_158 + 1893375*uk_159 + 4311115*uk_16 + 715275*uk_160 + 11390625*uk_161 + 4303125*uk_162 + 1625625*uk_163 + 614125*uk_164 + 3025*uk_17 + 1705*uk_18 + 4675*uk_19 + 55*uk_2 + 2200*uk_20 + 5445*uk_21 + 12375*uk_22 + 4675*uk_23 + 961*uk_24 + 2635*uk_25 + 1240*uk_26 + 3069*uk_27 + 6975*uk_28 + 2635*uk_29 + 31*uk_3 + 7225*uk_30 + 3400*uk_31 + 8415*uk_32 + 19125*uk_33 + 7225*uk_34 + 1600*uk_35 + 3960*uk_36 + 9000*uk_37 + 3400*uk_38 + 9801*uk_39 + 85*uk_4 + 22275*uk_40 + 8415*uk_41 + 50625*uk_42 + 19125*uk_43 + 7225*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 79744925791*uk_47 + 218655441685*uk_48 + 102896678440*uk_49 + 40*uk_5 + 254669279139*uk_50 + 578793816225*uk_51 + 218655441685*uk_52 + 153424975*uk_53 + 86475895*uk_54 + 237111325*uk_55 + 111581800*uk_56 + 276164955*uk_57 + 627647625*uk_58 + 237111325*uk_59 + 99*uk_6 + 48740959*uk_60 + 133644565*uk_61 + 62891560*uk_62 + 155656611*uk_63 + 353765025*uk_64 + 133644565*uk_65 + 366444775*uk_66 + 172444600*uk_67 + 426800385*uk_68 + 970000875*uk_69 + 225*uk_7 + 366444775*uk_70 + 81150400*uk_71 + 200847240*uk_72 + 456471000*uk_73 + 172444600*uk_74 + 497096919*uk_75 + 1129765725*uk_76 + 426800385*uk_77 + 2567649375*uk_78 + 970000875*uk_79 + 85*uk_8 + 366444775*uk_80 + 166375*uk_81 + 93775*uk_82 + 257125*uk_83 + 121000*uk_84 + 299475*uk_85 + 680625*uk_86 + 257125*uk_87 + 52855*uk_88 + 144925*uk_89 + 2572416961*uk_9 + 68200*uk_90 + 168795*uk_91 + 383625*uk_92 + 144925*uk_93 + 397375*uk_94 + 187000*uk_95 + 462825*uk_96 + 1051875*uk_97 + 397375*uk_98 + 88000*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 222200*uk_100 + 495000*uk_101 + 68200*uk_102 + 561055*uk_103 + 1249875*uk_104 + 172205*uk_105 + 2784375*uk_106 + 383625*uk_107 + 52855*uk_108 + 4913*uk_109 + 862223*uk_11 + 8959*uk_110 + 11560*uk_111 + 29189*uk_112 + 65025*uk_113 + 8959*uk_114 + 16337*uk_115 + 21080*uk_116 + 53227*uk_117 + 118575*uk_118 + 16337*uk_119 + 1572289*uk_12 + 27200*uk_120 + 68680*uk_121 + 153000*uk_122 + 21080*uk_123 + 173417*uk_124 + 386325*uk_125 + 53227*uk_126 + 860625*uk_127 + 118575*uk_128 + 16337*uk_129 + 2028760*uk_13 + 29791*uk_130 + 38440*uk_131 + 97061*uk_132 + 216225*uk_133 + 29791*uk_134 + 49600*uk_135 + 125240*uk_136 + 279000*uk_137 + 38440*uk_138 + 316231*uk_139 + 5122619*uk_14 + 704475*uk_140 + 97061*uk_141 + 1569375*uk_142 + 216225*uk_143 + 29791*uk_144 + 64000*uk_145 + 161600*uk_146 + 360000*uk_147 + 49600*uk_148 + 408040*uk_149 + 11411775*uk_15 + 909000*uk_150 + 125240*uk_151 + 2025000*uk_152 + 279000*uk_153 + 38440*uk_154 + 1030301*uk_155 + 2295225*uk_156 + 316231*uk_157 + 5113125*uk_158 + 704475*uk_159 + 1572289*uk_16 + 97061*uk_160 + 11390625*uk_161 + 1569375*uk_162 + 216225*uk_163 + 29791*uk_164 + 3025*uk_17 + 935*uk_18 + 1705*uk_19 + 55*uk_2 + 2200*uk_20 + 5555*uk_21 + 12375*uk_22 + 1705*uk_23 + 289*uk_24 + 527*uk_25 + 680*uk_26 + 1717*uk_27 + 3825*uk_28 + 527*uk_29 + 17*uk_3 + 961*uk_30 + 1240*uk_31 + 3131*uk_32 + 6975*uk_33 + 961*uk_34 + 1600*uk_35 + 4040*uk_36 + 9000*uk_37 + 1240*uk_38 + 10201*uk_39 + 31*uk_4 + 22725*uk_40 + 3131*uk_41 + 50625*uk_42 + 6975*uk_43 + 961*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 43731088337*uk_47 + 79744925791*uk_48 + 102896678440*uk_49 + 40*uk_5 + 259814113061*uk_50 + 578793816225*uk_51 + 79744925791*uk_52 + 153424975*uk_53 + 47422265*uk_54 + 86475895*uk_55 + 111581800*uk_56 + 281744045*uk_57 + 627647625*uk_58 + 86475895*uk_59 + 101*uk_6 + 14657791*uk_60 + 26728913*uk_61 + 34488920*uk_62 + 87084523*uk_63 + 194000175*uk_64 + 26728913*uk_65 + 48740959*uk_66 + 62891560*uk_67 + 158801189*uk_68 + 353765025*uk_69 + 225*uk_7 + 48740959*uk_70 + 81150400*uk_71 + 204904760*uk_72 + 456471000*uk_73 + 62891560*uk_74 + 517384519*uk_75 + 1152589275*uk_76 + 158801189*uk_77 + 2567649375*uk_78 + 353765025*uk_79 + 31*uk_8 + 48740959*uk_80 + 166375*uk_81 + 51425*uk_82 + 93775*uk_83 + 121000*uk_84 + 305525*uk_85 + 680625*uk_86 + 93775*uk_87 + 15895*uk_88 + 28985*uk_89 + 2572416961*uk_9 + 37400*uk_90 + 94435*uk_91 + 210375*uk_92 + 28985*uk_93 + 52855*uk_94 + 68200*uk_95 + 172205*uk_96 + 383625*uk_97 + 52855*uk_98 + 88000*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 226600*uk_100 + 495000*uk_101 + 37400*uk_102 + 583495*uk_103 + 1274625*uk_104 + 96305*uk_105 + 2784375*uk_106 + 210375*uk_107 + 15895*uk_108 + 79507*uk_109 + 2180917*uk_11 + 31433*uk_110 + 73960*uk_111 + 190447*uk_112 + 416025*uk_113 + 31433*uk_114 + 12427*uk_115 + 29240*uk_116 + 75293*uk_117 + 164475*uk_118 + 12427*uk_119 + 862223*uk_12 + 68800*uk_120 + 177160*uk_121 + 387000*uk_122 + 29240*uk_123 + 456187*uk_124 + 996525*uk_125 + 75293*uk_126 + 2176875*uk_127 + 164475*uk_128 + 12427*uk_129 + 2028760*uk_13 + 4913*uk_130 + 11560*uk_131 + 29767*uk_132 + 65025*uk_133 + 4913*uk_134 + 27200*uk_135 + 70040*uk_136 + 153000*uk_137 + 11560*uk_138 + 180353*uk_139 + 5224057*uk_14 + 393975*uk_140 + 29767*uk_141 + 860625*uk_142 + 65025*uk_143 + 4913*uk_144 + 64000*uk_145 + 164800*uk_146 + 360000*uk_147 + 27200*uk_148 + 424360*uk_149 + 11411775*uk_15 + 927000*uk_150 + 70040*uk_151 + 2025000*uk_152 + 153000*uk_153 + 11560*uk_154 + 1092727*uk_155 + 2387025*uk_156 + 180353*uk_157 + 5214375*uk_158 + 393975*uk_159 + 862223*uk_16 + 29767*uk_160 + 11390625*uk_161 + 860625*uk_162 + 65025*uk_163 + 4913*uk_164 + 3025*uk_17 + 2365*uk_18 + 935*uk_19 + 55*uk_2 + 2200*uk_20 + 5665*uk_21 + 12375*uk_22 + 935*uk_23 + 1849*uk_24 + 731*uk_25 + 1720*uk_26 + 4429*uk_27 + 9675*uk_28 + 731*uk_29 + 43*uk_3 + 289*uk_30 + 680*uk_31 + 1751*uk_32 + 3825*uk_33 + 289*uk_34 + 1600*uk_35 + 4120*uk_36 + 9000*uk_37 + 680*uk_38 + 10609*uk_39 + 17*uk_4 + 23175*uk_40 + 1751*uk_41 + 50625*uk_42 + 3825*uk_43 + 289*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 110613929323*uk_47 + 43731088337*uk_48 + 102896678440*uk_49 + 40*uk_5 + 264958946983*uk_50 + 578793816225*uk_51 + 43731088337*uk_52 + 153424975*uk_53 + 119950435*uk_54 + 47422265*uk_55 + 111581800*uk_56 + 287323135*uk_57 + 627647625*uk_58 + 47422265*uk_59 + 103*uk_6 + 93779431*uk_60 + 37075589*uk_61 + 87236680*uk_62 + 224634451*uk_63 + 490706325*uk_64 + 37075589*uk_65 + 14657791*uk_66 + 34488920*uk_67 + 88808969*uk_68 + 194000175*uk_69 + 225*uk_7 + 14657791*uk_70 + 81150400*uk_71 + 208962280*uk_72 + 456471000*uk_73 + 34488920*uk_74 + 538077871*uk_75 + 1175412825*uk_76 + 88808969*uk_77 + 2567649375*uk_78 + 194000175*uk_79 + 17*uk_8 + 14657791*uk_80 + 166375*uk_81 + 130075*uk_82 + 51425*uk_83 + 121000*uk_84 + 311575*uk_85 + 680625*uk_86 + 51425*uk_87 + 101695*uk_88 + 40205*uk_89 + 2572416961*uk_9 + 94600*uk_90 + 243595*uk_91 + 532125*uk_92 + 40205*uk_93 + 15895*uk_94 + 37400*uk_95 + 96305*uk_96 + 210375*uk_97 + 15895*uk_98 + 88000*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 207900*uk_100 + 445500*uk_101 + 85140*uk_102 + 606375*uk_103 + 1299375*uk_104 + 248325*uk_105 + 2784375*uk_106 + 532125*uk_107 + 101695*uk_108 + 64*uk_109 + 202876*uk_11 + 688*uk_110 + 576*uk_111 + 1680*uk_112 + 3600*uk_113 + 688*uk_114 + 7396*uk_115 + 6192*uk_116 + 18060*uk_117 + 38700*uk_118 + 7396*uk_119 + 2180917*uk_12 + 5184*uk_120 + 15120*uk_121 + 32400*uk_122 + 6192*uk_123 + 44100*uk_124 + 94500*uk_125 + 18060*uk_126 + 202500*uk_127 + 38700*uk_128 + 7396*uk_129 + 1825884*uk_13 + 79507*uk_130 + 66564*uk_131 + 194145*uk_132 + 416025*uk_133 + 79507*uk_134 + 55728*uk_135 + 162540*uk_136 + 348300*uk_137 + 66564*uk_138 + 474075*uk_139 + 5325495*uk_14 + 1015875*uk_140 + 194145*uk_141 + 2176875*uk_142 + 416025*uk_143 + 79507*uk_144 + 46656*uk_145 + 136080*uk_146 + 291600*uk_147 + 55728*uk_148 + 396900*uk_149 + 11411775*uk_15 + 850500*uk_150 + 162540*uk_151 + 1822500*uk_152 + 348300*uk_153 + 66564*uk_154 + 1157625*uk_155 + 2480625*uk_156 + 474075*uk_157 + 5315625*uk_158 + 1015875*uk_159 + 2180917*uk_16 + 194145*uk_160 + 11390625*uk_161 + 2176875*uk_162 + 416025*uk_163 + 79507*uk_164 + 3025*uk_17 + 220*uk_18 + 2365*uk_19 + 55*uk_2 + 1980*uk_20 + 5775*uk_21 + 12375*uk_22 + 2365*uk_23 + 16*uk_24 + 172*uk_25 + 144*uk_26 + 420*uk_27 + 900*uk_28 + 172*uk_29 + 4*uk_3 + 1849*uk_30 + 1548*uk_31 + 4515*uk_32 + 9675*uk_33 + 1849*uk_34 + 1296*uk_35 + 3780*uk_36 + 8100*uk_37 + 1548*uk_38 + 11025*uk_39 + 43*uk_4 + 23625*uk_40 + 4515*uk_41 + 50625*uk_42 + 9675*uk_43 + 1849*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 10289667844*uk_47 + 110613929323*uk_48 + 92607010596*uk_49 + 36*uk_5 + 270103780905*uk_50 + 578793816225*uk_51 + 110613929323*uk_52 + 153424975*uk_53 + 11158180*uk_54 + 119950435*uk_55 + 100423620*uk_56 + 292902225*uk_57 + 627647625*uk_58 + 119950435*uk_59 + 105*uk_6 + 811504*uk_60 + 8723668*uk_61 + 7303536*uk_62 + 21301980*uk_63 + 45647100*uk_64 + 8723668*uk_65 + 93779431*uk_66 + 78513012*uk_67 + 228996285*uk_68 + 490706325*uk_69 + 225*uk_7 + 93779431*uk_70 + 65731824*uk_71 + 191717820*uk_72 + 410823900*uk_73 + 78513012*uk_74 + 559176975*uk_75 + 1198236375*uk_76 + 228996285*uk_77 + 2567649375*uk_78 + 490706325*uk_79 + 43*uk_8 + 93779431*uk_80 + 166375*uk_81 + 12100*uk_82 + 130075*uk_83 + 108900*uk_84 + 317625*uk_85 + 680625*uk_86 + 130075*uk_87 + 880*uk_88 + 9460*uk_89 + 2572416961*uk_9 + 7920*uk_90 + 23100*uk_91 + 49500*uk_92 + 9460*uk_93 + 101695*uk_94 + 85140*uk_95 + 248325*uk_96 + 532125*uk_97 + 101695*uk_98 + 71280*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 211860*uk_100 + 445500*uk_101 + 7920*uk_102 + 629695*uk_103 + 1324125*uk_104 + 23540*uk_105 + 2784375*uk_106 + 49500*uk_107 + 880*uk_108 + uk_109 + 50719*uk_11 + 4*uk_110 + 36*uk_111 + 107*uk_112 + 225*uk_113 + 4*uk_114 + 16*uk_115 + 144*uk_116 + 428*uk_117 + 900*uk_118 + 16*uk_119 + 202876*uk_12 + 1296*uk_120 + 3852*uk_121 + 8100*uk_122 + 144*uk_123 + 11449*uk_124 + 24075*uk_125 + 428*uk_126 + 50625*uk_127 + 900*uk_128 + 16*uk_129 + 1825884*uk_13 + 64*uk_130 + 576*uk_131 + 1712*uk_132 + 3600*uk_133 + 64*uk_134 + 5184*uk_135 + 15408*uk_136 + 32400*uk_137 + 576*uk_138 + 45796*uk_139 + 5426933*uk_14 + 96300*uk_140 + 1712*uk_141 + 202500*uk_142 + 3600*uk_143 + 64*uk_144 + 46656*uk_145 + 138672*uk_146 + 291600*uk_147 + 5184*uk_148 + 412164*uk_149 + 11411775*uk_15 + 866700*uk_150 + 15408*uk_151 + 1822500*uk_152 + 32400*uk_153 + 576*uk_154 + 1225043*uk_155 + 2576025*uk_156 + 45796*uk_157 + 5416875*uk_158 + 96300*uk_159 + 202876*uk_16 + 1712*uk_160 + 11390625*uk_161 + 202500*uk_162 + 3600*uk_163 + 64*uk_164 + 3025*uk_17 + 55*uk_18 + 220*uk_19 + 55*uk_2 + 1980*uk_20 + 5885*uk_21 + 12375*uk_22 + 220*uk_23 + uk_24 + 4*uk_25 + 36*uk_26 + 107*uk_27 + 225*uk_28 + 4*uk_29 + uk_3 + 16*uk_30 + 144*uk_31 + 428*uk_32 + 900*uk_33 + 16*uk_34 + 1296*uk_35 + 3852*uk_36 + 8100*uk_37 + 144*uk_38 + 11449*uk_39 + 4*uk_4 + 24075*uk_40 + 428*uk_41 + 50625*uk_42 + 900*uk_43 + 16*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 2572416961*uk_47 + 10289667844*uk_48 + 92607010596*uk_49 + 36*uk_5 + 275248614827*uk_50 + 578793816225*uk_51 + 10289667844*uk_52 + 153424975*uk_53 + 2789545*uk_54 + 11158180*uk_55 + 100423620*uk_56 + 298481315*uk_57 + 627647625*uk_58 + 11158180*uk_59 + 107*uk_6 + 50719*uk_60 + 202876*uk_61 + 1825884*uk_62 + 5426933*uk_63 + 11411775*uk_64 + 202876*uk_65 + 811504*uk_66 + 7303536*uk_67 + 21707732*uk_68 + 45647100*uk_69 + 225*uk_7 + 811504*uk_70 + 65731824*uk_71 + 195369588*uk_72 + 410823900*uk_73 + 7303536*uk_74 + 580681831*uk_75 + 1221059925*uk_76 + 21707732*uk_77 + 2567649375*uk_78 + 45647100*uk_79 + 4*uk_8 + 811504*uk_80 + 166375*uk_81 + 3025*uk_82 + 12100*uk_83 + 108900*uk_84 + 323675*uk_85 + 680625*uk_86 + 12100*uk_87 + 55*uk_88 + 220*uk_89 + 2572416961*uk_9 + 1980*uk_90 + 5885*uk_91 + 12375*uk_92 + 220*uk_93 + 880*uk_94 + 7920*uk_95 + 23540*uk_96 + 49500*uk_97 + 880*uk_98 + 71280*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 215820*uk_100 + 445500*uk_101 + 1980*uk_102 + 653455*uk_103 + 1348875*uk_104 + 5995*uk_105 + 2784375*uk_106 + 12375*uk_107 + 55*uk_108 + 39304*uk_109 + 1724446*uk_11 + 1156*uk_110 + 41616*uk_111 + 126004*uk_112 + 260100*uk_113 + 1156*uk_114 + 34*uk_115 + 1224*uk_116 + 3706*uk_117 + 7650*uk_118 + 34*uk_119 + 50719*uk_12 + 44064*uk_120 + 133416*uk_121 + 275400*uk_122 + 1224*uk_123 + 403954*uk_124 + 833850*uk_125 + 3706*uk_126 + 1721250*uk_127 + 7650*uk_128 + 34*uk_129 + 1825884*uk_13 + uk_130 + 36*uk_131 + 109*uk_132 + 225*uk_133 + uk_134 + 1296*uk_135 + 3924*uk_136 + 8100*uk_137 + 36*uk_138 + 11881*uk_139 + 5528371*uk_14 + 24525*uk_140 + 109*uk_141 + 50625*uk_142 + 225*uk_143 + uk_144 + 46656*uk_145 + 141264*uk_146 + 291600*uk_147 + 1296*uk_148 + 427716*uk_149 + 11411775*uk_15 + 882900*uk_150 + 3924*uk_151 + 1822500*uk_152 + 8100*uk_153 + 36*uk_154 + 1295029*uk_155 + 2673225*uk_156 + 11881*uk_157 + 5518125*uk_158 + 24525*uk_159 + 50719*uk_16 + 109*uk_160 + 11390625*uk_161 + 50625*uk_162 + 225*uk_163 + uk_164 + 3025*uk_17 + 1870*uk_18 + 55*uk_19 + 55*uk_2 + 1980*uk_20 + 5995*uk_21 + 12375*uk_22 + 55*uk_23 + 1156*uk_24 + 34*uk_25 + 1224*uk_26 + 3706*uk_27 + 7650*uk_28 + 34*uk_29 + 34*uk_3 + uk_30 + 36*uk_31 + 109*uk_32 + 225*uk_33 + uk_34 + 1296*uk_35 + 3924*uk_36 + 8100*uk_37 + 36*uk_38 + 11881*uk_39 + uk_4 + 24525*uk_40 + 109*uk_41 + 50625*uk_42 + 225*uk_43 + uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 87462176674*uk_47 + 2572416961*uk_48 + 92607010596*uk_49 + 36*uk_5 + 280393448749*uk_50 + 578793816225*uk_51 + 2572416961*uk_52 + 153424975*uk_53 + 94844530*uk_54 + 2789545*uk_55 + 100423620*uk_56 + 304060405*uk_57 + 627647625*uk_58 + 2789545*uk_59 + 109*uk_6 + 58631164*uk_60 + 1724446*uk_61 + 62080056*uk_62 + 187964614*uk_63 + 388000350*uk_64 + 1724446*uk_65 + 50719*uk_66 + 1825884*uk_67 + 5528371*uk_68 + 11411775*uk_69 + 225*uk_7 + 50719*uk_70 + 65731824*uk_71 + 199021356*uk_72 + 410823900*uk_73 + 1825884*uk_74 + 602592439*uk_75 + 1243883475*uk_76 + 5528371*uk_77 + 2567649375*uk_78 + 11411775*uk_79 + uk_8 + 50719*uk_80 + 166375*uk_81 + 102850*uk_82 + 3025*uk_83 + 108900*uk_84 + 329725*uk_85 + 680625*uk_86 + 3025*uk_87 + 63580*uk_88 + 1870*uk_89 + 2572416961*uk_9 + 67320*uk_90 + 203830*uk_91 + 420750*uk_92 + 1870*uk_93 + 55*uk_94 + 1980*uk_95 + 5995*uk_96 + 12375*uk_97 + 55*uk_98 + 71280*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 219780*uk_100 + 445500*uk_101 + 67320*uk_102 + 677655*uk_103 + 1373625*uk_104 + 207570*uk_105 + 2784375*uk_106 + 420750*uk_107 + 63580*uk_108 + 1092727*uk_109 + 5224057*uk_11 + 360706*uk_110 + 381924*uk_111 + 1177599*uk_112 + 2387025*uk_113 + 360706*uk_114 + 119068*uk_115 + 126072*uk_116 + 388722*uk_117 + 787950*uk_118 + 119068*uk_119 + 1724446*uk_12 + 133488*uk_120 + 411588*uk_121 + 834300*uk_122 + 126072*uk_123 + 1269063*uk_124 + 2572425*uk_125 + 388722*uk_126 + 5214375*uk_127 + 787950*uk_128 + 119068*uk_129 + 1825884*uk_13 + 39304*uk_130 + 41616*uk_131 + 128316*uk_132 + 260100*uk_133 + 39304*uk_134 + 44064*uk_135 + 135864*uk_136 + 275400*uk_137 + 41616*uk_138 + 418914*uk_139 + 5629809*uk_14 + 849150*uk_140 + 128316*uk_141 + 1721250*uk_142 + 260100*uk_143 + 39304*uk_144 + 46656*uk_145 + 143856*uk_146 + 291600*uk_147 + 44064*uk_148 + 443556*uk_149 + 11411775*uk_15 + 899100*uk_150 + 135864*uk_151 + 1822500*uk_152 + 275400*uk_153 + 41616*uk_154 + 1367631*uk_155 + 2772225*uk_156 + 418914*uk_157 + 5619375*uk_158 + 849150*uk_159 + 1724446*uk_16 + 128316*uk_160 + 11390625*uk_161 + 1721250*uk_162 + 260100*uk_163 + 39304*uk_164 + 3025*uk_17 + 5665*uk_18 + 1870*uk_19 + 55*uk_2 + 1980*uk_20 + 6105*uk_21 + 12375*uk_22 + 1870*uk_23 + 10609*uk_24 + 3502*uk_25 + 3708*uk_26 + 11433*uk_27 + 23175*uk_28 + 3502*uk_29 + 103*uk_3 + 1156*uk_30 + 1224*uk_31 + 3774*uk_32 + 7650*uk_33 + 1156*uk_34 + 1296*uk_35 + 3996*uk_36 + 8100*uk_37 + 1224*uk_38 + 12321*uk_39 + 34*uk_4 + 24975*uk_40 + 3774*uk_41 + 50625*uk_42 + 7650*uk_43 + 1156*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 264958946983*uk_47 + 87462176674*uk_48 + 92607010596*uk_49 + 36*uk_5 + 285538282671*uk_50 + 578793816225*uk_51 + 87462176674*uk_52 + 153424975*uk_53 + 287323135*uk_54 + 94844530*uk_55 + 100423620*uk_56 + 309639495*uk_57 + 627647625*uk_58 + 94844530*uk_59 + 111*uk_6 + 538077871*uk_60 + 177617938*uk_61 + 188066052*uk_62 + 579870327*uk_63 + 1175412825*uk_64 + 177617938*uk_65 + 58631164*uk_66 + 62080056*uk_67 + 191413506*uk_68 + 388000350*uk_69 + 225*uk_7 + 58631164*uk_70 + 65731824*uk_71 + 202673124*uk_72 + 410823900*uk_73 + 62080056*uk_74 + 624908799*uk_75 + 1266707025*uk_76 + 191413506*uk_77 + 2567649375*uk_78 + 388000350*uk_79 + 34*uk_8 + 58631164*uk_80 + 166375*uk_81 + 311575*uk_82 + 102850*uk_83 + 108900*uk_84 + 335775*uk_85 + 680625*uk_86 + 102850*uk_87 + 583495*uk_88 + 192610*uk_89 + 2572416961*uk_9 + 203940*uk_90 + 628815*uk_91 + 1274625*uk_92 + 192610*uk_93 + 63580*uk_94 + 67320*uk_95 + 207570*uk_96 + 420750*uk_97 + 63580*uk_98 + 71280*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 198880*uk_100 + 396000*uk_101 + 181280*uk_102 + 702295*uk_103 + 1398375*uk_104 + 640145*uk_105 + 2784375*uk_106 + 1274625*uk_107 + 583495*uk_108 + 857375*uk_109 + 4818305*uk_11 + 929575*uk_110 + 288800*uk_111 + 1019825*uk_112 + 2030625*uk_113 + 929575*uk_114 + 1007855*uk_115 + 313120*uk_116 + 1105705*uk_117 + 2201625*uk_118 + 1007855*uk_119 + 5224057*uk_12 + 97280*uk_120 + 343520*uk_121 + 684000*uk_122 + 313120*uk_123 + 1213055*uk_124 + 2415375*uk_125 + 1105705*uk_126 + 4809375*uk_127 + 2201625*uk_128 + 1007855*uk_129 + 1623008*uk_13 + 1092727*uk_130 + 339488*uk_131 + 1198817*uk_132 + 2387025*uk_133 + 1092727*uk_134 + 105472*uk_135 + 372448*uk_136 + 741600*uk_137 + 339488*uk_138 + 1315207*uk_139 + 5731247*uk_14 + 2618775*uk_140 + 1198817*uk_141 + 5214375*uk_142 + 2387025*uk_143 + 1092727*uk_144 + 32768*uk_145 + 115712*uk_146 + 230400*uk_147 + 105472*uk_148 + 408608*uk_149 + 11411775*uk_15 + 813600*uk_150 + 372448*uk_151 + 1620000*uk_152 + 741600*uk_153 + 339488*uk_154 + 1442897*uk_155 + 2873025*uk_156 + 1315207*uk_157 + 5720625*uk_158 + 2618775*uk_159 + 5224057*uk_16 + 1198817*uk_160 + 11390625*uk_161 + 5214375*uk_162 + 2387025*uk_163 + 1092727*uk_164 + 3025*uk_17 + 5225*uk_18 + 5665*uk_19 + 55*uk_2 + 1760*uk_20 + 6215*uk_21 + 12375*uk_22 + 5665*uk_23 + 9025*uk_24 + 9785*uk_25 + 3040*uk_26 + 10735*uk_27 + 21375*uk_28 + 9785*uk_29 + 95*uk_3 + 10609*uk_30 + 3296*uk_31 + 11639*uk_32 + 23175*uk_33 + 10609*uk_34 + 1024*uk_35 + 3616*uk_36 + 7200*uk_37 + 3296*uk_38 + 12769*uk_39 + 103*uk_4 + 25425*uk_40 + 11639*uk_41 + 50625*uk_42 + 23175*uk_43 + 10609*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 244379611295*uk_47 + 264958946983*uk_48 + 82317342752*uk_49 + 32*uk_5 + 290683116593*uk_50 + 578793816225*uk_51 + 264958946983*uk_52 + 153424975*uk_53 + 265006775*uk_54 + 287323135*uk_55 + 89265440*uk_56 + 315218585*uk_57 + 627647625*uk_58 + 287323135*uk_59 + 113*uk_6 + 457738975*uk_60 + 496285415*uk_61 + 154185760*uk_62 + 544468465*uk_63 + 1084118625*uk_64 + 496285415*uk_65 + 538077871*uk_66 + 167169824*uk_67 + 590318441*uk_68 + 1175412825*uk_69 + 225*uk_7 + 538077871*uk_70 + 51936256*uk_71 + 183399904*uk_72 + 365176800*uk_73 + 167169824*uk_74 + 647630911*uk_75 + 1289530575*uk_76 + 590318441*uk_77 + 2567649375*uk_78 + 1175412825*uk_79 + 103*uk_8 + 538077871*uk_80 + 166375*uk_81 + 287375*uk_82 + 311575*uk_83 + 96800*uk_84 + 341825*uk_85 + 680625*uk_86 + 311575*uk_87 + 496375*uk_88 + 538175*uk_89 + 2572416961*uk_9 + 167200*uk_90 + 590425*uk_91 + 1175625*uk_92 + 538175*uk_93 + 583495*uk_94 + 181280*uk_95 + 640145*uk_96 + 1274625*uk_97 + 583495*uk_98 + 56320*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 177100*uk_100 + 346500*uk_101 + 146300*uk_102 + 727375*uk_103 + 1423125*uk_104 + 600875*uk_105 + 2784375*uk_106 + 1175625*uk_107 + 496375*uk_108 + 64*uk_109 + 202876*uk_11 + 1520*uk_110 + 448*uk_111 + 1840*uk_112 + 3600*uk_113 + 1520*uk_114 + 36100*uk_115 + 10640*uk_116 + 43700*uk_117 + 85500*uk_118 + 36100*uk_119 + 4818305*uk_12 + 3136*uk_120 + 12880*uk_121 + 25200*uk_122 + 10640*uk_123 + 52900*uk_124 + 103500*uk_125 + 43700*uk_126 + 202500*uk_127 + 85500*uk_128 + 36100*uk_129 + 1420132*uk_13 + 857375*uk_130 + 252700*uk_131 + 1037875*uk_132 + 2030625*uk_133 + 857375*uk_134 + 74480*uk_135 + 305900*uk_136 + 598500*uk_137 + 252700*uk_138 + 1256375*uk_139 + 5832685*uk_14 + 2458125*uk_140 + 1037875*uk_141 + 4809375*uk_142 + 2030625*uk_143 + 857375*uk_144 + 21952*uk_145 + 90160*uk_146 + 176400*uk_147 + 74480*uk_148 + 370300*uk_149 + 11411775*uk_15 + 724500*uk_150 + 305900*uk_151 + 1417500*uk_152 + 598500*uk_153 + 252700*uk_154 + 1520875*uk_155 + 2975625*uk_156 + 1256375*uk_157 + 5821875*uk_158 + 2458125*uk_159 + 4818305*uk_16 + 1037875*uk_160 + 11390625*uk_161 + 4809375*uk_162 + 2030625*uk_163 + 857375*uk_164 + 3025*uk_17 + 220*uk_18 + 5225*uk_19 + 55*uk_2 + 1540*uk_20 + 6325*uk_21 + 12375*uk_22 + 5225*uk_23 + 16*uk_24 + 380*uk_25 + 112*uk_26 + 460*uk_27 + 900*uk_28 + 380*uk_29 + 4*uk_3 + 9025*uk_30 + 2660*uk_31 + 10925*uk_32 + 21375*uk_33 + 9025*uk_34 + 784*uk_35 + 3220*uk_36 + 6300*uk_37 + 2660*uk_38 + 13225*uk_39 + 95*uk_4 + 25875*uk_40 + 10925*uk_41 + 50625*uk_42 + 21375*uk_43 + 9025*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 10289667844*uk_47 + 244379611295*uk_48 + 72027674908*uk_49 + 28*uk_5 + 295827950515*uk_50 + 578793816225*uk_51 + 244379611295*uk_52 + 153424975*uk_53 + 11158180*uk_54 + 265006775*uk_55 + 78107260*uk_56 + 320797675*uk_57 + 627647625*uk_58 + 265006775*uk_59 + 115*uk_6 + 811504*uk_60 + 19273220*uk_61 + 5680528*uk_62 + 23330740*uk_63 + 45647100*uk_64 + 19273220*uk_65 + 457738975*uk_66 + 134912540*uk_67 + 554105075*uk_68 + 1084118625*uk_69 + 225*uk_7 + 457738975*uk_70 + 39763696*uk_71 + 163315180*uk_72 + 319529700*uk_73 + 134912540*uk_74 + 670758775*uk_75 + 1312354125*uk_76 + 554105075*uk_77 + 2567649375*uk_78 + 1084118625*uk_79 + 95*uk_8 + 457738975*uk_80 + 166375*uk_81 + 12100*uk_82 + 287375*uk_83 + 84700*uk_84 + 347875*uk_85 + 680625*uk_86 + 287375*uk_87 + 880*uk_88 + 20900*uk_89 + 2572416961*uk_9 + 6160*uk_90 + 25300*uk_91 + 49500*uk_92 + 20900*uk_93 + 496375*uk_94 + 146300*uk_95 + 600875*uk_96 + 1175625*uk_97 + 496375*uk_98 + 43120*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 205920*uk_100 + 396000*uk_101 + 7040*uk_102 + 752895*uk_103 + 1447875*uk_104 + 25740*uk_105 + 2784375*uk_106 + 49500*uk_107 + 880*uk_108 + 195112*uk_109 + 2941702*uk_11 + 13456*uk_110 + 107648*uk_111 + 393588*uk_112 + 756900*uk_113 + 13456*uk_114 + 928*uk_115 + 7424*uk_116 + 27144*uk_117 + 52200*uk_118 + 928*uk_119 + 202876*uk_12 + 59392*uk_120 + 217152*uk_121 + 417600*uk_122 + 7424*uk_123 + 793962*uk_124 + 1526850*uk_125 + 27144*uk_126 + 2936250*uk_127 + 52200*uk_128 + 928*uk_129 + 1623008*uk_13 + 64*uk_130 + 512*uk_131 + 1872*uk_132 + 3600*uk_133 + 64*uk_134 + 4096*uk_135 + 14976*uk_136 + 28800*uk_137 + 512*uk_138 + 54756*uk_139 + 5934123*uk_14 + 105300*uk_140 + 1872*uk_141 + 202500*uk_142 + 3600*uk_143 + 64*uk_144 + 32768*uk_145 + 119808*uk_146 + 230400*uk_147 + 4096*uk_148 + 438048*uk_149 + 11411775*uk_15 + 842400*uk_150 + 14976*uk_151 + 1620000*uk_152 + 28800*uk_153 + 512*uk_154 + 1601613*uk_155 + 3080025*uk_156 + 54756*uk_157 + 5923125*uk_158 + 105300*uk_159 + 202876*uk_16 + 1872*uk_160 + 11390625*uk_161 + 202500*uk_162 + 3600*uk_163 + 64*uk_164 + 3025*uk_17 + 3190*uk_18 + 220*uk_19 + 55*uk_2 + 1760*uk_20 + 6435*uk_21 + 12375*uk_22 + 220*uk_23 + 3364*uk_24 + 232*uk_25 + 1856*uk_26 + 6786*uk_27 + 13050*uk_28 + 232*uk_29 + 58*uk_3 + 16*uk_30 + 128*uk_31 + 468*uk_32 + 900*uk_33 + 16*uk_34 + 1024*uk_35 + 3744*uk_36 + 7200*uk_37 + 128*uk_38 + 13689*uk_39 + 4*uk_4 + 26325*uk_40 + 468*uk_41 + 50625*uk_42 + 900*uk_43 + 16*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 149200183738*uk_47 + 10289667844*uk_48 + 82317342752*uk_49 + 32*uk_5 + 300972784437*uk_50 + 578793816225*uk_51 + 10289667844*uk_52 + 153424975*uk_53 + 161793610*uk_54 + 11158180*uk_55 + 89265440*uk_56 + 326376765*uk_57 + 627647625*uk_58 + 11158180*uk_59 + 117*uk_6 + 170618716*uk_60 + 11766808*uk_61 + 94134464*uk_62 + 344179134*uk_63 + 661882950*uk_64 + 11766808*uk_65 + 811504*uk_66 + 6492032*uk_67 + 23736492*uk_68 + 45647100*uk_69 + 225*uk_7 + 811504*uk_70 + 51936256*uk_71 + 189891936*uk_72 + 365176800*uk_73 + 6492032*uk_74 + 694292391*uk_75 + 1335177675*uk_76 + 23736492*uk_77 + 2567649375*uk_78 + 45647100*uk_79 + 4*uk_8 + 811504*uk_80 + 166375*uk_81 + 175450*uk_82 + 12100*uk_83 + 96800*uk_84 + 353925*uk_85 + 680625*uk_86 + 12100*uk_87 + 185020*uk_88 + 12760*uk_89 + 2572416961*uk_9 + 102080*uk_90 + 373230*uk_91 + 717750*uk_92 + 12760*uk_93 + 880*uk_94 + 7040*uk_95 + 25740*uk_96 + 49500*uk_97 + 880*uk_98 + 56320*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 183260*uk_100 + 346500*uk_101 + 89320*uk_102 + 778855*uk_103 + 1472625*uk_104 + 379610*uk_105 + 2784375*uk_106 + 717750*uk_107 + 185020*uk_108 + 15625*uk_109 + 1267975*uk_11 + 36250*uk_110 + 17500*uk_111 + 74375*uk_112 + 140625*uk_113 + 36250*uk_114 + 84100*uk_115 + 40600*uk_116 + 172550*uk_117 + 326250*uk_118 + 84100*uk_119 + 2941702*uk_12 + 19600*uk_120 + 83300*uk_121 + 157500*uk_122 + 40600*uk_123 + 354025*uk_124 + 669375*uk_125 + 172550*uk_126 + 1265625*uk_127 + 326250*uk_128 + 84100*uk_129 + 1420132*uk_13 + 195112*uk_130 + 94192*uk_131 + 400316*uk_132 + 756900*uk_133 + 195112*uk_134 + 45472*uk_135 + 193256*uk_136 + 365400*uk_137 + 94192*uk_138 + 821338*uk_139 + 6035561*uk_14 + 1552950*uk_140 + 400316*uk_141 + 2936250*uk_142 + 756900*uk_143 + 195112*uk_144 + 21952*uk_145 + 93296*uk_146 + 176400*uk_147 + 45472*uk_148 + 396508*uk_149 + 11411775*uk_15 + 749700*uk_150 + 193256*uk_151 + 1417500*uk_152 + 365400*uk_153 + 94192*uk_154 + 1685159*uk_155 + 3186225*uk_156 + 821338*uk_157 + 6024375*uk_158 + 1552950*uk_159 + 2941702*uk_16 + 400316*uk_160 + 11390625*uk_161 + 2936250*uk_162 + 756900*uk_163 + 195112*uk_164 + 3025*uk_17 + 1375*uk_18 + 3190*uk_19 + 55*uk_2 + 1540*uk_20 + 6545*uk_21 + 12375*uk_22 + 3190*uk_23 + 625*uk_24 + 1450*uk_25 + 700*uk_26 + 2975*uk_27 + 5625*uk_28 + 1450*uk_29 + 25*uk_3 + 3364*uk_30 + 1624*uk_31 + 6902*uk_32 + 13050*uk_33 + 3364*uk_34 + 784*uk_35 + 3332*uk_36 + 6300*uk_37 + 1624*uk_38 + 14161*uk_39 + 58*uk_4 + 26775*uk_40 + 6902*uk_41 + 50625*uk_42 + 13050*uk_43 + 3364*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 64310424025*uk_47 + 149200183738*uk_48 + 72027674908*uk_49 + 28*uk_5 + 306117618359*uk_50 + 578793816225*uk_51 + 149200183738*uk_52 + 153424975*uk_53 + 69738625*uk_54 + 161793610*uk_55 + 78107260*uk_56 + 331955855*uk_57 + 627647625*uk_58 + 161793610*uk_59 + 119*uk_6 + 31699375*uk_60 + 73542550*uk_61 + 35503300*uk_62 + 150889025*uk_63 + 285294375*uk_64 + 73542550*uk_65 + 170618716*uk_66 + 82367656*uk_67 + 350062538*uk_68 + 661882950*uk_69 + 225*uk_7 + 170618716*uk_70 + 39763696*uk_71 + 168995708*uk_72 + 319529700*uk_73 + 82367656*uk_74 + 718231759*uk_75 + 1358001225*uk_76 + 350062538*uk_77 + 2567649375*uk_78 + 661882950*uk_79 + 58*uk_8 + 170618716*uk_80 + 166375*uk_81 + 75625*uk_82 + 175450*uk_83 + 84700*uk_84 + 359975*uk_85 + 680625*uk_86 + 175450*uk_87 + 34375*uk_88 + 79750*uk_89 + 2572416961*uk_9 + 38500*uk_90 + 163625*uk_91 + 309375*uk_92 + 79750*uk_93 + 185020*uk_94 + 89320*uk_95 + 379610*uk_96 + 717750*uk_97 + 185020*uk_98 + 43120*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 186340*uk_100 + 346500*uk_101 + 38500*uk_102 + 805255*uk_103 + 1497375*uk_104 + 166375*uk_105 + 2784375*uk_106 + 309375*uk_107 + 34375*uk_108 + 8000*uk_109 + 1014380*uk_11 + 10000*uk_110 + 11200*uk_111 + 48400*uk_112 + 90000*uk_113 + 10000*uk_114 + 12500*uk_115 + 14000*uk_116 + 60500*uk_117 + 112500*uk_118 + 12500*uk_119 + 1267975*uk_12 + 15680*uk_120 + 67760*uk_121 + 126000*uk_122 + 14000*uk_123 + 292820*uk_124 + 544500*uk_125 + 60500*uk_126 + 1012500*uk_127 + 112500*uk_128 + 12500*uk_129 + 1420132*uk_13 + 15625*uk_130 + 17500*uk_131 + 75625*uk_132 + 140625*uk_133 + 15625*uk_134 + 19600*uk_135 + 84700*uk_136 + 157500*uk_137 + 17500*uk_138 + 366025*uk_139 + 6136999*uk_14 + 680625*uk_140 + 75625*uk_141 + 1265625*uk_142 + 140625*uk_143 + 15625*uk_144 + 21952*uk_145 + 94864*uk_146 + 176400*uk_147 + 19600*uk_148 + 409948*uk_149 + 11411775*uk_15 + 762300*uk_150 + 84700*uk_151 + 1417500*uk_152 + 157500*uk_153 + 17500*uk_154 + 1771561*uk_155 + 3294225*uk_156 + 366025*uk_157 + 6125625*uk_158 + 680625*uk_159 + 1267975*uk_16 + 75625*uk_160 + 11390625*uk_161 + 1265625*uk_162 + 140625*uk_163 + 15625*uk_164 + 3025*uk_17 + 1100*uk_18 + 1375*uk_19 + 55*uk_2 + 1540*uk_20 + 6655*uk_21 + 12375*uk_22 + 1375*uk_23 + 400*uk_24 + 500*uk_25 + 560*uk_26 + 2420*uk_27 + 4500*uk_28 + 500*uk_29 + 20*uk_3 + 625*uk_30 + 700*uk_31 + 3025*uk_32 + 5625*uk_33 + 625*uk_34 + 784*uk_35 + 3388*uk_36 + 6300*uk_37 + 700*uk_38 + 14641*uk_39 + 25*uk_4 + 27225*uk_40 + 3025*uk_41 + 50625*uk_42 + 5625*uk_43 + 625*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 51448339220*uk_47 + 64310424025*uk_48 + 72027674908*uk_49 + 28*uk_5 + 311262452281*uk_50 + 578793816225*uk_51 + 64310424025*uk_52 + 153424975*uk_53 + 55790900*uk_54 + 69738625*uk_55 + 78107260*uk_56 + 337534945*uk_57 + 627647625*uk_58 + 69738625*uk_59 + 121*uk_6 + 20287600*uk_60 + 25359500*uk_61 + 28402640*uk_62 + 122739980*uk_63 + 228235500*uk_64 + 25359500*uk_65 + 31699375*uk_66 + 35503300*uk_67 + 153424975*uk_68 + 285294375*uk_69 + 225*uk_7 + 31699375*uk_70 + 39763696*uk_71 + 171835972*uk_72 + 319529700*uk_73 + 35503300*uk_74 + 742576879*uk_75 + 1380824775*uk_76 + 153424975*uk_77 + 2567649375*uk_78 + 285294375*uk_79 + 25*uk_8 + 31699375*uk_80 + 166375*uk_81 + 60500*uk_82 + 75625*uk_83 + 84700*uk_84 + 366025*uk_85 + 680625*uk_86 + 75625*uk_87 + 22000*uk_88 + 27500*uk_89 + 2572416961*uk_9 + 30800*uk_90 + 133100*uk_91 + 247500*uk_92 + 27500*uk_93 + 34375*uk_94 + 38500*uk_95 + 166375*uk_96 + 309375*uk_97 + 34375*uk_98 + 43120*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 189420*uk_100 + 346500*uk_101 + 30800*uk_102 + 832095*uk_103 + 1522125*uk_104 + 135300*uk_105 + 2784375*uk_106 + 247500*uk_107 + 22000*uk_108 + 79507*uk_109 + 2180917*uk_11 + 36980*uk_110 + 51772*uk_111 + 227427*uk_112 + 416025*uk_113 + 36980*uk_114 + 17200*uk_115 + 24080*uk_116 + 105780*uk_117 + 193500*uk_118 + 17200*uk_119 + 1014380*uk_12 + 33712*uk_120 + 148092*uk_121 + 270900*uk_122 + 24080*uk_123 + 650547*uk_124 + 1190025*uk_125 + 105780*uk_126 + 2176875*uk_127 + 193500*uk_128 + 17200*uk_129 + 1420132*uk_13 + 8000*uk_130 + 11200*uk_131 + 49200*uk_132 + 90000*uk_133 + 8000*uk_134 + 15680*uk_135 + 68880*uk_136 + 126000*uk_137 + 11200*uk_138 + 302580*uk_139 + 6238437*uk_14 + 553500*uk_140 + 49200*uk_141 + 1012500*uk_142 + 90000*uk_143 + 8000*uk_144 + 21952*uk_145 + 96432*uk_146 + 176400*uk_147 + 15680*uk_148 + 423612*uk_149 + 11411775*uk_15 + 774900*uk_150 + 68880*uk_151 + 1417500*uk_152 + 126000*uk_153 + 11200*uk_154 + 1860867*uk_155 + 3404025*uk_156 + 302580*uk_157 + 6226875*uk_158 + 553500*uk_159 + 1014380*uk_16 + 49200*uk_160 + 11390625*uk_161 + 1012500*uk_162 + 90000*uk_163 + 8000*uk_164 + 3025*uk_17 + 2365*uk_18 + 1100*uk_19 + 55*uk_2 + 1540*uk_20 + 6765*uk_21 + 12375*uk_22 + 1100*uk_23 + 1849*uk_24 + 860*uk_25 + 1204*uk_26 + 5289*uk_27 + 9675*uk_28 + 860*uk_29 + 43*uk_3 + 400*uk_30 + 560*uk_31 + 2460*uk_32 + 4500*uk_33 + 400*uk_34 + 784*uk_35 + 3444*uk_36 + 6300*uk_37 + 560*uk_38 + 15129*uk_39 + 20*uk_4 + 27675*uk_40 + 2460*uk_41 + 50625*uk_42 + 4500*uk_43 + 400*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 110613929323*uk_47 + 51448339220*uk_48 + 72027674908*uk_49 + 28*uk_5 + 316407286203*uk_50 + 578793816225*uk_51 + 51448339220*uk_52 + 153424975*uk_53 + 119950435*uk_54 + 55790900*uk_55 + 78107260*uk_56 + 343114035*uk_57 + 627647625*uk_58 + 55790900*uk_59 + 123*uk_6 + 93779431*uk_60 + 43618340*uk_61 + 61065676*uk_62 + 268252791*uk_63 + 490706325*uk_64 + 43618340*uk_65 + 20287600*uk_66 + 28402640*uk_67 + 124768740*uk_68 + 228235500*uk_69 + 225*uk_7 + 20287600*uk_70 + 39763696*uk_71 + 174676236*uk_72 + 319529700*uk_73 + 28402640*uk_74 + 767327751*uk_75 + 1403648325*uk_76 + 124768740*uk_77 + 2567649375*uk_78 + 228235500*uk_79 + 20*uk_8 + 20287600*uk_80 + 166375*uk_81 + 130075*uk_82 + 60500*uk_83 + 84700*uk_84 + 372075*uk_85 + 680625*uk_86 + 60500*uk_87 + 101695*uk_88 + 47300*uk_89 + 2572416961*uk_9 + 66220*uk_90 + 290895*uk_91 + 532125*uk_92 + 47300*uk_93 + 22000*uk_94 + 30800*uk_95 + 135300*uk_96 + 247500*uk_97 + 22000*uk_98 + 43120*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 192500*uk_100 + 346500*uk_101 + 66220*uk_102 + 859375*uk_103 + 1546875*uk_104 + 295625*uk_105 + 2784375*uk_106 + 532125*uk_107 + 101695*uk_108 + 830584*uk_109 + 4767586*uk_11 + 379948*uk_110 + 247408*uk_111 + 1104500*uk_112 + 1988100*uk_113 + 379948*uk_114 + 173806*uk_115 + 113176*uk_116 + 505250*uk_117 + 909450*uk_118 + 173806*uk_119 + 2180917*uk_12 + 73696*uk_120 + 329000*uk_121 + 592200*uk_122 + 113176*uk_123 + 1468750*uk_124 + 2643750*uk_125 + 505250*uk_126 + 4758750*uk_127 + 909450*uk_128 + 173806*uk_129 + 1420132*uk_13 + 79507*uk_130 + 51772*uk_131 + 231125*uk_132 + 416025*uk_133 + 79507*uk_134 + 33712*uk_135 + 150500*uk_136 + 270900*uk_137 + 51772*uk_138 + 671875*uk_139 + 6339875*uk_14 + 1209375*uk_140 + 231125*uk_141 + 2176875*uk_142 + 416025*uk_143 + 79507*uk_144 + 21952*uk_145 + 98000*uk_146 + 176400*uk_147 + 33712*uk_148 + 437500*uk_149 + 11411775*uk_15 + 787500*uk_150 + 150500*uk_151 + 1417500*uk_152 + 270900*uk_153 + 51772*uk_154 + 1953125*uk_155 + 3515625*uk_156 + 671875*uk_157 + 6328125*uk_158 + 1209375*uk_159 + 2180917*uk_16 + 231125*uk_160 + 11390625*uk_161 + 2176875*uk_162 + 416025*uk_163 + 79507*uk_164 + 3025*uk_17 + 5170*uk_18 + 2365*uk_19 + 55*uk_2 + 1540*uk_20 + 6875*uk_21 + 12375*uk_22 + 2365*uk_23 + 8836*uk_24 + 4042*uk_25 + 2632*uk_26 + 11750*uk_27 + 21150*uk_28 + 4042*uk_29 + 94*uk_3 + 1849*uk_30 + 1204*uk_31 + 5375*uk_32 + 9675*uk_33 + 1849*uk_34 + 784*uk_35 + 3500*uk_36 + 6300*uk_37 + 1204*uk_38 + 15625*uk_39 + 43*uk_4 + 28125*uk_40 + 5375*uk_41 + 50625*uk_42 + 9675*uk_43 + 1849*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 241807194334*uk_47 + 110613929323*uk_48 + 72027674908*uk_49 + 28*uk_5 + 321552120125*uk_50 + 578793816225*uk_51 + 110613929323*uk_52 + 153424975*uk_53 + 262217230*uk_54 + 119950435*uk_55 + 78107260*uk_56 + 348693125*uk_57 + 627647625*uk_58 + 119950435*uk_59 + 125*uk_6 + 448153084*uk_60 + 205006198*uk_61 + 133492408*uk_62 + 595948250*uk_63 + 1072706850*uk_64 + 205006198*uk_65 + 93779431*uk_66 + 61065676*uk_67 + 272614625*uk_68 + 490706325*uk_69 + 225*uk_7 + 93779431*uk_70 + 39763696*uk_71 + 177516500*uk_72 + 319529700*uk_73 + 61065676*uk_74 + 792484375*uk_75 + 1426471875*uk_76 + 272614625*uk_77 + 2567649375*uk_78 + 490706325*uk_79 + 43*uk_8 + 93779431*uk_80 + 166375*uk_81 + 284350*uk_82 + 130075*uk_83 + 84700*uk_84 + 378125*uk_85 + 680625*uk_86 + 130075*uk_87 + 485980*uk_88 + 222310*uk_89 + 2572416961*uk_9 + 144760*uk_90 + 646250*uk_91 + 1163250*uk_92 + 222310*uk_93 + 101695*uk_94 + 66220*uk_95 + 295625*uk_96 + 532125*uk_97 + 101695*uk_98 + 43120*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 167640*uk_100 + 297000*uk_101 + 124080*uk_102 + 887095*uk_103 + 1571625*uk_104 + 656590*uk_105 + 2784375*uk_106 + 1163250*uk_107 + 485980*uk_108 + 97336*uk_109 + 2333074*uk_11 + 198904*uk_110 + 50784*uk_111 + 268732*uk_112 + 476100*uk_113 + 198904*uk_114 + 406456*uk_115 + 103776*uk_116 + 549148*uk_117 + 972900*uk_118 + 406456*uk_119 + 4767586*uk_12 + 26496*uk_120 + 140208*uk_121 + 248400*uk_122 + 103776*uk_123 + 741934*uk_124 + 1314450*uk_125 + 549148*uk_126 + 2328750*uk_127 + 972900*uk_128 + 406456*uk_129 + 1217256*uk_13 + 830584*uk_130 + 212064*uk_131 + 1122172*uk_132 + 1988100*uk_133 + 830584*uk_134 + 54144*uk_135 + 286512*uk_136 + 507600*uk_137 + 212064*uk_138 + 1516126*uk_139 + 6441313*uk_14 + 2686050*uk_140 + 1122172*uk_141 + 4758750*uk_142 + 1988100*uk_143 + 830584*uk_144 + 13824*uk_145 + 73152*uk_146 + 129600*uk_147 + 54144*uk_148 + 387096*uk_149 + 11411775*uk_15 + 685800*uk_150 + 286512*uk_151 + 1215000*uk_152 + 507600*uk_153 + 212064*uk_154 + 2048383*uk_155 + 3629025*uk_156 + 1516126*uk_157 + 6429375*uk_158 + 2686050*uk_159 + 4767586*uk_16 + 1122172*uk_160 + 11390625*uk_161 + 4758750*uk_162 + 1988100*uk_163 + 830584*uk_164 + 3025*uk_17 + 2530*uk_18 + 5170*uk_19 + 55*uk_2 + 1320*uk_20 + 6985*uk_21 + 12375*uk_22 + 5170*uk_23 + 2116*uk_24 + 4324*uk_25 + 1104*uk_26 + 5842*uk_27 + 10350*uk_28 + 4324*uk_29 + 46*uk_3 + 8836*uk_30 + 2256*uk_31 + 11938*uk_32 + 21150*uk_33 + 8836*uk_34 + 576*uk_35 + 3048*uk_36 + 5400*uk_37 + 2256*uk_38 + 16129*uk_39 + 94*uk_4 + 28575*uk_40 + 11938*uk_41 + 50625*uk_42 + 21150*uk_43 + 8836*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 118331180206*uk_47 + 241807194334*uk_48 + 61738007064*uk_49 + 24*uk_5 + 326696954047*uk_50 + 578793816225*uk_51 + 241807194334*uk_52 + 153424975*uk_53 + 128319070*uk_54 + 262217230*uk_55 + 66949080*uk_56 + 354272215*uk_57 + 627647625*uk_58 + 262217230*uk_59 + 127*uk_6 + 107321404*uk_60 + 219308956*uk_61 + 55993776*uk_62 + 296300398*uk_63 + 524941650*uk_64 + 219308956*uk_65 + 448153084*uk_66 + 114422064*uk_67 + 605483422*uk_68 + 1072706850*uk_69 + 225*uk_7 + 448153084*uk_70 + 29214144*uk_71 + 154591512*uk_72 + 273882600*uk_73 + 114422064*uk_74 + 818046751*uk_75 + 1449295425*uk_76 + 605483422*uk_77 + 2567649375*uk_78 + 1072706850*uk_79 + 94*uk_8 + 448153084*uk_80 + 166375*uk_81 + 139150*uk_82 + 284350*uk_83 + 72600*uk_84 + 384175*uk_85 + 680625*uk_86 + 284350*uk_87 + 116380*uk_88 + 237820*uk_89 + 2572416961*uk_9 + 60720*uk_90 + 321310*uk_91 + 569250*uk_92 + 237820*uk_93 + 485980*uk_94 + 124080*uk_95 + 656590*uk_96 + 1163250*uk_97 + 485980*uk_98 + 31680*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 170280*uk_100 + 297000*uk_101 + 60720*uk_102 + 915255*uk_103 + 1596375*uk_104 + 326370*uk_105 + 2784375*uk_106 + 569250*uk_107 + 116380*uk_108 + 10648*uk_109 + 1115818*uk_11 + 22264*uk_110 + 11616*uk_111 + 62436*uk_112 + 108900*uk_113 + 22264*uk_114 + 46552*uk_115 + 24288*uk_116 + 130548*uk_117 + 227700*uk_118 + 46552*uk_119 + 2333074*uk_12 + 12672*uk_120 + 68112*uk_121 + 118800*uk_122 + 24288*uk_123 + 366102*uk_124 + 638550*uk_125 + 130548*uk_126 + 1113750*uk_127 + 227700*uk_128 + 46552*uk_129 + 1217256*uk_13 + 97336*uk_130 + 50784*uk_131 + 272964*uk_132 + 476100*uk_133 + 97336*uk_134 + 26496*uk_135 + 142416*uk_136 + 248400*uk_137 + 50784*uk_138 + 765486*uk_139 + 6542751*uk_14 + 1335150*uk_140 + 272964*uk_141 + 2328750*uk_142 + 476100*uk_143 + 97336*uk_144 + 13824*uk_145 + 74304*uk_146 + 129600*uk_147 + 26496*uk_148 + 399384*uk_149 + 11411775*uk_15 + 696600*uk_150 + 142416*uk_151 + 1215000*uk_152 + 248400*uk_153 + 50784*uk_154 + 2146689*uk_155 + 3744225*uk_156 + 765486*uk_157 + 6530625*uk_158 + 1335150*uk_159 + 2333074*uk_16 + 272964*uk_160 + 11390625*uk_161 + 2328750*uk_162 + 476100*uk_163 + 97336*uk_164 + 3025*uk_17 + 1210*uk_18 + 2530*uk_19 + 55*uk_2 + 1320*uk_20 + 7095*uk_21 + 12375*uk_22 + 2530*uk_23 + 484*uk_24 + 1012*uk_25 + 528*uk_26 + 2838*uk_27 + 4950*uk_28 + 1012*uk_29 + 22*uk_3 + 2116*uk_30 + 1104*uk_31 + 5934*uk_32 + 10350*uk_33 + 2116*uk_34 + 576*uk_35 + 3096*uk_36 + 5400*uk_37 + 1104*uk_38 + 16641*uk_39 + 46*uk_4 + 29025*uk_40 + 5934*uk_41 + 50625*uk_42 + 10350*uk_43 + 2116*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 56593173142*uk_47 + 118331180206*uk_48 + 61738007064*uk_49 + 24*uk_5 + 331841787969*uk_50 + 578793816225*uk_51 + 118331180206*uk_52 + 153424975*uk_53 + 61369990*uk_54 + 128319070*uk_55 + 66949080*uk_56 + 359851305*uk_57 + 627647625*uk_58 + 128319070*uk_59 + 129*uk_6 + 24547996*uk_60 + 51327628*uk_61 + 26779632*uk_62 + 143940522*uk_63 + 251059050*uk_64 + 51327628*uk_65 + 107321404*uk_66 + 55993776*uk_67 + 300966546*uk_68 + 524941650*uk_69 + 225*uk_7 + 107321404*uk_70 + 29214144*uk_71 + 157026024*uk_72 + 273882600*uk_73 + 55993776*uk_74 + 844014879*uk_75 + 1472118975*uk_76 + 300966546*uk_77 + 2567649375*uk_78 + 524941650*uk_79 + 46*uk_8 + 107321404*uk_80 + 166375*uk_81 + 66550*uk_82 + 139150*uk_83 + 72600*uk_84 + 390225*uk_85 + 680625*uk_86 + 139150*uk_87 + 26620*uk_88 + 55660*uk_89 + 2572416961*uk_9 + 29040*uk_90 + 156090*uk_91 + 272250*uk_92 + 55660*uk_93 + 116380*uk_94 + 60720*uk_95 + 326370*uk_96 + 569250*uk_97 + 116380*uk_98 + 31680*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 172920*uk_100 + 297000*uk_101 + 29040*uk_102 + 943855*uk_103 + 1621125*uk_104 + 158510*uk_105 + 2784375*uk_106 + 272250*uk_107 + 26620*uk_108 + 10648*uk_109 + 1115818*uk_11 + 10648*uk_110 + 11616*uk_111 + 63404*uk_112 + 108900*uk_113 + 10648*uk_114 + 10648*uk_115 + 11616*uk_116 + 63404*uk_117 + 108900*uk_118 + 10648*uk_119 + 1115818*uk_12 + 12672*uk_120 + 69168*uk_121 + 118800*uk_122 + 11616*uk_123 + 377542*uk_124 + 648450*uk_125 + 63404*uk_126 + 1113750*uk_127 + 108900*uk_128 + 10648*uk_129 + 1217256*uk_13 + 10648*uk_130 + 11616*uk_131 + 63404*uk_132 + 108900*uk_133 + 10648*uk_134 + 12672*uk_135 + 69168*uk_136 + 118800*uk_137 + 11616*uk_138 + 377542*uk_139 + 6644189*uk_14 + 648450*uk_140 + 63404*uk_141 + 1113750*uk_142 + 108900*uk_143 + 10648*uk_144 + 13824*uk_145 + 75456*uk_146 + 129600*uk_147 + 12672*uk_148 + 411864*uk_149 + 11411775*uk_15 + 707400*uk_150 + 69168*uk_151 + 1215000*uk_152 + 118800*uk_153 + 11616*uk_154 + 2248091*uk_155 + 3861225*uk_156 + 377542*uk_157 + 6631875*uk_158 + 648450*uk_159 + 1115818*uk_16 + 63404*uk_160 + 11390625*uk_161 + 1113750*uk_162 + 108900*uk_163 + 10648*uk_164 + 3025*uk_17 + 1210*uk_18 + 1210*uk_19 + 55*uk_2 + 1320*uk_20 + 7205*uk_21 + 12375*uk_22 + 1210*uk_23 + 484*uk_24 + 484*uk_25 + 528*uk_26 + 2882*uk_27 + 4950*uk_28 + 484*uk_29 + 22*uk_3 + 484*uk_30 + 528*uk_31 + 2882*uk_32 + 4950*uk_33 + 484*uk_34 + 576*uk_35 + 3144*uk_36 + 5400*uk_37 + 528*uk_38 + 17161*uk_39 + 22*uk_4 + 29475*uk_40 + 2882*uk_41 + 50625*uk_42 + 4950*uk_43 + 484*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 56593173142*uk_47 + 56593173142*uk_48 + 61738007064*uk_49 + 24*uk_5 + 336986621891*uk_50 + 578793816225*uk_51 + 56593173142*uk_52 + 153424975*uk_53 + 61369990*uk_54 + 61369990*uk_55 + 66949080*uk_56 + 365430395*uk_57 + 627647625*uk_58 + 61369990*uk_59 + 131*uk_6 + 24547996*uk_60 + 24547996*uk_61 + 26779632*uk_62 + 146172158*uk_63 + 251059050*uk_64 + 24547996*uk_65 + 24547996*uk_66 + 26779632*uk_67 + 146172158*uk_68 + 251059050*uk_69 + 225*uk_7 + 24547996*uk_70 + 29214144*uk_71 + 159460536*uk_72 + 273882600*uk_73 + 26779632*uk_74 + 870388759*uk_75 + 1494942525*uk_76 + 146172158*uk_77 + 2567649375*uk_78 + 251059050*uk_79 + 22*uk_8 + 24547996*uk_80 + 166375*uk_81 + 66550*uk_82 + 66550*uk_83 + 72600*uk_84 + 396275*uk_85 + 680625*uk_86 + 66550*uk_87 + 26620*uk_88 + 26620*uk_89 + 2572416961*uk_9 + 29040*uk_90 + 158510*uk_91 + 272250*uk_92 + 26620*uk_93 + 26620*uk_94 + 29040*uk_95 + 158510*uk_96 + 272250*uk_97 + 26620*uk_98 + 31680*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 175560*uk_100 + 297000*uk_101 + 29040*uk_102 + 972895*uk_103 + 1645875*uk_104 + 160930*uk_105 + 2784375*uk_106 + 272250*uk_107 + 26620*uk_108 + 97336*uk_109 + 2333074*uk_11 + 46552*uk_110 + 50784*uk_111 + 281428*uk_112 + 476100*uk_113 + 46552*uk_114 + 22264*uk_115 + 24288*uk_116 + 134596*uk_117 + 227700*uk_118 + 22264*uk_119 + 1115818*uk_12 + 26496*uk_120 + 146832*uk_121 + 248400*uk_122 + 24288*uk_123 + 813694*uk_124 + 1376550*uk_125 + 134596*uk_126 + 2328750*uk_127 + 227700*uk_128 + 22264*uk_129 + 1217256*uk_13 + 10648*uk_130 + 11616*uk_131 + 64372*uk_132 + 108900*uk_133 + 10648*uk_134 + 12672*uk_135 + 70224*uk_136 + 118800*uk_137 + 11616*uk_138 + 389158*uk_139 + 6745627*uk_14 + 658350*uk_140 + 64372*uk_141 + 1113750*uk_142 + 108900*uk_143 + 10648*uk_144 + 13824*uk_145 + 76608*uk_146 + 129600*uk_147 + 12672*uk_148 + 424536*uk_149 + 11411775*uk_15 + 718200*uk_150 + 70224*uk_151 + 1215000*uk_152 + 118800*uk_153 + 11616*uk_154 + 2352637*uk_155 + 3980025*uk_156 + 389158*uk_157 + 6733125*uk_158 + 658350*uk_159 + 1115818*uk_16 + 64372*uk_160 + 11390625*uk_161 + 1113750*uk_162 + 108900*uk_163 + 10648*uk_164 + 3025*uk_17 + 2530*uk_18 + 1210*uk_19 + 55*uk_2 + 1320*uk_20 + 7315*uk_21 + 12375*uk_22 + 1210*uk_23 + 2116*uk_24 + 1012*uk_25 + 1104*uk_26 + 6118*uk_27 + 10350*uk_28 + 1012*uk_29 + 46*uk_3 + 484*uk_30 + 528*uk_31 + 2926*uk_32 + 4950*uk_33 + 484*uk_34 + 576*uk_35 + 3192*uk_36 + 5400*uk_37 + 528*uk_38 + 17689*uk_39 + 22*uk_4 + 29925*uk_40 + 2926*uk_41 + 50625*uk_42 + 4950*uk_43 + 484*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 118331180206*uk_47 + 56593173142*uk_48 + 61738007064*uk_49 + 24*uk_5 + 342131455813*uk_50 + 578793816225*uk_51 + 56593173142*uk_52 + 153424975*uk_53 + 128319070*uk_54 + 61369990*uk_55 + 66949080*uk_56 + 371009485*uk_57 + 627647625*uk_58 + 61369990*uk_59 + 133*uk_6 + 107321404*uk_60 + 51327628*uk_61 + 55993776*uk_62 + 310298842*uk_63 + 524941650*uk_64 + 51327628*uk_65 + 24547996*uk_66 + 26779632*uk_67 + 148403794*uk_68 + 251059050*uk_69 + 225*uk_7 + 24547996*uk_70 + 29214144*uk_71 + 161895048*uk_72 + 273882600*uk_73 + 26779632*uk_74 + 897168391*uk_75 + 1517766075*uk_76 + 148403794*uk_77 + 2567649375*uk_78 + 251059050*uk_79 + 22*uk_8 + 24547996*uk_80 + 166375*uk_81 + 139150*uk_82 + 66550*uk_83 + 72600*uk_84 + 402325*uk_85 + 680625*uk_86 + 66550*uk_87 + 116380*uk_88 + 55660*uk_89 + 2572416961*uk_9 + 60720*uk_90 + 336490*uk_91 + 569250*uk_92 + 55660*uk_93 + 26620*uk_94 + 29040*uk_95 + 160930*uk_96 + 272250*uk_97 + 26620*uk_98 + 31680*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 178200*uk_100 + 297000*uk_101 + 60720*uk_102 + 1002375*uk_103 + 1670625*uk_104 + 341550*uk_105 + 2784375*uk_106 + 569250*uk_107 + 116380*uk_108 + 830584*uk_109 + 4767586*uk_11 + 406456*uk_110 + 212064*uk_111 + 1192860*uk_112 + 1988100*uk_113 + 406456*uk_114 + 198904*uk_115 + 103776*uk_116 + 583740*uk_117 + 972900*uk_118 + 198904*uk_119 + 2333074*uk_12 + 54144*uk_120 + 304560*uk_121 + 507600*uk_122 + 103776*uk_123 + 1713150*uk_124 + 2855250*uk_125 + 583740*uk_126 + 4758750*uk_127 + 972900*uk_128 + 198904*uk_129 + 1217256*uk_13 + 97336*uk_130 + 50784*uk_131 + 285660*uk_132 + 476100*uk_133 + 97336*uk_134 + 26496*uk_135 + 149040*uk_136 + 248400*uk_137 + 50784*uk_138 + 838350*uk_139 + 6847065*uk_14 + 1397250*uk_140 + 285660*uk_141 + 2328750*uk_142 + 476100*uk_143 + 97336*uk_144 + 13824*uk_145 + 77760*uk_146 + 129600*uk_147 + 26496*uk_148 + 437400*uk_149 + 11411775*uk_15 + 729000*uk_150 + 149040*uk_151 + 1215000*uk_152 + 248400*uk_153 + 50784*uk_154 + 2460375*uk_155 + 4100625*uk_156 + 838350*uk_157 + 6834375*uk_158 + 1397250*uk_159 + 2333074*uk_16 + 285660*uk_160 + 11390625*uk_161 + 2328750*uk_162 + 476100*uk_163 + 97336*uk_164 + 3025*uk_17 + 5170*uk_18 + 2530*uk_19 + 55*uk_2 + 1320*uk_20 + 7425*uk_21 + 12375*uk_22 + 2530*uk_23 + 8836*uk_24 + 4324*uk_25 + 2256*uk_26 + 12690*uk_27 + 21150*uk_28 + 4324*uk_29 + 94*uk_3 + 2116*uk_30 + 1104*uk_31 + 6210*uk_32 + 10350*uk_33 + 2116*uk_34 + 576*uk_35 + 3240*uk_36 + 5400*uk_37 + 1104*uk_38 + 18225*uk_39 + 46*uk_4 + 30375*uk_40 + 6210*uk_41 + 50625*uk_42 + 10350*uk_43 + 2116*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 241807194334*uk_47 + 118331180206*uk_48 + 61738007064*uk_49 + 24*uk_5 + 347276289735*uk_50 + 578793816225*uk_51 + 118331180206*uk_52 + 153424975*uk_53 + 262217230*uk_54 + 128319070*uk_55 + 66949080*uk_56 + 376588575*uk_57 + 627647625*uk_58 + 128319070*uk_59 + 135*uk_6 + 448153084*uk_60 + 219308956*uk_61 + 114422064*uk_62 + 643624110*uk_63 + 1072706850*uk_64 + 219308956*uk_65 + 107321404*uk_66 + 55993776*uk_67 + 314964990*uk_68 + 524941650*uk_69 + 225*uk_7 + 107321404*uk_70 + 29214144*uk_71 + 164329560*uk_72 + 273882600*uk_73 + 55993776*uk_74 + 924353775*uk_75 + 1540589625*uk_76 + 314964990*uk_77 + 2567649375*uk_78 + 524941650*uk_79 + 46*uk_8 + 107321404*uk_80 + 166375*uk_81 + 284350*uk_82 + 139150*uk_83 + 72600*uk_84 + 408375*uk_85 + 680625*uk_86 + 139150*uk_87 + 485980*uk_88 + 237820*uk_89 + 2572416961*uk_9 + 124080*uk_90 + 697950*uk_91 + 1163250*uk_92 + 237820*uk_93 + 116380*uk_94 + 60720*uk_95 + 341550*uk_96 + 569250*uk_97 + 116380*uk_98 + 31680*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 150700*uk_100 + 247500*uk_101 + 103400*uk_102 + 1032295*uk_103 + 1695375*uk_104 + 708290*uk_105 + 2784375*uk_106 + 1163250*uk_107 + 485980*uk_108 + 24389*uk_109 + 1470851*uk_11 + 79054*uk_110 + 16820*uk_111 + 115217*uk_112 + 189225*uk_113 + 79054*uk_114 + 256244*uk_115 + 54520*uk_116 + 373462*uk_117 + 613350*uk_118 + 256244*uk_119 + 4767586*uk_12 + 11600*uk_120 + 79460*uk_121 + 130500*uk_122 + 54520*uk_123 + 544301*uk_124 + 893925*uk_125 + 373462*uk_126 + 1468125*uk_127 + 613350*uk_128 + 256244*uk_129 + 1014380*uk_13 + 830584*uk_130 + 176720*uk_131 + 1210532*uk_132 + 1988100*uk_133 + 830584*uk_134 + 37600*uk_135 + 257560*uk_136 + 423000*uk_137 + 176720*uk_138 + 1764286*uk_139 + 6948503*uk_14 + 2897550*uk_140 + 1210532*uk_141 + 4758750*uk_142 + 1988100*uk_143 + 830584*uk_144 + 8000*uk_145 + 54800*uk_146 + 90000*uk_147 + 37600*uk_148 + 375380*uk_149 + 11411775*uk_15 + 616500*uk_150 + 257560*uk_151 + 1012500*uk_152 + 423000*uk_153 + 176720*uk_154 + 2571353*uk_155 + 4223025*uk_156 + 1764286*uk_157 + 6935625*uk_158 + 2897550*uk_159 + 4767586*uk_16 + 1210532*uk_160 + 11390625*uk_161 + 4758750*uk_162 + 1988100*uk_163 + 830584*uk_164 + 3025*uk_17 + 1595*uk_18 + 5170*uk_19 + 55*uk_2 + 1100*uk_20 + 7535*uk_21 + 12375*uk_22 + 5170*uk_23 + 841*uk_24 + 2726*uk_25 + 580*uk_26 + 3973*uk_27 + 6525*uk_28 + 2726*uk_29 + 29*uk_3 + 8836*uk_30 + 1880*uk_31 + 12878*uk_32 + 21150*uk_33 + 8836*uk_34 + 400*uk_35 + 2740*uk_36 + 4500*uk_37 + 1880*uk_38 + 18769*uk_39 + 94*uk_4 + 30825*uk_40 + 12878*uk_41 + 50625*uk_42 + 21150*uk_43 + 8836*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 74600091869*uk_47 + 241807194334*uk_48 + 51448339220*uk_49 + 20*uk_5 + 352421123657*uk_50 + 578793816225*uk_51 + 241807194334*uk_52 + 153424975*uk_53 + 80896805*uk_54 + 262217230*uk_55 + 55790900*uk_56 + 382167665*uk_57 + 627647625*uk_58 + 262217230*uk_59 + 137*uk_6 + 42654679*uk_60 + 138259994*uk_61 + 29417020*uk_62 + 201506587*uk_63 + 330941475*uk_64 + 138259994*uk_65 + 448153084*uk_66 + 95351720*uk_67 + 653159282*uk_68 + 1072706850*uk_69 + 225*uk_7 + 448153084*uk_70 + 20287600*uk_71 + 138970060*uk_72 + 228235500*uk_73 + 95351720*uk_74 + 951944911*uk_75 + 1563413175*uk_76 + 653159282*uk_77 + 2567649375*uk_78 + 1072706850*uk_79 + 94*uk_8 + 448153084*uk_80 + 166375*uk_81 + 87725*uk_82 + 284350*uk_83 + 60500*uk_84 + 414425*uk_85 + 680625*uk_86 + 284350*uk_87 + 46255*uk_88 + 149930*uk_89 + 2572416961*uk_9 + 31900*uk_90 + 218515*uk_91 + 358875*uk_92 + 149930*uk_93 + 485980*uk_94 + 103400*uk_95 + 708290*uk_96 + 1163250*uk_97 + 485980*uk_98 + 22000*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 183480*uk_100 + 297000*uk_101 + 38280*uk_102 + 1062655*uk_103 + 1720125*uk_104 + 221705*uk_105 + 2784375*uk_106 + 358875*uk_107 + 46255*uk_108 + 1860867*uk_109 + 6238437*uk_11 + 438741*uk_110 + 363096*uk_111 + 2102931*uk_112 + 3404025*uk_113 + 438741*uk_114 + 103443*uk_115 + 85608*uk_116 + 495813*uk_117 + 802575*uk_118 + 103443*uk_119 + 1470851*uk_12 + 70848*uk_120 + 410328*uk_121 + 664200*uk_122 + 85608*uk_123 + 2376483*uk_124 + 3846825*uk_125 + 495813*uk_126 + 6226875*uk_127 + 802575*uk_128 + 103443*uk_129 + 1217256*uk_13 + 24389*uk_130 + 20184*uk_131 + 116899*uk_132 + 189225*uk_133 + 24389*uk_134 + 16704*uk_135 + 96744*uk_136 + 156600*uk_137 + 20184*uk_138 + 560309*uk_139 + 7049941*uk_14 + 906975*uk_140 + 116899*uk_141 + 1468125*uk_142 + 189225*uk_143 + 24389*uk_144 + 13824*uk_145 + 80064*uk_146 + 129600*uk_147 + 16704*uk_148 + 463704*uk_149 + 11411775*uk_15 + 750600*uk_150 + 96744*uk_151 + 1215000*uk_152 + 156600*uk_153 + 20184*uk_154 + 2685619*uk_155 + 4347225*uk_156 + 560309*uk_157 + 7036875*uk_158 + 906975*uk_159 + 1470851*uk_16 + 116899*uk_160 + 11390625*uk_161 + 1468125*uk_162 + 189225*uk_163 + 24389*uk_164 + 3025*uk_17 + 6765*uk_18 + 1595*uk_19 + 55*uk_2 + 1320*uk_20 + 7645*uk_21 + 12375*uk_22 + 1595*uk_23 + 15129*uk_24 + 3567*uk_25 + 2952*uk_26 + 17097*uk_27 + 27675*uk_28 + 3567*uk_29 + 123*uk_3 + 841*uk_30 + 696*uk_31 + 4031*uk_32 + 6525*uk_33 + 841*uk_34 + 576*uk_35 + 3336*uk_36 + 5400*uk_37 + 696*uk_38 + 19321*uk_39 + 29*uk_4 + 31275*uk_40 + 4031*uk_41 + 50625*uk_42 + 6525*uk_43 + 841*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 316407286203*uk_47 + 74600091869*uk_48 + 61738007064*uk_49 + 24*uk_5 + 357565957579*uk_50 + 578793816225*uk_51 + 74600091869*uk_52 + 153424975*uk_53 + 343114035*uk_54 + 80896805*uk_55 + 66949080*uk_56 + 387746755*uk_57 + 627647625*uk_58 + 80896805*uk_59 + 139*uk_6 + 767327751*uk_60 + 180914673*uk_61 + 149722488*uk_62 + 867142743*uk_63 + 1403648325*uk_64 + 180914673*uk_65 + 42654679*uk_66 + 35300424*uk_67 + 204448289*uk_68 + 330941475*uk_69 + 225*uk_7 + 42654679*uk_70 + 29214144*uk_71 + 169198584*uk_72 + 273882600*uk_73 + 35300424*uk_74 + 979941799*uk_75 + 1586236725*uk_76 + 204448289*uk_77 + 2567649375*uk_78 + 330941475*uk_79 + 29*uk_8 + 42654679*uk_80 + 166375*uk_81 + 372075*uk_82 + 87725*uk_83 + 72600*uk_84 + 420475*uk_85 + 680625*uk_86 + 87725*uk_87 + 832095*uk_88 + 196185*uk_89 + 2572416961*uk_9 + 162360*uk_90 + 940335*uk_91 + 1522125*uk_92 + 196185*uk_93 + 46255*uk_94 + 38280*uk_95 + 221705*uk_96 + 358875*uk_97 + 46255*uk_98 + 31680*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 155100*uk_100 + 247500*uk_101 + 135300*uk_102 + 1093455*uk_103 + 1744875*uk_104 + 953865*uk_105 + 2784375*uk_106 + 1522125*uk_107 + 832095*uk_108 + 1000000*uk_109 + 5071900*uk_11 + 1230000*uk_110 + 200000*uk_111 + 1410000*uk_112 + 2250000*uk_113 + 1230000*uk_114 + 1512900*uk_115 + 246000*uk_116 + 1734300*uk_117 + 2767500*uk_118 + 1512900*uk_119 + 6238437*uk_12 + 40000*uk_120 + 282000*uk_121 + 450000*uk_122 + 246000*uk_123 + 1988100*uk_124 + 3172500*uk_125 + 1734300*uk_126 + 5062500*uk_127 + 2767500*uk_128 + 1512900*uk_129 + 1014380*uk_13 + 1860867*uk_130 + 302580*uk_131 + 2133189*uk_132 + 3404025*uk_133 + 1860867*uk_134 + 49200*uk_135 + 346860*uk_136 + 553500*uk_137 + 302580*uk_138 + 2445363*uk_139 + 7151379*uk_14 + 3902175*uk_140 + 2133189*uk_141 + 6226875*uk_142 + 3404025*uk_143 + 1860867*uk_144 + 8000*uk_145 + 56400*uk_146 + 90000*uk_147 + 49200*uk_148 + 397620*uk_149 + 11411775*uk_15 + 634500*uk_150 + 346860*uk_151 + 1012500*uk_152 + 553500*uk_153 + 302580*uk_154 + 2803221*uk_155 + 4473225*uk_156 + 2445363*uk_157 + 7138125*uk_158 + 3902175*uk_159 + 6238437*uk_16 + 2133189*uk_160 + 11390625*uk_161 + 6226875*uk_162 + 3404025*uk_163 + 1860867*uk_164 + 3025*uk_17 + 5500*uk_18 + 6765*uk_19 + 55*uk_2 + 1100*uk_20 + 7755*uk_21 + 12375*uk_22 + 6765*uk_23 + 10000*uk_24 + 12300*uk_25 + 2000*uk_26 + 14100*uk_27 + 22500*uk_28 + 12300*uk_29 + 100*uk_3 + 15129*uk_30 + 2460*uk_31 + 17343*uk_32 + 27675*uk_33 + 15129*uk_34 + 400*uk_35 + 2820*uk_36 + 4500*uk_37 + 2460*uk_38 + 19881*uk_39 + 123*uk_4 + 31725*uk_40 + 17343*uk_41 + 50625*uk_42 + 27675*uk_43 + 15129*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 257241696100*uk_47 + 316407286203*uk_48 + 51448339220*uk_49 + 20*uk_5 + 362710791501*uk_50 + 578793816225*uk_51 + 316407286203*uk_52 + 153424975*uk_53 + 278954500*uk_54 + 343114035*uk_55 + 55790900*uk_56 + 393325845*uk_57 + 627647625*uk_58 + 343114035*uk_59 + 141*uk_6 + 507190000*uk_60 + 623843700*uk_61 + 101438000*uk_62 + 715137900*uk_63 + 1141177500*uk_64 + 623843700*uk_65 + 767327751*uk_66 + 124768740*uk_67 + 879619617*uk_68 + 1403648325*uk_69 + 225*uk_7 + 767327751*uk_70 + 20287600*uk_71 + 143027580*uk_72 + 228235500*uk_73 + 124768740*uk_74 + 1008344439*uk_75 + 1609060275*uk_76 + 879619617*uk_77 + 2567649375*uk_78 + 1403648325*uk_79 + 123*uk_8 + 767327751*uk_80 + 166375*uk_81 + 302500*uk_82 + 372075*uk_83 + 60500*uk_84 + 426525*uk_85 + 680625*uk_86 + 372075*uk_87 + 550000*uk_88 + 676500*uk_89 + 2572416961*uk_9 + 110000*uk_90 + 775500*uk_91 + 1237500*uk_92 + 676500*uk_93 + 832095*uk_94 + 135300*uk_95 + 953865*uk_96 + 1522125*uk_97 + 832095*uk_98 + 22000*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 157300*uk_100 + 247500*uk_101 + 110000*uk_102 + 1124695*uk_103 + 1769625*uk_104 + 786500*uk_105 + 2784375*uk_106 + 1237500*uk_107 + 550000*uk_108 + 912673*uk_109 + 4919743*uk_11 + 940900*uk_110 + 188180*uk_111 + 1345487*uk_112 + 2117025*uk_113 + 940900*uk_114 + 970000*uk_115 + 194000*uk_116 + 1387100*uk_117 + 2182500*uk_118 + 970000*uk_119 + 5071900*uk_12 + 38800*uk_120 + 277420*uk_121 + 436500*uk_122 + 194000*uk_123 + 1983553*uk_124 + 3120975*uk_125 + 1387100*uk_126 + 4910625*uk_127 + 2182500*uk_128 + 970000*uk_129 + 1014380*uk_13 + 1000000*uk_130 + 200000*uk_131 + 1430000*uk_132 + 2250000*uk_133 + 1000000*uk_134 + 40000*uk_135 + 286000*uk_136 + 450000*uk_137 + 200000*uk_138 + 2044900*uk_139 + 7252817*uk_14 + 3217500*uk_140 + 1430000*uk_141 + 5062500*uk_142 + 2250000*uk_143 + 1000000*uk_144 + 8000*uk_145 + 57200*uk_146 + 90000*uk_147 + 40000*uk_148 + 408980*uk_149 + 11411775*uk_15 + 643500*uk_150 + 286000*uk_151 + 1012500*uk_152 + 450000*uk_153 + 200000*uk_154 + 2924207*uk_155 + 4601025*uk_156 + 2044900*uk_157 + 7239375*uk_158 + 3217500*uk_159 + 5071900*uk_16 + 1430000*uk_160 + 11390625*uk_161 + 5062500*uk_162 + 2250000*uk_163 + 1000000*uk_164 + 3025*uk_17 + 5335*uk_18 + 5500*uk_19 + 55*uk_2 + 1100*uk_20 + 7865*uk_21 + 12375*uk_22 + 5500*uk_23 + 9409*uk_24 + 9700*uk_25 + 1940*uk_26 + 13871*uk_27 + 21825*uk_28 + 9700*uk_29 + 97*uk_3 + 10000*uk_30 + 2000*uk_31 + 14300*uk_32 + 22500*uk_33 + 10000*uk_34 + 400*uk_35 + 2860*uk_36 + 4500*uk_37 + 2000*uk_38 + 20449*uk_39 + 100*uk_4 + 32175*uk_40 + 14300*uk_41 + 50625*uk_42 + 22500*uk_43 + 10000*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 249524445217*uk_47 + 257241696100*uk_48 + 51448339220*uk_49 + 20*uk_5 + 367855625423*uk_50 + 578793816225*uk_51 + 257241696100*uk_52 + 153424975*uk_53 + 270585865*uk_54 + 278954500*uk_55 + 55790900*uk_56 + 398904935*uk_57 + 627647625*uk_58 + 278954500*uk_59 + 143*uk_6 + 477215071*uk_60 + 491974300*uk_61 + 98394860*uk_62 + 703523249*uk_63 + 1106942175*uk_64 + 491974300*uk_65 + 507190000*uk_66 + 101438000*uk_67 + 725281700*uk_68 + 1141177500*uk_69 + 225*uk_7 + 507190000*uk_70 + 20287600*uk_71 + 145056340*uk_72 + 228235500*uk_73 + 101438000*uk_74 + 1037152831*uk_75 + 1631883825*uk_76 + 725281700*uk_77 + 2567649375*uk_78 + 1141177500*uk_79 + 100*uk_8 + 507190000*uk_80 + 166375*uk_81 + 293425*uk_82 + 302500*uk_83 + 60500*uk_84 + 432575*uk_85 + 680625*uk_86 + 302500*uk_87 + 517495*uk_88 + 533500*uk_89 + 2572416961*uk_9 + 106700*uk_90 + 762905*uk_91 + 1200375*uk_92 + 533500*uk_93 + 550000*uk_94 + 110000*uk_95 + 786500*uk_96 + 1237500*uk_97 + 550000*uk_98 + 22000*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 159500*uk_100 + 247500*uk_101 + 106700*uk_102 + 1156375*uk_103 + 1794375*uk_104 + 773575*uk_105 + 2784375*uk_106 + 1200375*uk_107 + 517495*uk_108 + 1481544*uk_109 + 5781966*uk_11 + 1260612*uk_110 + 259920*uk_111 + 1884420*uk_112 + 2924100*uk_113 + 1260612*uk_114 + 1072626*uk_115 + 221160*uk_116 + 1603410*uk_117 + 2488050*uk_118 + 1072626*uk_119 + 4919743*uk_12 + 45600*uk_120 + 330600*uk_121 + 513000*uk_122 + 221160*uk_123 + 2396850*uk_124 + 3719250*uk_125 + 1603410*uk_126 + 5771250*uk_127 + 2488050*uk_128 + 1072626*uk_129 + 1014380*uk_13 + 912673*uk_130 + 188180*uk_131 + 1364305*uk_132 + 2117025*uk_133 + 912673*uk_134 + 38800*uk_135 + 281300*uk_136 + 436500*uk_137 + 188180*uk_138 + 2039425*uk_139 + 7354255*uk_14 + 3164625*uk_140 + 1364305*uk_141 + 4910625*uk_142 + 2117025*uk_143 + 912673*uk_144 + 8000*uk_145 + 58000*uk_146 + 90000*uk_147 + 38800*uk_148 + 420500*uk_149 + 11411775*uk_15 + 652500*uk_150 + 281300*uk_151 + 1012500*uk_152 + 436500*uk_153 + 188180*uk_154 + 3048625*uk_155 + 4730625*uk_156 + 2039425*uk_157 + 7340625*uk_158 + 3164625*uk_159 + 4919743*uk_16 + 1364305*uk_160 + 11390625*uk_161 + 4910625*uk_162 + 2117025*uk_163 + 912673*uk_164 + 3025*uk_17 + 6270*uk_18 + 5335*uk_19 + 55*uk_2 + 1100*uk_20 + 7975*uk_21 + 12375*uk_22 + 5335*uk_23 + 12996*uk_24 + 11058*uk_25 + 2280*uk_26 + 16530*uk_27 + 25650*uk_28 + 11058*uk_29 + 114*uk_3 + 9409*uk_30 + 1940*uk_31 + 14065*uk_32 + 21825*uk_33 + 9409*uk_34 + 400*uk_35 + 2900*uk_36 + 4500*uk_37 + 1940*uk_38 + 21025*uk_39 + 97*uk_4 + 32625*uk_40 + 14065*uk_41 + 50625*uk_42 + 21825*uk_43 + 9409*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 293255533554*uk_47 + 249524445217*uk_48 + 51448339220*uk_49 + 20*uk_5 + 373000459345*uk_50 + 578793816225*uk_51 + 249524445217*uk_52 + 153424975*uk_53 + 318008130*uk_54 + 270585865*uk_55 + 55790900*uk_56 + 404484025*uk_57 + 627647625*uk_58 + 270585865*uk_59 + 145*uk_6 + 659144124*uk_60 + 560850702*uk_61 + 115639320*uk_62 + 838385070*uk_63 + 1300942350*uk_64 + 560850702*uk_65 + 477215071*uk_66 + 98394860*uk_67 + 713362735*uk_68 + 1106942175*uk_69 + 225*uk_7 + 477215071*uk_70 + 20287600*uk_71 + 147085100*uk_72 + 228235500*uk_73 + 98394860*uk_74 + 1066366975*uk_75 + 1654707375*uk_76 + 713362735*uk_77 + 2567649375*uk_78 + 1106942175*uk_79 + 97*uk_8 + 477215071*uk_80 + 166375*uk_81 + 344850*uk_82 + 293425*uk_83 + 60500*uk_84 + 438625*uk_85 + 680625*uk_86 + 293425*uk_87 + 714780*uk_88 + 608190*uk_89 + 2572416961*uk_9 + 125400*uk_90 + 909150*uk_91 + 1410750*uk_92 + 608190*uk_93 + 517495*uk_94 + 106700*uk_95 + 773575*uk_96 + 1200375*uk_97 + 517495*uk_98 + 22000*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 129360*uk_100 + 198000*uk_101 + 100320*uk_102 + 1188495*uk_103 + 1819125*uk_104 + 921690*uk_105 + 2784375*uk_106 + 1410750*uk_107 + 714780*uk_108 + 64*uk_109 + 202876*uk_11 + 1824*uk_110 + 256*uk_111 + 2352*uk_112 + 3600*uk_113 + 1824*uk_114 + 51984*uk_115 + 7296*uk_116 + 67032*uk_117 + 102600*uk_118 + 51984*uk_119 + 5781966*uk_12 + 1024*uk_120 + 9408*uk_121 + 14400*uk_122 + 7296*uk_123 + 86436*uk_124 + 132300*uk_125 + 67032*uk_126 + 202500*uk_127 + 102600*uk_128 + 51984*uk_129 + 811504*uk_13 + 1481544*uk_130 + 207936*uk_131 + 1910412*uk_132 + 2924100*uk_133 + 1481544*uk_134 + 29184*uk_135 + 268128*uk_136 + 410400*uk_137 + 207936*uk_138 + 2463426*uk_139 + 7455693*uk_14 + 3770550*uk_140 + 1910412*uk_141 + 5771250*uk_142 + 2924100*uk_143 + 1481544*uk_144 + 4096*uk_145 + 37632*uk_146 + 57600*uk_147 + 29184*uk_148 + 345744*uk_149 + 11411775*uk_15 + 529200*uk_150 + 268128*uk_151 + 810000*uk_152 + 410400*uk_153 + 207936*uk_154 + 3176523*uk_155 + 4862025*uk_156 + 2463426*uk_157 + 7441875*uk_158 + 3770550*uk_159 + 5781966*uk_16 + 1910412*uk_160 + 11390625*uk_161 + 5771250*uk_162 + 2924100*uk_163 + 1481544*uk_164 + 3025*uk_17 + 220*uk_18 + 6270*uk_19 + 55*uk_2 + 880*uk_20 + 8085*uk_21 + 12375*uk_22 + 6270*uk_23 + 16*uk_24 + 456*uk_25 + 64*uk_26 + 588*uk_27 + 900*uk_28 + 456*uk_29 + 4*uk_3 + 12996*uk_30 + 1824*uk_31 + 16758*uk_32 + 25650*uk_33 + 12996*uk_34 + 256*uk_35 + 2352*uk_36 + 3600*uk_37 + 1824*uk_38 + 21609*uk_39 + 114*uk_4 + 33075*uk_40 + 16758*uk_41 + 50625*uk_42 + 25650*uk_43 + 12996*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 10289667844*uk_47 + 293255533554*uk_48 + 41158671376*uk_49 + 16*uk_5 + 378145293267*uk_50 + 578793816225*uk_51 + 293255533554*uk_52 + 153424975*uk_53 + 11158180*uk_54 + 318008130*uk_55 + 44632720*uk_56 + 410063115*uk_57 + 627647625*uk_58 + 318008130*uk_59 + 147*uk_6 + 811504*uk_60 + 23127864*uk_61 + 3246016*uk_62 + 29822772*uk_63 + 45647100*uk_64 + 23127864*uk_65 + 659144124*uk_66 + 92511456*uk_67 + 849949002*uk_68 + 1300942350*uk_69 + 225*uk_7 + 659144124*uk_70 + 12984064*uk_71 + 119291088*uk_72 + 182588400*uk_73 + 92511456*uk_74 + 1095986871*uk_75 + 1677530925*uk_76 + 849949002*uk_77 + 2567649375*uk_78 + 1300942350*uk_79 + 114*uk_8 + 659144124*uk_80 + 166375*uk_81 + 12100*uk_82 + 344850*uk_83 + 48400*uk_84 + 444675*uk_85 + 680625*uk_86 + 344850*uk_87 + 880*uk_88 + 25080*uk_89 + 2572416961*uk_9 + 3520*uk_90 + 32340*uk_91 + 49500*uk_92 + 25080*uk_93 + 714780*uk_94 + 100320*uk_95 + 921690*uk_96 + 1410750*uk_97 + 714780*uk_98 + 14080*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 163900*uk_100 + 247500*uk_101 + 4400*uk_102 + 1221055*uk_103 + 1843875*uk_104 + 32780*uk_105 + 2784375*uk_106 + 49500*uk_107 + 880*uk_108 + 205379*uk_109 + 2992421*uk_11 + 13924*uk_110 + 69620*uk_111 + 518669*uk_112 + 783225*uk_113 + 13924*uk_114 + 944*uk_115 + 4720*uk_116 + 35164*uk_117 + 53100*uk_118 + 944*uk_119 + 202876*uk_12 + 23600*uk_120 + 175820*uk_121 + 265500*uk_122 + 4720*uk_123 + 1309859*uk_124 + 1977975*uk_125 + 35164*uk_126 + 2986875*uk_127 + 53100*uk_128 + 944*uk_129 + 1014380*uk_13 + 64*uk_130 + 320*uk_131 + 2384*uk_132 + 3600*uk_133 + 64*uk_134 + 1600*uk_135 + 11920*uk_136 + 18000*uk_137 + 320*uk_138 + 88804*uk_139 + 7557131*uk_14 + 134100*uk_140 + 2384*uk_141 + 202500*uk_142 + 3600*uk_143 + 64*uk_144 + 8000*uk_145 + 59600*uk_146 + 90000*uk_147 + 1600*uk_148 + 444020*uk_149 + 11411775*uk_15 + 670500*uk_150 + 11920*uk_151 + 1012500*uk_152 + 18000*uk_153 + 320*uk_154 + 3307949*uk_155 + 4995225*uk_156 + 88804*uk_157 + 7543125*uk_158 + 134100*uk_159 + 202876*uk_16 + 2384*uk_160 + 11390625*uk_161 + 202500*uk_162 + 3600*uk_163 + 64*uk_164 + 3025*uk_17 + 3245*uk_18 + 220*uk_19 + 55*uk_2 + 1100*uk_20 + 8195*uk_21 + 12375*uk_22 + 220*uk_23 + 3481*uk_24 + 236*uk_25 + 1180*uk_26 + 8791*uk_27 + 13275*uk_28 + 236*uk_29 + 59*uk_3 + 16*uk_30 + 80*uk_31 + 596*uk_32 + 900*uk_33 + 16*uk_34 + 400*uk_35 + 2980*uk_36 + 4500*uk_37 + 80*uk_38 + 22201*uk_39 + 4*uk_4 + 33525*uk_40 + 596*uk_41 + 50625*uk_42 + 900*uk_43 + 16*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 151772600699*uk_47 + 10289667844*uk_48 + 51448339220*uk_49 + 20*uk_5 + 383290127189*uk_50 + 578793816225*uk_51 + 10289667844*uk_52 + 153424975*uk_53 + 164583155*uk_54 + 11158180*uk_55 + 55790900*uk_56 + 415642205*uk_57 + 627647625*uk_58 + 11158180*uk_59 + 149*uk_6 + 176552839*uk_60 + 11969684*uk_61 + 59848420*uk_62 + 445870729*uk_63 + 673294725*uk_64 + 11969684*uk_65 + 811504*uk_66 + 4057520*uk_67 + 30228524*uk_68 + 45647100*uk_69 + 225*uk_7 + 811504*uk_70 + 20287600*uk_71 + 151142620*uk_72 + 228235500*uk_73 + 4057520*uk_74 + 1126012519*uk_75 + 1700354475*uk_76 + 30228524*uk_77 + 2567649375*uk_78 + 45647100*uk_79 + 4*uk_8 + 811504*uk_80 + 166375*uk_81 + 178475*uk_82 + 12100*uk_83 + 60500*uk_84 + 450725*uk_85 + 680625*uk_86 + 12100*uk_87 + 191455*uk_88 + 12980*uk_89 + 2572416961*uk_9 + 64900*uk_90 + 483505*uk_91 + 730125*uk_92 + 12980*uk_93 + 880*uk_94 + 4400*uk_95 + 32780*uk_96 + 49500*uk_97 + 880*uk_98 + 22000*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 166100*uk_100 + 247500*uk_101 + 64900*uk_102 + 1254055*uk_103 + 1868625*uk_104 + 489995*uk_105 + 2784375*uk_106 + 730125*uk_107 + 191455*uk_108 + 2406104*uk_109 + 6796346*uk_11 + 1059404*uk_110 + 359120*uk_111 + 2711356*uk_112 + 4040100*uk_113 + 1059404*uk_114 + 466454*uk_115 + 158120*uk_116 + 1193806*uk_117 + 1778850*uk_118 + 466454*uk_119 + 2992421*uk_12 + 53600*uk_120 + 404680*uk_121 + 603000*uk_122 + 158120*uk_123 + 3055334*uk_124 + 4552650*uk_125 + 1193806*uk_126 + 6783750*uk_127 + 1778850*uk_128 + 466454*uk_129 + 1014380*uk_13 + 205379*uk_130 + 69620*uk_131 + 525631*uk_132 + 783225*uk_133 + 205379*uk_134 + 23600*uk_135 + 178180*uk_136 + 265500*uk_137 + 69620*uk_138 + 1345259*uk_139 + 7658569*uk_14 + 2004525*uk_140 + 525631*uk_141 + 2986875*uk_142 + 783225*uk_143 + 205379*uk_144 + 8000*uk_145 + 60400*uk_146 + 90000*uk_147 + 23600*uk_148 + 456020*uk_149 + 11411775*uk_15 + 679500*uk_150 + 178180*uk_151 + 1012500*uk_152 + 265500*uk_153 + 69620*uk_154 + 3442951*uk_155 + 5130225*uk_156 + 1345259*uk_157 + 7644375*uk_158 + 2004525*uk_159 + 2992421*uk_16 + 525631*uk_160 + 11390625*uk_161 + 2986875*uk_162 + 783225*uk_163 + 205379*uk_164 + 3025*uk_17 + 7370*uk_18 + 3245*uk_19 + 55*uk_2 + 1100*uk_20 + 8305*uk_21 + 12375*uk_22 + 3245*uk_23 + 17956*uk_24 + 7906*uk_25 + 2680*uk_26 + 20234*uk_27 + 30150*uk_28 + 7906*uk_29 + 134*uk_3 + 3481*uk_30 + 1180*uk_31 + 8909*uk_32 + 13275*uk_33 + 3481*uk_34 + 400*uk_35 + 3020*uk_36 + 4500*uk_37 + 1180*uk_38 + 22801*uk_39 + 59*uk_4 + 33975*uk_40 + 8909*uk_41 + 50625*uk_42 + 13275*uk_43 + 3481*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 344703872774*uk_47 + 151772600699*uk_48 + 51448339220*uk_49 + 20*uk_5 + 388434961111*uk_50 + 578793816225*uk_51 + 151772600699*uk_52 + 153424975*uk_53 + 373799030*uk_54 + 164583155*uk_55 + 55790900*uk_56 + 421221295*uk_57 + 627647625*uk_58 + 164583155*uk_59 + 151*uk_6 + 910710364*uk_60 + 400984414*uk_61 + 135926920*uk_62 + 1026248246*uk_63 + 1529177850*uk_64 + 400984414*uk_65 + 176552839*uk_66 + 59848420*uk_67 + 451855571*uk_68 + 673294725*uk_69 + 225*uk_7 + 176552839*uk_70 + 20287600*uk_71 + 153171380*uk_72 + 228235500*uk_73 + 59848420*uk_74 + 1156443919*uk_75 + 1723178025*uk_76 + 451855571*uk_77 + 2567649375*uk_78 + 673294725*uk_79 + 59*uk_8 + 176552839*uk_80 + 166375*uk_81 + 405350*uk_82 + 178475*uk_83 + 60500*uk_84 + 456775*uk_85 + 680625*uk_86 + 178475*uk_87 + 987580*uk_88 + 434830*uk_89 + 2572416961*uk_9 + 147400*uk_90 + 1112870*uk_91 + 1658250*uk_92 + 434830*uk_93 + 191455*uk_94 + 64900*uk_95 + 489995*uk_96 + 730125*uk_97 + 191455*uk_98 + 22000*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 134640*uk_100 + 198000*uk_101 + 117920*uk_102 + 1287495*uk_103 + 1893375*uk_104 + 1127610*uk_105 + 2784375*uk_106 + 1658250*uk_107 + 987580*uk_108 + 438976*uk_109 + 3854644*uk_11 + 773984*uk_110 + 92416*uk_111 + 883728*uk_112 + 1299600*uk_113 + 773984*uk_114 + 1364656*uk_115 + 162944*uk_116 + 1558152*uk_117 + 2291400*uk_118 + 1364656*uk_119 + 6796346*uk_12 + 19456*uk_120 + 186048*uk_121 + 273600*uk_122 + 162944*uk_123 + 1779084*uk_124 + 2616300*uk_125 + 1558152*uk_126 + 3847500*uk_127 + 2291400*uk_128 + 1364656*uk_129 + 811504*uk_13 + 2406104*uk_130 + 287296*uk_131 + 2747268*uk_132 + 4040100*uk_133 + 2406104*uk_134 + 34304*uk_135 + 328032*uk_136 + 482400*uk_137 + 287296*uk_138 + 3136806*uk_139 + 7760007*uk_14 + 4612950*uk_140 + 2747268*uk_141 + 6783750*uk_142 + 4040100*uk_143 + 2406104*uk_144 + 4096*uk_145 + 39168*uk_146 + 57600*uk_147 + 34304*uk_148 + 374544*uk_149 + 11411775*uk_15 + 550800*uk_150 + 328032*uk_151 + 810000*uk_152 + 482400*uk_153 + 287296*uk_154 + 3581577*uk_155 + 5267025*uk_156 + 3136806*uk_157 + 7745625*uk_158 + 4612950*uk_159 + 6796346*uk_16 + 2747268*uk_160 + 11390625*uk_161 + 6783750*uk_162 + 4040100*uk_163 + 2406104*uk_164 + 3025*uk_17 + 4180*uk_18 + 7370*uk_19 + 55*uk_2 + 880*uk_20 + 8415*uk_21 + 12375*uk_22 + 7370*uk_23 + 5776*uk_24 + 10184*uk_25 + 1216*uk_26 + 11628*uk_27 + 17100*uk_28 + 10184*uk_29 + 76*uk_3 + 17956*uk_30 + 2144*uk_31 + 20502*uk_32 + 30150*uk_33 + 17956*uk_34 + 256*uk_35 + 2448*uk_36 + 3600*uk_37 + 2144*uk_38 + 23409*uk_39 + 134*uk_4 + 34425*uk_40 + 20502*uk_41 + 50625*uk_42 + 30150*uk_43 + 17956*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 195503689036*uk_47 + 344703872774*uk_48 + 41158671376*uk_49 + 16*uk_5 + 393579795033*uk_50 + 578793816225*uk_51 + 344703872774*uk_52 + 153424975*uk_53 + 212005420*uk_54 + 373799030*uk_55 + 44632720*uk_56 + 426800385*uk_57 + 627647625*uk_58 + 373799030*uk_59 + 153*uk_6 + 292952944*uk_60 + 516522296*uk_61 + 61674304*uk_62 + 589760532*uk_63 + 867294900*uk_64 + 516522296*uk_65 + 910710364*uk_66 + 108741536*uk_67 + 1039840938*uk_68 + 1529177850*uk_69 + 225*uk_7 + 910710364*uk_70 + 12984064*uk_71 + 124160112*uk_72 + 182588400*uk_73 + 108741536*uk_74 + 1187281071*uk_75 + 1746001575*uk_76 + 1039840938*uk_77 + 2567649375*uk_78 + 1529177850*uk_79 + 134*uk_8 + 910710364*uk_80 + 166375*uk_81 + 229900*uk_82 + 405350*uk_83 + 48400*uk_84 + 462825*uk_85 + 680625*uk_86 + 405350*uk_87 + 317680*uk_88 + 560120*uk_89 + 2572416961*uk_9 + 66880*uk_90 + 639540*uk_91 + 940500*uk_92 + 560120*uk_93 + 987580*uk_94 + 117920*uk_95 + 1127610*uk_96 + 1658250*uk_97 + 987580*uk_98 + 14080*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 136400*uk_100 + 198000*uk_101 + 66880*uk_102 + 1321375*uk_103 + 1918125*uk_104 + 647900*uk_105 + 2784375*uk_106 + 940500*uk_107 + 317680*uk_108 + 39304*uk_109 + 1724446*uk_11 + 87856*uk_110 + 18496*uk_111 + 179180*uk_112 + 260100*uk_113 + 87856*uk_114 + 196384*uk_115 + 41344*uk_116 + 400520*uk_117 + 581400*uk_118 + 196384*uk_119 + 3854644*uk_12 + 8704*uk_120 + 84320*uk_121 + 122400*uk_122 + 41344*uk_123 + 816850*uk_124 + 1185750*uk_125 + 400520*uk_126 + 1721250*uk_127 + 581400*uk_128 + 196384*uk_129 + 811504*uk_13 + 438976*uk_130 + 92416*uk_131 + 895280*uk_132 + 1299600*uk_133 + 438976*uk_134 + 19456*uk_135 + 188480*uk_136 + 273600*uk_137 + 92416*uk_138 + 1825900*uk_139 + 7861445*uk_14 + 2650500*uk_140 + 895280*uk_141 + 3847500*uk_142 + 1299600*uk_143 + 438976*uk_144 + 4096*uk_145 + 39680*uk_146 + 57600*uk_147 + 19456*uk_148 + 384400*uk_149 + 11411775*uk_15 + 558000*uk_150 + 188480*uk_151 + 810000*uk_152 + 273600*uk_153 + 92416*uk_154 + 3723875*uk_155 + 5405625*uk_156 + 1825900*uk_157 + 7846875*uk_158 + 2650500*uk_159 + 3854644*uk_16 + 895280*uk_160 + 11390625*uk_161 + 3847500*uk_162 + 1299600*uk_163 + 438976*uk_164 + 3025*uk_17 + 1870*uk_18 + 4180*uk_19 + 55*uk_2 + 880*uk_20 + 8525*uk_21 + 12375*uk_22 + 4180*uk_23 + 1156*uk_24 + 2584*uk_25 + 544*uk_26 + 5270*uk_27 + 7650*uk_28 + 2584*uk_29 + 34*uk_3 + 5776*uk_30 + 1216*uk_31 + 11780*uk_32 + 17100*uk_33 + 5776*uk_34 + 256*uk_35 + 2480*uk_36 + 3600*uk_37 + 1216*uk_38 + 24025*uk_39 + 76*uk_4 + 34875*uk_40 + 11780*uk_41 + 50625*uk_42 + 17100*uk_43 + 5776*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 87462176674*uk_47 + 195503689036*uk_48 + 41158671376*uk_49 + 16*uk_5 + 398724628955*uk_50 + 578793816225*uk_51 + 195503689036*uk_52 + 153424975*uk_53 + 94844530*uk_54 + 212005420*uk_55 + 44632720*uk_56 + 432379475*uk_57 + 627647625*uk_58 + 212005420*uk_59 + 155*uk_6 + 58631164*uk_60 + 131057896*uk_61 + 27591136*uk_62 + 267289130*uk_63 + 388000350*uk_64 + 131057896*uk_65 + 292952944*uk_66 + 61674304*uk_67 + 597469820*uk_68 + 867294900*uk_69 + 225*uk_7 + 292952944*uk_70 + 12984064*uk_71 + 125783120*uk_72 + 182588400*uk_73 + 61674304*uk_74 + 1218523975*uk_75 + 1768825125*uk_76 + 597469820*uk_77 + 2567649375*uk_78 + 867294900*uk_79 + 76*uk_8 + 292952944*uk_80 + 166375*uk_81 + 102850*uk_82 + 229900*uk_83 + 48400*uk_84 + 468875*uk_85 + 680625*uk_86 + 229900*uk_87 + 63580*uk_88 + 142120*uk_89 + 2572416961*uk_9 + 29920*uk_90 + 289850*uk_91 + 420750*uk_92 + 142120*uk_93 + 317680*uk_94 + 66880*uk_95 + 647900*uk_96 + 940500*uk_97 + 317680*uk_98 + 14080*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 138160*uk_100 + 198000*uk_101 + 29920*uk_102 + 1355695*uk_103 + 1942875*uk_104 + 293590*uk_105 + 2784375*uk_106 + 420750*uk_107 + 63580*uk_108 + 512*uk_109 + 405752*uk_11 + 2176*uk_110 + 1024*uk_111 + 10048*uk_112 + 14400*uk_113 + 2176*uk_114 + 9248*uk_115 + 4352*uk_116 + 42704*uk_117 + 61200*uk_118 + 9248*uk_119 + 1724446*uk_12 + 2048*uk_120 + 20096*uk_121 + 28800*uk_122 + 4352*uk_123 + 197192*uk_124 + 282600*uk_125 + 42704*uk_126 + 405000*uk_127 + 61200*uk_128 + 9248*uk_129 + 811504*uk_13 + 39304*uk_130 + 18496*uk_131 + 181492*uk_132 + 260100*uk_133 + 39304*uk_134 + 8704*uk_135 + 85408*uk_136 + 122400*uk_137 + 18496*uk_138 + 838066*uk_139 + 7962883*uk_14 + 1201050*uk_140 + 181492*uk_141 + 1721250*uk_142 + 260100*uk_143 + 39304*uk_144 + 4096*uk_145 + 40192*uk_146 + 57600*uk_147 + 8704*uk_148 + 394384*uk_149 + 11411775*uk_15 + 565200*uk_150 + 85408*uk_151 + 810000*uk_152 + 122400*uk_153 + 18496*uk_154 + 3869893*uk_155 + 5546025*uk_156 + 838066*uk_157 + 7948125*uk_158 + 1201050*uk_159 + 1724446*uk_16 + 181492*uk_160 + 11390625*uk_161 + 1721250*uk_162 + 260100*uk_163 + 39304*uk_164 + 3025*uk_17 + 440*uk_18 + 1870*uk_19 + 55*uk_2 + 880*uk_20 + 8635*uk_21 + 12375*uk_22 + 1870*uk_23 + 64*uk_24 + 272*uk_25 + 128*uk_26 + 1256*uk_27 + 1800*uk_28 + 272*uk_29 + 8*uk_3 + 1156*uk_30 + 544*uk_31 + 5338*uk_32 + 7650*uk_33 + 1156*uk_34 + 256*uk_35 + 2512*uk_36 + 3600*uk_37 + 544*uk_38 + 24649*uk_39 + 34*uk_4 + 35325*uk_40 + 5338*uk_41 + 50625*uk_42 + 7650*uk_43 + 1156*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 20579335688*uk_47 + 87462176674*uk_48 + 41158671376*uk_49 + 16*uk_5 + 403869462877*uk_50 + 578793816225*uk_51 + 87462176674*uk_52 + 153424975*uk_53 + 22316360*uk_54 + 94844530*uk_55 + 44632720*uk_56 + 437958565*uk_57 + 627647625*uk_58 + 94844530*uk_59 + 157*uk_6 + 3246016*uk_60 + 13795568*uk_61 + 6492032*uk_62 + 63703064*uk_63 + 91294200*uk_64 + 13795568*uk_65 + 58631164*uk_66 + 27591136*uk_67 + 270738022*uk_68 + 388000350*uk_69 + 225*uk_7 + 58631164*uk_70 + 12984064*uk_71 + 127406128*uk_72 + 182588400*uk_73 + 27591136*uk_74 + 1250172631*uk_75 + 1791648675*uk_76 + 270738022*uk_77 + 2567649375*uk_78 + 388000350*uk_79 + 34*uk_8 + 58631164*uk_80 + 166375*uk_81 + 24200*uk_82 + 102850*uk_83 + 48400*uk_84 + 474925*uk_85 + 680625*uk_86 + 102850*uk_87 + 3520*uk_88 + 14960*uk_89 + 2572416961*uk_9 + 7040*uk_90 + 69080*uk_91 + 99000*uk_92 + 14960*uk_93 + 63580*uk_94 + 29920*uk_95 + 293590*uk_96 + 420750*uk_97 + 63580*uk_98 + 14080*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 174900*uk_100 + 247500*uk_101 + 8800*uk_102 + 1390455*uk_103 + 1967625*uk_104 + 69960*uk_105 + 2784375*uk_106 + 99000*uk_107 + 3520*uk_108 + 3869893*uk_109 + 7962883*uk_11 + 197192*uk_110 + 492980*uk_111 + 3919191*uk_112 + 5546025*uk_113 + 197192*uk_114 + 10048*uk_115 + 25120*uk_116 + 199704*uk_117 + 282600*uk_118 + 10048*uk_119 + 405752*uk_12 + 62800*uk_120 + 499260*uk_121 + 706500*uk_122 + 25120*uk_123 + 3969117*uk_124 + 5616675*uk_125 + 199704*uk_126 + 7948125*uk_127 + 282600*uk_128 + 10048*uk_129 + 1014380*uk_13 + 512*uk_130 + 1280*uk_131 + 10176*uk_132 + 14400*uk_133 + 512*uk_134 + 3200*uk_135 + 25440*uk_136 + 36000*uk_137 + 1280*uk_138 + 202248*uk_139 + 8064321*uk_14 + 286200*uk_140 + 10176*uk_141 + 405000*uk_142 + 14400*uk_143 + 512*uk_144 + 8000*uk_145 + 63600*uk_146 + 90000*uk_147 + 3200*uk_148 + 505620*uk_149 + 11411775*uk_15 + 715500*uk_150 + 25440*uk_151 + 1012500*uk_152 + 36000*uk_153 + 1280*uk_154 + 4019679*uk_155 + 5688225*uk_156 + 202248*uk_157 + 8049375*uk_158 + 286200*uk_159 + 405752*uk_16 + 10176*uk_160 + 11390625*uk_161 + 405000*uk_162 + 14400*uk_163 + 512*uk_164 + 3025*uk_17 + 8635*uk_18 + 440*uk_19 + 55*uk_2 + 1100*uk_20 + 8745*uk_21 + 12375*uk_22 + 440*uk_23 + 24649*uk_24 + 1256*uk_25 + 3140*uk_26 + 24963*uk_27 + 35325*uk_28 + 1256*uk_29 + 157*uk_3 + 64*uk_30 + 160*uk_31 + 1272*uk_32 + 1800*uk_33 + 64*uk_34 + 400*uk_35 + 3180*uk_36 + 4500*uk_37 + 160*uk_38 + 25281*uk_39 + 8*uk_4 + 35775*uk_40 + 1272*uk_41 + 50625*uk_42 + 1800*uk_43 + 64*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 403869462877*uk_47 + 20579335688*uk_48 + 51448339220*uk_49 + 20*uk_5 + 409014296799*uk_50 + 578793816225*uk_51 + 20579335688*uk_52 + 153424975*uk_53 + 437958565*uk_54 + 22316360*uk_55 + 55790900*uk_56 + 443537655*uk_57 + 627647625*uk_58 + 22316360*uk_59 + 159*uk_6 + 1250172631*uk_60 + 63703064*uk_61 + 159257660*uk_62 + 1266098397*uk_63 + 1791648675*uk_64 + 63703064*uk_65 + 3246016*uk_66 + 8115040*uk_67 + 64514568*uk_68 + 91294200*uk_69 + 225*uk_7 + 3246016*uk_70 + 20287600*uk_71 + 161286420*uk_72 + 228235500*uk_73 + 8115040*uk_74 + 1282227039*uk_75 + 1814472225*uk_76 + 64514568*uk_77 + 2567649375*uk_78 + 91294200*uk_79 + 8*uk_8 + 3246016*uk_80 + 166375*uk_81 + 474925*uk_82 + 24200*uk_83 + 60500*uk_84 + 480975*uk_85 + 680625*uk_86 + 24200*uk_87 + 1355695*uk_88 + 69080*uk_89 + 2572416961*uk_9 + 172700*uk_90 + 1372965*uk_91 + 1942875*uk_92 + 69080*uk_93 + 3520*uk_94 + 8800*uk_95 + 69960*uk_96 + 99000*uk_97 + 3520*uk_98 + 22000*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 106260*uk_100 + 148500*uk_101 + 103620*uk_102 + 1425655*uk_103 + 1992375*uk_104 + 1390235*uk_105 + 2784375*uk_106 + 1942875*uk_107 + 1355695*uk_108 + 64*uk_109 + 202876*uk_11 + 2512*uk_110 + 192*uk_111 + 2576*uk_112 + 3600*uk_113 + 2512*uk_114 + 98596*uk_115 + 7536*uk_116 + 101108*uk_117 + 141300*uk_118 + 98596*uk_119 + 7962883*uk_12 + 576*uk_120 + 7728*uk_121 + 10800*uk_122 + 7536*uk_123 + 103684*uk_124 + 144900*uk_125 + 101108*uk_126 + 202500*uk_127 + 141300*uk_128 + 98596*uk_129 + 608628*uk_13 + 3869893*uk_130 + 295788*uk_131 + 3968489*uk_132 + 5546025*uk_133 + 3869893*uk_134 + 22608*uk_135 + 303324*uk_136 + 423900*uk_137 + 295788*uk_138 + 4069597*uk_139 + 8165759*uk_14 + 5687325*uk_140 + 3968489*uk_141 + 7948125*uk_142 + 5546025*uk_143 + 3869893*uk_144 + 1728*uk_145 + 23184*uk_146 + 32400*uk_147 + 22608*uk_148 + 311052*uk_149 + 11411775*uk_15 + 434700*uk_150 + 303324*uk_151 + 607500*uk_152 + 423900*uk_153 + 295788*uk_154 + 4173281*uk_155 + 5832225*uk_156 + 4069597*uk_157 + 8150625*uk_158 + 5687325*uk_159 + 7962883*uk_16 + 3968489*uk_160 + 11390625*uk_161 + 7948125*uk_162 + 5546025*uk_163 + 3869893*uk_164 + 3025*uk_17 + 220*uk_18 + 8635*uk_19 + 55*uk_2 + 660*uk_20 + 8855*uk_21 + 12375*uk_22 + 8635*uk_23 + 16*uk_24 + 628*uk_25 + 48*uk_26 + 644*uk_27 + 900*uk_28 + 628*uk_29 + 4*uk_3 + 24649*uk_30 + 1884*uk_31 + 25277*uk_32 + 35325*uk_33 + 24649*uk_34 + 144*uk_35 + 1932*uk_36 + 2700*uk_37 + 1884*uk_38 + 25921*uk_39 + 157*uk_4 + 36225*uk_40 + 25277*uk_41 + 50625*uk_42 + 35325*uk_43 + 24649*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 10289667844*uk_47 + 403869462877*uk_48 + 30869003532*uk_49 + 12*uk_5 + 414159130721*uk_50 + 578793816225*uk_51 + 403869462877*uk_52 + 153424975*uk_53 + 11158180*uk_54 + 437958565*uk_55 + 33474540*uk_56 + 449116745*uk_57 + 627647625*uk_58 + 437958565*uk_59 + 161*uk_6 + 811504*uk_60 + 31851532*uk_61 + 2434512*uk_62 + 32663036*uk_63 + 45647100*uk_64 + 31851532*uk_65 + 1250172631*uk_66 + 95554596*uk_67 + 1282024163*uk_68 + 1791648675*uk_69 + 225*uk_7 + 1250172631*uk_70 + 7303536*uk_71 + 97989108*uk_72 + 136941300*uk_73 + 95554596*uk_74 + 1314687199*uk_75 + 1837295775*uk_76 + 1282024163*uk_77 + 2567649375*uk_78 + 1791648675*uk_79 + 157*uk_8 + 1250172631*uk_80 + 166375*uk_81 + 12100*uk_82 + 474925*uk_83 + 36300*uk_84 + 487025*uk_85 + 680625*uk_86 + 474925*uk_87 + 880*uk_88 + 34540*uk_89 + 2572416961*uk_9 + 2640*uk_90 + 35420*uk_91 + 49500*uk_92 + 34540*uk_93 + 1355695*uk_94 + 103620*uk_95 + 1390235*uk_96 + 1942875*uk_97 + 1355695*uk_98 + 7920*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 143440*uk_100 + 198000*uk_101 + 3520*uk_102 + 1461295*uk_103 + 2017125*uk_104 + 35860*uk_105 + 2784375*uk_106 + 49500*uk_107 + 880*uk_108 + 17576*uk_109 + 1318694*uk_11 + 2704*uk_110 + 10816*uk_111 + 110188*uk_112 + 152100*uk_113 + 2704*uk_114 + 416*uk_115 + 1664*uk_116 + 16952*uk_117 + 23400*uk_118 + 416*uk_119 + 202876*uk_12 + 6656*uk_120 + 67808*uk_121 + 93600*uk_122 + 1664*uk_123 + 690794*uk_124 + 953550*uk_125 + 16952*uk_126 + 1316250*uk_127 + 23400*uk_128 + 416*uk_129 + 811504*uk_13 + 64*uk_130 + 256*uk_131 + 2608*uk_132 + 3600*uk_133 + 64*uk_134 + 1024*uk_135 + 10432*uk_136 + 14400*uk_137 + 256*uk_138 + 106276*uk_139 + 8267197*uk_14 + 146700*uk_140 + 2608*uk_141 + 202500*uk_142 + 3600*uk_143 + 64*uk_144 + 4096*uk_145 + 41728*uk_146 + 57600*uk_147 + 1024*uk_148 + 425104*uk_149 + 11411775*uk_15 + 586800*uk_150 + 10432*uk_151 + 810000*uk_152 + 14400*uk_153 + 256*uk_154 + 4330747*uk_155 + 5978025*uk_156 + 106276*uk_157 + 8251875*uk_158 + 146700*uk_159 + 202876*uk_16 + 2608*uk_160 + 11390625*uk_161 + 202500*uk_162 + 3600*uk_163 + 64*uk_164 + 3025*uk_17 + 1430*uk_18 + 220*uk_19 + 55*uk_2 + 880*uk_20 + 8965*uk_21 + 12375*uk_22 + 220*uk_23 + 676*uk_24 + 104*uk_25 + 416*uk_26 + 4238*uk_27 + 5850*uk_28 + 104*uk_29 + 26*uk_3 + 16*uk_30 + 64*uk_31 + 652*uk_32 + 900*uk_33 + 16*uk_34 + 256*uk_35 + 2608*uk_36 + 3600*uk_37 + 64*uk_38 + 26569*uk_39 + 4*uk_4 + 36675*uk_40 + 652*uk_41 + 50625*uk_42 + 900*uk_43 + 16*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 66882840986*uk_47 + 10289667844*uk_48 + 41158671376*uk_49 + 16*uk_5 + 419303964643*uk_50 + 578793816225*uk_51 + 10289667844*uk_52 + 153424975*uk_53 + 72528170*uk_54 + 11158180*uk_55 + 44632720*uk_56 + 454695835*uk_57 + 627647625*uk_58 + 11158180*uk_59 + 163*uk_6 + 34286044*uk_60 + 5274776*uk_61 + 21099104*uk_62 + 214947122*uk_63 + 296706150*uk_64 + 5274776*uk_65 + 811504*uk_66 + 3246016*uk_67 + 33068788*uk_68 + 45647100*uk_69 + 225*uk_7 + 811504*uk_70 + 12984064*uk_71 + 132275152*uk_72 + 182588400*uk_73 + 3246016*uk_74 + 1347553111*uk_75 + 1860119325*uk_76 + 33068788*uk_77 + 2567649375*uk_78 + 45647100*uk_79 + 4*uk_8 + 811504*uk_80 + 166375*uk_81 + 78650*uk_82 + 12100*uk_83 + 48400*uk_84 + 493075*uk_85 + 680625*uk_86 + 12100*uk_87 + 37180*uk_88 + 5720*uk_89 + 2572416961*uk_9 + 22880*uk_90 + 233090*uk_91 + 321750*uk_92 + 5720*uk_93 + 880*uk_94 + 3520*uk_95 + 35860*uk_96 + 49500*uk_97 + 880*uk_98 + 14080*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 145200*uk_100 + 198000*uk_101 + 22880*uk_102 + 1497375*uk_103 + 2041875*uk_104 + 235950*uk_105 + 2784375*uk_106 + 321750*uk_107 + 37180*uk_108 + 262144*uk_109 + 3246016*uk_11 + 106496*uk_110 + 65536*uk_111 + 675840*uk_112 + 921600*uk_113 + 106496*uk_114 + 43264*uk_115 + 26624*uk_116 + 274560*uk_117 + 374400*uk_118 + 43264*uk_119 + 1318694*uk_12 + 16384*uk_120 + 168960*uk_121 + 230400*uk_122 + 26624*uk_123 + 1742400*uk_124 + 2376000*uk_125 + 274560*uk_126 + 3240000*uk_127 + 374400*uk_128 + 43264*uk_129 + 811504*uk_13 + 17576*uk_130 + 10816*uk_131 + 111540*uk_132 + 152100*uk_133 + 17576*uk_134 + 6656*uk_135 + 68640*uk_136 + 93600*uk_137 + 10816*uk_138 + 707850*uk_139 + 8368635*uk_14 + 965250*uk_140 + 111540*uk_141 + 1316250*uk_142 + 152100*uk_143 + 17576*uk_144 + 4096*uk_145 + 42240*uk_146 + 57600*uk_147 + 6656*uk_148 + 435600*uk_149 + 11411775*uk_15 + 594000*uk_150 + 68640*uk_151 + 810000*uk_152 + 93600*uk_153 + 10816*uk_154 + 4492125*uk_155 + 6125625*uk_156 + 707850*uk_157 + 8353125*uk_158 + 965250*uk_159 + 1318694*uk_16 + 111540*uk_160 + 11390625*uk_161 + 1316250*uk_162 + 152100*uk_163 + 17576*uk_164 + 3025*uk_17 + 3520*uk_18 + 1430*uk_19 + 55*uk_2 + 880*uk_20 + 9075*uk_21 + 12375*uk_22 + 1430*uk_23 + 4096*uk_24 + 1664*uk_25 + 1024*uk_26 + 10560*uk_27 + 14400*uk_28 + 1664*uk_29 + 64*uk_3 + 676*uk_30 + 416*uk_31 + 4290*uk_32 + 5850*uk_33 + 676*uk_34 + 256*uk_35 + 2640*uk_36 + 3600*uk_37 + 416*uk_38 + 27225*uk_39 + 26*uk_4 + 37125*uk_40 + 4290*uk_41 + 50625*uk_42 + 5850*uk_43 + 676*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 164634685504*uk_47 + 66882840986*uk_48 + 41158671376*uk_49 + 16*uk_5 + 424448798565*uk_50 + 578793816225*uk_51 + 66882840986*uk_52 + 153424975*uk_53 + 178530880*uk_54 + 72528170*uk_55 + 44632720*uk_56 + 460274925*uk_57 + 627647625*uk_58 + 72528170*uk_59 + 165*uk_6 + 207745024*uk_60 + 84396416*uk_61 + 51936256*uk_62 + 535592640*uk_63 + 730353600*uk_64 + 84396416*uk_65 + 34286044*uk_66 + 21099104*uk_67 + 217584510*uk_68 + 296706150*uk_69 + 225*uk_7 + 34286044*uk_70 + 12984064*uk_71 + 133898160*uk_72 + 182588400*uk_73 + 21099104*uk_74 + 1380824775*uk_75 + 1882942875*uk_76 + 217584510*uk_77 + 2567649375*uk_78 + 296706150*uk_79 + 26*uk_8 + 34286044*uk_80 + 166375*uk_81 + 193600*uk_82 + 78650*uk_83 + 48400*uk_84 + 499125*uk_85 + 680625*uk_86 + 78650*uk_87 + 225280*uk_88 + 91520*uk_89 + 2572416961*uk_9 + 56320*uk_90 + 580800*uk_91 + 792000*uk_92 + 91520*uk_93 + 37180*uk_94 + 22880*uk_95 + 235950*uk_96 + 321750*uk_97 + 37180*uk_98 + 14080*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 146960*uk_100 + 198000*uk_101 + 56320*uk_102 + 1533895*uk_103 + 2066625*uk_104 + 587840*uk_105 + 2784375*uk_106 + 792000*uk_107 + 225280*uk_108 + 1643032*uk_109 + 5984842*uk_11 + 891136*uk_110 + 222784*uk_111 + 2325308*uk_112 + 3132900*uk_113 + 891136*uk_114 + 483328*uk_115 + 120832*uk_116 + 1261184*uk_117 + 1699200*uk_118 + 483328*uk_119 + 3246016*uk_12 + 30208*uk_120 + 315296*uk_121 + 424800*uk_122 + 120832*uk_123 + 3290902*uk_124 + 4433850*uk_125 + 1261184*uk_126 + 5973750*uk_127 + 1699200*uk_128 + 483328*uk_129 + 811504*uk_13 + 262144*uk_130 + 65536*uk_131 + 684032*uk_132 + 921600*uk_133 + 262144*uk_134 + 16384*uk_135 + 171008*uk_136 + 230400*uk_137 + 65536*uk_138 + 1784896*uk_139 + 8470073*uk_14 + 2404800*uk_140 + 684032*uk_141 + 3240000*uk_142 + 921600*uk_143 + 262144*uk_144 + 4096*uk_145 + 42752*uk_146 + 57600*uk_147 + 16384*uk_148 + 446224*uk_149 + 11411775*uk_15 + 601200*uk_150 + 171008*uk_151 + 810000*uk_152 + 230400*uk_153 + 65536*uk_154 + 4657463*uk_155 + 6275025*uk_156 + 1784896*uk_157 + 8454375*uk_158 + 2404800*uk_159 + 3246016*uk_16 + 684032*uk_160 + 11390625*uk_161 + 3240000*uk_162 + 921600*uk_163 + 262144*uk_164 + 3025*uk_17 + 6490*uk_18 + 3520*uk_19 + 55*uk_2 + 880*uk_20 + 9185*uk_21 + 12375*uk_22 + 3520*uk_23 + 13924*uk_24 + 7552*uk_25 + 1888*uk_26 + 19706*uk_27 + 26550*uk_28 + 7552*uk_29 + 118*uk_3 + 4096*uk_30 + 1024*uk_31 + 10688*uk_32 + 14400*uk_33 + 4096*uk_34 + 256*uk_35 + 2672*uk_36 + 3600*uk_37 + 1024*uk_38 + 27889*uk_39 + 64*uk_4 + 37575*uk_40 + 10688*uk_41 + 50625*uk_42 + 14400*uk_43 + 4096*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 303545201398*uk_47 + 164634685504*uk_48 + 41158671376*uk_49 + 16*uk_5 + 429593632487*uk_50 + 578793816225*uk_51 + 164634685504*uk_52 + 153424975*uk_53 + 329166310*uk_54 + 178530880*uk_55 + 44632720*uk_56 + 465854015*uk_57 + 627647625*uk_58 + 178530880*uk_59 + 167*uk_6 + 706211356*uk_60 + 383029888*uk_61 + 95757472*uk_62 + 999468614*uk_63 + 1346589450*uk_64 + 383029888*uk_65 + 207745024*uk_66 + 51936256*uk_67 + 542084672*uk_68 + 730353600*uk_69 + 225*uk_7 + 207745024*uk_70 + 12984064*uk_71 + 135521168*uk_72 + 182588400*uk_73 + 51936256*uk_74 + 1414502191*uk_75 + 1905766425*uk_76 + 542084672*uk_77 + 2567649375*uk_78 + 730353600*uk_79 + 64*uk_8 + 207745024*uk_80 + 166375*uk_81 + 356950*uk_82 + 193600*uk_83 + 48400*uk_84 + 505175*uk_85 + 680625*uk_86 + 193600*uk_87 + 765820*uk_88 + 415360*uk_89 + 2572416961*uk_9 + 103840*uk_90 + 1083830*uk_91 + 1460250*uk_92 + 415360*uk_93 + 225280*uk_94 + 56320*uk_95 + 587840*uk_96 + 792000*uk_97 + 225280*uk_98 + 14080*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 111540*uk_100 + 148500*uk_101 + 77880*uk_102 + 1570855*uk_103 + 2091375*uk_104 + 1096810*uk_105 + 2784375*uk_106 + 1460250*uk_107 + 765820*uk_108 + 6859*uk_109 + 963661*uk_11 + 42598*uk_110 + 4332*uk_111 + 61009*uk_112 + 81225*uk_113 + 42598*uk_114 + 264556*uk_115 + 26904*uk_116 + 378898*uk_117 + 504450*uk_118 + 264556*uk_119 + 5984842*uk_12 + 2736*uk_120 + 38532*uk_121 + 51300*uk_122 + 26904*uk_123 + 542659*uk_124 + 722475*uk_125 + 378898*uk_126 + 961875*uk_127 + 504450*uk_128 + 264556*uk_129 + 608628*uk_13 + 1643032*uk_130 + 167088*uk_131 + 2353156*uk_132 + 3132900*uk_133 + 1643032*uk_134 + 16992*uk_135 + 239304*uk_136 + 318600*uk_137 + 167088*uk_138 + 3370198*uk_139 + 8571511*uk_14 + 4486950*uk_140 + 2353156*uk_141 + 5973750*uk_142 + 3132900*uk_143 + 1643032*uk_144 + 1728*uk_145 + 24336*uk_146 + 32400*uk_147 + 16992*uk_148 + 342732*uk_149 + 11411775*uk_15 + 456300*uk_150 + 239304*uk_151 + 607500*uk_152 + 318600*uk_153 + 167088*uk_154 + 4826809*uk_155 + 6426225*uk_156 + 3370198*uk_157 + 8555625*uk_158 + 4486950*uk_159 + 5984842*uk_16 + 2353156*uk_160 + 11390625*uk_161 + 5973750*uk_162 + 3132900*uk_163 + 1643032*uk_164 + 3025*uk_17 + 1045*uk_18 + 6490*uk_19 + 55*uk_2 + 660*uk_20 + 9295*uk_21 + 12375*uk_22 + 6490*uk_23 + 361*uk_24 + 2242*uk_25 + 228*uk_26 + 3211*uk_27 + 4275*uk_28 + 2242*uk_29 + 19*uk_3 + 13924*uk_30 + 1416*uk_31 + 19942*uk_32 + 26550*uk_33 + 13924*uk_34 + 144*uk_35 + 2028*uk_36 + 2700*uk_37 + 1416*uk_38 + 28561*uk_39 + 118*uk_4 + 38025*uk_40 + 19942*uk_41 + 50625*uk_42 + 26550*uk_43 + 13924*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 48875922259*uk_47 + 303545201398*uk_48 + 30869003532*uk_49 + 12*uk_5 + 434738466409*uk_50 + 578793816225*uk_51 + 303545201398*uk_52 + 153424975*uk_53 + 53001355*uk_54 + 329166310*uk_55 + 33474540*uk_56 + 471433105*uk_57 + 627647625*uk_58 + 329166310*uk_59 + 169*uk_6 + 18309559*uk_60 + 113711998*uk_61 + 11563932*uk_62 + 162858709*uk_63 + 216823725*uk_64 + 113711998*uk_65 + 706211356*uk_66 + 71818104*uk_67 + 1011438298*uk_68 + 1346589450*uk_69 + 225*uk_7 + 706211356*uk_70 + 7303536*uk_71 + 102858132*uk_72 + 136941300*uk_73 + 71818104*uk_74 + 1448585359*uk_75 + 1928589975*uk_76 + 1011438298*uk_77 + 2567649375*uk_78 + 1346589450*uk_79 + 118*uk_8 + 706211356*uk_80 + 166375*uk_81 + 57475*uk_82 + 356950*uk_83 + 36300*uk_84 + 511225*uk_85 + 680625*uk_86 + 356950*uk_87 + 19855*uk_88 + 123310*uk_89 + 2572416961*uk_9 + 12540*uk_90 + 176605*uk_91 + 235125*uk_92 + 123310*uk_93 + 765820*uk_94 + 77880*uk_95 + 1096810*uk_96 + 1460250*uk_97 + 765820*uk_98 + 7920*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 150480*uk_100 + 198000*uk_101 + 16720*uk_102 + 1608255*uk_103 + 2116125*uk_104 + 178695*uk_105 + 2784375*uk_106 + 235125*uk_107 + 19855*uk_108 + 1092727*uk_109 + 5224057*uk_11 + 201571*uk_110 + 169744*uk_111 + 1814139*uk_112 + 2387025*uk_113 + 201571*uk_114 + 37183*uk_115 + 31312*uk_116 + 334647*uk_117 + 440325*uk_118 + 37183*uk_119 + 963661*uk_12 + 26368*uk_120 + 281808*uk_121 + 370800*uk_122 + 31312*uk_123 + 3011823*uk_124 + 3962925*uk_125 + 334647*uk_126 + 5214375*uk_127 + 440325*uk_128 + 37183*uk_129 + 811504*uk_13 + 6859*uk_130 + 5776*uk_131 + 61731*uk_132 + 81225*uk_133 + 6859*uk_134 + 4864*uk_135 + 51984*uk_136 + 68400*uk_137 + 5776*uk_138 + 555579*uk_139 + 8672949*uk_14 + 731025*uk_140 + 61731*uk_141 + 961875*uk_142 + 81225*uk_143 + 6859*uk_144 + 4096*uk_145 + 43776*uk_146 + 57600*uk_147 + 4864*uk_148 + 467856*uk_149 + 11411775*uk_15 + 615600*uk_150 + 51984*uk_151 + 810000*uk_152 + 68400*uk_153 + 5776*uk_154 + 5000211*uk_155 + 6579225*uk_156 + 555579*uk_157 + 8656875*uk_158 + 731025*uk_159 + 963661*uk_16 + 61731*uk_160 + 11390625*uk_161 + 961875*uk_162 + 81225*uk_163 + 6859*uk_164 + 3025*uk_17 + 5665*uk_18 + 1045*uk_19 + 55*uk_2 + 880*uk_20 + 9405*uk_21 + 12375*uk_22 + 1045*uk_23 + 10609*uk_24 + 1957*uk_25 + 1648*uk_26 + 17613*uk_27 + 23175*uk_28 + 1957*uk_29 + 103*uk_3 + 361*uk_30 + 304*uk_31 + 3249*uk_32 + 4275*uk_33 + 361*uk_34 + 256*uk_35 + 2736*uk_36 + 3600*uk_37 + 304*uk_38 + 29241*uk_39 + 19*uk_4 + 38475*uk_40 + 3249*uk_41 + 50625*uk_42 + 4275*uk_43 + 361*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 264958946983*uk_47 + 48875922259*uk_48 + 41158671376*uk_49 + 16*uk_5 + 439883300331*uk_50 + 578793816225*uk_51 + 48875922259*uk_52 + 153424975*uk_53 + 287323135*uk_54 + 53001355*uk_55 + 44632720*uk_56 + 477012195*uk_57 + 627647625*uk_58 + 53001355*uk_59 + 171*uk_6 + 538077871*uk_60 + 99257083*uk_61 + 83584912*uk_62 + 893313747*uk_63 + 1175412825*uk_64 + 99257083*uk_65 + 18309559*uk_66 + 15418576*uk_67 + 164786031*uk_68 + 216823725*uk_69 + 225*uk_7 + 18309559*uk_70 + 12984064*uk_71 + 138767184*uk_72 + 182588400*uk_73 + 15418576*uk_74 + 1483074279*uk_75 + 1951413525*uk_76 + 164786031*uk_77 + 2567649375*uk_78 + 216823725*uk_79 + 19*uk_8 + 18309559*uk_80 + 166375*uk_81 + 311575*uk_82 + 57475*uk_83 + 48400*uk_84 + 517275*uk_85 + 680625*uk_86 + 57475*uk_87 + 583495*uk_88 + 107635*uk_89 + 2572416961*uk_9 + 90640*uk_90 + 968715*uk_91 + 1274625*uk_92 + 107635*uk_93 + 19855*uk_94 + 16720*uk_95 + 178695*uk_96 + 235125*uk_97 + 19855*uk_98 + 14080*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 114180*uk_100 + 148500*uk_101 + 67980*uk_102 + 1646095*uk_103 + 2140875*uk_104 + 980045*uk_105 + 2784375*uk_106 + 1274625*uk_107 + 583495*uk_108 + 27000*uk_109 + 1521570*uk_11 + 92700*uk_110 + 10800*uk_111 + 155700*uk_112 + 202500*uk_113 + 92700*uk_114 + 318270*uk_115 + 37080*uk_116 + 534570*uk_117 + 695250*uk_118 + 318270*uk_119 + 5224057*uk_12 + 4320*uk_120 + 62280*uk_121 + 81000*uk_122 + 37080*uk_123 + 897870*uk_124 + 1167750*uk_125 + 534570*uk_126 + 1518750*uk_127 + 695250*uk_128 + 318270*uk_129 + 608628*uk_13 + 1092727*uk_130 + 127308*uk_131 + 1835357*uk_132 + 2387025*uk_133 + 1092727*uk_134 + 14832*uk_135 + 213828*uk_136 + 278100*uk_137 + 127308*uk_138 + 3082687*uk_139 + 8774387*uk_14 + 4009275*uk_140 + 1835357*uk_141 + 5214375*uk_142 + 2387025*uk_143 + 1092727*uk_144 + 1728*uk_145 + 24912*uk_146 + 32400*uk_147 + 14832*uk_148 + 359148*uk_149 + 11411775*uk_15 + 467100*uk_150 + 213828*uk_151 + 607500*uk_152 + 278100*uk_153 + 127308*uk_154 + 5177717*uk_155 + 6734025*uk_156 + 3082687*uk_157 + 8758125*uk_158 + 4009275*uk_159 + 5224057*uk_16 + 1835357*uk_160 + 11390625*uk_161 + 5214375*uk_162 + 2387025*uk_163 + 1092727*uk_164 + 3025*uk_17 + 1650*uk_18 + 5665*uk_19 + 55*uk_2 + 660*uk_20 + 9515*uk_21 + 12375*uk_22 + 5665*uk_23 + 900*uk_24 + 3090*uk_25 + 360*uk_26 + 5190*uk_27 + 6750*uk_28 + 3090*uk_29 + 30*uk_3 + 10609*uk_30 + 1236*uk_31 + 17819*uk_32 + 23175*uk_33 + 10609*uk_34 + 144*uk_35 + 2076*uk_36 + 2700*uk_37 + 1236*uk_38 + 29929*uk_39 + 103*uk_4 + 38925*uk_40 + 17819*uk_41 + 50625*uk_42 + 23175*uk_43 + 10609*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 77172508830*uk_47 + 264958946983*uk_48 + 30869003532*uk_49 + 12*uk_5 + 445028134253*uk_50 + 578793816225*uk_51 + 264958946983*uk_52 + 153424975*uk_53 + 83686350*uk_54 + 287323135*uk_55 + 33474540*uk_56 + 482591285*uk_57 + 627647625*uk_58 + 287323135*uk_59 + 173*uk_6 + 45647100*uk_60 + 156721710*uk_61 + 18258840*uk_62 + 263231610*uk_63 + 342353250*uk_64 + 156721710*uk_65 + 538077871*uk_66 + 62688684*uk_67 + 903761861*uk_68 + 1175412825*uk_69 + 225*uk_7 + 538077871*uk_70 + 7303536*uk_71 + 105292644*uk_72 + 136941300*uk_73 + 62688684*uk_74 + 1517968951*uk_75 + 1974237075*uk_76 + 903761861*uk_77 + 2567649375*uk_78 + 1175412825*uk_79 + 103*uk_8 + 538077871*uk_80 + 166375*uk_81 + 90750*uk_82 + 311575*uk_83 + 36300*uk_84 + 523325*uk_85 + 680625*uk_86 + 311575*uk_87 + 49500*uk_88 + 169950*uk_89 + 2572416961*uk_9 + 19800*uk_90 + 285450*uk_91 + 371250*uk_92 + 169950*uk_93 + 583495*uk_94 + 67980*uk_95 + 980045*uk_96 + 1274625*uk_97 + 583495*uk_98 + 7920*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 154000*uk_100 + 198000*uk_101 + 26400*uk_102 + 1684375*uk_103 + 2165625*uk_104 + 288750*uk_105 + 2784375*uk_106 + 371250*uk_107 + 49500*uk_108 + 2985984*uk_109 + 7303536*uk_11 + 622080*uk_110 + 331776*uk_111 + 3628800*uk_112 + 4665600*uk_113 + 622080*uk_114 + 129600*uk_115 + 69120*uk_116 + 756000*uk_117 + 972000*uk_118 + 129600*uk_119 + 1521570*uk_12 + 36864*uk_120 + 403200*uk_121 + 518400*uk_122 + 69120*uk_123 + 4410000*uk_124 + 5670000*uk_125 + 756000*uk_126 + 7290000*uk_127 + 972000*uk_128 + 129600*uk_129 + 811504*uk_13 + 27000*uk_130 + 14400*uk_131 + 157500*uk_132 + 202500*uk_133 + 27000*uk_134 + 7680*uk_135 + 84000*uk_136 + 108000*uk_137 + 14400*uk_138 + 918750*uk_139 + 8875825*uk_14 + 1181250*uk_140 + 157500*uk_141 + 1518750*uk_142 + 202500*uk_143 + 27000*uk_144 + 4096*uk_145 + 44800*uk_146 + 57600*uk_147 + 7680*uk_148 + 490000*uk_149 + 11411775*uk_15 + 630000*uk_150 + 84000*uk_151 + 810000*uk_152 + 108000*uk_153 + 14400*uk_154 + 5359375*uk_155 + 6890625*uk_156 + 918750*uk_157 + 8859375*uk_158 + 1181250*uk_159 + 1521570*uk_16 + 157500*uk_160 + 11390625*uk_161 + 1518750*uk_162 + 202500*uk_163 + 27000*uk_164 + 3025*uk_17 + 7920*uk_18 + 1650*uk_19 + 55*uk_2 + 880*uk_20 + 9625*uk_21 + 12375*uk_22 + 1650*uk_23 + 20736*uk_24 + 4320*uk_25 + 2304*uk_26 + 25200*uk_27 + 32400*uk_28 + 4320*uk_29 + 144*uk_3 + 900*uk_30 + 480*uk_31 + 5250*uk_32 + 6750*uk_33 + 900*uk_34 + 256*uk_35 + 2800*uk_36 + 3600*uk_37 + 480*uk_38 + 30625*uk_39 + 30*uk_4 + 39375*uk_40 + 5250*uk_41 + 50625*uk_42 + 6750*uk_43 + 900*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 370428042384*uk_47 + 77172508830*uk_48 + 41158671376*uk_49 + 16*uk_5 + 450172968175*uk_50 + 578793816225*uk_51 + 77172508830*uk_52 + 153424975*uk_53 + 401694480*uk_54 + 83686350*uk_55 + 44632720*uk_56 + 488170375*uk_57 + 627647625*uk_58 + 83686350*uk_59 + 175*uk_6 + 1051709184*uk_60 + 219106080*uk_61 + 116856576*uk_62 + 1278118800*uk_63 + 1643295600*uk_64 + 219106080*uk_65 + 45647100*uk_66 + 24345120*uk_67 + 266274750*uk_68 + 342353250*uk_69 + 225*uk_7 + 45647100*uk_70 + 12984064*uk_71 + 142013200*uk_72 + 182588400*uk_73 + 24345120*uk_74 + 1553269375*uk_75 + 1997060625*uk_76 + 266274750*uk_77 + 2567649375*uk_78 + 342353250*uk_79 + 30*uk_8 + 45647100*uk_80 + 166375*uk_81 + 435600*uk_82 + 90750*uk_83 + 48400*uk_84 + 529375*uk_85 + 680625*uk_86 + 90750*uk_87 + 1140480*uk_88 + 237600*uk_89 + 2572416961*uk_9 + 126720*uk_90 + 1386000*uk_91 + 1782000*uk_92 + 237600*uk_93 + 49500*uk_94 + 26400*uk_95 + 288750*uk_96 + 371250*uk_97 + 49500*uk_98 + 14080*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 116820*uk_100 + 148500*uk_101 + 95040*uk_102 + 1723095*uk_103 + 2190375*uk_104 + 1401840*uk_105 + 2784375*uk_106 + 1782000*uk_107 + 1140480*uk_108 + 912673*uk_109 + 4919743*uk_11 + 1354896*uk_110 + 112908*uk_111 + 1665393*uk_112 + 2117025*uk_113 + 1354896*uk_114 + 2011392*uk_115 + 167616*uk_116 + 2472336*uk_117 + 3142800*uk_118 + 2011392*uk_119 + 7303536*uk_12 + 13968*uk_120 + 206028*uk_121 + 261900*uk_122 + 167616*uk_123 + 3038913*uk_124 + 3863025*uk_125 + 2472336*uk_126 + 4910625*uk_127 + 3142800*uk_128 + 2011392*uk_129 + 608628*uk_13 + 2985984*uk_130 + 248832*uk_131 + 3670272*uk_132 + 4665600*uk_133 + 2985984*uk_134 + 20736*uk_135 + 305856*uk_136 + 388800*uk_137 + 248832*uk_138 + 4511376*uk_139 + 8977263*uk_14 + 5734800*uk_140 + 3670272*uk_141 + 7290000*uk_142 + 4665600*uk_143 + 2985984*uk_144 + 1728*uk_145 + 25488*uk_146 + 32400*uk_147 + 20736*uk_148 + 375948*uk_149 + 11411775*uk_15 + 477900*uk_150 + 305856*uk_151 + 607500*uk_152 + 388800*uk_153 + 248832*uk_154 + 5545233*uk_155 + 7049025*uk_156 + 4511376*uk_157 + 8960625*uk_158 + 5734800*uk_159 + 7303536*uk_16 + 3670272*uk_160 + 11390625*uk_161 + 7290000*uk_162 + 4665600*uk_163 + 2985984*uk_164 + 3025*uk_17 + 5335*uk_18 + 7920*uk_19 + 55*uk_2 + 660*uk_20 + 9735*uk_21 + 12375*uk_22 + 7920*uk_23 + 9409*uk_24 + 13968*uk_25 + 1164*uk_26 + 17169*uk_27 + 21825*uk_28 + 13968*uk_29 + 97*uk_3 + 20736*uk_30 + 1728*uk_31 + 25488*uk_32 + 32400*uk_33 + 20736*uk_34 + 144*uk_35 + 2124*uk_36 + 2700*uk_37 + 1728*uk_38 + 31329*uk_39 + 144*uk_4 + 39825*uk_40 + 25488*uk_41 + 50625*uk_42 + 32400*uk_43 + 20736*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 249524445217*uk_47 + 370428042384*uk_48 + 30869003532*uk_49 + 12*uk_5 + 455317802097*uk_50 + 578793816225*uk_51 + 370428042384*uk_52 + 153424975*uk_53 + 270585865*uk_54 + 401694480*uk_55 + 33474540*uk_56 + 493749465*uk_57 + 627647625*uk_58 + 401694480*uk_59 + 177*uk_6 + 477215071*uk_60 + 708442992*uk_61 + 59036916*uk_62 + 870794511*uk_63 + 1106942175*uk_64 + 708442992*uk_65 + 1051709184*uk_66 + 87642432*uk_67 + 1292725872*uk_68 + 1643295600*uk_69 + 225*uk_7 + 1051709184*uk_70 + 7303536*uk_71 + 107727156*uk_72 + 136941300*uk_73 + 87642432*uk_74 + 1588975551*uk_75 + 2019884175*uk_76 + 1292725872*uk_77 + 2567649375*uk_78 + 1643295600*uk_79 + 144*uk_8 + 1051709184*uk_80 + 166375*uk_81 + 293425*uk_82 + 435600*uk_83 + 36300*uk_84 + 535425*uk_85 + 680625*uk_86 + 435600*uk_87 + 517495*uk_88 + 768240*uk_89 + 2572416961*uk_9 + 64020*uk_90 + 944295*uk_91 + 1200375*uk_92 + 768240*uk_93 + 1140480*uk_94 + 95040*uk_95 + 1401840*uk_96 + 1782000*uk_97 + 1140480*uk_98 + 7920*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 118140*uk_100 + 148500*uk_101 + 64020*uk_102 + 1762255*uk_103 + 2215125*uk_104 + 954965*uk_105 + 2784375*uk_106 + 1200375*uk_107 + 517495*uk_108 + 238328*uk_109 + 3144578*uk_11 + 372868*uk_110 + 46128*uk_111 + 688076*uk_112 + 864900*uk_113 + 372868*uk_114 + 583358*uk_115 + 72168*uk_116 + 1076506*uk_117 + 1353150*uk_118 + 583358*uk_119 + 4919743*uk_12 + 8928*uk_120 + 133176*uk_121 + 167400*uk_122 + 72168*uk_123 + 1986542*uk_124 + 2497050*uk_125 + 1076506*uk_126 + 3138750*uk_127 + 1353150*uk_128 + 583358*uk_129 + 608628*uk_13 + 912673*uk_130 + 112908*uk_131 + 1684211*uk_132 + 2117025*uk_133 + 912673*uk_134 + 13968*uk_135 + 208356*uk_136 + 261900*uk_137 + 112908*uk_138 + 3107977*uk_139 + 9078701*uk_14 + 3906675*uk_140 + 1684211*uk_141 + 4910625*uk_142 + 2117025*uk_143 + 912673*uk_144 + 1728*uk_145 + 25776*uk_146 + 32400*uk_147 + 13968*uk_148 + 384492*uk_149 + 11411775*uk_15 + 483300*uk_150 + 208356*uk_151 + 607500*uk_152 + 261900*uk_153 + 112908*uk_154 + 5735339*uk_155 + 7209225*uk_156 + 3107977*uk_157 + 9061875*uk_158 + 3906675*uk_159 + 4919743*uk_16 + 1684211*uk_160 + 11390625*uk_161 + 4910625*uk_162 + 2117025*uk_163 + 912673*uk_164 + 3025*uk_17 + 3410*uk_18 + 5335*uk_19 + 55*uk_2 + 660*uk_20 + 9845*uk_21 + 12375*uk_22 + 5335*uk_23 + 3844*uk_24 + 6014*uk_25 + 744*uk_26 + 11098*uk_27 + 13950*uk_28 + 6014*uk_29 + 62*uk_3 + 9409*uk_30 + 1164*uk_31 + 17363*uk_32 + 21825*uk_33 + 9409*uk_34 + 144*uk_35 + 2148*uk_36 + 2700*uk_37 + 1164*uk_38 + 32041*uk_39 + 97*uk_4 + 40275*uk_40 + 17363*uk_41 + 50625*uk_42 + 21825*uk_43 + 9409*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 159489851582*uk_47 + 249524445217*uk_48 + 30869003532*uk_49 + 12*uk_5 + 460462636019*uk_50 + 578793816225*uk_51 + 249524445217*uk_52 + 153424975*uk_53 + 172951790*uk_54 + 270585865*uk_55 + 33474540*uk_56 + 499328555*uk_57 + 627647625*uk_58 + 270585865*uk_59 + 179*uk_6 + 194963836*uk_60 + 305024066*uk_61 + 37734936*uk_62 + 562879462*uk_63 + 707530050*uk_64 + 305024066*uk_65 + 477215071*uk_66 + 59036916*uk_67 + 880633997*uk_68 + 1106942175*uk_69 + 225*uk_7 + 477215071*uk_70 + 7303536*uk_71 + 108944412*uk_72 + 136941300*uk_73 + 59036916*uk_74 + 1625087479*uk_75 + 2042707725*uk_76 + 880633997*uk_77 + 2567649375*uk_78 + 1106942175*uk_79 + 97*uk_8 + 477215071*uk_80 + 166375*uk_81 + 187550*uk_82 + 293425*uk_83 + 36300*uk_84 + 541475*uk_85 + 680625*uk_86 + 293425*uk_87 + 211420*uk_88 + 330770*uk_89 + 2572416961*uk_9 + 40920*uk_90 + 610390*uk_91 + 767250*uk_92 + 330770*uk_93 + 517495*uk_94 + 64020*uk_95 + 954965*uk_96 + 1200375*uk_97 + 517495*uk_98 + 7920*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 119460*uk_100 + 148500*uk_101 + 40920*uk_102 + 1801855*uk_103 + 2239875*uk_104 + 617210*uk_105 + 2784375*uk_106 + 767250*uk_107 + 211420*uk_108 + 59319*uk_109 + 1978041*uk_11 + 94302*uk_110 + 18252*uk_111 + 275301*uk_112 + 342225*uk_113 + 94302*uk_114 + 149916*uk_115 + 29016*uk_116 + 437658*uk_117 + 544050*uk_118 + 149916*uk_119 + 3144578*uk_12 + 5616*uk_120 + 84708*uk_121 + 105300*uk_122 + 29016*uk_123 + 1277679*uk_124 + 1588275*uk_125 + 437658*uk_126 + 1974375*uk_127 + 544050*uk_128 + 149916*uk_129 + 608628*uk_13 + 238328*uk_130 + 46128*uk_131 + 695764*uk_132 + 864900*uk_133 + 238328*uk_134 + 8928*uk_135 + 134664*uk_136 + 167400*uk_137 + 46128*uk_138 + 2031182*uk_139 + 9180139*uk_14 + 2524950*uk_140 + 695764*uk_141 + 3138750*uk_142 + 864900*uk_143 + 238328*uk_144 + 1728*uk_145 + 26064*uk_146 + 32400*uk_147 + 8928*uk_148 + 393132*uk_149 + 11411775*uk_15 + 488700*uk_150 + 134664*uk_151 + 607500*uk_152 + 167400*uk_153 + 46128*uk_154 + 5929741*uk_155 + 7371225*uk_156 + 2031182*uk_157 + 9163125*uk_158 + 2524950*uk_159 + 3144578*uk_16 + 695764*uk_160 + 11390625*uk_161 + 3138750*uk_162 + 864900*uk_163 + 238328*uk_164 + 3025*uk_17 + 2145*uk_18 + 3410*uk_19 + 55*uk_2 + 660*uk_20 + 9955*uk_21 + 12375*uk_22 + 3410*uk_23 + 1521*uk_24 + 2418*uk_25 + 468*uk_26 + 7059*uk_27 + 8775*uk_28 + 2418*uk_29 + 39*uk_3 + 3844*uk_30 + 744*uk_31 + 11222*uk_32 + 13950*uk_33 + 3844*uk_34 + 144*uk_35 + 2172*uk_36 + 2700*uk_37 + 744*uk_38 + 32761*uk_39 + 62*uk_4 + 40725*uk_40 + 11222*uk_41 + 50625*uk_42 + 13950*uk_43 + 3844*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 100324261479*uk_47 + 159489851582*uk_48 + 30869003532*uk_49 + 12*uk_5 + 465607469941*uk_50 + 578793816225*uk_51 + 159489851582*uk_52 + 153424975*uk_53 + 108792255*uk_54 + 172951790*uk_55 + 33474540*uk_56 + 504907645*uk_57 + 627647625*uk_58 + 172951790*uk_59 + 181*uk_6 + 77143599*uk_60 + 122638542*uk_61 + 23736492*uk_62 + 358025421*uk_63 + 445059225*uk_64 + 122638542*uk_65 + 194963836*uk_66 + 37734936*uk_67 + 569168618*uk_68 + 707530050*uk_69 + 225*uk_7 + 194963836*uk_70 + 7303536*uk_71 + 110161668*uk_72 + 136941300*uk_73 + 37734936*uk_74 + 1661605159*uk_75 + 2065531275*uk_76 + 569168618*uk_77 + 2567649375*uk_78 + 707530050*uk_79 + 62*uk_8 + 194963836*uk_80 + 166375*uk_81 + 117975*uk_82 + 187550*uk_83 + 36300*uk_84 + 547525*uk_85 + 680625*uk_86 + 187550*uk_87 + 83655*uk_88 + 132990*uk_89 + 2572416961*uk_9 + 25740*uk_90 + 388245*uk_91 + 482625*uk_92 + 132990*uk_93 + 211420*uk_94 + 40920*uk_95 + 617210*uk_96 + 767250*uk_97 + 211420*uk_98 + 7920*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 120780*uk_100 + 148500*uk_101 + 25740*uk_102 + 1841895*uk_103 + 2264625*uk_104 + 392535*uk_105 + 2784375*uk_106 + 482625*uk_107 + 83655*uk_108 + 21952*uk_109 + 1420132*uk_11 + 30576*uk_110 + 9408*uk_111 + 143472*uk_112 + 176400*uk_113 + 30576*uk_114 + 42588*uk_115 + 13104*uk_116 + 199836*uk_117 + 245700*uk_118 + 42588*uk_119 + 1978041*uk_12 + 4032*uk_120 + 61488*uk_121 + 75600*uk_122 + 13104*uk_123 + 937692*uk_124 + 1152900*uk_125 + 199836*uk_126 + 1417500*uk_127 + 245700*uk_128 + 42588*uk_129 + 608628*uk_13 + 59319*uk_130 + 18252*uk_131 + 278343*uk_132 + 342225*uk_133 + 59319*uk_134 + 5616*uk_135 + 85644*uk_136 + 105300*uk_137 + 18252*uk_138 + 1306071*uk_139 + 9281577*uk_14 + 1605825*uk_140 + 278343*uk_141 + 1974375*uk_142 + 342225*uk_143 + 59319*uk_144 + 1728*uk_145 + 26352*uk_146 + 32400*uk_147 + 5616*uk_148 + 401868*uk_149 + 11411775*uk_15 + 494100*uk_150 + 85644*uk_151 + 607500*uk_152 + 105300*uk_153 + 18252*uk_154 + 6128487*uk_155 + 7535025*uk_156 + 1306071*uk_157 + 9264375*uk_158 + 1605825*uk_159 + 1978041*uk_16 + 278343*uk_160 + 11390625*uk_161 + 1974375*uk_162 + 342225*uk_163 + 59319*uk_164 + 3025*uk_17 + 1540*uk_18 + 2145*uk_19 + 55*uk_2 + 660*uk_20 + 10065*uk_21 + 12375*uk_22 + 2145*uk_23 + 784*uk_24 + 1092*uk_25 + 336*uk_26 + 5124*uk_27 + 6300*uk_28 + 1092*uk_29 + 28*uk_3 + 1521*uk_30 + 468*uk_31 + 7137*uk_32 + 8775*uk_33 + 1521*uk_34 + 144*uk_35 + 2196*uk_36 + 2700*uk_37 + 468*uk_38 + 33489*uk_39 + 39*uk_4 + 41175*uk_40 + 7137*uk_41 + 50625*uk_42 + 8775*uk_43 + 1521*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 72027674908*uk_47 + 100324261479*uk_48 + 30869003532*uk_49 + 12*uk_5 + 470752303863*uk_50 + 578793816225*uk_51 + 100324261479*uk_52 + 153424975*uk_53 + 78107260*uk_54 + 108792255*uk_55 + 33474540*uk_56 + 510486735*uk_57 + 627647625*uk_58 + 108792255*uk_59 + 183*uk_6 + 39763696*uk_60 + 55385148*uk_61 + 17041584*uk_62 + 259884156*uk_63 + 319529700*uk_64 + 55385148*uk_65 + 77143599*uk_66 + 23736492*uk_67 + 361981503*uk_68 + 445059225*uk_69 + 225*uk_7 + 77143599*uk_70 + 7303536*uk_71 + 111378924*uk_72 + 136941300*uk_73 + 23736492*uk_74 + 1698528591*uk_75 + 2088354825*uk_76 + 361981503*uk_77 + 2567649375*uk_78 + 445059225*uk_79 + 39*uk_8 + 77143599*uk_80 + 166375*uk_81 + 84700*uk_82 + 117975*uk_83 + 36300*uk_84 + 553575*uk_85 + 680625*uk_86 + 117975*uk_87 + 43120*uk_88 + 60060*uk_89 + 2572416961*uk_9 + 18480*uk_90 + 281820*uk_91 + 346500*uk_92 + 60060*uk_93 + 83655*uk_94 + 25740*uk_95 + 392535*uk_96 + 482625*uk_97 + 83655*uk_98 + 7920*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 122100*uk_100 + 148500*uk_101 + 18480*uk_102 + 1882375*uk_103 + 2289375*uk_104 + 284900*uk_105 + 2784375*uk_106 + 346500*uk_107 + 43120*uk_108 + 24389*uk_109 + 1470851*uk_11 + 23548*uk_110 + 10092*uk_111 + 155585*uk_112 + 189225*uk_113 + 23548*uk_114 + 22736*uk_115 + 9744*uk_116 + 150220*uk_117 + 182700*uk_118 + 22736*uk_119 + 1420132*uk_12 + 4176*uk_120 + 64380*uk_121 + 78300*uk_122 + 9744*uk_123 + 992525*uk_124 + 1207125*uk_125 + 150220*uk_126 + 1468125*uk_127 + 182700*uk_128 + 22736*uk_129 + 608628*uk_13 + 21952*uk_130 + 9408*uk_131 + 145040*uk_132 + 176400*uk_133 + 21952*uk_134 + 4032*uk_135 + 62160*uk_136 + 75600*uk_137 + 9408*uk_138 + 958300*uk_139 + 9383015*uk_14 + 1165500*uk_140 + 145040*uk_141 + 1417500*uk_142 + 176400*uk_143 + 21952*uk_144 + 1728*uk_145 + 26640*uk_146 + 32400*uk_147 + 4032*uk_148 + 410700*uk_149 + 11411775*uk_15 + 499500*uk_150 + 62160*uk_151 + 607500*uk_152 + 75600*uk_153 + 9408*uk_154 + 6331625*uk_155 + 7700625*uk_156 + 958300*uk_157 + 9365625*uk_158 + 1165500*uk_159 + 1420132*uk_16 + 145040*uk_160 + 11390625*uk_161 + 1417500*uk_162 + 176400*uk_163 + 21952*uk_164 + 3025*uk_17 + 1595*uk_18 + 1540*uk_19 + 55*uk_2 + 660*uk_20 + 10175*uk_21 + 12375*uk_22 + 1540*uk_23 + 841*uk_24 + 812*uk_25 + 348*uk_26 + 5365*uk_27 + 6525*uk_28 + 812*uk_29 + 29*uk_3 + 784*uk_30 + 336*uk_31 + 5180*uk_32 + 6300*uk_33 + 784*uk_34 + 144*uk_35 + 2220*uk_36 + 2700*uk_37 + 336*uk_38 + 34225*uk_39 + 28*uk_4 + 41625*uk_40 + 5180*uk_41 + 50625*uk_42 + 6300*uk_43 + 784*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 74600091869*uk_47 + 72027674908*uk_48 + 30869003532*uk_49 + 12*uk_5 + 475897137785*uk_50 + 578793816225*uk_51 + 72027674908*uk_52 + 153424975*uk_53 + 80896805*uk_54 + 78107260*uk_55 + 33474540*uk_56 + 516065825*uk_57 + 627647625*uk_58 + 78107260*uk_59 + 185*uk_6 + 42654679*uk_60 + 41183828*uk_61 + 17650212*uk_62 + 272107435*uk_63 + 330941475*uk_64 + 41183828*uk_65 + 39763696*uk_66 + 17041584*uk_67 + 262724420*uk_68 + 319529700*uk_69 + 225*uk_7 + 39763696*uk_70 + 7303536*uk_71 + 112596180*uk_72 + 136941300*uk_73 + 17041584*uk_74 + 1735857775*uk_75 + 2111178375*uk_76 + 262724420*uk_77 + 2567649375*uk_78 + 319529700*uk_79 + 28*uk_8 + 39763696*uk_80 + 166375*uk_81 + 87725*uk_82 + 84700*uk_83 + 36300*uk_84 + 559625*uk_85 + 680625*uk_86 + 84700*uk_87 + 46255*uk_88 + 44660*uk_89 + 2572416961*uk_9 + 19140*uk_90 + 295075*uk_91 + 358875*uk_92 + 44660*uk_93 + 43120*uk_94 + 18480*uk_95 + 284900*uk_96 + 346500*uk_97 + 43120*uk_98 + 7920*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 123420*uk_100 + 148500*uk_101 + 19140*uk_102 + 1923295*uk_103 + 2314125*uk_104 + 298265*uk_105 + 2784375*uk_106 + 358875*uk_107 + 46255*uk_108 + 74088*uk_109 + 2130198*uk_11 + 51156*uk_110 + 21168*uk_111 + 329868*uk_112 + 396900*uk_113 + 51156*uk_114 + 35322*uk_115 + 14616*uk_116 + 227766*uk_117 + 274050*uk_118 + 35322*uk_119 + 1470851*uk_12 + 6048*uk_120 + 94248*uk_121 + 113400*uk_122 + 14616*uk_123 + 1468698*uk_124 + 1767150*uk_125 + 227766*uk_126 + 2126250*uk_127 + 274050*uk_128 + 35322*uk_129 + 608628*uk_13 + 24389*uk_130 + 10092*uk_131 + 157267*uk_132 + 189225*uk_133 + 24389*uk_134 + 4176*uk_135 + 65076*uk_136 + 78300*uk_137 + 10092*uk_138 + 1014101*uk_139 + 9484453*uk_14 + 1220175*uk_140 + 157267*uk_141 + 1468125*uk_142 + 189225*uk_143 + 24389*uk_144 + 1728*uk_145 + 26928*uk_146 + 32400*uk_147 + 4176*uk_148 + 419628*uk_149 + 11411775*uk_15 + 504900*uk_150 + 65076*uk_151 + 607500*uk_152 + 78300*uk_153 + 10092*uk_154 + 6539203*uk_155 + 7868025*uk_156 + 1014101*uk_157 + 9466875*uk_158 + 1220175*uk_159 + 1470851*uk_16 + 157267*uk_160 + 11390625*uk_161 + 1468125*uk_162 + 189225*uk_163 + 24389*uk_164 + 3025*uk_17 + 2310*uk_18 + 1595*uk_19 + 55*uk_2 + 660*uk_20 + 10285*uk_21 + 12375*uk_22 + 1595*uk_23 + 1764*uk_24 + 1218*uk_25 + 504*uk_26 + 7854*uk_27 + 9450*uk_28 + 1218*uk_29 + 42*uk_3 + 841*uk_30 + 348*uk_31 + 5423*uk_32 + 6525*uk_33 + 841*uk_34 + 144*uk_35 + 2244*uk_36 + 2700*uk_37 + 348*uk_38 + 34969*uk_39 + 29*uk_4 + 42075*uk_40 + 5423*uk_41 + 50625*uk_42 + 6525*uk_43 + 841*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 108041512362*uk_47 + 74600091869*uk_48 + 30869003532*uk_49 + 12*uk_5 + 481041971707*uk_50 + 578793816225*uk_51 + 74600091869*uk_52 + 153424975*uk_53 + 117160890*uk_54 + 80896805*uk_55 + 33474540*uk_56 + 521644915*uk_57 + 627647625*uk_58 + 80896805*uk_59 + 187*uk_6 + 89468316*uk_60 + 61775742*uk_61 + 25562376*uk_62 + 398347026*uk_63 + 479294550*uk_64 + 61775742*uk_65 + 42654679*uk_66 + 17650212*uk_67 + 275049137*uk_68 + 330941475*uk_69 + 225*uk_7 + 42654679*uk_70 + 7303536*uk_71 + 113813436*uk_72 + 136941300*uk_73 + 17650212*uk_74 + 1773592711*uk_75 + 2134001925*uk_76 + 275049137*uk_77 + 2567649375*uk_78 + 330941475*uk_79 + 29*uk_8 + 42654679*uk_80 + 166375*uk_81 + 127050*uk_82 + 87725*uk_83 + 36300*uk_84 + 565675*uk_85 + 680625*uk_86 + 87725*uk_87 + 97020*uk_88 + 66990*uk_89 + 2572416961*uk_9 + 27720*uk_90 + 431970*uk_91 + 519750*uk_92 + 66990*uk_93 + 46255*uk_94 + 19140*uk_95 + 298265*uk_96 + 358875*uk_97 + 46255*uk_98 + 7920*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 124740*uk_100 + 148500*uk_101 + 27720*uk_102 + 1964655*uk_103 + 2338875*uk_104 + 436590*uk_105 + 2784375*uk_106 + 519750*uk_107 + 97020*uk_108 + 300763*uk_109 + 3398173*uk_11 + 188538*uk_110 + 53868*uk_111 + 848421*uk_112 + 1010025*uk_113 + 188538*uk_114 + 118188*uk_115 + 33768*uk_116 + 531846*uk_117 + 633150*uk_118 + 118188*uk_119 + 2130198*uk_12 + 9648*uk_120 + 151956*uk_121 + 180900*uk_122 + 33768*uk_123 + 2393307*uk_124 + 2849175*uk_125 + 531846*uk_126 + 3391875*uk_127 + 633150*uk_128 + 118188*uk_129 + 608628*uk_13 + 74088*uk_130 + 21168*uk_131 + 333396*uk_132 + 396900*uk_133 + 74088*uk_134 + 6048*uk_135 + 95256*uk_136 + 113400*uk_137 + 21168*uk_138 + 1500282*uk_139 + 9585891*uk_14 + 1786050*uk_140 + 333396*uk_141 + 2126250*uk_142 + 396900*uk_143 + 74088*uk_144 + 1728*uk_145 + 27216*uk_146 + 32400*uk_147 + 6048*uk_148 + 428652*uk_149 + 11411775*uk_15 + 510300*uk_150 + 95256*uk_151 + 607500*uk_152 + 113400*uk_153 + 21168*uk_154 + 6751269*uk_155 + 8037225*uk_156 + 1500282*uk_157 + 9568125*uk_158 + 1786050*uk_159 + 2130198*uk_16 + 333396*uk_160 + 11390625*uk_161 + 2126250*uk_162 + 396900*uk_163 + 74088*uk_164 + 3025*uk_17 + 3685*uk_18 + 2310*uk_19 + 55*uk_2 + 660*uk_20 + 10395*uk_21 + 12375*uk_22 + 2310*uk_23 + 4489*uk_24 + 2814*uk_25 + 804*uk_26 + 12663*uk_27 + 15075*uk_28 + 2814*uk_29 + 67*uk_3 + 1764*uk_30 + 504*uk_31 + 7938*uk_32 + 9450*uk_33 + 1764*uk_34 + 144*uk_35 + 2268*uk_36 + 2700*uk_37 + 504*uk_38 + 35721*uk_39 + 42*uk_4 + 42525*uk_40 + 7938*uk_41 + 50625*uk_42 + 9450*uk_43 + 1764*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 172351936387*uk_47 + 108041512362*uk_48 + 30869003532*uk_49 + 12*uk_5 + 486186805629*uk_50 + 578793816225*uk_51 + 108041512362*uk_52 + 153424975*uk_53 + 186899515*uk_54 + 117160890*uk_55 + 33474540*uk_56 + 527224005*uk_57 + 627647625*uk_58 + 117160890*uk_59 + 189*uk_6 + 227677591*uk_60 + 142723266*uk_61 + 40778076*uk_62 + 642254697*uk_63 + 764588925*uk_64 + 142723266*uk_65 + 89468316*uk_66 + 25562376*uk_67 + 402607422*uk_68 + 479294550*uk_69 + 225*uk_7 + 89468316*uk_70 + 7303536*uk_71 + 115030692*uk_72 + 136941300*uk_73 + 25562376*uk_74 + 1811733399*uk_75 + 2156825475*uk_76 + 402607422*uk_77 + 2567649375*uk_78 + 479294550*uk_79 + 42*uk_8 + 89468316*uk_80 + 166375*uk_81 + 202675*uk_82 + 127050*uk_83 + 36300*uk_84 + 571725*uk_85 + 680625*uk_86 + 127050*uk_87 + 246895*uk_88 + 154770*uk_89 + 2572416961*uk_9 + 44220*uk_90 + 696465*uk_91 + 829125*uk_92 + 154770*uk_93 + 97020*uk_94 + 27720*uk_95 + 436590*uk_96 + 519750*uk_97 + 97020*uk_98 + 7920*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 126060*uk_100 + 148500*uk_101 + 44220*uk_102 + 2006455*uk_103 + 2363625*uk_104 + 703835*uk_105 + 2784375*uk_106 + 829125*uk_107 + 246895*uk_108 + 1124864*uk_109 + 5274776*uk_11 + 724672*uk_110 + 129792*uk_111 + 2065856*uk_112 + 2433600*uk_113 + 724672*uk_114 + 466856*uk_115 + 83616*uk_116 + 1330888*uk_117 + 1567800*uk_118 + 466856*uk_119 + 3398173*uk_12 + 14976*uk_120 + 238368*uk_121 + 280800*uk_122 + 83616*uk_123 + 3794024*uk_124 + 4469400*uk_125 + 1330888*uk_126 + 5265000*uk_127 + 1567800*uk_128 + 466856*uk_129 + 608628*uk_13 + 300763*uk_130 + 53868*uk_131 + 857399*uk_132 + 1010025*uk_133 + 300763*uk_134 + 9648*uk_135 + 153564*uk_136 + 180900*uk_137 + 53868*uk_138 + 2444227*uk_139 + 9687329*uk_14 + 2879325*uk_140 + 857399*uk_141 + 3391875*uk_142 + 1010025*uk_143 + 300763*uk_144 + 1728*uk_145 + 27504*uk_146 + 32400*uk_147 + 9648*uk_148 + 437772*uk_149 + 11411775*uk_15 + 515700*uk_150 + 153564*uk_151 + 607500*uk_152 + 180900*uk_153 + 53868*uk_154 + 6967871*uk_155 + 8208225*uk_156 + 2444227*uk_157 + 9669375*uk_158 + 2879325*uk_159 + 3398173*uk_16 + 857399*uk_160 + 11390625*uk_161 + 3391875*uk_162 + 1010025*uk_163 + 300763*uk_164 + 3025*uk_17 + 5720*uk_18 + 3685*uk_19 + 55*uk_2 + 660*uk_20 + 10505*uk_21 + 12375*uk_22 + 3685*uk_23 + 10816*uk_24 + 6968*uk_25 + 1248*uk_26 + 19864*uk_27 + 23400*uk_28 + 6968*uk_29 + 104*uk_3 + 4489*uk_30 + 804*uk_31 + 12797*uk_32 + 15075*uk_33 + 4489*uk_34 + 144*uk_35 + 2292*uk_36 + 2700*uk_37 + 804*uk_38 + 36481*uk_39 + 67*uk_4 + 42975*uk_40 + 12797*uk_41 + 50625*uk_42 + 15075*uk_43 + 4489*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 267531363944*uk_47 + 172351936387*uk_48 + 30869003532*uk_49 + 12*uk_5 + 491331639551*uk_50 + 578793816225*uk_51 + 172351936387*uk_52 + 153424975*uk_53 + 290112680*uk_54 + 186899515*uk_55 + 33474540*uk_56 + 532803095*uk_57 + 627647625*uk_58 + 186899515*uk_59 + 191*uk_6 + 548576704*uk_60 + 353409992*uk_61 + 63297312*uk_62 + 1007482216*uk_63 + 1186824600*uk_64 + 353409992*uk_65 + 227677591*uk_66 + 40778076*uk_67 + 649051043*uk_68 + 764588925*uk_69 + 225*uk_7 + 227677591*uk_70 + 7303536*uk_71 + 116247948*uk_72 + 136941300*uk_73 + 40778076*uk_74 + 1850279839*uk_75 + 2179649025*uk_76 + 649051043*uk_77 + 2567649375*uk_78 + 764588925*uk_79 + 67*uk_8 + 227677591*uk_80 + 166375*uk_81 + 314600*uk_82 + 202675*uk_83 + 36300*uk_84 + 577775*uk_85 + 680625*uk_86 + 202675*uk_87 + 594880*uk_88 + 383240*uk_89 + 2572416961*uk_9 + 68640*uk_90 + 1092520*uk_91 + 1287000*uk_92 + 383240*uk_93 + 246895*uk_94 + 44220*uk_95 + 703835*uk_96 + 829125*uk_97 + 246895*uk_98 + 7920*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 127380*uk_100 + 148500*uk_101 + 68640*uk_102 + 2048695*uk_103 + 2388375*uk_104 + 1103960*uk_105 + 2784375*uk_106 + 1287000*uk_107 + 594880*uk_108 + 3581577*uk_109 + 7760007*uk_11 + 2434536*uk_110 + 280908*uk_111 + 4517937*uk_112 + 5267025*uk_113 + 2434536*uk_114 + 1654848*uk_115 + 190944*uk_116 + 3071016*uk_117 + 3580200*uk_118 + 1654848*uk_119 + 5274776*uk_12 + 22032*uk_120 + 354348*uk_121 + 413100*uk_122 + 190944*uk_123 + 5699097*uk_124 + 6644025*uk_125 + 3071016*uk_126 + 7745625*uk_127 + 3580200*uk_128 + 1654848*uk_129 + 608628*uk_13 + 1124864*uk_130 + 129792*uk_131 + 2087488*uk_132 + 2433600*uk_133 + 1124864*uk_134 + 14976*uk_135 + 240864*uk_136 + 280800*uk_137 + 129792*uk_138 + 3873896*uk_139 + 9788767*uk_14 + 4516200*uk_140 + 2087488*uk_141 + 5265000*uk_142 + 2433600*uk_143 + 1124864*uk_144 + 1728*uk_145 + 27792*uk_146 + 32400*uk_147 + 14976*uk_148 + 446988*uk_149 + 11411775*uk_15 + 521100*uk_150 + 240864*uk_151 + 607500*uk_152 + 280800*uk_153 + 129792*uk_154 + 7189057*uk_155 + 8381025*uk_156 + 3873896*uk_157 + 9770625*uk_158 + 4516200*uk_159 + 5274776*uk_16 + 2087488*uk_160 + 11390625*uk_161 + 5265000*uk_162 + 2433600*uk_163 + 1124864*uk_164 + 3025*uk_17 + 8415*uk_18 + 5720*uk_19 + 55*uk_2 + 660*uk_20 + 10615*uk_21 + 12375*uk_22 + 5720*uk_23 + 23409*uk_24 + 15912*uk_25 + 1836*uk_26 + 29529*uk_27 + 34425*uk_28 + 15912*uk_29 + 153*uk_3 + 10816*uk_30 + 1248*uk_31 + 20072*uk_32 + 23400*uk_33 + 10816*uk_34 + 144*uk_35 + 2316*uk_36 + 2700*uk_37 + 1248*uk_38 + 37249*uk_39 + 104*uk_4 + 43425*uk_40 + 20072*uk_41 + 50625*uk_42 + 23400*uk_43 + 10816*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 393579795033*uk_47 + 267531363944*uk_48 + 30869003532*uk_49 + 12*uk_5 + 496476473473*uk_50 + 578793816225*uk_51 + 267531363944*uk_52 + 153424975*uk_53 + 426800385*uk_54 + 290112680*uk_55 + 33474540*uk_56 + 538382185*uk_57 + 627647625*uk_58 + 290112680*uk_59 + 193*uk_6 + 1187281071*uk_60 + 807040728*uk_61 + 93120084*uk_62 + 1497681351*uk_63 + 1746001575*uk_64 + 807040728*uk_65 + 548576704*uk_66 + 63297312*uk_67 + 1018031768*uk_68 + 1186824600*uk_69 + 225*uk_7 + 548576704*uk_70 + 7303536*uk_71 + 117465204*uk_72 + 136941300*uk_73 + 63297312*uk_74 + 1889232031*uk_75 + 2202472575*uk_76 + 1018031768*uk_77 + 2567649375*uk_78 + 1186824600*uk_79 + 104*uk_8 + 548576704*uk_80 + 166375*uk_81 + 462825*uk_82 + 314600*uk_83 + 36300*uk_84 + 583825*uk_85 + 680625*uk_86 + 314600*uk_87 + 1287495*uk_88 + 875160*uk_89 + 2572416961*uk_9 + 100980*uk_90 + 1624095*uk_91 + 1893375*uk_92 + 875160*uk_93 + 594880*uk_94 + 68640*uk_95 + 1103960*uk_96 + 1287000*uk_97 + 594880*uk_98 + 7920*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 85800*uk_100 + 99000*uk_101 + 67320*uk_102 + 2091375*uk_103 + 2413125*uk_104 + 1640925*uk_105 + 2784375*uk_106 + 1893375*uk_107 + 1287495*uk_108 + 6859*uk_109 + 963661*uk_11 + 55233*uk_110 + 2888*uk_111 + 70395*uk_112 + 81225*uk_113 + 55233*uk_114 + 444771*uk_115 + 23256*uk_116 + 566865*uk_117 + 654075*uk_118 + 444771*uk_119 + 7760007*uk_12 + 1216*uk_120 + 29640*uk_121 + 34200*uk_122 + 23256*uk_123 + 722475*uk_124 + 833625*uk_125 + 566865*uk_126 + 961875*uk_127 + 654075*uk_128 + 444771*uk_129 + 405752*uk_13 + 3581577*uk_130 + 187272*uk_131 + 4564755*uk_132 + 5267025*uk_133 + 3581577*uk_134 + 9792*uk_135 + 238680*uk_136 + 275400*uk_137 + 187272*uk_138 + 5817825*uk_139 + 9890205*uk_14 + 6712875*uk_140 + 4564755*uk_141 + 7745625*uk_142 + 5267025*uk_143 + 3581577*uk_144 + 512*uk_145 + 12480*uk_146 + 14400*uk_147 + 9792*uk_148 + 304200*uk_149 + 11411775*uk_15 + 351000*uk_150 + 238680*uk_151 + 405000*uk_152 + 275400*uk_153 + 187272*uk_154 + 7414875*uk_155 + 8555625*uk_156 + 5817825*uk_157 + 9871875*uk_158 + 6712875*uk_159 + 7760007*uk_16 + 4564755*uk_160 + 11390625*uk_161 + 7745625*uk_162 + 5267025*uk_163 + 3581577*uk_164 + 3025*uk_17 + 1045*uk_18 + 8415*uk_19 + 55*uk_2 + 440*uk_20 + 10725*uk_21 + 12375*uk_22 + 8415*uk_23 + 361*uk_24 + 2907*uk_25 + 152*uk_26 + 3705*uk_27 + 4275*uk_28 + 2907*uk_29 + 19*uk_3 + 23409*uk_30 + 1224*uk_31 + 29835*uk_32 + 34425*uk_33 + 23409*uk_34 + 64*uk_35 + 1560*uk_36 + 1800*uk_37 + 1224*uk_38 + 38025*uk_39 + 153*uk_4 + 43875*uk_40 + 29835*uk_41 + 50625*uk_42 + 34425*uk_43 + 23409*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 48875922259*uk_47 + 393579795033*uk_48 + 20579335688*uk_49 + 8*uk_5 + 501621307395*uk_50 + 578793816225*uk_51 + 393579795033*uk_52 + 153424975*uk_53 + 53001355*uk_54 + 426800385*uk_55 + 22316360*uk_56 + 543961275*uk_57 + 627647625*uk_58 + 426800385*uk_59 + 195*uk_6 + 18309559*uk_60 + 147440133*uk_61 + 7709288*uk_62 + 187913895*uk_63 + 216823725*uk_64 + 147440133*uk_65 + 1187281071*uk_66 + 62080056*uk_67 + 1513201365*uk_68 + 1746001575*uk_69 + 225*uk_7 + 1187281071*uk_70 + 3246016*uk_71 + 79121640*uk_72 + 91294200*uk_73 + 62080056*uk_74 + 1928589975*uk_75 + 2225296125*uk_76 + 1513201365*uk_77 + 2567649375*uk_78 + 1746001575*uk_79 + 153*uk_8 + 1187281071*uk_80 + 166375*uk_81 + 57475*uk_82 + 462825*uk_83 + 24200*uk_84 + 589875*uk_85 + 680625*uk_86 + 462825*uk_87 + 19855*uk_88 + 159885*uk_89 + 2572416961*uk_9 + 8360*uk_90 + 203775*uk_91 + 235125*uk_92 + 159885*uk_93 + 1287495*uk_94 + 67320*uk_95 + 1640925*uk_96 + 1893375*uk_97 + 1287495*uk_98 + 3520*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 130020*uk_100 + 148500*uk_101 + 12540*uk_102 + 2134495*uk_103 + 2437875*uk_104 + 205865*uk_105 + 2784375*uk_106 + 235125*uk_107 + 19855*uk_108 + 729000*uk_109 + 4564710*uk_11 + 153900*uk_110 + 97200*uk_111 + 1595700*uk_112 + 1822500*uk_113 + 153900*uk_114 + 32490*uk_115 + 20520*uk_116 + 336870*uk_117 + 384750*uk_118 + 32490*uk_119 + 963661*uk_12 + 12960*uk_120 + 212760*uk_121 + 243000*uk_122 + 20520*uk_123 + 3492810*uk_124 + 3989250*uk_125 + 336870*uk_126 + 4556250*uk_127 + 384750*uk_128 + 32490*uk_129 + 608628*uk_13 + 6859*uk_130 + 4332*uk_131 + 71117*uk_132 + 81225*uk_133 + 6859*uk_134 + 2736*uk_135 + 44916*uk_136 + 51300*uk_137 + 4332*uk_138 + 737371*uk_139 + 9991643*uk_14 + 842175*uk_140 + 71117*uk_141 + 961875*uk_142 + 81225*uk_143 + 6859*uk_144 + 1728*uk_145 + 28368*uk_146 + 32400*uk_147 + 2736*uk_148 + 465708*uk_149 + 11411775*uk_15 + 531900*uk_150 + 44916*uk_151 + 607500*uk_152 + 51300*uk_153 + 4332*uk_154 + 7645373*uk_155 + 8732025*uk_156 + 737371*uk_157 + 9973125*uk_158 + 842175*uk_159 + 963661*uk_16 + 71117*uk_160 + 11390625*uk_161 + 961875*uk_162 + 81225*uk_163 + 6859*uk_164 + 3025*uk_17 + 4950*uk_18 + 1045*uk_19 + 55*uk_2 + 660*uk_20 + 10835*uk_21 + 12375*uk_22 + 1045*uk_23 + 8100*uk_24 + 1710*uk_25 + 1080*uk_26 + 17730*uk_27 + 20250*uk_28 + 1710*uk_29 + 90*uk_3 + 361*uk_30 + 228*uk_31 + 3743*uk_32 + 4275*uk_33 + 361*uk_34 + 144*uk_35 + 2364*uk_36 + 2700*uk_37 + 228*uk_38 + 38809*uk_39 + 19*uk_4 + 44325*uk_40 + 3743*uk_41 + 50625*uk_42 + 4275*uk_43 + 361*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 231517526490*uk_47 + 48875922259*uk_48 + 30869003532*uk_49 + 12*uk_5 + 506766141317*uk_50 + 578793816225*uk_51 + 48875922259*uk_52 + 153424975*uk_53 + 251059050*uk_54 + 53001355*uk_55 + 33474540*uk_56 + 549540365*uk_57 + 627647625*uk_58 + 53001355*uk_59 + 197*uk_6 + 410823900*uk_60 + 86729490*uk_61 + 54776520*uk_62 + 899247870*uk_63 + 1027059750*uk_64 + 86729490*uk_65 + 18309559*uk_66 + 11563932*uk_67 + 189841217*uk_68 + 216823725*uk_69 + 225*uk_7 + 18309559*uk_70 + 7303536*uk_71 + 119899716*uk_72 + 136941300*uk_73 + 11563932*uk_74 + 1968353671*uk_75 + 2248119675*uk_76 + 189841217*uk_77 + 2567649375*uk_78 + 216823725*uk_79 + 19*uk_8 + 18309559*uk_80 + 166375*uk_81 + 272250*uk_82 + 57475*uk_83 + 36300*uk_84 + 595925*uk_85 + 680625*uk_86 + 57475*uk_87 + 445500*uk_88 + 94050*uk_89 + 2572416961*uk_9 + 59400*uk_90 + 975150*uk_91 + 1113750*uk_92 + 94050*uk_93 + 19855*uk_94 + 12540*uk_95 + 205865*uk_96 + 235125*uk_97 + 19855*uk_98 + 7920*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 131340*uk_100 + 148500*uk_101 + 59400*uk_102 + 2178055*uk_103 + 2462625*uk_104 + 985050*uk_105 + 2784375*uk_106 + 1113750*uk_107 + 445500*uk_108 + 5177717*uk_109 + 8774387*uk_11 + 2693610*uk_110 + 359148*uk_111 + 5955871*uk_112 + 6734025*uk_113 + 2693610*uk_114 + 1401300*uk_115 + 186840*uk_116 + 3098430*uk_117 + 3503250*uk_118 + 1401300*uk_119 + 4564710*uk_12 + 24912*uk_120 + 413124*uk_121 + 467100*uk_122 + 186840*uk_123 + 6850973*uk_124 + 7746075*uk_125 + 3098430*uk_126 + 8758125*uk_127 + 3503250*uk_128 + 1401300*uk_129 + 608628*uk_13 + 729000*uk_130 + 97200*uk_131 + 1611900*uk_132 + 1822500*uk_133 + 729000*uk_134 + 12960*uk_135 + 214920*uk_136 + 243000*uk_137 + 97200*uk_138 + 3564090*uk_139 + 10093081*uk_14 + 4029750*uk_140 + 1611900*uk_141 + 4556250*uk_142 + 1822500*uk_143 + 729000*uk_144 + 1728*uk_145 + 28656*uk_146 + 32400*uk_147 + 12960*uk_148 + 475212*uk_149 + 11411775*uk_15 + 537300*uk_150 + 214920*uk_151 + 607500*uk_152 + 243000*uk_153 + 97200*uk_154 + 7880599*uk_155 + 8910225*uk_156 + 3564090*uk_157 + 10074375*uk_158 + 4029750*uk_159 + 4564710*uk_16 + 1611900*uk_160 + 11390625*uk_161 + 4556250*uk_162 + 1822500*uk_163 + 729000*uk_164 + 3025*uk_17 + 9515*uk_18 + 4950*uk_19 + 55*uk_2 + 660*uk_20 + 10945*uk_21 + 12375*uk_22 + 4950*uk_23 + 29929*uk_24 + 15570*uk_25 + 2076*uk_26 + 34427*uk_27 + 38925*uk_28 + 15570*uk_29 + 173*uk_3 + 8100*uk_30 + 1080*uk_31 + 17910*uk_32 + 20250*uk_33 + 8100*uk_34 + 144*uk_35 + 2388*uk_36 + 2700*uk_37 + 1080*uk_38 + 39601*uk_39 + 90*uk_4 + 44775*uk_40 + 17910*uk_41 + 50625*uk_42 + 20250*uk_43 + 8100*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 445028134253*uk_47 + 231517526490*uk_48 + 30869003532*uk_49 + 12*uk_5 + 511910975239*uk_50 + 578793816225*uk_51 + 231517526490*uk_52 + 153424975*uk_53 + 482591285*uk_54 + 251059050*uk_55 + 33474540*uk_56 + 555119455*uk_57 + 627647625*uk_58 + 251059050*uk_59 + 199*uk_6 + 1517968951*uk_60 + 789694830*uk_61 + 105292644*uk_62 + 1746103013*uk_63 + 1974237075*uk_64 + 789694830*uk_65 + 410823900*uk_66 + 54776520*uk_67 + 908377290*uk_68 + 1027059750*uk_69 + 225*uk_7 + 410823900*uk_70 + 7303536*uk_71 + 121116972*uk_72 + 136941300*uk_73 + 54776520*uk_74 + 2008523119*uk_75 + 2270943225*uk_76 + 908377290*uk_77 + 2567649375*uk_78 + 1027059750*uk_79 + 90*uk_8 + 410823900*uk_80 + 166375*uk_81 + 523325*uk_82 + 272250*uk_83 + 36300*uk_84 + 601975*uk_85 + 680625*uk_86 + 272250*uk_87 + 1646095*uk_88 + 856350*uk_89 + 2572416961*uk_9 + 114180*uk_90 + 1893485*uk_91 + 2140875*uk_92 + 856350*uk_93 + 445500*uk_94 + 59400*uk_95 + 985050*uk_96 + 1113750*uk_97 + 445500*uk_98 + 7920*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 88440*uk_100 + 99000*uk_101 + 76120*uk_102 + 2222055*uk_103 + 2487375*uk_104 + 1912515*uk_105 + 2784375*uk_106 + 2140875*uk_107 + 1646095*uk_108 + 300763*uk_109 + 3398173*uk_11 + 776597*uk_110 + 35912*uk_111 + 902289*uk_112 + 1010025*uk_113 + 776597*uk_114 + 2005243*uk_115 + 92728*uk_116 + 2329791*uk_117 + 2607975*uk_118 + 2005243*uk_119 + 8774387*uk_12 + 4288*uk_120 + 107736*uk_121 + 120600*uk_122 + 92728*uk_123 + 2706867*uk_124 + 3030075*uk_125 + 2329791*uk_126 + 3391875*uk_127 + 2607975*uk_128 + 2005243*uk_129 + 405752*uk_13 + 5177717*uk_130 + 239432*uk_131 + 6015729*uk_132 + 6734025*uk_133 + 5177717*uk_134 + 11072*uk_135 + 278184*uk_136 + 311400*uk_137 + 239432*uk_138 + 6989373*uk_139 + 10194519*uk_14 + 7823925*uk_140 + 6015729*uk_141 + 8758125*uk_142 + 6734025*uk_143 + 5177717*uk_144 + 512*uk_145 + 12864*uk_146 + 14400*uk_147 + 11072*uk_148 + 323208*uk_149 + 11411775*uk_15 + 361800*uk_150 + 278184*uk_151 + 405000*uk_152 + 311400*uk_153 + 239432*uk_154 + 8120601*uk_155 + 9090225*uk_156 + 6989373*uk_157 + 10175625*uk_158 + 7823925*uk_159 + 8774387*uk_16 + 6015729*uk_160 + 11390625*uk_161 + 8758125*uk_162 + 6734025*uk_163 + 5177717*uk_164 + 3025*uk_17 + 3685*uk_18 + 9515*uk_19 + 55*uk_2 + 440*uk_20 + 11055*uk_21 + 12375*uk_22 + 9515*uk_23 + 4489*uk_24 + 11591*uk_25 + 536*uk_26 + 13467*uk_27 + 15075*uk_28 + 11591*uk_29 + 67*uk_3 + 29929*uk_30 + 1384*uk_31 + 34773*uk_32 + 38925*uk_33 + 29929*uk_34 + 64*uk_35 + 1608*uk_36 + 1800*uk_37 + 1384*uk_38 + 40401*uk_39 + 173*uk_4 + 45225*uk_40 + 34773*uk_41 + 50625*uk_42 + 38925*uk_43 + 29929*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 172351936387*uk_47 + 445028134253*uk_48 + 20579335688*uk_49 + 8*uk_5 + 517055809161*uk_50 + 578793816225*uk_51 + 445028134253*uk_52 + 153424975*uk_53 + 186899515*uk_54 + 482591285*uk_55 + 22316360*uk_56 + 560698545*uk_57 + 627647625*uk_58 + 482591285*uk_59 + 201*uk_6 + 227677591*uk_60 + 587883929*uk_61 + 27185384*uk_62 + 683032773*uk_63 + 764588925*uk_64 + 587883929*uk_65 + 1517968951*uk_66 + 70195096*uk_67 + 1763651787*uk_68 + 1974237075*uk_69 + 225*uk_7 + 1517968951*uk_70 + 3246016*uk_71 + 81556152*uk_72 + 91294200*uk_73 + 70195096*uk_74 + 2049098319*uk_75 + 2293766775*uk_76 + 1763651787*uk_77 + 2567649375*uk_78 + 1974237075*uk_79 + 173*uk_8 + 1517968951*uk_80 + 166375*uk_81 + 202675*uk_82 + 523325*uk_83 + 24200*uk_84 + 608025*uk_85 + 680625*uk_86 + 523325*uk_87 + 246895*uk_88 + 637505*uk_89 + 2572416961*uk_9 + 29480*uk_90 + 740685*uk_91 + 829125*uk_92 + 637505*uk_93 + 1646095*uk_94 + 76120*uk_95 + 1912515*uk_96 + 2140875*uk_97 + 1646095*uk_98 + 3520*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 133980*uk_100 + 148500*uk_101 + 44220*uk_102 + 2266495*uk_103 + 2512125*uk_104 + 748055*uk_105 + 2784375*uk_106 + 829125*uk_107 + 246895*uk_108 + 5088448*uk_109 + 8723668*uk_11 + 1982128*uk_110 + 355008*uk_111 + 6005552*uk_112 + 6656400*uk_113 + 1982128*uk_114 + 772108*uk_115 + 138288*uk_116 + 2339372*uk_117 + 2592900*uk_118 + 772108*uk_119 + 3398173*uk_12 + 24768*uk_120 + 418992*uk_121 + 464400*uk_122 + 138288*uk_123 + 7087948*uk_124 + 7856100*uk_125 + 2339372*uk_126 + 8707500*uk_127 + 2592900*uk_128 + 772108*uk_129 + 608628*uk_13 + 300763*uk_130 + 53868*uk_131 + 911267*uk_132 + 1010025*uk_133 + 300763*uk_134 + 9648*uk_135 + 163212*uk_136 + 180900*uk_137 + 53868*uk_138 + 2761003*uk_139 + 10295957*uk_14 + 3060225*uk_140 + 911267*uk_141 + 3391875*uk_142 + 1010025*uk_143 + 300763*uk_144 + 1728*uk_145 + 29232*uk_146 + 32400*uk_147 + 9648*uk_148 + 494508*uk_149 + 11411775*uk_15 + 548100*uk_150 + 163212*uk_151 + 607500*uk_152 + 180900*uk_153 + 53868*uk_154 + 8365427*uk_155 + 9272025*uk_156 + 2761003*uk_157 + 10276875*uk_158 + 3060225*uk_159 + 3398173*uk_16 + 911267*uk_160 + 11390625*uk_161 + 3391875*uk_162 + 1010025*uk_163 + 300763*uk_164 + 3025*uk_17 + 9460*uk_18 + 3685*uk_19 + 55*uk_2 + 660*uk_20 + 11165*uk_21 + 12375*uk_22 + 3685*uk_23 + 29584*uk_24 + 11524*uk_25 + 2064*uk_26 + 34916*uk_27 + 38700*uk_28 + 11524*uk_29 + 172*uk_3 + 4489*uk_30 + 804*uk_31 + 13601*uk_32 + 15075*uk_33 + 4489*uk_34 + 144*uk_35 + 2436*uk_36 + 2700*uk_37 + 804*uk_38 + 41209*uk_39 + 67*uk_4 + 45675*uk_40 + 13601*uk_41 + 50625*uk_42 + 15075*uk_43 + 4489*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 442455717292*uk_47 + 172351936387*uk_48 + 30869003532*uk_49 + 12*uk_5 + 522200643083*uk_50 + 578793816225*uk_51 + 172351936387*uk_52 + 153424975*uk_53 + 479801740*uk_54 + 186899515*uk_55 + 33474540*uk_56 + 566277635*uk_57 + 627647625*uk_58 + 186899515*uk_59 + 203*uk_6 + 1500470896*uk_60 + 584485756*uk_61 + 104684016*uk_62 + 1770904604*uk_63 + 1962825300*uk_64 + 584485756*uk_65 + 227677591*uk_66 + 40778076*uk_67 + 689829119*uk_68 + 764588925*uk_69 + 225*uk_7 + 227677591*uk_70 + 7303536*uk_71 + 123551484*uk_72 + 136941300*uk_73 + 40778076*uk_74 + 2090079271*uk_75 + 2316590325*uk_76 + 689829119*uk_77 + 2567649375*uk_78 + 764588925*uk_79 + 67*uk_8 + 227677591*uk_80 + 166375*uk_81 + 520300*uk_82 + 202675*uk_83 + 36300*uk_84 + 614075*uk_85 + 680625*uk_86 + 202675*uk_87 + 1627120*uk_88 + 633820*uk_89 + 2572416961*uk_9 + 113520*uk_90 + 1920380*uk_91 + 2128500*uk_92 + 633820*uk_93 + 246895*uk_94 + 44220*uk_95 + 748055*uk_96 + 829125*uk_97 + 246895*uk_98 + 7920*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 90200*uk_100 + 99000*uk_101 + 75680*uk_102 + 2311375*uk_103 + 2536875*uk_104 + 1939300*uk_105 + 2784375*uk_106 + 2128500*uk_107 + 1627120*uk_108 + 592704*uk_109 + 4260396*uk_11 + 1213632*uk_110 + 56448*uk_111 + 1446480*uk_112 + 1587600*uk_113 + 1213632*uk_114 + 2485056*uk_115 + 115584*uk_116 + 2961840*uk_117 + 3250800*uk_118 + 2485056*uk_119 + 8723668*uk_12 + 5376*uk_120 + 137760*uk_121 + 151200*uk_122 + 115584*uk_123 + 3530100*uk_124 + 3874500*uk_125 + 2961840*uk_126 + 4252500*uk_127 + 3250800*uk_128 + 2485056*uk_129 + 405752*uk_13 + 5088448*uk_130 + 236672*uk_131 + 6064720*uk_132 + 6656400*uk_133 + 5088448*uk_134 + 11008*uk_135 + 282080*uk_136 + 309600*uk_137 + 236672*uk_138 + 7228300*uk_139 + 10397395*uk_14 + 7933500*uk_140 + 6064720*uk_141 + 8707500*uk_142 + 6656400*uk_143 + 5088448*uk_144 + 512*uk_145 + 13120*uk_146 + 14400*uk_147 + 11008*uk_148 + 336200*uk_149 + 11411775*uk_15 + 369000*uk_150 + 282080*uk_151 + 405000*uk_152 + 309600*uk_153 + 236672*uk_154 + 8615125*uk_155 + 9455625*uk_156 + 7228300*uk_157 + 10378125*uk_158 + 7933500*uk_159 + 8723668*uk_16 + 6064720*uk_160 + 11390625*uk_161 + 8707500*uk_162 + 6656400*uk_163 + 5088448*uk_164 + 3025*uk_17 + 4620*uk_18 + 9460*uk_19 + 55*uk_2 + 440*uk_20 + 11275*uk_21 + 12375*uk_22 + 9460*uk_23 + 7056*uk_24 + 14448*uk_25 + 672*uk_26 + 17220*uk_27 + 18900*uk_28 + 14448*uk_29 + 84*uk_3 + 29584*uk_30 + 1376*uk_31 + 35260*uk_32 + 38700*uk_33 + 29584*uk_34 + 64*uk_35 + 1640*uk_36 + 1800*uk_37 + 1376*uk_38 + 42025*uk_39 + 172*uk_4 + 46125*uk_40 + 35260*uk_41 + 50625*uk_42 + 38700*uk_43 + 29584*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 216083024724*uk_47 + 442455717292*uk_48 + 20579335688*uk_49 + 8*uk_5 + 527345477005*uk_50 + 578793816225*uk_51 + 442455717292*uk_52 + 153424975*uk_53 + 234321780*uk_54 + 479801740*uk_55 + 22316360*uk_56 + 571856725*uk_57 + 627647625*uk_58 + 479801740*uk_59 + 205*uk_6 + 357873264*uk_60 + 732788112*uk_61 + 34083168*uk_62 + 873381180*uk_63 + 958589100*uk_64 + 732788112*uk_65 + 1500470896*uk_66 + 69789344*uk_67 + 1788351940*uk_68 + 1962825300*uk_69 + 225*uk_7 + 1500470896*uk_70 + 3246016*uk_71 + 83179160*uk_72 + 91294200*uk_73 + 69789344*uk_74 + 2131465975*uk_75 + 2339413875*uk_76 + 1788351940*uk_77 + 2567649375*uk_78 + 1962825300*uk_79 + 172*uk_8 + 1500470896*uk_80 + 166375*uk_81 + 254100*uk_82 + 520300*uk_83 + 24200*uk_84 + 620125*uk_85 + 680625*uk_86 + 520300*uk_87 + 388080*uk_88 + 794640*uk_89 + 2572416961*uk_9 + 36960*uk_90 + 947100*uk_91 + 1039500*uk_92 + 794640*uk_93 + 1627120*uk_94 + 75680*uk_95 + 1939300*uk_96 + 2128500*uk_97 + 1627120*uk_98 + 3520*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 91080*uk_100 + 99000*uk_101 + 36960*uk_102 + 2356695*uk_103 + 2561625*uk_104 + 956340*uk_105 + 2784375*uk_106 + 1039500*uk_107 + 388080*uk_108 + 64*uk_109 + 202876*uk_11 + 1344*uk_110 + 128*uk_111 + 3312*uk_112 + 3600*uk_113 + 1344*uk_114 + 28224*uk_115 + 2688*uk_116 + 69552*uk_117 + 75600*uk_118 + 28224*uk_119 + 4260396*uk_12 + 256*uk_120 + 6624*uk_121 + 7200*uk_122 + 2688*uk_123 + 171396*uk_124 + 186300*uk_125 + 69552*uk_126 + 202500*uk_127 + 75600*uk_128 + 28224*uk_129 + 405752*uk_13 + 592704*uk_130 + 56448*uk_131 + 1460592*uk_132 + 1587600*uk_133 + 592704*uk_134 + 5376*uk_135 + 139104*uk_136 + 151200*uk_137 + 56448*uk_138 + 3599316*uk_139 + 10498833*uk_14 + 3912300*uk_140 + 1460592*uk_141 + 4252500*uk_142 + 1587600*uk_143 + 592704*uk_144 + 512*uk_145 + 13248*uk_146 + 14400*uk_147 + 5376*uk_148 + 342792*uk_149 + 11411775*uk_15 + 372600*uk_150 + 139104*uk_151 + 405000*uk_152 + 151200*uk_153 + 56448*uk_154 + 8869743*uk_155 + 9641025*uk_156 + 3599316*uk_157 + 10479375*uk_158 + 3912300*uk_159 + 4260396*uk_16 + 1460592*uk_160 + 11390625*uk_161 + 4252500*uk_162 + 1587600*uk_163 + 592704*uk_164 + 3025*uk_17 + 220*uk_18 + 4620*uk_19 + 55*uk_2 + 440*uk_20 + 11385*uk_21 + 12375*uk_22 + 4620*uk_23 + 16*uk_24 + 336*uk_25 + 32*uk_26 + 828*uk_27 + 900*uk_28 + 336*uk_29 + 4*uk_3 + 7056*uk_30 + 672*uk_31 + 17388*uk_32 + 18900*uk_33 + 7056*uk_34 + 64*uk_35 + 1656*uk_36 + 1800*uk_37 + 672*uk_38 + 42849*uk_39 + 84*uk_4 + 46575*uk_40 + 17388*uk_41 + 50625*uk_42 + 18900*uk_43 + 7056*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 10289667844*uk_47 + 216083024724*uk_48 + 20579335688*uk_49 + 8*uk_5 + 532490310927*uk_50 + 578793816225*uk_51 + 216083024724*uk_52 + 153424975*uk_53 + 11158180*uk_54 + 234321780*uk_55 + 22316360*uk_56 + 577435815*uk_57 + 627647625*uk_58 + 234321780*uk_59 + 207*uk_6 + 811504*uk_60 + 17041584*uk_61 + 1623008*uk_62 + 41995332*uk_63 + 45647100*uk_64 + 17041584*uk_65 + 357873264*uk_66 + 34083168*uk_67 + 881901972*uk_68 + 958589100*uk_69 + 225*uk_7 + 357873264*uk_70 + 3246016*uk_71 + 83990664*uk_72 + 91294200*uk_73 + 34083168*uk_74 + 2173258431*uk_75 + 2362237425*uk_76 + 881901972*uk_77 + 2567649375*uk_78 + 958589100*uk_79 + 84*uk_8 + 357873264*uk_80 + 166375*uk_81 + 12100*uk_82 + 254100*uk_83 + 24200*uk_84 + 626175*uk_85 + 680625*uk_86 + 254100*uk_87 + 880*uk_88 + 18480*uk_89 + 2572416961*uk_9 + 1760*uk_90 + 45540*uk_91 + 49500*uk_92 + 18480*uk_93 + 388080*uk_94 + 36960*uk_95 + 956340*uk_96 + 1039500*uk_97 + 388080*uk_98 + 3520*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 137940*uk_100 + 148500*uk_101 + 2640*uk_102 + 2402455*uk_103 + 2586375*uk_104 + 45980*uk_105 + 2784375*uk_106 + 49500*uk_107 + 880*uk_108 + 2803221*uk_109 + 7151379*uk_11 + 79524*uk_110 + 238572*uk_111 + 4155129*uk_112 + 4473225*uk_113 + 79524*uk_114 + 2256*uk_115 + 6768*uk_116 + 117876*uk_117 + 126900*uk_118 + 2256*uk_119 + 202876*uk_12 + 20304*uk_120 + 353628*uk_121 + 380700*uk_122 + 6768*uk_123 + 6159021*uk_124 + 6630525*uk_125 + 117876*uk_126 + 7138125*uk_127 + 126900*uk_128 + 2256*uk_129 + 608628*uk_13 + 64*uk_130 + 192*uk_131 + 3344*uk_132 + 3600*uk_133 + 64*uk_134 + 576*uk_135 + 10032*uk_136 + 10800*uk_137 + 192*uk_138 + 174724*uk_139 + 10600271*uk_14 + 188100*uk_140 + 3344*uk_141 + 202500*uk_142 + 3600*uk_143 + 64*uk_144 + 1728*uk_145 + 30096*uk_146 + 32400*uk_147 + 576*uk_148 + 524172*uk_149 + 11411775*uk_15 + 564300*uk_150 + 10032*uk_151 + 607500*uk_152 + 10800*uk_153 + 192*uk_154 + 9129329*uk_155 + 9828225*uk_156 + 174724*uk_157 + 10580625*uk_158 + 188100*uk_159 + 202876*uk_16 + 3344*uk_160 + 11390625*uk_161 + 202500*uk_162 + 3600*uk_163 + 64*uk_164 + 3025*uk_17 + 7755*uk_18 + 220*uk_19 + 55*uk_2 + 660*uk_20 + 11495*uk_21 + 12375*uk_22 + 220*uk_23 + 19881*uk_24 + 564*uk_25 + 1692*uk_26 + 29469*uk_27 + 31725*uk_28 + 564*uk_29 + 141*uk_3 + 16*uk_30 + 48*uk_31 + 836*uk_32 + 900*uk_33 + 16*uk_34 + 144*uk_35 + 2508*uk_36 + 2700*uk_37 + 48*uk_38 + 43681*uk_39 + 4*uk_4 + 47025*uk_40 + 836*uk_41 + 50625*uk_42 + 900*uk_43 + 16*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 362710791501*uk_47 + 10289667844*uk_48 + 30869003532*uk_49 + 12*uk_5 + 537635144849*uk_50 + 578793816225*uk_51 + 10289667844*uk_52 + 153424975*uk_53 + 393325845*uk_54 + 11158180*uk_55 + 33474540*uk_56 + 583014905*uk_57 + 627647625*uk_58 + 11158180*uk_59 + 209*uk_6 + 1008344439*uk_60 + 28605516*uk_61 + 85816548*uk_62 + 1494638211*uk_63 + 1609060275*uk_64 + 28605516*uk_65 + 811504*uk_66 + 2434512*uk_67 + 42401084*uk_68 + 45647100*uk_69 + 225*uk_7 + 811504*uk_70 + 7303536*uk_71 + 127203252*uk_72 + 136941300*uk_73 + 2434512*uk_74 + 2215456639*uk_75 + 2385060975*uk_76 + 42401084*uk_77 + 2567649375*uk_78 + 45647100*uk_79 + 4*uk_8 + 811504*uk_80 + 166375*uk_81 + 426525*uk_82 + 12100*uk_83 + 36300*uk_84 + 632225*uk_85 + 680625*uk_86 + 12100*uk_87 + 1093455*uk_88 + 31020*uk_89 + 2572416961*uk_9 + 93060*uk_90 + 1620795*uk_91 + 1744875*uk_92 + 31020*uk_93 + 880*uk_94 + 2640*uk_95 + 45980*uk_96 + 49500*uk_97 + 880*uk_98 + 7920*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 92840*uk_100 + 99000*uk_101 + 62040*uk_102 + 2448655*uk_103 + 2611125*uk_104 + 1636305*uk_105 + 2784375*uk_106 + 1744875*uk_107 + 1093455*uk_108 + 493039*uk_109 + 4006801*uk_11 + 879981*uk_110 + 49928*uk_111 + 1316851*uk_112 + 1404225*uk_113 + 879981*uk_114 + 1570599*uk_115 + 89112*uk_116 + 2350329*uk_117 + 2506275*uk_118 + 1570599*uk_119 + 7151379*uk_12 + 5056*uk_120 + 133352*uk_121 + 142200*uk_122 + 89112*uk_123 + 3517159*uk_124 + 3750525*uk_125 + 2350329*uk_126 + 3999375*uk_127 + 2506275*uk_128 + 1570599*uk_129 + 405752*uk_13 + 2803221*uk_130 + 159048*uk_131 + 4194891*uk_132 + 4473225*uk_133 + 2803221*uk_134 + 9024*uk_135 + 238008*uk_136 + 253800*uk_137 + 159048*uk_138 + 6277461*uk_139 + 10701709*uk_14 + 6693975*uk_140 + 4194891*uk_141 + 7138125*uk_142 + 4473225*uk_143 + 2803221*uk_144 + 512*uk_145 + 13504*uk_146 + 14400*uk_147 + 9024*uk_148 + 356168*uk_149 + 11411775*uk_15 + 379800*uk_150 + 238008*uk_151 + 405000*uk_152 + 253800*uk_153 + 159048*uk_154 + 9393931*uk_155 + 10017225*uk_156 + 6277461*uk_157 + 10681875*uk_158 + 6693975*uk_159 + 7151379*uk_16 + 4194891*uk_160 + 11390625*uk_161 + 7138125*uk_162 + 4473225*uk_163 + 2803221*uk_164 + 3025*uk_17 + 4345*uk_18 + 7755*uk_19 + 55*uk_2 + 440*uk_20 + 11605*uk_21 + 12375*uk_22 + 7755*uk_23 + 6241*uk_24 + 11139*uk_25 + 632*uk_26 + 16669*uk_27 + 17775*uk_28 + 11139*uk_29 + 79*uk_3 + 19881*uk_30 + 1128*uk_31 + 29751*uk_32 + 31725*uk_33 + 19881*uk_34 + 64*uk_35 + 1688*uk_36 + 1800*uk_37 + 1128*uk_38 + 44521*uk_39 + 141*uk_4 + 47475*uk_40 + 29751*uk_41 + 50625*uk_42 + 31725*uk_43 + 19881*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 203220939919*uk_47 + 362710791501*uk_48 + 20579335688*uk_49 + 8*uk_5 + 542779978771*uk_50 + 578793816225*uk_51 + 362710791501*uk_52 + 153424975*uk_53 + 220374055*uk_54 + 393325845*uk_55 + 22316360*uk_56 + 588593995*uk_57 + 627647625*uk_58 + 393325845*uk_59 + 211*uk_6 + 316537279*uk_60 + 564958941*uk_61 + 32054408*uk_62 + 845435011*uk_63 + 901530225*uk_64 + 564958941*uk_65 + 1008344439*uk_66 + 57211032*uk_67 + 1508940969*uk_68 + 1609060275*uk_69 + 225*uk_7 + 1008344439*uk_70 + 3246016*uk_71 + 85613672*uk_72 + 91294200*uk_73 + 57211032*uk_74 + 2258060599*uk_75 + 2407884525*uk_76 + 1508940969*uk_77 + 2567649375*uk_78 + 1609060275*uk_79 + 141*uk_8 + 1008344439*uk_80 + 166375*uk_81 + 238975*uk_82 + 426525*uk_83 + 24200*uk_84 + 638275*uk_85 + 680625*uk_86 + 426525*uk_87 + 343255*uk_88 + 612645*uk_89 + 2572416961*uk_9 + 34760*uk_90 + 916795*uk_91 + 977625*uk_92 + 612645*uk_93 + 1093455*uk_94 + 62040*uk_95 + 1636305*uk_96 + 1744875*uk_97 + 1093455*uk_98 + 3520*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 93720*uk_100 + 99000*uk_101 + 34760*uk_102 + 2495295*uk_103 + 2635875*uk_104 + 925485*uk_105 + 2784375*uk_106 + 977625*uk_107 + 343255*uk_108 + 15625*uk_109 + 1267975*uk_11 + 49375*uk_110 + 5000*uk_111 + 133125*uk_112 + 140625*uk_113 + 49375*uk_114 + 156025*uk_115 + 15800*uk_116 + 420675*uk_117 + 444375*uk_118 + 156025*uk_119 + 4006801*uk_12 + 1600*uk_120 + 42600*uk_121 + 45000*uk_122 + 15800*uk_123 + 1134225*uk_124 + 1198125*uk_125 + 420675*uk_126 + 1265625*uk_127 + 444375*uk_128 + 156025*uk_129 + 405752*uk_13 + 493039*uk_130 + 49928*uk_131 + 1329333*uk_132 + 1404225*uk_133 + 493039*uk_134 + 5056*uk_135 + 134616*uk_136 + 142200*uk_137 + 49928*uk_138 + 3584151*uk_139 + 10803147*uk_14 + 3786075*uk_140 + 1329333*uk_141 + 3999375*uk_142 + 1404225*uk_143 + 493039*uk_144 + 512*uk_145 + 13632*uk_146 + 14400*uk_147 + 5056*uk_148 + 362952*uk_149 + 11411775*uk_15 + 383400*uk_150 + 134616*uk_151 + 405000*uk_152 + 142200*uk_153 + 49928*uk_154 + 9663597*uk_155 + 10208025*uk_156 + 3584151*uk_157 + 10783125*uk_158 + 3786075*uk_159 + 4006801*uk_16 + 1329333*uk_160 + 11390625*uk_161 + 3999375*uk_162 + 1404225*uk_163 + 493039*uk_164 + 3025*uk_17 + 1375*uk_18 + 4345*uk_19 + 55*uk_2 + 440*uk_20 + 11715*uk_21 + 12375*uk_22 + 4345*uk_23 + 625*uk_24 + 1975*uk_25 + 200*uk_26 + 5325*uk_27 + 5625*uk_28 + 1975*uk_29 + 25*uk_3 + 6241*uk_30 + 632*uk_31 + 16827*uk_32 + 17775*uk_33 + 6241*uk_34 + 64*uk_35 + 1704*uk_36 + 1800*uk_37 + 632*uk_38 + 45369*uk_39 + 79*uk_4 + 47925*uk_40 + 16827*uk_41 + 50625*uk_42 + 17775*uk_43 + 6241*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 64310424025*uk_47 + 203220939919*uk_48 + 20579335688*uk_49 + 8*uk_5 + 547924812693*uk_50 + 578793816225*uk_51 + 203220939919*uk_52 + 153424975*uk_53 + 69738625*uk_54 + 220374055*uk_55 + 22316360*uk_56 + 594173085*uk_57 + 627647625*uk_58 + 220374055*uk_59 + 213*uk_6 + 31699375*uk_60 + 100170025*uk_61 + 10143800*uk_62 + 270078675*uk_63 + 285294375*uk_64 + 100170025*uk_65 + 316537279*uk_66 + 32054408*uk_67 + 853448613*uk_68 + 901530225*uk_69 + 225*uk_7 + 316537279*uk_70 + 3246016*uk_71 + 86425176*uk_72 + 91294200*uk_73 + 32054408*uk_74 + 2301070311*uk_75 + 2430708075*uk_76 + 853448613*uk_77 + 2567649375*uk_78 + 901530225*uk_79 + 79*uk_8 + 316537279*uk_80 + 166375*uk_81 + 75625*uk_82 + 238975*uk_83 + 24200*uk_84 + 644325*uk_85 + 680625*uk_86 + 238975*uk_87 + 34375*uk_88 + 108625*uk_89 + 2572416961*uk_9 + 11000*uk_90 + 292875*uk_91 + 309375*uk_92 + 108625*uk_93 + 343255*uk_94 + 34760*uk_95 + 925485*uk_96 + 977625*uk_97 + 343255*uk_98 + 3520*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 141900*uk_100 + 148500*uk_101 + 16500*uk_102 + 2542375*uk_103 + 2660625*uk_104 + 295625*uk_105 + 2784375*uk_106 + 309375*uk_107 + 34375*uk_108 + 7301384*uk_109 + 9839486*uk_11 + 940900*uk_110 + 451632*uk_111 + 8091740*uk_112 + 8468100*uk_113 + 940900*uk_114 + 121250*uk_115 + 58200*uk_116 + 1042750*uk_117 + 1091250*uk_118 + 121250*uk_119 + 1267975*uk_12 + 27936*uk_120 + 500520*uk_121 + 523800*uk_122 + 58200*uk_123 + 8967650*uk_124 + 9384750*uk_125 + 1042750*uk_126 + 9821250*uk_127 + 1091250*uk_128 + 121250*uk_129 + 608628*uk_13 + 15625*uk_130 + 7500*uk_131 + 134375*uk_132 + 140625*uk_133 + 15625*uk_134 + 3600*uk_135 + 64500*uk_136 + 67500*uk_137 + 7500*uk_138 + 1155625*uk_139 + 10904585*uk_14 + 1209375*uk_140 + 134375*uk_141 + 1265625*uk_142 + 140625*uk_143 + 15625*uk_144 + 1728*uk_145 + 30960*uk_146 + 32400*uk_147 + 3600*uk_148 + 554700*uk_149 + 11411775*uk_15 + 580500*uk_150 + 64500*uk_151 + 607500*uk_152 + 67500*uk_153 + 7500*uk_154 + 9938375*uk_155 + 10400625*uk_156 + 1155625*uk_157 + 10884375*uk_158 + 1209375*uk_159 + 1267975*uk_16 + 134375*uk_160 + 11390625*uk_161 + 1265625*uk_162 + 140625*uk_163 + 15625*uk_164 + 3025*uk_17 + 10670*uk_18 + 1375*uk_19 + 55*uk_2 + 660*uk_20 + 11825*uk_21 + 12375*uk_22 + 1375*uk_23 + 37636*uk_24 + 4850*uk_25 + 2328*uk_26 + 41710*uk_27 + 43650*uk_28 + 4850*uk_29 + 194*uk_3 + 625*uk_30 + 300*uk_31 + 5375*uk_32 + 5625*uk_33 + 625*uk_34 + 144*uk_35 + 2580*uk_36 + 2700*uk_37 + 300*uk_38 + 46225*uk_39 + 25*uk_4 + 48375*uk_40 + 5375*uk_41 + 50625*uk_42 + 5625*uk_43 + 625*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 499048890434*uk_47 + 64310424025*uk_48 + 30869003532*uk_49 + 12*uk_5 + 553069646615*uk_50 + 578793816225*uk_51 + 64310424025*uk_52 + 153424975*uk_53 + 541171730*uk_54 + 69738625*uk_55 + 33474540*uk_56 + 599752175*uk_57 + 627647625*uk_58 + 69738625*uk_59 + 215*uk_6 + 1908860284*uk_60 + 245987150*uk_61 + 118073832*uk_62 + 2115489490*uk_63 + 2213884350*uk_64 + 245987150*uk_65 + 31699375*uk_66 + 15215700*uk_67 + 272614625*uk_68 + 285294375*uk_69 + 225*uk_7 + 31699375*uk_70 + 7303536*uk_71 + 130855020*uk_72 + 136941300*uk_73 + 15215700*uk_74 + 2344485775*uk_75 + 2453531625*uk_76 + 272614625*uk_77 + 2567649375*uk_78 + 285294375*uk_79 + 25*uk_8 + 31699375*uk_80 + 166375*uk_81 + 586850*uk_82 + 75625*uk_83 + 36300*uk_84 + 650375*uk_85 + 680625*uk_86 + 75625*uk_87 + 2069980*uk_88 + 266750*uk_89 + 2572416961*uk_9 + 128040*uk_90 + 2294050*uk_91 + 2400750*uk_92 + 266750*uk_93 + 34375*uk_94 + 16500*uk_95 + 295625*uk_96 + 309375*uk_97 + 34375*uk_98 + 7920*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 95480*uk_100 + 99000*uk_101 + 85360*uk_102 + 2589895*uk_103 + 2685375*uk_104 + 2315390*uk_105 + 2784375*uk_106 + 2400750*uk_107 + 2069980*uk_108 + 3944312*uk_109 + 8013602*uk_11 + 4843016*uk_110 + 199712*uk_111 + 5417188*uk_112 + 5616900*uk_113 + 4843016*uk_114 + 5946488*uk_115 + 245216*uk_116 + 6651484*uk_117 + 6896700*uk_118 + 5946488*uk_119 + 9839486*uk_12 + 10112*uk_120 + 274288*uk_121 + 284400*uk_122 + 245216*uk_123 + 7440062*uk_124 + 7714350*uk_125 + 6651484*uk_126 + 7998750*uk_127 + 6896700*uk_128 + 5946488*uk_129 + 405752*uk_13 + 7301384*uk_130 + 301088*uk_131 + 8167012*uk_132 + 8468100*uk_133 + 7301384*uk_134 + 12416*uk_135 + 336784*uk_136 + 349200*uk_137 + 301088*uk_138 + 9135266*uk_139 + 11006023*uk_14 + 9472050*uk_140 + 8167012*uk_141 + 9821250*uk_142 + 8468100*uk_143 + 7301384*uk_144 + 512*uk_145 + 13888*uk_146 + 14400*uk_147 + 12416*uk_148 + 376712*uk_149 + 11411775*uk_15 + 390600*uk_150 + 336784*uk_151 + 405000*uk_152 + 349200*uk_153 + 301088*uk_154 + 10218313*uk_155 + 10595025*uk_156 + 9135266*uk_157 + 10985625*uk_158 + 9472050*uk_159 + 9839486*uk_16 + 8167012*uk_160 + 11390625*uk_161 + 9821250*uk_162 + 8468100*uk_163 + 7301384*uk_164 + 3025*uk_17 + 8690*uk_18 + 10670*uk_19 + 55*uk_2 + 440*uk_20 + 11935*uk_21 + 12375*uk_22 + 10670*uk_23 + 24964*uk_24 + 30652*uk_25 + 1264*uk_26 + 34286*uk_27 + 35550*uk_28 + 30652*uk_29 + 158*uk_3 + 37636*uk_30 + 1552*uk_31 + 42098*uk_32 + 43650*uk_33 + 37636*uk_34 + 64*uk_35 + 1736*uk_36 + 1800*uk_37 + 1552*uk_38 + 47089*uk_39 + 194*uk_4 + 48825*uk_40 + 42098*uk_41 + 50625*uk_42 + 43650*uk_43 + 37636*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 406441879838*uk_47 + 499048890434*uk_48 + 20579335688*uk_49 + 8*uk_5 + 558214480537*uk_50 + 578793816225*uk_51 + 499048890434*uk_52 + 153424975*uk_53 + 440748110*uk_54 + 541171730*uk_55 + 22316360*uk_56 + 605331265*uk_57 + 627647625*uk_58 + 541171730*uk_59 + 217*uk_6 + 1266149116*uk_60 + 1554638788*uk_61 + 64108816*uk_62 + 1738951634*uk_63 + 1803060450*uk_64 + 1554638788*uk_65 + 1908860284*uk_66 + 78715888*uk_67 + 2135168462*uk_68 + 2213884350*uk_69 + 225*uk_7 + 1908860284*uk_70 + 3246016*uk_71 + 88048184*uk_72 + 91294200*uk_73 + 78715888*uk_74 + 2388306991*uk_75 + 2476355175*uk_76 + 2135168462*uk_77 + 2567649375*uk_78 + 2213884350*uk_79 + 194*uk_8 + 1908860284*uk_80 + 166375*uk_81 + 477950*uk_82 + 586850*uk_83 + 24200*uk_84 + 656425*uk_85 + 680625*uk_86 + 586850*uk_87 + 1373020*uk_88 + 1685860*uk_89 + 2572416961*uk_9 + 69520*uk_90 + 1885730*uk_91 + 1955250*uk_92 + 1685860*uk_93 + 2069980*uk_94 + 85360*uk_95 + 2315390*uk_96 + 2400750*uk_97 + 2069980*uk_98 + 3520*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 96360*uk_100 + 99000*uk_101 + 69520*uk_102 + 2637855*uk_103 + 2710125*uk_104 + 1903110*uk_105 + 2784375*uk_106 + 1955250*uk_107 + 1373020*uk_108 + 2197000*uk_109 + 6593470*uk_11 + 2670200*uk_110 + 135200*uk_111 + 3701100*uk_112 + 3802500*uk_113 + 2670200*uk_114 + 3245320*uk_115 + 164320*uk_116 + 4498260*uk_117 + 4621500*uk_118 + 3245320*uk_119 + 8013602*uk_12 + 8320*uk_120 + 227760*uk_121 + 234000*uk_122 + 164320*uk_123 + 6234930*uk_124 + 6405750*uk_125 + 4498260*uk_126 + 6581250*uk_127 + 4621500*uk_128 + 3245320*uk_129 + 405752*uk_13 + 3944312*uk_130 + 199712*uk_131 + 5467116*uk_132 + 5616900*uk_133 + 3944312*uk_134 + 10112*uk_135 + 276816*uk_136 + 284400*uk_137 + 199712*uk_138 + 7577838*uk_139 + 11107461*uk_14 + 7785450*uk_140 + 5467116*uk_141 + 7998750*uk_142 + 5616900*uk_143 + 3944312*uk_144 + 512*uk_145 + 14016*uk_146 + 14400*uk_147 + 10112*uk_148 + 383688*uk_149 + 11411775*uk_15 + 394200*uk_150 + 276816*uk_151 + 405000*uk_152 + 284400*uk_153 + 199712*uk_154 + 10503459*uk_155 + 10791225*uk_156 + 7577838*uk_157 + 11086875*uk_158 + 7785450*uk_159 + 8013602*uk_16 + 5467116*uk_160 + 11390625*uk_161 + 7998750*uk_162 + 5616900*uk_163 + 3944312*uk_164 + 3025*uk_17 + 7150*uk_18 + 8690*uk_19 + 55*uk_2 + 440*uk_20 + 12045*uk_21 + 12375*uk_22 + 8690*uk_23 + 16900*uk_24 + 20540*uk_25 + 1040*uk_26 + 28470*uk_27 + 29250*uk_28 + 20540*uk_29 + 130*uk_3 + 24964*uk_30 + 1264*uk_31 + 34602*uk_32 + 35550*uk_33 + 24964*uk_34 + 64*uk_35 + 1752*uk_36 + 1800*uk_37 + 1264*uk_38 + 47961*uk_39 + 158*uk_4 + 49275*uk_40 + 34602*uk_41 + 50625*uk_42 + 35550*uk_43 + 24964*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 334414204930*uk_47 + 406441879838*uk_48 + 20579335688*uk_49 + 8*uk_5 + 563359314459*uk_50 + 578793816225*uk_51 + 406441879838*uk_52 + 153424975*uk_53 + 362640850*uk_54 + 440748110*uk_55 + 22316360*uk_56 + 610910355*uk_57 + 627647625*uk_58 + 440748110*uk_59 + 219*uk_6 + 857151100*uk_60 + 1041768260*uk_61 + 52747760*uk_62 + 1443969930*uk_63 + 1483530750*uk_64 + 1041768260*uk_65 + 1266149116*uk_66 + 64108816*uk_67 + 1754978838*uk_68 + 1803060450*uk_69 + 225*uk_7 + 1266149116*uk_70 + 3246016*uk_71 + 88859688*uk_72 + 91294200*uk_73 + 64108816*uk_74 + 2432533959*uk_75 + 2499178725*uk_76 + 1754978838*uk_77 + 2567649375*uk_78 + 1803060450*uk_79 + 158*uk_8 + 1266149116*uk_80 + 166375*uk_81 + 393250*uk_82 + 477950*uk_83 + 24200*uk_84 + 662475*uk_85 + 680625*uk_86 + 477950*uk_87 + 929500*uk_88 + 1129700*uk_89 + 2572416961*uk_9 + 57200*uk_90 + 1565850*uk_91 + 1608750*uk_92 + 1129700*uk_93 + 1373020*uk_94 + 69520*uk_95 + 1903110*uk_96 + 1955250*uk_97 + 1373020*uk_98 + 3520*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 97240*uk_100 + 99000*uk_101 + 57200*uk_102 + 2686255*uk_103 + 2734875*uk_104 + 1580150*uk_105 + 2784375*uk_106 + 1608750*uk_107 + 929500*uk_108 + 1331000*uk_109 + 5579090*uk_11 + 1573000*uk_110 + 96800*uk_111 + 2674100*uk_112 + 2722500*uk_113 + 1573000*uk_114 + 1859000*uk_115 + 114400*uk_116 + 3160300*uk_117 + 3217500*uk_118 + 1859000*uk_119 + 6593470*uk_12 + 7040*uk_120 + 194480*uk_121 + 198000*uk_122 + 114400*uk_123 + 5372510*uk_124 + 5469750*uk_125 + 3160300*uk_126 + 5568750*uk_127 + 3217500*uk_128 + 1859000*uk_129 + 405752*uk_13 + 2197000*uk_130 + 135200*uk_131 + 3734900*uk_132 + 3802500*uk_133 + 2197000*uk_134 + 8320*uk_135 + 229840*uk_136 + 234000*uk_137 + 135200*uk_138 + 6349330*uk_139 + 11208899*uk_14 + 6464250*uk_140 + 3734900*uk_141 + 6581250*uk_142 + 3802500*uk_143 + 2197000*uk_144 + 512*uk_145 + 14144*uk_146 + 14400*uk_147 + 8320*uk_148 + 390728*uk_149 + 11411775*uk_15 + 397800*uk_150 + 229840*uk_151 + 405000*uk_152 + 234000*uk_153 + 135200*uk_154 + 10793861*uk_155 + 10989225*uk_156 + 6349330*uk_157 + 11188125*uk_158 + 6464250*uk_159 + 6593470*uk_16 + 3734900*uk_160 + 11390625*uk_161 + 6581250*uk_162 + 3802500*uk_163 + 2197000*uk_164 + 3025*uk_17 + 6050*uk_18 + 7150*uk_19 + 55*uk_2 + 440*uk_20 + 12155*uk_21 + 12375*uk_22 + 7150*uk_23 + 12100*uk_24 + 14300*uk_25 + 880*uk_26 + 24310*uk_27 + 24750*uk_28 + 14300*uk_29 + 110*uk_3 + 16900*uk_30 + 1040*uk_31 + 28730*uk_32 + 29250*uk_33 + 16900*uk_34 + 64*uk_35 + 1768*uk_36 + 1800*uk_37 + 1040*uk_38 + 48841*uk_39 + 130*uk_4 + 49725*uk_40 + 28730*uk_41 + 50625*uk_42 + 29250*uk_43 + 16900*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 282965865710*uk_47 + 334414204930*uk_48 + 20579335688*uk_49 + 8*uk_5 + 568504148381*uk_50 + 578793816225*uk_51 + 334414204930*uk_52 + 153424975*uk_53 + 306849950*uk_54 + 362640850*uk_55 + 22316360*uk_56 + 616489445*uk_57 + 627647625*uk_58 + 362640850*uk_59 + 221*uk_6 + 613699900*uk_60 + 725281700*uk_61 + 44632720*uk_62 + 1232978890*uk_63 + 1255295250*uk_64 + 725281700*uk_65 + 857151100*uk_66 + 52747760*uk_67 + 1457156870*uk_68 + 1483530750*uk_69 + 225*uk_7 + 857151100*uk_70 + 3246016*uk_71 + 89671192*uk_72 + 91294200*uk_73 + 52747760*uk_74 + 2477166679*uk_75 + 2522002275*uk_76 + 1457156870*uk_77 + 2567649375*uk_78 + 1483530750*uk_79 + 130*uk_8 + 857151100*uk_80 + 166375*uk_81 + 332750*uk_82 + 393250*uk_83 + 24200*uk_84 + 668525*uk_85 + 680625*uk_86 + 393250*uk_87 + 665500*uk_88 + 786500*uk_89 + 2572416961*uk_9 + 48400*uk_90 + 1337050*uk_91 + 1361250*uk_92 + 786500*uk_93 + 929500*uk_94 + 57200*uk_95 + 1580150*uk_96 + 1608750*uk_97 + 929500*uk_98 + 3520*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 98120*uk_100 + 99000*uk_101 + 48400*uk_102 + 2735095*uk_103 + 2759625*uk_104 + 1349150*uk_105 + 2784375*uk_106 + 1361250*uk_107 + 665500*uk_108 + 941192*uk_109 + 4970462*uk_11 + 1056440*uk_110 + 76832*uk_111 + 2141692*uk_112 + 2160900*uk_113 + 1056440*uk_114 + 1185800*uk_115 + 86240*uk_116 + 2403940*uk_117 + 2425500*uk_118 + 1185800*uk_119 + 5579090*uk_12 + 6272*uk_120 + 174832*uk_121 + 176400*uk_122 + 86240*uk_123 + 4873442*uk_124 + 4917150*uk_125 + 2403940*uk_126 + 4961250*uk_127 + 2425500*uk_128 + 1185800*uk_129 + 405752*uk_13 + 1331000*uk_130 + 96800*uk_131 + 2698300*uk_132 + 2722500*uk_133 + 1331000*uk_134 + 7040*uk_135 + 196240*uk_136 + 198000*uk_137 + 96800*uk_138 + 5470190*uk_139 + 11310337*uk_14 + 5519250*uk_140 + 2698300*uk_141 + 5568750*uk_142 + 2722500*uk_143 + 1331000*uk_144 + 512*uk_145 + 14272*uk_146 + 14400*uk_147 + 7040*uk_148 + 397832*uk_149 + 11411775*uk_15 + 401400*uk_150 + 196240*uk_151 + 405000*uk_152 + 198000*uk_153 + 96800*uk_154 + 11089567*uk_155 + 11189025*uk_156 + 5470190*uk_157 + 11289375*uk_158 + 5519250*uk_159 + 5579090*uk_16 + 2698300*uk_160 + 11390625*uk_161 + 5568750*uk_162 + 2722500*uk_163 + 1331000*uk_164 + 3025*uk_17 + 5390*uk_18 + 6050*uk_19 + 55*uk_2 + 440*uk_20 + 12265*uk_21 + 12375*uk_22 + 6050*uk_23 + 9604*uk_24 + 10780*uk_25 + 784*uk_26 + 21854*uk_27 + 22050*uk_28 + 10780*uk_29 + 98*uk_3 + 12100*uk_30 + 880*uk_31 + 24530*uk_32 + 24750*uk_33 + 12100*uk_34 + 64*uk_35 + 1784*uk_36 + 1800*uk_37 + 880*uk_38 + 49729*uk_39 + 110*uk_4 + 50175*uk_40 + 24530*uk_41 + 50625*uk_42 + 24750*uk_43 + 12100*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 252096862178*uk_47 + 282965865710*uk_48 + 20579335688*uk_49 + 8*uk_5 + 573648982303*uk_50 + 578793816225*uk_51 + 282965865710*uk_52 + 153424975*uk_53 + 273375410*uk_54 + 306849950*uk_55 + 22316360*uk_56 + 622068535*uk_57 + 627647625*uk_58 + 306849950*uk_59 + 223*uk_6 + 487105276*uk_60 + 546750820*uk_61 + 39763696*uk_62 + 1108413026*uk_63 + 1118353950*uk_64 + 546750820*uk_65 + 613699900*uk_66 + 44632720*uk_67 + 1244137070*uk_68 + 1255295250*uk_69 + 225*uk_7 + 613699900*uk_70 + 3246016*uk_71 + 90482696*uk_72 + 91294200*uk_73 + 44632720*uk_74 + 2522205151*uk_75 + 2544825825*uk_76 + 1244137070*uk_77 + 2567649375*uk_78 + 1255295250*uk_79 + 110*uk_8 + 613699900*uk_80 + 166375*uk_81 + 296450*uk_82 + 332750*uk_83 + 24200*uk_84 + 674575*uk_85 + 680625*uk_86 + 332750*uk_87 + 528220*uk_88 + 592900*uk_89 + 2572416961*uk_9 + 43120*uk_90 + 1201970*uk_91 + 1212750*uk_92 + 592900*uk_93 + 665500*uk_94 + 48400*uk_95 + 1349150*uk_96 + 1361250*uk_97 + 665500*uk_98 + 3520*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 99000*uk_100 + 99000*uk_101 + 43120*uk_102 + 2784375*uk_103 + 2784375*uk_104 + 1212750*uk_105 + 2784375*uk_106 + 1212750*uk_107 + 528220*uk_108 + 830584*uk_109 + 4767586*uk_11 + 865928*uk_110 + 70688*uk_111 + 1988100*uk_112 + 1988100*uk_113 + 865928*uk_114 + 902776*uk_115 + 73696*uk_116 + 2072700*uk_117 + 2072700*uk_118 + 902776*uk_119 + 4970462*uk_12 + 6016*uk_120 + 169200*uk_121 + 169200*uk_122 + 73696*uk_123 + 4758750*uk_124 + 4758750*uk_125 + 2072700*uk_126 + 4758750*uk_127 + 2072700*uk_128 + 902776*uk_129 + 405752*uk_13 + 941192*uk_130 + 76832*uk_131 + 2160900*uk_132 + 2160900*uk_133 + 941192*uk_134 + 6272*uk_135 + 176400*uk_136 + 176400*uk_137 + 76832*uk_138 + 4961250*uk_139 + 11411775*uk_14 + 4961250*uk_140 + 2160900*uk_141 + 4961250*uk_142 + 2160900*uk_143 + 941192*uk_144 + 512*uk_145 + 14400*uk_146 + 14400*uk_147 + 6272*uk_148 + 405000*uk_149 + 11411775*uk_15 + 405000*uk_150 + 176400*uk_151 + 405000*uk_152 + 176400*uk_153 + 76832*uk_154 + 11390625*uk_155 + 11390625*uk_156 + 4961250*uk_157 + 11390625*uk_158 + 4961250*uk_159 + 4970462*uk_16 + 2160900*uk_160 + 11390625*uk_161 + 4961250*uk_162 + 2160900*uk_163 + 941192*uk_164 + 3025*uk_17 + 5170*uk_18 + 5390*uk_19 + 55*uk_2 + 440*uk_20 + 12375*uk_21 + 12375*uk_22 + 5390*uk_23 + 8836*uk_24 + 9212*uk_25 + 752*uk_26 + 21150*uk_27 + 21150*uk_28 + 9212*uk_29 + 94*uk_3 + 9604*uk_30 + 784*uk_31 + 22050*uk_32 + 22050*uk_33 + 9604*uk_34 + 64*uk_35 + 1800*uk_36 + 1800*uk_37 + 784*uk_38 + 50625*uk_39 + 98*uk_4 + 50625*uk_40 + 22050*uk_41 + 50625*uk_42 + 22050*uk_43 + 9604*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 241807194334*uk_47 + 252096862178*uk_48 + 20579335688*uk_49 + 8*uk_5 + 578793816225*uk_50 + 578793816225*uk_51 + 252096862178*uk_52 + 153424975*uk_53 + 262217230*uk_54 + 273375410*uk_55 + 22316360*uk_56 + 627647625*uk_57 + 627647625*uk_58 + 273375410*uk_59 + 225*uk_6 + 448153084*uk_60 + 467223428*uk_61 + 38140688*uk_62 + 1072706850*uk_63 + 1072706850*uk_64 + 467223428*uk_65 + 487105276*uk_66 + 39763696*uk_67 + 1118353950*uk_68 + 1118353950*uk_69 + 225*uk_7 + 487105276*uk_70 + 3246016*uk_71 + 91294200*uk_72 + 91294200*uk_73 + 39763696*uk_74 + 2567649375*uk_75 + 2567649375*uk_76 + 1118353950*uk_77 + 2567649375*uk_78 + 1118353950*uk_79 + 98*uk_8 + 487105276*uk_80 + 166375*uk_81 + 284350*uk_82 + 296450*uk_83 + 24200*uk_84 + 680625*uk_85 + 680625*uk_86 + 296450*uk_87 + 485980*uk_88 + 506660*uk_89 + 2572416961*uk_9 + 41360*uk_90 + 1163250*uk_91 + 1163250*uk_92 + 506660*uk_93 + 528220*uk_94 + 43120*uk_95 + 1212750*uk_96 + 1212750*uk_97 + 528220*uk_98 + 3520*uk_99,
        uk_0 + 50719*uk_1 + 2789545*uk_10 + 99880*uk_100 + 99000*uk_101 + 41360*uk_102 + 2834095*uk_103 + 2809125*uk_104 + 1173590*uk_105 + 2784375*uk_106 + 1163250*uk_107 + 485980*uk_108 + 941192*uk_109 + 4970462*uk_11 + 902776*uk_110 + 76832*uk_111 + 2180108*uk_112 + 2160900*uk_113 + 902776*uk_114 + 865928*uk_115 + 73696*uk_116 + 2091124*uk_117 + 2072700*uk_118 + 865928*uk_119 + 4767586*uk_12 + 6272*uk_120 + 177968*uk_121 + 176400*uk_122 + 73696*uk_123 + 5049842*uk_124 + 5005350*uk_125 + 2091124*uk_126 + 4961250*uk_127 + 2072700*uk_128 + 865928*uk_129 + 405752*uk_13 + 830584*uk_130 + 70688*uk_131 + 2005772*uk_132 + 1988100*uk_133 + 830584*uk_134 + 6016*uk_135 + 170704*uk_136 + 169200*uk_137 + 70688*uk_138 + 4843726*uk_139 + 11513213*uk_14 + 4801050*uk_140 + 2005772*uk_141 + 4758750*uk_142 + 1988100*uk_143 + 830584*uk_144 + 512*uk_145 + 14528*uk_146 + 14400*uk_147 + 6016*uk_148 + 412232*uk_149 + 11411775*uk_15 + 408600*uk_150 + 170704*uk_151 + 405000*uk_152 + 169200*uk_153 + 70688*uk_154 + 11697083*uk_155 + 11594025*uk_156 + 4843726*uk_157 + 11491875*uk_158 + 4801050*uk_159 + 4767586*uk_16 + 2005772*uk_160 + 11390625*uk_161 + 4758750*uk_162 + 1988100*uk_163 + 830584*uk_164 + 3025*uk_17 + 5390*uk_18 + 5170*uk_19 + 55*uk_2 + 440*uk_20 + 12485*uk_21 + 12375*uk_22 + 5170*uk_23 + 9604*uk_24 + 9212*uk_25 + 784*uk_26 + 22246*uk_27 + 22050*uk_28 + 9212*uk_29 + 98*uk_3 + 8836*uk_30 + 752*uk_31 + 21338*uk_32 + 21150*uk_33 + 8836*uk_34 + 64*uk_35 + 1816*uk_36 + 1800*uk_37 + 752*uk_38 + 51529*uk_39 + 94*uk_4 + 51075*uk_40 + 21338*uk_41 + 50625*uk_42 + 21150*uk_43 + 8836*uk_44 + 130470415844959*uk_45 + 141482932855*uk_46 + 252096862178*uk_47 + 241807194334*uk_48 + 20579335688*uk_49 + 8*uk_5 + 583938650147*uk_50 + 578793816225*uk_51 + 241807194334*uk_52 + 153424975*uk_53 + 273375410*uk_54 + 262217230*uk_55 + 22316360*uk_56 + 633226715*uk_57 + 627647625*uk_58 + 262217230*uk_59 + 227*uk_6 + 487105276*uk_60 + 467223428*uk_61 + 39763696*uk_62 + 1128294874*uk_63 + 1118353950*uk_64 + 467223428*uk_65 + 448153084*uk_66 + 38140688*uk_67 + 1082242022*uk_68 + 1072706850*uk_69 + 225*uk_7 + 448153084*uk_70 + 3246016*uk_71 + 92105704*uk_72 + 91294200*uk_73 + 38140688*uk_74 + 2613499351*uk_75 + 2590472925*uk_76 + 1082242022*uk_77 + 2567649375*uk_78 + 1072706850*uk_79 + 94*uk_8 + 448153084*uk_80 + 166375*uk_81 + 296450*uk_82 + 284350*uk_83 + 24200*uk_84 + 686675*uk_85 + 680625*uk_86 + 284350*uk_87 + 528220*uk_88 + 506660*uk_89 + 2572416961*uk_9 + 43120*uk_90 + 1223530*uk_91 + 1212750*uk_92 + 506660*uk_93 + 485980*uk_94 + 41360*uk_95 + 1173590*uk_96 + 1163250*uk_97 + 485980*uk_98 + 3520*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 396900*uk_100 + 1367100*uk_101 + 250047*uk_103 + 861273*uk_104 + 2966607*uk_106 + 64000*uk_109 + 1894120*uk_11 + 27200*uk_110 + 160000*uk_111 + 100800*uk_112 + 347200*uk_113 + 11560*uk_115 + 68000*uk_116 + 42840*uk_117 + 147560*uk_118 + 805001*uk_12 + 400000*uk_120 + 252000*uk_121 + 868000*uk_122 + 158760*uk_124 + 546840*uk_125 + 1883560*uk_127 + 4735300*uk_13 + 4913*uk_130 + 28900*uk_131 + 18207*uk_132 + 62713*uk_133 + 170000*uk_135 + 107100*uk_136 + 368900*uk_137 + 67473*uk_139 + 2983239*uk_14 + 232407*uk_140 + 800513*uk_142 + 1000000*uk_145 + 630000*uk_146 + 2170000*uk_147 + 396900*uk_149 + 10275601*uk_15 + 1367100*uk_150 + 4708900*uk_152 + 250047*uk_155 + 861273*uk_156 + 2966607*uk_158 + 10218313*uk_161 + 3969*uk_17 + 2520*uk_18 + 1071*uk_19 + 63*uk_2 + 6300*uk_20 + 3969*uk_21 + 13671*uk_22 + 1600*uk_24 + 680*uk_25 + 4000*uk_26 + 2520*uk_27 + 8680*uk_28 + 40*uk_3 + 289*uk_30 + 1700*uk_31 + 1071*uk_32 + 3689*uk_33 + 10000*uk_35 + 6300*uk_36 + 21700*uk_37 + 3969*uk_39 + 17*uk_4 + 13671*uk_40 + 47089*uk_42 + 106179944855977*uk_45 + 141265316367*uk_46 + 89692264360*uk_47 + 38119212353*uk_48 + 224230660900*uk_49 + 100*uk_5 + 141265316367*uk_50 + 486580534153*uk_51 + 187944057*uk_53 + 119329560*uk_54 + 50715063*uk_55 + 298323900*uk_56 + 187944057*uk_57 + 647362863*uk_58 + 63*uk_6 + 75764800*uk_60 + 32200040*uk_61 + 189412000*uk_62 + 119329560*uk_63 + 411024040*uk_64 + 13685017*uk_66 + 80500100*uk_67 + 50715063*uk_68 + 174685217*uk_69 + 217*uk_7 + 473530000*uk_71 + 298323900*uk_72 + 1027560100*uk_73 + 187944057*uk_75 + 647362863*uk_76 + 2229805417*uk_78 + 250047*uk_81 + 158760*uk_82 + 67473*uk_83 + 396900*uk_84 + 250047*uk_85 + 861273*uk_86 + 100800*uk_88 + 42840*uk_89 + 2242306609*uk_9 + 252000*uk_90 + 158760*uk_91 + 546840*uk_92 + 18207*uk_94 + 107100*uk_95 + 67473*uk_96 + 232407*uk_97 + 630000*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 376740*uk_100 + 1257732*uk_101 + 231840*uk_102 + 266175*uk_103 + 888615*uk_104 + 163800*uk_105 + 2966607*uk_106 + 546840*uk_107 + 100800*uk_108 + 35937*uk_109 + 1562649*uk_11 + 43560*uk_110 + 100188*uk_111 + 70785*uk_112 + 236313*uk_113 + 43560*uk_114 + 52800*uk_115 + 121440*uk_116 + 85800*uk_117 + 286440*uk_118 + 52800*uk_119 + 1894120*uk_12 + 279312*uk_120 + 197340*uk_121 + 658812*uk_122 + 121440*uk_123 + 139425*uk_124 + 465465*uk_125 + 85800*uk_126 + 1553937*uk_127 + 286440*uk_128 + 52800*uk_129 + 4356476*uk_13 + 64000*uk_130 + 147200*uk_131 + 104000*uk_132 + 347200*uk_133 + 64000*uk_134 + 338560*uk_135 + 239200*uk_136 + 798560*uk_137 + 147200*uk_138 + 169000*uk_139 + 3077945*uk_14 + 564200*uk_140 + 104000*uk_141 + 1883560*uk_142 + 347200*uk_143 + 64000*uk_144 + 778688*uk_145 + 550160*uk_146 + 1836688*uk_147 + 338560*uk_148 + 388700*uk_149 + 10275601*uk_15 + 1297660*uk_150 + 239200*uk_151 + 4332188*uk_152 + 798560*uk_153 + 147200*uk_154 + 274625*uk_155 + 916825*uk_156 + 169000*uk_157 + 3060785*uk_158 + 564200*uk_159 + 1894120*uk_16 + 104000*uk_160 + 10218313*uk_161 + 1883560*uk_162 + 347200*uk_163 + 64000*uk_164 + 3969*uk_17 + 2079*uk_18 + 2520*uk_19 + 63*uk_2 + 5796*uk_20 + 4095*uk_21 + 13671*uk_22 + 2520*uk_23 + 1089*uk_24 + 1320*uk_25 + 3036*uk_26 + 2145*uk_27 + 7161*uk_28 + 1320*uk_29 + 33*uk_3 + 1600*uk_30 + 3680*uk_31 + 2600*uk_32 + 8680*uk_33 + 1600*uk_34 + 8464*uk_35 + 5980*uk_36 + 19964*uk_37 + 3680*uk_38 + 4225*uk_39 + 40*uk_4 + 14105*uk_40 + 2600*uk_41 + 47089*uk_42 + 8680*uk_43 + 1600*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 73996118097*uk_47 + 89692264360*uk_48 + 206292208028*uk_49 + 92*uk_5 + 145749929585*uk_50 + 486580534153*uk_51 + 89692264360*uk_52 + 187944057*uk_53 + 98446887*uk_54 + 119329560*uk_55 + 274457988*uk_56 + 193910535*uk_57 + 647362863*uk_58 + 119329560*uk_59 + 65*uk_6 + 51567417*uk_60 + 62505960*uk_61 + 143763708*uk_62 + 101572185*uk_63 + 339094833*uk_64 + 62505960*uk_65 + 75764800*uk_66 + 174259040*uk_67 + 123117800*uk_68 + 411024040*uk_69 + 217*uk_7 + 75764800*uk_70 + 400795792*uk_71 + 283170940*uk_72 + 945355292*uk_73 + 174259040*uk_74 + 200066425*uk_75 + 667914065*uk_76 + 123117800*uk_77 + 2229805417*uk_78 + 411024040*uk_79 + 40*uk_8 + 75764800*uk_80 + 250047*uk_81 + 130977*uk_82 + 158760*uk_83 + 365148*uk_84 + 257985*uk_85 + 861273*uk_86 + 158760*uk_87 + 68607*uk_88 + 83160*uk_89 + 2242306609*uk_9 + 191268*uk_90 + 135135*uk_91 + 451143*uk_92 + 83160*uk_93 + 100800*uk_94 + 231840*uk_95 + 163800*uk_96 + 546840*uk_97 + 100800*uk_98 + 533232*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 371448*uk_100 + 1203048*uk_101 + 182952*uk_102 + 282807*uk_103 + 915957*uk_104 + 139293*uk_105 + 2966607*uk_106 + 451143*uk_107 + 68607*uk_108 + 132651*uk_109 + 2415003*uk_11 + 85833*uk_110 + 228888*uk_111 + 174267*uk_112 + 564417*uk_113 + 85833*uk_114 + 55539*uk_115 + 148104*uk_116 + 112761*uk_117 + 365211*uk_118 + 55539*uk_119 + 1562649*uk_12 + 394944*uk_120 + 300696*uk_121 + 973896*uk_122 + 148104*uk_123 + 228939*uk_124 + 741489*uk_125 + 112761*uk_126 + 2401539*uk_127 + 365211*uk_128 + 55539*uk_129 + 4167064*uk_13 + 35937*uk_130 + 95832*uk_131 + 72963*uk_132 + 236313*uk_133 + 35937*uk_134 + 255552*uk_135 + 194568*uk_136 + 630168*uk_137 + 95832*uk_138 + 148137*uk_139 + 3172651*uk_14 + 479787*uk_140 + 72963*uk_141 + 1553937*uk_142 + 236313*uk_143 + 35937*uk_144 + 681472*uk_145 + 518848*uk_146 + 1680448*uk_147 + 255552*uk_148 + 395032*uk_149 + 10275601*uk_15 + 1279432*uk_150 + 194568*uk_151 + 4143832*uk_152 + 630168*uk_153 + 95832*uk_154 + 300763*uk_155 + 974113*uk_156 + 148137*uk_157 + 3154963*uk_158 + 479787*uk_159 + 1562649*uk_16 + 72963*uk_160 + 10218313*uk_161 + 1553937*uk_162 + 236313*uk_163 + 35937*uk_164 + 3969*uk_17 + 3213*uk_18 + 2079*uk_19 + 63*uk_2 + 5544*uk_20 + 4221*uk_21 + 13671*uk_22 + 2079*uk_23 + 2601*uk_24 + 1683*uk_25 + 4488*uk_26 + 3417*uk_27 + 11067*uk_28 + 1683*uk_29 + 51*uk_3 + 1089*uk_30 + 2904*uk_31 + 2211*uk_32 + 7161*uk_33 + 1089*uk_34 + 7744*uk_35 + 5896*uk_36 + 19096*uk_37 + 2904*uk_38 + 4489*uk_39 + 33*uk_4 + 14539*uk_40 + 2211*uk_41 + 47089*uk_42 + 7161*uk_43 + 1089*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 114357637059*uk_47 + 73996118097*uk_48 + 197322981592*uk_49 + 88*uk_5 + 150234542803*uk_50 + 486580534153*uk_51 + 73996118097*uk_52 + 187944057*uk_53 + 152145189*uk_54 + 98446887*uk_55 + 262525032*uk_56 + 199877013*uk_57 + 647362863*uk_58 + 98446887*uk_59 + 67*uk_6 + 123165153*uk_60 + 79695099*uk_61 + 212520264*uk_62 + 161805201*uk_63 + 524055651*uk_64 + 79695099*uk_65 + 51567417*uk_66 + 137513112*uk_67 + 104697483*uk_68 + 339094833*uk_69 + 217*uk_7 + 51567417*uk_70 + 366701632*uk_71 + 279193288*uk_72 + 904252888*uk_73 + 137513112*uk_74 + 212567617*uk_75 + 688465267*uk_76 + 104697483*uk_77 + 2229805417*uk_78 + 339094833*uk_79 + 33*uk_8 + 51567417*uk_80 + 250047*uk_81 + 202419*uk_82 + 130977*uk_83 + 349272*uk_84 + 265923*uk_85 + 861273*uk_86 + 130977*uk_87 + 163863*uk_88 + 106029*uk_89 + 2242306609*uk_9 + 282744*uk_90 + 215271*uk_91 + 697221*uk_92 + 106029*uk_93 + 68607*uk_94 + 182952*uk_95 + 139293*uk_96 + 451143*uk_97 + 68607*uk_98 + 487872*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 347760*uk_100 + 1093680*uk_101 + 257040*uk_102 + 299943*uk_103 + 943299*uk_104 + 221697*uk_105 + 2966607*uk_106 + 697221*uk_107 + 163863*uk_108 + 6859*uk_109 + 899707*uk_11 + 18411*uk_110 + 28880*uk_111 + 24909*uk_112 + 78337*uk_113 + 18411*uk_114 + 49419*uk_115 + 77520*uk_116 + 66861*uk_117 + 210273*uk_118 + 49419*uk_119 + 2415003*uk_12 + 121600*uk_120 + 104880*uk_121 + 329840*uk_122 + 77520*uk_123 + 90459*uk_124 + 284487*uk_125 + 66861*uk_126 + 894691*uk_127 + 210273*uk_128 + 49419*uk_129 + 3788240*uk_13 + 132651*uk_130 + 208080*uk_131 + 179469*uk_132 + 564417*uk_133 + 132651*uk_134 + 326400*uk_135 + 281520*uk_136 + 885360*uk_137 + 208080*uk_138 + 242811*uk_139 + 3267357*uk_14 + 763623*uk_140 + 179469*uk_141 + 2401539*uk_142 + 564417*uk_143 + 132651*uk_144 + 512000*uk_145 + 441600*uk_146 + 1388800*uk_147 + 326400*uk_148 + 380880*uk_149 + 10275601*uk_15 + 1197840*uk_150 + 281520*uk_151 + 3767120*uk_152 + 885360*uk_153 + 208080*uk_154 + 328509*uk_155 + 1033137*uk_156 + 242811*uk_157 + 3249141*uk_158 + 763623*uk_159 + 2415003*uk_16 + 179469*uk_160 + 10218313*uk_161 + 2401539*uk_162 + 564417*uk_163 + 132651*uk_164 + 3969*uk_17 + 1197*uk_18 + 3213*uk_19 + 63*uk_2 + 5040*uk_20 + 4347*uk_21 + 13671*uk_22 + 3213*uk_23 + 361*uk_24 + 969*uk_25 + 1520*uk_26 + 1311*uk_27 + 4123*uk_28 + 969*uk_29 + 19*uk_3 + 2601*uk_30 + 4080*uk_31 + 3519*uk_32 + 11067*uk_33 + 2601*uk_34 + 6400*uk_35 + 5520*uk_36 + 17360*uk_37 + 4080*uk_38 + 4761*uk_39 + 51*uk_4 + 14973*uk_40 + 3519*uk_41 + 47089*uk_42 + 11067*uk_43 + 2601*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 42603825571*uk_47 + 114357637059*uk_48 + 179384528720*uk_49 + 80*uk_5 + 154719156021*uk_50 + 486580534153*uk_51 + 114357637059*uk_52 + 187944057*uk_53 + 56681541*uk_54 + 152145189*uk_55 + 238659120*uk_56 + 205843491*uk_57 + 647362863*uk_58 + 152145189*uk_59 + 69*uk_6 + 17094433*uk_60 + 45885057*uk_61 + 71976560*uk_62 + 62079783*uk_63 + 195236419*uk_64 + 45885057*uk_65 + 123165153*uk_66 + 193200240*uk_67 + 166635207*uk_68 + 524055651*uk_69 + 217*uk_7 + 123165153*uk_70 + 303059200*uk_71 + 261388560*uk_72 + 822048080*uk_73 + 193200240*uk_74 + 225447633*uk_75 + 709016469*uk_76 + 166635207*uk_77 + 2229805417*uk_78 + 524055651*uk_79 + 51*uk_8 + 123165153*uk_80 + 250047*uk_81 + 75411*uk_82 + 202419*uk_83 + 317520*uk_84 + 273861*uk_85 + 861273*uk_86 + 202419*uk_87 + 22743*uk_88 + 61047*uk_89 + 2242306609*uk_9 + 95760*uk_90 + 82593*uk_91 + 259749*uk_92 + 61047*uk_93 + 163863*uk_94 + 257040*uk_95 + 221697*uk_96 + 697221*uk_97 + 163863*uk_98 + 403200*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 357840*uk_100 + 1093680*uk_101 + 95760*uk_102 + 317583*uk_103 + 970641*uk_104 + 84987*uk_105 + 2966607*uk_106 + 259749*uk_107 + 22743*uk_108 + 300763*uk_109 + 3172651*uk_11 + 85291*uk_110 + 359120*uk_111 + 318719*uk_112 + 974113*uk_113 + 85291*uk_114 + 24187*uk_115 + 101840*uk_116 + 90383*uk_117 + 276241*uk_118 + 24187*uk_119 + 899707*uk_12 + 428800*uk_120 + 380560*uk_121 + 1163120*uk_122 + 101840*uk_123 + 337747*uk_124 + 1032269*uk_125 + 90383*uk_126 + 3154963*uk_127 + 276241*uk_128 + 24187*uk_129 + 3788240*uk_13 + 6859*uk_130 + 28880*uk_131 + 25631*uk_132 + 78337*uk_133 + 6859*uk_134 + 121600*uk_135 + 107920*uk_136 + 329840*uk_137 + 28880*uk_138 + 95779*uk_139 + 3362063*uk_14 + 292733*uk_140 + 25631*uk_141 + 894691*uk_142 + 78337*uk_143 + 6859*uk_144 + 512000*uk_145 + 454400*uk_146 + 1388800*uk_147 + 121600*uk_148 + 403280*uk_149 + 10275601*uk_15 + 1232560*uk_150 + 107920*uk_151 + 3767120*uk_152 + 329840*uk_153 + 28880*uk_154 + 357911*uk_155 + 1093897*uk_156 + 95779*uk_157 + 3343319*uk_158 + 292733*uk_159 + 899707*uk_16 + 25631*uk_160 + 10218313*uk_161 + 894691*uk_162 + 78337*uk_163 + 6859*uk_164 + 3969*uk_17 + 4221*uk_18 + 1197*uk_19 + 63*uk_2 + 5040*uk_20 + 4473*uk_21 + 13671*uk_22 + 1197*uk_23 + 4489*uk_24 + 1273*uk_25 + 5360*uk_26 + 4757*uk_27 + 14539*uk_28 + 1273*uk_29 + 67*uk_3 + 361*uk_30 + 1520*uk_31 + 1349*uk_32 + 4123*uk_33 + 361*uk_34 + 6400*uk_35 + 5680*uk_36 + 17360*uk_37 + 1520*uk_38 + 5041*uk_39 + 19*uk_4 + 15407*uk_40 + 1349*uk_41 + 47089*uk_42 + 4123*uk_43 + 361*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 150234542803*uk_47 + 42603825571*uk_48 + 179384528720*uk_49 + 80*uk_5 + 159203769239*uk_50 + 486580534153*uk_51 + 42603825571*uk_52 + 187944057*uk_53 + 199877013*uk_54 + 56681541*uk_55 + 238659120*uk_56 + 211809969*uk_57 + 647362863*uk_58 + 56681541*uk_59 + 71*uk_6 + 212567617*uk_60 + 60280369*uk_61 + 253812080*uk_62 + 225258221*uk_63 + 688465267*uk_64 + 60280369*uk_65 + 17094433*uk_66 + 71976560*uk_67 + 63879197*uk_68 + 195236419*uk_69 + 217*uk_7 + 17094433*uk_70 + 303059200*uk_71 + 268965040*uk_72 + 822048080*uk_73 + 71976560*uk_74 + 238706473*uk_75 + 729567671*uk_76 + 63879197*uk_77 + 2229805417*uk_78 + 195236419*uk_79 + 19*uk_8 + 17094433*uk_80 + 250047*uk_81 + 265923*uk_82 + 75411*uk_83 + 317520*uk_84 + 281799*uk_85 + 861273*uk_86 + 75411*uk_87 + 282807*uk_88 + 80199*uk_89 + 2242306609*uk_9 + 337680*uk_90 + 299691*uk_91 + 915957*uk_92 + 80199*uk_93 + 22743*uk_94 + 95760*uk_95 + 84987*uk_96 + 259749*uk_97 + 22743*uk_98 + 403200*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 331128*uk_100 + 984312*uk_101 + 303912*uk_102 + 335727*uk_103 + 997983*uk_104 + 308133*uk_105 + 2966607*uk_106 + 915957*uk_107 + 282807*uk_108 + 117649*uk_109 + 2320297*uk_11 + 160867*uk_110 + 172872*uk_111 + 175273*uk_112 + 521017*uk_113 + 160867*uk_114 + 219961*uk_115 + 236376*uk_116 + 239659*uk_117 + 712411*uk_118 + 219961*uk_119 + 3172651*uk_12 + 254016*uk_120 + 257544*uk_121 + 765576*uk_122 + 236376*uk_123 + 261121*uk_124 + 776209*uk_125 + 239659*uk_126 + 2307361*uk_127 + 712411*uk_128 + 219961*uk_129 + 3409416*uk_13 + 300763*uk_130 + 323208*uk_131 + 327697*uk_132 + 974113*uk_133 + 300763*uk_134 + 347328*uk_135 + 352152*uk_136 + 1046808*uk_137 + 323208*uk_138 + 357043*uk_139 + 3456769*uk_14 + 1061347*uk_140 + 327697*uk_141 + 3154963*uk_142 + 974113*uk_143 + 300763*uk_144 + 373248*uk_145 + 378432*uk_146 + 1124928*uk_147 + 347328*uk_148 + 383688*uk_149 + 10275601*uk_15 + 1140552*uk_150 + 352152*uk_151 + 3390408*uk_152 + 1046808*uk_153 + 323208*uk_154 + 389017*uk_155 + 1156393*uk_156 + 357043*uk_157 + 3437497*uk_158 + 1061347*uk_159 + 3172651*uk_16 + 327697*uk_160 + 10218313*uk_161 + 3154963*uk_162 + 974113*uk_163 + 300763*uk_164 + 3969*uk_17 + 3087*uk_18 + 4221*uk_19 + 63*uk_2 + 4536*uk_20 + 4599*uk_21 + 13671*uk_22 + 4221*uk_23 + 2401*uk_24 + 3283*uk_25 + 3528*uk_26 + 3577*uk_27 + 10633*uk_28 + 3283*uk_29 + 49*uk_3 + 4489*uk_30 + 4824*uk_31 + 4891*uk_32 + 14539*uk_33 + 4489*uk_34 + 5184*uk_35 + 5256*uk_36 + 15624*uk_37 + 4824*uk_38 + 5329*uk_39 + 67*uk_4 + 15841*uk_40 + 4891*uk_41 + 47089*uk_42 + 14539*uk_43 + 4489*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 109873023841*uk_47 + 150234542803*uk_48 + 161446075848*uk_49 + 72*uk_5 + 163688382457*uk_50 + 486580534153*uk_51 + 150234542803*uk_52 + 187944057*uk_53 + 146178711*uk_54 + 199877013*uk_55 + 214793208*uk_56 + 217776447*uk_57 + 647362863*uk_58 + 199877013*uk_59 + 73*uk_6 + 113694553*uk_60 + 155459899*uk_61 + 167061384*uk_62 + 169381681*uk_63 + 503504449*uk_64 + 155459899*uk_65 + 212567617*uk_66 + 228430872*uk_67 + 231603523*uk_68 + 688465267*uk_69 + 217*uk_7 + 212567617*uk_70 + 245477952*uk_71 + 248887368*uk_72 + 739843272*uk_73 + 228430872*uk_74 + 252344137*uk_75 + 750118873*uk_76 + 231603523*uk_77 + 2229805417*uk_78 + 688465267*uk_79 + 67*uk_8 + 212567617*uk_80 + 250047*uk_81 + 194481*uk_82 + 265923*uk_83 + 285768*uk_84 + 289737*uk_85 + 861273*uk_86 + 265923*uk_87 + 151263*uk_88 + 206829*uk_89 + 2242306609*uk_9 + 222264*uk_90 + 225351*uk_91 + 669879*uk_92 + 206829*uk_93 + 282807*uk_94 + 303912*uk_95 + 308133*uk_96 + 915957*uk_97 + 282807*uk_98 + 326592*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 321300*uk_100 + 929628*uk_101 + 209916*uk_102 + 354375*uk_103 + 1025325*uk_104 + 231525*uk_105 + 2966607*uk_106 + 669879*uk_107 + 151263*uk_108 + 21952*uk_109 + 1325884*uk_11 + 38416*uk_110 + 53312*uk_111 + 58800*uk_112 + 170128*uk_113 + 38416*uk_114 + 67228*uk_115 + 93296*uk_116 + 102900*uk_117 + 297724*uk_118 + 67228*uk_119 + 2320297*uk_12 + 129472*uk_120 + 142800*uk_121 + 413168*uk_122 + 93296*uk_123 + 157500*uk_124 + 455700*uk_125 + 102900*uk_126 + 1318492*uk_127 + 297724*uk_128 + 67228*uk_129 + 3220004*uk_13 + 117649*uk_130 + 163268*uk_131 + 180075*uk_132 + 521017*uk_133 + 117649*uk_134 + 226576*uk_135 + 249900*uk_136 + 723044*uk_137 + 163268*uk_138 + 275625*uk_139 + 3551475*uk_14 + 797475*uk_140 + 180075*uk_141 + 2307361*uk_142 + 521017*uk_143 + 117649*uk_144 + 314432*uk_145 + 346800*uk_146 + 1003408*uk_147 + 226576*uk_148 + 382500*uk_149 + 10275601*uk_15 + 1106700*uk_150 + 249900*uk_151 + 3202052*uk_152 + 723044*uk_153 + 163268*uk_154 + 421875*uk_155 + 1220625*uk_156 + 275625*uk_157 + 3531675*uk_158 + 797475*uk_159 + 2320297*uk_16 + 180075*uk_160 + 10218313*uk_161 + 2307361*uk_162 + 521017*uk_163 + 117649*uk_164 + 3969*uk_17 + 1764*uk_18 + 3087*uk_19 + 63*uk_2 + 4284*uk_20 + 4725*uk_21 + 13671*uk_22 + 3087*uk_23 + 784*uk_24 + 1372*uk_25 + 1904*uk_26 + 2100*uk_27 + 6076*uk_28 + 1372*uk_29 + 28*uk_3 + 2401*uk_30 + 3332*uk_31 + 3675*uk_32 + 10633*uk_33 + 2401*uk_34 + 4624*uk_35 + 5100*uk_36 + 14756*uk_37 + 3332*uk_38 + 5625*uk_39 + 49*uk_4 + 16275*uk_40 + 3675*uk_41 + 47089*uk_42 + 10633*uk_43 + 2401*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 62784585052*uk_47 + 109873023841*uk_48 + 152476849412*uk_49 + 68*uk_5 + 168172995675*uk_50 + 486580534153*uk_51 + 109873023841*uk_52 + 187944057*uk_53 + 83530692*uk_54 + 146178711*uk_55 + 202860252*uk_56 + 223742925*uk_57 + 647362863*uk_58 + 146178711*uk_59 + 75*uk_6 + 37124752*uk_60 + 64968316*uk_61 + 90160112*uk_62 + 99441300*uk_63 + 287716828*uk_64 + 64968316*uk_65 + 113694553*uk_66 + 157780196*uk_67 + 174022275*uk_68 + 503504449*uk_69 + 217*uk_7 + 113694553*uk_70 + 218960272*uk_71 + 241500300*uk_72 + 698740868*uk_73 + 157780196*uk_74 + 266360625*uk_75 + 770670075*uk_76 + 174022275*uk_77 + 2229805417*uk_78 + 503504449*uk_79 + 49*uk_8 + 113694553*uk_80 + 250047*uk_81 + 111132*uk_82 + 194481*uk_83 + 269892*uk_84 + 297675*uk_85 + 861273*uk_86 + 194481*uk_87 + 49392*uk_88 + 86436*uk_89 + 2242306609*uk_9 + 119952*uk_90 + 132300*uk_91 + 382788*uk_92 + 86436*uk_93 + 151263*uk_94 + 209916*uk_95 + 231525*uk_96 + 669879*uk_97 + 151263*uk_98 + 291312*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 329868*uk_100 + 929628*uk_101 + 119952*uk_102 + 373527*uk_103 + 1052667*uk_104 + 135828*uk_105 + 2966607*uk_106 + 382788*uk_107 + 49392*uk_108 + 421875*uk_109 + 3551475*uk_11 + 157500*uk_110 + 382500*uk_111 + 433125*uk_112 + 1220625*uk_113 + 157500*uk_114 + 58800*uk_115 + 142800*uk_116 + 161700*uk_117 + 455700*uk_118 + 58800*uk_119 + 1325884*uk_12 + 346800*uk_120 + 392700*uk_121 + 1106700*uk_122 + 142800*uk_123 + 444675*uk_124 + 1253175*uk_125 + 161700*uk_126 + 3531675*uk_127 + 455700*uk_128 + 58800*uk_129 + 3220004*uk_13 + 21952*uk_130 + 53312*uk_131 + 60368*uk_132 + 170128*uk_133 + 21952*uk_134 + 129472*uk_135 + 146608*uk_136 + 413168*uk_137 + 53312*uk_138 + 166012*uk_139 + 3646181*uk_14 + 467852*uk_140 + 60368*uk_141 + 1318492*uk_142 + 170128*uk_143 + 21952*uk_144 + 314432*uk_145 + 356048*uk_146 + 1003408*uk_147 + 129472*uk_148 + 403172*uk_149 + 10275601*uk_15 + 1136212*uk_150 + 146608*uk_151 + 3202052*uk_152 + 413168*uk_153 + 53312*uk_154 + 456533*uk_155 + 1286593*uk_156 + 166012*uk_157 + 3625853*uk_158 + 467852*uk_159 + 1325884*uk_16 + 60368*uk_160 + 10218313*uk_161 + 1318492*uk_162 + 170128*uk_163 + 21952*uk_164 + 3969*uk_17 + 4725*uk_18 + 1764*uk_19 + 63*uk_2 + 4284*uk_20 + 4851*uk_21 + 13671*uk_22 + 1764*uk_23 + 5625*uk_24 + 2100*uk_25 + 5100*uk_26 + 5775*uk_27 + 16275*uk_28 + 2100*uk_29 + 75*uk_3 + 784*uk_30 + 1904*uk_31 + 2156*uk_32 + 6076*uk_33 + 784*uk_34 + 4624*uk_35 + 5236*uk_36 + 14756*uk_37 + 1904*uk_38 + 5929*uk_39 + 28*uk_4 + 16709*uk_40 + 2156*uk_41 + 47089*uk_42 + 6076*uk_43 + 784*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 168172995675*uk_47 + 62784585052*uk_48 + 152476849412*uk_49 + 68*uk_5 + 172657608893*uk_50 + 486580534153*uk_51 + 62784585052*uk_52 + 187944057*uk_53 + 223742925*uk_54 + 83530692*uk_55 + 202860252*uk_56 + 229709403*uk_57 + 647362863*uk_58 + 83530692*uk_59 + 77*uk_6 + 266360625*uk_60 + 99441300*uk_61 + 241500300*uk_62 + 273463575*uk_63 + 770670075*uk_64 + 99441300*uk_65 + 37124752*uk_66 + 90160112*uk_67 + 102093068*uk_68 + 287716828*uk_69 + 217*uk_7 + 37124752*uk_70 + 218960272*uk_71 + 247940308*uk_72 + 698740868*uk_73 + 90160112*uk_74 + 280755937*uk_75 + 791221277*uk_76 + 102093068*uk_77 + 2229805417*uk_78 + 287716828*uk_79 + 28*uk_8 + 37124752*uk_80 + 250047*uk_81 + 297675*uk_82 + 111132*uk_83 + 269892*uk_84 + 305613*uk_85 + 861273*uk_86 + 111132*uk_87 + 354375*uk_88 + 132300*uk_89 + 2242306609*uk_9 + 321300*uk_90 + 363825*uk_91 + 1025325*uk_92 + 132300*uk_93 + 49392*uk_94 + 119952*uk_95 + 135828*uk_96 + 382788*uk_97 + 49392*uk_98 + 291312*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 298620*uk_100 + 820260*uk_101 + 283500*uk_102 + 393183*uk_103 + 1080009*uk_104 + 373275*uk_105 + 2966607*uk_106 + 1025325*uk_107 + 354375*uk_108 + 32768*uk_109 + 1515296*uk_11 + 76800*uk_110 + 61440*uk_111 + 80896*uk_112 + 222208*uk_113 + 76800*uk_114 + 180000*uk_115 + 144000*uk_116 + 189600*uk_117 + 520800*uk_118 + 180000*uk_119 + 3551475*uk_12 + 115200*uk_120 + 151680*uk_121 + 416640*uk_122 + 144000*uk_123 + 199712*uk_124 + 548576*uk_125 + 189600*uk_126 + 1506848*uk_127 + 520800*uk_128 + 180000*uk_129 + 2841180*uk_13 + 421875*uk_130 + 337500*uk_131 + 444375*uk_132 + 1220625*uk_133 + 421875*uk_134 + 270000*uk_135 + 355500*uk_136 + 976500*uk_137 + 337500*uk_138 + 468075*uk_139 + 3740887*uk_14 + 1285725*uk_140 + 444375*uk_141 + 3531675*uk_142 + 1220625*uk_143 + 421875*uk_144 + 216000*uk_145 + 284400*uk_146 + 781200*uk_147 + 270000*uk_148 + 374460*uk_149 + 10275601*uk_15 + 1028580*uk_150 + 355500*uk_151 + 2825340*uk_152 + 976500*uk_153 + 337500*uk_154 + 493039*uk_155 + 1354297*uk_156 + 468075*uk_157 + 3720031*uk_158 + 1285725*uk_159 + 3551475*uk_16 + 444375*uk_160 + 10218313*uk_161 + 3531675*uk_162 + 1220625*uk_163 + 421875*uk_164 + 3969*uk_17 + 2016*uk_18 + 4725*uk_19 + 63*uk_2 + 3780*uk_20 + 4977*uk_21 + 13671*uk_22 + 4725*uk_23 + 1024*uk_24 + 2400*uk_25 + 1920*uk_26 + 2528*uk_27 + 6944*uk_28 + 2400*uk_29 + 32*uk_3 + 5625*uk_30 + 4500*uk_31 + 5925*uk_32 + 16275*uk_33 + 5625*uk_34 + 3600*uk_35 + 4740*uk_36 + 13020*uk_37 + 4500*uk_38 + 6241*uk_39 + 75*uk_4 + 17143*uk_40 + 5925*uk_41 + 47089*uk_42 + 16275*uk_43 + 5625*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 71753811488*uk_47 + 168172995675*uk_48 + 134538396540*uk_49 + 60*uk_5 + 177142222111*uk_50 + 486580534153*uk_51 + 168172995675*uk_52 + 187944057*uk_53 + 95463648*uk_54 + 223742925*uk_55 + 178994340*uk_56 + 235675881*uk_57 + 647362863*uk_58 + 223742925*uk_59 + 79*uk_6 + 48489472*uk_60 + 113647200*uk_61 + 90917760*uk_62 + 119708384*uk_63 + 328819232*uk_64 + 113647200*uk_65 + 266360625*uk_66 + 213088500*uk_67 + 280566525*uk_68 + 770670075*uk_69 + 217*uk_7 + 266360625*uk_70 + 170470800*uk_71 + 224453220*uk_72 + 616536060*uk_73 + 213088500*uk_74 + 295530073*uk_75 + 811772479*uk_76 + 280566525*uk_77 + 2229805417*uk_78 + 770670075*uk_79 + 75*uk_8 + 266360625*uk_80 + 250047*uk_81 + 127008*uk_82 + 297675*uk_83 + 238140*uk_84 + 313551*uk_85 + 861273*uk_86 + 297675*uk_87 + 64512*uk_88 + 151200*uk_89 + 2242306609*uk_9 + 120960*uk_90 + 159264*uk_91 + 437472*uk_92 + 151200*uk_93 + 354375*uk_94 + 283500*uk_95 + 373275*uk_96 + 1025325*uk_97 + 354375*uk_98 + 226800*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 306180*uk_100 + 820260*uk_101 + 120960*uk_102 + 413343*uk_103 + 1107351*uk_104 + 163296*uk_105 + 2966607*uk_106 + 437472*uk_107 + 64512*uk_108 + 117649*uk_109 + 2320297*uk_11 + 76832*uk_110 + 144060*uk_111 + 194481*uk_112 + 521017*uk_113 + 76832*uk_114 + 50176*uk_115 + 94080*uk_116 + 127008*uk_117 + 340256*uk_118 + 50176*uk_119 + 1515296*uk_12 + 176400*uk_120 + 238140*uk_121 + 637980*uk_122 + 94080*uk_123 + 321489*uk_124 + 861273*uk_125 + 127008*uk_126 + 2307361*uk_127 + 340256*uk_128 + 50176*uk_129 + 2841180*uk_13 + 32768*uk_130 + 61440*uk_131 + 82944*uk_132 + 222208*uk_133 + 32768*uk_134 + 115200*uk_135 + 155520*uk_136 + 416640*uk_137 + 61440*uk_138 + 209952*uk_139 + 3835593*uk_14 + 562464*uk_140 + 82944*uk_141 + 1506848*uk_142 + 222208*uk_143 + 32768*uk_144 + 216000*uk_145 + 291600*uk_146 + 781200*uk_147 + 115200*uk_148 + 393660*uk_149 + 10275601*uk_15 + 1054620*uk_150 + 155520*uk_151 + 2825340*uk_152 + 416640*uk_153 + 61440*uk_154 + 531441*uk_155 + 1423737*uk_156 + 209952*uk_157 + 3814209*uk_158 + 562464*uk_159 + 1515296*uk_16 + 82944*uk_160 + 10218313*uk_161 + 1506848*uk_162 + 222208*uk_163 + 32768*uk_164 + 3969*uk_17 + 3087*uk_18 + 2016*uk_19 + 63*uk_2 + 3780*uk_20 + 5103*uk_21 + 13671*uk_22 + 2016*uk_23 + 2401*uk_24 + 1568*uk_25 + 2940*uk_26 + 3969*uk_27 + 10633*uk_28 + 1568*uk_29 + 49*uk_3 + 1024*uk_30 + 1920*uk_31 + 2592*uk_32 + 6944*uk_33 + 1024*uk_34 + 3600*uk_35 + 4860*uk_36 + 13020*uk_37 + 1920*uk_38 + 6561*uk_39 + 32*uk_4 + 17577*uk_40 + 2592*uk_41 + 47089*uk_42 + 6944*uk_43 + 1024*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 109873023841*uk_47 + 71753811488*uk_48 + 134538396540*uk_49 + 60*uk_5 + 181626835329*uk_50 + 486580534153*uk_51 + 71753811488*uk_52 + 187944057*uk_53 + 146178711*uk_54 + 95463648*uk_55 + 178994340*uk_56 + 241642359*uk_57 + 647362863*uk_58 + 95463648*uk_59 + 81*uk_6 + 113694553*uk_60 + 74249504*uk_61 + 139217820*uk_62 + 187944057*uk_63 + 503504449*uk_64 + 74249504*uk_65 + 48489472*uk_66 + 90917760*uk_67 + 122738976*uk_68 + 328819232*uk_69 + 217*uk_7 + 48489472*uk_70 + 170470800*uk_71 + 230135580*uk_72 + 616536060*uk_73 + 90917760*uk_74 + 310683033*uk_75 + 832323681*uk_76 + 122738976*uk_77 + 2229805417*uk_78 + 328819232*uk_79 + 32*uk_8 + 48489472*uk_80 + 250047*uk_81 + 194481*uk_82 + 127008*uk_83 + 238140*uk_84 + 321489*uk_85 + 861273*uk_86 + 127008*uk_87 + 151263*uk_88 + 98784*uk_89 + 2242306609*uk_9 + 185220*uk_90 + 250047*uk_91 + 669879*uk_92 + 98784*uk_93 + 64512*uk_94 + 120960*uk_95 + 163296*uk_96 + 437472*uk_97 + 64512*uk_98 + 226800*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 292824*uk_100 + 765576*uk_101 + 172872*uk_102 + 434007*uk_103 + 1134693*uk_104 + 256221*uk_105 + 2966607*uk_106 + 669879*uk_107 + 151263*uk_108 + 79507*uk_109 + 2036179*uk_11 + 90601*uk_110 + 103544*uk_111 + 153467*uk_112 + 401233*uk_113 + 90601*uk_114 + 103243*uk_115 + 117992*uk_116 + 174881*uk_117 + 457219*uk_118 + 103243*uk_119 + 2320297*uk_12 + 134848*uk_120 + 199864*uk_121 + 522536*uk_122 + 117992*uk_123 + 296227*uk_124 + 774473*uk_125 + 174881*uk_126 + 2024827*uk_127 + 457219*uk_128 + 103243*uk_129 + 2651768*uk_13 + 117649*uk_130 + 134456*uk_131 + 199283*uk_132 + 521017*uk_133 + 117649*uk_134 + 153664*uk_135 + 227752*uk_136 + 595448*uk_137 + 134456*uk_138 + 337561*uk_139 + 3930299*uk_14 + 882539*uk_140 + 199283*uk_141 + 2307361*uk_142 + 521017*uk_143 + 117649*uk_144 + 175616*uk_145 + 260288*uk_146 + 680512*uk_147 + 153664*uk_148 + 385784*uk_149 + 10275601*uk_15 + 1008616*uk_150 + 227752*uk_151 + 2636984*uk_152 + 595448*uk_153 + 134456*uk_154 + 571787*uk_155 + 1494913*uk_156 + 337561*uk_157 + 3908387*uk_158 + 882539*uk_159 + 2320297*uk_16 + 199283*uk_160 + 10218313*uk_161 + 2307361*uk_162 + 521017*uk_163 + 117649*uk_164 + 3969*uk_17 + 2709*uk_18 + 3087*uk_19 + 63*uk_2 + 3528*uk_20 + 5229*uk_21 + 13671*uk_22 + 3087*uk_23 + 1849*uk_24 + 2107*uk_25 + 2408*uk_26 + 3569*uk_27 + 9331*uk_28 + 2107*uk_29 + 43*uk_3 + 2401*uk_30 + 2744*uk_31 + 4067*uk_32 + 10633*uk_33 + 2401*uk_34 + 3136*uk_35 + 4648*uk_36 + 12152*uk_37 + 2744*uk_38 + 6889*uk_39 + 49*uk_4 + 18011*uk_40 + 4067*uk_41 + 47089*uk_42 + 10633*uk_43 + 2401*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 96419184187*uk_47 + 109873023841*uk_48 + 125569170104*uk_49 + 56*uk_5 + 186111448547*uk_50 + 486580534153*uk_51 + 109873023841*uk_52 + 187944057*uk_53 + 128279277*uk_54 + 146178711*uk_55 + 167061384*uk_56 + 247608837*uk_57 + 647362863*uk_58 + 146178711*uk_59 + 83*uk_6 + 87555697*uk_60 + 99772771*uk_61 + 114026024*uk_62 + 169002857*uk_63 + 441850843*uk_64 + 99772771*uk_65 + 113694553*uk_66 + 129936632*uk_67 + 192584651*uk_68 + 503504449*uk_69 + 217*uk_7 + 113694553*uk_70 + 148499008*uk_71 + 220096744*uk_72 + 575433656*uk_73 + 129936632*uk_74 + 326214817*uk_75 + 852874883*uk_76 + 192584651*uk_77 + 2229805417*uk_78 + 503504449*uk_79 + 49*uk_8 + 113694553*uk_80 + 250047*uk_81 + 170667*uk_82 + 194481*uk_83 + 222264*uk_84 + 329427*uk_85 + 861273*uk_86 + 194481*uk_87 + 116487*uk_88 + 132741*uk_89 + 2242306609*uk_9 + 151704*uk_90 + 224847*uk_91 + 587853*uk_92 + 132741*uk_93 + 151263*uk_94 + 172872*uk_95 + 256221*uk_96 + 669879*uk_97 + 151263*uk_98 + 197568*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 278460*uk_100 + 710892*uk_101 + 140868*uk_102 + 455175*uk_103 + 1162035*uk_104 + 230265*uk_105 + 2966607*uk_106 + 587853*uk_107 + 116487*uk_108 + 512*uk_109 + 378824*uk_11 + 2752*uk_110 + 3328*uk_111 + 5440*uk_112 + 13888*uk_113 + 2752*uk_114 + 14792*uk_115 + 17888*uk_116 + 29240*uk_117 + 74648*uk_118 + 14792*uk_119 + 2036179*uk_12 + 21632*uk_120 + 35360*uk_121 + 90272*uk_122 + 17888*uk_123 + 57800*uk_124 + 147560*uk_125 + 29240*uk_126 + 376712*uk_127 + 74648*uk_128 + 14792*uk_129 + 2462356*uk_13 + 79507*uk_130 + 96148*uk_131 + 157165*uk_132 + 401233*uk_133 + 79507*uk_134 + 116272*uk_135 + 190060*uk_136 + 485212*uk_137 + 96148*uk_138 + 310675*uk_139 + 4025005*uk_14 + 793135*uk_140 + 157165*uk_141 + 2024827*uk_142 + 401233*uk_143 + 79507*uk_144 + 140608*uk_145 + 229840*uk_146 + 586768*uk_147 + 116272*uk_148 + 375700*uk_149 + 10275601*uk_15 + 959140*uk_150 + 190060*uk_151 + 2448628*uk_152 + 485212*uk_153 + 96148*uk_154 + 614125*uk_155 + 1567825*uk_156 + 310675*uk_157 + 4002565*uk_158 + 793135*uk_159 + 2036179*uk_16 + 157165*uk_160 + 10218313*uk_161 + 2024827*uk_162 + 401233*uk_163 + 79507*uk_164 + 3969*uk_17 + 504*uk_18 + 2709*uk_19 + 63*uk_2 + 3276*uk_20 + 5355*uk_21 + 13671*uk_22 + 2709*uk_23 + 64*uk_24 + 344*uk_25 + 416*uk_26 + 680*uk_27 + 1736*uk_28 + 344*uk_29 + 8*uk_3 + 1849*uk_30 + 2236*uk_31 + 3655*uk_32 + 9331*uk_33 + 1849*uk_34 + 2704*uk_35 + 4420*uk_36 + 11284*uk_37 + 2236*uk_38 + 7225*uk_39 + 43*uk_4 + 18445*uk_40 + 3655*uk_41 + 47089*uk_42 + 9331*uk_43 + 1849*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 17938452872*uk_47 + 96419184187*uk_48 + 116599943668*uk_49 + 52*uk_5 + 190596061765*uk_50 + 486580534153*uk_51 + 96419184187*uk_52 + 187944057*uk_53 + 23865912*uk_54 + 128279277*uk_55 + 155128428*uk_56 + 253575315*uk_57 + 647362863*uk_58 + 128279277*uk_59 + 85*uk_6 + 3030592*uk_60 + 16289432*uk_61 + 19698848*uk_62 + 32200040*uk_63 + 82204808*uk_64 + 16289432*uk_65 + 87555697*uk_66 + 105881308*uk_67 + 173075215*uk_68 + 441850843*uk_69 + 217*uk_7 + 87555697*uk_70 + 128042512*uk_71 + 209300260*uk_72 + 534331252*uk_73 + 105881308*uk_74 + 342125425*uk_75 + 873426085*uk_76 + 173075215*uk_77 + 2229805417*uk_78 + 441850843*uk_79 + 43*uk_8 + 87555697*uk_80 + 250047*uk_81 + 31752*uk_82 + 170667*uk_83 + 206388*uk_84 + 337365*uk_85 + 861273*uk_86 + 170667*uk_87 + 4032*uk_88 + 21672*uk_89 + 2242306609*uk_9 + 26208*uk_90 + 42840*uk_91 + 109368*uk_92 + 21672*uk_93 + 116487*uk_94 + 140868*uk_95 + 230265*uk_96 + 587853*uk_97 + 116487*uk_98 + 170352*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 285012*uk_100 + 710892*uk_101 + 26208*uk_102 + 476847*uk_103 + 1189377*uk_104 + 43848*uk_105 + 2966607*uk_106 + 109368*uk_107 + 4032*uk_108 + 15625*uk_109 + 1183825*uk_11 + 5000*uk_110 + 32500*uk_111 + 54375*uk_112 + 135625*uk_113 + 5000*uk_114 + 1600*uk_115 + 10400*uk_116 + 17400*uk_117 + 43400*uk_118 + 1600*uk_119 + 378824*uk_12 + 67600*uk_120 + 113100*uk_121 + 282100*uk_122 + 10400*uk_123 + 189225*uk_124 + 471975*uk_125 + 17400*uk_126 + 1177225*uk_127 + 43400*uk_128 + 1600*uk_129 + 2462356*uk_13 + 512*uk_130 + 3328*uk_131 + 5568*uk_132 + 13888*uk_133 + 512*uk_134 + 21632*uk_135 + 36192*uk_136 + 90272*uk_137 + 3328*uk_138 + 60552*uk_139 + 4119711*uk_14 + 151032*uk_140 + 5568*uk_141 + 376712*uk_142 + 13888*uk_143 + 512*uk_144 + 140608*uk_145 + 235248*uk_146 + 586768*uk_147 + 21632*uk_148 + 393588*uk_149 + 10275601*uk_15 + 981708*uk_150 + 36192*uk_151 + 2448628*uk_152 + 90272*uk_153 + 3328*uk_154 + 658503*uk_155 + 1642473*uk_156 + 60552*uk_157 + 4096743*uk_158 + 151032*uk_159 + 378824*uk_16 + 5568*uk_160 + 10218313*uk_161 + 376712*uk_162 + 13888*uk_163 + 512*uk_164 + 3969*uk_17 + 1575*uk_18 + 504*uk_19 + 63*uk_2 + 3276*uk_20 + 5481*uk_21 + 13671*uk_22 + 504*uk_23 + 625*uk_24 + 200*uk_25 + 1300*uk_26 + 2175*uk_27 + 5425*uk_28 + 200*uk_29 + 25*uk_3 + 64*uk_30 + 416*uk_31 + 696*uk_32 + 1736*uk_33 + 64*uk_34 + 2704*uk_35 + 4524*uk_36 + 11284*uk_37 + 416*uk_38 + 7569*uk_39 + 8*uk_4 + 18879*uk_40 + 696*uk_41 + 47089*uk_42 + 1736*uk_43 + 64*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 56057665225*uk_47 + 17938452872*uk_48 + 116599943668*uk_49 + 52*uk_5 + 195080674983*uk_50 + 486580534153*uk_51 + 17938452872*uk_52 + 187944057*uk_53 + 74580975*uk_54 + 23865912*uk_55 + 155128428*uk_56 + 259541793*uk_57 + 647362863*uk_58 + 23865912*uk_59 + 87*uk_6 + 29595625*uk_60 + 9470600*uk_61 + 61558900*uk_62 + 102992775*uk_63 + 256890025*uk_64 + 9470600*uk_65 + 3030592*uk_66 + 19698848*uk_67 + 32957688*uk_68 + 82204808*uk_69 + 217*uk_7 + 3030592*uk_70 + 128042512*uk_71 + 214224972*uk_72 + 534331252*uk_73 + 19698848*uk_74 + 358414857*uk_75 + 893977287*uk_76 + 32957688*uk_77 + 2229805417*uk_78 + 82204808*uk_79 + 8*uk_8 + 3030592*uk_80 + 250047*uk_81 + 99225*uk_82 + 31752*uk_83 + 206388*uk_84 + 345303*uk_85 + 861273*uk_86 + 31752*uk_87 + 39375*uk_88 + 12600*uk_89 + 2242306609*uk_9 + 81900*uk_90 + 137025*uk_91 + 341775*uk_92 + 12600*uk_93 + 4032*uk_94 + 26208*uk_95 + 43848*uk_96 + 109368*uk_97 + 4032*uk_98 + 170352*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 269136*uk_100 + 656208*uk_101 + 75600*uk_102 + 499023*uk_103 + 1216719*uk_104 + 140175*uk_105 + 2966607*uk_106 + 341775*uk_107 + 39375*uk_108 + 125*uk_109 + 236765*uk_11 + 625*uk_110 + 1200*uk_111 + 2225*uk_112 + 5425*uk_113 + 625*uk_114 + 3125*uk_115 + 6000*uk_116 + 11125*uk_117 + 27125*uk_118 + 3125*uk_119 + 1183825*uk_12 + 11520*uk_120 + 21360*uk_121 + 52080*uk_122 + 6000*uk_123 + 39605*uk_124 + 96565*uk_125 + 11125*uk_126 + 235445*uk_127 + 27125*uk_128 + 3125*uk_129 + 2272944*uk_13 + 15625*uk_130 + 30000*uk_131 + 55625*uk_132 + 135625*uk_133 + 15625*uk_134 + 57600*uk_135 + 106800*uk_136 + 260400*uk_137 + 30000*uk_138 + 198025*uk_139 + 4214417*uk_14 + 482825*uk_140 + 55625*uk_141 + 1177225*uk_142 + 135625*uk_143 + 15625*uk_144 + 110592*uk_145 + 205056*uk_146 + 499968*uk_147 + 57600*uk_148 + 380208*uk_149 + 10275601*uk_15 + 927024*uk_150 + 106800*uk_151 + 2260272*uk_152 + 260400*uk_153 + 30000*uk_154 + 704969*uk_155 + 1718857*uk_156 + 198025*uk_157 + 4190921*uk_158 + 482825*uk_159 + 1183825*uk_16 + 55625*uk_160 + 10218313*uk_161 + 1177225*uk_162 + 135625*uk_163 + 15625*uk_164 + 3969*uk_17 + 315*uk_18 + 1575*uk_19 + 63*uk_2 + 3024*uk_20 + 5607*uk_21 + 13671*uk_22 + 1575*uk_23 + 25*uk_24 + 125*uk_25 + 240*uk_26 + 445*uk_27 + 1085*uk_28 + 125*uk_29 + 5*uk_3 + 625*uk_30 + 1200*uk_31 + 2225*uk_32 + 5425*uk_33 + 625*uk_34 + 2304*uk_35 + 4272*uk_36 + 10416*uk_37 + 1200*uk_38 + 7921*uk_39 + 25*uk_4 + 19313*uk_40 + 2225*uk_41 + 47089*uk_42 + 5425*uk_43 + 625*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 11211533045*uk_47 + 56057665225*uk_48 + 107630717232*uk_49 + 48*uk_5 + 199565288201*uk_50 + 486580534153*uk_51 + 56057665225*uk_52 + 187944057*uk_53 + 14916195*uk_54 + 74580975*uk_55 + 143195472*uk_56 + 265508271*uk_57 + 647362863*uk_58 + 74580975*uk_59 + 89*uk_6 + 1183825*uk_60 + 5919125*uk_61 + 11364720*uk_62 + 21072085*uk_63 + 51378005*uk_64 + 5919125*uk_65 + 29595625*uk_66 + 56823600*uk_67 + 105360425*uk_68 + 256890025*uk_69 + 217*uk_7 + 29595625*uk_70 + 109101312*uk_71 + 202292016*uk_72 + 493228848*uk_73 + 56823600*uk_74 + 375083113*uk_75 + 914528489*uk_76 + 105360425*uk_77 + 2229805417*uk_78 + 256890025*uk_79 + 25*uk_8 + 29595625*uk_80 + 250047*uk_81 + 19845*uk_82 + 99225*uk_83 + 190512*uk_84 + 353241*uk_85 + 861273*uk_86 + 99225*uk_87 + 1575*uk_88 + 7875*uk_89 + 2242306609*uk_9 + 15120*uk_90 + 28035*uk_91 + 68355*uk_92 + 7875*uk_93 + 39375*uk_94 + 75600*uk_95 + 140175*uk_96 + 341775*uk_97 + 39375*uk_98 + 145152*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 275184*uk_100 + 656208*uk_101 + 15120*uk_102 + 521703*uk_103 + 1244061*uk_104 + 28665*uk_105 + 2966607*uk_106 + 68355*uk_107 + 1575*uk_108 + 35937*uk_109 + 1562649*uk_11 + 5445*uk_110 + 52272*uk_111 + 99099*uk_112 + 236313*uk_113 + 5445*uk_114 + 825*uk_115 + 7920*uk_116 + 15015*uk_117 + 35805*uk_118 + 825*uk_119 + 236765*uk_12 + 76032*uk_120 + 144144*uk_121 + 343728*uk_122 + 7920*uk_123 + 273273*uk_124 + 651651*uk_125 + 15015*uk_126 + 1553937*uk_127 + 35805*uk_128 + 825*uk_129 + 2272944*uk_13 + 125*uk_130 + 1200*uk_131 + 2275*uk_132 + 5425*uk_133 + 125*uk_134 + 11520*uk_135 + 21840*uk_136 + 52080*uk_137 + 1200*uk_138 + 41405*uk_139 + 4309123*uk_14 + 98735*uk_140 + 2275*uk_141 + 235445*uk_142 + 5425*uk_143 + 125*uk_144 + 110592*uk_145 + 209664*uk_146 + 499968*uk_147 + 11520*uk_148 + 397488*uk_149 + 10275601*uk_15 + 947856*uk_150 + 21840*uk_151 + 2260272*uk_152 + 52080*uk_153 + 1200*uk_154 + 753571*uk_155 + 1796977*uk_156 + 41405*uk_157 + 4285099*uk_158 + 98735*uk_159 + 236765*uk_16 + 2275*uk_160 + 10218313*uk_161 + 235445*uk_162 + 5425*uk_163 + 125*uk_164 + 3969*uk_17 + 2079*uk_18 + 315*uk_19 + 63*uk_2 + 3024*uk_20 + 5733*uk_21 + 13671*uk_22 + 315*uk_23 + 1089*uk_24 + 165*uk_25 + 1584*uk_26 + 3003*uk_27 + 7161*uk_28 + 165*uk_29 + 33*uk_3 + 25*uk_30 + 240*uk_31 + 455*uk_32 + 1085*uk_33 + 25*uk_34 + 2304*uk_35 + 4368*uk_36 + 10416*uk_37 + 240*uk_38 + 8281*uk_39 + 5*uk_4 + 19747*uk_40 + 455*uk_41 + 47089*uk_42 + 1085*uk_43 + 25*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 73996118097*uk_47 + 11211533045*uk_48 + 107630717232*uk_49 + 48*uk_5 + 204049901419*uk_50 + 486580534153*uk_51 + 11211533045*uk_52 + 187944057*uk_53 + 98446887*uk_54 + 14916195*uk_55 + 143195472*uk_56 + 271474749*uk_57 + 647362863*uk_58 + 14916195*uk_59 + 91*uk_6 + 51567417*uk_60 + 7813245*uk_61 + 75007152*uk_62 + 142201059*uk_63 + 339094833*uk_64 + 7813245*uk_65 + 1183825*uk_66 + 11364720*uk_67 + 21545615*uk_68 + 51378005*uk_69 + 217*uk_7 + 1183825*uk_70 + 109101312*uk_71 + 206837904*uk_72 + 493228848*uk_73 + 11364720*uk_74 + 392130193*uk_75 + 935079691*uk_76 + 21545615*uk_77 + 2229805417*uk_78 + 51378005*uk_79 + 5*uk_8 + 1183825*uk_80 + 250047*uk_81 + 130977*uk_82 + 19845*uk_83 + 190512*uk_84 + 361179*uk_85 + 861273*uk_86 + 19845*uk_87 + 68607*uk_88 + 10395*uk_89 + 2242306609*uk_9 + 99792*uk_90 + 189189*uk_91 + 451143*uk_92 + 10395*uk_93 + 1575*uk_94 + 15120*uk_95 + 28665*uk_96 + 68355*uk_97 + 1575*uk_98 + 145152*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 257796*uk_100 + 601524*uk_101 + 91476*uk_102 + 544887*uk_103 + 1271403*uk_104 + 193347*uk_105 + 2966607*uk_106 + 451143*uk_107 + 68607*uk_108 + 4096*uk_109 + 757648*uk_11 + 8448*uk_110 + 11264*uk_111 + 23808*uk_112 + 55552*uk_113 + 8448*uk_114 + 17424*uk_115 + 23232*uk_116 + 49104*uk_117 + 114576*uk_118 + 17424*uk_119 + 1562649*uk_12 + 30976*uk_120 + 65472*uk_121 + 152768*uk_122 + 23232*uk_123 + 138384*uk_124 + 322896*uk_125 + 49104*uk_126 + 753424*uk_127 + 114576*uk_128 + 17424*uk_129 + 2083532*uk_13 + 35937*uk_130 + 47916*uk_131 + 101277*uk_132 + 236313*uk_133 + 35937*uk_134 + 63888*uk_135 + 135036*uk_136 + 315084*uk_137 + 47916*uk_138 + 285417*uk_139 + 4403829*uk_14 + 665973*uk_140 + 101277*uk_141 + 1553937*uk_142 + 236313*uk_143 + 35937*uk_144 + 85184*uk_145 + 180048*uk_146 + 420112*uk_147 + 63888*uk_148 + 380556*uk_149 + 10275601*uk_15 + 887964*uk_150 + 135036*uk_151 + 2071916*uk_152 + 315084*uk_153 + 47916*uk_154 + 804357*uk_155 + 1876833*uk_156 + 285417*uk_157 + 4379277*uk_158 + 665973*uk_159 + 1562649*uk_16 + 101277*uk_160 + 10218313*uk_161 + 1553937*uk_162 + 236313*uk_163 + 35937*uk_164 + 3969*uk_17 + 1008*uk_18 + 2079*uk_19 + 63*uk_2 + 2772*uk_20 + 5859*uk_21 + 13671*uk_22 + 2079*uk_23 + 256*uk_24 + 528*uk_25 + 704*uk_26 + 1488*uk_27 + 3472*uk_28 + 528*uk_29 + 16*uk_3 + 1089*uk_30 + 1452*uk_31 + 3069*uk_32 + 7161*uk_33 + 1089*uk_34 + 1936*uk_35 + 4092*uk_36 + 9548*uk_37 + 1452*uk_38 + 8649*uk_39 + 33*uk_4 + 20181*uk_40 + 3069*uk_41 + 47089*uk_42 + 7161*uk_43 + 1089*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 35876905744*uk_47 + 73996118097*uk_48 + 98661490796*uk_49 + 44*uk_5 + 208534514637*uk_50 + 486580534153*uk_51 + 73996118097*uk_52 + 187944057*uk_53 + 47731824*uk_54 + 98446887*uk_55 + 131262516*uk_56 + 277441227*uk_57 + 647362863*uk_58 + 98446887*uk_59 + 93*uk_6 + 12122368*uk_60 + 25002384*uk_61 + 33336512*uk_62 + 70461264*uk_63 + 164409616*uk_64 + 25002384*uk_65 + 51567417*uk_66 + 68756556*uk_67 + 145326357*uk_68 + 339094833*uk_69 + 217*uk_7 + 51567417*uk_70 + 91675408*uk_71 + 193768476*uk_72 + 452126444*uk_73 + 68756556*uk_74 + 409556097*uk_75 + 955630893*uk_76 + 145326357*uk_77 + 2229805417*uk_78 + 339094833*uk_79 + 33*uk_8 + 51567417*uk_80 + 250047*uk_81 + 63504*uk_82 + 130977*uk_83 + 174636*uk_84 + 369117*uk_85 + 861273*uk_86 + 130977*uk_87 + 16128*uk_88 + 33264*uk_89 + 2242306609*uk_9 + 44352*uk_90 + 93744*uk_91 + 218736*uk_92 + 33264*uk_93 + 68607*uk_94 + 91476*uk_95 + 193347*uk_96 + 451143*uk_97 + 68607*uk_98 + 121968*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 263340*uk_100 + 601524*uk_101 + 44352*uk_102 + 568575*uk_103 + 1298745*uk_104 + 95760*uk_105 + 2966607*uk_106 + 218736*uk_107 + 16128*uk_108 + 79507*uk_109 + 2036179*uk_11 + 29584*uk_110 + 81356*uk_111 + 175655*uk_112 + 401233*uk_113 + 29584*uk_114 + 11008*uk_115 + 30272*uk_116 + 65360*uk_117 + 149296*uk_118 + 11008*uk_119 + 757648*uk_12 + 83248*uk_120 + 179740*uk_121 + 410564*uk_122 + 30272*uk_123 + 388075*uk_124 + 886445*uk_125 + 65360*uk_126 + 2024827*uk_127 + 149296*uk_128 + 11008*uk_129 + 2083532*uk_13 + 4096*uk_130 + 11264*uk_131 + 24320*uk_132 + 55552*uk_133 + 4096*uk_134 + 30976*uk_135 + 66880*uk_136 + 152768*uk_137 + 11264*uk_138 + 144400*uk_139 + 4498535*uk_14 + 329840*uk_140 + 24320*uk_141 + 753424*uk_142 + 55552*uk_143 + 4096*uk_144 + 85184*uk_145 + 183920*uk_146 + 420112*uk_147 + 30976*uk_148 + 397100*uk_149 + 10275601*uk_15 + 907060*uk_150 + 66880*uk_151 + 2071916*uk_152 + 152768*uk_153 + 11264*uk_154 + 857375*uk_155 + 1958425*uk_156 + 144400*uk_157 + 4473455*uk_158 + 329840*uk_159 + 757648*uk_16 + 24320*uk_160 + 10218313*uk_161 + 753424*uk_162 + 55552*uk_163 + 4096*uk_164 + 3969*uk_17 + 2709*uk_18 + 1008*uk_19 + 63*uk_2 + 2772*uk_20 + 5985*uk_21 + 13671*uk_22 + 1008*uk_23 + 1849*uk_24 + 688*uk_25 + 1892*uk_26 + 4085*uk_27 + 9331*uk_28 + 688*uk_29 + 43*uk_3 + 256*uk_30 + 704*uk_31 + 1520*uk_32 + 3472*uk_33 + 256*uk_34 + 1936*uk_35 + 4180*uk_36 + 9548*uk_37 + 704*uk_38 + 9025*uk_39 + 16*uk_4 + 20615*uk_40 + 1520*uk_41 + 47089*uk_42 + 3472*uk_43 + 256*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 96419184187*uk_47 + 35876905744*uk_48 + 98661490796*uk_49 + 44*uk_5 + 213019127855*uk_50 + 486580534153*uk_51 + 35876905744*uk_52 + 187944057*uk_53 + 128279277*uk_54 + 47731824*uk_55 + 131262516*uk_56 + 283407705*uk_57 + 647362863*uk_58 + 47731824*uk_59 + 95*uk_6 + 87555697*uk_60 + 32578864*uk_61 + 89591876*uk_62 + 193437005*uk_63 + 441850843*uk_64 + 32578864*uk_65 + 12122368*uk_66 + 33336512*uk_67 + 71976560*uk_68 + 164409616*uk_69 + 217*uk_7 + 12122368*uk_70 + 91675408*uk_71 + 197935540*uk_72 + 452126444*uk_73 + 33336512*uk_74 + 427360825*uk_75 + 976182095*uk_76 + 71976560*uk_77 + 2229805417*uk_78 + 164409616*uk_79 + 16*uk_8 + 12122368*uk_80 + 250047*uk_81 + 170667*uk_82 + 63504*uk_83 + 174636*uk_84 + 377055*uk_85 + 861273*uk_86 + 63504*uk_87 + 116487*uk_88 + 43344*uk_89 + 2242306609*uk_9 + 119196*uk_90 + 257355*uk_91 + 587853*uk_92 + 43344*uk_93 + 16128*uk_94 + 44352*uk_95 + 95760*uk_96 + 218736*uk_97 + 16128*uk_98 + 121968*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 244440*uk_100 + 546840*uk_101 + 108360*uk_102 + 592767*uk_103 + 1326087*uk_104 + 262773*uk_105 + 2966607*uk_106 + 587853*uk_107 + 116487*uk_108 + 4913*uk_109 + 805001*uk_11 + 12427*uk_110 + 11560*uk_111 + 28033*uk_112 + 62713*uk_113 + 12427*uk_114 + 31433*uk_115 + 29240*uk_116 + 70907*uk_117 + 158627*uk_118 + 31433*uk_119 + 2036179*uk_12 + 27200*uk_120 + 65960*uk_121 + 147560*uk_122 + 29240*uk_123 + 159953*uk_124 + 357833*uk_125 + 70907*uk_126 + 800513*uk_127 + 158627*uk_128 + 31433*uk_129 + 1894120*uk_13 + 79507*uk_130 + 73960*uk_131 + 179353*uk_132 + 401233*uk_133 + 79507*uk_134 + 68800*uk_135 + 166840*uk_136 + 373240*uk_137 + 73960*uk_138 + 404587*uk_139 + 4593241*uk_14 + 905107*uk_140 + 179353*uk_141 + 2024827*uk_142 + 401233*uk_143 + 79507*uk_144 + 64000*uk_145 + 155200*uk_146 + 347200*uk_147 + 68800*uk_148 + 376360*uk_149 + 10275601*uk_15 + 841960*uk_150 + 166840*uk_151 + 1883560*uk_152 + 373240*uk_153 + 73960*uk_154 + 912673*uk_155 + 2041753*uk_156 + 404587*uk_157 + 4567633*uk_158 + 905107*uk_159 + 2036179*uk_16 + 179353*uk_160 + 10218313*uk_161 + 2024827*uk_162 + 401233*uk_163 + 79507*uk_164 + 3969*uk_17 + 1071*uk_18 + 2709*uk_19 + 63*uk_2 + 2520*uk_20 + 6111*uk_21 + 13671*uk_22 + 2709*uk_23 + 289*uk_24 + 731*uk_25 + 680*uk_26 + 1649*uk_27 + 3689*uk_28 + 731*uk_29 + 17*uk_3 + 1849*uk_30 + 1720*uk_31 + 4171*uk_32 + 9331*uk_33 + 1849*uk_34 + 1600*uk_35 + 3880*uk_36 + 8680*uk_37 + 1720*uk_38 + 9409*uk_39 + 43*uk_4 + 21049*uk_40 + 4171*uk_41 + 47089*uk_42 + 9331*uk_43 + 1849*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 38119212353*uk_47 + 96419184187*uk_48 + 89692264360*uk_49 + 40*uk_5 + 217503741073*uk_50 + 486580534153*uk_51 + 96419184187*uk_52 + 187944057*uk_53 + 50715063*uk_54 + 128279277*uk_55 + 119329560*uk_56 + 289374183*uk_57 + 647362863*uk_58 + 128279277*uk_59 + 97*uk_6 + 13685017*uk_60 + 34615043*uk_61 + 32200040*uk_62 + 78085097*uk_63 + 174685217*uk_64 + 34615043*uk_65 + 87555697*uk_66 + 81447160*uk_67 + 197509363*uk_68 + 441850843*uk_69 + 217*uk_7 + 87555697*uk_70 + 75764800*uk_71 + 183729640*uk_72 + 411024040*uk_73 + 81447160*uk_74 + 445544377*uk_75 + 996733297*uk_76 + 197509363*uk_77 + 2229805417*uk_78 + 441850843*uk_79 + 43*uk_8 + 87555697*uk_80 + 250047*uk_81 + 67473*uk_82 + 170667*uk_83 + 158760*uk_84 + 384993*uk_85 + 861273*uk_86 + 170667*uk_87 + 18207*uk_88 + 46053*uk_89 + 2242306609*uk_9 + 42840*uk_90 + 103887*uk_91 + 232407*uk_92 + 46053*uk_93 + 116487*uk_94 + 108360*uk_95 + 262773*uk_96 + 587853*uk_97 + 116487*uk_98 + 100800*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 249480*uk_100 + 546840*uk_101 + 42840*uk_102 + 617463*uk_103 + 1353429*uk_104 + 106029*uk_105 + 2966607*uk_106 + 232407*uk_107 + 18207*uk_108 + 29791*uk_109 + 1467943*uk_11 + 16337*uk_110 + 38440*uk_111 + 95139*uk_112 + 208537*uk_113 + 16337*uk_114 + 8959*uk_115 + 21080*uk_116 + 52173*uk_117 + 114359*uk_118 + 8959*uk_119 + 805001*uk_12 + 49600*uk_120 + 122760*uk_121 + 269080*uk_122 + 21080*uk_123 + 303831*uk_124 + 665973*uk_125 + 52173*uk_126 + 1459759*uk_127 + 114359*uk_128 + 8959*uk_129 + 1894120*uk_13 + 4913*uk_130 + 11560*uk_131 + 28611*uk_132 + 62713*uk_133 + 4913*uk_134 + 27200*uk_135 + 67320*uk_136 + 147560*uk_137 + 11560*uk_138 + 166617*uk_139 + 4687947*uk_14 + 365211*uk_140 + 28611*uk_141 + 800513*uk_142 + 62713*uk_143 + 4913*uk_144 + 64000*uk_145 + 158400*uk_146 + 347200*uk_147 + 27200*uk_148 + 392040*uk_149 + 10275601*uk_15 + 859320*uk_150 + 67320*uk_151 + 1883560*uk_152 + 147560*uk_153 + 11560*uk_154 + 970299*uk_155 + 2126817*uk_156 + 166617*uk_157 + 4661811*uk_158 + 365211*uk_159 + 805001*uk_16 + 28611*uk_160 + 10218313*uk_161 + 800513*uk_162 + 62713*uk_163 + 4913*uk_164 + 3969*uk_17 + 1953*uk_18 + 1071*uk_19 + 63*uk_2 + 2520*uk_20 + 6237*uk_21 + 13671*uk_22 + 1071*uk_23 + 961*uk_24 + 527*uk_25 + 1240*uk_26 + 3069*uk_27 + 6727*uk_28 + 527*uk_29 + 31*uk_3 + 289*uk_30 + 680*uk_31 + 1683*uk_32 + 3689*uk_33 + 289*uk_34 + 1600*uk_35 + 3960*uk_36 + 8680*uk_37 + 680*uk_38 + 9801*uk_39 + 17*uk_4 + 21483*uk_40 + 1683*uk_41 + 47089*uk_42 + 3689*uk_43 + 289*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 69511504879*uk_47 + 38119212353*uk_48 + 89692264360*uk_49 + 40*uk_5 + 221988354291*uk_50 + 486580534153*uk_51 + 38119212353*uk_52 + 187944057*uk_53 + 92480409*uk_54 + 50715063*uk_55 + 119329560*uk_56 + 295340661*uk_57 + 647362863*uk_58 + 50715063*uk_59 + 99*uk_6 + 45506233*uk_60 + 24955031*uk_61 + 58717720*uk_62 + 145326357*uk_63 + 318543631*uk_64 + 24955031*uk_65 + 13685017*uk_66 + 32200040*uk_67 + 79695099*uk_68 + 174685217*uk_69 + 217*uk_7 + 13685017*uk_70 + 75764800*uk_71 + 187517880*uk_72 + 411024040*uk_73 + 32200040*uk_74 + 464106753*uk_75 + 1017284499*uk_76 + 79695099*uk_77 + 2229805417*uk_78 + 174685217*uk_79 + 17*uk_8 + 13685017*uk_80 + 250047*uk_81 + 123039*uk_82 + 67473*uk_83 + 158760*uk_84 + 392931*uk_85 + 861273*uk_86 + 67473*uk_87 + 60543*uk_88 + 33201*uk_89 + 2242306609*uk_9 + 78120*uk_90 + 193347*uk_91 + 423801*uk_92 + 33201*uk_93 + 18207*uk_94 + 42840*uk_95 + 106029*uk_96 + 232407*uk_97 + 18207*uk_98 + 100800*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 254520*uk_100 + 546840*uk_101 + 78120*uk_102 + 642663*uk_103 + 1380771*uk_104 + 197253*uk_105 + 2966607*uk_106 + 423801*uk_107 + 60543*uk_108 + 614125*uk_109 + 4025005*uk_11 + 223975*uk_110 + 289000*uk_111 + 729725*uk_112 + 1567825*uk_113 + 223975*uk_114 + 81685*uk_115 + 105400*uk_116 + 266135*uk_117 + 571795*uk_118 + 81685*uk_119 + 1467943*uk_12 + 136000*uk_120 + 343400*uk_121 + 737800*uk_122 + 105400*uk_123 + 867085*uk_124 + 1862945*uk_125 + 266135*uk_126 + 4002565*uk_127 + 571795*uk_128 + 81685*uk_129 + 1894120*uk_13 + 29791*uk_130 + 38440*uk_131 + 97061*uk_132 + 208537*uk_133 + 29791*uk_134 + 49600*uk_135 + 125240*uk_136 + 269080*uk_137 + 38440*uk_138 + 316231*uk_139 + 4782653*uk_14 + 679427*uk_140 + 97061*uk_141 + 1459759*uk_142 + 208537*uk_143 + 29791*uk_144 + 64000*uk_145 + 161600*uk_146 + 347200*uk_147 + 49600*uk_148 + 408040*uk_149 + 10275601*uk_15 + 876680*uk_150 + 125240*uk_151 + 1883560*uk_152 + 269080*uk_153 + 38440*uk_154 + 1030301*uk_155 + 2213617*uk_156 + 316231*uk_157 + 4755989*uk_158 + 679427*uk_159 + 1467943*uk_16 + 97061*uk_160 + 10218313*uk_161 + 1459759*uk_162 + 208537*uk_163 + 29791*uk_164 + 3969*uk_17 + 5355*uk_18 + 1953*uk_19 + 63*uk_2 + 2520*uk_20 + 6363*uk_21 + 13671*uk_22 + 1953*uk_23 + 7225*uk_24 + 2635*uk_25 + 3400*uk_26 + 8585*uk_27 + 18445*uk_28 + 2635*uk_29 + 85*uk_3 + 961*uk_30 + 1240*uk_31 + 3131*uk_32 + 6727*uk_33 + 961*uk_34 + 1600*uk_35 + 4040*uk_36 + 8680*uk_37 + 1240*uk_38 + 10201*uk_39 + 31*uk_4 + 21917*uk_40 + 3131*uk_41 + 47089*uk_42 + 6727*uk_43 + 961*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 190596061765*uk_47 + 69511504879*uk_48 + 89692264360*uk_49 + 40*uk_5 + 226472967509*uk_50 + 486580534153*uk_51 + 69511504879*uk_52 + 187944057*uk_53 + 253575315*uk_54 + 92480409*uk_55 + 119329560*uk_56 + 301307139*uk_57 + 647362863*uk_58 + 92480409*uk_59 + 101*uk_6 + 342125425*uk_60 + 124775155*uk_61 + 161000200*uk_62 + 406525505*uk_63 + 873426085*uk_64 + 124775155*uk_65 + 45506233*uk_66 + 58717720*uk_67 + 148262243*uk_68 + 318543631*uk_69 + 217*uk_7 + 45506233*uk_70 + 75764800*uk_71 + 191306120*uk_72 + 411024040*uk_73 + 58717720*uk_74 + 483047953*uk_75 + 1037835701*uk_76 + 148262243*uk_77 + 2229805417*uk_78 + 318543631*uk_79 + 31*uk_8 + 45506233*uk_80 + 250047*uk_81 + 337365*uk_82 + 123039*uk_83 + 158760*uk_84 + 400869*uk_85 + 861273*uk_86 + 123039*uk_87 + 455175*uk_88 + 166005*uk_89 + 2242306609*uk_9 + 214200*uk_90 + 540855*uk_91 + 1162035*uk_92 + 166005*uk_93 + 60543*uk_94 + 78120*uk_95 + 197253*uk_96 + 423801*uk_97 + 60543*uk_98 + 100800*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 233604*uk_100 + 492156*uk_101 + 192780*uk_102 + 668367*uk_103 + 1408113*uk_104 + 551565*uk_105 + 2966607*uk_106 + 1162035*uk_107 + 455175*uk_108 + 438976*uk_109 + 3598828*uk_11 + 490960*uk_110 + 207936*uk_111 + 594928*uk_112 + 1253392*uk_113 + 490960*uk_114 + 549100*uk_115 + 232560*uk_116 + 665380*uk_117 + 1401820*uk_118 + 549100*uk_119 + 4025005*uk_12 + 98496*uk_120 + 281808*uk_121 + 593712*uk_122 + 232560*uk_123 + 806284*uk_124 + 1698676*uk_125 + 665380*uk_126 + 3578764*uk_127 + 1401820*uk_128 + 549100*uk_129 + 1704708*uk_13 + 614125*uk_130 + 260100*uk_131 + 744175*uk_132 + 1567825*uk_133 + 614125*uk_134 + 110160*uk_135 + 315180*uk_136 + 664020*uk_137 + 260100*uk_138 + 901765*uk_139 + 4877359*uk_14 + 1899835*uk_140 + 744175*uk_141 + 4002565*uk_142 + 1567825*uk_143 + 614125*uk_144 + 46656*uk_145 + 133488*uk_146 + 281232*uk_147 + 110160*uk_148 + 381924*uk_149 + 10275601*uk_15 + 804636*uk_150 + 315180*uk_151 + 1695204*uk_152 + 664020*uk_153 + 260100*uk_154 + 1092727*uk_155 + 2302153*uk_156 + 901765*uk_157 + 4850167*uk_158 + 1899835*uk_159 + 4025005*uk_16 + 744175*uk_160 + 10218313*uk_161 + 4002565*uk_162 + 1567825*uk_163 + 614125*uk_164 + 3969*uk_17 + 4788*uk_18 + 5355*uk_19 + 63*uk_2 + 2268*uk_20 + 6489*uk_21 + 13671*uk_22 + 5355*uk_23 + 5776*uk_24 + 6460*uk_25 + 2736*uk_26 + 7828*uk_27 + 16492*uk_28 + 6460*uk_29 + 76*uk_3 + 7225*uk_30 + 3060*uk_31 + 8755*uk_32 + 18445*uk_33 + 7225*uk_34 + 1296*uk_35 + 3708*uk_36 + 7812*uk_37 + 3060*uk_38 + 10609*uk_39 + 85*uk_4 + 22351*uk_40 + 8755*uk_41 + 47089*uk_42 + 18445*uk_43 + 7225*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 170415302284*uk_47 + 190596061765*uk_48 + 80723037924*uk_49 + 36*uk_5 + 230957580727*uk_50 + 486580534153*uk_51 + 190596061765*uk_52 + 187944057*uk_53 + 226726164*uk_54 + 253575315*uk_55 + 107396604*uk_56 + 307273617*uk_57 + 647362863*uk_58 + 253575315*uk_59 + 103*uk_6 + 273510928*uk_60 + 305900380*uk_61 + 129557808*uk_62 + 370679284*uk_63 + 780945676*uk_64 + 305900380*uk_65 + 342125425*uk_66 + 144900180*uk_67 + 414575515*uk_68 + 873426085*uk_69 + 217*uk_7 + 342125425*uk_70 + 61369488*uk_71 + 175584924*uk_72 + 369921636*uk_73 + 144900180*uk_74 + 502367977*uk_75 + 1058386903*uk_76 + 414575515*uk_77 + 2229805417*uk_78 + 873426085*uk_79 + 85*uk_8 + 342125425*uk_80 + 250047*uk_81 + 301644*uk_82 + 337365*uk_83 + 142884*uk_84 + 408807*uk_85 + 861273*uk_86 + 337365*uk_87 + 363888*uk_88 + 406980*uk_89 + 2242306609*uk_9 + 172368*uk_90 + 493164*uk_91 + 1038996*uk_92 + 406980*uk_93 + 455175*uk_94 + 192780*uk_95 + 551565*uk_96 + 1162035*uk_97 + 455175*uk_98 + 81648*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 238140*uk_100 + 492156*uk_101 + 172368*uk_102 + 694575*uk_103 + 1435455*uk_104 + 502740*uk_105 + 2966607*uk_106 + 1038996*uk_107 + 363888*uk_108 + 1092727*uk_109 + 4877359*uk_11 + 806284*uk_110 + 381924*uk_111 + 1113945*uk_112 + 2302153*uk_113 + 806284*uk_114 + 594928*uk_115 + 281808*uk_116 + 821940*uk_117 + 1698676*uk_118 + 594928*uk_119 + 3598828*uk_12 + 133488*uk_120 + 389340*uk_121 + 804636*uk_122 + 281808*uk_123 + 1135575*uk_124 + 2346855*uk_125 + 821940*uk_126 + 4850167*uk_127 + 1698676*uk_128 + 594928*uk_129 + 1704708*uk_13 + 438976*uk_130 + 207936*uk_131 + 606480*uk_132 + 1253392*uk_133 + 438976*uk_134 + 98496*uk_135 + 287280*uk_136 + 593712*uk_137 + 207936*uk_138 + 837900*uk_139 + 4972065*uk_14 + 1731660*uk_140 + 606480*uk_141 + 3578764*uk_142 + 1253392*uk_143 + 438976*uk_144 + 46656*uk_145 + 136080*uk_146 + 281232*uk_147 + 98496*uk_148 + 396900*uk_149 + 10275601*uk_15 + 820260*uk_150 + 287280*uk_151 + 1695204*uk_152 + 593712*uk_153 + 207936*uk_154 + 1157625*uk_155 + 2392425*uk_156 + 837900*uk_157 + 4944345*uk_158 + 1731660*uk_159 + 3598828*uk_16 + 606480*uk_160 + 10218313*uk_161 + 3578764*uk_162 + 1253392*uk_163 + 438976*uk_164 + 3969*uk_17 + 6489*uk_18 + 4788*uk_19 + 63*uk_2 + 2268*uk_20 + 6615*uk_21 + 13671*uk_22 + 4788*uk_23 + 10609*uk_24 + 7828*uk_25 + 3708*uk_26 + 10815*uk_27 + 22351*uk_28 + 7828*uk_29 + 103*uk_3 + 5776*uk_30 + 2736*uk_31 + 7980*uk_32 + 16492*uk_33 + 5776*uk_34 + 1296*uk_35 + 3780*uk_36 + 7812*uk_37 + 2736*uk_38 + 11025*uk_39 + 76*uk_4 + 22785*uk_40 + 7980*uk_41 + 47089*uk_42 + 16492*uk_43 + 5776*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 230957580727*uk_47 + 170415302284*uk_48 + 80723037924*uk_49 + 36*uk_5 + 235442193945*uk_50 + 486580534153*uk_51 + 170415302284*uk_52 + 187944057*uk_53 + 307273617*uk_54 + 226726164*uk_55 + 107396604*uk_56 + 313240095*uk_57 + 647362863*uk_58 + 226726164*uk_59 + 105*uk_6 + 502367977*uk_60 + 370679284*uk_61 + 175584924*uk_62 + 512122695*uk_63 + 1058386903*uk_64 + 370679284*uk_65 + 273510928*uk_66 + 129557808*uk_67 + 377876940*uk_68 + 780945676*uk_69 + 217*uk_7 + 273510928*uk_70 + 61369488*uk_71 + 178994340*uk_72 + 369921636*uk_73 + 129557808*uk_74 + 522066825*uk_75 + 1078938105*uk_76 + 377876940*uk_77 + 2229805417*uk_78 + 780945676*uk_79 + 76*uk_8 + 273510928*uk_80 + 250047*uk_81 + 408807*uk_82 + 301644*uk_83 + 142884*uk_84 + 416745*uk_85 + 861273*uk_86 + 301644*uk_87 + 668367*uk_88 + 493164*uk_89 + 2242306609*uk_9 + 233604*uk_90 + 681345*uk_91 + 1408113*uk_92 + 493164*uk_93 + 363888*uk_94 + 172368*uk_95 + 502740*uk_96 + 1038996*uk_97 + 363888*uk_98 + 81648*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 215712*uk_100 + 437472*uk_101 + 207648*uk_102 + 721287*uk_103 + 1462797*uk_104 + 694323*uk_105 + 2966607*uk_106 + 1408113*uk_107 + 668367*uk_108 + 205379*uk_109 + 2793827*uk_11 + 358543*uk_110 + 111392*uk_111 + 372467*uk_112 + 755377*uk_113 + 358543*uk_114 + 625931*uk_115 + 194464*uk_116 + 650239*uk_117 + 1318709*uk_118 + 625931*uk_119 + 4877359*uk_12 + 60416*uk_120 + 202016*uk_121 + 409696*uk_122 + 194464*uk_123 + 675491*uk_124 + 1369921*uk_125 + 650239*uk_126 + 2778251*uk_127 + 1318709*uk_128 + 625931*uk_129 + 1515296*uk_13 + 1092727*uk_130 + 339488*uk_131 + 1135163*uk_132 + 2302153*uk_133 + 1092727*uk_134 + 105472*uk_135 + 352672*uk_136 + 715232*uk_137 + 339488*uk_138 + 1179247*uk_139 + 5066771*uk_14 + 2391557*uk_140 + 1135163*uk_141 + 4850167*uk_142 + 2302153*uk_143 + 1092727*uk_144 + 32768*uk_145 + 109568*uk_146 + 222208*uk_147 + 105472*uk_148 + 366368*uk_149 + 10275601*uk_15 + 743008*uk_150 + 352672*uk_151 + 1506848*uk_152 + 715232*uk_153 + 339488*uk_154 + 1225043*uk_155 + 2484433*uk_156 + 1179247*uk_157 + 5038523*uk_158 + 2391557*uk_159 + 4877359*uk_16 + 1135163*uk_160 + 10218313*uk_161 + 4850167*uk_162 + 2302153*uk_163 + 1092727*uk_164 + 3969*uk_17 + 3717*uk_18 + 6489*uk_19 + 63*uk_2 + 2016*uk_20 + 6741*uk_21 + 13671*uk_22 + 6489*uk_23 + 3481*uk_24 + 6077*uk_25 + 1888*uk_26 + 6313*uk_27 + 12803*uk_28 + 6077*uk_29 + 59*uk_3 + 10609*uk_30 + 3296*uk_31 + 11021*uk_32 + 22351*uk_33 + 10609*uk_34 + 1024*uk_35 + 3424*uk_36 + 6944*uk_37 + 3296*uk_38 + 11449*uk_39 + 103*uk_4 + 23219*uk_40 + 11021*uk_41 + 47089*uk_42 + 22351*uk_43 + 10609*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 132296089931*uk_47 + 230957580727*uk_48 + 71753811488*uk_49 + 32*uk_5 + 239926807163*uk_50 + 486580534153*uk_51 + 230957580727*uk_52 + 187944057*uk_53 + 176011101*uk_54 + 307273617*uk_55 + 95463648*uk_56 + 319206573*uk_57 + 647362863*uk_58 + 307273617*uk_59 + 107*uk_6 + 164835793*uk_60 + 287764181*uk_61 + 89402464*uk_62 + 298939489*uk_63 + 606260459*uk_64 + 287764181*uk_65 + 502367977*uk_66 + 156075488*uk_67 + 521877413*uk_68 + 1058386903*uk_69 + 217*uk_7 + 502367977*uk_70 + 48489472*uk_71 + 162136672*uk_72 + 328819232*uk_73 + 156075488*uk_74 + 542144497*uk_75 + 1099489307*uk_76 + 521877413*uk_77 + 2229805417*uk_78 + 1058386903*uk_79 + 103*uk_8 + 502367977*uk_80 + 250047*uk_81 + 234171*uk_82 + 408807*uk_83 + 127008*uk_84 + 424683*uk_85 + 861273*uk_86 + 408807*uk_87 + 219303*uk_88 + 382851*uk_89 + 2242306609*uk_9 + 118944*uk_90 + 397719*uk_91 + 806589*uk_92 + 382851*uk_93 + 668367*uk_94 + 207648*uk_95 + 694323*uk_96 + 1408113*uk_97 + 668367*uk_98 + 64512*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 219744*uk_100 + 437472*uk_101 + 118944*uk_102 + 748503*uk_103 + 1490139*uk_104 + 405153*uk_105 + 2966607*uk_106 + 806589*uk_107 + 219303*uk_108 + 103823*uk_109 + 2225591*uk_11 + 130331*uk_110 + 70688*uk_111 + 240781*uk_112 + 479353*uk_113 + 130331*uk_114 + 163607*uk_115 + 88736*uk_116 + 302257*uk_117 + 601741*uk_118 + 163607*uk_119 + 2793827*uk_12 + 48128*uk_120 + 163936*uk_121 + 326368*uk_122 + 88736*uk_123 + 558407*uk_124 + 1111691*uk_125 + 302257*uk_126 + 2213183*uk_127 + 601741*uk_128 + 163607*uk_129 + 1515296*uk_13 + 205379*uk_130 + 111392*uk_131 + 379429*uk_132 + 755377*uk_133 + 205379*uk_134 + 60416*uk_135 + 205792*uk_136 + 409696*uk_137 + 111392*uk_138 + 700979*uk_139 + 5161477*uk_14 + 1395527*uk_140 + 379429*uk_141 + 2778251*uk_142 + 755377*uk_143 + 205379*uk_144 + 32768*uk_145 + 111616*uk_146 + 222208*uk_147 + 60416*uk_148 + 380192*uk_149 + 10275601*uk_15 + 756896*uk_150 + 205792*uk_151 + 1506848*uk_152 + 409696*uk_153 + 111392*uk_154 + 1295029*uk_155 + 2578177*uk_156 + 700979*uk_157 + 5132701*uk_158 + 1395527*uk_159 + 2793827*uk_16 + 379429*uk_160 + 10218313*uk_161 + 2778251*uk_162 + 755377*uk_163 + 205379*uk_164 + 3969*uk_17 + 2961*uk_18 + 3717*uk_19 + 63*uk_2 + 2016*uk_20 + 6867*uk_21 + 13671*uk_22 + 3717*uk_23 + 2209*uk_24 + 2773*uk_25 + 1504*uk_26 + 5123*uk_27 + 10199*uk_28 + 2773*uk_29 + 47*uk_3 + 3481*uk_30 + 1888*uk_31 + 6431*uk_32 + 12803*uk_33 + 3481*uk_34 + 1024*uk_35 + 3488*uk_36 + 6944*uk_37 + 1888*uk_38 + 11881*uk_39 + 59*uk_4 + 23653*uk_40 + 6431*uk_41 + 47089*uk_42 + 12803*uk_43 + 3481*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 105388410623*uk_47 + 132296089931*uk_48 + 71753811488*uk_49 + 32*uk_5 + 244411420381*uk_50 + 486580534153*uk_51 + 132296089931*uk_52 + 187944057*uk_53 + 140212233*uk_54 + 176011101*uk_55 + 95463648*uk_56 + 325173051*uk_57 + 647362863*uk_58 + 176011101*uk_59 + 109*uk_6 + 104602777*uk_60 + 131309869*uk_61 + 71218912*uk_62 + 242589419*uk_63 + 482953247*uk_64 + 131309869*uk_65 + 164835793*uk_66 + 89402464*uk_67 + 304527143*uk_68 + 606260459*uk_69 + 217*uk_7 + 164835793*uk_70 + 48489472*uk_71 + 165167264*uk_72 + 328819232*uk_73 + 89402464*uk_74 + 562600993*uk_75 + 1120040509*uk_76 + 304527143*uk_77 + 2229805417*uk_78 + 606260459*uk_79 + 59*uk_8 + 164835793*uk_80 + 250047*uk_81 + 186543*uk_82 + 234171*uk_83 + 127008*uk_84 + 432621*uk_85 + 861273*uk_86 + 234171*uk_87 + 139167*uk_88 + 174699*uk_89 + 2242306609*uk_9 + 94752*uk_90 + 322749*uk_91 + 642537*uk_92 + 174699*uk_93 + 219303*uk_94 + 118944*uk_95 + 405153*uk_96 + 806589*uk_97 + 219303*uk_98 + 64512*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 223776*uk_100 + 437472*uk_101 + 94752*uk_102 + 776223*uk_103 + 1517481*uk_104 + 328671*uk_105 + 2966607*uk_106 + 642537*uk_107 + 139167*uk_108 + 300763*uk_109 + 3172651*uk_11 + 210983*uk_110 + 143648*uk_111 + 498279*uk_112 + 974113*uk_113 + 210983*uk_114 + 148003*uk_115 + 100768*uk_116 + 349539*uk_117 + 683333*uk_118 + 148003*uk_119 + 2225591*uk_12 + 68608*uk_120 + 237984*uk_121 + 465248*uk_122 + 100768*uk_123 + 825507*uk_124 + 1613829*uk_125 + 349539*uk_126 + 3154963*uk_127 + 683333*uk_128 + 148003*uk_129 + 1515296*uk_13 + 103823*uk_130 + 70688*uk_131 + 245199*uk_132 + 479353*uk_133 + 103823*uk_134 + 48128*uk_135 + 166944*uk_136 + 326368*uk_137 + 70688*uk_138 + 579087*uk_139 + 5256183*uk_14 + 1132089*uk_140 + 245199*uk_141 + 2213183*uk_142 + 479353*uk_143 + 103823*uk_144 + 32768*uk_145 + 113664*uk_146 + 222208*uk_147 + 48128*uk_148 + 394272*uk_149 + 10275601*uk_15 + 770784*uk_150 + 166944*uk_151 + 1506848*uk_152 + 326368*uk_153 + 70688*uk_154 + 1367631*uk_155 + 2673657*uk_156 + 579087*uk_157 + 5226879*uk_158 + 1132089*uk_159 + 2225591*uk_16 + 245199*uk_160 + 10218313*uk_161 + 2213183*uk_162 + 479353*uk_163 + 103823*uk_164 + 3969*uk_17 + 4221*uk_18 + 2961*uk_19 + 63*uk_2 + 2016*uk_20 + 6993*uk_21 + 13671*uk_22 + 2961*uk_23 + 4489*uk_24 + 3149*uk_25 + 2144*uk_26 + 7437*uk_27 + 14539*uk_28 + 3149*uk_29 + 67*uk_3 + 2209*uk_30 + 1504*uk_31 + 5217*uk_32 + 10199*uk_33 + 2209*uk_34 + 1024*uk_35 + 3552*uk_36 + 6944*uk_37 + 1504*uk_38 + 12321*uk_39 + 47*uk_4 + 24087*uk_40 + 5217*uk_41 + 47089*uk_42 + 10199*uk_43 + 2209*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 150234542803*uk_47 + 105388410623*uk_48 + 71753811488*uk_49 + 32*uk_5 + 248896033599*uk_50 + 486580534153*uk_51 + 105388410623*uk_52 + 187944057*uk_53 + 199877013*uk_54 + 140212233*uk_55 + 95463648*uk_56 + 331139529*uk_57 + 647362863*uk_58 + 140212233*uk_59 + 111*uk_6 + 212567617*uk_60 + 149114597*uk_61 + 101524832*uk_62 + 352164261*uk_63 + 688465267*uk_64 + 149114597*uk_65 + 104602777*uk_66 + 71218912*uk_67 + 247040601*uk_68 + 482953247*uk_69 + 217*uk_7 + 104602777*uk_70 + 48489472*uk_71 + 168197856*uk_72 + 328819232*uk_73 + 71218912*uk_74 + 583436313*uk_75 + 1140591711*uk_76 + 247040601*uk_77 + 2229805417*uk_78 + 482953247*uk_79 + 47*uk_8 + 104602777*uk_80 + 250047*uk_81 + 265923*uk_82 + 186543*uk_83 + 127008*uk_84 + 440559*uk_85 + 861273*uk_86 + 186543*uk_87 + 282807*uk_88 + 198387*uk_89 + 2242306609*uk_9 + 135072*uk_90 + 468531*uk_91 + 915957*uk_92 + 198387*uk_93 + 139167*uk_94 + 94752*uk_95 + 328671*uk_96 + 642537*uk_97 + 139167*uk_98 + 64512*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 199332*uk_100 + 382788*uk_101 + 118188*uk_102 + 804447*uk_103 + 1544823*uk_104 + 476973*uk_105 + 2966607*uk_106 + 915957*uk_107 + 282807*uk_108 + 216*uk_109 + 284118*uk_11 + 2412*uk_110 + 1008*uk_111 + 4068*uk_112 + 7812*uk_113 + 2412*uk_114 + 26934*uk_115 + 11256*uk_116 + 45426*uk_117 + 87234*uk_118 + 26934*uk_119 + 3172651*uk_12 + 4704*uk_120 + 18984*uk_121 + 36456*uk_122 + 11256*uk_123 + 76614*uk_124 + 147126*uk_125 + 45426*uk_126 + 282534*uk_127 + 87234*uk_128 + 26934*uk_129 + 1325884*uk_13 + 300763*uk_130 + 125692*uk_131 + 507257*uk_132 + 974113*uk_133 + 300763*uk_134 + 52528*uk_135 + 211988*uk_136 + 407092*uk_137 + 125692*uk_138 + 855523*uk_139 + 5350889*uk_14 + 1642907*uk_140 + 507257*uk_141 + 3154963*uk_142 + 974113*uk_143 + 300763*uk_144 + 21952*uk_145 + 88592*uk_146 + 170128*uk_147 + 52528*uk_148 + 357532*uk_149 + 10275601*uk_15 + 686588*uk_150 + 211988*uk_151 + 1318492*uk_152 + 407092*uk_153 + 125692*uk_154 + 1442897*uk_155 + 2770873*uk_156 + 855523*uk_157 + 5321057*uk_158 + 1642907*uk_159 + 3172651*uk_16 + 507257*uk_160 + 10218313*uk_161 + 3154963*uk_162 + 974113*uk_163 + 300763*uk_164 + 3969*uk_17 + 378*uk_18 + 4221*uk_19 + 63*uk_2 + 1764*uk_20 + 7119*uk_21 + 13671*uk_22 + 4221*uk_23 + 36*uk_24 + 402*uk_25 + 168*uk_26 + 678*uk_27 + 1302*uk_28 + 402*uk_29 + 6*uk_3 + 4489*uk_30 + 1876*uk_31 + 7571*uk_32 + 14539*uk_33 + 4489*uk_34 + 784*uk_35 + 3164*uk_36 + 6076*uk_37 + 1876*uk_38 + 12769*uk_39 + 67*uk_4 + 24521*uk_40 + 7571*uk_41 + 47089*uk_42 + 14539*uk_43 + 4489*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 13453839654*uk_47 + 150234542803*uk_48 + 62784585052*uk_49 + 28*uk_5 + 253380646817*uk_50 + 486580534153*uk_51 + 150234542803*uk_52 + 187944057*uk_53 + 17899434*uk_54 + 199877013*uk_55 + 83530692*uk_56 + 337106007*uk_57 + 647362863*uk_58 + 199877013*uk_59 + 113*uk_6 + 1704708*uk_60 + 19035906*uk_61 + 7955304*uk_62 + 32105334*uk_63 + 61653606*uk_64 + 19035906*uk_65 + 212567617*uk_66 + 88834228*uk_67 + 358509563*uk_68 + 688465267*uk_69 + 217*uk_7 + 212567617*uk_70 + 37124752*uk_71 + 149824892*uk_72 + 287716828*uk_73 + 88834228*uk_74 + 604650457*uk_75 + 1161142913*uk_76 + 358509563*uk_77 + 2229805417*uk_78 + 688465267*uk_79 + 67*uk_8 + 212567617*uk_80 + 250047*uk_81 + 23814*uk_82 + 265923*uk_83 + 111132*uk_84 + 448497*uk_85 + 861273*uk_86 + 265923*uk_87 + 2268*uk_88 + 25326*uk_89 + 2242306609*uk_9 + 10584*uk_90 + 42714*uk_91 + 82026*uk_92 + 25326*uk_93 + 282807*uk_94 + 118188*uk_95 + 476973*uk_96 + 915957*uk_97 + 282807*uk_98 + 49392*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 231840*uk_100 + 437472*uk_101 + 12096*uk_102 + 833175*uk_103 + 1572165*uk_104 + 43470*uk_105 + 2966607*uk_106 + 82026*uk_107 + 2268*uk_108 + 681472*uk_109 + 4167064*uk_11 + 46464*uk_110 + 247808*uk_111 + 890560*uk_112 + 1680448*uk_113 + 46464*uk_114 + 3168*uk_115 + 16896*uk_116 + 60720*uk_117 + 114576*uk_118 + 3168*uk_119 + 284118*uk_12 + 90112*uk_120 + 323840*uk_121 + 611072*uk_122 + 16896*uk_123 + 1163800*uk_124 + 2196040*uk_125 + 60720*uk_126 + 4143832*uk_127 + 114576*uk_128 + 3168*uk_129 + 1515296*uk_13 + 216*uk_130 + 1152*uk_131 + 4140*uk_132 + 7812*uk_133 + 216*uk_134 + 6144*uk_135 + 22080*uk_136 + 41664*uk_137 + 1152*uk_138 + 79350*uk_139 + 5445595*uk_14 + 149730*uk_140 + 4140*uk_141 + 282534*uk_142 + 7812*uk_143 + 216*uk_144 + 32768*uk_145 + 117760*uk_146 + 222208*uk_147 + 6144*uk_148 + 423200*uk_149 + 10275601*uk_15 + 798560*uk_150 + 22080*uk_151 + 1506848*uk_152 + 41664*uk_153 + 1152*uk_154 + 1520875*uk_155 + 2869825*uk_156 + 79350*uk_157 + 5415235*uk_158 + 149730*uk_159 + 284118*uk_16 + 4140*uk_160 + 10218313*uk_161 + 282534*uk_162 + 7812*uk_163 + 216*uk_164 + 3969*uk_17 + 5544*uk_18 + 378*uk_19 + 63*uk_2 + 2016*uk_20 + 7245*uk_21 + 13671*uk_22 + 378*uk_23 + 7744*uk_24 + 528*uk_25 + 2816*uk_26 + 10120*uk_27 + 19096*uk_28 + 528*uk_29 + 88*uk_3 + 36*uk_30 + 192*uk_31 + 690*uk_32 + 1302*uk_33 + 36*uk_34 + 1024*uk_35 + 3680*uk_36 + 6944*uk_37 + 192*uk_38 + 13225*uk_39 + 6*uk_4 + 24955*uk_40 + 690*uk_41 + 47089*uk_42 + 1302*uk_43 + 36*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 197322981592*uk_47 + 13453839654*uk_48 + 71753811488*uk_49 + 32*uk_5 + 257865260035*uk_50 + 486580534153*uk_51 + 13453839654*uk_52 + 187944057*uk_53 + 262525032*uk_54 + 17899434*uk_55 + 95463648*uk_56 + 343072485*uk_57 + 647362863*uk_58 + 17899434*uk_59 + 115*uk_6 + 366701632*uk_60 + 25002384*uk_61 + 133346048*uk_62 + 479212360*uk_63 + 904252888*uk_64 + 25002384*uk_65 + 1704708*uk_66 + 9091776*uk_67 + 32673570*uk_68 + 61653606*uk_69 + 217*uk_7 + 1704708*uk_70 + 48489472*uk_71 + 174259040*uk_72 + 328819232*uk_73 + 9091776*uk_74 + 626243425*uk_75 + 1181694115*uk_76 + 32673570*uk_77 + 2229805417*uk_78 + 61653606*uk_79 + 6*uk_8 + 1704708*uk_80 + 250047*uk_81 + 349272*uk_82 + 23814*uk_83 + 127008*uk_84 + 456435*uk_85 + 861273*uk_86 + 23814*uk_87 + 487872*uk_88 + 33264*uk_89 + 2242306609*uk_9 + 177408*uk_90 + 637560*uk_91 + 1203048*uk_92 + 33264*uk_93 + 2268*uk_94 + 12096*uk_95 + 43470*uk_96 + 82026*uk_97 + 2268*uk_98 + 64512*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 206388*uk_100 + 382788*uk_101 + 155232*uk_102 + 862407*uk_103 + 1599507*uk_104 + 648648*uk_105 + 2966607*uk_106 + 1203048*uk_107 + 487872*uk_108 + 614125*uk_109 + 4025005*uk_11 + 635800*uk_110 + 202300*uk_111 + 845325*uk_112 + 1567825*uk_113 + 635800*uk_114 + 658240*uk_115 + 209440*uk_116 + 875160*uk_117 + 1623160*uk_118 + 658240*uk_119 + 4167064*uk_12 + 66640*uk_120 + 278460*uk_121 + 516460*uk_122 + 209440*uk_123 + 1163565*uk_124 + 2158065*uk_125 + 875160*uk_126 + 4002565*uk_127 + 1623160*uk_128 + 658240*uk_129 + 1325884*uk_13 + 681472*uk_130 + 216832*uk_131 + 906048*uk_132 + 1680448*uk_133 + 681472*uk_134 + 68992*uk_135 + 288288*uk_136 + 534688*uk_137 + 216832*uk_138 + 1204632*uk_139 + 5540301*uk_14 + 2234232*uk_140 + 906048*uk_141 + 4143832*uk_142 + 1680448*uk_143 + 681472*uk_144 + 21952*uk_145 + 91728*uk_146 + 170128*uk_147 + 68992*uk_148 + 383292*uk_149 + 10275601*uk_15 + 710892*uk_150 + 288288*uk_151 + 1318492*uk_152 + 534688*uk_153 + 216832*uk_154 + 1601613*uk_155 + 2970513*uk_156 + 1204632*uk_157 + 5509413*uk_158 + 2234232*uk_159 + 4167064*uk_16 + 906048*uk_160 + 10218313*uk_161 + 4143832*uk_162 + 1680448*uk_163 + 681472*uk_164 + 3969*uk_17 + 5355*uk_18 + 5544*uk_19 + 63*uk_2 + 1764*uk_20 + 7371*uk_21 + 13671*uk_22 + 5544*uk_23 + 7225*uk_24 + 7480*uk_25 + 2380*uk_26 + 9945*uk_27 + 18445*uk_28 + 7480*uk_29 + 85*uk_3 + 7744*uk_30 + 2464*uk_31 + 10296*uk_32 + 19096*uk_33 + 7744*uk_34 + 784*uk_35 + 3276*uk_36 + 6076*uk_37 + 2464*uk_38 + 13689*uk_39 + 88*uk_4 + 25389*uk_40 + 10296*uk_41 + 47089*uk_42 + 19096*uk_43 + 7744*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 190596061765*uk_47 + 197322981592*uk_48 + 62784585052*uk_49 + 28*uk_5 + 262349873253*uk_50 + 486580534153*uk_51 + 197322981592*uk_52 + 187944057*uk_53 + 253575315*uk_54 + 262525032*uk_55 + 83530692*uk_56 + 349038963*uk_57 + 647362863*uk_58 + 262525032*uk_59 + 117*uk_6 + 342125425*uk_60 + 354200440*uk_61 + 112700140*uk_62 + 470925585*uk_63 + 873426085*uk_64 + 354200440*uk_65 + 366701632*uk_66 + 116677792*uk_67 + 487546488*uk_68 + 904252888*uk_69 + 217*uk_7 + 366701632*uk_70 + 37124752*uk_71 + 155128428*uk_72 + 287716828*uk_73 + 116677792*uk_74 + 648215217*uk_75 + 1202245317*uk_76 + 487546488*uk_77 + 2229805417*uk_78 + 904252888*uk_79 + 88*uk_8 + 366701632*uk_80 + 250047*uk_81 + 337365*uk_82 + 349272*uk_83 + 111132*uk_84 + 464373*uk_85 + 861273*uk_86 + 349272*uk_87 + 455175*uk_88 + 471240*uk_89 + 2242306609*uk_9 + 149940*uk_90 + 626535*uk_91 + 1162035*uk_92 + 471240*uk_93 + 487872*uk_94 + 155232*uk_95 + 648648*uk_96 + 1203048*uk_97 + 487872*uk_98 + 49392*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 209916*uk_100 + 382788*uk_101 + 149940*uk_102 + 892143*uk_103 + 1626849*uk_104 + 637245*uk_105 + 2966607*uk_106 + 1162035*uk_107 + 455175*uk_108 + 1331000*uk_109 + 5208830*uk_11 + 1028500*uk_110 + 338800*uk_111 + 1439900*uk_112 + 2625700*uk_113 + 1028500*uk_114 + 794750*uk_115 + 261800*uk_116 + 1112650*uk_117 + 2028950*uk_118 + 794750*uk_119 + 4025005*uk_12 + 86240*uk_120 + 366520*uk_121 + 668360*uk_122 + 261800*uk_123 + 1557710*uk_124 + 2840530*uk_125 + 1112650*uk_126 + 5179790*uk_127 + 2028950*uk_128 + 794750*uk_129 + 1325884*uk_13 + 614125*uk_130 + 202300*uk_131 + 859775*uk_132 + 1567825*uk_133 + 614125*uk_134 + 66640*uk_135 + 283220*uk_136 + 516460*uk_137 + 202300*uk_138 + 1203685*uk_139 + 5635007*uk_14 + 2194955*uk_140 + 859775*uk_141 + 4002565*uk_142 + 1567825*uk_143 + 614125*uk_144 + 21952*uk_145 + 93296*uk_146 + 170128*uk_147 + 66640*uk_148 + 396508*uk_149 + 10275601*uk_15 + 723044*uk_150 + 283220*uk_151 + 1318492*uk_152 + 516460*uk_153 + 202300*uk_154 + 1685159*uk_155 + 3072937*uk_156 + 1203685*uk_157 + 5603591*uk_158 + 2194955*uk_159 + 4025005*uk_16 + 859775*uk_160 + 10218313*uk_161 + 4002565*uk_162 + 1567825*uk_163 + 614125*uk_164 + 3969*uk_17 + 6930*uk_18 + 5355*uk_19 + 63*uk_2 + 1764*uk_20 + 7497*uk_21 + 13671*uk_22 + 5355*uk_23 + 12100*uk_24 + 9350*uk_25 + 3080*uk_26 + 13090*uk_27 + 23870*uk_28 + 9350*uk_29 + 110*uk_3 + 7225*uk_30 + 2380*uk_31 + 10115*uk_32 + 18445*uk_33 + 7225*uk_34 + 784*uk_35 + 3332*uk_36 + 6076*uk_37 + 2380*uk_38 + 14161*uk_39 + 85*uk_4 + 25823*uk_40 + 10115*uk_41 + 47089*uk_42 + 18445*uk_43 + 7225*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 246653726990*uk_47 + 190596061765*uk_48 + 62784585052*uk_49 + 28*uk_5 + 266834486471*uk_50 + 486580534153*uk_51 + 190596061765*uk_52 + 187944057*uk_53 + 328156290*uk_54 + 253575315*uk_55 + 83530692*uk_56 + 355005441*uk_57 + 647362863*uk_58 + 253575315*uk_59 + 119*uk_6 + 572971300*uk_60 + 442750550*uk_61 + 145847240*uk_62 + 619850770*uk_63 + 1130316110*uk_64 + 442750550*uk_65 + 342125425*uk_66 + 112700140*uk_67 + 478975595*uk_68 + 873426085*uk_69 + 217*uk_7 + 342125425*uk_70 + 37124752*uk_71 + 157780196*uk_72 + 287716828*uk_73 + 112700140*uk_74 + 670565833*uk_75 + 1222796519*uk_76 + 478975595*uk_77 + 2229805417*uk_78 + 873426085*uk_79 + 85*uk_8 + 342125425*uk_80 + 250047*uk_81 + 436590*uk_82 + 337365*uk_83 + 111132*uk_84 + 472311*uk_85 + 861273*uk_86 + 337365*uk_87 + 762300*uk_88 + 589050*uk_89 + 2242306609*uk_9 + 194040*uk_90 + 824670*uk_91 + 1503810*uk_92 + 589050*uk_93 + 455175*uk_94 + 149940*uk_95 + 637245*uk_96 + 1162035*uk_97 + 455175*uk_98 + 49392*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 182952*uk_100 + 328104*uk_101 + 166320*uk_102 + 922383*uk_103 + 1654191*uk_104 + 838530*uk_105 + 2966607*uk_106 + 1503810*uk_107 + 762300*uk_108 + 74088*uk_109 + 1988826*uk_11 + 194040*uk_110 + 42336*uk_111 + 213444*uk_112 + 382788*uk_113 + 194040*uk_114 + 508200*uk_115 + 110880*uk_116 + 559020*uk_117 + 1002540*uk_118 + 508200*uk_119 + 5208830*uk_12 + 24192*uk_120 + 121968*uk_121 + 218736*uk_122 + 110880*uk_123 + 614922*uk_124 + 1102794*uk_125 + 559020*uk_126 + 1977738*uk_127 + 1002540*uk_128 + 508200*uk_129 + 1136472*uk_13 + 1331000*uk_130 + 290400*uk_131 + 1464100*uk_132 + 2625700*uk_133 + 1331000*uk_134 + 63360*uk_135 + 319440*uk_136 + 572880*uk_137 + 290400*uk_138 + 1610510*uk_139 + 5729713*uk_14 + 2888270*uk_140 + 1464100*uk_141 + 5179790*uk_142 + 2625700*uk_143 + 1331000*uk_144 + 13824*uk_145 + 69696*uk_146 + 124992*uk_147 + 63360*uk_148 + 351384*uk_149 + 10275601*uk_15 + 630168*uk_150 + 319440*uk_151 + 1130136*uk_152 + 572880*uk_153 + 290400*uk_154 + 1771561*uk_155 + 3177097*uk_156 + 1610510*uk_157 + 5697769*uk_158 + 2888270*uk_159 + 5208830*uk_16 + 1464100*uk_160 + 10218313*uk_161 + 5179790*uk_162 + 2625700*uk_163 + 1331000*uk_164 + 3969*uk_17 + 2646*uk_18 + 6930*uk_19 + 63*uk_2 + 1512*uk_20 + 7623*uk_21 + 13671*uk_22 + 6930*uk_23 + 1764*uk_24 + 4620*uk_25 + 1008*uk_26 + 5082*uk_27 + 9114*uk_28 + 4620*uk_29 + 42*uk_3 + 12100*uk_30 + 2640*uk_31 + 13310*uk_32 + 23870*uk_33 + 12100*uk_34 + 576*uk_35 + 2904*uk_36 + 5208*uk_37 + 2640*uk_38 + 14641*uk_39 + 110*uk_4 + 26257*uk_40 + 13310*uk_41 + 47089*uk_42 + 23870*uk_43 + 12100*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 94176877578*uk_47 + 246653726990*uk_48 + 53815358616*uk_49 + 24*uk_5 + 271319099689*uk_50 + 486580534153*uk_51 + 246653726990*uk_52 + 187944057*uk_53 + 125296038*uk_54 + 328156290*uk_55 + 71597736*uk_56 + 360971919*uk_57 + 647362863*uk_58 + 328156290*uk_59 + 121*uk_6 + 83530692*uk_60 + 218770860*uk_61 + 47731824*uk_62 + 240647946*uk_63 + 431575242*uk_64 + 218770860*uk_65 + 572971300*uk_66 + 125011920*uk_67 + 630268430*uk_68 + 1130316110*uk_69 + 217*uk_7 + 572971300*uk_70 + 27275328*uk_71 + 137513112*uk_72 + 246614424*uk_73 + 125011920*uk_74 + 693295273*uk_75 + 1243347721*uk_76 + 630268430*uk_77 + 2229805417*uk_78 + 1130316110*uk_79 + 110*uk_8 + 572971300*uk_80 + 250047*uk_81 + 166698*uk_82 + 436590*uk_83 + 95256*uk_84 + 480249*uk_85 + 861273*uk_86 + 436590*uk_87 + 111132*uk_88 + 291060*uk_89 + 2242306609*uk_9 + 63504*uk_90 + 320166*uk_91 + 574182*uk_92 + 291060*uk_93 + 762300*uk_94 + 166320*uk_95 + 838530*uk_96 + 1503810*uk_97 + 762300*uk_98 + 36288*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 216972*uk_100 + 382788*uk_101 + 74088*uk_102 + 953127*uk_103 + 1681533*uk_104 + 325458*uk_105 + 2966607*uk_106 + 574182*uk_107 + 111132*uk_108 + 1771561*uk_109 + 5729713*uk_11 + 614922*uk_110 + 409948*uk_111 + 1800843*uk_112 + 3177097*uk_113 + 614922*uk_114 + 213444*uk_115 + 142296*uk_116 + 625086*uk_117 + 1102794*uk_118 + 213444*uk_119 + 1988826*uk_12 + 94864*uk_120 + 416724*uk_121 + 735196*uk_122 + 142296*uk_123 + 1830609*uk_124 + 3229611*uk_125 + 625086*uk_126 + 5697769*uk_127 + 1102794*uk_128 + 213444*uk_129 + 1325884*uk_13 + 74088*uk_130 + 49392*uk_131 + 216972*uk_132 + 382788*uk_133 + 74088*uk_134 + 32928*uk_135 + 144648*uk_136 + 255192*uk_137 + 49392*uk_138 + 635418*uk_139 + 5824419*uk_14 + 1121022*uk_140 + 216972*uk_141 + 1977738*uk_142 + 382788*uk_143 + 74088*uk_144 + 21952*uk_145 + 96432*uk_146 + 170128*uk_147 + 32928*uk_148 + 423612*uk_149 + 10275601*uk_15 + 747348*uk_150 + 144648*uk_151 + 1318492*uk_152 + 255192*uk_153 + 49392*uk_154 + 1860867*uk_155 + 3282993*uk_156 + 635418*uk_157 + 5791947*uk_158 + 1121022*uk_159 + 1988826*uk_16 + 216972*uk_160 + 10218313*uk_161 + 1977738*uk_162 + 382788*uk_163 + 74088*uk_164 + 3969*uk_17 + 7623*uk_18 + 2646*uk_19 + 63*uk_2 + 1764*uk_20 + 7749*uk_21 + 13671*uk_22 + 2646*uk_23 + 14641*uk_24 + 5082*uk_25 + 3388*uk_26 + 14883*uk_27 + 26257*uk_28 + 5082*uk_29 + 121*uk_3 + 1764*uk_30 + 1176*uk_31 + 5166*uk_32 + 9114*uk_33 + 1764*uk_34 + 784*uk_35 + 3444*uk_36 + 6076*uk_37 + 1176*uk_38 + 15129*uk_39 + 42*uk_4 + 26691*uk_40 + 5166*uk_41 + 47089*uk_42 + 9114*uk_43 + 1764*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 271319099689*uk_47 + 94176877578*uk_48 + 62784585052*uk_49 + 28*uk_5 + 275803712907*uk_50 + 486580534153*uk_51 + 94176877578*uk_52 + 187944057*uk_53 + 360971919*uk_54 + 125296038*uk_55 + 83530692*uk_56 + 366938397*uk_57 + 647362863*uk_58 + 125296038*uk_59 + 123*uk_6 + 693295273*uk_60 + 240647946*uk_61 + 160431964*uk_62 + 704754699*uk_63 + 1243347721*uk_64 + 240647946*uk_65 + 83530692*uk_66 + 55687128*uk_67 + 244625598*uk_68 + 431575242*uk_69 + 217*uk_7 + 83530692*uk_70 + 37124752*uk_71 + 163083732*uk_72 + 287716828*uk_73 + 55687128*uk_74 + 716403537*uk_75 + 1263898923*uk_76 + 244625598*uk_77 + 2229805417*uk_78 + 431575242*uk_79 + 42*uk_8 + 83530692*uk_80 + 250047*uk_81 + 480249*uk_82 + 166698*uk_83 + 111132*uk_84 + 488187*uk_85 + 861273*uk_86 + 166698*uk_87 + 922383*uk_88 + 320166*uk_89 + 2242306609*uk_9 + 213444*uk_90 + 937629*uk_91 + 1654191*uk_92 + 320166*uk_93 + 111132*uk_94 + 74088*uk_95 + 325458*uk_96 + 574182*uk_97 + 111132*uk_98 + 49392*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 189000*uk_100 + 328104*uk_101 + 182952*uk_102 + 984375*uk_103 + 1708875*uk_104 + 952875*uk_105 + 2966607*uk_106 + 1654191*uk_107 + 922383*uk_108 + 1092727*uk_109 + 4877359*uk_11 + 1283689*uk_110 + 254616*uk_111 + 1326125*uk_112 + 2302153*uk_113 + 1283689*uk_114 + 1508023*uk_115 + 299112*uk_116 + 1557875*uk_117 + 2704471*uk_118 + 1508023*uk_119 + 5729713*uk_12 + 59328*uk_120 + 309000*uk_121 + 536424*uk_122 + 299112*uk_123 + 1609375*uk_124 + 2793875*uk_125 + 1557875*uk_126 + 4850167*uk_127 + 2704471*uk_128 + 1508023*uk_129 + 1136472*uk_13 + 1771561*uk_130 + 351384*uk_131 + 1830125*uk_132 + 3177097*uk_133 + 1771561*uk_134 + 69696*uk_135 + 363000*uk_136 + 630168*uk_137 + 351384*uk_138 + 1890625*uk_139 + 5919125*uk_14 + 3282125*uk_140 + 1830125*uk_141 + 5697769*uk_142 + 3177097*uk_143 + 1771561*uk_144 + 13824*uk_145 + 72000*uk_146 + 124992*uk_147 + 69696*uk_148 + 375000*uk_149 + 10275601*uk_15 + 651000*uk_150 + 363000*uk_151 + 1130136*uk_152 + 630168*uk_153 + 351384*uk_154 + 1953125*uk_155 + 3390625*uk_156 + 1890625*uk_157 + 5886125*uk_158 + 3282125*uk_159 + 5729713*uk_16 + 1830125*uk_160 + 10218313*uk_161 + 5697769*uk_162 + 3177097*uk_163 + 1771561*uk_164 + 3969*uk_17 + 6489*uk_18 + 7623*uk_19 + 63*uk_2 + 1512*uk_20 + 7875*uk_21 + 13671*uk_22 + 7623*uk_23 + 10609*uk_24 + 12463*uk_25 + 2472*uk_26 + 12875*uk_27 + 22351*uk_28 + 12463*uk_29 + 103*uk_3 + 14641*uk_30 + 2904*uk_31 + 15125*uk_32 + 26257*uk_33 + 14641*uk_34 + 576*uk_35 + 3000*uk_36 + 5208*uk_37 + 2904*uk_38 + 15625*uk_39 + 121*uk_4 + 27125*uk_40 + 15125*uk_41 + 47089*uk_42 + 26257*uk_43 + 14641*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 230957580727*uk_47 + 271319099689*uk_48 + 53815358616*uk_49 + 24*uk_5 + 280288326125*uk_50 + 486580534153*uk_51 + 271319099689*uk_52 + 187944057*uk_53 + 307273617*uk_54 + 360971919*uk_55 + 71597736*uk_56 + 372904875*uk_57 + 647362863*uk_58 + 360971919*uk_59 + 125*uk_6 + 502367977*uk_60 + 590160439*uk_61 + 117056616*uk_62 + 609669875*uk_63 + 1058386903*uk_64 + 590160439*uk_65 + 693295273*uk_66 + 137513112*uk_67 + 716214125*uk_68 + 1243347721*uk_69 + 217*uk_7 + 693295273*uk_70 + 27275328*uk_71 + 142059000*uk_72 + 246614424*uk_73 + 137513112*uk_74 + 739890625*uk_75 + 1284450125*uk_76 + 716214125*uk_77 + 2229805417*uk_78 + 1243347721*uk_79 + 121*uk_8 + 693295273*uk_80 + 250047*uk_81 + 408807*uk_82 + 480249*uk_83 + 95256*uk_84 + 496125*uk_85 + 861273*uk_86 + 480249*uk_87 + 668367*uk_88 + 785169*uk_89 + 2242306609*uk_9 + 155736*uk_90 + 811125*uk_91 + 1408113*uk_92 + 785169*uk_93 + 922383*uk_94 + 182952*uk_95 + 952875*uk_96 + 1654191*uk_97 + 922383*uk_98 + 36288*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 192024*uk_100 + 328104*uk_101 + 155736*uk_102 + 1016127*uk_103 + 1736217*uk_104 + 824103*uk_105 + 2966607*uk_106 + 1408113*uk_107 + 668367*uk_108 + 1295029*uk_109 + 5161477*uk_11 + 1223743*uk_110 + 285144*uk_111 + 1508887*uk_112 + 2578177*uk_113 + 1223743*uk_114 + 1156381*uk_115 + 269448*uk_116 + 1425829*uk_117 + 2436259*uk_118 + 1156381*uk_119 + 4877359*uk_12 + 62784*uk_120 + 332232*uk_121 + 567672*uk_122 + 269448*uk_123 + 1758061*uk_124 + 3003931*uk_125 + 1425829*uk_126 + 5132701*uk_127 + 2436259*uk_128 + 1156381*uk_129 + 1136472*uk_13 + 1092727*uk_130 + 254616*uk_131 + 1347343*uk_132 + 2302153*uk_133 + 1092727*uk_134 + 59328*uk_135 + 313944*uk_136 + 536424*uk_137 + 254616*uk_138 + 1661287*uk_139 + 6013831*uk_14 + 2838577*uk_140 + 1347343*uk_141 + 4850167*uk_142 + 2302153*uk_143 + 1092727*uk_144 + 13824*uk_145 + 73152*uk_146 + 124992*uk_147 + 59328*uk_148 + 387096*uk_149 + 10275601*uk_15 + 661416*uk_150 + 313944*uk_151 + 1130136*uk_152 + 536424*uk_153 + 254616*uk_154 + 2048383*uk_155 + 3499993*uk_156 + 1661287*uk_157 + 5980303*uk_158 + 2838577*uk_159 + 4877359*uk_16 + 1347343*uk_160 + 10218313*uk_161 + 4850167*uk_162 + 2302153*uk_163 + 1092727*uk_164 + 3969*uk_17 + 6867*uk_18 + 6489*uk_19 + 63*uk_2 + 1512*uk_20 + 8001*uk_21 + 13671*uk_22 + 6489*uk_23 + 11881*uk_24 + 11227*uk_25 + 2616*uk_26 + 13843*uk_27 + 23653*uk_28 + 11227*uk_29 + 109*uk_3 + 10609*uk_30 + 2472*uk_31 + 13081*uk_32 + 22351*uk_33 + 10609*uk_34 + 576*uk_35 + 3048*uk_36 + 5208*uk_37 + 2472*uk_38 + 16129*uk_39 + 103*uk_4 + 27559*uk_40 + 13081*uk_41 + 47089*uk_42 + 22351*uk_43 + 10609*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 244411420381*uk_47 + 230957580727*uk_48 + 53815358616*uk_49 + 24*uk_5 + 284772939343*uk_50 + 486580534153*uk_51 + 230957580727*uk_52 + 187944057*uk_53 + 325173051*uk_54 + 307273617*uk_55 + 71597736*uk_56 + 378871353*uk_57 + 647362863*uk_58 + 307273617*uk_59 + 127*uk_6 + 562600993*uk_60 + 531632131*uk_61 + 123875448*uk_62 + 655507579*uk_63 + 1120040509*uk_64 + 531632131*uk_65 + 502367977*uk_66 + 117056616*uk_67 + 619424593*uk_68 + 1058386903*uk_69 + 217*uk_7 + 502367977*uk_70 + 27275328*uk_71 + 144331944*uk_72 + 246614424*uk_73 + 117056616*uk_74 + 763756537*uk_75 + 1305001327*uk_76 + 619424593*uk_77 + 2229805417*uk_78 + 1058386903*uk_79 + 103*uk_8 + 502367977*uk_80 + 250047*uk_81 + 432621*uk_82 + 408807*uk_83 + 95256*uk_84 + 504063*uk_85 + 861273*uk_86 + 408807*uk_87 + 748503*uk_88 + 707301*uk_89 + 2242306609*uk_9 + 164808*uk_90 + 872109*uk_91 + 1490139*uk_92 + 707301*uk_93 + 668367*uk_94 + 155736*uk_95 + 824103*uk_96 + 1408113*uk_97 + 668367*uk_98 + 36288*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 162540*uk_100 + 273420*uk_101 + 137340*uk_102 + 1048383*uk_103 + 1763559*uk_104 + 885843*uk_105 + 2966607*uk_106 + 1490139*uk_107 + 748503*uk_108 + 1000*uk_109 + 473530*uk_11 + 10900*uk_110 + 2000*uk_111 + 12900*uk_112 + 21700*uk_113 + 10900*uk_114 + 118810*uk_115 + 21800*uk_116 + 140610*uk_117 + 236530*uk_118 + 118810*uk_119 + 5161477*uk_12 + 4000*uk_120 + 25800*uk_121 + 43400*uk_122 + 21800*uk_123 + 166410*uk_124 + 279930*uk_125 + 140610*uk_126 + 470890*uk_127 + 236530*uk_128 + 118810*uk_129 + 947060*uk_13 + 1295029*uk_130 + 237620*uk_131 + 1532649*uk_132 + 2578177*uk_133 + 1295029*uk_134 + 43600*uk_135 + 281220*uk_136 + 473060*uk_137 + 237620*uk_138 + 1813869*uk_139 + 6108537*uk_14 + 3051237*uk_140 + 1532649*uk_141 + 5132701*uk_142 + 2578177*uk_143 + 1295029*uk_144 + 8000*uk_145 + 51600*uk_146 + 86800*uk_147 + 43600*uk_148 + 332820*uk_149 + 10275601*uk_15 + 559860*uk_150 + 281220*uk_151 + 941780*uk_152 + 473060*uk_153 + 237620*uk_154 + 2146689*uk_155 + 3611097*uk_156 + 1813869*uk_157 + 6074481*uk_158 + 3051237*uk_159 + 5161477*uk_16 + 1532649*uk_160 + 10218313*uk_161 + 5132701*uk_162 + 2578177*uk_163 + 1295029*uk_164 + 3969*uk_17 + 630*uk_18 + 6867*uk_19 + 63*uk_2 + 1260*uk_20 + 8127*uk_21 + 13671*uk_22 + 6867*uk_23 + 100*uk_24 + 1090*uk_25 + 200*uk_26 + 1290*uk_27 + 2170*uk_28 + 1090*uk_29 + 10*uk_3 + 11881*uk_30 + 2180*uk_31 + 14061*uk_32 + 23653*uk_33 + 11881*uk_34 + 400*uk_35 + 2580*uk_36 + 4340*uk_37 + 2180*uk_38 + 16641*uk_39 + 109*uk_4 + 27993*uk_40 + 14061*uk_41 + 47089*uk_42 + 23653*uk_43 + 11881*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 22423066090*uk_47 + 244411420381*uk_48 + 44846132180*uk_49 + 20*uk_5 + 289257552561*uk_50 + 486580534153*uk_51 + 244411420381*uk_52 + 187944057*uk_53 + 29832390*uk_54 + 325173051*uk_55 + 59664780*uk_56 + 384837831*uk_57 + 647362863*uk_58 + 325173051*uk_59 + 129*uk_6 + 4735300*uk_60 + 51614770*uk_61 + 9470600*uk_62 + 61085370*uk_63 + 102756010*uk_64 + 51614770*uk_65 + 562600993*uk_66 + 103229540*uk_67 + 665830533*uk_68 + 1120040509*uk_69 + 217*uk_7 + 562600993*uk_70 + 18941200*uk_71 + 122170740*uk_72 + 205512020*uk_73 + 103229540*uk_74 + 788001273*uk_75 + 1325552529*uk_76 + 665830533*uk_77 + 2229805417*uk_78 + 1120040509*uk_79 + 109*uk_8 + 562600993*uk_80 + 250047*uk_81 + 39690*uk_82 + 432621*uk_83 + 79380*uk_84 + 512001*uk_85 + 861273*uk_86 + 432621*uk_87 + 6300*uk_88 + 68670*uk_89 + 2242306609*uk_9 + 12600*uk_90 + 81270*uk_91 + 136710*uk_92 + 68670*uk_93 + 748503*uk_94 + 137340*uk_95 + 885843*uk_96 + 1490139*uk_97 + 748503*uk_98 + 25200*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 198072*uk_100 + 328104*uk_101 + 15120*uk_102 + 1081143*uk_103 + 1790901*uk_104 + 82530*uk_105 + 2966607*uk_106 + 136710*uk_107 + 6300*uk_108 + 238328*uk_109 + 2935886*uk_11 + 38440*uk_110 + 92256*uk_111 + 503564*uk_112 + 834148*uk_113 + 38440*uk_114 + 6200*uk_115 + 14880*uk_116 + 81220*uk_117 + 134540*uk_118 + 6200*uk_119 + 473530*uk_12 + 35712*uk_120 + 194928*uk_121 + 322896*uk_122 + 14880*uk_123 + 1063982*uk_124 + 1762474*uk_125 + 81220*uk_126 + 2919518*uk_127 + 134540*uk_128 + 6200*uk_129 + 1136472*uk_13 + 1000*uk_130 + 2400*uk_131 + 13100*uk_132 + 21700*uk_133 + 1000*uk_134 + 5760*uk_135 + 31440*uk_136 + 52080*uk_137 + 2400*uk_138 + 171610*uk_139 + 6203243*uk_14 + 284270*uk_140 + 13100*uk_141 + 470890*uk_142 + 21700*uk_143 + 1000*uk_144 + 13824*uk_145 + 75456*uk_146 + 124992*uk_147 + 5760*uk_148 + 411864*uk_149 + 10275601*uk_15 + 682248*uk_150 + 31440*uk_151 + 1130136*uk_152 + 52080*uk_153 + 2400*uk_154 + 2248091*uk_155 + 3723937*uk_156 + 171610*uk_157 + 6168659*uk_158 + 284270*uk_159 + 473530*uk_16 + 13100*uk_160 + 10218313*uk_161 + 470890*uk_162 + 21700*uk_163 + 1000*uk_164 + 3969*uk_17 + 3906*uk_18 + 630*uk_19 + 63*uk_2 + 1512*uk_20 + 8253*uk_21 + 13671*uk_22 + 630*uk_23 + 3844*uk_24 + 620*uk_25 + 1488*uk_26 + 8122*uk_27 + 13454*uk_28 + 620*uk_29 + 62*uk_3 + 100*uk_30 + 240*uk_31 + 1310*uk_32 + 2170*uk_33 + 100*uk_34 + 576*uk_35 + 3144*uk_36 + 5208*uk_37 + 240*uk_38 + 17161*uk_39 + 10*uk_4 + 28427*uk_40 + 1310*uk_41 + 47089*uk_42 + 2170*uk_43 + 100*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 139023009758*uk_47 + 22423066090*uk_48 + 53815358616*uk_49 + 24*uk_5 + 293742165779*uk_50 + 486580534153*uk_51 + 22423066090*uk_52 + 187944057*uk_53 + 184960818*uk_54 + 29832390*uk_55 + 71597736*uk_56 + 390804309*uk_57 + 647362863*uk_58 + 29832390*uk_59 + 131*uk_6 + 182024932*uk_60 + 29358860*uk_61 + 70461264*uk_62 + 384601066*uk_63 + 637087262*uk_64 + 29358860*uk_65 + 4735300*uk_66 + 11364720*uk_67 + 62032430*uk_68 + 102756010*uk_69 + 217*uk_7 + 4735300*uk_70 + 27275328*uk_71 + 148877832*uk_72 + 246614424*uk_73 + 11364720*uk_74 + 812624833*uk_75 + 1346103731*uk_76 + 62032430*uk_77 + 2229805417*uk_78 + 102756010*uk_79 + 10*uk_8 + 4735300*uk_80 + 250047*uk_81 + 246078*uk_82 + 39690*uk_83 + 95256*uk_84 + 519939*uk_85 + 861273*uk_86 + 39690*uk_87 + 242172*uk_88 + 39060*uk_89 + 2242306609*uk_9 + 93744*uk_90 + 511686*uk_91 + 847602*uk_92 + 39060*uk_93 + 6300*uk_94 + 15120*uk_95 + 82530*uk_96 + 136710*uk_97 + 6300*uk_98 + 36288*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 167580*uk_100 + 273420*uk_101 + 78120*uk_102 + 1114407*uk_103 + 1818243*uk_104 + 519498*uk_105 + 2966607*uk_106 + 847602*uk_107 + 242172*uk_108 + 125*uk_109 + 236765*uk_11 + 1550*uk_110 + 500*uk_111 + 3325*uk_112 + 5425*uk_113 + 1550*uk_114 + 19220*uk_115 + 6200*uk_116 + 41230*uk_117 + 67270*uk_118 + 19220*uk_119 + 2935886*uk_12 + 2000*uk_120 + 13300*uk_121 + 21700*uk_122 + 6200*uk_123 + 88445*uk_124 + 144305*uk_125 + 41230*uk_126 + 235445*uk_127 + 67270*uk_128 + 19220*uk_129 + 947060*uk_13 + 238328*uk_130 + 76880*uk_131 + 511252*uk_132 + 834148*uk_133 + 238328*uk_134 + 24800*uk_135 + 164920*uk_136 + 269080*uk_137 + 76880*uk_138 + 1096718*uk_139 + 6297949*uk_14 + 1789382*uk_140 + 511252*uk_141 + 2919518*uk_142 + 834148*uk_143 + 238328*uk_144 + 8000*uk_145 + 53200*uk_146 + 86800*uk_147 + 24800*uk_148 + 353780*uk_149 + 10275601*uk_15 + 577220*uk_150 + 164920*uk_151 + 941780*uk_152 + 269080*uk_153 + 76880*uk_154 + 2352637*uk_155 + 3838513*uk_156 + 1096718*uk_157 + 6262837*uk_158 + 1789382*uk_159 + 2935886*uk_16 + 511252*uk_160 + 10218313*uk_161 + 2919518*uk_162 + 834148*uk_163 + 238328*uk_164 + 3969*uk_17 + 315*uk_18 + 3906*uk_19 + 63*uk_2 + 1260*uk_20 + 8379*uk_21 + 13671*uk_22 + 3906*uk_23 + 25*uk_24 + 310*uk_25 + 100*uk_26 + 665*uk_27 + 1085*uk_28 + 310*uk_29 + 5*uk_3 + 3844*uk_30 + 1240*uk_31 + 8246*uk_32 + 13454*uk_33 + 3844*uk_34 + 400*uk_35 + 2660*uk_36 + 4340*uk_37 + 1240*uk_38 + 17689*uk_39 + 62*uk_4 + 28861*uk_40 + 8246*uk_41 + 47089*uk_42 + 13454*uk_43 + 3844*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 11211533045*uk_47 + 139023009758*uk_48 + 44846132180*uk_49 + 20*uk_5 + 298226778997*uk_50 + 486580534153*uk_51 + 139023009758*uk_52 + 187944057*uk_53 + 14916195*uk_54 + 184960818*uk_55 + 59664780*uk_56 + 396770787*uk_57 + 647362863*uk_58 + 184960818*uk_59 + 133*uk_6 + 1183825*uk_60 + 14679430*uk_61 + 4735300*uk_62 + 31489745*uk_63 + 51378005*uk_64 + 14679430*uk_65 + 182024932*uk_66 + 58717720*uk_67 + 390472838*uk_68 + 637087262*uk_69 + 217*uk_7 + 182024932*uk_70 + 18941200*uk_71 + 125958980*uk_72 + 205512020*uk_73 + 58717720*uk_74 + 837627217*uk_75 + 1366654933*uk_76 + 390472838*uk_77 + 2229805417*uk_78 + 637087262*uk_79 + 62*uk_8 + 182024932*uk_80 + 250047*uk_81 + 19845*uk_82 + 246078*uk_83 + 79380*uk_84 + 527877*uk_85 + 861273*uk_86 + 246078*uk_87 + 1575*uk_88 + 19530*uk_89 + 2242306609*uk_9 + 6300*uk_90 + 41895*uk_91 + 68355*uk_92 + 19530*uk_93 + 242172*uk_94 + 78120*uk_95 + 519498*uk_96 + 847602*uk_97 + 242172*uk_98 + 25200*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 204120*uk_100 + 328104*uk_101 + 7560*uk_102 + 1148175*uk_103 + 1845585*uk_104 + 42525*uk_105 + 2966607*uk_106 + 68355*uk_107 + 1575*uk_108 + 1092727*uk_109 + 4877359*uk_11 + 53045*uk_110 + 254616*uk_111 + 1432215*uk_112 + 2302153*uk_113 + 53045*uk_114 + 2575*uk_115 + 12360*uk_116 + 69525*uk_117 + 111755*uk_118 + 2575*uk_119 + 236765*uk_12 + 59328*uk_120 + 333720*uk_121 + 536424*uk_122 + 12360*uk_123 + 1877175*uk_124 + 3017385*uk_125 + 69525*uk_126 + 4850167*uk_127 + 111755*uk_128 + 2575*uk_129 + 1136472*uk_13 + 125*uk_130 + 600*uk_131 + 3375*uk_132 + 5425*uk_133 + 125*uk_134 + 2880*uk_135 + 16200*uk_136 + 26040*uk_137 + 600*uk_138 + 91125*uk_139 + 6392655*uk_14 + 146475*uk_140 + 3375*uk_141 + 235445*uk_142 + 5425*uk_143 + 125*uk_144 + 13824*uk_145 + 77760*uk_146 + 124992*uk_147 + 2880*uk_148 + 437400*uk_149 + 10275601*uk_15 + 703080*uk_150 + 16200*uk_151 + 1130136*uk_152 + 26040*uk_153 + 600*uk_154 + 2460375*uk_155 + 3954825*uk_156 + 91125*uk_157 + 6357015*uk_158 + 146475*uk_159 + 236765*uk_16 + 3375*uk_160 + 10218313*uk_161 + 235445*uk_162 + 5425*uk_163 + 125*uk_164 + 3969*uk_17 + 6489*uk_18 + 315*uk_19 + 63*uk_2 + 1512*uk_20 + 8505*uk_21 + 13671*uk_22 + 315*uk_23 + 10609*uk_24 + 515*uk_25 + 2472*uk_26 + 13905*uk_27 + 22351*uk_28 + 515*uk_29 + 103*uk_3 + 25*uk_30 + 120*uk_31 + 675*uk_32 + 1085*uk_33 + 25*uk_34 + 576*uk_35 + 3240*uk_36 + 5208*uk_37 + 120*uk_38 + 18225*uk_39 + 5*uk_4 + 29295*uk_40 + 675*uk_41 + 47089*uk_42 + 1085*uk_43 + 25*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 230957580727*uk_47 + 11211533045*uk_48 + 53815358616*uk_49 + 24*uk_5 + 302711392215*uk_50 + 486580534153*uk_51 + 11211533045*uk_52 + 187944057*uk_53 + 307273617*uk_54 + 14916195*uk_55 + 71597736*uk_56 + 402737265*uk_57 + 647362863*uk_58 + 14916195*uk_59 + 135*uk_6 + 502367977*uk_60 + 24386795*uk_61 + 117056616*uk_62 + 658443465*uk_63 + 1058386903*uk_64 + 24386795*uk_65 + 1183825*uk_66 + 5682360*uk_67 + 31963275*uk_68 + 51378005*uk_69 + 217*uk_7 + 1183825*uk_70 + 27275328*uk_71 + 153423720*uk_72 + 246614424*uk_73 + 5682360*uk_74 + 863008425*uk_75 + 1387206135*uk_76 + 31963275*uk_77 + 2229805417*uk_78 + 51378005*uk_79 + 5*uk_8 + 1183825*uk_80 + 250047*uk_81 + 408807*uk_82 + 19845*uk_83 + 95256*uk_84 + 535815*uk_85 + 861273*uk_86 + 19845*uk_87 + 668367*uk_88 + 32445*uk_89 + 2242306609*uk_9 + 155736*uk_90 + 876015*uk_91 + 1408113*uk_92 + 32445*uk_93 + 1575*uk_94 + 7560*uk_95 + 42525*uk_96 + 68355*uk_97 + 1575*uk_98 + 36288*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 172620*uk_100 + 273420*uk_101 + 129780*uk_102 + 1182447*uk_103 + 1872927*uk_104 + 888993*uk_105 + 2966607*uk_106 + 1408113*uk_107 + 668367*uk_108 + 681472*uk_109 + 4167064*uk_11 + 797632*uk_110 + 154880*uk_111 + 1060928*uk_112 + 1680448*uk_113 + 797632*uk_114 + 933592*uk_115 + 181280*uk_116 + 1241768*uk_117 + 1966888*uk_118 + 933592*uk_119 + 4877359*uk_12 + 35200*uk_120 + 241120*uk_121 + 381920*uk_122 + 181280*uk_123 + 1651672*uk_124 + 2616152*uk_125 + 1241768*uk_126 + 4143832*uk_127 + 1966888*uk_128 + 933592*uk_129 + 947060*uk_13 + 1092727*uk_130 + 212180*uk_131 + 1453433*uk_132 + 2302153*uk_133 + 1092727*uk_134 + 41200*uk_135 + 282220*uk_136 + 447020*uk_137 + 212180*uk_138 + 1933207*uk_139 + 6487361*uk_14 + 3062087*uk_140 + 1453433*uk_141 + 4850167*uk_142 + 2302153*uk_143 + 1092727*uk_144 + 8000*uk_145 + 54800*uk_146 + 86800*uk_147 + 41200*uk_148 + 375380*uk_149 + 10275601*uk_15 + 594580*uk_150 + 282220*uk_151 + 941780*uk_152 + 447020*uk_153 + 212180*uk_154 + 2571353*uk_155 + 4072873*uk_156 + 1933207*uk_157 + 6451193*uk_158 + 3062087*uk_159 + 4877359*uk_16 + 1453433*uk_160 + 10218313*uk_161 + 4850167*uk_162 + 2302153*uk_163 + 1092727*uk_164 + 3969*uk_17 + 5544*uk_18 + 6489*uk_19 + 63*uk_2 + 1260*uk_20 + 8631*uk_21 + 13671*uk_22 + 6489*uk_23 + 7744*uk_24 + 9064*uk_25 + 1760*uk_26 + 12056*uk_27 + 19096*uk_28 + 9064*uk_29 + 88*uk_3 + 10609*uk_30 + 2060*uk_31 + 14111*uk_32 + 22351*uk_33 + 10609*uk_34 + 400*uk_35 + 2740*uk_36 + 4340*uk_37 + 2060*uk_38 + 18769*uk_39 + 103*uk_4 + 29729*uk_40 + 14111*uk_41 + 47089*uk_42 + 22351*uk_43 + 10609*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 197322981592*uk_47 + 230957580727*uk_48 + 44846132180*uk_49 + 20*uk_5 + 307196005433*uk_50 + 486580534153*uk_51 + 230957580727*uk_52 + 187944057*uk_53 + 262525032*uk_54 + 307273617*uk_55 + 59664780*uk_56 + 408703743*uk_57 + 647362863*uk_58 + 307273617*uk_59 + 137*uk_6 + 366701632*uk_60 + 429207592*uk_61 + 83341280*uk_62 + 570887768*uk_63 + 904252888*uk_64 + 429207592*uk_65 + 502367977*uk_66 + 97547180*uk_67 + 668198183*uk_68 + 1058386903*uk_69 + 217*uk_7 + 502367977*uk_70 + 18941200*uk_71 + 129747220*uk_72 + 205512020*uk_73 + 97547180*uk_74 + 888768457*uk_75 + 1407757337*uk_76 + 668198183*uk_77 + 2229805417*uk_78 + 1058386903*uk_79 + 103*uk_8 + 502367977*uk_80 + 250047*uk_81 + 349272*uk_82 + 408807*uk_83 + 79380*uk_84 + 543753*uk_85 + 861273*uk_86 + 408807*uk_87 + 487872*uk_88 + 571032*uk_89 + 2242306609*uk_9 + 110880*uk_90 + 759528*uk_91 + 1203048*uk_92 + 571032*uk_93 + 668367*uk_94 + 129780*uk_95 + 888993*uk_96 + 1408113*uk_97 + 668367*uk_98 + 25200*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 175140*uk_100 + 273420*uk_101 + 110880*uk_102 + 1217223*uk_103 + 1900269*uk_104 + 770616*uk_105 + 2966607*uk_106 + 1203048*uk_107 + 487872*uk_108 + 804357*uk_109 + 4403829*uk_11 + 761112*uk_110 + 172980*uk_111 + 1202211*uk_112 + 1876833*uk_113 + 761112*uk_114 + 720192*uk_115 + 163680*uk_116 + 1137576*uk_117 + 1775928*uk_118 + 720192*uk_119 + 4167064*uk_12 + 37200*uk_120 + 258540*uk_121 + 403620*uk_122 + 163680*uk_123 + 1796853*uk_124 + 2805159*uk_125 + 1137576*uk_126 + 4379277*uk_127 + 1775928*uk_128 + 720192*uk_129 + 947060*uk_13 + 681472*uk_130 + 154880*uk_131 + 1076416*uk_132 + 1680448*uk_133 + 681472*uk_134 + 35200*uk_135 + 244640*uk_136 + 381920*uk_137 + 154880*uk_138 + 1700248*uk_139 + 6582067*uk_14 + 2654344*uk_140 + 1076416*uk_141 + 4143832*uk_142 + 1680448*uk_143 + 681472*uk_144 + 8000*uk_145 + 55600*uk_146 + 86800*uk_147 + 35200*uk_148 + 386420*uk_149 + 10275601*uk_15 + 603260*uk_150 + 244640*uk_151 + 941780*uk_152 + 381920*uk_153 + 154880*uk_154 + 2685619*uk_155 + 4192657*uk_156 + 1700248*uk_157 + 6545371*uk_158 + 2654344*uk_159 + 4167064*uk_16 + 1076416*uk_160 + 10218313*uk_161 + 4143832*uk_162 + 1680448*uk_163 + 681472*uk_164 + 3969*uk_17 + 5859*uk_18 + 5544*uk_19 + 63*uk_2 + 1260*uk_20 + 8757*uk_21 + 13671*uk_22 + 5544*uk_23 + 8649*uk_24 + 8184*uk_25 + 1860*uk_26 + 12927*uk_27 + 20181*uk_28 + 8184*uk_29 + 93*uk_3 + 7744*uk_30 + 1760*uk_31 + 12232*uk_32 + 19096*uk_33 + 7744*uk_34 + 400*uk_35 + 2780*uk_36 + 4340*uk_37 + 1760*uk_38 + 19321*uk_39 + 88*uk_4 + 30163*uk_40 + 12232*uk_41 + 47089*uk_42 + 19096*uk_43 + 7744*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 208534514637*uk_47 + 197322981592*uk_48 + 44846132180*uk_49 + 20*uk_5 + 311680618651*uk_50 + 486580534153*uk_51 + 197322981592*uk_52 + 187944057*uk_53 + 277441227*uk_54 + 262525032*uk_55 + 59664780*uk_56 + 414670221*uk_57 + 647362863*uk_58 + 262525032*uk_59 + 139*uk_6 + 409556097*uk_60 + 387536952*uk_61 + 88076580*uk_62 + 612132231*uk_63 + 955630893*uk_64 + 387536952*uk_65 + 366701632*uk_66 + 83341280*uk_67 + 579221896*uk_68 + 904252888*uk_69 + 217*uk_7 + 366701632*uk_70 + 18941200*uk_71 + 131641340*uk_72 + 205512020*uk_73 + 83341280*uk_74 + 914907313*uk_75 + 1428308539*uk_76 + 579221896*uk_77 + 2229805417*uk_78 + 904252888*uk_79 + 88*uk_8 + 366701632*uk_80 + 250047*uk_81 + 369117*uk_82 + 349272*uk_83 + 79380*uk_84 + 551691*uk_85 + 861273*uk_86 + 349272*uk_87 + 544887*uk_88 + 515592*uk_89 + 2242306609*uk_9 + 117180*uk_90 + 814401*uk_91 + 1271403*uk_92 + 515592*uk_93 + 487872*uk_94 + 110880*uk_95 + 770616*uk_96 + 1203048*uk_97 + 487872*uk_98 + 25200*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 177660*uk_100 + 273420*uk_101 + 117180*uk_102 + 1252503*uk_103 + 1927611*uk_104 + 826119*uk_105 + 2966607*uk_106 + 1271403*uk_107 + 544887*uk_108 + 1643032*uk_109 + 5587654*uk_11 + 1294932*uk_110 + 278480*uk_111 + 1963284*uk_112 + 3021508*uk_113 + 1294932*uk_114 + 1020582*uk_115 + 219480*uk_116 + 1547334*uk_117 + 2381358*uk_118 + 1020582*uk_119 + 4403829*uk_12 + 47200*uk_120 + 332760*uk_121 + 512120*uk_122 + 219480*uk_123 + 2345958*uk_124 + 3610446*uk_125 + 1547334*uk_126 + 5556502*uk_127 + 2381358*uk_128 + 1020582*uk_129 + 947060*uk_13 + 804357*uk_130 + 172980*uk_131 + 1219509*uk_132 + 1876833*uk_133 + 804357*uk_134 + 37200*uk_135 + 262260*uk_136 + 403620*uk_137 + 172980*uk_138 + 1848933*uk_139 + 6676773*uk_14 + 2845521*uk_140 + 1219509*uk_141 + 4379277*uk_142 + 1876833*uk_143 + 804357*uk_144 + 8000*uk_145 + 56400*uk_146 + 86800*uk_147 + 37200*uk_148 + 397620*uk_149 + 10275601*uk_15 + 611940*uk_150 + 262260*uk_151 + 941780*uk_152 + 403620*uk_153 + 172980*uk_154 + 2803221*uk_155 + 4314177*uk_156 + 1848933*uk_157 + 6639549*uk_158 + 2845521*uk_159 + 4403829*uk_16 + 1219509*uk_160 + 10218313*uk_161 + 4379277*uk_162 + 1876833*uk_163 + 804357*uk_164 + 3969*uk_17 + 7434*uk_18 + 5859*uk_19 + 63*uk_2 + 1260*uk_20 + 8883*uk_21 + 13671*uk_22 + 5859*uk_23 + 13924*uk_24 + 10974*uk_25 + 2360*uk_26 + 16638*uk_27 + 25606*uk_28 + 10974*uk_29 + 118*uk_3 + 8649*uk_30 + 1860*uk_31 + 13113*uk_32 + 20181*uk_33 + 8649*uk_34 + 400*uk_35 + 2820*uk_36 + 4340*uk_37 + 1860*uk_38 + 19881*uk_39 + 93*uk_4 + 30597*uk_40 + 13113*uk_41 + 47089*uk_42 + 20181*uk_43 + 8649*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 264592179862*uk_47 + 208534514637*uk_48 + 44846132180*uk_49 + 20*uk_5 + 316165231869*uk_50 + 486580534153*uk_51 + 208534514637*uk_52 + 187944057*uk_53 + 352022202*uk_54 + 277441227*uk_55 + 59664780*uk_56 + 420636699*uk_57 + 647362863*uk_58 + 277441227*uk_59 + 141*uk_6 + 659343172*uk_60 + 519651822*uk_61 + 111753080*uk_62 + 787859214*uk_63 + 1212520918*uk_64 + 519651822*uk_65 + 409556097*uk_66 + 88076580*uk_67 + 620939889*uk_68 + 955630893*uk_69 + 217*uk_7 + 409556097*uk_70 + 18941200*uk_71 + 133535460*uk_72 + 205512020*uk_73 + 88076580*uk_74 + 941424993*uk_75 + 1448859741*uk_76 + 620939889*uk_77 + 2229805417*uk_78 + 955630893*uk_79 + 93*uk_8 + 409556097*uk_80 + 250047*uk_81 + 468342*uk_82 + 369117*uk_83 + 79380*uk_84 + 559629*uk_85 + 861273*uk_86 + 369117*uk_87 + 877212*uk_88 + 691362*uk_89 + 2242306609*uk_9 + 148680*uk_90 + 1048194*uk_91 + 1613178*uk_92 + 691362*uk_93 + 544887*uk_94 + 117180*uk_95 + 826119*uk_96 + 1271403*uk_97 + 544887*uk_98 + 25200*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 144144*uk_100 + 218736*uk_101 + 118944*uk_102 + 1288287*uk_103 + 1954953*uk_104 + 1063062*uk_105 + 2966607*uk_106 + 1613178*uk_107 + 877212*uk_108 + 8000*uk_109 + 947060*uk_11 + 47200*uk_110 + 6400*uk_111 + 57200*uk_112 + 86800*uk_113 + 47200*uk_114 + 278480*uk_115 + 37760*uk_116 + 337480*uk_117 + 512120*uk_118 + 278480*uk_119 + 5587654*uk_12 + 5120*uk_120 + 45760*uk_121 + 69440*uk_122 + 37760*uk_123 + 408980*uk_124 + 620620*uk_125 + 337480*uk_126 + 941780*uk_127 + 512120*uk_128 + 278480*uk_129 + 757648*uk_13 + 1643032*uk_130 + 222784*uk_131 + 1991132*uk_132 + 3021508*uk_133 + 1643032*uk_134 + 30208*uk_135 + 269984*uk_136 + 409696*uk_137 + 222784*uk_138 + 2412982*uk_139 + 6771479*uk_14 + 3661658*uk_140 + 1991132*uk_141 + 5556502*uk_142 + 3021508*uk_143 + 1643032*uk_144 + 4096*uk_145 + 36608*uk_146 + 55552*uk_147 + 30208*uk_148 + 327184*uk_149 + 10275601*uk_15 + 496496*uk_150 + 269984*uk_151 + 753424*uk_152 + 409696*uk_153 + 222784*uk_154 + 2924207*uk_155 + 4437433*uk_156 + 2412982*uk_157 + 6733727*uk_158 + 3661658*uk_159 + 5587654*uk_16 + 1991132*uk_160 + 10218313*uk_161 + 5556502*uk_162 + 3021508*uk_163 + 1643032*uk_164 + 3969*uk_17 + 1260*uk_18 + 7434*uk_19 + 63*uk_2 + 1008*uk_20 + 9009*uk_21 + 13671*uk_22 + 7434*uk_23 + 400*uk_24 + 2360*uk_25 + 320*uk_26 + 2860*uk_27 + 4340*uk_28 + 2360*uk_29 + 20*uk_3 + 13924*uk_30 + 1888*uk_31 + 16874*uk_32 + 25606*uk_33 + 13924*uk_34 + 256*uk_35 + 2288*uk_36 + 3472*uk_37 + 1888*uk_38 + 20449*uk_39 + 118*uk_4 + 31031*uk_40 + 16874*uk_41 + 47089*uk_42 + 25606*uk_43 + 13924*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 44846132180*uk_47 + 264592179862*uk_48 + 35876905744*uk_49 + 16*uk_5 + 320649845087*uk_50 + 486580534153*uk_51 + 264592179862*uk_52 + 187944057*uk_53 + 59664780*uk_54 + 352022202*uk_55 + 47731824*uk_56 + 426603177*uk_57 + 647362863*uk_58 + 352022202*uk_59 + 143*uk_6 + 18941200*uk_60 + 111753080*uk_61 + 15152960*uk_62 + 135429580*uk_63 + 205512020*uk_64 + 111753080*uk_65 + 659343172*uk_66 + 89402464*uk_67 + 799034522*uk_68 + 1212520918*uk_69 + 217*uk_7 + 659343172*uk_70 + 12122368*uk_71 + 108343664*uk_72 + 164409616*uk_73 + 89402464*uk_74 + 968321497*uk_75 + 1469410943*uk_76 + 799034522*uk_77 + 2229805417*uk_78 + 1212520918*uk_79 + 118*uk_8 + 659343172*uk_80 + 250047*uk_81 + 79380*uk_82 + 468342*uk_83 + 63504*uk_84 + 567567*uk_85 + 861273*uk_86 + 468342*uk_87 + 25200*uk_88 + 148680*uk_89 + 2242306609*uk_9 + 20160*uk_90 + 180180*uk_91 + 273420*uk_92 + 148680*uk_93 + 877212*uk_94 + 118944*uk_95 + 1063062*uk_96 + 1613178*uk_97 + 877212*uk_98 + 16128*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 182700*uk_100 + 273420*uk_101 + 25200*uk_102 + 1324575*uk_103 + 1982295*uk_104 + 182700*uk_105 + 2966607*uk_106 + 273420*uk_107 + 25200*uk_108 + 571787*uk_109 + 3930299*uk_11 + 137780*uk_110 + 137780*uk_111 + 998905*uk_112 + 1494913*uk_113 + 137780*uk_114 + 33200*uk_115 + 33200*uk_116 + 240700*uk_117 + 360220*uk_118 + 33200*uk_119 + 947060*uk_12 + 33200*uk_120 + 240700*uk_121 + 360220*uk_122 + 33200*uk_123 + 1745075*uk_124 + 2611595*uk_125 + 240700*uk_126 + 3908387*uk_127 + 360220*uk_128 + 33200*uk_129 + 947060*uk_13 + 8000*uk_130 + 8000*uk_131 + 58000*uk_132 + 86800*uk_133 + 8000*uk_134 + 8000*uk_135 + 58000*uk_136 + 86800*uk_137 + 8000*uk_138 + 420500*uk_139 + 6866185*uk_14 + 629300*uk_140 + 58000*uk_141 + 941780*uk_142 + 86800*uk_143 + 8000*uk_144 + 8000*uk_145 + 58000*uk_146 + 86800*uk_147 + 8000*uk_148 + 420500*uk_149 + 10275601*uk_15 + 629300*uk_150 + 58000*uk_151 + 941780*uk_152 + 86800*uk_153 + 8000*uk_154 + 3048625*uk_155 + 4562425*uk_156 + 420500*uk_157 + 6827905*uk_158 + 629300*uk_159 + 947060*uk_16 + 58000*uk_160 + 10218313*uk_161 + 941780*uk_162 + 86800*uk_163 + 8000*uk_164 + 3969*uk_17 + 5229*uk_18 + 1260*uk_19 + 63*uk_2 + 1260*uk_20 + 9135*uk_21 + 13671*uk_22 + 1260*uk_23 + 6889*uk_24 + 1660*uk_25 + 1660*uk_26 + 12035*uk_27 + 18011*uk_28 + 1660*uk_29 + 83*uk_3 + 400*uk_30 + 400*uk_31 + 2900*uk_32 + 4340*uk_33 + 400*uk_34 + 400*uk_35 + 2900*uk_36 + 4340*uk_37 + 400*uk_38 + 21025*uk_39 + 20*uk_4 + 31465*uk_40 + 2900*uk_41 + 47089*uk_42 + 4340*uk_43 + 400*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 186111448547*uk_47 + 44846132180*uk_48 + 44846132180*uk_49 + 20*uk_5 + 325134458305*uk_50 + 486580534153*uk_51 + 44846132180*uk_52 + 187944057*uk_53 + 247608837*uk_54 + 59664780*uk_55 + 59664780*uk_56 + 432569655*uk_57 + 647362863*uk_58 + 59664780*uk_59 + 145*uk_6 + 326214817*uk_60 + 78605980*uk_61 + 78605980*uk_62 + 569893355*uk_63 + 852874883*uk_64 + 78605980*uk_65 + 18941200*uk_66 + 18941200*uk_67 + 137323700*uk_68 + 205512020*uk_69 + 217*uk_7 + 18941200*uk_70 + 18941200*uk_71 + 137323700*uk_72 + 205512020*uk_73 + 18941200*uk_74 + 995596825*uk_75 + 1489962145*uk_76 + 137323700*uk_77 + 2229805417*uk_78 + 205512020*uk_79 + 20*uk_8 + 18941200*uk_80 + 250047*uk_81 + 329427*uk_82 + 79380*uk_83 + 79380*uk_84 + 575505*uk_85 + 861273*uk_86 + 79380*uk_87 + 434007*uk_88 + 104580*uk_89 + 2242306609*uk_9 + 104580*uk_90 + 758205*uk_91 + 1134693*uk_92 + 104580*uk_93 + 25200*uk_94 + 25200*uk_95 + 182700*uk_96 + 273420*uk_97 + 25200*uk_98 + 25200*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 148176*uk_100 + 218736*uk_101 + 83664*uk_102 + 1361367*uk_103 + 2009637*uk_104 + 768663*uk_105 + 2966607*uk_106 + 1134693*uk_107 + 434007*uk_108 + 6859*uk_109 + 899707*uk_11 + 29963*uk_110 + 5776*uk_111 + 53067*uk_112 + 78337*uk_113 + 29963*uk_114 + 130891*uk_115 + 25232*uk_116 + 231819*uk_117 + 342209*uk_118 + 130891*uk_119 + 3930299*uk_12 + 4864*uk_120 + 44688*uk_121 + 65968*uk_122 + 25232*uk_123 + 410571*uk_124 + 606081*uk_125 + 231819*uk_126 + 894691*uk_127 + 342209*uk_128 + 130891*uk_129 + 757648*uk_13 + 571787*uk_130 + 110224*uk_131 + 1012683*uk_132 + 1494913*uk_133 + 571787*uk_134 + 21248*uk_135 + 195216*uk_136 + 288176*uk_137 + 110224*uk_138 + 1793547*uk_139 + 6960891*uk_14 + 2647617*uk_140 + 1012683*uk_141 + 3908387*uk_142 + 1494913*uk_143 + 571787*uk_144 + 4096*uk_145 + 37632*uk_146 + 55552*uk_147 + 21248*uk_148 + 345744*uk_149 + 10275601*uk_15 + 510384*uk_150 + 195216*uk_151 + 753424*uk_152 + 288176*uk_153 + 110224*uk_154 + 3176523*uk_155 + 4689153*uk_156 + 1793547*uk_157 + 6922083*uk_158 + 2647617*uk_159 + 3930299*uk_16 + 1012683*uk_160 + 10218313*uk_161 + 3908387*uk_162 + 1494913*uk_163 + 571787*uk_164 + 3969*uk_17 + 1197*uk_18 + 5229*uk_19 + 63*uk_2 + 1008*uk_20 + 9261*uk_21 + 13671*uk_22 + 5229*uk_23 + 361*uk_24 + 1577*uk_25 + 304*uk_26 + 2793*uk_27 + 4123*uk_28 + 1577*uk_29 + 19*uk_3 + 6889*uk_30 + 1328*uk_31 + 12201*uk_32 + 18011*uk_33 + 6889*uk_34 + 256*uk_35 + 2352*uk_36 + 3472*uk_37 + 1328*uk_38 + 21609*uk_39 + 83*uk_4 + 31899*uk_40 + 12201*uk_41 + 47089*uk_42 + 18011*uk_43 + 6889*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 42603825571*uk_47 + 186111448547*uk_48 + 35876905744*uk_49 + 16*uk_5 + 329619071523*uk_50 + 486580534153*uk_51 + 186111448547*uk_52 + 187944057*uk_53 + 56681541*uk_54 + 247608837*uk_55 + 47731824*uk_56 + 438536133*uk_57 + 647362863*uk_58 + 247608837*uk_59 + 147*uk_6 + 17094433*uk_60 + 74675681*uk_61 + 14395312*uk_62 + 132256929*uk_63 + 195236419*uk_64 + 74675681*uk_65 + 326214817*uk_66 + 62884784*uk_67 + 577753953*uk_68 + 852874883*uk_69 + 217*uk_7 + 326214817*uk_70 + 12122368*uk_71 + 111374256*uk_72 + 164409616*uk_73 + 62884784*uk_74 + 1023250977*uk_75 + 1510513347*uk_76 + 577753953*uk_77 + 2229805417*uk_78 + 852874883*uk_79 + 83*uk_8 + 326214817*uk_80 + 250047*uk_81 + 75411*uk_82 + 329427*uk_83 + 63504*uk_84 + 583443*uk_85 + 861273*uk_86 + 329427*uk_87 + 22743*uk_88 + 99351*uk_89 + 2242306609*uk_9 + 19152*uk_90 + 175959*uk_91 + 259749*uk_92 + 99351*uk_93 + 434007*uk_94 + 83664*uk_95 + 768663*uk_96 + 1134693*uk_97 + 434007*uk_98 + 16128*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 187740*uk_100 + 273420*uk_101 + 23940*uk_102 + 1398663*uk_103 + 2036979*uk_104 + 178353*uk_105 + 2966607*uk_106 + 259749*uk_107 + 22743*uk_108 + 1728000*uk_109 + 5682360*uk_11 + 273600*uk_110 + 288000*uk_111 + 2145600*uk_112 + 3124800*uk_113 + 273600*uk_114 + 43320*uk_115 + 45600*uk_116 + 339720*uk_117 + 494760*uk_118 + 43320*uk_119 + 899707*uk_12 + 48000*uk_120 + 357600*uk_121 + 520800*uk_122 + 45600*uk_123 + 2664120*uk_124 + 3879960*uk_125 + 339720*uk_126 + 5650680*uk_127 + 494760*uk_128 + 43320*uk_129 + 947060*uk_13 + 6859*uk_130 + 7220*uk_131 + 53789*uk_132 + 78337*uk_133 + 6859*uk_134 + 7600*uk_135 + 56620*uk_136 + 82460*uk_137 + 7220*uk_138 + 421819*uk_139 + 7055597*uk_14 + 614327*uk_140 + 53789*uk_141 + 894691*uk_142 + 78337*uk_143 + 6859*uk_144 + 8000*uk_145 + 59600*uk_146 + 86800*uk_147 + 7600*uk_148 + 444020*uk_149 + 10275601*uk_15 + 646660*uk_150 + 56620*uk_151 + 941780*uk_152 + 82460*uk_153 + 7220*uk_154 + 3307949*uk_155 + 4817617*uk_156 + 421819*uk_157 + 7016261*uk_158 + 614327*uk_159 + 899707*uk_16 + 53789*uk_160 + 10218313*uk_161 + 894691*uk_162 + 78337*uk_163 + 6859*uk_164 + 3969*uk_17 + 7560*uk_18 + 1197*uk_19 + 63*uk_2 + 1260*uk_20 + 9387*uk_21 + 13671*uk_22 + 1197*uk_23 + 14400*uk_24 + 2280*uk_25 + 2400*uk_26 + 17880*uk_27 + 26040*uk_28 + 2280*uk_29 + 120*uk_3 + 361*uk_30 + 380*uk_31 + 2831*uk_32 + 4123*uk_33 + 361*uk_34 + 400*uk_35 + 2980*uk_36 + 4340*uk_37 + 380*uk_38 + 22201*uk_39 + 19*uk_4 + 32333*uk_40 + 2831*uk_41 + 47089*uk_42 + 4123*uk_43 + 361*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 269076793080*uk_47 + 42603825571*uk_48 + 44846132180*uk_49 + 20*uk_5 + 334103684741*uk_50 + 486580534153*uk_51 + 42603825571*uk_52 + 187944057*uk_53 + 357988680*uk_54 + 56681541*uk_55 + 59664780*uk_56 + 444502611*uk_57 + 647362863*uk_58 + 56681541*uk_59 + 149*uk_6 + 681883200*uk_60 + 107964840*uk_61 + 113647200*uk_62 + 846671640*uk_63 + 1233072120*uk_64 + 107964840*uk_65 + 17094433*uk_66 + 17994140*uk_67 + 134056343*uk_68 + 195236419*uk_69 + 217*uk_7 + 17094433*uk_70 + 18941200*uk_71 + 141111940*uk_72 + 205512020*uk_73 + 17994140*uk_74 + 1051283953*uk_75 + 1531064549*uk_76 + 134056343*uk_77 + 2229805417*uk_78 + 195236419*uk_79 + 19*uk_8 + 17094433*uk_80 + 250047*uk_81 + 476280*uk_82 + 75411*uk_83 + 79380*uk_84 + 591381*uk_85 + 861273*uk_86 + 75411*uk_87 + 907200*uk_88 + 143640*uk_89 + 2242306609*uk_9 + 151200*uk_90 + 1126440*uk_91 + 1640520*uk_92 + 143640*uk_93 + 22743*uk_94 + 23940*uk_95 + 178353*uk_96 + 259749*uk_97 + 22743*uk_98 + 25200*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 152208*uk_100 + 218736*uk_101 + 120960*uk_102 + 1436463*uk_103 + 2064321*uk_104 + 1141560*uk_105 + 2966607*uk_106 + 1640520*uk_107 + 907200*uk_108 + 729000*uk_109 + 4261770*uk_11 + 972000*uk_110 + 129600*uk_111 + 1223100*uk_112 + 1757700*uk_113 + 972000*uk_114 + 1296000*uk_115 + 172800*uk_116 + 1630800*uk_117 + 2343600*uk_118 + 1296000*uk_119 + 5682360*uk_12 + 23040*uk_120 + 217440*uk_121 + 312480*uk_122 + 172800*uk_123 + 2052090*uk_124 + 2949030*uk_125 + 1630800*uk_126 + 4238010*uk_127 + 2343600*uk_128 + 1296000*uk_129 + 757648*uk_13 + 1728000*uk_130 + 230400*uk_131 + 2174400*uk_132 + 3124800*uk_133 + 1728000*uk_134 + 30720*uk_135 + 289920*uk_136 + 416640*uk_137 + 230400*uk_138 + 2736120*uk_139 + 7150303*uk_14 + 3932040*uk_140 + 2174400*uk_141 + 5650680*uk_142 + 3124800*uk_143 + 1728000*uk_144 + 4096*uk_145 + 38656*uk_146 + 55552*uk_147 + 30720*uk_148 + 364816*uk_149 + 10275601*uk_15 + 524272*uk_150 + 289920*uk_151 + 753424*uk_152 + 416640*uk_153 + 230400*uk_154 + 3442951*uk_155 + 4947817*uk_156 + 2736120*uk_157 + 7110439*uk_158 + 3932040*uk_159 + 5682360*uk_16 + 2174400*uk_160 + 10218313*uk_161 + 5650680*uk_162 + 3124800*uk_163 + 1728000*uk_164 + 3969*uk_17 + 5670*uk_18 + 7560*uk_19 + 63*uk_2 + 1008*uk_20 + 9513*uk_21 + 13671*uk_22 + 7560*uk_23 + 8100*uk_24 + 10800*uk_25 + 1440*uk_26 + 13590*uk_27 + 19530*uk_28 + 10800*uk_29 + 90*uk_3 + 14400*uk_30 + 1920*uk_31 + 18120*uk_32 + 26040*uk_33 + 14400*uk_34 + 256*uk_35 + 2416*uk_36 + 3472*uk_37 + 1920*uk_38 + 22801*uk_39 + 120*uk_4 + 32767*uk_40 + 18120*uk_41 + 47089*uk_42 + 26040*uk_43 + 14400*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 201807594810*uk_47 + 269076793080*uk_48 + 35876905744*uk_49 + 16*uk_5 + 338588297959*uk_50 + 486580534153*uk_51 + 269076793080*uk_52 + 187944057*uk_53 + 268491510*uk_54 + 357988680*uk_55 + 47731824*uk_56 + 450469089*uk_57 + 647362863*uk_58 + 357988680*uk_59 + 151*uk_6 + 383559300*uk_60 + 511412400*uk_61 + 68188320*uk_62 + 643527270*uk_63 + 924804090*uk_64 + 511412400*uk_65 + 681883200*uk_66 + 90917760*uk_67 + 858036360*uk_68 + 1233072120*uk_69 + 217*uk_7 + 681883200*uk_70 + 12122368*uk_71 + 114404848*uk_72 + 164409616*uk_73 + 90917760*uk_74 + 1079695753*uk_75 + 1551615751*uk_76 + 858036360*uk_77 + 2229805417*uk_78 + 1233072120*uk_79 + 120*uk_8 + 681883200*uk_80 + 250047*uk_81 + 357210*uk_82 + 476280*uk_83 + 63504*uk_84 + 599319*uk_85 + 861273*uk_86 + 476280*uk_87 + 510300*uk_88 + 680400*uk_89 + 2242306609*uk_9 + 90720*uk_90 + 856170*uk_91 + 1230390*uk_92 + 680400*uk_93 + 907200*uk_94 + 120960*uk_95 + 1141560*uk_96 + 1640520*uk_97 + 907200*uk_98 + 16128*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 154224*uk_100 + 218736*uk_101 + 90720*uk_102 + 1474767*uk_103 + 2091663*uk_104 + 867510*uk_105 + 2966607*uk_106 + 1230390*uk_107 + 510300*uk_108 + 438976*uk_109 + 3598828*uk_11 + 519840*uk_110 + 92416*uk_111 + 883728*uk_112 + 1253392*uk_113 + 519840*uk_114 + 615600*uk_115 + 109440*uk_116 + 1046520*uk_117 + 1484280*uk_118 + 615600*uk_119 + 4261770*uk_12 + 19456*uk_120 + 186048*uk_121 + 263872*uk_122 + 109440*uk_123 + 1779084*uk_124 + 2523276*uk_125 + 1046520*uk_126 + 3578764*uk_127 + 1484280*uk_128 + 615600*uk_129 + 757648*uk_13 + 729000*uk_130 + 129600*uk_131 + 1239300*uk_132 + 1757700*uk_133 + 729000*uk_134 + 23040*uk_135 + 220320*uk_136 + 312480*uk_137 + 129600*uk_138 + 2106810*uk_139 + 7245009*uk_14 + 2988090*uk_140 + 1239300*uk_141 + 4238010*uk_142 + 1757700*uk_143 + 729000*uk_144 + 4096*uk_145 + 39168*uk_146 + 55552*uk_147 + 23040*uk_148 + 374544*uk_149 + 10275601*uk_15 + 531216*uk_150 + 220320*uk_151 + 753424*uk_152 + 312480*uk_153 + 129600*uk_154 + 3581577*uk_155 + 5079753*uk_156 + 2106810*uk_157 + 7204617*uk_158 + 2988090*uk_159 + 4261770*uk_16 + 1239300*uk_160 + 10218313*uk_161 + 4238010*uk_162 + 1757700*uk_163 + 729000*uk_164 + 3969*uk_17 + 4788*uk_18 + 5670*uk_19 + 63*uk_2 + 1008*uk_20 + 9639*uk_21 + 13671*uk_22 + 5670*uk_23 + 5776*uk_24 + 6840*uk_25 + 1216*uk_26 + 11628*uk_27 + 16492*uk_28 + 6840*uk_29 + 76*uk_3 + 8100*uk_30 + 1440*uk_31 + 13770*uk_32 + 19530*uk_33 + 8100*uk_34 + 256*uk_35 + 2448*uk_36 + 3472*uk_37 + 1440*uk_38 + 23409*uk_39 + 90*uk_4 + 33201*uk_40 + 13770*uk_41 + 47089*uk_42 + 19530*uk_43 + 8100*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 170415302284*uk_47 + 201807594810*uk_48 + 35876905744*uk_49 + 16*uk_5 + 343072911177*uk_50 + 486580534153*uk_51 + 201807594810*uk_52 + 187944057*uk_53 + 226726164*uk_54 + 268491510*uk_55 + 47731824*uk_56 + 456435567*uk_57 + 647362863*uk_58 + 268491510*uk_59 + 153*uk_6 + 273510928*uk_60 + 323894520*uk_61 + 57581248*uk_62 + 550620684*uk_63 + 780945676*uk_64 + 323894520*uk_65 + 383559300*uk_66 + 68188320*uk_67 + 652050810*uk_68 + 924804090*uk_69 + 217*uk_7 + 383559300*uk_70 + 12122368*uk_71 + 115920144*uk_72 + 164409616*uk_73 + 68188320*uk_74 + 1108486377*uk_75 + 1572166953*uk_76 + 652050810*uk_77 + 2229805417*uk_78 + 924804090*uk_79 + 90*uk_8 + 383559300*uk_80 + 250047*uk_81 + 301644*uk_82 + 357210*uk_83 + 63504*uk_84 + 607257*uk_85 + 861273*uk_86 + 357210*uk_87 + 363888*uk_88 + 430920*uk_89 + 2242306609*uk_9 + 76608*uk_90 + 732564*uk_91 + 1038996*uk_92 + 430920*uk_93 + 510300*uk_94 + 90720*uk_95 + 867510*uk_96 + 1230390*uk_97 + 510300*uk_98 + 16128*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 156240*uk_100 + 218736*uk_101 + 76608*uk_102 + 1513575*uk_103 + 2119005*uk_104 + 742140*uk_105 + 2966607*uk_106 + 1038996*uk_107 + 363888*uk_108 + 474552*uk_109 + 3693534*uk_11 + 462384*uk_110 + 97344*uk_111 + 943020*uk_112 + 1320228*uk_113 + 462384*uk_114 + 450528*uk_115 + 94848*uk_116 + 918840*uk_117 + 1286376*uk_118 + 450528*uk_119 + 3598828*uk_12 + 19968*uk_120 + 193440*uk_121 + 270816*uk_122 + 94848*uk_123 + 1873950*uk_124 + 2623530*uk_125 + 918840*uk_126 + 3672942*uk_127 + 1286376*uk_128 + 450528*uk_129 + 757648*uk_13 + 438976*uk_130 + 92416*uk_131 + 895280*uk_132 + 1253392*uk_133 + 438976*uk_134 + 19456*uk_135 + 188480*uk_136 + 263872*uk_137 + 92416*uk_138 + 1825900*uk_139 + 7339715*uk_14 + 2556260*uk_140 + 895280*uk_141 + 3578764*uk_142 + 1253392*uk_143 + 438976*uk_144 + 4096*uk_145 + 39680*uk_146 + 55552*uk_147 + 19456*uk_148 + 384400*uk_149 + 10275601*uk_15 + 538160*uk_150 + 188480*uk_151 + 753424*uk_152 + 263872*uk_153 + 92416*uk_154 + 3723875*uk_155 + 5213425*uk_156 + 1825900*uk_157 + 7298795*uk_158 + 2556260*uk_159 + 3598828*uk_16 + 895280*uk_160 + 10218313*uk_161 + 3578764*uk_162 + 1253392*uk_163 + 438976*uk_164 + 3969*uk_17 + 4914*uk_18 + 4788*uk_19 + 63*uk_2 + 1008*uk_20 + 9765*uk_21 + 13671*uk_22 + 4788*uk_23 + 6084*uk_24 + 5928*uk_25 + 1248*uk_26 + 12090*uk_27 + 16926*uk_28 + 5928*uk_29 + 78*uk_3 + 5776*uk_30 + 1216*uk_31 + 11780*uk_32 + 16492*uk_33 + 5776*uk_34 + 256*uk_35 + 2480*uk_36 + 3472*uk_37 + 1216*uk_38 + 24025*uk_39 + 76*uk_4 + 33635*uk_40 + 11780*uk_41 + 47089*uk_42 + 16492*uk_43 + 5776*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 174899915502*uk_47 + 170415302284*uk_48 + 35876905744*uk_49 + 16*uk_5 + 347557524395*uk_50 + 486580534153*uk_51 + 170415302284*uk_52 + 187944057*uk_53 + 232692642*uk_54 + 226726164*uk_55 + 47731824*uk_56 + 462402045*uk_57 + 647362863*uk_58 + 226726164*uk_59 + 155*uk_6 + 288095652*uk_60 + 280708584*uk_61 + 59096544*uk_62 + 572497770*uk_63 + 801496878*uk_64 + 280708584*uk_65 + 273510928*uk_66 + 57581248*uk_67 + 557818340*uk_68 + 780945676*uk_69 + 217*uk_7 + 273510928*uk_70 + 12122368*uk_71 + 117435440*uk_72 + 164409616*uk_73 + 57581248*uk_74 + 1137655825*uk_75 + 1592718155*uk_76 + 557818340*uk_77 + 2229805417*uk_78 + 780945676*uk_79 + 76*uk_8 + 273510928*uk_80 + 250047*uk_81 + 309582*uk_82 + 301644*uk_83 + 63504*uk_84 + 615195*uk_85 + 861273*uk_86 + 301644*uk_87 + 383292*uk_88 + 373464*uk_89 + 2242306609*uk_9 + 78624*uk_90 + 761670*uk_91 + 1066338*uk_92 + 373464*uk_93 + 363888*uk_94 + 76608*uk_95 + 742140*uk_96 + 1038996*uk_97 + 363888*uk_98 + 16128*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 158256*uk_100 + 218736*uk_101 + 78624*uk_102 + 1552887*uk_103 + 2146347*uk_104 + 771498*uk_105 + 2966607*uk_106 + 1066338*uk_107 + 383292*uk_108 + 884736*uk_109 + 4545888*uk_11 + 718848*uk_110 + 147456*uk_111 + 1446912*uk_112 + 1999872*uk_113 + 718848*uk_114 + 584064*uk_115 + 119808*uk_116 + 1175616*uk_117 + 1624896*uk_118 + 584064*uk_119 + 3693534*uk_12 + 24576*uk_120 + 241152*uk_121 + 333312*uk_122 + 119808*uk_123 + 2366304*uk_124 + 3270624*uk_125 + 1175616*uk_126 + 4520544*uk_127 + 1624896*uk_128 + 584064*uk_129 + 757648*uk_13 + 474552*uk_130 + 97344*uk_131 + 955188*uk_132 + 1320228*uk_133 + 474552*uk_134 + 19968*uk_135 + 195936*uk_136 + 270816*uk_137 + 97344*uk_138 + 1922622*uk_139 + 7434421*uk_14 + 2657382*uk_140 + 955188*uk_141 + 3672942*uk_142 + 1320228*uk_143 + 474552*uk_144 + 4096*uk_145 + 40192*uk_146 + 55552*uk_147 + 19968*uk_148 + 394384*uk_149 + 10275601*uk_15 + 545104*uk_150 + 195936*uk_151 + 753424*uk_152 + 270816*uk_153 + 97344*uk_154 + 3869893*uk_155 + 5348833*uk_156 + 1922622*uk_157 + 7392973*uk_158 + 2657382*uk_159 + 3693534*uk_16 + 955188*uk_160 + 10218313*uk_161 + 3672942*uk_162 + 1320228*uk_163 + 474552*uk_164 + 3969*uk_17 + 6048*uk_18 + 4914*uk_19 + 63*uk_2 + 1008*uk_20 + 9891*uk_21 + 13671*uk_22 + 4914*uk_23 + 9216*uk_24 + 7488*uk_25 + 1536*uk_26 + 15072*uk_27 + 20832*uk_28 + 7488*uk_29 + 96*uk_3 + 6084*uk_30 + 1248*uk_31 + 12246*uk_32 + 16926*uk_33 + 6084*uk_34 + 256*uk_35 + 2512*uk_36 + 3472*uk_37 + 1248*uk_38 + 24649*uk_39 + 78*uk_4 + 34069*uk_40 + 12246*uk_41 + 47089*uk_42 + 16926*uk_43 + 6084*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 215261434464*uk_47 + 174899915502*uk_48 + 35876905744*uk_49 + 16*uk_5 + 352042137613*uk_50 + 486580534153*uk_51 + 174899915502*uk_52 + 187944057*uk_53 + 286390944*uk_54 + 232692642*uk_55 + 47731824*uk_56 + 468368523*uk_57 + 647362863*uk_58 + 232692642*uk_59 + 157*uk_6 + 436405248*uk_60 + 354579264*uk_61 + 72734208*uk_62 + 713704416*uk_63 + 986457696*uk_64 + 354579264*uk_65 + 288095652*uk_66 + 59096544*uk_67 + 579884838*uk_68 + 801496878*uk_69 + 217*uk_7 + 288095652*uk_70 + 12122368*uk_71 + 118950736*uk_72 + 164409616*uk_73 + 59096544*uk_74 + 1167204097*uk_75 + 1613269357*uk_76 + 579884838*uk_77 + 2229805417*uk_78 + 801496878*uk_79 + 78*uk_8 + 288095652*uk_80 + 250047*uk_81 + 381024*uk_82 + 309582*uk_83 + 63504*uk_84 + 623133*uk_85 + 861273*uk_86 + 309582*uk_87 + 580608*uk_88 + 471744*uk_89 + 2242306609*uk_9 + 96768*uk_90 + 949536*uk_91 + 1312416*uk_92 + 471744*uk_93 + 383292*uk_94 + 78624*uk_95 + 771498*uk_96 + 1066338*uk_97 + 383292*uk_98 + 16128*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 160272*uk_100 + 218736*uk_101 + 96768*uk_102 + 1592703*uk_103 + 2173689*uk_104 + 961632*uk_105 + 2966607*uk_106 + 1312416*uk_107 + 580608*uk_108 + 2197000*uk_109 + 6155890*uk_11 + 1622400*uk_110 + 270400*uk_111 + 2687100*uk_112 + 3667300*uk_113 + 1622400*uk_114 + 1198080*uk_115 + 199680*uk_116 + 1984320*uk_117 + 2708160*uk_118 + 1198080*uk_119 + 4545888*uk_12 + 33280*uk_120 + 330720*uk_121 + 451360*uk_122 + 199680*uk_123 + 3286530*uk_124 + 4485390*uk_125 + 1984320*uk_126 + 6121570*uk_127 + 2708160*uk_128 + 1198080*uk_129 + 757648*uk_13 + 884736*uk_130 + 147456*uk_131 + 1465344*uk_132 + 1999872*uk_133 + 884736*uk_134 + 24576*uk_135 + 244224*uk_136 + 333312*uk_137 + 147456*uk_138 + 2426976*uk_139 + 7529127*uk_14 + 3312288*uk_140 + 1465344*uk_141 + 4520544*uk_142 + 1999872*uk_143 + 884736*uk_144 + 4096*uk_145 + 40704*uk_146 + 55552*uk_147 + 24576*uk_148 + 404496*uk_149 + 10275601*uk_15 + 552048*uk_150 + 244224*uk_151 + 753424*uk_152 + 333312*uk_153 + 147456*uk_154 + 4019679*uk_155 + 5485977*uk_156 + 2426976*uk_157 + 7487151*uk_158 + 3312288*uk_159 + 4545888*uk_16 + 1465344*uk_160 + 10218313*uk_161 + 4520544*uk_162 + 1999872*uk_163 + 884736*uk_164 + 3969*uk_17 + 8190*uk_18 + 6048*uk_19 + 63*uk_2 + 1008*uk_20 + 10017*uk_21 + 13671*uk_22 + 6048*uk_23 + 16900*uk_24 + 12480*uk_25 + 2080*uk_26 + 20670*uk_27 + 28210*uk_28 + 12480*uk_29 + 130*uk_3 + 9216*uk_30 + 1536*uk_31 + 15264*uk_32 + 20832*uk_33 + 9216*uk_34 + 256*uk_35 + 2544*uk_36 + 3472*uk_37 + 1536*uk_38 + 25281*uk_39 + 96*uk_4 + 34503*uk_40 + 15264*uk_41 + 47089*uk_42 + 20832*uk_43 + 9216*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 291499859170*uk_47 + 215261434464*uk_48 + 35876905744*uk_49 + 16*uk_5 + 356526750831*uk_50 + 486580534153*uk_51 + 215261434464*uk_52 + 187944057*uk_53 + 387821070*uk_54 + 286390944*uk_55 + 47731824*uk_56 + 474335001*uk_57 + 647362863*uk_58 + 286390944*uk_59 + 159*uk_6 + 800265700*uk_60 + 590965440*uk_61 + 98494240*uk_62 + 978786510*uk_63 + 1335828130*uk_64 + 590965440*uk_65 + 436405248*uk_66 + 72734208*uk_67 + 722796192*uk_68 + 986457696*uk_69 + 217*uk_7 + 436405248*uk_70 + 12122368*uk_71 + 120466032*uk_72 + 164409616*uk_73 + 72734208*uk_74 + 1197131193*uk_75 + 1633820559*uk_76 + 722796192*uk_77 + 2229805417*uk_78 + 986457696*uk_79 + 96*uk_8 + 436405248*uk_80 + 250047*uk_81 + 515970*uk_82 + 381024*uk_83 + 63504*uk_84 + 631071*uk_85 + 861273*uk_86 + 381024*uk_87 + 1064700*uk_88 + 786240*uk_89 + 2242306609*uk_9 + 131040*uk_90 + 1302210*uk_91 + 1777230*uk_92 + 786240*uk_93 + 580608*uk_94 + 96768*uk_95 + 961632*uk_96 + 1312416*uk_97 + 580608*uk_98 + 16128*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 121716*uk_100 + 164052*uk_101 + 98280*uk_102 + 1633023*uk_103 + 2201031*uk_104 + 1318590*uk_105 + 2966607*uk_106 + 1777230*uk_107 + 1064700*uk_108 + 6859*uk_109 + 899707*uk_11 + 46930*uk_110 + 4332*uk_111 + 58121*uk_112 + 78337*uk_113 + 46930*uk_114 + 321100*uk_115 + 29640*uk_116 + 397670*uk_117 + 535990*uk_118 + 321100*uk_119 + 6155890*uk_12 + 2736*uk_120 + 36708*uk_121 + 49476*uk_122 + 29640*uk_123 + 492499*uk_124 + 663803*uk_125 + 397670*uk_126 + 894691*uk_127 + 535990*uk_128 + 321100*uk_129 + 568236*uk_13 + 2197000*uk_130 + 202800*uk_131 + 2720900*uk_132 + 3667300*uk_133 + 2197000*uk_134 + 18720*uk_135 + 251160*uk_136 + 338520*uk_137 + 202800*uk_138 + 3369730*uk_139 + 7623833*uk_14 + 4541810*uk_140 + 2720900*uk_141 + 6121570*uk_142 + 3667300*uk_143 + 2197000*uk_144 + 1728*uk_145 + 23184*uk_146 + 31248*uk_147 + 18720*uk_148 + 311052*uk_149 + 10275601*uk_15 + 419244*uk_150 + 251160*uk_151 + 565068*uk_152 + 338520*uk_153 + 202800*uk_154 + 4173281*uk_155 + 5624857*uk_156 + 3369730*uk_157 + 7581329*uk_158 + 4541810*uk_159 + 6155890*uk_16 + 2720900*uk_160 + 10218313*uk_161 + 6121570*uk_162 + 3667300*uk_163 + 2197000*uk_164 + 3969*uk_17 + 1197*uk_18 + 8190*uk_19 + 63*uk_2 + 756*uk_20 + 10143*uk_21 + 13671*uk_22 + 8190*uk_23 + 361*uk_24 + 2470*uk_25 + 228*uk_26 + 3059*uk_27 + 4123*uk_28 + 2470*uk_29 + 19*uk_3 + 16900*uk_30 + 1560*uk_31 + 20930*uk_32 + 28210*uk_33 + 16900*uk_34 + 144*uk_35 + 1932*uk_36 + 2604*uk_37 + 1560*uk_38 + 25921*uk_39 + 130*uk_4 + 34937*uk_40 + 20930*uk_41 + 47089*uk_42 + 28210*uk_43 + 16900*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 42603825571*uk_47 + 291499859170*uk_48 + 26907679308*uk_49 + 12*uk_5 + 361011364049*uk_50 + 486580534153*uk_51 + 291499859170*uk_52 + 187944057*uk_53 + 56681541*uk_54 + 387821070*uk_55 + 35798868*uk_56 + 480301479*uk_57 + 647362863*uk_58 + 387821070*uk_59 + 161*uk_6 + 17094433*uk_60 + 116961910*uk_61 + 10796484*uk_62 + 144852827*uk_63 + 195236419*uk_64 + 116961910*uk_65 + 800265700*uk_66 + 73870680*uk_67 + 991098290*uk_68 + 1335828130*uk_69 + 217*uk_7 + 800265700*uk_70 + 6818832*uk_71 + 91485996*uk_72 + 123307212*uk_73 + 73870680*uk_74 + 1227437113*uk_75 + 1654371761*uk_76 + 991098290*uk_77 + 2229805417*uk_78 + 1335828130*uk_79 + 130*uk_8 + 800265700*uk_80 + 250047*uk_81 + 75411*uk_82 + 515970*uk_83 + 47628*uk_84 + 639009*uk_85 + 861273*uk_86 + 515970*uk_87 + 22743*uk_88 + 155610*uk_89 + 2242306609*uk_9 + 14364*uk_90 + 192717*uk_91 + 259749*uk_92 + 155610*uk_93 + 1064700*uk_94 + 98280*uk_95 + 1318590*uk_96 + 1777230*uk_97 + 1064700*uk_98 + 9072*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 164304*uk_100 + 218736*uk_101 + 19152*uk_102 + 1673847*uk_103 + 2228373*uk_104 + 195111*uk_105 + 2966607*uk_106 + 259749*uk_107 + 22743*uk_108 + 571787*uk_109 + 3930299*uk_11 + 130891*uk_110 + 110224*uk_111 + 1122907*uk_112 + 1494913*uk_113 + 130891*uk_114 + 29963*uk_115 + 25232*uk_116 + 257051*uk_117 + 342209*uk_118 + 29963*uk_119 + 899707*uk_12 + 21248*uk_120 + 216464*uk_121 + 288176*uk_122 + 25232*uk_123 + 2205227*uk_124 + 2935793*uk_125 + 257051*uk_126 + 3908387*uk_127 + 342209*uk_128 + 29963*uk_129 + 757648*uk_13 + 6859*uk_130 + 5776*uk_131 + 58843*uk_132 + 78337*uk_133 + 6859*uk_134 + 4864*uk_135 + 49552*uk_136 + 65968*uk_137 + 5776*uk_138 + 504811*uk_139 + 7718539*uk_14 + 672049*uk_140 + 58843*uk_141 + 894691*uk_142 + 78337*uk_143 + 6859*uk_144 + 4096*uk_145 + 41728*uk_146 + 55552*uk_147 + 4864*uk_148 + 425104*uk_149 + 10275601*uk_15 + 565936*uk_150 + 49552*uk_151 + 753424*uk_152 + 65968*uk_153 + 5776*uk_154 + 4330747*uk_155 + 5765473*uk_156 + 504811*uk_157 + 7675507*uk_158 + 672049*uk_159 + 899707*uk_16 + 58843*uk_160 + 10218313*uk_161 + 894691*uk_162 + 78337*uk_163 + 6859*uk_164 + 3969*uk_17 + 5229*uk_18 + 1197*uk_19 + 63*uk_2 + 1008*uk_20 + 10269*uk_21 + 13671*uk_22 + 1197*uk_23 + 6889*uk_24 + 1577*uk_25 + 1328*uk_26 + 13529*uk_27 + 18011*uk_28 + 1577*uk_29 + 83*uk_3 + 361*uk_30 + 304*uk_31 + 3097*uk_32 + 4123*uk_33 + 361*uk_34 + 256*uk_35 + 2608*uk_36 + 3472*uk_37 + 304*uk_38 + 26569*uk_39 + 19*uk_4 + 35371*uk_40 + 3097*uk_41 + 47089*uk_42 + 4123*uk_43 + 361*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 186111448547*uk_47 + 42603825571*uk_48 + 35876905744*uk_49 + 16*uk_5 + 365495977267*uk_50 + 486580534153*uk_51 + 42603825571*uk_52 + 187944057*uk_53 + 247608837*uk_54 + 56681541*uk_55 + 47731824*uk_56 + 486267957*uk_57 + 647362863*uk_58 + 56681541*uk_59 + 163*uk_6 + 326214817*uk_60 + 74675681*uk_61 + 62884784*uk_62 + 640638737*uk_63 + 852874883*uk_64 + 74675681*uk_65 + 17094433*uk_66 + 14395312*uk_67 + 146652241*uk_68 + 195236419*uk_69 + 217*uk_7 + 17094433*uk_70 + 12122368*uk_71 + 123496624*uk_72 + 164409616*uk_73 + 14395312*uk_74 + 1258121857*uk_75 + 1674922963*uk_76 + 146652241*uk_77 + 2229805417*uk_78 + 195236419*uk_79 + 19*uk_8 + 17094433*uk_80 + 250047*uk_81 + 329427*uk_82 + 75411*uk_83 + 63504*uk_84 + 646947*uk_85 + 861273*uk_86 + 75411*uk_87 + 434007*uk_88 + 99351*uk_89 + 2242306609*uk_9 + 83664*uk_90 + 852327*uk_91 + 1134693*uk_92 + 99351*uk_93 + 22743*uk_94 + 19152*uk_95 + 195111*uk_96 + 259749*uk_97 + 22743*uk_98 + 16128*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 166320*uk_100 + 218736*uk_101 + 83664*uk_102 + 1715175*uk_103 + 2255715*uk_104 + 862785*uk_105 + 2966607*uk_106 + 1134693*uk_107 + 434007*uk_108 + 4330747*uk_109 + 7718539*uk_11 + 2205227*uk_110 + 425104*uk_111 + 4383885*uk_112 + 5765473*uk_113 + 2205227*uk_114 + 1122907*uk_115 + 216464*uk_116 + 2232285*uk_117 + 2935793*uk_118 + 1122907*uk_119 + 3930299*uk_12 + 41728*uk_120 + 430320*uk_121 + 565936*uk_122 + 216464*uk_123 + 4437675*uk_124 + 5836215*uk_125 + 2232285*uk_126 + 7675507*uk_127 + 2935793*uk_128 + 1122907*uk_129 + 757648*uk_13 + 571787*uk_130 + 110224*uk_131 + 1136685*uk_132 + 1494913*uk_133 + 571787*uk_134 + 21248*uk_135 + 219120*uk_136 + 288176*uk_137 + 110224*uk_138 + 2259675*uk_139 + 7813245*uk_14 + 2971815*uk_140 + 1136685*uk_141 + 3908387*uk_142 + 1494913*uk_143 + 571787*uk_144 + 4096*uk_145 + 42240*uk_146 + 55552*uk_147 + 21248*uk_148 + 435600*uk_149 + 10275601*uk_15 + 572880*uk_150 + 219120*uk_151 + 753424*uk_152 + 288176*uk_153 + 110224*uk_154 + 4492125*uk_155 + 5907825*uk_156 + 2259675*uk_157 + 7769685*uk_158 + 2971815*uk_159 + 3930299*uk_16 + 1136685*uk_160 + 10218313*uk_161 + 3908387*uk_162 + 1494913*uk_163 + 571787*uk_164 + 3969*uk_17 + 10269*uk_18 + 5229*uk_19 + 63*uk_2 + 1008*uk_20 + 10395*uk_21 + 13671*uk_22 + 5229*uk_23 + 26569*uk_24 + 13529*uk_25 + 2608*uk_26 + 26895*uk_27 + 35371*uk_28 + 13529*uk_29 + 163*uk_3 + 6889*uk_30 + 1328*uk_31 + 13695*uk_32 + 18011*uk_33 + 6889*uk_34 + 256*uk_35 + 2640*uk_36 + 3472*uk_37 + 1328*uk_38 + 27225*uk_39 + 83*uk_4 + 35805*uk_40 + 13695*uk_41 + 47089*uk_42 + 18011*uk_43 + 6889*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 365495977267*uk_47 + 186111448547*uk_48 + 35876905744*uk_49 + 16*uk_5 + 369980590485*uk_50 + 486580534153*uk_51 + 186111448547*uk_52 + 187944057*uk_53 + 486267957*uk_54 + 247608837*uk_55 + 47731824*uk_56 + 492234435*uk_57 + 647362863*uk_58 + 247608837*uk_59 + 165*uk_6 + 1258121857*uk_60 + 640638737*uk_61 + 123496624*uk_62 + 1273558935*uk_63 + 1674922963*uk_64 + 640638737*uk_65 + 326214817*uk_66 + 62884784*uk_67 + 648499335*uk_68 + 852874883*uk_69 + 217*uk_7 + 326214817*uk_70 + 12122368*uk_71 + 125011920*uk_72 + 164409616*uk_73 + 62884784*uk_74 + 1289185425*uk_75 + 1695474165*uk_76 + 648499335*uk_77 + 2229805417*uk_78 + 852874883*uk_79 + 83*uk_8 + 326214817*uk_80 + 250047*uk_81 + 646947*uk_82 + 329427*uk_83 + 63504*uk_84 + 654885*uk_85 + 861273*uk_86 + 329427*uk_87 + 1673847*uk_88 + 852327*uk_89 + 2242306609*uk_9 + 164304*uk_90 + 1694385*uk_91 + 2228373*uk_92 + 852327*uk_93 + 434007*uk_94 + 83664*uk_95 + 862785*uk_96 + 1134693*uk_97 + 434007*uk_98 + 16128*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 126252*uk_100 + 164052*uk_101 + 123228*uk_102 + 1757007*uk_103 + 2283057*uk_104 + 1714923*uk_105 + 2966607*uk_106 + 2228373*uk_107 + 1673847*uk_108 + 778688*uk_109 + 4356476*uk_11 + 1379632*uk_110 + 101568*uk_111 + 1413488*uk_112 + 1836688*uk_113 + 1379632*uk_114 + 2444348*uk_115 + 179952*uk_116 + 2504332*uk_117 + 3254132*uk_118 + 2444348*uk_119 + 7718539*uk_12 + 13248*uk_120 + 184368*uk_121 + 239568*uk_122 + 179952*uk_123 + 2565788*uk_124 + 3333988*uk_125 + 2504332*uk_126 + 4332188*uk_127 + 3254132*uk_128 + 2444348*uk_129 + 568236*uk_13 + 4330747*uk_130 + 318828*uk_131 + 4437023*uk_132 + 5765473*uk_133 + 4330747*uk_134 + 23472*uk_135 + 326652*uk_136 + 424452*uk_137 + 318828*uk_138 + 4545907*uk_139 + 7907951*uk_14 + 5906957*uk_140 + 4437023*uk_141 + 7675507*uk_142 + 5765473*uk_143 + 4330747*uk_144 + 1728*uk_145 + 24048*uk_146 + 31248*uk_147 + 23472*uk_148 + 334668*uk_149 + 10275601*uk_15 + 434868*uk_150 + 326652*uk_151 + 565068*uk_152 + 424452*uk_153 + 318828*uk_154 + 4657463*uk_155 + 6051913*uk_156 + 4545907*uk_157 + 7863863*uk_158 + 5906957*uk_159 + 7718539*uk_16 + 4437023*uk_160 + 10218313*uk_161 + 7675507*uk_162 + 5765473*uk_163 + 4330747*uk_164 + 3969*uk_17 + 5796*uk_18 + 10269*uk_19 + 63*uk_2 + 756*uk_20 + 10521*uk_21 + 13671*uk_22 + 10269*uk_23 + 8464*uk_24 + 14996*uk_25 + 1104*uk_26 + 15364*uk_27 + 19964*uk_28 + 14996*uk_29 + 92*uk_3 + 26569*uk_30 + 1956*uk_31 + 27221*uk_32 + 35371*uk_33 + 26569*uk_34 + 144*uk_35 + 2004*uk_36 + 2604*uk_37 + 1956*uk_38 + 27889*uk_39 + 163*uk_4 + 36239*uk_40 + 27221*uk_41 + 47089*uk_42 + 35371*uk_43 + 26569*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 206292208028*uk_47 + 365495977267*uk_48 + 26907679308*uk_49 + 12*uk_5 + 374465203703*uk_50 + 486580534153*uk_51 + 365495977267*uk_52 + 187944057*uk_53 + 274457988*uk_54 + 486267957*uk_55 + 35798868*uk_56 + 498200913*uk_57 + 647362863*uk_58 + 486267957*uk_59 + 167*uk_6 + 400795792*uk_60 + 710105588*uk_61 + 52277712*uk_62 + 727531492*uk_63 + 945355292*uk_64 + 710105588*uk_65 + 1258121857*uk_66 + 92622468*uk_67 + 1288996013*uk_68 + 1674922963*uk_69 + 217*uk_7 + 1258121857*uk_70 + 6818832*uk_71 + 94895412*uk_72 + 123307212*uk_73 + 92622468*uk_74 + 1320627817*uk_75 + 1716025367*uk_76 + 1288996013*uk_77 + 2229805417*uk_78 + 1674922963*uk_79 + 163*uk_8 + 1258121857*uk_80 + 250047*uk_81 + 365148*uk_82 + 646947*uk_83 + 47628*uk_84 + 662823*uk_85 + 861273*uk_86 + 646947*uk_87 + 533232*uk_88 + 944748*uk_89 + 2242306609*uk_9 + 69552*uk_90 + 967932*uk_91 + 1257732*uk_92 + 944748*uk_93 + 1673847*uk_94 + 123228*uk_95 + 1714923*uk_96 + 2228373*uk_97 + 1673847*uk_98 + 9072*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 127764*uk_100 + 164052*uk_101 + 69552*uk_102 + 1799343*uk_103 + 2310399*uk_104 + 979524*uk_105 + 2966607*uk_106 + 1257732*uk_107 + 533232*uk_108 + 35937*uk_109 + 1562649*uk_11 + 100188*uk_110 + 13068*uk_111 + 184041*uk_112 + 236313*uk_113 + 100188*uk_114 + 279312*uk_115 + 36432*uk_116 + 513084*uk_117 + 658812*uk_118 + 279312*uk_119 + 4356476*uk_12 + 4752*uk_120 + 66924*uk_121 + 85932*uk_122 + 36432*uk_123 + 942513*uk_124 + 1210209*uk_125 + 513084*uk_126 + 1553937*uk_127 + 658812*uk_128 + 279312*uk_129 + 568236*uk_13 + 778688*uk_130 + 101568*uk_131 + 1430416*uk_132 + 1836688*uk_133 + 778688*uk_134 + 13248*uk_135 + 186576*uk_136 + 239568*uk_137 + 101568*uk_138 + 2627612*uk_139 + 8002657*uk_14 + 3373916*uk_140 + 1430416*uk_141 + 4332188*uk_142 + 1836688*uk_143 + 778688*uk_144 + 1728*uk_145 + 24336*uk_146 + 31248*uk_147 + 13248*uk_148 + 342732*uk_149 + 10275601*uk_15 + 440076*uk_150 + 186576*uk_151 + 565068*uk_152 + 239568*uk_153 + 101568*uk_154 + 4826809*uk_155 + 6197737*uk_156 + 2627612*uk_157 + 7958041*uk_158 + 3373916*uk_159 + 4356476*uk_16 + 1430416*uk_160 + 10218313*uk_161 + 4332188*uk_162 + 1836688*uk_163 + 778688*uk_164 + 3969*uk_17 + 2079*uk_18 + 5796*uk_19 + 63*uk_2 + 756*uk_20 + 10647*uk_21 + 13671*uk_22 + 5796*uk_23 + 1089*uk_24 + 3036*uk_25 + 396*uk_26 + 5577*uk_27 + 7161*uk_28 + 3036*uk_29 + 33*uk_3 + 8464*uk_30 + 1104*uk_31 + 15548*uk_32 + 19964*uk_33 + 8464*uk_34 + 144*uk_35 + 2028*uk_36 + 2604*uk_37 + 1104*uk_38 + 28561*uk_39 + 92*uk_4 + 36673*uk_40 + 15548*uk_41 + 47089*uk_42 + 19964*uk_43 + 8464*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 73996118097*uk_47 + 206292208028*uk_48 + 26907679308*uk_49 + 12*uk_5 + 378949816921*uk_50 + 486580534153*uk_51 + 206292208028*uk_52 + 187944057*uk_53 + 98446887*uk_54 + 274457988*uk_55 + 35798868*uk_56 + 504167391*uk_57 + 647362863*uk_58 + 274457988*uk_59 + 169*uk_6 + 51567417*uk_60 + 143763708*uk_61 + 18751788*uk_62 + 264087681*uk_63 + 339094833*uk_64 + 143763708*uk_65 + 400795792*uk_66 + 52277712*uk_67 + 736244444*uk_68 + 945355292*uk_69 + 217*uk_7 + 400795792*uk_70 + 6818832*uk_71 + 96031884*uk_72 + 123307212*uk_73 + 52277712*uk_74 + 1352449033*uk_75 + 1736576569*uk_76 + 736244444*uk_77 + 2229805417*uk_78 + 945355292*uk_79 + 92*uk_8 + 400795792*uk_80 + 250047*uk_81 + 130977*uk_82 + 365148*uk_83 + 47628*uk_84 + 670761*uk_85 + 861273*uk_86 + 365148*uk_87 + 68607*uk_88 + 191268*uk_89 + 2242306609*uk_9 + 24948*uk_90 + 351351*uk_91 + 451143*uk_92 + 191268*uk_93 + 533232*uk_94 + 69552*uk_95 + 979524*uk_96 + 1257732*uk_97 + 533232*uk_98 + 9072*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 172368*uk_100 + 218736*uk_101 + 33264*uk_102 + 1842183*uk_103 + 2337741*uk_104 + 355509*uk_105 + 2966607*uk_106 + 451143*uk_107 + 68607*uk_108 + 3869893*uk_109 + 7434421*uk_11 + 813417*uk_110 + 394384*uk_111 + 4214979*uk_112 + 5348833*uk_113 + 813417*uk_114 + 170973*uk_115 + 82896*uk_116 + 885951*uk_117 + 1124277*uk_118 + 170973*uk_119 + 1562649*uk_12 + 40192*uk_120 + 429552*uk_121 + 545104*uk_122 + 82896*uk_123 + 4590837*uk_124 + 5825799*uk_125 + 885951*uk_126 + 7392973*uk_127 + 1124277*uk_128 + 170973*uk_129 + 757648*uk_13 + 35937*uk_130 + 17424*uk_131 + 186219*uk_132 + 236313*uk_133 + 35937*uk_134 + 8448*uk_135 + 90288*uk_136 + 114576*uk_137 + 17424*uk_138 + 964953*uk_139 + 8097363*uk_14 + 1224531*uk_140 + 186219*uk_141 + 1553937*uk_142 + 236313*uk_143 + 35937*uk_144 + 4096*uk_145 + 43776*uk_146 + 55552*uk_147 + 8448*uk_148 + 467856*uk_149 + 10275601*uk_15 + 593712*uk_150 + 90288*uk_151 + 753424*uk_152 + 114576*uk_153 + 17424*uk_154 + 5000211*uk_155 + 6345297*uk_156 + 964953*uk_157 + 8052219*uk_158 + 1224531*uk_159 + 1562649*uk_16 + 186219*uk_160 + 10218313*uk_161 + 1553937*uk_162 + 236313*uk_163 + 35937*uk_164 + 3969*uk_17 + 9891*uk_18 + 2079*uk_19 + 63*uk_2 + 1008*uk_20 + 10773*uk_21 + 13671*uk_22 + 2079*uk_23 + 24649*uk_24 + 5181*uk_25 + 2512*uk_26 + 26847*uk_27 + 34069*uk_28 + 5181*uk_29 + 157*uk_3 + 1089*uk_30 + 528*uk_31 + 5643*uk_32 + 7161*uk_33 + 1089*uk_34 + 256*uk_35 + 2736*uk_36 + 3472*uk_37 + 528*uk_38 + 29241*uk_39 + 33*uk_4 + 37107*uk_40 + 5643*uk_41 + 47089*uk_42 + 7161*uk_43 + 1089*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 352042137613*uk_47 + 73996118097*uk_48 + 35876905744*uk_49 + 16*uk_5 + 383434430139*uk_50 + 486580534153*uk_51 + 73996118097*uk_52 + 187944057*uk_53 + 468368523*uk_54 + 98446887*uk_55 + 47731824*uk_56 + 510133869*uk_57 + 647362863*uk_58 + 98446887*uk_59 + 171*uk_6 + 1167204097*uk_60 + 245335893*uk_61 + 118950736*uk_62 + 1271285991*uk_63 + 1613269357*uk_64 + 245335893*uk_65 + 51567417*uk_66 + 25002384*uk_67 + 267212979*uk_68 + 339094833*uk_69 + 217*uk_7 + 51567417*uk_70 + 12122368*uk_71 + 129557808*uk_72 + 164409616*uk_73 + 25002384*uk_74 + 1384649073*uk_75 + 1757127771*uk_76 + 267212979*uk_77 + 2229805417*uk_78 + 339094833*uk_79 + 33*uk_8 + 51567417*uk_80 + 250047*uk_81 + 623133*uk_82 + 130977*uk_83 + 63504*uk_84 + 678699*uk_85 + 861273*uk_86 + 130977*uk_87 + 1552887*uk_88 + 326403*uk_89 + 2242306609*uk_9 + 158256*uk_90 + 1691361*uk_91 + 2146347*uk_92 + 326403*uk_93 + 68607*uk_94 + 33264*uk_95 + 355509*uk_96 + 451143*uk_97 + 68607*uk_98 + 16128*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 130788*uk_100 + 164052*uk_101 + 118692*uk_102 + 1885527*uk_103 + 2365083*uk_104 + 1711143*uk_105 + 2966607*uk_106 + 2146347*uk_107 + 1552887*uk_108 + 1906624*uk_109 + 5871772*uk_11 + 2414032*uk_110 + 184512*uk_111 + 2660048*uk_112 + 3336592*uk_113 + 2414032*uk_114 + 3056476*uk_115 + 233616*uk_116 + 3367964*uk_117 + 4224556*uk_118 + 3056476*uk_119 + 7434421*uk_12 + 17856*uk_120 + 257424*uk_121 + 322896*uk_122 + 233616*uk_123 + 3711196*uk_124 + 4655084*uk_125 + 3367964*uk_126 + 5839036*uk_127 + 4224556*uk_128 + 3056476*uk_129 + 568236*uk_13 + 3869893*uk_130 + 295788*uk_131 + 4264277*uk_132 + 5348833*uk_133 + 3869893*uk_134 + 22608*uk_135 + 325932*uk_136 + 408828*uk_137 + 295788*uk_138 + 4698853*uk_139 + 8192069*uk_14 + 5893937*uk_140 + 4264277*uk_141 + 7392973*uk_142 + 5348833*uk_143 + 3869893*uk_144 + 1728*uk_145 + 24912*uk_146 + 31248*uk_147 + 22608*uk_148 + 359148*uk_149 + 10275601*uk_15 + 450492*uk_150 + 325932*uk_151 + 565068*uk_152 + 408828*uk_153 + 295788*uk_154 + 5177717*uk_155 + 6494593*uk_156 + 4698853*uk_157 + 8146397*uk_158 + 5893937*uk_159 + 7434421*uk_16 + 4264277*uk_160 + 10218313*uk_161 + 7392973*uk_162 + 5348833*uk_163 + 3869893*uk_164 + 3969*uk_17 + 7812*uk_18 + 9891*uk_19 + 63*uk_2 + 756*uk_20 + 10899*uk_21 + 13671*uk_22 + 9891*uk_23 + 15376*uk_24 + 19468*uk_25 + 1488*uk_26 + 21452*uk_27 + 26908*uk_28 + 19468*uk_29 + 124*uk_3 + 24649*uk_30 + 1884*uk_31 + 27161*uk_32 + 34069*uk_33 + 24649*uk_34 + 144*uk_35 + 2076*uk_36 + 2604*uk_37 + 1884*uk_38 + 29929*uk_39 + 157*uk_4 + 37541*uk_40 + 27161*uk_41 + 47089*uk_42 + 34069*uk_43 + 24649*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 278046019516*uk_47 + 352042137613*uk_48 + 26907679308*uk_49 + 12*uk_5 + 387919043357*uk_50 + 486580534153*uk_51 + 352042137613*uk_52 + 187944057*uk_53 + 369921636*uk_54 + 468368523*uk_55 + 35798868*uk_56 + 516100347*uk_57 + 647362863*uk_58 + 468368523*uk_59 + 173*uk_6 + 728099728*uk_60 + 921868204*uk_61 + 70461264*uk_62 + 1015816556*uk_63 + 1274174524*uk_64 + 921868204*uk_65 + 1167204097*uk_66 + 89213052*uk_67 + 1286154833*uk_68 + 1613269357*uk_69 + 217*uk_7 + 1167204097*uk_70 + 6818832*uk_71 + 98304828*uk_72 + 123307212*uk_73 + 89213052*uk_74 + 1417227937*uk_75 + 1777678973*uk_76 + 1286154833*uk_77 + 2229805417*uk_78 + 1613269357*uk_79 + 157*uk_8 + 1167204097*uk_80 + 250047*uk_81 + 492156*uk_82 + 623133*uk_83 + 47628*uk_84 + 686637*uk_85 + 861273*uk_86 + 623133*uk_87 + 968688*uk_88 + 1226484*uk_89 + 2242306609*uk_9 + 93744*uk_90 + 1351476*uk_91 + 1695204*uk_92 + 1226484*uk_93 + 1552887*uk_94 + 118692*uk_95 + 1711143*uk_96 + 2146347*uk_97 + 1552887*uk_98 + 9072*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 132300*uk_100 + 164052*uk_101 + 93744*uk_102 + 1929375*uk_103 + 2392425*uk_104 + 1367100*uk_105 + 2966607*uk_106 + 1695204*uk_107 + 968688*uk_108 + 1092727*uk_109 + 4877359*uk_11 + 1315516*uk_110 + 127308*uk_111 + 1856575*uk_112 + 2302153*uk_113 + 1315516*uk_114 + 1583728*uk_115 + 153264*uk_116 + 2235100*uk_117 + 2771524*uk_118 + 1583728*uk_119 + 5871772*uk_12 + 14832*uk_120 + 216300*uk_121 + 268212*uk_122 + 153264*uk_123 + 3154375*uk_124 + 3911425*uk_125 + 2235100*uk_126 + 4850167*uk_127 + 2771524*uk_128 + 1583728*uk_129 + 568236*uk_13 + 1906624*uk_130 + 184512*uk_131 + 2690800*uk_132 + 3336592*uk_133 + 1906624*uk_134 + 17856*uk_135 + 260400*uk_136 + 322896*uk_137 + 184512*uk_138 + 3797500*uk_139 + 8286775*uk_14 + 4708900*uk_140 + 2690800*uk_141 + 5839036*uk_142 + 3336592*uk_143 + 1906624*uk_144 + 1728*uk_145 + 25200*uk_146 + 31248*uk_147 + 17856*uk_148 + 367500*uk_149 + 10275601*uk_15 + 455700*uk_150 + 260400*uk_151 + 565068*uk_152 + 322896*uk_153 + 184512*uk_154 + 5359375*uk_155 + 6645625*uk_156 + 3797500*uk_157 + 8240575*uk_158 + 4708900*uk_159 + 5871772*uk_16 + 2690800*uk_160 + 10218313*uk_161 + 5839036*uk_162 + 3336592*uk_163 + 1906624*uk_164 + 3969*uk_17 + 6489*uk_18 + 7812*uk_19 + 63*uk_2 + 756*uk_20 + 11025*uk_21 + 13671*uk_22 + 7812*uk_23 + 10609*uk_24 + 12772*uk_25 + 1236*uk_26 + 18025*uk_27 + 22351*uk_28 + 12772*uk_29 + 103*uk_3 + 15376*uk_30 + 1488*uk_31 + 21700*uk_32 + 26908*uk_33 + 15376*uk_34 + 144*uk_35 + 2100*uk_36 + 2604*uk_37 + 1488*uk_38 + 30625*uk_39 + 124*uk_4 + 37975*uk_40 + 21700*uk_41 + 47089*uk_42 + 26908*uk_43 + 15376*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 230957580727*uk_47 + 278046019516*uk_48 + 26907679308*uk_49 + 12*uk_5 + 392403656575*uk_50 + 486580534153*uk_51 + 278046019516*uk_52 + 187944057*uk_53 + 307273617*uk_54 + 369921636*uk_55 + 35798868*uk_56 + 522066825*uk_57 + 647362863*uk_58 + 369921636*uk_59 + 175*uk_6 + 502367977*uk_60 + 604792516*uk_61 + 58528308*uk_62 + 853537825*uk_63 + 1058386903*uk_64 + 604792516*uk_65 + 728099728*uk_66 + 70461264*uk_67 + 1027560100*uk_68 + 1274174524*uk_69 + 217*uk_7 + 728099728*uk_70 + 6818832*uk_71 + 99441300*uk_72 + 123307212*uk_73 + 70461264*uk_74 + 1450185625*uk_75 + 1798230175*uk_76 + 1027560100*uk_77 + 2229805417*uk_78 + 1274174524*uk_79 + 124*uk_8 + 728099728*uk_80 + 250047*uk_81 + 408807*uk_82 + 492156*uk_83 + 47628*uk_84 + 694575*uk_85 + 861273*uk_86 + 492156*uk_87 + 668367*uk_88 + 804636*uk_89 + 2242306609*uk_9 + 77868*uk_90 + 1135575*uk_91 + 1408113*uk_92 + 804636*uk_93 + 968688*uk_94 + 93744*uk_95 + 1367100*uk_96 + 1695204*uk_97 + 968688*uk_98 + 9072*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 133812*uk_100 + 164052*uk_101 + 77868*uk_102 + 1973727*uk_103 + 2419767*uk_104 + 1148553*uk_105 + 2966607*uk_106 + 1408113*uk_107 + 668367*uk_108 + 830584*uk_109 + 4451182*uk_11 + 910108*uk_110 + 106032*uk_111 + 1563972*uk_112 + 1917412*uk_113 + 910108*uk_114 + 997246*uk_115 + 116184*uk_116 + 1713714*uk_117 + 2100994*uk_118 + 997246*uk_119 + 4877359*uk_12 + 13536*uk_120 + 199656*uk_121 + 244776*uk_122 + 116184*uk_123 + 2944926*uk_124 + 3610446*uk_125 + 1713714*uk_126 + 4426366*uk_127 + 2100994*uk_128 + 997246*uk_129 + 568236*uk_13 + 1092727*uk_130 + 127308*uk_131 + 1877793*uk_132 + 2302153*uk_133 + 1092727*uk_134 + 14832*uk_135 + 218772*uk_136 + 268212*uk_137 + 127308*uk_138 + 3226887*uk_139 + 8381481*uk_14 + 3956127*uk_140 + 1877793*uk_141 + 4850167*uk_142 + 2302153*uk_143 + 1092727*uk_144 + 1728*uk_145 + 25488*uk_146 + 31248*uk_147 + 14832*uk_148 + 375948*uk_149 + 10275601*uk_15 + 460908*uk_150 + 218772*uk_151 + 565068*uk_152 + 268212*uk_153 + 127308*uk_154 + 5545233*uk_155 + 6798393*uk_156 + 3226887*uk_157 + 8334753*uk_158 + 3956127*uk_159 + 4877359*uk_16 + 1877793*uk_160 + 10218313*uk_161 + 4850167*uk_162 + 2302153*uk_163 + 1092727*uk_164 + 3969*uk_17 + 5922*uk_18 + 6489*uk_19 + 63*uk_2 + 756*uk_20 + 11151*uk_21 + 13671*uk_22 + 6489*uk_23 + 8836*uk_24 + 9682*uk_25 + 1128*uk_26 + 16638*uk_27 + 20398*uk_28 + 9682*uk_29 + 94*uk_3 + 10609*uk_30 + 1236*uk_31 + 18231*uk_32 + 22351*uk_33 + 10609*uk_34 + 144*uk_35 + 2124*uk_36 + 2604*uk_37 + 1236*uk_38 + 31329*uk_39 + 103*uk_4 + 38409*uk_40 + 18231*uk_41 + 47089*uk_42 + 22351*uk_43 + 10609*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 210776821246*uk_47 + 230957580727*uk_48 + 26907679308*uk_49 + 12*uk_5 + 396888269793*uk_50 + 486580534153*uk_51 + 230957580727*uk_52 + 187944057*uk_53 + 280424466*uk_54 + 307273617*uk_55 + 35798868*uk_56 + 528033303*uk_57 + 647362863*uk_58 + 307273617*uk_59 + 177*uk_6 + 418411108*uk_60 + 458471746*uk_61 + 53414184*uk_62 + 787859214*uk_63 + 965906494*uk_64 + 458471746*uk_65 + 502367977*uk_66 + 58528308*uk_67 + 863292543*uk_68 + 1058386903*uk_69 + 217*uk_7 + 502367977*uk_70 + 6818832*uk_71 + 100577772*uk_72 + 123307212*uk_73 + 58528308*uk_74 + 1483522137*uk_75 + 1818781377*uk_76 + 863292543*uk_77 + 2229805417*uk_78 + 1058386903*uk_79 + 103*uk_8 + 502367977*uk_80 + 250047*uk_81 + 373086*uk_82 + 408807*uk_83 + 47628*uk_84 + 702513*uk_85 + 861273*uk_86 + 408807*uk_87 + 556668*uk_88 + 609966*uk_89 + 2242306609*uk_9 + 71064*uk_90 + 1048194*uk_91 + 1285074*uk_92 + 609966*uk_93 + 668367*uk_94 + 77868*uk_95 + 1148553*uk_96 + 1408113*uk_97 + 668367*uk_98 + 9072*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 135324*uk_100 + 164052*uk_101 + 71064*uk_102 + 2018583*uk_103 + 2447109*uk_104 + 1060038*uk_105 + 2966607*uk_106 + 1285074*uk_107 + 556668*uk_108 + 912673*uk_109 + 4593241*uk_11 + 884446*uk_110 + 112908*uk_111 + 1684211*uk_112 + 2041753*uk_113 + 884446*uk_114 + 857092*uk_115 + 109416*uk_116 + 1632122*uk_117 + 1978606*uk_118 + 857092*uk_119 + 4451182*uk_12 + 13968*uk_120 + 208356*uk_121 + 252588*uk_122 + 109416*uk_123 + 3107977*uk_124 + 3767771*uk_125 + 1632122*uk_126 + 4567633*uk_127 + 1978606*uk_128 + 857092*uk_129 + 568236*uk_13 + 830584*uk_130 + 106032*uk_131 + 1581644*uk_132 + 1917412*uk_133 + 830584*uk_134 + 13536*uk_135 + 201912*uk_136 + 244776*uk_137 + 106032*uk_138 + 3011854*uk_139 + 8476187*uk_14 + 3651242*uk_140 + 1581644*uk_141 + 4426366*uk_142 + 1917412*uk_143 + 830584*uk_144 + 1728*uk_145 + 25776*uk_146 + 31248*uk_147 + 13536*uk_148 + 384492*uk_149 + 10275601*uk_15 + 466116*uk_150 + 201912*uk_151 + 565068*uk_152 + 244776*uk_153 + 106032*uk_154 + 5735339*uk_155 + 6952897*uk_156 + 3011854*uk_157 + 8428931*uk_158 + 3651242*uk_159 + 4451182*uk_16 + 1581644*uk_160 + 10218313*uk_161 + 4426366*uk_162 + 1917412*uk_163 + 830584*uk_164 + 3969*uk_17 + 6111*uk_18 + 5922*uk_19 + 63*uk_2 + 756*uk_20 + 11277*uk_21 + 13671*uk_22 + 5922*uk_23 + 9409*uk_24 + 9118*uk_25 + 1164*uk_26 + 17363*uk_27 + 21049*uk_28 + 9118*uk_29 + 97*uk_3 + 8836*uk_30 + 1128*uk_31 + 16826*uk_32 + 20398*uk_33 + 8836*uk_34 + 144*uk_35 + 2148*uk_36 + 2604*uk_37 + 1128*uk_38 + 32041*uk_39 + 94*uk_4 + 38843*uk_40 + 16826*uk_41 + 47089*uk_42 + 20398*uk_43 + 8836*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 217503741073*uk_47 + 210776821246*uk_48 + 26907679308*uk_49 + 12*uk_5 + 401372883011*uk_50 + 486580534153*uk_51 + 210776821246*uk_52 + 187944057*uk_53 + 289374183*uk_54 + 280424466*uk_55 + 35798868*uk_56 + 533999781*uk_57 + 647362863*uk_58 + 280424466*uk_59 + 179*uk_6 + 445544377*uk_60 + 431764654*uk_61 + 55118892*uk_62 + 822190139*uk_63 + 996733297*uk_64 + 431764654*uk_65 + 418411108*uk_66 + 53414184*uk_67 + 796761578*uk_68 + 965906494*uk_69 + 217*uk_7 + 418411108*uk_70 + 6818832*uk_71 + 101714244*uk_72 + 123307212*uk_73 + 53414184*uk_74 + 1517237473*uk_75 + 1839332579*uk_76 + 796761578*uk_77 + 2229805417*uk_78 + 965906494*uk_79 + 94*uk_8 + 418411108*uk_80 + 250047*uk_81 + 384993*uk_82 + 373086*uk_83 + 47628*uk_84 + 710451*uk_85 + 861273*uk_86 + 373086*uk_87 + 592767*uk_88 + 574434*uk_89 + 2242306609*uk_9 + 73332*uk_90 + 1093869*uk_91 + 1326087*uk_92 + 574434*uk_93 + 556668*uk_94 + 71064*uk_95 + 1060038*uk_96 + 1285074*uk_97 + 556668*uk_98 + 9072*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 136836*uk_100 + 164052*uk_101 + 73332*uk_102 + 2063943*uk_103 + 2474451*uk_104 + 1106091*uk_105 + 2966607*uk_106 + 1326087*uk_107 + 592767*uk_108 + 1404928*uk_109 + 5303536*uk_11 + 1216768*uk_110 + 150528*uk_111 + 2270464*uk_112 + 2722048*uk_113 + 1216768*uk_114 + 1053808*uk_115 + 130368*uk_116 + 1966384*uk_117 + 2357488*uk_118 + 1053808*uk_119 + 4593241*uk_12 + 16128*uk_120 + 243264*uk_121 + 291648*uk_122 + 130368*uk_123 + 3669232*uk_124 + 4399024*uk_125 + 1966384*uk_126 + 5273968*uk_127 + 2357488*uk_128 + 1053808*uk_129 + 568236*uk_13 + 912673*uk_130 + 112908*uk_131 + 1703029*uk_132 + 2041753*uk_133 + 912673*uk_134 + 13968*uk_135 + 210684*uk_136 + 252588*uk_137 + 112908*uk_138 + 3177817*uk_139 + 8570893*uk_14 + 3809869*uk_140 + 1703029*uk_141 + 4567633*uk_142 + 2041753*uk_143 + 912673*uk_144 + 1728*uk_145 + 26064*uk_146 + 31248*uk_147 + 13968*uk_148 + 393132*uk_149 + 10275601*uk_15 + 471324*uk_150 + 210684*uk_151 + 565068*uk_152 + 252588*uk_153 + 112908*uk_154 + 5929741*uk_155 + 7109137*uk_156 + 3177817*uk_157 + 8523109*uk_158 + 3809869*uk_159 + 4593241*uk_16 + 1703029*uk_160 + 10218313*uk_161 + 4567633*uk_162 + 2041753*uk_163 + 912673*uk_164 + 3969*uk_17 + 7056*uk_18 + 6111*uk_19 + 63*uk_2 + 756*uk_20 + 11403*uk_21 + 13671*uk_22 + 6111*uk_23 + 12544*uk_24 + 10864*uk_25 + 1344*uk_26 + 20272*uk_27 + 24304*uk_28 + 10864*uk_29 + 112*uk_3 + 9409*uk_30 + 1164*uk_31 + 17557*uk_32 + 21049*uk_33 + 9409*uk_34 + 144*uk_35 + 2172*uk_36 + 2604*uk_37 + 1164*uk_38 + 32761*uk_39 + 97*uk_4 + 39277*uk_40 + 17557*uk_41 + 47089*uk_42 + 21049*uk_43 + 9409*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 251138340208*uk_47 + 217503741073*uk_48 + 26907679308*uk_49 + 12*uk_5 + 405857496229*uk_50 + 486580534153*uk_51 + 217503741073*uk_52 + 187944057*uk_53 + 334122768*uk_54 + 289374183*uk_55 + 35798868*uk_56 + 539966259*uk_57 + 647362863*uk_58 + 289374183*uk_59 + 181*uk_6 + 593996032*uk_60 + 514442992*uk_61 + 63642432*uk_62 + 959940016*uk_63 + 1150867312*uk_64 + 514442992*uk_65 + 445544377*uk_66 + 55118892*uk_67 + 831376621*uk_68 + 996733297*uk_69 + 217*uk_7 + 445544377*uk_70 + 6818832*uk_71 + 102850716*uk_72 + 123307212*uk_73 + 55118892*uk_74 + 1551331633*uk_75 + 1859883781*uk_76 + 831376621*uk_77 + 2229805417*uk_78 + 996733297*uk_79 + 97*uk_8 + 445544377*uk_80 + 250047*uk_81 + 444528*uk_82 + 384993*uk_83 + 47628*uk_84 + 718389*uk_85 + 861273*uk_86 + 384993*uk_87 + 790272*uk_88 + 684432*uk_89 + 2242306609*uk_9 + 84672*uk_90 + 1277136*uk_91 + 1531152*uk_92 + 684432*uk_93 + 592767*uk_94 + 73332*uk_95 + 1106091*uk_96 + 1326087*uk_97 + 592767*uk_98 + 9072*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 138348*uk_100 + 164052*uk_101 + 84672*uk_102 + 2109807*uk_103 + 2501793*uk_104 + 1291248*uk_105 + 2966607*uk_106 + 1531152*uk_107 + 790272*uk_108 + 2685619*uk_109 + 6582067*uk_11 + 2163952*uk_110 + 231852*uk_111 + 3535743*uk_112 + 4192657*uk_113 + 2163952*uk_114 + 1743616*uk_115 + 186816*uk_116 + 2848944*uk_117 + 3378256*uk_118 + 1743616*uk_119 + 5303536*uk_12 + 20016*uk_120 + 305244*uk_121 + 361956*uk_122 + 186816*uk_123 + 4654971*uk_124 + 5519829*uk_125 + 2848944*uk_126 + 6545371*uk_127 + 3378256*uk_128 + 1743616*uk_129 + 568236*uk_13 + 1404928*uk_130 + 150528*uk_131 + 2295552*uk_132 + 2722048*uk_133 + 1404928*uk_134 + 16128*uk_135 + 245952*uk_136 + 291648*uk_137 + 150528*uk_138 + 3750768*uk_139 + 8665599*uk_14 + 4447632*uk_140 + 2295552*uk_141 + 5273968*uk_142 + 2722048*uk_143 + 1404928*uk_144 + 1728*uk_145 + 26352*uk_146 + 31248*uk_147 + 16128*uk_148 + 401868*uk_149 + 10275601*uk_15 + 476532*uk_150 + 245952*uk_151 + 565068*uk_152 + 291648*uk_153 + 150528*uk_154 + 6128487*uk_155 + 7267113*uk_156 + 3750768*uk_157 + 8617287*uk_158 + 4447632*uk_159 + 5303536*uk_16 + 2295552*uk_160 + 10218313*uk_161 + 5273968*uk_162 + 2722048*uk_163 + 1404928*uk_164 + 3969*uk_17 + 8757*uk_18 + 7056*uk_19 + 63*uk_2 + 756*uk_20 + 11529*uk_21 + 13671*uk_22 + 7056*uk_23 + 19321*uk_24 + 15568*uk_25 + 1668*uk_26 + 25437*uk_27 + 30163*uk_28 + 15568*uk_29 + 139*uk_3 + 12544*uk_30 + 1344*uk_31 + 20496*uk_32 + 24304*uk_33 + 12544*uk_34 + 144*uk_35 + 2196*uk_36 + 2604*uk_37 + 1344*uk_38 + 33489*uk_39 + 112*uk_4 + 39711*uk_40 + 20496*uk_41 + 47089*uk_42 + 24304*uk_43 + 12544*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 311680618651*uk_47 + 251138340208*uk_48 + 26907679308*uk_49 + 12*uk_5 + 410342109447*uk_50 + 486580534153*uk_51 + 251138340208*uk_52 + 187944057*uk_53 + 414670221*uk_54 + 334122768*uk_55 + 35798868*uk_56 + 545932737*uk_57 + 647362863*uk_58 + 334122768*uk_59 + 183*uk_6 + 914907313*uk_60 + 737191504*uk_61 + 78984804*uk_62 + 1204518261*uk_63 + 1428308539*uk_64 + 737191504*uk_65 + 593996032*uk_66 + 63642432*uk_67 + 970547088*uk_68 + 1150867312*uk_69 + 217*uk_7 + 593996032*uk_70 + 6818832*uk_71 + 103987188*uk_72 + 123307212*uk_73 + 63642432*uk_74 + 1585804617*uk_75 + 1880434983*uk_76 + 970547088*uk_77 + 2229805417*uk_78 + 1150867312*uk_79 + 112*uk_8 + 593996032*uk_80 + 250047*uk_81 + 551691*uk_82 + 444528*uk_83 + 47628*uk_84 + 726327*uk_85 + 861273*uk_86 + 444528*uk_87 + 1217223*uk_88 + 980784*uk_89 + 2242306609*uk_9 + 105084*uk_90 + 1602531*uk_91 + 1900269*uk_92 + 980784*uk_93 + 790272*uk_94 + 84672*uk_95 + 1291248*uk_96 + 1531152*uk_97 + 790272*uk_98 + 9072*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 139860*uk_100 + 164052*uk_101 + 105084*uk_102 + 2156175*uk_103 + 2529135*uk_104 + 1620045*uk_105 + 2966607*uk_106 + 1900269*uk_107 + 1217223*uk_108 + 5639752*uk_109 + 8428834*uk_11 + 4404076*uk_110 + 380208*uk_111 + 5861540*uk_112 + 6875428*uk_113 + 4404076*uk_114 + 3439138*uk_115 + 296904*uk_116 + 4577270*uk_117 + 5369014*uk_118 + 3439138*uk_119 + 6582067*uk_12 + 25632*uk_120 + 395160*uk_121 + 463512*uk_122 + 296904*uk_123 + 6092050*uk_124 + 7145810*uk_125 + 4577270*uk_126 + 8381842*uk_127 + 5369014*uk_128 + 3439138*uk_129 + 568236*uk_13 + 2685619*uk_130 + 231852*uk_131 + 3574385*uk_132 + 4192657*uk_133 + 2685619*uk_134 + 20016*uk_135 + 308580*uk_136 + 361956*uk_137 + 231852*uk_138 + 4757275*uk_139 + 8760305*uk_14 + 5580155*uk_140 + 3574385*uk_141 + 6545371*uk_142 + 4192657*uk_143 + 2685619*uk_144 + 1728*uk_145 + 26640*uk_146 + 31248*uk_147 + 20016*uk_148 + 410700*uk_149 + 10275601*uk_15 + 481740*uk_150 + 308580*uk_151 + 565068*uk_152 + 361956*uk_153 + 231852*uk_154 + 6331625*uk_155 + 7426825*uk_156 + 4757275*uk_157 + 8711465*uk_158 + 5580155*uk_159 + 6582067*uk_16 + 3574385*uk_160 + 10218313*uk_161 + 6545371*uk_162 + 4192657*uk_163 + 2685619*uk_164 + 3969*uk_17 + 11214*uk_18 + 8757*uk_19 + 63*uk_2 + 756*uk_20 + 11655*uk_21 + 13671*uk_22 + 8757*uk_23 + 31684*uk_24 + 24742*uk_25 + 2136*uk_26 + 32930*uk_27 + 38626*uk_28 + 24742*uk_29 + 178*uk_3 + 19321*uk_30 + 1668*uk_31 + 25715*uk_32 + 30163*uk_33 + 19321*uk_34 + 144*uk_35 + 2220*uk_36 + 2604*uk_37 + 1668*uk_38 + 34225*uk_39 + 139*uk_4 + 40145*uk_40 + 25715*uk_41 + 47089*uk_42 + 30163*uk_43 + 19321*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 399130576402*uk_47 + 311680618651*uk_48 + 26907679308*uk_49 + 12*uk_5 + 414826722665*uk_50 + 486580534153*uk_51 + 311680618651*uk_52 + 187944057*uk_53 + 531016542*uk_54 + 414670221*uk_55 + 35798868*uk_56 + 551899215*uk_57 + 647362863*uk_58 + 414670221*uk_59 + 185*uk_6 + 1500332452*uk_60 + 1171607926*uk_61 + 101146008*uk_62 + 1559334290*uk_63 + 1829056978*uk_64 + 1171607926*uk_65 + 914907313*uk_66 + 78984804*uk_67 + 1217682395*uk_68 + 1428308539*uk_69 + 217*uk_7 + 914907313*uk_70 + 6818832*uk_71 + 105123660*uk_72 + 123307212*uk_73 + 78984804*uk_74 + 1620656425*uk_75 + 1900986185*uk_76 + 1217682395*uk_77 + 2229805417*uk_78 + 1428308539*uk_79 + 139*uk_8 + 914907313*uk_80 + 250047*uk_81 + 706482*uk_82 + 551691*uk_83 + 47628*uk_84 + 734265*uk_85 + 861273*uk_86 + 551691*uk_87 + 1996092*uk_88 + 1558746*uk_89 + 2242306609*uk_9 + 134568*uk_90 + 2074590*uk_91 + 2433438*uk_92 + 1558746*uk_93 + 1217223*uk_94 + 105084*uk_95 + 1620045*uk_96 + 1900269*uk_97 + 1217223*uk_98 + 9072*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 94248*uk_100 + 109368*uk_101 + 89712*uk_102 + 2203047*uk_103 + 2556477*uk_104 + 2097018*uk_105 + 2966607*uk_106 + 2433438*uk_107 + 1996092*uk_108 + 74088*uk_109 + 1988826*uk_11 + 313992*uk_110 + 14112*uk_111 + 329868*uk_112 + 382788*uk_113 + 313992*uk_114 + 1330728*uk_115 + 59808*uk_116 + 1398012*uk_117 + 1622292*uk_118 + 1330728*uk_119 + 8428834*uk_12 + 2688*uk_120 + 62832*uk_121 + 72912*uk_122 + 59808*uk_123 + 1468698*uk_124 + 1704318*uk_125 + 1398012*uk_126 + 1977738*uk_127 + 1622292*uk_128 + 1330728*uk_129 + 378824*uk_13 + 5639752*uk_130 + 253472*uk_131 + 5924908*uk_132 + 6875428*uk_133 + 5639752*uk_134 + 11392*uk_135 + 266288*uk_136 + 309008*uk_137 + 253472*uk_138 + 6224482*uk_139 + 8855011*uk_14 + 7223062*uk_140 + 5924908*uk_141 + 8381842*uk_142 + 6875428*uk_143 + 5639752*uk_144 + 512*uk_145 + 11968*uk_146 + 13888*uk_147 + 11392*uk_148 + 279752*uk_149 + 10275601*uk_15 + 324632*uk_150 + 266288*uk_151 + 376712*uk_152 + 309008*uk_153 + 253472*uk_154 + 6539203*uk_155 + 7588273*uk_156 + 6224482*uk_157 + 8805643*uk_158 + 7223062*uk_159 + 8428834*uk_16 + 5924908*uk_160 + 10218313*uk_161 + 8381842*uk_162 + 6875428*uk_163 + 5639752*uk_164 + 3969*uk_17 + 2646*uk_18 + 11214*uk_19 + 63*uk_2 + 504*uk_20 + 11781*uk_21 + 13671*uk_22 + 11214*uk_23 + 1764*uk_24 + 7476*uk_25 + 336*uk_26 + 7854*uk_27 + 9114*uk_28 + 7476*uk_29 + 42*uk_3 + 31684*uk_30 + 1424*uk_31 + 33286*uk_32 + 38626*uk_33 + 31684*uk_34 + 64*uk_35 + 1496*uk_36 + 1736*uk_37 + 1424*uk_38 + 34969*uk_39 + 178*uk_4 + 40579*uk_40 + 33286*uk_41 + 47089*uk_42 + 38626*uk_43 + 31684*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 94176877578*uk_47 + 399130576402*uk_48 + 17938452872*uk_49 + 8*uk_5 + 419311335883*uk_50 + 486580534153*uk_51 + 399130576402*uk_52 + 187944057*uk_53 + 125296038*uk_54 + 531016542*uk_55 + 23865912*uk_56 + 557865693*uk_57 + 647362863*uk_58 + 531016542*uk_59 + 187*uk_6 + 83530692*uk_60 + 354011028*uk_61 + 15910608*uk_62 + 371910462*uk_63 + 431575242*uk_64 + 354011028*uk_65 + 1500332452*uk_66 + 67430672*uk_67 + 1576191958*uk_68 + 1829056978*uk_69 + 217*uk_7 + 1500332452*uk_70 + 3030592*uk_71 + 70840088*uk_72 + 82204808*uk_73 + 67430672*uk_74 + 1655887057*uk_75 + 1921537387*uk_76 + 1576191958*uk_77 + 2229805417*uk_78 + 1829056978*uk_79 + 178*uk_8 + 1500332452*uk_80 + 250047*uk_81 + 166698*uk_82 + 706482*uk_83 + 31752*uk_84 + 742203*uk_85 + 861273*uk_86 + 706482*uk_87 + 111132*uk_88 + 470988*uk_89 + 2242306609*uk_9 + 21168*uk_90 + 494802*uk_91 + 574182*uk_92 + 470988*uk_93 + 1996092*uk_94 + 89712*uk_95 + 2097018*uk_96 + 2433438*uk_97 + 1996092*uk_98 + 4032*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 142884*uk_100 + 164052*uk_101 + 31752*uk_102 + 2250423*uk_103 + 2583819*uk_104 + 500094*uk_105 + 2966607*uk_106 + 574182*uk_107 + 111132*uk_108 + 1092727*uk_109 + 4877359*uk_11 + 445578*uk_110 + 127308*uk_111 + 2005101*uk_112 + 2302153*uk_113 + 445578*uk_114 + 181692*uk_115 + 51912*uk_116 + 817614*uk_117 + 938742*uk_118 + 181692*uk_119 + 1988826*uk_12 + 14832*uk_120 + 233604*uk_121 + 268212*uk_122 + 51912*uk_123 + 3679263*uk_124 + 4224339*uk_125 + 817614*uk_126 + 4850167*uk_127 + 938742*uk_128 + 181692*uk_129 + 568236*uk_13 + 74088*uk_130 + 21168*uk_131 + 333396*uk_132 + 382788*uk_133 + 74088*uk_134 + 6048*uk_135 + 95256*uk_136 + 109368*uk_137 + 21168*uk_138 + 1500282*uk_139 + 8949717*uk_14 + 1722546*uk_140 + 333396*uk_141 + 1977738*uk_142 + 382788*uk_143 + 74088*uk_144 + 1728*uk_145 + 27216*uk_146 + 31248*uk_147 + 6048*uk_148 + 428652*uk_149 + 10275601*uk_15 + 492156*uk_150 + 95256*uk_151 + 565068*uk_152 + 109368*uk_153 + 21168*uk_154 + 6751269*uk_155 + 7751457*uk_156 + 1500282*uk_157 + 8899821*uk_158 + 1722546*uk_159 + 1988826*uk_16 + 333396*uk_160 + 10218313*uk_161 + 1977738*uk_162 + 382788*uk_163 + 74088*uk_164 + 3969*uk_17 + 6489*uk_18 + 2646*uk_19 + 63*uk_2 + 756*uk_20 + 11907*uk_21 + 13671*uk_22 + 2646*uk_23 + 10609*uk_24 + 4326*uk_25 + 1236*uk_26 + 19467*uk_27 + 22351*uk_28 + 4326*uk_29 + 103*uk_3 + 1764*uk_30 + 504*uk_31 + 7938*uk_32 + 9114*uk_33 + 1764*uk_34 + 144*uk_35 + 2268*uk_36 + 2604*uk_37 + 504*uk_38 + 35721*uk_39 + 42*uk_4 + 41013*uk_40 + 7938*uk_41 + 47089*uk_42 + 9114*uk_43 + 1764*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 230957580727*uk_47 + 94176877578*uk_48 + 26907679308*uk_49 + 12*uk_5 + 423795949101*uk_50 + 486580534153*uk_51 + 94176877578*uk_52 + 187944057*uk_53 + 307273617*uk_54 + 125296038*uk_55 + 35798868*uk_56 + 563832171*uk_57 + 647362863*uk_58 + 125296038*uk_59 + 189*uk_6 + 502367977*uk_60 + 204849078*uk_61 + 58528308*uk_62 + 921820851*uk_63 + 1058386903*uk_64 + 204849078*uk_65 + 83530692*uk_66 + 23865912*uk_67 + 375888114*uk_68 + 431575242*uk_69 + 217*uk_7 + 83530692*uk_70 + 6818832*uk_71 + 107396604*uk_72 + 123307212*uk_73 + 23865912*uk_74 + 1691496513*uk_75 + 1942088589*uk_76 + 375888114*uk_77 + 2229805417*uk_78 + 431575242*uk_79 + 42*uk_8 + 83530692*uk_80 + 250047*uk_81 + 408807*uk_82 + 166698*uk_83 + 47628*uk_84 + 750141*uk_85 + 861273*uk_86 + 166698*uk_87 + 668367*uk_88 + 272538*uk_89 + 2242306609*uk_9 + 77868*uk_90 + 1226421*uk_91 + 1408113*uk_92 + 272538*uk_93 + 111132*uk_94 + 31752*uk_95 + 500094*uk_96 + 574182*uk_97 + 111132*uk_98 + 9072*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 144396*uk_100 + 164052*uk_101 + 77868*uk_102 + 2298303*uk_103 + 2611161*uk_104 + 1239399*uk_105 + 2966607*uk_106 + 1408113*uk_107 + 668367*uk_108 + 5451776*uk_109 + 8334128*uk_11 + 3190528*uk_110 + 371712*uk_111 + 5916416*uk_112 + 6721792*uk_113 + 3190528*uk_114 + 1867184*uk_115 + 217536*uk_116 + 3462448*uk_117 + 3933776*uk_118 + 1867184*uk_119 + 4877359*uk_12 + 25344*uk_120 + 403392*uk_121 + 458304*uk_122 + 217536*uk_123 + 6420656*uk_124 + 7294672*uk_125 + 3462448*uk_126 + 8287664*uk_127 + 3933776*uk_128 + 1867184*uk_129 + 568236*uk_13 + 1092727*uk_130 + 127308*uk_131 + 2026319*uk_132 + 2302153*uk_133 + 1092727*uk_134 + 14832*uk_135 + 236076*uk_136 + 268212*uk_137 + 127308*uk_138 + 3757543*uk_139 + 9044423*uk_14 + 4269041*uk_140 + 2026319*uk_141 + 4850167*uk_142 + 2302153*uk_143 + 1092727*uk_144 + 1728*uk_145 + 27504*uk_146 + 31248*uk_147 + 14832*uk_148 + 437772*uk_149 + 10275601*uk_15 + 497364*uk_150 + 236076*uk_151 + 565068*uk_152 + 268212*uk_153 + 127308*uk_154 + 6967871*uk_155 + 7916377*uk_156 + 3757543*uk_157 + 8993999*uk_158 + 4269041*uk_159 + 4877359*uk_16 + 2026319*uk_160 + 10218313*uk_161 + 4850167*uk_162 + 2302153*uk_163 + 1092727*uk_164 + 3969*uk_17 + 11088*uk_18 + 6489*uk_19 + 63*uk_2 + 756*uk_20 + 12033*uk_21 + 13671*uk_22 + 6489*uk_23 + 30976*uk_24 + 18128*uk_25 + 2112*uk_26 + 33616*uk_27 + 38192*uk_28 + 18128*uk_29 + 176*uk_3 + 10609*uk_30 + 1236*uk_31 + 19673*uk_32 + 22351*uk_33 + 10609*uk_34 + 144*uk_35 + 2292*uk_36 + 2604*uk_37 + 1236*uk_38 + 36481*uk_39 + 103*uk_4 + 41447*uk_40 + 19673*uk_41 + 47089*uk_42 + 22351*uk_43 + 10609*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 394645963184*uk_47 + 230957580727*uk_48 + 26907679308*uk_49 + 12*uk_5 + 428280562319*uk_50 + 486580534153*uk_51 + 230957580727*uk_52 + 187944057*uk_53 + 525050064*uk_54 + 307273617*uk_55 + 35798868*uk_56 + 569798649*uk_57 + 647362863*uk_58 + 307273617*uk_59 + 191*uk_6 + 1466806528*uk_60 + 858415184*uk_61 + 100009536*uk_62 + 1591818448*uk_63 + 1808505776*uk_64 + 858415184*uk_65 + 502367977*uk_66 + 58528308*uk_67 + 931575569*uk_68 + 1058386903*uk_69 + 217*uk_7 + 502367977*uk_70 + 6818832*uk_71 + 108533076*uk_72 + 123307212*uk_73 + 58528308*uk_74 + 1727484793*uk_75 + 1962639791*uk_76 + 931575569*uk_77 + 2229805417*uk_78 + 1058386903*uk_79 + 103*uk_8 + 502367977*uk_80 + 250047*uk_81 + 698544*uk_82 + 408807*uk_83 + 47628*uk_84 + 758079*uk_85 + 861273*uk_86 + 408807*uk_87 + 1951488*uk_88 + 1142064*uk_89 + 2242306609*uk_9 + 133056*uk_90 + 2117808*uk_91 + 2406096*uk_92 + 1142064*uk_93 + 668367*uk_94 + 77868*uk_95 + 1239399*uk_96 + 1408113*uk_97 + 668367*uk_98 + 9072*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 97272*uk_100 + 109368*uk_101 + 88704*uk_102 + 2346687*uk_103 + 2638503*uk_104 + 2139984*uk_105 + 2966607*uk_106 + 2406096*uk_107 + 1951488*uk_108 + 314432*uk_109 + 3220004*uk_11 + 813824*uk_110 + 36992*uk_111 + 892432*uk_112 + 1003408*uk_113 + 813824*uk_114 + 2106368*uk_115 + 95744*uk_116 + 2309824*uk_117 + 2597056*uk_118 + 2106368*uk_119 + 8334128*uk_12 + 4352*uk_120 + 104992*uk_121 + 118048*uk_122 + 95744*uk_123 + 2532932*uk_124 + 2847908*uk_125 + 2309824*uk_126 + 3202052*uk_127 + 2597056*uk_128 + 2106368*uk_129 + 378824*uk_13 + 5451776*uk_130 + 247808*uk_131 + 5978368*uk_132 + 6721792*uk_133 + 5451776*uk_134 + 11264*uk_135 + 271744*uk_136 + 305536*uk_137 + 247808*uk_138 + 6555824*uk_139 + 9139129*uk_14 + 7371056*uk_140 + 5978368*uk_141 + 8287664*uk_142 + 6721792*uk_143 + 5451776*uk_144 + 512*uk_145 + 12352*uk_146 + 13888*uk_147 + 11264*uk_148 + 297992*uk_149 + 10275601*uk_15 + 335048*uk_150 + 271744*uk_151 + 376712*uk_152 + 305536*uk_153 + 247808*uk_154 + 7189057*uk_155 + 8083033*uk_156 + 6555824*uk_157 + 9088177*uk_158 + 7371056*uk_159 + 8334128*uk_16 + 5978368*uk_160 + 10218313*uk_161 + 8287664*uk_162 + 6721792*uk_163 + 5451776*uk_164 + 3969*uk_17 + 4284*uk_18 + 11088*uk_19 + 63*uk_2 + 504*uk_20 + 12159*uk_21 + 13671*uk_22 + 11088*uk_23 + 4624*uk_24 + 11968*uk_25 + 544*uk_26 + 13124*uk_27 + 14756*uk_28 + 11968*uk_29 + 68*uk_3 + 30976*uk_30 + 1408*uk_31 + 33968*uk_32 + 38192*uk_33 + 30976*uk_34 + 64*uk_35 + 1544*uk_36 + 1736*uk_37 + 1408*uk_38 + 37249*uk_39 + 176*uk_4 + 41881*uk_40 + 33968*uk_41 + 47089*uk_42 + 38192*uk_43 + 30976*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 152476849412*uk_47 + 394645963184*uk_48 + 17938452872*uk_49 + 8*uk_5 + 432765175537*uk_50 + 486580534153*uk_51 + 394645963184*uk_52 + 187944057*uk_53 + 202860252*uk_54 + 525050064*uk_55 + 23865912*uk_56 + 575765127*uk_57 + 647362863*uk_58 + 525050064*uk_59 + 193*uk_6 + 218960272*uk_60 + 566720704*uk_61 + 25760032*uk_62 + 621460772*uk_63 + 698740868*uk_64 + 566720704*uk_65 + 1466806528*uk_66 + 66673024*uk_67 + 1608486704*uk_68 + 1808505776*uk_69 + 217*uk_7 + 1466806528*uk_70 + 3030592*uk_71 + 73113032*uk_72 + 82204808*uk_73 + 66673024*uk_74 + 1763851897*uk_75 + 1983190993*uk_76 + 1608486704*uk_77 + 2229805417*uk_78 + 1808505776*uk_79 + 176*uk_8 + 1466806528*uk_80 + 250047*uk_81 + 269892*uk_82 + 698544*uk_83 + 31752*uk_84 + 766017*uk_85 + 861273*uk_86 + 698544*uk_87 + 291312*uk_88 + 753984*uk_89 + 2242306609*uk_9 + 34272*uk_90 + 826812*uk_91 + 929628*uk_92 + 753984*uk_93 + 1951488*uk_94 + 88704*uk_95 + 2139984*uk_96 + 2406096*uk_97 + 1951488*uk_98 + 4032*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 147420*uk_100 + 164052*uk_101 + 51408*uk_102 + 2395575*uk_103 + 2665845*uk_104 + 835380*uk_105 + 2966607*uk_106 + 929628*uk_107 + 291312*uk_108 + 4330747*uk_109 + 7718539*uk_11 + 1806692*uk_110 + 318828*uk_111 + 5180955*uk_112 + 5765473*uk_113 + 1806692*uk_114 + 753712*uk_115 + 133008*uk_116 + 2161380*uk_117 + 2405228*uk_118 + 753712*uk_119 + 3220004*uk_12 + 23472*uk_120 + 381420*uk_121 + 424452*uk_122 + 133008*uk_123 + 6198075*uk_124 + 6897345*uk_125 + 2161380*uk_126 + 7675507*uk_127 + 2405228*uk_128 + 753712*uk_129 + 568236*uk_13 + 314432*uk_130 + 55488*uk_131 + 901680*uk_132 + 1003408*uk_133 + 314432*uk_134 + 9792*uk_135 + 159120*uk_136 + 177072*uk_137 + 55488*uk_138 + 2585700*uk_139 + 9233835*uk_14 + 2877420*uk_140 + 901680*uk_141 + 3202052*uk_142 + 1003408*uk_143 + 314432*uk_144 + 1728*uk_145 + 28080*uk_146 + 31248*uk_147 + 9792*uk_148 + 456300*uk_149 + 10275601*uk_15 + 507780*uk_150 + 159120*uk_151 + 565068*uk_152 + 177072*uk_153 + 55488*uk_154 + 7414875*uk_155 + 8251425*uk_156 + 2585700*uk_157 + 9182355*uk_158 + 2877420*uk_159 + 3220004*uk_16 + 901680*uk_160 + 10218313*uk_161 + 3202052*uk_162 + 1003408*uk_163 + 314432*uk_164 + 3969*uk_17 + 10269*uk_18 + 4284*uk_19 + 63*uk_2 + 756*uk_20 + 12285*uk_21 + 13671*uk_22 + 4284*uk_23 + 26569*uk_24 + 11084*uk_25 + 1956*uk_26 + 31785*uk_27 + 35371*uk_28 + 11084*uk_29 + 163*uk_3 + 4624*uk_30 + 816*uk_31 + 13260*uk_32 + 14756*uk_33 + 4624*uk_34 + 144*uk_35 + 2340*uk_36 + 2604*uk_37 + 816*uk_38 + 38025*uk_39 + 68*uk_4 + 42315*uk_40 + 13260*uk_41 + 47089*uk_42 + 14756*uk_43 + 4624*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 365495977267*uk_47 + 152476849412*uk_48 + 26907679308*uk_49 + 12*uk_5 + 437249788755*uk_50 + 486580534153*uk_51 + 152476849412*uk_52 + 187944057*uk_53 + 486267957*uk_54 + 202860252*uk_55 + 35798868*uk_56 + 581731605*uk_57 + 647362863*uk_58 + 202860252*uk_59 + 195*uk_6 + 1258121857*uk_60 + 524860652*uk_61 + 92622468*uk_62 + 1505115105*uk_63 + 1674922963*uk_64 + 524860652*uk_65 + 218960272*uk_66 + 38640048*uk_67 + 627900780*uk_68 + 698740868*uk_69 + 217*uk_7 + 218960272*uk_70 + 6818832*uk_71 + 110806020*uk_72 + 123307212*uk_73 + 38640048*uk_74 + 1800597825*uk_75 + 2003742195*uk_76 + 627900780*uk_77 + 2229805417*uk_78 + 698740868*uk_79 + 68*uk_8 + 218960272*uk_80 + 250047*uk_81 + 646947*uk_82 + 269892*uk_83 + 47628*uk_84 + 773955*uk_85 + 861273*uk_86 + 269892*uk_87 + 1673847*uk_88 + 698292*uk_89 + 2242306609*uk_9 + 123228*uk_90 + 2002455*uk_91 + 2228373*uk_92 + 698292*uk_93 + 291312*uk_94 + 51408*uk_95 + 835380*uk_96 + 929628*uk_97 + 291312*uk_98 + 9072*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 99288*uk_100 + 109368*uk_101 + 82152*uk_102 + 2444967*uk_103 + 2693187*uk_104 + 2022993*uk_105 + 2966607*uk_106 + 2228373*uk_107 + 1673847*uk_108 + 389017*uk_109 + 3456769*uk_11 + 868627*uk_110 + 42632*uk_111 + 1049813*uk_112 + 1156393*uk_113 + 868627*uk_114 + 1939537*uk_115 + 95192*uk_116 + 2344103*uk_117 + 2582083*uk_118 + 1939537*uk_119 + 7718539*uk_12 + 4672*uk_120 + 115048*uk_121 + 126728*uk_122 + 95192*uk_123 + 2833057*uk_124 + 3120677*uk_125 + 2344103*uk_126 + 3437497*uk_127 + 2582083*uk_128 + 1939537*uk_129 + 378824*uk_13 + 4330747*uk_130 + 212552*uk_131 + 5234093*uk_132 + 5765473*uk_133 + 4330747*uk_134 + 10432*uk_135 + 256888*uk_136 + 282968*uk_137 + 212552*uk_138 + 6325867*uk_139 + 9328541*uk_14 + 6968087*uk_140 + 5234093*uk_141 + 7675507*uk_142 + 5765473*uk_143 + 4330747*uk_144 + 512*uk_145 + 12608*uk_146 + 13888*uk_147 + 10432*uk_148 + 310472*uk_149 + 10275601*uk_15 + 341992*uk_150 + 256888*uk_151 + 376712*uk_152 + 282968*uk_153 + 212552*uk_154 + 7645373*uk_155 + 8421553*uk_156 + 6325867*uk_157 + 9276533*uk_158 + 6968087*uk_159 + 7718539*uk_16 + 5234093*uk_160 + 10218313*uk_161 + 7675507*uk_162 + 5765473*uk_163 + 4330747*uk_164 + 3969*uk_17 + 4599*uk_18 + 10269*uk_19 + 63*uk_2 + 504*uk_20 + 12411*uk_21 + 13671*uk_22 + 10269*uk_23 + 5329*uk_24 + 11899*uk_25 + 584*uk_26 + 14381*uk_27 + 15841*uk_28 + 11899*uk_29 + 73*uk_3 + 26569*uk_30 + 1304*uk_31 + 32111*uk_32 + 35371*uk_33 + 26569*uk_34 + 64*uk_35 + 1576*uk_36 + 1736*uk_37 + 1304*uk_38 + 38809*uk_39 + 163*uk_4 + 42749*uk_40 + 32111*uk_41 + 47089*uk_42 + 35371*uk_43 + 26569*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 163688382457*uk_47 + 365495977267*uk_48 + 17938452872*uk_49 + 8*uk_5 + 441734401973*uk_50 + 486580534153*uk_51 + 365495977267*uk_52 + 187944057*uk_53 + 217776447*uk_54 + 486267957*uk_55 + 23865912*uk_56 + 587698083*uk_57 + 647362863*uk_58 + 486267957*uk_59 + 197*uk_6 + 252344137*uk_60 + 563453347*uk_61 + 27654152*uk_62 + 680983493*uk_63 + 750118873*uk_64 + 563453347*uk_65 + 1258121857*uk_66 + 61748312*uk_67 + 1520552183*uk_68 + 1674922963*uk_69 + 217*uk_7 + 1258121857*uk_70 + 3030592*uk_71 + 74628328*uk_72 + 82204808*uk_73 + 61748312*uk_74 + 1837722577*uk_75 + 2024293397*uk_76 + 1520552183*uk_77 + 2229805417*uk_78 + 1674922963*uk_79 + 163*uk_8 + 1258121857*uk_80 + 250047*uk_81 + 289737*uk_82 + 646947*uk_83 + 31752*uk_84 + 781893*uk_85 + 861273*uk_86 + 646947*uk_87 + 335727*uk_88 + 749637*uk_89 + 2242306609*uk_9 + 36792*uk_90 + 906003*uk_91 + 997983*uk_92 + 749637*uk_93 + 1673847*uk_94 + 82152*uk_95 + 2022993*uk_96 + 2228373*uk_97 + 1673847*uk_98 + 4032*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 150444*uk_100 + 164052*uk_101 + 55188*uk_102 + 2494863*uk_103 + 2720529*uk_104 + 915201*uk_105 + 2966607*uk_106 + 997983*uk_107 + 335727*uk_108 + 6859000*uk_109 + 8997070*uk_11 + 2635300*uk_110 + 433200*uk_111 + 7183900*uk_112 + 7833700*uk_113 + 2635300*uk_114 + 1012510*uk_115 + 166440*uk_116 + 2760130*uk_117 + 3009790*uk_118 + 1012510*uk_119 + 3456769*uk_12 + 27360*uk_120 + 453720*uk_121 + 494760*uk_122 + 166440*uk_123 + 7524190*uk_124 + 8204770*uk_125 + 2760130*uk_126 + 8946910*uk_127 + 3009790*uk_128 + 1012510*uk_129 + 568236*uk_13 + 389017*uk_130 + 63948*uk_131 + 1060471*uk_132 + 1156393*uk_133 + 389017*uk_134 + 10512*uk_135 + 174324*uk_136 + 190092*uk_137 + 63948*uk_138 + 2890873*uk_139 + 9423247*uk_14 + 3152359*uk_140 + 1060471*uk_141 + 3437497*uk_142 + 1156393*uk_143 + 389017*uk_144 + 1728*uk_145 + 28656*uk_146 + 31248*uk_147 + 10512*uk_148 + 475212*uk_149 + 10275601*uk_15 + 518196*uk_150 + 174324*uk_151 + 565068*uk_152 + 190092*uk_153 + 63948*uk_154 + 7880599*uk_155 + 8593417*uk_156 + 2890873*uk_157 + 9370711*uk_158 + 3152359*uk_159 + 3456769*uk_16 + 1060471*uk_160 + 10218313*uk_161 + 3437497*uk_162 + 1156393*uk_163 + 389017*uk_164 + 3969*uk_17 + 11970*uk_18 + 4599*uk_19 + 63*uk_2 + 756*uk_20 + 12537*uk_21 + 13671*uk_22 + 4599*uk_23 + 36100*uk_24 + 13870*uk_25 + 2280*uk_26 + 37810*uk_27 + 41230*uk_28 + 13870*uk_29 + 190*uk_3 + 5329*uk_30 + 876*uk_31 + 14527*uk_32 + 15841*uk_33 + 5329*uk_34 + 144*uk_35 + 2388*uk_36 + 2604*uk_37 + 876*uk_38 + 39601*uk_39 + 73*uk_4 + 43183*uk_40 + 14527*uk_41 + 47089*uk_42 + 15841*uk_43 + 5329*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 426038255710*uk_47 + 163688382457*uk_48 + 26907679308*uk_49 + 12*uk_5 + 446219015191*uk_50 + 486580534153*uk_51 + 163688382457*uk_52 + 187944057*uk_53 + 566815410*uk_54 + 217776447*uk_55 + 35798868*uk_56 + 593664561*uk_57 + 647362863*uk_58 + 217776447*uk_59 + 199*uk_6 + 1709443300*uk_60 + 656786110*uk_61 + 107964840*uk_62 + 1790416930*uk_63 + 1952364190*uk_64 + 656786110*uk_65 + 252344137*uk_66 + 41481228*uk_67 + 687897031*uk_68 + 750118873*uk_69 + 217*uk_7 + 252344137*uk_70 + 6818832*uk_71 + 113078964*uk_72 + 123307212*uk_73 + 41481228*uk_74 + 1875226153*uk_75 + 2044844599*uk_76 + 687897031*uk_77 + 2229805417*uk_78 + 750118873*uk_79 + 73*uk_8 + 252344137*uk_80 + 250047*uk_81 + 754110*uk_82 + 289737*uk_83 + 47628*uk_84 + 789831*uk_85 + 861273*uk_86 + 289737*uk_87 + 2274300*uk_88 + 873810*uk_89 + 2242306609*uk_9 + 143640*uk_90 + 2382030*uk_91 + 2597490*uk_92 + 873810*uk_93 + 335727*uk_94 + 55188*uk_95 + 915201*uk_96 + 997983*uk_97 + 335727*uk_98 + 9072*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 101304*uk_100 + 109368*uk_101 + 95760*uk_102 + 2545263*uk_103 + 2747871*uk_104 + 2405970*uk_105 + 2966607*uk_106 + 2597490*uk_107 + 2274300*uk_108 + 1643032*uk_109 + 5587654*uk_11 + 2645560*uk_110 + 111392*uk_111 + 2798724*uk_112 + 3021508*uk_113 + 2645560*uk_114 + 4259800*uk_115 + 179360*uk_116 + 4506420*uk_117 + 4865140*uk_118 + 4259800*uk_119 + 8997070*uk_12 + 7552*uk_120 + 189744*uk_121 + 204848*uk_122 + 179360*uk_123 + 4767318*uk_124 + 5146806*uk_125 + 4506420*uk_126 + 5556502*uk_127 + 4865140*uk_128 + 4259800*uk_129 + 378824*uk_13 + 6859000*uk_130 + 288800*uk_131 + 7256100*uk_132 + 7833700*uk_133 + 6859000*uk_134 + 12160*uk_135 + 305520*uk_136 + 329840*uk_137 + 288800*uk_138 + 7676190*uk_139 + 9517953*uk_14 + 8287230*uk_140 + 7256100*uk_141 + 8946910*uk_142 + 7833700*uk_143 + 6859000*uk_144 + 512*uk_145 + 12864*uk_146 + 13888*uk_147 + 12160*uk_148 + 323208*uk_149 + 10275601*uk_15 + 348936*uk_150 + 305520*uk_151 + 376712*uk_152 + 329840*uk_153 + 288800*uk_154 + 8120601*uk_155 + 8767017*uk_156 + 7676190*uk_157 + 9464889*uk_158 + 8287230*uk_159 + 8997070*uk_16 + 7256100*uk_160 + 10218313*uk_161 + 8946910*uk_162 + 7833700*uk_163 + 6859000*uk_164 + 3969*uk_17 + 7434*uk_18 + 11970*uk_19 + 63*uk_2 + 504*uk_20 + 12663*uk_21 + 13671*uk_22 + 11970*uk_23 + 13924*uk_24 + 22420*uk_25 + 944*uk_26 + 23718*uk_27 + 25606*uk_28 + 22420*uk_29 + 118*uk_3 + 36100*uk_30 + 1520*uk_31 + 38190*uk_32 + 41230*uk_33 + 36100*uk_34 + 64*uk_35 + 1608*uk_36 + 1736*uk_37 + 1520*uk_38 + 40401*uk_39 + 190*uk_4 + 43617*uk_40 + 38190*uk_41 + 47089*uk_42 + 41230*uk_43 + 36100*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 264592179862*uk_47 + 426038255710*uk_48 + 17938452872*uk_49 + 8*uk_5 + 450703628409*uk_50 + 486580534153*uk_51 + 426038255710*uk_52 + 187944057*uk_53 + 352022202*uk_54 + 566815410*uk_55 + 23865912*uk_56 + 599631039*uk_57 + 647362863*uk_58 + 566815410*uk_59 + 201*uk_6 + 659343172*uk_60 + 1061654260*uk_61 + 44701232*uk_62 + 1123118454*uk_63 + 1212520918*uk_64 + 1061654260*uk_65 + 1709443300*uk_66 + 71976560*uk_67 + 1808411070*uk_68 + 1952364190*uk_69 + 217*uk_7 + 1709443300*uk_70 + 3030592*uk_71 + 76143624*uk_72 + 82204808*uk_73 + 71976560*uk_74 + 1913108553*uk_75 + 2065395801*uk_76 + 1808411070*uk_77 + 2229805417*uk_78 + 1952364190*uk_79 + 190*uk_8 + 1709443300*uk_80 + 250047*uk_81 + 468342*uk_82 + 754110*uk_83 + 31752*uk_84 + 797769*uk_85 + 861273*uk_86 + 754110*uk_87 + 877212*uk_88 + 1412460*uk_89 + 2242306609*uk_9 + 59472*uk_90 + 1494234*uk_91 + 1613178*uk_92 + 1412460*uk_93 + 2274300*uk_94 + 95760*uk_95 + 2405970*uk_96 + 2597490*uk_97 + 2274300*uk_98 + 4032*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 102312*uk_100 + 109368*uk_101 + 59472*uk_102 + 2596167*uk_103 + 2775213*uk_104 + 1509102*uk_105 + 2966607*uk_106 + 1613178*uk_107 + 877212*uk_108 + 157464*uk_109 + 2557062*uk_11 + 344088*uk_110 + 23328*uk_111 + 591948*uk_112 + 632772*uk_113 + 344088*uk_114 + 751896*uk_115 + 50976*uk_116 + 1293516*uk_117 + 1382724*uk_118 + 751896*uk_119 + 5587654*uk_12 + 3456*uk_120 + 87696*uk_121 + 93744*uk_122 + 50976*uk_123 + 2225286*uk_124 + 2378754*uk_125 + 1293516*uk_126 + 2542806*uk_127 + 1382724*uk_128 + 751896*uk_129 + 378824*uk_13 + 1643032*uk_130 + 111392*uk_131 + 2826572*uk_132 + 3021508*uk_133 + 1643032*uk_134 + 7552*uk_135 + 191632*uk_136 + 204848*uk_137 + 111392*uk_138 + 4862662*uk_139 + 9612659*uk_14 + 5198018*uk_140 + 2826572*uk_141 + 5556502*uk_142 + 3021508*uk_143 + 1643032*uk_144 + 512*uk_145 + 12992*uk_146 + 13888*uk_147 + 7552*uk_148 + 329672*uk_149 + 10275601*uk_15 + 352408*uk_150 + 191632*uk_151 + 376712*uk_152 + 204848*uk_153 + 111392*uk_154 + 8365427*uk_155 + 8942353*uk_156 + 4862662*uk_157 + 9559067*uk_158 + 5198018*uk_159 + 5587654*uk_16 + 2826572*uk_160 + 10218313*uk_161 + 5556502*uk_162 + 3021508*uk_163 + 1643032*uk_164 + 3969*uk_17 + 3402*uk_18 + 7434*uk_19 + 63*uk_2 + 504*uk_20 + 12789*uk_21 + 13671*uk_22 + 7434*uk_23 + 2916*uk_24 + 6372*uk_25 + 432*uk_26 + 10962*uk_27 + 11718*uk_28 + 6372*uk_29 + 54*uk_3 + 13924*uk_30 + 944*uk_31 + 23954*uk_32 + 25606*uk_33 + 13924*uk_34 + 64*uk_35 + 1624*uk_36 + 1736*uk_37 + 944*uk_38 + 41209*uk_39 + 118*uk_4 + 44051*uk_40 + 23954*uk_41 + 47089*uk_42 + 25606*uk_43 + 13924*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 121084556886*uk_47 + 264592179862*uk_48 + 17938452872*uk_49 + 8*uk_5 + 455188241627*uk_50 + 486580534153*uk_51 + 264592179862*uk_52 + 187944057*uk_53 + 161094906*uk_54 + 352022202*uk_55 + 23865912*uk_56 + 605597517*uk_57 + 647362863*uk_58 + 352022202*uk_59 + 203*uk_6 + 138081348*uk_60 + 301733316*uk_61 + 20456496*uk_62 + 519083586*uk_63 + 554882454*uk_64 + 301733316*uk_65 + 659343172*uk_66 + 44701232*uk_67 + 1134293762*uk_68 + 1212520918*uk_69 + 217*uk_7 + 659343172*uk_70 + 3030592*uk_71 + 76901272*uk_72 + 82204808*uk_73 + 44701232*uk_74 + 1951369777*uk_75 + 2085947003*uk_76 + 1134293762*uk_77 + 2229805417*uk_78 + 1212520918*uk_79 + 118*uk_8 + 659343172*uk_80 + 250047*uk_81 + 214326*uk_82 + 468342*uk_83 + 31752*uk_84 + 805707*uk_85 + 861273*uk_86 + 468342*uk_87 + 183708*uk_88 + 401436*uk_89 + 2242306609*uk_9 + 27216*uk_90 + 690606*uk_91 + 738234*uk_92 + 401436*uk_93 + 877212*uk_94 + 59472*uk_95 + 1509102*uk_96 + 1613178*uk_97 + 877212*uk_98 + 4032*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 154980*uk_100 + 164052*uk_101 + 40824*uk_102 + 2647575*uk_103 + 2802555*uk_104 + 697410*uk_105 + 2966607*uk_106 + 738234*uk_107 + 183708*uk_108 + 8365427*uk_109 + 9612659*uk_11 + 2225286*uk_110 + 494508*uk_111 + 8447845*uk_112 + 8942353*uk_113 + 2225286*uk_114 + 591948*uk_115 + 131544*uk_116 + 2247210*uk_117 + 2378754*uk_118 + 591948*uk_119 + 2557062*uk_12 + 29232*uk_120 + 499380*uk_121 + 528612*uk_122 + 131544*uk_123 + 8531075*uk_124 + 9030455*uk_125 + 2247210*uk_126 + 9559067*uk_127 + 2378754*uk_128 + 591948*uk_129 + 568236*uk_13 + 157464*uk_130 + 34992*uk_131 + 597780*uk_132 + 632772*uk_133 + 157464*uk_134 + 7776*uk_135 + 132840*uk_136 + 140616*uk_137 + 34992*uk_138 + 2269350*uk_139 + 9707365*uk_14 + 2402190*uk_140 + 597780*uk_141 + 2542806*uk_142 + 632772*uk_143 + 157464*uk_144 + 1728*uk_145 + 29520*uk_146 + 31248*uk_147 + 7776*uk_148 + 504300*uk_149 + 10275601*uk_15 + 533820*uk_150 + 132840*uk_151 + 565068*uk_152 + 140616*uk_153 + 34992*uk_154 + 8615125*uk_155 + 9119425*uk_156 + 2269350*uk_157 + 9653245*uk_158 + 2402190*uk_159 + 2557062*uk_16 + 597780*uk_160 + 10218313*uk_161 + 2542806*uk_162 + 632772*uk_163 + 157464*uk_164 + 3969*uk_17 + 12789*uk_18 + 3402*uk_19 + 63*uk_2 + 756*uk_20 + 12915*uk_21 + 13671*uk_22 + 3402*uk_23 + 41209*uk_24 + 10962*uk_25 + 2436*uk_26 + 41615*uk_27 + 44051*uk_28 + 10962*uk_29 + 203*uk_3 + 2916*uk_30 + 648*uk_31 + 11070*uk_32 + 11718*uk_33 + 2916*uk_34 + 144*uk_35 + 2460*uk_36 + 2604*uk_37 + 648*uk_38 + 42025*uk_39 + 54*uk_4 + 44485*uk_40 + 11070*uk_41 + 47089*uk_42 + 11718*uk_43 + 2916*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 455188241627*uk_47 + 121084556886*uk_48 + 26907679308*uk_49 + 12*uk_5 + 459672854845*uk_50 + 486580534153*uk_51 + 121084556886*uk_52 + 187944057*uk_53 + 605597517*uk_54 + 161094906*uk_55 + 35798868*uk_56 + 611563995*uk_57 + 647362863*uk_58 + 161094906*uk_59 + 205*uk_6 + 1951369777*uk_60 + 519083586*uk_61 + 115351908*uk_62 + 1970595095*uk_63 + 2085947003*uk_64 + 519083586*uk_65 + 138081348*uk_66 + 30684744*uk_67 + 524197710*uk_68 + 554882454*uk_69 + 217*uk_7 + 138081348*uk_70 + 6818832*uk_71 + 116488380*uk_72 + 123307212*uk_73 + 30684744*uk_74 + 1990009825*uk_75 + 2106498205*uk_76 + 524197710*uk_77 + 2229805417*uk_78 + 554882454*uk_79 + 54*uk_8 + 138081348*uk_80 + 250047*uk_81 + 805707*uk_82 + 214326*uk_83 + 47628*uk_84 + 813645*uk_85 + 861273*uk_86 + 214326*uk_87 + 2596167*uk_88 + 690606*uk_89 + 2242306609*uk_9 + 153468*uk_90 + 2621745*uk_91 + 2775213*uk_92 + 690606*uk_93 + 183708*uk_94 + 40824*uk_95 + 697410*uk_96 + 738234*uk_97 + 183708*uk_98 + 9072*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 104328*uk_100 + 109368*uk_101 + 102312*uk_102 + 2699487*uk_103 + 2829897*uk_104 + 2647323*uk_105 + 2966607*uk_106 + 2775213*uk_107 + 2596167*uk_108 + 3869893*uk_109 + 7434421*uk_11 + 5003747*uk_110 + 197192*uk_111 + 5102343*uk_112 + 5348833*uk_113 + 5003747*uk_114 + 6469813*uk_115 + 254968*uk_116 + 6597297*uk_117 + 6916007*uk_118 + 6469813*uk_119 + 9612659*uk_12 + 10048*uk_120 + 259992*uk_121 + 272552*uk_122 + 254968*uk_123 + 6727293*uk_124 + 7052283*uk_125 + 6597297*uk_126 + 7392973*uk_127 + 6916007*uk_128 + 6469813*uk_129 + 378824*uk_13 + 8365427*uk_130 + 329672*uk_131 + 8530263*uk_132 + 8942353*uk_133 + 8365427*uk_134 + 12992*uk_135 + 336168*uk_136 + 352408*uk_137 + 329672*uk_138 + 8698347*uk_139 + 9802071*uk_14 + 9118557*uk_140 + 8530263*uk_141 + 9559067*uk_142 + 8942353*uk_143 + 8365427*uk_144 + 512*uk_145 + 13248*uk_146 + 13888*uk_147 + 12992*uk_148 + 342792*uk_149 + 10275601*uk_15 + 359352*uk_150 + 336168*uk_151 + 376712*uk_152 + 352408*uk_153 + 329672*uk_154 + 8869743*uk_155 + 9298233*uk_156 + 8698347*uk_157 + 9747423*uk_158 + 9118557*uk_159 + 9612659*uk_16 + 8530263*uk_160 + 10218313*uk_161 + 9559067*uk_162 + 8942353*uk_163 + 8365427*uk_164 + 3969*uk_17 + 9891*uk_18 + 12789*uk_19 + 63*uk_2 + 504*uk_20 + 13041*uk_21 + 13671*uk_22 + 12789*uk_23 + 24649*uk_24 + 31871*uk_25 + 1256*uk_26 + 32499*uk_27 + 34069*uk_28 + 31871*uk_29 + 157*uk_3 + 41209*uk_30 + 1624*uk_31 + 42021*uk_32 + 44051*uk_33 + 41209*uk_34 + 64*uk_35 + 1656*uk_36 + 1736*uk_37 + 1624*uk_38 + 42849*uk_39 + 203*uk_4 + 44919*uk_40 + 42021*uk_41 + 47089*uk_42 + 44051*uk_43 + 41209*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 352042137613*uk_47 + 455188241627*uk_48 + 17938452872*uk_49 + 8*uk_5 + 464157468063*uk_50 + 486580534153*uk_51 + 455188241627*uk_52 + 187944057*uk_53 + 468368523*uk_54 + 605597517*uk_55 + 23865912*uk_56 + 617530473*uk_57 + 647362863*uk_58 + 605597517*uk_59 + 207*uk_6 + 1167204097*uk_60 + 1509187463*uk_61 + 59475368*uk_62 + 1538925147*uk_63 + 1613269357*uk_64 + 1509187463*uk_65 + 1951369777*uk_66 + 76901272*uk_67 + 1989820413*uk_68 + 2085947003*uk_69 + 217*uk_7 + 1951369777*uk_70 + 3030592*uk_71 + 78416568*uk_72 + 82204808*uk_73 + 76901272*uk_74 + 2029028697*uk_75 + 2127049407*uk_76 + 1989820413*uk_77 + 2229805417*uk_78 + 2085947003*uk_79 + 203*uk_8 + 1951369777*uk_80 + 250047*uk_81 + 623133*uk_82 + 805707*uk_83 + 31752*uk_84 + 821583*uk_85 + 861273*uk_86 + 805707*uk_87 + 1552887*uk_88 + 2007873*uk_89 + 2242306609*uk_9 + 79128*uk_90 + 2047437*uk_91 + 2146347*uk_92 + 2007873*uk_93 + 2596167*uk_94 + 102312*uk_95 + 2647323*uk_96 + 2775213*uk_97 + 2596167*uk_98 + 4032*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 105336*uk_100 + 109368*uk_101 + 79128*uk_102 + 2751903*uk_103 + 2857239*uk_104 + 2067219*uk_105 + 2966607*uk_106 + 2146347*uk_107 + 1552887*uk_108 + 1685159*uk_109 + 5635007*uk_11 + 2223277*uk_110 + 113288*uk_111 + 2959649*uk_112 + 3072937*uk_113 + 2223277*uk_114 + 2933231*uk_115 + 149464*uk_116 + 3904747*uk_117 + 4054211*uk_118 + 2933231*uk_119 + 7434421*uk_12 + 7616*uk_120 + 198968*uk_121 + 206584*uk_122 + 149464*uk_123 + 5198039*uk_124 + 5397007*uk_125 + 3904747*uk_126 + 5603591*uk_127 + 4054211*uk_128 + 2933231*uk_129 + 378824*uk_13 + 3869893*uk_130 + 197192*uk_131 + 5151641*uk_132 + 5348833*uk_133 + 3869893*uk_134 + 10048*uk_135 + 262504*uk_136 + 272552*uk_137 + 197192*uk_138 + 6857917*uk_139 + 9896777*uk_14 + 7120421*uk_140 + 5151641*uk_141 + 7392973*uk_142 + 5348833*uk_143 + 3869893*uk_144 + 512*uk_145 + 13376*uk_146 + 13888*uk_147 + 10048*uk_148 + 349448*uk_149 + 10275601*uk_15 + 362824*uk_150 + 262504*uk_151 + 376712*uk_152 + 272552*uk_153 + 197192*uk_154 + 9129329*uk_155 + 9478777*uk_156 + 6857917*uk_157 + 9841601*uk_158 + 7120421*uk_159 + 7434421*uk_16 + 5151641*uk_160 + 10218313*uk_161 + 7392973*uk_162 + 5348833*uk_163 + 3869893*uk_164 + 3969*uk_17 + 7497*uk_18 + 9891*uk_19 + 63*uk_2 + 504*uk_20 + 13167*uk_21 + 13671*uk_22 + 9891*uk_23 + 14161*uk_24 + 18683*uk_25 + 952*uk_26 + 24871*uk_27 + 25823*uk_28 + 18683*uk_29 + 119*uk_3 + 24649*uk_30 + 1256*uk_31 + 32813*uk_32 + 34069*uk_33 + 24649*uk_34 + 64*uk_35 + 1672*uk_36 + 1736*uk_37 + 1256*uk_38 + 43681*uk_39 + 157*uk_4 + 45353*uk_40 + 32813*uk_41 + 47089*uk_42 + 34069*uk_43 + 24649*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 266834486471*uk_47 + 352042137613*uk_48 + 17938452872*uk_49 + 8*uk_5 + 468642081281*uk_50 + 486580534153*uk_51 + 352042137613*uk_52 + 187944057*uk_53 + 355005441*uk_54 + 468368523*uk_55 + 23865912*uk_56 + 623496951*uk_57 + 647362863*uk_58 + 468368523*uk_59 + 209*uk_6 + 670565833*uk_60 + 884696099*uk_61 + 45080056*uk_62 + 1177716463*uk_63 + 1222796519*uk_64 + 884696099*uk_65 + 1167204097*uk_66 + 59475368*uk_67 + 1553793989*uk_68 + 1613269357*uk_69 + 217*uk_7 + 1167204097*uk_70 + 3030592*uk_71 + 79174216*uk_72 + 82204808*uk_73 + 59475368*uk_74 + 2068426393*uk_75 + 2147600609*uk_76 + 1553793989*uk_77 + 2229805417*uk_78 + 1613269357*uk_79 + 157*uk_8 + 1167204097*uk_80 + 250047*uk_81 + 472311*uk_82 + 623133*uk_83 + 31752*uk_84 + 829521*uk_85 + 861273*uk_86 + 623133*uk_87 + 892143*uk_88 + 1177029*uk_89 + 2242306609*uk_9 + 59976*uk_90 + 1566873*uk_91 + 1626849*uk_92 + 1177029*uk_93 + 1552887*uk_94 + 79128*uk_95 + 2067219*uk_96 + 2146347*uk_97 + 1552887*uk_98 + 4032*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 106344*uk_100 + 109368*uk_101 + 59976*uk_102 + 2804823*uk_103 + 2884581*uk_104 + 1581867*uk_105 + 2966607*uk_106 + 1626849*uk_107 + 892143*uk_108 + 704969*uk_109 + 4214417*uk_11 + 942599*uk_110 + 63368*uk_111 + 1671331*uk_112 + 1718857*uk_113 + 942599*uk_114 + 1260329*uk_115 + 84728*uk_116 + 2234701*uk_117 + 2298247*uk_118 + 1260329*uk_119 + 5635007*uk_12 + 5696*uk_120 + 150232*uk_121 + 154504*uk_122 + 84728*uk_123 + 3962369*uk_124 + 4075043*uk_125 + 2234701*uk_126 + 4190921*uk_127 + 2298247*uk_128 + 1260329*uk_129 + 378824*uk_13 + 1685159*uk_130 + 113288*uk_131 + 2987971*uk_132 + 3072937*uk_133 + 1685159*uk_134 + 7616*uk_135 + 200872*uk_136 + 206584*uk_137 + 113288*uk_138 + 5297999*uk_139 + 9991483*uk_14 + 5448653*uk_140 + 2987971*uk_141 + 5603591*uk_142 + 3072937*uk_143 + 1685159*uk_144 + 512*uk_145 + 13504*uk_146 + 13888*uk_147 + 7616*uk_148 + 356168*uk_149 + 10275601*uk_15 + 366296*uk_150 + 200872*uk_151 + 376712*uk_152 + 206584*uk_153 + 113288*uk_154 + 9393931*uk_155 + 9661057*uk_156 + 5297999*uk_157 + 9935779*uk_158 + 5448653*uk_159 + 5635007*uk_16 + 2987971*uk_160 + 10218313*uk_161 + 5603591*uk_162 + 3072937*uk_163 + 1685159*uk_164 + 3969*uk_17 + 5607*uk_18 + 7497*uk_19 + 63*uk_2 + 504*uk_20 + 13293*uk_21 + 13671*uk_22 + 7497*uk_23 + 7921*uk_24 + 10591*uk_25 + 712*uk_26 + 18779*uk_27 + 19313*uk_28 + 10591*uk_29 + 89*uk_3 + 14161*uk_30 + 952*uk_31 + 25109*uk_32 + 25823*uk_33 + 14161*uk_34 + 64*uk_35 + 1688*uk_36 + 1736*uk_37 + 952*uk_38 + 44521*uk_39 + 119*uk_4 + 45787*uk_40 + 25109*uk_41 + 47089*uk_42 + 25823*uk_43 + 14161*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 199565288201*uk_47 + 266834486471*uk_48 + 17938452872*uk_49 + 8*uk_5 + 473126694499*uk_50 + 486580534153*uk_51 + 266834486471*uk_52 + 187944057*uk_53 + 265508271*uk_54 + 355005441*uk_55 + 23865912*uk_56 + 629463429*uk_57 + 647362863*uk_58 + 355005441*uk_59 + 211*uk_6 + 375083113*uk_60 + 501515623*uk_61 + 33715336*uk_62 + 889241987*uk_63 + 914528489*uk_64 + 501515623*uk_65 + 670565833*uk_66 + 45080056*uk_67 + 1188986477*uk_68 + 1222796519*uk_69 + 217*uk_7 + 670565833*uk_70 + 3030592*uk_71 + 79931864*uk_72 + 82204808*uk_73 + 45080056*uk_74 + 2108202913*uk_75 + 2168151811*uk_76 + 1188986477*uk_77 + 2229805417*uk_78 + 1222796519*uk_79 + 119*uk_8 + 670565833*uk_80 + 250047*uk_81 + 353241*uk_82 + 472311*uk_83 + 31752*uk_84 + 837459*uk_85 + 861273*uk_86 + 472311*uk_87 + 499023*uk_88 + 667233*uk_89 + 2242306609*uk_9 + 44856*uk_90 + 1183077*uk_91 + 1216719*uk_92 + 667233*uk_93 + 892143*uk_94 + 59976*uk_95 + 1581867*uk_96 + 1626849*uk_97 + 892143*uk_98 + 4032*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 107352*uk_100 + 109368*uk_101 + 44856*uk_102 + 2858247*uk_103 + 2911923*uk_104 + 1194291*uk_105 + 2966607*uk_106 + 1216719*uk_107 + 499023*uk_108 + 300763*uk_109 + 3172651*uk_11 + 399521*uk_110 + 35912*uk_111 + 956157*uk_112 + 974113*uk_113 + 399521*uk_114 + 530707*uk_115 + 47704*uk_116 + 1270119*uk_117 + 1293971*uk_118 + 530707*uk_119 + 4214417*uk_12 + 4288*uk_120 + 114168*uk_121 + 116312*uk_122 + 47704*uk_123 + 3039723*uk_124 + 3096807*uk_125 + 1270119*uk_126 + 3154963*uk_127 + 1293971*uk_128 + 530707*uk_129 + 378824*uk_13 + 704969*uk_130 + 63368*uk_131 + 1687173*uk_132 + 1718857*uk_133 + 704969*uk_134 + 5696*uk_135 + 151656*uk_136 + 154504*uk_137 + 63368*uk_138 + 4037841*uk_139 + 10086189*uk_14 + 4113669*uk_140 + 1687173*uk_141 + 4190921*uk_142 + 1718857*uk_143 + 704969*uk_144 + 512*uk_145 + 13632*uk_146 + 13888*uk_147 + 5696*uk_148 + 362952*uk_149 + 10275601*uk_15 + 369768*uk_150 + 151656*uk_151 + 376712*uk_152 + 154504*uk_153 + 63368*uk_154 + 9663597*uk_155 + 9845073*uk_156 + 4037841*uk_157 + 10029957*uk_158 + 4113669*uk_159 + 4214417*uk_16 + 1687173*uk_160 + 10218313*uk_161 + 4190921*uk_162 + 1718857*uk_163 + 704969*uk_164 + 3969*uk_17 + 4221*uk_18 + 5607*uk_19 + 63*uk_2 + 504*uk_20 + 13419*uk_21 + 13671*uk_22 + 5607*uk_23 + 4489*uk_24 + 5963*uk_25 + 536*uk_26 + 14271*uk_27 + 14539*uk_28 + 5963*uk_29 + 67*uk_3 + 7921*uk_30 + 712*uk_31 + 18957*uk_32 + 19313*uk_33 + 7921*uk_34 + 64*uk_35 + 1704*uk_36 + 1736*uk_37 + 712*uk_38 + 45369*uk_39 + 89*uk_4 + 46221*uk_40 + 18957*uk_41 + 47089*uk_42 + 19313*uk_43 + 7921*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 150234542803*uk_47 + 199565288201*uk_48 + 17938452872*uk_49 + 8*uk_5 + 477611307717*uk_50 + 486580534153*uk_51 + 199565288201*uk_52 + 187944057*uk_53 + 199877013*uk_54 + 265508271*uk_55 + 23865912*uk_56 + 635429907*uk_57 + 647362863*uk_58 + 265508271*uk_59 + 213*uk_6 + 212567617*uk_60 + 282365939*uk_61 + 25381208*uk_62 + 675774663*uk_63 + 688465267*uk_64 + 282365939*uk_65 + 375083113*uk_66 + 33715336*uk_67 + 897670821*uk_68 + 914528489*uk_69 + 217*uk_7 + 375083113*uk_70 + 3030592*uk_71 + 80689512*uk_72 + 82204808*uk_73 + 33715336*uk_74 + 2148358257*uk_75 + 2188703013*uk_76 + 897670821*uk_77 + 2229805417*uk_78 + 914528489*uk_79 + 89*uk_8 + 375083113*uk_80 + 250047*uk_81 + 265923*uk_82 + 353241*uk_83 + 31752*uk_84 + 845397*uk_85 + 861273*uk_86 + 353241*uk_87 + 282807*uk_88 + 375669*uk_89 + 2242306609*uk_9 + 33768*uk_90 + 899073*uk_91 + 915957*uk_92 + 375669*uk_93 + 499023*uk_94 + 44856*uk_95 + 1194291*uk_96 + 1216719*uk_97 + 499023*uk_98 + 4032*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 108360*uk_100 + 109368*uk_101 + 33768*uk_102 + 2912175*uk_103 + 2939265*uk_104 + 907515*uk_105 + 2966607*uk_106 + 915957*uk_107 + 282807*uk_108 + 148877*uk_109 + 2509709*uk_11 + 188203*uk_110 + 22472*uk_111 + 603935*uk_112 + 609553*uk_113 + 188203*uk_114 + 237917*uk_115 + 28408*uk_116 + 763465*uk_117 + 770567*uk_118 + 237917*uk_119 + 3172651*uk_12 + 3392*uk_120 + 91160*uk_121 + 92008*uk_122 + 28408*uk_123 + 2449925*uk_124 + 2472715*uk_125 + 763465*uk_126 + 2495717*uk_127 + 770567*uk_128 + 237917*uk_129 + 378824*uk_13 + 300763*uk_130 + 35912*uk_131 + 965135*uk_132 + 974113*uk_133 + 300763*uk_134 + 4288*uk_135 + 115240*uk_136 + 116312*uk_137 + 35912*uk_138 + 3097075*uk_139 + 10180895*uk_14 + 3125885*uk_140 + 965135*uk_141 + 3154963*uk_142 + 974113*uk_143 + 300763*uk_144 + 512*uk_145 + 13760*uk_146 + 13888*uk_147 + 4288*uk_148 + 369800*uk_149 + 10275601*uk_15 + 373240*uk_150 + 115240*uk_151 + 376712*uk_152 + 116312*uk_153 + 35912*uk_154 + 9938375*uk_155 + 10030825*uk_156 + 3097075*uk_157 + 10124135*uk_158 + 3125885*uk_159 + 3172651*uk_16 + 965135*uk_160 + 10218313*uk_161 + 3154963*uk_162 + 974113*uk_163 + 300763*uk_164 + 3969*uk_17 + 3339*uk_18 + 4221*uk_19 + 63*uk_2 + 504*uk_20 + 13545*uk_21 + 13671*uk_22 + 4221*uk_23 + 2809*uk_24 + 3551*uk_25 + 424*uk_26 + 11395*uk_27 + 11501*uk_28 + 3551*uk_29 + 53*uk_3 + 4489*uk_30 + 536*uk_31 + 14405*uk_32 + 14539*uk_33 + 4489*uk_34 + 64*uk_35 + 1720*uk_36 + 1736*uk_37 + 536*uk_38 + 46225*uk_39 + 67*uk_4 + 46655*uk_40 + 14405*uk_41 + 47089*uk_42 + 14539*uk_43 + 4489*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 118842250277*uk_47 + 150234542803*uk_48 + 17938452872*uk_49 + 8*uk_5 + 482095920935*uk_50 + 486580534153*uk_51 + 150234542803*uk_52 + 187944057*uk_53 + 158111667*uk_54 + 199877013*uk_55 + 23865912*uk_56 + 641396385*uk_57 + 647362863*uk_58 + 199877013*uk_59 + 215*uk_6 + 133014577*uk_60 + 168150503*uk_61 + 20077672*uk_62 + 539587435*uk_63 + 544606853*uk_64 + 168150503*uk_65 + 212567617*uk_66 + 25381208*uk_67 + 682119965*uk_68 + 688465267*uk_69 + 217*uk_7 + 212567617*uk_70 + 3030592*uk_71 + 81447160*uk_72 + 82204808*uk_73 + 25381208*uk_74 + 2188892425*uk_75 + 2209254215*uk_76 + 682119965*uk_77 + 2229805417*uk_78 + 688465267*uk_79 + 67*uk_8 + 212567617*uk_80 + 250047*uk_81 + 210357*uk_82 + 265923*uk_83 + 31752*uk_84 + 853335*uk_85 + 861273*uk_86 + 265923*uk_87 + 176967*uk_88 + 223713*uk_89 + 2242306609*uk_9 + 26712*uk_90 + 717885*uk_91 + 724563*uk_92 + 223713*uk_93 + 282807*uk_94 + 33768*uk_95 + 907515*uk_96 + 915957*uk_97 + 282807*uk_98 + 4032*uk_99,
        uk_0 + 47353*uk_1 + 2983239*uk_10 + 109368*uk_100 + 109368*uk_101 + 26712*uk_102 + 2966607*uk_103 + 2966607*uk_104 + 724563*uk_105 + 2966607*uk_106 + 724563*uk_107 + 176967*uk_108 + 103823*uk_109 + 2225591*uk_11 + 117077*uk_110 + 17672*uk_111 + 479353*uk_112 + 479353*uk_113 + 117077*uk_114 + 132023*uk_115 + 19928*uk_116 + 540547*uk_117 + 540547*uk_118 + 132023*uk_119 + 2509709*uk_12 + 3008*uk_120 + 81592*uk_121 + 81592*uk_122 + 19928*uk_123 + 2213183*uk_124 + 2213183*uk_125 + 540547*uk_126 + 2213183*uk_127 + 540547*uk_128 + 132023*uk_129 + 378824*uk_13 + 148877*uk_130 + 22472*uk_131 + 609553*uk_132 + 609553*uk_133 + 148877*uk_134 + 3392*uk_135 + 92008*uk_136 + 92008*uk_137 + 22472*uk_138 + 2495717*uk_139 + 10275601*uk_14 + 2495717*uk_140 + 609553*uk_141 + 2495717*uk_142 + 609553*uk_143 + 148877*uk_144 + 512*uk_145 + 13888*uk_146 + 13888*uk_147 + 3392*uk_148 + 376712*uk_149 + 10275601*uk_15 + 376712*uk_150 + 92008*uk_151 + 376712*uk_152 + 92008*uk_153 + 22472*uk_154 + 10218313*uk_155 + 10218313*uk_156 + 2495717*uk_157 + 10218313*uk_158 + 2495717*uk_159 + 2509709*uk_16 + 609553*uk_160 + 10218313*uk_161 + 2495717*uk_162 + 609553*uk_163 + 148877*uk_164 + 3969*uk_17 + 2961*uk_18 + 3339*uk_19 + 63*uk_2 + 504*uk_20 + 13671*uk_21 + 13671*uk_22 + 3339*uk_23 + 2209*uk_24 + 2491*uk_25 + 376*uk_26 + 10199*uk_27 + 10199*uk_28 + 2491*uk_29 + 47*uk_3 + 2809*uk_30 + 424*uk_31 + 11501*uk_32 + 11501*uk_33 + 2809*uk_34 + 64*uk_35 + 1736*uk_36 + 1736*uk_37 + 424*uk_38 + 47089*uk_39 + 53*uk_4 + 47089*uk_40 + 11501*uk_41 + 47089*uk_42 + 11501*uk_43 + 2809*uk_44 + 106179944855977*uk_45 + 141265316367*uk_46 + 105388410623*uk_47 + 118842250277*uk_48 + 17938452872*uk_49 + 8*uk_5 + 486580534153*uk_50 + 486580534153*uk_51 + 118842250277*uk_52 + 187944057*uk_53 + 140212233*uk_54 + 158111667*uk_55 + 23865912*uk_56 + 647362863*uk_57 + 647362863*uk_58 + 158111667*uk_59 + 217*uk_6 + 104602777*uk_60 + 117956323*uk_61 + 17804728*uk_62 + 482953247*uk_63 + 482953247*uk_64 + 117956323*uk_65 + 133014577*uk_66 + 20077672*uk_67 + 544606853*uk_68 + 544606853*uk_69 + 217*uk_7 + 133014577*uk_70 + 3030592*uk_71 + 82204808*uk_72 + 82204808*uk_73 + 20077672*uk_74 + 2229805417*uk_75 + 2229805417*uk_76 + 544606853*uk_77 + 2229805417*uk_78 + 544606853*uk_79 + 53*uk_8 + 133014577*uk_80 + 250047*uk_81 + 186543*uk_82 + 210357*uk_83 + 31752*uk_84 + 861273*uk_85 + 861273*uk_86 + 210357*uk_87 + 139167*uk_88 + 156933*uk_89 + 2242306609*uk_9 + 23688*uk_90 + 642537*uk_91 + 642537*uk_92 + 156933*uk_93 + 176967*uk_94 + 26712*uk_95 + 724563*uk_96 + 724563*uk_97 + 176967*uk_98 + 4032*uk_99,
    ]

def sol_165x165():
    return {
        uk_0: -QQ(295441,1683)*uk_2 - QQ(175799,1683)*uk_7 + QQ(2401696807,1)*uk_9 - QQ(9606787228,1683)*uk_10 + QQ(9606787228,1683)*uk_15 - QQ(29030443,1683)*uk_17 - QQ(5965893,187)*uk_22 + QQ(262901,99)*uk_42 + QQ(235539209256104,1)*uk_45 - QQ(232597130667529,1683)*uk_46 + QQ(1364372733998209,1683)*uk_51 - QQ(1133600892904,1683)*uk_53 - QQ(172922170104,187)*uk_58 + QQ(249776467928,99)*uk_78 - QQ(2401889209,1683)*uk_81 - QQ(636292759,187)*uk_86 - QQ(1034157281,187)*uk_106 + QQ(10558824289,1683)*uk_161,
        uk_1: QQ(4,1683)*uk_2 - QQ(4,1683)*uk_7 - QQ(98072,1)*uk_9 + QQ(96847,1683)*uk_10 - QQ(568087,1683)*uk_15 + QQ(472,1683)*uk_17 + QQ(72,187)*uk_22 - QQ(104,99)*uk_42 - QQ(7216420377,1)*uk_45 - QQ(108808244,1683)*uk_46 - QQ(46106641036,1683)*uk_51 + QQ(17259541,1683)*uk_53 + QQ(1095291,187)*uk_58 - QQ(9936587,99)*uk_78 + QQ(41836,1683)*uk_81 + QQ(10036,187)*uk_86 + QQ(10124,187)*uk_106 - QQ(8,1)*uk_149 - QQ(586156,1683)*uk_161,
        uk_3: -QQ(295441,1683)*uk_18 - QQ(175799,1683)*uk_28 + QQ(2401696807,1)*uk_47 - QQ(9606787228,1683)*uk_54 + QQ(9606787228,1683)*uk_64 - QQ(29030443,1683)*uk_82 - QQ(5965893,187)*uk_92 + QQ(262901,99)*uk_127 + QQ(8,1)*uk_149,
        uk_4: -QQ(295441,1683)*uk_19 + QQ(1602583,3366)*uk_29 - QQ(175799,1683)*uk_33 - QQ(45670,99)*uk_34 - QQ(76006,187)*uk_38 + QQ(295441,1683)*uk_41 - QQ(45670,99)*uk_44 + QQ(2401696807,1)*uk_48 - QQ(9606787228,1683)*uk_55 + QQ(74452601017,3366)*uk_65 + QQ(9606787228,1683)*uk_69 - QQ(2401696807,99)*uk_70 - QQ(4803393614,187)*uk_74 + QQ(9606787228,1683)*uk_77 - QQ(2401696807,99)*uk_80 - QQ(29030443,1683)*uk_83 + QQ(11596905,374)*uk_93 - QQ(5965893,187)*uk_97 - QQ(769658,33)*uk_98 - QQ(17335370,1683)*uk_102 + QQ(29030443,1683)*uk_105 - QQ(769658,33)*uk_108 + QQ(77314807,3366)*uk_114 + QQ(750229,198)*uk_119 + QQ(72457964,1683)*uk_123 + QQ(11596905,374)*uk_126 + QQ(31304645,306)*uk_128 + QQ(750229,198)*uk_129 - QQ(3191393,99)*uk_134 - QQ(647642,9)*uk_138 - QQ(769658,33)*uk_141 + QQ(262901,99)*uk_142 - QQ(10478626,99)*uk_143 - QQ(3191393,99)*uk_144 - QQ(20480616,187)*uk_148 - QQ(17335370,1683)*uk_151 - QQ(174199750,1683)*uk_153 - QQ(647642,9)*uk_154 + QQ(29030443,1683)*uk_157 + QQ(5965893,187)*uk_159 - QQ(769658,33)*uk_160 - QQ(10478626,99)*uk_163 - QQ(3191393,99)*uk_164,
        uk_5: -QQ(295441,1683)*uk_20 - QQ(175799,1683)*uk_37 + QQ(2401696807,1)*uk_49 - QQ(9606787228,1683)*uk_56 + QQ(9606787228,1683)*uk_73 - QQ(29030443,1683)*uk_84 - QQ(5965893,187)*uk_101 + QQ(262901,99)*uk_152,
        uk_6: -QQ(295441,1683)*uk_21 - QQ(175799,1683)*uk_40 + QQ(2401696807,1)*uk_50 - QQ(9606787228,1683)*uk_57 + QQ(9606787228,1683)*uk_76 - QQ(29030443,1683)*uk_85 - QQ(5965893,187)*uk_104 + QQ(262901,99)*uk_158,
        uk_8: -QQ(295441,1683)*uk_23 - QQ(1602583,3366)*uk_29 + QQ(45670,99)*uk_34 + QQ(76006,187)*uk_38 - QQ(295441,1683)*uk_41 - QQ(175799,1683)*uk_43 + QQ(45670,99)*uk_44 + QQ(2401696807,1)*uk_52 - QQ(9606787228,1683)*uk_59 - QQ(74452601017,3366)*uk_65 + QQ(2401696807,99)*uk_70 + QQ(4803393614,187)*uk_74 - QQ(9606787228,1683)*uk_77 + QQ(9606787228,1683)*uk_79 + QQ(2401696807,99)*uk_80 - QQ(29030443,1683)*uk_87 - QQ(11596905,374)*uk_93 + QQ(769658,33)*uk_98 + QQ(17335370,1683)*uk_102 - QQ(29030443,1683)*uk_105 - QQ(5965893,187)*uk_107 + QQ(769658,33)*uk_108 - QQ(77314807,3366)*uk_114 - QQ(750229,198)*uk_119 - QQ(72457964,1683)*uk_123 - QQ(11596905,374)*uk_126 - QQ(31304645,306)*uk_128 - QQ(750229,198)*uk_129 + QQ(3191393,99)*uk_134 + QQ(647642,9)*uk_138 + QQ(769658,33)*uk_141 + QQ(10478626,99)*uk_143 + QQ(3191393,99)*uk_144 + QQ(20480616,187)*uk_148 + QQ(17335370,1683)*uk_151 + QQ(174199750,1683)*uk_153 + QQ(647642,9)*uk_154 - QQ(29030443,1683)*uk_157 - QQ(5965893,187)*uk_159 + QQ(769658,33)*uk_160 + QQ(262901,99)*uk_162 + QQ(10478626,99)*uk_163 + QQ(3191393,99)*uk_164,
        uk_11: QQ(4,1683)*uk_18 - QQ(4,1683)*uk_28 - QQ(98072,1)*uk_47 + QQ(96847,1683)*uk_54 - QQ(568087,1683)*uk_64 + QQ(472,1683)*uk_82 + QQ(72,187)*uk_92 - QQ(104,99)*uk_127,
        uk_12: QQ(4,1683)*uk_19 - QQ(31,3366)*uk_29 - QQ(4,1683)*uk_33 + QQ(1,99)*uk_34 + QQ(2,187)*uk_38 - QQ(4,1683)*uk_41 + QQ(1,99)*uk_44 - QQ(98072,1)*uk_48 + QQ(96847,1683)*uk_55 - QQ(1437649,3366)*uk_65 - QQ(568087,1683)*uk_69 + QQ(52402,99)*uk_70 + QQ(120138,187)*uk_74 - QQ(96847,1683)*uk_77 + QQ(52402,99)*uk_80 + QQ(472,1683)*uk_83 - QQ(225,374)*uk_93 + QQ(72,187)*uk_97 + QQ(17,33)*uk_98 + QQ(590,1683)*uk_102 - QQ(472,1683)*uk_105 + QQ(17,33)*uk_108 - QQ(1519,3366)*uk_114 - QQ(13,198)*uk_119 - QQ(1388,1683)*uk_123 - QQ(225,374)*uk_126 - QQ(605,306)*uk_128 - QQ(13,198)*uk_129 + QQ(68,99)*uk_134 + QQ(14,9)*uk_138 + QQ(17,33)*uk_141 - QQ(104,99)*uk_142 + QQ(229,99)*uk_143 + QQ(68,99)*uk_144 + QQ(472,187)*uk_148 + QQ(590,1683)*uk_151 + QQ(4450,1683)*uk_153 + QQ(14,9)*uk_154 - QQ(472,1683)*uk_157 - QQ(72,187)*uk_159 + QQ(17,33)*uk_160 + QQ(229,99)*uk_163 + QQ(68,99)*uk_164,
        uk_13: QQ(4,1683)*uk_20 - QQ(4,1683)*uk_37 - QQ(98072,1)*uk_49 + QQ(96847,1683)*uk_56 - QQ(568087,1683)*uk_73 + QQ(472,1683)*uk_84 + QQ(72,187)*uk_101 - QQ(104,99)*uk_152,
        uk_14: QQ(4,1683)*uk_21 - QQ(4,1683)*uk_40 - QQ(98072,1)*uk_50 + QQ(96847,1683)*uk_57 - QQ(568087,1683)*uk_76 + QQ(472,1683)*uk_85 + QQ(72,187)*uk_104 - QQ(104,99)*uk_158,
        uk_16: QQ(4,1683)*uk_23 + QQ(31,3366)*uk_29 - QQ(1,99)*uk_34 - QQ(2,187)*uk_38 + QQ(4,1683)*uk_41 - QQ(4,1683)*uk_43 - QQ(1,99)*uk_44 - QQ(98072,1)*uk_52 + QQ(96847,1683)*uk_59 + QQ(1437649,3366)*uk_65 - QQ(52402,99)*uk_70 - QQ(120138,187)*uk_74 + QQ(96847,1683)*uk_77 - QQ(568087,1683)*uk_79 - QQ(52402,99)*uk_80 + QQ(472,1683)*uk_87 + QQ(225,374)*uk_93 - QQ(17,33)*uk_98 - QQ(590,1683)*uk_102 + QQ(472,1683)*uk_105 + QQ(72,187)*uk_107 - QQ(17,33)*uk_108 + QQ(1519,3366)*uk_114 + QQ(13,198)*uk_119 + QQ(1388,1683)*uk_123 + QQ(225,374)*uk_126 + QQ(605,306)*uk_128 + QQ(13,198)*uk_129 - QQ(68,99)*uk_134 - QQ(14,9)*uk_138 - QQ(17,33)*uk_141 - QQ(229,99)*uk_143 - QQ(68,99)*uk_144 - QQ(472,187)*uk_148 - QQ(590,1683)*uk_151 - QQ(4450,1683)*uk_153 - QQ(14,9)*uk_154 + QQ(472,1683)*uk_157 + QQ(72,187)*uk_159 - QQ(17,33)*uk_160 - QQ(104,99)*uk_162 - QQ(229,99)*uk_163 - QQ(68,99)*uk_164,
        uk_24: -QQ(295441,1683)*uk_88 - QQ(175799,1683)*uk_113,
        uk_26: -QQ(295441,1683)*uk_90 - QQ(175799,1683)*uk_122, uk_25: -uk_29 - QQ(295441,1683)*uk_89 - QQ(295441,1683)*uk_93 - QQ(175799,1683)*uk_118 - QQ(175799,1683)*uk_128,
        uk_27: -QQ(295441,1683)*uk_91 - QQ(175799,1683)*uk_125 - QQ(4,1)*uk_149,
        uk_30: -uk_34 - uk_44 - QQ(295441,1683)*uk_94 - QQ(295441,1683)*uk_98 - QQ(295441,1683)*uk_108 - QQ(175799,1683)*uk_133 - QQ(175799,1683)*uk_143 - QQ(175799,1683)*uk_163,
        uk_31: -uk_38 - QQ(295441,1683)*uk_95 - QQ(295441,1683)*uk_102 - QQ(175799,1683)*uk_137 - QQ(175799,1683)*uk_153,
        uk_32: -uk_41 - QQ(295441,1683)*uk_96 - QQ(295441,1683)*uk_105 - QQ(175799,1683)*uk_140 + QQ(4,1)*uk_149 - QQ(175799,1683)*uk_159,
        uk_35: -QQ(295441,1683)*uk_99 - QQ(175799,1683)*uk_147,
        uk_36: -QQ(295441,1683)*uk_100 - QQ(2,1)*uk_149 - QQ(175799,1683)*uk_150,
        uk_39: -QQ(295441,1683)*uk_103 - QQ(175799,1683)*uk_156,
        uk_60: QQ(4,1683)*uk_88 - QQ(4,1683)*uk_113,
        uk_61: -uk_65 + QQ(4,1683)*uk_89 + QQ(4,1683)*uk_93 - QQ(4,1683)*uk_118 - QQ(4,1683)*uk_128,
        uk_62: QQ(4,1683)*uk_90 - QQ(4,1683)*uk_122,
        uk_63: QQ(4,1683)*uk_91 - QQ(4,1683)*uk_125,
        uk_66: -uk_70 - uk_80 + QQ(4,1683)*uk_94 + QQ(4,1683)*uk_98 + QQ(4,1683)*uk_108 - QQ(4,1683)*uk_133 - QQ(4,1683)*uk_143 - QQ(4,1683)*uk_163,
        uk_67: -uk_74 + QQ(4,1683)*uk_95 + QQ(4,1683)*uk_102 - QQ(4,1683)*uk_137 - QQ(4,1683)*uk_153,
        uk_68: -uk_77 + QQ(4,1683)*uk_96 + QQ(4,1683)*uk_105 - QQ(4,1683)*uk_140 - QQ(4,1683)*uk_159,
        uk_71: QQ(4,1683)*uk_99 - QQ(4,1683)*uk_147,
        uk_72: QQ(4,1683)*uk_100 - QQ(4,1683)*uk_150,
        uk_75: QQ(4,1683)*uk_103 - QQ(4,1683)*uk_156,
        uk_109: 0,
        uk_110: -uk_114,
        uk_111: 0,
        uk_112: 0,
        uk_115: -uk_119 - uk_129,
        uk_116: -uk_123,
        uk_117: -uk_126,
        uk_120: 0,
        uk_121: 0,
        uk_124: 0,
        uk_130: -uk_134 - uk_144 - uk_164,
        uk_131: -uk_138 - uk_154,
        uk_132: -uk_141 - uk_160,
        uk_135: -uk_148,
        uk_136: -uk_151,
        uk_139: -uk_157,
        uk_145: 0,
        uk_146: 0,
        uk_155: 0,
    }

def time_eqs_165x165():
    if len(eqs_165x165()) != 165:
        raise ValueError("length should be 165")

def time_solve_lin_sys_165x165():
    eqs = eqs_165x165()
    sol = solve_lin_sys(eqs, R_165)
    if sol != sol_165x165():
        raise ValueError("Value should be equal")

def time_verify_sol_165x165():
    eqs = eqs_165x165()
    sol = sol_165x165()
    zeros = [ eq.compose(sol) for eq in eqs ]
    if not all(zero == 0 for zero in zeros):
        raise ValueError("All should be 0")

def time_to_expr_eqs_165x165():
    eqs = eqs_165x165()
    assert [ R_165.from_expr(eq.as_expr()) for eq in eqs ] == eqs

# Benchmark R_49: shows how fast are arithmetics in rational function fields.
F_abc, a, b, c = field("a,b,c", ZZ)
R_49, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31, k32, k33, k34, k35, k36, k37, k38, k39, k40, k41, k42, k43, k44, k45, k46, k47, k48, k49 = ring("k1:50", F_abc)

def eqs_189x49():
    return [
        -b*k8/a+c*k8/a,
        -b*k11/a+c*k11/a,
        -b*k10/a+c*k10/a+k2,
        -k3-b*k9/a+c*k9/a,
        -b*k14/a+c*k14/a,
        -b*k15/a+c*k15/a,
        -b*k18/a+c*k18/a-k2,
        -b*k17/a+c*k17/a,
        -b*k16/a+c*k16/a+k4,
        -b*k13/a+c*k13/a-b*k21/a+c*k21/a+b*k5/a-c*k5/a,
        b*k44/a-c*k44/a,
        -b*k45/a+c*k45/a,
        -b*k20/a+c*k20/a,
        -b*k44/a+c*k44/a,
        b*k46/a-c*k46/a,
        b**2*k47/a**2-2*b*c*k47/a**2+c**2*k47/a**2,
        k3,
        -k4,
        -b*k12/a+c*k12/a-a*k6/b+c*k6/b,
        -b*k19/a+c*k19/a+a*k7/c-b*k7/c,
        b*k45/a-c*k45/a,
        -b*k46/a+c*k46/a,
        -k48+c*k48/a+c*k48/b-c**2*k48/(a*b),
        -k49+b*k49/a+b*k49/c-b**2*k49/(a*c),
        a*k1/b-c*k1/b,
        a*k4/b-c*k4/b,
        a*k3/b-c*k3/b+k9,
        -k10+a*k2/b-c*k2/b,
        a*k7/b-c*k7/b,
        -k9,
        k11,
        b*k12/a-c*k12/a+a*k6/b-c*k6/b,
        a*k15/b-c*k15/b,
        k10+a*k18/b-c*k18/b,
        -k11+a*k17/b-c*k17/b,
        a*k16/b-c*k16/b,
        -a*k13/b+c*k13/b+a*k21/b-c*k21/b+a*k5/b-c*k5/b,
        -a*k44/b+c*k44/b,
        a*k45/b-c*k45/b,
        a*k14/c-b*k14/c+a*k20/b-c*k20/b,
        a*k44/b-c*k44/b,
        -a*k46/b+c*k46/b,
        -k47+c*k47/a+c*k47/b-c**2*k47/(a*b),
        a*k19/b-c*k19/b,
        -a*k45/b+c*k45/b,
        a*k46/b-c*k46/b,
        a**2*k48/b**2-2*a*c*k48/b**2+c**2*k48/b**2,
        -k49+a*k49/b+a*k49/c-a**2*k49/(b*c),
        k16,
        -k17,
        -a*k1/c+b*k1/c,
        -k16-a*k4/c+b*k4/c,
        -a*k3/c+b*k3/c,
        k18-a*k2/c+b*k2/c,
        b*k19/a-c*k19/a-a*k7/c+b*k7/c,
        -a*k6/c+b*k6/c,
        -a*k8/c+b*k8/c,
        -a*k11/c+b*k11/c+k17,
        -a*k10/c+b*k10/c-k18,
        -a*k9/c+b*k9/c,
        -a*k14/c+b*k14/c-a*k20/b+c*k20/b,
        -a*k13/c+b*k13/c+a*k21/c-b*k21/c-a*k5/c+b*k5/c,
        a*k44/c-b*k44/c,
        -a*k45/c+b*k45/c,
        -a*k44/c+b*k44/c,
        a*k46/c-b*k46/c,
        -k47+b*k47/a+b*k47/c-b**2*k47/(a*c),
        -a*k12/c+b*k12/c,
        a*k45/c-b*k45/c,
        -a*k46/c+b*k46/c,
        -k48+a*k48/b+a*k48/c-a**2*k48/(b*c),
        a**2*k49/c**2-2*a*b*k49/c**2+b**2*k49/c**2,
        k8,
        k11,
        -k15,
        k10-k18,
        -k17,
        k9,
        -k16,
        -k29,
        k14-k32,
        -k21+k23-k31,
        -k24-k30,
        -k35,
        k44,
        -k45,
        k36,
        k13-k23+k39,
        -k20+k38,
        k25+k37,
        b*k26/a-c*k26/a-k34+k42,
        -2*k44,
        k45,
        k46,
        b*k47/a-c*k47/a,
        k41,
        k44,
        -k46,
        -b*k47/a+c*k47/a,
        k12+k24,
        -k19-k25,
        -a*k27/b+c*k27/b-k33,
        k45,
        -k46,
        -a*k48/b+c*k48/b,
        a*k28/c-b*k28/c+k40,
        -k45,
        k46,
        a*k48/b-c*k48/b,
        a*k49/c-b*k49/c,
        -a*k49/c+b*k49/c,
        -k1,
        -k4,
        -k3,
        k15,
        k18-k2,
        k17,
        k16,
        k22,
        k25-k7,
        k24+k30,
        k21+k23-k31,
        k28,
        -k44,
        k45,
        -k30-k6,
        k20+k32,
        k27+b*k33/a-c*k33/a,
        k44,
        -k46,
        -b*k47/a+c*k47/a,
        -k36,
        k31-k39-k5,
        -k32-k38,
        k19-k37,
        k26-a*k34/b+c*k34/b-k42,
        k44,
        -2*k45,
        k46,
        a*k48/b-c*k48/b,
        a*k35/c-b*k35/c-k41,
        -k44,
        k46,
        b*k47/a-c*k47/a,
        -a*k49/c+b*k49/c,
        -k40,
        k45,
        -k46,
        -a*k48/b+c*k48/b,
        a*k49/c-b*k49/c,
        k1,
        k4,
        k3,
        -k8,
        -k11,
        -k10+k2,
        -k9,
        k37+k7,
        -k14-k38,
        -k22,
        -k25-k37,
        -k24+k6,
        -k13-k23+k39,
        -k28+b*k40/a-c*k40/a,
        k44,
        -k45,
        -k27,
        -k44,
        k46,
        b*k47/a-c*k47/a,
        k29,
        k32+k38,
        k31-k39+k5,
        -k12+k30,
        k35-a*k41/b+c*k41/b,
        -k44,
        k45,
        -k26+k34+a*k42/c-b*k42/c,
        k44,
        k45,
        -2*k46,
        -b*k47/a+c*k47/a,
        -a*k48/b+c*k48/b,
        a*k49/c-b*k49/c,
        k33,
        -k45,
        k46,
        a*k48/b-c*k48/b,
        -a*k49/c+b*k49/c,
    ]

def sol_189x49():
    return {
        k49: 0, k48: 0, k47: 0, k46: 0, k45: 0, k44: 0, k41: 0, k40: 0,
        k38: 0, k37: 0, k36: 0, k35: 0, k33: 0, k32: 0, k30: 0, k29: 0,
        k28: 0, k27: 0, k25: 0, k24: 0, k22: 0, k21: 0, k20: 0, k19: 0,
        k18: 0, k17: 0, k16: 0, k15: 0, k14: 0, k13: 0, k12: 0, k11: 0,
        k10: 0, k9:  0, k8:  0, k7:  0, k6:  0, k5:  0, k4:  0, k3:  0,
        k2:  0, k1:  0,
        k34: b/c*k42,
        k31: k39,
        k26: a/c*k42,
        k23: k39,
    }

def time_eqs_189x49():
    if len(eqs_189x49()) != 189:
        raise ValueError("Length should be equal to 189")

def time_solve_lin_sys_189x49():
    eqs = eqs_189x49()
    sol = solve_lin_sys(eqs, R_49)
    if sol != sol_189x49():
        raise ValueError("Values should be equal")

def time_verify_sol_189x49():
    eqs = eqs_189x49()
    sol = sol_189x49()
    zeros = [ eq.compose(sol) for eq in eqs ]
    assert all(zero == 0 for zero in zeros)

def time_to_expr_eqs_189x49():
    eqs = eqs_189x49()
    assert [ R_49.from_expr(eq.as_expr()) for eq in eqs ] == eqs

# Benchmark R_8: shows how fast polynomial GCDs are computed.

F_a5_5, a_11, a_12, a_13, a_14, a_21, a_22, a_23, a_24, a_31, a_32, a_33, a_34, a_41, a_42, a_43, a_44 = field("a_(1:5)(1:5)", ZZ)
R_8, x0, x1, x2, x3, x4, x5, x6, x7 = ring("x:8", F_a5_5)

def eqs_10x8():
    return [
        (a_33*a_34 + a_33*a_44 + a_43*a_44)*x3 + (a_33*a_34 + a_33*a_44 + a_43*a_44)*x4 + (a_12*a_34 + a_12*a_44 + a_22*a_34 + a_22*a_44)*x5 + (a_12*a_44 + a_22*a_44)*x6 + (a_12*a_33 + a_22*a_33)*x7 - a_12*a_33 - a_12*a_43 - a_22*a_33 - a_22*a_43,
        (a_33 + a_34 + a_43 + a_44)*x3 + (a_33 + a_34 + a_43 + a_44)*x4 + (a_12 + a_22 + a_34 + a_44)*x5 + (a_12 + a_22 + a_44)*x6 + (a_12 + a_22 + a_33)*x7 - a_12 - a_22 - a_33 - a_43,
        x3 + x4 + x5 + x6 + x7 - 1,
        (a_12*a_33*a_34 + a_12*a_33*a_44 + a_12*a_43*a_44 + a_22*a_33*a_34 + a_22*a_33*a_44 + a_22*a_43*a_44)*x0 + (a_22*a_33*a_34 + a_22*a_33*a_44 + a_22*a_43*a_44)*x1 + (a_12*a_33*a_34 + a_12*a_33*a_44 + a_12*a_43*a_44 + a_22*a_33*a_34 + a_22*a_33*a_44 + a_22*a_43*a_44)*x2 + (a_11*a_33*a_34 + a_11*a_33*a_44 + a_11*a_43*a_44 + a_31*a_33*a_34 + a_31*a_33*a_44 + a_31*a_43*a_44)*x3 + (a_11*a_33*a_34 + a_11*a_33*a_44 + a_11*a_43*a_44 + a_21*a_33*a_34 + a_21*a_33*a_44 + a_21*a_43*a_44 + a_31*a_33*a_34 + a_31*a_33*a_44 + a_31*a_43*a_44)*x4 + (a_11*a_12*a_34 + a_11*a_12*a_44 + a_11*a_22*a_34 + a_11*a_22*a_44 + a_12*a_31*a_34 + a_12*a_31*a_44 + a_21*a_22*a_34 + a_21*a_22*a_44 + a_22*a_31*a_34 + a_22*a_31*a_44)*x5 + (a_11*a_12*a_44 + a_11*a_22*a_44 + a_12*a_31*a_44 + a_21*a_22*a_44 + a_22*a_31*a_44)*x6 + (a_11*a_12*a_33 + a_11*a_22*a_33 + a_12*a_31*a_33 + a_21*a_22*a_33 + a_22*a_31*a_33)*x7 - a_11*a_12*a_33 - a_11*a_12*a_43 - a_11*a_22*a_33 - a_11*a_22*a_43 - a_12*a_31*a_33 - a_12*a_31*a_43 - a_21*a_22*a_33 - a_21*a_22*a_43 - a_22*a_31*a_33 - a_22*a_31*a_43,
        (a_12*a_33 + a_12*a_34 + a_12*a_43 + a_12*a_44 + a_22*a_33 + a_22*a_34 + a_22*a_43 + a_22*a_44 + a_33*a_34 + a_33*a_44 + a_43*a_44)*x0 + (a_22*a_33 + a_22*a_34 + a_22*a_43 + a_22*a_44 + a_33*a_34 + a_33*a_44 + a_43*a_44)*x1 + (a_12*a_33 + a_12*a_34 + a_12*a_43 + a_12*a_44 + a_22*a_33 + a_22*a_34 + a_22*a_43 + a_22*a_44 + a_33*a_34 + a_33*a_44 + a_43*a_44)*x2 + (a_11*a_33 + a_11*a_34 + a_11*a_43 + a_11*a_44 + a_31*a_33 + a_31*a_34 + a_31*a_43 + a_31*a_44 + a_33*a_34 + a_33*a_44 + a_43*a_44)*x3 + (a_11*a_33 + a_11*a_34 + a_11*a_43 + a_11*a_44 + a_21*a_33 + a_21*a_34 + a_21*a_43 + a_21*a_44 + a_31*a_33 + a_31*a_34 + a_31*a_43 + a_31*a_44 + a_33*a_34 + a_33*a_44 + a_43*a_44)*x4 + (a_11*a_12 + a_11*a_22 + a_11*a_34 + a_11*a_44 + a_12*a_31 + a_12*a_34 + a_12*a_44 + a_21*a_22 + a_21*a_34 + a_21*a_44 + a_22*a_31 + a_22*a_34 + a_22*a_44 + a_31*a_34 + a_31*a_44)*x5 + (a_11*a_12 + a_11*a_22 + a_11*a_44 + a_12*a_31 + a_12*a_44 + a_21*a_22 + a_21*a_44 + a_22*a_31 + a_22*a_44 + a_31*a_44)*x6 + (a_11*a_12 + a_11*a_22 + a_11*a_33 + a_12*a_31 + a_12*a_33 + a_21*a_22 + a_21*a_33 + a_22*a_31 + a_22*a_33 + a_31*a_33)*x7 - a_11*a_12 - a_11*a_22 - a_11*a_33 - a_11*a_43 - a_12*a_31 - a_12*a_33 - a_12*a_43 - a_21*a_22 - a_21*a_33 - a_21*a_43 - a_22*a_31 - a_22*a_33 - a_22*a_43 - a_31*a_33 - a_31*a_43,
        (a_12 + a_22 + a_33 + a_34 + a_43 + a_44)*x0 + (a_22 + a_33 + a_34 + a_43 + a_44)*x1 + (a_12 + a_22 + a_33 + a_34 + a_43 + a_44)*x2 + (a_11 + a_31 + a_33 + a_34 + a_43 + a_44)*x3 + (a_11 + a_21 + a_31 + a_33 + a_34 + a_43 + a_44)*x4 + (a_11 + a_12 + a_21 + a_22 + a_31 + a_34 + a_44)*x5 + (a_11 + a_12 + a_21 + a_22 + a_31 + a_44)*x6 + (a_11 + a_12 + a_21 + a_22 + a_31 + a_33)*x7 - a_11 - a_12 - a_21 - a_22 - a_31 - a_33 - a_43,
        x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 - 1,
        (a_12*a_34 + a_12*a_44 + a_22*a_34 + a_22*a_44)*x2 + (a_31*a_34 + a_31*a_44)*x3 + (a_31*a_34 + a_31*a_44)*x4 + (a_12*a_31 + a_22*a_31)*x7 - a_12*a_31 - a_22*a_31,
        (a_12 + a_22 + a_34 + a_44)*x2 + a_31*x3 + a_31*x4 + a_31*x7 - a_31,
        x2,
    ]

def sol_10x8():
    return {
        x0: -a_21/a_12*x4,
        x1: a_21/a_12*x4,
        x2: 0,
        x3: -x4,
        x5: a_43/a_34,
        x6: -a_43/a_34,
        x7: 1,
    }

def time_eqs_10x8():
    if len(eqs_10x8()) != 10:
        raise ValueError("Value should be equal to 10")

def time_solve_lin_sys_10x8():
    eqs = eqs_10x8()
    sol = solve_lin_sys(eqs, R_8)
    if sol != sol_10x8():
        raise ValueError("Values should be equal")

def time_verify_sol_10x8():
    eqs = eqs_10x8()
    sol = sol_10x8()
    zeros = [ eq.compose(sol) for eq in eqs ]
    if not all(zero == 0 for zero in zeros):
        raise ValueError("All values in zero should be 0")

def time_to_expr_eqs_10x8():
    eqs = eqs_10x8()
    assert [ R_8.from_expr(eq.as_expr()) for eq in eqs ] == eqs

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\resolvelib\resolvers\resolution.py ===
from __future__ import annotations

import collections
import itertools
import operator
from typing import TYPE_CHECKING, Collection, Generic, Iterable, Mapping

from ..structs import (
    CT,
    KT,
    RT,
    DirectedGraph,
    IterableView,
    IteratorMapping,
    RequirementInformation,
    State,
    build_iter_view,
)
from .abstract import AbstractResolver, Result
from .criterion import Criterion
from .exceptions import (
    InconsistentCandidate,
    RequirementsConflicted,
    ResolutionImpossible,
    ResolutionTooDeep,
    ResolverException,
)

if TYPE_CHECKING:
    from ..providers import AbstractProvider, Preference
    from ..reporters import BaseReporter


def _build_result(state: State[RT, CT, KT]) -> Result[RT, CT, KT]:
    mapping = state.mapping
    all_keys: dict[int, KT | None] = {id(v): k for k, v in mapping.items()}
    all_keys[id(None)] = None

    graph: DirectedGraph[KT | None] = DirectedGraph()
    graph.add(None)  # Sentinel as root dependencies' parent.

    connected: set[KT | None] = {None}
    for key, criterion in state.criteria.items():
        if not _has_route_to_root(state.criteria, key, all_keys, connected):
            continue
        if key not in graph:
            graph.add(key)
        for p in criterion.iter_parent():
            try:
                pkey = all_keys[id(p)]
            except KeyError:
                continue
            if pkey not in graph:
                graph.add(pkey)
            graph.connect(pkey, key)

    return Result(
        mapping={k: v for k, v in mapping.items() if k in connected},
        graph=graph,
        criteria=state.criteria,
    )


class Resolution(Generic[RT, CT, KT]):
    """Stateful resolution object.

    This is designed as a one-off object that holds information to kick start
    the resolution process, and holds the results afterwards.
    """

    def __init__(
        self,
        provider: AbstractProvider[RT, CT, KT],
        reporter: BaseReporter[RT, CT, KT],
    ) -> None:
        self._p = provider
        self._r = reporter
        self._states: list[State[RT, CT, KT]] = []

    @property
    def state(self) -> State[RT, CT, KT]:
        try:
            return self._states[-1]
        except IndexError as e:
            raise AttributeError("state") from e

    def _push_new_state(self) -> None:
        """Push a new state into history.

        This new state will be used to hold resolution results of the next
        coming round.
        """
        base = self._states[-1]
        state = State(
            mapping=base.mapping.copy(),
            criteria=base.criteria.copy(),
            backtrack_causes=base.backtrack_causes[:],
        )
        self._states.append(state)

    def _add_to_criteria(
        self,
        criteria: dict[KT, Criterion[RT, CT]],
        requirement: RT,
        parent: CT | None,
    ) -> None:
        self._r.adding_requirement(requirement=requirement, parent=parent)

        identifier = self._p.identify(requirement_or_candidate=requirement)
        criterion = criteria.get(identifier)
        if criterion:
            incompatibilities = list(criterion.incompatibilities)
        else:
            incompatibilities = []

        matches = self._p.find_matches(
            identifier=identifier,
            requirements=IteratorMapping(
                criteria,
                operator.methodcaller("iter_requirement"),
                {identifier: [requirement]},
            ),
            incompatibilities=IteratorMapping(
                criteria,
                operator.attrgetter("incompatibilities"),
                {identifier: incompatibilities},
            ),
        )

        if criterion:
            information = list(criterion.information)
            information.append(RequirementInformation(requirement, parent))
        else:
            information = [RequirementInformation(requirement, parent)]

        criterion = Criterion(
            candidates=build_iter_view(matches),
            information=information,
            incompatibilities=incompatibilities,
        )
        if not criterion.candidates:
            raise RequirementsConflicted(criterion)
        criteria[identifier] = criterion

    def _remove_information_from_criteria(
        self, criteria: dict[KT, Criterion[RT, CT]], parents: Collection[KT]
    ) -> None:
        """Remove information from parents of criteria.

        Concretely, removes all values from each criterion's ``information``
        field that have one of ``parents`` as provider of the requirement.

        :param criteria: The criteria to update.
        :param parents: Identifiers for which to remove information from all criteria.
        """
        if not parents:
            return
        for key, criterion in criteria.items():
            criteria[key] = Criterion(
                criterion.candidates,
                [
                    information
                    for information in criterion.information
                    if (
                        information.parent is None
                        or self._p.identify(information.parent) not in parents
                    )
                ],
                criterion.incompatibilities,
            )

    def _get_preference(self, name: KT) -> Preference:
        return self._p.get_preference(
            identifier=name,
            resolutions=self.state.mapping,
            candidates=IteratorMapping(
                self.state.criteria,
                operator.attrgetter("candidates"),
            ),
            information=IteratorMapping(
                self.state.criteria,
                operator.attrgetter("information"),
            ),
            backtrack_causes=self.state.backtrack_causes,
        )

    def _is_current_pin_satisfying(
        self, name: KT, criterion: Criterion[RT, CT]
    ) -> bool:
        try:
            current_pin = self.state.mapping[name]
        except KeyError:
            return False
        return all(
            self._p.is_satisfied_by(requirement=r, candidate=current_pin)
            for r in criterion.iter_requirement()
        )

    def _get_updated_criteria(self, candidate: CT) -> dict[KT, Criterion[RT, CT]]:
        criteria = self.state.criteria.copy()
        for requirement in self._p.get_dependencies(candidate=candidate):
            self._add_to_criteria(criteria, requirement, parent=candidate)
        return criteria

    def _attempt_to_pin_criterion(self, name: KT) -> list[Criterion[RT, CT]]:
        criterion = self.state.criteria[name]

        causes: list[Criterion[RT, CT]] = []
        for candidate in criterion.candidates:
            try:
                criteria = self._get_updated_criteria(candidate)
            except RequirementsConflicted as e:
                self._r.rejecting_candidate(e.criterion, candidate)
                causes.append(e.criterion)
                continue

            # Check the newly-pinned candidate actually works. This should
            # always pass under normal circumstances, but in the case of a
            # faulty provider, we will raise an error to notify the implementer
            # to fix find_matches() and/or is_satisfied_by().
            satisfied = all(
                self._p.is_satisfied_by(requirement=r, candidate=candidate)
                for r in criterion.iter_requirement()
            )
            if not satisfied:
                raise InconsistentCandidate(candidate, criterion)

            self._r.pinning(candidate=candidate)
            self.state.criteria.update(criteria)

            # Put newly-pinned candidate at the end. This is essential because
            # backtracking looks at this mapping to get the last pin.
            self.state.mapping.pop(name, None)
            self.state.mapping[name] = candidate

            return []

        # All candidates tried, nothing works. This criterion is a dead
        # end, signal for backtracking.
        return causes

    def _patch_criteria(
        self, incompatibilities_from_broken: list[tuple[KT, list[CT]]]
    ) -> bool:
        # Create a new state from the last known-to-work one, and apply
        # the previously gathered incompatibility information.
        for k, incompatibilities in incompatibilities_from_broken:
            if not incompatibilities:
                continue
            try:
                criterion = self.state.criteria[k]
            except KeyError:
                continue
            matches = self._p.find_matches(
                identifier=k,
                requirements=IteratorMapping(
                    self.state.criteria,
                    operator.methodcaller("iter_requirement"),
                ),
                incompatibilities=IteratorMapping(
                    self.state.criteria,
                    operator.attrgetter("incompatibilities"),
                    {k: incompatibilities},
                ),
            )
            candidates: IterableView[CT] = build_iter_view(matches)
            if not candidates:
                return False
            incompatibilities.extend(criterion.incompatibilities)
            self.state.criteria[k] = Criterion(
                candidates=candidates,
                information=list(criterion.information),
                incompatibilities=incompatibilities,
            )
        return True

    def _backjump(self, causes: list[RequirementInformation[RT, CT]]) -> bool:
        """Perform backjumping.

        When we enter here, the stack is like this::

            [ state Z ]
            [ state Y ]
            [ state X ]
            .... earlier states are irrelevant.

        1. No pins worked for Z, so it does not have a pin.
        2. We want to reset state Y to unpinned, and pin another candidate.
        3. State X holds what state Y was before the pin, but does not
           have the incompatibility information gathered in state Y.

        Each iteration of the loop will:

        1.  Identify Z. The incompatibility is not always caused by the latest
            state. For example, given three requirements A, B and C, with
            dependencies A1, B1 and C1, where A1 and B1 are incompatible: the
            last state might be related to C, so we want to discard the
            previous state.
        2.  Discard Z.
        3.  Discard Y but remember its incompatibility information gathered
            previously, and the failure we're dealing with right now.
        4.  Push a new state Y' based on X, and apply the incompatibility
            information from Y to Y'.
        5a. If this causes Y' to conflict, we need to backtrack again. Make Y'
            the new Z and go back to step 2.
        5b. If the incompatibilities apply cleanly, end backtracking.
        """
        incompatible_reqs: Iterable[CT | RT] = itertools.chain(
            (c.parent for c in causes if c.parent is not None),
            (c.requirement for c in causes),
        )
        incompatible_deps = {self._p.identify(r) for r in incompatible_reqs}
        while len(self._states) >= 3:
            # Remove the state that triggered backtracking.
            del self._states[-1]

            # Optimistically backtrack to a state that caused the incompatibility
            broken_state = self.state
            while True:
                # Retrieve the last candidate pin and known incompatibilities.
                try:
                    broken_state = self._states.pop()
                    name, candidate = broken_state.mapping.popitem()
                except (IndexError, KeyError):
                    raise ResolutionImpossible(causes) from None

                # Only backjump if the current broken state is
                # an incompatible dependency
                if name not in incompatible_deps:
                    break

                # If the current dependencies and the incompatible dependencies
                # are overlapping then we have found a cause of the incompatibility
                current_dependencies = {
                    self._p.identify(d) for d in self._p.get_dependencies(candidate)
                }
                if not current_dependencies.isdisjoint(incompatible_deps):
                    break

                # Fallback: We should not backtrack to the point where
                # broken_state.mapping is empty, so stop backtracking for
                # a chance for the resolution to recover
                if not broken_state.mapping:
                    break

            incompatibilities_from_broken = [
                (k, list(v.incompatibilities)) for k, v in broken_state.criteria.items()
            ]

            # Also mark the newly known incompatibility.
            incompatibilities_from_broken.append((name, [candidate]))

            self._push_new_state()
            success = self._patch_criteria(incompatibilities_from_broken)

            # It works! Let's work on this new state.
            if success:
                return True

            # State does not work after applying known incompatibilities.
            # Try the still previous state.

        # No way to backtrack anymore.
        return False

    def _extract_causes(
        self, criteron: list[Criterion[RT, CT]]
    ) -> list[RequirementInformation[RT, CT]]:
        """Extract causes from list of criterion and deduplicate"""
        return list({id(i): i for c in criteron for i in c.information}.values())

    def resolve(self, requirements: Iterable[RT], max_rounds: int) -> State[RT, CT, KT]:
        if self._states:
            raise RuntimeError("already resolved")

        self._r.starting()

        # Initialize the root state.
        self._states = [
            State(
                mapping=collections.OrderedDict(),
                criteria={},
                backtrack_causes=[],
            )
        ]
        for r in requirements:
            try:
                self._add_to_criteria(self.state.criteria, r, parent=None)
            except RequirementsConflicted as e:
                raise ResolutionImpossible(e.criterion.information) from e

        # The root state is saved as a sentinel so the first ever pin can have
        # something to backtrack to if it fails. The root state is basically
        # pinning the virtual "root" package in the graph.
        self._push_new_state()

        for round_index in range(max_rounds):
            self._r.starting_round(index=round_index)

            unsatisfied_names = [
                key
                for key, criterion in self.state.criteria.items()
                if not self._is_current_pin_satisfying(key, criterion)
            ]

            # All criteria are accounted for. Nothing more to pin, we are done!
            if not unsatisfied_names:
                self._r.ending(state=self.state)
                return self.state

            # keep track of satisfied names to calculate diff after pinning
            satisfied_names = set(self.state.criteria.keys()) - set(unsatisfied_names)

            if len(unsatisfied_names) > 1:
                narrowed_unstatisfied_names = list(
                    self._p.narrow_requirement_selection(
                        identifiers=unsatisfied_names,
                        resolutions=self.state.mapping,
                        candidates=IteratorMapping(
                            self.state.criteria,
                            operator.attrgetter("candidates"),
                        ),
                        information=IteratorMapping(
                            self.state.criteria,
                            operator.attrgetter("information"),
                        ),
                        backtrack_causes=self.state.backtrack_causes,
                    )
                )
            else:
                narrowed_unstatisfied_names = unsatisfied_names

            # If there are no unsatisfied names use unsatisfied names
            if not narrowed_unstatisfied_names:
                raise RuntimeError("narrow_requirement_selection returned 0 names")

            # If there is only 1 unsatisfied name skip calling self._get_preference
            if len(narrowed_unstatisfied_names) > 1:
                # Choose the most preferred unpinned criterion to try.
                name = min(narrowed_unstatisfied_names, key=self._get_preference)
            else:
                name = narrowed_unstatisfied_names[0]

            failure_criterion = self._attempt_to_pin_criterion(name)

            if failure_criterion:
                causes = self._extract_causes(failure_criterion)
                # Backjump if pinning fails. The backjump process puts us in
                # an unpinned state, so we can work on it in the next round.
                self._r.resolving_conflicts(causes=causes)
                success = self._backjump(causes)
                self.state.backtrack_causes[:] = causes

                # Dead ends everywhere. Give up.
                if not success:
                    raise ResolutionImpossible(self.state.backtrack_causes)
            else:
                # discard as information sources any invalidated names
                # (unsatisfied names that were previously satisfied)
                newly_unsatisfied_names = {
                    key
                    for key, criterion in self.state.criteria.items()
                    if key in satisfied_names
                    and not self._is_current_pin_satisfying(key, criterion)
                }
                self._remove_information_from_criteria(
                    self.state.criteria, newly_unsatisfied_names
                )
                # Pinning was successful. Push a new state to do another pin.
                self._push_new_state()

            self._r.ending_round(index=round_index, state=self.state)

        raise ResolutionTooDeep(max_rounds)


class Resolver(AbstractResolver[RT, CT, KT]):
    """The thing that performs the actual resolution work."""

    base_exception = ResolverException

    def resolve(  # type: ignore[override]
        self,
        requirements: Iterable[RT],
        max_rounds: int = 100,
    ) -> Result[RT, CT, KT]:
        """Take a collection of constraints, spit out the resolution result.

        The return value is a representation to the final resolution result. It
        is a tuple subclass with three public members:

        * `mapping`: A dict of resolved candidates. Each key is an identifier
            of a requirement (as returned by the provider's `identify` method),
            and the value is the resolved candidate.
        * `graph`: A `DirectedGraph` instance representing the dependency tree.
            The vertices are keys of `mapping`, and each edge represents *why*
            a particular package is included. A special vertex `None` is
            included to represent parents of user-supplied requirements.
        * `criteria`: A dict of "criteria" that hold detailed information on
            how edges in the graph are derived. Each key is an identifier of a
            requirement, and the value is a `Criterion` instance.

        The following exceptions may be raised if a resolution cannot be found:

        * `ResolutionImpossible`: A resolution cannot be found for the given
            combination of requirements. The `causes` attribute of the
            exception is a list of (requirement, parent), giving the
            requirements that could not be satisfied.
        * `ResolutionTooDeep`: The dependency tree is too deeply nested and
            the resolver gave up. This is usually caused by a circular
            dependency, but you can try to resolve this by increasing the
            `max_rounds` argument.
        """
        resolution = Resolution(self.provider, self.reporter)
        state = resolution.resolve(requirements, max_rounds=max_rounds)
        return _build_result(state)


def _has_route_to_root(
    criteria: Mapping[KT, Criterion[RT, CT]],
    key: KT | None,
    all_keys: dict[int, KT | None],
    connected: set[KT | None],
) -> bool:
    if key in connected:
        return True
    if key not in criteria:
        return False
    assert key is not None
    for p in criteria[key].iter_parent():
        try:
            pkey = all_keys[id(p)]
        except KeyError:
            continue
        if pkey in connected:
            connected.add(key)
            return True
        if _has_route_to_root(criteria, pkey, all_keys, connected):
            connected.add(key)
            return True
    return False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\util\_py_collections.py ===
# util/_py_collections.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

from __future__ import annotations

from itertools import filterfalse
from typing import AbstractSet
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
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from ..util.typing import Self

_T = TypeVar("_T", bound=Any)
_S = TypeVar("_S", bound=Any)
_KT = TypeVar("_KT", bound=Any)
_VT = TypeVar("_VT", bound=Any)


class ReadOnlyContainer:
    __slots__ = ()

    def _readonly(self, *arg: Any, **kw: Any) -> NoReturn:
        raise TypeError(
            "%s object is immutable and/or readonly" % self.__class__.__name__
        )

    def _immutable(self, *arg: Any, **kw: Any) -> NoReturn:
        raise TypeError("%s object is immutable" % self.__class__.__name__)

    def __delitem__(self, key: Any) -> NoReturn:
        self._readonly()

    def __setitem__(self, key: Any, value: Any) -> NoReturn:
        self._readonly()

    def __setattr__(self, key: str, value: Any) -> NoReturn:
        self._readonly()


class ImmutableDictBase(ReadOnlyContainer, Dict[_KT, _VT]):
    if TYPE_CHECKING:

        def __new__(cls, *args: Any) -> Self: ...

        def __init__(cls, *args: Any): ...

    def _readonly(self, *arg: Any, **kw: Any) -> NoReturn:
        self._immutable()

    def clear(self) -> NoReturn:
        self._readonly()

    def pop(self, key: Any, default: Optional[Any] = None) -> NoReturn:
        self._readonly()

    def popitem(self) -> NoReturn:
        self._readonly()

    def setdefault(self, key: Any, default: Optional[Any] = None) -> NoReturn:
        self._readonly()

    def update(self, *arg: Any, **kw: Any) -> NoReturn:
        self._readonly()


class immutabledict(ImmutableDictBase[_KT, _VT]):
    def __new__(cls, *args):
        new = ImmutableDictBase.__new__(cls)
        dict.__init__(new, *args)
        return new

    def __init__(
        self, *args: Union[Mapping[_KT, _VT], Iterable[Tuple[_KT, _VT]]]
    ):
        pass

    def __reduce__(self):
        return immutabledict, (dict(self),)

    def union(
        self, __d: Optional[Mapping[_KT, _VT]] = None
    ) -> immutabledict[_KT, _VT]:
        if not __d:
            return self

        new = ImmutableDictBase.__new__(self.__class__)
        dict.__init__(new, self)
        dict.update(new, __d)  # type: ignore
        return new

    def _union_w_kw(
        self, __d: Optional[Mapping[_KT, _VT]] = None, **kw: _VT
    ) -> immutabledict[_KT, _VT]:
        # not sure if C version works correctly w/ this yet
        if not __d and not kw:
            return self

        new = ImmutableDictBase.__new__(self.__class__)
        dict.__init__(new, self)
        if __d:
            dict.update(new, __d)  # type: ignore
        dict.update(new, kw)  # type: ignore
        return new

    def merge_with(
        self, *dicts: Optional[Mapping[_KT, _VT]]
    ) -> immutabledict[_KT, _VT]:
        new = None
        for d in dicts:
            if d:
                if new is None:
                    new = ImmutableDictBase.__new__(self.__class__)
                    dict.__init__(new, self)
                dict.update(new, d)  # type: ignore
        if new is None:
            return self

        return new

    def __repr__(self) -> str:
        return "immutabledict(%s)" % dict.__repr__(self)

    # PEP 584
    def __ior__(self, __value: Any) -> NoReturn:  # type: ignore
        self._readonly()

    def __or__(  # type: ignore[override]
        self, __value: Mapping[_KT, _VT]
    ) -> immutabledict[_KT, _VT]:
        return immutabledict(
            super().__or__(__value),  # type: ignore[call-overload]
        )

    def __ror__(  # type: ignore[override]
        self, __value: Mapping[_KT, _VT]
    ) -> immutabledict[_KT, _VT]:
        return immutabledict(
            super().__ror__(__value),  # type: ignore[call-overload]
        )


class OrderedSet(Set[_T]):
    __slots__ = ("_list",)

    _list: List[_T]

    def __init__(self, d: Optional[Iterable[_T]] = None) -> None:
        if d is not None:
            self._list = unique_list(d)
            super().update(self._list)
        else:
            self._list = []

    def copy(self) -> OrderedSet[_T]:
        cp = self.__class__()
        cp._list = self._list.copy()
        set.update(cp, cp._list)
        return cp

    def add(self, element: _T) -> None:
        if element not in self:
            self._list.append(element)
        super().add(element)

    def remove(self, element: _T) -> None:
        super().remove(element)
        self._list.remove(element)

    def pop(self) -> _T:
        try:
            value = self._list.pop()
        except IndexError:
            raise KeyError("pop from an empty set") from None
        super().remove(value)
        return value

    def insert(self, pos: int, element: _T) -> None:
        if element not in self:
            self._list.insert(pos, element)
        super().add(element)

    def discard(self, element: _T) -> None:
        if element in self:
            self._list.remove(element)
            super().remove(element)

    def clear(self) -> None:
        super().clear()
        self._list = []

    def __getitem__(self, key: int) -> _T:
        return self._list[key]

    def __iter__(self) -> Iterator[_T]:
        return iter(self._list)

    def __add__(self, other: Iterator[_T]) -> OrderedSet[_T]:
        return self.union(other)

    def __repr__(self) -> str:
        return "%s(%r)" % (self.__class__.__name__, self._list)

    __str__ = __repr__

    def update(self, *iterables: Iterable[_T]) -> None:
        for iterable in iterables:
            for e in iterable:
                if e not in self:
                    self._list.append(e)
                    super().add(e)

    def __ior__(self, other: AbstractSet[_S]) -> OrderedSet[Union[_T, _S]]:
        self.update(other)
        return self

    def union(self, *other: Iterable[_S]) -> OrderedSet[Union[_T, _S]]:
        result: OrderedSet[Union[_T, _S]] = self.copy()
        result.update(*other)
        return result

    def __or__(self, other: AbstractSet[_S]) -> OrderedSet[Union[_T, _S]]:
        return self.union(other)

    def intersection(self, *other: Iterable[Any]) -> OrderedSet[_T]:
        other_set: Set[Any] = set()
        other_set.update(*other)
        return self.__class__(a for a in self if a in other_set)

    def __and__(self, other: AbstractSet[object]) -> OrderedSet[_T]:
        return self.intersection(other)

    def symmetric_difference(self, other: Iterable[_T]) -> OrderedSet[_T]:
        collection: Collection[_T]
        if isinstance(other, set):
            collection = other_set = other
        elif isinstance(other, Collection):
            collection = other
            other_set = set(other)
        else:
            collection = list(other)
            other_set = set(collection)
        result = self.__class__(a for a in self if a not in other_set)
        result.update(a for a in collection if a not in self)
        return result

    def __xor__(self, other: AbstractSet[_S]) -> OrderedSet[Union[_T, _S]]:
        return cast(OrderedSet[Union[_T, _S]], self).symmetric_difference(
            other
        )

    def difference(self, *other: Iterable[Any]) -> OrderedSet[_T]:
        other_set = super().difference(*other)
        return self.__class__(a for a in self._list if a in other_set)

    def __sub__(self, other: AbstractSet[Optional[_T]]) -> OrderedSet[_T]:
        return self.difference(other)

    def intersection_update(self, *other: Iterable[Any]) -> None:
        super().intersection_update(*other)
        self._list = [a for a in self._list if a in self]

    def __iand__(self, other: AbstractSet[object]) -> OrderedSet[_T]:
        self.intersection_update(other)
        return self

    def symmetric_difference_update(self, other: Iterable[Any]) -> None:
        collection = other if isinstance(other, Collection) else list(other)
        super().symmetric_difference_update(collection)
        self._list = [a for a in self._list if a in self]
        self._list += [a for a in collection if a in self]

    def __ixor__(self, other: AbstractSet[_S]) -> OrderedSet[Union[_T, _S]]:
        self.symmetric_difference_update(other)
        return cast(OrderedSet[Union[_T, _S]], self)

    def difference_update(self, *other: Iterable[Any]) -> None:
        super().difference_update(*other)
        self._list = [a for a in self._list if a in self]

    def __isub__(self, other: AbstractSet[Optional[_T]]) -> OrderedSet[_T]:  # type: ignore  # noqa: E501
        self.difference_update(other)
        return self


class IdentitySet:
    """A set that considers only object id() for uniqueness.

    This strategy has edge cases for builtin types- it's possible to have
    two 'foo' strings in one of these sets, for example.  Use sparingly.

    """

    _members: Dict[int, Any]

    def __init__(self, iterable: Optional[Iterable[Any]] = None):
        self._members = dict()
        if iterable:
            self.update(iterable)

    def add(self, value: Any) -> None:
        self._members[id(value)] = value

    def __contains__(self, value: Any) -> bool:
        return id(value) in self._members

    def remove(self, value: Any) -> None:
        del self._members[id(value)]

    def discard(self, value: Any) -> None:
        try:
            self.remove(value)
        except KeyError:
            pass

    def pop(self) -> Any:
        try:
            pair = self._members.popitem()
            return pair[1]
        except KeyError:
            raise KeyError("pop from an empty set")

    def clear(self) -> None:
        self._members.clear()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, IdentitySet):
            return self._members == other._members
        else:
            return False

    def __ne__(self, other: Any) -> bool:
        if isinstance(other, IdentitySet):
            return self._members != other._members
        else:
            return True

    def issubset(self, iterable: Iterable[Any]) -> bool:
        if isinstance(iterable, self.__class__):
            other = iterable
        else:
            other = self.__class__(iterable)

        if len(self) > len(other):
            return False
        for m in filterfalse(
            other._members.__contains__, iter(self._members.keys())
        ):
            return False
        return True

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, IdentitySet):
            return NotImplemented
        return self.issubset(other)

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, IdentitySet):
            return NotImplemented
        return len(self) < len(other) and self.issubset(other)

    def issuperset(self, iterable: Iterable[Any]) -> bool:
        if isinstance(iterable, self.__class__):
            other = iterable
        else:
            other = self.__class__(iterable)

        if len(self) < len(other):
            return False

        for m in filterfalse(
            self._members.__contains__, iter(other._members.keys())
        ):
            return False
        return True

    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, IdentitySet):
            return NotImplemented
        return self.issuperset(other)

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, IdentitySet):
            return NotImplemented
        return len(self) > len(other) and self.issuperset(other)

    def union(self, iterable: Iterable[Any]) -> IdentitySet:
        result = self.__class__()
        members = self._members
        result._members.update(members)
        result._members.update((id(obj), obj) for obj in iterable)
        return result

    def __or__(self, other: Any) -> IdentitySet:
        if not isinstance(other, IdentitySet):
            return NotImplemented
        return self.union(other)

    def update(self, iterable: Iterable[Any]) -> None:
        self._members.update((id(obj), obj) for obj in iterable)

    def __ior__(self, other: Any) -> IdentitySet:
        if not isinstance(other, IdentitySet):
            return NotImplemented
        self.update(other)
        return self

    def difference(self, iterable: Iterable[Any]) -> IdentitySet:
        result = self.__new__(self.__class__)
        other: Collection[Any]

        if isinstance(iterable, self.__class__):
            other = iterable._members
        else:
            other = {id(obj) for obj in iterable}
        result._members = {
            k: v for k, v in self._members.items() if k not in other
        }
        return result

    def __sub__(self, other: IdentitySet) -> IdentitySet:
        if not isinstance(other, IdentitySet):
            return NotImplemented
        return self.difference(other)

    def difference_update(self, iterable: Iterable[Any]) -> None:
        self._members = self.difference(iterable)._members

    def __isub__(self, other: IdentitySet) -> IdentitySet:
        if not isinstance(other, IdentitySet):
            return NotImplemented
        self.difference_update(other)
        return self

    def intersection(self, iterable: Iterable[Any]) -> IdentitySet:
        result = self.__new__(self.__class__)

        other: Collection[Any]

        if isinstance(iterable, self.__class__):
            other = iterable._members
        else:
            other = {id(obj) for obj in iterable}
        result._members = {
            k: v for k, v in self._members.items() if k in other
        }
        return result

    def __and__(self, other: IdentitySet) -> IdentitySet:
        if not isinstance(other, IdentitySet):
            return NotImplemented
        return self.intersection(other)

    def intersection_update(self, iterable: Iterable[Any]) -> None:
        self._members = self.intersection(iterable)._members

    def __iand__(self, other: IdentitySet) -> IdentitySet:
        if not isinstance(other, IdentitySet):
            return NotImplemented
        self.intersection_update(other)
        return self

    def symmetric_difference(self, iterable: Iterable[Any]) -> IdentitySet:
        result = self.__new__(self.__class__)
        if isinstance(iterable, self.__class__):
            other = iterable._members
        else:
            other = {id(obj): obj for obj in iterable}
        result._members = {
            k: v for k, v in self._members.items() if k not in other
        }
        result._members.update(
            (k, v) for k, v in other.items() if k not in self._members
        )
        return result

    def __xor__(self, other: IdentitySet) -> IdentitySet:
        if not isinstance(other, IdentitySet):
            return NotImplemented
        return self.symmetric_difference(other)

    def symmetric_difference_update(self, iterable: Iterable[Any]) -> None:
        self._members = self.symmetric_difference(iterable)._members

    def __ixor__(self, other: IdentitySet) -> IdentitySet:
        if not isinstance(other, IdentitySet):
            return NotImplemented
        self.symmetric_difference(other)
        return self

    def copy(self) -> IdentitySet:
        result = self.__new__(self.__class__)
        result._members = self._members.copy()
        return result

    __copy__ = copy

    def __len__(self) -> int:
        return len(self._members)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._members.values())

    def __hash__(self) -> NoReturn:
        raise TypeError("set objects are unhashable")

    def __repr__(self) -> str:
        return "%s(%r)" % (type(self).__name__, list(self._members.values()))


def unique_list(
    seq: Iterable[_T], hashfunc: Optional[Callable[[_T], int]] = None
) -> List[_T]:
    seen: Set[Any] = set()
    seen_add = seen.add
    if not hashfunc:
        return [x for x in seq if x not in seen and not seen_add(x)]
    else:
        return [
            x
            for x in seq
            if hashfunc(x) not in seen and not seen_add(hashfunc(x))
        ]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\numpy.py ===
from sympy.core import S
from sympy.core.function import Lambda
from sympy.core.power import Pow
from .pycode import PythonCodePrinter, _known_functions_math, _print_known_const, _print_known_func, _unpack_integral_limits, ArrayPrinter
from .codeprinter import CodePrinter


_not_in_numpy = 'erf erfc factorial gamma loggamma'.split()
_in_numpy = [(k, v) for k, v in _known_functions_math.items() if k not in _not_in_numpy]
_known_functions_numpy = dict(_in_numpy, **{
    'acos': 'arccos',
    'acosh': 'arccosh',
    'asin': 'arcsin',
    'asinh': 'arcsinh',
    'atan': 'arctan',
    'atan2': 'arctan2',
    'atanh': 'arctanh',
    'exp2': 'exp2',
    'sign': 'sign',
    'logaddexp': 'logaddexp',
    'logaddexp2': 'logaddexp2',
    'isinf': 'isinf',
    'isnan': 'isnan',

})
_known_constants_numpy = {
    'Exp1': 'e',
    'Pi': 'pi',
    'EulerGamma': 'euler_gamma',
    'NaN': 'nan',
    'Infinity': 'inf',
}

_numpy_known_functions = {k: 'numpy.' + v for k, v in _known_functions_numpy.items()}
_numpy_known_constants = {k: 'numpy.' + v for k, v in _known_constants_numpy.items()}

class NumPyPrinter(ArrayPrinter, PythonCodePrinter):
    """
    Numpy printer which handles vectorized piecewise functions,
    logical operators, etc.
    """

    _module = 'numpy'
    _kf = _numpy_known_functions
    _kc = _numpy_known_constants

    def __init__(self, settings=None):
        """
        `settings` is passed to CodePrinter.__init__()
        `module` specifies the array module to use, currently 'NumPy', 'CuPy'
        or 'JAX'.
        """
        self.language = "Python with {}".format(self._module)
        self.printmethod = "_{}code".format(self._module)

        self._kf = {**PythonCodePrinter._kf, **self._kf}

        super().__init__(settings=settings)


    def _print_seq(self, seq):
        "General sequence printer: converts to tuple"
        # Print tuples here instead of lists because numba supports
        #     tuples in nopython mode.
        delimiter=', '
        return '({},)'.format(delimiter.join(self._print(item) for item in seq))

    def _print_NegativeInfinity(self, expr):
        return '-' + self._print(S.Infinity)

    def _print_MatMul(self, expr):
        "Matrix multiplication printer"
        if expr.as_coeff_matrices()[0] is not S.One:
            expr_list = expr.as_coeff_matrices()[1]+[(expr.as_coeff_matrices()[0])]
            return '({})'.format(').dot('.join(self._print(i) for i in expr_list))
        return '({})'.format(').dot('.join(self._print(i) for i in expr.args))

    def _print_MatPow(self, expr):
        "Matrix power printer"
        return '{}({}, {})'.format(self._module_format(self._module + '.linalg.matrix_power'),
            self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_Inverse(self, expr):
        "Matrix inverse printer"
        return '{}({})'.format(self._module_format(self._module + '.linalg.inv'),
            self._print(expr.args[0]))

    def _print_DotProduct(self, expr):
        # DotProduct allows any shape order, but numpy.dot does matrix
        # multiplication, so we have to make sure it gets 1 x n by n x 1.
        arg1, arg2 = expr.args
        if arg1.shape[0] != 1:
            arg1 = arg1.T
        if arg2.shape[1] != 1:
            arg2 = arg2.T

        return "%s(%s, %s)" % (self._module_format(self._module + '.dot'),
                               self._print(arg1),
                               self._print(arg2))

    def _print_MatrixSolve(self, expr):
        return "%s(%s, %s)" % (self._module_format(self._module + '.linalg.solve'),
                               self._print(expr.matrix),
                               self._print(expr.vector))

    def _print_ZeroMatrix(self, expr):
        return '{}({})'.format(self._module_format(self._module + '.zeros'),
            self._print(expr.shape))

    def _print_OneMatrix(self, expr):
        return '{}({})'.format(self._module_format(self._module + '.ones'),
            self._print(expr.shape))

    def _print_FunctionMatrix(self, expr):
        from sympy.abc import i, j
        lamda = expr.lamda
        if not isinstance(lamda, Lambda):
            lamda = Lambda((i, j), lamda(i, j))
        return '{}(lambda {}: {}, {})'.format(self._module_format(self._module + '.fromfunction'),
            ', '.join(self._print(arg) for arg in lamda.args[0]),
            self._print(lamda.args[1]), self._print(expr.shape))

    def _print_HadamardProduct(self, expr):
        func = self._module_format(self._module + '.multiply')
        return ''.join('{}({}, '.format(func, self._print(arg)) \
            for arg in expr.args[:-1]) + "{}{}".format(self._print(expr.args[-1]),
            ')' * (len(expr.args) - 1))

    def _print_KroneckerProduct(self, expr):
        func = self._module_format(self._module + '.kron')
        return ''.join('{}({}, '.format(func, self._print(arg)) \
            for arg in expr.args[:-1]) + "{}{}".format(self._print(expr.args[-1]),
            ')' * (len(expr.args) - 1))

    def _print_Adjoint(self, expr):
        return '{}({}({}))'.format(
            self._module_format(self._module + '.conjugate'),
            self._module_format(self._module + '.transpose'),
            self._print(expr.args[0]))

    def _print_DiagonalOf(self, expr):
        vect = '{}({})'.format(
            self._module_format(self._module + '.diag'),
            self._print(expr.arg))
        return '{}({}, (-1, 1))'.format(
            self._module_format(self._module + '.reshape'), vect)

    def _print_DiagMatrix(self, expr):
        return '{}({})'.format(self._module_format(self._module + '.diagflat'),
            self._print(expr.args[0]))

    def _print_DiagonalMatrix(self, expr):
        return '{}({}, {}({}, {}))'.format(self._module_format(self._module + '.multiply'),
            self._print(expr.arg), self._module_format(self._module + '.eye'),
            self._print(expr.shape[0]), self._print(expr.shape[1]))

    def _print_Piecewise(self, expr):
        "Piecewise function printer"
        from sympy.logic.boolalg import ITE, simplify_logic
        def print_cond(cond):
            """ Problem having an ITE in the cond. """
            if cond.has(ITE):
                return self._print(simplify_logic(cond))
            else:
                return self._print(cond)
        exprs = '[{}]'.format(','.join(self._print(arg.expr) for arg in expr.args))
        conds = '[{}]'.format(','.join(print_cond(arg.cond) for arg in expr.args))
        # If [default_value, True] is a (expr, cond) sequence in a Piecewise object
        #     it will behave the same as passing the 'default' kwarg to select()
        #     *as long as* it is the last element in expr.args.
        # If this is not the case, it may be triggered prematurely.
        return '{}({}, {}, default={})'.format(
            self._module_format(self._module + '.select'), conds, exprs,
            self._print(S.NaN))

    def _print_Relational(self, expr):
        "Relational printer for Equality and Unequality"
        op = {
            '==' :'equal',
            '!=' :'not_equal',
            '<'  :'less',
            '<=' :'less_equal',
            '>'  :'greater',
            '>=' :'greater_equal',
        }
        if expr.rel_op in op:
            lhs = self._print(expr.lhs)
            rhs = self._print(expr.rhs)
            return '{op}({lhs}, {rhs})'.format(op=self._module_format(self._module + '.'+op[expr.rel_op]),
                                               lhs=lhs, rhs=rhs)
        return super()._print_Relational(expr)

    def _print_And(self, expr):
        "Logical And printer"
        # We have to override LambdaPrinter because it uses Python 'and' keyword.
        # If LambdaPrinter didn't define it, we could use StrPrinter's
        # version of the function and add 'logical_and' to NUMPY_TRANSLATIONS.
        return '{}.reduce(({}))'.format(self._module_format(self._module + '.logical_and'), ','.join(self._print(i) for i in expr.args))

    def _print_Or(self, expr):
        "Logical Or printer"
        # We have to override LambdaPrinter because it uses Python 'or' keyword.
        # If LambdaPrinter didn't define it, we could use StrPrinter's
        # version of the function and add 'logical_or' to NUMPY_TRANSLATIONS.
        return '{}.reduce(({}))'.format(self._module_format(self._module + '.logical_or'), ','.join(self._print(i) for i in expr.args))

    def _print_Not(self, expr):
        "Logical Not printer"
        # We have to override LambdaPrinter because it uses Python 'not' keyword.
        # If LambdaPrinter didn't define it, we would still have to define our
        #     own because StrPrinter doesn't define it.
        return '{}({})'.format(self._module_format(self._module + '.logical_not'), ','.join(self._print(i) for i in expr.args))

    def _print_Pow(self, expr, rational=False):
        # XXX Workaround for negative integer power error
        if expr.exp.is_integer and expr.exp.is_negative:
            expr = Pow(expr.base, expr.exp.evalf(), evaluate=False)
        return self._hprint_Pow(expr, rational=rational, sqrt=self._module + '.sqrt')

    def _helper_minimum_maximum(self, op: str, *args):
        if len(args) == 0:
            raise NotImplementedError(f"Need at least one argument for {op}")
        elif len(args) == 1:
            return self._print(args[0])
        _reduce = self._module_format('functools.reduce')
        s_args = [self._print(arg) for arg in args]
        return f"{_reduce}({op}, [{', '.join(s_args)}])"

    def _print_Min(self, expr):
        return self._print_minimum(expr)

    def _print_amin(self, expr):
        return '{}({}, axis={})'.format(self._module_format(self._module + '.amin'), self._print(expr.array), self._print(expr.axis))

    def _print_minimum(self, expr):
        op = self._module_format(self._module + '.minimum')
        return self._helper_minimum_maximum(op, *expr.args)

    def _print_Max(self, expr):
        return self._print_maximum(expr)

    def _print_amax(self, expr):
        return '{}({}, axis={})'.format(self._module_format(self._module + '.amax'), self._print(expr.array), self._print(expr.axis))

    def _print_maximum(self, expr):
        op = self._module_format(self._module + '.maximum')
        return self._helper_minimum_maximum(op, *expr.args)

    def _print_arg(self, expr):
        return "%s(%s)" % (self._module_format(self._module + '.angle'), self._print(expr.args[0]))

    def _print_im(self, expr):
        return "%s(%s)" % (self._module_format(self._module + '.imag'), self._print(expr.args[0]))

    def _print_Mod(self, expr):
        return "%s(%s)" % (self._module_format(self._module + '.mod'), ', '.join(
            (self._print(arg) for arg in expr.args)))

    def _print_re(self, expr):
        return "%s(%s)" % (self._module_format(self._module + '.real'), self._print(expr.args[0]))

    def _print_sinc(self, expr):
        return "%s(%s)" % (self._module_format(self._module + '.sinc'), self._print(expr.args[0]/S.Pi))

    def _print_MatrixBase(self, expr):
        if 0 in expr.shape:
            func = self._module_format(f'{self._module}.{self._zeros}')
            return f"{func}({self._print(expr.shape)})"
        func = self.known_functions.get(expr.__class__.__name__, None)
        if func is None:
            func = self._module_format(f'{self._module}.array')
        return "%s(%s)" % (func, self._print(expr.tolist()))

    def _print_Identity(self, expr):
        shape = expr.shape
        if all(dim.is_Integer for dim in shape):
            return "%s(%s)" % (self._module_format(self._module + '.eye'), self._print(expr.shape[0]))
        else:
            raise NotImplementedError("Symbolic matrix dimensions are not yet supported for identity matrices")

    def _print_BlockMatrix(self, expr):
        return '{}({})'.format(self._module_format(self._module + '.block'),
                                 self._print(expr.args[0].tolist()))

    def _print_NDimArray(self, expr):
        if expr.rank() == 0:
            func = self._module_format(f'{self._module}.array')
            return f"{func}({self._print(expr[()])})"
        if 0 in expr.shape:
            func = self._module_format(f'{self._module}.{self._zeros}')
            return f"{func}({self._print(expr.shape)})"
        func = self._module_format(f'{self._module}.array')
        return f"{func}({self._print(expr.tolist())})"

    _add = "add"
    _einsum = "einsum"
    _transpose = "transpose"
    _ones = "ones"
    _zeros = "zeros"

    _print_lowergamma = CodePrinter._print_not_supported
    _print_uppergamma = CodePrinter._print_not_supported
    _print_fresnelc = CodePrinter._print_not_supported
    _print_fresnels = CodePrinter._print_not_supported

for func in _numpy_known_functions:
    setattr(NumPyPrinter, f'_print_{func}', _print_known_func)

for const in _numpy_known_constants:
    setattr(NumPyPrinter, f'_print_{const}', _print_known_const)


_known_functions_scipy_special = {
    'Ei': 'expi',
    'erf': 'erf',
    'erfc': 'erfc',
    'besselj': 'jv',
    'bessely': 'yv',
    'besseli': 'iv',
    'besselk': 'kv',
    'cosm1': 'cosm1',
    'powm1': 'powm1',
    'factorial': 'factorial',
    'gamma': 'gamma',
    'loggamma': 'gammaln',
    'digamma': 'psi',
    'polygamma': 'polygamma',
    'RisingFactorial': 'poch',
    'jacobi': 'eval_jacobi',
    'gegenbauer': 'eval_gegenbauer',
    'chebyshevt': 'eval_chebyt',
    'chebyshevu': 'eval_chebyu',
    'legendre': 'eval_legendre',
    'hermite': 'eval_hermite',
    'laguerre': 'eval_laguerre',
    'assoc_laguerre': 'eval_genlaguerre',
    'beta': 'beta',
    'LambertW' : 'lambertw',
}

_known_constants_scipy_constants = {
    'GoldenRatio': 'golden_ratio',
    'Pi': 'pi',
}
_scipy_known_functions = {k : "scipy.special." + v for k, v in _known_functions_scipy_special.items()}
_scipy_known_constants = {k : "scipy.constants." + v for k, v in _known_constants_scipy_constants.items()}

class SciPyPrinter(NumPyPrinter):

    _kf = {**NumPyPrinter._kf, **_scipy_known_functions}
    _kc = {**NumPyPrinter._kc, **_scipy_known_constants}

    def __init__(self, settings=None):
        super().__init__(settings=settings)
        self.language = "Python with SciPy and NumPy"

    def _print_SparseRepMatrix(self, expr):
        i, j, data = [], [], []
        for (r, c), v in expr.todok().items():
            i.append(r)
            j.append(c)
            data.append(v)

        return "{name}(({data}, ({i}, {j})), shape={shape})".format(
            name=self._module_format('scipy.sparse.coo_matrix'),
            data=data, i=i, j=j, shape=expr.shape
        )

    _print_ImmutableSparseMatrix = _print_SparseRepMatrix

    # SciPy's lpmv has a different order of arguments from assoc_legendre
    def _print_assoc_legendre(self, expr):
        return "{0}({2}, {1}, {3})".format(
            self._module_format('scipy.special.lpmv'),
            self._print(expr.args[0]),
            self._print(expr.args[1]),
            self._print(expr.args[2]))

    def _print_lowergamma(self, expr):
        return "{0}({2})*{1}({2}, {3})".format(
            self._module_format('scipy.special.gamma'),
            self._module_format('scipy.special.gammainc'),
            self._print(expr.args[0]),
            self._print(expr.args[1]))

    def _print_uppergamma(self, expr):
        return "{0}({2})*{1}({2}, {3})".format(
            self._module_format('scipy.special.gamma'),
            self._module_format('scipy.special.gammaincc'),
            self._print(expr.args[0]),
            self._print(expr.args[1]))

    def _print_betainc(self, expr):
        betainc = self._module_format('scipy.special.betainc')
        beta = self._module_format('scipy.special.beta')
        args = [self._print(arg) for arg in expr.args]
        return f"({betainc}({args[0]}, {args[1]}, {args[3]}) - {betainc}({args[0]}, {args[1]}, {args[2]})) \
            * {beta}({args[0]}, {args[1]})"

    def _print_betainc_regularized(self, expr):
        return "{0}({1}, {2}, {4}) - {0}({1}, {2}, {3})".format(
            self._module_format('scipy.special.betainc'),
            self._print(expr.args[0]),
            self._print(expr.args[1]),
            self._print(expr.args[2]),
            self._print(expr.args[3]))

    def _print_fresnels(self, expr):
        return "{}({})[0]".format(
                self._module_format("scipy.special.fresnel"),
                self._print(expr.args[0]))

    def _print_fresnelc(self, expr):
        return "{}({})[1]".format(
                self._module_format("scipy.special.fresnel"),
                self._print(expr.args[0]))

    def _print_airyai(self, expr):
        return "{}({})[0]".format(
                self._module_format("scipy.special.airy"),
                self._print(expr.args[0]))

    def _print_airyaiprime(self, expr):
        return "{}({})[1]".format(
                self._module_format("scipy.special.airy"),
                self._print(expr.args[0]))

    def _print_airybi(self, expr):
        return "{}({})[2]".format(
                self._module_format("scipy.special.airy"),
                self._print(expr.args[0]))

    def _print_airybiprime(self, expr):
        return "{}({})[3]".format(
                self._module_format("scipy.special.airy"),
                self._print(expr.args[0]))

    def _print_bernoulli(self, expr):
        # scipy's bernoulli is inconsistent with SymPy's so rewrite
        return self._print(expr._eval_rewrite_as_zeta(*expr.args))

    def _print_harmonic(self, expr):
        return self._print(expr._eval_rewrite_as_zeta(*expr.args))

    def _print_Integral(self, e):
        integration_vars, limits = _unpack_integral_limits(e)

        if len(limits) == 1:
            # nicer (but not necessary) to prefer quad over nquad for 1D case
            module_str = self._module_format("scipy.integrate.quad")
            limit_str = "%s, %s" % tuple(map(self._print, limits[0]))
        else:
            module_str = self._module_format("scipy.integrate.nquad")
            limit_str = "({})".format(", ".join(
                "(%s, %s)" % tuple(map(self._print, l)) for l in limits))

        return "{}(lambda {}: {}, {})[0]".format(
                module_str,
                ", ".join(map(self._print, integration_vars)),
                self._print(e.args[0]),
                limit_str)

    def _print_Si(self, expr):
        return "{}({})[0]".format(
                self._module_format("scipy.special.sici"),
                self._print(expr.args[0]))

    def _print_Ci(self, expr):
        return "{}({})[1]".format(
                self._module_format("scipy.special.sici"),
                self._print(expr.args[0]))

for func in _scipy_known_functions:
    setattr(SciPyPrinter, f'_print_{func}', _print_known_func)

for const in _scipy_known_constants:
    setattr(SciPyPrinter, f'_print_{const}', _print_known_const)


_cupy_known_functions = {k : "cupy." + v for k, v in _known_functions_numpy.items()}
_cupy_known_constants = {k : "cupy." + v for k, v in _known_constants_numpy.items()}

class CuPyPrinter(NumPyPrinter):
    """
    CuPy printer which handles vectorized piecewise functions,
    logical operators, etc.
    """

    _module = 'cupy'
    _kf = _cupy_known_functions
    _kc = _cupy_known_constants

    def __init__(self, settings=None):
        super().__init__(settings=settings)

for func in _cupy_known_functions:
    setattr(CuPyPrinter, f'_print_{func}', _print_known_func)

for const in _cupy_known_constants:
    setattr(CuPyPrinter, f'_print_{const}', _print_known_const)


_jax_known_functions = {k: 'jax.numpy.' + v for k, v in _known_functions_numpy.items()}
_jax_known_constants = {k: 'jax.numpy.' + v for k, v in _known_constants_numpy.items()}

class JaxPrinter(NumPyPrinter):
    """
    JAX printer which handles vectorized piecewise functions,
    logical operators, etc.
    """
    _module = "jax.numpy"

    _kf = _jax_known_functions
    _kc = _jax_known_constants

    def __init__(self, settings=None):
        super().__init__(settings=settings)
        self.printmethod = '_jaxcode'

    # These need specific override to allow for the lack of "jax.numpy.reduce"
    def _print_And(self, expr):
        "Logical And printer"
        return "{}({}.asarray([{}]), axis=0)".format(
            self._module_format(self._module + ".all"),
            self._module_format(self._module),
            ",".join(self._print(i) for i in expr.args),
        )

    def _print_Or(self, expr):
        "Logical Or printer"
        return "{}({}.asarray([{}]), axis=0)".format(
            self._module_format(self._module + ".any"),
            self._module_format(self._module),
            ",".join(self._print(i) for i in expr.args),
        )

for func in _jax_known_functions:
    setattr(JaxPrinter, f'_print_{func}', _print_known_func)

for const in _jax_known_constants:
    setattr(JaxPrinter, f'_print_{const}', _print_known_const)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\poolmanager.py ===
from __future__ import absolute_import

import collections
import functools
import logging

from ._collections import HTTPHeaderDict, RecentlyUsedContainer
from .connectionpool import HTTPConnectionPool, HTTPSConnectionPool, port_by_scheme
from .exceptions import (
    LocationValueError,
    MaxRetryError,
    ProxySchemeUnknown,
    ProxySchemeUnsupported,
    URLSchemeUnknown,
)
from .packages import six
from .packages.six.moves.urllib.parse import urljoin
from .request import RequestMethods
from .util.proxy import connection_requires_http_tunnel
from .util.retry import Retry
from .util.url import parse_url

__all__ = ["PoolManager", "ProxyManager", "proxy_from_url"]


log = logging.getLogger(__name__)

SSL_KEYWORDS = (
    "key_file",
    "cert_file",
    "cert_reqs",
    "ca_certs",
    "ssl_version",
    "ca_cert_dir",
    "ssl_context",
    "key_password",
    "server_hostname",
)

# All known keyword arguments that could be provided to the pool manager, its
# pools, or the underlying connections. This is used to construct a pool key.
_key_fields = (
    "key_scheme",  # str
    "key_host",  # str
    "key_port",  # int
    "key_timeout",  # int or float or Timeout
    "key_retries",  # int or Retry
    "key_strict",  # bool
    "key_block",  # bool
    "key_source_address",  # str
    "key_key_file",  # str
    "key_key_password",  # str
    "key_cert_file",  # str
    "key_cert_reqs",  # str
    "key_ca_certs",  # str
    "key_ssl_version",  # str
    "key_ca_cert_dir",  # str
    "key_ssl_context",  # instance of ssl.SSLContext or urllib3.util.ssl_.SSLContext
    "key_maxsize",  # int
    "key_headers",  # dict
    "key__proxy",  # parsed proxy url
    "key__proxy_headers",  # dict
    "key__proxy_config",  # class
    "key_socket_options",  # list of (level (int), optname (int), value (int or str)) tuples
    "key__socks_options",  # dict
    "key_assert_hostname",  # bool or string
    "key_assert_fingerprint",  # str
    "key_server_hostname",  # str
)

#: The namedtuple class used to construct keys for the connection pool.
#: All custom key schemes should include the fields in this key at a minimum.
PoolKey = collections.namedtuple("PoolKey", _key_fields)

_proxy_config_fields = ("ssl_context", "use_forwarding_for_https")
ProxyConfig = collections.namedtuple("ProxyConfig", _proxy_config_fields)


def _default_key_normalizer(key_class, request_context):
    """
    Create a pool key out of a request context dictionary.

    According to RFC 3986, both the scheme and host are case-insensitive.
    Therefore, this function normalizes both before constructing the pool
    key for an HTTPS request. If you wish to change this behaviour, provide
    alternate callables to ``key_fn_by_scheme``.

    :param key_class:
        The class to use when constructing the key. This should be a namedtuple
        with the ``scheme`` and ``host`` keys at a minimum.
    :type  key_class: namedtuple
    :param request_context:
        A dictionary-like object that contain the context for a request.
    :type  request_context: dict

    :return: A namedtuple that can be used as a connection pool key.
    :rtype:  PoolKey
    """
    # Since we mutate the dictionary, make a copy first
    context = request_context.copy()
    context["scheme"] = context["scheme"].lower()
    context["host"] = context["host"].lower()

    # These are both dictionaries and need to be transformed into frozensets
    for key in ("headers", "_proxy_headers", "_socks_options"):
        if key in context and context[key] is not None:
            context[key] = frozenset(context[key].items())

    # The socket_options key may be a list and needs to be transformed into a
    # tuple.
    socket_opts = context.get("socket_options")
    if socket_opts is not None:
        context["socket_options"] = tuple(socket_opts)

    # Map the kwargs to the names in the namedtuple - this is necessary since
    # namedtuples can't have fields starting with '_'.
    for key in list(context.keys()):
        context["key_" + key] = context.pop(key)

    # Default to ``None`` for keys missing from the context
    for field in key_class._fields:
        if field not in context:
            context[field] = None

    return key_class(**context)


#: A dictionary that maps a scheme to a callable that creates a pool key.
#: This can be used to alter the way pool keys are constructed, if desired.
#: Each PoolManager makes a copy of this dictionary so they can be configured
#: globally here, or individually on the instance.
key_fn_by_scheme = {
    "http": functools.partial(_default_key_normalizer, PoolKey),
    "https": functools.partial(_default_key_normalizer, PoolKey),
}

pool_classes_by_scheme = {"http": HTTPConnectionPool, "https": HTTPSConnectionPool}


class PoolManager(RequestMethods):
    """
    Allows for arbitrary requests while transparently keeping track of
    necessary connection pools for you.

    :param num_pools:
        Number of connection pools to cache before discarding the least
        recently used pool.

    :param headers:
        Headers to include with all requests, unless other headers are given
        explicitly.

    :param \\**connection_pool_kw:
        Additional parameters are used to create fresh
        :class:`urllib3.connectionpool.ConnectionPool` instances.

    Example::

        >>> manager = PoolManager(num_pools=2)
        >>> r = manager.request('GET', 'http://google.com/')
        >>> r = manager.request('GET', 'http://google.com/mail')
        >>> r = manager.request('GET', 'http://yahoo.com/')
        >>> len(manager.pools)
        2

    """

    proxy = None
    proxy_config = None

    def __init__(self, num_pools=10, headers=None, **connection_pool_kw):
        RequestMethods.__init__(self, headers)
        self.connection_pool_kw = connection_pool_kw
        self.pools = RecentlyUsedContainer(num_pools)

        # Locally set the pool classes and keys so other PoolManagers can
        # override them.
        self.pool_classes_by_scheme = pool_classes_by_scheme
        self.key_fn_by_scheme = key_fn_by_scheme.copy()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()
        # Return False to re-raise any potential exceptions
        return False

    def _new_pool(self, scheme, host, port, request_context=None):
        """
        Create a new :class:`urllib3.connectionpool.ConnectionPool` based on host, port, scheme, and
        any additional pool keyword arguments.

        If ``request_context`` is provided, it is provided as keyword arguments
        to the pool class used. This method is used to actually create the
        connection pools handed out by :meth:`connection_from_url` and
        companion methods. It is intended to be overridden for customization.
        """
        pool_cls = self.pool_classes_by_scheme[scheme]
        if request_context is None:
            request_context = self.connection_pool_kw.copy()

        # Although the context has everything necessary to create the pool,
        # this function has historically only used the scheme, host, and port
        # in the positional args. When an API change is acceptable these can
        # be removed.
        for key in ("scheme", "host", "port"):
            request_context.pop(key, None)

        if scheme == "http":
            for kw in SSL_KEYWORDS:
                request_context.pop(kw, None)

        return pool_cls(host, port, **request_context)

    def clear(self):
        """
        Empty our store of pools and direct them all to close.

        This will not affect in-flight connections, but they will not be
        re-used after completion.
        """
        self.pools.clear()

    def connection_from_host(self, host, port=None, scheme="http", pool_kwargs=None):
        """
        Get a :class:`urllib3.connectionpool.ConnectionPool` based on the host, port, and scheme.

        If ``port`` isn't given, it will be derived from the ``scheme`` using
        ``urllib3.connectionpool.port_by_scheme``. If ``pool_kwargs`` is
        provided, it is merged with the instance's ``connection_pool_kw``
        variable and used to create the new connection pool, if one is
        needed.
        """

        if not host:
            raise LocationValueError("No host specified.")

        request_context = self._merge_pool_kwargs(pool_kwargs)
        request_context["scheme"] = scheme or "http"
        if not port:
            port = port_by_scheme.get(request_context["scheme"].lower(), 80)
        request_context["port"] = port
        request_context["host"] = host

        return self.connection_from_context(request_context)

    def connection_from_context(self, request_context):
        """
        Get a :class:`urllib3.connectionpool.ConnectionPool` based on the request context.

        ``request_context`` must at least contain the ``scheme`` key and its
        value must be a key in ``key_fn_by_scheme`` instance variable.
        """
        scheme = request_context["scheme"].lower()
        pool_key_constructor = self.key_fn_by_scheme.get(scheme)
        if not pool_key_constructor:
            raise URLSchemeUnknown(scheme)
        pool_key = pool_key_constructor(request_context)

        return self.connection_from_pool_key(pool_key, request_context=request_context)

    def connection_from_pool_key(self, pool_key, request_context=None):
        """
        Get a :class:`urllib3.connectionpool.ConnectionPool` based on the provided pool key.

        ``pool_key`` should be a namedtuple that only contains immutable
        objects. At a minimum it must have the ``scheme``, ``host``, and
        ``port`` fields.
        """
        with self.pools.lock:
            # If the scheme, host, or port doesn't match existing open
            # connections, open a new ConnectionPool.
            pool = self.pools.get(pool_key)
            if pool:
                return pool

            # Make a fresh ConnectionPool of the desired type
            scheme = request_context["scheme"]
            host = request_context["host"]
            port = request_context["port"]
            pool = self._new_pool(scheme, host, port, request_context=request_context)
            self.pools[pool_key] = pool

        return pool

    def connection_from_url(self, url, pool_kwargs=None):
        """
        Similar to :func:`urllib3.connectionpool.connection_from_url`.

        If ``pool_kwargs`` is not provided and a new pool needs to be
        constructed, ``self.connection_pool_kw`` is used to initialize
        the :class:`urllib3.connectionpool.ConnectionPool`. If ``pool_kwargs``
        is provided, it is used instead. Note that if a new pool does not
        need to be created for the request, the provided ``pool_kwargs`` are
        not used.
        """
        u = parse_url(url)
        return self.connection_from_host(
            u.host, port=u.port, scheme=u.scheme, pool_kwargs=pool_kwargs
        )

    def _merge_pool_kwargs(self, override):
        """
        Merge a dictionary of override values for self.connection_pool_kw.

        This does not modify self.connection_pool_kw and returns a new dict.
        Any keys in the override dictionary with a value of ``None`` are
        removed from the merged dictionary.
        """
        base_pool_kwargs = self.connection_pool_kw.copy()
        if override:
            for key, value in override.items():
                if value is None:
                    try:
                        del base_pool_kwargs[key]
                    except KeyError:
                        pass
                else:
                    base_pool_kwargs[key] = value
        return base_pool_kwargs

    def _proxy_requires_url_absolute_form(self, parsed_url):
        """
        Indicates if the proxy requires the complete destination URL in the
        request.  Normally this is only needed when not using an HTTP CONNECT
        tunnel.
        """
        if self.proxy is None:
            return False

        return not connection_requires_http_tunnel(
            self.proxy, self.proxy_config, parsed_url.scheme
        )

    def _validate_proxy_scheme_url_selection(self, url_scheme):
        """
        Validates that were not attempting to do TLS in TLS connections on
        Python2 or with unsupported SSL implementations.
        """
        if self.proxy is None or url_scheme != "https":
            return

        if self.proxy.scheme != "https":
            return

        if six.PY2 and not self.proxy_config.use_forwarding_for_https:
            raise ProxySchemeUnsupported(
                "Contacting HTTPS destinations through HTTPS proxies "
                "'via CONNECT tunnels' is not supported in Python 2"
            )

    def urlopen(self, method, url, redirect=True, **kw):
        """
        Same as :meth:`urllib3.HTTPConnectionPool.urlopen`
        with custom cross-host redirect logic and only sends the request-uri
        portion of the ``url``.

        The given ``url`` parameter must be absolute, such that an appropriate
        :class:`urllib3.connectionpool.ConnectionPool` can be chosen for it.
        """
        u = parse_url(url)
        self._validate_proxy_scheme_url_selection(u.scheme)

        conn = self.connection_from_host(u.host, port=u.port, scheme=u.scheme)

        kw["assert_same_host"] = False
        kw["redirect"] = False

        if "headers" not in kw:
            kw["headers"] = self.headers.copy()

        if self._proxy_requires_url_absolute_form(u):
            response = conn.urlopen(method, url, **kw)
        else:
            response = conn.urlopen(method, u.request_uri, **kw)

        redirect_location = redirect and response.get_redirect_location()
        if not redirect_location:
            return response

        # Support relative URLs for redirecting.
        redirect_location = urljoin(url, redirect_location)

        if response.status == 303:
            # Change the method according to RFC 9110, Section 15.4.4.
            method = "GET"
            # And lose the body not to transfer anything sensitive.
            kw["body"] = None
            kw["headers"] = HTTPHeaderDict(kw["headers"])._prepare_for_method_change()

        retries = kw.get("retries")
        if not isinstance(retries, Retry):
            retries = Retry.from_int(retries, redirect=redirect)

        # Strip headers marked as unsafe to forward to the redirected location.
        # Check remove_headers_on_redirect to avoid a potential network call within
        # conn.is_same_host() which may use socket.gethostbyname() in the future.
        if retries.remove_headers_on_redirect and not conn.is_same_host(
            redirect_location
        ):
            headers = list(six.iterkeys(kw["headers"]))
            for header in headers:
                if header.lower() in retries.remove_headers_on_redirect:
                    kw["headers"].pop(header, None)

        try:
            retries = retries.increment(method, url, response=response, _pool=conn)
        except MaxRetryError:
            if retries.raise_on_redirect:
                response.drain_conn()
                raise
            return response

        kw["retries"] = retries
        kw["redirect"] = redirect

        log.info("Redirecting %s -> %s", url, redirect_location)

        response.drain_conn()
        return self.urlopen(method, redirect_location, **kw)


class ProxyManager(PoolManager):
    """
    Behaves just like :class:`PoolManager`, but sends all requests through
    the defined proxy, using the CONNECT method for HTTPS URLs.

    :param proxy_url:
        The URL of the proxy to be used.

    :param proxy_headers:
        A dictionary containing headers that will be sent to the proxy. In case
        of HTTP they are being sent with each request, while in the
        HTTPS/CONNECT case they are sent only once. Could be used for proxy
        authentication.

    :param proxy_ssl_context:
        The proxy SSL context is used to establish the TLS connection to the
        proxy when using HTTPS proxies.

    :param use_forwarding_for_https:
        (Defaults to False) If set to True will forward requests to the HTTPS
        proxy to be made on behalf of the client instead of creating a TLS
        tunnel via the CONNECT method. **Enabling this flag means that request
        and response headers and content will be visible from the HTTPS proxy**
        whereas tunneling keeps request and response headers and content
        private.  IP address, target hostname, SNI, and port are always visible
        to an HTTPS proxy even when this flag is disabled.

    Example:
        >>> proxy = urllib3.ProxyManager('http://localhost:3128/')
        >>> r1 = proxy.request('GET', 'http://google.com/')
        >>> r2 = proxy.request('GET', 'http://httpbin.org/')
        >>> len(proxy.pools)
        1
        >>> r3 = proxy.request('GET', 'https://httpbin.org/')
        >>> r4 = proxy.request('GET', 'https://twitter.com/')
        >>> len(proxy.pools)
        3

    """

    def __init__(
        self,
        proxy_url,
        num_pools=10,
        headers=None,
        proxy_headers=None,
        proxy_ssl_context=None,
        use_forwarding_for_https=False,
        **connection_pool_kw
    ):

        if isinstance(proxy_url, HTTPConnectionPool):
            proxy_url = "%s://%s:%i" % (
                proxy_url.scheme,
                proxy_url.host,
                proxy_url.port,
            )
        proxy = parse_url(proxy_url)

        if proxy.scheme not in ("http", "https"):
            raise ProxySchemeUnknown(proxy.scheme)

        if not proxy.port:
            port = port_by_scheme.get(proxy.scheme, 80)
            proxy = proxy._replace(port=port)

        self.proxy = proxy
        self.proxy_headers = proxy_headers or {}
        self.proxy_ssl_context = proxy_ssl_context
        self.proxy_config = ProxyConfig(proxy_ssl_context, use_forwarding_for_https)

        connection_pool_kw["_proxy"] = self.proxy
        connection_pool_kw["_proxy_headers"] = self.proxy_headers
        connection_pool_kw["_proxy_config"] = self.proxy_config

        super(ProxyManager, self).__init__(num_pools, headers, **connection_pool_kw)

    def connection_from_host(self, host, port=None, scheme="http", pool_kwargs=None):
        if scheme == "https":
            return super(ProxyManager, self).connection_from_host(
                host, port, scheme, pool_kwargs=pool_kwargs
            )

        return super(ProxyManager, self).connection_from_host(
            self.proxy.host, self.proxy.port, self.proxy.scheme, pool_kwargs=pool_kwargs
        )

    def _set_proxy_headers(self, url, headers=None):
        """
        Sets headers needed by proxies: specifically, the Accept and Host
        headers. Only sets headers not provided by the user.
        """
        headers_ = {"Accept": "*/*"}

        netloc = parse_url(url).netloc
        if netloc:
            headers_["Host"] = netloc

        if headers:
            headers_.update(headers)
        return headers_

    def urlopen(self, method, url, redirect=True, **kw):
        "Same as HTTP(S)ConnectionPool.urlopen, ``url`` must be absolute."
        u = parse_url(url)
        if not connection_requires_http_tunnel(self.proxy, self.proxy_config, u.scheme):
            # For connections using HTTP CONNECT, httplib sets the necessary
            # headers on the CONNECT to the proxy. If we're not using CONNECT,
            # we'll definitely need to set 'Host' at the very least.
            headers = kw.get("headers", self.headers)
            kw["headers"] = self._set_proxy_headers(url, headers)

        return super(ProxyManager, self).urlopen(method, url, redirect=redirect, **kw)


def proxy_from_url(url, **kw):
    return ProxyManager(proxy_url=url, **kw)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\bluetooth_emulation.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: BluetoothEmulation (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

class CentralState(enum.Enum):
    '''
    Indicates the various states of Central.
    '''
    ABSENT = "absent"
    POWERED_OFF = "powered-off"
    POWERED_ON = "powered-on"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class GATTOperationType(enum.Enum):
    '''
    Indicates the various types of GATT event.
    '''
    CONNECTION = "connection"
    DISCOVERY = "discovery"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class CharacteristicWriteType(enum.Enum):
    '''
    Indicates the various types of characteristic write.
    '''
    WRITE_DEFAULT_DEPRECATED = "write-default-deprecated"
    WRITE_WITH_RESPONSE = "write-with-response"
    WRITE_WITHOUT_RESPONSE = "write-without-response"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class CharacteristicOperationType(enum.Enum):
    '''
    Indicates the various types of characteristic operation.
    '''
    READ = "read"
    WRITE = "write"
    SUBSCRIBE_TO_NOTIFICATIONS = "subscribe-to-notifications"
    UNSUBSCRIBE_FROM_NOTIFICATIONS = "unsubscribe-from-notifications"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class ManufacturerData:
    '''
    Stores the manufacturer data
    '''
    #: Company identifier
    #: https://bitbucket.org/bluetooth-SIG/public/src/main/assigned_numbers/company_identifiers/company_identifiers.yaml
    #: https://usb.org/developers
    key: int

    #: Manufacturer-specific data
    data: str

    def to_json(self):
        json = dict()
        json['key'] = self.key
        json['data'] = self.data
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            key=int(json['key']),
            data=str(json['data']),
        )


@dataclass
class ScanRecord:
    '''
    Stores the byte data of the advertisement packet sent by a Bluetooth device.
    '''
    name: typing.Optional[str] = None

    uuids: typing.Optional[typing.List[str]] = None

    #: Stores the external appearance description of the device.
    appearance: typing.Optional[int] = None

    #: Stores the transmission power of a broadcasting device.
    tx_power: typing.Optional[int] = None

    #: Key is the company identifier and the value is an array of bytes of
    #: manufacturer specific data.
    manufacturer_data: typing.Optional[typing.List[ManufacturerData]] = None

    def to_json(self):
        json = dict()
        if self.name is not None:
            json['name'] = self.name
        if self.uuids is not None:
            json['uuids'] = [i for i in self.uuids]
        if self.appearance is not None:
            json['appearance'] = self.appearance
        if self.tx_power is not None:
            json['txPower'] = self.tx_power
        if self.manufacturer_data is not None:
            json['manufacturerData'] = [i.to_json() for i in self.manufacturer_data]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']) if 'name' in json else None,
            uuids=[str(i) for i in json['uuids']] if 'uuids' in json else None,
            appearance=int(json['appearance']) if 'appearance' in json else None,
            tx_power=int(json['txPower']) if 'txPower' in json else None,
            manufacturer_data=[ManufacturerData.from_json(i) for i in json['manufacturerData']] if 'manufacturerData' in json else None,
        )


@dataclass
class ScanEntry:
    '''
    Stores the advertisement packet information that is sent by a Bluetooth device.
    '''
    device_address: str

    rssi: int

    scan_record: ScanRecord

    def to_json(self):
        json = dict()
        json['deviceAddress'] = self.device_address
        json['rssi'] = self.rssi
        json['scanRecord'] = self.scan_record.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            device_address=str(json['deviceAddress']),
            rssi=int(json['rssi']),
            scan_record=ScanRecord.from_json(json['scanRecord']),
        )


@dataclass
class CharacteristicProperties:
    '''
    Describes the properties of a characteristic. This follows Bluetooth Core
    Specification BT 4.2 Vol 3 Part G 3.3.1. Characteristic Properties.
    '''
    broadcast: typing.Optional[bool] = None

    read: typing.Optional[bool] = None

    write_without_response: typing.Optional[bool] = None

    write: typing.Optional[bool] = None

    notify: typing.Optional[bool] = None

    indicate: typing.Optional[bool] = None

    authenticated_signed_writes: typing.Optional[bool] = None

    extended_properties: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        if self.broadcast is not None:
            json['broadcast'] = self.broadcast
        if self.read is not None:
            json['read'] = self.read
        if self.write_without_response is not None:
            json['writeWithoutResponse'] = self.write_without_response
        if self.write is not None:
            json['write'] = self.write
        if self.notify is not None:
            json['notify'] = self.notify
        if self.indicate is not None:
            json['indicate'] = self.indicate
        if self.authenticated_signed_writes is not None:
            json['authenticatedSignedWrites'] = self.authenticated_signed_writes
        if self.extended_properties is not None:
            json['extendedProperties'] = self.extended_properties
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            broadcast=bool(json['broadcast']) if 'broadcast' in json else None,
            read=bool(json['read']) if 'read' in json else None,
            write_without_response=bool(json['writeWithoutResponse']) if 'writeWithoutResponse' in json else None,
            write=bool(json['write']) if 'write' in json else None,
            notify=bool(json['notify']) if 'notify' in json else None,
            indicate=bool(json['indicate']) if 'indicate' in json else None,
            authenticated_signed_writes=bool(json['authenticatedSignedWrites']) if 'authenticatedSignedWrites' in json else None,
            extended_properties=bool(json['extendedProperties']) if 'extendedProperties' in json else None,
        )


def enable(
        state: CentralState,
        le_supported: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enable the BluetoothEmulation domain.

    :param state: State of the simulated central.
    :param le_supported: If the simulated central supports low-energy.
    '''
    params: T_JSON_DICT = dict()
    params['state'] = state.to_json()
    params['leSupported'] = le_supported
    cmd_dict: T_JSON_DICT = {
        'method': 'BluetoothEmulation.enable',
        'params': params,
    }
    json = yield cmd_dict


def set_simulated_central_state(
        state: CentralState
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Set the state of the simulated central.

    :param state: State of the simulated central.
    '''
    params: T_JSON_DICT = dict()
    params['state'] = state.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'BluetoothEmulation.setSimulatedCentralState',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disable the BluetoothEmulation domain.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'BluetoothEmulation.disable',
    }
    json = yield cmd_dict


def simulate_preconnected_peripheral(
        address: str,
        name: str,
        manufacturer_data: typing.List[ManufacturerData],
        known_service_uuids: typing.List[str]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Simulates a peripheral with ``address``, ``name`` and ``knownServiceUuids``
    that has already been connected to the system.

    :param address:
    :param name:
    :param manufacturer_data:
    :param known_service_uuids:
    '''
    params: T_JSON_DICT = dict()
    params['address'] = address
    params['name'] = name
    params['manufacturerData'] = [i.to_json() for i in manufacturer_data]
    params['knownServiceUuids'] = [i for i in known_service_uuids]
    cmd_dict: T_JSON_DICT = {
        'method': 'BluetoothEmulation.simulatePreconnectedPeripheral',
        'params': params,
    }
    json = yield cmd_dict


def simulate_advertisement(
        entry: ScanEntry
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Simulates an advertisement packet described in ``entry`` being received by
    the central.

    :param entry:
    '''
    params: T_JSON_DICT = dict()
    params['entry'] = entry.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'BluetoothEmulation.simulateAdvertisement',
        'params': params,
    }
    json = yield cmd_dict


def simulate_gatt_operation_response(
        address: str,
        type_: GATTOperationType,
        code: int
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Simulates the response code from the peripheral with ``address`` for a
    GATT operation of ``type``. The ``code`` value follows the HCI Error Codes from
    Bluetooth Core Specification Vol 2 Part D 1.3 List Of Error Codes.

    :param address:
    :param type_:
    :param code:
    '''
    params: T_JSON_DICT = dict()
    params['address'] = address
    params['type'] = type_.to_json()
    params['code'] = code
    cmd_dict: T_JSON_DICT = {
        'method': 'BluetoothEmulation.simulateGATTOperationResponse',
        'params': params,
    }
    json = yield cmd_dict


def simulate_characteristic_operation_response(
        characteristic_id: str,
        type_: CharacteristicOperationType,
        code: int,
        data: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Simulates the response from the characteristic with ``characteristicId`` for a
    characteristic operation of ``type``. The ``code`` value follows the Error
    Codes from Bluetooth Core Specification Vol 3 Part F 3.4.1.1 Error Response.
    The ``data`` is expected to exist when simulating a successful read operation
    response.

    :param characteristic_id:
    :param type_:
    :param code:
    :param data: *(Optional)*
    '''
    params: T_JSON_DICT = dict()
    params['characteristicId'] = characteristic_id
    params['type'] = type_.to_json()
    params['code'] = code
    if data is not None:
        params['data'] = data
    cmd_dict: T_JSON_DICT = {
        'method': 'BluetoothEmulation.simulateCharacteristicOperationResponse',
        'params': params,
    }
    json = yield cmd_dict


def add_service(
        address: str,
        service_uuid: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,str]:
    '''
    Adds a service with ``serviceUuid`` to the peripheral with ``address``.

    :param address:
    :param service_uuid:
    :returns: An identifier that uniquely represents this service.
    '''
    params: T_JSON_DICT = dict()
    params['address'] = address
    params['serviceUuid'] = service_uuid
    cmd_dict: T_JSON_DICT = {
        'method': 'BluetoothEmulation.addService',
        'params': params,
    }
    json = yield cmd_dict
    return str(json['serviceId'])


def remove_service(
        service_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes the service respresented by ``serviceId`` from the simulated central.

    :param service_id:
    '''
    params: T_JSON_DICT = dict()
    params['serviceId'] = service_id
    cmd_dict: T_JSON_DICT = {
        'method': 'BluetoothEmulation.removeService',
        'params': params,
    }
    json = yield cmd_dict


def add_characteristic(
        service_id: str,
        characteristic_uuid: str,
        properties: CharacteristicProperties
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,str]:
    '''
    Adds a characteristic with ``characteristicUuid`` and ``properties`` to the
    service represented by ``serviceId``.

    :param service_id:
    :param characteristic_uuid:
    :param properties:
    :returns: An identifier that uniquely represents this characteristic.
    '''
    params: T_JSON_DICT = dict()
    params['serviceId'] = service_id
    params['characteristicUuid'] = characteristic_uuid
    params['properties'] = properties.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'BluetoothEmulation.addCharacteristic',
        'params': params,
    }
    json = yield cmd_dict
    return str(json['characteristicId'])


def remove_characteristic(
        characteristic_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes the characteristic respresented by ``characteristicId`` from the
    simulated central.

    :param characteristic_id:
    '''
    params: T_JSON_DICT = dict()
    params['characteristicId'] = characteristic_id
    cmd_dict: T_JSON_DICT = {
        'method': 'BluetoothEmulation.removeCharacteristic',
        'params': params,
    }
    json = yield cmd_dict


def add_descriptor(
        characteristic_id: str,
        descriptor_uuid: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,str]:
    '''
    Adds a descriptor with ``descriptorUuid`` to the characteristic respresented
    by ``characteristicId``.

    :param characteristic_id:
    :param descriptor_uuid:
    :returns: An identifier that uniquely represents this descriptor.
    '''
    params: T_JSON_DICT = dict()
    params['characteristicId'] = characteristic_id
    params['descriptorUuid'] = descriptor_uuid
    cmd_dict: T_JSON_DICT = {
        'method': 'BluetoothEmulation.addDescriptor',
        'params': params,
    }
    json = yield cmd_dict
    return str(json['descriptorId'])


def remove_descriptor(
        descriptor_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes the descriptor with ``descriptorId`` from the simulated central.

    :param descriptor_id:
    '''
    params: T_JSON_DICT = dict()
    params['descriptorId'] = descriptor_id
    cmd_dict: T_JSON_DICT = {
        'method': 'BluetoothEmulation.removeDescriptor',
        'params': params,
    }
    json = yield cmd_dict


@event_class('BluetoothEmulation.gattOperationReceived')
@dataclass
class GattOperationReceived:
    '''
    Event for when a GATT operation of ``type`` to the peripheral with ``address``
    happened.
    '''
    address: str
    type_: GATTOperationType

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> GattOperationReceived:
        return cls(
            address=str(json['address']),
            type_=GATTOperationType.from_json(json['type'])
        )


@event_class('BluetoothEmulation.characteristicOperationReceived')
@dataclass
class CharacteristicOperationReceived:
    '''
    Event for when a characteristic operation of ``type`` to the characteristic
    respresented by ``characteristicId`` happened. ``data`` and ``writeType`` is
    expected to exist when ``type`` is write.
    '''
    characteristic_id: str
    type_: CharacteristicOperationType
    data: typing.Optional[str]
    write_type: typing.Optional[CharacteristicWriteType]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CharacteristicOperationReceived:
        return cls(
            characteristic_id=str(json['characteristicId']),
            type_=CharacteristicOperationType.from_json(json['type']),
            data=str(json['data']) if 'data' in json else None,
            write_type=CharacteristicWriteType.from_json(json['writeType']) if 'writeType' in json else None
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\matrices\normalforms.py ===
'''Functions returning normal forms of matrices'''

from collections import defaultdict

from .domainmatrix import DomainMatrix
from .exceptions import DMDomainError, DMShapeError
from sympy.ntheory.modular import symmetric_residue
from sympy.polys.domains import QQ, ZZ


# TODO (future work):
#  There are faster algorithms for Smith and Hermite normal forms, which
#  we should implement. See e.g. the Kannan-Bachem algorithm:
#  <https://www.researchgate.net/publication/220617516_Polynomial_Algorithms_for_Computing_the_Smith_and_Hermite_Normal_Forms_of_an_Integer_Matrix>


def smith_normal_form(m):
    '''
    Return the Smith Normal Form of a matrix `m` over the ring `domain`.
    This will only work if the ring is a principal ideal domain.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.matrices import DomainMatrix
    >>> from sympy.polys.matrices.normalforms import smith_normal_form
    >>> m = DomainMatrix([[ZZ(12), ZZ(6), ZZ(4)],
    ...                   [ZZ(3), ZZ(9), ZZ(6)],
    ...                   [ZZ(2), ZZ(16), ZZ(14)]], (3, 3), ZZ)
    >>> print(smith_normal_form(m).to_Matrix())
    Matrix([[1, 0, 0], [0, 10, 0], [0, 0, 30]])

    '''
    invs = invariant_factors(m)
    smf = DomainMatrix.diag(invs, m.domain, m.shape)
    return smf


def is_smith_normal_form(m):
    '''
    Checks that the matrix is in Smith Normal Form
    '''
    domain = m.domain
    shape = m.shape
    zero = domain.zero
    m = m.to_list()

    for i in range(shape[0]):
        for j in range(shape[1]):
            if i == j:
                continue
            if not m[i][j] == zero:
                return False

    upper = min(shape[0], shape[1])
    for i in range(1, upper):
        if m[i-1][i-1] == zero:
            if m[i][i] != zero:
                return False
        else:
            r = domain.div(m[i][i], m[i-1][i-1])[1]
            if r != zero:
                return False

    return True


def add_columns(m, i, j, a, b, c, d):
    # replace m[:, i] by a*m[:, i] + b*m[:, j]
    # and m[:, j] by c*m[:, i] + d*m[:, j]
    for k in range(len(m)):
        e = m[k][i]
        m[k][i] = a*e + b*m[k][j]
        m[k][j] = c*e + d*m[k][j]


def invariant_factors(m):
    '''
    Return the tuple of abelian invariants for a matrix `m`
    (as in the Smith-Normal form)

    References
    ==========

    [1] https://en.wikipedia.org/wiki/Smith_normal_form#Algorithm
    [2] https://web.archive.org/web/20200331143852/https://sierra.nmsu.edu/morandi/notes/SmithNormalForm.pdf

    '''
    domain = m.domain
    shape = m.shape
    m = m.to_list()
    return _smith_normal_decomp(m, domain, shape=shape, full=False)


def smith_normal_decomp(m):
    '''
    Return the Smith-Normal form decomposition of matrix `m`.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.matrices import DomainMatrix
    >>> from sympy.polys.matrices.normalforms import smith_normal_decomp
    >>> m = DomainMatrix([[ZZ(12), ZZ(6), ZZ(4)],
    ...                   [ZZ(3), ZZ(9), ZZ(6)],
    ...                   [ZZ(2), ZZ(16), ZZ(14)]], (3, 3), ZZ)
    >>> a, s, t = smith_normal_decomp(m)
    >>> assert a == s * m * t
    '''
    domain = m.domain
    rows, cols = shape = m.shape
    m = m.to_list()

    invs, s, t = _smith_normal_decomp(m, domain, shape=shape, full=True)
    smf = DomainMatrix.diag(invs, domain, shape).to_dense()

    s = DomainMatrix(s, domain=domain, shape=(rows, rows))
    t = DomainMatrix(t, domain=domain, shape=(cols, cols))
    return smf, s, t


def _smith_normal_decomp(m, domain, shape, full):
    '''
    Return the tuple of abelian invariants for a matrix `m`
    (as in the Smith-Normal form). If `full=True` then invertible matrices
    ``s, t`` such that the product ``s, m, t`` is the Smith Normal Form
    are also returned.
    '''
    if not domain.is_PID:
        msg = f"The matrix entries must be over a principal ideal domain, but got {domain}"
        raise ValueError(msg)

    rows, cols = shape
    zero = domain.zero
    one = domain.one

    def eye(n):
        return [[one if i == j else zero for i in range(n)] for j in range(n)]

    if 0 in shape:
        if full:
            return (), eye(rows), eye(cols)
        else:
            return ()

    if full:
        s = eye(rows)
        t = eye(cols)

    def add_rows(m, i, j, a, b, c, d):
        # replace m[i, :] by a*m[i, :] + b*m[j, :]
        # and m[j, :] by c*m[i, :] + d*m[j, :]
        for k in range(len(m[0])):
            e = m[i][k]
            m[i][k] = a*e + b*m[j][k]
            m[j][k] = c*e + d*m[j][k]

    def clear_column():
        # make m[1:, 0] zero by row and column operations
        pivot = m[0][0]
        for j in range(1, rows):
            if m[j][0] == zero:
                continue
            d, r = domain.div(m[j][0], pivot)
            if r == zero:
                add_rows(m, 0, j, 1, 0, -d, 1)
                if full:
                    add_rows(s, 0, j, 1, 0, -d, 1)
            else:
                a, b, g = domain.gcdex(pivot, m[j][0])
                d_0 = domain.exquo(m[j][0], g)
                d_j = domain.exquo(pivot, g)
                add_rows(m, 0, j, a, b, d_0, -d_j)
                if full:
                    add_rows(s, 0, j, a, b, d_0, -d_j)
                pivot = g

    def clear_row():
        # make m[0, 1:] zero by row and column operations
        pivot = m[0][0]
        for j in range(1, cols):
            if m[0][j] == zero:
                continue
            d, r = domain.div(m[0][j], pivot)
            if r == zero:
                add_columns(m, 0, j, 1, 0, -d, 1)
                if full:
                    add_columns(t, 0, j, 1, 0, -d, 1)
            else:
                a, b, g = domain.gcdex(pivot, m[0][j])
                d_0 = domain.exquo(m[0][j], g)
                d_j = domain.exquo(pivot, g)
                add_columns(m, 0, j, a, b, d_0, -d_j)
                if full:
                    add_columns(t, 0, j, a, b, d_0, -d_j)
                pivot = g

    # permute the rows and columns until m[0,0] is non-zero if possible
    ind = [i for i in range(rows) if m[i][0] != zero]
    if ind and ind[0] != zero:
        m[0], m[ind[0]] = m[ind[0]], m[0]
        if full:
            s[0], s[ind[0]] = s[ind[0]], s[0]
    else:
        ind = [j for j in range(cols) if m[0][j] != zero]
        if ind and ind[0] != zero:
            for row in m:
                row[0], row[ind[0]] = row[ind[0]], row[0]
            if full:
                for row in t:
                    row[0], row[ind[0]] = row[ind[0]], row[0]

    # make the first row and column except m[0,0] zero
    while (any(m[0][i] != zero for i in range(1,cols)) or
           any(m[i][0] != zero for i in range(1,rows))):
        clear_column()
        clear_row()

    def to_domain_matrix(m):
        return DomainMatrix(m, shape=(len(m), len(m[0])), domain=domain)

    if m[0][0] != 0:
        c = domain.canonical_unit(m[0][0])
        if domain.is_Field:
            c = 1 / m[0][0]
        if c != domain.one:
            m[0][0] *= c
            if full:
                s[0] = [elem * c for elem in s[0]]

    if 1 in shape:
        invs = ()
    else:
        lower_right = [r[1:] for r in m[1:]]
        ret = _smith_normal_decomp(lower_right, domain,
                shape=(rows - 1, cols - 1), full=full)
        if full:
            invs, s_small, t_small = ret
            s2 = [[1] + [0]*(rows-1)] + [[0] + row for row in s_small]
            t2 = [[1] + [0]*(cols-1)] + [[0] + row for row in t_small]
            s, s2, t, t2 = list(map(to_domain_matrix, [s, s2, t, t2]))
            s = s2 * s
            t = t * t2
            s = s.to_list()
            t = t.to_list()
        else:
            invs = ret

    if m[0][0]:
        result = [m[0][0]]
        result.extend(invs)
        # in case m[0] doesn't divide the invariants of the rest of the matrix
        for i in range(len(result)-1):
            a, b = result[i], result[i+1]
            if b and domain.div(b, a)[1] != zero:
                if full:
                    x, y, d = domain.gcdex(a, b)
                else:
                    d = domain.gcd(a, b)

                alpha = domain.div(a, d)[0]
                if full:
                    beta = domain.div(b, d)[0]
                    add_rows(s, i, i + 1, 1, 0, x, 1)
                    add_columns(t, i, i + 1, 1, y, 0, 1)
                    add_rows(s, i, i + 1, 1, -alpha, 0, 1)
                    add_columns(t, i, i + 1, 1, 0, -beta, 1)
                    add_rows(s, i, i + 1, 0, 1, -1, 0)

                result[i+1] = b * alpha
                result[i] = d
            else:
                break
    else:
        if full:
            if rows > 1:
                s = s[1:] + [s[0]]
            if cols > 1:
                t = [row[1:] + [row[0]] for row in t]
        result = invs + (m[0][0],)

    if full:
        return tuple(result), s, t
    else:
        return tuple(result)


def _gcdex(a, b):
    r"""
    This supports the functions that compute Hermite Normal Form.

    Explanation
    ===========

    Let x, y be the coefficients returned by the extended Euclidean
    Algorithm, so that x*a + y*b = g. In the algorithms for computing HNF,
    it is critical that x, y not only satisfy the condition of being small
    in magnitude -- namely that |x| <= |b|/g, |y| <- |a|/g -- but also that
    y == 0 when a | b.

    """
    x, y, g = ZZ.gcdex(a, b)
    if a != 0 and b % a == 0:
        y = 0
        x = -1 if a < 0 else 1
    return x, y, g


def _hermite_normal_form(A):
    r"""
    Compute the Hermite Normal Form of DomainMatrix *A* over :ref:`ZZ`.

    Parameters
    ==========

    A : :py:class:`~.DomainMatrix` over domain :ref:`ZZ`.

    Returns
    =======

    :py:class:`~.DomainMatrix`
        The HNF of matrix *A*.

    Raises
    ======

    DMDomainError
        If the domain of the matrix is not :ref:`ZZ`.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Algorithm 2.4.5.)

    """
    if not A.domain.is_ZZ:
        raise DMDomainError('Matrix must be over domain ZZ.')
    # We work one row at a time, starting from the bottom row, and working our
    # way up.
    m, n = A.shape
    A = A.to_ddm().copy()
    # Our goal is to put pivot entries in the rightmost columns.
    # Invariant: Before processing each row, k should be the index of the
    # leftmost column in which we have so far put a pivot.
    k = n
    for i in range(m - 1, -1, -1):
        if k == 0:
            # This case can arise when n < m and we've already found n pivots.
            # We don't need to consider any more rows, because this is already
            # the maximum possible number of pivots.
            break
        k -= 1
        # k now points to the column in which we want to put a pivot.
        # We want zeros in all entries to the left of the pivot column.
        for j in range(k - 1, -1, -1):
            if A[i][j] != 0:
                # Replace cols j, k by lin combs of these cols such that, in row i,
                # col j has 0, while col k has the gcd of their row i entries. Note
                # that this ensures a nonzero entry in col k.
                u, v, d = _gcdex(A[i][k], A[i][j])
                r, s = A[i][k] // d, A[i][j] // d
                add_columns(A, k, j, u, v, -s, r)
        b = A[i][k]
        # Do not want the pivot entry to be negative.
        if b < 0:
            add_columns(A, k, k, -1, 0, -1, 0)
            b = -b
        # The pivot entry will be 0 iff the row was 0 from the pivot col all the
        # way to the left. In this case, we are still working on the same pivot
        # col for the next row. Therefore:
        if b == 0:
            k += 1
        # If the pivot entry is nonzero, then we want to reduce all entries to its
        # right in the sense of the division algorithm, i.e. make them all remainders
        # w.r.t. the pivot as divisor.
        else:
            for j in range(k + 1, n):
                q = A[i][j] // b
                add_columns(A, j, k, 1, -q, 0, 1)
    # Finally, the HNF consists of those columns of A in which we succeeded in making
    # a nonzero pivot.
    return DomainMatrix.from_rep(A.to_dfm_or_ddm())[:, k:]


def _hermite_normal_form_modulo_D(A, D):
    r"""
    Perform the mod *D* Hermite Normal Form reduction algorithm on
    :py:class:`~.DomainMatrix` *A*.

    Explanation
    ===========

    If *A* is an $m \times n$ matrix of rank $m$, having Hermite Normal Form
    $W$, and if *D* is any positive integer known in advance to be a multiple
    of $\det(W)$, then the HNF of *A* can be computed by an algorithm that
    works mod *D* in order to prevent coefficient explosion.

    Parameters
    ==========

    A : :py:class:`~.DomainMatrix` over :ref:`ZZ`
        $m \times n$ matrix, having rank $m$.
    D : :ref:`ZZ`
        Positive integer, known to be a multiple of the determinant of the
        HNF of *A*.

    Returns
    =======

    :py:class:`~.DomainMatrix`
        The HNF of matrix *A*.

    Raises
    ======

    DMDomainError
        If the domain of the matrix is not :ref:`ZZ`, or
        if *D* is given but is not in :ref:`ZZ`.

    DMShapeError
        If the matrix has more rows than columns.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Algorithm 2.4.8.)

    """
    if not A.domain.is_ZZ:
        raise DMDomainError('Matrix must be over domain ZZ.')
    if not ZZ.of_type(D) or D < 1:
        raise DMDomainError('Modulus D must be positive element of domain ZZ.')

    def add_columns_mod_R(m, R, i, j, a, b, c, d):
        # replace m[:, i] by (a*m[:, i] + b*m[:, j]) % R
        # and m[:, j] by (c*m[:, i] + d*m[:, j]) % R
        for k in range(len(m)):
            e = m[k][i]
            m[k][i] = symmetric_residue((a * e + b * m[k][j]) % R, R)
            m[k][j] = symmetric_residue((c * e + d * m[k][j]) % R, R)

    W = defaultdict(dict)

    m, n = A.shape
    if n < m:
        raise DMShapeError('Matrix must have at least as many columns as rows.')
    A = A.to_list()
    k = n
    R = D
    for i in range(m - 1, -1, -1):
        k -= 1
        for j in range(k - 1, -1, -1):
            if A[i][j] != 0:
                u, v, d = _gcdex(A[i][k], A[i][j])
                r, s = A[i][k] // d, A[i][j] // d
                add_columns_mod_R(A, R, k, j, u, v, -s, r)
        b = A[i][k]
        if b == 0:
            A[i][k] = b = R
        u, v, d = _gcdex(b, R)
        for ii in range(m):
            W[ii][i] = u*A[ii][k] % R
        if W[i][i] == 0:
            W[i][i] = R
        for j in range(i + 1, m):
            q = W[i][j] // W[i][i]
            add_columns(W, j, i, 1, -q, 0, 1)
        R //= d
    return DomainMatrix(W, (m, m), ZZ).to_dense()


def hermite_normal_form(A, *, D=None, check_rank=False):
    r"""
    Compute the Hermite Normal Form of :py:class:`~.DomainMatrix` *A* over
    :ref:`ZZ`.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.matrices import DomainMatrix
    >>> from sympy.polys.matrices.normalforms import hermite_normal_form
    >>> m = DomainMatrix([[ZZ(12), ZZ(6), ZZ(4)],
    ...                   [ZZ(3), ZZ(9), ZZ(6)],
    ...                   [ZZ(2), ZZ(16), ZZ(14)]], (3, 3), ZZ)
    >>> print(hermite_normal_form(m).to_Matrix())
    Matrix([[10, 0, 2], [0, 15, 3], [0, 0, 2]])

    Parameters
    ==========

    A : $m \times n$ ``DomainMatrix`` over :ref:`ZZ`.

    D : :ref:`ZZ`, optional
        Let $W$ be the HNF of *A*. If known in advance, a positive integer *D*
        being any multiple of $\det(W)$ may be provided. In this case, if *A*
        also has rank $m$, then we may use an alternative algorithm that works
        mod *D* in order to prevent coefficient explosion.

    check_rank : boolean, optional (default=False)
        The basic assumption is that, if you pass a value for *D*, then
        you already believe that *A* has rank $m$, so we do not waste time
        checking it for you. If you do want this to be checked (and the
        ordinary, non-modulo *D* algorithm to be used if the check fails), then
        set *check_rank* to ``True``.

    Returns
    =======

    :py:class:`~.DomainMatrix`
        The HNF of matrix *A*.

    Raises
    ======

    DMDomainError
        If the domain of the matrix is not :ref:`ZZ`, or
        if *D* is given but is not in :ref:`ZZ`.

    DMShapeError
        If the mod *D* algorithm is used but the matrix has more rows than
        columns.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Algorithms 2.4.5 and 2.4.8.)

    """
    if not A.domain.is_ZZ:
        raise DMDomainError('Matrix must be over domain ZZ.')
    if D is not None and (not check_rank or A.convert_to(QQ).rank() == A.shape[0]):
        return _hermite_normal_form_modulo_D(A, D)
    else:
        return _hermite_normal_form(A)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\postgresql\named_types.py ===
# dialects/postgresql/named_types.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors
from __future__ import annotations

from types import ModuleType
from typing import Any
from typing import Dict
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING
from typing import Union

from ... import schema
from ... import util
from ...sql import coercions
from ...sql import elements
from ...sql import roles
from ...sql import sqltypes
from ...sql import type_api
from ...sql.base import _NoArg
from ...sql.ddl import InvokeCreateDDLBase
from ...sql.ddl import InvokeDropDDLBase

if TYPE_CHECKING:
    from ...sql._typing import _CreateDropBind
    from ...sql._typing import _TypeEngineArgument


class NamedType(schema.SchemaVisitable, sqltypes.TypeEngine):
    """Base for named types."""

    __abstract__ = True
    DDLGenerator: Type[NamedTypeGenerator]
    DDLDropper: Type[NamedTypeDropper]
    create_type: bool

    def create(
        self, bind: _CreateDropBind, checkfirst: bool = True, **kw: Any
    ) -> None:
        """Emit ``CREATE`` DDL for this type.

        :param bind: a connectable :class:`_engine.Engine`,
         :class:`_engine.Connection`, or similar object to emit
         SQL.
        :param checkfirst: if ``True``, a query against
         the PG catalog will be first performed to see
         if the type does not exist already before
         creating.

        """
        bind._run_ddl_visitor(self.DDLGenerator, self, checkfirst=checkfirst)

    def drop(
        self, bind: _CreateDropBind, checkfirst: bool = True, **kw: Any
    ) -> None:
        """Emit ``DROP`` DDL for this type.

        :param bind: a connectable :class:`_engine.Engine`,
         :class:`_engine.Connection`, or similar object to emit
         SQL.
        :param checkfirst: if ``True``, a query against
         the PG catalog will be first performed to see
         if the type actually exists before dropping.

        """
        bind._run_ddl_visitor(self.DDLDropper, self, checkfirst=checkfirst)

    def _check_for_name_in_memos(
        self, checkfirst: bool, kw: Dict[str, Any]
    ) -> bool:
        """Look in the 'ddl runner' for 'memos', then
        note our name in that collection.

        This to ensure a particular named type is operated
        upon only once within any kind of create/drop
        sequence without relying upon "checkfirst".

        """
        if not self.create_type:
            return True
        if "_ddl_runner" in kw:
            ddl_runner = kw["_ddl_runner"]
            type_name = f"pg_{self.__visit_name__}"
            if type_name in ddl_runner.memo:
                existing = ddl_runner.memo[type_name]
            else:
                existing = ddl_runner.memo[type_name] = set()
            present = (self.schema, self.name) in existing
            existing.add((self.schema, self.name))
            return present
        else:
            return False

    def _on_table_create(
        self,
        target: Any,
        bind: _CreateDropBind,
        checkfirst: bool = False,
        **kw: Any,
    ) -> None:
        if (
            checkfirst
            or (
                not self.metadata
                and not kw.get("_is_metadata_operation", False)
            )
        ) and not self._check_for_name_in_memos(checkfirst, kw):
            self.create(bind=bind, checkfirst=checkfirst)

    def _on_table_drop(
        self,
        target: Any,
        bind: _CreateDropBind,
        checkfirst: bool = False,
        **kw: Any,
    ) -> None:
        if (
            not self.metadata
            and not kw.get("_is_metadata_operation", False)
            and not self._check_for_name_in_memos(checkfirst, kw)
        ):
            self.drop(bind=bind, checkfirst=checkfirst)

    def _on_metadata_create(
        self,
        target: Any,
        bind: _CreateDropBind,
        checkfirst: bool = False,
        **kw: Any,
    ) -> None:
        if not self._check_for_name_in_memos(checkfirst, kw):
            self.create(bind=bind, checkfirst=checkfirst)

    def _on_metadata_drop(
        self,
        target: Any,
        bind: _CreateDropBind,
        checkfirst: bool = False,
        **kw: Any,
    ) -> None:
        if not self._check_for_name_in_memos(checkfirst, kw):
            self.drop(bind=bind, checkfirst=checkfirst)


class NamedTypeGenerator(InvokeCreateDDLBase):
    def __init__(self, dialect, connection, checkfirst=False, **kwargs):
        super().__init__(connection, **kwargs)
        self.checkfirst = checkfirst

    def _can_create_type(self, type_):
        if not self.checkfirst:
            return True

        effective_schema = self.connection.schema_for_object(type_)
        return not self.connection.dialect.has_type(
            self.connection, type_.name, schema=effective_schema
        )


class NamedTypeDropper(InvokeDropDDLBase):
    def __init__(self, dialect, connection, checkfirst=False, **kwargs):
        super().__init__(connection, **kwargs)
        self.checkfirst = checkfirst

    def _can_drop_type(self, type_):
        if not self.checkfirst:
            return True

        effective_schema = self.connection.schema_for_object(type_)
        return self.connection.dialect.has_type(
            self.connection, type_.name, schema=effective_schema
        )


class EnumGenerator(NamedTypeGenerator):
    def visit_enum(self, enum):
        if not self._can_create_type(enum):
            return

        with self.with_ddl_events(enum):
            self.connection.execute(CreateEnumType(enum))


class EnumDropper(NamedTypeDropper):
    def visit_enum(self, enum):
        if not self._can_drop_type(enum):
            return

        with self.with_ddl_events(enum):
            self.connection.execute(DropEnumType(enum))


class ENUM(NamedType, type_api.NativeForEmulated, sqltypes.Enum):
    """PostgreSQL ENUM type.

    This is a subclass of :class:`_types.Enum` which includes
    support for PG's ``CREATE TYPE`` and ``DROP TYPE``.

    When the builtin type :class:`_types.Enum` is used and the
    :paramref:`.Enum.native_enum` flag is left at its default of
    True, the PostgreSQL backend will use a :class:`_postgresql.ENUM`
    type as the implementation, so the special create/drop rules
    will be used.

    The create/drop behavior of ENUM is necessarily intricate, due to the
    awkward relationship the ENUM type has in relationship to the
    parent table, in that it may be "owned" by just a single table, or
    may be shared among many tables.

    When using :class:`_types.Enum` or :class:`_postgresql.ENUM`
    in an "inline" fashion, the ``CREATE TYPE`` and ``DROP TYPE`` is emitted
    corresponding to when the :meth:`_schema.Table.create` and
    :meth:`_schema.Table.drop`
    methods are called::

        table = Table(
            "sometable",
            metadata,
            Column("some_enum", ENUM("a", "b", "c", name="myenum")),
        )

        table.create(engine)  # will emit CREATE ENUM and CREATE TABLE
        table.drop(engine)  # will emit DROP TABLE and DROP ENUM

    To use a common enumerated type between multiple tables, the best
    practice is to declare the :class:`_types.Enum` or
    :class:`_postgresql.ENUM` independently, and associate it with the
    :class:`_schema.MetaData` object itself::

        my_enum = ENUM("a", "b", "c", name="myenum", metadata=metadata)

        t1 = Table("sometable_one", metadata, Column("some_enum", myenum))

        t2 = Table("sometable_two", metadata, Column("some_enum", myenum))

    When this pattern is used, care must still be taken at the level
    of individual table creates.  Emitting CREATE TABLE without also
    specifying ``checkfirst=True`` will still cause issues::

        t1.create(engine)  # will fail: no such type 'myenum'

    If we specify ``checkfirst=True``, the individual table-level create
    operation will check for the ``ENUM`` and create if not exists::

        # will check if enum exists, and emit CREATE TYPE if not
        t1.create(engine, checkfirst=True)

    When using a metadata-level ENUM type, the type will always be created
    and dropped if either the metadata-wide create/drop is called::

        metadata.create_all(engine)  # will emit CREATE TYPE
        metadata.drop_all(engine)  # will emit DROP TYPE

    The type can also be created and dropped directly::

        my_enum.create(engine)
        my_enum.drop(engine)

    """

    native_enum = True
    DDLGenerator = EnumGenerator
    DDLDropper = EnumDropper

    def __init__(
        self,
        *enums,
        name: Union[str, _NoArg, None] = _NoArg.NO_ARG,
        create_type: bool = True,
        **kw,
    ):
        """Construct an :class:`_postgresql.ENUM`.

        Arguments are the same as that of
        :class:`_types.Enum`, but also including
        the following parameters.

        :param create_type: Defaults to True.
         Indicates that ``CREATE TYPE`` should be
         emitted, after optionally checking for the
         presence of the type, when the parent
         table is being created; and additionally
         that ``DROP TYPE`` is called when the table
         is dropped.    When ``False``, no check
         will be performed and no ``CREATE TYPE``
         or ``DROP TYPE`` is emitted, unless
         :meth:`~.postgresql.ENUM.create`
         or :meth:`~.postgresql.ENUM.drop`
         are called directly.
         Setting to ``False`` is helpful
         when invoking a creation scheme to a SQL file
         without access to the actual database -
         the :meth:`~.postgresql.ENUM.create` and
         :meth:`~.postgresql.ENUM.drop` methods can
         be used to emit SQL to a target bind.

        """
        native_enum = kw.pop("native_enum", None)
        if native_enum is False:
            util.warn(
                "the native_enum flag does not apply to the "
                "sqlalchemy.dialects.postgresql.ENUM datatype; this type "
                "always refers to ENUM.   Use sqlalchemy.types.Enum for "
                "non-native enum."
            )
        self.create_type = create_type
        if name is not _NoArg.NO_ARG:
            kw["name"] = name
        super().__init__(*enums, **kw)

    def coerce_compared_value(self, op, value):
        super_coerced_type = super().coerce_compared_value(op, value)
        if (
            super_coerced_type._type_affinity
            is type_api.STRINGTYPE._type_affinity
        ):
            return self
        else:
            return super_coerced_type

    @classmethod
    def __test_init__(cls):
        return cls(name="name")

    @classmethod
    def adapt_emulated_to_native(cls, impl, **kw):
        """Produce a PostgreSQL native :class:`_postgresql.ENUM` from plain
        :class:`.Enum`.

        """
        kw.setdefault("validate_strings", impl.validate_strings)
        kw.setdefault("name", impl.name)
        kw.setdefault("schema", impl.schema)
        kw.setdefault("inherit_schema", impl.inherit_schema)
        kw.setdefault("metadata", impl.metadata)
        kw.setdefault("_create_events", False)
        kw.setdefault("values_callable", impl.values_callable)
        kw.setdefault("omit_aliases", impl._omit_aliases)
        kw.setdefault("_adapted_from", impl)
        if type_api._is_native_for_emulated(impl.__class__):
            kw.setdefault("create_type", impl.create_type)

        return cls(**kw)

    def create(self, bind: _CreateDropBind, checkfirst: bool = True) -> None:
        """Emit ``CREATE TYPE`` for this
        :class:`_postgresql.ENUM`.

        If the underlying dialect does not support
        PostgreSQL CREATE TYPE, no action is taken.

        :param bind: a connectable :class:`_engine.Engine`,
         :class:`_engine.Connection`, or similar object to emit
         SQL.
        :param checkfirst: if ``True``, a query against
         the PG catalog will be first performed to see
         if the type does not exist already before
         creating.

        """
        if not bind.dialect.supports_native_enum:
            return

        super().create(bind, checkfirst=checkfirst)

    def drop(self, bind: _CreateDropBind, checkfirst: bool = True) -> None:
        """Emit ``DROP TYPE`` for this
        :class:`_postgresql.ENUM`.

        If the underlying dialect does not support
        PostgreSQL DROP TYPE, no action is taken.

        :param bind: a connectable :class:`_engine.Engine`,
         :class:`_engine.Connection`, or similar object to emit
         SQL.
        :param checkfirst: if ``True``, a query against
         the PG catalog will be first performed to see
         if the type actually exists before dropping.

        """
        if not bind.dialect.supports_native_enum:
            return

        super().drop(bind, checkfirst=checkfirst)

    def get_dbapi_type(self, dbapi: ModuleType) -> None:
        """dont return dbapi.STRING for ENUM in PostgreSQL, since that's
        a different type"""

        return None


class DomainGenerator(NamedTypeGenerator):
    def visit_DOMAIN(self, domain):
        if not self._can_create_type(domain):
            return
        with self.with_ddl_events(domain):
            self.connection.execute(CreateDomainType(domain))


class DomainDropper(NamedTypeDropper):
    def visit_DOMAIN(self, domain):
        if not self._can_drop_type(domain):
            return

        with self.with_ddl_events(domain):
            self.connection.execute(DropDomainType(domain))


class DOMAIN(NamedType, sqltypes.SchemaType):
    r"""Represent the DOMAIN PostgreSQL type.

    A domain is essentially a data type with optional constraints
    that restrict the allowed set of values. E.g.::

        PositiveInt = DOMAIN("pos_int", Integer, check="VALUE > 0", not_null=True)

        UsPostalCode = DOMAIN(
            "us_postal_code",
            Text,
            check="VALUE ~ '^\d{5}$' OR VALUE ~ '^\d{5}-\d{4}$'",
        )

    See the `PostgreSQL documentation`__ for additional details

    __ https://www.postgresql.org/docs/current/sql-createdomain.html

    .. versionadded:: 2.0

    """  # noqa: E501

    DDLGenerator = DomainGenerator
    DDLDropper = DomainDropper

    __visit_name__ = "DOMAIN"

    def __init__(
        self,
        name: str,
        data_type: _TypeEngineArgument[Any],
        *,
        collation: Optional[str] = None,
        default: Union[elements.TextClause, str, None] = None,
        constraint_name: Optional[str] = None,
        not_null: Optional[bool] = None,
        check: Union[elements.TextClause, str, None] = None,
        create_type: bool = True,
        **kw: Any,
    ):
        """
        Construct a DOMAIN.

        :param name: the name of the domain
        :param data_type: The underlying data type of the domain.
          This can include array specifiers.
        :param collation: An optional collation for the domain.
          If no collation is specified, the underlying data type's default
          collation is used. The underlying type must be collatable if
          ``collation`` is specified.
        :param default: The DEFAULT clause specifies a default value for
          columns of the domain data type. The default should be a string
          or a :func:`_expression.text` value.
          If no default value is specified, then the default value is
          the null value.
        :param constraint_name: An optional name for a constraint.
          If not specified, the backend generates a name.
        :param not_null: Values of this domain are prevented from being null.
          By default domain are allowed to be null. If not specified
          no nullability clause will be emitted.
        :param check: CHECK clause specify integrity constraint or test
          which values of the domain must satisfy. A constraint must be
          an expression producing a Boolean result that can use the key
          word VALUE to refer to the value being tested.
          Differently from PostgreSQL, only a single check clause is
          currently allowed in SQLAlchemy.
        :param schema: optional schema name
        :param metadata: optional :class:`_schema.MetaData` object which
         this :class:`_postgresql.DOMAIN` will be directly associated
        :param create_type: Defaults to True.
         Indicates that ``CREATE TYPE`` should be emitted, after optionally
         checking for the presence of the type, when the parent table is
         being created; and additionally that ``DROP TYPE`` is called
         when the table is dropped.

        """
        self.data_type = type_api.to_instance(data_type)
        self.default = default
        self.collation = collation
        self.constraint_name = constraint_name
        self.not_null = bool(not_null)
        if check is not None:
            check = coercions.expect(roles.DDLExpressionRole, check)
        self.check = check
        self.create_type = create_type
        super().__init__(name=name, **kw)

    @classmethod
    def __test_init__(cls):
        return cls("name", sqltypes.Integer)

    def adapt(self, impl, **kw):
        if self.default:
            kw["default"] = self.default
        if self.constraint_name is not None:
            kw["constraint_name"] = self.constraint_name
        if self.not_null:
            kw["not_null"] = self.not_null
        if self.check is not None:
            kw["check"] = str(self.check)
        if self.create_type:
            kw["create_type"] = self.create_type

        return super().adapt(impl, **kw)


class CreateEnumType(schema._CreateDropBase):
    __visit_name__ = "create_enum_type"


class DropEnumType(schema._CreateDropBase):
    __visit_name__ = "drop_enum_type"


class CreateDomainType(schema._CreateDropBase):
    """Represent a CREATE DOMAIN statement."""

    __visit_name__ = "create_domain_type"


class DropDomainType(schema._CreateDropBase):
    """Represent a DROP DOMAIN statement."""

    __visit_name__ = "drop_domain_type"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\util.py ===
# testing/util.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


from __future__ import annotations

from collections import deque
import contextlib
import decimal
import gc
from itertools import chain
import random
import sys
from sys import getsizeof
import time
import types
from typing import Any

from . import config
from . import mock
from .. import inspect
from ..engine import Connection
from ..schema import Column
from ..schema import DropConstraint
from ..schema import DropTable
from ..schema import ForeignKeyConstraint
from ..schema import MetaData
from ..schema import Table
from ..sql import schema
from ..sql.sqltypes import Integer
from ..util import decorator
from ..util import defaultdict
from ..util import has_refcount_gc
from ..util import inspect_getfullargspec


if not has_refcount_gc:

    def non_refcount_gc_collect(*args):
        gc.collect()
        gc.collect()

    gc_collect = lazy_gc = non_refcount_gc_collect
else:
    # assume CPython - straight gc.collect, lazy_gc() is a pass
    gc_collect = gc.collect

    def lazy_gc():
        pass


def picklers():
    picklers = set()
    import pickle

    picklers.add(pickle)

    # yes, this thing needs this much testing
    for pickle_ in picklers:
        for protocol in range(-2, pickle.HIGHEST_PROTOCOL + 1):
            yield pickle_.loads, lambda d: pickle_.dumps(d, protocol)


def random_choices(population, k=1):
    return random.choices(population, k=k)


def round_decimal(value, prec):
    if isinstance(value, float):
        return round(value, prec)

    # can also use shift() here but that is 2.6 only
    return (value * decimal.Decimal("1" + "0" * prec)).to_integral(
        decimal.ROUND_FLOOR
    ) / pow(10, prec)


class RandomSet(set):
    def __iter__(self):
        l = list(set.__iter__(self))
        random.shuffle(l)
        return iter(l)

    def pop(self):
        index = random.randint(0, len(self) - 1)
        item = list(set.__iter__(self))[index]
        self.remove(item)
        return item

    def union(self, other):
        return RandomSet(set.union(self, other))

    def difference(self, other):
        return RandomSet(set.difference(self, other))

    def intersection(self, other):
        return RandomSet(set.intersection(self, other))

    def copy(self):
        return RandomSet(self)


def conforms_partial_ordering(tuples, sorted_elements):
    """True if the given sorting conforms to the given partial ordering."""

    deps = defaultdict(set)
    for parent, child in tuples:
        deps[parent].add(child)
    for i, node in enumerate(sorted_elements):
        for n in sorted_elements[i:]:
            if node in deps[n]:
                return False
    else:
        return True


def all_partial_orderings(tuples, elements):
    edges = defaultdict(set)
    for parent, child in tuples:
        edges[child].add(parent)

    def _all_orderings(elements):
        if len(elements) == 1:
            yield list(elements)
        else:
            for elem in elements:
                subset = set(elements).difference([elem])
                if not subset.intersection(edges[elem]):
                    for sub_ordering in _all_orderings(subset):
                        yield [elem] + sub_ordering

    return iter(_all_orderings(elements))


def function_named(fn, name):
    """Return a function with a given __name__.

    Will assign to __name__ and return the original function if possible on
    the Python implementation, otherwise a new function will be constructed.

    This function should be phased out as much as possible
    in favor of @decorator.   Tests that "generate" many named tests
    should be modernized.

    """
    try:
        fn.__name__ = name
    except TypeError:
        fn = types.FunctionType(
            fn.__code__, fn.__globals__, name, fn.__defaults__, fn.__closure__
        )
    return fn


def run_as_contextmanager(ctx, fn, *arg, **kw):
    """Run the given function under the given contextmanager,
    simulating the behavior of 'with' to support older
    Python versions.

    This is not necessary anymore as we have placed 2.6
    as minimum Python version, however some tests are still using
    this structure.

    """

    obj = ctx.__enter__()
    try:
        result = fn(obj, *arg, **kw)
        ctx.__exit__(None, None, None)
        return result
    except:
        exc_info = sys.exc_info()
        raise_ = ctx.__exit__(*exc_info)
        if not raise_:
            raise
        else:
            return raise_


def rowset(results):
    """Converts the results of sql execution into a plain set of column tuples.

    Useful for asserting the results of an unordered query.
    """

    return {tuple(row) for row in results}


def fail(msg):
    assert False, msg


@decorator
def provide_metadata(fn, *args, **kw):
    """Provide bound MetaData for a single test, dropping afterwards.

    Legacy; use the "metadata" pytest fixture.

    """

    from . import fixtures

    metadata = schema.MetaData()
    self = args[0]
    prev_meta = getattr(self, "metadata", None)
    self.metadata = metadata
    try:
        return fn(*args, **kw)
    finally:
        # close out some things that get in the way of dropping tables.
        # when using the "metadata" fixture, there is a set ordering
        # of things that makes sure things are cleaned up in order, however
        # the simple "decorator" nature of this legacy function means
        # we have to hardcode some of that cleanup ahead of time.

        # close ORM sessions
        fixtures.close_all_sessions()

        # integrate with the "connection" fixture as there are many
        # tests where it is used along with provide_metadata
        cfc = fixtures.base._connection_fixture_connection
        if cfc:
            # TODO: this warning can be used to find all the places
            # this is used with connection fixture
            # warn("mixing legacy provide metadata with connection fixture")
            drop_all_tables_from_metadata(metadata, cfc)
            # as the provide_metadata fixture is often used with "testing.db",
            # when we do the drop we have to commit the transaction so that
            # the DB is actually updated as the CREATE would have been
            # committed
            cfc.get_transaction().commit()
        else:
            drop_all_tables_from_metadata(metadata, config.db)
        self.metadata = prev_meta


def flag_combinations(*combinations):
    """A facade around @testing.combinations() oriented towards boolean
    keyword-based arguments.

    Basically generates a nice looking identifier based on the keywords
    and also sets up the argument names.

    E.g.::

        @testing.flag_combinations(
            dict(lazy=False, passive=False),
            dict(lazy=True, passive=False),
            dict(lazy=False, passive=True),
            dict(lazy=False, passive=True, raiseload=True),
        )
        def test_fn(lazy, passive, raiseload): ...

    would result in::

        @testing.combinations(
            ("", False, False, False),
            ("lazy", True, False, False),
            ("lazy_passive", True, True, False),
            ("lazy_passive", True, True, True),
            id_="iaaa",
            argnames="lazy,passive,raiseload",
        )
        def test_fn(lazy, passive, raiseload): ...

    """

    keys = set()

    for d in combinations:
        keys.update(d)

    keys = sorted(keys)

    return config.combinations(
        *[
            ("_".join(k for k in keys if d.get(k, False)),)
            + tuple(d.get(k, False) for k in keys)
            for d in combinations
        ],
        id_="i" + ("a" * len(keys)),
        argnames=",".join(keys),
    )


def lambda_combinations(lambda_arg_sets, **kw):
    args = inspect_getfullargspec(lambda_arg_sets)

    arg_sets = lambda_arg_sets(*[mock.Mock() for arg in args[0]])

    def create_fixture(pos):
        def fixture(**kw):
            return lambda_arg_sets(**kw)[pos]

        fixture.__name__ = "fixture_%3.3d" % pos
        return fixture

    return config.combinations(
        *[(create_fixture(i),) for i in range(len(arg_sets))], **kw
    )


def resolve_lambda(__fn, **kw):
    """Given a no-arg lambda and a namespace, return a new lambda that
    has all the values filled in.

    This is used so that we can have module-level fixtures that
    refer to instance-level variables using lambdas.

    """

    pos_args = inspect_getfullargspec(__fn)[0]
    pass_pos_args = {arg: kw.pop(arg) for arg in pos_args}
    glb = dict(__fn.__globals__)
    glb.update(kw)
    new_fn = types.FunctionType(__fn.__code__, glb)
    return new_fn(**pass_pos_args)


def metadata_fixture(ddl="function"):
    """Provide MetaData for a pytest fixture."""

    def decorate(fn):
        def run_ddl(self):
            metadata = self.metadata = schema.MetaData()
            try:
                result = fn(self, metadata)
                metadata.create_all(config.db)
                # TODO:
                # somehow get a per-function dml erase fixture here
                yield result
            finally:
                metadata.drop_all(config.db)

        return config.fixture(scope=ddl)(run_ddl)

    return decorate


def force_drop_names(*names):
    """Force the given table names to be dropped after test complete,
    isolating for foreign key cycles

    """

    @decorator
    def go(fn, *args, **kw):
        try:
            return fn(*args, **kw)
        finally:
            drop_all_tables(config.db, inspect(config.db), include_names=names)

    return go


class adict(dict):
    """Dict keys available as attributes.  Shadows."""

    def __getattribute__(self, key):
        try:
            return self[key]
        except KeyError:
            return dict.__getattribute__(self, key)

    def __call__(self, *keys):
        return tuple([self[key] for key in keys])

    get_all = __call__


def drop_all_tables_from_metadata(metadata, engine_or_connection):
    from . import engines

    def go(connection):
        engines.testing_reaper.prepare_for_drop_tables(connection)

        if not connection.dialect.supports_alter:
            from . import assertions

            with assertions.expect_warnings(
                "Can't sort tables", assert_=False
            ):
                metadata.drop_all(connection)
        else:
            metadata.drop_all(connection)

    if not isinstance(engine_or_connection, Connection):
        with engine_or_connection.begin() as connection:
            go(connection)
    else:
        go(engine_or_connection)


def drop_all_tables(
    engine,
    inspector,
    schema=None,
    consider_schemas=(None,),
    include_names=None,
):
    if include_names is not None:
        include_names = set(include_names)

    if schema is not None:
        assert consider_schemas == (
            None,
        ), "consider_schemas and schema are mutually exclusive"
        consider_schemas = (schema,)

    with engine.begin() as conn:
        for table_key, fkcs in reversed(
            inspector.sort_tables_on_foreign_key_dependency(
                consider_schemas=consider_schemas
            )
        ):
            if table_key:
                if (
                    include_names is not None
                    and table_key[1] not in include_names
                ):
                    continue
                conn.execute(
                    DropTable(
                        Table(table_key[1], MetaData(), schema=table_key[0])
                    )
                )
            elif fkcs:
                if not engine.dialect.supports_alter:
                    continue
                for t_key, fkc in fkcs:
                    if (
                        include_names is not None
                        and t_key[1] not in include_names
                    ):
                        continue
                    tb = Table(
                        t_key[1],
                        MetaData(),
                        Column("x", Integer),
                        Column("y", Integer),
                        schema=t_key[0],
                    )
                    conn.execute(
                        DropConstraint(
                            ForeignKeyConstraint([tb.c.x], [tb.c.y], name=fkc)
                        )
                    )


def teardown_events(event_cls):
    @decorator
    def decorate(fn, *arg, **kw):
        try:
            return fn(*arg, **kw)
        finally:
            event_cls._clear()

    return decorate


def total_size(o):
    """Returns the approximate memory footprint an object and all of its
    contents.

    source: https://code.activestate.com/recipes/577504/


    """

    def dict_handler(d):
        return chain.from_iterable(d.items())

    all_handlers = {
        tuple: iter,
        list: iter,
        deque: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


def count_cache_key_tuples(tup):
    """given a cache key tuple, counts how many instances of actual
    tuples are found.

    used to alert large jumps in cache key complexity.

    """
    stack = [tup]

    sentinel = object()
    num_elements = 0

    while stack:
        elem = stack.pop(0)
        if elem is sentinel:
            num_elements += 1
        elif isinstance(elem, tuple):
            if elem:
                stack = list(elem) + [sentinel] + stack
    return num_elements


@contextlib.contextmanager
def skip_if_timeout(seconds: float, cleanup: Any = None):

    now = time.time()
    yield
    sec = time.time() - now
    if sec > seconds:
        try:
            cleanup()
        finally:
            config.skip_test(
                f"test took too long ({sec:.4f} seconds > {seconds})"
            )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\pretty\stringpict.py ===
"""Prettyprinter by Jurjen Bos.
(I hate spammers: mail me at pietjepuk314 at the reverse of ku.oc.oohay).
All objects have a method that create a "stringPict",
that can be used in the str method for pretty printing.

Updates by Jason Gedge (email <my last name> at cs mun ca)
    - terminal_string() method
    - minor fixes and changes (mostly to prettyForm)

TODO:
    - Allow left/center/right alignment options for above/below and
      top/center/bottom alignment options for left/right
"""

import shutil

from .pretty_symbology import hobj, vobj, xsym, xobj, pretty_use_unicode, line_width, center
from sympy.utilities.exceptions import sympy_deprecation_warning

_GLOBAL_WRAP_LINE = None

class stringPict:
    """An ASCII picture.
    The pictures are represented as a list of equal length strings.
    """
    #special value for stringPict.below
    LINE = 'line'

    def __init__(self, s, baseline=0):
        """Initialize from string.
        Multiline strings are centered.
        """
        self.s = s
        #picture is a string that just can be printed
        self.picture = stringPict.equalLengths(s.splitlines())
        #baseline is the line number of the "base line"
        self.baseline = baseline
        self.binding = None

    @staticmethod
    def equalLengths(lines):
        # empty lines
        if not lines:
            return ['']

        width = max(line_width(line) for line in lines)
        return [center(line, width) for line in lines]

    def height(self):
        """The height of the picture in characters."""
        return len(self.picture)

    def width(self):
        """The width of the picture in characters."""
        return line_width(self.picture[0])

    @staticmethod
    def next(*args):
        """Put a string of stringPicts next to each other.
        Returns string, baseline arguments for stringPict.
        """
        #convert everything to stringPicts
        objects = []
        for arg in args:
            if isinstance(arg, str):
                arg = stringPict(arg)
            objects.append(arg)

        #make a list of pictures, with equal height and baseline
        newBaseline = max(obj.baseline for obj in objects)
        newHeightBelowBaseline = max(
            obj.height() - obj.baseline
            for obj in objects)
        newHeight = newBaseline + newHeightBelowBaseline

        pictures = []
        for obj in objects:
            oneEmptyLine = [' '*obj.width()]
            basePadding = newBaseline - obj.baseline
            totalPadding = newHeight - obj.height()
            pictures.append(
                oneEmptyLine * basePadding +
                obj.picture +
                oneEmptyLine * (totalPadding - basePadding))

        result = [''.join(lines) for lines in zip(*pictures)]
        return '\n'.join(result), newBaseline

    def right(self, *args):
        r"""Put pictures next to this one.
        Returns string, baseline arguments for stringPict.
        (Multiline) strings are allowed, and are given a baseline of 0.

        Examples
        ========

        >>> from sympy.printing.pretty.stringpict import stringPict
        >>> print(stringPict("10").right(" + ",stringPict("1\r-\r2",1))[0])
             1
        10 + -
             2

        """
        return stringPict.next(self, *args)

    def left(self, *args):
        """Put pictures (left to right) at left.
        Returns string, baseline arguments for stringPict.
        """
        return stringPict.next(*(args + (self,)))

    @staticmethod
    def stack(*args):
        """Put pictures on top of each other,
        from top to bottom.
        Returns string, baseline arguments for stringPict.
        The baseline is the baseline of the second picture.
        Everything is centered.
        Baseline is the baseline of the second picture.
        Strings are allowed.
        The special value stringPict.LINE is a row of '-' extended to the width.
        """
        #convert everything to stringPicts; keep LINE
        objects = []
        for arg in args:
            if arg is not stringPict.LINE and isinstance(arg, str):
                arg = stringPict(arg)
            objects.append(arg)

        #compute new width
        newWidth = max(
            obj.width()
            for obj in objects
            if obj is not stringPict.LINE)

        lineObj = stringPict(hobj('-', newWidth))

        #replace LINE with proper lines
        for i, obj in enumerate(objects):
            if obj is stringPict.LINE:
                objects[i] = lineObj

        #stack the pictures, and center the result
        newPicture = [center(line, newWidth) for obj in objects for line in obj.picture]
        newBaseline = objects[0].height() + objects[1].baseline
        return '\n'.join(newPicture), newBaseline

    def below(self, *args):
        """Put pictures under this picture.
        Returns string, baseline arguments for stringPict.
        Baseline is baseline of top picture

        Examples
        ========

        >>> from sympy.printing.pretty.stringpict import stringPict
        >>> print(stringPict("x+3").below(
        ...       stringPict.LINE, '3')[0]) #doctest: +NORMALIZE_WHITESPACE
        x+3
        ---
         3

        """
        s, baseline = stringPict.stack(self, *args)
        return s, self.baseline

    def above(self, *args):
        """Put pictures above this picture.
        Returns string, baseline arguments for stringPict.
        Baseline is baseline of bottom picture.
        """
        string, baseline = stringPict.stack(*(args + (self,)))
        baseline = len(string.splitlines()) - self.height() + self.baseline
        return string, baseline

    def parens(self, left='(', right=')', ifascii_nougly=False):
        """Put parentheses around self.
        Returns string, baseline arguments for stringPict.

        left or right can be None or empty string which means 'no paren from
        that side'
        """
        h = self.height()
        b = self.baseline

        # XXX this is a hack -- ascii parens are ugly!
        if ifascii_nougly and not pretty_use_unicode():
            h = 1
            b = 0

        res = self

        if left:
            lparen = stringPict(vobj(left, h), baseline=b)
            res = stringPict(*lparen.right(self))
        if right:
            rparen = stringPict(vobj(right, h), baseline=b)
            res = stringPict(*res.right(rparen))

        return ('\n'.join(res.picture), res.baseline)

    def leftslash(self):
        """Precede object by a slash of the proper size.
        """
        # XXX not used anywhere ?
        height = max(
            self.baseline,
            self.height() - 1 - self.baseline)*2 + 1
        slash = '\n'.join(
            ' '*(height - i - 1) + xobj('/', 1) + ' '*i
            for i in range(height)
        )
        return self.left(stringPict(slash, height//2))

    def root(self, n=None):
        """Produce a nice root symbol.
        Produces ugly results for big n inserts.
        """
        # XXX not used anywhere
        # XXX duplicate of root drawing in pretty.py
        #put line over expression
        result = self.above('_'*self.width())
        #construct right half of root symbol
        height = self.height()
        slash = '\n'.join(
            ' ' * (height - i - 1) + '/' + ' ' * i
            for i in range(height)
        )
        slash = stringPict(slash, height - 1)
        #left half of root symbol
        if height > 2:
            downline = stringPict('\\ \n \\', 1)
        else:
            downline = stringPict('\\')
        #put n on top, as low as possible
        if n is not None and n.width() > downline.width():
            downline = downline.left(' '*(n.width() - downline.width()))
            downline = downline.above(n)
        #build root symbol
        root = downline.right(slash)
        #glue it on at the proper height
        #normally, the root symbel is as high as self
        #which is one less than result
        #this moves the root symbol one down
        #if the root became higher, the baseline has to grow too
        root.baseline = result.baseline - result.height() + root.height()
        return result.left(root)

    def render(self, * args, **kwargs):
        """Return the string form of self.

           Unless the argument line_break is set to False, it will
           break the expression in a form that can be printed
           on the terminal without being broken up.
         """
        if _GLOBAL_WRAP_LINE is not None:
            kwargs["wrap_line"] = _GLOBAL_WRAP_LINE

        if kwargs["wrap_line"] is False:
            return "\n".join(self.picture)

        if kwargs["num_columns"] is not None:
            # Read the argument num_columns if it is not None
            ncols = kwargs["num_columns"]
        else:
            # Attempt to get a terminal width
            ncols = self.terminal_width()

        if ncols <= 0:
            ncols = 80

        # If smaller than the terminal width, no need to correct
        if self.width() <= ncols:
            return type(self.picture[0])(self)

        """
        Break long-lines in a visually pleasing format.
        without overflow indicators | with overflow indicators
        |   2  2        3     |     |   2  2        3    ↪|
        |6*x *y  + 4*x*y  +   |     |6*x *y  + 4*x*y  +  ↪|
        |                     |     |                     |
        |     3    4    4     |     |↪      3    4    4   |
        |4*y*x  + x  + y      |     |↪ 4*y*x  + x  + y    |
        |a*c*e + a*c*f + a*d  |     |a*c*e + a*c*f + a*d ↪|
        |*e + a*d*f + b*c*e   |     |                     |
        |+ b*c*f + b*d*e + b  |     |↪ *e + a*d*f + b*c* ↪|
        |*d*f                 |     |                     |
        |                     |     |↪ e + b*c*f + b*d*e ↪|
        |                     |     |                     |
        |                     |     |↪ + b*d*f            |
        """

        overflow_first = ""
        if kwargs["use_unicode"] or pretty_use_unicode():
            overflow_start = "\N{RIGHTWARDS ARROW WITH HOOK} "
            overflow_end   = " \N{RIGHTWARDS ARROW WITH HOOK}"
        else:
            overflow_start = "> "
            overflow_end   = " >"

        def chunks(line):
            """Yields consecutive chunks of line_width ncols"""
            prefix = overflow_first
            width, start = line_width(prefix + overflow_end), 0
            for i, x in enumerate(line):
                wx = line_width(x)
                # Only flush the screen when the current character overflows.
                # This way, combining marks can be appended even when width == ncols.
                if width + wx > ncols:
                    yield prefix + line[start:i] + overflow_end
                    prefix = overflow_start
                    width, start = line_width(prefix + overflow_end), i
                width += wx
            yield prefix + line[start:]

        # Concurrently assemble chunks of all lines into individual screens
        pictures = zip(*map(chunks, self.picture))

        # Join lines of each screen into sub-pictures
        pictures = ["\n".join(picture) for picture in pictures]

        # Add spacers between sub-pictures
        return "\n\n".join(pictures)

    def terminal_width(self):
        """Return the terminal width if possible, otherwise return 0.
        """
        size = shutil.get_terminal_size(fallback=(0, 0))
        return size.columns

    def __eq__(self, o):
        if isinstance(o, str):
            return '\n'.join(self.picture) == o
        elif isinstance(o, stringPict):
            return o.picture == self.picture
        return False

    def __hash__(self):
        return super().__hash__()

    def __str__(self):
        return '\n'.join(self.picture)

    def __repr__(self):
        return "stringPict(%r,%d)" % ('\n'.join(self.picture), self.baseline)

    def __getitem__(self, index):
        return self.picture[index]

    def __len__(self):
        return len(self.s)


class prettyForm(stringPict):
    """
    Extension of the stringPict class that knows about basic math applications,
    optimizing double minus signs.

    "Binding" is interpreted as follows::

        ATOM this is an atom: never needs to be parenthesized
        FUNC this is a function application: parenthesize if added (?)
        DIV  this is a division: make wider division if divided
        POW  this is a power: only parenthesize if exponent
        MUL  this is a multiplication: parenthesize if powered
        ADD  this is an addition: parenthesize if multiplied or powered
        NEG  this is a negative number: optimize if added, parenthesize if
             multiplied or powered
        OPEN this is an open object: parenthesize if added, multiplied, or
             powered (example: Piecewise)
    """
    ATOM, FUNC, DIV, POW, MUL, ADD, NEG, OPEN = range(8)

    def __init__(self, s, baseline=0, binding=0, unicode=None):
        """Initialize from stringPict and binding power."""
        stringPict.__init__(self, s, baseline)
        self.binding = binding
        if unicode is not None:
            sympy_deprecation_warning(
                """
                The unicode argument to prettyForm is deprecated. Only the s
                argument (the first positional argument) should be passed.
                """,
                deprecated_since_version="1.7",
                active_deprecations_target="deprecated-pretty-printing-functions")
        self._unicode = unicode or s

    @property
    def unicode(self):
        sympy_deprecation_warning(
            """
            The prettyForm.unicode attribute is deprecated. Use the
            prettyForm.s attribute instead.
            """,
            deprecated_since_version="1.7",
            active_deprecations_target="deprecated-pretty-printing-functions")
        return self._unicode

    # Note: code to handle subtraction is in _print_Add

    def __add__(self, *others):
        """Make a pretty addition.
        Addition of negative numbers is simplified.
        """
        arg = self
        if arg.binding > prettyForm.NEG:
            arg = stringPict(*arg.parens())
        result = [arg]
        for arg in others:
            #add parentheses for weak binders
            if arg.binding > prettyForm.NEG:
                arg = stringPict(*arg.parens())
            #use existing minus sign if available
            if arg.binding != prettyForm.NEG:
                result.append(' + ')
            result.append(arg)
        return prettyForm(binding=prettyForm.ADD, *stringPict.next(*result))

    def __truediv__(self, den, slashed=False):
        """Make a pretty division; stacked or slashed.
        """
        if slashed:
            raise NotImplementedError("Can't do slashed fraction yet")
        num = self
        if num.binding == prettyForm.DIV:
            num = stringPict(*num.parens())
        if den.binding == prettyForm.DIV:
            den = stringPict(*den.parens())

        if num.binding==prettyForm.NEG:
            num = num.right(" ")[0]

        return prettyForm(binding=prettyForm.DIV, *stringPict.stack(
            num,
            stringPict.LINE,
            den))

    def __mul__(self, *others):
        """Make a pretty multiplication.
        Parentheses are needed around +, - and neg.
        """
        quantity = {
            'degree': "\N{DEGREE SIGN}"
        }

        if len(others) == 0:
            return self  # We aren't actually multiplying... So nothing to do here.

        # add parens on args that need them
        arg = self
        if arg.binding > prettyForm.MUL and arg.binding != prettyForm.NEG:
            arg = stringPict(*arg.parens())
        result = [arg]
        for arg in others:
            if arg.picture[0] not in quantity.values():
                result.append(xsym('*'))
            #add parentheses for weak binders
            if arg.binding > prettyForm.MUL and arg.binding != prettyForm.NEG:
                arg = stringPict(*arg.parens())
            result.append(arg)

        len_res = len(result)
        for i in range(len_res):
            if i < len_res - 1 and result[i] == '-1' and result[i + 1] == xsym('*'):
                # substitute -1 by -, like in -1*x -> -x
                result.pop(i)
                result.pop(i)
                result.insert(i, '-')
        if result[0][0] == '-':
            # if there is a - sign in front of all
            # This test was failing to catch a prettyForm.__mul__(prettyForm("-1", 0, 6)) being negative
            bin = prettyForm.NEG
            if result[0] == '-':
                right = result[1]
                if right.picture[right.baseline][0] == '-':
                    result[0] = '- '
        else:
            bin = prettyForm.MUL
        return prettyForm(binding=bin, *stringPict.next(*result))

    def __repr__(self):
        return "prettyForm(%r,%d,%d)" % (
            '\n'.join(self.picture),
            self.baseline,
            self.binding)

    def __pow__(self, b):
        """Make a pretty power.
        """
        a = self
        use_inline_func_form = False
        if b.binding == prettyForm.POW:
            b = stringPict(*b.parens())
        if a.binding > prettyForm.FUNC:
            a = stringPict(*a.parens())
        elif a.binding == prettyForm.FUNC:
            # heuristic for when to use inline power
            if b.height() > 1:
                a = stringPict(*a.parens())
            else:
                use_inline_func_form = True

        if use_inline_func_form:
            #         2
            #  sin  +   + (x)
            b.baseline = a.prettyFunc.baseline + b.height()
            func = stringPict(*a.prettyFunc.right(b))
            return prettyForm(*func.right(a.prettyArgs))
        else:
            #      2    <-- top
            # (x+y)     <-- bot
            top = stringPict(*b.left(' '*a.width()))
            bot = stringPict(*a.right(' '*b.width()))

        return prettyForm(binding=prettyForm.POW, *bot.above(top))

    simpleFunctions = ["sin", "cos", "tan"]

    @staticmethod
    def apply(function, *args):
        """Functions of one or more variables.
        """
        if function in prettyForm.simpleFunctions:
            #simple function: use only space if possible
            assert len(
                args) == 1, "Simple function %s must have 1 argument" % function
            arg = args[0].__pretty__()
            if arg.binding <= prettyForm.DIV:
                #optimization: no parentheses necessary
                return prettyForm(binding=prettyForm.FUNC, *arg.left(function + ' '))
        argumentList = []
        for arg in args:
            argumentList.append(',')
            argumentList.append(arg.__pretty__())
        argumentList = stringPict(*stringPict.next(*argumentList[1:]))
        argumentList = stringPict(*argumentList.parens())
        return prettyForm(binding=prettyForm.ATOM, *argumentList.left(function))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\vcs\git.py ===
import logging
import os.path
import pathlib
import re
import urllib.parse
import urllib.request
from dataclasses import replace
from typing import Any, List, Optional, Tuple

from pip._internal.exceptions import BadCommand, InstallationError
from pip._internal.utils.misc import HiddenText, display_path, hide_url
from pip._internal.utils.subprocess import make_command
from pip._internal.vcs.versioncontrol import (
    AuthInfo,
    RemoteNotFoundError,
    RemoteNotValidError,
    RevOptions,
    VersionControl,
    find_path_to_project_root_from_repo_root,
    vcs,
)

urlsplit = urllib.parse.urlsplit
urlunsplit = urllib.parse.urlunsplit


logger = logging.getLogger(__name__)


GIT_VERSION_REGEX = re.compile(
    r"^git version "  # Prefix.
    r"(\d+)"  # Major.
    r"\.(\d+)"  # Dot, minor.
    r"(?:\.(\d+))?"  # Optional dot, patch.
    r".*$"  # Suffix, including any pre- and post-release segments we don't care about.
)

HASH_REGEX = re.compile("^[a-fA-F0-9]{40}$")

# SCP (Secure copy protocol) shorthand. e.g. 'git@example.com:foo/bar.git'
SCP_REGEX = re.compile(
    r"""^
    # Optional user, e.g. 'git@'
    (\w+@)?
    # Server, e.g. 'github.com'.
    ([^/:]+):
    # The server-side path. e.g. 'user/project.git'. Must start with an
    # alphanumeric character so as not to be confusable with a Windows paths
    # like 'C:/foo/bar' or 'C:\foo\bar'.
    (\w[^:]*)
    $""",
    re.VERBOSE,
)


def looks_like_hash(sha: str) -> bool:
    return bool(HASH_REGEX.match(sha))


class Git(VersionControl):
    name = "git"
    dirname = ".git"
    repo_name = "clone"
    schemes = (
        "git+http",
        "git+https",
        "git+ssh",
        "git+git",
        "git+file",
    )
    # Prevent the user's environment variables from interfering with pip:
    # https://github.com/pypa/pip/issues/1130
    unset_environ = ("GIT_DIR", "GIT_WORK_TREE")
    default_arg_rev = "HEAD"

    @staticmethod
    def get_base_rev_args(rev: str) -> List[str]:
        return [rev]

    @classmethod
    def run_command(cls, *args: Any, **kwargs: Any) -> str:
        if os.environ.get("PIP_NO_INPUT"):
            extra_environ = kwargs.get("extra_environ", {})
            extra_environ["GIT_TERMINAL_PROMPT"] = "0"
            extra_environ["GIT_SSH_COMMAND"] = "ssh -oBatchMode=yes"
            kwargs["extra_environ"] = extra_environ
        return super().run_command(*args, **kwargs)

    def is_immutable_rev_checkout(self, url: str, dest: str) -> bool:
        _, rev_options = self.get_url_rev_options(hide_url(url))
        if not rev_options.rev:
            return False
        if not self.is_commit_id_equal(dest, rev_options.rev):
            # the current commit is different from rev,
            # which means rev was something else than a commit hash
            return False
        # return False in the rare case rev is both a commit hash
        # and a tag or a branch; we don't want to cache in that case
        # because that branch/tag could point to something else in the future
        is_tag_or_branch = bool(self.get_revision_sha(dest, rev_options.rev)[0])
        return not is_tag_or_branch

    def get_git_version(self) -> Tuple[int, ...]:
        version = self.run_command(
            ["version"],
            command_desc="git version",
            show_stdout=False,
            stdout_only=True,
        )
        match = GIT_VERSION_REGEX.match(version)
        if not match:
            logger.warning("Can't parse git version: %s", version)
            return ()
        return (int(match.group(1)), int(match.group(2)))

    @classmethod
    def get_current_branch(cls, location: str) -> Optional[str]:
        """
        Return the current branch, or None if HEAD isn't at a branch
        (e.g. detached HEAD).
        """
        # git-symbolic-ref exits with empty stdout if "HEAD" is a detached
        # HEAD rather than a symbolic ref.  In addition, the -q causes the
        # command to exit with status code 1 instead of 128 in this case
        # and to suppress the message to stderr.
        args = ["symbolic-ref", "-q", "HEAD"]
        output = cls.run_command(
            args,
            extra_ok_returncodes=(1,),
            show_stdout=False,
            stdout_only=True,
            cwd=location,
        )
        ref = output.strip()

        if ref.startswith("refs/heads/"):
            return ref[len("refs/heads/") :]

        return None

    @classmethod
    def get_revision_sha(cls, dest: str, rev: str) -> Tuple[Optional[str], bool]:
        """
        Return (sha_or_none, is_branch), where sha_or_none is a commit hash
        if the revision names a remote branch or tag, otherwise None.

        Args:
          dest: the repository directory.
          rev: the revision name.
        """
        # Pass rev to pre-filter the list.
        output = cls.run_command(
            ["show-ref", rev],
            cwd=dest,
            show_stdout=False,
            stdout_only=True,
            on_returncode="ignore",
        )
        refs = {}
        # NOTE: We do not use splitlines here since that would split on other
        #       unicode separators, which can be maliciously used to install a
        #       different revision.
        for line in output.strip().split("\n"):
            line = line.rstrip("\r")
            if not line:
                continue
            try:
                ref_sha, ref_name = line.split(" ", maxsplit=2)
            except ValueError:
                # Include the offending line to simplify troubleshooting if
                # this error ever occurs.
                raise ValueError(f"unexpected show-ref line: {line!r}")

            refs[ref_name] = ref_sha

        branch_ref = f"refs/remotes/origin/{rev}"
        tag_ref = f"refs/tags/{rev}"

        sha = refs.get(branch_ref)
        if sha is not None:
            return (sha, True)

        sha = refs.get(tag_ref)

        return (sha, False)

    @classmethod
    def _should_fetch(cls, dest: str, rev: str) -> bool:
        """
        Return true if rev is a ref or is a commit that we don't have locally.

        Branches and tags are not considered in this method because they are
        assumed to be always available locally (which is a normal outcome of
        ``git clone`` and ``git fetch --tags``).
        """
        if rev.startswith("refs/"):
            # Always fetch remote refs.
            return True

        if not looks_like_hash(rev):
            # Git fetch would fail with abbreviated commits.
            return False

        if cls.has_commit(dest, rev):
            # Don't fetch if we have the commit locally.
            return False

        return True

    @classmethod
    def resolve_revision(
        cls, dest: str, url: HiddenText, rev_options: RevOptions
    ) -> RevOptions:
        """
        Resolve a revision to a new RevOptions object with the SHA1 of the
        branch, tag, or ref if found.

        Args:
          rev_options: a RevOptions object.
        """
        rev = rev_options.arg_rev
        # The arg_rev property's implementation for Git ensures that the
        # rev return value is always non-None.
        assert rev is not None

        sha, is_branch = cls.get_revision_sha(dest, rev)

        if sha is not None:
            rev_options = rev_options.make_new(sha)
            rev_options = replace(rev_options, branch_name=(rev if is_branch else None))

            return rev_options

        # Do not show a warning for the common case of something that has
        # the form of a Git commit hash.
        if not looks_like_hash(rev):
            logger.warning(
                "Did not find branch or tag '%s', assuming revision or ref.",
                rev,
            )

        if not cls._should_fetch(dest, rev):
            return rev_options

        # fetch the requested revision
        cls.run_command(
            make_command("fetch", "-q", url, rev_options.to_args()),
            cwd=dest,
        )
        # Change the revision to the SHA of the ref we fetched
        sha = cls.get_revision(dest, rev="FETCH_HEAD")
        rev_options = rev_options.make_new(sha)

        return rev_options

    @classmethod
    def is_commit_id_equal(cls, dest: str, name: Optional[str]) -> bool:
        """
        Return whether the current commit hash equals the given name.

        Args:
          dest: the repository directory.
          name: a string name.
        """
        if not name:
            # Then avoid an unnecessary subprocess call.
            return False

        return cls.get_revision(dest) == name

    def fetch_new(
        self, dest: str, url: HiddenText, rev_options: RevOptions, verbosity: int
    ) -> None:
        rev_display = rev_options.to_display()
        logger.info("Cloning %s%s to %s", url, rev_display, display_path(dest))
        if verbosity <= 0:
            flags: Tuple[str, ...] = ("--quiet",)
        elif verbosity == 1:
            flags = ()
        else:
            flags = ("--verbose", "--progress")
        if self.get_git_version() >= (2, 17):
            # Git added support for partial clone in 2.17
            # https://git-scm.com/docs/partial-clone
            # Speeds up cloning by functioning without a complete copy of repository
            self.run_command(
                make_command(
                    "clone",
                    "--filter=blob:none",
                    *flags,
                    url,
                    dest,
                )
            )
        else:
            self.run_command(make_command("clone", *flags, url, dest))

        if rev_options.rev:
            # Then a specific revision was requested.
            rev_options = self.resolve_revision(dest, url, rev_options)
            branch_name = getattr(rev_options, "branch_name", None)
            logger.debug("Rev options %s, branch_name %s", rev_options, branch_name)
            if branch_name is None:
                # Only do a checkout if the current commit id doesn't match
                # the requested revision.
                if not self.is_commit_id_equal(dest, rev_options.rev):
                    cmd_args = make_command(
                        "checkout",
                        "-q",
                        rev_options.to_args(),
                    )
                    self.run_command(cmd_args, cwd=dest)
            elif self.get_current_branch(dest) != branch_name:
                # Then a specific branch was requested, and that branch
                # is not yet checked out.
                track_branch = f"origin/{branch_name}"
                cmd_args = [
                    "checkout",
                    "-b",
                    branch_name,
                    "--track",
                    track_branch,
                ]
                self.run_command(cmd_args, cwd=dest)
        else:
            sha = self.get_revision(dest)
            rev_options = rev_options.make_new(sha)

        logger.info("Resolved %s to commit %s", url, rev_options.rev)

        #: repo may contain submodules
        self.update_submodules(dest)

    def switch(self, dest: str, url: HiddenText, rev_options: RevOptions) -> None:
        self.run_command(
            make_command("config", "remote.origin.url", url),
            cwd=dest,
        )
        cmd_args = make_command("checkout", "-q", rev_options.to_args())
        self.run_command(cmd_args, cwd=dest)

        self.update_submodules(dest)

    def update(self, dest: str, url: HiddenText, rev_options: RevOptions) -> None:
        # First fetch changes from the default remote
        if self.get_git_version() >= (1, 9):
            # fetch tags in addition to everything else
            self.run_command(["fetch", "-q", "--tags"], cwd=dest)
        else:
            self.run_command(["fetch", "-q"], cwd=dest)
        # Then reset to wanted revision (maybe even origin/master)
        rev_options = self.resolve_revision(dest, url, rev_options)
        cmd_args = make_command("reset", "--hard", "-q", rev_options.to_args())
        self.run_command(cmd_args, cwd=dest)
        #: update submodules
        self.update_submodules(dest)

    @classmethod
    def get_remote_url(cls, location: str) -> str:
        """
        Return URL of the first remote encountered.

        Raises RemoteNotFoundError if the repository does not have a remote
        url configured.
        """
        # We need to pass 1 for extra_ok_returncodes since the command
        # exits with return code 1 if there are no matching lines.
        stdout = cls.run_command(
            ["config", "--get-regexp", r"remote\..*\.url"],
            extra_ok_returncodes=(1,),
            show_stdout=False,
            stdout_only=True,
            cwd=location,
        )
        remotes = stdout.splitlines()
        try:
            found_remote = remotes[0]
        except IndexError:
            raise RemoteNotFoundError

        for remote in remotes:
            if remote.startswith("remote.origin.url "):
                found_remote = remote
                break
        url = found_remote.split(" ")[1]
        return cls._git_remote_to_pip_url(url.strip())

    @staticmethod
    def _git_remote_to_pip_url(url: str) -> str:
        """
        Convert a remote url from what git uses to what pip accepts.

        There are 3 legal forms **url** may take:

            1. A fully qualified url: ssh://git@example.com/foo/bar.git
            2. A local project.git folder: /path/to/bare/repository.git
            3. SCP shorthand for form 1: git@example.com:foo/bar.git

        Form 1 is output as-is. Form 2 must be converted to URI and form 3 must
        be converted to form 1.

        See the corresponding test test_git_remote_url_to_pip() for examples of
        sample inputs/outputs.
        """
        if re.match(r"\w+://", url):
            # This is already valid. Pass it though as-is.
            return url
        if os.path.exists(url):
            # A local bare remote (git clone --mirror).
            # Needs a file:// prefix.
            return pathlib.PurePath(url).as_uri()
        scp_match = SCP_REGEX.match(url)
        if scp_match:
            # Add an ssh:// prefix and replace the ':' with a '/'.
            return scp_match.expand(r"ssh://\1\2/\3")
        # Otherwise, bail out.
        raise RemoteNotValidError(url)

    @classmethod
    def has_commit(cls, location: str, rev: str) -> bool:
        """
        Check if rev is a commit that is available in the local repository.
        """
        try:
            cls.run_command(
                ["rev-parse", "-q", "--verify", "sha^" + rev],
                cwd=location,
                log_failed_cmd=False,
            )
        except InstallationError:
            return False
        else:
            return True

    @classmethod
    def get_revision(cls, location: str, rev: Optional[str] = None) -> str:
        if rev is None:
            rev = "HEAD"
        current_rev = cls.run_command(
            ["rev-parse", rev],
            show_stdout=False,
            stdout_only=True,
            cwd=location,
        )
        return current_rev.strip()

    @classmethod
    def get_subdirectory(cls, location: str) -> Optional[str]:
        """
        Return the path to Python project root, relative to the repo root.
        Return None if the project root is in the repo root.
        """
        # find the repo root
        git_dir = cls.run_command(
            ["rev-parse", "--git-dir"],
            show_stdout=False,
            stdout_only=True,
            cwd=location,
        ).strip()
        if not os.path.isabs(git_dir):
            git_dir = os.path.join(location, git_dir)
        repo_root = os.path.abspath(os.path.join(git_dir, ".."))
        return find_path_to_project_root_from_repo_root(location, repo_root)

    @classmethod
    def get_url_rev_and_auth(cls, url: str) -> Tuple[str, Optional[str], AuthInfo]:
        """
        Prefixes stub URLs like 'user@hostname:user/repo.git' with 'ssh://'.
        That's required because although they use SSH they sometimes don't
        work with a ssh:// scheme (e.g. GitHub). But we need a scheme for
        parsing. Hence we remove it again afterwards and return it as a stub.
        """
        # Works around an apparent Git bug
        # (see https://article.gmane.org/gmane.comp.version-control.git/146500)
        scheme, netloc, path, query, fragment = urlsplit(url)
        if scheme.endswith("file"):
            initial_slashes = path[: -len(path.lstrip("/"))]
            newpath = initial_slashes + urllib.request.url2pathname(path).replace(
                "\\", "/"
            ).lstrip("/")
            after_plus = scheme.find("+") + 1
            url = scheme[:after_plus] + urlunsplit(
                (scheme[after_plus:], netloc, newpath, query, fragment),
            )

        if "://" not in url:
            assert "file:" not in url
            url = url.replace("git+", "git+ssh://")
            url, rev, user_pass = super().get_url_rev_and_auth(url)
            url = url.replace("ssh://", "")
        else:
            url, rev, user_pass = super().get_url_rev_and_auth(url)

        return url, rev, user_pass

    @classmethod
    def update_submodules(cls, location: str) -> None:
        if not os.path.exists(os.path.join(location, ".gitmodules")):
            return
        cls.run_command(
            ["submodule", "update", "--init", "--recursive", "-q"],
            cwd=location,
        )

    @classmethod
    def get_repository_root(cls, location: str) -> Optional[str]:
        loc = super().get_repository_root(location)
        if loc:
            return loc
        try:
            r = cls.run_command(
                ["rev-parse", "--show-toplevel"],
                cwd=location,
                show_stdout=False,
                stdout_only=True,
                on_returncode="raise",
                log_failed_cmd=False,
            )
        except BadCommand:
            logger.debug(
                "could not determine if %s is under git control "
                "because git is not available",
                location,
            )
            return None
        except InstallationError:
            return None
        return os.path.normpath(r.rstrip("\r\n"))

    @staticmethod
    def should_add_vcs_url_prefix(repo_url: str) -> bool:
        """In either https or ssh form, requirements must be prefixed with git+."""
        return True


vcs.register(Git)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\postgresql\ext.py ===
# dialects/postgresql/ext.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors
from __future__ import annotations

from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import overload
from typing import TYPE_CHECKING
from typing import TypeVar

from . import types
from .array import ARRAY
from ...sql import coercions
from ...sql import elements
from ...sql import expression
from ...sql import functions
from ...sql import roles
from ...sql import schema
from ...sql.schema import ColumnCollectionConstraint
from ...sql.sqltypes import TEXT
from ...sql.visitors import InternalTraversal

if TYPE_CHECKING:
    from ...sql._typing import _ColumnExpressionArgument
    from ...sql.elements import ClauseElement
    from ...sql.elements import ColumnElement
    from ...sql.operators import OperatorType
    from ...sql.selectable import FromClause
    from ...sql.visitors import _CloneCallableType
    from ...sql.visitors import _TraverseInternalsType

_T = TypeVar("_T", bound=Any)


class aggregate_order_by(expression.ColumnElement[_T]):
    """Represent a PostgreSQL aggregate order by expression.

    E.g.::

        from sqlalchemy.dialects.postgresql import aggregate_order_by

        expr = func.array_agg(aggregate_order_by(table.c.a, table.c.b.desc()))
        stmt = select(expr)

    would represent the expression:

    .. sourcecode:: sql

        SELECT array_agg(a ORDER BY b DESC) FROM table;

    Similarly::

        expr = func.string_agg(
            table.c.a, aggregate_order_by(literal_column("','"), table.c.a)
        )
        stmt = select(expr)

    Would represent:

    .. sourcecode:: sql

        SELECT string_agg(a, ',' ORDER BY a) FROM table;

    .. versionchanged:: 1.2.13 - the ORDER BY argument may be multiple terms

    .. seealso::

        :class:`_functions.array_agg`

    """

    __visit_name__ = "aggregate_order_by"

    stringify_dialect = "postgresql"
    _traverse_internals: _TraverseInternalsType = [
        ("target", InternalTraversal.dp_clauseelement),
        ("type", InternalTraversal.dp_type),
        ("order_by", InternalTraversal.dp_clauseelement),
    ]

    @overload
    def __init__(
        self,
        target: ColumnElement[_T],
        *order_by: _ColumnExpressionArgument[Any],
    ): ...

    @overload
    def __init__(
        self,
        target: _ColumnExpressionArgument[_T],
        *order_by: _ColumnExpressionArgument[Any],
    ): ...

    def __init__(
        self,
        target: _ColumnExpressionArgument[_T],
        *order_by: _ColumnExpressionArgument[Any],
    ):
        self.target: ClauseElement = coercions.expect(
            roles.ExpressionElementRole, target
        )
        self.type = self.target.type

        _lob = len(order_by)
        self.order_by: ClauseElement
        if _lob == 0:
            raise TypeError("at least one ORDER BY element is required")
        elif _lob == 1:
            self.order_by = coercions.expect(
                roles.ExpressionElementRole, order_by[0]
            )
        else:
            self.order_by = elements.ClauseList(
                *order_by, _literal_as_text_role=roles.ExpressionElementRole
            )

    def self_group(
        self, against: Optional[OperatorType] = None
    ) -> ClauseElement:
        return self

    def get_children(self, **kwargs: Any) -> Iterable[ClauseElement]:
        return self.target, self.order_by

    def _copy_internals(
        self, clone: _CloneCallableType = elements._clone, **kw: Any
    ) -> None:
        self.target = clone(self.target, **kw)
        self.order_by = clone(self.order_by, **kw)

    @property
    def _from_objects(self) -> List[FromClause]:
        return self.target._from_objects + self.order_by._from_objects


class ExcludeConstraint(ColumnCollectionConstraint):
    """A table-level EXCLUDE constraint.

    Defines an EXCLUDE constraint as described in the `PostgreSQL
    documentation`__.

    __ https://www.postgresql.org/docs/current/static/sql-createtable.html#SQL-CREATETABLE-EXCLUDE

    """  # noqa

    __visit_name__ = "exclude_constraint"

    where = None
    inherit_cache = False

    create_drop_stringify_dialect = "postgresql"

    @elements._document_text_coercion(
        "where",
        ":class:`.ExcludeConstraint`",
        ":paramref:`.ExcludeConstraint.where`",
    )
    def __init__(self, *elements, **kw):
        r"""
        Create an :class:`.ExcludeConstraint` object.

        E.g.::

            const = ExcludeConstraint(
                (Column("period"), "&&"),
                (Column("group"), "="),
                where=(Column("group") != "some group"),
                ops={"group": "my_operator_class"},
            )

        The constraint is normally embedded into the :class:`_schema.Table`
        construct
        directly, or added later using :meth:`.append_constraint`::

            some_table = Table(
                "some_table",
                metadata,
                Column("id", Integer, primary_key=True),
                Column("period", TSRANGE()),
                Column("group", String),
            )

            some_table.append_constraint(
                ExcludeConstraint(
                    (some_table.c.period, "&&"),
                    (some_table.c.group, "="),
                    where=some_table.c.group != "some group",
                    name="some_table_excl_const",
                    ops={"group": "my_operator_class"},
                )
            )

        The exclude constraint defined in this example requires the
        ``btree_gist`` extension, that can be created using the
        command ``CREATE EXTENSION btree_gist;``.

        :param \*elements:

          A sequence of two tuples of the form ``(column, operator)`` where
          "column" is either a :class:`_schema.Column` object, or a SQL
          expression element (e.g. ``func.int8range(table.from, table.to)``)
          or the name of a column as string, and "operator" is a string
          containing the operator to use (e.g. `"&&"` or `"="`).

          In order to specify a column name when a :class:`_schema.Column`
          object is not available, while ensuring
          that any necessary quoting rules take effect, an ad-hoc
          :class:`_schema.Column` or :func:`_expression.column`
          object should be used.
          The ``column`` may also be a string SQL expression when
          passed as :func:`_expression.literal_column` or
          :func:`_expression.text`

        :param name:
          Optional, the in-database name of this constraint.

        :param deferrable:
          Optional bool.  If set, emit DEFERRABLE or NOT DEFERRABLE when
          issuing DDL for this constraint.

        :param initially:
          Optional string.  If set, emit INITIALLY <value> when issuing DDL
          for this constraint.

        :param using:
          Optional string.  If set, emit USING <index_method> when issuing DDL
          for this constraint. Defaults to 'gist'.

        :param where:
          Optional SQL expression construct or literal SQL string.
          If set, emit WHERE <predicate> when issuing DDL
          for this constraint.

        :param ops:
          Optional dictionary.  Used to define operator classes for the
          elements; works the same way as that of the
          :ref:`postgresql_ops <postgresql_operator_classes>`
          parameter specified to the :class:`_schema.Index` construct.

          .. versionadded:: 1.3.21

          .. seealso::

            :ref:`postgresql_operator_classes` - general description of how
            PostgreSQL operator classes are specified.

        """
        columns = []
        render_exprs = []
        self.operators = {}

        expressions, operators = zip(*elements)

        for (expr, column, strname, add_element), operator in zip(
            coercions.expect_col_expression_collection(
                roles.DDLConstraintColumnRole, expressions
            ),
            operators,
        ):
            if add_element is not None:
                columns.append(add_element)

            name = column.name if column is not None else strname

            if name is not None:
                # backwards compat
                self.operators[name] = operator

            render_exprs.append((expr, name, operator))

        self._render_exprs = render_exprs

        ColumnCollectionConstraint.__init__(
            self,
            *columns,
            name=kw.get("name"),
            deferrable=kw.get("deferrable"),
            initially=kw.get("initially"),
        )
        self.using = kw.get("using", "gist")
        where = kw.get("where")
        if where is not None:
            self.where = coercions.expect(roles.StatementOptionRole, where)

        self.ops = kw.get("ops", {})

    def _set_parent(self, table, **kw):
        super()._set_parent(table)

        self._render_exprs = [
            (
                expr if not isinstance(expr, str) else table.c[expr],
                name,
                operator,
            )
            for expr, name, operator in (self._render_exprs)
        ]

    def _copy(self, target_table=None, **kw):
        elements = [
            (
                schema._copy_expression(expr, self.parent, target_table),
                operator,
            )
            for expr, _, operator in self._render_exprs
        ]
        c = self.__class__(
            *elements,
            name=self.name,
            deferrable=self.deferrable,
            initially=self.initially,
            where=self.where,
            using=self.using,
        )
        c.dispatch._update(self.dispatch)
        return c


def array_agg(*arg, **kw):
    """PostgreSQL-specific form of :class:`_functions.array_agg`, ensures
    return type is :class:`_postgresql.ARRAY` and not
    the plain :class:`_types.ARRAY`, unless an explicit ``type_``
    is passed.

    """
    kw["_default_array_type"] = ARRAY
    return functions.func.array_agg(*arg, **kw)


class _regconfig_fn(functions.GenericFunction[_T]):
    inherit_cache = True

    def __init__(self, *args, **kwargs):
        args = list(args)
        if len(args) > 1:
            initial_arg = coercions.expect(
                roles.ExpressionElementRole,
                args.pop(0),
                name=getattr(self, "name", None),
                apply_propagate_attrs=self,
                type_=types.REGCONFIG,
            )
            initial_arg = [initial_arg]
        else:
            initial_arg = []

        addtl_args = [
            coercions.expect(
                roles.ExpressionElementRole,
                c,
                name=getattr(self, "name", None),
                apply_propagate_attrs=self,
            )
            for c in args
        ]
        super().__init__(*(initial_arg + addtl_args), **kwargs)


class to_tsvector(_regconfig_fn):
    """The PostgreSQL ``to_tsvector`` SQL function.

    This function applies automatic casting of the REGCONFIG argument
    to use the :class:`_postgresql.REGCONFIG` datatype automatically,
    and applies a return type of :class:`_postgresql.TSVECTOR`.

    Assuming the PostgreSQL dialect has been imported, either by invoking
    ``from sqlalchemy.dialects import postgresql``, or by creating a PostgreSQL
    engine using ``create_engine("postgresql...")``,
    :class:`_postgresql.to_tsvector` will be used automatically when invoking
    ``sqlalchemy.func.to_tsvector()``, ensuring the correct argument and return
    type handlers are used at compile and execution time.

    .. versionadded:: 2.0.0rc1

    """

    inherit_cache = True
    type = types.TSVECTOR


class to_tsquery(_regconfig_fn):
    """The PostgreSQL ``to_tsquery`` SQL function.

    This function applies automatic casting of the REGCONFIG argument
    to use the :class:`_postgresql.REGCONFIG` datatype automatically,
    and applies a return type of :class:`_postgresql.TSQUERY`.

    Assuming the PostgreSQL dialect has been imported, either by invoking
    ``from sqlalchemy.dialects import postgresql``, or by creating a PostgreSQL
    engine using ``create_engine("postgresql...")``,
    :class:`_postgresql.to_tsquery` will be used automatically when invoking
    ``sqlalchemy.func.to_tsquery()``, ensuring the correct argument and return
    type handlers are used at compile and execution time.

    .. versionadded:: 2.0.0rc1

    """

    inherit_cache = True
    type = types.TSQUERY


class plainto_tsquery(_regconfig_fn):
    """The PostgreSQL ``plainto_tsquery`` SQL function.

    This function applies automatic casting of the REGCONFIG argument
    to use the :class:`_postgresql.REGCONFIG` datatype automatically,
    and applies a return type of :class:`_postgresql.TSQUERY`.

    Assuming the PostgreSQL dialect has been imported, either by invoking
    ``from sqlalchemy.dialects import postgresql``, or by creating a PostgreSQL
    engine using ``create_engine("postgresql...")``,
    :class:`_postgresql.plainto_tsquery` will be used automatically when
    invoking ``sqlalchemy.func.plainto_tsquery()``, ensuring the correct
    argument and return type handlers are used at compile and execution time.

    .. versionadded:: 2.0.0rc1

    """

    inherit_cache = True
    type = types.TSQUERY


class phraseto_tsquery(_regconfig_fn):
    """The PostgreSQL ``phraseto_tsquery`` SQL function.

    This function applies automatic casting of the REGCONFIG argument
    to use the :class:`_postgresql.REGCONFIG` datatype automatically,
    and applies a return type of :class:`_postgresql.TSQUERY`.

    Assuming the PostgreSQL dialect has been imported, either by invoking
    ``from sqlalchemy.dialects import postgresql``, or by creating a PostgreSQL
    engine using ``create_engine("postgresql...")``,
    :class:`_postgresql.phraseto_tsquery` will be used automatically when
    invoking ``sqlalchemy.func.phraseto_tsquery()``, ensuring the correct
    argument and return type handlers are used at compile and execution time.

    .. versionadded:: 2.0.0rc1

    """

    inherit_cache = True
    type = types.TSQUERY


class websearch_to_tsquery(_regconfig_fn):
    """The PostgreSQL ``websearch_to_tsquery`` SQL function.

    This function applies automatic casting of the REGCONFIG argument
    to use the :class:`_postgresql.REGCONFIG` datatype automatically,
    and applies a return type of :class:`_postgresql.TSQUERY`.

    Assuming the PostgreSQL dialect has been imported, either by invoking
    ``from sqlalchemy.dialects import postgresql``, or by creating a PostgreSQL
    engine using ``create_engine("postgresql...")``,
    :class:`_postgresql.websearch_to_tsquery` will be used automatically when
    invoking ``sqlalchemy.func.websearch_to_tsquery()``, ensuring the correct
    argument and return type handlers are used at compile and execution time.

    .. versionadded:: 2.0.0rc1

    """

    inherit_cache = True
    type = types.TSQUERY


class ts_headline(_regconfig_fn):
    """The PostgreSQL ``ts_headline`` SQL function.

    This function applies automatic casting of the REGCONFIG argument
    to use the :class:`_postgresql.REGCONFIG` datatype automatically,
    and applies a return type of :class:`_types.TEXT`.

    Assuming the PostgreSQL dialect has been imported, either by invoking
    ``from sqlalchemy.dialects import postgresql``, or by creating a PostgreSQL
    engine using ``create_engine("postgresql...")``,
    :class:`_postgresql.ts_headline` will be used automatically when invoking
    ``sqlalchemy.func.ts_headline()``, ensuring the correct argument and return
    type handlers are used at compile and execution time.

    .. versionadded:: 2.0.0rc1

    """

    inherit_cache = True
    type = TEXT

    def __init__(self, *args, **kwargs):
        args = list(args)

        # parse types according to
        # https://www.postgresql.org/docs/current/textsearch-controls.html#TEXTSEARCH-HEADLINE
        if len(args) < 2:
            # invalid args; don't do anything
            has_regconfig = False
        elif (
            isinstance(args[1], elements.ColumnElement)
            and args[1].type._type_affinity is types.TSQUERY
        ):
            # tsquery is second argument, no regconfig argument
            has_regconfig = False
        else:
            has_regconfig = True

        if has_regconfig:
            initial_arg = coercions.expect(
                roles.ExpressionElementRole,
                args.pop(0),
                apply_propagate_attrs=self,
                name=getattr(self, "name", None),
                type_=types.REGCONFIG,
            )
            initial_arg = [initial_arg]
        else:
            initial_arg = []

        addtl_args = [
            coercions.expect(
                roles.ExpressionElementRole,
                c,
                name=getattr(self, "name", None),
                apply_propagate_attrs=self,
            )
            for c in args
        ]
        super().__init__(*(initial_arg + addtl_args), **kwargs)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\strings\object_array.py ===
from __future__ import annotations

import functools
import re
import textwrap
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
    cast,
)
import unicodedata
import warnings

import numpy as np

from pandas._libs import lib
import pandas._libs.missing as libmissing
import pandas._libs.ops as libops
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.missing import isna

from pandas.core.strings.base import BaseStringArrayMethods

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pandas._typing import (
        NpDtype,
        Scalar,
    )

    from pandas import Series


class ObjectStringArrayMixin(BaseStringArrayMethods):
    """
    String Methods operating on object-dtype ndarrays.
    """

    def __len__(self) -> int:
        # For typing, _str_map relies on the object being sized.
        raise NotImplementedError

    def _str_map(
        self,
        f,
        na_value=lib.no_default,
        dtype: NpDtype | None = None,
        convert: bool = True,
    ):
        """
        Map a callable over valid elements of the array.

        Parameters
        ----------
        f : Callable
            A function to call on each non-NA element.
        na_value : Scalar, optional
            The value to set for NA values. Might also be used for the
            fill value if the callable `f` raises an exception.
            This defaults to ``self.dtype.na_value`` which is ``np.nan``
            for object-dtype and Categorical and ``pd.NA`` for StringArray.
        dtype : Dtype, optional
            The dtype of the result array.
        convert : bool, default True
            Whether to call `maybe_convert_objects` on the resulting ndarray
        """
        if dtype is None:
            dtype = np.dtype("object")
        if na_value is lib.no_default:
            na_value = self.dtype.na_value  # type: ignore[attr-defined]

        if not len(self):
            return np.array([], dtype=dtype)

        arr = np.asarray(self, dtype=object)
        mask = isna(arr)
        map_convert = convert and not np.all(mask)
        try:
            result = lib.map_infer_mask(arr, f, mask.view(np.uint8), map_convert)
        except (TypeError, AttributeError) as err:
            # Reraise the exception if callable `f` got wrong number of args.
            # The user may want to be warned by this, instead of getting NaN
            p_err = (
                r"((takes)|(missing)) (?(2)from \d+ to )?\d+ "
                r"(?(3)required )positional arguments?"
            )

            if len(err.args) >= 1 and re.search(p_err, err.args[0]):
                # FIXME: this should be totally avoidable
                raise err

            def g(x):
                # This type of fallback behavior can be removed once
                # we remove object-dtype .str accessor.
                try:
                    return f(x)
                except (TypeError, AttributeError):
                    return na_value

            return self._str_map(g, na_value=na_value, dtype=dtype)
        if not isinstance(result, np.ndarray):
            return result
        if na_value is not np.nan:
            np.putmask(result, mask, na_value)
            if convert and result.dtype == object:
                result = lib.maybe_convert_objects(result)
        return result

    def _str_count(self, pat, flags: int = 0):
        regex = re.compile(pat, flags=flags)
        f = lambda x: len(regex.findall(x))
        return self._str_map(f, dtype="int64")

    def _str_pad(
        self,
        width: int,
        side: Literal["left", "right", "both"] = "left",
        fillchar: str = " ",
    ):
        if side == "left":
            f = lambda x: x.rjust(width, fillchar)
        elif side == "right":
            f = lambda x: x.ljust(width, fillchar)
        elif side == "both":
            f = lambda x: x.center(width, fillchar)
        else:  # pragma: no cover
            raise ValueError("Invalid side")
        return self._str_map(f)

    def _str_contains(
        self,
        pat,
        case: bool = True,
        flags: int = 0,
        na=lib.no_default,
        regex: bool = True,
    ):
        if regex:
            if not case:
                flags |= re.IGNORECASE

            pat = re.compile(pat, flags=flags)

            f = lambda x: pat.search(x) is not None
        else:
            if case:
                f = lambda x: pat in x
            else:
                upper_pat = pat.upper()
                f = lambda x: upper_pat in x.upper()
        if na is not lib.no_default and not isna(na) and not isinstance(na, bool):
            # GH#59561
            warnings.warn(
                "Allowing a non-bool 'na' in obj.str.contains is deprecated "
                "and will raise in a future version.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        return self._str_map(f, na, dtype=np.dtype("bool"))

    def _str_startswith(self, pat, na=lib.no_default):
        f = lambda x: x.startswith(pat)
        if na is not lib.no_default and not isna(na) and not isinstance(na, bool):
            # GH#59561
            warnings.warn(
                "Allowing a non-bool 'na' in obj.str.startswith is deprecated "
                "and will raise in a future version.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def _str_endswith(self, pat, na=lib.no_default):
        f = lambda x: x.endswith(pat)
        if na is not lib.no_default and not isna(na) and not isinstance(na, bool):
            # GH#59561
            warnings.warn(
                "Allowing a non-bool 'na' in obj.str.endswith is deprecated "
                "and will raise in a future version.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def _str_replace(
        self,
        pat: str | re.Pattern,
        repl: str | Callable,
        n: int = -1,
        case: bool = True,
        flags: int = 0,
        regex: bool = True,
    ):
        if case is False:
            # add case flag, if provided
            flags |= re.IGNORECASE

        if regex or flags or callable(repl):
            if not isinstance(pat, re.Pattern):
                if regex is False:
                    pat = re.escape(pat)
                pat = re.compile(pat, flags=flags)

            n = n if n >= 0 else 0
            f = lambda x: pat.sub(repl=repl, string=x, count=n)
        else:
            f = lambda x: x.replace(pat, repl, n)

        return self._str_map(f, dtype=str)

    def _str_repeat(self, repeats: int | Sequence[int]):
        if lib.is_integer(repeats):
            rint = cast(int, repeats)

            def scalar_rep(x):
                try:
                    return bytes.__mul__(x, rint)
                except TypeError:
                    return str.__mul__(x, rint)

            return self._str_map(scalar_rep, dtype=str)
        else:
            from pandas.core.arrays.string_ import BaseStringArray

            def rep(x, r):
                if x is libmissing.NA:
                    return x
                try:
                    return bytes.__mul__(x, r)
                except TypeError:
                    return str.__mul__(x, r)

            result = libops.vec_binop(
                np.asarray(self),
                np.asarray(repeats, dtype=object),
                rep,
            )
            if isinstance(self, BaseStringArray):
                # Not going through map, so we have to do this here.
                result = type(self)._from_sequence(result, dtype=self.dtype)
            return result

    def _str_match(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
    ):
        if not case:
            flags |= re.IGNORECASE

        regex = re.compile(pat, flags=flags)

        f = lambda x: regex.match(x) is not None
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def _str_fullmatch(
        self,
        pat: str | re.Pattern,
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
    ):
        if not case:
            flags |= re.IGNORECASE

        regex = re.compile(pat, flags=flags)

        f = lambda x: regex.fullmatch(x) is not None
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def _str_encode(self, encoding, errors: str = "strict"):
        f = lambda x: x.encode(encoding, errors=errors)
        return self._str_map(f, dtype=object)

    def _str_find(self, sub, start: int = 0, end=None):
        return self._str_find_(sub, start, end, side="left")

    def _str_rfind(self, sub, start: int = 0, end=None):
        return self._str_find_(sub, start, end, side="right")

    def _str_find_(self, sub, start, end, side):
        if side == "left":
            method = "find"
        elif side == "right":
            method = "rfind"
        else:  # pragma: no cover
            raise ValueError("Invalid side")

        if end is None:
            f = lambda x: getattr(x, method)(sub, start)
        else:
            f = lambda x: getattr(x, method)(sub, start, end)
        return self._str_map(f, dtype="int64")

    def _str_findall(self, pat, flags: int = 0):
        regex = re.compile(pat, flags=flags)
        return self._str_map(regex.findall, dtype="object")

    def _str_get(self, i):
        def f(x):
            if isinstance(x, dict):
                return x.get(i)
            elif len(x) > i >= -len(x):
                return x[i]
            return self.dtype.na_value  # type: ignore[attr-defined]

        return self._str_map(f)

    def _str_index(self, sub, start: int = 0, end=None):
        if end:
            f = lambda x: x.index(sub, start, end)
        else:
            f = lambda x: x.index(sub, start, end)
        return self._str_map(f, dtype="int64")

    def _str_rindex(self, sub, start: int = 0, end=None):
        if end:
            f = lambda x: x.rindex(sub, start, end)
        else:
            f = lambda x: x.rindex(sub, start, end)
        return self._str_map(f, dtype="int64")

    def _str_join(self, sep: str):
        return self._str_map(sep.join)

    def _str_partition(self, sep: str, expand):
        result = self._str_map(lambda x: x.partition(sep), dtype="object")
        return result

    def _str_rpartition(self, sep: str, expand):
        return self._str_map(lambda x: x.rpartition(sep), dtype="object")

    def _str_len(self):
        return self._str_map(len, dtype="int64")

    def _str_slice(self, start=None, stop=None, step=None):
        obj = slice(start, stop, step)
        return self._str_map(lambda x: x[obj])

    def _str_slice_replace(self, start=None, stop=None, repl=None):
        if repl is None:
            repl = ""

        def f(x):
            if x[start:stop] == "":
                local_stop = start
            else:
                local_stop = stop
            y = ""
            if start is not None:
                y += x[:start]
            y += repl
            if stop is not None:
                y += x[local_stop:]
            return y

        return self._str_map(f)

    def _str_split(
        self,
        pat: str | re.Pattern | None = None,
        n=-1,
        expand: bool = False,
        regex: bool | None = None,
    ):
        if pat is None:
            if n is None or n == 0:
                n = -1
            f = lambda x: x.split(pat, n)
        else:
            new_pat: str | re.Pattern
            if regex is True or isinstance(pat, re.Pattern):
                new_pat = re.compile(pat)
            elif regex is False:
                new_pat = pat
            # regex is None so link to old behavior #43563
            else:
                if len(pat) == 1:
                    new_pat = pat
                else:
                    new_pat = re.compile(pat)

            if isinstance(new_pat, re.Pattern):
                if n is None or n == -1:
                    n = 0
                f = lambda x: new_pat.split(x, maxsplit=n)
            else:
                if n is None or n == 0:
                    n = -1
                f = lambda x: x.split(pat, n)
        return self._str_map(f, dtype=object)

    def _str_rsplit(self, pat=None, n=-1):
        if n is None or n == 0:
            n = -1
        f = lambda x: x.rsplit(pat, n)
        return self._str_map(f, dtype="object")

    def _str_translate(self, table):
        return self._str_map(lambda x: x.translate(table))

    def _str_wrap(self, width: int, **kwargs):
        kwargs["width"] = width
        tw = textwrap.TextWrapper(**kwargs)
        return self._str_map(lambda s: "\n".join(tw.wrap(s)))

    def _str_get_dummies(self, sep: str = "|"):
        from pandas import Series

        arr = Series(self).fillna("")
        try:
            arr = sep + arr + sep
        except (TypeError, NotImplementedError):
            arr = sep + arr.astype(str) + sep

        tags: set[str] = set()
        for ts in Series(arr, copy=False).str.split(sep):
            tags.update(ts)
        tags2 = sorted(tags - {""})

        dummies = np.empty((len(arr), len(tags2)), dtype=np.int64)

        def _isin(test_elements: str, element: str) -> bool:
            return element in test_elements

        for i, t in enumerate(tags2):
            pat = sep + t + sep
            dummies[:, i] = lib.map_infer(
                arr.to_numpy(), functools.partial(_isin, element=pat)
            )
        return dummies, tags2

    def _str_upper(self):
        return self._str_map(lambda x: x.upper())

    def _str_isalnum(self):
        return self._str_map(str.isalnum, dtype="bool")

    def _str_isalpha(self):
        return self._str_map(str.isalpha, dtype="bool")

    def _str_isdecimal(self):
        return self._str_map(str.isdecimal, dtype="bool")

    def _str_isdigit(self):
        return self._str_map(str.isdigit, dtype="bool")

    def _str_islower(self):
        return self._str_map(str.islower, dtype="bool")

    def _str_isnumeric(self):
        return self._str_map(str.isnumeric, dtype="bool")

    def _str_isspace(self):
        return self._str_map(str.isspace, dtype="bool")

    def _str_istitle(self):
        return self._str_map(str.istitle, dtype="bool")

    def _str_isupper(self):
        return self._str_map(str.isupper, dtype="bool")

    def _str_capitalize(self):
        return self._str_map(str.capitalize)

    def _str_casefold(self):
        return self._str_map(str.casefold)

    def _str_title(self):
        return self._str_map(str.title)

    def _str_swapcase(self):
        return self._str_map(str.swapcase)

    def _str_lower(self):
        return self._str_map(str.lower)

    def _str_normalize(self, form):
        f = lambda x: unicodedata.normalize(form, x)
        return self._str_map(f)

    def _str_strip(self, to_strip=None):
        return self._str_map(lambda x: x.strip(to_strip))

    def _str_lstrip(self, to_strip=None):
        return self._str_map(lambda x: x.lstrip(to_strip))

    def _str_rstrip(self, to_strip=None):
        return self._str_map(lambda x: x.rstrip(to_strip))

    def _str_removeprefix(self, prefix: str) -> Series:
        # outstanding question on whether to use native methods for users on Python 3.9+
        # https://github.com/pandas-dev/pandas/pull/39226#issuecomment-836719770,
        # in which case we could do return self._str_map(str.removeprefix)

        def removeprefix(text: str) -> str:
            if text.startswith(prefix):
                return text[len(prefix) :]
            return text

        return self._str_map(removeprefix)

    def _str_removesuffix(self, suffix: str) -> Series:
        return self._str_map(lambda x: x.removesuffix(suffix))

    def _str_extract(self, pat: str, flags: int = 0, expand: bool = True):
        regex = re.compile(pat, flags=flags)
        na_value = self.dtype.na_value  # type: ignore[attr-defined]

        if not expand:

            def g(x):
                m = regex.search(x)
                return m.groups()[0] if m else na_value

            return self._str_map(g, convert=False)

        empty_row = [na_value] * regex.groups

        def f(x):
            if not isinstance(x, str):
                return empty_row
            m = regex.search(x)
            if m:
                return [na_value if item is None else item for item in m.groups()]
            else:
                return empty_row

        return [f(val) for val in np.asarray(self)]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\sansio\request.py ===
from __future__ import annotations

import typing as t
from datetime import datetime
from urllib.parse import parse_qsl

from ..datastructures import Accept
from ..datastructures import Authorization
from ..datastructures import CharsetAccept
from ..datastructures import ETags
from ..datastructures import Headers
from ..datastructures import HeaderSet
from ..datastructures import IfRange
from ..datastructures import ImmutableList
from ..datastructures import ImmutableMultiDict
from ..datastructures import LanguageAccept
from ..datastructures import MIMEAccept
from ..datastructures import MultiDict
from ..datastructures import Range
from ..datastructures import RequestCacheControl
from ..http import parse_accept_header
from ..http import parse_cache_control_header
from ..http import parse_date
from ..http import parse_etags
from ..http import parse_if_range_header
from ..http import parse_list_header
from ..http import parse_options_header
from ..http import parse_range_header
from ..http import parse_set_header
from ..user_agent import UserAgent
from ..utils import cached_property
from ..utils import header_property
from .http import parse_cookie
from .utils import get_content_length
from .utils import get_current_url
from .utils import get_host


class Request:
    """Represents the non-IO parts of a HTTP request, including the
    method, URL info, and headers.

    This class is not meant for general use. It should only be used when
    implementing WSGI, ASGI, or another HTTP application spec. Werkzeug
    provides a WSGI implementation at :cls:`werkzeug.wrappers.Request`.

    :param method: The method the request was made with, such as
        ``GET``.
    :param scheme: The URL scheme of the protocol the request used, such
        as ``https`` or ``wss``.
    :param server: The address of the server. ``(host, port)``,
        ``(path, None)`` for unix sockets, or ``None`` if not known.
    :param root_path: The prefix that the application is mounted under.
        This is prepended to generated URLs, but is not part of route
        matching.
    :param path: The path part of the URL after ``root_path``.
    :param query_string: The part of the URL after the "?".
    :param headers: The headers received with the request.
    :param remote_addr: The address of the client sending the request.

    .. versionchanged:: 3.0
        The ``charset``, ``url_charset``, and ``encoding_errors`` attributes
        were removed.

    .. versionadded:: 2.0
    """

    #: the class to use for `args` and `form`.  The default is an
    #: :class:`~werkzeug.datastructures.ImmutableMultiDict` which supports
    #: multiple values per key. A :class:`~werkzeug.datastructures.ImmutableDict`
    #: is faster but only remembers the last key. It is also
    #: possible to use mutable structures, but this is not recommended.
    #:
    #: .. versionadded:: 0.6
    parameter_storage_class: type[MultiDict[str, t.Any]] = ImmutableMultiDict

    #: The type to be used for dict values from the incoming WSGI
    #: environment. (For example for :attr:`cookies`.) By default an
    #: :class:`~werkzeug.datastructures.ImmutableMultiDict` is used.
    #:
    #: .. versionchanged:: 1.0.0
    #:     Changed to ``ImmutableMultiDict`` to support multiple values.
    #:
    #: .. versionadded:: 0.6
    dict_storage_class: type[MultiDict[str, t.Any]] = ImmutableMultiDict

    #: the type to be used for list values from the incoming WSGI environment.
    #: By default an :class:`~werkzeug.datastructures.ImmutableList` is used
    #: (for example for :attr:`access_list`).
    #:
    #: .. versionadded:: 0.6
    list_storage_class: type[list[t.Any]] = ImmutableList

    user_agent_class: type[UserAgent] = UserAgent
    """The class used and returned by the :attr:`user_agent` property to
    parse the header. Defaults to
    :class:`~werkzeug.user_agent.UserAgent`, which does no parsing. An
    extension can provide a subclass that uses a parser to provide other
    data.

    .. versionadded:: 2.0
    """

    #: Valid host names when handling requests. By default all hosts are
    #: trusted, which means that whatever the client says the host is
    #: will be accepted.
    #:
    #: Because ``Host`` and ``X-Forwarded-Host`` headers can be set to
    #: any value by a malicious client, it is recommended to either set
    #: this property or implement similar validation in the proxy (if
    #: the application is being run behind one).
    #:
    #: .. versionadded:: 0.9
    trusted_hosts: list[str] | None = None

    def __init__(
        self,
        method: str,
        scheme: str,
        server: tuple[str, int | None] | None,
        root_path: str,
        path: str,
        query_string: bytes,
        headers: Headers,
        remote_addr: str | None,
    ) -> None:
        #: The method the request was made with, such as ``GET``.
        self.method = method.upper()
        #: The URL scheme of the protocol the request used, such as
        #: ``https`` or ``wss``.
        self.scheme = scheme
        #: The address of the server. ``(host, port)``, ``(path, None)``
        #: for unix sockets, or ``None`` if not known.
        self.server = server
        #: The prefix that the application is mounted under, without a
        #: trailing slash. :attr:`path` comes after this.
        self.root_path = root_path.rstrip("/")
        #: The path part of the URL after :attr:`root_path`. This is the
        #: path used for routing within the application.
        self.path = "/" + path.lstrip("/")
        #: The part of the URL after the "?". This is the raw value, use
        #: :attr:`args` for the parsed values.
        self.query_string = query_string
        #: The headers received with the request.
        self.headers = headers
        #: The address of the client sending the request.
        self.remote_addr = remote_addr

    def __repr__(self) -> str:
        try:
            url = self.url
        except Exception as e:
            url = f"(invalid URL: {e})"

        return f"<{type(self).__name__} {url!r} [{self.method}]>"

    @cached_property
    def args(self) -> MultiDict[str, str]:
        """The parsed URL parameters (the part in the URL after the question
        mark).

        By default an
        :class:`~werkzeug.datastructures.ImmutableMultiDict`
        is returned from this function.  This can be changed by setting
        :attr:`parameter_storage_class` to a different type.  This might
        be necessary if the order of the form data is important.

        .. versionchanged:: 2.3
            Invalid bytes remain percent encoded.
        """
        return self.parameter_storage_class(
            parse_qsl(
                self.query_string.decode(),
                keep_blank_values=True,
                errors="werkzeug.url_quote",
            )
        )

    @cached_property
    def access_route(self) -> list[str]:
        """If a forwarded header exists this is a list of all ip addresses
        from the client ip to the last proxy server.
        """
        if "X-Forwarded-For" in self.headers:
            return self.list_storage_class(
                parse_list_header(self.headers["X-Forwarded-For"])
            )
        elif self.remote_addr is not None:
            return self.list_storage_class([self.remote_addr])
        return self.list_storage_class()

    @cached_property
    def full_path(self) -> str:
        """Requested path, including the query string."""
        return f"{self.path}?{self.query_string.decode()}"

    @property
    def is_secure(self) -> bool:
        """``True`` if the request was made with a secure protocol
        (HTTPS or WSS).
        """
        return self.scheme in {"https", "wss"}

    @cached_property
    def url(self) -> str:
        """The full request URL with the scheme, host, root path, path,
        and query string."""
        return get_current_url(
            self.scheme, self.host, self.root_path, self.path, self.query_string
        )

    @cached_property
    def base_url(self) -> str:
        """Like :attr:`url` but without the query string."""
        return get_current_url(self.scheme, self.host, self.root_path, self.path)

    @cached_property
    def root_url(self) -> str:
        """The request URL scheme, host, and root path. This is the root
        that the application is accessed from.
        """
        return get_current_url(self.scheme, self.host, self.root_path)

    @cached_property
    def host_url(self) -> str:
        """The request URL scheme and host only."""
        return get_current_url(self.scheme, self.host)

    @cached_property
    def host(self) -> str:
        """The host name the request was made to, including the port if
        it's non-standard. Validated with :attr:`trusted_hosts`.
        """
        return get_host(
            self.scheme, self.headers.get("host"), self.server, self.trusted_hosts
        )

    @cached_property
    def cookies(self) -> ImmutableMultiDict[str, str]:
        """A :class:`dict` with the contents of all cookies transmitted with
        the request."""
        wsgi_combined_cookie = ";".join(self.headers.getlist("Cookie"))
        return parse_cookie(  # type: ignore
            wsgi_combined_cookie, cls=self.dict_storage_class
        )

    # Common Descriptors

    content_type = header_property[str](
        "Content-Type",
        doc="""The Content-Type entity-header field indicates the media
        type of the entity-body sent to the recipient or, in the case of
        the HEAD method, the media type that would have been sent had
        the request been a GET.""",
        read_only=True,
    )

    @cached_property
    def content_length(self) -> int | None:
        """The Content-Length entity-header field indicates the size of the
        entity-body in bytes or, in the case of the HEAD method, the size of
        the entity-body that would have been sent had the request been a
        GET.
        """
        return get_content_length(
            http_content_length=self.headers.get("Content-Length"),
            http_transfer_encoding=self.headers.get("Transfer-Encoding"),
        )

    content_encoding = header_property[str](
        "Content-Encoding",
        doc="""The Content-Encoding entity-header field is used as a
        modifier to the media-type. When present, its value indicates
        what additional content codings have been applied to the
        entity-body, and thus what decoding mechanisms must be applied
        in order to obtain the media-type referenced by the Content-Type
        header field.

        .. versionadded:: 0.9""",
        read_only=True,
    )
    content_md5 = header_property[str](
        "Content-MD5",
        doc="""The Content-MD5 entity-header field, as defined in
        RFC 1864, is an MD5 digest of the entity-body for the purpose of
        providing an end-to-end message integrity check (MIC) of the
        entity-body. (Note: a MIC is good for detecting accidental
        modification of the entity-body in transit, but is not proof
        against malicious attacks.)

        .. versionadded:: 0.9""",
        read_only=True,
    )
    referrer = header_property[str](
        "Referer",
        doc="""The Referer[sic] request-header field allows the client
        to specify, for the server's benefit, the address (URI) of the
        resource from which the Request-URI was obtained (the
        "referrer", although the header field is misspelled).""",
        read_only=True,
    )
    date = header_property(
        "Date",
        None,
        parse_date,
        doc="""The Date general-header field represents the date and
        time at which the message was originated, having the same
        semantics as orig-date in RFC 822.

        .. versionchanged:: 2.0
            The datetime object is timezone-aware.
        """,
        read_only=True,
    )
    max_forwards = header_property(
        "Max-Forwards",
        None,
        int,
        doc="""The Max-Forwards request-header field provides a
        mechanism with the TRACE and OPTIONS methods to limit the number
        of proxies or gateways that can forward the request to the next
        inbound server.""",
        read_only=True,
    )

    def _parse_content_type(self) -> None:
        if not hasattr(self, "_parsed_content_type"):
            self._parsed_content_type = parse_options_header(
                self.headers.get("Content-Type", "")
            )

    @property
    def mimetype(self) -> str:
        """Like :attr:`content_type`, but without parameters (eg, without
        charset, type etc.) and always lowercase.  For example if the content
        type is ``text/HTML; charset=utf-8`` the mimetype would be
        ``'text/html'``.
        """
        self._parse_content_type()
        return self._parsed_content_type[0].lower()

    @property
    def mimetype_params(self) -> dict[str, str]:
        """The mimetype parameters as dict.  For example if the content
        type is ``text/html; charset=utf-8`` the params would be
        ``{'charset': 'utf-8'}``.
        """
        self._parse_content_type()
        return self._parsed_content_type[1]

    @cached_property
    def pragma(self) -> HeaderSet:
        """The Pragma general-header field is used to include
        implementation-specific directives that might apply to any recipient
        along the request/response chain.  All pragma directives specify
        optional behavior from the viewpoint of the protocol; however, some
        systems MAY require that behavior be consistent with the directives.
        """
        return parse_set_header(self.headers.get("Pragma", ""))

    # Accept

    @cached_property
    def accept_mimetypes(self) -> MIMEAccept:
        """List of mimetypes this client supports as
        :class:`~werkzeug.datastructures.MIMEAccept` object.
        """
        return parse_accept_header(self.headers.get("Accept"), MIMEAccept)

    @cached_property
    def accept_charsets(self) -> CharsetAccept:
        """List of charsets this client supports as
        :class:`~werkzeug.datastructures.CharsetAccept` object.
        """
        return parse_accept_header(self.headers.get("Accept-Charset"), CharsetAccept)

    @cached_property
    def accept_encodings(self) -> Accept:
        """List of encodings this client accepts.  Encodings in a HTTP term
        are compression encodings such as gzip.  For charsets have a look at
        :attr:`accept_charset`.
        """
        return parse_accept_header(self.headers.get("Accept-Encoding"))

    @cached_property
    def accept_languages(self) -> LanguageAccept:
        """List of languages this client accepts as
        :class:`~werkzeug.datastructures.LanguageAccept` object.

        .. versionchanged 0.5
           In previous versions this was a regular
           :class:`~werkzeug.datastructures.Accept` object.
        """
        return parse_accept_header(self.headers.get("Accept-Language"), LanguageAccept)

    # ETag

    @cached_property
    def cache_control(self) -> RequestCacheControl:
        """A :class:`~werkzeug.datastructures.RequestCacheControl` object
        for the incoming cache control headers.
        """
        cache_control = self.headers.get("Cache-Control")
        return parse_cache_control_header(cache_control, None, RequestCacheControl)

    @cached_property
    def if_match(self) -> ETags:
        """An object containing all the etags in the `If-Match` header.

        :rtype: :class:`~werkzeug.datastructures.ETags`
        """
        return parse_etags(self.headers.get("If-Match"))

    @cached_property
    def if_none_match(self) -> ETags:
        """An object containing all the etags in the `If-None-Match` header.

        :rtype: :class:`~werkzeug.datastructures.ETags`
        """
        return parse_etags(self.headers.get("If-None-Match"))

    @cached_property
    def if_modified_since(self) -> datetime | None:
        """The parsed `If-Modified-Since` header as a datetime object.

        .. versionchanged:: 2.0
            The datetime object is timezone-aware.
        """
        return parse_date(self.headers.get("If-Modified-Since"))

    @cached_property
    def if_unmodified_since(self) -> datetime | None:
        """The parsed `If-Unmodified-Since` header as a datetime object.

        .. versionchanged:: 2.0
            The datetime object is timezone-aware.
        """
        return parse_date(self.headers.get("If-Unmodified-Since"))

    @cached_property
    def if_range(self) -> IfRange:
        """The parsed ``If-Range`` header.

        .. versionchanged:: 2.0
            ``IfRange.date`` is timezone-aware.

        .. versionadded:: 0.7
        """
        return parse_if_range_header(self.headers.get("If-Range"))

    @cached_property
    def range(self) -> Range | None:
        """The parsed `Range` header.

        .. versionadded:: 0.7

        :rtype: :class:`~werkzeug.datastructures.Range`
        """
        return parse_range_header(self.headers.get("Range"))

    # User Agent

    @cached_property
    def user_agent(self) -> UserAgent:
        """The user agent. Use ``user_agent.string`` to get the header
        value. Set :attr:`user_agent_class` to a subclass of
        :class:`~werkzeug.user_agent.UserAgent` to provide parsing for
        the other properties or other extended data.

        .. versionchanged:: 2.1
            The built-in parser was removed. Set ``user_agent_class`` to a ``UserAgent``
            subclass to parse data from the string.
        """
        return self.user_agent_class(self.headers.get("User-Agent", ""))

    # Authorization

    @cached_property
    def authorization(self) -> Authorization | None:
        """The ``Authorization`` header parsed into an :class:`.Authorization` object.
        ``None`` if the header is not present.

        .. versionchanged:: 2.3
            :class:`Authorization` is no longer a ``dict``. The ``token`` attribute
            was added for auth schemes that use a token instead of parameters.
        """
        return Authorization.from_header(self.headers.get("Authorization"))

    # CORS

    origin = header_property[str](
        "Origin",
        doc=(
            "The host that the request originated from. Set"
            " :attr:`~CORSResponseMixin.access_control_allow_origin` on"
            " the response to indicate which origins are allowed."
        ),
        read_only=True,
    )

    access_control_request_headers = header_property(
        "Access-Control-Request-Headers",
        load_func=parse_set_header,
        doc=(
            "Sent with a preflight request to indicate which headers"
            " will be sent with the cross origin request. Set"
            " :attr:`~CORSResponseMixin.access_control_allow_headers`"
            " on the response to indicate which headers are allowed."
        ),
        read_only=True,
    )

    access_control_request_method = header_property[str](
        "Access-Control-Request-Method",
        doc=(
            "Sent with a preflight request to indicate which method"
            " will be used for the cross origin request. Set"
            " :attr:`~CORSResponseMixin.access_control_allow_methods`"
            " on the response to indicate which methods are allowed."
        ),
        read_only=True,
    )

    @property
    def is_json(self) -> bool:
        """Check if the mimetype indicates JSON data, either
        :mimetype:`application/json` or :mimetype:`application/*+json`.
        """
        mt = self.mimetype
        return (
            mt == "application/json"
            or mt.startswith("application/")
            and mt.endswith("+json")
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\fusion_attention_sam2.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger

import numpy as np
from fusion_base import Fusion
from fusion_utils import NumpyHelper
from onnx import NodeProto, helper, numpy_helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionMultiHeadAttentionSam2(Fusion):
    """
    Fuse MultiHeadAttention subgraph of Segment Anything v2 (SAM2).
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
    ):
        super().__init__(model, "MultiHeadAttention", ["LayerNormalization"])
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Flags to show warning only once
        self.num_heads_warning = True
        self.hidden_size_warning = True

    def get_decoder_num_heads(self, reshape_q: NodeProto) -> int:
        """Detect num_heads from a reshape node.

        Args:
            reshape_q (NodeProto): reshape node for Q
        Returns:
            int: num_heads, or 0 if not found
        """
        num_heads = 0

        # we assume that reshape fusion has done, so the shape is a tensor like [0, 0, num_heads, head_size]
        shape_value = self.model.get_constant_value(reshape_q.input[1])
        if shape_value is not None:
            if isinstance(shape_value, np.ndarray) and list(shape_value.shape) == [4]:
                num_heads = int(shape_value[2])

        if isinstance(num_heads, int) and num_heads > 0:
            return num_heads

        return 0

    def get_encoder_num_heads(self, reshape_in: NodeProto) -> int:
        """Detect num_heads from a reshape node.

        Args:
            reshape_q (NodeProto): reshape node for Q
        Returns:
            int: num_heads, or 0 if not found
        """
        num_heads = 0

        shape_value = self.model.get_constant_value(reshape_in.input[1])
        if shape_value is not None:
            if isinstance(shape_value, np.ndarray) and list(shape_value.shape) == [5]:
                num_heads = int(shape_value[3])
        else:
            concat_shape = self.model.match_parent(reshape_in, "Concat", 1)
            if concat_shape is not None and len(concat_shape.input) == 5:
                # we assume that reshape fusion has done, so the shape is a tensor like [0, 0, num_heads, head_size]
                shape_value = self.model.get_constant_value(concat_shape.input[3])
                if shape_value is not None:
                    if isinstance(shape_value, np.ndarray) and list(shape_value.shape) == [1]:
                        num_heads = int(shape_value[0])

        if isinstance(num_heads, int) and num_heads > 0:
            return num_heads

        return 0

    def get_hidden_size(self, layernorm_node):
        """Detect hidden_size from LayerNormalization node.
        Args:
            layernorm_node (NodeProto): LayerNormalization node before Q, K and V
        Returns:
            int: hidden_size, or 0 if not found
        """
        layernorm_bias = self.model.get_initializer(layernorm_node.input[2])
        if layernorm_bias:
            return NumpyHelper.to_array(layernorm_bias).shape[0]

        return 0

    def get_num_heads_and_hidden_size(
        self, reshape_q: NodeProto, layernorm_node: NodeProto, is_encoder: bool = False
    ) -> tuple[int, int]:
        """Detect num_heads and hidden_size.

        Args:
            reshape_q (NodeProto): reshape node for Q
            layernorm_node (NodeProto): LayerNormalization node before Q, K, V
        Returns:
            Tuple[int, int]: num_heads and hidden_size
        """
        if is_encoder:
            num_heads = self.get_encoder_num_heads(reshape_q)
        else:
            num_heads = self.get_decoder_num_heads(reshape_q)
        if num_heads <= 0:
            num_heads = self.num_heads  # Fall back to user specified value

        if self.num_heads > 0 and num_heads != self.num_heads:
            if self.num_heads_warning:
                logger.warning(f"--num_heads is {self.num_heads}. Detected value is {num_heads}. Using detected value.")
                self.num_heads_warning = False  # Do not show the warning more than once

        hidden_size = self.get_hidden_size(layernorm_node)
        if hidden_size <= 0:
            hidden_size = self.hidden_size  # Fall back to user specified value

        if self.hidden_size > 0 and hidden_size != self.hidden_size:
            if self.hidden_size_warning:
                logger.warning(
                    f"--hidden_size is {self.hidden_size}. Detected value is {hidden_size}. Using detected value."
                )
                self.hidden_size_warning = False  # Do not show the warning more than once

        return num_heads, hidden_size

    def create_attention_node(
        self,
        q_matmul: NodeProto,
        q_add: NodeProto,
        k_matmul: NodeProto,
        k_add: NodeProto,
        v_matmul: NodeProto,
        v_add: NodeProto,
        num_heads: int,
        hidden_size: int,
        output: str,
    ) -> NodeProto | None:
        """Create an Attention node.

        Args:
            q_matmul (NodeProto): MatMul node in fully connection for Q
            q_add (NodeProto): Add bias node in fully connection for Q
            k_matmul (NodeProto): MatMul node in fully connection for K
            k_add (NodeProto): Add bias node in fully connection for K
            v_matmul (NodeProto): MatMul node in fully connection for V
            v_add (NodeProto): Add bias node in fully connection for V
            num_heads (int): number of attention heads. If a model is pruned, it is the number of heads after pruning.
            hidden_size (int): hidden dimension. If a model is pruned, it is the hidden dimension after pruning.
            output (str): output name

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        if hidden_size > 0 and (hidden_size % num_heads) != 0:
            logger.debug(f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}")
            return None

        q_weight = self.model.get_initializer(q_matmul.input[1])
        k_weight = self.model.get_initializer(k_matmul.input[1])
        v_weight = self.model.get_initializer(v_matmul.input[1])
        if not (q_weight and k_weight and v_weight):
            return None

        qw = NumpyHelper.to_array(q_weight)
        kw = NumpyHelper.to_array(k_weight)
        vw = NumpyHelper.to_array(v_weight)
        logger.debug(f"qw={qw.shape} kw={kw.shape} vw={vw.shape} hidden_size={hidden_size}")

        attention_node_name = self.model.create_node_name("MultiHeadAttention")

        attention_inputs = [
            q_add.output[0],
            k_add.output[0],
            v_add.output[0],
        ]

        attention_node = helper.make_node(
            "MultiHeadAttention",
            inputs=attention_inputs,
            outputs=[output],
            name=attention_node_name,
        )
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])

        counter_name = "MultiHeadAttention ({})".format("cross attention")
        self.increase_counter(counter_name)
        return attention_node

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        if self.fuse_sam_encoder_pattern(normalize_node, input_name_to_nodes, output_name_to_node):
            return

        match_qkv = self.match_attention_subgraph(normalize_node)
        if match_qkv is None:
            if normalize_node.input[0] not in output_name_to_node:
                return

            skip_add = output_name_to_node[normalize_node.input[0]]
            if skip_add.op_type != "Add":
                return

            match_qkv = self.match_attention_subgraph(skip_add)

            if match_qkv is None:
                return

        reshape_qkv, transpose_qkv, reshape_q, matmul_q, add_q, matmul_k, add_k, matmul_v, add_v = match_qkv

        attention_last_node = reshape_qkv

        q_num_heads, q_hidden_size = self.get_num_heads_and_hidden_size(reshape_q, normalize_node, False)
        if q_num_heads <= 0:
            logger.debug("fuse_attention: failed to detect num_heads")
            return

        # number of heads are same for all the paths, hence to create attention node, we pass the q_num_heads
        new_node = self.create_attention_node(
            matmul_q,
            add_q,
            matmul_k,
            add_k,
            matmul_v,
            add_v,
            q_num_heads,
            q_hidden_size,
            output=attention_last_node.output[0],
        )
        if new_node is None:
            return

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

        self.nodes_to_remove.extend([attention_last_node, transpose_qkv])

        # Use prune graph to remove nodes since they are shared by all attention nodes.
        self.prune_graph = True

    def match_attention_subgraph(self, node_after_output_projection):
        """Match Q, K and V paths exported by PyTorch 2.*"""
        qkv_nodes = self.model.match_parent_path(
            node_after_output_projection,
            ["Add", "MatMul", "Reshape", "Transpose", "MatMul"],
            [None, None, None, 0, 0],
        )

        if qkv_nodes is None:
            return None

        (_, _, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes

        v_nodes = self.model.match_parent_path(matmul_qkv, ["Transpose", "Reshape", "Add", "MatMul"], [1, 0, 0, None])
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return None
        (_, _, add_v, matmul_v) = v_nodes

        qk_nodes = self.model.match_parent_path(matmul_qkv, ["Softmax", "MatMul"], [0, 0])
        if qk_nodes is not None:
            (_softmax_qk, matmul_qk) = qk_nodes
        else:
            logger.debug("fuse_attention: failed to match qk path")
            return None

        q_nodes = self.model.match_parent_path(
            matmul_qk, ["Mul", "Transpose", "Reshape", "Add", "MatMul"], [0, None, 0, 0, None]
        )
        if q_nodes is None:
            logger.debug("fuse_attention: failed to match q path")
            return None
        (mul_q, _transpose_q, reshape_q, add_q, matmul_q) = q_nodes

        k_nodes = self.model.match_parent_path(
            matmul_qk, ["Mul", "Transpose", "Reshape", "Add", "MatMul"], [1, None, 0, 0, None]
        )
        if k_nodes is None:
            logger.debug("fuse_attention: failed to match k path")
            return None

        (_mul_k, _, _, add_k, matmul_k) = k_nodes

        # The scalar for Q and K is sqrt(1.0/sqrt(head_size)).
        mul_q_nodes = self.model.match_parent_path(
            mul_q,
            ["Sqrt", "Div", "Sqrt", "Cast", "Slice", "Shape", "Transpose", "Reshape"],
            [None, 0, 1, 0, 0, 0, 0, 0],
        )
        if mul_q_nodes is None or mul_q_nodes[-1] != reshape_q:
            logger.debug("fuse_attention: failed to match mul_q path")
            return None

        return reshape_qkv, transpose_qkv, reshape_q, matmul_q, add_q, matmul_k, add_k, matmul_v, add_v

    # --------------------------------------------------------
    # The following are for SAM encoder
    # --------------------------------------------------------
    def fuse_sam_encoder_pattern(self, normalize_node, input_name_to_nodes, output_name_to_node) -> bool:
        # SAM encoder attention layer pattern:
        #           Add -----------+
        #            |             |
        #        LayerNorm         |
        #            |             |
        #        Reshape           |
        #            |             |
        #        Transpose         |
        #            |             |
        #        MatMul            |
        #            |             |
        #           Add            |
        #            |             |
        #         Reshape          |
        #            |             |
        #          Split           |
        #            |             |
        #  Self Attention subgraph |
        #            |             |
        #        Reshape           |
        #            |             |
        #        Transpose         |
        #            |             |
        #        Reshape           |
        #            |             |
        #            Add ----------+
        #            |
        #         LayerNorm (starts from here)

        nodes = self.model.match_parent_path(
            normalize_node,
            ["Add", "Reshape", "Transpose", "Reshape"],
            [0, None, 0, 0],
        )
        if nodes is None:
            nodes = self.model.match_parent_path(
                normalize_node,
                ["Add", "Slice", "Slice", "Reshape", "Transpose", "Reshape"],
                [0, None, 0, 0, 0, 0],
            )
        if nodes is None:
            nodes = self.model.match_parent_path(
                normalize_node,
                ["Add"],
                [0],
            )
        if nodes is None:
            return False

        node_after_output_projection = nodes[-1]
        matched_sdpa = self.match_sam_encoder_attention_subgraph(
            node_after_output_projection, input_index=1 if len(nodes) == 1 else None
        )
        if matched_sdpa is None:
            return False

        reshape_out, transpose_out, split_qkv, transpose_q, transpose_k, transpose_v = matched_sdpa

        # B, S, N, H => B, N, S, H
        permutation_q = OnnxModel.get_node_attribute(transpose_q, "perm")
        if (not isinstance(permutation_q, list)) or permutation_q != [0, 2, 1, 3]:
            return False

        # B, S, N, H => B, N, H, S
        permutation_k = OnnxModel.get_node_attribute(transpose_k, "perm")
        if (not isinstance(permutation_k, list)) or permutation_k != [0, 2, 3, 1]:
            return False

        # B, S, N, H => B, N, S, H
        permutation_v = OnnxModel.get_node_attribute(transpose_v, "perm")
        if (not isinstance(permutation_v, list)) or permutation_v != [0, 2, 1, 3]:
            return False

        input_projection_nodes = self.model.match_parent_path(
            split_qkv,
            ["Reshape", "Add", "MatMul"],
            [0, 0, None],
        )
        if input_projection_nodes is None:
            return False
        reshape_in, add_in, matmul_in = input_projection_nodes
        q_num_heads, q_hidden_size = self.get_num_heads_and_hidden_size(reshape_in, normalize_node, True)
        if q_num_heads <= 0:
            logger.debug("fuse_attention: failed to detect num_heads")
            return False

        # Add a shape to convert 4D BxSxNxH to 3D BxSxD, which is required by MHA operator.
        new_dims_name = "bsnh_to_bsd_reshape_dims"
        new_dims = self.model.get_initializer(new_dims_name)
        if new_dims is None:
            new_dims = numpy_helper.from_array(np.array([0, 0, -1], dtype="int64"), name=new_dims_name)
            self.model.add_initializer(new_dims, self.this_graph_name)
        reshape_q_name = self.model.create_node_name("Reshape")
        reshape_q = helper.make_node(
            "Reshape",
            inputs=[transpose_q.input[0], new_dims_name],
            outputs=[transpose_q.input[0] + "_BSD"],
            name=reshape_q_name,
        )
        self.nodes_to_add.append(reshape_q)
        self.node_name_to_graph_name[reshape_q.name] = self.this_graph_name

        # Reuse the transpose_q node to transpose K from BSNH to BNSH. Here we update the input and output of the node.
        transpose_k_bnsh = transpose_q
        transpose_k_bnsh.input[0] = transpose_k.input[0]
        transpose_k_bnsh.output[0] = transpose_k.input[0] + "_BNSH"

        logger.debug(f"Found MHA: {q_num_heads=} {q_hidden_size=}")

        # number of heads are same for all the paths, hence to create attention node, we pass the q_num_heads
        new_node = self.create_mha_node(
            reshape_q,
            transpose_k_bnsh,
            transpose_v,
            q_num_heads,
        )
        if new_node is None:
            return False

        # Update the input of the next node that consumes the output of the MHA.
        assert len(self.model.get_children(transpose_out, input_name_to_nodes)) == 1
        reshape_out.input[0] = new_node.output[0]

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name
        self.nodes_to_remove.extend([transpose_out])

        # Use prune graph to remove nodes since they are shared by all attention nodes.
        self.prune_graph = True
        return True

    def match_sam_encoder_attention_subgraph(self, node_after_output_projection, input_index=None):
        """Match SDPA pattern in SAM2 enconder.*"""

        # nodes of output projection and the second MatMul in SDPA.
        out_nodes = self.model.match_parent_path(
            node_after_output_projection,
            ["Add", "MatMul", "Reshape", "Transpose", "MatMul"],
            [input_index, None, None, 0, 0],
        )

        if out_nodes is None:
            return None

        (_, _, reshape_out, transpose_out, matmul_qk_v) = out_nodes

        # Split and Reshape is for packed QKV
        v_nodes = self.model.match_parent_path(matmul_qk_v, ["Transpose", "Squeeze", "Split", "Reshape"], [1, 0, 0, 0])
        if v_nodes is None:
            logger.debug("failed to match v path")
            return None
        (transpose_v, _, split_qkv, reshape_qkv) = v_nodes

        qk_nodes = self.model.match_parent_path(matmul_qk_v, ["Softmax", "MatMul"], [0, 0])
        if qk_nodes is not None:
            (_softmax_qk, matmul_qk) = qk_nodes
        else:
            logger.debug("failed to match qk path")
            return None

        q_nodes = self.model.match_parent_path(matmul_qk, ["Mul", "Transpose", "Squeeze", "Split"], [0, None, 0, 0])
        if q_nodes is None:
            q_nodes = self.model.match_parent_path(
                matmul_qk,
                ["Mul", "Transpose", "Reshape", "Transpose", "MaxPool", "Transpose", "Reshape", "Squeeze", "Split"],
                [0, None, 0, 0, 0, 0, 0, 0, 0],
            )
            if q_nodes is None:
                logger.debug("failed to match q path")
                return None

        if q_nodes[-1] != split_qkv:
            return None
        transpose_q = q_nodes[1]

        k_nodes = self.model.match_parent_path(matmul_qk, ["Mul", "Transpose", "Squeeze", "Split"], [1, None, 0, 0])
        if k_nodes is None:
            logger.debug("failed to match k path")
            return None

        if k_nodes[-1] != split_qkv:
            return None
        (mul_k, transpose_k, _squeeze_k, _) = k_nodes

        return reshape_out, transpose_out, split_qkv, transpose_q, transpose_k, transpose_v

    def create_mha_node(
        self,
        reshape_q: NodeProto,
        transpose_k: NodeProto,
        transpose_v: NodeProto,
        num_heads: int,
    ) -> NodeProto:
        """Create a MultiHeadAttention node for SAM2 encoder.

        Args:
            reshape_q (NodeProto): Reshape node for Q, output is 3D BxSxNH format
            transpose_k (NodeProto): Transpose node for K, output is BNSH format
            transpose_v (NodeProto): Transpose node for V, output is BNSH format
            num_heads (int): number of attention heads. If a model is pruned, it is the number of heads after pruning.

        Returns:
            NodeProto: the MultiHeadAttention node created.
        """

        attention_node_name = self.model.create_node_name("MultiHeadAttention")

        inputs = [
            reshape_q.output[0],
            transpose_k.output[0],
            transpose_v.output[0],
        ]

        # Create a new output name since the shape is 3D, which is different from the original output shape (4D).
        output = attention_node_name + "_out"

        attention_node = helper.make_node(
            "MultiHeadAttention",
            inputs=inputs,
            outputs=[output],
            name=attention_node_name,
        )
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])

        counter_name = "MultiHeadAttention ({})".format("self attention")
        self.increase_counter(counter_name)
        return attention_node

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\sets\handlers\intersection.py ===
from sympy.core.basic import _aresame
from sympy.core.function import Lambda, expand_complex
from sympy.core.mul import Mul
from sympy.core.numbers import ilcm, Float
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.core.sorting import ordered
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.integers import floor, ceiling
from sympy.sets.fancysets import ComplexRegion
from sympy.sets.sets import (FiniteSet, Intersection, Interval, Set, Union)
from sympy.multipledispatch import Dispatcher
from sympy.sets.conditionset import ConditionSet
from sympy.sets.fancysets import (Integers, Naturals, Reals, Range,
    ImageSet, Rationals)
from sympy.sets.sets import EmptySet, UniversalSet, imageset, ProductSet
from sympy.simplify.radsimp import numer


intersection_sets = Dispatcher('intersection_sets')


@intersection_sets.register(ConditionSet, ConditionSet)
def _(a, b):
    return None

@intersection_sets.register(ConditionSet, Set)
def _(a, b):
    return ConditionSet(a.sym, a.condition, Intersection(a.base_set, b))

@intersection_sets.register(Naturals, Integers)
def _(a, b):
    return a

@intersection_sets.register(Naturals, Naturals)
def _(a, b):
    return a if a is S.Naturals else b

@intersection_sets.register(Interval, Naturals)
def _(a, b):
    return intersection_sets(b, a)

@intersection_sets.register(ComplexRegion, Set)
def _(self, other):
    if other.is_ComplexRegion:
        # self in rectangular form
        if (not self.polar) and (not other.polar):
            return ComplexRegion(Intersection(self.sets, other.sets))

        # self in polar form
        elif self.polar and other.polar:
            r1, theta1 = self.a_interval, self.b_interval
            r2, theta2 = other.a_interval, other.b_interval
            new_r_interval = Intersection(r1, r2)
            new_theta_interval = Intersection(theta1, theta2)

            # 0 and 2*Pi means the same
            if ((2*S.Pi in theta1 and S.Zero in theta2) or
               (2*S.Pi in theta2 and S.Zero in theta1)):
                new_theta_interval = Union(new_theta_interval,
                                           FiniteSet(0))
            return ComplexRegion(new_r_interval*new_theta_interval,
                                polar=True)


    if other.is_subset(S.Reals):
        new_interval = []
        x = symbols("x", cls=Dummy, real=True)

        # self in rectangular form
        if not self.polar:
            for element in self.psets:
                if S.Zero in element.args[1]:
                    new_interval.append(element.args[0])
            new_interval = Union(*new_interval)
            return Intersection(new_interval, other)

        # self in polar form
        elif self.polar:
            for element in self.psets:
                if S.Zero in element.args[1]:
                    new_interval.append(element.args[0])
                if S.Pi in element.args[1]:
                    new_interval.append(ImageSet(Lambda(x, -x), element.args[0]))
                if S.Zero in element.args[0]:
                    new_interval.append(FiniteSet(0))
            new_interval = Union(*new_interval)
            return Intersection(new_interval, other)

@intersection_sets.register(Integers, Reals)
def _(a, b):
    return a

@intersection_sets.register(Range, Interval)
def _(a, b):
    # Check that there are no symbolic arguments
    if not all(i.is_number for i in a.args + b.args[:2]):
        return

    # In case of null Range, return an EmptySet.
    if a.size == 0:
        return S.EmptySet

    # trim down to self's size, and represent
    # as a Range with step 1.
    start = ceiling(max(b.inf, a.inf))
    if start not in b:
        start += 1
    end = floor(min(b.sup, a.sup))
    if end not in b:
        end -= 1
    return intersection_sets(a, Range(start, end + 1))

@intersection_sets.register(Range, Naturals)
def _(a, b):
    return intersection_sets(a, Interval(b.inf, S.Infinity))

@intersection_sets.register(Range, Range)
def _(a, b):
    # Check that there are no symbolic range arguments
    if not all(all(v.is_number for v in r.args) for r in [a, b]):
        return None

    # non-overlap quick exits
    if not b:
        return S.EmptySet
    if not a:
        return S.EmptySet
    if b.sup < a.inf:
        return S.EmptySet
    if b.inf > a.sup:
        return S.EmptySet

    # work with finite end at the start
    r1 = a
    if r1.start.is_infinite:
        r1 = r1.reversed
    r2 = b
    if r2.start.is_infinite:
        r2 = r2.reversed

    # If both ends are infinite then it means that one Range is just the set
    # of all integers (the step must be 1).
    if r1.start.is_infinite:
        return b
    if r2.start.is_infinite:
        return a

    from sympy.solvers.diophantine.diophantine import diop_linear

    # this equation represents the values of the Range;
    # it's a linear equation
    eq = lambda r, i: r.start + i*r.step

    # we want to know when the two equations might
    # have integer solutions so we use the diophantine
    # solver
    va, vb = diop_linear(eq(r1, Dummy('a')) - eq(r2, Dummy('b')))

    # check for no solution
    no_solution = va is None and vb is None
    if no_solution:
        return S.EmptySet

    # there is a solution
    # -------------------

    # find the coincident point, c
    a0 = va.as_coeff_Add()[0]
    c = eq(r1, a0)

    # find the first point, if possible, in each range
    # since c may not be that point
    def _first_finite_point(r1, c):
        if c == r1.start:
            return c
        # st is the signed step we need to take to
        # get from c to r1.start
        st = sign(r1.start - c)*step
        # use Range to calculate the first point:
        # we want to get as close as possible to
        # r1.start; the Range will not be null since
        # it will at least contain c
        s1 = Range(c, r1.start + st, st)[-1]
        if s1 == r1.start:
            pass
        else:
            # if we didn't hit r1.start then, if the
            # sign of st didn't match the sign of r1.step
            # we are off by one and s1 is not in r1
            if sign(r1.step) != sign(st):
                s1 -= st
        if s1 not in r1:
            return
        return s1

    # calculate the step size of the new Range
    step = abs(ilcm(r1.step, r2.step))
    s1 = _first_finite_point(r1, c)
    if s1 is None:
        return S.EmptySet
    s2 = _first_finite_point(r2, c)
    if s2 is None:
        return S.EmptySet

    # replace the corresponding start or stop in
    # the original Ranges with these points; the
    # result must have at least one point since
    # we know that s1 and s2 are in the Ranges
    def _updated_range(r, first):
        st = sign(r.step)*step
        if r.start.is_finite:
            rv = Range(first, r.stop, st)
        else:
            rv = Range(r.start, first + st, st)
        return rv
    r1 = _updated_range(a, s1)
    r2 = _updated_range(b, s2)

    # work with them both in the increasing direction
    if sign(r1.step) < 0:
        r1 = r1.reversed
    if sign(r2.step) < 0:
        r2 = r2.reversed

    # return clipped Range with positive step; it
    # can't be empty at this point
    start = max(r1.start, r2.start)
    stop = min(r1.stop, r2.stop)
    return Range(start, stop, step)


@intersection_sets.register(Range, Integers)
def _(a, b):
    return a


@intersection_sets.register(Range, Rationals)
def _(a, b):
    return a


@intersection_sets.register(ImageSet, Set)
def _(self, other):
    from sympy.solvers.diophantine import diophantine

    # Only handle the straight-forward univariate case
    if (len(self.lamda.variables) > 1
            or self.lamda.signature != self.lamda.variables):
        return None
    base_set = self.base_sets[0]

    # Intersection between ImageSets with Integers as base set
    # For {f(n) : n in Integers} & {g(m) : m in Integers} we solve the
    # diophantine equations f(n)=g(m).
    # If the solutions for n are {h(t) : t in Integers} then we return
    # {f(h(t)) : t in integers}.
    # If the solutions for n are {n_1, n_2, ..., n_k} then we return
    # {f(n_i) : 1 <= i <= k}.
    if base_set is S.Integers:
        gm = None
        if isinstance(other, ImageSet) and other.base_sets == (S.Integers,):
            gm = other.lamda.expr
            var = other.lamda.variables[0]
            # Symbol of second ImageSet lambda must be distinct from first
            m = Dummy('m')
            gm = gm.subs(var, m)
        elif other is S.Integers:
            m = gm = Dummy('m')
        if gm is not None:
            fn = self.lamda.expr
            n = self.lamda.variables[0]
            try:
                solns = list(diophantine(fn - gm, syms=(n, m), permute=True))
            except (TypeError, NotImplementedError):
                # TypeError if equation not polynomial with rational coeff.
                # NotImplementedError if correct format but no solver.
                return
            # 3 cases are possible for solns:
            # - empty set,
            # - one or more parametric (infinite) solutions,
            # - a finite number of (non-parametric) solution couples.
            # Among those, there is one type of solution set that is
            # not helpful here: multiple parametric solutions.
            if len(solns) == 0:
                return S.EmptySet
            elif any(s.free_symbols for tupl in solns for s in tupl):
                if len(solns) == 1:
                    soln, solm = solns[0]
                    (t,) = soln.free_symbols
                    expr = fn.subs(n, soln.subs(t, n)).expand()
                    return imageset(Lambda(n, expr), S.Integers)
                else:
                    return
            else:
                return FiniteSet(*(fn.subs(n, s[0]) for s in solns))

    if other == S.Reals:
        from sympy.solvers.solvers import denoms, solve_linear

        def _solution_union(exprs, sym):
            # return a union of linear solutions to i in expr;
            # if i cannot be solved, use a ConditionSet for solution
            sols = []
            for i in exprs:
                x, xis = solve_linear(i, 0, [sym])
                if x == sym:
                    sols.append(FiniteSet(xis))
                else:
                    sols.append(ConditionSet(sym, Eq(i, 0)))
            return Union(*sols)

        f = self.lamda.expr
        n = self.lamda.variables[0]

        n_ = Dummy(n.name, real=True)
        f_ = f.subs(n, n_)

        re, im = f_.as_real_imag()
        im = expand_complex(im)

        re = re.subs(n_, n)
        im = im.subs(n_, n)
        ifree = im.free_symbols
        lam = Lambda(n, re)
        if im.is_zero:
            # allow re-evaluation
            # of self in this case to make
            # the result canonical
            pass
        elif im.is_zero is False:
            return S.EmptySet
        elif ifree != {n}:
            return None
        else:
            # univarite imaginary part in same variable;
            # use numer instead of as_numer_denom to keep
            # this as fast as possible while still handling
            # simple cases
            base_set &= _solution_union(
                Mul.make_args(numer(im)), n)
        # exclude values that make denominators 0
        base_set -= _solution_union(denoms(f), n)
        return imageset(lam, base_set)

    elif isinstance(other, Interval):
        from sympy.solvers.solveset import (invert_real, invert_complex,
                                            solveset)

        f = self.lamda.expr
        n = self.lamda.variables[0]
        new_inf, new_sup = None, None
        new_lopen, new_ropen = other.left_open, other.right_open

        if f.is_real:
            inverter = invert_real
        else:
            inverter = invert_complex

        g1, h1 = inverter(f, other.inf, n)
        g2, h2 = inverter(f, other.sup, n)

        if all(isinstance(i, FiniteSet) for i in (h1, h2)):
            if g1 == n:
                if len(h1) == 1:
                    new_inf = h1.args[0]
            if g2 == n:
                if len(h2) == 1:
                    new_sup = h2.args[0]
            # TODO: Design a technique to handle multiple-inverse
            # functions

            # Any of the new boundary values cannot be determined
            if any(i is None for i in (new_sup, new_inf)):
                return


            range_set = S.EmptySet

            if all(i.is_real for i in (new_sup, new_inf)):
                # this assumes continuity of underlying function
                # however fixes the case when it is decreasing
                if new_inf > new_sup:
                    new_inf, new_sup = new_sup, new_inf
                new_interval = Interval(new_inf, new_sup, new_lopen, new_ropen)
                range_set = base_set.intersect(new_interval)
            else:
                if other.is_subset(S.Reals):
                    solutions = solveset(f, n, S.Reals)
                    if not isinstance(range_set, (ImageSet, ConditionSet)):
                        range_set = solutions.intersect(other)
                    else:
                        return

            if range_set is S.EmptySet:
                return S.EmptySet
            elif isinstance(range_set, Range) and range_set.size is not S.Infinity:
                range_set = FiniteSet(*list(range_set))

            if range_set is not None:
                return imageset(Lambda(n, f), range_set)
            return
        else:
            return


@intersection_sets.register(ProductSet, ProductSet)
def _(a, b):
    if len(b.args) != len(a.args):
        return S.EmptySet
    return ProductSet(*(i.intersect(j) for i, j in zip(a.sets, b.sets)))


@intersection_sets.register(Interval, Interval)
def _(a, b):
    # handle (-oo, oo)
    infty = S.NegativeInfinity, S.Infinity
    if a == Interval(*infty):
        l, r = a.left, a.right
        if l.is_real or l in infty or r.is_real or r in infty:
            return b

    # We can't intersect [0,3] with [x,6] -- we don't know if x>0 or x<0
    if not a._is_comparable(b):
        return None

    empty = False

    if a.start <= b.end and b.start <= a.end:
        # Get topology right.
        if a.start < b.start:
            start = b.start
            left_open = b.left_open
        elif a.start > b.start:
            start = a.start
            left_open = a.left_open
        else:
            start = a.start
            if not _aresame(a.start, b.start):
                # For example Integer(2) != Float(2)
                # Prefer the Float boundary because Floats should be
                # contagious in calculations.
                if b.start.has(Float) and not a.start.has(Float):
                    start = b.start
                elif a.start.has(Float) and not b.start.has(Float):
                    start = a.start
                else:
                    #this is to ensure that if Eq(a.start, b.start) but
                    #type(a.start) != type(b.start) the order of a and b
                    #does not matter for the result
                    start = list(ordered([a,b]))[0].start
            left_open = a.left_open or b.left_open

        if a.end < b.end:
            end = a.end
            right_open = a.right_open
        elif a.end > b.end:
            end = b.end
            right_open = b.right_open
        else:
            # see above for logic with start
            end = a.end
            if not _aresame(a.end, b.end):
                if b.end.has(Float) and not a.end.has(Float):
                    end = b.end
                elif a.end.has(Float) and not b.end.has(Float):
                    end = a.end
                else:
                    end = list(ordered([a,b]))[0].end
            right_open = a.right_open or b.right_open

        if end - start == 0 and (left_open or right_open):
            empty = True
    else:
        empty = True

    if empty:
        return S.EmptySet

    return Interval(start, end, left_open, right_open)

@intersection_sets.register(EmptySet, Set)
def _(a, b):
    return S.EmptySet

@intersection_sets.register(UniversalSet, Set)
def _(a, b):
    return b

@intersection_sets.register(FiniteSet, FiniteSet)
def _(a, b):
    return FiniteSet(*(a._elements & b._elements))

@intersection_sets.register(FiniteSet, Set)
def _(a, b):
    try:
        return FiniteSet(*[el for el in a if el in b])
    except TypeError:
        return None  # could not evaluate `el in b` due to symbolic ranges.

@intersection_sets.register(Set, Set)
def _(a, b):
    return None

@intersection_sets.register(Integers, Rationals)
def _(a, b):
    return a

@intersection_sets.register(Naturals, Rationals)
def _(a, b):
    return a

@intersection_sets.register(Rationals, Reals)
def _(a, b):
    return a

def _intlike_interval(a, b):
    try:
        if b._inf is S.NegativeInfinity and b._sup is S.Infinity:
            return a
        s = Range(max(a.inf, ceiling(b.left)), floor(b.right) + 1)
        return intersection_sets(s, b)  # take out endpoints if open interval
    except ValueError:
        return None

@intersection_sets.register(Integers, Interval)
def _(a, b):
    return _intlike_interval(a, b)

@intersection_sets.register(Naturals, Interval)
def _(a, b):
    return _intlike_interval(a, b)