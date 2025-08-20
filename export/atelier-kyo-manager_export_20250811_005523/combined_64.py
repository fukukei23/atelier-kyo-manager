
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\nanops.py ===
from __future__ import annotations

import functools
import itertools
from typing import (
    Any,
    Callable,
    cast,
)
import warnings

import numpy as np

from pandas._config import get_option

from pandas._libs import (
    NaT,
    NaTType,
    iNaT,
    lib,
)
from pandas._typing import (
    ArrayLike,
    AxisInt,
    CorrelationMethod,
    Dtype,
    DtypeObj,
    F,
    Scalar,
    Shape,
    npt,
)
from pandas.compat._optional import import_optional_dependency
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    is_complex,
    is_float,
    is_float_dtype,
    is_integer,
    is_numeric_dtype,
    is_object_dtype,
    needs_i8_conversion,
    pandas_dtype,
)
from pandas.core.dtypes.missing import (
    isna,
    na_value_for_dtype,
    notna,
)

bn = import_optional_dependency("bottleneck", errors="warn")
_BOTTLENECK_INSTALLED = bn is not None
_USE_BOTTLENECK = False


def set_use_bottleneck(v: bool = True) -> None:
    # set/unset to use bottleneck
    global _USE_BOTTLENECK
    if _BOTTLENECK_INSTALLED:
        _USE_BOTTLENECK = v


set_use_bottleneck(get_option("compute.use_bottleneck"))


class disallow:
    def __init__(self, *dtypes: Dtype) -> None:
        super().__init__()
        self.dtypes = tuple(pandas_dtype(dtype).type for dtype in dtypes)

    def check(self, obj) -> bool:
        return hasattr(obj, "dtype") and issubclass(obj.dtype.type, self.dtypes)

    def __call__(self, f: F) -> F:
        @functools.wraps(f)
        def _f(*args, **kwargs):
            obj_iter = itertools.chain(args, kwargs.values())
            if any(self.check(obj) for obj in obj_iter):
                f_name = f.__name__.replace("nan", "")
                raise TypeError(
                    f"reduction operation '{f_name}' not allowed for this dtype"
                )
            try:
                return f(*args, **kwargs)
            except ValueError as e:
                # we want to transform an object array
                # ValueError message to the more typical TypeError
                # e.g. this is normally a disallowed function on
                # object arrays that contain strings
                if is_object_dtype(args[0]):
                    raise TypeError(e) from e
                raise

        return cast(F, _f)


class bottleneck_switch:
    def __init__(self, name=None, **kwargs) -> None:
        self.name = name
        self.kwargs = kwargs

    def __call__(self, alt: F) -> F:
        bn_name = self.name or alt.__name__

        try:
            bn_func = getattr(bn, bn_name)
        except (AttributeError, NameError):  # pragma: no cover
            bn_func = None

        @functools.wraps(alt)
        def f(
            values: np.ndarray,
            *,
            axis: AxisInt | None = None,
            skipna: bool = True,
            **kwds,
        ):
            if len(self.kwargs) > 0:
                for k, v in self.kwargs.items():
                    if k not in kwds:
                        kwds[k] = v

            if values.size == 0 and kwds.get("min_count") is None:
                # We are empty, returning NA for our type
                # Only applies for the default `min_count` of None
                # since that affects how empty arrays are handled.
                # TODO(GH-18976) update all the nanops methods to
                # correctly handle empty inputs and remove this check.
                # It *may* just be `var`
                return _na_for_min_count(values, axis)

            if _USE_BOTTLENECK and skipna and _bn_ok_dtype(values.dtype, bn_name):
                if kwds.get("mask", None) is None:
                    # `mask` is not recognised by bottleneck, would raise
                    #  TypeError if called
                    kwds.pop("mask", None)
                    result = bn_func(values, axis=axis, **kwds)

                    # prefer to treat inf/-inf as NA, but must compute the func
                    # twice :(
                    if _has_infs(result):
                        result = alt(values, axis=axis, skipna=skipna, **kwds)
                else:
                    result = alt(values, axis=axis, skipna=skipna, **kwds)
            else:
                result = alt(values, axis=axis, skipna=skipna, **kwds)

            return result

        return cast(F, f)


def _bn_ok_dtype(dtype: DtypeObj, name: str) -> bool:
    # Bottleneck chokes on datetime64, PeriodDtype (or and EA)
    if dtype != object and not needs_i8_conversion(dtype):
        # GH 42878
        # Bottleneck uses naive summation leading to O(n) loss of precision
        # unlike numpy which implements pairwise summation, which has O(log(n)) loss
        # crossref: https://github.com/pydata/bottleneck/issues/379

        # GH 15507
        # bottleneck does not properly upcast during the sum
        # so can overflow

        # GH 9422
        # further we also want to preserve NaN when all elements
        # are NaN, unlike bottleneck/numpy which consider this
        # to be 0
        return name not in ["nansum", "nanprod", "nanmean"]
    return False


def _has_infs(result) -> bool:
    if isinstance(result, np.ndarray):
        if result.dtype in ("f8", "f4"):
            # Note: outside of an nanops-specific test, we always have
            #  result.ndim == 1, so there is no risk of this ravel making a copy.
            return lib.has_infs(result.ravel("K"))
    try:
        return np.isinf(result).any()
    except (TypeError, NotImplementedError):
        # if it doesn't support infs, then it can't have infs
        return False


def _get_fill_value(
    dtype: DtypeObj, fill_value: Scalar | None = None, fill_value_typ=None
):
    """return the correct fill value for the dtype of the values"""
    if fill_value is not None:
        return fill_value
    if _na_ok_dtype(dtype):
        if fill_value_typ is None:
            return np.nan
        else:
            if fill_value_typ == "+inf":
                return np.inf
            else:
                return -np.inf
    else:
        if fill_value_typ == "+inf":
            # need the max int here
            return lib.i8max
        else:
            return iNaT


def _maybe_get_mask(
    values: np.ndarray, skipna: bool, mask: npt.NDArray[np.bool_] | None
) -> npt.NDArray[np.bool_] | None:
    """
    Compute a mask if and only if necessary.

    This function will compute a mask iff it is necessary. Otherwise,
    return the provided mask (potentially None) when a mask does not need to be
    computed.

    A mask is never necessary if the values array is of boolean or integer
    dtypes, as these are incapable of storing NaNs. If passing a NaN-capable
    dtype that is interpretable as either boolean or integer data (eg,
    timedelta64), a mask must be provided.

    If the skipna parameter is False, a new mask will not be computed.

    The mask is computed using isna() by default. Setting invert=True selects
    notna() as the masking function.

    Parameters
    ----------
    values : ndarray
        input array to potentially compute mask for
    skipna : bool
        boolean for whether NaNs should be skipped
    mask : Optional[ndarray]
        nan-mask if known

    Returns
    -------
    Optional[np.ndarray[bool]]
    """
    if mask is None:
        if values.dtype.kind in "biu":
            # Boolean data cannot contain nulls, so signal via mask being None
            return None

        if skipna or values.dtype.kind in "mM":
            mask = isna(values)

    return mask


def _get_values(
    values: np.ndarray,
    skipna: bool,
    fill_value: Any = None,
    fill_value_typ: str | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[np.ndarray, npt.NDArray[np.bool_] | None]:
    """
    Utility to get the values view, mask, dtype, dtype_max, and fill_value.

    If both mask and fill_value/fill_value_typ are not None and skipna is True,
    the values array will be copied.

    For input arrays of boolean or integer dtypes, copies will only occur if a
    precomputed mask, a fill_value/fill_value_typ, and skipna=True are
    provided.

    Parameters
    ----------
    values : ndarray
        input array to potentially compute mask for
    skipna : bool
        boolean for whether NaNs should be skipped
    fill_value : Any
        value to fill NaNs with
    fill_value_typ : str
        Set to '+inf' or '-inf' to handle dtype-specific infinities
    mask : Optional[np.ndarray[bool]]
        nan-mask if known

    Returns
    -------
    values : ndarray
        Potential copy of input value array
    mask : Optional[ndarray[bool]]
        Mask for values, if deemed necessary to compute
    """
    # In _get_values is only called from within nanops, and in all cases
    #  with scalar fill_value.  This guarantee is important for the
    #  np.where call below

    mask = _maybe_get_mask(values, skipna, mask)

    dtype = values.dtype

    datetimelike = False
    if values.dtype.kind in "mM":
        # changing timedelta64/datetime64 to int64 needs to happen after
        #  finding `mask` above
        values = np.asarray(values.view("i8"))
        datetimelike = True

    if skipna and (mask is not None):
        # get our fill value (in case we need to provide an alternative
        # dtype for it)
        fill_value = _get_fill_value(
            dtype, fill_value=fill_value, fill_value_typ=fill_value_typ
        )

        if fill_value is not None:
            if mask.any():
                if datetimelike or _na_ok_dtype(dtype):
                    values = values.copy()
                    np.putmask(values, mask, fill_value)
                else:
                    # np.where will promote if needed
                    values = np.where(~mask, values, fill_value)

    return values, mask


def _get_dtype_max(dtype: np.dtype) -> np.dtype:
    # return a platform independent precision dtype
    dtype_max = dtype
    if dtype.kind in "bi":
        dtype_max = np.dtype(np.int64)
    elif dtype.kind == "u":
        dtype_max = np.dtype(np.uint64)
    elif dtype.kind == "f":
        dtype_max = np.dtype(np.float64)
    return dtype_max


def _na_ok_dtype(dtype: DtypeObj) -> bool:
    if needs_i8_conversion(dtype):
        return False
    return not issubclass(dtype.type, np.integer)


def _wrap_results(result, dtype: np.dtype, fill_value=None):
    """wrap our results if needed"""
    if result is NaT:
        pass

    elif dtype.kind == "M":
        if fill_value is None:
            # GH#24293
            fill_value = iNaT
        if not isinstance(result, np.ndarray):
            assert not isna(fill_value), "Expected non-null fill_value"
            if result == fill_value:
                result = np.nan

            if isna(result):
                result = np.datetime64("NaT", "ns").astype(dtype)
            else:
                result = np.int64(result).view(dtype)
            # retain original unit
            result = result.astype(dtype, copy=False)
        else:
            # If we have float dtype, taking a view will give the wrong result
            result = result.astype(dtype)
    elif dtype.kind == "m":
        if not isinstance(result, np.ndarray):
            if result == fill_value or np.isnan(result):
                result = np.timedelta64("NaT").astype(dtype)

            elif np.fabs(result) > lib.i8max:
                # raise if we have a timedelta64[ns] which is too large
                raise ValueError("overflow in timedelta operation")
            else:
                # return a timedelta64 with the original unit
                result = np.int64(result).astype(dtype, copy=False)

        else:
            result = result.astype("m8[ns]").view(dtype)

    return result


def _datetimelike_compat(func: F) -> F:
    """
    If we have datetime64 or timedelta64 values, ensure we have a correct
    mask before calling the wrapped function, then cast back afterwards.
    """

    @functools.wraps(func)
    def new_func(
        values: np.ndarray,
        *,
        axis: AxisInt | None = None,
        skipna: bool = True,
        mask: npt.NDArray[np.bool_] | None = None,
        **kwargs,
    ):
        orig_values = values

        datetimelike = values.dtype.kind in "mM"
        if datetimelike and mask is None:
            mask = isna(values)

        result = func(values, axis=axis, skipna=skipna, mask=mask, **kwargs)

        if datetimelike:
            result = _wrap_results(result, orig_values.dtype, fill_value=iNaT)
            if not skipna:
                assert mask is not None  # checked above
                result = _mask_datetimelike_result(result, axis, mask, orig_values)

        return result

    return cast(F, new_func)


def _na_for_min_count(values: np.ndarray, axis: AxisInt | None) -> Scalar | np.ndarray:
    """
    Return the missing value for `values`.

    Parameters
    ----------
    values : ndarray
    axis : int or None
        axis for the reduction, required if values.ndim > 1.

    Returns
    -------
    result : scalar or ndarray
        For 1-D values, returns a scalar of the correct missing type.
        For 2-D values, returns a 1-D array where each element is missing.
    """
    # we either return np.nan or pd.NaT
    if values.dtype.kind in "iufcb":
        values = values.astype("float64")
    fill_value = na_value_for_dtype(values.dtype)

    if values.ndim == 1:
        return fill_value
    elif axis is None:
        return fill_value
    else:
        result_shape = values.shape[:axis] + values.shape[axis + 1 :]

        return np.full(result_shape, fill_value, dtype=values.dtype)


def maybe_operate_rowwise(func: F) -> F:
    """
    NumPy operations on C-contiguous ndarrays with axis=1 can be
    very slow if axis 1 >> axis 0.
    Operate row-by-row and concatenate the results.
    """

    @functools.wraps(func)
    def newfunc(values: np.ndarray, *, axis: AxisInt | None = None, **kwargs):
        if (
            axis == 1
            and values.ndim == 2
            and values.flags["C_CONTIGUOUS"]
            # only takes this path for wide arrays (long dataframes), for threshold see
            # https://github.com/pandas-dev/pandas/pull/43311#issuecomment-974891737
            and (values.shape[1] / 1000) > values.shape[0]
            and values.dtype != object
            and values.dtype != bool
        ):
            arrs = list(values)
            if kwargs.get("mask") is not None:
                mask = kwargs.pop("mask")
                results = [
                    func(arrs[i], mask=mask[i], **kwargs) for i in range(len(arrs))
                ]
            else:
                results = [func(x, **kwargs) for x in arrs]
            return np.array(results)

        return func(values, axis=axis, **kwargs)

    return cast(F, newfunc)


def nanany(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> bool:
    """
    Check if any elements along an axis evaluate to True.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : bool

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2])
    >>> nanops.nanany(s.values)
    True

    >>> from pandas.core import nanops
    >>> s = pd.Series([np.nan])
    >>> nanops.nanany(s.values)
    False
    """
    if values.dtype.kind in "iub" and mask is None:
        # GH#26032 fastpath
        # error: Incompatible return value type (got "Union[bool_, ndarray]",
        # expected "bool")
        return values.any(axis)  # type: ignore[return-value]

    if values.dtype.kind == "M":
        # GH#34479
        warnings.warn(
            "'any' with datetime64 dtypes is deprecated and will raise in a "
            "future version. Use (obj != pd.Timestamp(0)).any() instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

    values, _ = _get_values(values, skipna, fill_value=False, mask=mask)

    # For object type, any won't necessarily return
    # boolean values (numpy/numpy#4352)
    if values.dtype == object:
        values = values.astype(bool)

    # error: Incompatible return value type (got "Union[bool_, ndarray]", expected
    # "bool")
    return values.any(axis)  # type: ignore[return-value]


def nanall(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> bool:
    """
    Check if all elements along an axis evaluate to True.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : bool

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2, np.nan])
    >>> nanops.nanall(s.values)
    True

    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 0])
    >>> nanops.nanall(s.values)
    False
    """
    if values.dtype.kind in "iub" and mask is None:
        # GH#26032 fastpath
        # error: Incompatible return value type (got "Union[bool_, ndarray]",
        # expected "bool")
        return values.all(axis)  # type: ignore[return-value]

    if values.dtype.kind == "M":
        # GH#34479
        warnings.warn(
            "'all' with datetime64 dtypes is deprecated and will raise in a "
            "future version. Use (obj != pd.Timestamp(0)).all() instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

    values, _ = _get_values(values, skipna, fill_value=True, mask=mask)

    # For object type, all won't necessarily return
    # boolean values (numpy/numpy#4352)
    if values.dtype == object:
        values = values.astype(bool)

    # error: Incompatible return value type (got "Union[bool_, ndarray]", expected
    # "bool")
    return values.all(axis)  # type: ignore[return-value]


@disallow("M8")
@_datetimelike_compat
@maybe_operate_rowwise
def nansum(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    min_count: int = 0,
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    """
    Sum the elements along an axis ignoring NaNs

    Parameters
    ----------
    values : ndarray[dtype]
    axis : int, optional
    skipna : bool, default True
    min_count: int, default 0
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : dtype

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2, np.nan])
    >>> nanops.nansum(s.values)
    3.0
    """
    dtype = values.dtype
    values, mask = _get_values(values, skipna, fill_value=0, mask=mask)
    dtype_sum = _get_dtype_max(dtype)
    if dtype.kind == "f":
        dtype_sum = dtype
    elif dtype.kind == "m":
        dtype_sum = np.dtype(np.float64)

    the_sum = values.sum(axis, dtype=dtype_sum)
    the_sum = _maybe_null_out(the_sum, axis, mask, values.shape, min_count=min_count)

    return the_sum


def _mask_datetimelike_result(
    result: np.ndarray | np.datetime64 | np.timedelta64,
    axis: AxisInt | None,
    mask: npt.NDArray[np.bool_],
    orig_values: np.ndarray,
) -> np.ndarray | np.datetime64 | np.timedelta64 | NaTType:
    if isinstance(result, np.ndarray):
        # we need to apply the mask
        result = result.astype("i8").view(orig_values.dtype)
        axis_mask = mask.any(axis=axis)
        # error: Unsupported target for indexed assignment ("Union[ndarray[Any, Any],
        # datetime64, timedelta64]")
        result[axis_mask] = iNaT  # type: ignore[index]
    else:
        if mask.any():
            return np.int64(iNaT).view(orig_values.dtype)
    return result


@bottleneck_switch()
@_datetimelike_compat
def nanmean(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    """
    Compute the mean of the element along an axis ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    float
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2, np.nan])
    >>> nanops.nanmean(s.values)
    1.5
    """
    dtype = values.dtype
    values, mask = _get_values(values, skipna, fill_value=0, mask=mask)
    dtype_sum = _get_dtype_max(dtype)
    dtype_count = np.dtype(np.float64)

    # not using needs_i8_conversion because that includes period
    if dtype.kind in "mM":
        dtype_sum = np.dtype(np.float64)
    elif dtype.kind in "iu":
        dtype_sum = np.dtype(np.float64)
    elif dtype.kind == "f":
        dtype_sum = dtype
        dtype_count = dtype

    count = _get_counts(values.shape, mask, axis, dtype=dtype_count)
    the_sum = values.sum(axis, dtype=dtype_sum)
    the_sum = _ensure_numeric(the_sum)

    if axis is not None and getattr(the_sum, "ndim", False):
        count = cast(np.ndarray, count)
        with np.errstate(all="ignore"):
            # suppress division by zero warnings
            the_mean = the_sum / count
        ct_mask = count == 0
        if ct_mask.any():
            the_mean[ct_mask] = np.nan
    else:
        the_mean = the_sum / count if count > 0 else np.nan

    return the_mean


@bottleneck_switch()
def nanmedian(values, *, axis: AxisInt | None = None, skipna: bool = True, mask=None):
    """
    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 2, 2])
    >>> nanops.nanmedian(s.values)
    2.0
    """
    # for floats without mask, the data already uses NaN as missing value
    # indicator, and `mask` will be calculated from that below -> in those
    # cases we never need to set NaN to the masked values
    using_nan_sentinel = values.dtype.kind == "f" and mask is None

    def get_median(x, _mask=None):
        if _mask is None:
            _mask = notna(x)
        else:
            _mask = ~_mask
        if not skipna and not _mask.all():
            return np.nan
        with warnings.catch_warnings():
            # Suppress RuntimeWarning about All-NaN slice
            warnings.filterwarnings(
                "ignore", "All-NaN slice encountered", RuntimeWarning
            )
            res = np.nanmedian(x[_mask])
        return res

    dtype = values.dtype
    values, mask = _get_values(values, skipna, mask=mask, fill_value=None)
    if values.dtype.kind != "f":
        if values.dtype == object:
            # GH#34671 avoid casting strings to numeric
            inferred = lib.infer_dtype(values)
            if inferred in ["string", "mixed"]:
                raise TypeError(f"Cannot convert {values} to numeric")
        try:
            values = values.astype("f8")
        except ValueError as err:
            # e.g. "could not convert string to float: 'a'"
            raise TypeError(str(err)) from err
    if not using_nan_sentinel and mask is not None:
        if not values.flags.writeable:
            values = values.copy()
        values[mask] = np.nan

    notempty = values.size

    # an array from a frame
    if values.ndim > 1 and axis is not None:
        # there's a non-empty array to apply over otherwise numpy raises
        if notempty:
            if not skipna:
                res = np.apply_along_axis(get_median, axis, values)

            else:
                # fastpath for the skipna case
                with warnings.catch_warnings():
                    # Suppress RuntimeWarning about All-NaN slice
                    warnings.filterwarnings(
                        "ignore", "All-NaN slice encountered", RuntimeWarning
                    )
                    if (values.shape[1] == 1 and axis == 0) or (
                        values.shape[0] == 1 and axis == 1
                    ):
                        # GH52788: fastpath when squeezable, nanmedian for 2D array slow
                        res = np.nanmedian(np.squeeze(values), keepdims=True)
                    else:
                        res = np.nanmedian(values, axis=axis)

        else:
            # must return the correct shape, but median is not defined for the
            # empty set so return nans of shape "everything but the passed axis"
            # since "axis" is where the reduction would occur if we had a nonempty
            # array
            res = _get_empty_reduction_result(values.shape, axis)

    else:
        # otherwise return a scalar value
        res = get_median(values, mask) if notempty else np.nan
    return _wrap_results(res, dtype)


def _get_empty_reduction_result(
    shape: Shape,
    axis: AxisInt,
) -> np.ndarray:
    """
    The result from a reduction on an empty ndarray.

    Parameters
    ----------
    shape : Tuple[int, ...]
    axis : int

    Returns
    -------
    np.ndarray
    """
    shp = np.array(shape)
    dims = np.arange(len(shape))
    ret = np.empty(shp[dims != axis], dtype=np.float64)
    ret.fill(np.nan)
    return ret


def _get_counts_nanvar(
    values_shape: Shape,
    mask: npt.NDArray[np.bool_] | None,
    axis: AxisInt | None,
    ddof: int,
    dtype: np.dtype = np.dtype(np.float64),
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Get the count of non-null values along an axis, accounting
    for degrees of freedom.

    Parameters
    ----------
    values_shape : Tuple[int, ...]
        shape tuple from values ndarray, used if mask is None
    mask : Optional[ndarray[bool]]
        locations in values that should be considered missing
    axis : Optional[int]
        axis to count along
    ddof : int
        degrees of freedom
    dtype : type, optional
        type to use for count

    Returns
    -------
    count : int, np.nan or np.ndarray
    d : int, np.nan or np.ndarray
    """
    count = _get_counts(values_shape, mask, axis, dtype=dtype)
    d = count - dtype.type(ddof)

    # always return NaN, never inf
    if is_float(count):
        if count <= ddof:
            # error: Incompatible types in assignment (expression has type
            # "float", variable has type "Union[floating[Any], ndarray[Any,
            # dtype[floating[Any]]]]")
            count = np.nan  # type: ignore[assignment]
            d = np.nan
    else:
        # count is not narrowed by is_float check
        count = cast(np.ndarray, count)
        mask = count <= ddof
        if mask.any():
            np.putmask(d, mask, np.nan)
            np.putmask(count, mask, np.nan)
    return count, d


@bottleneck_switch(ddof=1)
def nanstd(
    values,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    ddof: int = 1,
    mask=None,
):
    """
    Compute the standard deviation along given axis while ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 2, 3])
    >>> nanops.nanstd(s.values)
    1.0
    """
    if values.dtype == "M8[ns]":
        values = values.view("m8[ns]")

    orig_dtype = values.dtype
    values, mask = _get_values(values, skipna, mask=mask)

    result = np.sqrt(nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask))
    return _wrap_results(result, orig_dtype)


@disallow("M8", "m8")
@bottleneck_switch(ddof=1)
def nanvar(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    ddof: int = 1,
    mask=None,
):
    """
    Compute the variance along given axis while ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 2, 3])
    >>> nanops.nanvar(s.values)
    1.0
    """
    dtype = values.dtype
    mask = _maybe_get_mask(values, skipna, mask)
    if dtype.kind in "iu":
        values = values.astype("f8")
        if mask is not None:
            values[mask] = np.nan

    if values.dtype.kind == "f":
        count, d = _get_counts_nanvar(values.shape, mask, axis, ddof, values.dtype)
    else:
        count, d = _get_counts_nanvar(values.shape, mask, axis, ddof)

    if skipna and mask is not None:
        values = values.copy()
        np.putmask(values, mask, 0)

    # xref GH10242
    # Compute variance via two-pass algorithm, which is stable against
    # cancellation errors and relatively accurate for small numbers of
    # observations.
    #
    # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    avg = _ensure_numeric(values.sum(axis=axis, dtype=np.float64)) / count
    if axis is not None:
        avg = np.expand_dims(avg, axis)
    sqr = _ensure_numeric((avg - values) ** 2)
    if mask is not None:
        np.putmask(sqr, mask, 0)
    result = sqr.sum(axis=axis, dtype=np.float64) / d

    # Return variance as np.float64 (the datatype used in the accumulator),
    # unless we were dealing with a float array, in which case use the same
    # precision as the original values array.
    if dtype.kind == "f":
        result = result.astype(dtype, copy=False)
    return result


@disallow("M8", "m8")
def nansem(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    ddof: int = 1,
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    """
    Compute the standard error in the mean along given axis while ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float64
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 2, 3])
    >>> nanops.nansem(s.values)
     0.5773502691896258
    """
    # This checks if non-numeric-like data is passed with numeric_only=False
    # and raises a TypeError otherwise
    nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask)

    mask = _maybe_get_mask(values, skipna, mask)
    if values.dtype.kind != "f":
        values = values.astype("f8")

    if not skipna and mask is not None and mask.any():
        return np.nan

    count, _ = _get_counts_nanvar(values.shape, mask, axis, ddof, values.dtype)
    var = nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask)

    return np.sqrt(var) / np.sqrt(count)


def _nanminmax(meth, fill_value_typ):
    @bottleneck_switch(name=f"nan{meth}")
    @_datetimelike_compat
    def reduction(
        values: np.ndarray,
        *,
        axis: AxisInt | None = None,
        skipna: bool = True,
        mask: npt.NDArray[np.bool_] | None = None,
    ):
        if values.size == 0:
            return _na_for_min_count(values, axis)

        values, mask = _get_values(
            values, skipna, fill_value_typ=fill_value_typ, mask=mask
        )
        result = getattr(values, meth)(axis)
        result = _maybe_null_out(result, axis, mask, values.shape)
        return result

    return reduction


nanmin = _nanminmax("min", fill_value_typ="+inf")
nanmax = _nanminmax("max", fill_value_typ="-inf")


def nanargmax(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> int | np.ndarray:
    """
    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : int or ndarray[int]
        The index/indices  of max value in specified axis or -1 in the NA case

    Examples
    --------
    >>> from pandas.core import nanops
    >>> arr = np.array([1, 2, 3, np.nan, 4])
    >>> nanops.nanargmax(arr)
    4

    >>> arr = np.array(range(12), dtype=np.float64).reshape(4, 3)
    >>> arr[2:, 2] = np.nan
    >>> arr
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [ 6.,  7., nan],
           [ 9., 10., nan]])
    >>> nanops.nanargmax(arr, axis=1)
    array([2, 2, 1, 1])
    """
    values, mask = _get_values(values, True, fill_value_typ="-inf", mask=mask)
    result = values.argmax(axis)
    # error: Argument 1 to "_maybe_arg_null_out" has incompatible type "Any |
    # signedinteger[Any]"; expected "ndarray[Any, Any]"
    result = _maybe_arg_null_out(result, axis, mask, skipna)  # type: ignore[arg-type]
    return result


def nanargmin(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> int | np.ndarray:
    """
    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : int or ndarray[int]
        The index/indices of min value in specified axis or -1 in the NA case

    Examples
    --------
    >>> from pandas.core import nanops
    >>> arr = np.array([1, 2, 3, np.nan, 4])
    >>> nanops.nanargmin(arr)
    0

    >>> arr = np.array(range(12), dtype=np.float64).reshape(4, 3)
    >>> arr[2:, 0] = np.nan
    >>> arr
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [nan,  7.,  8.],
           [nan, 10., 11.]])
    >>> nanops.nanargmin(arr, axis=1)
    array([0, 0, 1, 1])
    """
    values, mask = _get_values(values, True, fill_value_typ="+inf", mask=mask)
    result = values.argmin(axis)
    # error: Argument 1 to "_maybe_arg_null_out" has incompatible type "Any |
    # signedinteger[Any]"; expected "ndarray[Any, Any]"
    result = _maybe_arg_null_out(result, axis, mask, skipna)  # type: ignore[arg-type]
    return result


@disallow("M8", "m8")
@maybe_operate_rowwise
def nanskew(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    """
    Compute the sample skewness.

    The statistic computed here is the adjusted Fisher-Pearson standardized
    moment coefficient G1. The algorithm computes this coefficient directly
    from the second and third central moment.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float64
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 1, 2])
    >>> nanops.nanskew(s.values)
    1.7320508075688787
    """
    mask = _maybe_get_mask(values, skipna, mask)
    if values.dtype.kind != "f":
        values = values.astype("f8")
        count = _get_counts(values.shape, mask, axis)
    else:
        count = _get_counts(values.shape, mask, axis, dtype=values.dtype)

    if skipna and mask is not None:
        values = values.copy()
        np.putmask(values, mask, 0)
    elif not skipna and mask is not None and mask.any():
        return np.nan

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = values.sum(axis, dtype=np.float64) / count
    if axis is not None:
        mean = np.expand_dims(mean, axis)

    adjusted = values - mean
    if skipna and mask is not None:
        np.putmask(adjusted, mask, 0)
    adjusted2 = adjusted**2
    adjusted3 = adjusted2 * adjusted
    m2 = adjusted2.sum(axis, dtype=np.float64)
    m3 = adjusted3.sum(axis, dtype=np.float64)

    # floating point error
    #
    # #18044 in _libs/windows.pyx calc_skew follow this behavior
    # to fix the fperr to treat m2 <1e-14 as zero
    m2 = _zero_out_fperr(m2)
    m3 = _zero_out_fperr(m3)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (count * (count - 1) ** 0.5 / (count - 2)) * (m3 / m2**1.5)

    dtype = values.dtype
    if dtype.kind == "f":
        result = result.astype(dtype, copy=False)

    if isinstance(result, np.ndarray):
        result = np.where(m2 == 0, 0, result)
        result[count < 3] = np.nan
    else:
        result = dtype.type(0) if m2 == 0 else result
        if count < 3:
            return np.nan

    return result


@disallow("M8", "m8")
@maybe_operate_rowwise
def nankurt(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    """
    Compute the sample excess kurtosis

    The statistic computed here is the adjusted Fisher-Pearson standardized
    moment coefficient G2, computed directly from the second and fourth
    central moment.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float64
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 1, 3, 2])
    >>> nanops.nankurt(s.values)
    -1.2892561983471076
    """
    mask = _maybe_get_mask(values, skipna, mask)
    if values.dtype.kind != "f":
        values = values.astype("f8")
        count = _get_counts(values.shape, mask, axis)
    else:
        count = _get_counts(values.shape, mask, axis, dtype=values.dtype)

    if skipna and mask is not None:
        values = values.copy()
        np.putmask(values, mask, 0)
    elif not skipna and mask is not None and mask.any():
        return np.nan

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = values.sum(axis, dtype=np.float64) / count
    if axis is not None:
        mean = np.expand_dims(mean, axis)

    adjusted = values - mean
    if skipna and mask is not None:
        np.putmask(adjusted, mask, 0)
    adjusted2 = adjusted**2
    adjusted4 = adjusted2**2
    m2 = adjusted2.sum(axis, dtype=np.float64)
    m4 = adjusted4.sum(axis, dtype=np.float64)

    with np.errstate(invalid="ignore", divide="ignore"):
        adj = 3 * (count - 1) ** 2 / ((count - 2) * (count - 3))
        numerator = count * (count + 1) * (count - 1) * m4
        denominator = (count - 2) * (count - 3) * m2**2

    # floating point error
    #
    # #18044 in _libs/windows.pyx calc_kurt follow this behavior
    # to fix the fperr to treat denom <1e-14 as zero
    numerator = _zero_out_fperr(numerator)
    denominator = _zero_out_fperr(denominator)

    if not isinstance(denominator, np.ndarray):
        # if ``denom`` is a scalar, check these corner cases first before
        # doing division
        if count < 4:
            return np.nan
        if denominator == 0:
            return values.dtype.type(0)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = numerator / denominator - adj

    dtype = values.dtype
    if dtype.kind == "f":
        result = result.astype(dtype, copy=False)

    if isinstance(result, np.ndarray):
        result = np.where(denominator == 0, 0, result)
        result[count < 4] = np.nan

    return result


@disallow("M8", "m8")
@maybe_operate_rowwise
def nanprod(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    min_count: int = 0,
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    """
    Parameters
    ----------
    values : ndarray[dtype]
    axis : int, optional
    skipna : bool, default True
    min_count: int, default 0
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    Dtype
        The product of all elements on a given axis. ( NaNs are treated as 1)

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2, 3, np.nan])
    >>> nanops.nanprod(s.values)
    6.0
    """
    mask = _maybe_get_mask(values, skipna, mask)

    if skipna and mask is not None:
        values = values.copy()
        values[mask] = 1
    result = values.prod(axis)
    # error: Incompatible return value type (got "Union[ndarray, float]", expected
    # "float")
    return _maybe_null_out(  # type: ignore[return-value]
        result, axis, mask, values.shape, min_count=min_count
    )


def _maybe_arg_null_out(
    result: np.ndarray,
    axis: AxisInt | None,
    mask: npt.NDArray[np.bool_] | None,
    skipna: bool,
) -> np.ndarray | int:
    # helper function for nanargmin/nanargmax
    if mask is None:
        return result

    if axis is None or not getattr(result, "ndim", False):
        if skipna:
            if mask.all():
                return -1
        else:
            if mask.any():
                return -1
    else:
        if skipna:
            na_mask = mask.all(axis)
        else:
            na_mask = mask.any(axis)
        if na_mask.any():
            result[na_mask] = -1
    return result


def _get_counts(
    values_shape: Shape,
    mask: npt.NDArray[np.bool_] | None,
    axis: AxisInt | None,
    dtype: np.dtype[np.floating] = np.dtype(np.float64),
) -> np.floating | npt.NDArray[np.floating]:
    """
    Get the count of non-null values along an axis

    Parameters
    ----------
    values_shape : tuple of int
        shape tuple from values ndarray, used if mask is None
    mask : Optional[ndarray[bool]]
        locations in values that should be considered missing
    axis : Optional[int]
        axis to count along
    dtype : type, optional
        type to use for count

    Returns
    -------
    count : scalar or array
    """
    if axis is None:
        if mask is not None:
            n = mask.size - mask.sum()
        else:
            n = np.prod(values_shape)
        return dtype.type(n)

    if mask is not None:
        count = mask.shape[axis] - mask.sum(axis)
    else:
        count = values_shape[axis]

    if is_integer(count):
        return dtype.type(count)
    return count.astype(dtype, copy=False)


def _maybe_null_out(
    result: np.ndarray | float | NaTType,
    axis: AxisInt | None,
    mask: npt.NDArray[np.bool_] | None,
    shape: tuple[int, ...],
    min_count: int = 1,
) -> np.ndarray | float | NaTType:
    """
    Returns
    -------
    Dtype
        The product of all elements on a given axis. ( NaNs are treated as 1)
    """
    if mask is None and min_count == 0:
        # nothing to check; short-circuit
        return result

    if axis is not None and isinstance(result, np.ndarray):
        if mask is not None:
            null_mask = (mask.shape[axis] - mask.sum(axis) - min_count) < 0
        else:
            # we have no nulls, kept mask=None in _maybe_get_mask
            below_count = shape[axis] - min_count < 0
            new_shape = shape[:axis] + shape[axis + 1 :]
            null_mask = np.broadcast_to(below_count, new_shape)

        if np.any(null_mask):
            if is_numeric_dtype(result):
                if np.iscomplexobj(result):
                    result = result.astype("c16")
                elif not is_float_dtype(result):
                    result = result.astype("f8", copy=False)
                result[null_mask] = np.nan
            else:
                # GH12941, use None to auto cast null
                result[null_mask] = None
    elif result is not NaT:
        if check_below_min_count(shape, mask, min_count):
            result_dtype = getattr(result, "dtype", None)
            if is_float_dtype(result_dtype):
                # error: Item "None" of "Optional[Any]" has no attribute "type"
                result = result_dtype.type("nan")  # type: ignore[union-attr]
            else:
                result = np.nan

    return result


def check_below_min_count(
    shape: tuple[int, ...], mask: npt.NDArray[np.bool_] | None, min_count: int
) -> bool:
    """
    Check for the `min_count` keyword. Returns True if below `min_count` (when
    missing value should be returned from the reduction).

    Parameters
    ----------
    shape : tuple
        The shape of the values (`values.shape`).
    mask : ndarray[bool] or None
        Boolean numpy array (typically of same shape as `shape`) or None.
    min_count : int
        Keyword passed through from sum/prod call.

    Returns
    -------
    bool
    """
    if min_count > 0:
        if mask is None:
            # no missing values, only check size
            non_nulls = np.prod(shape)
        else:
            non_nulls = mask.size - mask.sum()
        if non_nulls < min_count:
            return True
    return False


def _zero_out_fperr(arg):
    # #18044 reference this behavior to fix rolling skew/kurt issue
    if isinstance(arg, np.ndarray):
        return np.where(np.abs(arg) < 1e-14, 0, arg)
    else:
        return arg.dtype.type(0) if np.abs(arg) < 1e-14 else arg


@disallow("M8", "m8")
def nancorr(
    a: np.ndarray,
    b: np.ndarray,
    *,
    method: CorrelationMethod = "pearson",
    min_periods: int | None = None,
) -> float:
    """
    a, b: ndarrays
    """
    if len(a) != len(b):
        raise AssertionError("Operands to nancorr must have same size")

    if min_periods is None:
        min_periods = 1

    valid = notna(a) & notna(b)
    if not valid.all():
        a = a[valid]
        b = b[valid]

    if len(a) < min_periods:
        return np.nan

    a = _ensure_numeric(a)
    b = _ensure_numeric(b)

    f = get_corr_func(method)
    return f(a, b)


def get_corr_func(
    method: CorrelationMethod,
) -> Callable[[np.ndarray, np.ndarray], float]:
    if method == "kendall":
        from scipy.stats import kendalltau

        def func(a, b):
            return kendalltau(a, b)[0]

        return func
    elif method == "spearman":
        from scipy.stats import spearmanr

        def func(a, b):
            return spearmanr(a, b)[0]

        return func
    elif method == "pearson":

        def func(a, b):
            return np.corrcoef(a, b)[0, 1]

        return func
    elif callable(method):
        return method

    raise ValueError(
        f"Unknown method '{method}', expected one of "
        "'kendall', 'spearman', 'pearson', or callable"
    )


@disallow("M8", "m8")
def nancov(
    a: np.ndarray,
    b: np.ndarray,
    *,
    min_periods: int | None = None,
    ddof: int | None = 1,
) -> float:
    if len(a) != len(b):
        raise AssertionError("Operands to nancov must have same size")

    if min_periods is None:
        min_periods = 1

    valid = notna(a) & notna(b)
    if not valid.all():
        a = a[valid]
        b = b[valid]

    if len(a) < min_periods:
        return np.nan

    a = _ensure_numeric(a)
    b = _ensure_numeric(b)

    return np.cov(a, b, ddof=ddof)[0, 1]


def _ensure_numeric(x):
    if isinstance(x, np.ndarray):
        if x.dtype.kind in "biu":
            x = x.astype(np.float64)
        elif x.dtype == object:
            inferred = lib.infer_dtype(x)
            if inferred in ["string", "mixed"]:
                # GH#44008, GH#36703 avoid casting e.g. strings to numeric
                raise TypeError(f"Could not convert {x} to numeric")
            try:
                x = x.astype(np.complex128)
            except (TypeError, ValueError):
                try:
                    x = x.astype(np.float64)
                except ValueError as err:
                    # GH#29941 we get here with object arrays containing strs
                    raise TypeError(f"Could not convert {x} to numeric") from err
            else:
                if not np.any(np.imag(x)):
                    x = x.real
    elif not (is_float(x) or is_integer(x) or is_complex(x)):
        if isinstance(x, str):
            # GH#44008, GH#36703 avoid casting e.g. strings to numeric
            raise TypeError(f"Could not convert string '{x}' to numeric")
        try:
            x = float(x)
        except (TypeError, ValueError):
            # e.g. "1+1j" or "foo"
            try:
                x = complex(x)
            except ValueError as err:
                # e.g. "foo"
                raise TypeError(f"Could not convert {x} to numeric") from err
    return x


def na_accum_func(values: ArrayLike, accum_func, *, skipna: bool) -> ArrayLike:
    """
    Cumulative function with skipna support.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
    accum_func : {np.cumprod, np.maximum.accumulate, np.cumsum, np.minimum.accumulate}
    skipna : bool

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    mask_a, mask_b = {
        np.cumprod: (1.0, np.nan),
        np.maximum.accumulate: (-np.inf, np.nan),
        np.cumsum: (0.0, np.nan),
        np.minimum.accumulate: (np.inf, np.nan),
    }[accum_func]

    # This should go through ea interface
    assert values.dtype.kind not in "mM"

    # We will be applying this function to block values
    if skipna and not issubclass(values.dtype.type, (np.integer, np.bool_)):
        vals = values.copy()
        mask = isna(vals)
        vals[mask] = mask_a
        result = accum_func(vals, axis=0)
        result[mask] = mask_b
    else:
        result = accum_func(values, axis=0)

    return result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\algorithms.py ===
"""
Generic data algorithms. This module is experimental at the moment and not
intended for public consumption
"""
from __future__ import annotations

import decimal
import operator
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Literal,
    cast,
)
import warnings

import numpy as np

from pandas._libs import (
    algos,
    hashtable as htable,
    iNaT,
    lib,
)
from pandas._typing import (
    AnyArrayLike,
    ArrayLike,
    AxisInt,
    DtypeObj,
    TakeIndexer,
    npt,
)
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import (
    construct_1d_object_array_from_listlike,
    np_find_common_type,
)
from pandas.core.dtypes.common import (
    ensure_float64,
    ensure_object,
    ensure_platform_int,
    is_array_like,
    is_bool_dtype,
    is_complex_dtype,
    is_dict_like,
    is_extension_array_dtype,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_object_dtype,
    is_signed_integer_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
    BaseMaskedDtype,
    CategoricalDtype,
    ExtensionDtype,
    NumpyEADtype,
)
from pandas.core.dtypes.generic import (
    ABCDatetimeArray,
    ABCExtensionArray,
    ABCIndex,
    ABCMultiIndex,
    ABCSeries,
    ABCTimedeltaArray,
)
from pandas.core.dtypes.missing import (
    isna,
    na_value_for_dtype,
)

from pandas.core.array_algos.take import take_nd
from pandas.core.construction import (
    array as pd_array,
    ensure_wrapped_if_datetimelike,
    extract_array,
)
from pandas.core.indexers import validate_indices

if TYPE_CHECKING:
    from pandas._typing import (
        ListLike,
        NumpySorter,
        NumpyValueArrayLike,
    )

    from pandas import (
        Categorical,
        Index,
        Series,
    )
    from pandas.core.arrays import (
        BaseMaskedArray,
        ExtensionArray,
    )


# --------------- #
# dtype access    #
# --------------- #
def _ensure_data(values: ArrayLike) -> np.ndarray:
    """
    routine to ensure that our data is of the correct
    input dtype for lower-level routines

    This will coerce:
    - ints -> int64
    - uint -> uint64
    - bool -> uint8
    - datetimelike -> i8
    - datetime64tz -> i8 (in local tz)
    - categorical -> codes

    Parameters
    ----------
    values : np.ndarray or ExtensionArray

    Returns
    -------
    np.ndarray
    """

    if not isinstance(values, ABCMultiIndex):
        # extract_array would raise
        values = extract_array(values, extract_numpy=True)

    if is_object_dtype(values.dtype):
        return ensure_object(np.asarray(values))

    elif isinstance(values.dtype, BaseMaskedDtype):
        # i.e. BooleanArray, FloatingArray, IntegerArray
        values = cast("BaseMaskedArray", values)
        if not values._hasna:
            # No pd.NAs -> We can avoid an object-dtype cast (and copy) GH#41816
            #  recurse to avoid re-implementing logic for eg bool->uint8
            return _ensure_data(values._data)
        return np.asarray(values)

    elif isinstance(values.dtype, CategoricalDtype):
        # NB: cases that go through here should NOT be using _reconstruct_data
        #  on the back-end.
        values = cast("Categorical", values)
        return values.codes

    elif is_bool_dtype(values.dtype):
        if isinstance(values, np.ndarray):
            # i.e. actually dtype == np.dtype("bool")
            return np.asarray(values).view("uint8")
        else:
            # e.g. Sparse[bool, False]  # TODO: no test cases get here
            return np.asarray(values).astype("uint8", copy=False)

    elif is_integer_dtype(values.dtype):
        return np.asarray(values)

    elif is_float_dtype(values.dtype):
        # Note: checking `values.dtype == "float128"` raises on Windows and 32bit
        # error: Item "ExtensionDtype" of "Union[Any, ExtensionDtype, dtype[Any]]"
        # has no attribute "itemsize"
        if values.dtype.itemsize in [2, 12, 16]:  # type: ignore[union-attr]
            # we dont (yet) have float128 hashtable support
            return ensure_float64(values)
        return np.asarray(values)

    elif is_complex_dtype(values.dtype):
        return cast(np.ndarray, values)

    # datetimelike
    elif needs_i8_conversion(values.dtype):
        npvalues = values.view("i8")
        npvalues = cast(np.ndarray, npvalues)
        return npvalues

    # we have failed, return object
    values = np.asarray(values, dtype=object)
    return ensure_object(values)


def _reconstruct_data(
    values: ArrayLike, dtype: DtypeObj, original: AnyArrayLike
) -> ArrayLike:
    """
    reverse of _ensure_data

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
    dtype : np.dtype or ExtensionDtype
    original : AnyArrayLike

    Returns
    -------
    ExtensionArray or np.ndarray
    """
    if isinstance(values, ABCExtensionArray) and values.dtype == dtype:
        # Catch DatetimeArray/TimedeltaArray
        return values

    if not isinstance(dtype, np.dtype):
        # i.e. ExtensionDtype; note we have ruled out above the possibility
        #  that values.dtype == dtype
        cls = dtype.construct_array_type()

        values = cls._from_sequence(values, dtype=dtype)

    else:
        values = values.astype(dtype, copy=False)

    return values


def _ensure_arraylike(values, func_name: str) -> ArrayLike:
    """
    ensure that we are arraylike if not already
    """
    if not isinstance(values, (ABCIndex, ABCSeries, ABCExtensionArray, np.ndarray)):
        # GH#52986
        if func_name != "isin-targets":
            # Make an exception for the comps argument in isin.
            warnings.warn(
                f"{func_name} with argument that is not not a Series, Index, "
                "ExtensionArray, or np.ndarray is deprecated and will raise in a "
                "future version.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

        inferred = lib.infer_dtype(values, skipna=False)
        if inferred in ["mixed", "string", "mixed-integer"]:
            # "mixed-integer" to ensure we do not cast ["ss", 42] to str GH#22160
            if isinstance(values, tuple):
                values = list(values)
            values = construct_1d_object_array_from_listlike(values)
        else:
            values = np.asarray(values)
    return values


_hashtables = {
    "complex128": htable.Complex128HashTable,
    "complex64": htable.Complex64HashTable,
    "float64": htable.Float64HashTable,
    "float32": htable.Float32HashTable,
    "uint64": htable.UInt64HashTable,
    "uint32": htable.UInt32HashTable,
    "uint16": htable.UInt16HashTable,
    "uint8": htable.UInt8HashTable,
    "int64": htable.Int64HashTable,
    "int32": htable.Int32HashTable,
    "int16": htable.Int16HashTable,
    "int8": htable.Int8HashTable,
    "string": htable.StringHashTable,
    "object": htable.PyObjectHashTable,
}


def _get_hashtable_algo(values: np.ndarray):
    """
    Parameters
    ----------
    values : np.ndarray

    Returns
    -------
    htable : HashTable subclass
    values : ndarray
    """
    values = _ensure_data(values)

    ndtype = _check_object_for_strings(values)
    hashtable = _hashtables[ndtype]
    return hashtable, values


def _check_object_for_strings(values: np.ndarray) -> str:
    """
    Check if we can use string hashtable instead of object hashtable.

    Parameters
    ----------
    values : ndarray

    Returns
    -------
    str
    """
    ndtype = values.dtype.name
    if ndtype == "object":
        # it's cheaper to use a String Hash Table than Object; we infer
        # including nulls because that is the only difference between
        # StringHashTable and ObjectHashtable
        if lib.is_string_array(values, skipna=False):
            ndtype = "string"
    return ndtype


# --------------- #
# top-level algos #
# --------------- #


def unique(values):
    """
    Return unique values based on a hash table.

    Uniques are returned in order of appearance. This does NOT sort.

    Significantly faster than numpy.unique for long enough sequences.
    Includes NA values.

    Parameters
    ----------
    values : 1d array-like

    Returns
    -------
    numpy.ndarray or ExtensionArray

        The return can be:

        * Index : when the input is an Index
        * Categorical : when the input is a Categorical dtype
        * ndarray : when the input is a Series/ndarray

        Return numpy.ndarray or ExtensionArray.

    See Also
    --------
    Index.unique : Return unique values from an Index.
    Series.unique : Return unique values of Series object.

    Examples
    --------
    >>> pd.unique(pd.Series([2, 1, 3, 3]))
    array([2, 1, 3])

    >>> pd.unique(pd.Series([2] + [1] * 5))
    array([2, 1])

    >>> pd.unique(pd.Series([pd.Timestamp("20160101"), pd.Timestamp("20160101")]))
    array(['2016-01-01T00:00:00.000000000'], dtype='datetime64[ns]')

    >>> pd.unique(
    ...     pd.Series(
    ...         [
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...         ]
    ...     )
    ... )
    <DatetimeArray>
    ['2016-01-01 00:00:00-05:00']
    Length: 1, dtype: datetime64[ns, US/Eastern]

    >>> pd.unique(
    ...     pd.Index(
    ...         [
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...         ]
    ...     )
    ... )
    DatetimeIndex(['2016-01-01 00:00:00-05:00'],
            dtype='datetime64[ns, US/Eastern]',
            freq=None)

    >>> pd.unique(np.array(list("baabc"), dtype="O"))
    array(['b', 'a', 'c'], dtype=object)

    An unordered Categorical will return categories in the
    order of appearance.

    >>> pd.unique(pd.Series(pd.Categorical(list("baabc"))))
    ['b', 'a', 'c']
    Categories (3, object): ['a', 'b', 'c']

    >>> pd.unique(pd.Series(pd.Categorical(list("baabc"), categories=list("abc"))))
    ['b', 'a', 'c']
    Categories (3, object): ['a', 'b', 'c']

    An ordered Categorical preserves the category ordering.

    >>> pd.unique(
    ...     pd.Series(
    ...         pd.Categorical(list("baabc"), categories=list("abc"), ordered=True)
    ...     )
    ... )
    ['b', 'a', 'c']
    Categories (3, object): ['a' < 'b' < 'c']

    An array of tuples

    >>> pd.unique(pd.Series([("a", "b"), ("b", "a"), ("a", "c"), ("b", "a")]).values)
    array([('a', 'b'), ('b', 'a'), ('a', 'c')], dtype=object)
    """
    return unique_with_mask(values)


def nunique_ints(values: ArrayLike) -> int:
    """
    Return the number of unique values for integer array-likes.

    Significantly faster than pandas.unique for long enough sequences.
    No checks are done to ensure input is integral.

    Parameters
    ----------
    values : 1d array-like

    Returns
    -------
    int : The number of unique values in ``values``
    """
    if len(values) == 0:
        return 0
    values = _ensure_data(values)
    # bincount requires intp
    result = (np.bincount(values.ravel().astype("intp")) != 0).sum()
    return result


def unique_with_mask(values, mask: npt.NDArray[np.bool_] | None = None):
    """See algorithms.unique for docs. Takes a mask for masked arrays."""
    values = _ensure_arraylike(values, func_name="unique")

    if isinstance(values.dtype, ExtensionDtype):
        # Dispatch to extension dtype's unique.
        return values.unique()

    original = values
    hashtable, values = _get_hashtable_algo(values)

    table = hashtable(len(values))
    if mask is None:
        uniques = table.unique(values)
        uniques = _reconstruct_data(uniques, original.dtype, original)
        return uniques

    else:
        uniques, mask = table.unique(values, mask=mask)
        uniques = _reconstruct_data(uniques, original.dtype, original)
        assert mask is not None  # for mypy
        return uniques, mask.astype("bool")


unique1d = unique


_MINIMUM_COMP_ARR_LEN = 1_000_000


def isin(comps: ListLike, values: ListLike) -> npt.NDArray[np.bool_]:
    """
    Compute the isin boolean array.

    Parameters
    ----------
    comps : list-like
    values : list-like

    Returns
    -------
    ndarray[bool]
        Same length as `comps`.
    """
    if not is_list_like(comps):
        raise TypeError(
            "only list-like objects are allowed to be passed "
            f"to isin(), you passed a `{type(comps).__name__}`"
        )
    if not is_list_like(values):
        raise TypeError(
            "only list-like objects are allowed to be passed "
            f"to isin(), you passed a `{type(values).__name__}`"
        )

    if not isinstance(values, (ABCIndex, ABCSeries, ABCExtensionArray, np.ndarray)):
        orig_values = list(values)
        values = _ensure_arraylike(orig_values, func_name="isin-targets")

        if (
            len(values) > 0
            and values.dtype.kind in "iufcb"
            and not is_signed_integer_dtype(comps)
        ):
            # GH#46485 Use object to avoid upcast to float64 later
            # TODO: Share with _find_common_type_compat
            values = construct_1d_object_array_from_listlike(orig_values)

    elif isinstance(values, ABCMultiIndex):
        # Avoid raising in extract_array
        values = np.array(values)
    else:
        values = extract_array(values, extract_numpy=True, extract_range=True)

    comps_array = _ensure_arraylike(comps, func_name="isin")
    comps_array = extract_array(comps_array, extract_numpy=True)
    if not isinstance(comps_array, np.ndarray):
        # i.e. Extension Array
        return comps_array.isin(values)

    elif needs_i8_conversion(comps_array.dtype):
        # Dispatch to DatetimeLikeArrayMixin.isin
        return pd_array(comps_array).isin(values)
    elif needs_i8_conversion(values.dtype) and not is_object_dtype(comps_array.dtype):
        # e.g. comps_array are integers and values are datetime64s
        return np.zeros(comps_array.shape, dtype=bool)
        # TODO: not quite right ... Sparse/Categorical
    elif needs_i8_conversion(values.dtype):
        return isin(comps_array, values.astype(object))

    elif isinstance(values.dtype, ExtensionDtype):
        return isin(np.asarray(comps_array), np.asarray(values))

    # GH16012
    # Ensure np.isin doesn't get object types or it *may* throw an exception
    # Albeit hashmap has O(1) look-up (vs. O(logn) in sorted array),
    # isin is faster for small sizes
    if (
        len(comps_array) > _MINIMUM_COMP_ARR_LEN
        and len(values) <= 26
        and comps_array.dtype != object
    ):
        # If the values include nan we need to check for nan explicitly
        # since np.nan it not equal to np.nan
        if isna(values).any():

            def f(c, v):
                return np.logical_or(np.isin(c, v).ravel(), np.isnan(c))

        else:
            f = lambda a, b: np.isin(a, b).ravel()

    else:
        common = np_find_common_type(values.dtype, comps_array.dtype)
        values = values.astype(common, copy=False)
        comps_array = comps_array.astype(common, copy=False)
        f = htable.ismember

    return f(comps_array, values)


def factorize_array(
    values: np.ndarray,
    use_na_sentinel: bool = True,
    size_hint: int | None = None,
    na_value: object = None,
    mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[npt.NDArray[np.intp], np.ndarray]:
    """
    Factorize a numpy array to codes and uniques.

    This doesn't do any coercion of types or unboxing before factorization.

    Parameters
    ----------
    values : ndarray
    use_na_sentinel : bool, default True
        If True, the sentinel -1 will be used for NaN values. If False,
        NaN values will be encoded as non-negative integers and will not drop the
        NaN from the uniques of the values.
    size_hint : int, optional
        Passed through to the hashtable's 'get_labels' method
    na_value : object, optional
        A value in `values` to consider missing. Note: only use this
        parameter when you know that you don't have any values pandas would
        consider missing in the array (NaN for float data, iNaT for
        datetimes, etc.).
    mask : ndarray[bool], optional
        If not None, the mask is used as indicator for missing values
        (True = missing, False = valid) instead of `na_value` or
        condition "val != val".

    Returns
    -------
    codes : ndarray[np.intp]
    uniques : ndarray
    """
    original = values
    if values.dtype.kind in "mM":
        # _get_hashtable_algo will cast dt64/td64 to i8 via _ensure_data, so we
        #  need to do the same to na_value. We are assuming here that the passed
        #  na_value is an appropriately-typed NaT.
        # e.g. test_where_datetimelike_categorical
        na_value = iNaT

    hash_klass, values = _get_hashtable_algo(values)

    table = hash_klass(size_hint or len(values))
    uniques, codes = table.factorize(
        values,
        na_sentinel=-1,
        na_value=na_value,
        mask=mask,
        ignore_na=use_na_sentinel,
    )

    # re-cast e.g. i8->dt64/td64, uint8->bool
    uniques = _reconstruct_data(uniques, original.dtype, original)

    codes = ensure_platform_int(codes)
    return codes, uniques


@doc(
    values=dedent(
        """\
    values : sequence
        A 1-D sequence. Sequences that aren't pandas objects are
        coerced to ndarrays before factorization.
    """
    ),
    sort=dedent(
        """\
    sort : bool, default False
        Sort `uniques` and shuffle `codes` to maintain the
        relationship.
    """
    ),
    size_hint=dedent(
        """\
    size_hint : int, optional
        Hint to the hashtable sizer.
    """
    ),
)
def factorize(
    values,
    sort: bool = False,
    use_na_sentinel: bool = True,
    size_hint: int | None = None,
) -> tuple[np.ndarray, np.ndarray | Index]:
    """
    Encode the object as an enumerated type or categorical variable.

    This method is useful for obtaining a numeric representation of an
    array when all that matters is identifying distinct values. `factorize`
    is available as both a top-level function :func:`pandas.factorize`,
    and as a method :meth:`Series.factorize` and :meth:`Index.factorize`.

    Parameters
    ----------
    {values}{sort}
    use_na_sentinel : bool, default True
        If True, the sentinel -1 will be used for NaN values. If False,
        NaN values will be encoded as non-negative integers and will not drop the
        NaN from the uniques of the values.

        .. versionadded:: 1.5.0
    {size_hint}\

    Returns
    -------
    codes : ndarray
        An integer ndarray that's an indexer into `uniques`.
        ``uniques.take(codes)`` will have the same values as `values`.
    uniques : ndarray, Index, or Categorical
        The unique valid values. When `values` is Categorical, `uniques`
        is a Categorical. When `values` is some other pandas object, an
        `Index` is returned. Otherwise, a 1-D ndarray is returned.

        .. note::

           Even if there's a missing value in `values`, `uniques` will
           *not* contain an entry for it.

    See Also
    --------
    cut : Discretize continuous-valued array.
    unique : Find the unique value in an array.

    Notes
    -----
    Reference :ref:`the user guide <reshaping.factorize>` for more examples.

    Examples
    --------
    These examples all show factorize as a top-level method like
    ``pd.factorize(values)``. The results are identical for methods like
    :meth:`Series.factorize`.

    >>> codes, uniques = pd.factorize(np.array(['b', 'b', 'a', 'c', 'b'], dtype="O"))
    >>> codes
    array([0, 0, 1, 2, 0])
    >>> uniques
    array(['b', 'a', 'c'], dtype=object)

    With ``sort=True``, the `uniques` will be sorted, and `codes` will be
    shuffled so that the relationship is the maintained.

    >>> codes, uniques = pd.factorize(np.array(['b', 'b', 'a', 'c', 'b'], dtype="O"),
    ...                               sort=True)
    >>> codes
    array([1, 1, 0, 2, 1])
    >>> uniques
    array(['a', 'b', 'c'], dtype=object)

    When ``use_na_sentinel=True`` (the default), missing values are indicated in
    the `codes` with the sentinel value ``-1`` and missing values are not
    included in `uniques`.

    >>> codes, uniques = pd.factorize(np.array(['b', None, 'a', 'c', 'b'], dtype="O"))
    >>> codes
    array([ 0, -1,  1,  2,  0])
    >>> uniques
    array(['b', 'a', 'c'], dtype=object)

    Thus far, we've only factorized lists (which are internally coerced to
    NumPy arrays). When factorizing pandas objects, the type of `uniques`
    will differ. For Categoricals, a `Categorical` is returned.

    >>> cat = pd.Categorical(['a', 'a', 'c'], categories=['a', 'b', 'c'])
    >>> codes, uniques = pd.factorize(cat)
    >>> codes
    array([0, 0, 1])
    >>> uniques
    ['a', 'c']
    Categories (3, object): ['a', 'b', 'c']

    Notice that ``'b'`` is in ``uniques.categories``, despite not being
    present in ``cat.values``.

    For all other pandas objects, an Index of the appropriate type is
    returned.

    >>> cat = pd.Series(['a', 'a', 'c'])
    >>> codes, uniques = pd.factorize(cat)
    >>> codes
    array([0, 0, 1])
    >>> uniques
    Index(['a', 'c'], dtype='object')

    If NaN is in the values, and we want to include NaN in the uniques of the
    values, it can be achieved by setting ``use_na_sentinel=False``.

    >>> values = np.array([1, 2, 1, np.nan])
    >>> codes, uniques = pd.factorize(values)  # default: use_na_sentinel=True
    >>> codes
    array([ 0,  1,  0, -1])
    >>> uniques
    array([1., 2.])

    >>> codes, uniques = pd.factorize(values, use_na_sentinel=False)
    >>> codes
    array([0, 1, 0, 2])
    >>> uniques
    array([ 1.,  2., nan])
    """
    # Implementation notes: This method is responsible for 3 things
    # 1.) coercing data to array-like (ndarray, Index, extension array)
    # 2.) factorizing codes and uniques
    # 3.) Maybe boxing the uniques in an Index
    #
    # Step 2 is dispatched to extension types (like Categorical). They are
    # responsible only for factorization. All data coercion, sorting and boxing
    # should happen here.
    if isinstance(values, (ABCIndex, ABCSeries)):
        return values.factorize(sort=sort, use_na_sentinel=use_na_sentinel)

    values = _ensure_arraylike(values, func_name="factorize")
    original = values

    if (
        isinstance(values, (ABCDatetimeArray, ABCTimedeltaArray))
        and values.freq is not None
    ):
        # The presence of 'freq' means we can fast-path sorting and know there
        #  aren't NAs
        codes, uniques = values.factorize(sort=sort)
        return codes, uniques

    elif not isinstance(values, np.ndarray):
        # i.e. ExtensionArray
        codes, uniques = values.factorize(use_na_sentinel=use_na_sentinel)

    else:
        values = np.asarray(values)  # convert DTA/TDA/MultiIndex

        if not use_na_sentinel and values.dtype == object:
            # factorize can now handle differentiating various types of null values.
            # These can only occur when the array has object dtype.
            # However, for backwards compatibility we only use the null for the
            # provided dtype. This may be revisited in the future, see GH#48476.
            null_mask = isna(values)
            if null_mask.any():
                na_value = na_value_for_dtype(values.dtype, compat=False)
                # Don't modify (potentially user-provided) array
                values = np.where(null_mask, na_value, values)

        codes, uniques = factorize_array(
            values,
            use_na_sentinel=use_na_sentinel,
            size_hint=size_hint,
        )

    if sort and len(uniques) > 0:
        uniques, codes = safe_sort(
            uniques,
            codes,
            use_na_sentinel=use_na_sentinel,
            assume_unique=True,
            verify=False,
        )

    uniques = _reconstruct_data(uniques, original.dtype, original)

    return codes, uniques


def value_counts(
    values,
    sort: bool = True,
    ascending: bool = False,
    normalize: bool = False,
    bins=None,
    dropna: bool = True,
) -> Series:
    """
    Compute a histogram of the counts of non-null values.

    Parameters
    ----------
    values : ndarray (1-d)
    sort : bool, default True
        Sort by values
    ascending : bool, default False
        Sort in ascending order
    normalize: bool, default False
        If True then compute a relative histogram
    bins : integer, optional
        Rather than count values, group them into half-open bins,
        convenience for pd.cut, only works with numeric data
    dropna : bool, default True
        Don't include counts of NaN

    Returns
    -------
    Series
    """
    warnings.warn(
        # GH#53493
        "pandas.value_counts is deprecated and will be removed in a "
        "future version. Use pd.Series(obj).value_counts() instead.",
        FutureWarning,
        stacklevel=find_stack_level(),
    )
    return value_counts_internal(
        values,
        sort=sort,
        ascending=ascending,
        normalize=normalize,
        bins=bins,
        dropna=dropna,
    )


def value_counts_internal(
    values,
    sort: bool = True,
    ascending: bool = False,
    normalize: bool = False,
    bins=None,
    dropna: bool = True,
) -> Series:
    from pandas import (
        Index,
        Series,
    )

    index_name = getattr(values, "name", None)
    name = "proportion" if normalize else "count"

    if bins is not None:
        from pandas.core.reshape.tile import cut

        if isinstance(values, Series):
            values = values._values

        try:
            ii = cut(values, bins, include_lowest=True)
        except TypeError as err:
            raise TypeError("bins argument only works with numeric data.") from err

        # count, remove nulls (from the index), and but the bins
        result = ii.value_counts(dropna=dropna)
        result.name = name
        result = result[result.index.notna()]
        result.index = result.index.astype("interval")
        result = result.sort_index()

        # if we are dropna and we have NO values
        if dropna and (result._values == 0).all():
            result = result.iloc[0:0]

        # normalizing is by len of all (regardless of dropna)
        counts = np.array([len(ii)])

    else:
        if is_extension_array_dtype(values):
            # handle Categorical and sparse,
            result = Series(values, copy=False)._values.value_counts(dropna=dropna)
            result.name = name
            result.index.name = index_name
            counts = result._values
            if not isinstance(counts, np.ndarray):
                # e.g. ArrowExtensionArray
                counts = np.asarray(counts)

        elif isinstance(values, ABCMultiIndex):
            # GH49558
            levels = list(range(values.nlevels))
            result = (
                Series(index=values, name=name)
                .groupby(level=levels, dropna=dropna)
                .size()
            )
            result.index.names = values.names
            counts = result._values

        else:
            values = _ensure_arraylike(values, func_name="value_counts")
            keys, counts, _ = value_counts_arraylike(values, dropna)
            if keys.dtype == np.float16:
                keys = keys.astype(np.float32)

            # For backwards compatibility, we let Index do its normal type
            #  inference, _except_ for if if infers from object to bool.
            idx = Index(keys)
            if idx.dtype in [bool, "string"] and keys.dtype == object:
                idx = idx.astype(object)
            elif (
                idx.dtype != keys.dtype  # noqa: PLR1714  # # pylint: disable=R1714
                and idx.dtype != "string"
            ):
                warnings.warn(
                    # GH#56161
                    "The behavior of value_counts with object-dtype is deprecated. "
                    "In a future version, this will *not* perform dtype inference "
                    "on the resulting index. To retain the old behavior, use "
                    "`result.index = result.index.infer_objects()`",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
            idx.name = index_name

            result = Series(counts, index=idx, name=name, copy=False)

    if sort:
        result = result.sort_values(ascending=ascending)

    if normalize:
        result = result / counts.sum()

    return result


# Called once from SparseArray, otherwise could be private
def value_counts_arraylike(
    values: np.ndarray, dropna: bool, mask: npt.NDArray[np.bool_] | None = None
) -> tuple[ArrayLike, npt.NDArray[np.int64], int]:
    """
    Parameters
    ----------
    values : np.ndarray
    dropna : bool
    mask : np.ndarray[bool] or None, default None

    Returns
    -------
    uniques : np.ndarray
    counts : np.ndarray[np.int64]
    """
    original = values
    values = _ensure_data(values)

    keys, counts, na_counter = htable.value_count(values, dropna, mask=mask)

    if needs_i8_conversion(original.dtype):
        # datetime, timedelta, or period

        if dropna:
            mask = keys != iNaT
            keys, counts = keys[mask], counts[mask]

    res_keys = _reconstruct_data(keys, original.dtype, original)
    return res_keys, counts, na_counter


def duplicated(
    values: ArrayLike,
    keep: Literal["first", "last", False] = "first",
    mask: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.bool_]:
    """
    Return boolean ndarray denoting duplicate values.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
        Array over which to check for duplicate values.
    keep : {'first', 'last', False}, default 'first'
        - ``first`` : Mark duplicates as ``True`` except for the first
          occurrence.
        - ``last`` : Mark duplicates as ``True`` except for the last
          occurrence.
        - False : Mark all duplicates as ``True``.
    mask : ndarray[bool], optional
        array indicating which elements to exclude from checking

    Returns
    -------
    duplicated : ndarray[bool]
    """
    values = _ensure_data(values)
    return htable.duplicated(values, keep=keep, mask=mask)


def mode(
    values: ArrayLike, dropna: bool = True, mask: npt.NDArray[np.bool_] | None = None
) -> ArrayLike:
    """
    Returns the mode(s) of an array.

    Parameters
    ----------
    values : array-like
        Array over which to check for duplicate values.
    dropna : bool, default True
        Don't consider counts of NaN/NaT.

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    values = _ensure_arraylike(values, func_name="mode")
    original = values

    if needs_i8_conversion(values.dtype):
        # Got here with ndarray; dispatch to DatetimeArray/TimedeltaArray.
        values = ensure_wrapped_if_datetimelike(values)
        values = cast("ExtensionArray", values)
        return values._mode(dropna=dropna)

    values = _ensure_data(values)

    npresult, res_mask = htable.mode(values, dropna=dropna, mask=mask)
    if res_mask is not None:
        return npresult, res_mask  # type: ignore[return-value]

    try:
        npresult = safe_sort(npresult)
    except TypeError as err:
        warnings.warn(
            f"Unable to sort modes: {err}",
            stacklevel=find_stack_level(),
        )

    result = _reconstruct_data(npresult, original.dtype, original)
    return result


def rank(
    values: ArrayLike,
    axis: AxisInt = 0,
    method: str = "average",
    na_option: str = "keep",
    ascending: bool = True,
    pct: bool = False,
) -> npt.NDArray[np.float64]:
    """
    Rank the values along a given axis.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
        Array whose values will be ranked. The number of dimensions in this
        array must not exceed 2.
    axis : int, default 0
        Axis over which to perform rankings.
    method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
        The method by which tiebreaks are broken during the ranking.
    na_option : {'keep', 'top'}, default 'keep'
        The method by which NaNs are placed in the ranking.
        - ``keep``: rank each NaN value with a NaN ranking
        - ``top``: replace each NaN with either +/- inf so that they
                   there are ranked at the top
    ascending : bool, default True
        Whether or not the elements should be ranked in ascending order.
    pct : bool, default False
        Whether or not to the display the returned rankings in integer form
        (e.g. 1, 2, 3) or in percentile form (e.g. 0.333..., 0.666..., 1).
    """
    is_datetimelike = needs_i8_conversion(values.dtype)
    values = _ensure_data(values)

    if values.ndim == 1:
        ranks = algos.rank_1d(
            values,
            is_datetimelike=is_datetimelike,
            ties_method=method,
            ascending=ascending,
            na_option=na_option,
            pct=pct,
        )
    elif values.ndim == 2:
        ranks = algos.rank_2d(
            values,
            axis=axis,
            is_datetimelike=is_datetimelike,
            ties_method=method,
            ascending=ascending,
            na_option=na_option,
            pct=pct,
        )
    else:
        raise TypeError("Array with ndim > 2 are not supported.")

    return ranks


# ---- #
# take #
# ---- #


def take(
    arr,
    indices: TakeIndexer,
    axis: AxisInt = 0,
    allow_fill: bool = False,
    fill_value=None,
):
    """
    Take elements from an array.

    Parameters
    ----------
    arr : array-like or scalar value
        Non array-likes (sequences/scalars without a dtype) are coerced
        to an ndarray.

        .. deprecated:: 2.1.0
            Passing an argument other than a numpy.ndarray, ExtensionArray,
            Index, or Series is deprecated.

    indices : sequence of int or one-dimensional np.ndarray of int
        Indices to be taken.
    axis : int, default 0
        The axis over which to select values.
    allow_fill : bool, default False
        How to handle negative values in `indices`.

        * False: negative values in `indices` indicate positional indices
          from the right (the default). This is similar to :func:`numpy.take`.

        * True: negative values in `indices` indicate
          missing values. These values are set to `fill_value`. Any other
          negative values raise a ``ValueError``.

    fill_value : any, optional
        Fill value to use for NA-indices when `allow_fill` is True.
        This may be ``None``, in which case the default NA value for
        the type (``self.dtype.na_value``) is used.

        For multi-dimensional `arr`, each *element* is filled with
        `fill_value`.

    Returns
    -------
    ndarray or ExtensionArray
        Same type as the input.

    Raises
    ------
    IndexError
        When `indices` is out of bounds for the array.
    ValueError
        When the indexer contains negative values other than ``-1``
        and `allow_fill` is True.

    Notes
    -----
    When `allow_fill` is False, `indices` may be whatever dimensionality
    is accepted by NumPy for `arr`.

    When `allow_fill` is True, `indices` should be 1-D.

    See Also
    --------
    numpy.take : Take elements from an array along an axis.

    Examples
    --------
    >>> import pandas as pd

    With the default ``allow_fill=False``, negative numbers indicate
    positional indices from the right.

    >>> pd.api.extensions.take(np.array([10, 20, 30]), [0, 0, -1])
    array([10, 10, 30])

    Setting ``allow_fill=True`` will place `fill_value` in those positions.

    >>> pd.api.extensions.take(np.array([10, 20, 30]), [0, 0, -1], allow_fill=True)
    array([10., 10., nan])

    >>> pd.api.extensions.take(np.array([10, 20, 30]), [0, 0, -1], allow_fill=True,
    ...      fill_value=-10)
    array([ 10,  10, -10])
    """
    if not isinstance(arr, (np.ndarray, ABCExtensionArray, ABCIndex, ABCSeries)):
        # GH#52981
        warnings.warn(
            "pd.api.extensions.take accepting non-standard inputs is deprecated "
            "and will raise in a future version. Pass either a numpy.ndarray, "
            "ExtensionArray, Index, or Series instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

    if not is_array_like(arr):
        arr = np.asarray(arr)

    indices = ensure_platform_int(indices)

    if allow_fill:
        # Pandas style, -1 means NA
        validate_indices(indices, arr.shape[axis])
        result = take_nd(
            arr, indices, axis=axis, allow_fill=True, fill_value=fill_value
        )
    else:
        # NumPy style
        result = arr.take(indices, axis=axis)
    return result


# ------------ #
# searchsorted #
# ------------ #


def searchsorted(
    arr: ArrayLike,
    value: NumpyValueArrayLike | ExtensionArray,
    side: Literal["left", "right"] = "left",
    sorter: NumpySorter | None = None,
) -> npt.NDArray[np.intp] | np.intp:
    """
    Find indices where elements should be inserted to maintain order.

    Find the indices into a sorted array `arr` (a) such that, if the
    corresponding elements in `value` were inserted before the indices,
    the order of `arr` would be preserved.

    Assuming that `arr` is sorted:

    ======  ================================
    `side`  returned index `i` satisfies
    ======  ================================
    left    ``arr[i-1] < value <= self[i]``
    right   ``arr[i-1] <= value < self[i]``
    ======  ================================

    Parameters
    ----------
    arr: np.ndarray, ExtensionArray, Series
        Input array. If `sorter` is None, then it must be sorted in
        ascending order, otherwise `sorter` must be an array of indices
        that sort it.
    value : array-like or scalar
        Values to insert into `arr`.
    side : {'left', 'right'}, optional
        If 'left', the index of the first suitable location found is given.
        If 'right', return the last such index.  If there is no suitable
        index, return either 0 or N (where N is the length of `self`).
    sorter : 1-D array-like, optional
        Optional array of integer indices that sort array a into ascending
        order. They are typically the result of argsort.

    Returns
    -------
    array of ints or int
        If value is array-like, array of insertion points.
        If value is scalar, a single integer.

    See Also
    --------
    numpy.searchsorted : Similar method from NumPy.
    """
    if sorter is not None:
        sorter = ensure_platform_int(sorter)

    if (
        isinstance(arr, np.ndarray)
        and arr.dtype.kind in "iu"
        and (is_integer(value) or is_integer_dtype(value))
    ):
        # if `arr` and `value` have different dtypes, `arr` would be
        # recast by numpy, causing a slow search.
        # Before searching below, we therefore try to give `value` the
        # same dtype as `arr`, while guarding against integer overflows.
        iinfo = np.iinfo(arr.dtype.type)
        value_arr = np.array([value]) if is_integer(value) else np.array(value)
        if (value_arr >= iinfo.min).all() and (value_arr <= iinfo.max).all():
            # value within bounds, so no overflow, so can convert value dtype
            # to dtype of arr
            dtype = arr.dtype
        else:
            dtype = value_arr.dtype

        if is_integer(value):
            # We know that value is int
            value = cast(int, dtype.type(value))
        else:
            value = pd_array(cast(ArrayLike, value), dtype=dtype)
    else:
        # E.g. if `arr` is an array with dtype='datetime64[ns]'
        # and `value` is a pd.Timestamp, we may need to convert value
        arr = ensure_wrapped_if_datetimelike(arr)

    # Argument 1 to "searchsorted" of "ndarray" has incompatible type
    # "Union[NumpyValueArrayLike, ExtensionArray]"; expected "NumpyValueArrayLike"
    return arr.searchsorted(value, side=side, sorter=sorter)  # type: ignore[arg-type]


# ---- #
# diff #
# ---- #

_diff_special = {"float64", "float32", "int64", "int32", "int16", "int8"}


def diff(arr, n: int, axis: AxisInt = 0):
    """
    difference of n between self,
    analogous to s-s.shift(n)

    Parameters
    ----------
    arr : ndarray or ExtensionArray
    n : int
        number of periods
    axis : {0, 1}
        axis to shift on
    stacklevel : int, default 3
        The stacklevel for the lost dtype warning.

    Returns
    -------
    shifted
    """

    n = int(n)
    na = np.nan
    dtype = arr.dtype

    is_bool = is_bool_dtype(dtype)
    if is_bool:
        op = operator.xor
    else:
        op = operator.sub

    if isinstance(dtype, NumpyEADtype):
        # NumpyExtensionArray cannot necessarily hold shifted versions of itself.
        arr = arr.to_numpy()
        dtype = arr.dtype

    if not isinstance(arr, np.ndarray):
        # i.e ExtensionArray
        if hasattr(arr, f"__{op.__name__}__"):
            if axis != 0:
                raise ValueError(f"cannot diff {type(arr).__name__} on axis={axis}")
            return op(arr, arr.shift(n))
        else:
            raise TypeError(
                f"{type(arr).__name__} has no 'diff' method. "
                "Convert to a suitable dtype prior to calling 'diff'."
            )

    is_timedelta = False
    if arr.dtype.kind in "mM":
        dtype = np.int64
        arr = arr.view("i8")
        na = iNaT
        is_timedelta = True

    elif is_bool:
        # We have to cast in order to be able to hold np.nan
        dtype = np.object_

    elif dtype.kind in "iu":
        # We have to cast in order to be able to hold np.nan

        # int8, int16 are incompatible with float64,
        # see https://github.com/cython/cython/issues/2646
        if arr.dtype.name in ["int8", "int16"]:
            dtype = np.float32
        else:
            dtype = np.float64

    orig_ndim = arr.ndim
    if orig_ndim == 1:
        # reshape so we can always use algos.diff_2d
        arr = arr.reshape(-1, 1)
        # TODO: require axis == 0

    dtype = np.dtype(dtype)
    out_arr = np.empty(arr.shape, dtype=dtype)

    na_indexer = [slice(None)] * 2
    na_indexer[axis] = slice(None, n) if n >= 0 else slice(n, None)
    out_arr[tuple(na_indexer)] = na

    if arr.dtype.name in _diff_special:
        # TODO: can diff_2d dtype specialization troubles be fixed by defining
        #  out_arr inside diff_2d?
        algos.diff_2d(arr, out_arr, n, axis, datetimelike=is_timedelta)
    else:
        # To keep mypy happy, _res_indexer is a list while res_indexer is
        #  a tuple, ditto for lag_indexer.
        _res_indexer = [slice(None)] * 2
        _res_indexer[axis] = slice(n, None) if n >= 0 else slice(None, n)
        res_indexer = tuple(_res_indexer)

        _lag_indexer = [slice(None)] * 2
        _lag_indexer[axis] = slice(None, -n) if n > 0 else slice(-n, None)
        lag_indexer = tuple(_lag_indexer)

        out_arr[res_indexer] = op(arr[res_indexer], arr[lag_indexer])

    if is_timedelta:
        out_arr = out_arr.view("timedelta64[ns]")

    if orig_ndim == 1:
        out_arr = out_arr[:, 0]
    return out_arr


# --------------------------------------------------------------------
# Helper functions


# Note: safe_sort is in algorithms.py instead of sorting.py because it is
#  low-dependency, is used in this module, and used private methods from
#  this module.
def safe_sort(
    values: Index | ArrayLike,
    codes: npt.NDArray[np.intp] | None = None,
    use_na_sentinel: bool = True,
    assume_unique: bool = False,
    verify: bool = True,
) -> AnyArrayLike | tuple[AnyArrayLike, np.ndarray]:
    """
    Sort ``values`` and reorder corresponding ``codes``.

    ``values`` should be unique if ``codes`` is not None.
    Safe for use with mixed types (int, str), orders ints before strs.

    Parameters
    ----------
    values : list-like
        Sequence; must be unique if ``codes`` is not None.
    codes : np.ndarray[intp] or None, default None
        Indices to ``values``. All out of bound indices are treated as
        "not found" and will be masked with ``-1``.
    use_na_sentinel : bool, default True
        If True, the sentinel -1 will be used for NaN values. If False,
        NaN values will be encoded as non-negative integers and will not drop the
        NaN from the uniques of the values.
    assume_unique : bool, default False
        When True, ``values`` are assumed to be unique, which can speed up
        the calculation. Ignored when ``codes`` is None.
    verify : bool, default True
        Check if codes are out of bound for the values and put out of bound
        codes equal to ``-1``. If ``verify=False``, it is assumed there
        are no out of bound codes. Ignored when ``codes`` is None.

    Returns
    -------
    ordered : AnyArrayLike
        Sorted ``values``
    new_codes : ndarray
        Reordered ``codes``; returned when ``codes`` is not None.

    Raises
    ------
    TypeError
        * If ``values`` is not list-like or if ``codes`` is neither None
        nor list-like
        * If ``values`` cannot be sorted
    ValueError
        * If ``codes`` is not None and ``values`` contain duplicates.
    """
    if not isinstance(values, (np.ndarray, ABCExtensionArray, ABCIndex)):
        raise TypeError(
            "Only np.ndarray, ExtensionArray, and Index objects are allowed to "
            "be passed to safe_sort as values"
        )

    sorter = None
    ordered: AnyArrayLike

    if (
        not isinstance(values.dtype, ExtensionDtype)
        and lib.infer_dtype(values, skipna=False) == "mixed-integer"
    ):
        ordered = _sort_mixed(values)
    else:
        try:
            sorter = values.argsort()
            ordered = values.take(sorter)
        except (TypeError, decimal.InvalidOperation):
            # Previous sorters failed or were not applicable, try `_sort_mixed`
            # which would work, but which fails for special case of 1d arrays
            # with tuples.
            if values.size and isinstance(values[0], tuple):
                # error: Argument 1 to "_sort_tuples" has incompatible type
                # "Union[Index, ExtensionArray, ndarray[Any, Any]]"; expected
                # "ndarray[Any, Any]"
                ordered = _sort_tuples(values)  # type: ignore[arg-type]
            else:
                ordered = _sort_mixed(values)

    # codes:

    if codes is None:
        return ordered

    if not is_list_like(codes):
        raise TypeError(
            "Only list-like objects or None are allowed to "
            "be passed to safe_sort as codes"
        )
    codes = ensure_platform_int(np.asarray(codes))

    if not assume_unique and not len(unique(values)) == len(values):
        raise ValueError("values should be unique if codes is not None")

    if sorter is None:
        # mixed types
        # error: Argument 1 to "_get_hashtable_algo" has incompatible type
        # "Union[Index, ExtensionArray, ndarray[Any, Any]]"; expected
        # "ndarray[Any, Any]"
        hash_klass, values = _get_hashtable_algo(values)  # type: ignore[arg-type]
        t = hash_klass(len(values))
        t.map_locations(values)
        sorter = ensure_platform_int(t.lookup(ordered))

    if use_na_sentinel:
        # take_nd is faster, but only works for na_sentinels of -1
        order2 = sorter.argsort()
        if verify:
            mask = (codes < -len(values)) | (codes >= len(values))
            codes[mask] = 0
        else:
            mask = None
        new_codes = take_nd(order2, codes, fill_value=-1)
    else:
        reverse_indexer = np.empty(len(sorter), dtype=int)
        reverse_indexer.put(sorter, np.arange(len(sorter)))
        # Out of bound indices will be masked with `-1` next, so we
        # may deal with them here without performance loss using `mode='wrap'`
        new_codes = reverse_indexer.take(codes, mode="wrap")

        if use_na_sentinel:
            mask = codes == -1
            if verify:
                mask = mask | (codes < -len(values)) | (codes >= len(values))

    if use_na_sentinel and mask is not None:
        np.putmask(new_codes, mask, -1)

    return ordered, ensure_platform_int(new_codes)


def _sort_mixed(values) -> AnyArrayLike:
    """order ints before strings before nulls in 1d arrays"""
    str_pos = np.array([isinstance(x, str) for x in values], dtype=bool)
    null_pos = np.array([isna(x) for x in values], dtype=bool)
    num_pos = ~str_pos & ~null_pos
    str_argsort = np.argsort(values[str_pos])
    num_argsort = np.argsort(values[num_pos])
    # convert boolean arrays to positional indices, then order by underlying values
    str_locs = str_pos.nonzero()[0].take(str_argsort)
    num_locs = num_pos.nonzero()[0].take(num_argsort)
    null_locs = null_pos.nonzero()[0]
    locs = np.concatenate([num_locs, str_locs, null_locs])
    return values.take(locs)


def _sort_tuples(values: np.ndarray) -> np.ndarray:
    """
    Convert array of tuples (1d) to array of arrays (2d).
    We need to keep the columns separately as they contain different types and
    nans (can't use `np.sort` as it may fail when str and nan are mixed in a
    column as types cannot be compared).
    """
    from pandas.core.internals.construction import to_arrays
    from pandas.core.sorting import lexsort_indexer

    arrays, _ = to_arrays(values, None)
    indexer = lexsort_indexer(arrays, orders=True)
    return values[indexer]


def union_with_duplicates(
    lvals: ArrayLike | Index, rvals: ArrayLike | Index
) -> ArrayLike | Index:
    """
    Extracts the union from lvals and rvals with respect to duplicates and nans in
    both arrays.

    Parameters
    ----------
    lvals: np.ndarray or ExtensionArray
        left values which is ordered in front.
    rvals: np.ndarray or ExtensionArray
        right values ordered after lvals.

    Returns
    -------
    np.ndarray or ExtensionArray
        Containing the unsorted union of both arrays.

    Notes
    -----
    Caller is responsible for ensuring lvals.dtype == rvals.dtype.
    """
    from pandas import Series

    with warnings.catch_warnings():
        # filter warning from object dtype inference; we will end up discarding
        # the index here, so the deprecation does not affect the end result here.
        warnings.filterwarnings(
            "ignore",
            "The behavior of value_counts with object-dtype is deprecated",
            category=FutureWarning,
        )
        l_count = value_counts_internal(lvals, dropna=False)
        r_count = value_counts_internal(rvals, dropna=False)
    l_count, r_count = l_count.align(r_count, fill_value=0)
    final_count = np.maximum(l_count.values, r_count.values)
    final_count = Series(final_count, index=l_count.index, dtype="int", copy=False)
    if isinstance(lvals, ABCMultiIndex) and isinstance(rvals, ABCMultiIndex):
        unique_vals = lvals.append(rvals).unique()
    else:
        if isinstance(lvals, ABCIndex):
            lvals = lvals._values
        if isinstance(rvals, ABCIndex):
            rvals = rvals._values
        # error: List item 0 has incompatible type "Union[ExtensionArray,
        # ndarray[Any, Any], Index]"; expected "Union[ExtensionArray,
        # ndarray[Any, Any]]"
        combined = concat_compat([lvals, rvals])  # type: ignore[list-item]
        unique_vals = unique(combined)
        unique_vals = ensure_wrapped_if_datetimelike(unique_vals)
    repeats = final_count.reindex(unique_vals).values
    return np.repeat(unique_vals, repeats)


def map_array(
    arr: ArrayLike,
    mapper,
    na_action: Literal["ignore"] | None = None,
    convert: bool = True,
) -> np.ndarray | ExtensionArray | Index:
    """
    Map values using an input mapping or function.

    Parameters
    ----------
    mapper : function, dict, or Series
        Mapping correspondence.
    na_action : {None, 'ignore'}, default None
        If 'ignore', propagate NA values, without passing them to the
        mapping correspondence.
    convert : bool, default True
        Try to find better dtype for elementwise function results. If
        False, leave as dtype=object.

    Returns
    -------
    Union[ndarray, Index, ExtensionArray]
        The output of the mapping function applied to the array.
        If the function returns a tuple with more than one element
        a MultiIndex will be returned.
    """
    if na_action not in (None, "ignore"):
        msg = f"na_action must either be 'ignore' or None, {na_action} was passed"
        raise ValueError(msg)

    # we can fastpath dict/Series to an efficient map
    # as we know that we are not going to have to yield
    # python types
    if is_dict_like(mapper):
        if isinstance(mapper, dict) and hasattr(mapper, "__missing__"):
            # If a dictionary subclass defines a default value method,
            # convert mapper to a lookup function (GH #15999).
            dict_with_default = mapper
            mapper = lambda x: dict_with_default[
                np.nan if isinstance(x, float) and np.isnan(x) else x
            ]
        else:
            # Dictionary does not have a default. Thus it's safe to
            # convert to an Series for efficiency.
            # we specify the keys here to handle the
            # possibility that they are tuples

            # The return value of mapping with an empty mapper is
            # expected to be pd.Series(np.nan, ...). As np.nan is
            # of dtype float64 the return value of this method should
            # be float64 as well
            from pandas import Series

            if len(mapper) == 0:
                mapper = Series(mapper, dtype=np.float64)
            else:
                mapper = Series(mapper)

    if isinstance(mapper, ABCSeries):
        if na_action == "ignore":
            mapper = mapper[mapper.index.notna()]

        # Since values were input this means we came from either
        # a dict or a series and mapper should be an index
        indexer = mapper.index.get_indexer(arr)
        new_values = take_nd(mapper._values, indexer)

        return new_values

    if not len(arr):
        return arr.copy()

    # we must convert to python types
    values = arr.astype(object, copy=False)
    if na_action is None:
        return lib.map_infer(values, mapper, convert=convert)
    else:
        return lib.map_infer_mask(
            values, mapper, mask=isna(values).view(np.uint8), convert=convert
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\audits.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Audits (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
from . import runtime


@dataclass
class AffectedCookie:
    '''
    Information about a cookie that is affected by an inspector issue.
    '''
    #: The following three properties uniquely identify a cookie
    name: str

    path: str

    domain: str

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['path'] = self.path
        json['domain'] = self.domain
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            path=str(json['path']),
            domain=str(json['domain']),
        )


@dataclass
class AffectedRequest:
    '''
    Information about a request that is affected by an inspector issue.
    '''
    url: str

    #: The unique request id.
    request_id: typing.Optional[network.RequestId] = None

    def to_json(self):
        json = dict()
        json['url'] = self.url
        if self.request_id is not None:
            json['requestId'] = self.request_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url=str(json['url']),
            request_id=network.RequestId.from_json(json['requestId']) if 'requestId' in json else None,
        )


@dataclass
class AffectedFrame:
    '''
    Information about the frame affected by an inspector issue.
    '''
    frame_id: page.FrameId

    def to_json(self):
        json = dict()
        json['frameId'] = self.frame_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            frame_id=page.FrameId.from_json(json['frameId']),
        )


class CookieExclusionReason(enum.Enum):
    EXCLUDE_SAME_SITE_UNSPECIFIED_TREATED_AS_LAX = "ExcludeSameSiteUnspecifiedTreatedAsLax"
    EXCLUDE_SAME_SITE_NONE_INSECURE = "ExcludeSameSiteNoneInsecure"
    EXCLUDE_SAME_SITE_LAX = "ExcludeSameSiteLax"
    EXCLUDE_SAME_SITE_STRICT = "ExcludeSameSiteStrict"
    EXCLUDE_INVALID_SAME_PARTY = "ExcludeInvalidSameParty"
    EXCLUDE_SAME_PARTY_CROSS_PARTY_CONTEXT = "ExcludeSamePartyCrossPartyContext"
    EXCLUDE_DOMAIN_NON_ASCII = "ExcludeDomainNonASCII"
    EXCLUDE_THIRD_PARTY_COOKIE_BLOCKED_IN_FIRST_PARTY_SET = "ExcludeThirdPartyCookieBlockedInFirstPartySet"
    EXCLUDE_THIRD_PARTY_PHASEOUT = "ExcludeThirdPartyPhaseout"
    EXCLUDE_PORT_MISMATCH = "ExcludePortMismatch"
    EXCLUDE_SCHEME_MISMATCH = "ExcludeSchemeMismatch"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class CookieWarningReason(enum.Enum):
    WARN_SAME_SITE_UNSPECIFIED_CROSS_SITE_CONTEXT = "WarnSameSiteUnspecifiedCrossSiteContext"
    WARN_SAME_SITE_NONE_INSECURE = "WarnSameSiteNoneInsecure"
    WARN_SAME_SITE_UNSPECIFIED_LAX_ALLOW_UNSAFE = "WarnSameSiteUnspecifiedLaxAllowUnsafe"
    WARN_SAME_SITE_STRICT_LAX_DOWNGRADE_STRICT = "WarnSameSiteStrictLaxDowngradeStrict"
    WARN_SAME_SITE_STRICT_CROSS_DOWNGRADE_STRICT = "WarnSameSiteStrictCrossDowngradeStrict"
    WARN_SAME_SITE_STRICT_CROSS_DOWNGRADE_LAX = "WarnSameSiteStrictCrossDowngradeLax"
    WARN_SAME_SITE_LAX_CROSS_DOWNGRADE_STRICT = "WarnSameSiteLaxCrossDowngradeStrict"
    WARN_SAME_SITE_LAX_CROSS_DOWNGRADE_LAX = "WarnSameSiteLaxCrossDowngradeLax"
    WARN_ATTRIBUTE_VALUE_EXCEEDS_MAX_SIZE = "WarnAttributeValueExceedsMaxSize"
    WARN_DOMAIN_NON_ASCII = "WarnDomainNonASCII"
    WARN_THIRD_PARTY_PHASEOUT = "WarnThirdPartyPhaseout"
    WARN_CROSS_SITE_REDIRECT_DOWNGRADE_CHANGES_INCLUSION = "WarnCrossSiteRedirectDowngradeChangesInclusion"
    WARN_DEPRECATION_TRIAL_METADATA = "WarnDeprecationTrialMetadata"
    WARN_THIRD_PARTY_COOKIE_HEURISTIC = "WarnThirdPartyCookieHeuristic"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class CookieOperation(enum.Enum):
    SET_COOKIE = "SetCookie"
    READ_COOKIE = "ReadCookie"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class InsightType(enum.Enum):
    '''
    Represents the category of insight that a cookie issue falls under.
    '''
    GIT_HUB_RESOURCE = "GitHubResource"
    GRACE_PERIOD = "GracePeriod"
    HEURISTICS = "Heuristics"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class CookieIssueInsight:
    '''
    Information about the suggested solution to a cookie issue.
    '''
    type_: InsightType

    #: Link to table entry in third-party cookie migration readiness list.
    table_entry_url: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_.to_json()
        if self.table_entry_url is not None:
            json['tableEntryUrl'] = self.table_entry_url
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=InsightType.from_json(json['type']),
            table_entry_url=str(json['tableEntryUrl']) if 'tableEntryUrl' in json else None,
        )


@dataclass
class CookieIssueDetails:
    '''
    This information is currently necessary, as the front-end has a difficult
    time finding a specific cookie. With this, we can convey specific error
    information without the cookie.
    '''
    cookie_warning_reasons: typing.List[CookieWarningReason]

    cookie_exclusion_reasons: typing.List[CookieExclusionReason]

    #: Optionally identifies the site-for-cookies and the cookie url, which
    #: may be used by the front-end as additional context.
    operation: CookieOperation

    #: If AffectedCookie is not set then rawCookieLine contains the raw
    #: Set-Cookie header string. This hints at a problem where the
    #: cookie line is syntactically or semantically malformed in a way
    #: that no valid cookie could be created.
    cookie: typing.Optional[AffectedCookie] = None

    raw_cookie_line: typing.Optional[str] = None

    site_for_cookies: typing.Optional[str] = None

    cookie_url: typing.Optional[str] = None

    request: typing.Optional[AffectedRequest] = None

    #: The recommended solution to the issue.
    insight: typing.Optional[CookieIssueInsight] = None

    def to_json(self):
        json = dict()
        json['cookieWarningReasons'] = [i.to_json() for i in self.cookie_warning_reasons]
        json['cookieExclusionReasons'] = [i.to_json() for i in self.cookie_exclusion_reasons]
        json['operation'] = self.operation.to_json()
        if self.cookie is not None:
            json['cookie'] = self.cookie.to_json()
        if self.raw_cookie_line is not None:
            json['rawCookieLine'] = self.raw_cookie_line
        if self.site_for_cookies is not None:
            json['siteForCookies'] = self.site_for_cookies
        if self.cookie_url is not None:
            json['cookieUrl'] = self.cookie_url
        if self.request is not None:
            json['request'] = self.request.to_json()
        if self.insight is not None:
            json['insight'] = self.insight.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            cookie_warning_reasons=[CookieWarningReason.from_json(i) for i in json['cookieWarningReasons']],
            cookie_exclusion_reasons=[CookieExclusionReason.from_json(i) for i in json['cookieExclusionReasons']],
            operation=CookieOperation.from_json(json['operation']),
            cookie=AffectedCookie.from_json(json['cookie']) if 'cookie' in json else None,
            raw_cookie_line=str(json['rawCookieLine']) if 'rawCookieLine' in json else None,
            site_for_cookies=str(json['siteForCookies']) if 'siteForCookies' in json else None,
            cookie_url=str(json['cookieUrl']) if 'cookieUrl' in json else None,
            request=AffectedRequest.from_json(json['request']) if 'request' in json else None,
            insight=CookieIssueInsight.from_json(json['insight']) if 'insight' in json else None,
        )


class MixedContentResolutionStatus(enum.Enum):
    MIXED_CONTENT_BLOCKED = "MixedContentBlocked"
    MIXED_CONTENT_AUTOMATICALLY_UPGRADED = "MixedContentAutomaticallyUpgraded"
    MIXED_CONTENT_WARNING = "MixedContentWarning"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class MixedContentResourceType(enum.Enum):
    ATTRIBUTION_SRC = "AttributionSrc"
    AUDIO = "Audio"
    BEACON = "Beacon"
    CSP_REPORT = "CSPReport"
    DOWNLOAD = "Download"
    EVENT_SOURCE = "EventSource"
    FAVICON = "Favicon"
    FONT = "Font"
    FORM = "Form"
    FRAME = "Frame"
    IMAGE = "Image"
    IMPORT = "Import"
    JSON = "JSON"
    MANIFEST = "Manifest"
    PING = "Ping"
    PLUGIN_DATA = "PluginData"
    PLUGIN_RESOURCE = "PluginResource"
    PREFETCH = "Prefetch"
    RESOURCE = "Resource"
    SCRIPT = "Script"
    SERVICE_WORKER = "ServiceWorker"
    SHARED_WORKER = "SharedWorker"
    SPECULATION_RULES = "SpeculationRules"
    STYLESHEET = "Stylesheet"
    TRACK = "Track"
    VIDEO = "Video"
    WORKER = "Worker"
    XML_HTTP_REQUEST = "XMLHttpRequest"
    XSLT = "XSLT"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class MixedContentIssueDetails:
    #: The way the mixed content issue is being resolved.
    resolution_status: MixedContentResolutionStatus

    #: The unsafe http url causing the mixed content issue.
    insecure_url: str

    #: The url responsible for the call to an unsafe url.
    main_resource_url: str

    #: The type of resource causing the mixed content issue (css, js, iframe,
    #: form,...). Marked as optional because it is mapped to from
    #: blink::mojom::RequestContextType, which will be replaced
    #: by network::mojom::RequestDestination
    resource_type: typing.Optional[MixedContentResourceType] = None

    #: The mixed content request.
    #: Does not always exist (e.g. for unsafe form submission urls).
    request: typing.Optional[AffectedRequest] = None

    #: Optional because not every mixed content issue is necessarily linked to a frame.
    frame: typing.Optional[AffectedFrame] = None

    def to_json(self):
        json = dict()
        json['resolutionStatus'] = self.resolution_status.to_json()
        json['insecureURL'] = self.insecure_url
        json['mainResourceURL'] = self.main_resource_url
        if self.resource_type is not None:
            json['resourceType'] = self.resource_type.to_json()
        if self.request is not None:
            json['request'] = self.request.to_json()
        if self.frame is not None:
            json['frame'] = self.frame.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            resolution_status=MixedContentResolutionStatus.from_json(json['resolutionStatus']),
            insecure_url=str(json['insecureURL']),
            main_resource_url=str(json['mainResourceURL']),
            resource_type=MixedContentResourceType.from_json(json['resourceType']) if 'resourceType' in json else None,
            request=AffectedRequest.from_json(json['request']) if 'request' in json else None,
            frame=AffectedFrame.from_json(json['frame']) if 'frame' in json else None,
        )


class BlockedByResponseReason(enum.Enum):
    '''
    Enum indicating the reason a response has been blocked. These reasons are
    refinements of the net error BLOCKED_BY_RESPONSE.
    '''
    COEP_FRAME_RESOURCE_NEEDS_COEP_HEADER = "CoepFrameResourceNeedsCoepHeader"
    COOP_SANDBOXED_I_FRAME_CANNOT_NAVIGATE_TO_COOP_PAGE = "CoopSandboxedIFrameCannotNavigateToCoopPage"
    CORP_NOT_SAME_ORIGIN = "CorpNotSameOrigin"
    CORP_NOT_SAME_ORIGIN_AFTER_DEFAULTED_TO_SAME_ORIGIN_BY_COEP = "CorpNotSameOriginAfterDefaultedToSameOriginByCoep"
    CORP_NOT_SAME_ORIGIN_AFTER_DEFAULTED_TO_SAME_ORIGIN_BY_DIP = "CorpNotSameOriginAfterDefaultedToSameOriginByDip"
    CORP_NOT_SAME_ORIGIN_AFTER_DEFAULTED_TO_SAME_ORIGIN_BY_COEP_AND_DIP = "CorpNotSameOriginAfterDefaultedToSameOriginByCoepAndDip"
    CORP_NOT_SAME_SITE = "CorpNotSameSite"
    SRI_MESSAGE_SIGNATURE_MISMATCH = "SRIMessageSignatureMismatch"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class BlockedByResponseIssueDetails:
    '''
    Details for a request that has been blocked with the BLOCKED_BY_RESPONSE
    code. Currently only used for COEP/COOP, but may be extended to include
    some CSP errors in the future.
    '''
    request: AffectedRequest

    reason: BlockedByResponseReason

    parent_frame: typing.Optional[AffectedFrame] = None

    blocked_frame: typing.Optional[AffectedFrame] = None

    def to_json(self):
        json = dict()
        json['request'] = self.request.to_json()
        json['reason'] = self.reason.to_json()
        if self.parent_frame is not None:
            json['parentFrame'] = self.parent_frame.to_json()
        if self.blocked_frame is not None:
            json['blockedFrame'] = self.blocked_frame.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            request=AffectedRequest.from_json(json['request']),
            reason=BlockedByResponseReason.from_json(json['reason']),
            parent_frame=AffectedFrame.from_json(json['parentFrame']) if 'parentFrame' in json else None,
            blocked_frame=AffectedFrame.from_json(json['blockedFrame']) if 'blockedFrame' in json else None,
        )


class HeavyAdResolutionStatus(enum.Enum):
    HEAVY_AD_BLOCKED = "HeavyAdBlocked"
    HEAVY_AD_WARNING = "HeavyAdWarning"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class HeavyAdReason(enum.Enum):
    NETWORK_TOTAL_LIMIT = "NetworkTotalLimit"
    CPU_TOTAL_LIMIT = "CpuTotalLimit"
    CPU_PEAK_LIMIT = "CpuPeakLimit"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class HeavyAdIssueDetails:
    #: The resolution status, either blocking the content or warning.
    resolution: HeavyAdResolutionStatus

    #: The reason the ad was blocked, total network or cpu or peak cpu.
    reason: HeavyAdReason

    #: The frame that was blocked.
    frame: AffectedFrame

    def to_json(self):
        json = dict()
        json['resolution'] = self.resolution.to_json()
        json['reason'] = self.reason.to_json()
        json['frame'] = self.frame.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            resolution=HeavyAdResolutionStatus.from_json(json['resolution']),
            reason=HeavyAdReason.from_json(json['reason']),
            frame=AffectedFrame.from_json(json['frame']),
        )


class ContentSecurityPolicyViolationType(enum.Enum):
    K_INLINE_VIOLATION = "kInlineViolation"
    K_EVAL_VIOLATION = "kEvalViolation"
    K_URL_VIOLATION = "kURLViolation"
    K_SRI_VIOLATION = "kSRIViolation"
    K_TRUSTED_TYPES_SINK_VIOLATION = "kTrustedTypesSinkViolation"
    K_TRUSTED_TYPES_POLICY_VIOLATION = "kTrustedTypesPolicyViolation"
    K_WASM_EVAL_VIOLATION = "kWasmEvalViolation"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class SourceCodeLocation:
    url: str

    line_number: int

    column_number: int

    script_id: typing.Optional[runtime.ScriptId] = None

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['lineNumber'] = self.line_number
        json['columnNumber'] = self.column_number
        if self.script_id is not None:
            json['scriptId'] = self.script_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url=str(json['url']),
            line_number=int(json['lineNumber']),
            column_number=int(json['columnNumber']),
            script_id=runtime.ScriptId.from_json(json['scriptId']) if 'scriptId' in json else None,
        )


@dataclass
class ContentSecurityPolicyIssueDetails:
    #: Specific directive that is violated, causing the CSP issue.
    violated_directive: str

    is_report_only: bool

    content_security_policy_violation_type: ContentSecurityPolicyViolationType

    #: The url not included in allowed sources.
    blocked_url: typing.Optional[str] = None

    frame_ancestor: typing.Optional[AffectedFrame] = None

    source_code_location: typing.Optional[SourceCodeLocation] = None

    violating_node_id: typing.Optional[dom.BackendNodeId] = None

    def to_json(self):
        json = dict()
        json['violatedDirective'] = self.violated_directive
        json['isReportOnly'] = self.is_report_only
        json['contentSecurityPolicyViolationType'] = self.content_security_policy_violation_type.to_json()
        if self.blocked_url is not None:
            json['blockedURL'] = self.blocked_url
        if self.frame_ancestor is not None:
            json['frameAncestor'] = self.frame_ancestor.to_json()
        if self.source_code_location is not None:
            json['sourceCodeLocation'] = self.source_code_location.to_json()
        if self.violating_node_id is not None:
            json['violatingNodeId'] = self.violating_node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            violated_directive=str(json['violatedDirective']),
            is_report_only=bool(json['isReportOnly']),
            content_security_policy_violation_type=ContentSecurityPolicyViolationType.from_json(json['contentSecurityPolicyViolationType']),
            blocked_url=str(json['blockedURL']) if 'blockedURL' in json else None,
            frame_ancestor=AffectedFrame.from_json(json['frameAncestor']) if 'frameAncestor' in json else None,
            source_code_location=SourceCodeLocation.from_json(json['sourceCodeLocation']) if 'sourceCodeLocation' in json else None,
            violating_node_id=dom.BackendNodeId.from_json(json['violatingNodeId']) if 'violatingNodeId' in json else None,
        )


class SharedArrayBufferIssueType(enum.Enum):
    TRANSFER_ISSUE = "TransferIssue"
    CREATION_ISSUE = "CreationIssue"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class SharedArrayBufferIssueDetails:
    '''
    Details for a issue arising from an SAB being instantiated in, or
    transferred to a context that is not cross-origin isolated.
    '''
    source_code_location: SourceCodeLocation

    is_warning: bool

    type_: SharedArrayBufferIssueType

    def to_json(self):
        json = dict()
        json['sourceCodeLocation'] = self.source_code_location.to_json()
        json['isWarning'] = self.is_warning
        json['type'] = self.type_.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            source_code_location=SourceCodeLocation.from_json(json['sourceCodeLocation']),
            is_warning=bool(json['isWarning']),
            type_=SharedArrayBufferIssueType.from_json(json['type']),
        )


@dataclass
class LowTextContrastIssueDetails:
    violating_node_id: dom.BackendNodeId

    violating_node_selector: str

    contrast_ratio: float

    threshold_aa: float

    threshold_aaa: float

    font_size: str

    font_weight: str

    def to_json(self):
        json = dict()
        json['violatingNodeId'] = self.violating_node_id.to_json()
        json['violatingNodeSelector'] = self.violating_node_selector
        json['contrastRatio'] = self.contrast_ratio
        json['thresholdAA'] = self.threshold_aa
        json['thresholdAAA'] = self.threshold_aaa
        json['fontSize'] = self.font_size
        json['fontWeight'] = self.font_weight
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            violating_node_id=dom.BackendNodeId.from_json(json['violatingNodeId']),
            violating_node_selector=str(json['violatingNodeSelector']),
            contrast_ratio=float(json['contrastRatio']),
            threshold_aa=float(json['thresholdAA']),
            threshold_aaa=float(json['thresholdAAA']),
            font_size=str(json['fontSize']),
            font_weight=str(json['fontWeight']),
        )


@dataclass
class CorsIssueDetails:
    '''
    Details for a CORS related issue, e.g. a warning or error related to
    CORS RFC1918 enforcement.
    '''
    cors_error_status: network.CorsErrorStatus

    is_warning: bool

    request: AffectedRequest

    location: typing.Optional[SourceCodeLocation] = None

    initiator_origin: typing.Optional[str] = None

    resource_ip_address_space: typing.Optional[network.IPAddressSpace] = None

    client_security_state: typing.Optional[network.ClientSecurityState] = None

    def to_json(self):
        json = dict()
        json['corsErrorStatus'] = self.cors_error_status.to_json()
        json['isWarning'] = self.is_warning
        json['request'] = self.request.to_json()
        if self.location is not None:
            json['location'] = self.location.to_json()
        if self.initiator_origin is not None:
            json['initiatorOrigin'] = self.initiator_origin
        if self.resource_ip_address_space is not None:
            json['resourceIPAddressSpace'] = self.resource_ip_address_space.to_json()
        if self.client_security_state is not None:
            json['clientSecurityState'] = self.client_security_state.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            cors_error_status=network.CorsErrorStatus.from_json(json['corsErrorStatus']),
            is_warning=bool(json['isWarning']),
            request=AffectedRequest.from_json(json['request']),
            location=SourceCodeLocation.from_json(json['location']) if 'location' in json else None,
            initiator_origin=str(json['initiatorOrigin']) if 'initiatorOrigin' in json else None,
            resource_ip_address_space=network.IPAddressSpace.from_json(json['resourceIPAddressSpace']) if 'resourceIPAddressSpace' in json else None,
            client_security_state=network.ClientSecurityState.from_json(json['clientSecurityState']) if 'clientSecurityState' in json else None,
        )


class AttributionReportingIssueType(enum.Enum):
    PERMISSION_POLICY_DISABLED = "PermissionPolicyDisabled"
    UNTRUSTWORTHY_REPORTING_ORIGIN = "UntrustworthyReportingOrigin"
    INSECURE_CONTEXT = "InsecureContext"
    INVALID_HEADER = "InvalidHeader"
    INVALID_REGISTER_TRIGGER_HEADER = "InvalidRegisterTriggerHeader"
    SOURCE_AND_TRIGGER_HEADERS = "SourceAndTriggerHeaders"
    SOURCE_IGNORED = "SourceIgnored"
    TRIGGER_IGNORED = "TriggerIgnored"
    OS_SOURCE_IGNORED = "OsSourceIgnored"
    OS_TRIGGER_IGNORED = "OsTriggerIgnored"
    INVALID_REGISTER_OS_SOURCE_HEADER = "InvalidRegisterOsSourceHeader"
    INVALID_REGISTER_OS_TRIGGER_HEADER = "InvalidRegisterOsTriggerHeader"
    WEB_AND_OS_HEADERS = "WebAndOsHeaders"
    NO_WEB_OR_OS_SUPPORT = "NoWebOrOsSupport"
    NAVIGATION_REGISTRATION_WITHOUT_TRANSIENT_USER_ACTIVATION = "NavigationRegistrationWithoutTransientUserActivation"
    INVALID_INFO_HEADER = "InvalidInfoHeader"
    NO_REGISTER_SOURCE_HEADER = "NoRegisterSourceHeader"
    NO_REGISTER_TRIGGER_HEADER = "NoRegisterTriggerHeader"
    NO_REGISTER_OS_SOURCE_HEADER = "NoRegisterOsSourceHeader"
    NO_REGISTER_OS_TRIGGER_HEADER = "NoRegisterOsTriggerHeader"
    NAVIGATION_REGISTRATION_UNIQUE_SCOPE_ALREADY_SET = "NavigationRegistrationUniqueScopeAlreadySet"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class SharedDictionaryError(enum.Enum):
    USE_ERROR_CROSS_ORIGIN_NO_CORS_REQUEST = "UseErrorCrossOriginNoCorsRequest"
    USE_ERROR_DICTIONARY_LOAD_FAILURE = "UseErrorDictionaryLoadFailure"
    USE_ERROR_MATCHING_DICTIONARY_NOT_USED = "UseErrorMatchingDictionaryNotUsed"
    USE_ERROR_UNEXPECTED_CONTENT_DICTIONARY_HEADER = "UseErrorUnexpectedContentDictionaryHeader"
    WRITE_ERROR_COSS_ORIGIN_NO_CORS_REQUEST = "WriteErrorCossOriginNoCorsRequest"
    WRITE_ERROR_DISALLOWED_BY_SETTINGS = "WriteErrorDisallowedBySettings"
    WRITE_ERROR_EXPIRED_RESPONSE = "WriteErrorExpiredResponse"
    WRITE_ERROR_FEATURE_DISABLED = "WriteErrorFeatureDisabled"
    WRITE_ERROR_INSUFFICIENT_RESOURCES = "WriteErrorInsufficientResources"
    WRITE_ERROR_INVALID_MATCH_FIELD = "WriteErrorInvalidMatchField"
    WRITE_ERROR_INVALID_STRUCTURED_HEADER = "WriteErrorInvalidStructuredHeader"
    WRITE_ERROR_NAVIGATION_REQUEST = "WriteErrorNavigationRequest"
    WRITE_ERROR_NO_MATCH_FIELD = "WriteErrorNoMatchField"
    WRITE_ERROR_NON_LIST_MATCH_DEST_FIELD = "WriteErrorNonListMatchDestField"
    WRITE_ERROR_NON_SECURE_CONTEXT = "WriteErrorNonSecureContext"
    WRITE_ERROR_NON_STRING_ID_FIELD = "WriteErrorNonStringIdField"
    WRITE_ERROR_NON_STRING_IN_MATCH_DEST_LIST = "WriteErrorNonStringInMatchDestList"
    WRITE_ERROR_NON_STRING_MATCH_FIELD = "WriteErrorNonStringMatchField"
    WRITE_ERROR_NON_TOKEN_TYPE_FIELD = "WriteErrorNonTokenTypeField"
    WRITE_ERROR_REQUEST_ABORTED = "WriteErrorRequestAborted"
    WRITE_ERROR_SHUTTING_DOWN = "WriteErrorShuttingDown"
    WRITE_ERROR_TOO_LONG_ID_FIELD = "WriteErrorTooLongIdField"
    WRITE_ERROR_UNSUPPORTED_TYPE = "WriteErrorUnsupportedType"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class SRIMessageSignatureError(enum.Enum):
    MISSING_SIGNATURE_HEADER = "MissingSignatureHeader"
    MISSING_SIGNATURE_INPUT_HEADER = "MissingSignatureInputHeader"
    INVALID_SIGNATURE_HEADER = "InvalidSignatureHeader"
    INVALID_SIGNATURE_INPUT_HEADER = "InvalidSignatureInputHeader"
    SIGNATURE_HEADER_VALUE_IS_NOT_BYTE_SEQUENCE = "SignatureHeaderValueIsNotByteSequence"
    SIGNATURE_HEADER_VALUE_IS_PARAMETERIZED = "SignatureHeaderValueIsParameterized"
    SIGNATURE_HEADER_VALUE_IS_INCORRECT_LENGTH = "SignatureHeaderValueIsIncorrectLength"
    SIGNATURE_INPUT_HEADER_MISSING_LABEL = "SignatureInputHeaderMissingLabel"
    SIGNATURE_INPUT_HEADER_VALUE_NOT_INNER_LIST = "SignatureInputHeaderValueNotInnerList"
    SIGNATURE_INPUT_HEADER_VALUE_MISSING_COMPONENTS = "SignatureInputHeaderValueMissingComponents"
    SIGNATURE_INPUT_HEADER_INVALID_COMPONENT_TYPE = "SignatureInputHeaderInvalidComponentType"
    SIGNATURE_INPUT_HEADER_INVALID_COMPONENT_NAME = "SignatureInputHeaderInvalidComponentName"
    SIGNATURE_INPUT_HEADER_INVALID_HEADER_COMPONENT_PARAMETER = "SignatureInputHeaderInvalidHeaderComponentParameter"
    SIGNATURE_INPUT_HEADER_INVALID_DERIVED_COMPONENT_PARAMETER = "SignatureInputHeaderInvalidDerivedComponentParameter"
    SIGNATURE_INPUT_HEADER_KEY_ID_LENGTH = "SignatureInputHeaderKeyIdLength"
    SIGNATURE_INPUT_HEADER_INVALID_PARAMETER = "SignatureInputHeaderInvalidParameter"
    SIGNATURE_INPUT_HEADER_MISSING_REQUIRED_PARAMETERS = "SignatureInputHeaderMissingRequiredParameters"
    VALIDATION_FAILED_SIGNATURE_EXPIRED = "ValidationFailedSignatureExpired"
    VALIDATION_FAILED_INVALID_LENGTH = "ValidationFailedInvalidLength"
    VALIDATION_FAILED_SIGNATURE_MISMATCH = "ValidationFailedSignatureMismatch"
    VALIDATION_FAILED_INTEGRITY_MISMATCH = "ValidationFailedIntegrityMismatch"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class AttributionReportingIssueDetails:
    '''
    Details for issues around "Attribution Reporting API" usage.
    Explainer: https://github.com/WICG/attribution-reporting-api
    '''
    violation_type: AttributionReportingIssueType

    request: typing.Optional[AffectedRequest] = None

    violating_node_id: typing.Optional[dom.BackendNodeId] = None

    invalid_parameter: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['violationType'] = self.violation_type.to_json()
        if self.request is not None:
            json['request'] = self.request.to_json()
        if self.violating_node_id is not None:
            json['violatingNodeId'] = self.violating_node_id.to_json()
        if self.invalid_parameter is not None:
            json['invalidParameter'] = self.invalid_parameter
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            violation_type=AttributionReportingIssueType.from_json(json['violationType']),
            request=AffectedRequest.from_json(json['request']) if 'request' in json else None,
            violating_node_id=dom.BackendNodeId.from_json(json['violatingNodeId']) if 'violatingNodeId' in json else None,
            invalid_parameter=str(json['invalidParameter']) if 'invalidParameter' in json else None,
        )


@dataclass
class QuirksModeIssueDetails:
    '''
    Details for issues about documents in Quirks Mode
    or Limited Quirks Mode that affects page layouting.
    '''
    #: If false, it means the document's mode is "quirks"
    #: instead of "limited-quirks".
    is_limited_quirks_mode: bool

    document_node_id: dom.BackendNodeId

    url: str

    frame_id: page.FrameId

    loader_id: network.LoaderId

    def to_json(self):
        json = dict()
        json['isLimitedQuirksMode'] = self.is_limited_quirks_mode
        json['documentNodeId'] = self.document_node_id.to_json()
        json['url'] = self.url
        json['frameId'] = self.frame_id.to_json()
        json['loaderId'] = self.loader_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            is_limited_quirks_mode=bool(json['isLimitedQuirksMode']),
            document_node_id=dom.BackendNodeId.from_json(json['documentNodeId']),
            url=str(json['url']),
            frame_id=page.FrameId.from_json(json['frameId']),
            loader_id=network.LoaderId.from_json(json['loaderId']),
        )


@dataclass
class NavigatorUserAgentIssueDetails:
    url: str

    location: typing.Optional[SourceCodeLocation] = None

    def to_json(self):
        json = dict()
        json['url'] = self.url
        if self.location is not None:
            json['location'] = self.location.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url=str(json['url']),
            location=SourceCodeLocation.from_json(json['location']) if 'location' in json else None,
        )


@dataclass
class SharedDictionaryIssueDetails:
    shared_dictionary_error: SharedDictionaryError

    request: AffectedRequest

    def to_json(self):
        json = dict()
        json['sharedDictionaryError'] = self.shared_dictionary_error.to_json()
        json['request'] = self.request.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            shared_dictionary_error=SharedDictionaryError.from_json(json['sharedDictionaryError']),
            request=AffectedRequest.from_json(json['request']),
        )


@dataclass
class SRIMessageSignatureIssueDetails:
    error: SRIMessageSignatureError

    signature_base: str

    integrity_assertions: typing.List[str]

    request: AffectedRequest

    def to_json(self):
        json = dict()
        json['error'] = self.error.to_json()
        json['signatureBase'] = self.signature_base
        json['integrityAssertions'] = [i for i in self.integrity_assertions]
        json['request'] = self.request.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            error=SRIMessageSignatureError.from_json(json['error']),
            signature_base=str(json['signatureBase']),
            integrity_assertions=[str(i) for i in json['integrityAssertions']],
            request=AffectedRequest.from_json(json['request']),
        )


class GenericIssueErrorType(enum.Enum):
    FORM_LABEL_FOR_NAME_ERROR = "FormLabelForNameError"
    FORM_DUPLICATE_ID_FOR_INPUT_ERROR = "FormDuplicateIdForInputError"
    FORM_INPUT_WITH_NO_LABEL_ERROR = "FormInputWithNoLabelError"
    FORM_AUTOCOMPLETE_ATTRIBUTE_EMPTY_ERROR = "FormAutocompleteAttributeEmptyError"
    FORM_EMPTY_ID_AND_NAME_ATTRIBUTES_FOR_INPUT_ERROR = "FormEmptyIdAndNameAttributesForInputError"
    FORM_ARIA_LABELLED_BY_TO_NON_EXISTING_ID = "FormAriaLabelledByToNonExistingId"
    FORM_INPUT_ASSIGNED_AUTOCOMPLETE_VALUE_TO_ID_OR_NAME_ATTRIBUTE_ERROR = "FormInputAssignedAutocompleteValueToIdOrNameAttributeError"
    FORM_LABEL_HAS_NEITHER_FOR_NOR_NESTED_INPUT = "FormLabelHasNeitherForNorNestedInput"
    FORM_LABEL_FOR_MATCHES_NON_EXISTING_ID_ERROR = "FormLabelForMatchesNonExistingIdError"
    FORM_INPUT_HAS_WRONG_BUT_WELL_INTENDED_AUTOCOMPLETE_VALUE_ERROR = "FormInputHasWrongButWellIntendedAutocompleteValueError"
    RESPONSE_WAS_BLOCKED_BY_ORB = "ResponseWasBlockedByORB"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class GenericIssueDetails:
    '''
    Depending on the concrete errorType, different properties are set.
    '''
    #: Issues with the same errorType are aggregated in the frontend.
    error_type: GenericIssueErrorType

    frame_id: typing.Optional[page.FrameId] = None

    violating_node_id: typing.Optional[dom.BackendNodeId] = None

    violating_node_attribute: typing.Optional[str] = None

    request: typing.Optional[AffectedRequest] = None

    def to_json(self):
        json = dict()
        json['errorType'] = self.error_type.to_json()
        if self.frame_id is not None:
            json['frameId'] = self.frame_id.to_json()
        if self.violating_node_id is not None:
            json['violatingNodeId'] = self.violating_node_id.to_json()
        if self.violating_node_attribute is not None:
            json['violatingNodeAttribute'] = self.violating_node_attribute
        if self.request is not None:
            json['request'] = self.request.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            error_type=GenericIssueErrorType.from_json(json['errorType']),
            frame_id=page.FrameId.from_json(json['frameId']) if 'frameId' in json else None,
            violating_node_id=dom.BackendNodeId.from_json(json['violatingNodeId']) if 'violatingNodeId' in json else None,
            violating_node_attribute=str(json['violatingNodeAttribute']) if 'violatingNodeAttribute' in json else None,
            request=AffectedRequest.from_json(json['request']) if 'request' in json else None,
        )


@dataclass
class DeprecationIssueDetails:
    '''
    This issue tracks information needed to print a deprecation message.
    https://source.chromium.org/chromium/chromium/src/+/main:third_party/blink/renderer/core/frame/third_party/blink/renderer/core/frame/deprecation/README.md
    '''
    source_code_location: SourceCodeLocation

    #: One of the deprecation names from third_party/blink/renderer/core/frame/deprecation/deprecation.json5
    type_: str

    affected_frame: typing.Optional[AffectedFrame] = None

    def to_json(self):
        json = dict()
        json['sourceCodeLocation'] = self.source_code_location.to_json()
        json['type'] = self.type_
        if self.affected_frame is not None:
            json['affectedFrame'] = self.affected_frame.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            source_code_location=SourceCodeLocation.from_json(json['sourceCodeLocation']),
            type_=str(json['type']),
            affected_frame=AffectedFrame.from_json(json['affectedFrame']) if 'affectedFrame' in json else None,
        )


@dataclass
class BounceTrackingIssueDetails:
    '''
    This issue warns about sites in the redirect chain of a finished navigation
    that may be flagged as trackers and have their state cleared if they don't
    receive a user interaction. Note that in this context 'site' means eTLD+1.
    For example, if the URL ``https://example.test:80/bounce`` was in the
    redirect chain, the site reported would be ``example.test``.
    '''
    tracking_sites: typing.List[str]

    def to_json(self):
        json = dict()
        json['trackingSites'] = [i for i in self.tracking_sites]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            tracking_sites=[str(i) for i in json['trackingSites']],
        )


@dataclass
class CookieDeprecationMetadataIssueDetails:
    '''
    This issue warns about third-party sites that are accessing cookies on the
    current page, and have been permitted due to having a global metadata grant.
    Note that in this context 'site' means eTLD+1. For example, if the URL
    ``https://example.test:80/web_page`` was accessing cookies, the site reported
    would be ``example.test``.
    '''
    allowed_sites: typing.List[str]

    opt_out_percentage: float

    is_opt_out_top_level: bool

    operation: CookieOperation

    def to_json(self):
        json = dict()
        json['allowedSites'] = [i for i in self.allowed_sites]
        json['optOutPercentage'] = self.opt_out_percentage
        json['isOptOutTopLevel'] = self.is_opt_out_top_level
        json['operation'] = self.operation.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            allowed_sites=[str(i) for i in json['allowedSites']],
            opt_out_percentage=float(json['optOutPercentage']),
            is_opt_out_top_level=bool(json['isOptOutTopLevel']),
            operation=CookieOperation.from_json(json['operation']),
        )


class ClientHintIssueReason(enum.Enum):
    META_TAG_ALLOW_LIST_INVALID_ORIGIN = "MetaTagAllowListInvalidOrigin"
    META_TAG_MODIFIED_HTML = "MetaTagModifiedHTML"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class FederatedAuthRequestIssueDetails:
    federated_auth_request_issue_reason: FederatedAuthRequestIssueReason

    def to_json(self):
        json = dict()
        json['federatedAuthRequestIssueReason'] = self.federated_auth_request_issue_reason.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            federated_auth_request_issue_reason=FederatedAuthRequestIssueReason.from_json(json['federatedAuthRequestIssueReason']),
        )


class FederatedAuthRequestIssueReason(enum.Enum):
    '''
    Represents the failure reason when a federated authentication reason fails.
    Should be updated alongside RequestIdTokenStatus in
    third_party/blink/public/mojom/devtools/inspector_issue.mojom to include
    all cases except for success.
    '''
    SHOULD_EMBARGO = "ShouldEmbargo"
    TOO_MANY_REQUESTS = "TooManyRequests"
    WELL_KNOWN_HTTP_NOT_FOUND = "WellKnownHttpNotFound"
    WELL_KNOWN_NO_RESPONSE = "WellKnownNoResponse"
    WELL_KNOWN_INVALID_RESPONSE = "WellKnownInvalidResponse"
    WELL_KNOWN_LIST_EMPTY = "WellKnownListEmpty"
    WELL_KNOWN_INVALID_CONTENT_TYPE = "WellKnownInvalidContentType"
    CONFIG_NOT_IN_WELL_KNOWN = "ConfigNotInWellKnown"
    WELL_KNOWN_TOO_BIG = "WellKnownTooBig"
    CONFIG_HTTP_NOT_FOUND = "ConfigHttpNotFound"
    CONFIG_NO_RESPONSE = "ConfigNoResponse"
    CONFIG_INVALID_RESPONSE = "ConfigInvalidResponse"
    CONFIG_INVALID_CONTENT_TYPE = "ConfigInvalidContentType"
    CLIENT_METADATA_HTTP_NOT_FOUND = "ClientMetadataHttpNotFound"
    CLIENT_METADATA_NO_RESPONSE = "ClientMetadataNoResponse"
    CLIENT_METADATA_INVALID_RESPONSE = "ClientMetadataInvalidResponse"
    CLIENT_METADATA_INVALID_CONTENT_TYPE = "ClientMetadataInvalidContentType"
    IDP_NOT_POTENTIALLY_TRUSTWORTHY = "IdpNotPotentiallyTrustworthy"
    DISABLED_IN_SETTINGS = "DisabledInSettings"
    DISABLED_IN_FLAGS = "DisabledInFlags"
    ERROR_FETCHING_SIGNIN = "ErrorFetchingSignin"
    INVALID_SIGNIN_RESPONSE = "InvalidSigninResponse"
    ACCOUNTS_HTTP_NOT_FOUND = "AccountsHttpNotFound"
    ACCOUNTS_NO_RESPONSE = "AccountsNoResponse"
    ACCOUNTS_INVALID_RESPONSE = "AccountsInvalidResponse"
    ACCOUNTS_LIST_EMPTY = "AccountsListEmpty"
    ACCOUNTS_INVALID_CONTENT_TYPE = "AccountsInvalidContentType"
    ID_TOKEN_HTTP_NOT_FOUND = "IdTokenHttpNotFound"
    ID_TOKEN_NO_RESPONSE = "IdTokenNoResponse"
    ID_TOKEN_INVALID_RESPONSE = "IdTokenInvalidResponse"
    ID_TOKEN_IDP_ERROR_RESPONSE = "IdTokenIdpErrorResponse"
    ID_TOKEN_CROSS_SITE_IDP_ERROR_RESPONSE = "IdTokenCrossSiteIdpErrorResponse"
    ID_TOKEN_INVALID_REQUEST = "IdTokenInvalidRequest"
    ID_TOKEN_INVALID_CONTENT_TYPE = "IdTokenInvalidContentType"
    ERROR_ID_TOKEN = "ErrorIdToken"
    CANCELED = "Canceled"
    RP_PAGE_NOT_VISIBLE = "RpPageNotVisible"
    SILENT_MEDIATION_FAILURE = "SilentMediationFailure"
    THIRD_PARTY_COOKIES_BLOCKED = "ThirdPartyCookiesBlocked"
    NOT_SIGNED_IN_WITH_IDP = "NotSignedInWithIdp"
    MISSING_TRANSIENT_USER_ACTIVATION = "MissingTransientUserActivation"
    REPLACED_BY_ACTIVE_MODE = "ReplacedByActiveMode"
    INVALID_FIELDS_SPECIFIED = "InvalidFieldsSpecified"
    RELYING_PARTY_ORIGIN_IS_OPAQUE = "RelyingPartyOriginIsOpaque"
    TYPE_NOT_MATCHING = "TypeNotMatching"
    UI_DISMISSED_NO_EMBARGO = "UiDismissedNoEmbargo"
    CORS_ERROR = "CorsError"
    SUPPRESSED_BY_SEGMENTATION_PLATFORM = "SuppressedBySegmentationPlatform"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class FederatedAuthUserInfoRequestIssueDetails:
    federated_auth_user_info_request_issue_reason: FederatedAuthUserInfoRequestIssueReason

    def to_json(self):
        json = dict()
        json['federatedAuthUserInfoRequestIssueReason'] = self.federated_auth_user_info_request_issue_reason.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            federated_auth_user_info_request_issue_reason=FederatedAuthUserInfoRequestIssueReason.from_json(json['federatedAuthUserInfoRequestIssueReason']),
        )


class FederatedAuthUserInfoRequestIssueReason(enum.Enum):
    '''
    Represents the failure reason when a getUserInfo() call fails.
    Should be updated alongside FederatedAuthUserInfoRequestResult in
    third_party/blink/public/mojom/devtools/inspector_issue.mojom.
    '''
    NOT_SAME_ORIGIN = "NotSameOrigin"
    NOT_IFRAME = "NotIframe"
    NOT_POTENTIALLY_TRUSTWORTHY = "NotPotentiallyTrustworthy"
    NO_API_PERMISSION = "NoApiPermission"
    NOT_SIGNED_IN_WITH_IDP = "NotSignedInWithIdp"
    NO_ACCOUNT_SHARING_PERMISSION = "NoAccountSharingPermission"
    INVALID_CONFIG_OR_WELL_KNOWN = "InvalidConfigOrWellKnown"
    INVALID_ACCOUNTS_RESPONSE = "InvalidAccountsResponse"
    NO_RETURNING_USER_FROM_FETCHED_ACCOUNTS = "NoReturningUserFromFetchedAccounts"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class ClientHintIssueDetails:
    '''
    This issue tracks client hints related issues. It's used to deprecate old
    features, encourage the use of new ones, and provide general guidance.
    '''
    source_code_location: SourceCodeLocation

    client_hint_issue_reason: ClientHintIssueReason

    def to_json(self):
        json = dict()
        json['sourceCodeLocation'] = self.source_code_location.to_json()
        json['clientHintIssueReason'] = self.client_hint_issue_reason.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            source_code_location=SourceCodeLocation.from_json(json['sourceCodeLocation']),
            client_hint_issue_reason=ClientHintIssueReason.from_json(json['clientHintIssueReason']),
        )


@dataclass
class FailedRequestInfo:
    #: The URL that failed to load.
    url: str

    #: The failure message for the failed request.
    failure_message: str

    request_id: typing.Optional[network.RequestId] = None

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['failureMessage'] = self.failure_message
        if self.request_id is not None:
            json['requestId'] = self.request_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url=str(json['url']),
            failure_message=str(json['failureMessage']),
            request_id=network.RequestId.from_json(json['requestId']) if 'requestId' in json else None,
        )


class PartitioningBlobURLInfo(enum.Enum):
    BLOCKED_CROSS_PARTITION_FETCHING = "BlockedCrossPartitionFetching"
    ENFORCE_NOOPENER_FOR_NAVIGATION = "EnforceNoopenerForNavigation"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class PartitioningBlobURLIssueDetails:
    #: The BlobURL that failed to load.
    url: str

    #: Additional information about the Partitioning Blob URL issue.
    partitioning_blob_url_info: PartitioningBlobURLInfo

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['partitioningBlobURLInfo'] = self.partitioning_blob_url_info.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url=str(json['url']),
            partitioning_blob_url_info=PartitioningBlobURLInfo.from_json(json['partitioningBlobURLInfo']),
        )


class SelectElementAccessibilityIssueReason(enum.Enum):
    DISALLOWED_SELECT_CHILD = "DisallowedSelectChild"
    DISALLOWED_OPT_GROUP_CHILD = "DisallowedOptGroupChild"
    NON_PHRASING_CONTENT_OPTION_CHILD = "NonPhrasingContentOptionChild"
    INTERACTIVE_CONTENT_OPTION_CHILD = "InteractiveContentOptionChild"
    INTERACTIVE_CONTENT_LEGEND_CHILD = "InteractiveContentLegendChild"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class SelectElementAccessibilityIssueDetails:
    '''
    This issue warns about errors in the select element content model.
    '''
    node_id: dom.BackendNodeId

    select_element_accessibility_issue_reason: SelectElementAccessibilityIssueReason

    has_disallowed_attributes: bool

    def to_json(self):
        json = dict()
        json['nodeId'] = self.node_id.to_json()
        json['selectElementAccessibilityIssueReason'] = self.select_element_accessibility_issue_reason.to_json()
        json['hasDisallowedAttributes'] = self.has_disallowed_attributes
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            node_id=dom.BackendNodeId.from_json(json['nodeId']),
            select_element_accessibility_issue_reason=SelectElementAccessibilityIssueReason.from_json(json['selectElementAccessibilityIssueReason']),
            has_disallowed_attributes=bool(json['hasDisallowedAttributes']),
        )


class StyleSheetLoadingIssueReason(enum.Enum):
    LATE_IMPORT_RULE = "LateImportRule"
    REQUEST_FAILED = "RequestFailed"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class StylesheetLoadingIssueDetails:
    '''
    This issue warns when a referenced stylesheet couldn't be loaded.
    '''
    #: Source code position that referenced the failing stylesheet.
    source_code_location: SourceCodeLocation

    #: Reason why the stylesheet couldn't be loaded.
    style_sheet_loading_issue_reason: StyleSheetLoadingIssueReason

    #: Contains additional info when the failure was due to a request.
    failed_request_info: typing.Optional[FailedRequestInfo] = None

    def to_json(self):
        json = dict()
        json['sourceCodeLocation'] = self.source_code_location.to_json()
        json['styleSheetLoadingIssueReason'] = self.style_sheet_loading_issue_reason.to_json()
        if self.failed_request_info is not None:
            json['failedRequestInfo'] = self.failed_request_info.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            source_code_location=SourceCodeLocation.from_json(json['sourceCodeLocation']),
            style_sheet_loading_issue_reason=StyleSheetLoadingIssueReason.from_json(json['styleSheetLoadingIssueReason']),
            failed_request_info=FailedRequestInfo.from_json(json['failedRequestInfo']) if 'failedRequestInfo' in json else None,
        )


class PropertyRuleIssueReason(enum.Enum):
    INVALID_SYNTAX = "InvalidSyntax"
    INVALID_INITIAL_VALUE = "InvalidInitialValue"
    INVALID_INHERITS = "InvalidInherits"
    INVALID_NAME = "InvalidName"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class PropertyRuleIssueDetails:
    '''
    This issue warns about errors in property rules that lead to property
    registrations being ignored.
    '''
    #: Source code position of the property rule.
    source_code_location: SourceCodeLocation

    #: Reason why the property rule was discarded.
    property_rule_issue_reason: PropertyRuleIssueReason

    #: The value of the property rule property that failed to parse
    property_value: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['sourceCodeLocation'] = self.source_code_location.to_json()
        json['propertyRuleIssueReason'] = self.property_rule_issue_reason.to_json()
        if self.property_value is not None:
            json['propertyValue'] = self.property_value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            source_code_location=SourceCodeLocation.from_json(json['sourceCodeLocation']),
            property_rule_issue_reason=PropertyRuleIssueReason.from_json(json['propertyRuleIssueReason']),
            property_value=str(json['propertyValue']) if 'propertyValue' in json else None,
        )


class InspectorIssueCode(enum.Enum):
    '''
    A unique identifier for the type of issue. Each type may use one of the
    optional fields in InspectorIssueDetails to convey more specific
    information about the kind of issue.
    '''
    COOKIE_ISSUE = "CookieIssue"
    MIXED_CONTENT_ISSUE = "MixedContentIssue"
    BLOCKED_BY_RESPONSE_ISSUE = "BlockedByResponseIssue"
    HEAVY_AD_ISSUE = "HeavyAdIssue"
    CONTENT_SECURITY_POLICY_ISSUE = "ContentSecurityPolicyIssue"
    SHARED_ARRAY_BUFFER_ISSUE = "SharedArrayBufferIssue"
    LOW_TEXT_CONTRAST_ISSUE = "LowTextContrastIssue"
    CORS_ISSUE = "CorsIssue"
    ATTRIBUTION_REPORTING_ISSUE = "AttributionReportingIssue"
    QUIRKS_MODE_ISSUE = "QuirksModeIssue"
    PARTITIONING_BLOB_URL_ISSUE = "PartitioningBlobURLIssue"
    NAVIGATOR_USER_AGENT_ISSUE = "NavigatorUserAgentIssue"
    GENERIC_ISSUE = "GenericIssue"
    DEPRECATION_ISSUE = "DeprecationIssue"
    CLIENT_HINT_ISSUE = "ClientHintIssue"
    FEDERATED_AUTH_REQUEST_ISSUE = "FederatedAuthRequestIssue"
    BOUNCE_TRACKING_ISSUE = "BounceTrackingIssue"
    COOKIE_DEPRECATION_METADATA_ISSUE = "CookieDeprecationMetadataIssue"
    STYLESHEET_LOADING_ISSUE = "StylesheetLoadingIssue"
    FEDERATED_AUTH_USER_INFO_REQUEST_ISSUE = "FederatedAuthUserInfoRequestIssue"
    PROPERTY_RULE_ISSUE = "PropertyRuleIssue"
    SHARED_DICTIONARY_ISSUE = "SharedDictionaryIssue"
    SELECT_ELEMENT_ACCESSIBILITY_ISSUE = "SelectElementAccessibilityIssue"
    SRI_MESSAGE_SIGNATURE_ISSUE = "SRIMessageSignatureIssue"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class InspectorIssueDetails:
    '''
    This struct holds a list of optional fields with additional information
    specific to the kind of issue. When adding a new issue code, please also
    add a new optional field to this type.
    '''
    cookie_issue_details: typing.Optional[CookieIssueDetails] = None

    mixed_content_issue_details: typing.Optional[MixedContentIssueDetails] = None

    blocked_by_response_issue_details: typing.Optional[BlockedByResponseIssueDetails] = None

    heavy_ad_issue_details: typing.Optional[HeavyAdIssueDetails] = None

    content_security_policy_issue_details: typing.Optional[ContentSecurityPolicyIssueDetails] = None

    shared_array_buffer_issue_details: typing.Optional[SharedArrayBufferIssueDetails] = None

    low_text_contrast_issue_details: typing.Optional[LowTextContrastIssueDetails] = None

    cors_issue_details: typing.Optional[CorsIssueDetails] = None

    attribution_reporting_issue_details: typing.Optional[AttributionReportingIssueDetails] = None

    quirks_mode_issue_details: typing.Optional[QuirksModeIssueDetails] = None

    partitioning_blob_url_issue_details: typing.Optional[PartitioningBlobURLIssueDetails] = None

    navigator_user_agent_issue_details: typing.Optional[NavigatorUserAgentIssueDetails] = None

    generic_issue_details: typing.Optional[GenericIssueDetails] = None

    deprecation_issue_details: typing.Optional[DeprecationIssueDetails] = None

    client_hint_issue_details: typing.Optional[ClientHintIssueDetails] = None

    federated_auth_request_issue_details: typing.Optional[FederatedAuthRequestIssueDetails] = None

    bounce_tracking_issue_details: typing.Optional[BounceTrackingIssueDetails] = None

    cookie_deprecation_metadata_issue_details: typing.Optional[CookieDeprecationMetadataIssueDetails] = None

    stylesheet_loading_issue_details: typing.Optional[StylesheetLoadingIssueDetails] = None

    property_rule_issue_details: typing.Optional[PropertyRuleIssueDetails] = None

    federated_auth_user_info_request_issue_details: typing.Optional[FederatedAuthUserInfoRequestIssueDetails] = None

    shared_dictionary_issue_details: typing.Optional[SharedDictionaryIssueDetails] = None

    select_element_accessibility_issue_details: typing.Optional[SelectElementAccessibilityIssueDetails] = None

    sri_message_signature_issue_details: typing.Optional[SRIMessageSignatureIssueDetails] = None

    def to_json(self):
        json = dict()
        if self.cookie_issue_details is not None:
            json['cookieIssueDetails'] = self.cookie_issue_details.to_json()
        if self.mixed_content_issue_details is not None:
            json['mixedContentIssueDetails'] = self.mixed_content_issue_details.to_json()
        if self.blocked_by_response_issue_details is not None:
            json['blockedByResponseIssueDetails'] = self.blocked_by_response_issue_details.to_json()
        if self.heavy_ad_issue_details is not None:
            json['heavyAdIssueDetails'] = self.heavy_ad_issue_details.to_json()
        if self.content_security_policy_issue_details is not None:
            json['contentSecurityPolicyIssueDetails'] = self.content_security_policy_issue_details.to_json()
        if self.shared_array_buffer_issue_details is not None:
            json['sharedArrayBufferIssueDetails'] = self.shared_array_buffer_issue_details.to_json()
        if self.low_text_contrast_issue_details is not None:
            json['lowTextContrastIssueDetails'] = self.low_text_contrast_issue_details.to_json()
        if self.cors_issue_details is not None:
            json['corsIssueDetails'] = self.cors_issue_details.to_json()
        if self.attribution_reporting_issue_details is not None:
            json['attributionReportingIssueDetails'] = self.attribution_reporting_issue_details.to_json()
        if self.quirks_mode_issue_details is not None:
            json['quirksModeIssueDetails'] = self.quirks_mode_issue_details.to_json()
        if self.partitioning_blob_url_issue_details is not None:
            json['partitioningBlobURLIssueDetails'] = self.partitioning_blob_url_issue_details.to_json()
        if self.navigator_user_agent_issue_details is not None:
            json['navigatorUserAgentIssueDetails'] = self.navigator_user_agent_issue_details.to_json()
        if self.generic_issue_details is not None:
            json['genericIssueDetails'] = self.generic_issue_details.to_json()
        if self.deprecation_issue_details is not None:
            json['deprecationIssueDetails'] = self.deprecation_issue_details.to_json()
        if self.client_hint_issue_details is not None:
            json['clientHintIssueDetails'] = self.client_hint_issue_details.to_json()
        if self.federated_auth_request_issue_details is not None:
            json['federatedAuthRequestIssueDetails'] = self.federated_auth_request_issue_details.to_json()
        if self.bounce_tracking_issue_details is not None:
            json['bounceTrackingIssueDetails'] = self.bounce_tracking_issue_details.to_json()
        if self.cookie_deprecation_metadata_issue_details is not None:
            json['cookieDeprecationMetadataIssueDetails'] = self.cookie_deprecation_metadata_issue_details.to_json()
        if self.stylesheet_loading_issue_details is not None:
            json['stylesheetLoadingIssueDetails'] = self.stylesheet_loading_issue_details.to_json()
        if self.property_rule_issue_details is not None:
            json['propertyRuleIssueDetails'] = self.property_rule_issue_details.to_json()
        if self.federated_auth_user_info_request_issue_details is not None:
            json['federatedAuthUserInfoRequestIssueDetails'] = self.federated_auth_user_info_request_issue_details.to_json()
        if self.shared_dictionary_issue_details is not None:
            json['sharedDictionaryIssueDetails'] = self.shared_dictionary_issue_details.to_json()
        if self.select_element_accessibility_issue_details is not None:
            json['selectElementAccessibilityIssueDetails'] = self.select_element_accessibility_issue_details.to_json()
        if self.sri_message_signature_issue_details is not None:
            json['sriMessageSignatureIssueDetails'] = self.sri_message_signature_issue_details.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            cookie_issue_details=CookieIssueDetails.from_json(json['cookieIssueDetails']) if 'cookieIssueDetails' in json else None,
            mixed_content_issue_details=MixedContentIssueDetails.from_json(json['mixedContentIssueDetails']) if 'mixedContentIssueDetails' in json else None,
            blocked_by_response_issue_details=BlockedByResponseIssueDetails.from_json(json['blockedByResponseIssueDetails']) if 'blockedByResponseIssueDetails' in json else None,
            heavy_ad_issue_details=HeavyAdIssueDetails.from_json(json['heavyAdIssueDetails']) if 'heavyAdIssueDetails' in json else None,
            content_security_policy_issue_details=ContentSecurityPolicyIssueDetails.from_json(json['contentSecurityPolicyIssueDetails']) if 'contentSecurityPolicyIssueDetails' in json else None,
            shared_array_buffer_issue_details=SharedArrayBufferIssueDetails.from_json(json['sharedArrayBufferIssueDetails']) if 'sharedArrayBufferIssueDetails' in json else None,
            low_text_contrast_issue_details=LowTextContrastIssueDetails.from_json(json['lowTextContrastIssueDetails']) if 'lowTextContrastIssueDetails' in json else None,
            cors_issue_details=CorsIssueDetails.from_json(json['corsIssueDetails']) if 'corsIssueDetails' in json else None,
            attribution_reporting_issue_details=AttributionReportingIssueDetails.from_json(json['attributionReportingIssueDetails']) if 'attributionReportingIssueDetails' in json else None,
            quirks_mode_issue_details=QuirksModeIssueDetails.from_json(json['quirksModeIssueDetails']) if 'quirksModeIssueDetails' in json else None,
            partitioning_blob_url_issue_details=PartitioningBlobURLIssueDetails.from_json(json['partitioningBlobURLIssueDetails']) if 'partitioningBlobURLIssueDetails' in json else None,
            navigator_user_agent_issue_details=NavigatorUserAgentIssueDetails.from_json(json['navigatorUserAgentIssueDetails']) if 'navigatorUserAgentIssueDetails' in json else None,
            generic_issue_details=GenericIssueDetails.from_json(json['genericIssueDetails']) if 'genericIssueDetails' in json else None,
            deprecation_issue_details=DeprecationIssueDetails.from_json(json['deprecationIssueDetails']) if 'deprecationIssueDetails' in json else None,
            client_hint_issue_details=ClientHintIssueDetails.from_json(json['clientHintIssueDetails']) if 'clientHintIssueDetails' in json else None,
            federated_auth_request_issue_details=FederatedAuthRequestIssueDetails.from_json(json['federatedAuthRequestIssueDetails']) if 'federatedAuthRequestIssueDetails' in json else None,
            bounce_tracking_issue_details=BounceTrackingIssueDetails.from_json(json['bounceTrackingIssueDetails']) if 'bounceTrackingIssueDetails' in json else None,
            cookie_deprecation_metadata_issue_details=CookieDeprecationMetadataIssueDetails.from_json(json['cookieDeprecationMetadataIssueDetails']) if 'cookieDeprecationMetadataIssueDetails' in json else None,
            stylesheet_loading_issue_details=StylesheetLoadingIssueDetails.from_json(json['stylesheetLoadingIssueDetails']) if 'stylesheetLoadingIssueDetails' in json else None,
            property_rule_issue_details=PropertyRuleIssueDetails.from_json(json['propertyRuleIssueDetails']) if 'propertyRuleIssueDetails' in json else None,
            federated_auth_user_info_request_issue_details=FederatedAuthUserInfoRequestIssueDetails.from_json(json['federatedAuthUserInfoRequestIssueDetails']) if 'federatedAuthUserInfoRequestIssueDetails' in json else None,
            shared_dictionary_issue_details=SharedDictionaryIssueDetails.from_json(json['sharedDictionaryIssueDetails']) if 'sharedDictionaryIssueDetails' in json else None,
            select_element_accessibility_issue_details=SelectElementAccessibilityIssueDetails.from_json(json['selectElementAccessibilityIssueDetails']) if 'selectElementAccessibilityIssueDetails' in json else None,
            sri_message_signature_issue_details=SRIMessageSignatureIssueDetails.from_json(json['sriMessageSignatureIssueDetails']) if 'sriMessageSignatureIssueDetails' in json else None,
        )


class IssueId(str):
    '''
    A unique id for a DevTools inspector issue. Allows other entities (e.g.
    exceptions, CDP message, console messages, etc.) to reference an issue.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> IssueId:
        return cls(json)

    def __repr__(self):
        return 'IssueId({})'.format(super().__repr__())


@dataclass
class InspectorIssue:
    '''
    An inspector issue reported from the back-end.
    '''
    code: InspectorIssueCode

    details: InspectorIssueDetails

    #: A unique id for this issue. May be omitted if no other entity (e.g.
    #: exception, CDP message, etc.) is referencing this issue.
    issue_id: typing.Optional[IssueId] = None

    def to_json(self):
        json = dict()
        json['code'] = self.code.to_json()
        json['details'] = self.details.to_json()
        if self.issue_id is not None:
            json['issueId'] = self.issue_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            code=InspectorIssueCode.from_json(json['code']),
            details=InspectorIssueDetails.from_json(json['details']),
            issue_id=IssueId.from_json(json['issueId']) if 'issueId' in json else None,
        )


def get_encoded_response(
        request_id: network.RequestId,
        encoding: str,
        quality: typing.Optional[float] = None,
        size_only: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.Optional[str], int, int]]:
    '''
    Returns the response body and size if it were re-encoded with the specified settings. Only
    applies to images.

    :param request_id: Identifier of the network request to get content for.
    :param encoding: The encoding to use.
    :param quality: *(Optional)* The quality of the encoding (0-1). (defaults to 1)
    :param size_only: *(Optional)* Whether to only return the size information (defaults to false).
    :returns: A tuple with the following items:

        0. **body** - *(Optional)* The encoded body as a base64 string. Omitted if sizeOnly is true.
        1. **originalSize** - Size before re-encoding.
        2. **encodedSize** - Size after re-encoding.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    params['encoding'] = encoding
    if quality is not None:
        params['quality'] = quality
    if size_only is not None:
        params['sizeOnly'] = size_only
    cmd_dict: T_JSON_DICT = {
        'method': 'Audits.getEncodedResponse',
        'params': params,
    }
    json = yield cmd_dict
    return (
        str(json['body']) if 'body' in json else None,
        int(json['originalSize']),
        int(json['encodedSize'])
    )


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables issues domain, prevents further issues from being reported to the client.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Audits.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables issues domain, sends the issues collected so far to the client by means of the
    ``issueAdded`` event.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Audits.enable',
    }
    json = yield cmd_dict


def check_contrast(
        report_aaa: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Runs the contrast check for the target page. Found issues are reported
    using Audits.issueAdded event.

    :param report_aaa: *(Optional)* Whether to report WCAG AAA level issues. Default is false.
    '''
    params: T_JSON_DICT = dict()
    if report_aaa is not None:
        params['reportAAA'] = report_aaa
    cmd_dict: T_JSON_DICT = {
        'method': 'Audits.checkContrast',
        'params': params,
    }
    json = yield cmd_dict


def check_forms_issues() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[GenericIssueDetails]]:
    '''
    Runs the form issues check for the target page. Found issues are reported
    using Audits.issueAdded event.

    :returns: 
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Audits.checkFormsIssues',
    }
    json = yield cmd_dict
    return [GenericIssueDetails.from_json(i) for i in json['formIssues']]


@event_class('Audits.issueAdded')
@dataclass
class IssueAdded:
    issue: InspectorIssue

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> IssueAdded:
        return cls(
            issue=InspectorIssue.from_json(json['issue'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\audits.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Audits (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
from . import runtime


@dataclass
class AffectedCookie:
    '''
    Information about a cookie that is affected by an inspector issue.
    '''
    #: The following three properties uniquely identify a cookie
    name: str

    path: str

    domain: str

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['path'] = self.path
        json['domain'] = self.domain
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            path=str(json['path']),
            domain=str(json['domain']),
        )


@dataclass
class AffectedRequest:
    '''
    Information about a request that is affected by an inspector issue.
    '''
    url: str

    #: The unique request id.
    request_id: typing.Optional[network.RequestId] = None

    def to_json(self):
        json = dict()
        json['url'] = self.url
        if self.request_id is not None:
            json['requestId'] = self.request_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url=str(json['url']),
            request_id=network.RequestId.from_json(json['requestId']) if 'requestId' in json else None,
        )


@dataclass
class AffectedFrame:
    '''
    Information about the frame affected by an inspector issue.
    '''
    frame_id: page.FrameId

    def to_json(self):
        json = dict()
        json['frameId'] = self.frame_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            frame_id=page.FrameId.from_json(json['frameId']),
        )


class CookieExclusionReason(enum.Enum):
    EXCLUDE_SAME_SITE_UNSPECIFIED_TREATED_AS_LAX = "ExcludeSameSiteUnspecifiedTreatedAsLax"
    EXCLUDE_SAME_SITE_NONE_INSECURE = "ExcludeSameSiteNoneInsecure"
    EXCLUDE_SAME_SITE_LAX = "ExcludeSameSiteLax"
    EXCLUDE_SAME_SITE_STRICT = "ExcludeSameSiteStrict"
    EXCLUDE_INVALID_SAME_PARTY = "ExcludeInvalidSameParty"
    EXCLUDE_SAME_PARTY_CROSS_PARTY_CONTEXT = "ExcludeSamePartyCrossPartyContext"
    EXCLUDE_DOMAIN_NON_ASCII = "ExcludeDomainNonASCII"
    EXCLUDE_THIRD_PARTY_COOKIE_BLOCKED_IN_FIRST_PARTY_SET = "ExcludeThirdPartyCookieBlockedInFirstPartySet"
    EXCLUDE_THIRD_PARTY_PHASEOUT = "ExcludeThirdPartyPhaseout"
    EXCLUDE_PORT_MISMATCH = "ExcludePortMismatch"
    EXCLUDE_SCHEME_MISMATCH = "ExcludeSchemeMismatch"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class CookieWarningReason(enum.Enum):
    WARN_SAME_SITE_UNSPECIFIED_CROSS_SITE_CONTEXT = "WarnSameSiteUnspecifiedCrossSiteContext"
    WARN_SAME_SITE_NONE_INSECURE = "WarnSameSiteNoneInsecure"
    WARN_SAME_SITE_UNSPECIFIED_LAX_ALLOW_UNSAFE = "WarnSameSiteUnspecifiedLaxAllowUnsafe"
    WARN_SAME_SITE_STRICT_LAX_DOWNGRADE_STRICT = "WarnSameSiteStrictLaxDowngradeStrict"
    WARN_SAME_SITE_STRICT_CROSS_DOWNGRADE_STRICT = "WarnSameSiteStrictCrossDowngradeStrict"
    WARN_SAME_SITE_STRICT_CROSS_DOWNGRADE_LAX = "WarnSameSiteStrictCrossDowngradeLax"
    WARN_SAME_SITE_LAX_CROSS_DOWNGRADE_STRICT = "WarnSameSiteLaxCrossDowngradeStrict"
    WARN_SAME_SITE_LAX_CROSS_DOWNGRADE_LAX = "WarnSameSiteLaxCrossDowngradeLax"
    WARN_ATTRIBUTE_VALUE_EXCEEDS_MAX_SIZE = "WarnAttributeValueExceedsMaxSize"
    WARN_DOMAIN_NON_ASCII = "WarnDomainNonASCII"
    WARN_THIRD_PARTY_PHASEOUT = "WarnThirdPartyPhaseout"
    WARN_CROSS_SITE_REDIRECT_DOWNGRADE_CHANGES_INCLUSION = "WarnCrossSiteRedirectDowngradeChangesInclusion"
    WARN_DEPRECATION_TRIAL_METADATA = "WarnDeprecationTrialMetadata"
    WARN_THIRD_PARTY_COOKIE_HEURISTIC = "WarnThirdPartyCookieHeuristic"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class CookieOperation(enum.Enum):
    SET_COOKIE = "SetCookie"
    READ_COOKIE = "ReadCookie"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class InsightType(enum.Enum):
    '''
    Represents the category of insight that a cookie issue falls under.
    '''
    GIT_HUB_RESOURCE = "GitHubResource"
    GRACE_PERIOD = "GracePeriod"
    HEURISTICS = "Heuristics"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class CookieIssueInsight:
    '''
    Information about the suggested solution to a cookie issue.
    '''
    type_: InsightType

    #: Link to table entry in third-party cookie migration readiness list.
    table_entry_url: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_.to_json()
        if self.table_entry_url is not None:
            json['tableEntryUrl'] = self.table_entry_url
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=InsightType.from_json(json['type']),
            table_entry_url=str(json['tableEntryUrl']) if 'tableEntryUrl' in json else None,
        )


@dataclass
class CookieIssueDetails:
    '''
    This information is currently necessary, as the front-end has a difficult
    time finding a specific cookie. With this, we can convey specific error
    information without the cookie.
    '''
    cookie_warning_reasons: typing.List[CookieWarningReason]

    cookie_exclusion_reasons: typing.List[CookieExclusionReason]

    #: Optionally identifies the site-for-cookies and the cookie url, which
    #: may be used by the front-end as additional context.
    operation: CookieOperation

    #: If AffectedCookie is not set then rawCookieLine contains the raw
    #: Set-Cookie header string. This hints at a problem where the
    #: cookie line is syntactically or semantically malformed in a way
    #: that no valid cookie could be created.
    cookie: typing.Optional[AffectedCookie] = None

    raw_cookie_line: typing.Optional[str] = None

    site_for_cookies: typing.Optional[str] = None

    cookie_url: typing.Optional[str] = None

    request: typing.Optional[AffectedRequest] = None

    #: The recommended solution to the issue.
    insight: typing.Optional[CookieIssueInsight] = None

    def to_json(self):
        json = dict()
        json['cookieWarningReasons'] = [i.to_json() for i in self.cookie_warning_reasons]
        json['cookieExclusionReasons'] = [i.to_json() for i in self.cookie_exclusion_reasons]
        json['operation'] = self.operation.to_json()
        if self.cookie is not None:
            json['cookie'] = self.cookie.to_json()
        if self.raw_cookie_line is not None:
            json['rawCookieLine'] = self.raw_cookie_line
        if self.site_for_cookies is not None:
            json['siteForCookies'] = self.site_for_cookies
        if self.cookie_url is not None:
            json['cookieUrl'] = self.cookie_url
        if self.request is not None:
            json['request'] = self.request.to_json()
        if self.insight is not None:
            json['insight'] = self.insight.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            cookie_warning_reasons=[CookieWarningReason.from_json(i) for i in json['cookieWarningReasons']],
            cookie_exclusion_reasons=[CookieExclusionReason.from_json(i) for i in json['cookieExclusionReasons']],
            operation=CookieOperation.from_json(json['operation']),
            cookie=AffectedCookie.from_json(json['cookie']) if 'cookie' in json else None,
            raw_cookie_line=str(json['rawCookieLine']) if 'rawCookieLine' in json else None,
            site_for_cookies=str(json['siteForCookies']) if 'siteForCookies' in json else None,
            cookie_url=str(json['cookieUrl']) if 'cookieUrl' in json else None,
            request=AffectedRequest.from_json(json['request']) if 'request' in json else None,
            insight=CookieIssueInsight.from_json(json['insight']) if 'insight' in json else None,
        )


class MixedContentResolutionStatus(enum.Enum):
    MIXED_CONTENT_BLOCKED = "MixedContentBlocked"
    MIXED_CONTENT_AUTOMATICALLY_UPGRADED = "MixedContentAutomaticallyUpgraded"
    MIXED_CONTENT_WARNING = "MixedContentWarning"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class MixedContentResourceType(enum.Enum):
    ATTRIBUTION_SRC = "AttributionSrc"
    AUDIO = "Audio"
    BEACON = "Beacon"
    CSP_REPORT = "CSPReport"
    DOWNLOAD = "Download"
    EVENT_SOURCE = "EventSource"
    FAVICON = "Favicon"
    FONT = "Font"
    FORM = "Form"
    FRAME = "Frame"
    IMAGE = "Image"
    IMPORT = "Import"
    JSON = "JSON"
    MANIFEST = "Manifest"
    PING = "Ping"
    PLUGIN_DATA = "PluginData"
    PLUGIN_RESOURCE = "PluginResource"
    PREFETCH = "Prefetch"
    RESOURCE = "Resource"
    SCRIPT = "Script"
    SERVICE_WORKER = "ServiceWorker"
    SHARED_WORKER = "SharedWorker"
    SPECULATION_RULES = "SpeculationRules"
    STYLESHEET = "Stylesheet"
    TRACK = "Track"
    VIDEO = "Video"
    WORKER = "Worker"
    XML_HTTP_REQUEST = "XMLHttpRequest"
    XSLT = "XSLT"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class MixedContentIssueDetails:
    #: The way the mixed content issue is being resolved.
    resolution_status: MixedContentResolutionStatus

    #: The unsafe http url causing the mixed content issue.
    insecure_url: str

    #: The url responsible for the call to an unsafe url.
    main_resource_url: str

    #: The type of resource causing the mixed content issue (css, js, iframe,
    #: form,...). Marked as optional because it is mapped to from
    #: blink::mojom::RequestContextType, which will be replaced
    #: by network::mojom::RequestDestination
    resource_type: typing.Optional[MixedContentResourceType] = None

    #: The mixed content request.
    #: Does not always exist (e.g. for unsafe form submission urls).
    request: typing.Optional[AffectedRequest] = None

    #: Optional because not every mixed content issue is necessarily linked to a frame.
    frame: typing.Optional[AffectedFrame] = None

    def to_json(self):
        json = dict()
        json['resolutionStatus'] = self.resolution_status.to_json()
        json['insecureURL'] = self.insecure_url
        json['mainResourceURL'] = self.main_resource_url
        if self.resource_type is not None:
            json['resourceType'] = self.resource_type.to_json()
        if self.request is not None:
            json['request'] = self.request.to_json()
        if self.frame is not None:
            json['frame'] = self.frame.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            resolution_status=MixedContentResolutionStatus.from_json(json['resolutionStatus']),
            insecure_url=str(json['insecureURL']),
            main_resource_url=str(json['mainResourceURL']),
            resource_type=MixedContentResourceType.from_json(json['resourceType']) if 'resourceType' in json else None,
            request=AffectedRequest.from_json(json['request']) if 'request' in json else None,
            frame=AffectedFrame.from_json(json['frame']) if 'frame' in json else None,
        )


class BlockedByResponseReason(enum.Enum):
    '''
    Enum indicating the reason a response has been blocked. These reasons are
    refinements of the net error BLOCKED_BY_RESPONSE.
    '''
    COEP_FRAME_RESOURCE_NEEDS_COEP_HEADER = "CoepFrameResourceNeedsCoepHeader"
    COOP_SANDBOXED_I_FRAME_CANNOT_NAVIGATE_TO_COOP_PAGE = "CoopSandboxedIFrameCannotNavigateToCoopPage"
    CORP_NOT_SAME_ORIGIN = "CorpNotSameOrigin"
    CORP_NOT_SAME_ORIGIN_AFTER_DEFAULTED_TO_SAME_ORIGIN_BY_COEP = "CorpNotSameOriginAfterDefaultedToSameOriginByCoep"
    CORP_NOT_SAME_ORIGIN_AFTER_DEFAULTED_TO_SAME_ORIGIN_BY_DIP = "CorpNotSameOriginAfterDefaultedToSameOriginByDip"
    CORP_NOT_SAME_ORIGIN_AFTER_DEFAULTED_TO_SAME_ORIGIN_BY_COEP_AND_DIP = "CorpNotSameOriginAfterDefaultedToSameOriginByCoepAndDip"
    CORP_NOT_SAME_SITE = "CorpNotSameSite"
    SRI_MESSAGE_SIGNATURE_MISMATCH = "SRIMessageSignatureMismatch"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class BlockedByResponseIssueDetails:
    '''
    Details for a request that has been blocked with the BLOCKED_BY_RESPONSE
    code. Currently only used for COEP/COOP, but may be extended to include
    some CSP errors in the future.
    '''
    request: AffectedRequest

    reason: BlockedByResponseReason

    parent_frame: typing.Optional[AffectedFrame] = None

    blocked_frame: typing.Optional[AffectedFrame] = None

    def to_json(self):
        json = dict()
        json['request'] = self.request.to_json()
        json['reason'] = self.reason.to_json()
        if self.parent_frame is not None:
            json['parentFrame'] = self.parent_frame.to_json()
        if self.blocked_frame is not None:
            json['blockedFrame'] = self.blocked_frame.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            request=AffectedRequest.from_json(json['request']),
            reason=BlockedByResponseReason.from_json(json['reason']),
            parent_frame=AffectedFrame.from_json(json['parentFrame']) if 'parentFrame' in json else None,
            blocked_frame=AffectedFrame.from_json(json['blockedFrame']) if 'blockedFrame' in json else None,
        )


class HeavyAdResolutionStatus(enum.Enum):
    HEAVY_AD_BLOCKED = "HeavyAdBlocked"
    HEAVY_AD_WARNING = "HeavyAdWarning"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class HeavyAdReason(enum.Enum):
    NETWORK_TOTAL_LIMIT = "NetworkTotalLimit"
    CPU_TOTAL_LIMIT = "CpuTotalLimit"
    CPU_PEAK_LIMIT = "CpuPeakLimit"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class HeavyAdIssueDetails:
    #: The resolution status, either blocking the content or warning.
    resolution: HeavyAdResolutionStatus

    #: The reason the ad was blocked, total network or cpu or peak cpu.
    reason: HeavyAdReason

    #: The frame that was blocked.
    frame: AffectedFrame

    def to_json(self):
        json = dict()
        json['resolution'] = self.resolution.to_json()
        json['reason'] = self.reason.to_json()
        json['frame'] = self.frame.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            resolution=HeavyAdResolutionStatus.from_json(json['resolution']),
            reason=HeavyAdReason.from_json(json['reason']),
            frame=AffectedFrame.from_json(json['frame']),
        )


class ContentSecurityPolicyViolationType(enum.Enum):
    K_INLINE_VIOLATION = "kInlineViolation"
    K_EVAL_VIOLATION = "kEvalViolation"
    K_URL_VIOLATION = "kURLViolation"
    K_SRI_VIOLATION = "kSRIViolation"
    K_TRUSTED_TYPES_SINK_VIOLATION = "kTrustedTypesSinkViolation"
    K_TRUSTED_TYPES_POLICY_VIOLATION = "kTrustedTypesPolicyViolation"
    K_WASM_EVAL_VIOLATION = "kWasmEvalViolation"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class SourceCodeLocation:
    url: str

    line_number: int

    column_number: int

    script_id: typing.Optional[runtime.ScriptId] = None

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['lineNumber'] = self.line_number
        json['columnNumber'] = self.column_number
        if self.script_id is not None:
            json['scriptId'] = self.script_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url=str(json['url']),
            line_number=int(json['lineNumber']),
            column_number=int(json['columnNumber']),
            script_id=runtime.ScriptId.from_json(json['scriptId']) if 'scriptId' in json else None,
        )


@dataclass
class ContentSecurityPolicyIssueDetails:
    #: Specific directive that is violated, causing the CSP issue.
    violated_directive: str

    is_report_only: bool

    content_security_policy_violation_type: ContentSecurityPolicyViolationType

    #: The url not included in allowed sources.
    blocked_url: typing.Optional[str] = None

    frame_ancestor: typing.Optional[AffectedFrame] = None

    source_code_location: typing.Optional[SourceCodeLocation] = None

    violating_node_id: typing.Optional[dom.BackendNodeId] = None

    def to_json(self):
        json = dict()
        json['violatedDirective'] = self.violated_directive
        json['isReportOnly'] = self.is_report_only
        json['contentSecurityPolicyViolationType'] = self.content_security_policy_violation_type.to_json()
        if self.blocked_url is not None:
            json['blockedURL'] = self.blocked_url
        if self.frame_ancestor is not None:
            json['frameAncestor'] = self.frame_ancestor.to_json()
        if self.source_code_location is not None:
            json['sourceCodeLocation'] = self.source_code_location.to_json()
        if self.violating_node_id is not None:
            json['violatingNodeId'] = self.violating_node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            violated_directive=str(json['violatedDirective']),
            is_report_only=bool(json['isReportOnly']),
            content_security_policy_violation_type=ContentSecurityPolicyViolationType.from_json(json['contentSecurityPolicyViolationType']),
            blocked_url=str(json['blockedURL']) if 'blockedURL' in json else None,
            frame_ancestor=AffectedFrame.from_json(json['frameAncestor']) if 'frameAncestor' in json else None,
            source_code_location=SourceCodeLocation.from_json(json['sourceCodeLocation']) if 'sourceCodeLocation' in json else None,
            violating_node_id=dom.BackendNodeId.from_json(json['violatingNodeId']) if 'violatingNodeId' in json else None,
        )


class SharedArrayBufferIssueType(enum.Enum):
    TRANSFER_ISSUE = "TransferIssue"
    CREATION_ISSUE = "CreationIssue"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class SharedArrayBufferIssueDetails:
    '''
    Details for a issue arising from an SAB being instantiated in, or
    transferred to a context that is not cross-origin isolated.
    '''
    source_code_location: SourceCodeLocation

    is_warning: bool

    type_: SharedArrayBufferIssueType

    def to_json(self):
        json = dict()
        json['sourceCodeLocation'] = self.source_code_location.to_json()
        json['isWarning'] = self.is_warning
        json['type'] = self.type_.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            source_code_location=SourceCodeLocation.from_json(json['sourceCodeLocation']),
            is_warning=bool(json['isWarning']),
            type_=SharedArrayBufferIssueType.from_json(json['type']),
        )


@dataclass
class LowTextContrastIssueDetails:
    violating_node_id: dom.BackendNodeId

    violating_node_selector: str

    contrast_ratio: float

    threshold_aa: float

    threshold_aaa: float

    font_size: str

    font_weight: str

    def to_json(self):
        json = dict()
        json['violatingNodeId'] = self.violating_node_id.to_json()
        json['violatingNodeSelector'] = self.violating_node_selector
        json['contrastRatio'] = self.contrast_ratio
        json['thresholdAA'] = self.threshold_aa
        json['thresholdAAA'] = self.threshold_aaa
        json['fontSize'] = self.font_size
        json['fontWeight'] = self.font_weight
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            violating_node_id=dom.BackendNodeId.from_json(json['violatingNodeId']),
            violating_node_selector=str(json['violatingNodeSelector']),
            contrast_ratio=float(json['contrastRatio']),
            threshold_aa=float(json['thresholdAA']),
            threshold_aaa=float(json['thresholdAAA']),
            font_size=str(json['fontSize']),
            font_weight=str(json['fontWeight']),
        )


@dataclass
class CorsIssueDetails:
    '''
    Details for a CORS related issue, e.g. a warning or error related to
    CORS RFC1918 enforcement.
    '''
    cors_error_status: network.CorsErrorStatus

    is_warning: bool

    request: AffectedRequest

    location: typing.Optional[SourceCodeLocation] = None

    initiator_origin: typing.Optional[str] = None

    resource_ip_address_space: typing.Optional[network.IPAddressSpace] = None

    client_security_state: typing.Optional[network.ClientSecurityState] = None

    def to_json(self):
        json = dict()
        json['corsErrorStatus'] = self.cors_error_status.to_json()
        json['isWarning'] = self.is_warning
        json['request'] = self.request.to_json()
        if self.location is not None:
            json['location'] = self.location.to_json()
        if self.initiator_origin is not None:
            json['initiatorOrigin'] = self.initiator_origin
        if self.resource_ip_address_space is not None:
            json['resourceIPAddressSpace'] = self.resource_ip_address_space.to_json()
        if self.client_security_state is not None:
            json['clientSecurityState'] = self.client_security_state.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            cors_error_status=network.CorsErrorStatus.from_json(json['corsErrorStatus']),
            is_warning=bool(json['isWarning']),
            request=AffectedRequest.from_json(json['request']),
            location=SourceCodeLocation.from_json(json['location']) if 'location' in json else None,
            initiator_origin=str(json['initiatorOrigin']) if 'initiatorOrigin' in json else None,
            resource_ip_address_space=network.IPAddressSpace.from_json(json['resourceIPAddressSpace']) if 'resourceIPAddressSpace' in json else None,
            client_security_state=network.ClientSecurityState.from_json(json['clientSecurityState']) if 'clientSecurityState' in json else None,
        )


class AttributionReportingIssueType(enum.Enum):
    PERMISSION_POLICY_DISABLED = "PermissionPolicyDisabled"
    UNTRUSTWORTHY_REPORTING_ORIGIN = "UntrustworthyReportingOrigin"
    INSECURE_CONTEXT = "InsecureContext"
    INVALID_HEADER = "InvalidHeader"
    INVALID_REGISTER_TRIGGER_HEADER = "InvalidRegisterTriggerHeader"
    SOURCE_AND_TRIGGER_HEADERS = "SourceAndTriggerHeaders"
    SOURCE_IGNORED = "SourceIgnored"
    TRIGGER_IGNORED = "TriggerIgnored"
    OS_SOURCE_IGNORED = "OsSourceIgnored"
    OS_TRIGGER_IGNORED = "OsTriggerIgnored"
    INVALID_REGISTER_OS_SOURCE_HEADER = "InvalidRegisterOsSourceHeader"
    INVALID_REGISTER_OS_TRIGGER_HEADER = "InvalidRegisterOsTriggerHeader"
    WEB_AND_OS_HEADERS = "WebAndOsHeaders"
    NO_WEB_OR_OS_SUPPORT = "NoWebOrOsSupport"
    NAVIGATION_REGISTRATION_WITHOUT_TRANSIENT_USER_ACTIVATION = "NavigationRegistrationWithoutTransientUserActivation"
    INVALID_INFO_HEADER = "InvalidInfoHeader"
    NO_REGISTER_SOURCE_HEADER = "NoRegisterSourceHeader"
    NO_REGISTER_TRIGGER_HEADER = "NoRegisterTriggerHeader"
    NO_REGISTER_OS_SOURCE_HEADER = "NoRegisterOsSourceHeader"
    NO_REGISTER_OS_TRIGGER_HEADER = "NoRegisterOsTriggerHeader"
    NAVIGATION_REGISTRATION_UNIQUE_SCOPE_ALREADY_SET = "NavigationRegistrationUniqueScopeAlreadySet"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class SharedDictionaryError(enum.Enum):
    USE_ERROR_CROSS_ORIGIN_NO_CORS_REQUEST = "UseErrorCrossOriginNoCorsRequest"
    USE_ERROR_DICTIONARY_LOAD_FAILURE = "UseErrorDictionaryLoadFailure"
    USE_ERROR_MATCHING_DICTIONARY_NOT_USED = "UseErrorMatchingDictionaryNotUsed"
    USE_ERROR_UNEXPECTED_CONTENT_DICTIONARY_HEADER = "UseErrorUnexpectedContentDictionaryHeader"
    WRITE_ERROR_COSS_ORIGIN_NO_CORS_REQUEST = "WriteErrorCossOriginNoCorsRequest"
    WRITE_ERROR_DISALLOWED_BY_SETTINGS = "WriteErrorDisallowedBySettings"
    WRITE_ERROR_EXPIRED_RESPONSE = "WriteErrorExpiredResponse"
    WRITE_ERROR_FEATURE_DISABLED = "WriteErrorFeatureDisabled"
    WRITE_ERROR_INSUFFICIENT_RESOURCES = "WriteErrorInsufficientResources"
    WRITE_ERROR_INVALID_MATCH_FIELD = "WriteErrorInvalidMatchField"
    WRITE_ERROR_INVALID_STRUCTURED_HEADER = "WriteErrorInvalidStructuredHeader"
    WRITE_ERROR_NAVIGATION_REQUEST = "WriteErrorNavigationRequest"
    WRITE_ERROR_NO_MATCH_FIELD = "WriteErrorNoMatchField"
    WRITE_ERROR_NON_LIST_MATCH_DEST_FIELD = "WriteErrorNonListMatchDestField"
    WRITE_ERROR_NON_SECURE_CONTEXT = "WriteErrorNonSecureContext"
    WRITE_ERROR_NON_STRING_ID_FIELD = "WriteErrorNonStringIdField"
    WRITE_ERROR_NON_STRING_IN_MATCH_DEST_LIST = "WriteErrorNonStringInMatchDestList"
    WRITE_ERROR_NON_STRING_MATCH_FIELD = "WriteErrorNonStringMatchField"
    WRITE_ERROR_NON_TOKEN_TYPE_FIELD = "WriteErrorNonTokenTypeField"
    WRITE_ERROR_REQUEST_ABORTED = "WriteErrorRequestAborted"
    WRITE_ERROR_SHUTTING_DOWN = "WriteErrorShuttingDown"
    WRITE_ERROR_TOO_LONG_ID_FIELD = "WriteErrorTooLongIdField"
    WRITE_ERROR_UNSUPPORTED_TYPE = "WriteErrorUnsupportedType"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class SRIMessageSignatureError(enum.Enum):
    MISSING_SIGNATURE_HEADER = "MissingSignatureHeader"
    MISSING_SIGNATURE_INPUT_HEADER = "MissingSignatureInputHeader"
    INVALID_SIGNATURE_HEADER = "InvalidSignatureHeader"
    INVALID_SIGNATURE_INPUT_HEADER = "InvalidSignatureInputHeader"
    SIGNATURE_HEADER_VALUE_IS_NOT_BYTE_SEQUENCE = "SignatureHeaderValueIsNotByteSequence"
    SIGNATURE_HEADER_VALUE_IS_PARAMETERIZED = "SignatureHeaderValueIsParameterized"
    SIGNATURE_HEADER_VALUE_IS_INCORRECT_LENGTH = "SignatureHeaderValueIsIncorrectLength"
    SIGNATURE_INPUT_HEADER_MISSING_LABEL = "SignatureInputHeaderMissingLabel"
    SIGNATURE_INPUT_HEADER_VALUE_NOT_INNER_LIST = "SignatureInputHeaderValueNotInnerList"
    SIGNATURE_INPUT_HEADER_VALUE_MISSING_COMPONENTS = "SignatureInputHeaderValueMissingComponents"
    SIGNATURE_INPUT_HEADER_INVALID_COMPONENT_TYPE = "SignatureInputHeaderInvalidComponentType"
    SIGNATURE_INPUT_HEADER_INVALID_COMPONENT_NAME = "SignatureInputHeaderInvalidComponentName"
    SIGNATURE_INPUT_HEADER_INVALID_HEADER_COMPONENT_PARAMETER = "SignatureInputHeaderInvalidHeaderComponentParameter"
    SIGNATURE_INPUT_HEADER_INVALID_DERIVED_COMPONENT_PARAMETER = "SignatureInputHeaderInvalidDerivedComponentParameter"
    SIGNATURE_INPUT_HEADER_KEY_ID_LENGTH = "SignatureInputHeaderKeyIdLength"
    SIGNATURE_INPUT_HEADER_INVALID_PARAMETER = "SignatureInputHeaderInvalidParameter"
    SIGNATURE_INPUT_HEADER_MISSING_REQUIRED_PARAMETERS = "SignatureInputHeaderMissingRequiredParameters"
    VALIDATION_FAILED_SIGNATURE_EXPIRED = "ValidationFailedSignatureExpired"
    VALIDATION_FAILED_INVALID_LENGTH = "ValidationFailedInvalidLength"
    VALIDATION_FAILED_SIGNATURE_MISMATCH = "ValidationFailedSignatureMismatch"
    VALIDATION_FAILED_INTEGRITY_MISMATCH = "ValidationFailedIntegrityMismatch"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class AttributionReportingIssueDetails:
    '''
    Details for issues around "Attribution Reporting API" usage.
    Explainer: https://github.com/WICG/attribution-reporting-api
    '''
    violation_type: AttributionReportingIssueType

    request: typing.Optional[AffectedRequest] = None

    violating_node_id: typing.Optional[dom.BackendNodeId] = None

    invalid_parameter: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['violationType'] = self.violation_type.to_json()
        if self.request is not None:
            json['request'] = self.request.to_json()
        if self.violating_node_id is not None:
            json['violatingNodeId'] = self.violating_node_id.to_json()
        if self.invalid_parameter is not None:
            json['invalidParameter'] = self.invalid_parameter
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            violation_type=AttributionReportingIssueType.from_json(json['violationType']),
            request=AffectedRequest.from_json(json['request']) if 'request' in json else None,
            violating_node_id=dom.BackendNodeId.from_json(json['violatingNodeId']) if 'violatingNodeId' in json else None,
            invalid_parameter=str(json['invalidParameter']) if 'invalidParameter' in json else None,
        )


@dataclass
class QuirksModeIssueDetails:
    '''
    Details for issues about documents in Quirks Mode
    or Limited Quirks Mode that affects page layouting.
    '''
    #: If false, it means the document's mode is "quirks"
    #: instead of "limited-quirks".
    is_limited_quirks_mode: bool

    document_node_id: dom.BackendNodeId

    url: str

    frame_id: page.FrameId

    loader_id: network.LoaderId

    def to_json(self):
        json = dict()
        json['isLimitedQuirksMode'] = self.is_limited_quirks_mode
        json['documentNodeId'] = self.document_node_id.to_json()
        json['url'] = self.url
        json['frameId'] = self.frame_id.to_json()
        json['loaderId'] = self.loader_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            is_limited_quirks_mode=bool(json['isLimitedQuirksMode']),
            document_node_id=dom.BackendNodeId.from_json(json['documentNodeId']),
            url=str(json['url']),
            frame_id=page.FrameId.from_json(json['frameId']),
            loader_id=network.LoaderId.from_json(json['loaderId']),
        )


@dataclass
class NavigatorUserAgentIssueDetails:
    url: str

    location: typing.Optional[SourceCodeLocation] = None

    def to_json(self):
        json = dict()
        json['url'] = self.url
        if self.location is not None:
            json['location'] = self.location.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url=str(json['url']),
            location=SourceCodeLocation.from_json(json['location']) if 'location' in json else None,
        )


@dataclass
class SharedDictionaryIssueDetails:
    shared_dictionary_error: SharedDictionaryError

    request: AffectedRequest

    def to_json(self):
        json = dict()
        json['sharedDictionaryError'] = self.shared_dictionary_error.to_json()
        json['request'] = self.request.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            shared_dictionary_error=SharedDictionaryError.from_json(json['sharedDictionaryError']),
            request=AffectedRequest.from_json(json['request']),
        )


@dataclass
class SRIMessageSignatureIssueDetails:
    error: SRIMessageSignatureError

    signature_base: str

    integrity_assertions: typing.List[str]

    request: AffectedRequest

    def to_json(self):
        json = dict()
        json['error'] = self.error.to_json()
        json['signatureBase'] = self.signature_base
        json['integrityAssertions'] = [i for i in self.integrity_assertions]
        json['request'] = self.request.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            error=SRIMessageSignatureError.from_json(json['error']),
            signature_base=str(json['signatureBase']),
            integrity_assertions=[str(i) for i in json['integrityAssertions']],
            request=AffectedRequest.from_json(json['request']),
        )


class GenericIssueErrorType(enum.Enum):
    FORM_LABEL_FOR_NAME_ERROR = "FormLabelForNameError"
    FORM_DUPLICATE_ID_FOR_INPUT_ERROR = "FormDuplicateIdForInputError"
    FORM_INPUT_WITH_NO_LABEL_ERROR = "FormInputWithNoLabelError"
    FORM_AUTOCOMPLETE_ATTRIBUTE_EMPTY_ERROR = "FormAutocompleteAttributeEmptyError"
    FORM_EMPTY_ID_AND_NAME_ATTRIBUTES_FOR_INPUT_ERROR = "FormEmptyIdAndNameAttributesForInputError"
    FORM_ARIA_LABELLED_BY_TO_NON_EXISTING_ID = "FormAriaLabelledByToNonExistingId"
    FORM_INPUT_ASSIGNED_AUTOCOMPLETE_VALUE_TO_ID_OR_NAME_ATTRIBUTE_ERROR = "FormInputAssignedAutocompleteValueToIdOrNameAttributeError"
    FORM_LABEL_HAS_NEITHER_FOR_NOR_NESTED_INPUT = "FormLabelHasNeitherForNorNestedInput"
    FORM_LABEL_FOR_MATCHES_NON_EXISTING_ID_ERROR = "FormLabelForMatchesNonExistingIdError"
    FORM_INPUT_HAS_WRONG_BUT_WELL_INTENDED_AUTOCOMPLETE_VALUE_ERROR = "FormInputHasWrongButWellIntendedAutocompleteValueError"
    RESPONSE_WAS_BLOCKED_BY_ORB = "ResponseWasBlockedByORB"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class GenericIssueDetails:
    '''
    Depending on the concrete errorType, different properties are set.
    '''
    #: Issues with the same errorType are aggregated in the frontend.
    error_type: GenericIssueErrorType

    frame_id: typing.Optional[page.FrameId] = None

    violating_node_id: typing.Optional[dom.BackendNodeId] = None

    violating_node_attribute: typing.Optional[str] = None

    request: typing.Optional[AffectedRequest] = None

    def to_json(self):
        json = dict()
        json['errorType'] = self.error_type.to_json()
        if self.frame_id is not None:
            json['frameId'] = self.frame_id.to_json()
        if self.violating_node_id is not None:
            json['violatingNodeId'] = self.violating_node_id.to_json()
        if self.violating_node_attribute is not None:
            json['violatingNodeAttribute'] = self.violating_node_attribute
        if self.request is not None:
            json['request'] = self.request.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            error_type=GenericIssueErrorType.from_json(json['errorType']),
            frame_id=page.FrameId.from_json(json['frameId']) if 'frameId' in json else None,
            violating_node_id=dom.BackendNodeId.from_json(json['violatingNodeId']) if 'violatingNodeId' in json else None,
            violating_node_attribute=str(json['violatingNodeAttribute']) if 'violatingNodeAttribute' in json else None,
            request=AffectedRequest.from_json(json['request']) if 'request' in json else None,
        )


@dataclass
class DeprecationIssueDetails:
    '''
    This issue tracks information needed to print a deprecation message.
    https://source.chromium.org/chromium/chromium/src/+/main:third_party/blink/renderer/core/frame/third_party/blink/renderer/core/frame/deprecation/README.md
    '''
    source_code_location: SourceCodeLocation

    #: One of the deprecation names from third_party/blink/renderer/core/frame/deprecation/deprecation.json5
    type_: str

    affected_frame: typing.Optional[AffectedFrame] = None

    def to_json(self):
        json = dict()
        json['sourceCodeLocation'] = self.source_code_location.to_json()
        json['type'] = self.type_
        if self.affected_frame is not None:
            json['affectedFrame'] = self.affected_frame.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            source_code_location=SourceCodeLocation.from_json(json['sourceCodeLocation']),
            type_=str(json['type']),
            affected_frame=AffectedFrame.from_json(json['affectedFrame']) if 'affectedFrame' in json else None,
        )


@dataclass
class BounceTrackingIssueDetails:
    '''
    This issue warns about sites in the redirect chain of a finished navigation
    that may be flagged as trackers and have their state cleared if they don't
    receive a user interaction. Note that in this context 'site' means eTLD+1.
    For example, if the URL ``https://example.test:80/bounce`` was in the
    redirect chain, the site reported would be ``example.test``.
    '''
    tracking_sites: typing.List[str]

    def to_json(self):
        json = dict()
        json['trackingSites'] = [i for i in self.tracking_sites]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            tracking_sites=[str(i) for i in json['trackingSites']],
        )


@dataclass
class CookieDeprecationMetadataIssueDetails:
    '''
    This issue warns about third-party sites that are accessing cookies on the
    current page, and have been permitted due to having a global metadata grant.
    Note that in this context 'site' means eTLD+1. For example, if the URL
    ``https://example.test:80/web_page`` was accessing cookies, the site reported
    would be ``example.test``.
    '''
    allowed_sites: typing.List[str]

    opt_out_percentage: float

    is_opt_out_top_level: bool

    operation: CookieOperation

    def to_json(self):
        json = dict()
        json['allowedSites'] = [i for i in self.allowed_sites]
        json['optOutPercentage'] = self.opt_out_percentage
        json['isOptOutTopLevel'] = self.is_opt_out_top_level
        json['operation'] = self.operation.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            allowed_sites=[str(i) for i in json['allowedSites']],
            opt_out_percentage=float(json['optOutPercentage']),
            is_opt_out_top_level=bool(json['isOptOutTopLevel']),
            operation=CookieOperation.from_json(json['operation']),
        )


class ClientHintIssueReason(enum.Enum):
    META_TAG_ALLOW_LIST_INVALID_ORIGIN = "MetaTagAllowListInvalidOrigin"
    META_TAG_MODIFIED_HTML = "MetaTagModifiedHTML"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class FederatedAuthRequestIssueDetails:
    federated_auth_request_issue_reason: FederatedAuthRequestIssueReason

    def to_json(self):
        json = dict()
        json['federatedAuthRequestIssueReason'] = self.federated_auth_request_issue_reason.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            federated_auth_request_issue_reason=FederatedAuthRequestIssueReason.from_json(json['federatedAuthRequestIssueReason']),
        )


class FederatedAuthRequestIssueReason(enum.Enum):
    '''
    Represents the failure reason when a federated authentication reason fails.
    Should be updated alongside RequestIdTokenStatus in
    third_party/blink/public/mojom/devtools/inspector_issue.mojom to include
    all cases except for success.
    '''
    SHOULD_EMBARGO = "ShouldEmbargo"
    TOO_MANY_REQUESTS = "TooManyRequests"
    WELL_KNOWN_HTTP_NOT_FOUND = "WellKnownHttpNotFound"
    WELL_KNOWN_NO_RESPONSE = "WellKnownNoResponse"
    WELL_KNOWN_INVALID_RESPONSE = "WellKnownInvalidResponse"
    WELL_KNOWN_LIST_EMPTY = "WellKnownListEmpty"
    WELL_KNOWN_INVALID_CONTENT_TYPE = "WellKnownInvalidContentType"
    CONFIG_NOT_IN_WELL_KNOWN = "ConfigNotInWellKnown"
    WELL_KNOWN_TOO_BIG = "WellKnownTooBig"
    CONFIG_HTTP_NOT_FOUND = "ConfigHttpNotFound"
    CONFIG_NO_RESPONSE = "ConfigNoResponse"
    CONFIG_INVALID_RESPONSE = "ConfigInvalidResponse"
    CONFIG_INVALID_CONTENT_TYPE = "ConfigInvalidContentType"
    CLIENT_METADATA_HTTP_NOT_FOUND = "ClientMetadataHttpNotFound"
    CLIENT_METADATA_NO_RESPONSE = "ClientMetadataNoResponse"
    CLIENT_METADATA_INVALID_RESPONSE = "ClientMetadataInvalidResponse"
    CLIENT_METADATA_INVALID_CONTENT_TYPE = "ClientMetadataInvalidContentType"
    IDP_NOT_POTENTIALLY_TRUSTWORTHY = "IdpNotPotentiallyTrustworthy"
    DISABLED_IN_SETTINGS = "DisabledInSettings"
    DISABLED_IN_FLAGS = "DisabledInFlags"
    ERROR_FETCHING_SIGNIN = "ErrorFetchingSignin"
    INVALID_SIGNIN_RESPONSE = "InvalidSigninResponse"
    ACCOUNTS_HTTP_NOT_FOUND = "AccountsHttpNotFound"
    ACCOUNTS_NO_RESPONSE = "AccountsNoResponse"
    ACCOUNTS_INVALID_RESPONSE = "AccountsInvalidResponse"
    ACCOUNTS_LIST_EMPTY = "AccountsListEmpty"
    ACCOUNTS_INVALID_CONTENT_TYPE = "AccountsInvalidContentType"
    ID_TOKEN_HTTP_NOT_FOUND = "IdTokenHttpNotFound"
    ID_TOKEN_NO_RESPONSE = "IdTokenNoResponse"
    ID_TOKEN_INVALID_RESPONSE = "IdTokenInvalidResponse"
    ID_TOKEN_IDP_ERROR_RESPONSE = "IdTokenIdpErrorResponse"
    ID_TOKEN_CROSS_SITE_IDP_ERROR_RESPONSE = "IdTokenCrossSiteIdpErrorResponse"
    ID_TOKEN_INVALID_REQUEST = "IdTokenInvalidRequest"
    ID_TOKEN_INVALID_CONTENT_TYPE = "IdTokenInvalidContentType"
    ERROR_ID_TOKEN = "ErrorIdToken"
    CANCELED = "Canceled"
    RP_PAGE_NOT_VISIBLE = "RpPageNotVisible"
    SILENT_MEDIATION_FAILURE = "SilentMediationFailure"
    THIRD_PARTY_COOKIES_BLOCKED = "ThirdPartyCookiesBlocked"
    NOT_SIGNED_IN_WITH_IDP = "NotSignedInWithIdp"
    MISSING_TRANSIENT_USER_ACTIVATION = "MissingTransientUserActivation"
    REPLACED_BY_ACTIVE_MODE = "ReplacedByActiveMode"
    INVALID_FIELDS_SPECIFIED = "InvalidFieldsSpecified"
    RELYING_PARTY_ORIGIN_IS_OPAQUE = "RelyingPartyOriginIsOpaque"
    TYPE_NOT_MATCHING = "TypeNotMatching"
    UI_DISMISSED_NO_EMBARGO = "UiDismissedNoEmbargo"
    CORS_ERROR = "CorsError"
    SUPPRESSED_BY_SEGMENTATION_PLATFORM = "SuppressedBySegmentationPlatform"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class FederatedAuthUserInfoRequestIssueDetails:
    federated_auth_user_info_request_issue_reason: FederatedAuthUserInfoRequestIssueReason

    def to_json(self):
        json = dict()
        json['federatedAuthUserInfoRequestIssueReason'] = self.federated_auth_user_info_request_issue_reason.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            federated_auth_user_info_request_issue_reason=FederatedAuthUserInfoRequestIssueReason.from_json(json['federatedAuthUserInfoRequestIssueReason']),
        )


class FederatedAuthUserInfoRequestIssueReason(enum.Enum):
    '''
    Represents the failure reason when a getUserInfo() call fails.
    Should be updated alongside FederatedAuthUserInfoRequestResult in
    third_party/blink/public/mojom/devtools/inspector_issue.mojom.
    '''
    NOT_SAME_ORIGIN = "NotSameOrigin"
    NOT_IFRAME = "NotIframe"
    NOT_POTENTIALLY_TRUSTWORTHY = "NotPotentiallyTrustworthy"
    NO_API_PERMISSION = "NoApiPermission"
    NOT_SIGNED_IN_WITH_IDP = "NotSignedInWithIdp"
    NO_ACCOUNT_SHARING_PERMISSION = "NoAccountSharingPermission"
    INVALID_CONFIG_OR_WELL_KNOWN = "InvalidConfigOrWellKnown"
    INVALID_ACCOUNTS_RESPONSE = "InvalidAccountsResponse"
    NO_RETURNING_USER_FROM_FETCHED_ACCOUNTS = "NoReturningUserFromFetchedAccounts"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class ClientHintIssueDetails:
    '''
    This issue tracks client hints related issues. It's used to deprecate old
    features, encourage the use of new ones, and provide general guidance.
    '''
    source_code_location: SourceCodeLocation

    client_hint_issue_reason: ClientHintIssueReason

    def to_json(self):
        json = dict()
        json['sourceCodeLocation'] = self.source_code_location.to_json()
        json['clientHintIssueReason'] = self.client_hint_issue_reason.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            source_code_location=SourceCodeLocation.from_json(json['sourceCodeLocation']),
            client_hint_issue_reason=ClientHintIssueReason.from_json(json['clientHintIssueReason']),
        )


@dataclass
class FailedRequestInfo:
    #: The URL that failed to load.
    url: str

    #: The failure message for the failed request.
    failure_message: str

    request_id: typing.Optional[network.RequestId] = None

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['failureMessage'] = self.failure_message
        if self.request_id is not None:
            json['requestId'] = self.request_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url=str(json['url']),
            failure_message=str(json['failureMessage']),
            request_id=network.RequestId.from_json(json['requestId']) if 'requestId' in json else None,
        )


class PartitioningBlobURLInfo(enum.Enum):
    BLOCKED_CROSS_PARTITION_FETCHING = "BlockedCrossPartitionFetching"
    ENFORCE_NOOPENER_FOR_NAVIGATION = "EnforceNoopenerForNavigation"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class PartitioningBlobURLIssueDetails:
    #: The BlobURL that failed to load.
    url: str

    #: Additional information about the Partitioning Blob URL issue.
    partitioning_blob_url_info: PartitioningBlobURLInfo

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['partitioningBlobURLInfo'] = self.partitioning_blob_url_info.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url=str(json['url']),
            partitioning_blob_url_info=PartitioningBlobURLInfo.from_json(json['partitioningBlobURLInfo']),
        )


class SelectElementAccessibilityIssueReason(enum.Enum):
    DISALLOWED_SELECT_CHILD = "DisallowedSelectChild"
    DISALLOWED_OPT_GROUP_CHILD = "DisallowedOptGroupChild"
    NON_PHRASING_CONTENT_OPTION_CHILD = "NonPhrasingContentOptionChild"
    INTERACTIVE_CONTENT_OPTION_CHILD = "InteractiveContentOptionChild"
    INTERACTIVE_CONTENT_LEGEND_CHILD = "InteractiveContentLegendChild"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class SelectElementAccessibilityIssueDetails:
    '''
    This issue warns about errors in the select element content model.
    '''
    node_id: dom.BackendNodeId

    select_element_accessibility_issue_reason: SelectElementAccessibilityIssueReason

    has_disallowed_attributes: bool

    def to_json(self):
        json = dict()
        json['nodeId'] = self.node_id.to_json()
        json['selectElementAccessibilityIssueReason'] = self.select_element_accessibility_issue_reason.to_json()
        json['hasDisallowedAttributes'] = self.has_disallowed_attributes
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            node_id=dom.BackendNodeId.from_json(json['nodeId']),
            select_element_accessibility_issue_reason=SelectElementAccessibilityIssueReason.from_json(json['selectElementAccessibilityIssueReason']),
            has_disallowed_attributes=bool(json['hasDisallowedAttributes']),
        )


class StyleSheetLoadingIssueReason(enum.Enum):
    LATE_IMPORT_RULE = "LateImportRule"
    REQUEST_FAILED = "RequestFailed"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class StylesheetLoadingIssueDetails:
    '''
    This issue warns when a referenced stylesheet couldn't be loaded.
    '''
    #: Source code position that referenced the failing stylesheet.
    source_code_location: SourceCodeLocation

    #: Reason why the stylesheet couldn't be loaded.
    style_sheet_loading_issue_reason: StyleSheetLoadingIssueReason

    #: Contains additional info when the failure was due to a request.
    failed_request_info: typing.Optional[FailedRequestInfo] = None

    def to_json(self):
        json = dict()
        json['sourceCodeLocation'] = self.source_code_location.to_json()
        json['styleSheetLoadingIssueReason'] = self.style_sheet_loading_issue_reason.to_json()
        if self.failed_request_info is not None:
            json['failedRequestInfo'] = self.failed_request_info.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            source_code_location=SourceCodeLocation.from_json(json['sourceCodeLocation']),
            style_sheet_loading_issue_reason=StyleSheetLoadingIssueReason.from_json(json['styleSheetLoadingIssueReason']),
            failed_request_info=FailedRequestInfo.from_json(json['failedRequestInfo']) if 'failedRequestInfo' in json else None,
        )


class PropertyRuleIssueReason(enum.Enum):
    INVALID_SYNTAX = "InvalidSyntax"
    INVALID_INITIAL_VALUE = "InvalidInitialValue"
    INVALID_INHERITS = "InvalidInherits"
    INVALID_NAME = "InvalidName"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class PropertyRuleIssueDetails:
    '''
    This issue warns about errors in property rules that lead to property
    registrations being ignored.
    '''
    #: Source code position of the property rule.
    source_code_location: SourceCodeLocation

    #: Reason why the property rule was discarded.
    property_rule_issue_reason: PropertyRuleIssueReason

    #: The value of the property rule property that failed to parse
    property_value: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['sourceCodeLocation'] = self.source_code_location.to_json()
        json['propertyRuleIssueReason'] = self.property_rule_issue_reason.to_json()
        if self.property_value is not None:
            json['propertyValue'] = self.property_value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            source_code_location=SourceCodeLocation.from_json(json['sourceCodeLocation']),
            property_rule_issue_reason=PropertyRuleIssueReason.from_json(json['propertyRuleIssueReason']),
            property_value=str(json['propertyValue']) if 'propertyValue' in json else None,
        )


class InspectorIssueCode(enum.Enum):
    '''
    A unique identifier for the type of issue. Each type may use one of the
    optional fields in InspectorIssueDetails to convey more specific
    information about the kind of issue.
    '''
    COOKIE_ISSUE = "CookieIssue"
    MIXED_CONTENT_ISSUE = "MixedContentIssue"
    BLOCKED_BY_RESPONSE_ISSUE = "BlockedByResponseIssue"
    HEAVY_AD_ISSUE = "HeavyAdIssue"
    CONTENT_SECURITY_POLICY_ISSUE = "ContentSecurityPolicyIssue"
    SHARED_ARRAY_BUFFER_ISSUE = "SharedArrayBufferIssue"
    LOW_TEXT_CONTRAST_ISSUE = "LowTextContrastIssue"
    CORS_ISSUE = "CorsIssue"
    ATTRIBUTION_REPORTING_ISSUE = "AttributionReportingIssue"
    QUIRKS_MODE_ISSUE = "QuirksModeIssue"
    PARTITIONING_BLOB_URL_ISSUE = "PartitioningBlobURLIssue"
    NAVIGATOR_USER_AGENT_ISSUE = "NavigatorUserAgentIssue"
    GENERIC_ISSUE = "GenericIssue"
    DEPRECATION_ISSUE = "DeprecationIssue"
    CLIENT_HINT_ISSUE = "ClientHintIssue"
    FEDERATED_AUTH_REQUEST_ISSUE = "FederatedAuthRequestIssue"
    BOUNCE_TRACKING_ISSUE = "BounceTrackingIssue"
    COOKIE_DEPRECATION_METADATA_ISSUE = "CookieDeprecationMetadataIssue"
    STYLESHEET_LOADING_ISSUE = "StylesheetLoadingIssue"
    FEDERATED_AUTH_USER_INFO_REQUEST_ISSUE = "FederatedAuthUserInfoRequestIssue"
    PROPERTY_RULE_ISSUE = "PropertyRuleIssue"
    SHARED_DICTIONARY_ISSUE = "SharedDictionaryIssue"
    SELECT_ELEMENT_ACCESSIBILITY_ISSUE = "SelectElementAccessibilityIssue"
    SRI_MESSAGE_SIGNATURE_ISSUE = "SRIMessageSignatureIssue"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class InspectorIssueDetails:
    '''
    This struct holds a list of optional fields with additional information
    specific to the kind of issue. When adding a new issue code, please also
    add a new optional field to this type.
    '''
    cookie_issue_details: typing.Optional[CookieIssueDetails] = None

    mixed_content_issue_details: typing.Optional[MixedContentIssueDetails] = None

    blocked_by_response_issue_details: typing.Optional[BlockedByResponseIssueDetails] = None

    heavy_ad_issue_details: typing.Optional[HeavyAdIssueDetails] = None

    content_security_policy_issue_details: typing.Optional[ContentSecurityPolicyIssueDetails] = None

    shared_array_buffer_issue_details: typing.Optional[SharedArrayBufferIssueDetails] = None

    low_text_contrast_issue_details: typing.Optional[LowTextContrastIssueDetails] = None

    cors_issue_details: typing.Optional[CorsIssueDetails] = None

    attribution_reporting_issue_details: typing.Optional[AttributionReportingIssueDetails] = None

    quirks_mode_issue_details: typing.Optional[QuirksModeIssueDetails] = None

    partitioning_blob_url_issue_details: typing.Optional[PartitioningBlobURLIssueDetails] = None

    navigator_user_agent_issue_details: typing.Optional[NavigatorUserAgentIssueDetails] = None

    generic_issue_details: typing.Optional[GenericIssueDetails] = None

    deprecation_issue_details: typing.Optional[DeprecationIssueDetails] = None

    client_hint_issue_details: typing.Optional[ClientHintIssueDetails] = None

    federated_auth_request_issue_details: typing.Optional[FederatedAuthRequestIssueDetails] = None

    bounce_tracking_issue_details: typing.Optional[BounceTrackingIssueDetails] = None

    cookie_deprecation_metadata_issue_details: typing.Optional[CookieDeprecationMetadataIssueDetails] = None

    stylesheet_loading_issue_details: typing.Optional[StylesheetLoadingIssueDetails] = None

    property_rule_issue_details: typing.Optional[PropertyRuleIssueDetails] = None

    federated_auth_user_info_request_issue_details: typing.Optional[FederatedAuthUserInfoRequestIssueDetails] = None

    shared_dictionary_issue_details: typing.Optional[SharedDictionaryIssueDetails] = None

    select_element_accessibility_issue_details: typing.Optional[SelectElementAccessibilityIssueDetails] = None

    sri_message_signature_issue_details: typing.Optional[SRIMessageSignatureIssueDetails] = None

    def to_json(self):
        json = dict()
        if self.cookie_issue_details is not None:
            json['cookieIssueDetails'] = self.cookie_issue_details.to_json()
        if self.mixed_content_issue_details is not None:
            json['mixedContentIssueDetails'] = self.mixed_content_issue_details.to_json()
        if self.blocked_by_response_issue_details is not None:
            json['blockedByResponseIssueDetails'] = self.blocked_by_response_issue_details.to_json()
        if self.heavy_ad_issue_details is not None:
            json['heavyAdIssueDetails'] = self.heavy_ad_issue_details.to_json()
        if self.content_security_policy_issue_details is not None:
            json['contentSecurityPolicyIssueDetails'] = self.content_security_policy_issue_details.to_json()
        if self.shared_array_buffer_issue_details is not None:
            json['sharedArrayBufferIssueDetails'] = self.shared_array_buffer_issue_details.to_json()
        if self.low_text_contrast_issue_details is not None:
            json['lowTextContrastIssueDetails'] = self.low_text_contrast_issue_details.to_json()
        if self.cors_issue_details is not None:
            json['corsIssueDetails'] = self.cors_issue_details.to_json()
        if self.attribution_reporting_issue_details is not None:
            json['attributionReportingIssueDetails'] = self.attribution_reporting_issue_details.to_json()
        if self.quirks_mode_issue_details is not None:
            json['quirksModeIssueDetails'] = self.quirks_mode_issue_details.to_json()
        if self.partitioning_blob_url_issue_details is not None:
            json['partitioningBlobURLIssueDetails'] = self.partitioning_blob_url_issue_details.to_json()
        if self.navigator_user_agent_issue_details is not None:
            json['navigatorUserAgentIssueDetails'] = self.navigator_user_agent_issue_details.to_json()
        if self.generic_issue_details is not None:
            json['genericIssueDetails'] = self.generic_issue_details.to_json()
        if self.deprecation_issue_details is not None:
            json['deprecationIssueDetails'] = self.deprecation_issue_details.to_json()
        if self.client_hint_issue_details is not None:
            json['clientHintIssueDetails'] = self.client_hint_issue_details.to_json()
        if self.federated_auth_request_issue_details is not None:
            json['federatedAuthRequestIssueDetails'] = self.federated_auth_request_issue_details.to_json()
        if self.bounce_tracking_issue_details is not None:
            json['bounceTrackingIssueDetails'] = self.bounce_tracking_issue_details.to_json()
        if self.cookie_deprecation_metadata_issue_details is not None:
            json['cookieDeprecationMetadataIssueDetails'] = self.cookie_deprecation_metadata_issue_details.to_json()
        if self.stylesheet_loading_issue_details is not None:
            json['stylesheetLoadingIssueDetails'] = self.stylesheet_loading_issue_details.to_json()
        if self.property_rule_issue_details is not None:
            json['propertyRuleIssueDetails'] = self.property_rule_issue_details.to_json()
        if self.federated_auth_user_info_request_issue_details is not None:
            json['federatedAuthUserInfoRequestIssueDetails'] = self.federated_auth_user_info_request_issue_details.to_json()
        if self.shared_dictionary_issue_details is not None:
            json['sharedDictionaryIssueDetails'] = self.shared_dictionary_issue_details.to_json()
        if self.select_element_accessibility_issue_details is not None:
            json['selectElementAccessibilityIssueDetails'] = self.select_element_accessibility_issue_details.to_json()
        if self.sri_message_signature_issue_details is not None:
            json['sriMessageSignatureIssueDetails'] = self.sri_message_signature_issue_details.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            cookie_issue_details=CookieIssueDetails.from_json(json['cookieIssueDetails']) if 'cookieIssueDetails' in json else None,
            mixed_content_issue_details=MixedContentIssueDetails.from_json(json['mixedContentIssueDetails']) if 'mixedContentIssueDetails' in json else None,
            blocked_by_response_issue_details=BlockedByResponseIssueDetails.from_json(json['blockedByResponseIssueDetails']) if 'blockedByResponseIssueDetails' in json else None,
            heavy_ad_issue_details=HeavyAdIssueDetails.from_json(json['heavyAdIssueDetails']) if 'heavyAdIssueDetails' in json else None,
            content_security_policy_issue_details=ContentSecurityPolicyIssueDetails.from_json(json['contentSecurityPolicyIssueDetails']) if 'contentSecurityPolicyIssueDetails' in json else None,
            shared_array_buffer_issue_details=SharedArrayBufferIssueDetails.from_json(json['sharedArrayBufferIssueDetails']) if 'sharedArrayBufferIssueDetails' in json else None,
            low_text_contrast_issue_details=LowTextContrastIssueDetails.from_json(json['lowTextContrastIssueDetails']) if 'lowTextContrastIssueDetails' in json else None,
            cors_issue_details=CorsIssueDetails.from_json(json['corsIssueDetails']) if 'corsIssueDetails' in json else None,
            attribution_reporting_issue_details=AttributionReportingIssueDetails.from_json(json['attributionReportingIssueDetails']) if 'attributionReportingIssueDetails' in json else None,
            quirks_mode_issue_details=QuirksModeIssueDetails.from_json(json['quirksModeIssueDetails']) if 'quirksModeIssueDetails' in json else None,
            partitioning_blob_url_issue_details=PartitioningBlobURLIssueDetails.from_json(json['partitioningBlobURLIssueDetails']) if 'partitioningBlobURLIssueDetails' in json else None,
            navigator_user_agent_issue_details=NavigatorUserAgentIssueDetails.from_json(json['navigatorUserAgentIssueDetails']) if 'navigatorUserAgentIssueDetails' in json else None,
            generic_issue_details=GenericIssueDetails.from_json(json['genericIssueDetails']) if 'genericIssueDetails' in json else None,
            deprecation_issue_details=DeprecationIssueDetails.from_json(json['deprecationIssueDetails']) if 'deprecationIssueDetails' in json else None,
            client_hint_issue_details=ClientHintIssueDetails.from_json(json['clientHintIssueDetails']) if 'clientHintIssueDetails' in json else None,
            federated_auth_request_issue_details=FederatedAuthRequestIssueDetails.from_json(json['federatedAuthRequestIssueDetails']) if 'federatedAuthRequestIssueDetails' in json else None,
            bounce_tracking_issue_details=BounceTrackingIssueDetails.from_json(json['bounceTrackingIssueDetails']) if 'bounceTrackingIssueDetails' in json else None,
            cookie_deprecation_metadata_issue_details=CookieDeprecationMetadataIssueDetails.from_json(json['cookieDeprecationMetadataIssueDetails']) if 'cookieDeprecationMetadataIssueDetails' in json else None,
            stylesheet_loading_issue_details=StylesheetLoadingIssueDetails.from_json(json['stylesheetLoadingIssueDetails']) if 'stylesheetLoadingIssueDetails' in json else None,
            property_rule_issue_details=PropertyRuleIssueDetails.from_json(json['propertyRuleIssueDetails']) if 'propertyRuleIssueDetails' in json else None,
            federated_auth_user_info_request_issue_details=FederatedAuthUserInfoRequestIssueDetails.from_json(json['federatedAuthUserInfoRequestIssueDetails']) if 'federatedAuthUserInfoRequestIssueDetails' in json else None,
            shared_dictionary_issue_details=SharedDictionaryIssueDetails.from_json(json['sharedDictionaryIssueDetails']) if 'sharedDictionaryIssueDetails' in json else None,
            select_element_accessibility_issue_details=SelectElementAccessibilityIssueDetails.from_json(json['selectElementAccessibilityIssueDetails']) if 'selectElementAccessibilityIssueDetails' in json else None,
            sri_message_signature_issue_details=SRIMessageSignatureIssueDetails.from_json(json['sriMessageSignatureIssueDetails']) if 'sriMessageSignatureIssueDetails' in json else None,
        )


class IssueId(str):
    '''
    A unique id for a DevTools inspector issue. Allows other entities (e.g.
    exceptions, CDP message, console messages, etc.) to reference an issue.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> IssueId:
        return cls(json)

    def __repr__(self):
        return 'IssueId({})'.format(super().__repr__())


@dataclass
class InspectorIssue:
    '''
    An inspector issue reported from the back-end.
    '''
    code: InspectorIssueCode

    details: InspectorIssueDetails

    #: A unique id for this issue. May be omitted if no other entity (e.g.
    #: exception, CDP message, etc.) is referencing this issue.
    issue_id: typing.Optional[IssueId] = None

    def to_json(self):
        json = dict()
        json['code'] = self.code.to_json()
        json['details'] = self.details.to_json()
        if self.issue_id is not None:
            json['issueId'] = self.issue_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            code=InspectorIssueCode.from_json(json['code']),
            details=InspectorIssueDetails.from_json(json['details']),
            issue_id=IssueId.from_json(json['issueId']) if 'issueId' in json else None,
        )


def get_encoded_response(
        request_id: network.RequestId,
        encoding: str,
        quality: typing.Optional[float] = None,
        size_only: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.Optional[str], int, int]]:
    '''
    Returns the response body and size if it were re-encoded with the specified settings. Only
    applies to images.

    :param request_id: Identifier of the network request to get content for.
    :param encoding: The encoding to use.
    :param quality: *(Optional)* The quality of the encoding (0-1). (defaults to 1)
    :param size_only: *(Optional)* Whether to only return the size information (defaults to false).
    :returns: A tuple with the following items:

        0. **body** - *(Optional)* The encoded body as a base64 string. Omitted if sizeOnly is true.
        1. **originalSize** - Size before re-encoding.
        2. **encodedSize** - Size after re-encoding.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    params['encoding'] = encoding
    if quality is not None:
        params['quality'] = quality
    if size_only is not None:
        params['sizeOnly'] = size_only
    cmd_dict: T_JSON_DICT = {
        'method': 'Audits.getEncodedResponse',
        'params': params,
    }
    json = yield cmd_dict
    return (
        str(json['body']) if 'body' in json else None,
        int(json['originalSize']),
        int(json['encodedSize'])
    )


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables issues domain, prevents further issues from being reported to the client.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Audits.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables issues domain, sends the issues collected so far to the client by means of the
    ``issueAdded`` event.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Audits.enable',
    }
    json = yield cmd_dict


def check_contrast(
        report_aaa: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Runs the contrast check for the target page. Found issues are reported
    using Audits.issueAdded event.

    :param report_aaa: *(Optional)* Whether to report WCAG AAA level issues. Default is false.
    '''
    params: T_JSON_DICT = dict()
    if report_aaa is not None:
        params['reportAAA'] = report_aaa
    cmd_dict: T_JSON_DICT = {
        'method': 'Audits.checkContrast',
        'params': params,
    }
    json = yield cmd_dict


def check_forms_issues() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[GenericIssueDetails]]:
    '''
    Runs the form issues check for the target page. Found issues are reported
    using Audits.issueAdded event.

    :returns: 
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Audits.checkFormsIssues',
    }
    json = yield cmd_dict
    return [GenericIssueDetails.from_json(i) for i in json['formIssues']]


@event_class('Audits.issueAdded')
@dataclass
class IssueAdded:
    issue: InspectorIssue

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> IssueAdded:
        return cls(
            issue=InspectorIssue.from_json(json['issue'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\polynomial\hermite.py ===
"""
==============================================================
Hermite Series, "Physicists" (:mod:`numpy.polynomial.hermite`)
==============================================================

This module provides a number of objects (mostly functions) useful for
dealing with Hermite series, including a `Hermite` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

Classes
-------
.. autosummary::
   :toctree: generated/

   Hermite

Constants
---------
.. autosummary::
   :toctree: generated/

   hermdomain
   hermzero
   hermone
   hermx

Arithmetic
----------
.. autosummary::
   :toctree: generated/

   hermadd
   hermsub
   hermmulx
   hermmul
   hermdiv
   hermpow
   hermval
   hermval2d
   hermval3d
   hermgrid2d
   hermgrid3d

Calculus
--------
.. autosummary::
   :toctree: generated/

   hermder
   hermint

Misc Functions
--------------
.. autosummary::
   :toctree: generated/

   hermfromroots
   hermroots
   hermvander
   hermvander2d
   hermvander3d
   hermgauss
   hermweight
   hermcompanion
   hermfit
   hermtrim
   hermline
   herm2poly
   poly2herm

See also
--------
`numpy.polynomial`

"""
import numpy as np
import numpy.linalg as la
from numpy.lib.array_utils import normalize_axis_index

from . import polyutils as pu
from ._polybase import ABCPolyBase

__all__ = [
    'hermzero', 'hermone', 'hermx', 'hermdomain', 'hermline', 'hermadd',
    'hermsub', 'hermmulx', 'hermmul', 'hermdiv', 'hermpow', 'hermval',
    'hermder', 'hermint', 'herm2poly', 'poly2herm', 'hermfromroots',
    'hermvander', 'hermfit', 'hermtrim', 'hermroots', 'Hermite',
    'hermval2d', 'hermval3d', 'hermgrid2d', 'hermgrid3d', 'hermvander2d',
    'hermvander3d', 'hermcompanion', 'hermgauss', 'hermweight']

hermtrim = pu.trimcoef


def poly2herm(pol):
    """
    poly2herm(pol)

    Convert a polynomial to a Hermite series.

    Convert an array representing the coefficients of a polynomial (relative
    to the "standard" basis) ordered from lowest degree to highest, to an
    array of the coefficients of the equivalent Hermite series, ordered
    from lowest to highest degree.

    Parameters
    ----------
    pol : array_like
        1-D array containing the polynomial coefficients

    Returns
    -------
    c : ndarray
        1-D array containing the coefficients of the equivalent Hermite
        series.

    See Also
    --------
    herm2poly

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy.polynomial.hermite import poly2herm
    >>> poly2herm(np.arange(4))
    array([1.   ,  2.75 ,  0.5  ,  0.375])

    """
    [pol] = pu.as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = hermadd(hermmulx(res), pol[i])
    return res


def herm2poly(c):
    """
    Convert a Hermite series to a polynomial.

    Convert an array representing the coefficients of a Hermite series,
    ordered from lowest degree to highest, to an array of the coefficients
    of the equivalent polynomial (relative to the "standard" basis) ordered
    from lowest to highest degree.

    Parameters
    ----------
    c : array_like
        1-D array containing the Hermite series coefficients, ordered
        from lowest order term to highest.

    Returns
    -------
    pol : ndarray
        1-D array containing the coefficients of the equivalent polynomial
        (relative to the "standard" basis) ordered from lowest order term
        to highest.

    See Also
    --------
    poly2herm

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy.polynomial.hermite import herm2poly
    >>> herm2poly([ 1.   ,  2.75 ,  0.5  ,  0.375])
    array([0., 1., 2., 3.])

    """
    from .polynomial import polyadd, polymulx, polysub

    [c] = pu.as_series([c])
    n = len(c)
    if n == 1:
        return c
    if n == 2:
        c[1] *= 2
        return c
    else:
        c0 = c[-2]
        c1 = c[-1]
        # i is the current degree of c1
        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(c[i - 2], c1 * (2 * (i - 1)))
            c1 = polyadd(tmp, polymulx(c1) * 2)
        return polyadd(c0, polymulx(c1) * 2)


#
# These are constant arrays are of integer type so as to be compatible
# with the widest range of other types, such as Decimal.
#

# Hermite
hermdomain = np.array([-1., 1.])

# Hermite coefficients representing zero.
hermzero = np.array([0])

# Hermite coefficients representing one.
hermone = np.array([1])

# Hermite coefficients representing the identity x.
hermx = np.array([0, 1 / 2])


def hermline(off, scl):
    """
    Hermite series whose graph is a straight line.



    Parameters
    ----------
    off, scl : scalars
        The specified line is given by ``off + scl*x``.

    Returns
    -------
    y : ndarray
        This module's representation of the Hermite series for
        ``off + scl*x``.

    See Also
    --------
    numpy.polynomial.polynomial.polyline
    numpy.polynomial.chebyshev.chebline
    numpy.polynomial.legendre.legline
    numpy.polynomial.laguerre.lagline
    numpy.polynomial.hermite_e.hermeline

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermline, hermval
    >>> hermval(0,hermline(3, 2))
    3.0
    >>> hermval(1,hermline(3, 2))
    5.0

    """
    if scl != 0:
        return np.array([off, scl / 2])
    else:
        return np.array([off])


def hermfromroots(roots):
    """
    Generate a Hermite series with given roots.

    The function returns the coefficients of the polynomial

    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),

    in Hermite form, where the :math:`r_n` are the roots specified in `roots`.
    If a zero has multiplicity n, then it must appear in `roots` n times.
    For instance, if 2 is a root of multiplicity three and 3 is a root of
    multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3]. The
    roots can appear in any order.

    If the returned coefficients are `c`, then

    .. math:: p(x) = c_0 + c_1 * H_1(x) + ... +  c_n * H_n(x)

    The coefficient of the last term is not generally 1 for monic
    polynomials in Hermite form.

    Parameters
    ----------
    roots : array_like
        Sequence containing the roots.

    Returns
    -------
    out : ndarray
        1-D array of coefficients.  If all roots are real then `out` is a
        real array, if some of the roots are complex, then `out` is complex
        even if all the coefficients in the result are real (see Examples
        below).

    See Also
    --------
    numpy.polynomial.polynomial.polyfromroots
    numpy.polynomial.legendre.legfromroots
    numpy.polynomial.laguerre.lagfromroots
    numpy.polynomial.chebyshev.chebfromroots
    numpy.polynomial.hermite_e.hermefromroots

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermfromroots, hermval
    >>> coef = hermfromroots((-1, 0, 1))
    >>> hermval((-1, 0, 1), coef)
    array([0.,  0.,  0.])
    >>> coef = hermfromroots((-1j, 1j))
    >>> hermval((-1j, 1j), coef)
    array([0.+0.j, 0.+0.j])

    """
    return pu._fromroots(hermline, hermmul, roots)


def hermadd(c1, c2):
    """
    Add one Hermite series to another.

    Returns the sum of two Hermite series `c1` + `c2`.  The arguments
    are sequences of coefficients ordered from lowest order term to
    highest, i.e., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the Hermite series of their sum.

    See Also
    --------
    hermsub, hermmulx, hermmul, hermdiv, hermpow

    Notes
    -----
    Unlike multiplication, division, etc., the sum of two Hermite series
    is a Hermite series (without having to "reproject" the result onto
    the basis set) so addition, just like that of "standard" polynomials,
    is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermadd
    >>> hermadd([1, 2, 3], [1, 2, 3, 4])
    array([2., 4., 6., 4.])

    """
    return pu._add(c1, c2)


def hermsub(c1, c2):
    """
    Subtract one Hermite series from another.

    Returns the difference of two Hermite series `c1` - `c2`.  The
    sequences of coefficients are from lowest order term to highest, i.e.,
    [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Hermite series coefficients representing their difference.

    See Also
    --------
    hermadd, hermmulx, hermmul, hermdiv, hermpow

    Notes
    -----
    Unlike multiplication, division, etc., the difference of two Hermite
    series is a Hermite series (without having to "reproject" the result
    onto the basis set) so subtraction, just like that of "standard"
    polynomials, is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermsub
    >>> hermsub([1, 2, 3, 4], [1, 2, 3])
    array([0.,  0.,  0.,  4.])

    """
    return pu._sub(c1, c2)


def hermmulx(c):
    """Multiply a Hermite series by x.

    Multiply the Hermite series `c` by x, where x is the independent
    variable.


    Parameters
    ----------
    c : array_like
        1-D array of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the result of the multiplication.

    See Also
    --------
    hermadd, hermsub, hermmul, hermdiv, hermpow

    Notes
    -----
    The multiplication uses the recursion relationship for Hermite
    polynomials in the form

    .. math::

        xP_i(x) = (P_{i + 1}(x)/2 + i*P_{i - 1}(x))

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermmulx
    >>> hermmulx([1, 2, 3])
    array([2. , 6.5, 1. , 1.5])

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    # The zero series needs special treatment
    if len(c) == 1 and c[0] == 0:
        return c

    prd = np.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0] * 0
    prd[1] = c[0] / 2
    for i in range(1, len(c)):
        prd[i + 1] = c[i] / 2
        prd[i - 1] += c[i] * i
    return prd


def hermmul(c1, c2):
    """
    Multiply one Hermite series by another.

    Returns the product of two Hermite series `c1` * `c2`.  The arguments
    are sequences of coefficients, from lowest order "term" to highest,
    e.g., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Hermite series coefficients representing their product.

    See Also
    --------
    hermadd, hermsub, hermmulx, hermdiv, hermpow

    Notes
    -----
    In general, the (polynomial) product of two C-series results in terms
    that are not in the Hermite polynomial basis set.  Thus, to express
    the product as a Hermite series, it is necessary to "reproject" the
    product onto said basis set, which may produce "unintuitive" (but
    correct) results; see Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermmul
    >>> hermmul([1, 2, 3], [0, 1, 2])
    array([52.,  29.,  52.,   7.,   6.])

    """
    # s1, s2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])

    if len(c1) > len(c2):
        c = c2
        xs = c1
    else:
        c = c1
        xs = c2

    if len(c) == 1:
        c0 = c[0] * xs
        c1 = 0
    elif len(c) == 2:
        c0 = c[0] * xs
        c1 = c[1] * xs
    else:
        nd = len(c)
        c0 = c[-2] * xs
        c1 = c[-1] * xs
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = hermsub(c[-i] * xs, c1 * (2 * (nd - 1)))
            c1 = hermadd(tmp, hermmulx(c1) * 2)
    return hermadd(c0, hermmulx(c1) * 2)


def hermdiv(c1, c2):
    """
    Divide one Hermite series by another.

    Returns the quotient-with-remainder of two Hermite series
    `c1` / `c2`.  The arguments are sequences of coefficients from lowest
    order "term" to highest, e.g., [1,2,3] represents the series
    ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    [quo, rem] : ndarrays
        Of Hermite series coefficients representing the quotient and
        remainder.

    See Also
    --------
    hermadd, hermsub, hermmulx, hermmul, hermpow

    Notes
    -----
    In general, the (polynomial) division of one Hermite series by another
    results in quotient and remainder terms that are not in the Hermite
    polynomial basis set.  Thus, to express these results as a Hermite
    series, it is necessary to "reproject" the results onto the Hermite
    basis set, which may produce "unintuitive" (but correct) results; see
    Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermdiv
    >>> hermdiv([ 52.,  29.,  52.,   7.,   6.], [0, 1, 2])
    (array([1., 2., 3.]), array([0.]))
    >>> hermdiv([ 54.,  31.,  52.,   7.,   6.], [0, 1, 2])
    (array([1., 2., 3.]), array([2., 2.]))
    >>> hermdiv([ 53.,  30.,  52.,   7.,   6.], [0, 1, 2])
    (array([1., 2., 3.]), array([1., 1.]))

    """
    return pu._div(hermmul, c1, c2)


def hermpow(c, pow, maxpower=16):
    """Raise a Hermite series to a power.

    Returns the Hermite series `c` raised to the power `pow`. The
    argument `c` is a sequence of coefficients ordered from low to high.
    i.e., [1,2,3] is the series  ``P_0 + 2*P_1 + 3*P_2.``

    Parameters
    ----------
    c : array_like
        1-D array of Hermite series coefficients ordered from low to
        high.
    pow : integer
        Power to which the series will be raised
    maxpower : integer, optional
        Maximum power allowed. This is mainly to limit growth of the series
        to unmanageable size. Default is 16

    Returns
    -------
    coef : ndarray
        Hermite series of power.

    See Also
    --------
    hermadd, hermsub, hermmulx, hermmul, hermdiv

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermpow
    >>> hermpow([1, 2, 3], 2)
    array([81.,  52.,  82.,  12.,   9.])

    """
    return pu._pow(hermmul, c, pow, maxpower)


def hermder(c, m=1, scl=1, axis=0):
    """
    Differentiate a Hermite series.

    Returns the Hermite series coefficients `c` differentiated `m` times
    along `axis`.  At each iteration the result is multiplied by `scl` (the
    scaling factor is for use in a linear change of variable). The argument
    `c` is an array of coefficients from low to high degree along each
    axis, e.g., [1,2,3] represents the series ``1*H_0 + 2*H_1 + 3*H_2``
    while [[1,2],[1,2]] represents ``1*H_0(x)*H_0(y) + 1*H_1(x)*H_0(y) +
    2*H_0(x)*H_1(y) + 2*H_1(x)*H_1(y)`` if axis=0 is ``x`` and axis=1 is
    ``y``.

    Parameters
    ----------
    c : array_like
        Array of Hermite series coefficients. If `c` is multidimensional the
        different axis correspond to different variables with the degree in
        each axis given by the corresponding index.
    m : int, optional
        Number of derivatives taken, must be non-negative. (Default: 1)
    scl : scalar, optional
        Each differentiation is multiplied by `scl`.  The end result is
        multiplication by ``scl**m``.  This is for use in a linear change of
        variable. (Default: 1)
    axis : int, optional
        Axis over which the derivative is taken. (Default: 0).

    Returns
    -------
    der : ndarray
        Hermite series of the derivative.

    See Also
    --------
    hermint

    Notes
    -----
    In general, the result of differentiating a Hermite series does not
    resemble the same operation on a power series. Thus the result of this
    function may be "unintuitive," albeit correct; see Examples section
    below.

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermder
    >>> hermder([ 1. ,  0.5,  0.5,  0.5])
    array([1., 2., 3.])
    >>> hermder([-0.5,  1./2.,  1./8.,  1./12.,  1./16.], m=2)
    array([1., 2., 3.])

    """
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    cnt = pu._as_int(m, "the order of derivation")
    iaxis = pu._as_int(axis, "the axis")
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = np.moveaxis(c, iaxis, 0)
    n = len(c)
    if cnt >= n:
        c = c[:1] * 0
    else:
        for i in range(cnt):
            n = n - 1
            c *= scl
            der = np.empty((n,) + c.shape[1:], dtype=c.dtype)
            for j in range(n, 0, -1):
                der[j - 1] = (2 * j) * c[j]
            c = der
    c = np.moveaxis(c, 0, iaxis)
    return c


def hermint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    """
    Integrate a Hermite series.

    Returns the Hermite series coefficients `c` integrated `m` times from
    `lbnd` along `axis`. At each iteration the resulting series is
    **multiplied** by `scl` and an integration constant, `k`, is added.
    The scaling factor is for use in a linear change of variable.  ("Buyer
    beware": note that, depending on what one is doing, one may want `scl`
    to be the reciprocal of what one might expect; for more information,
    see the Notes section below.)  The argument `c` is an array of
    coefficients from low to high degree along each axis, e.g., [1,2,3]
    represents the series ``H_0 + 2*H_1 + 3*H_2`` while [[1,2],[1,2]]
    represents ``1*H_0(x)*H_0(y) + 1*H_1(x)*H_0(y) + 2*H_0(x)*H_1(y) +
    2*H_1(x)*H_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.

    Parameters
    ----------
    c : array_like
        Array of Hermite series coefficients. If c is multidimensional the
        different axis correspond to different variables with the degree in
        each axis given by the corresponding index.
    m : int, optional
        Order of integration, must be positive. (Default: 1)
    k : {[], list, scalar}, optional
        Integration constant(s).  The value of the first integral at
        ``lbnd`` is the first value in the list, the value of the second
        integral at ``lbnd`` is the second value, etc.  If ``k == []`` (the
        default), all constants are set to zero.  If ``m == 1``, a single
        scalar can be given instead of a list.
    lbnd : scalar, optional
        The lower bound of the integral. (Default: 0)
    scl : scalar, optional
        Following each integration the result is *multiplied* by `scl`
        before the integration constant is added. (Default: 1)
    axis : int, optional
        Axis over which the integral is taken. (Default: 0).

    Returns
    -------
    S : ndarray
        Hermite series coefficients of the integral.

    Raises
    ------
    ValueError
        If ``m < 0``, ``len(k) > m``, ``np.ndim(lbnd) != 0``, or
        ``np.ndim(scl) != 0``.

    See Also
    --------
    hermder

    Notes
    -----
    Note that the result of each integration is *multiplied* by `scl`.
    Why is this important to note?  Say one is making a linear change of
    variable :math:`u = ax + b` in an integral relative to `x`.  Then
    :math:`dx = du/a`, so one will need to set `scl` equal to
    :math:`1/a` - perhaps not what one would have first thought.

    Also note that, in general, the result of integrating a C-series needs
    to be "reprojected" onto the C-series basis set.  Thus, typically,
    the result of this function is "unintuitive," albeit correct; see
    Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermint
    >>> hermint([1,2,3]) # integrate once, value 0 at 0.
    array([1. , 0.5, 0.5, 0.5])
    >>> hermint([1,2,3], m=2) # integrate twice, value & deriv 0 at 0
    array([-0.5       ,  0.5       ,  0.125     ,  0.08333333,  0.0625    ]) # may vary
    >>> hermint([1,2,3], k=1) # integrate once, value 1 at 0.
    array([2. , 0.5, 0.5, 0.5])
    >>> hermint([1,2,3], lbnd=-1) # integrate once, value 0 at -1
    array([-2. ,  0.5,  0.5,  0.5])
    >>> hermint([1,2,3], m=2, k=[1,2], lbnd=-1)
    array([ 1.66666667, -0.5       ,  0.125     ,  0.08333333,  0.0625    ]) # may vary

    """
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    if not np.iterable(k):
        k = [k]
    cnt = pu._as_int(m, "the order of integration")
    iaxis = pu._as_int(axis, "the axis")
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    if np.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if np.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = np.moveaxis(c, iaxis, 0)
    k = list(k) + [0] * (cnt - len(k))
    for i in range(cnt):
        n = len(c)
        c *= scl
        if n == 1 and np.all(c[0] == 0):
            c[0] += k[i]
        else:
            tmp = np.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
            tmp[0] = c[0] * 0
            tmp[1] = c[0] / 2
            for j in range(1, n):
                tmp[j + 1] = c[j] / (2 * (j + 1))
            tmp[0] += k[i] - hermval(lbnd, tmp)
            c = tmp
    c = np.moveaxis(c, 0, iaxis)
    return c


def hermval(x, c, tensor=True):
    """
    Evaluate an Hermite series at points x.

    If `c` is of length ``n + 1``, this function returns the value:

    .. math:: p(x) = c_0 * H_0(x) + c_1 * H_1(x) + ... + c_n * H_n(x)

    The parameter `x` is converted to an array only if it is a tuple or a
    list, otherwise it is treated as a scalar. In either case, either `x`
    or its elements must support multiplication and addition both with
    themselves and with the elements of `c`.

    If `c` is a 1-D array, then ``p(x)`` will have the same shape as `x`.  If
    `c` is multidimensional, then the shape of the result depends on the
    value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +
    x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that
    scalars have shape (,).

    Trailing zeros in the coefficients will be used in the evaluation, so
    they should be avoided if efficiency is a concern.

    Parameters
    ----------
    x : array_like, compatible object
        If `x` is a list or tuple, it is converted to an ndarray, otherwise
        it is left unchanged and treated as a scalar. In either case, `x`
        or its elements must support addition and multiplication with
        themselves and with the elements of `c`.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree n are contained in c[n]. If `c` is multidimensional the
        remaining indices enumerate multiple polynomials. In the two
        dimensional case the coefficients may be thought of as stored in
        the columns of `c`.
    tensor : boolean, optional
        If True, the shape of the coefficient array is extended with ones
        on the right, one for each dimension of `x`. Scalars have dimension 0
        for this action. The result is that every column of coefficients in
        `c` is evaluated for every element of `x`. If False, `x` is broadcast
        over the columns of `c` for the evaluation.  This keyword is useful
        when `c` is multidimensional. The default value is True.

    Returns
    -------
    values : ndarray, algebra_like
        The shape of the return value is described above.

    See Also
    --------
    hermval2d, hermgrid2d, hermval3d, hermgrid3d

    Notes
    -----
    The evaluation uses Clenshaw recursion, aka synthetic division.

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermval
    >>> coef = [1,2,3]
    >>> hermval(1, coef)
    11.0
    >>> hermval([[1,2],[3,4]], coef)
    array([[ 11.,   51.],
           [115.,  203.]])

    """
    c = np.array(c, ndmin=1, copy=None)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    if isinstance(x, np.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    x2 = x * 2
    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - c1 * (2 * (nd - 1))
            c1 = tmp + c1 * x2
    return c0 + c1 * x2


def hermval2d(x, y, c):
    """
    Evaluate a 2-D Hermite series at points (x, y).

    This function returns the values:

    .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * H_i(x) * H_j(y)

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as a scalars and they
    must have the same shape after conversion. In either case, either `x`
    and `y` or their elements must support multiplication and addition both
    with themselves and with the elements of `c`.

    If `c` is a 1-D array a one is implicitly appended to its shape to make
    it 2-D. The shape of the result will be c.shape[2:] + x.shape.

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two dimensional series is evaluated at the points ``(x, y)``,
        where `x` and `y` must have the same shape. If `x` or `y` is a list
        or tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and if it isn't an ndarray it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term
        of multi-degree i,j is contained in ``c[i,j]``. If `c` has
        dimension greater than two the remaining indices enumerate multiple
        sets of coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points formed with
        pairs of corresponding values from `x` and `y`.

    See Also
    --------
    hermval, hermgrid2d, hermval3d, hermgrid3d

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermval2d
    >>> x = [1, 2]
    >>> y = [4, 5]
    >>> c = [[1, 2, 3], [4, 5, 6]]
    >>> hermval2d(x, y, c)
    array([1035., 2883.])

    """
    return pu._valnd(hermval, c, x, y)


def hermgrid2d(x, y, c):
    """
    Evaluate a 2-D Hermite series on the Cartesian product of x and y.

    This function returns the values:

    .. math:: p(a,b) = \\sum_{i,j} c_{i,j} * H_i(a) * H_j(b)

    where the points ``(a, b)`` consist of all pairs formed by taking
    `a` from `x` and `b` from `y`. The resulting points form a grid with
    `x` in the first dimension and `y` in the second.

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as a scalars. In either
    case, either `x` and `y` or their elements must support multiplication
    and addition both with themselves and with the elements of `c`.

    If `c` has fewer than two dimensions, ones are implicitly appended to
    its shape to make it 2-D. The shape of the result will be c.shape[2:] +
    x.shape.

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two dimensional series is evaluated at the points in the
        Cartesian product of `x` and `y`.  If `x` or `y` is a list or
        tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and, if it isn't an ndarray, it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree i,j are contained in ``c[i,j]``. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points in the Cartesian
        product of `x` and `y`.

    See Also
    --------
    hermval, hermval2d, hermval3d, hermgrid3d

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermgrid2d
    >>> x = [1, 2, 3]
    >>> y = [4, 5]
    >>> c = [[1, 2, 3], [4, 5, 6]]
    >>> hermgrid2d(x, y, c)
    array([[1035., 1599.],
           [1867., 2883.],
           [2699., 4167.]])

    """
    return pu._gridnd(hermval, c, x, y)


def hermval3d(x, y, z, c):
    """
    Evaluate a 3-D Hermite series at points (x, y, z).

    This function returns the values:

    .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * H_i(x) * H_j(y) * H_k(z)

    The parameters `x`, `y`, and `z` are converted to arrays only if
    they are tuples or a lists, otherwise they are treated as a scalars and
    they must have the same shape after conversion. In either case, either
    `x`, `y`, and `z` or their elements must support multiplication and
    addition both with themselves and with the elements of `c`.

    If `c` has fewer than 3 dimensions, ones are implicitly appended to its
    shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape.

    Parameters
    ----------
    x, y, z : array_like, compatible object
        The three dimensional series is evaluated at the points
        ``(x, y, z)``, where `x`, `y`, and `z` must have the same shape.  If
        any of `x`, `y`, or `z` is a list or tuple, it is first converted
        to an ndarray, otherwise it is left unchanged and if it isn't an
        ndarray it is  treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term of
        multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension
        greater than 3 the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the multidimensional polynomial on points formed with
        triples of corresponding values from `x`, `y`, and `z`.

    See Also
    --------
    hermval, hermval2d, hermgrid2d, hermgrid3d

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermval3d
    >>> x = [1, 2]
    >>> y = [4, 5]
    >>> z = [6, 7]
    >>> c = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
    >>> hermval3d(x, y, z, c)
    array([ 40077., 120131.])

    """
    return pu._valnd(hermval, c, x, y, z)


def hermgrid3d(x, y, z, c):
    """
    Evaluate a 3-D Hermite series on the Cartesian product of x, y, and z.

    This function returns the values:

    .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * H_i(a) * H_j(b) * H_k(c)

    where the points ``(a, b, c)`` consist of all triples formed by taking
    `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form
    a grid with `x` in the first dimension, `y` in the second, and `z` in
    the third.

    The parameters `x`, `y`, and `z` are converted to arrays only if they
    are tuples or a lists, otherwise they are treated as a scalars. In
    either case, either `x`, `y`, and `z` or their elements must support
    multiplication and addition both with themselves and with the elements
    of `c`.

    If `c` has fewer than three dimensions, ones are implicitly appended to
    its shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape + y.shape + z.shape.

    Parameters
    ----------
    x, y, z : array_like, compatible objects
        The three dimensional series is evaluated at the points in the
        Cartesian product of `x`, `y`, and `z`.  If `x`, `y`, or `z` is a
        list or tuple, it is first converted to an ndarray, otherwise it is
        left unchanged and, if it isn't an ndarray, it is treated as a
        scalar.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree i,j are contained in ``c[i,j]``. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points in the Cartesian
        product of `x` and `y`.

    See Also
    --------
    hermval, hermval2d, hermgrid2d, hermval3d

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermgrid3d
    >>> x = [1, 2]
    >>> y = [4, 5]
    >>> z = [6, 7]
    >>> c = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
    >>> hermgrid3d(x, y, z, c)
    array([[[ 40077.,  54117.],
            [ 49293.,  66561.]],
           [[ 72375.,  97719.],
            [ 88975., 120131.]]])

    """
    return pu._gridnd(hermval, c, x, y, z)


def hermvander(x, deg):
    """Pseudo-Vandermonde matrix of given degree.

    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
    `x`. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., i] = H_i(x),

    where ``0 <= i <= deg``. The leading indices of `V` index the elements of
    `x` and the last index is the degree of the Hermite polynomial.

    If `c` is a 1-D array of coefficients of length ``n + 1`` and `V` is the
    array ``V = hermvander(x, n)``, then ``np.dot(V, c)`` and
    ``hermval(x, c)`` are the same up to roundoff. This equivalence is
    useful both for least squares fitting and for the evaluation of a large
    number of Hermite series of the same degree and sample points.

    Parameters
    ----------
    x : array_like
        Array of points. The dtype is converted to float64 or complex128
        depending on whether any of the elements are complex. If `x` is
        scalar it is converted to a 1-D array.
    deg : int
        Degree of the resulting matrix.

    Returns
    -------
    vander : ndarray
        The pseudo-Vandermonde matrix. The shape of the returned matrix is
        ``x.shape + (deg + 1,)``, where The last index is the degree of the
        corresponding Hermite polynomial.  The dtype will be the same as
        the converted `x`.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial.hermite import hermvander
    >>> x = np.array([-1, 0, 1])
    >>> hermvander(x, 3)
    array([[ 1., -2.,  2.,  4.],
           [ 1.,  0., -2., -0.],
           [ 1.,  2.,  2., -4.]])

    """
    ideg = pu._as_int(deg, "deg")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = np.array(x, copy=None, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = np.empty(dims, dtype=dtyp)
    v[0] = x * 0 + 1
    if ideg > 0:
        x2 = x * 2
        v[1] = x2
        for i in range(2, ideg + 1):
            v[i] = (v[i - 1] * x2 - v[i - 2] * (2 * (i - 1)))
    return np.moveaxis(v, 0, -1)


def hermvander2d(x, y, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y)``. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (deg[1] + 1)*i + j] = H_i(x) * H_j(y),

    where ``0 <= i <= deg[0]`` and ``0 <= j <= deg[1]``. The leading indices of
    `V` index the points ``(x, y)`` and the last index encodes the degrees of
    the Hermite polynomials.

    If ``V = hermvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
    correspond to the elements of a 2-D coefficient array `c` of shape
    (xdeg + 1, ydeg + 1) in the order

    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...

    and ``np.dot(V, c.flat)`` and ``hermval2d(x, y, c)`` will be the same
    up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 2-D Hermite
    series of the same degrees and sample points.

    Parameters
    ----------
    x, y : array_like
        Arrays of point coordinates, all of the same shape. The dtypes
        will be converted to either float64 or complex128 depending on
        whether any of the elements are complex. Scalars are converted to 1-D
        arrays.
    deg : list of ints
        List of maximum degrees of the form [x_deg, y_deg].

    Returns
    -------
    vander2d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg[1]+1)`.  The dtype will be the same
        as the converted `x` and `y`.

    See Also
    --------
    hermvander, hermvander3d, hermval2d, hermval3d

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial.hermite import hermvander2d
    >>> x = np.array([-1, 0, 1])
    >>> y = np.array([-1, 0, 1])
    >>> hermvander2d(x, y, [2, 2])
    array([[ 1., -2.,  2., -2.,  4., -4.,  2., -4.,  4.],
           [ 1.,  0., -2.,  0.,  0., -0., -2., -0.,  4.],
           [ 1.,  2.,  2.,  2.,  4.,  4.,  2.,  4.,  4.]])

    """
    return pu._vander_nd_flat((hermvander, hermvander), (x, y), deg)


def hermvander3d(x, y, z, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y, z)``. If `l`, `m`, `n` are the given degrees in `x`, `y`, `z`,
    then The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = H_i(x)*H_j(y)*H_k(z),

    where ``0 <= i <= l``, ``0 <= j <= m``, and ``0 <= j <= n``.  The leading
    indices of `V` index the points ``(x, y, z)`` and the last index encodes
    the degrees of the Hermite polynomials.

    If ``V = hermvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
    of `V` correspond to the elements of a 3-D coefficient array `c` of
    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order

    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...

    and  ``np.dot(V, c.flat)`` and ``hermval3d(x, y, z, c)`` will be the
    same up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 3-D Hermite
    series of the same degrees and sample points.

    Parameters
    ----------
    x, y, z : array_like
        Arrays of point coordinates, all of the same shape. The dtypes will
        be converted to either float64 or complex128 depending on whether
        any of the elements are complex. Scalars are converted to 1-D
        arrays.
    deg : list of ints
        List of maximum degrees of the form [x_deg, y_deg, z_deg].

    Returns
    -------
    vander3d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg[1]+1)*(deg[2]+1)`.  The dtype will
        be the same as the converted `x`, `y`, and `z`.

    See Also
    --------
    hermvander, hermvander3d, hermval2d, hermval3d

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermvander3d
    >>> x = np.array([-1, 0, 1])
    >>> y = np.array([-1, 0, 1])
    >>> z = np.array([-1, 0, 1])
    >>> hermvander3d(x, y, z, [0, 1, 2])
    array([[ 1., -2.,  2., -2.,  4., -4.],
           [ 1.,  0., -2.,  0.,  0., -0.],
           [ 1.,  2.,  2.,  2.,  4.,  4.]])

    """
    return pu._vander_nd_flat((hermvander, hermvander, hermvander), (x, y, z), deg)


def hermfit(x, y, deg, rcond=None, full=False, w=None):
    """
    Least squares fit of Hermite series to data.

    Return the coefficients of a Hermite series of degree `deg` that is the
    least squares fit to the data values `y` given at points `x`. If `y` is
    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
    fits are done, one for each column of `y`, and the resulting
    coefficients are stored in the corresponding columns of a 2-D return.
    The fitted polynomial(s) are in the form

    .. math::  p(x) = c_0 + c_1 * H_1(x) + ... + c_n * H_n(x),

    where `n` is `deg`.

    Parameters
    ----------
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample
        points sharing the same x-coordinates can be fitted at once by
        passing in a 2D-array that contains one dataset per column.
    deg : int or 1-D array_like
        Degree(s) of the fitting polynomials. If `deg` is a single integer
        all terms up to and including the `deg`'th term are included in the
        fit. For NumPy versions >= 1.11.0 a list of integers specifying the
        degrees of the terms to include may be used instead.
    rcond : float, optional
        Relative condition number of the fit. Singular values smaller than
        this relative to the largest singular value will be ignored. The
        default value is len(x)*eps, where eps is the relative precision of
        the float type, about 2e-16 in most cases.
    full : bool, optional
        Switch determining nature of return value. When it is False (the
        default) just the coefficients are returned, when True diagnostic
        information from the singular value decomposition is also returned.
    w : array_like, shape (`M`,), optional
        Weights. If not None, the weight ``w[i]`` applies to the unsquared
        residual ``y[i] - y_hat[i]`` at ``x[i]``. Ideally the weights are
        chosen so that the errors of the products ``w[i]*y[i]`` all have the
        same variance.  When using inverse-variance weighting, use
        ``w[i] = 1/sigma(y[i])``.  The default value is None.

    Returns
    -------
    coef : ndarray, shape (M,) or (M, K)
        Hermite coefficients ordered from low to high. If `y` was 2-D,
        the coefficients for the data in column k  of `y` are in column
        `k`.

    [residuals, rank, singular_values, rcond] : list
        These values are only returned if ``full == True``

        - residuals -- sum of squared residuals of the least squares fit
        - rank -- the numerical rank of the scaled Vandermonde matrix
        - singular_values -- singular values of the scaled Vandermonde matrix
        - rcond -- value of `rcond`.

        For more details, see `numpy.linalg.lstsq`.

    Warns
    -----
    RankWarning
        The rank of the coefficient matrix in the least-squares fit is
        deficient. The warning is only raised if ``full == False``.  The
        warnings can be turned off by

        >>> import warnings
        >>> warnings.simplefilter('ignore', np.exceptions.RankWarning)

    See Also
    --------
    numpy.polynomial.chebyshev.chebfit
    numpy.polynomial.legendre.legfit
    numpy.polynomial.laguerre.lagfit
    numpy.polynomial.polynomial.polyfit
    numpy.polynomial.hermite_e.hermefit
    hermval : Evaluates a Hermite series.
    hermvander : Vandermonde matrix of Hermite series.
    hermweight : Hermite weight function
    numpy.linalg.lstsq : Computes a least-squares fit from the matrix.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    The solution is the coefficients of the Hermite series `p` that
    minimizes the sum of the weighted squared errors

    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,

    where the :math:`w_j` are the weights. This problem is solved by
    setting up the (typically) overdetermined matrix equation

    .. math:: V(x) * c = w * y,

    where `V` is the weighted pseudo Vandermonde matrix of `x`, `c` are the
    coefficients to be solved for, `w` are the weights, `y` are the
    observed values.  This equation is then solved using the singular value
    decomposition of `V`.

    If some of the singular values of `V` are so small that they are
    neglected, then a `~exceptions.RankWarning` will be issued. This means that
    the coefficient values may be poorly determined. Using a lower order fit
    will usually get rid of the warning.  The `rcond` parameter can also be
    set to a value smaller than its default, but the resulting fit may be
    spurious and have large contributions from roundoff error.

    Fits using Hermite series are probably most useful when the data can be
    approximated by ``sqrt(w(x)) * p(x)``, where ``w(x)`` is the Hermite
    weight. In that case the weight ``sqrt(w(x[i]))`` should be used
    together with data values ``y[i]/sqrt(w(x[i]))``. The weight function is
    available as `hermweight`.

    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           https://en.wikipedia.org/wiki/Curve_fitting

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial.hermite import hermfit, hermval
    >>> x = np.linspace(-10, 10)
    >>> rng = np.random.default_rng()
    >>> err = rng.normal(scale=1./10, size=len(x))
    >>> y = hermval(x, [1, 2, 3]) + err
    >>> hermfit(x, y, 2)
    array([1.02294967, 2.00016403, 2.99994614]) # may vary

    """
    return pu._fit(hermvander, x, y, deg, rcond, full, w)


def hermcompanion(c):
    """Return the scaled companion matrix of c.

    The basis polynomials are scaled so that the companion matrix is
    symmetric when `c` is an Hermite basis polynomial. This provides
    better eigenvalue estimates than the unscaled case and for basis
    polynomials the eigenvalues are guaranteed to be real if
    `numpy.linalg.eigvalsh` is used to obtain them.

    Parameters
    ----------
    c : array_like
        1-D array of Hermite series coefficients ordered from low to high
        degree.

    Returns
    -------
    mat : ndarray
        Scaled companion matrix of dimensions (deg, deg).

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermcompanion
    >>> hermcompanion([1, 0, 1])
    array([[0.        , 0.35355339],
           [0.70710678, 0.        ]])

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    if len(c) < 2:
        raise ValueError('Series must have maximum degree of at least 1.')
    if len(c) == 2:
        return np.array([[-.5 * c[0] / c[1]]])

    n = len(c) - 1
    mat = np.zeros((n, n), dtype=c.dtype)
    scl = np.hstack((1., 1. / np.sqrt(2. * np.arange(n - 1, 0, -1))))
    scl = np.multiply.accumulate(scl)[::-1]
    top = mat.reshape(-1)[1::n + 1]
    bot = mat.reshape(-1)[n::n + 1]
    top[...] = np.sqrt(.5 * np.arange(1, n))
    bot[...] = top
    mat[:, -1] -= scl * c[:-1] / (2.0 * c[-1])
    return mat


def hermroots(c):
    """
    Compute the roots of a Hermite series.

    Return the roots (a.k.a. "zeros") of the polynomial

    .. math:: p(x) = \\sum_i c[i] * H_i(x).

    Parameters
    ----------
    c : 1-D array_like
        1-D array of coefficients.

    Returns
    -------
    out : ndarray
        Array of the roots of the series. If all the roots are real,
        then `out` is also real, otherwise it is complex.

    See Also
    --------
    numpy.polynomial.polynomial.polyroots
    numpy.polynomial.legendre.legroots
    numpy.polynomial.laguerre.lagroots
    numpy.polynomial.chebyshev.chebroots
    numpy.polynomial.hermite_e.hermeroots

    Notes
    -----
    The root estimates are obtained as the eigenvalues of the companion
    matrix, Roots far from the origin of the complex plane may have large
    errors due to the numerical instability of the series for such
    values. Roots with multiplicity greater than 1 will also show larger
    errors as the value of the series near such points is relatively
    insensitive to errors in the roots. Isolated roots near the origin can
    be improved by a few iterations of Newton's method.

    The Hermite series basis polynomials aren't powers of `x` so the
    results of this function may seem unintuitive.

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermroots, hermfromroots
    >>> coef = hermfromroots([-1, 0, 1])
    >>> coef
    array([0.   ,  0.25 ,  0.   ,  0.125])
    >>> hermroots(coef)
    array([-1.00000000e+00, -1.38777878e-17,  1.00000000e+00])

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    if len(c) <= 1:
        return np.array([], dtype=c.dtype)
    if len(c) == 2:
        return np.array([-.5 * c[0] / c[1]])

    # rotated companion matrix reduces error
    m = hermcompanion(c)[::-1, ::-1]
    r = la.eigvals(m)
    r.sort()
    return r


def _normed_hermite_n(x, n):
    """
    Evaluate a normalized Hermite polynomial.

    Compute the value of the normalized Hermite polynomial of degree ``n``
    at the points ``x``.


    Parameters
    ----------
    x : ndarray of double.
        Points at which to evaluate the function
    n : int
        Degree of the normalized Hermite function to be evaluated.

    Returns
    -------
    values : ndarray
        The shape of the return value is described above.

    Notes
    -----
    This function is needed for finding the Gauss points and integration
    weights for high degrees. The values of the standard Hermite functions
    overflow when n >= 207.

    """
    if n == 0:
        return np.full(x.shape, 1 / np.sqrt(np.sqrt(np.pi)))

    c0 = 0.
    c1 = 1. / np.sqrt(np.sqrt(np.pi))
    nd = float(n)
    for i in range(n - 1):
        tmp = c0
        c0 = -c1 * np.sqrt((nd - 1.) / nd)
        c1 = tmp + c1 * x * np.sqrt(2. / nd)
        nd = nd - 1.0
    return c0 + c1 * x * np.sqrt(2)


def hermgauss(deg):
    """
    Gauss-Hermite quadrature.

    Computes the sample points and weights for Gauss-Hermite quadrature.
    These sample points and weights will correctly integrate polynomials of
    degree :math:`2*deg - 1` or less over the interval :math:`[-\\inf, \\inf]`
    with the weight function :math:`f(x) = \\exp(-x^2)`.

    Parameters
    ----------
    deg : int
        Number of sample points and weights. It must be >= 1.

    Returns
    -------
    x : ndarray
        1-D ndarray containing the sample points.
    y : ndarray
        1-D ndarray containing the weights.

    Notes
    -----
    The results have only been tested up to degree 100, higher degrees may
    be problematic. The weights are determined by using the fact that

    .. math:: w_k = c / (H'_n(x_k) * H_{n-1}(x_k))

    where :math:`c` is a constant independent of :math:`k` and :math:`x_k`
    is the k'th root of :math:`H_n`, and then scaling the results to get
    the right value when integrating 1.

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermgauss
    >>> hermgauss(2)
    (array([-0.70710678,  0.70710678]), array([0.88622693, 0.88622693]))

    """
    ideg = pu._as_int(deg, "deg")
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    # first approximation of roots. We use the fact that the companion
    # matrix is symmetric in this case in order to obtain better zeros.
    c = np.array([0] * deg + [1], dtype=np.float64)
    m = hermcompanion(c)
    x = la.eigvalsh(m)

    # improve roots by one application of Newton
    dy = _normed_hermite_n(x, ideg)
    df = _normed_hermite_n(x, ideg - 1) * np.sqrt(2 * ideg)
    x -= dy / df

    # compute the weights. We scale the factor to avoid possible numerical
    # overflow.
    fm = _normed_hermite_n(x, ideg - 1)
    fm /= np.abs(fm).max()
    w = 1 / (fm * fm)

    # for Hermite we can also symmetrize
    w = (w + w[::-1]) / 2
    x = (x - x[::-1]) / 2

    # scale w to get the right value
    w *= np.sqrt(np.pi) / w.sum()

    return x, w


def hermweight(x):
    """
    Weight function of the Hermite polynomials.

    The weight function is :math:`\\exp(-x^2)` and the interval of
    integration is :math:`[-\\inf, \\inf]`. the Hermite polynomials are
    orthogonal, but not normalized, with respect to this weight function.

    Parameters
    ----------
    x : array_like
       Values at which the weight function will be computed.

    Returns
    -------
    w : ndarray
       The weight function at `x`.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial.hermite import hermweight
    >>> x = np.arange(-2, 2)
    >>> hermweight(x)
    array([0.01831564, 0.36787944, 1.        , 0.36787944])

    """
    w = np.exp(-x**2)
    return w


#
# Hermite series class
#

class Hermite(ABCPolyBase):
    """An Hermite series class.

    The Hermite class provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
    attributes and methods listed below.

    Parameters
    ----------
    coef : array_like
        Hermite coefficients in order of increasing degree, i.e,
        ``(1, 2, 3)`` gives ``1*H_0(x) + 2*H_1(x) + 3*H_2(x)``.
    domain : (2,) array_like, optional
        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
        to the interval ``[window[0], window[1]]`` by shifting and scaling.
        The default value is [-1., 1.].
    window : (2,) array_like, optional
        Window, see `domain` for its use. The default value is [-1., 1.].
    symbol : str, optional
        Symbol used to represent the independent variable in string
        representations of the polynomial expression, e.g. for printing.
        The symbol must be a valid Python identifier. Default value is 'x'.

        .. versionadded:: 1.24

    """
    # Virtual Functions
    _add = staticmethod(hermadd)
    _sub = staticmethod(hermsub)
    _mul = staticmethod(hermmul)
    _div = staticmethod(hermdiv)
    _pow = staticmethod(hermpow)
    _val = staticmethod(hermval)
    _int = staticmethod(hermint)
    _der = staticmethod(hermder)
    _fit = staticmethod(hermfit)
    _line = staticmethod(hermline)
    _roots = staticmethod(hermroots)
    _fromroots = staticmethod(hermfromroots)

    # Virtual properties
    domain = np.array(hermdomain)
    window = np.array(hermdomain)
    basis_name = 'H'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\dateutil\rrule.py ===
# -*- coding: utf-8 -*-
"""
The rrule module offers a small, complete, and very fast, implementation of
the recurrence rules documented in the
`iCalendar RFC <https://tools.ietf.org/html/rfc5545>`_,
including support for caching of results.
"""
import calendar
import datetime
import heapq
import itertools
import re
import sys
from functools import wraps
# For warning about deprecation of until and count
from warnings import warn

from six import advance_iterator, integer_types

from six.moves import _thread, range

from ._common import weekday as weekdaybase

try:
    from math import gcd
except ImportError:
    from fractions import gcd

__all__ = ["rrule", "rruleset", "rrulestr",
           "YEARLY", "MONTHLY", "WEEKLY", "DAILY",
           "HOURLY", "MINUTELY", "SECONDLY",
           "MO", "TU", "WE", "TH", "FR", "SA", "SU"]

# Every mask is 7 days longer to handle cross-year weekly periods.
M366MASK = tuple([1]*31+[2]*29+[3]*31+[4]*30+[5]*31+[6]*30 +
                 [7]*31+[8]*31+[9]*30+[10]*31+[11]*30+[12]*31+[1]*7)
M365MASK = list(M366MASK)
M29, M30, M31 = list(range(1, 30)), list(range(1, 31)), list(range(1, 32))
MDAY366MASK = tuple(M31+M29+M31+M30+M31+M30+M31+M31+M30+M31+M30+M31+M31[:7])
MDAY365MASK = list(MDAY366MASK)
M29, M30, M31 = list(range(-29, 0)), list(range(-30, 0)), list(range(-31, 0))
NMDAY366MASK = tuple(M31+M29+M31+M30+M31+M30+M31+M31+M30+M31+M30+M31+M31[:7])
NMDAY365MASK = list(NMDAY366MASK)
M366RANGE = (0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366)
M365RANGE = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365)
WDAYMASK = [0, 1, 2, 3, 4, 5, 6]*55
del M29, M30, M31, M365MASK[59], MDAY365MASK[59], NMDAY365MASK[31]
MDAY365MASK = tuple(MDAY365MASK)
M365MASK = tuple(M365MASK)

FREQNAMES = ['YEARLY', 'MONTHLY', 'WEEKLY', 'DAILY', 'HOURLY', 'MINUTELY', 'SECONDLY']

(YEARLY,
 MONTHLY,
 WEEKLY,
 DAILY,
 HOURLY,
 MINUTELY,
 SECONDLY) = list(range(7))

# Imported on demand.
easter = None
parser = None


class weekday(weekdaybase):
    """
    This version of weekday does not allow n = 0.
    """
    def __init__(self, wkday, n=None):
        if n == 0:
            raise ValueError("Can't create weekday with n==0")

        super(weekday, self).__init__(wkday, n)


MO, TU, WE, TH, FR, SA, SU = weekdays = tuple(weekday(x) for x in range(7))


def _invalidates_cache(f):
    """
    Decorator for rruleset methods which may invalidate the
    cached length.
    """
    @wraps(f)
    def inner_func(self, *args, **kwargs):
        rv = f(self, *args, **kwargs)
        self._invalidate_cache()
        return rv

    return inner_func


class rrulebase(object):
    def __init__(self, cache=False):
        if cache:
            self._cache = []
            self._cache_lock = _thread.allocate_lock()
            self._invalidate_cache()
        else:
            self._cache = None
            self._cache_complete = False
            self._len = None

    def __iter__(self):
        if self._cache_complete:
            return iter(self._cache)
        elif self._cache is None:
            return self._iter()
        else:
            return self._iter_cached()

    def _invalidate_cache(self):
        if self._cache is not None:
            self._cache = []
            self._cache_complete = False
            self._cache_gen = self._iter()

            if self._cache_lock.locked():
                self._cache_lock.release()

        self._len = None

    def _iter_cached(self):
        i = 0
        gen = self._cache_gen
        cache = self._cache
        acquire = self._cache_lock.acquire
        release = self._cache_lock.release
        while gen:
            if i == len(cache):
                acquire()
                if self._cache_complete:
                    break
                try:
                    for j in range(10):
                        cache.append(advance_iterator(gen))
                except StopIteration:
                    self._cache_gen = gen = None
                    self._cache_complete = True
                    break
                release()
            yield cache[i]
            i += 1
        while i < self._len:
            yield cache[i]
            i += 1

    def __getitem__(self, item):
        if self._cache_complete:
            return self._cache[item]
        elif isinstance(item, slice):
            if item.step and item.step < 0:
                return list(iter(self))[item]
            else:
                return list(itertools.islice(self,
                                             item.start or 0,
                                             item.stop or sys.maxsize,
                                             item.step or 1))
        elif item >= 0:
            gen = iter(self)
            try:
                for i in range(item+1):
                    res = advance_iterator(gen)
            except StopIteration:
                raise IndexError
            return res
        else:
            return list(iter(self))[item]

    def __contains__(self, item):
        if self._cache_complete:
            return item in self._cache
        else:
            for i in self:
                if i == item:
                    return True
                elif i > item:
                    return False
        return False

    # __len__() introduces a large performance penalty.
    def count(self):
        """ Returns the number of recurrences in this set. It will have go
            through the whole recurrence, if this hasn't been done before. """
        if self._len is None:
            for x in self:
                pass
        return self._len

    def before(self, dt, inc=False):
        """ Returns the last recurrence before the given datetime instance. The
            inc keyword defines what happens if dt is an occurrence. With
            inc=True, if dt itself is an occurrence, it will be returned. """
        if self._cache_complete:
            gen = self._cache
        else:
            gen = self
        last = None
        if inc:
            for i in gen:
                if i > dt:
                    break
                last = i
        else:
            for i in gen:
                if i >= dt:
                    break
                last = i
        return last

    def after(self, dt, inc=False):
        """ Returns the first recurrence after the given datetime instance. The
            inc keyword defines what happens if dt is an occurrence. With
            inc=True, if dt itself is an occurrence, it will be returned.  """
        if self._cache_complete:
            gen = self._cache
        else:
            gen = self
        if inc:
            for i in gen:
                if i >= dt:
                    return i
        else:
            for i in gen:
                if i > dt:
                    return i
        return None

    def xafter(self, dt, count=None, inc=False):
        """
        Generator which yields up to `count` recurrences after the given
        datetime instance, equivalent to `after`.

        :param dt:
            The datetime at which to start generating recurrences.

        :param count:
            The maximum number of recurrences to generate. If `None` (default),
            dates are generated until the recurrence rule is exhausted.

        :param inc:
            If `dt` is an instance of the rule and `inc` is `True`, it is
            included in the output.

        :yields: Yields a sequence of `datetime` objects.
        """

        if self._cache_complete:
            gen = self._cache
        else:
            gen = self

        # Select the comparison function
        if inc:
            comp = lambda dc, dtc: dc >= dtc
        else:
            comp = lambda dc, dtc: dc > dtc

        # Generate dates
        n = 0
        for d in gen:
            if comp(d, dt):
                if count is not None:
                    n += 1
                    if n > count:
                        break

                yield d

    def between(self, after, before, inc=False, count=1):
        """ Returns all the occurrences of the rrule between after and before.
        The inc keyword defines what happens if after and/or before are
        themselves occurrences. With inc=True, they will be included in the
        list, if they are found in the recurrence set. """
        if self._cache_complete:
            gen = self._cache
        else:
            gen = self
        started = False
        l = []
        if inc:
            for i in gen:
                if i > before:
                    break
                elif not started:
                    if i >= after:
                        started = True
                        l.append(i)
                else:
                    l.append(i)
        else:
            for i in gen:
                if i >= before:
                    break
                elif not started:
                    if i > after:
                        started = True
                        l.append(i)
                else:
                    l.append(i)
        return l


class rrule(rrulebase):
    """
    That's the base of the rrule operation. It accepts all the keywords
    defined in the RFC as its constructor parameters (except byday,
    which was renamed to byweekday) and more. The constructor prototype is::

            rrule(freq)

    Where freq must be one of YEARLY, MONTHLY, WEEKLY, DAILY, HOURLY, MINUTELY,
    or SECONDLY.

    .. note::
        Per RFC section 3.3.10, recurrence instances falling on invalid dates
        and times are ignored rather than coerced:

            Recurrence rules may generate recurrence instances with an invalid
            date (e.g., February 30) or nonexistent local time (e.g., 1:30 AM
            on a day where the local time is moved forward by an hour at 1:00
            AM).  Such recurrence instances MUST be ignored and MUST NOT be
            counted as part of the recurrence set.

        This can lead to possibly surprising behavior when, for example, the
        start date occurs at the end of the month:

        >>> from dateutil.rrule import rrule, MONTHLY
        >>> from datetime import datetime
        >>> start_date = datetime(2014, 12, 31)
        >>> list(rrule(freq=MONTHLY, count=4, dtstart=start_date))
        ... # doctest: +NORMALIZE_WHITESPACE
        [datetime.datetime(2014, 12, 31, 0, 0),
         datetime.datetime(2015, 1, 31, 0, 0),
         datetime.datetime(2015, 3, 31, 0, 0),
         datetime.datetime(2015, 5, 31, 0, 0)]

    Additionally, it supports the following keyword arguments:

    :param dtstart:
        The recurrence start. Besides being the base for the recurrence,
        missing parameters in the final recurrence instances will also be
        extracted from this date. If not given, datetime.now() will be used
        instead.
    :param interval:
        The interval between each freq iteration. For example, when using
        YEARLY, an interval of 2 means once every two years, but with HOURLY,
        it means once every two hours. The default interval is 1.
    :param wkst:
        The week start day. Must be one of the MO, TU, WE constants, or an
        integer, specifying the first day of the week. This will affect
        recurrences based on weekly periods. The default week start is got
        from calendar.firstweekday(), and may be modified by
        calendar.setfirstweekday().
    :param count:
        If given, this determines how many occurrences will be generated.

        .. note::
            As of version 2.5.0, the use of the keyword ``until`` in conjunction
            with ``count`` is deprecated, to make sure ``dateutil`` is fully
            compliant with `RFC-5545 Sec. 3.3.10 <https://tools.ietf.org/
            html/rfc5545#section-3.3.10>`_. Therefore, ``until`` and ``count``
            **must not** occur in the same call to ``rrule``.
    :param until:
        If given, this must be a datetime instance specifying the upper-bound
        limit of the recurrence. The last recurrence in the rule is the greatest
        datetime that is less than or equal to the value specified in the
        ``until`` parameter.

        .. note::
            As of version 2.5.0, the use of the keyword ``until`` in conjunction
            with ``count`` is deprecated, to make sure ``dateutil`` is fully
            compliant with `RFC-5545 Sec. 3.3.10 <https://tools.ietf.org/
            html/rfc5545#section-3.3.10>`_. Therefore, ``until`` and ``count``
            **must not** occur in the same call to ``rrule``.
    :param bysetpos:
        If given, it must be either an integer, or a sequence of integers,
        positive or negative. Each given integer will specify an occurrence
        number, corresponding to the nth occurrence of the rule inside the
        frequency period. For example, a bysetpos of -1 if combined with a
        MONTHLY frequency, and a byweekday of (MO, TU, WE, TH, FR), will
        result in the last work day of every month.
    :param bymonth:
        If given, it must be either an integer, or a sequence of integers,
        meaning the months to apply the recurrence to.
    :param bymonthday:
        If given, it must be either an integer, or a sequence of integers,
        meaning the month days to apply the recurrence to.
    :param byyearday:
        If given, it must be either an integer, or a sequence of integers,
        meaning the year days to apply the recurrence to.
    :param byeaster:
        If given, it must be either an integer, or a sequence of integers,
        positive or negative. Each integer will define an offset from the
        Easter Sunday. Passing the offset 0 to byeaster will yield the Easter
        Sunday itself. This is an extension to the RFC specification.
    :param byweekno:
        If given, it must be either an integer, or a sequence of integers,
        meaning the week numbers to apply the recurrence to. Week numbers
        have the meaning described in ISO8601, that is, the first week of
        the year is that containing at least four days of the new year.
    :param byweekday:
        If given, it must be either an integer (0 == MO), a sequence of
        integers, one of the weekday constants (MO, TU, etc), or a sequence
        of these constants. When given, these variables will define the
        weekdays where the recurrence will be applied. It's also possible to
        use an argument n for the weekday instances, which will mean the nth
        occurrence of this weekday in the period. For example, with MONTHLY,
        or with YEARLY and BYMONTH, using FR(+1) in byweekday will specify the
        first friday of the month where the recurrence happens. Notice that in
        the RFC documentation, this is specified as BYDAY, but was renamed to
        avoid the ambiguity of that keyword.
    :param byhour:
        If given, it must be either an integer, or a sequence of integers,
        meaning the hours to apply the recurrence to.
    :param byminute:
        If given, it must be either an integer, or a sequence of integers,
        meaning the minutes to apply the recurrence to.
    :param bysecond:
        If given, it must be either an integer, or a sequence of integers,
        meaning the seconds to apply the recurrence to.
    :param cache:
        If given, it must be a boolean value specifying to enable or disable
        caching of results. If you will use the same rrule instance multiple
        times, enabling caching will improve the performance considerably.
     """
    def __init__(self, freq, dtstart=None,
                 interval=1, wkst=None, count=None, until=None, bysetpos=None,
                 bymonth=None, bymonthday=None, byyearday=None, byeaster=None,
                 byweekno=None, byweekday=None,
                 byhour=None, byminute=None, bysecond=None,
                 cache=False):
        super(rrule, self).__init__(cache)
        global easter
        if not dtstart:
            if until and until.tzinfo:
                dtstart = datetime.datetime.now(tz=until.tzinfo).replace(microsecond=0)
            else:
                dtstart = datetime.datetime.now().replace(microsecond=0)
        elif not isinstance(dtstart, datetime.datetime):
            dtstart = datetime.datetime.fromordinal(dtstart.toordinal())
        else:
            dtstart = dtstart.replace(microsecond=0)
        self._dtstart = dtstart
        self._tzinfo = dtstart.tzinfo
        self._freq = freq
        self._interval = interval
        self._count = count

        # Cache the original byxxx rules, if they are provided, as the _byxxx
        # attributes do not necessarily map to the inputs, and this can be
        # a problem in generating the strings. Only store things if they've
        # been supplied (the string retrieval will just use .get())
        self._original_rule = {}

        if until and not isinstance(until, datetime.datetime):
            until = datetime.datetime.fromordinal(until.toordinal())
        self._until = until

        if self._dtstart and self._until:
            if (self._dtstart.tzinfo is not None) != (self._until.tzinfo is not None):
                # According to RFC5545 Section 3.3.10:
                # https://tools.ietf.org/html/rfc5545#section-3.3.10
                #
                # > If the "DTSTART" property is specified as a date with UTC
                # > time or a date with local time and time zone reference,
                # > then the UNTIL rule part MUST be specified as a date with
                # > UTC time.
                raise ValueError(
                    'RRULE UNTIL values must be specified in UTC when DTSTART '
                    'is timezone-aware'
                )

        if count is not None and until:
            warn("Using both 'count' and 'until' is inconsistent with RFC 5545"
                 " and has been deprecated in dateutil. Future versions will "
                 "raise an error.", DeprecationWarning)

        if wkst is None:
            self._wkst = calendar.firstweekday()
        elif isinstance(wkst, integer_types):
            self._wkst = wkst
        else:
            self._wkst = wkst.weekday

        if bysetpos is None:
            self._bysetpos = None
        elif isinstance(bysetpos, integer_types):
            if bysetpos == 0 or not (-366 <= bysetpos <= 366):
                raise ValueError("bysetpos must be between 1 and 366, "
                                 "or between -366 and -1")
            self._bysetpos = (bysetpos,)
        else:
            self._bysetpos = tuple(bysetpos)
            for pos in self._bysetpos:
                if pos == 0 or not (-366 <= pos <= 366):
                    raise ValueError("bysetpos must be between 1 and 366, "
                                     "or between -366 and -1")

        if self._bysetpos:
            self._original_rule['bysetpos'] = self._bysetpos

        if (byweekno is None and byyearday is None and bymonthday is None and
                byweekday is None and byeaster is None):
            if freq == YEARLY:
                if bymonth is None:
                    bymonth = dtstart.month
                    self._original_rule['bymonth'] = None
                bymonthday = dtstart.day
                self._original_rule['bymonthday'] = None
            elif freq == MONTHLY:
                bymonthday = dtstart.day
                self._original_rule['bymonthday'] = None
            elif freq == WEEKLY:
                byweekday = dtstart.weekday()
                self._original_rule['byweekday'] = None

        # bymonth
        if bymonth is None:
            self._bymonth = None
        else:
            if isinstance(bymonth, integer_types):
                bymonth = (bymonth,)

            self._bymonth = tuple(sorted(set(bymonth)))

            if 'bymonth' not in self._original_rule:
                self._original_rule['bymonth'] = self._bymonth

        # byyearday
        if byyearday is None:
            self._byyearday = None
        else:
            if isinstance(byyearday, integer_types):
                byyearday = (byyearday,)

            self._byyearday = tuple(sorted(set(byyearday)))
            self._original_rule['byyearday'] = self._byyearday

        # byeaster
        if byeaster is not None:
            if not easter:
                from dateutil import easter
            if isinstance(byeaster, integer_types):
                self._byeaster = (byeaster,)
            else:
                self._byeaster = tuple(sorted(byeaster))

            self._original_rule['byeaster'] = self._byeaster
        else:
            self._byeaster = None

        # bymonthday
        if bymonthday is None:
            self._bymonthday = ()
            self._bynmonthday = ()
        else:
            if isinstance(bymonthday, integer_types):
                bymonthday = (bymonthday,)

            bymonthday = set(bymonthday)            # Ensure it's unique

            self._bymonthday = tuple(sorted(x for x in bymonthday if x > 0))
            self._bynmonthday = tuple(sorted(x for x in bymonthday if x < 0))

            # Storing positive numbers first, then negative numbers
            if 'bymonthday' not in self._original_rule:
                self._original_rule['bymonthday'] = tuple(
                    itertools.chain(self._bymonthday, self._bynmonthday))

        # byweekno
        if byweekno is None:
            self._byweekno = None
        else:
            if isinstance(byweekno, integer_types):
                byweekno = (byweekno,)

            self._byweekno = tuple(sorted(set(byweekno)))

            self._original_rule['byweekno'] = self._byweekno

        # byweekday / bynweekday
        if byweekday is None:
            self._byweekday = None
            self._bynweekday = None
        else:
            # If it's one of the valid non-sequence types, convert to a
            # single-element sequence before the iterator that builds the
            # byweekday set.
            if isinstance(byweekday, integer_types) or hasattr(byweekday, "n"):
                byweekday = (byweekday,)

            self._byweekday = set()
            self._bynweekday = set()
            for wday in byweekday:
                if isinstance(wday, integer_types):
                    self._byweekday.add(wday)
                elif not wday.n or freq > MONTHLY:
                    self._byweekday.add(wday.weekday)
                else:
                    self._bynweekday.add((wday.weekday, wday.n))

            if not self._byweekday:
                self._byweekday = None
            elif not self._bynweekday:
                self._bynweekday = None

            if self._byweekday is not None:
                self._byweekday = tuple(sorted(self._byweekday))
                orig_byweekday = [weekday(x) for x in self._byweekday]
            else:
                orig_byweekday = ()

            if self._bynweekday is not None:
                self._bynweekday = tuple(sorted(self._bynweekday))
                orig_bynweekday = [weekday(*x) for x in self._bynweekday]
            else:
                orig_bynweekday = ()

            if 'byweekday' not in self._original_rule:
                self._original_rule['byweekday'] = tuple(itertools.chain(
                    orig_byweekday, orig_bynweekday))

        # byhour
        if byhour is None:
            if freq < HOURLY:
                self._byhour = {dtstart.hour}
            else:
                self._byhour = None
        else:
            if isinstance(byhour, integer_types):
                byhour = (byhour,)

            if freq == HOURLY:
                self._byhour = self.__construct_byset(start=dtstart.hour,
                                                      byxxx=byhour,
                                                      base=24)
            else:
                self._byhour = set(byhour)

            self._byhour = tuple(sorted(self._byhour))
            self._original_rule['byhour'] = self._byhour

        # byminute
        if byminute is None:
            if freq < MINUTELY:
                self._byminute = {dtstart.minute}
            else:
                self._byminute = None
        else:
            if isinstance(byminute, integer_types):
                byminute = (byminute,)

            if freq == MINUTELY:
                self._byminute = self.__construct_byset(start=dtstart.minute,
                                                        byxxx=byminute,
                                                        base=60)
            else:
                self._byminute = set(byminute)

            self._byminute = tuple(sorted(self._byminute))
            self._original_rule['byminute'] = self._byminute

        # bysecond
        if bysecond is None:
            if freq < SECONDLY:
                self._bysecond = ((dtstart.second,))
            else:
                self._bysecond = None
        else:
            if isinstance(bysecond, integer_types):
                bysecond = (bysecond,)

            self._bysecond = set(bysecond)

            if freq == SECONDLY:
                self._bysecond = self.__construct_byset(start=dtstart.second,
                                                        byxxx=bysecond,
                                                        base=60)
            else:
                self._bysecond = set(bysecond)

            self._bysecond = tuple(sorted(self._bysecond))
            self._original_rule['bysecond'] = self._bysecond

        if self._freq >= HOURLY:
            self._timeset = None
        else:
            self._timeset = []
            for hour in self._byhour:
                for minute in self._byminute:
                    for second in self._bysecond:
                        self._timeset.append(
                            datetime.time(hour, minute, second,
                                          tzinfo=self._tzinfo))
            self._timeset.sort()
            self._timeset = tuple(self._timeset)

    def __str__(self):
        """
        Output a string that would generate this RRULE if passed to rrulestr.
        This is mostly compatible with RFC5545, except for the
        dateutil-specific extension BYEASTER.
        """

        output = []
        h, m, s = [None] * 3
        if self._dtstart:
            output.append(self._dtstart.strftime('DTSTART:%Y%m%dT%H%M%S'))
            h, m, s = self._dtstart.timetuple()[3:6]

        parts = ['FREQ=' + FREQNAMES[self._freq]]
        if self._interval != 1:
            parts.append('INTERVAL=' + str(self._interval))

        if self._wkst:
            parts.append('WKST=' + repr(weekday(self._wkst))[0:2])

        if self._count is not None:
            parts.append('COUNT=' + str(self._count))

        if self._until:
            parts.append(self._until.strftime('UNTIL=%Y%m%dT%H%M%S'))

        if self._original_rule.get('byweekday') is not None:
            # The str() method on weekday objects doesn't generate
            # RFC5545-compliant strings, so we should modify that.
            original_rule = dict(self._original_rule)
            wday_strings = []
            for wday in original_rule['byweekday']:
                if wday.n:
                    wday_strings.append('{n:+d}{wday}'.format(
                        n=wday.n,
                        wday=repr(wday)[0:2]))
                else:
                    wday_strings.append(repr(wday))

            original_rule['byweekday'] = wday_strings
        else:
            original_rule = self._original_rule

        partfmt = '{name}={vals}'
        for name, key in [('BYSETPOS', 'bysetpos'),
                          ('BYMONTH', 'bymonth'),
                          ('BYMONTHDAY', 'bymonthday'),
                          ('BYYEARDAY', 'byyearday'),
                          ('BYWEEKNO', 'byweekno'),
                          ('BYDAY', 'byweekday'),
                          ('BYHOUR', 'byhour'),
                          ('BYMINUTE', 'byminute'),
                          ('BYSECOND', 'bysecond'),
                          ('BYEASTER', 'byeaster')]:
            value = original_rule.get(key)
            if value:
                parts.append(partfmt.format(name=name, vals=(','.join(str(v)
                                                             for v in value))))

        output.append('RRULE:' + ';'.join(parts))
        return '\n'.join(output)

    def replace(self, **kwargs):
        """Return new rrule with same attributes except for those attributes given new
           values by whichever keyword arguments are specified."""
        new_kwargs = {"interval": self._interval,
                      "count": self._count,
                      "dtstart": self._dtstart,
                      "freq": self._freq,
                      "until": self._until,
                      "wkst": self._wkst,
                      "cache": False if self._cache is None else True }
        new_kwargs.update(self._original_rule)
        new_kwargs.update(kwargs)
        return rrule(**new_kwargs)

    def _iter(self):
        year, month, day, hour, minute, second, weekday, yearday, _ = \
            self._dtstart.timetuple()

        # Some local variables to speed things up a bit
        freq = self._freq
        interval = self._interval
        wkst = self._wkst
        until = self._until
        bymonth = self._bymonth
        byweekno = self._byweekno
        byyearday = self._byyearday
        byweekday = self._byweekday
        byeaster = self._byeaster
        bymonthday = self._bymonthday
        bynmonthday = self._bynmonthday
        bysetpos = self._bysetpos
        byhour = self._byhour
        byminute = self._byminute
        bysecond = self._bysecond

        ii = _iterinfo(self)
        ii.rebuild(year, month)

        getdayset = {YEARLY: ii.ydayset,
                     MONTHLY: ii.mdayset,
                     WEEKLY: ii.wdayset,
                     DAILY: ii.ddayset,
                     HOURLY: ii.ddayset,
                     MINUTELY: ii.ddayset,
                     SECONDLY: ii.ddayset}[freq]

        if freq < HOURLY:
            timeset = self._timeset
        else:
            gettimeset = {HOURLY: ii.htimeset,
                          MINUTELY: ii.mtimeset,
                          SECONDLY: ii.stimeset}[freq]
            if ((freq >= HOURLY and
                 self._byhour and hour not in self._byhour) or
                (freq >= MINUTELY and
                 self._byminute and minute not in self._byminute) or
                (freq >= SECONDLY and
                 self._bysecond and second not in self._bysecond)):
                timeset = ()
            else:
                timeset = gettimeset(hour, minute, second)

        total = 0
        count = self._count
        while True:
            # Get dayset with the right frequency
            dayset, start, end = getdayset(year, month, day)

            # Do the "hard" work ;-)
            filtered = False
            for i in dayset[start:end]:
                if ((bymonth and ii.mmask[i] not in bymonth) or
                    (byweekno and not ii.wnomask[i]) or
                    (byweekday and ii.wdaymask[i] not in byweekday) or
                    (ii.nwdaymask and not ii.nwdaymask[i]) or
                    (byeaster and not ii.eastermask[i]) or
                    ((bymonthday or bynmonthday) and
                     ii.mdaymask[i] not in bymonthday and
                     ii.nmdaymask[i] not in bynmonthday) or
                    (byyearday and
                     ((i < ii.yearlen and i+1 not in byyearday and
                       -ii.yearlen+i not in byyearday) or
                      (i >= ii.yearlen and i+1-ii.yearlen not in byyearday and
                       -ii.nextyearlen+i-ii.yearlen not in byyearday)))):
                    dayset[i] = None
                    filtered = True

            # Output results
            if bysetpos and timeset:
                poslist = []
                for pos in bysetpos:
                    if pos < 0:
                        daypos, timepos = divmod(pos, len(timeset))
                    else:
                        daypos, timepos = divmod(pos-1, len(timeset))
                    try:
                        i = [x for x in dayset[start:end]
                             if x is not None][daypos]
                        time = timeset[timepos]
                    except IndexError:
                        pass
                    else:
                        date = datetime.date.fromordinal(ii.yearordinal+i)
                        res = datetime.datetime.combine(date, time)
                        if res not in poslist:
                            poslist.append(res)
                poslist.sort()
                for res in poslist:
                    if until and res > until:
                        self._len = total
                        return
                    elif res >= self._dtstart:
                        if count is not None:
                            count -= 1
                            if count < 0:
                                self._len = total
                                return
                        total += 1
                        yield res
            else:
                for i in dayset[start:end]:
                    if i is not None:
                        date = datetime.date.fromordinal(ii.yearordinal + i)
                        for time in timeset:
                            res = datetime.datetime.combine(date, time)
                            if until and res > until:
                                self._len = total
                                return
                            elif res >= self._dtstart:
                                if count is not None:
                                    count -= 1
                                    if count < 0:
                                        self._len = total
                                        return

                                total += 1
                                yield res

            # Handle frequency and interval
            fixday = False
            if freq == YEARLY:
                year += interval
                if year > datetime.MAXYEAR:
                    self._len = total
                    return
                ii.rebuild(year, month)
            elif freq == MONTHLY:
                month += interval
                if month > 12:
                    div, mod = divmod(month, 12)
                    month = mod
                    year += div
                    if month == 0:
                        month = 12
                        year -= 1
                    if year > datetime.MAXYEAR:
                        self._len = total
                        return
                ii.rebuild(year, month)
            elif freq == WEEKLY:
                if wkst > weekday:
                    day += -(weekday+1+(6-wkst))+self._interval*7
                else:
                    day += -(weekday-wkst)+self._interval*7
                weekday = wkst
                fixday = True
            elif freq == DAILY:
                day += interval
                fixday = True
            elif freq == HOURLY:
                if filtered:
                    # Jump to one iteration before next day
                    hour += ((23-hour)//interval)*interval

                if byhour:
                    ndays, hour = self.__mod_distance(value=hour,
                                                      byxxx=self._byhour,
                                                      base=24)
                else:
                    ndays, hour = divmod(hour+interval, 24)

                if ndays:
                    day += ndays
                    fixday = True

                timeset = gettimeset(hour, minute, second)
            elif freq == MINUTELY:
                if filtered:
                    # Jump to one iteration before next day
                    minute += ((1439-(hour*60+minute))//interval)*interval

                valid = False
                rep_rate = (24*60)
                for j in range(rep_rate // gcd(interval, rep_rate)):
                    if byminute:
                        nhours, minute = \
                            self.__mod_distance(value=minute,
                                                byxxx=self._byminute,
                                                base=60)
                    else:
                        nhours, minute = divmod(minute+interval, 60)

                    div, hour = divmod(hour+nhours, 24)
                    if div:
                        day += div
                        fixday = True
                        filtered = False

                    if not byhour or hour in byhour:
                        valid = True
                        break

                if not valid:
                    raise ValueError('Invalid combination of interval and ' +
                                     'byhour resulting in empty rule.')

                timeset = gettimeset(hour, minute, second)
            elif freq == SECONDLY:
                if filtered:
                    # Jump to one iteration before next day
                    second += (((86399 - (hour * 3600 + minute * 60 + second))
                                // interval) * interval)

                rep_rate = (24 * 3600)
                valid = False
                for j in range(0, rep_rate // gcd(interval, rep_rate)):
                    if bysecond:
                        nminutes, second = \
                            self.__mod_distance(value=second,
                                                byxxx=self._bysecond,
                                                base=60)
                    else:
                        nminutes, second = divmod(second+interval, 60)

                    div, minute = divmod(minute+nminutes, 60)
                    if div:
                        hour += div
                        div, hour = divmod(hour, 24)
                        if div:
                            day += div
                            fixday = True

                    if ((not byhour or hour in byhour) and
                            (not byminute or minute in byminute) and
                            (not bysecond or second in bysecond)):
                        valid = True
                        break

                if not valid:
                    raise ValueError('Invalid combination of interval, ' +
                                     'byhour and byminute resulting in empty' +
                                     ' rule.')

                timeset = gettimeset(hour, minute, second)

            if fixday and day > 28:
                daysinmonth = calendar.monthrange(year, month)[1]
                if day > daysinmonth:
                    while day > daysinmonth:
                        day -= daysinmonth
                        month += 1
                        if month == 13:
                            month = 1
                            year += 1
                            if year > datetime.MAXYEAR:
                                self._len = total
                                return
                        daysinmonth = calendar.monthrange(year, month)[1]
                    ii.rebuild(year, month)

    def __construct_byset(self, start, byxxx, base):
        """
        If a `BYXXX` sequence is passed to the constructor at the same level as
        `FREQ` (e.g. `FREQ=HOURLY,BYHOUR={2,4,7},INTERVAL=3`), there are some
        specifications which cannot be reached given some starting conditions.

        This occurs whenever the interval is not coprime with the base of a
        given unit and the difference between the starting position and the
        ending position is not coprime with the greatest common denominator
        between the interval and the base. For example, with a FREQ of hourly
        starting at 17:00 and an interval of 4, the only valid values for
        BYHOUR would be {21, 1, 5, 9, 13, 17}, because 4 and 24 are not
        coprime.

        :param start:
            Specifies the starting position.
        :param byxxx:
            An iterable containing the list of allowed values.
        :param base:
            The largest allowable value for the specified frequency (e.g.
            24 hours, 60 minutes).

        This does not preserve the type of the iterable, returning a set, since
        the values should be unique and the order is irrelevant, this will
        speed up later lookups.

        In the event of an empty set, raises a :exception:`ValueError`, as this
        results in an empty rrule.
        """

        cset = set()

        # Support a single byxxx value.
        if isinstance(byxxx, integer_types):
            byxxx = (byxxx, )

        for num in byxxx:
            i_gcd = gcd(self._interval, base)
            # Use divmod rather than % because we need to wrap negative nums.
            if i_gcd == 1 or divmod(num - start, i_gcd)[1] == 0:
                cset.add(num)

        if len(cset) == 0:
            raise ValueError("Invalid rrule byxxx generates an empty set.")

        return cset

    def __mod_distance(self, value, byxxx, base):
        """
        Calculates the next value in a sequence where the `FREQ` parameter is
        specified along with a `BYXXX` parameter at the same "level"
        (e.g. `HOURLY` specified with `BYHOUR`).

        :param value:
            The old value of the component.
        :param byxxx:
            The `BYXXX` set, which should have been generated by
            `rrule._construct_byset`, or something else which checks that a
            valid rule is present.
        :param base:
            The largest allowable value for the specified frequency (e.g.
            24 hours, 60 minutes).

        If a valid value is not found after `base` iterations (the maximum
        number before the sequence would start to repeat), this raises a
        :exception:`ValueError`, as no valid values were found.

        This returns a tuple of `divmod(n*interval, base)`, where `n` is the
        smallest number of `interval` repetitions until the next specified
        value in `byxxx` is found.
        """
        accumulator = 0
        for ii in range(1, base + 1):
            # Using divmod() over % to account for negative intervals
            div, value = divmod(value + self._interval, base)
            accumulator += div
            if value in byxxx:
                return (accumulator, value)


class _iterinfo(object):
    __slots__ = ["rrule", "lastyear", "lastmonth",
                 "yearlen", "nextyearlen", "yearordinal", "yearweekday",
                 "mmask", "mrange", "mdaymask", "nmdaymask",
                 "wdaymask", "wnomask", "nwdaymask", "eastermask"]

    def __init__(self, rrule):
        for attr in self.__slots__:
            setattr(self, attr, None)
        self.rrule = rrule

    def rebuild(self, year, month):
        # Every mask is 7 days longer to handle cross-year weekly periods.
        rr = self.rrule
        if year != self.lastyear:
            self.yearlen = 365 + calendar.isleap(year)
            self.nextyearlen = 365 + calendar.isleap(year + 1)
            firstyday = datetime.date(year, 1, 1)
            self.yearordinal = firstyday.toordinal()
            self.yearweekday = firstyday.weekday()

            wday = datetime.date(year, 1, 1).weekday()
            if self.yearlen == 365:
                self.mmask = M365MASK
                self.mdaymask = MDAY365MASK
                self.nmdaymask = NMDAY365MASK
                self.wdaymask = WDAYMASK[wday:]
                self.mrange = M365RANGE
            else:
                self.mmask = M366MASK
                self.mdaymask = MDAY366MASK
                self.nmdaymask = NMDAY366MASK
                self.wdaymask = WDAYMASK[wday:]
                self.mrange = M366RANGE

            if not rr._byweekno:
                self.wnomask = None
            else:
                self.wnomask = [0]*(self.yearlen+7)
                # no1wkst = firstwkst = self.wdaymask.index(rr._wkst)
                no1wkst = firstwkst = (7-self.yearweekday+rr._wkst) % 7
                if no1wkst >= 4:
                    no1wkst = 0
                    # Number of days in the year, plus the days we got
                    # from last year.
                    wyearlen = self.yearlen+(self.yearweekday-rr._wkst) % 7
                else:
                    # Number of days in the year, minus the days we
                    # left in last year.
                    wyearlen = self.yearlen-no1wkst
                div, mod = divmod(wyearlen, 7)
                numweeks = div+mod//4
                for n in rr._byweekno:
                    if n < 0:
                        n += numweeks+1
                    if not (0 < n <= numweeks):
                        continue
                    if n > 1:
                        i = no1wkst+(n-1)*7
                        if no1wkst != firstwkst:
                            i -= 7-firstwkst
                    else:
                        i = no1wkst
                    for j in range(7):
                        self.wnomask[i] = 1
                        i += 1
                        if self.wdaymask[i] == rr._wkst:
                            break
                if 1 in rr._byweekno:
                    # Check week number 1 of next year as well
                    # TODO: Check -numweeks for next year.
                    i = no1wkst+numweeks*7
                    if no1wkst != firstwkst:
                        i -= 7-firstwkst
                    if i < self.yearlen:
                        # If week starts in next year, we
                        # don't care about it.
                        for j in range(7):
                            self.wnomask[i] = 1
                            i += 1
                            if self.wdaymask[i] == rr._wkst:
                                break
                if no1wkst:
                    # Check last week number of last year as
                    # well. If no1wkst is 0, either the year
                    # started on week start, or week number 1
                    # got days from last year, so there are no
                    # days from last year's last week number in
                    # this year.
                    if -1 not in rr._byweekno:
                        lyearweekday = datetime.date(year-1, 1, 1).weekday()
                        lno1wkst = (7-lyearweekday+rr._wkst) % 7
                        lyearlen = 365+calendar.isleap(year-1)
                        if lno1wkst >= 4:
                            lno1wkst = 0
                            lnumweeks = 52+(lyearlen +
                                            (lyearweekday-rr._wkst) % 7) % 7//4
                        else:
                            lnumweeks = 52+(self.yearlen-no1wkst) % 7//4
                    else:
                        lnumweeks = -1
                    if lnumweeks in rr._byweekno:
                        for i in range(no1wkst):
                            self.wnomask[i] = 1

        if (rr._bynweekday and (month != self.lastmonth or
                                year != self.lastyear)):
            ranges = []
            if rr._freq == YEARLY:
                if rr._bymonth:
                    for month in rr._bymonth:
                        ranges.append(self.mrange[month-1:month+1])
                else:
                    ranges = [(0, self.yearlen)]
            elif rr._freq == MONTHLY:
                ranges = [self.mrange[month-1:month+1]]
            if ranges:
                # Weekly frequency won't get here, so we may not
                # care about cross-year weekly periods.
                self.nwdaymask = [0]*self.yearlen
                for first, last in ranges:
                    last -= 1
                    for wday, n in rr._bynweekday:
                        if n < 0:
                            i = last+(n+1)*7
                            i -= (self.wdaymask[i]-wday) % 7
                        else:
                            i = first+(n-1)*7
                            i += (7-self.wdaymask[i]+wday) % 7
                        if first <= i <= last:
                            self.nwdaymask[i] = 1

        if rr._byeaster:
            self.eastermask = [0]*(self.yearlen+7)
            eyday = easter.easter(year).toordinal()-self.yearordinal
            for offset in rr._byeaster:
                self.eastermask[eyday+offset] = 1

        self.lastyear = year
        self.lastmonth = month

    def ydayset(self, year, month, day):
        return list(range(self.yearlen)), 0, self.yearlen

    def mdayset(self, year, month, day):
        dset = [None]*self.yearlen
        start, end = self.mrange[month-1:month+1]
        for i in range(start, end):
            dset[i] = i
        return dset, start, end

    def wdayset(self, year, month, day):
        # We need to handle cross-year weeks here.
        dset = [None]*(self.yearlen+7)
        i = datetime.date(year, month, day).toordinal()-self.yearordinal
        start = i
        for j in range(7):
            dset[i] = i
            i += 1
            # if (not (0 <= i < self.yearlen) or
            #    self.wdaymask[i] == self.rrule._wkst):
            # This will cross the year boundary, if necessary.
            if self.wdaymask[i] == self.rrule._wkst:
                break
        return dset, start, i

    def ddayset(self, year, month, day):
        dset = [None] * self.yearlen
        i = datetime.date(year, month, day).toordinal() - self.yearordinal
        dset[i] = i
        return dset, i, i + 1

    def htimeset(self, hour, minute, second):
        tset = []
        rr = self.rrule
        for minute in rr._byminute:
            for second in rr._bysecond:
                tset.append(datetime.time(hour, minute, second,
                                          tzinfo=rr._tzinfo))
        tset.sort()
        return tset

    def mtimeset(self, hour, minute, second):
        tset = []
        rr = self.rrule
        for second in rr._bysecond:
            tset.append(datetime.time(hour, minute, second, tzinfo=rr._tzinfo))
        tset.sort()
        return tset

    def stimeset(self, hour, minute, second):
        return (datetime.time(hour, minute, second,
                tzinfo=self.rrule._tzinfo),)


class rruleset(rrulebase):
    """ The rruleset type allows more complex recurrence setups, mixing
    multiple rules, dates, exclusion rules, and exclusion dates. The type
    constructor takes the following keyword arguments:

    :param cache: If True, caching of results will be enabled, improving
                  performance of multiple queries considerably. """

    class _genitem(object):
        def __init__(self, genlist, gen):
            try:
                self.dt = advance_iterator(gen)
                genlist.append(self)
            except StopIteration:
                pass
            self.genlist = genlist
            self.gen = gen

        def __next__(self):
            try:
                self.dt = advance_iterator(self.gen)
            except StopIteration:
                if self.genlist[0] is self:
                    heapq.heappop(self.genlist)
                else:
                    self.genlist.remove(self)
                    heapq.heapify(self.genlist)

        next = __next__

        def __lt__(self, other):
            return self.dt < other.dt

        def __gt__(self, other):
            return self.dt > other.dt

        def __eq__(self, other):
            return self.dt == other.dt

        def __ne__(self, other):
            return self.dt != other.dt

    def __init__(self, cache=False):
        super(rruleset, self).__init__(cache)
        self._rrule = []
        self._rdate = []
        self._exrule = []
        self._exdate = []

    @_invalidates_cache
    def rrule(self, rrule):
        """ Include the given :py:class:`rrule` instance in the recurrence set
            generation. """
        self._rrule.append(rrule)

    @_invalidates_cache
    def rdate(self, rdate):
        """ Include the given :py:class:`datetime` instance in the recurrence
            set generation. """
        self._rdate.append(rdate)

    @_invalidates_cache
    def exrule(self, exrule):
        """ Include the given rrule instance in the recurrence set exclusion
            list. Dates which are part of the given recurrence rules will not
            be generated, even if some inclusive rrule or rdate matches them.
        """
        self._exrule.append(exrule)

    @_invalidates_cache
    def exdate(self, exdate):
        """ Include the given datetime instance in the recurrence set
            exclusion list. Dates included that way will not be generated,
            even if some inclusive rrule or rdate matches them. """
        self._exdate.append(exdate)

    def _iter(self):
        rlist = []
        self._rdate.sort()
        self._genitem(rlist, iter(self._rdate))
        for gen in [iter(x) for x in self._rrule]:
            self._genitem(rlist, gen)
        exlist = []
        self._exdate.sort()
        self._genitem(exlist, iter(self._exdate))
        for gen in [iter(x) for x in self._exrule]:
            self._genitem(exlist, gen)
        lastdt = None
        total = 0
        heapq.heapify(rlist)
        heapq.heapify(exlist)
        while rlist:
            ritem = rlist[0]
            if not lastdt or lastdt != ritem.dt:
                while exlist and exlist[0] < ritem:
                    exitem = exlist[0]
                    advance_iterator(exitem)
                    if exlist and exlist[0] is exitem:
                        heapq.heapreplace(exlist, exitem)
                if not exlist or ritem != exlist[0]:
                    total += 1
                    yield ritem.dt
                lastdt = ritem.dt
            advance_iterator(ritem)
            if rlist and rlist[0] is ritem:
                heapq.heapreplace(rlist, ritem)
        self._len = total




class _rrulestr(object):
    """ Parses a string representation of a recurrence rule or set of
    recurrence rules.

    :param s:
        Required, a string defining one or more recurrence rules.

    :param dtstart:
        If given, used as the default recurrence start if not specified in the
        rule string.

    :param cache:
        If set ``True`` caching of results will be enabled, improving
        performance of multiple queries considerably.

    :param unfold:
        If set ``True`` indicates that a rule string is split over more
        than one line and should be joined before processing.

    :param forceset:
        If set ``True`` forces a :class:`dateutil.rrule.rruleset` to
        be returned.

    :param compatible:
        If set ``True`` forces ``unfold`` and ``forceset`` to be ``True``.

    :param ignoretz:
        If set ``True``, time zones in parsed strings are ignored and a naive
        :class:`datetime.datetime` object is returned.

    :param tzids:
        If given, a callable or mapping used to retrieve a
        :class:`datetime.tzinfo` from a string representation.
        Defaults to :func:`dateutil.tz.gettz`.

    :param tzinfos:
        Additional time zone names / aliases which may be present in a string
        representation.  See :func:`dateutil.parser.parse` for more
        information.

    :return:
        Returns a :class:`dateutil.rrule.rruleset` or
        :class:`dateutil.rrule.rrule`
    """

    _freq_map = {"YEARLY": YEARLY,
                 "MONTHLY": MONTHLY,
                 "WEEKLY": WEEKLY,
                 "DAILY": DAILY,
                 "HOURLY": HOURLY,
                 "MINUTELY": MINUTELY,
                 "SECONDLY": SECONDLY}

    _weekday_map = {"MO": 0, "TU": 1, "WE": 2, "TH": 3,
                    "FR": 4, "SA": 5, "SU": 6}

    def _handle_int(self, rrkwargs, name, value, **kwargs):
        rrkwargs[name.lower()] = int(value)

    def _handle_int_list(self, rrkwargs, name, value, **kwargs):
        rrkwargs[name.lower()] = [int(x) for x in value.split(',')]

    _handle_INTERVAL = _handle_int
    _handle_COUNT = _handle_int
    _handle_BYSETPOS = _handle_int_list
    _handle_BYMONTH = _handle_int_list
    _handle_BYMONTHDAY = _handle_int_list
    _handle_BYYEARDAY = _handle_int_list
    _handle_BYEASTER = _handle_int_list
    _handle_BYWEEKNO = _handle_int_list
    _handle_BYHOUR = _handle_int_list
    _handle_BYMINUTE = _handle_int_list
    _handle_BYSECOND = _handle_int_list

    def _handle_FREQ(self, rrkwargs, name, value, **kwargs):
        rrkwargs["freq"] = self._freq_map[value]

    def _handle_UNTIL(self, rrkwargs, name, value, **kwargs):
        global parser
        if not parser:
            from dateutil import parser
        try:
            rrkwargs["until"] = parser.parse(value,
                                             ignoretz=kwargs.get("ignoretz"),
                                             tzinfos=kwargs.get("tzinfos"))
        except ValueError:
            raise ValueError("invalid until date")

    def _handle_WKST(self, rrkwargs, name, value, **kwargs):
        rrkwargs["wkst"] = self._weekday_map[value]

    def _handle_BYWEEKDAY(self, rrkwargs, name, value, **kwargs):
        """
        Two ways to specify this: +1MO or MO(+1)
        """
        l = []
        for wday in value.split(','):
            if '(' in wday:
                # If it's of the form TH(+1), etc.
                splt = wday.split('(')
                w = splt[0]
                n = int(splt[1][:-1])
            elif len(wday):
                # If it's of the form +1MO
                for i in range(len(wday)):
                    if wday[i] not in '+-0123456789':
                        break
                n = wday[:i] or None
                w = wday[i:]
                if n:
                    n = int(n)
            else:
                raise ValueError("Invalid (empty) BYDAY specification.")

            l.append(weekdays[self._weekday_map[w]](n))
        rrkwargs["byweekday"] = l

    _handle_BYDAY = _handle_BYWEEKDAY

    def _parse_rfc_rrule(self, line,
                         dtstart=None,
                         cache=False,
                         ignoretz=False,
                         tzinfos=None):
        if line.find(':') != -1:
            name, value = line.split(':')
            if name != "RRULE":
                raise ValueError("unknown parameter name")
        else:
            value = line
        rrkwargs = {}
        for pair in value.split(';'):
            name, value = pair.split('=')
            name = name.upper()
            value = value.upper()
            try:
                getattr(self, "_handle_"+name)(rrkwargs, name, value,
                                               ignoretz=ignoretz,
                                               tzinfos=tzinfos)
            except AttributeError:
                raise ValueError("unknown parameter '%s'" % name)
            except (KeyError, ValueError):
                raise ValueError("invalid '%s': %s" % (name, value))
        return rrule(dtstart=dtstart, cache=cache, **rrkwargs)

    def _parse_date_value(self, date_value, parms, rule_tzids,
                          ignoretz, tzids, tzinfos):
        global parser
        if not parser:
            from dateutil import parser

        datevals = []
        value_found = False
        TZID = None

        for parm in parms:
            if parm.startswith("TZID="):
                try:
                    tzkey = rule_tzids[parm.split('TZID=')[-1]]
                except KeyError:
                    continue
                if tzids is None:
                    from . import tz
                    tzlookup = tz.gettz
                elif callable(tzids):
                    tzlookup = tzids
                else:
                    tzlookup = getattr(tzids, 'get', None)
                    if tzlookup is None:
                        msg = ('tzids must be a callable, mapping, or None, '
                               'not %s' % tzids)
                        raise ValueError(msg)

                TZID = tzlookup(tzkey)
                continue

            # RFC 5445 3.8.2.4: The VALUE parameter is optional, but may be found
            # only once.
            if parm not in {"VALUE=DATE-TIME", "VALUE=DATE"}:
                raise ValueError("unsupported parm: " + parm)
            else:
                if value_found:
                    msg = ("Duplicate value parameter found in: " + parm)
                    raise ValueError(msg)
                value_found = True

        for datestr in date_value.split(','):
            date = parser.parse(datestr, ignoretz=ignoretz, tzinfos=tzinfos)
            if TZID is not None:
                if date.tzinfo is None:
                    date = date.replace(tzinfo=TZID)
                else:
                    raise ValueError('DTSTART/EXDATE specifies multiple timezone')
            datevals.append(date)

        return datevals

    def _parse_rfc(self, s,
                   dtstart=None,
                   cache=False,
                   unfold=False,
                   forceset=False,
                   compatible=False,
                   ignoretz=False,
                   tzids=None,
                   tzinfos=None):
        global parser
        if compatible:
            forceset = True
            unfold = True

        TZID_NAMES = dict(map(
            lambda x: (x.upper(), x),
            re.findall('TZID=(?P<name>[^:]+):', s)
        ))
        s = s.upper()
        if not s.strip():
            raise ValueError("empty string")
        if unfold:
            lines = s.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i].rstrip()
                if not line:
                    del lines[i]
                elif i > 0 and line[0] == " ":
                    lines[i-1] += line[1:]
                    del lines[i]
                else:
                    i += 1
        else:
            lines = s.split()
        if (not forceset and len(lines) == 1 and (s.find(':') == -1 or
                                                  s.startswith('RRULE:'))):
            return self._parse_rfc_rrule(lines[0], cache=cache,
                                         dtstart=dtstart, ignoretz=ignoretz,
                                         tzinfos=tzinfos)
        else:
            rrulevals = []
            rdatevals = []
            exrulevals = []
            exdatevals = []
            for line in lines:
                if not line:
                    continue
                if line.find(':') == -1:
                    name = "RRULE"
                    value = line
                else:
                    name, value = line.split(':', 1)
                parms = name.split(';')
                if not parms:
                    raise ValueError("empty property name")
                name = parms[0]
                parms = parms[1:]
                if name == "RRULE":
                    for parm in parms:
                        raise ValueError("unsupported RRULE parm: "+parm)
                    rrulevals.append(value)
                elif name == "RDATE":
                    for parm in parms:
                        if parm != "VALUE=DATE-TIME":
                            raise ValueError("unsupported RDATE parm: "+parm)
                    rdatevals.append(value)
                elif name == "EXRULE":
                    for parm in parms:
                        raise ValueError("unsupported EXRULE parm: "+parm)
                    exrulevals.append(value)
                elif name == "EXDATE":
                    exdatevals.extend(
                        self._parse_date_value(value, parms,
                                               TZID_NAMES, ignoretz,
                                               tzids, tzinfos)
                    )
                elif name == "DTSTART":
                    dtvals = self._parse_date_value(value, parms, TZID_NAMES,
                                                    ignoretz, tzids, tzinfos)
                    if len(dtvals) != 1:
                        raise ValueError("Multiple DTSTART values specified:" +
                                         value)
                    dtstart = dtvals[0]
                else:
                    raise ValueError("unsupported property: "+name)
            if (forceset or len(rrulevals) > 1 or rdatevals
                    or exrulevals or exdatevals):
                if not parser and (rdatevals or exdatevals):
                    from dateutil import parser
                rset = rruleset(cache=cache)
                for value in rrulevals:
                    rset.rrule(self._parse_rfc_rrule(value, dtstart=dtstart,
                                                     ignoretz=ignoretz,
                                                     tzinfos=tzinfos))
                for value in rdatevals:
                    for datestr in value.split(','):
                        rset.rdate(parser.parse(datestr,
                                                ignoretz=ignoretz,
                                                tzinfos=tzinfos))
                for value in exrulevals:
                    rset.exrule(self._parse_rfc_rrule(value, dtstart=dtstart,
                                                      ignoretz=ignoretz,
                                                      tzinfos=tzinfos))
                for value in exdatevals:
                    rset.exdate(value)
                if compatible and dtstart:
                    rset.rdate(dtstart)
                return rset
            else:
                return self._parse_rfc_rrule(rrulevals[0],
                                             dtstart=dtstart,
                                             cache=cache,
                                             ignoretz=ignoretz,
                                             tzinfos=tzinfos)

    def __call__(self, s, **kwargs):
        return self._parse_rfc(s, **kwargs)


rrulestr = _rrulestr()

# vim:ts=4:sw=4:et