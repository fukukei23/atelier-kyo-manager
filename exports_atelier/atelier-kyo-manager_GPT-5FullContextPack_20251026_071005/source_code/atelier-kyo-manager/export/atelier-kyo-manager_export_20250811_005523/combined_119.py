
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\array_algos\take.py ===
from __future__ import annotations

import functools
from typing import (
    TYPE_CHECKING,
    cast,
    overload,
)

import numpy as np

from pandas._libs import (
    algos as libalgos,
    lib,
)

from pandas.core.dtypes.cast import maybe_promote
from pandas.core.dtypes.common import (
    ensure_platform_int,
    is_1d_only_ea_dtype,
)
from pandas.core.dtypes.missing import na_value_for_dtype

from pandas.core.construction import ensure_wrapped_if_datetimelike

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        AxisInt,
        npt,
    )

    from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
    from pandas.core.arrays.base import ExtensionArray


@overload
def take_nd(
    arr: np.ndarray,
    indexer,
    axis: AxisInt = ...,
    fill_value=...,
    allow_fill: bool = ...,
) -> np.ndarray:
    ...


@overload
def take_nd(
    arr: ExtensionArray,
    indexer,
    axis: AxisInt = ...,
    fill_value=...,
    allow_fill: bool = ...,
) -> ArrayLike:
    ...


def take_nd(
    arr: ArrayLike,
    indexer,
    axis: AxisInt = 0,
    fill_value=lib.no_default,
    allow_fill: bool = True,
) -> ArrayLike:
    """
    Specialized Cython take which sets NaN values in one pass

    This dispatches to ``take`` defined on ExtensionArrays.

    Note: this function assumes that the indexer is a valid(ated) indexer with
    no out of bound indices.

    Parameters
    ----------
    arr : np.ndarray or ExtensionArray
        Input array.
    indexer : ndarray
        1-D array of indices to take, subarrays corresponding to -1 value
        indices are filed with fill_value
    axis : int, default 0
        Axis to take from
    fill_value : any, default np.nan
        Fill value to replace -1 values with
    allow_fill : bool, default True
        If False, indexer is assumed to contain no -1 values so no filling
        will be done.  This short-circuits computation of a mask.  Result is
        undefined if allow_fill == False and -1 is present in indexer.

    Returns
    -------
    subarray : np.ndarray or ExtensionArray
        May be the same type as the input, or cast to an ndarray.
    """
    if fill_value is lib.no_default:
        fill_value = na_value_for_dtype(arr.dtype, compat=False)
    elif lib.is_np_dtype(arr.dtype, "mM"):
        dtype, fill_value = maybe_promote(arr.dtype, fill_value)
        if arr.dtype != dtype:
            # EA.take is strict about returning a new object of the same type
            # so for that case cast upfront
            arr = arr.astype(dtype)

    if not isinstance(arr, np.ndarray):
        # i.e. ExtensionArray,
        # includes for EA to catch DatetimeArray, TimedeltaArray
        if not is_1d_only_ea_dtype(arr.dtype):
            # i.e. DatetimeArray, TimedeltaArray
            arr = cast("NDArrayBackedExtensionArray", arr)
            return arr.take(
                indexer, fill_value=fill_value, allow_fill=allow_fill, axis=axis
            )

        return arr.take(indexer, fill_value=fill_value, allow_fill=allow_fill)

    arr = np.asarray(arr)
    return _take_nd_ndarray(arr, indexer, axis, fill_value, allow_fill)


def _take_nd_ndarray(
    arr: np.ndarray,
    indexer: npt.NDArray[np.intp] | None,
    axis: AxisInt,
    fill_value,
    allow_fill: bool,
) -> np.ndarray:
    if indexer is None:
        indexer = np.arange(arr.shape[axis], dtype=np.intp)
        dtype, fill_value = arr.dtype, arr.dtype.type()
    else:
        indexer = ensure_platform_int(indexer)

    dtype, fill_value, mask_info = _take_preprocess_indexer_and_fill_value(
        arr, indexer, fill_value, allow_fill
    )

    flip_order = False
    if arr.ndim == 2 and arr.flags.f_contiguous:
        flip_order = True

    if flip_order:
        arr = arr.T
        axis = arr.ndim - axis - 1

    # at this point, it's guaranteed that dtype can hold both the arr values
    # and the fill_value
    out_shape_ = list(arr.shape)
    out_shape_[axis] = len(indexer)
    out_shape = tuple(out_shape_)
    if arr.flags.f_contiguous and axis == arr.ndim - 1:
        # minor tweak that can make an order-of-magnitude difference
        # for dataframes initialized directly from 2-d ndarrays
        # (s.t. df.values is c-contiguous and df._mgr.blocks[0] is its
        # f-contiguous transpose)
        out = np.empty(out_shape, dtype=dtype, order="F")
    else:
        out = np.empty(out_shape, dtype=dtype)

    func = _get_take_nd_function(
        arr.ndim, arr.dtype, out.dtype, axis=axis, mask_info=mask_info
    )
    func(arr, indexer, out, fill_value)

    if flip_order:
        out = out.T
    return out


def take_1d(
    arr: ArrayLike,
    indexer: npt.NDArray[np.intp],
    fill_value=None,
    allow_fill: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> ArrayLike:
    """
    Specialized version for 1D arrays. Differences compared to `take_nd`:

    - Assumes input array has already been converted to numpy array / EA
    - Assumes indexer is already guaranteed to be intp dtype ndarray
    - Only works for 1D arrays

    To ensure the lowest possible overhead.

    Note: similarly to `take_nd`, this function assumes that the indexer is
    a valid(ated) indexer with no out of bound indices.

    Parameters
    ----------
    arr : np.ndarray or ExtensionArray
        Input array.
    indexer : ndarray
        1-D array of indices to take (validated indices, intp dtype).
    fill_value : any, default np.nan
        Fill value to replace -1 values with
    allow_fill : bool, default True
        If False, indexer is assumed to contain no -1 values so no filling
        will be done.  This short-circuits computation of a mask. Result is
        undefined if allow_fill == False and -1 is present in indexer.
    mask : np.ndarray, optional, default None
        If `allow_fill` is True, and the mask (where indexer == -1) is already
        known, it can be passed to avoid recomputation.
    """
    if not isinstance(arr, np.ndarray):
        # ExtensionArray -> dispatch to their method
        return arr.take(indexer, fill_value=fill_value, allow_fill=allow_fill)

    if not allow_fill:
        return arr.take(indexer)

    dtype, fill_value, mask_info = _take_preprocess_indexer_and_fill_value(
        arr, indexer, fill_value, True, mask
    )

    # at this point, it's guaranteed that dtype can hold both the arr values
    # and the fill_value
    out = np.empty(indexer.shape, dtype=dtype)

    func = _get_take_nd_function(
        arr.ndim, arr.dtype, out.dtype, axis=0, mask_info=mask_info
    )
    func(arr, indexer, out, fill_value)

    return out


def take_2d_multi(
    arr: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    fill_value=np.nan,
) -> np.ndarray:
    """
    Specialized Cython take which sets NaN values in one pass.
    """
    # This is only called from one place in DataFrame._reindex_multi,
    #  so we know indexer is well-behaved.
    assert indexer is not None
    assert indexer[0] is not None
    assert indexer[1] is not None

    row_idx, col_idx = indexer

    row_idx = ensure_platform_int(row_idx)
    col_idx = ensure_platform_int(col_idx)
    indexer = row_idx, col_idx
    mask_info = None

    # check for promotion based on types only (do this first because
    # it's faster than computing a mask)
    dtype, fill_value = maybe_promote(arr.dtype, fill_value)
    if dtype != arr.dtype:
        # check if promotion is actually required based on indexer
        row_mask = row_idx == -1
        col_mask = col_idx == -1
        row_needs = row_mask.any()
        col_needs = col_mask.any()
        mask_info = (row_mask, col_mask), (row_needs, col_needs)

        if not (row_needs or col_needs):
            # if not, then depromote, set fill_value to dummy
            # (it won't be used but we don't want the cython code
            # to crash when trying to cast it to dtype)
            dtype, fill_value = arr.dtype, arr.dtype.type()

    # at this point, it's guaranteed that dtype can hold both the arr values
    # and the fill_value
    out_shape = len(row_idx), len(col_idx)
    out = np.empty(out_shape, dtype=dtype)

    func = _take_2d_multi_dict.get((arr.dtype.name, out.dtype.name), None)
    if func is None and arr.dtype != out.dtype:
        func = _take_2d_multi_dict.get((out.dtype.name, out.dtype.name), None)
        if func is not None:
            func = _convert_wrapper(func, out.dtype)

    if func is not None:
        func(arr, indexer, out=out, fill_value=fill_value)
    else:
        # test_reindex_multi
        _take_2d_multi_object(
            arr, indexer, out, fill_value=fill_value, mask_info=mask_info
        )

    return out


@functools.lru_cache
def _get_take_nd_function_cached(
    ndim: int, arr_dtype: np.dtype, out_dtype: np.dtype, axis: AxisInt
):
    """
    Part of _get_take_nd_function below that doesn't need `mask_info` and thus
    can be cached (mask_info potentially contains a numpy ndarray which is not
    hashable and thus cannot be used as argument for cached function).
    """
    tup = (arr_dtype.name, out_dtype.name)
    if ndim == 1:
        func = _take_1d_dict.get(tup, None)
    elif ndim == 2:
        if axis == 0:
            func = _take_2d_axis0_dict.get(tup, None)
        else:
            func = _take_2d_axis1_dict.get(tup, None)
    if func is not None:
        return func

    # We get here with string, uint, float16, and complex dtypes that could
    #  potentially be handled in algos_take_helper.
    #  Also a couple with (M8[ns], object) and (m8[ns], object)
    tup = (out_dtype.name, out_dtype.name)
    if ndim == 1:
        func = _take_1d_dict.get(tup, None)
    elif ndim == 2:
        if axis == 0:
            func = _take_2d_axis0_dict.get(tup, None)
        else:
            func = _take_2d_axis1_dict.get(tup, None)
    if func is not None:
        func = _convert_wrapper(func, out_dtype)
        return func

    return None


def _get_take_nd_function(
    ndim: int,
    arr_dtype: np.dtype,
    out_dtype: np.dtype,
    axis: AxisInt = 0,
    mask_info=None,
):
    """
    Get the appropriate "take" implementation for the given dimension, axis
    and dtypes.
    """
    func = None
    if ndim <= 2:
        # for this part we don't need `mask_info` -> use the cached algo lookup
        func = _get_take_nd_function_cached(ndim, arr_dtype, out_dtype, axis)

    if func is None:

        def func(arr, indexer, out, fill_value=np.nan) -> None:
            indexer = ensure_platform_int(indexer)
            _take_nd_object(
                arr, indexer, out, axis=axis, fill_value=fill_value, mask_info=mask_info
            )

    return func


def _view_wrapper(f, arr_dtype=None, out_dtype=None, fill_wrap=None):
    def wrapper(
        arr: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=np.nan
    ) -> None:
        if arr_dtype is not None:
            arr = arr.view(arr_dtype)
        if out_dtype is not None:
            out = out.view(out_dtype)
        if fill_wrap is not None:
            # FIXME: if we get here with dt64/td64 we need to be sure we have
            #  matching resos
            if fill_value.dtype.kind == "m":
                fill_value = fill_value.astype("m8[ns]")
            else:
                fill_value = fill_value.astype("M8[ns]")
            fill_value = fill_wrap(fill_value)

        f(arr, indexer, out, fill_value=fill_value)

    return wrapper


def _convert_wrapper(f, conv_dtype):
    def wrapper(
        arr: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=np.nan
    ) -> None:
        if conv_dtype == object:
            # GH#39755 avoid casting dt64/td64 to integers
            arr = ensure_wrapped_if_datetimelike(arr)
        arr = arr.astype(conv_dtype)
        f(arr, indexer, out, fill_value=fill_value)

    return wrapper


_take_1d_dict = {
    ("int8", "int8"): libalgos.take_1d_int8_int8,
    ("int8", "int32"): libalgos.take_1d_int8_int32,
    ("int8", "int64"): libalgos.take_1d_int8_int64,
    ("int8", "float64"): libalgos.take_1d_int8_float64,
    ("int16", "int16"): libalgos.take_1d_int16_int16,
    ("int16", "int32"): libalgos.take_1d_int16_int32,
    ("int16", "int64"): libalgos.take_1d_int16_int64,
    ("int16", "float64"): libalgos.take_1d_int16_float64,
    ("int32", "int32"): libalgos.take_1d_int32_int32,
    ("int32", "int64"): libalgos.take_1d_int32_int64,
    ("int32", "float64"): libalgos.take_1d_int32_float64,
    ("int64", "int64"): libalgos.take_1d_int64_int64,
    ("int64", "float64"): libalgos.take_1d_int64_float64,
    ("float32", "float32"): libalgos.take_1d_float32_float32,
    ("float32", "float64"): libalgos.take_1d_float32_float64,
    ("float64", "float64"): libalgos.take_1d_float64_float64,
    ("object", "object"): libalgos.take_1d_object_object,
    ("bool", "bool"): _view_wrapper(libalgos.take_1d_bool_bool, np.uint8, np.uint8),
    ("bool", "object"): _view_wrapper(libalgos.take_1d_bool_object, np.uint8, None),
    ("datetime64[ns]", "datetime64[ns]"): _view_wrapper(
        libalgos.take_1d_int64_int64, np.int64, np.int64, np.int64
    ),
    ("timedelta64[ns]", "timedelta64[ns]"): _view_wrapper(
        libalgos.take_1d_int64_int64, np.int64, np.int64, np.int64
    ),
}

_take_2d_axis0_dict = {
    ("int8", "int8"): libalgos.take_2d_axis0_int8_int8,
    ("int8", "int32"): libalgos.take_2d_axis0_int8_int32,
    ("int8", "int64"): libalgos.take_2d_axis0_int8_int64,
    ("int8", "float64"): libalgos.take_2d_axis0_int8_float64,
    ("int16", "int16"): libalgos.take_2d_axis0_int16_int16,
    ("int16", "int32"): libalgos.take_2d_axis0_int16_int32,
    ("int16", "int64"): libalgos.take_2d_axis0_int16_int64,
    ("int16", "float64"): libalgos.take_2d_axis0_int16_float64,
    ("int32", "int32"): libalgos.take_2d_axis0_int32_int32,
    ("int32", "int64"): libalgos.take_2d_axis0_int32_int64,
    ("int32", "float64"): libalgos.take_2d_axis0_int32_float64,
    ("int64", "int64"): libalgos.take_2d_axis0_int64_int64,
    ("int64", "float64"): libalgos.take_2d_axis0_int64_float64,
    ("float32", "float32"): libalgos.take_2d_axis0_float32_float32,
    ("float32", "float64"): libalgos.take_2d_axis0_float32_float64,
    ("float64", "float64"): libalgos.take_2d_axis0_float64_float64,
    ("object", "object"): libalgos.take_2d_axis0_object_object,
    ("bool", "bool"): _view_wrapper(
        libalgos.take_2d_axis0_bool_bool, np.uint8, np.uint8
    ),
    ("bool", "object"): _view_wrapper(
        libalgos.take_2d_axis0_bool_object, np.uint8, None
    ),
    ("datetime64[ns]", "datetime64[ns]"): _view_wrapper(
        libalgos.take_2d_axis0_int64_int64, np.int64, np.int64, fill_wrap=np.int64
    ),
    ("timedelta64[ns]", "timedelta64[ns]"): _view_wrapper(
        libalgos.take_2d_axis0_int64_int64, np.int64, np.int64, fill_wrap=np.int64
    ),
}

_take_2d_axis1_dict = {
    ("int8", "int8"): libalgos.take_2d_axis1_int8_int8,
    ("int8", "int32"): libalgos.take_2d_axis1_int8_int32,
    ("int8", "int64"): libalgos.take_2d_axis1_int8_int64,
    ("int8", "float64"): libalgos.take_2d_axis1_int8_float64,
    ("int16", "int16"): libalgos.take_2d_axis1_int16_int16,
    ("int16", "int32"): libalgos.take_2d_axis1_int16_int32,
    ("int16", "int64"): libalgos.take_2d_axis1_int16_int64,
    ("int16", "float64"): libalgos.take_2d_axis1_int16_float64,
    ("int32", "int32"): libalgos.take_2d_axis1_int32_int32,
    ("int32", "int64"): libalgos.take_2d_axis1_int32_int64,
    ("int32", "float64"): libalgos.take_2d_axis1_int32_float64,
    ("int64", "int64"): libalgos.take_2d_axis1_int64_int64,
    ("int64", "float64"): libalgos.take_2d_axis1_int64_float64,
    ("float32", "float32"): libalgos.take_2d_axis1_float32_float32,
    ("float32", "float64"): libalgos.take_2d_axis1_float32_float64,
    ("float64", "float64"): libalgos.take_2d_axis1_float64_float64,
    ("object", "object"): libalgos.take_2d_axis1_object_object,
    ("bool", "bool"): _view_wrapper(
        libalgos.take_2d_axis1_bool_bool, np.uint8, np.uint8
    ),
    ("bool", "object"): _view_wrapper(
        libalgos.take_2d_axis1_bool_object, np.uint8, None
    ),
    ("datetime64[ns]", "datetime64[ns]"): _view_wrapper(
        libalgos.take_2d_axis1_int64_int64, np.int64, np.int64, fill_wrap=np.int64
    ),
    ("timedelta64[ns]", "timedelta64[ns]"): _view_wrapper(
        libalgos.take_2d_axis1_int64_int64, np.int64, np.int64, fill_wrap=np.int64
    ),
}

_take_2d_multi_dict = {
    ("int8", "int8"): libalgos.take_2d_multi_int8_int8,
    ("int8", "int32"): libalgos.take_2d_multi_int8_int32,
    ("int8", "int64"): libalgos.take_2d_multi_int8_int64,
    ("int8", "float64"): libalgos.take_2d_multi_int8_float64,
    ("int16", "int16"): libalgos.take_2d_multi_int16_int16,
    ("int16", "int32"): libalgos.take_2d_multi_int16_int32,
    ("int16", "int64"): libalgos.take_2d_multi_int16_int64,
    ("int16", "float64"): libalgos.take_2d_multi_int16_float64,
    ("int32", "int32"): libalgos.take_2d_multi_int32_int32,
    ("int32", "int64"): libalgos.take_2d_multi_int32_int64,
    ("int32", "float64"): libalgos.take_2d_multi_int32_float64,
    ("int64", "int64"): libalgos.take_2d_multi_int64_int64,
    ("int64", "float64"): libalgos.take_2d_multi_int64_float64,
    ("float32", "float32"): libalgos.take_2d_multi_float32_float32,
    ("float32", "float64"): libalgos.take_2d_multi_float32_float64,
    ("float64", "float64"): libalgos.take_2d_multi_float64_float64,
    ("object", "object"): libalgos.take_2d_multi_object_object,
    ("bool", "bool"): _view_wrapper(
        libalgos.take_2d_multi_bool_bool, np.uint8, np.uint8
    ),
    ("bool", "object"): _view_wrapper(
        libalgos.take_2d_multi_bool_object, np.uint8, None
    ),
    ("datetime64[ns]", "datetime64[ns]"): _view_wrapper(
        libalgos.take_2d_multi_int64_int64, np.int64, np.int64, fill_wrap=np.int64
    ),
    ("timedelta64[ns]", "timedelta64[ns]"): _view_wrapper(
        libalgos.take_2d_multi_int64_int64, np.int64, np.int64, fill_wrap=np.int64
    ),
}


def _take_nd_object(
    arr: np.ndarray,
    indexer: npt.NDArray[np.intp],
    out: np.ndarray,
    axis: AxisInt,
    fill_value,
    mask_info,
) -> None:
    if mask_info is not None:
        mask, needs_masking = mask_info
    else:
        mask = indexer == -1
        needs_masking = mask.any()
    if arr.dtype != out.dtype:
        arr = arr.astype(out.dtype)
    if arr.shape[axis] > 0:
        arr.take(indexer, axis=axis, out=out)
    if needs_masking:
        outindexer = [slice(None)] * arr.ndim
        outindexer[axis] = mask
        out[tuple(outindexer)] = fill_value


def _take_2d_multi_object(
    arr: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    out: np.ndarray,
    fill_value,
    mask_info,
) -> None:
    # this is not ideal, performance-wise, but it's better than raising
    # an exception (best to optimize in Cython to avoid getting here)
    row_idx, col_idx = indexer  # both np.intp
    if mask_info is not None:
        (row_mask, col_mask), (row_needs, col_needs) = mask_info
    else:
        row_mask = row_idx == -1
        col_mask = col_idx == -1
        row_needs = row_mask.any()
        col_needs = col_mask.any()
    if fill_value is not None:
        if row_needs:
            out[row_mask, :] = fill_value
        if col_needs:
            out[:, col_mask] = fill_value
    for i, u_ in enumerate(row_idx):
        if u_ != -1:
            for j, v in enumerate(col_idx):
                if v != -1:
                    out[i, j] = arr[u_, v]


def _take_preprocess_indexer_and_fill_value(
    arr: np.ndarray,
    indexer: npt.NDArray[np.intp],
    fill_value,
    allow_fill: bool,
    mask: npt.NDArray[np.bool_] | None = None,
):
    mask_info: tuple[np.ndarray | None, bool] | None = None

    if not allow_fill:
        dtype, fill_value = arr.dtype, arr.dtype.type()
        mask_info = None, False
    else:
        # check for promotion based on types only (do this first because
        # it's faster than computing a mask)
        dtype, fill_value = maybe_promote(arr.dtype, fill_value)
        if dtype != arr.dtype:
            # check if promotion is actually required based on indexer
            if mask is not None:
                needs_masking = True
            else:
                mask = indexer == -1
                needs_masking = bool(mask.any())
            mask_info = mask, needs_masking
            if not needs_masking:
                # if not, then depromote, set fill_value to dummy
                # (it won't be used but we don't want the cython code
                # to crash when trying to cast it to dtype)
                dtype, fill_value = arr.dtype, arr.dtype.type()

    return dtype, fill_value, mask_info

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\ext\mypy\infer.py ===
# ext/mypy/infer.py
# Copyright (C) 2021-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

from __future__ import annotations

from typing import Optional
from typing import Sequence

from mypy.maptype import map_instance_to_supertype
from mypy.nodes import AssignmentStmt
from mypy.nodes import CallExpr
from mypy.nodes import Expression
from mypy.nodes import FuncDef
from mypy.nodes import LambdaExpr
from mypy.nodes import MemberExpr
from mypy.nodes import NameExpr
from mypy.nodes import RefExpr
from mypy.nodes import StrExpr
from mypy.nodes import TypeInfo
from mypy.nodes import Var
from mypy.plugin import SemanticAnalyzerPluginInterface
from mypy.subtypes import is_subtype
from mypy.types import AnyType
from mypy.types import CallableType
from mypy.types import get_proper_type
from mypy.types import Instance
from mypy.types import NoneType
from mypy.types import ProperType
from mypy.types import TypeOfAny
from mypy.types import UnionType

from . import names
from . import util


def infer_type_from_right_hand_nameexpr(
    api: SemanticAnalyzerPluginInterface,
    stmt: AssignmentStmt,
    node: Var,
    left_hand_explicit_type: Optional[ProperType],
    infer_from_right_side: RefExpr,
) -> Optional[ProperType]:
    type_id = names.type_id_for_callee(infer_from_right_side)
    if type_id is None:
        return None
    elif type_id is names.MAPPED:
        python_type_for_type = _infer_type_from_mapped(
            api, stmt, node, left_hand_explicit_type, infer_from_right_side
        )
    elif type_id is names.COLUMN:
        python_type_for_type = _infer_type_from_decl_column(
            api, stmt, node, left_hand_explicit_type
        )
    elif type_id is names.RELATIONSHIP:
        python_type_for_type = _infer_type_from_relationship(
            api, stmt, node, left_hand_explicit_type
        )
    elif type_id is names.COLUMN_PROPERTY:
        python_type_for_type = _infer_type_from_decl_column_property(
            api, stmt, node, left_hand_explicit_type
        )
    elif type_id is names.SYNONYM_PROPERTY:
        python_type_for_type = infer_type_from_left_hand_type_only(
            api, node, left_hand_explicit_type
        )
    elif type_id is names.COMPOSITE_PROPERTY:
        python_type_for_type = _infer_type_from_decl_composite_property(
            api, stmt, node, left_hand_explicit_type
        )
    else:
        return None

    return python_type_for_type


def _infer_type_from_relationship(
    api: SemanticAnalyzerPluginInterface,
    stmt: AssignmentStmt,
    node: Var,
    left_hand_explicit_type: Optional[ProperType],
) -> Optional[ProperType]:
    """Infer the type of mapping from a relationship.

    E.g.::

        @reg.mapped
        class MyClass:
            # ...

            addresses = relationship(Address, uselist=True)

            order: Mapped["Order"] = relationship("Order")

    Will resolve in mypy as::

        @reg.mapped
        class MyClass:
            # ...

            addresses: Mapped[List[Address]]

            order: Mapped["Order"]

    """

    assert isinstance(stmt.rvalue, CallExpr)
    target_cls_arg = stmt.rvalue.args[0]
    python_type_for_type: Optional[ProperType] = None

    if isinstance(target_cls_arg, NameExpr) and isinstance(
        target_cls_arg.node, TypeInfo
    ):
        # type
        related_object_type = target_cls_arg.node
        python_type_for_type = Instance(related_object_type, [])

    # other cases not covered - an error message directs the user
    # to set an explicit type annotation
    #
    # node.type == str, it's a string
    # if isinstance(target_cls_arg, NameExpr) and isinstance(
    #     target_cls_arg.node, Var
    # )
    # points to a type
    # isinstance(target_cls_arg, NameExpr) and isinstance(
    #     target_cls_arg.node, TypeAlias
    # )
    # string expression
    # isinstance(target_cls_arg, StrExpr)

    uselist_arg = util.get_callexpr_kwarg(stmt.rvalue, "uselist")
    collection_cls_arg: Optional[Expression] = util.get_callexpr_kwarg(
        stmt.rvalue, "collection_class"
    )
    type_is_a_collection = False

    # this can be used to determine Optional for a many-to-one
    # in the same way nullable=False could be used, if we start supporting
    # that.
    # innerjoin_arg = util.get_callexpr_kwarg(stmt.rvalue, "innerjoin")

    if (
        uselist_arg is not None
        and api.parse_bool(uselist_arg) is True
        and collection_cls_arg is None
    ):
        type_is_a_collection = True
        if python_type_for_type is not None:
            python_type_for_type = api.named_type(
                names.NAMED_TYPE_BUILTINS_LIST, [python_type_for_type]
            )
    elif (
        uselist_arg is None or api.parse_bool(uselist_arg) is True
    ) and collection_cls_arg is not None:
        type_is_a_collection = True
        if isinstance(collection_cls_arg, CallExpr):
            collection_cls_arg = collection_cls_arg.callee

        if isinstance(collection_cls_arg, NameExpr) and isinstance(
            collection_cls_arg.node, TypeInfo
        ):
            if python_type_for_type is not None:
                # this can still be overridden by the left hand side
                # within _infer_Type_from_left_and_inferred_right
                python_type_for_type = Instance(
                    collection_cls_arg.node, [python_type_for_type]
                )
        elif (
            isinstance(collection_cls_arg, NameExpr)
            and isinstance(collection_cls_arg.node, FuncDef)
            and collection_cls_arg.node.type is not None
        ):
            if python_type_for_type is not None:
                # this can still be overridden by the left hand side
                # within _infer_Type_from_left_and_inferred_right

                # TODO: handle mypy.types.Overloaded
                if isinstance(collection_cls_arg.node.type, CallableType):
                    rt = get_proper_type(collection_cls_arg.node.type.ret_type)

                    if isinstance(rt, CallableType):
                        callable_ret_type = get_proper_type(rt.ret_type)
                        if isinstance(callable_ret_type, Instance):
                            python_type_for_type = Instance(
                                callable_ret_type.type,
                                [python_type_for_type],
                            )
        else:
            util.fail(
                api,
                "Expected Python collection type for "
                "collection_class parameter",
                stmt.rvalue,
            )
            python_type_for_type = None
    elif uselist_arg is not None and api.parse_bool(uselist_arg) is False:
        if collection_cls_arg is not None:
            util.fail(
                api,
                "Sending uselist=False and collection_class at the same time "
                "does not make sense",
                stmt.rvalue,
            )
        if python_type_for_type is not None:
            python_type_for_type = UnionType(
                [python_type_for_type, NoneType()]
            )

    else:
        if left_hand_explicit_type is None:
            msg = (
                "Can't infer scalar or collection for ORM mapped expression "
                "assigned to attribute '{}' if both 'uselist' and "
                "'collection_class' arguments are absent from the "
                "relationship(); please specify a "
                "type annotation on the left hand side."
            )
            util.fail(api, msg.format(node.name), node)

    if python_type_for_type is None:
        return infer_type_from_left_hand_type_only(
            api, node, left_hand_explicit_type
        )
    elif left_hand_explicit_type is not None:
        if type_is_a_collection:
            assert isinstance(left_hand_explicit_type, Instance)
            assert isinstance(python_type_for_type, Instance)
            return _infer_collection_type_from_left_and_inferred_right(
                api, node, left_hand_explicit_type, python_type_for_type
            )
        else:
            return _infer_type_from_left_and_inferred_right(
                api,
                node,
                left_hand_explicit_type,
                python_type_for_type,
            )
    else:
        return python_type_for_type


def _infer_type_from_decl_composite_property(
    api: SemanticAnalyzerPluginInterface,
    stmt: AssignmentStmt,
    node: Var,
    left_hand_explicit_type: Optional[ProperType],
) -> Optional[ProperType]:
    """Infer the type of mapping from a Composite."""

    assert isinstance(stmt.rvalue, CallExpr)
    target_cls_arg = stmt.rvalue.args[0]
    python_type_for_type = None

    if isinstance(target_cls_arg, NameExpr) and isinstance(
        target_cls_arg.node, TypeInfo
    ):
        related_object_type = target_cls_arg.node
        python_type_for_type = Instance(related_object_type, [])
    else:
        python_type_for_type = None

    if python_type_for_type is None:
        return infer_type_from_left_hand_type_only(
            api, node, left_hand_explicit_type
        )
    elif left_hand_explicit_type is not None:
        return _infer_type_from_left_and_inferred_right(
            api, node, left_hand_explicit_type, python_type_for_type
        )
    else:
        return python_type_for_type


def _infer_type_from_mapped(
    api: SemanticAnalyzerPluginInterface,
    stmt: AssignmentStmt,
    node: Var,
    left_hand_explicit_type: Optional[ProperType],
    infer_from_right_side: RefExpr,
) -> Optional[ProperType]:
    """Infer the type of mapping from a right side expression
    that returns Mapped.


    """
    assert isinstance(stmt.rvalue, CallExpr)

    # (Pdb) print(stmt.rvalue.callee)
    # NameExpr(query_expression [sqlalchemy.orm._orm_constructors.query_expression])  # noqa: E501
    # (Pdb) stmt.rvalue.callee.node
    # <mypy.nodes.FuncDef object at 0x7f8d92fb5940>
    # (Pdb) stmt.rvalue.callee.node.type
    # def [_T] (default_expr: sqlalchemy.sql.elements.ColumnElement[_T`-1] =) -> sqlalchemy.orm.base.Mapped[_T`-1]  # noqa: E501
    # sqlalchemy.orm.base.Mapped[_T`-1]
    # the_mapped_type = stmt.rvalue.callee.node.type.ret_type

    # TODO: look at generic ref and either use that,
    # or reconcile w/ what's present, etc.
    the_mapped_type = util.type_for_callee(infer_from_right_side)  # noqa

    return infer_type_from_left_hand_type_only(
        api, node, left_hand_explicit_type
    )


def _infer_type_from_decl_column_property(
    api: SemanticAnalyzerPluginInterface,
    stmt: AssignmentStmt,
    node: Var,
    left_hand_explicit_type: Optional[ProperType],
) -> Optional[ProperType]:
    """Infer the type of mapping from a ColumnProperty.

    This includes mappings against ``column_property()`` as well as the
    ``deferred()`` function.

    """
    assert isinstance(stmt.rvalue, CallExpr)

    if stmt.rvalue.args:
        first_prop_arg = stmt.rvalue.args[0]

        if isinstance(first_prop_arg, CallExpr):
            type_id = names.type_id_for_callee(first_prop_arg.callee)

            # look for column_property() / deferred() etc with Column as first
            # argument
            if type_id is names.COLUMN:
                return _infer_type_from_decl_column(
                    api,
                    stmt,
                    node,
                    left_hand_explicit_type,
                    right_hand_expression=first_prop_arg,
                )

    if isinstance(stmt.rvalue, CallExpr):
        type_id = names.type_id_for_callee(stmt.rvalue.callee)
        # this is probably not strictly necessary as we have to use the left
        # hand type for query expression in any case.  any other no-arg
        # column prop objects would go here also
        if type_id is names.QUERY_EXPRESSION:
            return _infer_type_from_decl_column(
                api,
                stmt,
                node,
                left_hand_explicit_type,
            )

    return infer_type_from_left_hand_type_only(
        api, node, left_hand_explicit_type
    )


def _infer_type_from_decl_column(
    api: SemanticAnalyzerPluginInterface,
    stmt: AssignmentStmt,
    node: Var,
    left_hand_explicit_type: Optional[ProperType],
    right_hand_expression: Optional[CallExpr] = None,
) -> Optional[ProperType]:
    """Infer the type of mapping from a Column.

    E.g.::

        @reg.mapped
        class MyClass:
            # ...

            a = Column(Integer)

            b = Column("b", String)

            c: Mapped[int] = Column(Integer)

            d: bool = Column(Boolean)

    Will resolve in MyPy as::

        @reg.mapped
        class MyClass:
            # ...

            a: Mapped[int]

            b: Mapped[str]

            c: Mapped[int]

            d: Mapped[bool]

    """
    assert isinstance(node, Var)

    callee = None

    if right_hand_expression is None:
        if not isinstance(stmt.rvalue, CallExpr):
            return None

        right_hand_expression = stmt.rvalue

    for column_arg in right_hand_expression.args[0:2]:
        if isinstance(column_arg, CallExpr):
            if isinstance(column_arg.callee, RefExpr):
                # x = Column(String(50))
                callee = column_arg.callee
                type_args: Sequence[Expression] = column_arg.args
                break
        elif isinstance(column_arg, (NameExpr, MemberExpr)):
            if isinstance(column_arg.node, TypeInfo):
                # x = Column(String)
                callee = column_arg
                type_args = ()
                break
            else:
                # x = Column(some_name, String), go to next argument
                continue
        elif isinstance(column_arg, (StrExpr,)):
            # x = Column("name", String), go to next argument
            continue
        elif isinstance(column_arg, (LambdaExpr,)):
            # x = Column("name", String, default=lambda: uuid.uuid4())
            # go to next argument
            continue
        else:
            assert False

    if callee is None:
        return None

    if isinstance(callee.node, TypeInfo) and names.mro_has_id(
        callee.node.mro, names.TYPEENGINE
    ):
        python_type_for_type = extract_python_type_from_typeengine(
            api, callee.node, type_args
        )

        if left_hand_explicit_type is not None:
            return _infer_type_from_left_and_inferred_right(
                api, node, left_hand_explicit_type, python_type_for_type
            )

        else:
            return UnionType([python_type_for_type, NoneType()])
    else:
        # it's not TypeEngine, it's typically implicitly typed
        # like ForeignKey.  we can't infer from the right side.
        return infer_type_from_left_hand_type_only(
            api, node, left_hand_explicit_type
        )


def _infer_type_from_left_and_inferred_right(
    api: SemanticAnalyzerPluginInterface,
    node: Var,
    left_hand_explicit_type: ProperType,
    python_type_for_type: ProperType,
    orig_left_hand_type: Optional[ProperType] = None,
    orig_python_type_for_type: Optional[ProperType] = None,
) -> Optional[ProperType]:
    """Validate type when a left hand annotation is present and we also
    could infer the right hand side::

        attrname: SomeType = Column(SomeDBType)

    """

    if orig_left_hand_type is None:
        orig_left_hand_type = left_hand_explicit_type
    if orig_python_type_for_type is None:
        orig_python_type_for_type = python_type_for_type

    if not is_subtype(left_hand_explicit_type, python_type_for_type):
        effective_type = api.named_type(
            names.NAMED_TYPE_SQLA_MAPPED, [orig_python_type_for_type]
        )

        msg = (
            "Left hand assignment '{}: {}' not compatible "
            "with ORM mapped expression of type {}"
        )
        util.fail(
            api,
            msg.format(
                node.name,
                util.format_type(orig_left_hand_type, api.options),
                util.format_type(effective_type, api.options),
            ),
            node,
        )

    return orig_left_hand_type


def _infer_collection_type_from_left_and_inferred_right(
    api: SemanticAnalyzerPluginInterface,
    node: Var,
    left_hand_explicit_type: Instance,
    python_type_for_type: Instance,
) -> Optional[ProperType]:
    orig_left_hand_type = left_hand_explicit_type
    orig_python_type_for_type = python_type_for_type

    if left_hand_explicit_type.args:
        left_hand_arg = get_proper_type(left_hand_explicit_type.args[0])
        python_type_arg = get_proper_type(python_type_for_type.args[0])
    else:
        left_hand_arg = left_hand_explicit_type
        python_type_arg = python_type_for_type

    assert isinstance(left_hand_arg, (Instance, UnionType))
    assert isinstance(python_type_arg, (Instance, UnionType))

    return _infer_type_from_left_and_inferred_right(
        api,
        node,
        left_hand_arg,
        python_type_arg,
        orig_left_hand_type=orig_left_hand_type,
        orig_python_type_for_type=orig_python_type_for_type,
    )


def infer_type_from_left_hand_type_only(
    api: SemanticAnalyzerPluginInterface,
    node: Var,
    left_hand_explicit_type: Optional[ProperType],
) -> Optional[ProperType]:
    """Determine the type based on explicit annotation only.

    if no annotation were present, note that we need one there to know
    the type.

    """
    if left_hand_explicit_type is None:
        msg = (
            "Can't infer type from ORM mapped expression "
            "assigned to attribute '{}'; please specify a "
            "Python type or "
            "Mapped[<python type>] on the left hand side."
        )
        util.fail(api, msg.format(node.name), node)

        return api.named_type(
            names.NAMED_TYPE_SQLA_MAPPED, [AnyType(TypeOfAny.special_form)]
        )

    else:
        # use type from the left hand side
        return left_hand_explicit_type


def extract_python_type_from_typeengine(
    api: SemanticAnalyzerPluginInterface,
    node: TypeInfo,
    type_args: Sequence[Expression],
) -> ProperType:
    if node.fullname == "sqlalchemy.sql.sqltypes.Enum" and type_args:
        first_arg = type_args[0]
        if isinstance(first_arg, RefExpr) and isinstance(
            first_arg.node, TypeInfo
        ):
            for base_ in first_arg.node.mro:
                if base_.fullname == "enum.Enum":
                    return Instance(first_arg.node, [])
            # TODO: support other pep-435 types here
        else:
            return api.named_type(names.NAMED_TYPE_BUILTINS_STR, [])

    assert node.has_base("sqlalchemy.sql.type_api.TypeEngine"), (
        "could not extract Python type from node: %s" % node
    )

    type_engine_sym = api.lookup_fully_qualified_or_none(
        "sqlalchemy.sql.type_api.TypeEngine"
    )

    assert type_engine_sym is not None and isinstance(
        type_engine_sym.node, TypeInfo
    )
    type_engine = map_instance_to_supertype(
        Instance(node, []),
        type_engine_sym.node,
    )
    return get_proper_type(type_engine.args[-1])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\units\dimensions.py ===
"""
Definition of physical dimensions.

Unit systems will be constructed on top of these dimensions.

Most of the examples in the doc use MKS system and are presented from the
computer point of view: from a human point, adding length to time is not legal
in MKS but it is in natural system; for a computer in natural system there is
no time dimension (but a velocity dimension instead) - in the basis - so the
question of adding time to length has no meaning.
"""

from __future__ import annotations

import collections
from functools import reduce

from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.matrices.dense import Matrix
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.core.expr import Expr
from sympy.core.power import Pow


class _QuantityMapper:

    _quantity_scale_factors_global: dict[Expr, Expr] = {}
    _quantity_dimensional_equivalence_map_global: dict[Expr, Expr] = {}
    _quantity_dimension_global: dict[Expr, Expr] = {}

    def __init__(self, *args, **kwargs):
        self._quantity_dimension_map = {}
        self._quantity_scale_factors = {}

    def set_quantity_dimension(self, quantity, dimension):
        """
        Set the dimension for the quantity in a unit system.

        If this relation is valid in every unit system, use
        ``quantity.set_global_dimension(dimension)`` instead.
        """
        from sympy.physics.units import Quantity
        dimension = sympify(dimension)
        if not isinstance(dimension, Dimension):
            if dimension == 1:
                dimension = Dimension(1)
            else:
                raise ValueError("expected dimension or 1")
        elif isinstance(dimension, Quantity):
            dimension = self.get_quantity_dimension(dimension)
        self._quantity_dimension_map[quantity] = dimension

    def set_quantity_scale_factor(self, quantity, scale_factor):
        """
        Set the scale factor of a quantity relative to another quantity.

        It should be used only once per quantity to just one other quantity,
        the algorithm will then be able to compute the scale factors to all
        other quantities.

        In case the scale factor is valid in every unit system, please use
        ``quantity.set_global_relative_scale_factor(scale_factor)`` instead.
        """
        from sympy.physics.units import Quantity
        from sympy.physics.units.prefixes import Prefix
        scale_factor = sympify(scale_factor)
        # replace all prefixes by their ratio to canonical units:
        scale_factor = scale_factor.replace(
            lambda x: isinstance(x, Prefix),
            lambda x: x.scale_factor
        )
        # replace all quantities by their ratio to canonical units:
        scale_factor = scale_factor.replace(
            lambda x: isinstance(x, Quantity),
            lambda x: self.get_quantity_scale_factor(x)
        )
        self._quantity_scale_factors[quantity] = scale_factor

    def get_quantity_dimension(self, unit):
        from sympy.physics.units import Quantity
        # First look-up the local dimension map, then the global one:
        if unit in self._quantity_dimension_map:
            return self._quantity_dimension_map[unit]
        if unit in self._quantity_dimension_global:
            return self._quantity_dimension_global[unit]
        if unit in self._quantity_dimensional_equivalence_map_global:
            dep_unit = self._quantity_dimensional_equivalence_map_global[unit]
            if isinstance(dep_unit, Quantity):
                return self.get_quantity_dimension(dep_unit)
            else:
                return Dimension(self.get_dimensional_expr(dep_unit))
        if isinstance(unit, Quantity):
            return Dimension(unit.name)
        else:
            return Dimension(1)

    def get_quantity_scale_factor(self, unit):
        if unit in self._quantity_scale_factors:
            return self._quantity_scale_factors[unit]
        if unit in self._quantity_scale_factors_global:
            mul_factor, other_unit = self._quantity_scale_factors_global[unit]
            return mul_factor*self.get_quantity_scale_factor(other_unit)
        return S.One


class Dimension(Expr):
    """
    This class represent the dimension of a physical quantities.

    The ``Dimension`` constructor takes as parameters a name and an optional
    symbol.

    For example, in classical mechanics we know that time is different from
    temperature and dimensions make this difference (but they do not provide
    any measure of these quantities.

        >>> from sympy.physics.units import Dimension
        >>> length = Dimension('length')
        >>> length
        Dimension(length)
        >>> time = Dimension('time')
        >>> time
        Dimension(time)

    Dimensions can be composed using multiplication, division and
    exponentiation (by a number) to give new dimensions. Addition and
    subtraction is defined only when the two objects are the same dimension.

        >>> velocity = length / time
        >>> velocity
        Dimension(length/time)

    It is possible to use a dimension system object to get the dimensionsal
    dependencies of a dimension, for example the dimension system used by the
    SI units convention can be used:

        >>> from sympy.physics.units.systems.si import dimsys_SI
        >>> dimsys_SI.get_dimensional_dependencies(velocity)
        {Dimension(length, L): 1, Dimension(time, T): -1}
        >>> length + length
        Dimension(length)
        >>> l2 = length**2
        >>> l2
        Dimension(length**2)
        >>> dimsys_SI.get_dimensional_dependencies(l2)
        {Dimension(length, L): 2}

    """

    _op_priority = 13.0

    # XXX: This doesn't seem to be used anywhere...
    _dimensional_dependencies = {}  # type: ignore

    is_commutative = True
    is_number = False
    # make sqrt(M**2) --> M
    is_positive = True
    is_real = True

    def __new__(cls, name, symbol=None):

        if isinstance(name, str):
            name = Symbol(name)
        else:
            name = sympify(name)

        if not isinstance(name, Expr):
            raise TypeError("Dimension name needs to be a valid math expression")

        if isinstance(symbol, str):
            symbol = Symbol(symbol)
        elif symbol is not None:
            assert isinstance(symbol, Symbol)

        obj = Expr.__new__(cls, name)

        obj._name = name
        obj._symbol = symbol
        return obj

    @property
    def name(self):
        return self._name

    @property
    def symbol(self):
        return self._symbol

    def __str__(self):
        """
        Display the string representation of the dimension.
        """
        if self.symbol is None:
            return "Dimension(%s)" % (self.name)
        else:
            return "Dimension(%s, %s)" % (self.name, self.symbol)

    def __repr__(self):
        return self.__str__()

    def __neg__(self):
        return self

    def __add__(self, other):
        from sympy.physics.units.quantities import Quantity
        other = sympify(other)
        if isinstance(other, Basic):
            if other.has(Quantity):
                raise TypeError("cannot sum dimension and quantity")
            if isinstance(other, Dimension) and self == other:
                return self
            return super().__add__(other)
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # there is no notion of ordering (or magnitude) among dimension,
        # subtraction is equivalent to addition when the operation is legal
        return self + other

    def __rsub__(self, other):
        # there is no notion of ordering (or magnitude) among dimension,
        # subtraction is equivalent to addition when the operation is legal
        return self + other

    def __pow__(self, other):
        return self._eval_power(other)

    def _eval_power(self, other):
        other = sympify(other)
        return Dimension(self.name**other)

    def __mul__(self, other):
        from sympy.physics.units.quantities import Quantity
        if isinstance(other, Basic):
            if other.has(Quantity):
                raise TypeError("cannot sum dimension and quantity")
            if isinstance(other, Dimension):
                return Dimension(self.name*other.name)
            if not other.free_symbols:  # other.is_number cannot be used
                return self
            return super().__mul__(other)
        return self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self*Pow(other, -1)

    def __rtruediv__(self, other):
        return other * pow(self, -1)

    @classmethod
    def _from_dimensional_dependencies(cls, dependencies):
        return reduce(lambda x, y: x * y, (
            d**e for d, e in dependencies.items()
        ), 1)

    def has_integer_powers(self, dim_sys):
        """
        Check if the dimension object has only integer powers.

        All the dimension powers should be integers, but rational powers may
        appear in intermediate steps. This method may be used to check that the
        final result is well-defined.
        """

        return all(dpow.is_Integer for dpow in dim_sys.get_dimensional_dependencies(self).values())


# Create dimensions according to the base units in MKSA.
# For other unit systems, they can be derived by transforming the base
# dimensional dependency dictionary.


class DimensionSystem(Basic, _QuantityMapper):
    r"""
    DimensionSystem represents a coherent set of dimensions.

    The constructor takes three parameters:

    - base dimensions;
    - derived dimensions: these are defined in terms of the base dimensions
      (for example velocity is defined from the division of length by time);
    - dependency of dimensions: how the derived dimensions depend
      on the base dimensions.

    Optionally either the ``derived_dims`` or the ``dimensional_dependencies``
    may be omitted.
    """

    def __new__(cls, base_dims, derived_dims=(), dimensional_dependencies={}):
        dimensional_dependencies = dict(dimensional_dependencies)

        def parse_dim(dim):
            if isinstance(dim, str):
                dim = Dimension(Symbol(dim))
            elif isinstance(dim, Dimension):
                pass
            elif isinstance(dim, Symbol):
                dim = Dimension(dim)
            else:
                raise TypeError("%s wrong type" % dim)
            return dim

        base_dims = [parse_dim(i) for i in base_dims]
        derived_dims = [parse_dim(i) for i in derived_dims]

        for dim in base_dims:
            if (dim in dimensional_dependencies
                and (len(dimensional_dependencies[dim]) != 1 or
                dimensional_dependencies[dim].get(dim, None) != 1)):
                raise IndexError("Repeated value in base dimensions")
            dimensional_dependencies[dim] = Dict({dim: 1})

        def parse_dim_name(dim):
            if isinstance(dim, Dimension):
                return dim
            elif isinstance(dim, str):
                return Dimension(Symbol(dim))
            elif isinstance(dim, Symbol):
                return Dimension(dim)
            else:
                raise TypeError("unrecognized type %s for %s" % (type(dim), dim))

        for dim in dimensional_dependencies.keys():
            dim = parse_dim(dim)
            if (dim not in derived_dims) and (dim not in base_dims):
                derived_dims.append(dim)

        def parse_dict(d):
            return Dict({parse_dim_name(i): j for i, j in d.items()})

        # Make sure everything is a SymPy type:
        dimensional_dependencies = {parse_dim_name(i): parse_dict(j) for i, j in
                                    dimensional_dependencies.items()}

        for dim in derived_dims:
            if dim in base_dims:
                raise ValueError("Dimension %s both in base and derived" % dim)
            if dim not in dimensional_dependencies:
                # TODO: should this raise a warning?
                dimensional_dependencies[dim] = Dict({dim: 1})

        base_dims.sort(key=default_sort_key)
        derived_dims.sort(key=default_sort_key)

        base_dims = Tuple(*base_dims)
        derived_dims = Tuple(*derived_dims)
        dimensional_dependencies = Dict({i: Dict(j) for i, j in dimensional_dependencies.items()})
        obj = Basic.__new__(cls, base_dims, derived_dims, dimensional_dependencies)
        return obj

    @property
    def base_dims(self):
        return self.args[0]

    @property
    def derived_dims(self):
        return self.args[1]

    @property
    def dimensional_dependencies(self):
        return self.args[2]

    def _get_dimensional_dependencies_for_name(self, dimension):
        if isinstance(dimension, str):
            dimension = Dimension(Symbol(dimension))
        elif not isinstance(dimension, Dimension):
            dimension = Dimension(dimension)

        if dimension.name.is_Symbol:
            # Dimensions not included in the dependencies are considered
            # as base dimensions:
            return dict(self.dimensional_dependencies.get(dimension, {dimension: 1}))

        if dimension.name.is_number or dimension.name.is_NumberSymbol:
            return {}

        get_for_name = self._get_dimensional_dependencies_for_name

        if dimension.name.is_Mul:
            ret = collections.defaultdict(int)
            dicts = [get_for_name(i) for i in dimension.name.args]
            for d in dicts:
                for k, v in d.items():
                    ret[k] += v
            return {k: v for (k, v) in ret.items() if v != 0}

        if dimension.name.is_Add:
            dicts = [get_for_name(i) for i in dimension.name.args]
            if all(d == dicts[0] for d in dicts[1:]):
                return dicts[0]
            raise TypeError("Only equivalent dimensions can be added or subtracted.")

        if dimension.name.is_Pow:
            dim_base = get_for_name(dimension.name.base)
            dim_exp = get_for_name(dimension.name.exp)
            if dim_exp == {} or dimension.name.exp.is_Symbol:
                return {k: v * dimension.name.exp for (k, v) in dim_base.items()}
            else:
                raise TypeError("The exponent for the power operator must be a Symbol or dimensionless.")

        if dimension.name.is_Function:
            args = (Dimension._from_dimensional_dependencies(
                get_for_name(arg)) for arg in dimension.name.args)
            result = dimension.name.func(*args)

            dicts = [get_for_name(i) for i in dimension.name.args]

            if isinstance(result, Dimension):
                return self.get_dimensional_dependencies(result)
            elif result.func == dimension.name.func:
                if isinstance(dimension.name, TrigonometricFunction):
                    if dicts[0] in ({}, {Dimension('angle'): 1}):
                        return {}
                    else:
                        raise TypeError("The input argument for the function {} must be dimensionless or have dimensions of angle.".format(dimension.func))
                else:
                    if all(item == {} for item in dicts):
                        return {}
                    else:
                        raise TypeError("The input arguments for the function {} must be dimensionless.".format(dimension.func))
            else:
                return get_for_name(result)

        raise TypeError("Type {} not implemented for get_dimensional_dependencies".format(type(dimension.name)))

    def get_dimensional_dependencies(self, name, mark_dimensionless=False):
        dimdep = self._get_dimensional_dependencies_for_name(name)
        if mark_dimensionless and dimdep == {}:
            return {Dimension(1): 1}
        return dict(dimdep.items())

    def equivalent_dims(self, dim1, dim2):
        deps1 = self.get_dimensional_dependencies(dim1)
        deps2 = self.get_dimensional_dependencies(dim2)
        return deps1 == deps2

    def extend(self, new_base_dims, new_derived_dims=(), new_dim_deps=None):
        deps = dict(self.dimensional_dependencies)
        if new_dim_deps:
            deps.update(new_dim_deps)

        new_dim_sys = DimensionSystem(
            tuple(self.base_dims) + tuple(new_base_dims),
            tuple(self.derived_dims) + tuple(new_derived_dims),
            deps
        )
        new_dim_sys._quantity_dimension_map.update(self._quantity_dimension_map)
        new_dim_sys._quantity_scale_factors.update(self._quantity_scale_factors)
        return new_dim_sys

    def is_dimensionless(self, dimension):
        """
        Check if the dimension object really has a dimension.

        A dimension should have at least one component with non-zero power.
        """
        if dimension.name == 1:
            return True
        return self.get_dimensional_dependencies(dimension) == {}

    @property
    def list_can_dims(self):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        List all canonical dimension names.
        """
        dimset = set()
        for i in self.base_dims:
            dimset.update(set(self.get_dimensional_dependencies(i).keys()))
        return tuple(sorted(dimset, key=str))

    @property
    def inv_can_transf_matrix(self):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Compute the inverse transformation matrix from the base to the
        canonical dimension basis.

        It corresponds to the matrix where columns are the vector of base
        dimensions in canonical basis.

        This matrix will almost never be used because dimensions are always
        defined with respect to the canonical basis, so no work has to be done
        to get them in this basis. Nonetheless if this matrix is not square
        (or not invertible) it means that we have chosen a bad basis.
        """
        matrix = reduce(lambda x, y: x.row_join(y),
                        [self.dim_can_vector(d) for d in self.base_dims])
        return matrix

    @property
    def can_transf_matrix(self):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Return the canonical transformation matrix from the canonical to the
        base dimension basis.

        It is the inverse of the matrix computed with inv_can_transf_matrix().
        """

        #TODO: the inversion will fail if the system is inconsistent, for
        #      example if the matrix is not a square
        return reduce(lambda x, y: x.row_join(y),
                      [self.dim_can_vector(d) for d in sorted(self.base_dims, key=str)]
                      ).inv()

    def dim_can_vector(self, dim):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Dimensional representation in terms of the canonical base dimensions.
        """

        vec = []
        for d in self.list_can_dims:
            vec.append(self.get_dimensional_dependencies(dim).get(d, 0))
        return Matrix(vec)

    def dim_vector(self, dim):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.


        Vector representation in terms of the base dimensions.
        """
        return self.can_transf_matrix * Matrix(self.dim_can_vector(dim))

    def print_dim_base(self, dim):
        """
        Give the string expression of a dimension in term of the basis symbols.
        """
        dims = self.dim_vector(dim)
        symbols = [i.symbol if i.symbol is not None else i.name for i in self.base_dims]
        res = S.One
        for (s, p) in zip(symbols, dims):
            res *= s**p
        return res

    @property
    def dim(self):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Give the dimension of the system.

        That is return the number of dimensions forming the basis.
        """
        return len(self.base_dims)

    @property
    def is_consistent(self):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Check if the system is well defined.
        """

        # not enough or too many base dimensions compared to independent
        # dimensions
        # in vector language: the set of vectors do not form a basis
        return self.inv_can_transf_matrix.is_square

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\socket.py ===
from __future__ import annotations

# This is a public namespace, so we don't want to expose any non-underscored
# attributes that aren't actually part of our public API. But it's very
# annoying to carefully always use underscored names for module-level
# temporaries, imports, etc. when implementing the module. So we put the
# implementation in an underscored module, and then re-export the public parts
# here.
# We still have some underscore names though but only a few.
import socket as _stdlib_socket

# static checkers don't understand if importing this as _sys, so it's deleted later
import sys
import typing as _t

from . import _socket

_bad_symbols: set[str] = set()
if sys.platform == "win32":
    # See https://github.com/python-trio/trio/issues/39
    # Do not import for windows platform
    # (you can still get it from stdlib socket, of course, if you want it)
    _bad_symbols.add("SO_REUSEADDR")

# Dynamically re-export whatever constants this particular Python happens to
# have:
globals().update(
    {
        _name: getattr(_stdlib_socket, _name)
        for _name in _stdlib_socket.__all__
        if _name.isupper() and _name not in _bad_symbols
    },
)

# import the overwrites
from contextlib import suppress as _suppress

# Uses `from x import y as y` for compatibility with `pyright --verifytypes` (#2625)
from ._socket import (
    SocketType as SocketType,
    from_stdlib_socket as from_stdlib_socket,
    fromfd as fromfd,
    getaddrinfo as getaddrinfo,
    getnameinfo as getnameinfo,
    getprotobyname as getprotobyname,
    set_custom_hostname_resolver as set_custom_hostname_resolver,
    set_custom_socket_factory as set_custom_socket_factory,
    socket as socket,
    socketpair as socketpair,
)

# not always available so expose only if
if sys.platform == "win32" or not _t.TYPE_CHECKING:
    with _suppress(ImportError):
        from ._socket import fromshare as fromshare

# expose these functions to trio.socket
from socket import (
    gaierror as gaierror,
    gethostname as gethostname,
    herror as herror,
    htonl as htonl,
    htons as htons,
    inet_aton as inet_aton,
    inet_ntoa as inet_ntoa,
    inet_ntop as inet_ntop,
    inet_pton as inet_pton,
    ntohs as ntohs,
)

if sys.implementation.name == "cpython":
    from socket import (
        if_indextoname as if_indextoname,
        if_nametoindex as if_nametoindex,
    )

    # For android devices, if_nameindex support was introduced in API 24,
    # so it doesn't exist for any version prior.
    with _suppress(ImportError):
        from socket import (
            if_nameindex as if_nameindex,
        )


# not always available so expose only if
if sys.platform != "win32" or not _t.TYPE_CHECKING:
    with _suppress(ImportError):
        from socket import (
            sethostname as sethostname,
        )

if _t.TYPE_CHECKING:
    IP_BIND_ADDRESS_NO_PORT: int
else:
    try:
        IP_BIND_ADDRESS_NO_PORT  # noqa: B018  # "useless expression"
    except NameError:
        if sys.platform == "linux":
            IP_BIND_ADDRESS_NO_PORT = 24

del sys


# The socket module exports a bunch of platform-specific constants. We want to
# re-export them. Since the exact set of constants varies depending on Python
# version, platform, the libc installed on the system where Python was built,
# etc., we figure out which constants to re-export dynamically at runtime (see
# above). But that confuses static analysis tools like jedi and mypy. So this
# import statement statically lists every constant that *could* be
# exported. There's a test in test_exports.py to make sure that the list is
# kept up to date.
if _t.TYPE_CHECKING:
    from socket import (  # type: ignore[attr-defined]
        AF_ALG as AF_ALG,
        AF_APPLETALK as AF_APPLETALK,
        AF_ASH as AF_ASH,
        AF_ATMPVC as AF_ATMPVC,
        AF_ATMSVC as AF_ATMSVC,
        AF_AX25 as AF_AX25,
        AF_BLUETOOTH as AF_BLUETOOTH,
        AF_BRIDGE as AF_BRIDGE,
        AF_CAN as AF_CAN,
        AF_ECONET as AF_ECONET,
        AF_HYPERV as AF_HYPERV,
        AF_INET as AF_INET,
        AF_INET6 as AF_INET6,
        AF_IPX as AF_IPX,
        AF_IRDA as AF_IRDA,
        AF_KEY as AF_KEY,
        AF_LINK as AF_LINK,
        AF_LLC as AF_LLC,
        AF_NETBEUI as AF_NETBEUI,
        AF_NETLINK as AF_NETLINK,
        AF_NETROM as AF_NETROM,
        AF_PACKET as AF_PACKET,
        AF_PPPOX as AF_PPPOX,
        AF_QIPCRTR as AF_QIPCRTR,
        AF_RDS as AF_RDS,
        AF_ROSE as AF_ROSE,
        AF_ROUTE as AF_ROUTE,
        AF_SECURITY as AF_SECURITY,
        AF_SNA as AF_SNA,
        AF_SYSTEM as AF_SYSTEM,
        AF_TIPC as AF_TIPC,
        AF_UNIX as AF_UNIX,
        AF_UNSPEC as AF_UNSPEC,
        AF_VSOCK as AF_VSOCK,
        AF_WANPIPE as AF_WANPIPE,
        AF_X25 as AF_X25,
        AI_ADDRCONFIG as AI_ADDRCONFIG,
        AI_ALL as AI_ALL,
        AI_CANONNAME as AI_CANONNAME,
        AI_DEFAULT as AI_DEFAULT,
        AI_MASK as AI_MASK,
        AI_NUMERICHOST as AI_NUMERICHOST,
        AI_NUMERICSERV as AI_NUMERICSERV,
        AI_PASSIVE as AI_PASSIVE,
        AI_V4MAPPED as AI_V4MAPPED,
        AI_V4MAPPED_CFG as AI_V4MAPPED_CFG,
        ALG_OP_DECRYPT as ALG_OP_DECRYPT,
        ALG_OP_ENCRYPT as ALG_OP_ENCRYPT,
        ALG_OP_SIGN as ALG_OP_SIGN,
        ALG_OP_VERIFY as ALG_OP_VERIFY,
        ALG_SET_AEAD_ASSOCLEN as ALG_SET_AEAD_ASSOCLEN,
        ALG_SET_AEAD_AUTHSIZE as ALG_SET_AEAD_AUTHSIZE,
        ALG_SET_IV as ALG_SET_IV,
        ALG_SET_KEY as ALG_SET_KEY,
        ALG_SET_OP as ALG_SET_OP,
        ALG_SET_PUBKEY as ALG_SET_PUBKEY,
        BDADDR_ANY as BDADDR_ANY,
        BDADDR_LOCAL as BDADDR_LOCAL,
        BTPROTO_HCI as BTPROTO_HCI,
        BTPROTO_L2CAP as BTPROTO_L2CAP,
        BTPROTO_RFCOMM as BTPROTO_RFCOMM,
        BTPROTO_SCO as BTPROTO_SCO,
        CAN_BCM as CAN_BCM,
        CAN_BCM_CAN_FD_FRAME as CAN_BCM_CAN_FD_FRAME,
        CAN_BCM_RX_ANNOUNCE_RESUME as CAN_BCM_RX_ANNOUNCE_RESUME,
        CAN_BCM_RX_CHANGED as CAN_BCM_RX_CHANGED,
        CAN_BCM_RX_CHECK_DLC as CAN_BCM_RX_CHECK_DLC,
        CAN_BCM_RX_DELETE as CAN_BCM_RX_DELETE,
        CAN_BCM_RX_FILTER_ID as CAN_BCM_RX_FILTER_ID,
        CAN_BCM_RX_NO_AUTOTIMER as CAN_BCM_RX_NO_AUTOTIMER,
        CAN_BCM_RX_READ as CAN_BCM_RX_READ,
        CAN_BCM_RX_RTR_FRAME as CAN_BCM_RX_RTR_FRAME,
        CAN_BCM_RX_SETUP as CAN_BCM_RX_SETUP,
        CAN_BCM_RX_STATUS as CAN_BCM_RX_STATUS,
        CAN_BCM_RX_TIMEOUT as CAN_BCM_RX_TIMEOUT,
        CAN_BCM_SETTIMER as CAN_BCM_SETTIMER,
        CAN_BCM_STARTTIMER as CAN_BCM_STARTTIMER,
        CAN_BCM_TX_ANNOUNCE as CAN_BCM_TX_ANNOUNCE,
        CAN_BCM_TX_COUNTEVT as CAN_BCM_TX_COUNTEVT,
        CAN_BCM_TX_CP_CAN_ID as CAN_BCM_TX_CP_CAN_ID,
        CAN_BCM_TX_DELETE as CAN_BCM_TX_DELETE,
        CAN_BCM_TX_EXPIRED as CAN_BCM_TX_EXPIRED,
        CAN_BCM_TX_READ as CAN_BCM_TX_READ,
        CAN_BCM_TX_RESET_MULTI_IDX as CAN_BCM_TX_RESET_MULTI_IDX,
        CAN_BCM_TX_SEND as CAN_BCM_TX_SEND,
        CAN_BCM_TX_SETUP as CAN_BCM_TX_SETUP,
        CAN_BCM_TX_STATUS as CAN_BCM_TX_STATUS,
        CAN_EFF_FLAG as CAN_EFF_FLAG,
        CAN_EFF_MASK as CAN_EFF_MASK,
        CAN_ERR_FLAG as CAN_ERR_FLAG,
        CAN_ERR_MASK as CAN_ERR_MASK,
        CAN_ISOTP as CAN_ISOTP,
        CAN_J1939 as CAN_J1939,
        CAN_RAW as CAN_RAW,
        CAN_RAW_ERR_FILTER as CAN_RAW_ERR_FILTER,
        CAN_RAW_FD_FRAMES as CAN_RAW_FD_FRAMES,
        CAN_RAW_FILTER as CAN_RAW_FILTER,
        CAN_RAW_JOIN_FILTERS as CAN_RAW_JOIN_FILTERS,
        CAN_RAW_LOOPBACK as CAN_RAW_LOOPBACK,
        CAN_RAW_RECV_OWN_MSGS as CAN_RAW_RECV_OWN_MSGS,
        CAN_RTR_FLAG as CAN_RTR_FLAG,
        CAN_SFF_MASK as CAN_SFF_MASK,
        CAPI as CAPI,
        CMSG_LEN as CMSG_LEN,
        CMSG_SPACE as CMSG_SPACE,
        EAGAIN as EAGAIN,
        EAI_ADDRFAMILY as EAI_ADDRFAMILY,
        EAI_AGAIN as EAI_AGAIN,
        EAI_BADFLAGS as EAI_BADFLAGS,
        EAI_BADHINTS as EAI_BADHINTS,
        EAI_FAIL as EAI_FAIL,
        EAI_FAMILY as EAI_FAMILY,
        EAI_MAX as EAI_MAX,
        EAI_MEMORY as EAI_MEMORY,
        EAI_NODATA as EAI_NODATA,
        EAI_NONAME as EAI_NONAME,
        EAI_OVERFLOW as EAI_OVERFLOW,
        EAI_PROTOCOL as EAI_PROTOCOL,
        EAI_SERVICE as EAI_SERVICE,
        EAI_SOCKTYPE as EAI_SOCKTYPE,
        EAI_SYSTEM as EAI_SYSTEM,
        EBADF as EBADF,
        ETH_P_ALL as ETH_P_ALL,
        ETHERTYPE_ARP as ETHERTYPE_ARP,
        ETHERTYPE_IP as ETHERTYPE_IP,
        ETHERTYPE_IPV6 as ETHERTYPE_IPV6,
        ETHERTYPE_VLAN as ETHERTYPE_VLAN,
        EWOULDBLOCK as EWOULDBLOCK,
        FD_ACCEPT as FD_ACCEPT,
        FD_CLOSE as FD_CLOSE,
        FD_CLOSE_BIT as FD_CLOSE_BIT,
        FD_CONNECT as FD_CONNECT,
        FD_CONNECT_BIT as FD_CONNECT_BIT,
        FD_READ as FD_READ,
        FD_SETSIZE as FD_SETSIZE,
        FD_WRITE as FD_WRITE,
        HCI_DATA_DIR as HCI_DATA_DIR,
        HCI_FILTER as HCI_FILTER,
        HCI_TIME_STAMP as HCI_TIME_STAMP,
        HV_GUID_BROADCAST as HV_GUID_BROADCAST,
        HV_GUID_CHILDREN as HV_GUID_CHILDREN,
        HV_GUID_LOOPBACK as HV_GUID_LOOPBACK,
        HV_GUID_PARENT as HV_GUID_PARENT,
        HV_GUID_WILDCARD as HV_GUID_WILDCARD,
        HV_GUID_ZERO as HV_GUID_ZERO,
        HV_PROTOCOL_RAW as HV_PROTOCOL_RAW,
        HVSOCKET_ADDRESS_FLAG_PASSTHRU as HVSOCKET_ADDRESS_FLAG_PASSTHRU,
        HVSOCKET_CONNECT_TIMEOUT as HVSOCKET_CONNECT_TIMEOUT,
        HVSOCKET_CONNECT_TIMEOUT_MAX as HVSOCKET_CONNECT_TIMEOUT_MAX,
        HVSOCKET_CONNECTED_SUSPEND as HVSOCKET_CONNECTED_SUSPEND,
        INADDR_ALLHOSTS_GROUP as INADDR_ALLHOSTS_GROUP,
        INADDR_ANY as INADDR_ANY,
        INADDR_BROADCAST as INADDR_BROADCAST,
        INADDR_LOOPBACK as INADDR_LOOPBACK,
        INADDR_MAX_LOCAL_GROUP as INADDR_MAX_LOCAL_GROUP,
        INADDR_NONE as INADDR_NONE,
        INADDR_UNSPEC_GROUP as INADDR_UNSPEC_GROUP,
        INFINITE as INFINITE,
        IOCTL_VM_SOCKETS_GET_LOCAL_CID as IOCTL_VM_SOCKETS_GET_LOCAL_CID,
        IP_ADD_MEMBERSHIP as IP_ADD_MEMBERSHIP,
        IP_ADD_SOURCE_MEMBERSHIP as IP_ADD_SOURCE_MEMBERSHIP,
        IP_BLOCK_SOURCE as IP_BLOCK_SOURCE,
        IP_DEFAULT_MULTICAST_LOOP as IP_DEFAULT_MULTICAST_LOOP,
        IP_DEFAULT_MULTICAST_TTL as IP_DEFAULT_MULTICAST_TTL,
        IP_DROP_MEMBERSHIP as IP_DROP_MEMBERSHIP,
        IP_DROP_SOURCE_MEMBERSHIP as IP_DROP_SOURCE_MEMBERSHIP,
        IP_HDRINCL as IP_HDRINCL,
        IP_MAX_MEMBERSHIPS as IP_MAX_MEMBERSHIPS,
        IP_MULTICAST_IF as IP_MULTICAST_IF,
        IP_MULTICAST_LOOP as IP_MULTICAST_LOOP,
        IP_MULTICAST_TTL as IP_MULTICAST_TTL,
        IP_OPTIONS as IP_OPTIONS,
        IP_PKTINFO as IP_PKTINFO,
        IP_RECVDSTADDR as IP_RECVDSTADDR,
        IP_RECVOPTS as IP_RECVOPTS,
        IP_RECVRETOPTS as IP_RECVRETOPTS,
        IP_RECVTOS as IP_RECVTOS,
        IP_RETOPTS as IP_RETOPTS,
        IP_TOS as IP_TOS,
        IP_TRANSPARENT as IP_TRANSPARENT,
        IP_TTL as IP_TTL,
        IP_UNBLOCK_SOURCE as IP_UNBLOCK_SOURCE,
        IPPORT_RESERVED as IPPORT_RESERVED,
        IPPORT_USERRESERVED as IPPORT_USERRESERVED,
        IPPROTO_AH as IPPROTO_AH,
        IPPROTO_CBT as IPPROTO_CBT,
        IPPROTO_DSTOPTS as IPPROTO_DSTOPTS,
        IPPROTO_EGP as IPPROTO_EGP,
        IPPROTO_EON as IPPROTO_EON,
        IPPROTO_ESP as IPPROTO_ESP,
        IPPROTO_FRAGMENT as IPPROTO_FRAGMENT,
        IPPROTO_GGP as IPPROTO_GGP,
        IPPROTO_GRE as IPPROTO_GRE,
        IPPROTO_HELLO as IPPROTO_HELLO,
        IPPROTO_HOPOPTS as IPPROTO_HOPOPTS,
        IPPROTO_ICLFXBM as IPPROTO_ICLFXBM,
        IPPROTO_ICMP as IPPROTO_ICMP,
        IPPROTO_ICMPV6 as IPPROTO_ICMPV6,
        IPPROTO_IDP as IPPROTO_IDP,
        IPPROTO_IGMP as IPPROTO_IGMP,
        IPPROTO_IGP as IPPROTO_IGP,
        IPPROTO_IP as IPPROTO_IP,
        IPPROTO_IPCOMP as IPPROTO_IPCOMP,
        IPPROTO_IPIP as IPPROTO_IPIP,
        IPPROTO_IPV4 as IPPROTO_IPV4,
        IPPROTO_IPV6 as IPPROTO_IPV6,
        IPPROTO_L2TP as IPPROTO_L2TP,
        IPPROTO_MAX as IPPROTO_MAX,
        IPPROTO_MOBILE as IPPROTO_MOBILE,
        IPPROTO_MPTCP as IPPROTO_MPTCP,
        IPPROTO_ND as IPPROTO_ND,
        IPPROTO_NONE as IPPROTO_NONE,
        IPPROTO_PGM as IPPROTO_PGM,
        IPPROTO_PIM as IPPROTO_PIM,
        IPPROTO_PUP as IPPROTO_PUP,
        IPPROTO_RAW as IPPROTO_RAW,
        IPPROTO_RDP as IPPROTO_RDP,
        IPPROTO_ROUTING as IPPROTO_ROUTING,
        IPPROTO_RSVP as IPPROTO_RSVP,
        IPPROTO_SCTP as IPPROTO_SCTP,
        IPPROTO_ST as IPPROTO_ST,
        IPPROTO_TCP as IPPROTO_TCP,
        IPPROTO_TP as IPPROTO_TP,
        IPPROTO_UDP as IPPROTO_UDP,
        IPPROTO_UDPLITE as IPPROTO_UDPLITE,
        IPPROTO_XTP as IPPROTO_XTP,
        IPV6_CHECKSUM as IPV6_CHECKSUM,
        IPV6_DONTFRAG as IPV6_DONTFRAG,
        IPV6_DSTOPTS as IPV6_DSTOPTS,
        IPV6_HOPLIMIT as IPV6_HOPLIMIT,
        IPV6_HOPOPTS as IPV6_HOPOPTS,
        IPV6_JOIN_GROUP as IPV6_JOIN_GROUP,
        IPV6_LEAVE_GROUP as IPV6_LEAVE_GROUP,
        IPV6_MULTICAST_HOPS as IPV6_MULTICAST_HOPS,
        IPV6_MULTICAST_IF as IPV6_MULTICAST_IF,
        IPV6_MULTICAST_LOOP as IPV6_MULTICAST_LOOP,
        IPV6_NEXTHOP as IPV6_NEXTHOP,
        IPV6_PATHMTU as IPV6_PATHMTU,
        IPV6_PKTINFO as IPV6_PKTINFO,
        IPV6_RECVDSTOPTS as IPV6_RECVDSTOPTS,
        IPV6_RECVHOPLIMIT as IPV6_RECVHOPLIMIT,
        IPV6_RECVHOPOPTS as IPV6_RECVHOPOPTS,
        IPV6_RECVPATHMTU as IPV6_RECVPATHMTU,
        IPV6_RECVPKTINFO as IPV6_RECVPKTINFO,
        IPV6_RECVRTHDR as IPV6_RECVRTHDR,
        IPV6_RECVTCLASS as IPV6_RECVTCLASS,
        IPV6_RTHDR as IPV6_RTHDR,
        IPV6_RTHDR_TYPE_0 as IPV6_RTHDR_TYPE_0,
        IPV6_RTHDRDSTOPTS as IPV6_RTHDRDSTOPTS,
        IPV6_TCLASS as IPV6_TCLASS,
        IPV6_UNICAST_HOPS as IPV6_UNICAST_HOPS,
        IPV6_USE_MIN_MTU as IPV6_USE_MIN_MTU,
        IPV6_V6ONLY as IPV6_V6ONLY,
        J1939_EE_INFO_NONE as J1939_EE_INFO_NONE,
        J1939_EE_INFO_TX_ABORT as J1939_EE_INFO_TX_ABORT,
        J1939_FILTER_MAX as J1939_FILTER_MAX,
        J1939_IDLE_ADDR as J1939_IDLE_ADDR,
        J1939_MAX_UNICAST_ADDR as J1939_MAX_UNICAST_ADDR,
        J1939_NLA_BYTES_ACKED as J1939_NLA_BYTES_ACKED,
        J1939_NLA_PAD as J1939_NLA_PAD,
        J1939_NO_ADDR as J1939_NO_ADDR,
        J1939_NO_NAME as J1939_NO_NAME,
        J1939_NO_PGN as J1939_NO_PGN,
        J1939_PGN_ADDRESS_CLAIMED as J1939_PGN_ADDRESS_CLAIMED,
        J1939_PGN_ADDRESS_COMMANDED as J1939_PGN_ADDRESS_COMMANDED,
        J1939_PGN_MAX as J1939_PGN_MAX,
        J1939_PGN_PDU1_MAX as J1939_PGN_PDU1_MAX,
        J1939_PGN_REQUEST as J1939_PGN_REQUEST,
        LOCAL_PEERCRED as LOCAL_PEERCRED,
        MSG_BCAST as MSG_BCAST,
        MSG_CMSG_CLOEXEC as MSG_CMSG_CLOEXEC,
        MSG_CONFIRM as MSG_CONFIRM,
        MSG_CTRUNC as MSG_CTRUNC,
        MSG_DONTROUTE as MSG_DONTROUTE,
        MSG_DONTWAIT as MSG_DONTWAIT,
        MSG_EOF as MSG_EOF,
        MSG_EOR as MSG_EOR,
        MSG_ERRQUEUE as MSG_ERRQUEUE,
        MSG_FASTOPEN as MSG_FASTOPEN,
        MSG_MCAST as MSG_MCAST,
        MSG_MORE as MSG_MORE,
        MSG_NOSIGNAL as MSG_NOSIGNAL,
        MSG_NOTIFICATION as MSG_NOTIFICATION,
        MSG_OOB as MSG_OOB,
        MSG_PEEK as MSG_PEEK,
        MSG_TRUNC as MSG_TRUNC,
        MSG_WAITALL as MSG_WAITALL,
        NETLINK_CRYPTO as NETLINK_CRYPTO,
        NETLINK_DNRTMSG as NETLINK_DNRTMSG,
        NETLINK_FIREWALL as NETLINK_FIREWALL,
        NETLINK_IP6_FW as NETLINK_IP6_FW,
        NETLINK_NFLOG as NETLINK_NFLOG,
        NETLINK_ROUTE as NETLINK_ROUTE,
        NETLINK_USERSOCK as NETLINK_USERSOCK,
        NETLINK_XFRM as NETLINK_XFRM,
        NI_DGRAM as NI_DGRAM,
        NI_IDN as NI_IDN,
        NI_MAXHOST as NI_MAXHOST,
        NI_MAXSERV as NI_MAXSERV,
        NI_NAMEREQD as NI_NAMEREQD,
        NI_NOFQDN as NI_NOFQDN,
        NI_NUMERICHOST as NI_NUMERICHOST,
        NI_NUMERICSERV as NI_NUMERICSERV,
        PACKET_BROADCAST as PACKET_BROADCAST,
        PACKET_FASTROUTE as PACKET_FASTROUTE,
        PACKET_HOST as PACKET_HOST,
        PACKET_LOOPBACK as PACKET_LOOPBACK,
        PACKET_MULTICAST as PACKET_MULTICAST,
        PACKET_OTHERHOST as PACKET_OTHERHOST,
        PACKET_OUTGOING as PACKET_OUTGOING,
        PF_CAN as PF_CAN,
        PF_PACKET as PF_PACKET,
        PF_RDS as PF_RDS,
        PF_SYSTEM as PF_SYSTEM,
        POLLERR as POLLERR,
        POLLHUP as POLLHUP,
        POLLIN as POLLIN,
        POLLMSG as POLLMSG,
        POLLNVAL as POLLNVAL,
        POLLOUT as POLLOUT,
        POLLPRI as POLLPRI,
        POLLRDBAND as POLLRDBAND,
        POLLRDNORM as POLLRDNORM,
        POLLWRNORM as POLLWRNORM,
        RCVALL_MAX as RCVALL_MAX,
        RCVALL_OFF as RCVALL_OFF,
        RCVALL_ON as RCVALL_ON,
        RCVALL_SOCKETLEVELONLY as RCVALL_SOCKETLEVELONLY,
        SCM_CREDENTIALS as SCM_CREDENTIALS,
        SCM_CREDS as SCM_CREDS,
        SCM_J1939_DEST_ADDR as SCM_J1939_DEST_ADDR,
        SCM_J1939_DEST_NAME as SCM_J1939_DEST_NAME,
        SCM_J1939_ERRQUEUE as SCM_J1939_ERRQUEUE,
        SCM_J1939_PRIO as SCM_J1939_PRIO,
        SCM_RIGHTS as SCM_RIGHTS,
        SHUT_RD as SHUT_RD,
        SHUT_RDWR as SHUT_RDWR,
        SHUT_WR as SHUT_WR,
        SIO_KEEPALIVE_VALS as SIO_KEEPALIVE_VALS,
        SIO_LOOPBACK_FAST_PATH as SIO_LOOPBACK_FAST_PATH,
        SIO_RCVALL as SIO_RCVALL,
        SIOCGIFINDEX as SIOCGIFINDEX,
        SIOCGIFNAME as SIOCGIFNAME,
        SO_ACCEPTCONN as SO_ACCEPTCONN,
        SO_BINDTODEVICE as SO_BINDTODEVICE,
        SO_BINDTOIFINDEX as SO_BINDTOIFINDEX,
        SO_BROADCAST as SO_BROADCAST,
        SO_DEBUG as SO_DEBUG,
        SO_DOMAIN as SO_DOMAIN,
        SO_DONTROUTE as SO_DONTROUTE,
        SO_ERROR as SO_ERROR,
        SO_EXCLUSIVEADDRUSE as SO_EXCLUSIVEADDRUSE,
        SO_INCOMING_CPU as SO_INCOMING_CPU,
        SO_J1939_ERRQUEUE as SO_J1939_ERRQUEUE,
        SO_J1939_FILTER as SO_J1939_FILTER,
        SO_J1939_PROMISC as SO_J1939_PROMISC,
        SO_J1939_SEND_PRIO as SO_J1939_SEND_PRIO,
        SO_KEEPALIVE as SO_KEEPALIVE,
        SO_LINGER as SO_LINGER,
        SO_MARK as SO_MARK,
        SO_OOBINLINE as SO_OOBINLINE,
        SO_PASSCRED as SO_PASSCRED,
        SO_PASSSEC as SO_PASSSEC,
        SO_PEERCRED as SO_PEERCRED,
        SO_PEERSEC as SO_PEERSEC,
        SO_PRIORITY as SO_PRIORITY,
        SO_PROTOCOL as SO_PROTOCOL,
        SO_RCVBUF as SO_RCVBUF,
        SO_RCVLOWAT as SO_RCVLOWAT,
        SO_RCVTIMEO as SO_RCVTIMEO,
        SO_REUSEADDR as SO_REUSEADDR,
        SO_REUSEPORT as SO_REUSEPORT,
        SO_SETFIB as SO_SETFIB,
        SO_SNDBUF as SO_SNDBUF,
        SO_SNDLOWAT as SO_SNDLOWAT,
        SO_SNDTIMEO as SO_SNDTIMEO,
        SO_TYPE as SO_TYPE,
        SO_USELOOPBACK as SO_USELOOPBACK,
        SO_VM_SOCKETS_BUFFER_MAX_SIZE as SO_VM_SOCKETS_BUFFER_MAX_SIZE,
        SO_VM_SOCKETS_BUFFER_MIN_SIZE as SO_VM_SOCKETS_BUFFER_MIN_SIZE,
        SO_VM_SOCKETS_BUFFER_SIZE as SO_VM_SOCKETS_BUFFER_SIZE,
        SOCK_CLOEXEC as SOCK_CLOEXEC,
        SOCK_DGRAM as SOCK_DGRAM,
        SOCK_NONBLOCK as SOCK_NONBLOCK,
        SOCK_RAW as SOCK_RAW,
        SOCK_RDM as SOCK_RDM,
        SOCK_SEQPACKET as SOCK_SEQPACKET,
        SOCK_STREAM as SOCK_STREAM,
        SOL_ALG as SOL_ALG,
        SOL_CAN_BASE as SOL_CAN_BASE,
        SOL_CAN_RAW as SOL_CAN_RAW,
        SOL_HCI as SOL_HCI,
        SOL_IP as SOL_IP,
        SOL_RDS as SOL_RDS,
        SOL_SOCKET as SOL_SOCKET,
        SOL_TCP as SOL_TCP,
        SOL_TIPC as SOL_TIPC,
        SOL_UDP as SOL_UDP,
        SOMAXCONN as SOMAXCONN,
        SYSPROTO_CONTROL as SYSPROTO_CONTROL,
        TCP_CC_INFO as TCP_CC_INFO,
        TCP_CONGESTION as TCP_CONGESTION,
        TCP_CONNECTION_INFO as TCP_CONNECTION_INFO,
        TCP_CORK as TCP_CORK,
        TCP_DEFER_ACCEPT as TCP_DEFER_ACCEPT,
        TCP_FASTOPEN as TCP_FASTOPEN,
        TCP_FASTOPEN_CONNECT as TCP_FASTOPEN_CONNECT,
        TCP_FASTOPEN_KEY as TCP_FASTOPEN_KEY,
        TCP_FASTOPEN_NO_COOKIE as TCP_FASTOPEN_NO_COOKIE,
        TCP_INFO as TCP_INFO,
        TCP_INQ as TCP_INQ,
        TCP_KEEPALIVE as TCP_KEEPALIVE,
        TCP_KEEPCNT as TCP_KEEPCNT,
        TCP_KEEPIDLE as TCP_KEEPIDLE,
        TCP_KEEPINTVL as TCP_KEEPINTVL,
        TCP_LINGER2 as TCP_LINGER2,
        TCP_MAXSEG as TCP_MAXSEG,
        TCP_MD5SIG as TCP_MD5SIG,
        TCP_MD5SIG_EXT as TCP_MD5SIG_EXT,
        TCP_NODELAY as TCP_NODELAY,
        TCP_NOTSENT_LOWAT as TCP_NOTSENT_LOWAT,
        TCP_QUEUE_SEQ as TCP_QUEUE_SEQ,
        TCP_QUICKACK as TCP_QUICKACK,
        TCP_REPAIR as TCP_REPAIR,
        TCP_REPAIR_OPTIONS as TCP_REPAIR_OPTIONS,
        TCP_REPAIR_QUEUE as TCP_REPAIR_QUEUE,
        TCP_REPAIR_WINDOW as TCP_REPAIR_WINDOW,
        TCP_SAVE_SYN as TCP_SAVE_SYN,
        TCP_SAVED_SYN as TCP_SAVED_SYN,
        TCP_SYNCNT as TCP_SYNCNT,
        TCP_THIN_DUPACK as TCP_THIN_DUPACK,
        TCP_THIN_LINEAR_TIMEOUTS as TCP_THIN_LINEAR_TIMEOUTS,
        TCP_TIMESTAMP as TCP_TIMESTAMP,
        TCP_TX_DELAY as TCP_TX_DELAY,
        TCP_ULP as TCP_ULP,
        TCP_USER_TIMEOUT as TCP_USER_TIMEOUT,
        TCP_WINDOW_CLAMP as TCP_WINDOW_CLAMP,
        TCP_ZEROCOPY_RECEIVE as TCP_ZEROCOPY_RECEIVE,
        TIPC_ADDR_ID as TIPC_ADDR_ID,
        TIPC_ADDR_NAME as TIPC_ADDR_NAME,
        TIPC_ADDR_NAMESEQ as TIPC_ADDR_NAMESEQ,
        TIPC_CFG_SRV as TIPC_CFG_SRV,
        TIPC_CLUSTER_SCOPE as TIPC_CLUSTER_SCOPE,
        TIPC_CONN_TIMEOUT as TIPC_CONN_TIMEOUT,
        TIPC_CRITICAL_IMPORTANCE as TIPC_CRITICAL_IMPORTANCE,
        TIPC_DEST_DROPPABLE as TIPC_DEST_DROPPABLE,
        TIPC_HIGH_IMPORTANCE as TIPC_HIGH_IMPORTANCE,
        TIPC_IMPORTANCE as TIPC_IMPORTANCE,
        TIPC_LOW_IMPORTANCE as TIPC_LOW_IMPORTANCE,
        TIPC_MEDIUM_IMPORTANCE as TIPC_MEDIUM_IMPORTANCE,
        TIPC_NODE_SCOPE as TIPC_NODE_SCOPE,
        TIPC_PUBLISHED as TIPC_PUBLISHED,
        TIPC_SRC_DROPPABLE as TIPC_SRC_DROPPABLE,
        TIPC_SUB_CANCEL as TIPC_SUB_CANCEL,
        TIPC_SUB_PORTS as TIPC_SUB_PORTS,
        TIPC_SUB_SERVICE as TIPC_SUB_SERVICE,
        TIPC_SUBSCR_TIMEOUT as TIPC_SUBSCR_TIMEOUT,
        TIPC_TOP_SRV as TIPC_TOP_SRV,
        TIPC_WAIT_FOREVER as TIPC_WAIT_FOREVER,
        TIPC_WITHDRAWN as TIPC_WITHDRAWN,
        TIPC_ZONE_SCOPE as TIPC_ZONE_SCOPE,
        UDPLITE_RECV_CSCOV as UDPLITE_RECV_CSCOV,
        UDPLITE_SEND_CSCOV as UDPLITE_SEND_CSCOV,
        VM_SOCKETS_INVALID_VERSION as VM_SOCKETS_INVALID_VERSION,
        VMADDR_CID_ANY as VMADDR_CID_ANY,
        VMADDR_CID_HOST as VMADDR_CID_HOST,
        VMADDR_PORT_ANY as VMADDR_PORT_ANY,
        WSA_FLAG_OVERLAPPED as WSA_FLAG_OVERLAPPED,
        WSA_INVALID_HANDLE as WSA_INVALID_HANDLE,
        WSA_INVALID_PARAMETER as WSA_INVALID_PARAMETER,
        WSA_IO_INCOMPLETE as WSA_IO_INCOMPLETE,
        WSA_IO_PENDING as WSA_IO_PENDING,
        WSA_NOT_ENOUGH_MEMORY as WSA_NOT_ENOUGH_MEMORY,
        WSA_OPERATION_ABORTED as WSA_OPERATION_ABORTED,
        WSA_WAIT_FAILED as WSA_WAIT_FAILED,
        WSA_WAIT_TIMEOUT as WSA_WAIT_TIMEOUT,
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\testing\_fake_net.py ===
# This should eventually be cleaned up and become public, but for right now I'm just
# implementing enough to test DTLS.

# TODO:
# - user-defined routers
# - TCP
# - UDP broadcast

from __future__ import annotations

import contextlib
import errno
import ipaddress
import os
import socket
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    NoReturn,
    Union,
    overload,
)

import attrs

import trio
from trio._util import NoPublicConstructor, final

if TYPE_CHECKING:
    import builtins
    from collections.abc import Iterable
    from socket import AddressFamily, SocketKind
    from types import TracebackType

    from typing_extensions import Buffer, Self, TypeAlias

    from trio._socket import AddressFormat

IPAddress: TypeAlias = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]


def _family_for(ip: IPAddress) -> int:
    if isinstance(ip, ipaddress.IPv4Address):
        return trio.socket.AF_INET
    elif isinstance(ip, ipaddress.IPv6Address):
        return trio.socket.AF_INET6
    raise NotImplementedError("Unhandled IPAddress instance type")  # pragma: no cover


def _wildcard_ip_for(family: int) -> IPAddress:
    if family == trio.socket.AF_INET:
        return ipaddress.ip_address("0.0.0.0")
    elif family == trio.socket.AF_INET6:
        return ipaddress.ip_address("::")
    raise NotImplementedError("Unhandled ip address family")  # pragma: no cover


# not used anywhere
def _localhost_ip_for(family: int) -> IPAddress:  # pragma: no cover
    if family == trio.socket.AF_INET:
        return ipaddress.ip_address("127.0.0.1")
    elif family == trio.socket.AF_INET6:
        return ipaddress.ip_address("::1")
    raise NotImplementedError("Unhandled ip address family")


def _fake_err(code: int) -> NoReturn:
    raise OSError(code, os.strerror(code))


def _scatter(data: bytes, buffers: Iterable[Buffer]) -> int:
    written = 0
    for buf in buffers:  # pragma: no branch
        next_piece = data[written : written + memoryview(buf).nbytes]
        with memoryview(buf) as mbuf:
            mbuf[: len(next_piece)] = next_piece
        written += len(next_piece)
        if written == len(data):  # pragma: no branch
            break
    return written


@attrs.frozen
class UDPEndpoint:
    ip: IPAddress
    port: int

    def as_python_sockaddr(self) -> tuple[str, int] | tuple[str, int, int, int]:
        sockaddr: tuple[str, int] | tuple[str, int, int, int] = (
            self.ip.compressed,
            self.port,
        )
        if isinstance(self.ip, ipaddress.IPv6Address):
            sockaddr += (0, 0)  # type: ignore[assignment]
        return sockaddr

    @classmethod
    def from_python_sockaddr(
        cls,
        sockaddr: tuple[str, int] | tuple[str, int, int, int],
    ) -> UDPEndpoint:
        ip, port = sockaddr[:2]
        return cls(ip=ipaddress.ip_address(ip), port=port)


@attrs.frozen
class UDPBinding:
    local: UDPEndpoint
    # remote: UDPEndpoint # ??


@attrs.frozen
class UDPPacket:
    source: UDPEndpoint
    destination: UDPEndpoint
    payload: bytes = attrs.field(repr=lambda p: p.hex())

    # not used/tested anywhere
    def reply(self, payload: bytes) -> UDPPacket:  # pragma: no cover
        return UDPPacket(
            source=self.destination,
            destination=self.source,
            payload=payload,
        )


@attrs.frozen
class FakeSocketFactory(trio.abc.SocketFactory):
    fake_net: FakeNet

    def socket(self, family: int, type_: int, proto: int) -> FakeSocket:  # type: ignore[override]
        return FakeSocket._create(self.fake_net, family, type_, proto)


@attrs.frozen
class FakeHostnameResolver(trio.abc.HostnameResolver):
    fake_net: FakeNet

    async def getaddrinfo(
        self,
        host: bytes | None,
        port: bytes | str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[
        tuple[
            AddressFamily,
            SocketKind,
            int,
            str,
            tuple[str, int] | tuple[str, int, int, int] | tuple[int, bytes],
        ]
    ]:
        raise NotImplementedError("FakeNet doesn't do fake DNS yet")

    async def getnameinfo(
        self,
        sockaddr: tuple[str, int] | tuple[str, int, int, int],
        flags: int,
    ) -> tuple[str, str]:
        raise NotImplementedError("FakeNet doesn't do fake DNS yet")


@final
class FakeNet:
    def __init__(self) -> None:
        # When we need to pick an arbitrary unique ip address/port, use these:
        self._auto_ipv4_iter = ipaddress.IPv4Network("1.0.0.0/8").hosts()  # untested
        self._auto_ipv6_iter = ipaddress.IPv6Network("1::/16").hosts()  # untested
        self._auto_port_iter = iter(range(50000, 65535))

        self._bound: dict[UDPBinding, FakeSocket] = {}

        self.route_packet = None

    def _bind(self, binding: UDPBinding, socket: FakeSocket) -> None:
        if binding in self._bound:
            _fake_err(errno.EADDRINUSE)
        self._bound[binding] = socket

    def enable(self) -> None:
        trio.socket.set_custom_socket_factory(FakeSocketFactory(self))
        trio.socket.set_custom_hostname_resolver(FakeHostnameResolver(self))

    def send_packet(self, packet: UDPPacket) -> None:
        if self.route_packet is None:
            self.deliver_packet(packet)
        else:
            self.route_packet(packet)

    def deliver_packet(self, packet: UDPPacket) -> None:
        binding = UDPBinding(local=packet.destination)
        if binding in self._bound:
            self._bound[binding]._deliver_packet(packet)
        else:
            # No valid destination, so drop it
            pass


@final
class FakeSocket(trio.socket.SocketType, metaclass=NoPublicConstructor):
    def __init__(
        self,
        fake_net: FakeNet,
        family: AddressFamily,
        type: SocketKind,
        proto: int,
    ) -> None:
        self._fake_net = fake_net

        if not family:  # pragma: no cover
            family = trio.socket.AF_INET
        if not type:  # pragma: no cover
            type = trio.socket.SOCK_STREAM  # noqa: A001  # name shadowing builtin

        if family not in (trio.socket.AF_INET, trio.socket.AF_INET6):
            raise NotImplementedError(f"FakeNet doesn't (yet) support family={family}")
        if type != trio.socket.SOCK_DGRAM:
            raise NotImplementedError(f"FakeNet doesn't (yet) support type={type}")

        self._family = family
        self._type = type
        self._proto = proto

        self._closed = False

        self._packet_sender, self._packet_receiver = trio.open_memory_channel[
            UDPPacket
        ](float("inf"))

        # This is the source-of-truth for what port etc. this socket is bound to
        self._binding: UDPBinding | None = None

    @property
    def type(self) -> SocketKind:
        return self._type

    @property
    def family(self) -> AddressFamily:
        return self._family

    @property
    def proto(self) -> int:
        return self._proto

    def _check_closed(self) -> None:
        if self._closed:
            _fake_err(errno.EBADF)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._binding is not None:
            del self._fake_net._bound[self._binding]
        self._packet_receiver.close()

    async def _resolve_address_nocp(
        self,
        address: object,
        *,
        local: bool,
    ) -> tuple[str, int]:
        return await trio._socket._resolve_address_nocp(  # type: ignore[no-any-return]
            self.type,
            self.family,
            self.proto,
            address=address,
            ipv6_v6only=False,
            local=local,
        )

    def _deliver_packet(self, packet: UDPPacket) -> None:
        # sending to a closed socket -- UDP packets get dropped
        with contextlib.suppress(trio.BrokenResourceError):
            self._packet_sender.send_nowait(packet)

    ################################################################
    # Actual IO operation implementations
    ################################################################

    async def bind(self, addr: object) -> None:
        self._check_closed()
        if self._binding is not None:
            _fake_err(errno.EINVAL)
        await trio.lowlevel.checkpoint()
        ip_str, port, *_ = await self._resolve_address_nocp(addr, local=True)
        assert _ == [], "TODO: handle other values?"

        ip = ipaddress.ip_address(ip_str)
        assert _family_for(ip) == self.family
        # We convert binds to INET_ANY into binds to localhost
        if ip == ipaddress.ip_address("0.0.0.0"):
            ip = ipaddress.ip_address("127.0.0.1")
        elif ip == ipaddress.ip_address("::"):
            ip = ipaddress.ip_address("::1")
        if port == 0:
            port = next(self._fake_net._auto_port_iter)
        binding = UDPBinding(local=UDPEndpoint(ip, port))
        self._fake_net._bind(binding, self)
        self._binding = binding

    async def connect(self, peer: object) -> NoReturn:
        raise NotImplementedError("FakeNet does not (yet) support connected sockets")

    async def _sendmsg(
        self,
        buffers: Iterable[Buffer],
        ancdata: Iterable[tuple[int, int, Buffer]] = (),
        flags: int = 0,
        address: AddressFormat | None = None,
    ) -> int:
        self._check_closed()

        await trio.lowlevel.checkpoint()

        if address is not None:
            address = await self._resolve_address_nocp(address, local=False)
        if ancdata:
            raise NotImplementedError("FakeNet doesn't support ancillary data")
        if flags:
            raise NotImplementedError(f"FakeNet send flags must be 0, not {flags}")

        if address is None:
            _fake_err(errno.ENOTCONN)

        destination = UDPEndpoint.from_python_sockaddr(address)

        if self._binding is None:
            await self.bind((_wildcard_ip_for(self.family).compressed, 0))

        payload = b"".join(buffers)

        assert self._binding is not None
        packet = UDPPacket(
            source=self._binding.local,
            destination=destination,
            payload=payload,
        )

        self._fake_net.send_packet(packet)

        return len(payload)

    if sys.platform != "win32" or (
        not TYPE_CHECKING and hasattr(socket.socket, "sendmsg")
    ):
        sendmsg = _sendmsg

    async def _recvmsg_into(
        self,
        buffers: Iterable[Buffer],
        ancbufsize: int = 0,
        flags: int = 0,
    ) -> tuple[
        int,
        list[tuple[int, int, bytes]],
        int,
        tuple[str, int] | tuple[str, int, int, int],
    ]:
        if ancbufsize != 0:
            raise NotImplementedError("FakeNet doesn't support ancillary data")
        if flags != 0:
            raise NotImplementedError("FakeNet doesn't support any recv flags")
        if self._binding is None:
            # I messed this up a few times when writing tests ... but it also never happens
            # in any of the existing tests, so maybe it could be intentional...
            raise NotImplementedError(
                "The code will most likely hang if you try to receive on a fakesocket "
                "without a binding. If that is not the case, or you explicitly want to "
                "test that, remove this warning.",
            )

        self._check_closed()

        ancdata: list[tuple[int, int, bytes]] = []
        msg_flags = 0

        packet = await self._packet_receiver.receive()
        address = packet.source.as_python_sockaddr()
        written = _scatter(packet.payload, buffers)
        if written < len(packet.payload):
            msg_flags |= trio.socket.MSG_TRUNC
        return written, ancdata, msg_flags, address

    if sys.platform != "win32" or (
        not TYPE_CHECKING and hasattr(socket.socket, "sendmsg")
    ):
        recvmsg_into = _recvmsg_into

    ################################################################
    # Simple state query stuff
    ################################################################

    def getsockname(self) -> tuple[str, int] | tuple[str, int, int, int]:
        self._check_closed()
        if self._binding is not None:
            return self._binding.local.as_python_sockaddr()
        elif self.family == trio.socket.AF_INET:
            return ("0.0.0.0", 0)
        else:
            assert self.family == trio.socket.AF_INET6
            return ("::", 0)

    # TODO: This method is not tested, and seems to make incorrect assumptions. It should maybe raise NotImplementedError.
    def getpeername(self) -> tuple[str, int] | tuple[str, int, int, int]:
        self._check_closed()
        if self._binding is not None:
            assert hasattr(
                self._binding,
                "remote",
            ), "This method seems to assume that self._binding has a remote UDPEndpoint"
            if self._binding.remote is not None:  # pragma: no cover
                assert isinstance(
                    self._binding.remote,
                    UDPEndpoint,
                ), "Self._binding.remote should be a UDPEndpoint"
                return self._binding.remote.as_python_sockaddr()
        _fake_err(errno.ENOTCONN)

    @overload
    def getsockopt(self, /, level: int, optname: int) -> int: ...

    @overload
    def getsockopt(self, /, level: int, optname: int, buflen: int) -> bytes: ...

    def getsockopt(
        self,
        /,
        level: int,
        optname: int,
        buflen: int | None = None,
    ) -> int | bytes:
        self._check_closed()
        raise OSError(f"FakeNet doesn't implement getsockopt({level}, {optname})")

    @overload
    def setsockopt(self, /, level: int, optname: int, value: int | Buffer) -> None: ...

    @overload
    def setsockopt(
        self,
        /,
        level: int,
        optname: int,
        value: None,
        optlen: int,
    ) -> None: ...

    def setsockopt(
        self,
        /,
        level: int,
        optname: int,
        value: int | Buffer | None,
        optlen: int | None = None,
    ) -> None:
        self._check_closed()

        if (level, optname) == (
            trio.socket.IPPROTO_IPV6,
            trio.socket.IPV6_V6ONLY,
        ) and not value:
            raise NotImplementedError("FakeNet always has IPV6_V6ONLY=True")

        raise OSError(f"FakeNet doesn't implement setsockopt({level}, {optname}, ...)")

    ################################################################
    # Various boilerplate and trivial stubs
    ################################################################

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: builtins.type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    async def send(self, data: Buffer, flags: int = 0) -> int:
        return await self.sendto(data, flags, None)

    # __ prefixed arguments because typeshed uses that and typechecker issues
    @overload
    async def sendto(
        self,
        __data: Buffer,  # noqa: PYI063
        __address: tuple[object, ...] | str | Buffer,
    ) -> int: ...

    # __ prefixed arguments because typeshed uses that and typechecker issues
    @overload
    async def sendto(
        self,
        __data: Buffer,  # noqa: PYI063
        __flags: int,
        __address: tuple[object, ...] | str | Buffer | None,
    ) -> int: ...

    async def sendto(  # type: ignore[explicit-any]
        self,
        *args: Any,
    ) -> int:
        data: Buffer
        flags: int
        address: tuple[object, ...] | str | Buffer
        if len(args) == 2:
            data, address = args
            flags = 0
        elif len(args) == 3:
            data, flags, address = args
        else:
            raise TypeError("wrong number of arguments")
        return await self._sendmsg([data], [], flags, address)

    async def recv(self, bufsize: int, flags: int = 0) -> bytes:
        data, _address = await self.recvfrom(bufsize, flags)
        return data

    async def recv_into(self, buf: Buffer, nbytes: int = 0, flags: int = 0) -> int:
        got_bytes, _address = await self.recvfrom_into(buf, nbytes, flags)
        return got_bytes

    async def recvfrom(
        self,
        bufsize: int,
        flags: int = 0,
    ) -> tuple[bytes, AddressFormat]:
        data, _ancdata, _msg_flags, address = await self._recvmsg(bufsize, flags)
        return data, address

    async def recvfrom_into(
        self,
        buf: Buffer,
        nbytes: int = 0,
        flags: int = 0,
    ) -> tuple[int, AddressFormat]:
        if nbytes != 0 and nbytes != memoryview(buf).nbytes:
            raise NotImplementedError("partial recvfrom_into")
        got_nbytes, _ancdata, _msg_flags, address = await self._recvmsg_into(
            [buf],
            0,
            flags,
        )
        return got_nbytes, address

    async def _recvmsg(
        self,
        bufsize: int,
        ancbufsize: int = 0,
        flags: int = 0,
    ) -> tuple[bytes, list[tuple[int, int, bytes]], int, AddressFormat]:
        buf = bytearray(bufsize)
        got_nbytes, ancdata, msg_flags, address = await self._recvmsg_into(
            [buf],
            ancbufsize,
            flags,
        )
        return (bytes(buf[:got_nbytes]), ancdata, msg_flags, address)

    if sys.platform != "win32" or (
        not TYPE_CHECKING and hasattr(socket.socket, "sendmsg")
    ):
        recvmsg = _recvmsg

    def fileno(self) -> int:
        raise NotImplementedError("can't get fileno() for FakeNet sockets")

    def detach(self) -> int:
        raise NotImplementedError("can't detach() a FakeNet socket")

    def get_inheritable(self) -> bool:
        return False

    def set_inheritable(self, inheritable: bool) -> None:
        if inheritable:
            raise NotImplementedError("FakeNet can't make inheritable sockets")

    if sys.platform == "win32" or (
        not TYPE_CHECKING and hasattr(socket.socket, "share")
    ):

        def share(self, process_id: int) -> bytes:
            raise NotImplementedError("FakeNet can't share sockets")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\onnx_model_bert_tf.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx_model_bert import BertOnnxModel

logger = logging.getLogger(__name__)


class BertOnnxModelTF(BertOnnxModel):
    def __init__(self, model, num_heads, hidden_size):
        super().__init__(model, num_heads, hidden_size)

    def remove_identity(self):
        nodes_to_remove = []
        for node in self.nodes():
            if node.op_type == "Identity":
                if not self.find_graph_output(node.output[0]):
                    self.replace_input_of_all_nodes(node.output[0], node.input[0])
                    nodes_to_remove.append(node)
        self.remove_nodes(nodes_to_remove)
        logger.info(f"Removed Identity count: {len(nodes_to_remove)}")

    def match_mask_path(self, add_or_sub_before_softmax):
        mask_nodes = self.match_parent_path(
            add_or_sub_before_softmax,
            ["Mul", "Sub", "Reshape", "Cast"],
            [1, None, 1, 0],
        )
        if mask_nodes is not None:
            return mask_nodes

        mask_nodes = self.match_parent_path(
            add_or_sub_before_softmax,
            ["Mul", "Sub", "Cast", "Slice", "Unsqueeze"],
            [1, 0, 1, 0, 0],
        )
        if mask_nodes is not None:
            return mask_nodes

        mask_nodes = self.match_parent_path(
            add_or_sub_before_softmax,
            ["Mul", "Sub", "Cast", "Unsqueeze", "Unsqueeze"],
            [1, None, 1, 0, 0],
        )

        return mask_nodes

    def get_2d_initializers_from_parent_subgraphs(self, current_node):
        """
        Find initializers that is 2D. Returns a dictionary with name as key and shape as value.
        """
        parent_nodes = self.get_parent_subgraph_nodes(current_node, [])
        initializers = {}
        for node in parent_nodes:
            for input in node.input:
                initializer = self.get_initializer(input)
                if initializer:
                    temp = numpy_helper.to_array(initializer)
                    if len(temp.shape) == 2:
                        initializers[initializer.name] = temp.shape

        return initializers

    def find_segment_ids(self, segment_embedding, input_ids):
        input_name_to_nodes = self.input_name_to_nodes()
        if segment_embedding not in input_name_to_nodes:
            return None

        nodes = input_name_to_nodes[segment_embedding]
        if len(nodes) != 1:
            return None

        graph_inputs = self.get_graph_inputs(nodes[0], recursive=True)
        if len(graph_inputs) > 1:
            print("Found multiple candidates of segment_ids", graph_inputs)
            return None
        # Find segment ids in graph inputs. The segment id input must not be the same as input_ids.
        if len(graph_inputs) == 1 and graph_inputs[0] != input_ids:
            return graph_inputs[0]

        # If the segment id candidate is the same as the input_ids, try to assign alternative segment ids and simplify the graph if needed.
        segment_ids = nodes[0].input[1]
        _, segment_id_path, _ = self.match_parent_paths(
            nodes[0],
            [
                (
                    ["ConstantOfShape", "Cast", "Concat", "Slice", "Cast", "Shape"],
                    [1, 0, 0, 0, 0, 0],
                ),
                (
                    [
                        "ConstantOfShape",
                        "Cast",
                        "Concat",
                        "Unsqueeze",
                        "Squeeze",
                        "Slice",
                        "Cast",
                        "Shape",
                    ],
                    [1, 0, 0, 0, 0, 0, 0, 0],
                ),
            ],
            None,
        )

        if segment_id_path and input_ids and input_ids == segment_id_path[-1].input[0]:
            logger.debug("Simplify semgent id path...")
            constantofshape_node = segment_id_path[0]
            graph_name = self.get_graph_by_node(constantofshape_node).name
            self.add_node(
                helper.make_node("Shape", inputs=[input_ids], outputs=["input_shape"]),
                graph_name,
            )
            constantofshape_value = helper.get_attribute_value(constantofshape_node.attribute[0])
            self.add_node(
                helper.make_node(
                    "ConstantOfShape",
                    inputs=["input_shape"],
                    outputs=["zeros_for_input_shape"],
                    value=constantofshape_value,
                ),
                graph_name,
            )
            segment_ids = "zeros_for_input_shape"
        return segment_ids

    def find_input_ids(self, word_embedding):
        input_name_to_nodes = self.input_name_to_nodes()
        if word_embedding not in input_name_to_nodes:
            return None

        nodes = input_name_to_nodes[word_embedding]
        if len(nodes) != 1:
            return None

        graph_inputs = self.get_graph_inputs(nodes[0], recursive=True)
        if len(graph_inputs) == 1:
            return graph_inputs[0]

        print("Found multiple candidates of input_ids", graph_inputs)
        return None

    def find_mask_input(self, excluded_graph_inputs):
        for node in self.nodes():
            if node.op_type == "Softmax":
                mask_path = self.match_parent_path(
                    node,
                    ["Add", "Mul", "Sub", "Cast", "Slice", "Unsqueeze"],
                    [0, 1, None, 1, 0, 0],
                )
                if mask_path is None:
                    continue
                (
                    add_node,
                    mul_node,
                    sub_node,
                    cast_node,
                    slice_node,
                    unsqueeze_node,
                ) = mask_path
                if self.has_constant_input(mul_node, -10000) and self.has_constant_input(sub_node, 1):
                    graph_inputs = self.get_graph_inputs(sub_node, recursive=True)
                    inputs = [input for input in graph_inputs if input not in excluded_graph_inputs]
                    if len(inputs) > 1:
                        print("Found multiple candidates of mask input", inputs)
                        return None
                    if len(inputs) == 1:
                        return inputs[0]
                    # Duplicated input found. Try to simplify the graph.
                    path_to_be_simplified = self.match_parent_path(
                        mask_path[-1],
                        [
                            "ConstantOfShape",
                            "Cast",
                            "Concat",
                            "Unsqueeze",
                            "Squeeze",
                            "Slice",
                            "Cast",
                            "Shape",
                        ],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                    )
                    duplicated_inputs = [input for input in graph_inputs if input in excluded_graph_inputs]
                    # Simplify graph for dynamic axes.
                    if (
                        path_to_be_simplified
                        and duplicated_inputs
                        and len(duplicated_inputs) == 1
                        and duplicated_inputs[0] == path_to_be_simplified[-1].input[0]
                    ):
                        logger.debug("Simplify semgent id path...")
                        constantofshape_node = path_to_be_simplified[0]
                        constantofshape_value = helper.get_attribute_value(constantofshape_node.attribute[0])
                        graph_name = self.get_graph_by_node(constantofshape_node).name
                        self.add_node(
                            helper.make_node(
                                "Shape",
                                inputs=[duplicated_inputs[0]],
                                outputs=["input_shape_for_mask"],
                            ),
                            graph_name,
                        )
                        self.add_node(
                            helper.make_node(
                                "ConstantOfShape",
                                inputs=["input_shape_for_mask"],
                                outputs=[unsqueeze_node.input[0]],
                                value=constantofshape_value,
                            ),
                            graph_name,
                        )
                    return unsqueeze_node.input[0]
        return None

    def create_embedding_subgraph(self, normalize_node, word_embedding, segment_embedding, position_embedding):
        input_ids = self.find_input_ids(word_embedding)
        if input_ids is None:
            logger.info("Failed to find input_ids. Cannot fuse embedding layer.")
            return False

        segment_ids = self.find_segment_ids(segment_embedding, input_ids)
        if segment_ids is None:
            logger.info("Failed to find segment_ids. Cannot fuse embedding layer.")
            return False

        mask_input = self.find_mask_input([segment_ids, input_ids])
        if mask_input is None:
            logger.info("Failed to find input_mask. Cannot fuse embedding layer.")
            return False

        self.bert_inputs = [input_ids, segment_ids, mask_input]

        mask_index = self.create_node_name("mask_index")
        self.attention_mask.set_mask_indice(mask_input, mask_index)

        if self.find_graph_input(input_ids).type.tensor_type.elem_type != TensorProto.INT32:
            casted, input_ids = self.utils.cast_graph_input_to_int32(input_ids)

        if self.find_graph_input(segment_ids):
            casted, segment_ids = self.utils.cast_graph_input_to_int32(segment_ids)
        else:
            segment_ids, segment_id_cast_node = self.utils.cast_input_to_int32(segment_ids)

        if self.find_graph_input(mask_input):
            casted, mask_input = self.utils.cast_graph_input_to_int32(mask_input)
        else:
            mask_input, mask_input_cast_node = self.utils.cast_input_to_int32(mask_input)

        embed_output = self.create_node_name("embed_output")
        embed_node = onnx.helper.make_node(
            "EmbedLayerNormalization",
            inputs=[
                input_ids,
                segment_ids,
                word_embedding,
                position_embedding,
                segment_embedding,
                normalize_node.input[1],  # gamma
                normalize_node.input[2],  # beta
                mask_input,
            ],
            outputs=[embed_output, mask_index],
            name="EmbedLayer",
        )
        embed_node.domain = "com.microsoft"
        self.replace_input_of_all_nodes(normalize_node.output[0], embed_output)
        self.add_node(embed_node, self.get_graph_by_node(normalize_node).name)

    def process_embedding(self):
        """
        Automatically detect word, segment and position embeddings.
        """
        logger.info("start processing embedding layer...")
        output_name_to_node = self.output_name_to_node()

        layer_norm_nodes = self.get_nodes_by_op_type("LayerNormalization")
        for layer_norm_node in layer_norm_nodes:
            pos_embed_path = self.match_parent_path(
                layer_norm_node,
                ["Add", "Reshape", "Slice"],
                [0, 1, 0],
                output_name_to_node,
            )
            if pos_embed_path is None:
                continue

            add_node, reshape_node, slice_node = pos_embed_path
            initializer = self.get_initializer(slice_node.input[0])
            if initializer is None:
                continue

            temp = numpy_helper.to_array(initializer)
            if len(temp.shape) == 2:
                logger.info(f"Found position embedding. name:{initializer.name}, shape:{temp.shape}")
                position_embedding = initializer.name
            else:
                logger.info(f"Failed to find position embedding. name:{initializer.name}, shape:{temp.shape}")
                return

            first_parent = self.get_parent(add_node, 0, output_name_to_node)
            if first_parent is not None and first_parent.op_type == "Add":
                embeddings = self.get_2d_initializers_from_parent_subgraphs(first_parent)
                if len(embeddings) != 2:
                    logger.warning(
                        f"Failed to find two embeddings (word and segment) from Add node. Found {embeddings}"
                    )
                    return

                word_embedding = None
                segment_embedding = None
                for name, shape in embeddings.items():
                    if shape[0] == 2:
                        segment_embedding = name
                        logger.info(f"Found segment embedding. name:{name}, shape:{shape}")
                    else:
                        word_embedding = name
                        logger.info(f"Found words embedding. name:{name}, shape:{shape}")

                if word_embedding is None or segment_embedding is None:
                    logger.info("Failed to find both word and segment embedding")
                    return

                logger.info("Create Embedding node")
                self.create_embedding_subgraph(
                    layer_norm_node,
                    word_embedding,
                    segment_embedding,
                    position_embedding,
                )
                # Prune graph to remove those original embedding nodes.
                self.prune_graph()
                break

    def check_attention_input(self, matmul_q, matmul_k, matmul_v, parent, output_name_to_node):
        for x in [matmul_q, matmul_k, matmul_v]:
            root_input = x.input[0]
            root_node = output_name_to_node[root_input]
            if root_node == parent:
                continue
            logger.debug(f"Check attention input failed:{root_input}, {parent.output[0]}")
            return False

        return True

    def fuse_attention(self):
        output_name_to_node = self.output_name_to_node()

        nodes_to_remove = []
        attention_count = 0

        start_nodes = []
        skip_layer_norm_nodes = self.get_nodes_by_op_type("SkipLayerNormalization")
        layer_norm_nodes = self.get_nodes_by_op_type("LayerNormalization")
        # Sometimes we can not fuse skiplayernormalization since the add before layernorm has an output that used by nodes outside skiplayernorm
        # Conceptually we treat add before layernorm as skiplayernorm node since they share the same pattern
        start_nodes.extend(skip_layer_norm_nodes)
        start_nodes.extend(layer_norm_nodes)

        for normalize_node in start_nodes:
            graph_name = self.get_graph_by_node(normalize_node).name
            # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
            if normalize_node.op_type == "LayerNormalization":
                add_before_layernorm = self.match_parent(normalize_node, "Add", 0)
                if add_before_layernorm is not None:
                    normalize_node = add_before_layernorm  # noqa: PLW2901
                else:
                    continue
            parent = self.get_parent(normalize_node, 1)
            if parent is None or parent.op_type not in [
                "SkipLayerNormalization",
                "LayerNormalization",
                "Reshape",
            ]:
                parent = self.get_parent(normalize_node, 0)
                if parent is None or parent.op_type not in [
                    "SkipLayerNormalization",
                    "LayerNormalization",
                    "Reshape",
                ]:
                    logger.debug("Failed to match parent of normalize_node")
                    continue

            qkv_nodes = self.match_parent_path(
                normalize_node,
                ["Add", "MatMul", "Reshape", "Transpose", "MatMul"],
                [0, 0, 0, 0, 0],
            )
            if qkv_nodes is None:
                qkv_nodes = self.match_parent_path(
                    normalize_node,
                    ["MatMul", "Reshape", "Transpose", "MatMul"],
                    [1, 0, 0, 0],
                )
                if qkv_nodes is None:
                    qkv_nodes = self.match_parent_path(normalize_node, ["Add", "Einsum", "Einsum"], [0, 0, 0])
                    if qkv_nodes is None:
                        logger.debug("Failed to match qkv nodes")
                        continue

            matmul_qkv = qkv_nodes[-1]
            v_nodes = self.match_parent_path(matmul_qkv, ["Transpose", "Reshape", "Add", "MatMul"], [1, 0, 0, 0])
            if v_nodes is None:
                v_nodes = self.match_parent_path(matmul_qkv, ["Add", "Einsum"], [1, 0])
                if v_nodes is None:
                    logger.debug("Failed to match v path")
                    continue

            add_v = v_nodes[-2]
            matmul_v = v_nodes[-1]
            qk_nodes = self.match_parent_path(matmul_qkv, ["Softmax", "Add", "Mul", "MatMul"], [0, 0, 0, 0])
            if qk_nodes is None:
                qk_nodes = self.match_parent_path(matmul_qkv, ["Softmax", "Add", "Einsum"], [0, 0, 0])
                if qk_nodes is None:
                    logger.debug("Failed to match qk_paths")
                    continue
            matmul_qk = qk_nodes[-1]

            q_nodes = self.match_parent_path(matmul_qk, ["Transpose", "Reshape", "Add", "MatMul"], [0, 0, 0, 0])
            if q_nodes is None:
                q_nodes = self.match_parent_path(matmul_qk, ["Add", "Einsum"], [0, 0])
                if q_nodes is None:
                    logger.debug("Failed to match q path")
                    continue

            add_q = q_nodes[-2]
            matmul_q = q_nodes[-1]

            k_nodes = self.match_parent_path(matmul_qk, ["Transpose", "Reshape", "Add", "MatMul"], [1, 0, 0, 0])
            if k_nodes is None:
                k_nodes = self.match_parent_path(matmul_qk, ["Mul", "Add", "Einsum"], [1, 0, 0])
                if k_nodes is None:
                    logger.debug("Failed to match k path")
                    continue
            add_k = k_nodes[-2]
            matmul_k = k_nodes[-1]

            mask_nodes = self.match_mask_path(qk_nodes[1])

            if mask_nodes is None:
                logger.debug("Cannot find mask_nodes.")
                continue

            if not self.has_constant_input(mask_nodes[1], 1):
                logger.debug("Sub node expected to have an input with constant value 1.0.")
                continue

            # add a squeeze node to convert a 3-d mask to 2-d
            squeeze_node = self.match_parent_path(mask_nodes[-1], ["Squeeze"], [0]) or self.match_parent_path(
                mask_nodes[-1], ["Expand"], [0]
            )
            squeeze_node_name = "Squeeze_3d_to_2d_mask"
            squeeze_output_name = squeeze_node_name + "_output"
            if squeeze_node is None and len(mask_nodes) == 5 and self.find_graph_input(mask_nodes[-1].input[0]) is None:
                mask_input = mask_nodes[-1].input[1]
                self.add_node(
                    helper.make_node(
                        "Squeeze",
                        [mask_input],
                        [squeeze_output_name],
                        squeeze_node_name,
                        axes=[1],
                    ),
                    graph_name,
                )
                mask_nodes[-1].input[0] = squeeze_output_name

            is_same_root = self.check_attention_input(matmul_q, matmul_k, matmul_v, parent, output_name_to_node)
            if is_same_root:
                mask_index = self.attention_mask.process_mask(mask_nodes[-1].input[0])
                logger.debug("Create an Attention node.")

                # For tf models, q and v are flipped.
                attention_node = self.attention_fusion.create_attention_node(
                    mask_index=mask_index,
                    q_matmul=matmul_k,
                    k_matmul=matmul_q,
                    v_matmul=matmul_v,
                    q_add=add_k,
                    k_add=add_q,
                    v_add=add_v,
                    num_heads=self.num_heads,
                    hidden_size=self.hidden_size,
                    first_input=parent.output[0],
                    output=qkv_nodes[2].output[0],
                )
                if attention_node is None:
                    continue

                if qkv_nodes[1].op_type == "Einsum":
                    # add reshape before einsum
                    tensor = helper.make_tensor(
                        name=qkv_nodes[1].name + "_newshape",
                        data_type=TensorProto.INT64,
                        dims=[4],
                        vals=np.int64(
                            [
                                [
                                    0,
                                    0,
                                    self.num_heads,
                                    int(self.hidden_size / self.num_heads),
                                ]
                            ]
                        ).tobytes(),
                        raw=True,
                    )
                    self.add_initializer(tensor, graph_name)
                    reshape_ = helper.make_node(
                        "Reshape",
                        inputs=[
                            attention_node.output[0],
                            qkv_nodes[1].name + "_newshape",
                        ],
                        outputs=[qkv_nodes[1].name + "_reshape_output"],
                        name=qkv_nodes[1].name + "_reshape",
                    )
                    qkv_nodes[1].input[0] = qkv_nodes[1].name + "_reshape_output"
                    self.add_node(reshape_, graph_name)
                if parent.op_type == "Reshape":
                    # Temporary work around: we require the skiplayernorm and attention op be fed with 3-d input
                    hidden_size = numpy_helper.to_array(self.get_initializer(parent.input[1]))[1]
                    tensor = helper.make_tensor(
                        name=parent.name + "_modified",
                        data_type=TensorProto.INT64,
                        dims=[3],
                        vals=np.int64([[1, -1, hidden_size]]).tobytes(),
                        raw=True,
                    )
                    self.add_initializer(tensor, graph_name)
                    parent.input[1] = parent.name + "_modified"

                self.add_node(attention_node, graph_name)
                attention_count += 1

                nodes_to_remove.extend(qkv_nodes[2:])
                nodes_to_remove.extend(qk_nodes)
                nodes_to_remove.extend(q_nodes)
                nodes_to_remove.extend(k_nodes)
                nodes_to_remove.extend(v_nodes)
                nodes_to_remove.extend(mask_nodes)
            else:
                logger.debug("Root node not matched.")
                continue
        self.remove_nodes(nodes_to_remove)
        self.update_graph()
        logger.info(f"Fused Attention count:{attention_count}")

    def preprocess(self):
        self.remove_identity()
        self.process_embedding()
        self.skip_reshape()

    def skip_reshape(self):
        count = 0
        reshape_nodes = self.get_nodes_by_op_type("Reshape")
        for reshape_node in reshape_nodes:
            parent = self.get_parent(reshape_node, 0)
            if parent is not None and parent.op_type == "Reshape":
                reshape_node.input[0] = parent.input[0]
                count += 1

        if count > 0:
            logger.info(f"Skip consequent Reshape count: {count}")

    def remove_reshape_before_first_attention(self):
        attention_nodes = self.get_nodes_by_op_type("Attention")
        for attention_node in attention_nodes:
            path = self.match_parent_path(attention_node, ["Reshape", "EmbedLayerNormalization"], [0, 0])
            if path is None:
                continue
            logger.info("Remove Reshape before first Attention node.")
            reshape, _ = path
            self.replace_input_of_all_nodes(reshape.output[0], reshape.input[0])
            self.remove_node(reshape)
            break

    def postprocess(self):
        self.remove_reshape_before_first_attention()
        self.prune_graph()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_channel.py ===
from __future__ import annotations

import sys
from collections import OrderedDict, deque
from collections.abc import AsyncGenerator, Callable  # noqa: TC003  # Needed for Sphinx
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from functools import wraps
from math import inf
from typing import (
    TYPE_CHECKING,
    Generic,
)

import attrs
from outcome import Error, Value

import trio

from ._abc import ReceiveChannel, ReceiveType, SendChannel, SendType, T
from ._core import Abort, RaiseCancelT, Task, enable_ki_protection
from ._util import (
    MultipleExceptionError,
    NoPublicConstructor,
    final,
    generic_function,
    raise_single_exception_from_group,
)

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup

if TYPE_CHECKING:
    from types import TracebackType

    from typing_extensions import ParamSpec, Self

    P = ParamSpec("P")
elif "sphinx" in sys.modules:
    # P needs to exist for Sphinx to parse the type hints successfully.
    try:
        from typing_extensions import ParamSpec
    except ImportError:
        P = ...  # This is valid in Callable, though not correct
    else:
        P = ParamSpec("P")


def _open_memory_channel(
    max_buffer_size: int | float,  # noqa: PYI041
) -> tuple[MemorySendChannel[T], MemoryReceiveChannel[T]]:
    """Open a channel for passing objects between tasks within a process.

    Memory channels are lightweight, cheap to allocate, and entirely
    in-memory. They don't involve any operating-system resources, or any kind
    of serialization. They just pass Python objects directly between tasks
    (with a possible stop in an internal buffer along the way).

    Channel objects can be closed by calling `~trio.abc.AsyncResource.aclose`
    or using ``async with``. They are *not* automatically closed when garbage
    collected. Closing memory channels isn't mandatory, but it is generally a
    good idea, because it helps avoid situations where tasks get stuck waiting
    on a channel when there's no-one on the other side. See
    :ref:`channel-shutdown` for details.

    Memory channel operations are all atomic with respect to
    cancellation, either `~trio.abc.ReceiveChannel.receive` will
    successfully return an object, or it will raise :exc:`Cancelled`
    while leaving the channel unchanged.

    Args:
      max_buffer_size (int or math.inf): The maximum number of items that can
        be buffered in the channel before :meth:`~trio.abc.SendChannel.send`
        blocks. Choosing a sensible value here is important to ensure that
        backpressure is communicated promptly and avoid unnecessary latency;
        see :ref:`channel-buffering` for more details. If in doubt, use 0.

    Returns:
      A pair ``(send_channel, receive_channel)``. If you have
      trouble remembering which order these go in, remember: data
      flows from left → right.

    In addition to the standard channel methods, all memory channel objects
    provide a ``statistics()`` method, which returns an object with the
    following fields:

    * ``current_buffer_used``: The number of items currently stored in the
      channel buffer.
    * ``max_buffer_size``: The maximum number of items allowed in the buffer,
      as passed to :func:`open_memory_channel`.
    * ``open_send_channels``: The number of open
      :class:`MemorySendChannel` endpoints pointing to this channel.
      Initially 1, but can be increased by
      :meth:`MemorySendChannel.clone`.
    * ``open_receive_channels``: Likewise, but for open
      :class:`MemoryReceiveChannel` endpoints.
    * ``tasks_waiting_send``: The number of tasks blocked in ``send`` on this
      channel (summing over all clones).
    * ``tasks_waiting_receive``: The number of tasks blocked in ``receive`` on
      this channel (summing over all clones).

    """
    if max_buffer_size != inf and not isinstance(max_buffer_size, int):
        raise TypeError("max_buffer_size must be an integer or math.inf")
    if max_buffer_size < 0:
        raise ValueError("max_buffer_size must be >= 0")
    state: MemoryChannelState[T] = MemoryChannelState(max_buffer_size)
    return (
        MemorySendChannel[T]._create(state),
        MemoryReceiveChannel[T]._create(state),
    )


# This workaround requires python3.9+, once older python versions are not supported
# or there's a better way of achieving type-checking on a generic factory function,
# it could replace the normal function header
if TYPE_CHECKING:
    # written as a class so you can say open_memory_channel[int](5)
    class open_memory_channel(tuple["MemorySendChannel[T]", "MemoryReceiveChannel[T]"]):
        def __new__(  # type: ignore[misc]  # "must return a subtype"
            cls,
            max_buffer_size: int | float,  # noqa: PYI041
        ) -> tuple[MemorySendChannel[T], MemoryReceiveChannel[T]]:
            return _open_memory_channel(max_buffer_size)

        def __init__(self, max_buffer_size: int | float) -> None:  # noqa: PYI041
            ...

else:
    # apply the generic_function decorator to make open_memory_channel indexable
    # so it's valid to say e.g. ``open_memory_channel[bytes](5)`` at runtime
    open_memory_channel = generic_function(_open_memory_channel)


@attrs.frozen
class MemoryChannelStatistics:
    current_buffer_used: int
    max_buffer_size: int | float
    open_send_channels: int
    open_receive_channels: int
    tasks_waiting_send: int
    tasks_waiting_receive: int


@attrs.define
class MemoryChannelState(Generic[T]):
    max_buffer_size: int | float
    data: deque[T] = attrs.Factory(deque)
    # Counts of open endpoints using this state
    open_send_channels: int = 0
    open_receive_channels: int = 0
    # {task: value}
    send_tasks: OrderedDict[Task, T] = attrs.Factory(OrderedDict)
    # {task: None}
    receive_tasks: OrderedDict[Task, None] = attrs.Factory(OrderedDict)

    def statistics(self) -> MemoryChannelStatistics:
        return MemoryChannelStatistics(
            current_buffer_used=len(self.data),
            max_buffer_size=self.max_buffer_size,
            open_send_channels=self.open_send_channels,
            open_receive_channels=self.open_receive_channels,
            tasks_waiting_send=len(self.send_tasks),
            tasks_waiting_receive=len(self.receive_tasks),
        )


@final
@attrs.define(eq=False, repr=False, slots=False)
class MemorySendChannel(SendChannel[SendType], metaclass=NoPublicConstructor):
    _state: MemoryChannelState[SendType]
    _closed: bool = False
    # This is just the tasks waiting on *this* object. As compared to
    # self._state.send_tasks, which includes tasks from this object and
    # all clones.
    _tasks: set[Task] = attrs.Factory(set)

    def __attrs_post_init__(self) -> None:
        self._state.open_send_channels += 1

    def __repr__(self) -> str:
        return f"<send channel at {id(self):#x}, using buffer at {id(self._state):#x}>"

    def statistics(self) -> MemoryChannelStatistics:
        """Returns a `MemoryChannelStatistics` for the memory channel this is
        associated with."""
        # XX should we also report statistics specific to this object?
        return self._state.statistics()

    @enable_ki_protection
    def send_nowait(self, value: SendType) -> None:
        """Like `~trio.abc.SendChannel.send`, but if the channel's buffer is
        full, raises `WouldBlock` instead of blocking.

        """
        if self._closed:
            raise trio.ClosedResourceError
        if self._state.open_receive_channels == 0:
            raise trio.BrokenResourceError
        if self._state.receive_tasks:
            assert not self._state.data
            task, _ = self._state.receive_tasks.popitem(last=False)
            task.custom_sleep_data._tasks.remove(task)
            trio.lowlevel.reschedule(task, Value(value))
        elif len(self._state.data) < self._state.max_buffer_size:
            self._state.data.append(value)
        else:
            raise trio.WouldBlock

    @enable_ki_protection
    async def send(self, value: SendType) -> None:
        """See `SendChannel.send <trio.abc.SendChannel.send>`.

        Memory channels allow multiple tasks to call `send` at the same time.

        """
        await trio.lowlevel.checkpoint_if_cancelled()
        try:
            self.send_nowait(value)
        except trio.WouldBlock:
            pass
        else:
            await trio.lowlevel.cancel_shielded_checkpoint()
            return

        task = trio.lowlevel.current_task()
        self._tasks.add(task)
        self._state.send_tasks[task] = value
        task.custom_sleep_data = self

        def abort_fn(_: RaiseCancelT) -> Abort:
            self._tasks.remove(task)
            del self._state.send_tasks[task]
            return trio.lowlevel.Abort.SUCCEEDED

        await trio.lowlevel.wait_task_rescheduled(abort_fn)

    # Return type must be stringified or use a TypeVar
    @enable_ki_protection
    def clone(self) -> MemorySendChannel[SendType]:
        """Clone this send channel object.

        This returns a new `MemorySendChannel` object, which acts as a
        duplicate of the original: sending on the new object does exactly the
        same thing as sending on the old object. (If you're familiar with
        `os.dup`, then this is a similar idea.)

        However, closing one of the objects does not close the other, and
        receivers don't get `EndOfChannel` until *all* clones have been
        closed.

        This is useful for communication patterns that involve multiple
        producers all sending objects to the same destination. If you give
        each producer its own clone of the `MemorySendChannel`, and then make
        sure to close each `MemorySendChannel` when it's finished, receivers
        will automatically get notified when all producers are finished. See
        :ref:`channel-mpmc` for examples.

        Raises:
          trio.ClosedResourceError: if you already closed this
              `MemorySendChannel` object.

        """
        if self._closed:
            raise trio.ClosedResourceError
        return MemorySendChannel._create(self._state)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @enable_ki_protection
    def close(self) -> None:
        """Close this send channel object synchronously.

        All channel objects have an asynchronous `~.AsyncResource.aclose` method.
        Memory channels can also be closed synchronously. This has the same
        effect on the channel and other tasks using it, but `close` is not a
        trio checkpoint. This simplifies cleaning up in cancelled tasks.

        Using ``with send_channel:`` will close the channel object on leaving
        the with block.

        """
        if self._closed:
            return
        self._closed = True
        for task in self._tasks:
            trio.lowlevel.reschedule(task, Error(trio.ClosedResourceError()))
            del self._state.send_tasks[task]
        self._tasks.clear()
        self._state.open_send_channels -= 1
        if self._state.open_send_channels == 0:
            assert not self._state.send_tasks
            for task in self._state.receive_tasks:
                task.custom_sleep_data._tasks.remove(task)
                trio.lowlevel.reschedule(task, Error(trio.EndOfChannel()))
            self._state.receive_tasks.clear()

    @enable_ki_protection
    async def aclose(self) -> None:
        """Close this send channel object asynchronously.

        See `MemorySendChannel.close`."""
        self.close()
        await trio.lowlevel.checkpoint()


@final
@attrs.define(eq=False, repr=False, slots=False)
class MemoryReceiveChannel(ReceiveChannel[ReceiveType], metaclass=NoPublicConstructor):
    _state: MemoryChannelState[ReceiveType]
    _closed: bool = False
    _tasks: set[trio._core._run.Task] = attrs.Factory(set)

    def __attrs_post_init__(self) -> None:
        self._state.open_receive_channels += 1

    def statistics(self) -> MemoryChannelStatistics:
        """Returns a `MemoryChannelStatistics` for the memory channel this is
        associated with."""
        return self._state.statistics()

    def __repr__(self) -> str:
        return (
            f"<receive channel at {id(self):#x}, using buffer at {id(self._state):#x}>"
        )

    @enable_ki_protection
    def receive_nowait(self) -> ReceiveType:
        """Like `~trio.abc.ReceiveChannel.receive`, but if there's nothing
        ready to receive, raises `WouldBlock` instead of blocking.

        """
        if self._closed:
            raise trio.ClosedResourceError
        if self._state.send_tasks:
            task, value = self._state.send_tasks.popitem(last=False)
            task.custom_sleep_data._tasks.remove(task)
            trio.lowlevel.reschedule(task)
            self._state.data.append(value)
            # Fall through
        if self._state.data:
            return self._state.data.popleft()
        if not self._state.open_send_channels:
            raise trio.EndOfChannel
        raise trio.WouldBlock

    @enable_ki_protection
    async def receive(self) -> ReceiveType:
        """See `ReceiveChannel.receive <trio.abc.ReceiveChannel.receive>`.

        Memory channels allow multiple tasks to call `receive` at the same
        time. The first task will get the first item sent, the second task
        will get the second item sent, and so on.

        """
        await trio.lowlevel.checkpoint_if_cancelled()
        try:
            value = self.receive_nowait()
        except trio.WouldBlock:
            pass
        else:
            await trio.lowlevel.cancel_shielded_checkpoint()
            return value

        task = trio.lowlevel.current_task()
        self._tasks.add(task)
        self._state.receive_tasks[task] = None
        task.custom_sleep_data = self

        def abort_fn(_: RaiseCancelT) -> Abort:
            self._tasks.remove(task)
            del self._state.receive_tasks[task]
            return trio.lowlevel.Abort.SUCCEEDED

        # Not strictly guaranteed to return ReceiveType, but will do so unless
        # you intentionally reschedule with a bad value.
        return await trio.lowlevel.wait_task_rescheduled(abort_fn)  # type: ignore[no-any-return]

    @enable_ki_protection
    def clone(self) -> MemoryReceiveChannel[ReceiveType]:
        """Clone this receive channel object.

        This returns a new `MemoryReceiveChannel` object, which acts as a
        duplicate of the original: receiving on the new object does exactly
        the same thing as receiving on the old object.

        However, closing one of the objects does not close the other, and the
        underlying channel is not closed until all clones are closed. (If
        you're familiar with `os.dup`, then this is a similar idea.)

        This is useful for communication patterns that involve multiple
        consumers all receiving objects from the same underlying channel. See
        :ref:`channel-mpmc` for examples.

        .. warning:: The clones all share the same underlying channel.
           Whenever a clone :meth:`receive`\\s a value, it is removed from the
           channel and the other clones do *not* receive that value. If you
           want to send multiple copies of the same stream of values to
           multiple destinations, like :func:`itertools.tee`, then you need to
           find some other solution; this method does *not* do that.

        Raises:
          trio.ClosedResourceError: if you already closed this
              `MemoryReceiveChannel` object.

        """
        if self._closed:
            raise trio.ClosedResourceError
        return MemoryReceiveChannel._create(self._state)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @enable_ki_protection
    def close(self) -> None:
        """Close this receive channel object synchronously.

        All channel objects have an asynchronous `~.AsyncResource.aclose` method.
        Memory channels can also be closed synchronously. This has the same
        effect on the channel and other tasks using it, but `close` is not a
        trio checkpoint. This simplifies cleaning up in cancelled tasks.

        Using ``with receive_channel:`` will close the channel object on
        leaving the with block.

        """
        if self._closed:
            return
        self._closed = True
        for task in self._tasks:
            trio.lowlevel.reschedule(task, Error(trio.ClosedResourceError()))
            del self._state.receive_tasks[task]
        self._tasks.clear()
        self._state.open_receive_channels -= 1
        if self._state.open_receive_channels == 0:
            assert not self._state.receive_tasks
            for task in self._state.send_tasks:
                task.custom_sleep_data._tasks.remove(task)
                trio.lowlevel.reschedule(task, Error(trio.BrokenResourceError()))
            self._state.send_tasks.clear()
            self._state.data.clear()

    @enable_ki_protection
    async def aclose(self) -> None:
        """Close this receive channel object asynchronously.

        See `MemoryReceiveChannel.close`."""
        self.close()
        await trio.lowlevel.checkpoint()


class RecvChanWrapper(ReceiveChannel[T]):
    def __init__(
        self, recv_chan: MemoryReceiveChannel[T], send_semaphore: trio.Semaphore
    ) -> None:
        self._recv_chan = recv_chan
        self._send_semaphore = send_semaphore

    async def receive(self) -> T:
        self._send_semaphore.release()
        return await self._recv_chan.receive()

    async def aclose(self) -> None:
        await self._recv_chan.aclose()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._recv_chan.close()


def as_safe_channel(
    fn: Callable[P, AsyncGenerator[T, None]],
) -> Callable[P, AbstractAsyncContextManager[ReceiveChannel[T]]]:
    """Decorate an async generator function to make it cancellation-safe.

    The ``yield`` keyword offers a very convenient way to write iterators...
    which makes it really unfortunate that async generators are so difficult
    to call correctly.  Yielding from the inside of a cancel scope or a nursery
    to the outside `violates structured concurrency <https://xkcd.com/292/>`_
    with consequences explained in :pep:`789`.  Even then, resource cleanup
    errors remain common (:pep:`533`) unless you wrap every call in
    :func:`~contextlib.aclosing`.

    This decorator gives you the best of both worlds: with careful exception
    handling and a background task we preserve structured concurrency by
    offering only the safe interface, and you can still write your iterables
    with the convenience of ``yield``.  For example::

        @as_safe_channel
        async def my_async_iterable(arg, *, kwarg=True):
            while ...:
                item = await ...
                yield item

        async with my_async_iterable(...) as recv_chan:
            async for item in recv_chan:
                ...

    While the combined async-with-async-for can be inconvenient at first,
    the context manager is indispensable for both correctness and for prompt
    cleanup of resources.
    """
    # Perhaps a future PEP will adopt `async with for` syntax, like
    # https://coconut.readthedocs.io/en/master/DOCS.html#async-with-for

    @asynccontextmanager
    @wraps(fn)
    async def context_manager(
        *args: P.args, **kwargs: P.kwargs
    ) -> AsyncGenerator[trio._channel.RecvChanWrapper[T], None]:
        send_chan, recv_chan = trio.open_memory_channel[T](0)
        try:
            async with trio.open_nursery(strict_exception_groups=True) as nursery:
                agen = fn(*args, **kwargs)
                send_semaphore = trio.Semaphore(0)
                # `nursery.start` to make sure that we will clean up send_chan & agen
                # If this errors we don't close `recv_chan`, but the caller
                # never gets access to it, so that's not a problem.
                await nursery.start(
                    _move_elems_to_channel, agen, send_chan, send_semaphore
                )
                # `async with recv_chan` could eat exceptions, so use sync cm
                with RecvChanWrapper(recv_chan, send_semaphore) as wrapped_recv_chan:
                    yield wrapped_recv_chan
                # User has exited context manager, cancel to immediately close the
                # abandoned generator if it's still alive.
                nursery.cancel_scope.cancel()
        except BaseExceptionGroup as eg:
            try:
                raise_single_exception_from_group(eg)
            except MultipleExceptionError:
                # In case user has except* we make it possible for them to handle the
                # exceptions.
                raise BaseExceptionGroup(
                    "Encountered exception during cleanup of generator object, as well as exception in the contextmanager body - unable to unwrap.",
                    [eg],
                ) from None

    async def _move_elems_to_channel(
        agen: AsyncGenerator[T, None],
        send_chan: trio.MemorySendChannel[T],
        send_semaphore: trio.Semaphore,
        task_status: trio.TaskStatus,
    ) -> None:
        # `async with send_chan` will eat exceptions,
        # see https://github.com/python-trio/trio/issues/1559
        with send_chan:
            try:
                task_status.started()
                while True:
                    # wait for receiver to call next on the aiter
                    await send_semaphore.acquire()
                    try:
                        value = await agen.__anext__()
                    except StopAsyncIteration:
                        return
                    # Send the value to the channel
                    await send_chan.send(value)
            finally:
                # replace try-finally with contextlib.aclosing once python39 is dropped
                await agen.aclose()

    return context_manager

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\stable_diffusion\optimize_pipeline.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
#
# This script converts stable diffusion onnx models from float to half (mixed) precision for GPU inference.
#
# Before running this script, follow README.md to setup python environment and convert stable diffusion checkpoint
# to float32 onnx models.
#
# For example, the float32 ONNX pipeline is saved to ./sd-v1-5 directory, you can optimize and convert it to float16
# like the following:
#    python optimize_pipeline.py -i ./sd-v1-5 -o ./sd-v1-5-fp16 --float16
#
# Note that the optimizations are carried out for CUDA Execution Provider at first, other EPs may not have the support
# for the fused operators. The users could disable the operator fusion manually to workaround.

import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path

import __init__  # noqa: F401. Walk-around to run this script directly
import coloredlogs
import onnx
from fusion_options import FusionOptions
from onnx_model_clip import ClipOnnxModel
from onnx_model_mmdit import MmditOnnxModel
from onnx_model_t5 import T5OnnxModel
from onnx_model_unet import UnetOnnxModel
from onnx_model_vae import VaeOnnxModel
from optimizer import optimize_by_onnxruntime, optimize_model
from packaging import version

import onnxruntime

logger = logging.getLogger(__name__)


def has_external_data(onnx_model_path):
    original_model = onnx.load_model(str(onnx_model_path), load_external_data=False)
    for initializer in original_model.graph.initializer:
        if initializer.HasField("data_location") and initializer.data_location == onnx.TensorProto.EXTERNAL:
            return True
    return False


def is_sd_3(source_dir: Path):
    return (source_dir / "text_encoder_3").exists()


def is_sdxl(source_dir: Path):
    return (
        (source_dir / "text_encoder_2").exists()
        and not (source_dir / "text_encoder_3").exists()
        and not (source_dir / "transformer").exists()
    )


def is_flux(source_dir: Path):
    return (
        (source_dir / "text_encoder_2").exists()
        and not (source_dir / "text_encoder_3").exists()
        and (source_dir / "transformer").exists()
    )


def _classify_pipeline_type(source_dir: Path):
    # May also check _class_name in model_index.json like `StableDiffusion3Pipeline` or `FluxPipeline` etc to classify.
    if is_sd_3(source_dir):
        return "sd3"

    if is_flux(source_dir):
        return "flux"

    if is_sdxl(source_dir):
        return "sdxl"

    # sd 1.x and 2.x
    return "sd"


def _get_model_list(pipeline_type: str):
    if pipeline_type == "sd3":
        return ["text_encoder", "text_encoder_2", "text_encoder_3", "transformer", "vae_encoder", "vae_decoder"]

    if pipeline_type == "flux":
        return ["text_encoder", "text_encoder_2", "transformer", "vae_encoder", "vae_decoder"]

    if pipeline_type == "sdxl":
        return ["text_encoder", "text_encoder_2", "unet", "vae_encoder", "vae_decoder"]

    assert pipeline_type == "sd"
    return ["text_encoder", "unet", "vae_encoder", "vae_decoder"]


def _optimize_sd_pipeline(
    source_dir: Path,
    target_dir: Path,
    pipeline_type: str,
    model_list: list[str],
    use_external_data_format: bool | None,
    float16: bool,
    bfloat16: bool,
    force_fp32_ops: list[str],
    enable_runtime_optimization: bool,
    args,
):
    """Optimize onnx models used in stable diffusion onnx pipeline and optionally convert to float16.

    Args:
        source_dir (Path): Root of input directory of stable diffusion onnx pipeline with float32 models.
        target_dir (Path): Root of output directory of stable diffusion onnx pipeline with optimized models.
        model_list (List[str]): list of directory names with onnx model.
        use_external_data_format (Optional[bool]): use external data format.
        float16 (bool): use half precision
        bfloat16 (bool): use bfloat16 as fallback if float16 is also provided.
        force_fp32_ops(List[str]): operators that are forced to run in float32.
        enable_runtime_optimization(bool): run graph optimization using Onnx Runtime.

    Raises:
        RuntimeError: input onnx model does not exist
        RuntimeError: output onnx model path existed
    """
    is_flux_pipeline = pipeline_type == "flux"
    model_type_mapping = {
        "transformer": "mmdit",
        "unet": "unet",
        "vae_encoder": "vae",
        "vae_decoder": "vae",
        "text_encoder": "clip",
        "text_encoder_2": "t5" if is_flux_pipeline else "clip",
        "text_encoder_3": "t5",  # t5-v1_1-xxl is used in SD 3.x text_encoder_3 and Flux text_encoder_2.
        "safety_checker": "unet",
    }

    model_type_class_mapping = {
        "unet": UnetOnnxModel,
        "vae": VaeOnnxModel,
        "clip": ClipOnnxModel,
        "t5": T5OnnxModel,
        "mmdit": MmditOnnxModel,
    }

    force_fp32_operators = {
        "unet": [],
        "vae_encoder": [],
        "vae_decoder": [],
        "text_encoder": [],
        "text_encoder_2": [],
        "safety_checker": [],
        "text_encoder_3": [],
        "transformer": [],
    }

    # The node block list is generated by running the fp32 model and get statistics of node inputs and outputs.
    # Nodes with any input or output of float or double data type, but value ouf of range of float16 are candidates.
    #   python optimize_pipeline.py -i ./flux1_schnell_onnx/fp32 -o ./flux1_schnell_onnx/fp32_opt
    #   export ORT_DEBUG_NODE_IO_DUMP_STATISTICS_DATA=1
    #   export ORT_DEBUG_NODE_IO_DUMP_INPUT_DATA=1
    #   export ORT_DEBUG_NODE_IO_DUMP_OUTPUT_DATA=1
    #   python benchmark.py --height 1024 --width 1024 --steps 4 -b 1 -v Flux.1S -p flux1_schnell_onnx/fp32_opt -e optimum >stdout.txt 2>stderr.txt
    # Warning: The node name might change in different export settings. See benchmark_flux.sh for the settings.
    flux_node_block_list = {
        "text_encoder_2": [
            "/encoder/block.10/layer.1/DenseReluDense/wo/MatMul",
            "SkipLayerNorm_20",
            "SkipLayerNorm_21",
            "SkipLayerNorm_22",
            "SkipLayerNorm_23",
            "SkipLayerNorm_24",
            "SkipLayerNorm_25",
            "SkipLayerNorm_26",
            "SkipLayerNorm_27",
            "SkipLayerNorm_28",
            "SkipLayerNorm_29",
            "SkipLayerNorm_30",
            "SkipLayerNorm_31",
            "SkipLayerNorm_32",
            "SkipLayerNorm_33",
            "SkipLayerNorm_34",
            "SkipLayerNorm_35",
            "SkipLayerNorm_36",
            "SkipLayerNorm_37",
            "SkipLayerNorm_38",
            "SkipLayerNorm_39",
            "SkipLayerNorm_40",
            "SkipLayerNorm_41",
            "SkipLayerNorm_42",
            "SkipLayerNorm_43",
            "SkipLayerNorm_44",
            "SkipLayerNorm_45",
            "/encoder/block.23/layer.1/DenseReluDense/wo/MatMul",
            "SkipLayerNorm_46",
        ],
        "vae_decoder": [
            "/decoder/mid_block/attentions.0/MatMul",
            "/decoder/mid_block/attentions.0/Softmax",
        ],
        "transformer": [
            "/transformer_blocks.18/Mul_5",
            "/transformer_blocks.18/Add_7",
            "/Concat_1",
            "LayerNorm_76",
            "/single_transformer_blocks.0/Add",
            "LayerNorm_77",
            "/single_transformer_blocks.1/Add",
            "LayerNorm_78",
            "/single_transformer_blocks.2/Add",
            "LayerNorm_79",
            "/single_transformer_blocks.3/Add",
            "LayerNorm_80",
            "/single_transformer_blocks.4/Add",
            "LayerNorm_81",
            "/single_transformer_blocks.5/Add",
            "LayerNorm_82",
            "/single_transformer_blocks.6/Add",
            "LayerNorm_83",
            "/single_transformer_blocks.7/Add",
            "LayerNorm_84",
            "/single_transformer_blocks.8/Add",
            "LayerNorm_85",
            "/single_transformer_blocks.9/Add",
            "LayerNorm_86",
            "/single_transformer_blocks.10/Add",
            "LayerNorm_87",
            "/single_transformer_blocks.11/Add",
            "LayerNorm_88",
            "/single_transformer_blocks.12/Add",
            "LayerNorm_89",
            "/single_transformer_blocks.13/Add",
            "LayerNorm_90",
            "/single_transformer_blocks.14/Add",
            "LayerNorm_91",
            "/single_transformer_blocks.15/Add",
            "LayerNorm_92",
            "/single_transformer_blocks.16/Add",
            "LayerNorm_93",
            "/single_transformer_blocks.17/Add",
            "LayerNorm_94",
            "/single_transformer_blocks.18/Add",
            "LayerNorm_95",
            "/single_transformer_blocks.19/Add",
            "LayerNorm_96",
            "/single_transformer_blocks.20/Add",
            "LayerNorm_97",
            "/single_transformer_blocks.21/Add",
            "LayerNorm_98",
            "/single_transformer_blocks.22/Add",
            "LayerNorm_99",
            "/single_transformer_blocks.23/Add",
            "LayerNorm_100",
            "/single_transformer_blocks.24/Add",
            "LayerNorm_101",
            "/single_transformer_blocks.25/Add",
            "LayerNorm_102",
            "/single_transformer_blocks.26/Add",
            "LayerNorm_103",
            "/single_transformer_blocks.27/Add",
            "LayerNorm_104",
            "/single_transformer_blocks.28/Add",
            "LayerNorm_105",
            "/single_transformer_blocks.29/Add",
            "LayerNorm_106",
            "/single_transformer_blocks.30/Add",
            "LayerNorm_107",
            "/single_transformer_blocks.31/Add",
            "LayerNorm_108",
            "/single_transformer_blocks.32/Add",
            "LayerNorm_109",
            "/single_transformer_blocks.33/Add",
            "LayerNorm_110",
            "/single_transformer_blocks.34/Add",
            "LayerNorm_111",
            "/single_transformer_blocks.35/Add",
            "LayerNorm_112",
            "/single_transformer_blocks.36/Add",
            "LayerNorm_113",
            "/single_transformer_blocks.37/Add",
            "/Shape",
            "/Slice",
        ],
    }

    sd3_node_block_list = {"text_encoder_3": flux_node_block_list["text_encoder_2"]}

    if force_fp32_ops:
        for fp32_operator in force_fp32_ops:
            parts = fp32_operator.split(":")
            if len(parts) == 2 and parts[0] in force_fp32_operators and (parts[1] and parts[1][0].isupper()):
                force_fp32_operators[parts[0]].append(parts[1])
            else:
                raise ValueError(
                    f"--force_fp32_ops shall be in the format of module:operator like unet:Attention, got {fp32_operator}"
                )

    op_counters = {}
    for name, model_type in model_type_mapping.items():
        onnx_model_path = source_dir / name / "model.onnx"
        if not os.path.exists(onnx_model_path):
            if name != "safety_checker" and name in model_list:
                logger.warning("input onnx model does not exist: %s", onnx_model_path)
            # some model are optional so we do not raise error here.
            continue

        # Prepare output directory
        optimized_model_path = target_dir / name / "model.onnx"
        if os.path.exists(optimized_model_path):
            if not args.overwrite:
                logger.warning("Skipped optimization since the target file existed: %s", optimized_model_path)
            continue
        output_dir = optimized_model_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        if use_external_data_format is None:
            use_external_data_format = has_external_data(onnx_model_path)

        # Graph fusion before fp16 conversion, otherwise they cannot be fused later.
        logger.info("Optimize %s ...", onnx_model_path)

        args.model_type = model_type
        fusion_options = FusionOptions.parse(args)

        if model_type in ["unet"]:
            # Some optimizations are not available in v1.14 or older version: packed QKV and BiasAdd
            has_all_optimizations = version.parse(onnxruntime.__version__) >= version.parse("1.15.0")
            fusion_options.enable_packed_kv = float16 and fusion_options.enable_packed_kv
            fusion_options.enable_packed_qkv = float16 and has_all_optimizations and fusion_options.enable_packed_qkv
            fusion_options.enable_bias_add = has_all_optimizations and fusion_options.enable_bias_add

        m = optimize_model(
            str(onnx_model_path),
            model_type=model_type,
            num_heads=0,  # will be deduced from graph
            hidden_size=0,  # will be deduced from graph
            opt_level=0,
            optimization_options=fusion_options,
            use_gpu=True,
            provider=args.provider,
        )

        if float16:
            model_node_block_list = (
                flux_node_block_list if is_flux_pipeline else sd3_node_block_list if pipeline_type == "sd3" else {}
            )
            if name in model_node_block_list:
                # Opset 12 does not support bfloat16.
                # By default, optimum exports T5 model with opset 12. So we need to check the opset version.
                use_bfloat16 = bfloat16
                if use_bfloat16:
                    for opset in m.model.opset_import:
                        if opset.domain in ["", "ai.onnx"] and opset.version < 13:
                            logger.warning(
                                "onnx model requires opset 13 or higher to use bfloat16. Fall back to float32."
                            )
                            use_bfloat16 = False

                m.convert_float_to_float16(
                    keep_io_types=False,
                    node_block_list=model_node_block_list[name],
                    use_bfloat16_as_blocked_nodes_dtype=use_bfloat16,
                )
            # For SD-XL, use FP16 in VAE decoder will cause NaN and black image so we keep it in FP32.
            elif pipeline_type in ["sdxl"] and name in ["vae_decoder"]:
                logger.info("Skip converting %s to float16 to avoid NaN", name)
            else:
                logger.info("Convert %s to float16 ...", name)
                m.convert_float_to_float16(
                    keep_io_types=False,
                    op_block_list=force_fp32_operators[name],
                )

        if enable_runtime_optimization:
            # Use this step to see the final graph that executed by Onnx Runtime.
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Save to a temporary file so that we can load it with Onnx Runtime.
                logger.info("Saving a temporary model to run OnnxRuntime graph optimizations...")
                tmp_model_path = Path(tmp_dir) / "model.onnx"
                m.save_model_to_file(str(tmp_model_path), use_external_data_format=use_external_data_format)
                ort_optimized_model_path = Path(tmp_dir) / "optimized.onnx"
                optimize_by_onnxruntime(
                    str(tmp_model_path),
                    use_gpu=True,
                    provider=args.provider,
                    optimized_model_path=str(ort_optimized_model_path),
                    save_as_external_data=use_external_data_format,
                )
                model = onnx.load(str(ort_optimized_model_path), load_external_data=True)
                m = model_type_class_mapping[model_type](model)

        m.get_operator_statistics()
        op_counters[name] = m.get_fused_operator_statistics()
        m.save_model_to_file(str(optimized_model_path), use_external_data_format=use_external_data_format)
        logger.info("%s is optimized", name)
        logger.info("*" * 20)

    return op_counters


def _copy_extra_directory(source_dir: Path, target_dir: Path, model_list: list[str]):
    """Copy extra directory that does not have onnx model

    Args:
        source_dir (Path): source directory
        target_dir (Path): target directory
        model_list (List[str]): list of directory names with onnx model.

    Raises:
        RuntimeError: source path does not exist
    """
    extra_dirs = ["scheduler", "tokenizer", "tokenizer_2", "tokenizer_3", "feature_extractor"]

    for name in extra_dirs:
        source_path = source_dir / name
        if not os.path.exists(source_path):
            continue

        target_path = target_dir / name
        if target_path.exists():
            shutil.rmtree(target_path)
        shutil.copytree(source_path, target_path)
        logger.info("%s => %s", source_path, target_path)

    extra_files = ["model_index.json"]
    for name in extra_files:
        source_path = source_dir / name
        if not os.path.exists(source_path):
            raise RuntimeError(f"source path does not exist: {source_path}")

        target_path = target_dir / name
        shutil.copyfile(source_path, target_path)
        logger.info("%s => %s", source_path, target_path)

    # Some directory are optional
    for onnx_model_dir in model_list:
        source_path = source_dir / onnx_model_dir / "config.json"
        target_path = target_dir / onnx_model_dir / "config.json"
        if source_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target_path)
            logger.info("%s => %s", source_path, target_path)


def optimize_stable_diffusion_pipeline(
    input_dir: str,
    output_dir: str,
    overwrite: bool,
    use_external_data_format: bool | None,
    float16: bool,
    enable_runtime_optimization: bool,
    args,
):
    if os.path.exists(output_dir):
        if overwrite:
            shutil.rmtree(output_dir, ignore_errors=True)

    source_dir = Path(input_dir)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    pipeline_type = _classify_pipeline_type(source_dir)
    model_list = _get_model_list(pipeline_type)

    _copy_extra_directory(source_dir, target_dir, model_list)

    return _optimize_sd_pipeline(
        source_dir,
        target_dir,
        pipeline_type,
        model_list,
        use_external_data_format,
        float16,
        args.bfloat16,
        args.force_fp32_ops,
        enable_runtime_optimization,
        args,
    )


def parse_arguments(argv: list[str] | None = None):
    """Parse arguments

    Returns:
        Namespace: arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Root of input directory of stable diffusion onnx pipeline with float32 models.",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Root of output directory of stable diffusion onnx pipeline with optimized models.",
    )

    parser.add_argument(
        "--float16",
        required=False,
        action="store_true",
        help="Output models of float16, except some nodes falls back to float32 or bfloat16 to avoid overflow.",
    )
    parser.set_defaults(float16=False)

    parser.add_argument(
        "--bfloat16",
        required=False,
        action="store_true",
        help="Allow bfloat16 as fallback if --float16 is also provided.",
    )
    parser.set_defaults(bfloat16=False)

    parser.add_argument(
        "--force_fp32_ops",
        required=False,
        nargs="+",
        type=str,
        help="Force given operators (like unet:Attention) to run in float32. It is case sensitive!",
    )

    parser.add_argument(
        "--inspect",
        required=False,
        action="store_true",
        help="Save the optimized graph from Onnx Runtime. "
        "This option has no impact on inference performance except it might reduce session creation time.",
    )
    parser.set_defaults(inspect=False)

    parser.add_argument(
        "--overwrite",
        required=False,
        action="store_true",
        help="Overwrite exists files.",
    )
    parser.set_defaults(overwrite=False)

    parser.add_argument(
        "-e",
        "--use_external_data_format",
        required=False,
        action="store_true",
        help="Onnx model larger than 2GB need to use external data format. "
        "If specified, save each onnx model to two files: one for onnx graph, another for weights. "
        "If not specified, use same format as original model by default. ",
    )
    parser.set_defaults(use_external_data_format=None)

    parser.add_argument(
        "--provider",
        required=False,
        type=str,
        default=None,
        help="Execution provider to use.",
    )

    FusionOptions.add_arguments(parser)

    args = parser.parse_args(argv)
    return args


def main(argv: list[str] | None = None):
    args = parse_arguments(argv)

    logger.info("Arguments: %s", str(args))

    # Return op counters for testing purpose.
    return optimize_stable_diffusion_pipeline(
        args.input, args.output, args.overwrite, args.use_external_data_format, args.float16, args.inspect, args
    )


if __name__ == "__main__":
    coloredlogs.install(fmt="%(funcName)20s: %(message)s")
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\_numba\extensions.py ===
# Disable type checking for this module since numba's internals
# are not typed, and we use numba's internals via its extension API
# mypy: ignore-errors
"""
Utility classes/functions to let numba recognize
pandas Index/Series/DataFrame

Mostly vendored from https://github.com/numba/numba/blob/main/numba/tests/pdlike_usecase.py
"""

from __future__ import annotations

from contextlib import contextmanager
import operator

import numba
from numba import types
from numba.core import cgutils
from numba.core.datamodel import models
from numba.core.extending import (
    NativeValue,
    box,
    lower_builtin,
    make_attribute_wrapper,
    overload,
    overload_attribute,
    overload_method,
    register_model,
    type_callable,
    typeof_impl,
    unbox,
)
from numba.core.imputils import impl_ret_borrowed
import numpy as np

from pandas._libs import lib

from pandas.core.indexes.base import Index
from pandas.core.indexing import _iLocIndexer
from pandas.core.internals import SingleBlockManager
from pandas.core.series import Series


# Helper function to hack around fact that Index casts numpy string dtype to object
#
# Idea is to set an attribute on a Index called _numba_data
# that is the original data, or the object data casted to numpy string dtype,
# with a context manager that is unset afterwards
@contextmanager
def set_numba_data(index: Index):
    numba_data = index._data
    if numba_data.dtype in (object, "string"):
        numba_data = np.asarray(numba_data)
        if not lib.is_string_array(numba_data):
            raise ValueError(
                "The numba engine only supports using string or numeric column names"
            )
        numba_data = numba_data.astype("U")
    try:
        index._numba_data = numba_data
        yield index
    finally:
        del index._numba_data


# TODO: Range index support
# (this currently lowers OK, but does not round-trip)
class IndexType(types.Type):
    """
    The type class for Index objects.
    """

    def __init__(self, dtype, layout, pyclass: any) -> None:
        self.pyclass = pyclass
        name = f"index({dtype}, {layout})"
        self.dtype = dtype
        self.layout = layout
        super().__init__(name)

    @property
    def key(self):
        return self.pyclass, self.dtype, self.layout

    @property
    def as_array(self):
        return types.Array(self.dtype, 1, self.layout)

    def copy(self, dtype=None, ndim: int = 1, layout=None):
        assert ndim == 1
        if dtype is None:
            dtype = self.dtype
        layout = layout or self.layout
        return type(self)(dtype, layout, self.pyclass)


class SeriesType(types.Type):
    """
    The type class for Series objects.
    """

    def __init__(self, dtype, index, namety) -> None:
        assert isinstance(index, IndexType)
        self.dtype = dtype
        self.index = index
        self.values = types.Array(self.dtype, 1, "C")
        self.namety = namety
        name = f"series({dtype}, {index}, {namety})"
        super().__init__(name)

    @property
    def key(self):
        return self.dtype, self.index, self.namety

    @property
    def as_array(self):
        return self.values

    def copy(self, dtype=None, ndim: int = 1, layout: str = "C"):
        assert ndim == 1
        assert layout == "C"
        if dtype is None:
            dtype = self.dtype
        return type(self)(dtype, self.index, self.namety)


@typeof_impl.register(Index)
def typeof_index(val, c):
    """
    This will assume that only strings are in object dtype
    index.
    (you should check this before this gets lowered down to numba)
    """
    # arrty = typeof_impl(val._data, c)
    arrty = typeof_impl(val._numba_data, c)
    assert arrty.ndim == 1
    return IndexType(arrty.dtype, arrty.layout, type(val))


@typeof_impl.register(Series)
def typeof_series(val, c):
    index = typeof_impl(val.index, c)
    arrty = typeof_impl(val.values, c)
    namety = typeof_impl(val.name, c)
    assert arrty.ndim == 1
    assert arrty.layout == "C"
    return SeriesType(arrty.dtype, index, namety)


@type_callable(Series)
def type_series_constructor(context):
    def typer(data, index, name=None):
        if isinstance(index, IndexType) and isinstance(data, types.Array):
            assert data.ndim == 1
            if name is None:
                name = types.intp
            return SeriesType(data.dtype, index, name)

    return typer


@type_callable(Index)
def type_index_constructor(context):
    def typer(data, hashmap=None):
        if isinstance(data, types.Array):
            assert data.layout == "C"
            assert data.ndim == 1
            assert hashmap is None or isinstance(hashmap, types.DictType)
            return IndexType(data.dtype, layout=data.layout, pyclass=Index)

    return typer


# Backend extensions for Index and Series and Frame
@register_model(IndexType)
class IndexModel(models.StructModel):
    def __init__(self, dmm, fe_type) -> None:
        # We don't want the numpy string scalar type in our hashmap
        members = [
            ("data", fe_type.as_array),
            # This is an attempt to emulate our hashtable code with a numba
            # typed dict
            # It maps from values in the index to their integer positions in the array
            ("hashmap", types.DictType(fe_type.dtype, types.intp)),
            # Pointer to the Index object this was created from, or that it
            # boxes to
            # https://numba.discourse.group/t/qst-how-to-cache-the-boxing-of-an-object/2128/2?u=lithomas1
            ("parent", types.pyobject),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@register_model(SeriesType)
class SeriesModel(models.StructModel):
    def __init__(self, dmm, fe_type) -> None:
        members = [
            ("index", fe_type.index),
            ("values", fe_type.as_array),
            ("name", fe_type.namety),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(IndexType, "data", "_data")
make_attribute_wrapper(IndexType, "hashmap", "hashmap")

make_attribute_wrapper(SeriesType, "index", "index")
make_attribute_wrapper(SeriesType, "values", "values")
make_attribute_wrapper(SeriesType, "name", "name")


@lower_builtin(Series, types.Array, IndexType)
def pdseries_constructor(context, builder, sig, args):
    data, index = args
    series = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    series.index = index
    series.values = data
    series.name = context.get_constant(types.intp, 0)
    return impl_ret_borrowed(context, builder, sig.return_type, series._getvalue())


@lower_builtin(Series, types.Array, IndexType, types.intp)
@lower_builtin(Series, types.Array, IndexType, types.float64)
@lower_builtin(Series, types.Array, IndexType, types.unicode_type)
def pdseries_constructor_with_name(context, builder, sig, args):
    data, index, name = args
    series = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    series.index = index
    series.values = data
    series.name = name
    return impl_ret_borrowed(context, builder, sig.return_type, series._getvalue())


@lower_builtin(Index, types.Array, types.DictType, types.pyobject)
def index_constructor_2arg(context, builder, sig, args):
    (data, hashmap, parent) = args
    index = cgutils.create_struct_proxy(sig.return_type)(context, builder)

    index.data = data
    index.hashmap = hashmap
    index.parent = parent
    return impl_ret_borrowed(context, builder, sig.return_type, index._getvalue())


@lower_builtin(Index, types.Array, types.DictType)
def index_constructor_2arg_parent(context, builder, sig, args):
    # Basically same as index_constructor_1arg, but also lets you specify the
    # parent object
    (data, hashmap) = args
    index = cgutils.create_struct_proxy(sig.return_type)(context, builder)

    index.data = data
    index.hashmap = hashmap
    return impl_ret_borrowed(context, builder, sig.return_type, index._getvalue())


@lower_builtin(Index, types.Array)
def index_constructor_1arg(context, builder, sig, args):
    from numba.typed import Dict

    key_type = sig.return_type.dtype
    value_type = types.intp

    def index_impl(data):
        return Index(data, Dict.empty(key_type, value_type))

    return context.compile_internal(builder, index_impl, sig, args)


# Helper to convert the unicodecharseq (numpy string scalar) into a unicode_type
# (regular string)
def maybe_cast_str(x):
    # Dummy function that numba can overload
    pass


@overload(maybe_cast_str)
def maybe_cast_str_impl(x):
    """Converts numba UnicodeCharSeq (numpy string scalar) -> unicode type (string).
    Is a no-op for other types."""
    if isinstance(x, types.UnicodeCharSeq):
        return lambda x: str(x)
    else:
        return lambda x: x


@unbox(IndexType)
def unbox_index(typ, obj, c):
    """
    Convert a Index object to a native structure.

    Note: Object dtype is not allowed here
    """
    data_obj = c.pyapi.object_getattr_string(obj, "_numba_data")
    index = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    # If we see an object array, assume its been validated as only containing strings
    # We still need to do the conversion though
    index.data = c.unbox(typ.as_array, data_obj).value
    typed_dict_obj = c.pyapi.unserialize(c.pyapi.serialize_object(numba.typed.Dict))
    # Create an empty typed dict in numba for the hashmap for indexing
    # equiv of numba.typed.Dict.empty(typ.dtype, types.intp)
    arr_type_obj = c.pyapi.unserialize(c.pyapi.serialize_object(typ.dtype))
    intp_type_obj = c.pyapi.unserialize(c.pyapi.serialize_object(types.intp))
    hashmap_obj = c.pyapi.call_method(
        typed_dict_obj, "empty", (arr_type_obj, intp_type_obj)
    )
    index.hashmap = c.unbox(types.DictType(typ.dtype, types.intp), hashmap_obj).value
    # Set the parent for speedy boxing.
    index.parent = obj

    # Decrefs
    c.pyapi.decref(data_obj)
    c.pyapi.decref(arr_type_obj)
    c.pyapi.decref(intp_type_obj)
    c.pyapi.decref(typed_dict_obj)

    return NativeValue(index._getvalue())


@unbox(SeriesType)
def unbox_series(typ, obj, c):
    """
    Convert a Series object to a native structure.
    """
    index_obj = c.pyapi.object_getattr_string(obj, "index")
    values_obj = c.pyapi.object_getattr_string(obj, "values")
    name_obj = c.pyapi.object_getattr_string(obj, "name")

    series = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    series.index = c.unbox(typ.index, index_obj).value
    series.values = c.unbox(typ.values, values_obj).value
    series.name = c.unbox(typ.namety, name_obj).value

    # Decrefs
    c.pyapi.decref(index_obj)
    c.pyapi.decref(values_obj)
    c.pyapi.decref(name_obj)

    return NativeValue(series._getvalue())


@box(IndexType)
def box_index(typ, val, c):
    """
    Convert a native index structure to a Index object.

    If our native index is of a numpy string dtype, we'll cast it to
    object.
    """
    # First build a Numpy array object, then wrap it in a Index
    index = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)

    res = cgutils.alloca_once_value(c.builder, index.parent)

    # Does parent exist?
    # (it means already boxed once, or Index same as original df.index or df.columns)
    # xref https://github.com/numba/numba/blob/596e8a55334cc46854e3192766e643767bd7c934/numba/core/boxing.py#L593C17-L593C17
    with c.builder.if_else(cgutils.is_not_null(c.builder, index.parent)) as (
        has_parent,
        otherwise,
    ):
        with has_parent:
            c.pyapi.incref(index.parent)
        with otherwise:
            # TODO: preserve the original class for the index
            # Also need preserve the name of the Index
            # class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(typ.pyclass))
            class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Index))
            array_obj = c.box(typ.as_array, index.data)
            if isinstance(typ.dtype, types.UnicodeCharSeq):
                # We converted to numpy string dtype, convert back
                # to object since _simple_new won't do that for uss
                object_str_obj = c.pyapi.unserialize(c.pyapi.serialize_object("object"))
                array_obj = c.pyapi.call_method(array_obj, "astype", (object_str_obj,))
                c.pyapi.decref(object_str_obj)
            # this is basically Index._simple_new(array_obj, name_obj) in python
            index_obj = c.pyapi.call_method(class_obj, "_simple_new", (array_obj,))
            index.parent = index_obj
            c.builder.store(index_obj, res)

            # Decrefs
            c.pyapi.decref(class_obj)
            c.pyapi.decref(array_obj)
    return c.builder.load(res)


@box(SeriesType)
def box_series(typ, val, c):
    """
    Convert a native series structure to a Series object.
    """
    series = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    series_const_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Series._from_mgr))
    mgr_const_obj = c.pyapi.unserialize(
        c.pyapi.serialize_object(SingleBlockManager.from_array)
    )
    index_obj = c.box(typ.index, series.index)
    array_obj = c.box(typ.as_array, series.values)
    name_obj = c.box(typ.namety, series.name)
    # This is basically equivalent of
    # pd.Series(data=array_obj, index=index_obj)
    # To improve perf, we will construct the Series from a manager
    # object to avoid checks.
    # We'll also set the name attribute manually to avoid validation
    mgr_obj = c.pyapi.call_function_objargs(
        mgr_const_obj,
        (
            array_obj,
            index_obj,
        ),
    )
    mgr_axes_obj = c.pyapi.object_getattr_string(mgr_obj, "axes")
    # Series._constructor_from_mgr(mgr, axes)
    series_obj = c.pyapi.call_function_objargs(
        series_const_obj, (mgr_obj, mgr_axes_obj)
    )
    c.pyapi.object_setattr_string(series_obj, "_name", name_obj)

    # Decrefs
    c.pyapi.decref(series_const_obj)
    c.pyapi.decref(mgr_axes_obj)
    c.pyapi.decref(mgr_obj)
    c.pyapi.decref(mgr_const_obj)
    c.pyapi.decref(index_obj)
    c.pyapi.decref(array_obj)
    c.pyapi.decref(name_obj)

    return series_obj


# Add common series reductions (e.g. mean, sum),
# and also add common binops (e.g. add, sub, mul, div)
def generate_series_reduction(ser_reduction, ser_method):
    @overload_method(SeriesType, ser_reduction)
    def series_reduction(series):
        def series_reduction_impl(series):
            return ser_method(series.values)

        return series_reduction_impl

    return series_reduction


def generate_series_binop(binop):
    @overload(binop)
    def series_binop(series1, value):
        if isinstance(series1, SeriesType):
            if isinstance(value, SeriesType):

                def series_binop_impl(series1, series2):
                    # TODO: Check index matching?
                    return Series(
                        binop(series1.values, series2.values),
                        series1.index,
                        series1.name,
                    )

                return series_binop_impl
            else:

                def series_binop_impl(series1, value):
                    return Series(
                        binop(series1.values, value), series1.index, series1.name
                    )

                return series_binop_impl

    return series_binop


series_reductions = [
    ("sum", np.sum),
    ("mean", np.mean),
    # Disabled due to discrepancies between numba std. dev
    # and pandas std. dev (no way to specify dof)
    # ("std", np.std),
    # ("var", np.var),
    ("min", np.min),
    ("max", np.max),
]
for reduction, reduction_method in series_reductions:
    generate_series_reduction(reduction, reduction_method)

series_binops = [operator.add, operator.sub, operator.mul, operator.truediv]

for ser_binop in series_binops:
    generate_series_binop(ser_binop)


# get_loc on Index
@overload_method(IndexType, "get_loc")
def index_get_loc(index, item):
    def index_get_loc_impl(index, item):
        # Initialize the hash table if not initialized
        if len(index.hashmap) == 0:
            for i, val in enumerate(index._data):
                index.hashmap[val] = i
        return index.hashmap[item]

    return index_get_loc_impl


# Indexing for Series/Index
@overload(operator.getitem)
def series_indexing(series, item):
    if isinstance(series, SeriesType):

        def series_getitem(series, item):
            loc = series.index.get_loc(item)
            return series.iloc[loc]

        return series_getitem


@overload(operator.getitem)
def index_indexing(index, idx):
    if isinstance(index, IndexType):

        def index_getitem(index, idx):
            return index._data[idx]

        return index_getitem


class IlocType(types.Type):
    def __init__(self, obj_type) -> None:
        self.obj_type = obj_type
        name = f"iLocIndexer({obj_type})"
        super().__init__(name=name)

    @property
    def key(self):
        return self.obj_type


@typeof_impl.register(_iLocIndexer)
def typeof_iloc(val, c):
    objtype = typeof_impl(val.obj, c)
    return IlocType(objtype)


@type_callable(_iLocIndexer)
def type_iloc_constructor(context):
    def typer(obj):
        if isinstance(obj, SeriesType):
            return IlocType(obj)

    return typer


@lower_builtin(_iLocIndexer, SeriesType)
def iloc_constructor(context, builder, sig, args):
    (obj,) = args
    iloc_indexer = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    iloc_indexer.obj = obj
    return impl_ret_borrowed(
        context, builder, sig.return_type, iloc_indexer._getvalue()
    )


@register_model(IlocType)
class ILocModel(models.StructModel):
    def __init__(self, dmm, fe_type) -> None:
        members = [("obj", fe_type.obj_type)]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(IlocType, "obj", "obj")


@overload_attribute(SeriesType, "iloc")
def series_iloc(series):
    def get(series):
        return _iLocIndexer(series)

    return get


@overload(operator.getitem)
def iloc_getitem(iloc_indexer, i):
    if isinstance(iloc_indexer, IlocType):

        def getitem_impl(iloc_indexer, i):
            return iloc_indexer.obj.values[i]

        return getitem_impl

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\sql\annotation.py ===
# sql/annotation.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""The :class:`.Annotated` class and related routines; creates hash-equivalent
copies of SQL constructs which contain context-specific markers and
associations.

Note that the :class:`.Annotated` concept as implemented in this module is not
related in any way to the pep-593 concept of "Annotated".


"""

from __future__ import annotations

import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar

from . import operators
from .cache_key import HasCacheKey
from .visitors import anon_map
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .. import util
from ..util.typing import Literal
from ..util.typing import Self

if TYPE_CHECKING:
    from .base import _EntityNamespace
    from .visitors import _TraverseInternalsType

_AnnotationDict = Mapping[str, Any]

EMPTY_ANNOTATIONS: util.immutabledict[str, Any] = util.EMPTY_DICT


class SupportsAnnotations(ExternallyTraversible):
    __slots__ = ()

    _annotations: util.immutabledict[str, Any] = EMPTY_ANNOTATIONS

    proxy_set: util.generic_fn_descriptor[FrozenSet[Any]]

    _is_immutable: bool

    def _annotate(self, values: _AnnotationDict) -> Self:
        raise NotImplementedError()

    @overload
    def _deannotate(
        self,
        values: Literal[None] = ...,
        clone: bool = ...,
    ) -> Self: ...

    @overload
    def _deannotate(
        self,
        values: Sequence[str] = ...,
        clone: bool = ...,
    ) -> SupportsAnnotations: ...

    def _deannotate(
        self,
        values: Optional[Sequence[str]] = None,
        clone: bool = False,
    ) -> SupportsAnnotations:
        raise NotImplementedError()

    @util.memoized_property
    def _annotations_cache_key(self) -> Tuple[Any, ...]:
        anon_map_ = anon_map()

        return self._gen_annotations_cache_key(anon_map_)

    def _gen_annotations_cache_key(
        self, anon_map: anon_map
    ) -> Tuple[Any, ...]:
        return (
            "_annotations",
            tuple(
                (
                    key,
                    (
                        value._gen_cache_key(anon_map, [])
                        if isinstance(value, HasCacheKey)
                        else value
                    ),
                )
                for key, value in [
                    (key, self._annotations[key])
                    for key in sorted(self._annotations)
                ]
            ),
        )


class SupportsWrappingAnnotations(SupportsAnnotations):
    __slots__ = ()

    _constructor: Callable[..., SupportsWrappingAnnotations]

    if TYPE_CHECKING:

        @util.ro_non_memoized_property
        def entity_namespace(self) -> _EntityNamespace: ...

    def _annotate(self, values: _AnnotationDict) -> Self:
        """return a copy of this ClauseElement with annotations
        updated by the given dictionary.

        """
        return Annotated._as_annotated_instance(self, values)  # type: ignore

    def _with_annotations(self, values: _AnnotationDict) -> Self:
        """return a copy of this ClauseElement with annotations
        replaced by the given dictionary.

        """
        return Annotated._as_annotated_instance(self, values)  # type: ignore

    @overload
    def _deannotate(
        self,
        values: Literal[None] = ...,
        clone: bool = ...,
    ) -> Self: ...

    @overload
    def _deannotate(
        self,
        values: Sequence[str] = ...,
        clone: bool = ...,
    ) -> SupportsAnnotations: ...

    def _deannotate(
        self,
        values: Optional[Sequence[str]] = None,
        clone: bool = False,
    ) -> SupportsAnnotations:
        """return a copy of this :class:`_expression.ClauseElement`
        with annotations
        removed.

        :param values: optional tuple of individual values
         to remove.

        """
        if clone:
            s = self._clone()
            return s
        else:
            return self


class SupportsCloneAnnotations(SupportsWrappingAnnotations):
    # SupportsCloneAnnotations extends from SupportsWrappingAnnotations
    # to support the structure of having the base ClauseElement
    # be a subclass of SupportsWrappingAnnotations.  Any ClauseElement
    # subclass that wants to extend from SupportsCloneAnnotations
    # will inherently also be subclassing SupportsWrappingAnnotations, so
    # make that specific here.

    if not typing.TYPE_CHECKING:
        __slots__ = ()

    _clone_annotations_traverse_internals: _TraverseInternalsType = [
        ("_annotations", InternalTraversal.dp_annotations_key)
    ]

    def _annotate(self, values: _AnnotationDict) -> Self:
        """return a copy of this ClauseElement with annotations
        updated by the given dictionary.

        """
        new = self._clone()
        new._annotations = new._annotations.union(values)
        new.__dict__.pop("_annotations_cache_key", None)
        new.__dict__.pop("_generate_cache_key", None)
        return new

    def _with_annotations(self, values: _AnnotationDict) -> Self:
        """return a copy of this ClauseElement with annotations
        replaced by the given dictionary.

        """
        new = self._clone()
        new._annotations = util.immutabledict(values)
        new.__dict__.pop("_annotations_cache_key", None)
        new.__dict__.pop("_generate_cache_key", None)
        return new

    @overload
    def _deannotate(
        self,
        values: Literal[None] = ...,
        clone: bool = ...,
    ) -> Self: ...

    @overload
    def _deannotate(
        self,
        values: Sequence[str] = ...,
        clone: bool = ...,
    ) -> SupportsAnnotations: ...

    def _deannotate(
        self,
        values: Optional[Sequence[str]] = None,
        clone: bool = False,
    ) -> SupportsAnnotations:
        """return a copy of this :class:`_expression.ClauseElement`
        with annotations
        removed.

        :param values: optional tuple of individual values
         to remove.

        """
        if clone or self._annotations:
            # clone is used when we are also copying
            # the expression for a deep deannotation
            new = self._clone()
            new._annotations = util.immutabledict()
            new.__dict__.pop("_annotations_cache_key", None)
            return new
        else:
            return self


class Annotated(SupportsAnnotations):
    """clones a SupportsAnnotations and applies an 'annotations' dictionary.

    Unlike regular clones, this clone also mimics __hash__() and
    __eq__() of the original element so that it takes its place
    in hashed collections.

    A reference to the original element is maintained, for the important
    reason of keeping its hash value current.  When GC'ed, the
    hash value may be reused, causing conflicts.

    .. note::  The rationale for Annotated producing a brand new class,
       rather than placing the functionality directly within ClauseElement,
       is **performance**.  The __hash__() method is absent on plain
       ClauseElement which leads to significantly reduced function call
       overhead, as the use of sets and dictionaries against ClauseElement
       objects is prevalent, but most are not "annotated".

    """

    _is_column_operators = False

    @classmethod
    def _as_annotated_instance(
        cls, element: SupportsWrappingAnnotations, values: _AnnotationDict
    ) -> Annotated:
        try:
            cls = annotated_classes[element.__class__]
        except KeyError:
            cls = _new_annotation_type(element.__class__, cls)
        return cls(element, values)

    _annotations: util.immutabledict[str, Any]
    __element: SupportsWrappingAnnotations
    _hash: int

    def __new__(cls: Type[Self], *args: Any) -> Self:
        return object.__new__(cls)

    def __init__(
        self, element: SupportsWrappingAnnotations, values: _AnnotationDict
    ):
        self.__dict__ = element.__dict__.copy()
        self.__dict__.pop("_annotations_cache_key", None)
        self.__dict__.pop("_generate_cache_key", None)
        self.__element = element
        self._annotations = util.immutabledict(values)
        self._hash = hash(element)

    def _annotate(self, values: _AnnotationDict) -> Self:
        _values = self._annotations.union(values)
        new = self._with_annotations(_values)
        return new

    def _with_annotations(self, values: _AnnotationDict) -> Self:
        clone = self.__class__.__new__(self.__class__)
        clone.__dict__ = self.__dict__.copy()
        clone.__dict__.pop("_annotations_cache_key", None)
        clone.__dict__.pop("_generate_cache_key", None)
        clone._annotations = util.immutabledict(values)
        return clone

    @overload
    def _deannotate(
        self,
        values: Literal[None] = ...,
        clone: bool = ...,
    ) -> Self: ...

    @overload
    def _deannotate(
        self,
        values: Sequence[str] = ...,
        clone: bool = ...,
    ) -> Annotated: ...

    def _deannotate(
        self,
        values: Optional[Sequence[str]] = None,
        clone: bool = True,
    ) -> SupportsAnnotations:
        if values is None:
            return self.__element
        else:
            return self._with_annotations(
                util.immutabledict(
                    {
                        key: value
                        for key, value in self._annotations.items()
                        if key not in values
                    }
                )
            )

    if not typing.TYPE_CHECKING:
        # manually proxy some methods that need extra attention
        def _compiler_dispatch(self, visitor: Any, **kw: Any) -> Any:
            return self.__element.__class__._compiler_dispatch(
                self, visitor, **kw
            )

        @property
        def _constructor(self):
            return self.__element._constructor

    def _clone(self, **kw: Any) -> Self:
        clone = self.__element._clone(**kw)
        if clone is self.__element:
            # detect immutable, don't change anything
            return self
        else:
            # update the clone with any changes that have occurred
            # to this object's __dict__.
            clone.__dict__.update(self.__dict__)
            return self.__class__(clone, self._annotations)

    def __reduce__(self) -> Tuple[Type[Annotated], Tuple[Any, ...]]:
        return self.__class__, (self.__element, self._annotations)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: Any) -> bool:
        if self._is_column_operators:
            return self.__element.__class__.__eq__(self, other)
        else:
            return hash(other) == hash(self)

    @util.ro_non_memoized_property
    def entity_namespace(self) -> _EntityNamespace:
        if "entity_namespace" in self._annotations:
            return cast(
                SupportsWrappingAnnotations,
                self._annotations["entity_namespace"],
            ).entity_namespace
        else:
            return self.__element.entity_namespace


# hard-generate Annotated subclasses.  this technique
# is used instead of on-the-fly types (i.e. type.__new__())
# so that the resulting objects are pickleable; additionally, other
# decisions can be made up front about the type of object being annotated
# just once per class rather than per-instance.
annotated_classes: Dict[Type[SupportsWrappingAnnotations], Type[Annotated]] = (
    {}
)

_SA = TypeVar("_SA", bound="SupportsAnnotations")


def _safe_annotate(to_annotate: _SA, annotations: _AnnotationDict) -> _SA:
    try:
        _annotate = to_annotate._annotate
    except AttributeError:
        # skip objects that don't actually have an `_annotate`
        # attribute, namely QueryableAttribute inside of a join
        # condition
        return to_annotate
    else:
        return _annotate(annotations)


def _deep_annotate(
    element: _SA,
    annotations: _AnnotationDict,
    exclude: Optional[Sequence[SupportsAnnotations]] = None,
    *,
    detect_subquery_cols: bool = False,
    ind_cols_on_fromclause: bool = False,
    annotate_callable: Optional[
        Callable[[SupportsAnnotations, _AnnotationDict], SupportsAnnotations]
    ] = None,
) -> _SA:
    """Deep copy the given ClauseElement, annotating each element
    with the given annotations dictionary.

    Elements within the exclude collection will be cloned but not annotated.

    """

    # annotated objects hack the __hash__() method so if we want to
    # uniquely process them we have to use id()

    cloned_ids: Dict[int, SupportsAnnotations] = {}

    def clone(elem: SupportsAnnotations, **kw: Any) -> SupportsAnnotations:
        # ind_cols_on_fromclause means make sure an AnnotatedFromClause
        # has its own .c collection independent of that which its proxying.
        # this is used specifically by orm.LoaderCriteriaOption to break
        # a reference cycle that it's otherwise prone to building,
        # see test_relationship_criteria->
        # test_loader_criteria_subquery_w_same_entity.  logic here was
        # changed for #8796 and made explicit; previously it occurred
        # by accident

        kw["detect_subquery_cols"] = detect_subquery_cols
        id_ = id(elem)

        if id_ in cloned_ids:
            return cloned_ids[id_]

        if (
            exclude
            and hasattr(elem, "proxy_set")
            and elem.proxy_set.intersection(exclude)
        ):
            newelem = elem._clone(clone=clone, **kw)
        elif annotations != elem._annotations:
            if detect_subquery_cols and elem._is_immutable:
                to_annotate = elem._clone(clone=clone, **kw)
            else:
                to_annotate = elem
            if annotate_callable:
                newelem = annotate_callable(to_annotate, annotations)
            else:
                newelem = _safe_annotate(to_annotate, annotations)
        else:
            newelem = elem

        newelem._copy_internals(
            clone=clone, ind_cols_on_fromclause=ind_cols_on_fromclause
        )

        cloned_ids[id_] = newelem
        return newelem

    if element is not None:
        element = cast(_SA, clone(element))
    clone = None  # type: ignore  # remove gc cycles
    return element


@overload
def _deep_deannotate(
    element: Literal[None], values: Optional[Sequence[str]] = None
) -> Literal[None]: ...


@overload
def _deep_deannotate(
    element: _SA, values: Optional[Sequence[str]] = None
) -> _SA: ...


def _deep_deannotate(
    element: Optional[_SA], values: Optional[Sequence[str]] = None
) -> Optional[_SA]:
    """Deep copy the given element, removing annotations."""

    cloned: Dict[Any, SupportsAnnotations] = {}

    def clone(elem: SupportsAnnotations, **kw: Any) -> SupportsAnnotations:
        key: Any
        if values:
            key = id(elem)
        else:
            key = elem

        if key not in cloned:
            newelem = elem._deannotate(values=values, clone=True)
            newelem._copy_internals(clone=clone)
            cloned[key] = newelem
            return newelem
        else:
            return cloned[key]

    if element is not None:
        element = cast(_SA, clone(element))
    clone = None  # type: ignore  # remove gc cycles
    return element


def _shallow_annotate(element: _SA, annotations: _AnnotationDict) -> _SA:
    """Annotate the given ClauseElement and copy its internals so that
    internal objects refer to the new annotated object.

    Basically used to apply a "don't traverse" annotation to a
    selectable, without digging throughout the whole
    structure wasting time.
    """
    element = element._annotate(annotations)
    element._copy_internals()
    return element


def _new_annotation_type(
    cls: Type[SupportsWrappingAnnotations], base_cls: Type[Annotated]
) -> Type[Annotated]:
    """Generates a new class that subclasses Annotated and proxies a given
    element type.

    """
    if issubclass(cls, Annotated):
        return cls
    elif cls in annotated_classes:
        return annotated_classes[cls]

    for super_ in cls.__mro__:
        # check if an Annotated subclass more specific than
        # the given base_cls is already registered, such
        # as AnnotatedColumnElement.
        if super_ in annotated_classes:
            base_cls = annotated_classes[super_]
            break

    annotated_classes[cls] = anno_cls = cast(
        Type[Annotated],
        type("Annotated%s" % cls.__name__, (base_cls, cls), {}),
    )
    globals()["Annotated%s" % cls.__name__] = anno_cls

    if "_traverse_internals" in cls.__dict__:
        anno_cls._traverse_internals = list(cls._traverse_internals) + [
            ("_annotations", InternalTraversal.dp_annotations_key)
        ]
    elif cls.__dict__.get("inherit_cache", False):
        anno_cls._traverse_internals = list(cls._traverse_internals) + [
            ("_annotations", InternalTraversal.dp_annotations_key)
        ]

    # some classes include this even if they have traverse_internals
    # e.g. BindParameter, add it if present.
    if cls.__dict__.get("inherit_cache", False):
        anno_cls.inherit_cache = True  # type: ignore
    elif "inherit_cache" in cls.__dict__:
        anno_cls.inherit_cache = cls.__dict__["inherit_cache"]  # type: ignore

    anno_cls._is_column_operators = issubclass(cls, operators.ColumnOperators)

    return anno_cls


def _prepare_annotations(
    target_hierarchy: Type[SupportsWrappingAnnotations],
    base_cls: Type[Annotated],
) -> None:
    for cls in util.walk_subclasses(target_hierarchy):
        _new_annotation_type(cls, base_cls)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\libmp\libintmath.py ===
"""
Utility functions for integer math.

TODO: rename, cleanup, perhaps move the gmpy wrapper code
here from settings.py

"""

import math
from bisect import bisect

from .backend import xrange
from .backend import BACKEND, gmpy, sage, sage_utils, MPZ, MPZ_ONE, MPZ_ZERO

small_trailing = [0] * 256
for j in range(1,8):
    small_trailing[1<<j::1<<(j+1)] = [j] * (1<<(7-j))

def giant_steps(start, target, n=2):
    """
    Return a list of integers ~=

    [start, n*start, ..., target/n^2, target/n, target]

    but conservatively rounded so that the quotient between two
    successive elements is actually slightly less than n.

    With n = 2, this describes suitable precision steps for a
    quadratically convergent algorithm such as Newton's method;
    with n = 3 steps for cubic convergence (Halley's method), etc.

        >>> giant_steps(50,1000)
        [66, 128, 253, 502, 1000]
        >>> giant_steps(50,1000,4)
        [65, 252, 1000]

    """
    L = [target]
    while L[-1] > start*n:
        L = L + [L[-1]//n + 2]
    return L[::-1]

def rshift(x, n):
    """For an integer x, calculate x >> n with the fastest (floor)
    rounding. Unlike the plain Python expression (x >> n), n is
    allowed to be negative, in which case a left shift is performed."""
    if n >= 0: return x >> n
    else:      return x << (-n)

def lshift(x, n):
    """For an integer x, calculate x << n. Unlike the plain Python
    expression (x << n), n is allowed to be negative, in which case a
    right shift with default (floor) rounding is performed."""
    if n >= 0: return x << n
    else:      return x >> (-n)

if BACKEND == 'sage':
    import operator
    rshift = operator.rshift
    lshift = operator.lshift

def python_trailing(n):
    """Count the number of trailing zero bits in abs(n)."""
    if not n:
        return 0
    low_byte = n & 0xff
    if low_byte:
        return small_trailing[low_byte]
    t = 8
    n >>= 8
    while not n & 0xff:
        n >>= 8
        t += 8
    return t + small_trailing[n & 0xff]

if BACKEND == 'gmpy':
    if gmpy.version() >= '2':
        def gmpy_trailing(n):
            """Count the number of trailing zero bits in abs(n) using gmpy."""
            if n: return MPZ(n).bit_scan1()
            else: return 0
    else:
        def gmpy_trailing(n):
            """Count the number of trailing zero bits in abs(n) using gmpy."""
            if n: return MPZ(n).scan1()
            else: return 0

# Small powers of 2
powers = [1<<_ for _ in range(300)]

def python_bitcount(n):
    """Calculate bit size of the nonnegative integer n."""
    bc = bisect(powers, n)
    if bc != 300:
        return bc
    bc = int(math.log(n, 2)) - 4
    return bc + bctable[n>>bc]

def gmpy_bitcount(n):
    """Calculate bit size of the nonnegative integer n."""
    if n: return MPZ(n).numdigits(2)
    else: return 0

#def sage_bitcount(n):
#    if n: return MPZ(n).nbits()
#    else: return 0

def sage_trailing(n):
    return MPZ(n).trailing_zero_bits()

if BACKEND == 'gmpy':
    bitcount = gmpy_bitcount
    trailing = gmpy_trailing
elif BACKEND == 'sage':
    sage_bitcount = sage_utils.bitcount
    bitcount = sage_bitcount
    trailing = sage_trailing
else:
    bitcount = python_bitcount
    trailing = python_trailing

if BACKEND == 'gmpy' and 'bit_length' in dir(gmpy):
    bitcount = gmpy.bit_length

# Used to avoid slow function calls as far as possible
trailtable = [trailing(n) for n in range(256)]
bctable = [bitcount(n) for n in range(1024)]

# TODO: speed up for bases 2, 4, 8, 16, ...

def bin_to_radix(x, xbits, base, bdigits):
    """Changes radix of a fixed-point number; i.e., converts
    x * 2**xbits to floor(x * 10**bdigits)."""
    return x * (MPZ(base)**bdigits) >> xbits

stddigits = '0123456789abcdefghijklmnopqrstuvwxyz'

def small_numeral(n, base=10, digits=stddigits):
    """Return the string numeral of a positive integer in an arbitrary
    base. Most efficient for small input."""
    if base == 10:
        return str(n)
    digs = []
    while n:
        n, digit = divmod(n, base)
        digs.append(digits[digit])
    return "".join(digs[::-1])

def numeral_python(n, base=10, size=0, digits=stddigits):
    """Represent the integer n as a string of digits in the given base.
    Recursive division is used to make this function about 3x faster
    than Python's str() for converting integers to decimal strings.

    The 'size' parameters specifies the number of digits in n; this
    number is only used to determine splitting points and need not be
    exact."""
    if n <= 0:
        if not n:
            return "0"
        return "-" + numeral(-n, base, size, digits)
    # Fast enough to do directly
    if size < 250:
        return small_numeral(n, base, digits)
    # Divide in half
    half = (size // 2) + (size & 1)
    A, B = divmod(n, base**half)
    ad = numeral(A, base, half, digits)
    bd = numeral(B, base, half, digits).rjust(half, "0")
    return ad + bd

def numeral_gmpy(n, base=10, size=0, digits=stddigits):
    """Represent the integer n as a string of digits in the given base.
    Recursive division is used to make this function about 3x faster
    than Python's str() for converting integers to decimal strings.

    The 'size' parameters specifies the number of digits in n; this
    number is only used to determine splitting points and need not be
    exact."""
    if n < 0:
        return "-" + numeral(-n, base, size, digits)
    # gmpy.digits() may cause a segmentation fault when trying to convert
    # extremely large values to a string. The size limit may need to be
    # adjusted on some platforms, but 1500000 works on Windows and Linux.
    if size < 1500000:
        return gmpy.digits(n, base)
    # Divide in half
    half = (size // 2) + (size & 1)
    A, B = divmod(n, MPZ(base)**half)
    ad = numeral(A, base, half, digits)
    bd = numeral(B, base, half, digits).rjust(half, "0")
    return ad + bd

if BACKEND == "gmpy":
    numeral = numeral_gmpy
else:
    numeral = numeral_python

_1_800 = 1<<800
_1_600 = 1<<600
_1_400 = 1<<400
_1_200 = 1<<200
_1_100 = 1<<100
_1_50 = 1<<50

def isqrt_small_python(x):
    """
    Correctly (floor) rounded integer square root, using
    division. Fast up to ~200 digits.
    """
    if not x:
        return x
    if x < _1_800:
        # Exact with IEEE double precision arithmetic
        if x < _1_50:
            return int(x**0.5)
        # Initial estimate can be any integer >= the true root; round up
        r = int(x**0.5 * 1.00000000000001) + 1
    else:
        bc = bitcount(x)
        n = bc//2
        r = int((x>>(2*n-100))**0.5+2)<<(n-50)  # +2 is to round up
    # The following iteration now precisely computes floor(sqrt(x))
    # See e.g. Crandall & Pomerance, "Prime Numbers: A Computational
    # Perspective"
    while 1:
        y = (r+x//r)>>1
        if y >= r:
            return r
        r = y

def isqrt_fast_python(x):
    """
    Fast approximate integer square root, computed using division-free
    Newton iteration for large x. For random integers the result is almost
    always correct (floor(sqrt(x))), but is 1 ulp too small with a roughly
    0.1% probability. If x is very close to an exact square, the answer is
    1 ulp wrong with high probability.

    With 0 guard bits, the largest error over a set of 10^5 random
    inputs of size 1-10^5 bits was 3 ulp. The use of 10 guard bits
    almost certainly guarantees a max 1 ulp error.
    """
    # Use direct division-based iteration if sqrt(x) < 2^400
    # Assume floating-point square root accurate to within 1 ulp, then:
    # 0 Newton iterations good to 52 bits
    # 1 Newton iterations good to 104 bits
    # 2 Newton iterations good to 208 bits
    # 3 Newton iterations good to 416 bits
    if x < _1_800:
        y = int(x**0.5)
        if x >= _1_100:
            y = (y + x//y) >> 1
            if x >= _1_200:
                y = (y + x//y) >> 1
                if x >= _1_400:
                    y = (y + x//y) >> 1
        return y
    bc = bitcount(x)
    guard_bits = 10
    x <<= 2*guard_bits
    bc += 2*guard_bits
    bc += (bc&1)
    hbc = bc//2
    startprec = min(50, hbc)
    # Newton iteration for 1/sqrt(x), with floating-point starting value
    r = int(2.0**(2*startprec) * (x >> (bc-2*startprec)) ** -0.5)
    pp = startprec
    for p in giant_steps(startprec, hbc):
        # r**2, scaled from real size 2**(-bc) to 2**p
        r2 = (r*r) >> (2*pp - p)
        # x*r**2, scaled from real size ~1.0 to 2**p
        xr2 = ((x >> (bc-p)) * r2) >> p
        # New value of r, scaled from real size 2**(-bc/2) to 2**p
        r = (r * ((3<<p) - xr2)) >> (pp+1)
        pp = p
    # (1/sqrt(x))*x = sqrt(x)
    return (r*(x>>hbc)) >> (p+guard_bits)

def sqrtrem_python(x):
    """Correctly rounded integer (floor) square root with remainder."""
    # to check cutoff:
    # plot(lambda x: timing(isqrt, 2**int(x)), [0,2000])
    if x < _1_600:
        y = isqrt_small_python(x)
        return y, x - y*y
    y = isqrt_fast_python(x) + 1
    rem = x - y*y
    # Correct remainder
    while rem < 0:
        y -= 1
        rem += (1+2*y)
    else:
        if rem:
            while rem > 2*(1+y):
                y += 1
                rem -= (1+2*y)
    return y, rem

def isqrt_python(x):
    """Integer square root with correct (floor) rounding."""
    return sqrtrem_python(x)[0]

def sqrt_fixed(x, prec):
    return isqrt_fast(x<<prec)

sqrt_fixed2 = sqrt_fixed

if BACKEND == 'gmpy':
    if gmpy.version() >= '2':
        isqrt_small = isqrt_fast = isqrt = gmpy.isqrt
        sqrtrem = gmpy.isqrt_rem
    else:
        isqrt_small = isqrt_fast = isqrt = gmpy.sqrt
        sqrtrem = gmpy.sqrtrem
elif BACKEND == 'sage':
    isqrt_small = isqrt_fast = isqrt = \
        getattr(sage_utils, "isqrt", lambda n: MPZ(n).isqrt())
    sqrtrem = lambda n: MPZ(n).sqrtrem()
else:
    isqrt_small = isqrt_small_python
    isqrt_fast = isqrt_fast_python
    isqrt = isqrt_python
    sqrtrem = sqrtrem_python


def ifib(n, _cache={}):
    """Computes the nth Fibonacci number as an integer, for
    integer n."""
    if n < 0:
        return (-1)**(-n+1) * ifib(-n)
    if n in _cache:
        return _cache[n]
    m = n
    # Use Dijkstra's logarithmic algorithm
    # The following implementation is basically equivalent to
    # http://en.literateprograms.org/Fibonacci_numbers_(Scheme)
    a, b, p, q = MPZ_ONE, MPZ_ZERO, MPZ_ZERO, MPZ_ONE
    while n:
        if n & 1:
            aq = a*q
            a, b = b*q+aq+a*p, b*p+aq
            n -= 1
        else:
            qq = q*q
            p, q = p*p+qq, qq+2*p*q
            n >>= 1
    if m < 250:
        _cache[m] = b
    return b

MAX_FACTORIAL_CACHE = 1000

def ifac(n, memo={0:1, 1:1}):
    """Return n factorial (for integers n >= 0 only)."""
    f = memo.get(n)
    if f:
        return f
    k = len(memo)
    p = memo[k-1]
    MAX = MAX_FACTORIAL_CACHE
    while k <= n:
        p *= k
        if k <= MAX:
            memo[k] = p
        k += 1
    return p

def ifac2(n, memo_pair=[{0:1}, {1:1}]):
    """Return n!! (double factorial), integers n >= 0 only."""
    memo = memo_pair[n&1]
    f = memo.get(n)
    if f:
        return f
    k = max(memo)
    p = memo[k]
    MAX = MAX_FACTORIAL_CACHE
    while k < n:
        k += 2
        p *= k
        if k <= MAX:
            memo[k] = p
    return p

if BACKEND == 'gmpy':
    ifac = gmpy.fac
elif BACKEND == 'sage':
    ifac = lambda n: int(sage.factorial(n))
    ifib = sage.fibonacci

def list_primes(n):
    n = n + 1
    sieve = list(xrange(n))
    sieve[:2] = [0, 0]
    for i in xrange(2, int(n**0.5)+1):
        if sieve[i]:
            for j in xrange(i**2, n, i):
                sieve[j] = 0
    return [p for p in sieve if p]

if BACKEND == 'sage':
    # Note: it is *VERY* important for performance that we convert
    # the list to Python ints.
    def list_primes(n):
        return [int(_) for _ in sage.primes(n+1)]

small_odd_primes = (3,5,7,11,13,17,19,23,29,31,37,41,43,47)
small_odd_primes_set = set(small_odd_primes)

def isprime(n):
    """
    Determines whether n is a prime number. A probabilistic test is
    performed if n is very large. No special trick is used for detecting
    perfect powers.

        >>> sum(list_primes(100000))
        454396537
        >>> sum(n*isprime(n) for n in range(100000))
        454396537

    """
    n = int(n)
    if not n & 1:
        return n == 2
    if n < 50:
        return n in small_odd_primes_set
    for p in small_odd_primes:
        if not n % p:
            return False
    m = n-1
    s = trailing(m)
    d = m >> s
    def test(a):
        x = pow(a,d,n)
        if x == 1 or x == m:
            return True
        for r in xrange(1,s):
            x = x**2 % n
            if x == m:
                return True
        return False
    # See http://primes.utm.edu/prove/prove2_3.html
    if n < 1373653:
        witnesses = [2,3]
    elif n < 341550071728321:
        witnesses = [2,3,5,7,11,13,17]
    else:
        witnesses = small_odd_primes
    for a in witnesses:
        if not test(a):
            return False
    return True

def moebius(n):
    """
    Evaluates the Moebius function which is `mu(n) = (-1)^k` if `n`
    is a product of `k` distinct primes and `mu(n) = 0` otherwise.

    TODO: speed up using factorization
    """
    n = abs(int(n))
    if n < 2:
        return n
    factors = []
    for p in xrange(2, n+1):
        if not (n % p):
            if not (n % p**2):
                return 0
            if not sum(p % f for f in factors):
                factors.append(p)
    return (-1)**len(factors)

def gcd(*args):
    a = 0
    for b in args:
        if a:
            while b:
                a, b = b, a % b
        else:
            a = b
    return a


#  Comment by Juan Arias de Reyna:
#
#  I learn this method to compute EulerE[2n] from van de Lune.
#
#  We apply the formula   EulerE[2n] = (-1)^n 2**(-2n) sum_{j=0}^n a(2n,2j+1)
#
#  where the numbers a(n,j) vanish for  j > n+1 or j <= -1  and satisfies
#
#  a(0,-1) = a(0,0) = 0;  a(0,1)= 1; a(0,2) = a(0,3) = 0
#
#  a(n,j) = a(n-1,j)                              when n+j is even
#  a(n,j) = (j-1) a(n-1,j-1) + (j+1) a(n-1,j+1)   when n+j is odd
#
#
#  But we can use only one array unidimensional a(j) since to compute
#  a(n,j) we only need to know a(n-1,k) where k and j are of different parity
#  and we have not to conserve the used values.
#
#  We cached up the values of Euler numbers to sufficiently high order.
#
#  Important Observation: If we pretend to use the numbers
#     EulerE[1], EulerE[2], ... , EulerE[n]
#     it is convenient to compute first EulerE[n], since the algorithm
#     computes first all
#     the previous ones, and keeps them in the CACHE

MAX_EULER_CACHE = 500

def eulernum(m, _cache={0:MPZ_ONE}):
    r"""
    Computes the Euler numbers `E(n)`, which can be defined as
    coefficients of the Taylor expansion of `1/cosh x`:

    .. math ::

        \frac{1}{\cosh x} = \sum_{n=0}^\infty \frac{E_n}{n!} x^n

    Example::

        >>> [int(eulernum(n)) for n in range(11)]
        [1, 0, -1, 0, 5, 0, -61, 0, 1385, 0, -50521]
        >>> [int(eulernum(n)) for n in range(11)]   # test cache
        [1, 0, -1, 0, 5, 0, -61, 0, 1385, 0, -50521]

    """
    # for odd m > 1, the Euler numbers are zero
    if m & 1:
        return MPZ_ZERO
    f = _cache.get(m)
    if f:
        return f
    MAX = MAX_EULER_CACHE
    n = m
    a = [MPZ(_) for _ in [0,0,1,0,0,0]]
    for  n in range(1, m+1):
        for j in range(n+1, -1, -2):
            a[j+1] = (j-1)*a[j] + (j+1)*a[j+2]
        a.append(0)
        suma = 0
        for k in range(n+1, -1, -2):
            suma += a[k+1]
            if n <= MAX:
                _cache[n] = ((-1)**(n//2))*(suma // 2**n)
        if n == m:
            return ((-1)**(n//2))*suma // 2**n

def stirling1(n, k):
    """
    Stirling number of the first kind.
    """
    if n < 0 or k < 0:
        raise ValueError
    if k >= n:
        return MPZ(n == k)
    if k < 1:
        return MPZ_ZERO
    L = [MPZ_ZERO] * (k+1)
    L[1] = MPZ_ONE
    for m in xrange(2, n+1):
        for j in xrange(min(k, m), 0, -1):
            L[j] = (m-1) * L[j] + L[j-1]
    return (-1)**(n+k) * L[k]

def stirling2(n, k):
    """
    Stirling number of the second kind.
    """
    if n < 0 or k < 0:
        raise ValueError
    if k >= n:
        return MPZ(n == k)
    if k <= 1:
        return MPZ(k == 1)
    s = MPZ_ZERO
    t = MPZ_ONE
    for j in xrange(k+1):
        if (k + j) & 1:
            s -= t * MPZ(j)**n
        else:
            s += t * MPZ(j)**n
        t = t * (k - j) // (j + 1)
    return s // ifac(k)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\drawing\geometry.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    Float,
    Integer,
    Bool,
    MinMax,
    Set,
    NoneSet,
    String,
    Alias,
)
from openpyxl.descriptors.excel import Coordinate, Percentage
from openpyxl.descriptors.excel import ExtensionList as OfficeArtExtensionList
from .line import LineProperties

from openpyxl.styles.colors import Color
from openpyxl.xml.constants import DRAWING_NS


class Point2D(Serialisable):

    tagname = "off"
    namespace = DRAWING_NS

    x = Coordinate()
    y = Coordinate()

    def __init__(self,
                 x=None,
                 y=None,
                ):
        self.x = x
        self.y = y


class PositiveSize2D(Serialisable):

    tagname = "ext"
    namespace = DRAWING_NS

    """
    Dimensions in EMUs
    """

    cx = Integer()
    width = Alias('cx')
    cy = Integer()
    height = Alias('cy')

    def __init__(self,
                 cx=None,
                 cy=None,
                ):
        self.cx = cx
        self.cy = cy


class Transform2D(Serialisable):

    tagname = "xfrm"
    namespace = DRAWING_NS

    rot = Integer(allow_none=True)
    flipH = Bool(allow_none=True)
    flipV = Bool(allow_none=True)
    off = Typed(expected_type=Point2D, allow_none=True)
    ext = Typed(expected_type=PositiveSize2D, allow_none=True)
    chOff = Typed(expected_type=Point2D, allow_none=True)
    chExt = Typed(expected_type=PositiveSize2D, allow_none=True)

    __elements__ = ('off', 'ext', 'chOff', 'chExt')

    def __init__(self,
                 rot=None,
                 flipH=None,
                 flipV=None,
                 off=None,
                 ext=None,
                 chOff=None,
                 chExt=None,
                ):
        self.rot = rot
        self.flipH = flipH
        self.flipV = flipV
        self.off = off
        self.ext = ext
        self.chOff = chOff
        self.chExt = chExt


class GroupTransform2D(Serialisable):

    tagname = "xfrm"
    namespace = DRAWING_NS

    rot = Integer(allow_none=True)
    flipH = Bool(allow_none=True)
    flipV = Bool(allow_none=True)
    off = Typed(expected_type=Point2D, allow_none=True)
    ext = Typed(expected_type=PositiveSize2D, allow_none=True)
    chOff = Typed(expected_type=Point2D, allow_none=True)
    chExt = Typed(expected_type=PositiveSize2D, allow_none=True)

    __elements__ = ("off", "ext", "chOff", "chExt")

    def __init__(self,
                 rot=0,
                 flipH=None,
                 flipV=None,
                 off=None,
                 ext=None,
                 chOff=None,
                 chExt=None,
                ):
        self.rot = rot
        self.flipH = flipH
        self.flipV = flipV
        self.off = off
        self.ext = ext
        self.chOff = chOff
        self.chExt = chExt


class SphereCoords(Serialisable):

    tagname = "sphereCoords" # usually

    lat = Integer()
    lon = Integer()
    rev = Integer()

    def __init__(self,
                 lat=None,
                 lon=None,
                 rev=None,
                ):
        self.lat = lat
        self.lon = lon
        self.rev = rev


class Camera(Serialisable):

    tagname = "camera"

    prst = Set(values=[
        'legacyObliqueTopLeft', 'legacyObliqueTop', 'legacyObliqueTopRight', 'legacyObliqueLeft',
         'legacyObliqueFront', 'legacyObliqueRight', 'legacyObliqueBottomLeft',
         'legacyObliqueBottom', 'legacyObliqueBottomRight', 'legacyPerspectiveTopLeft',
         'legacyPerspectiveTop', 'legacyPerspectiveTopRight', 'legacyPerspectiveLeft',
         'legacyPerspectiveFront', 'legacyPerspectiveRight', 'legacyPerspectiveBottomLeft',
         'legacyPerspectiveBottom', 'legacyPerspectiveBottomRight', 'orthographicFront',
         'isometricTopUp', 'isometricTopDown', 'isometricBottomUp', 'isometricBottomDown',
         'isometricLeftUp', 'isometricLeftDown', 'isometricRightUp', 'isometricRightDown',
         'isometricOffAxis1Left', 'isometricOffAxis1Right', 'isometricOffAxis1Top',
         'isometricOffAxis2Left', 'isometricOffAxis2Right', 'isometricOffAxis2Top',
         'isometricOffAxis3Left', 'isometricOffAxis3Right', 'isometricOffAxis3Bottom',
         'isometricOffAxis4Left', 'isometricOffAxis4Right', 'isometricOffAxis4Bottom',
         'obliqueTopLeft',  'obliqueTop', 'obliqueTopRight', 'obliqueLeft', 'obliqueRight',
         'obliqueBottomLeft', 'obliqueBottom', 'obliqueBottomRight', 'perspectiveFront',
         'perspectiveLeft', 'perspectiveRight', 'perspectiveAbove', 'perspectiveBelow',
         'perspectiveAboveLeftFacing', 'perspectiveAboveRightFacing',
         'perspectiveContrastingLeftFacing', 'perspectiveContrastingRightFacing',
         'perspectiveHeroicLeftFacing', 'perspectiveHeroicRightFacing',
         'perspectiveHeroicExtremeLeftFacing', 'perspectiveHeroicExtremeRightFacing',
         'perspectiveRelaxed', 'perspectiveRelaxedModerately'])
    fov = Integer(allow_none=True)
    zoom = Typed(expected_type=Percentage, allow_none=True)
    rot = Typed(expected_type=SphereCoords, allow_none=True)


    def __init__(self,
                 prst=None,
                 fov=None,
                 zoom=None,
                 rot=None,
                ):
        self.prst = prst
        self.fov = fov
        self.zoom = zoom
        self.rot = rot


class LightRig(Serialisable):

    tagname = "lightRig"

    rig = Set(values=['legacyFlat1', 'legacyFlat2', 'legacyFlat3', 'legacyFlat4', 'legacyNormal1',
         'legacyNormal2', 'legacyNormal3', 'legacyNormal4', 'legacyHarsh1',
         'legacyHarsh2', 'legacyHarsh3', 'legacyHarsh4', 'threePt', 'balanced',
         'soft', 'harsh', 'flood', 'contrasting', 'morning', 'sunrise', 'sunset',
         'chilly', 'freezing', 'flat', 'twoPt', 'glow', 'brightRoom']
    )
    dir = Set(values=(['tl', 't', 'tr', 'l', 'r', 'bl', 'b', 'br']))
    rot = Typed(expected_type=SphereCoords, allow_none=True)

    def __init__(self,
                 rig=None,
                 dir=None,
                 rot=None,
                ):
        self.rig = rig
        self.dir = dir
        self.rot = rot


class Vector3D(Serialisable):

    tagname = "vector"

    dx = Integer() # can be in or universl measure :-/
    dy = Integer()
    dz = Integer()

    def __init__(self,
                 dx=None,
                 dy=None,
                 dz=None,
                ):
        self.dx = dx
        self.dy = dy
        self.dz = dz


class Point3D(Serialisable):

    tagname = "anchor"

    x = Integer()
    y = Integer()
    z = Integer()

    def __init__(self,
                 x=None,
                 y=None,
                 z=None,
                ):
        self.x = x
        self.y = y
        self.z = z


class Backdrop(Serialisable):

    anchor = Typed(expected_type=Point3D, )
    norm = Typed(expected_type=Vector3D, )
    up = Typed(expected_type=Vector3D, )
    extLst = Typed(expected_type=OfficeArtExtensionList, allow_none=True)

    def __init__(self,
                 anchor=None,
                 norm=None,
                 up=None,
                 extLst=None,
                ):
        self.anchor = anchor
        self.norm = norm
        self.up = up
        self.extLst = extLst


class Scene3D(Serialisable):

    camera = Typed(expected_type=Camera, )
    lightRig = Typed(expected_type=LightRig, )
    backdrop = Typed(expected_type=Backdrop, allow_none=True)
    extLst = Typed(expected_type=OfficeArtExtensionList, allow_none=True)

    def __init__(self,
                 camera=None,
                 lightRig=None,
                 backdrop=None,
                 extLst=None,
                ):
        self.camera = camera
        self.lightRig = lightRig
        self.backdrop = backdrop
        self.extLst = extLst


class Bevel(Serialisable):

    tagname = "bevel"

    w = Integer()
    h = Integer()
    prst = NoneSet(values=
               ['relaxedInset', 'circle', 'slope', 'cross', 'angle',
                'softRound', 'convex', 'coolSlant', 'divot', 'riblet',
                 'hardEdge', 'artDeco']
               )

    def __init__(self,
                 w=None,
                 h=None,
                 prst=None,
                ):
        self.w = w
        self.h = h
        self.prst = prst


class Shape3D(Serialisable):

    namespace = DRAWING_NS

    z = Typed(expected_type=Coordinate, allow_none=True)
    extrusionH = Integer(allow_none=True)
    contourW = Integer(allow_none=True)
    prstMaterial = NoneSet(values=[
        'legacyMatte','legacyPlastic', 'legacyMetal', 'legacyWireframe', 'matte', 'plastic',
        'metal', 'warmMatte', 'translucentPowder', 'powder', 'dkEdge',
        'softEdge', 'clear', 'flat', 'softmetal']
                       )
    bevelT = Typed(expected_type=Bevel, allow_none=True)
    bevelB = Typed(expected_type=Bevel, allow_none=True)
    extrusionClr = Typed(expected_type=Color, allow_none=True)
    contourClr = Typed(expected_type=Color, allow_none=True)
    extLst = Typed(expected_type=OfficeArtExtensionList, allow_none=True)

    def __init__(self,
                 z=None,
                 extrusionH=None,
                 contourW=None,
                 prstMaterial=None,
                 bevelT=None,
                 bevelB=None,
                 extrusionClr=None,
                 contourClr=None,
                 extLst=None,
                ):
        self.z = z
        self.extrusionH = extrusionH
        self.contourW = contourW
        self.prstMaterial = prstMaterial
        self.bevelT = bevelT
        self.bevelB = bevelB
        self.extrusionClr = extrusionClr
        self.contourClr = contourClr
        self.extLst = extLst


class Path2D(Serialisable):

    w = Float()
    h = Float()
    fill = NoneSet(values=(['norm', 'lighten', 'lightenLess', 'darken', 'darkenLess']))
    stroke = Bool(allow_none=True)
    extrusionOk = Bool(allow_none=True)

    def __init__(self,
                 w=None,
                 h=None,
                 fill=None,
                 stroke=None,
                 extrusionOk=None,
                ):
        self.w = w
        self.h = h
        self.fill = fill
        self.stroke = stroke
        self.extrusionOk = extrusionOk


class Path2DList(Serialisable):

    path = Typed(expected_type=Path2D, allow_none=True)

    def __init__(self,
                 path=None,
                ):
        self.path = path


class GeomRect(Serialisable):

    l = Coordinate()
    t = Coordinate()
    r = Coordinate()
    b = Coordinate()

    def __init__(self,
                 l=None,
                 t=None,
                 r=None,
                 b=None,
                ):
        self.l = l
        self.t = t
        self.r = r
        self.b = b


class AdjPoint2D(Serialisable):

    x = Coordinate()
    y = Coordinate()

    def __init__(self,
                 x=None,
                 y=None,
                ):
        self.x = x
        self.y = y


class ConnectionSite(Serialisable):

    ang = MinMax(min=0, max=360) # guess work, can also be a name
    pos = Typed(expected_type=AdjPoint2D, )

    def __init__(self,
                 ang=None,
                 pos=None,
                ):
        self.ang = ang
        self.pos = pos


class ConnectionSiteList(Serialisable):

    cxn = Typed(expected_type=ConnectionSite, allow_none=True)

    def __init__(self,
                 cxn=None,
                ):
        self.cxn = cxn


class AdjustHandleList(Serialisable):

    pass

class GeomGuide(Serialisable):

    name = String()
    fmla = String()

    def __init__(self,
                 name=None,
                 fmla=None,
                ):
        self.name = name
        self.fmla = fmla


class GeomGuideList(Serialisable):

    gd = Typed(expected_type=GeomGuide, allow_none=True)

    def __init__(self,
                 gd=None,
                ):
        self.gd = gd


class CustomGeometry2D(Serialisable):

    avLst = Typed(expected_type=GeomGuideList, allow_none=True)
    gdLst = Typed(expected_type=GeomGuideList, allow_none=True)
    ahLst = Typed(expected_type=AdjustHandleList, allow_none=True)
    cxnLst = Typed(expected_type=ConnectionSiteList, allow_none=True)
    #rect = Typed(expected_type=GeomRect, allow_none=True)
    pathLst = Typed(expected_type=Path2DList, )

    def __init__(self,
                 avLst=None,
                 gdLst=None,
                 ahLst=None,
                 cxnLst=None,
                 rect=None,
                 pathLst=None,
                ):
        self.avLst = avLst
        self.gdLst = gdLst
        self.ahLst = ahLst
        self.cxnLst = cxnLst
        self.rect = None
        self.pathLst = pathLst


class PresetGeometry2D(Serialisable):

    namespace = DRAWING_NS

    prst = Set(values=(
        ['line', 'lineInv', 'triangle', 'rtTriangle', 'rect',
         'diamond', 'parallelogram', 'trapezoid', 'nonIsoscelesTrapezoid',
         'pentagon', 'hexagon', 'heptagon', 'octagon', 'decagon', 'dodecagon',
         'star4', 'star5', 'star6', 'star7', 'star8', 'star10', 'star12',
         'star16', 'star24', 'star32', 'roundRect', 'round1Rect',
         'round2SameRect', 'round2DiagRect', 'snipRoundRect', 'snip1Rect',
         'snip2SameRect', 'snip2DiagRect', 'plaque', 'ellipse', 'teardrop',
         'homePlate', 'chevron', 'pieWedge', 'pie', 'blockArc', 'donut',
         'noSmoking', 'rightArrow', 'leftArrow', 'upArrow', 'downArrow',
         'stripedRightArrow', 'notchedRightArrow', 'bentUpArrow',
         'leftRightArrow', 'upDownArrow', 'leftUpArrow', 'leftRightUpArrow',
         'quadArrow', 'leftArrowCallout', 'rightArrowCallout', 'upArrowCallout',
         'downArrowCallout', 'leftRightArrowCallout', 'upDownArrowCallout',
         'quadArrowCallout', 'bentArrow', 'uturnArrow', 'circularArrow',
         'leftCircularArrow', 'leftRightCircularArrow', 'curvedRightArrow',
         'curvedLeftArrow', 'curvedUpArrow', 'curvedDownArrow', 'swooshArrow',
         'cube', 'can', 'lightningBolt', 'heart', 'sun', 'moon', 'smileyFace',
         'irregularSeal1', 'irregularSeal2', 'foldedCorner', 'bevel', 'frame',
         'halfFrame', 'corner', 'diagStripe', 'chord', 'arc', 'leftBracket',
         'rightBracket', 'leftBrace', 'rightBrace', 'bracketPair', 'bracePair',
         'straightConnector1', 'bentConnector2', 'bentConnector3',
         'bentConnector4', 'bentConnector5', 'curvedConnector2',
         'curvedConnector3', 'curvedConnector4', 'curvedConnector5', 'callout1',
         'callout2', 'callout3', 'accentCallout1', 'accentCallout2',
         'accentCallout3', 'borderCallout1', 'borderCallout2', 'borderCallout3',
         'accentBorderCallout1', 'accentBorderCallout2', 'accentBorderCallout3',
         'wedgeRectCallout', 'wedgeRoundRectCallout', 'wedgeEllipseCallout',
         'cloudCallout', 'cloud', 'ribbon', 'ribbon2', 'ellipseRibbon',
         'ellipseRibbon2', 'leftRightRibbon', 'verticalScroll',
         'horizontalScroll', 'wave', 'doubleWave', 'plus', 'flowChartProcess',
         'flowChartDecision', 'flowChartInputOutput',
         'flowChartPredefinedProcess', 'flowChartInternalStorage',
         'flowChartDocument', 'flowChartMultidocument', 'flowChartTerminator',
         'flowChartPreparation', 'flowChartManualInput',
         'flowChartManualOperation', 'flowChartConnector', 'flowChartPunchedCard',
         'flowChartPunchedTape', 'flowChartSummingJunction', 'flowChartOr',
         'flowChartCollate', 'flowChartSort', 'flowChartExtract',
         'flowChartMerge', 'flowChartOfflineStorage', 'flowChartOnlineStorage',
         'flowChartMagneticTape', 'flowChartMagneticDisk',
         'flowChartMagneticDrum', 'flowChartDisplay', 'flowChartDelay',
         'flowChartAlternateProcess', 'flowChartOffpageConnector',
         'actionButtonBlank', 'actionButtonHome', 'actionButtonHelp',
         'actionButtonInformation', 'actionButtonForwardNext',
         'actionButtonBackPrevious', 'actionButtonEnd', 'actionButtonBeginning',
         'actionButtonReturn', 'actionButtonDocument', 'actionButtonSound',
         'actionButtonMovie', 'gear6', 'gear9', 'funnel', 'mathPlus', 'mathMinus',
         'mathMultiply', 'mathDivide', 'mathEqual', 'mathNotEqual', 'cornerTabs',
         'squareTabs', 'plaqueTabs', 'chartX', 'chartStar', 'chartPlus']))
    avLst = Typed(expected_type=GeomGuideList, allow_none=True)

    def __init__(self,
                 prst=None,
                 avLst=None,
                ):
        self.prst = prst
        self.avLst = avLst


class FontReference(Serialisable):

    idx = NoneSet(values=(['major', 'minor']))

    def __init__(self,
                 idx=None,
                ):
        self.idx = idx


class StyleMatrixReference(Serialisable):

    idx = Integer()

    def __init__(self,
                 idx=None,
                ):
        self.idx = idx


class ShapeStyle(Serialisable):

    lnRef = Typed(expected_type=StyleMatrixReference, )
    fillRef = Typed(expected_type=StyleMatrixReference, )
    effectRef = Typed(expected_type=StyleMatrixReference, )
    fontRef = Typed(expected_type=FontReference, )

    def __init__(self,
                 lnRef=None,
                 fillRef=None,
                 effectRef=None,
                 fontRef=None,
                ):
        self.lnRef = lnRef
        self.fillRef = fillRef
        self.effectRef = effectRef
        self.fontRef = fontRef

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\polyutils.py ===
"""Useful utilities for higher level polynomial classes. """

from __future__ import annotations

from sympy.external.gmpy import GROUND_TYPES

from sympy.core import (S, Add, Mul, Pow, Eq, Expr,
    expand_mul, expand_multinomial)
from sympy.core.exprtools import decompose_power, decompose_power_rat
from sympy.core.numbers import _illegal
from sympy.polys.polyerrors import PolynomialError, GeneratorsError
from sympy.polys.polyoptions import build_options

import re


_gens_order = {
    'a': 301, 'b': 302, 'c': 303, 'd': 304,
    'e': 305, 'f': 306, 'g': 307, 'h': 308,
    'i': 309, 'j': 310, 'k': 311, 'l': 312,
    'm': 313, 'n': 314, 'o': 315, 'p': 216,
    'q': 217, 'r': 218, 's': 219, 't': 220,
    'u': 221, 'v': 222, 'w': 223, 'x': 124,
    'y': 125, 'z': 126,
}

_max_order = 1000
_re_gen = re.compile(r"^(.*?)(\d*)$", re.MULTILINE)


def _nsort(roots, separated=False):
    """Sort the numerical roots putting the real roots first, then sorting
    according to real and imaginary parts. If ``separated`` is True, then
    the real and imaginary roots will be returned in two lists, respectively.

    This routine tries to avoid issue 6137 by separating the roots into real
    and imaginary parts before evaluation. In addition, the sorting will raise
    an error if any computation cannot be done with precision.
    """
    if not all(r.is_number for r in roots):
        raise NotImplementedError
    if not len(roots):
        return [] if not separated else ([], [])
    # see issue 6137:
    # get the real part of the evaluated real and imaginary parts of each root
    key = [[i.n(2).as_real_imag()[0] for i in r.as_real_imag()] for r in roots]
    # make sure the parts were computed with precision
    if len(roots) > 1 and any(i._prec == 1 for k in key for i in k):
        raise NotImplementedError("could not compute root with precision")
    # insert a key to indicate if the root has an imaginary part
    key = [(1 if i else 0, r, i) for r, i in key]
    key = sorted(zip(key, roots))
    # return the real and imaginary roots separately if desired
    if separated:
        r = []
        i = []
        for (im, _, _), v in key:
            if im:
                i.append(v)
            else:
                r.append(v)
        return r, i
    _, roots = zip(*key)
    return list(roots)


def _sort_gens(gens, **args):
    """Sort generators in a reasonably intelligent way. """
    opt = build_options(args)

    gens_order, wrt = {}, None

    if opt is not None:
        gens_order, wrt = {}, opt.wrt

        for i, gen in enumerate(opt.sort):
            gens_order[gen] = i + 1

    def order_key(gen):
        gen = str(gen)

        if wrt is not None:
            try:
                return (-len(wrt) + wrt.index(gen), gen, 0)
            except ValueError:
                pass

        name, index = _re_gen.match(gen).groups()

        if index:
            index = int(index)
        else:
            index = 0

        try:
            return ( gens_order[name], name, index)
        except KeyError:
            pass

        try:
            return (_gens_order[name], name, index)
        except KeyError:
            pass

        return (_max_order, name, index)

    try:
        gens = sorted(gens, key=order_key)
    except TypeError:  # pragma: no cover
        pass

    return tuple(gens)


def _unify_gens(f_gens, g_gens):
    """Unify generators in a reasonably intelligent way. """
    f_gens = list(f_gens)
    g_gens = list(g_gens)

    if f_gens == g_gens:
        return tuple(f_gens)

    gens, common, k = [], [], 0

    for gen in f_gens:
        if gen in g_gens:
            common.append(gen)

    for i, gen in enumerate(g_gens):
        if gen in common:
            g_gens[i], k = common[k], k + 1

    for gen in common:
        i = f_gens.index(gen)

        gens.extend(f_gens[:i])
        f_gens = f_gens[i + 1:]

        i = g_gens.index(gen)

        gens.extend(g_gens[:i])
        g_gens = g_gens[i + 1:]

        gens.append(gen)

    gens.extend(f_gens)
    gens.extend(g_gens)

    return tuple(gens)


def _analyze_gens(gens):
    """Support for passing generators as `*gens` and `[gens]`. """
    if len(gens) == 1 and hasattr(gens[0], '__iter__'):
        return tuple(gens[0])
    else:
        return tuple(gens)


def _sort_factors(factors, **args):
    """Sort low-level factors in increasing 'complexity' order. """

    # XXX: GF(p) does not support comparisons so we need a key function to sort
    # the factors if python-flint is being used. A better solution might be to
    # add a sort key method to each domain.
    def order_key(factor):
        if isinstance(factor, _GF_types):
            return int(factor)
        elif isinstance(factor, list):
            return [order_key(f) for f in factor]
        else:
            return factor

    def order_if_multiple_key(factor):
        (f, n) = factor
        return (len(f), n, order_key(f))

    def order_no_multiple_key(f):
        return (len(f), order_key(f))

    if args.get('multiple', True):
        return sorted(factors, key=order_if_multiple_key)
    else:
        return sorted(factors, key=order_no_multiple_key)


illegal_types = [type(obj) for obj in _illegal]
finf = [float(i) for i in _illegal[1:3]]


def _not_a_coeff(expr):
    """Do not treat NaN and infinities as valid polynomial coefficients. """
    if type(expr) in illegal_types or expr in finf:
        return True
    if isinstance(expr, float) and float(expr) != expr:
        return True  # nan
    return  # could be


def _parallel_dict_from_expr_if_gens(exprs, opt):
    """Transform expressions into a multinomial form given generators. """
    k, indices = len(opt.gens), {}

    for i, g in enumerate(opt.gens):
        indices[g] = i

    polys = []

    for expr in exprs:
        poly = {}

        if expr.is_Equality:
            expr = expr.lhs - expr.rhs

        for term in Add.make_args(expr):
            coeff, monom = [], [0]*k

            for factor in Mul.make_args(term):
                if not _not_a_coeff(factor) and factor.is_Number:
                    coeff.append(factor)
                else:
                    try:
                        if opt.series is False:
                            base, exp = decompose_power(factor)

                            if exp < 0:
                                exp, base = -exp, Pow(base, -S.One)
                        else:
                            base, exp = decompose_power_rat(factor)

                        monom[indices[base]] = exp
                    except KeyError:
                        if not factor.has_free(*opt.gens):
                            coeff.append(factor)
                        else:
                            raise PolynomialError("%s contains an element of "
                                                  "the set of generators." % factor)

            monom = tuple(monom)

            if monom in poly:
                poly[monom] += Mul(*coeff)
            else:
                poly[monom] = Mul(*coeff)

        polys.append(poly)

    return polys, opt.gens


def _parallel_dict_from_expr_no_gens(exprs, opt):
    """Transform expressions into a multinomial form and figure out generators. """
    if opt.domain is not None:
        def _is_coeff(factor):
            return factor in opt.domain
    elif opt.extension is True:
        def _is_coeff(factor):
            return factor.is_algebraic
    elif opt.greedy is not False:
        def _is_coeff(factor):
            return factor is S.ImaginaryUnit
    else:
        def _is_coeff(factor):
            return factor.is_number

    gens, reprs = set(), []

    for expr in exprs:
        terms = []

        if expr.is_Equality:
            expr = expr.lhs - expr.rhs

        for term in Add.make_args(expr):
            coeff, elements = [], {}

            for factor in Mul.make_args(term):
                if not _not_a_coeff(factor) and (factor.is_Number or _is_coeff(factor)):
                    coeff.append(factor)
                else:
                    if opt.series is False:
                        base, exp = decompose_power(factor)

                        if exp < 0:
                            exp, base = -exp, Pow(base, -S.One)
                    else:
                        base, exp = decompose_power_rat(factor)

                    elements[base] = elements.setdefault(base, 0) + exp
                    gens.add(base)

            terms.append((coeff, elements))

        reprs.append(terms)

    gens = _sort_gens(gens, opt=opt)
    k, indices = len(gens), {}

    for i, g in enumerate(gens):
        indices[g] = i

    polys = []

    for terms in reprs:
        poly = {}

        for coeff, term in terms:
            monom = [0]*k

            for base, exp in term.items():
                monom[indices[base]] = exp

            monom = tuple(monom)

            if monom in poly:
                poly[monom] += Mul(*coeff)
            else:
                poly[monom] = Mul(*coeff)

        polys.append(poly)

    return polys, tuple(gens)


def _dict_from_expr_if_gens(expr, opt):
    """Transform an expression into a multinomial form given generators. """
    (poly,), gens = _parallel_dict_from_expr_if_gens((expr,), opt)
    return poly, gens


def _dict_from_expr_no_gens(expr, opt):
    """Transform an expression into a multinomial form and figure out generators. """
    (poly,), gens = _parallel_dict_from_expr_no_gens((expr,), opt)
    return poly, gens


def parallel_dict_from_expr(exprs, **args):
    """Transform expressions into a multinomial form. """
    reps, opt = _parallel_dict_from_expr(exprs, build_options(args))
    return reps, opt.gens


def _parallel_dict_from_expr(exprs, opt):
    """Transform expressions into a multinomial form. """
    if opt.expand is not False:
        exprs = [ expr.expand() for expr in exprs ]

    if any(expr.is_commutative is False for expr in exprs):
        raise PolynomialError('non-commutative expressions are not supported')

    if opt.gens:
        reps, gens = _parallel_dict_from_expr_if_gens(exprs, opt)
    else:
        reps, gens = _parallel_dict_from_expr_no_gens(exprs, opt)

    return reps, opt.clone({'gens': gens})


def dict_from_expr(expr, **args):
    """Transform an expression into a multinomial form. """
    rep, opt = _dict_from_expr(expr, build_options(args))
    return rep, opt.gens


def _dict_from_expr(expr, opt):
    """Transform an expression into a multinomial form. """
    if expr.is_commutative is False:
        raise PolynomialError('non-commutative expressions are not supported')

    def _is_expandable_pow(expr):
        return (expr.is_Pow and expr.exp.is_positive and expr.exp.is_Integer
                and expr.base.is_Add)

    if opt.expand is not False:
        if not isinstance(expr, (Expr, Eq)):
            raise PolynomialError('expression must be of type Expr')
        expr = expr.expand()
        # TODO: Integrate this into expand() itself
        while any(_is_expandable_pow(i) or i.is_Mul and
            any(_is_expandable_pow(j) for j in i.args) for i in
                Add.make_args(expr)):

            expr = expand_multinomial(expr)
        while any(i.is_Mul and any(j.is_Add for j in i.args) for i in Add.make_args(expr)):
            expr = expand_mul(expr)

    if opt.gens:
        rep, gens = _dict_from_expr_if_gens(expr, opt)
    else:
        rep, gens = _dict_from_expr_no_gens(expr, opt)

    return rep, opt.clone({'gens': gens})


def expr_from_dict(rep, *gens):
    """Convert a multinomial form into an expression. """
    result = []

    for monom, coeff in rep.items():
        term = [coeff]
        for g, m in zip(gens, monom):
            if m:
                term.append(Pow(g, m))

        result.append(Mul(*term))

    return Add(*result)

parallel_dict_from_basic = parallel_dict_from_expr
dict_from_basic = dict_from_expr
basic_from_dict = expr_from_dict


def _dict_reorder(rep, gens, new_gens):
    """Reorder levels using dict representation. """
    gens = list(gens)

    monoms = rep.keys()
    coeffs = rep.values()

    new_monoms = [ [] for _ in range(len(rep)) ]
    used_indices = set()

    for gen in new_gens:
        try:
            j = gens.index(gen)
            used_indices.add(j)

            for M, new_M in zip(monoms, new_monoms):
                new_M.append(M[j])
        except ValueError:
            for new_M in new_monoms:
                new_M.append(0)

    for i, _ in enumerate(gens):
        if i not in used_indices:
            for monom in monoms:
                if monom[i]:
                    raise GeneratorsError("unable to drop generators")

    return map(tuple, new_monoms), coeffs


class PicklableWithSlots:
    """
    Mixin class that allows to pickle objects with ``__slots__``.

    Examples
    ========

    First define a class that mixes :class:`PicklableWithSlots` in::

        >>> from sympy.polys.polyutils import PicklableWithSlots
        >>> class Some(PicklableWithSlots):
        ...     __slots__ = ('foo', 'bar')
        ...
        ...     def __init__(self, foo, bar):
        ...         self.foo = foo
        ...         self.bar = bar

    To make :mod:`pickle` happy in doctest we have to use these hacks::

        >>> import builtins
        >>> builtins.Some = Some
        >>> from sympy.polys import polyutils
        >>> polyutils.Some = Some

    Next lets see if we can create an instance, pickle it and unpickle::

        >>> some = Some('abc', 10)
        >>> some.foo, some.bar
        ('abc', 10)

        >>> from pickle import dumps, loads
        >>> some2 = loads(dumps(some))

        >>> some2.foo, some2.bar
        ('abc', 10)

    """

    __slots__ = ()

    def __getstate__(self, cls=None):
        if cls is None:
            # This is the case for the instance that gets pickled
            cls = self.__class__

        d = {}

        # Get all data that should be stored from super classes
        for c in cls.__bases__:
            # XXX: Python 3.11 defines object.__getstate__ and it does not
            # accept any arguments so we need to make sure not to call it with
            # an argument here. To be compatible with Python < 3.11 we need to
            # be careful not to assume that c or object has a __getstate__
            # method though.
            getstate = getattr(c, "__getstate__", None)
            objstate = getattr(object, "__getstate__", None)
            if getstate is not None and getstate is not objstate:
                d.update(getstate(self, c))

        # Get all information that should be stored from cls and return the dict
        for name in cls.__slots__:
            if hasattr(self, name):
                d[name] = getattr(self, name)

        return d

    def __setstate__(self, d):
        # All values that were pickled are now assigned to a fresh instance
        for name, value in d.items():
            setattr(self, name, value)


class IntegerPowerable:
    r"""
    Mixin class for classes that define a `__mul__` method, and want to be
    raised to integer powers in the natural way that follows. Implements
    powering via binary expansion, for efficiency.

    By default, only integer powers $\geq 2$ are supported. To support the
    first, zeroth, or negative powers, override the corresponding methods,
    `_first_power`, `_zeroth_power`, `_negative_power`, below.
    """

    def __pow__(self, e, modulo=None):
        if e < 2:
            try:
                if e == 1:
                    return self._first_power()
                elif e == 0:
                    return self._zeroth_power()
                else:
                    return self._negative_power(e, modulo=modulo)
            except NotImplementedError:
                return NotImplemented
        else:
            bits = [int(d) for d in reversed(bin(e)[2:])]
            n = len(bits)
            p = self
            first = True
            for i in range(n):
                if bits[i]:
                    if first:
                        r = p
                        first = False
                    else:
                        r *= p
                        if modulo is not None:
                            r %= modulo
                if i < n - 1:
                    p *= p
                    if modulo is not None:
                        p %= modulo
            return r

    def _negative_power(self, e, modulo=None):
        """
        Compute inverse of self, then raise that to the abs(e) power.
        For example, if the class has an `inv()` method,
            return self.inv() ** abs(e) % modulo
        """
        raise NotImplementedError

    def _zeroth_power(self):
        """Return unity element of algebraic struct to which self belongs."""
        raise NotImplementedError

    def _first_power(self):
        """Return a copy of self."""
        raise NotImplementedError


_GF_types: tuple[type, ...]


if GROUND_TYPES == 'flint':
    import flint
    _GF_types = (flint.nmod, flint.fmpz_mod)
else:
    from sympy.polys.domains.modularinteger import ModularInteger
    flint = None
    _GF_types = (ModularInteger,)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\dtypes\base.py ===
"""
Extend pandas with custom array types.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    cast,
    overload,
)

import numpy as np

from pandas._libs import missing as libmissing
from pandas._libs.hashtable import object_hash
from pandas._libs.properties import cache_readonly
from pandas.errors import AbstractMethodError

from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCSeries,
)

if TYPE_CHECKING:
    from pandas._typing import (
        DtypeObj,
        Self,
        Shape,
        npt,
        type_t,
    )

    from pandas import Index
    from pandas.core.arrays import ExtensionArray

    # To parameterize on same ExtensionDtype
    ExtensionDtypeT = TypeVar("ExtensionDtypeT", bound="ExtensionDtype")


class ExtensionDtype:
    """
    A custom data type, to be paired with an ExtensionArray.

    See Also
    --------
    extensions.register_extension_dtype: Register an ExtensionType
        with pandas as class decorator.
    extensions.ExtensionArray: Abstract base class for custom 1-D array types.

    Notes
    -----
    The interface includes the following abstract methods that must
    be implemented by subclasses:

    * type
    * name
    * construct_array_type

    The following attributes and methods influence the behavior of the dtype in
    pandas operations

    * _is_numeric
    * _is_boolean
    * _get_common_dtype

    The `na_value` class attribute can be used to set the default NA value
    for this type. :attr:`numpy.nan` is used by default.

    ExtensionDtypes are required to be hashable. The base class provides
    a default implementation, which relies on the ``_metadata`` class
    attribute. ``_metadata`` should be a tuple containing the strings
    that define your data type. For example, with ``PeriodDtype`` that's
    the ``freq`` attribute.

    **If you have a parametrized dtype you should set the ``_metadata``
    class property**.

    Ideally, the attributes in ``_metadata`` will match the
    parameters to your ``ExtensionDtype.__init__`` (if any). If any of
    the attributes in ``_metadata`` don't implement the standard
    ``__eq__`` or ``__hash__``, the default implementations here will not
    work.

    Examples
    --------

    For interaction with Apache Arrow (pyarrow), a ``__from_arrow__`` method
    can be implemented: this method receives a pyarrow Array or ChunkedArray
    as only argument and is expected to return the appropriate pandas
    ExtensionArray for this dtype and the passed values:

    >>> import pyarrow
    >>> from pandas.api.extensions import ExtensionArray
    >>> class ExtensionDtype:
    ...     def __from_arrow__(
    ...         self,
    ...         array: pyarrow.Array | pyarrow.ChunkedArray
    ...     ) -> ExtensionArray:
    ...         ...

    This class does not inherit from 'abc.ABCMeta' for performance reasons.
    Methods and properties required by the interface raise
    ``pandas.errors.AbstractMethodError`` and no ``register`` method is
    provided for registering virtual subclasses.
    """

    _metadata: tuple[str, ...] = ()

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        """
        Check whether 'other' is equal to self.

        By default, 'other' is considered equal if either

        * it's a string matching 'self.name'.
        * it's an instance of this type and all of the attributes
          in ``self._metadata`` are equal between `self` and `other`.

        Parameters
        ----------
        other : Any

        Returns
        -------
        bool
        """
        if isinstance(other, str):
            try:
                other = self.construct_from_string(other)
            except TypeError:
                return False
        if isinstance(other, type(self)):
            return all(
                getattr(self, attr) == getattr(other, attr) for attr in self._metadata
            )
        return False

    def __hash__(self) -> int:
        # for python>=3.10, different nan objects have different hashes
        # we need to avoid that and thus use hash function with old behavior
        return object_hash(tuple(getattr(self, attr) for attr in self._metadata))

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    @property
    def na_value(self) -> object:
        """
        Default NA value to use for this type.

        This is used in e.g. ExtensionArray.take. This should be the
        user-facing "boxed" version of the NA value, not the physical NA value
        for storage.  e.g. for JSONArray, this is an empty dictionary.
        """
        return np.nan

    @property
    def type(self) -> type_t[Any]:
        """
        The scalar type for the array, e.g. ``int``

        It's expected ``ExtensionArray[item]`` returns an instance
        of ``ExtensionDtype.type`` for scalar ``item``, assuming
        that value is valid (not NA). NA values do not need to be
        instances of `type`.
        """
        raise AbstractMethodError(self)

    @property
    def kind(self) -> str:
        """
        A character code (one of 'biufcmMOSUV'), default 'O'

        This should match the NumPy dtype used when the array is
        converted to an ndarray, which is probably 'O' for object if
        the extension type cannot be represented as a built-in NumPy
        type.

        See Also
        --------
        numpy.dtype.kind
        """
        return "O"

    @property
    def name(self) -> str:
        """
        A string identifying the data type.

        Will be used for display in, e.g. ``Series.dtype``
        """
        raise AbstractMethodError(self)

    @property
    def names(self) -> list[str] | None:
        """
        Ordered list of field names, or None if there are no fields.

        This is for compatibility with NumPy arrays, and may be removed in the
        future.
        """
        return None

    @classmethod
    def construct_array_type(cls) -> type_t[ExtensionArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        raise AbstractMethodError(cls)

    def empty(self, shape: Shape) -> ExtensionArray:
        """
        Construct an ExtensionArray of this dtype with the given shape.

        Analogous to numpy.empty.

        Parameters
        ----------
        shape : int or tuple[int]

        Returns
        -------
        ExtensionArray
        """
        cls = self.construct_array_type()
        return cls._empty(shape, dtype=self)

    @classmethod
    def construct_from_string(cls, string: str) -> Self:
        r"""
        Construct this type from a string.

        This is useful mainly for data types that accept parameters.
        For example, a period dtype accepts a frequency parameter that
        can be set as ``period[h]`` (where H means hourly frequency).

        By default, in the abstract class, just the name of the type is
        expected. But subclasses can overwrite this method to accept
        parameters.

        Parameters
        ----------
        string : str
            The name of the type, for example ``category``.

        Returns
        -------
        ExtensionDtype
            Instance of the dtype.

        Raises
        ------
        TypeError
            If a class cannot be constructed from this 'string'.

        Examples
        --------
        For extension dtypes with arguments the following may be an
        adequate implementation.

        >>> import re
        >>> @classmethod
        ... def construct_from_string(cls, string):
        ...     pattern = re.compile(r"^my_type\[(?P<arg_name>.+)\]$")
        ...     match = pattern.match(string)
        ...     if match:
        ...         return cls(**match.groupdict())
        ...     else:
        ...         raise TypeError(
        ...             f"Cannot construct a '{cls.__name__}' from '{string}'"
        ...         )
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        # error: Non-overlapping equality check (left operand type: "str", right
        #  operand type: "Callable[[ExtensionDtype], str]")  [comparison-overlap]
        assert isinstance(cls.name, str), (cls, type(cls.name))
        if string != cls.name:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")
        return cls()

    @classmethod
    def is_dtype(cls, dtype: object) -> bool:
        """
        Check if we match 'dtype'.

        Parameters
        ----------
        dtype : object
            The object to check.

        Returns
        -------
        bool

        Notes
        -----
        The default implementation is True if

        1. ``cls.construct_from_string(dtype)`` is an instance
           of ``cls``.
        2. ``dtype`` is an object and is an instance of ``cls``
        3. ``dtype`` has a ``dtype`` attribute, and any of the above
           conditions is true for ``dtype.dtype``.
        """
        dtype = getattr(dtype, "dtype", dtype)

        if isinstance(dtype, (ABCSeries, ABCIndex, ABCDataFrame, np.dtype)):
            # https://github.com/pandas-dev/pandas/issues/22960
            # avoid passing data to `construct_from_string`. This could
            # cause a FutureWarning from numpy about failing elementwise
            # comparison from, e.g., comparing DataFrame == 'category'.
            return False
        elif dtype is None:
            return False
        elif isinstance(dtype, cls):
            return True
        if isinstance(dtype, str):
            try:
                return cls.construct_from_string(dtype) is not None
            except TypeError:
                return False
        return False

    @property
    def _is_numeric(self) -> bool:
        """
        Whether columns with this dtype should be considered numeric.

        By default ExtensionDtypes are assumed to be non-numeric.
        They'll be excluded from operations that exclude non-numeric
        columns, like (groupby) reductions, plotting, etc.
        """
        return False

    @property
    def _is_boolean(self) -> bool:
        """
        Whether this dtype should be considered boolean.

        By default, ExtensionDtypes are assumed to be non-numeric.
        Setting this to True will affect the behavior of several places,
        e.g.

        * is_bool
        * boolean indexing

        Returns
        -------
        bool
        """
        return False

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        """
        Return the common dtype, if one exists.

        Used in `find_common_type` implementation. This is for example used
        to determine the resulting dtype in a concat operation.

        If no common dtype exists, return None (which gives the other dtypes
        the chance to determine a common dtype). If all dtypes in the list
        return None, then the common dtype will be "object" dtype (this means
        it is never needed to return "object" dtype from this method itself).

        Parameters
        ----------
        dtypes : list of dtypes
            The dtypes for which to determine a common dtype. This is a list
            of np.dtype or ExtensionDtype instances.

        Returns
        -------
        Common dtype (np.dtype or ExtensionDtype) or None
        """
        if len(set(dtypes)) == 1:
            # only itself
            return self
        else:
            return None

    @property
    def _can_hold_na(self) -> bool:
        """
        Can arrays of this dtype hold NA values?
        """
        return True

    @property
    def _is_immutable(self) -> bool:
        """
        Can arrays with this dtype be modified with __setitem__? If not, return
        True.

        Immutable arrays are expected to raise TypeError on __setitem__ calls.
        """
        return False

    @cache_readonly
    def index_class(self) -> type_t[Index]:
        """
        The Index subclass to return from Index.__new__ when this dtype is
        encountered.
        """
        from pandas import Index

        return Index

    @property
    def _supports_2d(self) -> bool:
        """
        Do ExtensionArrays with this dtype support 2D arrays?

        Historically ExtensionArrays were limited to 1D. By returning True here,
        authors can indicate that their arrays support 2D instances. This can
        improve performance in some cases, particularly operations with `axis=1`.

        Arrays that support 2D values should:

            - implement Array.reshape
            - subclass the Dim2CompatTests in tests.extension.base
            - _concat_same_type should support `axis` keyword
            - _reduce and reductions should support `axis` keyword
        """
        return False

    @property
    def _can_fast_transpose(self) -> bool:
        """
        Is transposing an array with this dtype zero-copy?

        Only relevant for cases where _supports_2d is True.
        """
        return False


class StorageExtensionDtype(ExtensionDtype):
    """ExtensionDtype that may be backed by more than one implementation."""

    name: str
    _metadata = ("storage",)

    def __init__(self, storage: str | None = None) -> None:
        self.storage = storage

    def __repr__(self) -> str:
        return f"{self.name}[{self.storage}]"

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str) and other == self.name:
            return True
        return super().__eq__(other)

    def __hash__(self) -> int:
        # custom __eq__ so have to override __hash__
        return super().__hash__()

    @property
    def na_value(self) -> libmissing.NAType:
        return libmissing.NA


def register_extension_dtype(cls: type_t[ExtensionDtypeT]) -> type_t[ExtensionDtypeT]:
    """
    Register an ExtensionType with pandas as class decorator.

    This enables operations like ``.astype(name)`` for the name
    of the ExtensionDtype.

    Returns
    -------
    callable
        A class decorator.

    Examples
    --------
    >>> from pandas.api.extensions import register_extension_dtype, ExtensionDtype
    >>> @register_extension_dtype
    ... class MyExtensionDtype(ExtensionDtype):
    ...     name = "myextension"
    """
    _registry.register(cls)
    return cls


class Registry:
    """
    Registry for dtype inference.

    The registry allows one to map a string repr of a extension
    dtype to an extension dtype. The string alias can be used in several
    places, including

    * Series and Index constructors
    * :meth:`pandas.array`
    * :meth:`pandas.Series.astype`

    Multiple extension types can be registered.
    These are tried in order.
    """

    def __init__(self) -> None:
        self.dtypes: list[type_t[ExtensionDtype]] = []

    def register(self, dtype: type_t[ExtensionDtype]) -> None:
        """
        Parameters
        ----------
        dtype : ExtensionDtype class
        """
        if not issubclass(dtype, ExtensionDtype):
            raise ValueError("can only register pandas extension dtypes")

        self.dtypes.append(dtype)

    @overload
    def find(self, dtype: type_t[ExtensionDtypeT]) -> type_t[ExtensionDtypeT]:
        ...

    @overload
    def find(self, dtype: ExtensionDtypeT) -> ExtensionDtypeT:
        ...

    @overload
    def find(self, dtype: str) -> ExtensionDtype | None:
        ...

    @overload
    def find(
        self, dtype: npt.DTypeLike
    ) -> type_t[ExtensionDtype] | ExtensionDtype | None:
        ...

    def find(
        self, dtype: type_t[ExtensionDtype] | ExtensionDtype | npt.DTypeLike
    ) -> type_t[ExtensionDtype] | ExtensionDtype | None:
        """
        Parameters
        ----------
        dtype : ExtensionDtype class or instance or str or numpy dtype or python type

        Returns
        -------
        return the first matching dtype, otherwise return None
        """
        if not isinstance(dtype, str):
            dtype_type: type_t
            if not isinstance(dtype, type):
                dtype_type = type(dtype)
            else:
                dtype_type = dtype
            if issubclass(dtype_type, ExtensionDtype):
                # cast needed here as mypy doesn't know we have figured
                # out it is an ExtensionDtype or type_t[ExtensionDtype]
                return cast("ExtensionDtype | type_t[ExtensionDtype]", dtype)

            return None

        for dtype_type in self.dtypes:
            try:
                return dtype_type.construct_from_string(dtype)
            except TypeError:
                pass

        return None


_registry = Registry()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\smtlib.py ===
import typing

import sympy
from sympy.core import Add, Mul
from sympy.core import Symbol, Expr, Float, Rational, Integer, Basic
from sympy.core.function import UndefinedFunction, Function
from sympy.core.relational import Relational, Unequality, Equality, LessThan, GreaterThan, StrictLessThan, StrictGreaterThan
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp, log, Pow
from sympy.functions.elementary.hyperbolic import sinh, cosh, tanh
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import sin, cos, tan, asin, acos, atan, atan2
from sympy.logic.boolalg import And, Or, Xor, Implies, Boolean
from sympy.logic.boolalg import BooleanTrue, BooleanFalse, BooleanFunction, Not, ITE
from sympy.printing.printer import Printer
from sympy.sets import Interval
from mpmath.libmp.libmpf import prec_to_dps, to_str as mlib_to_str
from sympy.assumptions.assume import AppliedPredicate
from sympy.assumptions.relation.binrel import AppliedBinaryRelation
from sympy.assumptions.ask import Q
from sympy.assumptions.relation.equality import StrictGreaterThanPredicate, StrictLessThanPredicate, GreaterThanPredicate, LessThanPredicate, EqualityPredicate


class SMTLibPrinter(Printer):
    printmethod = "_smtlib"

    # based on dReal, an automated reasoning tool for solving problems that can be encoded as first-order logic formulas over the real numbers.
    # dReal's special strength is in handling problems that involve a wide range of nonlinear real functions.
    _default_settings: dict = {
        'precision': None,
        'known_types': {
            bool: 'Bool',
            int: 'Int',
            float: 'Real'
        },
        'known_constants': {
            # pi: 'MY_VARIABLE_PI_DECLARED_ELSEWHERE',
        },
        'known_functions': {
            Add: '+',
            Mul: '*',

            Equality: '=',
            LessThan: '<=',
            GreaterThan: '>=',
            StrictLessThan: '<',
            StrictGreaterThan: '>',

            EqualityPredicate(): '=',
            LessThanPredicate(): '<=',
            GreaterThanPredicate(): '>=',
            StrictLessThanPredicate(): '<',
            StrictGreaterThanPredicate(): '>',

            exp: 'exp',
            log: 'log',
            Abs: 'abs',
            sin: 'sin',
            cos: 'cos',
            tan: 'tan',
            asin: 'arcsin',
            acos: 'arccos',
            atan: 'arctan',
            atan2: 'arctan2',
            sinh: 'sinh',
            cosh: 'cosh',
            tanh: 'tanh',
            Min: 'min',
            Max: 'max',
            Pow: 'pow',

            And: 'and',
            Or: 'or',
            Xor: 'xor',
            Not: 'not',
            ITE: 'ite',
            Implies: '=>',
        }
    }

    symbol_table: dict

    def __init__(self, settings: typing.Optional[dict] = None,
                 symbol_table=None):
        settings = settings or {}
        self.symbol_table = symbol_table or {}
        Printer.__init__(self, settings)
        self._precision = self._settings['precision']
        self._known_types = dict(self._settings['known_types'])
        self._known_constants = dict(self._settings['known_constants'])
        self._known_functions = dict(self._settings['known_functions'])

        for _ in self._known_types.values(): assert self._is_legal_name(_)
        for _ in self._known_constants.values(): assert self._is_legal_name(_)
        # for _ in self._known_functions.values(): assert self._is_legal_name(_)  # +, *, <, >, etc.

    def _is_legal_name(self, s: str):
        if not s: return False
        if s[0].isnumeric(): return False
        return all(_.isalnum() or _ == '_' for _ in s)

    def _s_expr(self, op: str, args: typing.Union[list, tuple]) -> str:
        args_str = ' '.join(
            a if isinstance(a, str)
            else self._print(a)
            for a in args
        )
        return f'({op} {args_str})'

    def _print_Function(self, e):
        if e in self._known_functions:
            op = self._known_functions[e]
        elif type(e) in self._known_functions:
            op = self._known_functions[type(e)]
        elif type(type(e)) == UndefinedFunction:
            op = e.name
        elif isinstance(e, AppliedBinaryRelation) and e.function in self._known_functions:
            op = self._known_functions[e.function]
            return self._s_expr(op, e.arguments)
        else:
            op = self._known_functions[e]  # throw KeyError

        return self._s_expr(op, e.args)

    def _print_Relational(self, e: Relational):
        return self._print_Function(e)

    def _print_BooleanFunction(self, e: BooleanFunction):
        return self._print_Function(e)

    def _print_Expr(self, e: Expr):
        return self._print_Function(e)

    def _print_Unequality(self, e: Unequality):
        if type(e) in self._known_functions:
            return self._print_Relational(e)  # default
        else:
            eq_op = self._known_functions[Equality]
            not_op = self._known_functions[Not]
            return self._s_expr(not_op, [self._s_expr(eq_op, e.args)])

    def _print_Piecewise(self, e: Piecewise):
        def _print_Piecewise_recursive(args: typing.Union[list, tuple]):
            e, c = args[0]
            if len(args) == 1:
                assert (c is True) or isinstance(c, BooleanTrue)
                return self._print(e)
            else:
                ite = self._known_functions[ITE]
                return self._s_expr(ite, [
                    c, e, _print_Piecewise_recursive(args[1:])
                ])

        return _print_Piecewise_recursive(e.args)

    def _print_Interval(self, e: Interval):
        if e.start.is_infinite and e.end.is_infinite:
            return ''
        elif e.start.is_infinite != e.end.is_infinite:
            raise ValueError(f'One-sided intervals (`{e}`) are not supported in SMT.')
        else:
            return f'[{e.start}, {e.end}]'

    def _print_AppliedPredicate(self, e: AppliedPredicate):
        if e.function == Q.positive:
            rel = Q.gt(e.arguments[0],0)
        elif e.function == Q.negative:
            rel = Q.lt(e.arguments[0], 0)
        elif e.function == Q.zero:
            rel = Q.eq(e.arguments[0], 0)
        elif e.function == Q.nonpositive:
            rel = Q.le(e.arguments[0], 0)
        elif e.function == Q.nonnegative:
            rel = Q.ge(e.arguments[0], 0)
        elif e.function == Q.nonzero:
            rel = Q.ne(e.arguments[0], 0)
        else:
            raise ValueError(f"Predicate (`{e}`) is not handled.")

        return self._print_AppliedBinaryRelation(rel)

    def _print_AppliedBinaryRelation(self, e: AppliedPredicate):
        if e.function == Q.ne:
            return self._print_Unequality(Unequality(*e.arguments))
        else:
            return self._print_Function(e)

    # todo: Sympy does not support quantifiers yet as of 2022, but quantifiers can be handy in SMT.
    # For now, users can extend this class and build in their own quantifier support.
    # See `test_quantifier_extensions()` in test_smtlib.py for an example of how this might look.

    # def _print_ForAll(self, e: ForAll):
    #     return self._s('forall', [
    #         self._s('', [
    #             self._s(sym.name, [self._type_name(sym), Interval(start, end)])
    #             for sym, start, end in e.limits
    #         ]),
    #         e.function
    #     ])

    def _print_BooleanTrue(self, x: BooleanTrue):
        return 'true'

    def _print_BooleanFalse(self, x: BooleanFalse):
        return 'false'

    def _print_Float(self, x: Float):
        dps = prec_to_dps(x._prec)
        str_real = mlib_to_str(x._mpf_, dps, strip_zeros=True, min_fixed=None, max_fixed=None)

        if 'e' in str_real:
            (mant, exp) = str_real.split('e')

            if exp[0] == '+':
                exp = exp[1:]

            mul = self._known_functions[Mul]
            pow = self._known_functions[Pow]

            return r"(%s %s (%s 10 %s))" % (mul, mant, pow, exp)
        elif str_real in ["+inf", "-inf"]:
            raise ValueError("Infinite values are not supported in SMT.")
        else:
            return str_real

    def _print_float(self, x: float):
        return self._print(Float(x))

    def _print_Rational(self, x: Rational):
        return self._s_expr('/', [x.p, x.q])

    def _print_Integer(self, x: Integer):
        assert x.q == 1
        return str(x.p)

    def _print_int(self, x: int):
        return str(x)

    def _print_Symbol(self, x: Symbol):
        assert self._is_legal_name(x.name)
        return x.name

    def _print_NumberSymbol(self, x):
        name = self._known_constants.get(x)
        if name:
            return name
        else:
            f = x.evalf(self._precision) if self._precision else x.evalf()
            return self._print_Float(f)

    def _print_UndefinedFunction(self, x):
        assert self._is_legal_name(x.name)
        return x.name

    def _print_Exp1(self, x):
        return (
            self._print_Function(exp(1, evaluate=False))
            if exp in self._known_functions else
            self._print_NumberSymbol(x)
        )

    def emptyPrinter(self, expr):
        raise NotImplementedError(f'Cannot convert `{repr(expr)}` of type `{type(expr)}` to SMT.')


def smtlib_code(
    expr,
    auto_assert=True, auto_declare=True,
    precision=None,
    symbol_table=None,
    known_types=None, known_constants=None, known_functions=None,
    prefix_expressions=None, suffix_expressions=None,
    log_warn=None
):
    r"""Converts ``expr`` to a string of smtlib code.

    Parameters
    ==========

    expr : Expr | List[Expr]
        A SymPy expression or system to be converted.
    auto_assert : bool, optional
        If false, do not modify expr and produce only the S-Expression equivalent of expr.
        If true, assume expr is a system and assert each boolean element.
    auto_declare : bool, optional
        If false, do not produce declarations for the symbols used in expr.
        If true, prepend all necessary declarations for variables used in expr based on symbol_table.
    precision : integer, optional
        The ``evalf(..)`` precision for numbers such as pi.
    symbol_table : dict, optional
        A dictionary where keys are ``Symbol`` or ``Function`` instances and values are their Python type i.e. ``bool``, ``int``, ``float``, or ``Callable[...]``.
        If incomplete, an attempt will be made to infer types from ``expr``.
    known_types: dict, optional
        A dictionary where keys are ``bool``, ``int``, ``float`` etc. and values are their corresponding SMT type names.
        If not given, a partial listing compatible with several solvers will be used.
    known_functions : dict, optional
        A dictionary where keys are ``Function``, ``Relational``, ``BooleanFunction``, or ``Expr`` instances and values are their SMT string representations.
        If not given, a partial listing optimized for dReal solver (but compatible with others) will be used.
    known_constants: dict, optional
        A dictionary where keys are ``NumberSymbol`` instances and values are their SMT variable names.
        When using this feature, extra caution must be taken to avoid naming collisions between user symbols and listed constants.
        If not given, constants will be expanded inline i.e. ``3.14159`` instead of ``MY_SMT_VARIABLE_FOR_PI``.
    prefix_expressions: list, optional
        A list of lists of ``str`` and/or expressions to convert into SMTLib and prefix to the output.
    suffix_expressions: list, optional
        A list of lists of ``str`` and/or expressions to convert into SMTLib and postfix to the output.
    log_warn: lambda function, optional
        A function to record all warnings during potentially risky operations.
        Soundness is a core value in SMT solving, so it is good to log all assumptions made.

    Examples
    ========
    >>> from sympy import smtlib_code, symbols, sin, Eq
    >>> x = symbols('x')
    >>> smtlib_code(sin(x).series(x).removeO(), log_warn=print)
    Could not infer type of `x`. Defaulting to float.
    Non-Boolean expression `x**5/120 - x**3/6 + x` will not be asserted. Converting to SMTLib verbatim.
    '(declare-const x Real)\n(+ x (* (/ -1 6) (pow x 3)) (* (/ 1 120) (pow x 5)))'

    >>> from sympy import Rational
    >>> x, y, tau = symbols("x, y, tau")
    >>> smtlib_code((2*tau)**Rational(7, 2), log_warn=print)
    Could not infer type of `tau`. Defaulting to float.
    Non-Boolean expression `8*sqrt(2)*tau**(7/2)` will not be asserted. Converting to SMTLib verbatim.
    '(declare-const tau Real)\n(* 8 (pow 2 (/ 1 2)) (pow tau (/ 7 2)))'

    ``Piecewise`` expressions are implemented with ``ite`` expressions by default.
    Note that if the ``Piecewise`` lacks a default term, represented by
    ``(expr, True)`` then an error will be thrown.  This is to prevent
    generating an expression that may not evaluate to anything.

    >>> from sympy import Piecewise
    >>> pw = Piecewise((x + 1, x > 0), (x, True))
    >>> smtlib_code(Eq(pw, 3), symbol_table={x: float}, log_warn=print)
    '(declare-const x Real)\n(assert (= (ite (> x 0) (+ 1 x) x) 3))'

    Custom printing can be defined for certain types by passing a dictionary of
    PythonType : "SMT Name" to the ``known_types``, ``known_constants``, and ``known_functions`` kwargs.

    >>> from typing import Callable
    >>> from sympy import Function, Add
    >>> f = Function('f')
    >>> g = Function('g')
    >>> smt_builtin_funcs = {  # functions our SMT solver will understand
    ...   f: "existing_smtlib_fcn",
    ...   Add: "sum",
    ... }
    >>> user_def_funcs = {  # functions defined by the user must have their types specified explicitly
    ...   g: Callable[[int], float],
    ... }
    >>> smtlib_code(f(x) + g(x), symbol_table=user_def_funcs, known_functions=smt_builtin_funcs, log_warn=print)
    Non-Boolean expression `f(x) + g(x)` will not be asserted. Converting to SMTLib verbatim.
    '(declare-const x Int)\n(declare-fun g (Int) Real)\n(sum (existing_smtlib_fcn x) (g x))'
    """
    log_warn = log_warn or (lambda _: None)

    if not isinstance(expr, list): expr = [expr]
    expr = [
        sympy.sympify(_, strict=True, evaluate=False, convert_xor=False)
        for _ in expr
    ]

    if not symbol_table: symbol_table = {}
    symbol_table = _auto_infer_smtlib_types(
        *expr, symbol_table=symbol_table
    )
    # See [FALLBACK RULES]
    # Need SMTLibPrinter to populate known_functions and known_constants first.

    settings = {}
    if precision: settings['precision'] = precision
    del precision

    if known_types: settings['known_types'] = known_types
    del known_types

    if known_functions: settings['known_functions'] = known_functions
    del known_functions

    if known_constants: settings['known_constants'] = known_constants
    del known_constants

    if not prefix_expressions: prefix_expressions = []
    if not suffix_expressions: suffix_expressions = []

    p = SMTLibPrinter(settings, symbol_table)
    del symbol_table

    # [FALLBACK RULES]
    for e in expr:
        for sym in e.atoms(Symbol, Function):
            if (
                sym.is_Symbol and
                sym not in p._known_constants and
                sym not in p.symbol_table
            ):
                log_warn(f"Could not infer type of `{sym}`. Defaulting to float.")
                p.symbol_table[sym] = float
            if (
                sym.is_Function and
                type(sym) not in p._known_functions and
                type(sym) not in p.symbol_table and
                not sym.is_Piecewise
            ): raise TypeError(
                f"Unknown type of undefined function `{sym}`. "
                f"Must be mapped to ``str`` in known_functions or mapped to ``Callable[..]`` in symbol_table."
            )

    declarations = []
    if auto_declare:
        constants = {sym.name: sym for e in expr for sym in e.free_symbols
                     if sym not in p._known_constants}
        functions = {fnc.name: fnc for e in expr for fnc in e.atoms(Function)
                     if type(fnc) not in p._known_functions and not fnc.is_Piecewise}
        declarations = \
            [
                _auto_declare_smtlib(sym, p, log_warn)
                for sym in constants.values()
            ] + [
                _auto_declare_smtlib(fnc, p, log_warn)
                for fnc in functions.values()
            ]
        declarations = [decl for decl in declarations if decl]

    if auto_assert:
        expr = [_auto_assert_smtlib(e, p, log_warn) for e in expr]

    # return SMTLibPrinter().doprint(expr)
    return '\n'.join([
        # ';; PREFIX EXPRESSIONS',
        *[
            e if isinstance(e, str) else p.doprint(e)
            for e in prefix_expressions
        ],

        # ';; DECLARATIONS',
        *sorted(e for e in declarations),

        # ';; EXPRESSIONS',
        *[
            e if isinstance(e, str) else p.doprint(e)
            for e in expr
        ],

        # ';; SUFFIX EXPRESSIONS',
        *[
            e if isinstance(e, str) else p.doprint(e)
            for e in suffix_expressions
        ],
    ])


def _auto_declare_smtlib(sym: typing.Union[Symbol, Function], p: SMTLibPrinter, log_warn: typing.Callable[[str], None]):
    if sym.is_Symbol:
        type_signature = p.symbol_table[sym]
        assert isinstance(type_signature, type)
        type_signature = p._known_types[type_signature]
        return p._s_expr('declare-const', [sym, type_signature])

    elif sym.is_Function:
        type_signature = p.symbol_table[type(sym)]
        assert callable(type_signature)
        type_signature = [p._known_types[_] for _ in type_signature.__args__]
        assert len(type_signature) > 0
        params_signature = f"({' '.join(type_signature[:-1])})"
        return_signature = type_signature[-1]
        return p._s_expr('declare-fun', [type(sym), params_signature, return_signature])

    else:
        log_warn(f"Non-Symbol/Function `{sym}` will not be declared.")
        return None


def _auto_assert_smtlib(e: Expr, p: SMTLibPrinter, log_warn: typing.Callable[[str], None]):
    if isinstance(e, Boolean) or (
        e in p.symbol_table and p.symbol_table[e] == bool
    ) or (
        e.is_Function and
        type(e) in p.symbol_table and
        p.symbol_table[type(e)].__args__[-1] == bool
    ):
        return p._s_expr('assert', [e])
    else:
        log_warn(f"Non-Boolean expression `{e}` will not be asserted. Converting to SMTLib verbatim.")
        return e


def _auto_infer_smtlib_types(
    *exprs: Basic,
    symbol_table: typing.Optional[dict] = None
) -> dict:
    # [TYPE INFERENCE RULES]
    # X is alone in an expr => X is bool
    # X in BooleanFunction.args => X is bool
    # X matches to a bool param of a symbol_table function => X is bool
    # X matches to an int param of a symbol_table function => X is int
    # X.is_integer => X is int
    # X == Y, where X is T => Y is T

    # [FALLBACK RULES]
    # see _auto_declare_smtlib(..)
    # X is not bool and X is not int and X is Symbol => X is float
    # else (e.g. X is Function) => error. must be specified explicitly.

    _symbols = dict(symbol_table) if symbol_table else {}

    def safe_update(syms: set, inf):
        for s in syms:
            assert s.is_Symbol
            if (old_type := _symbols.setdefault(s, inf)) != inf:
                raise TypeError(f"Could not infer type of `{s}`. Apparently both `{old_type}` and `{inf}`?")

    # EXPLICIT TYPES
    safe_update({
        e
        for e in exprs
        if e.is_Symbol
    }, bool)

    safe_update({
        symbol
        for e in exprs
        for boolfunc in e.atoms(BooleanFunction)
        for symbol in boolfunc.args
        if symbol.is_Symbol
    }, bool)

    safe_update({
        symbol
        for e in exprs
        for boolfunc in e.atoms(Function)
        if type(boolfunc) in _symbols
        for symbol, param in zip(boolfunc.args, _symbols[type(boolfunc)].__args__)
        if symbol.is_Symbol and param == bool
    }, bool)

    safe_update({
        symbol
        for e in exprs
        for intfunc in e.atoms(Function)
        if type(intfunc) in _symbols
        for symbol, param in zip(intfunc.args, _symbols[type(intfunc)].__args__)
        if symbol.is_Symbol and param == int
    }, int)

    safe_update({
        symbol
        for e in exprs
        for symbol in e.atoms(Symbol)
        if symbol.is_integer
    }, int)

    safe_update({
        symbol
        for e in exprs
        for symbol in e.atoms(Symbol)
        if symbol.is_real and not symbol.is_integer
    }, float)

    # EQUALITY RELATION RULE
    rels_eq = [rel for expr in exprs for rel in expr.atoms(Equality)]
    rels = [
               (rel.lhs, rel.rhs) for rel in rels_eq if rel.lhs.is_Symbol
           ] + [
               (rel.rhs, rel.lhs) for rel in rels_eq if rel.rhs.is_Symbol
           ]
    for infer, reltd in rels:
        inference = (
            _symbols[infer] if infer in _symbols else
            _symbols[reltd] if reltd in _symbols else

            _symbols[type(reltd)].__args__[-1]
            if reltd.is_Function and type(reltd) in _symbols else

            bool if reltd.is_Boolean else
            int if reltd.is_integer or reltd.is_Integer else
            float if reltd.is_real else
            None
        )
        if inference: safe_update({infer}, inference)

    return _symbols

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\phi2\convert_to_onnx.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import onnx
import torch
from benchmark_helper import Precision
from fusion_options import AttentionOpType
from onnx_model import OnnxModel
from packaging import version
from transformers import AutoConfig, AutoModelForCausalLM

from onnxruntime import __version__ as ort_version

if version.parse(ort_version) < version.parse("1.22.0"):
    from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer as MatMulNBitsQuantizer
else:
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer


class ConvertPhi2ToONNX:
    def __init__(
        self,
        device: torch.device,
        model_class: str = "microsoft/phi-2",
        cache_dir: str = "./cache",
    ):
        self.model_class = model_class
        self.device = device
        self.cache_dir = cache_dir
        self.phi_config = AutoConfig.from_pretrained(self.model_class, trust_remote_code=True, cache_dir=self.cache_dir)
        self.phi_model = None
        self.batch_size = 2
        self.sequence_length = 8
        self.attn_op_type = None
        self.precision = None
        self.block_size = 16
        self.accuracy_level = None

    def set_quantization_params(self, block_size: int, accuracy_level: int | None):
        self.block_size = block_size
        self.accuracy_level = accuracy_level

    def init_attn_type_and_precision(self, attn_op_type: AttentionOpType, precision: Precision):
        self.attn_op_type = attn_op_type
        self.precision = precision

    def erase_onnx_model(self, onnx_path: str) -> None:
        assert onnx_path.endswith(".onnx")
        if not os.path.exists(onnx_path):
            return

        model = onnx.load_model(onnx_path, load_external_data=False)
        onnx_data_path = None
        for initializer in model.graph.initializer:
            if initializer.data_location == 1 and initializer.external_data[0].key == "location":
                onnx_data_path = "./" + initializer.external_data[0].value
                break
        logging.info(f"Erasing {onnx_path}...")
        os.remove(onnx_path)
        if onnx_data_path is not None:
            onnx_data_path = os.path.join(Path(onnx_path).parent, onnx_data_path)
            logging.info(f"Erasing {onnx_data_path}...")
            os.remove(onnx_data_path)

    def get_phi2_torch_model(self):
        logging.info("Loading phi2 torch model...")
        if self.phi_model is not None:
            return
        self.phi_model = AutoModelForCausalLM.from_pretrained(
            self.model_class, trust_remote_code=True, cache_dir=self.cache_dir
        )
        self.phi_model.eval()
        self.phi_model.to(self.device)

    def get_phi2_torch_inputs(self, batch_size: int, sequence_length: int):
        input_ids = torch.randint(
            low=0,
            high=self.phi_config.vocab_size,
            size=(batch_size, sequence_length),
            dtype=torch.int64,
            device=self.device,
        )
        self.get_phi2_torch_model()
        torch_inputs = self.phi_model.prepare_inputs_for_generation(
            input_ids, past_key_values=self.phi_model(input_ids, use_cache=True)["past_key_values"]
        )
        return torch_inputs["input_ids"], torch_inputs["attention_mask"], torch_inputs["past_key_values"]

    def dynamo_export(self, onnx_path: str):
        input_ids, attention_mask, past_key_values = self.get_phi2_torch_inputs(self.batch_size, self.sequence_length)
        self.phi_model(input_ids, attention_mask=attention_mask, past_key_values=past_key_values)

        from torch._dynamo import config

        config.capture_scalar_outputs = True

        logging.info("Exporting Phi2 torch model to ONNX...")
        torch.onnx.dynamo_export(
            self.phi_model,
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            export_options=torch.onnx.ExportOptions(dynamic_shapes=True),
        ).save(onnx_path)
        onnx.checker.check_model(onnx_path)
        onnx.shape_inference.infer_shapes_path(onnx_path)

    def optimize_phi2_onnx(self, onnx_path: str, onnx_path_opt: str):
        from fusion_options import FusionOptions
        from optimizer import optimize_model

        optimization_options = FusionOptions("phi")
        optimization_options.set_attention_op_type(self.attn_op_type)
        optimizer = optimize_model(
            onnx_path,
            model_type="phi",
            num_heads=self.phi_config.num_attention_heads,
            hidden_size=self.phi_config.hidden_size,
            opt_level=0,
            optimization_options=optimization_options,
            only_onnxruntime=False,
        )

        fused_op_count = optimizer.get_fused_operator_statistics()
        if optimizer.is_fully_optimized(fused_op_count):
            logging.info("Model is fully optimized.")
        else:
            logging.info("Model is not fully optimized.")

        if self.precision == Precision.FLOAT32:
            optimizer.save_model_to_file(onnx_path_opt, use_external_data_format=True)
            return

        if (
            self.precision == Precision.FLOAT16 or self.precision == Precision.INT4
        ) and self.attn_op_type != AttentionOpType.MultiHeadAttention:
            # We keep last three layers of Attention as float32 or bfloat16 to avoid overflow.
            node_block_list = (
                [
                    "Attention_29",
                    "Attention_30",
                    "Attention_31",
                ]
                if self.attn_op_type != AttentionOpType.PagedAttention
                else []
            )  # TODO: temp setting for paged attention
            logging.info("Converting onnx model to float16/bfloat16...")
            optimizer.convert_float_to_float16(
                keep_io_types=False,
                node_block_list=node_block_list,
                use_symbolic_shape_infer=True,
                use_bfloat16_as_blocked_nodes_dtype=self.attn_op_type == AttentionOpType.GroupQueryAttention,
            )
            logging.info("Converting onnx model to float16/bfloat16 done.")

        if self.precision == Precision.FLOAT16:
            optimizer.save_model_to_file(onnx_path_opt, use_external_data_format=True)
            return
        else:
            assert self.precision == Precision.INT4
            quant = MatMulNBitsQuantizer(
                model=optimizer.model,
                block_size=self.block_size,
                is_symmetric=True,
                accuracy_level=self.accuracy_level,
            )
            quant.process()
            quant.model.save_model_to_file(onnx_path_opt, use_external_data_format=True)

    # This function currently only works for phi2 model
    def convert_to_use_cuda_graph(self, in_onnx_path: str, out_onnx_path: str):
        onnx_model = OnnxModel(onnx.load(in_onnx_path, load_external_data=True))

        from onnx import TensorProto, helper

        graph = onnx_model.graph()
        new_inputs = []
        for vi in graph.input:
            if "attention_mask" in vi.name:
                vi_seqlen_k = helper.make_tensor_value_info(
                    "seqlens_k",
                    elem_type=TensorProto.INT32,
                    shape=["batch_size"],
                )
                vi_total_seq_len = helper.make_tensor_value_info(
                    "total_sequence_length",
                    elem_type=TensorProto.INT32,
                    shape=[1],
                )
                new_inputs.extend([vi_seqlen_k, vi_total_seq_len])
            else:
                new_inputs.append(vi)

        graph.ClearField("input")
        graph.input.extend(new_inputs)

        gqas = onnx_model.get_nodes_by_op_type("GroupQueryAttention")
        gqa = gqas[0]
        seqlens_path = onnx_model.match_parent_path(
            gqa,
            ["Cast", "Sub", "ReduceSum", "Cast"],
            [5, 0, 0, 0],
        )
        if seqlens_path is None:
            raise RuntimeError("Failed to find seqlens path for GroupQueryAttention node.")
        total_seq_len_path = onnx_model.match_parent_path(
            gqa,
            ["Cast", "Gather", "Shape"],
            [6, 0, 0],
        )
        if total_seq_len_path is None:
            raise RuntimeError("Failed to find total_seq_len path for GroupQueryAttention node.")
        onnx_model.remove_nodes(seqlens_path)
        onnx_model.remove_nodes(total_seq_len_path)

        for gqa in gqas:
            gqa.input[5] = "seqlens_k"
            gqa.input[6] = "total_sequence_length"

        onnx_model.save(onnx_model.model, out_onnx_path, save_as_external_data=True)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fp32_cpu",
        required=False,
        action="store_true",
        help="Generate fp32 ONNX model for CPU",
    )

    parser.add_argument(
        "--int4_cpu",
        required=False,
        action="store_true",
        help="Generate int4 ONNX model for CPU",
    )

    parser.add_argument(
        "--fp32_gpu",
        required=False,
        action="store_true",
        help="Generate fp32 ONNX model for Nvidia GPUs",
    )

    parser.add_argument(
        "--fp16_gpu",
        required=False,
        action="store_true",
        help="Generate fp16 ONNX model for Nvidia GPUs",
    )

    parser.add_argument(
        "--int4_gpu",
        required=False,
        action="store_true",
        help="Generate int4 ONNX model for Nvidia GPUs",
    )

    parser.add_argument(
        "--fp16_gpu_sm8x",
        required=False,
        action="store_true",
        help="Generate fp16 ONNX model for Nvidia GPUs with CUDA architecture SM=80~89",
    )

    parser.add_argument(
        "--int4_gpu_sm8x",
        required=False,
        action="store_true",
        help="Generate int4 ONNX model for Nvidia GPUs with CUDA architecture SM=80~89",
    )

    parser.add_argument(
        "--fp16_vllm",
        required=False,
        action="store_true",
        help="Generate fp16 ONNX model for ORT VLLM",
    )

    parser.add_argument(
        "--int4_vllm",
        required=False,
        action="store_true",
        help="Generate int4 ONNX model for ORT VLLM",
    )

    parser.add_argument(
        "--use_cuda_graph",
        required=False,
        action="store_true",
        help="Use CUDA Graph in decoding process",
    )

    parser.add_argument(
        "--overwrite",
        required=False,
        action="store_true",
        help="Overwrite existing ONNX models",
    )

    parser.add_argument(
        "--cache_dir",
        required=False,
        type=str,
        default="./cache",
        help="The cache directory for the pytorch model",
    )

    parser.add_argument(
        "--device_id",
        required=False,
        type=int,
        default=0,
        help="The device id for the pytorch model",
    )

    parser.add_argument(
        "--run_example",
        required=False,
        action="store_true",
        help="Run ORT inference example",
    )

    parser.add_argument(
        "--run_benchmark",
        required=False,
        action="store_true",
        help="Run ORT benchmark",
    )

    parser.add_argument(
        "--skip_export",
        required=False,
        action="store_true",
        help="Skip exporting ONNX model",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="The output directory for the ONNX models",
        default="phi2_onnx_models",
    )

    parser.add_argument(
        "--block_size",
        required=False,
        default=16,
        type=int,
        help="Block size to quantize with. See https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/matmul_nbits_quantizer.py for details.",
    )

    parser.add_argument(
        "--int4_accuracy_level",
        required=False,
        type=int,
        help="Accuracy level of the 4-bit quantized MatMul computation. "
        "Refer to the MatMulNBits contrib op's 'accuracy_level' attribute for details "
        "(https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftmatmulnbits).",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    device = torch.device("cuda", args.device_id) if torch.cuda.is_available() else torch.device("cpu")

    converter = ConvertPhi2ToONNX(device, cache_dir=args.cache_dir)
    converter.set_quantization_params(args.block_size, args.int4_accuracy_level)

    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    original_onnx_path = os.path.join(output_dir, "phi2_original.onnx")

    if not args.skip_export:
        if not os.path.exists(original_onnx_path) or args.overwrite:
            converter.dynamo_export(original_onnx_path)

    model_type_to_args = {
        "fp32_cpu": (
            AttentionOpType.MultiHeadAttention,
            Precision.FLOAT32,
            os.path.join(output_dir, "phi2_decoder_fp32_cpu.onnx"),
        ),
        "int4_cpu": (
            AttentionOpType.MultiHeadAttention,
            Precision.INT4,
            os.path.join(output_dir, "phi2_decoder_int4_cpu.onnx"),
        ),
        "fp32_gpu": (
            AttentionOpType.Attention,
            Precision.FLOAT32,
            os.path.join(output_dir, "phi2_decoder_fp32_gpu.onnx"),
        ),
        "fp16_gpu": (
            AttentionOpType.Attention,
            Precision.FLOAT16,
            os.path.join(output_dir, "phi2_decoder_fp16_gpu.onnx"),
        ),
        "int4_gpu": (AttentionOpType.Attention, Precision.INT4, os.path.join(output_dir, "phi2_decoder_int4_gpu.onnx")),
        "fp16_gpu_sm8x": (
            AttentionOpType.GroupQueryAttention,
            Precision.FLOAT16,
            os.path.join(output_dir, "phi2_decoder_fp16_gpu_sm8x.onnx"),
        ),
        "int4_gpu_sm8x": (
            AttentionOpType.GroupQueryAttention,
            Precision.INT4,
            os.path.join(output_dir, "phi2_decoder_int4_gpu_sm8x.onnx"),
        ),
        "fp16_vllm": (
            AttentionOpType.PagedAttention,
            Precision.FLOAT16,
            os.path.join(output_dir, "phi2_decoder_fp16_vllm.onnx"),
        ),
        "int4_vllm": (
            AttentionOpType.PagedAttention,
            Precision.INT4,
            os.path.join(output_dir, "phi2_decoder_int4_vllm.onnx"),
        ),
    }

    if not args.skip_export:
        from multiprocessing import Process

        def run_optimize_phi2_onnx(
            converter: ConvertPhi2ToONNX,
            original_onnx_path: str,
            attention_type: AttentionOpType,
            precision: Precision,
            optimized_onnx_path: str,
        ):
            converter.init_attn_type_and_precision(attention_type, precision)
            converter.optimize_phi2_onnx(original_onnx_path, optimized_onnx_path)
            if args.use_cuda_graph:
                assert args.fp16_gpu_sm8x or args.int4_gpu_sm8x
                converter.convert_to_use_cuda_graph(optimized_onnx_path, optimized_onnx_path)

        processes = []
        if args.fp32_cpu:
            processes.append(
                Process(
                    target=run_optimize_phi2_onnx, args=(converter, original_onnx_path, *model_type_to_args["fp32_cpu"])
                )
            )

        if args.int4_cpu:
            processes.append(
                Process(
                    target=run_optimize_phi2_onnx, args=(converter, original_onnx_path, *model_type_to_args["int4_cpu"])
                )
            )

        if args.fp32_gpu:
            processes.append(
                Process(
                    target=run_optimize_phi2_onnx, args=(converter, original_onnx_path, *model_type_to_args["fp32_gpu"])
                )
            )

        if args.fp16_gpu:
            processes.append(
                Process(
                    target=run_optimize_phi2_onnx, args=(converter, original_onnx_path, *model_type_to_args["fp16_gpu"])
                )
            )

        if args.int4_gpu:
            processes.append(
                Process(
                    target=run_optimize_phi2_onnx, args=(converter, original_onnx_path, *model_type_to_args["int4_gpu"])
                )
            )

        if args.fp16_gpu_sm8x:
            processes.append(
                Process(
                    target=run_optimize_phi2_onnx,
                    args=(converter, original_onnx_path, *model_type_to_args["fp16_gpu_sm8x"]),
                )
            )

        if args.int4_gpu_sm8x:
            processes.append(
                Process(
                    target=run_optimize_phi2_onnx,
                    args=(converter, original_onnx_path, *model_type_to_args["int4_gpu_sm8x"]),
                )
            )

        if args.fp16_vllm:
            processes.append(
                Process(
                    target=run_optimize_phi2_onnx,
                    args=(converter, original_onnx_path, *model_type_to_args["fp16_vllm"]),
                )
            )

        if args.int4_vllm:
            processes.append(
                Process(
                    target=run_optimize_phi2_onnx,
                    args=(converter, original_onnx_path, *model_type_to_args["int4_vllm"]),
                )
            )

        [p.start() for p in processes]
        [p.join() for p in processes]

    if args.run_example or args.run_benchmark:
        from inference_example import run_phi2

        if args.fp16_gpu_sm8x:
            logging.info("Running fp16_gpu_sm8x example...")
            run_phi2(
                onnx_model_path=model_type_to_args["fp16_gpu_sm8x"][2],
                use_buffer_share=True,
                device_id=args.device_id,
                use_step=True,
                use_cuda_graph=args.use_cuda_graph,
                run_benchmark=args.run_benchmark,
            )
        if args.int4_gpu_sm8x:
            logging.info("Running int4_gpu_sm8x example...")
            run_phi2(
                onnx_model_path=model_type_to_args["int4_gpu_sm8x"][2],
                use_buffer_share=True,
                device_id=args.device_id,
                use_step=True,
                use_cuda_graph=args.use_cuda_graph,
                run_benchmark=args.run_benchmark,
            )
        if args.fp32_gpu:
            logging.info("Running fp32_gpu example...")
            run_phi2(
                onnx_model_path=model_type_to_args["fp32_gpu"][2],
                use_buffer_share=False,
                device_id=args.device_id,
                packed_kv=True,
                use_fp16=False,
                run_benchmark=args.run_benchmark,
            )
        if args.fp16_gpu:
            logging.info("Running fp16_gpu example...")
            run_phi2(
                onnx_model_path=model_type_to_args["fp16_gpu"][2],
                use_buffer_share=False,
                device_id=args.device_id,
                packed_kv=True,
                run_benchmark=args.run_benchmark,
            )
        if args.int4_gpu:
            logging.info("Running int4_gpu example...")
            run_phi2(
                onnx_model_path=model_type_to_args["int4_gpu"][2],
                use_buffer_share=False,
                device_id=args.device_id,
                packed_kv=True,
                run_benchmark=args.run_benchmark,
            )
        if args.fp32_cpu or args.int4_cpu or args.fp16_vllm or args.int4_vllm:
            raise NotImplementedError("CPU/vllm inference example is not implemented yet.")


if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\packaging\version.py ===
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.
"""
.. testsetup::

    from packaging.version import parse, Version
"""

from __future__ import annotations

import itertools
import re
from typing import Any, Callable, NamedTuple, SupportsInt, Tuple, Union

from ._structures import Infinity, InfinityType, NegativeInfinity, NegativeInfinityType

__all__ = ["VERSION_PATTERN", "InvalidVersion", "Version", "parse"]

LocalType = Tuple[Union[int, str], ...]

CmpPrePostDevType = Union[InfinityType, NegativeInfinityType, Tuple[str, int]]
CmpLocalType = Union[
    NegativeInfinityType,
    Tuple[Union[Tuple[int, str], Tuple[NegativeInfinityType, Union[int, str]]], ...],
]
CmpKey = Tuple[
    int,
    Tuple[int, ...],
    CmpPrePostDevType,
    CmpPrePostDevType,
    CmpPrePostDevType,
    CmpLocalType,
]
VersionComparisonMethod = Callable[[CmpKey, CmpKey], bool]


class _Version(NamedTuple):
    epoch: int
    release: tuple[int, ...]
    dev: tuple[str, int] | None
    pre: tuple[str, int] | None
    post: tuple[str, int] | None
    local: LocalType | None


def parse(version: str) -> Version:
    """Parse the given version string.

    >>> parse('1.0.dev1')
    <Version('1.0.dev1')>

    :param version: The version string to parse.
    :raises InvalidVersion: When the version string is not a valid version.
    """
    return Version(version)


class InvalidVersion(ValueError):
    """Raised when a version string is not a valid version.

    >>> Version("invalid")
    Traceback (most recent call last):
        ...
    packaging.version.InvalidVersion: Invalid version: 'invalid'
    """


class _BaseVersion:
    _key: tuple[Any, ...]

    def __hash__(self) -> int:
        return hash(self._key)

    # Please keep the duplicated `isinstance` check
    # in the six comparisons hereunder
    # unless you find a way to avoid adding overhead function calls.
    def __lt__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key < other._key

    def __le__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key <= other._key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key == other._key

    def __ge__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key >= other._key

    def __gt__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key > other._key

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key != other._key


# Deliberately not anchored to the start and end of the string, to make it
# easier for 3rd party code to reuse
_VERSION_PATTERN = r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>alpha|a|beta|b|preview|pre|c|rc)
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""

VERSION_PATTERN = _VERSION_PATTERN
"""
A string containing the regular expression used to match a valid version.

The pattern is not anchored at either end, and is intended for embedding in larger
expressions (for example, matching a version number as part of a file name). The
regular expression should be compiled with the ``re.VERBOSE`` and ``re.IGNORECASE``
flags set.

:meta hide-value:
"""


class Version(_BaseVersion):
    """This class abstracts handling of a project's versions.

    A :class:`Version` instance is comparison aware and can be compared and
    sorted using the standard Python interfaces.

    >>> v1 = Version("1.0a5")
    >>> v2 = Version("1.0")
    >>> v1
    <Version('1.0a5')>
    >>> v2
    <Version('1.0')>
    >>> v1 < v2
    True
    >>> v1 == v2
    False
    >>> v1 > v2
    False
    >>> v1 >= v2
    False
    >>> v1 <= v2
    True
    """

    _regex = re.compile(r"^\s*" + VERSION_PATTERN + r"\s*$", re.VERBOSE | re.IGNORECASE)
    _key: CmpKey

    def __init__(self, version: str) -> None:
        """Initialize a Version object.

        :param version:
            The string representation of a version which will be parsed and normalized
            before use.
        :raises InvalidVersion:
            If the ``version`` does not conform to PEP 440 in any way then this
            exception will be raised.
        """

        # Validate the version and parse it into pieces
        match = self._regex.search(version)
        if not match:
            raise InvalidVersion(f"Invalid version: {version!r}")

        # Store the parsed out pieces of the version
        self._version = _Version(
            epoch=int(match.group("epoch")) if match.group("epoch") else 0,
            release=tuple(int(i) for i in match.group("release").split(".")),
            pre=_parse_letter_version(match.group("pre_l"), match.group("pre_n")),
            post=_parse_letter_version(
                match.group("post_l"), match.group("post_n1") or match.group("post_n2")
            ),
            dev=_parse_letter_version(match.group("dev_l"), match.group("dev_n")),
            local=_parse_local_version(match.group("local")),
        )

        # Generate a key which will be used for sorting
        self._key = _cmpkey(
            self._version.epoch,
            self._version.release,
            self._version.pre,
            self._version.post,
            self._version.dev,
            self._version.local,
        )

    def __repr__(self) -> str:
        """A representation of the Version that shows all internal state.

        >>> Version('1.0.0')
        <Version('1.0.0')>
        """
        return f"<Version('{self}')>"

    def __str__(self) -> str:
        """A string representation of the version that can be round-tripped.

        >>> str(Version("1.0a5"))
        '1.0a5'
        """
        parts = []

        # Epoch
        if self.epoch != 0:
            parts.append(f"{self.epoch}!")

        # Release segment
        parts.append(".".join(str(x) for x in self.release))

        # Pre-release
        if self.pre is not None:
            parts.append("".join(str(x) for x in self.pre))

        # Post-release
        if self.post is not None:
            parts.append(f".post{self.post}")

        # Development release
        if self.dev is not None:
            parts.append(f".dev{self.dev}")

        # Local version segment
        if self.local is not None:
            parts.append(f"+{self.local}")

        return "".join(parts)

    @property
    def epoch(self) -> int:
        """The epoch of the version.

        >>> Version("2.0.0").epoch
        0
        >>> Version("1!2.0.0").epoch
        1
        """
        return self._version.epoch

    @property
    def release(self) -> tuple[int, ...]:
        """The components of the "release" segment of the version.

        >>> Version("1.2.3").release
        (1, 2, 3)
        >>> Version("2.0.0").release
        (2, 0, 0)
        >>> Version("1!2.0.0.post0").release
        (2, 0, 0)

        Includes trailing zeroes but not the epoch or any pre-release / development /
        post-release suffixes.
        """
        return self._version.release

    @property
    def pre(self) -> tuple[str, int] | None:
        """The pre-release segment of the version.

        >>> print(Version("1.2.3").pre)
        None
        >>> Version("1.2.3a1").pre
        ('a', 1)
        >>> Version("1.2.3b1").pre
        ('b', 1)
        >>> Version("1.2.3rc1").pre
        ('rc', 1)
        """
        return self._version.pre

    @property
    def post(self) -> int | None:
        """The post-release number of the version.

        >>> print(Version("1.2.3").post)
        None
        >>> Version("1.2.3.post1").post
        1
        """
        return self._version.post[1] if self._version.post else None

    @property
    def dev(self) -> int | None:
        """The development number of the version.

        >>> print(Version("1.2.3").dev)
        None
        >>> Version("1.2.3.dev1").dev
        1
        """
        return self._version.dev[1] if self._version.dev else None

    @property
    def local(self) -> str | None:
        """The local version segment of the version.

        >>> print(Version("1.2.3").local)
        None
        >>> Version("1.2.3+abc").local
        'abc'
        """
        if self._version.local:
            return ".".join(str(x) for x in self._version.local)
        else:
            return None

    @property
    def public(self) -> str:
        """The public portion of the version.

        >>> Version("1.2.3").public
        '1.2.3'
        >>> Version("1.2.3+abc").public
        '1.2.3'
        >>> Version("1!1.2.3dev1+abc").public
        '1!1.2.3.dev1'
        """
        return str(self).split("+", 1)[0]

    @property
    def base_version(self) -> str:
        """The "base version" of the version.

        >>> Version("1.2.3").base_version
        '1.2.3'
        >>> Version("1.2.3+abc").base_version
        '1.2.3'
        >>> Version("1!1.2.3dev1+abc").base_version
        '1!1.2.3'

        The "base version" is the public version of the project without any pre or post
        release markers.
        """
        parts = []

        # Epoch
        if self.epoch != 0:
            parts.append(f"{self.epoch}!")

        # Release segment
        parts.append(".".join(str(x) for x in self.release))

        return "".join(parts)

    @property
    def is_prerelease(self) -> bool:
        """Whether this version is a pre-release.

        >>> Version("1.2.3").is_prerelease
        False
        >>> Version("1.2.3a1").is_prerelease
        True
        >>> Version("1.2.3b1").is_prerelease
        True
        >>> Version("1.2.3rc1").is_prerelease
        True
        >>> Version("1.2.3dev1").is_prerelease
        True
        """
        return self.dev is not None or self.pre is not None

    @property
    def is_postrelease(self) -> bool:
        """Whether this version is a post-release.

        >>> Version("1.2.3").is_postrelease
        False
        >>> Version("1.2.3.post1").is_postrelease
        True
        """
        return self.post is not None

    @property
    def is_devrelease(self) -> bool:
        """Whether this version is a development release.

        >>> Version("1.2.3").is_devrelease
        False
        >>> Version("1.2.3.dev1").is_devrelease
        True
        """
        return self.dev is not None

    @property
    def major(self) -> int:
        """The first item of :attr:`release` or ``0`` if unavailable.

        >>> Version("1.2.3").major
        1
        """
        return self.release[0] if len(self.release) >= 1 else 0

    @property
    def minor(self) -> int:
        """The second item of :attr:`release` or ``0`` if unavailable.

        >>> Version("1.2.3").minor
        2
        >>> Version("1").minor
        0
        """
        return self.release[1] if len(self.release) >= 2 else 0

    @property
    def micro(self) -> int:
        """The third item of :attr:`release` or ``0`` if unavailable.

        >>> Version("1.2.3").micro
        3
        >>> Version("1").micro
        0
        """
        return self.release[2] if len(self.release) >= 3 else 0


class _TrimmedRelease(Version):
    @property
    def release(self) -> tuple[int, ...]:
        """
        Release segment without any trailing zeros.

        >>> _TrimmedRelease('1.0.0').release
        (1,)
        >>> _TrimmedRelease('0.0').release
        (0,)
        """
        rel = super().release
        nonzeros = (index for index, val in enumerate(rel) if val)
        last_nonzero = max(nonzeros, default=0)
        return rel[: last_nonzero + 1]


def _parse_letter_version(
    letter: str | None, number: str | bytes | SupportsInt | None
) -> tuple[str, int] | None:
    if letter:
        # We consider there to be an implicit 0 in a pre-release if there is
        # not a numeral associated with it.
        if number is None:
            number = 0

        # We normalize any letters to their lower case form
        letter = letter.lower()

        # We consider some words to be alternate spellings of other words and
        # in those cases we want to normalize the spellings to our preferred
        # spelling.
        if letter == "alpha":
            letter = "a"
        elif letter == "beta":
            letter = "b"
        elif letter in ["c", "pre", "preview"]:
            letter = "rc"
        elif letter in ["rev", "r"]:
            letter = "post"

        return letter, int(number)

    assert not letter
    if number:
        # We assume if we are given a number, but we are not given a letter
        # then this is using the implicit post release syntax (e.g. 1.0-1)
        letter = "post"

        return letter, int(number)

    return None


_local_version_separators = re.compile(r"[\._-]")


def _parse_local_version(local: str | None) -> LocalType | None:
    """
    Takes a string like abc.1.twelve and turns it into ("abc", 1, "twelve").
    """
    if local is not None:
        return tuple(
            part.lower() if not part.isdigit() else int(part)
            for part in _local_version_separators.split(local)
        )
    return None


def _cmpkey(
    epoch: int,
    release: tuple[int, ...],
    pre: tuple[str, int] | None,
    post: tuple[str, int] | None,
    dev: tuple[str, int] | None,
    local: LocalType | None,
) -> CmpKey:
    # When we compare a release version, we want to compare it with all of the
    # trailing zeros removed. So we'll use a reverse the list, drop all the now
    # leading zeros until we come to something non zero, then take the rest
    # re-reverse it back into the correct order and make it a tuple and use
    # that for our sorting key.
    _release = tuple(
        reversed(list(itertools.dropwhile(lambda x: x == 0, reversed(release))))
    )

    # We need to "trick" the sorting algorithm to put 1.0.dev0 before 1.0a0.
    # We'll do this by abusing the pre segment, but we _only_ want to do this
    # if there is not a pre or a post segment. If we have one of those then
    # the normal sorting rules will handle this case correctly.
    if pre is None and post is None and dev is not None:
        _pre: CmpPrePostDevType = NegativeInfinity
    # Versions without a pre-release (except as noted above) should sort after
    # those with one.
    elif pre is None:
        _pre = Infinity
    else:
        _pre = pre

    # Versions without a post segment should sort before those with one.
    if post is None:
        _post: CmpPrePostDevType = NegativeInfinity

    else:
        _post = post

    # Versions without a development segment should sort after those with one.
    if dev is None:
        _dev: CmpPrePostDevType = Infinity

    else:
        _dev = dev

    if local is None:
        # Versions without a local segment should sort before those with one.
        _local: CmpLocalType = NegativeInfinity
    else:
        # Versions with a local segment need that segment parsed to implement
        # the sorting rules in PEP440.
        # - Alpha numeric segments sort before numeric segments
        # - Alpha numeric segments sort lexicographically
        # - Numeric segments sort numerically
        # - Shorter versions sort before longer versions when the prefixes
        #   match exactly
        _local = tuple(
            (i, "") if isinstance(i, int) else (NegativeInfinity, i) for i in local
        )

    return epoch, _release, _pre, _post, _dev, _local

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\packaging\version.py ===
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.
"""
.. testsetup::

    from pip._vendor.packaging.version import parse, Version
"""

from __future__ import annotations

import itertools
import re
from typing import Any, Callable, NamedTuple, SupportsInt, Tuple, Union

from ._structures import Infinity, InfinityType, NegativeInfinity, NegativeInfinityType

__all__ = ["VERSION_PATTERN", "InvalidVersion", "Version", "parse"]

LocalType = Tuple[Union[int, str], ...]

CmpPrePostDevType = Union[InfinityType, NegativeInfinityType, Tuple[str, int]]
CmpLocalType = Union[
    NegativeInfinityType,
    Tuple[Union[Tuple[int, str], Tuple[NegativeInfinityType, Union[int, str]]], ...],
]
CmpKey = Tuple[
    int,
    Tuple[int, ...],
    CmpPrePostDevType,
    CmpPrePostDevType,
    CmpPrePostDevType,
    CmpLocalType,
]
VersionComparisonMethod = Callable[[CmpKey, CmpKey], bool]


class _Version(NamedTuple):
    epoch: int
    release: tuple[int, ...]
    dev: tuple[str, int] | None
    pre: tuple[str, int] | None
    post: tuple[str, int] | None
    local: LocalType | None


def parse(version: str) -> Version:
    """Parse the given version string.

    >>> parse('1.0.dev1')
    <Version('1.0.dev1')>

    :param version: The version string to parse.
    :raises InvalidVersion: When the version string is not a valid version.
    """
    return Version(version)


class InvalidVersion(ValueError):
    """Raised when a version string is not a valid version.

    >>> Version("invalid")
    Traceback (most recent call last):
        ...
    packaging.version.InvalidVersion: Invalid version: 'invalid'
    """


class _BaseVersion:
    _key: tuple[Any, ...]

    def __hash__(self) -> int:
        return hash(self._key)

    # Please keep the duplicated `isinstance` check
    # in the six comparisons hereunder
    # unless you find a way to avoid adding overhead function calls.
    def __lt__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key < other._key

    def __le__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key <= other._key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key == other._key

    def __ge__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key >= other._key

    def __gt__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key > other._key

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key != other._key


# Deliberately not anchored to the start and end of the string, to make it
# easier for 3rd party code to reuse
_VERSION_PATTERN = r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>alpha|a|beta|b|preview|pre|c|rc)
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""

VERSION_PATTERN = _VERSION_PATTERN
"""
A string containing the regular expression used to match a valid version.

The pattern is not anchored at either end, and is intended for embedding in larger
expressions (for example, matching a version number as part of a file name). The
regular expression should be compiled with the ``re.VERBOSE`` and ``re.IGNORECASE``
flags set.

:meta hide-value:
"""


class Version(_BaseVersion):
    """This class abstracts handling of a project's versions.

    A :class:`Version` instance is comparison aware and can be compared and
    sorted using the standard Python interfaces.

    >>> v1 = Version("1.0a5")
    >>> v2 = Version("1.0")
    >>> v1
    <Version('1.0a5')>
    >>> v2
    <Version('1.0')>
    >>> v1 < v2
    True
    >>> v1 == v2
    False
    >>> v1 > v2
    False
    >>> v1 >= v2
    False
    >>> v1 <= v2
    True
    """

    _regex = re.compile(r"^\s*" + VERSION_PATTERN + r"\s*$", re.VERBOSE | re.IGNORECASE)
    _key: CmpKey

    def __init__(self, version: str) -> None:
        """Initialize a Version object.

        :param version:
            The string representation of a version which will be parsed and normalized
            before use.
        :raises InvalidVersion:
            If the ``version`` does not conform to PEP 440 in any way then this
            exception will be raised.
        """

        # Validate the version and parse it into pieces
        match = self._regex.search(version)
        if not match:
            raise InvalidVersion(f"Invalid version: {version!r}")

        # Store the parsed out pieces of the version
        self._version = _Version(
            epoch=int(match.group("epoch")) if match.group("epoch") else 0,
            release=tuple(int(i) for i in match.group("release").split(".")),
            pre=_parse_letter_version(match.group("pre_l"), match.group("pre_n")),
            post=_parse_letter_version(
                match.group("post_l"), match.group("post_n1") or match.group("post_n2")
            ),
            dev=_parse_letter_version(match.group("dev_l"), match.group("dev_n")),
            local=_parse_local_version(match.group("local")),
        )

        # Generate a key which will be used for sorting
        self._key = _cmpkey(
            self._version.epoch,
            self._version.release,
            self._version.pre,
            self._version.post,
            self._version.dev,
            self._version.local,
        )

    def __repr__(self) -> str:
        """A representation of the Version that shows all internal state.

        >>> Version('1.0.0')
        <Version('1.0.0')>
        """
        return f"<Version('{self}')>"

    def __str__(self) -> str:
        """A string representation of the version that can be round-tripped.

        >>> str(Version("1.0a5"))
        '1.0a5'
        """
        parts = []

        # Epoch
        if self.epoch != 0:
            parts.append(f"{self.epoch}!")

        # Release segment
        parts.append(".".join(str(x) for x in self.release))

        # Pre-release
        if self.pre is not None:
            parts.append("".join(str(x) for x in self.pre))

        # Post-release
        if self.post is not None:
            parts.append(f".post{self.post}")

        # Development release
        if self.dev is not None:
            parts.append(f".dev{self.dev}")

        # Local version segment
        if self.local is not None:
            parts.append(f"+{self.local}")

        return "".join(parts)

    @property
    def epoch(self) -> int:
        """The epoch of the version.

        >>> Version("2.0.0").epoch
        0
        >>> Version("1!2.0.0").epoch
        1
        """
        return self._version.epoch

    @property
    def release(self) -> tuple[int, ...]:
        """The components of the "release" segment of the version.

        >>> Version("1.2.3").release
        (1, 2, 3)
        >>> Version("2.0.0").release
        (2, 0, 0)
        >>> Version("1!2.0.0.post0").release
        (2, 0, 0)

        Includes trailing zeroes but not the epoch or any pre-release / development /
        post-release suffixes.
        """
        return self._version.release

    @property
    def pre(self) -> tuple[str, int] | None:
        """The pre-release segment of the version.

        >>> print(Version("1.2.3").pre)
        None
        >>> Version("1.2.3a1").pre
        ('a', 1)
        >>> Version("1.2.3b1").pre
        ('b', 1)
        >>> Version("1.2.3rc1").pre
        ('rc', 1)
        """
        return self._version.pre

    @property
    def post(self) -> int | None:
        """The post-release number of the version.

        >>> print(Version("1.2.3").post)
        None
        >>> Version("1.2.3.post1").post
        1
        """
        return self._version.post[1] if self._version.post else None

    @property
    def dev(self) -> int | None:
        """The development number of the version.

        >>> print(Version("1.2.3").dev)
        None
        >>> Version("1.2.3.dev1").dev
        1
        """
        return self._version.dev[1] if self._version.dev else None

    @property
    def local(self) -> str | None:
        """The local version segment of the version.

        >>> print(Version("1.2.3").local)
        None
        >>> Version("1.2.3+abc").local
        'abc'
        """
        if self._version.local:
            return ".".join(str(x) for x in self._version.local)
        else:
            return None

    @property
    def public(self) -> str:
        """The public portion of the version.

        >>> Version("1.2.3").public
        '1.2.3'
        >>> Version("1.2.3+abc").public
        '1.2.3'
        >>> Version("1!1.2.3dev1+abc").public
        '1!1.2.3.dev1'
        """
        return str(self).split("+", 1)[0]

    @property
    def base_version(self) -> str:
        """The "base version" of the version.

        >>> Version("1.2.3").base_version
        '1.2.3'
        >>> Version("1.2.3+abc").base_version
        '1.2.3'
        >>> Version("1!1.2.3dev1+abc").base_version
        '1!1.2.3'

        The "base version" is the public version of the project without any pre or post
        release markers.
        """
        parts = []

        # Epoch
        if self.epoch != 0:
            parts.append(f"{self.epoch}!")

        # Release segment
        parts.append(".".join(str(x) for x in self.release))

        return "".join(parts)

    @property
    def is_prerelease(self) -> bool:
        """Whether this version is a pre-release.

        >>> Version("1.2.3").is_prerelease
        False
        >>> Version("1.2.3a1").is_prerelease
        True
        >>> Version("1.2.3b1").is_prerelease
        True
        >>> Version("1.2.3rc1").is_prerelease
        True
        >>> Version("1.2.3dev1").is_prerelease
        True
        """
        return self.dev is not None or self.pre is not None

    @property
    def is_postrelease(self) -> bool:
        """Whether this version is a post-release.

        >>> Version("1.2.3").is_postrelease
        False
        >>> Version("1.2.3.post1").is_postrelease
        True
        """
        return self.post is not None

    @property
    def is_devrelease(self) -> bool:
        """Whether this version is a development release.

        >>> Version("1.2.3").is_devrelease
        False
        >>> Version("1.2.3.dev1").is_devrelease
        True
        """
        return self.dev is not None

    @property
    def major(self) -> int:
        """The first item of :attr:`release` or ``0`` if unavailable.

        >>> Version("1.2.3").major
        1
        """
        return self.release[0] if len(self.release) >= 1 else 0

    @property
    def minor(self) -> int:
        """The second item of :attr:`release` or ``0`` if unavailable.

        >>> Version("1.2.3").minor
        2
        >>> Version("1").minor
        0
        """
        return self.release[1] if len(self.release) >= 2 else 0

    @property
    def micro(self) -> int:
        """The third item of :attr:`release` or ``0`` if unavailable.

        >>> Version("1.2.3").micro
        3
        >>> Version("1").micro
        0
        """
        return self.release[2] if len(self.release) >= 3 else 0


class _TrimmedRelease(Version):
    @property
    def release(self) -> tuple[int, ...]:
        """
        Release segment without any trailing zeros.

        >>> _TrimmedRelease('1.0.0').release
        (1,)
        >>> _TrimmedRelease('0.0').release
        (0,)
        """
        rel = super().release
        nonzeros = (index for index, val in enumerate(rel) if val)
        last_nonzero = max(nonzeros, default=0)
        return rel[: last_nonzero + 1]


def _parse_letter_version(
    letter: str | None, number: str | bytes | SupportsInt | None
) -> tuple[str, int] | None:
    if letter:
        # We consider there to be an implicit 0 in a pre-release if there is
        # not a numeral associated with it.
        if number is None:
            number = 0

        # We normalize any letters to their lower case form
        letter = letter.lower()

        # We consider some words to be alternate spellings of other words and
        # in those cases we want to normalize the spellings to our preferred
        # spelling.
        if letter == "alpha":
            letter = "a"
        elif letter == "beta":
            letter = "b"
        elif letter in ["c", "pre", "preview"]:
            letter = "rc"
        elif letter in ["rev", "r"]:
            letter = "post"

        return letter, int(number)

    assert not letter
    if number:
        # We assume if we are given a number, but we are not given a letter
        # then this is using the implicit post release syntax (e.g. 1.0-1)
        letter = "post"

        return letter, int(number)

    return None


_local_version_separators = re.compile(r"[\._-]")


def _parse_local_version(local: str | None) -> LocalType | None:
    """
    Takes a string like abc.1.twelve and turns it into ("abc", 1, "twelve").
    """
    if local is not None:
        return tuple(
            part.lower() if not part.isdigit() else int(part)
            for part in _local_version_separators.split(local)
        )
    return None


def _cmpkey(
    epoch: int,
    release: tuple[int, ...],
    pre: tuple[str, int] | None,
    post: tuple[str, int] | None,
    dev: tuple[str, int] | None,
    local: LocalType | None,
) -> CmpKey:
    # When we compare a release version, we want to compare it with all of the
    # trailing zeros removed. So we'll use a reverse the list, drop all the now
    # leading zeros until we come to something non zero, then take the rest
    # re-reverse it back into the correct order and make it a tuple and use
    # that for our sorting key.
    _release = tuple(
        reversed(list(itertools.dropwhile(lambda x: x == 0, reversed(release))))
    )

    # We need to "trick" the sorting algorithm to put 1.0.dev0 before 1.0a0.
    # We'll do this by abusing the pre segment, but we _only_ want to do this
    # if there is not a pre or a post segment. If we have one of those then
    # the normal sorting rules will handle this case correctly.
    if pre is None and post is None and dev is not None:
        _pre: CmpPrePostDevType = NegativeInfinity
    # Versions without a pre-release (except as noted above) should sort after
    # those with one.
    elif pre is None:
        _pre = Infinity
    else:
        _pre = pre

    # Versions without a post segment should sort before those with one.
    if post is None:
        _post: CmpPrePostDevType = NegativeInfinity

    else:
        _post = post

    # Versions without a development segment should sort after those with one.
    if dev is None:
        _dev: CmpPrePostDevType = Infinity

    else:
        _dev = dev

    if local is None:
        # Versions without a local segment should sort before those with one.
        _local: CmpLocalType = NegativeInfinity
    else:
        # Versions with a local segment need that segment parsed to implement
        # the sorting rules in PEP440.
        # - Alpha numeric segments sort before numeric segments
        # - Alpha numeric segments sort lexicographically
        # - Numeric segments sort numerically
        # - Shorter versions sort before longer versions when the prefixes
        #   match exactly
        _local = tuple(
            (i, "") if isinstance(i, int) else (NegativeInfinity, i) for i in local
        )

    return epoch, _release, _pre, _post, _dev, _local