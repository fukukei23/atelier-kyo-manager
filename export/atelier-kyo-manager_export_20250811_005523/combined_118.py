
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\ops\array_ops.py ===
"""
Functions for arithmetic and comparison operations on NumPy arrays and
ExtensionArrays.
"""
from __future__ import annotations

import datetime
from functools import partial
import operator
from typing import (
    TYPE_CHECKING,
    Any,
)
import warnings

import numpy as np

from pandas._libs import (
    NaT,
    Timedelta,
    Timestamp,
    lib,
    ops as libops,
)
from pandas._libs.tslibs import (
    BaseOffset,
    get_supported_dtype,
    is_supported_dtype,
    is_unitless,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import (
    construct_1d_object_array_from_listlike,
    find_common_type,
)
from pandas.core.dtypes.common import (
    ensure_object,
    is_bool_dtype,
    is_list_like,
    is_numeric_v_string_like,
    is_object_dtype,
    is_scalar,
)
from pandas.core.dtypes.generic import (
    ABCExtensionArray,
    ABCIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import (
    isna,
    notna,
)

from pandas.core import roperator
from pandas.core.computation import expressions
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.ops import missing
from pandas.core.ops.dispatch import should_extension_dispatch
from pandas.core.ops.invalid import invalid_comparison

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        Shape,
    )

# -----------------------------------------------------------------------------
# Masking NA values and fallbacks for operations numpy does not support


def fill_binop(left, right, fill_value):
    """
    If a non-None fill_value is given, replace null entries in left and right
    with this value, but only in positions where _one_ of left/right is null,
    not both.

    Parameters
    ----------
    left : array-like
    right : array-like
    fill_value : object

    Returns
    -------
    left : array-like
    right : array-like

    Notes
    -----
    Makes copies if fill_value is not None and NAs are present.
    """
    if fill_value is not None:
        left_mask = isna(left)
        right_mask = isna(right)

        # one but not both
        mask = left_mask ^ right_mask

        if left_mask.any():
            # Avoid making a copy if we can
            left = left.copy()
            left[left_mask & mask] = fill_value

        if right_mask.any():
            # Avoid making a copy if we can
            right = right.copy()
            right[right_mask & mask] = fill_value

    return left, right


def comp_method_OBJECT_ARRAY(op, x, y):
    if isinstance(y, list):
        # e.g. test_tuple_categories
        y = construct_1d_object_array_from_listlike(y)

    if isinstance(y, (np.ndarray, ABCSeries, ABCIndex)):
        if not is_object_dtype(y.dtype):
            y = y.astype(np.object_)

        if isinstance(y, (ABCSeries, ABCIndex)):
            y = y._values

        if x.shape != y.shape:
            raise ValueError("Shapes must match", x.shape, y.shape)
        result = libops.vec_compare(x.ravel(), y.ravel(), op)
    else:
        result = libops.scalar_compare(x.ravel(), y, op)
    return result.reshape(x.shape)


def _masked_arith_op(x: np.ndarray, y, op):
    """
    If the given arithmetic operation fails, attempt it again on
    only the non-null elements of the input array(s).

    Parameters
    ----------
    x : np.ndarray
    y : np.ndarray, Series, Index
    op : binary operator
    """
    # For Series `x` is 1D so ravel() is a no-op; calling it anyway makes
    # the logic valid for both Series and DataFrame ops.
    xrav = x.ravel()

    if isinstance(y, np.ndarray):
        dtype = find_common_type([x.dtype, y.dtype])
        result = np.empty(x.size, dtype=dtype)

        if len(x) != len(y):
            raise ValueError(x.shape, y.shape)
        ymask = notna(y)

        # NB: ravel() is only safe since y is ndarray; for e.g. PeriodIndex
        #  we would get int64 dtype, see GH#19956
        yrav = y.ravel()
        mask = notna(xrav) & ymask.ravel()

        # See GH#5284, GH#5035, GH#19448 for historical reference
        if mask.any():
            result[mask] = op(xrav[mask], yrav[mask])

    else:
        if not is_scalar(y):
            raise TypeError(
                f"Cannot broadcast np.ndarray with operand of type { type(y) }"
            )

        # mask is only meaningful for x
        result = np.empty(x.size, dtype=x.dtype)
        mask = notna(xrav)

        # 1 ** np.nan is 1. So we have to unmask those.
        if op is pow:
            mask = np.where(x == 1, False, mask)
        elif op is roperator.rpow:
            mask = np.where(y == 1, False, mask)

        if mask.any():
            result[mask] = op(xrav[mask], y)

    np.putmask(result, ~mask, np.nan)
    result = result.reshape(x.shape)  # 2D compat
    return result


def _na_arithmetic_op(left: np.ndarray, right, op, is_cmp: bool = False):
    """
    Return the result of evaluating op on the passed in values.

    If native types are not compatible, try coercion to object dtype.

    Parameters
    ----------
    left : np.ndarray
    right : np.ndarray or scalar
        Excludes DataFrame, Series, Index, ExtensionArray.
    is_cmp : bool, default False
        If this a comparison operation.

    Returns
    -------
    array-like

    Raises
    ------
    TypeError : invalid operation
    """
    if isinstance(right, str):
        # can never use numexpr
        func = op
    else:
        func = partial(expressions.evaluate, op)

    try:
        result = func(left, right)
    except TypeError:
        if not is_cmp and (
            left.dtype == object or getattr(right, "dtype", None) == object
        ):
            # For object dtype, fallback to a masked operation (only operating
            #  on the non-missing values)
            # Don't do this for comparisons, as that will handle complex numbers
            #  incorrectly, see GH#32047
            result = _masked_arith_op(left, right, op)
        else:
            raise

    if is_cmp and (is_scalar(result) or result is NotImplemented):
        # numpy returned a scalar instead of operating element-wise
        # e.g. numeric array vs str
        # TODO: can remove this after dropping some future numpy version?
        return invalid_comparison(left, right, op)

    return missing.dispatch_fill_zeros(op, left, right, result)


def arithmetic_op(left: ArrayLike, right: Any, op):
    """
    Evaluate an arithmetic operation `+`, `-`, `*`, `/`, `//`, `%`, `**`, ...

    Note: the caller is responsible for ensuring that numpy warnings are
    suppressed (with np.errstate(all="ignore")) if needed.

    Parameters
    ----------
    left : np.ndarray or ExtensionArray
    right : object
        Cannot be a DataFrame or Index.  Series is *not* excluded.
    op : {operator.add, operator.sub, ...}
        Or one of the reversed variants from roperator.

    Returns
    -------
    ndarray or ExtensionArray
        Or a 2-tuple of these in the case of divmod or rdivmod.
    """
    # NB: We assume that extract_array and ensure_wrapped_if_datetimelike
    #  have already been called on `left` and `right`,
    #  and `maybe_prepare_scalar_for_op` has already been called on `right`
    # We need to special-case datetime64/timedelta64 dtypes (e.g. because numpy
    # casts integer dtypes to timedelta64 when operating with timedelta64 - GH#22390)

    if (
        should_extension_dispatch(left, right)
        or isinstance(right, (Timedelta, BaseOffset, Timestamp))
        or right is NaT
    ):
        # Timedelta/Timestamp and other custom scalars are included in the check
        # because numexpr will fail on it, see GH#31457
        res_values = op(left, right)
    else:
        # TODO we should handle EAs consistently and move this check before the if/else
        # (https://github.com/pandas-dev/pandas/issues/41165)
        # error: Argument 2 to "_bool_arith_check" has incompatible type
        # "Union[ExtensionArray, ndarray[Any, Any]]"; expected "ndarray[Any, Any]"
        _bool_arith_check(op, left, right)  # type: ignore[arg-type]

        # error: Argument 1 to "_na_arithmetic_op" has incompatible type
        # "Union[ExtensionArray, ndarray[Any, Any]]"; expected "ndarray[Any, Any]"
        res_values = _na_arithmetic_op(left, right, op)  # type: ignore[arg-type]

    return res_values


def comparison_op(left: ArrayLike, right: Any, op) -> ArrayLike:
    """
    Evaluate a comparison operation `=`, `!=`, `>=`, `>`, `<=`, or `<`.

    Note: the caller is responsible for ensuring that numpy warnings are
    suppressed (with np.errstate(all="ignore")) if needed.

    Parameters
    ----------
    left : np.ndarray or ExtensionArray
    right : object
        Cannot be a DataFrame, Series, or Index.
    op : {operator.eq, operator.ne, operator.gt, operator.ge, operator.lt, operator.le}

    Returns
    -------
    ndarray or ExtensionArray
    """
    # NB: We assume extract_array has already been called on left and right
    lvalues = ensure_wrapped_if_datetimelike(left)
    rvalues = ensure_wrapped_if_datetimelike(right)

    rvalues = lib.item_from_zerodim(rvalues)
    if isinstance(rvalues, list):
        # We don't catch tuple here bc we may be comparing e.g. MultiIndex
        #  to a tuple that represents a single entry, see test_compare_tuple_strs
        rvalues = np.asarray(rvalues)

    if isinstance(rvalues, (np.ndarray, ABCExtensionArray)):
        # TODO: make this treatment consistent across ops and classes.
        #  We are not catching all listlikes here (e.g. frozenset, tuple)
        #  The ambiguous case is object-dtype.  See GH#27803
        if len(lvalues) != len(rvalues):
            raise ValueError(
                "Lengths must match to compare", lvalues.shape, rvalues.shape
            )

    if should_extension_dispatch(lvalues, rvalues) or (
        (isinstance(rvalues, (Timedelta, BaseOffset, Timestamp)) or right is NaT)
        and lvalues.dtype != object
    ):
        # Call the method on lvalues
        res_values = op(lvalues, rvalues)

    elif is_scalar(rvalues) and isna(rvalues):  # TODO: but not pd.NA?
        # numpy does not like comparisons vs None
        if op is operator.ne:
            res_values = np.ones(lvalues.shape, dtype=bool)
        else:
            res_values = np.zeros(lvalues.shape, dtype=bool)

    elif is_numeric_v_string_like(lvalues, rvalues):
        # GH#36377 going through the numexpr path would incorrectly raise
        return invalid_comparison(lvalues, rvalues, op)

    elif lvalues.dtype == object or isinstance(rvalues, str):
        res_values = comp_method_OBJECT_ARRAY(op, lvalues, rvalues)

    else:
        res_values = _na_arithmetic_op(lvalues, rvalues, op, is_cmp=True)

    return res_values


def na_logical_op(x: np.ndarray, y, op):
    try:
        # For exposition, write:
        #  yarr = isinstance(y, np.ndarray)
        #  yint = is_integer(y) or (yarr and y.dtype.kind == "i")
        #  ybool = is_bool(y) or (yarr and y.dtype.kind == "b")
        #  xint = x.dtype.kind == "i"
        #  xbool = x.dtype.kind == "b"
        # Then Cases where this goes through without raising include:
        #  (xint or xbool) and (yint or bool)
        result = op(x, y)
    except TypeError:
        if isinstance(y, np.ndarray):
            # bool-bool dtype operations should be OK, should not get here
            assert not (x.dtype.kind == "b" and y.dtype.kind == "b")
            x = ensure_object(x)
            y = ensure_object(y)
            result = libops.vec_binop(x.ravel(), y.ravel(), op)
        else:
            # let null fall thru
            assert lib.is_scalar(y)
            if not isna(y):
                y = bool(y)
            try:
                result = libops.scalar_binop(x, y, op)
            except (
                TypeError,
                ValueError,
                AttributeError,
                OverflowError,
                NotImplementedError,
            ) as err:
                typ = type(y).__name__
                raise TypeError(
                    f"Cannot perform '{op.__name__}' with a dtyped [{x.dtype}] array "
                    f"and scalar of type [{typ}]"
                ) from err

    return result.reshape(x.shape)


def logical_op(left: ArrayLike, right: Any, op) -> ArrayLike:
    """
    Evaluate a logical operation `|`, `&`, or `^`.

    Parameters
    ----------
    left : np.ndarray or ExtensionArray
    right : object
        Cannot be a DataFrame, Series, or Index.
    op : {operator.and_, operator.or_, operator.xor}
        Or one of the reversed variants from roperator.

    Returns
    -------
    ndarray or ExtensionArray
    """

    def fill_bool(x, left=None):
        # if `left` is specifically not-boolean, we do not cast to bool
        if x.dtype.kind in "cfO":
            # dtypes that can hold NA
            mask = isna(x)
            if mask.any():
                x = x.astype(object)
                x[mask] = False

        if left is None or left.dtype.kind == "b":
            x = x.astype(bool)
        return x

    right = lib.item_from_zerodim(right)
    if is_list_like(right) and not hasattr(right, "dtype"):
        # e.g. list, tuple
        warnings.warn(
            "Logical ops (and, or, xor) between Pandas objects and dtype-less "
            "sequences (e.g. list, tuple) are deprecated and will raise in a "
            "future version. Wrap the object in a Series, Index, or np.array "
            "before operating instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        right = construct_1d_object_array_from_listlike(right)

    # NB: We assume extract_array has already been called on left and right
    lvalues = ensure_wrapped_if_datetimelike(left)
    rvalues = right

    if should_extension_dispatch(lvalues, rvalues):
        # Call the method on lvalues
        res_values = op(lvalues, rvalues)

    else:
        if isinstance(rvalues, np.ndarray):
            is_other_int_dtype = rvalues.dtype.kind in "iu"
            if not is_other_int_dtype:
                rvalues = fill_bool(rvalues, lvalues)

        else:
            # i.e. scalar
            is_other_int_dtype = lib.is_integer(rvalues)

        res_values = na_logical_op(lvalues, rvalues, op)

        # For int vs int `^`, `|`, `&` are bitwise operators and return
        #   integer dtypes.  Otherwise these are boolean ops
        if not (left.dtype.kind in "iu" and is_other_int_dtype):
            res_values = fill_bool(res_values)

    return res_values


def get_array_op(op):
    """
    Return a binary array operation corresponding to the given operator op.

    Parameters
    ----------
    op : function
        Binary operator from operator or roperator module.

    Returns
    -------
    functools.partial
    """
    if isinstance(op, partial):
        # We get here via dispatch_to_series in DataFrame case
        # e.g. test_rolling_consistency_var_debiasing_factors
        return op

    op_name = op.__name__.strip("_").lstrip("r")
    if op_name == "arith_op":
        # Reached via DataFrame._combine_frame i.e. flex methods
        # e.g. test_df_add_flex_filled_mixed_dtypes
        return op

    if op_name in {"eq", "ne", "lt", "le", "gt", "ge"}:
        return partial(comparison_op, op=op)
    elif op_name in {"and", "or", "xor", "rand", "ror", "rxor"}:
        return partial(logical_op, op=op)
    elif op_name in {
        "add",
        "sub",
        "mul",
        "truediv",
        "floordiv",
        "mod",
        "divmod",
        "pow",
    }:
        return partial(arithmetic_op, op=op)
    else:
        raise NotImplementedError(op_name)


def maybe_prepare_scalar_for_op(obj, shape: Shape):
    """
    Cast non-pandas objects to pandas types to unify behavior of arithmetic
    and comparison operations.

    Parameters
    ----------
    obj: object
    shape : tuple[int]

    Returns
    -------
    out : object

    Notes
    -----
    Be careful to call this *after* determining the `name` attribute to be
    attached to the result of the arithmetic operation.
    """
    if type(obj) is datetime.timedelta:
        # GH#22390  cast up to Timedelta to rely on Timedelta
        # implementation; otherwise operation against numeric-dtype
        # raises TypeError
        return Timedelta(obj)
    elif type(obj) is datetime.datetime:
        # cast up to Timestamp to rely on Timestamp implementation, see Timedelta above
        return Timestamp(obj)
    elif isinstance(obj, np.datetime64):
        # GH#28080 numpy casts integer-dtype to datetime64 when doing
        #  array[int] + datetime64, which we do not allow
        if isna(obj):
            from pandas.core.arrays import DatetimeArray

            # Avoid possible ambiguities with pd.NaT
            # GH 52295
            if is_unitless(obj.dtype):
                obj = obj.astype("datetime64[ns]")
            elif not is_supported_dtype(obj.dtype):
                new_dtype = get_supported_dtype(obj.dtype)
                obj = obj.astype(new_dtype)
            right = np.broadcast_to(obj, shape)
            return DatetimeArray._simple_new(right, dtype=right.dtype)

        return Timestamp(obj)

    elif isinstance(obj, np.timedelta64):
        if isna(obj):
            from pandas.core.arrays import TimedeltaArray

            # wrapping timedelta64("NaT") in Timedelta returns NaT,
            #  which would incorrectly be treated as a datetime-NaT, so
            #  we broadcast and wrap in a TimedeltaArray
            # GH 52295
            if is_unitless(obj.dtype):
                obj = obj.astype("timedelta64[ns]")
            elif not is_supported_dtype(obj.dtype):
                new_dtype = get_supported_dtype(obj.dtype)
                obj = obj.astype(new_dtype)
            right = np.broadcast_to(obj, shape)
            return TimedeltaArray._simple_new(right, dtype=right.dtype)

        # In particular non-nanosecond timedelta64 needs to be cast to
        #  nanoseconds, or else we get undesired behavior like
        #  np.timedelta64(3, 'D') / 2 == np.timedelta64(1, 'D')
        return Timedelta(obj)

    # We want NumPy numeric scalars to behave like Python scalars
    # post NEP 50
    elif isinstance(obj, np.integer):
        return int(obj)

    elif isinstance(obj, np.floating):
        return float(obj)

    return obj


_BOOL_OP_NOT_ALLOWED = {
    operator.truediv,
    roperator.rtruediv,
    operator.floordiv,
    roperator.rfloordiv,
    operator.pow,
    roperator.rpow,
}


def _bool_arith_check(op, a: np.ndarray, b):
    """
    In contrast to numpy, pandas raises an error for certain operations
    with booleans.
    """
    if op in _BOOL_OP_NOT_ALLOWED:
        if a.dtype.kind == "b" and (is_bool_dtype(b) or lib.is_bool(b)):
            op_name = op.__name__.strip("_").lstrip("r")
            raise NotImplementedError(
                f"operator '{op_name}' not implemented for bool dtypes"
            )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\web_audio.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: WebAudio (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

class GraphObjectId(str):
    '''
    An unique ID for a graph object (AudioContext, AudioNode, AudioParam) in Web Audio API
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> GraphObjectId:
        return cls(json)

    def __repr__(self):
        return 'GraphObjectId({})'.format(super().__repr__())


class ContextType(enum.Enum):
    '''
    Enum of BaseAudioContext types
    '''
    REALTIME = "realtime"
    OFFLINE = "offline"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class ContextState(enum.Enum):
    '''
    Enum of AudioContextState from the spec
    '''
    SUSPENDED = "suspended"
    RUNNING = "running"
    CLOSED = "closed"
    INTERRUPTED = "interrupted"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class NodeType(str):
    '''
    Enum of AudioNode types
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> NodeType:
        return cls(json)

    def __repr__(self):
        return 'NodeType({})'.format(super().__repr__())


class ChannelCountMode(enum.Enum):
    '''
    Enum of AudioNode::ChannelCountMode from the spec
    '''
    CLAMPED_MAX = "clamped-max"
    EXPLICIT = "explicit"
    MAX_ = "max"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class ChannelInterpretation(enum.Enum):
    '''
    Enum of AudioNode::ChannelInterpretation from the spec
    '''
    DISCRETE = "discrete"
    SPEAKERS = "speakers"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class ParamType(str):
    '''
    Enum of AudioParam types
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> ParamType:
        return cls(json)

    def __repr__(self):
        return 'ParamType({})'.format(super().__repr__())


class AutomationRate(enum.Enum):
    '''
    Enum of AudioParam::AutomationRate from the spec
    '''
    A_RATE = "a-rate"
    K_RATE = "k-rate"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class ContextRealtimeData:
    '''
    Fields in AudioContext that change in real-time.
    '''
    #: The current context time in second in BaseAudioContext.
    current_time: float

    #: The time spent on rendering graph divided by render quantum duration,
    #: and multiplied by 100. 100 means the audio renderer reached the full
    #: capacity and glitch may occur.
    render_capacity: float

    #: A running mean of callback interval.
    callback_interval_mean: float

    #: A running variance of callback interval.
    callback_interval_variance: float

    def to_json(self):
        json = dict()
        json['currentTime'] = self.current_time
        json['renderCapacity'] = self.render_capacity
        json['callbackIntervalMean'] = self.callback_interval_mean
        json['callbackIntervalVariance'] = self.callback_interval_variance
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            current_time=float(json['currentTime']),
            render_capacity=float(json['renderCapacity']),
            callback_interval_mean=float(json['callbackIntervalMean']),
            callback_interval_variance=float(json['callbackIntervalVariance']),
        )


@dataclass
class BaseAudioContext:
    '''
    Protocol object for BaseAudioContext
    '''
    context_id: GraphObjectId

    context_type: ContextType

    context_state: ContextState

    #: Platform-dependent callback buffer size.
    callback_buffer_size: float

    #: Number of output channels supported by audio hardware in use.
    max_output_channel_count: float

    #: Context sample rate.
    sample_rate: float

    realtime_data: typing.Optional[ContextRealtimeData] = None

    def to_json(self):
        json = dict()
        json['contextId'] = self.context_id.to_json()
        json['contextType'] = self.context_type.to_json()
        json['contextState'] = self.context_state.to_json()
        json['callbackBufferSize'] = self.callback_buffer_size
        json['maxOutputChannelCount'] = self.max_output_channel_count
        json['sampleRate'] = self.sample_rate
        if self.realtime_data is not None:
            json['realtimeData'] = self.realtime_data.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            context_id=GraphObjectId.from_json(json['contextId']),
            context_type=ContextType.from_json(json['contextType']),
            context_state=ContextState.from_json(json['contextState']),
            callback_buffer_size=float(json['callbackBufferSize']),
            max_output_channel_count=float(json['maxOutputChannelCount']),
            sample_rate=float(json['sampleRate']),
            realtime_data=ContextRealtimeData.from_json(json['realtimeData']) if 'realtimeData' in json else None,
        )


@dataclass
class AudioListener:
    '''
    Protocol object for AudioListener
    '''
    listener_id: GraphObjectId

    context_id: GraphObjectId

    def to_json(self):
        json = dict()
        json['listenerId'] = self.listener_id.to_json()
        json['contextId'] = self.context_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            listener_id=GraphObjectId.from_json(json['listenerId']),
            context_id=GraphObjectId.from_json(json['contextId']),
        )


@dataclass
class AudioNode:
    '''
    Protocol object for AudioNode
    '''
    node_id: GraphObjectId

    context_id: GraphObjectId

    node_type: NodeType

    number_of_inputs: float

    number_of_outputs: float

    channel_count: float

    channel_count_mode: ChannelCountMode

    channel_interpretation: ChannelInterpretation

    def to_json(self):
        json = dict()
        json['nodeId'] = self.node_id.to_json()
        json['contextId'] = self.context_id.to_json()
        json['nodeType'] = self.node_type.to_json()
        json['numberOfInputs'] = self.number_of_inputs
        json['numberOfOutputs'] = self.number_of_outputs
        json['channelCount'] = self.channel_count
        json['channelCountMode'] = self.channel_count_mode.to_json()
        json['channelInterpretation'] = self.channel_interpretation.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            node_id=GraphObjectId.from_json(json['nodeId']),
            context_id=GraphObjectId.from_json(json['contextId']),
            node_type=NodeType.from_json(json['nodeType']),
            number_of_inputs=float(json['numberOfInputs']),
            number_of_outputs=float(json['numberOfOutputs']),
            channel_count=float(json['channelCount']),
            channel_count_mode=ChannelCountMode.from_json(json['channelCountMode']),
            channel_interpretation=ChannelInterpretation.from_json(json['channelInterpretation']),
        )


@dataclass
class AudioParam:
    '''
    Protocol object for AudioParam
    '''
    param_id: GraphObjectId

    node_id: GraphObjectId

    context_id: GraphObjectId

    param_type: ParamType

    rate: AutomationRate

    default_value: float

    min_value: float

    max_value: float

    def to_json(self):
        json = dict()
        json['paramId'] = self.param_id.to_json()
        json['nodeId'] = self.node_id.to_json()
        json['contextId'] = self.context_id.to_json()
        json['paramType'] = self.param_type.to_json()
        json['rate'] = self.rate.to_json()
        json['defaultValue'] = self.default_value
        json['minValue'] = self.min_value
        json['maxValue'] = self.max_value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            param_id=GraphObjectId.from_json(json['paramId']),
            node_id=GraphObjectId.from_json(json['nodeId']),
            context_id=GraphObjectId.from_json(json['contextId']),
            param_type=ParamType.from_json(json['paramType']),
            rate=AutomationRate.from_json(json['rate']),
            default_value=float(json['defaultValue']),
            min_value=float(json['minValue']),
            max_value=float(json['maxValue']),
        )


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables the WebAudio domain and starts sending context lifetime events.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAudio.enable',
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables the WebAudio domain.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAudio.disable',
    }
    json = yield cmd_dict


def get_realtime_data(
        context_id: GraphObjectId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,ContextRealtimeData]:
    '''
    Fetch the realtime data from the registered contexts.

    :param context_id:
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['contextId'] = context_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAudio.getRealtimeData',
        'params': params,
    }
    json = yield cmd_dict
    return ContextRealtimeData.from_json(json['realtimeData'])


@event_class('WebAudio.contextCreated')
@dataclass
class ContextCreated:
    '''
    Notifies that a new BaseAudioContext has been created.
    '''
    context: BaseAudioContext

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ContextCreated:
        return cls(
            context=BaseAudioContext.from_json(json['context'])
        )


@event_class('WebAudio.contextWillBeDestroyed')
@dataclass
class ContextWillBeDestroyed:
    '''
    Notifies that an existing BaseAudioContext will be destroyed.
    '''
    context_id: GraphObjectId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ContextWillBeDestroyed:
        return cls(
            context_id=GraphObjectId.from_json(json['contextId'])
        )


@event_class('WebAudio.contextChanged')
@dataclass
class ContextChanged:
    '''
    Notifies that existing BaseAudioContext has changed some properties (id stays the same)..
    '''
    context: BaseAudioContext

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ContextChanged:
        return cls(
            context=BaseAudioContext.from_json(json['context'])
        )


@event_class('WebAudio.audioListenerCreated')
@dataclass
class AudioListenerCreated:
    '''
    Notifies that the construction of an AudioListener has finished.
    '''
    listener: AudioListener

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AudioListenerCreated:
        return cls(
            listener=AudioListener.from_json(json['listener'])
        )


@event_class('WebAudio.audioListenerWillBeDestroyed')
@dataclass
class AudioListenerWillBeDestroyed:
    '''
    Notifies that a new AudioListener has been created.
    '''
    context_id: GraphObjectId
    listener_id: GraphObjectId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AudioListenerWillBeDestroyed:
        return cls(
            context_id=GraphObjectId.from_json(json['contextId']),
            listener_id=GraphObjectId.from_json(json['listenerId'])
        )


@event_class('WebAudio.audioNodeCreated')
@dataclass
class AudioNodeCreated:
    '''
    Notifies that a new AudioNode has been created.
    '''
    node: AudioNode

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AudioNodeCreated:
        return cls(
            node=AudioNode.from_json(json['node'])
        )


@event_class('WebAudio.audioNodeWillBeDestroyed')
@dataclass
class AudioNodeWillBeDestroyed:
    '''
    Notifies that an existing AudioNode has been destroyed.
    '''
    context_id: GraphObjectId
    node_id: GraphObjectId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AudioNodeWillBeDestroyed:
        return cls(
            context_id=GraphObjectId.from_json(json['contextId']),
            node_id=GraphObjectId.from_json(json['nodeId'])
        )


@event_class('WebAudio.audioParamCreated')
@dataclass
class AudioParamCreated:
    '''
    Notifies that a new AudioParam has been created.
    '''
    param: AudioParam

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AudioParamCreated:
        return cls(
            param=AudioParam.from_json(json['param'])
        )


@event_class('WebAudio.audioParamWillBeDestroyed')
@dataclass
class AudioParamWillBeDestroyed:
    '''
    Notifies that an existing AudioParam has been destroyed.
    '''
    context_id: GraphObjectId
    node_id: GraphObjectId
    param_id: GraphObjectId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AudioParamWillBeDestroyed:
        return cls(
            context_id=GraphObjectId.from_json(json['contextId']),
            node_id=GraphObjectId.from_json(json['nodeId']),
            param_id=GraphObjectId.from_json(json['paramId'])
        )


@event_class('WebAudio.nodesConnected')
@dataclass
class NodesConnected:
    '''
    Notifies that two AudioNodes are connected.
    '''
    context_id: GraphObjectId
    source_id: GraphObjectId
    destination_id: GraphObjectId
    source_output_index: typing.Optional[float]
    destination_input_index: typing.Optional[float]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NodesConnected:
        return cls(
            context_id=GraphObjectId.from_json(json['contextId']),
            source_id=GraphObjectId.from_json(json['sourceId']),
            destination_id=GraphObjectId.from_json(json['destinationId']),
            source_output_index=float(json['sourceOutputIndex']) if 'sourceOutputIndex' in json else None,
            destination_input_index=float(json['destinationInputIndex']) if 'destinationInputIndex' in json else None
        )


@event_class('WebAudio.nodesDisconnected')
@dataclass
class NodesDisconnected:
    '''
    Notifies that AudioNodes are disconnected. The destination can be null, and it means all the outgoing connections from the source are disconnected.
    '''
    context_id: GraphObjectId
    source_id: GraphObjectId
    destination_id: GraphObjectId
    source_output_index: typing.Optional[float]
    destination_input_index: typing.Optional[float]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NodesDisconnected:
        return cls(
            context_id=GraphObjectId.from_json(json['contextId']),
            source_id=GraphObjectId.from_json(json['sourceId']),
            destination_id=GraphObjectId.from_json(json['destinationId']),
            source_output_index=float(json['sourceOutputIndex']) if 'sourceOutputIndex' in json else None,
            destination_input_index=float(json['destinationInputIndex']) if 'destinationInputIndex' in json else None
        )


@event_class('WebAudio.nodeParamConnected')
@dataclass
class NodeParamConnected:
    '''
    Notifies that an AudioNode is connected to an AudioParam.
    '''
    context_id: GraphObjectId
    source_id: GraphObjectId
    destination_id: GraphObjectId
    source_output_index: typing.Optional[float]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NodeParamConnected:
        return cls(
            context_id=GraphObjectId.from_json(json['contextId']),
            source_id=GraphObjectId.from_json(json['sourceId']),
            destination_id=GraphObjectId.from_json(json['destinationId']),
            source_output_index=float(json['sourceOutputIndex']) if 'sourceOutputIndex' in json else None
        )


@event_class('WebAudio.nodeParamDisconnected')
@dataclass
class NodeParamDisconnected:
    '''
    Notifies that an AudioNode is disconnected to an AudioParam.
    '''
    context_id: GraphObjectId
    source_id: GraphObjectId
    destination_id: GraphObjectId
    source_output_index: typing.Optional[float]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NodeParamDisconnected:
        return cls(
            context_id=GraphObjectId.from_json(json['contextId']),
            source_id=GraphObjectId.from_json(json['sourceId']),
            destination_id=GraphObjectId.from_json(json['destinationId']),
            source_output_index=float(json['sourceOutputIndex']) if 'sourceOutputIndex' in json else None
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\web_audio.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: WebAudio (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

class GraphObjectId(str):
    '''
    An unique ID for a graph object (AudioContext, AudioNode, AudioParam) in Web Audio API
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> GraphObjectId:
        return cls(json)

    def __repr__(self):
        return 'GraphObjectId({})'.format(super().__repr__())


class ContextType(enum.Enum):
    '''
    Enum of BaseAudioContext types
    '''
    REALTIME = "realtime"
    OFFLINE = "offline"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class ContextState(enum.Enum):
    '''
    Enum of AudioContextState from the spec
    '''
    SUSPENDED = "suspended"
    RUNNING = "running"
    CLOSED = "closed"
    INTERRUPTED = "interrupted"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class NodeType(str):
    '''
    Enum of AudioNode types
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> NodeType:
        return cls(json)

    def __repr__(self):
        return 'NodeType({})'.format(super().__repr__())


class ChannelCountMode(enum.Enum):
    '''
    Enum of AudioNode::ChannelCountMode from the spec
    '''
    CLAMPED_MAX = "clamped-max"
    EXPLICIT = "explicit"
    MAX_ = "max"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class ChannelInterpretation(enum.Enum):
    '''
    Enum of AudioNode::ChannelInterpretation from the spec
    '''
    DISCRETE = "discrete"
    SPEAKERS = "speakers"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class ParamType(str):
    '''
    Enum of AudioParam types
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> ParamType:
        return cls(json)

    def __repr__(self):
        return 'ParamType({})'.format(super().__repr__())


class AutomationRate(enum.Enum):
    '''
    Enum of AudioParam::AutomationRate from the spec
    '''
    A_RATE = "a-rate"
    K_RATE = "k-rate"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class ContextRealtimeData:
    '''
    Fields in AudioContext that change in real-time.
    '''
    #: The current context time in second in BaseAudioContext.
    current_time: float

    #: The time spent on rendering graph divided by render quantum duration,
    #: and multiplied by 100. 100 means the audio renderer reached the full
    #: capacity and glitch may occur.
    render_capacity: float

    #: A running mean of callback interval.
    callback_interval_mean: float

    #: A running variance of callback interval.
    callback_interval_variance: float

    def to_json(self):
        json = dict()
        json['currentTime'] = self.current_time
        json['renderCapacity'] = self.render_capacity
        json['callbackIntervalMean'] = self.callback_interval_mean
        json['callbackIntervalVariance'] = self.callback_interval_variance
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            current_time=float(json['currentTime']),
            render_capacity=float(json['renderCapacity']),
            callback_interval_mean=float(json['callbackIntervalMean']),
            callback_interval_variance=float(json['callbackIntervalVariance']),
        )


@dataclass
class BaseAudioContext:
    '''
    Protocol object for BaseAudioContext
    '''
    context_id: GraphObjectId

    context_type: ContextType

    context_state: ContextState

    #: Platform-dependent callback buffer size.
    callback_buffer_size: float

    #: Number of output channels supported by audio hardware in use.
    max_output_channel_count: float

    #: Context sample rate.
    sample_rate: float

    realtime_data: typing.Optional[ContextRealtimeData] = None

    def to_json(self):
        json = dict()
        json['contextId'] = self.context_id.to_json()
        json['contextType'] = self.context_type.to_json()
        json['contextState'] = self.context_state.to_json()
        json['callbackBufferSize'] = self.callback_buffer_size
        json['maxOutputChannelCount'] = self.max_output_channel_count
        json['sampleRate'] = self.sample_rate
        if self.realtime_data is not None:
            json['realtimeData'] = self.realtime_data.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            context_id=GraphObjectId.from_json(json['contextId']),
            context_type=ContextType.from_json(json['contextType']),
            context_state=ContextState.from_json(json['contextState']),
            callback_buffer_size=float(json['callbackBufferSize']),
            max_output_channel_count=float(json['maxOutputChannelCount']),
            sample_rate=float(json['sampleRate']),
            realtime_data=ContextRealtimeData.from_json(json['realtimeData']) if 'realtimeData' in json else None,
        )


@dataclass
class AudioListener:
    '''
    Protocol object for AudioListener
    '''
    listener_id: GraphObjectId

    context_id: GraphObjectId

    def to_json(self):
        json = dict()
        json['listenerId'] = self.listener_id.to_json()
        json['contextId'] = self.context_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            listener_id=GraphObjectId.from_json(json['listenerId']),
            context_id=GraphObjectId.from_json(json['contextId']),
        )


@dataclass
class AudioNode:
    '''
    Protocol object for AudioNode
    '''
    node_id: GraphObjectId

    context_id: GraphObjectId

    node_type: NodeType

    number_of_inputs: float

    number_of_outputs: float

    channel_count: float

    channel_count_mode: ChannelCountMode

    channel_interpretation: ChannelInterpretation

    def to_json(self):
        json = dict()
        json['nodeId'] = self.node_id.to_json()
        json['contextId'] = self.context_id.to_json()
        json['nodeType'] = self.node_type.to_json()
        json['numberOfInputs'] = self.number_of_inputs
        json['numberOfOutputs'] = self.number_of_outputs
        json['channelCount'] = self.channel_count
        json['channelCountMode'] = self.channel_count_mode.to_json()
        json['channelInterpretation'] = self.channel_interpretation.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            node_id=GraphObjectId.from_json(json['nodeId']),
            context_id=GraphObjectId.from_json(json['contextId']),
            node_type=NodeType.from_json(json['nodeType']),
            number_of_inputs=float(json['numberOfInputs']),
            number_of_outputs=float(json['numberOfOutputs']),
            channel_count=float(json['channelCount']),
            channel_count_mode=ChannelCountMode.from_json(json['channelCountMode']),
            channel_interpretation=ChannelInterpretation.from_json(json['channelInterpretation']),
        )


@dataclass
class AudioParam:
    '''
    Protocol object for AudioParam
    '''
    param_id: GraphObjectId

    node_id: GraphObjectId

    context_id: GraphObjectId

    param_type: ParamType

    rate: AutomationRate

    default_value: float

    min_value: float

    max_value: float

    def to_json(self):
        json = dict()
        json['paramId'] = self.param_id.to_json()
        json['nodeId'] = self.node_id.to_json()
        json['contextId'] = self.context_id.to_json()
        json['paramType'] = self.param_type.to_json()
        json['rate'] = self.rate.to_json()
        json['defaultValue'] = self.default_value
        json['minValue'] = self.min_value
        json['maxValue'] = self.max_value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            param_id=GraphObjectId.from_json(json['paramId']),
            node_id=GraphObjectId.from_json(json['nodeId']),
            context_id=GraphObjectId.from_json(json['contextId']),
            param_type=ParamType.from_json(json['paramType']),
            rate=AutomationRate.from_json(json['rate']),
            default_value=float(json['defaultValue']),
            min_value=float(json['minValue']),
            max_value=float(json['maxValue']),
        )


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables the WebAudio domain and starts sending context lifetime events.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAudio.enable',
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables the WebAudio domain.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAudio.disable',
    }
    json = yield cmd_dict


def get_realtime_data(
        context_id: GraphObjectId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,ContextRealtimeData]:
    '''
    Fetch the realtime data from the registered contexts.

    :param context_id:
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['contextId'] = context_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAudio.getRealtimeData',
        'params': params,
    }
    json = yield cmd_dict
    return ContextRealtimeData.from_json(json['realtimeData'])


@event_class('WebAudio.contextCreated')
@dataclass
class ContextCreated:
    '''
    Notifies that a new BaseAudioContext has been created.
    '''
    context: BaseAudioContext

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ContextCreated:
        return cls(
            context=BaseAudioContext.from_json(json['context'])
        )


@event_class('WebAudio.contextWillBeDestroyed')
@dataclass
class ContextWillBeDestroyed:
    '''
    Notifies that an existing BaseAudioContext will be destroyed.
    '''
    context_id: GraphObjectId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ContextWillBeDestroyed:
        return cls(
            context_id=GraphObjectId.from_json(json['contextId'])
        )


@event_class('WebAudio.contextChanged')
@dataclass
class ContextChanged:
    '''
    Notifies that existing BaseAudioContext has changed some properties (id stays the same)..
    '''
    context: BaseAudioContext

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ContextChanged:
        return cls(
            context=BaseAudioContext.from_json(json['context'])
        )


@event_class('WebAudio.audioListenerCreated')
@dataclass
class AudioListenerCreated:
    '''
    Notifies that the construction of an AudioListener has finished.
    '''
    listener: AudioListener

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AudioListenerCreated:
        return cls(
            listener=AudioListener.from_json(json['listener'])
        )


@event_class('WebAudio.audioListenerWillBeDestroyed')
@dataclass
class AudioListenerWillBeDestroyed:
    '''
    Notifies that a new AudioListener has been created.
    '''
    context_id: GraphObjectId
    listener_id: GraphObjectId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AudioListenerWillBeDestroyed:
        return cls(
            context_id=GraphObjectId.from_json(json['contextId']),
            listener_id=GraphObjectId.from_json(json['listenerId'])
        )


@event_class('WebAudio.audioNodeCreated')
@dataclass
class AudioNodeCreated:
    '''
    Notifies that a new AudioNode has been created.
    '''
    node: AudioNode

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AudioNodeCreated:
        return cls(
            node=AudioNode.from_json(json['node'])
        )


@event_class('WebAudio.audioNodeWillBeDestroyed')
@dataclass
class AudioNodeWillBeDestroyed:
    '''
    Notifies that an existing AudioNode has been destroyed.
    '''
    context_id: GraphObjectId
    node_id: GraphObjectId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AudioNodeWillBeDestroyed:
        return cls(
            context_id=GraphObjectId.from_json(json['contextId']),
            node_id=GraphObjectId.from_json(json['nodeId'])
        )


@event_class('WebAudio.audioParamCreated')
@dataclass
class AudioParamCreated:
    '''
    Notifies that a new AudioParam has been created.
    '''
    param: AudioParam

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AudioParamCreated:
        return cls(
            param=AudioParam.from_json(json['param'])
        )


@event_class('WebAudio.audioParamWillBeDestroyed')
@dataclass
class AudioParamWillBeDestroyed:
    '''
    Notifies that an existing AudioParam has been destroyed.
    '''
    context_id: GraphObjectId
    node_id: GraphObjectId
    param_id: GraphObjectId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AudioParamWillBeDestroyed:
        return cls(
            context_id=GraphObjectId.from_json(json['contextId']),
            node_id=GraphObjectId.from_json(json['nodeId']),
            param_id=GraphObjectId.from_json(json['paramId'])
        )


@event_class('WebAudio.nodesConnected')
@dataclass
class NodesConnected:
    '''
    Notifies that two AudioNodes are connected.
    '''
    context_id: GraphObjectId
    source_id: GraphObjectId
    destination_id: GraphObjectId
    source_output_index: typing.Optional[float]
    destination_input_index: typing.Optional[float]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NodesConnected:
        return cls(
            context_id=GraphObjectId.from_json(json['contextId']),
            source_id=GraphObjectId.from_json(json['sourceId']),
            destination_id=GraphObjectId.from_json(json['destinationId']),
            source_output_index=float(json['sourceOutputIndex']) if 'sourceOutputIndex' in json else None,
            destination_input_index=float(json['destinationInputIndex']) if 'destinationInputIndex' in json else None
        )


@event_class('WebAudio.nodesDisconnected')
@dataclass
class NodesDisconnected:
    '''
    Notifies that AudioNodes are disconnected. The destination can be null, and it means all the outgoing connections from the source are disconnected.
    '''
    context_id: GraphObjectId
    source_id: GraphObjectId
    destination_id: GraphObjectId
    source_output_index: typing.Optional[float]
    destination_input_index: typing.Optional[float]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NodesDisconnected:
        return cls(
            context_id=GraphObjectId.from_json(json['contextId']),
            source_id=GraphObjectId.from_json(json['sourceId']),
            destination_id=GraphObjectId.from_json(json['destinationId']),
            source_output_index=float(json['sourceOutputIndex']) if 'sourceOutputIndex' in json else None,
            destination_input_index=float(json['destinationInputIndex']) if 'destinationInputIndex' in json else None
        )


@event_class('WebAudio.nodeParamConnected')
@dataclass
class NodeParamConnected:
    '''
    Notifies that an AudioNode is connected to an AudioParam.
    '''
    context_id: GraphObjectId
    source_id: GraphObjectId
    destination_id: GraphObjectId
    source_output_index: typing.Optional[float]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NodeParamConnected:
        return cls(
            context_id=GraphObjectId.from_json(json['contextId']),
            source_id=GraphObjectId.from_json(json['sourceId']),
            destination_id=GraphObjectId.from_json(json['destinationId']),
            source_output_index=float(json['sourceOutputIndex']) if 'sourceOutputIndex' in json else None
        )


@event_class('WebAudio.nodeParamDisconnected')
@dataclass
class NodeParamDisconnected:
    '''
    Notifies that an AudioNode is disconnected to an AudioParam.
    '''
    context_id: GraphObjectId
    source_id: GraphObjectId
    destination_id: GraphObjectId
    source_output_index: typing.Optional[float]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NodeParamDisconnected:
        return cls(
            context_id=GraphObjectId.from_json(json['contextId']),
            source_id=GraphObjectId.from_json(json['sourceId']),
            destination_id=GraphObjectId.from_json(json['destinationId']),
            source_output_index=float(json['sourceOutputIndex']) if 'sourceOutputIndex' in json else None
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\web_audio.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: WebAudio (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

class GraphObjectId(str):
    '''
    An unique ID for a graph object (AudioContext, AudioNode, AudioParam) in Web Audio API
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> GraphObjectId:
        return cls(json)

    def __repr__(self):
        return 'GraphObjectId({})'.format(super().__repr__())


class ContextType(enum.Enum):
    '''
    Enum of BaseAudioContext types
    '''
    REALTIME = "realtime"
    OFFLINE = "offline"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class ContextState(enum.Enum):
    '''
    Enum of AudioContextState from the spec
    '''
    SUSPENDED = "suspended"
    RUNNING = "running"
    CLOSED = "closed"
    INTERRUPTED = "interrupted"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class NodeType(str):
    '''
    Enum of AudioNode types
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> NodeType:
        return cls(json)

    def __repr__(self):
        return 'NodeType({})'.format(super().__repr__())


class ChannelCountMode(enum.Enum):
    '''
    Enum of AudioNode::ChannelCountMode from the spec
    '''
    CLAMPED_MAX = "clamped-max"
    EXPLICIT = "explicit"
    MAX_ = "max"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class ChannelInterpretation(enum.Enum):
    '''
    Enum of AudioNode::ChannelInterpretation from the spec
    '''
    DISCRETE = "discrete"
    SPEAKERS = "speakers"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class ParamType(str):
    '''
    Enum of AudioParam types
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> ParamType:
        return cls(json)

    def __repr__(self):
        return 'ParamType({})'.format(super().__repr__())


class AutomationRate(enum.Enum):
    '''
    Enum of AudioParam::AutomationRate from the spec
    '''
    A_RATE = "a-rate"
    K_RATE = "k-rate"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class ContextRealtimeData:
    '''
    Fields in AudioContext that change in real-time.
    '''
    #: The current context time in second in BaseAudioContext.
    current_time: float

    #: The time spent on rendering graph divided by render quantum duration,
    #: and multiplied by 100. 100 means the audio renderer reached the full
    #: capacity and glitch may occur.
    render_capacity: float

    #: A running mean of callback interval.
    callback_interval_mean: float

    #: A running variance of callback interval.
    callback_interval_variance: float

    def to_json(self):
        json = dict()
        json['currentTime'] = self.current_time
        json['renderCapacity'] = self.render_capacity
        json['callbackIntervalMean'] = self.callback_interval_mean
        json['callbackIntervalVariance'] = self.callback_interval_variance
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            current_time=float(json['currentTime']),
            render_capacity=float(json['renderCapacity']),
            callback_interval_mean=float(json['callbackIntervalMean']),
            callback_interval_variance=float(json['callbackIntervalVariance']),
        )


@dataclass
class BaseAudioContext:
    '''
    Protocol object for BaseAudioContext
    '''
    context_id: GraphObjectId

    context_type: ContextType

    context_state: ContextState

    #: Platform-dependent callback buffer size.
    callback_buffer_size: float

    #: Number of output channels supported by audio hardware in use.
    max_output_channel_count: float

    #: Context sample rate.
    sample_rate: float

    realtime_data: typing.Optional[ContextRealtimeData] = None

    def to_json(self):
        json = dict()
        json['contextId'] = self.context_id.to_json()
        json['contextType'] = self.context_type.to_json()
        json['contextState'] = self.context_state.to_json()
        json['callbackBufferSize'] = self.callback_buffer_size
        json['maxOutputChannelCount'] = self.max_output_channel_count
        json['sampleRate'] = self.sample_rate
        if self.realtime_data is not None:
            json['realtimeData'] = self.realtime_data.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            context_id=GraphObjectId.from_json(json['contextId']),
            context_type=ContextType.from_json(json['contextType']),
            context_state=ContextState.from_json(json['contextState']),
            callback_buffer_size=float(json['callbackBufferSize']),
            max_output_channel_count=float(json['maxOutputChannelCount']),
            sample_rate=float(json['sampleRate']),
            realtime_data=ContextRealtimeData.from_json(json['realtimeData']) if 'realtimeData' in json else None,
        )


@dataclass
class AudioListener:
    '''
    Protocol object for AudioListener
    '''
    listener_id: GraphObjectId

    context_id: GraphObjectId

    def to_json(self):
        json = dict()
        json['listenerId'] = self.listener_id.to_json()
        json['contextId'] = self.context_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            listener_id=GraphObjectId.from_json(json['listenerId']),
            context_id=GraphObjectId.from_json(json['contextId']),
        )


@dataclass
class AudioNode:
    '''
    Protocol object for AudioNode
    '''
    node_id: GraphObjectId

    context_id: GraphObjectId

    node_type: NodeType

    number_of_inputs: float

    number_of_outputs: float

    channel_count: float

    channel_count_mode: ChannelCountMode

    channel_interpretation: ChannelInterpretation

    def to_json(self):
        json = dict()
        json['nodeId'] = self.node_id.to_json()
        json['contextId'] = self.context_id.to_json()
        json['nodeType'] = self.node_type.to_json()
        json['numberOfInputs'] = self.number_of_inputs
        json['numberOfOutputs'] = self.number_of_outputs
        json['channelCount'] = self.channel_count
        json['channelCountMode'] = self.channel_count_mode.to_json()
        json['channelInterpretation'] = self.channel_interpretation.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            node_id=GraphObjectId.from_json(json['nodeId']),
            context_id=GraphObjectId.from_json(json['contextId']),
            node_type=NodeType.from_json(json['nodeType']),
            number_of_inputs=float(json['numberOfInputs']),
            number_of_outputs=float(json['numberOfOutputs']),
            channel_count=float(json['channelCount']),
            channel_count_mode=ChannelCountMode.from_json(json['channelCountMode']),
            channel_interpretation=ChannelInterpretation.from_json(json['channelInterpretation']),
        )


@dataclass
class AudioParam:
    '''
    Protocol object for AudioParam
    '''
    param_id: GraphObjectId

    node_id: GraphObjectId

    context_id: GraphObjectId

    param_type: ParamType

    rate: AutomationRate

    default_value: float

    min_value: float

    max_value: float

    def to_json(self):
        json = dict()
        json['paramId'] = self.param_id.to_json()
        json['nodeId'] = self.node_id.to_json()
        json['contextId'] = self.context_id.to_json()
        json['paramType'] = self.param_type.to_json()
        json['rate'] = self.rate.to_json()
        json['defaultValue'] = self.default_value
        json['minValue'] = self.min_value
        json['maxValue'] = self.max_value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            param_id=GraphObjectId.from_json(json['paramId']),
            node_id=GraphObjectId.from_json(json['nodeId']),
            context_id=GraphObjectId.from_json(json['contextId']),
            param_type=ParamType.from_json(json['paramType']),
            rate=AutomationRate.from_json(json['rate']),
            default_value=float(json['defaultValue']),
            min_value=float(json['minValue']),
            max_value=float(json['maxValue']),
        )


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables the WebAudio domain and starts sending context lifetime events.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAudio.enable',
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables the WebAudio domain.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAudio.disable',
    }
    json = yield cmd_dict


def get_realtime_data(
        context_id: GraphObjectId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,ContextRealtimeData]:
    '''
    Fetch the realtime data from the registered contexts.

    :param context_id:
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['contextId'] = context_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAudio.getRealtimeData',
        'params': params,
    }
    json = yield cmd_dict
    return ContextRealtimeData.from_json(json['realtimeData'])


@event_class('WebAudio.contextCreated')
@dataclass
class ContextCreated:
    '''
    Notifies that a new BaseAudioContext has been created.
    '''
    context: BaseAudioContext

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ContextCreated:
        return cls(
            context=BaseAudioContext.from_json(json['context'])
        )


@event_class('WebAudio.contextWillBeDestroyed')
@dataclass
class ContextWillBeDestroyed:
    '''
    Notifies that an existing BaseAudioContext will be destroyed.
    '''
    context_id: GraphObjectId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ContextWillBeDestroyed:
        return cls(
            context_id=GraphObjectId.from_json(json['contextId'])
        )


@event_class('WebAudio.contextChanged')
@dataclass
class ContextChanged:
    '''
    Notifies that existing BaseAudioContext has changed some properties (id stays the same)..
    '''
    context: BaseAudioContext

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ContextChanged:
        return cls(
            context=BaseAudioContext.from_json(json['context'])
        )


@event_class('WebAudio.audioListenerCreated')
@dataclass
class AudioListenerCreated:
    '''
    Notifies that the construction of an AudioListener has finished.
    '''
    listener: AudioListener

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AudioListenerCreated:
        return cls(
            listener=AudioListener.from_json(json['listener'])
        )


@event_class('WebAudio.audioListenerWillBeDestroyed')
@dataclass
class AudioListenerWillBeDestroyed:
    '''
    Notifies that a new AudioListener has been created.
    '''
    context_id: GraphObjectId
    listener_id: GraphObjectId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AudioListenerWillBeDestroyed:
        return cls(
            context_id=GraphObjectId.from_json(json['contextId']),
            listener_id=GraphObjectId.from_json(json['listenerId'])
        )


@event_class('WebAudio.audioNodeCreated')
@dataclass
class AudioNodeCreated:
    '''
    Notifies that a new AudioNode has been created.
    '''
    node: AudioNode

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AudioNodeCreated:
        return cls(
            node=AudioNode.from_json(json['node'])
        )


@event_class('WebAudio.audioNodeWillBeDestroyed')
@dataclass
class AudioNodeWillBeDestroyed:
    '''
    Notifies that an existing AudioNode has been destroyed.
    '''
    context_id: GraphObjectId
    node_id: GraphObjectId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AudioNodeWillBeDestroyed:
        return cls(
            context_id=GraphObjectId.from_json(json['contextId']),
            node_id=GraphObjectId.from_json(json['nodeId'])
        )


@event_class('WebAudio.audioParamCreated')
@dataclass
class AudioParamCreated:
    '''
    Notifies that a new AudioParam has been created.
    '''
    param: AudioParam

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AudioParamCreated:
        return cls(
            param=AudioParam.from_json(json['param'])
        )


@event_class('WebAudio.audioParamWillBeDestroyed')
@dataclass
class AudioParamWillBeDestroyed:
    '''
    Notifies that an existing AudioParam has been destroyed.
    '''
    context_id: GraphObjectId
    node_id: GraphObjectId
    param_id: GraphObjectId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AudioParamWillBeDestroyed:
        return cls(
            context_id=GraphObjectId.from_json(json['contextId']),
            node_id=GraphObjectId.from_json(json['nodeId']),
            param_id=GraphObjectId.from_json(json['paramId'])
        )


@event_class('WebAudio.nodesConnected')
@dataclass
class NodesConnected:
    '''
    Notifies that two AudioNodes are connected.
    '''
    context_id: GraphObjectId
    source_id: GraphObjectId
    destination_id: GraphObjectId
    source_output_index: typing.Optional[float]
    destination_input_index: typing.Optional[float]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NodesConnected:
        return cls(
            context_id=GraphObjectId.from_json(json['contextId']),
            source_id=GraphObjectId.from_json(json['sourceId']),
            destination_id=GraphObjectId.from_json(json['destinationId']),
            source_output_index=float(json['sourceOutputIndex']) if 'sourceOutputIndex' in json else None,
            destination_input_index=float(json['destinationInputIndex']) if 'destinationInputIndex' in json else None
        )


@event_class('WebAudio.nodesDisconnected')
@dataclass
class NodesDisconnected:
    '''
    Notifies that AudioNodes are disconnected. The destination can be null, and it means all the outgoing connections from the source are disconnected.
    '''
    context_id: GraphObjectId
    source_id: GraphObjectId
    destination_id: GraphObjectId
    source_output_index: typing.Optional[float]
    destination_input_index: typing.Optional[float]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NodesDisconnected:
        return cls(
            context_id=GraphObjectId.from_json(json['contextId']),
            source_id=GraphObjectId.from_json(json['sourceId']),
            destination_id=GraphObjectId.from_json(json['destinationId']),
            source_output_index=float(json['sourceOutputIndex']) if 'sourceOutputIndex' in json else None,
            destination_input_index=float(json['destinationInputIndex']) if 'destinationInputIndex' in json else None
        )


@event_class('WebAudio.nodeParamConnected')
@dataclass
class NodeParamConnected:
    '''
    Notifies that an AudioNode is connected to an AudioParam.
    '''
    context_id: GraphObjectId
    source_id: GraphObjectId
    destination_id: GraphObjectId
    source_output_index: typing.Optional[float]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NodeParamConnected:
        return cls(
            context_id=GraphObjectId.from_json(json['contextId']),
            source_id=GraphObjectId.from_json(json['sourceId']),
            destination_id=GraphObjectId.from_json(json['destinationId']),
            source_output_index=float(json['sourceOutputIndex']) if 'sourceOutputIndex' in json else None
        )


@event_class('WebAudio.nodeParamDisconnected')
@dataclass
class NodeParamDisconnected:
    '''
    Notifies that an AudioNode is disconnected to an AudioParam.
    '''
    context_id: GraphObjectId
    source_id: GraphObjectId
    destination_id: GraphObjectId
    source_output_index: typing.Optional[float]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NodeParamDisconnected:
        return cls(
            context_id=GraphObjectId.from_json(json['contextId']),
            source_id=GraphObjectId.from_json(json['sourceId']),
            destination_id=GraphObjectId.from_json(json['destinationId']),
            source_output_index=float(json['sourceOutputIndex']) if 'sourceOutputIndex' in json else None
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\ctypeslib\_ctypeslib.py ===
"""
============================
``ctypes`` Utility Functions
============================

See Also
--------
load_library : Load a C library.
ndpointer : Array restype/argtype with verification.
as_ctypes : Create a ctypes array from an ndarray.
as_array : Create an ndarray from a ctypes array.

References
----------
.. [1] "SciPy Cookbook: ctypes", https://scipy-cookbook.readthedocs.io/items/Ctypes.html

Examples
--------
Load the C library:

>>> _lib = np.ctypeslib.load_library('libmystuff', '.')     #doctest: +SKIP

Our result type, an ndarray that must be of type double, be 1-dimensional
and is C-contiguous in memory:

>>> array_1d_double = np.ctypeslib.ndpointer(
...                          dtype=np.double,
...                          ndim=1, flags='CONTIGUOUS')    #doctest: +SKIP

Our C-function typically takes an array and updates its values
in-place.  For example::

    void foo_func(double* x, int length)
    {
        int i;
        for (i = 0; i < length; i++) {
            x[i] = i*i;
        }
    }

We wrap it using:

>>> _lib.foo_func.restype = None                      #doctest: +SKIP
>>> _lib.foo_func.argtypes = [array_1d_double, c_int] #doctest: +SKIP

Then, we're ready to call ``foo_func``:

>>> out = np.empty(15, dtype=np.double)
>>> _lib.foo_func(out, len(out))                #doctest: +SKIP

"""
__all__ = ['load_library', 'ndpointer', 'c_intp', 'as_ctypes', 'as_array',
           'as_ctypes_type']

import os

import numpy as np
import numpy._core.multiarray as mu
from numpy._utils import set_module

try:
    import ctypes
except ImportError:
    ctypes = None

if ctypes is None:
    @set_module("numpy.ctypeslib")
    def _dummy(*args, **kwds):
        """
        Dummy object that raises an ImportError if ctypes is not available.

        Raises
        ------
        ImportError
            If ctypes is not available.

        """
        raise ImportError("ctypes is not available.")
    load_library = _dummy
    as_ctypes = _dummy
    as_ctypes_type = _dummy
    as_array = _dummy
    ndpointer = _dummy
    from numpy import intp as c_intp
    _ndptr_base = object
else:
    import numpy._core._internal as nic
    c_intp = nic._getintp_ctype()
    del nic
    _ndptr_base = ctypes.c_void_p

    # Adapted from Albert Strasheim
    @set_module("numpy.ctypeslib")
    def load_library(libname, loader_path):
        """
        It is possible to load a library using

        >>> lib = ctypes.cdll[<full_path_name>] # doctest: +SKIP

        But there are cross-platform considerations, such as library file extensions,
        plus the fact Windows will just load the first library it finds with that name.
        NumPy supplies the load_library function as a convenience.

        .. versionchanged:: 1.20.0
            Allow libname and loader_path to take any
            :term:`python:path-like object`.

        Parameters
        ----------
        libname : path-like
            Name of the library, which can have 'lib' as a prefix,
            but without an extension.
        loader_path : path-like
            Where the library can be found.

        Returns
        -------
        ctypes.cdll[libpath] : library object
           A ctypes library object

        Raises
        ------
        OSError
            If there is no library with the expected extension, or the
            library is defective and cannot be loaded.
        """
        # Convert path-like objects into strings
        libname = os.fsdecode(libname)
        loader_path = os.fsdecode(loader_path)

        ext = os.path.splitext(libname)[1]
        if not ext:
            import sys
            import sysconfig
            # Try to load library with platform-specific name, otherwise
            # default to libname.[so|dll|dylib].  Sometimes, these files are
            # built erroneously on non-linux platforms.
            base_ext = ".so"
            if sys.platform.startswith("darwin"):
                base_ext = ".dylib"
            elif sys.platform.startswith("win"):
                base_ext = ".dll"
            libname_ext = [libname + base_ext]
            so_ext = sysconfig.get_config_var("EXT_SUFFIX")
            if not so_ext == base_ext:
                libname_ext.insert(0, libname + so_ext)
        else:
            libname_ext = [libname]

        loader_path = os.path.abspath(loader_path)
        if not os.path.isdir(loader_path):
            libdir = os.path.dirname(loader_path)
        else:
            libdir = loader_path

        for ln in libname_ext:
            libpath = os.path.join(libdir, ln)
            if os.path.exists(libpath):
                try:
                    return ctypes.cdll[libpath]
                except OSError:
                    # defective lib file
                    raise
        # if no successful return in the libname_ext loop:
        raise OSError("no file with expected extension")


def _num_fromflags(flaglist):
    num = 0
    for val in flaglist:
        num += mu._flagdict[val]
    return num


_flagnames = ['C_CONTIGUOUS', 'F_CONTIGUOUS', 'ALIGNED', 'WRITEABLE',
              'OWNDATA', 'WRITEBACKIFCOPY']
def _flags_fromnum(num):
    res = []
    for key in _flagnames:
        value = mu._flagdict[key]
        if (num & value):
            res.append(key)
    return res


class _ndptr(_ndptr_base):
    @classmethod
    def from_param(cls, obj):
        if not isinstance(obj, np.ndarray):
            raise TypeError("argument must be an ndarray")
        if cls._dtype_ is not None \
               and obj.dtype != cls._dtype_:
            raise TypeError(f"array must have data type {cls._dtype_}")
        if cls._ndim_ is not None \
               and obj.ndim != cls._ndim_:
            raise TypeError("array must have %d dimension(s)" % cls._ndim_)
        if cls._shape_ is not None \
               and obj.shape != cls._shape_:
            raise TypeError(f"array must have shape {str(cls._shape_)}")
        if cls._flags_ is not None \
               and ((obj.flags.num & cls._flags_) != cls._flags_):
            raise TypeError(f"array must have flags {_flags_fromnum(cls._flags_)}")
        return obj.ctypes


class _concrete_ndptr(_ndptr):
    """
    Like _ndptr, but with `_shape_` and `_dtype_` specified.

    Notably, this means the pointer has enough information to reconstruct
    the array, which is not generally true.
    """
    def _check_retval_(self):
        """
        This method is called when this class is used as the .restype
        attribute for a shared-library function, to automatically wrap the
        pointer into an array.
        """
        return self.contents

    @property
    def contents(self):
        """
        Get an ndarray viewing the data pointed to by this pointer.

        This mirrors the `contents` attribute of a normal ctypes pointer
        """
        full_dtype = np.dtype((self._dtype_, self._shape_))
        full_ctype = ctypes.c_char * full_dtype.itemsize
        buffer = ctypes.cast(self, ctypes.POINTER(full_ctype)).contents
        return np.frombuffer(buffer, dtype=full_dtype).squeeze(axis=0)


# Factory for an array-checking class with from_param defined for
# use with ctypes argtypes mechanism
_pointer_type_cache = {}

@set_module("numpy.ctypeslib")
def ndpointer(dtype=None, ndim=None, shape=None, flags=None):
    """
    Array-checking restype/argtypes.

    An ndpointer instance is used to describe an ndarray in restypes
    and argtypes specifications.  This approach is more flexible than
    using, for example, ``POINTER(c_double)``, since several restrictions
    can be specified, which are verified upon calling the ctypes function.
    These include data type, number of dimensions, shape and flags.  If a
    given array does not satisfy the specified restrictions,
    a ``TypeError`` is raised.

    Parameters
    ----------
    dtype : data-type, optional
        Array data-type.
    ndim : int, optional
        Number of array dimensions.
    shape : tuple of ints, optional
        Array shape.
    flags : str or tuple of str
        Array flags; may be one or more of:

        - C_CONTIGUOUS / C / CONTIGUOUS
        - F_CONTIGUOUS / F / FORTRAN
        - OWNDATA / O
        - WRITEABLE / W
        - ALIGNED / A
        - WRITEBACKIFCOPY / X

    Returns
    -------
    klass : ndpointer type object
        A type object, which is an ``_ndtpr`` instance containing
        dtype, ndim, shape and flags information.

    Raises
    ------
    TypeError
        If a given array does not satisfy the specified restrictions.

    Examples
    --------
    >>> clib.somefunc.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64,
    ...                                                  ndim=1,
    ...                                                  flags='C_CONTIGUOUS')]
    ... #doctest: +SKIP
    >>> clib.somefunc(np.array([1, 2, 3], dtype=np.float64))
    ... #doctest: +SKIP

    """

    # normalize dtype to dtype | None
    if dtype is not None:
        dtype = np.dtype(dtype)

    # normalize flags to int | None
    num = None
    if flags is not None:
        if isinstance(flags, str):
            flags = flags.split(',')
        elif isinstance(flags, (int, np.integer)):
            num = flags
            flags = _flags_fromnum(num)
        elif isinstance(flags, mu.flagsobj):
            num = flags.num
            flags = _flags_fromnum(num)
        if num is None:
            try:
                flags = [x.strip().upper() for x in flags]
            except Exception as e:
                raise TypeError("invalid flags specification") from e
            num = _num_fromflags(flags)

    # normalize shape to tuple | None
    if shape is not None:
        try:
            shape = tuple(shape)
        except TypeError:
            # single integer -> 1-tuple
            shape = (shape,)

    cache_key = (dtype, ndim, shape, num)

    try:
        return _pointer_type_cache[cache_key]
    except KeyError:
        pass

    # produce a name for the new type
    if dtype is None:
        name = 'any'
    elif dtype.names is not None:
        name = str(id(dtype))
    else:
        name = dtype.str
    if ndim is not None:
        name += "_%dd" % ndim
    if shape is not None:
        name += "_" + "x".join(str(x) for x in shape)
    if flags is not None:
        name += "_" + "_".join(flags)

    if dtype is not None and shape is not None:
        base = _concrete_ndptr
    else:
        base = _ndptr

    klass = type(f"ndpointer_{name}", (base,),
                 {"_dtype_": dtype,
                  "_shape_": shape,
                  "_ndim_": ndim,
                  "_flags_": num})
    _pointer_type_cache[cache_key] = klass
    return klass


if ctypes is not None:
    def _ctype_ndarray(element_type, shape):
        """ Create an ndarray of the given element type and shape """
        for dim in shape[::-1]:
            element_type = dim * element_type
            # prevent the type name include np.ctypeslib
            element_type.__module__ = None
        return element_type

    def _get_scalar_type_map():
        """
        Return a dictionary mapping native endian scalar dtype to ctypes types
        """
        ct = ctypes
        simple_types = [
            ct.c_byte, ct.c_short, ct.c_int, ct.c_long, ct.c_longlong,
            ct.c_ubyte, ct.c_ushort, ct.c_uint, ct.c_ulong, ct.c_ulonglong,
            ct.c_float, ct.c_double,
            ct.c_bool,
        ]
        return {np.dtype(ctype): ctype for ctype in simple_types}

    _scalar_type_map = _get_scalar_type_map()

    def _ctype_from_dtype_scalar(dtype):
        # swapping twice ensure that `=` is promoted to <, >, or |
        dtype_with_endian = dtype.newbyteorder('S').newbyteorder('S')
        dtype_native = dtype.newbyteorder('=')
        try:
            ctype = _scalar_type_map[dtype_native]
        except KeyError as e:
            raise NotImplementedError(
                f"Converting {dtype!r} to a ctypes type"
            ) from None

        if dtype_with_endian.byteorder == '>':
            ctype = ctype.__ctype_be__
        elif dtype_with_endian.byteorder == '<':
            ctype = ctype.__ctype_le__

        return ctype

    def _ctype_from_dtype_subarray(dtype):
        element_dtype, shape = dtype.subdtype
        ctype = _ctype_from_dtype(element_dtype)
        return _ctype_ndarray(ctype, shape)

    def _ctype_from_dtype_structured(dtype):
        # extract offsets of each field
        field_data = []
        for name in dtype.names:
            field_dtype, offset = dtype.fields[name][:2]
            field_data.append((offset, name, _ctype_from_dtype(field_dtype)))

        # ctypes doesn't care about field order
        field_data = sorted(field_data, key=lambda f: f[0])

        if len(field_data) > 1 and all(offset == 0 for offset, _, _ in field_data):
            # union, if multiple fields all at address 0
            size = 0
            _fields_ = []
            for offset, name, ctype in field_data:
                _fields_.append((name, ctype))
                size = max(size, ctypes.sizeof(ctype))

            # pad to the right size
            if dtype.itemsize != size:
                _fields_.append(('', ctypes.c_char * dtype.itemsize))

            # we inserted manual padding, so always `_pack_`
            return type('union', (ctypes.Union,), {
                '_fields_': _fields_,
                '_pack_': 1,
                '__module__': None,
            })
        else:
            last_offset = 0
            _fields_ = []
            for offset, name, ctype in field_data:
                padding = offset - last_offset
                if padding < 0:
                    raise NotImplementedError("Overlapping fields")
                if padding > 0:
                    _fields_.append(('', ctypes.c_char * padding))

                _fields_.append((name, ctype))
                last_offset = offset + ctypes.sizeof(ctype)

            padding = dtype.itemsize - last_offset
            if padding > 0:
                _fields_.append(('', ctypes.c_char * padding))

            # we inserted manual padding, so always `_pack_`
            return type('struct', (ctypes.Structure,), {
                '_fields_': _fields_,
                '_pack_': 1,
                '__module__': None,
            })

    def _ctype_from_dtype(dtype):
        if dtype.fields is not None:
            return _ctype_from_dtype_structured(dtype)
        elif dtype.subdtype is not None:
            return _ctype_from_dtype_subarray(dtype)
        else:
            return _ctype_from_dtype_scalar(dtype)

    @set_module("numpy.ctypeslib")
    def as_ctypes_type(dtype):
        r"""
        Convert a dtype into a ctypes type.

        Parameters
        ----------
        dtype : dtype
            The dtype to convert

        Returns
        -------
        ctype
            A ctype scalar, union, array, or struct

        Raises
        ------
        NotImplementedError
            If the conversion is not possible

        Notes
        -----
        This function does not losslessly round-trip in either direction.

        ``np.dtype(as_ctypes_type(dt))`` will:

        - insert padding fields
        - reorder fields to be sorted by offset
        - discard field titles

        ``as_ctypes_type(np.dtype(ctype))`` will:

        - discard the class names of `ctypes.Structure`\ s and
          `ctypes.Union`\ s
        - convert single-element `ctypes.Union`\ s into single-element
          `ctypes.Structure`\ s
        - insert padding fields

        Examples
        --------
        Converting a simple dtype:

        >>> dt = np.dtype('int8')
        >>> ctype = np.ctypeslib.as_ctypes_type(dt)
        >>> ctype
        <class 'ctypes.c_byte'>

        Converting a structured dtype:

        >>> dt = np.dtype([('x', 'i4'), ('y', 'f4')])
        >>> ctype = np.ctypeslib.as_ctypes_type(dt)
        >>> ctype
        <class 'struct'>

        """
        return _ctype_from_dtype(np.dtype(dtype))

    @set_module("numpy.ctypeslib")
    def as_array(obj, shape=None):
        """
        Create a numpy array from a ctypes array or POINTER.

        The numpy array shares the memory with the ctypes object.

        The shape parameter must be given if converting from a ctypes POINTER.
        The shape parameter is ignored if converting from a ctypes array

        Examples
        --------
        Converting a ctypes integer array:

        >>> import ctypes
        >>> ctypes_array = (ctypes.c_int * 5)(0, 1, 2, 3, 4)
        >>> np_array = np.ctypeslib.as_array(ctypes_array)
        >>> np_array
        array([0, 1, 2, 3, 4], dtype=int32)

        Converting a ctypes POINTER:

        >>> import ctypes
        >>> buffer = (ctypes.c_int * 5)(0, 1, 2, 3, 4)
        >>> pointer = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_int))
        >>> np_array = np.ctypeslib.as_array(pointer, (5,))
        >>> np_array
        array([0, 1, 2, 3, 4], dtype=int32)

        """
        if isinstance(obj, ctypes._Pointer):
            # convert pointers to an array of the desired shape
            if shape is None:
                raise TypeError(
                    'as_array() requires a shape argument when called on a '
                    'pointer')
            p_arr_type = ctypes.POINTER(_ctype_ndarray(obj._type_, shape))
            obj = ctypes.cast(obj, p_arr_type).contents

        return np.asarray(obj)

    @set_module("numpy.ctypeslib")
    def as_ctypes(obj):
        """
        Create and return a ctypes object from a numpy array.  Actually
        anything that exposes the __array_interface__ is accepted.

        Examples
        --------
        Create ctypes object from inferred int ``np.array``:

        >>> inferred_int_array = np.array([1, 2, 3])
        >>> c_int_array = np.ctypeslib.as_ctypes(inferred_int_array)
        >>> type(c_int_array)
        <class 'c_long_Array_3'>
        >>> c_int_array[:]
        [1, 2, 3]

        Create ctypes object from explicit 8 bit unsigned int ``np.array`` :

        >>> exp_int_array = np.array([1, 2, 3], dtype=np.uint8)
        >>> c_int_array = np.ctypeslib.as_ctypes(exp_int_array)
        >>> type(c_int_array)
        <class 'c_ubyte_Array_3'>
        >>> c_int_array[:]
        [1, 2, 3]

        """
        ai = obj.__array_interface__
        if ai["strides"]:
            raise TypeError("strided arrays not supported")
        if ai["version"] != 3:
            raise TypeError("only __array_interface__ version 3 supported")
        addr, readonly = ai["data"]
        if readonly:
            raise TypeError("readonly arrays unsupported")

        # can't use `_dtype((ai["typestr"], ai["shape"]))` here, as it overflows
        # dtype.itemsize (gh-14214)
        ctype_scalar = as_ctypes_type(ai["typestr"])
        result_type = _ctype_ndarray(ctype_scalar, ai["shape"])
        result = result_type.from_address(addr)
        result.__keep = obj
        return result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\concrete\expr_with_limits.py ===
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, UndefinedFunction
from sympy.core.mul import Mul
from sympy.core.relational import Equality, Relational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify
from sympy.functions.elementary.piecewise import (piecewise_fold,
    Piecewise)
from sympy.logic.boolalg import BooleanFunction
from sympy.matrices.matrixbase import MatrixBase
from sympy.sets.sets import Interval, Set
from sympy.sets.fancysets import Range
from sympy.tensor.indexed import Idx
from sympy.utilities import flatten
from sympy.utilities.iterables import sift, is_sequence
from sympy.utilities.exceptions import sympy_deprecation_warning


def _common_new(cls, function, *symbols, discrete, **assumptions):
    """Return either a special return value or the tuple,
    (function, limits, orientation). This code is common to
    both ExprWithLimits and AddWithLimits."""
    function = sympify(function)

    if isinstance(function, Equality):
        # This transforms e.g. Integral(Eq(x, y)) to Eq(Integral(x), Integral(y))
        # but that is only valid for definite integrals.
        limits, orientation = _process_limits(*symbols, discrete=discrete)
        if not (limits and all(len(limit) == 3 for limit in limits)):
            sympy_deprecation_warning(
                """
                Creating a indefinite integral with an Eq() argument is
                deprecated.

                This is because indefinite integrals do not preserve equality
                due to the arbitrary constants. If you want an equality of
                indefinite integrals, use Eq(Integral(a, x), Integral(b, x))
                explicitly.
                """,
                deprecated_since_version="1.6",
                active_deprecations_target="deprecated-indefinite-integral-eq",
                stacklevel=5,
            )

        lhs = function.lhs
        rhs = function.rhs
        return Equality(cls(lhs, *symbols, **assumptions), \
                        cls(rhs, *symbols, **assumptions))

    if function is S.NaN:
        return S.NaN

    if symbols:
        limits, orientation = _process_limits(*symbols, discrete=discrete)
        for i, li in enumerate(limits):
            if len(li) == 4:
                function = function.subs(li[0], li[-1])
                limits[i] = Tuple(*li[:-1])
    else:
        # symbol not provided -- we can still try to compute a general form
        free = function.free_symbols
        if len(free) != 1:
            raise ValueError(
                "specify dummy variables for %s" % function)
        limits, orientation = [Tuple(s) for s in free], 1

    # denest any nested calls
    while cls == type(function):
        limits = list(function.limits) + limits
        function = function.function

    # Any embedded piecewise functions need to be brought out to the
    # top level. We only fold Piecewise that contain the integration
    # variable.
    reps = {}
    symbols_of_integration = {i[0] for i in limits}
    for p in function.atoms(Piecewise):
        if not p.has(*symbols_of_integration):
            reps[p] = Dummy()
    # mask off those that don't
    function = function.xreplace(reps)
    # do the fold
    function = piecewise_fold(function)
    # remove the masking
    function = function.xreplace({v: k for k, v in reps.items()})

    return function, limits, orientation


def _process_limits(*symbols, discrete=None):
    """Process the list of symbols and convert them to canonical limits,
    storing them as Tuple(symbol, lower, upper). The orientation of
    the function is also returned when the upper limit is missing
    so (x, 1, None) becomes (x, None, 1) and the orientation is changed.
    In the case that a limit is specified as (symbol, Range), a list of
    length 4 may be returned if a change of variables is needed; the
    expression that should replace the symbol in the expression is
    the fourth element in the list.
    """
    limits = []
    orientation = 1
    if discrete is None:
        err_msg = 'discrete must be True or False'
    elif discrete:
        err_msg = 'use Range, not Interval or Relational'
    else:
        err_msg = 'use Interval or Relational, not Range'
    for V in symbols:
        if isinstance(V, (Relational, BooleanFunction)):
            if discrete:
                raise TypeError(err_msg)
            variable = V.atoms(Symbol).pop()
            V = (variable, V.as_set())
        elif isinstance(V, Symbol) or getattr(V, '_diff_wrt', False):
            if isinstance(V, Idx):
                if V.lower is None or V.upper is None:
                    limits.append(Tuple(V))
                else:
                    limits.append(Tuple(V, V.lower, V.upper))
            else:
                limits.append(Tuple(V))
            continue
        if is_sequence(V) and not isinstance(V, Set):
            if len(V) == 2 and isinstance(V[1], Set):
                V = list(V)
                if isinstance(V[1], Interval):  # includes Reals
                    if discrete:
                        raise TypeError(err_msg)
                    V[1:] = V[1].inf, V[1].sup
                elif isinstance(V[1], Range):
                    if not discrete:
                        raise TypeError(err_msg)
                    lo = V[1].inf
                    hi = V[1].sup
                    dx = abs(V[1].step)  # direction doesn't matter
                    if dx == 1:
                        V[1:] = [lo, hi]
                    else:
                        if lo is not S.NegativeInfinity:
                            V = [V[0]] + [0, (hi - lo)//dx, dx*V[0] + lo]
                        else:
                            V = [V[0]] + [0, S.Infinity, -dx*V[0] + hi]
                else:
                    # more complicated sets would require splitting, e.g.
                    # Union(Interval(1, 3), interval(6,10))
                    raise NotImplementedError(
                        'expecting Range' if discrete else
                        'Relational or single Interval' )
            V = sympify(flatten(V))  # list of sympified elements/None
            if isinstance(V[0], (Symbol, Idx)) or getattr(V[0], '_diff_wrt', False):
                newsymbol = V[0]
                if len(V) == 3:
                    # general case
                    if V[2] is None and V[1] is not None:
                        orientation *= -1
                    V = [newsymbol] + [i for i in V[1:] if i is not None]

                lenV = len(V)
                if not isinstance(newsymbol, Idx) or lenV == 3:
                    if lenV == 4:
                        limits.append(Tuple(*V))
                        continue
                    if lenV == 3:
                        if isinstance(newsymbol, Idx):
                            # Idx represents an integer which may have
                            # specified values it can take on; if it is
                            # given such a value, an error is raised here
                            # if the summation would try to give it a larger
                            # or smaller value than permitted. None and Symbolic
                            # values will not raise an error.
                            lo, hi = newsymbol.lower, newsymbol.upper
                            try:
                                if lo is not None and not bool(V[1] >= lo):
                                    raise ValueError("Summation will set Idx value too low.")
                            except TypeError:
                                pass
                            try:
                                if hi is not None and not bool(V[2] <= hi):
                                    raise ValueError("Summation will set Idx value too high.")
                            except TypeError:
                                pass
                        limits.append(Tuple(*V))
                        continue
                    if lenV == 1 or (lenV == 2 and V[1] is None):
                        limits.append(Tuple(newsymbol))
                        continue
                    elif lenV == 2:
                        limits.append(Tuple(newsymbol, V[1]))
                        continue

        raise ValueError('Invalid limits given: %s' % str(symbols))

    return limits, orientation


class ExprWithLimits(Expr):
    __slots__ = ('is_commutative',)

    def __new__(cls, function, *symbols, **assumptions):
        from sympy.concrete.products import Product
        pre = _common_new(cls, function, *symbols,
            discrete=issubclass(cls, Product), **assumptions)
        if isinstance(pre, tuple):
            function, limits, _ = pre
        else:
            return pre

        # limits must have upper and lower bounds; the indefinite form
        # is not supported. This restriction does not apply to AddWithLimits
        if any(len(l) != 3 or None in l for l in limits):
            raise ValueError('ExprWithLimits requires values for lower and upper bounds.')

        obj = Expr.__new__(cls, **assumptions)
        arglist = [function]
        arglist.extend(limits)
        obj._args = tuple(arglist)
        obj.is_commutative = function.is_commutative  # limits already checked

        return obj

    @property
    def function(self):
        """Return the function applied across limits.

        Examples
        ========

        >>> from sympy import Integral
        >>> from sympy.abc import x
        >>> Integral(x**2, (x,)).function
        x**2

        See Also
        ========

        limits, variables, free_symbols
        """
        return self._args[0]

    @property
    def kind(self):
        return self.function.kind

    @property
    def limits(self):
        """Return the limits of expression.

        Examples
        ========

        >>> from sympy import Integral
        >>> from sympy.abc import x, i
        >>> Integral(x**i, (i, 1, 3)).limits
        ((i, 1, 3),)

        See Also
        ========

        function, variables, free_symbols
        """
        return self._args[1:]

    @property
    def variables(self):
        """Return a list of the limit variables.

        >>> from sympy import Sum
        >>> from sympy.abc import x, i
        >>> Sum(x**i, (i, 1, 3)).variables
        [i]

        See Also
        ========

        function, limits, free_symbols
        as_dummy : Rename dummy variables
        sympy.integrals.integrals.Integral.transform : Perform mapping on the dummy variable
        """
        return [l[0] for l in self.limits]

    @property
    def bound_symbols(self):
        """Return only variables that are dummy variables.

        Examples
        ========

        >>> from sympy import Integral
        >>> from sympy.abc import x, i, j, k
        >>> Integral(x**i, (i, 1, 3), (j, 2), k).bound_symbols
        [i, j]

        See Also
        ========

        function, limits, free_symbols
        as_dummy : Rename dummy variables
        sympy.integrals.integrals.Integral.transform : Perform mapping on the dummy variable
        """
        return [l[0] for l in self.limits if len(l) != 1]

    @property
    def free_symbols(self):
        """
        This method returns the symbols in the object, excluding those
        that take on a specific value (i.e. the dummy symbols).

        Examples
        ========

        >>> from sympy import Sum
        >>> from sympy.abc import x, y
        >>> Sum(x, (x, y, 1)).free_symbols
        {y}
        """
        # don't test for any special values -- nominal free symbols
        # should be returned, e.g. don't return set() if the
        # function is zero -- treat it like an unevaluated expression.
        function, limits = self.function, self.limits
        # mask off non-symbol integration variables that have
        # more than themself as a free symbol
        reps = {i[0]: i[0] if i[0].free_symbols == {i[0]} else Dummy()
            for i in self.limits}
        function = function.xreplace(reps)
        isyms = function.free_symbols
        for xab in limits:
            v = reps[xab[0]]
            if len(xab) == 1:
                isyms.add(v)
                continue
            # take out the target symbol
            if v in isyms:
                isyms.remove(v)
            # add in the new symbols
            for i in xab[1:]:
                isyms.update(i.free_symbols)
        reps = {v: k for k, v in reps.items()}
        return {reps.get(_, _) for _ in isyms}

    @property
    def is_number(self):
        """Return True if the Sum has no free symbols, else False."""
        return not self.free_symbols

    def _eval_interval(self, x, a, b):
        limits = [(i if i[0] != x else (x, a, b)) for i in self.limits]
        integrand = self.function
        return self.func(integrand, *limits)

    def _eval_subs(self, old, new):
        """
        Perform substitutions over non-dummy variables
        of an expression with limits.  Also, can be used
        to specify point-evaluation of an abstract antiderivative.

        Examples
        ========

        >>> from sympy import Sum, oo
        >>> from sympy.abc import s, n
        >>> Sum(1/n**s, (n, 1, oo)).subs(s, 2)
        Sum(n**(-2), (n, 1, oo))

        >>> from sympy import Integral
        >>> from sympy.abc import x, a
        >>> Integral(a*x**2, x).subs(x, 4)
        Integral(a*x**2, (x, 4))

        See Also
        ========

        variables : Lists the integration variables
        transform : Perform mapping on the dummy variable for integrals
        change_index : Perform mapping on the sum and product dummy variables

        """
        func, limits = self.function, list(self.limits)

        # If one of the expressions we are replacing is used as a func index
        # one of two things happens.
        #   - the old variable first appears as a free variable
        #     so we perform all free substitutions before it becomes
        #     a func index.
        #   - the old variable first appears as a func index, in
        #     which case we ignore.  See change_index.

        # Reorder limits to match standard mathematical practice for scoping
        limits.reverse()

        if not isinstance(old, Symbol) or \
                old.free_symbols.intersection(self.free_symbols):
            sub_into_func = True
            for i, xab in enumerate(limits):
                if 1 == len(xab) and old == xab[0]:
                    if new._diff_wrt:
                        xab = (new,)
                    else:
                        xab = (old, old)
                limits[i] = Tuple(xab[0], *[l._subs(old, new) for l in xab[1:]])
                if len(xab[0].free_symbols.intersection(old.free_symbols)) != 0:
                    sub_into_func = False
                    break
            if isinstance(old, (AppliedUndef, UndefinedFunction)):
                sy2 = set(self.variables).intersection(set(new.atoms(Symbol)))
                sy1 = set(self.variables).intersection(set(old.args))
                if not sy2.issubset(sy1):
                    raise ValueError(
                        "substitution cannot create dummy dependencies")
                sub_into_func = True
            if sub_into_func:
                func = func.subs(old, new)
        else:
            # old is a Symbol and a dummy variable of some limit
            for i, xab in enumerate(limits):
                if len(xab) == 3:
                    limits[i] = Tuple(xab[0], *[l._subs(old, new) for l in xab[1:]])
                    if old == xab[0]:
                        break
        # simplify redundant limits (x, x)  to (x, )
        for i, xab in enumerate(limits):
            if len(xab) == 2 and (xab[0] - xab[1]).is_zero:
                limits[i] = Tuple(xab[0], )

        # Reorder limits back to representation-form
        limits.reverse()

        return self.func(func, *limits)

    @property
    def has_finite_limits(self):
        """
        Returns True if the limits are known to be finite, either by the
        explicit bounds, assumptions on the bounds, or assumptions on the
        variables.  False if known to be infinite, based on the bounds.
        None if not enough information is available to determine.

        Examples
        ========

        >>> from sympy import Sum, Integral, Product, oo, Symbol
        >>> x = Symbol('x')
        >>> Sum(x, (x, 1, 8)).has_finite_limits
        True

        >>> Integral(x, (x, 1, oo)).has_finite_limits
        False

        >>> M = Symbol('M')
        >>> Sum(x, (x, 1, M)).has_finite_limits

        >>> N = Symbol('N', integer=True)
        >>> Product(x, (x, 1, N)).has_finite_limits
        True

        See Also
        ========

        has_reversed_limits

        """

        ret_None = False
        for lim in self.limits:
            if len(lim) == 3:
                if any(l.is_infinite for l in lim[1:]):
                    # Any of the bounds are +/-oo
                    return False
                elif any(l.is_infinite is None for l in lim[1:]):
                    # Maybe there are assumptions on the variable?
                    if lim[0].is_infinite is None:
                        ret_None = True
            else:
                if lim[0].is_infinite is None:
                    ret_None = True

        if ret_None:
            return None
        return True

    @property
    def has_reversed_limits(self):
        """
        Returns True if the limits are known to be in reversed order, either
        by the explicit bounds, assumptions on the bounds, or assumptions on the
        variables.  False if known to be in normal order, based on the bounds.
        None if not enough information is available to determine.

        Examples
        ========

        >>> from sympy import Sum, Integral, Product, oo, Symbol
        >>> x = Symbol('x')
        >>> Sum(x, (x, 8, 1)).has_reversed_limits
        True

        >>> Sum(x, (x, 1, oo)).has_reversed_limits
        False

        >>> M = Symbol('M')
        >>> Integral(x, (x, 1, M)).has_reversed_limits

        >>> N = Symbol('N', integer=True, positive=True)
        >>> Sum(x, (x, 1, N)).has_reversed_limits
        False

        >>> Product(x, (x, 2, N)).has_reversed_limits

        >>> Product(x, (x, 2, N)).subs(N, N + 2).has_reversed_limits
        False

        See Also
        ========

        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.has_empty_sequence

        """
        ret_None = False
        for lim in self.limits:
            if len(lim) == 3:
                var, a, b = lim
                dif = b - a
                if dif.is_extended_negative:
                    return True
                elif dif.is_extended_nonnegative:
                    continue
                else:
                    ret_None = True
            else:
                return None
        if ret_None:
            return None
        return False


class AddWithLimits(ExprWithLimits):
    r"""Represents unevaluated oriented additions.
        Parent class for Integral and Sum.
    """

    __slots__ = ()

    def __new__(cls, function, *symbols, **assumptions):
        from sympy.concrete.summations import Sum
        pre = _common_new(cls, function, *symbols,
            discrete=issubclass(cls, Sum), **assumptions)
        if isinstance(pre, tuple):
            function, limits, orientation = pre
        else:
            return pre

        obj = Expr.__new__(cls, **assumptions)
        arglist = [orientation*function]  # orientation not used in ExprWithLimits
        arglist.extend(limits)
        obj._args = tuple(arglist)
        obj.is_commutative = function.is_commutative  # limits already checked

        return obj

    def _eval_adjoint(self):
        if all(x.is_real for x in flatten(self.limits)):
            return self.func(self.function.adjoint(), *self.limits)
        return None

    def _eval_conjugate(self):
        if all(x.is_real for x in flatten(self.limits)):
            return self.func(self.function.conjugate(), *self.limits)
        return None

    def _eval_transpose(self):
        if all(x.is_real for x in flatten(self.limits)):
            return self.func(self.function.transpose(), *self.limits)
        return None

    def _eval_factor(self, **hints):
        if 1 == len(self.limits):
            summand = self.function.factor(**hints)
            if summand.is_Mul:
                out = sift(summand.args, lambda w: w.is_commutative \
                    and not set(self.variables) & w.free_symbols)
                return Mul(*out[True])*self.func(Mul(*out[False]), \
                    *self.limits)
        else:
            summand = self.func(self.function, *self.limits[0:-1]).factor()
            if not summand.has(self.variables[-1]):
                return self.func(1, [self.limits[-1]]).doit()*summand
            elif isinstance(summand, Mul):
                return self.func(summand, self.limits[-1]).factor()
        return self

    def _eval_expand_basic(self, **hints):
        summand = self.function.expand(**hints)
        force = hints.get('force', False)
        if (summand.is_Add and (force or summand.is_commutative and
                 self.has_finite_limits is not False)):
            return Add(*[self.func(i, *self.limits) for i in summand.args])
        elif isinstance(summand, MatrixBase):
            return summand.applyfunc(lambda x: self.func(x, *self.limits))
        elif summand != self.function:
            return self.func(summand, *self.limits)
        return self

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tseries\frequencies.py ===
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas._libs import lib
from pandas._libs.algos import unique_deltas
from pandas._libs.tslibs import (
    Timestamp,
    get_unit_from_dtype,
    periods_per_day,
    tz_convert_from_utc,
)
from pandas._libs.tslibs.ccalendar import (
    DAYS,
    MONTH_ALIASES,
    MONTH_NUMBERS,
    MONTHS,
    int_to_weekday,
)
from pandas._libs.tslibs.dtypes import (
    OFFSET_TO_PERIOD_FREQSTR,
    freq_to_period_freqstr,
)
from pandas._libs.tslibs.fields import (
    build_field_sarray,
    month_position_check,
)
from pandas._libs.tslibs.offsets import (
    DateOffset,
    Day,
    to_offset,
)
from pandas._libs.tslibs.parsing import get_rule_month
from pandas.util._decorators import cache_readonly

from pandas.core.dtypes.common import is_numeric_dtype
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    PeriodDtype,
)
from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCSeries,
)

from pandas.core.algorithms import unique

if TYPE_CHECKING:
    from pandas._typing import npt

    from pandas import (
        DatetimeIndex,
        Series,
        TimedeltaIndex,
    )
    from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
# --------------------------------------------------------------------
# Offset related functions

_need_suffix = ["QS", "BQE", "BQS", "YS", "BYE", "BYS"]

for _prefix in _need_suffix:
    for _m in MONTHS:
        key = f"{_prefix}-{_m}"
        OFFSET_TO_PERIOD_FREQSTR[key] = OFFSET_TO_PERIOD_FREQSTR[_prefix]

for _prefix in ["Y", "Q"]:
    for _m in MONTHS:
        _alias = f"{_prefix}-{_m}"
        OFFSET_TO_PERIOD_FREQSTR[_alias] = _alias

for _d in DAYS:
    OFFSET_TO_PERIOD_FREQSTR[f"W-{_d}"] = f"W-{_d}"


def get_period_alias(offset_str: str) -> str | None:
    """
    Alias to closest period strings BQ->Q etc.
    """
    return OFFSET_TO_PERIOD_FREQSTR.get(offset_str, None)


# ---------------------------------------------------------------------
# Period codes


def infer_freq(
    index: DatetimeIndex | TimedeltaIndex | Series | DatetimeLikeArrayMixin,
) -> str | None:
    """
    Infer the most likely frequency given the input index.

    Parameters
    ----------
    index : DatetimeIndex, TimedeltaIndex, Series or array-like
      If passed a Series will use the values of the series (NOT THE INDEX).

    Returns
    -------
    str or None
        None if no discernible frequency.

    Raises
    ------
    TypeError
        If the index is not datetime-like.
    ValueError
        If there are fewer than three values.

    Examples
    --------
    >>> idx = pd.date_range(start='2020/12/01', end='2020/12/30', periods=30)
    >>> pd.infer_freq(idx)
    'D'
    """
    from pandas.core.api import DatetimeIndex

    if isinstance(index, ABCSeries):
        values = index._values
        if not (
            lib.is_np_dtype(values.dtype, "mM")
            or isinstance(values.dtype, DatetimeTZDtype)
            or values.dtype == object
        ):
            raise TypeError(
                "cannot infer freq from a non-convertible dtype "
                f"on a Series of {index.dtype}"
            )
        index = values

    inferer: _FrequencyInferer

    if not hasattr(index, "dtype"):
        pass
    elif isinstance(index.dtype, PeriodDtype):
        raise TypeError(
            "PeriodIndex given. Check the `freq` attribute "
            "instead of using infer_freq."
        )
    elif lib.is_np_dtype(index.dtype, "m"):
        # Allow TimedeltaIndex and TimedeltaArray
        inferer = _TimedeltaFrequencyInferer(index)
        return inferer.get_freq()

    elif is_numeric_dtype(index.dtype):
        raise TypeError(
            f"cannot infer freq from a non-convertible index of dtype {index.dtype}"
        )

    if not isinstance(index, DatetimeIndex):
        index = DatetimeIndex(index)

    inferer = _FrequencyInferer(index)
    return inferer.get_freq()


class _FrequencyInferer:
    """
    Not sure if I can avoid the state machine here
    """

    def __init__(self, index) -> None:
        self.index = index
        self.i8values = index.asi8

        # For get_unit_from_dtype we need the dtype to the underlying ndarray,
        #  which for tz-aware is not the same as index.dtype
        if isinstance(index, ABCIndex):
            # error: Item "ndarray[Any, Any]" of "Union[ExtensionArray,
            # ndarray[Any, Any]]" has no attribute "_ndarray"
            self._creso = get_unit_from_dtype(
                index._data._ndarray.dtype  # type: ignore[union-attr]
            )
        else:
            # otherwise we have DTA/TDA
            self._creso = get_unit_from_dtype(index._ndarray.dtype)

        # This moves the values, which are implicitly in UTC, to the
        # the timezone so they are in local time
        if hasattr(index, "tz"):
            if index.tz is not None:
                self.i8values = tz_convert_from_utc(
                    self.i8values, index.tz, reso=self._creso
                )

        if len(index) < 3:
            raise ValueError("Need at least 3 dates to infer frequency")

        self.is_monotonic = (
            self.index._is_monotonic_increasing or self.index._is_monotonic_decreasing
        )

    @cache_readonly
    def deltas(self) -> npt.NDArray[np.int64]:
        return unique_deltas(self.i8values)

    @cache_readonly
    def deltas_asi8(self) -> npt.NDArray[np.int64]:
        # NB: we cannot use self.i8values here because we may have converted
        #  the tz in __init__
        return unique_deltas(self.index.asi8)

    @cache_readonly
    def is_unique(self) -> bool:
        return len(self.deltas) == 1

    @cache_readonly
    def is_unique_asi8(self) -> bool:
        return len(self.deltas_asi8) == 1

    def get_freq(self) -> str | None:
        """
        Find the appropriate frequency string to describe the inferred
        frequency of self.i8values

        Returns
        -------
        str or None
        """
        if not self.is_monotonic or not self.index._is_unique:
            return None

        delta = self.deltas[0]
        ppd = periods_per_day(self._creso)
        if delta and _is_multiple(delta, ppd):
            return self._infer_daily_rule()

        # Business hourly, maybe. 17: one day / 65: one weekend
        if self.hour_deltas in ([1, 17], [1, 65], [1, 17, 65]):
            return "bh"

        # Possibly intraday frequency.  Here we use the
        # original .asi8 values as the modified values
        # will not work around DST transitions.  See #8772
        if not self.is_unique_asi8:
            return None

        delta = self.deltas_asi8[0]
        pph = ppd // 24
        ppm = pph // 60
        pps = ppm // 60
        if _is_multiple(delta, pph):
            # Hours
            return _maybe_add_count("h", delta / pph)
        elif _is_multiple(delta, ppm):
            # Minutes
            return _maybe_add_count("min", delta / ppm)
        elif _is_multiple(delta, pps):
            # Seconds
            return _maybe_add_count("s", delta / pps)
        elif _is_multiple(delta, (pps // 1000)):
            # Milliseconds
            return _maybe_add_count("ms", delta / (pps // 1000))
        elif _is_multiple(delta, (pps // 1_000_000)):
            # Microseconds
            return _maybe_add_count("us", delta / (pps // 1_000_000))
        else:
            # Nanoseconds
            return _maybe_add_count("ns", delta)

    @cache_readonly
    def day_deltas(self) -> list[int]:
        ppd = periods_per_day(self._creso)
        return [x / ppd for x in self.deltas]

    @cache_readonly
    def hour_deltas(self) -> list[int]:
        pph = periods_per_day(self._creso) // 24
        return [x / pph for x in self.deltas]

    @cache_readonly
    def fields(self) -> np.ndarray:  # structured array of fields
        return build_field_sarray(self.i8values, reso=self._creso)

    @cache_readonly
    def rep_stamp(self) -> Timestamp:
        return Timestamp(self.i8values[0], unit=self.index.unit)

    def month_position_check(self) -> str | None:
        return month_position_check(self.fields, self.index.dayofweek)

    @cache_readonly
    def mdiffs(self) -> npt.NDArray[np.int64]:
        nmonths = self.fields["Y"] * 12 + self.fields["M"]
        return unique_deltas(nmonths.astype("i8"))

    @cache_readonly
    def ydiffs(self) -> npt.NDArray[np.int64]:
        return unique_deltas(self.fields["Y"].astype("i8"))

    def _infer_daily_rule(self) -> str | None:
        annual_rule = self._get_annual_rule()
        if annual_rule:
            nyears = self.ydiffs[0]
            month = MONTH_ALIASES[self.rep_stamp.month]
            alias = f"{annual_rule}-{month}"
            return _maybe_add_count(alias, nyears)

        quarterly_rule = self._get_quarterly_rule()
        if quarterly_rule:
            nquarters = self.mdiffs[0] / 3
            mod_dict = {0: 12, 2: 11, 1: 10}
            month = MONTH_ALIASES[mod_dict[self.rep_stamp.month % 3]]
            alias = f"{quarterly_rule}-{month}"
            return _maybe_add_count(alias, nquarters)

        monthly_rule = self._get_monthly_rule()
        if monthly_rule:
            return _maybe_add_count(monthly_rule, self.mdiffs[0])

        if self.is_unique:
            return self._get_daily_rule()

        if self._is_business_daily():
            return "B"

        wom_rule = self._get_wom_rule()
        if wom_rule:
            return wom_rule

        return None

    def _get_daily_rule(self) -> str | None:
        ppd = periods_per_day(self._creso)
        days = self.deltas[0] / ppd
        if days % 7 == 0:
            # Weekly
            wd = int_to_weekday[self.rep_stamp.weekday()]
            alias = f"W-{wd}"
            return _maybe_add_count(alias, days / 7)
        else:
            return _maybe_add_count("D", days)

    def _get_annual_rule(self) -> str | None:
        if len(self.ydiffs) > 1:
            return None

        if len(unique(self.fields["M"])) > 1:
            return None

        pos_check = self.month_position_check()

        if pos_check is None:
            return None
        else:
            return {"cs": "YS", "bs": "BYS", "ce": "YE", "be": "BYE"}.get(pos_check)

    def _get_quarterly_rule(self) -> str | None:
        if len(self.mdiffs) > 1:
            return None

        if not self.mdiffs[0] % 3 == 0:
            return None

        pos_check = self.month_position_check()

        if pos_check is None:
            return None
        else:
            return {"cs": "QS", "bs": "BQS", "ce": "QE", "be": "BQE"}.get(pos_check)

    def _get_monthly_rule(self) -> str | None:
        if len(self.mdiffs) > 1:
            return None
        pos_check = self.month_position_check()

        if pos_check is None:
            return None
        else:
            return {"cs": "MS", "bs": "BMS", "ce": "ME", "be": "BME"}.get(pos_check)

    def _is_business_daily(self) -> bool:
        # quick check: cannot be business daily
        if self.day_deltas != [1, 3]:
            return False

        # probably business daily, but need to confirm
        first_weekday = self.index[0].weekday()
        shifts = np.diff(self.i8values)
        ppd = periods_per_day(self._creso)
        shifts = np.floor_divide(shifts, ppd)
        weekdays = np.mod(first_weekday + np.cumsum(shifts), 7)

        return bool(
            np.all(
                ((weekdays == 0) & (shifts == 3))
                | ((weekdays > 0) & (weekdays <= 4) & (shifts == 1))
            )
        )

    def _get_wom_rule(self) -> str | None:
        weekdays = unique(self.index.weekday)
        if len(weekdays) > 1:
            return None

        week_of_months = unique((self.index.day - 1) // 7)
        # Only attempt to infer up to WOM-4. See #9425
        week_of_months = week_of_months[week_of_months < 4]
        if len(week_of_months) == 0 or len(week_of_months) > 1:
            return None

        # get which week
        week = week_of_months[0] + 1
        wd = int_to_weekday[weekdays[0]]

        return f"WOM-{week}{wd}"


class _TimedeltaFrequencyInferer(_FrequencyInferer):
    def _infer_daily_rule(self):
        if self.is_unique:
            return self._get_daily_rule()


def _is_multiple(us, mult: int) -> bool:
    return us % mult == 0


def _maybe_add_count(base: str, count: float) -> str:
    if count != 1:
        assert count == int(count)
        count = int(count)
        return f"{count}{base}"
    else:
        return base


# ----------------------------------------------------------------------
# Frequency comparison


def is_subperiod(source, target) -> bool:
    """
    Returns True if downsampling is possible between source and target
    frequencies

    Parameters
    ----------
    source : str or DateOffset
        Frequency converting from
    target : str or DateOffset
        Frequency converting to

    Returns
    -------
    bool
    """
    if target is None or source is None:
        return False
    source = _maybe_coerce_freq(source)
    target = _maybe_coerce_freq(target)

    if _is_annual(target):
        if _is_quarterly(source):
            return _quarter_months_conform(
                get_rule_month(source), get_rule_month(target)
            )
        return source in {"D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}
    elif _is_quarterly(target):
        return source in {"D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}
    elif _is_monthly(target):
        return source in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
    elif _is_weekly(target):
        return source in {target, "D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
    elif target == "B":
        return source in {"B", "h", "min", "s", "ms", "us", "ns"}
    elif target == "C":
        return source in {"C", "h", "min", "s", "ms", "us", "ns"}
    elif target == "D":
        return source in {"D", "h", "min", "s", "ms", "us", "ns"}
    elif target == "h":
        return source in {"h", "min", "s", "ms", "us", "ns"}
    elif target == "min":
        return source in {"min", "s", "ms", "us", "ns"}
    elif target == "s":
        return source in {"s", "ms", "us", "ns"}
    elif target == "ms":
        return source in {"ms", "us", "ns"}
    elif target == "us":
        return source in {"us", "ns"}
    elif target == "ns":
        return source in {"ns"}
    else:
        return False


def is_superperiod(source, target) -> bool:
    """
    Returns True if upsampling is possible between source and target
    frequencies

    Parameters
    ----------
    source : str or DateOffset
        Frequency converting from
    target : str or DateOffset
        Frequency converting to

    Returns
    -------
    bool
    """
    if target is None or source is None:
        return False
    source = _maybe_coerce_freq(source)
    target = _maybe_coerce_freq(target)

    if _is_annual(source):
        if _is_annual(target):
            return get_rule_month(source) == get_rule_month(target)

        if _is_quarterly(target):
            smonth = get_rule_month(source)
            tmonth = get_rule_month(target)
            return _quarter_months_conform(smonth, tmonth)
        return target in {"D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}
    elif _is_quarterly(source):
        return target in {"D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}
    elif _is_monthly(source):
        return target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
    elif _is_weekly(source):
        return target in {source, "D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
    elif source == "B":
        return target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
    elif source == "C":
        return target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
    elif source == "D":
        return target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
    elif source == "h":
        return target in {"h", "min", "s", "ms", "us", "ns"}
    elif source == "min":
        return target in {"min", "s", "ms", "us", "ns"}
    elif source == "s":
        return target in {"s", "ms", "us", "ns"}
    elif source == "ms":
        return target in {"ms", "us", "ns"}
    elif source == "us":
        return target in {"us", "ns"}
    elif source == "ns":
        return target in {"ns"}
    else:
        return False


def _maybe_coerce_freq(code) -> str:
    """we might need to coerce a code to a rule_code
    and uppercase it

    Parameters
    ----------
    source : str or DateOffset
        Frequency converting from

    Returns
    -------
    str
    """
    assert code is not None
    if isinstance(code, DateOffset):
        code = freq_to_period_freqstr(1, code.name)
    if code in {"h", "min", "s", "ms", "us", "ns"}:
        return code
    else:
        return code.upper()


def _quarter_months_conform(source: str, target: str) -> bool:
    snum = MONTH_NUMBERS[source]
    tnum = MONTH_NUMBERS[target]
    return snum % 3 == tnum % 3


def _is_annual(rule: str) -> bool:
    rule = rule.upper()
    return rule == "Y" or rule.startswith("Y-")


def _is_quarterly(rule: str) -> bool:
    rule = rule.upper()
    return rule == "Q" or rule.startswith(("Q-", "BQ"))


def _is_monthly(rule: str) -> bool:
    rule = rule.upper()
    return rule in ("M", "BM")


def _is_weekly(rule: str) -> bool:
    rule = rule.upper()
    return rule == "W" or rule.startswith("W-")


__all__ = [
    "Day",
    "get_period_alias",
    "infer_freq",
    "is_subperiod",
    "is_superperiod",
    "to_offset",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\pygments\lexers\_mapping.py ===
# Automatically generated by scripts/gen_mapfiles.py.
# DO NOT EDIT BY HAND; run `tox -e mapfiles` instead.

LEXERS = {
    'ABAPLexer': ('pip._vendor.pygments.lexers.business', 'ABAP', ('abap',), ('*.abap', '*.ABAP'), ('text/x-abap',)),
    'AMDGPULexer': ('pip._vendor.pygments.lexers.amdgpu', 'AMDGPU', ('amdgpu',), ('*.isa',), ()),
    'APLLexer': ('pip._vendor.pygments.lexers.apl', 'APL', ('apl',), ('*.apl', '*.aplf', '*.aplo', '*.apln', '*.aplc', '*.apli', '*.dyalog'), ()),
    'AbnfLexer': ('pip._vendor.pygments.lexers.grammar_notation', 'ABNF', ('abnf',), ('*.abnf',), ('text/x-abnf',)),
    'ActionScript3Lexer': ('pip._vendor.pygments.lexers.actionscript', 'ActionScript 3', ('actionscript3', 'as3'), ('*.as',), ('application/x-actionscript3', 'text/x-actionscript3', 'text/actionscript3')),
    'ActionScriptLexer': ('pip._vendor.pygments.lexers.actionscript', 'ActionScript', ('actionscript', 'as'), ('*.as',), ('application/x-actionscript', 'text/x-actionscript', 'text/actionscript')),
    'AdaLexer': ('pip._vendor.pygments.lexers.ada', 'Ada', ('ada', 'ada95', 'ada2005'), ('*.adb', '*.ads', '*.ada'), ('text/x-ada',)),
    'AdlLexer': ('pip._vendor.pygments.lexers.archetype', 'ADL', ('adl',), ('*.adl', '*.adls', '*.adlf', '*.adlx'), ()),
    'AgdaLexer': ('pip._vendor.pygments.lexers.haskell', 'Agda', ('agda',), ('*.agda',), ('text/x-agda',)),
    'AheuiLexer': ('pip._vendor.pygments.lexers.esoteric', 'Aheui', ('aheui',), ('*.aheui',), ()),
    'AlloyLexer': ('pip._vendor.pygments.lexers.dsls', 'Alloy', ('alloy',), ('*.als',), ('text/x-alloy',)),
    'AmbientTalkLexer': ('pip._vendor.pygments.lexers.ambient', 'AmbientTalk', ('ambienttalk', 'ambienttalk/2', 'at'), ('*.at',), ('text/x-ambienttalk',)),
    'AmplLexer': ('pip._vendor.pygments.lexers.ampl', 'Ampl', ('ampl',), ('*.run',), ()),
    'Angular2HtmlLexer': ('pip._vendor.pygments.lexers.templates', 'HTML + Angular2', ('html+ng2',), ('*.ng2',), ()),
    'Angular2Lexer': ('pip._vendor.pygments.lexers.templates', 'Angular2', ('ng2',), (), ()),
    'AntlrActionScriptLexer': ('pip._vendor.pygments.lexers.parsers', 'ANTLR With ActionScript Target', ('antlr-actionscript', 'antlr-as'), ('*.G', '*.g'), ()),
    'AntlrCSharpLexer': ('pip._vendor.pygments.lexers.parsers', 'ANTLR With C# Target', ('antlr-csharp', 'antlr-c#'), ('*.G', '*.g'), ()),
    'AntlrCppLexer': ('pip._vendor.pygments.lexers.parsers', 'ANTLR With CPP Target', ('antlr-cpp',), ('*.G', '*.g'), ()),
    'AntlrJavaLexer': ('pip._vendor.pygments.lexers.parsers', 'ANTLR With Java Target', ('antlr-java',), ('*.G', '*.g'), ()),
    'AntlrLexer': ('pip._vendor.pygments.lexers.parsers', 'ANTLR', ('antlr',), (), ()),
    'AntlrObjectiveCLexer': ('pip._vendor.pygments.lexers.parsers', 'ANTLR With ObjectiveC Target', ('antlr-objc',), ('*.G', '*.g'), ()),
    'AntlrPerlLexer': ('pip._vendor.pygments.lexers.parsers', 'ANTLR With Perl Target', ('antlr-perl',), ('*.G', '*.g'), ()),
    'AntlrPythonLexer': ('pip._vendor.pygments.lexers.parsers', 'ANTLR With Python Target', ('antlr-python',), ('*.G', '*.g'), ()),
    'AntlrRubyLexer': ('pip._vendor.pygments.lexers.parsers', 'ANTLR With Ruby Target', ('antlr-ruby', 'antlr-rb'), ('*.G', '*.g'), ()),
    'ApacheConfLexer': ('pip._vendor.pygments.lexers.configs', 'ApacheConf', ('apacheconf', 'aconf', 'apache'), ('.htaccess', 'apache.conf', 'apache2.conf'), ('text/x-apacheconf',)),
    'AppleScriptLexer': ('pip._vendor.pygments.lexers.scripting', 'AppleScript', ('applescript',), ('*.applescript',), ()),
    'ArduinoLexer': ('pip._vendor.pygments.lexers.c_like', 'Arduino', ('arduino',), ('*.ino',), ('text/x-arduino',)),
    'ArrowLexer': ('pip._vendor.pygments.lexers.arrow', 'Arrow', ('arrow',), ('*.arw',), ()),
    'ArturoLexer': ('pip._vendor.pygments.lexers.arturo', 'Arturo', ('arturo', 'art'), ('*.art',), ()),
    'AscLexer': ('pip._vendor.pygments.lexers.asc', 'ASCII armored', ('asc', 'pem'), ('*.asc', '*.pem', 'id_dsa', 'id_ecdsa', 'id_ecdsa_sk', 'id_ed25519', 'id_ed25519_sk', 'id_rsa'), ('application/pgp-keys', 'application/pgp-encrypted', 'application/pgp-signature', 'application/pem-certificate-chain')),
    'Asn1Lexer': ('pip._vendor.pygments.lexers.asn1', 'ASN.1', ('asn1',), ('*.asn1',), ()),
    'AspectJLexer': ('pip._vendor.pygments.lexers.jvm', 'AspectJ', ('aspectj',), ('*.aj',), ('text/x-aspectj',)),
    'AsymptoteLexer': ('pip._vendor.pygments.lexers.graphics', 'Asymptote', ('asymptote', 'asy'), ('*.asy',), ('text/x-asymptote',)),
    'AugeasLexer': ('pip._vendor.pygments.lexers.configs', 'Augeas', ('augeas',), ('*.aug',), ()),
    'AutoItLexer': ('pip._vendor.pygments.lexers.automation', 'AutoIt', ('autoit',), ('*.au3',), ('text/x-autoit',)),
    'AutohotkeyLexer': ('pip._vendor.pygments.lexers.automation', 'autohotkey', ('autohotkey', 'ahk'), ('*.ahk', '*.ahkl'), ('text/x-autohotkey',)),
    'AwkLexer': ('pip._vendor.pygments.lexers.textedit', 'Awk', ('awk', 'gawk', 'mawk', 'nawk'), ('*.awk',), ('application/x-awk',)),
    'BBCBasicLexer': ('pip._vendor.pygments.lexers.basic', 'BBC Basic', ('bbcbasic',), ('*.bbc',), ()),
    'BBCodeLexer': ('pip._vendor.pygments.lexers.markup', 'BBCode', ('bbcode',), (), ('text/x-bbcode',)),
    'BCLexer': ('pip._vendor.pygments.lexers.algebra', 'BC', ('bc',), ('*.bc',), ()),
    'BQNLexer': ('pip._vendor.pygments.lexers.bqn', 'BQN', ('bqn',), ('*.bqn',), ()),
    'BSTLexer': ('pip._vendor.pygments.lexers.bibtex', 'BST', ('bst', 'bst-pybtex'), ('*.bst',), ()),
    'BareLexer': ('pip._vendor.pygments.lexers.bare', 'BARE', ('bare',), ('*.bare',), ()),
    'BaseMakefileLexer': ('pip._vendor.pygments.lexers.make', 'Base Makefile', ('basemake',), (), ()),
    'BashLexer': ('pip._vendor.pygments.lexers.shell', 'Bash', ('bash', 'sh', 'ksh', 'zsh', 'shell', 'openrc'), ('*.sh', '*.ksh', '*.bash', '*.ebuild', '*.eclass', '*.exheres-0', '*.exlib', '*.zsh', '.bashrc', 'bashrc', '.bash_*', 'bash_*', 'zshrc', '.zshrc', '.kshrc', 'kshrc', 'PKGBUILD'), ('application/x-sh', 'application/x-shellscript', 'text/x-shellscript')),
    'BashSessionLexer': ('pip._vendor.pygments.lexers.shell', 'Bash Session', ('console', 'shell-session'), ('*.sh-session', '*.shell-session'), ('application/x-shell-session', 'application/x-sh-session')),
    'BatchLexer': ('pip._vendor.pygments.lexers.shell', 'Batchfile', ('batch', 'bat', 'dosbatch', 'winbatch'), ('*.bat', '*.cmd'), ('application/x-dos-batch',)),
    'BddLexer': ('pip._vendor.pygments.lexers.bdd', 'Bdd', ('bdd',), ('*.feature',), ('text/x-bdd',)),
    'BefungeLexer': ('pip._vendor.pygments.lexers.esoteric', 'Befunge', ('befunge',), ('*.befunge',), ('application/x-befunge',)),
    'BerryLexer': ('pip._vendor.pygments.lexers.berry', 'Berry', ('berry', 'be'), ('*.be',), ('text/x-berry', 'application/x-berry')),
    'BibTeXLexer': ('pip._vendor.pygments.lexers.bibtex', 'BibTeX', ('bibtex', 'bib'), ('*.bib',), ('text/x-bibtex',)),
    'BlitzBasicLexer': ('pip._vendor.pygments.lexers.basic', 'BlitzBasic', ('blitzbasic', 'b3d', 'bplus'), ('*.bb', '*.decls'), ('text/x-bb',)),
    'BlitzMaxLexer': ('pip._vendor.pygments.lexers.basic', 'BlitzMax', ('blitzmax', 'bmax'), ('*.bmx',), ('text/x-bmx',)),
    'BlueprintLexer': ('pip._vendor.pygments.lexers.blueprint', 'Blueprint', ('blueprint',), ('*.blp',), ('text/x-blueprint',)),
    'BnfLexer': ('pip._vendor.pygments.lexers.grammar_notation', 'BNF', ('bnf',), ('*.bnf',), ('text/x-bnf',)),
    'BoaLexer': ('pip._vendor.pygments.lexers.boa', 'Boa', ('boa',), ('*.boa',), ()),
    'BooLexer': ('pip._vendor.pygments.lexers.dotnet', 'Boo', ('boo',), ('*.boo',), ('text/x-boo',)),
    'BoogieLexer': ('pip._vendor.pygments.lexers.verification', 'Boogie', ('boogie',), ('*.bpl',), ()),
    'BrainfuckLexer': ('pip._vendor.pygments.lexers.esoteric', 'Brainfuck', ('brainfuck', 'bf'), ('*.bf', '*.b'), ('application/x-brainfuck',)),
    'BugsLexer': ('pip._vendor.pygments.lexers.modeling', 'BUGS', ('bugs', 'winbugs', 'openbugs'), ('*.bug',), ()),
    'CAmkESLexer': ('pip._vendor.pygments.lexers.esoteric', 'CAmkES', ('camkes', 'idl4'), ('*.camkes', '*.idl4'), ()),
    'CLexer': ('pip._vendor.pygments.lexers.c_cpp', 'C', ('c',), ('*.c', '*.h', '*.idc', '*.x[bp]m'), ('text/x-chdr', 'text/x-csrc', 'image/x-xbitmap', 'image/x-xpixmap')),
    'CMakeLexer': ('pip._vendor.pygments.lexers.make', 'CMake', ('cmake',), ('*.cmake', 'CMakeLists.txt'), ('text/x-cmake',)),
    'CObjdumpLexer': ('pip._vendor.pygments.lexers.asm', 'c-objdump', ('c-objdump',), ('*.c-objdump',), ('text/x-c-objdump',)),
    'CPSALexer': ('pip._vendor.pygments.lexers.lisp', 'CPSA', ('cpsa',), ('*.cpsa',), ()),
    'CSSUL4Lexer': ('pip._vendor.pygments.lexers.ul4', 'CSS+UL4', ('css+ul4',), ('*.cssul4',), ()),
    'CSharpAspxLexer': ('pip._vendor.pygments.lexers.dotnet', 'aspx-cs', ('aspx-cs',), ('*.aspx', '*.asax', '*.ascx', '*.ashx', '*.asmx', '*.axd'), ()),
    'CSharpLexer': ('pip._vendor.pygments.lexers.dotnet', 'C#', ('csharp', 'c#', 'cs'), ('*.cs',), ('text/x-csharp',)),
    'Ca65Lexer': ('pip._vendor.pygments.lexers.asm', 'ca65 assembler', ('ca65',), ('*.s',), ()),
    'CadlLexer': ('pip._vendor.pygments.lexers.archetype', 'cADL', ('cadl',), ('*.cadl',), ()),
    'CapDLLexer': ('pip._vendor.pygments.lexers.esoteric', 'CapDL', ('capdl',), ('*.cdl',), ()),
    'CapnProtoLexer': ('pip._vendor.pygments.lexers.capnproto', "Cap'n Proto", ('capnp',), ('*.capnp',), ()),
    'CarbonLexer': ('pip._vendor.pygments.lexers.carbon', 'Carbon', ('carbon',), ('*.carbon',), ('text/x-carbon',)),
    'CbmBasicV2Lexer': ('pip._vendor.pygments.lexers.basic', 'CBM BASIC V2', ('cbmbas',), ('*.bas',), ()),
    'CddlLexer': ('pip._vendor.pygments.lexers.cddl', 'CDDL', ('cddl',), ('*.cddl',), ('text/x-cddl',)),
    'CeylonLexer': ('pip._vendor.pygments.lexers.jvm', 'Ceylon', ('ceylon',), ('*.ceylon',), ('text/x-ceylon',)),
    'Cfengine3Lexer': ('pip._vendor.pygments.lexers.configs', 'CFEngine3', ('cfengine3', 'cf3'), ('*.cf',), ()),
    'ChaiscriptLexer': ('pip._vendor.pygments.lexers.scripting', 'ChaiScript', ('chaiscript', 'chai'), ('*.chai',), ('text/x-chaiscript', 'application/x-chaiscript')),
    'ChapelLexer': ('pip._vendor.pygments.lexers.chapel', 'Chapel', ('chapel', 'chpl'), ('*.chpl',), ()),
    'CharmciLexer': ('pip._vendor.pygments.lexers.c_like', 'Charmci', ('charmci',), ('*.ci',), ()),
    'CheetahHtmlLexer': ('pip._vendor.pygments.lexers.templates', 'HTML+Cheetah', ('html+cheetah', 'html+spitfire', 'htmlcheetah'), (), ('text/html+cheetah', 'text/html+spitfire')),
    'CheetahJavascriptLexer': ('pip._vendor.pygments.lexers.templates', 'JavaScript+Cheetah', ('javascript+cheetah', 'js+cheetah', 'javascript+spitfire', 'js+spitfire'), (), ('application/x-javascript+cheetah', 'text/x-javascript+cheetah', 'text/javascript+cheetah', 'application/x-javascript+spitfire', 'text/x-javascript+spitfire', 'text/javascript+spitfire')),
    'CheetahLexer': ('pip._vendor.pygments.lexers.templates', 'Cheetah', ('cheetah', 'spitfire'), ('*.tmpl', '*.spt'), ('application/x-cheetah', 'application/x-spitfire')),
    'CheetahXmlLexer': ('pip._vendor.pygments.lexers.templates', 'XML+Cheetah', ('xml+cheetah', 'xml+spitfire'), (), ('application/xml+cheetah', 'application/xml+spitfire')),
    'CirruLexer': ('pip._vendor.pygments.lexers.webmisc', 'Cirru', ('cirru',), ('*.cirru',), ('text/x-cirru',)),
    'ClayLexer': ('pip._vendor.pygments.lexers.c_like', 'Clay', ('clay',), ('*.clay',), ('text/x-clay',)),
    'CleanLexer': ('pip._vendor.pygments.lexers.clean', 'Clean', ('clean',), ('*.icl', '*.dcl'), ()),
    'ClojureLexer': ('pip._vendor.pygments.lexers.jvm', 'Clojure', ('clojure', 'clj'), ('*.clj', '*.cljc'), ('text/x-clojure', 'application/x-clojure')),
    'ClojureScriptLexer': ('pip._vendor.pygments.lexers.jvm', 'ClojureScript', ('clojurescript', 'cljs'), ('*.cljs',), ('text/x-clojurescript', 'application/x-clojurescript')),
    'CobolFreeformatLexer': ('pip._vendor.pygments.lexers.business', 'COBOLFree', ('cobolfree',), ('*.cbl', '*.CBL'), ()),
    'CobolLexer': ('pip._vendor.pygments.lexers.business', 'COBOL', ('cobol',), ('*.cob', '*.COB', '*.cpy', '*.CPY'), ('text/x-cobol',)),
    'CodeQLLexer': ('pip._vendor.pygments.lexers.codeql', 'CodeQL', ('codeql', 'ql'), ('*.ql', '*.qll'), ()),
    'CoffeeScriptLexer': ('pip._vendor.pygments.lexers.javascript', 'CoffeeScript', ('coffeescript', 'coffee-script', 'coffee'), ('*.coffee',), ('text/coffeescript',)),
    'ColdfusionCFCLexer': ('pip._vendor.pygments.lexers.templates', 'Coldfusion CFC', ('cfc',), ('*.cfc',), ()),
    'ColdfusionHtmlLexer': ('pip._vendor.pygments.lexers.templates', 'Coldfusion HTML', ('cfm',), ('*.cfm', '*.cfml'), ('application/x-coldfusion',)),
    'ColdfusionLexer': ('pip._vendor.pygments.lexers.templates', 'cfstatement', ('cfs',), (), ()),
    'Comal80Lexer': ('pip._vendor.pygments.lexers.comal', 'COMAL-80', ('comal', 'comal80'), ('*.cml', '*.comal'), ()),
    'CommonLispLexer': ('pip._vendor.pygments.lexers.lisp', 'Common Lisp', ('common-lisp', 'cl', 'lisp'), ('*.cl', '*.lisp'), ('text/x-common-lisp',)),
    'ComponentPascalLexer': ('pip._vendor.pygments.lexers.oberon', 'Component Pascal', ('componentpascal', 'cp'), ('*.cp', '*.cps'), ('text/x-component-pascal',)),
    'CoqLexer': ('pip._vendor.pygments.lexers.theorem', 'Coq', ('coq',), ('*.v',), ('text/x-coq',)),
    'CplintLexer': ('pip._vendor.pygments.lexers.cplint', 'cplint', ('cplint',), ('*.ecl', '*.prolog', '*.pro', '*.pl', '*.P', '*.lpad', '*.cpl'), ('text/x-cplint',)),
    'CppLexer': ('pip._vendor.pygments.lexers.c_cpp', 'C++', ('cpp', 'c++'), ('*.cpp', '*.hpp', '*.c++', '*.h++', '*.cc', '*.hh', '*.cxx', '*.hxx', '*.C', '*.H', '*.cp', '*.CPP', '*.tpp'), ('text/x-c++hdr', 'text/x-c++src')),
    'CppObjdumpLexer': ('pip._vendor.pygments.lexers.asm', 'cpp-objdump', ('cpp-objdump', 'c++-objdumb', 'cxx-objdump'), ('*.cpp-objdump', '*.c++-objdump', '*.cxx-objdump'), ('text/x-cpp-objdump',)),
    'CrmshLexer': ('pip._vendor.pygments.lexers.dsls', 'Crmsh', ('crmsh', 'pcmk'), ('*.crmsh', '*.pcmk'), ()),
    'CrocLexer': ('pip._vendor.pygments.lexers.d', 'Croc', ('croc',), ('*.croc',), ('text/x-crocsrc',)),
    'CryptolLexer': ('pip._vendor.pygments.lexers.haskell', 'Cryptol', ('cryptol', 'cry'), ('*.cry',), ('text/x-cryptol',)),
    'CrystalLexer': ('pip._vendor.pygments.lexers.crystal', 'Crystal', ('cr', 'crystal'), ('*.cr',), ('text/x-crystal',)),
    'CsoundDocumentLexer': ('pip._vendor.pygments.lexers.csound', 'Csound Document', ('csound-document', 'csound-csd'), ('*.csd',), ()),
    'CsoundOrchestraLexer': ('pip._vendor.pygments.lexers.csound', 'Csound Orchestra', ('csound', 'csound-orc'), ('*.orc', '*.udo'), ()),
    'CsoundScoreLexer': ('pip._vendor.pygments.lexers.csound', 'Csound Score', ('csound-score', 'csound-sco'), ('*.sco',), ()),
    'CssDjangoLexer': ('pip._vendor.pygments.lexers.templates', 'CSS+Django/Jinja', ('css+django', 'css+jinja'), ('*.css.j2', '*.css.jinja2'), ('text/css+django', 'text/css+jinja')),
    'CssErbLexer': ('pip._vendor.pygments.lexers.templates', 'CSS+Ruby', ('css+ruby', 'css+erb'), (), ('text/css+ruby',)),
    'CssGenshiLexer': ('pip._vendor.pygments.lexers.templates', 'CSS+Genshi Text', ('css+genshitext', 'css+genshi'), (), ('text/css+genshi',)),
    'CssLexer': ('pip._vendor.pygments.lexers.css', 'CSS', ('css',), ('*.css',), ('text/css',)),
    'CssPhpLexer': ('pip._vendor.pygments.lexers.templates', 'CSS+PHP', ('css+php',), (), ('text/css+php',)),
    'CssSmartyLexer': ('pip._vendor.pygments.lexers.templates', 'CSS+Smarty', ('css+smarty',), (), ('text/css+smarty',)),
    'CudaLexer': ('pip._vendor.pygments.lexers.c_like', 'CUDA', ('cuda', 'cu'), ('*.cu', '*.cuh'), ('text/x-cuda',)),
    'CypherLexer': ('pip._vendor.pygments.lexers.graph', 'Cypher', ('cypher',), ('*.cyp', '*.cypher'), ()),
    'CythonLexer': ('pip._vendor.pygments.lexers.python', 'Cython', ('cython', 'pyx', 'pyrex'), ('*.pyx', '*.pxd', '*.pxi'), ('text/x-cython', 'application/x-cython')),
    'DLexer': ('pip._vendor.pygments.lexers.d', 'D', ('d',), ('*.d', '*.di'), ('text/x-dsrc',)),
    'DObjdumpLexer': ('pip._vendor.pygments.lexers.asm', 'd-objdump', ('d-objdump',), ('*.d-objdump',), ('text/x-d-objdump',)),
    'DarcsPatchLexer': ('pip._vendor.pygments.lexers.diff', 'Darcs Patch', ('dpatch',), ('*.dpatch', '*.darcspatch'), ()),
    'DartLexer': ('pip._vendor.pygments.lexers.javascript', 'Dart', ('dart',), ('*.dart',), ('text/x-dart',)),
    'Dasm16Lexer': ('pip._vendor.pygments.lexers.asm', 'DASM16', ('dasm16',), ('*.dasm16', '*.dasm'), ('text/x-dasm16',)),
    'DaxLexer': ('pip._vendor.pygments.lexers.dax', 'Dax', ('dax',), ('*.dax',), ()),
    'DebianControlLexer': ('pip._vendor.pygments.lexers.installers', 'Debian Control file', ('debcontrol', 'control'), ('control',), ()),
    'DebianSourcesLexer': ('pip._vendor.pygments.lexers.installers', 'Debian Sources file', ('debian.sources',), ('*.sources',), ()),
    'DelphiLexer': ('pip._vendor.pygments.lexers.pascal', 'Delphi', ('delphi', 'pas', 'pascal', 'objectpascal'), ('*.pas', '*.dpr'), ('text/x-pascal',)),
    'DesktopLexer': ('pip._vendor.pygments.lexers.configs', 'Desktop file', ('desktop',), ('*.desktop',), ('application/x-desktop',)),
    'DevicetreeLexer': ('pip._vendor.pygments.lexers.devicetree', 'Devicetree', ('devicetree', 'dts'), ('*.dts', '*.dtsi'), ('text/x-c',)),
    'DgLexer': ('pip._vendor.pygments.lexers.python', 'dg', ('dg',), ('*.dg',), ('text/x-dg',)),
    'DiffLexer': ('pip._vendor.pygments.lexers.diff', 'Diff', ('diff', 'udiff'), ('*.diff', '*.patch'), ('text/x-diff', 'text/x-patch')),
    'DjangoLexer': ('pip._vendor.pygments.lexers.templates', 'Django/Jinja', ('django', 'jinja'), (), ('application/x-django-templating', 'application/x-jinja')),
    'DnsZoneLexer': ('pip._vendor.pygments.lexers.dns', 'Zone', ('zone',), ('*.zone',), ('text/dns',)),
    'DockerLexer': ('pip._vendor.pygments.lexers.configs', 'Docker', ('docker', 'dockerfile'), ('Dockerfile', '*.docker'), ('text/x-dockerfile-config',)),
    'DtdLexer': ('pip._vendor.pygments.lexers.html', 'DTD', ('dtd',), ('*.dtd',), ('application/xml-dtd',)),
    'DuelLexer': ('pip._vendor.pygments.lexers.webmisc', 'Duel', ('duel', 'jbst', 'jsonml+bst'), ('*.duel', '*.jbst'), ('text/x-duel', 'text/x-jbst')),
    'DylanConsoleLexer': ('pip._vendor.pygments.lexers.dylan', 'Dylan session', ('dylan-console', 'dylan-repl'), ('*.dylan-console',), ('text/x-dylan-console',)),
    'DylanLexer': ('pip._vendor.pygments.lexers.dylan', 'Dylan', ('dylan',), ('*.dylan', '*.dyl', '*.intr'), ('text/x-dylan',)),
    'DylanLidLexer': ('pip._vendor.pygments.lexers.dylan', 'DylanLID', ('dylan-lid', 'lid'), ('*.lid', '*.hdp'), ('text/x-dylan-lid',)),
    'ECLLexer': ('pip._vendor.pygments.lexers.ecl', 'ECL', ('ecl',), ('*.ecl',), ('application/x-ecl',)),
    'ECLexer': ('pip._vendor.pygments.lexers.c_like', 'eC', ('ec',), ('*.ec', '*.eh'), ('text/x-echdr', 'text/x-ecsrc')),
    'EarlGreyLexer': ('pip._vendor.pygments.lexers.javascript', 'Earl Grey', ('earl-grey', 'earlgrey', 'eg'), ('*.eg',), ('text/x-earl-grey',)),
    'EasytrieveLexer': ('pip._vendor.pygments.lexers.scripting', 'Easytrieve', ('easytrieve',), ('*.ezt', '*.mac'), ('text/x-easytrieve',)),
    'EbnfLexer': ('pip._vendor.pygments.lexers.parsers', 'EBNF', ('ebnf',), ('*.ebnf',), ('text/x-ebnf',)),
    'EiffelLexer': ('pip._vendor.pygments.lexers.eiffel', 'Eiffel', ('eiffel',), ('*.e',), ('text/x-eiffel',)),
    'ElixirConsoleLexer': ('pip._vendor.pygments.lexers.erlang', 'Elixir iex session', ('iex',), (), ('text/x-elixir-shellsession',)),
    'ElixirLexer': ('pip._vendor.pygments.lexers.erlang', 'Elixir', ('elixir', 'ex', 'exs'), ('*.ex', '*.eex', '*.exs', '*.leex'), ('text/x-elixir',)),
    'ElmLexer': ('pip._vendor.pygments.lexers.elm', 'Elm', ('elm',), ('*.elm',), ('text/x-elm',)),
    'ElpiLexer': ('pip._vendor.pygments.lexers.elpi', 'Elpi', ('elpi',), ('*.elpi',), ('text/x-elpi',)),
    'EmacsLispLexer': ('pip._vendor.pygments.lexers.lisp', 'EmacsLisp', ('emacs-lisp', 'elisp', 'emacs'), ('*.el',), ('text/x-elisp', 'application/x-elisp')),
    'EmailLexer': ('pip._vendor.pygments.lexers.email', 'E-mail', ('email', 'eml'), ('*.eml',), ('message/rfc822',)),
    'ErbLexer': ('pip._vendor.pygments.lexers.templates', 'ERB', ('erb',), (), ('application/x-ruby-templating',)),
    'ErlangLexer': ('pip._vendor.pygments.lexers.erlang', 'Erlang', ('erlang',), ('*.erl', '*.hrl', '*.es', '*.escript'), ('text/x-erlang',)),
    'ErlangShellLexer': ('pip._vendor.pygments.lexers.erlang', 'Erlang erl session', ('erl',), ('*.erl-sh',), ('text/x-erl-shellsession',)),
    'EvoqueHtmlLexer': ('pip._vendor.pygments.lexers.templates', 'HTML+Evoque', ('html+evoque',), (), ('text/html+evoque',)),
    'EvoqueLexer': ('pip._vendor.pygments.lexers.templates', 'Evoque', ('evoque',), ('*.evoque',), ('application/x-evoque',)),
    'EvoqueXmlLexer': ('pip._vendor.pygments.lexers.templates', 'XML+Evoque', ('xml+evoque',), (), ('application/xml+evoque',)),
    'ExeclineLexer': ('pip._vendor.pygments.lexers.shell', 'execline', ('execline',), ('*.exec',), ()),
    'EzhilLexer': ('pip._vendor.pygments.lexers.ezhil', 'Ezhil', ('ezhil',), ('*.n',), ('text/x-ezhil',)),
    'FSharpLexer': ('pip._vendor.pygments.lexers.dotnet', 'F#', ('fsharp', 'f#'), ('*.fs', '*.fsi', '*.fsx'), ('text/x-fsharp',)),
    'FStarLexer': ('pip._vendor.pygments.lexers.ml', 'FStar', ('fstar',), ('*.fst', '*.fsti'), ('text/x-fstar',)),
    'FactorLexer': ('pip._vendor.pygments.lexers.factor', 'Factor', ('factor',), ('*.factor',), ('text/x-factor',)),
    'FancyLexer': ('pip._vendor.pygments.lexers.ruby', 'Fancy', ('fancy', 'fy'), ('*.fy', '*.fancypack'), ('text/x-fancysrc',)),
    'FantomLexer': ('pip._vendor.pygments.lexers.fantom', 'Fantom', ('fan',), ('*.fan',), ('application/x-fantom',)),
    'FelixLexer': ('pip._vendor.pygments.lexers.felix', 'Felix', ('felix', 'flx'), ('*.flx', '*.flxh'), ('text/x-felix',)),
    'FennelLexer': ('pip._vendor.pygments.lexers.lisp', 'Fennel', ('fennel', 'fnl'), ('*.fnl',), ()),
    'FiftLexer': ('pip._vendor.pygments.lexers.fift', 'Fift', ('fift', 'fif'), ('*.fif',), ()),
    'FishShellLexer': ('pip._vendor.pygments.lexers.shell', 'Fish', ('fish', 'fishshell'), ('*.fish', '*.load'), ('application/x-fish',)),
    'FlatlineLexer': ('pip._vendor.pygments.lexers.dsls', 'Flatline', ('flatline',), (), ('text/x-flatline',)),
    'FloScriptLexer': ('pip._vendor.pygments.lexers.floscript', 'FloScript', ('floscript', 'flo'), ('*.flo',), ()),
    'ForthLexer': ('pip._vendor.pygments.lexers.forth', 'Forth', ('forth',), ('*.frt', '*.fs'), ('application/x-forth',)),
    'FortranFixedLexer': ('pip._vendor.pygments.lexers.fortran', 'FortranFixed', ('fortranfixed',), ('*.f', '*.F'), ()),
    'FortranLexer': ('pip._vendor.pygments.lexers.fortran', 'Fortran', ('fortran', 'f90'), ('*.f03', '*.f90', '*.F03', '*.F90'), ('text/x-fortran',)),
    'FoxProLexer': ('pip._vendor.pygments.lexers.foxpro', 'FoxPro', ('foxpro', 'vfp', 'clipper', 'xbase'), ('*.PRG', '*.prg'), ()),
    'FreeFemLexer': ('pip._vendor.pygments.lexers.freefem', 'Freefem', ('freefem',), ('*.edp',), ('text/x-freefem',)),
    'FuncLexer': ('pip._vendor.pygments.lexers.func', 'FunC', ('func', 'fc'), ('*.fc', '*.func'), ()),
    'FutharkLexer': ('pip._vendor.pygments.lexers.futhark', 'Futhark', ('futhark',), ('*.fut',), ('text/x-futhark',)),
    'GAPConsoleLexer': ('pip._vendor.pygments.lexers.algebra', 'GAP session', ('gap-console', 'gap-repl'), ('*.tst',), ()),
    'GAPLexer': ('pip._vendor.pygments.lexers.algebra', 'GAP', ('gap',), ('*.g', '*.gd', '*.gi', '*.gap'), ()),
    'GDScriptLexer': ('pip._vendor.pygments.lexers.gdscript', 'GDScript', ('gdscript', 'gd'), ('*.gd',), ('text/x-gdscript', 'application/x-gdscript')),
    'GLShaderLexer': ('pip._vendor.pygments.lexers.graphics', 'GLSL', ('glsl',), ('*.vert', '*.frag', '*.geo'), ('text/x-glslsrc',)),
    'GSQLLexer': ('pip._vendor.pygments.lexers.gsql', 'GSQL', ('gsql',), ('*.gsql',), ()),
    'GasLexer': ('pip._vendor.pygments.lexers.asm', 'GAS', ('gas', 'asm'), ('*.s', '*.S'), ('text/x-gas',)),
    'GcodeLexer': ('pip._vendor.pygments.lexers.gcodelexer', 'g-code', ('gcode',), ('*.gcode',), ()),
    'GenshiLexer': ('pip._vendor.pygments.lexers.templates', 'Genshi', ('genshi', 'kid', 'xml+genshi', 'xml+kid'), ('*.kid',), ('application/x-genshi', 'application/x-kid')),
    'GenshiTextLexer': ('pip._vendor.pygments.lexers.templates', 'Genshi Text', ('genshitext',), (), ('application/x-genshi-text', 'text/x-genshi')),
    'GettextLexer': ('pip._vendor.pygments.lexers.textfmts', 'Gettext Catalog', ('pot', 'po'), ('*.pot', '*.po'), ('application/x-gettext', 'text/x-gettext', 'text/gettext')),
    'GherkinLexer': ('pip._vendor.pygments.lexers.testing', 'Gherkin', ('gherkin', 'cucumber'), ('*.feature',), ('text/x-gherkin',)),
    'GleamLexer': ('pip._vendor.pygments.lexers.gleam', 'Gleam', ('gleam',), ('*.gleam',), ('text/x-gleam',)),
    'GnuplotLexer': ('pip._vendor.pygments.lexers.graphics', 'Gnuplot', ('gnuplot',), ('*.plot', '*.plt'), ('text/x-gnuplot',)),
    'GoLexer': ('pip._vendor.pygments.lexers.go', 'Go', ('go', 'golang'), ('*.go',), ('text/x-gosrc',)),
    'GoloLexer': ('pip._vendor.pygments.lexers.jvm', 'Golo', ('golo',), ('*.golo',), ()),
    'GoodDataCLLexer': ('pip._vendor.pygments.lexers.business', 'GoodData-CL', ('gooddata-cl',), ('*.gdc',), ('text/x-gooddata-cl',)),
    'GoogleSqlLexer': ('pip._vendor.pygments.lexers.sql', 'GoogleSQL', ('googlesql', 'zetasql'), ('*.googlesql', '*.googlesql.sql'), ('text/x-google-sql', 'text/x-google-sql-aux')),
    'GosuLexer': ('pip._vendor.pygments.lexers.jvm', 'Gosu', ('gosu',), ('*.gs', '*.gsx', '*.gsp', '*.vark'), ('text/x-gosu',)),
    'GosuTemplateLexer': ('pip._vendor.pygments.lexers.jvm', 'Gosu Template', ('gst',), ('*.gst',), ('text/x-gosu-template',)),
    'GraphQLLexer': ('pip._vendor.pygments.lexers.graphql', 'GraphQL', ('graphql',), ('*.graphql',), ()),
    'GraphvizLexer': ('pip._vendor.pygments.lexers.graphviz', 'Graphviz', ('graphviz', 'dot'), ('*.gv', '*.dot'), ('text/x-graphviz', 'text/vnd.graphviz')),
    'GroffLexer': ('pip._vendor.pygments.lexers.markup', 'Groff', ('groff', 'nroff', 'man'), ('*.[1-9]', '*.man', '*.1p', '*.3pm'), ('application/x-troff', 'text/troff')),
    'GroovyLexer': ('pip._vendor.pygments.lexers.jvm', 'Groovy', ('groovy',), ('*.groovy', '*.gradle'), ('text/x-groovy',)),
    'HLSLShaderLexer': ('pip._vendor.pygments.lexers.graphics', 'HLSL', ('hlsl',), ('*.hlsl', '*.hlsli'), ('text/x-hlsl',)),
    'HTMLUL4Lexer': ('pip._vendor.pygments.lexers.ul4', 'HTML+UL4', ('html+ul4',), ('*.htmlul4',), ()),
    'HamlLexer': ('pip._vendor.pygments.lexers.html', 'Haml', ('haml',), ('*.haml',), ('text/x-haml',)),
    'HandlebarsHtmlLexer': ('pip._vendor.pygments.lexers.templates', 'HTML+Handlebars', ('html+handlebars',), ('*.handlebars', '*.hbs'), ('text/html+handlebars', 'text/x-handlebars-template')),
    'HandlebarsLexer': ('pip._vendor.pygments.lexers.templates', 'Handlebars', ('handlebars',), (), ()),
    'HareLexer': ('pip._vendor.pygments.lexers.hare', 'Hare', ('hare',), ('*.ha',), ('text/x-hare',)),
    'HaskellLexer': ('pip._vendor.pygments.lexers.haskell', 'Haskell', ('haskell', 'hs'), ('*.hs',), ('text/x-haskell',)),
    'HaxeLexer': ('pip._vendor.pygments.lexers.haxe', 'Haxe', ('haxe', 'hxsl', 'hx'), ('*.hx', '*.hxsl'), ('text/haxe', 'text/x-haxe', 'text/x-hx')),
    'HexdumpLexer': ('pip._vendor.pygments.lexers.hexdump', 'Hexdump', ('hexdump',), (), ()),
    'HsailLexer': ('pip._vendor.pygments.lexers.asm', 'HSAIL', ('hsail', 'hsa'), ('*.hsail',), ('text/x-hsail',)),
    'HspecLexer': ('pip._vendor.pygments.lexers.haskell', 'Hspec', ('hspec',), ('*Spec.hs',), ()),
    'HtmlDjangoLexer': ('pip._vendor.pygments.lexers.templates', 'HTML+Django/Jinja', ('html+django', 'html+jinja', 'htmldjango'), ('*.html.j2', '*.htm.j2', '*.xhtml.j2', '*.html.jinja2', '*.htm.jinja2', '*.xhtml.jinja2'), ('text/html+django', 'text/html+jinja')),
    'HtmlGenshiLexer': ('pip._vendor.pygments.lexers.templates', 'HTML+Genshi', ('html+genshi', 'html+kid'), (), ('text/html+genshi',)),
    'HtmlLexer': ('pip._vendor.pygments.lexers.html', 'HTML', ('html',), ('*.html', '*.htm', '*.xhtml', '*.xslt'), ('text/html', 'application/xhtml+xml')),
    'HtmlPhpLexer': ('pip._vendor.pygments.lexers.templates', 'HTML+PHP', ('html+php',), ('*.phtml',), ('application/x-php', 'application/x-httpd-php', 'application/x-httpd-php3', 'application/x-httpd-php4', 'application/x-httpd-php5')),
    'HtmlSmartyLexer': ('pip._vendor.pygments.lexers.templates', 'HTML+Smarty', ('html+smarty',), (), ('text/html+smarty',)),
    'HttpLexer': ('pip._vendor.pygments.lexers.textfmts', 'HTTP', ('http',), (), ()),
    'HxmlLexer': ('pip._vendor.pygments.lexers.haxe', 'Hxml', ('haxeml', 'hxml'), ('*.hxml',), ()),
    'HyLexer': ('pip._vendor.pygments.lexers.lisp', 'Hy', ('hylang', 'hy'), ('*.hy',), ('text/x-hy', 'application/x-hy')),
    'HybrisLexer': ('pip._vendor.pygments.lexers.scripting', 'Hybris', ('hybris',), ('*.hyb',), ('text/x-hybris', 'application/x-hybris')),
    'IDLLexer': ('pip._vendor.pygments.lexers.idl', 'IDL', ('idl',), ('*.pro',), ('text/idl',)),
    'IconLexer': ('pip._vendor.pygments.lexers.unicon', 'Icon', ('icon',), ('*.icon', '*.ICON'), ()),
    'IdrisLexer': ('pip._vendor.pygments.lexers.haskell', 'Idris', ('idris', 'idr'), ('*.idr',), ('text/x-idris',)),
    'IgorLexer': ('pip._vendor.pygments.lexers.igor', 'Igor', ('igor', 'igorpro'), ('*.ipf',), ('text/ipf',)),
    'Inform6Lexer': ('pip._vendor.pygments.lexers.int_fiction', 'Inform 6', ('inform6', 'i6'), ('*.inf',), ()),
    'Inform6TemplateLexer': ('pip._vendor.pygments.lexers.int_fiction', 'Inform 6 template', ('i6t',), ('*.i6t',), ()),
    'Inform7Lexer': ('pip._vendor.pygments.lexers.int_fiction', 'Inform 7', ('inform7', 'i7'), ('*.ni', '*.i7x'), ()),
    'IniLexer': ('pip._vendor.pygments.lexers.configs', 'INI', ('ini', 'cfg', 'dosini'), ('*.ini', '*.cfg', '*.inf', '.editorconfig'), ('text/x-ini', 'text/inf')),
    'IoLexer': ('pip._vendor.pygments.lexers.iolang', 'Io', ('io',), ('*.io',), ('text/x-iosrc',)),
    'IokeLexer': ('pip._vendor.pygments.lexers.jvm', 'Ioke', ('ioke', 'ik'), ('*.ik',), ('text/x-iokesrc',)),
    'IrcLogsLexer': ('pip._vendor.pygments.lexers.textfmts', 'IRC logs', ('irc',), ('*.weechatlog',), ('text/x-irclog',)),
    'IsabelleLexer': ('pip._vendor.pygments.lexers.theorem', 'Isabelle', ('isabelle',), ('*.thy',), ('text/x-isabelle',)),
    'JLexer': ('pip._vendor.pygments.lexers.j', 'J', ('j',), ('*.ijs',), ('text/x-j',)),
    'JMESPathLexer': ('pip._vendor.pygments.lexers.jmespath', 'JMESPath', ('jmespath', 'jp'), ('*.jp',), ()),
    'JSLTLexer': ('pip._vendor.pygments.lexers.jslt', 'JSLT', ('jslt',), ('*.jslt',), ('text/x-jslt',)),
    'JagsLexer': ('pip._vendor.pygments.lexers.modeling', 'JAGS', ('jags',), ('*.jag', '*.bug'), ()),
    'JanetLexer': ('pip._vendor.pygments.lexers.lisp', 'Janet', ('janet',), ('*.janet', '*.jdn'), ('text/x-janet', 'application/x-janet')),
    'JasminLexer': ('pip._vendor.pygments.lexers.jvm', 'Jasmin', ('jasmin', 'jasminxt'), ('*.j',), ()),
    'JavaLexer': ('pip._vendor.pygments.lexers.jvm', 'Java', ('java',), ('*.java',), ('text/x-java',)),
    'JavascriptDjangoLexer': ('pip._vendor.pygments.lexers.templates', 'JavaScript+Django/Jinja', ('javascript+django', 'js+django', 'javascript+jinja', 'js+jinja'), ('*.js.j2', '*.js.jinja2'), ('application/x-javascript+django', 'application/x-javascript+jinja', 'text/x-javascript+django', 'text/x-javascript+jinja', 'text/javascript+django', 'text/javascript+jinja')),
    'JavascriptErbLexer': ('pip._vendor.pygments.lexers.templates', 'JavaScript+Ruby', ('javascript+ruby', 'js+ruby', 'javascript+erb', 'js+erb'), (), ('application/x-javascript+ruby', 'text/x-javascript+ruby', 'text/javascript+ruby')),
    'JavascriptGenshiLexer': ('pip._vendor.pygments.lexers.templates', 'JavaScript+Genshi Text', ('js+genshitext', 'js+genshi', 'javascript+genshitext', 'javascript+genshi'), (), ('application/x-javascript+genshi', 'text/x-javascript+genshi', 'text/javascript+genshi')),
    'JavascriptLexer': ('pip._vendor.pygments.lexers.javascript', 'JavaScript', ('javascript', 'js'), ('*.js', '*.jsm', '*.mjs', '*.cjs'), ('application/javascript', 'application/x-javascript', 'text/x-javascript', 'text/javascript')),
    'JavascriptPhpLexer': ('pip._vendor.pygments.lexers.templates', 'JavaScript+PHP', ('javascript+php', 'js+php'), (), ('application/x-javascript+php', 'text/x-javascript+php', 'text/javascript+php')),
    'JavascriptSmartyLexer': ('pip._vendor.pygments.lexers.templates', 'JavaScript+Smarty', ('javascript+smarty', 'js+smarty'), (), ('application/x-javascript+smarty', 'text/x-javascript+smarty', 'text/javascript+smarty')),
    'JavascriptUL4Lexer': ('pip._vendor.pygments.lexers.ul4', 'Javascript+UL4', ('js+ul4',), ('*.jsul4',), ()),
    'JclLexer': ('pip._vendor.pygments.lexers.scripting', 'JCL', ('jcl',), ('*.jcl',), ('text/x-jcl',)),
    'JsgfLexer': ('pip._vendor.pygments.lexers.grammar_notation', 'JSGF', ('jsgf',), ('*.jsgf',), ('application/jsgf', 'application/x-jsgf', 'text/jsgf')),
    'Json5Lexer': ('pip._vendor.pygments.lexers.json5', 'JSON5', ('json5',), ('*.json5',), ()),
    'JsonBareObjectLexer': ('pip._vendor.pygments.lexers.data', 'JSONBareObject', (), (), ()),
    'JsonLdLexer': ('pip._vendor.pygments.lexers.data', 'JSON-LD', ('jsonld', 'json-ld'), ('*.jsonld',), ('application/ld+json',)),
    'JsonLexer': ('pip._vendor.pygments.lexers.data', 'JSON', ('json', 'json-object'), ('*.json', '*.jsonl', '*.ndjson', 'Pipfile.lock'), ('application/json', 'application/json-object', 'application/x-ndjson', 'application/jsonl', 'application/json-seq')),
    'JsonnetLexer': ('pip._vendor.pygments.lexers.jsonnet', 'Jsonnet', ('jsonnet',), ('*.jsonnet', '*.libsonnet'), ()),
    'JspLexer': ('pip._vendor.pygments.lexers.templates', 'Java Server Page', ('jsp',), ('*.jsp',), ('application/x-jsp',)),
    'JsxLexer': ('pip._vendor.pygments.lexers.jsx', 'JSX', ('jsx', 'react'), ('*.jsx', '*.react'), ('text/jsx', 'text/typescript-jsx')),
    'JuliaConsoleLexer': ('pip._vendor.pygments.lexers.julia', 'Julia console', ('jlcon', 'julia-repl'), (), ()),
    'JuliaLexer': ('pip._vendor.pygments.lexers.julia', 'Julia', ('julia', 'jl'), ('*.jl',), ('text/x-julia', 'application/x-julia')),
    'JuttleLexer': ('pip._vendor.pygments.lexers.javascript', 'Juttle', ('juttle',), ('*.juttle',), ('application/juttle', 'application/x-juttle', 'text/x-juttle', 'text/juttle')),
    'KLexer': ('pip._vendor.pygments.lexers.q', 'K', ('k',), ('*.k',), ()),
    'KalLexer': ('pip._vendor.pygments.lexers.javascript', 'Kal', ('kal',), ('*.kal',), ('text/kal', 'application/kal')),
    'KconfigLexer': ('pip._vendor.pygments.lexers.configs', 'Kconfig', ('kconfig', 'menuconfig', 'linux-config', 'kernel-config'), ('Kconfig*', '*Config.in*', 'external.in*', 'standard-modules.in'), ('text/x-kconfig',)),
    'KernelLogLexer': ('pip._vendor.pygments.lexers.textfmts', 'Kernel log', ('kmsg', 'dmesg'), ('*.kmsg', '*.dmesg'), ()),
    'KokaLexer': ('pip._vendor.pygments.lexers.haskell', 'Koka', ('koka',), ('*.kk', '*.kki'), ('text/x-koka',)),
    'KotlinLexer': ('pip._vendor.pygments.lexers.jvm', 'Kotlin', ('kotlin',), ('*.kt', '*.kts'), ('text/x-kotlin',)),
    'KuinLexer': ('pip._vendor.pygments.lexers.kuin', 'Kuin', ('kuin',), ('*.kn',), ()),
    'KustoLexer': ('pip._vendor.pygments.lexers.kusto', 'Kusto', ('kql', 'kusto'), ('*.kql', '*.kusto', '.csl'), ()),
    'LSLLexer': ('pip._vendor.pygments.lexers.scripting', 'LSL', ('lsl',), ('*.lsl',), ('text/x-lsl',)),
    'LassoCssLexer': ('pip._vendor.pygments.lexers.templates', 'CSS+Lasso', ('css+lasso',), (), ('text/css+lasso',)),
    'LassoHtmlLexer': ('pip._vendor.pygments.lexers.templates', 'HTML+Lasso', ('html+lasso',), (), ('text/html+lasso', 'application/x-httpd-lasso', 'application/x-httpd-lasso[89]')),
    'LassoJavascriptLexer': ('pip._vendor.pygments.lexers.templates', 'JavaScript+Lasso', ('javascript+lasso', 'js+lasso'), (), ('application/x-javascript+lasso', 'text/x-javascript+lasso', 'text/javascript+lasso')),
    'LassoLexer': ('pip._vendor.pygments.lexers.javascript', 'Lasso', ('lasso', 'lassoscript'), ('*.lasso', '*.lasso[89]'), ('text/x-lasso',)),
    'LassoXmlLexer': ('pip._vendor.pygments.lexers.templates', 'XML+Lasso', ('xml+lasso',), (), ('application/xml+lasso',)),
    'LdaprcLexer': ('pip._vendor.pygments.lexers.ldap', 'LDAP configuration file', ('ldapconf', 'ldaprc'), ('.ldaprc', 'ldaprc', 'ldap.conf'), ('text/x-ldapconf',)),
    'LdifLexer': ('pip._vendor.pygments.lexers.ldap', 'LDIF', ('ldif',), ('*.ldif',), ('text/x-ldif',)),
    'Lean3Lexer': ('pip._vendor.pygments.lexers.lean', 'Lean', ('lean', 'lean3'), ('*.lean',), ('text/x-lean', 'text/x-lean3')),
    'Lean4Lexer': ('pip._vendor.pygments.lexers.lean', 'Lean4', ('lean4',), ('*.lean',), ('text/x-lean4',)),
    'LessCssLexer': ('pip._vendor.pygments.lexers.css', 'LessCss', ('less',), ('*.less',), ('text/x-less-css',)),
    'LighttpdConfLexer': ('pip._vendor.pygments.lexers.configs', 'Lighttpd configuration file', ('lighttpd', 'lighty'), ('lighttpd.conf',), ('text/x-lighttpd-conf',)),
    'LilyPondLexer': ('pip._vendor.pygments.lexers.lilypond', 'LilyPond', ('lilypond',), ('*.ly',), ()),
    'LimboLexer': ('pip._vendor.pygments.lexers.inferno', 'Limbo', ('limbo',), ('*.b',), ('text/limbo',)),
    'LiquidLexer': ('pip._vendor.pygments.lexers.templates', 'liquid', ('liquid',), ('*.liquid',), ()),
    'LiterateAgdaLexer': ('pip._vendor.pygments.lexers.haskell', 'Literate Agda', ('literate-agda', 'lagda'), ('*.lagda',), ('text/x-literate-agda',)),
    'LiterateCryptolLexer': ('pip._vendor.pygments.lexers.haskell', 'Literate Cryptol', ('literate-cryptol', 'lcryptol', 'lcry'), ('*.lcry',), ('text/x-literate-cryptol',)),
    'LiterateHaskellLexer': ('pip._vendor.pygments.lexers.haskell', 'Literate Haskell', ('literate-haskell', 'lhaskell', 'lhs'), ('*.lhs',), ('text/x-literate-haskell',)),
    'LiterateIdrisLexer': ('pip._vendor.pygments.lexers.haskell', 'Literate Idris', ('literate-idris', 'lidris', 'lidr'), ('*.lidr',), ('text/x-literate-idris',)),
    'LiveScriptLexer': ('pip._vendor.pygments.lexers.javascript', 'LiveScript', ('livescript', 'live-script'), ('*.ls',), ('text/livescript',)),
    'LlvmLexer': ('pip._vendor.pygments.lexers.asm', 'LLVM', ('llvm',), ('*.ll',), ('text/x-llvm',)),
    'LlvmMirBodyLexer': ('pip._vendor.pygments.lexers.asm', 'LLVM-MIR Body', ('llvm-mir-body',), (), ()),
    'LlvmMirLexer': ('pip._vendor.pygments.lexers.asm', 'LLVM-MIR', ('llvm-mir',), ('*.mir',), ()),
    'LogosLexer': ('pip._vendor.pygments.lexers.objective', 'Logos', ('logos',), ('*.x', '*.xi', '*.xm', '*.xmi'), ('text/x-logos',)),
    'LogtalkLexer': ('pip._vendor.pygments.lexers.prolog', 'Logtalk', ('logtalk',), ('*.lgt', '*.logtalk'), ('text/x-logtalk',)),
    'LuaLexer': ('pip._vendor.pygments.lexers.scripting', 'Lua', ('lua',), ('*.lua', '*.wlua'), ('text/x-lua', 'application/x-lua')),
    'LuauLexer': ('pip._vendor.pygments.lexers.scripting', 'Luau', ('luau',), ('*.luau',), ()),
    'MCFunctionLexer': ('pip._vendor.pygments.lexers.minecraft', 'MCFunction', ('mcfunction', 'mcf'), ('*.mcfunction',), ('text/mcfunction',)),
    'MCSchemaLexer': ('pip._vendor.pygments.lexers.minecraft', 'MCSchema', ('mcschema',), ('*.mcschema',), ('text/mcschema',)),
    'MIMELexer': ('pip._vendor.pygments.lexers.mime', 'MIME', ('mime',), (), ('multipart/mixed', 'multipart/related', 'multipart/alternative')),
    'MIPSLexer': ('pip._vendor.pygments.lexers.mips', 'MIPS', ('mips',), ('*.mips', '*.MIPS'), ()),
    'MOOCodeLexer': ('pip._vendor.pygments.lexers.scripting', 'MOOCode', ('moocode', 'moo'), ('*.moo',), ('text/x-moocode',)),
    'MSDOSSessionLexer': ('pip._vendor.pygments.lexers.shell', 'MSDOS Session', ('doscon',), (), ()),
    'Macaulay2Lexer': ('pip._vendor.pygments.lexers.macaulay2', 'Macaulay2', ('macaulay2',), ('*.m2',), ()),
    'MakefileLexer': ('pip._vendor.pygments.lexers.make', 'Makefile', ('make', 'makefile', 'mf', 'bsdmake'), ('*.mak', '*.mk', 'Makefile', 'makefile', 'Makefile.*', 'GNUmakefile'), ('text/x-makefile',)),
    'MakoCssLexer': ('pip._vendor.pygments.lexers.templates', 'CSS+Mako', ('css+mako',), (), ('text/css+mako',)),
    'MakoHtmlLexer': ('pip._vendor.pygments.lexers.templates', 'HTML+Mako', ('html+mako',), (), ('text/html+mako',)),
    'MakoJavascriptLexer': ('pip._vendor.pygments.lexers.templates', 'JavaScript+Mako', ('javascript+mako', 'js+mako'), (), ('application/x-javascript+mako', 'text/x-javascript+mako', 'text/javascript+mako')),
    'MakoLexer': ('pip._vendor.pygments.lexers.templates', 'Mako', ('mako',), ('*.mao',), ('application/x-mako',)),
    'MakoXmlLexer': ('pip._vendor.pygments.lexers.templates', 'XML+Mako', ('xml+mako',), (), ('application/xml+mako',)),
    'MapleLexer': ('pip._vendor.pygments.lexers.maple', 'Maple', ('maple',), ('*.mpl', '*.mi', '*.mm'), ('text/x-maple',)),
    'MaqlLexer': ('pip._vendor.pygments.lexers.business', 'MAQL', ('maql',), ('*.maql',), ('text/x-gooddata-maql', 'application/x-gooddata-maql')),
    'MarkdownLexer': ('pip._vendor.pygments.lexers.markup', 'Markdown', ('markdown', 'md'), ('*.md', '*.markdown'), ('text/x-markdown',)),
    'MaskLexer': ('pip._vendor.pygments.lexers.javascript', 'Mask', ('mask',), ('*.mask',), ('text/x-mask',)),
    'MasonLexer': ('pip._vendor.pygments.lexers.templates', 'Mason', ('mason',), ('*.m', '*.mhtml', '*.mc', '*.mi', 'autohandler', 'dhandler'), ('application/x-mason',)),
    'MathematicaLexer': ('pip._vendor.pygments.lexers.algebra', 'Mathematica', ('mathematica', 'mma', 'nb'), ('*.nb', '*.cdf', '*.nbp', '*.ma'), ('application/mathematica', 'application/vnd.wolfram.mathematica', 'application/vnd.wolfram.mathematica.package', 'application/vnd.wolfram.cdf')),
    'MatlabLexer': ('pip._vendor.pygments.lexers.matlab', 'Matlab', ('matlab',), ('*.m',), ('text/matlab',)),
    'MatlabSessionLexer': ('pip._vendor.pygments.lexers.matlab', 'Matlab session', ('matlabsession',), (), ()),
    'MaximaLexer': ('pip._vendor.pygments.lexers.maxima', 'Maxima', ('maxima', 'macsyma'), ('*.mac', '*.max'), ()),
    'MesonLexer': ('pip._vendor.pygments.lexers.meson', 'Meson', ('meson', 'meson.build'), ('meson.build', 'meson_options.txt'), ('text/x-meson',)),
    'MiniDLexer': ('pip._vendor.pygments.lexers.d', 'MiniD', ('minid',), (), ('text/x-minidsrc',)),
    'MiniScriptLexer': ('pip._vendor.pygments.lexers.scripting', 'MiniScript', ('miniscript', 'ms'), ('*.ms',), ('text/x-minicript', 'application/x-miniscript')),
    'ModelicaLexer': ('pip._vendor.pygments.lexers.modeling', 'Modelica', ('modelica',), ('*.mo',), ('text/x-modelica',)),
    'Modula2Lexer': ('pip._vendor.pygments.lexers.modula2', 'Modula-2', ('modula2', 'm2'), ('*.def', '*.mod'), ('text/x-modula2',)),
    'MoinWikiLexer': ('pip._vendor.pygments.lexers.markup', 'MoinMoin/Trac Wiki markup', ('trac-wiki', 'moin'), (), ('text/x-trac-wiki',)),
    'MojoLexer': ('pip._vendor.pygments.lexers.mojo', 'Mojo', ('mojo', '🔥'), ('*.mojo', '*.🔥'), ('text/x-mojo', 'application/x-mojo')),
    'MonkeyLexer': ('pip._vendor.pygments.lexers.basic', 'Monkey', ('monkey',), ('*.monkey',), ('text/x-monkey',)),
    'MonteLexer': ('pip._vendor.pygments.lexers.monte', 'Monte', ('monte',), ('*.mt',), ()),
    'MoonScriptLexer': ('pip._vendor.pygments.lexers.scripting', 'MoonScript', ('moonscript', 'moon'), ('*.moon',), ('text/x-moonscript', 'application/x-moonscript')),
    'MoselLexer': ('pip._vendor.pygments.lexers.mosel', 'Mosel', ('mosel',), ('*.mos',), ()),
    'MozPreprocCssLexer': ('pip._vendor.pygments.lexers.markup', 'CSS+mozpreproc', ('css+mozpreproc',), ('*.css.in',), ()),
    'MozPreprocHashLexer': ('pip._vendor.pygments.lexers.markup', 'mozhashpreproc', ('mozhashpreproc',), (), ()),
    'MozPreprocJavascriptLexer': ('pip._vendor.pygments.lexers.markup', 'Javascript+mozpreproc', ('javascript+mozpreproc',), ('*.js.in',), ()),
    'MozPreprocPercentLexer': ('pip._vendor.pygments.lexers.markup', 'mozpercentpreproc', ('mozpercentpreproc',), (), ()),
    'MozPreprocXulLexer': ('pip._vendor.pygments.lexers.markup', 'XUL+mozpreproc', ('xul+mozpreproc',), ('*.xul.in',), ()),
    'MqlLexer': ('pip._vendor.pygments.lexers.c_like', 'MQL', ('mql', 'mq4', 'mq5', 'mql4', 'mql5'), ('*.mq4', '*.mq5', '*.mqh'), ('text/x-mql',)),
    'MscgenLexer': ('pip._vendor.pygments.lexers.dsls', 'Mscgen', ('mscgen', 'msc'), ('*.msc',), ()),
    'MuPADLexer': ('pip._vendor.pygments.lexers.algebra', 'MuPAD', ('mupad',), ('*.mu',), ()),
    'MxmlLexer': ('pip._vendor.pygments.lexers.actionscript', 'MXML', ('mxml',), ('*.mxml',), ()),
    'MySqlLexer': ('pip._vendor.pygments.lexers.sql', 'MySQL', ('mysql',), (), ('text/x-mysql',)),
    'MyghtyCssLexer': ('pip._vendor.pygments.lexers.templates', 'CSS+Myghty', ('css+myghty',), (), ('text/css+myghty',)),
    'MyghtyHtmlLexer': ('pip._vendor.pygments.lexers.templates', 'HTML+Myghty', ('html+myghty',), (), ('text/html+myghty',)),
    'MyghtyJavascriptLexer': ('pip._vendor.pygments.lexers.templates', 'JavaScript+Myghty', ('javascript+myghty', 'js+myghty'), (), ('application/x-javascript+myghty', 'text/x-javascript+myghty', 'text/javascript+mygthy')),
    'MyghtyLexer': ('pip._vendor.pygments.lexers.templates', 'Myghty', ('myghty',), ('*.myt', 'autodelegate'), ('application/x-myghty',)),
    'MyghtyXmlLexer': ('pip._vendor.pygments.lexers.templates', 'XML+Myghty', ('xml+myghty',), (), ('application/xml+myghty',)),
    'NCLLexer': ('pip._vendor.pygments.lexers.ncl', 'NCL', ('ncl',), ('*.ncl',), ('text/ncl',)),
    'NSISLexer': ('pip._vendor.pygments.lexers.installers', 'NSIS', ('nsis', 'nsi', 'nsh'), ('*.nsi', '*.nsh'), ('text/x-nsis',)),
    'NasmLexer': ('pip._vendor.pygments.lexers.asm', 'NASM', ('nasm',), ('*.asm', '*.ASM', '*.nasm'), ('text/x-nasm',)),
    'NasmObjdumpLexer': ('pip._vendor.pygments.lexers.asm', 'objdump-nasm', ('objdump-nasm',), ('*.objdump-intel',), ('text/x-nasm-objdump',)),
    'NemerleLexer': ('pip._vendor.pygments.lexers.dotnet', 'Nemerle', ('nemerle',), ('*.n',), ('text/x-nemerle',)),
    'NesCLexer': ('pip._vendor.pygments.lexers.c_like', 'nesC', ('nesc',), ('*.nc',), ('text/x-nescsrc',)),
    'NestedTextLexer': ('pip._vendor.pygments.lexers.configs', 'NestedText', ('nestedtext', 'nt'), ('*.nt',), ()),
    'NewLispLexer': ('pip._vendor.pygments.lexers.lisp', 'NewLisp', ('newlisp',), ('*.lsp', '*.nl', '*.kif'), ('text/x-newlisp', 'application/x-newlisp')),
    'NewspeakLexer': ('pip._vendor.pygments.lexers.smalltalk', 'Newspeak', ('newspeak',), ('*.ns2',), ('text/x-newspeak',)),
    'NginxConfLexer': ('pip._vendor.pygments.lexers.configs', 'Nginx configuration file', ('nginx',), ('nginx.conf',), ('text/x-nginx-conf',)),
    'NimrodLexer': ('pip._vendor.pygments.lexers.nimrod', 'Nimrod', ('nimrod', 'nim'), ('*.nim', '*.nimrod'), ('text/x-nim',)),
    'NitLexer': ('pip._vendor.pygments.lexers.nit', 'Nit', ('nit',), ('*.nit',), ()),
    'NixLexer': ('pip._vendor.pygments.lexers.nix', 'Nix', ('nixos', 'nix'), ('*.nix',), ('text/x-nix',)),
    'NodeConsoleLexer': ('pip._vendor.pygments.lexers.javascript', 'Node.js REPL console session', ('nodejsrepl',), (), ('text/x-nodejsrepl',)),
    'NotmuchLexer': ('pip._vendor.pygments.lexers.textfmts', 'Notmuch', ('notmuch',), (), ()),
    'NuSMVLexer': ('pip._vendor.pygments.lexers.smv', 'NuSMV', ('nusmv',), ('*.smv',), ()),
    'NumPyLexer': ('pip._vendor.pygments.lexers.python', 'NumPy', ('numpy',), (), ()),
    'NumbaIRLexer': ('pip._vendor.pygments.lexers.numbair', 'Numba_IR', ('numba_ir', 'numbair'), ('*.numba_ir',), ('text/x-numba_ir', 'text/x-numbair')),
    'ObjdumpLexer': ('pip._vendor.pygments.lexers.asm', 'objdump', ('objdump',), ('*.objdump',), ('text/x-objdump',)),
    'ObjectiveCLexer': ('pip._vendor.pygments.lexers.objective', 'Objective-C', ('objective-c', 'objectivec', 'obj-c', 'objc'), ('*.m', '*.h'), ('text/x-objective-c',)),
    'ObjectiveCppLexer': ('pip._vendor.pygments.lexers.objective', 'Objective-C++', ('objective-c++', 'objectivec++', 'obj-c++', 'objc++'), ('*.mm', '*.hh'), ('text/x-objective-c++',)),
    'ObjectiveJLexer': ('pip._vendor.pygments.lexers.javascript', 'Objective-J', ('objective-j', 'objectivej', 'obj-j', 'objj'), ('*.j',), ('text/x-objective-j',)),
    'OcamlLexer': ('pip._vendor.pygments.lexers.ml', 'OCaml', ('ocaml',), ('*.ml', '*.mli', '*.mll', '*.mly'), ('text/x-ocaml',)),
    'OctaveLexer': ('pip._vendor.pygments.lexers.matlab', 'Octave', ('octave',), ('*.m',), ('text/octave',)),
    'OdinLexer': ('pip._vendor.pygments.lexers.archetype', 'ODIN', ('odin',), ('*.odin',), ('text/odin',)),
    'OmgIdlLexer': ('pip._vendor.pygments.lexers.c_like', 'OMG Interface Definition Language', ('omg-idl',), ('*.idl', '*.pidl'), ()),
    'OocLexer': ('pip._vendor.pygments.lexers.ooc', 'Ooc', ('ooc',), ('*.ooc',), ('text/x-ooc',)),
    'OpaLexer': ('pip._vendor.pygments.lexers.ml', 'Opa', ('opa',), ('*.opa',), ('text/x-opa',)),
    'OpenEdgeLexer': ('pip._vendor.pygments.lexers.business', 'OpenEdge ABL', ('openedge', 'abl', 'progress'), ('*.p', '*.cls'), ('text/x-openedge', 'application/x-openedge')),
    'OpenScadLexer': ('pip._vendor.pygments.lexers.openscad', 'OpenSCAD', ('openscad',), ('*.scad',), ('application/x-openscad',)),
    'OrgLexer': ('pip._vendor.pygments.lexers.markup', 'Org Mode', ('org', 'orgmode', 'org-mode'), ('*.org',), ('text/org',)),
    'OutputLexer': ('pip._vendor.pygments.lexers.special', 'Text output', ('output',), (), ()),
    'PacmanConfLexer': ('pip._vendor.pygments.lexers.configs', 'PacmanConf', ('pacmanconf',), ('pacman.conf',), ()),
    'PanLexer': ('pip._vendor.pygments.lexers.dsls', 'Pan', ('pan',), ('*.pan',), ()),
    'ParaSailLexer': ('pip._vendor.pygments.lexers.parasail', 'ParaSail', ('parasail',), ('*.psi', '*.psl'), ('text/x-parasail',)),
    'PawnLexer': ('pip._vendor.pygments.lexers.pawn', 'Pawn', ('pawn',), ('*.p', '*.pwn', '*.inc'), ('text/x-pawn',)),
    'PddlLexer': ('pip._vendor.pygments.lexers.pddl', 'PDDL', ('pddl',), ('*.pddl',), ()),
    'PegLexer': ('pip._vendor.pygments.lexers.grammar_notation', 'PEG', ('peg',), ('*.peg',), ('text/x-peg',)),
    'Perl6Lexer': ('pip._vendor.pygments.lexers.perl', 'Perl6', ('perl6', 'pl6', 'raku'), ('*.pl', '*.pm', '*.nqp', '*.p6', '*.6pl', '*.p6l', '*.pl6', '*.6pm', '*.p6m', '*.pm6', '*.t', '*.raku', '*.rakumod', '*.rakutest', '*.rakudoc'), ('text/x-perl6', 'application/x-perl6')),
    'PerlLexer': ('pip._vendor.pygments.lexers.perl', 'Perl', ('perl', 'pl'), ('*.pl', '*.pm', '*.t', '*.perl'), ('text/x-perl', 'application/x-perl')),
    'PhixLexer': ('pip._vendor.pygments.lexers.phix', 'Phix', ('phix',), ('*.exw',), ('text/x-phix',)),
    'PhpLexer': ('pip._vendor.pygments.lexers.php', 'PHP', ('php', 'php3', 'php4', 'php5'), ('*.php', '*.php[345]', '*.inc'), ('text/x-php',)),
    'PigLexer': ('pip._vendor.pygments.lexers.jvm', 'Pig', ('pig',), ('*.pig',), ('text/x-pig',)),
    'PikeLexer': ('pip._vendor.pygments.lexers.c_like', 'Pike', ('pike',), ('*.pike', '*.pmod'), ('text/x-pike',)),
    'PkgConfigLexer': ('pip._vendor.pygments.lexers.configs', 'PkgConfig', ('pkgconfig',), ('*.pc',), ()),
    'PlPgsqlLexer': ('pip._vendor.pygments.lexers.sql', 'PL/pgSQL', ('plpgsql',), (), ('text/x-plpgsql',)),
    'PointlessLexer': ('pip._vendor.pygments.lexers.pointless', 'Pointless', ('pointless',), ('*.ptls',), ()),
    'PonyLexer': ('pip._vendor.pygments.lexers.pony', 'Pony', ('pony',), ('*.pony',), ()),
    'PortugolLexer': ('pip._vendor.pygments.lexers.pascal', 'Portugol', ('portugol',), ('*.alg', '*.portugol'), ()),
    'PostScriptLexer': ('pip._vendor.pygments.lexers.graphics', 'PostScript', ('postscript', 'postscr'), ('*.ps', '*.eps'), ('application/postscript',)),
    'PostgresConsoleLexer': ('pip._vendor.pygments.lexers.sql', 'PostgreSQL console (psql)', ('psql', 'postgresql-console', 'postgres-console'), (), ('text/x-postgresql-psql',)),
    'PostgresExplainLexer': ('pip._vendor.pygments.lexers.sql', 'PostgreSQL EXPLAIN dialect', ('postgres-explain',), ('*.explain',), ('text/x-postgresql-explain',)),
    'PostgresLexer': ('pip._vendor.pygments.lexers.sql', 'PostgreSQL SQL dialect', ('postgresql', 'postgres'), (), ('text/x-postgresql',)),
    'PovrayLexer': ('pip._vendor.pygments.lexers.graphics', 'POVRay', ('pov',), ('*.pov', '*.inc'), ('text/x-povray',)),
    'PowerShellLexer': ('pip._vendor.pygments.lexers.shell', 'PowerShell', ('powershell', 'pwsh', 'posh', 'ps1', 'psm1'), ('*.ps1', '*.psm1'), ('text/x-powershell',)),
    'PowerShellSessionLexer': ('pip._vendor.pygments.lexers.shell', 'PowerShell Session', ('pwsh-session', 'ps1con'), (), ()),
    'PraatLexer': ('pip._vendor.pygments.lexers.praat', 'Praat', ('praat',), ('*.praat', '*.proc', '*.psc'), ()),
    'ProcfileLexer': ('pip._vendor.pygments.lexers.procfile', 'Procfile', ('procfile',), ('Procfile',), ()),
    'PrologLexer': ('pip._vendor.pygments.lexers.prolog', 'Prolog', ('prolog',), ('*.ecl', '*.prolog', '*.pro', '*.pl'), ('text/x-prolog',)),
    'PromQLLexer': ('pip._vendor.pygments.lexers.promql', 'PromQL', ('promql',), ('*.promql',), ()),
    'PromelaLexer': ('pip._vendor.pygments.lexers.c_like', 'Promela', ('promela',), ('*.pml', '*.prom', '*.prm', '*.promela', '*.pr', '*.pm'), ('text/x-promela',)),
    'PropertiesLexer': ('pip._vendor.pygments.lexers.configs', 'Properties', ('properties', 'jproperties'), ('*.properties',), ('text/x-java-properties',)),
    'ProtoBufLexer': ('pip._vendor.pygments.lexers.dsls', 'Protocol Buffer', ('protobuf', 'proto'), ('*.proto',), ()),
    'PrqlLexer': ('pip._vendor.pygments.lexers.prql', 'PRQL', ('prql',), ('*.prql',), ('application/prql', 'application/x-prql')),
    'PsyshConsoleLexer': ('pip._vendor.pygments.lexers.php', 'PsySH console session for PHP', ('psysh',), (), ()),
    'PtxLexer': ('pip._vendor.pygments.lexers.ptx', 'PTX', ('ptx',), ('*.ptx',), ('text/x-ptx',)),
    'PugLexer': ('pip._vendor.pygments.lexers.html', 'Pug', ('pug', 'jade'), ('*.pug', '*.jade'), ('text/x-pug', 'text/x-jade')),
    'PuppetLexer': ('pip._vendor.pygments.lexers.dsls', 'Puppet', ('puppet',), ('*.pp',), ()),
    'PyPyLogLexer': ('pip._vendor.pygments.lexers.console', 'PyPy Log', ('pypylog', 'pypy'), ('*.pypylog',), ('application/x-pypylog',)),
    'Python2Lexer': ('pip._vendor.pygments.lexers.python', 'Python 2.x', ('python2', 'py2'), (), ('text/x-python2', 'application/x-python2')),
    'Python2TracebackLexer': ('pip._vendor.pygments.lexers.python', 'Python 2.x Traceback', ('py2tb',), ('*.py2tb',), ('text/x-python2-traceback',)),
    'PythonConsoleLexer': ('pip._vendor.pygments.lexers.python', 'Python console session', ('pycon', 'python-console'), (), ('text/x-python-doctest',)),
    'PythonLexer': ('pip._vendor.pygments.lexers.python', 'Python', ('python', 'py', 'sage', 'python3', 'py3', 'bazel', 'starlark', 'pyi'), ('*.py', '*.pyw', '*.pyi', '*.jy', '*.sage', '*.sc', 'SConstruct', 'SConscript', '*.bzl', 'BUCK', 'BUILD', 'BUILD.bazel', 'WORKSPACE', '*.tac'), ('text/x-python', 'application/x-python', 'text/x-python3', 'application/x-python3')),
    'PythonTracebackLexer': ('pip._vendor.pygments.lexers.python', 'Python Traceback', ('pytb', 'py3tb'), ('*.pytb', '*.py3tb'), ('text/x-python-traceback', 'text/x-python3-traceback')),
    'PythonUL4Lexer': ('pip._vendor.pygments.lexers.ul4', 'Python+UL4', ('py+ul4',), ('*.pyul4',), ()),
    'QBasicLexer': ('pip._vendor.pygments.lexers.basic', 'QBasic', ('qbasic', 'basic'), ('*.BAS', '*.bas'), ('text/basic',)),
    'QLexer': ('pip._vendor.pygments.lexers.q', 'Q', ('q',), ('*.q',), ()),
    'QVToLexer': ('pip._vendor.pygments.lexers.qvt', 'QVTO', ('qvto', 'qvt'), ('*.qvto',), ()),
    'QlikLexer': ('pip._vendor.pygments.lexers.qlik', 'Qlik', ('qlik', 'qlikview', 'qliksense', 'qlikscript'), ('*.qvs', '*.qvw'), ()),
    'QmlLexer': ('pip._vendor.pygments.lexers.webmisc', 'QML', ('qml', 'qbs'), ('*.qml', '*.qbs'), ('application/x-qml', 'application/x-qt.qbs+qml')),
    'RConsoleLexer': ('pip._vendor.pygments.lexers.r', 'RConsole', ('rconsole', 'rout'), ('*.Rout',), ()),
    'RNCCompactLexer': ('pip._vendor.pygments.lexers.rnc', 'Relax-NG Compact', ('rng-compact', 'rnc'), ('*.rnc',), ()),
    'RPMSpecLexer': ('pip._vendor.pygments.lexers.installers', 'RPMSpec', ('spec',), ('*.spec',), ('text/x-rpm-spec',)),
    'RacketLexer': ('pip._vendor.pygments.lexers.lisp', 'Racket', ('racket', 'rkt'), ('*.rkt', '*.rktd', '*.rktl'), ('text/x-racket', 'application/x-racket')),
    'RagelCLexer': ('pip._vendor.pygments.lexers.parsers', 'Ragel in C Host', ('ragel-c',), ('*.rl',), ()),
    'RagelCppLexer': ('pip._vendor.pygments.lexers.parsers', 'Ragel in CPP Host', ('ragel-cpp',), ('*.rl',), ()),
    'RagelDLexer': ('pip._vendor.pygments.lexers.parsers', 'Ragel in D Host', ('ragel-d',), ('*.rl',), ()),
    'RagelEmbeddedLexer': ('pip._vendor.pygments.lexers.parsers', 'Embedded Ragel', ('ragel-em',), ('*.rl',), ()),
    'RagelJavaLexer': ('pip._vendor.pygments.lexers.parsers', 'Ragel in Java Host', ('ragel-java',), ('*.rl',), ()),
    'RagelLexer': ('pip._vendor.pygments.lexers.parsers', 'Ragel', ('ragel',), (), ()),
    'RagelObjectiveCLexer': ('pip._vendor.pygments.lexers.parsers', 'Ragel in Objective C Host', ('ragel-objc',), ('*.rl',), ()),
    'RagelRubyLexer': ('pip._vendor.pygments.lexers.parsers', 'Ragel in Ruby Host', ('ragel-ruby', 'ragel-rb'), ('*.rl',), ()),
    'RawTokenLexer': ('pip._vendor.pygments.lexers.special', 'Raw token data', (), (), ('application/x-pygments-tokens',)),
    'RdLexer': ('pip._vendor.pygments.lexers.r', 'Rd', ('rd',), ('*.Rd',), ('text/x-r-doc',)),
    'ReasonLexer': ('pip._vendor.pygments.lexers.ml', 'ReasonML', ('reasonml', 'reason'), ('*.re', '*.rei'), ('text/x-reasonml',)),
    'RebolLexer': ('pip._vendor.pygments.lexers.rebol', 'REBOL', ('rebol',), ('*.r', '*.r3', '*.reb'), ('text/x-rebol',)),
    'RedLexer': ('pip._vendor.pygments.lexers.rebol', 'Red', ('red', 'red/system'), ('*.red', '*.reds'), ('text/x-red', 'text/x-red-system')),
    'RedcodeLexer': ('pip._vendor.pygments.lexers.esoteric', 'Redcode', ('redcode',), ('*.cw',), ()),
    'RegeditLexer': ('pip._vendor.pygments.lexers.configs', 'reg', ('registry',), ('*.reg',), ('text/x-windows-registry',)),
    'RegoLexer': ('pip._vendor.pygments.lexers.rego', 'Rego', ('rego',), ('*.rego',), ('text/x-rego',)),
    'ResourceLexer': ('pip._vendor.pygments.lexers.resource', 'ResourceBundle', ('resourcebundle', 'resource'), (), ()),
    'RexxLexer': ('pip._vendor.pygments.lexers.scripting', 'Rexx', ('rexx', 'arexx'), ('*.rexx', '*.rex', '*.rx', '*.arexx'), ('text/x-rexx',)),
    'RhtmlLexer': ('pip._vendor.pygments.lexers.templates', 'RHTML', ('rhtml', 'html+erb', 'html+ruby'), ('*.rhtml',), ('text/html+ruby',)),
    'RideLexer': ('pip._vendor.pygments.lexers.ride', 'Ride', ('ride',), ('*.ride',), ('text/x-ride',)),
    'RitaLexer': ('pip._vendor.pygments.lexers.rita', 'Rita', ('rita',), ('*.rita',), ('text/rita',)),
    'RoboconfGraphLexer': ('pip._vendor.pygments.lexers.roboconf', 'Roboconf Graph', ('roboconf-graph',), ('*.graph',), ()),
    'RoboconfInstancesLexer': ('pip._vendor.pygments.lexers.roboconf', 'Roboconf Instances', ('roboconf-instances',), ('*.instances',), ()),
    'RobotFrameworkLexer': ('pip._vendor.pygments.lexers.robotframework', 'RobotFramework', ('robotframework',), ('*.robot', '*.resource'), ('text/x-robotframework',)),
    'RqlLexer': ('pip._vendor.pygments.lexers.sql', 'RQL', ('rql',), ('*.rql',), ('text/x-rql',)),
    'RslLexer': ('pip._vendor.pygments.lexers.dsls', 'RSL', ('rsl',), ('*.rsl',), ('text/rsl',)),
    'RstLexer': ('pip._vendor.pygments.lexers.markup', 'reStructuredText', ('restructuredtext', 'rst', 'rest'), ('*.rst', '*.rest'), ('text/x-rst', 'text/prs.fallenstein.rst')),
    'RtsLexer': ('pip._vendor.pygments.lexers.trafficscript', 'TrafficScript', ('trafficscript', 'rts'), ('*.rts',), ()),
    'RubyConsoleLexer': ('pip._vendor.pygments.lexers.ruby', 'Ruby irb session', ('rbcon', 'irb'), (), ('text/x-ruby-shellsession',)),
    'RubyLexer': ('pip._vendor.pygments.lexers.ruby', 'Ruby', ('ruby', 'rb', 'duby'), ('*.rb', '*.rbw', 'Rakefile', '*.rake', '*.gemspec', '*.rbx', '*.duby', 'Gemfile', 'Vagrantfile'), ('text/x-ruby', 'application/x-ruby')),
    'RustLexer': ('pip._vendor.pygments.lexers.rust', 'Rust', ('rust', 'rs'), ('*.rs', '*.rs.in'), ('text/rust', 'text/x-rust')),
    'SASLexer': ('pip._vendor.pygments.lexers.sas', 'SAS', ('sas',), ('*.SAS', '*.sas'), ('text/x-sas', 'text/sas', 'application/x-sas')),
    'SLexer': ('pip._vendor.pygments.lexers.r', 'S', ('splus', 's', 'r'), ('*.S', '*.R', '.Rhistory', '.Rprofile', '.Renviron'), ('text/S-plus', 'text/S', 'text/x-r-source', 'text/x-r', 'text/x-R', 'text/x-r-history', 'text/x-r-profile')),
    'SMLLexer': ('pip._vendor.pygments.lexers.ml', 'Standard ML', ('sml',), ('*.sml', '*.sig', '*.fun'), ('text/x-standardml', 'application/x-standardml')),
    'SNBTLexer': ('pip._vendor.pygments.lexers.minecraft', 'SNBT', ('snbt',), ('*.snbt',), ('text/snbt',)),
    'SarlLexer': ('pip._vendor.pygments.lexers.jvm', 'SARL', ('sarl',), ('*.sarl',), ('text/x-sarl',)),
    'SassLexer': ('pip._vendor.pygments.lexers.css', 'Sass', ('sass',), ('*.sass',), ('text/x-sass',)),
    'SaviLexer': ('pip._vendor.pygments.lexers.savi', 'Savi', ('savi',), ('*.savi',), ()),
    'ScalaLexer': ('pip._vendor.pygments.lexers.jvm', 'Scala', ('scala',), ('*.scala',), ('text/x-scala',)),
    'ScamlLexer': ('pip._vendor.pygments.lexers.html', 'Scaml', ('scaml',), ('*.scaml',), ('text/x-scaml',)),
    'ScdocLexer': ('pip._vendor.pygments.lexers.scdoc', 'scdoc', ('scdoc', 'scd'), ('*.scd', '*.scdoc'), ()),
    'SchemeLexer': ('pip._vendor.pygments.lexers.lisp', 'Scheme', ('scheme', 'scm'), ('*.scm', '*.ss'), ('text/x-scheme', 'application/x-scheme')),
    'ScilabLexer': ('pip._vendor.pygments.lexers.matlab', 'Scilab', ('scilab',), ('*.sci', '*.sce', '*.tst'), ('text/scilab',)),
    'ScssLexer': ('pip._vendor.pygments.lexers.css', 'SCSS', ('scss',), ('*.scss',), ('text/x-scss',)),
    'SedLexer': ('pip._vendor.pygments.lexers.textedit', 'Sed', ('sed', 'gsed', 'ssed'), ('*.sed', '*.[gs]sed'), ('text/x-sed',)),
    'ShExCLexer': ('pip._vendor.pygments.lexers.rdf', 'ShExC', ('shexc', 'shex'), ('*.shex',), ('text/shex',)),
    'ShenLexer': ('pip._vendor.pygments.lexers.lisp', 'Shen', ('shen',), ('*.shen',), ('text/x-shen', 'application/x-shen')),
    'SieveLexer': ('pip._vendor.pygments.lexers.sieve', 'Sieve', ('sieve',), ('*.siv', '*.sieve'), ()),
    'SilverLexer': ('pip._vendor.pygments.lexers.verification', 'Silver', ('silver',), ('*.sil', '*.vpr'), ()),
    'SingularityLexer': ('pip._vendor.pygments.lexers.configs', 'Singularity', ('singularity',), ('*.def', 'Singularity'), ()),
    'SlashLexer': ('pip._vendor.pygments.lexers.slash', 'Slash', ('slash',), ('*.sla',), ()),
    'SlimLexer': ('pip._vendor.pygments.lexers.webmisc', 'Slim', ('slim',), ('*.slim',), ('text/x-slim',)),
    'SlurmBashLexer': ('pip._vendor.pygments.lexers.shell', 'Slurm', ('slurm', 'sbatch'), ('*.sl',), ()),
    'SmaliLexer': ('pip._vendor.pygments.lexers.dalvik', 'Smali', ('smali',), ('*.smali',), ('text/smali',)),
    'SmalltalkLexer': ('pip._vendor.pygments.lexers.smalltalk', 'Smalltalk', ('smalltalk', 'squeak', 'st'), ('*.st',), ('text/x-smalltalk',)),
    'SmartGameFormatLexer': ('pip._vendor.pygments.lexers.sgf', 'SmartGameFormat', ('sgf',), ('*.sgf',), ()),
    'SmartyLexer': ('pip._vendor.pygments.lexers.templates', 'Smarty', ('smarty',), ('*.tpl',), ('application/x-smarty',)),
    'SmithyLexer': ('pip._vendor.pygments.lexers.smithy', 'Smithy', ('smithy',), ('*.smithy',), ()),
    'SnobolLexer': ('pip._vendor.pygments.lexers.snobol', 'Snobol', ('snobol',), ('*.snobol',), ('text/x-snobol',)),
    'SnowballLexer': ('pip._vendor.pygments.lexers.dsls', 'Snowball', ('snowball',), ('*.sbl',), ()),
    'SolidityLexer': ('pip._vendor.pygments.lexers.solidity', 'Solidity', ('solidity',), ('*.sol',), ()),
    'SoongLexer': ('pip._vendor.pygments.lexers.soong', 'Soong', ('androidbp', 'bp', 'soong'), ('Android.bp',), ()),
    'SophiaLexer': ('pip._vendor.pygments.lexers.sophia', 'Sophia', ('sophia',), ('*.aes',), ()),
    'SourcePawnLexer': ('pip._vendor.pygments.lexers.pawn', 'SourcePawn', ('sp',), ('*.sp',), ('text/x-sourcepawn',)),
    'SourcesListLexer': ('pip._vendor.pygments.lexers.installers', 'Debian Sourcelist', ('debsources', 'sourceslist', 'sources.list'), ('sources.list',), ()),
    'SparqlLexer': ('pip._vendor.pygments.lexers.rdf', 'SPARQL', ('sparql',), ('*.rq', '*.sparql'), ('application/sparql-query',)),
    'SpiceLexer': ('pip._vendor.pygments.lexers.spice', 'Spice', ('spice', 'spicelang'), ('*.spice',), ('text/x-spice',)),
    'SqlJinjaLexer': ('pip._vendor.pygments.lexers.templates', 'SQL+Jinja', ('sql+jinja',), ('*.sql', '*.sql.j2', '*.sql.jinja2'), ()),
    'SqlLexer': ('pip._vendor.pygments.lexers.sql', 'SQL', ('sql',), ('*.sql',), ('text/x-sql',)),
    'SqliteConsoleLexer': ('pip._vendor.pygments.lexers.sql', 'sqlite3con', ('sqlite3',), ('*.sqlite3-console',), ('text/x-sqlite3-console',)),
    'SquidConfLexer': ('pip._vendor.pygments.lexers.configs', 'SquidConf', ('squidconf', 'squid.conf', 'squid'), ('squid.conf',), ('text/x-squidconf',)),
    'SrcinfoLexer': ('pip._vendor.pygments.lexers.srcinfo', 'Srcinfo', ('srcinfo',), ('.SRCINFO',), ()),
    'SspLexer': ('pip._vendor.pygments.lexers.templates', 'Scalate Server Page', ('ssp',), ('*.ssp',), ('application/x-ssp',)),
    'StanLexer': ('pip._vendor.pygments.lexers.modeling', 'Stan', ('stan',), ('*.stan',), ()),
    'StataLexer': ('pip._vendor.pygments.lexers.stata', 'Stata', ('stata', 'do'), ('*.do', '*.ado'), ('text/x-stata', 'text/stata', 'application/x-stata')),
    'SuperColliderLexer': ('pip._vendor.pygments.lexers.supercollider', 'SuperCollider', ('supercollider', 'sc'), ('*.sc', '*.scd'), ('application/supercollider', 'text/supercollider')),
    'SwiftLexer': ('pip._vendor.pygments.lexers.objective', 'Swift', ('swift',), ('*.swift',), ('text/x-swift',)),
    'SwigLexer': ('pip._vendor.pygments.lexers.c_like', 'SWIG', ('swig',), ('*.swg', '*.i'), ('text/swig',)),
    'SystemVerilogLexer': ('pip._vendor.pygments.lexers.hdl', 'systemverilog', ('systemverilog', 'sv'), ('*.sv', '*.svh'), ('text/x-systemverilog',)),
    'SystemdLexer': ('pip._vendor.pygments.lexers.configs', 'Systemd', ('systemd',), ('*.service', '*.socket', '*.device', '*.mount', '*.automount', '*.swap', '*.target', '*.path', '*.timer', '*.slice', '*.scope'), ()),
    'TAPLexer': ('pip._vendor.pygments.lexers.testing', 'TAP', ('tap',), ('*.tap',), ()),
    'TNTLexer': ('pip._vendor.pygments.lexers.tnt', 'Typographic Number Theory', ('tnt',), ('*.tnt',), ()),
    'TOMLLexer': ('pip._vendor.pygments.lexers.configs', 'TOML', ('toml',), ('*.toml', 'Pipfile', 'poetry.lock'), ('application/toml',)),
    'TableGenLexer': ('pip._vendor.pygments.lexers.tablegen', 'TableGen', ('tablegen', 'td'), ('*.td',), ()),
    'TactLexer': ('pip._vendor.pygments.lexers.tact', 'Tact', ('tact',), ('*.tact',), ()),
    'Tads3Lexer': ('pip._vendor.pygments.lexers.int_fiction', 'TADS 3', ('tads3',), ('*.t',), ()),
    'TalLexer': ('pip._vendor.pygments.lexers.tal', 'Tal', ('tal', 'uxntal'), ('*.tal',), ('text/x-uxntal',)),
    'TasmLexer': ('pip._vendor.pygments.lexers.asm', 'TASM', ('tasm',), ('*.asm', '*.ASM', '*.tasm'), ('text/x-tasm',)),
    'TclLexer': ('pip._vendor.pygments.lexers.tcl', 'Tcl', ('tcl',), ('*.tcl', '*.rvt'), ('text/x-tcl', 'text/x-script.tcl', 'application/x-tcl')),
    'TcshLexer': ('pip._vendor.pygments.lexers.shell', 'Tcsh', ('tcsh', 'csh'), ('*.tcsh', '*.csh'), ('application/x-csh',)),
    'TcshSessionLexer': ('pip._vendor.pygments.lexers.shell', 'Tcsh Session', ('tcshcon',), (), ()),
    'TeaTemplateLexer': ('pip._vendor.pygments.lexers.templates', 'Tea', ('tea',), ('*.tea',), ('text/x-tea',)),
    'TealLexer': ('pip._vendor.pygments.lexers.teal', 'teal', ('teal',), ('*.teal',), ()),
    'TeraTermLexer': ('pip._vendor.pygments.lexers.teraterm', 'Tera Term macro', ('teratermmacro', 'teraterm', 'ttl'), ('*.ttl',), ('text/x-teratermmacro',)),
    'TermcapLexer': ('pip._vendor.pygments.lexers.configs', 'Termcap', ('termcap',), ('termcap', 'termcap.src'), ()),
    'TerminfoLexer': ('pip._vendor.pygments.lexers.configs', 'Terminfo', ('terminfo',), ('terminfo', 'terminfo.src'), ()),
    'TerraformLexer': ('pip._vendor.pygments.lexers.configs', 'Terraform', ('terraform', 'tf', 'hcl'), ('*.tf', '*.hcl'), ('application/x-tf', 'application/x-terraform')),
    'TexLexer': ('pip._vendor.pygments.lexers.markup', 'TeX', ('tex', 'latex'), ('*.tex', '*.aux', '*.toc'), ('text/x-tex', 'text/x-latex')),
    'TextLexer': ('pip._vendor.pygments.lexers.special', 'Text only', ('text',), ('*.txt',), ('text/plain',)),
    'ThingsDBLexer': ('pip._vendor.pygments.lexers.thingsdb', 'ThingsDB', ('ti', 'thingsdb'), ('*.ti',), ()),
    'ThriftLexer': ('pip._vendor.pygments.lexers.dsls', 'Thrift', ('thrift',), ('*.thrift',), ('application/x-thrift',)),
    'TiddlyWiki5Lexer': ('pip._vendor.pygments.lexers.markup', 'tiddler', ('tid',), ('*.tid',), ('text/vnd.tiddlywiki',)),
    'TlbLexer': ('pip._vendor.pygments.lexers.tlb', 'Tl-b', ('tlb',), ('*.tlb',), ()),
    'TlsLexer': ('pip._vendor.pygments.lexers.tls', 'TLS Presentation Language', ('tls',), (), ()),
    'TodotxtLexer': ('pip._vendor.pygments.lexers.textfmts', 'Todotxt', ('todotxt',), ('todo.txt', '*.todotxt'), ('text/x-todo',)),
    'TransactSqlLexer': ('pip._vendor.pygments.lexers.sql', 'Transact-SQL', ('tsql', 't-sql'), ('*.sql',), ('text/x-tsql',)),
    'TreetopLexer': ('pip._vendor.pygments.lexers.parsers', 'Treetop', ('treetop',), ('*.treetop', '*.tt'), ()),
    'TsxLexer': ('pip._vendor.pygments.lexers.jsx', 'TSX', ('tsx',), ('*.tsx',), ('text/typescript-tsx',)),
    'TurtleLexer': ('pip._vendor.pygments.lexers.rdf', 'Turtle', ('turtle',), ('*.ttl',), ('text/turtle', 'application/x-turtle')),
    'TwigHtmlLexer': ('pip._vendor.pygments.lexers.templates', 'HTML+Twig', ('html+twig',), ('*.twig',), ('text/html+twig',)),
    'TwigLexer': ('pip._vendor.pygments.lexers.templates', 'Twig', ('twig',), (), ('application/x-twig',)),
    'TypeScriptLexer': ('pip._vendor.pygments.lexers.javascript', 'TypeScript', ('typescript', 'ts'), ('*.ts',), ('application/x-typescript', 'text/x-typescript')),
    'TypoScriptCssDataLexer': ('pip._vendor.pygments.lexers.typoscript', 'TypoScriptCssData', ('typoscriptcssdata',), (), ()),
    'TypoScriptHtmlDataLexer': ('pip._vendor.pygments.lexers.typoscript', 'TypoScriptHtmlData', ('typoscripthtmldata',), (), ()),
    'TypoScriptLexer': ('pip._vendor.pygments.lexers.typoscript', 'TypoScript', ('typoscript',), ('*.typoscript',), ('text/x-typoscript',)),
    'TypstLexer': ('pip._vendor.pygments.lexers.typst', 'Typst', ('typst',), ('*.typ',), ('text/x-typst',)),
    'UL4Lexer': ('pip._vendor.pygments.lexers.ul4', 'UL4', ('ul4',), ('*.ul4',), ()),
    'UcodeLexer': ('pip._vendor.pygments.lexers.unicon', 'ucode', ('ucode',), ('*.u', '*.u1', '*.u2'), ()),
    'UniconLexer': ('pip._vendor.pygments.lexers.unicon', 'Unicon', ('unicon',), ('*.icn',), ('text/unicon',)),
    'UnixConfigLexer': ('pip._vendor.pygments.lexers.configs', 'Unix/Linux config files', ('unixconfig', 'linuxconfig'), (), ()),
    'UrbiscriptLexer': ('pip._vendor.pygments.lexers.urbi', 'UrbiScript', ('urbiscript',), ('*.u',), ('application/x-urbiscript',)),
    'UrlEncodedLexer': ('pip._vendor.pygments.lexers.html', 'urlencoded', ('urlencoded',), (), ('application/x-www-form-urlencoded',)),
    'UsdLexer': ('pip._vendor.pygments.lexers.usd', 'USD', ('usd', 'usda'), ('*.usd', '*.usda'), ()),
    'VBScriptLexer': ('pip._vendor.pygments.lexers.basic', 'VBScript', ('vbscript',), ('*.vbs', '*.VBS'), ()),
    'VCLLexer': ('pip._vendor.pygments.lexers.varnish', 'VCL', ('vcl',), ('*.vcl',), ('text/x-vclsrc',)),
    'VCLSnippetLexer': ('pip._vendor.pygments.lexers.varnish', 'VCLSnippets', ('vclsnippets', 'vclsnippet'), (), ('text/x-vclsnippet',)),
    'VCTreeStatusLexer': ('pip._vendor.pygments.lexers.console', 'VCTreeStatus', ('vctreestatus',), (), ()),
    'VGLLexer': ('pip._vendor.pygments.lexers.dsls', 'VGL', ('vgl',), ('*.rpf',), ()),
    'ValaLexer': ('pip._vendor.pygments.lexers.c_like', 'Vala', ('vala', 'vapi'), ('*.vala', '*.vapi'), ('text/x-vala',)),
    'VbNetAspxLexer': ('pip._vendor.pygments.lexers.dotnet', 'aspx-vb', ('aspx-vb',), ('*.aspx', '*.asax', '*.ascx', '*.ashx', '*.asmx', '*.axd'), ()),
    'VbNetLexer': ('pip._vendor.pygments.lexers.dotnet', 'VB.net', ('vb.net', 'vbnet', 'lobas', 'oobas', 'sobas', 'visual-basic', 'visualbasic'), ('*.vb', '*.bas'), ('text/x-vbnet', 'text/x-vba')),
    'VelocityHtmlLexer': ('pip._vendor.pygments.lexers.templates', 'HTML+Velocity', ('html+velocity',), (), ('text/html+velocity',)),
    'VelocityLexer': ('pip._vendor.pygments.lexers.templates', 'Velocity', ('velocity',), ('*.vm', '*.fhtml'), ()),
    'VelocityXmlLexer': ('pip._vendor.pygments.lexers.templates', 'XML+Velocity', ('xml+velocity',), (), ('application/xml+velocity',)),
    'VerifpalLexer': ('pip._vendor.pygments.lexers.verifpal', 'Verifpal', ('verifpal',), ('*.vp',), ('text/x-verifpal',)),
    'VerilogLexer': ('pip._vendor.pygments.lexers.hdl', 'verilog', ('verilog', 'v'), ('*.v',), ('text/x-verilog',)),
    'VhdlLexer': ('pip._vendor.pygments.lexers.hdl', 'vhdl', ('vhdl',), ('*.vhdl', '*.vhd'), ('text/x-vhdl',)),
    'VimLexer': ('pip._vendor.pygments.lexers.textedit', 'VimL', ('vim',), ('*.vim', '.vimrc', '.exrc', '.gvimrc', '_vimrc', '_exrc', '_gvimrc', 'vimrc', 'gvimrc'), ('text/x-vim',)),
    'VisualPrologGrammarLexer': ('pip._vendor.pygments.lexers.vip', 'Visual Prolog Grammar', ('visualprologgrammar',), ('*.vipgrm',), ()),
    'VisualPrologLexer': ('pip._vendor.pygments.lexers.vip', 'Visual Prolog', ('visualprolog',), ('*.pro', '*.cl', '*.i', '*.pack', '*.ph'), ()),
    'VueLexer': ('pip._vendor.pygments.lexers.html', 'Vue', ('vue',), ('*.vue',), ()),
    'VyperLexer': ('pip._vendor.pygments.lexers.vyper', 'Vyper', ('vyper',), ('*.vy',), ()),
    'WDiffLexer': ('pip._vendor.pygments.lexers.diff', 'WDiff', ('wdiff',), ('*.wdiff',), ()),
    'WatLexer': ('pip._vendor.pygments.lexers.webassembly', 'WebAssembly', ('wast', 'wat'), ('*.wat', '*.wast'), ()),
    'WebIDLLexer': ('pip._vendor.pygments.lexers.webidl', 'Web IDL', ('webidl',), ('*.webidl',), ()),
    'WgslLexer': ('pip._vendor.pygments.lexers.wgsl', 'WebGPU Shading Language', ('wgsl',), ('*.wgsl',), ('text/wgsl',)),
    'WhileyLexer': ('pip._vendor.pygments.lexers.whiley', 'Whiley', ('whiley',), ('*.whiley',), ('text/x-whiley',)),
    'WikitextLexer': ('pip._vendor.pygments.lexers.markup', 'Wikitext', ('wikitext', 'mediawiki'), (), ('text/x-wiki',)),
    'WoWTocLexer': ('pip._vendor.pygments.lexers.wowtoc', 'World of Warcraft TOC', ('wowtoc',), ('*.toc',), ()),
    'WrenLexer': ('pip._vendor.pygments.lexers.wren', 'Wren', ('wren',), ('*.wren',), ()),
    'X10Lexer': ('pip._vendor.pygments.lexers.x10', 'X10', ('x10', 'xten'), ('*.x10',), ('text/x-x10',)),
    'XMLUL4Lexer': ('pip._vendor.pygments.lexers.ul4', 'XML+UL4', ('xml+ul4',), ('*.xmlul4',), ()),
    'XQueryLexer': ('pip._vendor.pygments.lexers.webmisc', 'XQuery', ('xquery', 'xqy', 'xq', 'xql', 'xqm'), ('*.xqy', '*.xquery', '*.xq', '*.xql', '*.xqm'), ('text/xquery', 'application/xquery')),
    'XmlDjangoLexer': ('pip._vendor.pygments.lexers.templates', 'XML+Django/Jinja', ('xml+django', 'xml+jinja'), ('*.xml.j2', '*.xml.jinja2'), ('application/xml+django', 'application/xml+jinja')),
    'XmlErbLexer': ('pip._vendor.pygments.lexers.templates', 'XML+Ruby', ('xml+ruby', 'xml+erb'), (), ('application/xml+ruby',)),
    'XmlLexer': ('pip._vendor.pygments.lexers.html', 'XML', ('xml',), ('*.xml', '*.xsl', '*.rss', '*.xslt', '*.xsd', '*.wsdl', '*.wsf'), ('text/xml', 'application/xml', 'image/svg+xml', 'application/rss+xml', 'application/atom+xml')),
    'XmlPhpLexer': ('pip._vendor.pygments.lexers.templates', 'XML+PHP', ('xml+php',), (), ('application/xml+php',)),
    'XmlSmartyLexer': ('pip._vendor.pygments.lexers.templates', 'XML+Smarty', ('xml+smarty',), (), ('application/xml+smarty',)),
    'XorgLexer': ('pip._vendor.pygments.lexers.xorg', 'Xorg', ('xorg.conf',), ('xorg.conf',), ()),
    'XppLexer': ('pip._vendor.pygments.lexers.dotnet', 'X++', ('xpp', 'x++'), ('*.xpp',), ()),
    'XsltLexer': ('pip._vendor.pygments.lexers.html', 'XSLT', ('xslt',), ('*.xsl', '*.xslt', '*.xpl'), ('application/xsl+xml', 'application/xslt+xml')),
    'XtendLexer': ('pip._vendor.pygments.lexers.jvm', 'Xtend', ('xtend',), ('*.xtend',), ('text/x-xtend',)),
    'XtlangLexer': ('pip._vendor.pygments.lexers.lisp', 'xtlang', ('extempore',), ('*.xtm',), ()),
    'YamlJinjaLexer': ('pip._vendor.pygments.lexers.templates', 'YAML+Jinja', ('yaml+jinja', 'salt', 'sls'), ('*.sls', '*.yaml.j2', '*.yml.j2', '*.yaml.jinja2', '*.yml.jinja2'), ('text/x-yaml+jinja', 'text/x-sls')),
    'YamlLexer': ('pip._vendor.pygments.lexers.data', 'YAML', ('yaml',), ('*.yaml', '*.yml'), ('text/x-yaml',)),
    'YangLexer': ('pip._vendor.pygments.lexers.yang', 'YANG', ('yang',), ('*.yang',), ('application/yang',)),
    'YaraLexer': ('pip._vendor.pygments.lexers.yara', 'YARA', ('yara', 'yar'), ('*.yar',), ('text/x-yara',)),
    'ZeekLexer': ('pip._vendor.pygments.lexers.dsls', 'Zeek', ('zeek', 'bro'), ('*.zeek', '*.bro'), ()),
    'ZephirLexer': ('pip._vendor.pygments.lexers.php', 'Zephir', ('zephir',), ('*.zep',), ()),
    'ZigLexer': ('pip._vendor.pygments.lexers.zig', 'Zig', ('zig',), ('*.zig',), ('text/zig',)),
    'apdlexer': ('pip._vendor.pygments.lexers.apdlexer', 'ANSYS parametric design language', ('ansys', 'apdl'), ('*.ans',), ()),
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\ndim_array.py ===
from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.expr import Expr
from sympy.core.kind import Kind, NumberKind, UndefinedKind
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.printing.defaults import Printable

import itertools
from collections.abc import Iterable


class ArrayKind(Kind):
    """
    Kind for N-dimensional array in SymPy.

    This kind represents the multidimensional array that algebraic
    operations are defined. Basic class for this kind is ``NDimArray``,
    but any expression representing the array can have this.

    Parameters
    ==========

    element_kind : Kind
        Kind of the element. Default is :obj:NumberKind `<sympy.core.kind.NumberKind>`,
        which means that the array contains only numbers.

    Examples
    ========

    Any instance of array class has ``ArrayKind``.

    >>> from sympy import NDimArray
    >>> NDimArray([1,2,3]).kind
    ArrayKind(NumberKind)

    Although expressions representing an array may be not instance of
    array class, it will have ``ArrayKind`` as well.

    >>> from sympy import Integral
    >>> from sympy.tensor.array import NDimArray
    >>> from sympy.abc import x
    >>> intA = Integral(NDimArray([1,2,3]), x)
    >>> isinstance(intA, NDimArray)
    False
    >>> intA.kind
    ArrayKind(NumberKind)

    Use ``isinstance()`` to check for ``ArrayKind` without specifying
    the element kind. Use ``is`` with specifying the element kind.

    >>> from sympy.tensor.array import ArrayKind
    >>> from sympy.core import NumberKind
    >>> boolA = NDimArray([True, False])
    >>> isinstance(boolA.kind, ArrayKind)
    True
    >>> boolA.kind is ArrayKind(NumberKind)
    False

    See Also
    ========

    shape : Function to return the shape of objects with ``MatrixKind``.

    """
    def __new__(cls, element_kind=NumberKind):
        obj = super().__new__(cls, element_kind)
        obj.element_kind = element_kind
        return obj

    def __repr__(self):
        return "ArrayKind(%s)" % self.element_kind

    @classmethod
    def _union(cls, kinds) -> 'ArrayKind':
        elem_kinds = {e.kind for e in kinds}
        if len(elem_kinds) == 1:
            elemkind, = elem_kinds
        else:
            elemkind = UndefinedKind
        return ArrayKind(elemkind)


class NDimArray(Printable):
    """N-dimensional array.

    Examples
    ========

    Create an N-dim array of zeros:

    >>> from sympy import MutableDenseNDimArray
    >>> a = MutableDenseNDimArray.zeros(2, 3, 4)
    >>> a
    [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]

    Create an N-dim array from a list;

    >>> a = MutableDenseNDimArray([[2, 3], [4, 5]])
    >>> a
    [[2, 3], [4, 5]]

    >>> b = MutableDenseNDimArray([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])
    >>> b
    [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]

    Create an N-dim array from a flat list with dimension shape:

    >>> a = MutableDenseNDimArray([1, 2, 3, 4, 5, 6], (2, 3))
    >>> a
    [[1, 2, 3], [4, 5, 6]]

    Create an N-dim array from a matrix:

    >>> from sympy import Matrix
    >>> a = Matrix([[1,2],[3,4]])
    >>> a
    Matrix([
    [1, 2],
    [3, 4]])
    >>> b = MutableDenseNDimArray(a)
    >>> b
    [[1, 2], [3, 4]]

    Arithmetic operations on N-dim arrays

    >>> a = MutableDenseNDimArray([1, 1, 1, 1], (2, 2))
    >>> b = MutableDenseNDimArray([4, 4, 4, 4], (2, 2))
    >>> c = a + b
    >>> c
    [[5, 5], [5, 5]]
    >>> a - b
    [[-3, -3], [-3, -3]]

    """

    _diff_wrt = True
    is_scalar = False

    def __new__(cls, iterable, shape=None, **kwargs):
        from sympy.tensor.array import ImmutableDenseNDimArray
        return ImmutableDenseNDimArray(iterable, shape, **kwargs)

    def __getitem__(self, index):
        raise NotImplementedError("A subclass of NDimArray should implement __getitem__")

    def _parse_index(self, index):
        if isinstance(index, (SYMPY_INTS, Integer)):
            if index >= self._loop_size:
                raise ValueError("Only a tuple index is accepted")
            return index

        if self._loop_size == 0:
            raise ValueError("Index not valid with an empty array")

        if len(index) != self._rank:
            raise ValueError('Wrong number of array axes')

        real_index = 0
        # check if input index can exist in current indexing
        for i in range(self._rank):
            if (index[i] >= self.shape[i]) or (index[i] < -self.shape[i]):
                raise ValueError('Index ' + str(index) + ' out of border')
            if index[i] < 0:
                real_index += 1
            real_index = real_index*self.shape[i] + index[i]

        return real_index

    def _get_tuple_index(self, integer_index):
        index = []
        for sh in reversed(self.shape):
            index.append(integer_index % sh)
            integer_index //= sh
        index.reverse()
        return tuple(index)

    def _check_symbolic_index(self, index):
        # Check if any index is symbolic:
        tuple_index = (index if isinstance(index, tuple) else (index,))
        if any((isinstance(i, Expr) and (not i.is_number)) for i in tuple_index):
            for i, nth_dim in zip(tuple_index, self.shape):
                if ((i < 0) == True) or ((i >= nth_dim) == True):
                    raise ValueError("index out of range")
            from sympy.tensor import Indexed
            return Indexed(self, *tuple_index)
        return None

    def _setter_iterable_check(self, value):
        from sympy.matrices.matrixbase import MatrixBase
        if isinstance(value, (Iterable, MatrixBase, NDimArray)):
            raise NotImplementedError

    @classmethod
    def _scan_iterable_shape(cls, iterable):
        def f(pointer):
            if not isinstance(pointer, Iterable):
                return [pointer], ()

            if len(pointer) == 0:
                return [], (0,)

            result = []
            elems, shapes = zip(*[f(i) for i in pointer])
            if len(set(shapes)) != 1:
                raise ValueError("could not determine shape unambiguously")
            for i in elems:
                result.extend(i)
            return result, (len(shapes),)+shapes[0]

        return f(iterable)

    @classmethod
    def _handle_ndarray_creation_inputs(cls, iterable=None, shape=None, **kwargs):
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.tensor.array import SparseNDimArray

        if shape is None:
            if iterable is None:
                shape = ()
                iterable = ()
            # Construction of a sparse array from a sparse array
            elif isinstance(iterable, SparseNDimArray):
                return iterable._shape, iterable._sparse_array

            # Construct N-dim array from another N-dim array:
            elif isinstance(iterable, NDimArray):
                shape = iterable.shape

            # Construct N-dim array from an iterable (numpy arrays included):
            elif isinstance(iterable, Iterable):
                iterable, shape = cls._scan_iterable_shape(iterable)

            # Construct N-dim array from a Matrix:
            elif isinstance(iterable, MatrixBase):
                shape = iterable.shape

            else:
                shape = ()
                iterable = (iterable,)

        if isinstance(iterable, (Dict, dict)) and shape is not None:
            new_dict = iterable.copy()
            for k in new_dict:
                if isinstance(k, (tuple, Tuple)):
                    new_key = 0
                    for i, idx in enumerate(k):
                        new_key = new_key * shape[i] + idx
                    iterable[new_key] = iterable[k]
                    del iterable[k]

        if isinstance(shape, (SYMPY_INTS, Integer)):
            shape = (shape,)

        if not all(isinstance(dim, (SYMPY_INTS, Integer)) for dim in shape):
            raise TypeError("Shape should contain integers only.")

        return tuple(shape), iterable

    def __len__(self):
        """Overload common function len(). Returns number of elements in array.

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray.zeros(3, 3)
        >>> a
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        >>> len(a)
        9

        """
        return self._loop_size

    @property
    def shape(self):
        """
        Returns array shape (dimension).

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray.zeros(3, 3)
        >>> a.shape
        (3, 3)

        """
        return self._shape

    def rank(self):
        """
        Returns rank of array.

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray.zeros(3,4,5,6,3)
        >>> a.rank()
        5

        """
        return self._rank

    def diff(self, *args, **kwargs):
        """
        Calculate the derivative of each element in the array.

        Examples
        ========

        >>> from sympy import ImmutableDenseNDimArray
        >>> from sympy.abc import x, y
        >>> M = ImmutableDenseNDimArray([[x, y], [1, x*y]])
        >>> M.diff(x)
        [[1, 0], [0, y]]

        """
        from sympy.tensor.array.array_derivatives import ArrayDerivative
        kwargs.setdefault('evaluate', True)
        return ArrayDerivative(self.as_immutable(), *args, **kwargs)

    def _eval_derivative(self, base):
        # Types are (base: scalar, self: array)
        return self.applyfunc(lambda x: base.diff(x))

    def _eval_derivative_n_times(self, s, n):
        return Basic._eval_derivative_n_times(self, s, n)

    def applyfunc(self, f):
        """Apply a function to each element of the N-dim array.

        Examples
        ========

        >>> from sympy import ImmutableDenseNDimArray
        >>> m = ImmutableDenseNDimArray([i*2+j for i in range(2) for j in range(2)], (2, 2))
        >>> m
        [[0, 1], [2, 3]]
        >>> m.applyfunc(lambda i: 2*i)
        [[0, 2], [4, 6]]
        """
        from sympy.tensor.array import SparseNDimArray
        from sympy.tensor.array.arrayop import Flatten

        if isinstance(self, SparseNDimArray) and f(S.Zero) == 0:
            return type(self)({k: f(v) for k, v in self._sparse_array.items() if f(v) != 0}, self.shape)

        return type(self)(map(f, Flatten(self)), self.shape)

    def _sympystr(self, printer):
        def f(sh, shape_left, i, j):
            if len(shape_left) == 1:
                return "["+", ".join([printer._print(self[self._get_tuple_index(e)]) for e in range(i, j)])+"]"

            sh //= shape_left[0]
            return "[" + ", ".join([f(sh, shape_left[1:], i+e*sh, i+(e+1)*sh) for e in range(shape_left[0])]) + "]" # + "\n"*len(shape_left)

        if self.rank() == 0:
            return printer._print(self[()])
        if 0 in self.shape:
            return f"{self.__class__.__name__}([], {self.shape})"
        return f(self._loop_size, self.shape, 0, self._loop_size)

    def tolist(self):
        """
        Converting MutableDenseNDimArray to one-dim list

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray([1, 2, 3, 4], (2, 2))
        >>> a
        [[1, 2], [3, 4]]
        >>> b = a.tolist()
        >>> b
        [[1, 2], [3, 4]]
        """

        def f(sh, shape_left, i, j):
            if len(shape_left) == 1:
                return [self[self._get_tuple_index(e)] for e in range(i, j)]
            result = []
            sh //= shape_left[0]
            for e in range(shape_left[0]):
                result.append(f(sh, shape_left[1:], i+e*sh, i+(e+1)*sh))
            return result

        return f(self._loop_size, self.shape, 0, self._loop_size)

    def __add__(self, other):
        from sympy.tensor.array.arrayop import Flatten

        if not isinstance(other, NDimArray):
            return NotImplemented

        if self.shape != other.shape:
            raise ValueError("array shape mismatch")
        result_list = [i+j for i,j in zip(Flatten(self), Flatten(other))]

        return type(self)(result_list, self.shape)

    def __sub__(self, other):
        from sympy.tensor.array.arrayop import Flatten

        if not isinstance(other, NDimArray):
            return NotImplemented

        if self.shape != other.shape:
            raise ValueError("array shape mismatch")
        result_list = [i-j for i,j in zip(Flatten(self), Flatten(other))]

        return type(self)(result_list, self.shape)

    def __mul__(self, other):
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.tensor.array import SparseNDimArray
        from sympy.tensor.array.arrayop import Flatten

        if isinstance(other, (Iterable, NDimArray, MatrixBase)):
            raise ValueError("scalar expected, use tensorproduct(...) for tensorial product")

        other = sympify(other)
        if isinstance(self, SparseNDimArray):
            if other.is_zero:
                return type(self)({}, self.shape)
            return type(self)({k: other*v for (k, v) in self._sparse_array.items()}, self.shape)

        result_list = [i*other for i in Flatten(self)]
        return type(self)(result_list, self.shape)

    def __rmul__(self, other):
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.tensor.array import SparseNDimArray
        from sympy.tensor.array.arrayop import Flatten

        if isinstance(other, (Iterable, NDimArray, MatrixBase)):
            raise ValueError("scalar expected, use tensorproduct(...) for tensorial product")

        other = sympify(other)
        if isinstance(self, SparseNDimArray):
            if other.is_zero:
                return type(self)({}, self.shape)
            return type(self)({k: other*v for (k, v) in self._sparse_array.items()}, self.shape)

        result_list = [other*i for i in Flatten(self)]
        return type(self)(result_list, self.shape)

    def __truediv__(self, other):
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.tensor.array import SparseNDimArray
        from sympy.tensor.array.arrayop import Flatten

        if isinstance(other, (Iterable, NDimArray, MatrixBase)):
            raise ValueError("scalar expected")

        other = sympify(other)
        if isinstance(self, SparseNDimArray) and other != S.Zero:
            return type(self)({k: v/other for (k, v) in self._sparse_array.items()}, self.shape)

        result_list = [i/other for i in Flatten(self)]
        return type(self)(result_list, self.shape)

    def __rtruediv__(self, other):
        raise NotImplementedError('unsupported operation on NDimArray')

    def __neg__(self):
        from sympy.tensor.array import SparseNDimArray
        from sympy.tensor.array.arrayop import Flatten

        if isinstance(self, SparseNDimArray):
            return type(self)({k: -v for (k, v) in self._sparse_array.items()}, self.shape)

        result_list = [-i for i in Flatten(self)]
        return type(self)(result_list, self.shape)

    def __iter__(self):
        def iterator():
            if self._shape:
                for i in range(self._shape[0]):
                    yield self[i]
            else:
                yield self[()]

        return iterator()

    def __eq__(self, other):
        """
        NDimArray instances can be compared to each other.
        Instances equal if they have same shape and data.

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray.zeros(2, 3)
        >>> b = MutableDenseNDimArray.zeros(2, 3)
        >>> a == b
        True
        >>> c = a.reshape(3, 2)
        >>> c == b
        False
        >>> a[0,0] = 1
        >>> b[0,0] = 2
        >>> a == b
        False
        """
        from sympy.tensor.array import SparseNDimArray
        if not isinstance(other, NDimArray):
            return False

        if not self.shape == other.shape:
            return False

        if isinstance(self, SparseNDimArray) and isinstance(other, SparseNDimArray):
            return dict(self._sparse_array) == dict(other._sparse_array)

        return list(self) == list(other)

    def __ne__(self, other):
        return not self == other

    def _eval_transpose(self):
        if self.rank() != 2:
            raise ValueError("array rank not 2")
        from .arrayop import permutedims
        return permutedims(self, (1, 0))

    def transpose(self):
        return self._eval_transpose()

    def _eval_conjugate(self):
        from sympy.tensor.array.arrayop import Flatten

        return self.func([i.conjugate() for i in Flatten(self)], self.shape)

    def conjugate(self):
        return self._eval_conjugate()

    def _eval_adjoint(self):
        return self.transpose().conjugate()

    def adjoint(self):
        return self._eval_adjoint()

    def _slice_expand(self, s, dim):
        if not isinstance(s, slice):
                return (s,)
        start, stop, step = s.indices(dim)
        return [start + i*step for i in range((stop-start)//step)]

    def _get_slice_data_for_array_access(self, index):
        sl_factors = [self._slice_expand(i, dim) for (i, dim) in zip(index, self.shape)]
        eindices = itertools.product(*sl_factors)
        return sl_factors, eindices

    def _get_slice_data_for_array_assignment(self, index, value):
        if not isinstance(value, NDimArray):
            value = type(self)(value)
        sl_factors, eindices = self._get_slice_data_for_array_access(index)
        slice_offsets = [min(i) if isinstance(i, list) else None for i in sl_factors]
        # TODO: add checks for dimensions for `value`?
        return value, eindices, slice_offsets

    @classmethod
    def _check_special_bounds(cls, flat_list, shape):
        if shape == () and len(flat_list) != 1:
            raise ValueError("arrays without shape need one scalar value")
        if shape == (0,) and len(flat_list) > 0:
            raise ValueError("if array shape is (0,) there cannot be elements")

    def _check_index_for_getitem(self, index):
        if isinstance(index, (SYMPY_INTS, Integer, slice)):
            index = (index,)

        if len(index) < self.rank():
            index = tuple(index) + \
                          tuple(slice(None) for i in range(len(index), self.rank()))

        if len(index) > self.rank():
            raise ValueError('Dimension of index greater than rank of array')

        return index


class ImmutableNDimArray(NDimArray, Basic):
    _op_priority = 11.0

    def __hash__(self):
        return Basic.__hash__(self)

    def as_immutable(self):
        return self

    def as_mutable(self):
        raise NotImplementedError("abstract method")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\onnx_model.py ===
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import onnx
import onnx.helper as onnx_helper
import onnx.numpy_helper as onnx_numpy_helper
from onnx.onnx_pb import ModelProto

from .quant_utils import attribute_to_kwarg, find_by_name


def _clean_initializers_helper(graph, model):
    """Clean unused initializers from graph.

    Returns:
        A cleaned graph without unused initializers
        A list of tensor names, which are not produced by this graph and its subgraphes
    """
    requesting_tensor_names = set()
    requesting_tensor_names.update(input_name for node in graph.node for input_name in node.input if input_name)
    requesting_tensor_names.update(g_out.name for g_out in graph.output if g_out.name)

    new_nodes = []
    for node in graph.node:
        new_node = node
        graph_attrs = [
            attr
            for attr in node.attribute
            if attr.type == onnx.AttributeProto.GRAPH or attr.type == onnx.AttributeProto.GRAPHS
        ]
        if graph_attrs:
            kwargs = {}
            for attr in node.attribute:
                new_attribute = {}
                if attr.type == onnx.AttributeProto.GRAPH:
                    (
                        cleaned_sub_graph,
                        sub_requesting_tensor_names,
                    ) = _clean_initializers_helper(attr.g, model)
                    new_attribute = {attr.name: cleaned_sub_graph}
                    requesting_tensor_names.update(sub_requesting_tensor_names)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    cleaned_graphes = []
                    for subgraph in attr.graphs:
                        (
                            cleaned_sub_graph,
                            sub_requesting_tensor_names,
                        ) = _clean_initializers_helper(subgraph, model)
                        cleaned_graphes.append(cleaned_sub_graph)
                        requesting_tensor_names.update(sub_requesting_tensor_names)
                    new_attribute = {attr.name: cleaned_graphes}
                else:
                    new_attribute = attribute_to_kwarg(attr)
                kwargs.update(new_attribute)
            new_node = onnx_helper.make_node(node.op_type, node.input, node.output, name=node.name, **kwargs)
        new_nodes.append(new_node)

    graph.ClearField("node")
    graph.node.extend(new_nodes)

    requesting_tensor_names.difference_update(output for node in graph.node for output in node.output)

    unused_initializer = []
    for initializer in graph.initializer:
        if initializer.name in requesting_tensor_names:
            requesting_tensor_names.remove(initializer.name)
        else:
            # mark it to remove, remove here directly will cause mis-behavier
            unused_initializer.append(initializer)

    name_to_input = {input.name: input for input in graph.input}
    for initializer in unused_initializer:
        graph.initializer.remove(initializer)
        if initializer.name in name_to_input:
            try:
                graph.input.remove(name_to_input[initializer.name])
            except StopIteration:
                if model.ir_version < 4:
                    print(f"Warning: invalid weight name {initializer.name} found in the graph (not a graph input)")

    requesting_tensor_names.difference_update(input.name for input in graph.input)

    return graph, requesting_tensor_names


class ONNXModel:
    def __init__(self, model: ModelProto):
        self.model = model

    def nodes(self):
        return self.model.graph.node

    def initializer(self):
        return self.model.graph.initializer

    def initializer_extend(self, inits):
        if len(inits) == 0:
            raise ValueError("Can add an empty list.")
        for init in self.initializer():
            self._check_init(init, "gain")
        for init in inits:
            self._check_init(init)
            self.model.graph.initializer.append(init)

    def graph(self):
        return self.model.graph

    def ir_version(self):
        return self.model.ir_version

    def opset_import(self):
        return self.model.opset_import

    def set_opset_import(self, domain, version):
        for opset in self.model.opset_import:
            if opset.domain == domain:
                opset.version = version
                return

        self.model.opset_import.extend([onnx_helper.make_opsetid(domain, version)])

    def remove_node(self, node):
        if node in self.model.graph.node:
            self.model.graph.node.remove(node)

    def remove_nodes(self, nodes_to_remove):
        for node in nodes_to_remove:
            self.remove_node(node)

    def add_node(self, node):
        self.model.graph.node.extend([self._check_node(node)])

    def add_nodes(self, nodes_to_add):
        for node in nodes_to_add:
            self.add_node(node)

    def add_initializer(self, tensor):
        if find_by_name(tensor.name, self.model.graph.initializer) is None:
            self._check_init(tensor)
            self.model.graph.initializer.extend([tensor])

    def get_initializer(self, name):
        for tensor in self.model.graph.initializer:
            if tensor.name == name:
                return tensor
        return None

    def find_graph_input(self, input_name):
        for input in self.model.graph.input:
            if input.name == input_name:
                return input
        return None

    def find_graph_output(self, output_name):
        for output in self.model.graph.output:
            if output.name == output_name:
                return output
        return None

    def get_tensor_type(self, tensor_name: str):
        tensor_type_map = {obj.name: obj.type for obj in self.model.graph.value_info}

        if tensor_name in tensor_type_map:
            return tensor_type_map[tensor_name].tensor_type

        g_input = self.find_graph_input(tensor_name)
        if g_input:
            return g_input.type.tensor_type

        g_output = self.find_graph_output(tensor_name)
        if g_output:
            return g_output.type.tensor_type

        return None

    def get_constant_value(self, output_name):
        for node in self.model.graph.node:
            if node.op_type == "Constant":
                if node.output[0] == output_name:
                    for attr in node.attribute:
                        if attr.name == "value":
                            return onnx_numpy_helper.to_array(attr.t)

        # Fallback to initializer since constant folding may have been applied.
        initializer = self.get_initializer(output_name)
        if initializer is not None:
            return onnx_numpy_helper.to_array(initializer)

        return None

    def get_initializer_name_set(self):
        return {initializer.name for initializer in self.model.graph.initializer}

    def remove_initializer(self, tensor):
        if tensor in self.model.graph.initializer:
            self.model.graph.initializer.remove(tensor)
            for input in self.model.graph.input:
                if input.name == tensor.name:
                    self.model.graph.input.remove(input)
                    break

    def remove_initializers(self, init_to_remove):
        for initializer in init_to_remove:
            self.remove_initializer(initializer)

    def get_non_initializer_inputs(self):
        initializer_names = self.get_initializer_name_set()
        non_initializer_inputs = set()
        for input in self.model.graph.input:
            if input.name not in initializer_names:
                non_initializer_inputs.add(input.name)
        return non_initializer_inputs

    def input_name_to_nodes(self):
        input_name_to_nodes = {}
        for node in self.model.graph.node:
            for input_name in node.input:
                if input_name:  # Could be empty when it is optional
                    if input_name not in input_name_to_nodes:
                        input_name_to_nodes[input_name] = [node]
                    else:
                        input_name_to_nodes[input_name].append(node)
        return input_name_to_nodes

    def output_name_to_node(self):
        output_name_to_node = {}
        for node in self.model.graph.node:
            for output_name in node.output:
                if output_name:  # Could be empty when it is optional
                    output_name_to_node[output_name] = node
        return output_name_to_node

    def get_children(self, node, input_name_to_nodes=None):
        if input_name_to_nodes is None:
            input_name_to_nodes = self.input_name_to_nodes()

        children = []
        for output in node.output:
            if output in input_name_to_nodes:
                for node in input_name_to_nodes[output]:
                    children.append(node)  # noqa: PERF402
        return children

    def get_parents(self, node, output_name_to_node=None):
        if output_name_to_node is None:
            output_name_to_node = self.output_name_to_node()

        parents = []
        for input in node.input:
            if input in output_name_to_node:
                parents.append(output_name_to_node[input])
        return parents

    def get_parent(self, node, idx, output_name_to_node=None):
        if output_name_to_node is None:
            output_name_to_node = self.output_name_to_node()

        if len(node.input) <= idx:
            return None

        input = node.input[idx]
        if input not in output_name_to_node:
            return None

        return output_name_to_node[input]

    def find_node_by_name(self, node_name, new_nodes_list, graph):
        """Find out if a node exists in a graph or a node is in the
        new set of nodes created during quantization.

        Returns:
            The node found or None.
        """
        graph_nodes_list = list(graph.node)  # deep copy
        graph_nodes_list.extend(new_nodes_list)
        node = find_by_name(node_name, graph_nodes_list)
        return node

    def get_largest_node_name_suffix(self, node_name_prefix):
        """
        Gets the largest node name (int) suffix for all node names that begin with `node_name_prefix`.
        Example: for nodes my_prefix_0 and my_prefix_3, this method returns 3.
        """
        suffix = -1

        for node in self.model.graph.node:
            if node.name and node.name.startswith(node_name_prefix):
                try:
                    index = int(node.name[len(node_name_prefix) :])
                    suffix = max(index, suffix)
                except ValueError:
                    continue

        return suffix

    def get_largest_initializer_name_suffix(self, initializer_name_prefix):
        """
        Gets the largest initializer name integer suffix for all initializer names that begin
        with `initializer_name_prefix`. This can be used to create unique initializer names.

        Example: for initializer names 'my_weight_0' and 'my_weight_3', this method returns 3 if
                 `initializer_name_prefix` is 'my_weight_'.
        """
        suffix = -1

        for initializer in self.model.graph.initializer:
            if initializer.name.startswith(initializer_name_prefix):
                try:
                    index = int(initializer.name[len(initializer_name_prefix) :])
                    suffix = max(index, suffix)
                except ValueError:
                    continue

        return suffix

    def find_nodes_by_initializer(self, graph, initializer):
        """
        Find all nodes with given initializer as an input.
        """
        nodes = []
        for node in graph.node:
            for node_input in node.input:
                if node_input == initializer.name:
                    nodes.append(node)
        return nodes

    @staticmethod
    def __get_initializer(name, graph_path):
        for gid in range(len(graph_path) - 1, -1, -1):
            graph = graph_path[gid]
            for tensor in graph.initializer:
                if tensor.name == name:
                    return tensor, graph
        return None, None

    @staticmethod
    def __replace_gemm_with_matmul(graph_path):
        new_nodes = []
        graph = graph_path[-1]
        for node in graph.node:
            graph_attrs = [attr for attr in node.attribute if attr.type == 5 or attr.type == 10]
            if len(graph_attrs):
                kwargs = {}
                for attr in node.attribute:
                    if attr.type == 5:
                        graph_path.append(attr.g)
                        kv = {attr.name: ONNXModel.__replace_gemm_with_matmul(graph_path)}
                    elif attr.type == 10:
                        value = []
                        for subgraph in attr.graphs:
                            graph_path.append(subgraph)
                            value.extend([ONNXModel.__replace_gemm_with_matmul(graph_path)])
                        kv = {attr.name: value}
                    else:
                        kv = attribute_to_kwarg(attr)
                    kwargs.update(kv)
                node = onnx_helper.make_node(  # noqa: PLW2901
                    node.op_type, node.input, node.output, name=node.name, **kwargs
                )

            if node.op_type == "Gemm":
                alpha = 1.0
                beta = 1.0
                transA = 0  # noqa: N806
                transB = 0  # noqa: N806
                for attr in node.attribute:
                    if attr.name == "alpha":
                        alpha = onnx_helper.get_attribute_value(attr)
                    elif attr.name == "beta":
                        beta = onnx_helper.get_attribute_value(attr)
                    elif attr.name == "transA":
                        transA = onnx_helper.get_attribute_value(attr)  # noqa: N806
                    elif attr.name == "transB":
                        transB = onnx_helper.get_attribute_value(attr)  # noqa: N806
                if alpha == 1.0 and beta == 1.0 and transA == 0:
                    inputB = node.input[1]  # noqa: N806
                    if transB == 1:
                        B, Bs_graph = ONNXModel.__get_initializer(node.input[1], graph_path)  # noqa: N806
                        if B:
                            # assume B is not used by any other node
                            B_array = onnx_numpy_helper.to_array(B)  # noqa: N806
                            B_trans = onnx_numpy_helper.from_array(B_array.T)  # noqa: N806
                            B_trans.name = B.name
                            Bs_graph.initializer.remove(B)
                            for input in Bs_graph.input:
                                if input.name == inputB:
                                    Bs_graph.input.remove(input)
                                    break
                            Bs_graph.initializer.extend([B_trans])
                        else:
                            inputB += "_Transposed"  # noqa: N806
                            transpose_node = onnx_helper.make_node(
                                "Transpose",
                                inputs=[node.input[1]],
                                outputs=[inputB],
                                name=node.name + "_Transpose" if node.name else "",
                            )
                            new_nodes.append(transpose_node)

                    matmul_node = onnx_helper.make_node(
                        "MatMul",
                        inputs=[node.input[0], inputB],
                        outputs=[node.output[0] + ("_MatMul" if len(node.input) > 2 else "")],
                        name=node.name + "_MatMul" if node.name else "",
                    )
                    new_nodes.append(matmul_node)

                    if len(node.input) > 2:
                        add_node = onnx_helper.make_node(
                            "Add",
                            inputs=[node.output[0] + "_MatMul", node.input[2]],
                            outputs=node.output,
                            name=node.name + "_Add" if node.name else "",
                        )
                        new_nodes.append(add_node)

                # unsupported
                else:
                    new_nodes.append(node)

            # not GEMM
            else:
                new_nodes.append(node)

        graph.ClearField("node")
        graph.node.extend(new_nodes)
        graph_path.pop()
        return graph

    def replace_gemm_with_matmul(self):
        graph_path = [self.graph()]
        ONNXModel.__replace_gemm_with_matmul(graph_path)

    def save_model_to_file(self, output_path, use_external_data_format=False):
        """
        Save model to external data, which is needed for model size > 2GB
        """
        self.topological_sort()
        if use_external_data_format:
            onnx.external_data_helper.convert_model_to_external_data(
                self.model,
                all_tensors_to_one_file=True,
                location=Path(output_path).name + ".data",
                convert_attribute=True,
            )
        for init in self.model.graph.initializer:
            self._check_init(init, "end")
        onnx.save_model(self.model, output_path)

    @staticmethod
    def replace_node_input(node, old_input_name, new_input_name):
        assert isinstance(old_input_name, str) and isinstance(new_input_name, str)
        for j in range(len(node.input)):
            if node.input[j] == old_input_name:
                node.input[j] = new_input_name

    def replace_input_of_all_nodes(self, old_input_name, new_input_name):
        for node in self.model.graph.node:
            ONNXModel.replace_node_input(node, old_input_name, new_input_name)

    def replace_input_of_nodes(self, old_input_name, new_input_name, node_names_set):
        for node in self.model.graph.node:
            if node.name in node_names_set:
                ONNXModel.replace_node_input(node, old_input_name, new_input_name)

    @staticmethod
    def replace_node_output(node, old_output_name, new_output_name):
        assert isinstance(old_output_name, str) and isinstance(new_output_name, str)
        for j in range(len(node.output)):
            if node.output[j] == old_output_name:
                node.output[j] = new_output_name

    def replace_output_of_all_nodes(self, old_output_name, new_output_name):
        for node in self.model.graph.node:
            ONNXModel.replace_node_output(node, old_output_name, new_output_name)

    def replace_output_of_nodes(self, old_output_name, new_output_name, node_names_set):
        for node in self.model.graph.node:
            if node.name in node_names_set:
                ONNXModel.replace_node_output(node, old_output_name, new_output_name)

    def remove_unused_constant(self):
        input_name_to_nodes = self.input_name_to_nodes()

        # remove unused constant
        unused_nodes = []
        nodes = self.nodes()
        for node in nodes:
            if (
                node.op_type == "Constant"
                and not self.is_graph_output(node.output[0])
                and node.output[0] not in input_name_to_nodes
            ):
                unused_nodes.append(node)

        self.remove_nodes(unused_nodes)

        ununsed_weights = []
        for w in self.initializer():
            if w.name not in input_name_to_nodes and not self.is_graph_output(w.name):
                ununsed_weights.append(w)
                # Remove from graph.input
                for graph_input in self.graph().input:
                    if graph_input.name == w.name:
                        self.graph().input.remove(graph_input)

        self.remove_initializers(ununsed_weights)

    def is_graph_output(self, output_name):
        return any(output.name == output_name for output in self.model.graph.output)

    def is_graph_input(self, tensor_name: str) -> bool:
        return any(input.name == tensor_name for input in self.model.graph.input)

    # TODO:use OnnxModel.graph_topological_sort(self.model.graph) from transformers.onnx_model
    # Currently it breaks Openvino/Linux training gpu pipeline so hold off for 1.8 release
    def topological_sort(self):
        deps_count = [0] * len(self.nodes())  # dependency count of each node
        deps_to_nodes = {}  # input to node indice
        sorted_nodes = []  # initialize sorted_nodes
        for node_idx, node in enumerate(self.nodes()):
            # CANNOT use len(node.input) directly because input can be optional
            deps_count[node_idx] = sum(1 for _ in node.input if _)
            if deps_count[node_idx] == 0:  # Constant doesn't depend on any inputs
                sorted_nodes.append(self.nodes()[node_idx])
                continue

            for input_name in node.input:
                if not input_name:
                    continue
                if input_name not in deps_to_nodes:
                    deps_to_nodes[input_name] = [node_idx]
                else:
                    deps_to_nodes[input_name].append(node_idx)

        initializer_names = [init.name for init in self.initializer()]
        graph_input_names = [input.name for input in self.model.graph.input]
        input_names = initializer_names + graph_input_names
        input_names.sort()
        prev_input_name = None
        for input_name in input_names:
            if prev_input_name == input_name:
                continue

            prev_input_name = input_name
            if input_name in deps_to_nodes:
                for node_idx in deps_to_nodes[input_name]:
                    deps_count[node_idx] = deps_count[node_idx] - 1
                    if deps_count[node_idx] == 0:
                        sorted_nodes.append(self.nodes()[node_idx])

        start = 0
        end = len(sorted_nodes)

        while start < end:
            for output in sorted_nodes[start].output:
                if output in deps_to_nodes:
                    for node_idx in deps_to_nodes[output]:
                        deps_count[node_idx] = deps_count[node_idx] - 1
                        if deps_count[node_idx] == 0:
                            sorted_nodes.append(self.nodes()[node_idx])
                            end = end + 1
            start = start + 1

        assert end == len(self.graph().node), "Graph is not a DAG"
        self.graph().ClearField("node")
        self.graph().node.extend(sorted_nodes)

    def clean_initializers(self):
        return _clean_initializers_helper(self.graph(), self.model)

    def _check_init(self, init, test=None):
        if init.data_type == onnx.TensorProto.FLOAT8E4M3FN:
            if init.HasField("raw_data"):
                b = list(init.raw_data)
                if any((i & 127) == 127 for i in b):
                    raise ValueError(f"Initializer {init.name!r} has nan.")
        return init

    def _check_node(self, node):
        """
        A quantization to float 8 does not use quantized bias but float 16 bias.
        This function checks that DequantizeLinear is not used to
        dequantize from float 16.
        """
        if node.op_type == "DequantizeLinear":
            zero_point = node.input[2]
            init = self.get_initializer(zero_point)
            dtype = init.data_type
            if dtype in {
                onnx.TensorProto.FLOAT16,
                onnx.TensorProto.FLOAT,
                onnx.TensorProto.DOUBLE,
                onnx.TensorProto.BFLOAT16,
            }:
                raise RuntimeError(f"Unsupported DequantizeLinear operator, dequantization from {dtype}.")
        return node

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\ext\compiler.py ===
# ext/compiler.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

r"""Provides an API for creation of custom ClauseElements and compilers.

Synopsis
========

Usage involves the creation of one or more
:class:`~sqlalchemy.sql.expression.ClauseElement` subclasses and one or
more callables defining its compilation::

    from sqlalchemy.ext.compiler import compiles
    from sqlalchemy.sql.expression import ColumnClause


    class MyColumn(ColumnClause):
        inherit_cache = True


    @compiles(MyColumn)
    def compile_mycolumn(element, compiler, **kw):
        return "[%s]" % element.name

Above, ``MyColumn`` extends :class:`~sqlalchemy.sql.expression.ColumnClause`,
the base expression element for named column objects. The ``compiles``
decorator registers itself with the ``MyColumn`` class so that it is invoked
when the object is compiled to a string::

    from sqlalchemy import select

    s = select(MyColumn("x"), MyColumn("y"))
    print(str(s))

Produces:

.. sourcecode:: sql

    SELECT [x], [y]

Dialect-specific compilation rules
==================================

Compilers can also be made dialect-specific. The appropriate compiler will be
invoked for the dialect in use::

    from sqlalchemy.schema import DDLElement


    class AlterColumn(DDLElement):
        inherit_cache = False

        def __init__(self, column, cmd):
            self.column = column
            self.cmd = cmd


    @compiles(AlterColumn)
    def visit_alter_column(element, compiler, **kw):
        return "ALTER COLUMN %s ..." % element.column.name


    @compiles(AlterColumn, "postgresql")
    def visit_alter_column(element, compiler, **kw):
        return "ALTER TABLE %s ALTER COLUMN %s ..." % (
            element.table.name,
            element.column.name,
        )

The second ``visit_alter_table`` will be invoked when any ``postgresql``
dialect is used.

.. _compilerext_compiling_subelements:

Compiling sub-elements of a custom expression construct
=======================================================

The ``compiler`` argument is the
:class:`~sqlalchemy.engine.interfaces.Compiled` object in use. This object
can be inspected for any information about the in-progress compilation,
including ``compiler.dialect``, ``compiler.statement`` etc. The
:class:`~sqlalchemy.sql.compiler.SQLCompiler` and
:class:`~sqlalchemy.sql.compiler.DDLCompiler` both include a ``process()``
method which can be used for compilation of embedded attributes::

    from sqlalchemy.sql.expression import Executable, ClauseElement


    class InsertFromSelect(Executable, ClauseElement):
        inherit_cache = False

        def __init__(self, table, select):
            self.table = table
            self.select = select


    @compiles(InsertFromSelect)
    def visit_insert_from_select(element, compiler, **kw):
        return "INSERT INTO %s (%s)" % (
            compiler.process(element.table, asfrom=True, **kw),
            compiler.process(element.select, **kw),
        )


    insert = InsertFromSelect(t1, select(t1).where(t1.c.x > 5))
    print(insert)

Produces (formatted for readability):

.. sourcecode:: sql

    INSERT INTO mytable (
        SELECT mytable.x, mytable.y, mytable.z
        FROM mytable
        WHERE mytable.x > :x_1
    )

.. note::

    The above ``InsertFromSelect`` construct is only an example, this actual
    functionality is already available using the
    :meth:`_expression.Insert.from_select` method.


Cross Compiling between SQL and DDL compilers
---------------------------------------------

SQL and DDL constructs are each compiled using different base compilers -
``SQLCompiler`` and ``DDLCompiler``.   A common need is to access the
compilation rules of SQL expressions from within a DDL expression. The
``DDLCompiler`` includes an accessor ``sql_compiler`` for this reason, such as
below where we generate a CHECK constraint that embeds a SQL expression::

    @compiles(MyConstraint)
    def compile_my_constraint(constraint, ddlcompiler, **kw):
        kw["literal_binds"] = True
        return "CONSTRAINT %s CHECK (%s)" % (
            constraint.name,
            ddlcompiler.sql_compiler.process(constraint.expression, **kw),
        )

Above, we add an additional flag to the process step as called by
:meth:`.SQLCompiler.process`, which is the ``literal_binds`` flag.  This
indicates that any SQL expression which refers to a :class:`.BindParameter`
object or other "literal" object such as those which refer to strings or
integers should be rendered **in-place**, rather than being referred to as
a bound parameter;  when emitting DDL, bound parameters are typically not
supported.


Changing the default compilation of existing constructs
=======================================================

The compiler extension applies just as well to the existing constructs.  When
overriding the compilation of a built in SQL construct, the @compiles
decorator is invoked upon the appropriate class (be sure to use the class,
i.e. ``Insert`` or ``Select``, instead of the creation function such
as ``insert()`` or ``select()``).

Within the new compilation function, to get at the "original" compilation
routine, use the appropriate visit_XXX method - this
because compiler.process() will call upon the overriding routine and cause
an endless loop.   Such as, to add "prefix" to all insert statements::

    from sqlalchemy.sql.expression import Insert


    @compiles(Insert)
    def prefix_inserts(insert, compiler, **kw):
        return compiler.visit_insert(insert.prefix_with("some prefix"), **kw)

The above compiler will prefix all INSERT statements with "some prefix" when
compiled.

.. _type_compilation_extension:

Changing Compilation of Types
=============================

``compiler`` works for types, too, such as below where we implement the
MS-SQL specific 'max' keyword for ``String``/``VARCHAR``::

    @compiles(String, "mssql")
    @compiles(VARCHAR, "mssql")
    def compile_varchar(element, compiler, **kw):
        if element.length == "max":
            return "VARCHAR('max')"
        else:
            return compiler.visit_VARCHAR(element, **kw)


    foo = Table("foo", metadata, Column("data", VARCHAR("max")))

Subclassing Guidelines
======================

A big part of using the compiler extension is subclassing SQLAlchemy
expression constructs. To make this easier, the expression and
schema packages feature a set of "bases" intended for common tasks.
A synopsis is as follows:

* :class:`~sqlalchemy.sql.expression.ClauseElement` - This is the root
  expression class. Any SQL expression can be derived from this base, and is
  probably the best choice for longer constructs such as specialized INSERT
  statements.

* :class:`~sqlalchemy.sql.expression.ColumnElement` - The root of all
  "column-like" elements. Anything that you'd place in the "columns" clause of
  a SELECT statement (as well as order by and group by) can derive from this -
  the object will automatically have Python "comparison" behavior.

  :class:`~sqlalchemy.sql.expression.ColumnElement` classes want to have a
  ``type`` member which is expression's return type.  This can be established
  at the instance level in the constructor, or at the class level if its
  generally constant::

      class timestamp(ColumnElement):
          type = TIMESTAMP()
          inherit_cache = True

* :class:`~sqlalchemy.sql.functions.FunctionElement` - This is a hybrid of a
  ``ColumnElement`` and a "from clause" like object, and represents a SQL
  function or stored procedure type of call. Since most databases support
  statements along the line of "SELECT FROM <some function>"
  ``FunctionElement`` adds in the ability to be used in the FROM clause of a
  ``select()`` construct::

      from sqlalchemy.sql.expression import FunctionElement


      class coalesce(FunctionElement):
          name = "coalesce"
          inherit_cache = True


      @compiles(coalesce)
      def compile(element, compiler, **kw):
          return "coalesce(%s)" % compiler.process(element.clauses, **kw)


      @compiles(coalesce, "oracle")
      def compile(element, compiler, **kw):
          if len(element.clauses) > 2:
              raise TypeError(
                  "coalesce only supports two arguments on " "Oracle Database"
              )
          return "nvl(%s)" % compiler.process(element.clauses, **kw)

* :class:`.ExecutableDDLElement` - The root of all DDL expressions,
  like CREATE TABLE, ALTER TABLE, etc. Compilation of
  :class:`.ExecutableDDLElement` subclasses is issued by a
  :class:`.DDLCompiler` instead of a :class:`.SQLCompiler`.
  :class:`.ExecutableDDLElement` can also be used as an event hook in
  conjunction with event hooks like :meth:`.DDLEvents.before_create` and
  :meth:`.DDLEvents.after_create`, allowing the construct to be invoked
  automatically during CREATE TABLE and DROP TABLE sequences.

  .. seealso::

    :ref:`metadata_ddl_toplevel` - contains examples of associating
    :class:`.DDL` objects (which are themselves :class:`.ExecutableDDLElement`
    instances) with :class:`.DDLEvents` event hooks.

* :class:`~sqlalchemy.sql.expression.Executable` - This is a mixin which
  should be used with any expression class that represents a "standalone"
  SQL statement that can be passed directly to an ``execute()`` method.  It
  is already implicit within ``DDLElement`` and ``FunctionElement``.

Most of the above constructs also respond to SQL statement caching.   A
subclassed construct will want to define the caching behavior for the object,
which usually means setting the flag ``inherit_cache`` to the value of
``False`` or ``True``.  See the next section :ref:`compilerext_caching`
for background.


.. _compilerext_caching:

Enabling Caching Support for Custom Constructs
==============================================

SQLAlchemy as of version 1.4 includes a
:ref:`SQL compilation caching facility <sql_caching>` which will allow
equivalent SQL constructs to cache their stringified form, along with other
structural information used to fetch results from the statement.

For reasons discussed at :ref:`caching_caveats`, the implementation of this
caching system takes a conservative approach towards including custom SQL
constructs and/or subclasses within the caching system.   This includes that
any user-defined SQL constructs, including all the examples for this
extension, will not participate in caching by default unless they positively
assert that they are able to do so.  The :attr:`.HasCacheKey.inherit_cache`
attribute when set to ``True`` at the class level of a specific subclass
will indicate that instances of this class may be safely cached, using the
cache key generation scheme of the immediate superclass.  This applies
for example to the "synopsis" example indicated previously::

    class MyColumn(ColumnClause):
        inherit_cache = True


    @compiles(MyColumn)
    def compile_mycolumn(element, compiler, **kw):
        return "[%s]" % element.name

Above, the ``MyColumn`` class does not include any new state that
affects its SQL compilation; the cache key of ``MyColumn`` instances will
make use of that of the ``ColumnClause`` superclass, meaning it will take
into account the class of the object (``MyColumn``), the string name and
datatype of the object::

    >>> MyColumn("some_name", String())._generate_cache_key()
    CacheKey(
        key=('0', <class '__main__.MyColumn'>,
        'name', 'some_name',
        'type', (<class 'sqlalchemy.sql.sqltypes.String'>,
                 ('length', None), ('collation', None))
    ), bindparams=[])

For objects that are likely to be **used liberally as components within many
larger statements**, such as :class:`_schema.Column` subclasses and custom SQL
datatypes, it's important that **caching be enabled as much as possible**, as
this may otherwise negatively affect performance.

An example of an object that **does** contain state which affects its SQL
compilation is the one illustrated at :ref:`compilerext_compiling_subelements`;
this is an "INSERT FROM SELECT" construct that combines together a
:class:`_schema.Table` as well as a :class:`_sql.Select` construct, each of
which independently affect the SQL string generation of the construct.  For
this class, the example illustrates that it simply does not participate in
caching::

    class InsertFromSelect(Executable, ClauseElement):
        inherit_cache = False

        def __init__(self, table, select):
            self.table = table
            self.select = select


    @compiles(InsertFromSelect)
    def visit_insert_from_select(element, compiler, **kw):
        return "INSERT INTO %s (%s)" % (
            compiler.process(element.table, asfrom=True, **kw),
            compiler.process(element.select, **kw),
        )

While it is also possible that the above ``InsertFromSelect`` could be made to
produce a cache key that is composed of that of the :class:`_schema.Table` and
:class:`_sql.Select` components together, the API for this is not at the moment
fully public. However, for an "INSERT FROM SELECT" construct, which is only
used by itself for specific operations, caching is not as critical as in the
previous example.

For objects that are **used in relative isolation and are generally
standalone**, such as custom :term:`DML` constructs like an "INSERT FROM
SELECT", **caching is generally less critical** as the lack of caching for such
a construct will have only localized implications for that specific operation.


Further Examples
================

"UTC timestamp" function
-------------------------

A function that works like "CURRENT_TIMESTAMP" except applies the
appropriate conversions so that the time is in UTC time.   Timestamps are best
stored in relational databases as UTC, without time zones.   UTC so that your
database doesn't think time has gone backwards in the hour when daylight
savings ends, without timezones because timezones are like character
encodings - they're best applied only at the endpoints of an application
(i.e. convert to UTC upon user input, re-apply desired timezone upon display).

For PostgreSQL and Microsoft SQL Server::

    from sqlalchemy.sql import expression
    from sqlalchemy.ext.compiler import compiles
    from sqlalchemy.types import DateTime


    class utcnow(expression.FunctionElement):
        type = DateTime()
        inherit_cache = True


    @compiles(utcnow, "postgresql")
    def pg_utcnow(element, compiler, **kw):
        return "TIMEZONE('utc', CURRENT_TIMESTAMP)"


    @compiles(utcnow, "mssql")
    def ms_utcnow(element, compiler, **kw):
        return "GETUTCDATE()"

Example usage::

    from sqlalchemy import Table, Column, Integer, String, DateTime, MetaData

    metadata = MetaData()
    event = Table(
        "event",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("description", String(50), nullable=False),
        Column("timestamp", DateTime, server_default=utcnow()),
    )

"GREATEST" function
-------------------

The "GREATEST" function is given any number of arguments and returns the one
that is of the highest value - its equivalent to Python's ``max``
function.  A SQL standard version versus a CASE based version which only
accommodates two arguments::

    from sqlalchemy.sql import expression, case
    from sqlalchemy.ext.compiler import compiles
    from sqlalchemy.types import Numeric


    class greatest(expression.FunctionElement):
        type = Numeric()
        name = "greatest"
        inherit_cache = True


    @compiles(greatest)
    def default_greatest(element, compiler, **kw):
        return compiler.visit_function(element)


    @compiles(greatest, "sqlite")
    @compiles(greatest, "mssql")
    @compiles(greatest, "oracle")
    def case_greatest(element, compiler, **kw):
        arg1, arg2 = list(element.clauses)
        return compiler.process(case((arg1 > arg2, arg1), else_=arg2), **kw)

Example usage::

    Session.query(Account).filter(
        greatest(Account.checking_balance, Account.savings_balance) > 10000
    )

"false" expression
------------------

Render a "false" constant expression, rendering as "0" on platforms that
don't have a "false" constant::

    from sqlalchemy.sql import expression
    from sqlalchemy.ext.compiler import compiles


    class sql_false(expression.ColumnElement):
        inherit_cache = True


    @compiles(sql_false)
    def default_false(element, compiler, **kw):
        return "false"


    @compiles(sql_false, "mssql")
    @compiles(sql_false, "mysql")
    @compiles(sql_false, "oracle")
    def int_false(element, compiler, **kw):
        return "0"

Example usage::

    from sqlalchemy import select, union_all

    exp = union_all(
        select(users.c.name, sql_false().label("enrolled")),
        select(customers.c.name, customers.c.enrolled),
    )

"""
from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar

from .. import exc
from ..sql import sqltypes

if TYPE_CHECKING:
    from ..sql.compiler import SQLCompiler

_F = TypeVar("_F", bound=Callable[..., Any])


def compiles(class_: Type[Any], *specs: str) -> Callable[[_F], _F]:
    """Register a function as a compiler for a
    given :class:`_expression.ClauseElement` type."""

    def decorate(fn: _F) -> _F:
        # get an existing @compiles handler
        existing = class_.__dict__.get("_compiler_dispatcher", None)

        # get the original handler.  All ClauseElement classes have one
        # of these, but some TypeEngine classes will not.
        existing_dispatch = getattr(class_, "_compiler_dispatch", None)

        if not existing:
            existing = _dispatcher()

            if existing_dispatch:

                def _wrap_existing_dispatch(
                    element: Any, compiler: SQLCompiler, **kw: Any
                ) -> Any:
                    try:
                        return existing_dispatch(element, compiler, **kw)
                    except exc.UnsupportedCompilationError as uce:
                        raise exc.UnsupportedCompilationError(
                            compiler,
                            type(element),
                            message="%s construct has no default "
                            "compilation handler." % type(element),
                        ) from uce

                existing.specs["default"] = _wrap_existing_dispatch

            # TODO: why is the lambda needed ?
            setattr(
                class_,
                "_compiler_dispatch",
                lambda *arg, **kw: existing(*arg, **kw),
            )
            setattr(class_, "_compiler_dispatcher", existing)

        if specs:
            for s in specs:
                existing.specs[s] = fn

        else:
            existing.specs["default"] = fn
        return fn

    return decorate


def deregister(class_: Type[Any]) -> None:
    """Remove all custom compilers associated with a given
    :class:`_expression.ClauseElement` type.

    """

    if hasattr(class_, "_compiler_dispatcher"):
        class_._compiler_dispatch = class_._original_compiler_dispatch
        del class_._compiler_dispatcher


class _dispatcher:
    def __init__(self) -> None:
        self.specs: Dict[str, Callable[..., Any]] = {}

    def __call__(self, element: Any, compiler: SQLCompiler, **kw: Any) -> Any:
        # TODO: yes, this could also switch off of DBAPI in use.
        fn = self.specs.get(compiler.dialect.name, None)
        if not fn:
            try:
                fn = self.specs["default"]
            except KeyError as ke:
                raise exc.UnsupportedCompilationError(
                    compiler,
                    type(element),
                    message="%s construct has no default "
                    "compilation handler." % type(element),
                ) from ke

        # if compilation includes add_to_result_map, collect add_to_result_map
        # arguments from the user-defined callable, which are probably none
        # because this is not public API.  if it wasn't called, then call it
        # ourselves.
        arm = kw.get("add_to_result_map", None)
        if arm:
            arm_collection = []
            kw["add_to_result_map"] = lambda *args: arm_collection.append(args)

        expr = fn(element, compiler, **kw)

        if arm:
            if not arm_collection:
                arm_collection.append(
                    (None, None, (element,), sqltypes.NULLTYPE)
                )
            for tup in arm_collection:
                arm(*tup)
        return expr

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\dateutil\relativedelta.py ===
# -*- coding: utf-8 -*-
import datetime
import calendar

import operator
from math import copysign

from six import integer_types
from warnings import warn

from ._common import weekday

MO, TU, WE, TH, FR, SA, SU = weekdays = tuple(weekday(x) for x in range(7))

__all__ = ["relativedelta", "MO", "TU", "WE", "TH", "FR", "SA", "SU"]


class relativedelta(object):
    """
    The relativedelta type is designed to be applied to an existing datetime and
    can replace specific components of that datetime, or represents an interval
    of time.

    It is based on the specification of the excellent work done by M.-A. Lemburg
    in his
    `mx.DateTime <https://www.egenix.com/products/python/mxBase/mxDateTime/>`_ extension.
    However, notice that this type does *NOT* implement the same algorithm as
    his work. Do *NOT* expect it to behave like mx.DateTime's counterpart.

    There are two different ways to build a relativedelta instance. The
    first one is passing it two date/datetime classes::

        relativedelta(datetime1, datetime2)

    The second one is passing it any number of the following keyword arguments::

        relativedelta(arg1=x,arg2=y,arg3=z...)

        year, month, day, hour, minute, second, microsecond:
            Absolute information (argument is singular); adding or subtracting a
            relativedelta with absolute information does not perform an arithmetic
            operation, but rather REPLACES the corresponding value in the
            original datetime with the value(s) in relativedelta.

        years, months, weeks, days, hours, minutes, seconds, microseconds:
            Relative information, may be negative (argument is plural); adding
            or subtracting a relativedelta with relative information performs
            the corresponding arithmetic operation on the original datetime value
            with the information in the relativedelta.

        weekday:
            One of the weekday instances (MO, TU, etc) available in the
            relativedelta module. These instances may receive a parameter N,
            specifying the Nth weekday, which could be positive or negative
            (like MO(+1) or MO(-2)). Not specifying it is the same as specifying
            +1. You can also use an integer, where 0=MO. This argument is always
            relative e.g. if the calculated date is already Monday, using MO(1)
            or MO(-1) won't change the day. To effectively make it absolute, use
            it in combination with the day argument (e.g. day=1, MO(1) for first
            Monday of the month).

        leapdays:
            Will add given days to the date found, if year is a leap
            year, and the date found is post 28 of february.

        yearday, nlyearday:
            Set the yearday or the non-leap year day (jump leap days).
            These are converted to day/month/leapdays information.

    There are relative and absolute forms of the keyword
    arguments. The plural is relative, and the singular is
    absolute. For each argument in the order below, the absolute form
    is applied first (by setting each attribute to that value) and
    then the relative form (by adding the value to the attribute).

    The order of attributes considered when this relativedelta is
    added to a datetime is:

    1. Year
    2. Month
    3. Day
    4. Hours
    5. Minutes
    6. Seconds
    7. Microseconds

    Finally, weekday is applied, using the rule described above.

    For example

    >>> from datetime import datetime
    >>> from dateutil.relativedelta import relativedelta, MO
    >>> dt = datetime(2018, 4, 9, 13, 37, 0)
    >>> delta = relativedelta(hours=25, day=1, weekday=MO(1))
    >>> dt + delta
    datetime.datetime(2018, 4, 2, 14, 37)

    First, the day is set to 1 (the first of the month), then 25 hours
    are added, to get to the 2nd day and 14th hour, finally the
    weekday is applied, but since the 2nd is already a Monday there is
    no effect.

    """

    def __init__(self, dt1=None, dt2=None,
                 years=0, months=0, days=0, leapdays=0, weeks=0,
                 hours=0, minutes=0, seconds=0, microseconds=0,
                 year=None, month=None, day=None, weekday=None,
                 yearday=None, nlyearday=None,
                 hour=None, minute=None, second=None, microsecond=None):

        if dt1 and dt2:
            # datetime is a subclass of date. So both must be date
            if not (isinstance(dt1, datetime.date) and
                    isinstance(dt2, datetime.date)):
                raise TypeError("relativedelta only diffs datetime/date")

            # We allow two dates, or two datetimes, so we coerce them to be
            # of the same type
            if (isinstance(dt1, datetime.datetime) !=
                    isinstance(dt2, datetime.datetime)):
                if not isinstance(dt1, datetime.datetime):
                    dt1 = datetime.datetime.fromordinal(dt1.toordinal())
                elif not isinstance(dt2, datetime.datetime):
                    dt2 = datetime.datetime.fromordinal(dt2.toordinal())

            self.years = 0
            self.months = 0
            self.days = 0
            self.leapdays = 0
            self.hours = 0
            self.minutes = 0
            self.seconds = 0
            self.microseconds = 0
            self.year = None
            self.month = None
            self.day = None
            self.weekday = None
            self.hour = None
            self.minute = None
            self.second = None
            self.microsecond = None
            self._has_time = 0

            # Get year / month delta between the two
            months = (dt1.year - dt2.year) * 12 + (dt1.month - dt2.month)
            self._set_months(months)

            # Remove the year/month delta so the timedelta is just well-defined
            # time units (seconds, days and microseconds)
            dtm = self.__radd__(dt2)

            # If we've overshot our target, make an adjustment
            if dt1 < dt2:
                compare = operator.gt
                increment = 1
            else:
                compare = operator.lt
                increment = -1

            while compare(dt1, dtm):
                months += increment
                self._set_months(months)
                dtm = self.__radd__(dt2)

            # Get the timedelta between the "months-adjusted" date and dt1
            delta = dt1 - dtm
            self.seconds = delta.seconds + delta.days * 86400
            self.microseconds = delta.microseconds
        else:
            # Check for non-integer values in integer-only quantities
            if any(x is not None and x != int(x) for x in (years, months)):
                raise ValueError("Non-integer years and months are "
                                 "ambiguous and not currently supported.")

            # Relative information
            self.years = int(years)
            self.months = int(months)
            self.days = days + weeks * 7
            self.leapdays = leapdays
            self.hours = hours
            self.minutes = minutes
            self.seconds = seconds
            self.microseconds = microseconds

            # Absolute information
            self.year = year
            self.month = month
            self.day = day
            self.hour = hour
            self.minute = minute
            self.second = second
            self.microsecond = microsecond

            if any(x is not None and int(x) != x
                   for x in (year, month, day, hour,
                             minute, second, microsecond)):
                # For now we'll deprecate floats - later it'll be an error.
                warn("Non-integer value passed as absolute information. " +
                     "This is not a well-defined condition and will raise " +
                     "errors in future versions.", DeprecationWarning)

            if isinstance(weekday, integer_types):
                self.weekday = weekdays[weekday]
            else:
                self.weekday = weekday

            yday = 0
            if nlyearday:
                yday = nlyearday
            elif yearday:
                yday = yearday
                if yearday > 59:
                    self.leapdays = -1
            if yday:
                ydayidx = [31, 59, 90, 120, 151, 181, 212,
                           243, 273, 304, 334, 366]
                for idx, ydays in enumerate(ydayidx):
                    if yday <= ydays:
                        self.month = idx+1
                        if idx == 0:
                            self.day = yday
                        else:
                            self.day = yday-ydayidx[idx-1]
                        break
                else:
                    raise ValueError("invalid year day (%d)" % yday)

        self._fix()

    def _fix(self):
        if abs(self.microseconds) > 999999:
            s = _sign(self.microseconds)
            div, mod = divmod(self.microseconds * s, 1000000)
            self.microseconds = mod * s
            self.seconds += div * s
        if abs(self.seconds) > 59:
            s = _sign(self.seconds)
            div, mod = divmod(self.seconds * s, 60)
            self.seconds = mod * s
            self.minutes += div * s
        if abs(self.minutes) > 59:
            s = _sign(self.minutes)
            div, mod = divmod(self.minutes * s, 60)
            self.minutes = mod * s
            self.hours += div * s
        if abs(self.hours) > 23:
            s = _sign(self.hours)
            div, mod = divmod(self.hours * s, 24)
            self.hours = mod * s
            self.days += div * s
        if abs(self.months) > 11:
            s = _sign(self.months)
            div, mod = divmod(self.months * s, 12)
            self.months = mod * s
            self.years += div * s
        if (self.hours or self.minutes or self.seconds or self.microseconds
                or self.hour is not None or self.minute is not None or
                self.second is not None or self.microsecond is not None):
            self._has_time = 1
        else:
            self._has_time = 0

    @property
    def weeks(self):
        return int(self.days / 7.0)

    @weeks.setter
    def weeks(self, value):
        self.days = self.days - (self.weeks * 7) + value * 7

    def _set_months(self, months):
        self.months = months
        if abs(self.months) > 11:
            s = _sign(self.months)
            div, mod = divmod(self.months * s, 12)
            self.months = mod * s
            self.years = div * s
        else:
            self.years = 0

    def normalized(self):
        """
        Return a version of this object represented entirely using integer
        values for the relative attributes.

        >>> relativedelta(days=1.5, hours=2).normalized()
        relativedelta(days=+1, hours=+14)

        :return:
            Returns a :class:`dateutil.relativedelta.relativedelta` object.
        """
        # Cascade remainders down (rounding each to roughly nearest microsecond)
        days = int(self.days)

        hours_f = round(self.hours + 24 * (self.days - days), 11)
        hours = int(hours_f)

        minutes_f = round(self.minutes + 60 * (hours_f - hours), 10)
        minutes = int(minutes_f)

        seconds_f = round(self.seconds + 60 * (minutes_f - minutes), 8)
        seconds = int(seconds_f)

        microseconds = round(self.microseconds + 1e6 * (seconds_f - seconds))

        # Constructor carries overflow back up with call to _fix()
        return self.__class__(years=self.years, months=self.months,
                              days=days, hours=hours, minutes=minutes,
                              seconds=seconds, microseconds=microseconds,
                              leapdays=self.leapdays, year=self.year,
                              month=self.month, day=self.day,
                              weekday=self.weekday, hour=self.hour,
                              minute=self.minute, second=self.second,
                              microsecond=self.microsecond)

    def __add__(self, other):
        if isinstance(other, relativedelta):
            return self.__class__(years=other.years + self.years,
                                 months=other.months + self.months,
                                 days=other.days + self.days,
                                 hours=other.hours + self.hours,
                                 minutes=other.minutes + self.minutes,
                                 seconds=other.seconds + self.seconds,
                                 microseconds=(other.microseconds +
                                               self.microseconds),
                                 leapdays=other.leapdays or self.leapdays,
                                 year=(other.year if other.year is not None
                                       else self.year),
                                 month=(other.month if other.month is not None
                                        else self.month),
                                 day=(other.day if other.day is not None
                                      else self.day),
                                 weekday=(other.weekday if other.weekday is not None
                                          else self.weekday),
                                 hour=(other.hour if other.hour is not None
                                       else self.hour),
                                 minute=(other.minute if other.minute is not None
                                         else self.minute),
                                 second=(other.second if other.second is not None
                                         else self.second),
                                 microsecond=(other.microsecond if other.microsecond
                                              is not None else
                                              self.microsecond))
        if isinstance(other, datetime.timedelta):
            return self.__class__(years=self.years,
                                  months=self.months,
                                  days=self.days + other.days,
                                  hours=self.hours,
                                  minutes=self.minutes,
                                  seconds=self.seconds + other.seconds,
                                  microseconds=self.microseconds + other.microseconds,
                                  leapdays=self.leapdays,
                                  year=self.year,
                                  month=self.month,
                                  day=self.day,
                                  weekday=self.weekday,
                                  hour=self.hour,
                                  minute=self.minute,
                                  second=self.second,
                                  microsecond=self.microsecond)
        if not isinstance(other, datetime.date):
            return NotImplemented
        elif self._has_time and not isinstance(other, datetime.datetime):
            other = datetime.datetime.fromordinal(other.toordinal())
        year = (self.year or other.year)+self.years
        month = self.month or other.month
        if self.months:
            assert 1 <= abs(self.months) <= 12
            month += self.months
            if month > 12:
                year += 1
                month -= 12
            elif month < 1:
                year -= 1
                month += 12
        day = min(calendar.monthrange(year, month)[1],
                  self.day or other.day)
        repl = {"year": year, "month": month, "day": day}
        for attr in ["hour", "minute", "second", "microsecond"]:
            value = getattr(self, attr)
            if value is not None:
                repl[attr] = value
        days = self.days
        if self.leapdays and month > 2 and calendar.isleap(year):
            days += self.leapdays
        ret = (other.replace(**repl)
               + datetime.timedelta(days=days,
                                    hours=self.hours,
                                    minutes=self.minutes,
                                    seconds=self.seconds,
                                    microseconds=self.microseconds))
        if self.weekday:
            weekday, nth = self.weekday.weekday, self.weekday.n or 1
            jumpdays = (abs(nth) - 1) * 7
            if nth > 0:
                jumpdays += (7 - ret.weekday() + weekday) % 7
            else:
                jumpdays += (ret.weekday() - weekday) % 7
                jumpdays *= -1
            ret += datetime.timedelta(days=jumpdays)
        return ret

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__neg__().__radd__(other)

    def __sub__(self, other):
        if not isinstance(other, relativedelta):
            return NotImplemented   # In case the other object defines __rsub__
        return self.__class__(years=self.years - other.years,
                             months=self.months - other.months,
                             days=self.days - other.days,
                             hours=self.hours - other.hours,
                             minutes=self.minutes - other.minutes,
                             seconds=self.seconds - other.seconds,
                             microseconds=self.microseconds - other.microseconds,
                             leapdays=self.leapdays or other.leapdays,
                             year=(self.year if self.year is not None
                                   else other.year),
                             month=(self.month if self.month is not None else
                                    other.month),
                             day=(self.day if self.day is not None else
                                  other.day),
                             weekday=(self.weekday if self.weekday is not None else
                                      other.weekday),
                             hour=(self.hour if self.hour is not None else
                                   other.hour),
                             minute=(self.minute if self.minute is not None else
                                     other.minute),
                             second=(self.second if self.second is not None else
                                     other.second),
                             microsecond=(self.microsecond if self.microsecond
                                          is not None else
                                          other.microsecond))

    def __abs__(self):
        return self.__class__(years=abs(self.years),
                              months=abs(self.months),
                              days=abs(self.days),
                              hours=abs(self.hours),
                              minutes=abs(self.minutes),
                              seconds=abs(self.seconds),
                              microseconds=abs(self.microseconds),
                              leapdays=self.leapdays,
                              year=self.year,
                              month=self.month,
                              day=self.day,
                              weekday=self.weekday,
                              hour=self.hour,
                              minute=self.minute,
                              second=self.second,
                              microsecond=self.microsecond)

    def __neg__(self):
        return self.__class__(years=-self.years,
                             months=-self.months,
                             days=-self.days,
                             hours=-self.hours,
                             minutes=-self.minutes,
                             seconds=-self.seconds,
                             microseconds=-self.microseconds,
                             leapdays=self.leapdays,
                             year=self.year,
                             month=self.month,
                             day=self.day,
                             weekday=self.weekday,
                             hour=self.hour,
                             minute=self.minute,
                             second=self.second,
                             microsecond=self.microsecond)

    def __bool__(self):
        return not (not self.years and
                    not self.months and
                    not self.days and
                    not self.hours and
                    not self.minutes and
                    not self.seconds and
                    not self.microseconds and
                    not self.leapdays and
                    self.year is None and
                    self.month is None and
                    self.day is None and
                    self.weekday is None and
                    self.hour is None and
                    self.minute is None and
                    self.second is None and
                    self.microsecond is None)
    # Compatibility with Python 2.x
    __nonzero__ = __bool__

    def __mul__(self, other):
        try:
            f = float(other)
        except TypeError:
            return NotImplemented

        return self.__class__(years=int(self.years * f),
                             months=int(self.months * f),
                             days=int(self.days * f),
                             hours=int(self.hours * f),
                             minutes=int(self.minutes * f),
                             seconds=int(self.seconds * f),
                             microseconds=int(self.microseconds * f),
                             leapdays=self.leapdays,
                             year=self.year,
                             month=self.month,
                             day=self.day,
                             weekday=self.weekday,
                             hour=self.hour,
                             minute=self.minute,
                             second=self.second,
                             microsecond=self.microsecond)

    __rmul__ = __mul__

    def __eq__(self, other):
        if not isinstance(other, relativedelta):
            return NotImplemented
        if self.weekday or other.weekday:
            if not self.weekday or not other.weekday:
                return False
            if self.weekday.weekday != other.weekday.weekday:
                return False
            n1, n2 = self.weekday.n, other.weekday.n
            if n1 != n2 and not ((not n1 or n1 == 1) and (not n2 or n2 == 1)):
                return False
        return (self.years == other.years and
                self.months == other.months and
                self.days == other.days and
                self.hours == other.hours and
                self.minutes == other.minutes and
                self.seconds == other.seconds and
                self.microseconds == other.microseconds and
                self.leapdays == other.leapdays and
                self.year == other.year and
                self.month == other.month and
                self.day == other.day and
                self.hour == other.hour and
                self.minute == other.minute and
                self.second == other.second and
                self.microsecond == other.microsecond)

    def __hash__(self):
        return hash((
            self.weekday,
            self.years,
            self.months,
            self.days,
            self.hours,
            self.minutes,
            self.seconds,
            self.microseconds,
            self.leapdays,
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.microsecond,
        ))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __div__(self, other):
        try:
            reciprocal = 1 / float(other)
        except TypeError:
            return NotImplemented

        return self.__mul__(reciprocal)

    __truediv__ = __div__

    def __repr__(self):
        l = []
        for attr in ["years", "months", "days", "leapdays",
                     "hours", "minutes", "seconds", "microseconds"]:
            value = getattr(self, attr)
            if value:
                l.append("{attr}={value:+g}".format(attr=attr, value=value))
        for attr in ["year", "month", "day", "weekday",
                     "hour", "minute", "second", "microsecond"]:
            value = getattr(self, attr)
            if value is not None:
                l.append("{attr}={value}".format(attr=attr, value=repr(value)))
        return "{classname}({attrs})".format(classname=self.__class__.__name__,
                                             attrs=", ".join(l))


def _sign(x):
    return int(copysign(1, x))

# vim:ts=4:sw=4:et

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\internals\concat.py ===
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    cast,
)
import warnings

import numpy as np

from pandas._libs import (
    NaT,
    algos as libalgos,
    internals as libinternals,
    lib,
)
from pandas._libs.missing import NA
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import (
    ensure_dtype_can_hold_na,
    find_common_type,
)
from pandas.core.dtypes.common import (
    is_1d_only_ea_dtype,
    is_scalar,
    needs_i8_conversion,
)
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
    ExtensionDtype,
    SparseDtype,
)
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
    isna_all,
)

from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.internals.array_manager import ArrayManager
from pandas.core.internals.blocks import (
    ensure_block_shape,
    new_block_2d,
)
from pandas.core.internals.managers import (
    BlockManager,
    make_na_array,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pandas._typing import (
        ArrayLike,
        AxisInt,
        DtypeObj,
        Manager2D,
        Shape,
    )

    from pandas import Index
    from pandas.core.internals.blocks import (
        Block,
        BlockPlacement,
    )


def _concatenate_array_managers(
    mgrs: list[ArrayManager], axes: list[Index], concat_axis: AxisInt
) -> Manager2D:
    """
    Concatenate array managers into one.

    Parameters
    ----------
    mgrs_indexers : list of (ArrayManager, {axis: indexer,...}) tuples
    axes : list of Index
    concat_axis : int

    Returns
    -------
    ArrayManager
    """
    if concat_axis == 1:
        return mgrs[0].concat_vertical(mgrs, axes)
    else:
        # concatting along the columns -> combine reindexed arrays in a single manager
        assert concat_axis == 0
        return mgrs[0].concat_horizontal(mgrs, axes)


def concatenate_managers(
    mgrs_indexers, axes: list[Index], concat_axis: AxisInt, copy: bool
) -> Manager2D:
    """
    Concatenate block managers into one.

    Parameters
    ----------
    mgrs_indexers : list of (BlockManager, {axis: indexer,...}) tuples
    axes : list of Index
    concat_axis : int
    copy : bool

    Returns
    -------
    BlockManager
    """

    needs_copy = copy and concat_axis == 0

    # TODO(ArrayManager) this assumes that all managers are of the same type
    if isinstance(mgrs_indexers[0][0], ArrayManager):
        mgrs = _maybe_reindex_columns_na_proxy(axes, mgrs_indexers, needs_copy)
        # error: Argument 1 to "_concatenate_array_managers" has incompatible
        # type "List[BlockManager]"; expected "List[Union[ArrayManager,
        # SingleArrayManager, BlockManager, SingleBlockManager]]"
        return _concatenate_array_managers(
            mgrs, axes, concat_axis  # type: ignore[arg-type]
        )

    # Assertions disabled for performance
    # for tup in mgrs_indexers:
    #    # caller is responsible for ensuring this
    #    indexers = tup[1]
    #    assert concat_axis not in indexers

    if concat_axis == 0:
        mgrs = _maybe_reindex_columns_na_proxy(axes, mgrs_indexers, needs_copy)
        return mgrs[0].concat_horizontal(mgrs, axes)

    if len(mgrs_indexers) > 0 and mgrs_indexers[0][0].nblocks > 0:
        first_dtype = mgrs_indexers[0][0].blocks[0].dtype
        if first_dtype in [np.float64, np.float32]:
            # TODO: support more dtypes here.  This will be simpler once
            #  JoinUnit.is_na behavior is deprecated.
            if (
                all(_is_homogeneous_mgr(mgr, first_dtype) for mgr, _ in mgrs_indexers)
                and len(mgrs_indexers) > 1
            ):
                # Fastpath!
                # Length restriction is just to avoid having to worry about 'copy'
                shape = tuple(len(x) for x in axes)
                nb = _concat_homogeneous_fastpath(mgrs_indexers, shape, first_dtype)
                return BlockManager((nb,), axes)

    mgrs = _maybe_reindex_columns_na_proxy(axes, mgrs_indexers, needs_copy)

    if len(mgrs) == 1:
        mgr = mgrs[0]
        out = mgr.copy(deep=False)
        out.axes = axes
        return out

    concat_plan = _get_combined_plan(mgrs)

    blocks = []
    values: ArrayLike

    for placement, join_units in concat_plan:
        unit = join_units[0]
        blk = unit.block

        if _is_uniform_join_units(join_units):
            vals = [ju.block.values for ju in join_units]

            if not blk.is_extension:
                # _is_uniform_join_units ensures a single dtype, so
                #  we can use np.concatenate, which is more performant
                #  than concat_compat
                # error: Argument 1 to "concatenate" has incompatible type
                # "List[Union[ndarray[Any, Any], ExtensionArray]]";
                # expected "Union[_SupportsArray[dtype[Any]],
                # _NestedSequence[_SupportsArray[dtype[Any]]]]"
                values = np.concatenate(vals, axis=1)  # type: ignore[arg-type]
            elif is_1d_only_ea_dtype(blk.dtype):
                # TODO(EA2D): special-casing not needed with 2D EAs
                values = concat_compat(vals, axis=0, ea_compat_axis=True)
                values = ensure_block_shape(values, ndim=2)
            else:
                values = concat_compat(vals, axis=1)

            values = ensure_wrapped_if_datetimelike(values)

            fastpath = blk.values.dtype == values.dtype
        else:
            values = _concatenate_join_units(join_units, copy=copy)
            fastpath = False

        if fastpath:
            b = blk.make_block_same_class(values, placement=placement)
        else:
            b = new_block_2d(values, placement=placement)

        blocks.append(b)

    return BlockManager(tuple(blocks), axes)


def _maybe_reindex_columns_na_proxy(
    axes: list[Index],
    mgrs_indexers: list[tuple[BlockManager, dict[int, np.ndarray]]],
    needs_copy: bool,
) -> list[BlockManager]:
    """
    Reindex along columns so that all of the BlockManagers being concatenated
    have matching columns.

    Columns added in this reindexing have dtype=np.void, indicating they
    should be ignored when choosing a column's final dtype.
    """
    new_mgrs = []

    for mgr, indexers in mgrs_indexers:
        # For axis=0 (i.e. columns) we use_na_proxy and only_slice, so this
        #  is a cheap reindexing.
        for i, indexer in indexers.items():
            mgr = mgr.reindex_indexer(
                axes[i],
                indexers[i],
                axis=i,
                copy=False,
                only_slice=True,  # only relevant for i==0
                allow_dups=True,
                use_na_proxy=True,  # only relevant for i==0
            )
        if needs_copy and not indexers:
            mgr = mgr.copy()

        new_mgrs.append(mgr)
    return new_mgrs


def _is_homogeneous_mgr(mgr: BlockManager, first_dtype: DtypeObj) -> bool:
    """
    Check if this Manager can be treated as a single ndarray.
    """
    if mgr.nblocks != 1:
        return False
    blk = mgr.blocks[0]
    if not (blk.mgr_locs.is_slice_like and blk.mgr_locs.as_slice.step == 1):
        return False

    return blk.dtype == first_dtype


def _concat_homogeneous_fastpath(
    mgrs_indexers, shape: Shape, first_dtype: np.dtype
) -> Block:
    """
    With single-Block managers with homogeneous dtypes (that can already hold nan),
    we avoid [...]
    """
    # assumes
    #  all(_is_homogeneous_mgr(mgr, first_dtype) for mgr, _ in in mgrs_indexers)

    if all(not indexers for _, indexers in mgrs_indexers):
        # https://github.com/pandas-dev/pandas/pull/52685#issuecomment-1523287739
        arrs = [mgr.blocks[0].values.T for mgr, _ in mgrs_indexers]
        arr = np.concatenate(arrs).T
        bp = libinternals.BlockPlacement(slice(shape[0]))
        nb = new_block_2d(arr, bp)
        return nb

    arr = np.empty(shape, dtype=first_dtype)

    if first_dtype == np.float64:
        take_func = libalgos.take_2d_axis0_float64_float64
    else:
        take_func = libalgos.take_2d_axis0_float32_float32

    start = 0
    for mgr, indexers in mgrs_indexers:
        mgr_len = mgr.shape[1]
        end = start + mgr_len

        if 0 in indexers:
            take_func(
                mgr.blocks[0].values,
                indexers[0],
                arr[:, start:end],
            )
        else:
            # No reindexing necessary, we can copy values directly
            arr[:, start:end] = mgr.blocks[0].values

        start += mgr_len

    bp = libinternals.BlockPlacement(slice(shape[0]))
    nb = new_block_2d(arr, bp)
    return nb


def _get_combined_plan(
    mgrs: list[BlockManager],
) -> list[tuple[BlockPlacement, list[JoinUnit]]]:
    plan = []

    max_len = mgrs[0].shape[0]

    blknos_list = [mgr.blknos for mgr in mgrs]
    pairs = libinternals.get_concat_blkno_indexers(blknos_list)
    for ind, (blknos, bp) in enumerate(pairs):
        # assert bp.is_slice_like
        # assert len(bp) > 0

        units_for_bp = []
        for k, mgr in enumerate(mgrs):
            blkno = blknos[k]

            nb = _get_block_for_concat_plan(mgr, bp, blkno, max_len=max_len)
            unit = JoinUnit(nb)
            units_for_bp.append(unit)

        plan.append((bp, units_for_bp))

    return plan


def _get_block_for_concat_plan(
    mgr: BlockManager, bp: BlockPlacement, blkno: int, *, max_len: int
) -> Block:
    blk = mgr.blocks[blkno]
    # Assertions disabled for performance:
    #  assert bp.is_slice_like
    #  assert blkno != -1
    #  assert (mgr.blknos[bp] == blkno).all()

    if len(bp) == len(blk.mgr_locs) and (
        blk.mgr_locs.is_slice_like and blk.mgr_locs.as_slice.step == 1
    ):
        nb = blk
    else:
        ax0_blk_indexer = mgr.blklocs[bp.indexer]

        slc = lib.maybe_indices_to_slice(ax0_blk_indexer, max_len)
        # TODO: in all extant test cases 2023-04-08 we have a slice here.
        #  Will this always be the case?
        if isinstance(slc, slice):
            nb = blk.slice_block_columns(slc)
        else:
            nb = blk.take_block_columns(slc)

    # assert nb.shape == (len(bp), mgr.shape[1])
    return nb


class JoinUnit:
    def __init__(self, block: Block) -> None:
        self.block = block

    def __repr__(self) -> str:
        return f"{type(self).__name__}({repr(self.block)})"

    def _is_valid_na_for(self, dtype: DtypeObj) -> bool:
        """
        Check that we are all-NA of a type/dtype that is compatible with this dtype.
        Augments `self.is_na` with an additional check of the type of NA values.
        """
        if not self.is_na:
            return False

        blk = self.block
        if blk.dtype.kind == "V":
            return True

        if blk.dtype == object:
            values = blk.values
            return all(is_valid_na_for_dtype(x, dtype) for x in values.ravel(order="K"))

        na_value = blk.fill_value
        if na_value is NaT and blk.dtype != dtype:
            # e.g. we are dt64 and other is td64
            # fill_values match but we should not cast blk.values to dtype
            # TODO: this will need updating if we ever have non-nano dt64/td64
            return False

        if na_value is NA and needs_i8_conversion(dtype):
            # FIXME: kludge; test_append_empty_frame_with_timedelta64ns_nat
            #  e.g. blk.dtype == "Int64" and dtype is td64, we dont want
            #  to consider these as matching
            return False

        # TODO: better to use can_hold_element?
        return is_valid_na_for_dtype(na_value, dtype)

    @cache_readonly
    def is_na(self) -> bool:
        blk = self.block
        if blk.dtype.kind == "V":
            return True

        if not blk._can_hold_na:
            return False

        values = blk.values
        if values.size == 0:
            # GH#39122 this case will return False once deprecation is enforced
            return True

        if isinstance(values.dtype, SparseDtype):
            return False

        if values.ndim == 1:
            # TODO(EA2D): no need for special case with 2D EAs
            val = values[0]
            if not is_scalar(val) or not isna(val):
                # ideally isna_all would do this short-circuiting
                return False
            return isna_all(values)
        else:
            val = values[0][0]
            if not is_scalar(val) or not isna(val):
                # ideally isna_all would do this short-circuiting
                return False
            return all(isna_all(row) for row in values)

    @cache_readonly
    def is_na_after_size_and_isna_all_deprecation(self) -> bool:
        """
        Will self.is_na be True after values.size == 0 deprecation and isna_all
        deprecation are enforced?
        """
        blk = self.block
        if blk.dtype.kind == "V":
            return True
        return False

    def get_reindexed_values(self, empty_dtype: DtypeObj, upcasted_na) -> ArrayLike:
        values: ArrayLike

        if upcasted_na is None and self.block.dtype.kind != "V":
            # No upcasting is necessary
            return self.block.values
        else:
            fill_value = upcasted_na

            if self._is_valid_na_for(empty_dtype):
                # note: always holds when self.block.dtype.kind == "V"
                blk_dtype = self.block.dtype

                if blk_dtype == np.dtype("object"):
                    # we want to avoid filling with np.nan if we are
                    # using None; we already know that we are all
                    # nulls
                    values = cast(np.ndarray, self.block.values)
                    if values.size and values[0, 0] is None:
                        fill_value = None

                return make_na_array(empty_dtype, self.block.shape, fill_value)

            return self.block.values


def _concatenate_join_units(join_units: list[JoinUnit], copy: bool) -> ArrayLike:
    """
    Concatenate values from several join units along axis=1.
    """
    empty_dtype, empty_dtype_future = _get_empty_dtype(join_units)

    has_none_blocks = any(unit.block.dtype.kind == "V" for unit in join_units)
    upcasted_na = _dtype_to_na_value(empty_dtype, has_none_blocks)

    to_concat = [
        ju.get_reindexed_values(empty_dtype=empty_dtype, upcasted_na=upcasted_na)
        for ju in join_units
    ]

    if any(is_1d_only_ea_dtype(t.dtype) for t in to_concat):
        # TODO(EA2D): special case not needed if all EAs used HybridBlocks

        # error: No overload variant of "__getitem__" of "ExtensionArray" matches
        # argument type "Tuple[int, slice]"
        to_concat = [
            t
            if is_1d_only_ea_dtype(t.dtype)
            else t[0, :]  # type: ignore[call-overload]
            for t in to_concat
        ]
        concat_values = concat_compat(to_concat, axis=0, ea_compat_axis=True)
        concat_values = ensure_block_shape(concat_values, 2)

    else:
        concat_values = concat_compat(to_concat, axis=1)

    if empty_dtype != empty_dtype_future:
        if empty_dtype == concat_values.dtype:
            # GH#39122, GH#40893
            warnings.warn(
                "The behavior of DataFrame concatenation with empty or all-NA "
                "entries is deprecated. In a future version, this will no longer "
                "exclude empty or all-NA columns when determining the result dtypes. "
                "To retain the old behavior, exclude the relevant entries before "
                "the concat operation.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
    return concat_values


def _dtype_to_na_value(dtype: DtypeObj, has_none_blocks: bool):
    """
    Find the NA value to go with this dtype.
    """
    if isinstance(dtype, ExtensionDtype):
        return dtype.na_value
    elif dtype.kind in "mM":
        return dtype.type("NaT")
    elif dtype.kind in "fc":
        return dtype.type("NaN")
    elif dtype.kind == "b":
        # different from missing.na_value_for_dtype
        return None
    elif dtype.kind in "iu":
        if not has_none_blocks:
            # different from missing.na_value_for_dtype
            return None
        return np.nan
    elif dtype.kind == "O":
        return np.nan
    raise NotImplementedError


def _get_empty_dtype(join_units: Sequence[JoinUnit]) -> tuple[DtypeObj, DtypeObj]:
    """
    Return dtype and N/A values to use when concatenating specified units.

    Returned N/A value may be None which means there was no casting involved.

    Returns
    -------
    dtype
    """
    if lib.dtypes_all_equal([ju.block.dtype for ju in join_units]):
        empty_dtype = join_units[0].block.dtype
        return empty_dtype, empty_dtype

    has_none_blocks = any(unit.block.dtype.kind == "V" for unit in join_units)

    dtypes = [unit.block.dtype for unit in join_units if not unit.is_na]
    if not len(dtypes):
        dtypes = [
            unit.block.dtype for unit in join_units if unit.block.dtype.kind != "V"
        ]

    dtype = find_common_type(dtypes)
    if has_none_blocks:
        dtype = ensure_dtype_can_hold_na(dtype)

    dtype_future = dtype
    if len(dtypes) != len(join_units):
        dtypes_future = [
            unit.block.dtype
            for unit in join_units
            if not unit.is_na_after_size_and_isna_all_deprecation
        ]
        if not len(dtypes_future):
            dtypes_future = [
                unit.block.dtype for unit in join_units if unit.block.dtype.kind != "V"
            ]

        if len(dtypes) != len(dtypes_future):
            dtype_future = find_common_type(dtypes_future)
            if has_none_blocks:
                dtype_future = ensure_dtype_can_hold_na(dtype_future)

    return dtype, dtype_future


def _is_uniform_join_units(join_units: list[JoinUnit]) -> bool:
    """
    Check if the join units consist of blocks of uniform type that can
    be concatenated using Block.concat_same_type instead of the generic
    _concatenate_join_units (which uses `concat_compat`).

    """
    first = join_units[0].block
    if first.dtype.kind == "V":
        return False
    return (
        # exclude cases where a) ju.block is None or b) we have e.g. Int64+int64
        all(type(ju.block) is type(first) for ju in join_units)
        and
        # e.g. DatetimeLikeBlock can be dt64 or td64, but these are not uniform
        all(
            ju.block.dtype == first.dtype
            # GH#42092 we only want the dtype_equal check for non-numeric blocks
            #  (for now, may change but that would need a deprecation)
            or ju.block.dtype.kind in "iub"
            for ju in join_units
        )
        and
        # no blocks that would get missing values (can lead to type upcasts)
        # unless we're an extension dtype.
        all(not ju.is_na or ju.block.is_extension for ju in join_units)
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\resolution\legacy\resolver.py ===
"""Dependency Resolution

The dependency resolution in pip is performed as follows:

for top-level requirements:
    a. only one spec allowed per project, regardless of conflicts or not.
       otherwise a "double requirement" exception is raised
    b. they override sub-dependency requirements.
for sub-dependencies
    a. "first found, wins" (where the order is breadth first)
"""

import logging
import sys
from collections import defaultdict
from itertools import chain
from typing import DefaultDict, Iterable, List, Optional, Set, Tuple

from pip._vendor.packaging import specifiers
from pip._vendor.packaging.requirements import Requirement

from pip._internal.cache import WheelCache
from pip._internal.exceptions import (
    BestVersionAlreadyInstalled,
    DistributionNotFound,
    HashError,
    HashErrors,
    InstallationError,
    NoneMetadataError,
    UnsupportedPythonVersion,
)
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import BaseDistribution
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.operations.prepare import RequirementPreparer
from pip._internal.req.req_install import (
    InstallRequirement,
    check_invalid_constraint_type,
)
from pip._internal.req.req_set import RequirementSet
from pip._internal.resolution.base import BaseResolver, InstallRequirementProvider
from pip._internal.utils import compatibility_tags
from pip._internal.utils.compatibility_tags import get_supported
from pip._internal.utils.direct_url_helpers import direct_url_from_link
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import normalize_version_info
from pip._internal.utils.packaging import check_requires_python

logger = logging.getLogger(__name__)

DiscoveredDependencies = DefaultDict[Optional[str], List[InstallRequirement]]


def _check_dist_requires_python(
    dist: BaseDistribution,
    version_info: Tuple[int, int, int],
    ignore_requires_python: bool = False,
) -> None:
    """
    Check whether the given Python version is compatible with a distribution's
    "Requires-Python" value.

    :param version_info: A 3-tuple of ints representing the Python
        major-minor-micro version to check.
    :param ignore_requires_python: Whether to ignore the "Requires-Python"
        value if the given Python version isn't compatible.

    :raises UnsupportedPythonVersion: When the given Python version isn't
        compatible.
    """
    # This idiosyncratically converts the SpecifierSet to str and let
    # check_requires_python then parse it again into SpecifierSet. But this
    # is the legacy resolver so I'm just not going to bother refactoring.
    try:
        requires_python = str(dist.requires_python)
    except FileNotFoundError as e:
        raise NoneMetadataError(dist, str(e))
    try:
        is_compatible = check_requires_python(
            requires_python,
            version_info=version_info,
        )
    except specifiers.InvalidSpecifier as exc:
        logger.warning(
            "Package %r has an invalid Requires-Python: %s", dist.raw_name, exc
        )
        return

    if is_compatible:
        return

    version = ".".join(map(str, version_info))
    if ignore_requires_python:
        logger.debug(
            "Ignoring failed Requires-Python check for package %r: %s not in %r",
            dist.raw_name,
            version,
            requires_python,
        )
        return

    raise UnsupportedPythonVersion(
        f"Package {dist.raw_name!r} requires a different Python: "
        f"{version} not in {requires_python!r}"
    )


class Resolver(BaseResolver):
    """Resolves which packages need to be installed/uninstalled to perform \
    the requested operation without breaking the requirements of any package.
    """

    _allowed_strategies = {"eager", "only-if-needed", "to-satisfy-only"}

    def __init__(
        self,
        preparer: RequirementPreparer,
        finder: PackageFinder,
        wheel_cache: Optional[WheelCache],
        make_install_req: InstallRequirementProvider,
        use_user_site: bool,
        ignore_dependencies: bool,
        ignore_installed: bool,
        ignore_requires_python: bool,
        force_reinstall: bool,
        upgrade_strategy: str,
        py_version_info: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__()
        assert upgrade_strategy in self._allowed_strategies

        if py_version_info is None:
            py_version_info = sys.version_info[:3]
        else:
            py_version_info = normalize_version_info(py_version_info)

        self._py_version_info = py_version_info

        self.preparer = preparer
        self.finder = finder
        self.wheel_cache = wheel_cache

        self.upgrade_strategy = upgrade_strategy
        self.force_reinstall = force_reinstall
        self.ignore_dependencies = ignore_dependencies
        self.ignore_installed = ignore_installed
        self.ignore_requires_python = ignore_requires_python
        self.use_user_site = use_user_site
        self._make_install_req = make_install_req

        self._discovered_dependencies: DiscoveredDependencies = defaultdict(list)

    def resolve(
        self, root_reqs: List[InstallRequirement], check_supported_wheels: bool
    ) -> RequirementSet:
        """Resolve what operations need to be done

        As a side-effect of this method, the packages (and their dependencies)
        are downloaded, unpacked and prepared for installation. This
        preparation is done by ``pip.operations.prepare``.

        Once PyPI has static dependency metadata available, it would be
        possible to move the preparation to become a step separated from
        dependency resolution.
        """
        requirement_set = RequirementSet(check_supported_wheels=check_supported_wheels)
        for req in root_reqs:
            if req.constraint:
                check_invalid_constraint_type(req)
            self._add_requirement_to_set(requirement_set, req)

        # Actually prepare the files, and collect any exceptions. Most hash
        # exceptions cannot be checked ahead of time, because
        # _populate_link() needs to be called before we can make decisions
        # based on link type.
        discovered_reqs: List[InstallRequirement] = []
        hash_errors = HashErrors()
        for req in chain(requirement_set.all_requirements, discovered_reqs):
            try:
                discovered_reqs.extend(self._resolve_one(requirement_set, req))
            except HashError as exc:
                exc.req = req
                hash_errors.append(exc)

        if hash_errors:
            raise hash_errors

        return requirement_set

    def _add_requirement_to_set(
        self,
        requirement_set: RequirementSet,
        install_req: InstallRequirement,
        parent_req_name: Optional[str] = None,
        extras_requested: Optional[Iterable[str]] = None,
    ) -> Tuple[List[InstallRequirement], Optional[InstallRequirement]]:
        """Add install_req as a requirement to install.

        :param parent_req_name: The name of the requirement that needed this
            added. The name is used because when multiple unnamed requirements
            resolve to the same name, we could otherwise end up with dependency
            links that point outside the Requirements set. parent_req must
            already be added. Note that None implies that this is a user
            supplied requirement, vs an inferred one.
        :param extras_requested: an iterable of extras used to evaluate the
            environment markers.
        :return: Additional requirements to scan. That is either [] if
            the requirement is not applicable, or [install_req] if the
            requirement is applicable and has just been added.
        """
        # If the markers do not match, ignore this requirement.
        if not install_req.match_markers(extras_requested):
            logger.info(
                "Ignoring %s: markers '%s' don't match your environment",
                install_req.name,
                install_req.markers,
            )
            return [], None

        # If the wheel is not supported, raise an error.
        # Should check this after filtering out based on environment markers to
        # allow specifying different wheels based on the environment/OS, in a
        # single requirements file.
        if install_req.link and install_req.link.is_wheel:
            wheel = Wheel(install_req.link.filename)
            tags = compatibility_tags.get_supported()
            if requirement_set.check_supported_wheels and not wheel.supported(tags):
                raise InstallationError(
                    f"{wheel.filename} is not a supported wheel on this platform."
                )

        # This next bit is really a sanity check.
        assert (
            not install_req.user_supplied or parent_req_name is None
        ), "a user supplied req shouldn't have a parent"

        # Unnamed requirements are scanned again and the requirement won't be
        # added as a dependency until after scanning.
        if not install_req.name:
            requirement_set.add_unnamed_requirement(install_req)
            return [install_req], None

        try:
            existing_req: Optional[InstallRequirement] = (
                requirement_set.get_requirement(install_req.name)
            )
        except KeyError:
            existing_req = None

        has_conflicting_requirement = (
            parent_req_name is None
            and existing_req
            and not existing_req.constraint
            and existing_req.extras == install_req.extras
            and existing_req.req
            and install_req.req
            and existing_req.req.specifier != install_req.req.specifier
        )
        if has_conflicting_requirement:
            raise InstallationError(
                f"Double requirement given: {install_req} "
                f"(already in {existing_req}, name={install_req.name!r})"
            )

        # When no existing requirement exists, add the requirement as a
        # dependency and it will be scanned again after.
        if not existing_req:
            requirement_set.add_named_requirement(install_req)
            # We'd want to rescan this requirement later
            return [install_req], install_req

        # Assume there's no need to scan, and that we've already
        # encountered this for scanning.
        if install_req.constraint or not existing_req.constraint:
            return [], existing_req

        does_not_satisfy_constraint = install_req.link and not (
            existing_req.link and install_req.link.path == existing_req.link.path
        )
        if does_not_satisfy_constraint:
            raise InstallationError(
                f"Could not satisfy constraints for '{install_req.name}': "
                "installation from path or url cannot be "
                "constrained to a version"
            )
        # If we're now installing a constraint, mark the existing
        # object for real installation.
        existing_req.constraint = False
        # If we're now installing a user supplied requirement,
        # mark the existing object as such.
        if install_req.user_supplied:
            existing_req.user_supplied = True
        existing_req.extras = tuple(
            sorted(set(existing_req.extras) | set(install_req.extras))
        )
        logger.debug(
            "Setting %s extras to: %s",
            existing_req,
            existing_req.extras,
        )
        # Return the existing requirement for addition to the parent and
        # scanning again.
        return [existing_req], existing_req

    def _is_upgrade_allowed(self, req: InstallRequirement) -> bool:
        if self.upgrade_strategy == "to-satisfy-only":
            return False
        elif self.upgrade_strategy == "eager":
            return True
        else:
            assert self.upgrade_strategy == "only-if-needed"
            return req.user_supplied or req.constraint

    def _set_req_to_reinstall(self, req: InstallRequirement) -> None:
        """
        Set a requirement to be installed.
        """
        # Don't uninstall the conflict if doing a user install and the
        # conflict is not a user install.
        assert req.satisfied_by is not None
        if not self.use_user_site or req.satisfied_by.in_usersite:
            req.should_reinstall = True
        req.satisfied_by = None

    def _check_skip_installed(
        self, req_to_install: InstallRequirement
    ) -> Optional[str]:
        """Check if req_to_install should be skipped.

        This will check if the req is installed, and whether we should upgrade
        or reinstall it, taking into account all the relevant user options.

        After calling this req_to_install will only have satisfied_by set to
        None if the req_to_install is to be upgraded/reinstalled etc. Any
        other value will be a dist recording the current thing installed that
        satisfies the requirement.

        Note that for vcs urls and the like we can't assess skipping in this
        routine - we simply identify that we need to pull the thing down,
        then later on it is pulled down and introspected to assess upgrade/
        reinstalls etc.

        :return: A text reason for why it was skipped, or None.
        """
        if self.ignore_installed:
            return None

        req_to_install.check_if_exists(self.use_user_site)
        if not req_to_install.satisfied_by:
            return None

        if self.force_reinstall:
            self._set_req_to_reinstall(req_to_install)
            return None

        if not self._is_upgrade_allowed(req_to_install):
            if self.upgrade_strategy == "only-if-needed":
                return "already satisfied, skipping upgrade"
            return "already satisfied"

        # Check for the possibility of an upgrade.  For link-based
        # requirements we have to pull the tree down and inspect to assess
        # the version #, so it's handled way down.
        if not req_to_install.link:
            try:
                self.finder.find_requirement(req_to_install, upgrade=True)
            except BestVersionAlreadyInstalled:
                # Then the best version is installed.
                return "already up-to-date"
            except DistributionNotFound:
                # No distribution found, so we squash the error.  It will
                # be raised later when we re-try later to do the install.
                # Why don't we just raise here?
                pass

        self._set_req_to_reinstall(req_to_install)
        return None

    def _find_requirement_link(self, req: InstallRequirement) -> Optional[Link]:
        upgrade = self._is_upgrade_allowed(req)
        best_candidate = self.finder.find_requirement(req, upgrade)
        if not best_candidate:
            return None

        # Log a warning per PEP 592 if necessary before returning.
        link = best_candidate.link
        if link.is_yanked:
            reason = link.yanked_reason or "<none given>"
            msg = (
                # Mark this as a unicode string to prevent
                # "UnicodeEncodeError: 'ascii' codec can't encode character"
                # in Python 2 when the reason contains non-ascii characters.
                "The candidate selected for download or install is a "
                f"yanked version: {best_candidate}\n"
                f"Reason for being yanked: {reason}"
            )
            logger.warning(msg)

        return link

    def _populate_link(self, req: InstallRequirement) -> None:
        """Ensure that if a link can be found for this, that it is found.

        Note that req.link may still be None - if the requirement is already
        installed and not needed to be upgraded based on the return value of
        _is_upgrade_allowed().

        If preparer.require_hashes is True, don't use the wheel cache, because
        cached wheels, always built locally, have different hashes than the
        files downloaded from the index server and thus throw false hash
        mismatches. Furthermore, cached wheels at present have undeterministic
        contents due to file modification times.
        """
        if req.link is None:
            req.link = self._find_requirement_link(req)

        if self.wheel_cache is None or self.preparer.require_hashes:
            return

        assert req.link is not None, "_find_requirement_link unexpectedly returned None"
        cache_entry = self.wheel_cache.get_cache_entry(
            link=req.link,
            package_name=req.name,
            supported_tags=get_supported(),
        )
        if cache_entry is not None:
            logger.debug("Using cached wheel link: %s", cache_entry.link)
            if req.link is req.original_link and cache_entry.persistent:
                req.cached_wheel_source_link = req.link
            if cache_entry.origin is not None:
                req.download_info = cache_entry.origin
            else:
                # Legacy cache entry that does not have origin.json.
                # download_info may miss the archive_info.hashes field.
                req.download_info = direct_url_from_link(
                    req.link, link_is_in_wheel_cache=cache_entry.persistent
                )
            req.link = cache_entry.link

    def _get_dist_for(self, req: InstallRequirement) -> BaseDistribution:
        """Takes a InstallRequirement and returns a single AbstractDist \
        representing a prepared variant of the same.
        """
        if req.editable:
            return self.preparer.prepare_editable_requirement(req)

        # satisfied_by is only evaluated by calling _check_skip_installed,
        # so it must be None here.
        assert req.satisfied_by is None
        skip_reason = self._check_skip_installed(req)

        if req.satisfied_by:
            return self.preparer.prepare_installed_requirement(req, skip_reason)

        # We eagerly populate the link, since that's our "legacy" behavior.
        self._populate_link(req)
        dist = self.preparer.prepare_linked_requirement(req)

        # NOTE
        # The following portion is for determining if a certain package is
        # going to be re-installed/upgraded or not and reporting to the user.
        # This should probably get cleaned up in a future refactor.

        # req.req is only avail after unpack for URL
        # pkgs repeat check_if_exists to uninstall-on-upgrade
        # (#14)
        if not self.ignore_installed:
            req.check_if_exists(self.use_user_site)

        if req.satisfied_by:
            should_modify = (
                self.upgrade_strategy != "to-satisfy-only"
                or self.force_reinstall
                or self.ignore_installed
                or req.link.scheme == "file"
            )
            if should_modify:
                self._set_req_to_reinstall(req)
            else:
                logger.info(
                    "Requirement already satisfied (use --upgrade to upgrade): %s",
                    req,
                )
        return dist

    def _resolve_one(
        self,
        requirement_set: RequirementSet,
        req_to_install: InstallRequirement,
    ) -> List[InstallRequirement]:
        """Prepare a single requirements file.

        :return: A list of additional InstallRequirements to also install.
        """
        # Tell user what we are doing for this requirement:
        # obtain (editable), skipping, processing (local url), collecting
        # (remote url or package name)
        if req_to_install.constraint or req_to_install.prepared:
            return []

        req_to_install.prepared = True

        # Parse and return dependencies
        dist = self._get_dist_for(req_to_install)
        # This will raise UnsupportedPythonVersion if the given Python
        # version isn't compatible with the distribution's Requires-Python.
        _check_dist_requires_python(
            dist,
            version_info=self._py_version_info,
            ignore_requires_python=self.ignore_requires_python,
        )

        more_reqs: List[InstallRequirement] = []

        def add_req(subreq: Requirement, extras_requested: Iterable[str]) -> None:
            # This idiosyncratically converts the Requirement to str and let
            # make_install_req then parse it again into Requirement. But this is
            # the legacy resolver so I'm just not going to bother refactoring.
            sub_install_req = self._make_install_req(str(subreq), req_to_install)
            parent_req_name = req_to_install.name
            to_scan_again, add_to_parent = self._add_requirement_to_set(
                requirement_set,
                sub_install_req,
                parent_req_name=parent_req_name,
                extras_requested=extras_requested,
            )
            if parent_req_name and add_to_parent:
                self._discovered_dependencies[parent_req_name].append(add_to_parent)
            more_reqs.extend(to_scan_again)

        with indent_log():
            # We add req_to_install before its dependencies, so that we
            # can refer to it when adding dependencies.
            assert req_to_install.name is not None
            if not requirement_set.has_requirement(req_to_install.name):
                # 'unnamed' requirements will get added here
                # 'unnamed' requirements can only come from being directly
                # provided by the user.
                assert req_to_install.user_supplied
                self._add_requirement_to_set(
                    requirement_set, req_to_install, parent_req_name=None
                )

            if not self.ignore_dependencies:
                if req_to_install.extras:
                    logger.debug(
                        "Installing extra requirements: %r",
                        ",".join(req_to_install.extras),
                    )
                missing_requested = sorted(
                    set(req_to_install.extras) - set(dist.iter_provided_extras())
                )
                for missing in missing_requested:
                    logger.warning(
                        "%s %s does not provide the extra '%s'",
                        dist.raw_name,
                        dist.version,
                        missing,
                    )

                available_requested = sorted(
                    set(dist.iter_provided_extras()) & set(req_to_install.extras)
                )
                for subreq in dist.iter_dependencies(available_requested):
                    add_req(subreq, extras_requested=available_requested)

        return more_reqs

    def get_installation_order(
        self, req_set: RequirementSet
    ) -> List[InstallRequirement]:
        """Create the installation order.

        The installation order is topological - requirements are installed
        before the requiring thing. We break cycles at an arbitrary point,
        and make no other guarantees.
        """
        # The current implementation, which we may change at any point
        # installs the user specified things in the order given, except when
        # dependencies must come earlier to achieve topological order.
        order = []
        ordered_reqs: Set[InstallRequirement] = set()

        def schedule(req: InstallRequirement) -> None:
            if req.satisfied_by or req in ordered_reqs:
                return
            if req.constraint:
                return
            ordered_reqs.add(req)
            for dep in self._discovered_dependencies[req.name]:
                schedule(dep)
            order.append(req)

        for install_req in req_set.requirements.values():
            schedule(install_req)
        return order

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\discrete\convolutions.py ===
"""
Convolution (using **FFT**, **NTT**, **FWHT**), Subset Convolution,
Covering Product, Intersecting Product
"""

from sympy.core import S, sympify, Rational
from sympy.core.function import expand_mul
from sympy.discrete.transforms import (
    fft, ifft, ntt, intt, fwht, ifwht,
    mobius_transform, inverse_mobius_transform)
from sympy.external.gmpy import MPZ, lcm
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import as_int


def convolution(a, b, cycle=0, dps=None, prime=None, dyadic=None, subset=None):
    """
    Performs convolution by determining the type of desired
    convolution using hints.

    Exactly one of ``dps``, ``prime``, ``dyadic``, ``subset`` arguments
    should be specified explicitly for identifying the type of convolution,
    and the argument ``cycle`` can be specified optionally.

    For the default arguments, linear convolution is performed using **FFT**.

    Parameters
    ==========

    a, b : iterables
        The sequences for which convolution is performed.
    cycle : Integer
        Specifies the length for doing cyclic convolution.
    dps : Integer
        Specifies the number of decimal digits for precision for
        performing **FFT** on the sequence.
    prime : Integer
        Prime modulus of the form `(m 2^k + 1)` to be used for
        performing **NTT** on the sequence.
    dyadic : bool
        Identifies the convolution type as dyadic (*bitwise-XOR*)
        convolution, which is performed using **FWHT**.
    subset : bool
        Identifies the convolution type as subset convolution.

    Examples
    ========

    >>> from sympy import convolution, symbols, S, I
    >>> u, v, w, x, y, z = symbols('u v w x y z')

    >>> convolution([1 + 2*I, 4 + 3*I], [S(5)/4, 6], dps=3)
    [1.25 + 2.5*I, 11.0 + 15.8*I, 24.0 + 18.0*I]
    >>> convolution([1, 2, 3], [4, 5, 6], cycle=3)
    [31, 31, 28]

    >>> convolution([111, 777], [888, 444], prime=19*2**10 + 1)
    [1283, 19351, 14219]
    >>> convolution([111, 777], [888, 444], prime=19*2**10 + 1, cycle=2)
    [15502, 19351]

    >>> convolution([u, v], [x, y, z], dyadic=True)
    [u*x + v*y, u*y + v*x, u*z, v*z]
    >>> convolution([u, v], [x, y, z], dyadic=True, cycle=2)
    [u*x + u*z + v*y, u*y + v*x + v*z]

    >>> convolution([u, v, w], [x, y, z], subset=True)
    [u*x, u*y + v*x, u*z + w*x, v*z + w*y]
    >>> convolution([u, v, w], [x, y, z], subset=True, cycle=3)
    [u*x + v*z + w*y, u*y + v*x, u*z + w*x]

    """

    c = as_int(cycle)
    if c < 0:
        raise ValueError("The length for cyclic convolution "
                        "must be non-negative")

    dyadic = True if dyadic else None
    subset = True if subset else None
    if sum(x is not None for x in (prime, dps, dyadic, subset)) > 1:
        raise TypeError("Ambiguity in determining the type of convolution")

    if prime is not None:
        ls = convolution_ntt(a, b, prime=prime)
        return ls if not c else [sum(ls[i::c]) % prime for i in range(c)]

    if dyadic:
        ls = convolution_fwht(a, b)
    elif subset:
        ls = convolution_subset(a, b)
    else:
        def loop(a):
            dens = []
            for i in a:
                if isinstance(i, Rational) and i.q - 1:
                    dens.append(i.q)
                elif not isinstance(i, int):
                    return
            if dens:
                l = lcm(*dens)
                return [i*l if type(i) is int else i.p*(l//i.q) for i in a], l
            # no lcm of den to deal with
            return a, 1
        ls = None
        da = loop(a)
        if da is not None:
            db = loop(b)
            if db is not None:
                (ia, ma), (ib, mb) = da, db
                den = ma*mb
                ls = convolution_int(ia, ib)
                if den != 1:
                    ls = [Rational(i, den) for i in ls]
        if ls is None:
            ls = convolution_fft(a, b, dps)

    return ls if not c else [sum(ls[i::c]) for i in range(c)]


#----------------------------------------------------------------------------#
#                                                                            #
#                       Convolution for Complex domain                       #
#                                                                            #
#----------------------------------------------------------------------------#

def convolution_fft(a, b, dps=None):
    """
    Performs linear convolution using Fast Fourier Transform.

    Parameters
    ==========

    a, b : iterables
        The sequences for which convolution is performed.
    dps : Integer
        Specifies the number of decimal digits for precision.

    Examples
    ========

    >>> from sympy import S, I
    >>> from sympy.discrete.convolutions import convolution_fft

    >>> convolution_fft([2, 3], [4, 5])
    [8, 22, 15]
    >>> convolution_fft([2, 5], [6, 7, 3])
    [12, 44, 41, 15]
    >>> convolution_fft([1 + 2*I, 4 + 3*I], [S(5)/4, 6])
    [5/4 + 5*I/2, 11 + 63*I/4, 24 + 18*I]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Convolution_theorem
    .. [2] https://en.wikipedia.org/wiki/Discrete_Fourier_transform_(general%29

    """

    a, b = a[:], b[:]
    n = m = len(a) + len(b) - 1 # convolution size

    if n > 0 and n&(n - 1): # not a power of 2
        n = 2**n.bit_length()

    # padding with zeros
    a += [S.Zero]*(n - len(a))
    b += [S.Zero]*(n - len(b))

    a, b = fft(a, dps), fft(b, dps)
    a = [expand_mul(x*y) for x, y in zip(a, b)]
    a = ifft(a, dps)[:m]

    return a


#----------------------------------------------------------------------------#
#                                                                            #
#                           Convolution for GF(p)                            #
#                                                                            #
#----------------------------------------------------------------------------#

def convolution_ntt(a, b, prime):
    """
    Performs linear convolution using Number Theoretic Transform.

    Parameters
    ==========

    a, b : iterables
        The sequences for which convolution is performed.
    prime : Integer
        Prime modulus of the form `(m 2^k + 1)` to be used for performing
        **NTT** on the sequence.

    Examples
    ========

    >>> from sympy.discrete.convolutions import convolution_ntt
    >>> convolution_ntt([2, 3], [4, 5], prime=19*2**10 + 1)
    [8, 22, 15]
    >>> convolution_ntt([2, 5], [6, 7, 3], prime=19*2**10 + 1)
    [12, 44, 41, 15]
    >>> convolution_ntt([333, 555], [222, 666], prime=19*2**10 + 1)
    [15555, 14219, 19404]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Convolution_theorem
    .. [2] https://en.wikipedia.org/wiki/Discrete_Fourier_transform_(general%29

    """

    a, b, p = a[:], b[:], as_int(prime)
    n = m = len(a) + len(b) - 1 # convolution size

    if n > 0 and n&(n - 1): # not a power of 2
        n = 2**n.bit_length()

    # padding with zeros
    a += [0]*(n - len(a))
    b += [0]*(n - len(b))

    a, b = ntt(a, p), ntt(b, p)
    a = [x*y % p for x, y in zip(a, b)]
    a = intt(a, p)[:m]

    return a


#----------------------------------------------------------------------------#
#                                                                            #
#                         Convolution for 2**n-group                         #
#                                                                            #
#----------------------------------------------------------------------------#

def convolution_fwht(a, b):
    """
    Performs dyadic (*bitwise-XOR*) convolution using Fast Walsh Hadamard
    Transform.

    The convolution is automatically padded to the right with zeros, as the
    *radix-2 FWHT* requires the number of sample points to be a power of 2.

    Parameters
    ==========

    a, b : iterables
        The sequences for which convolution is performed.

    Examples
    ========

    >>> from sympy import symbols, S, I
    >>> from sympy.discrete.convolutions import convolution_fwht

    >>> u, v, x, y = symbols('u v x y')
    >>> convolution_fwht([u, v], [x, y])
    [u*x + v*y, u*y + v*x]

    >>> convolution_fwht([2, 3], [4, 5])
    [23, 22]
    >>> convolution_fwht([2, 5 + 4*I, 7], [6*I, 7, 3 + 4*I])
    [56 + 68*I, -10 + 30*I, 6 + 50*I, 48 + 32*I]

    >>> convolution_fwht([S(33)/7, S(55)/6, S(7)/4], [S(2)/3, 5])
    [2057/42, 1870/63, 7/6, 35/4]

    References
    ==========

    .. [1] https://www.radioeng.cz/fulltexts/2002/02_03_40_42.pdf
    .. [2] https://en.wikipedia.org/wiki/Hadamard_transform

    """

    if not a or not b:
        return []

    a, b = a[:], b[:]
    n = max(len(a), len(b))

    if n&(n - 1): # not a power of 2
        n = 2**n.bit_length()

    # padding with zeros
    a += [S.Zero]*(n - len(a))
    b += [S.Zero]*(n - len(b))

    a, b = fwht(a), fwht(b)
    a = [expand_mul(x*y) for x, y in zip(a, b)]
    a = ifwht(a)

    return a


#----------------------------------------------------------------------------#
#                                                                            #
#                            Subset Convolution                              #
#                                                                            #
#----------------------------------------------------------------------------#

def convolution_subset(a, b):
    """
    Performs Subset Convolution of given sequences.

    The indices of each argument, considered as bit strings, correspond to
    subsets of a finite set.

    The sequence is automatically padded to the right with zeros, as the
    definition of subset based on bitmasks (indices) requires the size of
    sequence to be a power of 2.

    Parameters
    ==========

    a, b : iterables
        The sequences for which convolution is performed.

    Examples
    ========

    >>> from sympy import symbols, S
    >>> from sympy.discrete.convolutions import convolution_subset
    >>> u, v, x, y, z = symbols('u v x y z')

    >>> convolution_subset([u, v], [x, y])
    [u*x, u*y + v*x]
    >>> convolution_subset([u, v, x], [y, z])
    [u*y, u*z + v*y, x*y, x*z]

    >>> convolution_subset([1, S(2)/3], [3, 4])
    [3, 6]
    >>> convolution_subset([1, 3, S(5)/7], [7])
    [7, 21, 5, 0]

    References
    ==========

    .. [1] https://people.csail.mit.edu/rrw/presentations/subset-conv.pdf

    """

    if not a or not b:
        return []

    if not iterable(a) or not iterable(b):
        raise TypeError("Expected a sequence of coefficients for convolution")

    a = [sympify(arg) for arg in a]
    b = [sympify(arg) for arg in b]
    n = max(len(a), len(b))

    if n&(n - 1): # not a power of 2
        n = 2**n.bit_length()

    # padding with zeros
    a += [S.Zero]*(n - len(a))
    b += [S.Zero]*(n - len(b))

    c = [S.Zero]*n

    for mask in range(n):
        smask = mask
        while smask > 0:
            c[mask] += expand_mul(a[smask] * b[mask^smask])
            smask = (smask - 1)&mask

        c[mask] += expand_mul(a[smask] * b[mask^smask])

    return c


#----------------------------------------------------------------------------#
#                                                                            #
#                              Covering Product                              #
#                                                                            #
#----------------------------------------------------------------------------#

def covering_product(a, b):
    """
    Returns the covering product of given sequences.

    The indices of each argument, considered as bit strings, correspond to
    subsets of a finite set.

    The covering product of given sequences is a sequence which contains
    the sum of products of the elements of the given sequences grouped by
    the *bitwise-OR* of the corresponding indices.

    The sequence is automatically padded to the right with zeros, as the
    definition of subset based on bitmasks (indices) requires the size of
    sequence to be a power of 2.

    Parameters
    ==========

    a, b : iterables
        The sequences for which covering product is to be obtained.

    Examples
    ========

    >>> from sympy import symbols, S, I, covering_product
    >>> u, v, x, y, z = symbols('u v x y z')

    >>> covering_product([u, v], [x, y])
    [u*x, u*y + v*x + v*y]
    >>> covering_product([u, v, x], [y, z])
    [u*y, u*z + v*y + v*z, x*y, x*z]

    >>> covering_product([1, S(2)/3], [3, 4 + 5*I])
    [3, 26/3 + 25*I/3]
    >>> covering_product([1, 3, S(5)/7], [7, 8])
    [7, 53, 5, 40/7]

    References
    ==========

    .. [1] https://people.csail.mit.edu/rrw/presentations/subset-conv.pdf

    """

    if not a or not b:
        return []

    a, b = a[:], b[:]
    n = max(len(a), len(b))

    if n&(n - 1): # not a power of 2
        n = 2**n.bit_length()

    # padding with zeros
    a += [S.Zero]*(n - len(a))
    b += [S.Zero]*(n - len(b))

    a, b = mobius_transform(a), mobius_transform(b)
    a = [expand_mul(x*y) for x, y in zip(a, b)]
    a = inverse_mobius_transform(a)

    return a


#----------------------------------------------------------------------------#
#                                                                            #
#                            Intersecting Product                            #
#                                                                            #
#----------------------------------------------------------------------------#

def intersecting_product(a, b):
    """
    Returns the intersecting product of given sequences.

    The indices of each argument, considered as bit strings, correspond to
    subsets of a finite set.

    The intersecting product of given sequences is the sequence which
    contains the sum of products of the elements of the given sequences
    grouped by the *bitwise-AND* of the corresponding indices.

    The sequence is automatically padded to the right with zeros, as the
    definition of subset based on bitmasks (indices) requires the size of
    sequence to be a power of 2.

    Parameters
    ==========

    a, b : iterables
        The sequences for which intersecting product is to be obtained.

    Examples
    ========

    >>> from sympy import symbols, S, I, intersecting_product
    >>> u, v, x, y, z = symbols('u v x y z')

    >>> intersecting_product([u, v], [x, y])
    [u*x + u*y + v*x, v*y]
    >>> intersecting_product([u, v, x], [y, z])
    [u*y + u*z + v*y + x*y + x*z, v*z, 0, 0]

    >>> intersecting_product([1, S(2)/3], [3, 4 + 5*I])
    [9 + 5*I, 8/3 + 10*I/3]
    >>> intersecting_product([1, 3, S(5)/7], [7, 8])
    [327/7, 24, 0, 0]

    References
    ==========

    .. [1] https://people.csail.mit.edu/rrw/presentations/subset-conv.pdf

    """

    if not a or not b:
        return []

    a, b = a[:], b[:]
    n = max(len(a), len(b))

    if n&(n - 1): # not a power of 2
        n = 2**n.bit_length()

    # padding with zeros
    a += [S.Zero]*(n - len(a))
    b += [S.Zero]*(n - len(b))

    a, b = mobius_transform(a, subset=False), mobius_transform(b, subset=False)
    a = [expand_mul(x*y) for x, y in zip(a, b)]
    a = inverse_mobius_transform(a, subset=False)

    return a


#----------------------------------------------------------------------------#
#                                                                            #
#                            Integer Convolutions                            #
#                                                                            #
#----------------------------------------------------------------------------#

def convolution_int(a, b):
    """Return the convolution of two sequences as a list.

    The iterables must consist solely of integers.

    Parameters
    ==========

    a, b : Sequence
        The sequences for which convolution is performed.

    Explanation
    ===========

    This function performs the convolution of ``a`` and ``b`` by packing
    each into a single integer, multiplying them together, and then
    unpacking the result from the product.  The intuition behind this is
    that if we evaluate some polynomial [1]:

    .. math ::
        1156x^6 + 3808x^5 + 8440x^4 + 14856x^3 + 16164x^2 + 14040x + 8100

    at say $x = 10^5$ we obtain $1156038080844014856161641404008100$.
    Note we can read of the coefficients for each term every five digits.
    If the $x$ we chose to evaluate at is large enough, the same will hold
    for the product.

    The idea now is since big integer multiplication in libraries such
    as GMP is highly optimised, this will be reasonably fast.

    Examples
    ========

    >>> from sympy.discrete.convolutions import convolution_int

    >>> convolution_int([2, 3], [4, 5])
    [8, 22, 15]
    >>> convolution_int([1, 1, -1], [1, 1])
    [1, 2, 0, -1]

    References
    ==========

    .. [1] Fateman, Richard J.
           Can you save time in multiplying polynomials by encoding them as integers?
           University of California, Berkeley, California (2004).
           https://people.eecs.berkeley.edu/~fateman/papers/polysbyGMP.pdf
    """
    # An upper bound on the largest coefficient in p(x)q(x) is given by (1 + min(dp, dq))N(p)N(q)
    # where dp = deg(p), dq = deg(q), N(f) denotes the coefficient of largest modulus in f [1]
    B = max(abs(c) for c in a)*max(abs(c) for c in b)*(1 + min(len(a) - 1, len(b) - 1))
    x, power = MPZ(1), 0
    while x <= (2*B):  # multiply by two for negative coefficients, see [1]
        x <<= 1
        power += 1

    def to_integer(poly):
        n, mul = MPZ(0), 0
        for c in reversed(poly):
            if c and not mul: mul = -1 if c < 0 else 1
            n <<= power
            n += mul*int(c)
        return mul, n

    # Perform packing and multiplication
    (a_mul, a_packed), (b_mul, b_packed) = to_integer(a), to_integer(b)
    result = a_packed * b_packed

    # Perform unpacking
    mul = a_mul * b_mul
    mask, half, borrow, poly = x - 1, x >> 1, 0, []
    while result or borrow:
        coeff = (result & mask) + borrow
        result >>= power
        borrow = coeff >= half
        poly.append(mul * int(coeff if coeff < half else coeff - x))
    return poly or [0]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\wsgi.py ===
from __future__ import annotations

import io
import typing as t
from functools import partial
from functools import update_wrapper

from .exceptions import ClientDisconnected
from .exceptions import RequestEntityTooLarge
from .sansio import utils as _sansio_utils
from .sansio.utils import host_is_trusted  # noqa: F401 # Imported as part of API

if t.TYPE_CHECKING:
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment


def responder(f: t.Callable[..., WSGIApplication]) -> WSGIApplication:
    """Marks a function as responder.  Decorate a function with it and it
    will automatically call the return value as WSGI application.

    Example::

        @responder
        def application(environ, start_response):
            return Response('Hello World!')
    """
    return update_wrapper(lambda *a: f(*a)(*a[-2:]), f)


def get_current_url(
    environ: WSGIEnvironment,
    root_only: bool = False,
    strip_querystring: bool = False,
    host_only: bool = False,
    trusted_hosts: t.Iterable[str] | None = None,
) -> str:
    """Recreate the URL for a request from the parts in a WSGI
    environment.

    The URL is an IRI, not a URI, so it may contain Unicode characters.
    Use :func:`~werkzeug.urls.iri_to_uri` to convert it to ASCII.

    :param environ: The WSGI environment to get the URL parts from.
    :param root_only: Only build the root path, don't include the
        remaining path or query string.
    :param strip_querystring: Don't include the query string.
    :param host_only: Only build the scheme and host.
    :param trusted_hosts: A list of trusted host names to validate the
        host against.
    """
    parts = {
        "scheme": environ["wsgi.url_scheme"],
        "host": get_host(environ, trusted_hosts),
    }

    if not host_only:
        parts["root_path"] = environ.get("SCRIPT_NAME", "")

        if not root_only:
            parts["path"] = environ.get("PATH_INFO", "")

            if not strip_querystring:
                parts["query_string"] = environ.get("QUERY_STRING", "").encode("latin1")

    return _sansio_utils.get_current_url(**parts)


def _get_server(
    environ: WSGIEnvironment,
) -> tuple[str, int | None] | None:
    name = environ.get("SERVER_NAME")

    if name is None:
        return None

    try:
        port: int | None = int(environ.get("SERVER_PORT", None))
    except (TypeError, ValueError):
        # unix socket
        port = None

    return name, port


def get_host(
    environ: WSGIEnvironment, trusted_hosts: t.Iterable[str] | None = None
) -> str:
    """Return the host for the given WSGI environment.

    The ``Host`` header is preferred, then ``SERVER_NAME`` if it's not
    set. The returned host will only contain the port if it is different
    than the standard port for the protocol.

    Optionally, verify that the host is trusted using
    :func:`host_is_trusted` and raise a
    :exc:`~werkzeug.exceptions.SecurityError` if it is not.

    :param environ: A WSGI environment dict.
    :param trusted_hosts: A list of trusted host names.

    :return: Host, with port if necessary.
    :raise ~werkzeug.exceptions.SecurityError: If the host is not
        trusted.
    """
    return _sansio_utils.get_host(
        environ["wsgi.url_scheme"],
        environ.get("HTTP_HOST"),
        _get_server(environ),
        trusted_hosts,
    )


def get_content_length(environ: WSGIEnvironment) -> int | None:
    """Return the ``Content-Length`` header value as an int. If the header is not given
    or the ``Transfer-Encoding`` header is ``chunked``, ``None`` is returned to indicate
    a streaming request. If the value is not an integer, or negative, 0 is returned.

    :param environ: The WSGI environ to get the content length from.

    .. versionadded:: 0.9
    """
    return _sansio_utils.get_content_length(
        http_content_length=environ.get("CONTENT_LENGTH"),
        http_transfer_encoding=environ.get("HTTP_TRANSFER_ENCODING"),
    )


def get_input_stream(
    environ: WSGIEnvironment,
    safe_fallback: bool = True,
    max_content_length: int | None = None,
) -> t.IO[bytes]:
    """Return the WSGI input stream, wrapped so that it may be read safely without going
    past the ``Content-Length`` header value or ``max_content_length``.

    If ``Content-Length`` exceeds ``max_content_length``, a
    :exc:`RequestEntityTooLarge`` ``413 Content Too Large`` error is raised.

    If the WSGI server sets ``environ["wsgi.input_terminated"]``, it indicates that the
    server handles terminating the stream, so it is safe to read directly. For example,
    a server that knows how to handle chunked requests safely would set this.

    If ``max_content_length`` is set, it can be enforced on streams if
    ``wsgi.input_terminated`` is set. Otherwise, an empty stream is returned unless the
    user explicitly disables this safe fallback.

    If the limit is reached before the underlying stream is exhausted (such as a file
    that is too large, or an infinite stream), the remaining contents of the stream
    cannot be read safely. Depending on how the server handles this, clients may show a
    "connection reset" failure instead of seeing the 413 response.

    :param environ: The WSGI environ containing the stream.
    :param safe_fallback: Return an empty stream when ``Content-Length`` is not set.
        Disabling this allows infinite streams, which can be a denial-of-service risk.
    :param max_content_length: The maximum length that content-length or streaming
        requests may not exceed.

    .. versionchanged:: 2.3.2
        ``max_content_length`` is only applied to streaming requests if the server sets
        ``wsgi.input_terminated``.

    .. versionchanged:: 2.3
        Check ``max_content_length`` and raise an error if it is exceeded.

    .. versionadded:: 0.9
    """
    stream = t.cast(t.IO[bytes], environ["wsgi.input"])
    content_length = get_content_length(environ)

    if content_length is not None and max_content_length is not None:
        if content_length > max_content_length:
            raise RequestEntityTooLarge()

    # A WSGI server can set this to indicate that it terminates the input stream. In
    # that case the stream is safe without wrapping, or can enforce a max length.
    if "wsgi.input_terminated" in environ:
        if max_content_length is not None:
            # If this is moved above, it can cause the stream to hang if a read attempt
            # is made when the client sends no data. For example, the development server
            # does not handle buffering except for chunked encoding.
            return t.cast(
                t.IO[bytes], LimitedStream(stream, max_content_length, is_max=True)
            )

        return stream

    # No limit given, return an empty stream unless the user explicitly allows the
    # potentially infinite stream. An infinite stream is dangerous if it's not expected,
    # as it can tie up a worker indefinitely.
    if content_length is None:
        return io.BytesIO() if safe_fallback else stream

    return t.cast(t.IO[bytes], LimitedStream(stream, content_length))


def get_path_info(environ: WSGIEnvironment) -> str:
    """Return ``PATH_INFO`` from  the WSGI environment.

    :param environ: WSGI environment to get the path from.

    .. versionchanged:: 3.0
        The ``charset`` and ``errors`` parameters were removed.

    .. versionadded:: 0.9
    """
    path: bytes = environ.get("PATH_INFO", "").encode("latin1")
    return path.decode(errors="replace")


class ClosingIterator:
    """The WSGI specification requires that all middlewares and gateways
    respect the `close` callback of the iterable returned by the application.
    Because it is useful to add another close action to a returned iterable
    and adding a custom iterable is a boring task this class can be used for
    that::

        return ClosingIterator(app(environ, start_response), [cleanup_session,
                                                              cleanup_locals])

    If there is just one close function it can be passed instead of the list.

    A closing iterator is not needed if the application uses response objects
    and finishes the processing if the response is started::

        try:
            return response(environ, start_response)
        finally:
            cleanup_session()
            cleanup_locals()
    """

    def __init__(
        self,
        iterable: t.Iterable[bytes],
        callbacks: None
        | (t.Callable[[], None] | t.Iterable[t.Callable[[], None]]) = None,
    ) -> None:
        iterator = iter(iterable)
        self._next = t.cast(t.Callable[[], bytes], partial(next, iterator))
        if callbacks is None:
            callbacks = []
        elif callable(callbacks):
            callbacks = [callbacks]
        else:
            callbacks = list(callbacks)
        iterable_close = getattr(iterable, "close", None)
        if iterable_close:
            callbacks.insert(0, iterable_close)
        self._callbacks = callbacks

    def __iter__(self) -> ClosingIterator:
        return self

    def __next__(self) -> bytes:
        return self._next()

    def close(self) -> None:
        for callback in self._callbacks:
            callback()


def wrap_file(
    environ: WSGIEnvironment, file: t.IO[bytes], buffer_size: int = 8192
) -> t.Iterable[bytes]:
    """Wraps a file.  This uses the WSGI server's file wrapper if available
    or otherwise the generic :class:`FileWrapper`.

    .. versionadded:: 0.5

    If the file wrapper from the WSGI server is used it's important to not
    iterate over it from inside the application but to pass it through
    unchanged.  If you want to pass out a file wrapper inside a response
    object you have to set :attr:`Response.direct_passthrough` to `True`.

    More information about file wrappers are available in :pep:`333`.

    :param file: a :class:`file`-like object with a :meth:`~file.read` method.
    :param buffer_size: number of bytes for one iteration.
    """
    return environ.get("wsgi.file_wrapper", FileWrapper)(  # type: ignore
        file, buffer_size
    )


class FileWrapper:
    """This class can be used to convert a :class:`file`-like object into
    an iterable.  It yields `buffer_size` blocks until the file is fully
    read.

    You should not use this class directly but rather use the
    :func:`wrap_file` function that uses the WSGI server's file wrapper
    support if it's available.

    .. versionadded:: 0.5

    If you're using this object together with a :class:`Response` you have
    to use the `direct_passthrough` mode.

    :param file: a :class:`file`-like object with a :meth:`~file.read` method.
    :param buffer_size: number of bytes for one iteration.
    """

    def __init__(self, file: t.IO[bytes], buffer_size: int = 8192) -> None:
        self.file = file
        self.buffer_size = buffer_size

    def close(self) -> None:
        if hasattr(self.file, "close"):
            self.file.close()

    def seekable(self) -> bool:
        if hasattr(self.file, "seekable"):
            return self.file.seekable()
        if hasattr(self.file, "seek"):
            return True
        return False

    def seek(self, *args: t.Any) -> None:
        if hasattr(self.file, "seek"):
            self.file.seek(*args)

    def tell(self) -> int | None:
        if hasattr(self.file, "tell"):
            return self.file.tell()
        return None

    def __iter__(self) -> FileWrapper:
        return self

    def __next__(self) -> bytes:
        data = self.file.read(self.buffer_size)
        if data:
            return data
        raise StopIteration()


class _RangeWrapper:
    # private for now, but should we make it public in the future ?

    """This class can be used to convert an iterable object into
    an iterable that will only yield a piece of the underlying content.
    It yields blocks until the underlying stream range is fully read.
    The yielded blocks will have a size that can't exceed the original
    iterator defined block size, but that can be smaller.

    If you're using this object together with a :class:`Response` you have
    to use the `direct_passthrough` mode.

    :param iterable: an iterable object with a :meth:`__next__` method.
    :param start_byte: byte from which read will start.
    :param byte_range: how many bytes to read.
    """

    def __init__(
        self,
        iterable: t.Iterable[bytes] | t.IO[bytes],
        start_byte: int = 0,
        byte_range: int | None = None,
    ):
        self.iterable = iter(iterable)
        self.byte_range = byte_range
        self.start_byte = start_byte
        self.end_byte = None

        if byte_range is not None:
            self.end_byte = start_byte + byte_range

        self.read_length = 0
        self.seekable = hasattr(iterable, "seekable") and iterable.seekable()
        self.end_reached = False

    def __iter__(self) -> _RangeWrapper:
        return self

    def _next_chunk(self) -> bytes:
        try:
            chunk = next(self.iterable)
            self.read_length += len(chunk)
            return chunk
        except StopIteration:
            self.end_reached = True
            raise

    def _first_iteration(self) -> tuple[bytes | None, int]:
        chunk = None
        if self.seekable:
            self.iterable.seek(self.start_byte)  # type: ignore
            self.read_length = self.iterable.tell()  # type: ignore
            contextual_read_length = self.read_length
        else:
            while self.read_length <= self.start_byte:
                chunk = self._next_chunk()
            if chunk is not None:
                chunk = chunk[self.start_byte - self.read_length :]
            contextual_read_length = self.start_byte
        return chunk, contextual_read_length

    def _next(self) -> bytes:
        if self.end_reached:
            raise StopIteration()
        chunk = None
        contextual_read_length = self.read_length
        if self.read_length == 0:
            chunk, contextual_read_length = self._first_iteration()
        if chunk is None:
            chunk = self._next_chunk()
        if self.end_byte is not None and self.read_length >= self.end_byte:
            self.end_reached = True
            return chunk[: self.end_byte - contextual_read_length]
        return chunk

    def __next__(self) -> bytes:
        chunk = self._next()
        if chunk:
            return chunk
        self.end_reached = True
        raise StopIteration()

    def close(self) -> None:
        if hasattr(self.iterable, "close"):
            self.iterable.close()


class LimitedStream(io.RawIOBase):
    """Wrap a stream so that it doesn't read more than a given limit. This is used to
    limit ``wsgi.input`` to the ``Content-Length`` header value or
    :attr:`.Request.max_content_length`.

    When attempting to read after the limit has been reached, :meth:`on_exhausted` is
    called. When the limit is a maximum, this raises :exc:`.RequestEntityTooLarge`.

    If reading from the stream returns zero bytes or raises an error,
    :meth:`on_disconnect` is called, which raises :exc:`.ClientDisconnected`. When the
    limit is a maximum and zero bytes were read, no error is raised, since it may be the
    end of the stream.

    If the limit is reached before the underlying stream is exhausted (such as a file
    that is too large, or an infinite stream), the remaining contents of the stream
    cannot be read safely. Depending on how the server handles this, clients may show a
    "connection reset" failure instead of seeing the 413 response.

    :param stream: The stream to read from. Must be a readable binary IO object.
    :param limit: The limit in bytes to not read past. Should be either the
        ``Content-Length`` header value or ``request.max_content_length``.
    :param is_max: Whether the given ``limit`` is ``request.max_content_length`` instead
        of the ``Content-Length`` header value. This changes how exhausted and
        disconnect events are handled.

    .. versionchanged:: 2.3
        Handle ``max_content_length`` differently than ``Content-Length``.

    .. versionchanged:: 2.3
        Implements ``io.RawIOBase`` rather than ``io.IOBase``.
    """

    def __init__(self, stream: t.IO[bytes], limit: int, is_max: bool = False) -> None:
        self._stream = stream
        self._pos = 0
        self.limit = limit
        self._limit_is_max = is_max

    @property
    def is_exhausted(self) -> bool:
        """Whether the current stream position has reached the limit."""
        return self._pos >= self.limit

    def on_exhausted(self) -> None:
        """Called when attempting to read after the limit has been reached.

        The default behavior is to do nothing, unless the limit is a maximum, in which
        case it raises :exc:`.RequestEntityTooLarge`.

        .. versionchanged:: 2.3
            Raises ``RequestEntityTooLarge`` if the limit is a maximum.

        .. versionchanged:: 2.3
            Any return value is ignored.
        """
        if self._limit_is_max:
            raise RequestEntityTooLarge()

    def on_disconnect(self, error: Exception | None = None) -> None:
        """Called when an attempted read receives zero bytes before the limit was
        reached. This indicates that the client disconnected before sending the full
        request body.

        The default behavior is to raise :exc:`.ClientDisconnected`, unless the limit is
        a maximum and no error was raised.

        .. versionchanged:: 2.3
            Added the ``error`` parameter. Do nothing if the limit is a maximum and no
            error was raised.

        .. versionchanged:: 2.3
            Any return value is ignored.
        """
        if not self._limit_is_max or error is not None:
            raise ClientDisconnected()

        # If the limit is a maximum, then we may have read zero bytes because the
        # streaming body is complete. There's no way to distinguish that from the
        # client disconnecting early.

    def exhaust(self) -> bytes:
        """Exhaust the stream by reading until the limit is reached or the client
        disconnects, returning the remaining data.

        .. versionchanged:: 2.3
            Return the remaining data.

        .. versionchanged:: 2.2.3
            Handle case where wrapped stream returns fewer bytes than requested.
        """
        if not self.is_exhausted:
            return self.readall()

        return b""

    def readinto(self, b: bytearray) -> int | None:  # type: ignore[override]
        size = len(b)
        remaining = self.limit - self._pos

        if remaining <= 0:
            self.on_exhausted()
            return 0

        if hasattr(self._stream, "readinto"):
            # Use stream.readinto if it's available.
            if size <= remaining:
                # The size fits in the remaining limit, use the buffer directly.
                try:
                    out_size: int | None = self._stream.readinto(b)
                except (OSError, ValueError) as e:
                    self.on_disconnect(error=e)
                    return 0
            else:
                # Use a temp buffer with the remaining limit as the size.
                temp_b = bytearray(remaining)

                try:
                    out_size = self._stream.readinto(temp_b)
                except (OSError, ValueError) as e:
                    self.on_disconnect(error=e)
                    return 0

                if out_size:
                    b[:out_size] = temp_b
        else:
            # WSGI requires that stream.read is available.
            try:
                data = self._stream.read(min(size, remaining))
            except (OSError, ValueError) as e:
                self.on_disconnect(error=e)
                return 0

            out_size = len(data)
            b[:out_size] = data

        if not out_size:
            # Read zero bytes from the stream.
            self.on_disconnect()
            return 0

        self._pos += out_size
        return out_size

    def readall(self) -> bytes:
        if self.is_exhausted:
            self.on_exhausted()
            return b""

        out = bytearray()

        # The parent implementation uses "while True", which results in an extra read.
        while not self.is_exhausted:
            data = self.read(1024 * 64)

            # Stream may return empty before a max limit is reached.
            if not data:
                break

            out.extend(data)

        return bytes(out)

    def tell(self) -> int:
        """Return the current stream position.

        .. versionadded:: 0.9
        """
        return self._pos

    def readable(self) -> bool:
        return True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\builder\_html5lib.py ===
# Use of this source code is governed by the MIT license.
__license__ = "MIT"

__all__ = [
    "HTML5TreeBuilder",
]

from typing import (
    Any,
    cast,
    Dict,
    Iterable,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Tuple,
    Union,
)
from typing_extensions import TypeAlias
from bs4._typing import (
    _AttributeValue,
    _AttributeValues,
    _Encoding,
    _Encodings,
    _NamespaceURL,
    _RawMarkup,
)

import warnings
from bs4.builder import (
    DetectsXMLParsedAsHTML,
    PERMISSIVE,
    HTML,
    HTML_5,
    HTMLTreeBuilder,
)
from bs4.element import (
    NamespacedAttribute,
    PageElement,
    nonwhitespace_re,
)
import html5lib
from html5lib.constants import (
    namespaces,
)
from bs4.element import (
    Comment,
    Doctype,
    NavigableString,
    Tag,
)

if TYPE_CHECKING:
    from bs4 import BeautifulSoup

from html5lib.treebuilders import base as treebuilder_base


class HTML5TreeBuilder(HTMLTreeBuilder):
    """Use `html5lib <https://github.com/html5lib/html5lib-python>`_ to
    build a tree.

    Note that `HTML5TreeBuilder` does not support some common HTML
    `TreeBuilder` features. Some of these features could theoretically
    be implemented, but at the very least it's quite difficult,
    because html5lib moves the parse tree around as it's being built.

    Specifically:

    * This `TreeBuilder` doesn't use different subclasses of
      `NavigableString` (e.g. `Script`) based on the name of the tag
      in which the string was found.
    * You can't use a `SoupStrainer` to parse only part of a document.
    """

    NAME: str = "html5lib"

    features: Sequence[str] = [NAME, PERMISSIVE, HTML_5, HTML]

    #: html5lib can tell us which line number and position in the
    #: original file is the source of an element.
    TRACKS_LINE_NUMBERS: bool = True

    underlying_builder: "TreeBuilderForHtml5lib"  #: :meta private:
    user_specified_encoding: Optional[_Encoding]

    def prepare_markup(
        self,
        markup: _RawMarkup,
        user_specified_encoding: Optional[_Encoding] = None,
        document_declared_encoding: Optional[_Encoding] = None,
        exclude_encodings: Optional[_Encodings] = None,
    ) -> Iterable[Tuple[_RawMarkup, Optional[_Encoding], Optional[_Encoding], bool]]:
        # Store the user-specified encoding for use later on.
        self.user_specified_encoding = user_specified_encoding

        # document_declared_encoding and exclude_encodings aren't used
        # ATM because the html5lib TreeBuilder doesn't use
        # UnicodeDammit.
        for variable, name in (
            (document_declared_encoding, "document_declared_encoding"),
            (exclude_encodings, "exclude_encodings"),
        ):
            if variable:
                warnings.warn(
                    f"You provided a value for {name}, but the html5lib tree builder doesn't support {name}.",
                    stacklevel=3,
                )

        # html5lib only parses HTML, so if it's given XML that's worth
        # noting.
        DetectsXMLParsedAsHTML.warn_if_markup_looks_like_xml(markup, stacklevel=3)

        yield (markup, None, None, False)

    # These methods are defined by Beautiful Soup.
    def feed(self, markup: _RawMarkup) -> None:
        """Run some incoming markup through some parsing process,
        populating the `BeautifulSoup` object in `HTML5TreeBuilder.soup`.
        """
        if self.soup is not None and self.soup.parse_only is not None:
            warnings.warn(
                "You provided a value for parse_only, but the html5lib tree builder doesn't support parse_only. The entire document will be parsed.",
                stacklevel=4,
            )

        # self.underlying_builder is probably None now, but it'll be set
        # when html5lib calls self.create_treebuilder().
        parser = html5lib.HTMLParser(tree=self.create_treebuilder)
        assert self.underlying_builder is not None
        self.underlying_builder.parser = parser
        extra_kwargs = dict()
        if not isinstance(markup, str):
            # kwargs, specifically override_encoding, will eventually
            # be passed in to html5lib's
            # HTMLBinaryInputStream.__init__.
            extra_kwargs["override_encoding"] = self.user_specified_encoding

        doc = parser.parse(markup, **extra_kwargs)

        # Set the character encoding detected by the tokenizer.
        if isinstance(markup, str):
            # We need to special-case this because html5lib sets
            # charEncoding to UTF-8 if it gets Unicode input.
            doc.original_encoding = None
        else:
            original_encoding = parser.tokenizer.stream.charEncoding[0]
            # The encoding is an html5lib Encoding object. We want to
            # use a string for compatibility with other tree builders.
            original_encoding = original_encoding.name
            doc.original_encoding = original_encoding
        self.underlying_builder.parser = None

    def create_treebuilder(
        self, namespaceHTMLElements: bool
    ) -> "TreeBuilderForHtml5lib":
        """Called by html5lib to instantiate the kind of class it
        calls a 'TreeBuilder'.

        :param namespaceHTMLElements: Whether or not to namespace HTML elements.

        :meta private:
        """
        self.underlying_builder = TreeBuilderForHtml5lib(
            namespaceHTMLElements, self.soup, store_line_numbers=self.store_line_numbers
        )
        return self.underlying_builder

    def test_fragment_to_document(self, fragment: str) -> str:
        """See `TreeBuilder`."""
        return "<html><head></head><body>%s</body></html>" % fragment


class TreeBuilderForHtml5lib(treebuilder_base.TreeBuilder):
    soup: "BeautifulSoup"  #: :meta private:
    parser: Optional[html5lib.HTMLParser]  #: :meta private:

    def __init__(
        self,
        namespaceHTMLElements: bool,
        soup: Optional["BeautifulSoup"] = None,
        store_line_numbers: bool = True,
        **kwargs: Any,
    ):
        if soup:
            self.soup = soup
        else:
            warnings.warn(
                "The optionality of the 'soup' argument to the TreeBuilderForHtml5lib constructor is deprecated as of Beautiful Soup 4.13.0: 'soup' is now required. If you can't pass in a BeautifulSoup object here, or you get this warning and it seems mysterious to you, please contact the Beautiful Soup developer team for possible un-deprecation.",
                DeprecationWarning,
                stacklevel=2,
            )
            from bs4 import BeautifulSoup

            # TODO: Why is the parser 'html.parser' here? Using
            # html5lib doesn't cause an infinite loop and is more
            # accurate. Best to get rid of this entire section, I think.
            self.soup = BeautifulSoup(
                "", "html.parser", store_line_numbers=store_line_numbers, **kwargs
            )
        # TODO: What are **kwargs exactly? Should they be passed in
        # here in addition to/instead of being passed to the BeautifulSoup
        # constructor?
        super(TreeBuilderForHtml5lib, self).__init__(namespaceHTMLElements)

        # This will be set later to a real html5lib HTMLParser object,
        # which we can use to track the current line number.
        self.parser = None
        self.store_line_numbers = store_line_numbers

    def documentClass(self) -> "Element":
        self.soup.reset()
        return Element(self.soup, self.soup, None)

    def insertDoctype(self, token: Dict[str, Any]) -> None:
        name: str = cast(str, token["name"])
        publicId: Optional[str] = cast(Optional[str], token["publicId"])
        systemId: Optional[str] = cast(Optional[str], token["systemId"])

        doctype = Doctype.for_name_and_ids(name, publicId, systemId)
        self.soup.object_was_parsed(doctype)

    def elementClass(self, name: str, namespace: str) -> "Element":
        sourceline: Optional[int] = None
        sourcepos: Optional[int] = None
        if self.parser is not None and self.store_line_numbers:
            # This represents the point immediately after the end of the
            # tag. We don't know when the tag started, but we do know
            # where it ended -- the character just before this one.
            sourceline, sourcepos = self.parser.tokenizer.stream.position()
            assert sourcepos is not None
            sourcepos = sourcepos - 1
        tag = self.soup.new_tag(
            name, namespace, sourceline=sourceline, sourcepos=sourcepos
        )

        return Element(tag, self.soup, namespace)

    def commentClass(self, data: str) -> "TextNode":
        return TextNode(Comment(data), self.soup)

    def fragmentClass(self) -> "Element":
        """This is only used by html5lib HTMLParser.parseFragment(),
        which is never used by Beautiful Soup, only by the html5lib
        unit tests. Since we don't currently hook into those tests,
        the implementation is left blank.
        """
        raise NotImplementedError()

    def getFragment(self) -> "Element":
        """This is only used by the html5lib unit tests. Since we
        don't currently hook into those tests, the implementation is
        left blank.
        """
        raise NotImplementedError()

    def appendChild(self, node: "Element") -> None:
        # TODO: This code is not covered by the BS4 tests, and
        # apparently not triggered by the html5lib test suite either.
        # But it doesn't seem test-specific and there are calls to it
        # (or a method with the same name) all over html5lib, so I'm
        # leaving the implementation in place rather than replacing it
        # with NotImplementedError()
        self.soup.append(node.element)

    def getDocument(self) -> "BeautifulSoup":
        return self.soup

    def testSerializer(self, element: "Element") -> str:
        """This is only used by the html5lib unit tests. Since we
        don't currently hook into those tests, the implementation is
        left blank.
        """
        raise NotImplementedError()


class AttrList(object):
    """Represents a Tag's attributes in a way compatible with html5lib."""

    element: Tag
    attrs: _AttributeValues

    def __init__(self, element: Tag):
        self.element = element
        self.attrs = dict(self.element.attrs)

    def __iter__(self) -> Iterable[Tuple[str, _AttributeValue]]:
        return list(self.attrs.items()).__iter__()

    def __setitem__(self, name: str, value: _AttributeValue) -> None:
        # If this attribute is a multi-valued attribute for this element,
        # turn its value into a list.
        list_attr = self.element.cdata_list_attributes or {}
        if name in list_attr.get("*", []) or (
            self.element.name in list_attr
            and name in list_attr.get(self.element.name, [])
        ):
            # A node that is being cloned may have already undergone
            # this procedure. Check for this and skip it.
            if not isinstance(value, list):
                assert isinstance(value, str)
                value = self.element.attribute_value_list_class(
                    nonwhitespace_re.findall(value)
                )
        self.element[name] = value

    def items(self) -> Iterable[Tuple[str, _AttributeValue]]:
        return list(self.attrs.items())

    def keys(self) -> Iterable[str]:
        return list(self.attrs.keys())

    def __len__(self) -> int:
        return len(self.attrs)

    def __getitem__(self, name: str) -> _AttributeValue:
        return self.attrs[name]

    def __contains__(self, name: str) -> bool:
        return name in list(self.attrs.keys())


class BeautifulSoupNode(treebuilder_base.Node):
    element: PageElement
    soup: "BeautifulSoup"
    namespace: Optional[_NamespaceURL]

    @property
    def nodeType(self) -> int:
        """Return the html5lib constant corresponding to the type of
        the underlying DOM object.

        NOTE: This property is only accessed by the html5lib test
        suite, not by Beautiful Soup proper.
        """
        raise NotImplementedError()

    # TODO-TYPING: typeshed stubs are incorrect about this;
    # cloneNode returns a new Node, not None.
    def cloneNode(self) -> treebuilder_base.Node:
        raise NotImplementedError()


class Element(BeautifulSoupNode):
    element: Tag
    namespace: Optional[_NamespaceURL]

    def __init__(
        self, element: Tag, soup: "BeautifulSoup", namespace: Optional[_NamespaceURL]
    ):
        treebuilder_base.Node.__init__(self, element.name)
        self.element = element
        self.soup = soup
        self.namespace = namespace

    def appendChild(self, node: "BeautifulSoupNode") -> None:
        string_child: Optional[NavigableString] = None
        child: PageElement
        if type(node.element) is NavigableString:
            string_child = child = node.element
        else:
            child = node.element
        node.parent = self

        if (
            child is not None
            and child.parent is not None
            and not isinstance(child, str)
        ):
            node.element.extract()

        if (
            string_child is not None
            and self.element.contents
            and type(self.element.contents[-1]) is NavigableString
        ):
            # We are appending a string onto another string.
            # TODO This has O(n^2) performance, for input like
            # "a</a>a</a>a</a>..."
            old_element = self.element.contents[-1]
            new_element = self.soup.new_string(old_element + string_child)
            old_element.replace_with(new_element)
            self.soup._most_recent_element = new_element
        else:
            if isinstance(node, str):
                # Create a brand new NavigableString from this string.
                child = self.soup.new_string(node)

            # Tell Beautiful Soup to act as if it parsed this element
            # immediately after the parent's last descendant. (Or
            # immediately after the parent, if it has no children.)
            if self.element.contents:
                most_recent_element = self.element._last_descendant(False)
            elif self.element.next_element is not None:
                # Something from further ahead in the parse tree is
                # being inserted into this earlier element. This is
                # very annoying because it means an expensive search
                # for the last element in the tree.
                most_recent_element = self.soup._last_descendant()
            else:
                most_recent_element = self.element

            self.soup.object_was_parsed(
                child, parent=self.element, most_recent_element=most_recent_element
            )

    def getAttributes(self) -> AttrList:
        if isinstance(self.element, Comment):
            return {}
        return AttrList(self.element)

    # An HTML5lib attribute name may either be a single string,
    # or a tuple (namespace, name).
    _Html5libAttributeName: TypeAlias = Union[str, Tuple[str, str]]
    # Now we can define the type this method accepts as a dictionary
    # mapping those attribute names to single string values.
    _Html5libAttributes: TypeAlias = Dict[_Html5libAttributeName, str]

    def setAttributes(self, attributes: Optional[_Html5libAttributes]) -> None:
        if attributes is not None and len(attributes) > 0:
            # Replace any namespaced attributes with
            # NamespacedAttribute objects.
            for name, value in list(attributes.items()):
                if isinstance(name, tuple):
                    new_name = NamespacedAttribute(*name)
                    del attributes[name]
                    attributes[new_name] = value

            # We can now cast attributes to the type of Dict
            # used by Beautiful Soup.
            normalized_attributes = cast(_AttributeValues, attributes)

            # Values for tags like 'class' came in as single strings;
            # replace them with lists of strings as appropriate.
            self.soup.builder._replace_cdata_list_attribute_values(
                self.name, normalized_attributes
            )

            # Then set the attributes on the Tag associated with this
            # BeautifulSoupNode.
            for name, value_or_values in list(normalized_attributes.items()):
                self.element[name] = value_or_values

            # The attributes may contain variables that need substitution.
            # Call set_up_substitutions manually.
            #
            # The Tag constructor called this method when the Tag was created,
            # but we just set/changed the attributes, so call it again.
            self.soup.builder.set_up_substitutions(self.element)

    attributes = property(getAttributes, setAttributes)

    def insertText(
        self, data: str, insertBefore: Optional["BeautifulSoupNode"] = None
    ) -> None:
        text = TextNode(self.soup.new_string(data), self.soup)
        if insertBefore:
            self.insertBefore(text, insertBefore)
        else:
            self.appendChild(text)

    def insertBefore(
        self, node: "BeautifulSoupNode", refNode: "BeautifulSoupNode"
    ) -> None:
        index = self.element.index(refNode.element)
        if (
            type(node.element) is NavigableString
            and self.element.contents
            and type(self.element.contents[index - 1]) is NavigableString
        ):
            # (See comments in appendChild)
            old_node = self.element.contents[index - 1]
            assert type(old_node) is NavigableString
            new_str = self.soup.new_string(old_node + node.element)
            old_node.replace_with(new_str)
        else:
            self.element.insert(index, node.element)
            node.parent = self

    def removeChild(self, node: "Element") -> None:
        node.element.extract()

    def reparentChildren(self, new_parent: "Element") -> None:
        """Move all of this tag's children into another tag."""
        # print("MOVE", self.element.contents)
        # print("FROM", self.element)
        # print("TO", new_parent.element)

        element = self.element
        new_parent_element = new_parent.element
        # Determine what this tag's next_element will be once all the children
        # are removed.
        final_next_element = element.next_sibling

        new_parents_last_descendant = new_parent_element._last_descendant(False, False)
        if len(new_parent_element.contents) > 0:
            # The new parent already contains children. We will be
            # appending this tag's children to the end.

            # We can make this assertion since we know new_parent has
            # children.
            assert new_parents_last_descendant is not None
            new_parents_last_child = new_parent_element.contents[-1]
            new_parents_last_descendant_next_element = (
                new_parents_last_descendant.next_element
            )
        else:
            # The new parent contains no children.
            new_parents_last_child = None
            new_parents_last_descendant_next_element = new_parent_element.next_element

        to_append = element.contents
        if len(to_append) > 0:
            # Set the first child's previous_element and previous_sibling
            # to elements within the new parent
            first_child = to_append[0]
            if new_parents_last_descendant is not None:
                first_child.previous_element = new_parents_last_descendant
            else:
                first_child.previous_element = new_parent_element
            first_child.previous_sibling = new_parents_last_child
            if new_parents_last_descendant is not None:
                new_parents_last_descendant.next_element = first_child
            else:
                new_parent_element.next_element = first_child
            if new_parents_last_child is not None:
                new_parents_last_child.next_sibling = first_child

            # Find the very last element being moved. It is now the
            # parent's last descendant. It has no .next_sibling and
            # its .next_element is whatever the previous last
            # descendant had.
            last_childs_last_descendant = to_append[-1]._last_descendant(
                is_initialized=False, accept_self=True
            )

            # Since we passed accept_self=True into _last_descendant,
            # there's no possibility that the result is None.
            assert last_childs_last_descendant is not None
            last_childs_last_descendant.next_element = (
                new_parents_last_descendant_next_element
            )
            if new_parents_last_descendant_next_element is not None:
                # TODO-COVERAGE: This code has no test coverage and
                # I'm not sure how to get html5lib to go through this
                # path, but it's just the other side of the previous
                # line.
                new_parents_last_descendant_next_element.previous_element = (
                    last_childs_last_descendant
                )
            last_childs_last_descendant.next_sibling = None

        for child in to_append:
            child.parent = new_parent_element
            new_parent_element.contents.append(child)

        # Now that this element has no children, change its .next_element.
        element.contents = []
        element.next_element = final_next_element

        # print("DONE WITH MOVE")
        # print("FROM", self.element)
        # print("TO", new_parent_element)

    # TODO-TYPING: typeshed stubs are incorrect about this;
    # hasContent returns a boolean, not None.
    def hasContent(self) -> bool:
        return len(self.element.contents) > 0

    # TODO-TYPING: typeshed stubs are incorrect about this;
    # cloneNode returns a new Node, not None.
    def cloneNode(self) -> treebuilder_base.Node:
        tag = self.soup.new_tag(self.element.name, self.namespace)
        node = Element(tag, self.soup, self.namespace)
        for key, value in self.attributes:
            node.attributes[key] = value
        return node

    def getNameTuple(self) -> Tuple[Optional[_NamespaceURL], str]:
        if self.namespace is None:
            return namespaces["html"], self.name
        else:
            return self.namespace, self.name

    nameTuple = property(getNameTuple)


class TextNode(BeautifulSoupNode):
    element: NavigableString

    def __init__(self, element: NavigableString, soup: "BeautifulSoup"):
        treebuilder_base.Node.__init__(self, None)
        self.element = element
        self.soup = soup