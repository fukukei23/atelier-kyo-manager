
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\_user_array_impl.py ===
"""
Container class for backward compatibility with NumArray.

The user_array.container class exists for backward compatibility with NumArray
and is not meant to be used in new code. If you need to create an array
container class, we recommend either creating a class that wraps an ndarray
or subclasses ndarray.

"""
from numpy._core import (
    absolute,
    add,
    arange,
    array,
    asarray,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    divide,
    equal,
    greater,
    greater_equal,
    invert,
    left_shift,
    less,
    less_equal,
    multiply,
    not_equal,
    power,
    remainder,
    reshape,
    right_shift,
    shape,
    sin,
    sqrt,
    subtract,
    transpose,
)
from numpy._core.overrides import set_module


@set_module("numpy.lib.user_array")
class container:
    """
    container(data, dtype=None, copy=True)

    Standard container-class for easy multiple-inheritance.

    Methods
    -------
    copy
    byteswap
    astype

    """
    def __init__(self, data, dtype=None, copy=True):
        self.array = array(data, dtype, copy=copy)

    def __repr__(self):
        if self.ndim > 0:
            return self.__class__.__name__ + repr(self.array)[len("array"):]
        else:
            return self.__class__.__name__ + "(" + repr(self.array) + ")"

    def __array__(self, t=None):
        if t:
            return self.array.astype(t)
        return self.array

    # Array as sequence
    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        return self._rc(self.array[index])

    def __setitem__(self, index, value):
        self.array[index] = asarray(value, self.dtype)

    def __abs__(self):
        return self._rc(absolute(self.array))

    def __neg__(self):
        return self._rc(-self.array)

    def __add__(self, other):
        return self._rc(self.array + asarray(other))

    __radd__ = __add__

    def __iadd__(self, other):
        add(self.array, other, self.array)
        return self

    def __sub__(self, other):
        return self._rc(self.array - asarray(other))

    def __rsub__(self, other):
        return self._rc(asarray(other) - self.array)

    def __isub__(self, other):
        subtract(self.array, other, self.array)
        return self

    def __mul__(self, other):
        return self._rc(multiply(self.array, asarray(other)))

    __rmul__ = __mul__

    def __imul__(self, other):
        multiply(self.array, other, self.array)
        return self

    def __mod__(self, other):
        return self._rc(remainder(self.array, other))

    def __rmod__(self, other):
        return self._rc(remainder(other, self.array))

    def __imod__(self, other):
        remainder(self.array, other, self.array)
        return self

    def __divmod__(self, other):
        return (self._rc(divide(self.array, other)),
                self._rc(remainder(self.array, other)))

    def __rdivmod__(self, other):
        return (self._rc(divide(other, self.array)),
                self._rc(remainder(other, self.array)))

    def __pow__(self, other):
        return self._rc(power(self.array, asarray(other)))

    def __rpow__(self, other):
        return self._rc(power(asarray(other), self.array))

    def __ipow__(self, other):
        power(self.array, other, self.array)
        return self

    def __lshift__(self, other):
        return self._rc(left_shift(self.array, other))

    def __rshift__(self, other):
        return self._rc(right_shift(self.array, other))

    def __rlshift__(self, other):
        return self._rc(left_shift(other, self.array))

    def __rrshift__(self, other):
        return self._rc(right_shift(other, self.array))

    def __ilshift__(self, other):
        left_shift(self.array, other, self.array)
        return self

    def __irshift__(self, other):
        right_shift(self.array, other, self.array)
        return self

    def __and__(self, other):
        return self._rc(bitwise_and(self.array, other))

    def __rand__(self, other):
        return self._rc(bitwise_and(other, self.array))

    def __iand__(self, other):
        bitwise_and(self.array, other, self.array)
        return self

    def __xor__(self, other):
        return self._rc(bitwise_xor(self.array, other))

    def __rxor__(self, other):
        return self._rc(bitwise_xor(other, self.array))

    def __ixor__(self, other):
        bitwise_xor(self.array, other, self.array)
        return self

    def __or__(self, other):
        return self._rc(bitwise_or(self.array, other))

    def __ror__(self, other):
        return self._rc(bitwise_or(other, self.array))

    def __ior__(self, other):
        bitwise_or(self.array, other, self.array)
        return self

    def __pos__(self):
        return self._rc(self.array)

    def __invert__(self):
        return self._rc(invert(self.array))

    def _scalarfunc(self, func):
        if self.ndim == 0:
            return func(self[0])
        else:
            raise TypeError(
                "only rank-0 arrays can be converted to Python scalars.")

    def __complex__(self):
        return self._scalarfunc(complex)

    def __float__(self):
        return self._scalarfunc(float)

    def __int__(self):
        return self._scalarfunc(int)

    def __hex__(self):
        return self._scalarfunc(hex)

    def __oct__(self):
        return self._scalarfunc(oct)

    def __lt__(self, other):
        return self._rc(less(self.array, other))

    def __le__(self, other):
        return self._rc(less_equal(self.array, other))

    def __eq__(self, other):
        return self._rc(equal(self.array, other))

    def __ne__(self, other):
        return self._rc(not_equal(self.array, other))

    def __gt__(self, other):
        return self._rc(greater(self.array, other))

    def __ge__(self, other):
        return self._rc(greater_equal(self.array, other))

    def copy(self):
        ""
        return self._rc(self.array.copy())

    def tobytes(self):
        ""
        return self.array.tobytes()

    def byteswap(self):
        ""
        return self._rc(self.array.byteswap())

    def astype(self, typecode):
        ""
        return self._rc(self.array.astype(typecode))

    def _rc(self, a):
        if len(shape(a)) == 0:
            return a
        else:
            return self.__class__(a)

    def __array_wrap__(self, *args):
        return self.__class__(args[0])

    def __setattr__(self, attr, value):
        if attr == 'array':
            object.__setattr__(self, attr, value)
            return
        try:
            self.array.__setattr__(attr, value)
        except AttributeError:
            object.__setattr__(self, attr, value)

    # Only called after other approaches fail.
    def __getattr__(self, attr):
        if (attr == 'array'):
            return object.__getattribute__(self, attr)
        return self.array.__getattribute__(attr)


#############################################################
# Test of class container
#############################################################
if __name__ == '__main__':
    temp = reshape(arange(10000), (100, 100))

    ua = container(temp)
    # new object created begin test
    print(dir(ua))
    print(shape(ua), ua.shape)  # I have changed Numeric.py

    ua_small = ua[:3, :5]
    print(ua_small)
    # this did not change ua[0,0], which is not normal behavior
    ua_small[0, 0] = 10
    print(ua_small[0, 0], ua[0, 0])
    print(sin(ua_small) / 3. * 6. + sqrt(ua_small ** 2))
    print(less(ua_small, 103), type(less(ua_small, 103)))
    print(type(ua_small * reshape(arange(15), shape(ua_small))))
    print(reshape(ua_small, (5, 3)))
    print(transpose(ua_small))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\ma\core.py ===
"""
numpy.ma : a package to handle missing or invalid values.

This package was initially written for numarray by Paul F. Dubois
at Lawrence Livermore National Laboratory.
In 2006, the package was completely rewritten by Pierre Gerard-Marchant
(University of Georgia) to make the MaskedArray class a subclass of ndarray,
and to improve support of structured arrays.


Copyright 1999, 2000, 2001 Regents of the University of California.
Released for unlimited redistribution.

* Adapted for numpy_core 2005 by Travis Oliphant and (mainly) Paul Dubois.
* Subclassing of the base `ndarray` 2006 by Pierre Gerard-Marchant
  (pgmdevlist_AT_gmail_DOT_com)
* Improvements suggested by Reggie Dugard (reggie_AT_merfinllc_DOT_com)

.. moduleauthor:: Pierre Gerard-Marchant

"""
import builtins
import functools
import inspect
import operator
import re
import textwrap
import warnings

import numpy as np
import numpy._core.numerictypes as ntypes
import numpy._core.umath as umath
from numpy import (
    _NoValue,
    amax,
    amin,
    angle,
    bool_,
    expand_dims,
    finfo,  # noqa: F401
    iinfo,  # noqa: F401
    iscomplexobj,
    ndarray,
)
from numpy import array as narray  # noqa: F401
from numpy._core import multiarray as mu
from numpy._core.numeric import normalize_axis_tuple
from numpy._utils import set_module
from numpy._utils._inspect import formatargspec, getargspec

__all__ = [
    'MAError', 'MaskError', 'MaskType', 'MaskedArray', 'abs', 'absolute',
    'add', 'all', 'allclose', 'allequal', 'alltrue', 'amax', 'amin',
    'angle', 'anom', 'anomalies', 'any', 'append', 'arange', 'arccos',
    'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2', 'arctanh',
    'argmax', 'argmin', 'argsort', 'around', 'array', 'asanyarray',
    'asarray', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'bool_', 'ceil',
    'choose', 'clip', 'common_fill_value', 'compress', 'compressed',
    'concatenate', 'conjugate', 'convolve', 'copy', 'correlate', 'cos', 'cosh',
    'count', 'cumprod', 'cumsum', 'default_fill_value', 'diag', 'diagonal',
    'diff', 'divide', 'empty', 'empty_like', 'equal', 'exp',
    'expand_dims', 'fabs', 'filled', 'fix_invalid', 'flatten_mask',
    'flatten_structured_array', 'floor', 'floor_divide', 'fmod',
    'frombuffer', 'fromflex', 'fromfunction', 'getdata', 'getmask',
    'getmaskarray', 'greater', 'greater_equal', 'harden_mask', 'hypot',
    'identity', 'ids', 'indices', 'inner', 'innerproduct', 'isMA',
    'isMaskedArray', 'is_mask', 'is_masked', 'isarray', 'left_shift',
    'less', 'less_equal', 'log', 'log10', 'log2',
    'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'make_mask',
    'make_mask_descr', 'make_mask_none', 'mask_or', 'masked',
    'masked_array', 'masked_equal', 'masked_greater',
    'masked_greater_equal', 'masked_inside', 'masked_invalid',
    'masked_less', 'masked_less_equal', 'masked_not_equal',
    'masked_object', 'masked_outside', 'masked_print_option',
    'masked_singleton', 'masked_values', 'masked_where', 'max', 'maximum',
    'maximum_fill_value', 'mean', 'min', 'minimum', 'minimum_fill_value',
    'mod', 'multiply', 'mvoid', 'ndim', 'negative', 'nomask', 'nonzero',
    'not_equal', 'ones', 'ones_like', 'outer', 'outerproduct', 'power', 'prod',
    'product', 'ptp', 'put', 'putmask', 'ravel', 'remainder',
    'repeat', 'reshape', 'resize', 'right_shift', 'round', 'round_',
    'set_fill_value', 'shape', 'sin', 'sinh', 'size', 'soften_mask',
    'sometrue', 'sort', 'sqrt', 'squeeze', 'std', 'subtract', 'sum',
    'swapaxes', 'take', 'tan', 'tanh', 'trace', 'transpose', 'true_divide',
    'var', 'where', 'zeros', 'zeros_like',
    ]

MaskType = np.bool
nomask = MaskType(0)

class MaskedArrayFutureWarning(FutureWarning):
    pass

def _deprecate_argsort_axis(arr):
    """
    Adjust the axis passed to argsort, warning if necessary

    Parameters
    ----------
    arr
        The array which argsort was called on

    np.ma.argsort has a long-term bug where the default of the axis argument
    is wrong (gh-8701), which now must be kept for backwards compatibility.
    Thankfully, this only makes a difference when arrays are 2- or more-
    dimensional, so we only need a warning then.
    """
    if arr.ndim <= 1:
        # no warning needed - but switch to -1 anyway, to avoid surprising
        # subclasses, which are more likely to implement scalar axes.
        return -1
    else:
        # 2017-04-11, Numpy 1.13.0, gh-8701: warn on axis default
        warnings.warn(
            "In the future the default for argsort will be axis=-1, not the "
            "current None, to match its documentation and np.argsort. "
            "Explicitly pass -1 or None to silence this warning.",
            MaskedArrayFutureWarning, stacklevel=3)
        return None


def doc_note(initialdoc, note):
    """
    Adds a Notes section to an existing docstring.

    """
    if initialdoc is None:
        return
    if note is None:
        return initialdoc

    notesplit = re.split(r'\n\s*?Notes\n\s*?-----', inspect.cleandoc(initialdoc))
    notedoc = f"\n\nNotes\n-----\n{inspect.cleandoc(note)}\n"

    return ''.join(notesplit[:1] + [notedoc] + notesplit[1:])


def get_object_signature(obj):
    """
    Get the signature from obj

    """
    try:
        sig = formatargspec(*getargspec(obj))
    except TypeError:
        sig = ''
    return sig


###############################################################################
#                              Exceptions                                     #
###############################################################################


class MAError(Exception):
    """
    Class for masked array related errors.

    """
    pass


class MaskError(MAError):
    """
    Class for mask related errors.

    """
    pass


###############################################################################
#                           Filling options                                   #
###############################################################################


# b: boolean - c: complex - f: floats - i: integer - O: object - S: string
default_filler = {'b': True,
                  'c': 1.e20 + 0.0j,
                  'f': 1.e20,
                  'i': 999999,
                  'O': '?',
                  'S': b'N/A',
                  'u': 999999,
                  'V': b'???',
                  'U': 'N/A'
                  }

# Add datetime64 and timedelta64 types
for v in ["Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps",
          "fs", "as"]:
    default_filler["M8[" + v + "]"] = np.datetime64("NaT", v)
    default_filler["m8[" + v + "]"] = np.timedelta64("NaT", v)

float_types_list = [np.half, np.single, np.double, np.longdouble,
                    np.csingle, np.cdouble, np.clongdouble]

_minvals: dict[type, int] = {}
_maxvals: dict[type, int] = {}

for sctype in ntypes.sctypeDict.values():
    scalar_dtype = np.dtype(sctype)

    if scalar_dtype.kind in "Mm":
        info = np.iinfo(np.int64)
        min_val, max_val = info.min + 1, info.max
    elif np.issubdtype(scalar_dtype, np.integer):
        info = np.iinfo(sctype)
        min_val, max_val = info.min, info.max
    elif np.issubdtype(scalar_dtype, np.floating):
        info = np.finfo(sctype)
        min_val, max_val = info.min, info.max
    elif scalar_dtype.kind == "b":
        min_val, max_val = 0, 1
    else:
        min_val, max_val = None, None

    _minvals[sctype] = min_val
    _maxvals[sctype] = max_val

max_filler = _minvals
max_filler.update([(k, -np.inf) for k in float_types_list[:4]])
max_filler.update([(k, complex(-np.inf, -np.inf)) for k in float_types_list[-3:]])

min_filler = _maxvals
min_filler.update([(k, +np.inf) for k in float_types_list[:4]])
min_filler.update([(k, complex(+np.inf, +np.inf)) for k in float_types_list[-3:]])

del float_types_list

def _recursive_fill_value(dtype, f):
    """
    Recursively produce a fill value for `dtype`, calling f on scalar dtypes
    """
    if dtype.names is not None:
        # We wrap into `array` here, which ensures we use NumPy cast rules
        # for integer casts, this allows the use of 99999 as a fill value
        # for int8.
        # TODO: This is probably a mess, but should best preserve behavior?
        vals = tuple(
                np.array(_recursive_fill_value(dtype[name], f))
                for name in dtype.names)
        return np.array(vals, dtype=dtype)[()]  # decay to void scalar from 0d
    elif dtype.subdtype:
        subtype, shape = dtype.subdtype
        subval = _recursive_fill_value(subtype, f)
        return np.full(shape, subval)
    else:
        return f(dtype)


def _get_dtype_of(obj):
    """ Convert the argument for *_fill_value into a dtype """
    if isinstance(obj, np.dtype):
        return obj
    elif hasattr(obj, 'dtype'):
        return obj.dtype
    else:
        return np.asanyarray(obj).dtype


def default_fill_value(obj):
    """
    Return the default fill value for the argument object.

    The default filling value depends on the datatype of the input
    array or the type of the input scalar:

       ========  ========
       datatype  default
       ========  ========
       bool      True
       int       999999
       float     1.e20
       complex   1.e20+0j
       object    '?'
       string    'N/A'
       ========  ========

    For structured types, a structured scalar is returned, with each field the
    default fill value for its type.

    For subarray types, the fill value is an array of the same size containing
    the default scalar fill value.

    Parameters
    ----------
    obj : ndarray, dtype or scalar
        The array data-type or scalar for which the default fill value
        is returned.

    Returns
    -------
    fill_value : scalar
        The default fill value.

    Examples
    --------
    >>> import numpy as np
    >>> np.ma.default_fill_value(1)
    999999
    >>> np.ma.default_fill_value(np.array([1.1, 2., np.pi]))
    1e+20
    >>> np.ma.default_fill_value(np.dtype(complex))
    (1e+20+0j)

    """
    def _scalar_fill_value(dtype):
        if dtype.kind in 'Mm':
            return default_filler.get(dtype.str[1:], '?')
        else:
            return default_filler.get(dtype.kind, '?')

    dtype = _get_dtype_of(obj)
    return _recursive_fill_value(dtype, _scalar_fill_value)


def _extremum_fill_value(obj, extremum, extremum_name):

    def _scalar_fill_value(dtype):
        try:
            return extremum[dtype.type]
        except KeyError as e:
            raise TypeError(
                f"Unsuitable type {dtype} for calculating {extremum_name}."
            ) from None

    dtype = _get_dtype_of(obj)
    return _recursive_fill_value(dtype, _scalar_fill_value)


def minimum_fill_value(obj):
    """
    Return the maximum value that can be represented by the dtype of an object.

    This function is useful for calculating a fill value suitable for
    taking the minimum of an array with a given dtype.

    Parameters
    ----------
    obj : ndarray, dtype or scalar
        An object that can be queried for it's numeric type.

    Returns
    -------
    val : scalar
        The maximum representable value.

    Raises
    ------
    TypeError
        If `obj` isn't a suitable numeric type.

    See Also
    --------
    maximum_fill_value : The inverse function.
    set_fill_value : Set the filling value of a masked array.
    MaskedArray.fill_value : Return current fill value.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> a = np.int8()
    >>> ma.minimum_fill_value(a)
    127
    >>> a = np.int32()
    >>> ma.minimum_fill_value(a)
    2147483647

    An array of numeric data can also be passed.

    >>> a = np.array([1, 2, 3], dtype=np.int8)
    >>> ma.minimum_fill_value(a)
    127
    >>> a = np.array([1, 2, 3], dtype=np.float32)
    >>> ma.minimum_fill_value(a)
    inf

    """
    return _extremum_fill_value(obj, min_filler, "minimum")


def maximum_fill_value(obj):
    """
    Return the minimum value that can be represented by the dtype of an object.

    This function is useful for calculating a fill value suitable for
    taking the maximum of an array with a given dtype.

    Parameters
    ----------
    obj : ndarray, dtype or scalar
        An object that can be queried for it's numeric type.

    Returns
    -------
    val : scalar
        The minimum representable value.

    Raises
    ------
    TypeError
        If `obj` isn't a suitable numeric type.

    See Also
    --------
    minimum_fill_value : The inverse function.
    set_fill_value : Set the filling value of a masked array.
    MaskedArray.fill_value : Return current fill value.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> a = np.int8()
    >>> ma.maximum_fill_value(a)
    -128
    >>> a = np.int32()
    >>> ma.maximum_fill_value(a)
    -2147483648

    An array of numeric data can also be passed.

    >>> a = np.array([1, 2, 3], dtype=np.int8)
    >>> ma.maximum_fill_value(a)
    -128
    >>> a = np.array([1, 2, 3], dtype=np.float32)
    >>> ma.maximum_fill_value(a)
    -inf

    """
    return _extremum_fill_value(obj, max_filler, "maximum")


def _recursive_set_fill_value(fillvalue, dt):
    """
    Create a fill value for a structured dtype.

    Parameters
    ----------
    fillvalue : scalar or array_like
        Scalar or array representing the fill value. If it is of shorter
        length than the number of fields in dt, it will be resized.
    dt : dtype
        The structured dtype for which to create the fill value.

    Returns
    -------
    val : tuple
        A tuple of values corresponding to the structured fill value.

    """
    fillvalue = np.resize(fillvalue, len(dt.names))
    output_value = []
    for (fval, name) in zip(fillvalue, dt.names):
        cdtype = dt[name]
        if cdtype.subdtype:
            cdtype = cdtype.subdtype[0]

        if cdtype.names is not None:
            output_value.append(tuple(_recursive_set_fill_value(fval, cdtype)))
        else:
            output_value.append(np.array(fval, dtype=cdtype).item())
    return tuple(output_value)


def _check_fill_value(fill_value, ndtype):
    """
    Private function validating the given `fill_value` for the given dtype.

    If fill_value is None, it is set to the default corresponding to the dtype.

    If fill_value is not None, its value is forced to the given dtype.

    The result is always a 0d array.

    """
    ndtype = np.dtype(ndtype)
    if fill_value is None:
        fill_value = default_fill_value(ndtype)
        # TODO: It seems better to always store a valid fill_value, the oddity
        #       about is that `_fill_value = None` would behave even more
        #       different then.
        #       (e.g. this allows arr_uint8.astype(int64) to have the default
        #       fill value again...)
        # The one thing that changed in 2.0/2.1 around cast safety is that the
        # default `int(99...)` is not a same-kind cast anymore, so if we
        # have a uint, use the default uint.
        if ndtype.kind == "u":
            fill_value = np.uint(fill_value)
    elif ndtype.names is not None:
        if isinstance(fill_value, (ndarray, np.void)):
            try:
                fill_value = np.asarray(fill_value, dtype=ndtype)
            except ValueError as e:
                err_msg = "Unable to transform %s to dtype %s"
                raise ValueError(err_msg % (fill_value, ndtype)) from e
        else:
            fill_value = np.asarray(fill_value, dtype=object)
            fill_value = np.array(_recursive_set_fill_value(fill_value, ndtype),
                                  dtype=ndtype)
    elif isinstance(fill_value, str) and (ndtype.char not in 'OSVU'):
        # Note this check doesn't work if fill_value is not a scalar
        err_msg = "Cannot set fill value of string with array of dtype %s"
        raise TypeError(err_msg % ndtype)
    else:
        # In case we want to convert 1e20 to int.
        # Also in case of converting string arrays.
        try:
            fill_value = np.asarray(fill_value, dtype=ndtype)
        except (OverflowError, ValueError) as e:
            # Raise TypeError instead of OverflowError or ValueError.
            # OverflowError is seldom used, and the real problem here is
            # that the passed fill_value is not compatible with the ndtype.
            err_msg = "Cannot convert fill_value %s to dtype %s"
            raise TypeError(err_msg % (fill_value, ndtype)) from e
    return np.array(fill_value)


def set_fill_value(a, fill_value):
    """
    Set the filling value of a, if a is a masked array.

    This function changes the fill value of the masked array `a` in place.
    If `a` is not a masked array, the function returns silently, without
    doing anything.

    Parameters
    ----------
    a : array_like
        Input array.
    fill_value : dtype
        Filling value. A consistency test is performed to make sure
        the value is compatible with the dtype of `a`.

    Returns
    -------
    None
        Nothing returned by this function.

    See Also
    --------
    maximum_fill_value : Return the default fill value for a dtype.
    MaskedArray.fill_value : Return current fill value.
    MaskedArray.set_fill_value : Equivalent method.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> a = np.arange(5)
    >>> a
    array([0, 1, 2, 3, 4])
    >>> a = ma.masked_where(a < 3, a)
    >>> a
    masked_array(data=[--, --, --, 3, 4],
                 mask=[ True,  True,  True, False, False],
           fill_value=999999)
    >>> ma.set_fill_value(a, -999)
    >>> a
    masked_array(data=[--, --, --, 3, 4],
                 mask=[ True,  True,  True, False, False],
           fill_value=-999)

    Nothing happens if `a` is not a masked array.

    >>> a = list(range(5))
    >>> a
    [0, 1, 2, 3, 4]
    >>> ma.set_fill_value(a, 100)
    >>> a
    [0, 1, 2, 3, 4]
    >>> a = np.arange(5)
    >>> a
    array([0, 1, 2, 3, 4])
    >>> ma.set_fill_value(a, 100)
    >>> a
    array([0, 1, 2, 3, 4])

    """
    if isinstance(a, MaskedArray):
        a.set_fill_value(fill_value)


def get_fill_value(a):
    """
    Return the filling value of a, if any.  Otherwise, returns the
    default filling value for that type.

    """
    if isinstance(a, MaskedArray):
        result = a.fill_value
    else:
        result = default_fill_value(a)
    return result


def common_fill_value(a, b):
    """
    Return the common filling value of two masked arrays, if any.

    If ``a.fill_value == b.fill_value``, return the fill value,
    otherwise return None.

    Parameters
    ----------
    a, b : MaskedArray
        The masked arrays for which to compare fill values.

    Returns
    -------
    fill_value : scalar or None
        The common fill value, or None.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.ma.array([0, 1.], fill_value=3)
    >>> y = np.ma.array([0, 1.], fill_value=3)
    >>> np.ma.common_fill_value(x, y)
    3.0

    """
    t1 = get_fill_value(a)
    t2 = get_fill_value(b)
    if t1 == t2:
        return t1
    return None


def filled(a, fill_value=None):
    """
    Return input as an `~numpy.ndarray`, with masked values replaced by
    `fill_value`.

    If `a` is not a `MaskedArray`, `a` itself is returned.
    If `a` is a `MaskedArray` with no masked values, then ``a.data`` is
    returned.
    If `a` is a `MaskedArray` and `fill_value` is None, `fill_value` is set to
    ``a.fill_value``.

    Parameters
    ----------
    a : MaskedArray or array_like
        An input object.
    fill_value : array_like, optional.
        Can be scalar or non-scalar. If non-scalar, the
        resulting filled array should be broadcastable
        over input array. Default is None.

    Returns
    -------
    a : ndarray
        The filled array.

    See Also
    --------
    compressed

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> x = ma.array(np.arange(9).reshape(3, 3), mask=[[1, 0, 0],
    ...                                                [1, 0, 0],
    ...                                                [0, 0, 0]])
    >>> x.filled()
    array([[999999,      1,      2],
           [999999,      4,      5],
           [     6,      7,      8]])
    >>> x.filled(fill_value=333)
    array([[333,   1,   2],
           [333,   4,   5],
           [  6,   7,   8]])
    >>> x.filled(fill_value=np.arange(3))
    array([[0, 1, 2],
           [0, 4, 5],
           [6, 7, 8]])

    """
    if hasattr(a, 'filled'):
        return a.filled(fill_value)

    elif isinstance(a, ndarray):
        # Should we check for contiguity ? and a.flags['CONTIGUOUS']:
        return a
    elif isinstance(a, dict):
        return np.array(a, 'O')
    else:
        return np.array(a)


def get_masked_subclass(*arrays):
    """
    Return the youngest subclass of MaskedArray from a list of (masked) arrays.

    In case of siblings, the first listed takes over.

    """
    if len(arrays) == 1:
        arr = arrays[0]
        if isinstance(arr, MaskedArray):
            rcls = type(arr)
        else:
            rcls = MaskedArray
    else:
        arrcls = [type(a) for a in arrays]
        rcls = arrcls[0]
        if not issubclass(rcls, MaskedArray):
            rcls = MaskedArray
        for cls in arrcls[1:]:
            if issubclass(cls, rcls):
                rcls = cls
    # Don't return MaskedConstant as result: revert to MaskedArray
    if rcls.__name__ == 'MaskedConstant':
        return MaskedArray
    return rcls


def getdata(a, subok=True):
    """
    Return the data of a masked array as an ndarray.

    Return the data of `a` (if any) as an ndarray if `a` is a ``MaskedArray``,
    else return `a` as a ndarray or subclass (depending on `subok`) if not.

    Parameters
    ----------
    a : array_like
        Input ``MaskedArray``, alternatively a ndarray or a subclass thereof.
    subok : bool
        Whether to force the output to be a `pure` ndarray (False) or to
        return a subclass of ndarray if appropriate (True, default).

    See Also
    --------
    getmask : Return the mask of a masked array, or nomask.
    getmaskarray : Return the mask of a masked array, or full array of False.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> a = ma.masked_equal([[1,2],[3,4]], 2)
    >>> a
    masked_array(
      data=[[1, --],
            [3, 4]],
      mask=[[False,  True],
            [False, False]],
      fill_value=2)
    >>> ma.getdata(a)
    array([[1, 2],
           [3, 4]])

    Equivalently use the ``MaskedArray`` `data` attribute.

    >>> a.data
    array([[1, 2],
           [3, 4]])

    """
    try:
        data = a._data
    except AttributeError:
        data = np.array(a, copy=None, subok=subok)
    if not subok:
        return data.view(ndarray)
    return data


get_data = getdata


def fix_invalid(a, mask=nomask, copy=True, fill_value=None):
    """
    Return input with invalid data masked and replaced by a fill value.

    Invalid data means values of `nan`, `inf`, etc.

    Parameters
    ----------
    a : array_like
        Input array, a (subclass of) ndarray.
    mask : sequence, optional
        Mask. Must be convertible to an array of booleans with the same
        shape as `data`. True indicates a masked (i.e. invalid) data.
    copy : bool, optional
        Whether to use a copy of `a` (True) or to fix `a` in place (False).
        Default is True.
    fill_value : scalar, optional
        Value used for fixing invalid data. Default is None, in which case
        the ``a.fill_value`` is used.

    Returns
    -------
    b : MaskedArray
        The input array with invalid entries fixed.

    Notes
    -----
    A copy is performed by default.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.ma.array([1., -1, np.nan, np.inf], mask=[1] + [0]*3)
    >>> x
    masked_array(data=[--, -1.0, nan, inf],
                 mask=[ True, False, False, False],
           fill_value=1e+20)
    >>> np.ma.fix_invalid(x)
    masked_array(data=[--, -1.0, --, --],
                 mask=[ True, False,  True,  True],
           fill_value=1e+20)

    >>> fixed = np.ma.fix_invalid(x)
    >>> fixed.data
    array([ 1.e+00, -1.e+00,  1.e+20,  1.e+20])
    >>> x.data
    array([ 1., -1., nan, inf])

    """
    a = masked_array(a, copy=copy, mask=mask, subok=True)
    invalid = np.logical_not(np.isfinite(a._data))
    if not invalid.any():
        return a
    a._mask |= invalid
    if fill_value is None:
        fill_value = a.fill_value
    a._data[invalid] = fill_value
    return a

def is_string_or_list_of_strings(val):
    return (isinstance(val, str) or
            (isinstance(val, list) and val and
             builtins.all(isinstance(s, str) for s in val)))

###############################################################################
#                                  Ufuncs                                     #
###############################################################################


ufunc_domain = {}
ufunc_fills = {}


class _DomainCheckInterval:
    """
    Define a valid interval, so that :

    ``domain_check_interval(a,b)(x) == True`` where
    ``x < a`` or ``x > b``.

    """

    def __init__(self, a, b):
        "domain_check_interval(a,b)(x) = true where x < a or y > b"
        if a > b:
            (a, b) = (b, a)
        self.a = a
        self.b = b

    def __call__(self, x):
        "Execute the call behavior."
        # nans at masked positions cause RuntimeWarnings, even though
        # they are masked. To avoid this we suppress warnings.
        with np.errstate(invalid='ignore'):
            return umath.logical_or(umath.greater(x, self.b),
                                    umath.less(x, self.a))


class _DomainTan:
    """
    Define a valid interval for the `tan` function, so that:

    ``domain_tan(eps) = True`` where ``abs(cos(x)) < eps``

    """

    def __init__(self, eps):
        "domain_tan(eps) = true where abs(cos(x)) < eps)"
        self.eps = eps

    def __call__(self, x):
        "Executes the call behavior."
        with np.errstate(invalid='ignore'):
            return umath.less(umath.absolute(umath.cos(x)), self.eps)


class _DomainSafeDivide:
    """
    Define a domain for safe division.

    """

    def __init__(self, tolerance=None):
        self.tolerance = tolerance

    def __call__(self, a, b):
        # Delay the selection of the tolerance to here in order to reduce numpy
        # import times. The calculation of these parameters is a substantial
        # component of numpy's import time.
        if self.tolerance is None:
            self.tolerance = np.finfo(float).tiny
        # don't call ma ufuncs from __array_wrap__ which would fail for scalars
        a, b = np.asarray(a), np.asarray(b)
        with np.errstate(all='ignore'):
            return umath.absolute(a) * self.tolerance >= umath.absolute(b)


class _DomainGreater:
    """
    DomainGreater(v)(x) is True where x <= v.

    """

    def __init__(self, critical_value):
        "DomainGreater(v)(x) = true where x <= v"
        self.critical_value = critical_value

    def __call__(self, x):
        "Executes the call behavior."
        with np.errstate(invalid='ignore'):
            return umath.less_equal(x, self.critical_value)


class _DomainGreaterEqual:
    """
    DomainGreaterEqual(v)(x) is True where x < v.

    """

    def __init__(self, critical_value):
        "DomainGreaterEqual(v)(x) = true where x < v"
        self.critical_value = critical_value

    def __call__(self, x):
        "Executes the call behavior."
        with np.errstate(invalid='ignore'):
            return umath.less(x, self.critical_value)


class _MaskedUFunc:
    def __init__(self, ufunc):
        self.f = ufunc
        self.__doc__ = ufunc.__doc__
        self.__name__ = ufunc.__name__
        self.__qualname__ = ufunc.__qualname__

    def __str__(self):
        return f"Masked version of {self.f}"


class _MaskedUnaryOperation(_MaskedUFunc):
    """
    Defines masked version of unary operations, where invalid values are
    pre-masked.

    Parameters
    ----------
    mufunc : callable
        The function for which to define a masked version. Made available
        as ``_MaskedUnaryOperation.f``.
    fill : scalar, optional
        Filling value, default is 0.
    domain : class instance
        Domain for the function. Should be one of the ``_Domain*``
        classes. Default is None.

    """

    def __init__(self, mufunc, fill=0, domain=None):
        super().__init__(mufunc)
        self.fill = fill
        self.domain = domain
        ufunc_domain[mufunc] = domain
        ufunc_fills[mufunc] = fill

    def __call__(self, a, *args, **kwargs):
        """
        Execute the call behavior.

        """
        d = getdata(a)
        # Deal with domain
        if self.domain is not None:
            # Case 1.1. : Domained function
            # nans at masked positions cause RuntimeWarnings, even though
            # they are masked. To avoid this we suppress warnings.
            with np.errstate(divide='ignore', invalid='ignore'):
                result = self.f(d, *args, **kwargs)
            # Make a mask
            m = ~umath.isfinite(result)
            m |= self.domain(d)
            m |= getmask(a)
        else:
            # Case 1.2. : Function without a domain
            # Get the result and the mask
            with np.errstate(divide='ignore', invalid='ignore'):
                result = self.f(d, *args, **kwargs)
            m = getmask(a)

        if not result.ndim:
            # Case 2.1. : The result is scalarscalar
            if m:
                return masked
            return result

        if m is not nomask:
            # Case 2.2. The result is an array
            # We need to fill the invalid data back w/ the input Now,
            # that's plain silly: in C, we would just skip the element and
            # keep the original, but we do have to do it that way in Python

            # In case result has a lower dtype than the inputs (as in
            # equal)
            try:
                np.copyto(result, d, where=m)
            except TypeError:
                pass
        # Transform to
        masked_result = result.view(get_masked_subclass(a))
        masked_result._mask = m
        masked_result._update_from(a)
        return masked_result


class _MaskedBinaryOperation(_MaskedUFunc):
    """
    Define masked version of binary operations, where invalid
    values are pre-masked.

    Parameters
    ----------
    mbfunc : function
        The function for which to define a masked version. Made available
        as ``_MaskedBinaryOperation.f``.
    domain : class instance
        Default domain for the function. Should be one of the ``_Domain*``
        classes. Default is None.
    fillx : scalar, optional
        Filling value for the first argument, default is 0.
    filly : scalar, optional
        Filling value for the second argument, default is 0.

    """

    def __init__(self, mbfunc, fillx=0, filly=0):
        """
        abfunc(fillx, filly) must be defined.

        abfunc(x, filly) = x for all x to enable reduce.

        """
        super().__init__(mbfunc)
        self.fillx = fillx
        self.filly = filly
        ufunc_domain[mbfunc] = None
        ufunc_fills[mbfunc] = (fillx, filly)

    def __call__(self, a, b, *args, **kwargs):
        """
        Execute the call behavior.

        """
        # Get the data, as ndarray
        (da, db) = (getdata(a), getdata(b))
        # Get the result
        with np.errstate():
            np.seterr(divide='ignore', invalid='ignore')
            result = self.f(da, db, *args, **kwargs)
        # Get the mask for the result
        (ma, mb) = (getmask(a), getmask(b))
        if ma is nomask:
            if mb is nomask:
                m = nomask
            else:
                m = umath.logical_or(getmaskarray(a), mb)
        elif mb is nomask:
            m = umath.logical_or(ma, getmaskarray(b))
        else:
            m = umath.logical_or(ma, mb)

        # Case 1. : scalar
        if not result.ndim:
            if m:
                return masked
            return result

        # Case 2. : array
        # Revert result to da where masked
        if m is not nomask and m.any():
            # any errors, just abort; impossible to guarantee masked values
            try:
                np.copyto(result, da, casting='unsafe', where=m)
            except Exception:
                pass

        # Transforms to a (subclass of) MaskedArray
        masked_result = result.view(get_masked_subclass(a, b))
        masked_result._mask = m
        if isinstance(a, MaskedArray):
            masked_result._update_from(a)
        elif isinstance(b, MaskedArray):
            masked_result._update_from(b)
        return masked_result

    def reduce(self, target, axis=0, dtype=None):
        """
        Reduce `target` along the given `axis`.

        """
        tclass = get_masked_subclass(target)
        m = getmask(target)
        t = filled(target, self.filly)
        if t.shape == ():
            t = t.reshape(1)
            if m is not nomask:
                m = make_mask(m, copy=True)
                m.shape = (1,)

        if m is nomask:
            tr = self.f.reduce(t, axis)
            mr = nomask
        else:
            tr = self.f.reduce(t, axis, dtype=dtype)
            mr = umath.logical_and.reduce(m, axis)

        if not tr.shape:
            if mr:
                return masked
            else:
                return tr
        masked_tr = tr.view(tclass)
        masked_tr._mask = mr
        return masked_tr

    def outer(self, a, b):
        """
        Return the function applied to the outer product of a and b.

        """
        (da, db) = (getdata(a), getdata(b))
        d = self.f.outer(da, db)
        ma = getmask(a)
        mb = getmask(b)
        if ma is nomask and mb is nomask:
            m = nomask
        else:
            ma = getmaskarray(a)
            mb = getmaskarray(b)
            m = umath.logical_or.outer(ma, mb)
        if (not m.ndim) and m:
            return masked
        if m is not nomask:
            np.copyto(d, da, where=m)
        if not d.shape:
            return d
        masked_d = d.view(get_masked_subclass(a, b))
        masked_d._mask = m
        return masked_d

    def accumulate(self, target, axis=0):
        """Accumulate `target` along `axis` after filling with y fill
        value.

        """
        tclass = get_masked_subclass(target)
        t = filled(target, self.filly)
        result = self.f.accumulate(t, axis)
        masked_result = result.view(tclass)
        return masked_result


class _DomainedBinaryOperation(_MaskedUFunc):
    """
    Define binary operations that have a domain, like divide.

    They have no reduce, outer or accumulate.

    Parameters
    ----------
    mbfunc : function
        The function for which to define a masked version. Made available
        as ``_DomainedBinaryOperation.f``.
    domain : class instance
        Default domain for the function. Should be one of the ``_Domain*``
        classes.
    fillx : scalar, optional
        Filling value for the first argument, default is 0.
    filly : scalar, optional
        Filling value for the second argument, default is 0.

    """

    def __init__(self, dbfunc, domain, fillx=0, filly=0):
        """abfunc(fillx, filly) must be defined.
           abfunc(x, filly) = x for all x to enable reduce.
        """
        super().__init__(dbfunc)
        self.domain = domain
        self.fillx = fillx
        self.filly = filly
        ufunc_domain[dbfunc] = domain
        ufunc_fills[dbfunc] = (fillx, filly)

    def __call__(self, a, b, *args, **kwargs):
        "Execute the call behavior."
        # Get the data
        (da, db) = (getdata(a), getdata(b))
        # Get the result
        with np.errstate(divide='ignore', invalid='ignore'):
            result = self.f(da, db, *args, **kwargs)
        # Get the mask as a combination of the source masks and invalid
        m = ~umath.isfinite(result)
        m |= getmask(a)
        m |= getmask(b)
        # Apply the domain
        domain = ufunc_domain.get(self.f, None)
        if domain is not None:
            m |= domain(da, db)
        # Take care of the scalar case first
        if not m.ndim:
            if m:
                return masked
            else:
                return result
        # When the mask is True, put back da if possible
        # any errors, just abort; impossible to guarantee masked values
        try:
            np.copyto(result, 0, casting='unsafe', where=m)
            # avoid using "*" since this may be overlaid
            masked_da = umath.multiply(m, da)
            # only add back if it can be cast safely
            if np.can_cast(masked_da.dtype, result.dtype, casting='safe'):
                result += masked_da
        except Exception:
            pass

        # Transforms to a (subclass of) MaskedArray
        masked_result = result.view(get_masked_subclass(a, b))
        masked_result._mask = m
        if isinstance(a, MaskedArray):
            masked_result._update_from(a)
        elif isinstance(b, MaskedArray):
            masked_result._update_from(b)
        return masked_result


# Unary ufuncs
exp = _MaskedUnaryOperation(umath.exp)
conjugate = _MaskedUnaryOperation(umath.conjugate)
sin = _MaskedUnaryOperation(umath.sin)
cos = _MaskedUnaryOperation(umath.cos)
arctan = _MaskedUnaryOperation(umath.arctan)
arcsinh = _MaskedUnaryOperation(umath.arcsinh)
sinh = _MaskedUnaryOperation(umath.sinh)
cosh = _MaskedUnaryOperation(umath.cosh)
tanh = _MaskedUnaryOperation(umath.tanh)
abs = absolute = _MaskedUnaryOperation(umath.absolute)
angle = _MaskedUnaryOperation(angle)
fabs = _MaskedUnaryOperation(umath.fabs)
negative = _MaskedUnaryOperation(umath.negative)
floor = _MaskedUnaryOperation(umath.floor)
ceil = _MaskedUnaryOperation(umath.ceil)
around = _MaskedUnaryOperation(np.around)
logical_not = _MaskedUnaryOperation(umath.logical_not)

# Domained unary ufuncs
sqrt = _MaskedUnaryOperation(umath.sqrt, 0.0,
                             _DomainGreaterEqual(0.0))
log = _MaskedUnaryOperation(umath.log, 1.0,
                            _DomainGreater(0.0))
log2 = _MaskedUnaryOperation(umath.log2, 1.0,
                             _DomainGreater(0.0))
log10 = _MaskedUnaryOperation(umath.log10, 1.0,
                              _DomainGreater(0.0))
tan = _MaskedUnaryOperation(umath.tan, 0.0,
                            _DomainTan(1e-35))
arcsin = _MaskedUnaryOperation(umath.arcsin, 0.0,
                               _DomainCheckInterval(-1.0, 1.0))
arccos = _MaskedUnaryOperation(umath.arccos, 0.0,
                               _DomainCheckInterval(-1.0, 1.0))
arccosh = _MaskedUnaryOperation(umath.arccosh, 1.0,
                                _DomainGreaterEqual(1.0))
arctanh = _MaskedUnaryOperation(umath.arctanh, 0.0,
                                _DomainCheckInterval(-1.0 + 1e-15, 1.0 - 1e-15))

# Binary ufuncs
add = _MaskedBinaryOperation(umath.add)
subtract = _MaskedBinaryOperation(umath.subtract)
multiply = _MaskedBinaryOperation(umath.multiply, 1, 1)
arctan2 = _MaskedBinaryOperation(umath.arctan2, 0.0, 1.0)
equal = _MaskedBinaryOperation(umath.equal)
equal.reduce = None
not_equal = _MaskedBinaryOperation(umath.not_equal)
not_equal.reduce = None
less_equal = _MaskedBinaryOperation(umath.less_equal)
less_equal.reduce = None
greater_equal = _MaskedBinaryOperation(umath.greater_equal)
greater_equal.reduce = None
less = _MaskedBinaryOperation(umath.less)
less.reduce = None
greater = _MaskedBinaryOperation(umath.greater)
greater.reduce = None
logical_and = _MaskedBinaryOperation(umath.logical_and)
alltrue = _MaskedBinaryOperation(umath.logical_and, 1, 1).reduce
logical_or = _MaskedBinaryOperation(umath.logical_or)
sometrue = logical_or.reduce
logical_xor = _MaskedBinaryOperation(umath.logical_xor)
bitwise_and = _MaskedBinaryOperation(umath.bitwise_and)
bitwise_or = _MaskedBinaryOperation(umath.bitwise_or)
bitwise_xor = _MaskedBinaryOperation(umath.bitwise_xor)
hypot = _MaskedBinaryOperation(umath.hypot)

# Domained binary ufuncs
divide = _DomainedBinaryOperation(umath.divide, _DomainSafeDivide(), 0, 1)
true_divide = divide  # Just an alias for divide.
floor_divide = _DomainedBinaryOperation(umath.floor_divide,
                                        _DomainSafeDivide(), 0, 1)
remainder = _DomainedBinaryOperation(umath.remainder,
                                     _DomainSafeDivide(), 0, 1)
fmod = _DomainedBinaryOperation(umath.fmod, _DomainSafeDivide(), 0, 1)
mod = remainder

###############################################################################
#                        Mask creation functions                              #
###############################################################################


def _replace_dtype_fields_recursive(dtype, primitive_dtype):
    "Private function allowing recursion in _replace_dtype_fields."
    _recurse = _replace_dtype_fields_recursive

    # Do we have some name fields ?
    if dtype.names is not None:
        descr = []
        for name in dtype.names:
            field = dtype.fields[name]
            if len(field) == 3:
                # Prepend the title to the name
                name = (field[-1], name)
            descr.append((name, _recurse(field[0], primitive_dtype)))
        new_dtype = np.dtype(descr)

    # Is this some kind of composite a la (float,2)
    elif dtype.subdtype:
        descr = list(dtype.subdtype)
        descr[0] = _recurse(dtype.subdtype[0], primitive_dtype)
        new_dtype = np.dtype(tuple(descr))

    # this is a primitive type, so do a direct replacement
    else:
        new_dtype = primitive_dtype

    # preserve identity of dtypes
    if new_dtype == dtype:
        new_dtype = dtype

    return new_dtype


def _replace_dtype_fields(dtype, primitive_dtype):
    """
    Construct a dtype description list from a given dtype.

    Returns a new dtype object, with all fields and subtypes in the given type
    recursively replaced with `primitive_dtype`.

    Arguments are coerced to dtypes first.
    """
    dtype = np.dtype(dtype)
    primitive_dtype = np.dtype(primitive_dtype)
    return _replace_dtype_fields_recursive(dtype, primitive_dtype)


def make_mask_descr(ndtype):
    """
    Construct a dtype description list from a given dtype.

    Returns a new dtype object, with the type of all fields in `ndtype` to a
    boolean type. Field names are not altered.

    Parameters
    ----------
    ndtype : dtype
        The dtype to convert.

    Returns
    -------
    result : dtype
        A dtype that looks like `ndtype`, the type of all fields is boolean.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> dtype = np.dtype({'names':['foo', 'bar'],
    ...                   'formats':[np.float32, np.int64]})
    >>> dtype
    dtype([('foo', '<f4'), ('bar', '<i8')])
    >>> ma.make_mask_descr(dtype)
    dtype([('foo', '|b1'), ('bar', '|b1')])
    >>> ma.make_mask_descr(np.float32)
    dtype('bool')

    """
    return _replace_dtype_fields(ndtype, MaskType)


def getmask(a):
    """
    Return the mask of a masked array, or nomask.

    Return the mask of `a` as an ndarray if `a` is a `MaskedArray` and the
    mask is not `nomask`, else return `nomask`. To guarantee a full array
    of booleans of the same shape as a, use `getmaskarray`.

    Parameters
    ----------
    a : array_like
        Input `MaskedArray` for which the mask is required.

    See Also
    --------
    getdata : Return the data of a masked array as an ndarray.
    getmaskarray : Return the mask of a masked array, or full array of False.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> a = ma.masked_equal([[1,2],[3,4]], 2)
    >>> a
    masked_array(
      data=[[1, --],
            [3, 4]],
      mask=[[False,  True],
            [False, False]],
      fill_value=2)
    >>> ma.getmask(a)
    array([[False,  True],
           [False, False]])

    Equivalently use the `MaskedArray` `mask` attribute.

    >>> a.mask
    array([[False,  True],
           [False, False]])

    Result when mask == `nomask`

    >>> b = ma.masked_array([[1,2],[3,4]])
    >>> b
    masked_array(
      data=[[1, 2],
            [3, 4]],
      mask=False,
      fill_value=999999)
    >>> ma.nomask
    False
    >>> ma.getmask(b) == ma.nomask
    True
    >>> b.mask == ma.nomask
    True

    """
    return getattr(a, '_mask', nomask)


get_mask = getmask


def getmaskarray(arr):
    """
    Return the mask of a masked array, or full boolean array of False.

    Return the mask of `arr` as an ndarray if `arr` is a `MaskedArray` and
    the mask is not `nomask`, else return a full boolean array of False of
    the same shape as `arr`.

    Parameters
    ----------
    arr : array_like
        Input `MaskedArray` for which the mask is required.

    See Also
    --------
    getmask : Return the mask of a masked array, or nomask.
    getdata : Return the data of a masked array as an ndarray.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> a = ma.masked_equal([[1,2],[3,4]], 2)
    >>> a
    masked_array(
      data=[[1, --],
            [3, 4]],
      mask=[[False,  True],
            [False, False]],
      fill_value=2)
    >>> ma.getmaskarray(a)
    array([[False,  True],
           [False, False]])

    Result when mask == ``nomask``

    >>> b = ma.masked_array([[1,2],[3,4]])
    >>> b
    masked_array(
      data=[[1, 2],
            [3, 4]],
      mask=False,
      fill_value=999999)
    >>> ma.getmaskarray(b)
    array([[False, False],
           [False, False]])

    """
    mask = getmask(arr)
    if mask is nomask:
        mask = make_mask_none(np.shape(arr), getattr(arr, 'dtype', None))
    return mask


def is_mask(m):
    """
    Return True if m is a valid, standard mask.

    This function does not check the contents of the input, only that the
    type is MaskType. In particular, this function returns False if the
    mask has a flexible dtype.

    Parameters
    ----------
    m : array_like
        Array to test.

    Returns
    -------
    result : bool
        True if `m.dtype.type` is MaskType, False otherwise.

    See Also
    --------
    ma.isMaskedArray : Test whether input is an instance of MaskedArray.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> m = ma.masked_equal([0, 1, 0, 2, 3], 0)
    >>> m
    masked_array(data=[--, 1, --, 2, 3],
                 mask=[ True, False,  True, False, False],
           fill_value=0)
    >>> ma.is_mask(m)
    False
    >>> ma.is_mask(m.mask)
    True

    Input must be an ndarray (or have similar attributes)
    for it to be considered a valid mask.

    >>> m = [False, True, False]
    >>> ma.is_mask(m)
    False
    >>> m = np.array([False, True, False])
    >>> m
    array([False,  True, False])
    >>> ma.is_mask(m)
    True

    Arrays with complex dtypes don't return True.

    >>> dtype = np.dtype({'names':['monty', 'pithon'],
    ...                   'formats':[bool, bool]})
    >>> dtype
    dtype([('monty', '|b1'), ('pithon', '|b1')])
    >>> m = np.array([(True, False), (False, True), (True, False)],
    ...              dtype=dtype)
    >>> m
    array([( True, False), (False,  True), ( True, False)],
          dtype=[('monty', '?'), ('pithon', '?')])
    >>> ma.is_mask(m)
    False

    """
    try:
        return m.dtype.type is MaskType
    except AttributeError:
        return False


def _shrink_mask(m):
    """
    Shrink a mask to nomask if possible
    """
    if m.dtype.names is None and not m.any():
        return nomask
    else:
        return m


def make_mask(m, copy=False, shrink=True, dtype=MaskType):
    """
    Create a boolean mask from an array.

    Return `m` as a boolean mask, creating a copy if necessary or requested.
    The function can accept any sequence that is convertible to integers,
    or ``nomask``.  Does not require that contents must be 0s and 1s, values
    of 0 are interpreted as False, everything else as True.

    Parameters
    ----------
    m : array_like
        Potential mask.
    copy : bool, optional
        Whether to return a copy of `m` (True) or `m` itself (False).
    shrink : bool, optional
        Whether to shrink `m` to ``nomask`` if all its values are False.
    dtype : dtype, optional
        Data-type of the output mask. By default, the output mask has a
        dtype of MaskType (bool). If the dtype is flexible, each field has
        a boolean dtype. This is ignored when `m` is ``nomask``, in which
        case ``nomask`` is always returned.

    Returns
    -------
    result : ndarray
        A boolean mask derived from `m`.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> m = [True, False, True, True]
    >>> ma.make_mask(m)
    array([ True, False,  True,  True])
    >>> m = [1, 0, 1, 1]
    >>> ma.make_mask(m)
    array([ True, False,  True,  True])
    >>> m = [1, 0, 2, -3]
    >>> ma.make_mask(m)
    array([ True, False,  True,  True])

    Effect of the `shrink` parameter.

    >>> m = np.zeros(4)
    >>> m
    array([0., 0., 0., 0.])
    >>> ma.make_mask(m)
    False
    >>> ma.make_mask(m, shrink=False)
    array([False, False, False, False])

    Using a flexible `dtype`.

    >>> m = [1, 0, 1, 1]
    >>> n = [0, 1, 0, 0]
    >>> arr = []
    >>> for man, mouse in zip(m, n):
    ...     arr.append((man, mouse))
    >>> arr
    [(1, 0), (0, 1), (1, 0), (1, 0)]
    >>> dtype = np.dtype({'names':['man', 'mouse'],
    ...                   'formats':[np.int64, np.int64]})
    >>> arr = np.array(arr, dtype=dtype)
    >>> arr
    array([(1, 0), (0, 1), (1, 0), (1, 0)],
          dtype=[('man', '<i8'), ('mouse', '<i8')])
    >>> ma.make_mask(arr, dtype=dtype)
    array([(True, False), (False, True), (True, False), (True, False)],
          dtype=[('man', '|b1'), ('mouse', '|b1')])

    """
    if m is nomask:
        return nomask

    # Make sure the input dtype is valid.
    dtype = make_mask_descr(dtype)

    # legacy boolean special case: "existence of fields implies true"
    if isinstance(m, ndarray) and m.dtype.fields and dtype == np.bool:
        return np.ones(m.shape, dtype=dtype)

    # Fill the mask in case there are missing data; turn it into an ndarray.
    copy = None if not copy else True
    result = np.array(filled(m, True), copy=copy, dtype=dtype, subok=True)
    # Bas les masques !
    if shrink:
        result = _shrink_mask(result)
    return result


def make_mask_none(newshape, dtype=None):
    """
    Return a boolean mask of the given shape, filled with False.

    This function returns a boolean ndarray with all entries False, that can
    be used in common mask manipulations. If a complex dtype is specified, the
    type of each field is converted to a boolean type.

    Parameters
    ----------
    newshape : tuple
        A tuple indicating the shape of the mask.
    dtype : {None, dtype}, optional
        If None, use a MaskType instance. Otherwise, use a new datatype with
        the same fields as `dtype`, converted to boolean types.

    Returns
    -------
    result : ndarray
        An ndarray of appropriate shape and dtype, filled with False.

    See Also
    --------
    make_mask : Create a boolean mask from an array.
    make_mask_descr : Construct a dtype description list from a given dtype.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> ma.make_mask_none((3,))
    array([False, False, False])

    Defining a more complex dtype.

    >>> dtype = np.dtype({'names':['foo', 'bar'],
    ...                   'formats':[np.float32, np.int64]})
    >>> dtype
    dtype([('foo', '<f4'), ('bar', '<i8')])
    >>> ma.make_mask_none((3,), dtype=dtype)
    array([(False, False), (False, False), (False, False)],
          dtype=[('foo', '|b1'), ('bar', '|b1')])

    """
    if dtype is None:
        result = np.zeros(newshape, dtype=MaskType)
    else:
        result = np.zeros(newshape, dtype=make_mask_descr(dtype))
    return result


def _recursive_mask_or(m1, m2, newmask):
    names = m1.dtype.names
    for name in names:
        current1 = m1[name]
        if current1.dtype.names is not None:
            _recursive_mask_or(current1, m2[name], newmask[name])
        else:
            umath.logical_or(current1, m2[name], newmask[name])


def mask_or(m1, m2, copy=False, shrink=True):
    """
    Combine two masks with the ``logical_or`` operator.

    The result may be a view on `m1` or `m2` if the other is `nomask`
    (i.e. False).

    Parameters
    ----------
    m1, m2 : array_like
        Input masks.
    copy : bool, optional
        If copy is False and one of the inputs is `nomask`, return a view
        of the other input mask. Defaults to False.
    shrink : bool, optional
        Whether to shrink the output to `nomask` if all its values are
        False. Defaults to True.

    Returns
    -------
    mask : output mask
        The result masks values that are masked in either `m1` or `m2`.

    Raises
    ------
    ValueError
        If `m1` and `m2` have different flexible dtypes.

    Examples
    --------
    >>> import numpy as np
    >>> m1 = np.ma.make_mask([0, 1, 1, 0])
    >>> m2 = np.ma.make_mask([1, 0, 0, 0])
    >>> np.ma.mask_or(m1, m2)
    array([ True,  True,  True, False])

    """

    if (m1 is nomask) or (m1 is False):
        dtype = getattr(m2, 'dtype', MaskType)
        return make_mask(m2, copy=copy, shrink=shrink, dtype=dtype)
    if (m2 is nomask) or (m2 is False):
        dtype = getattr(m1, 'dtype', MaskType)
        return make_mask(m1, copy=copy, shrink=shrink, dtype=dtype)
    if m1 is m2 and is_mask(m1):
        return _shrink_mask(m1) if shrink else m1
    (dtype1, dtype2) = (getattr(m1, 'dtype', None), getattr(m2, 'dtype', None))
    if dtype1 != dtype2:
        raise ValueError(f"Incompatible dtypes '{dtype1}'<>'{dtype2}'")
    if dtype1.names is not None:
        # Allocate an output mask array with the properly broadcast shape.
        newmask = np.empty(np.broadcast(m1, m2).shape, dtype1)
        _recursive_mask_or(m1, m2, newmask)
        return newmask
    return make_mask(umath.logical_or(m1, m2), copy=copy, shrink=shrink)


def flatten_mask(mask):
    """
    Returns a completely flattened version of the mask, where nested fields
    are collapsed.

    Parameters
    ----------
    mask : array_like
        Input array, which will be interpreted as booleans.

    Returns
    -------
    flattened_mask : ndarray of bools
        The flattened input.

    Examples
    --------
    >>> import numpy as np
    >>> mask = np.array([0, 0, 1])
    >>> np.ma.flatten_mask(mask)
    array([False, False,  True])

    >>> mask = np.array([(0, 0), (0, 1)], dtype=[('a', bool), ('b', bool)])
    >>> np.ma.flatten_mask(mask)
    array([False, False, False,  True])

    >>> mdtype = [('a', bool), ('b', [('ba', bool), ('bb', bool)])]
    >>> mask = np.array([(0, (0, 0)), (0, (0, 1))], dtype=mdtype)
    >>> np.ma.flatten_mask(mask)
    array([False, False, False, False, False,  True])

    """

    def _flatmask(mask):
        "Flatten the mask and returns a (maybe nested) sequence of booleans."
        mnames = mask.dtype.names
        if mnames is not None:
            return [flatten_mask(mask[name]) for name in mnames]
        else:
            return mask

    def _flatsequence(sequence):
        "Generates a flattened version of the sequence."
        try:
            for element in sequence:
                if hasattr(element, '__iter__'):
                    yield from _flatsequence(element)
                else:
                    yield element
        except TypeError:
            yield sequence

    mask = np.asarray(mask)
    flattened = _flatsequence(_flatmask(mask))
    return np.array(list(flattened), dtype=bool)


def _check_mask_axis(mask, axis, keepdims=np._NoValue):
    "Check whether there are masked values along the given axis"
    kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}
    if mask is not nomask:
        return mask.all(axis=axis, **kwargs)
    return nomask


###############################################################################
#                             Masking functions                               #
###############################################################################

def masked_where(condition, a, copy=True):
    """
    Mask an array where a condition is met.

    Return `a` as an array masked where `condition` is True.
    Any masked values of `a` or `condition` are also masked in the output.

    Parameters
    ----------
    condition : array_like
        Masking condition.  When `condition` tests floating point values for
        equality, consider using ``masked_values`` instead.
    a : array_like
        Array to mask.
    copy : bool
        If True (default) make a copy of `a` in the result.  If False modify
        `a` in place and return a view.

    Returns
    -------
    result : MaskedArray
        The result of masking `a` where `condition` is True.

    See Also
    --------
    masked_values : Mask using floating point equality.
    masked_equal : Mask where equal to a given value.
    masked_not_equal : Mask where *not* equal to a given value.
    masked_less_equal : Mask where less than or equal to a given value.
    masked_greater_equal : Mask where greater than or equal to a given value.
    masked_less : Mask where less than a given value.
    masked_greater : Mask where greater than a given value.
    masked_inside : Mask inside a given interval.
    masked_outside : Mask outside a given interval.
    masked_invalid : Mask invalid values (NaNs or infs).

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> a = np.arange(4)
    >>> a
    array([0, 1, 2, 3])
    >>> ma.masked_where(a <= 2, a)
    masked_array(data=[--, --, --, 3],
                 mask=[ True,  True,  True, False],
           fill_value=999999)

    Mask array `b` conditional on `a`.

    >>> b = ['a', 'b', 'c', 'd']
    >>> ma.masked_where(a == 2, b)
    masked_array(data=['a', 'b', --, 'd'],
                 mask=[False, False,  True, False],
           fill_value='N/A',
                dtype='<U1')

    Effect of the `copy` argument.

    >>> c = ma.masked_where(a <= 2, a)
    >>> c
    masked_array(data=[--, --, --, 3],
                 mask=[ True,  True,  True, False],
           fill_value=999999)
    >>> c[0] = 99
    >>> c
    masked_array(data=[99, --, --, 3],
                 mask=[False,  True,  True, False],
           fill_value=999999)
    >>> a
    array([0, 1, 2, 3])
    >>> c = ma.masked_where(a <= 2, a, copy=False)
    >>> c[0] = 99
    >>> c
    masked_array(data=[99, --, --, 3],
                 mask=[False,  True,  True, False],
           fill_value=999999)
    >>> a
    array([99,  1,  2,  3])

    When `condition` or `a` contain masked values.

    >>> a = np.arange(4)
    >>> a = ma.masked_where(a == 2, a)
    >>> a
    masked_array(data=[0, 1, --, 3],
                 mask=[False, False,  True, False],
           fill_value=999999)
    >>> b = np.arange(4)
    >>> b = ma.masked_where(b == 0, b)
    >>> b
    masked_array(data=[--, 1, 2, 3],
                 mask=[ True, False, False, False],
           fill_value=999999)
    >>> ma.masked_where(a == 3, b)
    masked_array(data=[--, 1, --, --],
                 mask=[ True, False,  True,  True],
           fill_value=999999)

    """
    # Make sure that condition is a valid standard-type mask.
    cond = make_mask(condition, shrink=False)
    a = np.array(a, copy=copy, subok=True)

    (cshape, ashape) = (cond.shape, a.shape)
    if cshape and cshape != ashape:
        raise IndexError("Inconsistent shape between the condition and the input"
                         " (got %s and %s)" % (cshape, ashape))
    if hasattr(a, '_mask'):
        cond = mask_or(cond, a._mask)
        cls = type(a)
    else:
        cls = MaskedArray
    result = a.view(cls)
    # Assign to *.mask so that structured masks are handled correctly.
    result.mask = _shrink_mask(cond)
    # There is no view of a boolean so when 'a' is a MaskedArray with nomask
    # the update to the result's mask has no effect.
    if not copy and hasattr(a, '_mask') and getmask(a) is nomask:
        a._mask = result._mask.view()
    return result


def masked_greater(x, value, copy=True):
    """
    Mask an array where greater than a given value.

    This function is a shortcut to ``masked_where``, with
    `condition` = (x > value).

    See Also
    --------
    masked_where : Mask where a condition is met.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> a = np.arange(4)
    >>> a
    array([0, 1, 2, 3])
    >>> ma.masked_greater(a, 2)
    masked_array(data=[0, 1, 2, --],
                 mask=[False, False, False,  True],
           fill_value=999999)

    """
    return masked_where(greater(x, value), x, copy=copy)


def masked_greater_equal(x, value, copy=True):
    """
    Mask an array where greater than or equal to a given value.

    This function is a shortcut to ``masked_where``, with
    `condition` = (x >= value).

    See Also
    --------
    masked_where : Mask where a condition is met.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> a = np.arange(4)
    >>> a
    array([0, 1, 2, 3])
    >>> ma.masked_greater_equal(a, 2)
    masked_array(data=[0, 1, --, --],
                 mask=[False, False,  True,  True],
           fill_value=999999)

    """
    return masked_where(greater_equal(x, value), x, copy=copy)


def masked_less(x, value, copy=True):
    """
    Mask an array where less than a given value.

    This function is a shortcut to ``masked_where``, with
    `condition` = (x < value).

    See Also
    --------
    masked_where : Mask where a condition is met.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> a = np.arange(4)
    >>> a
    array([0, 1, 2, 3])
    >>> ma.masked_less(a, 2)
    masked_array(data=[--, --, 2, 3],
                 mask=[ True,  True, False, False],
           fill_value=999999)

    """
    return masked_where(less(x, value), x, copy=copy)


def masked_less_equal(x, value, copy=True):
    """
    Mask an array where less than or equal to a given value.

    This function is a shortcut to ``masked_where``, with
    `condition` = (x <= value).

    See Also
    --------
    masked_where : Mask where a condition is met.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> a = np.arange(4)
    >>> a
    array([0, 1, 2, 3])
    >>> ma.masked_less_equal(a, 2)
    masked_array(data=[--, --, --, 3],
                 mask=[ True,  True,  True, False],
           fill_value=999999)

    """
    return masked_where(less_equal(x, value), x, copy=copy)


def masked_not_equal(x, value, copy=True):
    """
    Mask an array where *not* equal to a given value.

    This function is a shortcut to ``masked_where``, with
    `condition` = (x != value).

    See Also
    --------
    masked_where : Mask where a condition is met.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> a = np.arange(4)
    >>> a
    array([0, 1, 2, 3])
    >>> ma.masked_not_equal(a, 2)
    masked_array(data=[--, --, 2, --],
                 mask=[ True,  True, False,  True],
           fill_value=999999)

    """
    return masked_where(not_equal(x, value), x, copy=copy)


def masked_equal(x, value, copy=True):
    """
    Mask an array where equal to a given value.

    Return a MaskedArray, masked where the data in array `x` are
    equal to `value`. The fill_value of the returned MaskedArray
    is set to `value`.

    For floating point arrays, consider using ``masked_values(x, value)``.

    See Also
    --------
    masked_where : Mask where a condition is met.
    masked_values : Mask using floating point equality.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> a = np.arange(4)
    >>> a
    array([0, 1, 2, 3])
    >>> ma.masked_equal(a, 2)
    masked_array(data=[0, 1, --, 3],
                 mask=[False, False,  True, False],
           fill_value=2)

    """
    output = masked_where(equal(x, value), x, copy=copy)
    output.fill_value = value
    return output


def masked_inside(x, v1, v2, copy=True):
    """
    Mask an array inside a given interval.

    Shortcut to ``masked_where``, where `condition` is True for `x` inside
    the interval [v1,v2] (v1 <= x <= v2).  The boundaries `v1` and `v2`
    can be given in either order.

    See Also
    --------
    masked_where : Mask where a condition is met.

    Notes
    -----
    The array `x` is prefilled with its filling value.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> x = [0.31, 1.2, 0.01, 0.2, -0.4, -1.1]
    >>> ma.masked_inside(x, -0.3, 0.3)
    masked_array(data=[0.31, 1.2, --, --, -0.4, -1.1],
                 mask=[False, False,  True,  True, False, False],
           fill_value=1e+20)

    The order of `v1` and `v2` doesn't matter.

    >>> ma.masked_inside(x, 0.3, -0.3)
    masked_array(data=[0.31, 1.2, --, --, -0.4, -1.1],
                 mask=[False, False,  True,  True, False, False],
           fill_value=1e+20)

    """
    if v2 < v1:
        (v1, v2) = (v2, v1)
    xf = filled(x)
    condition = (xf >= v1) & (xf <= v2)
    return masked_where(condition, x, copy=copy)


def masked_outside(x, v1, v2, copy=True):
    """
    Mask an array outside a given interval.

    Shortcut to ``masked_where``, where `condition` is True for `x` outside
    the interval [v1,v2] (x < v1)|(x > v2).
    The boundaries `v1` and `v2` can be given in either order.

    See Also
    --------
    masked_where : Mask where a condition is met.

    Notes
    -----
    The array `x` is prefilled with its filling value.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> x = [0.31, 1.2, 0.01, 0.2, -0.4, -1.1]
    >>> ma.masked_outside(x, -0.3, 0.3)
    masked_array(data=[--, --, 0.01, 0.2, --, --],
                 mask=[ True,  True, False, False,  True,  True],
           fill_value=1e+20)

    The order of `v1` and `v2` doesn't matter.

    >>> ma.masked_outside(x, 0.3, -0.3)
    masked_array(data=[--, --, 0.01, 0.2, --, --],
                 mask=[ True,  True, False, False,  True,  True],
           fill_value=1e+20)

    """
    if v2 < v1:
        (v1, v2) = (v2, v1)
    xf = filled(x)
    condition = (xf < v1) | (xf > v2)
    return masked_where(condition, x, copy=copy)


def masked_object(x, value, copy=True, shrink=True):
    """
    Mask the array `x` where the data are exactly equal to value.

    This function is similar to `masked_values`, but only suitable
    for object arrays: for floating point, use `masked_values` instead.

    Parameters
    ----------
    x : array_like
        Array to mask
    value : object
        Comparison value
    copy : {True, False}, optional
        Whether to return a copy of `x`.
    shrink : {True, False}, optional
        Whether to collapse a mask full of False to nomask

    Returns
    -------
    result : MaskedArray
        The result of masking `x` where equal to `value`.

    See Also
    --------
    masked_where : Mask where a condition is met.
    masked_equal : Mask where equal to a given value (integers).
    masked_values : Mask using floating point equality.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> food = np.array(['green_eggs', 'ham'], dtype=object)
    >>> # don't eat spoiled food
    >>> eat = ma.masked_object(food, 'green_eggs')
    >>> eat
    masked_array(data=[--, 'ham'],
                 mask=[ True, False],
           fill_value='green_eggs',
                dtype=object)
    >>> # plain ol` ham is boring
    >>> fresh_food = np.array(['cheese', 'ham', 'pineapple'], dtype=object)
    >>> eat = ma.masked_object(fresh_food, 'green_eggs')
    >>> eat
    masked_array(data=['cheese', 'ham', 'pineapple'],
                 mask=False,
           fill_value='green_eggs',
                dtype=object)

    Note that `mask` is set to ``nomask`` if possible.

    >>> eat
    masked_array(data=['cheese', 'ham', 'pineapple'],
                 mask=False,
           fill_value='green_eggs',
                dtype=object)

    """
    if isMaskedArray(x):
        condition = umath.equal(x._data, value)
        mask = x._mask
    else:
        condition = umath.equal(np.asarray(x), value)
        mask = nomask
    mask = mask_or(mask, make_mask(condition, shrink=shrink))
    return masked_array(x, mask=mask, copy=copy, fill_value=value)


def masked_values(x, value, rtol=1e-5, atol=1e-8, copy=True, shrink=True):
    """
    Mask using floating point equality.

    Return a MaskedArray, masked where the data in array `x` are approximately
    equal to `value`, determined using `isclose`. The default tolerances for
    `masked_values` are the same as those for `isclose`.

    For integer types, exact equality is used, in the same way as
    `masked_equal`.

    The fill_value is set to `value` and the mask is set to ``nomask`` if
    possible.

    Parameters
    ----------
    x : array_like
        Array to mask.
    value : float
        Masking value.
    rtol, atol : float, optional
        Tolerance parameters passed on to `isclose`
    copy : bool, optional
        Whether to return a copy of `x`.
    shrink : bool, optional
        Whether to collapse a mask full of False to ``nomask``.

    Returns
    -------
    result : MaskedArray
        The result of masking `x` where approximately equal to `value`.

    See Also
    --------
    masked_where : Mask where a condition is met.
    masked_equal : Mask where equal to a given value (integers).

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> x = np.array([1, 1.1, 2, 1.1, 3])
    >>> ma.masked_values(x, 1.1)
    masked_array(data=[1.0, --, 2.0, --, 3.0],
                 mask=[False,  True, False,  True, False],
           fill_value=1.1)

    Note that `mask` is set to ``nomask`` if possible.

    >>> ma.masked_values(x, 2.1)
    masked_array(data=[1. , 1.1, 2. , 1.1, 3. ],
                 mask=False,
           fill_value=2.1)

    Unlike `masked_equal`, `masked_values` can perform approximate equalities.

    >>> ma.masked_values(x, 2.1, atol=1e-1)
    masked_array(data=[1.0, 1.1, --, 1.1, 3.0],
                 mask=[False, False,  True, False, False],
           fill_value=2.1)

    """
    xnew = filled(x, value)
    if np.issubdtype(xnew.dtype, np.floating):
        mask = np.isclose(xnew, value, atol=atol, rtol=rtol)
    else:
        mask = umath.equal(xnew, value)
    ret = masked_array(xnew, mask=mask, copy=copy, fill_value=value)
    if shrink:
        ret.shrink_mask()
    return ret


def masked_invalid(a, copy=True):
    """
    Mask an array where invalid values occur (NaNs or infs).

    This function is a shortcut to ``masked_where``, with
    `condition` = ~(np.isfinite(a)). Any pre-existing mask is conserved.
    Only applies to arrays with a dtype where NaNs or infs make sense
    (i.e. floating point types), but accepts any array_like object.

    See Also
    --------
    masked_where : Mask where a condition is met.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> a = np.arange(5, dtype=float)
    >>> a[2] = np.nan
    >>> a[3] = np.inf
    >>> a
    array([ 0.,  1., nan, inf,  4.])
    >>> ma.masked_invalid(a)
    masked_array(data=[0.0, 1.0, --, --, 4.0],
                 mask=[False, False,  True,  True, False],
           fill_value=1e+20)

    """
    a = np.array(a, copy=None, subok=True)
    res = masked_where(~(np.isfinite(a)), a, copy=copy)
    # masked_invalid previously never returned nomask as a mask and doing so
    # threw off matplotlib (gh-22842).  So use shrink=False:
    if res._mask is nomask:
        res._mask = make_mask_none(res.shape, res.dtype)
    return res

###############################################################################
#                            Printing options                                 #
###############################################################################


class _MaskedPrintOption:
    """
    Handle the string used to represent missing data in a masked array.

    """

    def __init__(self, display):
        """
        Create the masked_print_option object.

        """
        self._display = display
        self._enabled = True

    def display(self):
        """
        Display the string to print for masked values.

        """
        return self._display

    def set_display(self, s):
        """
        Set the string to print for masked values.

        """
        self._display = s

    def enabled(self):
        """
        Is the use of the display value enabled?

        """
        return self._enabled

    def enable(self, shrink=1):
        """
        Set the enabling shrink to `shrink`.

        """
        self._enabled = shrink

    def __str__(self):
        return str(self._display)

    __repr__ = __str__


# if you single index into a masked location you get this object.
masked_print_option = _MaskedPrintOption('--')


def _recursive_printoption(result, mask, printopt):
    """
    Puts printoptions in result where mask is True.

    Private function allowing for recursion

    """
    names = result.dtype.names
    if names is not None:
        for name in names:
            curdata = result[name]
            curmask = mask[name]
            _recursive_printoption(curdata, curmask, printopt)
    else:
        np.copyto(result, printopt, where=mask)


# For better or worse, these end in a newline
_legacy_print_templates = {
    'long_std': textwrap.dedent("""\
        masked_%(name)s(data =
         %(data)s,
        %(nlen)s        mask =
         %(mask)s,
        %(nlen)s  fill_value = %(fill)s)
        """),
    'long_flx': textwrap.dedent("""\
        masked_%(name)s(data =
         %(data)s,
        %(nlen)s        mask =
         %(mask)s,
        %(nlen)s  fill_value = %(fill)s,
        %(nlen)s       dtype = %(dtype)s)
        """),
    'short_std': textwrap.dedent("""\
        masked_%(name)s(data = %(data)s,
        %(nlen)s        mask = %(mask)s,
        %(nlen)s  fill_value = %(fill)s)
        """),
    'short_flx': textwrap.dedent("""\
        masked_%(name)s(data = %(data)s,
        %(nlen)s        mask = %(mask)s,
        %(nlen)s  fill_value = %(fill)s,
        %(nlen)s       dtype = %(dtype)s)
        """)
}

###############################################################################
#                          MaskedArray class                                  #
###############################################################################


def _recursive_filled(a, mask, fill_value):
    """
    Recursively fill `a` with `fill_value`.

    """
    names = a.dtype.names
    for name in names:
        current = a[name]
        if current.dtype.names is not None:
            _recursive_filled(current, mask[name], fill_value[name])
        else:
            np.copyto(current, fill_value[name], where=mask[name])


def flatten_structured_array(a):
    """
    Flatten a structured array.

    The data type of the output is chosen such that it can represent all of the
    (nested) fields.

    Parameters
    ----------
    a : structured array

    Returns
    -------
    output : masked array or ndarray
        A flattened masked array if the input is a masked array, otherwise a
        standard ndarray.

    Examples
    --------
    >>> import numpy as np
    >>> ndtype = [('a', int), ('b', float)]
    >>> a = np.array([(1, 1), (2, 2)], dtype=ndtype)
    >>> np.ma.flatten_structured_array(a)
    array([[1., 1.],
           [2., 2.]])

    """

    def flatten_sequence(iterable):
        """
        Flattens a compound of nested iterables.

        """
        for elm in iter(iterable):
            if hasattr(elm, '__iter__'):
                yield from flatten_sequence(elm)
            else:
                yield elm

    a = np.asanyarray(a)
    inishape = a.shape
    a = a.ravel()
    if isinstance(a, MaskedArray):
        out = np.array([tuple(flatten_sequence(d.item())) for d in a._data])
        out = out.view(MaskedArray)
        out._mask = np.array([tuple(flatten_sequence(d.item()))
                              for d in getmaskarray(a)])
    else:
        out = np.array([tuple(flatten_sequence(d.item())) for d in a])
    if len(inishape) > 1:
        newshape = list(out.shape)
        newshape[0] = inishape
        out.shape = tuple(flatten_sequence(newshape))
    return out


def _arraymethod(funcname, onmask=True):
    """
    Return a class method wrapper around a basic array method.

    Creates a class method which returns a masked array, where the new
    ``_data`` array is the output of the corresponding basic method called
    on the original ``_data``.

    If `onmask` is True, the new mask is the output of the method called
    on the initial mask. Otherwise, the new mask is just a reference
    to the initial mask.

    Parameters
    ----------
    funcname : str
        Name of the function to apply on data.
    onmask : bool
        Whether the mask must be processed also (True) or left
        alone (False). Default is True. Make available as `_onmask`
        attribute.

    Returns
    -------
    method : instancemethod
        Class method wrapper of the specified basic array method.

    """
    def wrapped_method(self, *args, **params):
        result = getattr(self._data, funcname)(*args, **params)
        result = result.view(type(self))
        result._update_from(self)
        mask = self._mask
        if not onmask:
            result.__setmask__(mask)
        elif mask is not nomask:
            # __setmask__ makes a copy, which we don't want
            result._mask = getattr(mask, funcname)(*args, **params)
        return result
    methdoc = getattr(ndarray, funcname, None) or getattr(np, funcname, None)
    if methdoc is not None:
        wrapped_method.__doc__ = methdoc.__doc__
    wrapped_method.__name__ = funcname
    return wrapped_method


class MaskedIterator:
    """
    Flat iterator object to iterate over masked arrays.

    A `MaskedIterator` iterator is returned by ``x.flat`` for any masked array
    `x`. It allows iterating over the array as if it were a 1-D array,
    either in a for-loop or by calling its `next` method.

    Iteration is done in C-contiguous style, with the last index varying the
    fastest. The iterator can also be indexed using basic slicing or
    advanced indexing.

    See Also
    --------
    MaskedArray.flat : Return a flat iterator over an array.
    MaskedArray.flatten : Returns a flattened copy of an array.

    Notes
    -----
    `MaskedIterator` is not exported by the `ma` module. Instead of
    instantiating a `MaskedIterator` directly, use `MaskedArray.flat`.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.ma.array(arange(6).reshape(2, 3))
    >>> fl = x.flat
    >>> type(fl)
    <class 'numpy.ma.MaskedIterator'>
    >>> for item in fl:
    ...     print(item)
    ...
    0
    1
    2
    3
    4
    5

    Extracting more than a single element b indexing the `MaskedIterator`
    returns a masked array:

    >>> fl[2:4]
    masked_array(data = [2 3],
                 mask = False,
           fill_value = 999999)

    """

    def __init__(self, ma):
        self.ma = ma
        self.dataiter = ma._data.flat

        if ma._mask is nomask:
            self.maskiter = None
        else:
            self.maskiter = ma._mask.flat

    def __iter__(self):
        return self

    def __getitem__(self, indx):
        result = self.dataiter.__getitem__(indx).view(type(self.ma))
        if self.maskiter is not None:
            _mask = self.maskiter.__getitem__(indx)
            if isinstance(_mask, ndarray):
                # set shape to match that of data; this is needed for matrices
                _mask.shape = result.shape
                result._mask = _mask
            elif isinstance(_mask, np.void):
                return mvoid(result, mask=_mask, hardmask=self.ma._hardmask)
            elif _mask:  # Just a scalar, masked
                return masked
        return result

    # This won't work if ravel makes a copy
    def __setitem__(self, index, value):
        self.dataiter[index] = getdata(value)
        if self.maskiter is not None:
            self.maskiter[index] = getmaskarray(value)

    def __next__(self):
        """
        Return the next value, or raise StopIteration.

        Examples
        --------
        >>> import numpy as np
        >>> x = np.ma.array([3, 2], mask=[0, 1])
        >>> fl = x.flat
        >>> next(fl)
        3
        >>> next(fl)
        masked
        >>> next(fl)
        Traceback (most recent call last):
          ...
        StopIteration

        """
        d = next(self.dataiter)
        if self.maskiter is not None:
            m = next(self.maskiter)
            if isinstance(m, np.void):
                return mvoid(d, mask=m, hardmask=self.ma._hardmask)
            elif m:  # Just a scalar, masked
                return masked
        return d


@set_module("numpy.ma")
class MaskedArray(ndarray):
    """
    An array class with possibly masked values.

    Masked values of True exclude the corresponding element from any
    computation.

    Construction::

      x = MaskedArray(data, mask=nomask, dtype=None, copy=False, subok=True,
                      ndmin=0, fill_value=None, keep_mask=True, hard_mask=None,
                      shrink=True, order=None)

    Parameters
    ----------
    data : array_like
        Input data.
    mask : sequence, optional
        Mask. Must be convertible to an array of booleans with the same
        shape as `data`. True indicates a masked (i.e. invalid) data.
    dtype : dtype, optional
        Data type of the output.
        If `dtype` is None, the type of the data argument (``data.dtype``)
        is used. If `dtype` is not None and different from ``data.dtype``,
        a copy is performed.
    copy : bool, optional
        Whether to copy the input data (True), or to use a reference instead.
        Default is False.
    subok : bool, optional
        Whether to return a subclass of `MaskedArray` if possible (True) or a
        plain `MaskedArray`. Default is True.
    ndmin : int, optional
        Minimum number of dimensions. Default is 0.
    fill_value : scalar, optional
        Value used to fill in the masked values when necessary.
        If None, a default based on the data-type is used.
    keep_mask : bool, optional
        Whether to combine `mask` with the mask of the input data, if any
        (True), or to use only `mask` for the output (False). Default is True.
    hard_mask : bool, optional
        Whether to use a hard mask or not. With a hard mask, masked values
        cannot be unmasked. Default is False.
    shrink : bool, optional
        Whether to force compression of an empty mask. Default is True.
    order : {'C', 'F', 'A'}, optional
        Specify the order of the array.  If order is 'C', then the array
        will be in C-contiguous order (last-index varies the fastest).
        If order is 'F', then the returned array will be in
        Fortran-contiguous order (first-index varies the fastest).
        If order is 'A' (default), then the returned array may be
        in any order (either C-, Fortran-contiguous, or even discontiguous),
        unless a copy is required, in which case it will be C-contiguous.

    Examples
    --------
    >>> import numpy as np

    The ``mask`` can be initialized with an array of boolean values
    with the same shape as ``data``.

    >>> data = np.arange(6).reshape((2, 3))
    >>> np.ma.MaskedArray(data, mask=[[False, True, False],
    ...                               [False, False, True]])
    masked_array(
      data=[[0, --, 2],
            [3, 4, --]],
      mask=[[False,  True, False],
            [False, False,  True]],
      fill_value=999999)

    Alternatively, the ``mask`` can be initialized to homogeneous boolean
    array with the same shape as ``data`` by passing in a scalar
    boolean value:

    >>> np.ma.MaskedArray(data, mask=False)
    masked_array(
      data=[[0, 1, 2],
            [3, 4, 5]],
      mask=[[False, False, False],
            [False, False, False]],
      fill_value=999999)

    >>> np.ma.MaskedArray(data, mask=True)
    masked_array(
      data=[[--, --, --],
            [--, --, --]],
      mask=[[ True,  True,  True],
            [ True,  True,  True]],
      fill_value=999999,
      dtype=int64)

    .. note::
        The recommended practice for initializing ``mask`` with a scalar
        boolean value is to use ``True``/``False`` rather than
        ``np.True_``/``np.False_``. The reason is :attr:`nomask`
        is represented internally as ``np.False_``.

        >>> np.False_ is np.ma.nomask
        True

    """

    __array_priority__ = 15
    _defaultmask = nomask
    _defaulthardmask = False
    _baseclass = ndarray

    # Maximum number of elements per axis used when printing an array. The
    # 1d case is handled separately because we need more values in this case.
    _print_width = 100
    _print_width_1d = 1500

    def __new__(cls, data=None, mask=nomask, dtype=None, copy=False,
                subok=True, ndmin=0, fill_value=None, keep_mask=True,
                hard_mask=None, shrink=True, order=None):
        """
        Create a new masked array from scratch.

        Notes
        -----
        A masked array can also be created by taking a .view(MaskedArray).

        """
        # Process data.
        copy = None if not copy else True
        _data = np.array(data, dtype=dtype, copy=copy,
                         order=order, subok=True, ndmin=ndmin)
        _baseclass = getattr(data, '_baseclass', type(_data))
        # Check that we're not erasing the mask.
        if isinstance(data, MaskedArray) and (data.shape != _data.shape):
            copy = True

        # Here, we copy the _view_, so that we can attach new properties to it
        # we must never do .view(MaskedConstant), as that would create a new
        # instance of np.ma.masked, which make identity comparison fail
        if isinstance(data, cls) and subok and not isinstance(data, MaskedConstant):
            _data = ndarray.view(_data, type(data))
        else:
            _data = ndarray.view(_data, cls)

        # Handle the case where data is not a subclass of ndarray, but
        # still has the _mask attribute like MaskedArrays
        if hasattr(data, '_mask') and not isinstance(data, ndarray):
            _data._mask = data._mask
            # FIXME: should we set `_data._sharedmask = True`?
        # Process mask.
        # Type of the mask
        mdtype = make_mask_descr(_data.dtype)
        if mask is nomask:
            # Case 1. : no mask in input.
            # Erase the current mask ?
            if not keep_mask:
                # With a reduced version
                if shrink:
                    _data._mask = nomask
                # With full version
                else:
                    _data._mask = np.zeros(_data.shape, dtype=mdtype)
            # Check whether we missed something
            elif isinstance(data, (tuple, list)):
                try:
                    # If data is a sequence of masked array
                    mask = np.array(
                        [getmaskarray(np.asanyarray(m, dtype=_data.dtype))
                         for m in data], dtype=mdtype)
                except (ValueError, TypeError):
                    # If data is nested
                    mask = nomask
                # Force shrinking of the mask if needed (and possible)
                if (mdtype == MaskType) and mask.any():
                    _data._mask = mask
                    _data._sharedmask = False
            else:
                _data._sharedmask = not copy
                if copy:
                    _data._mask = _data._mask.copy()
                    # Reset the shape of the original mask
                    if getmask(data) is not nomask:
                        # gh-21022 encounters an issue here
                        # because data._mask.shape is not writeable, but
                        # the op was also pointless in that case, because
                        # the shapes were the same, so we can at least
                        # avoid that path
                        if data._mask.shape != data.shape:
                            data._mask.shape = data.shape
        else:
            # Case 2. : With a mask in input.
            # If mask is boolean, create an array of True or False

            # if users pass `mask=None` be forgiving here and cast it False
            # for speed; although the default is `mask=nomask` and can differ.
            if mask is None:
                mask = False

            if mask is True and mdtype == MaskType:
                mask = np.ones(_data.shape, dtype=mdtype)
            elif mask is False and mdtype == MaskType:
                mask = np.zeros(_data.shape, dtype=mdtype)
            else:
                # Read the mask with the current mdtype
                try:
                    mask = np.array(mask, copy=copy, dtype=mdtype)
                # Or assume it's a sequence of bool/int
                except TypeError:
                    mask = np.array([tuple([m] * len(mdtype)) for m in mask],
                                    dtype=mdtype)
            # Make sure the mask and the data have the same shape
            if mask.shape != _data.shape:
                (nd, nm) = (_data.size, mask.size)
                if nm == 1:
                    mask = np.resize(mask, _data.shape)
                elif nm == nd:
                    mask = np.reshape(mask, _data.shape)
                else:
                    msg = (f"Mask and data not compatible:"
                           f" data size is {nd}, mask size is {nm}.")
                    raise MaskError(msg)
                copy = True
            # Set the mask to the new value
            if _data._mask is nomask:
                _data._mask = mask
                _data._sharedmask = not copy
            elif not keep_mask:
                _data._mask = mask
                _data._sharedmask = not copy
            else:
                if _data.dtype.names is not None:
                    def _recursive_or(a, b):
                        "do a|=b on each field of a, recursively"
                        for name in a.dtype.names:
                            (af, bf) = (a[name], b[name])
                            if af.dtype.names is not None:
                                _recursive_or(af, bf)
                            else:
                                af |= bf

                    _recursive_or(_data._mask, mask)
                else:
                    _data._mask = np.logical_or(mask, _data._mask)
                _data._sharedmask = False

        # Update fill_value.
        if fill_value is None:
            fill_value = getattr(data, '_fill_value', None)
        # But don't run the check unless we have something to check.
        if fill_value is not None:
            _data._fill_value = _check_fill_value(fill_value, _data.dtype)
        # Process extra options ..
        if hard_mask is None:
            _data._hardmask = getattr(data, '_hardmask', False)
        else:
            _data._hardmask = hard_mask
        _data._baseclass = _baseclass
        return _data

    def _update_from(self, obj):
        """
        Copies some attributes of obj to self.

        """
        if isinstance(obj, ndarray):
            _baseclass = type(obj)
        else:
            _baseclass = ndarray
        # We need to copy the _basedict to avoid backward propagation
        _optinfo = {}
        _optinfo.update(getattr(obj, '_optinfo', {}))
        _optinfo.update(getattr(obj, '_basedict', {}))
        if not isinstance(obj, MaskedArray):
            _optinfo.update(getattr(obj, '__dict__', {}))
        _dict = {'_fill_value': getattr(obj, '_fill_value', None),
                     '_hardmask': getattr(obj, '_hardmask', False),
                     '_sharedmask': getattr(obj, '_sharedmask', False),
                     '_isfield': getattr(obj, '_isfield', False),
                     '_baseclass': getattr(obj, '_baseclass', _baseclass),
                     '_optinfo': _optinfo,
                     '_basedict': _optinfo}
        self.__dict__.update(_dict)
        self.__dict__.update(_optinfo)

    def __array_finalize__(self, obj):
        """
        Finalizes the masked array.

        """
        # Get main attributes.
        self._update_from(obj)

        # We have to decide how to initialize self.mask, based on
        # obj.mask. This is very difficult.  There might be some
        # correspondence between the elements in the array we are being
        # created from (= obj) and us. Or there might not. This method can
        # be called in all kinds of places for all kinds of reasons -- could
        # be empty_like, could be slicing, could be a ufunc, could be a view.
        # The numpy subclassing interface simply doesn't give us any way
        # to know, which means that at best this method will be based on
        # guesswork and heuristics. To make things worse, there isn't even any
        # clear consensus about what the desired behavior is. For instance,
        # most users think that np.empty_like(marr) -- which goes via this
        # method -- should return a masked array with an empty mask (see
        # gh-3404 and linked discussions), but others disagree, and they have
        # existing code which depends on empty_like returning an array that
        # matches the input mask.
        #
        # Historically our algorithm was: if the template object mask had the
        # same *number of elements* as us, then we used *it's mask object
        # itself* as our mask, so that writes to us would also write to the
        # original array. This is horribly broken in multiple ways.
        #
        # Now what we do instead is, if the template object mask has the same
        # number of elements as us, and we do not have the same base pointer
        # as the template object (b/c views like arr[...] should keep the same
        # mask), then we make a copy of the template object mask and use
        # that. This is also horribly broken but somewhat less so. Maybe.
        if isinstance(obj, ndarray):
            # XX: This looks like a bug -- shouldn't it check self.dtype
            # instead?
            if obj.dtype.names is not None:
                _mask = getmaskarray(obj)
            else:
                _mask = getmask(obj)

            # If self and obj point to exactly the same data, then probably
            # self is a simple view of obj (e.g., self = obj[...]), so they
            # should share the same mask. (This isn't 100% reliable, e.g. self
            # could be the first row of obj, or have strange strides, but as a
            # heuristic it's not bad.) In all other cases, we make a copy of
            # the mask, so that future modifications to 'self' do not end up
            # side-effecting 'obj' as well.
            if (_mask is not nomask and obj.__array_interface__["data"][0]
                    != self.__array_interface__["data"][0]):
                # We should make a copy. But we could get here via astype,
                # in which case the mask might need a new dtype as well
                # (e.g., changing to or from a structured dtype), and the
                # order could have changed. So, change the mask type if
                # needed and use astype instead of copy.
                if self.dtype == obj.dtype:
                    _mask_dtype = _mask.dtype
                else:
                    _mask_dtype = make_mask_descr(self.dtype)

                if self.flags.c_contiguous:
                    order = "C"
                elif self.flags.f_contiguous:
                    order = "F"
                else:
                    order = "K"

                _mask = _mask.astype(_mask_dtype, order)
            else:
                # Take a view so shape changes, etc., do not propagate back.
                _mask = _mask.view()
        else:
            _mask = nomask

        self._mask = _mask
        # Finalize the mask
        if self._mask is not nomask:
            try:
                self._mask.shape = self.shape
            except ValueError:
                self._mask = nomask
            except (TypeError, AttributeError):
                # When _mask.shape is not writable (because it's a void)
                pass

        # Finalize the fill_value
        if self._fill_value is not None:
            self._fill_value = _check_fill_value(self._fill_value, self.dtype)
        elif self.dtype.names is not None:
            # Finalize the default fill_value for structured arrays
            self._fill_value = _check_fill_value(None, self.dtype)

    def __array_wrap__(self, obj, context=None, return_scalar=False):
        """
        Special hook for ufuncs.

        Wraps the numpy array and sets the mask according to context.

        """
        if obj is self:  # for in-place operations
            result = obj
        else:
            result = obj.view(type(self))
            result._update_from(self)

        if context is not None:
            result._mask = result._mask.copy()
            func, args, out_i = context
            # args sometimes contains outputs (gh-10459), which we don't want
            input_args = args[:func.nin]
            m = functools.reduce(mask_or, [getmaskarray(arg) for arg in input_args])
            # Get the domain mask
            domain = ufunc_domain.get(func)
            if domain is not None:
                # Take the domain, and make sure it's a ndarray
                with np.errstate(divide='ignore', invalid='ignore'):
                    # The result may be masked for two (unary) domains.
                    # That can't really be right as some domains drop
                    # the mask and some don't behaving differently here.
                    d = domain(*input_args).astype(bool, copy=False)
                    d = filled(d, True)

                if d.any():
                    # Fill the result where the domain is wrong
                    try:
                        # Binary domain: take the last value
                        fill_value = ufunc_fills[func][-1]
                    except TypeError:
                        # Unary domain: just use this one
                        fill_value = ufunc_fills[func]
                    except KeyError:
                        # Domain not recognized, use fill_value instead
                        fill_value = self.fill_value

                    np.copyto(result, fill_value, where=d)

                    # Update the mask
                    if m is nomask:
                        m = d
                    else:
                        # Don't modify inplace, we risk back-propagation
                        m = (m | d)

            # Make sure the mask has the proper size
            if result is not self and result.shape == () and m:
                return masked
            else:
                result._mask = m
                result._sharedmask = False

        return result

    def view(self, dtype=None, type=None, fill_value=None):
        """
        Return a view of the MaskedArray data.

        Parameters
        ----------
        dtype : data-type or ndarray sub-class, optional
            Data-type descriptor of the returned view, e.g., float32 or int16.
            The default, None, results in the view having the same data-type
            as `a`. As with ``ndarray.view``, dtype can also be specified as
            an ndarray sub-class, which then specifies the type of the
            returned object (this is equivalent to setting the ``type``
            parameter).
        type : Python type, optional
            Type of the returned view, either ndarray or a subclass.  The
            default None results in type preservation.
        fill_value : scalar, optional
            The value to use for invalid entries (None by default).
            If None, then this argument is inferred from the passed `dtype`, or
            in its absence the original array, as discussed in the notes below.

        See Also
        --------
        numpy.ndarray.view : Equivalent method on ndarray object.

        Notes
        -----

        ``a.view()`` is used two different ways:

        ``a.view(some_dtype)`` or ``a.view(dtype=some_dtype)`` constructs a view
        of the array's memory with a different data-type.  This can cause a
        reinterpretation of the bytes of memory.

        ``a.view(ndarray_subclass)`` or ``a.view(type=ndarray_subclass)`` just
        returns an instance of `ndarray_subclass` that looks at the same array
        (same shape, dtype, etc.)  This does not cause a reinterpretation of the
        memory.

        If `fill_value` is not specified, but `dtype` is specified (and is not
        an ndarray sub-class), the `fill_value` of the MaskedArray will be
        reset. If neither `fill_value` nor `dtype` are specified (or if
        `dtype` is an ndarray sub-class), then the fill value is preserved.
        Finally, if `fill_value` is specified, but `dtype` is not, the fill
        value is set to the specified value.

        For ``a.view(some_dtype)``, if ``some_dtype`` has a different number of
        bytes per entry than the previous dtype (for example, converting a
        regular array to a structured array), then the behavior of the view
        cannot be predicted just from the superficial appearance of ``a`` (shown
        by ``print(a)``). It also depends on exactly how ``a`` is stored in
        memory. Therefore if ``a`` is C-ordered versus fortran-ordered, versus
        defined as a slice or transpose, etc., the view may give different
        results.
        """

        if dtype is None:
            if type is None:
                output = ndarray.view(self)
            else:
                output = ndarray.view(self, type)
        elif type is None:
            try:
                if issubclass(dtype, ndarray):
                    output = ndarray.view(self, dtype)
                    dtype = None
                else:
                    output = ndarray.view(self, dtype)
            except TypeError:
                output = ndarray.view(self, dtype)
        else:
            output = ndarray.view(self, dtype, type)

        # also make the mask be a view (so attr changes to the view's
        # mask do no affect original object's mask)
        # (especially important to avoid affecting np.masked singleton)
        if getmask(output) is not nomask:
            output._mask = output._mask.view()

        # Make sure to reset the _fill_value if needed
        if getattr(output, '_fill_value', None) is not None:
            if fill_value is None:
                if dtype is None:
                    pass  # leave _fill_value as is
                else:
                    output._fill_value = None
            else:
                output.fill_value = fill_value
        return output

    def __getitem__(self, indx):
        """
        x.__getitem__(y) <==> x[y]

        Return the item described by i, as a masked array.

        """
        # We could directly use ndarray.__getitem__ on self.
        # But then we would have to modify __array_finalize__ to prevent the
        # mask of being reshaped if it hasn't been set up properly yet
        # So it's easier to stick to the current version
        dout = self.data[indx]
        _mask = self._mask

        def _is_scalar(m):
            return not isinstance(m, np.ndarray)

        def _scalar_heuristic(arr, elem):
            """
            Return whether `elem` is a scalar result of indexing `arr`, or None
            if undecidable without promoting nomask to a full mask
            """
            # obviously a scalar
            if not isinstance(elem, np.ndarray):
                return True

            # object array scalar indexing can return anything
            elif arr.dtype.type is np.object_:
                if arr.dtype is not elem.dtype:
                    # elem is an array, but dtypes do not match, so must be
                    # an element
                    return True

            # well-behaved subclass that only returns 0d arrays when
            # expected - this is not a scalar
            elif type(arr).__getitem__ == ndarray.__getitem__:
                return False

            return None

        if _mask is not nomask:
            # _mask cannot be a subclass, so it tells us whether we should
            # expect a scalar. It also cannot be of dtype object.
            mout = _mask[indx]
            scalar_expected = _is_scalar(mout)

        else:
            # attempt to apply the heuristic to avoid constructing a full mask
            mout = nomask
            scalar_expected = _scalar_heuristic(self.data, dout)
            if scalar_expected is None:
                # heuristics have failed
                # construct a full array, so we can be certain. This is costly.
                # we could also fall back on ndarray.__getitem__(self.data, indx)
                scalar_expected = _is_scalar(getmaskarray(self)[indx])

        # Did we extract a single item?
        if scalar_expected:
            # A record
            if isinstance(dout, np.void):
                # We should always re-cast to mvoid, otherwise users can
                # change masks on rows that already have masked values, but not
                # on rows that have no masked values, which is inconsistent.
                return mvoid(dout, mask=mout, hardmask=self._hardmask)

            # special case introduced in gh-5962
            elif (self.dtype.type is np.object_ and
                  isinstance(dout, np.ndarray) and
                  dout is not masked):
                # If masked, turn into a MaskedArray, with everything masked.
                if mout:
                    return MaskedArray(dout, mask=True)
                else:
                    return dout

            # Just a scalar
            elif mout:
                return masked
            else:
                return dout
        else:
            # Force dout to MA
            dout = dout.view(type(self))
            # Inherit attributes from self
            dout._update_from(self)
            # Check the fill_value
            if is_string_or_list_of_strings(indx):
                if self._fill_value is not None:
                    dout._fill_value = self._fill_value[indx]

                    # Something like gh-15895 has happened if this check fails.
                    # _fill_value should always be an ndarray.
                    if not isinstance(dout._fill_value, np.ndarray):
                        raise RuntimeError('Internal NumPy error.')
                    # If we're indexing a multidimensional field in a
                    # structured array (such as dtype("(2,)i2,(2,)i1")),
                    # dimensionality goes up (M[field].ndim == M.ndim +
                    # M.dtype[field].ndim).  That's fine for
                    # M[field] but problematic for M[field].fill_value
                    # which should have shape () to avoid breaking several
                    # methods. There is no great way out, so set to
                    # first element. See issue #6723.
                    if dout._fill_value.ndim > 0:
                        if not (dout._fill_value ==
                                dout._fill_value.flat[0]).all():
                            warnings.warn(
                                "Upon accessing multidimensional field "
                                f"{indx!s}, need to keep dimensionality "
                                "of fill_value at 0. Discarding "
                                "heterogeneous fill_value and setting "
                                f"all to {dout._fill_value[0]!s}.",
                                stacklevel=2)
                        # Need to use `.flat[0:1].squeeze(...)` instead of just
                        # `.flat[0]` to ensure the result is a 0d array and not
                        # a scalar.
                        dout._fill_value = dout._fill_value.flat[0:1].squeeze(axis=0)
                dout._isfield = True
            # Update the mask if needed
            if mout is not nomask:
                # set shape to match that of data; this is needed for matrices
                dout._mask = reshape(mout, dout.shape)
                dout._sharedmask = True
                # Note: Don't try to check for m.any(), that'll take too long
        return dout

    # setitem may put NaNs into integer arrays or occasionally overflow a
    # float.  But this may happen in masked values, so avoid otherwise
    # correct warnings (as is typical also in masked calculations).
    @np.errstate(over='ignore', invalid='ignore')
    def __setitem__(self, indx, value):
        """
        x.__setitem__(i, y) <==> x[i]=y

        Set item described by index. If value is masked, masks those
        locations.

        """
        if self is masked:
            raise MaskError('Cannot alter the masked element.')
        _data = self._data
        _mask = self._mask
        if isinstance(indx, str):
            _data[indx] = value
            if _mask is nomask:
                self._mask = _mask = make_mask_none(self.shape, self.dtype)
            _mask[indx] = getmask(value)
            return

        _dtype = _data.dtype

        if value is masked:
            # The mask wasn't set: create a full version.
            if _mask is nomask:
                _mask = self._mask = make_mask_none(self.shape, _dtype)
            # Now, set the mask to its value.
            if _dtype.names is not None:
                _mask[indx] = tuple([True] * len(_dtype.names))
            else:
                _mask[indx] = True
            return

        # Get the _data part of the new value
        dval = getattr(value, '_data', value)
        # Get the _mask part of the new value
        mval = getmask(value)
        if _dtype.names is not None and mval is nomask:
            mval = tuple([False] * len(_dtype.names))
        if _mask is nomask:
            # Set the data, then the mask
            _data[indx] = dval
            if mval is not nomask:
                _mask = self._mask = make_mask_none(self.shape, _dtype)
                _mask[indx] = mval
        elif not self._hardmask:
            # Set the data, then the mask
            if (isinstance(indx, masked_array) and
                    not isinstance(value, masked_array)):
                _data[indx.data] = dval
            else:
                _data[indx] = dval
                _mask[indx] = mval
        elif hasattr(indx, 'dtype') and (indx.dtype == MaskType):
            indx = indx * umath.logical_not(_mask)
            _data[indx] = dval
        else:
            if _dtype.names is not None:
                err_msg = "Flexible 'hard' masks are not yet supported."
                raise NotImplementedError(err_msg)
            mindx = mask_or(_mask[indx], mval, copy=True)
            dindx = self._data[indx]
            if dindx.size > 1:
                np.copyto(dindx, dval, where=~mindx)
            elif mindx is nomask:
                dindx = dval
            _data[indx] = dindx
            _mask[indx] = mindx
        return

    # Define so that we can overwrite the setter.
    @property
    def dtype(self):
        return super().dtype

    @dtype.setter
    def dtype(self, dtype):
        super(MaskedArray, type(self)).dtype.__set__(self, dtype)
        if self._mask is not nomask:
            self._mask = self._mask.view(make_mask_descr(dtype), ndarray)
            # Try to reset the shape of the mask (if we don't have a void).
            # This raises a ValueError if the dtype change won't work.
            try:
                self._mask.shape = self.shape
            except (AttributeError, TypeError):
                pass

    @property
    def shape(self):
        return super().shape

    @shape.setter
    def shape(self, shape):
        super(MaskedArray, type(self)).shape.__set__(self, shape)
        # Cannot use self._mask, since it may not (yet) exist when a
        # masked matrix sets the shape.
        if getmask(self) is not nomask:
            self._mask.shape = self.shape

    def __setmask__(self, mask, copy=False):
        """
        Set the mask.

        """
        idtype = self.dtype
        current_mask = self._mask
        if mask is masked:
            mask = True

        if current_mask is nomask:
            # Make sure the mask is set
            # Just don't do anything if there's nothing to do.
            if mask is nomask:
                return
            current_mask = self._mask = make_mask_none(self.shape, idtype)

        if idtype.names is None:
            # No named fields.
            # Hardmask: don't unmask the data
            if self._hardmask:
                current_mask |= mask
            # Softmask: set everything to False
            # If it's obviously a compatible scalar, use a quick update
            # method.
            elif isinstance(mask, (int, float, np.bool, np.number)):
                current_mask[...] = mask
            # Otherwise fall back to the slower, general purpose way.
            else:
                current_mask.flat = mask
        else:
            # Named fields w/
            mdtype = current_mask.dtype
            mask = np.asarray(mask)
            # Mask is a singleton
            if not mask.ndim:
                # It's a boolean : make a record
                if mask.dtype.kind == 'b':
                    mask = np.array(tuple([mask.item()] * len(mdtype)),
                                    dtype=mdtype)
                # It's a record: make sure the dtype is correct
                else:
                    mask = mask.astype(mdtype)
            # Mask is a sequence
            else:
                # Make sure the new mask is a ndarray with the proper dtype
                try:
                    copy = None if not copy else True
                    mask = np.array(mask, copy=copy, dtype=mdtype)
                # Or assume it's a sequence of bool/int
                except TypeError:
                    mask = np.array([tuple([m] * len(mdtype)) for m in mask],
                                    dtype=mdtype)
            # Hardmask: don't unmask the data
            if self._hardmask:
                for n in idtype.names:
                    current_mask[n] |= mask[n]
            # Softmask: set everything to False
            # If it's obviously a compatible scalar, use a quick update
            # method.
            elif isinstance(mask, (int, float, np.bool, np.number)):
                current_mask[...] = mask
            # Otherwise fall back to the slower, general purpose way.
            else:
                current_mask.flat = mask
        # Reshape if needed
        if current_mask.shape:
            current_mask.shape = self.shape
        return

    _set_mask = __setmask__

    @property
    def mask(self):
        """ Current mask. """

        # We could try to force a reshape, but that wouldn't work in some
        # cases.
        # Return a view so that the dtype and shape cannot be changed in place
        # This still preserves nomask by identity
        return self._mask.view()

    @mask.setter
    def mask(self, value):
        self.__setmask__(value)

    @property
    def recordmask(self):
        """
        Get or set the mask of the array if it has no named fields. For
        structured arrays, returns a ndarray of booleans where entries are
        ``True`` if **all** the fields are masked, ``False`` otherwise:

        >>> x = np.ma.array([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],
        ...         mask=[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)],
        ...        dtype=[('a', int), ('b', int)])
        >>> x.recordmask
        array([False, False,  True, False, False])
        """

        _mask = self._mask.view(ndarray)
        if _mask.dtype.names is None:
            return _mask
        return np.all(flatten_structured_array(_mask), axis=-1)

    @recordmask.setter
    def recordmask(self, mask):
        raise NotImplementedError("Coming soon: setting the mask per records!")

    def harden_mask(self):
        """
        Force the mask to hard, preventing unmasking by assignment.

        Whether the mask of a masked array is hard or soft is determined by
        its `~ma.MaskedArray.hardmask` property. `harden_mask` sets
        `~ma.MaskedArray.hardmask` to ``True`` (and returns the modified
        self).

        See Also
        --------
        ma.MaskedArray.hardmask
        ma.MaskedArray.soften_mask

        """
        self._hardmask = True
        return self

    def soften_mask(self):
        """
        Force the mask to soft (default), allowing unmasking by assignment.

        Whether the mask of a masked array is hard or soft is determined by
        its `~ma.MaskedArray.hardmask` property. `soften_mask` sets
        `~ma.MaskedArray.hardmask` to ``False`` (and returns the modified
        self).

        See Also
        --------
        ma.MaskedArray.hardmask
        ma.MaskedArray.harden_mask

        """
        self._hardmask = False
        return self

    @property
    def hardmask(self):
        """
        Specifies whether values can be unmasked through assignments.

        By default, assigning definite values to masked array entries will
        unmask them.  When `hardmask` is ``True``, the mask will not change
        through assignments.

        See Also
        --------
        ma.MaskedArray.harden_mask
        ma.MaskedArray.soften_mask

        Examples
        --------
        >>> import numpy as np
        >>> x = np.arange(10)
        >>> m = np.ma.masked_array(x, x>5)
        >>> assert not m.hardmask

        Since `m` has a soft mask, assigning an element value unmasks that
        element:

        >>> m[8] = 42
        >>> m
        masked_array(data=[0, 1, 2, 3, 4, 5, --, --, 42, --],
                     mask=[False, False, False, False, False, False,
                           True, True, False, True],
               fill_value=999999)

        After hardening, the mask is not affected by assignments:

        >>> hardened = np.ma.harden_mask(m)
        >>> assert m.hardmask and hardened is m
        >>> m[:] = 23
        >>> m
        masked_array(data=[23, 23, 23, 23, 23, 23, --, --, 23, --],
                     mask=[False, False, False, False, False, False,
                           True, True, False, True],
               fill_value=999999)

        """
        return self._hardmask

    def unshare_mask(self):
        """
        Copy the mask and set the `sharedmask` flag to ``False``.

        Whether the mask is shared between masked arrays can be seen from
        the `sharedmask` property. `unshare_mask` ensures the mask is not
        shared. A copy of the mask is only made if it was shared.

        See Also
        --------
        sharedmask

        """
        if self._sharedmask:
            self._mask = self._mask.copy()
            self._sharedmask = False
        return self

    @property
    def sharedmask(self):
        """ Share status of the mask (read-only). """
        return self._sharedmask

    def shrink_mask(self):
        """
        Reduce a mask to nomask when possible.

        Parameters
        ----------
        None

        Returns
        -------
        result : MaskedArray
            A :class:`~ma.MaskedArray` object.

        Examples
        --------
        >>> import numpy as np
        >>> x = np.ma.array([[1,2 ], [3, 4]], mask=[0]*4)
        >>> x.mask
        array([[False, False],
               [False, False]])
        >>> x.shrink_mask()
        masked_array(
          data=[[1, 2],
                [3, 4]],
          mask=False,
          fill_value=999999)
        >>> x.mask
        False

        """
        self._mask = _shrink_mask(self._mask)
        return self

    @property
    def baseclass(self):
        """ Class of the underlying data (read-only). """
        return self._baseclass

    def _get_data(self):
        """
        Returns the underlying data, as a view of the masked array.

        If the underlying data is a subclass of :class:`numpy.ndarray`, it is
        returned as such.

        >>> x = np.ma.array(np.matrix([[1, 2], [3, 4]]), mask=[[0, 1], [1, 0]])
        >>> x.data
        matrix([[1, 2],
                [3, 4]])

        The type of the data can be accessed through the :attr:`baseclass`
        attribute.
        """
        return ndarray.view(self, self._baseclass)

    _data = property(fget=_get_data)
    data = property(fget=_get_data)

    @property
    def flat(self):
        """ Return a flat iterator, or set a flattened version of self to value. """
        return MaskedIterator(self)

    @flat.setter
    def flat(self, value):
        y = self.ravel()
        y[:] = value

    @property
    def fill_value(self):
        """
        The filling value of the masked array is a scalar. When setting, None
        will set to a default based on the data type.

        Examples
        --------
        >>> import numpy as np
        >>> for dt in [np.int32, np.int64, np.float64, np.complex128]:
        ...     np.ma.array([0, 1], dtype=dt).get_fill_value()
        ...
        np.int64(999999)
        np.int64(999999)
        np.float64(1e+20)
        np.complex128(1e+20+0j)

        >>> x = np.ma.array([0, 1.], fill_value=-np.inf)
        >>> x.fill_value
        np.float64(-inf)
        >>> x.fill_value = np.pi
        >>> x.fill_value
        np.float64(3.1415926535897931)

        Reset to default:

        >>> x.fill_value = None
        >>> x.fill_value
        np.float64(1e+20)

        """
        if self._fill_value is None:
            self._fill_value = _check_fill_value(None, self.dtype)

        # Temporary workaround to account for the fact that str and bytes
        # scalars cannot be indexed with (), whereas all other numpy
        # scalars can. See issues #7259 and #7267.
        # The if-block can be removed after #7267 has been fixed.
        if isinstance(self._fill_value, ndarray):
            return self._fill_value[()]
        return self._fill_value

    @fill_value.setter
    def fill_value(self, value=None):
        target = _check_fill_value(value, self.dtype)
        if not target.ndim == 0:
            # 2019-11-12, 1.18.0
            warnings.warn(
                "Non-scalar arrays for the fill value are deprecated. Use "
                "arrays with scalar values instead. The filled function "
                "still supports any array as `fill_value`.",
                DeprecationWarning, stacklevel=2)

        _fill_value = self._fill_value
        if _fill_value is None:
            # Create the attribute if it was undefined
            self._fill_value = target
        else:
            # Don't overwrite the attribute, just fill it (for propagation)
            _fill_value[()] = target

    # kept for compatibility
    get_fill_value = fill_value.fget
    set_fill_value = fill_value.fset

    def filled(self, fill_value=None):
        """
        Return a copy of self, with masked values filled with a given value.
        **However**, if there are no masked values to fill, self will be
        returned instead as an ndarray.

        Parameters
        ----------
        fill_value : array_like, optional
            The value to use for invalid entries. Can be scalar or non-scalar.
            If non-scalar, the resulting ndarray must be broadcastable over
            input array. Default is None, in which case, the `fill_value`
            attribute of the array is used instead.

        Returns
        -------
        filled_array : ndarray
            A copy of ``self`` with invalid entries replaced by *fill_value*
            (be it the function argument or the attribute of ``self``), or
            ``self`` itself as an ndarray if there are no invalid entries to
            be replaced.

        Notes
        -----
        The result is **not** a MaskedArray!

        Examples
        --------
        >>> import numpy as np
        >>> x = np.ma.array([1,2,3,4,5], mask=[0,0,1,0,1], fill_value=-999)
        >>> x.filled()
        array([   1,    2, -999,    4, -999])
        >>> x.filled(fill_value=1000)
        array([   1,    2, 1000,    4, 1000])
        >>> type(x.filled())
        <class 'numpy.ndarray'>

        Subclassing is preserved. This means that if, e.g., the data part of
        the masked array is a recarray, `filled` returns a recarray:

        >>> x = np.array([(-1, 2), (-3, 4)], dtype='i8,i8').view(np.recarray)
        >>> m = np.ma.array(x, mask=[(True, False), (False, True)])
        >>> m.filled()
        rec.array([(999999,      2), (    -3, 999999)],
                  dtype=[('f0', '<i8'), ('f1', '<i8')])
        """
        m = self._mask
        if m is nomask:
            return self._data

        if fill_value is None:
            fill_value = self.fill_value
        else:
            fill_value = _check_fill_value(fill_value, self.dtype)

        if self is masked_singleton:
            return np.asanyarray(fill_value)

        if m.dtype.names is not None:
            result = self._data.copy('K')
            _recursive_filled(result, self._mask, fill_value)
        elif not m.any():
            return self._data
        else:
            result = self._data.copy('K')
            try:
                np.copyto(result, fill_value, where=m)
            except (TypeError, AttributeError):
                fill_value = narray(fill_value, dtype=object)
                d = result.astype(object)
                result = np.choose(m, (d, fill_value))
            except IndexError:
                # ok, if scalar
                if self._data.shape:
                    raise
                elif m:
                    result = np.array(fill_value, dtype=self.dtype)
                else:
                    result = self._data
        return result

    def compressed(self):
        """
        Return all the non-masked data as a 1-D array.

        Returns
        -------
        data : ndarray
            A new `ndarray` holding the non-masked data is returned.

        Notes
        -----
        The result is **not** a MaskedArray!

        Examples
        --------
        >>> import numpy as np
        >>> x = np.ma.array(np.arange(5), mask=[0]*2 + [1]*3)
        >>> x.compressed()
        array([0, 1])
        >>> type(x.compressed())
        <class 'numpy.ndarray'>

        N-D arrays are compressed to 1-D.

        >>> arr = [[1, 2], [3, 4]]
        >>> mask = [[1, 0], [0, 1]]
        >>> x = np.ma.array(arr, mask=mask)
        >>> x.compressed()
        array([2, 3])

        """
        data = ndarray.ravel(self._data)
        if self._mask is not nomask:
            data = data.compress(np.logical_not(ndarray.ravel(self._mask)))
        return data

    def compress(self, condition, axis=None, out=None):
        """
        Return `a` where condition is ``True``.

        If condition is a `~ma.MaskedArray`, missing values are considered
        as ``False``.

        Parameters
        ----------
        condition : var
            Boolean 1-d array selecting which entries to return. If len(condition)
            is less than the size of a along the axis, then output is truncated
            to length of condition array.
        axis : {None, int}, optional
            Axis along which the operation must be performed.
        out : {None, ndarray}, optional
            Alternative output array in which to place the result. It must have
            the same shape as the expected output but the type will be cast if
            necessary.

        Returns
        -------
        result : MaskedArray
            A :class:`~ma.MaskedArray` object.

        Notes
        -----
        Please note the difference with :meth:`compressed` !
        The output of :meth:`compress` has a mask, the output of
        :meth:`compressed` does not.

        Examples
        --------
        >>> import numpy as np
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> x
        masked_array(
          data=[[1, --, 3],
                [--, 5, --],
                [7, --, 9]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
        >>> x.compress([1, 0, 1])
        masked_array(data=[1, 3],
                     mask=[False, False],
               fill_value=999999)

        >>> x.compress([1, 0, 1], axis=1)
        masked_array(
          data=[[1, 3],
                [--, --],
                [7, 9]],
          mask=[[False, False],
                [ True,  True],
                [False, False]],
          fill_value=999999)

        """
        # Get the basic components
        (_data, _mask) = (self._data, self._mask)

        # Force the condition to a regular ndarray and forget the missing
        # values.
        condition = np.asarray(condition)

        _new = _data.compress(condition, axis=axis, out=out).view(type(self))
        _new._update_from(self)
        if _mask is not nomask:
            _new._mask = _mask.compress(condition, axis=axis)
        return _new

    def _insert_masked_print(self):
        """
        Replace masked values with masked_print_option, casting all innermost
        dtypes to object.
        """
        if masked_print_option.enabled():
            mask = self._mask
            if mask is nomask:
                res = self._data
            else:
                # convert to object array to make filled work
                data = self._data
                # For big arrays, to avoid a costly conversion to the
                # object dtype, extract the corners before the conversion.
                print_width = (self._print_width if self.ndim > 1
                               else self._print_width_1d)
                for axis in range(self.ndim):
                    if data.shape[axis] > print_width:
                        ind = print_width // 2
                        arr = np.split(data, (ind, -ind), axis=axis)
                        data = np.concatenate((arr[0], arr[2]), axis=axis)
                        arr = np.split(mask, (ind, -ind), axis=axis)
                        mask = np.concatenate((arr[0], arr[2]), axis=axis)

                rdtype = _replace_dtype_fields(self.dtype, "O")
                res = data.astype(rdtype)
                _recursive_printoption(res, mask, masked_print_option)
        else:
            res = self.filled(self.fill_value)
        return res

    def __str__(self):
        return str(self._insert_masked_print())

    def __repr__(self):
        """
        Literal string representation.

        """
        if self._baseclass is np.ndarray:
            name = 'array'
        else:
            name = self._baseclass.__name__

        # 2016-11-19: Demoted to legacy format
        if np._core.arrayprint._get_legacy_print_mode() <= 113:
            is_long = self.ndim > 1
            parameters = {
                'name': name,
                'nlen': " " * len(name),
                'data': str(self),
                'mask': str(self._mask),
                'fill': str(self.fill_value),
                'dtype': str(self.dtype)
            }
            is_structured = bool(self.dtype.names)
            key = '{}_{}'.format(
                'long' if is_long else 'short',
                'flx' if is_structured else 'std'
            )
            return _legacy_print_templates[key] % parameters

        prefix = f"masked_{name}("

        dtype_needed = (
            not np._core.arrayprint.dtype_is_implied(self.dtype) or
            np.all(self.mask) or
            self.size == 0
        )

        # determine which keyword args need to be shown
        keys = ['data', 'mask', 'fill_value']
        if dtype_needed:
            keys.append('dtype')

        # array has only one row (non-column)
        is_one_row = builtins.all(dim == 1 for dim in self.shape[:-1])

        # choose what to indent each keyword with
        min_indent = 2
        if is_one_row:
            # first key on the same line as the type, remaining keys
            # aligned by equals
            indents = {}
            indents[keys[0]] = prefix
            for k in keys[1:]:
                n = builtins.max(min_indent, len(prefix + keys[0]) - len(k))
                indents[k] = ' ' * n
            prefix = ''  # absorbed into the first indent
        else:
            # each key on its own line, indented by two spaces
            indents = dict.fromkeys(keys, ' ' * min_indent)
            prefix = prefix + '\n'  # first key on the next line

        # format the field values
        reprs = {}
        reprs['data'] = np.array2string(
            self._insert_masked_print(),
            separator=", ",
            prefix=indents['data'] + 'data=',
            suffix=',')
        reprs['mask'] = np.array2string(
            self._mask,
            separator=", ",
            prefix=indents['mask'] + 'mask=',
            suffix=',')

        if self._fill_value is None:
            self.fill_value  # initialize fill_value  # noqa: B018

        if (self._fill_value.dtype.kind in ("S", "U")
                and self.dtype.kind == self._fill_value.dtype.kind):
            # Allow strings: "N/A" has length 3 so would mismatch.
            fill_repr = repr(self.fill_value.item())
        elif self._fill_value.dtype == self.dtype and not self.dtype == object:
            # Guess that it is OK to use the string as item repr.  To really
            # fix this, it needs new logic (shared with structured scalars)
            fill_repr = str(self.fill_value)
        else:
            fill_repr = repr(self.fill_value)

        reprs['fill_value'] = fill_repr
        if dtype_needed:
            reprs['dtype'] = np._core.arrayprint.dtype_short_repr(self.dtype)

        # join keys with values and indentations
        result = ',\n'.join(
            f'{indents[k]}{k}={reprs[k]}'
            for k in keys
        )
        return prefix + result + ')'

    def _delegate_binop(self, other):
        # This emulates the logic in
        #     private/binop_override.h:forward_binop_should_defer
        if isinstance(other, type(self)):
            return False
        array_ufunc = getattr(other, "__array_ufunc__", False)
        if array_ufunc is False:
            other_priority = getattr(other, "__array_priority__", -1000000)
            return self.__array_priority__ < other_priority
        else:
            # If array_ufunc is not None, it will be called inside the ufunc;
            # None explicitly tells us to not call the ufunc, i.e., defer.
            return array_ufunc is None

    def _comparison(self, other, compare):
        """Compare self with other using operator.eq or operator.ne.

        When either of the elements is masked, the result is masked as well,
        but the underlying boolean data are still set, with self and other
        considered equal if both are masked, and unequal otherwise.

        For structured arrays, all fields are combined, with masked values
        ignored. The result is masked if all fields were masked, with self
        and other considered equal only if both were fully masked.
        """
        omask = getmask(other)
        smask = self.mask
        mask = mask_or(smask, omask, copy=True)

        odata = getdata(other)
        if mask.dtype.names is not None:
            # only == and != are reasonably defined for structured dtypes,
            # so give up early for all other comparisons:
            if compare not in (operator.eq, operator.ne):
                return NotImplemented
            # For possibly masked structured arrays we need to be careful,
            # since the standard structured array comparison will use all
            # fields, masked or not. To avoid masked fields influencing the
            # outcome, we set all masked fields in self to other, so they'll
            # count as equal.  To prepare, we ensure we have the right shape.
            broadcast_shape = np.broadcast(self, odata).shape
            sbroadcast = np.broadcast_to(self, broadcast_shape, subok=True)
            sbroadcast._mask = mask
            sdata = sbroadcast.filled(odata)
            # Now take care of the mask; the merged mask should have an item
            # masked if all fields were masked (in one and/or other).
            mask = (mask == np.ones((), mask.dtype))
            # Ensure we can compare masks below if other was not masked.
            if omask is np.False_:
                omask = np.zeros((), smask.dtype)

        else:
            # For regular arrays, just use the data as they come.
            sdata = self.data

        check = compare(sdata, odata)

        if isinstance(check, (np.bool, bool)):
            return masked if mask else check

        if mask is not nomask:
            if compare in (operator.eq, operator.ne):
                # Adjust elements that were masked, which should be treated
                # as equal if masked in both, unequal if masked in one.
                # Note that this works automatically for structured arrays too.
                # Ignore this for operations other than `==` and `!=`
                check = np.where(mask, compare(smask, omask), check)

            if mask.shape != check.shape:
                # Guarantee consistency of the shape, making a copy since the
                # the mask may need to get written to later.
                mask = np.broadcast_to(mask, check.shape).copy()

        check = check.view(type(self))
        check._update_from(self)
        check._mask = mask

        # Cast fill value to np.bool if needed. If it cannot be cast, the
        # default boolean fill value is used.
        if check._fill_value is not None:
            try:
                fill = _check_fill_value(check._fill_value, np.bool)
            except (TypeError, ValueError):
                fill = _check_fill_value(None, np.bool)
            check._fill_value = fill

        return check

    def __eq__(self, other):
        """Check whether other equals self elementwise.

        When either of the elements is masked, the result is masked as well,
        but the underlying boolean data are still set, with self and other
        considered equal if both are masked, and unequal otherwise.

        For structured arrays, all fields are combined, with masked values
        ignored. The result is masked if all fields were masked, with self
        and other considered equal only if both were fully masked.
        """
        return self._comparison(other, operator.eq)

    def __ne__(self, other):
        """Check whether other does not equal self elementwise.

        When either of the elements is masked, the result is masked as well,
        but the underlying boolean data are still set, with self and other
        considered equal if both are masked, and unequal otherwise.

        For structured arrays, all fields are combined, with masked values
        ignored. The result is masked if all fields were masked, with self
        and other considered equal only if both were fully masked.
        """
        return self._comparison(other, operator.ne)

    # All other comparisons:
    def __le__(self, other):
        return self._comparison(other, operator.le)

    def __lt__(self, other):
        return self._comparison(other, operator.lt)

    def __ge__(self, other):
        return self._comparison(other, operator.ge)

    def __gt__(self, other):
        return self._comparison(other, operator.gt)

    def __add__(self, other):
        """
        Add self to other, and return a new masked array.

        """
        if self._delegate_binop(other):
            return NotImplemented
        return add(self, other)

    def __radd__(self, other):
        """
        Add other to self, and return a new masked array.

        """
        # In analogy with __rsub__ and __rdiv__, use original order:
        # we get here from `other + self`.
        return add(other, self)

    def __sub__(self, other):
        """
        Subtract other from self, and return a new masked array.

        """
        if self._delegate_binop(other):
            return NotImplemented
        return subtract(self, other)

    def __rsub__(self, other):
        """
        Subtract self from other, and return a new masked array.

        """
        return subtract(other, self)

    def __mul__(self, other):
        "Multiply self by other, and return a new masked array."
        if self._delegate_binop(other):
            return NotImplemented
        return multiply(self, other)

    def __rmul__(self, other):
        """
        Multiply other by self, and return a new masked array.

        """
        # In analogy with __rsub__ and __rdiv__, use original order:
        # we get here from `other * self`.
        return multiply(other, self)

    def __truediv__(self, other):
        """
        Divide other into self, and return a new masked array.

        """
        if self._delegate_binop(other):
            return NotImplemented
        return true_divide(self, other)

    def __rtruediv__(self, other):
        """
        Divide self into other, and return a new masked array.

        """
        return true_divide(other, self)

    def __floordiv__(self, other):
        """
        Divide other into self, and return a new masked array.

        """
        if self._delegate_binop(other):
            return NotImplemented
        return floor_divide(self, other)

    def __rfloordiv__(self, other):
        """
        Divide self into other, and return a new masked array.

        """
        return floor_divide(other, self)

    def __pow__(self, other):
        """
        Raise self to the power other, masking the potential NaNs/Infs

        """
        if self._delegate_binop(other):
            return NotImplemented
        return power(self, other)

    def __rpow__(self, other):
        """
        Raise other to the power self, masking the potential NaNs/Infs

        """
        return power(other, self)

    def __iadd__(self, other):
        """
        Add other to self in-place.

        """
        m = getmask(other)
        if self._mask is nomask:
            if m is not nomask and m.any():
                self._mask = make_mask_none(self.shape, self.dtype)
                self._mask += m
        elif m is not nomask:
            self._mask += m
        other_data = getdata(other)
        other_data = np.where(self._mask, other_data.dtype.type(0), other_data)
        self._data.__iadd__(other_data)
        return self

    def __isub__(self, other):
        """
        Subtract other from self in-place.

        """
        m = getmask(other)
        if self._mask is nomask:
            if m is not nomask and m.any():
                self._mask = make_mask_none(self.shape, self.dtype)
                self._mask += m
        elif m is not nomask:
            self._mask += m
        other_data = getdata(other)
        other_data = np.where(self._mask, other_data.dtype.type(0), other_data)
        self._data.__isub__(other_data)
        return self

    def __imul__(self, other):
        """
        Multiply self by other in-place.

        """
        m = getmask(other)
        if self._mask is nomask:
            if m is not nomask and m.any():
                self._mask = make_mask_none(self.shape, self.dtype)
                self._mask += m
        elif m is not nomask:
            self._mask += m
        other_data = getdata(other)
        other_data = np.where(self._mask, other_data.dtype.type(1), other_data)
        self._data.__imul__(other_data)
        return self

    def __ifloordiv__(self, other):
        """
        Floor divide self by other in-place.

        """
        other_data = getdata(other)
        dom_mask = _DomainSafeDivide().__call__(self._data, other_data)
        other_mask = getmask(other)
        new_mask = mask_or(other_mask, dom_mask)
        # The following 3 lines control the domain filling
        if dom_mask.any():
            (_, fval) = ufunc_fills[np.floor_divide]
            other_data = np.where(
                    dom_mask, other_data.dtype.type(fval), other_data)
        self._mask |= new_mask
        other_data = np.where(self._mask, other_data.dtype.type(1), other_data)
        self._data.__ifloordiv__(other_data)
        return self

    def __itruediv__(self, other):
        """
        True divide self by other in-place.

        """
        other_data = getdata(other)
        dom_mask = _DomainSafeDivide().__call__(self._data, other_data)
        other_mask = getmask(other)
        new_mask = mask_or(other_mask, dom_mask)
        # The following 3 lines control the domain filling
        if dom_mask.any():
            (_, fval) = ufunc_fills[np.true_divide]
            other_data = np.where(
                    dom_mask, other_data.dtype.type(fval), other_data)
        self._mask |= new_mask
        other_data = np.where(self._mask, other_data.dtype.type(1), other_data)
        self._data.__itruediv__(other_data)
        return self

    def __ipow__(self, other):
        """
        Raise self to the power other, in place.

        """
        other_data = getdata(other)
        other_data = np.where(self._mask, other_data.dtype.type(1), other_data)
        other_mask = getmask(other)
        with np.errstate(divide='ignore', invalid='ignore'):
            self._data.__ipow__(other_data)
        invalid = np.logical_not(np.isfinite(self._data))
        if invalid.any():
            if self._mask is not nomask:
                self._mask |= invalid
            else:
                self._mask = invalid
            np.copyto(self._data, self.fill_value, where=invalid)
        new_mask = mask_or(other_mask, invalid)
        self._mask = mask_or(self._mask, new_mask)
        return self

    def __float__(self):
        """
        Convert to float.

        """
        if self.size > 1:
            raise TypeError("Only length-1 arrays can be converted "
                            "to Python scalars")
        elif self._mask:
            warnings.warn("Warning: converting a masked element to nan.", stacklevel=2)
            return np.nan
        return float(self.item())

    def __int__(self):
        """
        Convert to int.

        """
        if self.size > 1:
            raise TypeError("Only length-1 arrays can be converted "
                            "to Python scalars")
        elif self._mask:
            raise MaskError('Cannot convert masked element to a Python int.')
        return int(self.item())

    @property
    def imag(self):
        """
        The imaginary part of the masked array.

        This property is a view on the imaginary part of this `MaskedArray`.

        See Also
        --------
        real

        Examples
        --------
        >>> import numpy as np
        >>> x = np.ma.array([1+1.j, -2j, 3.45+1.6j], mask=[False, True, False])
        >>> x.imag
        masked_array(data=[1.0, --, 1.6],
                     mask=[False,  True, False],
               fill_value=1e+20)

        """
        result = self._data.imag.view(type(self))
        result.__setmask__(self._mask)
        return result

    # kept for compatibility
    get_imag = imag.fget

    @property
    def real(self):
        """
        The real part of the masked array.

        This property is a view on the real part of this `MaskedArray`.

        See Also
        --------
        imag

        Examples
        --------
        >>> import numpy as np
        >>> x = np.ma.array([1+1.j, -2j, 3.45+1.6j], mask=[False, True, False])
        >>> x.real
        masked_array(data=[1.0, --, 3.45],
                     mask=[False,  True, False],
               fill_value=1e+20)

        """
        result = self._data.real.view(type(self))
        result.__setmask__(self._mask)
        return result

    # kept for compatibility
    get_real = real.fget

    def count(self, axis=None, keepdims=np._NoValue):
        """
        Count the non-masked elements of the array along the given axis.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes along which the count is performed.
            The default, None, performs the count over all
            the dimensions of the input array. `axis` may be negative, in
            which case it counts from the last to the first axis.
            If this is a tuple of ints, the count is performed on multiple
            axes, instead of a single axis or all the axes as before.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the array.

        Returns
        -------
        result : ndarray or scalar
            An array with the same shape as the input array, with the specified
            axis removed. If the array is a 0-d array, or if `axis` is None, a
            scalar is returned.

        See Also
        --------
        ma.count_masked : Count masked elements in array or along a given axis.

        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = ma.arange(6).reshape((2, 3))
        >>> a[1, :] = ma.masked
        >>> a
        masked_array(
          data=[[0, 1, 2],
                [--, --, --]],
          mask=[[False, False, False],
                [ True,  True,  True]],
          fill_value=999999)
        >>> a.count()
        3

        When the `axis` keyword is specified an array of appropriate size is
        returned.

        >>> a.count(axis=0)
        array([1, 1, 1])
        >>> a.count(axis=1)
        array([3, 0])

        """
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}

        m = self._mask
        # special case for matrices (we assume no other subclasses modify
        # their dimensions)
        if isinstance(self.data, np.matrix):
            if m is nomask:
                m = np.zeros(self.shape, dtype=np.bool)
            m = m.view(type(self.data))

        if m is nomask:
            # compare to _count_reduce_items in _methods.py

            if self.shape == ():
                if axis not in (None, 0):
                    raise np.exceptions.AxisError(axis=axis, ndim=self.ndim)
                return 1
            elif axis is None:
                if kwargs.get('keepdims'):
                    return np.array(self.size, dtype=np.intp, ndmin=self.ndim)
                return self.size

            axes = normalize_axis_tuple(axis, self.ndim)
            items = 1
            for ax in axes:
                items *= self.shape[ax]

            if kwargs.get('keepdims'):
                out_dims = list(self.shape)
                for a in axes:
                    out_dims[a] = 1
            else:
                out_dims = [d for n, d in enumerate(self.shape)
                            if n not in axes]
            # make sure to return a 0-d array if axis is supplied
            return np.full(out_dims, items, dtype=np.intp)

        # take care of the masked singleton
        if self is masked:
            return 0

        return (~m).sum(axis=axis, dtype=np.intp, **kwargs)

    def ravel(self, order='C'):
        """
        Returns a 1D version of self, as a view.

        Parameters
        ----------
        order : {'C', 'F', 'A', 'K'}, optional
            The elements of `a` are read using this index order. 'C' means to
            index the elements in C-like order, with the last axis index
            changing fastest, back to the first axis index changing slowest.
            'F' means to index the elements in Fortran-like index order, with
            the first index changing fastest, and the last index changing
            slowest. Note that the 'C' and 'F' options take no account of the
            memory layout of the underlying array, and only refer to the order
            of axis indexing.  'A' means to read the elements in Fortran-like
            index order if `m` is Fortran *contiguous* in memory, C-like order
            otherwise.  'K' means to read the elements in the order they occur
            in memory, except for reversing the data when strides are negative.
            By default, 'C' index order is used.
            (Masked arrays currently use 'A' on the data when 'K' is passed.)

        Returns
        -------
        MaskedArray
            Output view is of shape ``(self.size,)`` (or
            ``(np.ma.product(self.shape),)``).

        Examples
        --------
        >>> import numpy as np
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> x
        masked_array(
          data=[[1, --, 3],
                [--, 5, --],
                [7, --, 9]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
        >>> x.ravel()
        masked_array(data=[1, --, 3, --, 5, --, 7, --, 9],
                     mask=[False,  True, False,  True, False,  True, False,  True,
                           False],
               fill_value=999999)

        """
        # The order of _data and _mask could be different (it shouldn't be
        # normally).  Passing order `K` or `A` would be incorrect.
        # So we ignore the mask memory order.
        # TODO: We don't actually support K, so use A instead.  We could
        #       try to guess this correct by sorting strides or deprecate.
        if order in "kKaA":
            order = "F" if self._data.flags.fnc else "C"
        r = ndarray.ravel(self._data, order=order).view(type(self))
        r._update_from(self)
        if self._mask is not nomask:
            r._mask = ndarray.ravel(self._mask, order=order).reshape(r.shape)
        else:
            r._mask = nomask
        return r

    def reshape(self, *s, **kwargs):
        """
        Give a new shape to the array without changing its data.

        Returns a masked array containing the same data, but with a new shape.
        The result is a view on the original array; if this is not possible, a
        ValueError is raised.

        Parameters
        ----------
        shape : int or tuple of ints
            The new shape should be compatible with the original shape. If an
            integer is supplied, then the result will be a 1-D array of that
            length.
        order : {'C', 'F'}, optional
            Determines whether the array data should be viewed as in C
            (row-major) or FORTRAN (column-major) order.

        Returns
        -------
        reshaped_array : array
            A new view on the array.

        See Also
        --------
        reshape : Equivalent function in the masked array module.
        numpy.ndarray.reshape : Equivalent method on ndarray object.
        numpy.reshape : Equivalent function in the NumPy module.

        Notes
        -----
        The reshaping operation cannot guarantee that a copy will not be made,
        to modify the shape in place, use ``a.shape = s``

        Examples
        --------
        >>> import numpy as np
        >>> x = np.ma.array([[1,2],[3,4]], mask=[1,0,0,1])
        >>> x
        masked_array(
          data=[[--, 2],
                [3, --]],
          mask=[[ True, False],
                [False,  True]],
          fill_value=999999)
        >>> x = x.reshape((4,1))
        >>> x
        masked_array(
          data=[[--],
                [2],
                [3],
                [--]],
          mask=[[ True],
                [False],
                [False],
                [ True]],
          fill_value=999999)

        """
        kwargs.update(order=kwargs.get('order', 'C'))
        result = self._data.reshape(*s, **kwargs).view(type(self))
        result._update_from(self)
        mask = self._mask
        if mask is not nomask:
            result._mask = mask.reshape(*s, **kwargs)
        return result

    def resize(self, newshape, refcheck=True, order=False):
        """
        .. warning::

            This method does nothing, except raise a ValueError exception. A
            masked array does not own its data and therefore cannot safely be
            resized in place. Use the `numpy.ma.resize` function instead.

        This method is difficult to implement safely and may be deprecated in
        future releases of NumPy.

        """
        # Note : the 'order' keyword looks broken, let's just drop it
        errmsg = "A masked array does not own its data "\
                 "and therefore cannot be resized.\n" \
                 "Use the numpy.ma.resize function instead."
        raise ValueError(errmsg)

    def put(self, indices, values, mode='raise'):
        """
        Set storage-indexed locations to corresponding values.

        Sets self._data.flat[n] = values[n] for each n in indices.
        If `values` is shorter than `indices` then it will repeat.
        If `values` has some masked values, the initial mask is updated
        in consequence, else the corresponding values are unmasked.

        Parameters
        ----------
        indices : 1-D array_like
            Target indices, interpreted as integers.
        values : array_like
            Values to place in self._data copy at target indices.
        mode : {'raise', 'wrap', 'clip'}, optional
            Specifies how out-of-bounds indices will behave.
            'raise' : raise an error.
            'wrap' : wrap around.
            'clip' : clip to the range.

        Notes
        -----
        `values` can be a scalar or length 1 array.

        Examples
        --------
        >>> import numpy as np
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> x
        masked_array(
          data=[[1, --, 3],
                [--, 5, --],
                [7, --, 9]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
        >>> x.put([0,4,8],[10,20,30])
        >>> x
        masked_array(
          data=[[10, --, 3],
                [--, 20, --],
                [7, --, 30]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)

        >>> x.put(4,999)
        >>> x
        masked_array(
          data=[[10, --, 3],
                [--, 999, --],
                [7, --, 30]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)

        """
        # Hard mask: Get rid of the values/indices that fall on masked data
        if self._hardmask and self._mask is not nomask:
            mask = self._mask[indices]
            indices = narray(indices, copy=None)
            values = narray(values, copy=None, subok=True)
            values.resize(indices.shape)
            indices = indices[~mask]
            values = values[~mask]

        self._data.put(indices, values, mode=mode)

        # short circuit if neither self nor values are masked
        if self._mask is nomask and getmask(values) is nomask:
            return

        m = getmaskarray(self)

        if getmask(values) is nomask:
            m.put(indices, False, mode=mode)
        else:
            m.put(indices, values._mask, mode=mode)
        m = make_mask(m, copy=False, shrink=True)
        self._mask = m
        return

    def ids(self):
        """
        Return the addresses of the data and mask areas.

        Parameters
        ----------
        None

        Examples
        --------
        >>> import numpy as np
        >>> x = np.ma.array([1, 2, 3], mask=[0, 1, 1])
        >>> x.ids()
        (166670640, 166659832) # may vary

        If the array has no mask, the address of `nomask` is returned. This address
        is typically not close to the data in memory:

        >>> x = np.ma.array([1, 2, 3])
        >>> x.ids()
        (166691080, 3083169284) # may vary

        """
        if self._mask is nomask:
            return (self.ctypes.data, id(nomask))
        return (self.ctypes.data, self._mask.ctypes.data)

    def iscontiguous(self):
        """
        Return a boolean indicating whether the data is contiguous.

        Parameters
        ----------
        None

        Examples
        --------
        >>> import numpy as np
        >>> x = np.ma.array([1, 2, 3])
        >>> x.iscontiguous()
        True

        `iscontiguous` returns one of the flags of the masked array:

        >>> x.flags
          C_CONTIGUOUS : True
          F_CONTIGUOUS : True
          OWNDATA : False
          WRITEABLE : True
          ALIGNED : True
          WRITEBACKIFCOPY : False

        """
        return self.flags['CONTIGUOUS']

    def all(self, axis=None, out=None, keepdims=np._NoValue):
        """
        Returns True if all elements evaluate to True.

        The output array is masked where all the values along the given axis
        are masked: if the output would have been a scalar and that all the
        values are masked, then the output is `masked`.

        Refer to `numpy.all` for full documentation.

        See Also
        --------
        numpy.ndarray.all : corresponding function for ndarrays
        numpy.all : equivalent function

        Examples
        --------
        >>> import numpy as np
        >>> np.ma.array([1,2,3]).all()
        True
        >>> a = np.ma.array([1,2,3], mask=True)
        >>> (a.all() is np.ma.masked)
        True

        """
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}

        mask = _check_mask_axis(self._mask, axis, **kwargs)
        if out is None:
            d = self.filled(True).all(axis=axis, **kwargs).view(type(self))
            if d.ndim:
                d.__setmask__(mask)
            elif mask:
                return masked
            return d
        self.filled(True).all(axis=axis, out=out, **kwargs)
        if isinstance(out, MaskedArray):
            if out.ndim or mask:
                out.__setmask__(mask)
        return out

    def any(self, axis=None, out=None, keepdims=np._NoValue):
        """
        Returns True if any of the elements of `a` evaluate to True.

        Masked values are considered as False during computation.

        Refer to `numpy.any` for full documentation.

        See Also
        --------
        numpy.ndarray.any : corresponding function for ndarrays
        numpy.any : equivalent function

        """
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}

        mask = _check_mask_axis(self._mask, axis, **kwargs)
        if out is None:
            d = self.filled(False).any(axis=axis, **kwargs).view(type(self))
            if d.ndim:
                d.__setmask__(mask)
            elif mask:
                d = masked
            return d
        self.filled(False).any(axis=axis, out=out, **kwargs)
        if isinstance(out, MaskedArray):
            if out.ndim or mask:
                out.__setmask__(mask)
        return out

    def nonzero(self):
        """
        Return the indices of unmasked elements that are not zero.

        Returns a tuple of arrays, one for each dimension, containing the
        indices of the non-zero elements in that dimension. The corresponding
        non-zero values can be obtained with::

            a[a.nonzero()]

        To group the indices by element, rather than dimension, use
        instead::

            np.transpose(a.nonzero())

        The result of this is always a 2d array, with a row for each non-zero
        element.

        Parameters
        ----------
        None

        Returns
        -------
        tuple_of_arrays : tuple
            Indices of elements that are non-zero.

        See Also
        --------
        numpy.nonzero :
            Function operating on ndarrays.
        flatnonzero :
            Return indices that are non-zero in the flattened version of the input
            array.
        numpy.ndarray.nonzero :
            Equivalent ndarray method.
        count_nonzero :
            Counts the number of non-zero elements in the input array.

        Examples
        --------
        >>> import numpy as np
        >>> import numpy.ma as ma
        >>> x = ma.array(np.eye(3))
        >>> x
        masked_array(
          data=[[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]],
          mask=False,
          fill_value=1e+20)
        >>> x.nonzero()
        (array([0, 1, 2]), array([0, 1, 2]))

        Masked elements are ignored.

        >>> x[1, 1] = ma.masked
        >>> x
        masked_array(
          data=[[1.0, 0.0, 0.0],
                [0.0, --, 0.0],
                [0.0, 0.0, 1.0]],
          mask=[[False, False, False],
                [False,  True, False],
                [False, False, False]],
          fill_value=1e+20)
        >>> x.nonzero()
        (array([0, 2]), array([0, 2]))

        Indices can also be grouped by element.

        >>> np.transpose(x.nonzero())
        array([[0, 0],
               [2, 2]])

        A common use for ``nonzero`` is to find the indices of an array, where
        a condition is True.  Given an array `a`, the condition `a` > 3 is a
        boolean array and since False is interpreted as 0, ma.nonzero(a > 3)
        yields the indices of the `a` where the condition is true.

        >>> a = ma.array([[1,2,3],[4,5,6],[7,8,9]])
        >>> a > 3
        masked_array(
          data=[[False, False, False],
                [ True,  True,  True],
                [ True,  True,  True]],
          mask=False,
          fill_value=True)
        >>> ma.nonzero(a > 3)
        (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

        The ``nonzero`` method of the condition array can also be called.

        >>> (a > 3).nonzero()
        (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

        """
        return np.asarray(self.filled(0)).nonzero()

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        """
        (this docstring should be overwritten)
        """
        # !!!: implement out + test!
        m = self._mask
        if m is nomask:
            result = super().trace(offset=offset, axis1=axis1, axis2=axis2,
                                   out=out)
            return result.astype(dtype)
        else:
            D = self.diagonal(offset=offset, axis1=axis1, axis2=axis2)
            return D.astype(dtype).filled(0).sum(axis=-1, out=out)
    trace.__doc__ = ndarray.trace.__doc__

    def dot(self, b, out=None, strict=False):
        """
        a.dot(b, out=None)

        Masked dot product of two arrays. Note that `out` and `strict` are
        located in different positions than in `ma.dot`. In order to
        maintain compatibility with the functional version, it is
        recommended that the optional arguments be treated as keyword only.
        At some point that may be mandatory.

        Parameters
        ----------
        b : masked_array_like
            Inputs array.
        out : masked_array, optional
            Output argument. This must have the exact kind that would be
            returned if it was not used. In particular, it must have the
            right type, must be C-contiguous, and its dtype must be the
            dtype that would be returned for `ma.dot(a,b)`. This is a
            performance feature. Therefore, if these conditions are not
            met, an exception is raised, instead of attempting to be
            flexible.
        strict : bool, optional
            Whether masked data are propagated (True) or set to 0 (False)
            for the computation. Default is False.  Propagating the mask
            means that if a masked value appears in a row or column, the
            whole row or column is considered masked.

        See Also
        --------
        numpy.ma.dot : equivalent function

        """
        return dot(self, b, out=out, strict=strict)

    def sum(self, axis=None, dtype=None, out=None, keepdims=np._NoValue):
        """
        Return the sum of the array elements over the given axis.

        Masked elements are set to 0 internally.

        Refer to `numpy.sum` for full documentation.

        See Also
        --------
        numpy.ndarray.sum : corresponding function for ndarrays
        numpy.sum : equivalent function

        Examples
        --------
        >>> import numpy as np
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> x
        masked_array(
          data=[[1, --, 3],
                [--, 5, --],
                [7, --, 9]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
        >>> x.sum()
        25
        >>> x.sum(axis=1)
        masked_array(data=[4, 5, 16],
                     mask=[False, False, False],
               fill_value=999999)
        >>> x.sum(axis=0)
        masked_array(data=[8, 5, 12],
                     mask=[False, False, False],
               fill_value=999999)
        >>> print(type(x.sum(axis=0, dtype=np.int64)[0]))
        <class 'numpy.int64'>

        """
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}

        _mask = self._mask
        newmask = _check_mask_axis(_mask, axis, **kwargs)
        # No explicit output
        if out is None:
            result = self.filled(0).sum(axis, dtype=dtype, **kwargs)
            rndim = getattr(result, 'ndim', 0)
            if rndim:
                result = result.view(type(self))
                result.__setmask__(newmask)
            elif newmask:
                result = masked
            return result
        # Explicit output
        result = self.filled(0).sum(axis, dtype=dtype, out=out, **kwargs)
        if isinstance(out, MaskedArray):
            outmask = getmask(out)
            if outmask is nomask:
                outmask = out._mask = make_mask_none(out.shape)
            outmask.flat = newmask
        return out

    def cumsum(self, axis=None, dtype=None, out=None):
        """
        Return the cumulative sum of the array elements over the given axis.

        Masked values are set to 0 internally during the computation.
        However, their position is saved, and the result will be masked at
        the same locations.

        Refer to `numpy.cumsum` for full documentation.

        Notes
        -----
        The mask is lost if `out` is not a valid :class:`ma.MaskedArray` !

        Arithmetic is modular when using integer types, and no error is
        raised on overflow.

        See Also
        --------
        numpy.ndarray.cumsum : corresponding function for ndarrays
        numpy.cumsum : equivalent function

        Examples
        --------
        >>> import numpy as np
        >>> marr = np.ma.array(np.arange(10), mask=[0,0,0,1,1,1,0,0,0,0])
        >>> marr.cumsum()
        masked_array(data=[0, 1, 3, --, --, --, 9, 16, 24, 33],
                     mask=[False, False, False,  True,  True,  True, False, False,
                           False, False],
               fill_value=999999)

        """
        result = self.filled(0).cumsum(axis=axis, dtype=dtype, out=out)
        if out is not None:
            if isinstance(out, MaskedArray):
                out.__setmask__(self.mask)
            return out
        result = result.view(type(self))
        result.__setmask__(self._mask)
        return result

    def prod(self, axis=None, dtype=None, out=None, keepdims=np._NoValue):
        """
        Return the product of the array elements over the given axis.

        Masked elements are set to 1 internally for computation.

        Refer to `numpy.prod` for full documentation.

        Notes
        -----
        Arithmetic is modular when using integer types, and no error is raised
        on overflow.

        See Also
        --------
        numpy.ndarray.prod : corresponding function for ndarrays
        numpy.prod : equivalent function
        """
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}

        _mask = self._mask
        newmask = _check_mask_axis(_mask, axis, **kwargs)
        # No explicit output
        if out is None:
            result = self.filled(1).prod(axis, dtype=dtype, **kwargs)
            rndim = getattr(result, 'ndim', 0)
            if rndim:
                result = result.view(type(self))
                result.__setmask__(newmask)
            elif newmask:
                result = masked
            return result
        # Explicit output
        result = self.filled(1).prod(axis, dtype=dtype, out=out, **kwargs)
        if isinstance(out, MaskedArray):
            outmask = getmask(out)
            if outmask is nomask:
                outmask = out._mask = make_mask_none(out.shape)
            outmask.flat = newmask
        return out
    product = prod

    def cumprod(self, axis=None, dtype=None, out=None):
        """
        Return the cumulative product of the array elements over the given axis.

        Masked values are set to 1 internally during the computation.
        However, their position is saved, and the result will be masked at
        the same locations.

        Refer to `numpy.cumprod` for full documentation.

        Notes
        -----
        The mask is lost if `out` is not a valid MaskedArray !

        Arithmetic is modular when using integer types, and no error is
        raised on overflow.

        See Also
        --------
        numpy.ndarray.cumprod : corresponding function for ndarrays
        numpy.cumprod : equivalent function
        """
        result = self.filled(1).cumprod(axis=axis, dtype=dtype, out=out)
        if out is not None:
            if isinstance(out, MaskedArray):
                out.__setmask__(self._mask)
            return out
        result = result.view(type(self))
        result.__setmask__(self._mask)
        return result

    def mean(self, axis=None, dtype=None, out=None, keepdims=np._NoValue):
        """
        Returns the average of the array elements along given axis.

        Masked entries are ignored, and result elements which are not
        finite will be masked.

        Refer to `numpy.mean` for full documentation.

        See Also
        --------
        numpy.ndarray.mean : corresponding function for ndarrays
        numpy.mean : Equivalent function
        numpy.ma.average : Weighted average.

        Examples
        --------
        >>> import numpy as np
        >>> a = np.ma.array([1,2,3], mask=[False, False, True])
        >>> a
        masked_array(data=[1, 2, --],
                     mask=[False, False,  True],
               fill_value=999999)
        >>> a.mean()
        1.5

        """
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}
        if self._mask is nomask:
            result = super().mean(axis=axis, dtype=dtype, **kwargs)[()]
        else:
            is_float16_result = False
            if dtype is None:
                if issubclass(self.dtype.type, (ntypes.integer, ntypes.bool)):
                    dtype = mu.dtype('f8')
                elif issubclass(self.dtype.type, ntypes.float16):
                    dtype = mu.dtype('f4')
                    is_float16_result = True
            dsum = self.sum(axis=axis, dtype=dtype, **kwargs)
            cnt = self.count(axis=axis, **kwargs)
            if cnt.shape == () and (cnt == 0):
                result = masked
            elif is_float16_result:
                result = self.dtype.type(dsum * 1. / cnt)
            else:
                result = dsum * 1. / cnt
        if out is not None:
            out.flat = result
            if isinstance(out, MaskedArray):
                outmask = getmask(out)
                if outmask is nomask:
                    outmask = out._mask = make_mask_none(out.shape)
                outmask.flat = getmask(result)
            return out
        return result

    def anom(self, axis=None, dtype=None):
        """
        Compute the anomalies (deviations from the arithmetic mean)
        along the given axis.

        Returns an array of anomalies, with the same shape as the input and
        where the arithmetic mean is computed along the given axis.

        Parameters
        ----------
        axis : int, optional
            Axis over which the anomalies are taken.
            The default is to use the mean of the flattened array as reference.
        dtype : dtype, optional
            Type to use in computing the variance. For arrays of integer type
             the default is float32; for arrays of float types it is the same as
             the array type.

        See Also
        --------
        mean : Compute the mean of the array.

        Examples
        --------
        >>> import numpy as np
        >>> a = np.ma.array([1,2,3])
        >>> a.anom()
        masked_array(data=[-1.,  0.,  1.],
                     mask=False,
               fill_value=1e+20)

        """
        m = self.mean(axis, dtype)
        if not axis:
            return self - m
        else:
            return self - expand_dims(m, axis)

    def var(self, axis=None, dtype=None, out=None, ddof=0,
            keepdims=np._NoValue, mean=np._NoValue):
        """
        Returns the variance of the array elements along given axis.

        Masked entries are ignored, and result elements which are not
        finite will be masked.

        Refer to `numpy.var` for full documentation.

        See Also
        --------
        numpy.ndarray.var : corresponding function for ndarrays
        numpy.var : Equivalent function
        """
        kwargs = {}

        if keepdims is not np._NoValue:
            kwargs['keepdims'] = keepdims

        # Easy case: nomask, business as usual
        if self._mask is nomask:

            if mean is not np._NoValue:
                kwargs['mean'] = mean

            ret = super().var(axis=axis, dtype=dtype, out=out, ddof=ddof,
                              **kwargs)[()]
            if out is not None:
                if isinstance(out, MaskedArray):
                    out.__setmask__(nomask)
                return out
            return ret

        # Some data are masked, yay!
        cnt = self.count(axis=axis, **kwargs) - ddof

        if mean is not np._NoValue:
            danom = self - mean
        else:
            danom = self - self.mean(axis, dtype, keepdims=True)

        if iscomplexobj(self):
            danom = umath.absolute(danom) ** 2
        else:
            danom *= danom
        dvar = divide(danom.sum(axis, **kwargs), cnt).view(type(self))
        # Apply the mask if it's not a scalar
        if dvar.ndim:
            dvar._mask = mask_or(self._mask.all(axis, **kwargs), (cnt <= 0))
            dvar._update_from(self)
        elif getmask(dvar):
            # Make sure that masked is returned when the scalar is masked.
            dvar = masked
            if out is not None:
                if isinstance(out, MaskedArray):
                    out.flat = 0
                    out.__setmask__(True)
                elif out.dtype.kind in 'biu':
                    errmsg = "Masked data information would be lost in one or "\
                             "more location."
                    raise MaskError(errmsg)
                else:
                    out.flat = np.nan
                return out
        # In case with have an explicit output
        if out is not None:
            # Set the data
            out.flat = dvar
            # Set the mask if needed
            if isinstance(out, MaskedArray):
                out.__setmask__(dvar.mask)
            return out
        return dvar
    var.__doc__ = np.var.__doc__

    def std(self, axis=None, dtype=None, out=None, ddof=0,
            keepdims=np._NoValue, mean=np._NoValue):
        """
        Returns the standard deviation of the array elements along given axis.

        Masked entries are ignored.

        Refer to `numpy.std` for full documentation.

        See Also
        --------
        numpy.ndarray.std : corresponding function for ndarrays
        numpy.std : Equivalent function
        """
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}

        dvar = self.var(axis, dtype, out, ddof, **kwargs)
        if dvar is not masked:
            if out is not None:
                np.power(out, 0.5, out=out, casting='unsafe')
                return out
            dvar = sqrt(dvar)
        return dvar

    def round(self, decimals=0, out=None):
        """
        Return each element rounded to the given number of decimals.

        Refer to `numpy.around` for full documentation.

        See Also
        --------
        numpy.ndarray.round : corresponding function for ndarrays
        numpy.around : equivalent function

        Examples
        --------
        >>> import numpy as np
        >>> import numpy.ma as ma
        >>> x = ma.array([1.35, 2.5, 1.5, 1.75, 2.25, 2.75],
        ...              mask=[0, 0, 0, 1, 0, 0])
        >>> ma.round(x)
        masked_array(data=[1.0, 2.0, 2.0, --, 2.0, 3.0],
                     mask=[False, False, False,  True, False, False],
                fill_value=1e+20)

        """
        result = self._data.round(decimals=decimals, out=out).view(type(self))
        if result.ndim > 0:
            result._mask = self._mask
            result._update_from(self)
        elif self._mask:
            # Return masked when the scalar is masked
            result = masked
        # No explicit output: we're done
        if out is None:
            return result
        if isinstance(out, MaskedArray):
            out.__setmask__(self._mask)
        return out

    def argsort(self, axis=np._NoValue, kind=None, order=None, endwith=True,
                fill_value=None, *, stable=False):
        """
        Return an ndarray of indices that sort the array along the
        specified axis.  Masked values are filled beforehand to
        `fill_value`.

        Parameters
        ----------
        axis : int, optional
            Axis along which to sort. If None, the default, the flattened array
            is used.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
            The sorting algorithm used.
        order : list, optional
            When `a` is an array with fields defined, this argument specifies
            which fields to compare first, second, etc.  Not all fields need be
            specified.
        endwith : {True, False}, optional
            Whether missing values (if any) should be treated as the largest values
            (True) or the smallest values (False)
            When the array contains unmasked values at the same extremes of the
            datatype, the ordering of these values and the masked values is
            undefined.
        fill_value : scalar or None, optional
            Value used internally for the masked values.
            If ``fill_value`` is not None, it supersedes ``endwith``.
        stable : bool, optional
            Only for compatibility with ``np.argsort``. Ignored.

        Returns
        -------
        index_array : ndarray, int
            Array of indices that sort `a` along the specified axis.
            In other words, ``a[index_array]`` yields a sorted `a`.

        See Also
        --------
        ma.MaskedArray.sort : Describes sorting algorithms used.
        lexsort : Indirect stable sort with multiple keys.
        numpy.ndarray.sort : Inplace sort.

        Notes
        -----
        See `sort` for notes on the different sorting algorithms.

        Examples
        --------
        >>> import numpy as np
        >>> a = np.ma.array([3,2,1], mask=[False, False, True])
        >>> a
        masked_array(data=[3, 2, --],
                     mask=[False, False,  True],
               fill_value=999999)
        >>> a.argsort()
        array([1, 0, 2])

        """
        if stable:
            raise ValueError(
                "`stable` parameter is not supported for masked arrays."
            )

        # 2017-04-11, Numpy 1.13.0, gh-8701: warn on axis default
        if axis is np._NoValue:
            axis = _deprecate_argsort_axis(self)

        if fill_value is None:
            if endwith:
                # nan > inf
                if np.issubdtype(self.dtype, np.floating):
                    fill_value = np.nan
                else:
                    fill_value = minimum_fill_value(self)
            else:
                fill_value = maximum_fill_value(self)

        filled = self.filled(fill_value)
        return filled.argsort(axis=axis, kind=kind, order=order)

    def argmin(self, axis=None, fill_value=None, out=None, *,
                keepdims=np._NoValue):
        """
        Return array of indices to the minimum values along the given axis.

        Parameters
        ----------
        axis : {None, integer}
            If None, the index is into the flattened array, otherwise along
            the specified axis
        fill_value : scalar or None, optional
            Value used to fill in the masked values.  If None, the output of
            minimum_fill_value(self._data) is used instead.
        out : {None, array}, optional
            Array into which the result can be placed. Its type is preserved
            and it must be of the right shape to hold the output.

        Returns
        -------
        ndarray or scalar
            If multi-dimension input, returns a new ndarray of indices to the
            minimum values along the given axis.  Otherwise, returns a scalar
            of index to the minimum values along the given axis.

        Examples
        --------
        >>> import numpy as np
        >>> x = np.ma.array(np.arange(4), mask=[1,1,0,0])
        >>> x.shape = (2,2)
        >>> x
        masked_array(
          data=[[--, --],
                [2, 3]],
          mask=[[ True,  True],
                [False, False]],
          fill_value=999999)
        >>> x.argmin(axis=0, fill_value=-1)
        array([0, 0])
        >>> x.argmin(axis=0, fill_value=9)
        array([1, 1])

        """
        if fill_value is None:
            fill_value = minimum_fill_value(self)
        d = self.filled(fill_value).view(ndarray)
        keepdims = False if keepdims is np._NoValue else bool(keepdims)
        return d.argmin(axis, out=out, keepdims=keepdims)

    def argmax(self, axis=None, fill_value=None, out=None, *,
                keepdims=np._NoValue):
        """
        Returns array of indices of the maximum values along the given axis.
        Masked values are treated as if they had the value fill_value.

        Parameters
        ----------
        axis : {None, integer}
            If None, the index is into the flattened array, otherwise along
            the specified axis
        fill_value : scalar or None, optional
            Value used to fill in the masked values.  If None, the output of
            maximum_fill_value(self._data) is used instead.
        out : {None, array}, optional
            Array into which the result can be placed. Its type is preserved
            and it must be of the right shape to hold the output.

        Returns
        -------
        index_array : {integer_array}

        Examples
        --------
        >>> import numpy as np
        >>> a = np.arange(6).reshape(2,3)
        >>> a.argmax()
        5
        >>> a.argmax(0)
        array([1, 1, 1])
        >>> a.argmax(1)
        array([2, 2])

        """
        if fill_value is None:
            fill_value = maximum_fill_value(self._data)
        d = self.filled(fill_value).view(ndarray)
        keepdims = False if keepdims is np._NoValue else bool(keepdims)
        return d.argmax(axis, out=out, keepdims=keepdims)

    def sort(self, axis=-1, kind=None, order=None, endwith=True,
             fill_value=None, *, stable=False):
        """
        Sort the array, in-place

        Parameters
        ----------
        a : array_like
            Array to be sorted.
        axis : int, optional
            Axis along which to sort. If None, the array is flattened before
            sorting. The default is -1, which sorts along the last axis.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
            The sorting algorithm used.
        order : list, optional
            When `a` is a structured array, this argument specifies which fields
            to compare first, second, and so on.  This list does not need to
            include all of the fields.
        endwith : {True, False}, optional
            Whether missing values (if any) should be treated as the largest values
            (True) or the smallest values (False)
            When the array contains unmasked values sorting at the same extremes of the
            datatype, the ordering of these values and the masked values is
            undefined.
        fill_value : scalar or None, optional
            Value used internally for the masked values.
            If ``fill_value`` is not None, it supersedes ``endwith``.
        stable : bool, optional
            Only for compatibility with ``np.sort``. Ignored.

        Returns
        -------
        sorted_array : ndarray
            Array of the same type and shape as `a`.

        See Also
        --------
        numpy.ndarray.sort : Method to sort an array in-place.
        argsort : Indirect sort.
        lexsort : Indirect stable sort on multiple keys.
        searchsorted : Find elements in a sorted array.

        Notes
        -----
        See ``sort`` for notes on the different sorting algorithms.

        Examples
        --------
        >>> import numpy as np
        >>> a = np.ma.array([1, 2, 5, 4, 3],mask=[0, 1, 0, 1, 0])
        >>> # Default
        >>> a.sort()
        >>> a
        masked_array(data=[1, 3, 5, --, --],
                     mask=[False, False, False,  True,  True],
               fill_value=999999)

        >>> a = np.ma.array([1, 2, 5, 4, 3],mask=[0, 1, 0, 1, 0])
        >>> # Put missing values in the front
        >>> a.sort(endwith=False)
        >>> a
        masked_array(data=[--, --, 1, 3, 5],
                     mask=[ True,  True, False, False, False],
               fill_value=999999)

        >>> a = np.ma.array([1, 2, 5, 4, 3],mask=[0, 1, 0, 1, 0])
        >>> # fill_value takes over endwith
        >>> a.sort(endwith=False, fill_value=3)
        >>> a
        masked_array(data=[1, --, --, 3, 5],
                     mask=[False,  True,  True, False, False],
               fill_value=999999)

        """
        if stable:
            raise ValueError(
                "`stable` parameter is not supported for masked arrays."
            )

        if self._mask is nomask:
            ndarray.sort(self, axis=axis, kind=kind, order=order)
            return

        if self is masked:
            return

        sidx = self.argsort(axis=axis, kind=kind, order=order,
                            fill_value=fill_value, endwith=endwith)

        self[...] = np.take_along_axis(self, sidx, axis=axis)

    def min(self, axis=None, out=None, fill_value=None, keepdims=np._NoValue):
        """
        Return the minimum along a given axis.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis along which to operate.  By default, ``axis`` is None and the
            flattened input is used.
            If this is a tuple of ints, the minimum is selected over multiple
            axes, instead of a single axis or all the axes as before.
        out : array_like, optional
            Alternative output array in which to place the result.  Must be of
            the same shape and buffer length as the expected output.
        fill_value : scalar or None, optional
            Value used to fill in the masked values.
            If None, use the output of `minimum_fill_value`.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the array.

        Returns
        -------
        amin : array_like
            New array holding the result.
            If ``out`` was specified, ``out`` is returned.

        See Also
        --------
        ma.minimum_fill_value
            Returns the minimum filling value for a given datatype.

        Examples
        --------
        >>> import numpy.ma as ma
        >>> x = [[1., -2., 3.], [0.2, -0.7, 0.1]]
        >>> mask = [[1, 1, 0], [0, 0, 1]]
        >>> masked_x = ma.masked_array(x, mask)
        >>> masked_x
        masked_array(
          data=[[--, --, 3.0],
                [0.2, -0.7, --]],
          mask=[[ True,  True, False],
                [False, False,  True]],
          fill_value=1e+20)
        >>> ma.min(masked_x)
        -0.7
        >>> ma.min(masked_x, axis=-1)
        masked_array(data=[3.0, -0.7],
                     mask=[False, False],
                fill_value=1e+20)
        >>> ma.min(masked_x, axis=0, keepdims=True)
        masked_array(data=[[0.2, -0.7, 3.0]],
                     mask=[[False, False, False]],
                fill_value=1e+20)
        >>> mask = [[1, 1, 1,], [1, 1, 1]]
        >>> masked_x = ma.masked_array(x, mask)
        >>> ma.min(masked_x, axis=0)
        masked_array(data=[--, --, --],
                     mask=[ True,  True,  True],
                fill_value=1e+20,
                    dtype=float64)
        """
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}

        _mask = self._mask
        newmask = _check_mask_axis(_mask, axis, **kwargs)
        if fill_value is None:
            fill_value = minimum_fill_value(self)
        # No explicit output
        if out is None:
            result = self.filled(fill_value).min(
                axis=axis, out=out, **kwargs).view(type(self))
            if result.ndim:
                # Set the mask
                result.__setmask__(newmask)
                # Get rid of Infs
                if newmask.ndim:
                    np.copyto(result, result.fill_value, where=newmask)
            elif newmask:
                result = masked
            return result
        # Explicit output
        self.filled(fill_value).min(axis=axis, out=out, **kwargs)
        if isinstance(out, MaskedArray):
            outmask = getmask(out)
            if outmask is nomask:
                outmask = out._mask = make_mask_none(out.shape)
            outmask.flat = newmask
        else:
            if out.dtype.kind in 'biu':
                errmsg = "Masked data information would be lost in one or more"\
                         " location."
                raise MaskError(errmsg)
            np.copyto(out, np.nan, where=newmask)
        return out

    def max(self, axis=None, out=None, fill_value=None, keepdims=np._NoValue):
        """
        Return the maximum along a given axis.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis along which to operate.  By default, ``axis`` is None and the
            flattened input is used.
            If this is a tuple of ints, the maximum is selected over multiple
            axes, instead of a single axis or all the axes as before.
        out : array_like, optional
            Alternative output array in which to place the result.  Must
            be of the same shape and buffer length as the expected output.
        fill_value : scalar or None, optional
            Value used to fill in the masked values.
            If None, use the output of maximum_fill_value().
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the array.

        Returns
        -------
        amax : array_like
            New array holding the result.
            If ``out`` was specified, ``out`` is returned.

        See Also
        --------
        ma.maximum_fill_value
            Returns the maximum filling value for a given datatype.

        Examples
        --------
        >>> import numpy.ma as ma
        >>> x = [[-1., 2.5], [4., -2.], [3., 0.]]
        >>> mask = [[0, 0], [1, 0], [1, 0]]
        >>> masked_x = ma.masked_array(x, mask)
        >>> masked_x
        masked_array(
          data=[[-1.0, 2.5],
                [--, -2.0],
                [--, 0.0]],
          mask=[[False, False],
                [ True, False],
                [ True, False]],
          fill_value=1e+20)
        >>> ma.max(masked_x)
        2.5
        >>> ma.max(masked_x, axis=0)
        masked_array(data=[-1.0, 2.5],
                     mask=[False, False],
               fill_value=1e+20)
        >>> ma.max(masked_x, axis=1, keepdims=True)
        masked_array(
          data=[[2.5],
                [-2.0],
                [0.0]],
          mask=[[False],
                [False],
                [False]],
          fill_value=1e+20)
        >>> mask = [[1, 1], [1, 1], [1, 1]]
        >>> masked_x = ma.masked_array(x, mask)
        >>> ma.max(masked_x, axis=1)
        masked_array(data=[--, --, --],
                     mask=[ True,  True,  True],
               fill_value=1e+20,
                    dtype=float64)
        """
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}

        _mask = self._mask
        newmask = _check_mask_axis(_mask, axis, **kwargs)
        if fill_value is None:
            fill_value = maximum_fill_value(self)
        # No explicit output
        if out is None:
            result = self.filled(fill_value).max(
                axis=axis, out=out, **kwargs).view(type(self))
            if result.ndim:
                # Set the mask
                result.__setmask__(newmask)
                # Get rid of Infs
                if newmask.ndim:
                    np.copyto(result, result.fill_value, where=newmask)
            elif newmask:
                result = masked
            return result
        # Explicit output
        self.filled(fill_value).max(axis=axis, out=out, **kwargs)
        if isinstance(out, MaskedArray):
            outmask = getmask(out)
            if outmask is nomask:
                outmask = out._mask = make_mask_none(out.shape)
            outmask.flat = newmask
        else:

            if out.dtype.kind in 'biu':
                errmsg = "Masked data information would be lost in one or more"\
                         " location."
                raise MaskError(errmsg)
            np.copyto(out, np.nan, where=newmask)
        return out

    def ptp(self, axis=None, out=None, fill_value=None, keepdims=False):
        """
        Return (maximum - minimum) along the given dimension
        (i.e. peak-to-peak value).

        .. warning::
            `ptp` preserves the data type of the array. This means the
            return value for an input of signed integers with n bits
            (e.g. `np.int8`, `np.int16`, etc) is also a signed integer
            with n bits.  In that case, peak-to-peak values greater than
            ``2**(n-1)-1`` will be returned as negative values. An example
            with a work-around is shown below.

        Parameters
        ----------
        axis : {None, int}, optional
            Axis along which to find the peaks.  If None (default) the
            flattened array is used.
        out : {None, array_like}, optional
            Alternative output array in which to place the result. It must
            have the same shape and buffer length as the expected output
            but the type will be cast if necessary.
        fill_value : scalar or None, optional
            Value used to fill in the masked values.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the array.

        Returns
        -------
        ptp : ndarray.
            A new array holding the result, unless ``out`` was
            specified, in which case a reference to ``out`` is returned.

        Examples
        --------
        >>> import numpy as np
        >>> x = np.ma.MaskedArray([[4, 9, 2, 10],
        ...                        [6, 9, 7, 12]])

        >>> x.ptp(axis=1)
        masked_array(data=[8, 6],
                     mask=False,
               fill_value=999999)

        >>> x.ptp(axis=0)
        masked_array(data=[2, 0, 5, 2],
                     mask=False,
               fill_value=999999)

        >>> x.ptp()
        10

        This example shows that a negative value can be returned when
        the input is an array of signed integers.

        >>> y = np.ma.MaskedArray([[1, 127],
        ...                        [0, 127],
        ...                        [-1, 127],
        ...                        [-2, 127]], dtype=np.int8)
        >>> y.ptp(axis=1)
        masked_array(data=[ 126,  127, -128, -127],
                     mask=False,
               fill_value=np.int64(999999),
                    dtype=int8)

        A work-around is to use the `view()` method to view the result as
        unsigned integers with the same bit width:

        >>> y.ptp(axis=1).view(np.uint8)
        masked_array(data=[126, 127, 128, 129],
                     mask=False,
               fill_value=np.uint64(999999),
                    dtype=uint8)
        """
        if out is None:
            result = self.max(axis=axis, fill_value=fill_value,
                              keepdims=keepdims)
            result -= self.min(axis=axis, fill_value=fill_value,
                               keepdims=keepdims)
            return result
        out.flat = self.max(axis=axis, out=out, fill_value=fill_value,
                            keepdims=keepdims)
        min_value = self.min(axis=axis, fill_value=fill_value,
                             keepdims=keepdims)
        np.subtract(out, min_value, out=out, casting='unsafe')
        return out

    def partition(self, *args, **kwargs):
        warnings.warn("Warning: 'partition' will ignore the 'mask' "
                      f"of the {self.__class__.__name__}.",
                      stacklevel=2)
        return super().partition(*args, **kwargs)

    def argpartition(self, *args, **kwargs):
        warnings.warn("Warning: 'argpartition' will ignore the 'mask' "
                      f"of the {self.__class__.__name__}.",
                      stacklevel=2)
        return super().argpartition(*args, **kwargs)

    def take(self, indices, axis=None, out=None, mode='raise'):
        """
        Take elements from a masked array along an axis.

        This function does the same thing as "fancy" indexing (indexing arrays
        using arrays) for masked arrays. It can be easier to use if you need
        elements along a given axis.

        Parameters
        ----------
        a : masked_array
            The source masked array.
        indices : array_like
            The indices of the values to extract. Also allow scalars for indices.
        axis : int, optional
            The axis over which to select values. By default, the flattened
            input array is used.
        out : MaskedArray, optional
            If provided, the result will be placed in this array. It should
            be of the appropriate shape and dtype. Note that `out` is always
            buffered if `mode='raise'`; use other modes for better performance.
        mode : {'raise', 'wrap', 'clip'}, optional
            Specifies how out-of-bounds indices will behave.

            * 'raise' -- raise an error (default)
            * 'wrap' -- wrap around
            * 'clip' -- clip to the range

            'clip' mode means that all indices that are too large are replaced
            by the index that addresses the last element along that axis. Note
            that this disables indexing with negative numbers.

        Returns
        -------
        out : MaskedArray
            The returned array has the same type as `a`.

        See Also
        --------
        numpy.take : Equivalent function for ndarrays.
        compress : Take elements using a boolean mask.
        take_along_axis : Take elements by matching the array and the index arrays.

        Notes
        -----
        This function behaves similarly to `numpy.take`, but it handles masked
        values. The mask is retained in the output array, and masked values
        in the input array remain masked in the output.

        Examples
        --------
        >>> import numpy as np
        >>> a = np.ma.array([4, 3, 5, 7, 6, 8], mask=[0, 0, 1, 0, 1, 0])
        >>> indices = [0, 1, 4]
        >>> np.ma.take(a, indices)
        masked_array(data=[4, 3, --],
                    mask=[False, False,  True],
            fill_value=999999)

        When `indices` is not one-dimensional, the output also has these dimensions:

        >>> np.ma.take(a, [[0, 1], [2, 3]])
        masked_array(data=[[4, 3],
                        [--, 7]],
                    mask=[[False, False],
                        [ True, False]],
            fill_value=999999)
        """
        (_data, _mask) = (self._data, self._mask)
        cls = type(self)
        # Make sure the indices are not masked
        maskindices = getmask(indices)
        if maskindices is not nomask:
            indices = indices.filled(0)
        # Get the data, promoting scalars to 0d arrays with [...] so that
        # .view works correctly
        if out is None:
            out = _data.take(indices, axis=axis, mode=mode)[...].view(cls)
        else:
            np.take(_data, indices, axis=axis, mode=mode, out=out)
        # Get the mask
        if isinstance(out, MaskedArray):
            if _mask is nomask:
                outmask = maskindices
            else:
                outmask = _mask.take(indices, axis=axis, mode=mode)
                outmask |= maskindices
            out.__setmask__(outmask)
        # demote 0d arrays back to scalars, for consistency with ndarray.take
        return out[()]

    # Array methods
    copy = _arraymethod('copy')
    diagonal = _arraymethod('diagonal')
    flatten = _arraymethod('flatten')
    repeat = _arraymethod('repeat')
    squeeze = _arraymethod('squeeze')
    swapaxes = _arraymethod('swapaxes')
    T = property(fget=lambda self: self.transpose())
    transpose = _arraymethod('transpose')

    @property
    def mT(self):
        """
        Return the matrix-transpose of the masked array.

        The matrix transpose is the transpose of the last two dimensions, even
        if the array is of higher dimension.

        .. versionadded:: 2.0

        Returns
        -------
        result: MaskedArray
            The masked array with the last two dimensions transposed

        Raises
        ------
        ValueError
            If the array is of dimension less than 2.

        See Also
        --------
        ndarray.mT:
            Equivalent method for arrays
        """

        if self.ndim < 2:
            raise ValueError("matrix transpose with ndim < 2 is undefined")

        if self._mask is nomask:
            return masked_array(data=self._data.mT)
        else:
            return masked_array(data=self.data.mT, mask=self.mask.mT)

    def tolist(self, fill_value=None):
        """
        Return the data portion of the masked array as a hierarchical Python list.

        Data items are converted to the nearest compatible Python type.
        Masked values are converted to `fill_value`. If `fill_value` is None,
        the corresponding entries in the output list will be ``None``.

        Parameters
        ----------
        fill_value : scalar, optional
            The value to use for invalid entries. Default is None.

        Returns
        -------
        result : list
            The Python list representation of the masked array.

        Examples
        --------
        >>> import numpy as np
        >>> x = np.ma.array([[1,2,3], [4,5,6], [7,8,9]], mask=[0] + [1,0]*4)
        >>> x.tolist()
        [[1, None, 3], [None, 5, None], [7, None, 9]]
        >>> x.tolist(-999)
        [[1, -999, 3], [-999, 5, -999], [7, -999, 9]]

        """
        _mask = self._mask
        # No mask ? Just return .data.tolist ?
        if _mask is nomask:
            return self._data.tolist()
        # Explicit fill_value: fill the array and get the list
        if fill_value is not None:
            return self.filled(fill_value).tolist()
        # Structured array.
        names = self.dtype.names
        if names:
            result = self._data.astype([(_, object) for _ in names])
            for n in names:
                result[n][_mask[n]] = None
            return result.tolist()
        # Standard arrays.
        if _mask is nomask:
            return [None]
        # Set temps to save time when dealing w/ marrays.
        inishape = self.shape
        result = np.array(self._data.ravel(), dtype=object)
        result[_mask.ravel()] = None
        result.shape = inishape
        return result.tolist()

    def tobytes(self, fill_value=None, order='C'):
        """
        Return the array data as a string containing the raw bytes in the array.

        The array is filled with a fill value before the string conversion.

        Parameters
        ----------
        fill_value : scalar, optional
            Value used to fill in the masked values. Default is None, in which
            case `MaskedArray.fill_value` is used.
        order : {'C','F','A'}, optional
            Order of the data item in the copy. Default is 'C'.

            - 'C'   -- C order (row major).
            - 'F'   -- Fortran order (column major).
            - 'A'   -- Any, current order of array.
            - None  -- Same as 'A'.

        See Also
        --------
        numpy.ndarray.tobytes
        tolist, tofile

        Notes
        -----
        As for `ndarray.tobytes`, information about the shape, dtype, etc.,
        but also about `fill_value`, will be lost.

        Examples
        --------
        >>> import numpy as np
        >>> x = np.ma.array(np.array([[1, 2], [3, 4]]), mask=[[0, 1], [1, 0]])
        >>> x.tobytes()
        b'\\x01\\x00\\x00\\x00\\x00\\x00\\x00\\x00?B\\x0f\\x00\\x00\\x00\\x00\\x00?B\\x0f\\x00\\x00\\x00\\x00\\x00\\x04\\x00\\x00\\x00\\x00\\x00\\x00\\x00'

        """
        return self.filled(fill_value).tobytes(order=order)

    def tofile(self, fid, sep="", format="%s"):
        """
        Save a masked array to a file in binary format.

        .. warning::
          This function is not implemented yet.

        Raises
        ------
        NotImplementedError
            When `tofile` is called.

        """
        raise NotImplementedError("MaskedArray.tofile() not implemented yet.")

    def toflex(self):
        """
        Transforms a masked array into a flexible-type array.

        The flexible type array that is returned will have two fields:

        * the ``_data`` field stores the ``_data`` part of the array.
        * the ``_mask`` field stores the ``_mask`` part of the array.

        Parameters
        ----------
        None

        Returns
        -------
        record : ndarray
            A new flexible-type `ndarray` with two fields: the first element
            containing a value, the second element containing the corresponding
            mask boolean. The returned record shape matches self.shape.

        Notes
        -----
        A side-effect of transforming a masked array into a flexible `ndarray` is
        that meta information (``fill_value``, ...) will be lost.

        Examples
        --------
        >>> import numpy as np
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> x
        masked_array(
          data=[[1, --, 3],
                [--, 5, --],
                [7, --, 9]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
        >>> x.toflex()
        array([[(1, False), (2,  True), (3, False)],
               [(4,  True), (5, False), (6,  True)],
               [(7, False), (8,  True), (9, False)]],
              dtype=[('_data', '<i8'), ('_mask', '?')])

        """
        # Get the basic dtype.
        ddtype = self.dtype
        # Make sure we have a mask
        _mask = self._mask
        if _mask is None:
            _mask = make_mask_none(self.shape, ddtype)
        # And get its dtype
        mdtype = self._mask.dtype

        record = np.ndarray(shape=self.shape,
                            dtype=[('_data', ddtype), ('_mask', mdtype)])
        record['_data'] = self._data
        record['_mask'] = self._mask
        return record
    torecords = toflex

    # Pickling
    def __getstate__(self):
        """Return the internal state of the masked array, for pickling
        purposes.

        """
        cf = 'CF'[self.flags.fnc]
        data_state = super().__reduce__()[2]
        return data_state + (getmaskarray(self).tobytes(cf), self._fill_value)

    def __setstate__(self, state):
        """Restore the internal state of the masked array, for
        pickling purposes.  ``state`` is typically the output of the
        ``__getstate__`` output, and is a 5-tuple:

        - class name
        - a tuple giving the shape of the data
        - a typecode for the data
        - a binary string for the data
        - a binary string for the mask.

        """
        (_, shp, typ, isf, raw, msk, flv) = state
        super().__setstate__((shp, typ, isf, raw))
        self._mask.__setstate__((shp, make_mask_descr(typ), isf, msk))
        self.fill_value = flv

    def __reduce__(self):
        """Return a 3-tuple for pickling a MaskedArray.

        """
        return (_mareconstruct,
                (self.__class__, self._baseclass, (0,), 'b',),
                self.__getstate__())

    def __deepcopy__(self, memo=None):
        from copy import deepcopy
        copied = MaskedArray.__new__(type(self), self, copy=True)
        if memo is None:
            memo = {}
        memo[id(self)] = copied
        for (k, v) in self.__dict__.items():
            copied.__dict__[k] = deepcopy(v, memo)
        # as clearly documented for np.copy(), you need to use
        # deepcopy() directly for arrays of object type that may
        # contain compound types--you cannot depend on normal
        # copy semantics to do the right thing here
        if self.dtype.hasobject:
            copied._data[...] = deepcopy(copied._data)
        return copied


def _mareconstruct(subtype, baseclass, baseshape, basetype,):
    """Internal function that builds a new MaskedArray from the
    information stored in a pickle.

    """
    _data = ndarray.__new__(baseclass, baseshape, basetype)
    _mask = ndarray.__new__(ndarray, baseshape, make_mask_descr(basetype))
    return subtype.__new__(subtype, _data, mask=_mask, dtype=basetype,)


class mvoid(MaskedArray):
    """
    Fake a 'void' object to use for masked array with structured dtypes.
    """

    def __new__(self, data, mask=nomask, dtype=None, fill_value=None,
                hardmask=False, copy=False, subok=True):
        copy = None if not copy else True
        _data = np.array(data, copy=copy, subok=subok, dtype=dtype)
        _data = _data.view(self)
        _data._hardmask = hardmask
        if mask is not nomask:
            if isinstance(mask, np.void):
                _data._mask = mask
            else:
                try:
                    # Mask is already a 0D array
                    _data._mask = np.void(mask)
                except TypeError:
                    # Transform the mask to a void
                    mdtype = make_mask_descr(dtype)
                    _data._mask = np.array(mask, dtype=mdtype)[()]
        if fill_value is not None:
            _data.fill_value = fill_value
        return _data

    @property
    def _data(self):
        # Make sure that the _data part is a np.void
        return super()._data[()]

    def __getitem__(self, indx):
        """
        Get the index.

        """
        m = self._mask
        if isinstance(m[indx], ndarray):
            # Can happen when indx is a multi-dimensional field:
            # A = ma.masked_array(data=[([0,1],)], mask=[([True,
            #                     False],)], dtype=[("A", ">i2", (2,))])
            # x = A[0]; y = x["A"]; then y.mask["A"].size==2
            # and we can not say masked/unmasked.
            # The result is no longer mvoid!
            # See also issue #6724.
            return masked_array(
                data=self._data[indx], mask=m[indx],
                fill_value=self._fill_value[indx],
                hard_mask=self._hardmask)
        if m is not nomask and m[indx]:
            return masked
        return self._data[indx]

    def __setitem__(self, indx, value):
        self._data[indx] = value
        if self._hardmask:
            self._mask[indx] |= getattr(value, "_mask", False)
        else:
            self._mask[indx] = getattr(value, "_mask", False)

    def __str__(self):
        m = self._mask
        if m is nomask:
            return str(self._data)

        rdtype = _replace_dtype_fields(self._data.dtype, "O")
        data_arr = super()._data
        res = data_arr.astype(rdtype)
        _recursive_printoption(res, self._mask, masked_print_option)
        return str(res)

    __repr__ = __str__

    def __iter__(self):
        "Defines an iterator for mvoid"
        (_data, _mask) = (self._data, self._mask)
        if _mask is nomask:
            yield from _data
        else:
            for (d, m) in zip(_data, _mask):
                if m:
                    yield masked
                else:
                    yield d

    def __len__(self):
        return self._data.__len__()

    def filled(self, fill_value=None):
        """
        Return a copy with masked fields filled with a given value.

        Parameters
        ----------
        fill_value : array_like, optional
            The value to use for invalid entries. Can be scalar or
            non-scalar. If latter is the case, the filled array should
            be broadcastable over input array. Default is None, in
            which case the `fill_value` attribute is used instead.

        Returns
        -------
        filled_void
            A `np.void` object

        See Also
        --------
        MaskedArray.filled

        """
        return asarray(self).filled(fill_value)[()]

    def tolist(self):
        """
    Transforms the mvoid object into a tuple.

    Masked fields are replaced by None.

    Returns
    -------
    returned_tuple
        Tuple of fields
        """
        _mask = self._mask
        if _mask is nomask:
            return self._data.tolist()
        result = []
        for (d, m) in zip(self._data, self._mask):
            if m:
                result.append(None)
            else:
                # .item() makes sure we return a standard Python object
                result.append(d.item())
        return tuple(result)


##############################################################################
#                                Shortcuts                                   #
##############################################################################


def isMaskedArray(x):
    """
    Test whether input is an instance of MaskedArray.

    This function returns True if `x` is an instance of MaskedArray
    and returns False otherwise.  Any object is accepted as input.

    Parameters
    ----------
    x : object
        Object to test.

    Returns
    -------
    result : bool
        True if `x` is a MaskedArray.

    See Also
    --------
    isMA : Alias to isMaskedArray.
    isarray : Alias to isMaskedArray.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> a = np.eye(3, 3)
    >>> a
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> m = ma.masked_values(a, 0)
    >>> m
    masked_array(
      data=[[1.0, --, --],
            [--, 1.0, --],
            [--, --, 1.0]],
      mask=[[False,  True,  True],
            [ True, False,  True],
            [ True,  True, False]],
      fill_value=0.0)
    >>> ma.isMaskedArray(a)
    False
    >>> ma.isMaskedArray(m)
    True
    >>> ma.isMaskedArray([0, 1, 2])
    False

    """
    return isinstance(x, MaskedArray)


isarray = isMaskedArray
isMA = isMaskedArray  # backward compatibility


class MaskedConstant(MaskedArray):
    # the lone np.ma.masked instance
    __singleton = None

    @classmethod
    def __has_singleton(cls):
        # second case ensures `cls.__singleton` is not just a view on the
        # superclass singleton
        return cls.__singleton is not None and type(cls.__singleton) is cls

    def __new__(cls):
        if not cls.__has_singleton():
            # We define the masked singleton as a float for higher precedence.
            # Note that it can be tricky sometimes w/ type comparison
            data = np.array(0.)
            mask = np.array(True)

            # prevent any modifications
            data.flags.writeable = False
            mask.flags.writeable = False

            # don't fall back on MaskedArray.__new__(MaskedConstant), since
            # that might confuse it - this way, the construction is entirely
            # within our control
            cls.__singleton = MaskedArray(data, mask=mask).view(cls)

        return cls.__singleton

    def __array_finalize__(self, obj):
        if not self.__has_singleton():
            # this handles the `.view` in __new__, which we want to copy across
            # properties normally
            return super().__array_finalize__(obj)
        elif self is self.__singleton:
            # not clear how this can happen, play it safe
            pass
        else:
            # everywhere else, we want to downcast to MaskedArray, to prevent a
            # duplicate maskedconstant.
            self.__class__ = MaskedArray
            MaskedArray.__array_finalize__(self, obj)

    def __array_wrap__(self, obj, context=None, return_scalar=False):
        return self.view(MaskedArray).__array_wrap__(obj, context)

    def __str__(self):
        return str(masked_print_option._display)

    def __repr__(self):
        if self is MaskedConstant.__singleton:
            return 'masked'
        else:
            # it's a subclass, or something is wrong, make it obvious
            return object.__repr__(self)

    def __format__(self, format_spec):
        # Replace ndarray.__format__ with the default, which supports no
        # format characters.
        # Supporting format characters is unwise here, because we do not know
        # what type the user was expecting - better to not guess.
        try:
            return object.__format__(self, format_spec)
        except TypeError:
            # 2020-03-23, NumPy 1.19.0
            warnings.warn(
                "Format strings passed to MaskedConstant are ignored,"
                " but in future may error or produce different behavior",
                FutureWarning, stacklevel=2
            )
            return object.__format__(self, "")

    def __reduce__(self):
        """Override of MaskedArray's __reduce__.
        """
        return (self.__class__, ())

    # inplace operations have no effect. We have to override them to avoid
    # trying to modify the readonly data and mask arrays
    def __iop__(self, other):
        return self
    __iadd__ = \
    __isub__ = \
    __imul__ = \
    __ifloordiv__ = \
    __itruediv__ = \
    __ipow__ = \
        __iop__
    del __iop__  # don't leave this around

    def copy(self, *args, **kwargs):
        """ Copy is a no-op on the maskedconstant, as it is a scalar """
        # maskedconstant is a scalar, so copy doesn't need to copy. There's
        # precedent for this with `np.bool` scalars.
        return self

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __setattr__(self, attr, value):
        if not self.__has_singleton():
            # allow the singleton to be initialized
            return super().__setattr__(attr, value)
        elif self is self.__singleton:
            raise AttributeError(
                f"attributes of {self!r} are not writeable")
        else:
            # duplicate instance - we can end up here from __array_finalize__,
            # where we set the __class__ attribute
            return super().__setattr__(attr, value)


masked = masked_singleton = MaskedConstant()
masked_array = MaskedArray


def array(data, dtype=None, copy=False, order=None,
          mask=nomask, fill_value=None, keep_mask=True,
          hard_mask=False, shrink=True, subok=True, ndmin=0):
    """
    Shortcut to MaskedArray.

    The options are in a different order for convenience and backwards
    compatibility.

    """
    return MaskedArray(data, mask=mask, dtype=dtype, copy=copy,
                       subok=subok, keep_mask=keep_mask,
                       hard_mask=hard_mask, fill_value=fill_value,
                       ndmin=ndmin, shrink=shrink, order=order)


array.__doc__ = masked_array.__doc__


def is_masked(x):
    """
    Determine whether input has masked values.

    Accepts any object as input, but always returns False unless the
    input is a MaskedArray containing masked values.

    Parameters
    ----------
    x : array_like
        Array to check for masked values.

    Returns
    -------
    result : bool
        True if `x` is a MaskedArray with masked values, False otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> x = ma.masked_equal([0, 1, 0, 2, 3], 0)
    >>> x
    masked_array(data=[--, 1, --, 2, 3],
                 mask=[ True, False,  True, False, False],
           fill_value=0)
    >>> ma.is_masked(x)
    True
    >>> x = ma.masked_equal([0, 1, 0, 2, 3], 42)
    >>> x
    masked_array(data=[0, 1, 0, 2, 3],
                 mask=False,
           fill_value=42)
    >>> ma.is_masked(x)
    False

    Always returns False if `x` isn't a MaskedArray.

    >>> x = [False, True, False]
    >>> ma.is_masked(x)
    False
    >>> x = 'a string'
    >>> ma.is_masked(x)
    False

    """
    m = getmask(x)
    if m is nomask:
        return False
    elif m.any():
        return True
    return False


##############################################################################
#                             Extrema functions                              #
##############################################################################


class _extrema_operation(_MaskedUFunc):
    """
    Generic class for maximum/minimum functions.

    .. note::
      This is the base class for `_maximum_operation` and
      `_minimum_operation`.

    """
    def __init__(self, ufunc, compare, fill_value):
        super().__init__(ufunc)
        self.compare = compare
        self.fill_value_func = fill_value

    def __call__(self, a, b):
        "Executes the call behavior."

        return where(self.compare(a, b), a, b)

    def reduce(self, target, axis=np._NoValue):
        "Reduce target along the given axis."
        target = narray(target, copy=None, subok=True)
        m = getmask(target)

        if axis is np._NoValue and target.ndim > 1:
            name = self.__name__
            # 2017-05-06, Numpy 1.13.0: warn on axis default
            warnings.warn(
                f"In the future the default for ma.{name}.reduce will be axis=0, "
                f"not the current None, to match np.{name}.reduce. "
                "Explicitly pass 0 or None to silence this warning.",
                MaskedArrayFutureWarning, stacklevel=2)
            axis = None

        if axis is not np._NoValue:
            kwargs = {'axis': axis}
        else:
            kwargs = {}

        if m is nomask:
            t = self.f.reduce(target, **kwargs)
        else:
            target = target.filled(
                self.fill_value_func(target)).view(type(target))
            t = self.f.reduce(target, **kwargs)
            m = umath.logical_and.reduce(m, **kwargs)
            if hasattr(t, '_mask'):
                t._mask = m
            elif m:
                t = masked
        return t

    def outer(self, a, b):
        "Return the function applied to the outer product of a and b."
        ma = getmask(a)
        mb = getmask(b)
        if ma is nomask and mb is nomask:
            m = nomask
        else:
            ma = getmaskarray(a)
            mb = getmaskarray(b)
            m = logical_or.outer(ma, mb)
        result = self.f.outer(filled(a), filled(b))
        if not isinstance(result, MaskedArray):
            result = result.view(MaskedArray)
        result._mask = m
        return result

def min(obj, axis=None, out=None, fill_value=None, keepdims=np._NoValue):
    kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}

    try:
        return obj.min(axis=axis, fill_value=fill_value, out=out, **kwargs)
    except (AttributeError, TypeError):
        # If obj doesn't have a min method, or if the method doesn't accept a
        # fill_value argument
        return asanyarray(obj).min(axis=axis, fill_value=fill_value,
                                   out=out, **kwargs)


min.__doc__ = MaskedArray.min.__doc__

def max(obj, axis=None, out=None, fill_value=None, keepdims=np._NoValue):
    kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}

    try:
        return obj.max(axis=axis, fill_value=fill_value, out=out, **kwargs)
    except (AttributeError, TypeError):
        # If obj doesn't have a max method, or if the method doesn't accept a
        # fill_value argument
        return asanyarray(obj).max(axis=axis, fill_value=fill_value,
                                   out=out, **kwargs)


max.__doc__ = MaskedArray.max.__doc__


def ptp(obj, axis=None, out=None, fill_value=None, keepdims=np._NoValue):
    kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}
    try:
        return obj.ptp(axis, out=out, fill_value=fill_value, **kwargs)
    except (AttributeError, TypeError):
        # If obj doesn't have a ptp method or if the method doesn't accept
        # a fill_value argument
        return asanyarray(obj).ptp(axis=axis, fill_value=fill_value,
                                   out=out, **kwargs)


ptp.__doc__ = MaskedArray.ptp.__doc__


##############################################################################
#           Definition of functions from the corresponding methods           #
##############################################################################


class _frommethod:
    """
    Define functions from existing MaskedArray methods.

    Parameters
    ----------
    methodname : str
        Name of the method to transform.

    """

    def __init__(self, methodname, reversed=False):
        self.__name__ = methodname
        self.__qualname__ = methodname
        self.__doc__ = self.getdoc()
        self.reversed = reversed

    def getdoc(self):
        "Return the doc of the function (from the doc of the method)."
        meth = getattr(MaskedArray, self.__name__, None) or\
            getattr(np, self.__name__, None)
        signature = self.__name__ + get_object_signature(meth)
        if meth is not None:
            doc = f"""    {signature}
{getattr(meth, '__doc__', None)}"""
            return doc

    def __call__(self, a, *args, **params):
        if self.reversed:
            args = list(args)
            a, args[0] = args[0], a

        marr = asanyarray(a)
        method_name = self.__name__
        method = getattr(type(marr), method_name, None)
        if method is None:
            # use the corresponding np function
            method = getattr(np, method_name)

        return method(marr, *args, **params)


all = _frommethod('all')
anomalies = anom = _frommethod('anom')
any = _frommethod('any')
compress = _frommethod('compress', reversed=True)
cumprod = _frommethod('cumprod')
cumsum = _frommethod('cumsum')
copy = _frommethod('copy')
diagonal = _frommethod('diagonal')
harden_mask = _frommethod('harden_mask')
ids = _frommethod('ids')
maximum = _extrema_operation(umath.maximum, greater, maximum_fill_value)
mean = _frommethod('mean')
minimum = _extrema_operation(umath.minimum, less, minimum_fill_value)
nonzero = _frommethod('nonzero')
prod = _frommethod('prod')
product = _frommethod('product')
ravel = _frommethod('ravel')
repeat = _frommethod('repeat')
shrink_mask = _frommethod('shrink_mask')
soften_mask = _frommethod('soften_mask')
std = _frommethod('std')
sum = _frommethod('sum')
swapaxes = _frommethod('swapaxes')
#take = _frommethod('take')
trace = _frommethod('trace')
var = _frommethod('var')

count = _frommethod('count')

def take(a, indices, axis=None, out=None, mode='raise'):
    """

    """
    a = masked_array(a)
    return a.take(indices, axis=axis, out=out, mode=mode)


def power(a, b, third=None):
    """
    Returns element-wise base array raised to power from second array.

    This is the masked array version of `numpy.power`. For details see
    `numpy.power`.

    See Also
    --------
    numpy.power

    Notes
    -----
    The *out* argument to `numpy.power` is not supported, `third` has to be
    None.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> x = [11.2, -3.973, 0.801, -1.41]
    >>> mask = [0, 0, 0, 1]
    >>> masked_x = ma.masked_array(x, mask)
    >>> masked_x
    masked_array(data=[11.2, -3.973, 0.801, --],
             mask=[False, False, False,  True],
       fill_value=1e+20)
    >>> ma.power(masked_x, 2)
    masked_array(data=[125.43999999999998, 15.784728999999999,
                   0.6416010000000001, --],
             mask=[False, False, False,  True],
       fill_value=1e+20)
    >>> y = [-0.5, 2, 0, 17]
    >>> masked_y = ma.masked_array(y, mask)
    >>> masked_y
    masked_array(data=[-0.5, 2.0, 0.0, --],
             mask=[False, False, False,  True],
       fill_value=1e+20)
    >>> ma.power(masked_x, masked_y)
    masked_array(data=[0.2988071523335984, 15.784728999999999, 1.0, --],
             mask=[False, False, False,  True],
       fill_value=1e+20)

    """
    if third is not None:
        raise MaskError("3-argument power not supported.")
    # Get the masks
    ma = getmask(a)
    mb = getmask(b)
    m = mask_or(ma, mb)
    # Get the rawdata
    fa = getdata(a)
    fb = getdata(b)
    # Get the type of the result (so that we preserve subclasses)
    if isinstance(a, MaskedArray):
        basetype = type(a)
    else:
        basetype = MaskedArray
    # Get the result and view it as a (subclass of) MaskedArray
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(m, fa, umath.power(fa, fb)).view(basetype)
    result._update_from(a)
    # Find where we're in trouble w/ NaNs and Infs
    invalid = np.logical_not(np.isfinite(result.view(ndarray)))
    # Add the initial mask
    if m is not nomask:
        if not result.ndim:
            return masked
        result._mask = np.logical_or(m, invalid)
    # Fix the invalid parts
    if invalid.any():
        if not result.ndim:
            return masked
        elif result._mask is nomask:
            result._mask = invalid
        result._data[invalid] = result.fill_value
    return result


argmin = _frommethod('argmin')
argmax = _frommethod('argmax')

def argsort(a, axis=np._NoValue, kind=None, order=None, endwith=True,
            fill_value=None, *, stable=None):
    "Function version of the eponymous method."
    a = np.asanyarray(a)

    # 2017-04-11, Numpy 1.13.0, gh-8701: warn on axis default
    if axis is np._NoValue:
        axis = _deprecate_argsort_axis(a)

    if isinstance(a, MaskedArray):
        return a.argsort(axis=axis, kind=kind, order=order, endwith=endwith,
                         fill_value=fill_value, stable=None)
    else:
        return a.argsort(axis=axis, kind=kind, order=order, stable=None)


argsort.__doc__ = MaskedArray.argsort.__doc__

def sort(a, axis=-1, kind=None, order=None, endwith=True, fill_value=None, *,
         stable=None):
    """
    Return a sorted copy of the masked array.

    Equivalent to creating a copy of the array
    and applying the  MaskedArray ``sort()`` method.

    Refer to ``MaskedArray.sort`` for the full documentation

    See Also
    --------
    MaskedArray.sort : equivalent method

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> x = [11.2, -3.973, 0.801, -1.41]
    >>> mask = [0, 0, 0, 1]
    >>> masked_x = ma.masked_array(x, mask)
    >>> masked_x
    masked_array(data=[11.2, -3.973, 0.801, --],
                 mask=[False, False, False,  True],
           fill_value=1e+20)
    >>> ma.sort(masked_x)
    masked_array(data=[-3.973, 0.801, 11.2, --],
                 mask=[False, False, False,  True],
           fill_value=1e+20)
    """
    a = np.array(a, copy=True, subok=True)
    if axis is None:
        a = a.flatten()
        axis = 0

    if isinstance(a, MaskedArray):
        a.sort(axis=axis, kind=kind, order=order, endwith=endwith,
               fill_value=fill_value, stable=stable)
    else:
        a.sort(axis=axis, kind=kind, order=order, stable=stable)
    return a


def compressed(x):
    """
    Return all the non-masked data as a 1-D array.

    This function is equivalent to calling the "compressed" method of a
    `ma.MaskedArray`, see `ma.MaskedArray.compressed` for details.

    See Also
    --------
    ma.MaskedArray.compressed : Equivalent method.

    Examples
    --------
    >>> import numpy as np

    Create an array with negative values masked:

    >>> import numpy as np
    >>> x = np.array([[1, -1, 0], [2, -1, 3], [7, 4, -1]])
    >>> masked_x = np.ma.masked_array(x, mask=x < 0)
    >>> masked_x
    masked_array(
      data=[[1, --, 0],
            [2, --, 3],
            [7, 4, --]],
      mask=[[False,  True, False],
            [False,  True, False],
            [False, False,  True]],
      fill_value=999999)

    Compress the masked array into a 1-D array of non-masked values:

    >>> np.ma.compressed(masked_x)
    array([1, 0, 2, 3, 7, 4])

    """
    return asanyarray(x).compressed()


def concatenate(arrays, axis=0):
    """
    Concatenate a sequence of arrays along the given axis.

    Parameters
    ----------
    arrays : sequence of array_like
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).
    axis : int, optional
        The axis along which the arrays will be joined. Default is 0.

    Returns
    -------
    result : MaskedArray
        The concatenated array with any masked entries preserved.

    See Also
    --------
    numpy.concatenate : Equivalent function in the top-level NumPy module.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> a = ma.arange(3)
    >>> a[1] = ma.masked
    >>> b = ma.arange(2, 5)
    >>> a
    masked_array(data=[0, --, 2],
                 mask=[False,  True, False],
           fill_value=999999)
    >>> b
    masked_array(data=[2, 3, 4],
                 mask=False,
           fill_value=999999)
    >>> ma.concatenate([a, b])
    masked_array(data=[0, --, 2, 2, 3, 4],
                 mask=[False,  True, False, False, False, False],
           fill_value=999999)

    """
    d = np.concatenate([getdata(a) for a in arrays], axis)
    rcls = get_masked_subclass(*arrays)
    data = d.view(rcls)
    # Check whether one of the arrays has a non-empty mask.
    for x in arrays:
        if getmask(x) is not nomask:
            break
    else:
        return data
    # OK, so we have to concatenate the masks
    dm = np.concatenate([getmaskarray(a) for a in arrays], axis)
    dm = dm.reshape(d.shape)

    # If we decide to keep a '_shrinkmask' option, we want to check that
    # all of them are True, and then check for dm.any()
    data._mask = _shrink_mask(dm)
    return data


def diag(v, k=0):
    """
    Extract a diagonal or construct a diagonal array.

    This function is the equivalent of `numpy.diag` that takes masked
    values into account, see `numpy.diag` for details.

    See Also
    --------
    numpy.diag : Equivalent function for ndarrays.

    Examples
    --------
    >>> import numpy as np

    Create an array with negative values masked:

    >>> import numpy as np
    >>> x = np.array([[11.2, -3.973, 18], [0.801, -1.41, 12], [7, 33, -12]])
    >>> masked_x = np.ma.masked_array(x, mask=x < 0)
    >>> masked_x
    masked_array(
      data=[[11.2, --, 18.0],
            [0.801, --, 12.0],
            [7.0, 33.0, --]],
      mask=[[False,  True, False],
            [False,  True, False],
            [False, False,  True]],
      fill_value=1e+20)

    Isolate the main diagonal from the masked array:

    >>> np.ma.diag(masked_x)
    masked_array(data=[11.2, --, --],
                 mask=[False,  True,  True],
           fill_value=1e+20)

    Isolate the first diagonal below the main diagonal:

    >>> np.ma.diag(masked_x, -1)
    masked_array(data=[0.801, 33.0],
                 mask=[False, False],
           fill_value=1e+20)

    """
    output = np.diag(v, k).view(MaskedArray)
    if getmask(v) is not nomask:
        output._mask = np.diag(v._mask, k)
    return output


def left_shift(a, n):
    """
    Shift the bits of an integer to the left.

    This is the masked array version of `numpy.left_shift`, for details
    see that function.

    See Also
    --------
    numpy.left_shift

    Examples
    --------
    Shift with a masked array:

    >>> arr = np.ma.array([10, 20, 30], mask=[False, True, False])
    >>> np.ma.left_shift(arr, 1)
    masked_array(data=[20, --, 60],
                 mask=[False,  True, False],
           fill_value=999999)

    Large shift:

    >>> np.ma.left_shift(10, 10)
    masked_array(data=10240,
                 mask=False,
           fill_value=999999)

    Shift with a scalar and an array:

    >>> scalar = 10
    >>> arr = np.ma.array([1, 2, 3], mask=[False, True, False])
    >>> np.ma.left_shift(scalar, arr)
    masked_array(data=[20, --, 80],
                 mask=[False,  True, False],
           fill_value=999999)


    """
    m = getmask(a)
    if m is nomask:
        d = umath.left_shift(filled(a), n)
        return masked_array(d)
    else:
        d = umath.left_shift(filled(a, 0), n)
        return masked_array(d, mask=m)


def right_shift(a, n):
    """
    Shift the bits of an integer to the right.

    This is the masked array version of `numpy.right_shift`, for details
    see that function.

    See Also
    --------
    numpy.right_shift

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> x = [11, 3, 8, 1]
    >>> mask = [0, 0, 0, 1]
    >>> masked_x = ma.masked_array(x, mask)
    >>> masked_x
    masked_array(data=[11, 3, 8, --],
                 mask=[False, False, False,  True],
           fill_value=999999)
    >>> ma.right_shift(masked_x,1)
    masked_array(data=[5, 1, 4, --],
                 mask=[False, False, False,  True],
           fill_value=999999)

    """
    m = getmask(a)
    if m is nomask:
        d = umath.right_shift(filled(a), n)
        return masked_array(d)
    else:
        d = umath.right_shift(filled(a, 0), n)
        return masked_array(d, mask=m)


def put(a, indices, values, mode='raise'):
    """
    Set storage-indexed locations to corresponding values.

    This function is equivalent to `MaskedArray.put`, see that method
    for details.

    See Also
    --------
    MaskedArray.put

    Examples
    --------
    Putting values in a masked array:

    >>> a = np.ma.array([1, 2, 3, 4], mask=[False, True, False, False])
    >>> np.ma.put(a, [1, 3], [10, 30])
    >>> a
    masked_array(data=[ 1, 10,  3, 30],
                 mask=False,
           fill_value=999999)

    Using put with a 2D array:

    >>> b = np.ma.array([[1, 2], [3, 4]], mask=[[False, True], [False, False]])
    >>> np.ma.put(b, [[0, 1], [1, 0]], [[10, 20], [30, 40]])
    >>> b
    masked_array(
      data=[[40, 30],
            [ 3,  4]],
      mask=False,
      fill_value=999999)

    """
    # We can't use 'frommethod', the order of arguments is different
    try:
        return a.put(indices, values, mode=mode)
    except AttributeError:
        return np.asarray(a).put(indices, values, mode=mode)


def putmask(a, mask, values):  # , mode='raise'):
    """
    Changes elements of an array based on conditional and input values.

    This is the masked array version of `numpy.putmask`, for details see
    `numpy.putmask`.

    See Also
    --------
    numpy.putmask

    Notes
    -----
    Using a masked array as `values` will **not** transform a `ndarray` into
    a `MaskedArray`.

    Examples
    --------
    >>> import numpy as np
    >>> arr = [[1, 2], [3, 4]]
    >>> mask = [[1, 0], [0, 0]]
    >>> x = np.ma.array(arr, mask=mask)
    >>> np.ma.putmask(x, x < 4, 10*x)
    >>> x
    masked_array(
      data=[[--, 20],
            [30, 4]],
      mask=[[ True, False],
            [False, False]],
      fill_value=999999)
    >>> x.data
    array([[10, 20],
           [30,  4]])

    """
    # We can't use 'frommethod', the order of arguments is different
    if not isinstance(a, MaskedArray):
        a = a.view(MaskedArray)
    (valdata, valmask) = (getdata(values), getmask(values))
    if getmask(a) is nomask:
        if valmask is not nomask:
            a._sharedmask = True
            a._mask = make_mask_none(a.shape, a.dtype)
            np.copyto(a._mask, valmask, where=mask)
    elif a._hardmask:
        if valmask is not nomask:
            m = a._mask.copy()
            np.copyto(m, valmask, where=mask)
            a.mask |= m
    else:
        if valmask is nomask:
            valmask = getmaskarray(values)
        np.copyto(a._mask, valmask, where=mask)
    np.copyto(a._data, valdata, where=mask)


def transpose(a, axes=None):
    """
    Permute the dimensions of an array.

    This function is exactly equivalent to `numpy.transpose`.

    See Also
    --------
    numpy.transpose : Equivalent function in top-level NumPy module.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> x = ma.arange(4).reshape((2,2))
    >>> x[1, 1] = ma.masked
    >>> x
    masked_array(
      data=[[0, 1],
            [2, --]],
      mask=[[False, False],
            [False,  True]],
      fill_value=999999)

    >>> ma.transpose(x)
    masked_array(
      data=[[0, 2],
            [1, --]],
      mask=[[False, False],
            [False,  True]],
      fill_value=999999)
    """
    # We can't use 'frommethod', as 'transpose' doesn't take keywords
    try:
        return a.transpose(axes)
    except AttributeError:
        return np.asarray(a).transpose(axes).view(MaskedArray)


def reshape(a, new_shape, order='C'):
    """
    Returns an array containing the same data with a new shape.

    Refer to `MaskedArray.reshape` for full documentation.

    See Also
    --------
    MaskedArray.reshape : equivalent function

    Examples
    --------
    Reshaping a 1-D array:

    >>> a = np.ma.array([1, 2, 3, 4])
    >>> np.ma.reshape(a, (2, 2))
    masked_array(
      data=[[1, 2],
            [3, 4]],
      mask=False,
      fill_value=999999)

    Reshaping a 2-D array:

    >>> b = np.ma.array([[1, 2], [3, 4]])
    >>> np.ma.reshape(b, (1, 4))
    masked_array(data=[[1, 2, 3, 4]],
                 mask=False,
           fill_value=999999)

    Reshaping a 1-D array with a mask:

    >>> c = np.ma.array([1, 2, 3, 4], mask=[False, True, False, False])
    >>> np.ma.reshape(c, (2, 2))
    masked_array(
      data=[[1, --],
            [3, 4]],
      mask=[[False,  True],
            [False, False]],
      fill_value=999999)

    """
    # We can't use 'frommethod', it whine about some parameters. Dmmit.
    try:
        return a.reshape(new_shape, order=order)
    except AttributeError:
        _tmp = np.asarray(a).reshape(new_shape, order=order)
        return _tmp.view(MaskedArray)


def resize(x, new_shape):
    """
    Return a new masked array with the specified size and shape.

    This is the masked equivalent of the `numpy.resize` function. The new
    array is filled with repeated copies of `x` (in the order that the
    data are stored in memory). If `x` is masked, the new array will be
    masked, and the new mask will be a repetition of the old one.

    See Also
    --------
    numpy.resize : Equivalent function in the top level NumPy module.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> a = ma.array([[1, 2] ,[3, 4]])
    >>> a[0, 1] = ma.masked
    >>> a
    masked_array(
      data=[[1, --],
            [3, 4]],
      mask=[[False,  True],
            [False, False]],
      fill_value=999999)
    >>> np.resize(a, (3, 3))
    masked_array(
      data=[[1, 2, 3],
            [4, 1, 2],
            [3, 4, 1]],
      mask=False,
      fill_value=999999)
    >>> ma.resize(a, (3, 3))
    masked_array(
      data=[[1, --, 3],
            [4, 1, --],
            [3, 4, 1]],
      mask=[[False,  True, False],
            [False, False,  True],
            [False, False, False]],
      fill_value=999999)

    A MaskedArray is always returned, regardless of the input type.

    >>> a = np.array([[1, 2] ,[3, 4]])
    >>> ma.resize(a, (3, 3))
    masked_array(
      data=[[1, 2, 3],
            [4, 1, 2],
            [3, 4, 1]],
      mask=False,
      fill_value=999999)

    """
    # We can't use _frommethods here, as N.resize is notoriously whiny.
    m = getmask(x)
    if m is not nomask:
        m = np.resize(m, new_shape)
    result = np.resize(x, new_shape).view(get_masked_subclass(x))
    if result.ndim:
        result._mask = m
    return result


def ndim(obj):
    """
    maskedarray version of the numpy function.

    """
    return np.ndim(getdata(obj))


ndim.__doc__ = np.ndim.__doc__


def shape(obj):
    "maskedarray version of the numpy function."
    return np.shape(getdata(obj))


shape.__doc__ = np.shape.__doc__


def size(obj, axis=None):
    "maskedarray version of the numpy function."
    return np.size(getdata(obj), axis)


size.__doc__ = np.size.__doc__


def diff(a, /, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue):
    """
    Calculate the n-th discrete difference along the given axis.
    The first difference is given by ``out[i] = a[i+1] - a[i]`` along
    the given axis, higher differences are calculated by using `diff`
    recursively.
    Preserves the input mask.

    Parameters
    ----------
    a : array_like
        Input array
    n : int, optional
        The number of times values are differenced. If zero, the input
        is returned as-is.
    axis : int, optional
        The axis along which the difference is taken, default is the
        last axis.
    prepend, append : array_like, optional
        Values to prepend or append to `a` along axis prior to
        performing the difference.  Scalar values are expanded to
        arrays with length 1 in the direction of axis and the shape
        of the input array in along all other axes.  Otherwise the
        dimension and shape must match `a` except along axis.

    Returns
    -------
    diff : MaskedArray
        The n-th differences. The shape of the output is the same as `a`
        except along `axis` where the dimension is smaller by `n`. The
        type of the output is the same as the type of the difference
        between any two elements of `a`. This is the same as the type of
        `a` in most cases. A notable exception is `datetime64`, which
        results in a `timedelta64` output array.

    See Also
    --------
    numpy.diff : Equivalent function in the top-level NumPy module.

    Notes
    -----
    Type is preserved for boolean arrays, so the result will contain
    `False` when consecutive elements are the same and `True` when they
    differ.

    For unsigned integer arrays, the results will also be unsigned. This
    should not be surprising, as the result is consistent with
    calculating the difference directly:

    >>> u8_arr = np.array([1, 0], dtype=np.uint8)
    >>> np.ma.diff(u8_arr)
    masked_array(data=[255],
                 mask=False,
           fill_value=np.uint64(999999),
                dtype=uint8)
    >>> u8_arr[1,...] - u8_arr[0,...]
    np.uint8(255)

    If this is not desirable, then the array should be cast to a larger
    integer type first:

    >>> i16_arr = u8_arr.astype(np.int16)
    >>> np.ma.diff(i16_arr)
    masked_array(data=[-1],
                 mask=False,
           fill_value=np.int64(999999),
                dtype=int16)

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, 2, 3, 4, 7, 0, 2, 3])
    >>> x = np.ma.masked_where(a < 2, a)
    >>> np.ma.diff(x)
    masked_array(data=[--, 1, 1, 3, --, --, 1],
            mask=[ True, False, False, False,  True,  True, False],
        fill_value=999999)

    >>> np.ma.diff(x, n=2)
    masked_array(data=[--, 0, 2, --, --, --],
                mask=[ True, False, False,  True,  True,  True],
        fill_value=999999)

    >>> a = np.array([[1, 3, 1, 5, 10], [0, 1, 5, 6, 8]])
    >>> x = np.ma.masked_equal(a, value=1)
    >>> np.ma.diff(x)
    masked_array(
        data=[[--, --, --, 5],
                [--, --, 1, 2]],
        mask=[[ True,  True,  True, False],
                [ True,  True, False, False]],
        fill_value=1)

    >>> np.ma.diff(x, axis=0)
    masked_array(data=[[--, --, --, 1, -2]],
            mask=[[ True,  True,  True, False, False]],
        fill_value=1)

    """
    if n == 0:
        return a
    if n < 0:
        raise ValueError("order must be non-negative but got " + repr(n))

    a = np.ma.asanyarray(a)
    if a.ndim == 0:
        raise ValueError(
            "diff requires input that is at least one dimensional"
            )

    combined = []
    if prepend is not np._NoValue:
        prepend = np.ma.asanyarray(prepend)
        if prepend.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            prepend = np.broadcast_to(prepend, tuple(shape))
        combined.append(prepend)

    combined.append(a)

    if append is not np._NoValue:
        append = np.ma.asanyarray(append)
        if append.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            append = np.broadcast_to(append, tuple(shape))
        combined.append(append)

    if len(combined) > 1:
        a = np.ma.concatenate(combined, axis)

    # GH 22465 np.diff without prepend/append preserves the mask
    return np.diff(a, n, axis)


##############################################################################
#                            Extra functions                                 #
##############################################################################


def where(condition, x=_NoValue, y=_NoValue):
    """
    Return a masked array with elements from `x` or `y`, depending on condition.

    .. note::
        When only `condition` is provided, this function is identical to
        `nonzero`. The rest of this documentation covers only the case where
        all three arguments are provided.

    Parameters
    ----------
    condition : array_like, bool
        Where True, yield `x`, otherwise yield `y`.
    x, y : array_like, optional
        Values from which to choose. `x`, `y` and `condition` need to be
        broadcastable to some shape.

    Returns
    -------
    out : MaskedArray
        An masked array with `masked` elements where the condition is masked,
        elements from `x` where `condition` is True, and elements from `y`
        elsewhere.

    See Also
    --------
    numpy.where : Equivalent function in the top-level NumPy module.
    nonzero : The function that is called when x and y are omitted

    Examples
    --------
    >>> import numpy as np
    >>> x = np.ma.array(np.arange(9.).reshape(3, 3), mask=[[0, 1, 0],
    ...                                                    [1, 0, 1],
    ...                                                    [0, 1, 0]])
    >>> x
    masked_array(
      data=[[0.0, --, 2.0],
            [--, 4.0, --],
            [6.0, --, 8.0]],
      mask=[[False,  True, False],
            [ True, False,  True],
            [False,  True, False]],
      fill_value=1e+20)
    >>> np.ma.where(x > 5, x, -3.1416)
    masked_array(
      data=[[-3.1416, --, -3.1416],
            [--, -3.1416, --],
            [6.0, --, 8.0]],
      mask=[[False,  True, False],
            [ True, False,  True],
            [False,  True, False]],
      fill_value=1e+20)

    """

    # handle the single-argument case
    missing = (x is _NoValue, y is _NoValue).count(True)
    if missing == 1:
        raise ValueError("Must provide both 'x' and 'y' or neither.")
    if missing == 2:
        return nonzero(condition)

    # we only care if the condition is true - false or masked pick y
    cf = filled(condition, False)
    xd = getdata(x)
    yd = getdata(y)

    # we need the full arrays here for correct final dimensions
    cm = getmaskarray(condition)
    xm = getmaskarray(x)
    ym = getmaskarray(y)

    # deal with the fact that masked.dtype == float64, but we don't actually
    # want to treat it as that.
    if x is masked and y is not masked:
        xd = np.zeros((), dtype=yd.dtype)
        xm = np.ones((),  dtype=ym.dtype)
    elif y is masked and x is not masked:
        yd = np.zeros((), dtype=xd.dtype)
        ym = np.ones((),  dtype=xm.dtype)

    data = np.where(cf, xd, yd)
    mask = np.where(cf, xm, ym)
    mask = np.where(cm, np.ones((), dtype=mask.dtype), mask)

    # collapse the mask, for backwards compatibility
    mask = _shrink_mask(mask)

    return masked_array(data, mask=mask)


def choose(indices, choices, out=None, mode='raise'):
    """
    Use an index array to construct a new array from a list of choices.

    Given an array of integers and a list of n choice arrays, this method
    will create a new array that merges each of the choice arrays.  Where a
    value in `index` is i, the new array will have the value that choices[i]
    contains in the same place.

    Parameters
    ----------
    indices : ndarray of ints
        This array must contain integers in ``[0, n-1]``, where n is the
        number of choices.
    choices : sequence of arrays
        Choice arrays. The index array and all of the choices should be
        broadcastable to the same shape.
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and `dtype`.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices will behave.

        * 'raise' : raise an error
        * 'wrap' : wrap around
        * 'clip' : clip to the range

    Returns
    -------
    merged_array : array

    See Also
    --------
    choose : equivalent function

    Examples
    --------
    >>> import numpy as np
    >>> choice = np.array([[1,1,1], [2,2,2], [3,3,3]])
    >>> a = np.array([2, 1, 0])
    >>> np.ma.choose(a, choice)
    masked_array(data=[3, 2, 1],
                 mask=False,
           fill_value=999999)

    """
    def fmask(x):
        "Returns the filled array, or True if masked."
        if x is masked:
            return True
        return filled(x)

    def nmask(x):
        "Returns the mask, True if ``masked``, False if ``nomask``."
        if x is masked:
            return True
        return getmask(x)
    # Get the indices.
    c = filled(indices, 0)
    # Get the masks.
    masks = [nmask(x) for x in choices]
    data = [fmask(x) for x in choices]
    # Construct the mask
    outputmask = np.choose(c, masks, mode=mode)
    outputmask = make_mask(mask_or(outputmask, getmask(indices)),
                           copy=False, shrink=True)
    # Get the choices.
    d = np.choose(c, data, mode=mode, out=out).view(MaskedArray)
    if out is not None:
        if isinstance(out, MaskedArray):
            out.__setmask__(outputmask)
        return out
    d.__setmask__(outputmask)
    return d


def round_(a, decimals=0, out=None):
    """
    Return a copy of a, rounded to 'decimals' places.

    When 'decimals' is negative, it specifies the number of positions
    to the left of the decimal point.  The real and imaginary parts of
    complex numbers are rounded separately. Nothing is done if the
    array is not of float type and 'decimals' is greater than or equal
    to 0.

    Parameters
    ----------
    decimals : int
        Number of decimals to round to. May be negative.
    out : array_like
        Existing array to use for output.
        If not given, returns a default copy of a.

    Notes
    -----
    If out is given and does not have a mask attribute, the mask of a
    is lost!

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> x = [11.2, -3.973, 0.801, -1.41]
    >>> mask = [0, 0, 0, 1]
    >>> masked_x = ma.masked_array(x, mask)
    >>> masked_x
    masked_array(data=[11.2, -3.973, 0.801, --],
                 mask=[False, False, False, True],
        fill_value=1e+20)
    >>> ma.round_(masked_x)
    masked_array(data=[11.0, -4.0, 1.0, --],
                 mask=[False, False, False, True],
        fill_value=1e+20)
    >>> ma.round(masked_x, decimals=1)
    masked_array(data=[11.2, -4.0, 0.8, --],
                 mask=[False, False, False, True],
        fill_value=1e+20)
    >>> ma.round_(masked_x, decimals=-1)
    masked_array(data=[10.0, -0.0, 0.0, --],
                 mask=[False, False, False, True],
        fill_value=1e+20)
    """
    if out is None:
        return np.round(a, decimals, out)
    else:
        np.round(getdata(a), decimals, out)
        if hasattr(out, '_mask'):
            out._mask = getmask(a)
        return out


round = round_


def _mask_propagate(a, axis):
    """
    Mask whole 1-d vectors of an array that contain masked values.
    """
    a = array(a, subok=False)
    m = getmask(a)
    if m is nomask or not m.any() or axis is None:
        return a
    a._mask = a._mask.copy()
    axes = normalize_axis_tuple(axis, a.ndim)
    for ax in axes:
        a._mask |= m.any(axis=ax, keepdims=True)
    return a


# Include masked dot here to avoid import problems in getting it from
# extras.py. Note that it is not included in __all__, but rather exported
# from extras in order to avoid backward compatibility problems.
def dot(a, b, strict=False, out=None):
    """
    Return the dot product of two arrays.

    This function is the equivalent of `numpy.dot` that takes masked values
    into account. Note that `strict` and `out` are in different position
    than in the method version. In order to maintain compatibility with the
    corresponding method, it is recommended that the optional arguments be
    treated as keyword only.  At some point that may be mandatory.

    Parameters
    ----------
    a, b : masked_array_like
        Inputs arrays.
    strict : bool, optional
        Whether masked data are propagated (True) or set to 0 (False) for
        the computation. Default is False.  Propagating the mask means that
        if a masked value appears in a row or column, the whole row or
        column is considered masked.
    out : masked_array, optional
        Output argument. This must have the exact kind that would be returned
        if it was not used. In particular, it must have the right type, must be
        C-contiguous, and its dtype must be the dtype that would be returned
        for `dot(a,b)`. This is a performance feature. Therefore, if these
        conditions are not met, an exception is raised, instead of attempting
        to be flexible.

    See Also
    --------
    numpy.dot : Equivalent function for ndarrays.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.ma.array([[1, 2, 3], [4, 5, 6]], mask=[[1, 0, 0], [0, 0, 0]])
    >>> b = np.ma.array([[1, 2], [3, 4], [5, 6]], mask=[[1, 0], [0, 0], [0, 0]])
    >>> np.ma.dot(a, b)
    masked_array(
      data=[[21, 26],
            [45, 64]],
      mask=[[False, False],
            [False, False]],
      fill_value=999999)
    >>> np.ma.dot(a, b, strict=True)
    masked_array(
      data=[[--, --],
            [--, 64]],
      mask=[[ True,  True],
            [ True, False]],
      fill_value=999999)

    """
    if strict is True:
        if np.ndim(a) == 0 or np.ndim(b) == 0:
            pass
        elif b.ndim == 1:
            a = _mask_propagate(a, a.ndim - 1)
            b = _mask_propagate(b, b.ndim - 1)
        else:
            a = _mask_propagate(a, a.ndim - 1)
            b = _mask_propagate(b, b.ndim - 2)
    am = ~getmaskarray(a)
    bm = ~getmaskarray(b)

    if out is None:
        d = np.dot(filled(a, 0), filled(b, 0))
        m = ~np.dot(am, bm)
        if np.ndim(d) == 0:
            d = np.asarray(d)
        r = d.view(get_masked_subclass(a, b))
        r.__setmask__(m)
        return r
    else:
        d = np.dot(filled(a, 0), filled(b, 0), out._data)
        if out.mask.shape != d.shape:
            out._mask = np.empty(d.shape, MaskType)
        np.dot(am, bm, out._mask)
        np.logical_not(out._mask, out._mask)
        return out


def inner(a, b):
    """
    Returns the inner product of a and b for arrays of floating point types.

    Like the generic NumPy equivalent the product sum is over the last dimension
    of a and b. The first argument is not conjugated.

    """
    fa = filled(a, 0)
    fb = filled(b, 0)
    if fa.ndim == 0:
        fa.shape = (1,)
    if fb.ndim == 0:
        fb.shape = (1,)
    return np.inner(fa, fb).view(MaskedArray)


inner.__doc__ = doc_note(np.inner.__doc__,
                         "Masked values are replaced by 0.")
innerproduct = inner


def outer(a, b):
    "maskedarray version of the numpy function."
    fa = filled(a, 0).ravel()
    fb = filled(b, 0).ravel()
    d = np.outer(fa, fb)
    ma = getmask(a)
    mb = getmask(b)
    if ma is nomask and mb is nomask:
        return masked_array(d)
    ma = getmaskarray(a)
    mb = getmaskarray(b)
    m = make_mask(1 - np.outer(1 - ma, 1 - mb), copy=False)
    return masked_array(d, mask=m)


outer.__doc__ = doc_note(np.outer.__doc__,
                         "Masked values are replaced by 0.")
outerproduct = outer


def _convolve_or_correlate(f, a, v, mode, propagate_mask):
    """
    Helper function for ma.correlate and ma.convolve
    """
    if propagate_mask:
        # results which are contributed to by either item in any pair being invalid
        mask = (
            f(getmaskarray(a), np.ones(np.shape(v), dtype=bool), mode=mode)
          | f(np.ones(np.shape(a), dtype=bool), getmaskarray(v), mode=mode)
        )
        data = f(getdata(a), getdata(v), mode=mode)
    else:
        # results which are not contributed to by any pair of valid elements
        mask = ~f(~getmaskarray(a), ~getmaskarray(v), mode=mode)
        data = f(filled(a, 0), filled(v, 0), mode=mode)

    return masked_array(data, mask=mask)


def correlate(a, v, mode='valid', propagate_mask=True):
    """
    Cross-correlation of two 1-dimensional sequences.

    Parameters
    ----------
    a, v : array_like
        Input sequences.
    mode : {'valid', 'same', 'full'}, optional
        Refer to the `np.convolve` docstring.  Note that the default
        is 'valid', unlike `convolve`, which uses 'full'.
    propagate_mask : bool
        If True, then a result element is masked if any masked element contributes
        towards it. If False, then a result element is only masked if no non-masked
        element contribute towards it

    Returns
    -------
    out : MaskedArray
        Discrete cross-correlation of `a` and `v`.

    See Also
    --------
    numpy.correlate : Equivalent function in the top-level NumPy module.

    Examples
    --------
    Basic correlation:

    >>> a = np.ma.array([1, 2, 3])
    >>> v = np.ma.array([0, 1, 0])
    >>> np.ma.correlate(a, v, mode='valid')
    masked_array(data=[2],
                 mask=[False],
           fill_value=999999)

    Correlation with masked elements:

    >>> a = np.ma.array([1, 2, 3], mask=[False, True, False])
    >>> v = np.ma.array([0, 1, 0])
    >>> np.ma.correlate(a, v, mode='valid', propagate_mask=True)
    masked_array(data=[--],
                 mask=[ True],
           fill_value=999999,
                dtype=int64)

    Correlation with different modes and mixed array types:

    >>> a = np.ma.array([1, 2, 3])
    >>> v = np.ma.array([0, 1, 0])
    >>> np.ma.correlate(a, v, mode='full')
    masked_array(data=[0, 1, 2, 3, 0],
                 mask=[False, False, False, False, False],
           fill_value=999999)

    """
    return _convolve_or_correlate(np.correlate, a, v, mode, propagate_mask)


def convolve(a, v, mode='full', propagate_mask=True):
    """
    Returns the discrete, linear convolution of two one-dimensional sequences.

    Parameters
    ----------
    a, v : array_like
        Input sequences.
    mode : {'valid', 'same', 'full'}, optional
        Refer to the `np.convolve` docstring.
    propagate_mask : bool
        If True, then if any masked element is included in the sum for a result
        element, then the result is masked.
        If False, then the result element is only masked if no non-masked cells
        contribute towards it

    Returns
    -------
    out : MaskedArray
        Discrete, linear convolution of `a` and `v`.

    See Also
    --------
    numpy.convolve : Equivalent function in the top-level NumPy module.
    """
    return _convolve_or_correlate(np.convolve, a, v, mode, propagate_mask)


def allequal(a, b, fill_value=True):
    """
    Return True if all entries of a and b are equal, using
    fill_value as a truth value where either or both are masked.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    fill_value : bool, optional
        Whether masked values in a or b are considered equal (True) or not
        (False).

    Returns
    -------
    y : bool
        Returns True if the two arrays are equal within the given
        tolerance, False otherwise. If either array contains NaN,
        then False is returned.

    See Also
    --------
    all, any
    numpy.ma.allclose

    Examples
    --------
    >>> import numpy as np
    >>> a = np.ma.array([1e10, 1e-7, 42.0], mask=[0, 0, 1])
    >>> a
    masked_array(data=[10000000000.0, 1e-07, --],
                 mask=[False, False,  True],
           fill_value=1e+20)

    >>> b = np.array([1e10, 1e-7, -42.0])
    >>> b
    array([  1.00000000e+10,   1.00000000e-07,  -4.20000000e+01])
    >>> np.ma.allequal(a, b, fill_value=False)
    False
    >>> np.ma.allequal(a, b)
    True

    """
    m = mask_or(getmask(a), getmask(b))
    if m is nomask:
        x = getdata(a)
        y = getdata(b)
        d = umath.equal(x, y)
        return d.all()
    elif fill_value:
        x = getdata(a)
        y = getdata(b)
        d = umath.equal(x, y)
        dm = array(d, mask=m, copy=False)
        return dm.filled(True).all(None)
    else:
        return False


def allclose(a, b, masked_equal=True, rtol=1e-5, atol=1e-8):
    """
    Returns True if two arrays are element-wise equal within a tolerance.

    This function is equivalent to `allclose` except that masked values
    are treated as equal (default) or unequal, depending on the `masked_equal`
    argument.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    masked_equal : bool, optional
        Whether masked values in `a` and `b` are considered equal (True) or not
        (False). They are considered equal by default.
    rtol : float, optional
        Relative tolerance. The relative difference is equal to ``rtol * b``.
        Default is 1e-5.
    atol : float, optional
        Absolute tolerance. The absolute difference is equal to `atol`.
        Default is 1e-8.

    Returns
    -------
    y : bool
        Returns True if the two arrays are equal within the given
        tolerance, False otherwise. If either array contains NaN, then
        False is returned.

    See Also
    --------
    all, any
    numpy.allclose : the non-masked `allclose`.

    Notes
    -----
    If the following equation is element-wise True, then `allclose` returns
    True::

      absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

    Return True if all elements of `a` and `b` are equal subject to
    given tolerances.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.ma.array([1e10, 1e-7, 42.0], mask=[0, 0, 1])
    >>> a
    masked_array(data=[10000000000.0, 1e-07, --],
                 mask=[False, False,  True],
           fill_value=1e+20)
    >>> b = np.ma.array([1e10, 1e-8, -42.0], mask=[0, 0, 1])
    >>> np.ma.allclose(a, b)
    False

    >>> a = np.ma.array([1e10, 1e-8, 42.0], mask=[0, 0, 1])
    >>> b = np.ma.array([1.00001e10, 1e-9, -42.0], mask=[0, 0, 1])
    >>> np.ma.allclose(a, b)
    True
    >>> np.ma.allclose(a, b, masked_equal=False)
    False

    Masked values are not compared directly.

    >>> a = np.ma.array([1e10, 1e-8, 42.0], mask=[0, 0, 1])
    >>> b = np.ma.array([1.00001e10, 1e-9, 42.0], mask=[0, 0, 1])
    >>> np.ma.allclose(a, b)
    True
    >>> np.ma.allclose(a, b, masked_equal=False)
    False

    """
    x = masked_array(a, copy=False)
    y = masked_array(b, copy=False)

    # make sure y is an inexact type to avoid abs(MIN_INT); will cause
    # casting of x later.
    # NOTE: We explicitly allow timedelta, which used to work. This could
    #       possibly be deprecated. See also gh-18286.
    #       timedelta works if `atol` is an integer or also a timedelta.
    #       Although, the default tolerances are unlikely to be useful
    if y.dtype.kind != "m":
        dtype = np.result_type(y, 1.)
        if y.dtype != dtype:
            y = masked_array(y, dtype=dtype, copy=False)

    m = mask_or(getmask(x), getmask(y))
    xinf = np.isinf(masked_array(x, copy=False, mask=m)).filled(False)
    # If we have some infs, they should fall at the same place.
    if not np.all(xinf == filled(np.isinf(y), False)):
        return False
    # No infs at all
    if not np.any(xinf):
        d = filled(less_equal(absolute(x - y), atol + rtol * absolute(y)),
                   masked_equal)
        return np.all(d)

    if not np.all(filled(x[xinf] == y[xinf], masked_equal)):
        return False
    x = x[~xinf]
    y = y[~xinf]

    d = filled(less_equal(absolute(x - y), atol + rtol * absolute(y)),
               masked_equal)

    return np.all(d)


def asarray(a, dtype=None, order=None):
    """
    Convert the input to a masked array of the given data-type.

    No copy is performed if the input is already an `ndarray`. If `a` is
    a subclass of `MaskedArray`, a base class `MaskedArray` is returned.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to a masked array. This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists, ndarrays and masked arrays.
    dtype : dtype, optional
        By default, the data-type is inferred from the input data.
    order : {'C', 'F'}, optional
        Whether to use row-major ('C') or column-major ('FORTRAN') memory
        representation.  Default is 'C'.

    Returns
    -------
    out : MaskedArray
        Masked array interpretation of `a`.

    See Also
    --------
    asanyarray : Similar to `asarray`, but conserves subclasses.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(10.).reshape(2, 5)
    >>> x
    array([[0., 1., 2., 3., 4.],
           [5., 6., 7., 8., 9.]])
    >>> np.ma.asarray(x)
    masked_array(
      data=[[0., 1., 2., 3., 4.],
            [5., 6., 7., 8., 9.]],
      mask=False,
      fill_value=1e+20)
    >>> type(np.ma.asarray(x))
    <class 'numpy.ma.MaskedArray'>

    """
    order = order or 'C'
    return masked_array(a, dtype=dtype, copy=False, keep_mask=True,
                        subok=False, order=order)


def asanyarray(a, dtype=None):
    """
    Convert the input to a masked array, conserving subclasses.

    If `a` is a subclass of `MaskedArray`, its class is conserved.
    No copy is performed if the input is already an `ndarray`.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array.
    dtype : dtype, optional
        By default, the data-type is inferred from the input data.
    order : {'C', 'F'}, optional
        Whether to use row-major ('C') or column-major ('FORTRAN') memory
        representation.  Default is 'C'.

    Returns
    -------
    out : MaskedArray
        MaskedArray interpretation of `a`.

    See Also
    --------
    asarray : Similar to `asanyarray`, but does not conserve subclass.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(10.).reshape(2, 5)
    >>> x
    array([[0., 1., 2., 3., 4.],
           [5., 6., 7., 8., 9.]])
    >>> np.ma.asanyarray(x)
    masked_array(
      data=[[0., 1., 2., 3., 4.],
            [5., 6., 7., 8., 9.]],
      mask=False,
      fill_value=1e+20)
    >>> type(np.ma.asanyarray(x))
    <class 'numpy.ma.MaskedArray'>

    """
    # workaround for #8666, to preserve identity. Ideally the bottom line
    # would handle this for us.
    if isinstance(a, MaskedArray) and (dtype is None or dtype == a.dtype):
        return a
    return masked_array(a, dtype=dtype, copy=False, keep_mask=True, subok=True)


##############################################################################
#                               Pickling                                     #
##############################################################################


def fromfile(file, dtype=float, count=-1, sep=''):
    raise NotImplementedError(
        "fromfile() not yet implemented for a MaskedArray.")


def fromflex(fxarray):
    """
    Build a masked array from a suitable flexible-type array.

    The input array has to have a data-type with ``_data`` and ``_mask``
    fields. This type of array is output by `MaskedArray.toflex`.

    Parameters
    ----------
    fxarray : ndarray
        The structured input array, containing ``_data`` and ``_mask``
        fields. If present, other fields are discarded.

    Returns
    -------
    result : MaskedArray
        The constructed masked array.

    See Also
    --------
    MaskedArray.toflex : Build a flexible-type array from a masked array.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.ma.array(np.arange(9).reshape(3, 3), mask=[0] + [1, 0] * 4)
    >>> rec = x.toflex()
    >>> rec
    array([[(0, False), (1,  True), (2, False)],
           [(3,  True), (4, False), (5,  True)],
           [(6, False), (7,  True), (8, False)]],
          dtype=[('_data', '<i8'), ('_mask', '?')])
    >>> x2 = np.ma.fromflex(rec)
    >>> x2
    masked_array(
      data=[[0, --, 2],
            [--, 4, --],
            [6, --, 8]],
      mask=[[False,  True, False],
            [ True, False,  True],
            [False,  True, False]],
      fill_value=999999)

    Extra fields can be present in the structured array but are discarded:

    >>> dt = [('_data', '<i4'), ('_mask', '|b1'), ('field3', '<f4')]
    >>> rec2 = np.zeros((2, 2), dtype=dt)
    >>> rec2
    array([[(0, False, 0.), (0, False, 0.)],
           [(0, False, 0.), (0, False, 0.)]],
          dtype=[('_data', '<i4'), ('_mask', '?'), ('field3', '<f4')])
    >>> y = np.ma.fromflex(rec2)
    >>> y
    masked_array(
      data=[[0, 0],
            [0, 0]],
      mask=[[False, False],
            [False, False]],
      fill_value=np.int64(999999),
      dtype=int32)

    """
    return masked_array(fxarray['_data'], mask=fxarray['_mask'])


class _convert2ma:

    """
    Convert functions from numpy to numpy.ma.

    Parameters
    ----------
        _methodname : string
            Name of the method to transform.

    """
    __doc__ = None

    def __init__(self, funcname, np_ret, np_ma_ret, params=None):
        self._func = getattr(np, funcname)
        self.__doc__ = self.getdoc(np_ret, np_ma_ret)
        self._extras = params or {}

    def getdoc(self, np_ret, np_ma_ret):
        "Return the doc of the function (from the doc of the method)."
        doc = getattr(self._func, '__doc__', None)
        sig = get_object_signature(self._func)
        if doc:
            doc = self._replace_return_type(doc, np_ret, np_ma_ret)
            # Add the signature of the function at the beginning of the doc
            if sig:
                sig = f"{self._func.__name__}{sig}\n"
            doc = sig + doc
        return doc

    def _replace_return_type(self, doc, np_ret, np_ma_ret):
        """
        Replace documentation of ``np`` function's return type.

        Replaces it with the proper type for the ``np.ma`` function.

        Parameters
        ----------
        doc : str
            The documentation of the ``np`` method.
        np_ret : str
            The return type string of the ``np`` method that we want to
            replace. (e.g. "out : ndarray")
        np_ma_ret : str
            The return type string of the ``np.ma`` method.
            (e.g. "out : MaskedArray")
        """
        if np_ret not in doc:
            raise RuntimeError(
                f"Failed to replace `{np_ret}` with `{np_ma_ret}`. "
                f"The documentation string for return type, {np_ret}, is not "
                f"found in the docstring for `np.{self._func.__name__}`. "
                f"Fix the docstring for `np.{self._func.__name__}` or "
                "update the expected string for return type."
            )

        return doc.replace(np_ret, np_ma_ret)

    def __call__(self, *args, **params):
        # Find the common parameters to the call and the definition
        _extras = self._extras
        common_params = set(params).intersection(_extras)
        # Drop the common parameters from the call
        for p in common_params:
            _extras[p] = params.pop(p)
        # Get the result
        result = self._func.__call__(*args, **params).view(MaskedArray)
        if "fill_value" in common_params:
            result.fill_value = _extras.get("fill_value", None)
        if "hardmask" in common_params:
            result._hardmask = bool(_extras.get("hard_mask", False))
        return result


arange = _convert2ma(
    'arange',
    params={'fill_value': None, 'hardmask': False},
    np_ret='arange : ndarray',
    np_ma_ret='arange : MaskedArray',
)
clip = _convert2ma(
    'clip',
    params={'fill_value': None, 'hardmask': False},
    np_ret='clipped_array : ndarray',
    np_ma_ret='clipped_array : MaskedArray',
)
empty = _convert2ma(
    'empty',
    params={'fill_value': None, 'hardmask': False},
    np_ret='out : ndarray',
    np_ma_ret='out : MaskedArray',
)
empty_like = _convert2ma(
    'empty_like',
    np_ret='out : ndarray',
    np_ma_ret='out : MaskedArray',
)
frombuffer = _convert2ma(
    'frombuffer',
    np_ret='out : ndarray',
    np_ma_ret='out: MaskedArray',
)
fromfunction = _convert2ma(
   'fromfunction',
   np_ret='fromfunction : any',
   np_ma_ret='fromfunction: MaskedArray',
)
identity = _convert2ma(
    'identity',
    params={'fill_value': None, 'hardmask': False},
    np_ret='out : ndarray',
    np_ma_ret='out : MaskedArray',
)
indices = _convert2ma(
    'indices',
    params={'fill_value': None, 'hardmask': False},
    np_ret='grid : one ndarray or tuple of ndarrays',
    np_ma_ret='grid : one MaskedArray or tuple of MaskedArrays',
)
ones = _convert2ma(
    'ones',
    params={'fill_value': None, 'hardmask': False},
    np_ret='out : ndarray',
    np_ma_ret='out : MaskedArray',
)
ones_like = _convert2ma(
    'ones_like',
    np_ret='out : ndarray',
    np_ma_ret='out : MaskedArray',
)
squeeze = _convert2ma(
    'squeeze',
    params={'fill_value': None, 'hardmask': False},
    np_ret='squeezed : ndarray',
    np_ma_ret='squeezed : MaskedArray',
)
zeros = _convert2ma(
    'zeros',
    params={'fill_value': None, 'hardmask': False},
    np_ret='out : ndarray',
    np_ma_ret='out : MaskedArray',
)
zeros_like = _convert2ma(
    'zeros_like',
    np_ret='out : ndarray',
    np_ma_ret='out : MaskedArray',
)


def append(a, b, axis=None):
    """Append values to the end of an array.

    Parameters
    ----------
    a : array_like
        Values are appended to a copy of this array.
    b : array_like
        These values are appended to a copy of `a`.  It must be of the
        correct shape (the same shape as `a`, excluding `axis`).  If `axis`
        is not specified, `b` can be any shape and will be flattened
        before use.
    axis : int, optional
        The axis along which `v` are appended.  If `axis` is not given,
        both `a` and `b` are flattened before use.

    Returns
    -------
    append : MaskedArray
        A copy of `a` with `b` appended to `axis`.  Note that `append`
        does not occur in-place: a new array is allocated and filled.  If
        `axis` is None, the result is a flattened array.

    See Also
    --------
    numpy.append : Equivalent function in the top-level NumPy module.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> a = ma.masked_values([1, 2, 3], 2)
    >>> b = ma.masked_values([[4, 5, 6], [7, 8, 9]], 7)
    >>> ma.append(a, b)
    masked_array(data=[1, --, 3, 4, 5, 6, --, 8, 9],
                 mask=[False,  True, False, False, False, False,  True, False,
                       False],
           fill_value=999999)
    """
    return concatenate([a, b], axis)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\idna\uts46data.py ===
# This file is automatically generated by tools/idna-data
# vim: set fileencoding=utf-8 :

from typing import List, Tuple, Union

"""IDNA Mapping Table from UTS46."""


__version__ = "15.1.0"


def _seg_0() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x0, "3"),
        (0x1, "3"),
        (0x2, "3"),
        (0x3, "3"),
        (0x4, "3"),
        (0x5, "3"),
        (0x6, "3"),
        (0x7, "3"),
        (0x8, "3"),
        (0x9, "3"),
        (0xA, "3"),
        (0xB, "3"),
        (0xC, "3"),
        (0xD, "3"),
        (0xE, "3"),
        (0xF, "3"),
        (0x10, "3"),
        (0x11, "3"),
        (0x12, "3"),
        (0x13, "3"),
        (0x14, "3"),
        (0x15, "3"),
        (0x16, "3"),
        (0x17, "3"),
        (0x18, "3"),
        (0x19, "3"),
        (0x1A, "3"),
        (0x1B, "3"),
        (0x1C, "3"),
        (0x1D, "3"),
        (0x1E, "3"),
        (0x1F, "3"),
        (0x20, "3"),
        (0x21, "3"),
        (0x22, "3"),
        (0x23, "3"),
        (0x24, "3"),
        (0x25, "3"),
        (0x26, "3"),
        (0x27, "3"),
        (0x28, "3"),
        (0x29, "3"),
        (0x2A, "3"),
        (0x2B, "3"),
        (0x2C, "3"),
        (0x2D, "V"),
        (0x2E, "V"),
        (0x2F, "3"),
        (0x30, "V"),
        (0x31, "V"),
        (0x32, "V"),
        (0x33, "V"),
        (0x34, "V"),
        (0x35, "V"),
        (0x36, "V"),
        (0x37, "V"),
        (0x38, "V"),
        (0x39, "V"),
        (0x3A, "3"),
        (0x3B, "3"),
        (0x3C, "3"),
        (0x3D, "3"),
        (0x3E, "3"),
        (0x3F, "3"),
        (0x40, "3"),
        (0x41, "M", "a"),
        (0x42, "M", "b"),
        (0x43, "M", "c"),
        (0x44, "M", "d"),
        (0x45, "M", "e"),
        (0x46, "M", "f"),
        (0x47, "M", "g"),
        (0x48, "M", "h"),
        (0x49, "M", "i"),
        (0x4A, "M", "j"),
        (0x4B, "M", "k"),
        (0x4C, "M", "l"),
        (0x4D, "M", "m"),
        (0x4E, "M", "n"),
        (0x4F, "M", "o"),
        (0x50, "M", "p"),
        (0x51, "M", "q"),
        (0x52, "M", "r"),
        (0x53, "M", "s"),
        (0x54, "M", "t"),
        (0x55, "M", "u"),
        (0x56, "M", "v"),
        (0x57, "M", "w"),
        (0x58, "M", "x"),
        (0x59, "M", "y"),
        (0x5A, "M", "z"),
        (0x5B, "3"),
        (0x5C, "3"),
        (0x5D, "3"),
        (0x5E, "3"),
        (0x5F, "3"),
        (0x60, "3"),
        (0x61, "V"),
        (0x62, "V"),
        (0x63, "V"),
    ]


def _seg_1() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x64, "V"),
        (0x65, "V"),
        (0x66, "V"),
        (0x67, "V"),
        (0x68, "V"),
        (0x69, "V"),
        (0x6A, "V"),
        (0x6B, "V"),
        (0x6C, "V"),
        (0x6D, "V"),
        (0x6E, "V"),
        (0x6F, "V"),
        (0x70, "V"),
        (0x71, "V"),
        (0x72, "V"),
        (0x73, "V"),
        (0x74, "V"),
        (0x75, "V"),
        (0x76, "V"),
        (0x77, "V"),
        (0x78, "V"),
        (0x79, "V"),
        (0x7A, "V"),
        (0x7B, "3"),
        (0x7C, "3"),
        (0x7D, "3"),
        (0x7E, "3"),
        (0x7F, "3"),
        (0x80, "X"),
        (0x81, "X"),
        (0x82, "X"),
        (0x83, "X"),
        (0x84, "X"),
        (0x85, "X"),
        (0x86, "X"),
        (0x87, "X"),
        (0x88, "X"),
        (0x89, "X"),
        (0x8A, "X"),
        (0x8B, "X"),
        (0x8C, "X"),
        (0x8D, "X"),
        (0x8E, "X"),
        (0x8F, "X"),
        (0x90, "X"),
        (0x91, "X"),
        (0x92, "X"),
        (0x93, "X"),
        (0x94, "X"),
        (0x95, "X"),
        (0x96, "X"),
        (0x97, "X"),
        (0x98, "X"),
        (0x99, "X"),
        (0x9A, "X"),
        (0x9B, "X"),
        (0x9C, "X"),
        (0x9D, "X"),
        (0x9E, "X"),
        (0x9F, "X"),
        (0xA0, "3", " "),
        (0xA1, "V"),
        (0xA2, "V"),
        (0xA3, "V"),
        (0xA4, "V"),
        (0xA5, "V"),
        (0xA6, "V"),
        (0xA7, "V"),
        (0xA8, "3", " ̈"),
        (0xA9, "V"),
        (0xAA, "M", "a"),
        (0xAB, "V"),
        (0xAC, "V"),
        (0xAD, "I"),
        (0xAE, "V"),
        (0xAF, "3", " ̄"),
        (0xB0, "V"),
        (0xB1, "V"),
        (0xB2, "M", "2"),
        (0xB3, "M", "3"),
        (0xB4, "3", " ́"),
        (0xB5, "M", "μ"),
        (0xB6, "V"),
        (0xB7, "V"),
        (0xB8, "3", " ̧"),
        (0xB9, "M", "1"),
        (0xBA, "M", "o"),
        (0xBB, "V"),
        (0xBC, "M", "1⁄4"),
        (0xBD, "M", "1⁄2"),
        (0xBE, "M", "3⁄4"),
        (0xBF, "V"),
        (0xC0, "M", "à"),
        (0xC1, "M", "á"),
        (0xC2, "M", "â"),
        (0xC3, "M", "ã"),
        (0xC4, "M", "ä"),
        (0xC5, "M", "å"),
        (0xC6, "M", "æ"),
        (0xC7, "M", "ç"),
    ]


def _seg_2() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0xC8, "M", "è"),
        (0xC9, "M", "é"),
        (0xCA, "M", "ê"),
        (0xCB, "M", "ë"),
        (0xCC, "M", "ì"),
        (0xCD, "M", "í"),
        (0xCE, "M", "î"),
        (0xCF, "M", "ï"),
        (0xD0, "M", "ð"),
        (0xD1, "M", "ñ"),
        (0xD2, "M", "ò"),
        (0xD3, "M", "ó"),
        (0xD4, "M", "ô"),
        (0xD5, "M", "õ"),
        (0xD6, "M", "ö"),
        (0xD7, "V"),
        (0xD8, "M", "ø"),
        (0xD9, "M", "ù"),
        (0xDA, "M", "ú"),
        (0xDB, "M", "û"),
        (0xDC, "M", "ü"),
        (0xDD, "M", "ý"),
        (0xDE, "M", "þ"),
        (0xDF, "D", "ss"),
        (0xE0, "V"),
        (0xE1, "V"),
        (0xE2, "V"),
        (0xE3, "V"),
        (0xE4, "V"),
        (0xE5, "V"),
        (0xE6, "V"),
        (0xE7, "V"),
        (0xE8, "V"),
        (0xE9, "V"),
        (0xEA, "V"),
        (0xEB, "V"),
        (0xEC, "V"),
        (0xED, "V"),
        (0xEE, "V"),
        (0xEF, "V"),
        (0xF0, "V"),
        (0xF1, "V"),
        (0xF2, "V"),
        (0xF3, "V"),
        (0xF4, "V"),
        (0xF5, "V"),
        (0xF6, "V"),
        (0xF7, "V"),
        (0xF8, "V"),
        (0xF9, "V"),
        (0xFA, "V"),
        (0xFB, "V"),
        (0xFC, "V"),
        (0xFD, "V"),
        (0xFE, "V"),
        (0xFF, "V"),
        (0x100, "M", "ā"),
        (0x101, "V"),
        (0x102, "M", "ă"),
        (0x103, "V"),
        (0x104, "M", "ą"),
        (0x105, "V"),
        (0x106, "M", "ć"),
        (0x107, "V"),
        (0x108, "M", "ĉ"),
        (0x109, "V"),
        (0x10A, "M", "ċ"),
        (0x10B, "V"),
        (0x10C, "M", "č"),
        (0x10D, "V"),
        (0x10E, "M", "ď"),
        (0x10F, "V"),
        (0x110, "M", "đ"),
        (0x111, "V"),
        (0x112, "M", "ē"),
        (0x113, "V"),
        (0x114, "M", "ĕ"),
        (0x115, "V"),
        (0x116, "M", "ė"),
        (0x117, "V"),
        (0x118, "M", "ę"),
        (0x119, "V"),
        (0x11A, "M", "ě"),
        (0x11B, "V"),
        (0x11C, "M", "ĝ"),
        (0x11D, "V"),
        (0x11E, "M", "ğ"),
        (0x11F, "V"),
        (0x120, "M", "ġ"),
        (0x121, "V"),
        (0x122, "M", "ģ"),
        (0x123, "V"),
        (0x124, "M", "ĥ"),
        (0x125, "V"),
        (0x126, "M", "ħ"),
        (0x127, "V"),
        (0x128, "M", "ĩ"),
        (0x129, "V"),
        (0x12A, "M", "ī"),
        (0x12B, "V"),
    ]


def _seg_3() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x12C, "M", "ĭ"),
        (0x12D, "V"),
        (0x12E, "M", "į"),
        (0x12F, "V"),
        (0x130, "M", "i̇"),
        (0x131, "V"),
        (0x132, "M", "ij"),
        (0x134, "M", "ĵ"),
        (0x135, "V"),
        (0x136, "M", "ķ"),
        (0x137, "V"),
        (0x139, "M", "ĺ"),
        (0x13A, "V"),
        (0x13B, "M", "ļ"),
        (0x13C, "V"),
        (0x13D, "M", "ľ"),
        (0x13E, "V"),
        (0x13F, "M", "l·"),
        (0x141, "M", "ł"),
        (0x142, "V"),
        (0x143, "M", "ń"),
        (0x144, "V"),
        (0x145, "M", "ņ"),
        (0x146, "V"),
        (0x147, "M", "ň"),
        (0x148, "V"),
        (0x149, "M", "ʼn"),
        (0x14A, "M", "ŋ"),
        (0x14B, "V"),
        (0x14C, "M", "ō"),
        (0x14D, "V"),
        (0x14E, "M", "ŏ"),
        (0x14F, "V"),
        (0x150, "M", "ő"),
        (0x151, "V"),
        (0x152, "M", "œ"),
        (0x153, "V"),
        (0x154, "M", "ŕ"),
        (0x155, "V"),
        (0x156, "M", "ŗ"),
        (0x157, "V"),
        (0x158, "M", "ř"),
        (0x159, "V"),
        (0x15A, "M", "ś"),
        (0x15B, "V"),
        (0x15C, "M", "ŝ"),
        (0x15D, "V"),
        (0x15E, "M", "ş"),
        (0x15F, "V"),
        (0x160, "M", "š"),
        (0x161, "V"),
        (0x162, "M", "ţ"),
        (0x163, "V"),
        (0x164, "M", "ť"),
        (0x165, "V"),
        (0x166, "M", "ŧ"),
        (0x167, "V"),
        (0x168, "M", "ũ"),
        (0x169, "V"),
        (0x16A, "M", "ū"),
        (0x16B, "V"),
        (0x16C, "M", "ŭ"),
        (0x16D, "V"),
        (0x16E, "M", "ů"),
        (0x16F, "V"),
        (0x170, "M", "ű"),
        (0x171, "V"),
        (0x172, "M", "ų"),
        (0x173, "V"),
        (0x174, "M", "ŵ"),
        (0x175, "V"),
        (0x176, "M", "ŷ"),
        (0x177, "V"),
        (0x178, "M", "ÿ"),
        (0x179, "M", "ź"),
        (0x17A, "V"),
        (0x17B, "M", "ż"),
        (0x17C, "V"),
        (0x17D, "M", "ž"),
        (0x17E, "V"),
        (0x17F, "M", "s"),
        (0x180, "V"),
        (0x181, "M", "ɓ"),
        (0x182, "M", "ƃ"),
        (0x183, "V"),
        (0x184, "M", "ƅ"),
        (0x185, "V"),
        (0x186, "M", "ɔ"),
        (0x187, "M", "ƈ"),
        (0x188, "V"),
        (0x189, "M", "ɖ"),
        (0x18A, "M", "ɗ"),
        (0x18B, "M", "ƌ"),
        (0x18C, "V"),
        (0x18E, "M", "ǝ"),
        (0x18F, "M", "ə"),
        (0x190, "M", "ɛ"),
        (0x191, "M", "ƒ"),
        (0x192, "V"),
        (0x193, "M", "ɠ"),
    ]


def _seg_4() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x194, "M", "ɣ"),
        (0x195, "V"),
        (0x196, "M", "ɩ"),
        (0x197, "M", "ɨ"),
        (0x198, "M", "ƙ"),
        (0x199, "V"),
        (0x19C, "M", "ɯ"),
        (0x19D, "M", "ɲ"),
        (0x19E, "V"),
        (0x19F, "M", "ɵ"),
        (0x1A0, "M", "ơ"),
        (0x1A1, "V"),
        (0x1A2, "M", "ƣ"),
        (0x1A3, "V"),
        (0x1A4, "M", "ƥ"),
        (0x1A5, "V"),
        (0x1A6, "M", "ʀ"),
        (0x1A7, "M", "ƨ"),
        (0x1A8, "V"),
        (0x1A9, "M", "ʃ"),
        (0x1AA, "V"),
        (0x1AC, "M", "ƭ"),
        (0x1AD, "V"),
        (0x1AE, "M", "ʈ"),
        (0x1AF, "M", "ư"),
        (0x1B0, "V"),
        (0x1B1, "M", "ʊ"),
        (0x1B2, "M", "ʋ"),
        (0x1B3, "M", "ƴ"),
        (0x1B4, "V"),
        (0x1B5, "M", "ƶ"),
        (0x1B6, "V"),
        (0x1B7, "M", "ʒ"),
        (0x1B8, "M", "ƹ"),
        (0x1B9, "V"),
        (0x1BC, "M", "ƽ"),
        (0x1BD, "V"),
        (0x1C4, "M", "dž"),
        (0x1C7, "M", "lj"),
        (0x1CA, "M", "nj"),
        (0x1CD, "M", "ǎ"),
        (0x1CE, "V"),
        (0x1CF, "M", "ǐ"),
        (0x1D0, "V"),
        (0x1D1, "M", "ǒ"),
        (0x1D2, "V"),
        (0x1D3, "M", "ǔ"),
        (0x1D4, "V"),
        (0x1D5, "M", "ǖ"),
        (0x1D6, "V"),
        (0x1D7, "M", "ǘ"),
        (0x1D8, "V"),
        (0x1D9, "M", "ǚ"),
        (0x1DA, "V"),
        (0x1DB, "M", "ǜ"),
        (0x1DC, "V"),
        (0x1DE, "M", "ǟ"),
        (0x1DF, "V"),
        (0x1E0, "M", "ǡ"),
        (0x1E1, "V"),
        (0x1E2, "M", "ǣ"),
        (0x1E3, "V"),
        (0x1E4, "M", "ǥ"),
        (0x1E5, "V"),
        (0x1E6, "M", "ǧ"),
        (0x1E7, "V"),
        (0x1E8, "M", "ǩ"),
        (0x1E9, "V"),
        (0x1EA, "M", "ǫ"),
        (0x1EB, "V"),
        (0x1EC, "M", "ǭ"),
        (0x1ED, "V"),
        (0x1EE, "M", "ǯ"),
        (0x1EF, "V"),
        (0x1F1, "M", "dz"),
        (0x1F4, "M", "ǵ"),
        (0x1F5, "V"),
        (0x1F6, "M", "ƕ"),
        (0x1F7, "M", "ƿ"),
        (0x1F8, "M", "ǹ"),
        (0x1F9, "V"),
        (0x1FA, "M", "ǻ"),
        (0x1FB, "V"),
        (0x1FC, "M", "ǽ"),
        (0x1FD, "V"),
        (0x1FE, "M", "ǿ"),
        (0x1FF, "V"),
        (0x200, "M", "ȁ"),
        (0x201, "V"),
        (0x202, "M", "ȃ"),
        (0x203, "V"),
        (0x204, "M", "ȅ"),
        (0x205, "V"),
        (0x206, "M", "ȇ"),
        (0x207, "V"),
        (0x208, "M", "ȉ"),
        (0x209, "V"),
        (0x20A, "M", "ȋ"),
        (0x20B, "V"),
        (0x20C, "M", "ȍ"),
    ]


def _seg_5() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x20D, "V"),
        (0x20E, "M", "ȏ"),
        (0x20F, "V"),
        (0x210, "M", "ȑ"),
        (0x211, "V"),
        (0x212, "M", "ȓ"),
        (0x213, "V"),
        (0x214, "M", "ȕ"),
        (0x215, "V"),
        (0x216, "M", "ȗ"),
        (0x217, "V"),
        (0x218, "M", "ș"),
        (0x219, "V"),
        (0x21A, "M", "ț"),
        (0x21B, "V"),
        (0x21C, "M", "ȝ"),
        (0x21D, "V"),
        (0x21E, "M", "ȟ"),
        (0x21F, "V"),
        (0x220, "M", "ƞ"),
        (0x221, "V"),
        (0x222, "M", "ȣ"),
        (0x223, "V"),
        (0x224, "M", "ȥ"),
        (0x225, "V"),
        (0x226, "M", "ȧ"),
        (0x227, "V"),
        (0x228, "M", "ȩ"),
        (0x229, "V"),
        (0x22A, "M", "ȫ"),
        (0x22B, "V"),
        (0x22C, "M", "ȭ"),
        (0x22D, "V"),
        (0x22E, "M", "ȯ"),
        (0x22F, "V"),
        (0x230, "M", "ȱ"),
        (0x231, "V"),
        (0x232, "M", "ȳ"),
        (0x233, "V"),
        (0x23A, "M", "ⱥ"),
        (0x23B, "M", "ȼ"),
        (0x23C, "V"),
        (0x23D, "M", "ƚ"),
        (0x23E, "M", "ⱦ"),
        (0x23F, "V"),
        (0x241, "M", "ɂ"),
        (0x242, "V"),
        (0x243, "M", "ƀ"),
        (0x244, "M", "ʉ"),
        (0x245, "M", "ʌ"),
        (0x246, "M", "ɇ"),
        (0x247, "V"),
        (0x248, "M", "ɉ"),
        (0x249, "V"),
        (0x24A, "M", "ɋ"),
        (0x24B, "V"),
        (0x24C, "M", "ɍ"),
        (0x24D, "V"),
        (0x24E, "M", "ɏ"),
        (0x24F, "V"),
        (0x2B0, "M", "h"),
        (0x2B1, "M", "ɦ"),
        (0x2B2, "M", "j"),
        (0x2B3, "M", "r"),
        (0x2B4, "M", "ɹ"),
        (0x2B5, "M", "ɻ"),
        (0x2B6, "M", "ʁ"),
        (0x2B7, "M", "w"),
        (0x2B8, "M", "y"),
        (0x2B9, "V"),
        (0x2D8, "3", " ̆"),
        (0x2D9, "3", " ̇"),
        (0x2DA, "3", " ̊"),
        (0x2DB, "3", " ̨"),
        (0x2DC, "3", " ̃"),
        (0x2DD, "3", " ̋"),
        (0x2DE, "V"),
        (0x2E0, "M", "ɣ"),
        (0x2E1, "M", "l"),
        (0x2E2, "M", "s"),
        (0x2E3, "M", "x"),
        (0x2E4, "M", "ʕ"),
        (0x2E5, "V"),
        (0x340, "M", "̀"),
        (0x341, "M", "́"),
        (0x342, "V"),
        (0x343, "M", "̓"),
        (0x344, "M", "̈́"),
        (0x345, "M", "ι"),
        (0x346, "V"),
        (0x34F, "I"),
        (0x350, "V"),
        (0x370, "M", "ͱ"),
        (0x371, "V"),
        (0x372, "M", "ͳ"),
        (0x373, "V"),
        (0x374, "M", "ʹ"),
        (0x375, "V"),
        (0x376, "M", "ͷ"),
        (0x377, "V"),
    ]


def _seg_6() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x378, "X"),
        (0x37A, "3", " ι"),
        (0x37B, "V"),
        (0x37E, "3", ";"),
        (0x37F, "M", "ϳ"),
        (0x380, "X"),
        (0x384, "3", " ́"),
        (0x385, "3", " ̈́"),
        (0x386, "M", "ά"),
        (0x387, "M", "·"),
        (0x388, "M", "έ"),
        (0x389, "M", "ή"),
        (0x38A, "M", "ί"),
        (0x38B, "X"),
        (0x38C, "M", "ό"),
        (0x38D, "X"),
        (0x38E, "M", "ύ"),
        (0x38F, "M", "ώ"),
        (0x390, "V"),
        (0x391, "M", "α"),
        (0x392, "M", "β"),
        (0x393, "M", "γ"),
        (0x394, "M", "δ"),
        (0x395, "M", "ε"),
        (0x396, "M", "ζ"),
        (0x397, "M", "η"),
        (0x398, "M", "θ"),
        (0x399, "M", "ι"),
        (0x39A, "M", "κ"),
        (0x39B, "M", "λ"),
        (0x39C, "M", "μ"),
        (0x39D, "M", "ν"),
        (0x39E, "M", "ξ"),
        (0x39F, "M", "ο"),
        (0x3A0, "M", "π"),
        (0x3A1, "M", "ρ"),
        (0x3A2, "X"),
        (0x3A3, "M", "σ"),
        (0x3A4, "M", "τ"),
        (0x3A5, "M", "υ"),
        (0x3A6, "M", "φ"),
        (0x3A7, "M", "χ"),
        (0x3A8, "M", "ψ"),
        (0x3A9, "M", "ω"),
        (0x3AA, "M", "ϊ"),
        (0x3AB, "M", "ϋ"),
        (0x3AC, "V"),
        (0x3C2, "D", "σ"),
        (0x3C3, "V"),
        (0x3CF, "M", "ϗ"),
        (0x3D0, "M", "β"),
        (0x3D1, "M", "θ"),
        (0x3D2, "M", "υ"),
        (0x3D3, "M", "ύ"),
        (0x3D4, "M", "ϋ"),
        (0x3D5, "M", "φ"),
        (0x3D6, "M", "π"),
        (0x3D7, "V"),
        (0x3D8, "M", "ϙ"),
        (0x3D9, "V"),
        (0x3DA, "M", "ϛ"),
        (0x3DB, "V"),
        (0x3DC, "M", "ϝ"),
        (0x3DD, "V"),
        (0x3DE, "M", "ϟ"),
        (0x3DF, "V"),
        (0x3E0, "M", "ϡ"),
        (0x3E1, "V"),
        (0x3E2, "M", "ϣ"),
        (0x3E3, "V"),
        (0x3E4, "M", "ϥ"),
        (0x3E5, "V"),
        (0x3E6, "M", "ϧ"),
        (0x3E7, "V"),
        (0x3E8, "M", "ϩ"),
        (0x3E9, "V"),
        (0x3EA, "M", "ϫ"),
        (0x3EB, "V"),
        (0x3EC, "M", "ϭ"),
        (0x3ED, "V"),
        (0x3EE, "M", "ϯ"),
        (0x3EF, "V"),
        (0x3F0, "M", "κ"),
        (0x3F1, "M", "ρ"),
        (0x3F2, "M", "σ"),
        (0x3F3, "V"),
        (0x3F4, "M", "θ"),
        (0x3F5, "M", "ε"),
        (0x3F6, "V"),
        (0x3F7, "M", "ϸ"),
        (0x3F8, "V"),
        (0x3F9, "M", "σ"),
        (0x3FA, "M", "ϻ"),
        (0x3FB, "V"),
        (0x3FD, "M", "ͻ"),
        (0x3FE, "M", "ͼ"),
        (0x3FF, "M", "ͽ"),
        (0x400, "M", "ѐ"),
        (0x401, "M", "ё"),
        (0x402, "M", "ђ"),
    ]


def _seg_7() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x403, "M", "ѓ"),
        (0x404, "M", "є"),
        (0x405, "M", "ѕ"),
        (0x406, "M", "і"),
        (0x407, "M", "ї"),
        (0x408, "M", "ј"),
        (0x409, "M", "љ"),
        (0x40A, "M", "њ"),
        (0x40B, "M", "ћ"),
        (0x40C, "M", "ќ"),
        (0x40D, "M", "ѝ"),
        (0x40E, "M", "ў"),
        (0x40F, "M", "џ"),
        (0x410, "M", "а"),
        (0x411, "M", "б"),
        (0x412, "M", "в"),
        (0x413, "M", "г"),
        (0x414, "M", "д"),
        (0x415, "M", "е"),
        (0x416, "M", "ж"),
        (0x417, "M", "з"),
        (0x418, "M", "и"),
        (0x419, "M", "й"),
        (0x41A, "M", "к"),
        (0x41B, "M", "л"),
        (0x41C, "M", "м"),
        (0x41D, "M", "н"),
        (0x41E, "M", "о"),
        (0x41F, "M", "п"),
        (0x420, "M", "р"),
        (0x421, "M", "с"),
        (0x422, "M", "т"),
        (0x423, "M", "у"),
        (0x424, "M", "ф"),
        (0x425, "M", "х"),
        (0x426, "M", "ц"),
        (0x427, "M", "ч"),
        (0x428, "M", "ш"),
        (0x429, "M", "щ"),
        (0x42A, "M", "ъ"),
        (0x42B, "M", "ы"),
        (0x42C, "M", "ь"),
        (0x42D, "M", "э"),
        (0x42E, "M", "ю"),
        (0x42F, "M", "я"),
        (0x430, "V"),
        (0x460, "M", "ѡ"),
        (0x461, "V"),
        (0x462, "M", "ѣ"),
        (0x463, "V"),
        (0x464, "M", "ѥ"),
        (0x465, "V"),
        (0x466, "M", "ѧ"),
        (0x467, "V"),
        (0x468, "M", "ѩ"),
        (0x469, "V"),
        (0x46A, "M", "ѫ"),
        (0x46B, "V"),
        (0x46C, "M", "ѭ"),
        (0x46D, "V"),
        (0x46E, "M", "ѯ"),
        (0x46F, "V"),
        (0x470, "M", "ѱ"),
        (0x471, "V"),
        (0x472, "M", "ѳ"),
        (0x473, "V"),
        (0x474, "M", "ѵ"),
        (0x475, "V"),
        (0x476, "M", "ѷ"),
        (0x477, "V"),
        (0x478, "M", "ѹ"),
        (0x479, "V"),
        (0x47A, "M", "ѻ"),
        (0x47B, "V"),
        (0x47C, "M", "ѽ"),
        (0x47D, "V"),
        (0x47E, "M", "ѿ"),
        (0x47F, "V"),
        (0x480, "M", "ҁ"),
        (0x481, "V"),
        (0x48A, "M", "ҋ"),
        (0x48B, "V"),
        (0x48C, "M", "ҍ"),
        (0x48D, "V"),
        (0x48E, "M", "ҏ"),
        (0x48F, "V"),
        (0x490, "M", "ґ"),
        (0x491, "V"),
        (0x492, "M", "ғ"),
        (0x493, "V"),
        (0x494, "M", "ҕ"),
        (0x495, "V"),
        (0x496, "M", "җ"),
        (0x497, "V"),
        (0x498, "M", "ҙ"),
        (0x499, "V"),
        (0x49A, "M", "қ"),
        (0x49B, "V"),
        (0x49C, "M", "ҝ"),
        (0x49D, "V"),
    ]


def _seg_8() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x49E, "M", "ҟ"),
        (0x49F, "V"),
        (0x4A0, "M", "ҡ"),
        (0x4A1, "V"),
        (0x4A2, "M", "ң"),
        (0x4A3, "V"),
        (0x4A4, "M", "ҥ"),
        (0x4A5, "V"),
        (0x4A6, "M", "ҧ"),
        (0x4A7, "V"),
        (0x4A8, "M", "ҩ"),
        (0x4A9, "V"),
        (0x4AA, "M", "ҫ"),
        (0x4AB, "V"),
        (0x4AC, "M", "ҭ"),
        (0x4AD, "V"),
        (0x4AE, "M", "ү"),
        (0x4AF, "V"),
        (0x4B0, "M", "ұ"),
        (0x4B1, "V"),
        (0x4B2, "M", "ҳ"),
        (0x4B3, "V"),
        (0x4B4, "M", "ҵ"),
        (0x4B5, "V"),
        (0x4B6, "M", "ҷ"),
        (0x4B7, "V"),
        (0x4B8, "M", "ҹ"),
        (0x4B9, "V"),
        (0x4BA, "M", "һ"),
        (0x4BB, "V"),
        (0x4BC, "M", "ҽ"),
        (0x4BD, "V"),
        (0x4BE, "M", "ҿ"),
        (0x4BF, "V"),
        (0x4C0, "X"),
        (0x4C1, "M", "ӂ"),
        (0x4C2, "V"),
        (0x4C3, "M", "ӄ"),
        (0x4C4, "V"),
        (0x4C5, "M", "ӆ"),
        (0x4C6, "V"),
        (0x4C7, "M", "ӈ"),
        (0x4C8, "V"),
        (0x4C9, "M", "ӊ"),
        (0x4CA, "V"),
        (0x4CB, "M", "ӌ"),
        (0x4CC, "V"),
        (0x4CD, "M", "ӎ"),
        (0x4CE, "V"),
        (0x4D0, "M", "ӑ"),
        (0x4D1, "V"),
        (0x4D2, "M", "ӓ"),
        (0x4D3, "V"),
        (0x4D4, "M", "ӕ"),
        (0x4D5, "V"),
        (0x4D6, "M", "ӗ"),
        (0x4D7, "V"),
        (0x4D8, "M", "ә"),
        (0x4D9, "V"),
        (0x4DA, "M", "ӛ"),
        (0x4DB, "V"),
        (0x4DC, "M", "ӝ"),
        (0x4DD, "V"),
        (0x4DE, "M", "ӟ"),
        (0x4DF, "V"),
        (0x4E0, "M", "ӡ"),
        (0x4E1, "V"),
        (0x4E2, "M", "ӣ"),
        (0x4E3, "V"),
        (0x4E4, "M", "ӥ"),
        (0x4E5, "V"),
        (0x4E6, "M", "ӧ"),
        (0x4E7, "V"),
        (0x4E8, "M", "ө"),
        (0x4E9, "V"),
        (0x4EA, "M", "ӫ"),
        (0x4EB, "V"),
        (0x4EC, "M", "ӭ"),
        (0x4ED, "V"),
        (0x4EE, "M", "ӯ"),
        (0x4EF, "V"),
        (0x4F0, "M", "ӱ"),
        (0x4F1, "V"),
        (0x4F2, "M", "ӳ"),
        (0x4F3, "V"),
        (0x4F4, "M", "ӵ"),
        (0x4F5, "V"),
        (0x4F6, "M", "ӷ"),
        (0x4F7, "V"),
        (0x4F8, "M", "ӹ"),
        (0x4F9, "V"),
        (0x4FA, "M", "ӻ"),
        (0x4FB, "V"),
        (0x4FC, "M", "ӽ"),
        (0x4FD, "V"),
        (0x4FE, "M", "ӿ"),
        (0x4FF, "V"),
        (0x500, "M", "ԁ"),
        (0x501, "V"),
        (0x502, "M", "ԃ"),
    ]


def _seg_9() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x503, "V"),
        (0x504, "M", "ԅ"),
        (0x505, "V"),
        (0x506, "M", "ԇ"),
        (0x507, "V"),
        (0x508, "M", "ԉ"),
        (0x509, "V"),
        (0x50A, "M", "ԋ"),
        (0x50B, "V"),
        (0x50C, "M", "ԍ"),
        (0x50D, "V"),
        (0x50E, "M", "ԏ"),
        (0x50F, "V"),
        (0x510, "M", "ԑ"),
        (0x511, "V"),
        (0x512, "M", "ԓ"),
        (0x513, "V"),
        (0x514, "M", "ԕ"),
        (0x515, "V"),
        (0x516, "M", "ԗ"),
        (0x517, "V"),
        (0x518, "M", "ԙ"),
        (0x519, "V"),
        (0x51A, "M", "ԛ"),
        (0x51B, "V"),
        (0x51C, "M", "ԝ"),
        (0x51D, "V"),
        (0x51E, "M", "ԟ"),
        (0x51F, "V"),
        (0x520, "M", "ԡ"),
        (0x521, "V"),
        (0x522, "M", "ԣ"),
        (0x523, "V"),
        (0x524, "M", "ԥ"),
        (0x525, "V"),
        (0x526, "M", "ԧ"),
        (0x527, "V"),
        (0x528, "M", "ԩ"),
        (0x529, "V"),
        (0x52A, "M", "ԫ"),
        (0x52B, "V"),
        (0x52C, "M", "ԭ"),
        (0x52D, "V"),
        (0x52E, "M", "ԯ"),
        (0x52F, "V"),
        (0x530, "X"),
        (0x531, "M", "ա"),
        (0x532, "M", "բ"),
        (0x533, "M", "գ"),
        (0x534, "M", "դ"),
        (0x535, "M", "ե"),
        (0x536, "M", "զ"),
        (0x537, "M", "է"),
        (0x538, "M", "ը"),
        (0x539, "M", "թ"),
        (0x53A, "M", "ժ"),
        (0x53B, "M", "ի"),
        (0x53C, "M", "լ"),
        (0x53D, "M", "խ"),
        (0x53E, "M", "ծ"),
        (0x53F, "M", "կ"),
        (0x540, "M", "հ"),
        (0x541, "M", "ձ"),
        (0x542, "M", "ղ"),
        (0x543, "M", "ճ"),
        (0x544, "M", "մ"),
        (0x545, "M", "յ"),
        (0x546, "M", "ն"),
        (0x547, "M", "շ"),
        (0x548, "M", "ո"),
        (0x549, "M", "չ"),
        (0x54A, "M", "պ"),
        (0x54B, "M", "ջ"),
        (0x54C, "M", "ռ"),
        (0x54D, "M", "ս"),
        (0x54E, "M", "վ"),
        (0x54F, "M", "տ"),
        (0x550, "M", "ր"),
        (0x551, "M", "ց"),
        (0x552, "M", "ւ"),
        (0x553, "M", "փ"),
        (0x554, "M", "ք"),
        (0x555, "M", "օ"),
        (0x556, "M", "ֆ"),
        (0x557, "X"),
        (0x559, "V"),
        (0x587, "M", "եւ"),
        (0x588, "V"),
        (0x58B, "X"),
        (0x58D, "V"),
        (0x590, "X"),
        (0x591, "V"),
        (0x5C8, "X"),
        (0x5D0, "V"),
        (0x5EB, "X"),
        (0x5EF, "V"),
        (0x5F5, "X"),
        (0x606, "V"),
        (0x61C, "X"),
        (0x61D, "V"),
    ]


def _seg_10() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x675, "M", "اٴ"),
        (0x676, "M", "وٴ"),
        (0x677, "M", "ۇٴ"),
        (0x678, "M", "يٴ"),
        (0x679, "V"),
        (0x6DD, "X"),
        (0x6DE, "V"),
        (0x70E, "X"),
        (0x710, "V"),
        (0x74B, "X"),
        (0x74D, "V"),
        (0x7B2, "X"),
        (0x7C0, "V"),
        (0x7FB, "X"),
        (0x7FD, "V"),
        (0x82E, "X"),
        (0x830, "V"),
        (0x83F, "X"),
        (0x840, "V"),
        (0x85C, "X"),
        (0x85E, "V"),
        (0x85F, "X"),
        (0x860, "V"),
        (0x86B, "X"),
        (0x870, "V"),
        (0x88F, "X"),
        (0x898, "V"),
        (0x8E2, "X"),
        (0x8E3, "V"),
        (0x958, "M", "क़"),
        (0x959, "M", "ख़"),
        (0x95A, "M", "ग़"),
        (0x95B, "M", "ज़"),
        (0x95C, "M", "ड़"),
        (0x95D, "M", "ढ़"),
        (0x95E, "M", "फ़"),
        (0x95F, "M", "य़"),
        (0x960, "V"),
        (0x984, "X"),
        (0x985, "V"),
        (0x98D, "X"),
        (0x98F, "V"),
        (0x991, "X"),
        (0x993, "V"),
        (0x9A9, "X"),
        (0x9AA, "V"),
        (0x9B1, "X"),
        (0x9B2, "V"),
        (0x9B3, "X"),
        (0x9B6, "V"),
        (0x9BA, "X"),
        (0x9BC, "V"),
        (0x9C5, "X"),
        (0x9C7, "V"),
        (0x9C9, "X"),
        (0x9CB, "V"),
        (0x9CF, "X"),
        (0x9D7, "V"),
        (0x9D8, "X"),
        (0x9DC, "M", "ড়"),
        (0x9DD, "M", "ঢ়"),
        (0x9DE, "X"),
        (0x9DF, "M", "য়"),
        (0x9E0, "V"),
        (0x9E4, "X"),
        (0x9E6, "V"),
        (0x9FF, "X"),
        (0xA01, "V"),
        (0xA04, "X"),
        (0xA05, "V"),
        (0xA0B, "X"),
        (0xA0F, "V"),
        (0xA11, "X"),
        (0xA13, "V"),
        (0xA29, "X"),
        (0xA2A, "V"),
        (0xA31, "X"),
        (0xA32, "V"),
        (0xA33, "M", "ਲ਼"),
        (0xA34, "X"),
        (0xA35, "V"),
        (0xA36, "M", "ਸ਼"),
        (0xA37, "X"),
        (0xA38, "V"),
        (0xA3A, "X"),
        (0xA3C, "V"),
        (0xA3D, "X"),
        (0xA3E, "V"),
        (0xA43, "X"),
        (0xA47, "V"),
        (0xA49, "X"),
        (0xA4B, "V"),
        (0xA4E, "X"),
        (0xA51, "V"),
        (0xA52, "X"),
        (0xA59, "M", "ਖ਼"),
        (0xA5A, "M", "ਗ਼"),
        (0xA5B, "M", "ਜ਼"),
        (0xA5C, "V"),
        (0xA5D, "X"),
    ]


def _seg_11() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0xA5E, "M", "ਫ਼"),
        (0xA5F, "X"),
        (0xA66, "V"),
        (0xA77, "X"),
        (0xA81, "V"),
        (0xA84, "X"),
        (0xA85, "V"),
        (0xA8E, "X"),
        (0xA8F, "V"),
        (0xA92, "X"),
        (0xA93, "V"),
        (0xAA9, "X"),
        (0xAAA, "V"),
        (0xAB1, "X"),
        (0xAB2, "V"),
        (0xAB4, "X"),
        (0xAB5, "V"),
        (0xABA, "X"),
        (0xABC, "V"),
        (0xAC6, "X"),
        (0xAC7, "V"),
        (0xACA, "X"),
        (0xACB, "V"),
        (0xACE, "X"),
        (0xAD0, "V"),
        (0xAD1, "X"),
        (0xAE0, "V"),
        (0xAE4, "X"),
        (0xAE6, "V"),
        (0xAF2, "X"),
        (0xAF9, "V"),
        (0xB00, "X"),
        (0xB01, "V"),
        (0xB04, "X"),
        (0xB05, "V"),
        (0xB0D, "X"),
        (0xB0F, "V"),
        (0xB11, "X"),
        (0xB13, "V"),
        (0xB29, "X"),
        (0xB2A, "V"),
        (0xB31, "X"),
        (0xB32, "V"),
        (0xB34, "X"),
        (0xB35, "V"),
        (0xB3A, "X"),
        (0xB3C, "V"),
        (0xB45, "X"),
        (0xB47, "V"),
        (0xB49, "X"),
        (0xB4B, "V"),
        (0xB4E, "X"),
        (0xB55, "V"),
        (0xB58, "X"),
        (0xB5C, "M", "ଡ଼"),
        (0xB5D, "M", "ଢ଼"),
        (0xB5E, "X"),
        (0xB5F, "V"),
        (0xB64, "X"),
        (0xB66, "V"),
        (0xB78, "X"),
        (0xB82, "V"),
        (0xB84, "X"),
        (0xB85, "V"),
        (0xB8B, "X"),
        (0xB8E, "V"),
        (0xB91, "X"),
        (0xB92, "V"),
        (0xB96, "X"),
        (0xB99, "V"),
        (0xB9B, "X"),
        (0xB9C, "V"),
        (0xB9D, "X"),
        (0xB9E, "V"),
        (0xBA0, "X"),
        (0xBA3, "V"),
        (0xBA5, "X"),
        (0xBA8, "V"),
        (0xBAB, "X"),
        (0xBAE, "V"),
        (0xBBA, "X"),
        (0xBBE, "V"),
        (0xBC3, "X"),
        (0xBC6, "V"),
        (0xBC9, "X"),
        (0xBCA, "V"),
        (0xBCE, "X"),
        (0xBD0, "V"),
        (0xBD1, "X"),
        (0xBD7, "V"),
        (0xBD8, "X"),
        (0xBE6, "V"),
        (0xBFB, "X"),
        (0xC00, "V"),
        (0xC0D, "X"),
        (0xC0E, "V"),
        (0xC11, "X"),
        (0xC12, "V"),
        (0xC29, "X"),
        (0xC2A, "V"),
    ]


def _seg_12() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0xC3A, "X"),
        (0xC3C, "V"),
        (0xC45, "X"),
        (0xC46, "V"),
        (0xC49, "X"),
        (0xC4A, "V"),
        (0xC4E, "X"),
        (0xC55, "V"),
        (0xC57, "X"),
        (0xC58, "V"),
        (0xC5B, "X"),
        (0xC5D, "V"),
        (0xC5E, "X"),
        (0xC60, "V"),
        (0xC64, "X"),
        (0xC66, "V"),
        (0xC70, "X"),
        (0xC77, "V"),
        (0xC8D, "X"),
        (0xC8E, "V"),
        (0xC91, "X"),
        (0xC92, "V"),
        (0xCA9, "X"),
        (0xCAA, "V"),
        (0xCB4, "X"),
        (0xCB5, "V"),
        (0xCBA, "X"),
        (0xCBC, "V"),
        (0xCC5, "X"),
        (0xCC6, "V"),
        (0xCC9, "X"),
        (0xCCA, "V"),
        (0xCCE, "X"),
        (0xCD5, "V"),
        (0xCD7, "X"),
        (0xCDD, "V"),
        (0xCDF, "X"),
        (0xCE0, "V"),
        (0xCE4, "X"),
        (0xCE6, "V"),
        (0xCF0, "X"),
        (0xCF1, "V"),
        (0xCF4, "X"),
        (0xD00, "V"),
        (0xD0D, "X"),
        (0xD0E, "V"),
        (0xD11, "X"),
        (0xD12, "V"),
        (0xD45, "X"),
        (0xD46, "V"),
        (0xD49, "X"),
        (0xD4A, "V"),
        (0xD50, "X"),
        (0xD54, "V"),
        (0xD64, "X"),
        (0xD66, "V"),
        (0xD80, "X"),
        (0xD81, "V"),
        (0xD84, "X"),
        (0xD85, "V"),
        (0xD97, "X"),
        (0xD9A, "V"),
        (0xDB2, "X"),
        (0xDB3, "V"),
        (0xDBC, "X"),
        (0xDBD, "V"),
        (0xDBE, "X"),
        (0xDC0, "V"),
        (0xDC7, "X"),
        (0xDCA, "V"),
        (0xDCB, "X"),
        (0xDCF, "V"),
        (0xDD5, "X"),
        (0xDD6, "V"),
        (0xDD7, "X"),
        (0xDD8, "V"),
        (0xDE0, "X"),
        (0xDE6, "V"),
        (0xDF0, "X"),
        (0xDF2, "V"),
        (0xDF5, "X"),
        (0xE01, "V"),
        (0xE33, "M", "ํา"),
        (0xE34, "V"),
        (0xE3B, "X"),
        (0xE3F, "V"),
        (0xE5C, "X"),
        (0xE81, "V"),
        (0xE83, "X"),
        (0xE84, "V"),
        (0xE85, "X"),
        (0xE86, "V"),
        (0xE8B, "X"),
        (0xE8C, "V"),
        (0xEA4, "X"),
        (0xEA5, "V"),
        (0xEA6, "X"),
        (0xEA7, "V"),
        (0xEB3, "M", "ໍາ"),
        (0xEB4, "V"),
    ]


def _seg_13() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0xEBE, "X"),
        (0xEC0, "V"),
        (0xEC5, "X"),
        (0xEC6, "V"),
        (0xEC7, "X"),
        (0xEC8, "V"),
        (0xECF, "X"),
        (0xED0, "V"),
        (0xEDA, "X"),
        (0xEDC, "M", "ຫນ"),
        (0xEDD, "M", "ຫມ"),
        (0xEDE, "V"),
        (0xEE0, "X"),
        (0xF00, "V"),
        (0xF0C, "M", "་"),
        (0xF0D, "V"),
        (0xF43, "M", "གྷ"),
        (0xF44, "V"),
        (0xF48, "X"),
        (0xF49, "V"),
        (0xF4D, "M", "ཌྷ"),
        (0xF4E, "V"),
        (0xF52, "M", "དྷ"),
        (0xF53, "V"),
        (0xF57, "M", "བྷ"),
        (0xF58, "V"),
        (0xF5C, "M", "ཛྷ"),
        (0xF5D, "V"),
        (0xF69, "M", "ཀྵ"),
        (0xF6A, "V"),
        (0xF6D, "X"),
        (0xF71, "V"),
        (0xF73, "M", "ཱི"),
        (0xF74, "V"),
        (0xF75, "M", "ཱུ"),
        (0xF76, "M", "ྲྀ"),
        (0xF77, "M", "ྲཱྀ"),
        (0xF78, "M", "ླྀ"),
        (0xF79, "M", "ླཱྀ"),
        (0xF7A, "V"),
        (0xF81, "M", "ཱྀ"),
        (0xF82, "V"),
        (0xF93, "M", "ྒྷ"),
        (0xF94, "V"),
        (0xF98, "X"),
        (0xF99, "V"),
        (0xF9D, "M", "ྜྷ"),
        (0xF9E, "V"),
        (0xFA2, "M", "ྡྷ"),
        (0xFA3, "V"),
        (0xFA7, "M", "ྦྷ"),
        (0xFA8, "V"),
        (0xFAC, "M", "ྫྷ"),
        (0xFAD, "V"),
        (0xFB9, "M", "ྐྵ"),
        (0xFBA, "V"),
        (0xFBD, "X"),
        (0xFBE, "V"),
        (0xFCD, "X"),
        (0xFCE, "V"),
        (0xFDB, "X"),
        (0x1000, "V"),
        (0x10A0, "X"),
        (0x10C7, "M", "ⴧ"),
        (0x10C8, "X"),
        (0x10CD, "M", "ⴭ"),
        (0x10CE, "X"),
        (0x10D0, "V"),
        (0x10FC, "M", "ნ"),
        (0x10FD, "V"),
        (0x115F, "X"),
        (0x1161, "V"),
        (0x1249, "X"),
        (0x124A, "V"),
        (0x124E, "X"),
        (0x1250, "V"),
        (0x1257, "X"),
        (0x1258, "V"),
        (0x1259, "X"),
        (0x125A, "V"),
        (0x125E, "X"),
        (0x1260, "V"),
        (0x1289, "X"),
        (0x128A, "V"),
        (0x128E, "X"),
        (0x1290, "V"),
        (0x12B1, "X"),
        (0x12B2, "V"),
        (0x12B6, "X"),
        (0x12B8, "V"),
        (0x12BF, "X"),
        (0x12C0, "V"),
        (0x12C1, "X"),
        (0x12C2, "V"),
        (0x12C6, "X"),
        (0x12C8, "V"),
        (0x12D7, "X"),
        (0x12D8, "V"),
        (0x1311, "X"),
        (0x1312, "V"),
    ]


def _seg_14() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1316, "X"),
        (0x1318, "V"),
        (0x135B, "X"),
        (0x135D, "V"),
        (0x137D, "X"),
        (0x1380, "V"),
        (0x139A, "X"),
        (0x13A0, "V"),
        (0x13F6, "X"),
        (0x13F8, "M", "Ᏸ"),
        (0x13F9, "M", "Ᏹ"),
        (0x13FA, "M", "Ᏺ"),
        (0x13FB, "M", "Ᏻ"),
        (0x13FC, "M", "Ᏼ"),
        (0x13FD, "M", "Ᏽ"),
        (0x13FE, "X"),
        (0x1400, "V"),
        (0x1680, "X"),
        (0x1681, "V"),
        (0x169D, "X"),
        (0x16A0, "V"),
        (0x16F9, "X"),
        (0x1700, "V"),
        (0x1716, "X"),
        (0x171F, "V"),
        (0x1737, "X"),
        (0x1740, "V"),
        (0x1754, "X"),
        (0x1760, "V"),
        (0x176D, "X"),
        (0x176E, "V"),
        (0x1771, "X"),
        (0x1772, "V"),
        (0x1774, "X"),
        (0x1780, "V"),
        (0x17B4, "X"),
        (0x17B6, "V"),
        (0x17DE, "X"),
        (0x17E0, "V"),
        (0x17EA, "X"),
        (0x17F0, "V"),
        (0x17FA, "X"),
        (0x1800, "V"),
        (0x1806, "X"),
        (0x1807, "V"),
        (0x180B, "I"),
        (0x180E, "X"),
        (0x180F, "I"),
        (0x1810, "V"),
        (0x181A, "X"),
        (0x1820, "V"),
        (0x1879, "X"),
        (0x1880, "V"),
        (0x18AB, "X"),
        (0x18B0, "V"),
        (0x18F6, "X"),
        (0x1900, "V"),
        (0x191F, "X"),
        (0x1920, "V"),
        (0x192C, "X"),
        (0x1930, "V"),
        (0x193C, "X"),
        (0x1940, "V"),
        (0x1941, "X"),
        (0x1944, "V"),
        (0x196E, "X"),
        (0x1970, "V"),
        (0x1975, "X"),
        (0x1980, "V"),
        (0x19AC, "X"),
        (0x19B0, "V"),
        (0x19CA, "X"),
        (0x19D0, "V"),
        (0x19DB, "X"),
        (0x19DE, "V"),
        (0x1A1C, "X"),
        (0x1A1E, "V"),
        (0x1A5F, "X"),
        (0x1A60, "V"),
        (0x1A7D, "X"),
        (0x1A7F, "V"),
        (0x1A8A, "X"),
        (0x1A90, "V"),
        (0x1A9A, "X"),
        (0x1AA0, "V"),
        (0x1AAE, "X"),
        (0x1AB0, "V"),
        (0x1ACF, "X"),
        (0x1B00, "V"),
        (0x1B4D, "X"),
        (0x1B50, "V"),
        (0x1B7F, "X"),
        (0x1B80, "V"),
        (0x1BF4, "X"),
        (0x1BFC, "V"),
        (0x1C38, "X"),
        (0x1C3B, "V"),
        (0x1C4A, "X"),
        (0x1C4D, "V"),
        (0x1C80, "M", "в"),
    ]


def _seg_15() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1C81, "M", "д"),
        (0x1C82, "M", "о"),
        (0x1C83, "M", "с"),
        (0x1C84, "M", "т"),
        (0x1C86, "M", "ъ"),
        (0x1C87, "M", "ѣ"),
        (0x1C88, "M", "ꙋ"),
        (0x1C89, "X"),
        (0x1C90, "M", "ა"),
        (0x1C91, "M", "ბ"),
        (0x1C92, "M", "გ"),
        (0x1C93, "M", "დ"),
        (0x1C94, "M", "ე"),
        (0x1C95, "M", "ვ"),
        (0x1C96, "M", "ზ"),
        (0x1C97, "M", "თ"),
        (0x1C98, "M", "ი"),
        (0x1C99, "M", "კ"),
        (0x1C9A, "M", "ლ"),
        (0x1C9B, "M", "მ"),
        (0x1C9C, "M", "ნ"),
        (0x1C9D, "M", "ო"),
        (0x1C9E, "M", "პ"),
        (0x1C9F, "M", "ჟ"),
        (0x1CA0, "M", "რ"),
        (0x1CA1, "M", "ს"),
        (0x1CA2, "M", "ტ"),
        (0x1CA3, "M", "უ"),
        (0x1CA4, "M", "ფ"),
        (0x1CA5, "M", "ქ"),
        (0x1CA6, "M", "ღ"),
        (0x1CA7, "M", "ყ"),
        (0x1CA8, "M", "შ"),
        (0x1CA9, "M", "ჩ"),
        (0x1CAA, "M", "ც"),
        (0x1CAB, "M", "ძ"),
        (0x1CAC, "M", "წ"),
        (0x1CAD, "M", "ჭ"),
        (0x1CAE, "M", "ხ"),
        (0x1CAF, "M", "ჯ"),
        (0x1CB0, "M", "ჰ"),
        (0x1CB1, "M", "ჱ"),
        (0x1CB2, "M", "ჲ"),
        (0x1CB3, "M", "ჳ"),
        (0x1CB4, "M", "ჴ"),
        (0x1CB5, "M", "ჵ"),
        (0x1CB6, "M", "ჶ"),
        (0x1CB7, "M", "ჷ"),
        (0x1CB8, "M", "ჸ"),
        (0x1CB9, "M", "ჹ"),
        (0x1CBA, "M", "ჺ"),
        (0x1CBB, "X"),
        (0x1CBD, "M", "ჽ"),
        (0x1CBE, "M", "ჾ"),
        (0x1CBF, "M", "ჿ"),
        (0x1CC0, "V"),
        (0x1CC8, "X"),
        (0x1CD0, "V"),
        (0x1CFB, "X"),
        (0x1D00, "V"),
        (0x1D2C, "M", "a"),
        (0x1D2D, "M", "æ"),
        (0x1D2E, "M", "b"),
        (0x1D2F, "V"),
        (0x1D30, "M", "d"),
        (0x1D31, "M", "e"),
        (0x1D32, "M", "ǝ"),
        (0x1D33, "M", "g"),
        (0x1D34, "M", "h"),
        (0x1D35, "M", "i"),
        (0x1D36, "M", "j"),
        (0x1D37, "M", "k"),
        (0x1D38, "M", "l"),
        (0x1D39, "M", "m"),
        (0x1D3A, "M", "n"),
        (0x1D3B, "V"),
        (0x1D3C, "M", "o"),
        (0x1D3D, "M", "ȣ"),
        (0x1D3E, "M", "p"),
        (0x1D3F, "M", "r"),
        (0x1D40, "M", "t"),
        (0x1D41, "M", "u"),
        (0x1D42, "M", "w"),
        (0x1D43, "M", "a"),
        (0x1D44, "M", "ɐ"),
        (0x1D45, "M", "ɑ"),
        (0x1D46, "M", "ᴂ"),
        (0x1D47, "M", "b"),
        (0x1D48, "M", "d"),
        (0x1D49, "M", "e"),
        (0x1D4A, "M", "ə"),
        (0x1D4B, "M", "ɛ"),
        (0x1D4C, "M", "ɜ"),
        (0x1D4D, "M", "g"),
        (0x1D4E, "V"),
        (0x1D4F, "M", "k"),
        (0x1D50, "M", "m"),
        (0x1D51, "M", "ŋ"),
        (0x1D52, "M", "o"),
        (0x1D53, "M", "ɔ"),
    ]


def _seg_16() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1D54, "M", "ᴖ"),
        (0x1D55, "M", "ᴗ"),
        (0x1D56, "M", "p"),
        (0x1D57, "M", "t"),
        (0x1D58, "M", "u"),
        (0x1D59, "M", "ᴝ"),
        (0x1D5A, "M", "ɯ"),
        (0x1D5B, "M", "v"),
        (0x1D5C, "M", "ᴥ"),
        (0x1D5D, "M", "β"),
        (0x1D5E, "M", "γ"),
        (0x1D5F, "M", "δ"),
        (0x1D60, "M", "φ"),
        (0x1D61, "M", "χ"),
        (0x1D62, "M", "i"),
        (0x1D63, "M", "r"),
        (0x1D64, "M", "u"),
        (0x1D65, "M", "v"),
        (0x1D66, "M", "β"),
        (0x1D67, "M", "γ"),
        (0x1D68, "M", "ρ"),
        (0x1D69, "M", "φ"),
        (0x1D6A, "M", "χ"),
        (0x1D6B, "V"),
        (0x1D78, "M", "н"),
        (0x1D79, "V"),
        (0x1D9B, "M", "ɒ"),
        (0x1D9C, "M", "c"),
        (0x1D9D, "M", "ɕ"),
        (0x1D9E, "M", "ð"),
        (0x1D9F, "M", "ɜ"),
        (0x1DA0, "M", "f"),
        (0x1DA1, "M", "ɟ"),
        (0x1DA2, "M", "ɡ"),
        (0x1DA3, "M", "ɥ"),
        (0x1DA4, "M", "ɨ"),
        (0x1DA5, "M", "ɩ"),
        (0x1DA6, "M", "ɪ"),
        (0x1DA7, "M", "ᵻ"),
        (0x1DA8, "M", "ʝ"),
        (0x1DA9, "M", "ɭ"),
        (0x1DAA, "M", "ᶅ"),
        (0x1DAB, "M", "ʟ"),
        (0x1DAC, "M", "ɱ"),
        (0x1DAD, "M", "ɰ"),
        (0x1DAE, "M", "ɲ"),
        (0x1DAF, "M", "ɳ"),
        (0x1DB0, "M", "ɴ"),
        (0x1DB1, "M", "ɵ"),
        (0x1DB2, "M", "ɸ"),
        (0x1DB3, "M", "ʂ"),
        (0x1DB4, "M", "ʃ"),
        (0x1DB5, "M", "ƫ"),
        (0x1DB6, "M", "ʉ"),
        (0x1DB7, "M", "ʊ"),
        (0x1DB8, "M", "ᴜ"),
        (0x1DB9, "M", "ʋ"),
        (0x1DBA, "M", "ʌ"),
        (0x1DBB, "M", "z"),
        (0x1DBC, "M", "ʐ"),
        (0x1DBD, "M", "ʑ"),
        (0x1DBE, "M", "ʒ"),
        (0x1DBF, "M", "θ"),
        (0x1DC0, "V"),
        (0x1E00, "M", "ḁ"),
        (0x1E01, "V"),
        (0x1E02, "M", "ḃ"),
        (0x1E03, "V"),
        (0x1E04, "M", "ḅ"),
        (0x1E05, "V"),
        (0x1E06, "M", "ḇ"),
        (0x1E07, "V"),
        (0x1E08, "M", "ḉ"),
        (0x1E09, "V"),
        (0x1E0A, "M", "ḋ"),
        (0x1E0B, "V"),
        (0x1E0C, "M", "ḍ"),
        (0x1E0D, "V"),
        (0x1E0E, "M", "ḏ"),
        (0x1E0F, "V"),
        (0x1E10, "M", "ḑ"),
        (0x1E11, "V"),
        (0x1E12, "M", "ḓ"),
        (0x1E13, "V"),
        (0x1E14, "M", "ḕ"),
        (0x1E15, "V"),
        (0x1E16, "M", "ḗ"),
        (0x1E17, "V"),
        (0x1E18, "M", "ḙ"),
        (0x1E19, "V"),
        (0x1E1A, "M", "ḛ"),
        (0x1E1B, "V"),
        (0x1E1C, "M", "ḝ"),
        (0x1E1D, "V"),
        (0x1E1E, "M", "ḟ"),
        (0x1E1F, "V"),
        (0x1E20, "M", "ḡ"),
        (0x1E21, "V"),
        (0x1E22, "M", "ḣ"),
        (0x1E23, "V"),
    ]


def _seg_17() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1E24, "M", "ḥ"),
        (0x1E25, "V"),
        (0x1E26, "M", "ḧ"),
        (0x1E27, "V"),
        (0x1E28, "M", "ḩ"),
        (0x1E29, "V"),
        (0x1E2A, "M", "ḫ"),
        (0x1E2B, "V"),
        (0x1E2C, "M", "ḭ"),
        (0x1E2D, "V"),
        (0x1E2E, "M", "ḯ"),
        (0x1E2F, "V"),
        (0x1E30, "M", "ḱ"),
        (0x1E31, "V"),
        (0x1E32, "M", "ḳ"),
        (0x1E33, "V"),
        (0x1E34, "M", "ḵ"),
        (0x1E35, "V"),
        (0x1E36, "M", "ḷ"),
        (0x1E37, "V"),
        (0x1E38, "M", "ḹ"),
        (0x1E39, "V"),
        (0x1E3A, "M", "ḻ"),
        (0x1E3B, "V"),
        (0x1E3C, "M", "ḽ"),
        (0x1E3D, "V"),
        (0x1E3E, "M", "ḿ"),
        (0x1E3F, "V"),
        (0x1E40, "M", "ṁ"),
        (0x1E41, "V"),
        (0x1E42, "M", "ṃ"),
        (0x1E43, "V"),
        (0x1E44, "M", "ṅ"),
        (0x1E45, "V"),
        (0x1E46, "M", "ṇ"),
        (0x1E47, "V"),
        (0x1E48, "M", "ṉ"),
        (0x1E49, "V"),
        (0x1E4A, "M", "ṋ"),
        (0x1E4B, "V"),
        (0x1E4C, "M", "ṍ"),
        (0x1E4D, "V"),
        (0x1E4E, "M", "ṏ"),
        (0x1E4F, "V"),
        (0x1E50, "M", "ṑ"),
        (0x1E51, "V"),
        (0x1E52, "M", "ṓ"),
        (0x1E53, "V"),
        (0x1E54, "M", "ṕ"),
        (0x1E55, "V"),
        (0x1E56, "M", "ṗ"),
        (0x1E57, "V"),
        (0x1E58, "M", "ṙ"),
        (0x1E59, "V"),
        (0x1E5A, "M", "ṛ"),
        (0x1E5B, "V"),
        (0x1E5C, "M", "ṝ"),
        (0x1E5D, "V"),
        (0x1E5E, "M", "ṟ"),
        (0x1E5F, "V"),
        (0x1E60, "M", "ṡ"),
        (0x1E61, "V"),
        (0x1E62, "M", "ṣ"),
        (0x1E63, "V"),
        (0x1E64, "M", "ṥ"),
        (0x1E65, "V"),
        (0x1E66, "M", "ṧ"),
        (0x1E67, "V"),
        (0x1E68, "M", "ṩ"),
        (0x1E69, "V"),
        (0x1E6A, "M", "ṫ"),
        (0x1E6B, "V"),
        (0x1E6C, "M", "ṭ"),
        (0x1E6D, "V"),
        (0x1E6E, "M", "ṯ"),
        (0x1E6F, "V"),
        (0x1E70, "M", "ṱ"),
        (0x1E71, "V"),
        (0x1E72, "M", "ṳ"),
        (0x1E73, "V"),
        (0x1E74, "M", "ṵ"),
        (0x1E75, "V"),
        (0x1E76, "M", "ṷ"),
        (0x1E77, "V"),
        (0x1E78, "M", "ṹ"),
        (0x1E79, "V"),
        (0x1E7A, "M", "ṻ"),
        (0x1E7B, "V"),
        (0x1E7C, "M", "ṽ"),
        (0x1E7D, "V"),
        (0x1E7E, "M", "ṿ"),
        (0x1E7F, "V"),
        (0x1E80, "M", "ẁ"),
        (0x1E81, "V"),
        (0x1E82, "M", "ẃ"),
        (0x1E83, "V"),
        (0x1E84, "M", "ẅ"),
        (0x1E85, "V"),
        (0x1E86, "M", "ẇ"),
        (0x1E87, "V"),
    ]


def _seg_18() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1E88, "M", "ẉ"),
        (0x1E89, "V"),
        (0x1E8A, "M", "ẋ"),
        (0x1E8B, "V"),
        (0x1E8C, "M", "ẍ"),
        (0x1E8D, "V"),
        (0x1E8E, "M", "ẏ"),
        (0x1E8F, "V"),
        (0x1E90, "M", "ẑ"),
        (0x1E91, "V"),
        (0x1E92, "M", "ẓ"),
        (0x1E93, "V"),
        (0x1E94, "M", "ẕ"),
        (0x1E95, "V"),
        (0x1E9A, "M", "aʾ"),
        (0x1E9B, "M", "ṡ"),
        (0x1E9C, "V"),
        (0x1E9E, "M", "ß"),
        (0x1E9F, "V"),
        (0x1EA0, "M", "ạ"),
        (0x1EA1, "V"),
        (0x1EA2, "M", "ả"),
        (0x1EA3, "V"),
        (0x1EA4, "M", "ấ"),
        (0x1EA5, "V"),
        (0x1EA6, "M", "ầ"),
        (0x1EA7, "V"),
        (0x1EA8, "M", "ẩ"),
        (0x1EA9, "V"),
        (0x1EAA, "M", "ẫ"),
        (0x1EAB, "V"),
        (0x1EAC, "M", "ậ"),
        (0x1EAD, "V"),
        (0x1EAE, "M", "ắ"),
        (0x1EAF, "V"),
        (0x1EB0, "M", "ằ"),
        (0x1EB1, "V"),
        (0x1EB2, "M", "ẳ"),
        (0x1EB3, "V"),
        (0x1EB4, "M", "ẵ"),
        (0x1EB5, "V"),
        (0x1EB6, "M", "ặ"),
        (0x1EB7, "V"),
        (0x1EB8, "M", "ẹ"),
        (0x1EB9, "V"),
        (0x1EBA, "M", "ẻ"),
        (0x1EBB, "V"),
        (0x1EBC, "M", "ẽ"),
        (0x1EBD, "V"),
        (0x1EBE, "M", "ế"),
        (0x1EBF, "V"),
        (0x1EC0, "M", "ề"),
        (0x1EC1, "V"),
        (0x1EC2, "M", "ể"),
        (0x1EC3, "V"),
        (0x1EC4, "M", "ễ"),
        (0x1EC5, "V"),
        (0x1EC6, "M", "ệ"),
        (0x1EC7, "V"),
        (0x1EC8, "M", "ỉ"),
        (0x1EC9, "V"),
        (0x1ECA, "M", "ị"),
        (0x1ECB, "V"),
        (0x1ECC, "M", "ọ"),
        (0x1ECD, "V"),
        (0x1ECE, "M", "ỏ"),
        (0x1ECF, "V"),
        (0x1ED0, "M", "ố"),
        (0x1ED1, "V"),
        (0x1ED2, "M", "ồ"),
        (0x1ED3, "V"),
        (0x1ED4, "M", "ổ"),
        (0x1ED5, "V"),
        (0x1ED6, "M", "ỗ"),
        (0x1ED7, "V"),
        (0x1ED8, "M", "ộ"),
        (0x1ED9, "V"),
        (0x1EDA, "M", "ớ"),
        (0x1EDB, "V"),
        (0x1EDC, "M", "ờ"),
        (0x1EDD, "V"),
        (0x1EDE, "M", "ở"),
        (0x1EDF, "V"),
        (0x1EE0, "M", "ỡ"),
        (0x1EE1, "V"),
        (0x1EE2, "M", "ợ"),
        (0x1EE3, "V"),
        (0x1EE4, "M", "ụ"),
        (0x1EE5, "V"),
        (0x1EE6, "M", "ủ"),
        (0x1EE7, "V"),
        (0x1EE8, "M", "ứ"),
        (0x1EE9, "V"),
        (0x1EEA, "M", "ừ"),
        (0x1EEB, "V"),
        (0x1EEC, "M", "ử"),
        (0x1EED, "V"),
        (0x1EEE, "M", "ữ"),
        (0x1EEF, "V"),
        (0x1EF0, "M", "ự"),
    ]


def _seg_19() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1EF1, "V"),
        (0x1EF2, "M", "ỳ"),
        (0x1EF3, "V"),
        (0x1EF4, "M", "ỵ"),
        (0x1EF5, "V"),
        (0x1EF6, "M", "ỷ"),
        (0x1EF7, "V"),
        (0x1EF8, "M", "ỹ"),
        (0x1EF9, "V"),
        (0x1EFA, "M", "ỻ"),
        (0x1EFB, "V"),
        (0x1EFC, "M", "ỽ"),
        (0x1EFD, "V"),
        (0x1EFE, "M", "ỿ"),
        (0x1EFF, "V"),
        (0x1F08, "M", "ἀ"),
        (0x1F09, "M", "ἁ"),
        (0x1F0A, "M", "ἂ"),
        (0x1F0B, "M", "ἃ"),
        (0x1F0C, "M", "ἄ"),
        (0x1F0D, "M", "ἅ"),
        (0x1F0E, "M", "ἆ"),
        (0x1F0F, "M", "ἇ"),
        (0x1F10, "V"),
        (0x1F16, "X"),
        (0x1F18, "M", "ἐ"),
        (0x1F19, "M", "ἑ"),
        (0x1F1A, "M", "ἒ"),
        (0x1F1B, "M", "ἓ"),
        (0x1F1C, "M", "ἔ"),
        (0x1F1D, "M", "ἕ"),
        (0x1F1E, "X"),
        (0x1F20, "V"),
        (0x1F28, "M", "ἠ"),
        (0x1F29, "M", "ἡ"),
        (0x1F2A, "M", "ἢ"),
        (0x1F2B, "M", "ἣ"),
        (0x1F2C, "M", "ἤ"),
        (0x1F2D, "M", "ἥ"),
        (0x1F2E, "M", "ἦ"),
        (0x1F2F, "M", "ἧ"),
        (0x1F30, "V"),
        (0x1F38, "M", "ἰ"),
        (0x1F39, "M", "ἱ"),
        (0x1F3A, "M", "ἲ"),
        (0x1F3B, "M", "ἳ"),
        (0x1F3C, "M", "ἴ"),
        (0x1F3D, "M", "ἵ"),
        (0x1F3E, "M", "ἶ"),
        (0x1F3F, "M", "ἷ"),
        (0x1F40, "V"),
        (0x1F46, "X"),
        (0x1F48, "M", "ὀ"),
        (0x1F49, "M", "ὁ"),
        (0x1F4A, "M", "ὂ"),
        (0x1F4B, "M", "ὃ"),
        (0x1F4C, "M", "ὄ"),
        (0x1F4D, "M", "ὅ"),
        (0x1F4E, "X"),
        (0x1F50, "V"),
        (0x1F58, "X"),
        (0x1F59, "M", "ὑ"),
        (0x1F5A, "X"),
        (0x1F5B, "M", "ὓ"),
        (0x1F5C, "X"),
        (0x1F5D, "M", "ὕ"),
        (0x1F5E, "X"),
        (0x1F5F, "M", "ὗ"),
        (0x1F60, "V"),
        (0x1F68, "M", "ὠ"),
        (0x1F69, "M", "ὡ"),
        (0x1F6A, "M", "ὢ"),
        (0x1F6B, "M", "ὣ"),
        (0x1F6C, "M", "ὤ"),
        (0x1F6D, "M", "ὥ"),
        (0x1F6E, "M", "ὦ"),
        (0x1F6F, "M", "ὧ"),
        (0x1F70, "V"),
        (0x1F71, "M", "ά"),
        (0x1F72, "V"),
        (0x1F73, "M", "έ"),
        (0x1F74, "V"),
        (0x1F75, "M", "ή"),
        (0x1F76, "V"),
        (0x1F77, "M", "ί"),
        (0x1F78, "V"),
        (0x1F79, "M", "ό"),
        (0x1F7A, "V"),
        (0x1F7B, "M", "ύ"),
        (0x1F7C, "V"),
        (0x1F7D, "M", "ώ"),
        (0x1F7E, "X"),
        (0x1F80, "M", "ἀι"),
        (0x1F81, "M", "ἁι"),
        (0x1F82, "M", "ἂι"),
        (0x1F83, "M", "ἃι"),
        (0x1F84, "M", "ἄι"),
        (0x1F85, "M", "ἅι"),
        (0x1F86, "M", "ἆι"),
        (0x1F87, "M", "ἇι"),
    ]


def _seg_20() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1F88, "M", "ἀι"),
        (0x1F89, "M", "ἁι"),
        (0x1F8A, "M", "ἂι"),
        (0x1F8B, "M", "ἃι"),
        (0x1F8C, "M", "ἄι"),
        (0x1F8D, "M", "ἅι"),
        (0x1F8E, "M", "ἆι"),
        (0x1F8F, "M", "ἇι"),
        (0x1F90, "M", "ἠι"),
        (0x1F91, "M", "ἡι"),
        (0x1F92, "M", "ἢι"),
        (0x1F93, "M", "ἣι"),
        (0x1F94, "M", "ἤι"),
        (0x1F95, "M", "ἥι"),
        (0x1F96, "M", "ἦι"),
        (0x1F97, "M", "ἧι"),
        (0x1F98, "M", "ἠι"),
        (0x1F99, "M", "ἡι"),
        (0x1F9A, "M", "ἢι"),
        (0x1F9B, "M", "ἣι"),
        (0x1F9C, "M", "ἤι"),
        (0x1F9D, "M", "ἥι"),
        (0x1F9E, "M", "ἦι"),
        (0x1F9F, "M", "ἧι"),
        (0x1FA0, "M", "ὠι"),
        (0x1FA1, "M", "ὡι"),
        (0x1FA2, "M", "ὢι"),
        (0x1FA3, "M", "ὣι"),
        (0x1FA4, "M", "ὤι"),
        (0x1FA5, "M", "ὥι"),
        (0x1FA6, "M", "ὦι"),
        (0x1FA7, "M", "ὧι"),
        (0x1FA8, "M", "ὠι"),
        (0x1FA9, "M", "ὡι"),
        (0x1FAA, "M", "ὢι"),
        (0x1FAB, "M", "ὣι"),
        (0x1FAC, "M", "ὤι"),
        (0x1FAD, "M", "ὥι"),
        (0x1FAE, "M", "ὦι"),
        (0x1FAF, "M", "ὧι"),
        (0x1FB0, "V"),
        (0x1FB2, "M", "ὰι"),
        (0x1FB3, "M", "αι"),
        (0x1FB4, "M", "άι"),
        (0x1FB5, "X"),
        (0x1FB6, "V"),
        (0x1FB7, "M", "ᾶι"),
        (0x1FB8, "M", "ᾰ"),
        (0x1FB9, "M", "ᾱ"),
        (0x1FBA, "M", "ὰ"),
        (0x1FBB, "M", "ά"),
        (0x1FBC, "M", "αι"),
        (0x1FBD, "3", " ̓"),
        (0x1FBE, "M", "ι"),
        (0x1FBF, "3", " ̓"),
        (0x1FC0, "3", " ͂"),
        (0x1FC1, "3", " ̈͂"),
        (0x1FC2, "M", "ὴι"),
        (0x1FC3, "M", "ηι"),
        (0x1FC4, "M", "ήι"),
        (0x1FC5, "X"),
        (0x1FC6, "V"),
        (0x1FC7, "M", "ῆι"),
        (0x1FC8, "M", "ὲ"),
        (0x1FC9, "M", "έ"),
        (0x1FCA, "M", "ὴ"),
        (0x1FCB, "M", "ή"),
        (0x1FCC, "M", "ηι"),
        (0x1FCD, "3", " ̓̀"),
        (0x1FCE, "3", " ̓́"),
        (0x1FCF, "3", " ̓͂"),
        (0x1FD0, "V"),
        (0x1FD3, "M", "ΐ"),
        (0x1FD4, "X"),
        (0x1FD6, "V"),
        (0x1FD8, "M", "ῐ"),
        (0x1FD9, "M", "ῑ"),
        (0x1FDA, "M", "ὶ"),
        (0x1FDB, "M", "ί"),
        (0x1FDC, "X"),
        (0x1FDD, "3", " ̔̀"),
        (0x1FDE, "3", " ̔́"),
        (0x1FDF, "3", " ̔͂"),
        (0x1FE0, "V"),
        (0x1FE3, "M", "ΰ"),
        (0x1FE4, "V"),
        (0x1FE8, "M", "ῠ"),
        (0x1FE9, "M", "ῡ"),
        (0x1FEA, "M", "ὺ"),
        (0x1FEB, "M", "ύ"),
        (0x1FEC, "M", "ῥ"),
        (0x1FED, "3", " ̈̀"),
        (0x1FEE, "3", " ̈́"),
        (0x1FEF, "3", "`"),
        (0x1FF0, "X"),
        (0x1FF2, "M", "ὼι"),
        (0x1FF3, "M", "ωι"),
        (0x1FF4, "M", "ώι"),
        (0x1FF5, "X"),
        (0x1FF6, "V"),
    ]


def _seg_21() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1FF7, "M", "ῶι"),
        (0x1FF8, "M", "ὸ"),
        (0x1FF9, "M", "ό"),
        (0x1FFA, "M", "ὼ"),
        (0x1FFB, "M", "ώ"),
        (0x1FFC, "M", "ωι"),
        (0x1FFD, "3", " ́"),
        (0x1FFE, "3", " ̔"),
        (0x1FFF, "X"),
        (0x2000, "3", " "),
        (0x200B, "I"),
        (0x200C, "D", ""),
        (0x200E, "X"),
        (0x2010, "V"),
        (0x2011, "M", "‐"),
        (0x2012, "V"),
        (0x2017, "3", " ̳"),
        (0x2018, "V"),
        (0x2024, "X"),
        (0x2027, "V"),
        (0x2028, "X"),
        (0x202F, "3", " "),
        (0x2030, "V"),
        (0x2033, "M", "′′"),
        (0x2034, "M", "′′′"),
        (0x2035, "V"),
        (0x2036, "M", "‵‵"),
        (0x2037, "M", "‵‵‵"),
        (0x2038, "V"),
        (0x203C, "3", "!!"),
        (0x203D, "V"),
        (0x203E, "3", " ̅"),
        (0x203F, "V"),
        (0x2047, "3", "??"),
        (0x2048, "3", "?!"),
        (0x2049, "3", "!?"),
        (0x204A, "V"),
        (0x2057, "M", "′′′′"),
        (0x2058, "V"),
        (0x205F, "3", " "),
        (0x2060, "I"),
        (0x2061, "X"),
        (0x2064, "I"),
        (0x2065, "X"),
        (0x2070, "M", "0"),
        (0x2071, "M", "i"),
        (0x2072, "X"),
        (0x2074, "M", "4"),
        (0x2075, "M", "5"),
        (0x2076, "M", "6"),
        (0x2077, "M", "7"),
        (0x2078, "M", "8"),
        (0x2079, "M", "9"),
        (0x207A, "3", "+"),
        (0x207B, "M", "−"),
        (0x207C, "3", "="),
        (0x207D, "3", "("),
        (0x207E, "3", ")"),
        (0x207F, "M", "n"),
        (0x2080, "M", "0"),
        (0x2081, "M", "1"),
        (0x2082, "M", "2"),
        (0x2083, "M", "3"),
        (0x2084, "M", "4"),
        (0x2085, "M", "5"),
        (0x2086, "M", "6"),
        (0x2087, "M", "7"),
        (0x2088, "M", "8"),
        (0x2089, "M", "9"),
        (0x208A, "3", "+"),
        (0x208B, "M", "−"),
        (0x208C, "3", "="),
        (0x208D, "3", "("),
        (0x208E, "3", ")"),
        (0x208F, "X"),
        (0x2090, "M", "a"),
        (0x2091, "M", "e"),
        (0x2092, "M", "o"),
        (0x2093, "M", "x"),
        (0x2094, "M", "ə"),
        (0x2095, "M", "h"),
        (0x2096, "M", "k"),
        (0x2097, "M", "l"),
        (0x2098, "M", "m"),
        (0x2099, "M", "n"),
        (0x209A, "M", "p"),
        (0x209B, "M", "s"),
        (0x209C, "M", "t"),
        (0x209D, "X"),
        (0x20A0, "V"),
        (0x20A8, "M", "rs"),
        (0x20A9, "V"),
        (0x20C1, "X"),
        (0x20D0, "V"),
        (0x20F1, "X"),
        (0x2100, "3", "a/c"),
        (0x2101, "3", "a/s"),
        (0x2102, "M", "c"),
        (0x2103, "M", "°c"),
        (0x2104, "V"),
    ]


def _seg_22() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x2105, "3", "c/o"),
        (0x2106, "3", "c/u"),
        (0x2107, "M", "ɛ"),
        (0x2108, "V"),
        (0x2109, "M", "°f"),
        (0x210A, "M", "g"),
        (0x210B, "M", "h"),
        (0x210F, "M", "ħ"),
        (0x2110, "M", "i"),
        (0x2112, "M", "l"),
        (0x2114, "V"),
        (0x2115, "M", "n"),
        (0x2116, "M", "no"),
        (0x2117, "V"),
        (0x2119, "M", "p"),
        (0x211A, "M", "q"),
        (0x211B, "M", "r"),
        (0x211E, "V"),
        (0x2120, "M", "sm"),
        (0x2121, "M", "tel"),
        (0x2122, "M", "tm"),
        (0x2123, "V"),
        (0x2124, "M", "z"),
        (0x2125, "V"),
        (0x2126, "M", "ω"),
        (0x2127, "V"),
        (0x2128, "M", "z"),
        (0x2129, "V"),
        (0x212A, "M", "k"),
        (0x212B, "M", "å"),
        (0x212C, "M", "b"),
        (0x212D, "M", "c"),
        (0x212E, "V"),
        (0x212F, "M", "e"),
        (0x2131, "M", "f"),
        (0x2132, "X"),
        (0x2133, "M", "m"),
        (0x2134, "M", "o"),
        (0x2135, "M", "א"),
        (0x2136, "M", "ב"),
        (0x2137, "M", "ג"),
        (0x2138, "M", "ד"),
        (0x2139, "M", "i"),
        (0x213A, "V"),
        (0x213B, "M", "fax"),
        (0x213C, "M", "π"),
        (0x213D, "M", "γ"),
        (0x213F, "M", "π"),
        (0x2140, "M", "∑"),
        (0x2141, "V"),
        (0x2145, "M", "d"),
        (0x2147, "M", "e"),
        (0x2148, "M", "i"),
        (0x2149, "M", "j"),
        (0x214A, "V"),
        (0x2150, "M", "1⁄7"),
        (0x2151, "M", "1⁄9"),
        (0x2152, "M", "1⁄10"),
        (0x2153, "M", "1⁄3"),
        (0x2154, "M", "2⁄3"),
        (0x2155, "M", "1⁄5"),
        (0x2156, "M", "2⁄5"),
        (0x2157, "M", "3⁄5"),
        (0x2158, "M", "4⁄5"),
        (0x2159, "M", "1⁄6"),
        (0x215A, "M", "5⁄6"),
        (0x215B, "M", "1⁄8"),
        (0x215C, "M", "3⁄8"),
        (0x215D, "M", "5⁄8"),
        (0x215E, "M", "7⁄8"),
        (0x215F, "M", "1⁄"),
        (0x2160, "M", "i"),
        (0x2161, "M", "ii"),
        (0x2162, "M", "iii"),
        (0x2163, "M", "iv"),
        (0x2164, "M", "v"),
        (0x2165, "M", "vi"),
        (0x2166, "M", "vii"),
        (0x2167, "M", "viii"),
        (0x2168, "M", "ix"),
        (0x2169, "M", "x"),
        (0x216A, "M", "xi"),
        (0x216B, "M", "xii"),
        (0x216C, "M", "l"),
        (0x216D, "M", "c"),
        (0x216E, "M", "d"),
        (0x216F, "M", "m"),
        (0x2170, "M", "i"),
        (0x2171, "M", "ii"),
        (0x2172, "M", "iii"),
        (0x2173, "M", "iv"),
        (0x2174, "M", "v"),
        (0x2175, "M", "vi"),
        (0x2176, "M", "vii"),
        (0x2177, "M", "viii"),
        (0x2178, "M", "ix"),
        (0x2179, "M", "x"),
        (0x217A, "M", "xi"),
        (0x217B, "M", "xii"),
        (0x217C, "M", "l"),
    ]


def _seg_23() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x217D, "M", "c"),
        (0x217E, "M", "d"),
        (0x217F, "M", "m"),
        (0x2180, "V"),
        (0x2183, "X"),
        (0x2184, "V"),
        (0x2189, "M", "0⁄3"),
        (0x218A, "V"),
        (0x218C, "X"),
        (0x2190, "V"),
        (0x222C, "M", "∫∫"),
        (0x222D, "M", "∫∫∫"),
        (0x222E, "V"),
        (0x222F, "M", "∮∮"),
        (0x2230, "M", "∮∮∮"),
        (0x2231, "V"),
        (0x2329, "M", "〈"),
        (0x232A, "M", "〉"),
        (0x232B, "V"),
        (0x2427, "X"),
        (0x2440, "V"),
        (0x244B, "X"),
        (0x2460, "M", "1"),
        (0x2461, "M", "2"),
        (0x2462, "M", "3"),
        (0x2463, "M", "4"),
        (0x2464, "M", "5"),
        (0x2465, "M", "6"),
        (0x2466, "M", "7"),
        (0x2467, "M", "8"),
        (0x2468, "M", "9"),
        (0x2469, "M", "10"),
        (0x246A, "M", "11"),
        (0x246B, "M", "12"),
        (0x246C, "M", "13"),
        (0x246D, "M", "14"),
        (0x246E, "M", "15"),
        (0x246F, "M", "16"),
        (0x2470, "M", "17"),
        (0x2471, "M", "18"),
        (0x2472, "M", "19"),
        (0x2473, "M", "20"),
        (0x2474, "3", "(1)"),
        (0x2475, "3", "(2)"),
        (0x2476, "3", "(3)"),
        (0x2477, "3", "(4)"),
        (0x2478, "3", "(5)"),
        (0x2479, "3", "(6)"),
        (0x247A, "3", "(7)"),
        (0x247B, "3", "(8)"),
        (0x247C, "3", "(9)"),
        (0x247D, "3", "(10)"),
        (0x247E, "3", "(11)"),
        (0x247F, "3", "(12)"),
        (0x2480, "3", "(13)"),
        (0x2481, "3", "(14)"),
        (0x2482, "3", "(15)"),
        (0x2483, "3", "(16)"),
        (0x2484, "3", "(17)"),
        (0x2485, "3", "(18)"),
        (0x2486, "3", "(19)"),
        (0x2487, "3", "(20)"),
        (0x2488, "X"),
        (0x249C, "3", "(a)"),
        (0x249D, "3", "(b)"),
        (0x249E, "3", "(c)"),
        (0x249F, "3", "(d)"),
        (0x24A0, "3", "(e)"),
        (0x24A1, "3", "(f)"),
        (0x24A2, "3", "(g)"),
        (0x24A3, "3", "(h)"),
        (0x24A4, "3", "(i)"),
        (0x24A5, "3", "(j)"),
        (0x24A6, "3", "(k)"),
        (0x24A7, "3", "(l)"),
        (0x24A8, "3", "(m)"),
        (0x24A9, "3", "(n)"),
        (0x24AA, "3", "(o)"),
        (0x24AB, "3", "(p)"),
        (0x24AC, "3", "(q)"),
        (0x24AD, "3", "(r)"),
        (0x24AE, "3", "(s)"),
        (0x24AF, "3", "(t)"),
        (0x24B0, "3", "(u)"),
        (0x24B1, "3", "(v)"),
        (0x24B2, "3", "(w)"),
        (0x24B3, "3", "(x)"),
        (0x24B4, "3", "(y)"),
        (0x24B5, "3", "(z)"),
        (0x24B6, "M", "a"),
        (0x24B7, "M", "b"),
        (0x24B8, "M", "c"),
        (0x24B9, "M", "d"),
        (0x24BA, "M", "e"),
        (0x24BB, "M", "f"),
        (0x24BC, "M", "g"),
        (0x24BD, "M", "h"),
        (0x24BE, "M", "i"),
        (0x24BF, "M", "j"),
        (0x24C0, "M", "k"),
    ]


def _seg_24() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x24C1, "M", "l"),
        (0x24C2, "M", "m"),
        (0x24C3, "M", "n"),
        (0x24C4, "M", "o"),
        (0x24C5, "M", "p"),
        (0x24C6, "M", "q"),
        (0x24C7, "M", "r"),
        (0x24C8, "M", "s"),
        (0x24C9, "M", "t"),
        (0x24CA, "M", "u"),
        (0x24CB, "M", "v"),
        (0x24CC, "M", "w"),
        (0x24CD, "M", "x"),
        (0x24CE, "M", "y"),
        (0x24CF, "M", "z"),
        (0x24D0, "M", "a"),
        (0x24D1, "M", "b"),
        (0x24D2, "M", "c"),
        (0x24D3, "M", "d"),
        (0x24D4, "M", "e"),
        (0x24D5, "M", "f"),
        (0x24D6, "M", "g"),
        (0x24D7, "M", "h"),
        (0x24D8, "M", "i"),
        (0x24D9, "M", "j"),
        (0x24DA, "M", "k"),
        (0x24DB, "M", "l"),
        (0x24DC, "M", "m"),
        (0x24DD, "M", "n"),
        (0x24DE, "M", "o"),
        (0x24DF, "M", "p"),
        (0x24E0, "M", "q"),
        (0x24E1, "M", "r"),
        (0x24E2, "M", "s"),
        (0x24E3, "M", "t"),
        (0x24E4, "M", "u"),
        (0x24E5, "M", "v"),
        (0x24E6, "M", "w"),
        (0x24E7, "M", "x"),
        (0x24E8, "M", "y"),
        (0x24E9, "M", "z"),
        (0x24EA, "M", "0"),
        (0x24EB, "V"),
        (0x2A0C, "M", "∫∫∫∫"),
        (0x2A0D, "V"),
        (0x2A74, "3", "::="),
        (0x2A75, "3", "=="),
        (0x2A76, "3", "==="),
        (0x2A77, "V"),
        (0x2ADC, "M", "⫝̸"),
        (0x2ADD, "V"),
        (0x2B74, "X"),
        (0x2B76, "V"),
        (0x2B96, "X"),
        (0x2B97, "V"),
        (0x2C00, "M", "ⰰ"),
        (0x2C01, "M", "ⰱ"),
        (0x2C02, "M", "ⰲ"),
        (0x2C03, "M", "ⰳ"),
        (0x2C04, "M", "ⰴ"),
        (0x2C05, "M", "ⰵ"),
        (0x2C06, "M", "ⰶ"),
        (0x2C07, "M", "ⰷ"),
        (0x2C08, "M", "ⰸ"),
        (0x2C09, "M", "ⰹ"),
        (0x2C0A, "M", "ⰺ"),
        (0x2C0B, "M", "ⰻ"),
        (0x2C0C, "M", "ⰼ"),
        (0x2C0D, "M", "ⰽ"),
        (0x2C0E, "M", "ⰾ"),
        (0x2C0F, "M", "ⰿ"),
        (0x2C10, "M", "ⱀ"),
        (0x2C11, "M", "ⱁ"),
        (0x2C12, "M", "ⱂ"),
        (0x2C13, "M", "ⱃ"),
        (0x2C14, "M", "ⱄ"),
        (0x2C15, "M", "ⱅ"),
        (0x2C16, "M", "ⱆ"),
        (0x2C17, "M", "ⱇ"),
        (0x2C18, "M", "ⱈ"),
        (0x2C19, "M", "ⱉ"),
        (0x2C1A, "M", "ⱊ"),
        (0x2C1B, "M", "ⱋ"),
        (0x2C1C, "M", "ⱌ"),
        (0x2C1D, "M", "ⱍ"),
        (0x2C1E, "M", "ⱎ"),
        (0x2C1F, "M", "ⱏ"),
        (0x2C20, "M", "ⱐ"),
        (0x2C21, "M", "ⱑ"),
        (0x2C22, "M", "ⱒ"),
        (0x2C23, "M", "ⱓ"),
        (0x2C24, "M", "ⱔ"),
        (0x2C25, "M", "ⱕ"),
        (0x2C26, "M", "ⱖ"),
        (0x2C27, "M", "ⱗ"),
        (0x2C28, "M", "ⱘ"),
        (0x2C29, "M", "ⱙ"),
        (0x2C2A, "M", "ⱚ"),
        (0x2C2B, "M", "ⱛ"),
        (0x2C2C, "M", "ⱜ"),
    ]


def _seg_25() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x2C2D, "M", "ⱝ"),
        (0x2C2E, "M", "ⱞ"),
        (0x2C2F, "M", "ⱟ"),
        (0x2C30, "V"),
        (0x2C60, "M", "ⱡ"),
        (0x2C61, "V"),
        (0x2C62, "M", "ɫ"),
        (0x2C63, "M", "ᵽ"),
        (0x2C64, "M", "ɽ"),
        (0x2C65, "V"),
        (0x2C67, "M", "ⱨ"),
        (0x2C68, "V"),
        (0x2C69, "M", "ⱪ"),
        (0x2C6A, "V"),
        (0x2C6B, "M", "ⱬ"),
        (0x2C6C, "V"),
        (0x2C6D, "M", "ɑ"),
        (0x2C6E, "M", "ɱ"),
        (0x2C6F, "M", "ɐ"),
        (0x2C70, "M", "ɒ"),
        (0x2C71, "V"),
        (0x2C72, "M", "ⱳ"),
        (0x2C73, "V"),
        (0x2C75, "M", "ⱶ"),
        (0x2C76, "V"),
        (0x2C7C, "M", "j"),
        (0x2C7D, "M", "v"),
        (0x2C7E, "M", "ȿ"),
        (0x2C7F, "M", "ɀ"),
        (0x2C80, "M", "ⲁ"),
        (0x2C81, "V"),
        (0x2C82, "M", "ⲃ"),
        (0x2C83, "V"),
        (0x2C84, "M", "ⲅ"),
        (0x2C85, "V"),
        (0x2C86, "M", "ⲇ"),
        (0x2C87, "V"),
        (0x2C88, "M", "ⲉ"),
        (0x2C89, "V"),
        (0x2C8A, "M", "ⲋ"),
        (0x2C8B, "V"),
        (0x2C8C, "M", "ⲍ"),
        (0x2C8D, "V"),
        (0x2C8E, "M", "ⲏ"),
        (0x2C8F, "V"),
        (0x2C90, "M", "ⲑ"),
        (0x2C91, "V"),
        (0x2C92, "M", "ⲓ"),
        (0x2C93, "V"),
        (0x2C94, "M", "ⲕ"),
        (0x2C95, "V"),
        (0x2C96, "M", "ⲗ"),
        (0x2C97, "V"),
        (0x2C98, "M", "ⲙ"),
        (0x2C99, "V"),
        (0x2C9A, "M", "ⲛ"),
        (0x2C9B, "V"),
        (0x2C9C, "M", "ⲝ"),
        (0x2C9D, "V"),
        (0x2C9E, "M", "ⲟ"),
        (0x2C9F, "V"),
        (0x2CA0, "M", "ⲡ"),
        (0x2CA1, "V"),
        (0x2CA2, "M", "ⲣ"),
        (0x2CA3, "V"),
        (0x2CA4, "M", "ⲥ"),
        (0x2CA5, "V"),
        (0x2CA6, "M", "ⲧ"),
        (0x2CA7, "V"),
        (0x2CA8, "M", "ⲩ"),
        (0x2CA9, "V"),
        (0x2CAA, "M", "ⲫ"),
        (0x2CAB, "V"),
        (0x2CAC, "M", "ⲭ"),
        (0x2CAD, "V"),
        (0x2CAE, "M", "ⲯ"),
        (0x2CAF, "V"),
        (0x2CB0, "M", "ⲱ"),
        (0x2CB1, "V"),
        (0x2CB2, "M", "ⲳ"),
        (0x2CB3, "V"),
        (0x2CB4, "M", "ⲵ"),
        (0x2CB5, "V"),
        (0x2CB6, "M", "ⲷ"),
        (0x2CB7, "V"),
        (0x2CB8, "M", "ⲹ"),
        (0x2CB9, "V"),
        (0x2CBA, "M", "ⲻ"),
        (0x2CBB, "V"),
        (0x2CBC, "M", "ⲽ"),
        (0x2CBD, "V"),
        (0x2CBE, "M", "ⲿ"),
        (0x2CBF, "V"),
        (0x2CC0, "M", "ⳁ"),
        (0x2CC1, "V"),
        (0x2CC2, "M", "ⳃ"),
        (0x2CC3, "V"),
        (0x2CC4, "M", "ⳅ"),
        (0x2CC5, "V"),
        (0x2CC6, "M", "ⳇ"),
    ]


def _seg_26() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x2CC7, "V"),
        (0x2CC8, "M", "ⳉ"),
        (0x2CC9, "V"),
        (0x2CCA, "M", "ⳋ"),
        (0x2CCB, "V"),
        (0x2CCC, "M", "ⳍ"),
        (0x2CCD, "V"),
        (0x2CCE, "M", "ⳏ"),
        (0x2CCF, "V"),
        (0x2CD0, "M", "ⳑ"),
        (0x2CD1, "V"),
        (0x2CD2, "M", "ⳓ"),
        (0x2CD3, "V"),
        (0x2CD4, "M", "ⳕ"),
        (0x2CD5, "V"),
        (0x2CD6, "M", "ⳗ"),
        (0x2CD7, "V"),
        (0x2CD8, "M", "ⳙ"),
        (0x2CD9, "V"),
        (0x2CDA, "M", "ⳛ"),
        (0x2CDB, "V"),
        (0x2CDC, "M", "ⳝ"),
        (0x2CDD, "V"),
        (0x2CDE, "M", "ⳟ"),
        (0x2CDF, "V"),
        (0x2CE0, "M", "ⳡ"),
        (0x2CE1, "V"),
        (0x2CE2, "M", "ⳣ"),
        (0x2CE3, "V"),
        (0x2CEB, "M", "ⳬ"),
        (0x2CEC, "V"),
        (0x2CED, "M", "ⳮ"),
        (0x2CEE, "V"),
        (0x2CF2, "M", "ⳳ"),
        (0x2CF3, "V"),
        (0x2CF4, "X"),
        (0x2CF9, "V"),
        (0x2D26, "X"),
        (0x2D27, "V"),
        (0x2D28, "X"),
        (0x2D2D, "V"),
        (0x2D2E, "X"),
        (0x2D30, "V"),
        (0x2D68, "X"),
        (0x2D6F, "M", "ⵡ"),
        (0x2D70, "V"),
        (0x2D71, "X"),
        (0x2D7F, "V"),
        (0x2D97, "X"),
        (0x2DA0, "V"),
        (0x2DA7, "X"),
        (0x2DA8, "V"),
        (0x2DAF, "X"),
        (0x2DB0, "V"),
        (0x2DB7, "X"),
        (0x2DB8, "V"),
        (0x2DBF, "X"),
        (0x2DC0, "V"),
        (0x2DC7, "X"),
        (0x2DC8, "V"),
        (0x2DCF, "X"),
        (0x2DD0, "V"),
        (0x2DD7, "X"),
        (0x2DD8, "V"),
        (0x2DDF, "X"),
        (0x2DE0, "V"),
        (0x2E5E, "X"),
        (0x2E80, "V"),
        (0x2E9A, "X"),
        (0x2E9B, "V"),
        (0x2E9F, "M", "母"),
        (0x2EA0, "V"),
        (0x2EF3, "M", "龟"),
        (0x2EF4, "X"),
        (0x2F00, "M", "一"),
        (0x2F01, "M", "丨"),
        (0x2F02, "M", "丶"),
        (0x2F03, "M", "丿"),
        (0x2F04, "M", "乙"),
        (0x2F05, "M", "亅"),
        (0x2F06, "M", "二"),
        (0x2F07, "M", "亠"),
        (0x2F08, "M", "人"),
        (0x2F09, "M", "儿"),
        (0x2F0A, "M", "入"),
        (0x2F0B, "M", "八"),
        (0x2F0C, "M", "冂"),
        (0x2F0D, "M", "冖"),
        (0x2F0E, "M", "冫"),
        (0x2F0F, "M", "几"),
        (0x2F10, "M", "凵"),
        (0x2F11, "M", "刀"),
        (0x2F12, "M", "力"),
        (0x2F13, "M", "勹"),
        (0x2F14, "M", "匕"),
        (0x2F15, "M", "匚"),
        (0x2F16, "M", "匸"),
        (0x2F17, "M", "十"),
        (0x2F18, "M", "卜"),
        (0x2F19, "M", "卩"),
    ]


def _seg_27() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x2F1A, "M", "厂"),
        (0x2F1B, "M", "厶"),
        (0x2F1C, "M", "又"),
        (0x2F1D, "M", "口"),
        (0x2F1E, "M", "囗"),
        (0x2F1F, "M", "土"),
        (0x2F20, "M", "士"),
        (0x2F21, "M", "夂"),
        (0x2F22, "M", "夊"),
        (0x2F23, "M", "夕"),
        (0x2F24, "M", "大"),
        (0x2F25, "M", "女"),
        (0x2F26, "M", "子"),
        (0x2F27, "M", "宀"),
        (0x2F28, "M", "寸"),
        (0x2F29, "M", "小"),
        (0x2F2A, "M", "尢"),
        (0x2F2B, "M", "尸"),
        (0x2F2C, "M", "屮"),
        (0x2F2D, "M", "山"),
        (0x2F2E, "M", "巛"),
        (0x2F2F, "M", "工"),
        (0x2F30, "M", "己"),
        (0x2F31, "M", "巾"),
        (0x2F32, "M", "干"),
        (0x2F33, "M", "幺"),
        (0x2F34, "M", "广"),
        (0x2F35, "M", "廴"),
        (0x2F36, "M", "廾"),
        (0x2F37, "M", "弋"),
        (0x2F38, "M", "弓"),
        (0x2F39, "M", "彐"),
        (0x2F3A, "M", "彡"),
        (0x2F3B, "M", "彳"),
        (0x2F3C, "M", "心"),
        (0x2F3D, "M", "戈"),
        (0x2F3E, "M", "戶"),
        (0x2F3F, "M", "手"),
        (0x2F40, "M", "支"),
        (0x2F41, "M", "攴"),
        (0x2F42, "M", "文"),
        (0x2F43, "M", "斗"),
        (0x2F44, "M", "斤"),
        (0x2F45, "M", "方"),
        (0x2F46, "M", "无"),
        (0x2F47, "M", "日"),
        (0x2F48, "M", "曰"),
        (0x2F49, "M", "月"),
        (0x2F4A, "M", "木"),
        (0x2F4B, "M", "欠"),
        (0x2F4C, "M", "止"),
        (0x2F4D, "M", "歹"),
        (0x2F4E, "M", "殳"),
        (0x2F4F, "M", "毋"),
        (0x2F50, "M", "比"),
        (0x2F51, "M", "毛"),
        (0x2F52, "M", "氏"),
        (0x2F53, "M", "气"),
        (0x2F54, "M", "水"),
        (0x2F55, "M", "火"),
        (0x2F56, "M", "爪"),
        (0x2F57, "M", "父"),
        (0x2F58, "M", "爻"),
        (0x2F59, "M", "爿"),
        (0x2F5A, "M", "片"),
        (0x2F5B, "M", "牙"),
        (0x2F5C, "M", "牛"),
        (0x2F5D, "M", "犬"),
        (0x2F5E, "M", "玄"),
        (0x2F5F, "M", "玉"),
        (0x2F60, "M", "瓜"),
        (0x2F61, "M", "瓦"),
        (0x2F62, "M", "甘"),
        (0x2F63, "M", "生"),
        (0x2F64, "M", "用"),
        (0x2F65, "M", "田"),
        (0x2F66, "M", "疋"),
        (0x2F67, "M", "疒"),
        (0x2F68, "M", "癶"),
        (0x2F69, "M", "白"),
        (0x2F6A, "M", "皮"),
        (0x2F6B, "M", "皿"),
        (0x2F6C, "M", "目"),
        (0x2F6D, "M", "矛"),
        (0x2F6E, "M", "矢"),
        (0x2F6F, "M", "石"),
        (0x2F70, "M", "示"),
        (0x2F71, "M", "禸"),
        (0x2F72, "M", "禾"),
        (0x2F73, "M", "穴"),
        (0x2F74, "M", "立"),
        (0x2F75, "M", "竹"),
        (0x2F76, "M", "米"),
        (0x2F77, "M", "糸"),
        (0x2F78, "M", "缶"),
        (0x2F79, "M", "网"),
        (0x2F7A, "M", "羊"),
        (0x2F7B, "M", "羽"),
        (0x2F7C, "M", "老"),
        (0x2F7D, "M", "而"),
    ]


def _seg_28() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x2F7E, "M", "耒"),
        (0x2F7F, "M", "耳"),
        (0x2F80, "M", "聿"),
        (0x2F81, "M", "肉"),
        (0x2F82, "M", "臣"),
        (0x2F83, "M", "自"),
        (0x2F84, "M", "至"),
        (0x2F85, "M", "臼"),
        (0x2F86, "M", "舌"),
        (0x2F87, "M", "舛"),
        (0x2F88, "M", "舟"),
        (0x2F89, "M", "艮"),
        (0x2F8A, "M", "色"),
        (0x2F8B, "M", "艸"),
        (0x2F8C, "M", "虍"),
        (0x2F8D, "M", "虫"),
        (0x2F8E, "M", "血"),
        (0x2F8F, "M", "行"),
        (0x2F90, "M", "衣"),
        (0x2F91, "M", "襾"),
        (0x2F92, "M", "見"),
        (0x2F93, "M", "角"),
        (0x2F94, "M", "言"),
        (0x2F95, "M", "谷"),
        (0x2F96, "M", "豆"),
        (0x2F97, "M", "豕"),
        (0x2F98, "M", "豸"),
        (0x2F99, "M", "貝"),
        (0x2F9A, "M", "赤"),
        (0x2F9B, "M", "走"),
        (0x2F9C, "M", "足"),
        (0x2F9D, "M", "身"),
        (0x2F9E, "M", "車"),
        (0x2F9F, "M", "辛"),
        (0x2FA0, "M", "辰"),
        (0x2FA1, "M", "辵"),
        (0x2FA2, "M", "邑"),
        (0x2FA3, "M", "酉"),
        (0x2FA4, "M", "釆"),
        (0x2FA5, "M", "里"),
        (0x2FA6, "M", "金"),
        (0x2FA7, "M", "長"),
        (0x2FA8, "M", "門"),
        (0x2FA9, "M", "阜"),
        (0x2FAA, "M", "隶"),
        (0x2FAB, "M", "隹"),
        (0x2FAC, "M", "雨"),
        (0x2FAD, "M", "靑"),
        (0x2FAE, "M", "非"),
        (0x2FAF, "M", "面"),
        (0x2FB0, "M", "革"),
        (0x2FB1, "M", "韋"),
        (0x2FB2, "M", "韭"),
        (0x2FB3, "M", "音"),
        (0x2FB4, "M", "頁"),
        (0x2FB5, "M", "風"),
        (0x2FB6, "M", "飛"),
        (0x2FB7, "M", "食"),
        (0x2FB8, "M", "首"),
        (0x2FB9, "M", "香"),
        (0x2FBA, "M", "馬"),
        (0x2FBB, "M", "骨"),
        (0x2FBC, "M", "高"),
        (0x2FBD, "M", "髟"),
        (0x2FBE, "M", "鬥"),
        (0x2FBF, "M", "鬯"),
        (0x2FC0, "M", "鬲"),
        (0x2FC1, "M", "鬼"),
        (0x2FC2, "M", "魚"),
        (0x2FC3, "M", "鳥"),
        (0x2FC4, "M", "鹵"),
        (0x2FC5, "M", "鹿"),
        (0x2FC6, "M", "麥"),
        (0x2FC7, "M", "麻"),
        (0x2FC8, "M", "黃"),
        (0x2FC9, "M", "黍"),
        (0x2FCA, "M", "黑"),
        (0x2FCB, "M", "黹"),
        (0x2FCC, "M", "黽"),
        (0x2FCD, "M", "鼎"),
        (0x2FCE, "M", "鼓"),
        (0x2FCF, "M", "鼠"),
        (0x2FD0, "M", "鼻"),
        (0x2FD1, "M", "齊"),
        (0x2FD2, "M", "齒"),
        (0x2FD3, "M", "龍"),
        (0x2FD4, "M", "龜"),
        (0x2FD5, "M", "龠"),
        (0x2FD6, "X"),
        (0x3000, "3", " "),
        (0x3001, "V"),
        (0x3002, "M", "."),
        (0x3003, "V"),
        (0x3036, "M", "〒"),
        (0x3037, "V"),
        (0x3038, "M", "十"),
        (0x3039, "M", "卄"),
        (0x303A, "M", "卅"),
        (0x303B, "V"),
        (0x3040, "X"),
    ]


def _seg_29() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x3041, "V"),
        (0x3097, "X"),
        (0x3099, "V"),
        (0x309B, "3", " ゙"),
        (0x309C, "3", " ゚"),
        (0x309D, "V"),
        (0x309F, "M", "より"),
        (0x30A0, "V"),
        (0x30FF, "M", "コト"),
        (0x3100, "X"),
        (0x3105, "V"),
        (0x3130, "X"),
        (0x3131, "M", "ᄀ"),
        (0x3132, "M", "ᄁ"),
        (0x3133, "M", "ᆪ"),
        (0x3134, "M", "ᄂ"),
        (0x3135, "M", "ᆬ"),
        (0x3136, "M", "ᆭ"),
        (0x3137, "M", "ᄃ"),
        (0x3138, "M", "ᄄ"),
        (0x3139, "M", "ᄅ"),
        (0x313A, "M", "ᆰ"),
        (0x313B, "M", "ᆱ"),
        (0x313C, "M", "ᆲ"),
        (0x313D, "M", "ᆳ"),
        (0x313E, "M", "ᆴ"),
        (0x313F, "M", "ᆵ"),
        (0x3140, "M", "ᄚ"),
        (0x3141, "M", "ᄆ"),
        (0x3142, "M", "ᄇ"),
        (0x3143, "M", "ᄈ"),
        (0x3144, "M", "ᄡ"),
        (0x3145, "M", "ᄉ"),
        (0x3146, "M", "ᄊ"),
        (0x3147, "M", "ᄋ"),
        (0x3148, "M", "ᄌ"),
        (0x3149, "M", "ᄍ"),
        (0x314A, "M", "ᄎ"),
        (0x314B, "M", "ᄏ"),
        (0x314C, "M", "ᄐ"),
        (0x314D, "M", "ᄑ"),
        (0x314E, "M", "ᄒ"),
        (0x314F, "M", "ᅡ"),
        (0x3150, "M", "ᅢ"),
        (0x3151, "M", "ᅣ"),
        (0x3152, "M", "ᅤ"),
        (0x3153, "M", "ᅥ"),
        (0x3154, "M", "ᅦ"),
        (0x3155, "M", "ᅧ"),
        (0x3156, "M", "ᅨ"),
        (0x3157, "M", "ᅩ"),
        (0x3158, "M", "ᅪ"),
        (0x3159, "M", "ᅫ"),
        (0x315A, "M", "ᅬ"),
        (0x315B, "M", "ᅭ"),
        (0x315C, "M", "ᅮ"),
        (0x315D, "M", "ᅯ"),
        (0x315E, "M", "ᅰ"),
        (0x315F, "M", "ᅱ"),
        (0x3160, "M", "ᅲ"),
        (0x3161, "M", "ᅳ"),
        (0x3162, "M", "ᅴ"),
        (0x3163, "M", "ᅵ"),
        (0x3164, "X"),
        (0x3165, "M", "ᄔ"),
        (0x3166, "M", "ᄕ"),
        (0x3167, "M", "ᇇ"),
        (0x3168, "M", "ᇈ"),
        (0x3169, "M", "ᇌ"),
        (0x316A, "M", "ᇎ"),
        (0x316B, "M", "ᇓ"),
        (0x316C, "M", "ᇗ"),
        (0x316D, "M", "ᇙ"),
        (0x316E, "M", "ᄜ"),
        (0x316F, "M", "ᇝ"),
        (0x3170, "M", "ᇟ"),
        (0x3171, "M", "ᄝ"),
        (0x3172, "M", "ᄞ"),
        (0x3173, "M", "ᄠ"),
        (0x3174, "M", "ᄢ"),
        (0x3175, "M", "ᄣ"),
        (0x3176, "M", "ᄧ"),
        (0x3177, "M", "ᄩ"),
        (0x3178, "M", "ᄫ"),
        (0x3179, "M", "ᄬ"),
        (0x317A, "M", "ᄭ"),
        (0x317B, "M", "ᄮ"),
        (0x317C, "M", "ᄯ"),
        (0x317D, "M", "ᄲ"),
        (0x317E, "M", "ᄶ"),
        (0x317F, "M", "ᅀ"),
        (0x3180, "M", "ᅇ"),
        (0x3181, "M", "ᅌ"),
        (0x3182, "M", "ᇱ"),
        (0x3183, "M", "ᇲ"),
        (0x3184, "M", "ᅗ"),
        (0x3185, "M", "ᅘ"),
        (0x3186, "M", "ᅙ"),
        (0x3187, "M", "ᆄ"),
        (0x3188, "M", "ᆅ"),
    ]


def _seg_30() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x3189, "M", "ᆈ"),
        (0x318A, "M", "ᆑ"),
        (0x318B, "M", "ᆒ"),
        (0x318C, "M", "ᆔ"),
        (0x318D, "M", "ᆞ"),
        (0x318E, "M", "ᆡ"),
        (0x318F, "X"),
        (0x3190, "V"),
        (0x3192, "M", "一"),
        (0x3193, "M", "二"),
        (0x3194, "M", "三"),
        (0x3195, "M", "四"),
        (0x3196, "M", "上"),
        (0x3197, "M", "中"),
        (0x3198, "M", "下"),
        (0x3199, "M", "甲"),
        (0x319A, "M", "乙"),
        (0x319B, "M", "丙"),
        (0x319C, "M", "丁"),
        (0x319D, "M", "天"),
        (0x319E, "M", "地"),
        (0x319F, "M", "人"),
        (0x31A0, "V"),
        (0x31E4, "X"),
        (0x31F0, "V"),
        (0x3200, "3", "(ᄀ)"),
        (0x3201, "3", "(ᄂ)"),
        (0x3202, "3", "(ᄃ)"),
        (0x3203, "3", "(ᄅ)"),
        (0x3204, "3", "(ᄆ)"),
        (0x3205, "3", "(ᄇ)"),
        (0x3206, "3", "(ᄉ)"),
        (0x3207, "3", "(ᄋ)"),
        (0x3208, "3", "(ᄌ)"),
        (0x3209, "3", "(ᄎ)"),
        (0x320A, "3", "(ᄏ)"),
        (0x320B, "3", "(ᄐ)"),
        (0x320C, "3", "(ᄑ)"),
        (0x320D, "3", "(ᄒ)"),
        (0x320E, "3", "(가)"),
        (0x320F, "3", "(나)"),
        (0x3210, "3", "(다)"),
        (0x3211, "3", "(라)"),
        (0x3212, "3", "(마)"),
        (0x3213, "3", "(바)"),
        (0x3214, "3", "(사)"),
        (0x3215, "3", "(아)"),
        (0x3216, "3", "(자)"),
        (0x3217, "3", "(차)"),
        (0x3218, "3", "(카)"),
        (0x3219, "3", "(타)"),
        (0x321A, "3", "(파)"),
        (0x321B, "3", "(하)"),
        (0x321C, "3", "(주)"),
        (0x321D, "3", "(오전)"),
        (0x321E, "3", "(오후)"),
        (0x321F, "X"),
        (0x3220, "3", "(一)"),
        (0x3221, "3", "(二)"),
        (0x3222, "3", "(三)"),
        (0x3223, "3", "(四)"),
        (0x3224, "3", "(五)"),
        (0x3225, "3", "(六)"),
        (0x3226, "3", "(七)"),
        (0x3227, "3", "(八)"),
        (0x3228, "3", "(九)"),
        (0x3229, "3", "(十)"),
        (0x322A, "3", "(月)"),
        (0x322B, "3", "(火)"),
        (0x322C, "3", "(水)"),
        (0x322D, "3", "(木)"),
        (0x322E, "3", "(金)"),
        (0x322F, "3", "(土)"),
        (0x3230, "3", "(日)"),
        (0x3231, "3", "(株)"),
        (0x3232, "3", "(有)"),
        (0x3233, "3", "(社)"),
        (0x3234, "3", "(名)"),
        (0x3235, "3", "(特)"),
        (0x3236, "3", "(財)"),
        (0x3237, "3", "(祝)"),
        (0x3238, "3", "(労)"),
        (0x3239, "3", "(代)"),
        (0x323A, "3", "(呼)"),
        (0x323B, "3", "(学)"),
        (0x323C, "3", "(監)"),
        (0x323D, "3", "(企)"),
        (0x323E, "3", "(資)"),
        (0x323F, "3", "(協)"),
        (0x3240, "3", "(祭)"),
        (0x3241, "3", "(休)"),
        (0x3242, "3", "(自)"),
        (0x3243, "3", "(至)"),
        (0x3244, "M", "問"),
        (0x3245, "M", "幼"),
        (0x3246, "M", "文"),
        (0x3247, "M", "箏"),
        (0x3248, "V"),
        (0x3250, "M", "pte"),
        (0x3251, "M", "21"),
    ]


def _seg_31() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x3252, "M", "22"),
        (0x3253, "M", "23"),
        (0x3254, "M", "24"),
        (0x3255, "M", "25"),
        (0x3256, "M", "26"),
        (0x3257, "M", "27"),
        (0x3258, "M", "28"),
        (0x3259, "M", "29"),
        (0x325A, "M", "30"),
        (0x325B, "M", "31"),
        (0x325C, "M", "32"),
        (0x325D, "M", "33"),
        (0x325E, "M", "34"),
        (0x325F, "M", "35"),
        (0x3260, "M", "ᄀ"),
        (0x3261, "M", "ᄂ"),
        (0x3262, "M", "ᄃ"),
        (0x3263, "M", "ᄅ"),
        (0x3264, "M", "ᄆ"),
        (0x3265, "M", "ᄇ"),
        (0x3266, "M", "ᄉ"),
        (0x3267, "M", "ᄋ"),
        (0x3268, "M", "ᄌ"),
        (0x3269, "M", "ᄎ"),
        (0x326A, "M", "ᄏ"),
        (0x326B, "M", "ᄐ"),
        (0x326C, "M", "ᄑ"),
        (0x326D, "M", "ᄒ"),
        (0x326E, "M", "가"),
        (0x326F, "M", "나"),
        (0x3270, "M", "다"),
        (0x3271, "M", "라"),
        (0x3272, "M", "마"),
        (0x3273, "M", "바"),
        (0x3274, "M", "사"),
        (0x3275, "M", "아"),
        (0x3276, "M", "자"),
        (0x3277, "M", "차"),
        (0x3278, "M", "카"),
        (0x3279, "M", "타"),
        (0x327A, "M", "파"),
        (0x327B, "M", "하"),
        (0x327C, "M", "참고"),
        (0x327D, "M", "주의"),
        (0x327E, "M", "우"),
        (0x327F, "V"),
        (0x3280, "M", "一"),
        (0x3281, "M", "二"),
        (0x3282, "M", "三"),
        (0x3283, "M", "四"),
        (0x3284, "M", "五"),
        (0x3285, "M", "六"),
        (0x3286, "M", "七"),
        (0x3287, "M", "八"),
        (0x3288, "M", "九"),
        (0x3289, "M", "十"),
        (0x328A, "M", "月"),
        (0x328B, "M", "火"),
        (0x328C, "M", "水"),
        (0x328D, "M", "木"),
        (0x328E, "M", "金"),
        (0x328F, "M", "土"),
        (0x3290, "M", "日"),
        (0x3291, "M", "株"),
        (0x3292, "M", "有"),
        (0x3293, "M", "社"),
        (0x3294, "M", "名"),
        (0x3295, "M", "特"),
        (0x3296, "M", "財"),
        (0x3297, "M", "祝"),
        (0x3298, "M", "労"),
        (0x3299, "M", "秘"),
        (0x329A, "M", "男"),
        (0x329B, "M", "女"),
        (0x329C, "M", "適"),
        (0x329D, "M", "優"),
        (0x329E, "M", "印"),
        (0x329F, "M", "注"),
        (0x32A0, "M", "項"),
        (0x32A1, "M", "休"),
        (0x32A2, "M", "写"),
        (0x32A3, "M", "正"),
        (0x32A4, "M", "上"),
        (0x32A5, "M", "中"),
        (0x32A6, "M", "下"),
        (0x32A7, "M", "左"),
        (0x32A8, "M", "右"),
        (0x32A9, "M", "医"),
        (0x32AA, "M", "宗"),
        (0x32AB, "M", "学"),
        (0x32AC, "M", "監"),
        (0x32AD, "M", "企"),
        (0x32AE, "M", "資"),
        (0x32AF, "M", "協"),
        (0x32B0, "M", "夜"),
        (0x32B1, "M", "36"),
        (0x32B2, "M", "37"),
        (0x32B3, "M", "38"),
        (0x32B4, "M", "39"),
        (0x32B5, "M", "40"),
    ]


def _seg_32() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x32B6, "M", "41"),
        (0x32B7, "M", "42"),
        (0x32B8, "M", "43"),
        (0x32B9, "M", "44"),
        (0x32BA, "M", "45"),
        (0x32BB, "M", "46"),
        (0x32BC, "M", "47"),
        (0x32BD, "M", "48"),
        (0x32BE, "M", "49"),
        (0x32BF, "M", "50"),
        (0x32C0, "M", "1月"),
        (0x32C1, "M", "2月"),
        (0x32C2, "M", "3月"),
        (0x32C3, "M", "4月"),
        (0x32C4, "M", "5月"),
        (0x32C5, "M", "6月"),
        (0x32C6, "M", "7月"),
        (0x32C7, "M", "8月"),
        (0x32C8, "M", "9月"),
        (0x32C9, "M", "10月"),
        (0x32CA, "M", "11月"),
        (0x32CB, "M", "12月"),
        (0x32CC, "M", "hg"),
        (0x32CD, "M", "erg"),
        (0x32CE, "M", "ev"),
        (0x32CF, "M", "ltd"),
        (0x32D0, "M", "ア"),
        (0x32D1, "M", "イ"),
        (0x32D2, "M", "ウ"),
        (0x32D3, "M", "エ"),
        (0x32D4, "M", "オ"),
        (0x32D5, "M", "カ"),
        (0x32D6, "M", "キ"),
        (0x32D7, "M", "ク"),
        (0x32D8, "M", "ケ"),
        (0x32D9, "M", "コ"),
        (0x32DA, "M", "サ"),
        (0x32DB, "M", "シ"),
        (0x32DC, "M", "ス"),
        (0x32DD, "M", "セ"),
        (0x32DE, "M", "ソ"),
        (0x32DF, "M", "タ"),
        (0x32E0, "M", "チ"),
        (0x32E1, "M", "ツ"),
        (0x32E2, "M", "テ"),
        (0x32E3, "M", "ト"),
        (0x32E4, "M", "ナ"),
        (0x32E5, "M", "ニ"),
        (0x32E6, "M", "ヌ"),
        (0x32E7, "M", "ネ"),
        (0x32E8, "M", "ノ"),
        (0x32E9, "M", "ハ"),
        (0x32EA, "M", "ヒ"),
        (0x32EB, "M", "フ"),
        (0x32EC, "M", "ヘ"),
        (0x32ED, "M", "ホ"),
        (0x32EE, "M", "マ"),
        (0x32EF, "M", "ミ"),
        (0x32F0, "M", "ム"),
        (0x32F1, "M", "メ"),
        (0x32F2, "M", "モ"),
        (0x32F3, "M", "ヤ"),
        (0x32F4, "M", "ユ"),
        (0x32F5, "M", "ヨ"),
        (0x32F6, "M", "ラ"),
        (0x32F7, "M", "リ"),
        (0x32F8, "M", "ル"),
        (0x32F9, "M", "レ"),
        (0x32FA, "M", "ロ"),
        (0x32FB, "M", "ワ"),
        (0x32FC, "M", "ヰ"),
        (0x32FD, "M", "ヱ"),
        (0x32FE, "M", "ヲ"),
        (0x32FF, "M", "令和"),
        (0x3300, "M", "アパート"),
        (0x3301, "M", "アルファ"),
        (0x3302, "M", "アンペア"),
        (0x3303, "M", "アール"),
        (0x3304, "M", "イニング"),
        (0x3305, "M", "インチ"),
        (0x3306, "M", "ウォン"),
        (0x3307, "M", "エスクード"),
        (0x3308, "M", "エーカー"),
        (0x3309, "M", "オンス"),
        (0x330A, "M", "オーム"),
        (0x330B, "M", "カイリ"),
        (0x330C, "M", "カラット"),
        (0x330D, "M", "カロリー"),
        (0x330E, "M", "ガロン"),
        (0x330F, "M", "ガンマ"),
        (0x3310, "M", "ギガ"),
        (0x3311, "M", "ギニー"),
        (0x3312, "M", "キュリー"),
        (0x3313, "M", "ギルダー"),
        (0x3314, "M", "キロ"),
        (0x3315, "M", "キログラム"),
        (0x3316, "M", "キロメートル"),
        (0x3317, "M", "キロワット"),
        (0x3318, "M", "グラム"),
        (0x3319, "M", "グラムトン"),
    ]


def _seg_33() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x331A, "M", "クルゼイロ"),
        (0x331B, "M", "クローネ"),
        (0x331C, "M", "ケース"),
        (0x331D, "M", "コルナ"),
        (0x331E, "M", "コーポ"),
        (0x331F, "M", "サイクル"),
        (0x3320, "M", "サンチーム"),
        (0x3321, "M", "シリング"),
        (0x3322, "M", "センチ"),
        (0x3323, "M", "セント"),
        (0x3324, "M", "ダース"),
        (0x3325, "M", "デシ"),
        (0x3326, "M", "ドル"),
        (0x3327, "M", "トン"),
        (0x3328, "M", "ナノ"),
        (0x3329, "M", "ノット"),
        (0x332A, "M", "ハイツ"),
        (0x332B, "M", "パーセント"),
        (0x332C, "M", "パーツ"),
        (0x332D, "M", "バーレル"),
        (0x332E, "M", "ピアストル"),
        (0x332F, "M", "ピクル"),
        (0x3330, "M", "ピコ"),
        (0x3331, "M", "ビル"),
        (0x3332, "M", "ファラッド"),
        (0x3333, "M", "フィート"),
        (0x3334, "M", "ブッシェル"),
        (0x3335, "M", "フラン"),
        (0x3336, "M", "ヘクタール"),
        (0x3337, "M", "ペソ"),
        (0x3338, "M", "ペニヒ"),
        (0x3339, "M", "ヘルツ"),
        (0x333A, "M", "ペンス"),
        (0x333B, "M", "ページ"),
        (0x333C, "M", "ベータ"),
        (0x333D, "M", "ポイント"),
        (0x333E, "M", "ボルト"),
        (0x333F, "M", "ホン"),
        (0x3340, "M", "ポンド"),
        (0x3341, "M", "ホール"),
        (0x3342, "M", "ホーン"),
        (0x3343, "M", "マイクロ"),
        (0x3344, "M", "マイル"),
        (0x3345, "M", "マッハ"),
        (0x3346, "M", "マルク"),
        (0x3347, "M", "マンション"),
        (0x3348, "M", "ミクロン"),
        (0x3349, "M", "ミリ"),
        (0x334A, "M", "ミリバール"),
        (0x334B, "M", "メガ"),
        (0x334C, "M", "メガトン"),
        (0x334D, "M", "メートル"),
        (0x334E, "M", "ヤード"),
        (0x334F, "M", "ヤール"),
        (0x3350, "M", "ユアン"),
        (0x3351, "M", "リットル"),
        (0x3352, "M", "リラ"),
        (0x3353, "M", "ルピー"),
        (0x3354, "M", "ルーブル"),
        (0x3355, "M", "レム"),
        (0x3356, "M", "レントゲン"),
        (0x3357, "M", "ワット"),
        (0x3358, "M", "0点"),
        (0x3359, "M", "1点"),
        (0x335A, "M", "2点"),
        (0x335B, "M", "3点"),
        (0x335C, "M", "4点"),
        (0x335D, "M", "5点"),
        (0x335E, "M", "6点"),
        (0x335F, "M", "7点"),
        (0x3360, "M", "8点"),
        (0x3361, "M", "9点"),
        (0x3362, "M", "10点"),
        (0x3363, "M", "11点"),
        (0x3364, "M", "12点"),
        (0x3365, "M", "13点"),
        (0x3366, "M", "14点"),
        (0x3367, "M", "15点"),
        (0x3368, "M", "16点"),
        (0x3369, "M", "17点"),
        (0x336A, "M", "18点"),
        (0x336B, "M", "19点"),
        (0x336C, "M", "20点"),
        (0x336D, "M", "21点"),
        (0x336E, "M", "22点"),
        (0x336F, "M", "23点"),
        (0x3370, "M", "24点"),
        (0x3371, "M", "hpa"),
        (0x3372, "M", "da"),
        (0x3373, "M", "au"),
        (0x3374, "M", "bar"),
        (0x3375, "M", "ov"),
        (0x3376, "M", "pc"),
        (0x3377, "M", "dm"),
        (0x3378, "M", "dm2"),
        (0x3379, "M", "dm3"),
        (0x337A, "M", "iu"),
        (0x337B, "M", "平成"),
        (0x337C, "M", "昭和"),
        (0x337D, "M", "大正"),
    ]


def _seg_34() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x337E, "M", "明治"),
        (0x337F, "M", "株式会社"),
        (0x3380, "M", "pa"),
        (0x3381, "M", "na"),
        (0x3382, "M", "μa"),
        (0x3383, "M", "ma"),
        (0x3384, "M", "ka"),
        (0x3385, "M", "kb"),
        (0x3386, "M", "mb"),
        (0x3387, "M", "gb"),
        (0x3388, "M", "cal"),
        (0x3389, "M", "kcal"),
        (0x338A, "M", "pf"),
        (0x338B, "M", "nf"),
        (0x338C, "M", "μf"),
        (0x338D, "M", "μg"),
        (0x338E, "M", "mg"),
        (0x338F, "M", "kg"),
        (0x3390, "M", "hz"),
        (0x3391, "M", "khz"),
        (0x3392, "M", "mhz"),
        (0x3393, "M", "ghz"),
        (0x3394, "M", "thz"),
        (0x3395, "M", "μl"),
        (0x3396, "M", "ml"),
        (0x3397, "M", "dl"),
        (0x3398, "M", "kl"),
        (0x3399, "M", "fm"),
        (0x339A, "M", "nm"),
        (0x339B, "M", "μm"),
        (0x339C, "M", "mm"),
        (0x339D, "M", "cm"),
        (0x339E, "M", "km"),
        (0x339F, "M", "mm2"),
        (0x33A0, "M", "cm2"),
        (0x33A1, "M", "m2"),
        (0x33A2, "M", "km2"),
        (0x33A3, "M", "mm3"),
        (0x33A4, "M", "cm3"),
        (0x33A5, "M", "m3"),
        (0x33A6, "M", "km3"),
        (0x33A7, "M", "m∕s"),
        (0x33A8, "M", "m∕s2"),
        (0x33A9, "M", "pa"),
        (0x33AA, "M", "kpa"),
        (0x33AB, "M", "mpa"),
        (0x33AC, "M", "gpa"),
        (0x33AD, "M", "rad"),
        (0x33AE, "M", "rad∕s"),
        (0x33AF, "M", "rad∕s2"),
        (0x33B0, "M", "ps"),
        (0x33B1, "M", "ns"),
        (0x33B2, "M", "μs"),
        (0x33B3, "M", "ms"),
        (0x33B4, "M", "pv"),
        (0x33B5, "M", "nv"),
        (0x33B6, "M", "μv"),
        (0x33B7, "M", "mv"),
        (0x33B8, "M", "kv"),
        (0x33B9, "M", "mv"),
        (0x33BA, "M", "pw"),
        (0x33BB, "M", "nw"),
        (0x33BC, "M", "μw"),
        (0x33BD, "M", "mw"),
        (0x33BE, "M", "kw"),
        (0x33BF, "M", "mw"),
        (0x33C0, "M", "kω"),
        (0x33C1, "M", "mω"),
        (0x33C2, "X"),
        (0x33C3, "M", "bq"),
        (0x33C4, "M", "cc"),
        (0x33C5, "M", "cd"),
        (0x33C6, "M", "c∕kg"),
        (0x33C7, "X"),
        (0x33C8, "M", "db"),
        (0x33C9, "M", "gy"),
        (0x33CA, "M", "ha"),
        (0x33CB, "M", "hp"),
        (0x33CC, "M", "in"),
        (0x33CD, "M", "kk"),
        (0x33CE, "M", "km"),
        (0x33CF, "M", "kt"),
        (0x33D0, "M", "lm"),
        (0x33D1, "M", "ln"),
        (0x33D2, "M", "log"),
        (0x33D3, "M", "lx"),
        (0x33D4, "M", "mb"),
        (0x33D5, "M", "mil"),
        (0x33D6, "M", "mol"),
        (0x33D7, "M", "ph"),
        (0x33D8, "X"),
        (0x33D9, "M", "ppm"),
        (0x33DA, "M", "pr"),
        (0x33DB, "M", "sr"),
        (0x33DC, "M", "sv"),
        (0x33DD, "M", "wb"),
        (0x33DE, "M", "v∕m"),
        (0x33DF, "M", "a∕m"),
        (0x33E0, "M", "1日"),
        (0x33E1, "M", "2日"),
    ]


def _seg_35() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x33E2, "M", "3日"),
        (0x33E3, "M", "4日"),
        (0x33E4, "M", "5日"),
        (0x33E5, "M", "6日"),
        (0x33E6, "M", "7日"),
        (0x33E7, "M", "8日"),
        (0x33E8, "M", "9日"),
        (0x33E9, "M", "10日"),
        (0x33EA, "M", "11日"),
        (0x33EB, "M", "12日"),
        (0x33EC, "M", "13日"),
        (0x33ED, "M", "14日"),
        (0x33EE, "M", "15日"),
        (0x33EF, "M", "16日"),
        (0x33F0, "M", "17日"),
        (0x33F1, "M", "18日"),
        (0x33F2, "M", "19日"),
        (0x33F3, "M", "20日"),
        (0x33F4, "M", "21日"),
        (0x33F5, "M", "22日"),
        (0x33F6, "M", "23日"),
        (0x33F7, "M", "24日"),
        (0x33F8, "M", "25日"),
        (0x33F9, "M", "26日"),
        (0x33FA, "M", "27日"),
        (0x33FB, "M", "28日"),
        (0x33FC, "M", "29日"),
        (0x33FD, "M", "30日"),
        (0x33FE, "M", "31日"),
        (0x33FF, "M", "gal"),
        (0x3400, "V"),
        (0xA48D, "X"),
        (0xA490, "V"),
        (0xA4C7, "X"),
        (0xA4D0, "V"),
        (0xA62C, "X"),
        (0xA640, "M", "ꙁ"),
        (0xA641, "V"),
        (0xA642, "M", "ꙃ"),
        (0xA643, "V"),
        (0xA644, "M", "ꙅ"),
        (0xA645, "V"),
        (0xA646, "M", "ꙇ"),
        (0xA647, "V"),
        (0xA648, "M", "ꙉ"),
        (0xA649, "V"),
        (0xA64A, "M", "ꙋ"),
        (0xA64B, "V"),
        (0xA64C, "M", "ꙍ"),
        (0xA64D, "V"),
        (0xA64E, "M", "ꙏ"),
        (0xA64F, "V"),
        (0xA650, "M", "ꙑ"),
        (0xA651, "V"),
        (0xA652, "M", "ꙓ"),
        (0xA653, "V"),
        (0xA654, "M", "ꙕ"),
        (0xA655, "V"),
        (0xA656, "M", "ꙗ"),
        (0xA657, "V"),
        (0xA658, "M", "ꙙ"),
        (0xA659, "V"),
        (0xA65A, "M", "ꙛ"),
        (0xA65B, "V"),
        (0xA65C, "M", "ꙝ"),
        (0xA65D, "V"),
        (0xA65E, "M", "ꙟ"),
        (0xA65F, "V"),
        (0xA660, "M", "ꙡ"),
        (0xA661, "V"),
        (0xA662, "M", "ꙣ"),
        (0xA663, "V"),
        (0xA664, "M", "ꙥ"),
        (0xA665, "V"),
        (0xA666, "M", "ꙧ"),
        (0xA667, "V"),
        (0xA668, "M", "ꙩ"),
        (0xA669, "V"),
        (0xA66A, "M", "ꙫ"),
        (0xA66B, "V"),
        (0xA66C, "M", "ꙭ"),
        (0xA66D, "V"),
        (0xA680, "M", "ꚁ"),
        (0xA681, "V"),
        (0xA682, "M", "ꚃ"),
        (0xA683, "V"),
        (0xA684, "M", "ꚅ"),
        (0xA685, "V"),
        (0xA686, "M", "ꚇ"),
        (0xA687, "V"),
        (0xA688, "M", "ꚉ"),
        (0xA689, "V"),
        (0xA68A, "M", "ꚋ"),
        (0xA68B, "V"),
        (0xA68C, "M", "ꚍ"),
        (0xA68D, "V"),
        (0xA68E, "M", "ꚏ"),
        (0xA68F, "V"),
        (0xA690, "M", "ꚑ"),
        (0xA691, "V"),
    ]


def _seg_36() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0xA692, "M", "ꚓ"),
        (0xA693, "V"),
        (0xA694, "M", "ꚕ"),
        (0xA695, "V"),
        (0xA696, "M", "ꚗ"),
        (0xA697, "V"),
        (0xA698, "M", "ꚙ"),
        (0xA699, "V"),
        (0xA69A, "M", "ꚛ"),
        (0xA69B, "V"),
        (0xA69C, "M", "ъ"),
        (0xA69D, "M", "ь"),
        (0xA69E, "V"),
        (0xA6F8, "X"),
        (0xA700, "V"),
        (0xA722, "M", "ꜣ"),
        (0xA723, "V"),
        (0xA724, "M", "ꜥ"),
        (0xA725, "V"),
        (0xA726, "M", "ꜧ"),
        (0xA727, "V"),
        (0xA728, "M", "ꜩ"),
        (0xA729, "V"),
        (0xA72A, "M", "ꜫ"),
        (0xA72B, "V"),
        (0xA72C, "M", "ꜭ"),
        (0xA72D, "V"),
        (0xA72E, "M", "ꜯ"),
        (0xA72F, "V"),
        (0xA732, "M", "ꜳ"),
        (0xA733, "V"),
        (0xA734, "M", "ꜵ"),
        (0xA735, "V"),
        (0xA736, "M", "ꜷ"),
        (0xA737, "V"),
        (0xA738, "M", "ꜹ"),
        (0xA739, "V"),
        (0xA73A, "M", "ꜻ"),
        (0xA73B, "V"),
        (0xA73C, "M", "ꜽ"),
        (0xA73D, "V"),
        (0xA73E, "M", "ꜿ"),
        (0xA73F, "V"),
        (0xA740, "M", "ꝁ"),
        (0xA741, "V"),
        (0xA742, "M", "ꝃ"),
        (0xA743, "V"),
        (0xA744, "M", "ꝅ"),
        (0xA745, "V"),
        (0xA746, "M", "ꝇ"),
        (0xA747, "V"),
        (0xA748, "M", "ꝉ"),
        (0xA749, "V"),
        (0xA74A, "M", "ꝋ"),
        (0xA74B, "V"),
        (0xA74C, "M", "ꝍ"),
        (0xA74D, "V"),
        (0xA74E, "M", "ꝏ"),
        (0xA74F, "V"),
        (0xA750, "M", "ꝑ"),
        (0xA751, "V"),
        (0xA752, "M", "ꝓ"),
        (0xA753, "V"),
        (0xA754, "M", "ꝕ"),
        (0xA755, "V"),
        (0xA756, "M", "ꝗ"),
        (0xA757, "V"),
        (0xA758, "M", "ꝙ"),
        (0xA759, "V"),
        (0xA75A, "M", "ꝛ"),
        (0xA75B, "V"),
        (0xA75C, "M", "ꝝ"),
        (0xA75D, "V"),
        (0xA75E, "M", "ꝟ"),
        (0xA75F, "V"),
        (0xA760, "M", "ꝡ"),
        (0xA761, "V"),
        (0xA762, "M", "ꝣ"),
        (0xA763, "V"),
        (0xA764, "M", "ꝥ"),
        (0xA765, "V"),
        (0xA766, "M", "ꝧ"),
        (0xA767, "V"),
        (0xA768, "M", "ꝩ"),
        (0xA769, "V"),
        (0xA76A, "M", "ꝫ"),
        (0xA76B, "V"),
        (0xA76C, "M", "ꝭ"),
        (0xA76D, "V"),
        (0xA76E, "M", "ꝯ"),
        (0xA76F, "V"),
        (0xA770, "M", "ꝯ"),
        (0xA771, "V"),
        (0xA779, "M", "ꝺ"),
        (0xA77A, "V"),
        (0xA77B, "M", "ꝼ"),
        (0xA77C, "V"),
        (0xA77D, "M", "ᵹ"),
        (0xA77E, "M", "ꝿ"),
        (0xA77F, "V"),
    ]


def _seg_37() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0xA780, "M", "ꞁ"),
        (0xA781, "V"),
        (0xA782, "M", "ꞃ"),
        (0xA783, "V"),
        (0xA784, "M", "ꞅ"),
        (0xA785, "V"),
        (0xA786, "M", "ꞇ"),
        (0xA787, "V"),
        (0xA78B, "M", "ꞌ"),
        (0xA78C, "V"),
        (0xA78D, "M", "ɥ"),
        (0xA78E, "V"),
        (0xA790, "M", "ꞑ"),
        (0xA791, "V"),
        (0xA792, "M", "ꞓ"),
        (0xA793, "V"),
        (0xA796, "M", "ꞗ"),
        (0xA797, "V"),
        (0xA798, "M", "ꞙ"),
        (0xA799, "V"),
        (0xA79A, "M", "ꞛ"),
        (0xA79B, "V"),
        (0xA79C, "M", "ꞝ"),
        (0xA79D, "V"),
        (0xA79E, "M", "ꞟ"),
        (0xA79F, "V"),
        (0xA7A0, "M", "ꞡ"),
        (0xA7A1, "V"),
        (0xA7A2, "M", "ꞣ"),
        (0xA7A3, "V"),
        (0xA7A4, "M", "ꞥ"),
        (0xA7A5, "V"),
        (0xA7A6, "M", "ꞧ"),
        (0xA7A7, "V"),
        (0xA7A8, "M", "ꞩ"),
        (0xA7A9, "V"),
        (0xA7AA, "M", "ɦ"),
        (0xA7AB, "M", "ɜ"),
        (0xA7AC, "M", "ɡ"),
        (0xA7AD, "M", "ɬ"),
        (0xA7AE, "M", "ɪ"),
        (0xA7AF, "V"),
        (0xA7B0, "M", "ʞ"),
        (0xA7B1, "M", "ʇ"),
        (0xA7B2, "M", "ʝ"),
        (0xA7B3, "M", "ꭓ"),
        (0xA7B4, "M", "ꞵ"),
        (0xA7B5, "V"),
        (0xA7B6, "M", "ꞷ"),
        (0xA7B7, "V"),
        (0xA7B8, "M", "ꞹ"),
        (0xA7B9, "V"),
        (0xA7BA, "M", "ꞻ"),
        (0xA7BB, "V"),
        (0xA7BC, "M", "ꞽ"),
        (0xA7BD, "V"),
        (0xA7BE, "M", "ꞿ"),
        (0xA7BF, "V"),
        (0xA7C0, "M", "ꟁ"),
        (0xA7C1, "V"),
        (0xA7C2, "M", "ꟃ"),
        (0xA7C3, "V"),
        (0xA7C4, "M", "ꞔ"),
        (0xA7C5, "M", "ʂ"),
        (0xA7C6, "M", "ᶎ"),
        (0xA7C7, "M", "ꟈ"),
        (0xA7C8, "V"),
        (0xA7C9, "M", "ꟊ"),
        (0xA7CA, "V"),
        (0xA7CB, "X"),
        (0xA7D0, "M", "ꟑ"),
        (0xA7D1, "V"),
        (0xA7D2, "X"),
        (0xA7D3, "V"),
        (0xA7D4, "X"),
        (0xA7D5, "V"),
        (0xA7D6, "M", "ꟗ"),
        (0xA7D7, "V"),
        (0xA7D8, "M", "ꟙ"),
        (0xA7D9, "V"),
        (0xA7DA, "X"),
        (0xA7F2, "M", "c"),
        (0xA7F3, "M", "f"),
        (0xA7F4, "M", "q"),
        (0xA7F5, "M", "ꟶ"),
        (0xA7F6, "V"),
        (0xA7F8, "M", "ħ"),
        (0xA7F9, "M", "œ"),
        (0xA7FA, "V"),
        (0xA82D, "X"),
        (0xA830, "V"),
        (0xA83A, "X"),
        (0xA840, "V"),
        (0xA878, "X"),
        (0xA880, "V"),
        (0xA8C6, "X"),
        (0xA8CE, "V"),
        (0xA8DA, "X"),
        (0xA8E0, "V"),
        (0xA954, "X"),
    ]


def _seg_38() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0xA95F, "V"),
        (0xA97D, "X"),
        (0xA980, "V"),
        (0xA9CE, "X"),
        (0xA9CF, "V"),
        (0xA9DA, "X"),
        (0xA9DE, "V"),
        (0xA9FF, "X"),
        (0xAA00, "V"),
        (0xAA37, "X"),
        (0xAA40, "V"),
        (0xAA4E, "X"),
        (0xAA50, "V"),
        (0xAA5A, "X"),
        (0xAA5C, "V"),
        (0xAAC3, "X"),
        (0xAADB, "V"),
        (0xAAF7, "X"),
        (0xAB01, "V"),
        (0xAB07, "X"),
        (0xAB09, "V"),
        (0xAB0F, "X"),
        (0xAB11, "V"),
        (0xAB17, "X"),
        (0xAB20, "V"),
        (0xAB27, "X"),
        (0xAB28, "V"),
        (0xAB2F, "X"),
        (0xAB30, "V"),
        (0xAB5C, "M", "ꜧ"),
        (0xAB5D, "M", "ꬷ"),
        (0xAB5E, "M", "ɫ"),
        (0xAB5F, "M", "ꭒ"),
        (0xAB60, "V"),
        (0xAB69, "M", "ʍ"),
        (0xAB6A, "V"),
        (0xAB6C, "X"),
        (0xAB70, "M", "Ꭰ"),
        (0xAB71, "M", "Ꭱ"),
        (0xAB72, "M", "Ꭲ"),
        (0xAB73, "M", "Ꭳ"),
        (0xAB74, "M", "Ꭴ"),
        (0xAB75, "M", "Ꭵ"),
        (0xAB76, "M", "Ꭶ"),
        (0xAB77, "M", "Ꭷ"),
        (0xAB78, "M", "Ꭸ"),
        (0xAB79, "M", "Ꭹ"),
        (0xAB7A, "M", "Ꭺ"),
        (0xAB7B, "M", "Ꭻ"),
        (0xAB7C, "M", "Ꭼ"),
        (0xAB7D, "M", "Ꭽ"),
        (0xAB7E, "M", "Ꭾ"),
        (0xAB7F, "M", "Ꭿ"),
        (0xAB80, "M", "Ꮀ"),
        (0xAB81, "M", "Ꮁ"),
        (0xAB82, "M", "Ꮂ"),
        (0xAB83, "M", "Ꮃ"),
        (0xAB84, "M", "Ꮄ"),
        (0xAB85, "M", "Ꮅ"),
        (0xAB86, "M", "Ꮆ"),
        (0xAB87, "M", "Ꮇ"),
        (0xAB88, "M", "Ꮈ"),
        (0xAB89, "M", "Ꮉ"),
        (0xAB8A, "M", "Ꮊ"),
        (0xAB8B, "M", "Ꮋ"),
        (0xAB8C, "M", "Ꮌ"),
        (0xAB8D, "M", "Ꮍ"),
        (0xAB8E, "M", "Ꮎ"),
        (0xAB8F, "M", "Ꮏ"),
        (0xAB90, "M", "Ꮐ"),
        (0xAB91, "M", "Ꮑ"),
        (0xAB92, "M", "Ꮒ"),
        (0xAB93, "M", "Ꮓ"),
        (0xAB94, "M", "Ꮔ"),
        (0xAB95, "M", "Ꮕ"),
        (0xAB96, "M", "Ꮖ"),
        (0xAB97, "M", "Ꮗ"),
        (0xAB98, "M", "Ꮘ"),
        (0xAB99, "M", "Ꮙ"),
        (0xAB9A, "M", "Ꮚ"),
        (0xAB9B, "M", "Ꮛ"),
        (0xAB9C, "M", "Ꮜ"),
        (0xAB9D, "M", "Ꮝ"),
        (0xAB9E, "M", "Ꮞ"),
        (0xAB9F, "M", "Ꮟ"),
        (0xABA0, "M", "Ꮠ"),
        (0xABA1, "M", "Ꮡ"),
        (0xABA2, "M", "Ꮢ"),
        (0xABA3, "M", "Ꮣ"),
        (0xABA4, "M", "Ꮤ"),
        (0xABA5, "M", "Ꮥ"),
        (0xABA6, "M", "Ꮦ"),
        (0xABA7, "M", "Ꮧ"),
        (0xABA8, "M", "Ꮨ"),
        (0xABA9, "M", "Ꮩ"),
        (0xABAA, "M", "Ꮪ"),
        (0xABAB, "M", "Ꮫ"),
        (0xABAC, "M", "Ꮬ"),
        (0xABAD, "M", "Ꮭ"),
        (0xABAE, "M", "Ꮮ"),
    ]


def _seg_39() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0xABAF, "M", "Ꮯ"),
        (0xABB0, "M", "Ꮰ"),
        (0xABB1, "M", "Ꮱ"),
        (0xABB2, "M", "Ꮲ"),
        (0xABB3, "M", "Ꮳ"),
        (0xABB4, "M", "Ꮴ"),
        (0xABB5, "M", "Ꮵ"),
        (0xABB6, "M", "Ꮶ"),
        (0xABB7, "M", "Ꮷ"),
        (0xABB8, "M", "Ꮸ"),
        (0xABB9, "M", "Ꮹ"),
        (0xABBA, "M", "Ꮺ"),
        (0xABBB, "M", "Ꮻ"),
        (0xABBC, "M", "Ꮼ"),
        (0xABBD, "M", "Ꮽ"),
        (0xABBE, "M", "Ꮾ"),
        (0xABBF, "M", "Ꮿ"),
        (0xABC0, "V"),
        (0xABEE, "X"),
        (0xABF0, "V"),
        (0xABFA, "X"),
        (0xAC00, "V"),
        (0xD7A4, "X"),
        (0xD7B0, "V"),
        (0xD7C7, "X"),
        (0xD7CB, "V"),
        (0xD7FC, "X"),
        (0xF900, "M", "豈"),
        (0xF901, "M", "更"),
        (0xF902, "M", "車"),
        (0xF903, "M", "賈"),
        (0xF904, "M", "滑"),
        (0xF905, "M", "串"),
        (0xF906, "M", "句"),
        (0xF907, "M", "龜"),
        (0xF909, "M", "契"),
        (0xF90A, "M", "金"),
        (0xF90B, "M", "喇"),
        (0xF90C, "M", "奈"),
        (0xF90D, "M", "懶"),
        (0xF90E, "M", "癩"),
        (0xF90F, "M", "羅"),
        (0xF910, "M", "蘿"),
        (0xF911, "M", "螺"),
        (0xF912, "M", "裸"),
        (0xF913, "M", "邏"),
        (0xF914, "M", "樂"),
        (0xF915, "M", "洛"),
        (0xF916, "M", "烙"),
        (0xF917, "M", "珞"),
        (0xF918, "M", "落"),
        (0xF919, "M", "酪"),
        (0xF91A, "M", "駱"),
        (0xF91B, "M", "亂"),
        (0xF91C, "M", "卵"),
        (0xF91D, "M", "欄"),
        (0xF91E, "M", "爛"),
        (0xF91F, "M", "蘭"),
        (0xF920, "M", "鸞"),
        (0xF921, "M", "嵐"),
        (0xF922, "M", "濫"),
        (0xF923, "M", "藍"),
        (0xF924, "M", "襤"),
        (0xF925, "M", "拉"),
        (0xF926, "M", "臘"),
        (0xF927, "M", "蠟"),
        (0xF928, "M", "廊"),
        (0xF929, "M", "朗"),
        (0xF92A, "M", "浪"),
        (0xF92B, "M", "狼"),
        (0xF92C, "M", "郎"),
        (0xF92D, "M", "來"),
        (0xF92E, "M", "冷"),
        (0xF92F, "M", "勞"),
        (0xF930, "M", "擄"),
        (0xF931, "M", "櫓"),
        (0xF932, "M", "爐"),
        (0xF933, "M", "盧"),
        (0xF934, "M", "老"),
        (0xF935, "M", "蘆"),
        (0xF936, "M", "虜"),
        (0xF937, "M", "路"),
        (0xF938, "M", "露"),
        (0xF939, "M", "魯"),
        (0xF93A, "M", "鷺"),
        (0xF93B, "M", "碌"),
        (0xF93C, "M", "祿"),
        (0xF93D, "M", "綠"),
        (0xF93E, "M", "菉"),
        (0xF93F, "M", "錄"),
        (0xF940, "M", "鹿"),
        (0xF941, "M", "論"),
        (0xF942, "M", "壟"),
        (0xF943, "M", "弄"),
        (0xF944, "M", "籠"),
        (0xF945, "M", "聾"),
        (0xF946, "M", "牢"),
        (0xF947, "M", "磊"),
        (0xF948, "M", "賂"),
        (0xF949, "M", "雷"),
    ]


def _seg_40() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0xF94A, "M", "壘"),
        (0xF94B, "M", "屢"),
        (0xF94C, "M", "樓"),
        (0xF94D, "M", "淚"),
        (0xF94E, "M", "漏"),
        (0xF94F, "M", "累"),
        (0xF950, "M", "縷"),
        (0xF951, "M", "陋"),
        (0xF952, "M", "勒"),
        (0xF953, "M", "肋"),
        (0xF954, "M", "凜"),
        (0xF955, "M", "凌"),
        (0xF956, "M", "稜"),
        (0xF957, "M", "綾"),
        (0xF958, "M", "菱"),
        (0xF959, "M", "陵"),
        (0xF95A, "M", "讀"),
        (0xF95B, "M", "拏"),
        (0xF95C, "M", "樂"),
        (0xF95D, "M", "諾"),
        (0xF95E, "M", "丹"),
        (0xF95F, "M", "寧"),
        (0xF960, "M", "怒"),
        (0xF961, "M", "率"),
        (0xF962, "M", "異"),
        (0xF963, "M", "北"),
        (0xF964, "M", "磻"),
        (0xF965, "M", "便"),
        (0xF966, "M", "復"),
        (0xF967, "M", "不"),
        (0xF968, "M", "泌"),
        (0xF969, "M", "數"),
        (0xF96A, "M", "索"),
        (0xF96B, "M", "參"),
        (0xF96C, "M", "塞"),
        (0xF96D, "M", "省"),
        (0xF96E, "M", "葉"),
        (0xF96F, "M", "說"),
        (0xF970, "M", "殺"),
        (0xF971, "M", "辰"),
        (0xF972, "M", "沈"),
        (0xF973, "M", "拾"),
        (0xF974, "M", "若"),
        (0xF975, "M", "掠"),
        (0xF976, "M", "略"),
        (0xF977, "M", "亮"),
        (0xF978, "M", "兩"),
        (0xF979, "M", "凉"),
        (0xF97A, "M", "梁"),
        (0xF97B, "M", "糧"),
        (0xF97C, "M", "良"),
        (0xF97D, "M", "諒"),
        (0xF97E, "M", "量"),
        (0xF97F, "M", "勵"),
        (0xF980, "M", "呂"),
        (0xF981, "M", "女"),
        (0xF982, "M", "廬"),
        (0xF983, "M", "旅"),
        (0xF984, "M", "濾"),
        (0xF985, "M", "礪"),
        (0xF986, "M", "閭"),
        (0xF987, "M", "驪"),
        (0xF988, "M", "麗"),
        (0xF989, "M", "黎"),
        (0xF98A, "M", "力"),
        (0xF98B, "M", "曆"),
        (0xF98C, "M", "歷"),
        (0xF98D, "M", "轢"),
        (0xF98E, "M", "年"),
        (0xF98F, "M", "憐"),
        (0xF990, "M", "戀"),
        (0xF991, "M", "撚"),
        (0xF992, "M", "漣"),
        (0xF993, "M", "煉"),
        (0xF994, "M", "璉"),
        (0xF995, "M", "秊"),
        (0xF996, "M", "練"),
        (0xF997, "M", "聯"),
        (0xF998, "M", "輦"),
        (0xF999, "M", "蓮"),
        (0xF99A, "M", "連"),
        (0xF99B, "M", "鍊"),
        (0xF99C, "M", "列"),
        (0xF99D, "M", "劣"),
        (0xF99E, "M", "咽"),
        (0xF99F, "M", "烈"),
        (0xF9A0, "M", "裂"),
        (0xF9A1, "M", "說"),
        (0xF9A2, "M", "廉"),
        (0xF9A3, "M", "念"),
        (0xF9A4, "M", "捻"),
        (0xF9A5, "M", "殮"),
        (0xF9A6, "M", "簾"),
        (0xF9A7, "M", "獵"),
        (0xF9A8, "M", "令"),
        (0xF9A9, "M", "囹"),
        (0xF9AA, "M", "寧"),
        (0xF9AB, "M", "嶺"),
        (0xF9AC, "M", "怜"),
        (0xF9AD, "M", "玲"),
    ]


def _seg_41() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0xF9AE, "M", "瑩"),
        (0xF9AF, "M", "羚"),
        (0xF9B0, "M", "聆"),
        (0xF9B1, "M", "鈴"),
        (0xF9B2, "M", "零"),
        (0xF9B3, "M", "靈"),
        (0xF9B4, "M", "領"),
        (0xF9B5, "M", "例"),
        (0xF9B6, "M", "禮"),
        (0xF9B7, "M", "醴"),
        (0xF9B8, "M", "隸"),
        (0xF9B9, "M", "惡"),
        (0xF9BA, "M", "了"),
        (0xF9BB, "M", "僚"),
        (0xF9BC, "M", "寮"),
        (0xF9BD, "M", "尿"),
        (0xF9BE, "M", "料"),
        (0xF9BF, "M", "樂"),
        (0xF9C0, "M", "燎"),
        (0xF9C1, "M", "療"),
        (0xF9C2, "M", "蓼"),
        (0xF9C3, "M", "遼"),
        (0xF9C4, "M", "龍"),
        (0xF9C5, "M", "暈"),
        (0xF9C6, "M", "阮"),
        (0xF9C7, "M", "劉"),
        (0xF9C8, "M", "杻"),
        (0xF9C9, "M", "柳"),
        (0xF9CA, "M", "流"),
        (0xF9CB, "M", "溜"),
        (0xF9CC, "M", "琉"),
        (0xF9CD, "M", "留"),
        (0xF9CE, "M", "硫"),
        (0xF9CF, "M", "紐"),
        (0xF9D0, "M", "類"),
        (0xF9D1, "M", "六"),
        (0xF9D2, "M", "戮"),
        (0xF9D3, "M", "陸"),
        (0xF9D4, "M", "倫"),
        (0xF9D5, "M", "崙"),
        (0xF9D6, "M", "淪"),
        (0xF9D7, "M", "輪"),
        (0xF9D8, "M", "律"),
        (0xF9D9, "M", "慄"),
        (0xF9DA, "M", "栗"),
        (0xF9DB, "M", "率"),
        (0xF9DC, "M", "隆"),
        (0xF9DD, "M", "利"),
        (0xF9DE, "M", "吏"),
        (0xF9DF, "M", "履"),
        (0xF9E0, "M", "易"),
        (0xF9E1, "M", "李"),
        (0xF9E2, "M", "梨"),
        (0xF9E3, "M", "泥"),
        (0xF9E4, "M", "理"),
        (0xF9E5, "M", "痢"),
        (0xF9E6, "M", "罹"),
        (0xF9E7, "M", "裏"),
        (0xF9E8, "M", "裡"),
        (0xF9E9, "M", "里"),
        (0xF9EA, "M", "離"),
        (0xF9EB, "M", "匿"),
        (0xF9EC, "M", "溺"),
        (0xF9ED, "M", "吝"),
        (0xF9EE, "M", "燐"),
        (0xF9EF, "M", "璘"),
        (0xF9F0, "M", "藺"),
        (0xF9F1, "M", "隣"),
        (0xF9F2, "M", "鱗"),
        (0xF9F3, "M", "麟"),
        (0xF9F4, "M", "林"),
        (0xF9F5, "M", "淋"),
        (0xF9F6, "M", "臨"),
        (0xF9F7, "M", "立"),
        (0xF9F8, "M", "笠"),
        (0xF9F9, "M", "粒"),
        (0xF9FA, "M", "狀"),
        (0xF9FB, "M", "炙"),
        (0xF9FC, "M", "識"),
        (0xF9FD, "M", "什"),
        (0xF9FE, "M", "茶"),
        (0xF9FF, "M", "刺"),
        (0xFA00, "M", "切"),
        (0xFA01, "M", "度"),
        (0xFA02, "M", "拓"),
        (0xFA03, "M", "糖"),
        (0xFA04, "M", "宅"),
        (0xFA05, "M", "洞"),
        (0xFA06, "M", "暴"),
        (0xFA07, "M", "輻"),
        (0xFA08, "M", "行"),
        (0xFA09, "M", "降"),
        (0xFA0A, "M", "見"),
        (0xFA0B, "M", "廓"),
        (0xFA0C, "M", "兀"),
        (0xFA0D, "M", "嗀"),
        (0xFA0E, "V"),
        (0xFA10, "M", "塚"),
        (0xFA11, "V"),
        (0xFA12, "M", "晴"),
    ]


def _seg_42() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0xFA13, "V"),
        (0xFA15, "M", "凞"),
        (0xFA16, "M", "猪"),
        (0xFA17, "M", "益"),
        (0xFA18, "M", "礼"),
        (0xFA19, "M", "神"),
        (0xFA1A, "M", "祥"),
        (0xFA1B, "M", "福"),
        (0xFA1C, "M", "靖"),
        (0xFA1D, "M", "精"),
        (0xFA1E, "M", "羽"),
        (0xFA1F, "V"),
        (0xFA20, "M", "蘒"),
        (0xFA21, "V"),
        (0xFA22, "M", "諸"),
        (0xFA23, "V"),
        (0xFA25, "M", "逸"),
        (0xFA26, "M", "都"),
        (0xFA27, "V"),
        (0xFA2A, "M", "飯"),
        (0xFA2B, "M", "飼"),
        (0xFA2C, "M", "館"),
        (0xFA2D, "M", "鶴"),
        (0xFA2E, "M", "郞"),
        (0xFA2F, "M", "隷"),
        (0xFA30, "M", "侮"),
        (0xFA31, "M", "僧"),
        (0xFA32, "M", "免"),
        (0xFA33, "M", "勉"),
        (0xFA34, "M", "勤"),
        (0xFA35, "M", "卑"),
        (0xFA36, "M", "喝"),
        (0xFA37, "M", "嘆"),
        (0xFA38, "M", "器"),
        (0xFA39, "M", "塀"),
        (0xFA3A, "M", "墨"),
        (0xFA3B, "M", "層"),
        (0xFA3C, "M", "屮"),
        (0xFA3D, "M", "悔"),
        (0xFA3E, "M", "慨"),
        (0xFA3F, "M", "憎"),
        (0xFA40, "M", "懲"),
        (0xFA41, "M", "敏"),
        (0xFA42, "M", "既"),
        (0xFA43, "M", "暑"),
        (0xFA44, "M", "梅"),
        (0xFA45, "M", "海"),
        (0xFA46, "M", "渚"),
        (0xFA47, "M", "漢"),
        (0xFA48, "M", "煮"),
        (0xFA49, "M", "爫"),
        (0xFA4A, "M", "琢"),
        (0xFA4B, "M", "碑"),
        (0xFA4C, "M", "社"),
        (0xFA4D, "M", "祉"),
        (0xFA4E, "M", "祈"),
        (0xFA4F, "M", "祐"),
        (0xFA50, "M", "祖"),
        (0xFA51, "M", "祝"),
        (0xFA52, "M", "禍"),
        (0xFA53, "M", "禎"),
        (0xFA54, "M", "穀"),
        (0xFA55, "M", "突"),
        (0xFA56, "M", "節"),
        (0xFA57, "M", "練"),
        (0xFA58, "M", "縉"),
        (0xFA59, "M", "繁"),
        (0xFA5A, "M", "署"),
        (0xFA5B, "M", "者"),
        (0xFA5C, "M", "臭"),
        (0xFA5D, "M", "艹"),
        (0xFA5F, "M", "著"),
        (0xFA60, "M", "褐"),
        (0xFA61, "M", "視"),
        (0xFA62, "M", "謁"),
        (0xFA63, "M", "謹"),
        (0xFA64, "M", "賓"),
        (0xFA65, "M", "贈"),
        (0xFA66, "M", "辶"),
        (0xFA67, "M", "逸"),
        (0xFA68, "M", "難"),
        (0xFA69, "M", "響"),
        (0xFA6A, "M", "頻"),
        (0xFA6B, "M", "恵"),
        (0xFA6C, "M", "𤋮"),
        (0xFA6D, "M", "舘"),
        (0xFA6E, "X"),
        (0xFA70, "M", "並"),
        (0xFA71, "M", "况"),
        (0xFA72, "M", "全"),
        (0xFA73, "M", "侀"),
        (0xFA74, "M", "充"),
        (0xFA75, "M", "冀"),
        (0xFA76, "M", "勇"),
        (0xFA77, "M", "勺"),
        (0xFA78, "M", "喝"),
        (0xFA79, "M", "啕"),
        (0xFA7A, "M", "喙"),
        (0xFA7B, "M", "嗢"),
        (0xFA7C, "M", "塚"),
    ]


def _seg_43() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0xFA7D, "M", "墳"),
        (0xFA7E, "M", "奄"),
        (0xFA7F, "M", "奔"),
        (0xFA80, "M", "婢"),
        (0xFA81, "M", "嬨"),
        (0xFA82, "M", "廒"),
        (0xFA83, "M", "廙"),
        (0xFA84, "M", "彩"),
        (0xFA85, "M", "徭"),
        (0xFA86, "M", "惘"),
        (0xFA87, "M", "慎"),
        (0xFA88, "M", "愈"),
        (0xFA89, "M", "憎"),
        (0xFA8A, "M", "慠"),
        (0xFA8B, "M", "懲"),
        (0xFA8C, "M", "戴"),
        (0xFA8D, "M", "揄"),
        (0xFA8E, "M", "搜"),
        (0xFA8F, "M", "摒"),
        (0xFA90, "M", "敖"),
        (0xFA91, "M", "晴"),
        (0xFA92, "M", "朗"),
        (0xFA93, "M", "望"),
        (0xFA94, "M", "杖"),
        (0xFA95, "M", "歹"),
        (0xFA96, "M", "殺"),
        (0xFA97, "M", "流"),
        (0xFA98, "M", "滛"),
        (0xFA99, "M", "滋"),
        (0xFA9A, "M", "漢"),
        (0xFA9B, "M", "瀞"),
        (0xFA9C, "M", "煮"),
        (0xFA9D, "M", "瞧"),
        (0xFA9E, "M", "爵"),
        (0xFA9F, "M", "犯"),
        (0xFAA0, "M", "猪"),
        (0xFAA1, "M", "瑱"),
        (0xFAA2, "M", "甆"),
        (0xFAA3, "M", "画"),
        (0xFAA4, "M", "瘝"),
        (0xFAA5, "M", "瘟"),
        (0xFAA6, "M", "益"),
        (0xFAA7, "M", "盛"),
        (0xFAA8, "M", "直"),
        (0xFAA9, "M", "睊"),
        (0xFAAA, "M", "着"),
        (0xFAAB, "M", "磌"),
        (0xFAAC, "M", "窱"),
        (0xFAAD, "M", "節"),
        (0xFAAE, "M", "类"),
        (0xFAAF, "M", "絛"),
        (0xFAB0, "M", "練"),
        (0xFAB1, "M", "缾"),
        (0xFAB2, "M", "者"),
        (0xFAB3, "M", "荒"),
        (0xFAB4, "M", "華"),
        (0xFAB5, "M", "蝹"),
        (0xFAB6, "M", "襁"),
        (0xFAB7, "M", "覆"),
        (0xFAB8, "M", "視"),
        (0xFAB9, "M", "調"),
        (0xFABA, "M", "諸"),
        (0xFABB, "M", "請"),
        (0xFABC, "M", "謁"),
        (0xFABD, "M", "諾"),
        (0xFABE, "M", "諭"),
        (0xFABF, "M", "謹"),
        (0xFAC0, "M", "變"),
        (0xFAC1, "M", "贈"),
        (0xFAC2, "M", "輸"),
        (0xFAC3, "M", "遲"),
        (0xFAC4, "M", "醙"),
        (0xFAC5, "M", "鉶"),
        (0xFAC6, "M", "陼"),
        (0xFAC7, "M", "難"),
        (0xFAC8, "M", "靖"),
        (0xFAC9, "M", "韛"),
        (0xFACA, "M", "響"),
        (0xFACB, "M", "頋"),
        (0xFACC, "M", "頻"),
        (0xFACD, "M", "鬒"),
        (0xFACE, "M", "龜"),
        (0xFACF, "M", "𢡊"),
        (0xFAD0, "M", "𢡄"),
        (0xFAD1, "M", "𣏕"),
        (0xFAD2, "M", "㮝"),
        (0xFAD3, "M", "䀘"),
        (0xFAD4, "M", "䀹"),
        (0xFAD5, "M", "𥉉"),
        (0xFAD6, "M", "𥳐"),
        (0xFAD7, "M", "𧻓"),
        (0xFAD8, "M", "齃"),
        (0xFAD9, "M", "龎"),
        (0xFADA, "X"),
        (0xFB00, "M", "ff"),
        (0xFB01, "M", "fi"),
        (0xFB02, "M", "fl"),
        (0xFB03, "M", "ffi"),
        (0xFB04, "M", "ffl"),
        (0xFB05, "M", "st"),
    ]


def _seg_44() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0xFB07, "X"),
        (0xFB13, "M", "մն"),
        (0xFB14, "M", "մե"),
        (0xFB15, "M", "մի"),
        (0xFB16, "M", "վն"),
        (0xFB17, "M", "մխ"),
        (0xFB18, "X"),
        (0xFB1D, "M", "יִ"),
        (0xFB1E, "V"),
        (0xFB1F, "M", "ײַ"),
        (0xFB20, "M", "ע"),
        (0xFB21, "M", "א"),
        (0xFB22, "M", "ד"),
        (0xFB23, "M", "ה"),
        (0xFB24, "M", "כ"),
        (0xFB25, "M", "ל"),
        (0xFB26, "M", "ם"),
        (0xFB27, "M", "ר"),
        (0xFB28, "M", "ת"),
        (0xFB29, "3", "+"),
        (0xFB2A, "M", "שׁ"),
        (0xFB2B, "M", "שׂ"),
        (0xFB2C, "M", "שּׁ"),
        (0xFB2D, "M", "שּׂ"),
        (0xFB2E, "M", "אַ"),
        (0xFB2F, "M", "אָ"),
        (0xFB30, "M", "אּ"),
        (0xFB31, "M", "בּ"),
        (0xFB32, "M", "גּ"),
        (0xFB33, "M", "דּ"),
        (0xFB34, "M", "הּ"),
        (0xFB35, "M", "וּ"),
        (0xFB36, "M", "זּ"),
        (0xFB37, "X"),
        (0xFB38, "M", "טּ"),
        (0xFB39, "M", "יּ"),
        (0xFB3A, "M", "ךּ"),
        (0xFB3B, "M", "כּ"),
        (0xFB3C, "M", "לּ"),
        (0xFB3D, "X"),
        (0xFB3E, "M", "מּ"),
        (0xFB3F, "X"),
        (0xFB40, "M", "נּ"),
        (0xFB41, "M", "סּ"),
        (0xFB42, "X"),
        (0xFB43, "M", "ףּ"),
        (0xFB44, "M", "פּ"),
        (0xFB45, "X"),
        (0xFB46, "M", "צּ"),
        (0xFB47, "M", "קּ"),
        (0xFB48, "M", "רּ"),
        (0xFB49, "M", "שּ"),
        (0xFB4A, "M", "תּ"),
        (0xFB4B, "M", "וֹ"),
        (0xFB4C, "M", "בֿ"),
        (0xFB4D, "M", "כֿ"),
        (0xFB4E, "M", "פֿ"),
        (0xFB4F, "M", "אל"),
        (0xFB50, "M", "ٱ"),
        (0xFB52, "M", "ٻ"),
        (0xFB56, "M", "پ"),
        (0xFB5A, "M", "ڀ"),
        (0xFB5E, "M", "ٺ"),
        (0xFB62, "M", "ٿ"),
        (0xFB66, "M", "ٹ"),
        (0xFB6A, "M", "ڤ"),
        (0xFB6E, "M", "ڦ"),
        (0xFB72, "M", "ڄ"),
        (0xFB76, "M", "ڃ"),
        (0xFB7A, "M", "چ"),
        (0xFB7E, "M", "ڇ"),
        (0xFB82, "M", "ڍ"),
        (0xFB84, "M", "ڌ"),
        (0xFB86, "M", "ڎ"),
        (0xFB88, "M", "ڈ"),
        (0xFB8A, "M", "ژ"),
        (0xFB8C, "M", "ڑ"),
        (0xFB8E, "M", "ک"),
        (0xFB92, "M", "گ"),
        (0xFB96, "M", "ڳ"),
        (0xFB9A, "M", "ڱ"),
        (0xFB9E, "M", "ں"),
        (0xFBA0, "M", "ڻ"),
        (0xFBA4, "M", "ۀ"),
        (0xFBA6, "M", "ہ"),
        (0xFBAA, "M", "ھ"),
        (0xFBAE, "M", "ے"),
        (0xFBB0, "M", "ۓ"),
        (0xFBB2, "V"),
        (0xFBC3, "X"),
        (0xFBD3, "M", "ڭ"),
        (0xFBD7, "M", "ۇ"),
        (0xFBD9, "M", "ۆ"),
        (0xFBDB, "M", "ۈ"),
        (0xFBDD, "M", "ۇٴ"),
        (0xFBDE, "M", "ۋ"),
        (0xFBE0, "M", "ۅ"),
        (0xFBE2, "M", "ۉ"),
        (0xFBE4, "M", "ې"),
        (0xFBE8, "M", "ى"),
    ]


def _seg_45() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0xFBEA, "M", "ئا"),
        (0xFBEC, "M", "ئە"),
        (0xFBEE, "M", "ئو"),
        (0xFBF0, "M", "ئۇ"),
        (0xFBF2, "M", "ئۆ"),
        (0xFBF4, "M", "ئۈ"),
        (0xFBF6, "M", "ئې"),
        (0xFBF9, "M", "ئى"),
        (0xFBFC, "M", "ی"),
        (0xFC00, "M", "ئج"),
        (0xFC01, "M", "ئح"),
        (0xFC02, "M", "ئم"),
        (0xFC03, "M", "ئى"),
        (0xFC04, "M", "ئي"),
        (0xFC05, "M", "بج"),
        (0xFC06, "M", "بح"),
        (0xFC07, "M", "بخ"),
        (0xFC08, "M", "بم"),
        (0xFC09, "M", "بى"),
        (0xFC0A, "M", "بي"),
        (0xFC0B, "M", "تج"),
        (0xFC0C, "M", "تح"),
        (0xFC0D, "M", "تخ"),
        (0xFC0E, "M", "تم"),
        (0xFC0F, "M", "تى"),
        (0xFC10, "M", "تي"),
        (0xFC11, "M", "ثج"),
        (0xFC12, "M", "ثم"),
        (0xFC13, "M", "ثى"),
        (0xFC14, "M", "ثي"),
        (0xFC15, "M", "جح"),
        (0xFC16, "M", "جم"),
        (0xFC17, "M", "حج"),
        (0xFC18, "M", "حم"),
        (0xFC19, "M", "خج"),
        (0xFC1A, "M", "خح"),
        (0xFC1B, "M", "خم"),
        (0xFC1C, "M", "سج"),
        (0xFC1D, "M", "سح"),
        (0xFC1E, "M", "سخ"),
        (0xFC1F, "M", "سم"),
        (0xFC20, "M", "صح"),
        (0xFC21, "M", "صم"),
        (0xFC22, "M", "ضج"),
        (0xFC23, "M", "ضح"),
        (0xFC24, "M", "ضخ"),
        (0xFC25, "M", "ضم"),
        (0xFC26, "M", "طح"),
        (0xFC27, "M", "طم"),
        (0xFC28, "M", "ظم"),
        (0xFC29, "M", "عج"),
        (0xFC2A, "M", "عم"),
        (0xFC2B, "M", "غج"),
        (0xFC2C, "M", "غم"),
        (0xFC2D, "M", "فج"),
        (0xFC2E, "M", "فح"),
        (0xFC2F, "M", "فخ"),
        (0xFC30, "M", "فم"),
        (0xFC31, "M", "فى"),
        (0xFC32, "M", "في"),
        (0xFC33, "M", "قح"),
        (0xFC34, "M", "قم"),
        (0xFC35, "M", "قى"),
        (0xFC36, "M", "قي"),
        (0xFC37, "M", "كا"),
        (0xFC38, "M", "كج"),
        (0xFC39, "M", "كح"),
        (0xFC3A, "M", "كخ"),
        (0xFC3B, "M", "كل"),
        (0xFC3C, "M", "كم"),
        (0xFC3D, "M", "كى"),
        (0xFC3E, "M", "كي"),
        (0xFC3F, "M", "لج"),
        (0xFC40, "M", "لح"),
        (0xFC41, "M", "لخ"),
        (0xFC42, "M", "لم"),
        (0xFC43, "M", "لى"),
        (0xFC44, "M", "لي"),
        (0xFC45, "M", "مج"),
        (0xFC46, "M", "مح"),
        (0xFC47, "M", "مخ"),
        (0xFC48, "M", "مم"),
        (0xFC49, "M", "مى"),
        (0xFC4A, "M", "مي"),
        (0xFC4B, "M", "نج"),
        (0xFC4C, "M", "نح"),
        (0xFC4D, "M", "نخ"),
        (0xFC4E, "M", "نم"),
        (0xFC4F, "M", "نى"),
        (0xFC50, "M", "ني"),
        (0xFC51, "M", "هج"),
        (0xFC52, "M", "هم"),
        (0xFC53, "M", "هى"),
        (0xFC54, "M", "هي"),
        (0xFC55, "M", "يج"),
        (0xFC56, "M", "يح"),
        (0xFC57, "M", "يخ"),
        (0xFC58, "M", "يم"),
        (0xFC59, "M", "يى"),
        (0xFC5A, "M", "يي"),
    ]


def _seg_46() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0xFC5B, "M", "ذٰ"),
        (0xFC5C, "M", "رٰ"),
        (0xFC5D, "M", "ىٰ"),
        (0xFC5E, "3", " ٌّ"),
        (0xFC5F, "3", " ٍّ"),
        (0xFC60, "3", " َّ"),
        (0xFC61, "3", " ُّ"),
        (0xFC62, "3", " ِّ"),
        (0xFC63, "3", " ّٰ"),
        (0xFC64, "M", "ئر"),
        (0xFC65, "M", "ئز"),
        (0xFC66, "M", "ئم"),
        (0xFC67, "M", "ئن"),
        (0xFC68, "M", "ئى"),
        (0xFC69, "M", "ئي"),
        (0xFC6A, "M", "بر"),
        (0xFC6B, "M", "بز"),
        (0xFC6C, "M", "بم"),
        (0xFC6D, "M", "بن"),
        (0xFC6E, "M", "بى"),
        (0xFC6F, "M", "بي"),
        (0xFC70, "M", "تر"),
        (0xFC71, "M", "تز"),
        (0xFC72, "M", "تم"),
        (0xFC73, "M", "تن"),
        (0xFC74, "M", "تى"),
        (0xFC75, "M", "تي"),
        (0xFC76, "M", "ثر"),
        (0xFC77, "M", "ثز"),
        (0xFC78, "M", "ثم"),
        (0xFC79, "M", "ثن"),
        (0xFC7A, "M", "ثى"),
        (0xFC7B, "M", "ثي"),
        (0xFC7C, "M", "فى"),
        (0xFC7D, "M", "في"),
        (0xFC7E, "M", "قى"),
        (0xFC7F, "M", "قي"),
        (0xFC80, "M", "كا"),
        (0xFC81, "M", "كل"),
        (0xFC82, "M", "كم"),
        (0xFC83, "M", "كى"),
        (0xFC84, "M", "كي"),
        (0xFC85, "M", "لم"),
        (0xFC86, "M", "لى"),
        (0xFC87, "M", "لي"),
        (0xFC88, "M", "ما"),
        (0xFC89, "M", "مم"),
        (0xFC8A, "M", "نر"),
        (0xFC8B, "M", "نز"),
        (0xFC8C, "M", "نم"),
        (0xFC8D, "M", "نن"),
        (0xFC8E, "M", "نى"),
        (0xFC8F, "M", "ني"),
        (0xFC90, "M", "ىٰ"),
        (0xFC91, "M", "ير"),
        (0xFC92, "M", "يز"),
        (0xFC93, "M", "يم"),
        (0xFC94, "M", "ين"),
        (0xFC95, "M", "يى"),
        (0xFC96, "M", "يي"),
        (0xFC97, "M", "ئج"),
        (0xFC98, "M", "ئح"),
        (0xFC99, "M", "ئخ"),
        (0xFC9A, "M", "ئم"),
        (0xFC9B, "M", "ئه"),
        (0xFC9C, "M", "بج"),
        (0xFC9D, "M", "بح"),
        (0xFC9E, "M", "بخ"),
        (0xFC9F, "M", "بم"),
        (0xFCA0, "M", "به"),
        (0xFCA1, "M", "تج"),
        (0xFCA2, "M", "تح"),
        (0xFCA3, "M", "تخ"),
        (0xFCA4, "M", "تم"),
        (0xFCA5, "M", "ته"),
        (0xFCA6, "M", "ثم"),
        (0xFCA7, "M", "جح"),
        (0xFCA8, "M", "جم"),
        (0xFCA9, "M", "حج"),
        (0xFCAA, "M", "حم"),
        (0xFCAB, "M", "خج"),
        (0xFCAC, "M", "خم"),
        (0xFCAD, "M", "سج"),
        (0xFCAE, "M", "سح"),
        (0xFCAF, "M", "سخ"),
        (0xFCB0, "M", "سم"),
        (0xFCB1, "M", "صح"),
        (0xFCB2, "M", "صخ"),
        (0xFCB3, "M", "صم"),
        (0xFCB4, "M", "ضج"),
        (0xFCB5, "M", "ضح"),
        (0xFCB6, "M", "ضخ"),
        (0xFCB7, "M", "ضم"),
        (0xFCB8, "M", "طح"),
        (0xFCB9, "M", "ظم"),
        (0xFCBA, "M", "عج"),
        (0xFCBB, "M", "عم"),
        (0xFCBC, "M", "غج"),
        (0xFCBD, "M", "غم"),
        (0xFCBE, "M", "فج"),
    ]


def _seg_47() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0xFCBF, "M", "فح"),
        (0xFCC0, "M", "فخ"),
        (0xFCC1, "M", "فم"),
        (0xFCC2, "M", "قح"),
        (0xFCC3, "M", "قم"),
        (0xFCC4, "M", "كج"),
        (0xFCC5, "M", "كح"),
        (0xFCC6, "M", "كخ"),
        (0xFCC7, "M", "كل"),
        (0xFCC8, "M", "كم"),
        (0xFCC9, "M", "لج"),
        (0xFCCA, "M", "لح"),
        (0xFCCB, "M", "لخ"),
        (0xFCCC, "M", "لم"),
        (0xFCCD, "M", "له"),
        (0xFCCE, "M", "مج"),
        (0xFCCF, "M", "مح"),
        (0xFCD0, "M", "مخ"),
        (0xFCD1, "M", "مم"),
        (0xFCD2, "M", "نج"),
        (0xFCD3, "M", "نح"),
        (0xFCD4, "M", "نخ"),
        (0xFCD5, "M", "نم"),
        (0xFCD6, "M", "نه"),
        (0xFCD7, "M", "هج"),
        (0xFCD8, "M", "هم"),
        (0xFCD9, "M", "هٰ"),
        (0xFCDA, "M", "يج"),
        (0xFCDB, "M", "يح"),
        (0xFCDC, "M", "يخ"),
        (0xFCDD, "M", "يم"),
        (0xFCDE, "M", "يه"),
        (0xFCDF, "M", "ئم"),
        (0xFCE0, "M", "ئه"),
        (0xFCE1, "M", "بم"),
        (0xFCE2, "M", "به"),
        (0xFCE3, "M", "تم"),
        (0xFCE4, "M", "ته"),
        (0xFCE5, "M", "ثم"),
        (0xFCE6, "M", "ثه"),
        (0xFCE7, "M", "سم"),
        (0xFCE8, "M", "سه"),
        (0xFCE9, "M", "شم"),
        (0xFCEA, "M", "شه"),
        (0xFCEB, "M", "كل"),
        (0xFCEC, "M", "كم"),
        (0xFCED, "M", "لم"),
        (0xFCEE, "M", "نم"),
        (0xFCEF, "M", "نه"),
        (0xFCF0, "M", "يم"),
        (0xFCF1, "M", "يه"),
        (0xFCF2, "M", "ـَّ"),
        (0xFCF3, "M", "ـُّ"),
        (0xFCF4, "M", "ـِّ"),
        (0xFCF5, "M", "طى"),
        (0xFCF6, "M", "طي"),
        (0xFCF7, "M", "عى"),
        (0xFCF8, "M", "عي"),
        (0xFCF9, "M", "غى"),
        (0xFCFA, "M", "غي"),
        (0xFCFB, "M", "سى"),
        (0xFCFC, "M", "سي"),
        (0xFCFD, "M", "شى"),
        (0xFCFE, "M", "شي"),
        (0xFCFF, "M", "حى"),
        (0xFD00, "M", "حي"),
        (0xFD01, "M", "جى"),
        (0xFD02, "M", "جي"),
        (0xFD03, "M", "خى"),
        (0xFD04, "M", "خي"),
        (0xFD05, "M", "صى"),
        (0xFD06, "M", "صي"),
        (0xFD07, "M", "ضى"),
        (0xFD08, "M", "ضي"),
        (0xFD09, "M", "شج"),
        (0xFD0A, "M", "شح"),
        (0xFD0B, "M", "شخ"),
        (0xFD0C, "M", "شم"),
        (0xFD0D, "M", "شر"),
        (0xFD0E, "M", "سر"),
        (0xFD0F, "M", "صر"),
        (0xFD10, "M", "ضر"),
        (0xFD11, "M", "طى"),
        (0xFD12, "M", "طي"),
        (0xFD13, "M", "عى"),
        (0xFD14, "M", "عي"),
        (0xFD15, "M", "غى"),
        (0xFD16, "M", "غي"),
        (0xFD17, "M", "سى"),
        (0xFD18, "M", "سي"),
        (0xFD19, "M", "شى"),
        (0xFD1A, "M", "شي"),
        (0xFD1B, "M", "حى"),
        (0xFD1C, "M", "حي"),
        (0xFD1D, "M", "جى"),
        (0xFD1E, "M", "جي"),
        (0xFD1F, "M", "خى"),
        (0xFD20, "M", "خي"),
        (0xFD21, "M", "صى"),
        (0xFD22, "M", "صي"),
    ]


def _seg_48() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0xFD23, "M", "ضى"),
        (0xFD24, "M", "ضي"),
        (0xFD25, "M", "شج"),
        (0xFD26, "M", "شح"),
        (0xFD27, "M", "شخ"),
        (0xFD28, "M", "شم"),
        (0xFD29, "M", "شر"),
        (0xFD2A, "M", "سر"),
        (0xFD2B, "M", "صر"),
        (0xFD2C, "M", "ضر"),
        (0xFD2D, "M", "شج"),
        (0xFD2E, "M", "شح"),
        (0xFD2F, "M", "شخ"),
        (0xFD30, "M", "شم"),
        (0xFD31, "M", "سه"),
        (0xFD32, "M", "شه"),
        (0xFD33, "M", "طم"),
        (0xFD34, "M", "سج"),
        (0xFD35, "M", "سح"),
        (0xFD36, "M", "سخ"),
        (0xFD37, "M", "شج"),
        (0xFD38, "M", "شح"),
        (0xFD39, "M", "شخ"),
        (0xFD3A, "M", "طم"),
        (0xFD3B, "M", "ظم"),
        (0xFD3C, "M", "اً"),
        (0xFD3E, "V"),
        (0xFD50, "M", "تجم"),
        (0xFD51, "M", "تحج"),
        (0xFD53, "M", "تحم"),
        (0xFD54, "M", "تخم"),
        (0xFD55, "M", "تمج"),
        (0xFD56, "M", "تمح"),
        (0xFD57, "M", "تمخ"),
        (0xFD58, "M", "جمح"),
        (0xFD5A, "M", "حمي"),
        (0xFD5B, "M", "حمى"),
        (0xFD5C, "M", "سحج"),
        (0xFD5D, "M", "سجح"),
        (0xFD5E, "M", "سجى"),
        (0xFD5F, "M", "سمح"),
        (0xFD61, "M", "سمج"),
        (0xFD62, "M", "سمم"),
        (0xFD64, "M", "صحح"),
        (0xFD66, "M", "صمم"),
        (0xFD67, "M", "شحم"),
        (0xFD69, "M", "شجي"),
        (0xFD6A, "M", "شمخ"),
        (0xFD6C, "M", "شمم"),
        (0xFD6E, "M", "ضحى"),
        (0xFD6F, "M", "ضخم"),
        (0xFD71, "M", "طمح"),
        (0xFD73, "M", "طمم"),
        (0xFD74, "M", "طمي"),
        (0xFD75, "M", "عجم"),
        (0xFD76, "M", "عمم"),
        (0xFD78, "M", "عمى"),
        (0xFD79, "M", "غمم"),
        (0xFD7A, "M", "غمي"),
        (0xFD7B, "M", "غمى"),
        (0xFD7C, "M", "فخم"),
        (0xFD7E, "M", "قمح"),
        (0xFD7F, "M", "قمم"),
        (0xFD80, "M", "لحم"),
        (0xFD81, "M", "لحي"),
        (0xFD82, "M", "لحى"),
        (0xFD83, "M", "لجج"),
        (0xFD85, "M", "لخم"),
        (0xFD87, "M", "لمح"),
        (0xFD89, "M", "محج"),
        (0xFD8A, "M", "محم"),
        (0xFD8B, "M", "محي"),
        (0xFD8C, "M", "مجح"),
        (0xFD8D, "M", "مجم"),
        (0xFD8E, "M", "مخج"),
        (0xFD8F, "M", "مخم"),
        (0xFD90, "X"),
        (0xFD92, "M", "مجخ"),
        (0xFD93, "M", "همج"),
        (0xFD94, "M", "همم"),
        (0xFD95, "M", "نحم"),
        (0xFD96, "M", "نحى"),
        (0xFD97, "M", "نجم"),
        (0xFD99, "M", "نجى"),
        (0xFD9A, "M", "نمي"),
        (0xFD9B, "M", "نمى"),
        (0xFD9C, "M", "يمم"),
        (0xFD9E, "M", "بخي"),
        (0xFD9F, "M", "تجي"),
        (0xFDA0, "M", "تجى"),
        (0xFDA1, "M", "تخي"),
        (0xFDA2, "M", "تخى"),
        (0xFDA3, "M", "تمي"),
        (0xFDA4, "M", "تمى"),
        (0xFDA5, "M", "جمي"),
        (0xFDA6, "M", "جحى"),
        (0xFDA7, "M", "جمى"),
        (0xFDA8, "M", "سخى"),
        (0xFDA9, "M", "صحي"),
        (0xFDAA, "M", "شحي"),
    ]


def _seg_49() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0xFDAB, "M", "ضحي"),
        (0xFDAC, "M", "لجي"),
        (0xFDAD, "M", "لمي"),
        (0xFDAE, "M", "يحي"),
        (0xFDAF, "M", "يجي"),
        (0xFDB0, "M", "يمي"),
        (0xFDB1, "M", "ممي"),
        (0xFDB2, "M", "قمي"),
        (0xFDB3, "M", "نحي"),
        (0xFDB4, "M", "قمح"),
        (0xFDB5, "M", "لحم"),
        (0xFDB6, "M", "عمي"),
        (0xFDB7, "M", "كمي"),
        (0xFDB8, "M", "نجح"),
        (0xFDB9, "M", "مخي"),
        (0xFDBA, "M", "لجم"),
        (0xFDBB, "M", "كمم"),
        (0xFDBC, "M", "لجم"),
        (0xFDBD, "M", "نجح"),
        (0xFDBE, "M", "جحي"),
        (0xFDBF, "M", "حجي"),
        (0xFDC0, "M", "مجي"),
        (0xFDC1, "M", "فمي"),
        (0xFDC2, "M", "بحي"),
        (0xFDC3, "M", "كمم"),
        (0xFDC4, "M", "عجم"),
        (0xFDC5, "M", "صمم"),
        (0xFDC6, "M", "سخي"),
        (0xFDC7, "M", "نجي"),
        (0xFDC8, "X"),
        (0xFDCF, "V"),
        (0xFDD0, "X"),
        (0xFDF0, "M", "صلے"),
        (0xFDF1, "M", "قلے"),
        (0xFDF2, "M", "الله"),
        (0xFDF3, "M", "اكبر"),
        (0xFDF4, "M", "محمد"),
        (0xFDF5, "M", "صلعم"),
        (0xFDF6, "M", "رسول"),
        (0xFDF7, "M", "عليه"),
        (0xFDF8, "M", "وسلم"),
        (0xFDF9, "M", "صلى"),
        (0xFDFA, "3", "صلى الله عليه وسلم"),
        (0xFDFB, "3", "جل جلاله"),
        (0xFDFC, "M", "ریال"),
        (0xFDFD, "V"),
        (0xFE00, "I"),
        (0xFE10, "3", ","),
        (0xFE11, "M", "、"),
        (0xFE12, "X"),
        (0xFE13, "3", ":"),
        (0xFE14, "3", ";"),
        (0xFE15, "3", "!"),
        (0xFE16, "3", "?"),
        (0xFE17, "M", "〖"),
        (0xFE18, "M", "〗"),
        (0xFE19, "X"),
        (0xFE20, "V"),
        (0xFE30, "X"),
        (0xFE31, "M", "—"),
        (0xFE32, "M", "–"),
        (0xFE33, "3", "_"),
        (0xFE35, "3", "("),
        (0xFE36, "3", ")"),
        (0xFE37, "3", "{"),
        (0xFE38, "3", "}"),
        (0xFE39, "M", "〔"),
        (0xFE3A, "M", "〕"),
        (0xFE3B, "M", "【"),
        (0xFE3C, "M", "】"),
        (0xFE3D, "M", "《"),
        (0xFE3E, "M", "》"),
        (0xFE3F, "M", "〈"),
        (0xFE40, "M", "〉"),
        (0xFE41, "M", "「"),
        (0xFE42, "M", "」"),
        (0xFE43, "M", "『"),
        (0xFE44, "M", "』"),
        (0xFE45, "V"),
        (0xFE47, "3", "["),
        (0xFE48, "3", "]"),
        (0xFE49, "3", " ̅"),
        (0xFE4D, "3", "_"),
        (0xFE50, "3", ","),
        (0xFE51, "M", "、"),
        (0xFE52, "X"),
        (0xFE54, "3", ";"),
        (0xFE55, "3", ":"),
        (0xFE56, "3", "?"),
        (0xFE57, "3", "!"),
        (0xFE58, "M", "—"),
        (0xFE59, "3", "("),
        (0xFE5A, "3", ")"),
        (0xFE5B, "3", "{"),
        (0xFE5C, "3", "}"),
        (0xFE5D, "M", "〔"),
        (0xFE5E, "M", "〕"),
        (0xFE5F, "3", "#"),
        (0xFE60, "3", "&"),
        (0xFE61, "3", "*"),
    ]


def _seg_50() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0xFE62, "3", "+"),
        (0xFE63, "M", "-"),
        (0xFE64, "3", "<"),
        (0xFE65, "3", ">"),
        (0xFE66, "3", "="),
        (0xFE67, "X"),
        (0xFE68, "3", "\\"),
        (0xFE69, "3", "$"),
        (0xFE6A, "3", "%"),
        (0xFE6B, "3", "@"),
        (0xFE6C, "X"),
        (0xFE70, "3", " ً"),
        (0xFE71, "M", "ـً"),
        (0xFE72, "3", " ٌ"),
        (0xFE73, "V"),
        (0xFE74, "3", " ٍ"),
        (0xFE75, "X"),
        (0xFE76, "3", " َ"),
        (0xFE77, "M", "ـَ"),
        (0xFE78, "3", " ُ"),
        (0xFE79, "M", "ـُ"),
        (0xFE7A, "3", " ِ"),
        (0xFE7B, "M", "ـِ"),
        (0xFE7C, "3", " ّ"),
        (0xFE7D, "M", "ـّ"),
        (0xFE7E, "3", " ْ"),
        (0xFE7F, "M", "ـْ"),
        (0xFE80, "M", "ء"),
        (0xFE81, "M", "آ"),
        (0xFE83, "M", "أ"),
        (0xFE85, "M", "ؤ"),
        (0xFE87, "M", "إ"),
        (0xFE89, "M", "ئ"),
        (0xFE8D, "M", "ا"),
        (0xFE8F, "M", "ب"),
        (0xFE93, "M", "ة"),
        (0xFE95, "M", "ت"),
        (0xFE99, "M", "ث"),
        (0xFE9D, "M", "ج"),
        (0xFEA1, "M", "ح"),
        (0xFEA5, "M", "خ"),
        (0xFEA9, "M", "د"),
        (0xFEAB, "M", "ذ"),
        (0xFEAD, "M", "ر"),
        (0xFEAF, "M", "ز"),
        (0xFEB1, "M", "س"),
        (0xFEB5, "M", "ش"),
        (0xFEB9, "M", "ص"),
        (0xFEBD, "M", "ض"),
        (0xFEC1, "M", "ط"),
        (0xFEC5, "M", "ظ"),
        (0xFEC9, "M", "ع"),
        (0xFECD, "M", "غ"),
        (0xFED1, "M", "ف"),
        (0xFED5, "M", "ق"),
        (0xFED9, "M", "ك"),
        (0xFEDD, "M", "ل"),
        (0xFEE1, "M", "م"),
        (0xFEE5, "M", "ن"),
        (0xFEE9, "M", "ه"),
        (0xFEED, "M", "و"),
        (0xFEEF, "M", "ى"),
        (0xFEF1, "M", "ي"),
        (0xFEF5, "M", "لآ"),
        (0xFEF7, "M", "لأ"),
        (0xFEF9, "M", "لإ"),
        (0xFEFB, "M", "لا"),
        (0xFEFD, "X"),
        (0xFEFF, "I"),
        (0xFF00, "X"),
        (0xFF01, "3", "!"),
        (0xFF02, "3", '"'),
        (0xFF03, "3", "#"),
        (0xFF04, "3", "$"),
        (0xFF05, "3", "%"),
        (0xFF06, "3", "&"),
        (0xFF07, "3", "'"),
        (0xFF08, "3", "("),
        (0xFF09, "3", ")"),
        (0xFF0A, "3", "*"),
        (0xFF0B, "3", "+"),
        (0xFF0C, "3", ","),
        (0xFF0D, "M", "-"),
        (0xFF0E, "M", "."),
        (0xFF0F, "3", "/"),
        (0xFF10, "M", "0"),
        (0xFF11, "M", "1"),
        (0xFF12, "M", "2"),
        (0xFF13, "M", "3"),
        (0xFF14, "M", "4"),
        (0xFF15, "M", "5"),
        (0xFF16, "M", "6"),
        (0xFF17, "M", "7"),
        (0xFF18, "M", "8"),
        (0xFF19, "M", "9"),
        (0xFF1A, "3", ":"),
        (0xFF1B, "3", ";"),
        (0xFF1C, "3", "<"),
        (0xFF1D, "3", "="),
        (0xFF1E, "3", ">"),
    ]


def _seg_51() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0xFF1F, "3", "?"),
        (0xFF20, "3", "@"),
        (0xFF21, "M", "a"),
        (0xFF22, "M", "b"),
        (0xFF23, "M", "c"),
        (0xFF24, "M", "d"),
        (0xFF25, "M", "e"),
        (0xFF26, "M", "f"),
        (0xFF27, "M", "g"),
        (0xFF28, "M", "h"),
        (0xFF29, "M", "i"),
        (0xFF2A, "M", "j"),
        (0xFF2B, "M", "k"),
        (0xFF2C, "M", "l"),
        (0xFF2D, "M", "m"),
        (0xFF2E, "M", "n"),
        (0xFF2F, "M", "o"),
        (0xFF30, "M", "p"),
        (0xFF31, "M", "q"),
        (0xFF32, "M", "r"),
        (0xFF33, "M", "s"),
        (0xFF34, "M", "t"),
        (0xFF35, "M", "u"),
        (0xFF36, "M", "v"),
        (0xFF37, "M", "w"),
        (0xFF38, "M", "x"),
        (0xFF39, "M", "y"),
        (0xFF3A, "M", "z"),
        (0xFF3B, "3", "["),
        (0xFF3C, "3", "\\"),
        (0xFF3D, "3", "]"),
        (0xFF3E, "3", "^"),
        (0xFF3F, "3", "_"),
        (0xFF40, "3", "`"),
        (0xFF41, "M", "a"),
        (0xFF42, "M", "b"),
        (0xFF43, "M", "c"),
        (0xFF44, "M", "d"),
        (0xFF45, "M", "e"),
        (0xFF46, "M", "f"),
        (0xFF47, "M", "g"),
        (0xFF48, "M", "h"),
        (0xFF49, "M", "i"),
        (0xFF4A, "M", "j"),
        (0xFF4B, "M", "k"),
        (0xFF4C, "M", "l"),
        (0xFF4D, "M", "m"),
        (0xFF4E, "M", "n"),
        (0xFF4F, "M", "o"),
        (0xFF50, "M", "p"),
        (0xFF51, "M", "q"),
        (0xFF52, "M", "r"),
        (0xFF53, "M", "s"),
        (0xFF54, "M", "t"),
        (0xFF55, "M", "u"),
        (0xFF56, "M", "v"),
        (0xFF57, "M", "w"),
        (0xFF58, "M", "x"),
        (0xFF59, "M", "y"),
        (0xFF5A, "M", "z"),
        (0xFF5B, "3", "{"),
        (0xFF5C, "3", "|"),
        (0xFF5D, "3", "}"),
        (0xFF5E, "3", "~"),
        (0xFF5F, "M", "⦅"),
        (0xFF60, "M", "⦆"),
        (0xFF61, "M", "."),
        (0xFF62, "M", "「"),
        (0xFF63, "M", "」"),
        (0xFF64, "M", "、"),
        (0xFF65, "M", "・"),
        (0xFF66, "M", "ヲ"),
        (0xFF67, "M", "ァ"),
        (0xFF68, "M", "ィ"),
        (0xFF69, "M", "ゥ"),
        (0xFF6A, "M", "ェ"),
        (0xFF6B, "M", "ォ"),
        (0xFF6C, "M", "ャ"),
        (0xFF6D, "M", "ュ"),
        (0xFF6E, "M", "ョ"),
        (0xFF6F, "M", "ッ"),
        (0xFF70, "M", "ー"),
        (0xFF71, "M", "ア"),
        (0xFF72, "M", "イ"),
        (0xFF73, "M", "ウ"),
        (0xFF74, "M", "エ"),
        (0xFF75, "M", "オ"),
        (0xFF76, "M", "カ"),
        (0xFF77, "M", "キ"),
        (0xFF78, "M", "ク"),
        (0xFF79, "M", "ケ"),
        (0xFF7A, "M", "コ"),
        (0xFF7B, "M", "サ"),
        (0xFF7C, "M", "シ"),
        (0xFF7D, "M", "ス"),
        (0xFF7E, "M", "セ"),
        (0xFF7F, "M", "ソ"),
        (0xFF80, "M", "タ"),
        (0xFF81, "M", "チ"),
        (0xFF82, "M", "ツ"),
    ]


def _seg_52() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0xFF83, "M", "テ"),
        (0xFF84, "M", "ト"),
        (0xFF85, "M", "ナ"),
        (0xFF86, "M", "ニ"),
        (0xFF87, "M", "ヌ"),
        (0xFF88, "M", "ネ"),
        (0xFF89, "M", "ノ"),
        (0xFF8A, "M", "ハ"),
        (0xFF8B, "M", "ヒ"),
        (0xFF8C, "M", "フ"),
        (0xFF8D, "M", "ヘ"),
        (0xFF8E, "M", "ホ"),
        (0xFF8F, "M", "マ"),
        (0xFF90, "M", "ミ"),
        (0xFF91, "M", "ム"),
        (0xFF92, "M", "メ"),
        (0xFF93, "M", "モ"),
        (0xFF94, "M", "ヤ"),
        (0xFF95, "M", "ユ"),
        (0xFF96, "M", "ヨ"),
        (0xFF97, "M", "ラ"),
        (0xFF98, "M", "リ"),
        (0xFF99, "M", "ル"),
        (0xFF9A, "M", "レ"),
        (0xFF9B, "M", "ロ"),
        (0xFF9C, "M", "ワ"),
        (0xFF9D, "M", "ン"),
        (0xFF9E, "M", "゙"),
        (0xFF9F, "M", "゚"),
        (0xFFA0, "X"),
        (0xFFA1, "M", "ᄀ"),
        (0xFFA2, "M", "ᄁ"),
        (0xFFA3, "M", "ᆪ"),
        (0xFFA4, "M", "ᄂ"),
        (0xFFA5, "M", "ᆬ"),
        (0xFFA6, "M", "ᆭ"),
        (0xFFA7, "M", "ᄃ"),
        (0xFFA8, "M", "ᄄ"),
        (0xFFA9, "M", "ᄅ"),
        (0xFFAA, "M", "ᆰ"),
        (0xFFAB, "M", "ᆱ"),
        (0xFFAC, "M", "ᆲ"),
        (0xFFAD, "M", "ᆳ"),
        (0xFFAE, "M", "ᆴ"),
        (0xFFAF, "M", "ᆵ"),
        (0xFFB0, "M", "ᄚ"),
        (0xFFB1, "M", "ᄆ"),
        (0xFFB2, "M", "ᄇ"),
        (0xFFB3, "M", "ᄈ"),
        (0xFFB4, "M", "ᄡ"),
        (0xFFB5, "M", "ᄉ"),
        (0xFFB6, "M", "ᄊ"),
        (0xFFB7, "M", "ᄋ"),
        (0xFFB8, "M", "ᄌ"),
        (0xFFB9, "M", "ᄍ"),
        (0xFFBA, "M", "ᄎ"),
        (0xFFBB, "M", "ᄏ"),
        (0xFFBC, "M", "ᄐ"),
        (0xFFBD, "M", "ᄑ"),
        (0xFFBE, "M", "ᄒ"),
        (0xFFBF, "X"),
        (0xFFC2, "M", "ᅡ"),
        (0xFFC3, "M", "ᅢ"),
        (0xFFC4, "M", "ᅣ"),
        (0xFFC5, "M", "ᅤ"),
        (0xFFC6, "M", "ᅥ"),
        (0xFFC7, "M", "ᅦ"),
        (0xFFC8, "X"),
        (0xFFCA, "M", "ᅧ"),
        (0xFFCB, "M", "ᅨ"),
        (0xFFCC, "M", "ᅩ"),
        (0xFFCD, "M", "ᅪ"),
        (0xFFCE, "M", "ᅫ"),
        (0xFFCF, "M", "ᅬ"),
        (0xFFD0, "X"),
        (0xFFD2, "M", "ᅭ"),
        (0xFFD3, "M", "ᅮ"),
        (0xFFD4, "M", "ᅯ"),
        (0xFFD5, "M", "ᅰ"),
        (0xFFD6, "M", "ᅱ"),
        (0xFFD7, "M", "ᅲ"),
        (0xFFD8, "X"),
        (0xFFDA, "M", "ᅳ"),
        (0xFFDB, "M", "ᅴ"),
        (0xFFDC, "M", "ᅵ"),
        (0xFFDD, "X"),
        (0xFFE0, "M", "¢"),
        (0xFFE1, "M", "£"),
        (0xFFE2, "M", "¬"),
        (0xFFE3, "3", " ̄"),
        (0xFFE4, "M", "¦"),
        (0xFFE5, "M", "¥"),
        (0xFFE6, "M", "₩"),
        (0xFFE7, "X"),
        (0xFFE8, "M", "│"),
        (0xFFE9, "M", "←"),
        (0xFFEA, "M", "↑"),
        (0xFFEB, "M", "→"),
        (0xFFEC, "M", "↓"),
        (0xFFED, "M", "■"),
    ]


def _seg_53() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0xFFEE, "M", "○"),
        (0xFFEF, "X"),
        (0x10000, "V"),
        (0x1000C, "X"),
        (0x1000D, "V"),
        (0x10027, "X"),
        (0x10028, "V"),
        (0x1003B, "X"),
        (0x1003C, "V"),
        (0x1003E, "X"),
        (0x1003F, "V"),
        (0x1004E, "X"),
        (0x10050, "V"),
        (0x1005E, "X"),
        (0x10080, "V"),
        (0x100FB, "X"),
        (0x10100, "V"),
        (0x10103, "X"),
        (0x10107, "V"),
        (0x10134, "X"),
        (0x10137, "V"),
        (0x1018F, "X"),
        (0x10190, "V"),
        (0x1019D, "X"),
        (0x101A0, "V"),
        (0x101A1, "X"),
        (0x101D0, "V"),
        (0x101FE, "X"),
        (0x10280, "V"),
        (0x1029D, "X"),
        (0x102A0, "V"),
        (0x102D1, "X"),
        (0x102E0, "V"),
        (0x102FC, "X"),
        (0x10300, "V"),
        (0x10324, "X"),
        (0x1032D, "V"),
        (0x1034B, "X"),
        (0x10350, "V"),
        (0x1037B, "X"),
        (0x10380, "V"),
        (0x1039E, "X"),
        (0x1039F, "V"),
        (0x103C4, "X"),
        (0x103C8, "V"),
        (0x103D6, "X"),
        (0x10400, "M", "𐐨"),
        (0x10401, "M", "𐐩"),
        (0x10402, "M", "𐐪"),
        (0x10403, "M", "𐐫"),
        (0x10404, "M", "𐐬"),
        (0x10405, "M", "𐐭"),
        (0x10406, "M", "𐐮"),
        (0x10407, "M", "𐐯"),
        (0x10408, "M", "𐐰"),
        (0x10409, "M", "𐐱"),
        (0x1040A, "M", "𐐲"),
        (0x1040B, "M", "𐐳"),
        (0x1040C, "M", "𐐴"),
        (0x1040D, "M", "𐐵"),
        (0x1040E, "M", "𐐶"),
        (0x1040F, "M", "𐐷"),
        (0x10410, "M", "𐐸"),
        (0x10411, "M", "𐐹"),
        (0x10412, "M", "𐐺"),
        (0x10413, "M", "𐐻"),
        (0x10414, "M", "𐐼"),
        (0x10415, "M", "𐐽"),
        (0x10416, "M", "𐐾"),
        (0x10417, "M", "𐐿"),
        (0x10418, "M", "𐑀"),
        (0x10419, "M", "𐑁"),
        (0x1041A, "M", "𐑂"),
        (0x1041B, "M", "𐑃"),
        (0x1041C, "M", "𐑄"),
        (0x1041D, "M", "𐑅"),
        (0x1041E, "M", "𐑆"),
        (0x1041F, "M", "𐑇"),
        (0x10420, "M", "𐑈"),
        (0x10421, "M", "𐑉"),
        (0x10422, "M", "𐑊"),
        (0x10423, "M", "𐑋"),
        (0x10424, "M", "𐑌"),
        (0x10425, "M", "𐑍"),
        (0x10426, "M", "𐑎"),
        (0x10427, "M", "𐑏"),
        (0x10428, "V"),
        (0x1049E, "X"),
        (0x104A0, "V"),
        (0x104AA, "X"),
        (0x104B0, "M", "𐓘"),
        (0x104B1, "M", "𐓙"),
        (0x104B2, "M", "𐓚"),
        (0x104B3, "M", "𐓛"),
        (0x104B4, "M", "𐓜"),
        (0x104B5, "M", "𐓝"),
        (0x104B6, "M", "𐓞"),
        (0x104B7, "M", "𐓟"),
        (0x104B8, "M", "𐓠"),
        (0x104B9, "M", "𐓡"),
    ]


def _seg_54() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x104BA, "M", "𐓢"),
        (0x104BB, "M", "𐓣"),
        (0x104BC, "M", "𐓤"),
        (0x104BD, "M", "𐓥"),
        (0x104BE, "M", "𐓦"),
        (0x104BF, "M", "𐓧"),
        (0x104C0, "M", "𐓨"),
        (0x104C1, "M", "𐓩"),
        (0x104C2, "M", "𐓪"),
        (0x104C3, "M", "𐓫"),
        (0x104C4, "M", "𐓬"),
        (0x104C5, "M", "𐓭"),
        (0x104C6, "M", "𐓮"),
        (0x104C7, "M", "𐓯"),
        (0x104C8, "M", "𐓰"),
        (0x104C9, "M", "𐓱"),
        (0x104CA, "M", "𐓲"),
        (0x104CB, "M", "𐓳"),
        (0x104CC, "M", "𐓴"),
        (0x104CD, "M", "𐓵"),
        (0x104CE, "M", "𐓶"),
        (0x104CF, "M", "𐓷"),
        (0x104D0, "M", "𐓸"),
        (0x104D1, "M", "𐓹"),
        (0x104D2, "M", "𐓺"),
        (0x104D3, "M", "𐓻"),
        (0x104D4, "X"),
        (0x104D8, "V"),
        (0x104FC, "X"),
        (0x10500, "V"),
        (0x10528, "X"),
        (0x10530, "V"),
        (0x10564, "X"),
        (0x1056F, "V"),
        (0x10570, "M", "𐖗"),
        (0x10571, "M", "𐖘"),
        (0x10572, "M", "𐖙"),
        (0x10573, "M", "𐖚"),
        (0x10574, "M", "𐖛"),
        (0x10575, "M", "𐖜"),
        (0x10576, "M", "𐖝"),
        (0x10577, "M", "𐖞"),
        (0x10578, "M", "𐖟"),
        (0x10579, "M", "𐖠"),
        (0x1057A, "M", "𐖡"),
        (0x1057B, "X"),
        (0x1057C, "M", "𐖣"),
        (0x1057D, "M", "𐖤"),
        (0x1057E, "M", "𐖥"),
        (0x1057F, "M", "𐖦"),
        (0x10580, "M", "𐖧"),
        (0x10581, "M", "𐖨"),
        (0x10582, "M", "𐖩"),
        (0x10583, "M", "𐖪"),
        (0x10584, "M", "𐖫"),
        (0x10585, "M", "𐖬"),
        (0x10586, "M", "𐖭"),
        (0x10587, "M", "𐖮"),
        (0x10588, "M", "𐖯"),
        (0x10589, "M", "𐖰"),
        (0x1058A, "M", "𐖱"),
        (0x1058B, "X"),
        (0x1058C, "M", "𐖳"),
        (0x1058D, "M", "𐖴"),
        (0x1058E, "M", "𐖵"),
        (0x1058F, "M", "𐖶"),
        (0x10590, "M", "𐖷"),
        (0x10591, "M", "𐖸"),
        (0x10592, "M", "𐖹"),
        (0x10593, "X"),
        (0x10594, "M", "𐖻"),
        (0x10595, "M", "𐖼"),
        (0x10596, "X"),
        (0x10597, "V"),
        (0x105A2, "X"),
        (0x105A3, "V"),
        (0x105B2, "X"),
        (0x105B3, "V"),
        (0x105BA, "X"),
        (0x105BB, "V"),
        (0x105BD, "X"),
        (0x10600, "V"),
        (0x10737, "X"),
        (0x10740, "V"),
        (0x10756, "X"),
        (0x10760, "V"),
        (0x10768, "X"),
        (0x10780, "V"),
        (0x10781, "M", "ː"),
        (0x10782, "M", "ˑ"),
        (0x10783, "M", "æ"),
        (0x10784, "M", "ʙ"),
        (0x10785, "M", "ɓ"),
        (0x10786, "X"),
        (0x10787, "M", "ʣ"),
        (0x10788, "M", "ꭦ"),
        (0x10789, "M", "ʥ"),
        (0x1078A, "M", "ʤ"),
        (0x1078B, "M", "ɖ"),
        (0x1078C, "M", "ɗ"),
    ]


def _seg_55() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1078D, "M", "ᶑ"),
        (0x1078E, "M", "ɘ"),
        (0x1078F, "M", "ɞ"),
        (0x10790, "M", "ʩ"),
        (0x10791, "M", "ɤ"),
        (0x10792, "M", "ɢ"),
        (0x10793, "M", "ɠ"),
        (0x10794, "M", "ʛ"),
        (0x10795, "M", "ħ"),
        (0x10796, "M", "ʜ"),
        (0x10797, "M", "ɧ"),
        (0x10798, "M", "ʄ"),
        (0x10799, "M", "ʪ"),
        (0x1079A, "M", "ʫ"),
        (0x1079B, "M", "ɬ"),
        (0x1079C, "M", "𝼄"),
        (0x1079D, "M", "ꞎ"),
        (0x1079E, "M", "ɮ"),
        (0x1079F, "M", "𝼅"),
        (0x107A0, "M", "ʎ"),
        (0x107A1, "M", "𝼆"),
        (0x107A2, "M", "ø"),
        (0x107A3, "M", "ɶ"),
        (0x107A4, "M", "ɷ"),
        (0x107A5, "M", "q"),
        (0x107A6, "M", "ɺ"),
        (0x107A7, "M", "𝼈"),
        (0x107A8, "M", "ɽ"),
        (0x107A9, "M", "ɾ"),
        (0x107AA, "M", "ʀ"),
        (0x107AB, "M", "ʨ"),
        (0x107AC, "M", "ʦ"),
        (0x107AD, "M", "ꭧ"),
        (0x107AE, "M", "ʧ"),
        (0x107AF, "M", "ʈ"),
        (0x107B0, "M", "ⱱ"),
        (0x107B1, "X"),
        (0x107B2, "M", "ʏ"),
        (0x107B3, "M", "ʡ"),
        (0x107B4, "M", "ʢ"),
        (0x107B5, "M", "ʘ"),
        (0x107B6, "M", "ǀ"),
        (0x107B7, "M", "ǁ"),
        (0x107B8, "M", "ǂ"),
        (0x107B9, "M", "𝼊"),
        (0x107BA, "M", "𝼞"),
        (0x107BB, "X"),
        (0x10800, "V"),
        (0x10806, "X"),
        (0x10808, "V"),
        (0x10809, "X"),
        (0x1080A, "V"),
        (0x10836, "X"),
        (0x10837, "V"),
        (0x10839, "X"),
        (0x1083C, "V"),
        (0x1083D, "X"),
        (0x1083F, "V"),
        (0x10856, "X"),
        (0x10857, "V"),
        (0x1089F, "X"),
        (0x108A7, "V"),
        (0x108B0, "X"),
        (0x108E0, "V"),
        (0x108F3, "X"),
        (0x108F4, "V"),
        (0x108F6, "X"),
        (0x108FB, "V"),
        (0x1091C, "X"),
        (0x1091F, "V"),
        (0x1093A, "X"),
        (0x1093F, "V"),
        (0x10940, "X"),
        (0x10980, "V"),
        (0x109B8, "X"),
        (0x109BC, "V"),
        (0x109D0, "X"),
        (0x109D2, "V"),
        (0x10A04, "X"),
        (0x10A05, "V"),
        (0x10A07, "X"),
        (0x10A0C, "V"),
        (0x10A14, "X"),
        (0x10A15, "V"),
        (0x10A18, "X"),
        (0x10A19, "V"),
        (0x10A36, "X"),
        (0x10A38, "V"),
        (0x10A3B, "X"),
        (0x10A3F, "V"),
        (0x10A49, "X"),
        (0x10A50, "V"),
        (0x10A59, "X"),
        (0x10A60, "V"),
        (0x10AA0, "X"),
        (0x10AC0, "V"),
        (0x10AE7, "X"),
        (0x10AEB, "V"),
        (0x10AF7, "X"),
        (0x10B00, "V"),
    ]


def _seg_56() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x10B36, "X"),
        (0x10B39, "V"),
        (0x10B56, "X"),
        (0x10B58, "V"),
        (0x10B73, "X"),
        (0x10B78, "V"),
        (0x10B92, "X"),
        (0x10B99, "V"),
        (0x10B9D, "X"),
        (0x10BA9, "V"),
        (0x10BB0, "X"),
        (0x10C00, "V"),
        (0x10C49, "X"),
        (0x10C80, "M", "𐳀"),
        (0x10C81, "M", "𐳁"),
        (0x10C82, "M", "𐳂"),
        (0x10C83, "M", "𐳃"),
        (0x10C84, "M", "𐳄"),
        (0x10C85, "M", "𐳅"),
        (0x10C86, "M", "𐳆"),
        (0x10C87, "M", "𐳇"),
        (0x10C88, "M", "𐳈"),
        (0x10C89, "M", "𐳉"),
        (0x10C8A, "M", "𐳊"),
        (0x10C8B, "M", "𐳋"),
        (0x10C8C, "M", "𐳌"),
        (0x10C8D, "M", "𐳍"),
        (0x10C8E, "M", "𐳎"),
        (0x10C8F, "M", "𐳏"),
        (0x10C90, "M", "𐳐"),
        (0x10C91, "M", "𐳑"),
        (0x10C92, "M", "𐳒"),
        (0x10C93, "M", "𐳓"),
        (0x10C94, "M", "𐳔"),
        (0x10C95, "M", "𐳕"),
        (0x10C96, "M", "𐳖"),
        (0x10C97, "M", "𐳗"),
        (0x10C98, "M", "𐳘"),
        (0x10C99, "M", "𐳙"),
        (0x10C9A, "M", "𐳚"),
        (0x10C9B, "M", "𐳛"),
        (0x10C9C, "M", "𐳜"),
        (0x10C9D, "M", "𐳝"),
        (0x10C9E, "M", "𐳞"),
        (0x10C9F, "M", "𐳟"),
        (0x10CA0, "M", "𐳠"),
        (0x10CA1, "M", "𐳡"),
        (0x10CA2, "M", "𐳢"),
        (0x10CA3, "M", "𐳣"),
        (0x10CA4, "M", "𐳤"),
        (0x10CA5, "M", "𐳥"),
        (0x10CA6, "M", "𐳦"),
        (0x10CA7, "M", "𐳧"),
        (0x10CA8, "M", "𐳨"),
        (0x10CA9, "M", "𐳩"),
        (0x10CAA, "M", "𐳪"),
        (0x10CAB, "M", "𐳫"),
        (0x10CAC, "M", "𐳬"),
        (0x10CAD, "M", "𐳭"),
        (0x10CAE, "M", "𐳮"),
        (0x10CAF, "M", "𐳯"),
        (0x10CB0, "M", "𐳰"),
        (0x10CB1, "M", "𐳱"),
        (0x10CB2, "M", "𐳲"),
        (0x10CB3, "X"),
        (0x10CC0, "V"),
        (0x10CF3, "X"),
        (0x10CFA, "V"),
        (0x10D28, "X"),
        (0x10D30, "V"),
        (0x10D3A, "X"),
        (0x10E60, "V"),
        (0x10E7F, "X"),
        (0x10E80, "V"),
        (0x10EAA, "X"),
        (0x10EAB, "V"),
        (0x10EAE, "X"),
        (0x10EB0, "V"),
        (0x10EB2, "X"),
        (0x10EFD, "V"),
        (0x10F28, "X"),
        (0x10F30, "V"),
        (0x10F5A, "X"),
        (0x10F70, "V"),
        (0x10F8A, "X"),
        (0x10FB0, "V"),
        (0x10FCC, "X"),
        (0x10FE0, "V"),
        (0x10FF7, "X"),
        (0x11000, "V"),
        (0x1104E, "X"),
        (0x11052, "V"),
        (0x11076, "X"),
        (0x1107F, "V"),
        (0x110BD, "X"),
        (0x110BE, "V"),
        (0x110C3, "X"),
        (0x110D0, "V"),
        (0x110E9, "X"),
        (0x110F0, "V"),
    ]


def _seg_57() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x110FA, "X"),
        (0x11100, "V"),
        (0x11135, "X"),
        (0x11136, "V"),
        (0x11148, "X"),
        (0x11150, "V"),
        (0x11177, "X"),
        (0x11180, "V"),
        (0x111E0, "X"),
        (0x111E1, "V"),
        (0x111F5, "X"),
        (0x11200, "V"),
        (0x11212, "X"),
        (0x11213, "V"),
        (0x11242, "X"),
        (0x11280, "V"),
        (0x11287, "X"),
        (0x11288, "V"),
        (0x11289, "X"),
        (0x1128A, "V"),
        (0x1128E, "X"),
        (0x1128F, "V"),
        (0x1129E, "X"),
        (0x1129F, "V"),
        (0x112AA, "X"),
        (0x112B0, "V"),
        (0x112EB, "X"),
        (0x112F0, "V"),
        (0x112FA, "X"),
        (0x11300, "V"),
        (0x11304, "X"),
        (0x11305, "V"),
        (0x1130D, "X"),
        (0x1130F, "V"),
        (0x11311, "X"),
        (0x11313, "V"),
        (0x11329, "X"),
        (0x1132A, "V"),
        (0x11331, "X"),
        (0x11332, "V"),
        (0x11334, "X"),
        (0x11335, "V"),
        (0x1133A, "X"),
        (0x1133B, "V"),
        (0x11345, "X"),
        (0x11347, "V"),
        (0x11349, "X"),
        (0x1134B, "V"),
        (0x1134E, "X"),
        (0x11350, "V"),
        (0x11351, "X"),
        (0x11357, "V"),
        (0x11358, "X"),
        (0x1135D, "V"),
        (0x11364, "X"),
        (0x11366, "V"),
        (0x1136D, "X"),
        (0x11370, "V"),
        (0x11375, "X"),
        (0x11400, "V"),
        (0x1145C, "X"),
        (0x1145D, "V"),
        (0x11462, "X"),
        (0x11480, "V"),
        (0x114C8, "X"),
        (0x114D0, "V"),
        (0x114DA, "X"),
        (0x11580, "V"),
        (0x115B6, "X"),
        (0x115B8, "V"),
        (0x115DE, "X"),
        (0x11600, "V"),
        (0x11645, "X"),
        (0x11650, "V"),
        (0x1165A, "X"),
        (0x11660, "V"),
        (0x1166D, "X"),
        (0x11680, "V"),
        (0x116BA, "X"),
        (0x116C0, "V"),
        (0x116CA, "X"),
        (0x11700, "V"),
        (0x1171B, "X"),
        (0x1171D, "V"),
        (0x1172C, "X"),
        (0x11730, "V"),
        (0x11747, "X"),
        (0x11800, "V"),
        (0x1183C, "X"),
        (0x118A0, "M", "𑣀"),
        (0x118A1, "M", "𑣁"),
        (0x118A2, "M", "𑣂"),
        (0x118A3, "M", "𑣃"),
        (0x118A4, "M", "𑣄"),
        (0x118A5, "M", "𑣅"),
        (0x118A6, "M", "𑣆"),
        (0x118A7, "M", "𑣇"),
        (0x118A8, "M", "𑣈"),
        (0x118A9, "M", "𑣉"),
        (0x118AA, "M", "𑣊"),
    ]


def _seg_58() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x118AB, "M", "𑣋"),
        (0x118AC, "M", "𑣌"),
        (0x118AD, "M", "𑣍"),
        (0x118AE, "M", "𑣎"),
        (0x118AF, "M", "𑣏"),
        (0x118B0, "M", "𑣐"),
        (0x118B1, "M", "𑣑"),
        (0x118B2, "M", "𑣒"),
        (0x118B3, "M", "𑣓"),
        (0x118B4, "M", "𑣔"),
        (0x118B5, "M", "𑣕"),
        (0x118B6, "M", "𑣖"),
        (0x118B7, "M", "𑣗"),
        (0x118B8, "M", "𑣘"),
        (0x118B9, "M", "𑣙"),
        (0x118BA, "M", "𑣚"),
        (0x118BB, "M", "𑣛"),
        (0x118BC, "M", "𑣜"),
        (0x118BD, "M", "𑣝"),
        (0x118BE, "M", "𑣞"),
        (0x118BF, "M", "𑣟"),
        (0x118C0, "V"),
        (0x118F3, "X"),
        (0x118FF, "V"),
        (0x11907, "X"),
        (0x11909, "V"),
        (0x1190A, "X"),
        (0x1190C, "V"),
        (0x11914, "X"),
        (0x11915, "V"),
        (0x11917, "X"),
        (0x11918, "V"),
        (0x11936, "X"),
        (0x11937, "V"),
        (0x11939, "X"),
        (0x1193B, "V"),
        (0x11947, "X"),
        (0x11950, "V"),
        (0x1195A, "X"),
        (0x119A0, "V"),
        (0x119A8, "X"),
        (0x119AA, "V"),
        (0x119D8, "X"),
        (0x119DA, "V"),
        (0x119E5, "X"),
        (0x11A00, "V"),
        (0x11A48, "X"),
        (0x11A50, "V"),
        (0x11AA3, "X"),
        (0x11AB0, "V"),
        (0x11AF9, "X"),
        (0x11B00, "V"),
        (0x11B0A, "X"),
        (0x11C00, "V"),
        (0x11C09, "X"),
        (0x11C0A, "V"),
        (0x11C37, "X"),
        (0x11C38, "V"),
        (0x11C46, "X"),
        (0x11C50, "V"),
        (0x11C6D, "X"),
        (0x11C70, "V"),
        (0x11C90, "X"),
        (0x11C92, "V"),
        (0x11CA8, "X"),
        (0x11CA9, "V"),
        (0x11CB7, "X"),
        (0x11D00, "V"),
        (0x11D07, "X"),
        (0x11D08, "V"),
        (0x11D0A, "X"),
        (0x11D0B, "V"),
        (0x11D37, "X"),
        (0x11D3A, "V"),
        (0x11D3B, "X"),
        (0x11D3C, "V"),
        (0x11D3E, "X"),
        (0x11D3F, "V"),
        (0x11D48, "X"),
        (0x11D50, "V"),
        (0x11D5A, "X"),
        (0x11D60, "V"),
        (0x11D66, "X"),
        (0x11D67, "V"),
        (0x11D69, "X"),
        (0x11D6A, "V"),
        (0x11D8F, "X"),
        (0x11D90, "V"),
        (0x11D92, "X"),
        (0x11D93, "V"),
        (0x11D99, "X"),
        (0x11DA0, "V"),
        (0x11DAA, "X"),
        (0x11EE0, "V"),
        (0x11EF9, "X"),
        (0x11F00, "V"),
        (0x11F11, "X"),
        (0x11F12, "V"),
        (0x11F3B, "X"),
        (0x11F3E, "V"),
    ]


def _seg_59() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x11F5A, "X"),
        (0x11FB0, "V"),
        (0x11FB1, "X"),
        (0x11FC0, "V"),
        (0x11FF2, "X"),
        (0x11FFF, "V"),
        (0x1239A, "X"),
        (0x12400, "V"),
        (0x1246F, "X"),
        (0x12470, "V"),
        (0x12475, "X"),
        (0x12480, "V"),
        (0x12544, "X"),
        (0x12F90, "V"),
        (0x12FF3, "X"),
        (0x13000, "V"),
        (0x13430, "X"),
        (0x13440, "V"),
        (0x13456, "X"),
        (0x14400, "V"),
        (0x14647, "X"),
        (0x16800, "V"),
        (0x16A39, "X"),
        (0x16A40, "V"),
        (0x16A5F, "X"),
        (0x16A60, "V"),
        (0x16A6A, "X"),
        (0x16A6E, "V"),
        (0x16ABF, "X"),
        (0x16AC0, "V"),
        (0x16ACA, "X"),
        (0x16AD0, "V"),
        (0x16AEE, "X"),
        (0x16AF0, "V"),
        (0x16AF6, "X"),
        (0x16B00, "V"),
        (0x16B46, "X"),
        (0x16B50, "V"),
        (0x16B5A, "X"),
        (0x16B5B, "V"),
        (0x16B62, "X"),
        (0x16B63, "V"),
        (0x16B78, "X"),
        (0x16B7D, "V"),
        (0x16B90, "X"),
        (0x16E40, "M", "𖹠"),
        (0x16E41, "M", "𖹡"),
        (0x16E42, "M", "𖹢"),
        (0x16E43, "M", "𖹣"),
        (0x16E44, "M", "𖹤"),
        (0x16E45, "M", "𖹥"),
        (0x16E46, "M", "𖹦"),
        (0x16E47, "M", "𖹧"),
        (0x16E48, "M", "𖹨"),
        (0x16E49, "M", "𖹩"),
        (0x16E4A, "M", "𖹪"),
        (0x16E4B, "M", "𖹫"),
        (0x16E4C, "M", "𖹬"),
        (0x16E4D, "M", "𖹭"),
        (0x16E4E, "M", "𖹮"),
        (0x16E4F, "M", "𖹯"),
        (0x16E50, "M", "𖹰"),
        (0x16E51, "M", "𖹱"),
        (0x16E52, "M", "𖹲"),
        (0x16E53, "M", "𖹳"),
        (0x16E54, "M", "𖹴"),
        (0x16E55, "M", "𖹵"),
        (0x16E56, "M", "𖹶"),
        (0x16E57, "M", "𖹷"),
        (0x16E58, "M", "𖹸"),
        (0x16E59, "M", "𖹹"),
        (0x16E5A, "M", "𖹺"),
        (0x16E5B, "M", "𖹻"),
        (0x16E5C, "M", "𖹼"),
        (0x16E5D, "M", "𖹽"),
        (0x16E5E, "M", "𖹾"),
        (0x16E5F, "M", "𖹿"),
        (0x16E60, "V"),
        (0x16E9B, "X"),
        (0x16F00, "V"),
        (0x16F4B, "X"),
        (0x16F4F, "V"),
        (0x16F88, "X"),
        (0x16F8F, "V"),
        (0x16FA0, "X"),
        (0x16FE0, "V"),
        (0x16FE5, "X"),
        (0x16FF0, "V"),
        (0x16FF2, "X"),
        (0x17000, "V"),
        (0x187F8, "X"),
        (0x18800, "V"),
        (0x18CD6, "X"),
        (0x18D00, "V"),
        (0x18D09, "X"),
        (0x1AFF0, "V"),
        (0x1AFF4, "X"),
        (0x1AFF5, "V"),
        (0x1AFFC, "X"),
        (0x1AFFD, "V"),
    ]


def _seg_60() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1AFFF, "X"),
        (0x1B000, "V"),
        (0x1B123, "X"),
        (0x1B132, "V"),
        (0x1B133, "X"),
        (0x1B150, "V"),
        (0x1B153, "X"),
        (0x1B155, "V"),
        (0x1B156, "X"),
        (0x1B164, "V"),
        (0x1B168, "X"),
        (0x1B170, "V"),
        (0x1B2FC, "X"),
        (0x1BC00, "V"),
        (0x1BC6B, "X"),
        (0x1BC70, "V"),
        (0x1BC7D, "X"),
        (0x1BC80, "V"),
        (0x1BC89, "X"),
        (0x1BC90, "V"),
        (0x1BC9A, "X"),
        (0x1BC9C, "V"),
        (0x1BCA0, "I"),
        (0x1BCA4, "X"),
        (0x1CF00, "V"),
        (0x1CF2E, "X"),
        (0x1CF30, "V"),
        (0x1CF47, "X"),
        (0x1CF50, "V"),
        (0x1CFC4, "X"),
        (0x1D000, "V"),
        (0x1D0F6, "X"),
        (0x1D100, "V"),
        (0x1D127, "X"),
        (0x1D129, "V"),
        (0x1D15E, "M", "𝅗𝅥"),
        (0x1D15F, "M", "𝅘𝅥"),
        (0x1D160, "M", "𝅘𝅥𝅮"),
        (0x1D161, "M", "𝅘𝅥𝅯"),
        (0x1D162, "M", "𝅘𝅥𝅰"),
        (0x1D163, "M", "𝅘𝅥𝅱"),
        (0x1D164, "M", "𝅘𝅥𝅲"),
        (0x1D165, "V"),
        (0x1D173, "X"),
        (0x1D17B, "V"),
        (0x1D1BB, "M", "𝆹𝅥"),
        (0x1D1BC, "M", "𝆺𝅥"),
        (0x1D1BD, "M", "𝆹𝅥𝅮"),
        (0x1D1BE, "M", "𝆺𝅥𝅮"),
        (0x1D1BF, "M", "𝆹𝅥𝅯"),
        (0x1D1C0, "M", "𝆺𝅥𝅯"),
        (0x1D1C1, "V"),
        (0x1D1EB, "X"),
        (0x1D200, "V"),
        (0x1D246, "X"),
        (0x1D2C0, "V"),
        (0x1D2D4, "X"),
        (0x1D2E0, "V"),
        (0x1D2F4, "X"),
        (0x1D300, "V"),
        (0x1D357, "X"),
        (0x1D360, "V"),
        (0x1D379, "X"),
        (0x1D400, "M", "a"),
        (0x1D401, "M", "b"),
        (0x1D402, "M", "c"),
        (0x1D403, "M", "d"),
        (0x1D404, "M", "e"),
        (0x1D405, "M", "f"),
        (0x1D406, "M", "g"),
        (0x1D407, "M", "h"),
        (0x1D408, "M", "i"),
        (0x1D409, "M", "j"),
        (0x1D40A, "M", "k"),
        (0x1D40B, "M", "l"),
        (0x1D40C, "M", "m"),
        (0x1D40D, "M", "n"),
        (0x1D40E, "M", "o"),
        (0x1D40F, "M", "p"),
        (0x1D410, "M", "q"),
        (0x1D411, "M", "r"),
        (0x1D412, "M", "s"),
        (0x1D413, "M", "t"),
        (0x1D414, "M", "u"),
        (0x1D415, "M", "v"),
        (0x1D416, "M", "w"),
        (0x1D417, "M", "x"),
        (0x1D418, "M", "y"),
        (0x1D419, "M", "z"),
        (0x1D41A, "M", "a"),
        (0x1D41B, "M", "b"),
        (0x1D41C, "M", "c"),
        (0x1D41D, "M", "d"),
        (0x1D41E, "M", "e"),
        (0x1D41F, "M", "f"),
        (0x1D420, "M", "g"),
        (0x1D421, "M", "h"),
        (0x1D422, "M", "i"),
        (0x1D423, "M", "j"),
        (0x1D424, "M", "k"),
    ]


def _seg_61() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1D425, "M", "l"),
        (0x1D426, "M", "m"),
        (0x1D427, "M", "n"),
        (0x1D428, "M", "o"),
        (0x1D429, "M", "p"),
        (0x1D42A, "M", "q"),
        (0x1D42B, "M", "r"),
        (0x1D42C, "M", "s"),
        (0x1D42D, "M", "t"),
        (0x1D42E, "M", "u"),
        (0x1D42F, "M", "v"),
        (0x1D430, "M", "w"),
        (0x1D431, "M", "x"),
        (0x1D432, "M", "y"),
        (0x1D433, "M", "z"),
        (0x1D434, "M", "a"),
        (0x1D435, "M", "b"),
        (0x1D436, "M", "c"),
        (0x1D437, "M", "d"),
        (0x1D438, "M", "e"),
        (0x1D439, "M", "f"),
        (0x1D43A, "M", "g"),
        (0x1D43B, "M", "h"),
        (0x1D43C, "M", "i"),
        (0x1D43D, "M", "j"),
        (0x1D43E, "M", "k"),
        (0x1D43F, "M", "l"),
        (0x1D440, "M", "m"),
        (0x1D441, "M", "n"),
        (0x1D442, "M", "o"),
        (0x1D443, "M", "p"),
        (0x1D444, "M", "q"),
        (0x1D445, "M", "r"),
        (0x1D446, "M", "s"),
        (0x1D447, "M", "t"),
        (0x1D448, "M", "u"),
        (0x1D449, "M", "v"),
        (0x1D44A, "M", "w"),
        (0x1D44B, "M", "x"),
        (0x1D44C, "M", "y"),
        (0x1D44D, "M", "z"),
        (0x1D44E, "M", "a"),
        (0x1D44F, "M", "b"),
        (0x1D450, "M", "c"),
        (0x1D451, "M", "d"),
        (0x1D452, "M", "e"),
        (0x1D453, "M", "f"),
        (0x1D454, "M", "g"),
        (0x1D455, "X"),
        (0x1D456, "M", "i"),
        (0x1D457, "M", "j"),
        (0x1D458, "M", "k"),
        (0x1D459, "M", "l"),
        (0x1D45A, "M", "m"),
        (0x1D45B, "M", "n"),
        (0x1D45C, "M", "o"),
        (0x1D45D, "M", "p"),
        (0x1D45E, "M", "q"),
        (0x1D45F, "M", "r"),
        (0x1D460, "M", "s"),
        (0x1D461, "M", "t"),
        (0x1D462, "M", "u"),
        (0x1D463, "M", "v"),
        (0x1D464, "M", "w"),
        (0x1D465, "M", "x"),
        (0x1D466, "M", "y"),
        (0x1D467, "M", "z"),
        (0x1D468, "M", "a"),
        (0x1D469, "M", "b"),
        (0x1D46A, "M", "c"),
        (0x1D46B, "M", "d"),
        (0x1D46C, "M", "e"),
        (0x1D46D, "M", "f"),
        (0x1D46E, "M", "g"),
        (0x1D46F, "M", "h"),
        (0x1D470, "M", "i"),
        (0x1D471, "M", "j"),
        (0x1D472, "M", "k"),
        (0x1D473, "M", "l"),
        (0x1D474, "M", "m"),
        (0x1D475, "M", "n"),
        (0x1D476, "M", "o"),
        (0x1D477, "M", "p"),
        (0x1D478, "M", "q"),
        (0x1D479, "M", "r"),
        (0x1D47A, "M", "s"),
        (0x1D47B, "M", "t"),
        (0x1D47C, "M", "u"),
        (0x1D47D, "M", "v"),
        (0x1D47E, "M", "w"),
        (0x1D47F, "M", "x"),
        (0x1D480, "M", "y"),
        (0x1D481, "M", "z"),
        (0x1D482, "M", "a"),
        (0x1D483, "M", "b"),
        (0x1D484, "M", "c"),
        (0x1D485, "M", "d"),
        (0x1D486, "M", "e"),
        (0x1D487, "M", "f"),
        (0x1D488, "M", "g"),
    ]


def _seg_62() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1D489, "M", "h"),
        (0x1D48A, "M", "i"),
        (0x1D48B, "M", "j"),
        (0x1D48C, "M", "k"),
        (0x1D48D, "M", "l"),
        (0x1D48E, "M", "m"),
        (0x1D48F, "M", "n"),
        (0x1D490, "M", "o"),
        (0x1D491, "M", "p"),
        (0x1D492, "M", "q"),
        (0x1D493, "M", "r"),
        (0x1D494, "M", "s"),
        (0x1D495, "M", "t"),
        (0x1D496, "M", "u"),
        (0x1D497, "M", "v"),
        (0x1D498, "M", "w"),
        (0x1D499, "M", "x"),
        (0x1D49A, "M", "y"),
        (0x1D49B, "M", "z"),
        (0x1D49C, "M", "a"),
        (0x1D49D, "X"),
        (0x1D49E, "M", "c"),
        (0x1D49F, "M", "d"),
        (0x1D4A0, "X"),
        (0x1D4A2, "M", "g"),
        (0x1D4A3, "X"),
        (0x1D4A5, "M", "j"),
        (0x1D4A6, "M", "k"),
        (0x1D4A7, "X"),
        (0x1D4A9, "M", "n"),
        (0x1D4AA, "M", "o"),
        (0x1D4AB, "M", "p"),
        (0x1D4AC, "M", "q"),
        (0x1D4AD, "X"),
        (0x1D4AE, "M", "s"),
        (0x1D4AF, "M", "t"),
        (0x1D4B0, "M", "u"),
        (0x1D4B1, "M", "v"),
        (0x1D4B2, "M", "w"),
        (0x1D4B3, "M", "x"),
        (0x1D4B4, "M", "y"),
        (0x1D4B5, "M", "z"),
        (0x1D4B6, "M", "a"),
        (0x1D4B7, "M", "b"),
        (0x1D4B8, "M", "c"),
        (0x1D4B9, "M", "d"),
        (0x1D4BA, "X"),
        (0x1D4BB, "M", "f"),
        (0x1D4BC, "X"),
        (0x1D4BD, "M", "h"),
        (0x1D4BE, "M", "i"),
        (0x1D4BF, "M", "j"),
        (0x1D4C0, "M", "k"),
        (0x1D4C1, "M", "l"),
        (0x1D4C2, "M", "m"),
        (0x1D4C3, "M", "n"),
        (0x1D4C4, "X"),
        (0x1D4C5, "M", "p"),
        (0x1D4C6, "M", "q"),
        (0x1D4C7, "M", "r"),
        (0x1D4C8, "M", "s"),
        (0x1D4C9, "M", "t"),
        (0x1D4CA, "M", "u"),
        (0x1D4CB, "M", "v"),
        (0x1D4CC, "M", "w"),
        (0x1D4CD, "M", "x"),
        (0x1D4CE, "M", "y"),
        (0x1D4CF, "M", "z"),
        (0x1D4D0, "M", "a"),
        (0x1D4D1, "M", "b"),
        (0x1D4D2, "M", "c"),
        (0x1D4D3, "M", "d"),
        (0x1D4D4, "M", "e"),
        (0x1D4D5, "M", "f"),
        (0x1D4D6, "M", "g"),
        (0x1D4D7, "M", "h"),
        (0x1D4D8, "M", "i"),
        (0x1D4D9, "M", "j"),
        (0x1D4DA, "M", "k"),
        (0x1D4DB, "M", "l"),
        (0x1D4DC, "M", "m"),
        (0x1D4DD, "M", "n"),
        (0x1D4DE, "M", "o"),
        (0x1D4DF, "M", "p"),
        (0x1D4E0, "M", "q"),
        (0x1D4E1, "M", "r"),
        (0x1D4E2, "M", "s"),
        (0x1D4E3, "M", "t"),
        (0x1D4E4, "M", "u"),
        (0x1D4E5, "M", "v"),
        (0x1D4E6, "M", "w"),
        (0x1D4E7, "M", "x"),
        (0x1D4E8, "M", "y"),
        (0x1D4E9, "M", "z"),
        (0x1D4EA, "M", "a"),
        (0x1D4EB, "M", "b"),
        (0x1D4EC, "M", "c"),
        (0x1D4ED, "M", "d"),
        (0x1D4EE, "M", "e"),
        (0x1D4EF, "M", "f"),
    ]


def _seg_63() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1D4F0, "M", "g"),
        (0x1D4F1, "M", "h"),
        (0x1D4F2, "M", "i"),
        (0x1D4F3, "M", "j"),
        (0x1D4F4, "M", "k"),
        (0x1D4F5, "M", "l"),
        (0x1D4F6, "M", "m"),
        (0x1D4F7, "M", "n"),
        (0x1D4F8, "M", "o"),
        (0x1D4F9, "M", "p"),
        (0x1D4FA, "M", "q"),
        (0x1D4FB, "M", "r"),
        (0x1D4FC, "M", "s"),
        (0x1D4FD, "M", "t"),
        (0x1D4FE, "M", "u"),
        (0x1D4FF, "M", "v"),
        (0x1D500, "M", "w"),
        (0x1D501, "M", "x"),
        (0x1D502, "M", "y"),
        (0x1D503, "M", "z"),
        (0x1D504, "M", "a"),
        (0x1D505, "M", "b"),
        (0x1D506, "X"),
        (0x1D507, "M", "d"),
        (0x1D508, "M", "e"),
        (0x1D509, "M", "f"),
        (0x1D50A, "M", "g"),
        (0x1D50B, "X"),
        (0x1D50D, "M", "j"),
        (0x1D50E, "M", "k"),
        (0x1D50F, "M", "l"),
        (0x1D510, "M", "m"),
        (0x1D511, "M", "n"),
        (0x1D512, "M", "o"),
        (0x1D513, "M", "p"),
        (0x1D514, "M", "q"),
        (0x1D515, "X"),
        (0x1D516, "M", "s"),
        (0x1D517, "M", "t"),
        (0x1D518, "M", "u"),
        (0x1D519, "M", "v"),
        (0x1D51A, "M", "w"),
        (0x1D51B, "M", "x"),
        (0x1D51C, "M", "y"),
        (0x1D51D, "X"),
        (0x1D51E, "M", "a"),
        (0x1D51F, "M", "b"),
        (0x1D520, "M", "c"),
        (0x1D521, "M", "d"),
        (0x1D522, "M", "e"),
        (0x1D523, "M", "f"),
        (0x1D524, "M", "g"),
        (0x1D525, "M", "h"),
        (0x1D526, "M", "i"),
        (0x1D527, "M", "j"),
        (0x1D528, "M", "k"),
        (0x1D529, "M", "l"),
        (0x1D52A, "M", "m"),
        (0x1D52B, "M", "n"),
        (0x1D52C, "M", "o"),
        (0x1D52D, "M", "p"),
        (0x1D52E, "M", "q"),
        (0x1D52F, "M", "r"),
        (0x1D530, "M", "s"),
        (0x1D531, "M", "t"),
        (0x1D532, "M", "u"),
        (0x1D533, "M", "v"),
        (0x1D534, "M", "w"),
        (0x1D535, "M", "x"),
        (0x1D536, "M", "y"),
        (0x1D537, "M", "z"),
        (0x1D538, "M", "a"),
        (0x1D539, "M", "b"),
        (0x1D53A, "X"),
        (0x1D53B, "M", "d"),
        (0x1D53C, "M", "e"),
        (0x1D53D, "M", "f"),
        (0x1D53E, "M", "g"),
        (0x1D53F, "X"),
        (0x1D540, "M", "i"),
        (0x1D541, "M", "j"),
        (0x1D542, "M", "k"),
        (0x1D543, "M", "l"),
        (0x1D544, "M", "m"),
        (0x1D545, "X"),
        (0x1D546, "M", "o"),
        (0x1D547, "X"),
        (0x1D54A, "M", "s"),
        (0x1D54B, "M", "t"),
        (0x1D54C, "M", "u"),
        (0x1D54D, "M", "v"),
        (0x1D54E, "M", "w"),
        (0x1D54F, "M", "x"),
        (0x1D550, "M", "y"),
        (0x1D551, "X"),
        (0x1D552, "M", "a"),
        (0x1D553, "M", "b"),
        (0x1D554, "M", "c"),
        (0x1D555, "M", "d"),
        (0x1D556, "M", "e"),
    ]


def _seg_64() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1D557, "M", "f"),
        (0x1D558, "M", "g"),
        (0x1D559, "M", "h"),
        (0x1D55A, "M", "i"),
        (0x1D55B, "M", "j"),
        (0x1D55C, "M", "k"),
        (0x1D55D, "M", "l"),
        (0x1D55E, "M", "m"),
        (0x1D55F, "M", "n"),
        (0x1D560, "M", "o"),
        (0x1D561, "M", "p"),
        (0x1D562, "M", "q"),
        (0x1D563, "M", "r"),
        (0x1D564, "M", "s"),
        (0x1D565, "M", "t"),
        (0x1D566, "M", "u"),
        (0x1D567, "M", "v"),
        (0x1D568, "M", "w"),
        (0x1D569, "M", "x"),
        (0x1D56A, "M", "y"),
        (0x1D56B, "M", "z"),
        (0x1D56C, "M", "a"),
        (0x1D56D, "M", "b"),
        (0x1D56E, "M", "c"),
        (0x1D56F, "M", "d"),
        (0x1D570, "M", "e"),
        (0x1D571, "M", "f"),
        (0x1D572, "M", "g"),
        (0x1D573, "M", "h"),
        (0x1D574, "M", "i"),
        (0x1D575, "M", "j"),
        (0x1D576, "M", "k"),
        (0x1D577, "M", "l"),
        (0x1D578, "M", "m"),
        (0x1D579, "M", "n"),
        (0x1D57A, "M", "o"),
        (0x1D57B, "M", "p"),
        (0x1D57C, "M", "q"),
        (0x1D57D, "M", "r"),
        (0x1D57E, "M", "s"),
        (0x1D57F, "M", "t"),
        (0x1D580, "M", "u"),
        (0x1D581, "M", "v"),
        (0x1D582, "M", "w"),
        (0x1D583, "M", "x"),
        (0x1D584, "M", "y"),
        (0x1D585, "M", "z"),
        (0x1D586, "M", "a"),
        (0x1D587, "M", "b"),
        (0x1D588, "M", "c"),
        (0x1D589, "M", "d"),
        (0x1D58A, "M", "e"),
        (0x1D58B, "M", "f"),
        (0x1D58C, "M", "g"),
        (0x1D58D, "M", "h"),
        (0x1D58E, "M", "i"),
        (0x1D58F, "M", "j"),
        (0x1D590, "M", "k"),
        (0x1D591, "M", "l"),
        (0x1D592, "M", "m"),
        (0x1D593, "M", "n"),
        (0x1D594, "M", "o"),
        (0x1D595, "M", "p"),
        (0x1D596, "M", "q"),
        (0x1D597, "M", "r"),
        (0x1D598, "M", "s"),
        (0x1D599, "M", "t"),
        (0x1D59A, "M", "u"),
        (0x1D59B, "M", "v"),
        (0x1D59C, "M", "w"),
        (0x1D59D, "M", "x"),
        (0x1D59E, "M", "y"),
        (0x1D59F, "M", "z"),
        (0x1D5A0, "M", "a"),
        (0x1D5A1, "M", "b"),
        (0x1D5A2, "M", "c"),
        (0x1D5A3, "M", "d"),
        (0x1D5A4, "M", "e"),
        (0x1D5A5, "M", "f"),
        (0x1D5A6, "M", "g"),
        (0x1D5A7, "M", "h"),
        (0x1D5A8, "M", "i"),
        (0x1D5A9, "M", "j"),
        (0x1D5AA, "M", "k"),
        (0x1D5AB, "M", "l"),
        (0x1D5AC, "M", "m"),
        (0x1D5AD, "M", "n"),
        (0x1D5AE, "M", "o"),
        (0x1D5AF, "M", "p"),
        (0x1D5B0, "M", "q"),
        (0x1D5B1, "M", "r"),
        (0x1D5B2, "M", "s"),
        (0x1D5B3, "M", "t"),
        (0x1D5B4, "M", "u"),
        (0x1D5B5, "M", "v"),
        (0x1D5B6, "M", "w"),
        (0x1D5B7, "M", "x"),
        (0x1D5B8, "M", "y"),
        (0x1D5B9, "M", "z"),
        (0x1D5BA, "M", "a"),
    ]


def _seg_65() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1D5BB, "M", "b"),
        (0x1D5BC, "M", "c"),
        (0x1D5BD, "M", "d"),
        (0x1D5BE, "M", "e"),
        (0x1D5BF, "M", "f"),
        (0x1D5C0, "M", "g"),
        (0x1D5C1, "M", "h"),
        (0x1D5C2, "M", "i"),
        (0x1D5C3, "M", "j"),
        (0x1D5C4, "M", "k"),
        (0x1D5C5, "M", "l"),
        (0x1D5C6, "M", "m"),
        (0x1D5C7, "M", "n"),
        (0x1D5C8, "M", "o"),
        (0x1D5C9, "M", "p"),
        (0x1D5CA, "M", "q"),
        (0x1D5CB, "M", "r"),
        (0x1D5CC, "M", "s"),
        (0x1D5CD, "M", "t"),
        (0x1D5CE, "M", "u"),
        (0x1D5CF, "M", "v"),
        (0x1D5D0, "M", "w"),
        (0x1D5D1, "M", "x"),
        (0x1D5D2, "M", "y"),
        (0x1D5D3, "M", "z"),
        (0x1D5D4, "M", "a"),
        (0x1D5D5, "M", "b"),
        (0x1D5D6, "M", "c"),
        (0x1D5D7, "M", "d"),
        (0x1D5D8, "M", "e"),
        (0x1D5D9, "M", "f"),
        (0x1D5DA, "M", "g"),
        (0x1D5DB, "M", "h"),
        (0x1D5DC, "M", "i"),
        (0x1D5DD, "M", "j"),
        (0x1D5DE, "M", "k"),
        (0x1D5DF, "M", "l"),
        (0x1D5E0, "M", "m"),
        (0x1D5E1, "M", "n"),
        (0x1D5E2, "M", "o"),
        (0x1D5E3, "M", "p"),
        (0x1D5E4, "M", "q"),
        (0x1D5E5, "M", "r"),
        (0x1D5E6, "M", "s"),
        (0x1D5E7, "M", "t"),
        (0x1D5E8, "M", "u"),
        (0x1D5E9, "M", "v"),
        (0x1D5EA, "M", "w"),
        (0x1D5EB, "M", "x"),
        (0x1D5EC, "M", "y"),
        (0x1D5ED, "M", "z"),
        (0x1D5EE, "M", "a"),
        (0x1D5EF, "M", "b"),
        (0x1D5F0, "M", "c"),
        (0x1D5F1, "M", "d"),
        (0x1D5F2, "M", "e"),
        (0x1D5F3, "M", "f"),
        (0x1D5F4, "M", "g"),
        (0x1D5F5, "M", "h"),
        (0x1D5F6, "M", "i"),
        (0x1D5F7, "M", "j"),
        (0x1D5F8, "M", "k"),
        (0x1D5F9, "M", "l"),
        (0x1D5FA, "M", "m"),
        (0x1D5FB, "M", "n"),
        (0x1D5FC, "M", "o"),
        (0x1D5FD, "M", "p"),
        (0x1D5FE, "M", "q"),
        (0x1D5FF, "M", "r"),
        (0x1D600, "M", "s"),
        (0x1D601, "M", "t"),
        (0x1D602, "M", "u"),
        (0x1D603, "M", "v"),
        (0x1D604, "M", "w"),
        (0x1D605, "M", "x"),
        (0x1D606, "M", "y"),
        (0x1D607, "M", "z"),
        (0x1D608, "M", "a"),
        (0x1D609, "M", "b"),
        (0x1D60A, "M", "c"),
        (0x1D60B, "M", "d"),
        (0x1D60C, "M", "e"),
        (0x1D60D, "M", "f"),
        (0x1D60E, "M", "g"),
        (0x1D60F, "M", "h"),
        (0x1D610, "M", "i"),
        (0x1D611, "M", "j"),
        (0x1D612, "M", "k"),
        (0x1D613, "M", "l"),
        (0x1D614, "M", "m"),
        (0x1D615, "M", "n"),
        (0x1D616, "M", "o"),
        (0x1D617, "M", "p"),
        (0x1D618, "M", "q"),
        (0x1D619, "M", "r"),
        (0x1D61A, "M", "s"),
        (0x1D61B, "M", "t"),
        (0x1D61C, "M", "u"),
        (0x1D61D, "M", "v"),
        (0x1D61E, "M", "w"),
    ]


def _seg_66() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1D61F, "M", "x"),
        (0x1D620, "M", "y"),
        (0x1D621, "M", "z"),
        (0x1D622, "M", "a"),
        (0x1D623, "M", "b"),
        (0x1D624, "M", "c"),
        (0x1D625, "M", "d"),
        (0x1D626, "M", "e"),
        (0x1D627, "M", "f"),
        (0x1D628, "M", "g"),
        (0x1D629, "M", "h"),
        (0x1D62A, "M", "i"),
        (0x1D62B, "M", "j"),
        (0x1D62C, "M", "k"),
        (0x1D62D, "M", "l"),
        (0x1D62E, "M", "m"),
        (0x1D62F, "M", "n"),
        (0x1D630, "M", "o"),
        (0x1D631, "M", "p"),
        (0x1D632, "M", "q"),
        (0x1D633, "M", "r"),
        (0x1D634, "M", "s"),
        (0x1D635, "M", "t"),
        (0x1D636, "M", "u"),
        (0x1D637, "M", "v"),
        (0x1D638, "M", "w"),
        (0x1D639, "M", "x"),
        (0x1D63A, "M", "y"),
        (0x1D63B, "M", "z"),
        (0x1D63C, "M", "a"),
        (0x1D63D, "M", "b"),
        (0x1D63E, "M", "c"),
        (0x1D63F, "M", "d"),
        (0x1D640, "M", "e"),
        (0x1D641, "M", "f"),
        (0x1D642, "M", "g"),
        (0x1D643, "M", "h"),
        (0x1D644, "M", "i"),
        (0x1D645, "M", "j"),
        (0x1D646, "M", "k"),
        (0x1D647, "M", "l"),
        (0x1D648, "M", "m"),
        (0x1D649, "M", "n"),
        (0x1D64A, "M", "o"),
        (0x1D64B, "M", "p"),
        (0x1D64C, "M", "q"),
        (0x1D64D, "M", "r"),
        (0x1D64E, "M", "s"),
        (0x1D64F, "M", "t"),
        (0x1D650, "M", "u"),
        (0x1D651, "M", "v"),
        (0x1D652, "M", "w"),
        (0x1D653, "M", "x"),
        (0x1D654, "M", "y"),
        (0x1D655, "M", "z"),
        (0x1D656, "M", "a"),
        (0x1D657, "M", "b"),
        (0x1D658, "M", "c"),
        (0x1D659, "M", "d"),
        (0x1D65A, "M", "e"),
        (0x1D65B, "M", "f"),
        (0x1D65C, "M", "g"),
        (0x1D65D, "M", "h"),
        (0x1D65E, "M", "i"),
        (0x1D65F, "M", "j"),
        (0x1D660, "M", "k"),
        (0x1D661, "M", "l"),
        (0x1D662, "M", "m"),
        (0x1D663, "M", "n"),
        (0x1D664, "M", "o"),
        (0x1D665, "M", "p"),
        (0x1D666, "M", "q"),
        (0x1D667, "M", "r"),
        (0x1D668, "M", "s"),
        (0x1D669, "M", "t"),
        (0x1D66A, "M", "u"),
        (0x1D66B, "M", "v"),
        (0x1D66C, "M", "w"),
        (0x1D66D, "M", "x"),
        (0x1D66E, "M", "y"),
        (0x1D66F, "M", "z"),
        (0x1D670, "M", "a"),
        (0x1D671, "M", "b"),
        (0x1D672, "M", "c"),
        (0x1D673, "M", "d"),
        (0x1D674, "M", "e"),
        (0x1D675, "M", "f"),
        (0x1D676, "M", "g"),
        (0x1D677, "M", "h"),
        (0x1D678, "M", "i"),
        (0x1D679, "M", "j"),
        (0x1D67A, "M", "k"),
        (0x1D67B, "M", "l"),
        (0x1D67C, "M", "m"),
        (0x1D67D, "M", "n"),
        (0x1D67E, "M", "o"),
        (0x1D67F, "M", "p"),
        (0x1D680, "M", "q"),
        (0x1D681, "M", "r"),
        (0x1D682, "M", "s"),
    ]


def _seg_67() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1D683, "M", "t"),
        (0x1D684, "M", "u"),
        (0x1D685, "M", "v"),
        (0x1D686, "M", "w"),
        (0x1D687, "M", "x"),
        (0x1D688, "M", "y"),
        (0x1D689, "M", "z"),
        (0x1D68A, "M", "a"),
        (0x1D68B, "M", "b"),
        (0x1D68C, "M", "c"),
        (0x1D68D, "M", "d"),
        (0x1D68E, "M", "e"),
        (0x1D68F, "M", "f"),
        (0x1D690, "M", "g"),
        (0x1D691, "M", "h"),
        (0x1D692, "M", "i"),
        (0x1D693, "M", "j"),
        (0x1D694, "M", "k"),
        (0x1D695, "M", "l"),
        (0x1D696, "M", "m"),
        (0x1D697, "M", "n"),
        (0x1D698, "M", "o"),
        (0x1D699, "M", "p"),
        (0x1D69A, "M", "q"),
        (0x1D69B, "M", "r"),
        (0x1D69C, "M", "s"),
        (0x1D69D, "M", "t"),
        (0x1D69E, "M", "u"),
        (0x1D69F, "M", "v"),
        (0x1D6A0, "M", "w"),
        (0x1D6A1, "M", "x"),
        (0x1D6A2, "M", "y"),
        (0x1D6A3, "M", "z"),
        (0x1D6A4, "M", "ı"),
        (0x1D6A5, "M", "ȷ"),
        (0x1D6A6, "X"),
        (0x1D6A8, "M", "α"),
        (0x1D6A9, "M", "β"),
        (0x1D6AA, "M", "γ"),
        (0x1D6AB, "M", "δ"),
        (0x1D6AC, "M", "ε"),
        (0x1D6AD, "M", "ζ"),
        (0x1D6AE, "M", "η"),
        (0x1D6AF, "M", "θ"),
        (0x1D6B0, "M", "ι"),
        (0x1D6B1, "M", "κ"),
        (0x1D6B2, "M", "λ"),
        (0x1D6B3, "M", "μ"),
        (0x1D6B4, "M", "ν"),
        (0x1D6B5, "M", "ξ"),
        (0x1D6B6, "M", "ο"),
        (0x1D6B7, "M", "π"),
        (0x1D6B8, "M", "ρ"),
        (0x1D6B9, "M", "θ"),
        (0x1D6BA, "M", "σ"),
        (0x1D6BB, "M", "τ"),
        (0x1D6BC, "M", "υ"),
        (0x1D6BD, "M", "φ"),
        (0x1D6BE, "M", "χ"),
        (0x1D6BF, "M", "ψ"),
        (0x1D6C0, "M", "ω"),
        (0x1D6C1, "M", "∇"),
        (0x1D6C2, "M", "α"),
        (0x1D6C3, "M", "β"),
        (0x1D6C4, "M", "γ"),
        (0x1D6C5, "M", "δ"),
        (0x1D6C6, "M", "ε"),
        (0x1D6C7, "M", "ζ"),
        (0x1D6C8, "M", "η"),
        (0x1D6C9, "M", "θ"),
        (0x1D6CA, "M", "ι"),
        (0x1D6CB, "M", "κ"),
        (0x1D6CC, "M", "λ"),
        (0x1D6CD, "M", "μ"),
        (0x1D6CE, "M", "ν"),
        (0x1D6CF, "M", "ξ"),
        (0x1D6D0, "M", "ο"),
        (0x1D6D1, "M", "π"),
        (0x1D6D2, "M", "ρ"),
        (0x1D6D3, "M", "σ"),
        (0x1D6D5, "M", "τ"),
        (0x1D6D6, "M", "υ"),
        (0x1D6D7, "M", "φ"),
        (0x1D6D8, "M", "χ"),
        (0x1D6D9, "M", "ψ"),
        (0x1D6DA, "M", "ω"),
        (0x1D6DB, "M", "∂"),
        (0x1D6DC, "M", "ε"),
        (0x1D6DD, "M", "θ"),
        (0x1D6DE, "M", "κ"),
        (0x1D6DF, "M", "φ"),
        (0x1D6E0, "M", "ρ"),
        (0x1D6E1, "M", "π"),
        (0x1D6E2, "M", "α"),
        (0x1D6E3, "M", "β"),
        (0x1D6E4, "M", "γ"),
        (0x1D6E5, "M", "δ"),
        (0x1D6E6, "M", "ε"),
        (0x1D6E7, "M", "ζ"),
        (0x1D6E8, "M", "η"),
    ]


def _seg_68() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1D6E9, "M", "θ"),
        (0x1D6EA, "M", "ι"),
        (0x1D6EB, "M", "κ"),
        (0x1D6EC, "M", "λ"),
        (0x1D6ED, "M", "μ"),
        (0x1D6EE, "M", "ν"),
        (0x1D6EF, "M", "ξ"),
        (0x1D6F0, "M", "ο"),
        (0x1D6F1, "M", "π"),
        (0x1D6F2, "M", "ρ"),
        (0x1D6F3, "M", "θ"),
        (0x1D6F4, "M", "σ"),
        (0x1D6F5, "M", "τ"),
        (0x1D6F6, "M", "υ"),
        (0x1D6F7, "M", "φ"),
        (0x1D6F8, "M", "χ"),
        (0x1D6F9, "M", "ψ"),
        (0x1D6FA, "M", "ω"),
        (0x1D6FB, "M", "∇"),
        (0x1D6FC, "M", "α"),
        (0x1D6FD, "M", "β"),
        (0x1D6FE, "M", "γ"),
        (0x1D6FF, "M", "δ"),
        (0x1D700, "M", "ε"),
        (0x1D701, "M", "ζ"),
        (0x1D702, "M", "η"),
        (0x1D703, "M", "θ"),
        (0x1D704, "M", "ι"),
        (0x1D705, "M", "κ"),
        (0x1D706, "M", "λ"),
        (0x1D707, "M", "μ"),
        (0x1D708, "M", "ν"),
        (0x1D709, "M", "ξ"),
        (0x1D70A, "M", "ο"),
        (0x1D70B, "M", "π"),
        (0x1D70C, "M", "ρ"),
        (0x1D70D, "M", "σ"),
        (0x1D70F, "M", "τ"),
        (0x1D710, "M", "υ"),
        (0x1D711, "M", "φ"),
        (0x1D712, "M", "χ"),
        (0x1D713, "M", "ψ"),
        (0x1D714, "M", "ω"),
        (0x1D715, "M", "∂"),
        (0x1D716, "M", "ε"),
        (0x1D717, "M", "θ"),
        (0x1D718, "M", "κ"),
        (0x1D719, "M", "φ"),
        (0x1D71A, "M", "ρ"),
        (0x1D71B, "M", "π"),
        (0x1D71C, "M", "α"),
        (0x1D71D, "M", "β"),
        (0x1D71E, "M", "γ"),
        (0x1D71F, "M", "δ"),
        (0x1D720, "M", "ε"),
        (0x1D721, "M", "ζ"),
        (0x1D722, "M", "η"),
        (0x1D723, "M", "θ"),
        (0x1D724, "M", "ι"),
        (0x1D725, "M", "κ"),
        (0x1D726, "M", "λ"),
        (0x1D727, "M", "μ"),
        (0x1D728, "M", "ν"),
        (0x1D729, "M", "ξ"),
        (0x1D72A, "M", "ο"),
        (0x1D72B, "M", "π"),
        (0x1D72C, "M", "ρ"),
        (0x1D72D, "M", "θ"),
        (0x1D72E, "M", "σ"),
        (0x1D72F, "M", "τ"),
        (0x1D730, "M", "υ"),
        (0x1D731, "M", "φ"),
        (0x1D732, "M", "χ"),
        (0x1D733, "M", "ψ"),
        (0x1D734, "M", "ω"),
        (0x1D735, "M", "∇"),
        (0x1D736, "M", "α"),
        (0x1D737, "M", "β"),
        (0x1D738, "M", "γ"),
        (0x1D739, "M", "δ"),
        (0x1D73A, "M", "ε"),
        (0x1D73B, "M", "ζ"),
        (0x1D73C, "M", "η"),
        (0x1D73D, "M", "θ"),
        (0x1D73E, "M", "ι"),
        (0x1D73F, "M", "κ"),
        (0x1D740, "M", "λ"),
        (0x1D741, "M", "μ"),
        (0x1D742, "M", "ν"),
        (0x1D743, "M", "ξ"),
        (0x1D744, "M", "ο"),
        (0x1D745, "M", "π"),
        (0x1D746, "M", "ρ"),
        (0x1D747, "M", "σ"),
        (0x1D749, "M", "τ"),
        (0x1D74A, "M", "υ"),
        (0x1D74B, "M", "φ"),
        (0x1D74C, "M", "χ"),
        (0x1D74D, "M", "ψ"),
        (0x1D74E, "M", "ω"),
    ]


def _seg_69() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1D74F, "M", "∂"),
        (0x1D750, "M", "ε"),
        (0x1D751, "M", "θ"),
        (0x1D752, "M", "κ"),
        (0x1D753, "M", "φ"),
        (0x1D754, "M", "ρ"),
        (0x1D755, "M", "π"),
        (0x1D756, "M", "α"),
        (0x1D757, "M", "β"),
        (0x1D758, "M", "γ"),
        (0x1D759, "M", "δ"),
        (0x1D75A, "M", "ε"),
        (0x1D75B, "M", "ζ"),
        (0x1D75C, "M", "η"),
        (0x1D75D, "M", "θ"),
        (0x1D75E, "M", "ι"),
        (0x1D75F, "M", "κ"),
        (0x1D760, "M", "λ"),
        (0x1D761, "M", "μ"),
        (0x1D762, "M", "ν"),
        (0x1D763, "M", "ξ"),
        (0x1D764, "M", "ο"),
        (0x1D765, "M", "π"),
        (0x1D766, "M", "ρ"),
        (0x1D767, "M", "θ"),
        (0x1D768, "M", "σ"),
        (0x1D769, "M", "τ"),
        (0x1D76A, "M", "υ"),
        (0x1D76B, "M", "φ"),
        (0x1D76C, "M", "χ"),
        (0x1D76D, "M", "ψ"),
        (0x1D76E, "M", "ω"),
        (0x1D76F, "M", "∇"),
        (0x1D770, "M", "α"),
        (0x1D771, "M", "β"),
        (0x1D772, "M", "γ"),
        (0x1D773, "M", "δ"),
        (0x1D774, "M", "ε"),
        (0x1D775, "M", "ζ"),
        (0x1D776, "M", "η"),
        (0x1D777, "M", "θ"),
        (0x1D778, "M", "ι"),
        (0x1D779, "M", "κ"),
        (0x1D77A, "M", "λ"),
        (0x1D77B, "M", "μ"),
        (0x1D77C, "M", "ν"),
        (0x1D77D, "M", "ξ"),
        (0x1D77E, "M", "ο"),
        (0x1D77F, "M", "π"),
        (0x1D780, "M", "ρ"),
        (0x1D781, "M", "σ"),
        (0x1D783, "M", "τ"),
        (0x1D784, "M", "υ"),
        (0x1D785, "M", "φ"),
        (0x1D786, "M", "χ"),
        (0x1D787, "M", "ψ"),
        (0x1D788, "M", "ω"),
        (0x1D789, "M", "∂"),
        (0x1D78A, "M", "ε"),
        (0x1D78B, "M", "θ"),
        (0x1D78C, "M", "κ"),
        (0x1D78D, "M", "φ"),
        (0x1D78E, "M", "ρ"),
        (0x1D78F, "M", "π"),
        (0x1D790, "M", "α"),
        (0x1D791, "M", "β"),
        (0x1D792, "M", "γ"),
        (0x1D793, "M", "δ"),
        (0x1D794, "M", "ε"),
        (0x1D795, "M", "ζ"),
        (0x1D796, "M", "η"),
        (0x1D797, "M", "θ"),
        (0x1D798, "M", "ι"),
        (0x1D799, "M", "κ"),
        (0x1D79A, "M", "λ"),
        (0x1D79B, "M", "μ"),
        (0x1D79C, "M", "ν"),
        (0x1D79D, "M", "ξ"),
        (0x1D79E, "M", "ο"),
        (0x1D79F, "M", "π"),
        (0x1D7A0, "M", "ρ"),
        (0x1D7A1, "M", "θ"),
        (0x1D7A2, "M", "σ"),
        (0x1D7A3, "M", "τ"),
        (0x1D7A4, "M", "υ"),
        (0x1D7A5, "M", "φ"),
        (0x1D7A6, "M", "χ"),
        (0x1D7A7, "M", "ψ"),
        (0x1D7A8, "M", "ω"),
        (0x1D7A9, "M", "∇"),
        (0x1D7AA, "M", "α"),
        (0x1D7AB, "M", "β"),
        (0x1D7AC, "M", "γ"),
        (0x1D7AD, "M", "δ"),
        (0x1D7AE, "M", "ε"),
        (0x1D7AF, "M", "ζ"),
        (0x1D7B0, "M", "η"),
        (0x1D7B1, "M", "θ"),
        (0x1D7B2, "M", "ι"),
        (0x1D7B3, "M", "κ"),
    ]


def _seg_70() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1D7B4, "M", "λ"),
        (0x1D7B5, "M", "μ"),
        (0x1D7B6, "M", "ν"),
        (0x1D7B7, "M", "ξ"),
        (0x1D7B8, "M", "ο"),
        (0x1D7B9, "M", "π"),
        (0x1D7BA, "M", "ρ"),
        (0x1D7BB, "M", "σ"),
        (0x1D7BD, "M", "τ"),
        (0x1D7BE, "M", "υ"),
        (0x1D7BF, "M", "φ"),
        (0x1D7C0, "M", "χ"),
        (0x1D7C1, "M", "ψ"),
        (0x1D7C2, "M", "ω"),
        (0x1D7C3, "M", "∂"),
        (0x1D7C4, "M", "ε"),
        (0x1D7C5, "M", "θ"),
        (0x1D7C6, "M", "κ"),
        (0x1D7C7, "M", "φ"),
        (0x1D7C8, "M", "ρ"),
        (0x1D7C9, "M", "π"),
        (0x1D7CA, "M", "ϝ"),
        (0x1D7CC, "X"),
        (0x1D7CE, "M", "0"),
        (0x1D7CF, "M", "1"),
        (0x1D7D0, "M", "2"),
        (0x1D7D1, "M", "3"),
        (0x1D7D2, "M", "4"),
        (0x1D7D3, "M", "5"),
        (0x1D7D4, "M", "6"),
        (0x1D7D5, "M", "7"),
        (0x1D7D6, "M", "8"),
        (0x1D7D7, "M", "9"),
        (0x1D7D8, "M", "0"),
        (0x1D7D9, "M", "1"),
        (0x1D7DA, "M", "2"),
        (0x1D7DB, "M", "3"),
        (0x1D7DC, "M", "4"),
        (0x1D7DD, "M", "5"),
        (0x1D7DE, "M", "6"),
        (0x1D7DF, "M", "7"),
        (0x1D7E0, "M", "8"),
        (0x1D7E1, "M", "9"),
        (0x1D7E2, "M", "0"),
        (0x1D7E3, "M", "1"),
        (0x1D7E4, "M", "2"),
        (0x1D7E5, "M", "3"),
        (0x1D7E6, "M", "4"),
        (0x1D7E7, "M", "5"),
        (0x1D7E8, "M", "6"),
        (0x1D7E9, "M", "7"),
        (0x1D7EA, "M", "8"),
        (0x1D7EB, "M", "9"),
        (0x1D7EC, "M", "0"),
        (0x1D7ED, "M", "1"),
        (0x1D7EE, "M", "2"),
        (0x1D7EF, "M", "3"),
        (0x1D7F0, "M", "4"),
        (0x1D7F1, "M", "5"),
        (0x1D7F2, "M", "6"),
        (0x1D7F3, "M", "7"),
        (0x1D7F4, "M", "8"),
        (0x1D7F5, "M", "9"),
        (0x1D7F6, "M", "0"),
        (0x1D7F7, "M", "1"),
        (0x1D7F8, "M", "2"),
        (0x1D7F9, "M", "3"),
        (0x1D7FA, "M", "4"),
        (0x1D7FB, "M", "5"),
        (0x1D7FC, "M", "6"),
        (0x1D7FD, "M", "7"),
        (0x1D7FE, "M", "8"),
        (0x1D7FF, "M", "9"),
        (0x1D800, "V"),
        (0x1DA8C, "X"),
        (0x1DA9B, "V"),
        (0x1DAA0, "X"),
        (0x1DAA1, "V"),
        (0x1DAB0, "X"),
        (0x1DF00, "V"),
        (0x1DF1F, "X"),
        (0x1DF25, "V"),
        (0x1DF2B, "X"),
        (0x1E000, "V"),
        (0x1E007, "X"),
        (0x1E008, "V"),
        (0x1E019, "X"),
        (0x1E01B, "V"),
        (0x1E022, "X"),
        (0x1E023, "V"),
        (0x1E025, "X"),
        (0x1E026, "V"),
        (0x1E02B, "X"),
        (0x1E030, "M", "а"),
        (0x1E031, "M", "б"),
        (0x1E032, "M", "в"),
        (0x1E033, "M", "г"),
        (0x1E034, "M", "д"),
        (0x1E035, "M", "е"),
        (0x1E036, "M", "ж"),
    ]


def _seg_71() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1E037, "M", "з"),
        (0x1E038, "M", "и"),
        (0x1E039, "M", "к"),
        (0x1E03A, "M", "л"),
        (0x1E03B, "M", "м"),
        (0x1E03C, "M", "о"),
        (0x1E03D, "M", "п"),
        (0x1E03E, "M", "р"),
        (0x1E03F, "M", "с"),
        (0x1E040, "M", "т"),
        (0x1E041, "M", "у"),
        (0x1E042, "M", "ф"),
        (0x1E043, "M", "х"),
        (0x1E044, "M", "ц"),
        (0x1E045, "M", "ч"),
        (0x1E046, "M", "ш"),
        (0x1E047, "M", "ы"),
        (0x1E048, "M", "э"),
        (0x1E049, "M", "ю"),
        (0x1E04A, "M", "ꚉ"),
        (0x1E04B, "M", "ә"),
        (0x1E04C, "M", "і"),
        (0x1E04D, "M", "ј"),
        (0x1E04E, "M", "ө"),
        (0x1E04F, "M", "ү"),
        (0x1E050, "M", "ӏ"),
        (0x1E051, "M", "а"),
        (0x1E052, "M", "б"),
        (0x1E053, "M", "в"),
        (0x1E054, "M", "г"),
        (0x1E055, "M", "д"),
        (0x1E056, "M", "е"),
        (0x1E057, "M", "ж"),
        (0x1E058, "M", "з"),
        (0x1E059, "M", "и"),
        (0x1E05A, "M", "к"),
        (0x1E05B, "M", "л"),
        (0x1E05C, "M", "о"),
        (0x1E05D, "M", "п"),
        (0x1E05E, "M", "с"),
        (0x1E05F, "M", "у"),
        (0x1E060, "M", "ф"),
        (0x1E061, "M", "х"),
        (0x1E062, "M", "ц"),
        (0x1E063, "M", "ч"),
        (0x1E064, "M", "ш"),
        (0x1E065, "M", "ъ"),
        (0x1E066, "M", "ы"),
        (0x1E067, "M", "ґ"),
        (0x1E068, "M", "і"),
        (0x1E069, "M", "ѕ"),
        (0x1E06A, "M", "џ"),
        (0x1E06B, "M", "ҫ"),
        (0x1E06C, "M", "ꙑ"),
        (0x1E06D, "M", "ұ"),
        (0x1E06E, "X"),
        (0x1E08F, "V"),
        (0x1E090, "X"),
        (0x1E100, "V"),
        (0x1E12D, "X"),
        (0x1E130, "V"),
        (0x1E13E, "X"),
        (0x1E140, "V"),
        (0x1E14A, "X"),
        (0x1E14E, "V"),
        (0x1E150, "X"),
        (0x1E290, "V"),
        (0x1E2AF, "X"),
        (0x1E2C0, "V"),
        (0x1E2FA, "X"),
        (0x1E2FF, "V"),
        (0x1E300, "X"),
        (0x1E4D0, "V"),
        (0x1E4FA, "X"),
        (0x1E7E0, "V"),
        (0x1E7E7, "X"),
        (0x1E7E8, "V"),
        (0x1E7EC, "X"),
        (0x1E7ED, "V"),
        (0x1E7EF, "X"),
        (0x1E7F0, "V"),
        (0x1E7FF, "X"),
        (0x1E800, "V"),
        (0x1E8C5, "X"),
        (0x1E8C7, "V"),
        (0x1E8D7, "X"),
        (0x1E900, "M", "𞤢"),
        (0x1E901, "M", "𞤣"),
        (0x1E902, "M", "𞤤"),
        (0x1E903, "M", "𞤥"),
        (0x1E904, "M", "𞤦"),
        (0x1E905, "M", "𞤧"),
        (0x1E906, "M", "𞤨"),
        (0x1E907, "M", "𞤩"),
        (0x1E908, "M", "𞤪"),
        (0x1E909, "M", "𞤫"),
        (0x1E90A, "M", "𞤬"),
        (0x1E90B, "M", "𞤭"),
        (0x1E90C, "M", "𞤮"),
        (0x1E90D, "M", "𞤯"),
    ]


def _seg_72() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1E90E, "M", "𞤰"),
        (0x1E90F, "M", "𞤱"),
        (0x1E910, "M", "𞤲"),
        (0x1E911, "M", "𞤳"),
        (0x1E912, "M", "𞤴"),
        (0x1E913, "M", "𞤵"),
        (0x1E914, "M", "𞤶"),
        (0x1E915, "M", "𞤷"),
        (0x1E916, "M", "𞤸"),
        (0x1E917, "M", "𞤹"),
        (0x1E918, "M", "𞤺"),
        (0x1E919, "M", "𞤻"),
        (0x1E91A, "M", "𞤼"),
        (0x1E91B, "M", "𞤽"),
        (0x1E91C, "M", "𞤾"),
        (0x1E91D, "M", "𞤿"),
        (0x1E91E, "M", "𞥀"),
        (0x1E91F, "M", "𞥁"),
        (0x1E920, "M", "𞥂"),
        (0x1E921, "M", "𞥃"),
        (0x1E922, "V"),
        (0x1E94C, "X"),
        (0x1E950, "V"),
        (0x1E95A, "X"),
        (0x1E95E, "V"),
        (0x1E960, "X"),
        (0x1EC71, "V"),
        (0x1ECB5, "X"),
        (0x1ED01, "V"),
        (0x1ED3E, "X"),
        (0x1EE00, "M", "ا"),
        (0x1EE01, "M", "ب"),
        (0x1EE02, "M", "ج"),
        (0x1EE03, "M", "د"),
        (0x1EE04, "X"),
        (0x1EE05, "M", "و"),
        (0x1EE06, "M", "ز"),
        (0x1EE07, "M", "ح"),
        (0x1EE08, "M", "ط"),
        (0x1EE09, "M", "ي"),
        (0x1EE0A, "M", "ك"),
        (0x1EE0B, "M", "ل"),
        (0x1EE0C, "M", "م"),
        (0x1EE0D, "M", "ن"),
        (0x1EE0E, "M", "س"),
        (0x1EE0F, "M", "ع"),
        (0x1EE10, "M", "ف"),
        (0x1EE11, "M", "ص"),
        (0x1EE12, "M", "ق"),
        (0x1EE13, "M", "ر"),
        (0x1EE14, "M", "ش"),
        (0x1EE15, "M", "ت"),
        (0x1EE16, "M", "ث"),
        (0x1EE17, "M", "خ"),
        (0x1EE18, "M", "ذ"),
        (0x1EE19, "M", "ض"),
        (0x1EE1A, "M", "ظ"),
        (0x1EE1B, "M", "غ"),
        (0x1EE1C, "M", "ٮ"),
        (0x1EE1D, "M", "ں"),
        (0x1EE1E, "M", "ڡ"),
        (0x1EE1F, "M", "ٯ"),
        (0x1EE20, "X"),
        (0x1EE21, "M", "ب"),
        (0x1EE22, "M", "ج"),
        (0x1EE23, "X"),
        (0x1EE24, "M", "ه"),
        (0x1EE25, "X"),
        (0x1EE27, "M", "ح"),
        (0x1EE28, "X"),
        (0x1EE29, "M", "ي"),
        (0x1EE2A, "M", "ك"),
        (0x1EE2B, "M", "ل"),
        (0x1EE2C, "M", "م"),
        (0x1EE2D, "M", "ن"),
        (0x1EE2E, "M", "س"),
        (0x1EE2F, "M", "ع"),
        (0x1EE30, "M", "ف"),
        (0x1EE31, "M", "ص"),
        (0x1EE32, "M", "ق"),
        (0x1EE33, "X"),
        (0x1EE34, "M", "ش"),
        (0x1EE35, "M", "ت"),
        (0x1EE36, "M", "ث"),
        (0x1EE37, "M", "خ"),
        (0x1EE38, "X"),
        (0x1EE39, "M", "ض"),
        (0x1EE3A, "X"),
        (0x1EE3B, "M", "غ"),
        (0x1EE3C, "X"),
        (0x1EE42, "M", "ج"),
        (0x1EE43, "X"),
        (0x1EE47, "M", "ح"),
        (0x1EE48, "X"),
        (0x1EE49, "M", "ي"),
        (0x1EE4A, "X"),
        (0x1EE4B, "M", "ل"),
        (0x1EE4C, "X"),
        (0x1EE4D, "M", "ن"),
        (0x1EE4E, "M", "س"),
    ]


def _seg_73() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1EE4F, "M", "ع"),
        (0x1EE50, "X"),
        (0x1EE51, "M", "ص"),
        (0x1EE52, "M", "ق"),
        (0x1EE53, "X"),
        (0x1EE54, "M", "ش"),
        (0x1EE55, "X"),
        (0x1EE57, "M", "خ"),
        (0x1EE58, "X"),
        (0x1EE59, "M", "ض"),
        (0x1EE5A, "X"),
        (0x1EE5B, "M", "غ"),
        (0x1EE5C, "X"),
        (0x1EE5D, "M", "ں"),
        (0x1EE5E, "X"),
        (0x1EE5F, "M", "ٯ"),
        (0x1EE60, "X"),
        (0x1EE61, "M", "ب"),
        (0x1EE62, "M", "ج"),
        (0x1EE63, "X"),
        (0x1EE64, "M", "ه"),
        (0x1EE65, "X"),
        (0x1EE67, "M", "ح"),
        (0x1EE68, "M", "ط"),
        (0x1EE69, "M", "ي"),
        (0x1EE6A, "M", "ك"),
        (0x1EE6B, "X"),
        (0x1EE6C, "M", "م"),
        (0x1EE6D, "M", "ن"),
        (0x1EE6E, "M", "س"),
        (0x1EE6F, "M", "ع"),
        (0x1EE70, "M", "ف"),
        (0x1EE71, "M", "ص"),
        (0x1EE72, "M", "ق"),
        (0x1EE73, "X"),
        (0x1EE74, "M", "ش"),
        (0x1EE75, "M", "ت"),
        (0x1EE76, "M", "ث"),
        (0x1EE77, "M", "خ"),
        (0x1EE78, "X"),
        (0x1EE79, "M", "ض"),
        (0x1EE7A, "M", "ظ"),
        (0x1EE7B, "M", "غ"),
        (0x1EE7C, "M", "ٮ"),
        (0x1EE7D, "X"),
        (0x1EE7E, "M", "ڡ"),
        (0x1EE7F, "X"),
        (0x1EE80, "M", "ا"),
        (0x1EE81, "M", "ب"),
        (0x1EE82, "M", "ج"),
        (0x1EE83, "M", "د"),
        (0x1EE84, "M", "ه"),
        (0x1EE85, "M", "و"),
        (0x1EE86, "M", "ز"),
        (0x1EE87, "M", "ح"),
        (0x1EE88, "M", "ط"),
        (0x1EE89, "M", "ي"),
        (0x1EE8A, "X"),
        (0x1EE8B, "M", "ل"),
        (0x1EE8C, "M", "م"),
        (0x1EE8D, "M", "ن"),
        (0x1EE8E, "M", "س"),
        (0x1EE8F, "M", "ع"),
        (0x1EE90, "M", "ف"),
        (0x1EE91, "M", "ص"),
        (0x1EE92, "M", "ق"),
        (0x1EE93, "M", "ر"),
        (0x1EE94, "M", "ش"),
        (0x1EE95, "M", "ت"),
        (0x1EE96, "M", "ث"),
        (0x1EE97, "M", "خ"),
        (0x1EE98, "M", "ذ"),
        (0x1EE99, "M", "ض"),
        (0x1EE9A, "M", "ظ"),
        (0x1EE9B, "M", "غ"),
        (0x1EE9C, "X"),
        (0x1EEA1, "M", "ب"),
        (0x1EEA2, "M", "ج"),
        (0x1EEA3, "M", "د"),
        (0x1EEA4, "X"),
        (0x1EEA5, "M", "و"),
        (0x1EEA6, "M", "ز"),
        (0x1EEA7, "M", "ح"),
        (0x1EEA8, "M", "ط"),
        (0x1EEA9, "M", "ي"),
        (0x1EEAA, "X"),
        (0x1EEAB, "M", "ل"),
        (0x1EEAC, "M", "م"),
        (0x1EEAD, "M", "ن"),
        (0x1EEAE, "M", "س"),
        (0x1EEAF, "M", "ع"),
        (0x1EEB0, "M", "ف"),
        (0x1EEB1, "M", "ص"),
        (0x1EEB2, "M", "ق"),
        (0x1EEB3, "M", "ر"),
        (0x1EEB4, "M", "ش"),
        (0x1EEB5, "M", "ت"),
        (0x1EEB6, "M", "ث"),
        (0x1EEB7, "M", "خ"),
        (0x1EEB8, "M", "ذ"),
    ]


def _seg_74() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1EEB9, "M", "ض"),
        (0x1EEBA, "M", "ظ"),
        (0x1EEBB, "M", "غ"),
        (0x1EEBC, "X"),
        (0x1EEF0, "V"),
        (0x1EEF2, "X"),
        (0x1F000, "V"),
        (0x1F02C, "X"),
        (0x1F030, "V"),
        (0x1F094, "X"),
        (0x1F0A0, "V"),
        (0x1F0AF, "X"),
        (0x1F0B1, "V"),
        (0x1F0C0, "X"),
        (0x1F0C1, "V"),
        (0x1F0D0, "X"),
        (0x1F0D1, "V"),
        (0x1F0F6, "X"),
        (0x1F101, "3", "0,"),
        (0x1F102, "3", "1,"),
        (0x1F103, "3", "2,"),
        (0x1F104, "3", "3,"),
        (0x1F105, "3", "4,"),
        (0x1F106, "3", "5,"),
        (0x1F107, "3", "6,"),
        (0x1F108, "3", "7,"),
        (0x1F109, "3", "8,"),
        (0x1F10A, "3", "9,"),
        (0x1F10B, "V"),
        (0x1F110, "3", "(a)"),
        (0x1F111, "3", "(b)"),
        (0x1F112, "3", "(c)"),
        (0x1F113, "3", "(d)"),
        (0x1F114, "3", "(e)"),
        (0x1F115, "3", "(f)"),
        (0x1F116, "3", "(g)"),
        (0x1F117, "3", "(h)"),
        (0x1F118, "3", "(i)"),
        (0x1F119, "3", "(j)"),
        (0x1F11A, "3", "(k)"),
        (0x1F11B, "3", "(l)"),
        (0x1F11C, "3", "(m)"),
        (0x1F11D, "3", "(n)"),
        (0x1F11E, "3", "(o)"),
        (0x1F11F, "3", "(p)"),
        (0x1F120, "3", "(q)"),
        (0x1F121, "3", "(r)"),
        (0x1F122, "3", "(s)"),
        (0x1F123, "3", "(t)"),
        (0x1F124, "3", "(u)"),
        (0x1F125, "3", "(v)"),
        (0x1F126, "3", "(w)"),
        (0x1F127, "3", "(x)"),
        (0x1F128, "3", "(y)"),
        (0x1F129, "3", "(z)"),
        (0x1F12A, "M", "〔s〕"),
        (0x1F12B, "M", "c"),
        (0x1F12C, "M", "r"),
        (0x1F12D, "M", "cd"),
        (0x1F12E, "M", "wz"),
        (0x1F12F, "V"),
        (0x1F130, "M", "a"),
        (0x1F131, "M", "b"),
        (0x1F132, "M", "c"),
        (0x1F133, "M", "d"),
        (0x1F134, "M", "e"),
        (0x1F135, "M", "f"),
        (0x1F136, "M", "g"),
        (0x1F137, "M", "h"),
        (0x1F138, "M", "i"),
        (0x1F139, "M", "j"),
        (0x1F13A, "M", "k"),
        (0x1F13B, "M", "l"),
        (0x1F13C, "M", "m"),
        (0x1F13D, "M", "n"),
        (0x1F13E, "M", "o"),
        (0x1F13F, "M", "p"),
        (0x1F140, "M", "q"),
        (0x1F141, "M", "r"),
        (0x1F142, "M", "s"),
        (0x1F143, "M", "t"),
        (0x1F144, "M", "u"),
        (0x1F145, "M", "v"),
        (0x1F146, "M", "w"),
        (0x1F147, "M", "x"),
        (0x1F148, "M", "y"),
        (0x1F149, "M", "z"),
        (0x1F14A, "M", "hv"),
        (0x1F14B, "M", "mv"),
        (0x1F14C, "M", "sd"),
        (0x1F14D, "M", "ss"),
        (0x1F14E, "M", "ppv"),
        (0x1F14F, "M", "wc"),
        (0x1F150, "V"),
        (0x1F16A, "M", "mc"),
        (0x1F16B, "M", "md"),
        (0x1F16C, "M", "mr"),
        (0x1F16D, "V"),
        (0x1F190, "M", "dj"),
        (0x1F191, "V"),
    ]


def _seg_75() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1F1AE, "X"),
        (0x1F1E6, "V"),
        (0x1F200, "M", "ほか"),
        (0x1F201, "M", "ココ"),
        (0x1F202, "M", "サ"),
        (0x1F203, "X"),
        (0x1F210, "M", "手"),
        (0x1F211, "M", "字"),
        (0x1F212, "M", "双"),
        (0x1F213, "M", "デ"),
        (0x1F214, "M", "二"),
        (0x1F215, "M", "多"),
        (0x1F216, "M", "解"),
        (0x1F217, "M", "天"),
        (0x1F218, "M", "交"),
        (0x1F219, "M", "映"),
        (0x1F21A, "M", "無"),
        (0x1F21B, "M", "料"),
        (0x1F21C, "M", "前"),
        (0x1F21D, "M", "後"),
        (0x1F21E, "M", "再"),
        (0x1F21F, "M", "新"),
        (0x1F220, "M", "初"),
        (0x1F221, "M", "終"),
        (0x1F222, "M", "生"),
        (0x1F223, "M", "販"),
        (0x1F224, "M", "声"),
        (0x1F225, "M", "吹"),
        (0x1F226, "M", "演"),
        (0x1F227, "M", "投"),
        (0x1F228, "M", "捕"),
        (0x1F229, "M", "一"),
        (0x1F22A, "M", "三"),
        (0x1F22B, "M", "遊"),
        (0x1F22C, "M", "左"),
        (0x1F22D, "M", "中"),
        (0x1F22E, "M", "右"),
        (0x1F22F, "M", "指"),
        (0x1F230, "M", "走"),
        (0x1F231, "M", "打"),
        (0x1F232, "M", "禁"),
        (0x1F233, "M", "空"),
        (0x1F234, "M", "合"),
        (0x1F235, "M", "満"),
        (0x1F236, "M", "有"),
        (0x1F237, "M", "月"),
        (0x1F238, "M", "申"),
        (0x1F239, "M", "割"),
        (0x1F23A, "M", "営"),
        (0x1F23B, "M", "配"),
        (0x1F23C, "X"),
        (0x1F240, "M", "〔本〕"),
        (0x1F241, "M", "〔三〕"),
        (0x1F242, "M", "〔二〕"),
        (0x1F243, "M", "〔安〕"),
        (0x1F244, "M", "〔点〕"),
        (0x1F245, "M", "〔打〕"),
        (0x1F246, "M", "〔盗〕"),
        (0x1F247, "M", "〔勝〕"),
        (0x1F248, "M", "〔敗〕"),
        (0x1F249, "X"),
        (0x1F250, "M", "得"),
        (0x1F251, "M", "可"),
        (0x1F252, "X"),
        (0x1F260, "V"),
        (0x1F266, "X"),
        (0x1F300, "V"),
        (0x1F6D8, "X"),
        (0x1F6DC, "V"),
        (0x1F6ED, "X"),
        (0x1F6F0, "V"),
        (0x1F6FD, "X"),
        (0x1F700, "V"),
        (0x1F777, "X"),
        (0x1F77B, "V"),
        (0x1F7DA, "X"),
        (0x1F7E0, "V"),
        (0x1F7EC, "X"),
        (0x1F7F0, "V"),
        (0x1F7F1, "X"),
        (0x1F800, "V"),
        (0x1F80C, "X"),
        (0x1F810, "V"),
        (0x1F848, "X"),
        (0x1F850, "V"),
        (0x1F85A, "X"),
        (0x1F860, "V"),
        (0x1F888, "X"),
        (0x1F890, "V"),
        (0x1F8AE, "X"),
        (0x1F8B0, "V"),
        (0x1F8B2, "X"),
        (0x1F900, "V"),
        (0x1FA54, "X"),
        (0x1FA60, "V"),
        (0x1FA6E, "X"),
        (0x1FA70, "V"),
        (0x1FA7D, "X"),
        (0x1FA80, "V"),
        (0x1FA89, "X"),
    ]


def _seg_76() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x1FA90, "V"),
        (0x1FABE, "X"),
        (0x1FABF, "V"),
        (0x1FAC6, "X"),
        (0x1FACE, "V"),
        (0x1FADC, "X"),
        (0x1FAE0, "V"),
        (0x1FAE9, "X"),
        (0x1FAF0, "V"),
        (0x1FAF9, "X"),
        (0x1FB00, "V"),
        (0x1FB93, "X"),
        (0x1FB94, "V"),
        (0x1FBCB, "X"),
        (0x1FBF0, "M", "0"),
        (0x1FBF1, "M", "1"),
        (0x1FBF2, "M", "2"),
        (0x1FBF3, "M", "3"),
        (0x1FBF4, "M", "4"),
        (0x1FBF5, "M", "5"),
        (0x1FBF6, "M", "6"),
        (0x1FBF7, "M", "7"),
        (0x1FBF8, "M", "8"),
        (0x1FBF9, "M", "9"),
        (0x1FBFA, "X"),
        (0x20000, "V"),
        (0x2A6E0, "X"),
        (0x2A700, "V"),
        (0x2B73A, "X"),
        (0x2B740, "V"),
        (0x2B81E, "X"),
        (0x2B820, "V"),
        (0x2CEA2, "X"),
        (0x2CEB0, "V"),
        (0x2EBE1, "X"),
        (0x2EBF0, "V"),
        (0x2EE5E, "X"),
        (0x2F800, "M", "丽"),
        (0x2F801, "M", "丸"),
        (0x2F802, "M", "乁"),
        (0x2F803, "M", "𠄢"),
        (0x2F804, "M", "你"),
        (0x2F805, "M", "侮"),
        (0x2F806, "M", "侻"),
        (0x2F807, "M", "倂"),
        (0x2F808, "M", "偺"),
        (0x2F809, "M", "備"),
        (0x2F80A, "M", "僧"),
        (0x2F80B, "M", "像"),
        (0x2F80C, "M", "㒞"),
        (0x2F80D, "M", "𠘺"),
        (0x2F80E, "M", "免"),
        (0x2F80F, "M", "兔"),
        (0x2F810, "M", "兤"),
        (0x2F811, "M", "具"),
        (0x2F812, "M", "𠔜"),
        (0x2F813, "M", "㒹"),
        (0x2F814, "M", "內"),
        (0x2F815, "M", "再"),
        (0x2F816, "M", "𠕋"),
        (0x2F817, "M", "冗"),
        (0x2F818, "M", "冤"),
        (0x2F819, "M", "仌"),
        (0x2F81A, "M", "冬"),
        (0x2F81B, "M", "况"),
        (0x2F81C, "M", "𩇟"),
        (0x2F81D, "M", "凵"),
        (0x2F81E, "M", "刃"),
        (0x2F81F, "M", "㓟"),
        (0x2F820, "M", "刻"),
        (0x2F821, "M", "剆"),
        (0x2F822, "M", "割"),
        (0x2F823, "M", "剷"),
        (0x2F824, "M", "㔕"),
        (0x2F825, "M", "勇"),
        (0x2F826, "M", "勉"),
        (0x2F827, "M", "勤"),
        (0x2F828, "M", "勺"),
        (0x2F829, "M", "包"),
        (0x2F82A, "M", "匆"),
        (0x2F82B, "M", "北"),
        (0x2F82C, "M", "卉"),
        (0x2F82D, "M", "卑"),
        (0x2F82E, "M", "博"),
        (0x2F82F, "M", "即"),
        (0x2F830, "M", "卽"),
        (0x2F831, "M", "卿"),
        (0x2F834, "M", "𠨬"),
        (0x2F835, "M", "灰"),
        (0x2F836, "M", "及"),
        (0x2F837, "M", "叟"),
        (0x2F838, "M", "𠭣"),
        (0x2F839, "M", "叫"),
        (0x2F83A, "M", "叱"),
        (0x2F83B, "M", "吆"),
        (0x2F83C, "M", "咞"),
        (0x2F83D, "M", "吸"),
        (0x2F83E, "M", "呈"),
        (0x2F83F, "M", "周"),
        (0x2F840, "M", "咢"),
    ]


def _seg_77() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x2F841, "M", "哶"),
        (0x2F842, "M", "唐"),
        (0x2F843, "M", "啓"),
        (0x2F844, "M", "啣"),
        (0x2F845, "M", "善"),
        (0x2F847, "M", "喙"),
        (0x2F848, "M", "喫"),
        (0x2F849, "M", "喳"),
        (0x2F84A, "M", "嗂"),
        (0x2F84B, "M", "圖"),
        (0x2F84C, "M", "嘆"),
        (0x2F84D, "M", "圗"),
        (0x2F84E, "M", "噑"),
        (0x2F84F, "M", "噴"),
        (0x2F850, "M", "切"),
        (0x2F851, "M", "壮"),
        (0x2F852, "M", "城"),
        (0x2F853, "M", "埴"),
        (0x2F854, "M", "堍"),
        (0x2F855, "M", "型"),
        (0x2F856, "M", "堲"),
        (0x2F857, "M", "報"),
        (0x2F858, "M", "墬"),
        (0x2F859, "M", "𡓤"),
        (0x2F85A, "M", "売"),
        (0x2F85B, "M", "壷"),
        (0x2F85C, "M", "夆"),
        (0x2F85D, "M", "多"),
        (0x2F85E, "M", "夢"),
        (0x2F85F, "M", "奢"),
        (0x2F860, "M", "𡚨"),
        (0x2F861, "M", "𡛪"),
        (0x2F862, "M", "姬"),
        (0x2F863, "M", "娛"),
        (0x2F864, "M", "娧"),
        (0x2F865, "M", "姘"),
        (0x2F866, "M", "婦"),
        (0x2F867, "M", "㛮"),
        (0x2F868, "X"),
        (0x2F869, "M", "嬈"),
        (0x2F86A, "M", "嬾"),
        (0x2F86C, "M", "𡧈"),
        (0x2F86D, "M", "寃"),
        (0x2F86E, "M", "寘"),
        (0x2F86F, "M", "寧"),
        (0x2F870, "M", "寳"),
        (0x2F871, "M", "𡬘"),
        (0x2F872, "M", "寿"),
        (0x2F873, "M", "将"),
        (0x2F874, "X"),
        (0x2F875, "M", "尢"),
        (0x2F876, "M", "㞁"),
        (0x2F877, "M", "屠"),
        (0x2F878, "M", "屮"),
        (0x2F879, "M", "峀"),
        (0x2F87A, "M", "岍"),
        (0x2F87B, "M", "𡷤"),
        (0x2F87C, "M", "嵃"),
        (0x2F87D, "M", "𡷦"),
        (0x2F87E, "M", "嵮"),
        (0x2F87F, "M", "嵫"),
        (0x2F880, "M", "嵼"),
        (0x2F881, "M", "巡"),
        (0x2F882, "M", "巢"),
        (0x2F883, "M", "㠯"),
        (0x2F884, "M", "巽"),
        (0x2F885, "M", "帨"),
        (0x2F886, "M", "帽"),
        (0x2F887, "M", "幩"),
        (0x2F888, "M", "㡢"),
        (0x2F889, "M", "𢆃"),
        (0x2F88A, "M", "㡼"),
        (0x2F88B, "M", "庰"),
        (0x2F88C, "M", "庳"),
        (0x2F88D, "M", "庶"),
        (0x2F88E, "M", "廊"),
        (0x2F88F, "M", "𪎒"),
        (0x2F890, "M", "廾"),
        (0x2F891, "M", "𢌱"),
        (0x2F893, "M", "舁"),
        (0x2F894, "M", "弢"),
        (0x2F896, "M", "㣇"),
        (0x2F897, "M", "𣊸"),
        (0x2F898, "M", "𦇚"),
        (0x2F899, "M", "形"),
        (0x2F89A, "M", "彫"),
        (0x2F89B, "M", "㣣"),
        (0x2F89C, "M", "徚"),
        (0x2F89D, "M", "忍"),
        (0x2F89E, "M", "志"),
        (0x2F89F, "M", "忹"),
        (0x2F8A0, "M", "悁"),
        (0x2F8A1, "M", "㤺"),
        (0x2F8A2, "M", "㤜"),
        (0x2F8A3, "M", "悔"),
        (0x2F8A4, "M", "𢛔"),
        (0x2F8A5, "M", "惇"),
        (0x2F8A6, "M", "慈"),
        (0x2F8A7, "M", "慌"),
        (0x2F8A8, "M", "慎"),
    ]


def _seg_78() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x2F8A9, "M", "慌"),
        (0x2F8AA, "M", "慺"),
        (0x2F8AB, "M", "憎"),
        (0x2F8AC, "M", "憲"),
        (0x2F8AD, "M", "憤"),
        (0x2F8AE, "M", "憯"),
        (0x2F8AF, "M", "懞"),
        (0x2F8B0, "M", "懲"),
        (0x2F8B1, "M", "懶"),
        (0x2F8B2, "M", "成"),
        (0x2F8B3, "M", "戛"),
        (0x2F8B4, "M", "扝"),
        (0x2F8B5, "M", "抱"),
        (0x2F8B6, "M", "拔"),
        (0x2F8B7, "M", "捐"),
        (0x2F8B8, "M", "𢬌"),
        (0x2F8B9, "M", "挽"),
        (0x2F8BA, "M", "拼"),
        (0x2F8BB, "M", "捨"),
        (0x2F8BC, "M", "掃"),
        (0x2F8BD, "M", "揤"),
        (0x2F8BE, "M", "𢯱"),
        (0x2F8BF, "M", "搢"),
        (0x2F8C0, "M", "揅"),
        (0x2F8C1, "M", "掩"),
        (0x2F8C2, "M", "㨮"),
        (0x2F8C3, "M", "摩"),
        (0x2F8C4, "M", "摾"),
        (0x2F8C5, "M", "撝"),
        (0x2F8C6, "M", "摷"),
        (0x2F8C7, "M", "㩬"),
        (0x2F8C8, "M", "敏"),
        (0x2F8C9, "M", "敬"),
        (0x2F8CA, "M", "𣀊"),
        (0x2F8CB, "M", "旣"),
        (0x2F8CC, "M", "書"),
        (0x2F8CD, "M", "晉"),
        (0x2F8CE, "M", "㬙"),
        (0x2F8CF, "M", "暑"),
        (0x2F8D0, "M", "㬈"),
        (0x2F8D1, "M", "㫤"),
        (0x2F8D2, "M", "冒"),
        (0x2F8D3, "M", "冕"),
        (0x2F8D4, "M", "最"),
        (0x2F8D5, "M", "暜"),
        (0x2F8D6, "M", "肭"),
        (0x2F8D7, "M", "䏙"),
        (0x2F8D8, "M", "朗"),
        (0x2F8D9, "M", "望"),
        (0x2F8DA, "M", "朡"),
        (0x2F8DB, "M", "杞"),
        (0x2F8DC, "M", "杓"),
        (0x2F8DD, "M", "𣏃"),
        (0x2F8DE, "M", "㭉"),
        (0x2F8DF, "M", "柺"),
        (0x2F8E0, "M", "枅"),
        (0x2F8E1, "M", "桒"),
        (0x2F8E2, "M", "梅"),
        (0x2F8E3, "M", "𣑭"),
        (0x2F8E4, "M", "梎"),
        (0x2F8E5, "M", "栟"),
        (0x2F8E6, "M", "椔"),
        (0x2F8E7, "M", "㮝"),
        (0x2F8E8, "M", "楂"),
        (0x2F8E9, "M", "榣"),
        (0x2F8EA, "M", "槪"),
        (0x2F8EB, "M", "檨"),
        (0x2F8EC, "M", "𣚣"),
        (0x2F8ED, "M", "櫛"),
        (0x2F8EE, "M", "㰘"),
        (0x2F8EF, "M", "次"),
        (0x2F8F0, "M", "𣢧"),
        (0x2F8F1, "M", "歔"),
        (0x2F8F2, "M", "㱎"),
        (0x2F8F3, "M", "歲"),
        (0x2F8F4, "M", "殟"),
        (0x2F8F5, "M", "殺"),
        (0x2F8F6, "M", "殻"),
        (0x2F8F7, "M", "𣪍"),
        (0x2F8F8, "M", "𡴋"),
        (0x2F8F9, "M", "𣫺"),
        (0x2F8FA, "M", "汎"),
        (0x2F8FB, "M", "𣲼"),
        (0x2F8FC, "M", "沿"),
        (0x2F8FD, "M", "泍"),
        (0x2F8FE, "M", "汧"),
        (0x2F8FF, "M", "洖"),
        (0x2F900, "M", "派"),
        (0x2F901, "M", "海"),
        (0x2F902, "M", "流"),
        (0x2F903, "M", "浩"),
        (0x2F904, "M", "浸"),
        (0x2F905, "M", "涅"),
        (0x2F906, "M", "𣴞"),
        (0x2F907, "M", "洴"),
        (0x2F908, "M", "港"),
        (0x2F909, "M", "湮"),
        (0x2F90A, "M", "㴳"),
        (0x2F90B, "M", "滋"),
        (0x2F90C, "M", "滇"),
    ]


def _seg_79() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x2F90D, "M", "𣻑"),
        (0x2F90E, "M", "淹"),
        (0x2F90F, "M", "潮"),
        (0x2F910, "M", "𣽞"),
        (0x2F911, "M", "𣾎"),
        (0x2F912, "M", "濆"),
        (0x2F913, "M", "瀹"),
        (0x2F914, "M", "瀞"),
        (0x2F915, "M", "瀛"),
        (0x2F916, "M", "㶖"),
        (0x2F917, "M", "灊"),
        (0x2F918, "M", "災"),
        (0x2F919, "M", "灷"),
        (0x2F91A, "M", "炭"),
        (0x2F91B, "M", "𠔥"),
        (0x2F91C, "M", "煅"),
        (0x2F91D, "M", "𤉣"),
        (0x2F91E, "M", "熜"),
        (0x2F91F, "X"),
        (0x2F920, "M", "爨"),
        (0x2F921, "M", "爵"),
        (0x2F922, "M", "牐"),
        (0x2F923, "M", "𤘈"),
        (0x2F924, "M", "犀"),
        (0x2F925, "M", "犕"),
        (0x2F926, "M", "𤜵"),
        (0x2F927, "M", "𤠔"),
        (0x2F928, "M", "獺"),
        (0x2F929, "M", "王"),
        (0x2F92A, "M", "㺬"),
        (0x2F92B, "M", "玥"),
        (0x2F92C, "M", "㺸"),
        (0x2F92E, "M", "瑇"),
        (0x2F92F, "M", "瑜"),
        (0x2F930, "M", "瑱"),
        (0x2F931, "M", "璅"),
        (0x2F932, "M", "瓊"),
        (0x2F933, "M", "㼛"),
        (0x2F934, "M", "甤"),
        (0x2F935, "M", "𤰶"),
        (0x2F936, "M", "甾"),
        (0x2F937, "M", "𤲒"),
        (0x2F938, "M", "異"),
        (0x2F939, "M", "𢆟"),
        (0x2F93A, "M", "瘐"),
        (0x2F93B, "M", "𤾡"),
        (0x2F93C, "M", "𤾸"),
        (0x2F93D, "M", "𥁄"),
        (0x2F93E, "M", "㿼"),
        (0x2F93F, "M", "䀈"),
        (0x2F940, "M", "直"),
        (0x2F941, "M", "𥃳"),
        (0x2F942, "M", "𥃲"),
        (0x2F943, "M", "𥄙"),
        (0x2F944, "M", "𥄳"),
        (0x2F945, "M", "眞"),
        (0x2F946, "M", "真"),
        (0x2F948, "M", "睊"),
        (0x2F949, "M", "䀹"),
        (0x2F94A, "M", "瞋"),
        (0x2F94B, "M", "䁆"),
        (0x2F94C, "M", "䂖"),
        (0x2F94D, "M", "𥐝"),
        (0x2F94E, "M", "硎"),
        (0x2F94F, "M", "碌"),
        (0x2F950, "M", "磌"),
        (0x2F951, "M", "䃣"),
        (0x2F952, "M", "𥘦"),
        (0x2F953, "M", "祖"),
        (0x2F954, "M", "𥚚"),
        (0x2F955, "M", "𥛅"),
        (0x2F956, "M", "福"),
        (0x2F957, "M", "秫"),
        (0x2F958, "M", "䄯"),
        (0x2F959, "M", "穀"),
        (0x2F95A, "M", "穊"),
        (0x2F95B, "M", "穏"),
        (0x2F95C, "M", "𥥼"),
        (0x2F95D, "M", "𥪧"),
        (0x2F95F, "X"),
        (0x2F960, "M", "䈂"),
        (0x2F961, "M", "𥮫"),
        (0x2F962, "M", "篆"),
        (0x2F963, "M", "築"),
        (0x2F964, "M", "䈧"),
        (0x2F965, "M", "𥲀"),
        (0x2F966, "M", "糒"),
        (0x2F967, "M", "䊠"),
        (0x2F968, "M", "糨"),
        (0x2F969, "M", "糣"),
        (0x2F96A, "M", "紀"),
        (0x2F96B, "M", "𥾆"),
        (0x2F96C, "M", "絣"),
        (0x2F96D, "M", "䌁"),
        (0x2F96E, "M", "緇"),
        (0x2F96F, "M", "縂"),
        (0x2F970, "M", "繅"),
        (0x2F971, "M", "䌴"),
        (0x2F972, "M", "𦈨"),
        (0x2F973, "M", "𦉇"),
    ]


def _seg_80() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x2F974, "M", "䍙"),
        (0x2F975, "M", "𦋙"),
        (0x2F976, "M", "罺"),
        (0x2F977, "M", "𦌾"),
        (0x2F978, "M", "羕"),
        (0x2F979, "M", "翺"),
        (0x2F97A, "M", "者"),
        (0x2F97B, "M", "𦓚"),
        (0x2F97C, "M", "𦔣"),
        (0x2F97D, "M", "聠"),
        (0x2F97E, "M", "𦖨"),
        (0x2F97F, "M", "聰"),
        (0x2F980, "M", "𣍟"),
        (0x2F981, "M", "䏕"),
        (0x2F982, "M", "育"),
        (0x2F983, "M", "脃"),
        (0x2F984, "M", "䐋"),
        (0x2F985, "M", "脾"),
        (0x2F986, "M", "媵"),
        (0x2F987, "M", "𦞧"),
        (0x2F988, "M", "𦞵"),
        (0x2F989, "M", "𣎓"),
        (0x2F98A, "M", "𣎜"),
        (0x2F98B, "M", "舁"),
        (0x2F98C, "M", "舄"),
        (0x2F98D, "M", "辞"),
        (0x2F98E, "M", "䑫"),
        (0x2F98F, "M", "芑"),
        (0x2F990, "M", "芋"),
        (0x2F991, "M", "芝"),
        (0x2F992, "M", "劳"),
        (0x2F993, "M", "花"),
        (0x2F994, "M", "芳"),
        (0x2F995, "M", "芽"),
        (0x2F996, "M", "苦"),
        (0x2F997, "M", "𦬼"),
        (0x2F998, "M", "若"),
        (0x2F999, "M", "茝"),
        (0x2F99A, "M", "荣"),
        (0x2F99B, "M", "莭"),
        (0x2F99C, "M", "茣"),
        (0x2F99D, "M", "莽"),
        (0x2F99E, "M", "菧"),
        (0x2F99F, "M", "著"),
        (0x2F9A0, "M", "荓"),
        (0x2F9A1, "M", "菊"),
        (0x2F9A2, "M", "菌"),
        (0x2F9A3, "M", "菜"),
        (0x2F9A4, "M", "𦰶"),
        (0x2F9A5, "M", "𦵫"),
        (0x2F9A6, "M", "𦳕"),
        (0x2F9A7, "M", "䔫"),
        (0x2F9A8, "M", "蓱"),
        (0x2F9A9, "M", "蓳"),
        (0x2F9AA, "M", "蔖"),
        (0x2F9AB, "M", "𧏊"),
        (0x2F9AC, "M", "蕤"),
        (0x2F9AD, "M", "𦼬"),
        (0x2F9AE, "M", "䕝"),
        (0x2F9AF, "M", "䕡"),
        (0x2F9B0, "M", "𦾱"),
        (0x2F9B1, "M", "𧃒"),
        (0x2F9B2, "M", "䕫"),
        (0x2F9B3, "M", "虐"),
        (0x2F9B4, "M", "虜"),
        (0x2F9B5, "M", "虧"),
        (0x2F9B6, "M", "虩"),
        (0x2F9B7, "M", "蚩"),
        (0x2F9B8, "M", "蚈"),
        (0x2F9B9, "M", "蜎"),
        (0x2F9BA, "M", "蛢"),
        (0x2F9BB, "M", "蝹"),
        (0x2F9BC, "M", "蜨"),
        (0x2F9BD, "M", "蝫"),
        (0x2F9BE, "M", "螆"),
        (0x2F9BF, "X"),
        (0x2F9C0, "M", "蟡"),
        (0x2F9C1, "M", "蠁"),
        (0x2F9C2, "M", "䗹"),
        (0x2F9C3, "M", "衠"),
        (0x2F9C4, "M", "衣"),
        (0x2F9C5, "M", "𧙧"),
        (0x2F9C6, "M", "裗"),
        (0x2F9C7, "M", "裞"),
        (0x2F9C8, "M", "䘵"),
        (0x2F9C9, "M", "裺"),
        (0x2F9CA, "M", "㒻"),
        (0x2F9CB, "M", "𧢮"),
        (0x2F9CC, "M", "𧥦"),
        (0x2F9CD, "M", "䚾"),
        (0x2F9CE, "M", "䛇"),
        (0x2F9CF, "M", "誠"),
        (0x2F9D0, "M", "諭"),
        (0x2F9D1, "M", "變"),
        (0x2F9D2, "M", "豕"),
        (0x2F9D3, "M", "𧲨"),
        (0x2F9D4, "M", "貫"),
        (0x2F9D5, "M", "賁"),
        (0x2F9D6, "M", "贛"),
        (0x2F9D7, "M", "起"),
    ]


def _seg_81() -> List[Union[Tuple[int, str], Tuple[int, str, str]]]:
    return [
        (0x2F9D8, "M", "𧼯"),
        (0x2F9D9, "M", "𠠄"),
        (0x2F9DA, "M", "跋"),
        (0x2F9DB, "M", "趼"),
        (0x2F9DC, "M", "跰"),
        (0x2F9DD, "M", "𠣞"),
        (0x2F9DE, "M", "軔"),
        (0x2F9DF, "M", "輸"),
        (0x2F9E0, "M", "𨗒"),
        (0x2F9E1, "M", "𨗭"),
        (0x2F9E2, "M", "邔"),
        (0x2F9E3, "M", "郱"),
        (0x2F9E4, "M", "鄑"),
        (0x2F9E5, "M", "𨜮"),
        (0x2F9E6, "M", "鄛"),
        (0x2F9E7, "M", "鈸"),
        (0x2F9E8, "M", "鋗"),
        (0x2F9E9, "M", "鋘"),
        (0x2F9EA, "M", "鉼"),
        (0x2F9EB, "M", "鏹"),
        (0x2F9EC, "M", "鐕"),
        (0x2F9ED, "M", "𨯺"),
        (0x2F9EE, "M", "開"),
        (0x2F9EF, "M", "䦕"),
        (0x2F9F0, "M", "閷"),
        (0x2F9F1, "M", "𨵷"),
        (0x2F9F2, "M", "䧦"),
        (0x2F9F3, "M", "雃"),
        (0x2F9F4, "M", "嶲"),
        (0x2F9F5, "M", "霣"),
        (0x2F9F6, "M", "𩅅"),
        (0x2F9F7, "M", "𩈚"),
        (0x2F9F8, "M", "䩮"),
        (0x2F9F9, "M", "䩶"),
        (0x2F9FA, "M", "韠"),
        (0x2F9FB, "M", "𩐊"),
        (0x2F9FC, "M", "䪲"),
        (0x2F9FD, "M", "𩒖"),
        (0x2F9FE, "M", "頋"),
        (0x2FA00, "M", "頩"),
        (0x2FA01, "M", "𩖶"),
        (0x2FA02, "M", "飢"),
        (0x2FA03, "M", "䬳"),
        (0x2FA04, "M", "餩"),
        (0x2FA05, "M", "馧"),
        (0x2FA06, "M", "駂"),
        (0x2FA07, "M", "駾"),
        (0x2FA08, "M", "䯎"),
        (0x2FA09, "M", "𩬰"),
        (0x2FA0A, "M", "鬒"),
        (0x2FA0B, "M", "鱀"),
        (0x2FA0C, "M", "鳽"),
        (0x2FA0D, "M", "䳎"),
        (0x2FA0E, "M", "䳭"),
        (0x2FA0F, "M", "鵧"),
        (0x2FA10, "M", "𪃎"),
        (0x2FA11, "M", "䳸"),
        (0x2FA12, "M", "𪄅"),
        (0x2FA13, "M", "𪈎"),
        (0x2FA14, "M", "𪊑"),
        (0x2FA15, "M", "麻"),
        (0x2FA16, "M", "䵖"),
        (0x2FA17, "M", "黹"),
        (0x2FA18, "M", "黾"),
        (0x2FA19, "M", "鼅"),
        (0x2FA1A, "M", "鼏"),
        (0x2FA1B, "M", "鼖"),
        (0x2FA1C, "M", "鼻"),
        (0x2FA1D, "M", "𪘀"),
        (0x2FA1E, "X"),
        (0x30000, "V"),
        (0x3134B, "X"),
        (0x31350, "V"),
        (0x323B0, "X"),
        (0xE0100, "I"),
        (0xE01F0, "X"),
    ]


uts46data = tuple(
    _seg_0()
    + _seg_1()
    + _seg_2()
    + _seg_3()
    + _seg_4()
    + _seg_5()
    + _seg_6()
    + _seg_7()
    + _seg_8()
    + _seg_9()
    + _seg_10()
    + _seg_11()
    + _seg_12()
    + _seg_13()
    + _seg_14()
    + _seg_15()
    + _seg_16()
    + _seg_17()
    + _seg_18()
    + _seg_19()
    + _seg_20()
    + _seg_21()
    + _seg_22()
    + _seg_23()
    + _seg_24()
    + _seg_25()
    + _seg_26()
    + _seg_27()
    + _seg_28()
    + _seg_29()
    + _seg_30()
    + _seg_31()
    + _seg_32()
    + _seg_33()
    + _seg_34()
    + _seg_35()
    + _seg_36()
    + _seg_37()
    + _seg_38()
    + _seg_39()
    + _seg_40()
    + _seg_41()
    + _seg_42()
    + _seg_43()
    + _seg_44()
    + _seg_45()
    + _seg_46()
    + _seg_47()
    + _seg_48()
    + _seg_49()
    + _seg_50()
    + _seg_51()
    + _seg_52()
    + _seg_53()
    + _seg_54()
    + _seg_55()
    + _seg_56()
    + _seg_57()
    + _seg_58()
    + _seg_59()
    + _seg_60()
    + _seg_61()
    + _seg_62()
    + _seg_63()
    + _seg_64()
    + _seg_65()
    + _seg_66()
    + _seg_67()
    + _seg_68()
    + _seg_69()
    + _seg_70()
    + _seg_71()
    + _seg_72()
    + _seg_73()
    + _seg_74()
    + _seg_75()
    + _seg_76()
    + _seg_77()
    + _seg_78()
    + _seg_79()
    + _seg_80()
    + _seg_81()
)  # type: Tuple[Union[Tuple[int, str], Tuple[int, str, str]], ...]