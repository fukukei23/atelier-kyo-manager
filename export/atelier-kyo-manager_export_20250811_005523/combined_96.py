
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\_internal.py ===
"""
A place for internal code

Some things are more easily handled Python.

"""
import ast
import math
import re
import sys
import warnings

from numpy import _NoValue
from numpy.exceptions import DTypePromotionError

from .multiarray import StringDType, array, dtype, promote_types

try:
    import ctypes
except ImportError:
    ctypes = None

IS_PYPY = sys.implementation.name == 'pypy'

if sys.byteorder == 'little':
    _nbo = '<'
else:
    _nbo = '>'

def _makenames_list(adict, align):
    allfields = []

    for fname, obj in adict.items():
        n = len(obj)
        if not isinstance(obj, tuple) or n not in (2, 3):
            raise ValueError("entry not a 2- or 3- tuple")
        if n > 2 and obj[2] == fname:
            continue
        num = int(obj[1])
        if num < 0:
            raise ValueError("invalid offset.")
        format = dtype(obj[0], align=align)
        if n > 2:
            title = obj[2]
        else:
            title = None
        allfields.append((fname, format, num, title))
    # sort by offsets
    allfields.sort(key=lambda x: x[2])
    names = [x[0] for x in allfields]
    formats = [x[1] for x in allfields]
    offsets = [x[2] for x in allfields]
    titles = [x[3] for x in allfields]

    return names, formats, offsets, titles

# Called in PyArray_DescrConverter function when
#  a dictionary without "names" and "formats"
#  fields is used as a data-type descriptor.
def _usefields(adict, align):
    try:
        names = adict[-1]
    except KeyError:
        names = None
    if names is None:
        names, formats, offsets, titles = _makenames_list(adict, align)
    else:
        formats = []
        offsets = []
        titles = []
        for name in names:
            res = adict[name]
            formats.append(res[0])
            offsets.append(res[1])
            if len(res) > 2:
                titles.append(res[2])
            else:
                titles.append(None)

    return dtype({"names": names,
                  "formats": formats,
                  "offsets": offsets,
                  "titles": titles}, align)


# construct an array_protocol descriptor list
#  from the fields attribute of a descriptor
# This calls itself recursively but should eventually hit
#  a descriptor that has no fields and then return
#  a simple typestring

def _array_descr(descriptor):
    fields = descriptor.fields
    if fields is None:
        subdtype = descriptor.subdtype
        if subdtype is None:
            if descriptor.metadata is None:
                return descriptor.str
            else:
                new = descriptor.metadata.copy()
                if new:
                    return (descriptor.str, new)
                else:
                    return descriptor.str
        else:
            return (_array_descr(subdtype[0]), subdtype[1])

    names = descriptor.names
    ordered_fields = [fields[x] + (x,) for x in names]
    result = []
    offset = 0
    for field in ordered_fields:
        if field[1] > offset:
            num = field[1] - offset
            result.append(('', f'|V{num}'))
            offset += num
        elif field[1] < offset:
            raise ValueError(
                "dtype.descr is not defined for types with overlapping or "
                "out-of-order fields")
        if len(field) > 3:
            name = (field[2], field[3])
        else:
            name = field[2]
        if field[0].subdtype:
            tup = (name, _array_descr(field[0].subdtype[0]),
                   field[0].subdtype[1])
        else:
            tup = (name, _array_descr(field[0]))
        offset += field[0].itemsize
        result.append(tup)

    if descriptor.itemsize > offset:
        num = descriptor.itemsize - offset
        result.append(('', f'|V{num}'))

    return result


# format_re was originally from numarray by J. Todd Miller

format_re = re.compile(r'(?P<order1>[<>|=]?)'
                       r'(?P<repeats> *[(]?[ ,0-9]*[)]? *)'
                       r'(?P<order2>[<>|=]?)'
                       r'(?P<dtype>[A-Za-z0-9.?]*(?:\[[a-zA-Z0-9,.]+\])?)')
sep_re = re.compile(r'\s*,\s*')
space_re = re.compile(r'\s+$')

# astr is a string (perhaps comma separated)

_convorder = {'=': _nbo}

def _commastring(astr):
    startindex = 0
    result = []
    islist = False
    while startindex < len(astr):
        mo = format_re.match(astr, pos=startindex)
        try:
            (order1, repeats, order2, dtype) = mo.groups()
        except (TypeError, AttributeError):
            raise ValueError(
                f'format number {len(result) + 1} of "{astr}" is not recognized'
                ) from None
        startindex = mo.end()
        # Separator or ending padding
        if startindex < len(astr):
            if space_re.match(astr, pos=startindex):
                startindex = len(astr)
            else:
                mo = sep_re.match(astr, pos=startindex)
                if not mo:
                    raise ValueError(
                        'format number %d of "%s" is not recognized' %
                        (len(result) + 1, astr))
                startindex = mo.end()
                islist = True

        if order2 == '':
            order = order1
        elif order1 == '':
            order = order2
        else:
            order1 = _convorder.get(order1, order1)
            order2 = _convorder.get(order2, order2)
            if (order1 != order2):
                raise ValueError(
                    f'inconsistent byte-order specification {order1} and {order2}')
            order = order1

        if order in ('|', '=', _nbo):
            order = ''
        dtype = order + dtype
        if repeats == '':
            newitem = dtype
        else:
            if (repeats[0] == "(" and repeats[-1] == ")"
                    and repeats[1:-1].strip() != ""
                    and "," not in repeats):
                warnings.warn(
                    'Passing in a parenthesized single number for repeats '
                    'is deprecated; pass either a single number or indicate '
                    'a tuple with a comma, like "(2,)".', DeprecationWarning,
                    stacklevel=2)
            newitem = (dtype, ast.literal_eval(repeats))

        result.append(newitem)

    return result if islist else result[0]

class dummy_ctype:

    def __init__(self, cls):
        self._cls = cls

    def __mul__(self, other):
        return self

    def __call__(self, *other):
        return self._cls(other)

    def __eq__(self, other):
        return self._cls == other._cls

    def __ne__(self, other):
        return self._cls != other._cls

def _getintp_ctype():
    val = _getintp_ctype.cache
    if val is not None:
        return val
    if ctypes is None:
        import numpy as np
        val = dummy_ctype(np.intp)
    else:
        char = dtype('n').char
        if char == 'i':
            val = ctypes.c_int
        elif char == 'l':
            val = ctypes.c_long
        elif char == 'q':
            val = ctypes.c_longlong
        else:
            val = ctypes.c_long
    _getintp_ctype.cache = val
    return val


_getintp_ctype.cache = None

# Used for .ctypes attribute of ndarray

class _missing_ctypes:
    def cast(self, num, obj):
        return num.value

    class c_void_p:
        def __init__(self, ptr):
            self.value = ptr


class _ctypes:
    def __init__(self, array, ptr=None):
        self._arr = array

        if ctypes:
            self._ctypes = ctypes
            self._data = self._ctypes.c_void_p(ptr)
        else:
            # fake a pointer-like object that holds onto the reference
            self._ctypes = _missing_ctypes()
            self._data = self._ctypes.c_void_p(ptr)
            self._data._objects = array

        if self._arr.ndim == 0:
            self._zerod = True
        else:
            self._zerod = False

    def data_as(self, obj):
        """
        Return the data pointer cast to a particular c-types object.
        For example, calling ``self._as_parameter_`` is equivalent to
        ``self.data_as(ctypes.c_void_p)``. Perhaps you want to use
        the data as a pointer to a ctypes array of floating-point data:
        ``self.data_as(ctypes.POINTER(ctypes.c_double))``.

        The returned pointer will keep a reference to the array.
        """
        # _ctypes.cast function causes a circular reference of self._data in
        # self._data._objects. Attributes of self._data cannot be released
        # until gc.collect is called. Make a copy of the pointer first then
        # let it hold the array reference. This is a workaround to circumvent
        # the CPython bug https://bugs.python.org/issue12836.
        ptr = self._ctypes.cast(self._data, obj)
        ptr._arr = self._arr
        return ptr

    def shape_as(self, obj):
        """
        Return the shape tuple as an array of some other c-types
        type. For example: ``self.shape_as(ctypes.c_short)``.
        """
        if self._zerod:
            return None
        return (obj * self._arr.ndim)(*self._arr.shape)

    def strides_as(self, obj):
        """
        Return the strides tuple as an array of some other
        c-types type. For example: ``self.strides_as(ctypes.c_longlong)``.
        """
        if self._zerod:
            return None
        return (obj * self._arr.ndim)(*self._arr.strides)

    @property
    def data(self):
        """
        A pointer to the memory area of the array as a Python integer.
        This memory area may contain data that is not aligned, or not in
        correct byte-order. The memory area may not even be writeable.
        The array flags and data-type of this array should be respected
        when passing this attribute to arbitrary C-code to avoid trouble
        that can include Python crashing. User Beware! The value of this
        attribute is exactly the same as:
        ``self._array_interface_['data'][0]``.

        Note that unlike ``data_as``, a reference won't be kept to the array:
        code like ``ctypes.c_void_p((a + b).ctypes.data)`` will result in a
        pointer to a deallocated array, and should be spelt
        ``(a + b).ctypes.data_as(ctypes.c_void_p)``
        """
        return self._data.value

    @property
    def shape(self):
        """
        (c_intp*self.ndim): A ctypes array of length self.ndim where
        the basetype is the C-integer corresponding to ``dtype('p')`` on this
        platform (see `~numpy.ctypeslib.c_intp`). This base-type could be
        `ctypes.c_int`, `ctypes.c_long`, or `ctypes.c_longlong` depending on
        the platform. The ctypes array contains the shape of
        the underlying array.
        """
        return self.shape_as(_getintp_ctype())

    @property
    def strides(self):
        """
        (c_intp*self.ndim): A ctypes array of length self.ndim where
        the basetype is the same as for the shape attribute. This ctypes
        array contains the strides information from the underlying array.
        This strides information is important for showing how many bytes
        must be jumped to get to the next element in the array.
        """
        return self.strides_as(_getintp_ctype())

    @property
    def _as_parameter_(self):
        """
        Overrides the ctypes semi-magic method

        Enables `c_func(some_array.ctypes)`
        """
        return self.data_as(ctypes.c_void_p)

    # Numpy 1.21.0, 2021-05-18

    def get_data(self):
        """Deprecated getter for the `_ctypes.data` property.

        .. deprecated:: 1.21
        """
        warnings.warn('"get_data" is deprecated. Use "data" instead',
                      DeprecationWarning, stacklevel=2)
        return self.data

    def get_shape(self):
        """Deprecated getter for the `_ctypes.shape` property.

        .. deprecated:: 1.21
        """
        warnings.warn('"get_shape" is deprecated. Use "shape" instead',
                      DeprecationWarning, stacklevel=2)
        return self.shape

    def get_strides(self):
        """Deprecated getter for the `_ctypes.strides` property.

        .. deprecated:: 1.21
        """
        warnings.warn('"get_strides" is deprecated. Use "strides" instead',
                      DeprecationWarning, stacklevel=2)
        return self.strides

    def get_as_parameter(self):
        """Deprecated getter for the `_ctypes._as_parameter_` property.

        .. deprecated:: 1.21
        """
        warnings.warn(
            '"get_as_parameter" is deprecated. Use "_as_parameter_" instead',
            DeprecationWarning, stacklevel=2,
        )
        return self._as_parameter_


def _newnames(datatype, order):
    """
    Given a datatype and an order object, return a new names tuple, with the
    order indicated
    """
    oldnames = datatype.names
    nameslist = list(oldnames)
    if isinstance(order, str):
        order = [order]
    seen = set()
    if isinstance(order, (list, tuple)):
        for name in order:
            try:
                nameslist.remove(name)
            except ValueError:
                if name in seen:
                    raise ValueError(f"duplicate field name: {name}") from None
                else:
                    raise ValueError(f"unknown field name: {name}") from None
            seen.add(name)
        return tuple(list(order) + nameslist)
    raise ValueError(f"unsupported order value: {order}")

def _copy_fields(ary):
    """Return copy of structured array with padding between fields removed.

    Parameters
    ----------
    ary : ndarray
       Structured array from which to remove padding bytes

    Returns
    -------
    ary_copy : ndarray
       Copy of ary with padding bytes removed
    """
    dt = ary.dtype
    copy_dtype = {'names': dt.names,
                  'formats': [dt.fields[name][0] for name in dt.names]}
    return array(ary, dtype=copy_dtype, copy=True)

def _promote_fields(dt1, dt2):
    """ Perform type promotion for two structured dtypes.

    Parameters
    ----------
    dt1 : structured dtype
        First dtype.
    dt2 : structured dtype
        Second dtype.

    Returns
    -------
    out : dtype
        The promoted dtype

    Notes
    -----
    If one of the inputs is aligned, the result will be.  The titles of
    both descriptors must match (point to the same field).
    """
    # Both must be structured and have the same names in the same order
    if (dt1.names is None or dt2.names is None) or dt1.names != dt2.names:
        raise DTypePromotionError(
                f"field names `{dt1.names}` and `{dt2.names}` mismatch.")

    # if both are identical, we can (maybe!) just return the same dtype.
    identical = dt1 is dt2
    new_fields = []
    for name in dt1.names:
        field1 = dt1.fields[name]
        field2 = dt2.fields[name]
        new_descr = promote_types(field1[0], field2[0])
        identical = identical and new_descr is field1[0]

        # Check that the titles match (if given):
        if field1[2:] != field2[2:]:
            raise DTypePromotionError(
                    f"field titles of field '{name}' mismatch")
        if len(field1) == 2:
            new_fields.append((name, new_descr))
        else:
            new_fields.append(((field1[2], name), new_descr))

    res = dtype(new_fields, align=dt1.isalignedstruct or dt2.isalignedstruct)

    # Might as well preserve identity (and metadata) if the dtype is identical
    # and the itemsize, offsets are also unmodified.  This could probably be
    # sped up, but also probably just be removed entirely.
    if identical and res.itemsize == dt1.itemsize:
        for name in dt1.names:
            if dt1.fields[name][1] != res.fields[name][1]:
                return res  # the dtype changed.
        return dt1

    return res


def _getfield_is_safe(oldtype, newtype, offset):
    """ Checks safety of getfield for object arrays.

    As in _view_is_safe, we need to check that memory containing objects is not
    reinterpreted as a non-object datatype and vice versa.

    Parameters
    ----------
    oldtype : data-type
        Data type of the original ndarray.
    newtype : data-type
        Data type of the field being accessed by ndarray.getfield
    offset : int
        Offset of the field being accessed by ndarray.getfield

    Raises
    ------
    TypeError
        If the field access is invalid

    """
    if newtype.hasobject or oldtype.hasobject:
        if offset == 0 and newtype == oldtype:
            return
        if oldtype.names is not None:
            for name in oldtype.names:
                if (oldtype.fields[name][1] == offset and
                        oldtype.fields[name][0] == newtype):
                    return
        raise TypeError("Cannot get/set field of an object array")
    return

def _view_is_safe(oldtype, newtype):
    """ Checks safety of a view involving object arrays, for example when
    doing::

        np.zeros(10, dtype=oldtype).view(newtype)

    Parameters
    ----------
    oldtype : data-type
        Data type of original ndarray
    newtype : data-type
        Data type of the view

    Raises
    ------
    TypeError
        If the new type is incompatible with the old type.

    """

    # if the types are equivalent, there is no problem.
    # for example: dtype((np.record, 'i4,i4')) == dtype((np.void, 'i4,i4'))
    if oldtype == newtype:
        return

    if newtype.hasobject or oldtype.hasobject:
        raise TypeError("Cannot change data-type for array of references.")
    return


# Given a string containing a PEP 3118 format specifier,
# construct a NumPy dtype

_pep3118_native_map = {
    '?': '?',
    'c': 'S1',
    'b': 'b',
    'B': 'B',
    'h': 'h',
    'H': 'H',
    'i': 'i',
    'I': 'I',
    'l': 'l',
    'L': 'L',
    'q': 'q',
    'Q': 'Q',
    'e': 'e',
    'f': 'f',
    'd': 'd',
    'g': 'g',
    'Zf': 'F',
    'Zd': 'D',
    'Zg': 'G',
    's': 'S',
    'w': 'U',
    'O': 'O',
    'x': 'V',  # padding
}
_pep3118_native_typechars = ''.join(_pep3118_native_map.keys())

_pep3118_standard_map = {
    '?': '?',
    'c': 'S1',
    'b': 'b',
    'B': 'B',
    'h': 'i2',
    'H': 'u2',
    'i': 'i4',
    'I': 'u4',
    'l': 'i4',
    'L': 'u4',
    'q': 'i8',
    'Q': 'u8',
    'e': 'f2',
    'f': 'f',
    'd': 'd',
    'Zf': 'F',
    'Zd': 'D',
    's': 'S',
    'w': 'U',
    'O': 'O',
    'x': 'V',  # padding
}
_pep3118_standard_typechars = ''.join(_pep3118_standard_map.keys())

_pep3118_unsupported_map = {
    'u': 'UCS-2 strings',
    '&': 'pointers',
    't': 'bitfields',
    'X': 'function pointers',
}

class _Stream:
    def __init__(self, s):
        self.s = s
        self.byteorder = '@'

    def advance(self, n):
        res = self.s[:n]
        self.s = self.s[n:]
        return res

    def consume(self, c):
        if self.s[:len(c)] == c:
            self.advance(len(c))
            return True
        return False

    def consume_until(self, c):
        if callable(c):
            i = 0
            while i < len(self.s) and not c(self.s[i]):
                i = i + 1
            return self.advance(i)
        else:
            i = self.s.index(c)
            res = self.advance(i)
            self.advance(len(c))
            return res

    @property
    def next(self):
        return self.s[0]

    def __bool__(self):
        return bool(self.s)


def _dtype_from_pep3118(spec):
    stream = _Stream(spec)
    dtype, align = __dtype_from_pep3118(stream, is_subdtype=False)
    return dtype

def __dtype_from_pep3118(stream, is_subdtype):
    field_spec = {
        'names': [],
        'formats': [],
        'offsets': [],
        'itemsize': 0
    }
    offset = 0
    common_alignment = 1
    is_padding = False

    # Parse spec
    while stream:
        value = None

        # End of structure, bail out to upper level
        if stream.consume('}'):
            break

        # Sub-arrays (1)
        shape = None
        if stream.consume('('):
            shape = stream.consume_until(')')
            shape = tuple(map(int, shape.split(',')))

        # Byte order
        if stream.next in ('@', '=', '<', '>', '^', '!'):
            byteorder = stream.advance(1)
            if byteorder == '!':
                byteorder = '>'
            stream.byteorder = byteorder

        # Byte order characters also control native vs. standard type sizes
        if stream.byteorder in ('@', '^'):
            type_map = _pep3118_native_map
            type_map_chars = _pep3118_native_typechars
        else:
            type_map = _pep3118_standard_map
            type_map_chars = _pep3118_standard_typechars

        # Item sizes
        itemsize_str = stream.consume_until(lambda c: not c.isdigit())
        if itemsize_str:
            itemsize = int(itemsize_str)
        else:
            itemsize = 1

        # Data types
        is_padding = False

        if stream.consume('T{'):
            value, align = __dtype_from_pep3118(
                stream, is_subdtype=True)
        elif stream.next in type_map_chars:
            if stream.next == 'Z':
                typechar = stream.advance(2)
            else:
                typechar = stream.advance(1)

            is_padding = (typechar == 'x')
            dtypechar = type_map[typechar]
            if dtypechar in 'USV':
                dtypechar += '%d' % itemsize
                itemsize = 1
            numpy_byteorder = {'@': '=', '^': '='}.get(
                stream.byteorder, stream.byteorder)
            value = dtype(numpy_byteorder + dtypechar)
            align = value.alignment
        elif stream.next in _pep3118_unsupported_map:
            desc = _pep3118_unsupported_map[stream.next]
            raise NotImplementedError(
                f"Unrepresentable PEP 3118 data type {stream.next!r} ({desc})")
        else:
            raise ValueError(
                f"Unknown PEP 3118 data type specifier {stream.s!r}"
            )

        #
        # Native alignment may require padding
        #
        # Here we assume that the presence of a '@' character implicitly
        # implies that the start of the array is *already* aligned.
        #
        extra_offset = 0
        if stream.byteorder == '@':
            start_padding = (-offset) % align
            intra_padding = (-value.itemsize) % align

            offset += start_padding

            if intra_padding != 0:
                if itemsize > 1 or (shape is not None and _prod(shape) > 1):
                    # Inject internal padding to the end of the sub-item
                    value = _add_trailing_padding(value, intra_padding)
                else:
                    # We can postpone the injection of internal padding,
                    # as the item appears at most once
                    extra_offset += intra_padding

            # Update common alignment
            common_alignment = _lcm(align, common_alignment)

        # Convert itemsize to sub-array
        if itemsize != 1:
            value = dtype((value, (itemsize,)))

        # Sub-arrays (2)
        if shape is not None:
            value = dtype((value, shape))

        # Field name
        if stream.consume(':'):
            name = stream.consume_until(':')
        else:
            name = None

        if not (is_padding and name is None):
            if name is not None and name in field_spec['names']:
                raise RuntimeError(
                    f"Duplicate field name '{name}' in PEP3118 format"
                )
            field_spec['names'].append(name)
            field_spec['formats'].append(value)
            field_spec['offsets'].append(offset)

        offset += value.itemsize
        offset += extra_offset

        field_spec['itemsize'] = offset

    # extra final padding for aligned types
    if stream.byteorder == '@':
        field_spec['itemsize'] += (-offset) % common_alignment

    # Check if this was a simple 1-item type, and unwrap it
    if (field_spec['names'] == [None]
            and field_spec['offsets'][0] == 0
            and field_spec['itemsize'] == field_spec['formats'][0].itemsize
            and not is_subdtype):
        ret = field_spec['formats'][0]
    else:
        _fix_names(field_spec)
        ret = dtype(field_spec)

    # Finished
    return ret, common_alignment

def _fix_names(field_spec):
    """ Replace names which are None with the next unused f%d name """
    names = field_spec['names']
    for i, name in enumerate(names):
        if name is not None:
            continue

        j = 0
        while True:
            name = f'f{j}'
            if name not in names:
                break
            j = j + 1
        names[i] = name

def _add_trailing_padding(value, padding):
    """Inject the specified number of padding bytes at the end of a dtype"""
    if value.fields is None:
        field_spec = {
            'names': ['f0'],
            'formats': [value],
            'offsets': [0],
            'itemsize': value.itemsize
        }
    else:
        fields = value.fields
        names = value.names
        field_spec = {
            'names': names,
            'formats': [fields[name][0] for name in names],
            'offsets': [fields[name][1] for name in names],
            'itemsize': value.itemsize
        }

    field_spec['itemsize'] += padding
    return dtype(field_spec)

def _prod(a):
    p = 1
    for x in a:
        p *= x
    return p

def _gcd(a, b):
    """Calculate the greatest common divisor of a and b"""
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError('Can only find greatest common divisor of '
                         f'finite arguments, found "{a}" and "{b}"')
    while b:
        a, b = b, a % b
    return a

def _lcm(a, b):
    return a // _gcd(a, b) * b

def array_ufunc_errmsg_formatter(dummy, ufunc, method, *inputs, **kwargs):
    """ Format the error message for when __array_ufunc__ gives up. """
    args_string = ', '.join([f'{arg!r}' for arg in inputs] +
                            [f'{k}={v!r}'
                             for k, v in kwargs.items()])
    args = inputs + kwargs.get('out', ())
    types_string = ', '.join(repr(type(arg).__name__) for arg in args)
    return ('operand type(s) all returned NotImplemented from '
            f'__array_ufunc__({ufunc!r}, {method!r}, {args_string}): {types_string}'
            )


def array_function_errmsg_formatter(public_api, types):
    """ Format the error message for when __array_ufunc__ gives up. """
    func_name = f'{public_api.__module__}.{public_api.__name__}'
    return (f"no implementation found for '{func_name}' on types that implement "
            f'__array_function__: {list(types)}')


def _ufunc_doc_signature_formatter(ufunc):
    """
    Builds a signature string which resembles PEP 457

    This is used to construct the first line of the docstring
    """

    # input arguments are simple
    if ufunc.nin == 1:
        in_args = 'x'
    else:
        in_args = ', '.join(f'x{i + 1}' for i in range(ufunc.nin))

    # output arguments are both keyword or positional
    if ufunc.nout == 0:
        out_args = ', /, out=()'
    elif ufunc.nout == 1:
        out_args = ', /, out=None'
    else:
        out_args = '[, {positional}], / [, out={default}]'.format(
            positional=', '.join(
                f'out{i + 1}' for i in range(ufunc.nout)),
            default=repr((None,) * ufunc.nout)
        )

    # keyword only args depend on whether this is a gufunc
    kwargs = (
        ", casting='same_kind'"
        ", order='K'"
        ", dtype=None"
        ", subok=True"
    )

    # NOTE: gufuncs may or may not support the `axis` parameter
    if ufunc.signature is None:
        kwargs = f", where=True{kwargs}[, signature]"
    else:
        kwargs += "[, signature, axes, axis]"

    # join all the parts together
    return f'{ufunc.__name__}({in_args}{out_args}, *{kwargs})'


def npy_ctypes_check(cls):
    # determine if a class comes from ctypes, in order to work around
    # a bug in the buffer protocol for those objects, bpo-10746
    try:
        # ctypes class are new-style, so have an __mro__. This probably fails
        # for ctypes classes with multiple inheritance.
        if IS_PYPY:
            # (..., _ctypes.basics._CData, Bufferable, object)
            ctype_base = cls.__mro__[-3]
        else:
            # # (..., _ctypes._CData, object)
            ctype_base = cls.__mro__[-2]
        # right now, they're part of the _ctypes module
        return '_ctypes' in ctype_base.__module__
    except Exception:
        return False

# used to handle the _NoValue default argument for na_object
# in the C implementation of the __reduce__ method for stringdtype
def _convert_to_stringdtype_kwargs(coerce, na_object=_NoValue):
    if na_object is _NoValue:
        return StringDType(coerce=coerce)
    return StringDType(coerce=coerce, na_object=na_object)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\shared_docs.py ===
from __future__ import annotations

_shared_docs: dict[str, str] = {}

_shared_docs[
    "aggregate"
] = """
Aggregate using one or more operations over the specified axis.

Parameters
----------
func : function, str, list or dict
    Function to use for aggregating the data. If a function, must either
    work when passed a {klass} or when passed to {klass}.apply.

    Accepted combinations are:

    - function
    - string function name
    - list of functions and/or function names, e.g. ``[np.sum, 'mean']``
    - dict of axis labels -> functions, function names or list of such.
{axis}
*args
    Positional arguments to pass to `func`.
**kwargs
    Keyword arguments to pass to `func`.

Returns
-------
scalar, Series or DataFrame

    The return can be:

    * scalar : when Series.agg is called with single function
    * Series : when DataFrame.agg is called with a single function
    * DataFrame : when DataFrame.agg is called with several functions
{see_also}
Notes
-----
The aggregation operations are always performed over an axis, either the
index (default) or the column axis. This behavior is different from
`numpy` aggregation functions (`mean`, `median`, `prod`, `sum`, `std`,
`var`), where the default is to compute the aggregation of the flattened
array, e.g., ``numpy.mean(arr_2d)`` as opposed to
``numpy.mean(arr_2d, axis=0)``.

`agg` is an alias for `aggregate`. Use the alias.

Functions that mutate the passed object can produce unexpected
behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
for more details.

A passed user-defined-function will be passed a Series for evaluation.
{examples}"""

_shared_docs[
    "compare"
] = """
Compare to another {klass} and show the differences.

Parameters
----------
other : {klass}
    Object to compare with.

align_axis : {{0 or 'index', 1 or 'columns'}}, default 1
    Determine which axis to align the comparison on.

    * 0, or 'index' : Resulting differences are stacked vertically
        with rows drawn alternately from self and other.
    * 1, or 'columns' : Resulting differences are aligned horizontally
        with columns drawn alternately from self and other.

keep_shape : bool, default False
    If true, all rows and columns are kept.
    Otherwise, only the ones with different values are kept.

keep_equal : bool, default False
    If true, the result keeps values that are equal.
    Otherwise, equal values are shown as NaNs.

result_names : tuple, default ('self', 'other')
    Set the dataframes names in the comparison.

    .. versionadded:: 1.5.0
"""

_shared_docs[
    "groupby"
] = """
Group %(klass)s using a mapper or by a Series of columns.

A groupby operation involves some combination of splitting the
object, applying a function, and combining the results. This can be
used to group large amounts of data and compute operations on these
groups.

Parameters
----------
by : mapping, function, label, pd.Grouper or list of such
    Used to determine the groups for the groupby.
    If ``by`` is a function, it's called on each value of the object's
    index. If a dict or Series is passed, the Series or dict VALUES
    will be used to determine the groups (the Series' values are first
    aligned; see ``.align()`` method). If a list or ndarray of length
    equal to the selected axis is passed (see the `groupby user guide
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#splitting-an-object-into-groups>`_),
    the values are used as-is to determine the groups. A label or list
    of labels may be passed to group by the columns in ``self``.
    Notice that a tuple is interpreted as a (single) key.
axis : {0 or 'index', 1 or 'columns'}, default 0
    Split along rows (0) or columns (1). For `Series` this parameter
    is unused and defaults to 0.

    .. deprecated:: 2.1.0

        Will be removed and behave like axis=0 in a future version.
        For ``axis=1``, do ``frame.T.groupby(...)`` instead.

level : int, level name, or sequence of such, default None
    If the axis is a MultiIndex (hierarchical), group by a particular
    level or levels. Do not specify both ``by`` and ``level``.
as_index : bool, default True
    Return object with group labels as the
    index. Only relevant for DataFrame input. as_index=False is
    effectively "SQL-style" grouped output. This argument has no effect
    on filtrations (see the `filtrations in the user guide
    <https://pandas.pydata.org/docs/dev/user_guide/groupby.html#filtration>`_),
    such as ``head()``, ``tail()``, ``nth()`` and in transformations
    (see the `transformations in the user guide
    <https://pandas.pydata.org/docs/dev/user_guide/groupby.html#transformation>`_).
sort : bool, default True
    Sort group keys. Get better performance by turning this off.
    Note this does not influence the order of observations within each
    group. Groupby preserves the order of rows within each group. If False,
    the groups will appear in the same order as they did in the original DataFrame.
    This argument has no effect on filtrations (see the `filtrations in the user guide
    <https://pandas.pydata.org/docs/dev/user_guide/groupby.html#filtration>`_),
    such as ``head()``, ``tail()``, ``nth()`` and in transformations
    (see the `transformations in the user guide
    <https://pandas.pydata.org/docs/dev/user_guide/groupby.html#transformation>`_).

    .. versionchanged:: 2.0.0

        Specifying ``sort=False`` with an ordered categorical grouper will no
        longer sort the values.

group_keys : bool, default True
    When calling apply and the ``by`` argument produces a like-indexed
    (i.e. :ref:`a transform <groupby.transform>`) result, add group keys to
    index to identify pieces. By default group keys are not included
    when the result's index (and column) labels match the inputs, and
    are included otherwise.

    .. versionchanged:: 1.5.0

       Warns that ``group_keys`` will no longer be ignored when the
       result from ``apply`` is a like-indexed Series or DataFrame.
       Specify ``group_keys`` explicitly to include the group keys or
       not.

    .. versionchanged:: 2.0.0

       ``group_keys`` now defaults to ``True``.

observed : bool, default False
    This only applies if any of the groupers are Categoricals.
    If True: only show observed values for categorical groupers.
    If False: show all values for categorical groupers.

    .. deprecated:: 2.1.0

        The default value will change to True in a future version of pandas.

dropna : bool, default True
    If True, and if group keys contain NA values, NA values together
    with row/column will be dropped.
    If False, NA values will also be treated as the key in groups.

Returns
-------
pandas.api.typing.%(klass)sGroupBy
    Returns a groupby object that contains information about the groups.

See Also
--------
resample : Convenience method for frequency conversion and resampling
    of time series.

Notes
-----
See the `user guide
<https://pandas.pydata.org/pandas-docs/stable/groupby.html>`__ for more
detailed usage and examples, including splitting an object into groups,
iterating through groups, selecting a group, aggregation, and more.
"""

_shared_docs[
    "melt"
] = """
Unpivot a DataFrame from wide to long format, optionally leaving identifiers set.

This function is useful to massage a DataFrame into a format where one
or more columns are identifier variables (`id_vars`), while all other
columns, considered measured variables (`value_vars`), are "unpivoted" to
the row axis, leaving just two non-identifier columns, 'variable' and
'value'.

Parameters
----------
id_vars : scalar, tuple, list, or ndarray, optional
    Column(s) to use as identifier variables.
value_vars : scalar, tuple, list, or ndarray, optional
    Column(s) to unpivot. If not specified, uses all columns that
    are not set as `id_vars`.
var_name : scalar, default None
    Name to use for the 'variable' column. If None it uses
    ``frame.columns.name`` or 'variable'.
value_name : scalar, default 'value'
    Name to use for the 'value' column, can't be an existing column label.
col_level : scalar, optional
    If columns are a MultiIndex then use this level to melt.
ignore_index : bool, default True
    If True, original index is ignored. If False, the original index is retained.
    Index labels will be repeated as necessary.

Returns
-------
DataFrame
    Unpivoted DataFrame.

See Also
--------
%(other)s : Identical method.
pivot_table : Create a spreadsheet-style pivot table as a DataFrame.
DataFrame.pivot : Return reshaped DataFrame organized
    by given index / column values.
DataFrame.explode : Explode a DataFrame from list-like
        columns to long format.

Notes
-----
Reference :ref:`the user guide <reshaping.melt>` for more examples.

Examples
--------
>>> df = pd.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'},
...                    'B': {0: 1, 1: 3, 2: 5},
...                    'C': {0: 2, 1: 4, 2: 6}})
>>> df
   A  B  C
0  a  1  2
1  b  3  4
2  c  5  6

>>> %(caller)sid_vars=['A'], value_vars=['B'])
   A variable  value
0  a        B      1
1  b        B      3
2  c        B      5

>>> %(caller)sid_vars=['A'], value_vars=['B', 'C'])
   A variable  value
0  a        B      1
1  b        B      3
2  c        B      5
3  a        C      2
4  b        C      4
5  c        C      6

The names of 'variable' and 'value' columns can be customized:

>>> %(caller)sid_vars=['A'], value_vars=['B'],
...         var_name='myVarname', value_name='myValname')
   A myVarname  myValname
0  a         B          1
1  b         B          3
2  c         B          5

Original index values can be kept around:

>>> %(caller)sid_vars=['A'], value_vars=['B', 'C'], ignore_index=False)
   A variable  value
0  a        B      1
1  b        B      3
2  c        B      5
0  a        C      2
1  b        C      4
2  c        C      6

If you have multi-index columns:

>>> df.columns = [list('ABC'), list('DEF')]
>>> df
   A  B  C
   D  E  F
0  a  1  2
1  b  3  4
2  c  5  6

>>> %(caller)scol_level=0, id_vars=['A'], value_vars=['B'])
   A variable  value
0  a        B      1
1  b        B      3
2  c        B      5

>>> %(caller)sid_vars=[('A', 'D')], value_vars=[('B', 'E')])
  (A, D) variable_0 variable_1  value
0      a          B          E      1
1      b          B          E      3
2      c          B          E      5
"""

_shared_docs[
    "transform"
] = """
Call ``func`` on self producing a {klass} with the same axis shape as self.

Parameters
----------
func : function, str, list-like or dict-like
    Function to use for transforming the data. If a function, must either
    work when passed a {klass} or when passed to {klass}.apply. If func
    is both list-like and dict-like, dict-like behavior takes precedence.

    Accepted combinations are:

    - function
    - string function name
    - list-like of functions and/or function names, e.g. ``[np.exp, 'sqrt']``
    - dict-like of axis labels -> functions, function names or list-like of such.
{axis}
*args
    Positional arguments to pass to `func`.
**kwargs
    Keyword arguments to pass to `func`.

Returns
-------
{klass}
    A {klass} that must have the same length as self.

Raises
------
ValueError : If the returned {klass} has a different length than self.

See Also
--------
{klass}.agg : Only perform aggregating type operations.
{klass}.apply : Invoke function on a {klass}.

Notes
-----
Functions that mutate the passed object can produce unexpected
behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
for more details.

Examples
--------
>>> df = pd.DataFrame({{'A': range(3), 'B': range(1, 4)}})
>>> df
   A  B
0  0  1
1  1  2
2  2  3
>>> df.transform(lambda x: x + 1)
   A  B
0  1  2
1  2  3
2  3  4

Even though the resulting {klass} must have the same length as the
input {klass}, it is possible to provide several input functions:

>>> s = pd.Series(range(3))
>>> s
0    0
1    1
2    2
dtype: int64
>>> s.transform([np.sqrt, np.exp])
       sqrt        exp
0  0.000000   1.000000
1  1.000000   2.718282
2  1.414214   7.389056

You can call transform on a GroupBy object:

>>> df = pd.DataFrame({{
...     "Date": [
...         "2015-05-08", "2015-05-07", "2015-05-06", "2015-05-05",
...         "2015-05-08", "2015-05-07", "2015-05-06", "2015-05-05"],
...     "Data": [5, 8, 6, 1, 50, 100, 60, 120],
... }})
>>> df
         Date  Data
0  2015-05-08     5
1  2015-05-07     8
2  2015-05-06     6
3  2015-05-05     1
4  2015-05-08    50
5  2015-05-07   100
6  2015-05-06    60
7  2015-05-05   120
>>> df.groupby('Date')['Data'].transform('sum')
0     55
1    108
2     66
3    121
4     55
5    108
6     66
7    121
Name: Data, dtype: int64

>>> df = pd.DataFrame({{
...     "c": [1, 1, 1, 2, 2, 2, 2],
...     "type": ["m", "n", "o", "m", "m", "n", "n"]
... }})
>>> df
   c type
0  1    m
1  1    n
2  1    o
3  2    m
4  2    m
5  2    n
6  2    n
>>> df['size'] = df.groupby('c')['type'].transform(len)
>>> df
   c type size
0  1    m    3
1  1    n    3
2  1    o    3
3  2    m    4
4  2    m    4
5  2    n    4
6  2    n    4
"""

_shared_docs[
    "storage_options"
] = """storage_options : dict, optional
    Extra options that make sense for a particular storage connection, e.g.
    host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
    are forwarded to ``urllib.request.Request`` as header options. For other
    URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
    forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
    details, and for more examples on storage options refer `here
    <https://pandas.pydata.org/docs/user_guide/io.html?
    highlight=storage_options#reading-writing-remote-files>`_."""

_shared_docs[
    "compression_options"
] = """compression : str or dict, default 'infer'
    For on-the-fly compression of the output data. If 'infer' and '%s' is
    path-like, then detect compression from the following extensions: '.gz',
    '.bz2', '.zip', '.xz', '.zst', '.tar', '.tar.gz', '.tar.xz' or '.tar.bz2'
    (otherwise no compression).
    Set to ``None`` for no compression.
    Can also be a dict with key ``'method'`` set
    to one of {``'zip'``, ``'gzip'``, ``'bz2'``, ``'zstd'``, ``'xz'``, ``'tar'``} and
    other key-value pairs are forwarded to
    ``zipfile.ZipFile``, ``gzip.GzipFile``,
    ``bz2.BZ2File``, ``zstandard.ZstdCompressor``, ``lzma.LZMAFile`` or
    ``tarfile.TarFile``, respectively.
    As an example, the following could be passed for faster compression and to create
    a reproducible gzip archive:
    ``compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1}``.

    .. versionadded:: 1.5.0
        Added support for `.tar` files."""

_shared_docs[
    "decompression_options"
] = """compression : str or dict, default 'infer'
    For on-the-fly decompression of on-disk data. If 'infer' and '%s' is
    path-like, then detect compression from the following extensions: '.gz',
    '.bz2', '.zip', '.xz', '.zst', '.tar', '.tar.gz', '.tar.xz' or '.tar.bz2'
    (otherwise no compression).
    If using 'zip' or 'tar', the ZIP file must contain only one data file to be read in.
    Set to ``None`` for no decompression.
    Can also be a dict with key ``'method'`` set
    to one of {``'zip'``, ``'gzip'``, ``'bz2'``, ``'zstd'``, ``'xz'``, ``'tar'``} and
    other key-value pairs are forwarded to
    ``zipfile.ZipFile``, ``gzip.GzipFile``,
    ``bz2.BZ2File``, ``zstandard.ZstdDecompressor``, ``lzma.LZMAFile`` or
    ``tarfile.TarFile``, respectively.
    As an example, the following could be passed for Zstandard decompression using a
    custom compression dictionary:
    ``compression={'method': 'zstd', 'dict_data': my_compression_dict}``.

    .. versionadded:: 1.5.0
        Added support for `.tar` files."""

_shared_docs[
    "replace"
] = """
    Replace values given in `to_replace` with `value`.

    Values of the {klass} are replaced with other values dynamically.
    This differs from updating with ``.loc`` or ``.iloc``, which require
    you to specify a location to update with some value.

    Parameters
    ----------
    to_replace : str, regex, list, dict, Series, int, float, or None
        How to find the values that will be replaced.

        * numeric, str or regex:

            - numeric: numeric values equal to `to_replace` will be
              replaced with `value`
            - str: string exactly matching `to_replace` will be replaced
              with `value`
            - regex: regexs matching `to_replace` will be replaced with
              `value`

        * list of str, regex, or numeric:

            - First, if `to_replace` and `value` are both lists, they
              **must** be the same length.
            - Second, if ``regex=True`` then all of the strings in **both**
              lists will be interpreted as regexs otherwise they will match
              directly. This doesn't matter much for `value` since there
              are only a few possible substitution regexes you can use.
            - str, regex and numeric rules apply as above.

        * dict:

            - Dicts can be used to specify different replacement values
              for different existing values. For example,
              ``{{'a': 'b', 'y': 'z'}}`` replaces the value 'a' with 'b' and
              'y' with 'z'. To use a dict in this way, the optional `value`
              parameter should not be given.
            - For a DataFrame a dict can specify that different values
              should be replaced in different columns. For example,
              ``{{'a': 1, 'b': 'z'}}`` looks for the value 1 in column 'a'
              and the value 'z' in column 'b' and replaces these values
              with whatever is specified in `value`. The `value` parameter
              should not be ``None`` in this case. You can treat this as a
              special case of passing two lists except that you are
              specifying the column to search in.
            - For a DataFrame nested dictionaries, e.g.,
              ``{{'a': {{'b': np.nan}}}}``, are read as follows: look in column
              'a' for the value 'b' and replace it with NaN. The optional `value`
              parameter should not be specified to use a nested dict in this
              way. You can nest regular expressions as well. Note that
              column names (the top-level dictionary keys in a nested
              dictionary) **cannot** be regular expressions.

        * None:

            - This means that the `regex` argument must be a string,
              compiled regular expression, or list, dict, ndarray or
              Series of such elements. If `value` is also ``None`` then
              this **must** be a nested dictionary or Series.

        See the examples section for examples of each of these.
    value : scalar, dict, list, str, regex, default None
        Value to replace any values matching `to_replace` with.
        For a DataFrame a dict of values can be used to specify which
        value to use for each column (columns not in the dict will not be
        filled). Regular expressions, strings and lists or dicts of such
        objects are also allowed.
    {inplace}
    limit : int, default None
        Maximum size gap to forward or backward fill.

        .. deprecated:: 2.1.0
    regex : bool or same types as `to_replace`, default False
        Whether to interpret `to_replace` and/or `value` as regular
        expressions. Alternatively, this could be a regular expression or a
        list, dict, or array of regular expressions in which case
        `to_replace` must be ``None``.
    method : {{'pad', 'ffill', 'bfill'}}
        The method to use when for replacement, when `to_replace` is a
        scalar, list or tuple and `value` is ``None``.

        .. deprecated:: 2.1.0

    Returns
    -------
    {klass}
        Object after replacement.

    Raises
    ------
    AssertionError
        * If `regex` is not a ``bool`` and `to_replace` is not
          ``None``.

    TypeError
        * If `to_replace` is not a scalar, array-like, ``dict``, or ``None``
        * If `to_replace` is a ``dict`` and `value` is not a ``list``,
          ``dict``, ``ndarray``, or ``Series``
        * If `to_replace` is ``None`` and `regex` is not compilable
          into a regular expression or is a list, dict, ndarray, or
          Series.
        * When replacing multiple ``bool`` or ``datetime64`` objects and
          the arguments to `to_replace` does not match the type of the
          value being replaced

    ValueError
        * If a ``list`` or an ``ndarray`` is passed to `to_replace` and
          `value` but they are not the same length.

    See Also
    --------
    Series.fillna : Fill NA values.
    DataFrame.fillna : Fill NA values.
    Series.where : Replace values based on boolean condition.
    DataFrame.where : Replace values based on boolean condition.
    DataFrame.map: Apply a function to a Dataframe elementwise.
    Series.map: Map values of Series according to an input mapping or function.
    Series.str.replace : Simple string replacement.

    Notes
    -----
    * Regex substitution is performed under the hood with ``re.sub``. The
      rules for substitution for ``re.sub`` are the same.
    * Regular expressions will only substitute on strings, meaning you
      cannot provide, for example, a regular expression matching floating
      point numbers and expect the columns in your frame that have a
      numeric dtype to be matched. However, if those floating point
      numbers *are* strings, then you can do this.
    * This method has *a lot* of options. You are encouraged to experiment
      and play with this method to gain intuition about how it works.
    * When dict is used as the `to_replace` value, it is like
      key(s) in the dict are the to_replace part and
      value(s) in the dict are the value parameter.

    Examples
    --------

    **Scalar `to_replace` and `value`**

    >>> s = pd.Series([1, 2, 3, 4, 5])
    >>> s.replace(1, 5)
    0    5
    1    2
    2    3
    3    4
    4    5
    dtype: int64

    >>> df = pd.DataFrame({{'A': [0, 1, 2, 3, 4],
    ...                    'B': [5, 6, 7, 8, 9],
    ...                    'C': ['a', 'b', 'c', 'd', 'e']}})
    >>> df.replace(0, 5)
        A  B  C
    0  5  5  a
    1  1  6  b
    2  2  7  c
    3  3  8  d
    4  4  9  e

    **List-like `to_replace`**

    >>> df.replace([0, 1, 2, 3], 4)
        A  B  C
    0  4  5  a
    1  4  6  b
    2  4  7  c
    3  4  8  d
    4  4  9  e

    >>> df.replace([0, 1, 2, 3], [4, 3, 2, 1])
        A  B  C
    0  4  5  a
    1  3  6  b
    2  2  7  c
    3  1  8  d
    4  4  9  e

    >>> s.replace([1, 2], method='bfill')
    0    3
    1    3
    2    3
    3    4
    4    5
    dtype: int64

    **dict-like `to_replace`**

    >>> df.replace({{0: 10, 1: 100}})
            A  B  C
    0   10  5  a
    1  100  6  b
    2    2  7  c
    3    3  8  d
    4    4  9  e

    >>> df.replace({{'A': 0, 'B': 5}}, 100)
            A    B  C
    0  100  100  a
    1    1    6  b
    2    2    7  c
    3    3    8  d
    4    4    9  e

    >>> df.replace({{'A': {{0: 100, 4: 400}}}})
            A  B  C
    0  100  5  a
    1    1  6  b
    2    2  7  c
    3    3  8  d
    4  400  9  e

    **Regular expression `to_replace`**

    >>> df = pd.DataFrame({{'A': ['bat', 'foo', 'bait'],
    ...                    'B': ['abc', 'bar', 'xyz']}})
    >>> df.replace(to_replace=r'^ba.$', value='new', regex=True)
            A    B
    0   new  abc
    1   foo  new
    2  bait  xyz

    >>> df.replace({{'A': r'^ba.$'}}, {{'A': 'new'}}, regex=True)
            A    B
    0   new  abc
    1   foo  bar
    2  bait  xyz

    >>> df.replace(regex=r'^ba.$', value='new')
            A    B
    0   new  abc
    1   foo  new
    2  bait  xyz

    >>> df.replace(regex={{r'^ba.$': 'new', 'foo': 'xyz'}})
            A    B
    0   new  abc
    1   xyz  new
    2  bait  xyz

    >>> df.replace(regex=[r'^ba.$', 'foo'], value='new')
            A    B
    0   new  abc
    1   new  new
    2  bait  xyz

    Compare the behavior of ``s.replace({{'a': None}})`` and
    ``s.replace('a', None)`` to understand the peculiarities
    of the `to_replace` parameter:

    >>> s = pd.Series([10, 'a', 'a', 'b', 'a'])

    When one uses a dict as the `to_replace` value, it is like the
    value(s) in the dict are equal to the `value` parameter.
    ``s.replace({{'a': None}})`` is equivalent to
    ``s.replace(to_replace={{'a': None}}, value=None, method=None)``:

    >>> s.replace({{'a': None}})
    0      10
    1    None
    2    None
    3       b
    4    None
    dtype: object

    When ``value`` is not explicitly passed and `to_replace` is a scalar, list
    or tuple, `replace` uses the method parameter (default 'pad') to do the
    replacement. So this is why the 'a' values are being replaced by 10
    in rows 1 and 2 and 'b' in row 4 in this case.

    >>> s.replace('a')
    0    10
    1    10
    2    10
    3     b
    4     b
    dtype: object

        .. deprecated:: 2.1.0
            The 'method' parameter and padding behavior are deprecated.

    On the other hand, if ``None`` is explicitly passed for ``value``, it will
    be respected:

    >>> s.replace('a', None)
    0      10
    1    None
    2    None
    3       b
    4    None
    dtype: object

        .. versionchanged:: 1.4.0
            Previously the explicit ``None`` was silently ignored.

    When ``regex=True``, ``value`` is not ``None`` and `to_replace` is a string,
    the replacement will be applied in all columns of the DataFrame.

    >>> df = pd.DataFrame({{'A': [0, 1, 2, 3, 4],
    ...                    'B': ['a', 'b', 'c', 'd', 'e'],
    ...                    'C': ['f', 'g', 'h', 'i', 'j']}})

    >>> df.replace(to_replace='^[a-g]', value='e', regex=True)
        A  B  C
    0  0  e  e
    1  1  e  e
    2  2  e  h
    3  3  e  i
    4  4  e  j

    If ``value`` is not ``None`` and `to_replace` is a dictionary, the dictionary
    keys will be the DataFrame columns that the replacement will be applied.

    >>> df.replace(to_replace={{'B': '^[a-c]', 'C': '^[h-j]'}}, value='e', regex=True)
        A  B  C
    0  0  e  f
    1  1  e  g
    2  2  e  e
    3  3  d  e
    4  4  e  e
"""

_shared_docs[
    "idxmin"
] = """
    Return index of first occurrence of minimum over requested axis.

    NA/null values are excluded.

    Parameters
    ----------
    axis : {{0 or 'index', 1 or 'columns'}}, default 0
        The axis to use. 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
    skipna : bool, default True
        Exclude NA/null values. If an entire row/column is NA, the result
        will be NA.
    numeric_only : bool, default {numeric_only_default}
        Include only `float`, `int` or `boolean` data.

        .. versionadded:: 1.5.0

    Returns
    -------
    Series
        Indexes of minima along the specified axis.

    Raises
    ------
    ValueError
        * If the row/column is empty

    See Also
    --------
    Series.idxmin : Return index of the minimum element.

    Notes
    -----
    This method is the DataFrame version of ``ndarray.argmin``.

    Examples
    --------
    Consider a dataset containing food consumption in Argentina.

    >>> df = pd.DataFrame({{'consumption': [10.51, 103.11, 55.48],
    ...                     'co2_emissions': [37.2, 19.66, 1712]}},
    ...                   index=['Pork', 'Wheat Products', 'Beef'])

    >>> df
                    consumption  co2_emissions
    Pork                  10.51         37.20
    Wheat Products       103.11         19.66
    Beef                  55.48       1712.00

    By default, it returns the index for the minimum value in each column.

    >>> df.idxmin()
    consumption                Pork
    co2_emissions    Wheat Products
    dtype: object

    To return the index for the minimum value in each row, use ``axis="columns"``.

    >>> df.idxmin(axis="columns")
    Pork                consumption
    Wheat Products    co2_emissions
    Beef                consumption
    dtype: object
"""

_shared_docs[
    "idxmax"
] = """
    Return index of first occurrence of maximum over requested axis.

    NA/null values are excluded.

    Parameters
    ----------
    axis : {{0 or 'index', 1 or 'columns'}}, default 0
        The axis to use. 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
    skipna : bool, default True
        Exclude NA/null values. If an entire row/column is NA, the result
        will be NA.
    numeric_only : bool, default {numeric_only_default}
        Include only `float`, `int` or `boolean` data.

        .. versionadded:: 1.5.0

    Returns
    -------
    Series
        Indexes of maxima along the specified axis.

    Raises
    ------
    ValueError
        * If the row/column is empty

    See Also
    --------
    Series.idxmax : Return index of the maximum element.

    Notes
    -----
    This method is the DataFrame version of ``ndarray.argmax``.

    Examples
    --------
    Consider a dataset containing food consumption in Argentina.

    >>> df = pd.DataFrame({{'consumption': [10.51, 103.11, 55.48],
    ...                     'co2_emissions': [37.2, 19.66, 1712]}},
    ...                   index=['Pork', 'Wheat Products', 'Beef'])

    >>> df
                    consumption  co2_emissions
    Pork                  10.51         37.20
    Wheat Products       103.11         19.66
    Beef                  55.48       1712.00

    By default, it returns the index for the maximum value in each column.

    >>> df.idxmax()
    consumption     Wheat Products
    co2_emissions             Beef
    dtype: object

    To return the index for the maximum value in each row, use ``axis="columns"``.

    >>> df.idxmax(axis="columns")
    Pork              co2_emissions
    Wheat Products     consumption
    Beef              co2_emissions
    dtype: object
"""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\matrices\_dfm.py ===
#
# sympy.polys.matrices.dfm
#
# This modules defines the DFM class which is a wrapper for dense flint
# matrices as found in python-flint.
#
# As of python-flint 0.4.1 matrices over the following domains can be supported
# by python-flint:
#
#   ZZ: flint.fmpz_mat
#   QQ: flint.fmpq_mat
#   GF(p): flint.nmod_mat (p prime and p < ~2**62)
#
# The underlying flint library has many more domains, but these are not yet
# supported by python-flint.
#
# The DFM class is a wrapper for the flint matrices and provides a common
# interface for all supported domains that is interchangeable with the DDM
# and SDM classes so that DomainMatrix can be used with any as its internal
# matrix representation.
#

# TODO:
#
# Implement the following methods that are provided by python-flint:
#
# - hnf (Hermite normal form)
# - snf (Smith normal form)
# - minpoly
# - is_hnf
# - is_snf
# - rank
#
# The other types DDM and SDM do not have these methods and the algorithms
# for hnf, snf and rank are already implemented. Algorithms for minpoly,
# is_hnf and is_snf would need to be added.
#
# Add more methods to python-flint to expose more of Flint's functionality
# and also to make some of the above methods simpler or more efficient e.g.
# slicing, fancy indexing etc.

from sympy.external.gmpy import GROUND_TYPES
from sympy.external.importtools import import_module
from sympy.utilities.decorator import doctest_depends_on

from sympy.polys.domains import ZZ, QQ

from .exceptions import (
    DMBadInputError,
    DMDomainError,
    DMNonSquareMatrixError,
    DMNonInvertibleMatrixError,
    DMRankError,
    DMShapeError,
    DMValueError,
)


if GROUND_TYPES != 'flint':
    __doctest_skip__ = ['*']


flint = import_module('flint')


__all__ = ['DFM']


@doctest_depends_on(ground_types=['flint'])
class DFM:
    """
    Dense FLINT matrix. This class is a wrapper for matrices from python-flint.

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.matrices.dfm import DFM
    >>> dfm = DFM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    >>> dfm
    [[1, 2], [3, 4]]
    >>> dfm.rep
    [1, 2]
    [3, 4]
    >>> type(dfm.rep)  # doctest: +SKIP
    <class 'flint._flint.fmpz_mat'>

    Usually, the DFM class is not instantiated directly, but is created as the
    internal representation of :class:`~.DomainMatrix`. When
    `SYMPY_GROUND_TYPES` is set to `flint` and `python-flint` is installed, the
    :class:`DFM` class is used automatically as the internal representation of
    :class:`~.DomainMatrix` in dense format if the domain is supported by
    python-flint.

    >>> from sympy.polys.matrices.domainmatrix import DM
    >>> dM = DM([[1, 2], [3, 4]], ZZ)
    >>> dM.rep
    [[1, 2], [3, 4]]

    A :class:`~.DomainMatrix` can be converted to :class:`DFM` by calling the
    :meth:`to_dfm` method:

    >>> dM.to_dfm()
    [[1, 2], [3, 4]]

    """

    fmt = 'dense'
    is_DFM = True
    is_DDM = False

    def __new__(cls, rowslist, shape, domain):
        """Construct from a nested list."""
        flint_mat = cls._get_flint_func(domain)

        if 0 not in shape:
            try:
                rep = flint_mat(rowslist)
            except (ValueError, TypeError):
                raise DMBadInputError(f"Input should be a list of list of {domain}")
        else:
            rep = flint_mat(*shape)

        return cls._new(rep, shape, domain)

    @classmethod
    def _new(cls, rep, shape, domain):
        """Internal constructor from a flint matrix."""
        cls._check(rep, shape, domain)
        obj = object.__new__(cls)
        obj.rep = rep
        obj.shape = obj.rows, obj.cols = shape
        obj.domain = domain
        return obj

    def _new_rep(self, rep):
        """Create a new DFM with the same shape and domain but a new rep."""
        return self._new(rep, self.shape, self.domain)

    @classmethod
    def _check(cls, rep, shape, domain):
        repshape = (rep.nrows(), rep.ncols())
        if repshape != shape:
            raise DMBadInputError("Shape of rep does not match shape of DFM")
        if domain == ZZ and not isinstance(rep, flint.fmpz_mat):
            raise RuntimeError("Rep is not a flint.fmpz_mat")
        elif domain == QQ and not isinstance(rep, flint.fmpq_mat):
            raise RuntimeError("Rep is not a flint.fmpq_mat")
        elif domain.is_FF and not isinstance(rep, (flint.fmpz_mod_mat, flint.nmod_mat)):
            raise RuntimeError("Rep is not a flint.fmpz_mod_mat or flint.nmod_mat")
        elif domain not in (ZZ, QQ) and not domain.is_FF:
            raise NotImplementedError("Only ZZ and QQ are supported by DFM")

    @classmethod
    def _supports_domain(cls, domain):
        """Return True if the given domain is supported by DFM."""
        return domain in (ZZ, QQ) or domain.is_FF and domain._is_flint

    @classmethod
    def _get_flint_func(cls, domain):
        """Return the flint matrix class for the given domain."""
        if domain == ZZ:
            return flint.fmpz_mat
        elif domain == QQ:
            return flint.fmpq_mat
        elif domain.is_FF:
            c = domain.characteristic()
            if isinstance(domain.one, flint.nmod):
                _cls = flint.nmod_mat
                def _func(*e):
                    if len(e) == 1 and isinstance(e[0], flint.nmod_mat):
                        return _cls(e[0])
                    else:
                        return _cls(*e, c)
            else:
                m = flint.fmpz_mod_ctx(c)
                _func = lambda *e: flint.fmpz_mod_mat(*e, m)
            return _func
        else:
            raise NotImplementedError("Only ZZ and QQ are supported by DFM")

    @property
    def _func(self):
        """Callable to create a flint matrix of the same domain."""
        return self._get_flint_func(self.domain)

    def __str__(self):
        """Return ``str(self)``."""
        return str(self.to_ddm())

    def __repr__(self):
        """Return ``repr(self)``."""
        return f'DFM{repr(self.to_ddm())[3:]}'

    def __eq__(self, other):
        """Return ``self == other``."""
        if not isinstance(other, DFM):
            return NotImplemented
        # Compare domains first because we do *not* want matrices with
        # different domains to be equal but e.g. a flint fmpz_mat and fmpq_mat
        # with the same entries will compare equal.
        return self.domain == other.domain and self.rep == other.rep

    @classmethod
    def from_list(cls, rowslist, shape, domain):
        """Construct from a nested list."""
        return cls(rowslist, shape, domain)

    def to_list(self):
        """Convert to a nested list."""
        return self.rep.tolist()

    def copy(self):
        """Return a copy of self."""
        return self._new_rep(self._func(self.rep))

    def to_ddm(self):
        """Convert to a DDM."""
        return DDM.from_list(self.to_list(), self.shape, self.domain)

    def to_sdm(self):
        """Convert to a SDM."""
        return SDM.from_list(self.to_list(), self.shape, self.domain)

    def to_dfm(self):
        """Return self."""
        return self

    def to_dfm_or_ddm(self):
        """
        Convert to a :class:`DFM`.

        This :class:`DFM` method exists to parallel the :class:`~.DDM` and
        :class:`~.SDM` methods. For :class:`DFM` it will always return self.

        See Also
        ========

        to_ddm
        to_sdm
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dfm_or_ddm
        """
        return self

    @classmethod
    def from_ddm(cls, ddm):
        """Convert from a DDM."""
        return cls.from_list(ddm.to_list(), ddm.shape, ddm.domain)

    @classmethod
    def from_list_flat(cls, elements, shape, domain):
        """Inverse of :meth:`to_list_flat`."""
        func = cls._get_flint_func(domain)
        try:
            rep = func(*shape, elements)
        except ValueError:
            raise DMBadInputError(f"Incorrect number of elements for shape {shape}")
        except TypeError:
            raise DMBadInputError(f"Input should be a list of {domain}")
        return cls(rep, shape, domain)

    def to_list_flat(self):
        """Convert to a flat list."""
        return self.rep.entries()

    def to_flat_nz(self):
        """Convert to a flat list of non-zeros."""
        return self.to_ddm().to_flat_nz()

    @classmethod
    def from_flat_nz(cls, elements, data, domain):
        """Inverse of :meth:`to_flat_nz`."""
        return DDM.from_flat_nz(elements, data, domain).to_dfm()

    def to_dod(self):
        """Convert to a DOD."""
        return self.to_ddm().to_dod()

    @classmethod
    def from_dod(cls, dod, shape, domain):
        """Inverse of :meth:`to_dod`."""
        return DDM.from_dod(dod, shape, domain).to_dfm()

    def to_dok(self):
        """Convert to a DOK."""
        return self.to_ddm().to_dok()

    @classmethod
    def from_dok(cls, dok, shape, domain):
        """Inverse of :math:`to_dod`."""
        return DDM.from_dok(dok, shape, domain).to_dfm()

    def iter_values(self):
        """Iterate over the non-zero values of the matrix."""
        m, n = self.shape
        rep = self.rep
        for i in range(m):
            for j in range(n):
                repij = rep[i, j]
                if repij:
                    yield rep[i, j]

    def iter_items(self):
        """Iterate over indices and values of nonzero elements of the matrix."""
        m, n = self.shape
        rep = self.rep
        for i in range(m):
            for j in range(n):
                repij = rep[i, j]
                if repij:
                    yield ((i, j), repij)

    def convert_to(self, domain):
        """Convert to a new domain."""
        if domain == self.domain:
            return self.copy()
        elif domain == QQ and self.domain == ZZ:
            return self._new(flint.fmpq_mat(self.rep), self.shape, domain)
        elif self._supports_domain(domain):
            # XXX: Use more efficient conversions when possible.
            return self.to_ddm().convert_to(domain).to_dfm()
        else:
            # It is the callers responsibility to convert to DDM before calling
            # this method if the domain is not supported by DFM.
            raise NotImplementedError("Only ZZ and QQ are supported by DFM")

    def getitem(self, i, j):
        """Get the ``(i, j)``-th entry."""
        # XXX: flint matrices do not support negative indices
        # XXX: They also raise ValueError instead of IndexError
        m, n = self.shape
        if i < 0:
            i += m
        if j < 0:
            j += n
        try:
            return self.rep[i, j]
        except ValueError:
            raise IndexError(f"Invalid indices ({i}, {j}) for Matrix of shape {self.shape}")

    def setitem(self, i, j, value):
        """Set the ``(i, j)``-th entry."""
        # XXX: flint matrices do not support negative indices
        # XXX: They also raise ValueError instead of IndexError
        m, n = self.shape
        if i < 0:
            i += m
        if j < 0:
            j += n
        try:
            self.rep[i, j] = value
        except ValueError:
            raise IndexError(f"Invalid indices ({i}, {j}) for Matrix of shape {self.shape}")

    def _extract(self, i_indices, j_indices):
        """Extract a submatrix with no checking."""
        # Indices must be positive and in range.
        M = self.rep
        lol = [[M[i, j] for j in j_indices] for i in i_indices]
        shape = (len(i_indices), len(j_indices))
        return self.from_list(lol, shape, self.domain)

    def extract(self, rowslist, colslist):
        """Extract a submatrix."""
        # XXX: flint matrices do not support fancy indexing or negative indices
        #
        # Check and convert negative indices before calling _extract.
        m, n = self.shape

        new_rows = []
        new_cols = []

        for i in rowslist:
            if i < 0:
                i_pos = i + m
            else:
                i_pos = i
            if not 0 <= i_pos < m:
                raise IndexError(f"Invalid row index {i} for Matrix of shape {self.shape}")
            new_rows.append(i_pos)

        for j in colslist:
            if j < 0:
                j_pos = j + n
            else:
                j_pos = j
            if not 0 <= j_pos < n:
                raise IndexError(f"Invalid column index {j} for Matrix of shape {self.shape}")
            new_cols.append(j_pos)

        return self._extract(new_rows, new_cols)

    def extract_slice(self, rowslice, colslice):
        """Slice a DFM."""
        # XXX: flint matrices do not support slicing
        m, n = self.shape
        i_indices = range(m)[rowslice]
        j_indices = range(n)[colslice]
        return self._extract(i_indices, j_indices)

    def neg(self):
        """Negate a DFM matrix."""
        return self._new_rep(-self.rep)

    def add(self, other):
        """Add two DFM matrices."""
        return self._new_rep(self.rep + other.rep)

    def sub(self, other):
        """Subtract two DFM matrices."""
        return self._new_rep(self.rep - other.rep)

    def mul(self, other):
        """Multiply a DFM matrix from the right by a scalar."""
        return self._new_rep(self.rep * other)

    def rmul(self, other):
        """Multiply a DFM matrix from the left by a scalar."""
        return self._new_rep(other * self.rep)

    def mul_elementwise(self, other):
        """Elementwise multiplication of two DFM matrices."""
        # XXX: flint matrices do not support elementwise multiplication
        return self.to_ddm().mul_elementwise(other.to_ddm()).to_dfm()

    def matmul(self, other):
        """Multiply two DFM matrices."""
        shape = (self.rows, other.cols)
        return self._new(self.rep * other.rep, shape, self.domain)

    # XXX: For the most part DomainMatrix does not expect DDM, SDM, or DFM to
    # have arithmetic operators defined. The only exception is negation.
    # Perhaps that should be removed.

    def __neg__(self):
        """Negate a DFM matrix."""
        return self.neg()

    @classmethod
    def zeros(cls, shape, domain):
        """Return a zero DFM matrix."""
        func = cls._get_flint_func(domain)
        return cls._new(func(*shape), shape, domain)

    # XXX: flint matrices do not have anything like ones or eye
    # In the methods below we convert to DDM and then back to DFM which is
    # probably about as efficient as implementing these methods directly.

    @classmethod
    def ones(cls, shape, domain):
        """Return a one DFM matrix."""
        # XXX: flint matrices do not have anything like ones
        return DDM.ones(shape, domain).to_dfm()

    @classmethod
    def eye(cls, n, domain):
        """Return the identity matrix of size n."""
        # XXX: flint matrices do not have anything like eye
        return DDM.eye(n, domain).to_dfm()

    @classmethod
    def diag(cls, elements, domain):
        """Return a diagonal matrix."""
        return DDM.diag(elements, domain).to_dfm()

    def applyfunc(self, func, domain):
        """Apply a function to each entry of a DFM matrix."""
        return self.to_ddm().applyfunc(func, domain).to_dfm()

    def transpose(self):
        """Transpose a DFM matrix."""
        return self._new(self.rep.transpose(), (self.cols, self.rows), self.domain)

    def hstack(self, *others):
        """Horizontally stack matrices."""
        return self.to_ddm().hstack(*[o.to_ddm() for o in others]).to_dfm()

    def vstack(self, *others):
        """Vertically stack matrices."""
        return self.to_ddm().vstack(*[o.to_ddm() for o in others]).to_dfm()

    def diagonal(self):
        """Return the diagonal of a DFM matrix."""
        M = self.rep
        m, n = self.shape
        return [M[i, i] for i in range(min(m, n))]

    def is_upper(self):
        """Return ``True`` if the matrix is upper triangular."""
        M = self.rep
        for i in range(self.rows):
            for j in range(min(i, self.cols)):
                if M[i, j]:
                    return False
        return True

    def is_lower(self):
        """Return ``True`` if the matrix is lower triangular."""
        M = self.rep
        for i in range(self.rows):
            for j in range(i + 1, self.cols):
                if M[i, j]:
                    return False
        return True

    def is_diagonal(self):
        """Return ``True`` if the matrix is diagonal."""
        return self.is_upper() and self.is_lower()

    def is_zero_matrix(self):
        """Return ``True`` if the matrix is the zero matrix."""
        M = self.rep
        for i in range(self.rows):
            for j in range(self.cols):
                if M[i, j]:
                    return False
        return True

    def nnz(self):
        """Return the number of non-zero elements in the matrix."""
        return self.to_ddm().nnz()

    def scc(self):
        """Return the strongly connected components of the matrix."""
        return self.to_ddm().scc()

    @doctest_depends_on(ground_types='flint')
    def det(self):
        """
        Compute the determinant of the matrix using FLINT.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 2], [3, 4]])
        >>> dfm = M.to_DM().to_dfm()
        >>> dfm
        [[1, 2], [3, 4]]
        >>> dfm.det()
        -2

        Notes
        =====

        Calls the ``.det()`` method of the underlying FLINT matrix.

        For :ref:`ZZ` or :ref:`QQ` this calls ``fmpz_mat_det`` or
        ``fmpq_mat_det`` respectively.

        At the time of writing the implementation of ``fmpz_mat_det`` uses one
        of several algorithms depending on the size of the matrix and bit size
        of the entries. The algorithms used are:

        - Cofactor for very small (up to 4x4) matrices.
        - Bareiss for small (up to 25x25) matrices.
        - Modular algorithms for larger matrices (up to 60x60) or for larger
          matrices with large bit sizes.
        - Modular "accelerated" for larger matrices (60x60 upwards) if the bit
          size is smaller than the dimensions of the matrix.

        The implementation of ``fmpq_mat_det`` clears denominators from each
        row (not the whole matrix) and then calls ``fmpz_mat_det`` and divides
        by the product of the denominators.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.det
            Higher level interface to compute the determinant of a matrix.
        """
        # XXX: At least the first three algorithms described above should also
        # be implemented in the pure Python DDM and SDM classes which at the
        # time of writng just use Bareiss for all matrices and domains.
        # Probably in Python the thresholds would be different though.
        return self.rep.det()

    @doctest_depends_on(ground_types='flint')
    def charpoly(self):
        """
        Compute the characteristic polynomial of the matrix using FLINT.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 2], [3, 4]])
        >>> dfm = M.to_DM().to_dfm()  # need ground types = 'flint'
        >>> dfm
        [[1, 2], [3, 4]]
        >>> dfm.charpoly()
        [1, -5, -2]

        Notes
        =====

        Calls the ``.charpoly()`` method of the underlying FLINT matrix.

        For :ref:`ZZ` or :ref:`QQ` this calls ``fmpz_mat_charpoly`` or
        ``fmpq_mat_charpoly`` respectively.

        At the time of writing the implementation of ``fmpq_mat_charpoly``
        clears a denominator from the whole matrix and then calls
        ``fmpz_mat_charpoly``. The coefficients of the characteristic
        polynomial are then multiplied by powers of the denominator.

        The ``fmpz_mat_charpoly`` method uses a modular algorithm with CRT
        reconstruction. The modular algorithm uses ``nmod_mat_charpoly`` which
        uses Berkowitz for small matrices and non-prime moduli or otherwise
        the Danilevsky method.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.charpoly
            Higher level interface to compute the characteristic polynomial of
            a matrix.
        """
        # FLINT polynomial coefficients are in reverse order compared to SymPy.
        return self.rep.charpoly().coeffs()[::-1]

    @doctest_depends_on(ground_types='flint')
    def inv(self):
        """
        Compute the inverse of a matrix using FLINT.

        Examples
        ========

        >>> from sympy import Matrix, QQ
        >>> M = Matrix([[1, 2], [3, 4]])
        >>> dfm = M.to_DM().to_dfm().convert_to(QQ)
        >>> dfm
        [[1, 2], [3, 4]]
        >>> dfm.inv()
        [[-2, 1], [3/2, -1/2]]
        >>> dfm.matmul(dfm.inv())
        [[1, 0], [0, 1]]

        Notes
        =====

        Calls the ``.inv()`` method of the underlying FLINT matrix.

        For now this will raise an error if the domain is :ref:`ZZ` but will
        use the FLINT method for :ref:`QQ`.

        The FLINT methods for :ref:`ZZ` and :ref:`QQ` are ``fmpz_mat_inv`` and
        ``fmpq_mat_inv`` respectively. The ``fmpz_mat_inv`` method computes an
        inverse with denominator. This is implemented by calling
        ``fmpz_mat_solve`` (see notes in :meth:`lu_solve` about the algorithm).

        The ``fmpq_mat_inv`` method clears denominators from each row and then
        multiplies those into the rhs identity matrix before calling
        ``fmpz_mat_solve``.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.inv
            Higher level method for computing the inverse of a matrix.
        """
        # TODO: Implement similar algorithms for DDM and SDM.
        #
        # XXX: The flint fmpz_mat and fmpq_mat inv methods both return fmpq_mat
        # by default. The fmpz_mat method has an optional argument to return
        # fmpz_mat instead for unimodular matrices.
        #
        # The convention in DomainMatrix is to raise an error if the matrix is
        # not over a field regardless of whether the matrix is invertible over
        # its domain or over any associated field. Maybe DomainMatrix.inv
        # should be changed to always return a matrix over an associated field
        # except with a unimodular argument for returning an inverse over a
        # ring if possible.
        #
        # For now we follow the existing DomainMatrix convention...
        K = self.domain
        m, n = self.shape

        if m != n:
            raise DMNonSquareMatrixError("cannot invert a non-square matrix")

        if K == ZZ:
            raise DMDomainError("field expected, got %s" % K)
        elif K == QQ or K.is_FF:
            try:
                return self._new_rep(self.rep.inv())
            except ZeroDivisionError:
                raise DMNonInvertibleMatrixError("matrix is not invertible")
        else:
            # If more domains are added for DFM then we will need to consider
            # what happens here.
            raise NotImplementedError("DFM.inv() is not implemented for %s" % K)

    def lu(self):
        """Return the LU decomposition of the matrix."""
        L, U, swaps = self.to_ddm().lu()
        return L.to_dfm(), U.to_dfm(), swaps

    def qr(self):
        """Return the QR decomposition of the matrix."""
        Q, R = self.to_ddm().qr()
        return Q.to_dfm(), R.to_dfm()

    # XXX: The lu_solve function should be renamed to solve. Whether or not it
    # uses an LU decomposition is an implementation detail. A method called
    # lu_solve would make sense for a situation in which an LU decomposition is
    # reused several times to solve with different rhs but that would imply a
    # different call signature.
    #
    # The underlying python-flint method has an algorithm= argument so we could
    # use that and have e.g. solve_lu and solve_modular or perhaps also a
    # method= argument to choose between the two. Flint itself has more
    # possible algorithms to choose from than are exposed by python-flint.

    @doctest_depends_on(ground_types='flint')
    def lu_solve(self, rhs):
        """
        Solve a matrix equation using FLINT.

        Examples
        ========

        >>> from sympy import Matrix, QQ
        >>> M = Matrix([[1, 2], [3, 4]])
        >>> dfm = M.to_DM().to_dfm().convert_to(QQ)
        >>> dfm
        [[1, 2], [3, 4]]
        >>> rhs = Matrix([1, 2]).to_DM().to_dfm().convert_to(QQ)
        >>> dfm.lu_solve(rhs)
        [[0], [1/2]]

        Notes
        =====

        Calls the ``.solve()`` method of the underlying FLINT matrix.

        For now this will raise an error if the domain is :ref:`ZZ` but will
        use the FLINT method for :ref:`QQ`.

        The FLINT methods for :ref:`ZZ` and :ref:`QQ` are ``fmpz_mat_solve``
        and ``fmpq_mat_solve`` respectively. The ``fmpq_mat_solve`` method
        uses one of two algorithms:

        - For small matrices (<25 rows) it clears denominators between the
          matrix and rhs and uses ``fmpz_mat_solve``.
        - For larger matrices it uses ``fmpq_mat_solve_dixon`` which is a
          modular approach with CRT reconstruction over :ref:`QQ`.

        The ``fmpz_mat_solve`` method uses one of four algorithms:

        - For very small (<= 3x3) matrices it uses a Cramer's rule.
        - For small (<= 15x15) matrices it uses a fraction-free LU solve.
        - Otherwise it uses either Dixon or another multimodular approach.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.lu_solve
            Higher level interface to solve a matrix equation.
        """
        if not self.domain == rhs.domain:
            raise DMDomainError("Domains must match: %s != %s" % (self.domain, rhs.domain))

        # XXX: As for inv we should consider whether to return a matrix over
        # over an associated field or attempt to find a solution in the ring.
        # For now we follow the existing DomainMatrix convention...
        if not self.domain.is_Field:
            raise DMDomainError("Field expected, got %s" % self.domain)

        m, n = self.shape
        j, k = rhs.shape
        if m != j:
            raise DMShapeError("Matrix size mismatch: %s * %s vs %s * %s" % (m, n, j, k))
        sol_shape = (n, k)

        # XXX: The Flint solve method only handles square matrices. Probably
        # Flint has functions that could be used to solve non-square systems
        # but they are not exposed in python-flint yet. Alternatively we could
        # put something here using the features that are available like rref.
        if m != n:
            return self.to_ddm().lu_solve(rhs.to_ddm()).to_dfm()

        try:
            sol = self.rep.solve(rhs.rep)
        except ZeroDivisionError:
            raise DMNonInvertibleMatrixError("Matrix det == 0; not invertible.")

        return self._new(sol, sol_shape, self.domain)

    def fflu(self):
        """
        Fraction-free LU decomposition of DFM.

        Explanation
        ===========

        Uses `python-flint` if possible for a matrix of
        integers otherwise uses the DDM method.

        See Also
        ========

        sympy.polys.matrices.ddm.DDM.fflu
        """
        if self.domain == ZZ:
            fflu = getattr(self.rep, 'fflu', None)
            if fflu is not None:
                P, L, D, U = self.rep.fflu()
                m, n = self.shape
                return (
                    self._new(P, (m, m), self.domain),
                    self._new(L, (m, m), self.domain),
                    self._new(D, (m, m), self.domain),
                    self._new(U, self.shape, self.domain)
                )
        ddm_p, ddm_l, ddm_d, ddm_u = self.to_ddm().fflu()
        P = ddm_p.to_dfm()
        L = ddm_l.to_dfm()
        D = ddm_d.to_dfm()
        U = ddm_u.to_dfm()
        return P, L, D, U

    def nullspace(self):
        """Return a basis for the nullspace of the matrix."""
        # Code to compute nullspace using flint:
        #
        # V, nullity = self.rep.nullspace()
        # V_dfm = self._new_rep(V)._extract(range(self.rows), range(nullity))
        #
        # XXX: That gives the nullspace but does not give us nonpivots. So we
        # use the slower DDM method anyway. It would be better to change the
        # signature of the nullspace method to not return nonpivots.
        #
        # XXX: Also python-flint exposes a nullspace method for fmpz_mat but
        # not for fmpq_mat. This is the reverse of the situation for DDM etc
        # which only allow nullspace over a field. The nullspace method for
        # DDM, SDM etc should be changed to allow nullspace over ZZ as well.
        # The DomainMatrix nullspace method does allow the domain to be a ring
        # but does not directly call the lower-level nullspace methods and uses
        # rref_den instead. Nullspace methods should also be added to all
        # matrix types in python-flint.
        ddm, nonpivots = self.to_ddm().nullspace()
        return ddm.to_dfm(), nonpivots

    def nullspace_from_rref(self, pivots=None):
        """Return a basis for the nullspace of the matrix."""
        # XXX: Use the flint nullspace method!!!
        sdm, nonpivots = self.to_sdm().nullspace_from_rref(pivots=pivots)
        return sdm.to_dfm(), nonpivots

    def particular(self):
        """Return a particular solution to the system."""
        return self.to_ddm().particular().to_dfm()

    def _lll(self, transform=False, delta=0.99, eta=0.51, rep='zbasis', gram='approx'):
        """Call the fmpz_mat.lll() method but check rank to avoid segfaults."""

        # XXX: There are tests that pass e.g. QQ(5,6) for delta. That fails
        # with a TypeError in flint because if QQ is fmpq then conversion with
        # float fails. We handle that here but there are two better fixes:
        #
        # - Make python-flint's fmpq convert with float(x)
        # - Change the tests because delta should just be a float.

        def to_float(x):
            if QQ.of_type(x):
                return float(x.numerator) / float(x.denominator)
            else:
                return float(x)

        delta = to_float(delta)
        eta = to_float(eta)

        if not 0.25 < delta < 1:
            raise DMValueError("delta must be between 0.25 and 1")

        # XXX: The flint lll method segfaults if the matrix is not full rank.
        m, n = self.shape
        if self.rep.rank() != m:
            raise DMRankError("Matrix must have full row rank for Flint LLL.")

        # Actually call the flint method.
        return self.rep.lll(transform=transform, delta=delta, eta=eta, rep=rep, gram=gram)

    @doctest_depends_on(ground_types='flint')
    def lll(self, delta=0.75):
        """Compute LLL-reduced basis using FLINT.

        See :meth:`lll_transform` for more information.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> M.to_DM().to_dfm().lll()
        [[2, 1, 0], [-1, 1, 3]]

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.lll
            Higher level interface to compute LLL-reduced basis.
        lll_transform
            Compute LLL-reduced basis and transform matrix.
        """
        if self.domain != ZZ:
            raise DMDomainError("ZZ expected, got %s" % self.domain)
        elif self.rows > self.cols:
            raise DMShapeError("Matrix must not have more rows than columns.")

        rep = self._lll(delta=delta)
        return self._new_rep(rep)

    @doctest_depends_on(ground_types='flint')
    def lll_transform(self, delta=0.75):
        """Compute LLL-reduced basis and transform using FLINT.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 2, 3], [4, 5, 6]]).to_DM().to_dfm()
        >>> M_lll, T = M.lll_transform()
        >>> M_lll
        [[2, 1, 0], [-1, 1, 3]]
        >>> T
        [[-2, 1], [3, -1]]
        >>> T.matmul(M) == M_lll
        True

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.lll
            Higher level interface to compute LLL-reduced basis.
        lll
            Compute LLL-reduced basis without transform matrix.
        """
        if self.domain != ZZ:
            raise DMDomainError("ZZ expected, got %s" % self.domain)
        elif self.rows > self.cols:
            raise DMShapeError("Matrix must not have more rows than columns.")

        rep, T = self._lll(transform=True, delta=delta)
        basis = self._new_rep(rep)
        T_dfm = self._new(T, (self.rows, self.rows), self.domain)
        return basis, T_dfm


# Avoid circular imports
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.ddm import SDM

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\routing\map.py ===
from __future__ import annotations

import typing as t
import warnings
from pprint import pformat
from threading import Lock
from urllib.parse import quote
from urllib.parse import urljoin
from urllib.parse import urlunsplit

from .._internal import _get_environ
from .._internal import _wsgi_decoding_dance
from ..datastructures import ImmutableDict
from ..datastructures import MultiDict
from ..exceptions import BadHost
from ..exceptions import HTTPException
from ..exceptions import MethodNotAllowed
from ..exceptions import NotFound
from ..urls import _urlencode
from ..wsgi import get_host
from .converters import DEFAULT_CONVERTERS
from .exceptions import BuildError
from .exceptions import NoMatch
from .exceptions import RequestAliasRedirect
from .exceptions import RequestPath
from .exceptions import RequestRedirect
from .exceptions import WebsocketMismatch
from .matcher import StateMachineMatcher
from .rules import _simple_rule_re
from .rules import Rule

if t.TYPE_CHECKING:
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment

    from ..wrappers.request import Request
    from .converters import BaseConverter
    from .rules import RuleFactory


class Map:
    """The map class stores all the URL rules and some configuration
    parameters.  Some of the configuration values are only stored on the
    `Map` instance since those affect all rules, others are just defaults
    and can be overridden for each rule.  Note that you have to specify all
    arguments besides the `rules` as keyword arguments!

    :param rules: sequence of url rules for this map.
    :param default_subdomain: The default subdomain for rules without a
                              subdomain defined.
    :param strict_slashes: If a rule ends with a slash but the matched
        URL does not, redirect to the URL with a trailing slash.
    :param merge_slashes: Merge consecutive slashes when matching or
        building URLs. Matches will redirect to the normalized URL.
        Slashes in variable parts are not merged.
    :param redirect_defaults: This will redirect to the default rule if it
                              wasn't visited that way. This helps creating
                              unique URLs.
    :param converters: A dict of converters that adds additional converters
                       to the list of converters. If you redefine one
                       converter this will override the original one.
    :param sort_parameters: If set to `True` the url parameters are sorted.
                            See `url_encode` for more details.
    :param sort_key: The sort key function for `url_encode`.
    :param host_matching: if set to `True` it enables the host matching
                          feature and disables the subdomain one.  If
                          enabled the `host` parameter to rules is used
                          instead of the `subdomain` one.

    .. versionchanged:: 3.0
        The ``charset`` and ``encoding_errors`` parameters were removed.

    .. versionchanged:: 1.0
        If ``url_scheme`` is ``ws`` or ``wss``, only WebSocket rules will match.

    .. versionchanged:: 1.0
        The ``merge_slashes`` parameter was added.

    .. versionchanged:: 0.7
        The ``encoding_errors`` and ``host_matching`` parameters were added.

    .. versionchanged:: 0.5
        The ``sort_parameters`` and ``sort_key``  paramters were added.
    """

    #: A dict of default converters to be used.
    default_converters = ImmutableDict(DEFAULT_CONVERTERS)

    #: The type of lock to use when updating.
    #:
    #: .. versionadded:: 1.0
    lock_class = Lock

    def __init__(
        self,
        rules: t.Iterable[RuleFactory] | None = None,
        default_subdomain: str = "",
        strict_slashes: bool = True,
        merge_slashes: bool = True,
        redirect_defaults: bool = True,
        converters: t.Mapping[str, type[BaseConverter]] | None = None,
        sort_parameters: bool = False,
        sort_key: t.Callable[[t.Any], t.Any] | None = None,
        host_matching: bool = False,
    ) -> None:
        self._matcher = StateMachineMatcher(merge_slashes)
        self._rules_by_endpoint: dict[t.Any, list[Rule]] = {}
        self._remap = True
        self._remap_lock = self.lock_class()

        self.default_subdomain = default_subdomain
        self.strict_slashes = strict_slashes
        self.redirect_defaults = redirect_defaults
        self.host_matching = host_matching

        self.converters = self.default_converters.copy()
        if converters:
            self.converters.update(converters)

        self.sort_parameters = sort_parameters
        self.sort_key = sort_key

        for rulefactory in rules or ():
            self.add(rulefactory)

    @property
    def merge_slashes(self) -> bool:
        return self._matcher.merge_slashes

    @merge_slashes.setter
    def merge_slashes(self, value: bool) -> None:
        self._matcher.merge_slashes = value

    def is_endpoint_expecting(self, endpoint: t.Any, *arguments: str) -> bool:
        """Iterate over all rules and check if the endpoint expects
        the arguments provided.  This is for example useful if you have
        some URLs that expect a language code and others that do not and
        you want to wrap the builder a bit so that the current language
        code is automatically added if not provided but endpoints expect
        it.

        :param endpoint: the endpoint to check.
        :param arguments: this function accepts one or more arguments
                          as positional arguments.  Each one of them is
                          checked.
        """
        self.update()
        arguments_set = set(arguments)
        for rule in self._rules_by_endpoint[endpoint]:
            if arguments_set.issubset(rule.arguments):
                return True
        return False

    @property
    def _rules(self) -> list[Rule]:
        return [rule for rules in self._rules_by_endpoint.values() for rule in rules]

    def iter_rules(self, endpoint: t.Any | None = None) -> t.Iterator[Rule]:
        """Iterate over all rules or the rules of an endpoint.

        :param endpoint: if provided only the rules for that endpoint
                         are returned.
        :return: an iterator
        """
        self.update()
        if endpoint is not None:
            return iter(self._rules_by_endpoint[endpoint])
        return iter(self._rules)

    def add(self, rulefactory: RuleFactory) -> None:
        """Add a new rule or factory to the map and bind it.  Requires that the
        rule is not bound to another map.

        :param rulefactory: a :class:`Rule` or :class:`RuleFactory`
        """
        for rule in rulefactory.get_rules(self):
            rule.bind(self)
            if not rule.build_only:
                self._matcher.add(rule)
            self._rules_by_endpoint.setdefault(rule.endpoint, []).append(rule)
        self._remap = True

    def bind(
        self,
        server_name: str,
        script_name: str | None = None,
        subdomain: str | None = None,
        url_scheme: str = "http",
        default_method: str = "GET",
        path_info: str | None = None,
        query_args: t.Mapping[str, t.Any] | str | None = None,
    ) -> MapAdapter:
        """Return a new :class:`MapAdapter` with the details specified to the
        call.  Note that `script_name` will default to ``'/'`` if not further
        specified or `None`.  The `server_name` at least is a requirement
        because the HTTP RFC requires absolute URLs for redirects and so all
        redirect exceptions raised by Werkzeug will contain the full canonical
        URL.

        If no path_info is passed to :meth:`match` it will use the default path
        info passed to bind.  While this doesn't really make sense for
        manual bind calls, it's useful if you bind a map to a WSGI
        environment which already contains the path info.

        `subdomain` will default to the `default_subdomain` for this map if
        no defined. If there is no `default_subdomain` you cannot use the
        subdomain feature.

        .. versionchanged:: 1.0
            If ``url_scheme`` is ``ws`` or ``wss``, only WebSocket rules
            will match.

        .. versionchanged:: 0.15
            ``path_info`` defaults to ``'/'`` if ``None``.

        .. versionchanged:: 0.8
            ``query_args`` can be a string.

        .. versionchanged:: 0.7
            Added ``query_args``.
        """
        server_name = server_name.lower()
        if self.host_matching:
            if subdomain is not None:
                raise RuntimeError("host matching enabled and a subdomain was provided")
        elif subdomain is None:
            subdomain = self.default_subdomain
        if script_name is None:
            script_name = "/"
        if path_info is None:
            path_info = "/"

        # Port isn't part of IDNA, and might push a name over the 63 octet limit.
        server_name, port_sep, port = server_name.partition(":")

        try:
            server_name = server_name.encode("idna").decode("ascii")
        except UnicodeError as e:
            raise BadHost() from e

        return MapAdapter(
            self,
            f"{server_name}{port_sep}{port}",
            script_name,
            subdomain,
            url_scheme,
            path_info,
            default_method,
            query_args,
        )

    def bind_to_environ(
        self,
        environ: WSGIEnvironment | Request,
        server_name: str | None = None,
        subdomain: str | None = None,
    ) -> MapAdapter:
        """Like :meth:`bind` but you can pass it an WSGI environment and it
        will fetch the information from that dictionary.  Note that because of
        limitations in the protocol there is no way to get the current
        subdomain and real `server_name` from the environment.  If you don't
        provide it, Werkzeug will use `SERVER_NAME` and `SERVER_PORT` (or
        `HTTP_HOST` if provided) as used `server_name` with disabled subdomain
        feature.

        If `subdomain` is `None` but an environment and a server name is
        provided it will calculate the current subdomain automatically.
        Example: `server_name` is ``'example.com'`` and the `SERVER_NAME`
        in the wsgi `environ` is ``'staging.dev.example.com'`` the calculated
        subdomain will be ``'staging.dev'``.

        If the object passed as environ has an environ attribute, the value of
        this attribute is used instead.  This allows you to pass request
        objects.  Additionally `PATH_INFO` added as a default of the
        :class:`MapAdapter` so that you don't have to pass the path info to
        the match method.

        .. versionchanged:: 1.0.0
            If the passed server name specifies port 443, it will match
            if the incoming scheme is ``https`` without a port.

        .. versionchanged:: 1.0.0
            A warning is shown when the passed server name does not
            match the incoming WSGI server name.

        .. versionchanged:: 0.8
           This will no longer raise a ValueError when an unexpected server
           name was passed.

        .. versionchanged:: 0.5
            previously this method accepted a bogus `calculate_subdomain`
            parameter that did not have any effect.  It was removed because
            of that.

        :param environ: a WSGI environment.
        :param server_name: an optional server name hint (see above).
        :param subdomain: optionally the current subdomain (see above).
        """
        env = _get_environ(environ)
        wsgi_server_name = get_host(env).lower()
        scheme = env["wsgi.url_scheme"]
        upgrade = any(
            v.strip() == "upgrade"
            for v in env.get("HTTP_CONNECTION", "").lower().split(",")
        )

        if upgrade and env.get("HTTP_UPGRADE", "").lower() == "websocket":
            scheme = "wss" if scheme == "https" else "ws"

        if server_name is None:
            server_name = wsgi_server_name
        else:
            server_name = server_name.lower()

            # strip standard port to match get_host()
            if scheme in {"http", "ws"} and server_name.endswith(":80"):
                server_name = server_name[:-3]
            elif scheme in {"https", "wss"} and server_name.endswith(":443"):
                server_name = server_name[:-4]

        if subdomain is None and not self.host_matching:
            cur_server_name = wsgi_server_name.split(".")
            real_server_name = server_name.split(".")
            offset = -len(real_server_name)

            if cur_server_name[offset:] != real_server_name:
                # This can happen even with valid configs if the server was
                # accessed directly by IP address under some situations.
                # Instead of raising an exception like in Werkzeug 0.7 or
                # earlier we go by an invalid subdomain which will result
                # in a 404 error on matching.
                warnings.warn(
                    f"Current server name {wsgi_server_name!r} doesn't match configured"
                    f" server name {server_name!r}",
                    stacklevel=2,
                )
                subdomain = "<invalid>"
            else:
                subdomain = ".".join(filter(None, cur_server_name[:offset]))

        def _get_wsgi_string(name: str) -> str | None:
            val = env.get(name)
            if val is not None:
                return _wsgi_decoding_dance(val)
            return None

        script_name = _get_wsgi_string("SCRIPT_NAME")
        path_info = _get_wsgi_string("PATH_INFO")
        query_args = _get_wsgi_string("QUERY_STRING")
        return Map.bind(
            self,
            server_name,
            script_name,
            subdomain,
            scheme,
            env["REQUEST_METHOD"],
            path_info,
            query_args=query_args,
        )

    def update(self) -> None:
        """Called before matching and building to keep the compiled rules
        in the correct order after things changed.
        """
        if not self._remap:
            return

        with self._remap_lock:
            if not self._remap:
                return

            self._matcher.update()
            for rules in self._rules_by_endpoint.values():
                rules.sort(key=lambda x: x.build_compare_key())
            self._remap = False

    def __repr__(self) -> str:
        rules = self.iter_rules()
        return f"{type(self).__name__}({pformat(list(rules))})"


class MapAdapter:
    """Returned by :meth:`Map.bind` or :meth:`Map.bind_to_environ` and does
    the URL matching and building based on runtime information.
    """

    def __init__(
        self,
        map: Map,
        server_name: str,
        script_name: str,
        subdomain: str | None,
        url_scheme: str,
        path_info: str,
        default_method: str,
        query_args: t.Mapping[str, t.Any] | str | None = None,
    ):
        self.map = map
        self.server_name = server_name

        if not script_name.endswith("/"):
            script_name += "/"

        self.script_name = script_name
        self.subdomain = subdomain
        self.url_scheme = url_scheme
        self.path_info = path_info
        self.default_method = default_method
        self.query_args = query_args
        self.websocket = self.url_scheme in {"ws", "wss"}

    def dispatch(
        self,
        view_func: t.Callable[[str, t.Mapping[str, t.Any]], WSGIApplication],
        path_info: str | None = None,
        method: str | None = None,
        catch_http_exceptions: bool = False,
    ) -> WSGIApplication:
        """Does the complete dispatching process.  `view_func` is called with
        the endpoint and a dict with the values for the view.  It should
        look up the view function, call it, and return a response object
        or WSGI application.  http exceptions are not caught by default
        so that applications can display nicer error messages by just
        catching them by hand.  If you want to stick with the default
        error messages you can pass it ``catch_http_exceptions=True`` and
        it will catch the http exceptions.

        Here a small example for the dispatch usage::

            from werkzeug.wrappers import Request, Response
            from werkzeug.wsgi import responder
            from werkzeug.routing import Map, Rule

            def on_index(request):
                return Response('Hello from the index')

            url_map = Map([Rule('/', endpoint='index')])
            views = {'index': on_index}

            @responder
            def application(environ, start_response):
                request = Request(environ)
                urls = url_map.bind_to_environ(environ)
                return urls.dispatch(lambda e, v: views[e](request, **v),
                                     catch_http_exceptions=True)

        Keep in mind that this method might return exception objects, too, so
        use :class:`Response.force_type` to get a response object.

        :param view_func: a function that is called with the endpoint as
                          first argument and the value dict as second.  Has
                          to dispatch to the actual view function with this
                          information.  (see above)
        :param path_info: the path info to use for matching.  Overrides the
                          path info specified on binding.
        :param method: the HTTP method used for matching.  Overrides the
                       method specified on binding.
        :param catch_http_exceptions: set to `True` to catch any of the
                                      werkzeug :class:`HTTPException`\\s.
        """
        try:
            try:
                endpoint, args = self.match(path_info, method)
            except RequestRedirect as e:
                return e
            return view_func(endpoint, args)
        except HTTPException as e:
            if catch_http_exceptions:
                return e
            raise

    @t.overload
    def match(
        self,
        path_info: str | None = None,
        method: str | None = None,
        return_rule: t.Literal[False] = False,
        query_args: t.Mapping[str, t.Any] | str | None = None,
        websocket: bool | None = None,
    ) -> tuple[t.Any, t.Mapping[str, t.Any]]: ...

    @t.overload
    def match(
        self,
        path_info: str | None = None,
        method: str | None = None,
        return_rule: t.Literal[True] = True,
        query_args: t.Mapping[str, t.Any] | str | None = None,
        websocket: bool | None = None,
    ) -> tuple[Rule, t.Mapping[str, t.Any]]: ...

    def match(
        self,
        path_info: str | None = None,
        method: str | None = None,
        return_rule: bool = False,
        query_args: t.Mapping[str, t.Any] | str | None = None,
        websocket: bool | None = None,
    ) -> tuple[t.Any | Rule, t.Mapping[str, t.Any]]:
        """The usage is simple: you just pass the match method the current
        path info as well as the method (which defaults to `GET`).  The
        following things can then happen:

        - you receive a `NotFound` exception that indicates that no URL is
          matching.  A `NotFound` exception is also a WSGI application you
          can call to get a default page not found page (happens to be the
          same object as `werkzeug.exceptions.NotFound`)

        - you receive a `MethodNotAllowed` exception that indicates that there
          is a match for this URL but not for the current request method.
          This is useful for RESTful applications.

        - you receive a `RequestRedirect` exception with a `new_url`
          attribute.  This exception is used to notify you about a request
          Werkzeug requests from your WSGI application.  This is for example the
          case if you request ``/foo`` although the correct URL is ``/foo/``
          You can use the `RequestRedirect` instance as response-like object
          similar to all other subclasses of `HTTPException`.

        - you receive a ``WebsocketMismatch`` exception if the only
          match is a WebSocket rule but the bind is an HTTP request, or
          if the match is an HTTP rule but the bind is a WebSocket
          request.

        - you get a tuple in the form ``(endpoint, arguments)`` if there is
          a match (unless `return_rule` is True, in which case you get a tuple
          in the form ``(rule, arguments)``)

        If the path info is not passed to the match method the default path
        info of the map is used (defaults to the root URL if not defined
        explicitly).

        All of the exceptions raised are subclasses of `HTTPException` so they
        can be used as WSGI responses. They will all render generic error or
        redirect pages.

        Here is a small example for matching:

        >>> m = Map([
        ...     Rule('/', endpoint='index'),
        ...     Rule('/downloads/', endpoint='downloads/index'),
        ...     Rule('/downloads/<int:id>', endpoint='downloads/show')
        ... ])
        >>> urls = m.bind("example.com", "/")
        >>> urls.match("/", "GET")
        ('index', {})
        >>> urls.match("/downloads/42")
        ('downloads/show', {'id': 42})

        And here is what happens on redirect and missing URLs:

        >>> urls.match("/downloads")
        Traceback (most recent call last):
          ...
        RequestRedirect: http://example.com/downloads/
        >>> urls.match("/missing")
        Traceback (most recent call last):
          ...
        NotFound: 404 Not Found

        :param path_info: the path info to use for matching.  Overrides the
                          path info specified on binding.
        :param method: the HTTP method used for matching.  Overrides the
                       method specified on binding.
        :param return_rule: return the rule that matched instead of just the
                            endpoint (defaults to `False`).
        :param query_args: optional query arguments that are used for
                           automatic redirects as string or dictionary.  It's
                           currently not possible to use the query arguments
                           for URL matching.
        :param websocket: Match WebSocket instead of HTTP requests. A
            websocket request has a ``ws`` or ``wss``
            :attr:`url_scheme`. This overrides that detection.

        .. versionadded:: 1.0
            Added ``websocket``.

        .. versionchanged:: 0.8
            ``query_args`` can be a string.

        .. versionadded:: 0.7
            Added ``query_args``.

        .. versionadded:: 0.6
            Added ``return_rule``.
        """
        self.map.update()
        if path_info is None:
            path_info = self.path_info
        if query_args is None:
            query_args = self.query_args or {}
        method = (method or self.default_method).upper()

        if websocket is None:
            websocket = self.websocket

        domain_part = self.server_name

        if not self.map.host_matching and self.subdomain is not None:
            domain_part = self.subdomain

        path_part = f"/{path_info.lstrip('/')}" if path_info else ""

        try:
            result = self.map._matcher.match(domain_part, path_part, method, websocket)
        except RequestPath as e:
            # safe = https://url.spec.whatwg.org/#url-path-segment-string
            new_path = quote(e.path_info, safe="!$&'()*+,/:;=@")
            raise RequestRedirect(
                self.make_redirect_url(new_path, query_args)
            ) from None
        except RequestAliasRedirect as e:
            raise RequestRedirect(
                self.make_alias_redirect_url(
                    f"{domain_part}|{path_part}",
                    e.endpoint,
                    e.matched_values,
                    method,
                    query_args,
                )
            ) from None
        except NoMatch as e:
            if e.have_match_for:
                raise MethodNotAllowed(valid_methods=list(e.have_match_for)) from None

            if e.websocket_mismatch:
                raise WebsocketMismatch() from None

            raise NotFound() from None
        else:
            rule, rv = result

            if self.map.redirect_defaults:
                redirect_url = self.get_default_redirect(rule, method, rv, query_args)
                if redirect_url is not None:
                    raise RequestRedirect(redirect_url)

            if rule.redirect_to is not None:
                if isinstance(rule.redirect_to, str):

                    def _handle_match(match: t.Match[str]) -> str:
                        value = rv[match.group(1)]
                        return rule._converters[match.group(1)].to_url(value)

                    redirect_url = _simple_rule_re.sub(_handle_match, rule.redirect_to)
                else:
                    redirect_url = rule.redirect_to(self, **rv)

                if self.subdomain:
                    netloc = f"{self.subdomain}.{self.server_name}"
                else:
                    netloc = self.server_name

                raise RequestRedirect(
                    urljoin(
                        f"{self.url_scheme or 'http'}://{netloc}{self.script_name}",
                        redirect_url,
                    )
                )

            if return_rule:
                return rule, rv
            else:
                return rule.endpoint, rv

    def test(self, path_info: str | None = None, method: str | None = None) -> bool:
        """Test if a rule would match.  Works like `match` but returns `True`
        if the URL matches, or `False` if it does not exist.

        :param path_info: the path info to use for matching.  Overrides the
                          path info specified on binding.
        :param method: the HTTP method used for matching.  Overrides the
                       method specified on binding.
        """
        try:
            self.match(path_info, method)
        except RequestRedirect:
            pass
        except HTTPException:
            return False
        return True

    def allowed_methods(self, path_info: str | None = None) -> t.Iterable[str]:
        """Returns the valid methods that match for a given path.

        .. versionadded:: 0.7
        """
        try:
            self.match(path_info, method="--")
        except MethodNotAllowed as e:
            return e.valid_methods  # type: ignore
        except HTTPException:
            pass
        return []

    def get_host(self, domain_part: str | None) -> str:
        """Figures out the full host name for the given domain part.  The
        domain part is a subdomain in case host matching is disabled or
        a full host name.
        """
        if self.map.host_matching:
            if domain_part is None:
                return self.server_name

            return domain_part

        if domain_part is None:
            subdomain = self.subdomain
        else:
            subdomain = domain_part

        if subdomain:
            return f"{subdomain}.{self.server_name}"
        else:
            return self.server_name

    def get_default_redirect(
        self,
        rule: Rule,
        method: str,
        values: t.MutableMapping[str, t.Any],
        query_args: t.Mapping[str, t.Any] | str,
    ) -> str | None:
        """A helper that returns the URL to redirect to if it finds one.
        This is used for default redirecting only.

        :internal:
        """
        assert self.map.redirect_defaults
        for r in self.map._rules_by_endpoint[rule.endpoint]:
            # every rule that comes after this one, including ourself
            # has a lower priority for the defaults.  We order the ones
            # with the highest priority up for building.
            if r is rule:
                break
            if r.provides_defaults_for(rule) and r.suitable_for(values, method):
                values.update(r.defaults)  # type: ignore
                domain_part, path = r.build(values)  # type: ignore
                return self.make_redirect_url(path, query_args, domain_part=domain_part)
        return None

    def encode_query_args(self, query_args: t.Mapping[str, t.Any] | str) -> str:
        if not isinstance(query_args, str):
            return _urlencode(query_args)
        return query_args

    def make_redirect_url(
        self,
        path_info: str,
        query_args: t.Mapping[str, t.Any] | str | None = None,
        domain_part: str | None = None,
    ) -> str:
        """Creates a redirect URL.

        :internal:
        """
        if query_args is None:
            query_args = self.query_args

        if query_args:
            query_str = self.encode_query_args(query_args)
        else:
            query_str = None

        scheme = self.url_scheme or "http"
        host = self.get_host(domain_part)
        path = "/".join((self.script_name.strip("/"), path_info.lstrip("/")))
        return urlunsplit((scheme, host, path, query_str, None))

    def make_alias_redirect_url(
        self,
        path: str,
        endpoint: t.Any,
        values: t.Mapping[str, t.Any],
        method: str,
        query_args: t.Mapping[str, t.Any] | str,
    ) -> str:
        """Internally called to make an alias redirect URL."""
        url = self.build(
            endpoint, values, method, append_unknown=False, force_external=True
        )
        if query_args:
            url += f"?{self.encode_query_args(query_args)}"
        assert url != path, "detected invalid alias setting. No canonical URL found"
        return url

    def _partial_build(
        self,
        endpoint: t.Any,
        values: t.Mapping[str, t.Any],
        method: str | None,
        append_unknown: bool,
    ) -> tuple[str, str, bool] | None:
        """Helper for :meth:`build`.  Returns subdomain and path for the
        rule that accepts this endpoint, values and method.

        :internal:
        """
        # in case the method is none, try with the default method first
        if method is None:
            rv = self._partial_build(
                endpoint, values, self.default_method, append_unknown
            )
            if rv is not None:
                return rv

        # Default method did not match or a specific method is passed.
        # Check all for first match with matching host. If no matching
        # host is found, go with first result.
        first_match = None

        for rule in self.map._rules_by_endpoint.get(endpoint, ()):
            if rule.suitable_for(values, method):
                build_rv = rule.build(values, append_unknown)

                if build_rv is not None:
                    rv = (build_rv[0], build_rv[1], rule.websocket)
                    if self.map.host_matching:
                        if rv[0] == self.server_name:
                            return rv
                        elif first_match is None:
                            first_match = rv
                    else:
                        return rv

        return first_match

    def build(
        self,
        endpoint: t.Any,
        values: t.Mapping[str, t.Any] | None = None,
        method: str | None = None,
        force_external: bool = False,
        append_unknown: bool = True,
        url_scheme: str | None = None,
    ) -> str:
        """Building URLs works pretty much the other way round.  Instead of
        `match` you call `build` and pass it the endpoint and a dict of
        arguments for the placeholders.

        The `build` function also accepts an argument called `force_external`
        which, if you set it to `True` will force external URLs. Per default
        external URLs (include the server name) will only be used if the
        target URL is on a different subdomain.

        >>> m = Map([
        ...     Rule('/', endpoint='index'),
        ...     Rule('/downloads/', endpoint='downloads/index'),
        ...     Rule('/downloads/<int:id>', endpoint='downloads/show')
        ... ])
        >>> urls = m.bind("example.com", "/")
        >>> urls.build("index", {})
        '/'
        >>> urls.build("downloads/show", {'id': 42})
        '/downloads/42'
        >>> urls.build("downloads/show", {'id': 42}, force_external=True)
        'http://example.com/downloads/42'

        Because URLs cannot contain non ASCII data you will always get
        bytes back.  Non ASCII characters are urlencoded with the
        charset defined on the map instance.

        Additional values are converted to strings and appended to the URL as
        URL querystring parameters:

        >>> urls.build("index", {'q': 'My Searchstring'})
        '/?q=My+Searchstring'

        When processing those additional values, lists are furthermore
        interpreted as multiple values (as per
        :py:class:`werkzeug.datastructures.MultiDict`):

        >>> urls.build("index", {'q': ['a', 'b', 'c']})
        '/?q=a&q=b&q=c'

        Passing a ``MultiDict`` will also add multiple values:

        >>> urls.build("index", MultiDict((('p', 'z'), ('q', 'a'), ('q', 'b'))))
        '/?p=z&q=a&q=b'

        If a rule does not exist when building a `BuildError` exception is
        raised.

        The build method accepts an argument called `method` which allows you
        to specify the method you want to have an URL built for if you have
        different methods for the same endpoint specified.

        :param endpoint: the endpoint of the URL to build.
        :param values: the values for the URL to build.  Unhandled values are
                       appended to the URL as query parameters.
        :param method: the HTTP method for the rule if there are different
                       URLs for different methods on the same endpoint.
        :param force_external: enforce full canonical external URLs. If the URL
                               scheme is not provided, this will generate
                               a protocol-relative URL.
        :param append_unknown: unknown parameters are appended to the generated
                               URL as query string argument.  Disable this
                               if you want the builder to ignore those.
        :param url_scheme: Scheme to use in place of the bound
            :attr:`url_scheme`.

        .. versionchanged:: 2.0
            Added the ``url_scheme`` parameter.

        .. versionadded:: 0.6
           Added the ``append_unknown`` parameter.
        """
        self.map.update()

        if values:
            if isinstance(values, MultiDict):
                values = {
                    k: (v[0] if len(v) == 1 else v)
                    for k, v in dict.items(values)
                    if len(v) != 0
                }
            else:  # plain dict
                values = {k: v for k, v in values.items() if v is not None}
        else:
            values = {}

        rv = self._partial_build(endpoint, values, method, append_unknown)
        if rv is None:
            raise BuildError(endpoint, values, method, self)

        domain_part, path, websocket = rv
        host = self.get_host(domain_part)

        if url_scheme is None:
            url_scheme = self.url_scheme

        # Always build WebSocket routes with the scheme (browsers
        # require full URLs). If bound to a WebSocket, ensure that HTTP
        # routes are built with an HTTP scheme.
        secure = url_scheme in {"https", "wss"}

        if websocket:
            force_external = True
            url_scheme = "wss" if secure else "ws"
        elif url_scheme:
            url_scheme = "https" if secure else "http"

        # shortcut this.
        if not force_external and (
            (self.map.host_matching and host == self.server_name)
            or (not self.map.host_matching and domain_part == self.subdomain)
        ):
            return f"{self.script_name.rstrip('/')}/{path.lstrip('/')}"

        scheme = f"{url_scheme}:" if url_scheme else ""
        return f"{scheme}//{host}{self.script_name[:-1]}/{path.lstrip('/')}"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\_config\config.py ===
"""
The config module holds package-wide configurables and provides
a uniform API for working with them.

Overview
========

This module supports the following requirements:
- options are referenced using keys in dot.notation, e.g. "x.y.option - z".
- keys are case-insensitive.
- functions should accept partial/regex keys, when unambiguous.
- options can be registered by modules at import time.
- options can be registered at init-time (via core.config_init)
- options have a default value, and (optionally) a description and
  validation function associated with them.
- options can be deprecated, in which case referencing them
  should produce a warning.
- deprecated options can optionally be rerouted to a replacement
  so that accessing a deprecated option reroutes to a differently
  named option.
- options can be reset to their default value.
- all option can be reset to their default value at once.
- all options in a certain sub - namespace can be reset at once.
- the user can set / get / reset or ask for the description of an option.
- a developer can register and mark an option as deprecated.
- you can register a callback to be invoked when the option value
  is set or reset. Changing the stored value is considered misuse, but
  is not verboten.

Implementation
==============

- Data is stored using nested dictionaries, and should be accessed
  through the provided API.

- "Registered options" and "Deprecated options" have metadata associated
  with them, which are stored in auxiliary dictionaries keyed on the
  fully-qualified key, e.g. "x.y.z.option".

- the config_init module is imported by the package's __init__.py file.
  placing any register_option() calls there will ensure those options
  are available as soon as pandas is loaded. If you use register_option
  in a module, it will only be available after that module is imported,
  which you should be aware of.

- `config_prefix` is a context_manager (for use with the `with` keyword)
  which can save developers some typing, see the docstring.

"""

from __future__ import annotations

from contextlib import (
    ContextDecorator,
    contextmanager,
)
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    NamedTuple,
    cast,
)
import warnings

from pandas._typing import (
    F,
    T,
)
from pandas.util._exceptions import find_stack_level

if TYPE_CHECKING:
    from collections.abc import (
        Generator,
        Iterable,
    )


class DeprecatedOption(NamedTuple):
    key: str
    msg: str | None
    rkey: str | None
    removal_ver: str | None


class RegisteredOption(NamedTuple):
    key: str
    defval: object
    doc: str
    validator: Callable[[object], Any] | None
    cb: Callable[[str], Any] | None


# holds deprecated option metadata
_deprecated_options: dict[str, DeprecatedOption] = {}

# holds registered option metadata
_registered_options: dict[str, RegisteredOption] = {}

# holds the current values for registered options
_global_config: dict[str, Any] = {}

# keys which have a special meaning
_reserved_keys: list[str] = ["all"]


class OptionError(AttributeError, KeyError):
    """
    Exception raised for pandas.options.

    Backwards compatible with KeyError checks.

    Examples
    --------
    >>> pd.options.context
    Traceback (most recent call last):
    OptionError: No such option
    """


#
# User API


def _get_single_key(pat: str, silent: bool) -> str:
    keys = _select_options(pat)
    if len(keys) == 0:
        if not silent:
            _warn_if_deprecated(pat)
        raise OptionError(f"No such keys(s): {repr(pat)}")
    if len(keys) > 1:
        raise OptionError("Pattern matched multiple keys")
    key = keys[0]

    if not silent:
        _warn_if_deprecated(key)

    key = _translate_key(key)

    return key


def _get_option(pat: str, silent: bool = False) -> Any:
    key = _get_single_key(pat, silent)

    # walk the nested dict
    root, k = _get_root(key)
    return root[k]


def _set_option(*args, **kwargs) -> None:
    # must at least 1 arg deal with constraints later
    nargs = len(args)
    if not nargs or nargs % 2 != 0:
        raise ValueError("Must provide an even number of non-keyword arguments")

    # default to false
    silent = kwargs.pop("silent", False)

    if kwargs:
        kwarg = next(iter(kwargs.keys()))
        raise TypeError(f'_set_option() got an unexpected keyword argument "{kwarg}"')

    for k, v in zip(args[::2], args[1::2]):
        key = _get_single_key(k, silent)

        o = _get_registered_option(key)
        if o and o.validator:
            o.validator(v)

        # walk the nested dict
        root, k_root = _get_root(key)
        root[k_root] = v

        if o.cb:
            if silent:
                with warnings.catch_warnings(record=True):
                    o.cb(key)
            else:
                o.cb(key)


def _describe_option(pat: str = "", _print_desc: bool = True) -> str | None:
    keys = _select_options(pat)
    if len(keys) == 0:
        raise OptionError("No such keys(s)")

    s = "\n".join([_build_option_description(k) for k in keys])

    if _print_desc:
        print(s)
        return None
    return s


def _reset_option(pat: str, silent: bool = False) -> None:
    keys = _select_options(pat)

    if len(keys) == 0:
        raise OptionError("No such keys(s)")

    if len(keys) > 1 and len(pat) < 4 and pat != "all":
        raise ValueError(
            "You must specify at least 4 characters when "
            "resetting multiple keys, use the special keyword "
            '"all" to reset all the options to their default value'
        )

    for k in keys:
        _set_option(k, _registered_options[k].defval, silent=silent)


def get_default_val(pat: str):
    key = _get_single_key(pat, silent=True)
    return _get_registered_option(key).defval


class DictWrapper:
    """provide attribute-style access to a nested dict"""

    d: dict[str, Any]

    def __init__(self, d: dict[str, Any], prefix: str = "") -> None:
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "prefix", prefix)

    def __setattr__(self, key: str, val: Any) -> None:
        prefix = object.__getattribute__(self, "prefix")
        if prefix:
            prefix += "."
        prefix += key
        # you can't set new keys
        # can you can't overwrite subtrees
        if key in self.d and not isinstance(self.d[key], dict):
            _set_option(prefix, val)
        else:
            raise OptionError("You can only set the value of existing options")

    def __getattr__(self, key: str):
        prefix = object.__getattribute__(self, "prefix")
        if prefix:
            prefix += "."
        prefix += key
        try:
            v = object.__getattribute__(self, "d")[key]
        except KeyError as err:
            raise OptionError("No such option") from err
        if isinstance(v, dict):
            return DictWrapper(v, prefix)
        else:
            return _get_option(prefix)

    def __dir__(self) -> list[str]:
        return list(self.d.keys())


# For user convenience,  we'd like to have the available options described
# in the docstring. For dev convenience we'd like to generate the docstrings
# dynamically instead of maintaining them by hand. To this, we use the
# class below which wraps functions inside a callable, and converts
# __doc__ into a property function. The doctsrings below are templates
# using the py2.6+ advanced formatting syntax to plug in a concise list
# of options, and option descriptions.


class CallableDynamicDoc(Generic[T]):
    def __init__(self, func: Callable[..., T], doc_tmpl: str) -> None:
        self.__doc_tmpl__ = doc_tmpl
        self.__func__ = func

    def __call__(self, *args, **kwds) -> T:
        return self.__func__(*args, **kwds)

    # error: Signature of "__doc__" incompatible with supertype "object"
    @property
    def __doc__(self) -> str:  # type: ignore[override]
        opts_desc = _describe_option("all", _print_desc=False)
        opts_list = pp_options_list(list(_registered_options.keys()))
        return self.__doc_tmpl__.format(opts_desc=opts_desc, opts_list=opts_list)


_get_option_tmpl = """
get_option(pat)

Retrieves the value of the specified option.

Available options:

{opts_list}

Parameters
----------
pat : str
    Regexp which should match a single option.
    Note: partial matches are supported for convenience, but unless you use the
    full option name (e.g. x.y.z.option_name), your code may break in future
    versions if new options with similar names are introduced.

Returns
-------
result : the value of the option

Raises
------
OptionError : if no such option exists

Notes
-----
Please reference the :ref:`User Guide <options>` for more information.

The available options with its descriptions:

{opts_desc}

Examples
--------
>>> pd.get_option('display.max_columns')  # doctest: +SKIP
4
"""

_set_option_tmpl = """
set_option(pat, value)

Sets the value of the specified option.

Available options:

{opts_list}

Parameters
----------
pat : str
    Regexp which should match a single option.
    Note: partial matches are supported for convenience, but unless you use the
    full option name (e.g. x.y.z.option_name), your code may break in future
    versions if new options with similar names are introduced.
value : object
    New value of option.

Returns
-------
None

Raises
------
OptionError if no such option exists

Notes
-----
Please reference the :ref:`User Guide <options>` for more information.

The available options with its descriptions:

{opts_desc}

Examples
--------
>>> pd.set_option('display.max_columns', 4)
>>> df = pd.DataFrame([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
>>> df
   0  1  ...  3   4
0  1  2  ...  4   5
1  6  7  ...  9  10
[2 rows x 5 columns]
>>> pd.reset_option('display.max_columns')
"""

_describe_option_tmpl = """
describe_option(pat, _print_desc=False)

Prints the description for one or more registered options.

Call with no arguments to get a listing for all registered options.

Available options:

{opts_list}

Parameters
----------
pat : str
    Regexp pattern. All matching keys will have their description displayed.
_print_desc : bool, default True
    If True (default) the description(s) will be printed to stdout.
    Otherwise, the description(s) will be returned as a unicode string
    (for testing).

Returns
-------
None by default, the description(s) as a unicode string if _print_desc
is False

Notes
-----
Please reference the :ref:`User Guide <options>` for more information.

The available options with its descriptions:

{opts_desc}

Examples
--------
>>> pd.describe_option('display.max_columns')  # doctest: +SKIP
display.max_columns : int
    If max_cols is exceeded, switch to truncate view...
"""

_reset_option_tmpl = """
reset_option(pat)

Reset one or more options to their default value.

Pass "all" as argument to reset all options.

Available options:

{opts_list}

Parameters
----------
pat : str/regex
    If specified only options matching `prefix*` will be reset.
    Note: partial matches are supported for convenience, but unless you
    use the full option name (e.g. x.y.z.option_name), your code may break
    in future versions if new options with similar names are introduced.

Returns
-------
None

Notes
-----
Please reference the :ref:`User Guide <options>` for more information.

The available options with its descriptions:

{opts_desc}

Examples
--------
>>> pd.reset_option('display.max_columns')  # doctest: +SKIP
"""

# bind the functions with their docstrings into a Callable
# and use that as the functions exposed in pd.api
get_option = CallableDynamicDoc(_get_option, _get_option_tmpl)
set_option = CallableDynamicDoc(_set_option, _set_option_tmpl)
reset_option = CallableDynamicDoc(_reset_option, _reset_option_tmpl)
describe_option = CallableDynamicDoc(_describe_option, _describe_option_tmpl)
options = DictWrapper(_global_config)

#
# Functions for use by pandas developers, in addition to User - api


class option_context(ContextDecorator):
    """
    Context manager to temporarily set options in the `with` statement context.

    You need to invoke as ``option_context(pat, val, [(pat, val), ...])``.

    Examples
    --------
    >>> from pandas import option_context
    >>> with option_context('display.max_rows', 10, 'display.max_columns', 5):
    ...     pass
    """

    def __init__(self, *args) -> None:
        if len(args) % 2 != 0 or len(args) < 2:
            raise ValueError(
                "Need to invoke as option_context(pat, val, [(pat, val), ...])."
            )

        self.ops = list(zip(args[::2], args[1::2]))

    def __enter__(self) -> None:
        self.undo = [(pat, _get_option(pat)) for pat, val in self.ops]

        for pat, val in self.ops:
            _set_option(pat, val, silent=True)

    def __exit__(self, *args) -> None:
        if self.undo:
            for pat, val in self.undo:
                _set_option(pat, val, silent=True)


def register_option(
    key: str,
    defval: object,
    doc: str = "",
    validator: Callable[[object], Any] | None = None,
    cb: Callable[[str], Any] | None = None,
) -> None:
    """
    Register an option in the package-wide pandas config object

    Parameters
    ----------
    key : str
        Fully-qualified key, e.g. "x.y.option - z".
    defval : object
        Default value of the option.
    doc : str
        Description of the option.
    validator : Callable, optional
        Function of a single argument, should raise `ValueError` if
        called with a value which is not a legal value for the option.
    cb
        a function of a single argument "key", which is called
        immediately after an option value is set/reset. key is
        the full name of the option.

    Raises
    ------
    ValueError if `validator` is specified and `defval` is not a valid value.

    """
    import keyword
    import tokenize

    key = key.lower()

    if key in _registered_options:
        raise OptionError(f"Option '{key}' has already been registered")
    if key in _reserved_keys:
        raise OptionError(f"Option '{key}' is a reserved key")

    # the default value should be legal
    if validator:
        validator(defval)

    # walk the nested dict, creating dicts as needed along the path
    path = key.split(".")

    for k in path:
        if not re.match("^" + tokenize.Name + "$", k):
            raise ValueError(f"{k} is not a valid identifier")
        if keyword.iskeyword(k):
            raise ValueError(f"{k} is a python keyword")

    cursor = _global_config
    msg = "Path prefix to option '{option}' is already an option"

    for i, p in enumerate(path[:-1]):
        if not isinstance(cursor, dict):
            raise OptionError(msg.format(option=".".join(path[:i])))
        if p not in cursor:
            cursor[p] = {}
        cursor = cursor[p]

    if not isinstance(cursor, dict):
        raise OptionError(msg.format(option=".".join(path[:-1])))

    cursor[path[-1]] = defval  # initialize

    # save the option metadata
    _registered_options[key] = RegisteredOption(
        key=key, defval=defval, doc=doc, validator=validator, cb=cb
    )


def deprecate_option(
    key: str,
    msg: str | None = None,
    rkey: str | None = None,
    removal_ver: str | None = None,
) -> None:
    """
    Mark option `key` as deprecated, if code attempts to access this option,
    a warning will be produced, using `msg` if given, or a default message
    if not.
    if `rkey` is given, any access to the key will be re-routed to `rkey`.

    Neither the existence of `key` nor that if `rkey` is checked. If they
    do not exist, any subsequence access will fail as usual, after the
    deprecation warning is given.

    Parameters
    ----------
    key : str
        Name of the option to be deprecated.
        must be a fully-qualified option name (e.g "x.y.z.rkey").
    msg : str, optional
        Warning message to output when the key is referenced.
        if no message is given a default message will be emitted.
    rkey : str, optional
        Name of an option to reroute access to.
        If specified, any referenced `key` will be
        re-routed to `rkey` including set/get/reset.
        rkey must be a fully-qualified option name (e.g "x.y.z.rkey").
        used by the default message if no `msg` is specified.
    removal_ver : str, optional
        Specifies the version in which this option will
        be removed. used by the default message if no `msg` is specified.

    Raises
    ------
    OptionError
        If the specified key has already been deprecated.
    """
    key = key.lower()

    if key in _deprecated_options:
        raise OptionError(f"Option '{key}' has already been defined as deprecated.")

    _deprecated_options[key] = DeprecatedOption(key, msg, rkey, removal_ver)


#
# functions internal to the module


def _select_options(pat: str) -> list[str]:
    """
    returns a list of keys matching `pat`

    if pat=="all", returns all registered options
    """
    # short-circuit for exact key
    if pat in _registered_options:
        return [pat]

    # else look through all of them
    keys = sorted(_registered_options.keys())
    if pat == "all":  # reserved key
        return keys

    return [k for k in keys if re.search(pat, k, re.I)]


def _get_root(key: str) -> tuple[dict[str, Any], str]:
    path = key.split(".")
    cursor = _global_config
    for p in path[:-1]:
        cursor = cursor[p]
    return cursor, path[-1]


def _is_deprecated(key: str) -> bool:
    """Returns True if the given option has been deprecated"""
    key = key.lower()
    return key in _deprecated_options


def _get_deprecated_option(key: str):
    """
    Retrieves the metadata for a deprecated option, if `key` is deprecated.

    Returns
    -------
    DeprecatedOption (namedtuple) if key is deprecated, None otherwise
    """
    try:
        d = _deprecated_options[key]
    except KeyError:
        return None
    else:
        return d


def _get_registered_option(key: str):
    """
    Retrieves the option metadata if `key` is a registered option.

    Returns
    -------
    RegisteredOption (namedtuple) if key is deprecated, None otherwise
    """
    return _registered_options.get(key)


def _translate_key(key: str) -> str:
    """
    if key id deprecated and a replacement key defined, will return the
    replacement key, otherwise returns `key` as - is
    """
    d = _get_deprecated_option(key)
    if d:
        return d.rkey or key
    else:
        return key


def _warn_if_deprecated(key: str) -> bool:
    """
    Checks if `key` is a deprecated option and if so, prints a warning.

    Returns
    -------
    bool - True if `key` is deprecated, False otherwise.
    """
    d = _get_deprecated_option(key)
    if d:
        if d.msg:
            warnings.warn(
                d.msg,
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        else:
            msg = f"'{key}' is deprecated"
            if d.removal_ver:
                msg += f" and will be removed in {d.removal_ver}"
            if d.rkey:
                msg += f", please use '{d.rkey}' instead."
            else:
                msg += ", please refrain from using it."

            warnings.warn(msg, FutureWarning, stacklevel=find_stack_level())
        return True
    return False


def _build_option_description(k: str) -> str:
    """Builds a formatted description of a registered option and prints it"""
    o = _get_registered_option(k)
    d = _get_deprecated_option(k)

    s = f"{k} "

    if o.doc:
        s += "\n".join(o.doc.strip().split("\n"))
    else:
        s += "No description available."

    if o:
        s += f"\n    [default: {o.defval}] [currently: {_get_option(k, True)}]"

    if d:
        rkey = d.rkey or ""
        s += "\n    (Deprecated"
        s += f", use `{rkey}` instead."
        s += ")"

    return s


def pp_options_list(keys: Iterable[str], width: int = 80, _print: bool = False):
    """Builds a concise listing of available options, grouped by prefix"""
    from itertools import groupby
    from textwrap import wrap

    def pp(name: str, ks: Iterable[str]) -> list[str]:
        pfx = "- " + name + ".[" if name else ""
        ls = wrap(
            ", ".join(ks),
            width,
            initial_indent=pfx,
            subsequent_indent="  ",
            break_long_words=False,
        )
        if ls and ls[-1] and name:
            ls[-1] = ls[-1] + "]"
        return ls

    ls: list[str] = []
    singles = [x for x in sorted(keys) if x.find(".") < 0]
    if singles:
        ls += pp("", singles)
    keys = [x for x in keys if x.find(".") >= 0]

    for k, g in groupby(sorted(keys), lambda x: x[: x.rfind(".")]):
        ks = [x[len(k) + 1 :] for x in list(g)]
        ls += pp(k, ks)
    s = "\n".join(ls)
    if _print:
        print(s)
    else:
        return s


#
# helpers


@contextmanager
def config_prefix(prefix: str) -> Generator[None, None, None]:
    """
    contextmanager for multiple invocations of API with a common prefix

    supported API functions: (register / get / set )__option

    Warning: This is not thread - safe, and won't work properly if you import
    the API functions into your module using the "from x import y" construct.

    Example
    -------
    import pandas._config.config as cf
    with cf.config_prefix("display.font"):
        cf.register_option("color", "red")
        cf.register_option("size", " 5 pt")
        cf.set_option(size, " 6 pt")
        cf.get_option(size)
        ...

        etc'

    will register options "display.font.color", "display.font.size", set the
    value of "display.font.size"... and so on.
    """
    # Note: reset_option relies on set_option, and on key directly
    # it does not fit in to this monkey-patching scheme

    global register_option, get_option, set_option

    def wrap(func: F) -> F:
        def inner(key: str, *args, **kwds):
            pkey = f"{prefix}.{key}"
            return func(pkey, *args, **kwds)

        return cast(F, inner)

    _register_option = register_option
    _get_option = get_option
    _set_option = set_option
    set_option = wrap(set_option)
    get_option = wrap(get_option)
    register_option = wrap(register_option)
    try:
        yield
    finally:
        set_option = _set_option
        get_option = _get_option
        register_option = _register_option


# These factories and methods are handy for use as the validator
# arg in register_option


def is_type_factory(_type: type[Any]) -> Callable[[Any], None]:
    """

    Parameters
    ----------
    `_type` - a type to be compared against (e.g. type(x) == `_type`)

    Returns
    -------
    validator - a function of a single argument x , which raises
                ValueError if type(x) is not equal to `_type`

    """

    def inner(x) -> None:
        if type(x) != _type:
            raise ValueError(f"Value must have type '{_type}'")

    return inner


def is_instance_factory(_type) -> Callable[[Any], None]:
    """

    Parameters
    ----------
    `_type` - the type to be checked against

    Returns
    -------
    validator - a function of a single argument x , which raises
                ValueError if x is not an instance of `_type`

    """
    if isinstance(_type, (tuple, list)):
        _type = tuple(_type)
        type_repr = "|".join(map(str, _type))
    else:
        type_repr = f"'{_type}'"

    def inner(x) -> None:
        if not isinstance(x, _type):
            raise ValueError(f"Value must be an instance of {type_repr}")

    return inner


def is_one_of_factory(legal_values) -> Callable[[Any], None]:
    callables = [c for c in legal_values if callable(c)]
    legal_values = [c for c in legal_values if not callable(c)]

    def inner(x) -> None:
        if x not in legal_values:
            if not any(c(x) for c in callables):
                uvals = [str(lval) for lval in legal_values]
                pp_values = "|".join(uvals)
                msg = f"Value must be one of {pp_values}"
                if len(callables):
                    msg += " or a callable"
                raise ValueError(msg)

    return inner


def is_nonnegative_int(value: object) -> None:
    """
    Verify that value is None or a positive int.

    Parameters
    ----------
    value : None or int
            The `value` to be checked.

    Raises
    ------
    ValueError
        When the value is not None or is a negative integer
    """
    if value is None:
        return

    elif isinstance(value, int):
        if value >= 0:
            return

    msg = "Value must be a nonnegative integer or None"
    raise ValueError(msg)


# common type validators, for convenience
# usage: register_option(... , validator = is_int)
is_int = is_type_factory(int)
is_bool = is_type_factory(bool)
is_float = is_type_factory(float)
is_str = is_type_factory(str)
is_text = is_instance_factory((str, bytes))


def is_callable(obj) -> bool:
    """

    Parameters
    ----------
    `obj` - the object to be checked

    Returns
    -------
    validator - returns True if object is callable
        raises ValueError otherwise.

    """
    if not callable(obj):
        raise ValueError("Value must be a callable")
    return True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\oracle\oracledb.py ===
# dialects/oracle/oracledb.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

r""".. dialect:: oracle+oracledb
    :name: python-oracledb
    :dbapi: oracledb
    :connectstring: oracle+oracledb://user:pass@hostname:port[/dbname][?service_name=<service>[&key=value&key=value...]]
    :url: https://oracle.github.io/python-oracledb/

Description
-----------

Python-oracledb is the Oracle Database driver for Python. It features a default
"thin" client mode that requires no dependencies, and an optional "thick" mode
that uses Oracle Client libraries.  It supports SQLAlchemy features including
two phase transactions and Asyncio.

Python-oracle is the renamed, updated cx_Oracle driver. Oracle is no longer
doing any releases in the cx_Oracle namespace.

The SQLAlchemy ``oracledb`` dialect provides both a sync and an async
implementation under the same dialect name. The proper version is
selected depending on how the engine is created:

* calling :func:`_sa.create_engine` with ``oracle+oracledb://...`` will
  automatically select the sync version::

    from sqlalchemy import create_engine

    sync_engine = create_engine(
        "oracle+oracledb://scott:tiger@localhost?service_name=FREEPDB1"
    )

* calling :func:`_asyncio.create_async_engine` with ``oracle+oracledb://...``
  will automatically select the async version::

    from sqlalchemy.ext.asyncio import create_async_engine

    asyncio_engine = create_async_engine(
        "oracle+oracledb://scott:tiger@localhost?service_name=FREEPDB1"
    )

  The asyncio version of the dialect may also be specified explicitly using the
  ``oracledb_async`` suffix::

      from sqlalchemy.ext.asyncio import create_async_engine

      asyncio_engine = create_async_engine(
          "oracle+oracledb_async://scott:tiger@localhost?service_name=FREEPDB1"
      )

.. versionadded:: 2.0.25 added support for the async version of oracledb.

Thick mode support
------------------

By default, the python-oracledb driver runs in a "thin" mode that does not
require Oracle Client libraries to be installed. The driver also supports a
"thick" mode that uses Oracle Client libraries to get functionality such as
Oracle Application Continuity.

To enable thick mode, call `oracledb.init_oracle_client()
<https://python-oracledb.readthedocs.io/en/latest/api_manual/module.html#oracledb.init_oracle_client>`_
explicitly, or pass the parameter ``thick_mode=True`` to
:func:`_sa.create_engine`. To pass custom arguments to
``init_oracle_client()``, like the ``lib_dir`` path, a dict may be passed, for
example::

    engine = sa.create_engine(
        "oracle+oracledb://...",
        thick_mode={
            "lib_dir": "/path/to/oracle/client/lib",
            "config_dir": "/path/to/network_config_file_directory",
            "driver_name": "my-app : 1.0.0",
        },
    )

Note that passing a ``lib_dir`` path should only be done on macOS or
Windows. On Linux it does not behave as you might expect.

.. seealso::

    python-oracledb documentation `Enabling python-oracledb Thick mode
    <https://python-oracledb.readthedocs.io/en/latest/user_guide/initialization.html#enabling-python-oracledb-thick-mode>`_

Connecting to Oracle Database
-----------------------------

python-oracledb provides several methods of indicating the target database.
The dialect translates from a series of different URL forms.

Given the hostname, port and service name of the target database, you can
connect in SQLAlchemy using the ``service_name`` query string parameter::

    engine = create_engine(
        "oracle+oracledb://scott:tiger@hostname:port?service_name=myservice"
    )

Connecting with Easy Connect strings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can pass any valid python-oracledb connection string as the ``dsn`` key
value in a :paramref:`_sa.create_engine.connect_args` dictionary.  See
python-oracledb documentation `Oracle Net Services Connection Strings
<https://python-oracledb.readthedocs.io/en/latest/user_guide/connection_handling.html#oracle-net-services-connection-strings>`_.

For example to use an `Easy Connect string
<https://download.oracle.com/ocomdocs/global/Oracle-Net-Easy-Connect-Plus.pdf>`_
with a timeout to prevent connection establishment from hanging if the network
transport to the database cannot be establishd in 30 seconds, and also setting
a keep-alive time of 60 seconds to stop idle network connections from being
terminated by a firewall::

    e = create_engine(
        "oracle+oracledb://@",
        connect_args={
            "user": "scott",
            "password": "tiger",
            "dsn": "hostname:port/myservice?transport_connect_timeout=30&expire_time=60",
        },
    )

The Easy Connect syntax has been enhanced during the life of Oracle Database.
Review the documentation for your database version.  The current documentation
is at `Understanding the Easy Connect Naming Method
<https://www.oracle.com/pls/topic/lookup?ctx=dblatest&id=GUID-B0437826-43C1-49EC-A94D-B650B6A4A6EE>`_.

The general syntax is similar to:

.. sourcecode:: text

    [[protocol:]//]host[:port][/[service_name]][?parameter_name=value{&parameter_name=value}]

Note that although the SQLAlchemy URL syntax ``hostname:port/dbname`` looks
like Oracle's Easy Connect syntax, it is different. SQLAlchemy's URL requires a
system identifier (SID) for the ``dbname`` component::

    engine = create_engine("oracle+oracledb://scott:tiger@hostname:port/sid")

Easy Connect syntax does not support SIDs. It uses services names, which are
the preferred choice for connecting to Oracle Database.

Passing python-oracledb connect arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Other python-oracledb driver `connection options
<https://python-oracledb.readthedocs.io/en/latest/api_manual/module.html#oracledb.connect>`_
can be passed in ``connect_args``.  For example::

    e = create_engine(
        "oracle+oracledb://@",
        connect_args={
            "user": "scott",
            "password": "tiger",
            "dsn": "hostname:port/myservice",
            "events": True,
            "mode": oracledb.AUTH_MODE_SYSDBA,
        },
    )

Connecting with tnsnames.ora TNS aliases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If no port, database name, or service name is provided, the dialect will use an
Oracle Database DSN "connection string".  This takes the "hostname" portion of
the URL as the data source name.  For example, if the ``tnsnames.ora`` file
contains a `TNS Alias
<https://python-oracledb.readthedocs.io/en/latest/user_guide/connection_handling.html#tns-aliases-for-connection-strings>`_
of ``myalias`` as below:

.. sourcecode:: text

    myalias =
      (DESCRIPTION =
        (ADDRESS = (PROTOCOL = TCP)(HOST = mymachine.example.com)(PORT = 1521))
        (CONNECT_DATA =
          (SERVER = DEDICATED)
          (SERVICE_NAME = orclpdb1)
        )
      )

The python-oracledb dialect connects to this database service when ``myalias`` is the
hostname portion of the URL, without specifying a port, database name or
``service_name``::

    engine = create_engine("oracle+oracledb://scott:tiger@myalias")

Connecting to Oracle Autonomous Database
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users of Oracle Autonomous Database should use either use the TNS Alias URL
shown above, or pass the TNS Alias as the ``dsn`` key value in a
:paramref:`_sa.create_engine.connect_args` dictionary.

If Oracle Autonomous Database is configured for mutual TLS ("mTLS")
connections, then additional configuration is required as shown in `Connecting
to Oracle Cloud Autonomous Databases
<https://python-oracledb.readthedocs.io/en/latest/user_guide/connection_handling.html#connecting-to-oracle-cloud-autonomous-databases>`_. In
summary, Thick mode users should configure file locations and set the wallet
path in ``sqlnet.ora`` appropriately::

    e = create_engine(
        "oracle+oracledb://@",
        thick_mode={
            # directory containing tnsnames.ora and cwallet.so
            "config_dir": "/opt/oracle/wallet_dir",
        },
        connect_args={
            "user": "scott",
            "password": "tiger",
            "dsn": "mydb_high",
        },
    )

Thin mode users of mTLS should pass the appropriate directories and PEM wallet
password when creating the engine, similar to::

    e = create_engine(
        "oracle+oracledb://@",
        connect_args={
            "user": "scott",
            "password": "tiger",
            "dsn": "mydb_high",
            "config_dir": "/opt/oracle/wallet_dir",  # directory containing tnsnames.ora
            "wallet_location": "/opt/oracle/wallet_dir",  # directory containing ewallet.pem
            "wallet_password": "top secret",  # password for the PEM file
        },
    )

Typically ``config_dir`` and ``wallet_location`` are the same directory, which
is where the Oracle Autonomous Database wallet zip file was extracted.  Note
this directory should be protected.

Connection Pooling
------------------

Applications with multiple concurrent users should use connection pooling. A
minimal sized connection pool is also beneficial for long-running, single-user
applications that do not frequently use a connection.

The python-oracledb driver provides its own connection pool implementation that
may be used in place of SQLAlchemy's pooling functionality.  The driver pool
gives support for high availability features such as dead connection detection,
connection draining for planned database downtime, support for Oracle
Application Continuity and Transparent Application Continuity, and gives
support for `Database Resident Connection Pooling (DRCP)
<https://python-oracledb.readthedocs.io/en/latest/user_guide/connection_handling.html#database-resident-connection-pooling-drcp>`_.

To take advantage of python-oracledb's pool, use the
:paramref:`_sa.create_engine.creator` parameter to provide a function that
returns a new connection, along with setting
:paramref:`_sa.create_engine.pool_class` to ``NullPool`` to disable
SQLAlchemy's pooling::

    import oracledb
    from sqlalchemy import create_engine
    from sqlalchemy import text
    from sqlalchemy.pool import NullPool

    # Uncomment to use the optional python-oracledb Thick mode.
    # Review the python-oracledb doc for the appropriate parameters
    # oracledb.init_oracle_client(<your parameters>)

    pool = oracledb.create_pool(
        user="scott",
        password="tiger",
        dsn="localhost:1521/freepdb1",
        min=1,
        max=4,
        increment=1,
    )
    engine = create_engine(
        "oracle+oracledb://", creator=pool.acquire, poolclass=NullPool
    )

The above engine may then be used normally. Internally, python-oracledb handles
connection pooling::

    with engine.connect() as conn:
        print(conn.scalar(text("select 1 from dual")))

Refer to the python-oracledb documentation for `oracledb.create_pool()
<https://python-oracledb.readthedocs.io/en/latest/api_manual/module.html#oracledb.create_pool>`_
for the arguments that can be used when creating a connection pool.

.. _drcp:

Using Oracle Database Resident Connection Pooling (DRCP)
--------------------------------------------------------

When using Oracle Database's Database Resident Connection Pooling (DRCP), the
best practice is to specify a connection class and "purity". Refer to the
`python-oracledb documentation on DRCP
<https://python-oracledb.readthedocs.io/en/latest/user_guide/connection_handling.html#database-resident-connection-pooling-drcp>`_.
For example::

    import oracledb
    from sqlalchemy import create_engine
    from sqlalchemy import text
    from sqlalchemy.pool import NullPool

    # Uncomment to use the optional python-oracledb Thick mode.
    # Review the python-oracledb doc for the appropriate parameters
    # oracledb.init_oracle_client(<your parameters>)

    pool = oracledb.create_pool(
        user="scott",
        password="tiger",
        dsn="localhost:1521/freepdb1",
        min=1,
        max=4,
        increment=1,
        cclass="MYCLASS",
        purity=oracledb.PURITY_SELF,
    )
    engine = create_engine(
        "oracle+oracledb://", creator=pool.acquire, poolclass=NullPool
    )

The above engine may then be used normally where python-oracledb handles
application connection pooling and Oracle Database additionally uses DRCP::

    with engine.connect() as conn:
        print(conn.scalar(text("select 1 from dual")))

If you wish to use different connection classes or purities for different
connections, then wrap ``pool.acquire()``::

    import oracledb
    from sqlalchemy import create_engine
    from sqlalchemy import text
    from sqlalchemy.pool import NullPool

    # Uncomment to use python-oracledb Thick mode.
    # Review the python-oracledb doc for the appropriate parameters
    # oracledb.init_oracle_client(<your parameters>)

    pool = oracledb.create_pool(
        user="scott",
        password="tiger",
        dsn="localhost:1521/freepdb1",
        min=1,
        max=4,
        increment=1,
        cclass="MYCLASS",
        purity=oracledb.PURITY_SELF,
    )


    def creator():
        return pool.acquire(cclass="MYOTHERCLASS", purity=oracledb.PURITY_NEW)


    engine = create_engine(
        "oracle+oracledb://", creator=creator, poolclass=NullPool
    )

Engine Options consumed by the SQLAlchemy oracledb dialect outside of the driver
--------------------------------------------------------------------------------

There are also options that are consumed by the SQLAlchemy oracledb dialect
itself.  These options are always passed directly to :func:`_sa.create_engine`,
such as::

    e = create_engine("oracle+oracledb://user:pass@tnsalias", arraysize=500)

The parameters accepted by the oracledb dialect are as follows:

* ``arraysize`` - set the driver cursor.arraysize value. It defaults to
  ``None``, indicating that the driver default value of 100 should be used.
  This setting controls how many rows are buffered when fetching rows, and can
  have a significant effect on performance if increased for queries that return
  large numbers of rows.

  .. versionchanged:: 2.0.26 - changed the default value from 50 to None,
    to use the default value of the driver itself.

* ``auto_convert_lobs`` - defaults to True; See :ref:`oracledb_lob`.

* ``coerce_to_decimal`` - see :ref:`oracledb_numeric` for detail.

* ``encoding_errors`` - see :ref:`oracledb_unicode_encoding_errors` for detail.

.. _oracledb_unicode:

Unicode
-------

As is the case for all DBAPIs under Python 3, all strings are inherently
Unicode strings.

Ensuring the Correct Client Encoding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In python-oracledb, the encoding used for all character data is "UTF-8".

Unicode-specific Column datatypes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Core expression language handles unicode data by use of the
:class:`.Unicode` and :class:`.UnicodeText` datatypes.  These types correspond
to the VARCHAR2 and CLOB Oracle Database datatypes by default.  When using
these datatypes with Unicode data, it is expected that the database is
configured with a Unicode-aware character set so that the VARCHAR2 and CLOB
datatypes can accommodate the data.

In the case that Oracle Database is not configured with a Unicode character
set, the two options are to use the :class:`_types.NCHAR` and
:class:`_oracle.NCLOB` datatypes explicitly, or to pass the flag
``use_nchar_for_unicode=True`` to :func:`_sa.create_engine`, which will cause
the SQLAlchemy dialect to use NCHAR/NCLOB for the :class:`.Unicode` /
:class:`.UnicodeText` datatypes instead of VARCHAR/CLOB.

.. versionchanged:: 1.3 The :class:`.Unicode` and :class:`.UnicodeText`
   datatypes now correspond to the ``VARCHAR2`` and ``CLOB`` Oracle Database
   datatypes unless the ``use_nchar_for_unicode=True`` is passed to the dialect
   when :func:`_sa.create_engine` is called.


.. _oracledb_unicode_encoding_errors:

Encoding Errors
^^^^^^^^^^^^^^^

For the unusual case that data in Oracle Database is present with a broken
encoding, the dialect accepts a parameter ``encoding_errors`` which will be
passed to Unicode decoding functions in order to affect how decoding errors are
handled.  The value is ultimately consumed by the Python `decode
<https://docs.python.org/3/library/stdtypes.html#bytes.decode>`_ function, and
is passed both via python-oracledb's ``encodingErrors`` parameter consumed by
``Cursor.var()``, as well as SQLAlchemy's own decoding function, as the
python-oracledb dialect makes use of both under different circumstances.

.. versionadded:: 1.3.11


.. _oracledb_setinputsizes:

Fine grained control over python-oracledb data binding with setinputsizes
-------------------------------------------------------------------------

The python-oracle DBAPI has a deep and fundamental reliance upon the usage of
the DBAPI ``setinputsizes()`` call.  The purpose of this call is to establish
the datatypes that are bound to a SQL statement for Python values being passed
as parameters.  While virtually no other DBAPI assigns any use to the
``setinputsizes()`` call, the python-oracledb DBAPI relies upon it heavily in
its interactions with the Oracle Database, and in some scenarios it is not
possible for SQLAlchemy to know exactly how data should be bound, as some
settings can cause profoundly different performance characteristics, while
altering the type coercion behavior at the same time.

Users of the oracledb dialect are **strongly encouraged** to read through
python-oracledb's list of built-in datatype symbols at `Database Types
<https://python-oracledb.readthedocs.io/en/latest/api_manual/module.html#database-types>`_
Note that in some cases, significant performance degradation can occur when
using these types vs. not.

On the SQLAlchemy side, the :meth:`.DialectEvents.do_setinputsizes` event can
be used both for runtime visibility (e.g. logging) of the setinputsizes step as
well as to fully control how ``setinputsizes()`` is used on a per-statement
basis.

.. versionadded:: 1.2.9 Added :meth:`.DialectEvents.setinputsizes`


Example 1 - logging all setinputsizes calls
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following example illustrates how to log the intermediary values from a
SQLAlchemy perspective before they are converted to the raw ``setinputsizes()``
parameter dictionary.  The keys of the dictionary are :class:`.BindParameter`
objects which have a ``.key`` and a ``.type`` attribute::

    from sqlalchemy import create_engine, event

    engine = create_engine(
        "oracle+oracledb://scott:tiger@localhost:1521?service_name=freepdb1"
    )


    @event.listens_for(engine, "do_setinputsizes")
    def _log_setinputsizes(inputsizes, cursor, statement, parameters, context):
        for bindparam, dbapitype in inputsizes.items():
            log.info(
                "Bound parameter name: %s  SQLAlchemy type: %r DBAPI object: %s",
                bindparam.key,
                bindparam.type,
                dbapitype,
            )

Example 2 - remove all bindings to CLOB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For performance, fetching LOB datatypes from Oracle Database is set by default
for the ``Text`` type within SQLAlchemy.  This setting can be modified as
follows::


    from sqlalchemy import create_engine, event
    from oracledb import CLOB

    engine = create_engine(
        "oracle+oracledb://scott:tiger@localhost:1521?service_name=freepdb1"
    )


    @event.listens_for(engine, "do_setinputsizes")
    def _remove_clob(inputsizes, cursor, statement, parameters, context):
        for bindparam, dbapitype in list(inputsizes.items()):
            if dbapitype is CLOB:
                del inputsizes[bindparam]

.. _oracledb_lob:

LOB Datatypes
--------------

LOB datatypes refer to the "large object" datatypes such as CLOB, NCLOB and
BLOB. Oracle Database can efficiently return these datatypes as a single
buffer. SQLAlchemy makes use of type handlers to do this by default.

To disable the use of the type handlers and deliver LOB objects as classic
buffered objects with a ``read()`` method, the parameter
``auto_convert_lobs=False`` may be passed to :func:`_sa.create_engine`.

.. _oracledb_returning:

RETURNING Support
-----------------

The oracledb dialect implements RETURNING using OUT parameters.  The dialect
supports RETURNING fully.

Two Phase Transaction Support
-----------------------------

Two phase transactions are fully supported with python-oracledb. (Thin mode
requires python-oracledb 2.3).  APIs for two phase transactions are provided at
the Core level via :meth:`_engine.Connection.begin_twophase` and
:paramref:`_orm.Session.twophase` for transparent ORM use.

.. versionchanged:: 2.0.32 added support for two phase transactions

.. _oracledb_numeric:

Precision Numerics
------------------

SQLAlchemy's numeric types can handle receiving and returning values as Python
``Decimal`` objects or float objects.  When a :class:`.Numeric` object, or a
subclass such as :class:`.Float`, :class:`_oracle.DOUBLE_PRECISION` etc. is in
use, the :paramref:`.Numeric.asdecimal` flag determines if values should be
coerced to ``Decimal`` upon return, or returned as float objects.  To make
matters more complicated under Oracle Database, the ``NUMBER`` type can also
represent integer values if the "scale" is zero, so the Oracle
Database-specific :class:`_oracle.NUMBER` type takes this into account as well.

The oracledb dialect makes extensive use of connection- and cursor-level
"outputtypehandler" callables in order to coerce numeric values as requested.
These callables are specific to the specific flavor of :class:`.Numeric` in
use, as well as if no SQLAlchemy typing objects are present.  There are
observed scenarios where Oracle Database may send incomplete or ambiguous
information about the numeric types being returned, such as a query where the
numeric types are buried under multiple levels of subquery.  The type handlers
do their best to make the right decision in all cases, deferring to the
underlying python-oracledb DBAPI for all those cases where the driver can make
the best decision.

When no typing objects are present, as when executing plain SQL strings, a
default "outputtypehandler" is present which will generally return numeric
values which specify precision and scale as Python ``Decimal`` objects.  To
disable this coercion to decimal for performance reasons, pass the flag
``coerce_to_decimal=False`` to :func:`_sa.create_engine`::

    engine = create_engine(
        "oracle+oracledb://scott:tiger@tnsalias", coerce_to_decimal=False
    )

The ``coerce_to_decimal`` flag only impacts the results of plain string
SQL statements that are not otherwise associated with a :class:`.Numeric`
SQLAlchemy type (or a subclass of such).

.. versionchanged:: 1.2 The numeric handling system for the oracle dialects has
   been reworked to take advantage of newer driver features as well as better
   integration of outputtypehandlers.

.. versionadded:: 2.0.0 added support for the python-oracledb driver.

"""  # noqa
from __future__ import annotations

import collections
import re
from typing import Any
from typing import TYPE_CHECKING

from . import cx_oracle as _cx_oracle
from ... import exc
from ... import pool
from ...connectors.asyncio import AsyncAdapt_dbapi_connection
from ...connectors.asyncio import AsyncAdapt_dbapi_cursor
from ...connectors.asyncio import AsyncAdapt_dbapi_ss_cursor
from ...connectors.asyncio import AsyncAdaptFallback_dbapi_connection
from ...engine import default
from ...util import asbool
from ...util import await_fallback
from ...util import await_only

if TYPE_CHECKING:
    from oracledb import AsyncConnection
    from oracledb import AsyncCursor


class OracleExecutionContext_oracledb(
    _cx_oracle.OracleExecutionContext_cx_oracle
):
    pass


class OracleDialect_oracledb(_cx_oracle.OracleDialect_cx_oracle):
    supports_statement_cache = True
    execution_ctx_cls = OracleExecutionContext_oracledb

    driver = "oracledb"
    _min_version = (1,)

    def __init__(
        self,
        auto_convert_lobs=True,
        coerce_to_decimal=True,
        arraysize=None,
        encoding_errors=None,
        thick_mode=None,
        **kwargs,
    ):
        super().__init__(
            auto_convert_lobs,
            coerce_to_decimal,
            arraysize,
            encoding_errors,
            **kwargs,
        )

        if self.dbapi is not None and (
            thick_mode or isinstance(thick_mode, dict)
        ):
            kw = thick_mode if isinstance(thick_mode, dict) else {}
            self.dbapi.init_oracle_client(**kw)

    @classmethod
    def import_dbapi(cls):
        import oracledb

        return oracledb

    @classmethod
    def is_thin_mode(cls, connection):
        return connection.connection.dbapi_connection.thin

    @classmethod
    def get_async_dialect_cls(cls, url):
        return OracleDialectAsync_oracledb

    def _load_version(self, dbapi_module):
        version = (0, 0, 0)
        if dbapi_module is not None:
            m = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", dbapi_module.version)
            if m:
                version = tuple(
                    int(x) for x in m.group(1, 2, 3) if x is not None
                )
        self.oracledb_ver = version
        if (
            self.oracledb_ver > (0, 0, 0)
            and self.oracledb_ver < self._min_version
        ):
            raise exc.InvalidRequestError(
                f"oracledb version {self._min_version} and above are supported"
            )

    def do_begin_twophase(self, connection, xid):
        conn_xis = connection.connection.xid(*xid)
        connection.connection.tpc_begin(conn_xis)
        connection.connection.info["oracledb_xid"] = conn_xis

    def do_prepare_twophase(self, connection, xid):
        should_commit = connection.connection.tpc_prepare()
        connection.info["oracledb_should_commit"] = should_commit

    def do_rollback_twophase(
        self, connection, xid, is_prepared=True, recover=False
    ):
        if recover:
            conn_xid = connection.connection.xid(*xid)
        else:
            conn_xid = None
        connection.connection.tpc_rollback(conn_xid)

    def do_commit_twophase(
        self, connection, xid, is_prepared=True, recover=False
    ):
        conn_xid = None
        if not is_prepared:
            should_commit = connection.connection.tpc_prepare()
        elif recover:
            conn_xid = connection.connection.xid(*xid)
            should_commit = True
        else:
            should_commit = connection.info["oracledb_should_commit"]
        if should_commit:
            connection.connection.tpc_commit(conn_xid)

    def do_recover_twophase(self, connection):
        return [
            # oracledb seems to return bytes
            (
                fi,
                gti.decode() if isinstance(gti, bytes) else gti,
                bq.decode() if isinstance(bq, bytes) else bq,
            )
            for fi, gti, bq in connection.connection.tpc_recover()
        ]

    def _check_max_identifier_length(self, connection):
        if self.oracledb_ver >= (2, 5):
            max_len = connection.connection.max_identifier_length
            if max_len is not None:
                return max_len
        return super()._check_max_identifier_length(connection)


class AsyncAdapt_oracledb_cursor(AsyncAdapt_dbapi_cursor):
    _cursor: AsyncCursor
    __slots__ = ()

    @property
    def outputtypehandler(self):
        return self._cursor.outputtypehandler

    @outputtypehandler.setter
    def outputtypehandler(self, value):
        self._cursor.outputtypehandler = value

    def var(self, *args, **kwargs):
        return self._cursor.var(*args, **kwargs)

    def close(self):
        self._rows.clear()
        self._cursor.close()

    def setinputsizes(self, *args: Any, **kwargs: Any) -> Any:
        return self._cursor.setinputsizes(*args, **kwargs)

    def _aenter_cursor(self, cursor: AsyncCursor) -> AsyncCursor:
        try:
            return cursor.__enter__()
        except Exception as error:
            self._adapt_connection._handle_exception(error)

    async def _execute_async(self, operation, parameters):
        # override to not use mutex, oracledb already has a mutex

        if parameters is None:
            result = await self._cursor.execute(operation)
        else:
            result = await self._cursor.execute(operation, parameters)

        if self._cursor.description and not self.server_side:
            self._rows = collections.deque(await self._cursor.fetchall())
        return result

    async def _executemany_async(
        self,
        operation,
        seq_of_parameters,
    ):
        # override to not use mutex, oracledb already has a mutex
        return await self._cursor.executemany(operation, seq_of_parameters)

    def __enter__(self):
        return self

    def __exit__(self, type_: Any, value: Any, traceback: Any) -> None:
        self.close()


class AsyncAdapt_oracledb_ss_cursor(
    AsyncAdapt_dbapi_ss_cursor, AsyncAdapt_oracledb_cursor
):
    __slots__ = ()

    def close(self) -> None:
        if self._cursor is not None:
            self._cursor.close()
            self._cursor = None  # type: ignore


class AsyncAdapt_oracledb_connection(AsyncAdapt_dbapi_connection):
    _connection: AsyncConnection
    __slots__ = ()

    thin = True

    _cursor_cls = AsyncAdapt_oracledb_cursor
    _ss_cursor_cls = None

    @property
    def autocommit(self):
        return self._connection.autocommit

    @autocommit.setter
    def autocommit(self, value):
        self._connection.autocommit = value

    @property
    def outputtypehandler(self):
        return self._connection.outputtypehandler

    @outputtypehandler.setter
    def outputtypehandler(self, value):
        self._connection.outputtypehandler = value

    @property
    def version(self):
        return self._connection.version

    @property
    def stmtcachesize(self):
        return self._connection.stmtcachesize

    @stmtcachesize.setter
    def stmtcachesize(self, value):
        self._connection.stmtcachesize = value

    @property
    def max_identifier_length(self):
        return self._connection.max_identifier_length

    def cursor(self):
        return AsyncAdapt_oracledb_cursor(self)

    def ss_cursor(self):
        return AsyncAdapt_oracledb_ss_cursor(self)

    def xid(self, *args: Any, **kwargs: Any) -> Any:
        return self._connection.xid(*args, **kwargs)

    def tpc_begin(self, *args: Any, **kwargs: Any) -> Any:
        return self.await_(self._connection.tpc_begin(*args, **kwargs))

    def tpc_commit(self, *args: Any, **kwargs: Any) -> Any:
        return self.await_(self._connection.tpc_commit(*args, **kwargs))

    def tpc_prepare(self, *args: Any, **kwargs: Any) -> Any:
        return self.await_(self._connection.tpc_prepare(*args, **kwargs))

    def tpc_recover(self, *args: Any, **kwargs: Any) -> Any:
        return self.await_(self._connection.tpc_recover(*args, **kwargs))

    def tpc_rollback(self, *args: Any, **kwargs: Any) -> Any:
        return self.await_(self._connection.tpc_rollback(*args, **kwargs))


class AsyncAdaptFallback_oracledb_connection(
    AsyncAdaptFallback_dbapi_connection, AsyncAdapt_oracledb_connection
):
    __slots__ = ()


class OracledbAdaptDBAPI:
    def __init__(self, oracledb) -> None:
        self.oracledb = oracledb

        for k, v in self.oracledb.__dict__.items():
            if k != "connect":
                self.__dict__[k] = v

    def connect(self, *arg, **kw):
        async_fallback = kw.pop("async_fallback", False)
        creator_fn = kw.pop("async_creator_fn", self.oracledb.connect_async)

        if asbool(async_fallback):
            return AsyncAdaptFallback_oracledb_connection(
                self, await_fallback(creator_fn(*arg, **kw))
            )

        else:
            return AsyncAdapt_oracledb_connection(
                self, await_only(creator_fn(*arg, **kw))
            )


class OracleExecutionContextAsync_oracledb(OracleExecutionContext_oracledb):
    # restore default create cursor
    create_cursor = default.DefaultExecutionContext.create_cursor

    def create_default_cursor(self):
        # copy of OracleExecutionContext_cx_oracle.create_cursor
        c = self._dbapi_connection.cursor()
        if self.dialect.arraysize:
            c.arraysize = self.dialect.arraysize

        return c

    def create_server_side_cursor(self):
        c = self._dbapi_connection.ss_cursor()
        if self.dialect.arraysize:
            c.arraysize = self.dialect.arraysize

        return c


class OracleDialectAsync_oracledb(OracleDialect_oracledb):
    is_async = True
    supports_server_side_cursors = True
    supports_statement_cache = True
    execution_ctx_cls = OracleExecutionContextAsync_oracledb

    _min_version = (2,)

    # thick_mode mode is not supported by asyncio, oracledb will raise
    @classmethod
    def import_dbapi(cls):
        import oracledb

        return OracledbAdaptDBAPI(oracledb)

    @classmethod
    def get_pool_class(cls, url):
        async_fallback = url.query.get("async_fallback", False)

        if asbool(async_fallback):
            return pool.FallbackAsyncAdaptedQueuePool
        else:
            return pool.AsyncAdaptedQueuePool

    def get_driver_connection(self, connection):
        return connection._connection


dialect = OracleDialect_oracledb
dialect_async = OracleDialectAsync_oracledb

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\benchmark.py ===
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Copyright 2018 The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmarking the inference of pretrained transformer models.
PyTorch/TorchScript benchmark is based on https://github.com/huggingface/transformers/blob/master/examples/benchmarks.py.
One difference is that random input_ids is generated in this benchmark.

For onnxruntime, this script will convert a pretrained model to ONNX, and optimize it when -o parameter is used.

Example commands:
    Export all models to ONNX, optimize and validate them:
        python benchmark.py -b 0 -o -v -i 1 2 3
    Run OnnxRuntime on GPU for all models:
        python benchmark.py -g
    Run OnnxRuntime on GPU for all models with fp32 optimization:
        python benchmark.py -g -o
    Run OnnxRuntime on GPU with fp16 optimization:
        python benchmark.py -g -o -p "fp16"
    Run TorchScript on GPU for all models:
        python benchmark.py -e torchscript -g
    Run TorchScript on GPU for all models with fp16:
        python benchmark.py -e torchscript -g -p "fp16"
    Run ONNXRuntime and TorchScript on CPU for all models with quantization:
        python benchmark.py -e torchscript onnxruntime -p "int8" -o
    Run OnnxRuntime with the ROCM provider and graph optimization script:
        python benchmark.py -g -m bert-base-cased --provider rocm --optimizer_info by_script --disable_embed_layer_norm
    Run OnnxRuntime with bfloat16 fastmath mode kernels on aarch64 platforms with bfloat16 support:
        python benchmark.py --enable_arm64_bfloat16_fastmath_mlas_gemm

It is recommended to use run_benchmark.sh to launch benchmark.
"""

import argparse
import logging
import os
import timeit
from datetime import datetime

import numpy
import psutil
from benchmark_helper import (
    ConfigModifier,
    OptimizerInfo,
    Precision,
    create_onnxruntime_session,
    get_latency_result,
    inference_ort,
    inference_ort_with_io_binding,
    output_details,
    output_fusion_statistics,
    output_summary,
    setup_logger,
)
from fusion_options import FusionOptions
from huggingface_models import MODEL_CLASSES, MODELS
from onnx_exporter import (
    create_onnxruntime_input,
    export_onnx_model_from_pt,
    export_onnx_model_from_tf,
    load_pretrained_model,
)
from packaging import version
from quantize_helper import QuantizeHelper

logger = logging.getLogger("")

cpu_count = psutil.cpu_count(logical=False)

# Set OMP environment variable before importing onnxruntime or torch.
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = str(cpu_count)

import torch  # noqa: E402
from transformers import AutoConfig, AutoTokenizer, LxmertConfig  # noqa: E402


def run_onnxruntime(
    use_gpu,
    provider,
    model_names,
    model_class,
    config_modifier,
    precision,
    num_threads,
    batch_sizes,
    sequence_lengths,
    repeat_times,
    input_counts,
    optimizer_info,
    validate_onnx,
    cache_dir,
    onnx_dir,
    verbose,
    overwrite,
    disable_ort_io_binding,
    use_raw_attention_mask,
    model_fusion_statistics,
    model_source,
    enable_arm64_bfloat16_fastmath_mlas_gemm,
    args,
):
    import onnxruntime

    results = []
    if (
        use_gpu
        and ("CUDAExecutionProvider" not in onnxruntime.get_available_providers())
        and ("MIGraphXExecutionProvider" not in onnxruntime.get_available_providers())
        and ("ROCMExecutionProvider" not in onnxruntime.get_available_providers())
        and ("DmlExecutionProvider" not in onnxruntime.get_available_providers())
    ):
        logger.error(
            "Please install onnxruntime-gpu or onnxruntime-directml package instead of onnxruntime, and use a machine with GPU for testing gpu performance."
        )
        return results

    warm_up_repeat = 0
    if provider == "tensorrt":
        optimizer_info = OptimizerInfo.NOOPT
        warm_up_repeat = 5
        if "TensorrtExecutionProvider" not in onnxruntime.get_available_providers():
            logger.error(
                "Please install onnxruntime-gpu-tensorrt package, and use a machine with GPU for testing gpu performance."
            )
            return results

    if optimizer_info == OptimizerInfo.NOOPT:
        logger.warning(
            f"OptimizerInfo is set to {optimizer_info}, graph optimizations specified in FusionOptions are not applied."
        )

    for model_name in model_names:
        all_input_names = MODELS[model_name][0]
        for num_inputs in input_counts:
            if num_inputs > len(all_input_names):
                break

            input_names = all_input_names[:num_inputs]
            args.model_type = MODELS[model_name][3]
            fusion_options = FusionOptions.parse(args)

            if "pt" in model_source:
                with torch.no_grad():
                    (
                        onnx_model_file,
                        is_valid_onnx_model,
                        vocab_size,
                        max_sequence_length,
                    ) = export_onnx_model_from_pt(
                        model_name,
                        MODELS[model_name][1],
                        MODELS[model_name][2],
                        MODELS[model_name][3],
                        model_class,
                        config_modifier,
                        cache_dir,
                        onnx_dir,
                        input_names,
                        use_gpu,
                        precision,
                        optimizer_info,
                        validate_onnx,
                        use_raw_attention_mask,
                        overwrite,
                        model_fusion_statistics,
                        fusion_options,
                    )
            if "tf" in model_source:
                (
                    onnx_model_file,
                    is_valid_onnx_model,
                    vocab_size,
                    max_sequence_length,
                ) = export_onnx_model_from_tf(
                    model_name,
                    MODELS[model_name][1],
                    MODELS[model_name][2],
                    MODELS[model_name][3],
                    model_class,
                    config_modifier,
                    cache_dir,
                    onnx_dir,
                    input_names,
                    use_gpu,
                    precision,
                    optimizer_info,
                    validate_onnx,
                    use_raw_attention_mask,
                    overwrite,
                    model_fusion_statistics,
                    fusion_options,
                )

            if not is_valid_onnx_model:
                continue

            ort_session = create_onnxruntime_session(
                onnx_model_file,
                use_gpu,
                provider,
                enable_all_optimization=True,
                num_threads=num_threads,
                verbose=verbose,
                enable_mlas_gemm_fastmath_arm64_bfloat16=enable_arm64_bfloat16_fastmath_mlas_gemm,
            )
            if ort_session is None:
                continue

            ort_output_names = [node_arg.name for node_arg in ort_session.get_outputs()]
            output_buffers = []
            device = "cuda" if use_gpu else "cpu"
            config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
            max_last_state_size = numpy.prod(
                [
                    max(batch_sizes),
                    max(sequence_lengths),
                    max(vocab_size, config.hidden_size),
                ]
            )
            max_pooler_size = numpy.prod([max(batch_sizes), config.hidden_size])
            for batch_size in batch_sizes:
                if batch_size <= 0:
                    continue
                for sequence_length in sequence_lengths:
                    if max_sequence_length is not None and sequence_length > max_sequence_length:
                        continue

                    input_value_type = numpy.int64 if "pt" in model_source else numpy.int32
                    ort_inputs = create_onnxruntime_input(
                        vocab_size,
                        batch_size,
                        sequence_length,
                        input_names,
                        config,
                        input_value_type,
                    )
                    result_template = {
                        "engine": "onnxruntime",
                        "version": onnxruntime.__version__,
                        "providers": provider,
                        "device": device,
                        "optimizer": optimizer_info,
                        "precision": precision,
                        "io_binding": not disable_ort_io_binding,
                        "model_name": model_name,
                        "inputs": num_inputs,
                        "threads": num_threads,
                        "batch_size": batch_size,
                        "sequence_length": sequence_length,
                        "custom_layer_num": config_modifier.get_layer_num(),
                        "datetime": str(datetime.now()),
                    }

                    if config.model_type in ["vit", "swin"]:
                        logger.info(
                            f"Run onnxruntime on {model_name} with input shape {[batch_size, 3, config.image_size, config.image_size]}"
                        )
                    else:
                        logger.info(f"Run onnxruntime on {model_name} with input shape {[batch_size, sequence_length]}")

                    if disable_ort_io_binding:
                        result = inference_ort(
                            ort_session,
                            ort_inputs,
                            result_template,
                            repeat_times,
                            batch_size,
                            warm_up_repeat,
                        )
                    else:
                        # Get output sizes from a dummy ort run
                        ort_outputs = ort_session.run(ort_output_names, ort_inputs)
                        output_buffer_max_sizes = [max_last_state_size]
                        for i in range(len(ort_outputs)):
                            if i == 2 and MODELS[model_name][3] == "gpt":
                                # past state output max size
                                output_buffer_max_sizes.append(max_pooler_size)
                            else:
                                output_buffer_max_sizes.append(max_last_state_size)

                        data_type = numpy.longlong if "pt" in model_source else numpy.intc
                        result = inference_ort_with_io_binding(
                            ort_session,
                            ort_inputs,
                            result_template,
                            repeat_times,
                            ort_output_names,
                            ort_outputs,
                            output_buffers,
                            output_buffer_max_sizes,
                            batch_size,
                            device,
                            data_type,
                            warm_up_repeat,
                        )
                    logger.info(result)
                    results.append(result)

    return results


def run_pytorch(
    use_gpu,
    model_names,
    model_class,
    config_modifier,
    precision,
    num_threads,
    batch_sizes,
    sequence_lengths,
    repeat_times,
    torchscript,
    torch2,
    cache_dir,
    verbose,
):
    results = []
    if use_gpu and not torch.cuda.is_available():
        logger.error("Please install PyTorch with Cuda, and use a machine with GPU for testing gpu performance.")
        return results

    torch.set_grad_enabled(False)

    for model_name in model_names:
        config = AutoConfig.from_pretrained(model_name, torchscript=torchscript, cache_dir=cache_dir)
        config_modifier.modify(config)
        model = load_pretrained_model(
            model_name,
            config=config,
            cache_dir=cache_dir,
            custom_model_class=model_class,
        )

        if config.model_type in ["vit", "swin"]:
            # These models don't use sequence lengths, so just pick the first sequence length so that the summary still works
            sequence_lengths = [sequence_lengths[0]]
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

            max_input_size = tokenizer.model_max_length

        logger.debug(f"Model {model}")
        logger.debug(f"Number of parameters {model.num_parameters()}")

        if precision == Precision.FLOAT16:
            model.half()

        device = torch.device("cuda:0" if use_gpu else "cpu")
        model.to(device)

        if precision == Precision.INT8:
            model = QuantizeHelper.quantize_torch_model(model)

        for batch_size in batch_sizes:
            if batch_size <= 0:
                continue

            for sequence_length in sequence_lengths:
                if config.model_type in ["vit", "swin"]:
                    logger.info(
                        f"Run PyTorch on {model_name} with input shape {[batch_size, 3, config.image_size, config.image_size]}"
                    )
                    input_ids = torch.randn(
                        size=(batch_size, 3, config.image_size, config.image_size),
                        dtype=torch.float16 if precision == Precision.FLOAT16 else torch.float32,
                        device=device,
                    )
                else:
                    if max_input_size is not None and sequence_length > max_input_size:
                        continue

                    logger.info(f"Run PyTorch on {model_name} with input shape {[batch_size, sequence_length]}")
                    input_ids = torch.randint(
                        low=0,
                        high=config.vocab_size - 1,
                        size=(batch_size, sequence_length),
                        dtype=torch.long,
                        device=device,
                    )
                try:
                    inference = (
                        torch.jit.trace(model, input_ids) if torchscript else torch.compile(model) if torch2 else model
                    )
                    inference(input_ids)

                    runtimes = timeit.repeat(lambda: inference(input_ids), repeat=repeat_times, number=1)  # noqa: B023

                    result = {
                        "engine": "torchscript" if torchscript else "torch2" if torch2 else "torch",
                        "version": torch.__version__,
                        "providers": "NA",
                        "device": "cuda" if use_gpu else "cpu",
                        "optimizer": "",
                        "precision": precision,
                        "io_binding": "",
                        "model_name": model_name,
                        "inputs": 1,
                        "threads": num_threads,
                        "batch_size": batch_size,
                        "sequence_length": sequence_length,
                        "custom_layer_num": config_modifier.get_layer_num(),
                        "datetime": str(datetime.now()),
                    }
                    result.update(get_latency_result(runtimes, batch_size))
                    logger.info(result)
                    results.append(result)
                except RuntimeError as e:
                    logger.exception(e)
                    torch.cuda.empty_cache()

    return results


def run_with_tf_optimizations(do_eager_mode: bool, use_xla: bool):
    from functools import wraps

    import tensorflow as tf

    def run_func(func):
        @wraps(func)
        def run_in_eager_mode(*args, **kwargs):
            return func(*args, **kwargs)

        @wraps(func)
        @tf.function(experimental_compile=use_xla)
        def run_in_graph_mode(*args, **kwargs):
            return func(*args, **kwargs)

        if do_eager_mode is True:
            assert use_xla is False, (
                "Cannot run model in XLA, if `args.eager_mode` is set to `True`. Please set `args.eager_mode=False`."
            )
            return run_in_eager_mode
        else:
            return run_in_graph_mode

    return run_func


def run_tensorflow(
    use_gpu,
    model_names,
    model_class,
    config_modifier,
    precision,
    num_threads,
    batch_sizes,
    sequence_lengths,
    repeat_times,
    cache_dir,
    verbose,
):
    results = []

    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(num_threads)

    if not use_gpu:
        tf.config.set_visible_devices([], "GPU")

    if use_gpu and not tf.test.is_built_with_cuda():
        logger.error("Please install Tensorflow-gpu, and use a machine with GPU for testing gpu performance.")
        return results

    if use_gpu:  # Restrict TensorFlow to only use the first GPU
        physical_devices = tf.config.list_physical_devices("GPU")
        try:
            tf.config.set_visible_devices(physical_devices[0], "GPU")
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            tf.distribute.OneDeviceStrategy(device="/gpu:0")
        except RuntimeError as e:
            logger.exception(e)

    if precision == Precision.FLOAT16 or precision == Precision.INT8:
        raise NotImplementedError("Mixed precision is currently not supported.")

    for model_name in model_names:
        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        config_modifier.modify(config)

        model = load_pretrained_model(
            model_name,
            config=config,
            cache_dir=cache_dir,
            custom_model_class=model_class,
            is_tf_model=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        max_input_size = tokenizer.model_max_length

        for batch_size in batch_sizes:
            if batch_size <= 0:
                continue

            for sequence_length in sequence_lengths:
                if max_input_size is not None and sequence_length > max_input_size:
                    continue

                logger.info(f"Run Tensorflow on {model_name} with input shape {[batch_size, sequence_length]}")

                import random

                rng = random.Random()
                values = [rng.randint(0, config.vocab_size - 1) for i in range(batch_size * sequence_length)]
                input_ids = tf.constant(values, shape=(batch_size, sequence_length), dtype=tf.int32)

                try:
                    # Disable both for better inference perf
                    @run_with_tf_optimizations(do_eager_mode=False, use_xla=False)
                    def encoder_forward():
                        return model(input_ids, training=False)  # noqa: B023

                    @run_with_tf_optimizations(do_eager_mode=False, use_xla=False)
                    def encoder_decoder_forward():
                        return model(input_ids, decoder_input_ids=input_ids, training=False)  # noqa: B023

                    @run_with_tf_optimizations(do_eager_mode=False, use_xla=False)
                    def lxmert_forward():
                        feats = tf.random.normal([1, 1, config.visual_feat_dim])  # noqa: B023
                        pos = tf.random.normal([1, 1, config.visual_pos_dim])  # noqa: B023
                        return model(  # noqa: B023
                            input_ids,  # noqa: B023
                            visual_feats=feats,
                            visual_pos=pos,
                            training=False,
                        )

                    inference = encoder_forward
                    if config.is_encoder_decoder:
                        inference = encoder_decoder_forward
                    elif isinstance(config, LxmertConfig):
                        inference = lxmert_forward

                    inference()

                    runtimes = timeit.repeat(lambda: inference(), repeat=repeat_times, number=1)  # noqa: B023

                    result = {
                        "engine": "tensorflow",
                        "version": tf.__version__,
                        "providers": "NA",
                        "device": "cuda" if use_gpu else "cpu",
                        "optimizer": "",
                        "precision": precision,
                        "io_binding": "",
                        "model_name": model_name,
                        "inputs": 1,
                        "threads": num_threads,
                        "batch_size": batch_size,
                        "sequence_length": sequence_length,
                        "custom_layer_num": config_modifier.get_layer_num(),
                        "datetime": str(datetime.now()),
                    }
                    result.update(get_latency_result(runtimes, batch_size))
                    logger.info(result)
                    results.append(result)
                except RuntimeError as e:
                    logger.exception(e)
                    from numba import cuda

                    device = cuda.get_current_device()
                    device.reset()

    return results


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--models",
        required=False,
        nargs="+",
        type=str,
        default=["bert-base-cased", "roberta-base", "gpt2"],
        choices=list(MODELS.keys()),
        help="Pre-trained models in the list: " + ", ".join(MODELS.keys()),
    )

    parser.add_argument(
        "--model_source",
        required=False,
        nargs=1,
        type=str,
        default="pt",
        choices=["pt", "tf"],
        help="Export onnx from pt or tf",
    )

    parser.add_argument(
        "--model_class",
        required=False,
        type=str,
        default=None,
        choices=list(MODEL_CLASSES),
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES),
    )

    parser.add_argument(
        "-e",
        "--engines",
        required=False,
        nargs="+",
        type=str,
        default=["onnxruntime"],
        choices=["onnxruntime", "torch", "torch2", "torchscript", "tensorflow"],
        help="Engines to benchmark",
    )

    parser.add_argument(
        "-c",
        "--cache_dir",
        required=False,
        type=str,
        default=os.path.join(".", "cache_models"),
        help="Directory to cache pre-trained models",
    )

    parser.add_argument(
        "--onnx_dir",
        required=False,
        type=str,
        default=os.path.join(".", "onnx_models"),
        help="Directory to store onnx models",
    )

    parser.add_argument("-g", "--use_gpu", required=False, action="store_true", help="Run on gpu device")

    parser.add_argument(
        "--provider",
        required=False,
        type=str,
        default=None,
        help="Execution provider to use",
    )

    parser.add_argument(
        "-p",
        "--precision",
        type=Precision,
        default=Precision.FLOAT32,
        choices=list(Precision),
        help="Precision of model to run. fp32 for full precision, fp16 for half precision, and int8 for quantization",
    )

    parser.add_argument("--verbose", required=False, action="store_true", help="Print more information")

    parser.add_argument(
        "--overwrite",
        required=False,
        action="store_true",
        help="Overwrite existing models",
    )

    parser.add_argument(
        "-o",
        "--optimizer_info",
        type=OptimizerInfo,
        default=OptimizerInfo.BYSCRIPT,
        choices=list(OptimizerInfo),
        help="Optimizer info: Use optimizer.py to optimize onnx model as default. Can also choose from by_ort and no_opt",
    )

    parser.add_argument(
        "-v",
        "--validate_onnx",
        required=False,
        action="store_true",
        help="Validate ONNX model",
    )

    parser.add_argument(
        "-f",
        "--fusion_csv",
        required=False,
        default=None,
        help="CSV file for saving summary results of graph optimization.",
    )

    parser.add_argument(
        "-d",
        "--detail_csv",
        required=False,
        default=None,
        help="CSV file for saving detail results.",
    )

    parser.add_argument(
        "-r",
        "--result_csv",
        required=False,
        default=None,
        help="CSV file for saving summary results.",
    )

    parser.add_argument(
        "-i",
        "--input_counts",
        required=False,
        nargs="+",
        default=[1],
        type=int,
        choices=[1, 2, 3],
        help="Number of ONNX model inputs. Please use 1 for fair comparison with Torch or TorchScript.",
    )

    parser.add_argument(
        "-t",
        "--test_times",
        required=False,
        default=100,
        type=int,
        help="Number of repeat times to get average inference latency.",
    )

    parser.add_argument("-b", "--batch_sizes", nargs="+", type=int, default=[1])

    parser.add_argument(
        "-s",
        "--sequence_lengths",
        nargs="+",
        type=int,
        default=[4, 8, 16, 32, 64, 128, 256],
    )

    parser.add_argument(
        "--disable_ort_io_binding",
        required=False,
        action="store_true",
        help="Disable running ONNX Runtime with binded inputs and outputs. ",
    )
    parser.set_defaults(disable_ort_io_binding=False)

    parser.add_argument(
        "-n",
        "--num_threads",
        required=False,
        nargs="+",
        type=int,
        default=[0],
        help="Threads to use",
    )

    parser.add_argument(
        "--force_num_layers",
        required=False,
        type=int,
        default=None,
        help="Manually set the model's layer number",
    )

    parser.add_argument(
        "--enable_arm64_bfloat16_fastmath_mlas_gemm",
        required=False,
        action="store_true",
        help="Enable bfloat16 mlas gemm kernels on aarch64. Supported only for CPU EP ",
    )
    parser.set_defaults(enable_arm64_bfloat16_fastmath_mlas_gemm=False)

    FusionOptions.add_arguments(parser)

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    setup_logger(args.verbose)

    if args.precision == Precision.FLOAT16 and not args.use_gpu:
        logger.error("fp16 is for GPU only")
        return

    if args.precision == Precision.INT8 and args.use_gpu and args.provider not in ["migraphx", "rocm"]:
        logger.error("int8 is for CPU only")
        return

    if len(args.models) == 1 and MODELS[args.models[0]][3] in ["vit", "swim"]:
        args.sequence_lengths = [""]

    args.num_threads = sorted({cpu_count if x <= 0 else x for x in args.num_threads})

    logger.info(f"Arguments: {args}")

    if not os.path.exists(args.cache_dir):
        try:
            os.mkdir(args.cache_dir)
        except OSError:
            logger.error("Creation of the directory %s failed", args.cache_dir)

    enable_torch = "torch" in args.engines
    enable_torch2 = "torch2" in args.engines
    enable_torchscript = "torchscript" in args.engines
    enable_onnxruntime = "onnxruntime" in args.engines
    enable_tensorflow = "tensorflow" in args.engines

    if enable_torch2 and version.parse(torch.__version__) < version.parse("2.0.0"):
        logger.error(f"PyTorch version must be >=2.0.0 and you are using {torch.__version__}")
        return

    config_modifier = ConfigModifier(args.force_num_layers)

    results = []

    for num_threads in args.num_threads:
        torch.set_num_threads(num_threads)
        logger.debug(torch.__config__.parallel_info())
        if enable_torch or enable_torch2 or enable_torchscript:
            if args.input_counts != [1]:
                logger.warning("--input_counts is not implemented for torch or torchscript engine.")

            if enable_torchscript:
                results += run_pytorch(
                    args.use_gpu,
                    args.models,
                    args.model_class,
                    config_modifier,
                    args.precision,
                    num_threads,
                    args.batch_sizes,
                    args.sequence_lengths,
                    args.test_times,
                    True,
                    False,
                    args.cache_dir,
                    args.verbose,
                )

            if enable_torch:
                results += run_pytorch(
                    args.use_gpu,
                    args.models,
                    args.model_class,
                    config_modifier,
                    args.precision,
                    num_threads,
                    args.batch_sizes,
                    args.sequence_lengths,
                    args.test_times,
                    False,
                    False,
                    args.cache_dir,
                    args.verbose,
                )

            if enable_torch2:
                results += run_pytorch(
                    args.use_gpu,
                    args.models,
                    args.model_class,
                    config_modifier,
                    args.precision,
                    num_threads,
                    args.batch_sizes,
                    args.sequence_lengths,
                    args.test_times,
                    False,
                    True,
                    args.cache_dir,
                    args.verbose,
                )

        if enable_tensorflow:
            results += run_tensorflow(
                args.use_gpu,
                args.models,
                args.model_class,
                config_modifier,
                args.precision,
                num_threads,
                args.batch_sizes,
                args.sequence_lengths,
                args.test_times,
                args.cache_dir,
                args.verbose,
            )

        model_fusion_statistics = {}
        if enable_onnxruntime:
            try:
                use_raw_attention_mask = not args.use_mask_index
                results += run_onnxruntime(
                    args.use_gpu,
                    args.provider,
                    args.models,
                    args.model_class,
                    config_modifier,
                    args.precision,
                    num_threads,
                    args.batch_sizes,
                    args.sequence_lengths,
                    args.test_times,
                    args.input_counts,
                    args.optimizer_info,
                    args.validate_onnx,
                    args.cache_dir,
                    args.onnx_dir,
                    args.verbose,
                    args.overwrite,
                    args.disable_ort_io_binding,
                    use_raw_attention_mask,
                    model_fusion_statistics,
                    args.model_source,
                    args.enable_arm64_bfloat16_fastmath_mlas_gemm,
                    args,
                )
            except Exception:
                logger.exception("Exception")

    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if model_fusion_statistics:
        csv_filename = args.fusion_csv or f"benchmark_fusion_{time_stamp}.csv"
        output_fusion_statistics(model_fusion_statistics, csv_filename)

    if len(results) == 0:
        if args.batch_sizes != [0]:
            logger.warning("No any result available.")
        return

    csv_filename = args.detail_csv or f"benchmark_detail_{time_stamp}.csv"
    output_details(results, csv_filename)

    csv_filename = args.result_csv or f"benchmark_summary_{time_stamp}.csv"
    output_summary(results, csv_filename, args)


if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\schedule\__init__.py ===
"""
Python job scheduling for humans.

github.com/dbader/schedule

An in-process scheduler for periodic jobs that uses the builder pattern
for configuration. Schedule lets you run Python functions (or any other
callable) periodically at pre-determined intervals using a simple,
human-friendly syntax.

Inspired by Addam Wiggins' article "Rethinking Cron" [1] and the
"clockwork" Ruby module [2][3].

Features:
    - A simple to use API for scheduling jobs.
    - Very lightweight and no external dependencies.
    - Excellent test coverage.
    - Tested on Python 3.7, 3.8, 3.9, 3.10, 3.11 and 3.12

Usage:
    >>> import schedule
    >>> import time

    >>> def job(message='stuff'):
    >>>     print("I'm working on:", message)

    >>> schedule.every(10).minutes.do(job)
    >>> schedule.every(5).to(10).days.do(job)
    >>> schedule.every().hour.do(job, message='things')
    >>> schedule.every().day.at("10:30").do(job)

    >>> while True:
    >>>     schedule.run_pending()
    >>>     time.sleep(1)

[1] https://adam.herokuapp.com/past/2010/4/13/rethinking_cron/
[2] https://github.com/Rykian/clockwork
[3] https://adam.herokuapp.com/past/2010/6/30/replace_cron_with_clockwork/
"""

from collections.abc import Hashable
import datetime
import functools
import logging
import random
import re
import time
from typing import Set, List, Optional, Callable, Union

logger = logging.getLogger("schedule")


class ScheduleError(Exception):
    """Base schedule exception"""

    pass


class ScheduleValueError(ScheduleError):
    """Base schedule value error"""

    pass


class IntervalError(ScheduleValueError):
    """An improper interval was used"""

    pass


class CancelJob:
    """
    Can be returned from a job to unschedule itself.
    """

    pass


class Scheduler:
    """
    Objects instantiated by the :class:`Scheduler <Scheduler>` are
    factories to create jobs, keep record of scheduled jobs and
    handle their execution.
    """

    def __init__(self) -> None:
        self.jobs: List[Job] = []

    def run_pending(self) -> None:
        """
        Run all jobs that are scheduled to run.

        Please note that it is *intended behavior that run_pending()
        does not run missed jobs*. For example, if you've registered a job
        that should run every minute and you only call run_pending()
        in one hour increments then your job won't be run 60 times in
        between but only once.
        """
        runnable_jobs = (job for job in self.jobs if job.should_run)
        for job in sorted(runnable_jobs):
            self._run_job(job)

    def run_all(self, delay_seconds: int = 0) -> None:
        """
        Run all jobs regardless if they are scheduled to run or not.

        A delay of `delay` seconds is added between each job. This helps
        distribute system load generated by the jobs more evenly
        over time.

        :param delay_seconds: A delay added between every executed job
        """
        logger.debug(
            "Running *all* %i jobs with %is delay in between",
            len(self.jobs),
            delay_seconds,
        )
        for job in self.jobs[:]:
            self._run_job(job)
            time.sleep(delay_seconds)

    def get_jobs(self, tag: Optional[Hashable] = None) -> List["Job"]:
        """
        Gets scheduled jobs marked with the given tag, or all jobs
        if tag is omitted.

        :param tag: An identifier used to identify a subset of
                    jobs to retrieve
        """
        if tag is None:
            return self.jobs[:]
        else:
            return [job for job in self.jobs if tag in job.tags]

    def clear(self, tag: Optional[Hashable] = None) -> None:
        """
        Deletes scheduled jobs marked with the given tag, or all jobs
        if tag is omitted.

        :param tag: An identifier used to identify a subset of
                    jobs to delete
        """
        if tag is None:
            logger.debug("Deleting *all* jobs")
            del self.jobs[:]
        else:
            logger.debug('Deleting all jobs tagged "%s"', tag)
            self.jobs[:] = (job for job in self.jobs if tag not in job.tags)

    def cancel_job(self, job: "Job") -> None:
        """
        Delete a scheduled job.

        :param job: The job to be unscheduled
        """
        try:
            logger.debug('Cancelling job "%s"', str(job))
            self.jobs.remove(job)
        except ValueError:
            logger.debug('Cancelling not-scheduled job "%s"', str(job))

    def every(self, interval: int = 1) -> "Job":
        """
        Schedule a new periodic job.

        :param interval: A quantity of a certain time unit
        :return: An unconfigured :class:`Job <Job>`
        """
        job = Job(interval, self)
        return job

    def _run_job(self, job: "Job") -> None:
        ret = job.run()
        if isinstance(ret, CancelJob) or ret is CancelJob:
            self.cancel_job(job)

    def get_next_run(
        self, tag: Optional[Hashable] = None
    ) -> Optional[datetime.datetime]:
        """
        Datetime when the next job should run.

        :param tag: Filter the next run for the given tag parameter

        :return: A :class:`~datetime.datetime` object
                 or None if no jobs scheduled
        """
        if not self.jobs:
            return None
        jobs_filtered = self.get_jobs(tag)
        if not jobs_filtered:
            return None
        return min(jobs_filtered).next_run

    next_run = property(get_next_run)

    @property
    def idle_seconds(self) -> Optional[float]:
        """
        :return: Number of seconds until
                 :meth:`next_run <Scheduler.next_run>`
                 or None if no jobs are scheduled
        """
        if not self.next_run:
            return None
        return (self.next_run - datetime.datetime.now()).total_seconds()


class Job:
    """
    A periodic job as used by :class:`Scheduler`.

    :param interval: A quantity of a certain time unit
    :param scheduler: The :class:`Scheduler <Scheduler>` instance that
                      this job will register itself with once it has
                      been fully configured in :meth:`Job.do()`.

    Every job runs at a given fixed time interval that is defined by:

    * a :meth:`time unit <Job.second>`
    * a quantity of `time units` defined by `interval`

    A job is usually created and returned by :meth:`Scheduler.every`
    method, which also defines its `interval`.
    """

    def __init__(self, interval: int, scheduler: Optional[Scheduler] = None):
        self.interval: int = interval  # pause interval * unit between runs
        self.latest: Optional[int] = None  # upper limit to the interval
        self.job_func: Optional[functools.partial] = None  # the job job_func to run

        # time units, e.g. 'minutes', 'hours', ...
        self.unit: Optional[str] = None

        # optional time at which this job runs
        self.at_time: Optional[datetime.time] = None

        # optional time zone of the self.at_time field. Only relevant when at_time is not None
        self.at_time_zone = None

        # datetime of the last run
        self.last_run: Optional[datetime.datetime] = None

        # datetime of the next run
        self.next_run: Optional[datetime.datetime] = None

        # Weekday to run the job at. Only relevant when unit is 'weeks'.
        # For example, when asking 'every week on tuesday' the start_day is 'tuesday'.
        self.start_day: Optional[str] = None

        # optional time of final run
        self.cancel_after: Optional[datetime.datetime] = None

        self.tags: Set[Hashable] = set()  # unique set of tags for the job
        self.scheduler: Optional[Scheduler] = scheduler  # scheduler to register with

    def __lt__(self, other) -> bool:
        """
        PeriodicJobs are sortable based on the scheduled time they
        run next.
        """
        return self.next_run < other.next_run

    def __str__(self) -> str:
        if hasattr(self.job_func, "__name__"):
            job_func_name = self.job_func.__name__  # type: ignore
        else:
            job_func_name = repr(self.job_func)

        return ("Job(interval={}, unit={}, do={}, args={}, kwargs={})").format(
            self.interval,
            self.unit,
            job_func_name,
            "()" if self.job_func is None else self.job_func.args,
            "{}" if self.job_func is None else self.job_func.keywords,
        )

    def __repr__(self):
        def format_time(t):
            return t.strftime("%Y-%m-%d %H:%M:%S") if t else "[never]"

        def is_repr(j):
            return not isinstance(j, Job)

        timestats = "(last run: %s, next run: %s)" % (
            format_time(self.last_run),
            format_time(self.next_run),
        )

        if hasattr(self.job_func, "__name__"):
            job_func_name = self.job_func.__name__
        else:
            job_func_name = repr(self.job_func)

        if self.job_func is not None:
            args = [repr(x) if is_repr(x) else str(x) for x in self.job_func.args]
            kwargs = ["%s=%s" % (k, repr(v)) for k, v in self.job_func.keywords.items()]
            call_repr = job_func_name + "(" + ", ".join(args + kwargs) + ")"
        else:
            call_repr = "[None]"

        if self.at_time is not None:
            return "Every %s %s at %s do %s %s" % (
                self.interval,
                self.unit[:-1] if self.interval == 1 else self.unit,
                self.at_time,
                call_repr,
                timestats,
            )
        else:
            fmt = (
                "Every %(interval)s "
                + ("to %(latest)s " if self.latest is not None else "")
                + "%(unit)s do %(call_repr)s %(timestats)s"
            )

            return fmt % dict(
                interval=self.interval,
                latest=self.latest,
                unit=(self.unit[:-1] if self.interval == 1 else self.unit),
                call_repr=call_repr,
                timestats=timestats,
            )

    @property
    def second(self):
        if self.interval != 1:
            raise IntervalError("Use seconds instead of second")
        return self.seconds

    @property
    def seconds(self):
        self.unit = "seconds"
        return self

    @property
    def minute(self):
        if self.interval != 1:
            raise IntervalError("Use minutes instead of minute")
        return self.minutes

    @property
    def minutes(self):
        self.unit = "minutes"
        return self

    @property
    def hour(self):
        if self.interval != 1:
            raise IntervalError("Use hours instead of hour")
        return self.hours

    @property
    def hours(self):
        self.unit = "hours"
        return self

    @property
    def day(self):
        if self.interval != 1:
            raise IntervalError("Use days instead of day")
        return self.days

    @property
    def days(self):
        self.unit = "days"
        return self

    @property
    def week(self):
        if self.interval != 1:
            raise IntervalError("Use weeks instead of week")
        return self.weeks

    @property
    def weeks(self):
        self.unit = "weeks"
        return self

    @property
    def monday(self):
        if self.interval != 1:
            raise IntervalError(
                "Scheduling .monday() jobs is only allowed for weekly jobs. "
                "Using .monday() on a job scheduled to run every 2 or more weeks "
                "is not supported."
            )
        self.start_day = "monday"
        return self.weeks

    @property
    def tuesday(self):
        if self.interval != 1:
            raise IntervalError(
                "Scheduling .tuesday() jobs is only allowed for weekly jobs. "
                "Using .tuesday() on a job scheduled to run every 2 or more weeks "
                "is not supported."
            )
        self.start_day = "tuesday"
        return self.weeks

    @property
    def wednesday(self):
        if self.interval != 1:
            raise IntervalError(
                "Scheduling .wednesday() jobs is only allowed for weekly jobs. "
                "Using .wednesday() on a job scheduled to run every 2 or more weeks "
                "is not supported."
            )
        self.start_day = "wednesday"
        return self.weeks

    @property
    def thursday(self):
        if self.interval != 1:
            raise IntervalError(
                "Scheduling .thursday() jobs is only allowed for weekly jobs. "
                "Using .thursday() on a job scheduled to run every 2 or more weeks "
                "is not supported."
            )
        self.start_day = "thursday"
        return self.weeks

    @property
    def friday(self):
        if self.interval != 1:
            raise IntervalError(
                "Scheduling .friday() jobs is only allowed for weekly jobs. "
                "Using .friday() on a job scheduled to run every 2 or more weeks "
                "is not supported."
            )
        self.start_day = "friday"
        return self.weeks

    @property
    def saturday(self):
        if self.interval != 1:
            raise IntervalError(
                "Scheduling .saturday() jobs is only allowed for weekly jobs. "
                "Using .saturday() on a job scheduled to run every 2 or more weeks "
                "is not supported."
            )
        self.start_day = "saturday"
        return self.weeks

    @property
    def sunday(self):
        if self.interval != 1:
            raise IntervalError(
                "Scheduling .sunday() jobs is only allowed for weekly jobs. "
                "Using .sunday() on a job scheduled to run every 2 or more weeks "
                "is not supported."
            )
        self.start_day = "sunday"
        return self.weeks

    def tag(self, *tags: Hashable):
        """
        Tags the job with one or more unique identifiers.

        Tags must be hashable. Duplicate tags are discarded.

        :param tags: A unique list of ``Hashable`` tags.
        :return: The invoked job instance
        """
        if not all(isinstance(tag, Hashable) for tag in tags):
            raise TypeError("Tags must be hashable")
        self.tags.update(tags)
        return self

    def at(self, time_str: str, tz: Optional[str] = None):
        """
        Specify a particular time that the job should be run at.

        :param time_str: A string in one of the following formats:

            - For daily jobs -> `HH:MM:SS` or `HH:MM`
            - For hourly jobs -> `MM:SS` or `:MM`
            - For minute jobs -> `:SS`

            The format must make sense given how often the job is
            repeating; for example, a job that repeats every minute
            should not be given a string in the form `HH:MM:SS`. The
            difference between `:MM` and `:SS` is inferred from the
            selected time-unit (e.g. `every().hour.at(':30')` vs.
            `every().minute.at(':30')`).

        :param tz: The timezone that this timestamp refers to. Can be
            a string that can be parsed by pytz.timezone(), or a pytz.BaseTzInfo object

        :return: The invoked job instance
        """
        if self.unit not in ("days", "hours", "minutes") and not self.start_day:
            raise ScheduleValueError(
                "Invalid unit (valid units are `days`, `hours`, and `minutes`)"
            )

        if tz is not None:
            import pytz

            if isinstance(tz, str):
                self.at_time_zone = pytz.timezone(tz)  # type: ignore
            elif isinstance(tz, pytz.BaseTzInfo):
                self.at_time_zone = tz
            else:
                raise ScheduleValueError(
                    "Timezone must be string or pytz.timezone object"
                )

        if not isinstance(time_str, str):
            raise TypeError("at() should be passed a string")
        if self.unit == "days" or self.start_day:
            if not re.match(r"^[0-2]\d:[0-5]\d(:[0-5]\d)?$", time_str):
                raise ScheduleValueError(
                    "Invalid time format for a daily job (valid format is HH:MM(:SS)?)"
                )
        if self.unit == "hours":
            if not re.match(r"^([0-5]\d)?:[0-5]\d$", time_str):
                raise ScheduleValueError(
                    "Invalid time format for an hourly job (valid format is (MM)?:SS)"
                )

        if self.unit == "minutes":
            if not re.match(r"^:[0-5]\d$", time_str):
                raise ScheduleValueError(
                    "Invalid time format for a minutely job (valid format is :SS)"
                )
        time_values = time_str.split(":")
        hour: Union[str, int]
        minute: Union[str, int]
        second: Union[str, int]
        if len(time_values) == 3:
            hour, minute, second = time_values
        elif len(time_values) == 2 and self.unit == "minutes":
            hour = 0
            minute = 0
            _, second = time_values
        elif len(time_values) == 2 and self.unit == "hours" and len(time_values[0]):
            hour = 0
            minute, second = time_values
        else:
            hour, minute = time_values
            second = 0
        if self.unit == "days" or self.start_day:
            hour = int(hour)
            if not (0 <= hour <= 23):
                raise ScheduleValueError(
                    "Invalid number of hours ({} is not between 0 and 23)"
                )
        elif self.unit == "hours":
            hour = 0
        elif self.unit == "minutes":
            hour = 0
            minute = 0
        hour = int(hour)
        minute = int(minute)
        second = int(second)
        self.at_time = datetime.time(hour, minute, second)
        return self

    def to(self, latest: int):
        """
        Schedule the job to run at an irregular (randomized) interval.

        The job's interval will randomly vary from the value given
        to  `every` to `latest`. The range defined is inclusive on
        both ends. For example, `every(A).to(B).seconds` executes
        the job function every N seconds such that A <= N <= B.

        :param latest: Maximum interval between randomized job runs
        :return: The invoked job instance
        """
        self.latest = latest
        return self

    def until(
        self,
        until_time: Union[datetime.datetime, datetime.timedelta, datetime.time, str],
    ):
        """
        Schedule job to run until the specified moment.

        The job is canceled whenever the next run is calculated and it turns out the
        next run is after the until_time. The job is also canceled right before it runs,
        if the current time is after until_time. This latter case can happen when the
        the job was scheduled to run before until_time, but runs after until_time.

        If until_time is a moment in the past, ScheduleValueError is thrown.

        :param until_time: A moment in the future representing the latest time a job can
           be run. If only a time is supplied, the date is set to today.
           The following formats are accepted:

           - datetime.datetime
           - datetime.timedelta
           - datetime.time
           - String in one of the following formats: "%Y-%m-%d %H:%M:%S",
             "%Y-%m-%d %H:%M", "%Y-%m-%d", "%H:%M:%S", "%H:%M"
             as defined by strptime() behaviour. If an invalid string format is passed,
             ScheduleValueError is thrown.

        :return: The invoked job instance
        """

        if isinstance(until_time, datetime.datetime):
            self.cancel_after = until_time
        elif isinstance(until_time, datetime.timedelta):
            self.cancel_after = datetime.datetime.now() + until_time
        elif isinstance(until_time, datetime.time):
            self.cancel_after = datetime.datetime.combine(
                datetime.datetime.now(), until_time
            )
        elif isinstance(until_time, str):
            cancel_after = self._decode_datetimestr(
                until_time,
                [
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d %H:%M",
                    "%Y-%m-%d",
                    "%H:%M:%S",
                    "%H:%M",
                ],
            )
            if cancel_after is None:
                raise ScheduleValueError("Invalid string format for until()")
            if "-" not in until_time:
                # the until_time is a time-only format. Set the date to today
                now = datetime.datetime.now()
                cancel_after = cancel_after.replace(
                    year=now.year, month=now.month, day=now.day
                )
            self.cancel_after = cancel_after
        else:
            raise TypeError(
                "until() takes a string, datetime.datetime, datetime.timedelta, "
                "datetime.time parameter"
            )
        if self.cancel_after < datetime.datetime.now():
            raise ScheduleValueError(
                "Cannot schedule a job to run until a time in the past"
            )
        return self

    def do(self, job_func: Callable, *args, **kwargs):
        """
        Specifies the job_func that should be called every time the
        job runs.

        Any additional arguments are passed on to job_func when
        the job runs.

        :param job_func: The function to be scheduled
        :return: The invoked job instance
        """
        self.job_func = functools.partial(job_func, *args, **kwargs)
        functools.update_wrapper(self.job_func, job_func)
        self._schedule_next_run()
        if self.scheduler is None:
            raise ScheduleError(
                "Unable to a add job to schedule. "
                "Job is not associated with an scheduler"
            )
        self.scheduler.jobs.append(self)
        return self

    @property
    def should_run(self) -> bool:
        """
        :return: ``True`` if the job should be run now.
        """
        assert self.next_run is not None, "must run _schedule_next_run before"
        return datetime.datetime.now() >= self.next_run

    def run(self):
        """
        Run the job and immediately reschedule it.
        If the job's deadline is reached (configured using .until()), the job is not
        run and CancelJob is returned immediately. If the next scheduled run exceeds
        the job's deadline, CancelJob is returned after the execution. In this latter
        case CancelJob takes priority over any other returned value.

        :return: The return value returned by the `job_func`, or CancelJob if the job's
                 deadline is reached.

        """
        if self._is_overdue(datetime.datetime.now()):
            logger.debug("Cancelling job %s", self)
            return CancelJob

        logger.debug("Running job %s", self)
        ret = self.job_func()
        self.last_run = datetime.datetime.now()
        self._schedule_next_run()

        if self._is_overdue(self.next_run):
            logger.debug("Cancelling job %s", self)
            return CancelJob
        return ret

    def _schedule_next_run(self) -> None:
        """
        Compute the instant when this job should run next.
        """
        if self.unit not in ("seconds", "minutes", "hours", "days", "weeks"):
            raise ScheduleValueError(
                "Invalid unit (valid units are `seconds`, `minutes`, `hours`, "
                "`days`, and `weeks`)"
            )
        if self.latest is not None:
            if not (self.latest >= self.interval):
                raise ScheduleError("`latest` is greater than `interval`")
            interval = random.randint(self.interval, self.latest)
        else:
            interval = self.interval

        # Do all computation in the context of the requested timezone
        now = datetime.datetime.now(self.at_time_zone)

        next_run = now

        if self.start_day is not None:
            if self.unit != "weeks":
                raise ScheduleValueError("`unit` should be 'weeks'")
            next_run = _move_to_next_weekday(next_run, self.start_day)

        if self.at_time is not None:
            next_run = self._move_to_at_time(next_run)

        period = datetime.timedelta(**{self.unit: interval})
        if interval != 1:
            next_run += period

        while next_run <= now:
            next_run += period

        next_run = self._correct_utc_offset(
            next_run, fixate_time=(self.at_time is not None)
        )

        # To keep the api consistent with older versions, we have to set the 'next_run' to a naive timestamp in the local timezone.
        # Because we want to stay backwards compatible with older versions.
        if self.at_time_zone is not None:
            # Convert back to the local timezone
            next_run = next_run.astimezone()

            next_run = next_run.replace(tzinfo=None)

        self.next_run = next_run

    def _move_to_at_time(self, moment: datetime.datetime) -> datetime.datetime:
        """
        Takes a datetime and moves the time-component to the job's at_time.
        """
        if self.at_time is None:
            return moment

        kwargs = {"second": self.at_time.second, "microsecond": 0}

        if self.unit == "days" or self.start_day is not None:
            kwargs["hour"] = self.at_time.hour

        if self.unit in ["days", "hours"] or self.start_day is not None:
            kwargs["minute"] = self.at_time.minute

        moment = moment.replace(**kwargs)  # type: ignore

        # When we set the time elements, we might end up in a different UTC-offset than the current offset.
        # This happens when we cross into or out of daylight saving time.
        moment = self._correct_utc_offset(moment, fixate_time=True)

        return moment

    def _correct_utc_offset(
        self, moment: datetime.datetime, fixate_time: bool
    ) -> datetime.datetime:
        """
        Given a datetime, corrects any mistakes in the utc offset.
        This is similar to pytz' normalize, but adds the ability to attempt
        keeping the time-component at the same hour/minute/second.
        """
        if self.at_time_zone is None:
            return moment
        # Normalize corrects the utc-offset to match the timezone
        # For example: When a date&time&offset does not exist within a timezone,
        # the normalization will change the utc-offset to where it is valid.
        # It does this while keeping the moment in time the same, by moving the
        # time component opposite of the utc-change.
        offset_before_normalize = moment.utcoffset()
        moment = self.at_time_zone.normalize(moment)
        offset_after_normalize = moment.utcoffset()

        if offset_before_normalize == offset_after_normalize:
            # There was no change in the utc-offset, datetime didn't change.
            return moment

        # The utc-offset and time-component has changed

        if not fixate_time:
            # No need to fixate the time.
            return moment

        offset_diff = offset_after_normalize - offset_before_normalize

        # Adjust the time to reset the date-time to have the same HH:mm components
        moment -= offset_diff

        # Check if moving the timestamp back by the utc-offset-difference made it end up
        # in a moment that does not exist within the current timezone/utc-offset
        re_normalized_offset = self.at_time_zone.normalize(moment).utcoffset()
        if re_normalized_offset != offset_after_normalize:
            # We ended up in a DST Gap. The requested 'at' time does not exist
            # within the current timezone/utc-offset. As a best effort, we will
            # schedule the job 1 offset later than possible.
            # For example, if 02:23 does not exist (because DST moves from 02:00
            # to 03:00), this will schedule the job at 03:23.
            moment += offset_diff
        return moment

    def _is_overdue(self, when: datetime.datetime):
        return self.cancel_after is not None and when > self.cancel_after

    def _decode_datetimestr(
        self, datetime_str: str, formats: List[str]
    ) -> Optional[datetime.datetime]:
        for f in formats:
            try:
                return datetime.datetime.strptime(datetime_str, f)
            except ValueError:
                pass
        return None


# The following methods are shortcuts for not having to
# create a Scheduler instance:

#: Default :class:`Scheduler <Scheduler>` object
default_scheduler = Scheduler()

#: Default :class:`Jobs <Job>` list
jobs = default_scheduler.jobs  # todo: should this be a copy, e.g. jobs()?


def every(interval: int = 1) -> Job:
    """Calls :meth:`every <Scheduler.every>` on the
    :data:`default scheduler instance <default_scheduler>`.
    """
    return default_scheduler.every(interval)


def run_pending() -> None:
    """Calls :meth:`run_pending <Scheduler.run_pending>` on the
    :data:`default scheduler instance <default_scheduler>`.
    """
    default_scheduler.run_pending()


def run_all(delay_seconds: int = 0) -> None:
    """Calls :meth:`run_all <Scheduler.run_all>` on the
    :data:`default scheduler instance <default_scheduler>`.
    """
    default_scheduler.run_all(delay_seconds=delay_seconds)


def get_jobs(tag: Optional[Hashable] = None) -> List[Job]:
    """Calls :meth:`get_jobs <Scheduler.get_jobs>` on the
    :data:`default scheduler instance <default_scheduler>`.
    """
    return default_scheduler.get_jobs(tag)


def clear(tag: Optional[Hashable] = None) -> None:
    """Calls :meth:`clear <Scheduler.clear>` on the
    :data:`default scheduler instance <default_scheduler>`.
    """
    default_scheduler.clear(tag)


def cancel_job(job: Job) -> None:
    """Calls :meth:`cancel_job <Scheduler.cancel_job>` on the
    :data:`default scheduler instance <default_scheduler>`.
    """
    default_scheduler.cancel_job(job)


def next_run(tag: Optional[Hashable] = None) -> Optional[datetime.datetime]:
    """Calls :meth:`next_run <Scheduler.next_run>` on the
    :data:`default scheduler instance <default_scheduler>`.
    """
    return default_scheduler.get_next_run(tag)


def idle_seconds() -> Optional[float]:
    """Calls :meth:`idle_seconds <Scheduler.idle_seconds>` on the
    :data:`default scheduler instance <default_scheduler>`.
    """
    return default_scheduler.idle_seconds


def repeat(job, *args, **kwargs):
    """
    Decorator to schedule a new periodic job.

    Any additional arguments are passed on to the decorated function
    when the job runs.

    :param job: a :class:`Jobs <Job>`
    """

    def _schedule_decorator(decorated_function):
        job.do(decorated_function, *args, **kwargs)
        return decorated_function

    return _schedule_decorator


def _move_to_next_weekday(moment: datetime.datetime, weekday: str):
    """
    Move the given timestamp to the nearest given weekday. May be this week
    or next week. If the timestamp is already at the given weekday, it is not
    moved.
    """
    weekday_index = _weekday_index(weekday)

    days_ahead = weekday_index - moment.weekday()
    if days_ahead < 0:
        # Target day already happened this week, move to next week
        days_ahead += 7
    return moment + datetime.timedelta(days=days_ahead)


def _weekday_index(day: str) -> int:
    weekdays = (
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    )
    if day not in weekdays:
        raise ScheduleValueError(
            "Invalid start day (valid start days are {})".format(weekdays)
        )
    return weekdays.index(day)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\cse_main.py ===
""" Tools for doing common subexpression elimination.
"""
from collections import defaultdict

from sympy.core import Basic, Mul, Add, Pow, sympify
from sympy.core.containers import Tuple, OrderedSet
from sympy.core.exprtools import factor_terms
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import symbols, Symbol
from sympy.matrices import (MatrixBase, Matrix, ImmutableMatrix,
                            SparseMatrix, ImmutableSparseMatrix)
from sympy.matrices.expressions import (MatrixExpr, MatrixSymbol, MatMul,
                                        MatAdd, MatPow, Inverse)
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.polys.rootoftools import RootOf
from sympy.utilities.iterables import numbered_symbols, sift, \
        topological_sort, iterable

from . import cse_opts

# (preprocessor, postprocessor) pairs which are commonly useful. They should
# each take a SymPy expression and return a possibly transformed expression.
# When used in the function ``cse()``, the target expressions will be transformed
# by each of the preprocessor functions in order. After the common
# subexpressions are eliminated, each resulting expression will have the
# postprocessor functions transform them in *reverse* order in order to undo the
# transformation if necessary. This allows the algorithm to operate on
# a representation of the expressions that allows for more optimization
# opportunities.
# ``None`` can be used to specify no transformation for either the preprocessor or
# postprocessor.


basic_optimizations = [(cse_opts.sub_pre, cse_opts.sub_post),
                       (factor_terms, None)]

# sometimes we want the output in a different format; non-trivial
# transformations can be put here for users
# ===============================================================


def reps_toposort(r):
    """Sort replacements ``r`` so (k1, v1) appears before (k2, v2)
    if k2 is in v1's free symbols. This orders items in the
    way that cse returns its results (hence, in order to use the
    replacements in a substitution option it would make sense
    to reverse the order).

    Examples
    ========

    >>> from sympy.simplify.cse_main import reps_toposort
    >>> from sympy.abc import x, y
    >>> from sympy import Eq
    >>> for l, r in reps_toposort([(x, y + 1), (y, 2)]):
    ...     print(Eq(l, r))
    ...
    Eq(y, 2)
    Eq(x, y + 1)

    """
    r = sympify(r)
    E = []
    for c1, (k1, v1) in enumerate(r):
        for c2, (k2, v2) in enumerate(r):
            if k1 in v2.free_symbols:
                E.append((c1, c2))
    return [r[i] for i in topological_sort((range(len(r)), E))]


def cse_separate(r, e):
    """Move expressions that are in the form (symbol, expr) out of the
    expressions and sort them into the replacements using the reps_toposort.

    Examples
    ========

    >>> from sympy.simplify.cse_main import cse_separate
    >>> from sympy.abc import x, y, z
    >>> from sympy import cos, exp, cse, Eq, symbols
    >>> x0, x1 = symbols('x:2')
    >>> eq = (x + 1 + exp((x + 1)/(y + 1)) + cos(y + 1))
    >>> cse([eq, Eq(x, z + 1), z - 2], postprocess=cse_separate) in [
    ... [[(x0, y + 1), (x, z + 1), (x1, x + 1)],
    ...  [x1 + exp(x1/x0) + cos(x0), z - 2]],
    ... [[(x1, y + 1), (x, z + 1), (x0, x + 1)],
    ...  [x0 + exp(x0/x1) + cos(x1), z - 2]]]
    ...
    True
    """
    d = sift(e, lambda w: w.is_Equality and w.lhs.is_Symbol)
    r = r + [w.args for w in d[True]]
    e = d[False]
    return [reps_toposort(r), e]


def cse_release_variables(r, e):
    """
    Return tuples giving ``(a, b)`` where ``a`` is a symbol and ``b`` is
    either an expression or None. The value of None is used when a
    symbol is no longer needed for subsequent expressions.

    Use of such output can reduce the memory footprint of lambdified
    expressions that contain large, repeated subexpressions.

    Examples
    ========

    >>> from sympy import cse
    >>> from sympy.simplify.cse_main import cse_release_variables
    >>> from sympy.abc import x, y
    >>> eqs = [(x + y - 1)**2, x, x + y, (x + y)/(2*x + 1) + (x + y - 1)**2, (2*x + 1)**(x + y)]
    >>> defs, rvs = cse_release_variables(*cse(eqs))
    >>> for i in defs:
    ...   print(i)
    ...
    (x0, x + y)
    (x1, (x0 - 1)**2)
    (x2, 2*x + 1)
    (_3, x0/x2 + x1)
    (_4, x2**x0)
    (x2, None)
    (_0, x1)
    (x1, None)
    (_2, x0)
    (x0, None)
    (_1, x)
    >>> print(rvs)
    (_0, _1, _2, _3, _4)
    """
    if not r:
        return r, e

    s, p = zip(*r)
    esyms = symbols('_:%d' % len(e))
    syms = list(esyms)
    s = list(s)
    in_use = set(s)
    p = list(p)
    # sort e so those with most sub-expressions appear first
    e = [(e[i], syms[i]) for i in range(len(e))]
    e, syms = zip(*sorted(e,
        key=lambda x: -sum(p[s.index(i)].count_ops()
        for i in x[0].free_symbols & in_use)))
    syms = list(syms)
    p += e
    rv = []
    i = len(p) - 1
    while i >= 0:
        _p = p.pop()
        c = in_use & _p.free_symbols
        if c: # sorting for canonical results
            rv.extend([(s, None) for s in sorted(c, key=str)])
        if i >= len(r):
            rv.append((syms.pop(), _p))
        else:
            rv.append((s[i], _p))
        in_use -= c
        i -= 1
    rv.reverse()
    return rv, esyms


# ====end of cse postprocess idioms===========================


def preprocess_for_cse(expr, optimizations):
    """ Preprocess an expression to optimize for common subexpression
    elimination.

    Parameters
    ==========

    expr : SymPy expression
        The target expression to optimize.
    optimizations : list of (callable, callable) pairs
        The (preprocessor, postprocessor) pairs.

    Returns
    =======

    expr : SymPy expression
        The transformed expression.
    """
    for pre, post in optimizations:
        if pre is not None:
            expr = pre(expr)
    return expr


def postprocess_for_cse(expr, optimizations):
    """Postprocess an expression after common subexpression elimination to
    return the expression to canonical SymPy form.

    Parameters
    ==========

    expr : SymPy expression
        The target expression to transform.
    optimizations : list of (callable, callable) pairs, optional
        The (preprocessor, postprocessor) pairs.  The postprocessors will be
        applied in reversed order to undo the effects of the preprocessors
        correctly.

    Returns
    =======

    expr : SymPy expression
        The transformed expression.
    """
    for pre, post in reversed(optimizations):
        if post is not None:
            expr = post(expr)
    return expr


class FuncArgTracker:
    """
    A class which manages a mapping from functions to arguments and an inverse
    mapping from arguments to functions.
    """

    def __init__(self, funcs):
        # To minimize the number of symbolic comparisons, all function arguments
        # get assigned a value number.
        self.value_numbers = {}
        self.value_number_to_value = []

        # Both of these maps use integer indices for arguments / functions.
        self.arg_to_funcset = []
        self.func_to_argset = []

        for func_i, func in enumerate(funcs):
            func_argset = OrderedSet()

            for func_arg in func.args:
                arg_number = self.get_or_add_value_number(func_arg)
                func_argset.add(arg_number)
                self.arg_to_funcset[arg_number].add(func_i)

            self.func_to_argset.append(func_argset)

    def get_args_in_value_order(self, argset):
        """
        Return the list of arguments in sorted order according to their value
        numbers.
        """
        return [self.value_number_to_value[argn] for argn in sorted(argset)]

    def get_or_add_value_number(self, value):
        """
        Return the value number for the given argument.
        """
        nvalues = len(self.value_numbers)
        value_number = self.value_numbers.setdefault(value, nvalues)
        if value_number == nvalues:
            self.value_number_to_value.append(value)
            self.arg_to_funcset.append(OrderedSet())
        return value_number

    def stop_arg_tracking(self, func_i):
        """
        Remove the function func_i from the argument to function mapping.
        """
        for arg in self.func_to_argset[func_i]:
            self.arg_to_funcset[arg].remove(func_i)


    def get_common_arg_candidates(self, argset, min_func_i=0):
        """Return a dict whose keys are function numbers. The entries of the dict are
        the number of arguments said function has in common with
        ``argset``. Entries have at least 2 items in common.  All keys have
        value at least ``min_func_i``.
        """
        count_map = defaultdict(lambda: 0)
        if not argset:
            return count_map

        funcsets = [self.arg_to_funcset[arg] for arg in argset]
        # As an optimization below, we handle the largest funcset separately from
        # the others.
        largest_funcset = max(funcsets, key=len)

        for funcset in funcsets:
            if largest_funcset is funcset:
                continue
            for func_i in funcset:
                if func_i >= min_func_i:
                    count_map[func_i] += 1

        # We pick the smaller of the two containers (count_map, largest_funcset)
        # to iterate over to reduce the number of iterations needed.
        (smaller_funcs_container,
         larger_funcs_container) = sorted(
                 [largest_funcset, count_map],
                 key=len)

        for func_i in smaller_funcs_container:
            # Not already in count_map? It can't possibly be in the output, so
            # skip it.
            if count_map[func_i] < 1:
                continue

            if func_i in larger_funcs_container:
                count_map[func_i] += 1

        return {k: v for k, v in count_map.items() if v >= 2}

    def get_subset_candidates(self, argset, restrict_to_funcset=None):
        """
        Return a set of functions each of which whose argument list contains
        ``argset``, optionally filtered only to contain functions in
        ``restrict_to_funcset``.
        """
        iarg = iter(argset)

        indices = OrderedSet(
            fi for fi in self.arg_to_funcset[next(iarg)])

        if restrict_to_funcset is not None:
            indices &= restrict_to_funcset

        for arg in iarg:
            indices &= self.arg_to_funcset[arg]

        return indices

    def update_func_argset(self, func_i, new_argset):
        """
        Update a function with a new set of arguments.
        """
        new_args = OrderedSet(new_argset)
        old_args = self.func_to_argset[func_i]

        for deleted_arg in old_args - new_args:
            self.arg_to_funcset[deleted_arg].remove(func_i)
        for added_arg in new_args - old_args:
            self.arg_to_funcset[added_arg].add(func_i)

        self.func_to_argset[func_i].clear()
        self.func_to_argset[func_i].update(new_args)


class Unevaluated:

    def __init__(self, func, args):
        self.func = func
        self.args = args

    def __str__(self):
        return "Uneval<{}>({})".format(
                self.func, ", ".join(str(a) for a in self.args))

    def as_unevaluated_basic(self):
        return self.func(*self.args, evaluate=False)

    @property
    def free_symbols(self):
        return set().union(*[a.free_symbols for a in self.args])

    __repr__ = __str__


def match_common_args(func_class, funcs, opt_subs):
    """
    Recognize and extract common subexpressions of function arguments within a
    set of function calls. For instance, for the following function calls::

        x + z + y
        sin(x + y)

    this will extract a common subexpression of `x + y`::

        w = x + y
        w + z
        sin(w)

    The function we work with is assumed to be associative and commutative.

    Parameters
    ==========

    func_class: class
        The function class (e.g. Add, Mul)
    funcs: list of functions
        A list of function calls.
    opt_subs: dict
        A dictionary of substitutions which this function may update.
    """

    # Sort to ensure that whole-function subexpressions come before the items
    # that use them.
    funcs = sorted(funcs, key=lambda f: len(f.args))
    arg_tracker = FuncArgTracker(funcs)

    changed = OrderedSet()

    for i in range(len(funcs)):
        common_arg_candidates_counts = arg_tracker.get_common_arg_candidates(
                arg_tracker.func_to_argset[i], min_func_i=i + 1)

        # Sort the candidates in order of match size.
        # This makes us try combining smaller matches first.
        common_arg_candidates = OrderedSet(sorted(
                common_arg_candidates_counts.keys(),
                key=lambda k: (common_arg_candidates_counts[k], k)))

        while common_arg_candidates:
            j = common_arg_candidates.pop(last=False)

            com_args = arg_tracker.func_to_argset[i].intersection(
                    arg_tracker.func_to_argset[j])

            if len(com_args) <= 1:
                # This may happen if a set of common arguments was already
                # combined in a previous iteration.
                continue

            # For all sets, replace the common symbols by the function
            # over them, to allow recursive matches.

            diff_i = arg_tracker.func_to_argset[i].difference(com_args)
            if diff_i:
                # com_func needs to be unevaluated to allow for recursive matches.
                com_func = Unevaluated(
                        func_class, arg_tracker.get_args_in_value_order(com_args))
                com_func_number = arg_tracker.get_or_add_value_number(com_func)
                arg_tracker.update_func_argset(i, diff_i | OrderedSet([com_func_number]))
                changed.add(i)
            else:
                # Treat the whole expression as a CSE.
                #
                # The reason this needs to be done is somewhat subtle. Within
                # tree_cse(), to_eliminate only contains expressions that are
                # seen more than once. The problem is unevaluated expressions
                # do not compare equal to the evaluated equivalent. So
                # tree_cse() won't mark funcs[i] as a CSE if we use an
                # unevaluated version.
                com_func_number = arg_tracker.get_or_add_value_number(funcs[i])

            diff_j = arg_tracker.func_to_argset[j].difference(com_args)
            arg_tracker.update_func_argset(j, diff_j | OrderedSet([com_func_number]))
            changed.add(j)

            for k in arg_tracker.get_subset_candidates(
                    com_args, common_arg_candidates):
                diff_k = arg_tracker.func_to_argset[k].difference(com_args)
                arg_tracker.update_func_argset(k, diff_k | OrderedSet([com_func_number]))
                changed.add(k)

        if i in changed:
            opt_subs[funcs[i]] = Unevaluated(func_class,
                arg_tracker.get_args_in_value_order(arg_tracker.func_to_argset[i]))

        arg_tracker.stop_arg_tracking(i)


def opt_cse(exprs, order='canonical'):
    """Find optimization opportunities in Adds, Muls, Pows and negative
    coefficient Muls.

    Parameters
    ==========

    exprs : list of SymPy expressions
        The expressions to optimize.
    order : string, 'none' or 'canonical'
        The order by which Mul and Add arguments are processed. For large
        expressions where speed is a concern, use the setting order='none'.

    Returns
    =======

    opt_subs : dictionary of expression substitutions
        The expression substitutions which can be useful to optimize CSE.

    Examples
    ========

    >>> from sympy.simplify.cse_main import opt_cse
    >>> from sympy.abc import x
    >>> opt_subs = opt_cse([x**-2])
    >>> k, v = list(opt_subs.keys())[0], list(opt_subs.values())[0]
    >>> print((k, v.as_unevaluated_basic()))
    (x**(-2), 1/(x**2))
    """
    opt_subs = {}

    adds = OrderedSet()
    muls = OrderedSet()

    seen_subexp = set()
    collapsible_subexp = set()

    def _find_opts(expr):

        if not isinstance(expr, (Basic, Unevaluated)):
            return

        if expr.is_Atom or expr.is_Order:
            return

        if iterable(expr):
            list(map(_find_opts, expr))
            return

        if expr in seen_subexp:
            return expr
        seen_subexp.add(expr)

        list(map(_find_opts, expr.args))

        if not isinstance(expr, MatrixExpr) and expr.could_extract_minus_sign():
            # XXX -expr does not always work rigorously for some expressions
            # containing UnevaluatedExpr.
            # https://github.com/sympy/sympy/issues/24818
            if isinstance(expr, Add):
                neg_expr = Add(*(-i for i in expr.args))
            else:
                neg_expr = -expr

            if not neg_expr.is_Atom:
                opt_subs[expr] = Unevaluated(Mul, (S.NegativeOne, neg_expr))
                seen_subexp.add(neg_expr)
                expr = neg_expr

        if isinstance(expr, (Mul, MatMul)):
            if len(expr.args) == 1:
                collapsible_subexp.add(expr)
            else:
                muls.add(expr)

        elif isinstance(expr, (Add, MatAdd)):
            if len(expr.args) == 1:
                collapsible_subexp.add(expr)
            else:
                adds.add(expr)

        elif isinstance(expr, Inverse):
            # Do not want to treat `Inverse` as a `MatPow`
            pass

        elif isinstance(expr, (Pow, MatPow)):
            base, exp = expr.base, expr.exp
            if exp.could_extract_minus_sign():
                opt_subs[expr] = Unevaluated(Pow, (Pow(base, -exp), -1))

    for e in exprs:
        if isinstance(e, (Basic, Unevaluated)):
            _find_opts(e)

    # Handle collapsing of multinary operations with single arguments
    edges = [(s, s.args[0]) for s in collapsible_subexp
             if s.args[0] in collapsible_subexp]
    for e in reversed(topological_sort((collapsible_subexp, edges))):
        opt_subs[e] = opt_subs.get(e.args[0], e.args[0])

    # split muls into commutative
    commutative_muls = OrderedSet()
    for m in muls:
        c, nc = m.args_cnc(cset=False)
        if c:
            c_mul = m.func(*c)
            if nc:
                if c_mul == 1:
                    new_obj = m.func(*nc)
                else:
                    if isinstance(m, MatMul):
                        new_obj = m.func(c_mul, *nc, evaluate=False)
                    else:
                        new_obj = m.func(c_mul, m.func(*nc), evaluate=False)
                opt_subs[m] = new_obj
            if len(c) > 1:
                commutative_muls.add(c_mul)

    match_common_args(Add, adds, opt_subs)
    match_common_args(Mul, commutative_muls, opt_subs)

    return opt_subs


def tree_cse(exprs, symbols, opt_subs=None, order='canonical', ignore=()):
    """Perform raw CSE on expression tree, taking opt_subs into account.

    Parameters
    ==========

    exprs : list of SymPy expressions
        The expressions to reduce.
    symbols : infinite iterator yielding unique Symbols
        The symbols used to label the common subexpressions which are pulled
        out.
    opt_subs : dictionary of expression substitutions
        The expressions to be substituted before any CSE action is performed.
    order : string, 'none' or 'canonical'
        The order by which Mul and Add arguments are processed. For large
        expressions where speed is a concern, use the setting order='none'.
    ignore : iterable of Symbols
        Substitutions containing any Symbol from ``ignore`` will be ignored.
    """
    if opt_subs is None:
        opt_subs = {}

    ## Find repeated sub-expressions

    to_eliminate = set()

    seen_subexp = set()
    excluded_symbols = set()

    def _find_repeated(expr):
        if not isinstance(expr, (Basic, Unevaluated)):
            return

        if isinstance(expr, RootOf):
            return

        if isinstance(expr, Basic) and (
                expr.is_Atom or
                expr.is_Order or
                isinstance(expr, (MatrixSymbol, MatrixElement))):
            if expr.is_Symbol:
                excluded_symbols.add(expr.name)
            return

        if iterable(expr):
            args = expr

        else:
            if expr in seen_subexp:
                for ign in ignore:
                    if ign in expr.free_symbols:
                        break
                else:
                    to_eliminate.add(expr)
                    return

            seen_subexp.add(expr)

            if expr in opt_subs:
                expr = opt_subs[expr]

            args = expr.args

        list(map(_find_repeated, args))

    for e in exprs:
        if isinstance(e, Basic):
            _find_repeated(e)

    ## Rebuild tree

    # Remove symbols from the generator that conflict with names in the expressions.
    symbols = (_ for _ in symbols if _.name not in excluded_symbols)

    replacements = []

    subs = {}

    def _rebuild(expr):
        if not isinstance(expr, (Basic, Unevaluated)):
            return expr

        if not expr.args:
            return expr

        if iterable(expr):
            new_args = [_rebuild(arg) for arg in expr.args]
            return expr.func(*new_args)

        if expr in subs:
            return subs[expr]

        orig_expr = expr
        if expr in opt_subs:
            expr = opt_subs[expr]

        # If enabled, parse Muls and Adds arguments by order to ensure
        # replacement order independent from hashes
        if order != 'none':
            if isinstance(expr, (Mul, MatMul)):
                c, nc = expr.args_cnc()
                if c == [1]:
                    args = nc
                else:
                    args = list(ordered(c)) + nc
            elif isinstance(expr, (Add, MatAdd)):
                args = list(ordered(expr.args))
            else:
                args = expr.args
        else:
            args = expr.args

        new_args = list(map(_rebuild, args))
        if isinstance(expr, Unevaluated) or new_args != args:
            new_expr = expr.func(*new_args)
        else:
            new_expr = expr

        if orig_expr in to_eliminate:
            try:
                sym = next(symbols)
            except StopIteration:
                raise ValueError("Symbols iterator ran out of symbols.")

            if isinstance(orig_expr, MatrixExpr):
                sym = MatrixSymbol(sym.name, orig_expr.rows,
                    orig_expr.cols)

            subs[orig_expr] = sym
            replacements.append((sym, new_expr))
            return sym

        else:
            return new_expr

    reduced_exprs = []
    for e in exprs:
        if isinstance(e, Basic):
            reduced_e = _rebuild(e)
        else:
            reduced_e = e
        reduced_exprs.append(reduced_e)
    return replacements, reduced_exprs


def cse(exprs, symbols=None, optimizations=None, postprocess=None,
        order='canonical', ignore=(), list=True):
    """ Perform common subexpression elimination on an expression.

    Parameters
    ==========

    exprs : list of SymPy expressions, or a single SymPy expression
        The expressions to reduce.
    symbols : infinite iterator yielding unique Symbols
        The symbols used to label the common subexpressions which are pulled
        out. The ``numbered_symbols`` generator is useful. The default is a
        stream of symbols of the form "x0", "x1", etc. This must be an
        infinite iterator.
    optimizations : list of (callable, callable) pairs
        The (preprocessor, postprocessor) pairs of external optimization
        functions. Optionally 'basic' can be passed for a set of predefined
        basic optimizations. Such 'basic' optimizations were used by default
        in old implementation, however they can be really slow on larger
        expressions. Now, no pre or post optimizations are made by default.
    postprocess : a function which accepts the two return values of cse and
        returns the desired form of output from cse, e.g. if you want the
        replacements reversed the function might be the following lambda:
        lambda r, e: return reversed(r), e
    order : string, 'none' or 'canonical'
        The order by which Mul and Add arguments are processed. If set to
        'canonical', arguments will be canonically ordered. If set to 'none',
        ordering will be faster but dependent on expressions hashes, thus
        machine dependent and variable. For large expressions where speed is a
        concern, use the setting order='none'.
    ignore : iterable of Symbols
        Substitutions containing any Symbol from ``ignore`` will be ignored.
    list : bool, (default True)
        Returns expression in list or else with same type as input (when False).

    Returns
    =======

    replacements : list of (Symbol, expression) pairs
        All of the common subexpressions that were replaced. Subexpressions
        earlier in this list might show up in subexpressions later in this
        list.
    reduced_exprs : list of SymPy expressions
        The reduced expressions with all of the replacements above.

    Examples
    ========

    >>> from sympy import cse, SparseMatrix
    >>> from sympy.abc import x, y, z, w
    >>> cse(((w + x + y + z)*(w + y + z))/(w + x)**3)
    ([(x0, y + z), (x1, w + x)], [(w + x0)*(x0 + x1)/x1**3])


    List of expressions with recursive substitutions:

    >>> m = SparseMatrix([x + y, x + y + z])
    >>> cse([(x+y)**2, x + y + z, y + z, x + z + y, m])
    ([(x0, x + y), (x1, x0 + z)], [x0**2, x1, y + z, x1, Matrix([
    [x0],
    [x1]])])

    Note: the type and mutability of input matrices is retained.

    >>> isinstance(_[1][-1], SparseMatrix)
    True

    The user may disallow substitutions containing certain symbols:

    >>> cse([y**2*(x + 1), 3*y**2*(x + 1)], ignore=(y,))
    ([(x0, x + 1)], [x0*y**2, 3*x0*y**2])

    The default return value for the reduced expression(s) is a list, even if there is only
    one expression. The `list` flag preserves the type of the input in the output:

    >>> cse(x)
    ([], [x])
    >>> cse(x, list=False)
    ([], x)
    """
    if not list:
        return _cse_homogeneous(exprs,
            symbols=symbols, optimizations=optimizations,
            postprocess=postprocess, order=order, ignore=ignore)

    if isinstance(exprs, (int, float)):
        exprs = sympify(exprs)

    # Handle the case if just one expression was passed.
    if isinstance(exprs, (Basic, MatrixBase)):
        exprs = [exprs]

    copy = exprs
    temp = []
    for e in exprs:
        if isinstance(e, (Matrix, ImmutableMatrix)):
            temp.append(Tuple(*e.flat()))
        elif isinstance(e, (SparseMatrix, ImmutableSparseMatrix)):
            temp.append(Tuple(*e.todok().items()))
        else:
            temp.append(e)
    exprs = temp
    del temp

    if optimizations is None:
        optimizations = []
    elif optimizations == 'basic':
        optimizations = basic_optimizations

    # Preprocess the expressions to give us better optimization opportunities.
    reduced_exprs = [preprocess_for_cse(e, optimizations) for e in exprs]

    if symbols is None:
        symbols = numbered_symbols(cls=Symbol)
    else:
        # In case we get passed an iterable with an __iter__ method instead of
        # an actual iterator.
        symbols = iter(symbols)

    # Find other optimization opportunities.
    opt_subs = opt_cse(reduced_exprs, order)

    # Main CSE algorithm.
    replacements, reduced_exprs = tree_cse(reduced_exprs, symbols, opt_subs,
                                           order, ignore)

    # Postprocess the expressions to return the expressions to canonical form.
    exprs = copy
    replacements = [(sym, postprocess_for_cse(subtree, optimizations))
                    for sym, subtree in replacements]
    reduced_exprs = [postprocess_for_cse(e, optimizations)
                     for e in reduced_exprs]

    # Get the matrices back
    for i, e in enumerate(exprs):
        if isinstance(e, (Matrix, ImmutableMatrix)):
            reduced_exprs[i] = Matrix(e.rows, e.cols, reduced_exprs[i])
            if isinstance(e, ImmutableMatrix):
                reduced_exprs[i] = reduced_exprs[i].as_immutable()
        elif isinstance(e, (SparseMatrix, ImmutableSparseMatrix)):
            m = SparseMatrix(e.rows, e.cols, {})
            for k, v in reduced_exprs[i]:
                m[k] = v
            if isinstance(e, ImmutableSparseMatrix):
                m = m.as_immutable()
            reduced_exprs[i] = m

    if postprocess is None:
        return replacements, reduced_exprs

    return postprocess(replacements, reduced_exprs)


def _cse_homogeneous(exprs, **kwargs):
    """
    Same as ``cse`` but the ``reduced_exprs`` are returned
    with the same type as ``exprs`` or a sympified version of the same.

    Parameters
    ==========

    exprs : an Expr, iterable of Expr or dictionary with Expr values
        the expressions in which repeated subexpressions will be identified
    kwargs : additional arguments for the ``cse`` function

    Returns
    =======

    replacements : list of (Symbol, expression) pairs
        All of the common subexpressions that were replaced. Subexpressions
        earlier in this list might show up in subexpressions later in this
        list.
    reduced_exprs : list of SymPy expressions
        The reduced expressions with all of the replacements above.

    Examples
    ========

    >>> from sympy.simplify.cse_main import cse
    >>> from sympy import cos, Tuple, Matrix
    >>> from sympy.abc import x
    >>> output = lambda x: type(cse(x, list=False)[1])
    >>> output(1)
    <class 'sympy.core.numbers.One'>
    >>> output('cos(x)')
    <class 'str'>
    >>> output(cos(x))
    cos
    >>> output(Tuple(1, x))
    <class 'sympy.core.containers.Tuple'>
    >>> output(Matrix([[1,0], [0,1]]))
    <class 'sympy.matrices.dense.MutableDenseMatrix'>
    >>> output([1, x])
    <class 'list'>
    >>> output((1, x))
    <class 'tuple'>
    >>> output({1, x})
    <class 'set'>
    """
    if isinstance(exprs, str):
        replacements, reduced_exprs = _cse_homogeneous(
            sympify(exprs), **kwargs)
        return replacements, repr(reduced_exprs)
    if isinstance(exprs, (list, tuple, set)):
        replacements, reduced_exprs = cse(exprs, **kwargs)
        return replacements, type(exprs)(reduced_exprs)
    if isinstance(exprs, dict):
        keys = list(exprs.keys()) # In order to guarantee the order of the elements.
        replacements, values = cse([exprs[k] for k in keys], **kwargs)
        reduced_exprs = dict(zip(keys, values))
        return replacements, reduced_exprs

    try:
        replacements, (reduced_exprs,) = cse(exprs, **kwargs)
    except TypeError: # For example 'mpf' objects
        return [], exprs
    else:
        return replacements, reduced_exprs

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\joint_rv_types.py ===
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import Lambda
from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, Rational, pi)
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (rf, factorial)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.bessel import besselk
from sympy.functions.special.gamma_functions import gamma
from sympy.matrices.dense import (Matrix, ones)
from sympy.sets.fancysets import Range
from sympy.sets.sets import (Intersection, Interval)
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.matrices import ImmutableMatrix, MatrixSymbol
from sympy.matrices.expressions.determinant import det
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.stats.joint_rv import JointDistribution, JointPSpace, MarginalDistribution
from sympy.stats.rv import _value_check, random_symbols

__all__ = ['JointRV',
'MultivariateNormal',
'MultivariateLaplace',
'Dirichlet',
'GeneralizedMultivariateLogGamma',
'GeneralizedMultivariateLogGammaOmega',
'Multinomial',
'MultivariateBeta',
'MultivariateEwens',
'MultivariateT',
'NegativeMultinomial',
'NormalGamma'
]

def multivariate_rv(cls, sym, *args):
    args = list(map(sympify, args))
    dist = cls(*args)
    args = dist.args
    dist.check(*args)
    return JointPSpace(sym, dist).value


def marginal_distribution(rv, *indices):
    """
    Marginal distribution function of a joint random variable.

    Parameters
    ==========

    rv : A random variable with a joint probability distribution.
    indices : Component indices or the indexed random symbol
        for which the joint distribution is to be calculated

    Returns
    =======

    A Lambda expression in `sym`.

    Examples
    ========

    >>> from sympy.stats import MultivariateNormal, marginal_distribution
    >>> m = MultivariateNormal('X', [1, 2], [[2, 1], [1, 2]])
    >>> marginal_distribution(m, m[0])(1)
    1/(2*sqrt(pi))

    """
    indices = list(indices)
    for i in range(len(indices)):
        if isinstance(indices[i], Indexed):
            indices[i] = indices[i].args[1]
    prob_space = rv.pspace
    if not indices:
        raise ValueError(
            "At least one component for marginal density is needed.")
    if hasattr(prob_space.distribution, '_marginal_distribution'):
        return prob_space.distribution._marginal_distribution(indices, rv.symbol)
    return prob_space.marginal_distribution(*indices)


class JointDistributionHandmade(JointDistribution):

    _argnames = ('pdf',)
    is_Continuous = True

    @property
    def set(self):
        return self.args[1]


def JointRV(symbol, pdf, _set=None):
    """
    Create a Joint Random Variable where each of its component is continuous,
    given the following:

    Parameters
    ==========

    symbol : Symbol
        Represents name of the random variable.
    pdf : A PDF in terms of indexed symbols of the symbol given
        as the first argument

    NOTE
    ====

    As of now, the set for each component for a ``JointRV`` is
    equal to the set of all integers, which cannot be changed.

    Examples
    ========

    >>> from sympy import exp, pi, Indexed, S
    >>> from sympy.stats import density, JointRV
    >>> x1, x2 = (Indexed('x', i) for i in (1, 2))
    >>> pdf = exp(-x1**2/2 + x1 - x2**2/2 - S(1)/2)/(2*pi)
    >>> N1 = JointRV('x', pdf) #Multivariate Normal distribution
    >>> density(N1)(1, 2)
    exp(-2)/(2*pi)

    Returns
    =======

    RandomSymbol

    """
    #TODO: Add support for sets provided by the user
    symbol = sympify(symbol)
    syms = [i for i in pdf.free_symbols if isinstance(i, Indexed)
        and i.base == IndexedBase(symbol)]
    syms = tuple(sorted(syms, key = lambda index: index.args[1]))
    _set = S.Reals**len(syms)
    pdf = Lambda(syms, pdf)
    dist = JointDistributionHandmade(pdf, _set)
    jrv = JointPSpace(symbol, dist).value
    rvs = random_symbols(pdf)
    if len(rvs) != 0:
        dist = MarginalDistribution(dist, (jrv,))
        return JointPSpace(symbol, dist).value
    return jrv

#-------------------------------------------------------------------------------
# Multivariate Normal distribution ---------------------------------------------

class MultivariateNormalDistribution(JointDistribution):
    _argnames = ('mu', 'sigma')

    is_Continuous=True

    @property
    def set(self):
        k = self.mu.shape[0]
        return S.Reals**k

    @staticmethod
    def check(mu, sigma):
        _value_check(mu.shape[0] == sigma.shape[0],
            "Size of the mean vector and covariance matrix are incorrect.")
        #check if covariance matrix is positive semi definite or not.
        if not isinstance(sigma, MatrixSymbol):
            _value_check(sigma.is_positive_semidefinite,
            "The covariance matrix must be positive semi definite. ")

    def pdf(self, *args):
        mu, sigma = self.mu, self.sigma
        k = mu.shape[0]
        if len(args) == 1 and args[0].is_Matrix:
            args = args[0]
        else:
            args = ImmutableMatrix(args)
        x = args - mu
        density = S.One/sqrt((2*pi)**(k)*det(sigma))*exp(
            Rational(-1, 2)*x.transpose()*(sigma.inv()*x))
        return MatrixElement(density, 0, 0)

    def _marginal_distribution(self, indices, sym):
        sym = ImmutableMatrix([Indexed(sym, i) for i in indices])
        _mu, _sigma = self.mu, self.sigma
        k = self.mu.shape[0]
        for i in range(k):
            if i not in indices:
                _mu = _mu.row_del(i)
                _sigma = _sigma.col_del(i)
                _sigma = _sigma.row_del(i)
        return Lambda(tuple(sym), S.One/sqrt((2*pi)**(len(_mu))*det(_sigma))*exp(
            Rational(-1, 2)*(_mu - sym).transpose()*(_sigma.inv()*\
                (_mu - sym)))[0])

def MultivariateNormal(name, mu, sigma):
    r"""
    Creates a continuous random variable with Multivariate Normal
    Distribution.

    The density of the multivariate normal distribution can be found at [1].

    Parameters
    ==========

    mu : List representing the mean or the mean vector
    sigma : Positive semidefinite square matrix
        Represents covariance Matrix.
        If `\sigma` is noninvertible then only sampling is supported currently

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import MultivariateNormal, density, marginal_distribution
    >>> from sympy import symbols, MatrixSymbol
    >>> X = MultivariateNormal('X', [3, 4], [[2, 1], [1, 2]])
    >>> y, z = symbols('y z')
    >>> density(X)(y, z)
    sqrt(3)*exp(-y**2/3 + y*z/3 + 2*y/3 - z**2/3 + 5*z/3 - 13/3)/(6*pi)
    >>> density(X)(1, 2)
    sqrt(3)*exp(-4/3)/(6*pi)
    >>> marginal_distribution(X, X[1])(y)
    exp(-(y - 4)**2/4)/(2*sqrt(pi))
    >>> marginal_distribution(X, X[0])(y)
    exp(-(y - 3)**2/4)/(2*sqrt(pi))

    The example below shows that it is also possible to use
    symbolic parameters to define the MultivariateNormal class.

    >>> n = symbols('n', integer=True, positive=True)
    >>> Sg = MatrixSymbol('Sg', n, n)
    >>> mu = MatrixSymbol('mu', n, 1)
    >>> obs = MatrixSymbol('obs', n, 1)
    >>> X = MultivariateNormal('X', mu, Sg)

    The density of a multivariate normal can be
    calculated using a matrix argument, as shown below.

    >>> density(X)(obs)
    (exp(((1/2)*mu.T - (1/2)*obs.T)*Sg**(-1)*(-mu + obs))/sqrt((2*pi)**n*Determinant(Sg)))[0, 0]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Multivariate_normal_distribution

    """
    return multivariate_rv(MultivariateNormalDistribution, name, mu, sigma)

#-------------------------------------------------------------------------------
# Multivariate Laplace distribution --------------------------------------------

class MultivariateLaplaceDistribution(JointDistribution):
    _argnames = ('mu', 'sigma')
    is_Continuous=True

    @property
    def set(self):
        k = self.mu.shape[0]
        return S.Reals**k

    @staticmethod
    def check(mu, sigma):
        _value_check(mu.shape[0] == sigma.shape[0],
                     "Size of the mean vector and covariance matrix are incorrect.")
        # check if covariance matrix is positive definite or not.
        if not isinstance(sigma, MatrixSymbol):
            _value_check(sigma.is_positive_definite,
                         "The covariance matrix must be positive definite. ")

    def pdf(self, *args):
        mu, sigma = self.mu, self.sigma
        mu_T = mu.transpose()
        k = S(mu.shape[0])
        sigma_inv = sigma.inv()
        args = ImmutableMatrix(args)
        args_T = args.transpose()
        x = (mu_T*sigma_inv*mu)[0]
        y = (args_T*sigma_inv*args)[0]
        v = 1 - k/2
        return (2 * (y/(2 + x))**(v/2) * besselk(v, sqrt((2 + x)*y)) *
                exp((args_T * sigma_inv * mu)[0]) /
                ((2 * pi)**(k/2) * sqrt(det(sigma))))


def MultivariateLaplace(name, mu, sigma):
    """
    Creates a continuous random variable with Multivariate Laplace
    Distribution.

    The density of the multivariate Laplace distribution can be found at [1].

    Parameters
    ==========

    mu : List representing the mean or the mean vector
    sigma : Positive definite square matrix
        Represents covariance Matrix

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import MultivariateLaplace, density
    >>> from sympy import symbols
    >>> y, z = symbols('y z')
    >>> X = MultivariateLaplace('X', [2, 4], [[3, 1], [1, 3]])
    >>> density(X)(y, z)
    sqrt(2)*exp(y/4 + 5*z/4)*besselk(0, sqrt(15*y*(3*y/8 - z/8)/2 + 15*z*(-y/8 + 3*z/8)/2))/(4*pi)
    >>> density(X)(1, 2)
    sqrt(2)*exp(11/4)*besselk(0, sqrt(165)/4)/(4*pi)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Multivariate_Laplace_distribution

    """
    return multivariate_rv(MultivariateLaplaceDistribution, name, mu, sigma)

#-------------------------------------------------------------------------------
# Multivariate StudentT distribution -------------------------------------------

class MultivariateTDistribution(JointDistribution):
    _argnames = ('mu', 'shape_mat', 'dof')
    is_Continuous=True

    @property
    def set(self):
        k = self.mu.shape[0]
        return S.Reals**k

    @staticmethod
    def check(mu, sigma, v):
        _value_check(mu.shape[0] == sigma.shape[0],
                     "Size of the location vector and shape matrix are incorrect.")
        # check if covariance matrix is positive definite or not.
        if not isinstance(sigma, MatrixSymbol):
            _value_check(sigma.is_positive_definite,
                         "The shape matrix must be positive definite. ")

    def pdf(self, *args):
        mu, sigma = self.mu, self.shape_mat
        v = S(self.dof)
        k = S(mu.shape[0])
        sigma_inv = sigma.inv()
        args = ImmutableMatrix(args)
        x = args - mu
        return gamma((k + v)/2)/(gamma(v/2)*(v*pi)**(k/2)*sqrt(det(sigma)))\
        *(1 + 1/v*(x.transpose()*sigma_inv*x)[0])**((-v - k)/2)

def MultivariateT(syms, mu, sigma, v):
    """
    Creates a joint random variable with multivariate T-distribution.

    Parameters
    ==========

    syms : A symbol/str
        For identifying the random variable.
    mu : A list/matrix
        Representing the location vector
    sigma : The shape matrix for the distribution

    Examples
    ========

    >>> from sympy.stats import density, MultivariateT
    >>> from sympy import Symbol

    >>> x = Symbol("x")
    >>> X = MultivariateT("x", [1, 1], [[1, 0], [0, 1]], 2)

    >>> density(X)(1, 2)
    2/(9*pi)

    Returns
    =======

    RandomSymbol

    """
    return multivariate_rv(MultivariateTDistribution, syms, mu, sigma, v)


#-------------------------------------------------------------------------------
# Multivariate Normal Gamma distribution ---------------------------------------

class NormalGammaDistribution(JointDistribution):

    _argnames = ('mu', 'lamda', 'alpha', 'beta')
    is_Continuous=True

    @staticmethod
    def check(mu, lamda, alpha, beta):
        _value_check(mu.is_real, "Location must be real.")
        _value_check(lamda > 0, "Lambda must be positive")
        _value_check(alpha > 0, "alpha must be positive")
        _value_check(beta > 0, "beta must be positive")

    @property
    def set(self):
        return S.Reals*Interval(0, S.Infinity)

    def pdf(self, x, tau):
        beta, alpha, lamda = self.beta, self.alpha, self.lamda
        mu = self.mu

        return beta**alpha*sqrt(lamda)/(gamma(alpha)*sqrt(2*pi))*\
        tau**(alpha - S.Half)*exp(-1*beta*tau)*\
        exp(-1*(lamda*tau*(x - mu)**2)/S(2))

    def _marginal_distribution(self, indices, *sym):
        if len(indices) == 2:
            return self.pdf(*sym)
        if indices[0] == 0:
            #For marginal over `x`, return non-standardized Student-T's
            #distribution
            x = sym[0]
            v, mu, sigma = self.alpha - S.Half, self.mu, \
                S(self.beta)/(self.lamda * self.alpha)
            return Lambda(sym, gamma((v + 1)/2)/(gamma(v/2)*sqrt(pi*v)*sigma)*\
                (1 + 1/v*((x - mu)/sigma)**2)**((-v -1)/2))
        #For marginal over `tau`, return Gamma distribution as per construction
        from sympy.stats.crv_types import GammaDistribution
        return Lambda(sym, GammaDistribution(self.alpha, self.beta)(sym[0]))

def NormalGamma(sym, mu, lamda, alpha, beta):
    """
    Creates a bivariate joint random variable with multivariate Normal gamma
    distribution.

    Parameters
    ==========

    sym : A symbol/str
        For identifying the random variable.
    mu : A real number
        The mean of the normal distribution
    lamda : A positive integer
        Parameter of joint distribution
    alpha : A positive integer
        Parameter of joint distribution
    beta : A positive integer
        Parameter of joint distribution

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, NormalGamma
    >>> from sympy import symbols

    >>> X = NormalGamma('x', 0, 1, 2, 3)
    >>> y, z = symbols('y z')

    >>> density(X)(y, z)
    9*sqrt(2)*z**(3/2)*exp(-3*z)*exp(-y**2*z/2)/(2*sqrt(pi))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Normal-gamma_distribution

    """
    return multivariate_rv(NormalGammaDistribution, sym, mu, lamda, alpha, beta)

#-------------------------------------------------------------------------------
# Multivariate Beta/Dirichlet distribution -------------------------------------

class MultivariateBetaDistribution(JointDistribution):

    _argnames = ('alpha',)
    is_Continuous = True

    @staticmethod
    def check(alpha):
        _value_check(len(alpha) >= 2, "At least two categories should be passed.")
        for a_k in alpha:
            _value_check((a_k > 0) != False, "Each concentration parameter"
                                            " should be positive.")

    @property
    def set(self):
        k = len(self.alpha)
        return Interval(0, 1)**k

    def pdf(self, *syms):
        alpha = self.alpha
        B = Mul.fromiter(map(gamma, alpha))/gamma(Add(*alpha))
        return Mul.fromiter(sym**(a_k - 1) for a_k, sym in zip(alpha, syms))/B

def MultivariateBeta(syms, *alpha):
    """
    Creates a continuous random variable with Dirichlet/Multivariate Beta
    Distribution.

    The density of the Dirichlet distribution can be found at [1].

    Parameters
    ==========

    alpha : Positive real numbers
        Signifies concentration numbers.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, MultivariateBeta, marginal_distribution
    >>> from sympy import Symbol
    >>> a1 = Symbol('a1', positive=True)
    >>> a2 = Symbol('a2', positive=True)
    >>> B = MultivariateBeta('B', [a1, a2])
    >>> C = MultivariateBeta('C', a1, a2)
    >>> x = Symbol('x')
    >>> y = Symbol('y')
    >>> density(B)(x, y)
    x**(a1 - 1)*y**(a2 - 1)*gamma(a1 + a2)/(gamma(a1)*gamma(a2))
    >>> marginal_distribution(C, C[0])(x)
    x**(a1 - 1)*gamma(a1 + a2)/(a2*gamma(a1)*gamma(a2))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Dirichlet_distribution
    .. [2] https://mathworld.wolfram.com/DirichletDistribution.html

    """
    if not isinstance(alpha[0], list):
        alpha = (list(alpha),)
    return multivariate_rv(MultivariateBetaDistribution, syms, alpha[0])

Dirichlet = MultivariateBeta

#-------------------------------------------------------------------------------
# Multivariate Ewens distribution ----------------------------------------------

class MultivariateEwensDistribution(JointDistribution):

    _argnames = ('n', 'theta')
    is_Discrete = True
    is_Continuous = False

    @staticmethod
    def check(n, theta):
        _value_check((n > 0),
                        "sample size should be positive integer.")
        _value_check(theta.is_positive, "mutation rate should be positive.")

    @property
    def set(self):
        if not isinstance(self.n, Integer):
            i = Symbol('i', integer=True, positive=True)
            return Product(Intersection(S.Naturals0, Interval(0, self.n//i)),
                                    (i, 1, self.n))
        prod_set = Range(0, self.n + 1)
        for i in range(2, self.n + 1):
            prod_set *= Range(0, self.n//i + 1)
        return prod_set.flatten()

    def pdf(self, *syms):
        n, theta = self.n, self.theta
        condi = isinstance(self.n, Integer)
        if not (isinstance(syms[0], IndexedBase) or condi):
            raise ValueError("Please use IndexedBase object for syms as "
                                "the dimension is symbolic")
        term_1 = factorial(n)/rf(theta, n)
        if condi:
            term_2 = Mul.fromiter(theta**syms[j]/((j+1)**syms[j]*factorial(syms[j]))
                                    for j in range(n))
            cond = Eq(sum((k + 1)*syms[k] for k in range(n)), n)
            return Piecewise((term_1 * term_2, cond), (0, True))
        syms = syms[0]
        j, k = symbols('j, k', positive=True, integer=True)
        term_2 = Product(theta**syms[j]/((j+1)**syms[j]*factorial(syms[j])),
                            (j, 0, n - 1))
        cond = Eq(Sum((k + 1)*syms[k], (k, 0, n - 1)), n)
        return Piecewise((term_1 * term_2, cond), (0, True))


def MultivariateEwens(syms, n, theta):
    """
    Creates a discrete random variable with Multivariate Ewens
    Distribution.

    The density of the said distribution can be found at [1].

    Parameters
    ==========

    n : Positive integer
        Size of the sample or the integer whose partitions are considered
    theta : Positive real number
        Denotes Mutation rate

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, marginal_distribution, MultivariateEwens
    >>> from sympy import Symbol
    >>> a1 = Symbol('a1', positive=True)
    >>> a2 = Symbol('a2', positive=True)
    >>> ed = MultivariateEwens('E', 2, 1)
    >>> density(ed)(a1, a2)
    Piecewise((1/(2**a2*factorial(a1)*factorial(a2)), Eq(a1 + 2*a2, 2)), (0, True))
    >>> marginal_distribution(ed, ed[0])(a1)
    Piecewise((1/factorial(a1), Eq(a1, 2)), (0, True))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Ewens%27s_sampling_formula
    .. [2] https://www.jstor.org/stable/24780825
    """
    return multivariate_rv(MultivariateEwensDistribution, syms, n, theta)

#-------------------------------------------------------------------------------
# Generalized Multivariate Log Gamma distribution ------------------------------

class GeneralizedMultivariateLogGammaDistribution(JointDistribution):

    _argnames = ('delta', 'v', 'lamda', 'mu')
    is_Continuous=True

    def check(self, delta, v, l, mu):
        _value_check((delta >= 0, delta <= 1), "delta must be in range [0, 1].")
        _value_check((v > 0), "v must be positive")
        for lk in l:
            _value_check((lk > 0), "lamda must be a positive vector.")
        for muk in mu:
            _value_check((muk > 0), "mu must be a positive vector.")
        _value_check(len(l) > 1,"the distribution should have at least"
                                " two random variables.")

    @property
    def set(self):
        return S.Reals**len(self.lamda)

    def pdf(self, *y):
        d, v, l, mu = self.delta, self.v, self.lamda, self.mu
        n = Symbol('n', negative=False, integer=True)
        k = len(l)
        sterm1 = Pow((1 - d), n)/\
                ((gamma(v + n)**(k - 1))*gamma(v)*gamma(n + 1))
        sterm2 = Mul.fromiter(mui*li**(-v - n) for mui, li in zip(mu, l))
        term1 = sterm1 * sterm2
        sterm3 = (v + n) * sum(mui * yi for mui, yi in zip(mu, y))
        sterm4 = sum(exp(mui * yi)/li for (mui, yi, li) in zip(mu, y, l))
        term2 = exp(sterm3 - sterm4)
        return Pow(d, v) * Sum(term1 * term2, (n, 0, S.Infinity))

def GeneralizedMultivariateLogGamma(syms, delta, v, lamda, mu):
    """
    Creates a joint random variable with generalized multivariate log gamma
    distribution.

    The joint pdf can be found at [1].

    Parameters
    ==========

    syms : list/tuple/set of symbols for identifying each component
    delta : A constant in range $[0, 1]$
    v : Positive real number
    lamda : List of positive real numbers
    mu : List of positive real numbers

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density
    >>> from sympy.stats.joint_rv_types import GeneralizedMultivariateLogGamma
    >>> from sympy import symbols, S
    >>> v = 1
    >>> l, mu = [1, 1, 1], [1, 1, 1]
    >>> d = S.Half
    >>> y = symbols('y_1:4', positive=True)
    >>> Gd = GeneralizedMultivariateLogGamma('G', d, v, l, mu)
    >>> density(Gd)(y[0], y[1], y[2])
    Sum(exp((n + 1)*(y_1 + y_2 + y_3) - exp(y_1) - exp(y_2) -
    exp(y_3))/(2**n*gamma(n + 1)**3), (n, 0, oo))/2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Generalized_multivariate_log-gamma_distribution
    .. [2] https://www.researchgate.net/publication/234137346_On_a_multivariate_log-gamma_distribution_and_the_use_of_the_distribution_in_the_Bayesian_analysis

    Note
    ====

    If the GeneralizedMultivariateLogGamma is too long to type use,

    >>> from sympy.stats.joint_rv_types import GeneralizedMultivariateLogGamma as GMVLG
    >>> Gd = GMVLG('G', d, v, l, mu)

    If you want to pass the matrix omega instead of the constant delta, then use
    ``GeneralizedMultivariateLogGammaOmega``.

    """
    return multivariate_rv(GeneralizedMultivariateLogGammaDistribution,
                            syms, delta, v, lamda, mu)

def GeneralizedMultivariateLogGammaOmega(syms, omega, v, lamda, mu):
    """
    Extends GeneralizedMultivariateLogGamma.

    Parameters
    ==========

    syms : list/tuple/set of symbols
        For identifying each component
    omega : A square matrix
           Every element of square matrix must be absolute value of
           square root of correlation coefficient
    v : Positive real number
    lamda : List of positive real numbers
    mu : List of positive real numbers

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density
    >>> from sympy.stats.joint_rv_types import GeneralizedMultivariateLogGammaOmega
    >>> from sympy import Matrix, symbols, S
    >>> omega = Matrix([[1, S.Half, S.Half], [S.Half, 1, S.Half], [S.Half, S.Half, 1]])
    >>> v = 1
    >>> l, mu = [1, 1, 1], [1, 1, 1]
    >>> G = GeneralizedMultivariateLogGammaOmega('G', omega, v, l, mu)
    >>> y = symbols('y_1:4', positive=True)
    >>> density(G)(y[0], y[1], y[2])
    sqrt(2)*Sum((1 - sqrt(2)/2)**n*exp((n + 1)*(y_1 + y_2 + y_3) - exp(y_1) -
    exp(y_2) - exp(y_3))/gamma(n + 1)**3, (n, 0, oo))/2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Generalized_multivariate_log-gamma_distribution
    .. [2] https://www.researchgate.net/publication/234137346_On_a_multivariate_log-gamma_distribution_and_the_use_of_the_distribution_in_the_Bayesian_analysis

    Notes
    =====

    If the GeneralizedMultivariateLogGammaOmega is too long to type use,

    >>> from sympy.stats.joint_rv_types import GeneralizedMultivariateLogGammaOmega as GMVLGO
    >>> G = GMVLGO('G', omega, v, l, mu)

    """
    _value_check((omega.is_square, isinstance(omega, Matrix)), "omega must be a"
                                                            " square matrix")
    for val in omega.values():
        _value_check((val >= 0, val <= 1),
            "all values in matrix must be between 0 and 1(both inclusive).")
    _value_check(omega.diagonal().equals(ones(1, omega.shape[0])),
                    "all the elements of diagonal should be 1.")
    _value_check((omega.shape[0] == len(lamda), len(lamda) == len(mu)),
                    "lamda, mu should be of same length and omega should "
                    " be of shape (length of lamda, length of mu)")
    _value_check(len(lamda) > 1,"the distribution should have at least"
                            " two random variables.")
    delta = Pow(Rational(omega.det()), Rational(1, len(lamda) - 1))
    return GeneralizedMultivariateLogGamma(syms, delta, v, lamda, mu)


#-------------------------------------------------------------------------------
# Multinomial distribution -----------------------------------------------------

class MultinomialDistribution(JointDistribution):

    _argnames = ('n', 'p')
    is_Continuous=False
    is_Discrete = True

    @staticmethod
    def check(n, p):
        _value_check(n > 0,
                        "number of trials must be a positive integer")
        for p_k in p:
            _value_check((p_k >= 0, p_k <= 1),
                        "probability must be in range [0, 1]")
        _value_check(Eq(sum(p), 1),
                        "probabilities must sum to 1")

    @property
    def set(self):
        return Intersection(S.Naturals0, Interval(0, self.n))**len(self.p)

    def pdf(self, *x):
        n, p = self.n, self.p
        term_1 = factorial(n)/Mul.fromiter(factorial(x_k) for x_k in x)
        term_2 = Mul.fromiter(p_k**x_k for p_k, x_k in zip(p, x))
        return Piecewise((term_1 * term_2, Eq(sum(x), n)), (0, True))

def Multinomial(syms, n, *p):
    """
    Creates a discrete random variable with Multinomial Distribution.

    The density of the said distribution can be found at [1].

    Parameters
    ==========

    n : Positive integer
        Represents number of trials
    p : List of event probabilities
        Must be in the range of $[0, 1]$.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, Multinomial, marginal_distribution
    >>> from sympy import symbols
    >>> x1, x2, x3 = symbols('x1, x2, x3', nonnegative=True, integer=True)
    >>> p1, p2, p3 = symbols('p1, p2, p3', positive=True)
    >>> M = Multinomial('M', 3, p1, p2, p3)
    >>> density(M)(x1, x2, x3)
    Piecewise((6*p1**x1*p2**x2*p3**x3/(factorial(x1)*factorial(x2)*factorial(x3)),
    Eq(x1 + x2 + x3, 3)), (0, True))
    >>> marginal_distribution(M, M[0])(x1).subs(x1, 1)
    3*p1*p2**2 + 6*p1*p2*p3 + 3*p1*p3**2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Multinomial_distribution
    .. [2] https://mathworld.wolfram.com/MultinomialDistribution.html

    """
    if not isinstance(p[0], list):
        p = (list(p), )
    return multivariate_rv(MultinomialDistribution, syms, n, p[0])

#-------------------------------------------------------------------------------
# Negative Multinomial Distribution --------------------------------------------

class NegativeMultinomialDistribution(JointDistribution):

    _argnames = ('k0', 'p')
    is_Continuous=False
    is_Discrete = True

    @staticmethod
    def check(k0, p):
        _value_check(k0 > 0,
                        "number of failures must be a positive integer")
        for p_k in p:
            _value_check((p_k >= 0, p_k <= 1),
                        "probability must be in range [0, 1].")
        _value_check(sum(p) <= 1,
                        "success probabilities must not be greater than 1.")

    @property
    def set(self):
        return Range(0, S.Infinity)**len(self.p)

    def pdf(self, *k):
        k0, p = self.k0, self.p
        term_1 = (gamma(k0 + sum(k))*(1 - sum(p))**k0)/gamma(k0)
        term_2 = Mul.fromiter(pi**ki/factorial(ki) for pi, ki in zip(p, k))
        return term_1 * term_2

def NegativeMultinomial(syms, k0, *p):
    """
    Creates a discrete random variable with Negative Multinomial Distribution.

    The density of the said distribution can be found at [1].

    Parameters
    ==========

    k0 : positive integer
        Represents number of failures before the experiment is stopped
    p : List of event probabilities
        Must be in the range of $[0, 1]$

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, NegativeMultinomial, marginal_distribution
    >>> from sympy import symbols
    >>> x1, x2, x3 = symbols('x1, x2, x3', nonnegative=True, integer=True)
    >>> p1, p2, p3 = symbols('p1, p2, p3', positive=True)
    >>> N = NegativeMultinomial('M', 3, p1, p2, p3)
    >>> N_c = NegativeMultinomial('M', 3, 0.1, 0.1, 0.1)
    >>> density(N)(x1, x2, x3)
    p1**x1*p2**x2*p3**x3*(-p1 - p2 - p3 + 1)**3*gamma(x1 + x2 +
    x3 + 3)/(2*factorial(x1)*factorial(x2)*factorial(x3))
    >>> marginal_distribution(N_c, N_c[0])(1).evalf().round(2)
    0.25


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Negative_multinomial_distribution
    .. [2] https://mathworld.wolfram.com/NegativeBinomialDistribution.html

    """
    if not isinstance(p[0], list):
        p = (list(p), )
    return multivariate_rv(NegativeMultinomialDistribution, syms, k0, p[0])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\quantize.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations

import copy
import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import onnx

from .calibrate import CalibrationDataReader, CalibrationMethod, TensorsData, create_calibrator
from .onnx_quantizer import ONNXQuantizer
from .qdq_quantizer import QDQQuantizer
from .quant_utils import (
    MODEL_SIZE_THRESHOLD,
    QuantFormat,
    QuantizationMode,
    QuantType,
    load_model_with_shape_infer,
    model_has_pre_process_metadata,
    save_and_reload_model_with_shape_infer,
)
from .registry import IntegerOpsRegistry, QDQRegistry, QLinearOpsRegistry
from .tensor_quant_overrides import TensorQuantOverridesHelper


class QuantConfig:
    def __init__(
        self,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=None,
        nodes_to_quantize=None,
        nodes_to_exclude=None,
        per_channel=False,
        reduce_range=False,
        use_external_data_format=False,
    ):
        """
        This is the Base class for both Static and Dynamic Quantize Configuration
        Args:
            activation_type:
                quantization data type of activation. Please refer to
                https://onnxruntime.ai/docs/performance/quantization.html for more details on data type selection
            weight_type:
                quantization data type of weight. Please refer to
                https://onnxruntime.ai/docs/performance/quantization.html for more details on data type selection
            op_types_to_quantize:
                specify the types of operators to quantize, like ['Conv'] to quantize Conv only.
                It quantizes all supported operators by default.
            nodes_to_quantize:
                List of nodes names to quantize. When this list is not None only the nodes in this list
                are quantized.
                example:
                [
                    'Conv__224',
                    'Conv__252'
                ]
            nodes_to_exclude:
                List of nodes names to exclude. The nodes in this list will be excluded from quantization
                when it is not None.
            per_channel: quantize weights per channel
            reduce_range:
                quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine,
                especially for per-channel mode
            use_external_data_format: option used for large size (>2GB) model. Set to False by default.
        """

        nodes_to_exclude = nodes_to_exclude or []
        nodes_to_quantize = nodes_to_quantize or []
        op_types_to_quantize = op_types_to_quantize or []
        self.op_types_to_quantize = op_types_to_quantize
        self.per_channel = per_channel
        self.reduce_range = reduce_range
        self.weight_type = weight_type
        self.activation_type = activation_type
        self.nodes_to_quantize = nodes_to_quantize
        self.nodes_to_exclude = nodes_to_exclude
        self.use_external_data_format = use_external_data_format


class StaticQuantConfig(QuantConfig):
    def __init__(
        self,
        calibration_data_reader: CalibrationDataReader,
        calibrate_method=CalibrationMethod.MinMax,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=None,
        nodes_to_quantize=None,
        nodes_to_exclude=None,
        per_channel=False,
        reduce_range=False,
        use_external_data_format=False,
        calibration_providers=None,
        extra_options=None,
    ):
        """
        This is the derived class for static Quantize Configuration

        Args:
            calibration_data_reader:
                a calibration data reader. It enumerates calibration data and generates inputs for the original model.
            calibrate_method:
                Current calibration methods supported are MinMax, Entropy and Percentile.
            quant_format: QuantFormat{QOperator, QDQ}.
                QOperator format quantizes the model with quantized operators directly.
                QDQ format quantize the model by inserting QuantizeLinear/DeQuantizeLinear on the tensor.
            calibration_providers: Execution providers to run the session during calibration. Default is None which uses
                [ "CPUExecutionProvider" ].
            extra_options:
                key value pair dictionary for various options in different case. Current used:
                    extra.Sigmoid.nnapi = True/False  (Default is False)
                    ActivationSymmetric = True/False: symmetrize calibration data for activations (default is False).
                    WeightSymmetric = True/False: symmetrize calibration data for weights (default is True).
                    EnableSubgraph = True/False : Default is False. If enabled, subgraph will be quantized.
                                                  Dyanmic mode currently is supported. Will support more in future.
                    ForceQuantizeNoInputCheck = True/False :
                        By default, some latent operators like maxpool, transpose, do not quantize if their input is not
                        quantized already. Setting to True to force such operator always quantize input and so generate
                        quantized output. Also the True behavior could be disabled per node using the nodes_to_exclude.
                    MatMulConstBOnly = True/False:
                        Default is False for static mode. If enabled, only MatMul with const B will be quantized.
                    AddQDQPairToWeight = True/False :
                        Default is False which quantizes floating-point weight and feeds it to solely inserted
                        DeQuantizeLinear node. If True, it remains floating-point weight and inserts both
                        QuantizeLinear/DeQuantizeLinear nodes to weight.
                    OpTypesToExcludeOutputQuantization = list of op type :
                        Default is []. If any op type is specified, it won't quantize the output of ops with this
                        specific op types.
                    DedicatedQDQPair = True/False :
                        Default is False. When inserting QDQ pair, multiple nodes can share a single QDQ pair as their
                        inputs. If True, it will create identical and dedicated QDQ pair for each node.
                    QDQOpTypePerChannelSupportToAxis = dictionary :
                        Default is {}. Set channel axis for specific op type, for example: {'MatMul': 1}, and it's
                        effective only when per channel quantization is supported and per_channel is True. If specific
                        op type supports per channel quantization but not explicitly specified with channel axis,
                        default channel axis will be used.
                    CalibTensorRangeSymmetric = True/False :
                        Default is False. If enabled, the final range of tensor during calibration will be explicitly
                        set to symmetric to central point "0".
                    CalibMovingAverage = True/False :
                        Default is False. If enabled, the moving average of the minimum and maximum values will be
                        computed when the calibration method selected is MinMax.
                    CalibMovingAverageConstant = float :
                        Default is 0.01. Constant smoothing factor to use when computing the moving average of the
                        minimum and maximum values. Effective only when the calibration method selected is MinMax and
                        when CalibMovingAverage is set to True.
                    QuantizeBias = True/False :
                        Default is True which quantizes floating-point biases and it solely inserts
                        a DeQuantizeLinear node. If False, it remains floating-point bias and does not insert
                        any quantization nodes associated with biases.
                        This extra option is only effective when quant_format is QuantFormat.QDQ.
                    SmoothQuant = True/False :
                        Default is False. If enabled, SmoothQuant algorithm will be applied before quantization to do
                        fake input channel quantization.
                    SmoothQuantAlpha = float :
                        Default is 0.5. It only works if SmoothQuant is True. It controls the difficulty of weight
                        and activation quantization. A larger alpha value could be used on models with more significant
                        activation outliers to migrate more quantization difficulty to weights.
                    SmoothQuantFolding = True/False :
                        Default is True. It only works if SmoothQuant is True. If enabled, inserted Mul ops during
                        SmoothQuant will be folded into the previous op if the previous op is foldable.
                    UseQDQContribOps = True/False :
                        Default is False. If enabled, the inserted QuantizeLinear and DequantizeLinear ops will have the
                        `com.microsoft` domain, which forces use of ONNX Runtime's QuantizeLinear and DequantizeLinear
                        contrib op implementations. The contrib op implementations may support features not standardized
                        into the ONNX specification (e.g., 16-bit quantization types).
                    MinimumRealRange = float|None :
                        Default is None. If set to a floating-point value, the calculation of the quantization parameters
                        (i.e., scale and zero point) will enforce a minimum range between rmin and rmax. If (rmax-rmin)
                        is less than the specified minimum range, rmax will be set to rmin + MinimumRealRange. This is
                        necessary for EPs like QNN that require a minimum floating-point range when determining
                        quantization parameters.
                    TensorQuantOverrides = dictionary :
                        Default is {}. Set tensor quantization overrides. The key is a tensor name and the value is a
                        list of dictionaries. For per-tensor quantization, the list contains a single dictionary. For
                        per-channel quantization, the list contains a dictionary for each channel in the tensor.
                        Each dictionary contains optional overrides with the following keys and values.
                            'quant_type' = QuantType : The tensor's quantization data type.
                            'scale' =  Float         : The scale value to use. Must also specify `zero_point` if set.
                            'zero_point' = Int       : The zero-point value to use. Must also specify `scale` is set.
                            'symmetric' = Bool       : If the tensor should use symmetric quantization. Invalid if also
                                                       set `scale` or `zero_point`.
                            'reduce_range' = Bool    : If the quantization range should be reduced. Invalid if also
                                                       set `scale` or `zero_point`.
                            'rmax' = Float           : Override the maximum real tensor value in calibration data.
                                                       Invalid if also set `scale` or `zero_point`.
                            'rmin' = Float           : Override the minimum real tensor value in calibration data.
                                                       Invalid if also set `scale` or `zero_point`.
                    QDQKeepRemovableActivations = True/False:
                        Default is False. If true, "removable" activations (e.g., Clip or Relu) will not be removed, and
                        will be explicitly represented in the QDQ model. If false, these activations are automatically
                        removed if activations are asymmetrically quantized. Keeping these activations is necessary if
                        optimizations or EP transformations will later remove QuantizeLinear/DequantizeLinear
                        operators from the model.
                    QDQDisableWeightAdjustForInt32Bias = True/False:
                        Default is False. If true, QDQ quantizer will not adjust the weight's scale when the bias
                        has a scale (input_scale * weight_scale) that is too small.
            execution_provider : A enum indicates the Execution Provider such as: CPU, TRT, NNAPI, SNE, etc.
        Raises:
            ValueError: Raise ValueError if execution provider is unknown
        """

        super().__init__(
            activation_type=activation_type,
            weight_type=weight_type,
            op_types_to_quantize=op_types_to_quantize,
            nodes_to_quantize=nodes_to_quantize,
            nodes_to_exclude=nodes_to_exclude,
            per_channel=per_channel,
            reduce_range=reduce_range,
            use_external_data_format=use_external_data_format,
        )
        self.calibration_data_reader = calibration_data_reader
        self.calibrate_method = calibrate_method
        self.quant_format = quant_format
        self.calibration_providers = calibration_providers
        self.extra_options = extra_options or {}


def get_qdq_config(
    model_input: str | Path | onnx.ModelProto,
    calibration_data_reader: CalibrationDataReader,
    calibrate_method=CalibrationMethod.MinMax,
    calibrate_args: dict[str, Any] | None = None,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    activation_symmetric: bool = False,
    weight_symmetric: bool | None = None,
    per_channel: bool = False,
    reduce_range: bool = False,
    keep_removable_activations: bool = False,
    min_real_range: float | None = None,
    tensor_quant_overrides: dict[str, list[dict[str, Any]]] | None = None,
    calibration_providers: list[str] | None = None,
    op_types_to_quantize: list[str] | None = None,
    nodes_to_exclude: list[str] | Callable[[onnx.ModelProto, onnx.NodeProto], bool] | None = None,
    extra_options: dict | None = None,
) -> StaticQuantConfig:
    """
    Returns a configuration suitable that quantizes the entire model to integer precision.

    Params:
        model_input: Path to the input model file or ModelProto.
        calibration_data_reader: Calibration data reader.
        calibrate_methode: The calibration method. Defaults to MinMax.
        activation_type: The default activation quantization type. Defaults to QUInt8.
        weight_type: The default weight quantization type. Defaults to QInt8.
        activation_symmetric: True if activations should be quantized symmetrically (i.e, rmax == -rmin) by default.
            Defaults to false. For int8 and int16, this results in zero-point values of 0. For uint8 and uint16,
            the zero-point values are 127 and 32,767, respectively.
        weight_symmetric: True if weights should be quantized symmetrically (i.e., rmax == -rmin) by default.
            Defaults to None. If set to None, weight_symmetric is assumed true if a weight's quant type is a signed int.
        per_channel: Global option that determines if a fixed set of operator types should be quantized per-channel.
            Defaults to false. Alternatively, use the tensor-level `tensor_quant_overrides` to select individual operators
            and their quantization axes.
        reduce_range: quantize weights with 1 less bit of precision (e.g., 7 bits for QInt8). Defaults to false.
            May improve the accuracy for some models running on non-VNNI machine, especially for per-channel mode.
        keep_removable_activations: Defaults to false. If true, "removable" activations (e.g., Clip or Relu) will not
                        be removed, and will be explicitly represented in the QDQ model. If false, these activations
                        are automatically removed if activations are asymmetrically quantized. Keeping these activations
                        is necessary if optimizations or EP transformations will later remove
                        QuantizeLinear/DequantizeLinear operators from the model.
        min_real_range: Default is None. If set to a floating-point value, the calculation of the quantization parameters
            (i.e., scale and zero point) will enforce a minimum range between rmin and rmax. If (rmax - rmin)
            is less than the specified minimum range, rmax will be set to rmin + min_real_range.
        tensor_quant_overrides: tensor-level quantization overrides. Defaults to None.
            The key is a tensor name and the value is a list of dictionaries. For per-tensor quantization, the list
            contains a single dictionary. For per-channel quantization, the list contains either a dictionary for
            each channel in the tensor or a single dictionary that is assumed to apply to all channels. An 'axis'
            key must be present in the first dictionary for per-channel quantization.

            Each dictionary contains optional overrides with the following keys and values.
                'quant_type' = QuantType : The tensor's quantization data type.
                'axis' = Int             : The per-channel axis. Must be present for per-channel weights.
                'scale' =  Float         : The scale value to use. Must also specify `zero_point` if set.
                'zero_point' = Int       : The zero-point value to use. Must also specify `scale` is set.
                'symmetric' = Bool       : If the tensor should use symmetric quantization. Invalid if also
                                            set `scale` or `zero_point`.
                'reduce_range' = Bool    : If the quantization range should be reduced. Invalid if also
                                            set `scale` or `zero_point`. Only valid for initializers.
                'rmax' = Float           : Override the maximum real tensor value in calibration data.
                                            Invalid if also set `scale` or `zero_point`.
                'rmin' = Float           : Override the minimum real tensor value in calibration data.
                                            Invalid if also set `scale` or `zero_point`.
                'convert' = Dict         : A nested dictionary with the same keys for an activation
                                           tensor that should be converted to another quantization type.
                'convert["recv_nodes"] = Set : Set of node names that consume the converted activation,
                                               other nodes get the original type. If not specified,
                                               assume all consumer nodes get the converted type.
        calibration_providers: Execution providers to run the session during calibration. Default is None which uses
            [ "CPUExecutionProvider" ].
        op_types_to_quantize: List of operator types to quantize. If None, all operators other than Cast, DequantizeLinear,
            and QuantizeLinear are quantized.
        nodes_to_exclude: List of nodes names to exclude from quantization. Alternatively, can provide a function that
            accepts an onnx.ModelProto and onnx.NodeProto as arguments and returns true if the give onnx.NodeProto
            should be excluded from quantization.
        extra_options: Additional options specified as string key/value pairs. Refer to the documentation for
            `quantize_static` for valid keys and values.

    Returns:
        A StaticQuantConfig object
    """
    q16_types = {QuantType.QInt16, QuantType.QUInt16}
    q4_types = {QuantType.QInt4, QuantType.QUInt4}
    op_types_to_exclude = {"Cast", "DequantizeLinear", "QuantizeLinear"}

    model = (
        model_input
        if isinstance(model_input, onnx.ModelProto)
        else onnx.load_model(model_input, load_external_data=False)
    )

    op_types = set()
    model_has_external_data = False
    overrides_helper = TensorQuantOverridesHelper(
        copy.deepcopy(tensor_quant_overrides) if tensor_quant_overrides else {}
    )

    # check if the model has external data.
    for initializer in model.graph.initializer:
        if onnx.external_data_helper.uses_external_data(initializer):
            model_has_external_data = True

    op_types_to_quantize_set = set(op_types_to_quantize) if op_types_to_quantize else None
    nodes_to_exclude_set = set(nodes_to_exclude) if isinstance(nodes_to_exclude, list) else set()

    # Iterate through nodes to get all operator types in the model and
    # call user's function to filter out nodes from quantization.
    for node in model.graph.node:
        if op_types_to_quantize_set and node.op_type not in op_types_to_quantize_set:
            continue
        if node.name in nodes_to_exclude_set:
            continue
        if callable(nodes_to_exclude) and nodes_to_exclude(model, node):
            nodes_to_exclude_set.add(node.name)
        else:
            op_types.add(node.op_type)

    final_extra_options = {
        "MinimumRealRange": min_real_range,
        "QDQKeepRemovableActivations": keep_removable_activations,
        "ActivationSymmetric": activation_symmetric,
        "WeightSymmetric": weight_symmetric,
        "ForceQuantizeNoInputCheck": True,
        "TensorQuantOverrides": overrides_helper.get_dict(),
    }

    # Pass along known calibration options
    if calibrate_args:
        calib_extra_options_keys = [
            ("symmetric", "CalibTensorRangeSymmetric"),
            ("moving_average", "CalibMovingAverage"),
            ("averaging_constant", "CalibMovingAverageConstant"),
            ("max_intermediate_outputs", "CalibMaxIntermediateOutputs"),
            ("percentile", "CalibPercentile"),
        ]
        calib_extra_options = {
            key: calibrate_args.get(name) for (name, key) in calib_extra_options_keys if name in calibrate_args
        }
        final_extra_options.update(calib_extra_options)

    # ONNX opset < 21 does not support 16-bit quantization, so must use 'com.microsoft' domain
    # on Q/DQ operators if using 16-bit or 4-bit quantization.
    onnx_opset = next(x for x in model.opset_import if x.domain == "" or x.domain == "ai.onnx")
    if onnx_opset.version < 21:
        opset21_types = q16_types.union(q4_types)
        overrides_have_opset21_types = any(t in opset21_types for t in overrides_helper.get_quant_types())
        if activation_type in opset21_types or weight_type in opset21_types or overrides_have_opset21_types:
            final_extra_options["UseQDQContribOps"] = True

    # Allow user's extra_options to override our final_extra_options.
    if extra_options:
        final_extra_options.update(extra_options)

    return StaticQuantConfig(
        calibration_data_reader,
        calibrate_method=calibrate_method,
        quant_format=QuantFormat.QDQ,
        activation_type=activation_type,
        weight_type=weight_type,
        op_types_to_quantize=(
            op_types_to_quantize if op_types_to_quantize else list(op_types.difference(op_types_to_exclude))
        ),
        nodes_to_exclude=list(nodes_to_exclude_set),
        per_channel=per_channel,
        reduce_range=reduce_range,
        use_external_data_format=(model_has_external_data or model.ByteSize() >= MODEL_SIZE_THRESHOLD),
        calibration_providers=calibration_providers,
        extra_options=final_extra_options,
    )


class DynamicQuantConfig(QuantConfig):
    def __init__(
        self,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=None,
        nodes_to_quantize=None,
        nodes_to_exclude=None,
        per_channel=False,
        reduce_range=False,
        use_external_data_format=False,
        extra_options=None,
    ):
        """
        This is a class for dynamic Quant Configuration

        Args:
            extra_options: key value pair dictionary for various options in different case. Current used:
                extra.Sigmoid.nnapi = True/False  (Default is False)
                ActivationSymmetric = True/False: symmetrize calibration data for activations (default is False).
                WeightSymmetric = True/False: symmetrize calibration data for weights (default is True).
                EnableSubgraph = True/False :
                    Default is False. If enabled, subgraph will be quantized. Dynamic mode currently is supported. Will
                    support more in the future.
                ForceQuantizeNoInputCheck = True/False :
                    By default, some latent operators like maxpool, transpose, do not quantize if their input is not
                    quantized already. Setting to True to force such operator always quantize input and so generate
                    quantized output. Also the True behavior could be disabled per node using the nodes_to_exclude.
                MatMulConstBOnly = True/False:
                    Default is True for dynamic mode. If enabled, only MatMul with const B will be quantized.
            execution_provider : A enum indicates the Execution Provider such as: CPU, TRT, NNAPI, SNE, etc.

        Raises:
            ValueError: Raise ValueError if execution provider is unknown
        """
        super().__init__(
            op_types_to_quantize=op_types_to_quantize,
            per_channel=per_channel,
            reduce_range=reduce_range,
            weight_type=weight_type,
            nodes_to_quantize=nodes_to_quantize,
            nodes_to_exclude=nodes_to_exclude,
            use_external_data_format=use_external_data_format,
        )
        self.extra_options = extra_options or {}


def check_static_quant_arguments(quant_format: QuantFormat, activation_type: QuantType, weight_type: QuantType):
    if activation_type == QuantType.QInt8 and weight_type == QuantType.QUInt8:
        raise ValueError(
            "ONNXRuntime quantization doesn't support data format:"
            "activation_type=QuantType.QInt8, weight_type=QuantType.QUInt8"
        )
    if activation_type != QuantType.QFLOAT8E4M3FN and weight_type == QuantType.QFLOAT8E4M3FN:
        raise ValueError(
            f"ONNXRuntime quantization doesn't support data format: activation_type={activation_type} "
            "!=QuantType.QFLOAT8E4M3FN, weight_type=QuantType.QFLOAT8E4M3FN."
        )

    if activation_type == QuantType.QFLOAT8E4M3FN and weight_type != QuantType.QFLOAT8E4M3FN:
        raise ValueError(
            "ONNXRuntime quantization doesn't support data format: activation_type=QuantType.QFLOAT8E4M3FN, "
            f"weight_type={weight_type}!=QuantType.QFLOAT8E4M3FN"
        )

    q16_types = [QuantType.QInt16, QuantType.QUInt16]

    if (activation_type in q16_types or weight_type in q16_types) and quant_format != QuantFormat.QDQ:
        raise ValueError("Only QuantFormat.QDQ supports 16-bit quantization types.")

    if activation_type == QuantType.QInt8 and weight_type == QuantType.QInt8 and quant_format != QuantFormat.QDQ:
        logging.warning(
            "Please use QuantFormat.QDQ for activation type QInt8 and weight type QInt8. "
            "Or it will lead to bad performance on x64."
        )


def quantize_static(
    model_input: str | Path | onnx.ModelProto,
    model_output: str | Path,
    calibration_data_reader: CalibrationDataReader,
    quant_format=QuantFormat.QDQ,
    op_types_to_quantize=None,
    per_channel=False,
    reduce_range=False,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    nodes_to_quantize=None,
    nodes_to_exclude=None,
    use_external_data_format=False,
    calibrate_method=CalibrationMethod.MinMax,
    calibration_providers=None,
    extra_options=None,
):
    """
    Given an onnx model and calibration data reader, create a quantized onnx model and save it into a file
    It is recommended to use QuantFormat.QDQ format from 1.11 with activation_type = QuantType.QInt8 and weight_type
    = QuantType.QInt8. If model is targeted to GPU/TRT, symmetric activation and weight are required. If model is
    targeted to CPU, asymmetric activation and symmetric weight are recommended for balance of performance and
    accuracy.

    Args:

        model_input: file path of model or ModelProto to quantize
        model_output: file path of quantized model
        calibration_data_reader: a calibration data reader. It
            enumerates calibration data and generates inputs for the
            original model.
        quant_format: QuantFormat{QOperator, QDQ}.
            QOperator format quantizes the model with quantized operators directly.
            QDQ format quantize the model by inserting QuantizeLinear/DeQuantizeLinear on the tensor.
        activation_type:
            quantization data type of activation. Please refer to
            https://onnxruntime.ai/docs/performance/quantization.html for more details on data type selection
        calibrate_method:
            Current calibration methods supported are MinMax and Entropy.
                Please use CalibrationMethod.MinMax or CalibrationMethod.Entropy as options.
        op_types_to_quantize:
                specify the types of operators to quantize, like ['Conv'] to quantize Conv only.
                It quantizes all supported operators by default.
        per_channel: quantize weights per channel
        reduce_range:
            quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine,
            especially for per-channel mode
        weight_type:
            quantization data type of weight. Please refer to
            https://onnxruntime.ai/docs/performance/quantization.html for more details on data type selection
        nodes_to_quantize:
            List of nodes names to quantize. When this list is not None only the nodes in this list
            are quantized.
            example:
            [
                'Conv__224',
                'Conv__252'
            ]
        nodes_to_exclude:
            List of nodes names to exclude. The nodes in this list will be excluded from quantization
            when it is not None.
        use_external_data_format: option used for large size (>2GB) model. Set to False by default.
        calibration_providers: Execution providers to run the session during calibration. Default is None which uses
            [ "CPUExecutionProvider" ]
        extra_options:
            key value pair dictionary for various options in different case. Current used:
                extra.Sigmoid.nnapi = True/False  (Default is False)
                ActivationSymmetric = True/False: symmetrize calibration data for activations (default is False).
                WeightSymmetric = True/False: symmetrize calibration data for weights (default is True).
                EnableSubgraph = True/False : Default is False. If enabled, subgraph will be quantized.
                                              Dyanmic mode currently is supported. Will support more in the future.
                ForceQuantizeNoInputCheck = True/False :
                    By default, some latent operators like maxpool, transpose, do not quantize if their input is not
                    quantized already. Setting to True to force such operator always quantize input and so generate
                    quantized output. Also, the True behavior could be disabled per node using the nodes_to_exclude.
                MatMulConstBOnly = True/False:
                    Default is False for static mode. If enabled, only MatMul with const B will be quantized.
                AddQDQPairToWeight = True/False :
                    Default is False which quantizes floating-point weight and feeds it to solely inserted
                    DeQuantizeLinear node. If True, it remains floating-point weight and inserts both
                    QuantizeLinear/DeQuantizeLinear nodes to weight.
                OpTypesToExcludeOutputQuantization = list of op type :
                    Default is []. If any op type is specified, it won't quantize the output of ops with this
                    specific op types.
                DedicatedQDQPair = True/False :
                    Default is False. When inserting QDQ pair, multiple nodes can share a single QDQ pair as their
                    inputs. If True, it will create identical and dedicated QDQ pair for each node.
                QDQOpTypePerChannelSupportToAxis = dictionary :
                    Default is {}. Set channel axis for specific op type, for example: {'MatMul': 1}, and it's
                    effective only when per channel quantization is supported and per_channel is True. If specific
                    op type supports per channel quantization but not explicitly specified with channel axis,
                    default channel axis will be used.
                CalibTensorRangeSymmetric = True/False :
                    Default is False. If enabled, the final range of tensor during calibration will be explicitly
                    set to symmetric to central point "0".
                CalibStridedMinMax = Optional[int] :
                    Default is None. If set to an integer, during calculation of the min-max, only stride amount of
                    data will be used and then all results will be merged in the end.
                CalibMovingAverage = True/False :
                    Default is False. If enabled, the moving average of the minimum and maximum values will be
                    computed when the calibration method selected is MinMax.
                CalibMovingAverageConstant = float :
                    Default is 0.01. Constant smoothing factor to use when computing the moving average of the
                    minimum and maximum values. Effective only when the calibration method selected is MinMax and
                    when CalibMovingAverage is set to True.
                CalibMaxIntermediateOutputs = Optional[int] :
                    Default is None. If set to an integer, during calculation of the min-max range of the tensors
                    it will load at max value number of outputs before computing and merging the range. This will
                    produce the same result as all computing with None, but is more memory efficient.
                SmoothQuant = True/False :
                    Default is False. If enabled, SmoothQuant algorithm will be applied before quantization to do
                    fake input channel quantization.
                SmoothQuantAlpha = float :
                    Default is 0.5. It only works if SmoothQuant is True. It controls the difficulty of weight
                    and activation quantization. A larger alpha value could be used on models with more significant
                    activation outliers to migrate more quantization difficulty to weights.
                SmoothQuantFolding = True/False :
                    Default is True. It only works if SmoothQuant is True. If enabled, inserted Mul ops during
                    SmoothQuant will be folded into the previous op if the previous op is foldable.
                UseQDQContribOps = True/False :
                    Default is False. If enabled, the inserted QuantizeLinear and DequantizeLinear ops will have the
                    `com.microsoft` domain, which forces use of ONNX Runtime's QuantizeLinear and DequantizeLinear
                    contrib op implementations. The contrib op implementations may support features not standardized
                    into the ONNX specification (e.g., 16-bit quantization types).
                MinimumRealRange = float|None :
                    Default is None. If set to a floating-point value, the calculation of the quantization parameters
                    (i.e., scale and zero point) will enforce a minimum range between rmin and rmax. If (rmax - rmin)
                    is less than the specified minimum range, rmax will be set to rmin + MinimumRealRange. This is
                    necessary for EPs like QNN that require a minimum floating-point range when determining
                    quantization parameters.
                TensorQuantOverrides = dictionary :
                    Default is {}. Set tensor quantization overrides. The key is a tensor name and the value is a
                    list of dictionaries. For per-tensor quantization, the list contains a single dictionary. For
                    per-channel quantization, the list contains a dictionary for each channel in the tensor.
                    Each dictionary contains optional overrides with the following keys and values.
                        'quant_type' = QuantType : The tensor's quantization data type.
                        'scale' =  Float         : The scale value to use. Must also specify `zero_point` if set.
                        'zero_point' = Int       : The zero-point value to use. Must also specify `scale` is set.
                        'symmetric' = Bool       : If the tensor should use symmetric quantization. Invalid if also
                                                   set `scale` or `zero_point`.
                        'reduce_range' = Bool    : If the quantization range should be reduced. Invalid if also
                                                   set `scale` or `zero_point`.
                        'rmax' = Float           : Override the maximum real tensor value in calibration data.
                                                   Invalid if also set `scale` or `zero_point`.
                        'rmin' = Float           : Override the minimum real tensor value in calibration data.
                                                   Invalid if also set `scale` or `zero_point`.
                QDQKeepRemovableActivations = True/False:
                    Default is False. If true, "removable" activations (e.g., Clip or Relu) will not be removed, and
                    will be explicitly represented in the QDQ model. If false, these activations are automatically
                    removed if activations are asymmetrically quantized. Keeping these activations is necessary if
                    optimizations or EP transformations will later remove QuantizeLinear/DequantizeLinear
                    operators from the model.
                QDQDisableWeightAdjustForInt32Bias = True/False:
                    Default is False. If true, QDQ quantizer will not adjust the weight's scale when the bias
                    has a scale (input_scale * weight_scale) that is too small.
    """
    if activation_type == QuantType.QFLOAT8E4M3FN or weight_type == QuantType.QFLOAT8E4M3FN:
        if calibrate_method != CalibrationMethod.Distribution:
            raise ValueError("Only Distribution calibration method is supported for float quantization.")

    extra_options = extra_options or {}
    nodes_to_exclude = nodes_to_exclude or []
    nodes_to_quantize = nodes_to_quantize or []
    op_types_to_quantize = op_types_to_quantize or []
    mode = QuantizationMode.QLinearOps

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        q_linear_ops = list(QLinearOpsRegistry.keys())
        qdq_ops = list(QDQRegistry.keys())
        op_types_to_quantize = list(set(q_linear_ops + qdq_ops))

    model = (
        save_and_reload_model_with_shape_infer(model_input)
        if isinstance(model_input, onnx.ModelProto)
        else load_model_with_shape_infer(Path(model_input))
    )

    pre_processed: bool = model_has_pre_process_metadata(model)
    if not pre_processed:
        logging.warning(
            "Please consider to run pre-processing before quantization. Refer to example: "
            "https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification"
            "/cpu/ReadMe.md "
        )

    calib_extra_options_keys = [
        ("CalibTensorRangeSymmetric", "symmetric"),
        ("CalibMovingAverage", "moving_average"),
        ("CalibMovingAverageConstant", "averaging_constant"),
        ("CalibMaxIntermediateOutputs", "max_intermediate_outputs"),
        ("CalibPercentile", "percentile"),
    ]
    calib_extra_options = {
        key: extra_options.get(name) for (name, key) in calib_extra_options_keys if name in extra_options
    }

    if extra_options.get("SmoothQuant", False):
        import importlib

        try:
            importlib.import_module("neural_compressor.adaptor.ox_utils.smooth_quant")
        except Exception as e:
            logging.error(f"{e}.")
            raise RuntimeError("neural-compressor is not correctly installed. Please check your environment.") from e

        import copy

        from neural_compressor.adaptor.ox_utils.smooth_quant import ORTSmoothQuant

        def inc_dataloader():
            data_reader = copy.deepcopy(calibration_data_reader)
            for data in data_reader:
                yield data, None

        orig_nodes = [i.name for i in model.graph.node]
        dataloader = inc_dataloader()
        sq = ORTSmoothQuant(model_input, dataloader, reduce_range)
        del dataloader
        model = sq.transform(extra_options.get("SmoothQuantAlpha", 0.5), extra_options.get("SmoothQuantFolding", True))
        sq_path = tempfile.TemporaryDirectory(prefix="ort.quant.")
        model_input = Path(sq_path.name).joinpath("sq_model.onnx").as_posix()
        model.save(model_input)
        nodes_to_exclude.extend([i.name for i in model.model.graph.node if i.name not in orig_nodes])
        model = load_model_with_shape_infer(Path(model_input))  # use smooth quant model for calibration

    with tempfile.TemporaryDirectory(prefix="ort.quant.") as quant_tmp_dir:
        if isinstance(model_input, onnx.ModelProto):
            output_path = str(Path(quant_tmp_dir) / "model_input.onnx")
            onnx.save_model(
                model_input,
                output_path,
                save_as_external_data=True,
            )
            model_input = output_path

        calibrator = create_calibrator(
            Path(model_input),
            op_types_to_quantize,
            augmented_model_path=Path(quant_tmp_dir).joinpath("augmented_model.onnx").as_posix(),
            calibrate_method=calibrate_method,
            use_external_data_format=use_external_data_format,
            providers=calibration_providers,
            extra_options=calib_extra_options,
        )

        stride = extra_options.get("CalibStridedMinMax", None)
        if stride:
            total_data_size = len(calibration_data_reader)
            if total_data_size % stride != 0:
                raise ValueError(f"Total data size ({total_data_size}) is not divisible by stride size ({stride}).")

            for start in range(0, total_data_size, stride):
                end_index = start + stride
                calibration_data_reader.set_range(start_index=start, end_index=end_index)
                calibrator.collect_data(calibration_data_reader)
        else:
            calibrator.collect_data(calibration_data_reader)
        tensors_range = calibrator.compute_data()
        if not isinstance(tensors_range, TensorsData):
            raise TypeError(
                f"Unexpected type {type(tensors_range)} for tensors_range and calibrator={type(calibrator)}."
            )
        del calibrator

    check_static_quant_arguments(quant_format, activation_type, weight_type)

    if quant_format is QuantFormat.QOperator:
        quantizer = ONNXQuantizer(
            model,
            per_channel,
            reduce_range,
            mode,
            True,  # static
            weight_type,
            activation_type,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )
    else:
        quantizer = QDQQuantizer(
            model,
            per_channel,
            reduce_range,
            weight_type,
            activation_type,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )

    quantizer.quantize_model()
    quantizer.model.save_model_to_file(model_output, use_external_data_format)
    if not pre_processed:
        logging.warning(
            "Please consider pre-processing before quantization. See "
            "https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification"
            "/cpu/ReadMe.md "
        )

    if extra_options.get("SmoothQuant", False):
        sq_path.cleanup()


def quantize_dynamic(
    model_input: str | Path | onnx.ModelProto,
    model_output: str | Path,
    op_types_to_quantize=None,
    per_channel=False,
    reduce_range=False,
    weight_type=QuantType.QInt8,
    nodes_to_quantize=None,
    nodes_to_exclude=None,
    use_external_data_format=False,
    extra_options=None,
):
    """Given an onnx model, create a quantized onnx model and save it into a file

    Args:
        model_input: file path of model or ModelProto to quantize
        model_output: file path of quantized model
        op_types_to_quantize:
            specify the types of operators to quantize, like ['Conv'] to quantize Conv only.
            It quantizes all supported operators by default.
        per_channel: quantize weights per channel
        reduce_range:
            quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine,
            especially for per-channel mode
        weight_type:
            quantization data type of weight. Please refer to
            https://onnxruntime.ai/docs/performance/quantization.html for more details on data type selection
        nodes_to_quantize:
            List of nodes names to quantize. When this list is not None only the nodes in this list
            are quantized.
            example:
            [
                'Conv__224',
                'Conv__252'
            ]
        nodes_to_exclude:
            List of nodes names to exclude. The nodes in this list will be excluded from quantization
            when it is not None.
        use_external_data_format: option used for large size (>2GB) model. Set to False by default.
        extra_options:
            key value pair dictionary for various options in different case. Current used:
                extra.Sigmoid.nnapi = True/False  (Default is False)
                ActivationSymmetric = True/False: symmetrize calibration data for activations (default is False).
                WeightSymmetric = True/False: symmetrize calibration data for weights (default is True).
                EnableSubgraph = True/False :
                    Default is False. If enabled, subgraph will be quantized. Dynamic mode currently is supported. Will
                    support more in the future.
                ForceQuantizeNoInputCheck = True/False :
                    By default, some latent operators like maxpool, transpose, do not quantize if their input is not
                    quantized already. Setting to True to force such operator always quantize input and so generate
                    quantized output. Also the True behavior could be disabled per node using the nodes_to_exclude.
                MatMulConstBOnly = True/False:
                    Default is True for dynamic mode. If enabled, only MatMul with const B will be quantized.
    """
    extra_options = extra_options or {}
    nodes_to_exclude = nodes_to_exclude or []
    nodes_to_quantize = nodes_to_quantize or []
    op_types_to_quantize = op_types_to_quantize or []

    mode = QuantizationMode.IntegerOps

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        op_types_to_quantize = list(IntegerOpsRegistry.keys())

    model = (
        save_and_reload_model_with_shape_infer(model_input)
        if isinstance(model_input, onnx.ModelProto)
        else load_model_with_shape_infer(Path(model_input))
    )

    pre_processed: bool = model_has_pre_process_metadata(model)
    if not pre_processed:
        logging.warning(
            "Please consider to run pre-processing before quantization. Refer to example: "
            "https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification"
            "/cpu/ReadMe.md "
        )

    if "MatMulConstBOnly" not in extra_options:
        extra_options["MatMulConstBOnly"] = True

    quantizer = ONNXQuantizer(
        model,
        per_channel,
        reduce_range,
        mode,
        False,  # static
        weight_type,
        QuantType.QUInt8,  # dynamic activation only supports uint8
        None,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        extra_options,
    )

    quantizer.quantize_model()
    quantizer.model.save_model_to_file(model_output, use_external_data_format)


def quantize(
    model_input: str | Path | onnx.ModelProto,
    model_output: str | Path,
    quant_config: QuantConfig,
):
    """Quantize a model with QuantConfig.

    Args:
        model_input (str | Path | ModelProto): Path to the model or ModelProto to quantize.
        model_output (str | Path): Path to save the quantized model.
        quant_config (QuantConfig | WeightOnlyQuantConfig): Quantization Configuration.
    """
    if isinstance(quant_config, StaticQuantConfig):
        quantize_static(
            model_input,
            model_output,
            quant_config.calibration_data_reader,
            calibrate_method=quant_config.calibrate_method,
            quant_format=quant_config.quant_format,
            activation_type=quant_config.activation_type,
            weight_type=quant_config.weight_type,
            op_types_to_quantize=quant_config.op_types_to_quantize,
            nodes_to_quantize=quant_config.nodes_to_quantize,
            nodes_to_exclude=quant_config.nodes_to_exclude,
            per_channel=quant_config.per_channel,
            reduce_range=quant_config.reduce_range,
            use_external_data_format=quant_config.use_external_data_format,
            calibration_providers=quant_config.calibration_providers,
            extra_options=quant_config.extra_options,
        )

    elif isinstance(quant_config, DynamicQuantConfig):
        quantize_dynamic(
            model_input,
            model_output,
            weight_type=quant_config.weight_type,
            op_types_to_quantize=quant_config.op_types_to_quantize,
            nodes_to_quantize=quant_config.nodes_to_quantize,
            nodes_to_exclude=quant_config.nodes_to_exclude,
            per_channel=quant_config.per_channel,
            reduce_range=quant_config.reduce_range,
            use_external_data_format=quant_config.use_external_data_format,
            extra_options=quant_config.extra_options,
        )
    else:
        # training package doesn't has quantize_matmul_4bits, avoid global import
        from .matmul_nbits_quantizer import MatMulNBitsQuantizer, WeightOnlyQuantConfig

        if isinstance(quant_config, WeightOnlyQuantConfig):
            model = model_input if isinstance(model_input, onnx.ModelProto) else onnx.load(model_input)
            quant = MatMulNBitsQuantizer(model, algo_config=quant_config)
            quant.process()
            quant.model.save_model_to_file(model_output, True)
        else:
            raise TypeError(
                "Invalid quantization config type, it must be either StaticQuantConfig, "
                "DynamicQuantConfig, or WeightOnlyQuantConfig."
            )