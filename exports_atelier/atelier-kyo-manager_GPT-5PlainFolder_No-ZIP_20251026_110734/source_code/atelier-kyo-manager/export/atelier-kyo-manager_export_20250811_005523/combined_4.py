
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\_index_tricks_impl.py ===
import functools
import math
import sys
import warnings

import numpy as np
import numpy._core.numeric as _nx
import numpy.matrixlib as matrixlib
from numpy._core import linspace, overrides
from numpy._core.multiarray import ravel_multi_index, unravel_index
from numpy._core.numeric import ScalarType, array
from numpy._core.numerictypes import issubdtype
from numpy._utils import set_module
from numpy.lib._function_base_impl import diff
from numpy.lib.stride_tricks import as_strided

array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


__all__ = [
    'ravel_multi_index', 'unravel_index', 'mgrid', 'ogrid', 'r_', 'c_',
    's_', 'index_exp', 'ix_', 'ndenumerate', 'ndindex', 'fill_diagonal',
    'diag_indices', 'diag_indices_from'
]


def _ix__dispatcher(*args):
    return args


@array_function_dispatch(_ix__dispatcher)
def ix_(*args):
    """
    Construct an open mesh from multiple sequences.

    This function takes N 1-D sequences and returns N outputs with N
    dimensions each, such that the shape is 1 in all but one dimension
    and the dimension with the non-unit shape value cycles through all
    N dimensions.

    Using `ix_` one can quickly construct index arrays that will index
    the cross product. ``a[np.ix_([1,3],[2,5])]`` returns the array
    ``[[a[1,2] a[1,5]], [a[3,2] a[3,5]]]``.

    Parameters
    ----------
    args : 1-D sequences
        Each sequence should be of integer or boolean type.
        Boolean sequences will be interpreted as boolean masks for the
        corresponding dimension (equivalent to passing in
        ``np.nonzero(boolean_sequence)``).

    Returns
    -------
    out : tuple of ndarrays
        N arrays with N dimensions each, with N the number of input
        sequences. Together these arrays form an open mesh.

    See Also
    --------
    ogrid, mgrid, meshgrid

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(10).reshape(2, 5)
    >>> a
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> ixgrid = np.ix_([0, 1], [2, 4])
    >>> ixgrid
    (array([[0],
           [1]]), array([[2, 4]]))
    >>> ixgrid[0].shape, ixgrid[1].shape
    ((2, 1), (1, 2))
    >>> a[ixgrid]
    array([[2, 4],
           [7, 9]])

    >>> ixgrid = np.ix_([True, True], [2, 4])
    >>> a[ixgrid]
    array([[2, 4],
           [7, 9]])
    >>> ixgrid = np.ix_([True, True], [False, False, True, False, True])
    >>> a[ixgrid]
    array([[2, 4],
           [7, 9]])

    """
    out = []
    nd = len(args)
    for k, new in enumerate(args):
        if not isinstance(new, _nx.ndarray):
            new = np.asarray(new)
            if new.size == 0:
                # Explicitly type empty arrays to avoid float default
                new = new.astype(_nx.intp)
        if new.ndim != 1:
            raise ValueError("Cross index must be 1 dimensional")
        if issubdtype(new.dtype, _nx.bool):
            new, = new.nonzero()
        new = new.reshape((1,) * k + (new.size,) + (1,) * (nd - k - 1))
        out.append(new)
    return tuple(out)


class nd_grid:
    """
    Construct a multi-dimensional "meshgrid".

    ``grid = nd_grid()`` creates an instance which will return a mesh-grid
    when indexed.  The dimension and number of the output arrays are equal
    to the number of indexing dimensions.  If the step length is not a
    complex number, then the stop is not inclusive.

    However, if the step length is a **complex number** (e.g. 5j), then the
    integer part of its magnitude is interpreted as specifying the
    number of points to create between the start and stop values, where
    the stop value **is inclusive**.

    If instantiated with an argument of ``sparse=True``, the mesh-grid is
    open (or not fleshed out) so that only one-dimension of each returned
    argument is greater than 1.

    Parameters
    ----------
    sparse : bool, optional
        Whether the grid is sparse or not. Default is False.

    Notes
    -----
    Two instances of `nd_grid` are made available in the NumPy namespace,
    `mgrid` and `ogrid`, approximately defined as::

        mgrid = nd_grid(sparse=False)
        ogrid = nd_grid(sparse=True)

    Users should use these pre-defined instances instead of using `nd_grid`
    directly.
    """
    __slots__ = ('sparse',)

    def __init__(self, sparse=False):
        self.sparse = sparse

    def __getitem__(self, key):
        try:
            size = []
            # Mimic the behavior of `np.arange` and use a data type
            # which is at least as large as `np.int_`
            num_list = [0]
            for k in range(len(key)):
                step = key[k].step
                start = key[k].start
                stop = key[k].stop
                if start is None:
                    start = 0
                if step is None:
                    step = 1
                if isinstance(step, (_nx.complexfloating, complex)):
                    step = abs(step)
                    size.append(int(step))
                else:
                    size.append(
                        math.ceil((stop - start) / step))
                num_list += [start, stop, step]
            typ = _nx.result_type(*num_list)
            if self.sparse:
                nn = [_nx.arange(_x, dtype=_t)
                      for _x, _t in zip(size, (typ,) * len(size))]
            else:
                nn = _nx.indices(size, typ)
            for k, kk in enumerate(key):
                step = kk.step
                start = kk.start
                if start is None:
                    start = 0
                if step is None:
                    step = 1
                if isinstance(step, (_nx.complexfloating, complex)):
                    step = int(abs(step))
                    if step != 1:
                        step = (kk.stop - start) / float(step - 1)
                nn[k] = (nn[k] * step + start)
            if self.sparse:
                slobj = [_nx.newaxis] * len(size)
                for k in range(len(size)):
                    slobj[k] = slice(None, None)
                    nn[k] = nn[k][tuple(slobj)]
                    slobj[k] = _nx.newaxis
                return tuple(nn)  # ogrid -> tuple of arrays
            return nn  # mgrid -> ndarray
        except (IndexError, TypeError):
            step = key.step
            stop = key.stop
            start = key.start
            if start is None:
                start = 0
            if isinstance(step, (_nx.complexfloating, complex)):
                # Prevent the (potential) creation of integer arrays
                step_float = abs(step)
                step = length = int(step_float)
                if step != 1:
                    step = (key.stop - start) / float(step - 1)
                typ = _nx.result_type(start, stop, step_float)
                return _nx.arange(0, length, 1, dtype=typ) * step + start
            else:
                return _nx.arange(start, stop, step)


class MGridClass(nd_grid):
    """
    An instance which returns a dense multi-dimensional "meshgrid".

    An instance which returns a dense (or fleshed out) mesh-grid
    when indexed, so that each returned argument has the same shape.
    The dimensions and number of the output arrays are equal to the
    number of indexing dimensions.  If the step length is not a complex
    number, then the stop is not inclusive.

    However, if the step length is a **complex number** (e.g. 5j), then
    the integer part of its magnitude is interpreted as specifying the
    number of points to create between the start and stop values, where
    the stop value **is inclusive**.

    Returns
    -------
    mesh-grid : ndarray
        A single array, containing a set of `ndarray`\\ s all of the same
        dimensions. stacked along the first axis.

    See Also
    --------
    ogrid : like `mgrid` but returns open (not fleshed out) mesh grids
    meshgrid: return coordinate matrices from coordinate vectors
    r_ : array concatenator
    :ref:`how-to-partition`

    Examples
    --------
    >>> import numpy as np
    >>> np.mgrid[0:5, 0:5]
    array([[[0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4]],
           [[0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4]]])
    >>> np.mgrid[-1:1:5j]
    array([-1. , -0.5,  0. ,  0.5,  1. ])

    >>> np.mgrid[0:4].shape
    (4,)
    >>> np.mgrid[0:4, 0:5].shape
    (2, 4, 5)
    >>> np.mgrid[0:4, 0:5, 0:6].shape
    (3, 4, 5, 6)

    """
    __slots__ = ()

    def __init__(self):
        super().__init__(sparse=False)


mgrid = MGridClass()


class OGridClass(nd_grid):
    """
    An instance which returns an open multi-dimensional "meshgrid".

    An instance which returns an open (i.e. not fleshed out) mesh-grid
    when indexed, so that only one dimension of each returned array is
    greater than 1.  The dimension and number of the output arrays are
    equal to the number of indexing dimensions.  If the step length is
    not a complex number, then the stop is not inclusive.

    However, if the step length is a **complex number** (e.g. 5j), then
    the integer part of its magnitude is interpreted as specifying the
    number of points to create between the start and stop values, where
    the stop value **is inclusive**.

    Returns
    -------
    mesh-grid : ndarray or tuple of ndarrays
        If the input is a single slice, returns an array.
        If the input is multiple slices, returns a tuple of arrays, with
        only one dimension not equal to 1.

    See Also
    --------
    mgrid : like `ogrid` but returns dense (or fleshed out) mesh grids
    meshgrid: return coordinate matrices from coordinate vectors
    r_ : array concatenator
    :ref:`how-to-partition`

    Examples
    --------
    >>> from numpy import ogrid
    >>> ogrid[-1:1:5j]
    array([-1. , -0.5,  0. ,  0.5,  1. ])
    >>> ogrid[0:5, 0:5]
    (array([[0],
            [1],
            [2],
            [3],
            [4]]),
     array([[0, 1, 2, 3, 4]]))

    """
    __slots__ = ()

    def __init__(self):
        super().__init__(sparse=True)


ogrid = OGridClass()


class AxisConcatenator:
    """
    Translates slice objects to concatenation along an axis.

    For detailed documentation on usage, see `r_`.
    """
    __slots__ = ('axis', 'matrix', 'ndmin', 'trans1d')

    # allow ma.mr_ to override this
    concatenate = staticmethod(_nx.concatenate)
    makemat = staticmethod(matrixlib.matrix)

    def __init__(self, axis=0, matrix=False, ndmin=1, trans1d=-1):
        self.axis = axis
        self.matrix = matrix
        self.trans1d = trans1d
        self.ndmin = ndmin

    def __getitem__(self, key):
        # handle matrix builder syntax
        if isinstance(key, str):
            frame = sys._getframe().f_back
            mymat = matrixlib.bmat(key, frame.f_globals, frame.f_locals)
            return mymat

        if not isinstance(key, tuple):
            key = (key,)

        # copy attributes, since they can be overridden in the first argument
        trans1d = self.trans1d
        ndmin = self.ndmin
        matrix = self.matrix
        axis = self.axis

        objs = []
        # dtypes or scalars for weak scalar handling in result_type
        result_type_objs = []

        for k, item in enumerate(key):
            scalar = False
            if isinstance(item, slice):
                step = item.step
                start = item.start
                stop = item.stop
                if start is None:
                    start = 0
                if step is None:
                    step = 1
                if isinstance(step, (_nx.complexfloating, complex)):
                    size = int(abs(step))
                    newobj = linspace(start, stop, num=size)
                else:
                    newobj = _nx.arange(start, stop, step)
                if ndmin > 1:
                    newobj = array(newobj, copy=None, ndmin=ndmin)
                    if trans1d != -1:
                        newobj = newobj.swapaxes(-1, trans1d)
            elif isinstance(item, str):
                if k != 0:
                    raise ValueError("special directives must be the "
                                     "first entry.")
                if item in ('r', 'c'):
                    matrix = True
                    col = (item == 'c')
                    continue
                if ',' in item:
                    vec = item.split(',')
                    try:
                        axis, ndmin = [int(x) for x in vec[:2]]
                        if len(vec) == 3:
                            trans1d = int(vec[2])
                        continue
                    except Exception as e:
                        raise ValueError(
                            f"unknown special directive {item!r}"
                        ) from e
                try:
                    axis = int(item)
                    continue
                except (ValueError, TypeError) as e:
                    raise ValueError("unknown special directive") from e
            elif type(item) in ScalarType:
                scalar = True
                newobj = item
            else:
                item_ndim = np.ndim(item)
                newobj = array(item, copy=None, subok=True, ndmin=ndmin)
                if trans1d != -1 and item_ndim < ndmin:
                    k2 = ndmin - item_ndim
                    k1 = trans1d
                    if k1 < 0:
                        k1 += k2 + 1
                    defaxes = list(range(ndmin))
                    axes = defaxes[:k1] + defaxes[k2:] + defaxes[k1:k2]
                    newobj = newobj.transpose(axes)

            objs.append(newobj)
            if scalar:
                result_type_objs.append(item)
            else:
                result_type_objs.append(newobj.dtype)

        # Ensure that scalars won't up-cast unless warranted, for 0, drops
        # through to error in concatenate.
        if len(result_type_objs) != 0:
            final_dtype = _nx.result_type(*result_type_objs)
            # concatenate could do cast, but that can be overridden:
            objs = [array(obj, copy=None, subok=True,
                          ndmin=ndmin, dtype=final_dtype) for obj in objs]

        res = self.concatenate(tuple(objs), axis=axis)

        if matrix:
            oldndim = res.ndim
            res = self.makemat(res)
            if oldndim == 1 and col:
                res = res.T
        return res

    def __len__(self):
        return 0

# separate classes are used here instead of just making r_ = concatenator(0),
# etc. because otherwise we couldn't get the doc string to come out right
# in help(r_)


class RClass(AxisConcatenator):
    """
    Translates slice objects to concatenation along the first axis.

    This is a simple way to build up arrays quickly. There are two use cases.

    1. If the index expression contains comma separated arrays, then stack
       them along their first axis.
    2. If the index expression contains slice notation or scalars then create
       a 1-D array with a range indicated by the slice notation.

    If slice notation is used, the syntax ``start:stop:step`` is equivalent
    to ``np.arange(start, stop, step)`` inside of the brackets. However, if
    ``step`` is an imaginary number (i.e. 100j) then its integer portion is
    interpreted as a number-of-points desired and the start and stop are
    inclusive. In other words ``start:stop:stepj`` is interpreted as
    ``np.linspace(start, stop, step, endpoint=1)`` inside of the brackets.
    After expansion of slice notation, all comma separated sequences are
    concatenated together.

    Optional character strings placed as the first element of the index
    expression can be used to change the output. The strings 'r' or 'c' result
    in matrix output. If the result is 1-D and 'r' is specified a 1 x N (row)
    matrix is produced. If the result is 1-D and 'c' is specified, then a N x 1
    (column) matrix is produced. If the result is 2-D then both provide the
    same matrix result.

    A string integer specifies which axis to stack multiple comma separated
    arrays along. A string of two comma-separated integers allows indication
    of the minimum number of dimensions to force each entry into as the
    second integer (the axis to concatenate along is still the first integer).

    A string with three comma-separated integers allows specification of the
    axis to concatenate along, the minimum number of dimensions to force the
    entries to, and which axis should contain the start of the arrays which
    are less than the specified number of dimensions. In other words the third
    integer allows you to specify where the 1's should be placed in the shape
    of the arrays that have their shapes upgraded. By default, they are placed
    in the front of the shape tuple. The third argument allows you to specify
    where the start of the array should be instead. Thus, a third argument of
    '0' would place the 1's at the end of the array shape. Negative integers
    specify where in the new shape tuple the last dimension of upgraded arrays
    should be placed, so the default is '-1'.

    Parameters
    ----------
    Not a function, so takes no parameters


    Returns
    -------
    A concatenated ndarray or matrix.

    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    c_ : Translates slice objects to concatenation along the second axis.

    Examples
    --------
    >>> import numpy as np
    >>> np.r_[np.array([1,2,3]), 0, 0, np.array([4,5,6])]
    array([1, 2, 3, ..., 4, 5, 6])
    >>> np.r_[-1:1:6j, [0]*3, 5, 6]
    array([-1. , -0.6, -0.2,  0.2,  0.6,  1. ,  0. ,  0. ,  0. ,  5. ,  6. ])

    String integers specify the axis to concatenate along or the minimum
    number of dimensions to force entries into.

    >>> a = np.array([[0, 1, 2], [3, 4, 5]])
    >>> np.r_['-1', a, a] # concatenate along last axis
    array([[0, 1, 2, 0, 1, 2],
           [3, 4, 5, 3, 4, 5]])
    >>> np.r_['0,2', [1,2,3], [4,5,6]] # concatenate along first axis, dim>=2
    array([[1, 2, 3],
           [4, 5, 6]])

    >>> np.r_['0,2,0', [1,2,3], [4,5,6]]
    array([[1],
           [2],
           [3],
           [4],
           [5],
           [6]])
    >>> np.r_['1,2,0', [1,2,3], [4,5,6]]
    array([[1, 4],
           [2, 5],
           [3, 6]])

    Using 'r' or 'c' as a first string argument creates a matrix.

    >>> np.r_['r',[1,2,3], [4,5,6]]
    matrix([[1, 2, 3, 4, 5, 6]])

    """
    __slots__ = ()

    def __init__(self):
        AxisConcatenator.__init__(self, 0)


r_ = RClass()


class CClass(AxisConcatenator):
    """
    Translates slice objects to concatenation along the second axis.

    This is short-hand for ``np.r_['-1,2,0', index expression]``, which is
    useful because of its common occurrence. In particular, arrays will be
    stacked along their last axis after being upgraded to at least 2-D with
    1's post-pended to the shape (column vectors made out of 1-D arrays).

    See Also
    --------
    column_stack : Stack 1-D arrays as columns into a 2-D array.
    r_ : For more detailed documentation.

    Examples
    --------
    >>> import numpy as np
    >>> np.c_[np.array([1,2,3]), np.array([4,5,6])]
    array([[1, 4],
           [2, 5],
           [3, 6]])
    >>> np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])]
    array([[1, 2, 3, ..., 4, 5, 6]])

    """
    __slots__ = ()

    def __init__(self):
        AxisConcatenator.__init__(self, -1, ndmin=2, trans1d=0)


c_ = CClass()


@set_module('numpy')
class ndenumerate:
    """
    Multidimensional index iterator.

    Return an iterator yielding pairs of array coordinates and values.

    Parameters
    ----------
    arr : ndarray
      Input array.

    See Also
    --------
    ndindex, flatiter

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> for index, x in np.ndenumerate(a):
    ...     print(index, x)
    (0, 0) 1
    (0, 1) 2
    (1, 0) 3
    (1, 1) 4

    """

    def __init__(self, arr):
        self.iter = np.asarray(arr).flat

    def __next__(self):
        """
        Standard iterator method, returns the index tuple and array value.

        Returns
        -------
        coords : tuple of ints
            The indices of the current iteration.
        val : scalar
            The array element of the current iteration.

        """
        return self.iter.coords, next(self.iter)

    def __iter__(self):
        return self


@set_module('numpy')
class ndindex:
    """
    An N-dimensional iterator object to index arrays.

    Given the shape of an array, an `ndindex` instance iterates over
    the N-dimensional index of the array. At each iteration a tuple
    of indices is returned, the last dimension is iterated over first.

    Parameters
    ----------
    shape : ints, or a single tuple of ints
        The size of each dimension of the array can be passed as
        individual parameters or as the elements of a tuple.

    See Also
    --------
    ndenumerate, flatiter

    Examples
    --------
    >>> import numpy as np

    Dimensions as individual arguments

    >>> for index in np.ndindex(3, 2, 1):
    ...     print(index)
    (0, 0, 0)
    (0, 1, 0)
    (1, 0, 0)
    (1, 1, 0)
    (2, 0, 0)
    (2, 1, 0)

    Same dimensions - but in a tuple ``(3, 2, 1)``

    >>> for index in np.ndindex((3, 2, 1)):
    ...     print(index)
    (0, 0, 0)
    (0, 1, 0)
    (1, 0, 0)
    (1, 1, 0)
    (2, 0, 0)
    (2, 1, 0)

    """

    def __init__(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        x = as_strided(_nx.zeros(1), shape=shape,
                       strides=_nx.zeros_like(shape))
        self._it = _nx.nditer(x, flags=['multi_index', 'zerosize_ok'],
                              order='C')

    def __iter__(self):
        return self

    def ndincr(self):
        """
        Increment the multi-dimensional index by one.

        This method is for backward compatibility only: do not use.

        .. deprecated:: 1.20.0
            This method has been advised against since numpy 1.8.0, but only
            started emitting DeprecationWarning as of this version.
        """
        # NumPy 1.20.0, 2020-09-08
        warnings.warn(
            "`ndindex.ndincr()` is deprecated, use `next(ndindex)` instead",
            DeprecationWarning, stacklevel=2)
        next(self)

    def __next__(self):
        """
        Standard iterator method, updates the index and returns the index
        tuple.

        Returns
        -------
        val : tuple of ints
            Returns a tuple containing the indices of the current
            iteration.

        """
        next(self._it)
        return self._it.multi_index


# You can do all this with slice() plus a few special objects,
# but there's a lot to remember. This version is simpler because
# it uses the standard array indexing syntax.
#
# Written by Konrad Hinsen <hinsen@cnrs-orleans.fr>
# last revision: 1999-7-23
#
# Cosmetic changes by T. Oliphant 2001
#
#

class IndexExpression:
    """
    A nicer way to build up index tuples for arrays.

    .. note::
       Use one of the two predefined instances ``index_exp`` or `s_`
       rather than directly using `IndexExpression`.

    For any index combination, including slicing and axis insertion,
    ``a[indices]`` is the same as ``a[np.index_exp[indices]]`` for any
    array `a`. However, ``np.index_exp[indices]`` can be used anywhere
    in Python code and returns a tuple of slice objects that can be
    used in the construction of complex index expressions.

    Parameters
    ----------
    maketuple : bool
        If True, always returns a tuple.

    See Also
    --------
    s_ : Predefined instance without tuple conversion:
       `s_ = IndexExpression(maketuple=False)`.
       The ``index_exp`` is another predefined instance that
       always returns a tuple:
       `index_exp = IndexExpression(maketuple=True)`.

    Notes
    -----
    You can do all this with :class:`slice` plus a few special objects,
    but there's a lot to remember and this version is simpler because
    it uses the standard array indexing syntax.

    Examples
    --------
    >>> import numpy as np
    >>> np.s_[2::2]
    slice(2, None, 2)
    >>> np.index_exp[2::2]
    (slice(2, None, 2),)

    >>> np.array([0, 1, 2, 3, 4])[np.s_[2::2]]
    array([2, 4])

    """
    __slots__ = ('maketuple',)

    def __init__(self, maketuple):
        self.maketuple = maketuple

    def __getitem__(self, item):
        if self.maketuple and not isinstance(item, tuple):
            return (item,)
        else:
            return item


index_exp = IndexExpression(maketuple=True)
s_ = IndexExpression(maketuple=False)

# End contribution from Konrad.


# The following functions complement those in twodim_base, but are
# applicable to N-dimensions.


def _fill_diagonal_dispatcher(a, val, wrap=None):
    return (a,)


@array_function_dispatch(_fill_diagonal_dispatcher)
def fill_diagonal(a, val, wrap=False):
    """Fill the main diagonal of the given array of any dimensionality.

    For an array `a` with ``a.ndim >= 2``, the diagonal is the list of
    values ``a[i, ..., i]`` with indices ``i`` all identical.  This function
    modifies the input array in-place without returning a value.

    Parameters
    ----------
    a : array, at least 2-D.
      Array whose diagonal is to be filled in-place.
    val : scalar or array_like
      Value(s) to write on the diagonal. If `val` is scalar, the value is
      written along the diagonal. If array-like, the flattened `val` is
      written along the diagonal, repeating if necessary to fill all
      diagonal entries.

    wrap : bool
      For tall matrices in NumPy version up to 1.6.2, the
      diagonal "wrapped" after N columns. You can have this behavior
      with this option. This affects only tall matrices.

    See also
    --------
    diag_indices, diag_indices_from

    Notes
    -----
    This functionality can be obtained via `diag_indices`, but internally
    this version uses a much faster implementation that never constructs the
    indices and uses simple slicing.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.zeros((3, 3), int)
    >>> np.fill_diagonal(a, 5)
    >>> a
    array([[5, 0, 0],
           [0, 5, 0],
           [0, 0, 5]])

    The same function can operate on a 4-D array:

    >>> a = np.zeros((3, 3, 3, 3), int)
    >>> np.fill_diagonal(a, 4)

    We only show a few blocks for clarity:

    >>> a[0, 0]
    array([[4, 0, 0],
           [0, 0, 0],
           [0, 0, 0]])
    >>> a[1, 1]
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 0]])
    >>> a[2, 2]
    array([[0, 0, 0],
           [0, 0, 0],
           [0, 0, 4]])

    The wrap option affects only tall matrices:

    >>> # tall matrices no wrap
    >>> a = np.zeros((5, 3), int)
    >>> np.fill_diagonal(a, 4)
    >>> a
    array([[4, 0, 0],
           [0, 4, 0],
           [0, 0, 4],
           [0, 0, 0],
           [0, 0, 0]])

    >>> # tall matrices wrap
    >>> a = np.zeros((5, 3), int)
    >>> np.fill_diagonal(a, 4, wrap=True)
    >>> a
    array([[4, 0, 0],
           [0, 4, 0],
           [0, 0, 4],
           [0, 0, 0],
           [4, 0, 0]])

    >>> # wide matrices
    >>> a = np.zeros((3, 5), int)
    >>> np.fill_diagonal(a, 4, wrap=True)
    >>> a
    array([[4, 0, 0, 0, 0],
           [0, 4, 0, 0, 0],
           [0, 0, 4, 0, 0]])

    The anti-diagonal can be filled by reversing the order of elements
    using either `numpy.flipud` or `numpy.fliplr`.

    >>> a = np.zeros((3, 3), int);
    >>> np.fill_diagonal(np.fliplr(a), [1,2,3])  # Horizontal flip
    >>> a
    array([[0, 0, 1],
           [0, 2, 0],
           [3, 0, 0]])
    >>> np.fill_diagonal(np.flipud(a), [1,2,3])  # Vertical flip
    >>> a
    array([[0, 0, 3],
           [0, 2, 0],
           [1, 0, 0]])

    Note that the order in which the diagonal is filled varies depending
    on the flip function.
    """
    if a.ndim < 2:
        raise ValueError("array must be at least 2-d")
    end = None
    if a.ndim == 2:
        # Explicit, fast formula for the common case.  For 2-d arrays, we
        # accept rectangular ones.
        step = a.shape[1] + 1
        # This is needed to don't have tall matrix have the diagonal wrap.
        if not wrap:
            end = a.shape[1] * a.shape[1]
    else:
        # For more than d=2, the strided formula is only valid for arrays with
        # all dimensions equal, so we check first.
        if not np.all(diff(a.shape) == 0):
            raise ValueError("All dimensions of input must be of equal length")
        step = 1 + (np.cumprod(a.shape[:-1])).sum()

    # Write the value out into the diagonal.
    a.flat[:end:step] = val


@set_module('numpy')
def diag_indices(n, ndim=2):
    """
    Return the indices to access the main diagonal of an array.

    This returns a tuple of indices that can be used to access the main
    diagonal of an array `a` with ``a.ndim >= 2`` dimensions and shape
    (n, n, ..., n). For ``a.ndim = 2`` this is the usual diagonal, for
    ``a.ndim > 2`` this is the set of indices to access ``a[i, i, ..., i]``
    for ``i = [0..n-1]``.

    Parameters
    ----------
    n : int
      The size, along each dimension, of the arrays for which the returned
      indices can be used.

    ndim : int, optional
      The number of dimensions.

    See Also
    --------
    diag_indices_from

    Examples
    --------
    >>> import numpy as np

    Create a set of indices to access the diagonal of a (4, 4) array:

    >>> di = np.diag_indices(4)
    >>> di
    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))
    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> a[di] = 100
    >>> a
    array([[100,   1,   2,   3],
           [  4, 100,   6,   7],
           [  8,   9, 100,  11],
           [ 12,  13,  14, 100]])

    Now, we create indices to manipulate a 3-D array:

    >>> d3 = np.diag_indices(2, 3)
    >>> d3
    (array([0, 1]), array([0, 1]), array([0, 1]))

    And use it to set the diagonal of an array of zeros to 1:

    >>> a = np.zeros((2, 2, 2), dtype=int)
    >>> a[d3] = 1
    >>> a
    array([[[1, 0],
            [0, 0]],
           [[0, 0],
            [0, 1]]])

    """
    idx = np.arange(n)
    return (idx,) * ndim


def _diag_indices_from(arr):
    return (arr,)


@array_function_dispatch(_diag_indices_from)
def diag_indices_from(arr):
    """
    Return the indices to access the main diagonal of an n-dimensional array.

    See `diag_indices` for full details.

    Parameters
    ----------
    arr : array, at least 2-D

    See Also
    --------
    diag_indices

    Examples
    --------
    >>> import numpy as np

    Create a 4 by 4 array.

    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])

    Get the indices of the diagonal elements.

    >>> di = np.diag_indices_from(a)
    >>> di
    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))

    >>> a[di]
    array([ 0,  5, 10, 15])

    This is simply syntactic sugar for diag_indices.

    >>> np.diag_indices(a.shape[0])
    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))

    """

    if not arr.ndim >= 2:
        raise ValueError("input array must be at least 2-d")
    # For more than d=2, the strided formula is only valid for arrays with
    # all dimensions equal, so we check first.
    if not np.all(diff(arr.shape) == 0):
        raise ValueError("All dimensions of input must be of equal length")

    return diag_indices(arr.shape[0], arr.ndim)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\tests\test_format.py ===
# doctest
r''' Test the .npy file format.

Set up:

    >>> import sys
    >>> from io import BytesIO
    >>> from numpy.lib import format
    >>>
    >>> scalars = [
    ...     np.uint8,
    ...     np.int8,
    ...     np.uint16,
    ...     np.int16,
    ...     np.uint32,
    ...     np.int32,
    ...     np.uint64,
    ...     np.int64,
    ...     np.float32,
    ...     np.float64,
    ...     np.complex64,
    ...     np.complex128,
    ...     object,
    ... ]
    >>>
    >>> basic_arrays = []
    >>>
    >>> for scalar in scalars:
    ...     for endian in '<>':
    ...         dtype = np.dtype(scalar).newbyteorder(endian)
    ...         basic = np.arange(15).astype(dtype)
    ...         basic_arrays.extend([
    ...             np.array([], dtype=dtype),
    ...             np.array(10, dtype=dtype),
    ...             basic,
    ...             basic.reshape((3,5)),
    ...             basic.reshape((3,5)).T,
    ...             basic.reshape((3,5))[::-1,::2],
    ...         ])
    ...
    >>>
    >>> Pdescr = [
    ...     ('x', 'i4', (2,)),
    ...     ('y', 'f8', (2, 2)),
    ...     ('z', 'u1')]
    >>>
    >>>
    >>> PbufferT = [
    ...     ([3,2], [[6.,4.],[6.,4.]], 8),
    ...     ([4,3], [[7.,5.],[7.,5.]], 9),
    ...     ]
    >>>
    >>>
    >>> Ndescr = [
    ...     ('x', 'i4', (2,)),
    ...     ('Info', [
    ...         ('value', 'c16'),
    ...         ('y2', 'f8'),
    ...         ('Info2', [
    ...             ('name', 'S2'),
    ...             ('value', 'c16', (2,)),
    ...             ('y3', 'f8', (2,)),
    ...             ('z3', 'u4', (2,))]),
    ...         ('name', 'S2'),
    ...         ('z2', 'b1')]),
    ...     ('color', 'S2'),
    ...     ('info', [
    ...         ('Name', 'U8'),
    ...         ('Value', 'c16')]),
    ...     ('y', 'f8', (2, 2)),
    ...     ('z', 'u1')]
    >>>
    >>>
    >>> NbufferT = [
    ...     ([3,2], (6j, 6., ('nn', [6j,4j], [6.,4.], [1,2]), 'NN', True), 'cc', ('NN', 6j), [[6.,4.],[6.,4.]], 8),
    ...     ([4,3], (7j, 7., ('oo', [7j,5j], [7.,5.], [2,1]), 'OO', False), 'dd', ('OO', 7j), [[7.,5.],[7.,5.]], 9),
    ...     ]
    >>>
    >>>
    >>> record_arrays = [
    ...     np.array(PbufferT, dtype=np.dtype(Pdescr).newbyteorder('<')),
    ...     np.array(NbufferT, dtype=np.dtype(Ndescr).newbyteorder('<')),
    ...     np.array(PbufferT, dtype=np.dtype(Pdescr).newbyteorder('>')),
    ...     np.array(NbufferT, dtype=np.dtype(Ndescr).newbyteorder('>')),
    ... ]

Test the magic string writing.

    >>> format.magic(1, 0)
    '\x93NUMPY\x01\x00'
    >>> format.magic(0, 0)
    '\x93NUMPY\x00\x00'
    >>> format.magic(255, 255)
    '\x93NUMPY\xff\xff'
    >>> format.magic(2, 5)
    '\x93NUMPY\x02\x05'

Test the magic string reading.

    >>> format.read_magic(BytesIO(format.magic(1, 0)))
    (1, 0)
    >>> format.read_magic(BytesIO(format.magic(0, 0)))
    (0, 0)
    >>> format.read_magic(BytesIO(format.magic(255, 255)))
    (255, 255)
    >>> format.read_magic(BytesIO(format.magic(2, 5)))
    (2, 5)

Test the header writing.

    >>> for arr in basic_arrays + record_arrays:
    ...     f = BytesIO()
    ...     format.write_array_header_1_0(f, arr)   # XXX: arr is not a dict, items gets called on it
    ...     print(repr(f.getvalue()))
    ...
    "F\x00{'descr': '|u1', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '|u1', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '|u1', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '|u1', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '|u1', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '|u1', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '|u1', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '|u1', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '|u1', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '|u1', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '|u1', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '|u1', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '|i1', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '|i1', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '|i1', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '|i1', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '|i1', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '|i1', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '|i1', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '|i1', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '|i1', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '|i1', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '|i1', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '|i1', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<u2', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<u2', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<u2', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<u2', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<u2', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<u2', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>u2', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>u2', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>u2', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '>u2', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>u2', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>u2', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<i2', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<i2', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<i2', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<i2', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<i2', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<i2', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>i2', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>i2', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>i2', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '>i2', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>i2', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>i2', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<u4', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<u4', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<u4', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<u4', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<u4', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<u4', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>u4', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>u4', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>u4', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '>u4', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>u4', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>u4', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<i4', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<i4', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<i4', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<i4', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<i4', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<i4', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>i4', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>i4', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>i4', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '>i4', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>i4', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>i4', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<u8', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<u8', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<u8', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<u8', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<u8', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<u8', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>u8', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>u8', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>u8', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '>u8', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>u8', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>u8', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<i8', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<i8', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<i8', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<i8', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<i8', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<i8', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>i8', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>i8', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>i8', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '>i8', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>i8', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>i8', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<f4', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<f4', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<f4', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<f4', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<f4', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<f4', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>f4', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>f4', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>f4', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '>f4', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>f4', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>f4', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<f8', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<f8', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<f8', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<f8', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<f8', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<f8', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>f8', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>f8', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>f8', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '>f8', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>f8', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>f8', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<c8', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<c8', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<c8', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<c8', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<c8', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<c8', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>c8', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>c8', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>c8', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '>c8', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>c8', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>c8', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<c16', 'fortran_order': False, 'shape': (0,)}             \n"
    "F\x00{'descr': '<c16', 'fortran_order': False, 'shape': ()}               \n"
    "F\x00{'descr': '<c16', 'fortran_order': False, 'shape': (15,)}            \n"
    "F\x00{'descr': '<c16', 'fortran_order': False, 'shape': (3, 5)}           \n"
    "F\x00{'descr': '<c16', 'fortran_order': True, 'shape': (5, 3)}            \n"
    "F\x00{'descr': '<c16', 'fortran_order': False, 'shape': (3, 3)}           \n"
    "F\x00{'descr': '>c16', 'fortran_order': False, 'shape': (0,)}             \n"
    "F\x00{'descr': '>c16', 'fortran_order': False, 'shape': ()}               \n"
    "F\x00{'descr': '>c16', 'fortran_order': False, 'shape': (15,)}            \n"
    "F\x00{'descr': '>c16', 'fortran_order': False, 'shape': (3, 5)}           \n"
    "F\x00{'descr': '>c16', 'fortran_order': True, 'shape': (5, 3)}            \n"
    "F\x00{'descr': '>c16', 'fortran_order': False, 'shape': (3, 3)}           \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': 'O', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': 'O', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "v\x00{'descr': [('x', '<i4', (2,)), ('y', '<f8', (2, 2)), ('z', '|u1')],\n 'fortran_order': False,\n 'shape': (2,)}         \n"
    "\x16\x02{'descr': [('x', '<i4', (2,)),\n           ('Info',\n            [('value', '<c16'),\n             ('y2', '<f8'),\n             ('Info2',\n              [('name', '|S2'),\n               ('value', '<c16', (2,)),\n               ('y3', '<f8', (2,)),\n               ('z3', '<u4', (2,))]),\n             ('name', '|S2'),\n             ('z2', '|b1')]),\n           ('color', '|S2'),\n           ('info', [('Name', '<U8'), ('Value', '<c16')]),\n           ('y', '<f8', (2, 2)),\n           ('z', '|u1')],\n 'fortran_order': False,\n 'shape': (2,)}      \n"
    "v\x00{'descr': [('x', '>i4', (2,)), ('y', '>f8', (2, 2)), ('z', '|u1')],\n 'fortran_order': False,\n 'shape': (2,)}         \n"
    "\x16\x02{'descr': [('x', '>i4', (2,)),\n           ('Info',\n            [('value', '>c16'),\n             ('y2', '>f8'),\n             ('Info2',\n              [('name', '|S2'),\n               ('value', '>c16', (2,)),\n               ('y3', '>f8', (2,)),\n               ('z3', '>u4', (2,))]),\n             ('name', '|S2'),\n             ('z2', '|b1')]),\n           ('color', '|S2'),\n           ('info', [('Name', '>U8'), ('Value', '>c16')]),\n           ('y', '>f8', (2, 2)),\n           ('z', '|u1')],\n 'fortran_order': False,\n 'shape': (2,)}      \n"
'''
import os
import sys
import warnings
from io import BytesIO

import pytest

import numpy as np
from numpy.lib import format
from numpy.testing import (
    IS_64BIT,
    IS_PYPY,
    IS_WASM,
    assert_,
    assert_array_equal,
    assert_raises,
    assert_raises_regex,
    assert_warns,
)
from numpy.testing._private.utils import requires_memory

# Generate some basic arrays to test with.
scalars = [
    np.uint8,
    np.int8,
    np.uint16,
    np.int16,
    np.uint32,
    np.int32,
    np.uint64,
    np.int64,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
    object,
]
basic_arrays = []
for scalar in scalars:
    for endian in '<>':
        dtype = np.dtype(scalar).newbyteorder(endian)
        basic = np.arange(1500).astype(dtype)
        basic_arrays.extend([
            # Empty
            np.array([], dtype=dtype),
            # Rank-0
            np.array(10, dtype=dtype),
            # 1-D
            basic,
            # 2-D C-contiguous
            basic.reshape((30, 50)),
            # 2-D F-contiguous
            basic.reshape((30, 50)).T,
            # 2-D non-contiguous
            basic.reshape((30, 50))[::-1, ::2],
        ])

# More complicated record arrays.
# This is the structure of the table used for plain objects:
#
# +-+-+-+
# |x|y|z|
# +-+-+-+

# Structure of a plain array description:
Pdescr = [
    ('x', 'i4', (2,)),
    ('y', 'f8', (2, 2)),
    ('z', 'u1')]

# A plain list of tuples with values for testing:
PbufferT = [
    # x     y                  z
    ([3, 2], [[6., 4.], [6., 4.]], 8),
    ([4, 3], [[7., 5.], [7., 5.]], 9),
    ]


# This is the structure of the table used for nested objects (DON'T PANIC!):
#
# +-+---------------------------------+-----+----------+-+-+
# |x|Info                             |color|info      |y|z|
# | +-----+--+----------------+----+--+     +----+-----+ | |
# | |value|y2|Info2           |name|z2|     |Name|Value| | |
# | |     |  +----+-----+--+--+    |  |     |    |     | | |
# | |     |  |name|value|y3|z3|    |  |     |    |     | | |
# +-+-----+--+----+-----+--+--+----+--+-----+----+-----+-+-+
#

# The corresponding nested array description:
Ndescr = [
    ('x', 'i4', (2,)),
    ('Info', [
        ('value', 'c16'),
        ('y2', 'f8'),
        ('Info2', [
            ('name', 'S2'),
            ('value', 'c16', (2,)),
            ('y3', 'f8', (2,)),
            ('z3', 'u4', (2,))]),
        ('name', 'S2'),
        ('z2', 'b1')]),
    ('color', 'S2'),
    ('info', [
        ('Name', 'U8'),
        ('Value', 'c16')]),
    ('y', 'f8', (2, 2)),
    ('z', 'u1')]

NbufferT = [
    # x     Info                                                color info        y                  z
    #       value y2 Info2                            name z2         Name Value
    #                name   value    y3       z3
    ([3, 2], (6j, 6., ('nn', [6j, 4j], [6., 4.], [1, 2]), 'NN', True),
     'cc', ('NN', 6j), [[6., 4.], [6., 4.]], 8),
    ([4, 3], (7j, 7., ('oo', [7j, 5j], [7., 5.], [2, 1]), 'OO', False),
     'dd', ('OO', 7j), [[7., 5.], [7., 5.]], 9),
    ]

record_arrays = [
    np.array(PbufferT, dtype=np.dtype(Pdescr).newbyteorder('<')),
    np.array(NbufferT, dtype=np.dtype(Ndescr).newbyteorder('<')),
    np.array(PbufferT, dtype=np.dtype(Pdescr).newbyteorder('>')),
    np.array(NbufferT, dtype=np.dtype(Ndescr).newbyteorder('>')),
    np.zeros(1, dtype=[('c', ('<f8', (5,)), (2,))])
]


# BytesIO that reads a random number of bytes at a time
class BytesIOSRandomSize(BytesIO):
    def read(self, size=None):
        import random
        size = random.randint(1, size)
        return super().read(size)


def roundtrip(arr):
    f = BytesIO()
    format.write_array(f, arr)
    f2 = BytesIO(f.getvalue())
    arr2 = format.read_array(f2, allow_pickle=True)
    return arr2


def roundtrip_randsize(arr):
    f = BytesIO()
    format.write_array(f, arr)
    f2 = BytesIOSRandomSize(f.getvalue())
    arr2 = format.read_array(f2)
    return arr2


def roundtrip_truncated(arr):
    f = BytesIO()
    format.write_array(f, arr)
    # BytesIO is one byte short
    f2 = BytesIO(f.getvalue()[0:-1])
    arr2 = format.read_array(f2)
    return arr2

def assert_equal_(o1, o2):
    assert_(o1 == o2)


def test_roundtrip():
    for arr in basic_arrays + record_arrays:
        arr2 = roundtrip(arr)
        assert_array_equal(arr, arr2)


def test_roundtrip_randsize():
    for arr in basic_arrays + record_arrays:
        if arr.dtype != object:
            arr2 = roundtrip_randsize(arr)
            assert_array_equal(arr, arr2)


def test_roundtrip_truncated():
    for arr in basic_arrays:
        if arr.dtype != object:
            assert_raises(ValueError, roundtrip_truncated, arr)

def test_file_truncated(tmp_path):
    path = tmp_path / "a.npy"
    for arr in basic_arrays:
        if arr.dtype != object:
            with open(path, 'wb') as f:
                format.write_array(f, arr)
            # truncate the file by one byte
            with open(path, 'rb+') as f:
                f.seek(-1, os.SEEK_END)
                f.truncate()
            with open(path, 'rb') as f:
                with pytest.raises(
                    ValueError,
                    match=(
                        r"EOF: reading array header, "
                        r"expected (\d+) bytes got (\d+)"
                    ) if arr.size == 0 else (
                        r"Failed to read all data for array\. "
                        r"Expected \(.*?\) = (\d+) elements, "
                        r"could only read (\d+) elements\. "
                        r"\(file seems not fully written\?\)"
                    )
                ):
                    _ = format.read_array(f)

def test_long_str():
    # check items larger than internal buffer size, gh-4027
    long_str_arr = np.ones(1, dtype=np.dtype((str, format.BUFFER_SIZE + 1)))
    long_str_arr2 = roundtrip(long_str_arr)
    assert_array_equal(long_str_arr, long_str_arr2)


@pytest.mark.skipif(IS_WASM, reason="memmap doesn't work correctly")
@pytest.mark.slow
def test_memmap_roundtrip(tmpdir):
    for i, arr in enumerate(basic_arrays + record_arrays):
        if arr.dtype.hasobject:
            # Skip these since they can't be mmap'ed.
            continue
        # Write it out normally and through mmap.
        nfn = os.path.join(tmpdir, f'normal{i}.npy')
        mfn = os.path.join(tmpdir, f'memmap{i}.npy')
        with open(nfn, 'wb') as fp:
            format.write_array(fp, arr)

        fortran_order = (
            arr.flags.f_contiguous and not arr.flags.c_contiguous)
        ma = format.open_memmap(mfn, mode='w+', dtype=arr.dtype,
                                shape=arr.shape, fortran_order=fortran_order)
        ma[...] = arr
        ma.flush()

        # Check that both of these files' contents are the same.
        with open(nfn, 'rb') as fp:
            normal_bytes = fp.read()
        with open(mfn, 'rb') as fp:
            memmap_bytes = fp.read()
        assert_equal_(normal_bytes, memmap_bytes)

        # Check that reading the file using memmap works.
        ma = format.open_memmap(nfn, mode='r')
        ma.flush()


def test_compressed_roundtrip(tmpdir):
    arr = np.random.rand(200, 200)
    npz_file = os.path.join(tmpdir, 'compressed.npz')
    np.savez_compressed(npz_file, arr=arr)
    with np.load(npz_file) as npz:
        arr1 = npz['arr']
    assert_array_equal(arr, arr1)


# aligned
dt1 = np.dtype('i1, i4, i1', align=True)
# non-aligned, explicit offsets
dt2 = np.dtype({'names': ['a', 'b'], 'formats': ['i4', 'i4'],
                'offsets': [1, 6]})
# nested struct-in-struct
dt3 = np.dtype({'names': ['c', 'd'], 'formats': ['i4', dt2]})
# field with '' name
dt4 = np.dtype({'names': ['a', '', 'b'], 'formats': ['i4'] * 3})
# titles
dt5 = np.dtype({'names': ['a', 'b'], 'formats': ['i4', 'i4'],
                'offsets': [1, 6], 'titles': ['aa', 'bb']})
# empty
dt6 = np.dtype({'names': [], 'formats': [], 'itemsize': 8})

@pytest.mark.parametrize("dt", [dt1, dt2, dt3, dt4, dt5, dt6])
def test_load_padded_dtype(tmpdir, dt):
    arr = np.zeros(3, dt)
    for i in range(3):
        arr[i] = i + 5
    npz_file = os.path.join(tmpdir, 'aligned.npz')
    np.savez(npz_file, arr=arr)
    with np.load(npz_file) as npz:
        arr1 = npz['arr']
    assert_array_equal(arr, arr1)


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="see gh-23988")
@pytest.mark.xfail(IS_WASM, reason="Emscripten NODEFS has a buggy dup")
def test_python2_python3_interoperability():
    fname = 'win64python2.npy'
    path = os.path.join(os.path.dirname(__file__), 'data', fname)
    with pytest.warns(UserWarning, match="Reading.*this warning\\."):
        data = np.load(path)
    assert_array_equal(data, np.ones(2))


def test_pickle_python2_python3():
    # Test that loading object arrays saved on Python 2 works both on
    # Python 2 and Python 3 and vice versa
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    expected = np.array([None, range, '\u512a\u826f',
                         b'\xe4\xb8\x8d\xe8\x89\xaf'],
                        dtype=object)

    for fname in ['py2-np0-objarr.npy', 'py2-objarr.npy', 'py2-objarr.npz',
                  'py3-objarr.npy', 'py3-objarr.npz']:
        path = os.path.join(data_dir, fname)

        for encoding in ['bytes', 'latin1']:
            data_f = np.load(path, allow_pickle=True, encoding=encoding)
            if fname.endswith('.npz'):
                data = data_f['x']
                data_f.close()
            else:
                data = data_f

            if encoding == 'latin1' and fname.startswith('py2'):
                assert_(isinstance(data[3], str))
                assert_array_equal(data[:-1], expected[:-1])
                # mojibake occurs
                assert_array_equal(data[-1].encode(encoding), expected[-1])
            else:
                assert_(isinstance(data[3], bytes))
                assert_array_equal(data, expected)

        if fname.startswith('py2'):
            if fname.endswith('.npz'):
                data = np.load(path, allow_pickle=True)
                assert_raises(UnicodeError, data.__getitem__, 'x')
                data.close()
                data = np.load(path, allow_pickle=True, fix_imports=False,
                               encoding='latin1')
                assert_raises(ImportError, data.__getitem__, 'x')
                data.close()
            else:
                assert_raises(UnicodeError, np.load, path,
                              allow_pickle=True)
                assert_raises(ImportError, np.load, path,
                              allow_pickle=True, fix_imports=False,
                              encoding='latin1')


def test_pickle_disallow(tmpdir):
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    path = os.path.join(data_dir, 'py2-objarr.npy')
    assert_raises(ValueError, np.load, path,
                  allow_pickle=False, encoding='latin1')

    path = os.path.join(data_dir, 'py2-objarr.npz')
    with np.load(path, allow_pickle=False, encoding='latin1') as f:
        assert_raises(ValueError, f.__getitem__, 'x')

    path = os.path.join(tmpdir, 'pickle-disabled.npy')
    assert_raises(ValueError, np.save, path, np.array([None], dtype=object),
                  allow_pickle=False)

@pytest.mark.parametrize('dt', [
    np.dtype(np.dtype([('a', np.int8),
                       ('b', np.int16),
                       ('c', np.int32),
                      ], align=True),
             (3,)),
    np.dtype([('x', np.dtype({'names': ['a', 'b'],
                              'formats': ['i1', 'i1'],
                              'offsets': [0, 4],
                              'itemsize': 8,
                             },
                    (3,)),
               (4,),
             )]),
    np.dtype([('x',
                   ('<f8', (5,)),
                   (2,),
               )]),
    np.dtype([('x', np.dtype((
        np.dtype((
            np.dtype({'names': ['a', 'b'],
                      'formats': ['i1', 'i1'],
                      'offsets': [0, 4],
                      'itemsize': 8}),
            (3,)
            )),
        (4,)
        )))
        ]),
    np.dtype([
        ('a', np.dtype((
            np.dtype((
                np.dtype((
                    np.dtype([
                        ('a', int),
                        ('b', np.dtype({'names': ['a', 'b'],
                                        'formats': ['i1', 'i1'],
                                        'offsets': [0, 4],
                                        'itemsize': 8})),
                    ]),
                    (3,),
                )),
                (4,),
            )),
            (5,),
        )))
        ]),
    ])
def test_descr_to_dtype(dt):
    dt1 = format.descr_to_dtype(dt.descr)
    assert_equal_(dt1, dt)
    arr1 = np.zeros(3, dt)
    arr2 = roundtrip(arr1)
    assert_array_equal(arr1, arr2)

def test_version_2_0():
    f = BytesIO()
    # requires more than 2 byte for header
    dt = [(("%d" % i) * 100, float) for i in range(500)]
    d = np.ones(1000, dtype=dt)

    format.write_array(f, d, version=(2, 0))
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('always', '', UserWarning)
        format.write_array(f, d)
        assert_(w[0].category is UserWarning)

    # check alignment of data portion
    f.seek(0)
    header = f.readline()
    assert_(len(header) % format.ARRAY_ALIGN == 0)

    f.seek(0)
    n = format.read_array(f, max_header_size=200000)
    assert_array_equal(d, n)

    # 1.0 requested but data cannot be saved this way
    assert_raises(ValueError, format.write_array, f, d, (1, 0))


@pytest.mark.skipif(IS_WASM, reason="memmap doesn't work correctly")
def test_version_2_0_memmap(tmpdir):
    # requires more than 2 byte for header
    dt = [(("%d" % i) * 100, float) for i in range(500)]
    d = np.ones(1000, dtype=dt)
    tf1 = os.path.join(tmpdir, 'version2_01.npy')
    tf2 = os.path.join(tmpdir, 'version2_02.npy')

    # 1.0 requested but data cannot be saved this way
    assert_raises(ValueError, format.open_memmap, tf1, mode='w+', dtype=d.dtype,
                            shape=d.shape, version=(1, 0))

    ma = format.open_memmap(tf1, mode='w+', dtype=d.dtype,
                            shape=d.shape, version=(2, 0))
    ma[...] = d
    ma.flush()
    ma = format.open_memmap(tf1, mode='r', max_header_size=200000)
    assert_array_equal(ma, d)

    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('always', '', UserWarning)
        ma = format.open_memmap(tf2, mode='w+', dtype=d.dtype,
                                shape=d.shape, version=None)
        assert_(w[0].category is UserWarning)
        ma[...] = d
        ma.flush()

    ma = format.open_memmap(tf2, mode='r', max_header_size=200000)

    assert_array_equal(ma, d)

@pytest.mark.parametrize("mmap_mode", ["r", None])
def test_huge_header(tmpdir, mmap_mode):
    f = os.path.join(tmpdir, 'large_header.npy')
    arr = np.array(1, dtype="i," * 10000 + "i")

    with pytest.warns(UserWarning, match=".*format 2.0"):
        np.save(f, arr)

    with pytest.raises(ValueError, match="Header.*large"):
        np.load(f, mmap_mode=mmap_mode)

    with pytest.raises(ValueError, match="Header.*large"):
        np.load(f, mmap_mode=mmap_mode, max_header_size=20000)

    res = np.load(f, mmap_mode=mmap_mode, allow_pickle=True)
    assert_array_equal(res, arr)

    res = np.load(f, mmap_mode=mmap_mode, max_header_size=180000)
    assert_array_equal(res, arr)

def test_huge_header_npz(tmpdir):
    f = os.path.join(tmpdir, 'large_header.npz')
    arr = np.array(1, dtype="i," * 10000 + "i")

    with pytest.warns(UserWarning, match=".*format 2.0"):
        np.savez(f, arr=arr)

    # Only getting the array from the file actually reads it
    with pytest.raises(ValueError, match="Header.*large"):
        np.load(f)["arr"]

    with pytest.raises(ValueError, match="Header.*large"):
        np.load(f, max_header_size=20000)["arr"]

    res = np.load(f, allow_pickle=True)["arr"]
    assert_array_equal(res, arr)

    res = np.load(f, max_header_size=180000)["arr"]
    assert_array_equal(res, arr)

def test_write_version():
    f = BytesIO()
    arr = np.arange(1)
    # These should pass.
    format.write_array(f, arr, version=(1, 0))
    format.write_array(f, arr)

    format.write_array(f, arr, version=None)
    format.write_array(f, arr)

    format.write_array(f, arr, version=(2, 0))
    format.write_array(f, arr)

    # These should all fail.
    bad_versions = [
        (1, 1),
        (0, 0),
        (0, 1),
        (2, 2),
        (255, 255),
    ]
    for version in bad_versions:
        with assert_raises_regex(ValueError,
                                 'we only support format version.*'):
            format.write_array(f, arr, version=version)


bad_version_magic = [
    b'\x93NUMPY\x01\x01',
    b'\x93NUMPY\x00\x00',
    b'\x93NUMPY\x00\x01',
    b'\x93NUMPY\x02\x00',
    b'\x93NUMPY\x02\x02',
    b'\x93NUMPY\xff\xff',
]
malformed_magic = [
    b'\x92NUMPY\x01\x00',
    b'\x00NUMPY\x01\x00',
    b'\x93numpy\x01\x00',
    b'\x93MATLB\x01\x00',
    b'\x93NUMPY\x01',
    b'\x93NUMPY',
    b'',
]

def test_read_magic():
    s1 = BytesIO()
    s2 = BytesIO()

    arr = np.ones((3, 6), dtype=float)

    format.write_array(s1, arr, version=(1, 0))
    format.write_array(s2, arr, version=(2, 0))

    s1.seek(0)
    s2.seek(0)

    version1 = format.read_magic(s1)
    version2 = format.read_magic(s2)

    assert_(version1 == (1, 0))
    assert_(version2 == (2, 0))

    assert_(s1.tell() == format.MAGIC_LEN)
    assert_(s2.tell() == format.MAGIC_LEN)

def test_read_magic_bad_magic():
    for magic in malformed_magic:
        f = BytesIO(magic)
        assert_raises(ValueError, format.read_array, f)


def test_read_version_1_0_bad_magic():
    for magic in bad_version_magic + malformed_magic:
        f = BytesIO(magic)
        assert_raises(ValueError, format.read_array, f)


def test_bad_magic_args():
    assert_raises(ValueError, format.magic, -1, 1)
    assert_raises(ValueError, format.magic, 256, 1)
    assert_raises(ValueError, format.magic, 1, -1)
    assert_raises(ValueError, format.magic, 1, 256)


def test_large_header():
    s = BytesIO()
    d = {'shape': (), 'fortran_order': False, 'descr': '<i8'}
    format.write_array_header_1_0(s, d)

    s = BytesIO()
    d['descr'] = [('x' * 256 * 256, '<i8')]
    assert_raises(ValueError, format.write_array_header_1_0, s, d)


def test_read_array_header_1_0():
    s = BytesIO()

    arr = np.ones((3, 6), dtype=float)
    format.write_array(s, arr, version=(1, 0))

    s.seek(format.MAGIC_LEN)
    shape, fortran, dtype = format.read_array_header_1_0(s)

    assert_(s.tell() % format.ARRAY_ALIGN == 0)
    assert_((shape, fortran, dtype) == ((3, 6), False, float))


def test_read_array_header_2_0():
    s = BytesIO()

    arr = np.ones((3, 6), dtype=float)
    format.write_array(s, arr, version=(2, 0))

    s.seek(format.MAGIC_LEN)
    shape, fortran, dtype = format.read_array_header_2_0(s)

    assert_(s.tell() % format.ARRAY_ALIGN == 0)
    assert_((shape, fortran, dtype) == ((3, 6), False, float))


def test_bad_header():
    # header of length less than 2 should fail
    s = BytesIO()
    assert_raises(ValueError, format.read_array_header_1_0, s)
    s = BytesIO(b'1')
    assert_raises(ValueError, format.read_array_header_1_0, s)

    # header shorter than indicated size should fail
    s = BytesIO(b'\x01\x00')
    assert_raises(ValueError, format.read_array_header_1_0, s)

    # headers without the exact keys required should fail
    # d = {"shape": (1, 2),
    #      "descr": "x"}
    s = BytesIO(
        b"\x93NUMPY\x01\x006\x00{'descr': 'x', 'shape': (1, 2), }"
        b"                    \n"
    )
    assert_raises(ValueError, format.read_array_header_1_0, s)

    d = {"shape": (1, 2),
         "fortran_order": False,
         "descr": "x",
         "extrakey": -1}
    s = BytesIO()
    format.write_array_header_1_0(s, d)
    assert_raises(ValueError, format.read_array_header_1_0, s)


def test_large_file_support(tmpdir):
    if (sys.platform == 'win32' or sys.platform == 'cygwin'):
        pytest.skip("Unknown if Windows has sparse filesystems")
    # try creating a large sparse file
    tf_name = os.path.join(tmpdir, 'sparse_file')
    try:
        # seek past end would work too, but linux truncate somewhat
        # increases the chances that we have a sparse filesystem and can
        # avoid actually writing 5GB
        import subprocess as sp
        sp.check_call(["truncate", "-s", "5368709120", tf_name])
    except Exception:
        pytest.skip("Could not create 5GB large file")
    # write a small array to the end
    with open(tf_name, "wb") as f:
        f.seek(5368709120)
        d = np.arange(5)
        np.save(f, d)
    # read it back
    with open(tf_name, "rb") as f:
        f.seek(5368709120)
        r = np.load(f)
    assert_array_equal(r, d)


@pytest.mark.skipif(IS_PYPY, reason="flaky on PyPy")
@pytest.mark.skipif(not IS_64BIT, reason="test requires 64-bit system")
@pytest.mark.slow
@requires_memory(free_bytes=2 * 2**30)
def test_large_archive(tmpdir):
    # Regression test for product of saving arrays with dimensions of array
    # having a product that doesn't fit in int32.  See gh-7598 for details.
    shape = (2**30, 2)
    try:
        a = np.empty(shape, dtype=np.uint8)
    except MemoryError:
        pytest.skip("Could not create large file")

    fname = os.path.join(tmpdir, "large_archive")

    with open(fname, "wb") as f:
        np.savez(f, arr=a)

    del a

    with open(fname, "rb") as f:
        new_a = np.load(f)["arr"]

    assert new_a.shape == shape


def test_empty_npz(tmpdir):
    # Test for gh-9989
    fname = os.path.join(tmpdir, "nothing.npz")
    np.savez(fname)
    with np.load(fname) as nps:
        pass


def test_unicode_field_names(tmpdir):
    # gh-7391
    arr = np.array([
        (1, 3),
        (1, 2),
        (1, 3),
        (1, 2)
    ], dtype=[
        ('int', int),
        ('\N{CJK UNIFIED IDEOGRAPH-6574}\N{CJK UNIFIED IDEOGRAPH-5F62}', int)
    ])
    fname = os.path.join(tmpdir, "unicode.npy")
    with open(fname, 'wb') as f:
        format.write_array(f, arr, version=(3, 0))
    with open(fname, 'rb') as f:
        arr2 = format.read_array(f)
    assert_array_equal(arr, arr2)

    # notifies the user that 3.0 is selected
    with open(fname, 'wb') as f:
        with assert_warns(UserWarning):
            format.write_array(f, arr, version=None)

def test_header_growth_axis():
    for is_fortran_array, dtype_space, expected_header_length in [
        [False, 22, 128], [False, 23, 192], [True, 23, 128], [True, 24, 192]
    ]:
        for size in [10**i for i in range(format.GROWTH_AXIS_MAX_DIGITS)]:
            fp = BytesIO()
            format.write_array_header_1_0(fp, {
                'shape': (2, size) if is_fortran_array else (size, 2),
                'fortran_order': is_fortran_array,
                'descr': np.dtype([(' ' * dtype_space, int)])
            })

            assert len(fp.getvalue()) == expected_header_length

@pytest.mark.parametrize('dt', [
    np.dtype({'names': ['a', 'b'], 'formats':  [float, np.dtype('S3',
                 metadata={'some': 'stuff'})]}),
    np.dtype(int, metadata={'some': 'stuff'}),
    np.dtype([('subarray', (int, (2,)))], metadata={'some': 'stuff'}),
    # recursive: metadata on the field of a dtype
    np.dtype({'names': ['a', 'b'], 'formats': [
        float, np.dtype({'names': ['c'], 'formats': [np.dtype(int, metadata={})]})
    ]}),
    ])
@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
        reason="PyPy bug in error formatting")
def test_metadata_dtype(dt):
    # gh-14142
    arr = np.ones(10, dtype=dt)
    buf = BytesIO()
    with assert_warns(UserWarning):
        np.save(buf, arr)
    buf.seek(0)

    # Loading should work (metadata was stripped):
    arr2 = np.load(buf)
    # BUG: assert_array_equal does not check metadata
    from numpy.lib._utils_impl import drop_metadata
    assert_array_equal(arr, arr2)
    assert drop_metadata(arr.dtype) is not arr.dtype
    assert drop_metadata(arr2.dtype) is arr2.dtype

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\tests\test_recfunctions.py ===

import numpy as np
import numpy.ma as ma
from numpy.lib.recfunctions import (
    append_fields,
    apply_along_fields,
    assign_fields_by_name,
    drop_fields,
    find_duplicates,
    get_fieldstructure,
    join_by,
    merge_arrays,
    recursive_fill_fields,
    rename_fields,
    repack_fields,
    require_fields,
    stack_arrays,
    structured_to_unstructured,
    unstructured_to_structured,
)
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises

get_fieldspec = np.lib.recfunctions._get_fieldspec
get_names = np.lib.recfunctions.get_names
get_names_flat = np.lib.recfunctions.get_names_flat
zip_descr = np.lib.recfunctions._zip_descr
zip_dtype = np.lib.recfunctions._zip_dtype


class TestRecFunctions:
    # Misc tests

    def setup_method(self):
        x = np.array([1, 2, ])
        y = np.array([10, 20, 30])
        z = np.array([('A', 1.), ('B', 2.)],
                     dtype=[('A', '|S3'), ('B', float)])
        w = np.array([(1, (2, 3.0)), (4, (5, 6.0))],
                     dtype=[('a', int), ('b', [('ba', float), ('bb', int)])])
        self.data = (w, x, y, z)

    def test_zip_descr(self):
        # Test zip_descr
        (w, x, y, z) = self.data

        # Std array
        test = zip_descr((x, x), flatten=True)
        assert_equal(test,
                     np.dtype([('', int), ('', int)]))
        test = zip_descr((x, x), flatten=False)
        assert_equal(test,
                     np.dtype([('', int), ('', int)]))

        # Std & flexible-dtype
        test = zip_descr((x, z), flatten=True)
        assert_equal(test,
                     np.dtype([('', int), ('A', '|S3'), ('B', float)]))
        test = zip_descr((x, z), flatten=False)
        assert_equal(test,
                     np.dtype([('', int),
                               ('', [('A', '|S3'), ('B', float)])]))

        # Standard & nested dtype
        test = zip_descr((x, w), flatten=True)
        assert_equal(test,
                     np.dtype([('', int),
                               ('a', int),
                               ('ba', float), ('bb', int)]))
        test = zip_descr((x, w), flatten=False)
        assert_equal(test,
                     np.dtype([('', int),
                               ('', [('a', int),
                                     ('b', [('ba', float), ('bb', int)])])]))

    def test_drop_fields(self):
        # Test drop_fields
        a = np.array([(1, (2, 3.0)), (4, (5, 6.0))],
                     dtype=[('a', int), ('b', [('ba', float), ('bb', int)])])

        # A basic field
        test = drop_fields(a, 'a')
        control = np.array([((2, 3.0),), ((5, 6.0),)],
                           dtype=[('b', [('ba', float), ('bb', int)])])
        assert_equal(test, control)

        # Another basic field (but nesting two fields)
        test = drop_fields(a, 'b')
        control = np.array([(1,), (4,)], dtype=[('a', int)])
        assert_equal(test, control)

        # A nested sub-field
        test = drop_fields(a, ['ba', ])
        control = np.array([(1, (3.0,)), (4, (6.0,))],
                           dtype=[('a', int), ('b', [('bb', int)])])
        assert_equal(test, control)

        # All the nested sub-field from a field: zap that field
        test = drop_fields(a, ['ba', 'bb'])
        control = np.array([(1,), (4,)], dtype=[('a', int)])
        assert_equal(test, control)

        # dropping all fields results in an array with no fields
        test = drop_fields(a, ['a', 'b'])
        control = np.array([(), ()], dtype=[])
        assert_equal(test, control)

    def test_rename_fields(self):
        # Test rename fields
        a = np.array([(1, (2, [3.0, 30.])), (4, (5, [6.0, 60.]))],
                     dtype=[('a', int),
                            ('b', [('ba', float), ('bb', (float, 2))])])
        test = rename_fields(a, {'a': 'A', 'bb': 'BB'})
        newdtype = [('A', int), ('b', [('ba', float), ('BB', (float, 2))])]
        control = a.view(newdtype)
        assert_equal(test.dtype, newdtype)
        assert_equal(test, control)

    def test_get_names(self):
        # Test get_names
        ndtype = np.dtype([('A', '|S3'), ('B', float)])
        test = get_names(ndtype)
        assert_equal(test, ('A', 'B'))

        ndtype = np.dtype([('a', int), ('b', [('ba', float), ('bb', int)])])
        test = get_names(ndtype)
        assert_equal(test, ('a', ('b', ('ba', 'bb'))))

        ndtype = np.dtype([('a', int), ('b', [])])
        test = get_names(ndtype)
        assert_equal(test, ('a', ('b', ())))

        ndtype = np.dtype([])
        test = get_names(ndtype)
        assert_equal(test, ())

    def test_get_names_flat(self):
        # Test get_names_flat
        ndtype = np.dtype([('A', '|S3'), ('B', float)])
        test = get_names_flat(ndtype)
        assert_equal(test, ('A', 'B'))

        ndtype = np.dtype([('a', int), ('b', [('ba', float), ('bb', int)])])
        test = get_names_flat(ndtype)
        assert_equal(test, ('a', 'b', 'ba', 'bb'))

        ndtype = np.dtype([('a', int), ('b', [])])
        test = get_names_flat(ndtype)
        assert_equal(test, ('a', 'b'))

        ndtype = np.dtype([])
        test = get_names_flat(ndtype)
        assert_equal(test, ())

    def test_get_fieldstructure(self):
        # Test get_fieldstructure

        # No nested fields
        ndtype = np.dtype([('A', '|S3'), ('B', float)])
        test = get_fieldstructure(ndtype)
        assert_equal(test, {'A': [], 'B': []})

        # One 1-nested field
        ndtype = np.dtype([('A', int), ('B', [('BA', float), ('BB', '|S1')])])
        test = get_fieldstructure(ndtype)
        assert_equal(test, {'A': [], 'B': [], 'BA': ['B', ], 'BB': ['B']})

        # One 2-nested fields
        ndtype = np.dtype([('A', int),
                           ('B', [('BA', int),
                                  ('BB', [('BBA', int), ('BBB', int)])])])
        test = get_fieldstructure(ndtype)
        control = {'A': [], 'B': [], 'BA': ['B'], 'BB': ['B'],
                   'BBA': ['B', 'BB'], 'BBB': ['B', 'BB']}
        assert_equal(test, control)

        # 0 fields
        ndtype = np.dtype([])
        test = get_fieldstructure(ndtype)
        assert_equal(test, {})

    def test_find_duplicates(self):
        # Test find_duplicates
        a = ma.array([(2, (2., 'B')), (1, (2., 'B')), (2, (2., 'B')),
                      (1, (1., 'B')), (2, (2., 'B')), (2, (2., 'C'))],
                     mask=[(0, (0, 0)), (0, (0, 0)), (0, (0, 0)),
                           (0, (0, 0)), (1, (0, 0)), (0, (1, 0))],
                     dtype=[('A', int), ('B', [('BA', float), ('BB', '|S1')])])

        test = find_duplicates(a, ignoremask=False, return_index=True)
        control = [0, 2]
        assert_equal(sorted(test[-1]), control)
        assert_equal(test[0], a[test[-1]])

        test = find_duplicates(a, key='A', return_index=True)
        control = [0, 1, 2, 3, 5]
        assert_equal(sorted(test[-1]), control)
        assert_equal(test[0], a[test[-1]])

        test = find_duplicates(a, key='B', return_index=True)
        control = [0, 1, 2, 4]
        assert_equal(sorted(test[-1]), control)
        assert_equal(test[0], a[test[-1]])

        test = find_duplicates(a, key='BA', return_index=True)
        control = [0, 1, 2, 4]
        assert_equal(sorted(test[-1]), control)
        assert_equal(test[0], a[test[-1]])

        test = find_duplicates(a, key='BB', return_index=True)
        control = [0, 1, 2, 3, 4]
        assert_equal(sorted(test[-1]), control)
        assert_equal(test[0], a[test[-1]])

    def test_find_duplicates_ignoremask(self):
        # Test the ignoremask option of find_duplicates
        ndtype = [('a', int)]
        a = ma.array([1, 1, 1, 2, 2, 3, 3],
                     mask=[0, 0, 1, 0, 0, 0, 1]).view(ndtype)
        test = find_duplicates(a, ignoremask=True, return_index=True)
        control = [0, 1, 3, 4]
        assert_equal(sorted(test[-1]), control)
        assert_equal(test[0], a[test[-1]])

        test = find_duplicates(a, ignoremask=False, return_index=True)
        control = [0, 1, 2, 3, 4, 6]
        assert_equal(sorted(test[-1]), control)
        assert_equal(test[0], a[test[-1]])

    def test_repack_fields(self):
        dt = np.dtype('u1,f4,i8', align=True)
        a = np.zeros(2, dtype=dt)

        assert_equal(repack_fields(dt), np.dtype('u1,f4,i8'))
        assert_equal(repack_fields(a).itemsize, 13)
        assert_equal(repack_fields(repack_fields(dt), align=True), dt)

        # make sure type is preserved
        dt = np.dtype((np.record, dt))
        assert_(repack_fields(dt).type is np.record)

    def test_structured_to_unstructured(self, tmp_path):
        a = np.zeros(4, dtype=[('a', 'i4'), ('b', 'f4,u2'), ('c', 'f4', 2)])
        out = structured_to_unstructured(a)
        assert_equal(out, np.zeros((4, 5), dtype='f8'))

        b = np.array([(1, 2, 5), (4, 5, 7), (7, 8, 11), (10, 11, 12)],
                     dtype=[('x', 'i4'), ('y', 'f4'), ('z', 'f8')])
        out = np.mean(structured_to_unstructured(b[['x', 'z']]), axis=-1)
        assert_equal(out, np.array([3.,  5.5,  9., 11.]))
        out = np.mean(structured_to_unstructured(b[['x']]), axis=-1)
        assert_equal(out, np.array([1.,  4. ,  7., 10.]))  # noqa: E203

        c = np.arange(20).reshape((4, 5))
        out = unstructured_to_structured(c, a.dtype)
        want = np.array([( 0, ( 1.,  2), [ 3.,  4.]),
                         ( 5, ( 6.,  7), [ 8.,  9.]),
                         (10, (11., 12), [13., 14.]),
                         (15, (16., 17), [18., 19.])],
                     dtype=[('a', 'i4'),
                            ('b', [('f0', 'f4'), ('f1', 'u2')]),
                            ('c', 'f4', (2,))])
        assert_equal(out, want)

        d = np.array([(1, 2, 5), (4, 5, 7), (7, 8, 11), (10, 11, 12)],
                     dtype=[('x', 'i4'), ('y', 'f4'), ('z', 'f8')])
        assert_equal(apply_along_fields(np.mean, d),
                     np.array([ 8.0 / 3,  16.0 / 3,  26.0 / 3, 11.]))
        assert_equal(apply_along_fields(np.mean, d[['x', 'z']]),
                     np.array([ 3.,  5.5,  9., 11.]))

        # check that for uniform field dtypes we get a view, not a copy:
        d = np.array([(1, 2, 5), (4, 5, 7), (7, 8, 11), (10, 11, 12)],
                     dtype=[('x', 'i4'), ('y', 'i4'), ('z', 'i4')])
        dd = structured_to_unstructured(d)
        ddd = unstructured_to_structured(dd, d.dtype)
        assert_(np.shares_memory(dd, d))
        assert_(np.shares_memory(ddd, d))

        # check that reversing the order of attributes works
        dd_attrib_rev = structured_to_unstructured(d[['z', 'x']])
        assert_equal(dd_attrib_rev, [[5, 1], [7, 4], [11, 7], [12, 10]])
        assert_(np.shares_memory(dd_attrib_rev, d))

        # including uniform fields with subarrays unpacked
        d = np.array([(1, [2,  3], [[ 4,  5], [ 6,  7]]),
                      (8, [9, 10], [[11, 12], [13, 14]])],
                     dtype=[('x0', 'i4'), ('x1', ('i4', 2)),
                            ('x2', ('i4', (2, 2)))])
        dd = structured_to_unstructured(d)
        ddd = unstructured_to_structured(dd, d.dtype)
        assert_(np.shares_memory(dd, d))
        assert_(np.shares_memory(ddd, d))

        # check that reversing with sub-arrays works as expected
        d_rev = d[::-1]
        dd_rev = structured_to_unstructured(d_rev)
        assert_equal(dd_rev, [[8, 9, 10, 11, 12, 13, 14],
                              [1, 2, 3, 4, 5, 6, 7]])

        # check that sub-arrays keep the order of their values
        d_attrib_rev = d[['x2', 'x1', 'x0']]
        dd_attrib_rev = structured_to_unstructured(d_attrib_rev)
        assert_equal(dd_attrib_rev, [[4, 5, 6, 7, 2, 3, 1],
                                     [11, 12, 13, 14, 9, 10, 8]])

        # with ignored field at the end
        d = np.array([(1, [2,  3], [[4, 5], [6, 7]], 32),
                      (8, [9, 10], [[11, 12], [13, 14]], 64)],
                     dtype=[('x0', 'i4'), ('x1', ('i4', 2)),
                            ('x2', ('i4', (2, 2))), ('ignored', 'u1')])
        dd = structured_to_unstructured(d[['x0', 'x1', 'x2']])
        assert_(np.shares_memory(dd, d))
        assert_equal(dd, [[1, 2, 3, 4, 5, 6, 7],
                          [8, 9, 10, 11, 12, 13, 14]])

        # test that nested fields with identical names don't break anything
        point = np.dtype([('x', int), ('y', int)])
        triangle = np.dtype([('a', point), ('b', point), ('c', point)])
        arr = np.zeros(10, triangle)
        res = structured_to_unstructured(arr, dtype=int)
        assert_equal(res, np.zeros((10, 6), dtype=int))

        # test nested combinations of subarrays and structured arrays, gh-13333
        def subarray(dt, shape):
            return np.dtype((dt, shape))

        def structured(*dts):
            return np.dtype([(f'x{i}', dt) for i, dt in enumerate(dts)])

        def inspect(dt, dtype=None):
            arr = np.zeros((), dt)
            ret = structured_to_unstructured(arr, dtype=dtype)
            backarr = unstructured_to_structured(ret, dt)
            return ret.shape, ret.dtype, backarr.dtype

        dt = structured(subarray(structured(np.int32, np.int32), 3))
        assert_equal(inspect(dt), ((6,), np.int32, dt))

        dt = structured(subarray(subarray(np.int32, 2), 2))
        assert_equal(inspect(dt), ((4,), np.int32, dt))

        dt = structured(np.int32)
        assert_equal(inspect(dt), ((1,), np.int32, dt))

        dt = structured(np.int32, subarray(subarray(np.int32, 2), 2))
        assert_equal(inspect(dt), ((5,), np.int32, dt))

        dt = structured()
        assert_raises(ValueError, structured_to_unstructured, np.zeros(3, dt))

        # these currently don't work, but we may make it work in the future
        assert_raises(NotImplementedError, structured_to_unstructured,
                                           np.zeros(3, dt), dtype=np.int32)
        assert_raises(NotImplementedError, unstructured_to_structured,
                                           np.zeros((3, 0), dtype=np.int32))

        # test supported ndarray subclasses
        d_plain = np.array([(1, 2), (3, 4)], dtype=[('a', 'i4'), ('b', 'i4')])
        dd_expected = structured_to_unstructured(d_plain, copy=True)

        # recarray
        d = d_plain.view(np.recarray)

        dd = structured_to_unstructured(d, copy=False)
        ddd = structured_to_unstructured(d, copy=True)
        assert_(np.shares_memory(d, dd))
        assert_(type(dd) is np.recarray)
        assert_(type(ddd) is np.recarray)
        assert_equal(dd, dd_expected)
        assert_equal(ddd, dd_expected)

        # memmap
        d = np.memmap(tmp_path / 'memmap',
                      mode='w+',
                      dtype=d_plain.dtype,
                      shape=d_plain.shape)
        d[:] = d_plain
        dd = structured_to_unstructured(d, copy=False)
        ddd = structured_to_unstructured(d, copy=True)
        assert_(np.shares_memory(d, dd))
        assert_(type(dd) is np.memmap)
        assert_(type(ddd) is np.memmap)
        assert_equal(dd, dd_expected)
        assert_equal(ddd, dd_expected)

    def test_unstructured_to_structured(self):
        # test if dtype is the args of np.dtype
        a = np.zeros((20, 2))
        test_dtype_args = [('x', float), ('y', float)]
        test_dtype = np.dtype(test_dtype_args)
        field1 = unstructured_to_structured(a, dtype=test_dtype_args)  # now
        field2 = unstructured_to_structured(a, dtype=test_dtype)  # before
        assert_equal(field1, field2)

    def test_field_assignment_by_name(self):
        a = np.ones(2, dtype=[('a', 'i4'), ('b', 'f8'), ('c', 'u1')])
        newdt = [('b', 'f4'), ('c', 'u1')]

        assert_equal(require_fields(a, newdt), np.ones(2, newdt))

        b = np.array([(1, 2), (3, 4)], dtype=newdt)
        assign_fields_by_name(a, b, zero_unassigned=False)
        assert_equal(a, np.array([(1, 1, 2), (1, 3, 4)], dtype=a.dtype))
        assign_fields_by_name(a, b)
        assert_equal(a, np.array([(0, 1, 2), (0, 3, 4)], dtype=a.dtype))

        # test nested fields
        a = np.ones(2, dtype=[('a', [('b', 'f8'), ('c', 'u1')])])
        newdt = [('a', [('c', 'u1')])]
        assert_equal(require_fields(a, newdt), np.ones(2, newdt))
        b = np.array([((2,),), ((3,),)], dtype=newdt)
        assign_fields_by_name(a, b, zero_unassigned=False)
        assert_equal(a, np.array([((1, 2),), ((1, 3),)], dtype=a.dtype))
        assign_fields_by_name(a, b)
        assert_equal(a, np.array([((0, 2),), ((0, 3),)], dtype=a.dtype))

        # test unstructured code path for 0d arrays
        a, b = np.array(3), np.array(0)
        assign_fields_by_name(b, a)
        assert_equal(b[()], 3)


class TestRecursiveFillFields:
    # Test recursive_fill_fields.
    def test_simple_flexible(self):
        # Test recursive_fill_fields on flexible-array
        a = np.array([(1, 10.), (2, 20.)], dtype=[('A', int), ('B', float)])
        b = np.zeros((3,), dtype=a.dtype)
        test = recursive_fill_fields(a, b)
        control = np.array([(1, 10.), (2, 20.), (0, 0.)],
                           dtype=[('A', int), ('B', float)])
        assert_equal(test, control)

    def test_masked_flexible(self):
        # Test recursive_fill_fields on masked flexible-array
        a = ma.array([(1, 10.), (2, 20.)], mask=[(0, 1), (1, 0)],
                     dtype=[('A', int), ('B', float)])
        b = ma.zeros((3,), dtype=a.dtype)
        test = recursive_fill_fields(a, b)
        control = ma.array([(1, 10.), (2, 20.), (0, 0.)],
                           mask=[(0, 1), (1, 0), (0, 0)],
                           dtype=[('A', int), ('B', float)])
        assert_equal(test, control)


class TestMergeArrays:
    # Test merge_arrays

    def setup_method(self):
        x = np.array([1, 2, ])
        y = np.array([10, 20, 30])
        z = np.array(
            [('A', 1.), ('B', 2.)], dtype=[('A', '|S3'), ('B', float)])
        w = np.array(
            [(1, (2, 3.0, ())), (4, (5, 6.0, ()))],
            dtype=[('a', int), ('b', [('ba', float), ('bb', int), ('bc', [])])])
        self.data = (w, x, y, z)

    def test_solo(self):
        # Test merge_arrays on a single array.
        (_, x, _, z) = self.data

        test = merge_arrays(x)
        control = np.array([(1,), (2,)], dtype=[('f0', int)])
        assert_equal(test, control)
        test = merge_arrays((x,))
        assert_equal(test, control)

        test = merge_arrays(z, flatten=False)
        assert_equal(test, z)
        test = merge_arrays(z, flatten=True)
        assert_equal(test, z)

    def test_solo_w_flatten(self):
        # Test merge_arrays on a single array w & w/o flattening
        w = self.data[0]
        test = merge_arrays(w, flatten=False)
        assert_equal(test, w)

        test = merge_arrays(w, flatten=True)
        control = np.array([(1, 2, 3.0), (4, 5, 6.0)],
                           dtype=[('a', int), ('ba', float), ('bb', int)])
        assert_equal(test, control)

    def test_standard(self):
        # Test standard & standard
        # Test merge arrays
        (_, x, y, _) = self.data
        test = merge_arrays((x, y), usemask=False)
        control = np.array([(1, 10), (2, 20), (-1, 30)],
                           dtype=[('f0', int), ('f1', int)])
        assert_equal(test, control)

        test = merge_arrays((x, y), usemask=True)
        control = ma.array([(1, 10), (2, 20), (-1, 30)],
                           mask=[(0, 0), (0, 0), (1, 0)],
                           dtype=[('f0', int), ('f1', int)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)

    def test_flatten(self):
        # Test standard & flexible
        (_, x, _, z) = self.data
        test = merge_arrays((x, z), flatten=True)
        control = np.array([(1, 'A', 1.), (2, 'B', 2.)],
                           dtype=[('f0', int), ('A', '|S3'), ('B', float)])
        assert_equal(test, control)

        test = merge_arrays((x, z), flatten=False)
        control = np.array([(1, ('A', 1.)), (2, ('B', 2.))],
                           dtype=[('f0', int),
                                  ('f1', [('A', '|S3'), ('B', float)])])
        assert_equal(test, control)

    def test_flatten_wflexible(self):
        # Test flatten standard & nested
        (w, x, _, _) = self.data
        test = merge_arrays((x, w), flatten=True)
        control = np.array([(1, 1, 2, 3.0), (2, 4, 5, 6.0)],
                           dtype=[('f0', int),
                                  ('a', int), ('ba', float), ('bb', int)])
        assert_equal(test, control)

        test = merge_arrays((x, w), flatten=False)
        controldtype = [('f0', int),
                                ('f1', [('a', int),
                                        ('b', [('ba', float), ('bb', int), ('bc', [])])])]
        control = np.array([(1., (1, (2, 3.0, ()))), (2, (4, (5, 6.0, ())))],
                           dtype=controldtype)
        assert_equal(test, control)

    def test_wmasked_arrays(self):
        # Test merge_arrays masked arrays
        (_, x, _, _) = self.data
        mx = ma.array([1, 2, 3], mask=[1, 0, 0])
        test = merge_arrays((x, mx), usemask=True)
        control = ma.array([(1, 1), (2, 2), (-1, 3)],
                           mask=[(0, 1), (0, 0), (1, 0)],
                           dtype=[('f0', int), ('f1', int)])
        assert_equal(test, control)
        test = merge_arrays((x, mx), usemask=True, asrecarray=True)
        assert_equal(test, control)
        assert_(isinstance(test, MaskedRecords))

    def test_w_singlefield(self):
        # Test single field
        test = merge_arrays((np.array([1, 2]).view([('a', int)]),
                             np.array([10., 20., 30.])),)
        control = ma.array([(1, 10.), (2, 20.), (-1, 30.)],
                           mask=[(0, 0), (0, 0), (1, 0)],
                           dtype=[('a', int), ('f1', float)])
        assert_equal(test, control)

    def test_w_shorter_flex(self):
        # Test merge_arrays w/ a shorter flexndarray.
        z = self.data[-1]

        # Fixme, this test looks incomplete and broken
        #test = merge_arrays((z, np.array([10, 20, 30]).view([('C', int)])))
        #control = np.array([('A', 1., 10), ('B', 2., 20), ('-1', -1, 20)],
        #                   dtype=[('A', '|S3'), ('B', float), ('C', int)])
        #assert_equal(test, control)

        merge_arrays((z, np.array([10, 20, 30]).view([('C', int)])))
        np.array([('A', 1., 10), ('B', 2., 20), ('-1', -1, 20)],
                 dtype=[('A', '|S3'), ('B', float), ('C', int)])

    def test_singlerecord(self):
        (_, x, y, z) = self.data
        test = merge_arrays((x[0], y[0], z[0]), usemask=False)
        control = np.array([(1, 10, ('A', 1))],
                           dtype=[('f0', int),
                                  ('f1', int),
                                  ('f2', [('A', '|S3'), ('B', float)])])
        assert_equal(test, control)


class TestAppendFields:
    # Test append_fields

    def setup_method(self):
        x = np.array([1, 2, ])
        y = np.array([10, 20, 30])
        z = np.array(
            [('A', 1.), ('B', 2.)], dtype=[('A', '|S3'), ('B', float)])
        w = np.array([(1, (2, 3.0)), (4, (5, 6.0))],
                     dtype=[('a', int), ('b', [('ba', float), ('bb', int)])])
        self.data = (w, x, y, z)

    def test_append_single(self):
        # Test simple case
        (_, x, _, _) = self.data
        test = append_fields(x, 'A', data=[10, 20, 30])
        control = ma.array([(1, 10), (2, 20), (-1, 30)],
                           mask=[(0, 0), (0, 0), (1, 0)],
                           dtype=[('f0', int), ('A', int)],)
        assert_equal(test, control)

    def test_append_double(self):
        # Test simple case
        (_, x, _, _) = self.data
        test = append_fields(x, ('A', 'B'), data=[[10, 20, 30], [100, 200]])
        control = ma.array([(1, 10, 100), (2, 20, 200), (-1, 30, -1)],
                           mask=[(0, 0, 0), (0, 0, 0), (1, 0, 1)],
                           dtype=[('f0', int), ('A', int), ('B', int)],)
        assert_equal(test, control)

    def test_append_on_flex(self):
        # Test append_fields on flexible type arrays
        z = self.data[-1]
        test = append_fields(z, 'C', data=[10, 20, 30])
        control = ma.array([('A', 1., 10), ('B', 2., 20), (-1, -1., 30)],
                           mask=[(0, 0, 0), (0, 0, 0), (1, 1, 0)],
                           dtype=[('A', '|S3'), ('B', float), ('C', int)],)
        assert_equal(test, control)

    def test_append_on_nested(self):
        # Test append_fields on nested fields
        w = self.data[0]
        test = append_fields(w, 'C', data=[10, 20, 30])
        control = ma.array([(1, (2, 3.0), 10),
                            (4, (5, 6.0), 20),
                            (-1, (-1, -1.), 30)],
                           mask=[(
                               0, (0, 0), 0), (0, (0, 0), 0), (1, (1, 1), 0)],
                           dtype=[('a', int),
                                  ('b', [('ba', float), ('bb', int)]),
                                  ('C', int)],)
        assert_equal(test, control)


class TestStackArrays:
    # Test stack_arrays
    def setup_method(self):
        x = np.array([1, 2, ])
        y = np.array([10, 20, 30])
        z = np.array(
            [('A', 1.), ('B', 2.)], dtype=[('A', '|S3'), ('B', float)])
        w = np.array([(1, (2, 3.0)), (4, (5, 6.0))],
                     dtype=[('a', int), ('b', [('ba', float), ('bb', int)])])
        self.data = (w, x, y, z)

    def test_solo(self):
        # Test stack_arrays on single arrays
        (_, x, _, _) = self.data
        test = stack_arrays((x,))
        assert_equal(test, x)
        assert_(test is x)

        test = stack_arrays(x)
        assert_equal(test, x)
        assert_(test is x)

    def test_unnamed_fields(self):
        # Tests combinations of arrays w/o named fields
        (_, x, y, _) = self.data

        test = stack_arrays((x, x), usemask=False)
        control = np.array([1, 2, 1, 2])
        assert_equal(test, control)

        test = stack_arrays((x, y), usemask=False)
        control = np.array([1, 2, 10, 20, 30])
        assert_equal(test, control)

        test = stack_arrays((y, x), usemask=False)
        control = np.array([10, 20, 30, 1, 2])
        assert_equal(test, control)

    def test_unnamed_and_named_fields(self):
        # Test combination of arrays w/ & w/o named fields
        (_, x, _, z) = self.data

        test = stack_arrays((x, z))
        control = ma.array([(1, -1, -1), (2, -1, -1),
                            (-1, 'A', 1), (-1, 'B', 2)],
                           mask=[(0, 1, 1), (0, 1, 1),
                                 (1, 0, 0), (1, 0, 0)],
                           dtype=[('f0', int), ('A', '|S3'), ('B', float)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)

        test = stack_arrays((z, x))
        control = ma.array([('A', 1, -1), ('B', 2, -1),
                            (-1, -1, 1), (-1, -1, 2), ],
                           mask=[(0, 0, 1), (0, 0, 1),
                                 (1, 1, 0), (1, 1, 0)],
                           dtype=[('A', '|S3'), ('B', float), ('f2', int)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)

        test = stack_arrays((z, z, x))
        control = ma.array([('A', 1, -1), ('B', 2, -1),
                            ('A', 1, -1), ('B', 2, -1),
                            (-1, -1, 1), (-1, -1, 2), ],
                           mask=[(0, 0, 1), (0, 0, 1),
                                 (0, 0, 1), (0, 0, 1),
                                 (1, 1, 0), (1, 1, 0)],
                           dtype=[('A', '|S3'), ('B', float), ('f2', int)])
        assert_equal(test, control)

    def test_matching_named_fields(self):
        # Test combination of arrays w/ matching field names
        (_, x, _, z) = self.data
        zz = np.array([('a', 10., 100.), ('b', 20., 200.), ('c', 30., 300.)],
                      dtype=[('A', '|S3'), ('B', float), ('C', float)])
        test = stack_arrays((z, zz))
        control = ma.array([('A', 1, -1), ('B', 2, -1),
                            (
                                'a', 10., 100.), ('b', 20., 200.), ('c', 30., 300.)],
                           dtype=[('A', '|S3'), ('B', float), ('C', float)],
                           mask=[(0, 0, 1), (0, 0, 1),
                                 (0, 0, 0), (0, 0, 0), (0, 0, 0)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)

        test = stack_arrays((z, zz, x))
        ndtype = [('A', '|S3'), ('B', float), ('C', float), ('f3', int)]
        control = ma.array([('A', 1, -1, -1), ('B', 2, -1, -1),
                            ('a', 10., 100., -1), ('b', 20., 200., -1),
                            ('c', 30., 300., -1),
                            (-1, -1, -1, 1), (-1, -1, -1, 2)],
                           dtype=ndtype,
                           mask=[(0, 0, 1, 1), (0, 0, 1, 1),
                                 (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1),
                                 (1, 1, 1, 0), (1, 1, 1, 0)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)

    def test_defaults(self):
        # Test defaults: no exception raised if keys of defaults are not fields.
        (_, _, _, z) = self.data
        zz = np.array([('a', 10., 100.), ('b', 20., 200.), ('c', 30., 300.)],
                      dtype=[('A', '|S3'), ('B', float), ('C', float)])
        defaults = {'A': '???', 'B': -999., 'C': -9999., 'D': -99999.}
        test = stack_arrays((z, zz), defaults=defaults)
        control = ma.array([('A', 1, -9999.), ('B', 2, -9999.),
                            (
                                'a', 10., 100.), ('b', 20., 200.), ('c', 30., 300.)],
                           dtype=[('A', '|S3'), ('B', float), ('C', float)],
                           mask=[(0, 0, 1), (0, 0, 1),
                                 (0, 0, 0), (0, 0, 0), (0, 0, 0)])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)

    def test_autoconversion(self):
        # Tests autoconversion
        adtype = [('A', int), ('B', bool), ('C', float)]
        a = ma.array([(1, 2, 3)], mask=[(0, 1, 0)], dtype=adtype)
        bdtype = [('A', int), ('B', float), ('C', float)]
        b = ma.array([(4, 5, 6)], dtype=bdtype)
        control = ma.array([(1, 2, 3), (4, 5, 6)], mask=[(0, 1, 0), (0, 0, 0)],
                           dtype=bdtype)
        test = stack_arrays((a, b), autoconvert=True)
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        with assert_raises(TypeError):
            stack_arrays((a, b), autoconvert=False)

    def test_checktitles(self):
        # Test using titles in the field names
        adtype = [(('a', 'A'), int), (('b', 'B'), bool), (('c', 'C'), float)]
        a = ma.array([(1, 2, 3)], mask=[(0, 1, 0)], dtype=adtype)
        bdtype = [(('a', 'A'), int), (('b', 'B'), bool), (('c', 'C'), float)]
        b = ma.array([(4, 5, 6)], dtype=bdtype)
        test = stack_arrays((a, b))
        control = ma.array([(1, 2, 3), (4, 5, 6)], mask=[(0, 1, 0), (0, 0, 0)],
                           dtype=bdtype)
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)

    def test_subdtype(self):
        z = np.array([
            ('A', 1), ('B', 2)
        ], dtype=[('A', '|S3'), ('B', float, (1,))])
        zz = np.array([
            ('a', [10.], 100.), ('b', [20.], 200.), ('c', [30.], 300.)
        ], dtype=[('A', '|S3'), ('B', float, (1,)), ('C', float)])

        res = stack_arrays((z, zz))
        expected = ma.array(
            data=[
                (b'A', [1.0], 0),
                (b'B', [2.0], 0),
                (b'a', [10.0], 100.0),
                (b'b', [20.0], 200.0),
                (b'c', [30.0], 300.0)],
            mask=[
                (False, [False], True),
                (False, [False], True),
                (False, [False], False),
                (False, [False], False),
                (False, [False], False)
            ],
            dtype=zz.dtype
        )
        assert_equal(res.dtype, expected.dtype)
        assert_equal(res, expected)
        assert_equal(res.mask, expected.mask)


class TestJoinBy:
    def setup_method(self):
        self.a = np.array(list(zip(np.arange(10), np.arange(50, 60),
                                   np.arange(100, 110))),
                          dtype=[('a', int), ('b', int), ('c', int)])
        self.b = np.array(list(zip(np.arange(5, 15), np.arange(65, 75),
                                   np.arange(100, 110))),
                          dtype=[('a', int), ('b', int), ('d', int)])

    def test_inner_join(self):
        # Basic test of join_by
        a, b = self.a, self.b

        test = join_by('a', a, b, jointype='inner')
        control = np.array([(5, 55, 65, 105, 100), (6, 56, 66, 106, 101),
                            (7, 57, 67, 107, 102), (8, 58, 68, 108, 103),
                            (9, 59, 69, 109, 104)],
                           dtype=[('a', int), ('b1', int), ('b2', int),
                                  ('c', int), ('d', int)])
        assert_equal(test, control)

    def test_join(self):
        a, b = self.a, self.b

        # Fixme, this test is broken
        #test = join_by(('a', 'b'), a, b)
        #control = np.array([(5, 55, 105, 100), (6, 56, 106, 101),
        #                    (7, 57, 107, 102), (8, 58, 108, 103),
        #                    (9, 59, 109, 104)],
        #                   dtype=[('a', int), ('b', int),
        #                          ('c', int), ('d', int)])
        #assert_equal(test, control)

        join_by(('a', 'b'), a, b)
        np.array([(5, 55, 105, 100), (6, 56, 106, 101),
                  (7, 57, 107, 102), (8, 58, 108, 103),
                  (9, 59, 109, 104)],
                  dtype=[('a', int), ('b', int),
                         ('c', int), ('d', int)])

    def test_join_subdtype(self):
        # tests the bug in https://stackoverflow.com/q/44769632/102441
        foo = np.array([(1,)],
                       dtype=[('key', int)])
        bar = np.array([(1, np.array([1, 2, 3]))],
                       dtype=[('key', int), ('value', 'uint16', 3)])
        res = join_by('key', foo, bar)
        assert_equal(res, bar.view(ma.MaskedArray))

    def test_outer_join(self):
        a, b = self.a, self.b

        test = join_by(('a', 'b'), a, b, 'outer')
        control = ma.array([(0, 50, 100, -1), (1, 51, 101, -1),
                            (2, 52, 102, -1), (3, 53, 103, -1),
                            (4, 54, 104, -1), (5, 55, 105, -1),
                            (5, 65, -1, 100), (6, 56, 106, -1),
                            (6, 66, -1, 101), (7, 57, 107, -1),
                            (7, 67, -1, 102), (8, 58, 108, -1),
                            (8, 68, -1, 103), (9, 59, 109, -1),
                            (9, 69, -1, 104), (10, 70, -1, 105),
                            (11, 71, -1, 106), (12, 72, -1, 107),
                            (13, 73, -1, 108), (14, 74, -1, 109)],
                           mask=[(0, 0, 0, 1), (0, 0, 0, 1),
                                 (0, 0, 0, 1), (0, 0, 0, 1),
                                 (0, 0, 0, 1), (0, 0, 0, 1),
                                 (0, 0, 1, 0), (0, 0, 0, 1),
                                 (0, 0, 1, 0), (0, 0, 0, 1),
                                 (0, 0, 1, 0), (0, 0, 0, 1),
                                 (0, 0, 1, 0), (0, 0, 0, 1),
                                 (0, 0, 1, 0), (0, 0, 1, 0),
                                 (0, 0, 1, 0), (0, 0, 1, 0),
                                 (0, 0, 1, 0), (0, 0, 1, 0)],
                           dtype=[('a', int), ('b', int),
                                  ('c', int), ('d', int)])
        assert_equal(test, control)

    def test_leftouter_join(self):
        a, b = self.a, self.b

        test = join_by(('a', 'b'), a, b, 'leftouter')
        control = ma.array([(0, 50, 100, -1), (1, 51, 101, -1),
                            (2, 52, 102, -1), (3, 53, 103, -1),
                            (4, 54, 104, -1), (5, 55, 105, -1),
                            (6, 56, 106, -1), (7, 57, 107, -1),
                            (8, 58, 108, -1), (9, 59, 109, -1)],
                           mask=[(0, 0, 0, 1), (0, 0, 0, 1),
                                 (0, 0, 0, 1), (0, 0, 0, 1),
                                 (0, 0, 0, 1), (0, 0, 0, 1),
                                 (0, 0, 0, 1), (0, 0, 0, 1),
                                 (0, 0, 0, 1), (0, 0, 0, 1)],
                           dtype=[('a', int), ('b', int), ('c', int), ('d', int)])
        assert_equal(test, control)

    def test_different_field_order(self):
        # gh-8940
        a = np.zeros(3, dtype=[('a', 'i4'), ('b', 'f4'), ('c', 'u1')])
        b = np.ones(3, dtype=[('c', 'u1'), ('b', 'f4'), ('a', 'i4')])
        # this should not give a FutureWarning:
        j = join_by(['c', 'b'], a, b, jointype='inner', usemask=False)
        assert_equal(j.dtype.names, ['b', 'c', 'a1', 'a2'])

    def test_duplicate_keys(self):
        a = np.zeros(3, dtype=[('a', 'i4'), ('b', 'f4'), ('c', 'u1')])
        b = np.ones(3, dtype=[('c', 'u1'), ('b', 'f4'), ('a', 'i4')])
        assert_raises(ValueError, join_by, ['a', 'b', 'b'], a, b)

    def test_same_name_different_dtypes_key(self):
        a_dtype = np.dtype([('key', 'S5'), ('value', '<f4')])
        b_dtype = np.dtype([('key', 'S10'), ('value', '<f4')])
        expected_dtype = np.dtype([
            ('key', 'S10'), ('value1', '<f4'), ('value2', '<f4')])

        a = np.array([('Sarah',  8.0), ('John', 6.0)], dtype=a_dtype)
        b = np.array([('Sarah', 10.0), ('John', 7.0)], dtype=b_dtype)
        res = join_by('key', a, b)

        assert_equal(res.dtype, expected_dtype)

    def test_same_name_different_dtypes(self):
        # gh-9338
        a_dtype = np.dtype([('key', 'S10'), ('value', '<f4')])
        b_dtype = np.dtype([('key', 'S10'), ('value', '<f8')])
        expected_dtype = np.dtype([
            ('key', '|S10'), ('value1', '<f4'), ('value2', '<f8')])

        a = np.array([('Sarah',  8.0), ('John', 6.0)], dtype=a_dtype)
        b = np.array([('Sarah', 10.0), ('John', 7.0)], dtype=b_dtype)
        res = join_by('key', a, b)

        assert_equal(res.dtype, expected_dtype)

    def test_subarray_key(self):
        a_dtype = np.dtype([('pos', int, 3), ('f', '<f4')])
        a = np.array([([1, 1, 1], np.pi), ([1, 2, 3], 0.0)], dtype=a_dtype)

        b_dtype = np.dtype([('pos', int, 3), ('g', '<f4')])
        b = np.array([([1, 1, 1], 3), ([3, 2, 1], 0.0)], dtype=b_dtype)

        expected_dtype = np.dtype([('pos', int, 3), ('f', '<f4'), ('g', '<f4')])
        expected = np.array([([1, 1, 1], np.pi, 3)], dtype=expected_dtype)

        res = join_by('pos', a, b)
        assert_equal(res.dtype, expected_dtype)
        assert_equal(res, expected)

    def test_padded_dtype(self):
        dt = np.dtype('i1,f4', align=True)
        dt.names = ('k', 'v')
        assert_(len(dt.descr), 3)  # padding field is inserted

        a = np.array([(1, 3), (3, 2)], dt)
        b = np.array([(1, 1), (2, 2)], dt)
        res = join_by('k', a, b)

        # no padding fields remain
        expected_dtype = np.dtype([
            ('k', 'i1'), ('v1', 'f4'), ('v2', 'f4')
        ])

        assert_equal(res.dtype, expected_dtype)


class TestJoinBy2:
    @classmethod
    def setup_method(cls):
        cls.a = np.array(list(zip(np.arange(10), np.arange(50, 60),
                                  np.arange(100, 110))),
                         dtype=[('a', int), ('b', int), ('c', int)])
        cls.b = np.array(list(zip(np.arange(10), np.arange(65, 75),
                                  np.arange(100, 110))),
                         dtype=[('a', int), ('b', int), ('d', int)])

    def test_no_r1postfix(self):
        # Basic test of join_by no_r1postfix
        a, b = self.a, self.b

        test = join_by(
            'a', a, b, r1postfix='', r2postfix='2', jointype='inner')
        control = np.array([(0, 50, 65, 100, 100), (1, 51, 66, 101, 101),
                            (2, 52, 67, 102, 102), (3, 53, 68, 103, 103),
                            (4, 54, 69, 104, 104), (5, 55, 70, 105, 105),
                            (6, 56, 71, 106, 106), (7, 57, 72, 107, 107),
                            (8, 58, 73, 108, 108), (9, 59, 74, 109, 109)],
                           dtype=[('a', int), ('b', int), ('b2', int),
                                  ('c', int), ('d', int)])
        assert_equal(test, control)

    def test_no_postfix(self):
        assert_raises(ValueError, join_by, 'a', self.a, self.b,
                      r1postfix='', r2postfix='')

    def test_no_r2postfix(self):
        # Basic test of join_by no_r2postfix
        a, b = self.a, self.b

        test = join_by(
            'a', a, b, r1postfix='1', r2postfix='', jointype='inner')
        control = np.array([(0, 50, 65, 100, 100), (1, 51, 66, 101, 101),
                            (2, 52, 67, 102, 102), (3, 53, 68, 103, 103),
                            (4, 54, 69, 104, 104), (5, 55, 70, 105, 105),
                            (6, 56, 71, 106, 106), (7, 57, 72, 107, 107),
                            (8, 58, 73, 108, 108), (9, 59, 74, 109, 109)],
                           dtype=[('a', int), ('b1', int), ('b', int),
                                  ('c', int), ('d', int)])
        assert_equal(test, control)

    def test_two_keys_two_vars(self):
        a = np.array(list(zip(np.tile([10, 11], 5), np.repeat(np.arange(5), 2),
                              np.arange(50, 60), np.arange(10, 20))),
                     dtype=[('k', int), ('a', int), ('b', int), ('c', int)])

        b = np.array(list(zip(np.tile([10, 11], 5), np.repeat(np.arange(5), 2),
                              np.arange(65, 75), np.arange(0, 10))),
                     dtype=[('k', int), ('a', int), ('b', int), ('c', int)])

        control = np.array([(10, 0, 50, 65, 10, 0), (11, 0, 51, 66, 11, 1),
                            (10, 1, 52, 67, 12, 2), (11, 1, 53, 68, 13, 3),
                            (10, 2, 54, 69, 14, 4), (11, 2, 55, 70, 15, 5),
                            (10, 3, 56, 71, 16, 6), (11, 3, 57, 72, 17, 7),
                            (10, 4, 58, 73, 18, 8), (11, 4, 59, 74, 19, 9)],
                           dtype=[('k', int), ('a', int), ('b1', int),
                                  ('b2', int), ('c1', int), ('c2', int)])
        test = join_by(
            ['a', 'k'], a, b, r1postfix='1', r2postfix='2', jointype='inner')
        assert_equal(test.dtype, control.dtype)
        assert_equal(test, control)

class TestAppendFieldsObj:
    """
    Test append_fields with arrays containing objects
    """
    # https://github.com/numpy/numpy/issues/2346

    def setup_method(self):
        from datetime import date
        self.data = {'obj': date(2000, 1, 1)}

    def test_append_to_objects(self):
        "Test append_fields when the base array contains objects"
        obj = self.data['obj']
        x = np.array([(obj, 1.), (obj, 2.)],
                      dtype=[('A', object), ('B', float)])
        y = np.array([10, 20], dtype=int)
        test = append_fields(x, 'C', data=y, usemask=False)
        control = np.array([(obj, 1.0, 10), (obj, 2.0, 20)],
                           dtype=[('A', object), ('B', float), ('C', int)])
        assert_equal(test, control)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\_format_impl.py ===
"""
Binary serialization

NPY format
==========

A simple format for saving numpy arrays to disk with the full
information about them.

The ``.npy`` format is the standard binary file format in NumPy for
persisting a *single* arbitrary NumPy array on disk. The format stores all
of the shape and dtype information necessary to reconstruct the array
correctly even on another machine with a different architecture.
The format is designed to be as simple as possible while achieving
its limited goals.

The ``.npz`` format is the standard format for persisting *multiple* NumPy
arrays on disk. A ``.npz`` file is a zip file containing multiple ``.npy``
files, one for each array.

Capabilities
------------

- Can represent all NumPy arrays including nested record arrays and
  object arrays.

- Represents the data in its native binary form.

- Supports Fortran-contiguous arrays directly.

- Stores all of the necessary information to reconstruct the array
  including shape and dtype on a machine of a different
  architecture.  Both little-endian and big-endian arrays are
  supported, and a file with little-endian numbers will yield
  a little-endian array on any machine reading the file. The
  types are described in terms of their actual sizes. For example,
  if a machine with a 64-bit C "long int" writes out an array with
  "long ints", a reading machine with 32-bit C "long ints" will yield
  an array with 64-bit integers.

- Is straightforward to reverse engineer. Datasets often live longer than
  the programs that created them. A competent developer should be
  able to create a solution in their preferred programming language to
  read most ``.npy`` files that they have been given without much
  documentation.

- Allows memory-mapping of the data. See `open_memmap`.

- Can be read from a filelike stream object instead of an actual file.

- Stores object arrays, i.e. arrays containing elements that are arbitrary
  Python objects. Files with object arrays are not to be mmapable, but
  can be read and written to disk.

Limitations
-----------

- Arbitrary subclasses of numpy.ndarray are not completely preserved.
  Subclasses will be accepted for writing, but only the array data will
  be written out. A regular numpy.ndarray object will be created
  upon reading the file.

.. warning::

  Due to limitations in the interpretation of structured dtypes, dtypes
  with fields with empty names will have the names replaced by 'f0', 'f1',
  etc. Such arrays will not round-trip through the format entirely
  accurately. The data is intact; only the field names will differ. We are
  working on a fix for this. This fix will not require a change in the
  file format. The arrays with such structures can still be saved and
  restored, and the correct dtype may be restored by using the
  ``loadedarray.view(correct_dtype)`` method.

File extensions
---------------

We recommend using the ``.npy`` and ``.npz`` extensions for files saved
in this format. This is by no means a requirement; applications may wish
to use these file formats but use an extension specific to the
application. In the absence of an obvious alternative, however,
we suggest using ``.npy`` and ``.npz``.

Version numbering
-----------------

The version numbering of these formats is independent of NumPy version
numbering. If the format is upgraded, the code in `numpy.io` will still
be able to read and write Version 1.0 files.

Format Version 1.0
------------------

The first 6 bytes are a magic string: exactly ``\\x93NUMPY``.

The next 1 byte is an unsigned byte: the major version number of the file
format, e.g. ``\\x01``.

The next 1 byte is an unsigned byte: the minor version number of the file
format, e.g. ``\\x00``. Note: the version of the file format is not tied
to the version of the numpy package.

The next 2 bytes form a little-endian unsigned short int: the length of
the header data HEADER_LEN.

The next HEADER_LEN bytes form the header data describing the array's
format. It is an ASCII string which contains a Python literal expression
of a dictionary. It is terminated by a newline (``\\n``) and padded with
spaces (``\\x20``) to make the total of
``len(magic string) + 2 + len(length) + HEADER_LEN`` be evenly divisible
by 64 for alignment purposes.

The dictionary contains three keys:

    "descr" : dtype.descr
      An object that can be passed as an argument to the `numpy.dtype`
      constructor to create the array's dtype.
    "fortran_order" : bool
      Whether the array data is Fortran-contiguous or not. Since
      Fortran-contiguous arrays are a common form of non-C-contiguity,
      we allow them to be written directly to disk for efficiency.
    "shape" : tuple of int
      The shape of the array.

For repeatability and readability, the dictionary keys are sorted in
alphabetic order. This is for convenience only. A writer SHOULD implement
this if possible. A reader MUST NOT depend on this.

Following the header comes the array data. If the dtype contains Python
objects (i.e. ``dtype.hasobject is True``), then the data is a Python
pickle of the array. Otherwise the data is the contiguous (either C-
or Fortran-, depending on ``fortran_order``) bytes of the array.
Consumers can figure out the number of bytes by multiplying the number
of elements given by the shape (noting that ``shape=()`` means there is
1 element) by ``dtype.itemsize``.

Format Version 2.0
------------------

The version 1.0 format only allowed the array header to have a total size of
65535 bytes.  This can be exceeded by structured arrays with a large number of
columns.  The version 2.0 format extends the header size to 4 GiB.
`numpy.save` will automatically save in 2.0 format if the data requires it,
else it will always use the more compatible 1.0 format.

The description of the fourth element of the header therefore has become:
"The next 4 bytes form a little-endian unsigned int: the length of the header
data HEADER_LEN."

Format Version 3.0
------------------

This version replaces the ASCII string (which in practice was latin1) with
a utf8-encoded string, so supports structured types with any unicode field
names.

Notes
-----
The ``.npy`` format, including motivation for creating it and a comparison of
alternatives, is described in the
:doc:`"npy-format" NEP <neps:nep-0001-npy-format>`, however details have
evolved with time and this document is more current.

"""
import io
import os
import pickle
import warnings

import numpy
from numpy._utils import set_module
from numpy.lib._utils_impl import drop_metadata

__all__ = []

drop_metadata.__module__ = "numpy.lib.format"

EXPECTED_KEYS = {'descr', 'fortran_order', 'shape'}
MAGIC_PREFIX = b'\x93NUMPY'
MAGIC_LEN = len(MAGIC_PREFIX) + 2
ARRAY_ALIGN = 64  # plausible values are powers of 2 between 16 and 4096
BUFFER_SIZE = 2**18  # size of buffer for reading npz files in bytes
# allow growth within the address space of a 64 bit machine along one axis
GROWTH_AXIS_MAX_DIGITS = 21  # = len(str(8*2**64-1)) hypothetical int1 dtype

# difference between version 1.0 and 2.0 is a 4 byte (I) header length
# instead of 2 bytes (H) allowing storage of large structured arrays
_header_size_info = {
    (1, 0): ('<H', 'latin1'),
    (2, 0): ('<I', 'latin1'),
    (3, 0): ('<I', 'utf8'),
}

# Python's literal_eval is not actually safe for large inputs, since parsing
# may become slow or even cause interpreter crashes.
# This is an arbitrary, low limit which should make it safe in practice.
_MAX_HEADER_SIZE = 10000


def _check_version(version):
    if version not in [(1, 0), (2, 0), (3, 0), None]:
        msg = "we only support format version (1,0), (2,0), and (3,0), not %s"
        raise ValueError(msg % (version,))


@set_module("numpy.lib.format")
def magic(major, minor):
    """ Return the magic string for the given file format version.

    Parameters
    ----------
    major : int in [0, 255]
    minor : int in [0, 255]

    Returns
    -------
    magic : str

    Raises
    ------
    ValueError if the version cannot be formatted.
    """
    if major < 0 or major > 255:
        raise ValueError("major version must be 0 <= major < 256")
    if minor < 0 or minor > 255:
        raise ValueError("minor version must be 0 <= minor < 256")
    return MAGIC_PREFIX + bytes([major, minor])


@set_module("numpy.lib.format")
def read_magic(fp):
    """ Read the magic string to get the version of the file format.

    Parameters
    ----------
    fp : filelike object

    Returns
    -------
    major : int
    minor : int
    """
    magic_str = _read_bytes(fp, MAGIC_LEN, "magic string")
    if magic_str[:-2] != MAGIC_PREFIX:
        msg = "the magic string is not correct; expected %r, got %r"
        raise ValueError(msg % (MAGIC_PREFIX, magic_str[:-2]))
    major, minor = magic_str[-2:]
    return major, minor


@set_module("numpy.lib.format")
def dtype_to_descr(dtype):
    """
    Get a serializable descriptor from the dtype.

    The .descr attribute of a dtype object cannot be round-tripped through
    the dtype() constructor. Simple types, like dtype('float32'), have
    a descr which looks like a record array with one field with '' as
    a name. The dtype() constructor interprets this as a request to give
    a default name.  Instead, we construct descriptor that can be passed to
    dtype().

    Parameters
    ----------
    dtype : dtype
        The dtype of the array that will be written to disk.

    Returns
    -------
    descr : object
        An object that can be passed to `numpy.dtype()` in order to
        replicate the input dtype.

    """
    # NOTE: that drop_metadata may not return the right dtype e.g. for user
    #       dtypes.  In that case our code below would fail the same, though.
    new_dtype = drop_metadata(dtype)
    if new_dtype is not dtype:
        warnings.warn("metadata on a dtype is not saved to an npy/npz. "
                      "Use another format (such as pickle) to store it.",
                      UserWarning, stacklevel=2)
    dtype = new_dtype

    if dtype.names is not None:
        # This is a record array. The .descr is fine.  XXX: parts of the
        # record array with an empty name, like padding bytes, still get
        # fiddled with. This needs to be fixed in the C implementation of
        # dtype().
        return dtype.descr
    elif not type(dtype)._legacy:
        # this must be a user-defined dtype since numpy does not yet expose any
        # non-legacy dtypes in the public API
        #
        # non-legacy dtypes don't yet have __array_interface__
        # support. Instead, as a hack, we use pickle to save the array, and lie
        # that the dtype is object. When the array is loaded, the descriptor is
        # unpickled with the array and the object dtype in the header is
        # discarded.
        #
        # a future NEP should define a way to serialize user-defined
        # descriptors and ideally work out the possible security implications
        warnings.warn("Custom dtypes are saved as python objects using the "
                      "pickle protocol. Loading this file requires "
                      "allow_pickle=True to be set.",
                      UserWarning, stacklevel=2)
        return "|O"
    else:
        return dtype.str


@set_module("numpy.lib.format")
def descr_to_dtype(descr):
    """
    Returns a dtype based off the given description.

    This is essentially the reverse of `~lib.format.dtype_to_descr`. It will
    remove the valueless padding fields created by, i.e. simple fields like
    dtype('float32'), and then convert the description to its corresponding
    dtype.

    Parameters
    ----------
    descr : object
        The object retrieved by dtype.descr. Can be passed to
        `numpy.dtype` in order to replicate the input dtype.

    Returns
    -------
    dtype : dtype
        The dtype constructed by the description.

    """
    if isinstance(descr, str):
        # No padding removal needed
        return numpy.dtype(descr)
    elif isinstance(descr, tuple):
        # subtype, will always have a shape descr[1]
        dt = descr_to_dtype(descr[0])
        return numpy.dtype((dt, descr[1]))

    titles = []
    names = []
    formats = []
    offsets = []
    offset = 0
    for field in descr:
        if len(field) == 2:
            name, descr_str = field
            dt = descr_to_dtype(descr_str)
        else:
            name, descr_str, shape = field
            dt = numpy.dtype((descr_to_dtype(descr_str), shape))

        # Ignore padding bytes, which will be void bytes with '' as name
        # Once support for blank names is removed, only "if name == ''" needed)
        is_pad = (name == '' and dt.type is numpy.void and dt.names is None)
        if not is_pad:
            title, name = name if isinstance(name, tuple) else (None, name)
            titles.append(title)
            names.append(name)
            formats.append(dt)
            offsets.append(offset)
        offset += dt.itemsize

    return numpy.dtype({'names': names, 'formats': formats, 'titles': titles,
                        'offsets': offsets, 'itemsize': offset})


@set_module("numpy.lib.format")
def header_data_from_array_1_0(array):
    """ Get the dictionary of header metadata from a numpy.ndarray.

    Parameters
    ----------
    array : numpy.ndarray

    Returns
    -------
    d : dict
        This has the appropriate entries for writing its string representation
        to the header of the file.
    """
    d = {'shape': array.shape}
    if array.flags.c_contiguous:
        d['fortran_order'] = False
    elif array.flags.f_contiguous:
        d['fortran_order'] = True
    else:
        # Totally non-contiguous data. We will have to make it C-contiguous
        # before writing. Note that we need to test for C_CONTIGUOUS first
        # because a 1-D array is both C_CONTIGUOUS and F_CONTIGUOUS.
        d['fortran_order'] = False

    d['descr'] = dtype_to_descr(array.dtype)
    return d


def _wrap_header(header, version):
    """
    Takes a stringified header, and attaches the prefix and padding to it
    """
    import struct
    assert version is not None
    fmt, encoding = _header_size_info[version]
    header = header.encode(encoding)
    hlen = len(header) + 1
    padlen = ARRAY_ALIGN - ((MAGIC_LEN + struct.calcsize(fmt) + hlen) % ARRAY_ALIGN)
    try:
        header_prefix = magic(*version) + struct.pack(fmt, hlen + padlen)
    except struct.error:
        msg = f"Header length {hlen} too big for version={version}"
        raise ValueError(msg) from None

    # Pad the header with spaces and a final newline such that the magic
    # string, the header-length short and the header are aligned on a
    # ARRAY_ALIGN byte boundary.  This supports memory mapping of dtypes
    # aligned up to ARRAY_ALIGN on systems like Linux where mmap()
    # offset must be page-aligned (i.e. the beginning of the file).
    return header_prefix + header + b' ' * padlen + b'\n'


def _wrap_header_guess_version(header):
    """
    Like `_wrap_header`, but chooses an appropriate version given the contents
    """
    try:
        return _wrap_header(header, (1, 0))
    except ValueError:
        pass

    try:
        ret = _wrap_header(header, (2, 0))
    except UnicodeEncodeError:
        pass
    else:
        warnings.warn("Stored array in format 2.0. It can only be"
                      "read by NumPy >= 1.9", UserWarning, stacklevel=2)
        return ret

    header = _wrap_header(header, (3, 0))
    warnings.warn("Stored array in format 3.0. It can only be "
                  "read by NumPy >= 1.17", UserWarning, stacklevel=2)
    return header


def _write_array_header(fp, d, version=None):
    """ Write the header for an array and returns the version used

    Parameters
    ----------
    fp : filelike object
    d : dict
        This has the appropriate entries for writing its string representation
        to the header of the file.
    version : tuple or None
        None means use oldest that works. Providing an explicit version will
        raise a ValueError if the format does not allow saving this data.
        Default: None
    """
    header = ["{"]
    for key, value in sorted(d.items()):
        # Need to use repr here, since we eval these when reading
        header.append(f"'{key}': {repr(value)}, ")
    header.append("}")
    header = "".join(header)

    # Add some spare space so that the array header can be modified in-place
    # when changing the array size, e.g. when growing it by appending data at
    # the end.
    shape = d['shape']
    header += " " * ((GROWTH_AXIS_MAX_DIGITS - len(repr(
        shape[-1 if d['fortran_order'] else 0]
    ))) if len(shape) > 0 else 0)

    if version is None:
        header = _wrap_header_guess_version(header)
    else:
        header = _wrap_header(header, version)
    fp.write(header)


@set_module("numpy.lib.format")
def write_array_header_1_0(fp, d):
    """ Write the header for an array using the 1.0 format.

    Parameters
    ----------
    fp : filelike object
    d : dict
        This has the appropriate entries for writing its string
        representation to the header of the file.
    """
    _write_array_header(fp, d, (1, 0))


@set_module("numpy.lib.format")
def write_array_header_2_0(fp, d):
    """ Write the header for an array using the 2.0 format.
        The 2.0 format allows storing very large structured arrays.

    Parameters
    ----------
    fp : filelike object
    d : dict
        This has the appropriate entries for writing its string
        representation to the header of the file.
    """
    _write_array_header(fp, d, (2, 0))


@set_module("numpy.lib.format")
def read_array_header_1_0(fp, max_header_size=_MAX_HEADER_SIZE):
    """
    Read an array header from a filelike object using the 1.0 file format
    version.

    This will leave the file object located just after the header.

    Parameters
    ----------
    fp : filelike object
        A file object or something with a `.read()` method like a file.

    Returns
    -------
    shape : tuple of int
        The shape of the array.
    fortran_order : bool
        The array data will be written out directly if it is either
        C-contiguous or Fortran-contiguous. Otherwise, it will be made
        contiguous before writing it out.
    dtype : dtype
        The dtype of the file's data.
    max_header_size : int, optional
        Maximum allowed size of the header.  Large headers may not be safe
        to load securely and thus require explicitly passing a larger value.
        See :py:func:`ast.literal_eval()` for details.

    Raises
    ------
    ValueError
        If the data is invalid.

    """
    return _read_array_header(
            fp, version=(1, 0), max_header_size=max_header_size)


@set_module("numpy.lib.format")
def read_array_header_2_0(fp, max_header_size=_MAX_HEADER_SIZE):
    """
    Read an array header from a filelike object using the 2.0 file format
    version.

    This will leave the file object located just after the header.

    Parameters
    ----------
    fp : filelike object
        A file object or something with a `.read()` method like a file.
    max_header_size : int, optional
        Maximum allowed size of the header.  Large headers may not be safe
        to load securely and thus require explicitly passing a larger value.
        See :py:func:`ast.literal_eval()` for details.

    Returns
    -------
    shape : tuple of int
        The shape of the array.
    fortran_order : bool
        The array data will be written out directly if it is either
        C-contiguous or Fortran-contiguous. Otherwise, it will be made
        contiguous before writing it out.
    dtype : dtype
        The dtype of the file's data.

    Raises
    ------
    ValueError
        If the data is invalid.

    """
    return _read_array_header(
            fp, version=(2, 0), max_header_size=max_header_size)


def _filter_header(s):
    """Clean up 'L' in npz header ints.

    Cleans up the 'L' in strings representing integers. Needed to allow npz
    headers produced in Python2 to be read in Python3.

    Parameters
    ----------
    s : string
        Npy file header.

    Returns
    -------
    header : str
        Cleaned up header.

    """
    import tokenize
    from io import StringIO

    tokens = []
    last_token_was_number = False
    for token in tokenize.generate_tokens(StringIO(s).readline):
        token_type = token[0]
        token_string = token[1]
        if (last_token_was_number and
                token_type == tokenize.NAME and
                token_string == "L"):
            continue
        else:
            tokens.append(token)
        last_token_was_number = (token_type == tokenize.NUMBER)
    return tokenize.untokenize(tokens)


def _read_array_header(fp, version, max_header_size=_MAX_HEADER_SIZE):
    """
    see read_array_header_1_0
    """
    # Read an unsigned, little-endian short int which has the length of the
    # header.
    import ast
    import struct
    hinfo = _header_size_info.get(version)
    if hinfo is None:
        raise ValueError(f"Invalid version {version!r}")
    hlength_type, encoding = hinfo

    hlength_str = _read_bytes(fp, struct.calcsize(hlength_type), "array header length")
    header_length = struct.unpack(hlength_type, hlength_str)[0]
    header = _read_bytes(fp, header_length, "array header")
    header = header.decode(encoding)
    if len(header) > max_header_size:
        raise ValueError(
            f"Header info length ({len(header)}) is large and may not be safe "
            "to load securely.\n"
            "To allow loading, adjust `max_header_size` or fully trust "
            "the `.npy` file using `allow_pickle=True`.\n"
            "For safety against large resource use or crashes, sandboxing "
            "may be necessary.")

    # The header is a pretty-printed string representation of a literal
    # Python dictionary with trailing newlines padded to a ARRAY_ALIGN byte
    # boundary. The keys are strings.
    #   "shape" : tuple of int
    #   "fortran_order" : bool
    #   "descr" : dtype.descr
    # Versions (2, 0) and (1, 0) could have been created by a Python 2
    # implementation before header filtering was implemented.
    #
    # For performance reasons, we try without _filter_header first though
    try:
        d = ast.literal_eval(header)
    except SyntaxError as e:
        if version <= (2, 0):
            header = _filter_header(header)
            try:
                d = ast.literal_eval(header)
            except SyntaxError as e2:
                msg = "Cannot parse header: {!r}"
                raise ValueError(msg.format(header)) from e2
            else:
                warnings.warn(
                    "Reading `.npy` or `.npz` file required additional "
                    "header parsing as it was created on Python 2. Save the "
                    "file again to speed up loading and avoid this warning.",
                    UserWarning, stacklevel=4)
        else:
            msg = "Cannot parse header: {!r}"
            raise ValueError(msg.format(header)) from e
    if not isinstance(d, dict):
        msg = "Header is not a dictionary: {!r}"
        raise ValueError(msg.format(d))

    if EXPECTED_KEYS != d.keys():
        keys = sorted(d.keys())
        msg = "Header does not contain the correct keys: {!r}"
        raise ValueError(msg.format(keys))

    # Sanity-check the values.
    if (not isinstance(d['shape'], tuple) or
            not all(isinstance(x, int) for x in d['shape'])):
        msg = "shape is not valid: {!r}"
        raise ValueError(msg.format(d['shape']))
    if not isinstance(d['fortran_order'], bool):
        msg = "fortran_order is not a valid bool: {!r}"
        raise ValueError(msg.format(d['fortran_order']))
    try:
        dtype = descr_to_dtype(d['descr'])
    except TypeError as e:
        msg = "descr is not a valid dtype descriptor: {!r}"
        raise ValueError(msg.format(d['descr'])) from e

    return d['shape'], d['fortran_order'], dtype


@set_module("numpy.lib.format")
def write_array(fp, array, version=None, allow_pickle=True, pickle_kwargs=None):
    """
    Write an array to an NPY file, including a header.

    If the array is neither C-contiguous nor Fortran-contiguous AND the
    file_like object is not a real file object, this function will have to
    copy data in memory.

    Parameters
    ----------
    fp : file_like object
        An open, writable file object, or similar object with a
        ``.write()`` method.
    array : ndarray
        The array to write to disk.
    version : (int, int) or None, optional
        The version number of the format. None means use the oldest
        supported version that is able to store the data.  Default: None
    allow_pickle : bool, optional
        Whether to allow writing pickled data. Default: True
    pickle_kwargs : dict, optional
        Additional keyword arguments to pass to pickle.dump, excluding
        'protocol'. These are only useful when pickling objects in object
        arrays to Python 2 compatible format.

    Raises
    ------
    ValueError
        If the array cannot be persisted. This includes the case of
        allow_pickle=False and array being an object array.
    Various other errors
        If the array contains Python objects as part of its dtype, the
        process of pickling them may raise various errors if the objects
        are not picklable.

    """
    _check_version(version)
    _write_array_header(fp, header_data_from_array_1_0(array), version)

    if array.itemsize == 0:
        buffersize = 0
    else:
        # Set buffer size to 16 MiB to hide the Python loop overhead.
        buffersize = max(16 * 1024 ** 2 // array.itemsize, 1)

    dtype_class = type(array.dtype)

    if array.dtype.hasobject or not dtype_class._legacy:
        # We contain Python objects so we cannot write out the data
        # directly.  Instead, we will pickle it out
        if not allow_pickle:
            if array.dtype.hasobject:
                raise ValueError("Object arrays cannot be saved when "
                                 "allow_pickle=False")
            if not dtype_class._legacy:
                raise ValueError("User-defined dtypes cannot be saved "
                                 "when allow_pickle=False")
        if pickle_kwargs is None:
            pickle_kwargs = {}
        pickle.dump(array, fp, protocol=4, **pickle_kwargs)
    elif array.flags.f_contiguous and not array.flags.c_contiguous:
        if isfileobj(fp):
            array.T.tofile(fp)
        else:
            for chunk in numpy.nditer(
                    array, flags=['external_loop', 'buffered', 'zerosize_ok'],
                    buffersize=buffersize, order='F'):
                fp.write(chunk.tobytes('C'))
    elif isfileobj(fp):
        array.tofile(fp)
    else:
        for chunk in numpy.nditer(
                array, flags=['external_loop', 'buffered', 'zerosize_ok'],
                buffersize=buffersize, order='C'):
            fp.write(chunk.tobytes('C'))


@set_module("numpy.lib.format")
def read_array(fp, allow_pickle=False, pickle_kwargs=None, *,
               max_header_size=_MAX_HEADER_SIZE):
    """
    Read an array from an NPY file.

    Parameters
    ----------
    fp : file_like object
        If this is not a real file object, then this may take extra memory
        and time.
    allow_pickle : bool, optional
        Whether to allow writing pickled data. Default: False
    pickle_kwargs : dict
        Additional keyword arguments to pass to pickle.load. These are only
        useful when loading object arrays saved on Python 2.
    max_header_size : int, optional
        Maximum allowed size of the header.  Large headers may not be safe
        to load securely and thus require explicitly passing a larger value.
        See :py:func:`ast.literal_eval()` for details.
        This option is ignored when `allow_pickle` is passed.  In that case
        the file is by definition trusted and the limit is unnecessary.

    Returns
    -------
    array : ndarray
        The array from the data on disk.

    Raises
    ------
    ValueError
        If the data is invalid, or allow_pickle=False and the file contains
        an object array.

    """
    if allow_pickle:
        # Effectively ignore max_header_size, since `allow_pickle` indicates
        # that the input is fully trusted.
        max_header_size = 2**64

    version = read_magic(fp)
    _check_version(version)
    shape, fortran_order, dtype = _read_array_header(
            fp, version, max_header_size=max_header_size)
    if len(shape) == 0:
        count = 1
    else:
        count = numpy.multiply.reduce(shape, dtype=numpy.int64)

    # Now read the actual data.
    if dtype.hasobject:
        # The array contained Python objects. We need to unpickle the data.
        if not allow_pickle:
            raise ValueError("Object arrays cannot be loaded when "
                             "allow_pickle=False")
        if pickle_kwargs is None:
            pickle_kwargs = {}
        try:
            array = pickle.load(fp, **pickle_kwargs)
        except UnicodeError as err:
            # Friendlier error message
            raise UnicodeError("Unpickling a python object failed: %r\n"
                               "You may need to pass the encoding= option "
                               "to numpy.load" % (err,)) from err
    else:
        if isfileobj(fp):
            # We can use the fast fromfile() function.
            array = numpy.fromfile(fp, dtype=dtype, count=count)
        else:
            # This is not a real file. We have to read it the
            # memory-intensive way.
            # crc32 module fails on reads greater than 2 ** 32 bytes,
            # breaking large reads from gzip streams. Chunk reads to
            # BUFFER_SIZE bytes to avoid issue and reduce memory overhead
            # of the read. In non-chunked case count < max_read_count, so
            # only one read is performed.

            # Use np.ndarray instead of np.empty since the latter does
            # not correctly instantiate zero-width string dtypes; see
            # https://github.com/numpy/numpy/pull/6430
            array = numpy.ndarray(count, dtype=dtype)

            if dtype.itemsize > 0:
                # If dtype.itemsize == 0 then there's nothing more to read
                max_read_count = BUFFER_SIZE // min(BUFFER_SIZE, dtype.itemsize)

                for i in range(0, count, max_read_count):
                    read_count = min(max_read_count, count - i)
                    read_size = int(read_count * dtype.itemsize)
                    data = _read_bytes(fp, read_size, "array data")
                    array[i:i + read_count] = numpy.frombuffer(data, dtype=dtype,
                                                             count=read_count)

        if array.size != count:
            raise ValueError(
                "Failed to read all data for array. "
                f"Expected {shape} = {count} elements, "
                f"could only read {array.size} elements. "
                "(file seems not fully written?)"
            )

        if fortran_order:
            array.shape = shape[::-1]
            array = array.transpose()
        else:
            array.shape = shape

    return array


@set_module("numpy.lib.format")
def open_memmap(filename, mode='r+', dtype=None, shape=None,
                fortran_order=False, version=None, *,
                max_header_size=_MAX_HEADER_SIZE):
    """
    Open a .npy file as a memory-mapped array.

    This may be used to read an existing file or create a new one.

    Parameters
    ----------
    filename : str or path-like
        The name of the file on disk.  This may *not* be a file-like
        object.
    mode : str, optional
        The mode in which to open the file; the default is 'r+'.  In
        addition to the standard file modes, 'c' is also accepted to mean
        "copy on write."  See `memmap` for the available mode strings.
    dtype : data-type, optional
        The data type of the array if we are creating a new file in "write"
        mode, if not, `dtype` is ignored.  The default value is None, which
        results in a data-type of `float64`.
    shape : tuple of int
        The shape of the array if we are creating a new file in "write"
        mode, in which case this parameter is required.  Otherwise, this
        parameter is ignored and is thus optional.
    fortran_order : bool, optional
        Whether the array should be Fortran-contiguous (True) or
        C-contiguous (False, the default) if we are creating a new file in
        "write" mode.
    version : tuple of int (major, minor) or None
        If the mode is a "write" mode, then this is the version of the file
        format used to create the file.  None means use the oldest
        supported version that is able to store the data.  Default: None
    max_header_size : int, optional
        Maximum allowed size of the header.  Large headers may not be safe
        to load securely and thus require explicitly passing a larger value.
        See :py:func:`ast.literal_eval()` for details.

    Returns
    -------
    marray : memmap
        The memory-mapped array.

    Raises
    ------
    ValueError
        If the data or the mode is invalid.
    OSError
        If the file is not found or cannot be opened correctly.

    See Also
    --------
    numpy.memmap

    """
    if isfileobj(filename):
        raise ValueError("Filename must be a string or a path-like object."
                         "  Memmap cannot use existing file handles.")

    if 'w' in mode:
        # We are creating the file, not reading it.
        # Check if we ought to create the file.
        _check_version(version)
        # Ensure that the given dtype is an authentic dtype object rather
        # than just something that can be interpreted as a dtype object.
        dtype = numpy.dtype(dtype)
        if dtype.hasobject:
            msg = "Array can't be memory-mapped: Python objects in dtype."
            raise ValueError(msg)
        d = {
            "descr": dtype_to_descr(dtype),
            "fortran_order": fortran_order,
            "shape": shape,
        }
        # If we got here, then it should be safe to create the file.
        with open(os.fspath(filename), mode + 'b') as fp:
            _write_array_header(fp, d, version)
            offset = fp.tell()
    else:
        # Read the header of the file first.
        with open(os.fspath(filename), 'rb') as fp:
            version = read_magic(fp)
            _check_version(version)

            shape, fortran_order, dtype = _read_array_header(
                    fp, version, max_header_size=max_header_size)
            if dtype.hasobject:
                msg = "Array can't be memory-mapped: Python objects in dtype."
                raise ValueError(msg)
            offset = fp.tell()

    if fortran_order:
        order = 'F'
    else:
        order = 'C'

    # We need to change a write-only mode to a read-write mode since we've
    # already written data to the file.
    if mode == 'w+':
        mode = 'r+'

    marray = numpy.memmap(filename, dtype=dtype, shape=shape, order=order,
        mode=mode, offset=offset)

    return marray


def _read_bytes(fp, size, error_template="ran out of data"):
    """
    Read from file-like object until size bytes are read.
    Raises ValueError if not EOF is encountered before size bytes are read.
    Non-blocking objects only supported if they derive from io objects.

    Required as e.g. ZipExtFile in python 2.6 can return less data than
    requested.
    """
    data = b""
    while True:
        # io files (default in python3) return None or raise on
        # would-block, python2 file will truncate, probably nothing can be
        # done about that.  note that regular files can't be non-blocking
        try:
            r = fp.read(size - len(data))
            data += r
            if len(r) == 0 or len(data) == size:
                break
        except BlockingIOError:
            pass
    if len(data) != size:
        msg = "EOF: reading %s, expected %d bytes got %d"
        raise ValueError(msg % (error_template, size, len(data)))
    else:
        return data


@set_module("numpy.lib.format")
def isfileobj(f):
    if not isinstance(f, (io.FileIO, io.BufferedReader, io.BufferedWriter)):
        return False
    try:
        # BufferedReader/Writer may raise OSError when
        # fetching `fileno()` (e.g. when wrapping BytesIO).
        f.fileno()
        return True
    except OSError:
        return False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\_iotools.py ===
"""A collection of functions designed to help I/O with ascii files.

"""
__docformat__ = "restructuredtext en"

import itertools

import numpy as np
import numpy._core.numeric as nx
from numpy._utils import asbytes, asunicode


def _decode_line(line, encoding=None):
    """Decode bytes from binary input streams.

    Defaults to decoding from 'latin1'.

    Parameters
    ----------
    line : str or bytes
         Line to be decoded.
    encoding : str
         Encoding used to decode `line`.

    Returns
    -------
    decoded_line : str

    """
    if type(line) is bytes:
        if encoding is None:
            encoding = "latin1"
        line = line.decode(encoding)

    return line


def _is_string_like(obj):
    """
    Check whether obj behaves like a string.
    """
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True


def _is_bytes_like(obj):
    """
    Check whether obj behaves like a bytes object.
    """
    try:
        obj + b''
    except (TypeError, ValueError):
        return False
    return True


def has_nested_fields(ndtype):
    """
    Returns whether one or several fields of a dtype are nested.

    Parameters
    ----------
    ndtype : dtype
        Data-type of a structured array.

    Raises
    ------
    AttributeError
        If `ndtype` does not have a `names` attribute.

    Examples
    --------
    >>> import numpy as np
    >>> dt = np.dtype([('name', 'S4'), ('x', float), ('y', float)])
    >>> np.lib._iotools.has_nested_fields(dt)
    False

    """
    return any(ndtype[name].names is not None for name in ndtype.names or ())


def flatten_dtype(ndtype, flatten_base=False):
    """
    Unpack a structured data-type by collapsing nested fields and/or fields
    with a shape.

    Note that the field names are lost.

    Parameters
    ----------
    ndtype : dtype
        The datatype to collapse
    flatten_base : bool, optional
       If True, transform a field with a shape into several fields. Default is
       False.

    Examples
    --------
    >>> import numpy as np
    >>> dt = np.dtype([('name', 'S4'), ('x', float), ('y', float),
    ...                ('block', int, (2, 3))])
    >>> np.lib._iotools.flatten_dtype(dt)
    [dtype('S4'), dtype('float64'), dtype('float64'), dtype('int64')]
    >>> np.lib._iotools.flatten_dtype(dt, flatten_base=True)
    [dtype('S4'),
     dtype('float64'),
     dtype('float64'),
     dtype('int64'),
     dtype('int64'),
     dtype('int64'),
     dtype('int64'),
     dtype('int64'),
     dtype('int64')]

    """
    names = ndtype.names
    if names is None:
        if flatten_base:
            return [ndtype.base] * int(np.prod(ndtype.shape))
        return [ndtype.base]
    else:
        types = []
        for field in names:
            info = ndtype.fields[field]
            flat_dt = flatten_dtype(info[0], flatten_base)
            types.extend(flat_dt)
        return types


class LineSplitter:
    """
    Object to split a string at a given delimiter or at given places.

    Parameters
    ----------
    delimiter : str, int, or sequence of ints, optional
        If a string, character used to delimit consecutive fields.
        If an integer or a sequence of integers, width(s) of each field.
    comments : str, optional
        Character used to mark the beginning of a comment. Default is '#'.
    autostrip : bool, optional
        Whether to strip each individual field. Default is True.

    """

    def autostrip(self, method):
        """
        Wrapper to strip each member of the output of `method`.

        Parameters
        ----------
        method : function
            Function that takes a single argument and returns a sequence of
            strings.

        Returns
        -------
        wrapped : function
            The result of wrapping `method`. `wrapped` takes a single input
            argument and returns a list of strings that are stripped of
            white-space.

        """
        return lambda input: [_.strip() for _ in method(input)]

    def __init__(self, delimiter=None, comments='#', autostrip=True,
                 encoding=None):
        delimiter = _decode_line(delimiter)
        comments = _decode_line(comments)

        self.comments = comments

        # Delimiter is a character
        if (delimiter is None) or isinstance(delimiter, str):
            delimiter = delimiter or None
            _handyman = self._delimited_splitter
        # Delimiter is a list of field widths
        elif hasattr(delimiter, '__iter__'):
            _handyman = self._variablewidth_splitter
            idx = np.cumsum([0] + list(delimiter))
            delimiter = [slice(i, j) for (i, j) in itertools.pairwise(idx)]
        # Delimiter is a single integer
        elif int(delimiter):
            (_handyman, delimiter) = (
                    self._fixedwidth_splitter, int(delimiter))
        else:
            (_handyman, delimiter) = (self._delimited_splitter, None)
        self.delimiter = delimiter
        if autostrip:
            self._handyman = self.autostrip(_handyman)
        else:
            self._handyman = _handyman
        self.encoding = encoding

    def _delimited_splitter(self, line):
        """Chop off comments, strip, and split at delimiter. """
        if self.comments is not None:
            line = line.split(self.comments)[0]
        line = line.strip(" \r\n")
        if not line:
            return []
        return line.split(self.delimiter)

    def _fixedwidth_splitter(self, line):
        if self.comments is not None:
            line = line.split(self.comments)[0]
        line = line.strip("\r\n")
        if not line:
            return []
        fixed = self.delimiter
        slices = [slice(i, i + fixed) for i in range(0, len(line), fixed)]
        return [line[s] for s in slices]

    def _variablewidth_splitter(self, line):
        if self.comments is not None:
            line = line.split(self.comments)[0]
        if not line:
            return []
        slices = self.delimiter
        return [line[s] for s in slices]

    def __call__(self, line):
        return self._handyman(_decode_line(line, self.encoding))


class NameValidator:
    """
    Object to validate a list of strings to use as field names.

    The strings are stripped of any non alphanumeric character, and spaces
    are replaced by '_'. During instantiation, the user can define a list
    of names to exclude, as well as a list of invalid characters. Names in
    the exclusion list are appended a '_' character.

    Once an instance has been created, it can be called with a list of
    names, and a list of valid names will be created.  The `__call__`
    method accepts an optional keyword "default" that sets the default name
    in case of ambiguity. By default this is 'f', so that names will
    default to `f0`, `f1`, etc.

    Parameters
    ----------
    excludelist : sequence, optional
        A list of names to exclude. This list is appended to the default
        list ['return', 'file', 'print']. Excluded names are appended an
        underscore: for example, `file` becomes `file_` if supplied.
    deletechars : str, optional
        A string combining invalid characters that must be deleted from the
        names.
    case_sensitive : {True, False, 'upper', 'lower'}, optional
        * If True, field names are case-sensitive.
        * If False or 'upper', field names are converted to upper case.
        * If 'lower', field names are converted to lower case.

        The default value is True.
    replace_space : '_', optional
        Character(s) used in replacement of white spaces.

    Notes
    -----
    Calling an instance of `NameValidator` is the same as calling its
    method `validate`.

    Examples
    --------
    >>> import numpy as np
    >>> validator = np.lib._iotools.NameValidator()
    >>> validator(['file', 'field2', 'with space', 'CaSe'])
    ('file_', 'field2', 'with_space', 'CaSe')

    >>> validator = np.lib._iotools.NameValidator(excludelist=['excl'],
    ...                                           deletechars='q',
    ...                                           case_sensitive=False)
    >>> validator(['excl', 'field2', 'no_q', 'with space', 'CaSe'])
    ('EXCL', 'FIELD2', 'NO_Q', 'WITH_SPACE', 'CASE')

    """

    defaultexcludelist = 'return', 'file', 'print'
    defaultdeletechars = frozenset(r"""~!@#$%^&*()-=+~\|]}[{';: /?.>,<""")

    def __init__(self, excludelist=None, deletechars=None,
                 case_sensitive=None, replace_space='_'):
        # Process the exclusion list ..
        if excludelist is None:
            excludelist = []
        excludelist.extend(self.defaultexcludelist)
        self.excludelist = excludelist
        # Process the list of characters to delete
        if deletechars is None:
            delete = set(self.defaultdeletechars)
        else:
            delete = set(deletechars)
        delete.add('"')
        self.deletechars = delete
        # Process the case option .....
        if (case_sensitive is None) or (case_sensitive is True):
            self.case_converter = lambda x: x
        elif (case_sensitive is False) or case_sensitive.startswith('u'):
            self.case_converter = lambda x: x.upper()
        elif case_sensitive.startswith('l'):
            self.case_converter = lambda x: x.lower()
        else:
            msg = f'unrecognized case_sensitive value {case_sensitive}.'
            raise ValueError(msg)

        self.replace_space = replace_space

    def validate(self, names, defaultfmt="f%i", nbfields=None):
        """
        Validate a list of strings as field names for a structured array.

        Parameters
        ----------
        names : sequence of str
            Strings to be validated.
        defaultfmt : str, optional
            Default format string, used if validating a given string
            reduces its length to zero.
        nbfields : integer, optional
            Final number of validated names, used to expand or shrink the
            initial list of names.

        Returns
        -------
        validatednames : list of str
            The list of validated field names.

        Notes
        -----
        A `NameValidator` instance can be called directly, which is the
        same as calling `validate`. For examples, see `NameValidator`.

        """
        # Initial checks ..............
        if (names is None):
            if (nbfields is None):
                return None
            names = []
        if isinstance(names, str):
            names = [names, ]
        if nbfields is not None:
            nbnames = len(names)
            if (nbnames < nbfields):
                names = list(names) + [''] * (nbfields - nbnames)
            elif (nbnames > nbfields):
                names = names[:nbfields]
        # Set some shortcuts ...........
        deletechars = self.deletechars
        excludelist = self.excludelist
        case_converter = self.case_converter
        replace_space = self.replace_space
        # Initializes some variables ...
        validatednames = []
        seen = {}
        nbempty = 0

        for item in names:
            item = case_converter(item).strip()
            if replace_space:
                item = item.replace(' ', replace_space)
            item = ''.join([c for c in item if c not in deletechars])
            if item == '':
                item = defaultfmt % nbempty
                while item in names:
                    nbempty += 1
                    item = defaultfmt % nbempty
                nbempty += 1
            elif item in excludelist:
                item += '_'
            cnt = seen.get(item, 0)
            if cnt > 0:
                validatednames.append(item + '_%d' % cnt)
            else:
                validatednames.append(item)
            seen[item] = cnt + 1
        return tuple(validatednames)

    def __call__(self, names, defaultfmt="f%i", nbfields=None):
        return self.validate(names, defaultfmt=defaultfmt, nbfields=nbfields)


def str2bool(value):
    """
    Tries to transform a string supposed to represent a boolean to a boolean.

    Parameters
    ----------
    value : str
        The string that is transformed to a boolean.

    Returns
    -------
    boolval : bool
        The boolean representation of `value`.

    Raises
    ------
    ValueError
        If the string is not 'True' or 'False' (case independent)

    Examples
    --------
    >>> import numpy as np
    >>> np.lib._iotools.str2bool('TRUE')
    True
    >>> np.lib._iotools.str2bool('false')
    False

    """
    value = value.upper()
    if value == 'TRUE':
        return True
    elif value == 'FALSE':
        return False
    else:
        raise ValueError("Invalid boolean")


class ConverterError(Exception):
    """
    Exception raised when an error occurs in a converter for string values.

    """
    pass


class ConverterLockError(ConverterError):
    """
    Exception raised when an attempt is made to upgrade a locked converter.

    """
    pass


class ConversionWarning(UserWarning):
    """
    Warning issued when a string converter has a problem.

    Notes
    -----
    In `genfromtxt` a `ConversionWarning` is issued if raising exceptions
    is explicitly suppressed with the "invalid_raise" keyword.

    """
    pass


class StringConverter:
    """
    Factory class for function transforming a string into another object
    (int, float).

    After initialization, an instance can be called to transform a string
    into another object. If the string is recognized as representing a
    missing value, a default value is returned.

    Attributes
    ----------
    func : function
        Function used for the conversion.
    default : any
        Default value to return when the input corresponds to a missing
        value.
    type : type
        Type of the output.
    _status : int
        Integer representing the order of the conversion.
    _mapper : sequence of tuples
        Sequence of tuples (dtype, function, default value) to evaluate in
        order.
    _locked : bool
        Holds `locked` parameter.

    Parameters
    ----------
    dtype_or_func : {None, dtype, function}, optional
        If a `dtype`, specifies the input data type, used to define a basic
        function and a default value for missing data. For example, when
        `dtype` is float, the `func` attribute is set to `float` and the
        default value to `np.nan`.  If a function, this function is used to
        convert a string to another object. In this case, it is recommended
        to give an associated default value as input.
    default : any, optional
        Value to return by default, that is, when the string to be
        converted is flagged as missing. If not given, `StringConverter`
        tries to supply a reasonable default value.
    missing_values : {None, sequence of str}, optional
        ``None`` or sequence of strings indicating a missing value. If ``None``
        then missing values are indicated by empty entries. The default is
        ``None``.
    locked : bool, optional
        Whether the StringConverter should be locked to prevent automatic
        upgrade or not. Default is False.

    """
    _mapper = [(nx.bool, str2bool, False),
               (nx.int_, int, -1),]

    # On 32-bit systems, we need to make sure that we explicitly include
    # nx.int64 since ns.int_ is nx.int32.
    if nx.dtype(nx.int_).itemsize < nx.dtype(nx.int64).itemsize:
        _mapper.append((nx.int64, int, -1))

    _mapper.extend([(nx.float64, float, nx.nan),
                    (nx.complex128, complex, nx.nan + 0j),
                    (nx.longdouble, nx.longdouble, nx.nan),
                    # If a non-default dtype is passed, fall back to generic
                    # ones (should only be used for the converter)
                    (nx.integer, int, -1),
                    (nx.floating, float, nx.nan),
                    (nx.complexfloating, complex, nx.nan + 0j),
                    # Last, try with the string types (must be last, because
                    # `_mapper[-1]` is used as default in some cases)
                    (nx.str_, asunicode, '???'),
                    (nx.bytes_, asbytes, '???'),
                    ])

    @classmethod
    def _getdtype(cls, val):
        """Returns the dtype of the input variable."""
        return np.array(val).dtype

    @classmethod
    def _getsubdtype(cls, val):
        """Returns the type of the dtype of the input variable."""
        return np.array(val).dtype.type

    @classmethod
    def _dtypeortype(cls, dtype):
        """Returns dtype for datetime64 and type of dtype otherwise."""

        # This is a bit annoying. We want to return the "general" type in most
        # cases (ie. "string" rather than "S10"), but we want to return the
        # specific type for datetime64 (ie. "datetime64[us]" rather than
        # "datetime64").
        if dtype.type == np.datetime64:
            return dtype
        return dtype.type

    @classmethod
    def upgrade_mapper(cls, func, default=None):
        """
        Upgrade the mapper of a StringConverter by adding a new function and
        its corresponding default.

        The input function (or sequence of functions) and its associated
        default value (if any) is inserted in penultimate position of the
        mapper.  The corresponding type is estimated from the dtype of the
        default value.

        Parameters
        ----------
        func : var
            Function, or sequence of functions

        Examples
        --------
        >>> import dateutil.parser
        >>> import datetime
        >>> dateparser = dateutil.parser.parse
        >>> defaultdate = datetime.date(2000, 1, 1)
        >>> StringConverter.upgrade_mapper(dateparser, default=defaultdate)
        """
        # Func is a single functions
        if callable(func):
            cls._mapper.insert(-1, (cls._getsubdtype(default), func, default))
            return
        elif hasattr(func, '__iter__'):
            if isinstance(func[0], (tuple, list)):
                for _ in func:
                    cls._mapper.insert(-1, _)
                return
            if default is None:
                default = [None] * len(func)
            else:
                default = list(default)
                default.append([None] * (len(func) - len(default)))
            for fct, dft in zip(func, default):
                cls._mapper.insert(-1, (cls._getsubdtype(dft), fct, dft))

    @classmethod
    def _find_map_entry(cls, dtype):
        # if a converter for the specific dtype is available use that
        for i, (deftype, func, default_def) in enumerate(cls._mapper):
            if dtype.type == deftype:
                return i, (deftype, func, default_def)

        # otherwise find an inexact match
        for i, (deftype, func, default_def) in enumerate(cls._mapper):
            if np.issubdtype(dtype.type, deftype):
                return i, (deftype, func, default_def)

        raise LookupError

    def __init__(self, dtype_or_func=None, default=None, missing_values=None,
                 locked=False):
        # Defines a lock for upgrade
        self._locked = bool(locked)
        # No input dtype: minimal initialization
        if dtype_or_func is None:
            self.func = str2bool
            self._status = 0
            self.default = default or False
            dtype = np.dtype('bool')
        else:
            # Is the input a np.dtype ?
            try:
                self.func = None
                dtype = np.dtype(dtype_or_func)
            except TypeError:
                # dtype_or_func must be a function, then
                if not callable(dtype_or_func):
                    errmsg = ("The input argument `dtype` is neither a"
                              " function nor a dtype (got '%s' instead)")
                    raise TypeError(errmsg % type(dtype_or_func))
                # Set the function
                self.func = dtype_or_func
                # If we don't have a default, try to guess it or set it to
                # None
                if default is None:
                    try:
                        default = self.func('0')
                    except ValueError:
                        default = None
                dtype = self._getdtype(default)

            # find the best match in our mapper
            try:
                self._status, (_, func, default_def) = self._find_map_entry(dtype)
            except LookupError:
                # no match
                self.default = default
                _, func, _ = self._mapper[-1]
                self._status = 0
            else:
                # use the found default only if we did not already have one
                if default is None:
                    self.default = default_def
                else:
                    self.default = default

            # If the input was a dtype, set the function to the last we saw
            if self.func is None:
                self.func = func

            # If the status is 1 (int), change the function to
            # something more robust.
            if self.func == self._mapper[1][1]:
                if issubclass(dtype.type, np.uint64):
                    self.func = np.uint64
                elif issubclass(dtype.type, np.int64):
                    self.func = np.int64
                else:
                    self.func = lambda x: int(float(x))
        # Store the list of strings corresponding to missing values.
        if missing_values is None:
            self.missing_values = {''}
        else:
            if isinstance(missing_values, str):
                missing_values = missing_values.split(",")
            self.missing_values = set(list(missing_values) + [''])

        self._callingfunction = self._strict_call
        self.type = self._dtypeortype(dtype)
        self._checked = False
        self._initial_default = default

    def _loose_call(self, value):
        try:
            return self.func(value)
        except ValueError:
            return self.default

    def _strict_call(self, value):
        try:

            # We check if we can convert the value using the current function
            new_value = self.func(value)

            # In addition to having to check whether func can convert the
            # value, we also have to make sure that we don't get overflow
            # errors for integers.
            if self.func is int:
                try:
                    np.array(value, dtype=self.type)
                except OverflowError:
                    raise ValueError

            # We're still here so we can now return the new value
            return new_value

        except ValueError:
            if value.strip() in self.missing_values:
                if not self._status:
                    self._checked = False
                return self.default
            raise ValueError(f"Cannot convert string '{value}'")

    def __call__(self, value):
        return self._callingfunction(value)

    def _do_upgrade(self):
        # Raise an exception if we locked the converter...
        if self._locked:
            errmsg = "Converter is locked and cannot be upgraded"
            raise ConverterLockError(errmsg)
        _statusmax = len(self._mapper)
        # Complains if we try to upgrade by the maximum
        _status = self._status
        if _status == _statusmax:
            errmsg = "Could not find a valid conversion function"
            raise ConverterError(errmsg)
        elif _status < _statusmax - 1:
            _status += 1
        self.type, self.func, default = self._mapper[_status]
        self._status = _status
        if self._initial_default is not None:
            self.default = self._initial_default
        else:
            self.default = default

    def upgrade(self, value):
        """
        Find the best converter for a given string, and return the result.

        The supplied string `value` is converted by testing different
        converters in order. First the `func` method of the
        `StringConverter` instance is tried, if this fails other available
        converters are tried.  The order in which these other converters
        are tried is determined by the `_status` attribute of the instance.

        Parameters
        ----------
        value : str
            The string to convert.

        Returns
        -------
        out : any
            The result of converting `value` with the appropriate converter.

        """
        self._checked = True
        try:
            return self._strict_call(value)
        except ValueError:
            self._do_upgrade()
            return self.upgrade(value)

    def iterupgrade(self, value):
        self._checked = True
        if not hasattr(value, '__iter__'):
            value = (value,)
        _strict_call = self._strict_call
        try:
            for _m in value:
                _strict_call(_m)
        except ValueError:
            self._do_upgrade()
            self.iterupgrade(value)

    def update(self, func, default=None, testing_value=None,
               missing_values='', locked=False):
        """
        Set StringConverter attributes directly.

        Parameters
        ----------
        func : function
            Conversion function.
        default : any, optional
            Value to return by default, that is, when the string to be
            converted is flagged as missing. If not given,
            `StringConverter` tries to supply a reasonable default value.
        testing_value : str, optional
            A string representing a standard input value of the converter.
            This string is used to help defining a reasonable default
            value.
        missing_values : {sequence of str, None}, optional
            Sequence of strings indicating a missing value. If ``None``, then
            the existing `missing_values` are cleared. The default is ``''``.
        locked : bool, optional
            Whether the StringConverter should be locked to prevent
            automatic upgrade or not. Default is False.

        Notes
        -----
        `update` takes the same parameters as the constructor of
        `StringConverter`, except that `func` does not accept a `dtype`
        whereas `dtype_or_func` in the constructor does.

        """
        self.func = func
        self._locked = locked

        # Don't reset the default to None if we can avoid it
        if default is not None:
            self.default = default
            self.type = self._dtypeortype(self._getdtype(default))
        else:
            try:
                tester = func(testing_value or '1')
            except (TypeError, ValueError):
                tester = None
            self.type = self._dtypeortype(self._getdtype(tester))

        # Add the missing values to the existing set or clear it.
        if missing_values is None:
            # Clear all missing values even though the ctor initializes it to
            # set(['']) when the argument is None.
            self.missing_values = set()
        else:
            if not np.iterable(missing_values):
                missing_values = [missing_values]
            if not all(isinstance(v, str) for v in missing_values):
                raise TypeError("missing_values must be strings or unicode")
            self.missing_values.update(missing_values)


def easy_dtype(ndtype, names=None, defaultfmt="f%i", **validationargs):
    """
    Convenience function to create a `np.dtype` object.

    The function processes the input `dtype` and matches it with the given
    names.

    Parameters
    ----------
    ndtype : var
        Definition of the dtype. Can be any string or dictionary recognized
        by the `np.dtype` function, or a sequence of types.
    names : str or sequence, optional
        Sequence of strings to use as field names for a structured dtype.
        For convenience, `names` can be a string of a comma-separated list
        of names.
    defaultfmt : str, optional
        Format string used to define missing names, such as ``"f%i"``
        (default) or ``"fields_%02i"``.
    validationargs : optional
        A series of optional arguments used to initialize a
        `NameValidator`.

    Examples
    --------
    >>> import numpy as np
    >>> np.lib._iotools.easy_dtype(float)
    dtype('float64')
    >>> np.lib._iotools.easy_dtype("i4, f8")
    dtype([('f0', '<i4'), ('f1', '<f8')])
    >>> np.lib._iotools.easy_dtype("i4, f8", defaultfmt="field_%03i")
    dtype([('field_000', '<i4'), ('field_001', '<f8')])

    >>> np.lib._iotools.easy_dtype((int, float, float), names="a,b,c")
    dtype([('a', '<i8'), ('b', '<f8'), ('c', '<f8')])
    >>> np.lib._iotools.easy_dtype(float, names="a,b,c")
    dtype([('a', '<f8'), ('b', '<f8'), ('c', '<f8')])

    """
    try:
        ndtype = np.dtype(ndtype)
    except TypeError:
        validate = NameValidator(**validationargs)
        nbfields = len(ndtype)
        if names is None:
            names = [''] * len(ndtype)
        elif isinstance(names, str):
            names = names.split(",")
        names = validate(names, nbfields=nbfields, defaultfmt=defaultfmt)
        ndtype = np.dtype({"formats": ndtype, "names": names})
    else:
        # Explicit names
        if names is not None:
            validate = NameValidator(**validationargs)
            if isinstance(names, str):
                names = names.split(",")
            # Simple dtype: repeat to match the nb of names
            if ndtype.names is None:
                formats = tuple([ndtype.type] * len(names))
                names = validate(names, defaultfmt=defaultfmt)
                ndtype = np.dtype(list(zip(names, formats)))
            # Structured dtype: just validate the names as needed
            else:
                ndtype.names = validate(names, nbfields=len(ndtype.names),
                                        defaultfmt=defaultfmt)
        # No implicit names
        elif ndtype.names is not None:
            validate = NameValidator(**validationargs)
            # Default initial names : should we change the format ?
            numbered_names = tuple(f"f{i}" for i in range(len(ndtype.names)))
            if ((ndtype.names == numbered_names) and (defaultfmt != "f%i")):
                ndtype.names = validate([''] * len(ndtype.names),
                                        defaultfmt=defaultfmt)
            # Explicit initial names : just validate
            else:
                ndtype.names = validate(ndtype.names, defaultfmt=defaultfmt)
    return ndtype

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\_arraypad_impl.py ===
"""
The arraypad module contains a group of functions to pad values onto the edges
of an n-dimensional array.

"""
import numpy as np
from numpy._core.overrides import array_function_dispatch
from numpy.lib._index_tricks_impl import ndindex

__all__ = ['pad']


###############################################################################
# Private utility functions.


def _round_if_needed(arr, dtype):
    """
    Rounds arr inplace if destination dtype is integer.

    Parameters
    ----------
    arr : ndarray
        Input array.
    dtype : dtype
        The dtype of the destination array.
    """
    if np.issubdtype(dtype, np.integer):
        arr.round(out=arr)


def _slice_at_axis(sl, axis):
    """
    Construct tuple of slices to slice an array in the given dimension.

    Parameters
    ----------
    sl : slice
        The slice for the given dimension.
    axis : int
        The axis to which `sl` is applied. All other dimensions are left
        "unsliced".

    Returns
    -------
    sl : tuple of slices
        A tuple with slices matching `shape` in length.

    Examples
    --------
    >>> np._slice_at_axis(slice(None, 3, -1), 1)
    (slice(None, None, None), slice(None, 3, -1), (...,))
    """
    return (slice(None),) * axis + (sl,) + (...,)


def _view_roi(array, original_area_slice, axis):
    """
    Get a view of the current region of interest during iterative padding.

    When padding multiple dimensions iteratively corner values are
    unnecessarily overwritten multiple times. This function reduces the
    working area for the first dimensions so that corners are excluded.

    Parameters
    ----------
    array : ndarray
        The array with the region of interest.
    original_area_slice : tuple of slices
        Denotes the area with original values of the unpadded array.
    axis : int
        The currently padded dimension assuming that `axis` is padded before
        `axis` + 1.

    Returns
    -------
    roi : ndarray
        The region of interest of the original `array`.
    """
    axis += 1
    sl = (slice(None),) * axis + original_area_slice[axis:]
    return array[sl]


def _pad_simple(array, pad_width, fill_value=None):
    """
    Pad array on all sides with either a single value or undefined values.

    Parameters
    ----------
    array : ndarray
        Array to grow.
    pad_width : sequence of tuple[int, int]
        Pad width on both sides for each dimension in `arr`.
    fill_value : scalar, optional
        If provided the padded area is filled with this value, otherwise
        the pad area left undefined.

    Returns
    -------
    padded : ndarray
        The padded array with the same dtype as`array`. Its order will default
        to C-style if `array` is not F-contiguous.
    original_area_slice : tuple
        A tuple of slices pointing to the area of the original array.
    """
    # Allocate grown array
    new_shape = tuple(
        left + size + right
        for size, (left, right) in zip(array.shape, pad_width)
    )
    order = 'F' if array.flags.fnc else 'C'  # Fortran and not also C-order
    padded = np.empty(new_shape, dtype=array.dtype, order=order)

    if fill_value is not None:
        padded.fill(fill_value)

    # Copy old array into correct space
    original_area_slice = tuple(
        slice(left, left + size)
        for size, (left, right) in zip(array.shape, pad_width)
    )
    padded[original_area_slice] = array

    return padded, original_area_slice


def _set_pad_area(padded, axis, width_pair, value_pair):
    """
    Set empty-padded area in given dimension.

    Parameters
    ----------
    padded : ndarray
        Array with the pad area which is modified inplace.
    axis : int
        Dimension with the pad area to set.
    width_pair : (int, int)
        Pair of widths that mark the pad area on both sides in the given
        dimension.
    value_pair : tuple of scalars or ndarrays
        Values inserted into the pad area on each side. It must match or be
        broadcastable to the shape of `arr`.
    """
    left_slice = _slice_at_axis(slice(None, width_pair[0]), axis)
    padded[left_slice] = value_pair[0]

    right_slice = _slice_at_axis(
        slice(padded.shape[axis] - width_pair[1], None), axis)
    padded[right_slice] = value_pair[1]


def _get_edges(padded, axis, width_pair):
    """
    Retrieve edge values from empty-padded array in given dimension.

    Parameters
    ----------
    padded : ndarray
        Empty-padded array.
    axis : int
        Dimension in which the edges are considered.
    width_pair : (int, int)
        Pair of widths that mark the pad area on both sides in the given
        dimension.

    Returns
    -------
    left_edge, right_edge : ndarray
        Edge values of the valid area in `padded` in the given dimension. Its
        shape will always match `padded` except for the dimension given by
        `axis` which will have a length of 1.
    """
    left_index = width_pair[0]
    left_slice = _slice_at_axis(slice(left_index, left_index + 1), axis)
    left_edge = padded[left_slice]

    right_index = padded.shape[axis] - width_pair[1]
    right_slice = _slice_at_axis(slice(right_index - 1, right_index), axis)
    right_edge = padded[right_slice]

    return left_edge, right_edge


def _get_linear_ramps(padded, axis, width_pair, end_value_pair):
    """
    Construct linear ramps for empty-padded array in given dimension.

    Parameters
    ----------
    padded : ndarray
        Empty-padded array.
    axis : int
        Dimension in which the ramps are constructed.
    width_pair : (int, int)
        Pair of widths that mark the pad area on both sides in the given
        dimension.
    end_value_pair : (scalar, scalar)
        End values for the linear ramps which form the edge of the fully padded
        array. These values are included in the linear ramps.

    Returns
    -------
    left_ramp, right_ramp : ndarray
        Linear ramps to set on both sides of `padded`.
    """
    edge_pair = _get_edges(padded, axis, width_pair)

    left_ramp, right_ramp = (
        np.linspace(
            start=end_value,
            stop=edge.squeeze(axis),  # Dimension is replaced by linspace
            num=width,
            endpoint=False,
            dtype=padded.dtype,
            axis=axis
        )
        for end_value, edge, width in zip(
            end_value_pair, edge_pair, width_pair
        )
    )

    # Reverse linear space in appropriate dimension
    right_ramp = right_ramp[_slice_at_axis(slice(None, None, -1), axis)]

    return left_ramp, right_ramp


def _get_stats(padded, axis, width_pair, length_pair, stat_func):
    """
    Calculate statistic for the empty-padded array in given dimension.

    Parameters
    ----------
    padded : ndarray
        Empty-padded array.
    axis : int
        Dimension in which the statistic is calculated.
    width_pair : (int, int)
        Pair of widths that mark the pad area on both sides in the given
        dimension.
    length_pair : 2-element sequence of None or int
        Gives the number of values in valid area from each side that is
        taken into account when calculating the statistic. If None the entire
        valid area in `padded` is considered.
    stat_func : function
        Function to compute statistic. The expected signature is
        ``stat_func(x: ndarray, axis: int, keepdims: bool) -> ndarray``.

    Returns
    -------
    left_stat, right_stat : ndarray
        Calculated statistic for both sides of `padded`.
    """
    # Calculate indices of the edges of the area with original values
    left_index = width_pair[0]
    right_index = padded.shape[axis] - width_pair[1]
    # as well as its length
    max_length = right_index - left_index

    # Limit stat_lengths to max_length
    left_length, right_length = length_pair
    if left_length is None or max_length < left_length:
        left_length = max_length
    if right_length is None or max_length < right_length:
        right_length = max_length

    if (left_length == 0 or right_length == 0) \
            and stat_func in {np.amax, np.amin}:
        # amax and amin can't operate on an empty array,
        # raise a more descriptive warning here instead of the default one
        raise ValueError("stat_length of 0 yields no value for padding")

    # Calculate statistic for the left side
    left_slice = _slice_at_axis(
        slice(left_index, left_index + left_length), axis)
    left_chunk = padded[left_slice]
    left_stat = stat_func(left_chunk, axis=axis, keepdims=True)
    _round_if_needed(left_stat, padded.dtype)

    if left_length == right_length == max_length:
        # return early as right_stat must be identical to left_stat
        return left_stat, left_stat

    # Calculate statistic for the right side
    right_slice = _slice_at_axis(
        slice(right_index - right_length, right_index), axis)
    right_chunk = padded[right_slice]
    right_stat = stat_func(right_chunk, axis=axis, keepdims=True)
    _round_if_needed(right_stat, padded.dtype)

    return left_stat, right_stat


def _set_reflect_both(padded, axis, width_pair, method,
                      original_period, include_edge=False):
    """
    Pad `axis` of `arr` with reflection.

    Parameters
    ----------
    padded : ndarray
        Input array of arbitrary shape.
    axis : int
        Axis along which to pad `arr`.
    width_pair : (int, int)
        Pair of widths that mark the pad area on both sides in the given
        dimension.
    method : str
        Controls method of reflection; options are 'even' or 'odd'.
    original_period : int
        Original length of data on `axis` of `arr`.
    include_edge : bool
        If true, edge value is included in reflection, otherwise the edge
        value forms the symmetric axis to the reflection.

    Returns
    -------
    pad_amt : tuple of ints, length 2
        New index positions of padding to do along the `axis`. If these are
        both 0, padding is done in this dimension.
    """
    left_pad, right_pad = width_pair
    old_length = padded.shape[axis] - right_pad - left_pad

    if include_edge:
        # Avoid wrapping with only a subset of the original area
        # by ensuring period can only be a multiple of the original
        # area's length.
        old_length = old_length // original_period * original_period
        # Edge is included, we need to offset the pad amount by 1
        edge_offset = 1
    else:
        # Avoid wrapping with only a subset of the original area
        # by ensuring period can only be a multiple of the original
        # area's length.
        old_length = ((old_length - 1) // (original_period - 1)
            * (original_period - 1) + 1)
        edge_offset = 0  # Edge is not included, no need to offset pad amount
        old_length -= 1  # but must be omitted from the chunk

    if left_pad > 0:
        # Pad with reflected values on left side:
        # First limit chunk size which can't be larger than pad area
        chunk_length = min(old_length, left_pad)
        # Slice right to left, stop on or next to edge, start relative to stop
        stop = left_pad - edge_offset
        start = stop + chunk_length
        left_slice = _slice_at_axis(slice(start, stop, -1), axis)
        left_chunk = padded[left_slice]

        if method == "odd":
            # Negate chunk and align with edge
            edge_slice = _slice_at_axis(slice(left_pad, left_pad + 1), axis)
            left_chunk = 2 * padded[edge_slice] - left_chunk

        # Insert chunk into padded area
        start = left_pad - chunk_length
        stop = left_pad
        pad_area = _slice_at_axis(slice(start, stop), axis)
        padded[pad_area] = left_chunk
        # Adjust pointer to left edge for next iteration
        left_pad -= chunk_length

    if right_pad > 0:
        # Pad with reflected values on right side:
        # First limit chunk size which can't be larger than pad area
        chunk_length = min(old_length, right_pad)
        # Slice right to left, start on or next to edge, stop relative to start
        start = -right_pad + edge_offset - 2
        stop = start - chunk_length
        right_slice = _slice_at_axis(slice(start, stop, -1), axis)
        right_chunk = padded[right_slice]

        if method == "odd":
            # Negate chunk and align with edge
            edge_slice = _slice_at_axis(
                slice(-right_pad - 1, -right_pad), axis)
            right_chunk = 2 * padded[edge_slice] - right_chunk

        # Insert chunk into padded area
        start = padded.shape[axis] - right_pad
        stop = start + chunk_length
        pad_area = _slice_at_axis(slice(start, stop), axis)
        padded[pad_area] = right_chunk
        # Adjust pointer to right edge for next iteration
        right_pad -= chunk_length

    return left_pad, right_pad


def _set_wrap_both(padded, axis, width_pair, original_period):
    """
    Pad `axis` of `arr` with wrapped values.

    Parameters
    ----------
    padded : ndarray
        Input array of arbitrary shape.
    axis : int
        Axis along which to pad `arr`.
    width_pair : (int, int)
        Pair of widths that mark the pad area on both sides in the given
        dimension.
    original_period : int
        Original length of data on `axis` of `arr`.

    Returns
    -------
    pad_amt : tuple of ints, length 2
        New index positions of padding to do along the `axis`. If these are
        both 0, padding is done in this dimension.
    """
    left_pad, right_pad = width_pair
    period = padded.shape[axis] - right_pad - left_pad
    # Avoid wrapping with only a subset of the original area by ensuring period
    # can only be a multiple of the original area's length.
    period = period // original_period * original_period

    # If the current dimension of `arr` doesn't contain enough valid values
    # (not part of the undefined pad area) we need to pad multiple times.
    # Each time the pad area shrinks on both sides which is communicated with
    # these variables.
    new_left_pad = 0
    new_right_pad = 0

    if left_pad > 0:
        # Pad with wrapped values on left side
        # First slice chunk from left side of the non-pad area.
        # Use min(period, left_pad) to ensure that chunk is not larger than
        # pad area.
        slice_end = left_pad + period
        slice_start = slice_end - min(period, left_pad)
        right_slice = _slice_at_axis(slice(slice_start, slice_end), axis)
        right_chunk = padded[right_slice]

        if left_pad > period:
            # Chunk is smaller than pad area
            pad_area = _slice_at_axis(slice(left_pad - period, left_pad), axis)
            new_left_pad = left_pad - period
        else:
            # Chunk matches pad area
            pad_area = _slice_at_axis(slice(None, left_pad), axis)
        padded[pad_area] = right_chunk

    if right_pad > 0:
        # Pad with wrapped values on right side
        # First slice chunk from right side of the non-pad area.
        # Use min(period, right_pad) to ensure that chunk is not larger than
        # pad area.
        slice_start = -right_pad - period
        slice_end = slice_start + min(period, right_pad)
        left_slice = _slice_at_axis(slice(slice_start, slice_end), axis)
        left_chunk = padded[left_slice]

        if right_pad > period:
            # Chunk is smaller than pad area
            pad_area = _slice_at_axis(
                slice(-right_pad, -right_pad + period), axis)
            new_right_pad = right_pad - period
        else:
            # Chunk matches pad area
            pad_area = _slice_at_axis(slice(-right_pad, None), axis)
        padded[pad_area] = left_chunk

    return new_left_pad, new_right_pad


def _as_pairs(x, ndim, as_index=False):
    """
    Broadcast `x` to an array with the shape (`ndim`, 2).

    A helper function for `pad` that prepares and validates arguments like
    `pad_width` for iteration in pairs.

    Parameters
    ----------
    x : {None, scalar, array-like}
        The object to broadcast to the shape (`ndim`, 2).
    ndim : int
        Number of pairs the broadcasted `x` will have.
    as_index : bool, optional
        If `x` is not None, try to round each element of `x` to an integer
        (dtype `np.intp`) and ensure every element is positive.

    Returns
    -------
    pairs : nested iterables, shape (`ndim`, 2)
        The broadcasted version of `x`.

    Raises
    ------
    ValueError
        If `as_index` is True and `x` contains negative elements.
        Or if `x` is not broadcastable to the shape (`ndim`, 2).
    """
    if x is None:
        # Pass through None as a special case, otherwise np.round(x) fails
        # with an AttributeError
        return ((None, None),) * ndim

    x = np.array(x)
    if as_index:
        x = np.round(x).astype(np.intp, copy=False)

    if x.ndim < 3:
        # Optimization: Possibly use faster paths for cases where `x` has
        # only 1 or 2 elements. `np.broadcast_to` could handle these as well
        # but is currently slower

        if x.size == 1:
            # x was supplied as a single value
            x = x.ravel()  # Ensure x[0] works for x.ndim == 0, 1, 2
            if as_index and x < 0:
                raise ValueError("index can't contain negative values")
            return ((x[0], x[0]),) * ndim

        if x.size == 2 and x.shape != (2, 1):
            # x was supplied with a single value for each side
            # but except case when each dimension has a single value
            # which should be broadcasted to a pair,
            # e.g. [[1], [2]] -> [[1, 1], [2, 2]] not [[1, 2], [1, 2]]
            x = x.ravel()  # Ensure x[0], x[1] works
            if as_index and (x[0] < 0 or x[1] < 0):
                raise ValueError("index can't contain negative values")
            return ((x[0], x[1]),) * ndim

    if as_index and x.min() < 0:
        raise ValueError("index can't contain negative values")

    # Converting the array with `tolist` seems to improve performance
    # when iterating and indexing the result (see usage in `pad`)
    return np.broadcast_to(x, (ndim, 2)).tolist()


def _pad_dispatcher(array, pad_width, mode=None, **kwargs):
    return (array,)


###############################################################################
# Public functions


@array_function_dispatch(_pad_dispatcher, module='numpy')
def pad(array, pad_width, mode='constant', **kwargs):
    """
    Pad an array.

    Parameters
    ----------
    array : array_like of rank N
        The array to pad.
    pad_width : {sequence, array_like, int}
        Number of values padded to the edges of each axis.
        ``((before_1, after_1), ... (before_N, after_N))`` unique pad widths
        for each axis.
        ``(before, after)`` or ``((before, after),)`` yields same before
        and after pad for each axis.
        ``(pad,)`` or ``int`` is a shortcut for before = after = pad width
        for all axes.
    mode : str or function, optional
        One of the following string values or a user supplied function.

        'constant' (default)
            Pads with a constant value.
        'edge'
            Pads with the edge values of array.
        'linear_ramp'
            Pads with the linear ramp between end_value and the
            array edge value.
        'maximum'
            Pads with the maximum value of all or part of the
            vector along each axis.
        'mean'
            Pads with the mean value of all or part of the
            vector along each axis.
        'median'
            Pads with the median value of all or part of the
            vector along each axis.
        'minimum'
            Pads with the minimum value of all or part of the
            vector along each axis.
        'reflect'
            Pads with the reflection of the vector mirrored on
            the first and last values of the vector along each
            axis.
        'symmetric'
            Pads with the reflection of the vector mirrored
            along the edge of the array.
        'wrap'
            Pads with the wrap of the vector along the axis.
            The first values are used to pad the end and the
            end values are used to pad the beginning.
        'empty'
            Pads with undefined values.

        <function>
            Padding function, see Notes.
    stat_length : sequence or int, optional
        Used in 'maximum', 'mean', 'median', and 'minimum'.  Number of
        values at edge of each axis used to calculate the statistic value.

        ``((before_1, after_1), ... (before_N, after_N))`` unique statistic
        lengths for each axis.

        ``(before, after)`` or ``((before, after),)`` yields same before
        and after statistic lengths for each axis.

        ``(stat_length,)`` or ``int`` is a shortcut for
        ``before = after = statistic`` length for all axes.

        Default is ``None``, to use the entire axis.
    constant_values : sequence or scalar, optional
        Used in 'constant'.  The values to set the padded values for each
        axis.

        ``((before_1, after_1), ... (before_N, after_N))`` unique pad constants
        for each axis.

        ``(before, after)`` or ``((before, after),)`` yields same before
        and after constants for each axis.

        ``(constant,)`` or ``constant`` is a shortcut for
        ``before = after = constant`` for all axes.

        Default is 0.
    end_values : sequence or scalar, optional
        Used in 'linear_ramp'.  The values used for the ending value of the
        linear_ramp and that will form the edge of the padded array.

        ``((before_1, after_1), ... (before_N, after_N))`` unique end values
        for each axis.

        ``(before, after)`` or ``((before, after),)`` yields same before
        and after end values for each axis.

        ``(constant,)`` or ``constant`` is a shortcut for
        ``before = after = constant`` for all axes.

        Default is 0.
    reflect_type : {'even', 'odd'}, optional
        Used in 'reflect', and 'symmetric'.  The 'even' style is the
        default with an unaltered reflection around the edge value.  For
        the 'odd' style, the extended part of the array is created by
        subtracting the reflected values from two times the edge value.

    Returns
    -------
    pad : ndarray
        Padded array of rank equal to `array` with shape increased
        according to `pad_width`.

    Notes
    -----
    For an array with rank greater than 1, some of the padding of later
    axes is calculated from padding of previous axes.  This is easiest to
    think about with a rank 2 array where the corners of the padded array
    are calculated by using padded values from the first axis.

    The padding function, if used, should modify a rank 1 array in-place. It
    has the following signature::

        padding_func(vector, iaxis_pad_width, iaxis, kwargs)

    where

    vector : ndarray
        A rank 1 array already padded with zeros.  Padded values are
        vector[:iaxis_pad_width[0]] and vector[-iaxis_pad_width[1]:].
    iaxis_pad_width : tuple
        A 2-tuple of ints, iaxis_pad_width[0] represents the number of
        values padded at the beginning of vector where
        iaxis_pad_width[1] represents the number of values padded at
        the end of vector.
    iaxis : int
        The axis currently being calculated.
    kwargs : dict
        Any keyword arguments the function requires.

    Examples
    --------
    >>> import numpy as np
    >>> a = [1, 2, 3, 4, 5]
    >>> np.pad(a, (2, 3), 'constant', constant_values=(4, 6))
    array([4, 4, 1, ..., 6, 6, 6])

    >>> np.pad(a, (2, 3), 'edge')
    array([1, 1, 1, ..., 5, 5, 5])

    >>> np.pad(a, (2, 3), 'linear_ramp', end_values=(5, -4))
    array([ 5,  3,  1,  2,  3,  4,  5,  2, -1, -4])

    >>> np.pad(a, (2,), 'maximum')
    array([5, 5, 1, 2, 3, 4, 5, 5, 5])

    >>> np.pad(a, (2,), 'mean')
    array([3, 3, 1, 2, 3, 4, 5, 3, 3])

    >>> np.pad(a, (2,), 'median')
    array([3, 3, 1, 2, 3, 4, 5, 3, 3])

    >>> a = [[1, 2], [3, 4]]
    >>> np.pad(a, ((3, 2), (2, 3)), 'minimum')
    array([[1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1],
           [3, 3, 3, 4, 3, 3, 3],
           [1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1]])

    >>> a = [1, 2, 3, 4, 5]
    >>> np.pad(a, (2, 3), 'reflect')
    array([3, 2, 1, 2, 3, 4, 5, 4, 3, 2])

    >>> np.pad(a, (2, 3), 'reflect', reflect_type='odd')
    array([-1,  0,  1,  2,  3,  4,  5,  6,  7,  8])

    >>> np.pad(a, (2, 3), 'symmetric')
    array([2, 1, 1, 2, 3, 4, 5, 5, 4, 3])

    >>> np.pad(a, (2, 3), 'symmetric', reflect_type='odd')
    array([0, 1, 1, 2, 3, 4, 5, 5, 6, 7])

    >>> np.pad(a, (2, 3), 'wrap')
    array([4, 5, 1, 2, 3, 4, 5, 1, 2, 3])

    >>> def pad_with(vector, pad_width, iaxis, kwargs):
    ...     pad_value = kwargs.get('padder', 10)
    ...     vector[:pad_width[0]] = pad_value
    ...     vector[-pad_width[1]:] = pad_value
    >>> a = np.arange(6)
    >>> a = a.reshape((2, 3))
    >>> np.pad(a, 2, pad_with)
    array([[10, 10, 10, 10, 10, 10, 10],
           [10, 10, 10, 10, 10, 10, 10],
           [10, 10,  0,  1,  2, 10, 10],
           [10, 10,  3,  4,  5, 10, 10],
           [10, 10, 10, 10, 10, 10, 10],
           [10, 10, 10, 10, 10, 10, 10]])
    >>> np.pad(a, 2, pad_with, padder=100)
    array([[100, 100, 100, 100, 100, 100, 100],
           [100, 100, 100, 100, 100, 100, 100],
           [100, 100,   0,   1,   2, 100, 100],
           [100, 100,   3,   4,   5, 100, 100],
           [100, 100, 100, 100, 100, 100, 100],
           [100, 100, 100, 100, 100, 100, 100]])
    """
    array = np.asarray(array)
    pad_width = np.asarray(pad_width)

    if not pad_width.dtype.kind == 'i':
        raise TypeError('`pad_width` must be of integral type.')

    # Broadcast to shape (array.ndim, 2)
    pad_width = _as_pairs(pad_width, array.ndim, as_index=True)

    if callable(mode):
        # Old behavior: Use user-supplied function with np.apply_along_axis
        function = mode
        # Create a new zero padded array
        padded, _ = _pad_simple(array, pad_width, fill_value=0)
        # And apply along each axis

        for axis in range(padded.ndim):
            # Iterate using ndindex as in apply_along_axis, but assuming that
            # function operates inplace on the padded array.

            # view with the iteration axis at the end
            view = np.moveaxis(padded, axis, -1)

            # compute indices for the iteration axes, and append a trailing
            # ellipsis to prevent 0d arrays decaying to scalars (gh-8642)
            inds = ndindex(view.shape[:-1])
            inds = (ind + (Ellipsis,) for ind in inds)
            for ind in inds:
                function(view[ind], pad_width[axis], axis, kwargs)

        return padded

    # Make sure that no unsupported keywords were passed for the current mode
    allowed_kwargs = {
        'empty': [], 'edge': [], 'wrap': [],
        'constant': ['constant_values'],
        'linear_ramp': ['end_values'],
        'maximum': ['stat_length'],
        'mean': ['stat_length'],
        'median': ['stat_length'],
        'minimum': ['stat_length'],
        'reflect': ['reflect_type'],
        'symmetric': ['reflect_type'],
    }
    try:
        unsupported_kwargs = set(kwargs) - set(allowed_kwargs[mode])
    except KeyError:
        raise ValueError(f"mode '{mode}' is not supported") from None
    if unsupported_kwargs:
        raise ValueError("unsupported keyword arguments for mode "
                         f"'{mode}': {unsupported_kwargs}")

    stat_functions = {"maximum": np.amax, "minimum": np.amin,
                      "mean": np.mean, "median": np.median}

    # Create array with final shape and original values
    # (padded area is undefined)
    padded, original_area_slice = _pad_simple(array, pad_width)
    # And prepare iteration over all dimensions
    # (zipping may be more readable than using enumerate)
    axes = range(padded.ndim)

    if mode == "constant":
        values = kwargs.get("constant_values", 0)
        values = _as_pairs(values, padded.ndim)
        for axis, width_pair, value_pair in zip(axes, pad_width, values):
            roi = _view_roi(padded, original_area_slice, axis)
            _set_pad_area(roi, axis, width_pair, value_pair)

    elif mode == "empty":
        pass  # Do nothing as _pad_simple already returned the correct result

    elif array.size == 0:
        # Only modes "constant" and "empty" can extend empty axes, all other
        # modes depend on `array` not being empty
        # -> ensure every empty axis is only "padded with 0"
        for axis, width_pair in zip(axes, pad_width):
            if array.shape[axis] == 0 and any(width_pair):
                raise ValueError(
                    f"can't extend empty axis {axis} using modes other than "
                    "'constant' or 'empty'"
                )
        # passed, don't need to do anything more as _pad_simple already
        # returned the correct result

    elif mode == "edge":
        for axis, width_pair in zip(axes, pad_width):
            roi = _view_roi(padded, original_area_slice, axis)
            edge_pair = _get_edges(roi, axis, width_pair)
            _set_pad_area(roi, axis, width_pair, edge_pair)

    elif mode == "linear_ramp":
        end_values = kwargs.get("end_values", 0)
        end_values = _as_pairs(end_values, padded.ndim)
        for axis, width_pair, value_pair in zip(axes, pad_width, end_values):
            roi = _view_roi(padded, original_area_slice, axis)
            ramp_pair = _get_linear_ramps(roi, axis, width_pair, value_pair)
            _set_pad_area(roi, axis, width_pair, ramp_pair)

    elif mode in stat_functions:
        func = stat_functions[mode]
        length = kwargs.get("stat_length")
        length = _as_pairs(length, padded.ndim, as_index=True)
        for axis, width_pair, length_pair in zip(axes, pad_width, length):
            roi = _view_roi(padded, original_area_slice, axis)
            stat_pair = _get_stats(roi, axis, width_pair, length_pair, func)
            _set_pad_area(roi, axis, width_pair, stat_pair)

    elif mode in {"reflect", "symmetric"}:
        method = kwargs.get("reflect_type", "even")
        include_edge = mode == "symmetric"
        for axis, (left_index, right_index) in zip(axes, pad_width):
            if array.shape[axis] == 1 and (left_index > 0 or right_index > 0):
                # Extending singleton dimension for 'reflect' is legacy
                # behavior; it really should raise an error.
                edge_pair = _get_edges(padded, axis, (left_index, right_index))
                _set_pad_area(
                    padded, axis, (left_index, right_index), edge_pair)
                continue

            roi = _view_roi(padded, original_area_slice, axis)
            while left_index > 0 or right_index > 0:
                # Iteratively pad until dimension is filled with reflected
                # values. This is necessary if the pad area is larger than
                # the length of the original values in the current dimension.
                left_index, right_index = _set_reflect_both(
                    roi, axis, (left_index, right_index),
                    method, array.shape[axis], include_edge
                )

    elif mode == "wrap":
        for axis, (left_index, right_index) in zip(axes, pad_width):
            roi = _view_roi(padded, original_area_slice, axis)
            original_period = padded.shape[axis] - right_index - left_index
            while left_index > 0 or right_index > 0:
                # Iteratively pad until dimension is filled with wrapped
                # values. This is necessary if the pad area is larger than
                # the length of the original values in the current dimension.
                left_index, right_index = _set_wrap_both(
                    roi, axis, (left_index, right_index), original_period)

    return padded

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\tests\test_histograms.py ===
import pytest

import numpy as np
from numpy import histogram, histogram_bin_edges, histogramdd
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_array_max_ulp,
    assert_equal,
    assert_raises,
    assert_raises_regex,
    suppress_warnings,
)


class TestHistogram:

    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    def test_simple(self):
        n = 100
        v = np.random.rand(n)
        (a, b) = histogram(v)
        # check if the sum of the bins equals the number of samples
        assert_equal(np.sum(a, axis=0), n)
        # check that the bin counts are evenly spaced when the data is from
        # a linear function
        (a, b) = histogram(np.linspace(0, 10, 100))
        assert_array_equal(a, 10)

    def test_one_bin(self):
        # Ticket 632
        hist, edges = histogram([1, 2, 3, 4], [1, 2])
        assert_array_equal(hist, [2, ])
        assert_array_equal(edges, [1, 2])
        assert_raises(ValueError, histogram, [1, 2], bins=0)
        h, e = histogram([1, 2], bins=1)
        assert_equal(h, np.array([2]))
        assert_allclose(e, np.array([1., 2.]))

    def test_density(self):
        # Check that the integral of the density equals 1.
        n = 100
        v = np.random.rand(n)
        a, b = histogram(v, density=True)
        area = np.sum(a * np.diff(b))
        assert_almost_equal(area, 1)

        # Check with non-constant bin widths
        v = np.arange(10)
        bins = [0, 1, 3, 6, 10]
        a, b = histogram(v, bins, density=True)
        assert_array_equal(a, .1)
        assert_equal(np.sum(a * np.diff(b)), 1)

        # Test that passing False works too
        a, b = histogram(v, bins, density=False)
        assert_array_equal(a, [1, 2, 3, 4])

        # Variable bin widths are especially useful to deal with
        # infinities.
        v = np.arange(10)
        bins = [0, 1, 3, 6, np.inf]
        a, b = histogram(v, bins, density=True)
        assert_array_equal(a, [.1, .1, .1, 0.])

        # Taken from a bug report from N. Becker on the numpy-discussion
        # mailing list Aug. 6, 2010.
        counts, dmy = np.histogram(
            [1, 2, 3, 4], [0.5, 1.5, np.inf], density=True)
        assert_equal(counts, [.25, 0])

    def test_outliers(self):
        # Check that outliers are not tallied
        a = np.arange(10) + .5

        # Lower outliers
        h, b = histogram(a, range=[0, 9])
        assert_equal(h.sum(), 9)

        # Upper outliers
        h, b = histogram(a, range=[1, 10])
        assert_equal(h.sum(), 9)

        # Normalization
        h, b = histogram(a, range=[1, 9], density=True)
        assert_almost_equal((h * np.diff(b)).sum(), 1, decimal=15)

        # Weights
        w = np.arange(10) + .5
        h, b = histogram(a, range=[1, 9], weights=w, density=True)
        assert_equal((h * np.diff(b)).sum(), 1)

        h, b = histogram(a, bins=8, range=[1, 9], weights=w)
        assert_equal(h, w[1:-1])

    def test_arr_weights_mismatch(self):
        a = np.arange(10) + .5
        w = np.arange(11) + .5
        with assert_raises_regex(ValueError, "same shape as"):
            h, b = histogram(a, range=[1, 9], weights=w, density=True)

    def test_type(self):
        # Check the type of the returned histogram
        a = np.arange(10) + .5
        h, b = histogram(a)
        assert_(np.issubdtype(h.dtype, np.integer))

        h, b = histogram(a, density=True)
        assert_(np.issubdtype(h.dtype, np.floating))

        h, b = histogram(a, weights=np.ones(10, int))
        assert_(np.issubdtype(h.dtype, np.integer))

        h, b = histogram(a, weights=np.ones(10, float))
        assert_(np.issubdtype(h.dtype, np.floating))

    def test_f32_rounding(self):
        # gh-4799, check that the rounding of the edges works with float32
        x = np.array([276.318359, -69.593948, 21.329449], dtype=np.float32)
        y = np.array([5005.689453, 4481.327637, 6010.369629], dtype=np.float32)
        counts_hist, xedges, yedges = np.histogram2d(x, y, bins=100)
        assert_equal(counts_hist.sum(), 3.)

    def test_bool_conversion(self):
        # gh-12107
        # Reference integer histogram
        a = np.array([1, 1, 0], dtype=np.uint8)
        int_hist, int_edges = np.histogram(a)

        # Should raise an warning on booleans
        # Ensure that the histograms are equivalent, need to suppress
        # the warnings to get the actual outputs
        with suppress_warnings() as sup:
            rec = sup.record(RuntimeWarning, 'Converting input from .*')
            hist, edges = np.histogram([True, True, False])
            # A warning should be issued
            assert_equal(len(rec), 1)
            assert_array_equal(hist, int_hist)
            assert_array_equal(edges, int_edges)

    def test_weights(self):
        v = np.random.rand(100)
        w = np.ones(100) * 5
        a, b = histogram(v)
        na, nb = histogram(v, density=True)
        wa, wb = histogram(v, weights=w)
        nwa, nwb = histogram(v, weights=w, density=True)
        assert_array_almost_equal(a * 5, wa)
        assert_array_almost_equal(na, nwa)

        # Check weights are properly applied.
        v = np.linspace(0, 10, 10)
        w = np.concatenate((np.zeros(5), np.ones(5)))
        wa, wb = histogram(v, bins=np.arange(11), weights=w)
        assert_array_almost_equal(wa, w)

        # Check with integer weights
        wa, wb = histogram([1, 2, 2, 4], bins=4, weights=[4, 3, 2, 1])
        assert_array_equal(wa, [4, 5, 0, 1])
        wa, wb = histogram(
            [1, 2, 2, 4], bins=4, weights=[4, 3, 2, 1], density=True)
        assert_array_almost_equal(wa, np.array([4, 5, 0, 1]) / 10. / 3. * 4)

        # Check weights with non-uniform bin widths
        a, b = histogram(
            np.arange(9), [0, 1, 3, 6, 10],
            weights=[2, 1, 1, 1, 1, 1, 1, 1, 1], density=True)
        assert_almost_equal(a, [.2, .1, .1, .075])

    def test_exotic_weights(self):

        # Test the use of weights that are not integer or floats, but e.g.
        # complex numbers or object types.

        # Complex weights
        values = np.array([1.3, 2.5, 2.3])
        weights = np.array([1, -1, 2]) + 1j * np.array([2, 1, 2])

        # Check with custom bins
        wa, wb = histogram(values, bins=[0, 2, 3], weights=weights)
        assert_array_almost_equal(wa, np.array([1, 1]) + 1j * np.array([2, 3]))

        # Check with even bins
        wa, wb = histogram(values, bins=2, range=[1, 3], weights=weights)
        assert_array_almost_equal(wa, np.array([1, 1]) + 1j * np.array([2, 3]))

        # Decimal weights
        from decimal import Decimal
        values = np.array([1.3, 2.5, 2.3])
        weights = np.array([Decimal(1), Decimal(2), Decimal(3)])

        # Check with custom bins
        wa, wb = histogram(values, bins=[0, 2, 3], weights=weights)
        assert_array_almost_equal(wa, [Decimal(1), Decimal(5)])

        # Check with even bins
        wa, wb = histogram(values, bins=2, range=[1, 3], weights=weights)
        assert_array_almost_equal(wa, [Decimal(1), Decimal(5)])

    def test_no_side_effects(self):
        # This is a regression test that ensures that values passed to
        # ``histogram`` are unchanged.
        values = np.array([1.3, 2.5, 2.3])
        np.histogram(values, range=[-10, 10], bins=100)
        assert_array_almost_equal(values, [1.3, 2.5, 2.3])

    def test_empty(self):
        a, b = histogram([], bins=([0, 1]))
        assert_array_equal(a, np.array([0]))
        assert_array_equal(b, np.array([0, 1]))

    def test_error_binnum_type(self):
        # Tests if right Error is raised if bins argument is float
        vals = np.linspace(0.0, 1.0, num=100)
        histogram(vals, 5)
        assert_raises(TypeError, histogram, vals, 2.4)

    def test_finite_range(self):
        # Normal ranges should be fine
        vals = np.linspace(0.0, 1.0, num=100)
        histogram(vals, range=[0.25, 0.75])
        assert_raises(ValueError, histogram, vals, range=[np.nan, 0.75])
        assert_raises(ValueError, histogram, vals, range=[0.25, np.inf])

    def test_invalid_range(self):
        # start of range must be < end of range
        vals = np.linspace(0.0, 1.0, num=100)
        with assert_raises_regex(ValueError, "max must be larger than"):
            np.histogram(vals, range=[0.1, 0.01])

    def test_bin_edge_cases(self):
        # Ensure that floating-point computations correctly place edge cases.
        arr = np.array([337, 404, 739, 806, 1007, 1811, 2012])
        hist, edges = np.histogram(arr, bins=8296, range=(2, 2280))
        mask = hist > 0
        left_edges = edges[:-1][mask]
        right_edges = edges[1:][mask]
        for x, left, right in zip(arr, left_edges, right_edges):
            assert_(x >= left)
            assert_(x < right)

    def test_last_bin_inclusive_range(self):
        arr = np.array([0.,  0.,  0.,  1.,  2.,  3.,  3.,  4.,  5.])
        hist, edges = np.histogram(arr, bins=30, range=(-0.5, 5))
        assert_equal(hist[-1], 1)

    def test_bin_array_dims(self):
        # gracefully handle bins object > 1 dimension
        vals = np.linspace(0.0, 1.0, num=100)
        bins = np.array([[0, 0.5], [0.6, 1.0]])
        with assert_raises_regex(ValueError, "must be 1d"):
            np.histogram(vals, bins=bins)

    def test_unsigned_monotonicity_check(self):
        # Ensures ValueError is raised if bins not increasing monotonically
        # when bins contain unsigned values (see #9222)
        arr = np.array([2])
        bins = np.array([1, 3, 1], dtype='uint64')
        with assert_raises(ValueError):
            hist, edges = np.histogram(arr, bins=bins)

    def test_object_array_of_0d(self):
        # gh-7864
        assert_raises(ValueError,
            histogram, [np.array(0.4) for i in range(10)] + [-np.inf])
        assert_raises(ValueError,
            histogram, [np.array(0.4) for i in range(10)] + [np.inf])

        # these should not crash
        np.histogram([np.array(0.5) for i in range(10)] + [.500000000000002])
        np.histogram([np.array(0.5) for i in range(10)] + [.5])

    def test_some_nan_values(self):
        # gh-7503
        one_nan = np.array([0, 1, np.nan])
        all_nan = np.array([np.nan, np.nan])

        # the internal comparisons with NaN give warnings
        sup = suppress_warnings()
        sup.filter(RuntimeWarning)
        with sup:
            # can't infer range with nan
            assert_raises(ValueError, histogram, one_nan, bins='auto')
            assert_raises(ValueError, histogram, all_nan, bins='auto')

            # explicit range solves the problem
            h, b = histogram(one_nan, bins='auto', range=(0, 1))
            assert_equal(h.sum(), 2)  # nan is not counted
            h, b = histogram(all_nan, bins='auto', range=(0, 1))
            assert_equal(h.sum(), 0)  # nan is not counted

            # as does an explicit set of bins
            h, b = histogram(one_nan, bins=[0, 1])
            assert_equal(h.sum(), 2)  # nan is not counted
            h, b = histogram(all_nan, bins=[0, 1])
            assert_equal(h.sum(), 0)  # nan is not counted

    def test_datetime(self):
        begin = np.datetime64('2000-01-01', 'D')
        offsets = np.array([0, 0, 1, 1, 2, 3, 5, 10, 20])
        bins = np.array([0, 2, 7, 20])
        dates = begin + offsets
        date_bins = begin + bins

        td = np.dtype('timedelta64[D]')

        # Results should be the same for integer offsets or datetime values.
        # For now, only explicit bins are supported, since linspace does not
        # work on datetimes or timedeltas
        d_count, d_edge = histogram(dates, bins=date_bins)
        t_count, t_edge = histogram(offsets.astype(td), bins=bins.astype(td))
        i_count, i_edge = histogram(offsets, bins=bins)

        assert_equal(d_count, i_count)
        assert_equal(t_count, i_count)

        assert_equal((d_edge - begin).astype(int), i_edge)
        assert_equal(t_edge.astype(int), i_edge)

        assert_equal(d_edge.dtype, dates.dtype)
        assert_equal(t_edge.dtype, td)

    def do_signed_overflow_bounds(self, dtype):
        exponent = 8 * np.dtype(dtype).itemsize - 1
        arr = np.array([-2**exponent + 4, 2**exponent - 4], dtype=dtype)
        hist, e = histogram(arr, bins=2)
        assert_equal(e, [-2**exponent + 4, 0, 2**exponent - 4])
        assert_equal(hist, [1, 1])

    def test_signed_overflow_bounds(self):
        self.do_signed_overflow_bounds(np.byte)
        self.do_signed_overflow_bounds(np.short)
        self.do_signed_overflow_bounds(np.intc)
        self.do_signed_overflow_bounds(np.int_)
        self.do_signed_overflow_bounds(np.longlong)

    def do_precision_lower_bound(self, float_small, float_large):
        eps = np.finfo(float_large).eps

        arr = np.array([1.0], float_small)
        range = np.array([1.0 + eps, 2.0], float_large)

        # test is looking for behavior when the bounds change between dtypes
        if range.astype(float_small)[0] != 1:
            return

        # previously crashed
        count, x_loc = np.histogram(arr, bins=1, range=range)
        assert_equal(count, [0])
        assert_equal(x_loc.dtype, float_large)

    def do_precision_upper_bound(self, float_small, float_large):
        eps = np.finfo(float_large).eps

        arr = np.array([1.0], float_small)
        range = np.array([0.0, 1.0 - eps], float_large)

        # test is looking for behavior when the bounds change between dtypes
        if range.astype(float_small)[-1] != 1:
            return

        # previously crashed
        count, x_loc = np.histogram(arr, bins=1, range=range)
        assert_equal(count, [0])

        assert_equal(x_loc.dtype, float_large)

    def do_precision(self, float_small, float_large):
        self.do_precision_lower_bound(float_small, float_large)
        self.do_precision_upper_bound(float_small, float_large)

    def test_precision(self):
        # not looping results in a useful stack trace upon failure
        self.do_precision(np.half, np.single)
        self.do_precision(np.half, np.double)
        self.do_precision(np.half, np.longdouble)
        self.do_precision(np.single, np.double)
        self.do_precision(np.single, np.longdouble)
        self.do_precision(np.double, np.longdouble)

    def test_histogram_bin_edges(self):
        hist, e = histogram([1, 2, 3, 4], [1, 2])
        edges = histogram_bin_edges([1, 2, 3, 4], [1, 2])
        assert_array_equal(edges, e)

        arr = np.array([0.,  0.,  0.,  1.,  2.,  3.,  3.,  4.,  5.])
        hist, e = histogram(arr, bins=30, range=(-0.5, 5))
        edges = histogram_bin_edges(arr, bins=30, range=(-0.5, 5))
        assert_array_equal(edges, e)

        hist, e = histogram(arr, bins='auto', range=(0, 1))
        edges = histogram_bin_edges(arr, bins='auto', range=(0, 1))
        assert_array_equal(edges, e)

    def test_small_value_range(self):
        arr = np.array([1, 1 + 2e-16] * 10)
        with pytest.raises(ValueError, match="Too many bins for data range"):
            histogram(arr, bins=10)

    # @requires_memory(free_bytes=1e10)
    # @pytest.mark.slow
    @pytest.mark.skip(reason="Bad memory reports lead to OOM in ci testing")
    def test_big_arrays(self):
        sample = np.zeros([100000000, 3])
        xbins = 400
        ybins = 400
        zbins = np.arange(16000)
        hist = np.histogramdd(sample=sample, bins=(xbins, ybins, zbins))
        assert_equal(type(hist), type((1, 2)))

    def test_gh_23110(self):
        hist, e = np.histogram(np.array([-0.9e-308], dtype='>f8'),
                               bins=2,
                               range=(-1e-308, -2e-313))
        expected_hist = np.array([1, 0])
        assert_array_equal(hist, expected_hist)

    def test_gh_28400(self):
        e = 1 + 1e-12
        Z = [0, 1, 1, 1, 1, 1, e, e, e, e, e, e, 2]
        counts, edges = np.histogram(Z, bins="auto")
        assert len(counts) < 10
        assert edges[0] == Z[0]
        assert edges[-1] == Z[-1]

class TestHistogramOptimBinNums:
    """
    Provide test coverage when using provided estimators for optimal number of
    bins
    """

    def test_empty(self):
        estimator_list = ['fd', 'scott', 'rice', 'sturges',
                          'doane', 'sqrt', 'auto', 'stone']
        # check it can deal with empty data
        for estimator in estimator_list:
            a, b = histogram([], bins=estimator)
            assert_array_equal(a, np.array([0]))
            assert_array_equal(b, np.array([0, 1]))

    def test_simple(self):
        """
        Straightforward testing with a mixture of linspace data (for
        consistency). All test values have been precomputed and the values
        shouldn't change
        """
        # Some basic sanity checking, with some fixed data.
        # Checking for the correct number of bins
        basic_test = {50:   {'fd': 4,  'scott': 4,  'rice': 8,  'sturges': 7,
                             'doane': 8, 'sqrt': 8, 'auto': 7, 'stone': 2},
                      500:  {'fd': 8,  'scott': 8,  'rice': 16, 'sturges': 10,
                             'doane': 12, 'sqrt': 23, 'auto': 10, 'stone': 9},
                      5000: {'fd': 17, 'scott': 17, 'rice': 35, 'sturges': 14,
                             'doane': 17, 'sqrt': 71, 'auto': 17, 'stone': 20}}

        for testlen, expectedResults in basic_test.items():
            # Create some sort of non uniform data to test with
            # (2 peak uniform mixture)
            x1 = np.linspace(-10, -1, testlen // 5 * 2)
            x2 = np.linspace(1, 10, testlen // 5 * 3)
            x = np.concatenate((x1, x2))
            for estimator, numbins in expectedResults.items():
                a, b = np.histogram(x, estimator)
                assert_equal(len(a), numbins, err_msg=f"For the {estimator} estimator "
                             f"with datasize of {testlen}")

    def test_small(self):
        """
        Smaller datasets have the potential to cause issues with the data
        adaptive methods, especially the FD method. All bin numbers have been
        precalculated.
        """
        small_dat = {1: {'fd': 1, 'scott': 1, 'rice': 1, 'sturges': 1,
                         'doane': 1, 'sqrt': 1, 'stone': 1},
                     2: {'fd': 2, 'scott': 1, 'rice': 3, 'sturges': 2,
                         'doane': 1, 'sqrt': 2, 'stone': 1},
                     3: {'fd': 2, 'scott': 2, 'rice': 3, 'sturges': 3,
                         'doane': 3, 'sqrt': 2, 'stone': 1}}

        for testlen, expectedResults in small_dat.items():
            testdat = np.arange(testlen).astype(float)
            for estimator, expbins in expectedResults.items():
                a, b = np.histogram(testdat, estimator)
                assert_equal(len(a), expbins, err_msg=f"For the {estimator} estimator "
                             f"with datasize of {testlen}")

    def test_incorrect_methods(self):
        """
        Check a Value Error is thrown when an unknown string is passed in
        """
        check_list = ['mad', 'freeman', 'histograms', 'IQR']
        for estimator in check_list:
            assert_raises(ValueError, histogram, [1, 2, 3], estimator)

    def test_novariance(self):
        """
        Check that methods handle no variance in data
        Primarily for Scott and FD as the SD and IQR are both 0 in this case
        """
        novar_dataset = np.ones(100)
        novar_resultdict = {'fd': 1, 'scott': 1, 'rice': 1, 'sturges': 1,
                            'doane': 1, 'sqrt': 1, 'auto': 1, 'stone': 1}

        for estimator, numbins in novar_resultdict.items():
            a, b = np.histogram(novar_dataset, estimator)
            assert_equal(len(a), numbins,
                         err_msg=f"{estimator} estimator, No Variance test")

    def test_limited_variance(self):
        """
        Check when IQR is 0, but variance exists, we return a reasonable value.
        """
        lim_var_data = np.ones(1000)
        lim_var_data[:3] = 0
        lim_var_data[-4:] = 100

        edges_auto = histogram_bin_edges(lim_var_data, 'auto')
        assert_equal(edges_auto[0], 0)
        assert_equal(edges_auto[-1], 100.)
        assert len(edges_auto) < 100

        edges_fd = histogram_bin_edges(lim_var_data, 'fd')
        assert_equal(edges_fd, np.array([0, 100]))

        edges_sturges = histogram_bin_edges(lim_var_data, 'sturges')
        assert_equal(edges_sturges, np.linspace(0, 100, 12))

    def test_outlier(self):
        """
        Check the FD, Scott and Doane with outliers.

        The FD estimates a smaller binwidth since it's less affected by
        outliers. Since the range is so (artificially) large, this means more
        bins, most of which will be empty, but the data of interest usually is
        unaffected. The Scott estimator is more affected and returns fewer bins,
        despite most of the variance being in one area of the data. The Doane
        estimator lies somewhere between the other two.
        """
        xcenter = np.linspace(-10, 10, 50)
        outlier_dataset = np.hstack((np.linspace(-110, -100, 5), xcenter))

        outlier_resultdict = {'fd': 21, 'scott': 5, 'doane': 11, 'stone': 6}

        for estimator, numbins in outlier_resultdict.items():
            a, b = np.histogram(outlier_dataset, estimator)
            assert_equal(len(a), numbins)

    def test_scott_vs_stone(self):
        """Verify that Scott's rule and Stone's rule converges for normally distributed data"""

        def nbins_ratio(seed, size):
            rng = np.random.RandomState(seed)
            x = rng.normal(loc=0, scale=2, size=size)
            a, b = len(np.histogram(x, 'stone')[0]), len(np.histogram(x, 'scott')[0])
            return a / (a + b)

        ll = [[nbins_ratio(seed, size) for size in np.geomspace(start=10, stop=100, num=4).round().astype(int)]
              for seed in range(10)]

        # the average difference between the two methods decreases as the dataset size increases.
        avg = abs(np.mean(ll, axis=0) - 0.5)
        assert_almost_equal(avg, [0.15, 0.09, 0.08, 0.03], decimal=2)

    def test_simple_range(self):
        """
        Straightforward testing with a mixture of linspace data (for
        consistency). Adding in a 3rd mixture that will then be
        completely ignored. All test values have been precomputed and
        the shouldn't change.
        """
        # some basic sanity checking, with some fixed data.
        # Checking for the correct number of bins
        basic_test = {
                      50:   {'fd': 8,  'scott': 8,  'rice': 15,
                             'sturges': 14, 'auto': 14, 'stone': 8},
                      500:  {'fd': 15, 'scott': 16, 'rice': 32,
                             'sturges': 20, 'auto': 20, 'stone': 80},
                      5000: {'fd': 33, 'scott': 33, 'rice': 69,
                             'sturges': 27, 'auto': 33, 'stone': 80}
                     }

        for testlen, expectedResults in basic_test.items():
            # create some sort of non uniform data to test with
            # (3 peak uniform mixture)
            x1 = np.linspace(-10, -1, testlen // 5 * 2)
            x2 = np.linspace(1, 10, testlen // 5 * 3)
            x3 = np.linspace(-100, -50, testlen)
            x = np.hstack((x1, x2, x3))
            for estimator, numbins in expectedResults.items():
                a, b = np.histogram(x, estimator, range=(-20, 20))
                msg = f"For the {estimator} estimator"
                msg += f" with datasize of {testlen}"
                assert_equal(len(a), numbins, err_msg=msg)

    @pytest.mark.parametrize("bins", ['auto', 'fd', 'doane', 'scott',
                                      'stone', 'rice', 'sturges'])
    def test_signed_integer_data(self, bins):
        # Regression test for gh-14379.
        a = np.array([-2, 0, 127], dtype=np.int8)
        hist, edges = np.histogram(a, bins=bins)
        hist32, edges32 = np.histogram(a.astype(np.int32), bins=bins)
        assert_array_equal(hist, hist32)
        assert_array_equal(edges, edges32)

    @pytest.mark.parametrize("bins", ['auto', 'fd', 'doane', 'scott',
                                      'stone', 'rice', 'sturges'])
    def test_integer(self, bins):
        """
        Test that bin width for integer data is at least 1.
        """
        with suppress_warnings() as sup:
            if bins == 'stone':
                sup.filter(RuntimeWarning)
            assert_equal(
                np.histogram_bin_edges(np.tile(np.arange(9), 1000), bins),
                np.arange(9))

    def test_integer_non_auto(self):
        """
        Test that the bin-width>=1 requirement *only* applies to auto binning.
        """
        assert_equal(
            np.histogram_bin_edges(np.tile(np.arange(9), 1000), 16),
            np.arange(17) / 2)
        assert_equal(
            np.histogram_bin_edges(np.tile(np.arange(9), 1000), [.1, .2]),
            [.1, .2])

    def test_simple_weighted(self):
        """
        Check that weighted data raises a TypeError
        """
        estimator_list = ['fd', 'scott', 'rice', 'sturges', 'auto']
        for estimator in estimator_list:
            assert_raises(TypeError, histogram, [1, 2, 3],
                          estimator, weights=[1, 2, 3])


class TestHistogramdd:

    def test_simple(self):
        x = np.array([[-.5, .5, 1.5], [-.5, 1.5, 2.5], [-.5, 2.5, .5],
                      [.5,  .5, 1.5], [.5,  1.5, 2.5], [.5,  2.5, 2.5]])
        H, edges = histogramdd(x, (2, 3, 3),
                               range=[[-1, 1], [0, 3], [0, 3]])
        answer = np.array([[[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                           [[0, 1, 0], [0, 0, 1], [0, 0, 1]]])
        assert_array_equal(H, answer)

        # Check normalization
        ed = [[-2, 0, 2], [0, 1, 2, 3], [0, 1, 2, 3]]
        H, edges = histogramdd(x, bins=ed, density=True)
        assert_(np.all(H == answer / 12.))

        # Check that H has the correct shape.
        H, edges = histogramdd(x, (2, 3, 4),
                               range=[[-1, 1], [0, 3], [0, 4]],
                               density=True)
        answer = np.array([[[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]],
                           [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]]])
        assert_array_almost_equal(H, answer / 6., 4)
        # Check that a sequence of arrays is accepted and H has the correct
        # shape.
        z = [np.squeeze(y) for y in np.split(x, 3, axis=1)]
        H, edges = histogramdd(
            z, bins=(4, 3, 2), range=[[-2, 2], [0, 3], [0, 2]])
        answer = np.array([[[0, 0], [0, 0], [0, 0]],
                           [[0, 1], [0, 0], [1, 0]],
                           [[0, 1], [0, 0], [0, 0]],
                           [[0, 0], [0, 0], [0, 0]]])
        assert_array_equal(H, answer)

        Z = np.zeros((5, 5, 5))
        Z[list(range(5)), list(range(5)), list(range(5))] = 1.
        H, edges = histogramdd([np.arange(5), np.arange(5), np.arange(5)], 5)
        assert_array_equal(H, Z)

    def test_shape_3d(self):
        # All possible permutations for bins of different lengths in 3D.
        bins = ((5, 4, 6), (6, 4, 5), (5, 6, 4), (4, 6, 5), (6, 5, 4),
                (4, 5, 6))
        r = np.random.rand(10, 3)
        for b in bins:
            H, edges = histogramdd(r, b)
            assert_(H.shape == b)

    def test_shape_4d(self):
        # All possible permutations for bins of different lengths in 4D.
        bins = ((7, 4, 5, 6), (4, 5, 7, 6), (5, 6, 4, 7), (7, 6, 5, 4),
                (5, 7, 6, 4), (4, 6, 7, 5), (6, 5, 7, 4), (7, 5, 4, 6),
                (7, 4, 6, 5), (6, 4, 7, 5), (6, 7, 5, 4), (4, 6, 5, 7),
                (4, 7, 5, 6), (5, 4, 6, 7), (5, 7, 4, 6), (6, 7, 4, 5),
                (6, 5, 4, 7), (4, 7, 6, 5), (4, 5, 6, 7), (7, 6, 4, 5),
                (5, 4, 7, 6), (5, 6, 7, 4), (6, 4, 5, 7), (7, 5, 6, 4))

        r = np.random.rand(10, 4)
        for b in bins:
            H, edges = histogramdd(r, b)
            assert_(H.shape == b)

    def test_weights(self):
        v = np.random.rand(100, 2)
        hist, edges = histogramdd(v)
        n_hist, edges = histogramdd(v, density=True)
        w_hist, edges = histogramdd(v, weights=np.ones(100))
        assert_array_equal(w_hist, hist)
        w_hist, edges = histogramdd(v, weights=np.ones(100) * 2, density=True)
        assert_array_equal(w_hist, n_hist)
        w_hist, edges = histogramdd(v, weights=np.ones(100, int) * 2)
        assert_array_equal(w_hist, 2 * hist)

    def test_identical_samples(self):
        x = np.zeros((10, 2), int)
        hist, edges = histogramdd(x, bins=2)
        assert_array_equal(edges[0], np.array([-0.5, 0., 0.5]))

    def test_empty(self):
        a, b = histogramdd([[], []], bins=([0, 1], [0, 1]))
        assert_array_max_ulp(a, np.array([[0.]]))
        a, b = np.histogramdd([[], [], []], bins=2)
        assert_array_max_ulp(a, np.zeros((2, 2, 2)))

    def test_bins_errors(self):
        # There are two ways to specify bins. Check for the right errors
        # when mixing those.
        x = np.arange(8).reshape(2, 4)
        assert_raises(ValueError, np.histogramdd, x, bins=[-1, 2, 4, 5])
        assert_raises(ValueError, np.histogramdd, x, bins=[1, 0.99, 1, 1])
        assert_raises(
            ValueError, np.histogramdd, x, bins=[1, 1, 1, [1, 2, 3, -3]])
        assert_(np.histogramdd(x, bins=[1, 1, 1, [1, 2, 3, 4]]))

    def test_inf_edges(self):
        # Test using +/-inf bin edges works. See #1788.
        with np.errstate(invalid='ignore'):
            x = np.arange(6).reshape(3, 2)
            expected = np.array([[1, 0], [0, 1], [0, 1]])
            h, e = np.histogramdd(x, bins=[3, [-np.inf, 2, 10]])
            assert_allclose(h, expected)
            h, e = np.histogramdd(x, bins=[3, np.array([-1, 2, np.inf])])
            assert_allclose(h, expected)
            h, e = np.histogramdd(x, bins=[3, [-np.inf, 3, np.inf]])
            assert_allclose(h, expected)

    def test_rightmost_binedge(self):
        # Test event very close to rightmost binedge. See Github issue #4266
        x = [0.9999999995]
        bins = [[0., 0.5, 1.0]]
        hist, _ = histogramdd(x, bins=bins)
        assert_(hist[0] == 0.0)
        assert_(hist[1] == 1.)
        x = [1.0]
        bins = [[0., 0.5, 1.0]]
        hist, _ = histogramdd(x, bins=bins)
        assert_(hist[0] == 0.0)
        assert_(hist[1] == 1.)
        x = [1.0000000001]
        bins = [[0., 0.5, 1.0]]
        hist, _ = histogramdd(x, bins=bins)
        assert_(hist[0] == 0.0)
        assert_(hist[1] == 0.0)
        x = [1.0001]
        bins = [[0., 0.5, 1.0]]
        hist, _ = histogramdd(x, bins=bins)
        assert_(hist[0] == 0.0)
        assert_(hist[1] == 0.0)

    def test_finite_range(self):
        vals = np.random.random((100, 3))
        histogramdd(vals, range=[[0.0, 1.0], [0.25, 0.75], [0.25, 0.5]])
        assert_raises(ValueError, histogramdd, vals,
                      range=[[0.0, 1.0], [0.25, 0.75], [0.25, np.inf]])
        assert_raises(ValueError, histogramdd, vals,
                      range=[[0.0, 1.0], [np.nan, 0.75], [0.25, 0.5]])

    def test_equal_edges(self):
        """ Test that adjacent entries in an edge array can be equal """
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 2])
        x_edges = np.array([0, 2, 2])
        y_edges = 1
        hist, edges = histogramdd((x, y), bins=(x_edges, y_edges))

        hist_expected = np.array([
            [2.],
            [1.],  # x == 2 falls in the final bin
        ])
        assert_equal(hist, hist_expected)

    def test_edge_dtype(self):
        """ Test that if an edge array is input, its type is preserved """
        x = np.array([0, 10, 20])
        y = x / 10
        x_edges = np.array([0, 5, 15, 20])
        y_edges = x_edges / 10
        hist, edges = histogramdd((x, y), bins=(x_edges, y_edges))

        assert_equal(edges[0].dtype, x_edges.dtype)
        assert_equal(edges[1].dtype, y_edges.dtype)

    def test_large_integers(self):
        big = 2**60  # Too large to represent with a full precision float

        x = np.array([0], np.int64)
        x_edges = np.array([-1, +1], np.int64)
        y = big + x
        y_edges = big + x_edges

        hist, edges = histogramdd((x, y), bins=(x_edges, y_edges))

        assert_equal(hist[0, 0], 1)

    def test_density_non_uniform_2d(self):
        # Defines the following grid:
        #
        #    0 2     8
        #   0+-+-----+
        #    + |     +
        #    + |     +
        #   6+-+-----+
        #   8+-+-----+
        x_edges = np.array([0, 2, 8])
        y_edges = np.array([0, 6, 8])
        relative_areas = np.array([
            [3, 9],
            [1, 3]])

        # ensure the number of points in each region is proportional to its area
        x = np.array([1] + [1] * 3 + [7] * 3 + [7] * 9)
        y = np.array([7] + [1] * 3 + [7] * 3 + [1] * 9)

        # sanity check that the above worked as intended
        hist, edges = histogramdd((y, x), bins=(y_edges, x_edges))
        assert_equal(hist, relative_areas)

        # resulting histogram should be uniform, since counts and areas are proportional
        hist, edges = histogramdd((y, x), bins=(y_edges, x_edges), density=True)
        assert_equal(hist, 1 / (8 * 8))

    def test_density_non_uniform_1d(self):
        # compare to histogram to show the results are the same
        v = np.arange(10)
        bins = np.array([0, 1, 3, 6, 10])
        hist, edges = histogram(v, bins, density=True)
        hist_dd, edges_dd = histogramdd((v,), (bins,), density=True)
        assert_equal(hist, hist_dd)
        assert_equal(edges, edges_dd[0])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\tests\test_shape_base.py ===
import functools
import sys

import pytest

import numpy as np
from numpy import (
    apply_along_axis,
    apply_over_axes,
    array_split,
    column_stack,
    dsplit,
    dstack,
    expand_dims,
    hsplit,
    kron,
    put_along_axis,
    split,
    take_along_axis,
    tile,
    vsplit,
)
from numpy.exceptions import AxisError
from numpy.testing import assert_, assert_array_equal, assert_equal, assert_raises

IS_64BIT = sys.maxsize > 2**32


def _add_keepdims(func):
    """ hack in keepdims behavior into a function taking an axis """
    @functools.wraps(func)
    def wrapped(a, axis, **kwargs):
        res = func(a, axis=axis, **kwargs)
        if axis is None:
            axis = 0  # res is now a scalar, so we can insert this anywhere
        return np.expand_dims(res, axis=axis)
    return wrapped


class TestTakeAlongAxis:
    def test_argequivalent(self):
        """ Test it translates from arg<func> to <func> """
        from numpy.random import rand
        a = rand(3, 4, 5)

        funcs = [
            (np.sort, np.argsort, {}),
            (_add_keepdims(np.min), _add_keepdims(np.argmin), {}),
            (_add_keepdims(np.max), _add_keepdims(np.argmax), {}),
            #(np.partition, np.argpartition, dict(kth=2)),
        ]

        for func, argfunc, kwargs in funcs:
            for axis in list(range(a.ndim)) + [None]:
                a_func = func(a, axis=axis, **kwargs)
                ai_func = argfunc(a, axis=axis, **kwargs)
                assert_equal(a_func, take_along_axis(a, ai_func, axis=axis))

    def test_invalid(self):
        """ Test it errors when indices has too few dimensions """
        a = np.ones((10, 10))
        ai = np.ones((10, 2), dtype=np.intp)

        # sanity check
        take_along_axis(a, ai, axis=1)

        # not enough indices
        assert_raises(ValueError, take_along_axis, a, np.array(1), axis=1)
        # bool arrays not allowed
        assert_raises(IndexError, take_along_axis, a, ai.astype(bool), axis=1)
        # float arrays not allowed
        assert_raises(IndexError, take_along_axis, a, ai.astype(float), axis=1)
        # invalid axis
        assert_raises(AxisError, take_along_axis, a, ai, axis=10)
        # invalid indices
        assert_raises(ValueError, take_along_axis, a, ai, axis=None)

    def test_empty(self):
        """ Test everything is ok with empty results, even with inserted dims """
        a = np.ones((3, 4, 5))
        ai = np.ones((3, 0, 5), dtype=np.intp)

        actual = take_along_axis(a, ai, axis=1)
        assert_equal(actual.shape, ai.shape)

    def test_broadcast(self):
        """ Test that non-indexing dimensions are broadcast in both directions """
        a = np.ones((3, 4, 1))
        ai = np.ones((1, 2, 5), dtype=np.intp)
        actual = take_along_axis(a, ai, axis=1)
        assert_equal(actual.shape, (3, 2, 5))


class TestPutAlongAxis:
    def test_replace_max(self):
        a_base = np.array([[10, 30, 20], [60, 40, 50]])

        for axis in list(range(a_base.ndim)) + [None]:
            # we mutate this in the loop
            a = a_base.copy()

            # replace the max with a small value
            i_max = _add_keepdims(np.argmax)(a, axis=axis)
            put_along_axis(a, i_max, -99, axis=axis)

            # find the new minimum, which should max
            i_min = _add_keepdims(np.argmin)(a, axis=axis)

            assert_equal(i_min, i_max)

    def test_broadcast(self):
        """ Test that non-indexing dimensions are broadcast in both directions """
        a = np.ones((3, 4, 1))
        ai = np.arange(10, dtype=np.intp).reshape((1, 2, 5)) % 4
        put_along_axis(a, ai, 20, axis=1)
        assert_equal(take_along_axis(a, ai, axis=1), 20)

    def test_invalid(self):
        """ Test invalid inputs """
        a_base = np.array([[10, 30, 20], [60, 40, 50]])
        indices = np.array([[0], [1]])
        values = np.array([[2], [1]])

        # sanity check
        a = a_base.copy()
        put_along_axis(a, indices, values, axis=0)
        assert np.all(a == [[2, 2, 2], [1, 1, 1]])

        # invalid indices
        a = a_base.copy()
        with assert_raises(ValueError) as exc:
            put_along_axis(a, indices, values, axis=None)
        assert "single dimension" in str(exc.exception)


class TestApplyAlongAxis:
    def test_simple(self):
        a = np.ones((20, 10), 'd')
        assert_array_equal(
            apply_along_axis(len, 0, a), len(a) * np.ones(a.shape[1]))

    def test_simple101(self):
        a = np.ones((10, 101), 'd')
        assert_array_equal(
            apply_along_axis(len, 0, a), len(a) * np.ones(a.shape[1]))

    def test_3d(self):
        a = np.arange(27).reshape((3, 3, 3))
        assert_array_equal(apply_along_axis(np.sum, 0, a),
                           [[27, 30, 33], [36, 39, 42], [45, 48, 51]])

    def test_preserve_subclass(self):
        def double(row):
            return row * 2

        class MyNDArray(np.ndarray):
            pass

        m = np.array([[0, 1], [2, 3]]).view(MyNDArray)
        expected = np.array([[0, 2], [4, 6]]).view(MyNDArray)

        result = apply_along_axis(double, 0, m)
        assert_(isinstance(result, MyNDArray))
        assert_array_equal(result, expected)

        result = apply_along_axis(double, 1, m)
        assert_(isinstance(result, MyNDArray))
        assert_array_equal(result, expected)

    def test_subclass(self):
        class MinimalSubclass(np.ndarray):
            data = 1

        def minimal_function(array):
            return array.data

        a = np.zeros((6, 3)).view(MinimalSubclass)

        assert_array_equal(
            apply_along_axis(minimal_function, 0, a), np.array([1, 1, 1])
        )

    def test_scalar_array(self, cls=np.ndarray):
        a = np.ones((6, 3)).view(cls)
        res = apply_along_axis(np.sum, 0, a)
        assert_(isinstance(res, cls))
        assert_array_equal(res, np.array([6, 6, 6]).view(cls))

    def test_0d_array(self, cls=np.ndarray):
        def sum_to_0d(x):
            """ Sum x, returning a 0d array of the same class """
            assert_equal(x.ndim, 1)
            return np.squeeze(np.sum(x, keepdims=True))
        a = np.ones((6, 3)).view(cls)
        res = apply_along_axis(sum_to_0d, 0, a)
        assert_(isinstance(res, cls))
        assert_array_equal(res, np.array([6, 6, 6]).view(cls))

        res = apply_along_axis(sum_to_0d, 1, a)
        assert_(isinstance(res, cls))
        assert_array_equal(res, np.array([3, 3, 3, 3, 3, 3]).view(cls))

    def test_axis_insertion(self, cls=np.ndarray):
        def f1to2(x):
            """produces an asymmetric non-square matrix from x"""
            assert_equal(x.ndim, 1)
            return (x[::-1] * x[1:, None]).view(cls)

        a2d = np.arange(6 * 3).reshape((6, 3))

        # 2d insertion along first axis
        actual = apply_along_axis(f1to2, 0, a2d)
        expected = np.stack([
            f1to2(a2d[:, i]) for i in range(a2d.shape[1])
        ], axis=-1).view(cls)
        assert_equal(type(actual), type(expected))
        assert_equal(actual, expected)

        # 2d insertion along last axis
        actual = apply_along_axis(f1to2, 1, a2d)
        expected = np.stack([
            f1to2(a2d[i, :]) for i in range(a2d.shape[0])
        ], axis=0).view(cls)
        assert_equal(type(actual), type(expected))
        assert_equal(actual, expected)

        # 3d insertion along middle axis
        a3d = np.arange(6 * 5 * 3).reshape((6, 5, 3))

        actual = apply_along_axis(f1to2, 1, a3d)
        expected = np.stack([
            np.stack([
                f1to2(a3d[i, :, j]) for i in range(a3d.shape[0])
            ], axis=0)
            for j in range(a3d.shape[2])
        ], axis=-1).view(cls)
        assert_equal(type(actual), type(expected))
        assert_equal(actual, expected)

    def test_subclass_preservation(self):
        class MinimalSubclass(np.ndarray):
            pass
        self.test_scalar_array(MinimalSubclass)
        self.test_0d_array(MinimalSubclass)
        self.test_axis_insertion(MinimalSubclass)

    def test_axis_insertion_ma(self):
        def f1to2(x):
            """produces an asymmetric non-square matrix from x"""
            assert_equal(x.ndim, 1)
            res = x[::-1] * x[1:, None]
            return np.ma.masked_where(res % 5 == 0, res)
        a = np.arange(6 * 3).reshape((6, 3))
        res = apply_along_axis(f1to2, 0, a)
        assert_(isinstance(res, np.ma.masked_array))
        assert_equal(res.ndim, 3)
        assert_array_equal(res[:, :, 0].mask, f1to2(a[:, 0]).mask)
        assert_array_equal(res[:, :, 1].mask, f1to2(a[:, 1]).mask)
        assert_array_equal(res[:, :, 2].mask, f1to2(a[:, 2]).mask)

    def test_tuple_func1d(self):
        def sample_1d(x):
            return x[1], x[0]
        res = np.apply_along_axis(sample_1d, 1, np.array([[1, 2], [3, 4]]))
        assert_array_equal(res, np.array([[2, 1], [4, 3]]))

    def test_empty(self):
        # can't apply_along_axis when there's no chance to call the function
        def never_call(x):
            assert_(False)  # should never be reached

        a = np.empty((0, 0))
        assert_raises(ValueError, np.apply_along_axis, never_call, 0, a)
        assert_raises(ValueError, np.apply_along_axis, never_call, 1, a)

        # but it's sometimes ok with some non-zero dimensions
        def empty_to_1(x):
            assert_(len(x) == 0)
            return 1

        a = np.empty((10, 0))
        actual = np.apply_along_axis(empty_to_1, 1, a)
        assert_equal(actual, np.ones(10))
        assert_raises(ValueError, np.apply_along_axis, empty_to_1, 0, a)

    def test_with_iterable_object(self):
        # from issue 5248
        d = np.array([
            [{1, 11}, {2, 22}, {3, 33}],
            [{4, 44}, {5, 55}, {6, 66}]
        ])
        actual = np.apply_along_axis(lambda a: set.union(*a), 0, d)
        expected = np.array([{1, 11, 4, 44}, {2, 22, 5, 55}, {3, 33, 6, 66}])

        assert_equal(actual, expected)

        # issue 8642 - assert_equal doesn't detect this!
        for i in np.ndindex(actual.shape):
            assert_equal(type(actual[i]), type(expected[i]))


class TestApplyOverAxes:
    def test_simple(self):
        a = np.arange(24).reshape(2, 3, 4)
        aoa_a = apply_over_axes(np.sum, a, [0, 2])
        assert_array_equal(aoa_a, np.array([[[60], [92], [124]]]))


class TestExpandDims:
    def test_functionality(self):
        s = (2, 3, 4, 5)
        a = np.empty(s)
        for axis in range(-5, 4):
            b = expand_dims(a, axis)
            assert_(b.shape[axis] == 1)
            assert_(np.squeeze(b).shape == s)

    def test_axis_tuple(self):
        a = np.empty((3, 3, 3))
        assert np.expand_dims(a, axis=(0, 1, 2)).shape == (1, 1, 1, 3, 3, 3)
        assert np.expand_dims(a, axis=(0, -1, -2)).shape == (1, 3, 3, 3, 1, 1)
        assert np.expand_dims(a, axis=(0, 3, 5)).shape == (1, 3, 3, 1, 3, 1)
        assert np.expand_dims(a, axis=(0, -3, -5)).shape == (1, 1, 3, 1, 3, 3)

    def test_axis_out_of_range(self):
        s = (2, 3, 4, 5)
        a = np.empty(s)
        assert_raises(AxisError, expand_dims, a, -6)
        assert_raises(AxisError, expand_dims, a, 5)

        a = np.empty((3, 3, 3))
        assert_raises(AxisError, expand_dims, a, (0, -6))
        assert_raises(AxisError, expand_dims, a, (0, 5))

    def test_repeated_axis(self):
        a = np.empty((3, 3, 3))
        assert_raises(ValueError, expand_dims, a, axis=(1, 1))

    def test_subclasses(self):
        a = np.arange(10).reshape((2, 5))
        a = np.ma.array(a, mask=a % 3 == 0)

        expanded = np.expand_dims(a, axis=1)
        assert_(isinstance(expanded, np.ma.MaskedArray))
        assert_equal(expanded.shape, (2, 1, 5))
        assert_equal(expanded.mask.shape, (2, 1, 5))


class TestArraySplit:
    def test_integer_0_split(self):
        a = np.arange(10)
        assert_raises(ValueError, array_split, a, 0)

    def test_integer_split(self):
        a = np.arange(10)
        res = array_split(a, 1)
        desired = [np.arange(10)]
        compare_results(res, desired)

        res = array_split(a, 2)
        desired = [np.arange(5), np.arange(5, 10)]
        compare_results(res, desired)

        res = array_split(a, 3)
        desired = [np.arange(4), np.arange(4, 7), np.arange(7, 10)]
        compare_results(res, desired)

        res = array_split(a, 4)
        desired = [np.arange(3), np.arange(3, 6), np.arange(6, 8),
                   np.arange(8, 10)]
        compare_results(res, desired)

        res = array_split(a, 5)
        desired = [np.arange(2), np.arange(2, 4), np.arange(4, 6),
                   np.arange(6, 8), np.arange(8, 10)]
        compare_results(res, desired)

        res = array_split(a, 6)
        desired = [np.arange(2), np.arange(2, 4), np.arange(4, 6),
                   np.arange(6, 8), np.arange(8, 9), np.arange(9, 10)]
        compare_results(res, desired)

        res = array_split(a, 7)
        desired = [np.arange(2), np.arange(2, 4), np.arange(4, 6),
                   np.arange(6, 7), np.arange(7, 8), np.arange(8, 9),
                   np.arange(9, 10)]
        compare_results(res, desired)

        res = array_split(a, 8)
        desired = [np.arange(2), np.arange(2, 4), np.arange(4, 5),
                   np.arange(5, 6), np.arange(6, 7), np.arange(7, 8),
                   np.arange(8, 9), np.arange(9, 10)]
        compare_results(res, desired)

        res = array_split(a, 9)
        desired = [np.arange(2), np.arange(2, 3), np.arange(3, 4),
                   np.arange(4, 5), np.arange(5, 6), np.arange(6, 7),
                   np.arange(7, 8), np.arange(8, 9), np.arange(9, 10)]
        compare_results(res, desired)

        res = array_split(a, 10)
        desired = [np.arange(1), np.arange(1, 2), np.arange(2, 3),
                   np.arange(3, 4), np.arange(4, 5), np.arange(5, 6),
                   np.arange(6, 7), np.arange(7, 8), np.arange(8, 9),
                   np.arange(9, 10)]
        compare_results(res, desired)

        res = array_split(a, 11)
        desired = [np.arange(1), np.arange(1, 2), np.arange(2, 3),
                   np.arange(3, 4), np.arange(4, 5), np.arange(5, 6),
                   np.arange(6, 7), np.arange(7, 8), np.arange(8, 9),
                   np.arange(9, 10), np.array([])]
        compare_results(res, desired)

    def test_integer_split_2D_rows(self):
        a = np.array([np.arange(10), np.arange(10)])
        res = array_split(a, 3, axis=0)
        tgt = [np.array([np.arange(10)]), np.array([np.arange(10)]),
                   np.zeros((0, 10))]
        compare_results(res, tgt)
        assert_(a.dtype.type is res[-1].dtype.type)

        # Same thing for manual splits:
        res = array_split(a, [0, 1], axis=0)
        tgt = [np.zeros((0, 10)), np.array([np.arange(10)]),
               np.array([np.arange(10)])]
        compare_results(res, tgt)
        assert_(a.dtype.type is res[-1].dtype.type)

    def test_integer_split_2D_cols(self):
        a = np.array([np.arange(10), np.arange(10)])
        res = array_split(a, 3, axis=-1)
        desired = [np.array([np.arange(4), np.arange(4)]),
                   np.array([np.arange(4, 7), np.arange(4, 7)]),
                   np.array([np.arange(7, 10), np.arange(7, 10)])]
        compare_results(res, desired)

    def test_integer_split_2D_default(self):
        """ This will fail if we change default axis
        """
        a = np.array([np.arange(10), np.arange(10)])
        res = array_split(a, 3)
        tgt = [np.array([np.arange(10)]), np.array([np.arange(10)]),
                   np.zeros((0, 10))]
        compare_results(res, tgt)
        assert_(a.dtype.type is res[-1].dtype.type)
        # perhaps should check higher dimensions

    @pytest.mark.skipif(not IS_64BIT, reason="Needs 64bit platform")
    def test_integer_split_2D_rows_greater_max_int32(self):
        a = np.broadcast_to([0], (1 << 32, 2))
        res = array_split(a, 4)
        chunk = np.broadcast_to([0], (1 << 30, 2))
        tgt = [chunk] * 4
        for i in range(len(tgt)):
            assert_equal(res[i].shape, tgt[i].shape)

    def test_index_split_simple(self):
        a = np.arange(10)
        indices = [1, 5, 7]
        res = array_split(a, indices, axis=-1)
        desired = [np.arange(0, 1), np.arange(1, 5), np.arange(5, 7),
                   np.arange(7, 10)]
        compare_results(res, desired)

    def test_index_split_low_bound(self):
        a = np.arange(10)
        indices = [0, 5, 7]
        res = array_split(a, indices, axis=-1)
        desired = [np.array([]), np.arange(0, 5), np.arange(5, 7),
                   np.arange(7, 10)]
        compare_results(res, desired)

    def test_index_split_high_bound(self):
        a = np.arange(10)
        indices = [0, 5, 7, 10, 12]
        res = array_split(a, indices, axis=-1)
        desired = [np.array([]), np.arange(0, 5), np.arange(5, 7),
                   np.arange(7, 10), np.array([]), np.array([])]
        compare_results(res, desired)


class TestSplit:
    # The split function is essentially the same as array_split,
    # except that it test if splitting will result in an
    # equal split.  Only test for this case.

    def test_equal_split(self):
        a = np.arange(10)
        res = split(a, 2)
        desired = [np.arange(5), np.arange(5, 10)]
        compare_results(res, desired)

    def test_unequal_split(self):
        a = np.arange(10)
        assert_raises(ValueError, split, a, 3)


class TestColumnStack:
    def test_non_iterable(self):
        assert_raises(TypeError, column_stack, 1)

    def test_1D_arrays(self):
        # example from docstring
        a = np.array((1, 2, 3))
        b = np.array((2, 3, 4))
        expected = np.array([[1, 2],
                             [2, 3],
                             [3, 4]])
        actual = np.column_stack((a, b))
        assert_equal(actual, expected)

    def test_2D_arrays(self):
        # same as hstack 2D docstring example
        a = np.array([[1], [2], [3]])
        b = np.array([[2], [3], [4]])
        expected = np.array([[1, 2],
                             [2, 3],
                             [3, 4]])
        actual = np.column_stack((a, b))
        assert_equal(actual, expected)

    def test_generator(self):
        with pytest.raises(TypeError, match="arrays to stack must be"):
            column_stack(np.arange(3) for _ in range(2))


class TestDstack:
    def test_non_iterable(self):
        assert_raises(TypeError, dstack, 1)

    def test_0D_array(self):
        a = np.array(1)
        b = np.array(2)
        res = dstack([a, b])
        desired = np.array([[[1, 2]]])
        assert_array_equal(res, desired)

    def test_1D_array(self):
        a = np.array([1])
        b = np.array([2])
        res = dstack([a, b])
        desired = np.array([[[1, 2]]])
        assert_array_equal(res, desired)

    def test_2D_array(self):
        a = np.array([[1], [2]])
        b = np.array([[1], [2]])
        res = dstack([a, b])
        desired = np.array([[[1, 1]], [[2, 2, ]]])
        assert_array_equal(res, desired)

    def test_2D_array2(self):
        a = np.array([1, 2])
        b = np.array([1, 2])
        res = dstack([a, b])
        desired = np.array([[[1, 1], [2, 2]]])
        assert_array_equal(res, desired)

    def test_generator(self):
        with pytest.raises(TypeError, match="arrays to stack must be"):
            dstack(np.arange(3) for _ in range(2))


# array_split has more comprehensive test of splitting.
# only do simple test on hsplit, vsplit, and dsplit
class TestHsplit:
    """Only testing for integer splits.

    """
    def test_non_iterable(self):
        assert_raises(ValueError, hsplit, 1, 1)

    def test_0D_array(self):
        a = np.array(1)
        try:
            hsplit(a, 2)
            assert_(0)
        except ValueError:
            pass

    def test_1D_array(self):
        a = np.array([1, 2, 3, 4])
        res = hsplit(a, 2)
        desired = [np.array([1, 2]), np.array([3, 4])]
        compare_results(res, desired)

    def test_2D_array(self):
        a = np.array([[1, 2, 3, 4],
                  [1, 2, 3, 4]])
        res = hsplit(a, 2)
        desired = [np.array([[1, 2], [1, 2]]), np.array([[3, 4], [3, 4]])]
        compare_results(res, desired)


class TestVsplit:
    """Only testing for integer splits.

    """
    def test_non_iterable(self):
        assert_raises(ValueError, vsplit, 1, 1)

    def test_0D_array(self):
        a = np.array(1)
        assert_raises(ValueError, vsplit, a, 2)

    def test_1D_array(self):
        a = np.array([1, 2, 3, 4])
        try:
            vsplit(a, 2)
            assert_(0)
        except ValueError:
            pass

    def test_2D_array(self):
        a = np.array([[1, 2, 3, 4],
                  [1, 2, 3, 4]])
        res = vsplit(a, 2)
        desired = [np.array([[1, 2, 3, 4]]), np.array([[1, 2, 3, 4]])]
        compare_results(res, desired)


class TestDsplit:
    # Only testing for integer splits.
    def test_non_iterable(self):
        assert_raises(ValueError, dsplit, 1, 1)

    def test_0D_array(self):
        a = np.array(1)
        assert_raises(ValueError, dsplit, a, 2)

    def test_1D_array(self):
        a = np.array([1, 2, 3, 4])
        assert_raises(ValueError, dsplit, a, 2)

    def test_2D_array(self):
        a = np.array([[1, 2, 3, 4],
                  [1, 2, 3, 4]])
        try:
            dsplit(a, 2)
            assert_(0)
        except ValueError:
            pass

    def test_3D_array(self):
        a = np.array([[[1, 2, 3, 4],
                   [1, 2, 3, 4]],
                  [[1, 2, 3, 4],
                   [1, 2, 3, 4]]])
        res = dsplit(a, 2)
        desired = [np.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]]),
                   np.array([[[3, 4], [3, 4]], [[3, 4], [3, 4]]])]
        compare_results(res, desired)


class TestSqueeze:
    def test_basic(self):
        from numpy.random import rand

        a = rand(20, 10, 10, 1, 1)
        b = rand(20, 1, 10, 1, 20)
        c = rand(1, 1, 20, 10)
        assert_array_equal(np.squeeze(a), np.reshape(a, (20, 10, 10)))
        assert_array_equal(np.squeeze(b), np.reshape(b, (20, 10, 20)))
        assert_array_equal(np.squeeze(c), np.reshape(c, (20, 10)))

        # Squeezing to 0-dim should still give an ndarray
        a = [[[1.5]]]
        res = np.squeeze(a)
        assert_equal(res, 1.5)
        assert_equal(res.ndim, 0)
        assert_equal(type(res), np.ndarray)


class TestKron:
    def test_basic(self):
        # Using 0-dimensional ndarray
        a = np.array(1)
        b = np.array([[1, 2], [3, 4]])
        k = np.array([[1, 2], [3, 4]])
        assert_array_equal(np.kron(a, b), k)
        a = np.array([[1, 2], [3, 4]])
        b = np.array(1)
        assert_array_equal(np.kron(a, b), k)

        # Using 1-dimensional ndarray
        a = np.array([3])
        b = np.array([[1, 2], [3, 4]])
        k = np.array([[3, 6], [9, 12]])
        assert_array_equal(np.kron(a, b), k)
        a = np.array([[1, 2], [3, 4]])
        b = np.array([3])
        assert_array_equal(np.kron(a, b), k)

        # Using 3-dimensional ndarray
        a = np.array([[[1]], [[2]]])
        b = np.array([[1, 2], [3, 4]])
        k = np.array([[[1, 2], [3, 4]], [[2, 4], [6, 8]]])
        assert_array_equal(np.kron(a, b), k)
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[[1]], [[2]]])
        k = np.array([[[1, 2], [3, 4]], [[2, 4], [6, 8]]])
        assert_array_equal(np.kron(a, b), k)

    def test_return_type(self):
        class myarray(np.ndarray):
            __array_priority__ = 1.0

        a = np.ones([2, 2])
        ma = myarray(a.shape, a.dtype, a.data)
        assert_equal(type(kron(a, a)), np.ndarray)
        assert_equal(type(kron(ma, ma)), myarray)
        assert_equal(type(kron(a, ma)), myarray)
        assert_equal(type(kron(ma, a)), myarray)

    @pytest.mark.parametrize(
        "array_class", [np.asarray, np.asmatrix]
    )
    def test_kron_smoke(self, array_class):
        a = array_class(np.ones([3, 3]))
        b = array_class(np.ones([3, 3]))
        k = array_class(np.ones([9, 9]))

        assert_array_equal(np.kron(a, b), k)

    def test_kron_ma(self):
        x = np.ma.array([[1, 2], [3, 4]], mask=[[0, 1], [1, 0]])
        k = np.ma.array(np.diag([1, 4, 4, 16]),
                mask=~np.array(np.identity(4), dtype=bool))

        assert_array_equal(k, np.kron(x, x))

    @pytest.mark.parametrize(
        "shape_a,shape_b", [
            ((1, 1), (1, 1)),
            ((1, 2, 3), (4, 5, 6)),
            ((2, 2), (2, 2, 2)),
            ((1, 0), (1, 1)),
            ((2, 0, 2), (2, 2)),
            ((2, 0, 0, 2), (2, 0, 2)),
        ])
    def test_kron_shape(self, shape_a, shape_b):
        a = np.ones(shape_a)
        b = np.ones(shape_b)
        normalised_shape_a = (1,) * max(0, len(shape_b) - len(shape_a)) + shape_a
        normalised_shape_b = (1,) * max(0, len(shape_a) - len(shape_b)) + shape_b
        expected_shape = np.multiply(normalised_shape_a, normalised_shape_b)

        k = np.kron(a, b)
        assert np.array_equal(
                k.shape, expected_shape), "Unexpected shape from kron"


class TestTile:
    def test_basic(self):
        a = np.array([0, 1, 2])
        b = [[1, 2], [3, 4]]
        assert_equal(tile(a, 2), [0, 1, 2, 0, 1, 2])
        assert_equal(tile(a, (2, 2)), [[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]])
        assert_equal(tile(a, (1, 2)), [[0, 1, 2, 0, 1, 2]])
        assert_equal(tile(b, 2), [[1, 2, 1, 2], [3, 4, 3, 4]])
        assert_equal(tile(b, (2, 1)), [[1, 2], [3, 4], [1, 2], [3, 4]])
        assert_equal(tile(b, (2, 2)), [[1, 2, 1, 2], [3, 4, 3, 4],
                                       [1, 2, 1, 2], [3, 4, 3, 4]])

    def test_tile_one_repetition_on_array_gh4679(self):
        a = np.arange(5)
        b = tile(a, 1)
        b += 2
        assert_equal(a, np.arange(5))

    def test_empty(self):
        a = np.array([[[]]])
        b = np.array([[], []])
        c = tile(b, 2).shape
        d = tile(a, (3, 2, 5)).shape
        assert_equal(c, (2, 0))
        assert_equal(d, (3, 2, 0))

    def test_kroncompare(self):
        from numpy.random import randint

        reps = [(2,), (1, 2), (2, 1), (2, 2), (2, 3, 2), (3, 2)]
        shape = [(3,), (2, 3), (3, 4, 3), (3, 2, 3), (4, 3, 2, 4), (2, 2)]
        for s in shape:
            b = randint(0, 10, size=s)
            for r in reps:
                a = np.ones(r, b.dtype)
                large = tile(b, r)
                klarge = kron(a, b)
                assert_equal(large, klarge)


class TestMayShareMemory:
    def test_basic(self):
        d = np.ones((50, 60))
        d2 = np.ones((30, 60, 6))
        assert_(np.may_share_memory(d, d))
        assert_(np.may_share_memory(d, d[::-1]))
        assert_(np.may_share_memory(d, d[::2]))
        assert_(np.may_share_memory(d, d[1:, ::-1]))

        assert_(not np.may_share_memory(d[::-1], d2))
        assert_(not np.may_share_memory(d[::2], d2))
        assert_(not np.may_share_memory(d[1:, ::-1], d2))
        assert_(np.may_share_memory(d2[1:, ::-1], d2))


# Utility
def compare_results(res, desired):
    """Compare lists of arrays."""
    for x, y in zip(res, desired, strict=False):
        assert_array_equal(x, y)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\_utils_impl.py ===
import functools
import os
import platform
import sys
import textwrap
import types
import warnings

import numpy as np
from numpy._core import ndarray
from numpy._utils import set_module

__all__ = [
    'get_include', 'info', 'show_runtime'
]


@set_module('numpy')
def show_runtime():
    """
    Print information about various resources in the system
    including available intrinsic support and BLAS/LAPACK library
    in use

    .. versionadded:: 1.24.0

    See Also
    --------
    show_config : Show libraries in the system on which NumPy was built.

    Notes
    -----
    1. Information is derived with the help of `threadpoolctl <https://pypi.org/project/threadpoolctl/>`_
       library if available.
    2. SIMD related information is derived from ``__cpu_features__``,
       ``__cpu_baseline__`` and ``__cpu_dispatch__``

    """
    from pprint import pprint

    from numpy._core._multiarray_umath import (
        __cpu_baseline__,
        __cpu_dispatch__,
        __cpu_features__,
    )
    config_found = [{
        "numpy_version": np.__version__,
        "python": sys.version,
        "uname": platform.uname(),
        }]
    features_found, features_not_found = [], []
    for feature in __cpu_dispatch__:
        if __cpu_features__[feature]:
            features_found.append(feature)
        else:
            features_not_found.append(feature)
    config_found.append({
        "simd_extensions": {
            "baseline": __cpu_baseline__,
            "found": features_found,
            "not_found": features_not_found
        }
    })
    try:
        from threadpoolctl import threadpool_info
        config_found.extend(threadpool_info())
    except ImportError:
        print("WARNING: `threadpoolctl` not found in system!"
              " Install it by `pip install threadpoolctl`."
              " Once installed, try `np.show_runtime` again"
              " for more detailed build information")
    pprint(config_found)


@set_module('numpy')
def get_include():
    """
    Return the directory that contains the NumPy \\*.h header files.

    Extension modules that need to compile against NumPy may need to use this
    function to locate the appropriate include directory.

    Notes
    -----
    When using ``setuptools``, for example in ``setup.py``::

        import numpy as np
        ...
        Extension('extension_name', ...
                  include_dirs=[np.get_include()])
        ...

    Note that a CLI tool ``numpy-config`` was introduced in NumPy 2.0, using
    that is likely preferred for build systems other than ``setuptools``::

        $ numpy-config --cflags
        -I/path/to/site-packages/numpy/_core/include

        # Or rely on pkg-config:
        $ export PKG_CONFIG_PATH=$(numpy-config --pkgconfigdir)
        $ pkg-config --cflags
        -I/path/to/site-packages/numpy/_core/include

    Examples
    --------
    >>> np.get_include()
    '.../site-packages/numpy/core/include'  # may vary

    """
    import numpy
    if numpy.show_config is None:
        # running from numpy source directory
        d = os.path.join(os.path.dirname(numpy.__file__), '_core', 'include')
    else:
        # using installed numpy core headers
        import numpy._core as _core
        d = os.path.join(os.path.dirname(_core.__file__), 'include')
    return d


class _Deprecate:
    """
    Decorator class to deprecate old functions.

    Refer to `deprecate` for details.

    See Also
    --------
    deprecate

    """

    def __init__(self, old_name=None, new_name=None, message=None):
        self.old_name = old_name
        self.new_name = new_name
        self.message = message

    def __call__(self, func, *args, **kwargs):
        """
        Decorator call.  Refer to ``decorate``.

        """
        old_name = self.old_name
        new_name = self.new_name
        message = self.message

        if old_name is None:
            old_name = func.__name__
        if new_name is None:
            depdoc = f"`{old_name}` is deprecated!"
        else:
            depdoc = f"`{old_name}` is deprecated, use `{new_name}` instead!"

        if message is not None:
            depdoc += "\n" + message

        @functools.wraps(func)
        def newfunc(*args, **kwds):
            warnings.warn(depdoc, DeprecationWarning, stacklevel=2)
            return func(*args, **kwds)

        newfunc.__name__ = old_name
        doc = func.__doc__
        if doc is None:
            doc = depdoc
        else:
            lines = doc.expandtabs().split('\n')
            indent = _get_indent(lines[1:])
            if lines[0].lstrip():
                # Indent the original first line to let inspect.cleandoc()
                # dedent the docstring despite the deprecation notice.
                doc = indent * ' ' + doc
            else:
                # Remove the same leading blank lines as cleandoc() would.
                skip = len(lines[0]) + 1
                for line in lines[1:]:
                    if len(line) > indent:
                        break
                    skip += len(line) + 1
                doc = doc[skip:]
            depdoc = textwrap.indent(depdoc, ' ' * indent)
            doc = f'{depdoc}\n\n{doc}'
        newfunc.__doc__ = doc

        return newfunc


def _get_indent(lines):
    """
    Determines the leading whitespace that could be removed from all the lines.
    """
    indent = sys.maxsize
    for line in lines:
        content = len(line.lstrip())
        if content:
            indent = min(indent, len(line) - content)
    if indent == sys.maxsize:
        indent = 0
    return indent


def deprecate(*args, **kwargs):
    """
    Issues a DeprecationWarning, adds warning to `old_name`'s
    docstring, rebinds ``old_name.__name__`` and returns the new
    function object.

    This function may also be used as a decorator.

    .. deprecated:: 2.0
        Use `~warnings.warn` with :exc:`DeprecationWarning` instead.

    Parameters
    ----------
    func : function
        The function to be deprecated.
    old_name : str, optional
        The name of the function to be deprecated. Default is None, in
        which case the name of `func` is used.
    new_name : str, optional
        The new name for the function. Default is None, in which case the
        deprecation message is that `old_name` is deprecated. If given, the
        deprecation message is that `old_name` is deprecated and `new_name`
        should be used instead.
    message : str, optional
        Additional explanation of the deprecation.  Displayed in the
        docstring after the warning.

    Returns
    -------
    old_func : function
        The deprecated function.

    Examples
    --------
    Note that ``olduint`` returns a value after printing Deprecation
    Warning:

    >>> olduint = np.lib.utils.deprecate(np.uint)
    DeprecationWarning: `uint64` is deprecated! # may vary
    >>> olduint(6)
    6

    """
    # Deprecate may be run as a function or as a decorator
    # If run as a function, we initialise the decorator class
    # and execute its __call__ method.

    # Deprecated in NumPy 2.0, 2023-07-11
    warnings.warn(
        "`deprecate` is deprecated, "
        "use `warn` with `DeprecationWarning` instead. "
        "(deprecated in NumPy 2.0)",
        DeprecationWarning,
        stacklevel=2
    )

    if args:
        fn = args[0]
        args = args[1:]

        return _Deprecate(*args, **kwargs)(fn)
    else:
        return _Deprecate(*args, **kwargs)


def deprecate_with_doc(msg):
    """
    Deprecates a function and includes the deprecation in its docstring.

    .. deprecated:: 2.0
        Use `~warnings.warn` with :exc:`DeprecationWarning` instead.

    This function is used as a decorator. It returns an object that can be
    used to issue a DeprecationWarning, by passing the to-be decorated
    function as argument, this adds warning to the to-be decorated function's
    docstring and returns the new function object.

    See Also
    --------
    deprecate : Decorate a function such that it issues a
                :exc:`DeprecationWarning`

    Parameters
    ----------
    msg : str
        Additional explanation of the deprecation. Displayed in the
        docstring after the warning.

    Returns
    -------
    obj : object

    """

    # Deprecated in NumPy 2.0, 2023-07-11
    warnings.warn(
        "`deprecate` is deprecated, "
        "use `warn` with `DeprecationWarning` instead. "
        "(deprecated in NumPy 2.0)",
        DeprecationWarning,
        stacklevel=2
    )

    return _Deprecate(message=msg)


#-----------------------------------------------------------------------------


# NOTE:  pydoc defines a help function which works similarly to this
#  except it uses a pager to take over the screen.

# combine name and arguments and split to multiple lines of width
# characters.  End lines on a comma and begin argument list indented with
# the rest of the arguments.
def _split_line(name, arguments, width):
    firstwidth = len(name)
    k = firstwidth
    newstr = name
    sepstr = ", "
    arglist = arguments.split(sepstr)
    for argument in arglist:
        if k == firstwidth:
            addstr = ""
        else:
            addstr = sepstr
        k = k + len(argument) + len(addstr)
        if k > width:
            k = firstwidth + 1 + len(argument)
            newstr = newstr + ",\n" + " " * (firstwidth + 2) + argument
        else:
            newstr = newstr + addstr + argument
    return newstr


_namedict = None
_dictlist = None

# Traverse all module directories underneath globals
# to see if something is defined
def _makenamedict(module='numpy'):
    module = __import__(module, globals(), locals(), [])
    thedict = {module.__name__: module.__dict__}
    dictlist = [module.__name__]
    totraverse = [module.__dict__]
    while True:
        if len(totraverse) == 0:
            break
        thisdict = totraverse.pop(0)
        for x in thisdict.keys():
            if isinstance(thisdict[x], types.ModuleType):
                modname = thisdict[x].__name__
                if modname not in dictlist:
                    moddict = thisdict[x].__dict__
                    dictlist.append(modname)
                    totraverse.append(moddict)
                    thedict[modname] = moddict
    return thedict, dictlist


def _info(obj, output=None):
    """Provide information about ndarray obj.

    Parameters
    ----------
    obj : ndarray
        Must be ndarray, not checked.
    output
        Where printed output goes.

    Notes
    -----
    Copied over from the numarray module prior to its removal.
    Adapted somewhat as only numpy is an option now.

    Called by info.

    """
    extra = ""
    tic = ""
    bp = lambda x: x
    cls = getattr(obj, '__class__', type(obj))
    nm = getattr(cls, '__name__', cls)
    strides = obj.strides
    endian = obj.dtype.byteorder

    if output is None:
        output = sys.stdout

    print("class: ", nm, file=output)
    print("shape: ", obj.shape, file=output)
    print("strides: ", strides, file=output)
    print("itemsize: ", obj.itemsize, file=output)
    print("aligned: ", bp(obj.flags.aligned), file=output)
    print("contiguous: ", bp(obj.flags.contiguous), file=output)
    print("fortran: ", obj.flags.fortran, file=output)
    print(
        f"data pointer: {hex(obj.ctypes._as_parameter_.value)}{extra}",
        file=output
        )
    print("byteorder: ", end=' ', file=output)
    if endian in ['|', '=']:
        print(f"{tic}{sys.byteorder}{tic}", file=output)
        byteswap = False
    elif endian == '>':
        print(f"{tic}big{tic}", file=output)
        byteswap = sys.byteorder != "big"
    else:
        print(f"{tic}little{tic}", file=output)
        byteswap = sys.byteorder != "little"
    print("byteswap: ", bp(byteswap), file=output)
    print(f"type: {obj.dtype}", file=output)


@set_module('numpy')
def info(object=None, maxwidth=76, output=None, toplevel='numpy'):
    """
    Get help information for an array, function, class, or module.

    Parameters
    ----------
    object : object or str, optional
        Input object or name to get information about. If `object` is
        an `ndarray` instance, information about the array is printed.
        If `object` is a numpy object, its docstring is given. If it is
        a string, available modules are searched for matching objects.
        If None, information about `info` itself is returned.
    maxwidth : int, optional
        Printing width.
    output : file like object, optional
        File like object that the output is written to, default is
        ``None``, in which case ``sys.stdout`` will be used.
        The object has to be opened in 'w' or 'a' mode.
    toplevel : str, optional
        Start search at this level.

    Notes
    -----
    When used interactively with an object, ``np.info(obj)`` is equivalent
    to ``help(obj)`` on the Python prompt or ``obj?`` on the IPython
    prompt.

    Examples
    --------
    >>> np.info(np.polyval) # doctest: +SKIP
       polyval(p, x)
         Evaluate the polynomial p at x.
         ...

    When using a string for `object` it is possible to get multiple results.

    >>> np.info('fft') # doctest: +SKIP
         *** Found in numpy ***
    Core FFT routines
    ...
         *** Found in numpy.fft ***
     fft(a, n=None, axis=-1)
    ...
         *** Repeat reference found in numpy.fft.fftpack ***
         *** Total of 3 references found. ***

    When the argument is an array, information about the array is printed.

    >>> a = np.array([[1 + 2j, 3, -4], [-5j, 6, 0]], dtype=np.complex64)
    >>> np.info(a)
    class:  ndarray
    shape:  (2, 3)
    strides:  (24, 8)
    itemsize:  8
    aligned:  True
    contiguous:  True
    fortran:  False
    data pointer: 0x562b6e0d2860  # may vary
    byteorder:  little
    byteswap:  False
    type: complex64

    """
    global _namedict, _dictlist
    # Local import to speed up numpy's import time.
    import inspect
    import pydoc

    if (hasattr(object, '_ppimport_importer') or
           hasattr(object, '_ppimport_module')):
        object = object._ppimport_module
    elif hasattr(object, '_ppimport_attr'):
        object = object._ppimport_attr

    if output is None:
        output = sys.stdout

    if object is None:
        info(info)
    elif isinstance(object, ndarray):
        _info(object, output=output)
    elif isinstance(object, str):
        if _namedict is None:
            _namedict, _dictlist = _makenamedict(toplevel)
        numfound = 0
        objlist = []
        for namestr in _dictlist:
            try:
                obj = _namedict[namestr][object]
                if id(obj) in objlist:
                    print(f"\n     *** Repeat reference found in {namestr} *** ",
                          file=output
                          )
                else:
                    objlist.append(id(obj))
                    print(f"     *** Found in {namestr} ***", file=output)
                    info(obj)
                    print("-" * maxwidth, file=output)
                numfound += 1
            except KeyError:
                pass
        if numfound == 0:
            print(f"Help for {object} not found.", file=output)
        else:
            print("\n     "
                  "*** Total of %d references found. ***" % numfound,
                  file=output
                  )

    elif inspect.isfunction(object) or inspect.ismethod(object):
        name = object.__name__
        try:
            arguments = str(inspect.signature(object))
        except Exception:
            arguments = "()"

        if len(name + arguments) > maxwidth:
            argstr = _split_line(name, arguments, maxwidth)
        else:
            argstr = name + arguments

        print(" " + argstr + "\n", file=output)
        print(inspect.getdoc(object), file=output)

    elif inspect.isclass(object):
        name = object.__name__
        try:
            arguments = str(inspect.signature(object))
        except Exception:
            arguments = "()"

        if len(name + arguments) > maxwidth:
            argstr = _split_line(name, arguments, maxwidth)
        else:
            argstr = name + arguments

        print(" " + argstr + "\n", file=output)
        doc1 = inspect.getdoc(object)
        if doc1 is None:
            if hasattr(object, '__init__'):
                print(inspect.getdoc(object.__init__), file=output)
        else:
            print(inspect.getdoc(object), file=output)

        methods = pydoc.allmethods(object)

        public_methods = [meth for meth in methods if meth[0] != '_']
        if public_methods:
            print("\n\nMethods:\n", file=output)
            for meth in public_methods:
                thisobj = getattr(object, meth, None)
                if thisobj is not None:
                    methstr, other = pydoc.splitdoc(
                            inspect.getdoc(thisobj) or "None"
                            )
                print(f"  {meth}  --  {methstr}", file=output)

    elif hasattr(object, '__doc__'):
        print(inspect.getdoc(object), file=output)


def safe_eval(source):
    """
    Protected string evaluation.

    .. deprecated:: 2.0
        Use `ast.literal_eval` instead.

    Evaluate a string containing a Python literal expression without
    allowing the execution of arbitrary non-literal code.

    .. warning::

        This function is identical to :py:meth:`ast.literal_eval` and
        has the same security implications.  It may not always be safe
        to evaluate large input strings.

    Parameters
    ----------
    source : str
        The string to evaluate.

    Returns
    -------
    obj : object
       The result of evaluating `source`.

    Raises
    ------
    SyntaxError
        If the code has invalid Python syntax, or if it contains
        non-literal code.

    Examples
    --------
    >>> np.safe_eval('1')
    1
    >>> np.safe_eval('[1, 2, 3]')
    [1, 2, 3]
    >>> np.safe_eval('{"foo": ("bar", 10.0)}')
    {'foo': ('bar', 10.0)}

    >>> np.safe_eval('import os')
    Traceback (most recent call last):
      ...
    SyntaxError: invalid syntax

    >>> np.safe_eval('open("/home/user/.ssh/id_dsa").read()')
    Traceback (most recent call last):
      ...
    ValueError: malformed node or string: <_ast.Call object at 0x...>

    """

    # Deprecated in NumPy 2.0, 2023-07-11
    warnings.warn(
        "`safe_eval` is deprecated. Use `ast.literal_eval` instead. "
        "Be aware of security implications, such as memory exhaustion "
        "based attacks (deprecated in NumPy 2.0)",
        DeprecationWarning,
        stacklevel=2
    )

    # Local import to speed up numpy's import time.
    import ast
    return ast.literal_eval(source)


def _median_nancheck(data, result, axis):
    """
    Utility function to check median result from data for NaN values at the end
    and return NaN in that case. Input result can also be a MaskedArray.

    Parameters
    ----------
    data : array
        Sorted input data to median function
    result : Array or MaskedArray
        Result of median function.
    axis : int
        Axis along which the median was computed.

    Returns
    -------
    result : scalar or ndarray
        Median or NaN in axes which contained NaN in the input.  If the input
        was an array, NaN will be inserted in-place.  If a scalar, either the
        input itself or a scalar NaN.
    """
    if data.size == 0:
        return result
    potential_nans = data.take(-1, axis=axis)
    n = np.isnan(potential_nans)
    # masked NaN values are ok, although for masked the copyto may fail for
    # unmasked ones (this was always broken) when the result is a scalar.
    if np.ma.isMaskedArray(n):
        n = n.filled(False)

    if not n.any():
        return result

    # Without given output, it is possible that the current result is a
    # numpy scalar, which is not writeable.  If so, just return nan.
    if isinstance(result, np.generic):
        return potential_nans

    # Otherwise copy NaNs (if there are any)
    np.copyto(result, potential_nans, where=n)
    return result

def _opt_info():
    """
    Returns a string containing the CPU features supported
    by the current build.

    The format of the string can be explained as follows:
        - Dispatched features supported by the running machine end with `*`.
        - Dispatched features not supported by the running machine
          end with `?`.
        - Remaining features represent the baseline.

    Returns:
        str: A formatted string indicating the supported CPU features.
    """
    from numpy._core._multiarray_umath import (
        __cpu_baseline__,
        __cpu_dispatch__,
        __cpu_features__,
    )

    if len(__cpu_baseline__) == 0 and len(__cpu_dispatch__) == 0:
        return ''

    enabled_features = ' '.join(__cpu_baseline__)
    for feature in __cpu_dispatch__:
        if __cpu_features__[feature]:
            enabled_features += f" {feature}*"
        else:
            enabled_features += f" {feature}?"

    return enabled_features

def drop_metadata(dtype, /):
    """
    Returns the dtype unchanged if it contained no metadata or a copy of the
    dtype if it (or any of its structure dtypes) contained metadata.

    This utility is used by `np.save` and `np.savez` to drop metadata before
    saving.

    .. note::

        Due to its limitation this function may move to a more appropriate
        home or change in the future and is considered semi-public API only.

    .. warning::

        This function does not preserve more strange things like record dtypes
        and user dtypes may simply return the wrong thing.  If you need to be
        sure about the latter, check the result with:
        ``np.can_cast(new_dtype, dtype, casting="no")``.

    """
    if dtype.fields is not None:
        found_metadata = dtype.metadata is not None

        names = []
        formats = []
        offsets = []
        titles = []
        for name, field in dtype.fields.items():
            field_dt = drop_metadata(field[0])
            if field_dt is not field[0]:
                found_metadata = True

            names.append(name)
            formats.append(field_dt)
            offsets.append(field[1])
            titles.append(None if len(field) < 3 else field[2])

        if not found_metadata:
            return dtype

        structure = {
            'names': names, 'formats': formats, 'offsets': offsets, 'titles': titles,
            'itemsize': dtype.itemsize}

        # NOTE: Could pass (dtype.type, structure) to preserve record dtypes...
        return np.dtype(structure, align=dtype.isalignedstruct)
    elif dtype.subdtype is not None:
        # subarray dtype
        subdtype, shape = dtype.subdtype
        new_subdtype = drop_metadata(subdtype)
        if dtype.metadata is None and new_subdtype is subdtype:
            return dtype

        return np.dtype((new_subdtype, shape))
    else:
        # Normal unstructured dtype
        if dtype.metadata is None:
            return dtype
        # Note that `dt.str` doesn't round-trip e.g. for user-dtypes.
        return np.dtype(dtype.str)

# === atelier-kyo-manager/app\routes\image_crawler_route.py ===
from flask import Blueprint, render_template, request
from app.utils.image_crawler import crawl_images

bp = Blueprint('image_crawler', __name__)

@bp.route('/image_crawler', methods=['GET', 'POST'])
def image_crawler():
    message = ''
    if request.method == 'POST':
        keyword = request.form.get('keyword')
        use_bing = request.form.get('use_bing') == 'on'
        use_google = request.form.get('use_google') == 'on'
        engines = []
        if use_bing:
            engines.append('bing')
        if use_google:
            engines.append('google')
        if keyword and engines:
            save_dir = crawl_images(keyword, max_num=50, engines=engines)
            message = f"「{keyword}」の画像を {', '.join(engines)} で収集し、{save_dir} に保存しました。"
        elif not engines:
            message = "検索エンジンを1つ以上選択してください。"
    return render_template('image_crawler.html', message=message)

# === atelier-kyo-manager/app\models_backup.py ===
# app/models.py
from app import db # app/__init__.py で定義されたdbオブジェクトをインポート

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    brand = db.Column(db.String(60))
    purchase_price = db.Column(db.Float)
    selling_price = db.Column(db.Float)
    supplier_url = db.Column(db.String(500))
    image_url = db.Column(db.String(500))
    stock_status = db.Column(db.Boolean, default=True)
    profit_margin = db.Column(db.Float) # これは手動で更新するか、calculate_profitで計算後に保存

    def calculate_profit(self):
        # 手数料15%仮定。必要であれば、config.py などで設定できるようにしても良い
        # shipping_costは別途考慮が必要な場合、引数で渡すか、Productモデルに含める
        return self.selling_price - self.purchase_price - (self.selling_price * 0.15)

    def __repr__(self):
        return f"<Product {self.name}>"

# === atelier-kyo-manager/app\__init__.py ===
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_wtf.csrf import CSRFProtect

db = SQLAlchemy()
migrate = Migrate()
csrf = CSRFProtect()

def create_app():
    app = Flask(__name__)
    app.config.from_object('config.Config')

    db.init_app(app)
    migrate.init_app(app, db)
    csrf.init_app(app)

    from app.routes.image_crawler_route import bp as image_crawler_bp
    app.register_blueprint(image_crawler_bp)

    return app

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\_datasource.py ===
"""A file interface for handling local and remote data files.

The goal of datasource is to abstract some of the file system operations
when dealing with data files so the researcher doesn't have to know all the
low-level details.  Through datasource, a researcher can obtain and use a
file with one function call, regardless of location of the file.

DataSource is meant to augment standard python libraries, not replace them.
It should work seamlessly with standard file IO operations and the os
module.

DataSource files can originate locally or remotely:

- local files : '/home/guido/src/local/data.txt'
- URLs (http, ftp, ...) : 'http://www.scipy.org/not/real/data.txt'

DataSource files can also be compressed or uncompressed.  Currently only
gzip, bz2 and xz are supported.

Example::

    >>> # Create a DataSource, use os.curdir (default) for local storage.
    >>> from numpy import DataSource
    >>> ds = DataSource()
    >>>
    >>> # Open a remote file.
    >>> # DataSource downloads the file, stores it locally in:
    >>> #     './www.google.com/index.html'
    >>> # opens the file and returns a file object.
    >>> fp = ds.open('http://www.google.com/') # doctest: +SKIP
    >>>
    >>> # Use the file as you normally would
    >>> fp.read() # doctest: +SKIP
    >>> fp.close() # doctest: +SKIP

"""
import os

from numpy._utils import set_module

_open = open


def _check_mode(mode, encoding, newline):
    """Check mode and that encoding and newline are compatible.

    Parameters
    ----------
    mode : str
        File open mode.
    encoding : str
        File encoding.
    newline : str
        Newline for text files.

    """
    if "t" in mode:
        if "b" in mode:
            raise ValueError(f"Invalid mode: {mode!r}")
    else:
        if encoding is not None:
            raise ValueError("Argument 'encoding' not supported in binary mode")
        if newline is not None:
            raise ValueError("Argument 'newline' not supported in binary mode")


# Using a class instead of a module-level dictionary
# to reduce the initial 'import numpy' overhead by
# deferring the import of lzma, bz2 and gzip until needed

# TODO: .zip support, .tar support?
class _FileOpeners:
    """
    Container for different methods to open (un-)compressed files.

    `_FileOpeners` contains a dictionary that holds one method for each
    supported file format. Attribute lookup is implemented in such a way
    that an instance of `_FileOpeners` itself can be indexed with the keys
    of that dictionary. Currently uncompressed files as well as files
    compressed with ``gzip``, ``bz2`` or ``xz`` compression are supported.

    Notes
    -----
    `_file_openers`, an instance of `_FileOpeners`, is made available for
    use in the `_datasource` module.

    Examples
    --------
    >>> import gzip
    >>> np.lib._datasource._file_openers.keys()
    [None, '.bz2', '.gz', '.xz', '.lzma']
    >>> np.lib._datasource._file_openers['.gz'] is gzip.open
    True

    """

    def __init__(self):
        self._loaded = False
        self._file_openers = {None: open}

    def _load(self):
        if self._loaded:
            return

        try:
            import bz2
            self._file_openers[".bz2"] = bz2.open
        except ImportError:
            pass

        try:
            import gzip
            self._file_openers[".gz"] = gzip.open
        except ImportError:
            pass

        try:
            import lzma
            self._file_openers[".xz"] = lzma.open
            self._file_openers[".lzma"] = lzma.open
        except (ImportError, AttributeError):
            # There are incompatible backports of lzma that do not have the
            # lzma.open attribute, so catch that as well as ImportError.
            pass

        self._loaded = True

    def keys(self):
        """
        Return the keys of currently supported file openers.

        Parameters
        ----------
        None

        Returns
        -------
        keys : list
            The keys are None for uncompressed files and the file extension
            strings (i.e. ``'.gz'``, ``'.xz'``) for supported compression
            methods.

        """
        self._load()
        return list(self._file_openers.keys())

    def __getitem__(self, key):
        self._load()
        return self._file_openers[key]


_file_openers = _FileOpeners()

def open(path, mode='r', destpath=os.curdir, encoding=None, newline=None):
    """
    Open `path` with `mode` and return the file object.

    If ``path`` is an URL, it will be downloaded, stored in the
    `DataSource` `destpath` directory and opened from there.

    Parameters
    ----------
    path : str or pathlib.Path
        Local file path or URL to open.
    mode : str, optional
        Mode to open `path`. Mode 'r' for reading, 'w' for writing, 'a' to
        append. Available modes depend on the type of object specified by
        path.  Default is 'r'.
    destpath : str, optional
        Path to the directory where the source file gets downloaded to for
        use.  If `destpath` is None, a temporary directory will be created.
        The default path is the current directory.
    encoding : {None, str}, optional
        Open text file with given encoding. The default encoding will be
        what `open` uses.
    newline : {None, str}, optional
        Newline to use when reading text file.

    Returns
    -------
    out : file object
        The opened file.

    Notes
    -----
    This is a convenience function that instantiates a `DataSource` and
    returns the file object from ``DataSource.open(path)``.

    """

    ds = DataSource(destpath)
    return ds.open(path, mode, encoding=encoding, newline=newline)


@set_module('numpy.lib.npyio')
class DataSource:
    """
    DataSource(destpath='.')

    A generic data source file (file, http, ftp, ...).

    DataSources can be local files or remote files/URLs.  The files may
    also be compressed or uncompressed. DataSource hides some of the
    low-level details of downloading the file, allowing you to simply pass
    in a valid file path (or URL) and obtain a file object.

    Parameters
    ----------
    destpath : str or None, optional
        Path to the directory where the source file gets downloaded to for
        use.  If `destpath` is None, a temporary directory will be created.
        The default path is the current directory.

    Notes
    -----
    URLs require a scheme string (``http://``) to be used, without it they
    will fail::

        >>> repos = np.lib.npyio.DataSource()
        >>> repos.exists('www.google.com/index.html')
        False
        >>> repos.exists('http://www.google.com/index.html')
        True

    Temporary directories are deleted when the DataSource is deleted.

    Examples
    --------
    ::

        >>> ds = np.lib.npyio.DataSource('/home/guido')
        >>> urlname = 'http://www.google.com/'
        >>> gfile = ds.open('http://www.google.com/')
        >>> ds.abspath(urlname)
        '/home/guido/www.google.com/index.html'

        >>> ds = np.lib.npyio.DataSource(None)  # use with temporary file
        >>> ds.open('/home/guido/foobar.txt')
        <open file '/home/guido.foobar.txt', mode 'r' at 0x91d4430>
        >>> ds.abspath('/home/guido/foobar.txt')
        '/tmp/.../home/guido/foobar.txt'

    """

    def __init__(self, destpath=os.curdir):
        """Create a DataSource with a local path at destpath."""
        if destpath:
            self._destpath = os.path.abspath(destpath)
            self._istmpdest = False
        else:
            import tempfile  # deferring import to improve startup time
            self._destpath = tempfile.mkdtemp()
            self._istmpdest = True

    def __del__(self):
        # Remove temp directories
        if hasattr(self, '_istmpdest') and self._istmpdest:
            import shutil

            shutil.rmtree(self._destpath)

    def _iszip(self, filename):
        """Test if the filename is a zip file by looking at the file extension.

        """
        fname, ext = os.path.splitext(filename)
        return ext in _file_openers.keys()

    def _iswritemode(self, mode):
        """Test if the given mode will open a file for writing."""

        # Currently only used to test the bz2 files.
        _writemodes = ("w", "+")
        return any(c in _writemodes for c in mode)

    def _splitzipext(self, filename):
        """Split zip extension from filename and return filename.

        Returns
        -------
        base, zip_ext : {tuple}

        """

        if self._iszip(filename):
            return os.path.splitext(filename)
        else:
            return filename, None

    def _possible_names(self, filename):
        """Return a tuple containing compressed filename variations."""
        names = [filename]
        if not self._iszip(filename):
            for zipext in _file_openers.keys():
                if zipext:
                    names.append(filename + zipext)
        return names

    def _isurl(self, path):
        """Test if path is a net location.  Tests the scheme and netloc."""

        # We do this here to reduce the 'import numpy' initial import time.
        from urllib.parse import urlparse

        # BUG : URLs require a scheme string ('http://') to be used.
        #       www.google.com will fail.
        #       Should we prepend the scheme for those that don't have it and
        #       test that also?  Similar to the way we append .gz and test for
        #       for compressed versions of files.

        scheme, netloc, upath, uparams, uquery, ufrag = urlparse(path)
        return bool(scheme and netloc)

    def _cache(self, path):
        """Cache the file specified by path.

        Creates a copy of the file in the datasource cache.

        """
        # We import these here because importing them is slow and
        # a significant fraction of numpy's total import time.
        import shutil
        from urllib.request import urlopen

        upath = self.abspath(path)

        # ensure directory exists
        if not os.path.exists(os.path.dirname(upath)):
            os.makedirs(os.path.dirname(upath))

        # TODO: Doesn't handle compressed files!
        if self._isurl(path):
            with urlopen(path) as openedurl:
                with _open(upath, 'wb') as f:
                    shutil.copyfileobj(openedurl, f)
        else:
            shutil.copyfile(path, upath)
        return upath

    def _findfile(self, path):
        """Searches for ``path`` and returns full path if found.

        If path is an URL, _findfile will cache a local copy and return the
        path to the cached file.  If path is a local file, _findfile will
        return a path to that local file.

        The search will include possible compressed versions of the file
        and return the first occurrence found.

        """

        # Build list of possible local file paths
        if not self._isurl(path):
            # Valid local paths
            filelist = self._possible_names(path)
            # Paths in self._destpath
            filelist += self._possible_names(self.abspath(path))
        else:
            # Cached URLs in self._destpath
            filelist = self._possible_names(self.abspath(path))
            # Remote URLs
            filelist = filelist + self._possible_names(path)

        for name in filelist:
            if self.exists(name):
                if self._isurl(name):
                    name = self._cache(name)
                return name
        return None

    def abspath(self, path):
        """
        Return absolute path of file in the DataSource directory.

        If `path` is an URL, then `abspath` will return either the location
        the file exists locally or the location it would exist when opened
        using the `open` method.

        Parameters
        ----------
        path : str or pathlib.Path
            Can be a local file or a remote URL.

        Returns
        -------
        out : str
            Complete path, including the `DataSource` destination directory.

        Notes
        -----
        The functionality is based on `os.path.abspath`.

        """
        # We do this here to reduce the 'import numpy' initial import time.
        from urllib.parse import urlparse

        # TODO:  This should be more robust.  Handles case where path includes
        #        the destpath, but not other sub-paths. Failing case:
        #        path = /home/guido/datafile.txt
        #        destpath = /home/alex/
        #        upath = self.abspath(path)
        #        upath == '/home/alex/home/guido/datafile.txt'

        # handle case where path includes self._destpath
        splitpath = path.split(self._destpath, 2)
        if len(splitpath) > 1:
            path = splitpath[1]
        scheme, netloc, upath, uparams, uquery, ufrag = urlparse(path)
        netloc = self._sanitize_relative_path(netloc)
        upath = self._sanitize_relative_path(upath)
        return os.path.join(self._destpath, netloc, upath)

    def _sanitize_relative_path(self, path):
        """Return a sanitised relative path for which
        os.path.abspath(os.path.join(base, path)).startswith(base)
        """
        last = None
        path = os.path.normpath(path)
        while path != last:
            last = path
            # Note: os.path.join treats '/' as os.sep on Windows
            path = path.lstrip(os.sep).lstrip('/')
            path = path.lstrip(os.pardir).removeprefix('..')
            drive, path = os.path.splitdrive(path)  # for Windows
        return path

    def exists(self, path):
        """
        Test if path exists.

        Test if `path` exists as (and in this order):

        - a local file.
        - a remote URL that has been downloaded and stored locally in the
          `DataSource` directory.
        - a remote URL that has not been downloaded, but is valid and
          accessible.

        Parameters
        ----------
        path : str or pathlib.Path
            Can be a local file or a remote URL.

        Returns
        -------
        out : bool
            True if `path` exists.

        Notes
        -----
        When `path` is an URL, `exists` will return True if it's either
        stored locally in the `DataSource` directory, or is a valid remote
        URL.  `DataSource` does not discriminate between the two, the file
        is accessible if it exists in either location.

        """

        # First test for local path
        if os.path.exists(path):
            return True

        # We import this here because importing urllib is slow and
        # a significant fraction of numpy's total import time.
        from urllib.error import URLError
        from urllib.request import urlopen

        # Test cached url
        upath = self.abspath(path)
        if os.path.exists(upath):
            return True

        # Test remote url
        if self._isurl(path):
            try:
                netfile = urlopen(path)
                netfile.close()
                del netfile
                return True
            except URLError:
                return False
        return False

    def open(self, path, mode='r', encoding=None, newline=None):
        """
        Open and return file-like object.

        If `path` is an URL, it will be downloaded, stored in the
        `DataSource` directory and opened from there.

        Parameters
        ----------
        path : str or pathlib.Path
            Local file path or URL to open.
        mode : {'r', 'w', 'a'}, optional
            Mode to open `path`.  Mode 'r' for reading, 'w' for writing,
            'a' to append. Available modes depend on the type of object
            specified by `path`. Default is 'r'.
        encoding : {None, str}, optional
            Open text file with given encoding. The default encoding will be
            what `open` uses.
        newline : {None, str}, optional
            Newline to use when reading text file.

        Returns
        -------
        out : file object
            File object.

        """

        # TODO: There is no support for opening a file for writing which
        #       doesn't exist yet (creating a file).  Should there be?

        # TODO: Add a ``subdir`` parameter for specifying the subdirectory
        #       used to store URLs in self._destpath.

        if self._isurl(path) and self._iswritemode(mode):
            raise ValueError("URLs are not writeable")

        # NOTE: _findfile will fail on a new file opened for writing.
        found = self._findfile(path)
        if found:
            _fname, ext = self._splitzipext(found)
            if ext == 'bz2':
                mode.replace("+", "")
            return _file_openers[ext](found, mode=mode,
                                      encoding=encoding, newline=newline)
        else:
            raise FileNotFoundError(f"{path} not found.")


class Repository (DataSource):
    """
    Repository(baseurl, destpath='.')

    A data repository where multiple DataSource's share a base
    URL/directory.

    `Repository` extends `DataSource` by prepending a base URL (or
    directory) to all the files it handles. Use `Repository` when you will
    be working with multiple files from one base URL.  Initialize
    `Repository` with the base URL, then refer to each file by its filename
    only.

    Parameters
    ----------
    baseurl : str
        Path to the local directory or remote location that contains the
        data files.
    destpath : str or None, optional
        Path to the directory where the source file gets downloaded to for
        use.  If `destpath` is None, a temporary directory will be created.
        The default path is the current directory.

    Examples
    --------
    To analyze all files in the repository, do something like this
    (note: this is not self-contained code)::

        >>> repos = np.lib._datasource.Repository('/home/user/data/dir/')
        >>> for filename in filelist:
        ...     fp = repos.open(filename)
        ...     fp.analyze()
        ...     fp.close()

    Similarly you could use a URL for a repository::

        >>> repos = np.lib._datasource.Repository('http://www.xyz.edu/data')

    """

    def __init__(self, baseurl, destpath=os.curdir):
        """Create a Repository with a shared url or directory of baseurl."""
        DataSource.__init__(self, destpath=destpath)
        self._baseurl = baseurl

    def __del__(self):
        DataSource.__del__(self)

    def _fullpath(self, path):
        """Return complete path for path.  Prepends baseurl if necessary."""
        splitpath = path.split(self._baseurl, 2)
        if len(splitpath) == 1:
            result = os.path.join(self._baseurl, path)
        else:
            result = path    # path contains baseurl already
        return result

    def _findfile(self, path):
        """Extend DataSource method to prepend baseurl to ``path``."""
        return DataSource._findfile(self, self._fullpath(path))

    def abspath(self, path):
        """
        Return absolute path of file in the Repository directory.

        If `path` is an URL, then `abspath` will return either the location
        the file exists locally or the location it would exist when opened
        using the `open` method.

        Parameters
        ----------
        path : str or pathlib.Path
            Can be a local file or a remote URL. This may, but does not
            have to, include the `baseurl` with which the `Repository` was
            initialized.

        Returns
        -------
        out : str
            Complete path, including the `DataSource` destination directory.

        """
        return DataSource.abspath(self, self._fullpath(path))

    def exists(self, path):
        """
        Test if path exists prepending Repository base URL to path.

        Test if `path` exists as (and in this order):

        - a local file.
        - a remote URL that has been downloaded and stored locally in the
          `DataSource` directory.
        - a remote URL that has not been downloaded, but is valid and
          accessible.

        Parameters
        ----------
        path : str or pathlib.Path
            Can be a local file or a remote URL. This may, but does not
            have to, include the `baseurl` with which the `Repository` was
            initialized.

        Returns
        -------
        out : bool
            True if `path` exists.

        Notes
        -----
        When `path` is an URL, `exists` will return True if it's either
        stored locally in the `DataSource` directory, or is a valid remote
        URL.  `DataSource` does not discriminate between the two, the file
        is accessible if it exists in either location.

        """
        return DataSource.exists(self, self._fullpath(path))

    def open(self, path, mode='r', encoding=None, newline=None):
        """
        Open and return file-like object prepending Repository base URL.

        If `path` is an URL, it will be downloaded, stored in the
        DataSource directory and opened from there.

        Parameters
        ----------
        path : str or pathlib.Path
            Local file path or URL to open. This may, but does not have to,
            include the `baseurl` with which the `Repository` was
            initialized.
        mode : {'r', 'w', 'a'}, optional
            Mode to open `path`.  Mode 'r' for reading, 'w' for writing,
            'a' to append. Available modes depend on the type of object
            specified by `path`. Default is 'r'.
        encoding : {None, str}, optional
            Open text file with given encoding. The default encoding will be
            what `open` uses.
        newline : {None, str}, optional
            Newline to use when reading text file.

        Returns
        -------
        out : file object
            File object.

        """
        return DataSource.open(self, self._fullpath(path), mode,
                               encoding=encoding, newline=newline)

    def listdir(self):
        """
        List files in the source Repository.

        Returns
        -------
        files : list of str or pathlib.Path
            List of file names (not containing a directory part).

        Notes
        -----
        Does not currently work for remote repositories.

        """
        if self._isurl(self._baseurl):
            raise NotImplementedError(
                  "Directory listing of URLs, not supported yet.")
        else:
            return os.listdir(self._baseurl)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\_type_check_impl.py ===
"""Automatically adapted for numpy Sep 19, 2005 by convertcode.py

"""
import functools

__all__ = ['iscomplexobj', 'isrealobj', 'imag', 'iscomplex',
           'isreal', 'nan_to_num', 'real', 'real_if_close',
           'typename', 'mintypecode',
           'common_type']

import numpy._core.numeric as _nx
from numpy._core import getlimits, overrides
from numpy._core.numeric import asanyarray, asarray, isnan, zeros
from numpy._utils import set_module

from ._ufunclike_impl import isneginf, isposinf

array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


_typecodes_by_elsize = 'GDFgdfQqLlIiHhBb?'


@set_module('numpy')
def mintypecode(typechars, typeset='GDFgdf', default='d'):
    """
    Return the character for the minimum-size type to which given types can
    be safely cast.

    The returned type character must represent the smallest size dtype such
    that an array of the returned type can handle the data from an array of
    all types in `typechars` (or if `typechars` is an array, then its
    dtype.char).

    Parameters
    ----------
    typechars : list of str or array_like
        If a list of strings, each string should represent a dtype.
        If array_like, the character representation of the array dtype is used.
    typeset : str or list of str, optional
        The set of characters that the returned character is chosen from.
        The default set is 'GDFgdf'.
    default : str, optional
        The default character, this is returned if none of the characters in
        `typechars` matches a character in `typeset`.

    Returns
    -------
    typechar : str
        The character representing the minimum-size type that was found.

    See Also
    --------
    dtype

    Examples
    --------
    >>> import numpy as np
    >>> np.mintypecode(['d', 'f', 'S'])
    'd'
    >>> x = np.array([1.1, 2-3.j])
    >>> np.mintypecode(x)
    'D'

    >>> np.mintypecode('abceh', default='G')
    'G'

    """
    typecodes = ((isinstance(t, str) and t) or asarray(t).dtype.char
                 for t in typechars)
    intersection = {t for t in typecodes if t in typeset}
    if not intersection:
        return default
    if 'F' in intersection and 'd' in intersection:
        return 'D'
    return min(intersection, key=_typecodes_by_elsize.index)


def _real_dispatcher(val):
    return (val,)


@array_function_dispatch(_real_dispatcher)
def real(val):
    """
    Return the real part of the complex argument.

    Parameters
    ----------
    val : array_like
        Input array.

    Returns
    -------
    out : ndarray or scalar
        The real component of the complex argument. If `val` is real, the type
        of `val` is used for the output.  If `val` has complex elements, the
        returned type is float.

    See Also
    --------
    real_if_close, imag, angle

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1+2j, 3+4j, 5+6j])
    >>> a.real
    array([1.,  3.,  5.])
    >>> a.real = 9
    >>> a
    array([9.+2.j,  9.+4.j,  9.+6.j])
    >>> a.real = np.array([9, 8, 7])
    >>> a
    array([9.+2.j,  8.+4.j,  7.+6.j])
    >>> np.real(1 + 1j)
    1.0

    """
    try:
        return val.real
    except AttributeError:
        return asanyarray(val).real


def _imag_dispatcher(val):
    return (val,)


@array_function_dispatch(_imag_dispatcher)
def imag(val):
    """
    Return the imaginary part of the complex argument.

    Parameters
    ----------
    val : array_like
        Input array.

    Returns
    -------
    out : ndarray or scalar
        The imaginary component of the complex argument. If `val` is real,
        the type of `val` is used for the output.  If `val` has complex
        elements, the returned type is float.

    See Also
    --------
    real, angle, real_if_close

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1+2j, 3+4j, 5+6j])
    >>> a.imag
    array([2.,  4.,  6.])
    >>> a.imag = np.array([8, 10, 12])
    >>> a
    array([1. +8.j,  3.+10.j,  5.+12.j])
    >>> np.imag(1 + 1j)
    1.0

    """
    try:
        return val.imag
    except AttributeError:
        return asanyarray(val).imag


def _is_type_dispatcher(x):
    return (x,)


@array_function_dispatch(_is_type_dispatcher)
def iscomplex(x):
    """
    Returns a bool array, where True if input element is complex.

    What is tested is whether the input has a non-zero imaginary part, not if
    the input type is complex.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    out : ndarray of bools
        Output array.

    See Also
    --------
    isreal
    iscomplexobj : Return True if x is a complex type or an array of complex
                   numbers.

    Examples
    --------
    >>> import numpy as np
    >>> np.iscomplex([1+1j, 1+0j, 4.5, 3, 2, 2j])
    array([ True, False, False, False, False,  True])

    """
    ax = asanyarray(x)
    if issubclass(ax.dtype.type, _nx.complexfloating):
        return ax.imag != 0
    res = zeros(ax.shape, bool)
    return res[()]   # convert to scalar if needed


@array_function_dispatch(_is_type_dispatcher)
def isreal(x):
    """
    Returns a bool array, where True if input element is real.

    If element has complex type with zero imaginary part, the return value
    for that element is True.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    out : ndarray, bool
        Boolean array of same shape as `x`.

    Notes
    -----
    `isreal` may behave unexpectedly for string or object arrays (see examples)

    See Also
    --------
    iscomplex
    isrealobj : Return True if x is not a complex type.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1+1j, 1+0j, 4.5, 3, 2, 2j], dtype=complex)
    >>> np.isreal(a)
    array([False,  True,  True,  True,  True, False])

    The function does not work on string arrays.

    >>> a = np.array([2j, "a"], dtype="U")
    >>> np.isreal(a)  # Warns about non-elementwise comparison
    False

    Returns True for all elements in input array of ``dtype=object`` even if
    any of the elements is complex.

    >>> a = np.array([1, "2", 3+4j], dtype=object)
    >>> np.isreal(a)
    array([ True,  True,  True])

    isreal should not be used with object arrays

    >>> a = np.array([1+2j, 2+1j], dtype=object)
    >>> np.isreal(a)
    array([ True,  True])

    """
    return imag(x) == 0


@array_function_dispatch(_is_type_dispatcher)
def iscomplexobj(x):
    """
    Check for a complex type or an array of complex numbers.

    The type of the input is checked, not the value. Even if the input
    has an imaginary part equal to zero, `iscomplexobj` evaluates to True.

    Parameters
    ----------
    x : any
        The input can be of any type and shape.

    Returns
    -------
    iscomplexobj : bool
        The return value, True if `x` is of a complex type or has at least
        one complex element.

    See Also
    --------
    isrealobj, iscomplex

    Examples
    --------
    >>> import numpy as np
    >>> np.iscomplexobj(1)
    False
    >>> np.iscomplexobj(1+0j)
    True
    >>> np.iscomplexobj([3, 1+0j, True])
    True

    """
    try:
        dtype = x.dtype
        type_ = dtype.type
    except AttributeError:
        type_ = asarray(x).dtype.type
    return issubclass(type_, _nx.complexfloating)


@array_function_dispatch(_is_type_dispatcher)
def isrealobj(x):
    """
    Return True if x is a not complex type or an array of complex numbers.

    The type of the input is checked, not the value. So even if the input
    has an imaginary part equal to zero, `isrealobj` evaluates to False
    if the data type is complex.

    Parameters
    ----------
    x : any
        The input can be of any type and shape.

    Returns
    -------
    y : bool
        The return value, False if `x` is of a complex type.

    See Also
    --------
    iscomplexobj, isreal

    Notes
    -----
    The function is only meant for arrays with numerical values but it
    accepts all other objects. Since it assumes array input, the return
    value of other objects may be True.

    >>> np.isrealobj('A string')
    True
    >>> np.isrealobj(False)
    True
    >>> np.isrealobj(None)
    True

    Examples
    --------
    >>> import numpy as np
    >>> np.isrealobj(1)
    True
    >>> np.isrealobj(1+0j)
    False
    >>> np.isrealobj([3, 1+0j, True])
    False

    """
    return not iscomplexobj(x)

#-----------------------------------------------------------------------------

def _getmaxmin(t):
    from numpy._core import getlimits
    f = getlimits.finfo(t)
    return f.max, f.min


def _nan_to_num_dispatcher(x, copy=None, nan=None, posinf=None, neginf=None):
    return (x,)


@array_function_dispatch(_nan_to_num_dispatcher)
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    """
    Replace NaN with zero and infinity with large finite numbers (default
    behaviour) or with the numbers defined by the user using the `nan`,
    `posinf` and/or `neginf` keywords.

    If `x` is inexact, NaN is replaced by zero or by the user defined value in
    `nan` keyword, infinity is replaced by the largest finite floating point
    values representable by ``x.dtype`` or by the user defined value in
    `posinf` keyword and -infinity is replaced by the most negative finite
    floating point values representable by ``x.dtype`` or by the user defined
    value in `neginf` keyword.

    For complex dtypes, the above is applied to each of the real and
    imaginary components of `x` separately.

    If `x` is not inexact, then no replacements are made.

    Parameters
    ----------
    x : scalar or array_like
        Input data.
    copy : bool, optional
        Whether to create a copy of `x` (True) or to replace values
        in-place (False). The in-place operation only occurs if
        casting to an array does not require a copy.
        Default is True.
    nan : int, float, optional
        Value to be used to fill NaN values. If no value is passed
        then NaN values will be replaced with 0.0.
    posinf : int, float, optional
        Value to be used to fill positive infinity values. If no value is
        passed then positive infinity values will be replaced with a very
        large number.
    neginf : int, float, optional
        Value to be used to fill negative infinity values. If no value is
        passed then negative infinity values will be replaced with a very
        small (or negative) number.

    Returns
    -------
    out : ndarray
        `x`, with the non-finite values replaced. If `copy` is False, this may
        be `x` itself.

    See Also
    --------
    isinf : Shows which elements are positive or negative infinity.
    isneginf : Shows which elements are negative infinity.
    isposinf : Shows which elements are positive infinity.
    isnan : Shows which elements are Not a Number (NaN).
    isfinite : Shows which elements are finite (not NaN, not infinity)

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.

    Examples
    --------
    >>> import numpy as np
    >>> np.nan_to_num(np.inf)
    1.7976931348623157e+308
    >>> np.nan_to_num(-np.inf)
    -1.7976931348623157e+308
    >>> np.nan_to_num(np.nan)
    0.0
    >>> x = np.array([np.inf, -np.inf, np.nan, -128, 128])
    >>> np.nan_to_num(x)
    array([ 1.79769313e+308, -1.79769313e+308,  0.00000000e+000, # may vary
           -1.28000000e+002,  1.28000000e+002])
    >>> np.nan_to_num(x, nan=-9999, posinf=33333333, neginf=33333333)
    array([ 3.3333333e+07,  3.3333333e+07, -9.9990000e+03,
           -1.2800000e+02,  1.2800000e+02])
    >>> y = np.array([complex(np.inf, np.nan), np.nan, complex(np.nan, np.inf)])
    array([  1.79769313e+308,  -1.79769313e+308,   0.00000000e+000, # may vary
         -1.28000000e+002,   1.28000000e+002])
    >>> np.nan_to_num(y)
    array([  1.79769313e+308 +0.00000000e+000j, # may vary
             0.00000000e+000 +0.00000000e+000j,
             0.00000000e+000 +1.79769313e+308j])
    >>> np.nan_to_num(y, nan=111111, posinf=222222)
    array([222222.+111111.j, 111111.     +0.j, 111111.+222222.j])
    """
    x = _nx.array(x, subok=True, copy=copy)
    xtype = x.dtype.type

    isscalar = (x.ndim == 0)

    if not issubclass(xtype, _nx.inexact):
        return x[()] if isscalar else x

    iscomplex = issubclass(xtype, _nx.complexfloating)

    dest = (x.real, x.imag) if iscomplex else (x,)
    maxf, minf = _getmaxmin(x.real.dtype)
    if posinf is not None:
        maxf = posinf
    if neginf is not None:
        minf = neginf
    for d in dest:
        idx_nan = isnan(d)
        idx_posinf = isposinf(d)
        idx_neginf = isneginf(d)
        _nx.copyto(d, nan, where=idx_nan)
        _nx.copyto(d, maxf, where=idx_posinf)
        _nx.copyto(d, minf, where=idx_neginf)
    return x[()] if isscalar else x

#-----------------------------------------------------------------------------

def _real_if_close_dispatcher(a, tol=None):
    return (a,)


@array_function_dispatch(_real_if_close_dispatcher)
def real_if_close(a, tol=100):
    """
    If input is complex with all imaginary parts close to zero, return
    real parts.

    "Close to zero" is defined as `tol` * (machine epsilon of the type for
    `a`).

    Parameters
    ----------
    a : array_like
        Input array.
    tol : float
        Tolerance in machine epsilons for the complex part of the elements
        in the array. If the tolerance is <=1, then the absolute tolerance
        is used.

    Returns
    -------
    out : ndarray
        If `a` is real, the type of `a` is used for the output.  If `a`
        has complex elements, the returned type is float.

    See Also
    --------
    real, imag, angle

    Notes
    -----
    Machine epsilon varies from machine to machine and between data types
    but Python floats on most platforms have a machine epsilon equal to
    2.2204460492503131e-16.  You can use 'np.finfo(float).eps' to print
    out the machine epsilon for floats.

    Examples
    --------
    >>> import numpy as np
    >>> np.finfo(float).eps
    2.2204460492503131e-16 # may vary

    >>> np.real_if_close([2.1 + 4e-14j, 5.2 + 3e-15j], tol=1000)
    array([2.1, 5.2])
    >>> np.real_if_close([2.1 + 4e-13j, 5.2 + 3e-15j], tol=1000)
    array([2.1+4.e-13j, 5.2 + 3e-15j])

    """
    a = asanyarray(a)
    type_ = a.dtype.type
    if not issubclass(type_, _nx.complexfloating):
        return a
    if tol > 1:
        f = getlimits.finfo(type_)
        tol = f.eps * tol
    if _nx.all(_nx.absolute(a.imag) < tol):
        a = a.real
    return a


#-----------------------------------------------------------------------------

_namefromtype = {'S1': 'character',
                 '?': 'bool',
                 'b': 'signed char',
                 'B': 'unsigned char',
                 'h': 'short',
                 'H': 'unsigned short',
                 'i': 'integer',
                 'I': 'unsigned integer',
                 'l': 'long integer',
                 'L': 'unsigned long integer',
                 'q': 'long long integer',
                 'Q': 'unsigned long long integer',
                 'f': 'single precision',
                 'd': 'double precision',
                 'g': 'long precision',
                 'F': 'complex single precision',
                 'D': 'complex double precision',
                 'G': 'complex long double precision',
                 'S': 'string',
                 'U': 'unicode',
                 'V': 'void',
                 'O': 'object'
                 }

@set_module('numpy')
def typename(char):
    """
    Return a description for the given data type code.

    Parameters
    ----------
    char : str
        Data type code.

    Returns
    -------
    out : str
        Description of the input data type code.

    See Also
    --------
    dtype

    Examples
    --------
    >>> import numpy as np
    >>> typechars = ['S1', '?', 'B', 'D', 'G', 'F', 'I', 'H', 'L', 'O', 'Q',
    ...              'S', 'U', 'V', 'b', 'd', 'g', 'f', 'i', 'h', 'l', 'q']
    >>> for typechar in typechars:
    ...     print(typechar, ' : ', np.typename(typechar))
    ...
    S1  :  character
    ?  :  bool
    B  :  unsigned char
    D  :  complex double precision
    G  :  complex long double precision
    F  :  complex single precision
    I  :  unsigned integer
    H  :  unsigned short
    L  :  unsigned long integer
    O  :  object
    Q  :  unsigned long long integer
    S  :  string
    U  :  unicode
    V  :  void
    b  :  signed char
    d  :  double precision
    g  :  long precision
    f  :  single precision
    i  :  integer
    h  :  short
    l  :  long integer
    q  :  long long integer

    """
    return _namefromtype[char]

#-----------------------------------------------------------------------------


#determine the "minimum common type" for a group of arrays.
array_type = [[_nx.float16, _nx.float32, _nx.float64, _nx.longdouble],
              [None, _nx.complex64, _nx.complex128, _nx.clongdouble]]
array_precision = {_nx.float16: 0,
                   _nx.float32: 1,
                   _nx.float64: 2,
                   _nx.longdouble: 3,
                   _nx.complex64: 1,
                   _nx.complex128: 2,
                   _nx.clongdouble: 3}


def _common_type_dispatcher(*arrays):
    return arrays


@array_function_dispatch(_common_type_dispatcher)
def common_type(*arrays):
    """
    Return a scalar type which is common to the input arrays.

    The return type will always be an inexact (i.e. floating point) scalar
    type, even if all the arrays are integer arrays. If one of the inputs is
    an integer array, the minimum precision type that is returned is a
    64-bit floating point dtype.

    All input arrays except int64 and uint64 can be safely cast to the
    returned dtype without loss of information.

    Parameters
    ----------
    array1, array2, ... : ndarrays
        Input arrays.

    Returns
    -------
    out : data type code
        Data type code.

    See Also
    --------
    dtype, mintypecode

    Examples
    --------
    >>> np.common_type(np.arange(2, dtype=np.float32))
    <class 'numpy.float32'>
    >>> np.common_type(np.arange(2, dtype=np.float32), np.arange(2))
    <class 'numpy.float64'>
    >>> np.common_type(np.arange(4), np.array([45, 6.j]), np.array([45.0]))
    <class 'numpy.complex128'>

    """
    is_complex = False
    precision = 0
    for a in arrays:
        t = a.dtype.type
        if iscomplexobj(a):
            is_complex = True
        if issubclass(t, _nx.integer):
            p = 2  # array_precision[_nx.double]
        else:
            p = array_precision.get(t)
            if p is None:
                raise TypeError("can't get common type for non-numeric array")
        precision = max(precision, p)
    if is_complex:
        return array_type[1][precision]
    else:
        return array_type[0][precision]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\tests\test_stride_tricks.py ===
import pytest

import numpy as np
from numpy._core._rational_tests import rational
from numpy.lib._stride_tricks_impl import (
    _broadcast_shape,
    as_strided,
    broadcast_arrays,
    broadcast_shapes,
    broadcast_to,
    sliding_window_view,
)
from numpy.testing import (
    assert_,
    assert_array_equal,
    assert_equal,
    assert_raises,
    assert_raises_regex,
    assert_warns,
)


def assert_shapes_correct(input_shapes, expected_shape):
    # Broadcast a list of arrays with the given input shapes and check the
    # common output shape.

    inarrays = [np.zeros(s) for s in input_shapes]
    outarrays = broadcast_arrays(*inarrays)
    outshapes = [a.shape for a in outarrays]
    expected = [expected_shape] * len(inarrays)
    assert_equal(outshapes, expected)


def assert_incompatible_shapes_raise(input_shapes):
    # Broadcast a list of arrays with the given (incompatible) input shapes
    # and check that they raise a ValueError.

    inarrays = [np.zeros(s) for s in input_shapes]
    assert_raises(ValueError, broadcast_arrays, *inarrays)


def assert_same_as_ufunc(shape0, shape1, transposed=False, flipped=False):
    # Broadcast two shapes against each other and check that the data layout
    # is the same as if a ufunc did the broadcasting.

    x0 = np.zeros(shape0, dtype=int)
    # Note that multiply.reduce's identity element is 1.0, so when shape1==(),
    # this gives the desired n==1.
    n = int(np.multiply.reduce(shape1))
    x1 = np.arange(n).reshape(shape1)
    if transposed:
        x0 = x0.T
        x1 = x1.T
    if flipped:
        x0 = x0[::-1]
        x1 = x1[::-1]
    # Use the add ufunc to do the broadcasting. Since we're adding 0s to x1, the
    # result should be exactly the same as the broadcasted view of x1.
    y = x0 + x1
    b0, b1 = broadcast_arrays(x0, x1)
    assert_array_equal(y, b1)


def test_same():
    x = np.arange(10)
    y = np.arange(10)
    bx, by = broadcast_arrays(x, y)
    assert_array_equal(x, bx)
    assert_array_equal(y, by)

def test_broadcast_kwargs():
    # ensure that a TypeError is appropriately raised when
    # np.broadcast_arrays() is called with any keyword
    # argument other than 'subok'
    x = np.arange(10)
    y = np.arange(10)

    with assert_raises_regex(TypeError, 'got an unexpected keyword'):
        broadcast_arrays(x, y, dtype='float64')


def test_one_off():
    x = np.array([[1, 2, 3]])
    y = np.array([[1], [2], [3]])
    bx, by = broadcast_arrays(x, y)
    bx0 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    by0 = bx0.T
    assert_array_equal(bx0, bx)
    assert_array_equal(by0, by)


def test_same_input_shapes():
    # Check that the final shape is just the input shape.

    data = [
        (),
        (1,),
        (3,),
        (0, 1),
        (0, 3),
        (1, 0),
        (3, 0),
        (1, 3),
        (3, 1),
        (3, 3),
    ]
    for shape in data:
        input_shapes = [shape]
        # Single input.
        assert_shapes_correct(input_shapes, shape)
        # Double input.
        input_shapes2 = [shape, shape]
        assert_shapes_correct(input_shapes2, shape)
        # Triple input.
        input_shapes3 = [shape, shape, shape]
        assert_shapes_correct(input_shapes3, shape)


def test_two_compatible_by_ones_input_shapes():
    # Check that two different input shapes of the same length, but some have
    # ones, broadcast to the correct shape.

    data = [
        [[(1,), (3,)], (3,)],
        [[(1, 3), (3, 3)], (3, 3)],
        [[(3, 1), (3, 3)], (3, 3)],
        [[(1, 3), (3, 1)], (3, 3)],
        [[(1, 1), (3, 3)], (3, 3)],
        [[(1, 1), (1, 3)], (1, 3)],
        [[(1, 1), (3, 1)], (3, 1)],
        [[(1, 0), (0, 0)], (0, 0)],
        [[(0, 1), (0, 0)], (0, 0)],
        [[(1, 0), (0, 1)], (0, 0)],
        [[(1, 1), (0, 0)], (0, 0)],
        [[(1, 1), (1, 0)], (1, 0)],
        [[(1, 1), (0, 1)], (0, 1)],
    ]
    for input_shapes, expected_shape in data:
        assert_shapes_correct(input_shapes, expected_shape)
        # Reverse the input shapes since broadcasting should be symmetric.
        assert_shapes_correct(input_shapes[::-1], expected_shape)


def test_two_compatible_by_prepending_ones_input_shapes():
    # Check that two different input shapes (of different lengths) broadcast
    # to the correct shape.

    data = [
        [[(), (3,)], (3,)],
        [[(3,), (3, 3)], (3, 3)],
        [[(3,), (3, 1)], (3, 3)],
        [[(1,), (3, 3)], (3, 3)],
        [[(), (3, 3)], (3, 3)],
        [[(1, 1), (3,)], (1, 3)],
        [[(1,), (3, 1)], (3, 1)],
        [[(1,), (1, 3)], (1, 3)],
        [[(), (1, 3)], (1, 3)],
        [[(), (3, 1)], (3, 1)],
        [[(), (0,)], (0,)],
        [[(0,), (0, 0)], (0, 0)],
        [[(0,), (0, 1)], (0, 0)],
        [[(1,), (0, 0)], (0, 0)],
        [[(), (0, 0)], (0, 0)],
        [[(1, 1), (0,)], (1, 0)],
        [[(1,), (0, 1)], (0, 1)],
        [[(1,), (1, 0)], (1, 0)],
        [[(), (1, 0)], (1, 0)],
        [[(), (0, 1)], (0, 1)],
    ]
    for input_shapes, expected_shape in data:
        assert_shapes_correct(input_shapes, expected_shape)
        # Reverse the input shapes since broadcasting should be symmetric.
        assert_shapes_correct(input_shapes[::-1], expected_shape)


def test_incompatible_shapes_raise_valueerror():
    # Check that a ValueError is raised for incompatible shapes.

    data = [
        [(3,), (4,)],
        [(2, 3), (2,)],
        [(3,), (3,), (4,)],
        [(1, 3, 4), (2, 3, 3)],
    ]
    for input_shapes in data:
        assert_incompatible_shapes_raise(input_shapes)
        # Reverse the input shapes since broadcasting should be symmetric.
        assert_incompatible_shapes_raise(input_shapes[::-1])


def test_same_as_ufunc():
    # Check that the data layout is the same as if a ufunc did the operation.

    data = [
        [[(1,), (3,)], (3,)],
        [[(1, 3), (3, 3)], (3, 3)],
        [[(3, 1), (3, 3)], (3, 3)],
        [[(1, 3), (3, 1)], (3, 3)],
        [[(1, 1), (3, 3)], (3, 3)],
        [[(1, 1), (1, 3)], (1, 3)],
        [[(1, 1), (3, 1)], (3, 1)],
        [[(1, 0), (0, 0)], (0, 0)],
        [[(0, 1), (0, 0)], (0, 0)],
        [[(1, 0), (0, 1)], (0, 0)],
        [[(1, 1), (0, 0)], (0, 0)],
        [[(1, 1), (1, 0)], (1, 0)],
        [[(1, 1), (0, 1)], (0, 1)],
        [[(), (3,)], (3,)],
        [[(3,), (3, 3)], (3, 3)],
        [[(3,), (3, 1)], (3, 3)],
        [[(1,), (3, 3)], (3, 3)],
        [[(), (3, 3)], (3, 3)],
        [[(1, 1), (3,)], (1, 3)],
        [[(1,), (3, 1)], (3, 1)],
        [[(1,), (1, 3)], (1, 3)],
        [[(), (1, 3)], (1, 3)],
        [[(), (3, 1)], (3, 1)],
        [[(), (0,)], (0,)],
        [[(0,), (0, 0)], (0, 0)],
        [[(0,), (0, 1)], (0, 0)],
        [[(1,), (0, 0)], (0, 0)],
        [[(), (0, 0)], (0, 0)],
        [[(1, 1), (0,)], (1, 0)],
        [[(1,), (0, 1)], (0, 1)],
        [[(1,), (1, 0)], (1, 0)],
        [[(), (1, 0)], (1, 0)],
        [[(), (0, 1)], (0, 1)],
    ]
    for input_shapes, expected_shape in data:
        assert_same_as_ufunc(input_shapes[0], input_shapes[1],
                             f"Shapes: {input_shapes[0]} {input_shapes[1]}")
        # Reverse the input shapes since broadcasting should be symmetric.
        assert_same_as_ufunc(input_shapes[1], input_shapes[0])
        # Try them transposed, too.
        assert_same_as_ufunc(input_shapes[0], input_shapes[1], True)
        # ... and flipped for non-rank-0 inputs in order to test negative
        # strides.
        if () not in input_shapes:
            assert_same_as_ufunc(input_shapes[0], input_shapes[1], False, True)
            assert_same_as_ufunc(input_shapes[0], input_shapes[1], True, True)


def test_broadcast_to_succeeds():
    data = [
        [np.array(0), (0,), np.array(0)],
        [np.array(0), (1,), np.zeros(1)],
        [np.array(0), (3,), np.zeros(3)],
        [np.ones(1), (1,), np.ones(1)],
        [np.ones(1), (2,), np.ones(2)],
        [np.ones(1), (1, 2, 3), np.ones((1, 2, 3))],
        [np.arange(3), (3,), np.arange(3)],
        [np.arange(3), (1, 3), np.arange(3).reshape(1, -1)],
        [np.arange(3), (2, 3), np.array([[0, 1, 2], [0, 1, 2]])],
        # test if shape is not a tuple
        [np.ones(0), 0, np.ones(0)],
        [np.ones(1), 1, np.ones(1)],
        [np.ones(1), 2, np.ones(2)],
        # these cases with size 0 are strange, but they reproduce the behavior
        # of broadcasting with ufuncs (see test_same_as_ufunc above)
        [np.ones(1), (0,), np.ones(0)],
        [np.ones((1, 2)), (0, 2), np.ones((0, 2))],
        [np.ones((2, 1)), (2, 0), np.ones((2, 0))],
    ]
    for input_array, shape, expected in data:
        actual = broadcast_to(input_array, shape)
        assert_array_equal(expected, actual)


def test_broadcast_to_raises():
    data = [
        [(0,), ()],
        [(1,), ()],
        [(3,), ()],
        [(3,), (1,)],
        [(3,), (2,)],
        [(3,), (4,)],
        [(1, 2), (2, 1)],
        [(1, 1), (1,)],
        [(1,), -1],
        [(1,), (-1,)],
        [(1, 2), (-1, 2)],
    ]
    for orig_shape, target_shape in data:
        arr = np.zeros(orig_shape)
        assert_raises(ValueError, lambda: broadcast_to(arr, target_shape))


def test_broadcast_shape():
    # tests internal _broadcast_shape
    # _broadcast_shape is already exercised indirectly by broadcast_arrays
    # _broadcast_shape is also exercised by the public broadcast_shapes function
    assert_equal(_broadcast_shape(), ())
    assert_equal(_broadcast_shape([1, 2]), (2,))
    assert_equal(_broadcast_shape(np.ones((1, 1))), (1, 1))
    assert_equal(_broadcast_shape(np.ones((1, 1)), np.ones((3, 4))), (3, 4))
    assert_equal(_broadcast_shape(*([np.ones((1, 2))] * 32)), (1, 2))
    assert_equal(_broadcast_shape(*([np.ones((1, 2))] * 100)), (1, 2))

    # regression tests for gh-5862
    assert_equal(_broadcast_shape(*([np.ones(2)] * 32 + [1])), (2,))
    bad_args = [np.ones(2)] * 32 + [np.ones(3)] * 32
    assert_raises(ValueError, lambda: _broadcast_shape(*bad_args))


def test_broadcast_shapes_succeeds():
    # tests public broadcast_shapes
    data = [
        [[], ()],
        [[()], ()],
        [[(7,)], (7,)],
        [[(1, 2), (2,)], (1, 2)],
        [[(1, 1)], (1, 1)],
        [[(1, 1), (3, 4)], (3, 4)],
        [[(6, 7), (5, 6, 1), (7,), (5, 1, 7)], (5, 6, 7)],
        [[(5, 6, 1)], (5, 6, 1)],
        [[(1, 3), (3, 1)], (3, 3)],
        [[(1, 0), (0, 0)], (0, 0)],
        [[(0, 1), (0, 0)], (0, 0)],
        [[(1, 0), (0, 1)], (0, 0)],
        [[(1, 1), (0, 0)], (0, 0)],
        [[(1, 1), (1, 0)], (1, 0)],
        [[(1, 1), (0, 1)], (0, 1)],
        [[(), (0,)], (0,)],
        [[(0,), (0, 0)], (0, 0)],
        [[(0,), (0, 1)], (0, 0)],
        [[(1,), (0, 0)], (0, 0)],
        [[(), (0, 0)], (0, 0)],
        [[(1, 1), (0,)], (1, 0)],
        [[(1,), (0, 1)], (0, 1)],
        [[(1,), (1, 0)], (1, 0)],
        [[(), (1, 0)], (1, 0)],
        [[(), (0, 1)], (0, 1)],
        [[(1,), (3,)], (3,)],
        [[2, (3, 2)], (3, 2)],
    ]
    for input_shapes, target_shape in data:
        assert_equal(broadcast_shapes(*input_shapes), target_shape)

    assert_equal(broadcast_shapes(*([(1, 2)] * 32)), (1, 2))
    assert_equal(broadcast_shapes(*([(1, 2)] * 100)), (1, 2))

    # regression tests for gh-5862
    assert_equal(broadcast_shapes(*([(2,)] * 32)), (2,))


def test_broadcast_shapes_raises():
    # tests public broadcast_shapes
    data = [
        [(3,), (4,)],
        [(2, 3), (2,)],
        [(3,), (3,), (4,)],
        [(1, 3, 4), (2, 3, 3)],
        [(1, 2), (3, 1), (3, 2), (10, 5)],
        [2, (2, 3)],
    ]
    for input_shapes in data:
        assert_raises(ValueError, lambda: broadcast_shapes(*input_shapes))

    bad_args = [(2,)] * 32 + [(3,)] * 32
    assert_raises(ValueError, lambda: broadcast_shapes(*bad_args))


def test_as_strided():
    a = np.array([None])
    a_view = as_strided(a)
    expected = np.array([None])
    assert_array_equal(a_view, np.array([None]))

    a = np.array([1, 2, 3, 4])
    a_view = as_strided(a, shape=(2,), strides=(2 * a.itemsize,))
    expected = np.array([1, 3])
    assert_array_equal(a_view, expected)

    a = np.array([1, 2, 3, 4])
    a_view = as_strided(a, shape=(3, 4), strides=(0, 1 * a.itemsize))
    expected = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    assert_array_equal(a_view, expected)

    # Regression test for gh-5081
    dt = np.dtype([('num', 'i4'), ('obj', 'O')])
    a = np.empty((4,), dtype=dt)
    a['num'] = np.arange(1, 5)
    a_view = as_strided(a, shape=(3, 4), strides=(0, a.itemsize))
    expected_num = [[1, 2, 3, 4]] * 3
    expected_obj = [[None] * 4] * 3
    assert_equal(a_view.dtype, dt)
    assert_array_equal(expected_num, a_view['num'])
    assert_array_equal(expected_obj, a_view['obj'])

    # Make sure that void types without fields are kept unchanged
    a = np.empty((4,), dtype='V4')
    a_view = as_strided(a, shape=(3, 4), strides=(0, a.itemsize))
    assert_equal(a.dtype, a_view.dtype)

    # Make sure that the only type that could fail is properly handled
    dt = np.dtype({'names': [''], 'formats': ['V4']})
    a = np.empty((4,), dtype=dt)
    a_view = as_strided(a, shape=(3, 4), strides=(0, a.itemsize))
    assert_equal(a.dtype, a_view.dtype)

    # Custom dtypes should not be lost (gh-9161)
    r = [rational(i) for i in range(4)]
    a = np.array(r, dtype=rational)
    a_view = as_strided(a, shape=(3, 4), strides=(0, a.itemsize))
    assert_equal(a.dtype, a_view.dtype)
    assert_array_equal([r] * 3, a_view)


class TestSlidingWindowView:
    def test_1d(self):
        arr = np.arange(5)
        arr_view = sliding_window_view(arr, 2)
        expected = np.array([[0, 1],
                             [1, 2],
                             [2, 3],
                             [3, 4]])
        assert_array_equal(arr_view, expected)

    def test_2d(self):
        i, j = np.ogrid[:3, :4]
        arr = 10 * i + j
        shape = (2, 2)
        arr_view = sliding_window_view(arr, shape)
        expected = np.array([[[[0, 1], [10, 11]],
                              [[1, 2], [11, 12]],
                              [[2, 3], [12, 13]]],
                             [[[10, 11], [20, 21]],
                              [[11, 12], [21, 22]],
                              [[12, 13], [22, 23]]]])
        assert_array_equal(arr_view, expected)

    def test_2d_with_axis(self):
        i, j = np.ogrid[:3, :4]
        arr = 10 * i + j
        arr_view = sliding_window_view(arr, 3, 0)
        expected = np.array([[[0, 10, 20],
                              [1, 11, 21],
                              [2, 12, 22],
                              [3, 13, 23]]])
        assert_array_equal(arr_view, expected)

    def test_2d_repeated_axis(self):
        i, j = np.ogrid[:3, :4]
        arr = 10 * i + j
        arr_view = sliding_window_view(arr, (2, 3), (1, 1))
        expected = np.array([[[[0, 1, 2],
                               [1, 2, 3]]],
                             [[[10, 11, 12],
                               [11, 12, 13]]],
                             [[[20, 21, 22],
                               [21, 22, 23]]]])
        assert_array_equal(arr_view, expected)

    def test_2d_without_axis(self):
        i, j = np.ogrid[:4, :4]
        arr = 10 * i + j
        shape = (2, 3)
        arr_view = sliding_window_view(arr, shape)
        expected = np.array([[[[0, 1, 2], [10, 11, 12]],
                              [[1, 2, 3], [11, 12, 13]]],
                             [[[10, 11, 12], [20, 21, 22]],
                              [[11, 12, 13], [21, 22, 23]]],
                             [[[20, 21, 22], [30, 31, 32]],
                              [[21, 22, 23], [31, 32, 33]]]])
        assert_array_equal(arr_view, expected)

    def test_errors(self):
        i, j = np.ogrid[:4, :4]
        arr = 10 * i + j
        with pytest.raises(ValueError, match='cannot contain negative values'):
            sliding_window_view(arr, (-1, 3))
        with pytest.raises(
                ValueError,
                match='must provide window_shape for all dimensions of `x`'):
            sliding_window_view(arr, (1,))
        with pytest.raises(
                ValueError,
                match='Must provide matching length window_shape and axis'):
            sliding_window_view(arr, (1, 3, 4), axis=(0, 1))
        with pytest.raises(
                ValueError,
                match='window shape cannot be larger than input array'):
            sliding_window_view(arr, (5, 5))

    def test_writeable(self):
        arr = np.arange(5)
        view = sliding_window_view(arr, 2, writeable=False)
        assert_(not view.flags.writeable)
        with pytest.raises(
                ValueError,
                match='assignment destination is read-only'):
            view[0, 0] = 3
        view = sliding_window_view(arr, 2, writeable=True)
        assert_(view.flags.writeable)
        view[0, 1] = 3
        assert_array_equal(arr, np.array([0, 3, 2, 3, 4]))

    def test_subok(self):
        class MyArray(np.ndarray):
            pass

        arr = np.arange(5).view(MyArray)
        assert_(not isinstance(sliding_window_view(arr, 2,
                                                   subok=False),
                               MyArray))
        assert_(isinstance(sliding_window_view(arr, 2, subok=True), MyArray))
        # Default behavior
        assert_(not isinstance(sliding_window_view(arr, 2), MyArray))


def as_strided_writeable():
    arr = np.ones(10)
    view = as_strided(arr, writeable=False)
    assert_(not view.flags.writeable)

    # Check that writeable also is fine:
    view = as_strided(arr, writeable=True)
    assert_(view.flags.writeable)
    view[...] = 3
    assert_array_equal(arr, np.full_like(arr, 3))

    # Test that things do not break down for readonly:
    arr.flags.writeable = False
    view = as_strided(arr, writeable=False)
    view = as_strided(arr, writeable=True)
    assert_(not view.flags.writeable)


class VerySimpleSubClass(np.ndarray):
    def __new__(cls, *args, **kwargs):
        return np.array(*args, subok=True, **kwargs).view(cls)


class SimpleSubClass(VerySimpleSubClass):
    def __new__(cls, *args, **kwargs):
        self = np.array(*args, subok=True, **kwargs).view(cls)
        self.info = 'simple'
        return self

    def __array_finalize__(self, obj):
        self.info = getattr(obj, 'info', '') + ' finalized'


def test_subclasses():
    # test that subclass is preserved only if subok=True
    a = VerySimpleSubClass([1, 2, 3, 4])
    assert_(type(a) is VerySimpleSubClass)
    a_view = as_strided(a, shape=(2,), strides=(2 * a.itemsize,))
    assert_(type(a_view) is np.ndarray)
    a_view = as_strided(a, shape=(2,), strides=(2 * a.itemsize,), subok=True)
    assert_(type(a_view) is VerySimpleSubClass)
    # test that if a subclass has __array_finalize__, it is used
    a = SimpleSubClass([1, 2, 3, 4])
    a_view = as_strided(a, shape=(2,), strides=(2 * a.itemsize,), subok=True)
    assert_(type(a_view) is SimpleSubClass)
    assert_(a_view.info == 'simple finalized')

    # similar tests for broadcast_arrays
    b = np.arange(len(a)).reshape(-1, 1)
    a_view, b_view = broadcast_arrays(a, b)
    assert_(type(a_view) is np.ndarray)
    assert_(type(b_view) is np.ndarray)
    assert_(a_view.shape == b_view.shape)
    a_view, b_view = broadcast_arrays(a, b, subok=True)
    assert_(type(a_view) is SimpleSubClass)
    assert_(a_view.info == 'simple finalized')
    assert_(type(b_view) is np.ndarray)
    assert_(a_view.shape == b_view.shape)

    # and for broadcast_to
    shape = (2, 4)
    a_view = broadcast_to(a, shape)
    assert_(type(a_view) is np.ndarray)
    assert_(a_view.shape == shape)
    a_view = broadcast_to(a, shape, subok=True)
    assert_(type(a_view) is SimpleSubClass)
    assert_(a_view.info == 'simple finalized')
    assert_(a_view.shape == shape)


def test_writeable():
    # broadcast_to should return a readonly array
    original = np.array([1, 2, 3])
    result = broadcast_to(original, (2, 3))
    assert_equal(result.flags.writeable, False)
    assert_raises(ValueError, result.__setitem__, slice(None), 0)

    # but the result of broadcast_arrays needs to be writeable, to
    # preserve backwards compatibility
    test_cases = [((False,), broadcast_arrays(original,)),
                  ((True, False), broadcast_arrays(0, original))]
    for is_broadcast, results in test_cases:
        for array_is_broadcast, result in zip(is_broadcast, results):
            # This will change to False in a future version
            if array_is_broadcast:
                with assert_warns(FutureWarning):
                    assert_equal(result.flags.writeable, True)
                with assert_warns(DeprecationWarning):
                    result[:] = 0
                # Warning not emitted, writing to the array resets it
                assert_equal(result.flags.writeable, True)
            else:
                # No warning:
                assert_equal(result.flags.writeable, True)

    for results in [broadcast_arrays(original),
                    broadcast_arrays(0, original)]:
        for result in results:
            # resets the warn_on_write DeprecationWarning
            result.flags.writeable = True
            # check: no warning emitted
            assert_equal(result.flags.writeable, True)
            result[:] = 0

    # keep readonly input readonly
    original.flags.writeable = False
    _, result = broadcast_arrays(0, original)
    assert_equal(result.flags.writeable, False)

    # regression test for GH6491
    shape = (2,)
    strides = [0]
    tricky_array = as_strided(np.array(0), shape, strides)
    other = np.zeros((1,))
    first, second = broadcast_arrays(tricky_array, other)
    assert_(first.shape == second.shape)


def test_writeable_memoryview():
    # The result of broadcast_arrays exports as a non-writeable memoryview
    # because otherwise there is no good way to opt in to the new behaviour
    # (i.e. you would need to set writeable to False explicitly).
    # See gh-13929.
    original = np.array([1, 2, 3])

    test_cases = [((False, ), broadcast_arrays(original,)),
                  ((True, False), broadcast_arrays(0, original))]
    for is_broadcast, results in test_cases:
        for array_is_broadcast, result in zip(is_broadcast, results):
            # This will change to False in a future version
            if array_is_broadcast:
                # memoryview(result, writable=True) will give warning but cannot
                # be tested using the python API.
                assert memoryview(result).readonly
            else:
                assert not memoryview(result).readonly


def test_reference_types():
    input_array = np.array('a', dtype=object)
    expected = np.array(['a'] * 3, dtype=object)
    actual = broadcast_to(input_array, (3,))
    assert_array_equal(expected, actual)

    actual, _ = broadcast_arrays(input_array, np.ones(3))
    assert_array_equal(expected, actual)