
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\tests\test_regression.py ===
import os

import numpy as np
from numpy.testing import (
    _assert_valid_refcount,
    assert_,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_raises,
)


class TestRegression:
    def test_poly1d(self):
        # Ticket #28
        assert_equal(np.poly1d([1]) - np.poly1d([1, 0]),
                     np.poly1d([-1, 1]))

    def test_cov_parameters(self):
        # Ticket #91
        x = np.random.random((3, 3))
        y = x.copy()
        np.cov(x, rowvar=True)
        np.cov(y, rowvar=False)
        assert_array_equal(x, y)

    def test_mem_digitize(self):
        # Ticket #95
        for i in range(100):
            np.digitize([1, 2, 3, 4], [1, 3])
            np.digitize([0, 1, 2, 3, 4], [1, 3])

    def test_unique_zero_sized(self):
        # Ticket #205
        assert_array_equal([], np.unique(np.array([])))

    def test_mem_vectorise(self):
        # Ticket #325
        vt = np.vectorize(lambda *args: args)
        vt(np.zeros((1, 2, 1)), np.zeros((2, 1, 1)), np.zeros((1, 1, 2)))
        vt(np.zeros((1, 2, 1)), np.zeros((2, 1, 1)), np.zeros((1,
           1, 2)), np.zeros((2, 2)))

    def test_mgrid_single_element(self):
        # Ticket #339
        assert_array_equal(np.mgrid[0:0:1j], [0])
        assert_array_equal(np.mgrid[0:0], [])

    def test_refcount_vectorize(self):
        # Ticket #378
        def p(x, y):
            return 123
        v = np.vectorize(p)
        _assert_valid_refcount(v)

    def test_poly1d_nan_roots(self):
        # Ticket #396
        p = np.poly1d([np.nan, np.nan, 1], r=False)
        assert_raises(np.linalg.LinAlgError, getattr, p, "r")

    def test_mem_polymul(self):
        # Ticket #448
        np.polymul([], [1.])

    def test_mem_string_concat(self):
        # Ticket #469
        x = np.array([])
        np.append(x, 'asdasd\tasdasd')

    def test_poly_div(self):
        # Ticket #553
        u = np.poly1d([1, 2, 3])
        v = np.poly1d([1, 2, 3, 4, 5])
        q, r = np.polydiv(u, v)
        assert_equal(q * v + r, u)

    def test_poly_eq(self):
        # Ticket #554
        x = np.poly1d([1, 2, 3])
        y = np.poly1d([3, 4])
        assert_(x != y)
        assert_(x == x)

    def test_polyfit_build(self):
        # Ticket #628
        ref = [-1.06123820e-06, 5.70886914e-04, -1.13822012e-01,
               9.95368241e+00, -3.14526520e+02]
        x = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
             104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
             116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 129,
             130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
             146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,
             158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169,
             170, 171, 172, 173, 174, 175, 176]
        y = [9.0, 3.0, 7.0, 4.0, 4.0, 8.0, 6.0, 11.0, 9.0, 8.0, 11.0, 5.0,
             6.0, 5.0, 9.0, 8.0, 6.0, 10.0, 6.0, 10.0, 7.0, 6.0, 6.0, 6.0,
             13.0, 4.0, 9.0, 11.0, 4.0, 5.0, 8.0, 5.0, 7.0, 7.0, 6.0, 12.0,
             7.0, 7.0, 9.0, 4.0, 12.0, 6.0, 6.0, 4.0, 3.0, 9.0, 8.0, 8.0,
             6.0, 7.0, 9.0, 10.0, 6.0, 8.0, 4.0, 7.0, 7.0, 10.0, 8.0, 8.0,
             6.0, 3.0, 8.0, 4.0, 5.0, 7.0, 8.0, 6.0, 6.0, 4.0, 12.0, 9.0,
             8.0, 8.0, 8.0, 6.0, 7.0, 4.0, 4.0, 5.0, 7.0]
        tested = np.polyfit(x, y, 4)
        assert_array_almost_equal(ref, tested)

    def test_polydiv_type(self):
        # Make polydiv work for complex types
        msg = "Wrong type, should be complex"
        x = np.ones(3, dtype=complex)
        q, r = np.polydiv(x, x)
        assert_(q.dtype == complex, msg)
        msg = "Wrong type, should be float"
        x = np.ones(3, dtype=int)
        q, r = np.polydiv(x, x)
        assert_(q.dtype == float, msg)

    def test_histogramdd_too_many_bins(self):
        # Ticket 928.
        assert_raises(ValueError, np.histogramdd, np.ones((1, 10)), bins=2**10)

    def test_polyint_type(self):
        # Ticket #944
        msg = "Wrong type, should be complex"
        x = np.ones(3, dtype=complex)
        assert_(np.polyint(x).dtype == complex, msg)
        msg = "Wrong type, should be float"
        x = np.ones(3, dtype=int)
        assert_(np.polyint(x).dtype == float, msg)

    def test_ndenumerate_crash(self):
        # Ticket 1140
        # Shouldn't crash:
        list(np.ndenumerate(np.array([[]])))

    def test_large_fancy_indexing(self):
        # Large enough to fail on 64-bit.
        nbits = np.dtype(np.intp).itemsize * 8
        thesize = int((2**nbits)**(1.0 / 5.0) + 1)

        def dp():
            n = 3
            a = np.ones((n,) * 5)
            i = np.random.randint(0, n, size=thesize)
            a[np.ix_(i, i, i, i, i)] = 0

        def dp2():
            n = 3
            a = np.ones((n,) * 5)
            i = np.random.randint(0, n, size=thesize)
            a[np.ix_(i, i, i, i, i)]

        assert_raises(ValueError, dp)
        assert_raises(ValueError, dp2)

    def test_void_coercion(self):
        dt = np.dtype([('a', 'f4'), ('b', 'i4')])
        x = np.zeros((1,), dt)
        assert_(np.r_[x, x].dtype == dt)

    def test_include_dirs(self):
        # As a sanity check, just test that get_include
        # includes something reasonable.  Somewhat
        # related to ticket #1405.
        include_dirs = [np.get_include()]
        for path in include_dirs:
            assert_(isinstance(path, str))
            assert_(path != '')

    def test_polyder_return_type(self):
        # Ticket #1249
        assert_(isinstance(np.polyder(np.poly1d([1]), 0), np.poly1d))
        assert_(isinstance(np.polyder([1], 0), np.ndarray))
        assert_(isinstance(np.polyder(np.poly1d([1]), 1), np.poly1d))
        assert_(isinstance(np.polyder([1], 1), np.ndarray))

    def test_append_fields_dtype_list(self):
        # Ticket #1676
        from numpy.lib.recfunctions import append_fields

        base = np.array([1, 2, 3], dtype=np.int32)
        names = ['a', 'b', 'c']
        data = np.eye(3).astype(np.int32)
        dlist = [np.float64, np.int32, np.int32]
        try:
            append_fields(base, names, data, dlist)
        except Exception:
            raise AssertionError

    def test_loadtxt_fields_subarrays(self):
        # For ticket #1936
        from io import StringIO

        dt = [("a", 'u1', 2), ("b", 'u1', 2)]
        x = np.loadtxt(StringIO("0 1 2 3"), dtype=dt)
        assert_equal(x, np.array([((0, 1), (2, 3))], dtype=dt))

        dt = [("a", [("a", 'u1', (1, 3)), ("b", 'u1')])]
        x = np.loadtxt(StringIO("0 1 2 3"), dtype=dt)
        assert_equal(x, np.array([(((0, 1, 2), 3),)], dtype=dt))

        dt = [("a", 'u1', (2, 2))]
        x = np.loadtxt(StringIO("0 1 2 3"), dtype=dt)
        assert_equal(x, np.array([(((0, 1), (2, 3)),)], dtype=dt))

        dt = [("a", 'u1', (2, 3, 2))]
        x = np.loadtxt(StringIO("0 1 2 3 4 5 6 7 8 9 10 11"), dtype=dt)
        data = [((((0, 1), (2, 3), (4, 5)), ((6, 7), (8, 9), (10, 11))),)]
        assert_equal(x, np.array(data, dtype=dt))

    def test_nansum_with_boolean(self):
        # gh-2978
        a = np.zeros(2, dtype=bool)
        try:
            np.nansum(a)
        except Exception:
            raise AssertionError

    def test_py3_compat(self):
        # gh-2561
        # Test if the oldstyle class test is bypassed in python3
        class C:
            """Old-style class in python2, normal class in python3"""
            pass

        out = open(os.devnull, 'w')
        try:
            np.info(C(), output=out)
        except AttributeError:
            raise AssertionError
        finally:
            out.close()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\_arrayterator_impl.py ===
"""
A buffered iterator for big arrays.

This module solves the problem of iterating over a big file-based array
without having to read it into memory. The `Arrayterator` class wraps
an array object, and when iterated it will return sub-arrays with at most
a user-specified number of elements.

"""
from functools import reduce
from operator import mul

__all__ = ['Arrayterator']


class Arrayterator:
    """
    Buffered iterator for big arrays.

    `Arrayterator` creates a buffered iterator for reading big arrays in small
    contiguous blocks. The class is useful for objects stored in the
    file system. It allows iteration over the object *without* reading
    everything in memory; instead, small blocks are read and iterated over.

    `Arrayterator` can be used with any object that supports multidimensional
    slices. This includes NumPy arrays, but also variables from
    Scientific.IO.NetCDF or pynetcdf for example.

    Parameters
    ----------
    var : array_like
        The object to iterate over.
    buf_size : int, optional
        The buffer size. If `buf_size` is supplied, the maximum amount of
        data that will be read into memory is `buf_size` elements.
        Default is None, which will read as many element as possible
        into memory.

    Attributes
    ----------
    var
    buf_size
    start
    stop
    step
    shape
    flat

    See Also
    --------
    numpy.ndenumerate : Multidimensional array iterator.
    numpy.flatiter : Flat array iterator.
    numpy.memmap : Create a memory-map to an array stored
                   in a binary file on disk.

    Notes
    -----
    The algorithm works by first finding a "running dimension", along which
    the blocks will be extracted. Given an array of dimensions
    ``(d1, d2, ..., dn)``, e.g. if `buf_size` is smaller than ``d1``, the
    first dimension will be used. If, on the other hand,
    ``d1 < buf_size < d1*d2`` the second dimension will be used, and so on.
    Blocks are extracted along this dimension, and when the last block is
    returned the process continues from the next dimension, until all
    elements have been read.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
    >>> a_itor = np.lib.Arrayterator(a, 2)
    >>> a_itor.shape
    (3, 4, 5, 6)

    Now we can iterate over ``a_itor``, and it will return arrays of size
    two. Since `buf_size` was smaller than any dimension, the first
    dimension will be iterated over first:

    >>> for subarr in a_itor:
    ...     if not subarr.all():
    ...         print(subarr, subarr.shape) # doctest: +SKIP
    >>> # [[[[0 1]]]] (1, 1, 1, 2)

    """

    __module__ = "numpy.lib"

    def __init__(self, var, buf_size=None):
        self.var = var
        self.buf_size = buf_size

        self.start = [0 for dim in var.shape]
        self.stop = list(var.shape)
        self.step = [1 for dim in var.shape]

    def __getattr__(self, attr):
        return getattr(self.var, attr)

    def __getitem__(self, index):
        """
        Return a new arrayterator.

        """
        # Fix index, handling ellipsis and incomplete slices.
        if not isinstance(index, tuple):
            index = (index,)
        fixed = []
        length, dims = len(index), self.ndim
        for slice_ in index:
            if slice_ is Ellipsis:
                fixed.extend([slice(None)] * (dims - length + 1))
                length = len(fixed)
            elif isinstance(slice_, int):
                fixed.append(slice(slice_, slice_ + 1, 1))
            else:
                fixed.append(slice_)
        index = tuple(fixed)
        if len(index) < dims:
            index += (slice(None),) * (dims - len(index))

        # Return a new arrayterator object.
        out = self.__class__(self.var, self.buf_size)
        for i, (start, stop, step, slice_) in enumerate(
                zip(self.start, self.stop, self.step, index)):
            out.start[i] = start + (slice_.start or 0)
            out.step[i] = step * (slice_.step or 1)
            out.stop[i] = start + (slice_.stop or stop - start)
            out.stop[i] = min(stop, out.stop[i])
        return out

    def __array__(self, dtype=None, copy=None):
        """
        Return corresponding data.

        """
        slice_ = tuple(slice(*t) for t in zip(
                self.start, self.stop, self.step))
        return self.var[slice_]

    @property
    def flat(self):
        """
        A 1-D flat iterator for Arrayterator objects.

        This iterator returns elements of the array to be iterated over in
        `~lib.Arrayterator` one by one.
        It is similar to `flatiter`.

        See Also
        --------
        lib.Arrayterator
        flatiter

        Examples
        --------
        >>> a = np.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
        >>> a_itor = np.lib.Arrayterator(a, 2)

        >>> for subarr in a_itor.flat:
        ...     if not subarr:
        ...         print(subarr, type(subarr))
        ...
        0 <class 'numpy.int64'>

        """
        for block in self:
            yield from block.flat

    @property
    def shape(self):
        """
        The shape of the array to be iterated over.

        For an example, see `Arrayterator`.

        """
        return tuple(((stop - start - 1) // step + 1) for start, stop, step in
                zip(self.start, self.stop, self.step))

    def __iter__(self):
        # Skip arrays with degenerate dimensions
        if [dim for dim in self.shape if dim <= 0]:
            return

        start = self.start[:]
        stop = self.stop[:]
        step = self.step[:]
        ndims = self.var.ndim

        while True:
            count = self.buf_size or reduce(mul, self.shape)

            # iterate over each dimension, looking for the
            # running dimension (ie, the dimension along which
            # the blocks will be built from)
            rundim = 0
            for i in range(ndims - 1, -1, -1):
                # if count is zero we ran out of elements to read
                # along higher dimensions, so we read only a single position
                if count == 0:
                    stop[i] = start[i] + 1
                elif count <= self.shape[i]:
                    # limit along this dimension
                    stop[i] = start[i] + count * step[i]
                    rundim = i
                else:
                    # read everything along this dimension
                    stop[i] = self.stop[i]
                stop[i] = min(self.stop[i], stop[i])
                count = count // self.shape[i]

            # yield a block
            slice_ = tuple(slice(*t) for t in zip(start, stop, step))
            yield self.var[slice_]

            # Update start position, taking care of overflow to
            # other dimensions
            start[rundim] = stop[rundim]  # start where we stopped
            for i in range(ndims - 1, 0, -1):
                if start[i] >= self.stop[i]:
                    start[i] = self.start[i]
                    start[i - 1] += self.step[i - 1]
            if start[0] >= self.stop[0]:
                return

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\sql\selectable.py ===
# sql/selectable.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""The :class:`_expression.FromClause` class of SQL expression elements,
representing
SQL tables and derived rowsets.

"""

from __future__ import annotations

import collections
from enum import Enum
import itertools
from typing import AbstractSet
from typing import Any as TODO_Any
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from . import cache_key
from . import coercions
from . import operators
from . import roles
from . import traversals
from . import type_api
from . import visitors
from ._typing import _ColumnsClauseArgument
from ._typing import _no_kw
from ._typing import _T
from ._typing import _TP
from ._typing import is_column_element
from ._typing import is_select_statement
from ._typing import is_subquery
from ._typing import is_table
from ._typing import is_text_clause
from .annotation import Annotated
from .annotation import SupportsCloneAnnotations
from .base import _clone
from .base import _cloned_difference
from .base import _cloned_intersection
from .base import _entity_namespace_key
from .base import _EntityNamespace
from .base import _expand_cloned
from .base import _from_objects
from .base import _generative
from .base import _never_select_column
from .base import _NoArg
from .base import _select_iterables
from .base import CacheableOptions
from .base import ColumnCollection
from .base import ColumnSet
from .base import CompileState
from .base import DedupeColumnCollection
from .base import DialectKWArgs
from .base import Executable
from .base import Generative
from .base import HasCompileState
from .base import HasMemoized
from .base import Immutable
from .coercions import _document_text_coercion
from .elements import _anonymous_label
from .elements import BindParameter
from .elements import BooleanClauseList
from .elements import ClauseElement
from .elements import ClauseList
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import DQLDMLClauseElement
from .elements import GroupedElement
from .elements import literal_column
from .elements import TableValuedColumn
from .elements import UnaryExpression
from .operators import OperatorType
from .sqltypes import NULLTYPE
from .visitors import _TraverseInternalsType
from .visitors import InternalTraversal
from .visitors import prefix_anon_map
from .. import exc
from .. import util
from ..util import HasMemoized_ro_memoized_attribute
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self


and_ = BooleanClauseList.and_


if TYPE_CHECKING:
    from ._typing import _ColumnExpressionArgument
    from ._typing import _ColumnExpressionOrStrLabelArgument
    from ._typing import _FromClauseArgument
    from ._typing import _JoinTargetArgument
    from ._typing import _LimitOffsetType
    from ._typing import _MAYBE_ENTITY
    from ._typing import _NOT_ENTITY
    from ._typing import _OnClauseArgument
    from ._typing import _SelectStatementForCompoundArgument
    from ._typing import _T0
    from ._typing import _T1
    from ._typing import _T2
    from ._typing import _T3
    from ._typing import _T4
    from ._typing import _T5
    from ._typing import _T6
    from ._typing import _T7
    from ._typing import _TextCoercedExpressionArgument
    from ._typing import _TypedColumnClauseArgument as _TCCA
    from ._typing import _TypeEngineArgument
    from .base import _AmbiguousTableNameMap
    from .base import ExecutableOption
    from .base import ReadOnlyColumnCollection
    from .cache_key import _CacheKeyTraversalType
    from .compiler import SQLCompiler
    from .dml import Delete
    from .dml import Update
    from .elements import BinaryExpression
    from .elements import KeyedColumnElement
    from .elements import Label
    from .elements import NamedColumn
    from .elements import TextClause
    from .functions import Function
    from .schema import ForeignKey
    from .schema import ForeignKeyConstraint
    from .sqltypes import TableValueType
    from .type_api import TypeEngine
    from .visitors import _CloneCallableType


_ColumnsClauseElement = Union["FromClause", ColumnElement[Any], "TextClause"]
_LabelConventionCallable = Callable[
    [Union["ColumnElement[Any]", "TextClause"]], Optional[str]
]


class _JoinTargetProtocol(Protocol):
    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]: ...

    @util.ro_non_memoized_property
    def entity_namespace(self) -> _EntityNamespace: ...


_JoinTargetElement = Union["FromClause", _JoinTargetProtocol]
_OnClauseElement = Union["ColumnElement[bool]", _JoinTargetProtocol]

_ForUpdateOfArgument = Union[
    # single column, Table, ORM Entity
    Union[
        "_ColumnExpressionArgument[Any]",
        "_FromClauseArgument",
    ],
    # or sequence of single column elements
    Sequence["_ColumnExpressionArgument[Any]"],
]


_SetupJoinsElement = Tuple[
    _JoinTargetElement,
    Optional[_OnClauseElement],
    Optional["FromClause"],
    Dict[str, Any],
]


_SelectIterable = Iterable[Union["ColumnElement[Any]", "TextClause"]]


class _OffsetLimitParam(BindParameter[int]):
    inherit_cache = True

    @property
    def _limit_offset_value(self) -> Optional[int]:
        return self.effective_value


class ReturnsRows(roles.ReturnsRowsRole, DQLDMLClauseElement):
    """The base-most class for Core constructs that have some concept of
    columns that can represent rows.

    While the SELECT statement and TABLE are the primary things we think
    of in this category,  DML like INSERT, UPDATE and DELETE can also specify
    RETURNING which means they can be used in CTEs and other forms, and
    PostgreSQL has functions that return rows also.

    .. versionadded:: 1.4

    """

    _is_returns_rows = True

    # sub-elements of returns_rows
    _is_from_clause = False
    _is_select_base = False
    _is_select_statement = False
    _is_lateral = False

    @property
    def selectable(self) -> ReturnsRows:
        return self

    @util.ro_non_memoized_property
    def _all_selected_columns(self) -> _SelectIterable:
        """A sequence of column expression objects that represents the
        "selected" columns of this :class:`_expression.ReturnsRows`.

        This is typically equivalent to .exported_columns except it is
        delivered in the form of a straight sequence and not  keyed
        :class:`_expression.ColumnCollection`.

        """
        raise NotImplementedError()

    def is_derived_from(self, fromclause: Optional[FromClause]) -> bool:
        """Return ``True`` if this :class:`.ReturnsRows` is
        'derived' from the given :class:`.FromClause`.

        An example would be an Alias of a Table is derived from that Table.

        """
        raise NotImplementedError()

    def _generate_fromclause_column_proxies(
        self,
        fromclause: FromClause,
        columns: ColumnCollection[str, KeyedColumnElement[Any]],
        primary_key: ColumnSet,
        foreign_keys: Set[KeyedColumnElement[Any]],
    ) -> None:
        """Populate columns into an :class:`.AliasedReturnsRows` object."""

        raise NotImplementedError()

    def _refresh_for_new_column(self, column: ColumnElement[Any]) -> None:
        """reset internal collections for an incoming column being added."""
        raise NotImplementedError()

    @property
    def exported_columns(self) -> ReadOnlyColumnCollection[Any, Any]:
        """A :class:`_expression.ColumnCollection`
        that represents the "exported"
        columns of this :class:`_expression.ReturnsRows`.

        The "exported" columns represent the collection of
        :class:`_expression.ColumnElement`
        expressions that are rendered by this SQL
        construct.   There are primary varieties which are the
        "FROM clause columns" of a FROM clause, such as a table, join,
        or subquery, the "SELECTed columns", which are the columns in
        the "columns clause" of a SELECT statement, and the RETURNING
        columns in a DML statement..

        .. versionadded:: 1.4

        .. seealso::

            :attr:`_expression.FromClause.exported_columns`

            :attr:`_expression.SelectBase.exported_columns`
        """

        raise NotImplementedError()


class ExecutableReturnsRows(Executable, ReturnsRows):
    """base for executable statements that return rows."""


class TypedReturnsRows(ExecutableReturnsRows, Generic[_TP]):
    """base for a typed executable statements that return rows."""


class Selectable(ReturnsRows):
    """Mark a class as being selectable."""

    __visit_name__ = "selectable"

    is_selectable = True

    def _refresh_for_new_column(self, column: ColumnElement[Any]) -> None:
        raise NotImplementedError()

    def lateral(self, name: Optional[str] = None) -> LateralFromClause:
        """Return a LATERAL alias of this :class:`_expression.Selectable`.

        The return value is the :class:`_expression.Lateral` construct also
        provided by the top-level :func:`_expression.lateral` function.

        .. seealso::

            :ref:`tutorial_lateral_correlation` -  overview of usage.

        """
        return Lateral._construct(self, name=name)

    @util.deprecated(
        "1.4",
        message="The :meth:`.Selectable.replace_selectable` method is "
        "deprecated, and will be removed in a future release.  Similar "
        "functionality is available via the sqlalchemy.sql.visitors module.",
    )
    @util.preload_module("sqlalchemy.sql.util")
    def replace_selectable(self, old: FromClause, alias: Alias) -> Self:
        """Replace all occurrences of :class:`_expression.FromClause`
        'old' with the given :class:`_expression.Alias`
        object, returning a copy of this :class:`_expression.FromClause`.

        """
        return util.preloaded.sql_util.ClauseAdapter(alias).traverse(self)

    def corresponding_column(
        self, column: KeyedColumnElement[Any], require_embedded: bool = False
    ) -> Optional[KeyedColumnElement[Any]]:
        """Given a :class:`_expression.ColumnElement`, return the exported
        :class:`_expression.ColumnElement` object from the
        :attr:`_expression.Selectable.exported_columns`
        collection of this :class:`_expression.Selectable`
        which corresponds to that
        original :class:`_expression.ColumnElement` via a common ancestor
        column.

        :param column: the target :class:`_expression.ColumnElement`
                      to be matched.

        :param require_embedded: only return corresponding columns for
         the given :class:`_expression.ColumnElement`, if the given
         :class:`_expression.ColumnElement`
         is actually present within a sub-element
         of this :class:`_expression.Selectable`.
         Normally the column will match if
         it merely shares a common ancestor with one of the exported
         columns of this :class:`_expression.Selectable`.

        .. seealso::

            :attr:`_expression.Selectable.exported_columns` - the
            :class:`_expression.ColumnCollection`
            that is used for the operation.

            :meth:`_expression.ColumnCollection.corresponding_column`
            - implementation
            method.

        """

        return self.exported_columns.corresponding_column(
            column, require_embedded
        )


class HasPrefixes:
    _prefixes: Tuple[Tuple[DQLDMLClauseElement, str], ...] = ()

    _has_prefixes_traverse_internals: _TraverseInternalsType = [
        ("_prefixes", InternalTraversal.dp_prefix_sequence)
    ]

    @_generative
    @_document_text_coercion(
        "prefixes",
        ":meth:`_expression.HasPrefixes.prefix_with`",
        ":paramref:`.HasPrefixes.prefix_with.*prefixes`",
    )
    def prefix_with(
        self,
        *prefixes: _TextCoercedExpressionArgument[Any],
        dialect: str = "*",
    ) -> Self:
        r"""Add one or more expressions following the statement keyword, i.e.
        SELECT, INSERT, UPDATE, or DELETE. Generative.

        This is used to support backend-specific prefix keywords such as those
        provided by MySQL.

        E.g.::

            stmt = table.insert().prefix_with("LOW_PRIORITY", dialect="mysql")

            # MySQL 5.7 optimizer hints
            stmt = select(table).prefix_with("/*+ BKA(t1) */", dialect="mysql")

        Multiple prefixes can be specified by multiple calls
        to :meth:`_expression.HasPrefixes.prefix_with`.

        :param \*prefixes: textual or :class:`_expression.ClauseElement`
         construct which
         will be rendered following the INSERT, UPDATE, or DELETE
         keyword.
        :param dialect: optional string dialect name which will
         limit rendering of this prefix to only that dialect.

        """
        self._prefixes = self._prefixes + tuple(
            [
                (coercions.expect(roles.StatementOptionRole, p), dialect)
                for p in prefixes
            ]
        )
        return self


class HasSuffixes:
    _suffixes: Tuple[Tuple[DQLDMLClauseElement, str], ...] = ()

    _has_suffixes_traverse_internals: _TraverseInternalsType = [
        ("_suffixes", InternalTraversal.dp_prefix_sequence)
    ]

    @_generative
    @_document_text_coercion(
        "suffixes",
        ":meth:`_expression.HasSuffixes.suffix_with`",
        ":paramref:`.HasSuffixes.suffix_with.*suffixes`",
    )
    def suffix_with(
        self,
        *suffixes: _TextCoercedExpressionArgument[Any],
        dialect: str = "*",
    ) -> Self:
        r"""Add one or more expressions following the statement as a whole.

        This is used to support backend-specific suffix keywords on
        certain constructs.

        E.g.::

            stmt = (
                select(col1, col2)
                .cte()
                .suffix_with(
                    "cycle empno set y_cycle to 1 default 0", dialect="oracle"
                )
            )

        Multiple suffixes can be specified by multiple calls
        to :meth:`_expression.HasSuffixes.suffix_with`.

        :param \*suffixes: textual or :class:`_expression.ClauseElement`
         construct which
         will be rendered following the target clause.
        :param dialect: Optional string dialect name which will
         limit rendering of this suffix to only that dialect.

        """
        self._suffixes = self._suffixes + tuple(
            [
                (coercions.expect(roles.StatementOptionRole, p), dialect)
                for p in suffixes
            ]
        )
        return self


class HasHints:
    _hints: util.immutabledict[Tuple[FromClause, str], str] = (
        util.immutabledict()
    )
    _statement_hints: Tuple[Tuple[str, str], ...] = ()

    _has_hints_traverse_internals: _TraverseInternalsType = [
        ("_statement_hints", InternalTraversal.dp_statement_hint_list),
        ("_hints", InternalTraversal.dp_table_hint_list),
    ]

    @_generative
    def with_statement_hint(self, text: str, dialect_name: str = "*") -> Self:
        """Add a statement hint to this :class:`_expression.Select` or
        other selectable object.

        .. tip::

            :meth:`_expression.Select.with_statement_hint` generally adds hints
            **at the trailing end** of a SELECT statement.  To place
            dialect-specific hints such as optimizer hints at the **front** of
            the SELECT statement after the SELECT keyword, use the
            :meth:`_expression.Select.prefix_with` method for an open-ended
            space, or for table-specific hints the
            :meth:`_expression.Select.with_hint` may be used, which places
            hints in a dialect-specific location.

        This method is similar to :meth:`_expression.Select.with_hint` except
        that it does not require an individual table, and instead applies to
        the statement as a whole.

        Hints here are specific to the backend database and may include
        directives such as isolation levels, file directives, fetch directives,
        etc.

        .. seealso::

            :meth:`_expression.Select.with_hint`

            :meth:`_expression.Select.prefix_with` - generic SELECT prefixing
            which also can suit some database-specific HINT syntaxes such as
            MySQL or Oracle Database optimizer hints

        """
        return self._with_hint(None, text, dialect_name)

    @_generative
    def with_hint(
        self,
        selectable: _FromClauseArgument,
        text: str,
        dialect_name: str = "*",
    ) -> Self:
        r"""Add an indexing or other executional context hint for the given
        selectable to this :class:`_expression.Select` or other selectable
        object.

        .. tip::

            The :meth:`_expression.Select.with_hint` method adds hints that are
            **specific to a single table** to a statement, in a location that
            is **dialect-specific**.  To add generic optimizer hints to the
            **beginning** of a statement ahead of the SELECT keyword such as
            for MySQL or Oracle Database, use the
            :meth:`_expression.Select.prefix_with` method.  To add optimizer
            hints to the **end** of a statement such as for PostgreSQL, use the
            :meth:`_expression.Select.with_statement_hint` method.

        The text of the hint is rendered in the appropriate
        location for the database backend in use, relative
        to the given :class:`_schema.Table` or :class:`_expression.Alias`
        passed as the
        ``selectable`` argument. The dialect implementation
        typically uses Python string substitution syntax
        with the token ``%(name)s`` to render the name of
        the table or alias. E.g. when using Oracle Database, the
        following::

            select(mytable).with_hint(mytable, "index(%(name)s ix_mytable)")

        Would render SQL as:

        .. sourcecode:: sql

            select /*+ index(mytable ix_mytable) */ ... from mytable

        The ``dialect_name`` option will limit the rendering of a particular
        hint to a particular backend. Such as, to add hints for both Oracle
        Database and MSSql simultaneously::

            select(mytable).with_hint(
                mytable, "index(%(name)s ix_mytable)", "oracle"
            ).with_hint(mytable, "WITH INDEX ix_mytable", "mssql")

        .. seealso::

            :meth:`_expression.Select.with_statement_hint`

            :meth:`_expression.Select.prefix_with` - generic SELECT prefixing
            which also can suit some database-specific HINT syntaxes such as
            MySQL or Oracle Database optimizer hints

        """

        return self._with_hint(selectable, text, dialect_name)

    def _with_hint(
        self,
        selectable: Optional[_FromClauseArgument],
        text: str,
        dialect_name: str,
    ) -> Self:
        if selectable is None:
            self._statement_hints += ((dialect_name, text),)
        else:
            self._hints = self._hints.union(
                {
                    (
                        coercions.expect(roles.FromClauseRole, selectable),
                        dialect_name,
                    ): text
                }
            )
        return self


class FromClause(roles.AnonymizedFromClauseRole, Selectable):
    """Represent an element that can be used within the ``FROM``
    clause of a ``SELECT`` statement.

    The most common forms of :class:`_expression.FromClause` are the
    :class:`_schema.Table` and the :func:`_expression.select` constructs.  Key
    features common to all :class:`_expression.FromClause` objects include:

    * a :attr:`.c` collection, which provides per-name access to a collection
      of :class:`_expression.ColumnElement` objects.
    * a :attr:`.primary_key` attribute, which is a collection of all those
      :class:`_expression.ColumnElement`
      objects that indicate the ``primary_key`` flag.
    * Methods to generate various derivations of a "from" clause, including
      :meth:`_expression.FromClause.alias`,
      :meth:`_expression.FromClause.join`,
      :meth:`_expression.FromClause.select`.


    """

    __visit_name__ = "fromclause"
    named_with_column = False

    @util.ro_non_memoized_property
    def _hide_froms(self) -> Iterable[FromClause]:
        return ()

    _is_clone_of: Optional[FromClause]

    _columns: ColumnCollection[Any, Any]

    schema: Optional[str] = None
    """Define the 'schema' attribute for this :class:`_expression.FromClause`.

    This is typically ``None`` for most objects except that of
    :class:`_schema.Table`, where it is taken as the value of the
    :paramref:`_schema.Table.schema` argument.

    """

    is_selectable = True
    _is_from_clause = True
    _is_join = False

    _use_schema_map = False

    def select(self) -> Select[Any]:
        r"""Return a SELECT of this :class:`_expression.FromClause`.


        e.g.::

            stmt = some_table.select().where(some_table.c.id == 5)

        .. seealso::

            :func:`_expression.select` - general purpose
            method which allows for arbitrary column lists.

        """
        return Select(self)

    def join(
        self,
        right: _FromClauseArgument,
        onclause: Optional[_ColumnExpressionArgument[bool]] = None,
        isouter: bool = False,
        full: bool = False,
    ) -> Join:
        """Return a :class:`_expression.Join` from this
        :class:`_expression.FromClause`
        to another :class:`FromClause`.

        E.g.::

            from sqlalchemy import join

            j = user_table.join(
                address_table, user_table.c.id == address_table.c.user_id
            )
            stmt = select(user_table).select_from(j)

        would emit SQL along the lines of:

        .. sourcecode:: sql

            SELECT user.id, user.name FROM user
            JOIN address ON user.id = address.user_id

        :param right: the right side of the join; this is any
         :class:`_expression.FromClause` object such as a
         :class:`_schema.Table` object, and
         may also be a selectable-compatible object such as an ORM-mapped
         class.

        :param onclause: a SQL expression representing the ON clause of the
         join.  If left at ``None``, :meth:`_expression.FromClause.join`
         will attempt to
         join the two tables based on a foreign key relationship.

        :param isouter: if True, render a LEFT OUTER JOIN, instead of JOIN.

        :param full: if True, render a FULL OUTER JOIN, instead of LEFT OUTER
         JOIN.  Implies :paramref:`.FromClause.join.isouter`.

        .. seealso::

            :func:`_expression.join` - standalone function

            :class:`_expression.Join` - the type of object produced

        """

        return Join(self, right, onclause, isouter, full)

    def outerjoin(
        self,
        right: _FromClauseArgument,
        onclause: Optional[_ColumnExpressionArgument[bool]] = None,
        full: bool = False,
    ) -> Join:
        """Return a :class:`_expression.Join` from this
        :class:`_expression.FromClause`
        to another :class:`FromClause`, with the "isouter" flag set to
        True.

        E.g.::

            from sqlalchemy import outerjoin

            j = user_table.outerjoin(
                address_table, user_table.c.id == address_table.c.user_id
            )

        The above is equivalent to::

            j = user_table.join(
                address_table, user_table.c.id == address_table.c.user_id, isouter=True
            )

        :param right: the right side of the join; this is any
         :class:`_expression.FromClause` object such as a
         :class:`_schema.Table` object, and
         may also be a selectable-compatible object such as an ORM-mapped
         class.

        :param onclause: a SQL expression representing the ON clause of the
         join.  If left at ``None``, :meth:`_expression.FromClause.join`
         will attempt to
         join the two tables based on a foreign key relationship.

        :param full: if True, render a FULL OUTER JOIN, instead of
         LEFT OUTER JOIN.

        .. seealso::

            :meth:`_expression.FromClause.join`

            :class:`_expression.Join`

        """  # noqa: E501

        return Join(self, right, onclause, True, full)

    def alias(
        self, name: Optional[str] = None, flat: bool = False
    ) -> NamedFromClause:
        """Return an alias of this :class:`_expression.FromClause`.

        E.g.::

            a2 = some_table.alias("a2")

        The above code creates an :class:`_expression.Alias`
        object which can be used
        as a FROM clause in any SELECT statement.

        .. seealso::

            :ref:`tutorial_using_aliases`

            :func:`_expression.alias`

        """

        return Alias._construct(self, name=name)

    def tablesample(
        self,
        sampling: Union[float, Function[Any]],
        name: Optional[str] = None,
        seed: Optional[roles.ExpressionElementRole[Any]] = None,
    ) -> TableSample:
        """Return a TABLESAMPLE alias of this :class:`_expression.FromClause`.

        The return value is the :class:`_expression.TableSample`
        construct also
        provided by the top-level :func:`_expression.tablesample` function.

        .. seealso::

            :func:`_expression.tablesample` - usage guidelines and parameters

        """
        return TableSample._construct(
            self, sampling=sampling, name=name, seed=seed
        )

    def is_derived_from(self, fromclause: Optional[FromClause]) -> bool:
        """Return ``True`` if this :class:`_expression.FromClause` is
        'derived' from the given ``FromClause``.

        An example would be an Alias of a Table is derived from that Table.

        """
        # this is essentially an "identity" check in the base class.
        # Other constructs override this to traverse through
        # contained elements.
        return fromclause in self._cloned_set

    def _is_lexical_equivalent(self, other: FromClause) -> bool:
        """Return ``True`` if this :class:`_expression.FromClause` and
        the other represent the same lexical identity.

        This tests if either one is a copy of the other, or
        if they are the same via annotation identity.

        """
        return bool(self._cloned_set.intersection(other._cloned_set))

    @util.ro_non_memoized_property
    def description(self) -> str:
        """A brief description of this :class:`_expression.FromClause`.

        Used primarily for error message formatting.

        """
        return getattr(self, "name", self.__class__.__name__ + " object")

    def _generate_fromclause_column_proxies(
        self,
        fromclause: FromClause,
        columns: ColumnCollection[str, KeyedColumnElement[Any]],
        primary_key: ColumnSet,
        foreign_keys: Set[KeyedColumnElement[Any]],
    ) -> None:
        columns._populate_separate_keys(
            col._make_proxy(
                fromclause, primary_key=primary_key, foreign_keys=foreign_keys
            )
            for col in self.c
        )

    @util.ro_non_memoized_property
    def exported_columns(
        self,
    ) -> ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]:
        """A :class:`_expression.ColumnCollection`
        that represents the "exported"
        columns of this :class:`_expression.Selectable`.

        The "exported" columns for a :class:`_expression.FromClause`
        object are synonymous
        with the :attr:`_expression.FromClause.columns` collection.

        .. versionadded:: 1.4

        .. seealso::

            :attr:`_expression.Selectable.exported_columns`

            :attr:`_expression.SelectBase.exported_columns`


        """
        return self.c

    @util.ro_non_memoized_property
    def columns(
        self,
    ) -> ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]:
        """A named-based collection of :class:`_expression.ColumnElement`
        objects maintained by this :class:`_expression.FromClause`.

        The :attr:`.columns`, or :attr:`.c` collection, is the gateway
        to the construction of SQL expressions using table-bound or
        other selectable-bound columns::

            select(mytable).where(mytable.c.somecolumn == 5)

        :return: a :class:`.ColumnCollection` object.

        """
        return self.c

    @util.ro_memoized_property
    def c(self) -> ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]:
        """
        A synonym for :attr:`.FromClause.columns`

        :return: a :class:`.ColumnCollection`

        """
        if "_columns" not in self.__dict__:
            self._setup_collections()
        return self._columns.as_readonly()

    def _setup_collections(self) -> None:
        assert "_columns" not in self.__dict__
        assert "primary_key" not in self.__dict__
        assert "foreign_keys" not in self.__dict__

        _columns: ColumnCollection[Any, Any] = ColumnCollection()
        primary_key = ColumnSet()
        foreign_keys: Set[KeyedColumnElement[Any]] = set()

        self._populate_column_collection(
            columns=_columns,
            primary_key=primary_key,
            foreign_keys=foreign_keys,
        )

        # assigning these three collections separately is not itself atomic,
        # but greatly reduces the surface for problems
        self._columns = _columns
        self.primary_key = primary_key  # type: ignore
        self.foreign_keys = foreign_keys  # type: ignore

    @util.ro_non_memoized_property
    def entity_namespace(self) -> _EntityNamespace:
        """Return a namespace used for name-based access in SQL expressions.

        This is the namespace that is used to resolve "filter_by()" type
        expressions, such as::

            stmt.filter_by(address="some address")

        It defaults to the ``.c`` collection, however internally it can
        be overridden using the "entity_namespace" annotation to deliver
        alternative results.

        """
        return self.c

    @util.ro_memoized_property
    def primary_key(self) -> Iterable[NamedColumn[Any]]:
        """Return the iterable collection of :class:`_schema.Column` objects
        which comprise the primary key of this :class:`_selectable.FromClause`.

        For a :class:`_schema.Table` object, this collection is represented
        by the :class:`_schema.PrimaryKeyConstraint` which itself is an
        iterable collection of :class:`_schema.Column` objects.

        """
        self._setup_collections()
        return self.primary_key

    @util.ro_memoized_property
    def foreign_keys(self) -> Iterable[ForeignKey]:
        """Return the collection of :class:`_schema.ForeignKey` marker objects
        which this FromClause references.

        Each :class:`_schema.ForeignKey` is a member of a
        :class:`_schema.Table`-wide
        :class:`_schema.ForeignKeyConstraint`.

        .. seealso::

            :attr:`_schema.Table.foreign_key_constraints`

        """
        self._setup_collections()
        return self.foreign_keys

    def _reset_column_collection(self) -> None:
        """Reset the attributes linked to the ``FromClause.c`` attribute.

        This collection is separate from all the other memoized things
        as it has shown to be sensitive to being cleared out in situations
        where enclosing code, typically in a replacement traversal scenario,
        has already established strong relationships
        with the exported columns.

        The collection is cleared for the case where a table is having a
        column added to it as well as within a Join during copy internals.

        """

        for key in ["_columns", "columns", "c", "primary_key", "foreign_keys"]:
            self.__dict__.pop(key, None)

    @util.ro_non_memoized_property
    def _select_iterable(self) -> _SelectIterable:
        return (c for c in self.c if not _never_select_column(c))

    @property
    def _cols_populated(self) -> bool:
        return "_columns" in self.__dict__

    def _populate_column_collection(
        self,
        columns: ColumnCollection[str, KeyedColumnElement[Any]],
        primary_key: ColumnSet,
        foreign_keys: Set[KeyedColumnElement[Any]],
    ) -> None:
        """Called on subclasses to establish the .c collection.

        Each implementation has a different way of establishing
        this collection.

        """

    def _refresh_for_new_column(self, column: ColumnElement[Any]) -> None:
        """Given a column added to the .c collection of an underlying
        selectable, produce the local version of that column, assuming this
        selectable ultimately should proxy this column.

        this is used to "ping" a derived selectable to add a new column
        to its .c. collection when a Column has been added to one of the
        Table objects it ultimately derives from.

        If the given selectable hasn't populated its .c. collection yet,
        it should at least pass on the message to the contained selectables,
        but it will return None.

        This method is currently used by Declarative to allow Table
        columns to be added to a partially constructed inheritance
        mapping that may have already produced joins.  The method
        isn't public right now, as the full span of implications
        and/or caveats aren't yet clear.

        It's also possible that this functionality could be invoked by
        default via an event, which would require that
        selectables maintain a weak referencing collection of all
        derivations.

        """
        self._reset_column_collection()

    def _anonymous_fromclause(
        self, *, name: Optional[str] = None, flat: bool = False
    ) -> FromClause:
        return self.alias(name=name)

    if TYPE_CHECKING:

        def self_group(
            self, against: Optional[OperatorType] = None
        ) -> Union[FromGrouping, Self]: ...


class NamedFromClause(FromClause):
    """A :class:`.FromClause` that has a name.

    Examples include tables, subqueries, CTEs, aliased tables.

    .. versionadded:: 2.0

    """

    named_with_column = True

    name: str

    @util.preload_module("sqlalchemy.sql.sqltypes")
    def table_valued(self) -> TableValuedColumn[Any]:
        """Return a :class:`_sql.TableValuedColumn` object for this
        :class:`_expression.FromClause`.

        A :class:`_sql.TableValuedColumn` is a :class:`_sql.ColumnElement` that
        represents a complete row in a table. Support for this construct is
        backend dependent, and is supported in various forms by backends
        such as PostgreSQL, Oracle Database and SQL Server.

        E.g.:

        .. sourcecode:: pycon+sql

            >>> from sqlalchemy import select, column, func, table
            >>> a = table("a", column("id"), column("x"), column("y"))
            >>> stmt = select(func.row_to_json(a.table_valued()))
            >>> print(stmt)
            {printsql}SELECT row_to_json(a) AS row_to_json_1
            FROM a

        .. versionadded:: 1.4.0b2

        .. seealso::

            :ref:`tutorial_functions` - in the :ref:`unified_tutorial`

        """
        return TableValuedColumn(self, type_api.TABLEVALUE)


class SelectLabelStyle(Enum):
    """Label style constants that may be passed to
    :meth:`_sql.Select.set_label_style`."""

    LABEL_STYLE_NONE = 0
    """Label style indicating no automatic labeling should be applied to the
    columns clause of a SELECT statement.

    Below, the columns named ``columna`` are both rendered as is, meaning that
    the name ``columna`` can only refer to the first occurrence of this name
    within a result set, as well as if the statement were used as a subquery:

    .. sourcecode:: pycon+sql

        >>> from sqlalchemy import table, column, select, true, LABEL_STYLE_NONE
        >>> table1 = table("table1", column("columna"), column("columnb"))
        >>> table2 = table("table2", column("columna"), column("columnc"))
        >>> print(
        ...     select(table1, table2)
        ...     .join(table2, true())
        ...     .set_label_style(LABEL_STYLE_NONE)
        ... )
        {printsql}SELECT table1.columna, table1.columnb, table2.columna, table2.columnc
        FROM table1 JOIN table2 ON true

    Used with the :meth:`_sql.Select.set_label_style` method.

    .. versionadded:: 1.4

    """  # noqa: E501

    LABEL_STYLE_TABLENAME_PLUS_COL = 1
    """Label style indicating all columns should be labeled as
    ``<tablename>_<columnname>`` when generating the columns clause of a SELECT
    statement, to disambiguate same-named columns referenced from different
    tables, aliases, or subqueries.

    Below, all column names are given a label so that the two same-named
    columns ``columna`` are disambiguated as ``table1_columna`` and
    ``table2_columna``:

    .. sourcecode:: pycon+sql

        >>> from sqlalchemy import (
        ...     table,
        ...     column,
        ...     select,
        ...     true,
        ...     LABEL_STYLE_TABLENAME_PLUS_COL,
        ... )
        >>> table1 = table("table1", column("columna"), column("columnb"))
        >>> table2 = table("table2", column("columna"), column("columnc"))
        >>> print(
        ...     select(table1, table2)
        ...     .join(table2, true())
        ...     .set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)
        ... )
        {printsql}SELECT table1.columna AS table1_columna, table1.columnb AS table1_columnb, table2.columna AS table2_columna, table2.columnc AS table2_columnc
        FROM table1 JOIN table2 ON true

    Used with the :meth:`_sql.GenerativeSelect.set_label_style` method.
    Equivalent to the legacy method ``Select.apply_labels()``;
    :data:`_sql.LABEL_STYLE_TABLENAME_PLUS_COL` is SQLAlchemy's legacy
    auto-labeling style. :data:`_sql.LABEL_STYLE_DISAMBIGUATE_ONLY` provides a
    less intrusive approach to disambiguation of same-named column expressions.


    .. versionadded:: 1.4

    """  # noqa: E501

    LABEL_STYLE_DISAMBIGUATE_ONLY = 2
    """Label style indicating that columns with a name that conflicts with
    an existing name should be labeled with a semi-anonymizing label
    when generating the columns clause of a SELECT statement.

    Below, most column names are left unaffected, except for the second
    occurrence of the name ``columna``, which is labeled using the
    label ``columna_1`` to disambiguate it from that of ``tablea.columna``:

    .. sourcecode:: pycon+sql

        >>> from sqlalchemy import (
        ...     table,
        ...     column,
        ...     select,
        ...     true,
        ...     LABEL_STYLE_DISAMBIGUATE_ONLY,
        ... )
        >>> table1 = table("table1", column("columna"), column("columnb"))
        >>> table2 = table("table2", column("columna"), column("columnc"))
        >>> print(
        ...     select(table1, table2)
        ...     .join(table2, true())
        ...     .set_label_style(LABEL_STYLE_DISAMBIGUATE_ONLY)
        ... )
        {printsql}SELECT table1.columna, table1.columnb, table2.columna AS columna_1, table2.columnc
        FROM table1 JOIN table2 ON true

    Used with the :meth:`_sql.GenerativeSelect.set_label_style` method,
    :data:`_sql.LABEL_STYLE_DISAMBIGUATE_ONLY` is the default labeling style
    for all SELECT statements outside of :term:`1.x style` ORM queries.

    .. versionadded:: 1.4

    """  # noqa: E501

    LABEL_STYLE_DEFAULT = LABEL_STYLE_DISAMBIGUATE_ONLY
    """The default label style, refers to
    :data:`_sql.LABEL_STYLE_DISAMBIGUATE_ONLY`.

    .. versionadded:: 1.4

    """

    LABEL_STYLE_LEGACY_ORM = 3


(
    LABEL_STYLE_NONE,
    LABEL_STYLE_TABLENAME_PLUS_COL,
    LABEL_STYLE_DISAMBIGUATE_ONLY,
    _,
) = list(SelectLabelStyle)

LABEL_STYLE_DEFAULT = LABEL_STYLE_DISAMBIGUATE_ONLY


class Join(roles.DMLTableRole, FromClause):
    """Represent a ``JOIN`` construct between two
    :class:`_expression.FromClause`
    elements.

    The public constructor function for :class:`_expression.Join`
    is the module-level
    :func:`_expression.join()` function, as well as the
    :meth:`_expression.FromClause.join` method
    of any :class:`_expression.FromClause` (e.g. such as
    :class:`_schema.Table`).

    .. seealso::

        :func:`_expression.join`

        :meth:`_expression.FromClause.join`

    """

    __visit_name__ = "join"

    _traverse_internals: _TraverseInternalsType = [
        ("left", InternalTraversal.dp_clauseelement),
        ("right", InternalTraversal.dp_clauseelement),
        ("onclause", InternalTraversal.dp_clauseelement),
        ("isouter", InternalTraversal.dp_boolean),
        ("full", InternalTraversal.dp_boolean),
    ]

    _is_join = True

    left: FromClause
    right: FromClause
    onclause: Optional[ColumnElement[bool]]
    isouter: bool
    full: bool

    def __init__(
        self,
        left: _FromClauseArgument,
        right: _FromClauseArgument,
        onclause: Optional[_OnClauseArgument] = None,
        isouter: bool = False,
        full: bool = False,
    ):
        """Construct a new :class:`_expression.Join`.

        The usual entrypoint here is the :func:`_expression.join`
        function or the :meth:`_expression.FromClause.join` method of any
        :class:`_expression.FromClause` object.

        """

        # when deannotate was removed here, callcounts went up for ORM
        # compilation of eager joins, since there were more comparisons of
        # annotated objects.   test_orm.py -> test_fetch_results
        # was therefore changed to show a more real-world use case, where the
        # compilation is cached; there's no change in post-cache callcounts.
        # callcounts for a single compilation in that particular test
        # that includes about eight joins about 1100 extra fn calls, from
        # 29200 -> 30373

        self.left = coercions.expect(
            roles.FromClauseRole,
            left,
        )
        self.right = coercions.expect(
            roles.FromClauseRole,
            right,
        ).self_group()

        if onclause is None:
            self.onclause = self._match_primaries(self.left, self.right)
        else:
            # note: taken from If91f61527236fd4d7ae3cad1f24c38be921c90ba
            # not merged yet
            self.onclause = coercions.expect(
                roles.OnClauseRole, onclause
            ).self_group(against=operators._asbool)

        self.isouter = isouter
        self.full = full

    @util.ro_non_memoized_property
    def description(self) -> str:
        return "Join object on %s(%d) and %s(%d)" % (
            self.left.description,
            id(self.left),
            self.right.description,
            id(self.right),
        )

    def is_derived_from(self, fromclause: Optional[FromClause]) -> bool:
        return (
            # use hash() to ensure direct comparison to annotated works
            # as well
            hash(fromclause) == hash(self)
            or self.left.is_derived_from(fromclause)
            or self.right.is_derived_from(fromclause)
        )

    def self_group(
        self, against: Optional[OperatorType] = None
    ) -> FromGrouping:
        return FromGrouping(self)

    @util.preload_module("sqlalchemy.sql.util")
    def _populate_column_collection(
        self,
        columns: ColumnCollection[str, KeyedColumnElement[Any]],
        primary_key: ColumnSet,
        foreign_keys: Set[KeyedColumnElement[Any]],
    ) -> None:
        sqlutil = util.preloaded.sql_util
        _columns: List[KeyedColumnElement[Any]] = [c for c in self.left.c] + [
            c for c in self.right.c
        ]

        primary_key.extend(  # type: ignore
            sqlutil.reduce_columns(
                (c for c in _columns if c.primary_key), self.onclause
            )
        )
        columns._populate_separate_keys(
            (col._tq_key_label, col) for col in _columns  # type: ignore
        )
        foreign_keys.update(
            itertools.chain(*[col.foreign_keys for col in _columns])  # type: ignore  # noqa: E501
        )

    def _copy_internals(
        self, clone: _CloneCallableType = _clone, **kw: Any
    ) -> None:
        # see Select._copy_internals() for similar concept

        # here we pre-clone "left" and "right" so that we can
        # determine the new FROM clauses
        all_the_froms = set(
            itertools.chain(
                _from_objects(self.left),
                _from_objects(self.right),
            )
        )

        # run the clone on those.  these will be placed in the
        # cache used by the clone function
        new_froms = {f: clone(f, **kw) for f in all_the_froms}

        # set up a special replace function that will replace for
        # ColumnClause with parent table referring to those
        # replaced FromClause objects
        def replace(
            obj: Union[BinaryExpression[Any], ColumnClause[Any]],
            **kw: Any,
        ) -> Optional[KeyedColumnElement[Any]]:
            if isinstance(obj, ColumnClause) and obj.table in new_froms:
                newelem = new_froms[obj.table].corresponding_column(obj)
                return newelem
            return None

        kw["replace"] = replace

        # run normal _copy_internals.  the clones for
        # left and right will come from the clone function's
        # cache
        super()._copy_internals(clone=clone, **kw)

        self._reset_memoizations()

    def _refresh_for_new_column(self, column: ColumnElement[Any]) -> None:
        super()._refresh_for_new_column(column)
        self.left._refresh_for_new_column(column)
        self.right._refresh_for_new_column(column)

    def _match_primaries(
        self,
        left: FromClause,
        right: FromClause,
    ) -> ColumnElement[bool]:
        if isinstance(left, Join):
            left_right = left.right
        else:
            left_right = None
        return self._join_condition(left, right, a_subset=left_right)

    @classmethod
    def _join_condition(
        cls,
        a: FromClause,
        b: FromClause,
        *,
        a_subset: Optional[FromClause] = None,
        consider_as_foreign_keys: Optional[
            AbstractSet[ColumnClause[Any]]
        ] = None,
    ) -> ColumnElement[bool]:
        """Create a join condition between two tables or selectables.

        See sqlalchemy.sql.util.join_condition() for full docs.

        """
        constraints = cls._joincond_scan_left_right(
            a, a_subset, b, consider_as_foreign_keys
        )

        if len(constraints) > 1:
            cls._joincond_trim_constraints(
                a, b, constraints, consider_as_foreign_keys
            )

        if len(constraints) == 0:
            if isinstance(b, FromGrouping):
                hint = (
                    " Perhaps you meant to convert the right side to a "
                    "subquery using alias()?"
                )
            else:
                hint = ""
            raise exc.NoForeignKeysError(
                "Can't find any foreign key relationships "
                "between '%s' and '%s'.%s"
                % (a.description, b.description, hint)
            )

        crit = [(x == y) for x, y in list(constraints.values())[0]]
        if len(crit) == 1:
            return crit[0]
        else:
            return and_(*crit)

    @classmethod
    def _can_join(
        cls,
        left: FromClause,
        right: FromClause,
        *,
        consider_as_foreign_keys: Optional[
            AbstractSet[ColumnClause[Any]]
        ] = None,
    ) -> bool:
        if isinstance(left, Join):
            left_right = left.right
        else:
            left_right = None

        constraints = cls._joincond_scan_left_right(
            a=left,
            b=right,
            a_subset=left_right,
            consider_as_foreign_keys=consider_as_foreign_keys,
        )

        return bool(constraints)

    @classmethod
    @util.preload_module("sqlalchemy.sql.util")
    def _joincond_scan_left_right(
        cls,
        a: FromClause,
        a_subset: Optional[FromClause],
        b: FromClause,
        consider_as_foreign_keys: Optional[AbstractSet[ColumnClause[Any]]],
    ) -> collections.defaultdict[
        Optional[ForeignKeyConstraint],
        List[Tuple[ColumnClause[Any], ColumnClause[Any]]],
    ]:
        sql_util = util.preloaded.sql_util

        a = coercions.expect(roles.FromClauseRole, a)
        b = coercions.expect(roles.FromClauseRole, b)

        constraints: collections.defaultdict[
            Optional[ForeignKeyConstraint],
            List[Tuple[ColumnClause[Any], ColumnClause[Any]]],
        ] = collections.defaultdict(list)

        for left in (a_subset, a):
            if left is None:
                continue
            for fk in sorted(
                b.foreign_keys,
                key=lambda fk: fk.parent._creation_order,
            ):
                if (
                    consider_as_foreign_keys is not None
                    and fk.parent not in consider_as_foreign_keys
                ):
                    continue
                try:
                    col = fk.get_referent(left)
                except exc.NoReferenceError as nrte:
                    table_names = {t.name for t in sql_util.find_tables(left)}
                    if nrte.table_name in table_names:
                        raise
                    else:
                        continue

                if col is not None:
                    constraints[fk.constraint].append((col, fk.parent))
            if left is not b:
                for fk in sorted(
                    left.foreign_keys,
                    key=lambda fk: fk.parent._creation_order,
                ):
                    if (
                        consider_as_foreign_keys is not None
                        and fk.parent not in consider_as_foreign_keys
                    ):
                        continue
                    try:
                        col = fk.get_referent(b)
                    except exc.NoReferenceError as nrte:
                        table_names = {t.name for t in sql_util.find_tables(b)}
                        if nrte.table_name in table_names:
                            raise
                        else:
                            continue

                    if col is not None:
                        constraints[fk.constraint].append((col, fk.parent))
            if constraints:
                break
        return constraints

    @classmethod
    def _joincond_trim_constraints(
        cls,
        a: FromClause,
        b: FromClause,
        constraints: Dict[Any, Any],
        consider_as_foreign_keys: Optional[Any],
    ) -> None:
        # more than one constraint matched.  narrow down the list
        # to include just those FKCs that match exactly to
        # "consider_as_foreign_keys".
        if consider_as_foreign_keys:
            for const in list(constraints):
                if {f.parent for f in const.elements} != set(
                    consider_as_foreign_keys
                ):
                    del constraints[const]

        # if still multiple constraints, but
        # they all refer to the exact same end result, use it.
        if len(constraints) > 1:
            dedupe = {tuple(crit) for crit in constraints.values()}
            if len(dedupe) == 1:
                key = list(constraints)[0]
                constraints = {key: constraints[key]}

        if len(constraints) != 1:
            raise exc.AmbiguousForeignKeysError(
                "Can't determine join between '%s' and '%s'; "
                "tables have more than one foreign key "
                "constraint relationship between them. "
                "Please specify the 'onclause' of this "
                "join explicitly." % (a.description, b.description)
            )

    def select(self) -> Select[Any]:
        r"""Create a :class:`_expression.Select` from this
        :class:`_expression.Join`.

        E.g.::

            stmt = table_a.join(table_b, table_a.c.id == table_b.c.a_id)

            stmt = stmt.select()

        The above will produce a SQL string resembling:

        .. sourcecode:: sql

            SELECT table_a.id, table_a.col, table_b.id, table_b.a_id
            FROM table_a JOIN table_b ON table_a.id = table_b.a_id

        """
        return Select(self.left, self.right).select_from(self)

    @util.preload_module("sqlalchemy.sql.util")
    def _anonymous_fromclause(
        self, name: Optional[str] = None, flat: bool = False
    ) -> TODO_Any:
        sqlutil = util.preloaded.sql_util
        if flat:
            if isinstance(self.left, (FromGrouping, Join)):
                left_name = name  # will recurse
            else:
                if name and isinstance(self.left, NamedFromClause):
                    left_name = f"{name}_{self.left.name}"
                else:
                    left_name = name
            if isinstance(self.right, (FromGrouping, Join)):
                right_name = name  # will recurse
            else:
                if name and isinstance(self.right, NamedFromClause):
                    right_name = f"{name}_{self.right.name}"
                else:
                    right_name = name
            left_a, right_a = (
                self.left._anonymous_fromclause(name=left_name, flat=flat),
                self.right._anonymous_fromclause(name=right_name, flat=flat),
            )
            adapter = sqlutil.ClauseAdapter(left_a).chain(
                sqlutil.ClauseAdapter(right_a)
            )

            return left_a.join(
                right_a,
                adapter.traverse(self.onclause),
                isouter=self.isouter,
                full=self.full,
            )
        else:
            return (
                self.select()
                .set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)
                .correlate(None)
                .alias(name)
            )

    @util.ro_non_memoized_property
    def _hide_froms(self) -> Iterable[FromClause]:
        return itertools.chain(
            *[_from_objects(x.left, x.right) for x in self._cloned_set]
        )

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        self_list: List[FromClause] = [self]
        return self_list + self.left._from_objects + self.right._from_objects


class NoInit:
    def __init__(self, *arg: Any, **kw: Any):
        raise NotImplementedError(
            "The %s class is not intended to be constructed "
            "directly.  Please use the %s() standalone "
            "function or the %s() method available from appropriate "
            "selectable objects."
            % (
                self.__class__.__name__,
                self.__class__.__name__.lower(),
                self.__class__.__name__.lower(),
            )
        )


class LateralFromClause(NamedFromClause):
    """mark a FROM clause as being able to render directly as LATERAL"""


# FromClause ->
#   AliasedReturnsRows
#        -> Alias   only for FromClause
#        -> Subquery  only for SelectBase
#        -> CTE only for HasCTE -> SelectBase, DML
#        -> Lateral -> FromClause, but we accept SelectBase
#           w/ non-deprecated coercion
#        -> TableSample -> only for FromClause


class AliasedReturnsRows(NoInit, NamedFromClause):
    """Base class of aliases against tables, subqueries, and other
    selectables."""

    _is_from_container = True

    _supports_derived_columns = False

    element: ReturnsRows

    _traverse_internals: _TraverseInternalsType = [
        ("element", InternalTraversal.dp_clauseelement),
        ("name", InternalTraversal.dp_anon_name),
    ]

    @classmethod
    def _construct(
        cls,
        selectable: Any,
        *,
        name: Optional[str] = None,
        **kw: Any,
    ) -> Self:
        obj = cls.__new__(cls)
        obj._init(selectable, name=name, **kw)
        return obj

    def _init(self, selectable: Any, *, name: Optional[str] = None) -> None:
        self.element = coercions.expect(
            roles.ReturnsRowsRole, selectable, apply_propagate_attrs=self
        )
        self.element = selectable
        self._orig_name = name
        if name is None:
            if (
                isinstance(selectable, FromClause)
                and selectable.named_with_column
            ):
                name = getattr(selectable, "name", None)
                if isinstance(name, _anonymous_label):
                    name = None
            name = _anonymous_label.safe_construct(id(self), name or "anon")
        self.name = name

    def _refresh_for_new_column(self, column: ColumnElement[Any]) -> None:
        super()._refresh_for_new_column(column)
        self.element._refresh_for_new_column(column)

    def _populate_column_collection(
        self,
        columns: ColumnCollection[str, KeyedColumnElement[Any]],
        primary_key: ColumnSet,
        foreign_keys: Set[KeyedColumnElement[Any]],
    ) -> None:
        self.element._generate_fromclause_column_proxies(
            self, columns, primary_key=primary_key, foreign_keys=foreign_keys
        )

    @util.ro_non_memoized_property
    def description(self) -> str:
        name = self.name
        if isinstance(name, _anonymous_label):
            name = "anon_1"

        return name

    @util.ro_non_memoized_property
    def implicit_returning(self) -> bool:
        return self.element.implicit_returning  # type: ignore

    @property
    def original(self) -> ReturnsRows:
        """Legacy for dialects that are referring to Alias.original."""
        return self.element

    def is_derived_from(self, fromclause: Optional[FromClause]) -> bool:
        if fromclause in self._cloned_set:
            return True
        return self.element.is_derived_from(fromclause)

    def _copy_internals(
        self, clone: _CloneCallableType = _clone, **kw: Any
    ) -> None:
        existing_element = self.element

        super()._copy_internals(clone=clone, **kw)

        # the element clone is usually against a Table that returns the
        # same object.  don't reset exported .c. collections and other
        # memoized details if it was not changed.  this saves a lot on
        # performance.
        if existing_element is not self.element:
            self._reset_column_collection()

    @property
    def _from_objects(self) -> List[FromClause]:
        return [self]


class FromClauseAlias(AliasedReturnsRows):
    element: FromClause


class Alias(roles.DMLTableRole, FromClauseAlias):
    """Represents an table or selectable alias (AS).

    Represents an alias, as typically applied to any table or
    sub-select within a SQL statement using the ``AS`` keyword (or
    without the keyword on certain databases such as Oracle Database).

    This object is constructed from the :func:`_expression.alias` module
    level function as well as the :meth:`_expression.FromClause.alias`
    method available
    on all :class:`_expression.FromClause` subclasses.

    .. seealso::

        :meth:`_expression.FromClause.alias`

    """

    __visit_name__ = "alias"

    inherit_cache = True

    element: FromClause

    @classmethod
    def _factory(
        cls,
        selectable: FromClause,
        name: Optional[str] = None,
        flat: bool = False,
    ) -> NamedFromClause:
        return coercions.expect(
            roles.FromClauseRole, selectable, allow_select=True
        ).alias(name=name, flat=flat)


class TableValuedAlias(LateralFromClause, Alias):
    """An alias against a "table valued" SQL function.

    This construct provides for a SQL function that returns columns
    to be used in the FROM clause of a SELECT statement.   The
    object is generated using the :meth:`_functions.FunctionElement.table_valued`
    method, e.g.:

    .. sourcecode:: pycon+sql

        >>> from sqlalchemy import select, func
        >>> fn = func.json_array_elements_text('["one", "two", "three"]').table_valued(
        ...     "value"
        ... )
        >>> print(select(fn.c.value))
        {printsql}SELECT anon_1.value
        FROM json_array_elements_text(:json_array_elements_text_1) AS anon_1

    .. versionadded:: 1.4.0b2

    .. seealso::

        :ref:`tutorial_functions_table_valued` - in the :ref:`unified_tutorial`

    """  # noqa: E501

    __visit_name__ = "table_valued_alias"

    _supports_derived_columns = True
    _render_derived = False
    _render_derived_w_types = False
    joins_implicitly = False

    _traverse_internals: _TraverseInternalsType = [
        ("element", InternalTraversal.dp_clauseelement),
        ("name", InternalTraversal.dp_anon_name),
        ("_tableval_type", InternalTraversal.dp_type),
        ("_render_derived", InternalTraversal.dp_boolean),
        ("_render_derived_w_types", InternalTraversal.dp_boolean),
    ]

    def _init(
        self,
        selectable: Any,
        *,
        name: Optional[str] = None,
        table_value_type: Optional[TableValueType] = None,
        joins_implicitly: bool = False,
    ) -> None:
        super()._init(selectable, name=name)

        self.joins_implicitly = joins_implicitly
        self._tableval_type = (
            type_api.TABLEVALUE
            if table_value_type is None
            else table_value_type
        )

    @HasMemoized.memoized_attribute
    def column(self) -> TableValuedColumn[Any]:
        """Return a column expression representing this
        :class:`_sql.TableValuedAlias`.

        This accessor is used to implement the
        :meth:`_functions.FunctionElement.column_valued` method. See that
        method for further details.

        E.g.:

        .. sourcecode:: pycon+sql

            >>> print(select(func.some_func().table_valued("value").column))
            {printsql}SELECT anon_1 FROM some_func() AS anon_1

        .. seealso::

            :meth:`_functions.FunctionElement.column_valued`

        """

        return TableValuedColumn(self, self._tableval_type)

    def alias(
        self, name: Optional[str] = None, flat: bool = False
    ) -> TableValuedAlias:
        """Return a new alias of this :class:`_sql.TableValuedAlias`.

        This creates a distinct FROM object that will be distinguished
        from the original one when used in a SQL statement.

        """

        tva: TableValuedAlias = TableValuedAlias._construct(
            self,
            name=name,
            table_value_type=self._tableval_type,
            joins_implicitly=self.joins_implicitly,
        )

        if self._render_derived:
            tva._render_derived = True
            tva._render_derived_w_types = self._render_derived_w_types

        return tva

    def lateral(self, name: Optional[str] = None) -> LateralFromClause:
        """Return a new :class:`_sql.TableValuedAlias` with the lateral flag
        set, so that it renders as LATERAL.

        .. seealso::

            :func:`_expression.lateral`

        """
        tva = self.alias(name=name)
        tva._is_lateral = True
        return tva

    def render_derived(
        self,
        name: Optional[str] = None,
        with_types: bool = False,
    ) -> TableValuedAlias:
        """Apply "render derived" to this :class:`_sql.TableValuedAlias`.

        This has the effect of the individual column names listed out
        after the alias name in the "AS" sequence, e.g.:

        .. sourcecode:: pycon+sql

            >>> print(
            ...     select(
            ...         func.unnest(array(["one", "two", "three"]))
            ...         .table_valued("x", with_ordinality="o")
            ...         .render_derived()
            ...     )
            ... )
            {printsql}SELECT anon_1.x, anon_1.o
            FROM unnest(ARRAY[%(param_1)s, %(param_2)s, %(param_3)s]) WITH ORDINALITY AS anon_1(x, o)

        The ``with_types`` keyword will render column types inline within
        the alias expression (this syntax currently applies to the
        PostgreSQL database):

        .. sourcecode:: pycon+sql

            >>> print(
            ...     select(
            ...         func.json_to_recordset('[{"a":1,"b":"foo"},{"a":"2","c":"bar"}]')
            ...         .table_valued(column("a", Integer), column("b", String))
            ...         .render_derived(with_types=True)
            ...     )
            ... )
            {printsql}SELECT anon_1.a, anon_1.b FROM json_to_recordset(:json_to_recordset_1)
            AS anon_1(a INTEGER, b VARCHAR)

        :param name: optional string name that will be applied to the alias
         generated.  If left as None, a unique anonymizing name will be used.

        :param with_types: if True, the derived columns will include the
         datatype specification with each column. This is a special syntax
         currently known to be required by PostgreSQL for some SQL functions.

        """  # noqa: E501

        # note: don't use the @_generative system here, keep a reference
        # to the original object.  otherwise you can have re-use of the
        # python id() of the original which can cause name conflicts if
        # a new anon-name grabs the same identifier as the local anon-name
        # (just saw it happen on CI)

        # construct against original to prevent memory growth
        # for repeated generations
        new_alias: TableValuedAlias = TableValuedAlias._construct(
            self.element,
            name=name,
            table_value_type=self._tableval_type,
            joins_implicitly=self.joins_implicitly,
        )
        new_alias._render_derived = True
        new_alias._render_derived_w_types = with_types
        return new_alias


class Lateral(FromClauseAlias, LateralFromClause):
    """Represent a LATERAL subquery.

    This object is constructed from the :func:`_expression.lateral` module
    level function as well as the :meth:`_expression.FromClause.lateral`
    method available
    on all :class:`_expression.FromClause` subclasses.

    While LATERAL is part of the SQL standard, currently only more recent
    PostgreSQL versions provide support for this keyword.

    .. seealso::

        :ref:`tutorial_lateral_correlation` -  overview of usage.

    """

    __visit_name__ = "lateral"
    _is_lateral = True

    inherit_cache = True

    @classmethod
    def _factory(
        cls,
        selectable: Union[SelectBase, _FromClauseArgument],
        name: Optional[str] = None,
    ) -> LateralFromClause:
        return coercions.expect(
            roles.FromClauseRole, selectable, explicit_subquery=True
        ).lateral(name=name)


class TableSample(FromClauseAlias):
    """Represent a TABLESAMPLE clause.

    This object is constructed from the :func:`_expression.tablesample` module
    level function as well as the :meth:`_expression.FromClause.tablesample`
    method
    available on all :class:`_expression.FromClause` subclasses.

    .. seealso::

        :func:`_expression.tablesample`

    """

    __visit_name__ = "tablesample"

    _traverse_internals: _TraverseInternalsType = (
        AliasedReturnsRows._traverse_internals
        + [
            ("sampling", InternalTraversal.dp_clauseelement),
            ("seed", InternalTraversal.dp_clauseelement),
        ]
    )

    @classmethod
    def _factory(
        cls,
        selectable: _FromClauseArgument,
        sampling: Union[float, Function[Any]],
        name: Optional[str] = None,
        seed: Optional[roles.ExpressionElementRole[Any]] = None,
    ) -> TableSample:
        return coercions.expect(roles.FromClauseRole, selectable).tablesample(
            sampling, name=name, seed=seed
        )

    @util.preload_module("sqlalchemy.sql.functions")
    def _init(  # type: ignore[override]
        self,
        selectable: Any,
        *,
        name: Optional[str] = None,
        sampling: Union[float, Function[Any]],
        seed: Optional[roles.ExpressionElementRole[Any]] = None,
    ) -> None:
        assert sampling is not None
        functions = util.preloaded.sql_functions
        if not isinstance(sampling, functions.Function):
            sampling = functions.func.system(sampling)

        self.sampling: Function[Any] = sampling
        self.seed = seed
        super()._init(selectable, name=name)

    def _get_method(self) -> Function[Any]:
        return self.sampling


class CTE(
    roles.DMLTableRole,
    roles.IsCTERole,
    Generative,
    HasPrefixes,
    HasSuffixes,
    AliasedReturnsRows,
):
    """Represent a Common Table Expression.

    The :class:`_expression.CTE` object is obtained using the
    :meth:`_sql.SelectBase.cte` method from any SELECT statement. A less often
    available syntax also allows use of the :meth:`_sql.HasCTE.cte` method
    present on :term:`DML` constructs such as :class:`_sql.Insert`,
    :class:`_sql.Update` and
    :class:`_sql.Delete`.   See the :meth:`_sql.HasCTE.cte` method for
    usage details on CTEs.

    .. seealso::

        :ref:`tutorial_subqueries_ctes` - in the 2.0 tutorial

        :meth:`_sql.HasCTE.cte` - examples of calling styles

    """

    __visit_name__ = "cte"

    _traverse_internals: _TraverseInternalsType = (
        AliasedReturnsRows._traverse_internals
        + [
            ("_cte_alias", InternalTraversal.dp_clauseelement),
            ("_restates", InternalTraversal.dp_clauseelement),
            ("recursive", InternalTraversal.dp_boolean),
            ("nesting", InternalTraversal.dp_boolean),
        ]
        + HasPrefixes._has_prefixes_traverse_internals
        + HasSuffixes._has_suffixes_traverse_internals
    )

    element: HasCTE

    @classmethod
    def _factory(
        cls,
        selectable: HasCTE,
        name: Optional[str] = None,
        recursive: bool = False,
    ) -> CTE:
        r"""Return a new :class:`_expression.CTE`,
        or Common Table Expression instance.

        Please see :meth:`_expression.HasCTE.cte` for detail on CTE usage.

        """
        return coercions.expect(roles.HasCTERole, selectable).cte(
            name=name, recursive=recursive
        )

    def _init(
        self,
        selectable: Select[Any],
        *,
        name: Optional[str] = None,
        recursive: bool = False,
        nesting: bool = False,
        _cte_alias: Optional[CTE] = None,
        _restates: Optional[CTE] = None,
        _prefixes: Optional[Tuple[()]] = None,
        _suffixes: Optional[Tuple[()]] = None,
    ) -> None:
        self.recursive = recursive
        self.nesting = nesting
        self._cte_alias = _cte_alias
        # Keep recursivity reference with union/union_all
        self._restates = _restates
        if _prefixes:
            self._prefixes = _prefixes
        if _suffixes:
            self._suffixes = _suffixes
        super()._init(selectable, name=name)

    def _populate_column_collection(
        self,
        columns: ColumnCollection[str, KeyedColumnElement[Any]],
        primary_key: ColumnSet,
        foreign_keys: Set[KeyedColumnElement[Any]],
    ) -> None:
        if self._cte_alias is not None:
            self._cte_alias._generate_fromclause_column_proxies(
                self,
                columns,
                primary_key=primary_key,
                foreign_keys=foreign_keys,
            )
        else:
            self.element._generate_fromclause_column_proxies(
                self,
                columns,
                primary_key=primary_key,
                foreign_keys=foreign_keys,
            )

    def alias(self, name: Optional[str] = None, flat: bool = False) -> CTE:
        """Return an :class:`_expression.Alias` of this
        :class:`_expression.CTE`.

        This method is a CTE-specific specialization of the
        :meth:`_expression.FromClause.alias` method.

        .. seealso::

            :ref:`tutorial_using_aliases`

            :func:`_expression.alias`

        """
        return CTE._construct(
            self.element,
            name=name,
            recursive=self.recursive,
            nesting=self.nesting,
            _cte_alias=self,
            _prefixes=self._prefixes,
            _suffixes=self._suffixes,
        )

    def union(self, *other: _SelectStatementForCompoundArgument[Any]) -> CTE:
        r"""Return a new :class:`_expression.CTE` with a SQL ``UNION``
        of the original CTE against the given selectables provided
        as positional arguments.

        :param \*other: one or more elements with which to create a
         UNION.

         .. versionchanged:: 1.4.28 multiple elements are now accepted.

        .. seealso::

            :meth:`_sql.HasCTE.cte` - examples of calling styles

        """
        assert is_select_statement(
            self.element
        ), f"CTE element f{self.element} does not support union()"

        return CTE._construct(
            self.element.union(*other),
            name=self.name,
            recursive=self.recursive,
            nesting=self.nesting,
            _restates=self,
            _prefixes=self._prefixes,
            _suffixes=self._suffixes,
        )

    def union_all(
        self, *other: _SelectStatementForCompoundArgument[Any]
    ) -> CTE:
        r"""Return a new :class:`_expression.CTE` with a SQL ``UNION ALL``
        of the original CTE against the given selectables provided
        as positional arguments.

        :param \*other: one or more elements with which to create a
         UNION.

         .. versionchanged:: 1.4.28 multiple elements are now accepted.

        .. seealso::

            :meth:`_sql.HasCTE.cte` - examples of calling styles

        """

        assert is_select_statement(
            self.element
        ), f"CTE element f{self.element} does not support union_all()"

        return CTE._construct(
            self.element.union_all(*other),
            name=self.name,
            recursive=self.recursive,
            nesting=self.nesting,
            _restates=self,
            _prefixes=self._prefixes,
            _suffixes=self._suffixes,
        )

    def _get_reference_cte(self) -> CTE:
        """
        A recursive CTE is updated to attach the recursive part.
        Updated CTEs should still refer to the original CTE.
        This function returns this reference identifier.
        """
        return self._restates if self._restates is not None else self


class _CTEOpts(NamedTuple):
    nesting: bool


class _ColumnsPlusNames(NamedTuple):
    required_label_name: Optional[str]
    """
    string label name, if non-None, must be rendered as a
    label, i.e. "AS <name>"
    """

    proxy_key: Optional[str]
    """
    proxy_key that is to be part of the result map for this
    col.  this is also the key in a fromclause.c or
    select.selected_columns collection
    """

    fallback_label_name: Optional[str]
    """
    name that can be used to render an "AS <name>" when
    we have to render a label even though
    required_label_name was not given
    """

    column: Union[ColumnElement[Any], TextClause]
    """
    the ColumnElement itself
    """

    repeated: bool
    """
    True if this is a duplicate of a previous column
    in the list of columns
    """


class SelectsRows(ReturnsRows):
    """Sub-base of ReturnsRows for elements that deliver rows
    directly, namely SELECT and INSERT/UPDATE/DELETE..RETURNING"""

    _label_style: SelectLabelStyle = LABEL_STYLE_NONE

    def _generate_columns_plus_names(
        self,
        anon_for_dupe_key: bool,
        cols: Optional[_SelectIterable] = None,
    ) -> List[_ColumnsPlusNames]:
        """Generate column names as rendered in a SELECT statement by
        the compiler.

        This is distinct from the _column_naming_convention generator that's
        intended for population of .c collections and similar, which has
        different rules.   the collection returned here calls upon the
        _column_naming_convention as well.

        """

        if cols is None:
            cols = self._all_selected_columns

        key_naming_convention = SelectState._column_naming_convention(
            self._label_style
        )

        names = {}

        result: List[_ColumnsPlusNames] = []
        result_append = result.append

        table_qualified = self._label_style is LABEL_STYLE_TABLENAME_PLUS_COL
        label_style_none = self._label_style is LABEL_STYLE_NONE

        # a counter used for "dedupe" labels, which have double underscores
        # in them and are never referred by name; they only act
        # as positional placeholders.  they need only be unique within
        # the single columns clause they're rendered within (required by
        # some dbs such as mysql).  So their anon identity is tracked against
        # a fixed counter rather than hash() identity.
        dedupe_hash = 1

        for c in cols:
            repeated = False

            if not c._render_label_in_columns_clause:
                effective_name = required_label_name = fallback_label_name = (
                    None
                )
            elif label_style_none:
                if TYPE_CHECKING:
                    assert is_column_element(c)

                effective_name = required_label_name = None
                fallback_label_name = c._non_anon_label or c._anon_name_label
            else:
                if TYPE_CHECKING:
                    assert is_column_element(c)

                if table_qualified:
                    required_label_name = effective_name = (
                        fallback_label_name
                    ) = c._tq_label
                else:
                    effective_name = fallback_label_name = c._non_anon_label
                    required_label_name = None

                if effective_name is None:
                    # it seems like this could be _proxy_key and we would
                    # not need _expression_label but it isn't
                    # giving us a clue when to use anon_label instead
                    expr_label = c._expression_label
                    if expr_label is None:
                        repeated = c._anon_name_label in names
                        names[c._anon_name_label] = c
                        effective_name = required_label_name = None

                        if repeated:
                            # here, "required_label_name" is sent as
                            # "None" and "fallback_label_name" is sent.
                            if table_qualified:
                                fallback_label_name = (
                                    c._dedupe_anon_tq_label_idx(dedupe_hash)
                                )
                                dedupe_hash += 1
                            else:
                                fallback_label_name = c._dedupe_anon_label_idx(
                                    dedupe_hash
                                )
                                dedupe_hash += 1
                        else:
                            fallback_label_name = c._anon_name_label
                    else:
                        required_label_name = effective_name = (
                            fallback_label_name
                        ) = expr_label

            if effective_name is not None:
                if TYPE_CHECKING:
                    assert is_column_element(c)

                if effective_name in names:
                    # when looking to see if names[name] is the same column as
                    # c, use hash(), so that an annotated version of the column
                    # is seen as the same as the non-annotated
                    if hash(names[effective_name]) != hash(c):
                        # different column under the same name.  apply
                        # disambiguating label
                        if table_qualified:
                            required_label_name = fallback_label_name = (
                                c._anon_tq_label
                            )
                        else:
                            required_label_name = fallback_label_name = (
                                c._anon_name_label
                            )

                        if anon_for_dupe_key and required_label_name in names:
                            # here, c._anon_tq_label is definitely unique to
                            # that column identity (or annotated version), so
                            # this should always be true.
                            # this is also an infrequent codepath because
                            # you need two levels of duplication to be here
                            assert hash(names[required_label_name]) == hash(c)

                            # the column under the disambiguating label is
                            # already present.  apply the "dedupe" label to
                            # subsequent occurrences of the column so that the
                            # original stays non-ambiguous
                            if table_qualified:
                                required_label_name = fallback_label_name = (
                                    c._dedupe_anon_tq_label_idx(dedupe_hash)
                                )
                                dedupe_hash += 1
                            else:
                                required_label_name = fallback_label_name = (
                                    c._dedupe_anon_label_idx(dedupe_hash)
                                )
                                dedupe_hash += 1
                            repeated = True
                        else:
                            names[required_label_name] = c
                    elif anon_for_dupe_key:
                        # same column under the same name. apply the "dedupe"
                        # label so that the original stays non-ambiguous
                        if table_qualified:
                            required_label_name = fallback_label_name = (
                                c._dedupe_anon_tq_label_idx(dedupe_hash)
                            )
                            dedupe_hash += 1
                        else:
                            required_label_name = fallback_label_name = (
                                c._dedupe_anon_label_idx(dedupe_hash)
                            )
                            dedupe_hash += 1
                        repeated = True
                else:
                    names[effective_name] = c

            result_append(
                _ColumnsPlusNames(
                    required_label_name,
                    key_naming_convention(c),
                    fallback_label_name,
                    c,
                    repeated,
                )
            )

        return result


class HasCTE(roles.HasCTERole, SelectsRows):
    """Mixin that declares a class to include CTE support."""

    _has_ctes_traverse_internals: _TraverseInternalsType = [
        ("_independent_ctes", InternalTraversal.dp_clauseelement_list),
        ("_independent_ctes_opts", InternalTraversal.dp_plain_obj),
    ]

    _independent_ctes: Tuple[CTE, ...] = ()
    _independent_ctes_opts: Tuple[_CTEOpts, ...] = ()

    @_generative
    def add_cte(self, *ctes: CTE, nest_here: bool = False) -> Self:
        r"""Add one or more :class:`_sql.CTE` constructs to this statement.

        This method will associate the given :class:`_sql.CTE` constructs with
        the parent statement such that they will each be unconditionally
        rendered in the WITH clause of the final statement, even if not
        referenced elsewhere within the statement or any sub-selects.

        The optional :paramref:`.HasCTE.add_cte.nest_here` parameter when set
        to True will have the effect that each given :class:`_sql.CTE` will
        render in a WITH clause rendered directly along with this statement,
        rather than being moved to the top of the ultimate rendered statement,
        even if this statement is rendered as a subquery within a larger
        statement.

        This method has two general uses. One is to embed CTE statements that
        serve some purpose without being referenced explicitly, such as the use
        case of embedding a DML statement such as an INSERT or UPDATE as a CTE
        inline with a primary statement that may draw from its results
        indirectly.  The other is to provide control over the exact placement
        of a particular series of CTE constructs that should remain rendered
        directly in terms of a particular statement that may be nested in a
        larger statement.

        E.g.::

            from sqlalchemy import table, column, select

            t = table("t", column("c1"), column("c2"))

            ins = t.insert().values({"c1": "x", "c2": "y"}).cte()

            stmt = select(t).add_cte(ins)

        Would render:

        .. sourcecode:: sql

            WITH anon_1 AS (
                INSERT INTO t (c1, c2) VALUES (:param_1, :param_2)
            )
            SELECT t.c1, t.c2
            FROM t

        Above, the "anon_1" CTE is not referenced in the SELECT
        statement, however still accomplishes the task of running an INSERT
        statement.

        Similarly in a DML-related context, using the PostgreSQL
        :class:`_postgresql.Insert` construct to generate an "upsert"::

            from sqlalchemy import table, column
            from sqlalchemy.dialects.postgresql import insert

            t = table("t", column("c1"), column("c2"))

            delete_statement_cte = t.delete().where(t.c.c1 < 1).cte("deletions")

            insert_stmt = insert(t).values({"c1": 1, "c2": 2})
            update_statement = insert_stmt.on_conflict_do_update(
                index_elements=[t.c.c1],
                set_={
                    "c1": insert_stmt.excluded.c1,
                    "c2": insert_stmt.excluded.c2,
                },
            ).add_cte(delete_statement_cte)

            print(update_statement)

        The above statement renders as:

        .. sourcecode:: sql

            WITH deletions AS (
                DELETE FROM t WHERE t.c1 < %(c1_1)s
            )
            INSERT INTO t (c1, c2) VALUES (%(c1)s, %(c2)s)
            ON CONFLICT (c1) DO UPDATE SET c1 = excluded.c1, c2 = excluded.c2

        .. versionadded:: 1.4.21

        :param \*ctes: zero or more :class:`.CTE` constructs.

         .. versionchanged:: 2.0  Multiple CTE instances are accepted

        :param nest_here: if True, the given CTE or CTEs will be rendered
         as though they specified the :paramref:`.HasCTE.cte.nesting` flag
         to ``True`` when they were added to this :class:`.HasCTE`.
         Assuming the given CTEs are not referenced in an outer-enclosing
         statement as well, the CTEs given should render at the level of
         this statement when this flag is given.

         .. versionadded:: 2.0

         .. seealso::

            :paramref:`.HasCTE.cte.nesting`


        """  # noqa: E501
        opt = _CTEOpts(nest_here)
        for cte in ctes:
            cte = coercions.expect(roles.IsCTERole, cte)
            self._independent_ctes += (cte,)
            self._independent_ctes_opts += (opt,)
        return self

    def cte(
        self,
        name: Optional[str] = None,
        recursive: bool = False,
        nesting: bool = False,
    ) -> CTE:
        r"""Return a new :class:`_expression.CTE`,
        or Common Table Expression instance.

        Common table expressions are a SQL standard whereby SELECT
        statements can draw upon secondary statements specified along
        with the primary statement, using a clause called "WITH".
        Special semantics regarding UNION can also be employed to
        allow "recursive" queries, where a SELECT statement can draw
        upon the set of rows that have previously been selected.

        CTEs can also be applied to DML constructs UPDATE, INSERT
        and DELETE on some databases, both as a source of CTE rows
        when combined with RETURNING, as well as a consumer of
        CTE rows.

        SQLAlchemy detects :class:`_expression.CTE` objects, which are treated
        similarly to :class:`_expression.Alias` objects, as special elements
        to be delivered to the FROM clause of the statement as well
        as to a WITH clause at the top of the statement.

        For special prefixes such as PostgreSQL "MATERIALIZED" and
        "NOT MATERIALIZED", the :meth:`_expression.CTE.prefix_with`
        method may be
        used to establish these.

        .. versionchanged:: 1.3.13 Added support for prefixes.
           In particular - MATERIALIZED and NOT MATERIALIZED.

        :param name: name given to the common table expression.  Like
         :meth:`_expression.FromClause.alias`, the name can be left as
         ``None`` in which case an anonymous symbol will be used at query
         compile time.
        :param recursive: if ``True``, will render ``WITH RECURSIVE``.
         A recursive common table expression is intended to be used in
         conjunction with UNION ALL in order to derive rows
         from those already selected.
        :param nesting: if ``True``, will render the CTE locally to the
         statement in which it is referenced.   For more complex scenarios,
         the :meth:`.HasCTE.add_cte` method using the
         :paramref:`.HasCTE.add_cte.nest_here`
         parameter may also be used to more carefully
         control the exact placement of a particular CTE.

         .. versionadded:: 1.4.24

         .. seealso::

            :meth:`.HasCTE.add_cte`

        The following examples include two from PostgreSQL's documentation at
        https://www.postgresql.org/docs/current/static/queries-with.html,
        as well as additional examples.

        Example 1, non recursive::

            from sqlalchemy import (
                Table,
                Column,
                String,
                Integer,
                MetaData,
                select,
                func,
            )

            metadata = MetaData()

            orders = Table(
                "orders",
                metadata,
                Column("region", String),
                Column("amount", Integer),
                Column("product", String),
                Column("quantity", Integer),
            )

            regional_sales = (
                select(orders.c.region, func.sum(orders.c.amount).label("total_sales"))
                .group_by(orders.c.region)
                .cte("regional_sales")
            )


            top_regions = (
                select(regional_sales.c.region)
                .where(
                    regional_sales.c.total_sales
                    > select(func.sum(regional_sales.c.total_sales) / 10)
                )
                .cte("top_regions")
            )

            statement = (
                select(
                    orders.c.region,
                    orders.c.product,
                    func.sum(orders.c.quantity).label("product_units"),
                    func.sum(orders.c.amount).label("product_sales"),
                )
                .where(orders.c.region.in_(select(top_regions.c.region)))
                .group_by(orders.c.region, orders.c.product)
            )

            result = conn.execute(statement).fetchall()

        Example 2, WITH RECURSIVE::

            from sqlalchemy import (
                Table,
                Column,
                String,
                Integer,
                MetaData,
                select,
                func,
            )

            metadata = MetaData()

            parts = Table(
                "parts",
                metadata,
                Column("part", String),
                Column("sub_part", String),
                Column("quantity", Integer),
            )

            included_parts = (
                select(parts.c.sub_part, parts.c.part, parts.c.quantity)
                .where(parts.c.part == "our part")
                .cte(recursive=True)
            )


            incl_alias = included_parts.alias()
            parts_alias = parts.alias()
            included_parts = included_parts.union_all(
                select(
                    parts_alias.c.sub_part, parts_alias.c.part, parts_alias.c.quantity
                ).where(parts_alias.c.part == incl_alias.c.sub_part)
            )

            statement = select(
                included_parts.c.sub_part,
                func.sum(included_parts.c.quantity).label("total_quantity"),
            ).group_by(included_parts.c.sub_part)

            result = conn.execute(statement).fetchall()

        Example 3, an upsert using UPDATE and INSERT with CTEs::

            from datetime import date
            from sqlalchemy import (
                MetaData,
                Table,
                Column,
                Integer,
                Date,
                select,
                literal,
                and_,
                exists,
            )

            metadata = MetaData()

            visitors = Table(
                "visitors",
                metadata,
                Column("product_id", Integer, primary_key=True),
                Column("date", Date, primary_key=True),
                Column("count", Integer),
            )

            # add 5 visitors for the product_id == 1
            product_id = 1
            day = date.today()
            count = 5

            update_cte = (
                visitors.update()
                .where(
                    and_(visitors.c.product_id == product_id, visitors.c.date == day)
                )
                .values(count=visitors.c.count + count)
                .returning(literal(1))
                .cte("update_cte")
            )

            upsert = visitors.insert().from_select(
                [visitors.c.product_id, visitors.c.date, visitors.c.count],
                select(literal(product_id), literal(day), literal(count)).where(
                    ~exists(update_cte.select())
                ),
            )

            connection.execute(upsert)

        Example 4, Nesting CTE (SQLAlchemy 1.4.24 and above)::

            value_a = select(literal("root").label("n")).cte("value_a")

            # A nested CTE with the same name as the root one
            value_a_nested = select(literal("nesting").label("n")).cte(
                "value_a", nesting=True
            )

            # Nesting CTEs takes ascendency locally
            # over the CTEs at a higher level
            value_b = select(value_a_nested.c.n).cte("value_b")

            value_ab = select(value_a.c.n.label("a"), value_b.c.n.label("b"))

        The above query will render the second CTE nested inside the first,
        shown with inline parameters below as:

        .. sourcecode:: sql

            WITH
                value_a AS
                    (SELECT 'root' AS n),
                value_b AS
                    (WITH value_a AS
                        (SELECT 'nesting' AS n)
                    SELECT value_a.n AS n FROM value_a)
            SELECT value_a.n AS a, value_b.n AS b
            FROM value_a, value_b

        The same CTE can be set up using the :meth:`.HasCTE.add_cte` method
        as follows (SQLAlchemy 2.0 and above)::

            value_a = select(literal("root").label("n")).cte("value_a")

            # A nested CTE with the same name as the root one
            value_a_nested = select(literal("nesting").label("n")).cte("value_a")

            # Nesting CTEs takes ascendency locally
            # over the CTEs at a higher level
            value_b = (
                select(value_a_nested.c.n)
                .add_cte(value_a_nested, nest_here=True)
                .cte("value_b")
            )

            value_ab = select(value_a.c.n.label("a"), value_b.c.n.label("b"))

        Example 5, Non-Linear CTE (SQLAlchemy 1.4.28 and above)::

            edge = Table(
                "edge",
                metadata,
                Column("id", Integer, primary_key=True),
                Column("left", Integer),
                Column("right", Integer),
            )

            root_node = select(literal(1).label("node")).cte("nodes", recursive=True)

            left_edge = select(edge.c.left).join(
                root_node, edge.c.right == root_node.c.node
            )
            right_edge = select(edge.c.right).join(
                root_node, edge.c.left == root_node.c.node
            )

            subgraph_cte = root_node.union(left_edge, right_edge)

            subgraph = select(subgraph_cte)

        The above query will render 2 UNIONs inside the recursive CTE:

        .. sourcecode:: sql

            WITH RECURSIVE nodes(node) AS (
                    SELECT 1 AS node
                UNION
                    SELECT edge."left" AS "left"
                    FROM edge JOIN nodes ON edge."right" = nodes.node
                UNION
                    SELECT edge."right" AS "right"
                    FROM edge JOIN nodes ON edge."left" = nodes.node
            )
            SELECT nodes.node FROM nodes

        .. seealso::

            :meth:`_orm.Query.cte` - ORM version of
            :meth:`_expression.HasCTE.cte`.

        """  # noqa: E501
        return CTE._construct(
            self, name=name, recursive=recursive, nesting=nesting
        )


class Subquery(AliasedReturnsRows):
    """Represent a subquery of a SELECT.

    A :class:`.Subquery` is created by invoking the
    :meth:`_expression.SelectBase.subquery` method, or for convenience the
    :meth:`_expression.SelectBase.alias` method, on any
    :class:`_expression.SelectBase` subclass
    which includes :class:`_expression.Select`,
    :class:`_expression.CompoundSelect`, and
    :class:`_expression.TextualSelect`.  As rendered in a FROM clause,
    it represents the
    body of the SELECT statement inside of parenthesis, followed by the usual
    "AS <somename>" that defines all "alias" objects.

    The :class:`.Subquery` object is very similar to the
    :class:`_expression.Alias`
    object and can be used in an equivalent way.    The difference between
    :class:`_expression.Alias` and :class:`.Subquery` is that
    :class:`_expression.Alias` always
    contains a :class:`_expression.FromClause` object whereas
    :class:`.Subquery`
    always contains a :class:`_expression.SelectBase` object.

    .. versionadded:: 1.4 The :class:`.Subquery` class was added which now
       serves the purpose of providing an aliased version of a SELECT
       statement.

    """

    __visit_name__ = "subquery"

    _is_subquery = True

    inherit_cache = True

    element: SelectBase

    @classmethod
    def _factory(
        cls, selectable: SelectBase, name: Optional[str] = None
    ) -> Subquery:
        """Return a :class:`.Subquery` object."""

        return coercions.expect(
            roles.SelectStatementRole, selectable
        ).subquery(name=name)

    @util.deprecated(
        "1.4",
        "The :meth:`.Subquery.as_scalar` method, which was previously "
        "``Alias.as_scalar()`` prior to version 1.4, is deprecated and "
        "will be removed in a future release; Please use the "
        ":meth:`_expression.Select.scalar_subquery` method of the "
        ":func:`_expression.select` "
        "construct before constructing a subquery object, or with the ORM "
        "use the :meth:`_query.Query.scalar_subquery` method.",
    )
    def as_scalar(self) -> ScalarSelect[Any]:
        return self.element.set_label_style(LABEL_STYLE_NONE).scalar_subquery()


class FromGrouping(GroupedElement, FromClause):
    """Represent a grouping of a FROM clause"""

    _traverse_internals: _TraverseInternalsType = [
        ("element", InternalTraversal.dp_clauseelement)
    ]

    element: FromClause

    def __init__(self, element: FromClause):
        self.element = coercions.expect(roles.FromClauseRole, element)

    @util.ro_non_memoized_property
    def columns(
        self,
    ) -> ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]:
        return self.element.columns

    @util.ro_non_memoized_property
    def c(self) -> ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]:
        return self.element.columns

    @property
    def primary_key(self) -> Iterable[NamedColumn[Any]]:
        return self.element.primary_key

    @property
    def foreign_keys(self) -> Iterable[ForeignKey]:
        return self.element.foreign_keys

    def is_derived_from(self, fromclause: Optional[FromClause]) -> bool:
        return self.element.is_derived_from(fromclause)

    def alias(
        self, name: Optional[str] = None, flat: bool = False
    ) -> NamedFromGrouping:
        return NamedFromGrouping(self.element.alias(name=name, flat=flat))

    def _anonymous_fromclause(self, **kw: Any) -> FromGrouping:
        return FromGrouping(self.element._anonymous_fromclause(**kw))

    @util.ro_non_memoized_property
    def _hide_froms(self) -> Iterable[FromClause]:
        return self.element._hide_froms

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return self.element._from_objects

    def __getstate__(self) -> Dict[str, FromClause]:
        return {"element": self.element}

    def __setstate__(self, state: Dict[str, FromClause]) -> None:
        self.element = state["element"]

    if TYPE_CHECKING:

        def self_group(
            self, against: Optional[OperatorType] = None
        ) -> Self: ...


class NamedFromGrouping(FromGrouping, NamedFromClause):
    """represent a grouping of a named FROM clause

    .. versionadded:: 2.0

    """

    inherit_cache = True

    if TYPE_CHECKING:

        def self_group(
            self, against: Optional[OperatorType] = None
        ) -> Self: ...


class TableClause(roles.DMLTableRole, Immutable, NamedFromClause):
    """Represents a minimal "table" construct.

    This is a lightweight table object that has only a name, a
    collection of columns, which are typically produced
    by the :func:`_expression.column` function, and a schema::

        from sqlalchemy import table, column

        user = table(
            "user",
            column("id"),
            column("name"),
            column("description"),
        )

    The :class:`_expression.TableClause` construct serves as the base for
    the more commonly used :class:`_schema.Table` object, providing
    the usual set of :class:`_expression.FromClause` services including
    the ``.c.`` collection and statement generation methods.

    It does **not** provide all the additional schema-level services
    of :class:`_schema.Table`, including constraints, references to other
    tables, or support for :class:`_schema.MetaData`-level services.
    It's useful
    on its own as an ad-hoc construct used to generate quick SQL
    statements when a more fully fledged :class:`_schema.Table`
    is not on hand.

    """

    __visit_name__ = "table"

    _traverse_internals: _TraverseInternalsType = [
        (
            "columns",
            InternalTraversal.dp_fromclause_canonical_column_collection,
        ),
        ("name", InternalTraversal.dp_string),
        ("schema", InternalTraversal.dp_string),
    ]

    _is_table = True

    fullname: str

    implicit_returning = False
    """:class:`_expression.TableClause`
    doesn't support having a primary key or column
    -level defaults, so implicit returning doesn't apply."""

    @util.ro_memoized_property
    def _autoincrement_column(self) -> Optional[ColumnClause[Any]]:
        """No PK or default support so no autoincrement column."""
        return None

    def __init__(self, name: str, *columns: ColumnClause[Any], **kw: Any):
        super().__init__()
        self.name = name
        self._columns = DedupeColumnCollection()
        self.primary_key = ColumnSet()  # type: ignore
        self.foreign_keys = set()  # type: ignore
        for c in columns:
            self.append_column(c)

        schema = kw.pop("schema", None)
        if schema is not None:
            self.schema = schema
        if self.schema is not None:
            self.fullname = "%s.%s" % (self.schema, self.name)
        else:
            self.fullname = self.name
        if kw:
            raise exc.ArgumentError("Unsupported argument(s): %s" % list(kw))

    if TYPE_CHECKING:

        @util.ro_non_memoized_property
        def columns(
            self,
        ) -> ReadOnlyColumnCollection[str, ColumnClause[Any]]: ...

        @util.ro_non_memoized_property
        def c(self) -> ReadOnlyColumnCollection[str, ColumnClause[Any]]: ...

    def __str__(self) -> str:
        if self.schema is not None:
            return self.schema + "." + self.name
        else:
            return self.name

    def _refresh_for_new_column(self, column: ColumnElement[Any]) -> None:
        pass

    @util.ro_memoized_property
    def description(self) -> str:
        return self.name

    def append_column(self, c: ColumnClause[Any]) -> None:
        existing = c.table
        if existing is not None and existing is not self:
            raise exc.ArgumentError(
                "column object '%s' already assigned to table '%s'"
                % (c.key, existing)
            )

        self._columns.add(c)
        c.table = self

    @util.preload_module("sqlalchemy.sql.dml")
    def insert(self) -> util.preloaded.sql_dml.Insert:
        """Generate an :class:`_sql.Insert` construct against this
        :class:`_expression.TableClause`.

        E.g.::

            table.insert().values(name="foo")

        See :func:`_expression.insert` for argument and usage information.

        """

        return util.preloaded.sql_dml.Insert(self)

    @util.preload_module("sqlalchemy.sql.dml")
    def update(self) -> Update:
        """Generate an :func:`_expression.update` construct against this
        :class:`_expression.TableClause`.

        E.g.::

            table.update().where(table.c.id == 7).values(name="foo")

        See :func:`_expression.update` for argument and usage information.

        """
        return util.preloaded.sql_dml.Update(
            self,
        )

    @util.preload_module("sqlalchemy.sql.dml")
    def delete(self) -> Delete:
        """Generate a :func:`_expression.delete` construct against this
        :class:`_expression.TableClause`.

        E.g.::

            table.delete().where(table.c.id == 7)

        See :func:`_expression.delete` for argument and usage information.

        """
        return util.preloaded.sql_dml.Delete(self)

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return [self]


ForUpdateParameter = Union["ForUpdateArg", None, bool, Dict[str, Any]]


class ForUpdateArg(ClauseElement):
    _traverse_internals: _TraverseInternalsType = [
        ("of", InternalTraversal.dp_clauseelement_list),
        ("nowait", InternalTraversal.dp_boolean),
        ("read", InternalTraversal.dp_boolean),
        ("skip_locked", InternalTraversal.dp_boolean),
        ("key_share", InternalTraversal.dp_boolean),
    ]

    of: Optional[Sequence[ClauseElement]]
    nowait: bool
    read: bool
    skip_locked: bool

    @classmethod
    def _from_argument(
        cls, with_for_update: ForUpdateParameter
    ) -> Optional[ForUpdateArg]:
        if isinstance(with_for_update, ForUpdateArg):
            return with_for_update
        elif with_for_update in (None, False):
            return None
        elif with_for_update is True:
            return ForUpdateArg()
        else:
            return ForUpdateArg(**cast("Dict[str, Any]", with_for_update))

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, ForUpdateArg)
            and other.nowait == self.nowait
            and other.read == self.read
            and other.skip_locked == self.skip_locked
            and other.key_share == self.key_share
            and other.of is self.of
        )

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return id(self)

    def __init__(
        self,
        *,
        nowait: bool = False,
        read: bool = False,
        of: Optional[_ForUpdateOfArgument] = None,
        skip_locked: bool = False,
        key_share: bool = False,
    ):
        """Represents arguments specified to
        :meth:`_expression.Select.for_update`.

        """

        self.nowait = nowait
        self.read = read
        self.skip_locked = skip_locked
        self.key_share = key_share
        if of is not None:
            self.of = [
                coercions.expect(roles.ColumnsClauseRole, elem)
                for elem in util.to_list(of)
            ]
        else:
            self.of = None


class Values(roles.InElementRole, Generative, LateralFromClause):
    """Represent a ``VALUES`` construct that can be used as a FROM element
    in a statement.

    The :class:`_expression.Values` object is created from the
    :func:`_expression.values` function.

    .. versionadded:: 1.4

    """

    __visit_name__ = "values"

    _data: Tuple[Sequence[Tuple[Any, ...]], ...] = ()

    _unnamed: bool
    _traverse_internals: _TraverseInternalsType = [
        ("_column_args", InternalTraversal.dp_clauseelement_list),
        ("_data", InternalTraversal.dp_dml_multi_values),
        ("name", InternalTraversal.dp_string),
        ("literal_binds", InternalTraversal.dp_boolean),
    ]

    def __init__(
        self,
        *columns: ColumnClause[Any],
        name: Optional[str] = None,
        literal_binds: bool = False,
    ):
        super().__init__()
        self._column_args = columns

        if name is None:
            self._unnamed = True
            self.name = _anonymous_label.safe_construct(id(self), "anon")
        else:
            self._unnamed = False
            self.name = name
        self.literal_binds = literal_binds
        self.named_with_column = not self._unnamed

    @property
    def _column_types(self) -> List[TypeEngine[Any]]:
        return [col.type for col in self._column_args]

    @_generative
    def alias(self, name: Optional[str] = None, flat: bool = False) -> Self:
        """Return a new :class:`_expression.Values`
        construct that is a copy of this
        one with the given name.

        This method is a VALUES-specific specialization of the
        :meth:`_expression.FromClause.alias` method.

        .. seealso::

            :ref:`tutorial_using_aliases`

            :func:`_expression.alias`

        """
        non_none_name: str

        if name is None:
            non_none_name = _anonymous_label.safe_construct(id(self), "anon")
        else:
            non_none_name = name

        self.name = non_none_name
        self.named_with_column = True
        self._unnamed = False
        return self

    @_generative
    def lateral(self, name: Optional[str] = None) -> LateralFromClause:
        """Return a new :class:`_expression.Values` with the lateral flag set,
        so that
        it renders as LATERAL.

        .. seealso::

            :func:`_expression.lateral`

        """
        non_none_name: str

        if name is None:
            non_none_name = self.name
        else:
            non_none_name = name

        self._is_lateral = True
        self.name = non_none_name
        self._unnamed = False
        return self

    @_generative
    def data(self, values: Sequence[Tuple[Any, ...]]) -> Self:
        """Return a new :class:`_expression.Values` construct,
        adding the given data to the data list.

        E.g.::

            my_values = my_values.data([(1, "value 1"), (2, "value2")])

        :param values: a sequence (i.e. list) of tuples that map to the
         column expressions given in the :class:`_expression.Values`
         constructor.

        """

        self._data += (values,)
        return self

    def scalar_values(self) -> ScalarValues:
        """Returns a scalar ``VALUES`` construct that can be used as a
        COLUMN element in a statement.

        .. versionadded:: 2.0.0b4

        """
        return ScalarValues(self._column_args, self._data, self.literal_binds)

    def _populate_column_collection(
        self,
        columns: ColumnCollection[str, KeyedColumnElement[Any]],
        primary_key: ColumnSet,
        foreign_keys: Set[KeyedColumnElement[Any]],
    ) -> None:
        for c in self._column_args:
            if c.table is not None and c.table is not self:
                _, c = c._make_proxy(
                    self, primary_key=primary_key, foreign_keys=foreign_keys
                )
            else:
                # if the column was used in other contexts, ensure
                # no memoizations of other FROM clauses.
                # see test_values.py -> test_auto_proxy_select_direct_col
                c._reset_memoizations()
            columns.add(c)
            c.table = self

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return [self]


class ScalarValues(roles.InElementRole, GroupedElement, ColumnElement[Any]):
    """Represent a scalar ``VALUES`` construct that can be used as a
    COLUMN element in a statement.

    The :class:`_expression.ScalarValues` object is created from the
    :meth:`_expression.Values.scalar_values` method. It's also
    automatically generated when a :class:`_expression.Values` is used in
    an ``IN`` or ``NOT IN`` condition.

    .. versionadded:: 2.0.0b4

    """

    __visit_name__ = "scalar_values"

    _traverse_internals: _TraverseInternalsType = [
        ("_column_args", InternalTraversal.dp_clauseelement_list),
        ("_data", InternalTraversal.dp_dml_multi_values),
        ("literal_binds", InternalTraversal.dp_boolean),
    ]

    def __init__(
        self,
        columns: Sequence[ColumnClause[Any]],
        data: Tuple[Sequence[Tuple[Any, ...]], ...],
        literal_binds: bool,
    ):
        super().__init__()
        self._column_args = columns
        self._data = data
        self.literal_binds = literal_binds

    @property
    def _column_types(self) -> List[TypeEngine[Any]]:
        return [col.type for col in self._column_args]

    def __clause_element__(self) -> ScalarValues:
        return self

    if TYPE_CHECKING:

        def self_group(
            self, against: Optional[OperatorType] = None
        ) -> Self: ...


class SelectBase(
    roles.SelectStatementRole,
    roles.DMLSelectRole,
    roles.CompoundElementRole,
    roles.InElementRole,
    HasCTE,
    SupportsCloneAnnotations,
    Selectable,
):
    """Base class for SELECT statements.


    This includes :class:`_expression.Select`,
    :class:`_expression.CompoundSelect` and
    :class:`_expression.TextualSelect`.


    """

    _is_select_base = True
    is_select = True

    _label_style: SelectLabelStyle = LABEL_STYLE_NONE

    def _refresh_for_new_column(self, column: ColumnElement[Any]) -> None:
        self._reset_memoizations()

    @util.ro_non_memoized_property
    def selected_columns(
        self,
    ) -> ColumnCollection[str, ColumnElement[Any]]:
        """A :class:`_expression.ColumnCollection`
        representing the columns that
        this SELECT statement or similar construct returns in its result set.

        This collection differs from the :attr:`_expression.FromClause.columns`
        collection of a :class:`_expression.FromClause` in that the columns
        within this collection cannot be directly nested inside another SELECT
        statement; a subquery must be applied first which provides for the
        necessary parenthesization required by SQL.

        .. note::

            The :attr:`_sql.SelectBase.selected_columns` collection does not
            include expressions established in the columns clause using the
            :func:`_sql.text` construct; these are silently omitted from the
            collection. To use plain textual column expressions inside of a
            :class:`_sql.Select` construct, use the :func:`_sql.literal_column`
            construct.

        .. seealso::

            :attr:`_sql.Select.selected_columns`

        .. versionadded:: 1.4

        """
        raise NotImplementedError()

    def _generate_fromclause_column_proxies(
        self,
        subquery: FromClause,
        columns: ColumnCollection[str, KeyedColumnElement[Any]],
        primary_key: ColumnSet,
        foreign_keys: Set[KeyedColumnElement[Any]],
        *,
        proxy_compound_columns: Optional[
            Iterable[Sequence[ColumnElement[Any]]]
        ] = None,
    ) -> None:
        raise NotImplementedError()

    @util.ro_non_memoized_property
    def _all_selected_columns(self) -> _SelectIterable:
        """A sequence of expressions that correspond to what is rendered
        in the columns clause, including :class:`_sql.TextClause`
        constructs.

        .. versionadded:: 1.4.12

        .. seealso::

            :attr:`_sql.SelectBase.exported_columns`

        """
        raise NotImplementedError()

    @property
    def exported_columns(
        self,
    ) -> ReadOnlyColumnCollection[str, ColumnElement[Any]]:
        """A :class:`_expression.ColumnCollection`
        that represents the "exported"
        columns of this :class:`_expression.Selectable`, not including
        :class:`_sql.TextClause` constructs.

        The "exported" columns for a :class:`_expression.SelectBase`
        object are synonymous
        with the :attr:`_expression.SelectBase.selected_columns` collection.

        .. versionadded:: 1.4

        .. seealso::

            :attr:`_expression.Select.exported_columns`

            :attr:`_expression.Selectable.exported_columns`

            :attr:`_expression.FromClause.exported_columns`


        """
        return self.selected_columns.as_readonly()

    @property
    @util.deprecated(
        "1.4",
        "The :attr:`_expression.SelectBase.c` and "
        ":attr:`_expression.SelectBase.columns` attributes "
        "are deprecated and will be removed in a future release; these "
        "attributes implicitly create a subquery that should be explicit.  "
        "Please call :meth:`_expression.SelectBase.subquery` "
        "first in order to create "
        "a subquery, which then contains this attribute.  To access the "
        "columns that this SELECT object SELECTs "
        "from, use the :attr:`_expression.SelectBase.selected_columns` "
        "attribute.",
    )
    def c(self) -> ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]:
        return self._implicit_subquery.columns

    @property
    def columns(
        self,
    ) -> ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]:
        return self.c

    def get_label_style(self) -> SelectLabelStyle:
        """
        Retrieve the current label style.

        Implemented by subclasses.

        """
        raise NotImplementedError()

    def set_label_style(self, style: SelectLabelStyle) -> Self:
        """Return a new selectable with the specified label style.

        Implemented by subclasses.

        """

        raise NotImplementedError()

    @util.deprecated(
        "1.4",
        "The :meth:`_expression.SelectBase.select` method is deprecated "
        "and will be removed in a future release; this method implicitly "
        "creates a subquery that should be explicit.  "
        "Please call :meth:`_expression.SelectBase.subquery` "
        "first in order to create "
        "a subquery, which then can be selected.",
    )
    def select(self, *arg: Any, **kw: Any) -> Select[Any]:
        return self._implicit_subquery.select(*arg, **kw)

    @HasMemoized.memoized_attribute
    def _implicit_subquery(self) -> Subquery:
        return self.subquery()

    def _scalar_type(self) -> TypeEngine[Any]:
        raise NotImplementedError()

    @util.deprecated(
        "1.4",
        "The :meth:`_expression.SelectBase.as_scalar` "
        "method is deprecated and will be "
        "removed in a future release.  Please refer to "
        ":meth:`_expression.SelectBase.scalar_subquery`.",
    )
    def as_scalar(self) -> ScalarSelect[Any]:
        return self.scalar_subquery()

    def exists(self) -> Exists:
        """Return an :class:`_sql.Exists` representation of this selectable,
        which can be used as a column expression.

        The returned object is an instance of :class:`_sql.Exists`.

        .. seealso::

            :func:`_sql.exists`

            :ref:`tutorial_exists` - in the :term:`2.0 style` tutorial.

        .. versionadded:: 1.4

        """
        return Exists(self)

    def scalar_subquery(self) -> ScalarSelect[Any]:
        """Return a 'scalar' representation of this selectable, which can be
        used as a column expression.

        The returned object is an instance of :class:`_sql.ScalarSelect`.

        Typically, a select statement which has only one column in its columns
        clause is eligible to be used as a scalar expression.  The scalar
        subquery can then be used in the WHERE clause or columns clause of
        an enclosing SELECT.

        Note that the scalar subquery differentiates from the FROM-level
        subquery that can be produced using the
        :meth:`_expression.SelectBase.subquery`
        method.

        .. versionchanged: 1.4 - the ``.as_scalar()`` method was renamed to
           :meth:`_expression.SelectBase.scalar_subquery`.

        .. seealso::

            :ref:`tutorial_scalar_subquery` - in the 2.0 tutorial

        """
        if self._label_style is not LABEL_STYLE_NONE:
            self = self.set_label_style(LABEL_STYLE_NONE)

        return ScalarSelect(self)

    def label(self, name: Optional[str]) -> Label[Any]:
        """Return a 'scalar' representation of this selectable, embedded as a
        subquery with a label.

        .. seealso::

            :meth:`_expression.SelectBase.scalar_subquery`.

        """
        return self.scalar_subquery().label(name)

    def lateral(self, name: Optional[str] = None) -> LateralFromClause:
        """Return a LATERAL alias of this :class:`_expression.Selectable`.

        The return value is the :class:`_expression.Lateral` construct also
        provided by the top-level :func:`_expression.lateral` function.

        .. seealso::

            :ref:`tutorial_lateral_correlation` -  overview of usage.

        """
        return Lateral._factory(self, name)

    def subquery(self, name: Optional[str] = None) -> Subquery:
        """Return a subquery of this :class:`_expression.SelectBase`.

        A subquery is from a SQL perspective a parenthesized, named
        construct that can be placed in the FROM clause of another
        SELECT statement.

        Given a SELECT statement such as::

            stmt = select(table.c.id, table.c.name)

        The above statement might look like:

        .. sourcecode:: sql

            SELECT table.id, table.name FROM table

        The subquery form by itself renders the same way, however when
        embedded into the FROM clause of another SELECT statement, it becomes
        a named sub-element::

            subq = stmt.subquery()
            new_stmt = select(subq)

        The above renders as:

        .. sourcecode:: sql

            SELECT anon_1.id, anon_1.name
            FROM (SELECT table.id, table.name FROM table) AS anon_1

        Historically, :meth:`_expression.SelectBase.subquery`
        is equivalent to calling
        the :meth:`_expression.FromClause.alias`
        method on a FROM object; however,
        as a :class:`_expression.SelectBase`
        object is not directly  FROM object,
        the :meth:`_expression.SelectBase.subquery`
        method provides clearer semantics.

        .. versionadded:: 1.4

        """

        return Subquery._construct(
            self._ensure_disambiguated_names(), name=name
        )

    def _ensure_disambiguated_names(self) -> Self:
        """Ensure that the names generated by this selectbase will be
        disambiguated in some way, if possible.

        """

        raise NotImplementedError()

    def alias(
        self, name: Optional[str] = None, flat: bool = False
    ) -> Subquery:
        """Return a named subquery against this
        :class:`_expression.SelectBase`.

        For a :class:`_expression.SelectBase` (as opposed to a
        :class:`_expression.FromClause`),
        this returns a :class:`.Subquery` object which behaves mostly the
        same as the :class:`_expression.Alias` object that is used with a
        :class:`_expression.FromClause`.

        .. versionchanged:: 1.4 The :meth:`_expression.SelectBase.alias`
           method is now
           a synonym for the :meth:`_expression.SelectBase.subquery` method.

        """
        return self.subquery(name=name)


_SB = TypeVar("_SB", bound=SelectBase)


class SelectStatementGrouping(GroupedElement, SelectBase, Generic[_SB]):
    """Represent a grouping of a :class:`_expression.SelectBase`.

    This differs from :class:`.Subquery` in that we are still
    an "inner" SELECT statement, this is strictly for grouping inside of
    compound selects.

    """

    __visit_name__ = "select_statement_grouping"
    _traverse_internals: _TraverseInternalsType = [
        ("element", InternalTraversal.dp_clauseelement)
    ] + SupportsCloneAnnotations._clone_annotations_traverse_internals

    _is_select_container = True

    element: _SB

    def __init__(self, element: _SB) -> None:
        self.element = cast(
            _SB, coercions.expect(roles.SelectStatementRole, element)
        )

    def _ensure_disambiguated_names(self) -> SelectStatementGrouping[_SB]:
        new_element = self.element._ensure_disambiguated_names()
        if new_element is not self.element:
            return SelectStatementGrouping(new_element)
        else:
            return self

    def get_label_style(self) -> SelectLabelStyle:
        return self.element.get_label_style()

    def set_label_style(
        self, label_style: SelectLabelStyle
    ) -> SelectStatementGrouping[_SB]:
        return SelectStatementGrouping(
            self.element.set_label_style(label_style)
        )

    @property
    def select_statement(self) -> _SB:
        return self.element

    def self_group(self, against: Optional[OperatorType] = None) -> Self:
        return self

    if TYPE_CHECKING:

        def _ungroup(self) -> _SB: ...

    # def _generate_columns_plus_names(
    #    self, anon_for_dupe_key: bool
    # ) -> List[Tuple[str, str, str, ColumnElement[Any], bool]]:
    #    return self.element._generate_columns_plus_names(anon_for_dupe_key)

    def _generate_fromclause_column_proxies(
        self,
        subquery: FromClause,
        columns: ColumnCollection[str, KeyedColumnElement[Any]],
        primary_key: ColumnSet,
        foreign_keys: Set[KeyedColumnElement[Any]],
        *,
        proxy_compound_columns: Optional[
            Iterable[Sequence[ColumnElement[Any]]]
        ] = None,
    ) -> None:
        self.element._generate_fromclause_column_proxies(
            subquery,
            columns,
            proxy_compound_columns=proxy_compound_columns,
            primary_key=primary_key,
            foreign_keys=foreign_keys,
        )

    @util.ro_non_memoized_property
    def _all_selected_columns(self) -> _SelectIterable:
        return self.element._all_selected_columns

    @util.ro_non_memoized_property
    def selected_columns(self) -> ColumnCollection[str, ColumnElement[Any]]:
        """A :class:`_expression.ColumnCollection`
        representing the columns that
        the embedded SELECT statement returns in its result set, not including
        :class:`_sql.TextClause` constructs.

        .. versionadded:: 1.4

        .. seealso::

            :attr:`_sql.Select.selected_columns`

        """
        return self.element.selected_columns

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return self.element._from_objects

    def add_cte(self, *ctes: CTE, nest_here: bool = False) -> Self:
        # SelectStatementGrouping not generative: has no attribute '_generate'
        raise NotImplementedError


class GenerativeSelect(DialectKWArgs, SelectBase, Generative):
    """Base class for SELECT statements where additional elements can be
    added.

    This serves as the base for :class:`_expression.Select` and
    :class:`_expression.CompoundSelect`
    where elements such as ORDER BY, GROUP BY can be added and column
    rendering can be controlled.  Compare to
    :class:`_expression.TextualSelect`, which,
    while it subclasses :class:`_expression.SelectBase`
    and is also a SELECT construct,
    represents a fixed textual string which cannot be altered at this level,
    only wrapped as a subquery.

    """

    _order_by_clauses: Tuple[ColumnElement[Any], ...] = ()
    _group_by_clauses: Tuple[ColumnElement[Any], ...] = ()
    _limit_clause: Optional[ColumnElement[Any]] = None
    _offset_clause: Optional[ColumnElement[Any]] = None
    _fetch_clause: Optional[ColumnElement[Any]] = None
    _fetch_clause_options: Optional[Dict[str, bool]] = None
    _for_update_arg: Optional[ForUpdateArg] = None

    def __init__(self, _label_style: SelectLabelStyle = LABEL_STYLE_DEFAULT):
        self._label_style = _label_style

    @_generative
    def with_for_update(
        self,
        *,
        nowait: bool = False,
        read: bool = False,
        of: Optional[_ForUpdateOfArgument] = None,
        skip_locked: bool = False,
        key_share: bool = False,
    ) -> Self:
        """Specify a ``FOR UPDATE`` clause for this
        :class:`_expression.GenerativeSelect`.

        E.g.::

            stmt = select(table).with_for_update(nowait=True)

        On a database like PostgreSQL or Oracle Database, the above would
        render a statement like:

        .. sourcecode:: sql

            SELECT table.a, table.b FROM table FOR UPDATE NOWAIT

        on other backends, the ``nowait`` option is ignored and instead
        would produce:

        .. sourcecode:: sql

            SELECT table.a, table.b FROM table FOR UPDATE

        When called with no arguments, the statement will render with
        the suffix ``FOR UPDATE``.   Additional arguments can then be
        provided which allow for common database-specific
        variants.

        :param nowait: boolean; will render ``FOR UPDATE NOWAIT`` on Oracle
         Database and PostgreSQL dialects.

        :param read: boolean; will render ``LOCK IN SHARE MODE`` on MySQL,
         ``FOR SHARE`` on PostgreSQL.  On PostgreSQL, when combined with
         ``nowait``, will render ``FOR SHARE NOWAIT``.

        :param of: SQL expression or list of SQL expression elements,
         (typically :class:`_schema.Column` objects or a compatible expression,
         for some backends may also be a table expression) which will render
         into a ``FOR UPDATE OF`` clause; supported by PostgreSQL, Oracle
         Database, some MySQL versions and possibly others. May render as a
         table or as a column depending on backend.

        :param skip_locked: boolean, will render ``FOR UPDATE SKIP LOCKED`` on
         Oracle Database and PostgreSQL dialects or ``FOR SHARE SKIP LOCKED``
         if ``read=True`` is also specified.

        :param key_share: boolean, will render ``FOR NO KEY UPDATE``,
         or if combined with ``read=True`` will render ``FOR KEY SHARE``,
         on the PostgreSQL dialect.

        """
        self._for_update_arg = ForUpdateArg(
            nowait=nowait,
            read=read,
            of=of,
            skip_locked=skip_locked,
            key_share=key_share,
        )
        return self

    def get_label_style(self) -> SelectLabelStyle:
        """
        Retrieve the current label style.

        .. versionadded:: 1.4

        """
        return self._label_style

    def set_label_style(self, style: SelectLabelStyle) -> Self:
        """Return a new selectable with the specified label style.

        There are three "label styles" available,
        :attr:`_sql.SelectLabelStyle.LABEL_STYLE_DISAMBIGUATE_ONLY`,
        :attr:`_sql.SelectLabelStyle.LABEL_STYLE_TABLENAME_PLUS_COL`, and
        :attr:`_sql.SelectLabelStyle.LABEL_STYLE_NONE`.   The default style is
        :attr:`_sql.SelectLabelStyle.LABEL_STYLE_DISAMBIGUATE_ONLY`.

        In modern SQLAlchemy, there is not generally a need to change the
        labeling style, as per-expression labels are more effectively used by
        making use of the :meth:`_sql.ColumnElement.label` method. In past
        versions, :data:`_sql.LABEL_STYLE_TABLENAME_PLUS_COL` was used to
        disambiguate same-named columns from different tables, aliases, or
        subqueries; the newer :data:`_sql.LABEL_STYLE_DISAMBIGUATE_ONLY` now
        applies labels only to names that conflict with an existing name so
        that the impact of this labeling is minimal.

        The rationale for disambiguation is mostly so that all column
        expressions are available from a given :attr:`_sql.FromClause.c`
        collection when a subquery is created.

        .. versionadded:: 1.4 - the
            :meth:`_sql.GenerativeSelect.set_label_style` method replaces the
            previous combination of ``.apply_labels()``, ``.with_labels()`` and
            ``use_labels=True`` methods and/or parameters.

        .. seealso::

            :data:`_sql.LABEL_STYLE_DISAMBIGUATE_ONLY`

            :data:`_sql.LABEL_STYLE_TABLENAME_PLUS_COL`

            :data:`_sql.LABEL_STYLE_NONE`

            :data:`_sql.LABEL_STYLE_DEFAULT`

        """
        if self._label_style is not style:
            self = self._generate()
            self._label_style = style
        return self

    @property
    def _group_by_clause(self) -> ClauseList:
        """ClauseList access to group_by_clauses for legacy dialects"""
        return ClauseList._construct_raw(
            operators.comma_op, self._group_by_clauses
        )

    @property
    def _order_by_clause(self) -> ClauseList:
        """ClauseList access to order_by_clauses for legacy dialects"""
        return ClauseList._construct_raw(
            operators.comma_op, self._order_by_clauses
        )

    def _offset_or_limit_clause(
        self,
        element: _LimitOffsetType,
        name: Optional[str] = None,
        type_: Optional[_TypeEngineArgument[int]] = None,
    ) -> ColumnElement[Any]:
        """Convert the given value to an "offset or limit" clause.

        This handles incoming integers and converts to an expression; if
        an expression is already given, it is passed through.

        """
        return coercions.expect(
            roles.LimitOffsetRole, element, name=name, type_=type_
        )

    @overload
    def _offset_or_limit_clause_asint(
        self, clause: ColumnElement[Any], attrname: str
    ) -> NoReturn: ...

    @overload
    def _offset_or_limit_clause_asint(
        self, clause: Optional[_OffsetLimitParam], attrname: str
    ) -> Optional[int]: ...

    def _offset_or_limit_clause_asint(
        self, clause: Optional[ColumnElement[Any]], attrname: str
    ) -> Union[NoReturn, Optional[int]]:
        """Convert the "offset or limit" clause of a select construct to an
        integer.

        This is only possible if the value is stored as a simple bound
        parameter. Otherwise, a compilation error is raised.

        """
        if clause is None:
            return None
        try:
            value = clause._limit_offset_value
        except AttributeError as err:
            raise exc.CompileError(
                "This SELECT structure does not use a simple "
                "integer value for %s" % attrname
            ) from err
        else:
            return util.asint(value)

    @property
    def _limit(self) -> Optional[int]:
        """Get an integer value for the limit.  This should only be used
        by code that cannot support a limit as a BindParameter or
        other custom clause as it will throw an exception if the limit
        isn't currently set to an integer.

        """
        return self._offset_or_limit_clause_asint(self._limit_clause, "limit")

    def _simple_int_clause(self, clause: ClauseElement) -> bool:
        """True if the clause is a simple integer, False
        if it is not present or is a SQL expression.
        """
        return isinstance(clause, _OffsetLimitParam)

    @property
    def _offset(self) -> Optional[int]:
        """Get an integer value for the offset.  This should only be used
        by code that cannot support an offset as a BindParameter or
        other custom clause as it will throw an exception if the
        offset isn't currently set to an integer.

        """
        return self._offset_or_limit_clause_asint(
            self._offset_clause, "offset"
        )

    @property
    def _has_row_limiting_clause(self) -> bool:
        return (
            self._limit_clause is not None
            or self._offset_clause is not None
            or self._fetch_clause is not None
        )

    @_generative
    def limit(self, limit: _LimitOffsetType) -> Self:
        """Return a new selectable with the given LIMIT criterion
        applied.

        This is a numerical value which usually renders as a ``LIMIT``
        expression in the resulting select.  Backends that don't
        support ``LIMIT`` will attempt to provide similar
        functionality.

        .. note::

           The :meth:`_sql.GenerativeSelect.limit` method will replace
           any clause applied with :meth:`_sql.GenerativeSelect.fetch`.

        :param limit: an integer LIMIT parameter, or a SQL expression
         that provides an integer result. Pass ``None`` to reset it.

        .. seealso::

           :meth:`_sql.GenerativeSelect.fetch`

           :meth:`_sql.GenerativeSelect.offset`

        """

        self._fetch_clause = self._fetch_clause_options = None
        self._limit_clause = self._offset_or_limit_clause(limit)
        return self

    @_generative
    def fetch(
        self,
        count: _LimitOffsetType,
        with_ties: bool = False,
        percent: bool = False,
        **dialect_kw: Any,
    ) -> Self:
        r"""Return a new selectable with the given FETCH FIRST criterion
        applied.

        This is a numeric value which usually renders as ``FETCH {FIRST | NEXT}
        [ count ] {ROW | ROWS} {ONLY | WITH TIES}`` expression in the resulting
        select. This functionality is is currently implemented for Oracle
        Database, PostgreSQL, MSSQL.

        Use :meth:`_sql.GenerativeSelect.offset` to specify the offset.

        .. note::

           The :meth:`_sql.GenerativeSelect.fetch` method will replace
           any clause applied with :meth:`_sql.GenerativeSelect.limit`.

        .. versionadded:: 1.4

        :param count: an integer COUNT parameter, or a SQL expression
         that provides an integer result. When ``percent=True`` this will
         represent the percentage of rows to return, not the absolute value.
         Pass ``None`` to reset it.

        :param with_ties: When ``True``, the WITH TIES option is used
         to return any additional rows that tie for the last place in the
         result set according to the ``ORDER BY`` clause. The
         ``ORDER BY`` may be mandatory in this case. Defaults to ``False``

        :param percent: When ``True``, ``count`` represents the percentage
         of the total number of selected rows to return. Defaults to ``False``

        :param \**dialect_kw: Additional dialect-specific keyword arguments
         may be accepted by dialects.

         .. versionadded:: 2.0.41

        .. seealso::

           :meth:`_sql.GenerativeSelect.limit`

           :meth:`_sql.GenerativeSelect.offset`

        """
        self._validate_dialect_kwargs(dialect_kw)
        self._limit_clause = None
        if count is None:
            self._fetch_clause = self._fetch_clause_options = None
        else:
            self._fetch_clause = self._offset_or_limit_clause(count)
            self._fetch_clause_options = {
                "with_ties": with_ties,
                "percent": percent,
            }
        return self

    @_generative
    def offset(self, offset: _LimitOffsetType) -> Self:
        """Return a new selectable with the given OFFSET criterion
        applied.


        This is a numeric value which usually renders as an ``OFFSET``
        expression in the resulting select.  Backends that don't
        support ``OFFSET`` will attempt to provide similar
        functionality.

        :param offset: an integer OFFSET parameter, or a SQL expression
         that provides an integer result. Pass ``None`` to reset it.

        .. seealso::

           :meth:`_sql.GenerativeSelect.limit`

           :meth:`_sql.GenerativeSelect.fetch`

        """

        self._offset_clause = self._offset_or_limit_clause(offset)
        return self

    @_generative
    @util.preload_module("sqlalchemy.sql.util")
    def slice(
        self,
        start: int,
        stop: int,
    ) -> Self:
        """Apply LIMIT / OFFSET to this statement based on a slice.

        The start and stop indices behave like the argument to Python's
        built-in :func:`range` function. This method provides an
        alternative to using ``LIMIT``/``OFFSET`` to get a slice of the
        query.

        For example, ::

            stmt = select(User).order_by(User.id).slice(1, 3)

        renders as

        .. sourcecode:: sql

           SELECT users.id AS users_id,
                  users.name AS users_name
           FROM users ORDER BY users.id
           LIMIT ? OFFSET ?
           (2, 1)

        .. note::

           The :meth:`_sql.GenerativeSelect.slice` method will replace
           any clause applied with :meth:`_sql.GenerativeSelect.fetch`.

        .. versionadded:: 1.4  Added the :meth:`_sql.GenerativeSelect.slice`
           method generalized from the ORM.

        .. seealso::

           :meth:`_sql.GenerativeSelect.limit`

           :meth:`_sql.GenerativeSelect.offset`

           :meth:`_sql.GenerativeSelect.fetch`

        """
        sql_util = util.preloaded.sql_util
        self._fetch_clause = self._fetch_clause_options = None
        self._limit_clause, self._offset_clause = sql_util._make_slice(
            self._limit_clause, self._offset_clause, start, stop
        )
        return self

    @_generative
    def order_by(
        self,
        __first: Union[
            Literal[None, _NoArg.NO_ARG],
            _ColumnExpressionOrStrLabelArgument[Any],
        ] = _NoArg.NO_ARG,
        *clauses: _ColumnExpressionOrStrLabelArgument[Any],
    ) -> Self:
        r"""Return a new selectable with the given list of ORDER BY
        criteria applied.

        e.g.::

            stmt = select(table).order_by(table.c.id, table.c.name)

        Calling this method multiple times is equivalent to calling it once
        with all the clauses concatenated. All existing ORDER BY criteria may
        be cancelled by passing ``None`` by itself.  New ORDER BY criteria may
        then be added by invoking :meth:`_orm.Query.order_by` again, e.g.::

            # will erase all ORDER BY and ORDER BY new_col alone
            stmt = stmt.order_by(None).order_by(new_col)

        :param \*clauses: a series of :class:`_expression.ColumnElement`
         constructs
         which will be used to generate an ORDER BY clause.

        .. seealso::

            :ref:`tutorial_order_by` - in the :ref:`unified_tutorial`

            :ref:`tutorial_order_by_label` - in the :ref:`unified_tutorial`

        """

        if not clauses and __first is None:
            self._order_by_clauses = ()
        elif __first is not _NoArg.NO_ARG:
            self._order_by_clauses += tuple(
                coercions.expect(
                    roles.OrderByRole, clause, apply_propagate_attrs=self
                )
                for clause in (__first,) + clauses
            )
        return self

    @_generative
    def group_by(
        self,
        __first: Union[
            Literal[None, _NoArg.NO_ARG],
            _ColumnExpressionOrStrLabelArgument[Any],
        ] = _NoArg.NO_ARG,
        *clauses: _ColumnExpressionOrStrLabelArgument[Any],
    ) -> Self:
        r"""Return a new selectable with the given list of GROUP BY
        criterion applied.

        All existing GROUP BY settings can be suppressed by passing ``None``.

        e.g.::

            stmt = select(table.c.name, func.max(table.c.stat)).group_by(table.c.name)

        :param \*clauses: a series of :class:`_expression.ColumnElement`
         constructs
         which will be used to generate an GROUP BY clause.

        .. seealso::

            :ref:`tutorial_group_by_w_aggregates` - in the
            :ref:`unified_tutorial`

            :ref:`tutorial_order_by_label` - in the :ref:`unified_tutorial`

        """  # noqa: E501

        if not clauses and __first is None:
            self._group_by_clauses = ()
        elif __first is not _NoArg.NO_ARG:
            self._group_by_clauses += tuple(
                coercions.expect(
                    roles.GroupByRole, clause, apply_propagate_attrs=self
                )
                for clause in (__first,) + clauses
            )
        return self


@CompileState.plugin_for("default", "compound_select")
class CompoundSelectState(CompileState):
    @util.memoized_property
    def _label_resolve_dict(
        self,
    ) -> Tuple[
        Dict[str, ColumnElement[Any]],
        Dict[str, ColumnElement[Any]],
        Dict[str, ColumnElement[Any]],
    ]:
        # TODO: this is hacky and slow
        hacky_subquery = self.statement.subquery()
        hacky_subquery.named_with_column = False
        d = {c.key: c for c in hacky_subquery.c}
        return d, d, d


class _CompoundSelectKeyword(Enum):
    UNION = "UNION"
    UNION_ALL = "UNION ALL"
    EXCEPT = "EXCEPT"
    EXCEPT_ALL = "EXCEPT ALL"
    INTERSECT = "INTERSECT"
    INTERSECT_ALL = "INTERSECT ALL"


class CompoundSelect(HasCompileState, GenerativeSelect, TypedReturnsRows[_TP]):
    """Forms the basis of ``UNION``, ``UNION ALL``, and other
    SELECT-based set operations.


    .. seealso::

        :func:`_expression.union`

        :func:`_expression.union_all`

        :func:`_expression.intersect`

        :func:`_expression.intersect_all`

        :func:`_expression.except`

        :func:`_expression.except_all`

    """

    __visit_name__ = "compound_select"

    _traverse_internals: _TraverseInternalsType = (
        [
            ("selects", InternalTraversal.dp_clauseelement_list),
            ("_limit_clause", InternalTraversal.dp_clauseelement),
            ("_offset_clause", InternalTraversal.dp_clauseelement),
            ("_fetch_clause", InternalTraversal.dp_clauseelement),
            ("_fetch_clause_options", InternalTraversal.dp_plain_dict),
            ("_order_by_clauses", InternalTraversal.dp_clauseelement_list),
            ("_group_by_clauses", InternalTraversal.dp_clauseelement_list),
            ("_for_update_arg", InternalTraversal.dp_clauseelement),
            ("keyword", InternalTraversal.dp_string),
        ]
        + SupportsCloneAnnotations._clone_annotations_traverse_internals
        + HasCTE._has_ctes_traverse_internals
        + DialectKWArgs._dialect_kwargs_traverse_internals
    )

    selects: List[SelectBase]

    _is_from_container = True
    _auto_correlate = False

    def __init__(
        self,
        keyword: _CompoundSelectKeyword,
        *selects: _SelectStatementForCompoundArgument[_TP],
    ):
        self.keyword = keyword
        self.selects = [
            coercions.expect(
                roles.CompoundElementRole, s, apply_propagate_attrs=self
            ).self_group(against=self)
            for s in selects
        ]

        GenerativeSelect.__init__(self)

    @classmethod
    def _create_union(
        cls, *selects: _SelectStatementForCompoundArgument[_TP]
    ) -> CompoundSelect[_TP]:
        return CompoundSelect(_CompoundSelectKeyword.UNION, *selects)

    @classmethod
    def _create_union_all(
        cls, *selects: _SelectStatementForCompoundArgument[_TP]
    ) -> CompoundSelect[_TP]:
        return CompoundSelect(_CompoundSelectKeyword.UNION_ALL, *selects)

    @classmethod
    def _create_except(
        cls, *selects: _SelectStatementForCompoundArgument[_TP]
    ) -> CompoundSelect[_TP]:
        return CompoundSelect(_CompoundSelectKeyword.EXCEPT, *selects)

    @classmethod
    def _create_except_all(
        cls, *selects: _SelectStatementForCompoundArgument[_TP]
    ) -> CompoundSelect[_TP]:
        return CompoundSelect(_CompoundSelectKeyword.EXCEPT_ALL, *selects)

    @classmethod
    def _create_intersect(
        cls, *selects: _SelectStatementForCompoundArgument[_TP]
    ) -> CompoundSelect[_TP]:
        return CompoundSelect(_CompoundSelectKeyword.INTERSECT, *selects)

    @classmethod
    def _create_intersect_all(
        cls, *selects: _SelectStatementForCompoundArgument[_TP]
    ) -> CompoundSelect[_TP]:
        return CompoundSelect(_CompoundSelectKeyword.INTERSECT_ALL, *selects)

    def _scalar_type(self) -> TypeEngine[Any]:
        return self.selects[0]._scalar_type()

    def self_group(
        self, against: Optional[OperatorType] = None
    ) -> GroupedElement:
        return SelectStatementGrouping(self)

    def is_derived_from(self, fromclause: Optional[FromClause]) -> bool:
        for s in self.selects:
            if s.is_derived_from(fromclause):
                return True
        return False

    def set_label_style(self, style: SelectLabelStyle) -> Self:
        if self._label_style is not style:
            self = self._generate()
            select_0 = self.selects[0].set_label_style(style)
            self.selects = [select_0] + self.selects[1:]

        return self

    def _ensure_disambiguated_names(self) -> Self:
        new_select = self.selects[0]._ensure_disambiguated_names()
        if new_select is not self.selects[0]:
            self = self._generate()
            self.selects = [new_select] + self.selects[1:]

        return self

    def _generate_fromclause_column_proxies(
        self,
        subquery: FromClause,
        columns: ColumnCollection[str, KeyedColumnElement[Any]],
        primary_key: ColumnSet,
        foreign_keys: Set[KeyedColumnElement[Any]],
        *,
        proxy_compound_columns: Optional[
            Iterable[Sequence[ColumnElement[Any]]]
        ] = None,
    ) -> None:
        # this is a slightly hacky thing - the union exports a
        # column that resembles just that of the *first* selectable.
        # to get at a "composite" column, particularly foreign keys,
        # you have to dig through the proxies collection which we
        # generate below.
        select_0 = self.selects[0]

        if self._label_style is not LABEL_STYLE_DEFAULT:
            select_0 = select_0.set_label_style(self._label_style)

        # hand-construct the "_proxies" collection to include all
        # derived columns place a 'weight' annotation corresponding
        # to how low in the list of select()s the column occurs, so
        # that the corresponding_column() operation can resolve
        # conflicts
        extra_col_iterator = zip(
            *[
                [
                    c._annotate(dd)
                    for c in stmt._all_selected_columns
                    if is_column_element(c)
                ]
                for dd, stmt in [
                    ({"weight": i + 1}, stmt)
                    for i, stmt in enumerate(self.selects)
                ]
            ]
        )

        # the incoming proxy_compound_columns can be present also if this is
        # a compound embedded in a compound.  it's probably more appropriate
        # that we generate new weights local to this nested compound, though
        # i haven't tried to think what it means for compound nested in
        # compound
        select_0._generate_fromclause_column_proxies(
            subquery,
            columns,
            proxy_compound_columns=extra_col_iterator,
            primary_key=primary_key,
            foreign_keys=foreign_keys,
        )

    def _refresh_for_new_column(self, column: ColumnElement[Any]) -> None:
        super()._refresh_for_new_column(column)
        for select in self.selects:
            select._refresh_for_new_column(column)

    @util.ro_non_memoized_property
    def _all_selected_columns(self) -> _SelectIterable:
        return self.selects[0]._all_selected_columns

    @util.ro_non_memoized_property
    def selected_columns(
        self,
    ) -> ColumnCollection[str, ColumnElement[Any]]:
        """A :class:`_expression.ColumnCollection`
        representing the columns that
        this SELECT statement or similar construct returns in its result set,
        not including :class:`_sql.TextClause` constructs.

        For a :class:`_expression.CompoundSelect`, the
        :attr:`_expression.CompoundSelect.selected_columns`
        attribute returns the selected
        columns of the first SELECT statement contained within the series of
        statements within the set operation.

        .. seealso::

            :attr:`_sql.Select.selected_columns`

        .. versionadded:: 1.4

        """
        return self.selects[0].selected_columns


# backwards compat
for elem in _CompoundSelectKeyword:
    setattr(CompoundSelect, elem.name, elem)


@CompileState.plugin_for("default", "select")
class SelectState(util.MemoizedSlots, CompileState):
    __slots__ = (
        "from_clauses",
        "froms",
        "columns_plus_names",
        "_label_resolve_dict",
    )

    if TYPE_CHECKING:
        default_select_compile_options: CacheableOptions
    else:

        class default_select_compile_options(CacheableOptions):
            _cache_key_traversal = []

    if TYPE_CHECKING:

        @classmethod
        def get_plugin_class(
            cls, statement: Executable
        ) -> Type[SelectState]: ...

    def __init__(
        self,
        statement: Select[Any],
        compiler: SQLCompiler,
        **kw: Any,
    ):
        self.statement = statement
        self.from_clauses = statement._from_obj

        for memoized_entities in statement._memoized_select_entities:
            self._setup_joins(
                memoized_entities._setup_joins, memoized_entities._raw_columns
            )

        if statement._setup_joins:
            self._setup_joins(statement._setup_joins, statement._raw_columns)

        self.froms = self._get_froms(statement)

        self.columns_plus_names = statement._generate_columns_plus_names(True)

    @classmethod
    def _plugin_not_implemented(cls) -> NoReturn:
        raise NotImplementedError(
            "The default SELECT construct without plugins does not "
            "implement this method."
        )

    @classmethod
    def get_column_descriptions(
        cls, statement: Select[Any]
    ) -> List[Dict[str, Any]]:
        return [
            {
                "name": name,
                "type": element.type,
                "expr": element,
            }
            for _, name, _, element, _ in (
                statement._generate_columns_plus_names(False)
            )
        ]

    @classmethod
    def from_statement(
        cls, statement: Select[Any], from_statement: roles.ReturnsRowsRole
    ) -> ExecutableReturnsRows:
        cls._plugin_not_implemented()

    @classmethod
    def get_columns_clause_froms(
        cls, statement: Select[Any]
    ) -> List[FromClause]:
        return cls._normalize_froms(
            itertools.chain.from_iterable(
                element._from_objects for element in statement._raw_columns
            )
        )

    @classmethod
    def _column_naming_convention(
        cls, label_style: SelectLabelStyle
    ) -> _LabelConventionCallable:
        table_qualified = label_style is LABEL_STYLE_TABLENAME_PLUS_COL

        dedupe = label_style is not LABEL_STYLE_NONE

        pa = prefix_anon_map()
        names = set()

        def go(
            c: Union[ColumnElement[Any], TextClause],
            col_name: Optional[str] = None,
        ) -> Optional[str]:
            if is_text_clause(c):
                return None
            elif TYPE_CHECKING:
                assert is_column_element(c)

            if not dedupe:
                name = c._proxy_key
                if name is None:
                    name = "_no_label"
                return name

            name = c._tq_key_label if table_qualified else c._proxy_key

            if name is None:
                name = "_no_label"
                if name in names:
                    return c._anon_label(name) % pa
                else:
                    names.add(name)
                    return name

            elif name in names:
                return (
                    c._anon_tq_key_label % pa
                    if table_qualified
                    else c._anon_key_label % pa
                )
            else:
                names.add(name)
                return name

        return go

    def _get_froms(self, statement: Select[Any]) -> List[FromClause]:
        ambiguous_table_name_map: _AmbiguousTableNameMap
        self._ambiguous_table_name_map = ambiguous_table_name_map = {}

        return self._normalize_froms(
            itertools.chain(
                self.from_clauses,
                itertools.chain.from_iterable(
                    [
                        element._from_objects
                        for element in statement._raw_columns
                    ]
                ),
                itertools.chain.from_iterable(
                    [
                        element._from_objects
                        for element in statement._where_criteria
                    ]
                ),
            ),
            check_statement=statement,
            ambiguous_table_name_map=ambiguous_table_name_map,
        )

    @classmethod
    def _normalize_froms(
        cls,
        iterable_of_froms: Iterable[FromClause],
        check_statement: Optional[Select[Any]] = None,
        ambiguous_table_name_map: Optional[_AmbiguousTableNameMap] = None,
    ) -> List[FromClause]:
        """given an iterable of things to select FROM, reduce them to what
        would actually render in the FROM clause of a SELECT.

        This does the job of checking for JOINs, tables, etc. that are in fact
        overlapping due to cloning, adaption, present in overlapping joins,
        etc.

        """
        seen: Set[FromClause] = set()
        froms: List[FromClause] = []

        for item in iterable_of_froms:
            if is_subquery(item) and item.element is check_statement:
                raise exc.InvalidRequestError(
                    "select() construct refers to itself as a FROM"
                )

            if not seen.intersection(item._cloned_set):
                froms.append(item)
                seen.update(item._cloned_set)

        if froms:
            toremove = set(
                itertools.chain.from_iterable(
                    [_expand_cloned(f._hide_froms) for f in froms]
                )
            )
            if toremove:
                # filter out to FROM clauses not in the list,
                # using a list to maintain ordering
                froms = [f for f in froms if f not in toremove]

            if ambiguous_table_name_map is not None:
                ambiguous_table_name_map.update(
                    (
                        fr.name,
                        _anonymous_label.safe_construct(
                            hash(fr.name), fr.name
                        ),
                    )
                    for item in froms
                    for fr in item._from_objects
                    if is_table(fr)
                    and fr.schema
                    and fr.name not in ambiguous_table_name_map
                )

        return froms

    def _get_display_froms(
        self,
        explicit_correlate_froms: Optional[Sequence[FromClause]] = None,
        implicit_correlate_froms: Optional[Sequence[FromClause]] = None,
    ) -> List[FromClause]:
        """Return the full list of 'from' clauses to be displayed.

        Takes into account a set of existing froms which may be
        rendered in the FROM clause of enclosing selects; this Select
        may want to leave those absent if it is automatically
        correlating.

        """

        froms = self.froms

        if self.statement._correlate:
            to_correlate = self.statement._correlate
            if to_correlate:
                froms = [
                    f
                    for f in froms
                    if f
                    not in _cloned_intersection(
                        _cloned_intersection(
                            froms, explicit_correlate_froms or ()
                        ),
                        to_correlate,
                    )
                ]

        if self.statement._correlate_except is not None:
            froms = [
                f
                for f in froms
                if f
                not in _cloned_difference(
                    _cloned_intersection(
                        froms, explicit_correlate_froms or ()
                    ),
                    self.statement._correlate_except,
                )
            ]

        if (
            self.statement._auto_correlate
            and implicit_correlate_froms
            and len(froms) > 1
        ):
            froms = [
                f
                for f in froms
                if f
                not in _cloned_intersection(froms, implicit_correlate_froms)
            ]

            if not len(froms):
                raise exc.InvalidRequestError(
                    "Select statement '%r"
                    "' returned no FROM clauses "
                    "due to auto-correlation; "
                    "specify correlate(<tables>) "
                    "to control correlation "
                    "manually." % self.statement
                )

        return froms

    def _memoized_attr__label_resolve_dict(
        self,
    ) -> Tuple[
        Dict[str, ColumnElement[Any]],
        Dict[str, ColumnElement[Any]],
        Dict[str, ColumnElement[Any]],
    ]:
        with_cols: Dict[str, ColumnElement[Any]] = {
            c._tq_label or c.key: c
            for c in self.statement._all_selected_columns
            if c._allow_label_resolve
        }
        only_froms: Dict[str, ColumnElement[Any]] = {
            c.key: c  # type: ignore
            for c in _select_iterables(self.froms)
            if c._allow_label_resolve
        }
        only_cols: Dict[str, ColumnElement[Any]] = with_cols.copy()
        for key, value in only_froms.items():
            with_cols.setdefault(key, value)

        return with_cols, only_froms, only_cols

    @classmethod
    def determine_last_joined_entity(
        cls, stmt: Select[Any]
    ) -> Optional[_JoinTargetElement]:
        if stmt._setup_joins:
            return stmt._setup_joins[-1][0]
        else:
            return None

    @classmethod
    def all_selected_columns(cls, statement: Select[Any]) -> _SelectIterable:
        return [c for c in _select_iterables(statement._raw_columns)]

    def _setup_joins(
        self,
        args: Tuple[_SetupJoinsElement, ...],
        raw_columns: List[_ColumnsClauseElement],
    ) -> None:
        for right, onclause, left, flags in args:
            if TYPE_CHECKING:
                if onclause is not None:
                    assert isinstance(onclause, ColumnElement)

            isouter = flags["isouter"]
            full = flags["full"]

            if left is None:
                (
                    left,
                    replace_from_obj_index,
                ) = self._join_determine_implicit_left_side(
                    raw_columns, left, right, onclause
                )
            else:
                (replace_from_obj_index) = self._join_place_explicit_left_side(
                    left
                )

            # these assertions can be made here, as if the right/onclause
            # contained ORM elements, the select() statement would have been
            # upgraded to an ORM select, and this method would not be called;
            # orm.context.ORMSelectCompileState._join() would be
            # used instead.
            if TYPE_CHECKING:
                assert isinstance(right, FromClause)
                if onclause is not None:
                    assert isinstance(onclause, ColumnElement)

            if replace_from_obj_index is not None:
                # splice into an existing element in the
                # self._from_obj list
                left_clause = self.from_clauses[replace_from_obj_index]

                self.from_clauses = (
                    self.from_clauses[:replace_from_obj_index]
                    + (
                        Join(
                            left_clause,
                            right,
                            onclause,
                            isouter=isouter,
                            full=full,
                        ),
                    )
                    + self.from_clauses[replace_from_obj_index + 1 :]
                )
            else:
                assert left is not None
                self.from_clauses = self.from_clauses + (
                    Join(left, right, onclause, isouter=isouter, full=full),
                )

    @util.preload_module("sqlalchemy.sql.util")
    def _join_determine_implicit_left_side(
        self,
        raw_columns: List[_ColumnsClauseElement],
        left: Optional[FromClause],
        right: _JoinTargetElement,
        onclause: Optional[ColumnElement[Any]],
    ) -> Tuple[Optional[FromClause], Optional[int]]:
        """When join conditions don't express the left side explicitly,
        determine if an existing FROM or entity in this query
        can serve as the left hand side.

        """

        sql_util = util.preloaded.sql_util

        replace_from_obj_index: Optional[int] = None

        from_clauses = self.from_clauses

        if from_clauses:
            indexes: List[int] = sql_util.find_left_clause_to_join_from(
                from_clauses, right, onclause
            )

            if len(indexes) == 1:
                replace_from_obj_index = indexes[0]
                left = from_clauses[replace_from_obj_index]
        else:
            potential = {}
            statement = self.statement

            for from_clause in itertools.chain(
                itertools.chain.from_iterable(
                    [element._from_objects for element in raw_columns]
                ),
                itertools.chain.from_iterable(
                    [
                        element._from_objects
                        for element in statement._where_criteria
                    ]
                ),
            ):
                potential[from_clause] = ()

            all_clauses = list(potential.keys())
            indexes = sql_util.find_left_clause_to_join_from(
                all_clauses, right, onclause
            )

            if len(indexes) == 1:
                left = all_clauses[indexes[0]]

        if len(indexes) > 1:
            raise exc.InvalidRequestError(
                "Can't determine which FROM clause to join "
                "from, there are multiple FROMS which can "
                "join to this entity. Please use the .select_from() "
                "method to establish an explicit left side, as well as "
                "providing an explicit ON clause if not present already to "
                "help resolve the ambiguity."
            )
        elif not indexes:
            raise exc.InvalidRequestError(
                "Don't know how to join to %r. "
                "Please use the .select_from() "
                "method to establish an explicit left side, as well as "
                "providing an explicit ON clause if not present already to "
                "help resolve the ambiguity." % (right,)
            )
        return left, replace_from_obj_index

    @util.preload_module("sqlalchemy.sql.util")
    def _join_place_explicit_left_side(
        self, left: FromClause
    ) -> Optional[int]:
        replace_from_obj_index: Optional[int] = None

        sql_util = util.preloaded.sql_util

        from_clauses = list(self.statement._iterate_from_elements())

        if from_clauses:
            indexes: List[int] = sql_util.find_left_clause_that_matches_given(
                self.from_clauses, left
            )
        else:
            indexes = []

        if len(indexes) > 1:
            raise exc.InvalidRequestError(
                "Can't identify which entity in which to assign the "
                "left side of this join.   Please use a more specific "
                "ON clause."
            )

        # have an index, means the left side is already present in
        # an existing FROM in the self._from_obj tuple
        if indexes:
            replace_from_obj_index = indexes[0]

        # no index, means we need to add a new element to the
        # self._from_obj tuple

        return replace_from_obj_index


class _SelectFromElements:
    __slots__ = ()

    _raw_columns: List[_ColumnsClauseElement]
    _where_criteria: Tuple[ColumnElement[Any], ...]
    _from_obj: Tuple[FromClause, ...]

    def _iterate_from_elements(self) -> Iterator[FromClause]:
        # note this does not include elements
        # in _setup_joins

        seen = set()
        for element in self._raw_columns:
            for fr in element._from_objects:
                if fr in seen:
                    continue
                seen.add(fr)
                yield fr
        for element in self._where_criteria:
            for fr in element._from_objects:
                if fr in seen:
                    continue
                seen.add(fr)
                yield fr
        for element in self._from_obj:
            if element in seen:
                continue
            seen.add(element)
            yield element


class _MemoizedSelectEntities(
    cache_key.HasCacheKey, traversals.HasCopyInternals, visitors.Traversible
):
    """represents partial state from a Select object, for the case
    where Select.columns() has redefined the set of columns/entities the
    statement will be SELECTing from.  This object represents
    the entities from the SELECT before that transformation was applied,
    so that transformations that were made in terms of the SELECT at that
    time, such as join() as well as options(), can access the correct context.

    In previous SQLAlchemy versions, this wasn't needed because these
    constructs calculated everything up front, like when you called join()
    or options(), it did everything to figure out how that would translate
    into specific SQL constructs that would be ready to send directly to the
    SQL compiler when needed.  But as of
    1.4, all of that stuff is done in the compilation phase, during the
    "compile state" portion of the process, so that the work can all be
    cached.  So it needs to be able to resolve joins/options2 based on what
    the list of entities was when those methods were called.


    """

    __visit_name__ = "memoized_select_entities"

    _traverse_internals: _TraverseInternalsType = [
        ("_raw_columns", InternalTraversal.dp_clauseelement_list),
        ("_setup_joins", InternalTraversal.dp_setup_join_tuple),
        ("_with_options", InternalTraversal.dp_executable_options),
    ]

    _is_clone_of: Optional[ClauseElement]
    _raw_columns: List[_ColumnsClauseElement]
    _setup_joins: Tuple[_SetupJoinsElement, ...]
    _with_options: Tuple[ExecutableOption, ...]

    _annotations = util.EMPTY_DICT

    def _clone(self, **kw: Any) -> Self:
        c = self.__class__.__new__(self.__class__)
        c.__dict__ = {k: v for k, v in self.__dict__.items()}

        c._is_clone_of = self.__dict__.get("_is_clone_of", self)
        return c

    @classmethod
    def _generate_for_statement(cls, select_stmt: Select[Any]) -> None:
        if select_stmt._setup_joins or select_stmt._with_options:
            self = _MemoizedSelectEntities()
            self._raw_columns = select_stmt._raw_columns
            self._setup_joins = select_stmt._setup_joins
            self._with_options = select_stmt._with_options

            select_stmt._memoized_select_entities += (self,)
            select_stmt._raw_columns = []
            select_stmt._setup_joins = select_stmt._with_options = ()


class Select(
    HasPrefixes,
    HasSuffixes,
    HasHints,
    HasCompileState,
    _SelectFromElements,
    GenerativeSelect,
    TypedReturnsRows[_TP],
):
    """Represents a ``SELECT`` statement.

    The :class:`_sql.Select` object is normally constructed using the
    :func:`_sql.select` function.  See that function for details.

    .. seealso::

        :func:`_sql.select`

        :ref:`tutorial_selecting_data` - in the 2.0 tutorial

    """

    __visit_name__ = "select"

    _setup_joins: Tuple[_SetupJoinsElement, ...] = ()
    _memoized_select_entities: Tuple[TODO_Any, ...] = ()

    _raw_columns: List[_ColumnsClauseElement]

    _distinct: bool = False
    _distinct_on: Tuple[ColumnElement[Any], ...] = ()
    _correlate: Tuple[FromClause, ...] = ()
    _correlate_except: Optional[Tuple[FromClause, ...]] = None
    _where_criteria: Tuple[ColumnElement[Any], ...] = ()
    _having_criteria: Tuple[ColumnElement[Any], ...] = ()
    _from_obj: Tuple[FromClause, ...] = ()
    _auto_correlate = True
    _is_select_statement = True
    _compile_options: CacheableOptions = (
        SelectState.default_select_compile_options
    )

    _traverse_internals: _TraverseInternalsType = (
        [
            ("_raw_columns", InternalTraversal.dp_clauseelement_list),
            (
                "_memoized_select_entities",
                InternalTraversal.dp_memoized_select_entities,
            ),
            ("_from_obj", InternalTraversal.dp_clauseelement_list),
            ("_where_criteria", InternalTraversal.dp_clauseelement_tuple),
            ("_having_criteria", InternalTraversal.dp_clauseelement_tuple),
            ("_order_by_clauses", InternalTraversal.dp_clauseelement_tuple),
            ("_group_by_clauses", InternalTraversal.dp_clauseelement_tuple),
            ("_setup_joins", InternalTraversal.dp_setup_join_tuple),
            ("_correlate", InternalTraversal.dp_clauseelement_tuple),
            ("_correlate_except", InternalTraversal.dp_clauseelement_tuple),
            ("_limit_clause", InternalTraversal.dp_clauseelement),
            ("_offset_clause", InternalTraversal.dp_clauseelement),
            ("_fetch_clause", InternalTraversal.dp_clauseelement),
            ("_fetch_clause_options", InternalTraversal.dp_plain_dict),
            ("_for_update_arg", InternalTraversal.dp_clauseelement),
            ("_distinct", InternalTraversal.dp_boolean),
            ("_distinct_on", InternalTraversal.dp_clauseelement_tuple),
            ("_label_style", InternalTraversal.dp_plain_obj),
        ]
        + HasCTE._has_ctes_traverse_internals
        + HasPrefixes._has_prefixes_traverse_internals
        + HasSuffixes._has_suffixes_traverse_internals
        + HasHints._has_hints_traverse_internals
        + SupportsCloneAnnotations._clone_annotations_traverse_internals
        + Executable._executable_traverse_internals
        + DialectKWArgs._dialect_kwargs_traverse_internals
    )

    _cache_key_traversal: _CacheKeyTraversalType = _traverse_internals + [
        ("_compile_options", InternalTraversal.dp_has_cache_key)
    ]

    _compile_state_factory: Type[SelectState]

    @classmethod
    def _create_raw_select(cls, **kw: Any) -> Select[Any]:
        """Create a :class:`.Select` using raw ``__new__`` with no coercions.

        Used internally to build up :class:`.Select` constructs with
        pre-established state.

        """

        stmt = Select.__new__(Select)
        stmt.__dict__.update(kw)
        return stmt

    def __init__(
        self, *entities: _ColumnsClauseArgument[Any], **dialect_kw: Any
    ):
        r"""Construct a new :class:`_expression.Select`.

        The public constructor for :class:`_expression.Select` is the
        :func:`_sql.select` function.

        """
        self._raw_columns = [
            coercions.expect(
                roles.ColumnsClauseRole, ent, apply_propagate_attrs=self
            )
            for ent in entities
        ]
        GenerativeSelect.__init__(self)

    def _scalar_type(self) -> TypeEngine[Any]:
        if not self._raw_columns:
            return NULLTYPE
        elem = self._raw_columns[0]
        cols = list(elem._select_iterable)
        return cols[0].type

    def filter(self, *criteria: _ColumnExpressionArgument[bool]) -> Self:
        """A synonym for the :meth:`_sql.Select.where` method."""

        return self.where(*criteria)

    def _filter_by_zero(
        self,
    ) -> Union[
        FromClause, _JoinTargetProtocol, ColumnElement[Any], TextClause
    ]:
        if self._setup_joins:
            meth = SelectState.get_plugin_class(
                self
            ).determine_last_joined_entity
            _last_joined_entity = meth(self)
            if _last_joined_entity is not None:
                return _last_joined_entity

        if self._from_obj:
            return self._from_obj[0]

        return self._raw_columns[0]

    if TYPE_CHECKING:

        @overload
        def scalar_subquery(
            self: Select[Tuple[_MAYBE_ENTITY]],
        ) -> ScalarSelect[Any]: ...

        @overload
        def scalar_subquery(
            self: Select[Tuple[_NOT_ENTITY]],
        ) -> ScalarSelect[_NOT_ENTITY]: ...

        @overload
        def scalar_subquery(self) -> ScalarSelect[Any]: ...

        def scalar_subquery(self) -> ScalarSelect[Any]: ...

    def filter_by(self, **kwargs: Any) -> Self:
        r"""apply the given filtering criterion as a WHERE clause
        to this select.

        """
        from_entity = self._filter_by_zero()

        clauses = [
            _entity_namespace_key(from_entity, key) == value
            for key, value in kwargs.items()
        ]
        return self.filter(*clauses)

    @property
    def column_descriptions(self) -> Any:
        """Return a :term:`plugin-enabled` 'column descriptions' structure
        referring to the columns which are SELECTed by this statement.

        This attribute is generally useful when using the ORM, as an
        extended structure which includes information about mapped
        entities is returned.  The section :ref:`queryguide_inspection`
        contains more background.

        For a Core-only statement, the structure returned by this accessor
        is derived from the same objects that are returned by the
        :attr:`.Select.selected_columns` accessor, formatted as a list of
        dictionaries which contain the keys ``name``, ``type`` and ``expr``,
        which indicate the column expressions to be selected::

            >>> stmt = select(user_table)
            >>> stmt.column_descriptions
            [
                {
                    'name': 'id',
                    'type': Integer(),
                    'expr': Column('id', Integer(), ...)},
                {
                    'name': 'name',
                    'type': String(length=30),
                    'expr': Column('name', String(length=30), ...)}
            ]

        .. versionchanged:: 1.4.33 The :attr:`.Select.column_descriptions`
           attribute returns a structure for a Core-only set of entities,
           not just ORM-only entities.

        .. seealso::

            :attr:`.UpdateBase.entity_description` - entity information for
            an :func:`.insert`, :func:`.update`, or :func:`.delete`

            :ref:`queryguide_inspection` - ORM background

        """
        meth = SelectState.get_plugin_class(self).get_column_descriptions
        return meth(self)

    def from_statement(
        self, statement: roles.ReturnsRowsRole
    ) -> ExecutableReturnsRows:
        """Apply the columns which this :class:`.Select` would select
        onto another statement.

        This operation is :term:`plugin-specific` and will raise a not
        supported exception if this :class:`_sql.Select` does not select from
        plugin-enabled entities.


        The statement is typically either a :func:`_expression.text` or
        :func:`_expression.select` construct, and should return the set of
        columns appropriate to the entities represented by this
        :class:`.Select`.

        .. seealso::

            :ref:`orm_queryguide_selecting_text` - usage examples in the
            ORM Querying Guide

        """
        meth = SelectState.get_plugin_class(self).from_statement
        return meth(self, statement)

    @_generative
    def join(
        self,
        target: _JoinTargetArgument,
        onclause: Optional[_OnClauseArgument] = None,
        *,
        isouter: bool = False,
        full: bool = False,
    ) -> Self:
        r"""Create a SQL JOIN against this :class:`_expression.Select`
        object's criterion
        and apply generatively, returning the newly resulting
        :class:`_expression.Select`.

        E.g.::

            stmt = select(user_table).join(
                address_table, user_table.c.id == address_table.c.user_id
            )

        The above statement generates SQL similar to:

        .. sourcecode:: sql

            SELECT user.id, user.name
            FROM user
            JOIN address ON user.id = address.user_id

        .. versionchanged:: 1.4 :meth:`_expression.Select.join` now creates
           a :class:`_sql.Join` object between a :class:`_sql.FromClause`
           source that is within the FROM clause of the existing SELECT,
           and a given target :class:`_sql.FromClause`, and then adds
           this :class:`_sql.Join` to the FROM clause of the newly generated
           SELECT statement.    This is completely reworked from the behavior
           in 1.3, which would instead create a subquery of the entire
           :class:`_expression.Select` and then join that subquery to the
           target.

           This is a **backwards incompatible change** as the previous behavior
           was mostly useless, producing an unnamed subquery rejected by
           most databases in any case.   The new behavior is modeled after
           that of the very successful :meth:`_orm.Query.join` method in the
           ORM, in order to support the functionality of :class:`_orm.Query`
           being available by using a :class:`_sql.Select` object with an
           :class:`_orm.Session`.

           See the notes for this change at :ref:`change_select_join`.


        :param target: target table to join towards

        :param onclause: ON clause of the join.  If omitted, an ON clause
         is generated automatically based on the :class:`_schema.ForeignKey`
         linkages between the two tables, if one can be unambiguously
         determined, otherwise an error is raised.

        :param isouter: if True, generate LEFT OUTER join.  Same as
         :meth:`_expression.Select.outerjoin`.

        :param full: if True, generate FULL OUTER join.

        .. seealso::

            :ref:`tutorial_select_join` - in the :doc:`/tutorial/index`

            :ref:`orm_queryguide_joins` - in the :ref:`queryguide_toplevel`

            :meth:`_expression.Select.join_from`

            :meth:`_expression.Select.outerjoin`

        """  # noqa: E501
        join_target = coercions.expect(
            roles.JoinTargetRole, target, apply_propagate_attrs=self
        )
        if onclause is not None:
            onclause_element = coercions.expect(roles.OnClauseRole, onclause)
        else:
            onclause_element = None

        self._setup_joins += (
            (
                join_target,
                onclause_element,
                None,
                {"isouter": isouter, "full": full},
            ),
        )
        return self

    def outerjoin_from(
        self,
        from_: _FromClauseArgument,
        target: _JoinTargetArgument,
        onclause: Optional[_OnClauseArgument] = None,
        *,
        full: bool = False,
    ) -> Self:
        r"""Create a SQL LEFT OUTER JOIN against this
        :class:`_expression.Select` object's criterion and apply generatively,
        returning the newly resulting :class:`_expression.Select`.

        Usage is the same as that of :meth:`_selectable.Select.join_from`.

        """
        return self.join_from(
            from_, target, onclause=onclause, isouter=True, full=full
        )

    @_generative
    def join_from(
        self,
        from_: _FromClauseArgument,
        target: _JoinTargetArgument,
        onclause: Optional[_OnClauseArgument] = None,
        *,
        isouter: bool = False,
        full: bool = False,
    ) -> Self:
        r"""Create a SQL JOIN against this :class:`_expression.Select`
        object's criterion
        and apply generatively, returning the newly resulting
        :class:`_expression.Select`.

        E.g.::

            stmt = select(user_table, address_table).join_from(
                user_table, address_table, user_table.c.id == address_table.c.user_id
            )

        The above statement generates SQL similar to:

        .. sourcecode:: sql

            SELECT user.id, user.name, address.id, address.email, address.user_id
            FROM user JOIN address ON user.id = address.user_id

        .. versionadded:: 1.4

        :param from\_: the left side of the join, will be rendered in the
         FROM clause and is roughly equivalent to using the
         :meth:`.Select.select_from` method.

        :param target: target table to join towards

        :param onclause: ON clause of the join.

        :param isouter: if True, generate LEFT OUTER join.  Same as
         :meth:`_expression.Select.outerjoin`.

        :param full: if True, generate FULL OUTER join.

        .. seealso::

            :ref:`tutorial_select_join` - in the :doc:`/tutorial/index`

            :ref:`orm_queryguide_joins` - in the :ref:`queryguide_toplevel`

            :meth:`_expression.Select.join`

        """  # noqa: E501

        # note the order of parsing from vs. target is important here, as we
        # are also deriving the source of the plugin (i.e. the subject mapper
        # in an ORM query) which should favor the "from_" over the "target"

        from_ = coercions.expect(
            roles.FromClauseRole, from_, apply_propagate_attrs=self
        )
        join_target = coercions.expect(
            roles.JoinTargetRole, target, apply_propagate_attrs=self
        )
        if onclause is not None:
            onclause_element = coercions.expect(roles.OnClauseRole, onclause)
        else:
            onclause_element = None

        self._setup_joins += (
            (
                join_target,
                onclause_element,
                from_,
                {"isouter": isouter, "full": full},
            ),
        )
        return self

    def outerjoin(
        self,
        target: _JoinTargetArgument,
        onclause: Optional[_OnClauseArgument] = None,
        *,
        full: bool = False,
    ) -> Self:
        """Create a left outer join.

        Parameters are the same as that of :meth:`_expression.Select.join`.

        .. versionchanged:: 1.4 :meth:`_expression.Select.outerjoin` now
           creates a :class:`_sql.Join` object between a
           :class:`_sql.FromClause` source that is within the FROM clause of
           the existing SELECT, and a given target :class:`_sql.FromClause`,
           and then adds this :class:`_sql.Join` to the FROM clause of the
           newly generated SELECT statement.    This is completely reworked
           from the behavior in 1.3, which would instead create a subquery of
           the entire
           :class:`_expression.Select` and then join that subquery to the
           target.

           This is a **backwards incompatible change** as the previous behavior
           was mostly useless, producing an unnamed subquery rejected by
           most databases in any case.   The new behavior is modeled after
           that of the very successful :meth:`_orm.Query.join` method in the
           ORM, in order to support the functionality of :class:`_orm.Query`
           being available by using a :class:`_sql.Select` object with an
           :class:`_orm.Session`.

           See the notes for this change at :ref:`change_select_join`.

        .. seealso::

            :ref:`tutorial_select_join` - in the :doc:`/tutorial/index`

            :ref:`orm_queryguide_joins` - in the :ref:`queryguide_toplevel`

            :meth:`_expression.Select.join`

        """
        return self.join(target, onclause=onclause, isouter=True, full=full)

    def get_final_froms(self) -> Sequence[FromClause]:
        """Compute the final displayed list of :class:`_expression.FromClause`
        elements.

        This method will run through the full computation required to
        determine what FROM elements will be displayed in the resulting
        SELECT statement, including shadowing individual tables with
        JOIN objects, as well as full computation for ORM use cases including
        eager loading clauses.

        For ORM use, this accessor returns the **post compilation**
        list of FROM objects; this collection will include elements such as
        eagerly loaded tables and joins.  The objects will **not** be
        ORM enabled and not work as a replacement for the
        :meth:`_sql.Select.select_froms` collection; additionally, the
        method is not well performing for an ORM enabled statement as it
        will incur the full ORM construction process.

        To retrieve the FROM list that's implied by the "columns" collection
        passed to the :class:`_sql.Select` originally, use the
        :attr:`_sql.Select.columns_clause_froms` accessor.

        To select from an alternative set of columns while maintaining the
        FROM list, use the :meth:`_sql.Select.with_only_columns` method and
        pass the
        :paramref:`_sql.Select.with_only_columns.maintain_column_froms`
        parameter.

        .. versionadded:: 1.4.23 - the :meth:`_sql.Select.get_final_froms`
           method replaces the previous :attr:`_sql.Select.froms` accessor,
           which is deprecated.

        .. seealso::

            :attr:`_sql.Select.columns_clause_froms`

        """
        compiler = self._default_compiler()

        return self._compile_state_factory(self, compiler)._get_display_froms()

    @property
    @util.deprecated(
        "1.4.23",
        "The :attr:`_expression.Select.froms` attribute is moved to "
        "the :meth:`_expression.Select.get_final_froms` method.",
    )
    def froms(self) -> Sequence[FromClause]:
        """Return the displayed list of :class:`_expression.FromClause`
        elements.


        """
        return self.get_final_froms()

    @property
    def columns_clause_froms(self) -> List[FromClause]:
        """Return the set of :class:`_expression.FromClause` objects implied
        by the columns clause of this SELECT statement.

        .. versionadded:: 1.4.23

        .. seealso::

            :attr:`_sql.Select.froms` - "final" FROM list taking the full
            statement into account

            :meth:`_sql.Select.with_only_columns` - makes use of this
            collection to set up a new FROM list

        """

        return SelectState.get_plugin_class(self).get_columns_clause_froms(
            self
        )

    @property
    def inner_columns(self) -> _SelectIterable:
        """An iterator of all :class:`_expression.ColumnElement`
        expressions which would
        be rendered into the columns clause of the resulting SELECT statement.

        This method is legacy as of 1.4 and is superseded by the
        :attr:`_expression.Select.exported_columns` collection.

        """

        return iter(self._all_selected_columns)

    def is_derived_from(self, fromclause: Optional[FromClause]) -> bool:
        if fromclause is not None and self in fromclause._cloned_set:
            return True

        for f in self._iterate_from_elements():
            if f.is_derived_from(fromclause):
                return True
        return False

    def _copy_internals(
        self, clone: _CloneCallableType = _clone, **kw: Any
    ) -> None:
        # Select() object has been cloned and probably adapted by the
        # given clone function.  Apply the cloning function to internal
        # objects

        # 1. keep a dictionary of the froms we've cloned, and what
        # they've become.  This allows us to ensure the same cloned from
        # is used when other items such as columns are "cloned"

        all_the_froms = set(
            itertools.chain(
                _from_objects(*self._raw_columns),
                _from_objects(*self._where_criteria),
                _from_objects(*[elem[0] for elem in self._setup_joins]),
            )
        )

        # do a clone for the froms we've gathered.  what is important here
        # is if any of the things we are selecting from, like tables,
        # were converted into Join objects.   if so, these need to be
        # added to _from_obj explicitly, because otherwise they won't be
        # part of the new state, as they don't associate themselves with
        # their columns.
        new_froms = {f: clone(f, **kw) for f in all_the_froms}

        # 2. copy FROM collections, adding in joins that we've created.
        existing_from_obj = [clone(f, **kw) for f in self._from_obj]
        add_froms = (
            {f for f in new_froms.values() if isinstance(f, Join)}
            .difference(all_the_froms)
            .difference(existing_from_obj)
        )

        self._from_obj = tuple(existing_from_obj) + tuple(add_froms)

        # 3. clone everything else, making sure we use columns
        # corresponding to the froms we just made.
        def replace(
            obj: Union[BinaryExpression[Any], ColumnClause[Any]],
            **kw: Any,
        ) -> Optional[KeyedColumnElement[Any]]:
            if isinstance(obj, ColumnClause) and obj.table in new_froms:
                newelem = new_froms[obj.table].corresponding_column(obj)
                return newelem
            return None

        kw["replace"] = replace

        # copy everything else.   for table-ish things like correlate,
        # correlate_except, setup_joins, these clone normally.  For
        # column-expression oriented things like raw_columns, where_criteria,
        # order by, we get this from the new froms.
        super()._copy_internals(clone=clone, omit_attrs=("_from_obj",), **kw)

        self._reset_memoizations()

    def get_children(self, **kw: Any) -> Iterable[ClauseElement]:
        return itertools.chain(
            super().get_children(
                omit_attrs=("_from_obj", "_correlate", "_correlate_except"),
                **kw,
            ),
            self._iterate_from_elements(),
        )

    @_generative
    def add_columns(
        self, *entities: _ColumnsClauseArgument[Any]
    ) -> Select[Any]:
        r"""Return a new :func:`_expression.select` construct with
        the given entities appended to its columns clause.

        E.g.::

            my_select = my_select.add_columns(table.c.new_column)

        The original expressions in the columns clause remain in place.
        To replace the original expressions with new ones, see the method
        :meth:`_expression.Select.with_only_columns`.

        :param \*entities: column, table, or other entity expressions to be
         added to the columns clause

        .. seealso::

            :meth:`_expression.Select.with_only_columns` - replaces existing
            expressions rather than appending.

            :ref:`orm_queryguide_select_multiple_entities` - ORM-centric
            example

        """
        self._reset_memoizations()

        self._raw_columns = self._raw_columns + [
            coercions.expect(
                roles.ColumnsClauseRole, column, apply_propagate_attrs=self
            )
            for column in entities
        ]
        return self

    def _set_entities(
        self, entities: Iterable[_ColumnsClauseArgument[Any]]
    ) -> None:
        self._raw_columns = [
            coercions.expect(
                roles.ColumnsClauseRole, ent, apply_propagate_attrs=self
            )
            for ent in util.to_list(entities)
        ]

    @util.deprecated(
        "1.4",
        "The :meth:`_expression.Select.column` method is deprecated and will "
        "be removed in a future release.  Please use "
        ":meth:`_expression.Select.add_columns`",
    )
    def column(self, column: _ColumnsClauseArgument[Any]) -> Select[Any]:
        """Return a new :func:`_expression.select` construct with
        the given column expression added to its columns clause.

        E.g.::

            my_select = my_select.column(table.c.new_column)

        See the documentation for
        :meth:`_expression.Select.with_only_columns`
        for guidelines on adding /replacing the columns of a
        :class:`_expression.Select` object.

        """
        return self.add_columns(column)

    @util.preload_module("sqlalchemy.sql.util")
    def reduce_columns(self, only_synonyms: bool = True) -> Select[Any]:
        """Return a new :func:`_expression.select` construct with redundantly
        named, equivalently-valued columns removed from the columns clause.

        "Redundant" here means two columns where one refers to the
        other either based on foreign key, or via a simple equality
        comparison in the WHERE clause of the statement.   The primary purpose
        of this method is to automatically construct a select statement
        with all uniquely-named columns, without the need to use
        table-qualified labels as
        :meth:`_expression.Select.set_label_style`
        does.

        When columns are omitted based on foreign key, the referred-to
        column is the one that's kept.  When columns are omitted based on
        WHERE equivalence, the first column in the columns clause is the
        one that's kept.

        :param only_synonyms: when True, limit the removal of columns
         to those which have the same name as the equivalent.   Otherwise,
         all columns that are equivalent to another are removed.

        """
        woc: Select[Any]
        woc = self.with_only_columns(
            *util.preloaded.sql_util.reduce_columns(
                self._all_selected_columns,
                only_synonyms=only_synonyms,
                *(self._where_criteria + self._from_obj),
            )
        )
        return woc

    # START OVERLOADED FUNCTIONS self.with_only_columns Select 1-8 ", *, maintain_column_froms: bool =..." # noqa: E501

    # code within this block is **programmatically,
    # statically generated** by tools/generate_tuple_map_overloads.py

    @overload
    def with_only_columns(
        self, __ent0: _TCCA[_T0], *, maintain_column_froms: bool = ...
    ) -> Select[Tuple[_T0]]: ...

    @overload
    def with_only_columns(
        self,
        __ent0: _TCCA[_T0],
        __ent1: _TCCA[_T1],
        *,
        maintain_column_froms: bool = ...,
    ) -> Select[Tuple[_T0, _T1]]: ...

    @overload
    def with_only_columns(
        self,
        __ent0: _TCCA[_T0],
        __ent1: _TCCA[_T1],
        __ent2: _TCCA[_T2],
        *,
        maintain_column_froms: bool = ...,
    ) -> Select[Tuple[_T0, _T1, _T2]]: ...

    @overload
    def with_only_columns(
        self,
        __ent0: _TCCA[_T0],
        __ent1: _TCCA[_T1],
        __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3],
        *,
        maintain_column_froms: bool = ...,
    ) -> Select[Tuple[_T0, _T1, _T2, _T3]]: ...

    @overload
    def with_only_columns(
        self,
        __ent0: _TCCA[_T0],
        __ent1: _TCCA[_T1],
        __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3],
        __ent4: _TCCA[_T4],
        *,
        maintain_column_froms: bool = ...,
    ) -> Select[Tuple[_T0, _T1, _T2, _T3, _T4]]: ...

    @overload
    def with_only_columns(
        self,
        __ent0: _TCCA[_T0],
        __ent1: _TCCA[_T1],
        __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3],
        __ent4: _TCCA[_T4],
        __ent5: _TCCA[_T5],
        *,
        maintain_column_froms: bool = ...,
    ) -> Select[Tuple[_T0, _T1, _T2, _T3, _T4, _T5]]: ...

    @overload
    def with_only_columns(
        self,
        __ent0: _TCCA[_T0],
        __ent1: _TCCA[_T1],
        __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3],
        __ent4: _TCCA[_T4],
        __ent5: _TCCA[_T5],
        __ent6: _TCCA[_T6],
        *,
        maintain_column_froms: bool = ...,
    ) -> Select[Tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6]]: ...

    @overload
    def with_only_columns(
        self,
        __ent0: _TCCA[_T0],
        __ent1: _TCCA[_T1],
        __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3],
        __ent4: _TCCA[_T4],
        __ent5: _TCCA[_T5],
        __ent6: _TCCA[_T6],
        __ent7: _TCCA[_T7],
        *,
        maintain_column_froms: bool = ...,
    ) -> Select[Tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7]]: ...

    # END OVERLOADED FUNCTIONS self.with_only_columns

    @overload
    def with_only_columns(
        self,
        *entities: _ColumnsClauseArgument[Any],
        maintain_column_froms: bool = False,
        **__kw: Any,
    ) -> Select[Any]: ...

    @_generative
    def with_only_columns(
        self,
        *entities: _ColumnsClauseArgument[Any],
        maintain_column_froms: bool = False,
        **__kw: Any,
    ) -> Select[Any]:
        r"""Return a new :func:`_expression.select` construct with its columns
        clause replaced with the given entities.

        By default, this method is exactly equivalent to as if the original
        :func:`_expression.select` had been called with the given entities.
        E.g. a statement::

            s = select(table1.c.a, table1.c.b)
            s = s.with_only_columns(table1.c.b)

        should be exactly equivalent to::

            s = select(table1.c.b)

        In this mode of operation, :meth:`_sql.Select.with_only_columns`
        will also dynamically alter the FROM clause of the
        statement if it is not explicitly stated.
        To maintain the existing set of FROMs including those implied by the
        current columns clause, add the
        :paramref:`_sql.Select.with_only_columns.maintain_column_froms`
        parameter::

            s = select(table1.c.a, table2.c.b)
            s = s.with_only_columns(table1.c.a, maintain_column_froms=True)

        The above parameter performs a transfer of the effective FROMs
        in the columns collection to the :meth:`_sql.Select.select_from`
        method, as though the following were invoked::

            s = select(table1.c.a, table2.c.b)
            s = s.select_from(table1, table2).with_only_columns(table1.c.a)

        The :paramref:`_sql.Select.with_only_columns.maintain_column_froms`
        parameter makes use of the :attr:`_sql.Select.columns_clause_froms`
        collection and performs an operation equivalent to the following::

            s = select(table1.c.a, table2.c.b)
            s = s.select_from(*s.columns_clause_froms).with_only_columns(table1.c.a)

        :param \*entities: column expressions to be used.

        :param maintain_column_froms: boolean parameter that will ensure the
         FROM list implied from the current columns clause will be transferred
         to the :meth:`_sql.Select.select_from` method first.

         .. versionadded:: 1.4.23

        """  # noqa: E501

        if __kw:
            raise _no_kw()

        # memoizations should be cleared here as of
        # I95c560ffcbfa30b26644999412fb6a385125f663 , asserting this
        # is the case for now.
        self._assert_no_memoizations()

        if maintain_column_froms:
            self.select_from.non_generative(  # type: ignore
                self, *self.columns_clause_froms
            )

        # then memoize the FROMs etc.
        _MemoizedSelectEntities._generate_for_statement(self)

        self._raw_columns = [
            coercions.expect(roles.ColumnsClauseRole, c)
            for c in coercions._expression_collection_was_a_list(
                "entities", "Select.with_only_columns", entities
            )
        ]
        return self

    @property
    def whereclause(self) -> Optional[ColumnElement[Any]]:
        """Return the completed WHERE clause for this
        :class:`_expression.Select` statement.

        This assembles the current collection of WHERE criteria
        into a single :class:`_expression.BooleanClauseList` construct.


        .. versionadded:: 1.4

        """

        return BooleanClauseList._construct_for_whereclause(
            self._where_criteria
        )

    _whereclause = whereclause

    @_generative
    def where(self, *whereclause: _ColumnExpressionArgument[bool]) -> Self:
        """Return a new :func:`_expression.select` construct with
        the given expression added to
        its WHERE clause, joined to the existing clause via AND, if any.

        """

        assert isinstance(self._where_criteria, tuple)

        for criterion in whereclause:
            where_criteria: ColumnElement[Any] = coercions.expect(
                roles.WhereHavingRole, criterion, apply_propagate_attrs=self
            )
            self._where_criteria += (where_criteria,)
        return self

    @_generative
    def having(self, *having: _ColumnExpressionArgument[bool]) -> Self:
        """Return a new :func:`_expression.select` construct with
        the given expression added to
        its HAVING clause, joined to the existing clause via AND, if any.

        """

        for criterion in having:
            having_criteria = coercions.expect(
                roles.WhereHavingRole, criterion, apply_propagate_attrs=self
            )
            self._having_criteria += (having_criteria,)
        return self

    @_generative
    def distinct(self, *expr: _ColumnExpressionArgument[Any]) -> Self:
        r"""Return a new :func:`_expression.select` construct which
        will apply DISTINCT to the SELECT statement overall.

        E.g.::

            from sqlalchemy import select

            stmt = select(users_table.c.id, users_table.c.name).distinct()

        The above would produce an statement resembling:

        .. sourcecode:: sql

            SELECT DISTINCT user.id, user.name FROM user

        The method also accepts an ``*expr`` parameter which produces the
        PostgreSQL dialect-specific ``DISTINCT ON`` expression.  Using this
        parameter on other backends which don't support this syntax will
        raise an error.

        :param \*expr: optional column expressions.  When present,
         the PostgreSQL dialect will render a ``DISTINCT ON (<expressions>)``
         construct.  A deprecation warning and/or :class:`_exc.CompileError`
         will be raised on other backends.

         .. deprecated:: 1.4 Using \*expr in other dialects is deprecated
            and will raise :class:`_exc.CompileError` in a future version.

        """
        if expr:
            self._distinct = True
            self._distinct_on = self._distinct_on + tuple(
                coercions.expect(roles.ByOfRole, e, apply_propagate_attrs=self)
                for e in expr
            )
        else:
            self._distinct = True
        return self

    @_generative
    def select_from(self, *froms: _FromClauseArgument) -> Self:
        r"""Return a new :func:`_expression.select` construct with the
        given FROM expression(s)
        merged into its list of FROM objects.

        E.g.::

            table1 = table("t1", column("a"))
            table2 = table("t2", column("b"))
            s = select(table1.c.a).select_from(
                table1.join(table2, table1.c.a == table2.c.b)
            )

        The "from" list is a unique set on the identity of each element,
        so adding an already present :class:`_schema.Table`
        or other selectable
        will have no effect.   Passing a :class:`_expression.Join` that refers
        to an already present :class:`_schema.Table`
        or other selectable will have
        the effect of concealing the presence of that selectable as
        an individual element in the rendered FROM list, instead
        rendering it into a JOIN clause.

        While the typical purpose of :meth:`_expression.Select.select_from`
        is to
        replace the default, derived FROM clause with a join, it can
        also be called with individual table elements, multiple times
        if desired, in the case that the FROM clause cannot be fully
        derived from the columns clause::

            select(func.count("*")).select_from(table1)

        """

        self._from_obj += tuple(
            coercions.expect(
                roles.FromClauseRole, fromclause, apply_propagate_attrs=self
            )
            for fromclause in froms
        )
        return self

    @_generative
    def correlate(
        self,
        *fromclauses: Union[Literal[None, False], _FromClauseArgument],
    ) -> Self:
        r"""Return a new :class:`_expression.Select`
        which will correlate the given FROM
        clauses to that of an enclosing :class:`_expression.Select`.

        Calling this method turns off the :class:`_expression.Select` object's
        default behavior of "auto-correlation".  Normally, FROM elements
        which appear in a :class:`_expression.Select`
        that encloses this one via
        its :term:`WHERE clause`, ORDER BY, HAVING or
        :term:`columns clause` will be omitted from this
        :class:`_expression.Select`
        object's :term:`FROM clause`.
        Setting an explicit correlation collection using the
        :meth:`_expression.Select.correlate`
        method provides a fixed list of FROM objects
        that can potentially take place in this process.

        When :meth:`_expression.Select.correlate`
        is used to apply specific FROM clauses
        for correlation, the FROM elements become candidates for
        correlation regardless of how deeply nested this
        :class:`_expression.Select`
        object is, relative to an enclosing :class:`_expression.Select`
        which refers to
        the same FROM object.  This is in contrast to the behavior of
        "auto-correlation" which only correlates to an immediate enclosing
        :class:`_expression.Select`.
        Multi-level correlation ensures that the link
        between enclosed and enclosing :class:`_expression.Select`
        is always via
        at least one WHERE/ORDER BY/HAVING/columns clause in order for
        correlation to take place.

        If ``None`` is passed, the :class:`_expression.Select`
        object will correlate
        none of its FROM entries, and all will render unconditionally
        in the local FROM clause.

        :param \*fromclauses: one or more :class:`.FromClause` or other
         FROM-compatible construct such as an ORM mapped entity to become part
         of the correlate collection; alternatively pass a single value
         ``None`` to remove all existing correlations.

        .. seealso::

            :meth:`_expression.Select.correlate_except`

            :ref:`tutorial_scalar_subquery`

        """

        # tests failing when we try to change how these
        # arguments are passed

        self._auto_correlate = False
        if not fromclauses or fromclauses[0] in {None, False}:
            if len(fromclauses) > 1:
                raise exc.ArgumentError(
                    "additional FROM objects not accepted when "
                    "passing None/False to correlate()"
                )
            self._correlate = ()
        else:
            self._correlate = self._correlate + tuple(
                coercions.expect(roles.FromClauseRole, f) for f in fromclauses
            )
        return self

    @_generative
    def correlate_except(
        self,
        *fromclauses: Union[Literal[None, False], _FromClauseArgument],
    ) -> Self:
        r"""Return a new :class:`_expression.Select`
        which will omit the given FROM
        clauses from the auto-correlation process.

        Calling :meth:`_expression.Select.correlate_except` turns off the
        :class:`_expression.Select` object's default behavior of
        "auto-correlation" for the given FROM elements.  An element
        specified here will unconditionally appear in the FROM list, while
        all other FROM elements remain subject to normal auto-correlation
        behaviors.

        If ``None`` is passed, or no arguments are passed,
        the :class:`_expression.Select` object will correlate all of its
        FROM entries.

        :param \*fromclauses: a list of one or more
         :class:`_expression.FromClause`
         constructs, or other compatible constructs (i.e. ORM-mapped
         classes) to become part of the correlate-exception collection.

        .. seealso::

            :meth:`_expression.Select.correlate`

            :ref:`tutorial_scalar_subquery`

        """

        self._auto_correlate = False
        if not fromclauses or fromclauses[0] in {None, False}:
            if len(fromclauses) > 1:
                raise exc.ArgumentError(
                    "additional FROM objects not accepted when "
                    "passing None/False to correlate_except()"
                )
            self._correlate_except = ()
        else:
            self._correlate_except = (self._correlate_except or ()) + tuple(
                coercions.expect(roles.FromClauseRole, f) for f in fromclauses
            )

        return self

    @HasMemoized_ro_memoized_attribute
    def selected_columns(
        self,
    ) -> ColumnCollection[str, ColumnElement[Any]]:
        """A :class:`_expression.ColumnCollection`
        representing the columns that
        this SELECT statement or similar construct returns in its result set,
        not including :class:`_sql.TextClause` constructs.

        This collection differs from the :attr:`_expression.FromClause.columns`
        collection of a :class:`_expression.FromClause` in that the columns
        within this collection cannot be directly nested inside another SELECT
        statement; a subquery must be applied first which provides for the
        necessary parenthesization required by SQL.

        For a :func:`_expression.select` construct, the collection here is
        exactly what would be rendered inside the "SELECT" statement, and the
        :class:`_expression.ColumnElement` objects are directly present as they
        were given, e.g.::

            col1 = column("q", Integer)
            col2 = column("p", Integer)
            stmt = select(col1, col2)

        Above, ``stmt.selected_columns`` would be a collection that contains
        the ``col1`` and ``col2`` objects directly. For a statement that is
        against a :class:`_schema.Table` or other
        :class:`_expression.FromClause`, the collection will use the
        :class:`_expression.ColumnElement` objects that are in the
        :attr:`_expression.FromClause.c` collection of the from element.

        A use case for the :attr:`_sql.Select.selected_columns` collection is
        to allow the existing columns to be referenced when adding additional
        criteria, e.g.::

            def filter_on_id(my_select, id):
                return my_select.where(my_select.selected_columns["id"] == id)


            stmt = select(MyModel)

            # adds "WHERE id=:param" to the statement
            stmt = filter_on_id(stmt, 42)

        .. note::

            The :attr:`_sql.Select.selected_columns` collection does not
            include expressions established in the columns clause using the
            :func:`_sql.text` construct; these are silently omitted from the
            collection. To use plain textual column expressions inside of a
            :class:`_sql.Select` construct, use the :func:`_sql.literal_column`
            construct.


        .. versionadded:: 1.4

        """

        # compare to SelectState._generate_columns_plus_names, which
        # generates the actual names used in the SELECT string.  that
        # method is more complex because it also renders columns that are
        # fully ambiguous, e.g. same column more than once.
        conv = cast(
            "Callable[[Any], str]",
            SelectState._column_naming_convention(self._label_style),
        )

        cc: ColumnCollection[str, ColumnElement[Any]] = ColumnCollection(
            [
                (conv(c), c)
                for c in self._all_selected_columns
                if is_column_element(c)
            ]
        )
        return cc.as_readonly()

    @HasMemoized_ro_memoized_attribute
    def _all_selected_columns(self) -> _SelectIterable:
        meth = SelectState.get_plugin_class(self).all_selected_columns
        return list(meth(self))

    def _ensure_disambiguated_names(self) -> Select[Any]:
        if self._label_style is LABEL_STYLE_NONE:
            self = self.set_label_style(LABEL_STYLE_DISAMBIGUATE_ONLY)
        return self

    def _generate_fromclause_column_proxies(
        self,
        subquery: FromClause,
        columns: ColumnCollection[str, KeyedColumnElement[Any]],
        primary_key: ColumnSet,
        foreign_keys: Set[KeyedColumnElement[Any]],
        *,
        proxy_compound_columns: Optional[
            Iterable[Sequence[ColumnElement[Any]]]
        ] = None,
    ) -> None:
        """Generate column proxies to place in the exported ``.c``
        collection of a subquery."""

        if proxy_compound_columns:
            extra_col_iterator = proxy_compound_columns
            prox = [
                c._make_proxy(
                    subquery,
                    key=proxy_key,
                    name=required_label_name,
                    name_is_truncatable=True,
                    compound_select_cols=extra_cols,
                    primary_key=primary_key,
                    foreign_keys=foreign_keys,
                )
                for (
                    (
                        required_label_name,
                        proxy_key,
                        fallback_label_name,
                        c,
                        repeated,
                    ),
                    extra_cols,
                ) in (
                    zip(
                        self._generate_columns_plus_names(False),
                        extra_col_iterator,
                    )
                )
                if is_column_element(c)
            ]
        else:
            prox = [
                c._make_proxy(
                    subquery,
                    key=proxy_key,
                    name=required_label_name,
                    name_is_truncatable=True,
                    primary_key=primary_key,
                    foreign_keys=foreign_keys,
                )
                for (
                    required_label_name,
                    proxy_key,
                    fallback_label_name,
                    c,
                    repeated,
                ) in (self._generate_columns_plus_names(False))
                if is_column_element(c)
            ]

        columns._populate_separate_keys(prox)

    def _needs_parens_for_grouping(self) -> bool:
        return self._has_row_limiting_clause or bool(
            self._order_by_clause.clauses
        )

    def self_group(
        self, against: Optional[OperatorType] = None
    ) -> Union[SelectStatementGrouping[Self], Self]:
        """Return a 'grouping' construct as per the
        :class:`_expression.ClauseElement` specification.

        This produces an element that can be embedded in an expression. Note
        that this method is called automatically as needed when constructing
        expressions and should not require explicit use.

        """
        if (
            isinstance(against, CompoundSelect)
            and not self._needs_parens_for_grouping()
        ):
            return self
        else:
            return SelectStatementGrouping(self)

    def union(
        self, *other: _SelectStatementForCompoundArgument[_TP]
    ) -> CompoundSelect[_TP]:
        r"""Return a SQL ``UNION`` of this select() construct against
        the given selectables provided as positional arguments.

        :param \*other: one or more elements with which to create a
         UNION.

         .. versionchanged:: 1.4.28

            multiple elements are now accepted.

        :param \**kwargs: keyword arguments are forwarded to the constructor
         for the newly created :class:`_sql.CompoundSelect` object.

        """
        return CompoundSelect._create_union(self, *other)

    def union_all(
        self, *other: _SelectStatementForCompoundArgument[_TP]
    ) -> CompoundSelect[_TP]:
        r"""Return a SQL ``UNION ALL`` of this select() construct against
        the given selectables provided as positional arguments.

        :param \*other: one or more elements with which to create a
         UNION.

         .. versionchanged:: 1.4.28

            multiple elements are now accepted.

        :param \**kwargs: keyword arguments are forwarded to the constructor
         for the newly created :class:`_sql.CompoundSelect` object.

        """
        return CompoundSelect._create_union_all(self, *other)

    def except_(
        self, *other: _SelectStatementForCompoundArgument[_TP]
    ) -> CompoundSelect[_TP]:
        r"""Return a SQL ``EXCEPT`` of this select() construct against
        the given selectable provided as positional arguments.

        :param \*other: one or more elements with which to create a
         UNION.

         .. versionchanged:: 1.4.28

            multiple elements are now accepted.

        """
        return CompoundSelect._create_except(self, *other)

    def except_all(
        self, *other: _SelectStatementForCompoundArgument[_TP]
    ) -> CompoundSelect[_TP]:
        r"""Return a SQL ``EXCEPT ALL`` of this select() construct against
        the given selectables provided as positional arguments.

        :param \*other: one or more elements with which to create a
         UNION.

         .. versionchanged:: 1.4.28

            multiple elements are now accepted.

        """
        return CompoundSelect._create_except_all(self, *other)

    def intersect(
        self, *other: _SelectStatementForCompoundArgument[_TP]
    ) -> CompoundSelect[_TP]:
        r"""Return a SQL ``INTERSECT`` of this select() construct against
        the given selectables provided as positional arguments.

        :param \*other: one or more elements with which to create a
         UNION.

         .. versionchanged:: 1.4.28

            multiple elements are now accepted.

        :param \**kwargs: keyword arguments are forwarded to the constructor
         for the newly created :class:`_sql.CompoundSelect` object.

        """
        return CompoundSelect._create_intersect(self, *other)

    def intersect_all(
        self, *other: _SelectStatementForCompoundArgument[_TP]
    ) -> CompoundSelect[_TP]:
        r"""Return a SQL ``INTERSECT ALL`` of this select() construct
        against the given selectables provided as positional arguments.

        :param \*other: one or more elements with which to create a
         UNION.

         .. versionchanged:: 1.4.28

            multiple elements are now accepted.

        :param \**kwargs: keyword arguments are forwarded to the constructor
         for the newly created :class:`_sql.CompoundSelect` object.

        """
        return CompoundSelect._create_intersect_all(self, *other)


class ScalarSelect(
    roles.InElementRole, Generative, GroupedElement, ColumnElement[_T]
):
    """Represent a scalar subquery.


    A :class:`_sql.ScalarSelect` is created by invoking the
    :meth:`_sql.SelectBase.scalar_subquery` method.   The object
    then participates in other SQL expressions as a SQL column expression
    within the :class:`_sql.ColumnElement` hierarchy.

    .. seealso::

        :meth:`_sql.SelectBase.scalar_subquery`

        :ref:`tutorial_scalar_subquery` - in the 2.0 tutorial

    """

    _traverse_internals: _TraverseInternalsType = [
        ("element", InternalTraversal.dp_clauseelement),
        ("type", InternalTraversal.dp_type),
    ]

    _from_objects: List[FromClause] = []
    _is_from_container = True
    if not TYPE_CHECKING:
        _is_implicitly_boolean = False
    inherit_cache = True

    element: SelectBase

    def __init__(self, element: SelectBase) -> None:
        self.element = element
        self.type = element._scalar_type()
        self._propagate_attrs = element._propagate_attrs

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.element, attr)

    def __getstate__(self) -> Dict[str, Any]:
        return {"element": self.element, "type": self.type}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.element = state["element"]
        self.type = state["type"]

    @property
    def columns(self) -> NoReturn:
        raise exc.InvalidRequestError(
            "Scalar Select expression has no "
            "columns; use this object directly "
            "within a column-level expression."
        )

    c = columns

    @_generative
    def where(self, crit: _ColumnExpressionArgument[bool]) -> Self:
        """Apply a WHERE clause to the SELECT statement referred to
        by this :class:`_expression.ScalarSelect`.

        """
        self.element = cast("Select[Any]", self.element).where(crit)
        return self

    def self_group(self, against: Optional[OperatorType] = None) -> Self:
        return self

    if TYPE_CHECKING:

        def _ungroup(self) -> Select[Any]: ...

    @_generative
    def correlate(
        self,
        *fromclauses: Union[Literal[None, False], _FromClauseArgument],
    ) -> Self:
        r"""Return a new :class:`_expression.ScalarSelect`
        which will correlate the given FROM
        clauses to that of an enclosing :class:`_expression.Select`.

        This method is mirrored from the :meth:`_sql.Select.correlate` method
        of the underlying :class:`_sql.Select`.  The method applies the
        :meth:_sql.Select.correlate` method, then returns a new
        :class:`_sql.ScalarSelect` against that statement.

        .. versionadded:: 1.4 Previously, the
           :meth:`_sql.ScalarSelect.correlate`
           method was only available from :class:`_sql.Select`.

        :param \*fromclauses: a list of one or more
         :class:`_expression.FromClause`
         constructs, or other compatible constructs (i.e. ORM-mapped
         classes) to become part of the correlate collection.

        .. seealso::

            :meth:`_expression.ScalarSelect.correlate_except`

            :ref:`tutorial_scalar_subquery` - in the 2.0 tutorial


        """
        self.element = cast("Select[Any]", self.element).correlate(
            *fromclauses
        )
        return self

    @_generative
    def correlate_except(
        self,
        *fromclauses: Union[Literal[None, False], _FromClauseArgument],
    ) -> Self:
        r"""Return a new :class:`_expression.ScalarSelect`
        which will omit the given FROM
        clauses from the auto-correlation process.

        This method is mirrored from the
        :meth:`_sql.Select.correlate_except` method of the underlying
        :class:`_sql.Select`.  The method applies the
        :meth:_sql.Select.correlate_except` method, then returns a new
        :class:`_sql.ScalarSelect` against that statement.

        .. versionadded:: 1.4 Previously, the
           :meth:`_sql.ScalarSelect.correlate_except`
           method was only available from :class:`_sql.Select`.

        :param \*fromclauses: a list of one or more
         :class:`_expression.FromClause`
         constructs, or other compatible constructs (i.e. ORM-mapped
         classes) to become part of the correlate-exception collection.

        .. seealso::

            :meth:`_expression.ScalarSelect.correlate`

            :ref:`tutorial_scalar_subquery` - in the 2.0 tutorial


        """

        self.element = cast("Select[Any]", self.element).correlate_except(
            *fromclauses
        )
        return self


class Exists(UnaryExpression[bool]):
    """Represent an ``EXISTS`` clause.

    See :func:`_sql.exists` for a description of usage.

    An ``EXISTS`` clause can also be constructed from a :func:`_sql.select`
    instance by calling :meth:`_sql.SelectBase.exists`.

    """

    inherit_cache = True
    element: Union[SelectStatementGrouping[Select[Any]], ScalarSelect[Any]]

    def __init__(
        self,
        __argument: Optional[
            Union[_ColumnsClauseArgument[Any], SelectBase, ScalarSelect[Any]]
        ] = None,
    ):
        s: ScalarSelect[Any]

        # TODO: this seems like we should be using coercions for this
        if __argument is None:
            s = Select(literal_column("*")).scalar_subquery()
        elif isinstance(__argument, SelectBase):
            s = __argument.scalar_subquery()
            s._propagate_attrs = __argument._propagate_attrs
        elif isinstance(__argument, ScalarSelect):
            s = __argument
        else:
            s = Select(__argument).scalar_subquery()

        UnaryExpression.__init__(
            self,
            s,
            operator=operators.exists,
            type_=type_api.BOOLEANTYPE,
            wraps_column_expression=True,
        )

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return []

    def _regroup(
        self, fn: Callable[[Select[Any]], Select[Any]]
    ) -> SelectStatementGrouping[Select[Any]]:
        element = self.element._ungroup()
        new_element = fn(element)

        return_value = new_element.self_group(against=operators.exists)
        assert isinstance(return_value, SelectStatementGrouping)
        return return_value

    def select(self) -> Select[Tuple[bool]]:
        r"""Return a SELECT of this :class:`_expression.Exists`.

        e.g.::

            stmt = exists(some_table.c.id).where(some_table.c.id == 5).select()

        This will produce a statement resembling:

        .. sourcecode:: sql

            SELECT EXISTS (SELECT id FROM some_table WHERE some_table = :param) AS anon_1

        .. seealso::

            :func:`_expression.select` - general purpose
            method which allows for arbitrary column lists.

        """  # noqa

        return Select(self)

    def correlate(
        self,
        *fromclauses: Union[Literal[None, False], _FromClauseArgument],
    ) -> Self:
        """Apply correlation to the subquery noted by this
        :class:`_sql.Exists`.

        .. seealso::

            :meth:`_sql.ScalarSelect.correlate`

        """
        e = self._clone()
        e.element = self._regroup(
            lambda element: element.correlate(*fromclauses)
        )
        return e

    def correlate_except(
        self,
        *fromclauses: Union[Literal[None, False], _FromClauseArgument],
    ) -> Self:
        """Apply correlation to the subquery noted by this
        :class:`_sql.Exists`.

        .. seealso::

            :meth:`_sql.ScalarSelect.correlate_except`

        """

        e = self._clone()
        e.element = self._regroup(
            lambda element: element.correlate_except(*fromclauses)
        )
        return e

    def select_from(self, *froms: _FromClauseArgument) -> Self:
        """Return a new :class:`_expression.Exists` construct,
        applying the given
        expression to the :meth:`_expression.Select.select_from`
        method of the select
        statement contained.

        .. note:: it is typically preferable to build a :class:`_sql.Select`
           statement first, including the desired WHERE clause, then use the
           :meth:`_sql.SelectBase.exists` method to produce an
           :class:`_sql.Exists` object at once.

        """
        e = self._clone()
        e.element = self._regroup(lambda element: element.select_from(*froms))
        return e

    def where(self, *clause: _ColumnExpressionArgument[bool]) -> Self:
        """Return a new :func:`_expression.exists` construct with the
        given expression added to
        its WHERE clause, joined to the existing clause via AND, if any.


        .. note:: it is typically preferable to build a :class:`_sql.Select`
           statement first, including the desired WHERE clause, then use the
           :meth:`_sql.SelectBase.exists` method to produce an
           :class:`_sql.Exists` object at once.

        """
        e = self._clone()
        e.element = self._regroup(lambda element: element.where(*clause))
        return e


class TextualSelect(SelectBase, ExecutableReturnsRows, Generative):
    """Wrap a :class:`_expression.TextClause` construct within a
    :class:`_expression.SelectBase`
    interface.

    This allows the :class:`_expression.TextClause` object to gain a
    ``.c`` collection
    and other FROM-like capabilities such as
    :meth:`_expression.FromClause.alias`,
    :meth:`_expression.SelectBase.cte`, etc.

    The :class:`_expression.TextualSelect` construct is produced via the
    :meth:`_expression.TextClause.columns`
    method - see that method for details.

    .. versionchanged:: 1.4 the :class:`_expression.TextualSelect`
       class was renamed
       from ``TextAsFrom``, to more correctly suit its role as a
       SELECT-oriented object and not a FROM clause.

    .. seealso::

        :func:`_expression.text`

        :meth:`_expression.TextClause.columns` - primary creation interface.

    """

    __visit_name__ = "textual_select"

    _label_style = LABEL_STYLE_NONE

    _traverse_internals: _TraverseInternalsType = (
        [
            ("element", InternalTraversal.dp_clauseelement),
            ("column_args", InternalTraversal.dp_clauseelement_list),
        ]
        + SupportsCloneAnnotations._clone_annotations_traverse_internals
        + HasCTE._has_ctes_traverse_internals
    )

    _is_textual = True

    is_text = True
    is_select = True

    def __init__(
        self,
        text: TextClause,
        columns: List[_ColumnExpressionArgument[Any]],
        positional: bool = False,
    ) -> None:
        self._init(
            text,
            # convert for ORM attributes->columns, etc
            [
                coercions.expect(roles.LabeledColumnExprRole, c)
                for c in columns
            ],
            positional,
        )

    def _init(
        self,
        text: TextClause,
        columns: List[NamedColumn[Any]],
        positional: bool = False,
    ) -> None:
        self.element = text
        self.column_args = columns
        self.positional = positional

    @HasMemoized_ro_memoized_attribute
    def selected_columns(
        self,
    ) -> ColumnCollection[str, KeyedColumnElement[Any]]:
        """A :class:`_expression.ColumnCollection`
        representing the columns that
        this SELECT statement or similar construct returns in its result set,
        not including :class:`_sql.TextClause` constructs.

        This collection differs from the :attr:`_expression.FromClause.columns`
        collection of a :class:`_expression.FromClause` in that the columns
        within this collection cannot be directly nested inside another SELECT
        statement; a subquery must be applied first which provides for the
        necessary parenthesization required by SQL.

        For a :class:`_expression.TextualSelect` construct, the collection
        contains the :class:`_expression.ColumnElement` objects that were
        passed to the constructor, typically via the
        :meth:`_expression.TextClause.columns` method.


        .. versionadded:: 1.4

        """
        return ColumnCollection(
            (c.key, c) for c in self.column_args
        ).as_readonly()

    @util.ro_non_memoized_property
    def _all_selected_columns(self) -> _SelectIterable:
        return self.column_args

    def set_label_style(self, style: SelectLabelStyle) -> TextualSelect:
        return self

    def _ensure_disambiguated_names(self) -> TextualSelect:
        return self

    @_generative
    def bindparams(
        self,
        *binds: BindParameter[Any],
        **bind_as_values: Any,
    ) -> Self:
        self.element = self.element.bindparams(*binds, **bind_as_values)
        return self

    def _generate_fromclause_column_proxies(
        self,
        fromclause: FromClause,
        columns: ColumnCollection[str, KeyedColumnElement[Any]],
        primary_key: ColumnSet,
        foreign_keys: Set[KeyedColumnElement[Any]],
        *,
        proxy_compound_columns: Optional[
            Iterable[Sequence[ColumnElement[Any]]]
        ] = None,
    ) -> None:
        if TYPE_CHECKING:
            assert isinstance(fromclause, Subquery)

        if proxy_compound_columns:
            columns._populate_separate_keys(
                c._make_proxy(
                    fromclause,
                    compound_select_cols=extra_cols,
                    primary_key=primary_key,
                    foreign_keys=foreign_keys,
                )
                for c, extra_cols in zip(
                    self.column_args, proxy_compound_columns
                )
            )
        else:
            columns._populate_separate_keys(
                c._make_proxy(
                    fromclause,
                    primary_key=primary_key,
                    foreign_keys=foreign_keys,
                )
                for c in self.column_args
            )

    def _scalar_type(self) -> Union[TypeEngine[Any], Any]:
        return self.column_args[0].type


TextAsFrom = TextualSelect
"""Backwards compatibility with the previous name"""


class AnnotatedFromClause(Annotated):
    def _copy_internals(self, **kw: Any) -> None:
        super()._copy_internals(**kw)
        if kw.get("ind_cols_on_fromclause", False):
            ee = self._Annotated__element  # type: ignore

            self.c = ee.__class__.c.fget(self)  # type: ignore

    @util.ro_memoized_property
    def c(self) -> ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]:
        """proxy the .c collection of the underlying FromClause.

        Originally implemented in 2008 as a simple load of the .c collection
        when the annotated construct was created (see d3621ae961a), in modern
        SQLAlchemy versions this can be expensive for statements constructed
        with ORM aliases.   So for #8796 SQLAlchemy 2.0 we instead proxy
        it, which works just as well.

        Two different use cases seem to require the collection either copied
        from the underlying one, or unique to this AnnotatedFromClause.

        See test_selectable->test_annotated_corresponding_column

        """
        ee = self._Annotated__element  # type: ignore
        return ee.c  # type: ignore

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\_add_newdocs.py ===
"""
This is only meant to add docs to objects defined in C-extension modules.
The purpose is to allow easier editing of the docstrings without
requiring a re-compile.

NOTE: Many of the methods of ndarray have corresponding functions.
      If you update these docstrings, please keep also the ones in
      _core/fromnumeric.py, matrixlib/defmatrix.py up-to-date.

"""

from numpy._core.function_base import add_newdoc
from numpy._core.overrides import get_array_function_like_doc  # noqa: F401

###############################################################################
#
# flatiter
#
# flatiter needs a toplevel description
#
###############################################################################

add_newdoc('numpy._core', 'flatiter',
    """
    Flat iterator object to iterate over arrays.

    A `flatiter` iterator is returned by ``x.flat`` for any array `x`.
    It allows iterating over the array as if it were a 1-D array,
    either in a for-loop or by calling its `next` method.

    Iteration is done in row-major, C-style order (the last
    index varying the fastest). The iterator can also be indexed using
    basic slicing or advanced indexing.

    See Also
    --------
    ndarray.flat : Return a flat iterator over an array.
    ndarray.flatten : Returns a flattened copy of an array.

    Notes
    -----
    A `flatiter` iterator can not be constructed directly from Python code
    by calling the `flatiter` constructor.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(6).reshape(2, 3)
    >>> fl = x.flat
    >>> type(fl)
    <class 'numpy.flatiter'>
    >>> for item in fl:
    ...     print(item)
    ...
    0
    1
    2
    3
    4
    5

    >>> fl[2:4]
    array([2, 3])

    """)

# flatiter attributes

add_newdoc('numpy._core', 'flatiter', ('base',
    """
    A reference to the array that is iterated over.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(5)
    >>> fl = x.flat
    >>> fl.base is x
    True

    """))


add_newdoc('numpy._core', 'flatiter', ('coords',
    """
    An N-dimensional tuple of current coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(6).reshape(2, 3)
    >>> fl = x.flat
    >>> fl.coords
    (0, 0)
    >>> next(fl)
    0
    >>> fl.coords
    (0, 1)

    """))


add_newdoc('numpy._core', 'flatiter', ('index',
    """
    Current flat index into the array.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(6).reshape(2, 3)
    >>> fl = x.flat
    >>> fl.index
    0
    >>> next(fl)
    0
    >>> fl.index
    1

    """))

# flatiter functions

add_newdoc('numpy._core', 'flatiter', ('__array__',
    """__array__(type=None) Get array from iterator

    """))


add_newdoc('numpy._core', 'flatiter', ('copy',
    """
    copy()

    Get a copy of the iterator as a 1-D array.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(6).reshape(2, 3)
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> fl = x.flat
    >>> fl.copy()
    array([0, 1, 2, 3, 4, 5])

    """))


###############################################################################
#
# nditer
#
###############################################################################

add_newdoc('numpy._core', 'nditer',
    """
    nditer(op, flags=None, op_flags=None, op_dtypes=None, order='K',
        casting='safe', op_axes=None, itershape=None, buffersize=0)

    Efficient multi-dimensional iterator object to iterate over arrays.
    To get started using this object, see the
    :ref:`introductory guide to array iteration <arrays.nditer>`.

    Parameters
    ----------
    op : ndarray or sequence of array_like
        The array(s) to iterate over.

    flags : sequence of str, optional
          Flags to control the behavior of the iterator.

          * ``buffered`` enables buffering when required.
          * ``c_index`` causes a C-order index to be tracked.
          * ``f_index`` causes a Fortran-order index to be tracked.
          * ``multi_index`` causes a multi-index, or a tuple of indices
            with one per iteration dimension, to be tracked.
          * ``common_dtype`` causes all the operands to be converted to
            a common data type, with copying or buffering as necessary.
          * ``copy_if_overlap`` causes the iterator to determine if read
            operands have overlap with write operands, and make temporary
            copies as necessary to avoid overlap. False positives (needless
            copying) are possible in some cases.
          * ``delay_bufalloc`` delays allocation of the buffers until
            a reset() call is made. Allows ``allocate`` operands to
            be initialized before their values are copied into the buffers.
          * ``external_loop`` causes the ``values`` given to be
            one-dimensional arrays with multiple values instead of
            zero-dimensional arrays.
          * ``grow_inner`` allows the ``value`` array sizes to be made
            larger than the buffer size when both ``buffered`` and
            ``external_loop`` is used.
          * ``ranged`` allows the iterator to be restricted to a sub-range
            of the iterindex values.
          * ``refs_ok`` enables iteration of reference types, such as
            object arrays.
          * ``reduce_ok`` enables iteration of ``readwrite`` operands
            which are broadcasted, also known as reduction operands.
          * ``zerosize_ok`` allows `itersize` to be zero.
    op_flags : list of list of str, optional
          This is a list of flags for each operand. At minimum, one of
          ``readonly``, ``readwrite``, or ``writeonly`` must be specified.

          * ``readonly`` indicates the operand will only be read from.
          * ``readwrite`` indicates the operand will be read from and written to.
          * ``writeonly`` indicates the operand will only be written to.
          * ``no_broadcast`` prevents the operand from being broadcasted.
          * ``contig`` forces the operand data to be contiguous.
          * ``aligned`` forces the operand data to be aligned.
          * ``nbo`` forces the operand data to be in native byte order.
          * ``copy`` allows a temporary read-only copy if required.
          * ``updateifcopy`` allows a temporary read-write copy if required.
          * ``allocate`` causes the array to be allocated if it is None
            in the ``op`` parameter.
          * ``no_subtype`` prevents an ``allocate`` operand from using a subtype.
          * ``arraymask`` indicates that this operand is the mask to use
            for selecting elements when writing to operands with the
            'writemasked' flag set. The iterator does not enforce this,
            but when writing from a buffer back to the array, it only
            copies those elements indicated by this mask.
          * ``writemasked`` indicates that only elements where the chosen
            ``arraymask`` operand is True will be written to.
          * ``overlap_assume_elementwise`` can be used to mark operands that are
            accessed only in the iterator order, to allow less conservative
            copying when ``copy_if_overlap`` is present.
    op_dtypes : dtype or tuple of dtype(s), optional
        The required data type(s) of the operands. If copying or buffering
        is enabled, the data will be converted to/from their original types.
    order : {'C', 'F', 'A', 'K'}, optional
        Controls the iteration order. 'C' means C order, 'F' means
        Fortran order, 'A' means 'F' order if all the arrays are Fortran
        contiguous, 'C' order otherwise, and 'K' means as close to the
        order the array elements appear in memory as possible. This also
        affects the element memory order of ``allocate`` operands, as they
        are allocated to be compatible with iteration order.
        Default is 'K'.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur when making a copy
        or buffering.  Setting this to 'unsafe' is not recommended,
        as it can adversely affect accumulations.

        * 'no' means the data types should not be cast at all.
        * 'equiv' means only byte-order changes are allowed.
        * 'safe' means only casts which can preserve values are allowed.
        * 'same_kind' means only safe casts or casts within a kind,
          like float64 to float32, are allowed.
        * 'unsafe' means any data conversions may be done.
    op_axes : list of list of ints, optional
        If provided, is a list of ints or None for each operands.
        The list of axes for an operand is a mapping from the dimensions
        of the iterator to the dimensions of the operand. A value of
        -1 can be placed for entries, causing that dimension to be
        treated as `newaxis`.
    itershape : tuple of ints, optional
        The desired shape of the iterator. This allows ``allocate`` operands
        with a dimension mapped by op_axes not corresponding to a dimension
        of a different operand to get a value not equal to 1 for that
        dimension.
    buffersize : int, optional
        When buffering is enabled, controls the size of the temporary
        buffers. Set to 0 for the default value.

    Attributes
    ----------
    dtypes : tuple of dtype(s)
        The data types of the values provided in `value`. This may be
        different from the operand data types if buffering is enabled.
        Valid only before the iterator is closed.
    finished : bool
        Whether the iteration over the operands is finished or not.
    has_delayed_bufalloc : bool
        If True, the iterator was created with the ``delay_bufalloc`` flag,
        and no reset() function was called on it yet.
    has_index : bool
        If True, the iterator was created with either the ``c_index`` or
        the ``f_index`` flag, and the property `index` can be used to
        retrieve it.
    has_multi_index : bool
        If True, the iterator was created with the ``multi_index`` flag,
        and the property `multi_index` can be used to retrieve it.
    index
        When the ``c_index`` or ``f_index`` flag was used, this property
        provides access to the index. Raises a ValueError if accessed
        and ``has_index`` is False.
    iterationneedsapi : bool
        Whether iteration requires access to the Python API, for example
        if one of the operands is an object array.
    iterindex : int
        An index which matches the order of iteration.
    itersize : int
        Size of the iterator.
    itviews
        Structured view(s) of `operands` in memory, matching the reordered
        and optimized iterator access pattern. Valid only before the iterator
        is closed.
    multi_index
        When the ``multi_index`` flag was used, this property
        provides access to the index. Raises a ValueError if accessed
        accessed and ``has_multi_index`` is False.
    ndim : int
        The dimensions of the iterator.
    nop : int
        The number of iterator operands.
    operands : tuple of operand(s)
        The array(s) to be iterated over. Valid only before the iterator is
        closed.
    shape : tuple of ints
        Shape tuple, the shape of the iterator.
    value
        Value of ``operands`` at current iteration. Normally, this is a
        tuple of array scalars, but if the flag ``external_loop`` is used,
        it is a tuple of one dimensional arrays.

    Notes
    -----
    `nditer` supersedes `flatiter`.  The iterator implementation behind
    `nditer` is also exposed by the NumPy C API.

    The Python exposure supplies two iteration interfaces, one which follows
    the Python iterator protocol, and another which mirrors the C-style
    do-while pattern.  The native Python approach is better in most cases, but
    if you need the coordinates or index of an iterator, use the C-style pattern.

    Examples
    --------
    Here is how we might write an ``iter_add`` function, using the
    Python iterator protocol:

    >>> import numpy as np

    >>> def iter_add_py(x, y, out=None):
    ...     addop = np.add
    ...     it = np.nditer([x, y, out], [],
    ...                 [['readonly'], ['readonly'], ['writeonly','allocate']])
    ...     with it:
    ...         for (a, b, c) in it:
    ...             addop(a, b, out=c)
    ...         return it.operands[2]

    Here is the same function, but following the C-style pattern:

    >>> def iter_add(x, y, out=None):
    ...    addop = np.add
    ...    it = np.nditer([x, y, out], [],
    ...                [['readonly'], ['readonly'], ['writeonly','allocate']])
    ...    with it:
    ...        while not it.finished:
    ...            addop(it[0], it[1], out=it[2])
    ...            it.iternext()
    ...        return it.operands[2]

    Here is an example outer product function:

    >>> def outer_it(x, y, out=None):
    ...     mulop = np.multiply
    ...     it = np.nditer([x, y, out], ['external_loop'],
    ...             [['readonly'], ['readonly'], ['writeonly', 'allocate']],
    ...             op_axes=[list(range(x.ndim)) + [-1] * y.ndim,
    ...                      [-1] * x.ndim + list(range(y.ndim)),
    ...                      None])
    ...     with it:
    ...         for (a, b, c) in it:
    ...             mulop(a, b, out=c)
    ...         return it.operands[2]

    >>> a = np.arange(2)+1
    >>> b = np.arange(3)+1
    >>> outer_it(a,b)
    array([[1, 2, 3],
           [2, 4, 6]])

    Here is an example function which operates like a "lambda" ufunc:

    >>> def luf(lamdaexpr, *args, **kwargs):
    ...    '''luf(lambdaexpr, op1, ..., opn, out=None, order='K', casting='safe', buffersize=0)'''
    ...    nargs = len(args)
    ...    op = (kwargs.get('out',None),) + args
    ...    it = np.nditer(op, ['buffered','external_loop'],
    ...            [['writeonly','allocate','no_broadcast']] +
    ...                            [['readonly','nbo','aligned']]*nargs,
    ...            order=kwargs.get('order','K'),
    ...            casting=kwargs.get('casting','safe'),
    ...            buffersize=kwargs.get('buffersize',0))
    ...    while not it.finished:
    ...        it[0] = lamdaexpr(*it[1:])
    ...        it.iternext()
    ...    return it.operands[0]

    >>> a = np.arange(5)
    >>> b = np.ones(5)
    >>> luf(lambda i,j:i*i + j/2, a, b)
    array([  0.5,   1.5,   4.5,   9.5,  16.5])

    If operand flags ``"writeonly"`` or ``"readwrite"`` are used the
    operands may be views into the original data with the
    `WRITEBACKIFCOPY` flag. In this case `nditer` must be used as a
    context manager or the `nditer.close` method must be called before
    using the result. The temporary data will be written back to the
    original data when the :meth:`~object.__exit__` function is called
    but not before:

    >>> a = np.arange(6, dtype='i4')[::-2]
    >>> with np.nditer(a, [],
    ...        [['writeonly', 'updateifcopy']],
    ...        casting='unsafe',
    ...        op_dtypes=[np.dtype('f4')]) as i:
    ...    x = i.operands[0]
    ...    x[:] = [-1, -2, -3]
    ...    # a still unchanged here
    >>> a, x
    (array([-1, -2, -3], dtype=int32), array([-1., -2., -3.], dtype=float32))

    It is important to note that once the iterator is exited, dangling
    references (like `x` in the example) may or may not share data with
    the original data `a`. If writeback semantics were active, i.e. if
    `x.base.flags.writebackifcopy` is `True`, then exiting the iterator
    will sever the connection between `x` and `a`, writing to `x` will
    no longer write to `a`. If writeback semantics are not active, then
    `x.data` will still point at some part of `a.data`, and writing to
    one will affect the other.

    Context management and the `close` method appeared in version 1.15.0.

    """)

# nditer methods

add_newdoc('numpy._core', 'nditer', ('copy',
    """
    copy()

    Get a copy of the iterator in its current state.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(10)
    >>> y = x + 1
    >>> it = np.nditer([x, y])
    >>> next(it)
    (array(0), array(1))
    >>> it2 = it.copy()
    >>> next(it2)
    (array(1), array(2))

    """))

add_newdoc('numpy._core', 'nditer', ('operands',
    """
    operands[`Slice`]

    The array(s) to be iterated over. Valid only before the iterator is closed.
    """))

add_newdoc('numpy._core', 'nditer', ('debug_print',
    """
    debug_print()

    Print the current state of the `nditer` instance and debug info to stdout.

    """))

add_newdoc('numpy._core', 'nditer', ('enable_external_loop',
    """
    enable_external_loop()

    When the "external_loop" was not used during construction, but
    is desired, this modifies the iterator to behave as if the flag
    was specified.

    """))

add_newdoc('numpy._core', 'nditer', ('iternext',
    """
    iternext()

    Check whether iterations are left, and perform a single internal iteration
    without returning the result.  Used in the C-style pattern do-while
    pattern.  For an example, see `nditer`.

    Returns
    -------
    iternext : bool
        Whether or not there are iterations left.

    """))

add_newdoc('numpy._core', 'nditer', ('remove_axis',
    """
    remove_axis(i, /)

    Removes axis `i` from the iterator. Requires that the flag "multi_index"
    be enabled.

    """))

add_newdoc('numpy._core', 'nditer', ('remove_multi_index',
    """
    remove_multi_index()

    When the "multi_index" flag was specified, this removes it, allowing
    the internal iteration structure to be optimized further.

    """))

add_newdoc('numpy._core', 'nditer', ('reset',
    """
    reset()

    Reset the iterator to its initial state.

    """))

add_newdoc('numpy._core', 'nested_iters',
    """
    nested_iters(op, axes, flags=None, op_flags=None, op_dtypes=None, \
    order="K", casting="safe", buffersize=0)

    Create nditers for use in nested loops

    Create a tuple of `nditer` objects which iterate in nested loops over
    different axes of the op argument. The first iterator is used in the
    outermost loop, the last in the innermost loop. Advancing one will change
    the subsequent iterators to point at its new element.

    Parameters
    ----------
    op : ndarray or sequence of array_like
        The array(s) to iterate over.

    axes : list of list of int
        Each item is used as an "op_axes" argument to an nditer

    flags, op_flags, op_dtypes, order, casting, buffersize (optional)
        See `nditer` parameters of the same name

    Returns
    -------
    iters : tuple of nditer
        An nditer for each item in `axes`, outermost first

    See Also
    --------
    nditer

    Examples
    --------

    Basic usage. Note how y is the "flattened" version of
    [a[:, 0, :], a[:, 1, 0], a[:, 2, :]] since we specified
    the first iter's axes as [1]

    >>> import numpy as np
    >>> a = np.arange(12).reshape(2, 3, 2)
    >>> i, j = np.nested_iters(a, [[1], [0, 2]], flags=["multi_index"])
    >>> for x in i:
    ...      print(i.multi_index)
    ...      for y in j:
    ...          print('', j.multi_index, y)
    (0,)
     (0, 0) 0
     (0, 1) 1
     (1, 0) 6
     (1, 1) 7
    (1,)
     (0, 0) 2
     (0, 1) 3
     (1, 0) 8
     (1, 1) 9
    (2,)
     (0, 0) 4
     (0, 1) 5
     (1, 0) 10
     (1, 1) 11

    """)

add_newdoc('numpy._core', 'nditer', ('close',
    """
    close()

    Resolve all writeback semantics in writeable operands.

    See Also
    --------

    :ref:`nditer-context-manager`

    """))


###############################################################################
#
# broadcast
#
###############################################################################

add_newdoc('numpy._core', 'broadcast',
    """
    Produce an object that mimics broadcasting.

    Parameters
    ----------
    in1, in2, ... : array_like
        Input parameters.

    Returns
    -------
    b : broadcast object
        Broadcast the input parameters against one another, and
        return an object that encapsulates the result.
        Amongst others, it has ``shape`` and ``nd`` properties, and
        may be used as an iterator.

    See Also
    --------
    broadcast_arrays
    broadcast_to
    broadcast_shapes

    Examples
    --------

    Manually adding two vectors, using broadcasting:

    >>> import numpy as np
    >>> x = np.array([[1], [2], [3]])
    >>> y = np.array([4, 5, 6])
    >>> b = np.broadcast(x, y)

    >>> out = np.empty(b.shape)
    >>> out.flat = [u+v for (u,v) in b]
    >>> out
    array([[5.,  6.,  7.],
           [6.,  7.,  8.],
           [7.,  8.,  9.]])

    Compare against built-in broadcasting:

    >>> x + y
    array([[5, 6, 7],
           [6, 7, 8],
           [7, 8, 9]])

    """)

# attributes

add_newdoc('numpy._core', 'broadcast', ('index',
    """
    current index in broadcasted result

    Examples
    --------

    >>> import numpy as np
    >>> x = np.array([[1], [2], [3]])
    >>> y = np.array([4, 5, 6])
    >>> b = np.broadcast(x, y)
    >>> b.index
    0
    >>> next(b), next(b), next(b)
    ((1, 4), (1, 5), (1, 6))
    >>> b.index
    3

    """))

add_newdoc('numpy._core', 'broadcast', ('iters',
    """
    tuple of iterators along ``self``'s "components."

    Returns a tuple of `numpy.flatiter` objects, one for each "component"
    of ``self``.

    See Also
    --------
    numpy.flatiter

    Examples
    --------

    >>> import numpy as np
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([[4], [5], [6]])
    >>> b = np.broadcast(x, y)
    >>> row, col = b.iters
    >>> next(row), next(col)
    (1, 4)

    """))

add_newdoc('numpy._core', 'broadcast', ('ndim',
    """
    Number of dimensions of broadcasted result. Alias for `nd`.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([[4], [5], [6]])
    >>> b = np.broadcast(x, y)
    >>> b.ndim
    2

    """))

add_newdoc('numpy._core', 'broadcast', ('nd',
    """
    Number of dimensions of broadcasted result. For code intended for NumPy
    1.12.0 and later the more consistent `ndim` is preferred.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([[4], [5], [6]])
    >>> b = np.broadcast(x, y)
    >>> b.nd
    2

    """))

add_newdoc('numpy._core', 'broadcast', ('numiter',
    """
    Number of iterators possessed by the broadcasted result.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([[4], [5], [6]])
    >>> b = np.broadcast(x, y)
    >>> b.numiter
    2

    """))

add_newdoc('numpy._core', 'broadcast', ('shape',
    """
    Shape of broadcasted result.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([[4], [5], [6]])
    >>> b = np.broadcast(x, y)
    >>> b.shape
    (3, 3)

    """))

add_newdoc('numpy._core', 'broadcast', ('size',
    """
    Total size of broadcasted result.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([[4], [5], [6]])
    >>> b = np.broadcast(x, y)
    >>> b.size
    9

    """))

add_newdoc('numpy._core', 'broadcast', ('reset',
    """
    reset()

    Reset the broadcasted result's iterator(s).

    Parameters
    ----------
    None

    Returns
    -------
    None

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([[4], [5], [6]])
    >>> b = np.broadcast(x, y)
    >>> b.index
    0
    >>> next(b), next(b), next(b)
    ((1, 4), (2, 4), (3, 4))
    >>> b.index
    3
    >>> b.reset()
    >>> b.index
    0

    """))

###############################################################################
#
# numpy functions
#
###############################################################################

add_newdoc('numpy._core.multiarray', 'array',
    """
    array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0,
          like=None)

    Create an array.

    Parameters
    ----------
    object : array_like
        An array, any object exposing the array interface, an object whose
        ``__array__`` method returns an array, or any (nested) sequence.
        If object is a scalar, a 0-dimensional array containing object is
        returned.
    dtype : data-type, optional
        The desired data-type for the array. If not given, NumPy will try to use
        a default ``dtype`` that can represent the values (by applying promotion
        rules when necessary.)
    copy : bool, optional
        If ``True`` (default), then the array data is copied. If ``None``,
        a copy will only be made if ``__array__`` returns a copy, if obj is
        a nested sequence, or if a copy is needed to satisfy any of the other
        requirements (``dtype``, ``order``, etc.). Note that any copy of
        the data is shallow, i.e., for arrays with object dtype, the new
        array will point to the same objects. See Examples for `ndarray.copy`.
        For ``False`` it raises a ``ValueError`` if a copy cannot be avoided.
        Default: ``True``.
    order : {'K', 'A', 'C', 'F'}, optional
        Specify the memory layout of the array. If object is not an array, the
        newly created array will be in C order (row major) unless 'F' is
        specified, in which case it will be in Fortran order (column major).
        If object is an array the following holds.

        ===== ========= ===================================================
        order  no copy                     copy=True
        ===== ========= ===================================================
        'K'   unchanged F & C order preserved, otherwise most similar order
        'A'   unchanged F order if input is F and not C, otherwise C order
        'C'   C order   C order
        'F'   F order   F order
        ===== ========= ===================================================

        When ``copy=None`` and a copy is made for other reasons, the result is
        the same as if ``copy=True``, with some exceptions for 'A', see the
        Notes section. The default order is 'K'.
    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise
        the returned array will be forced to be a base-class array (default).
    ndmin : int, optional
        Specifies the minimum number of dimensions that the resulting
        array should have.  Ones will be prepended to the shape as
        needed to meet this requirement.
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        An array object satisfying the specified requirements.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    full_like : Return a new array with shape of input filled with value.
    empty : Return a new uninitialized array.
    ones : Return a new array setting values to one.
    zeros : Return a new array setting values to zero.
    full : Return a new array of given shape filled with value.
    copy: Return an array copy of the given object.


    Notes
    -----
    When order is 'A' and ``object`` is an array in neither 'C' nor 'F' order,
    and a copy is forced by a change in dtype, then the order of the result is
    not necessarily 'C' as expected. This is likely a bug.

    Examples
    --------
    >>> import numpy as np
    >>> np.array([1, 2, 3])
    array([1, 2, 3])

    Upcasting:

    >>> np.array([1, 2, 3.0])
    array([ 1.,  2.,  3.])

    More than one dimension:

    >>> np.array([[1, 2], [3, 4]])
    array([[1, 2],
           [3, 4]])

    Minimum dimensions 2:

    >>> np.array([1, 2, 3], ndmin=2)
    array([[1, 2, 3]])

    Type provided:

    >>> np.array([1, 2, 3], dtype=complex)
    array([ 1.+0.j,  2.+0.j,  3.+0.j])

    Data-type consisting of more than one element:

    >>> x = np.array([(1,2),(3,4)],dtype=[('a','<i4'),('b','<i4')])
    >>> x['a']
    array([1, 3], dtype=int32)

    Creating an array from sub-classes:

    >>> np.array(np.asmatrix('1 2; 3 4'))
    array([[1, 2],
           [3, 4]])

    >>> np.array(np.asmatrix('1 2; 3 4'), subok=True)
    matrix([[1, 2],
            [3, 4]])

    """)

add_newdoc('numpy._core.multiarray', 'asarray',
    """
    asarray(a, dtype=None, order=None, *, device=None, copy=None, like=None)

    Convert the input to an array.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array.  This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists and ndarrays.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.
    order : {'C', 'F', 'A', 'K'}, optional
        Memory layout.  'A' and 'K' depend on the order of input array a.
        'C' row-major (C-style),
        'F' column-major (Fortran-style) memory representation.
        'A' (any) means 'F' if `a` is Fortran contiguous, 'C' otherwise
        'K' (keep) preserve input order
        Defaults to 'K'.
    device : str, optional
        The device on which to place the created array. Default: ``None``.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0
    copy : bool, optional
        If ``True``, then the object is copied. If ``None`` then the object is
        copied only if needed, i.e. if ``__array__`` returns a copy, if obj
        is a nested sequence, or if a copy is needed to satisfy any of
        the other requirements (``dtype``, ``order``, etc.).
        For ``False`` it raises a ``ValueError`` if a copy cannot be avoided.
        Default: ``None``.

        .. versionadded:: 2.0.0
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        Array interpretation of ``a``.  No copy is performed if the input
        is already an ndarray with matching dtype and order.  If ``a`` is a
        subclass of ndarray, a base class ndarray is returned.

    See Also
    --------
    asanyarray : Similar function which passes through subclasses.
    ascontiguousarray : Convert input to a contiguous array.
    asfortranarray : Convert input to an ndarray with column-major
                     memory order.
    asarray_chkfinite : Similar function which checks input for NaNs and Infs.
    fromiter : Create an array from an iterator.
    fromfunction : Construct an array by executing a function on grid
                   positions.

    Examples
    --------
    Convert a list into an array:

    >>> a = [1, 2]
    >>> import numpy as np
    >>> np.asarray(a)
    array([1, 2])

    Existing arrays are not copied:

    >>> a = np.array([1, 2])
    >>> np.asarray(a) is a
    True

    If `dtype` is set, array is copied only if dtype does not match:

    >>> a = np.array([1, 2], dtype=np.float32)
    >>> np.shares_memory(np.asarray(a, dtype=np.float32), a)
    True
    >>> np.shares_memory(np.asarray(a, dtype=np.float64), a)
    False

    Contrary to `asanyarray`, ndarray subclasses are not passed through:

    >>> issubclass(np.recarray, np.ndarray)
    True
    >>> a = np.array([(1., 2), (3., 4)], dtype='f4,i4').view(np.recarray)
    >>> np.asarray(a) is a
    False
    >>> np.asanyarray(a) is a
    True

    """)

add_newdoc('numpy._core.multiarray', 'asanyarray',
    """
    asanyarray(a, dtype=None, order=None, *, device=None, copy=None, like=None)

    Convert the input to an ndarray, but pass ndarray subclasses through.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array.  This
        includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        tuples of lists, and ndarrays.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.
    order : {'C', 'F', 'A', 'K'}, optional
        Memory layout.  'A' and 'K' depend on the order of input array a.
        'C' row-major (C-style),
        'F' column-major (Fortran-style) memory representation.
        'A' (any) means 'F' if `a` is Fortran contiguous, 'C' otherwise
        'K' (keep) preserve input order
        Defaults to 'C'.
    device : str, optional
        The device on which to place the created array. Default: ``None``.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.1.0

    copy : bool, optional
        If ``True``, then the object is copied. If ``None`` then the object is
        copied only if needed, i.e. if ``__array__`` returns a copy, if obj
        is a nested sequence, or if a copy is needed to satisfy any of
        the other requirements (``dtype``, ``order``, etc.).
        For ``False`` it raises a ``ValueError`` if a copy cannot be avoided.
        Default: ``None``.

        .. versionadded:: 2.1.0

    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray or an ndarray subclass
        Array interpretation of `a`.  If `a` is an ndarray or a subclass
        of ndarray, it is returned as-is and no copy is performed.

    See Also
    --------
    asarray : Similar function which always returns ndarrays.
    ascontiguousarray : Convert input to a contiguous array.
    asfortranarray : Convert input to an ndarray with column-major
                     memory order.
    asarray_chkfinite : Similar function which checks input for NaNs and
                        Infs.
    fromiter : Create an array from an iterator.
    fromfunction : Construct an array by executing a function on grid
                   positions.

    Examples
    --------
    Convert a list into an array:

    >>> a = [1, 2]
    >>> import numpy as np
    >>> np.asanyarray(a)
    array([1, 2])

    Instances of `ndarray` subclasses are passed through as-is:

    >>> a = np.array([(1., 2), (3., 4)], dtype='f4,i4').view(np.recarray)
    >>> np.asanyarray(a) is a
    True

    """)

add_newdoc('numpy._core.multiarray', 'ascontiguousarray',
    """
    ascontiguousarray(a, dtype=None, *, like=None)

    Return a contiguous array (ndim >= 1) in memory (C order).

    Parameters
    ----------
    a : array_like
        Input array.
    dtype : str or dtype object, optional
        Data-type of returned array.
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        Contiguous array of same shape and content as `a`, with type `dtype`
        if specified.

    See Also
    --------
    asfortranarray : Convert input to an ndarray with column-major
                     memory order.
    require : Return an ndarray that satisfies requirements.
    ndarray.flags : Information about the memory layout of the array.

    Examples
    --------
    Starting with a Fortran-contiguous array:

    >>> import numpy as np
    >>> x = np.ones((2, 3), order='F')
    >>> x.flags['F_CONTIGUOUS']
    True

    Calling ``ascontiguousarray`` makes a C-contiguous copy:

    >>> y = np.ascontiguousarray(x)
    >>> y.flags['C_CONTIGUOUS']
    True
    >>> np.may_share_memory(x, y)
    False

    Now, starting with a C-contiguous array:

    >>> x = np.ones((2, 3), order='C')
    >>> x.flags['C_CONTIGUOUS']
    True

    Then, calling ``ascontiguousarray`` returns the same object:

    >>> y = np.ascontiguousarray(x)
    >>> x is y
    True

    Note: This function returns an array with at least one-dimension (1-d)
    so it will not preserve 0-d arrays.

    """)

add_newdoc('numpy._core.multiarray', 'asfortranarray',
    """
    asfortranarray(a, dtype=None, *, like=None)

    Return an array (ndim >= 1) laid out in Fortran order in memory.

    Parameters
    ----------
    a : array_like
        Input array.
    dtype : str or dtype object, optional
        By default, the data-type is inferred from the input data.
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        The input `a` in Fortran, or column-major, order.

    See Also
    --------
    ascontiguousarray : Convert input to a contiguous (C order) array.
    asanyarray : Convert input to an ndarray with either row or
        column-major memory order.
    require : Return an ndarray that satisfies requirements.
    ndarray.flags : Information about the memory layout of the array.

    Examples
    --------
    Starting with a C-contiguous array:

    >>> import numpy as np
    >>> x = np.ones((2, 3), order='C')
    >>> x.flags['C_CONTIGUOUS']
    True

    Calling ``asfortranarray`` makes a Fortran-contiguous copy:

    >>> y = np.asfortranarray(x)
    >>> y.flags['F_CONTIGUOUS']
    True
    >>> np.may_share_memory(x, y)
    False

    Now, starting with a Fortran-contiguous array:

    >>> x = np.ones((2, 3), order='F')
    >>> x.flags['F_CONTIGUOUS']
    True

    Then, calling ``asfortranarray`` returns the same object:

    >>> y = np.asfortranarray(x)
    >>> x is y
    True

    Note: This function returns an array with at least one-dimension (1-d)
    so it will not preserve 0-d arrays.

    """)

add_newdoc('numpy._core.multiarray', 'empty',
    """
    empty(shape, dtype=float, order='C', *, device=None, like=None)

    Return a new array of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        Desired output data-type for the array, e.g, `numpy.int8`. Default is
        `numpy.float64`.
    order : {'C', 'F'}, optional, default: 'C'
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.
    device : str, optional
        The device on which to place the created array. Default: ``None``.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        Array of uninitialized (arbitrary) data of the given shape, dtype, and
        order.  Object arrays will be initialized to None.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    ones : Return a new array setting values to one.
    zeros : Return a new array setting values to zero.
    full : Return a new array of given shape filled with value.

    Notes
    -----
    Unlike other array creation functions (e.g. `zeros`, `ones`, `full`),
    `empty` does not initialize the values of the array, and may therefore be
    marginally faster. However, the values stored in the newly allocated array
    are arbitrary. For reproducible behavior, be sure to set each element of
    the array before reading.

    Examples
    --------
    >>> import numpy as np
    >>> np.empty([2, 2])
    array([[ -9.74499359e+001,   6.69583040e-309],
           [  2.13182611e-314,   3.06959433e-309]])         #uninitialized

    >>> np.empty([2, 2], dtype=int)
    array([[-1073741821, -1067949133],
           [  496041986,    19249760]])                     #uninitialized

    """)

add_newdoc('numpy._core.multiarray', 'scalar',
    """
    scalar(dtype, obj)

    Return a new scalar array of the given type initialized with obj.

    This function is meant mainly for pickle support. `dtype` must be a
    valid data-type descriptor. If `dtype` corresponds to an object
    descriptor, then `obj` can be any object, otherwise `obj` must be a
    string. If `obj` is not given, it will be interpreted as None for object
    type and as zeros for all other types.

    """)

add_newdoc('numpy._core.multiarray', 'zeros',
    """
    zeros(shape, dtype=float, order='C', *, like=None)

    Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    order : {'C', 'F'}, optional, default: 'C'
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        Array of zeros with the given shape, dtype, and order.

    See Also
    --------
    zeros_like : Return an array of zeros with shape and type of input.
    empty : Return a new uninitialized array.
    ones : Return a new array setting values to one.
    full : Return a new array of given shape filled with value.

    Examples
    --------
    >>> import numpy as np
    >>> np.zeros(5)
    array([ 0.,  0.,  0.,  0.,  0.])

    >>> np.zeros((5,), dtype=int)
    array([0, 0, 0, 0, 0])

    >>> np.zeros((2, 1))
    array([[ 0.],
           [ 0.]])

    >>> s = (2,2)
    >>> np.zeros(s)
    array([[ 0.,  0.],
           [ 0.,  0.]])

    >>> np.zeros((2,), dtype=[('x', 'i4'), ('y', 'i4')]) # custom dtype
    array([(0, 0), (0, 0)],
          dtype=[('x', '<i4'), ('y', '<i4')])

    """)

add_newdoc('numpy._core.multiarray', 'set_typeDict',
    """set_typeDict(dict)

    Set the internal dictionary that can look up an array type using a
    registered code.

    """)

add_newdoc('numpy._core.multiarray', 'fromstring',
    """
    fromstring(string, dtype=float, count=-1, *, sep, like=None)

    A new 1-D array initialized from text data in a string.

    Parameters
    ----------
    string : str
        A string containing the data.
    dtype : data-type, optional
        The data type of the array; default: float.  For binary input data,
        the data must be in exactly this format. Most builtin numeric types are
        supported and extension types may be supported.
    count : int, optional
        Read this number of `dtype` elements from the data.  If this is
        negative (the default), the count will be determined from the
        length of the data.
    sep : str, optional
        The string separating numbers in the data; extra whitespace between
        elements is also ignored.

        .. deprecated:: 1.14
            Passing ``sep=''``, the default, is deprecated since it will
            trigger the deprecated binary mode of this function. This mode
            interprets `string` as binary bytes, rather than ASCII text with
            decimal numbers, an operation which is better spelt
            ``frombuffer(string, dtype, count)``. If `string` contains unicode
            text, the binary mode of `fromstring` will first encode it into
            bytes using utf-8, which will not produce sane results.

    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    arr : ndarray
        The constructed array.

    Raises
    ------
    ValueError
        If the string is not the correct size to satisfy the requested
        `dtype` and `count`.

    See Also
    --------
    frombuffer, fromfile, fromiter

    Examples
    --------
    >>> import numpy as np
    >>> np.fromstring('1 2', dtype=int, sep=' ')
    array([1, 2])
    >>> np.fromstring('1, 2', dtype=int, sep=',')
    array([1, 2])

    """)

add_newdoc('numpy._core.multiarray', 'compare_chararrays',
    """
    compare_chararrays(a1, a2, cmp, rstrip)

    Performs element-wise comparison of two string arrays using the
    comparison operator specified by `cmp`.

    Parameters
    ----------
    a1, a2 : array_like
        Arrays to be compared.
    cmp : {"<", "<=", "==", ">=", ">", "!="}
        Type of comparison.
    rstrip : Boolean
        If True, the spaces at the end of Strings are removed before the comparison.

    Returns
    -------
    out : ndarray
        The output array of type Boolean with the same shape as a and b.

    Raises
    ------
    ValueError
        If `cmp` is not valid.
    TypeError
        If at least one of `a` or `b` is a non-string array

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(["a", "b", "cde"])
    >>> b = np.array(["a", "a", "dec"])
    >>> np.char.compare_chararrays(a, b, ">", True)
    array([False,  True, False])

    """)

add_newdoc('numpy._core.multiarray', 'fromiter',
    """
    fromiter(iter, dtype, count=-1, *, like=None)

    Create a new 1-dimensional array from an iterable object.

    Parameters
    ----------
    iter : iterable object
        An iterable object providing data for the array.
    dtype : data-type
        The data-type of the returned array.

        .. versionchanged:: 1.23
            Object and subarray dtypes are now supported (note that the final
            result is not 1-D for a subarray dtype).

    count : int, optional
        The number of items to read from *iterable*.  The default is -1,
        which means all data is read.
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        The output array.

    Notes
    -----
    Specify `count` to improve performance.  It allows ``fromiter`` to
    pre-allocate the output array, instead of resizing it on demand.

    Examples
    --------
    >>> import numpy as np
    >>> iterable = (x*x for x in range(5))
    >>> np.fromiter(iterable, float)
    array([  0.,   1.,   4.,   9.,  16.])

    A carefully constructed subarray dtype will lead to higher dimensional
    results:

    >>> iterable = ((x+1, x+2) for x in range(5))
    >>> np.fromiter(iterable, dtype=np.dtype((int, 2)))
    array([[1, 2],
           [2, 3],
           [3, 4],
           [4, 5],
           [5, 6]])


    """)

add_newdoc('numpy._core.multiarray', 'fromfile',
    """
    fromfile(file, dtype=float, count=-1, sep='', offset=0, *, like=None)

    Construct an array from data in a text or binary file.

    A highly efficient way of reading binary data with a known data-type,
    as well as parsing simply formatted text files.  Data written using the
    `tofile` method can be read using this function.

    Parameters
    ----------
    file : file or str or Path
        Open file object or filename.
    dtype : data-type
        Data type of the returned array.
        For binary files, it is used to determine the size and byte-order
        of the items in the file.
        Most builtin numeric types are supported and extension types may be supported.
    count : int
        Number of items to read. ``-1`` means all items (i.e., the complete
        file).
    sep : str
        Separator between items if file is a text file.
        Empty ("") separator means the file should be treated as binary.
        Spaces (" ") in the separator match zero or more whitespace characters.
        A separator consisting only of spaces must match at least one
        whitespace.
    offset : int
        The offset (in bytes) from the file's current position. Defaults to 0.
        Only permitted for binary files.
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    See also
    --------
    load, save
    ndarray.tofile
    loadtxt : More flexible way of loading data from a text file.

    Notes
    -----
    Do not rely on the combination of `tofile` and `fromfile` for
    data storage, as the binary files generated are not platform
    independent.  In particular, no byte-order or data-type information is
    saved.  Data can be stored in the platform independent ``.npy`` format
    using `save` and `load` instead.

    Examples
    --------
    Construct an ndarray:

    >>> import numpy as np
    >>> dt = np.dtype([('time', [('min', np.int64), ('sec', np.int64)]),
    ...                ('temp', float)])
    >>> x = np.zeros((1,), dtype=dt)
    >>> x['time']['min'] = 10; x['temp'] = 98.25
    >>> x
    array([((10, 0), 98.25)],
          dtype=[('time', [('min', '<i8'), ('sec', '<i8')]), ('temp', '<f8')])

    Save the raw data to disk:

    >>> import tempfile
    >>> fname = tempfile.mkstemp()[1]
    >>> x.tofile(fname)

    Read the raw data from disk:

    >>> np.fromfile(fname, dtype=dt)
    array([((10, 0), 98.25)],
          dtype=[('time', [('min', '<i8'), ('sec', '<i8')]), ('temp', '<f8')])

    The recommended way to store and load data:

    >>> np.save(fname, x)
    >>> np.load(fname + '.npy')
    array([((10, 0), 98.25)],
          dtype=[('time', [('min', '<i8'), ('sec', '<i8')]), ('temp', '<f8')])

    """)

add_newdoc('numpy._core.multiarray', 'frombuffer',
    """
    frombuffer(buffer, dtype=float, count=-1, offset=0, *, like=None)

    Interpret a buffer as a 1-dimensional array.

    Parameters
    ----------
    buffer : buffer_like
        An object that exposes the buffer interface.
    dtype : data-type, optional
        Data-type of the returned array; default: float.
    count : int, optional
        Number of items to read. ``-1`` means all data in the buffer.
    offset : int, optional
        Start reading the buffer from this offset (in bytes); default: 0.
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray

    See also
    --------
    ndarray.tobytes
        Inverse of this operation, construct Python bytes from the raw data
        bytes in the array.

    Notes
    -----
    If the buffer has data that is not in machine byte-order, this should
    be specified as part of the data-type, e.g.::

      >>> dt = np.dtype(int)
      >>> dt = dt.newbyteorder('>')
      >>> np.frombuffer(buf, dtype=dt) # doctest: +SKIP

    The data of the resulting array will not be byteswapped, but will be
    interpreted correctly.

    This function creates a view into the original object.  This should be safe
    in general, but it may make sense to copy the result when the original
    object is mutable or untrusted.

    Examples
    --------
    >>> import numpy as np
    >>> s = b'hello world'
    >>> np.frombuffer(s, dtype='S1', count=5, offset=6)
    array([b'w', b'o', b'r', b'l', b'd'], dtype='|S1')

    >>> np.frombuffer(b'\\x01\\x02', dtype=np.uint8)
    array([1, 2], dtype=uint8)
    >>> np.frombuffer(b'\\x01\\x02\\x03\\x04\\x05', dtype=np.uint8, count=3)
    array([1, 2, 3], dtype=uint8)

    """)

add_newdoc('numpy._core.multiarray', 'from_dlpack',
    """
    from_dlpack(x, /, *, device=None, copy=None)

    Create a NumPy array from an object implementing the ``__dlpack__``
    protocol. Generally, the returned NumPy array is a view of the input
    object. See [1]_ and [2]_ for more details.

    Parameters
    ----------
    x : object
        A Python object that implements the ``__dlpack__`` and
        ``__dlpack_device__`` methods.
    device : device, optional
        Device on which to place the created array. Default: ``None``.
        Must be ``"cpu"`` if passed which may allow importing an array
        that is not already CPU available.
    copy : bool, optional
        Boolean indicating whether or not to copy the input. If ``True``,
        the copy will be made. If ``False``, the function will never copy,
        and will raise ``BufferError`` in case a copy is deemed necessary.
        Passing it requests a copy from the exporter who may or may not
        implement the capability.
        If ``None``, the function will reuse the existing memory buffer if
        possible and copy otherwise. Default: ``None``.


    Returns
    -------
    out : ndarray

    References
    ----------
    .. [1] Array API documentation,
       https://data-apis.org/array-api/latest/design_topics/data_interchange.html#syntax-for-data-interchange-with-dlpack

    .. [2] Python specification for DLPack,
       https://dmlc.github.io/dlpack/latest/python_spec.html

    Examples
    --------
    >>> import torch  # doctest: +SKIP
    >>> x = torch.arange(10)  # doctest: +SKIP
    >>> # create a view of the torch tensor "x" in NumPy
    >>> y = np.from_dlpack(x)  # doctest: +SKIP
    """)

add_newdoc('numpy._core.multiarray', 'correlate',
    """cross_correlate(a,v, mode=0)""")

add_newdoc('numpy._core.multiarray', 'arange',
    """
    arange([start,] stop[, step,], dtype=None, *, device=None, like=None)

    Return evenly spaced values within a given interval.

    ``arange`` can be called with a varying number of positional arguments:

    * ``arange(stop)``: Values are generated within the half-open interval
      ``[0, stop)`` (in other words, the interval including `start` but
      excluding `stop`).
    * ``arange(start, stop)``: Values are generated within the half-open
      interval ``[start, stop)``.
    * ``arange(start, stop, step)`` Values are generated within the half-open
      interval ``[start, stop)``, with spacing between values given by
      ``step``.

    For integer arguments the function is roughly equivalent to the Python
    built-in :py:class:`range`, but returns an ndarray rather than a ``range``
    instance.

    When using a non-integer step, such as 0.1, it is often better to use
    `numpy.linspace`.

    See the Warning sections below for more information.

    Parameters
    ----------
    start : integer or real, optional
        Start of interval.  The interval includes this value.  The default
        start value is 0.
    stop : integer or real
        End of interval.  The interval does not include this value, except
        in some cases where `step` is not an integer and floating point
        round-off affects the length of `out`.
    step : integer or real, optional
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified as a position argument,
        `start` must also be given.
    dtype : dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.
    device : str, optional
        The device on which to place the created array. Default: ``None``.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    arange : ndarray
        Array of evenly spaced values.

        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.

    Warnings
    --------
    The length of the output might not be numerically stable.

    Another stability issue is due to the internal implementation of
    `numpy.arange`.
    The actual step value used to populate the array is
    ``dtype(start + step) - dtype(start)`` and not `step`. Precision loss
    can occur here, due to casting or due to using floating points when
    `start` is much larger than `step`. This can lead to unexpected
    behaviour. For example::

      >>> np.arange(0, 5, 0.5, dtype=int)
      array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
      >>> np.arange(-3, 3, 0.5, dtype=int)
      array([-3, -2, -1,  0,  1,  2,  3,  4,  5,  6,  7,  8])

    In such cases, the use of `numpy.linspace` should be preferred.

    The built-in :py:class:`range` generates :std:doc:`Python built-in integers
    that have arbitrary size <python:c-api/long>`, while `numpy.arange`
    produces `numpy.int32` or `numpy.int64` numbers. This may result in
    incorrect results for large integer values::

      >>> power = 40
      >>> modulo = 10000
      >>> x1 = [(n ** power) % modulo for n in range(8)]
      >>> x2 = [(n ** power) % modulo for n in np.arange(8)]
      >>> print(x1)
      [0, 1, 7776, 8801, 6176, 625, 6576, 4001]  # correct
      >>> print(x2)
      [0, 1, 7776, 7185, 0, 5969, 4816, 3361]  # incorrect

    See Also
    --------
    numpy.linspace : Evenly spaced numbers with careful handling of endpoints.
    numpy.ogrid: Arrays of evenly spaced numbers in N-dimensions.
    numpy.mgrid: Grid-shaped arrays of evenly spaced numbers in N-dimensions.
    :ref:`how-to-partition`

    Examples
    --------
    >>> import numpy as np
    >>> np.arange(3)
    array([0, 1, 2])
    >>> np.arange(3.0)
    array([ 0.,  1.,  2.])
    >>> np.arange(3,7)
    array([3, 4, 5, 6])
    >>> np.arange(3,7,2)
    array([3, 5])

    """)

add_newdoc('numpy._core.multiarray', '_get_ndarray_c_version',
    """_get_ndarray_c_version()

    Return the compile time NPY_VERSION (formerly called NDARRAY_VERSION) number.

    """)

add_newdoc('numpy._core.multiarray', '_reconstruct',
    """_reconstruct(subtype, shape, dtype)

    Construct an empty array. Used by Pickles.

    """)

add_newdoc('numpy._core.multiarray', 'promote_types',
    """
    promote_types(type1, type2)

    Returns the data type with the smallest size and smallest scalar
    kind to which both ``type1`` and ``type2`` may be safely cast.
    The returned data type is always considered "canonical", this mainly
    means that the promoted dtype will always be in native byte order.

    This function is symmetric, but rarely associative.

    Parameters
    ----------
    type1 : dtype or dtype specifier
        First data type.
    type2 : dtype or dtype specifier
        Second data type.

    Returns
    -------
    out : dtype
        The promoted data type.

    Notes
    -----
    Please see `numpy.result_type` for additional information about promotion.

    Starting in NumPy 1.9, promote_types function now returns a valid string
    length when given an integer or float dtype as one argument and a string
    dtype as another argument. Previously it always returned the input string
    dtype, even if it wasn't long enough to store the max integer/float value
    converted to a string.

    .. versionchanged:: 1.23.0

    NumPy now supports promotion for more structured dtypes.  It will now
    remove unnecessary padding from a structure dtype and promote included
    fields individually.

    See Also
    --------
    result_type, dtype, can_cast

    Examples
    --------
    >>> import numpy as np
    >>> np.promote_types('f4', 'f8')
    dtype('float64')

    >>> np.promote_types('i8', 'f4')
    dtype('float64')

    >>> np.promote_types('>i8', '<c8')
    dtype('complex128')

    >>> np.promote_types('i4', 'S8')
    dtype('S11')

    An example of a non-associative case:

    >>> p = np.promote_types
    >>> p('S', p('i1', 'u1'))
    dtype('S6')
    >>> p(p('S', 'i1'), 'u1')
    dtype('S4')

    """)

add_newdoc('numpy._core.multiarray', 'c_einsum',
    """
    c_einsum(subscripts, *operands, out=None, dtype=None, order='K',
           casting='safe')

    *This documentation shadows that of the native python implementation of the `einsum` function,
    except all references and examples related to the `optimize` argument (v 0.12.0) have been removed.*

    Evaluates the Einstein summation convention on the operands.

    Using the Einstein summation convention, many common multi-dimensional,
    linear algebraic array operations can be represented in a simple fashion.
    In *implicit* mode `einsum` computes these values.

    In *explicit* mode, `einsum` provides further flexibility to compute
    other array operations that might not be considered classical Einstein
    summation operations, by disabling, or forcing summation over specified
    subscript labels.

    See the notes and examples for clarification.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation as comma separated list of
        subscript labels. An implicit (classical Einstein summation)
        calculation is performed unless the explicit indicator '->' is
        included as well as subscript labels of the precise output form.
    operands : list of array_like
        These are the arrays for the operation.
    out : ndarray, optional
        If provided, the calculation is done into this array.
    dtype : {data-type, None}, optional
        If provided, forces the calculation to use the data type specified.
        Note that you may have to also give a more liberal `casting`
        parameter to allow the conversions. Default is None.
    order : {'C', 'F', 'A', 'K'}, optional
        Controls the memory layout of the output. 'C' means it should
        be C contiguous. 'F' means it should be Fortran contiguous,
        'A' means it should be 'F' if the inputs are all 'F', 'C' otherwise.
        'K' means it should be as close to the layout of the inputs as
        is possible, including arbitrarily permuted axes.
        Default is 'K'.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur.  Setting this to
        'unsafe' is not recommended, as it can adversely affect accumulations.

          * 'no' means the data types should not be cast at all.
          * 'equiv' means only byte-order changes are allowed.
          * 'safe' means only casts which can preserve values are allowed.
          * 'same_kind' means only safe casts or casts within a kind,
            like float64 to float32, are allowed.
          * 'unsafe' means any data conversions may be done.

        Default is 'safe'.
    optimize : {False, True, 'greedy', 'optimal'}, optional
        Controls if intermediate optimization should occur. No optimization
        will occur if False and True will default to the 'greedy' algorithm.
        Also accepts an explicit contraction list from the ``np.einsum_path``
        function. See ``np.einsum_path`` for more details. Defaults to False.

    Returns
    -------
    output : ndarray
        The calculation based on the Einstein summation convention.

    See Also
    --------
    einsum_path, dot, inner, outer, tensordot, linalg.multi_dot

    Notes
    -----
    The Einstein summation convention can be used to compute
    many multi-dimensional, linear algebraic array operations. `einsum`
    provides a succinct way of representing these.

    A non-exhaustive list of these operations,
    which can be computed by `einsum`, is shown below along with examples:

    * Trace of an array, :py:func:`numpy.trace`.
    * Return a diagonal, :py:func:`numpy.diag`.
    * Array axis summations, :py:func:`numpy.sum`.
    * Transpositions and permutations, :py:func:`numpy.transpose`.
    * Matrix multiplication and dot product, :py:func:`numpy.matmul` :py:func:`numpy.dot`.
    * Vector inner and outer products, :py:func:`numpy.inner` :py:func:`numpy.outer`.
    * Broadcasting, element-wise and scalar multiplication, :py:func:`numpy.multiply`.
    * Tensor contractions, :py:func:`numpy.tensordot`.
    * Chained array operations, in efficient calculation order, :py:func:`numpy.einsum_path`.

    The subscripts string is a comma-separated list of subscript labels,
    where each label refers to a dimension of the corresponding operand.
    Whenever a label is repeated it is summed, so ``np.einsum('i,i', a, b)``
    is equivalent to :py:func:`np.inner(a,b) <numpy.inner>`. If a label
    appears only once, it is not summed, so ``np.einsum('i', a)`` produces a
    view of ``a`` with no changes. A further example ``np.einsum('ij,jk', a, b)``
    describes traditional matrix multiplication and is equivalent to
    :py:func:`np.matmul(a,b) <numpy.matmul>`. Repeated subscript labels in one
    operand take the diagonal. For example, ``np.einsum('ii', a)`` is equivalent
    to :py:func:`np.trace(a) <numpy.trace>`.

    In *implicit mode*, the chosen subscripts are important
    since the axes of the output are reordered alphabetically.  This
    means that ``np.einsum('ij', a)`` doesn't affect a 2D array, while
    ``np.einsum('ji', a)`` takes its transpose. Additionally,
    ``np.einsum('ij,jk', a, b)`` returns a matrix multiplication, while,
    ``np.einsum('ij,jh', a, b)`` returns the transpose of the
    multiplication since subscript 'h' precedes subscript 'i'.

    In *explicit mode* the output can be directly controlled by
    specifying output subscript labels.  This requires the
    identifier '->' as well as the list of output subscript labels.
    This feature increases the flexibility of the function since
    summing can be disabled or forced when required. The call
    ``np.einsum('i->', a)`` is like :py:func:`np.sum(a) <numpy.sum>`
    if ``a`` is a 1-D array, and ``np.einsum('ii->i', a)``
    is like :py:func:`np.diag(a) <numpy.diag>` if ``a`` is a square 2-D array.
    The difference is that `einsum` does not allow broadcasting by default.
    Additionally ``np.einsum('ij,jh->ih', a, b)`` directly specifies the
    order of the output subscript labels and therefore returns matrix
    multiplication, unlike the example above in implicit mode.

    To enable and control broadcasting, use an ellipsis.  Default
    NumPy-style broadcasting is done by adding an ellipsis
    to the left of each term, like ``np.einsum('...ii->...i', a)``.
    ``np.einsum('...i->...', a)`` is like
    :py:func:`np.sum(a, axis=-1) <numpy.sum>` for array ``a`` of any shape.
    To take the trace along the first and last axes,
    you can do ``np.einsum('i...i', a)``, or to do a matrix-matrix
    product with the left-most indices instead of rightmost, one can do
    ``np.einsum('ij...,jk...->ik...', a, b)``.

    When there is only one operand, no axes are summed, and no output
    parameter is provided, a view into the operand is returned instead
    of a new array.  Thus, taking the diagonal as ``np.einsum('ii->i', a)``
    produces a view (changed in version 1.10.0).

    `einsum` also provides an alternative way to provide the subscripts
    and operands as ``einsum(op0, sublist0, op1, sublist1, ..., [sublistout])``.
    If the output shape is not provided in this format `einsum` will be
    calculated in implicit mode, otherwise it will be performed explicitly.
    The examples below have corresponding `einsum` calls with the two
    parameter methods.

    Views returned from einsum are now writeable whenever the input array
    is writeable. For example, ``np.einsum('ijk...->kji...', a)`` will now
    have the same effect as :py:func:`np.swapaxes(a, 0, 2) <numpy.swapaxes>`
    and ``np.einsum('ii->i', a)`` will return a writeable view of the diagonal
    of a 2D array.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(25).reshape(5,5)
    >>> b = np.arange(5)
    >>> c = np.arange(6).reshape(2,3)

    Trace of a matrix:

    >>> np.einsum('ii', a)
    60
    >>> np.einsum(a, [0,0])
    60
    >>> np.trace(a)
    60

    Extract the diagonal (requires explicit form):

    >>> np.einsum('ii->i', a)
    array([ 0,  6, 12, 18, 24])
    >>> np.einsum(a, [0,0], [0])
    array([ 0,  6, 12, 18, 24])
    >>> np.diag(a)
    array([ 0,  6, 12, 18, 24])

    Sum over an axis (requires explicit form):

    >>> np.einsum('ij->i', a)
    array([ 10,  35,  60,  85, 110])
    >>> np.einsum(a, [0,1], [0])
    array([ 10,  35,  60,  85, 110])
    >>> np.sum(a, axis=1)
    array([ 10,  35,  60,  85, 110])

    For higher dimensional arrays summing a single axis can be done with ellipsis:

    >>> np.einsum('...j->...', a)
    array([ 10,  35,  60,  85, 110])
    >>> np.einsum(a, [Ellipsis,1], [Ellipsis])
    array([ 10,  35,  60,  85, 110])

    Compute a matrix transpose, or reorder any number of axes:

    >>> np.einsum('ji', c)
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.einsum('ij->ji', c)
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.einsum(c, [1,0])
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.transpose(c)
    array([[0, 3],
           [1, 4],
           [2, 5]])

    Vector inner products:

    >>> np.einsum('i,i', b, b)
    30
    >>> np.einsum(b, [0], b, [0])
    30
    >>> np.inner(b,b)
    30

    Matrix vector multiplication:

    >>> np.einsum('ij,j', a, b)
    array([ 30,  80, 130, 180, 230])
    >>> np.einsum(a, [0,1], b, [1])
    array([ 30,  80, 130, 180, 230])
    >>> np.dot(a, b)
    array([ 30,  80, 130, 180, 230])
    >>> np.einsum('...j,j', a, b)
    array([ 30,  80, 130, 180, 230])

    Broadcasting and scalar multiplication:

    >>> np.einsum('..., ...', 3, c)
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> np.einsum(',ij', 3, c)
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> np.einsum(3, [Ellipsis], c, [Ellipsis])
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> np.multiply(3, c)
    array([[ 0,  3,  6],
           [ 9, 12, 15]])

    Vector outer product:

    >>> np.einsum('i,j', np.arange(2)+1, b)
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])
    >>> np.einsum(np.arange(2)+1, [0], b, [1])
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])
    >>> np.outer(np.arange(2)+1, b)
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])

    Tensor contraction:

    >>> a = np.arange(60.).reshape(3,4,5)
    >>> b = np.arange(24.).reshape(4,3,2)
    >>> np.einsum('ijk,jil->kl', a, b)
    array([[ 4400.,  4730.],
           [ 4532.,  4874.],
           [ 4664.,  5018.],
           [ 4796.,  5162.],
           [ 4928.,  5306.]])
    >>> np.einsum(a, [0,1,2], b, [1,0,3], [2,3])
    array([[ 4400.,  4730.],
           [ 4532.,  4874.],
           [ 4664.,  5018.],
           [ 4796.,  5162.],
           [ 4928.,  5306.]])
    >>> np.tensordot(a,b, axes=([1,0],[0,1]))
    array([[ 4400.,  4730.],
           [ 4532.,  4874.],
           [ 4664.,  5018.],
           [ 4796.,  5162.],
           [ 4928.,  5306.]])

    Writeable returned arrays (since version 1.10.0):

    >>> a = np.zeros((3, 3))
    >>> np.einsum('ii->i', a)[:] = 1
    >>> a
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])

    Example of ellipsis use:

    >>> a = np.arange(6).reshape((3,2))
    >>> b = np.arange(12).reshape((4,3))
    >>> np.einsum('ki,jk->ij', a, b)
    array([[10, 28, 46, 64],
           [13, 40, 67, 94]])
    >>> np.einsum('ki,...k->i...', a, b)
    array([[10, 28, 46, 64],
           [13, 40, 67, 94]])
    >>> np.einsum('k...,jk', a, b)
    array([[10, 28, 46, 64],
           [13, 40, 67, 94]])

    """)


##############################################################################
#
# Documentation for ndarray attributes and methods
#
##############################################################################


##############################################################################
#
# ndarray object
#
##############################################################################


add_newdoc('numpy._core.multiarray', 'ndarray',
    """
    ndarray(shape, dtype=float, buffer=None, offset=0,
            strides=None, order=None)

    An array object represents a multidimensional, homogeneous array
    of fixed-size items.  An associated data-type object describes the
    format of each element in the array (its byte-order, how many bytes it
    occupies in memory, whether it is an integer, a floating point number,
    or something else, etc.)

    Arrays should be constructed using `array`, `zeros` or `empty` (refer
    to the See Also section below).  The parameters given here refer to
    a low-level method (`ndarray(...)`) for instantiating an array.

    For more information, refer to the `numpy` module and examine the
    methods and attributes of an array.

    Parameters
    ----------
    (for the __new__ method; see Notes below)

    shape : tuple of ints
        Shape of created array.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type.
    buffer : object exposing buffer interface, optional
        Used to fill the array with data.
    offset : int, optional
        Offset of array data in buffer.
    strides : tuple of ints, optional
        Strides of data in memory.
    order : {'C', 'F'}, optional
        Row-major (C-style) or column-major (Fortran-style) order.

    Attributes
    ----------
    T : ndarray
        Transpose of the array.
    data : buffer
        The array's elements, in memory.
    dtype : dtype object
        Describes the format of the elements in the array.
    flags : dict
        Dictionary containing information related to memory use, e.g.,
        'C_CONTIGUOUS', 'OWNDATA', 'WRITEABLE', etc.
    flat : numpy.flatiter object
        Flattened version of the array as an iterator.  The iterator
        allows assignments, e.g., ``x.flat = 3`` (See `ndarray.flat` for
        assignment examples; TODO).
    imag : ndarray
        Imaginary part of the array.
    real : ndarray
        Real part of the array.
    size : int
        Number of elements in the array.
    itemsize : int
        The memory use of each array element in bytes.
    nbytes : int
        The total number of bytes required to store the array data,
        i.e., ``itemsize * size``.
    ndim : int
        The array's number of dimensions.
    shape : tuple of ints
        Shape of the array.
    strides : tuple of ints
        The step-size required to move from one element to the next in
        memory. For example, a contiguous ``(3, 4)`` array of type
        ``int16`` in C-order has strides ``(8, 2)``.  This implies that
        to move from element to element in memory requires jumps of 2 bytes.
        To move from row-to-row, one needs to jump 8 bytes at a time
        (``2 * 4``).
    ctypes : ctypes object
        Class containing properties of the array needed for interaction
        with ctypes.
    base : ndarray
        If the array is a view into another array, that array is its `base`
        (unless that array is also a view).  The `base` array is where the
        array data is actually stored.

    See Also
    --------
    array : Construct an array.
    zeros : Create an array, each element of which is zero.
    empty : Create an array, but leave its allocated memory unchanged (i.e.,
            it contains "garbage").
    dtype : Create a data-type.
    numpy.typing.NDArray : An ndarray alias :term:`generic <generic type>`
                           w.r.t. its `dtype.type <numpy.dtype.type>`.

    Notes
    -----
    There are two modes of creating an array using ``__new__``:

    1. If `buffer` is None, then only `shape`, `dtype`, and `order`
       are used.
    2. If `buffer` is an object exposing the buffer interface, then
       all keywords are interpreted.

    No ``__init__`` method is needed because the array is fully initialized
    after the ``__new__`` method.

    Examples
    --------
    These examples illustrate the low-level `ndarray` constructor.  Refer
    to the `See Also` section above for easier ways of constructing an
    ndarray.

    First mode, `buffer` is None:

    >>> import numpy as np
    >>> np.ndarray(shape=(2,2), dtype=float, order='F')
    array([[0.0e+000, 0.0e+000], # random
           [     nan, 2.5e-323]])

    Second mode:

    >>> np.ndarray((2,), buffer=np.array([1,2,3]),
    ...            offset=np.int_().itemsize,
    ...            dtype=int) # offset = 1*itemsize, i.e. skip first element
    array([2, 3])

    """)


##############################################################################
#
# ndarray attributes
#
##############################################################################


add_newdoc('numpy._core.multiarray', 'ndarray', ('__array_interface__',
    """Array protocol: Python side."""))


add_newdoc('numpy._core.multiarray', 'ndarray', ('__array_priority__',
    """Array priority."""))


add_newdoc('numpy._core.multiarray', 'ndarray', ('__array_struct__',
    """Array protocol: C-struct side."""))

add_newdoc('numpy._core.multiarray', 'ndarray', ('__dlpack__',
    """
    a.__dlpack__(*, stream=None, max_version=None, dl_device=None, copy=None)

    DLPack Protocol: Part of the Array API.

    """))

add_newdoc('numpy._core.multiarray', 'ndarray', ('__dlpack_device__',
    """
    a.__dlpack_device__()

    DLPack Protocol: Part of the Array API.

    """))

add_newdoc('numpy._core.multiarray', 'ndarray', ('base',
    """
    Base object if memory is from some other object.

    Examples
    --------
    The base of an array that owns its memory is None:

    >>> import numpy as np
    >>> x = np.array([1,2,3,4])
    >>> x.base is None
    True

    Slicing creates a view, whose memory is shared with x:

    >>> y = x[2:]
    >>> y.base is x
    True

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('ctypes',
    """
    An object to simplify the interaction of the array with the ctypes
    module.

    This attribute creates an object that makes it easier to use arrays
    when calling shared libraries with the ctypes module. The returned
    object has, among others, data, shape, and strides attributes (see
    Notes below) which themselves return ctypes objects that can be used
    as arguments to a shared library.

    Parameters
    ----------
    None

    Returns
    -------
    c : Python object
        Possessing attributes data, shape, strides, etc.

    See Also
    --------
    numpy.ctypeslib

    Notes
    -----
    Below are the public attributes of this object which were documented
    in "Guide to NumPy" (we have omitted undocumented public attributes,
    as well as documented private attributes):

    .. autoattribute:: numpy._core._internal._ctypes.data
        :noindex:

    .. autoattribute:: numpy._core._internal._ctypes.shape
        :noindex:

    .. autoattribute:: numpy._core._internal._ctypes.strides
        :noindex:

    .. automethod:: numpy._core._internal._ctypes.data_as
        :noindex:

    .. automethod:: numpy._core._internal._ctypes.shape_as
        :noindex:

    .. automethod:: numpy._core._internal._ctypes.strides_as
        :noindex:

    If the ctypes module is not available, then the ctypes attribute
    of array objects still returns something useful, but ctypes objects
    are not returned and errors may be raised instead. In particular,
    the object will still have the ``as_parameter`` attribute which will
    return an integer equal to the data attribute.

    Examples
    --------
    >>> import numpy as np
    >>> import ctypes
    >>> x = np.array([[0, 1], [2, 3]], dtype=np.int32)
    >>> x
    array([[0, 1],
           [2, 3]], dtype=int32)
    >>> x.ctypes.data
    31962608 # may vary
    >>> x.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    <__main__.LP_c_uint object at 0x7ff2fc1fc200> # may vary
    >>> x.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)).contents
    c_uint(0)
    >>> x.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)).contents
    c_ulong(4294967296)
    >>> x.ctypes.shape
    <numpy._core._internal.c_long_Array_2 object at 0x7ff2fc1fce60> # may vary
    >>> x.ctypes.strides
    <numpy._core._internal.c_long_Array_2 object at 0x7ff2fc1ff320> # may vary

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('data',
    """Python buffer object pointing to the start of the array's data."""))


add_newdoc('numpy._core.multiarray', 'ndarray', ('dtype',
    """
    Data-type of the array's elements.

    .. warning::

        Setting ``arr.dtype`` is discouraged and may be deprecated in the
        future.  Setting will replace the ``dtype`` without modifying the
        memory (see also `ndarray.view` and `ndarray.astype`).

    Parameters
    ----------
    None

    Returns
    -------
    d : numpy dtype object

    See Also
    --------
    ndarray.astype : Cast the values contained in the array to a new data-type.
    ndarray.view : Create a view of the same data but a different data-type.
    numpy.dtype

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(4).reshape((2, 2))
    >>> x
    array([[0, 1],
           [2, 3]])
    >>> x.dtype
    dtype('int64')   # may vary (OS, bitness)
    >>> isinstance(x.dtype, np.dtype)
    True

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('imag',
    """
    The imaginary part of the array.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.sqrt([1+0j, 0+1j])
    >>> x.imag
    array([ 0.        ,  0.70710678])
    >>> x.imag.dtype
    dtype('float64')

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('itemsize',
    """
    Length of one array element in bytes.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1,2,3], dtype=np.float64)
    >>> x.itemsize
    8
    >>> x = np.array([1,2,3], dtype=np.complex128)
    >>> x.itemsize
    16

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('flags',
    """
    Information about the memory layout of the array.

    Attributes
    ----------
    C_CONTIGUOUS (C)
        The data is in a single, C-style contiguous segment.
    F_CONTIGUOUS (F)
        The data is in a single, Fortran-style contiguous segment.
    OWNDATA (O)
        The array owns the memory it uses or borrows it from another object.
    WRITEABLE (W)
        The data area can be written to.  Setting this to False locks
        the data, making it read-only.  A view (slice, etc.) inherits WRITEABLE
        from its base array at creation time, but a view of a writeable
        array may be subsequently locked while the base array remains writeable.
        (The opposite is not true, in that a view of a locked array may not
        be made writeable.  However, currently, locking a base object does not
        lock any views that already reference it, so under that circumstance it
        is possible to alter the contents of a locked array via a previously
        created writeable view onto it.)  Attempting to change a non-writeable
        array raises a RuntimeError exception.
    ALIGNED (A)
        The data and all elements are aligned appropriately for the hardware.
    WRITEBACKIFCOPY (X)
        This array is a copy of some other array. The C-API function
        PyArray_ResolveWritebackIfCopy must be called before deallocating
        to the base array will be updated with the contents of this array.
    FNC
        F_CONTIGUOUS and not C_CONTIGUOUS.
    FORC
        F_CONTIGUOUS or C_CONTIGUOUS (one-segment test).
    BEHAVED (B)
        ALIGNED and WRITEABLE.
    CARRAY (CA)
        BEHAVED and C_CONTIGUOUS.
    FARRAY (FA)
        BEHAVED and F_CONTIGUOUS and not C_CONTIGUOUS.

    Notes
    -----
    The `flags` object can be accessed dictionary-like (as in ``a.flags['WRITEABLE']``),
    or by using lowercased attribute names (as in ``a.flags.writeable``). Short flag
    names are only supported in dictionary access.

    Only the WRITEBACKIFCOPY, WRITEABLE, and ALIGNED flags can be
    changed by the user, via direct assignment to the attribute or dictionary
    entry, or by calling `ndarray.setflags`.

    The array flags cannot be set arbitrarily:

    - WRITEBACKIFCOPY can only be set ``False``.
    - ALIGNED can only be set ``True`` if the data is truly aligned.
    - WRITEABLE can only be set ``True`` if the array owns its own memory
      or the ultimate owner of the memory exposes a writeable buffer
      interface or is a string.

    Arrays can be both C-style and Fortran-style contiguous simultaneously.
    This is clear for 1-dimensional arrays, but can also be true for higher
    dimensional arrays.

    Even for contiguous arrays a stride for a given dimension
    ``arr.strides[dim]`` may be *arbitrary* if ``arr.shape[dim] == 1``
    or the array has no elements.
    It does *not* generally hold that ``self.strides[-1] == self.itemsize``
    for C-style contiguous arrays or ``self.strides[0] == self.itemsize`` for
    Fortran-style contiguous arrays is true.
    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('flat',
    """
    A 1-D iterator over the array.

    This is a `numpy.flatiter` instance, which acts similarly to, but is not
    a subclass of, Python's built-in iterator object.

    See Also
    --------
    flatten : Return a copy of the array collapsed into one dimension.

    flatiter

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(1, 7).reshape(2, 3)
    >>> x
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> x.flat[3]
    4
    >>> x.T
    array([[1, 4],
           [2, 5],
           [3, 6]])
    >>> x.T.flat[3]
    5
    >>> type(x.flat)
    <class 'numpy.flatiter'>

    An assignment example:

    >>> x.flat = 3; x
    array([[3, 3, 3],
           [3, 3, 3]])
    >>> x.flat[[1,4]] = 1; x
    array([[3, 1, 3],
           [3, 1, 3]])

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('nbytes',
    """
    Total bytes consumed by the elements of the array.

    Notes
    -----
    Does not include memory consumed by non-element attributes of the
    array object.

    See Also
    --------
    sys.getsizeof
        Memory consumed by the object itself without parents in case view.
        This does include memory consumed by non-element attributes.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.zeros((3,5,2), dtype=np.complex128)
    >>> x.nbytes
    480
    >>> np.prod(x.shape) * x.itemsize
    480

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('ndim',
    """
    Number of array dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3])
    >>> x.ndim
    1
    >>> y = np.zeros((2, 3, 4))
    >>> y.ndim
    3

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('real',
    """
    The real part of the array.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.sqrt([1+0j, 0+1j])
    >>> x.real
    array([ 1.        ,  0.70710678])
    >>> x.real.dtype
    dtype('float64')

    See Also
    --------
    numpy.real : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('shape',
    """
    Tuple of array dimensions.

    The shape property is usually used to get the current shape of an array,
    but may also be used to reshape the array in-place by assigning a tuple of
    array dimensions to it.  As with `numpy.reshape`, one of the new shape
    dimensions can be -1, in which case its value is inferred from the size of
    the array and the remaining dimensions. Reshaping an array in-place will
    fail if a copy is required.

    .. warning::

        Setting ``arr.shape`` is discouraged and may be deprecated in the
        future.  Using `ndarray.reshape` is the preferred approach.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4])
    >>> x.shape
    (4,)
    >>> y = np.zeros((2, 3, 4))
    >>> y.shape
    (2, 3, 4)
    >>> y.shape = (3, 8)
    >>> y
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
    >>> y.shape = (3, 6)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: cannot reshape array of size 24 into shape (3,6)
    >>> np.zeros((4,2))[::2].shape = (-1,)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    AttributeError: Incompatible shape for in-place modification. Use
    `.reshape()` to make a copy with the desired shape.

    See Also
    --------
    numpy.shape : Equivalent getter function.
    numpy.reshape : Function similar to setting ``shape``.
    ndarray.reshape : Method similar to setting ``shape``.

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('size',
    """
    Number of elements in the array.

    Equal to ``np.prod(a.shape)``, i.e., the product of the array's
    dimensions.

    Notes
    -----
    `a.size` returns a standard arbitrary precision Python integer. This
    may not be the case with other methods of obtaining the same value
    (like the suggested ``np.prod(a.shape)``, which returns an instance
    of ``np.int_``), and may be relevant if the value is used further in
    calculations that may overflow a fixed size integer type.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.zeros((3, 5, 2), dtype=np.complex128)
    >>> x.size
    30
    >>> np.prod(x.shape)
    30

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('strides',
    """
    Tuple of bytes to step in each dimension when traversing an array.

    The byte offset of element ``(i[0], i[1], ..., i[n])`` in an array `a`
    is::

        offset = sum(np.array(i) * a.strides)

    A more detailed explanation of strides can be found in
    :ref:`arrays.ndarray`.

    .. warning::

        Setting ``arr.strides`` is discouraged and may be deprecated in the
        future.  `numpy.lib.stride_tricks.as_strided` should be preferred
        to create a new view of the same data in a safer way.

    Notes
    -----
    Imagine an array of 32-bit integers (each 4 bytes)::

      x = np.array([[0, 1, 2, 3, 4],
                    [5, 6, 7, 8, 9]], dtype=np.int32)

    This array is stored in memory as 40 bytes, one after the other
    (known as a contiguous block of memory).  The strides of an array tell
    us how many bytes we have to skip in memory to move to the next position
    along a certain axis.  For example, we have to skip 4 bytes (1 value) to
    move to the next column, but 20 bytes (5 values) to get to the same
    position in the next row.  As such, the strides for the array `x` will be
    ``(20, 4)``.

    See Also
    --------
    numpy.lib.stride_tricks.as_strided

    Examples
    --------
    >>> import numpy as np
    >>> y = np.reshape(np.arange(2 * 3 * 4, dtype=np.int32), (2, 3, 4))
    >>> y
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]], dtype=np.int32)
    >>> y.strides
    (48, 16, 4)
    >>> y[1, 1, 1]
    np.int32(17)
    >>> offset = sum(y.strides * np.array((1, 1, 1)))
    >>> offset // y.itemsize
    np.int64(17)

    >>> x = np.reshape(np.arange(5*6*7*8, dtype=np.int32), (5, 6, 7, 8))
    >>> x = x.transpose(2, 3, 1, 0)
    >>> x.strides
    (32, 4, 224, 1344)
    >>> i = np.array([3, 5, 2, 2], dtype=np.int32)
    >>> offset = sum(i * x.strides)
    >>> x[3, 5, 2, 2]
    np.int32(813)
    >>> offset // x.itemsize
    np.int64(813)

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('T',
    """
    View of the transposed array.

    Same as ``self.transpose()``.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> a
    array([[1, 2],
           [3, 4]])
    >>> a.T
    array([[1, 3],
           [2, 4]])

    >>> a = np.array([1, 2, 3, 4])
    >>> a
    array([1, 2, 3, 4])
    >>> a.T
    array([1, 2, 3, 4])

    See Also
    --------
    transpose

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('mT',
    """
    View of the matrix transposed array.

    The matrix transpose is the transpose of the last two dimensions, even
    if the array is of higher dimension.

    .. versionadded:: 2.0

    Raises
    ------
    ValueError
        If the array is of dimension less than 2.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> a
    array([[1, 2],
           [3, 4]])
    >>> a.mT
    array([[1, 3],
           [2, 4]])

    >>> a = np.arange(8).reshape((2, 2, 2))
    >>> a
    array([[[0, 1],
            [2, 3]],
    <BLANKLINE>
           [[4, 5],
            [6, 7]]])
    >>> a.mT
    array([[[0, 2],
            [1, 3]],
    <BLANKLINE>
           [[4, 6],
            [5, 7]]])

    """))
##############################################################################
#
# ndarray methods
#
##############################################################################


add_newdoc('numpy._core.multiarray', 'ndarray', ('__array__',
    """
    a.__array__([dtype], *, copy=None)

    For ``dtype`` parameter it returns a new reference to self if
    ``dtype`` is not given or it matches array's data type.
    A new array of provided data type is returned if ``dtype``
    is different from the current data type of the array.
    For ``copy`` parameter it returns a new reference to self if
    ``copy=False`` or ``copy=None`` and copying isn't enforced by ``dtype``
    parameter. The method returns a new array for ``copy=True``, regardless of
    ``dtype`` parameter.

    A more detailed explanation of the ``__array__`` interface
    can be found in :ref:`dunder_array.interface`.

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('__array_finalize__',
    """
    a.__array_finalize__(obj, /)

    Present so subclasses can call super. Does nothing.

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('__array_wrap__',
    """
    a.__array_wrap__(array[, context], /)

    Returns a view of `array` with the same type as self.

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('__copy__',
    """
    a.__copy__()

    Used if :func:`copy.copy` is called on an array. Returns a copy of the array.

    Equivalent to ``a.copy(order='K')``.

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('__class_getitem__',
    """
    a.__class_getitem__(item, /)

    Return a parametrized wrapper around the `~numpy.ndarray` type.

    .. versionadded:: 1.22

    Returns
    -------
    alias : types.GenericAlias
        A parametrized `~numpy.ndarray` type.

    Examples
    --------
    >>> from typing import Any
    >>> import numpy as np

    >>> np.ndarray[Any, np.dtype[np.uint8]]
    numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]

    See Also
    --------
    :pep:`585` : Type hinting generics in standard collections.
    numpy.typing.NDArray : An ndarray alias :term:`generic <generic type>`
                        w.r.t. its `dtype.type <numpy.dtype.type>`.

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('__deepcopy__',
    """
    a.__deepcopy__(memo, /)

    Used if :func:`copy.deepcopy` is called on an array.

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('__reduce__',
    """
    a.__reduce__()

    For pickling.

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('__setstate__',
    """
    a.__setstate__(state, /)

    For unpickling.

    The `state` argument must be a sequence that contains the following
    elements:

    Parameters
    ----------
    version : int
        optional pickle version. If omitted defaults to 0.
    shape : tuple
    dtype : data-type
    isFortran : bool
    rawdata : string or list
        a binary string with the data (or a list if 'a' is an object array)

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('all',
    """
    a.all(axis=None, out=None, keepdims=False, *, where=True)

    Returns True if all elements evaluate to True.

    Refer to `numpy.all` for full documentation.

    See Also
    --------
    numpy.all : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('any',
    """
    a.any(axis=None, out=None, keepdims=False, *, where=True)

    Returns True if any of the elements of `a` evaluate to True.

    Refer to `numpy.any` for full documentation.

    See Also
    --------
    numpy.any : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('argmax',
    """
    a.argmax(axis=None, out=None, *, keepdims=False)

    Return indices of the maximum values along the given axis.

    Refer to `numpy.argmax` for full documentation.

    See Also
    --------
    numpy.argmax : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('argmin',
    """
    a.argmin(axis=None, out=None, *, keepdims=False)

    Return indices of the minimum values along the given axis.

    Refer to `numpy.argmin` for detailed documentation.

    See Also
    --------
    numpy.argmin : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('argsort',
    """
    a.argsort(axis=-1, kind=None, order=None)

    Returns the indices that would sort this array.

    Refer to `numpy.argsort` for full documentation.

    See Also
    --------
    numpy.argsort : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('argpartition',
    """
    a.argpartition(kth, axis=-1, kind='introselect', order=None)

    Returns the indices that would partition this array.

    Refer to `numpy.argpartition` for full documentation.

    See Also
    --------
    numpy.argpartition : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('astype',
    """
    a.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)

    Copy of the array, cast to a specified type.

    Parameters
    ----------
    dtype : str or dtype
        Typecode or data-type to which the array is cast.
    order : {'C', 'F', 'A', 'K'}, optional
        Controls the memory layout order of the result.
        'C' means C order, 'F' means Fortran order, 'A'
        means 'F' order if all the arrays are Fortran contiguous,
        'C' order otherwise, and 'K' means as close to the
        order the array elements appear in memory as possible.
        Default is 'K'.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur. Defaults to 'unsafe'
        for backwards compatibility.

        * 'no' means the data types should not be cast at all.
        * 'equiv' means only byte-order changes are allowed.
        * 'safe' means only casts which can preserve values are allowed.
        * 'same_kind' means only safe casts or casts within a kind,
          like float64 to float32, are allowed.
        * 'unsafe' means any data conversions may be done.
    subok : bool, optional
        If True, then sub-classes will be passed-through (default), otherwise
        the returned array will be forced to be a base-class array.
    copy : bool, optional
        By default, astype always returns a newly allocated array. If this
        is set to false, and the `dtype`, `order`, and `subok`
        requirements are satisfied, the input array is returned instead
        of a copy.

    Returns
    -------
    arr_t : ndarray
        Unless `copy` is False and the other conditions for returning the input
        array are satisfied (see description for `copy` input parameter), `arr_t`
        is a new array of the same shape as the input array, with dtype, order
        given by `dtype`, `order`.

    Raises
    ------
    ComplexWarning
        When casting from complex to float or int. To avoid this,
        one should use ``a.real.astype(t)``.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 2.5])
    >>> x
    array([1. ,  2. ,  2.5])

    >>> x.astype(int)
    array([1, 2, 2])

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('byteswap',
    """
    a.byteswap(inplace=False)

    Swap the bytes of the array elements

    Toggle between low-endian and big-endian data representation by
    returning a byteswapped array, optionally swapped in-place.
    Arrays of byte-strings are not swapped. The real and imaginary
    parts of a complex number are swapped individually.

    Parameters
    ----------
    inplace : bool, optional
        If ``True``, swap bytes in-place, default is ``False``.

    Returns
    -------
    out : ndarray
        The byteswapped array. If `inplace` is ``True``, this is
        a view to self.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([1, 256, 8755], dtype=np.int16)
    >>> list(map(hex, A))
    ['0x1', '0x100', '0x2233']
    >>> A.byteswap(inplace=True)
    array([  256,     1, 13090], dtype=int16)
    >>> list(map(hex, A))
    ['0x100', '0x1', '0x3322']

    Arrays of byte-strings are not swapped

    >>> A = np.array([b'ceg', b'fac'])
    >>> A.byteswap()
    array([b'ceg', b'fac'], dtype='|S3')

    ``A.view(A.dtype.newbyteorder()).byteswap()`` produces an array with
    the same values but different representation in memory

    >>> A = np.array([1, 2, 3],dtype=np.int64)
    >>> A.view(np.uint8)
    array([1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0,
           0, 0], dtype=uint8)
    >>> A.view(A.dtype.newbyteorder()).byteswap(inplace=True)
    array([1, 2, 3], dtype='>i8')
    >>> A.view(np.uint8)
    array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
           0, 3], dtype=uint8)

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('choose',
    """
    a.choose(choices, out=None, mode='raise')

    Use an index array to construct a new array from a set of choices.

    Refer to `numpy.choose` for full documentation.

    See Also
    --------
    numpy.choose : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('clip',
    """
    a.clip(min=None, max=None, out=None, **kwargs)

    Return an array whose values are limited to ``[min, max]``.
    One of max or min must be given.

    Refer to `numpy.clip` for full documentation.

    See Also
    --------
    numpy.clip : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('compress',
    """
    a.compress(condition, axis=None, out=None)

    Return selected slices of this array along given axis.

    Refer to `numpy.compress` for full documentation.

    See Also
    --------
    numpy.compress : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('conj',
    """
    a.conj()

    Complex-conjugate all elements.

    Refer to `numpy.conjugate` for full documentation.

    See Also
    --------
    numpy.conjugate : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('conjugate',
    """
    a.conjugate()

    Return the complex conjugate, element-wise.

    Refer to `numpy.conjugate` for full documentation.

    See Also
    --------
    numpy.conjugate : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('copy',
    """
    a.copy(order='C')

    Return a copy of the array.

    Parameters
    ----------
    order : {'C', 'F', 'A', 'K'}, optional
        Controls the memory layout of the copy. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible. (Note that this function and :func:`numpy.copy` are very
        similar but have different default values for their order=
        arguments, and this function always passes sub-classes through.)

    See also
    --------
    numpy.copy : Similar function with different default behavior
    numpy.copyto

    Notes
    -----
    This function is the preferred method for creating an array copy.  The
    function :func:`numpy.copy` is similar, but it defaults to using order 'K',
    and will not pass sub-classes through by default.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[1,2,3],[4,5,6]], order='F')

    >>> y = x.copy()

    >>> x.fill(0)

    >>> x
    array([[0, 0, 0],
           [0, 0, 0]])

    >>> y
    array([[1, 2, 3],
           [4, 5, 6]])

    >>> y.flags['C_CONTIGUOUS']
    True

    For arrays containing Python objects (e.g. dtype=object),
    the copy is a shallow one. The new array will contain the
    same object which may lead to surprises if that object can
    be modified (is mutable):

    >>> a = np.array([1, 'm', [2, 3, 4]], dtype=object)
    >>> b = a.copy()
    >>> b[2][0] = 10
    >>> a
    array([1, 'm', list([10, 3, 4])], dtype=object)

    To ensure all elements within an ``object`` array are copied,
    use `copy.deepcopy`:

    >>> import copy
    >>> a = np.array([1, 'm', [2, 3, 4]], dtype=object)
    >>> c = copy.deepcopy(a)
    >>> c[2][0] = 10
    >>> c
    array([1, 'm', list([10, 3, 4])], dtype=object)
    >>> a
    array([1, 'm', list([2, 3, 4])], dtype=object)

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('cumprod',
    """
    a.cumprod(axis=None, dtype=None, out=None)

    Return the cumulative product of the elements along the given axis.

    Refer to `numpy.cumprod` for full documentation.

    See Also
    --------
    numpy.cumprod : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('cumsum',
    """
    a.cumsum(axis=None, dtype=None, out=None)

    Return the cumulative sum of the elements along the given axis.

    Refer to `numpy.cumsum` for full documentation.

    See Also
    --------
    numpy.cumsum : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('diagonal',
    """
    a.diagonal(offset=0, axis1=0, axis2=1)

    Return specified diagonals. In NumPy 1.9 the returned array is a
    read-only view instead of a copy as in previous NumPy versions.  In
    a future version the read-only restriction will be removed.

    Refer to :func:`numpy.diagonal` for full documentation.

    See Also
    --------
    numpy.diagonal : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('dot'))


add_newdoc('numpy._core.multiarray', 'ndarray', ('dump',
    """
    a.dump(file)

    Dump a pickle of the array to the specified file.
    The array can be read back with pickle.load or numpy.load.

    Parameters
    ----------
    file : str or Path
        A string naming the dump file.

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('dumps',
    """
    a.dumps()

    Returns the pickle of the array as a string.
    pickle.loads will convert the string back to an array.

    Parameters
    ----------
    None

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('fill',
    """
    a.fill(value)

    Fill the array with a scalar value.

    Parameters
    ----------
    value : scalar
        All elements of `a` will be assigned this value.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, 2])
    >>> a.fill(0)
    >>> a
    array([0, 0])
    >>> a = np.empty(2)
    >>> a.fill(1)
    >>> a
    array([1.,  1.])

    Fill expects a scalar value and always behaves the same as assigning
    to a single array element.  The following is a rare example where this
    distinction is important:

    >>> a = np.array([None, None], dtype=object)
    >>> a[0] = np.array(3)
    >>> a
    array([array(3), None], dtype=object)
    >>> a.fill(np.array(3))
    >>> a
    array([array(3), array(3)], dtype=object)

    Where other forms of assignments will unpack the array being assigned:

    >>> a[...] = np.array(3)
    >>> a
    array([3, 3], dtype=object)

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('flatten',
    """
    a.flatten(order='C')

    Return a copy of the array collapsed into one dimension.

    Parameters
    ----------
    order : {'C', 'F', 'A', 'K'}, optional
        'C' means to flatten in row-major (C-style) order.
        'F' means to flatten in column-major (Fortran-
        style) order. 'A' means to flatten in column-major
        order if `a` is Fortran *contiguous* in memory,
        row-major order otherwise. 'K' means to flatten
        `a` in the order the elements occur in memory.
        The default is 'C'.

    Returns
    -------
    y : ndarray
        A copy of the input array, flattened to one dimension.

    See Also
    --------
    ravel : Return a flattened array.
    flat : A 1-D flat iterator over the array.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1,2], [3,4]])
    >>> a.flatten()
    array([1, 2, 3, 4])
    >>> a.flatten('F')
    array([1, 3, 2, 4])

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('getfield',
    """
    a.getfield(dtype, offset=0)

    Returns a field of the given array as a certain type.

    A field is a view of the array data with a given data-type. The values in
    the view are determined by the given type and the offset into the current
    array in bytes. The offset needs to be such that the view dtype fits in the
    array dtype; for example an array of dtype complex128 has 16-byte elements.
    If taking a view with a 32-bit integer (4 bytes), the offset needs to be
    between 0 and 12 bytes.

    Parameters
    ----------
    dtype : str or dtype
        The data type of the view. The dtype size of the view can not be larger
        than that of the array itself.
    offset : int
        Number of bytes to skip before beginning the element view.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.diag([1.+1.j]*2)
    >>> x[1, 1] = 2 + 4.j
    >>> x
    array([[1.+1.j,  0.+0.j],
           [0.+0.j,  2.+4.j]])
    >>> x.getfield(np.float64)
    array([[1.,  0.],
           [0.,  2.]])

    By choosing an offset of 8 bytes we can select the complex part of the
    array for our view:

    >>> x.getfield(np.float64, offset=8)
    array([[1.,  0.],
           [0.,  4.]])

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('item',
    """
    a.item(*args)

    Copy an element of an array to a standard Python scalar and return it.

    Parameters
    ----------
    \\*args : Arguments (variable number and type)

        * none: in this case, the method only works for arrays
          with one element (`a.size == 1`), which element is
          copied into a standard Python scalar object and returned.

        * int_type: this argument is interpreted as a flat index into
          the array, specifying which element to copy and return.

        * tuple of int_types: functions as does a single int_type argument,
          except that the argument is interpreted as an nd-index into the
          array.

    Returns
    -------
    z : Standard Python scalar object
        A copy of the specified element of the array as a suitable
        Python scalar

    Notes
    -----
    When the data type of `a` is longdouble or clongdouble, item() returns
    a scalar array object because there is no available Python scalar that
    would not lose information. Void arrays return a buffer object for item(),
    unless fields are defined, in which case a tuple is returned.

    `item` is very similar to a[args], except, instead of an array scalar,
    a standard Python scalar is returned. This can be useful for speeding up
    access to elements of the array and doing arithmetic on elements of the
    array using Python's optimized math.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(123)
    >>> x = np.random.randint(9, size=(3, 3))
    >>> x
    array([[2, 2, 6],
           [1, 3, 6],
           [1, 0, 1]])
    >>> x.item(3)
    1
    >>> x.item(7)
    0
    >>> x.item((0, 1))
    2
    >>> x.item((2, 2))
    1

    For an array with object dtype, elements are returned as-is.

    >>> a = np.array([np.int64(1)], dtype=object)
    >>> a.item() #return np.int64
    np.int64(1)

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('max',
    """
    a.max(axis=None, out=None, keepdims=False, initial=<no value>, where=True)

    Return the maximum along a given axis.

    Refer to `numpy.amax` for full documentation.

    See Also
    --------
    numpy.amax : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('mean',
    """
    a.mean(axis=None, dtype=None, out=None, keepdims=False, *, where=True)

    Returns the average of the array elements along given axis.

    Refer to `numpy.mean` for full documentation.

    See Also
    --------
    numpy.mean : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('min',
    """
    a.min(axis=None, out=None, keepdims=False, initial=<no value>, where=True)

    Return the minimum along a given axis.

    Refer to `numpy.amin` for full documentation.

    See Also
    --------
    numpy.amin : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('nonzero',
    """
    a.nonzero()

    Return the indices of the elements that are non-zero.

    Refer to `numpy.nonzero` for full documentation.

    See Also
    --------
    numpy.nonzero : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('prod',
    """
    a.prod(axis=None, dtype=None, out=None, keepdims=False,
        initial=1, where=True)

    Return the product of the array elements over the given axis

    Refer to `numpy.prod` for full documentation.

    See Also
    --------
    numpy.prod : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('put',
    """
    a.put(indices, values, mode='raise')

    Set ``a.flat[n] = values[n]`` for all `n` in indices.

    Refer to `numpy.put` for full documentation.

    See Also
    --------
    numpy.put : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('ravel',
    """
    a.ravel([order])

    Return a flattened array.

    Refer to `numpy.ravel` for full documentation.

    See Also
    --------
    numpy.ravel : equivalent function

    ndarray.flat : a flat iterator on the array.

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('repeat',
    """
    a.repeat(repeats, axis=None)

    Repeat elements of an array.

    Refer to `numpy.repeat` for full documentation.

    See Also
    --------
    numpy.repeat : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('reshape',
    """
    a.reshape(shape, /, *, order='C', copy=None)

    Returns an array containing the same data with a new shape.

    Refer to `numpy.reshape` for full documentation.

    See Also
    --------
    numpy.reshape : equivalent function

    Notes
    -----
    Unlike the free function `numpy.reshape`, this method on `ndarray` allows
    the elements of the shape parameter to be passed in as separate arguments.
    For example, ``a.reshape(10, 11)`` is equivalent to
    ``a.reshape((10, 11))``.

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('resize',
    """
    a.resize(new_shape, refcheck=True)

    Change shape and size of array in-place.

    Parameters
    ----------
    new_shape : tuple of ints, or `n` ints
        Shape of resized array.
    refcheck : bool, optional
        If False, reference count will not be checked. Default is True.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `a` does not own its own data or references or views to it exist,
        and the data memory must be changed.
        PyPy only: will always raise if the data memory must be changed, since
        there is no reliable way to determine if references or views to it
        exist.

    SystemError
        If the `order` keyword argument is specified. This behaviour is a
        bug in NumPy.

    See Also
    --------
    resize : Return a new array with the specified shape.

    Notes
    -----
    This reallocates space for the data area if necessary.

    Only contiguous arrays (data elements consecutive in memory) can be
    resized.

    The purpose of the reference count check is to make sure you
    do not use this array as a buffer for another Python object and then
    reallocate the memory. However, reference counts can increase in
    other ways so if you are sure that you have not shared the memory
    for this array with another Python object, then you may safely set
    `refcheck` to False.

    Examples
    --------
    Shrinking an array: array is flattened (in the order that the data are
    stored in memory), resized, and reshaped:

    >>> import numpy as np

    >>> a = np.array([[0, 1], [2, 3]], order='C')
    >>> a.resize((2, 1))
    >>> a
    array([[0],
           [1]])

    >>> a = np.array([[0, 1], [2, 3]], order='F')
    >>> a.resize((2, 1))
    >>> a
    array([[0],
           [2]])

    Enlarging an array: as above, but missing entries are filled with zeros:

    >>> b = np.array([[0, 1], [2, 3]])
    >>> b.resize(2, 3) # new_shape parameter doesn't have to be a tuple
    >>> b
    array([[0, 1, 2],
           [3, 0, 0]])

    Referencing an array prevents resizing...

    >>> c = a
    >>> a.resize((1, 1))
    Traceback (most recent call last):
    ...
    ValueError: cannot resize an array that references or is referenced ...

    Unless `refcheck` is False:

    >>> a.resize((1, 1), refcheck=False)
    >>> a
    array([[0]])
    >>> c
    array([[0]])

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('round',
    """
    a.round(decimals=0, out=None)

    Return `a` with each element rounded to the given number of decimals.

    Refer to `numpy.around` for full documentation.

    See Also
    --------
    numpy.around : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('searchsorted',
    """
    a.searchsorted(v, side='left', sorter=None)

    Find indices where elements of v should be inserted in a to maintain order.

    For full documentation, see `numpy.searchsorted`

    See Also
    --------
    numpy.searchsorted : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('setfield',
    """
    a.setfield(val, dtype, offset=0)

    Put a value into a specified place in a field defined by a data-type.

    Place `val` into `a`'s field defined by `dtype` and beginning `offset`
    bytes into the field.

    Parameters
    ----------
    val : object
        Value to be placed in field.
    dtype : dtype object
        Data-type of the field in which to place `val`.
    offset : int, optional
        The number of bytes into the field at which to place `val`.

    Returns
    -------
    None

    See Also
    --------
    getfield

    Examples
    --------
    >>> import numpy as np
    >>> x = np.eye(3)
    >>> x.getfield(np.float64)
    array([[1.,  0.,  0.],
           [0.,  1.,  0.],
           [0.,  0.,  1.]])
    >>> x.setfield(3, np.int32)
    >>> x.getfield(np.int32)
    array([[3, 3, 3],
           [3, 3, 3],
           [3, 3, 3]], dtype=int32)
    >>> x
    array([[1.0e+000, 1.5e-323, 1.5e-323],
           [1.5e-323, 1.0e+000, 1.5e-323],
           [1.5e-323, 1.5e-323, 1.0e+000]])
    >>> x.setfield(np.eye(3), np.int32)
    >>> x
    array([[1.,  0.,  0.],
           [0.,  1.,  0.],
           [0.,  0.,  1.]])

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('setflags',
    """
    a.setflags(write=None, align=None, uic=None)

    Set array flags WRITEABLE, ALIGNED, WRITEBACKIFCOPY,
    respectively.

    These Boolean-valued flags affect how numpy interprets the memory
    area used by `a` (see Notes below). The ALIGNED flag can only
    be set to True if the data is actually aligned according to the type.
    The WRITEBACKIFCOPY flag can never be set
    to True. The flag WRITEABLE can only be set to True if the array owns its
    own memory, or the ultimate owner of the memory exposes a writeable buffer
    interface, or is a string. (The exception for string is made so that
    unpickling can be done without copying memory.)

    Parameters
    ----------
    write : bool, optional
        Describes whether or not `a` can be written to.
    align : bool, optional
        Describes whether or not `a` is aligned properly for its type.
    uic : bool, optional
        Describes whether or not `a` is a copy of another "base" array.

    Notes
    -----
    Array flags provide information about how the memory area used
    for the array is to be interpreted. There are 7 Boolean flags
    in use, only three of which can be changed by the user:
    WRITEBACKIFCOPY, WRITEABLE, and ALIGNED.

    WRITEABLE (W) the data area can be written to;

    ALIGNED (A) the data and strides are aligned appropriately for the hardware
    (as determined by the compiler);

    WRITEBACKIFCOPY (X) this array is a copy of some other array (referenced
    by .base). When the C-API function PyArray_ResolveWritebackIfCopy is
    called, the base array will be updated with the contents of this array.

    All flags can be accessed using the single (upper case) letter as well
    as the full name.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([[3, 1, 7],
    ...               [2, 0, 0],
    ...               [8, 5, 9]])
    >>> y
    array([[3, 1, 7],
           [2, 0, 0],
           [8, 5, 9]])
    >>> y.flags
      C_CONTIGUOUS : True
      F_CONTIGUOUS : False
      OWNDATA : True
      WRITEABLE : True
      ALIGNED : True
      WRITEBACKIFCOPY : False
    >>> y.setflags(write=0, align=0)
    >>> y.flags
      C_CONTIGUOUS : True
      F_CONTIGUOUS : False
      OWNDATA : True
      WRITEABLE : False
      ALIGNED : False
      WRITEBACKIFCOPY : False
    >>> y.setflags(uic=1)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: cannot set WRITEBACKIFCOPY flag to True

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('sort',
    """
    a.sort(axis=-1, kind=None, order=None)

    Sort an array in-place. Refer to `numpy.sort` for full documentation.

    Parameters
    ----------
    axis : int, optional
        Axis along which to sort. Default is -1, which means sort along the
        last axis.
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
        Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
        and 'mergesort' use timsort under the covers and, in general, the
        actual implementation will vary with datatype. The 'mergesort' option
        is retained for backwards compatibility.
    order : str or list of str, optional
        When `a` is an array with fields defined, this argument specifies
        which fields to compare first, second, etc.  A single field can
        be specified as a string, and not all fields need be specified,
        but unspecified fields will still be used, in the order in which
        they come up in the dtype, to break ties.

    See Also
    --------
    numpy.sort : Return a sorted copy of an array.
    numpy.argsort : Indirect sort.
    numpy.lexsort : Indirect stable sort on multiple keys.
    numpy.searchsorted : Find elements in sorted array.
    numpy.partition: Partial sort.

    Notes
    -----
    See `numpy.sort` for notes on the different sorting algorithms.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1,4], [3,1]])
    >>> a.sort(axis=1)
    >>> a
    array([[1, 4],
           [1, 3]])
    >>> a.sort(axis=0)
    >>> a
    array([[1, 3],
           [1, 4]])

    Use the `order` keyword to specify a field to use when sorting a
    structured array:

    >>> a = np.array([('a', 2), ('c', 1)], dtype=[('x', 'S1'), ('y', int)])
    >>> a.sort(order='y')
    >>> a
    array([(b'c', 1), (b'a', 2)],
          dtype=[('x', 'S1'), ('y', '<i8')])

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('partition',
    """
    a.partition(kth, axis=-1, kind='introselect', order=None)

    Partially sorts the elements in the array in such a way that the value of
    the element in k-th position is in the position it would be in a sorted
    array. In the output array, all elements smaller than the k-th element
    are located to the left of this element and all equal or greater are
    located to its right. The ordering of the elements in the two partitions
    on the either side of the k-th element in the output array is undefined.

    Parameters
    ----------
    kth : int or sequence of ints
        Element index to partition by. The kth element value will be in its
        final sorted position and all smaller elements will be moved before it
        and all equal or greater elements behind it.
        The order of all elements in the partitions is undefined.
        If provided with a sequence of kth it will partition all elements
        indexed by kth of them into their sorted position at once.

        .. deprecated:: 1.22.0
            Passing booleans as index is deprecated.
    axis : int, optional
        Axis along which to sort. Default is -1, which means sort along the
        last axis.
    kind : {'introselect'}, optional
        Selection algorithm. Default is 'introselect'.
    order : str or list of str, optional
        When `a` is an array with fields defined, this argument specifies
        which fields to compare first, second, etc. A single field can
        be specified as a string, and not all fields need to be specified,
        but unspecified fields will still be used, in the order in which
        they come up in the dtype, to break ties.

    See Also
    --------
    numpy.partition : Return a partitioned copy of an array.
    argpartition : Indirect partition.
    sort : Full sort.

    Notes
    -----
    See ``np.partition`` for notes on the different algorithms.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([3, 4, 2, 1])
    >>> a.partition(3)
    >>> a
    array([2, 1, 3, 4]) # may vary

    >>> a.partition((1, 3))
    >>> a
    array([1, 2, 3, 4])
    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('squeeze',
    """
    a.squeeze(axis=None)

    Remove axes of length one from `a`.

    Refer to `numpy.squeeze` for full documentation.

    See Also
    --------
    numpy.squeeze : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('std',
    """
    a.std(axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True)

    Returns the standard deviation of the array elements along given axis.

    Refer to `numpy.std` for full documentation.

    See Also
    --------
    numpy.std : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('sum',
    """
    a.sum(axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True)

    Return the sum of the array elements over the given axis.

    Refer to `numpy.sum` for full documentation.

    See Also
    --------
    numpy.sum : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('swapaxes',
    """
    a.swapaxes(axis1, axis2)

    Return a view of the array with `axis1` and `axis2` interchanged.

    Refer to `numpy.swapaxes` for full documentation.

    See Also
    --------
    numpy.swapaxes : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('take',
    """
    a.take(indices, axis=None, out=None, mode='raise')

    Return an array formed from the elements of `a` at the given indices.

    Refer to `numpy.take` for full documentation.

    See Also
    --------
    numpy.take : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('tofile',
    """
    a.tofile(fid, sep="", format="%s")

    Write array to a file as text or binary (default).

    Data is always written in 'C' order, independent of the order of `a`.
    The data produced by this method can be recovered using the function
    fromfile().

    Parameters
    ----------
    fid : file or str or Path
        An open file object, or a string containing a filename.
    sep : str
        Separator between array items for text output.
        If "" (empty), a binary file is written, equivalent to
        ``file.write(a.tobytes())``.
    format : str
        Format string for text file output.
        Each entry in the array is formatted to text by first converting
        it to the closest Python type, and then using "format" % item.

    Notes
    -----
    This is a convenience function for quick storage of array data.
    Information on endianness and precision is lost, so this method is not a
    good choice for files intended to archive data or transport data between
    machines with different endianness. Some of these problems can be overcome
    by outputting the data as text files, at the expense of speed and file
    size.

    When fid is a file object, array contents are directly written to the
    file, bypassing the file object's ``write`` method. As a result, tofile
    cannot be used with files objects supporting compression (e.g., GzipFile)
    or file-like objects that do not support ``fileno()`` (e.g., BytesIO).

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('tolist',
    """
    a.tolist()

    Return the array as an ``a.ndim``-levels deep nested list of Python scalars.

    Return a copy of the array data as a (nested) Python list.
    Data items are converted to the nearest compatible builtin Python type, via
    the `~numpy.ndarray.item` function.

    If ``a.ndim`` is 0, then since the depth of the nested list is 0, it will
    not be a list at all, but a simple Python scalar.

    Parameters
    ----------
    none

    Returns
    -------
    y : object, or list of object, or list of list of object, or ...
        The possibly nested list of array elements.

    Notes
    -----
    The array may be recreated via ``a = np.array(a.tolist())``, although this
    may sometimes lose precision.

    Examples
    --------
    For a 1D array, ``a.tolist()`` is almost the same as ``list(a)``,
    except that ``tolist`` changes numpy scalars to Python scalars:

    >>> import numpy as np
    >>> a = np.uint32([1, 2])
    >>> a_list = list(a)
    >>> a_list
    [np.uint32(1), np.uint32(2)]
    >>> type(a_list[0])
    <class 'numpy.uint32'>
    >>> a_tolist = a.tolist()
    >>> a_tolist
    [1, 2]
    >>> type(a_tolist[0])
    <class 'int'>

    Additionally, for a 2D array, ``tolist`` applies recursively:

    >>> a = np.array([[1, 2], [3, 4]])
    >>> list(a)
    [array([1, 2]), array([3, 4])]
    >>> a.tolist()
    [[1, 2], [3, 4]]

    The base case for this recursion is a 0D array:

    >>> a = np.array(1)
    >>> list(a)
    Traceback (most recent call last):
      ...
    TypeError: iteration over a 0-d array
    >>> a.tolist()
    1
    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('tobytes', """
    a.tobytes(order='C')

    Construct Python bytes containing the raw data bytes in the array.

    Constructs Python bytes showing a copy of the raw contents of
    data memory. The bytes object is produced in C-order by default.
    This behavior is controlled by the ``order`` parameter.

    Parameters
    ----------
    order : {'C', 'F', 'A'}, optional
        Controls the memory layout of the bytes object. 'C' means C-order,
        'F' means F-order, 'A' (short for *Any*) means 'F' if `a` is
        Fortran contiguous, 'C' otherwise. Default is 'C'.

    Returns
    -------
    s : bytes
        Python bytes exhibiting a copy of `a`'s raw data.

    See also
    --------
    frombuffer
        Inverse of this operation, construct a 1-dimensional array from Python
        bytes.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[0, 1], [2, 3]], dtype='<u2')
    >>> x.tobytes()
    b'\\x00\\x00\\x01\\x00\\x02\\x00\\x03\\x00'
    >>> x.tobytes('C') == x.tobytes()
    True
    >>> x.tobytes('F')
    b'\\x00\\x00\\x02\\x00\\x01\\x00\\x03\\x00'

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('trace',
    """
    a.trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)

    Return the sum along diagonals of the array.

    Refer to `numpy.trace` for full documentation.

    See Also
    --------
    numpy.trace : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('transpose',
    """
    a.transpose(*axes)

    Returns a view of the array with axes transposed.

    Refer to `numpy.transpose` for full documentation.

    Parameters
    ----------
    axes : None, tuple of ints, or `n` ints

     * None or no argument: reverses the order of the axes.

     * tuple of ints: `i` in the `j`-th place in the tuple means that the
       array's `i`-th axis becomes the transposed array's `j`-th axis.

     * `n` ints: same as an n-tuple of the same ints (this form is
       intended simply as a "convenience" alternative to the tuple form).

    Returns
    -------
    p : ndarray
        View of the array with its axes suitably permuted.

    See Also
    --------
    transpose : Equivalent function.
    ndarray.T : Array property returning the array transposed.
    ndarray.reshape : Give a new shape to an array without changing its data.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> a
    array([[1, 2],
           [3, 4]])
    >>> a.transpose()
    array([[1, 3],
           [2, 4]])
    >>> a.transpose((1, 0))
    array([[1, 3],
           [2, 4]])
    >>> a.transpose(1, 0)
    array([[1, 3],
           [2, 4]])

    >>> a = np.array([1, 2, 3, 4])
    >>> a
    array([1, 2, 3, 4])
    >>> a.transpose()
    array([1, 2, 3, 4])

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('var',
    """
    a.var(axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True)

    Returns the variance of the array elements, along given axis.

    Refer to `numpy.var` for full documentation.

    See Also
    --------
    numpy.var : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('view',
    """
    a.view([dtype][, type])

    New view of array with the same data.

    .. note::
        Passing None for ``dtype`` is different from omitting the parameter,
        since the former invokes ``dtype(None)`` which is an alias for
        ``dtype('float64')``.

    Parameters
    ----------
    dtype : data-type or ndarray sub-class, optional
        Data-type descriptor of the returned view, e.g., float32 or int16.
        Omitting it results in the view having the same data-type as `a`.
        This argument can also be specified as an ndarray sub-class, which
        then specifies the type of the returned object (this is equivalent to
        setting the ``type`` parameter).
    type : Python type, optional
        Type of the returned view, e.g., ndarray or matrix.  Again, omission
        of the parameter results in type preservation.

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

    For ``a.view(some_dtype)``, if ``some_dtype`` has a different number of
    bytes per entry than the previous dtype (for example, converting a regular
    array to a structured array), then the last axis of ``a`` must be
    contiguous. This axis will be resized in the result.

    .. versionchanged:: 1.23.0
       Only the last axis needs to be contiguous. Previously, the entire array
       had to be C-contiguous.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([(-1, 2)], dtype=[('a', np.int8), ('b', np.int8)])

    Viewing array data using a different type and dtype:

    >>> nonneg = np.dtype([("a", np.uint8), ("b", np.uint8)])
    >>> y = x.view(dtype=nonneg, type=np.recarray)
    >>> x["a"]
    array([-1], dtype=int8)
    >>> y.a
    array([255], dtype=uint8)

    Creating a view on a structured array so it can be used in calculations

    >>> x = np.array([(1, 2),(3,4)], dtype=[('a', np.int8), ('b', np.int8)])
    >>> xv = x.view(dtype=np.int8).reshape(-1,2)
    >>> xv
    array([[1, 2],
           [3, 4]], dtype=int8)
    >>> xv.mean(0)
    array([2.,  3.])

    Making changes to the view changes the underlying array

    >>> xv[0,1] = 20
    >>> x
    array([(1, 20), (3,  4)], dtype=[('a', 'i1'), ('b', 'i1')])

    Using a view to convert an array to a recarray:

    >>> z = x.view(np.recarray)
    >>> z.a
    array([1, 3], dtype=int8)

    Views share data:

    >>> x[0] = (9, 10)
    >>> z[0]
    np.record((9, 10), dtype=[('a', 'i1'), ('b', 'i1')])

    Views that change the dtype size (bytes per entry) should normally be
    avoided on arrays defined by slices, transposes, fortran-ordering, etc.:

    >>> x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
    >>> y = x[:, ::2]
    >>> y
    array([[1, 3],
           [4, 6]], dtype=int16)
    >>> y.view(dtype=[('width', np.int16), ('length', np.int16)])
    Traceback (most recent call last):
        ...
    ValueError: To change to a dtype of a different size, the last axis must be contiguous
    >>> z = y.copy()
    >>> z.view(dtype=[('width', np.int16), ('length', np.int16)])
    array([[(1, 3)],
           [(4, 6)]], dtype=[('width', '<i2'), ('length', '<i2')])

    However, views that change dtype are totally fine for arrays with a
    contiguous last axis, even if the rest of the axes are not C-contiguous:

    >>> x = np.arange(2 * 3 * 4, dtype=np.int8).reshape(2, 3, 4)
    >>> x.transpose(1, 0, 2).view(np.int16)
    array([[[ 256,  770],
            [3340, 3854]],
    <BLANKLINE>
           [[1284, 1798],
            [4368, 4882]],
    <BLANKLINE>
           [[2312, 2826],
            [5396, 5910]]], dtype=int16)

    """))


##############################################################################
#
# umath functions
#
##############################################################################

add_newdoc('numpy._core.umath', 'frompyfunc',
    """
    frompyfunc(func, /, nin, nout, *[, identity])

    Takes an arbitrary Python function and returns a NumPy ufunc.

    Can be used, for example, to add broadcasting to a built-in Python
    function (see Examples section).

    Parameters
    ----------
    func : Python function object
        An arbitrary Python function.
    nin : int
        The number of input arguments.
    nout : int
        The number of objects returned by `func`.
    identity : object, optional
        The value to use for the `~numpy.ufunc.identity` attribute of the resulting
        object. If specified, this is equivalent to setting the underlying
        C ``identity`` field to ``PyUFunc_IdentityValue``.
        If omitted, the identity is set to ``PyUFunc_None``. Note that this is
        _not_ equivalent to setting the identity to ``None``, which implies the
        operation is reorderable.

    Returns
    -------
    out : ufunc
        Returns a NumPy universal function (``ufunc``) object.

    See Also
    --------
    vectorize : Evaluates pyfunc over input arrays using broadcasting rules of numpy.

    Notes
    -----
    The returned ufunc always returns PyObject arrays.

    Examples
    --------
    Use frompyfunc to add broadcasting to the Python function ``oct``:

    >>> import numpy as np
    >>> oct_array = np.frompyfunc(oct, 1, 1)
    >>> oct_array(np.array((10, 30, 100)))
    array(['0o12', '0o36', '0o144'], dtype=object)
    >>> np.array((oct(10), oct(30), oct(100))) # for comparison
    array(['0o12', '0o36', '0o144'], dtype='<U5')

    """)


##############################################################################
#
# compiled_base functions
#
##############################################################################

add_newdoc('numpy._core.multiarray', 'add_docstring',
    """
    add_docstring(obj, docstring)

    Add a docstring to a built-in obj if possible.
    If the obj already has a docstring raise a RuntimeError
    If this routine does not know how to add a docstring to the object
    raise a TypeError
    """)

add_newdoc('numpy._core.umath', '_add_newdoc_ufunc',
    """
    add_ufunc_docstring(ufunc, new_docstring)

    Replace the docstring for a ufunc with new_docstring.
    This method will only work if the current docstring for
    the ufunc is NULL. (At the C level, i.e. when ufunc->doc is NULL.)

    Parameters
    ----------
    ufunc : numpy.ufunc
        A ufunc whose current doc is NULL.
    new_docstring : string
        The new docstring for the ufunc.

    Notes
    -----
    This method allocates memory for new_docstring on
    the heap. Technically this creates a memory leak, since this
    memory will not be reclaimed until the end of the program
    even if the ufunc itself is removed. However this will only
    be a problem if the user is repeatedly creating ufuncs with
    no documentation, adding documentation via add_newdoc_ufunc,
    and then throwing away the ufunc.
    """)

add_newdoc('numpy._core.multiarray', 'get_handler_name',
    """
    get_handler_name(a: ndarray) -> str,None

    Return the name of the memory handler used by `a`. If not provided, return
    the name of the memory handler that will be used to allocate data for the
    next `ndarray` in this context. May return None if `a` does not own its
    memory, in which case you can traverse ``a.base`` for a memory handler.
    """)

add_newdoc('numpy._core.multiarray', 'get_handler_version',
    """
    get_handler_version(a: ndarray) -> int,None

    Return the version of the memory handler used by `a`. If not provided,
    return the version of the memory handler that will be used to allocate data
    for the next `ndarray` in this context. May return None if `a` does not own
    its memory, in which case you can traverse ``a.base`` for a memory handler.
    """)

add_newdoc('numpy._core._multiarray_umath', '_array_converter',
    """
    _array_converter(*array_likes)

    Helper to convert one or more objects to arrays.  Integrates machinery
    to deal with the ``result_type`` and ``__array_wrap__``.

    The reason for this is that e.g. ``result_type`` needs to convert to arrays
    to find the ``dtype``.  But converting to an array before calling
    ``result_type`` would incorrectly "forget" whether it was a Python int,
    float, or complex.
    """)

add_newdoc(
    'numpy._core._multiarray_umath', '_array_converter', ('scalar_input',
    """
    A tuple which indicates for each input whether it was a scalar that
    was coerced to a 0-D array (and was not already an array or something
    converted via a protocol like ``__array__()``).
    """))

add_newdoc('numpy._core._multiarray_umath', '_array_converter', ('as_arrays',
    """
    as_arrays(/, subok=True, pyscalars="convert_if_no_array")

    Return the inputs as arrays or scalars.

    Parameters
    ----------
    subok : True or False, optional
        Whether array subclasses are preserved.
    pyscalars : {"convert", "preserve", "convert_if_no_array"}, optional
        To allow NEP 50 weak promotion later, it may be desirable to preserve
        Python scalars.  As default, these are preserved unless all inputs
        are Python scalars.  "convert" enforces an array return.
    """))

add_newdoc('numpy._core._multiarray_umath', '_array_converter', ('result_type',
    """result_type(/, extra_dtype=None, ensure_inexact=False)

    Find the ``result_type`` just as ``np.result_type`` would, but taking
    into account that the original inputs (before converting to an array) may
    have been Python scalars with weak promotion.

    Parameters
    ----------
    extra_dtype : dtype instance or class
        An additional DType or dtype instance to promote (e.g. could be used
        to ensure the result precision is at least float32).
    ensure_inexact : True or False
        When ``True``, ensures a floating point (or complex) result replacing
        the ``arr * 1.`` or ``result_type(..., 0.0)`` pattern.
    """))

add_newdoc('numpy._core._multiarray_umath', '_array_converter', ('wrap',
    """
    wrap(arr, /, to_scalar=None)

    Call ``__array_wrap__`` on ``arr`` if ``arr`` is not the same subclass
    as the input the ``__array_wrap__`` method was retrieved from.

    Parameters
    ----------
    arr : ndarray
        The object to be wrapped. Normally an ndarray or subclass,
        although for backward compatibility NumPy scalars are also accepted
        (these will be converted to a NumPy array before being passed on to
        the ``__array_wrap__`` method).
    to_scalar : {True, False, None}, optional
        When ``True`` will convert a 0-d array to a scalar via ``result[()]``
        (with a fast-path for non-subclasses).  If ``False`` the result should
        be an array-like (as ``__array_wrap__`` is free to return a non-array).
        By default (``None``), a scalar is returned if all inputs were scalar.
    """))


add_newdoc('numpy._core.multiarray', '_get_madvise_hugepage',
    """
    _get_madvise_hugepage() -> bool

    Get use of ``madvise (2)`` MADV_HUGEPAGE support when
    allocating the array data. Returns the currently set value.
    See `global_state` for more information.
    """)

add_newdoc('numpy._core.multiarray', '_set_madvise_hugepage',
    """
    _set_madvise_hugepage(enabled: bool) -> bool

    Set  or unset use of ``madvise (2)`` MADV_HUGEPAGE support when
    allocating the array data. Returns the previously set value.
    See `global_state` for more information.
    """)


##############################################################################
#
# Documentation for ufunc attributes and methods
#
##############################################################################


##############################################################################
#
# ufunc object
#
##############################################################################

add_newdoc('numpy._core', 'ufunc',
    """
    Functions that operate element by element on whole arrays.

    To see the documentation for a specific ufunc, use `info`.  For
    example, ``np.info(np.sin)``.  Because ufuncs are written in C
    (for speed) and linked into Python with NumPy's ufunc facility,
    Python's help() function finds this page whenever help() is called
    on a ufunc.

    A detailed explanation of ufuncs can be found in the docs for :ref:`ufuncs`.

    **Calling ufuncs:** ``op(*x[, out], where=True, **kwargs)``

    Apply `op` to the arguments `*x` elementwise, broadcasting the arguments.

    The broadcasting rules are:

    * Dimensions of length 1 may be prepended to either array.
    * Arrays may be repeated along dimensions of length 1.

    Parameters
    ----------
    *x : array_like
        Input arrays.
    out : ndarray, None, ..., or tuple of ndarray and None, optional
        Location(s) into which the result(s) are stored.
        If not provided or None, new array(s) are created by the ufunc.
        If passed as a keyword argument, can be Ellipses (``out=...``) to
        ensure an array is returned even if the result is 0-dimensional,
        or a tuple with length equal to the number of outputs (where None
        can be used for allocation by the ufunc).

        .. versionadded:: 2.3
            Support for ``out=...`` was added.

    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    r : ndarray or tuple of ndarray
        `r` will have the shape that the arrays in `x` broadcast to; if `out` is
        provided, it will be returned. If not, `r` will be allocated and
        may contain uninitialized values. If the function has more than one
        output, then the result will be a tuple of arrays.

    """)


##############################################################################
#
# ufunc attributes
#
##############################################################################

add_newdoc('numpy._core', 'ufunc', ('identity',
    """
    The identity value.

    Data attribute containing the identity element for the ufunc,
    if it has one. If it does not, the attribute value is None.

    Examples
    --------
    >>> import numpy as np
    >>> np.add.identity
    0
    >>> np.multiply.identity
    1
    >>> print(np.power.identity)
    None
    >>> print(np.exp.identity)
    None
    """))

add_newdoc('numpy._core', 'ufunc', ('nargs',
    """
    The number of arguments.

    Data attribute containing the number of arguments the ufunc takes, including
    optional ones.

    Notes
    -----
    Typically this value will be one more than what you might expect
    because all ufuncs take  the optional "out" argument.

    Examples
    --------
    >>> import numpy as np
    >>> np.add.nargs
    3
    >>> np.multiply.nargs
    3
    >>> np.power.nargs
    3
    >>> np.exp.nargs
    2
    """))

add_newdoc('numpy._core', 'ufunc', ('nin',
    """
    The number of inputs.

    Data attribute containing the number of arguments the ufunc treats as input.

    Examples
    --------
    >>> import numpy as np
    >>> np.add.nin
    2
    >>> np.multiply.nin
    2
    >>> np.power.nin
    2
    >>> np.exp.nin
    1
    """))

add_newdoc('numpy._core', 'ufunc', ('nout',
    """
    The number of outputs.

    Data attribute containing the number of arguments the ufunc treats as output.

    Notes
    -----
    Since all ufuncs can take output arguments, this will always be at least 1.

    Examples
    --------
    >>> import numpy as np
    >>> np.add.nout
    1
    >>> np.multiply.nout
    1
    >>> np.power.nout
    1
    >>> np.exp.nout
    1

    """))

add_newdoc('numpy._core', 'ufunc', ('ntypes',
    """
    The number of types.

    The number of numerical NumPy types - of which there are 18 total - on which
    the ufunc can operate.

    See Also
    --------
    numpy.ufunc.types

    Examples
    --------
    >>> import numpy as np
    >>> np.add.ntypes
    22
    >>> np.multiply.ntypes
    23
    >>> np.power.ntypes
    21
    >>> np.exp.ntypes
    10
    >>> np.remainder.ntypes
    16

    """))

add_newdoc('numpy._core', 'ufunc', ('types',
    """
    Returns a list with types grouped input->output.

    Data attribute listing the data-type "Domain-Range" groupings the ufunc can
    deliver. The data-types are given using the character codes.

    See Also
    --------
    numpy.ufunc.ntypes

    Examples
    --------
    >>> import numpy as np
    >>> np.add.types
    ['??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', ...

    >>> np.power.types
    ['bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', ...

    >>> np.exp.types
    ['e->e', 'f->f', 'd->d', 'f->f', 'd->d', 'g->g', 'F->F', 'D->D', 'G->G', 'O->O']

    >>> np.remainder.types
    ['bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', ...

    """))

add_newdoc('numpy._core', 'ufunc', ('signature',
    """
    Definition of the core elements a generalized ufunc operates on.

    The signature determines how the dimensions of each input/output array
    are split into core and loop dimensions:

    1. Each dimension in the signature is matched to a dimension of the
       corresponding passed-in array, starting from the end of the shape tuple.
    2. Core dimensions assigned to the same label in the signature must have
       exactly matching sizes, no broadcasting is performed.
    3. The core dimensions are removed from all inputs and the remaining
       dimensions are broadcast together, defining the loop dimensions.

    Notes
    -----
    Generalized ufuncs are used internally in many linalg functions, and in
    the testing suite; the examples below are taken from these.
    For ufuncs that operate on scalars, the signature is None, which is
    equivalent to '()' for every argument.

    Examples
    --------
    >>> import numpy as np
    >>> np.linalg._umath_linalg.det.signature
    '(m,m)->()'
    >>> np.matmul.signature
    '(n?,k),(k,m?)->(n?,m?)'
    >>> np.add.signature is None
    True  # equivalent to '(),()->()'
    """))

##############################################################################
#
# ufunc methods
#
##############################################################################

add_newdoc('numpy._core', 'ufunc', ('reduce',
    """
    reduce(array, axis=0, dtype=None, out=None, keepdims=False, initial=<no value>, where=True)

    Reduces `array`'s dimension by one, by applying ufunc along one axis.

    Let :math:`array.shape = (N_0, ..., N_i, ..., N_{M-1})`.  Then
    :math:`ufunc.reduce(array, axis=i)[k_0, ..,k_{i-1}, k_{i+1}, .., k_{M-1}]` =
    the result of iterating `j` over :math:`range(N_i)`, cumulatively applying
    ufunc to each :math:`array[k_0, ..,k_{i-1}, j, k_{i+1}, .., k_{M-1}]`.
    For a one-dimensional array, reduce produces results equivalent to:
    ::

     r = op.identity # op = ufunc
     for i in range(len(A)):
       r = op(r, A[i])
     return r

    For example, add.reduce() is equivalent to sum().

    Parameters
    ----------
    array : array_like
        The array to act on.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a reduction is performed.
        The default (`axis` = 0) is perform a reduction over the first
        dimension of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis.

        If this is None, a reduction is performed over all the axes.
        If this is a tuple of ints, a reduction is performed on multiple
        axes, instead of a single axis or all the axes as before.

        For operations which are either not commutative or not associative,
        doing a reduction over multiple axes is not well-defined. The
        ufuncs do not currently raise an exception in this case, but will
        likely do so in the future.
    dtype : data-type code, optional
        The data type used to perform the operation. Defaults to that of
        ``out`` if given, and the data type of ``array`` otherwise (though
        upcast to conserve precision for some cases, such as
        ``numpy.add.reduce`` for integer or boolean input).
    out : ndarray, None, ..., or tuple of ndarray and None, optional
        Location into which the result is stored.
        If not provided or None, a freshly-allocated array is returned.
        If passed as a keyword argument, can be Ellipses (``out=...``) to
        ensure an array is returned even if the result is 0-dimensional
        (which is useful especially for object dtype), or a 1-element tuple
        (latter for consistency with ``ufunc.__call__``).

        .. versionadded:: 2.3
            Support for ``out=...`` was added.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `array`.
    initial : scalar, optional
        The value with which to start the reduction.
        If the ufunc has no identity or the dtype is object, this defaults
        to None - otherwise it defaults to ufunc.identity.
        If ``None`` is given, the first element of the reduction is used,
        and an error is thrown if the reduction is empty.
    where : array_like of bool, optional
        A boolean array which is broadcasted to match the dimensions
        of `array`, and selects elements to include in the reduction. Note
        that for ufuncs like ``minimum`` that do not have an identity
        defined, one has to pass in also ``initial``.

    Returns
    -------
    r : ndarray
        The reduced array. If `out` was supplied, `r` is a reference to it.

    Examples
    --------
    >>> import numpy as np
    >>> np.multiply.reduce([2,3,5])
    30

    A multi-dimensional array example:

    >>> X = np.arange(8).reshape((2,2,2))
    >>> X
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> np.add.reduce(X, 0)
    array([[ 4,  6],
           [ 8, 10]])
    >>> np.add.reduce(X) # confirm: default axis value is 0
    array([[ 4,  6],
           [ 8, 10]])
    >>> np.add.reduce(X, 1)
    array([[ 2,  4],
           [10, 12]])
    >>> np.add.reduce(X, 2)
    array([[ 1,  5],
           [ 9, 13]])

    You can use the ``initial`` keyword argument to initialize the reduction
    with a different value, and ``where`` to select specific elements to include:

    >>> np.add.reduce([10], initial=5)
    15
    >>> np.add.reduce(np.ones((2, 2, 2)), axis=(0, 2), initial=10)
    array([14., 14.])
    >>> a = np.array([10., np.nan, 10])
    >>> np.add.reduce(a, where=~np.isnan(a))
    20.0

    Allows reductions of empty arrays where they would normally fail, i.e.
    for ufuncs without an identity.

    >>> np.minimum.reduce([], initial=np.inf)
    inf
    >>> np.minimum.reduce([[1., 2.], [3., 4.]], initial=10., where=[True, False])
    array([ 1., 10.])
    >>> np.minimum.reduce([])
    Traceback (most recent call last):
        ...
    ValueError: zero-size array to reduction operation minimum which has no identity
    """))

add_newdoc('numpy._core', 'ufunc', ('accumulate',
    """
    accumulate(array, axis=0, dtype=None, out=None)

    Accumulate the result of applying the operator to all elements.

    For a one-dimensional array, accumulate produces results equivalent to::

      r = np.empty(len(A))
      t = op.identity        # op = the ufunc being applied to A's  elements
      for i in range(len(A)):
          t = op(t, A[i])
          r[i] = t
      return r

    For example, add.accumulate() is equivalent to np.cumsum().

    For a multi-dimensional array, accumulate is applied along only one
    axis (axis zero by default; see Examples below) so repeated use is
    necessary if one wants to accumulate over multiple axes.

    Parameters
    ----------
    array : array_like
        The array to act on.
    axis : int, optional
        The axis along which to apply the accumulation; default is zero.
    dtype : data-type code, optional
        The data-type used to represent the intermediate results. Defaults
        to the data-type of the output array if such is provided, or the
        data-type of the input array if no output array is provided.
    out : ndarray, None, or tuple of ndarray and None, optional
        Location into which the result is stored.
        If not provided or None, a freshly-allocated array is returned.
        For consistency with ``ufunc.__call__``, if passed as a keyword
        argument, can be Ellipses (``out=...``, which has the same effect
        as None as an array is always returned), or a 1-element tuple.

    Returns
    -------
    r : ndarray
        The accumulated values. If `out` was supplied, `r` is a reference to
        `out`.

    Examples
    --------
    1-D array examples:

    >>> import numpy as np
    >>> np.add.accumulate([2, 3, 5])
    array([ 2,  5, 10])
    >>> np.multiply.accumulate([2, 3, 5])
    array([ 2,  6, 30])

    2-D array examples:

    >>> I = np.eye(2)
    >>> I
    array([[1.,  0.],
           [0.,  1.]])

    Accumulate along axis 0 (rows), down columns:

    >>> np.add.accumulate(I, 0)
    array([[1.,  0.],
           [1.,  1.]])
    >>> np.add.accumulate(I) # no axis specified = axis zero
    array([[1.,  0.],
           [1.,  1.]])

    Accumulate along axis 1 (columns), through rows:

    >>> np.add.accumulate(I, 1)
    array([[1.,  1.],
           [0.,  1.]])

    """))

add_newdoc('numpy._core', 'ufunc', ('reduceat',
    """
    reduceat(array, indices, axis=0, dtype=None, out=None)

    Performs a (local) reduce with specified slices over a single axis.

    For i in ``range(len(indices))``, `reduceat` computes
    ``ufunc.reduce(array[indices[i]:indices[i+1]])``, which becomes the i-th
    generalized "row" parallel to `axis` in the final result (i.e., in a
    2-D array, for example, if `axis = 0`, it becomes the i-th row, but if
    `axis = 1`, it becomes the i-th column).  There are three exceptions to this:

    * when ``i = len(indices) - 1`` (so for the last index),
      ``indices[i+1] = array.shape[axis]``.
    * if ``indices[i] >= indices[i + 1]``, the i-th generalized "row" is
      simply ``array[indices[i]]``.
    * if ``indices[i] >= len(array)`` or ``indices[i] < 0``, an error is raised.

    The shape of the output depends on the size of `indices`, and may be
    larger than `array` (this happens if ``len(indices) > array.shape[axis]``).

    Parameters
    ----------
    array : array_like
        The array to act on.
    indices : array_like
        Paired indices, comma separated (not colon), specifying slices to
        reduce.
    axis : int, optional
        The axis along which to apply the reduceat.
    dtype : data-type code, optional
        The data type used to perform the operation. Defaults to that of
        ``out`` if given, and the data type of ``array`` otherwise (though
        upcast to conserve precision for some cases, such as
        ``numpy.add.reduce`` for integer or boolean input).
    out : ndarray, None, or tuple of ndarray and None, optional
        Location into which the result is stored.
        If not provided or None, a freshly-allocated array is returned.
        For consistency with ``ufunc.__call__``, if passed as a keyword
        argument, can be Ellipses (``out=...``, which has the same effect
        as None as an array is always returned), or a 1-element tuple.

    Returns
    -------
    r : ndarray
        The reduced values. If `out` was supplied, `r` is a reference to
        `out`.

    Notes
    -----
    A descriptive example:

    If `array` is 1-D, the function `ufunc.accumulate(array)` is the same as
    ``ufunc.reduceat(array, indices)[::2]`` where `indices` is
    ``range(len(array) - 1)`` with a zero placed
    in every other element:
    ``indices = zeros(2 * len(array) - 1)``,
    ``indices[1::2] = range(1, len(array))``.

    Don't be fooled by this attribute's name: `reduceat(array)` is not
    necessarily smaller than `array`.

    Examples
    --------
    To take the running sum of four successive values:

    >>> import numpy as np
    >>> np.add.reduceat(np.arange(8),[0,4, 1,5, 2,6, 3,7])[::2]
    array([ 6, 10, 14, 18])

    A 2-D example:

    >>> x = np.linspace(0, 15, 16).reshape(4,4)
    >>> x
    array([[ 0.,   1.,   2.,   3.],
           [ 4.,   5.,   6.,   7.],
           [ 8.,   9.,  10.,  11.],
           [12.,  13.,  14.,  15.]])

    ::

     # reduce such that the result has the following five rows:
     # [row1 + row2 + row3]
     # [row4]
     # [row2]
     # [row3]
     # [row1 + row2 + row3 + row4]

    >>> np.add.reduceat(x, [0, 3, 1, 2, 0])
    array([[12.,  15.,  18.,  21.],
           [12.,  13.,  14.,  15.],
           [ 4.,   5.,   6.,   7.],
           [ 8.,   9.,  10.,  11.],
           [24.,  28.,  32.,  36.]])

    ::

     # reduce such that result has the following two columns:
     # [col1 * col2 * col3, col4]

    >>> np.multiply.reduceat(x, [0, 3], 1)
    array([[   0.,     3.],
           [ 120.,     7.],
           [ 720.,    11.],
           [2184.,    15.]])

    """))

add_newdoc('numpy._core', 'ufunc', ('outer',
    r"""
    outer(A, B, /, **kwargs)

    Apply the ufunc `op` to all pairs (a, b) with a in `A` and b in `B`.

    Let ``M = A.ndim``, ``N = B.ndim``. Then the result, `C`, of
    ``op.outer(A, B)`` is an array of dimension M + N such that:

    .. math:: C[i_0, ..., i_{M-1}, j_0, ..., j_{N-1}] =
       op(A[i_0, ..., i_{M-1}], B[j_0, ..., j_{N-1}])

    For `A` and `B` one-dimensional, this is equivalent to::

      r = empty(len(A),len(B))
      for i in range(len(A)):
          for j in range(len(B)):
              r[i,j] = op(A[i], B[j])  # op = ufunc in question

    Parameters
    ----------
    A : array_like
        First array
    B : array_like
        Second array
    kwargs : any
        Arguments to pass on to the ufunc. Typically `dtype` or `out`.
        See `ufunc` for a comprehensive overview of all available arguments.

    Returns
    -------
    r : ndarray
        Output array

    See Also
    --------
    numpy.outer : A less powerful version of ``np.multiply.outer``
                  that `ravel`\ s all inputs to 1D. This exists
                  primarily for compatibility with old code.

    tensordot : ``np.tensordot(a, b, axes=((), ()))`` and
                ``np.multiply.outer(a, b)`` behave same for all
                dimensions of a and b.

    Examples
    --------
    >>> np.multiply.outer([1, 2, 3], [4, 5, 6])
    array([[ 4,  5,  6],
           [ 8, 10, 12],
           [12, 15, 18]])

    A multi-dimensional example:

    >>> A = np.array([[1, 2, 3], [4, 5, 6]])
    >>> A.shape
    (2, 3)
    >>> B = np.array([[1, 2, 3, 4]])
    >>> B.shape
    (1, 4)
    >>> C = np.multiply.outer(A, B)
    >>> C.shape; C
    (2, 3, 1, 4)
    array([[[[ 1,  2,  3,  4]],
            [[ 2,  4,  6,  8]],
            [[ 3,  6,  9, 12]]],
           [[[ 4,  8, 12, 16]],
            [[ 5, 10, 15, 20]],
            [[ 6, 12, 18, 24]]]])

    """))

add_newdoc('numpy._core', 'ufunc', ('at',
    """
    at(a, indices, b=None, /)

    Performs unbuffered in place operation on operand 'a' for elements
    specified by 'indices'. For addition ufunc, this method is equivalent to
    ``a[indices] += b``, except that results are accumulated for elements that
    are indexed more than once. For example, ``a[[0,0]] += 1`` will only
    increment the first element once because of buffering, whereas
    ``add.at(a, [0,0], 1)`` will increment the first element twice.

    Parameters
    ----------
    a : array_like
        The array to perform in place operation on.
    indices : array_like or tuple
        Array like index object or slice object for indexing into first
        operand. If first operand has multiple dimensions, indices can be a
        tuple of array like index objects or slice objects.
    b : array_like
        Second operand for ufuncs requiring two operands. Operand must be
        broadcastable over first operand after indexing or slicing.

    Examples
    --------
    Set items 0 and 1 to their negative values:

    >>> import numpy as np
    >>> a = np.array([1, 2, 3, 4])
    >>> np.negative.at(a, [0, 1])
    >>> a
    array([-1, -2,  3,  4])

    Increment items 0 and 1, and increment item 2 twice:

    >>> a = np.array([1, 2, 3, 4])
    >>> np.add.at(a, [0, 1, 2, 2], 1)
    >>> a
    array([2, 3, 5, 4])

    Add items 0 and 1 in first array to second array,
    and store results in first array:

    >>> a = np.array([1, 2, 3, 4])
    >>> b = np.array([1, 2])
    >>> np.add.at(a, [0, 1], b)
    >>> a
    array([2, 4, 3, 4])

    """))

add_newdoc('numpy._core', 'ufunc', ('resolve_dtypes',
    """
    resolve_dtypes(dtypes, *, signature=None, casting=None, reduction=False)

    Find the dtypes NumPy will use for the operation.  Both input and
    output dtypes are returned and may differ from those provided.

    .. note::

        This function always applies NEP 50 rules since it is not provided
        any actual values.  The Python types ``int``, ``float``, and
        ``complex`` thus behave weak and should be passed for "untyped"
        Python input.

    Parameters
    ----------
    dtypes : tuple of dtypes, None, or literal int, float, complex
        The input dtypes for each operand.  Output operands can be
        None, indicating that the dtype must be found.
    signature : tuple of DTypes or None, optional
        If given, enforces exact DType (classes) of the specific operand.
        The ufunc ``dtype`` argument is equivalent to passing a tuple with
        only output dtypes set.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        The casting mode when casting is necessary.  This is identical to
        the ufunc call casting modes.
    reduction : boolean
        If given, the resolution assumes a reduce operation is happening
        which slightly changes the promotion and type resolution rules.
        `dtypes` is usually something like ``(None, np.dtype("i2"), None)``
        for reductions (first input is also the output).

        .. note::

            The default casting mode is "same_kind", however, as of
            NumPy 1.24, NumPy uses "unsafe" for reductions.

    Returns
    -------
    dtypes : tuple of dtypes
        The dtypes which NumPy would use for the calculation.  Note that
        dtypes may not match the passed in ones (casting is necessary).


    Examples
    --------
    This API requires passing dtypes, define them for convenience:

    >>> import numpy as np
    >>> int32 = np.dtype("int32")
    >>> float32 = np.dtype("float32")

    The typical ufunc call does not pass an output dtype.  `numpy.add` has two
    inputs and one output, so leave the output as ``None`` (not provided):

    >>> np.add.resolve_dtypes((int32, float32, None))
    (dtype('float64'), dtype('float64'), dtype('float64'))

    The loop found uses "float64" for all operands (including the output), the
    first input would be cast.

    ``resolve_dtypes`` supports "weak" handling for Python scalars by passing
    ``int``, ``float``, or ``complex``:

    >>> np.add.resolve_dtypes((float32, float, None))
    (dtype('float32'), dtype('float32'), dtype('float32'))

    Where the Python ``float`` behaves similar to a Python value ``0.0``
    in a ufunc call.  (See :ref:`NEP 50 <NEP50>` for details.)

    """))

add_newdoc('numpy._core', 'ufunc', ('_resolve_dtypes_and_context',
    """
    _resolve_dtypes_and_context(dtypes, *, signature=None, casting=None, reduction=False)

    See `numpy.ufunc.resolve_dtypes` for parameter information.  This
    function is considered *unstable*.  You may use it, but the returned
    information is NumPy version specific and expected to change.
    Large API/ABI changes are not expected, but a new NumPy version is
    expected to require updating code using this functionality.

    This function is designed to be used in conjunction with
    `numpy.ufunc._get_strided_loop`.  The calls are split to mirror the C API
    and allow future improvements.

    Returns
    -------
    dtypes : tuple of dtypes
    call_info :
        PyCapsule with all necessary information to get access to low level
        C calls.  See `numpy.ufunc._get_strided_loop` for more information.

    """))

add_newdoc('numpy._core', 'ufunc', ('_get_strided_loop',
    """
    _get_strided_loop(call_info, /, *, fixed_strides=None)

    This function fills in the ``call_info`` capsule to include all
    information necessary to call the low-level strided loop from NumPy.

    See notes for more information.

    Parameters
    ----------
    call_info : PyCapsule
        The PyCapsule returned by `numpy.ufunc._resolve_dtypes_and_context`.
    fixed_strides : tuple of int or None, optional
        A tuple with fixed byte strides of all input arrays.  NumPy may use
        this information to find specialized loops, so any call must follow
        the given stride.  Use ``None`` to indicate that the stride is not
        known (or not fixed) for all calls.

    Notes
    -----
    Together with `numpy.ufunc._resolve_dtypes_and_context` this function
    gives low-level access to the NumPy ufunc loops.
    The first function does general preparation and returns the required
    information. It returns this as a C capsule with the version specific
    name ``numpy_1.24_ufunc_call_info``.
    The NumPy 1.24 ufunc call info capsule has the following layout::

        typedef struct {
            PyArrayMethod_StridedLoop *strided_loop;
            PyArrayMethod_Context *context;
            NpyAuxData *auxdata;

            /* Flag information (expected to change) */
            npy_bool requires_pyapi;  /* GIL is required by loop */

            /* Loop doesn't set FPE flags; if not set check FPE flags */
            npy_bool no_floatingpoint_errors;
        } ufunc_call_info;

    Note that the first call only fills in the ``context``.  The call to
    ``_get_strided_loop`` fills in all other data.  The main thing to note is
    that the new-style loops return 0 on success, -1 on failure.  They are
    passed context as new first input and ``auxdata`` as (replaced) last.

    Only the ``strided_loop``signature is considered guaranteed stable
    for NumPy bug-fix releases.  All other API is tied to the experimental
    API versioning.

    The reason for the split call is that cast information is required to
    decide what the fixed-strides will be.

    NumPy ties the lifetime of the ``auxdata`` information to the capsule.

    """))


##############################################################################
#
# Documentation for dtype attributes and methods
#
##############################################################################

##############################################################################
#
# dtype object
#
##############################################################################

add_newdoc('numpy._core.multiarray', 'dtype',
    """
    dtype(dtype, align=False, copy=False, [metadata])

    Create a data type object.

    A numpy array is homogeneous, and contains elements described by a
    dtype object. A dtype object can be constructed from different
    combinations of fundamental numeric types.

    Parameters
    ----------
    dtype
        Object to be converted to a data type object.
    align : bool, optional
        Add padding to the fields to match what a C compiler would output
        for a similar C-struct. Can be ``True`` only if `obj` is a dictionary
        or a comma-separated string. If a struct dtype is being created,
        this also sets a sticky alignment flag ``isalignedstruct``.
    copy : bool, optional
        Make a new copy of the data-type object. If ``False``, the result
        may just be a reference to a built-in data-type object.
    metadata : dict, optional
        An optional dictionary with dtype metadata.

    See also
    --------
    result_type

    Examples
    --------
    Using array-scalar type:

    >>> import numpy as np
    >>> np.dtype(np.int16)
    dtype('int16')

    Structured type, one field name 'f1', containing int16:

    >>> np.dtype([('f1', np.int16)])
    dtype([('f1', '<i2')])

    Structured type, one field named 'f1', in itself containing a structured
    type with one field:

    >>> np.dtype([('f1', [('f1', np.int16)])])
    dtype([('f1', [('f1', '<i2')])])

    Structured type, two fields: the first field contains an unsigned int, the
    second an int32:

    >>> np.dtype([('f1', np.uint64), ('f2', np.int32)])
    dtype([('f1', '<u8'), ('f2', '<i4')])

    Using array-protocol type strings:

    >>> np.dtype([('a','f8'),('b','S10')])
    dtype([('a', '<f8'), ('b', 'S10')])

    Using comma-separated field formats.  The shape is (2,3):

    >>> np.dtype("i4, (2,3)f8")
    dtype([('f0', '<i4'), ('f1', '<f8', (2, 3))])

    Using tuples.  ``int`` is a fixed type, 3 the field's shape.  ``void``
    is a flexible type, here of size 10:

    >>> np.dtype([('hello',(np.int64,3)),('world',np.void,10)])
    dtype([('hello', '<i8', (3,)), ('world', 'V10')])

    Subdivide ``int16`` into 2 ``int8``'s, called x and y.  0 and 1 are
    the offsets in bytes:

    >>> np.dtype((np.int16, {'x':(np.int8,0), 'y':(np.int8,1)}))
    dtype((numpy.int16, [('x', 'i1'), ('y', 'i1')]))

    Using dictionaries.  Two fields named 'gender' and 'age':

    >>> np.dtype({'names':['gender','age'], 'formats':['S1',np.uint8]})
    dtype([('gender', 'S1'), ('age', 'u1')])

    Offsets in bytes, here 0 and 25:

    >>> np.dtype({'surname':('S25',0),'age':(np.uint8,25)})
    dtype([('surname', 'S25'), ('age', 'u1')])

    """)

##############################################################################
#
# dtype attributes
#
##############################################################################

add_newdoc('numpy._core.multiarray', 'dtype', ('alignment',
    """
    The required alignment (bytes) of this data-type according to the compiler.

    More information is available in the C-API section of the manual.

    Examples
    --------

    >>> import numpy as np
    >>> x = np.dtype('i4')
    >>> x.alignment
    4

    >>> x = np.dtype(float)
    >>> x.alignment
    8

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('byteorder',
    """
    A character indicating the byte-order of this data-type object.

    One of:

    ===  ==============
    '='  native
    '<'  little-endian
    '>'  big-endian
    '|'  not applicable
    ===  ==============

    All built-in data-type objects have byteorder either '=' or '|'.

    Examples
    --------

    >>> import numpy as np
    >>> dt = np.dtype('i2')
    >>> dt.byteorder
    '='
    >>> # endian is not relevant for 8 bit numbers
    >>> np.dtype('i1').byteorder
    '|'
    >>> # or ASCII strings
    >>> np.dtype('S2').byteorder
    '|'
    >>> # Even if specific code is given, and it is native
    >>> # '=' is the byteorder
    >>> import sys
    >>> sys_is_le = sys.byteorder == 'little'
    >>> native_code = '<' if sys_is_le else '>'
    >>> swapped_code = '>' if sys_is_le else '<'
    >>> dt = np.dtype(native_code + 'i2')
    >>> dt.byteorder
    '='
    >>> # Swapped code shows up as itself
    >>> dt = np.dtype(swapped_code + 'i2')
    >>> dt.byteorder == swapped_code
    True

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('char',
    """A unique character code for each of the 21 different built-in types.

    Examples
    --------

    >>> import numpy as np
    >>> x = np.dtype(float)
    >>> x.char
    'd'

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('descr',
    """
    `__array_interface__` description of the data-type.

    The format is that required by the 'descr' key in the
    `__array_interface__` attribute.

    Warning: This attribute exists specifically for `__array_interface__`,
    and passing it directly to `numpy.dtype` will not accurately reconstruct
    some dtypes (e.g., scalar and subarray dtypes).

    Examples
    --------

    >>> import numpy as np
    >>> x = np.dtype(float)
    >>> x.descr
    [('', '<f8')]

    >>> dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])
    >>> dt.descr
    [('name', '<U16'), ('grades', '<f8', (2,))]

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('fields',
    """
    Dictionary of named fields defined for this data type, or ``None``.

    The dictionary is indexed by keys that are the names of the fields.
    Each entry in the dictionary is a tuple fully describing the field::

      (dtype, offset[, title])

    Offset is limited to C int, which is signed and usually 32 bits.
    If present, the optional title can be any object (if it is a string
    or unicode then it will also be a key in the fields dictionary,
    otherwise it's meta-data). Notice also that the first two elements
    of the tuple can be passed directly as arguments to the
    ``ndarray.getfield`` and ``ndarray.setfield`` methods.

    See Also
    --------
    ndarray.getfield, ndarray.setfield

    Examples
    --------

    >>> import numpy as np
    >>> dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])
    >>> print(dt.fields)
    {'name': (dtype('|S16'), 0), 'grades': (dtype(('float64',(2,))), 16)}

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('flags',
    """
    Bit-flags describing how this data type is to be interpreted.

    Bit-masks are in ``numpy._core.multiarray`` as the constants
    `ITEM_HASOBJECT`, `LIST_PICKLE`, `ITEM_IS_POINTER`, `NEEDS_INIT`,
    `NEEDS_PYAPI`, `USE_GETITEM`, `USE_SETITEM`. A full explanation
    of these flags is in C-API documentation; they are largely useful
    for user-defined data-types.

    The following example demonstrates that operations on this particular
    dtype requires Python C-API.

    Examples
    --------

    >>> import numpy as np
    >>> x = np.dtype([('a', np.int32, 8), ('b', np.float64, 6)])
    >>> x.flags
    16
    >>> np._core.multiarray.NEEDS_PYAPI
    16

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('hasobject',
    """
    Boolean indicating whether this dtype contains any reference-counted
    objects in any fields or sub-dtypes.

    Recall that what is actually in the ndarray memory representing
    the Python object is the memory address of that object (a pointer).
    Special handling may be required, and this attribute is useful for
    distinguishing data types that may contain arbitrary Python objects
    and data-types that won't.

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('isbuiltin',
    """
    Integer indicating how this dtype relates to the built-in dtypes.

    Read-only.

    =  ========================================================================
    0  if this is a structured array type, with fields
    1  if this is a dtype compiled into numpy (such as ints, floats etc)
    2  if the dtype is for a user-defined numpy type
       A user-defined type uses the numpy C-API machinery to extend
       numpy to handle a new array type. See
       :ref:`user.user-defined-data-types` in the NumPy manual.
    =  ========================================================================

    Examples
    --------

    >>> import numpy as np
    >>> dt = np.dtype('i2')
    >>> dt.isbuiltin
    1
    >>> dt = np.dtype('f8')
    >>> dt.isbuiltin
    1
    >>> dt = np.dtype([('field1', 'f8')])
    >>> dt.isbuiltin
    0

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('isnative',
    """
    Boolean indicating whether the byte order of this dtype is native
    to the platform.

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('isalignedstruct',
    """
    Boolean indicating whether the dtype is a struct which maintains
    field alignment. This flag is sticky, so when combining multiple
    structs together, it is preserved and produces new dtypes which
    are also aligned.

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('itemsize',
    """
    The element size of this data-type object.

    For 18 of the 21 types this number is fixed by the data-type.
    For the flexible data-types, this number can be anything.

    Examples
    --------

    >>> import numpy as np
    >>> arr = np.array([[1, 2], [3, 4]])
    >>> arr.dtype
    dtype('int64')
    >>> arr.itemsize
    8

    >>> dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])
    >>> dt.itemsize
    80

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('kind',
    """
    A character code (one of 'biufcmMOSTUV') identifying the general kind of data.

    =  ======================
    b  boolean
    i  signed integer
    u  unsigned integer
    f  floating-point
    c  complex floating-point
    m  timedelta
    M  datetime
    O  object
    S  (byte-)string
    T  string (StringDType)
    U  Unicode
    V  void
    =  ======================

    Examples
    --------

    >>> import numpy as np
    >>> dt = np.dtype('i4')
    >>> dt.kind
    'i'
    >>> dt = np.dtype('f8')
    >>> dt.kind
    'f'
    >>> dt = np.dtype([('field1', 'f8')])
    >>> dt.kind
    'V'

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('metadata',
    """
    Either ``None`` or a readonly dictionary of metadata (mappingproxy).

    The metadata field can be set using any dictionary at data-type
    creation. NumPy currently has no uniform approach to propagating
    metadata; although some array operations preserve it, there is no
    guarantee that others will.

    .. warning::

        Although used in certain projects, this feature was long undocumented
        and is not well supported. Some aspects of metadata propagation
        are expected to change in the future.

    Examples
    --------

    >>> import numpy as np
    >>> dt = np.dtype(float, metadata={"key": "value"})
    >>> dt.metadata["key"]
    'value'
    >>> arr = np.array([1, 2, 3], dtype=dt)
    >>> arr.dtype.metadata
    mappingproxy({'key': 'value'})

    Adding arrays with identical datatypes currently preserves the metadata:

    >>> (arr + arr).dtype.metadata
    mappingproxy({'key': 'value'})

    If the arrays have different dtype metadata, the first one wins:

    >>> dt2 = np.dtype(float, metadata={"key2": "value2"})
    >>> arr2 = np.array([3, 2, 1], dtype=dt2)
    >>> print((arr + arr2).dtype.metadata)
    {'key': 'value'}
    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('name',
    """
    A bit-width name for this data-type.

    Un-sized flexible data-type objects do not have this attribute.

    Examples
    --------

    >>> import numpy as np
    >>> x = np.dtype(float)
    >>> x.name
    'float64'
    >>> x = np.dtype([('a', np.int32, 8), ('b', np.float64, 6)])
    >>> x.name
    'void640'

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('names',
    """
    Ordered list of field names, or ``None`` if there are no fields.

    The names are ordered according to increasing byte offset. This can be
    used, for example, to walk through all of the named fields in offset order.

    Examples
    --------
    >>> dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])
    >>> dt.names
    ('name', 'grades')

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('num',
    """
    A unique number for each of the 21 different built-in types.

    These are roughly ordered from least-to-most precision.

    Examples
    --------

    >>> import numpy as np
    >>> dt = np.dtype(str)
    >>> dt.num
    19

    >>> dt = np.dtype(float)
    >>> dt.num
    12

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('shape',
    """
    Shape tuple of the sub-array if this data type describes a sub-array,
    and ``()`` otherwise.

    Examples
    --------

    >>> import numpy as np
    >>> dt = np.dtype(('i4', 4))
    >>> dt.shape
    (4,)

    >>> dt = np.dtype(('i4', (2, 3)))
    >>> dt.shape
    (2, 3)

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('ndim',
    """
    Number of dimensions of the sub-array if this data type describes a
    sub-array, and ``0`` otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.dtype(float)
    >>> x.ndim
    0

    >>> x = np.dtype((float, 8))
    >>> x.ndim
    1

    >>> x = np.dtype(('i4', (3, 4)))
    >>> x.ndim
    2

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('str',
    """The array-protocol typestring of this data-type object."""))

add_newdoc('numpy._core.multiarray', 'dtype', ('subdtype',
    """
    Tuple ``(item_dtype, shape)`` if this `dtype` describes a sub-array, and
    None otherwise.

    The *shape* is the fixed shape of the sub-array described by this
    data type, and *item_dtype* the data type of the array.

    If a field whose dtype object has this attribute is retrieved,
    then the extra dimensions implied by *shape* are tacked on to
    the end of the retrieved array.

    See Also
    --------
    dtype.base

    Examples
    --------
    >>> import numpy as np
    >>> x = np.dtype('8f')
    >>> x.subdtype
    (dtype('float32'), (8,))

    >>> x =  np.dtype('i2')
    >>> x.subdtype
    >>>

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('base',
    """
    Returns dtype for the base element of the subarrays,
    regardless of their dimension or shape.

    See Also
    --------
    dtype.subdtype

    Examples
    --------
    >>> import numpy as np
    >>> x = np.dtype('8f')
    >>> x.base
    dtype('float32')

    >>> x =  np.dtype('i2')
    >>> x.base
    dtype('int16')

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('type',
    """The type object used to instantiate a scalar of this data-type."""))

##############################################################################
#
# dtype methods
#
##############################################################################

add_newdoc('numpy._core.multiarray', 'dtype', ('newbyteorder',
    """
    newbyteorder(new_order='S', /)

    Return a new dtype with a different byte order.

    Changes are also made in all fields and sub-arrays of the data type.

    Parameters
    ----------
    new_order : string, optional
        Byte order to force; a value from the byte order specifications
        below.  The default value ('S') results in swapping the current
        byte order.  `new_order` codes can be any of:

        * 'S' - swap dtype from current to opposite endian
        * {'<', 'little'} - little endian
        * {'>', 'big'} - big endian
        * {'=', 'native'} - native order
        * {'|', 'I'} - ignore (no change to byte order)

    Returns
    -------
    new_dtype : dtype
        New dtype object with the given change to the byte order.

    Notes
    -----
    Changes are also made in all fields and sub-arrays of the data type.

    Examples
    --------
    >>> import sys
    >>> sys_is_le = sys.byteorder == 'little'
    >>> native_code = '<' if sys_is_le else '>'
    >>> swapped_code = '>' if sys_is_le else '<'
    >>> import numpy as np
    >>> native_dt = np.dtype(native_code+'i2')
    >>> swapped_dt = np.dtype(swapped_code+'i2')
    >>> native_dt.newbyteorder('S') == swapped_dt
    True
    >>> native_dt.newbyteorder() == swapped_dt
    True
    >>> native_dt == swapped_dt.newbyteorder('S')
    True
    >>> native_dt == swapped_dt.newbyteorder('=')
    True
    >>> native_dt == swapped_dt.newbyteorder('N')
    True
    >>> native_dt == native_dt.newbyteorder('|')
    True
    >>> np.dtype('<i2') == native_dt.newbyteorder('<')
    True
    >>> np.dtype('<i2') == native_dt.newbyteorder('L')
    True
    >>> np.dtype('>i2') == native_dt.newbyteorder('>')
    True
    >>> np.dtype('>i2') == native_dt.newbyteorder('B')
    True

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('__class_getitem__',
    """
    __class_getitem__(item, /)

    Return a parametrized wrapper around the `~numpy.dtype` type.

    .. versionadded:: 1.22

    Returns
    -------
    alias : types.GenericAlias
        A parametrized `~numpy.dtype` type.

    Examples
    --------
    >>> import numpy as np

    >>> np.dtype[np.int64]
    numpy.dtype[numpy.int64]

    See Also
    --------
    :pep:`585` : Type hinting generics in standard collections.

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('__ge__',
    """
    __ge__(value, /)

    Return ``self >= value``.

    Equivalent to ``np.can_cast(value, self, casting="safe")``.

    See Also
    --------
    can_cast : Returns True if cast between data types can occur according to
               the casting rule.

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('__le__',
    """
    __le__(value, /)

    Return ``self <= value``.

    Equivalent to ``np.can_cast(self, value, casting="safe")``.

    See Also
    --------
    can_cast : Returns True if cast between data types can occur according to
               the casting rule.

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('__gt__',
    """
    __ge__(value, /)

    Return ``self > value``.

    Equivalent to
    ``self != value and np.can_cast(value, self, casting="safe")``.

    See Also
    --------
    can_cast : Returns True if cast between data types can occur according to
               the casting rule.

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('__lt__',
    """
    __lt__(value, /)

    Return ``self < value``.

    Equivalent to
    ``self != value and np.can_cast(self, value, casting="safe")``.

    See Also
    --------
    can_cast : Returns True if cast between data types can occur according to
               the casting rule.

    """))

##############################################################################
#
# Datetime-related Methods
#
##############################################################################

add_newdoc('numpy._core.multiarray', 'busdaycalendar',
    """
    busdaycalendar(weekmask='1111100', holidays=None)

    A business day calendar object that efficiently stores information
    defining valid days for the busday family of functions.

    The default valid days are Monday through Friday ("business days").
    A busdaycalendar object can be specified with any set of weekly
    valid days, plus an optional "holiday" dates that always will be invalid.

    Once a busdaycalendar object is created, the weekmask and holidays
    cannot be modified.

    Parameters
    ----------
    weekmask : str or array_like of bool, optional
        A seven-element array indicating which of Monday through Sunday are
        valid days. May be specified as a length-seven list or array, like
        [1,1,1,1,1,0,0]; a length-seven string, like '1111100'; or a string
        like "Mon Tue Wed Thu Fri", made up of 3-character abbreviations for
        weekdays, optionally separated by white space. Valid abbreviations
        are: Mon Tue Wed Thu Fri Sat Sun
    holidays : array_like of datetime64[D], optional
        An array of dates to consider as invalid dates, no matter which
        weekday they fall upon.  Holiday dates may be specified in any
        order, and NaT (not-a-time) dates are ignored.  This list is
        saved in a normalized form that is suited for fast calculations
        of valid days.

    Returns
    -------
    out : busdaycalendar
        A business day calendar object containing the specified
        weekmask and holidays values.

    See Also
    --------
    is_busday : Returns a boolean array indicating valid days.
    busday_offset : Applies an offset counted in valid days.
    busday_count : Counts how many valid days are in a half-open date range.

    Attributes
    ----------
    weekmask : (copy) seven-element array of bool
    holidays : (copy) sorted array of datetime64[D]

    Notes
    -----
    Once a busdaycalendar object is created, you cannot modify the
    weekmask or holidays.  The attributes return copies of internal data.

    Examples
    --------
    >>> import numpy as np
    >>> # Some important days in July
    ... bdd = np.busdaycalendar(
    ...             holidays=['2011-07-01', '2011-07-04', '2011-07-17'])
    >>> # Default is Monday to Friday weekdays
    ... bdd.weekmask
    array([ True,  True,  True,  True,  True, False, False])
    >>> # Any holidays already on the weekend are removed
    ... bdd.holidays
    array(['2011-07-01', '2011-07-04'], dtype='datetime64[D]')
    """)

add_newdoc('numpy._core.multiarray', 'busdaycalendar', ('weekmask',
    """A copy of the seven-element boolean mask indicating valid days."""))

add_newdoc('numpy._core.multiarray', 'busdaycalendar', ('holidays',
    """A copy of the holiday array indicating additional invalid days."""))

add_newdoc('numpy._core.multiarray', 'normalize_axis_index',
    """
    normalize_axis_index(axis, ndim, msg_prefix=None)

    Normalizes an axis index, `axis`, such that is a valid positive index into
    the shape of array with `ndim` dimensions. Raises an AxisError with an
    appropriate message if this is not possible.

    Used internally by all axis-checking logic.

    Parameters
    ----------
    axis : int
        The un-normalized index of the axis. Can be negative
    ndim : int
        The number of dimensions of the array that `axis` should be normalized
        against
    msg_prefix : str
        A prefix to put before the message, typically the name of the argument

    Returns
    -------
    normalized_axis : int
        The normalized axis index, such that `0 <= normalized_axis < ndim`

    Raises
    ------
    AxisError
        If the axis index is invalid, when `-ndim <= axis < ndim` is false.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.lib.array_utils import normalize_axis_index
    >>> normalize_axis_index(0, ndim=3)
    0
    >>> normalize_axis_index(1, ndim=3)
    1
    >>> normalize_axis_index(-1, ndim=3)
    2

    >>> normalize_axis_index(3, ndim=3)
    Traceback (most recent call last):
    ...
    numpy.exceptions.AxisError: axis 3 is out of bounds for array ...
    >>> normalize_axis_index(-4, ndim=3, msg_prefix='axes_arg')
    Traceback (most recent call last):
    ...
    numpy.exceptions.AxisError: axes_arg: axis -4 is out of bounds ...
    """)

add_newdoc('numpy._core.multiarray', 'datetime_data',
    """
    datetime_data(dtype, /)

    Get information about the step size of a date or time type.

    The returned tuple can be passed as the second argument of `numpy.datetime64` and
    `numpy.timedelta64`.

    Parameters
    ----------
    dtype : dtype
        The dtype object, which must be a `datetime64` or `timedelta64` type.

    Returns
    -------
    unit : str
        The :ref:`datetime unit <arrays.dtypes.dateunits>` on which this dtype
        is based.
    count : int
        The number of base units in a step.

    Examples
    --------
    >>> import numpy as np
    >>> dt_25s = np.dtype('timedelta64[25s]')
    >>> np.datetime_data(dt_25s)
    ('s', 25)
    >>> np.array(10, dt_25s).astype('timedelta64[s]')
    array(250, dtype='timedelta64[s]')

    The result can be used to construct a datetime that uses the same units
    as a timedelta

    >>> np.datetime64('2010', np.datetime_data(dt_25s))
    np.datetime64('2010-01-01T00:00:00','25s')
    """)


##############################################################################
#
# Documentation for `generic` attributes and methods
#
##############################################################################

add_newdoc('numpy._core.numerictypes', 'generic',
    """
    Base class for numpy scalar types.

    Class from which most (all?) numpy scalar types are derived.  For
    consistency, exposes the same API as `ndarray`, despite many
    consequent attributes being either "get-only," or completely irrelevant.
    This is the class from which it is strongly suggested users should derive
    custom scalar types.

    """)

# Attributes

def refer_to_array_attribute(attr, method=True):
    docstring = """
    Scalar {} identical to the corresponding array attribute.

    Please see `ndarray.{}`.
    """

    return attr, docstring.format("method" if method else "attribute", attr)


add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('T', method=False))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('base', method=False))

add_newdoc('numpy._core.numerictypes', 'generic', ('data',
    """Pointer to start of data."""))

add_newdoc('numpy._core.numerictypes', 'generic', ('dtype',
    """Get array data-descriptor."""))

add_newdoc('numpy._core.numerictypes', 'generic', ('flags',
    """The integer value of flags."""))

add_newdoc('numpy._core.numerictypes', 'generic', ('flat',
    """A 1-D view of the scalar."""))

add_newdoc('numpy._core.numerictypes', 'generic', ('imag',
    """The imaginary part of the scalar."""))

add_newdoc('numpy._core.numerictypes', 'generic', ('itemsize',
    """The length of one element in bytes."""))

add_newdoc('numpy._core.numerictypes', 'generic', ('ndim',
    """The number of array dimensions."""))

add_newdoc('numpy._core.numerictypes', 'generic', ('real',
    """The real part of the scalar."""))

add_newdoc('numpy._core.numerictypes', 'generic', ('shape',
    """Tuple of array dimensions."""))

add_newdoc('numpy._core.numerictypes', 'generic', ('size',
    """The number of elements in the gentype."""))

add_newdoc('numpy._core.numerictypes', 'generic', ('strides',
    """Tuple of bytes steps in each dimension."""))

# Methods

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('all'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('any'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('argmax'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('argmin'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('argsort'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('astype'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('byteswap'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('choose'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('clip'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('compress'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('conjugate'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('copy'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('cumprod'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('cumsum'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('diagonal'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('dump'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('dumps'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('fill'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('flatten'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('getfield'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('item'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('max'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('mean'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('min'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('nonzero'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('prod'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('put'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('ravel'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('repeat'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('reshape'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('resize'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('round'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('searchsorted'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('setfield'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('setflags'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('sort'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('squeeze'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('std'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('sum'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('swapaxes'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('take'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('tofile'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('tolist'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('tostring'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('trace'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('transpose'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('var'))

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('view'))

add_newdoc('numpy._core.numerictypes', 'number', ('__class_getitem__',
    """
    __class_getitem__(item, /)

    Return a parametrized wrapper around the `~numpy.number` type.

    .. versionadded:: 1.22

    Returns
    -------
    alias : types.GenericAlias
        A parametrized `~numpy.number` type.

    Examples
    --------
    >>> from typing import Any
    >>> import numpy as np

    >>> np.signedinteger[Any]
    numpy.signedinteger[typing.Any]

    See Also
    --------
    :pep:`585` : Type hinting generics in standard collections.

    """))

##############################################################################
#
# Documentation for scalar type abstract base classes in type hierarchy
#
##############################################################################


add_newdoc('numpy._core.numerictypes', 'number',
    """
    Abstract base class of all numeric scalar types.

    """)

add_newdoc('numpy._core.numerictypes', 'integer',
    """
    Abstract base class of all integer scalar types.

    """)

add_newdoc('numpy._core.numerictypes', 'signedinteger',
    """
    Abstract base class of all signed integer scalar types.

    """)

add_newdoc('numpy._core.numerictypes', 'unsignedinteger',
    """
    Abstract base class of all unsigned integer scalar types.

    """)

add_newdoc('numpy._core.numerictypes', 'inexact',
    """
    Abstract base class of all numeric scalar types with a (potentially)
    inexact representation of the values in its range, such as
    floating-point numbers.

    """)

add_newdoc('numpy._core.numerictypes', 'floating',
    """
    Abstract base class of all floating-point scalar types.

    """)

add_newdoc('numpy._core.numerictypes', 'complexfloating',
    """
    Abstract base class of all complex number scalar types that are made up of
    floating-point numbers.

    """)

add_newdoc('numpy._core.numerictypes', 'flexible',
    """
    Abstract base class of all scalar types without predefined length.
    The actual size of these types depends on the specific `numpy.dtype`
    instantiation.

    """)

add_newdoc('numpy._core.numerictypes', 'character',
    """
    Abstract base class of all character string scalar types.

    """)

add_newdoc('numpy._core.multiarray', 'StringDType',
    """
    StringDType(*, na_object=np._NoValue, coerce=True)

    Create a StringDType instance.

    StringDType can be used to store UTF-8 encoded variable-width strings in
    a NumPy array.

    Parameters
    ----------
    na_object : object, optional
        Object used to represent missing data. If unset, the array will not
        use a missing data sentinel.
    coerce : bool, optional
        Whether or not items in an array-like passed to an array creation
        function that are neither a str or str subtype should be coerced to
        str. Defaults to True. If set to False, creating a StringDType
        array from an array-like containing entries that are not already
        strings will raise an error.

    Examples
    --------

    >>> import numpy as np

    >>> from numpy.dtypes import StringDType
    >>> np.array(["hello", "world"], dtype=StringDType())
    array(["hello", "world"], dtype=StringDType())

    >>> arr = np.array(["hello", None, "world"],
    ...                dtype=StringDType(na_object=None))
    >>> arr
    array(["hello", None, "world"], dtype=StringDType(na_object=None))
    >>> arr[1] is None
    True

    >>> arr = np.array(["hello", np.nan, "world"],
    ...                dtype=StringDType(na_object=np.nan))
    >>> np.isnan(arr)
    array([False, True, False])

    >>> np.array([1.2, object(), "hello world"],
    ...          dtype=StringDType(coerce=False))
    Traceback (most recent call last):
        ...
    ValueError: StringDType only allows string data when string coercion is disabled.

    >>> np.array(["hello", "world"], dtype=StringDType(coerce=True))
    array(["hello", "world"], dtype=StringDType(coerce=True))
    """)