
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\predicates\matrices.py ===
from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher

class SquarePredicate(Predicate):
    """
    Square matrix predicate.

    Explanation
    ===========

    ``Q.square(x)`` is true iff ``x`` is a square matrix. A square matrix
    is a matrix with the same number of rows and columns.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, ZeroMatrix, Identity
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('X', 2, 3)
    >>> ask(Q.square(X))
    True
    >>> ask(Q.square(Y))
    False
    >>> ask(Q.square(ZeroMatrix(3, 3)))
    True
    >>> ask(Q.square(Identity(3)))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Square_matrix

    """
    name = 'square'
    handler = Dispatcher("SquareHandler", doc="Handler for Q.square.")


class SymmetricPredicate(Predicate):
    """
    Symmetric matrix predicate.

    Explanation
    ===========

    ``Q.symmetric(x)`` is true iff ``x`` is a square matrix and is equal to
    its transpose. Every square diagonal matrix is a symmetric matrix.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('Y', 2, 3)
    >>> Z = MatrixSymbol('Z', 2, 2)
    >>> ask(Q.symmetric(X*Z), Q.symmetric(X) & Q.symmetric(Z))
    True
    >>> ask(Q.symmetric(X + Z), Q.symmetric(X) & Q.symmetric(Z))
    True
    >>> ask(Q.symmetric(Y))
    False


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Symmetric_matrix

    """
    # TODO: Add handlers to make these keys work with
    # actual matrices and add more examples in the docstring.
    name = 'symmetric'
    handler = Dispatcher("SymmetricHandler", doc="Handler for Q.symmetric.")


class InvertiblePredicate(Predicate):
    """
    Invertible matrix predicate.

    Explanation
    ===========

    ``Q.invertible(x)`` is true iff ``x`` is an invertible matrix.
    A square matrix is called invertible only if its determinant is 0.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('Y', 2, 3)
    >>> Z = MatrixSymbol('Z', 2, 2)
    >>> ask(Q.invertible(X*Y), Q.invertible(X))
    False
    >>> ask(Q.invertible(X*Z), Q.invertible(X) & Q.invertible(Z))
    True
    >>> ask(Q.invertible(X), Q.fullrank(X) & Q.square(X))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Invertible_matrix

    """
    name = 'invertible'
    handler = Dispatcher("InvertibleHandler", doc="Handler for Q.invertible.")


class OrthogonalPredicate(Predicate):
    """
    Orthogonal matrix predicate.

    Explanation
    ===========

    ``Q.orthogonal(x)`` is true iff ``x`` is an orthogonal matrix.
    A square matrix ``M`` is an orthogonal matrix if it satisfies
    ``M^TM = MM^T = I`` where ``M^T`` is the transpose matrix of
    ``M`` and ``I`` is an identity matrix. Note that an orthogonal
    matrix is necessarily invertible.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, Identity
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('Y', 2, 3)
    >>> Z = MatrixSymbol('Z', 2, 2)
    >>> ask(Q.orthogonal(Y))
    False
    >>> ask(Q.orthogonal(X*Z*X), Q.orthogonal(X) & Q.orthogonal(Z))
    True
    >>> ask(Q.orthogonal(Identity(3)))
    True
    >>> ask(Q.invertible(X), Q.orthogonal(X))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Orthogonal_matrix

    """
    name = 'orthogonal'
    handler = Dispatcher("OrthogonalHandler", doc="Handler for key 'orthogonal'.")


class UnitaryPredicate(Predicate):
    """
    Unitary matrix predicate.

    Explanation
    ===========

    ``Q.unitary(x)`` is true iff ``x`` is a unitary matrix.
    Unitary matrix is an analogue to orthogonal matrix. A square
    matrix ``M`` with complex elements is unitary if :math:``M^TM = MM^T= I``
    where :math:``M^T`` is the conjugate transpose matrix of ``M``.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, Identity
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('Y', 2, 3)
    >>> Z = MatrixSymbol('Z', 2, 2)
    >>> ask(Q.unitary(Y))
    False
    >>> ask(Q.unitary(X*Z*X), Q.unitary(X) & Q.unitary(Z))
    True
    >>> ask(Q.unitary(Identity(3)))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Unitary_matrix

    """
    name = 'unitary'
    handler = Dispatcher("UnitaryHandler", doc="Handler for key 'unitary'.")


class FullRankPredicate(Predicate):
    """
    Fullrank matrix predicate.

    Explanation
    ===========

    ``Q.fullrank(x)`` is true iff ``x`` is a full rank matrix.
    A matrix is full rank if all rows and columns of the matrix
    are linearly independent. A square matrix is full rank iff
    its determinant is nonzero.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, ZeroMatrix, Identity
    >>> X = MatrixSymbol('X', 2, 2)
    >>> ask(Q.fullrank(X.T), Q.fullrank(X))
    True
    >>> ask(Q.fullrank(ZeroMatrix(3, 3)))
    False
    >>> ask(Q.fullrank(Identity(3)))
    True

    """
    name = 'fullrank'
    handler = Dispatcher("FullRankHandler", doc="Handler for key 'fullrank'.")


class PositiveDefinitePredicate(Predicate):
    r"""
    Positive definite matrix predicate.

    Explanation
    ===========

    If $M$ is a :math:`n \times n` symmetric real matrix, it is said
    to be positive definite if :math:`Z^TMZ` is positive for
    every non-zero column vector $Z$ of $n$ real numbers.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, Identity
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('Y', 2, 3)
    >>> Z = MatrixSymbol('Z', 2, 2)
    >>> ask(Q.positive_definite(Y))
    False
    >>> ask(Q.positive_definite(Identity(3)))
    True
    >>> ask(Q.positive_definite(X + Z), Q.positive_definite(X) &
    ...     Q.positive_definite(Z))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Positive-definite_matrix

    """
    name = "positive_definite"
    handler = Dispatcher("PositiveDefiniteHandler", doc="Handler for key 'positive_definite'.")


class UpperTriangularPredicate(Predicate):
    """
    Upper triangular matrix predicate.

    Explanation
    ===========

    A matrix $M$ is called upper triangular matrix if :math:`M_{ij}=0`
    for :math:`i<j`.

    Examples
    ========

    >>> from sympy import Q, ask, ZeroMatrix, Identity
    >>> ask(Q.upper_triangular(Identity(3)))
    True
    >>> ask(Q.upper_triangular(ZeroMatrix(3, 3)))
    True

    References
    ==========

    .. [1] https://mathworld.wolfram.com/UpperTriangularMatrix.html

    """
    name = "upper_triangular"
    handler = Dispatcher("UpperTriangularHandler", doc="Handler for key 'upper_triangular'.")


class LowerTriangularPredicate(Predicate):
    """
    Lower triangular matrix predicate.

    Explanation
    ===========

    A matrix $M$ is called lower triangular matrix if :math:`M_{ij}=0`
    for :math:`i>j`.

    Examples
    ========

    >>> from sympy import Q, ask, ZeroMatrix, Identity
    >>> ask(Q.lower_triangular(Identity(3)))
    True
    >>> ask(Q.lower_triangular(ZeroMatrix(3, 3)))
    True

    References
    ==========

    .. [1] https://mathworld.wolfram.com/LowerTriangularMatrix.html

    """
    name = "lower_triangular"
    handler = Dispatcher("LowerTriangularHandler", doc="Handler for key 'lower_triangular'.")


class DiagonalPredicate(Predicate):
    """
    Diagonal matrix predicate.

    Explanation
    ===========

    ``Q.diagonal(x)`` is true iff ``x`` is a diagonal matrix. A diagonal
    matrix is a matrix in which the entries outside the main diagonal
    are all zero.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, ZeroMatrix
    >>> X = MatrixSymbol('X', 2, 2)
    >>> ask(Q.diagonal(ZeroMatrix(3, 3)))
    True
    >>> ask(Q.diagonal(X), Q.lower_triangular(X) &
    ...     Q.upper_triangular(X))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Diagonal_matrix

    """
    name = "diagonal"
    handler = Dispatcher("DiagonalHandler", doc="Handler for key 'diagonal'.")


class IntegerElementsPredicate(Predicate):
    """
    Integer elements matrix predicate.

    Explanation
    ===========

    ``Q.integer_elements(x)`` is true iff all the elements of ``x``
    are integers.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.integer(X[1, 2]), Q.integer_elements(X))
    True

    """
    name = "integer_elements"
    handler = Dispatcher("IntegerElementsHandler", doc="Handler for key 'integer_elements'.")


class RealElementsPredicate(Predicate):
    """
    Real elements matrix predicate.

    Explanation
    ===========

    ``Q.real_elements(x)`` is true iff all the elements of ``x``
    are real numbers.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.real(X[1, 2]), Q.real_elements(X))
    True

    """
    name = "real_elements"
    handler = Dispatcher("RealElementsHandler", doc="Handler for key 'real_elements'.")


class ComplexElementsPredicate(Predicate):
    """
    Complex elements matrix predicate.

    Explanation
    ===========

    ``Q.complex_elements(x)`` is true iff all the elements of ``x``
    are complex numbers.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.complex(X[1, 2]), Q.complex_elements(X))
    True
    >>> ask(Q.complex_elements(X), Q.integer_elements(X))
    True

    """
    name = "complex_elements"
    handler = Dispatcher("ComplexElementsHandler", doc="Handler for key 'complex_elements'.")


class SingularPredicate(Predicate):
    """
    Singular matrix predicate.

    A matrix is singular iff the value of its determinant is 0.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.singular(X), Q.invertible(X))
    False
    >>> ask(Q.singular(X), ~Q.invertible(X))
    True

    References
    ==========

    .. [1] https://mathworld.wolfram.com/SingularMatrix.html

    """
    name = "singular"
    handler = Dispatcher("SingularHandler", doc="Predicate fore key 'singular'.")


class NormalPredicate(Predicate):
    """
    Normal matrix predicate.

    A matrix is normal if it commutes with its conjugate transpose.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.normal(X), Q.unitary(X))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Normal_matrix

    """
    name = "normal"
    handler = Dispatcher("NormalHandler", doc="Predicate fore key 'normal'.")


class TriangularPredicate(Predicate):
    """
    Triangular matrix predicate.

    Explanation
    ===========

    ``Q.triangular(X)`` is true if ``X`` is one that is either lower
    triangular or upper triangular.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.triangular(X), Q.upper_triangular(X))
    True
    >>> ask(Q.triangular(X), Q.lower_triangular(X))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Triangular_matrix

    """
    name = "triangular"
    handler = Dispatcher("TriangularHandler", doc="Predicate fore key 'triangular'.")


class UnitTriangularPredicate(Predicate):
    """
    Unit triangular matrix predicate.

    Explanation
    ===========

    A unit triangular matrix is a triangular matrix with 1s
    on the diagonal.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.triangular(X), Q.unit_triangular(X))
    True

    """
    name = "unit_triangular"
    handler = Dispatcher("UnitTriangularHandler", doc="Predicate fore key 'unit_triangular'.")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\postgresql\array.py ===
# dialects/postgresql/array.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php


from __future__ import annotations

import re
from typing import Any as typing_Any
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from .operators import CONTAINED_BY
from .operators import CONTAINS
from .operators import OVERLAP
from ... import types as sqltypes
from ... import util
from ...sql import expression
from ...sql import operators
from ...sql.visitors import InternalTraversal

if TYPE_CHECKING:
    from ...engine.interfaces import Dialect
    from ...sql._typing import _ColumnExpressionArgument
    from ...sql._typing import _TypeEngineArgument
    from ...sql.elements import ColumnElement
    from ...sql.elements import Grouping
    from ...sql.expression import BindParameter
    from ...sql.operators import OperatorType
    from ...sql.selectable import _SelectIterable
    from ...sql.type_api import _BindProcessorType
    from ...sql.type_api import _LiteralProcessorType
    from ...sql.type_api import _ResultProcessorType
    from ...sql.type_api import TypeEngine
    from ...sql.visitors import _TraverseInternalsType
    from ...util.typing import Self


_T = TypeVar("_T", bound=typing_Any)


def Any(
    other: typing_Any,
    arrexpr: _ColumnExpressionArgument[_T],
    operator: OperatorType = operators.eq,
) -> ColumnElement[bool]:
    """A synonym for the ARRAY-level :meth:`.ARRAY.Comparator.any` method.
    See that method for details.

    """

    return arrexpr.any(other, operator)  # type: ignore[no-any-return, union-attr]  # noqa: E501


def All(
    other: typing_Any,
    arrexpr: _ColumnExpressionArgument[_T],
    operator: OperatorType = operators.eq,
) -> ColumnElement[bool]:
    """A synonym for the ARRAY-level :meth:`.ARRAY.Comparator.all` method.
    See that method for details.

    """

    return arrexpr.all(other, operator)  # type: ignore[no-any-return, union-attr]  # noqa: E501


class array(expression.ExpressionClauseList[_T]):
    """A PostgreSQL ARRAY literal.

    This is used to produce ARRAY literals in SQL expressions, e.g.::

        from sqlalchemy.dialects.postgresql import array
        from sqlalchemy.dialects import postgresql
        from sqlalchemy import select, func

        stmt = select(array([1, 2]) + array([3, 4, 5]))

        print(stmt.compile(dialect=postgresql.dialect()))

    Produces the SQL:

    .. sourcecode:: sql

        SELECT ARRAY[%(param_1)s, %(param_2)s] ||
            ARRAY[%(param_3)s, %(param_4)s, %(param_5)s]) AS anon_1

    An instance of :class:`.array` will always have the datatype
    :class:`_types.ARRAY`.  The "inner" type of the array is inferred from the
    values present, unless the :paramref:`_postgresql.array.type_` keyword
    argument is passed::

        array(["foo", "bar"], type_=CHAR)

    When constructing an empty array, the :paramref:`_postgresql.array.type_`
    argument is particularly important as PostgreSQL server typically requires
    a cast to be rendered for the inner type in order to render an empty array.
    SQLAlchemy's compilation for the empty array will produce this cast so
    that::

        stmt = array([], type_=Integer)
        print(stmt.compile(dialect=postgresql.dialect()))

    Produces:

    .. sourcecode:: sql

        ARRAY[]::INTEGER[]

    As required by PostgreSQL for empty arrays.

    .. versionadded:: 2.0.40 added support to render empty PostgreSQL array
       literals with a required cast.

    Multidimensional arrays are produced by nesting :class:`.array` constructs.
    The dimensionality of the final :class:`_types.ARRAY`
    type is calculated by
    recursively adding the dimensions of the inner :class:`_types.ARRAY`
    type::

        stmt = select(
            array(
                [array([1, 2]), array([3, 4]), array([column("q"), column("x")])]
            )
        )
        print(stmt.compile(dialect=postgresql.dialect()))

    Produces:

    .. sourcecode:: sql

        SELECT ARRAY[
            ARRAY[%(param_1)s, %(param_2)s],
            ARRAY[%(param_3)s, %(param_4)s],
            ARRAY[q, x]
        ] AS anon_1

    .. versionadded:: 1.3.6 added support for multidimensional array literals

    .. seealso::

        :class:`_postgresql.ARRAY`

    """  # noqa: E501

    __visit_name__ = "array"

    stringify_dialect = "postgresql"

    _traverse_internals: _TraverseInternalsType = [
        ("clauses", InternalTraversal.dp_clauseelement_tuple),
        ("type", InternalTraversal.dp_type),
    ]

    def __init__(
        self,
        clauses: Iterable[_T],
        *,
        type_: Optional[_TypeEngineArgument[_T]] = None,
        **kw: typing_Any,
    ):
        r"""Construct an ARRAY literal.

        :param clauses: iterable, such as a list, containing elements to be
         rendered in the array
        :param type\_: optional type.  If omitted, the type is inferred
         from the contents of the array.

        """
        super().__init__(operators.comma_op, *clauses, **kw)

        main_type = (
            type_
            if type_ is not None
            else self.clauses[0].type if self.clauses else sqltypes.NULLTYPE
        )

        if isinstance(main_type, ARRAY):
            self.type = ARRAY(
                main_type.item_type,
                dimensions=(
                    main_type.dimensions + 1
                    if main_type.dimensions is not None
                    else 2
                ),
            )  # type: ignore[assignment]
        else:
            self.type = ARRAY(main_type)  # type: ignore[assignment]

    @property
    def _select_iterable(self) -> _SelectIterable:
        return (self,)

    def _bind_param(
        self,
        operator: OperatorType,
        obj: typing_Any,
        type_: Optional[TypeEngine[_T]] = None,
        _assume_scalar: bool = False,
    ) -> BindParameter[_T]:
        if _assume_scalar or operator is operators.getitem:
            return expression.BindParameter(
                None,
                obj,
                _compared_to_operator=operator,
                type_=type_,
                _compared_to_type=self.type,
                unique=True,
            )

        else:
            return array(
                [
                    self._bind_param(
                        operator, o, _assume_scalar=True, type_=type_
                    )
                    for o in obj
                ]
            )  # type: ignore[return-value]

    def self_group(
        self, against: Optional[OperatorType] = None
    ) -> Union[Self, Grouping[_T]]:
        if against in (operators.any_op, operators.all_op, operators.getitem):
            return expression.Grouping(self)
        else:
            return self


class ARRAY(sqltypes.ARRAY[_T]):
    """PostgreSQL ARRAY type.

    The :class:`_postgresql.ARRAY` type is constructed in the same way
    as the core :class:`_types.ARRAY` type; a member type is required, and a
    number of dimensions is recommended if the type is to be used for more
    than one dimension::

        from sqlalchemy.dialects import postgresql

        mytable = Table(
            "mytable",
            metadata,
            Column("data", postgresql.ARRAY(Integer, dimensions=2)),
        )

    The :class:`_postgresql.ARRAY` type provides all operations defined on the
    core :class:`_types.ARRAY` type, including support for "dimensions",
    indexed access, and simple matching such as
    :meth:`.types.ARRAY.Comparator.any` and
    :meth:`.types.ARRAY.Comparator.all`.  :class:`_postgresql.ARRAY`
    class also
    provides PostgreSQL-specific methods for containment operations, including
    :meth:`.postgresql.ARRAY.Comparator.contains`
    :meth:`.postgresql.ARRAY.Comparator.contained_by`, and
    :meth:`.postgresql.ARRAY.Comparator.overlap`, e.g.::

        mytable.c.data.contains([1, 2])

    Indexed access is one-based by default, to match that of PostgreSQL;
    for zero-based indexed access, set
    :paramref:`_postgresql.ARRAY.zero_indexes`.

    Additionally, the :class:`_postgresql.ARRAY`
    type does not work directly in
    conjunction with the :class:`.ENUM` type.  For a workaround, see the
    special type at :ref:`postgresql_array_of_enum`.

    .. container:: topic

        **Detecting Changes in ARRAY columns when using the ORM**

        The :class:`_postgresql.ARRAY` type, when used with the SQLAlchemy ORM,
        does not detect in-place mutations to the array. In order to detect
        these, the :mod:`sqlalchemy.ext.mutable` extension must be used, using
        the :class:`.MutableList` class::

            from sqlalchemy.dialects.postgresql import ARRAY
            from sqlalchemy.ext.mutable import MutableList


            class SomeOrmClass(Base):
                # ...

                data = Column(MutableList.as_mutable(ARRAY(Integer)))

        This extension will allow "in-place" changes such to the array
        such as ``.append()`` to produce events which will be detected by the
        unit of work.  Note that changes to elements **inside** the array,
        including subarrays that are mutated in place, are **not** detected.

        Alternatively, assigning a new array value to an ORM element that
        replaces the old one will always trigger a change event.

    .. seealso::

        :class:`_types.ARRAY` - base array type

        :class:`_postgresql.array` - produces a literal array value.

    """

    def __init__(
        self,
        item_type: _TypeEngineArgument[_T],
        as_tuple: bool = False,
        dimensions: Optional[int] = None,
        zero_indexes: bool = False,
    ):
        """Construct an ARRAY.

        E.g.::

          Column("myarray", ARRAY(Integer))

        Arguments are:

        :param item_type: The data type of items of this array. Note that
          dimensionality is irrelevant here, so multi-dimensional arrays like
          ``INTEGER[][]``, are constructed as ``ARRAY(Integer)``, not as
          ``ARRAY(ARRAY(Integer))`` or such.

        :param as_tuple=False: Specify whether return results
          should be converted to tuples from lists. DBAPIs such
          as psycopg2 return lists by default. When tuples are
          returned, the results are hashable.

        :param dimensions: if non-None, the ARRAY will assume a fixed
         number of dimensions.  This will cause the DDL emitted for this
         ARRAY to include the exact number of bracket clauses ``[]``,
         and will also optimize the performance of the type overall.
         Note that PG arrays are always implicitly "non-dimensioned",
         meaning they can store any number of dimensions no matter how
         they were declared.

        :param zero_indexes=False: when True, index values will be converted
         between Python zero-based and PostgreSQL one-based indexes, e.g.
         a value of one will be added to all index values before passing
         to the database.

        """
        if isinstance(item_type, ARRAY):
            raise ValueError(
                "Do not nest ARRAY types; ARRAY(basetype) "
                "handles multi-dimensional arrays of basetype"
            )
        if isinstance(item_type, type):
            item_type = item_type()
        self.item_type = item_type
        self.as_tuple = as_tuple
        self.dimensions = dimensions
        self.zero_indexes = zero_indexes

    class Comparator(sqltypes.ARRAY.Comparator[_T]):
        """Define comparison operations for :class:`_types.ARRAY`.

        Note that these operations are in addition to those provided
        by the base :class:`.types.ARRAY.Comparator` class, including
        :meth:`.types.ARRAY.Comparator.any` and
        :meth:`.types.ARRAY.Comparator.all`.

        """

        def contains(
            self, other: typing_Any, **kwargs: typing_Any
        ) -> ColumnElement[bool]:
            """Boolean expression.  Test if elements are a superset of the
            elements of the argument array expression.

            kwargs may be ignored by this operator but are required for API
            conformance.
            """
            return self.operate(CONTAINS, other, result_type=sqltypes.Boolean)

        def contained_by(self, other: typing_Any) -> ColumnElement[bool]:
            """Boolean expression.  Test if elements are a proper subset of the
            elements of the argument array expression.
            """
            return self.operate(
                CONTAINED_BY, other, result_type=sqltypes.Boolean
            )

        def overlap(self, other: typing_Any) -> ColumnElement[bool]:
            """Boolean expression.  Test if array has elements in common with
            an argument array expression.
            """
            return self.operate(OVERLAP, other, result_type=sqltypes.Boolean)

    comparator_factory = Comparator

    @util.memoized_property
    def _against_native_enum(self) -> bool:
        return (
            isinstance(self.item_type, sqltypes.Enum)
            and self.item_type.native_enum  # type: ignore[attr-defined]
        )

    def literal_processor(
        self, dialect: Dialect
    ) -> Optional[_LiteralProcessorType[_T]]:
        item_proc = self.item_type.dialect_impl(dialect).literal_processor(
            dialect
        )
        if item_proc is None:
            return None

        def to_str(elements: Iterable[typing_Any]) -> str:
            return f"ARRAY[{', '.join(elements)}]"

        def process(value: Sequence[typing_Any]) -> str:
            inner = self._apply_item_processor(
                value, item_proc, self.dimensions, to_str
            )
            return inner

        return process

    def bind_processor(
        self, dialect: Dialect
    ) -> Optional[_BindProcessorType[Sequence[typing_Any]]]:
        item_proc = self.item_type.dialect_impl(dialect).bind_processor(
            dialect
        )

        def process(
            value: Optional[Sequence[typing_Any]],
        ) -> Optional[list[typing_Any]]:
            if value is None:
                return value
            else:
                return self._apply_item_processor(
                    value, item_proc, self.dimensions, list
                )

        return process

    def result_processor(
        self, dialect: Dialect, coltype: object
    ) -> _ResultProcessorType[Sequence[typing_Any]]:
        item_proc = self.item_type.dialect_impl(dialect).result_processor(
            dialect, coltype
        )

        def process(
            value: Sequence[typing_Any],
        ) -> Optional[Sequence[typing_Any]]:
            if value is None:
                return value
            else:
                return self._apply_item_processor(
                    value,
                    item_proc,
                    self.dimensions,
                    tuple if self.as_tuple else list,
                )

        if self._against_native_enum:
            super_rp = process
            pattern = re.compile(r"^{(.*)}$")

            def handle_raw_string(value: str) -> list[str]:
                inner = pattern.match(value).group(1)  # type: ignore[union-attr]  # noqa: E501
                return _split_enum_values(inner)

            def process(
                value: Sequence[typing_Any],
            ) -> Optional[Sequence[typing_Any]]:
                if value is None:
                    return value
                # isinstance(value, str) is required to handle
                # the case where a TypeDecorator for and Array of Enum is
                # used like was required in sa < 1.3.17
                return super_rp(
                    handle_raw_string(value)
                    if isinstance(value, str)
                    else value
                )

        return process


def _split_enum_values(array_string: str) -> list[str]:
    if '"' not in array_string:
        # no escape char is present so it can just split on the comma
        return array_string.split(",") if array_string else []

    # handles quoted strings from:
    # r'abc,"quoted","also\\\\quoted", "quoted, comma", "esc \" quot", qpr'
    # returns
    # ['abc', 'quoted', 'also\\quoted', 'quoted, comma', 'esc " quot', 'qpr']
    text = array_string.replace(r"\"", "_$ESC_QUOTE$_")
    text = text.replace(r"\\", "\\")
    result = []
    on_quotes = re.split(r'(")', text)
    in_quotes = False
    for tok in on_quotes:
        if tok == '"':
            in_quotes = not in_quotes
        elif in_quotes:
            result.append(tok.replace("_$ESC_QUOTE$_", '"'))
        else:
            result.extend(re.findall(r"([^\s,]+),?", tok))
    return result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\bivariate.py ===
from sympy.core.add import Add
from sympy.core.exprtools import factor_terms
from sympy.core.function import expand_log, _mexpand
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.miscellaneous import root
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly, factor
from sympy.simplify.simplify import separatevars
from sympy.simplify.radsimp import collect
from sympy.simplify.simplify import powsimp
from sympy.solvers.solvers import solve, _invert
from sympy.utilities.iterables import uniq


def _filtered_gens(poly, symbol):
    """process the generators of ``poly``, returning the set of generators that
    have ``symbol``.  If there are two generators that are inverses of each other,
    prefer the one that has no denominator.

    Examples
    ========

    >>> from sympy.solvers.bivariate import _filtered_gens
    >>> from sympy import Poly, exp
    >>> from sympy.abc import x
    >>> _filtered_gens(Poly(x + 1/x + exp(x)), x)
    {x, exp(x)}

    """
    # TODO it would be good to pick the smallest divisible power
    # instead of the base for something like x**4 + x**2 -->
    # return x**2 not x
    gens = {g for g in poly.gens if symbol in g.free_symbols}
    for g in list(gens):
        ag = 1/g
        if g in gens and ag in gens:
            if ag.as_numer_denom()[1] is not S.One:
                g = ag
            gens.remove(g)
    return gens


def _mostfunc(lhs, func, X=None):
    """Returns the term in lhs which contains the most of the
    func-type things e.g. log(log(x)) wins over log(x) if both terms appear.

    ``func`` can be a function (exp, log, etc...) or any other SymPy object,
    like Pow.

    If ``X`` is not ``None``, then the function returns the term composed with the
    most ``func`` having the specified variable.

    Examples
    ========

    >>> from sympy.solvers.bivariate import _mostfunc
    >>> from sympy import exp
    >>> from sympy.abc import x, y
    >>> _mostfunc(exp(x) + exp(exp(x) + 2), exp)
    exp(exp(x) + 2)
    >>> _mostfunc(exp(x) + exp(exp(y) + 2), exp)
    exp(exp(y) + 2)
    >>> _mostfunc(exp(x) + exp(exp(y) + 2), exp, x)
    exp(x)
    >>> _mostfunc(x, exp, x) is None
    True
    >>> _mostfunc(exp(x) + exp(x*y), exp, x)
    exp(x)
    """
    fterms = [tmp for tmp in lhs.atoms(func) if (not X or
        X.is_Symbol and X in tmp.free_symbols or
        not X.is_Symbol and tmp.has(X))]
    if len(fterms) == 1:
        return fterms[0]
    elif fterms:
        return max(list(ordered(fterms)), key=lambda x: x.count(func))
    return None


def _linab(arg, symbol):
    """Return ``a, b, X`` assuming ``arg`` can be written as ``a*X + b``
    where ``X`` is a symbol-dependent factor and ``a`` and ``b`` are
    independent of ``symbol``.

    Examples
    ========

    >>> from sympy.solvers.bivariate import _linab
    >>> from sympy.abc import x, y
    >>> from sympy import exp, S
    >>> _linab(S(2), x)
    (2, 0, 1)
    >>> _linab(2*x, x)
    (2, 0, x)
    >>> _linab(y + y*x + 2*x, x)
    (y + 2, y, x)
    >>> _linab(3 + 2*exp(x), x)
    (2, 3, exp(x))
    """
    arg = factor_terms(arg.expand())
    ind, dep = arg.as_independent(symbol)
    if arg.is_Mul and dep.is_Add:
        a, b, x = _linab(dep, symbol)
        return ind*a, ind*b, x
    if not arg.is_Add:
        b = 0
        a, x = ind, dep
    else:
        b = ind
        a, x = separatevars(dep).as_independent(symbol, as_Add=False)
    if x.could_extract_minus_sign():
        a = -a
        x = -x
    return a, b, x


def _lambert(eq, x):
    """
    Given an expression assumed to be in the form
        ``F(X, a..f) = a*log(b*X + c) + d*X + f = 0``
    where X = g(x) and x = g^-1(X), return the Lambert solution,
        ``x = g^-1(-c/b + (a/d)*W(d/(a*b)*exp(c*d/a/b)*exp(-f/a)))``.
    """
    eq = _mexpand(expand_log(eq))
    mainlog = _mostfunc(eq, log, x)
    if not mainlog:
        return []  # violated assumptions
    other = eq.subs(mainlog, 0)
    if isinstance(-other, log):
        eq = (eq - other).subs(mainlog, mainlog.args[0])
        mainlog = mainlog.args[0]
        if not isinstance(mainlog, log):
            return []  # violated assumptions
        other = -(-other).args[0]
        eq += other
    if x not in other.free_symbols:
        return [] # violated assumptions
    d, f, X2 = _linab(other, x)
    logterm = collect(eq - other, mainlog)
    a = logterm.as_coefficient(mainlog)
    if a is None or x in a.free_symbols:
        return []  # violated assumptions
    logarg = mainlog.args[0]
    b, c, X1 = _linab(logarg, x)
    if X1 != X2:
        return []  # violated assumptions

    # invert the generator X1 so we have x(u)
    u = Dummy('rhs')
    xusolns = solve(X1 - u, x)

    # There are infinitely many branches for LambertW
    # but only branches for k = -1 and 0 might be real. The k = 0
    # branch is real and the k = -1 branch is real if the LambertW argument
    # in in range [-1/e, 0]. Since `solve` does not return infinite
    # solutions we will only include the -1 branch if it tests as real.
    # Otherwise, inclusion of any LambertW in the solution indicates to
    #  the user that there are imaginary solutions corresponding to
    # different k values.
    lambert_real_branches = [-1, 0]
    sol = []

    # solution of the given Lambert equation is like
    # sol = -c/b + (a/d)*LambertW(arg, k),
    # where arg = d/(a*b)*exp((c*d-b*f)/a/b) and k in lambert_real_branches.
    # Instead of considering the single arg, `d/(a*b)*exp((c*d-b*f)/a/b)`,
    # the individual `p` roots obtained when writing `exp((c*d-b*f)/a/b)`
    # as `exp(A/p) = exp(A)**(1/p)`, where `p` is an Integer, are used.

    # calculating args for LambertW
    num, den = ((c*d-b*f)/a/b).as_numer_denom()
    p, den = den.as_coeff_Mul()
    e = exp(num/den)
    t = Dummy('t')
    args = [d/(a*b)*t for t in roots(t**p - e, t).keys()]

    # calculating solutions from args
    for arg in args:
        for k in lambert_real_branches:
            w = LambertW(arg, k)
            if k and not w.is_real:
                continue
            rhs = -c/b + (a/d)*w

            sol.extend(xu.subs(u, rhs) for xu in xusolns)
    return sol


def _solve_lambert(f, symbol, gens):
    """Return solution to ``f`` if it is a Lambert-type expression
    else raise NotImplementedError.

    For ``f(X, a..f) = a*log(b*X + c) + d*X - f = 0`` the solution
    for ``X`` is ``X = -c/b + (a/d)*W(d/(a*b)*exp(c*d/a/b)*exp(f/a))``.
    There are a variety of forms for `f(X, a..f)` as enumerated below:

    1a1)
      if B**B = R for R not in [0, 1] (since those cases would already
      be solved before getting here) then log of both sides gives
      log(B) + log(log(B)) = log(log(R)) and
      X = log(B), a = 1, b = 1, c = 0, d = 1, f = log(log(R))
    1a2)
      if B*(b*log(B) + c)**a = R then log of both sides gives
      log(B) + a*log(b*log(B) + c) = log(R) and
      X = log(B), d=1, f=log(R)
    1b)
      if a*log(b*B + c) + d*B = R and
      X = B, f = R
    2a)
      if (b*B + c)*exp(d*B + g) = R then log of both sides gives
      log(b*B + c) + d*B + g = log(R) and
      X = B, a = 1, f = log(R) - g
    2b)
      if g*exp(d*B + h) - b*B = c then the log form is
      log(g) + d*B + h - log(b*B + c) = 0 and
      X = B, a = -1, f = -h - log(g)
    3)
      if d*p**(a*B + g) - b*B = c then the log form is
      log(d) + (a*B + g)*log(p) - log(b*B + c) = 0 and
      X = B, a = -1, d = a*log(p), f = -log(d) - g*log(p)
    """

    def _solve_even_degree_expr(expr, t, symbol):
        """Return the unique solutions of equations derived from
        ``expr`` by replacing ``t`` with ``+/- symbol``.

        Parameters
        ==========

        expr : Expr
            The expression which includes a dummy variable t to be
            replaced with +symbol and -symbol.

        symbol : Symbol
            The symbol for which a solution is being sought.

        Returns
        =======

        List of unique solution of the two equations generated by
        replacing ``t`` with positive and negative ``symbol``.

        Notes
        =====

        If ``expr = 2*log(t) + x/2` then solutions for
        ``2*log(x) + x/2 = 0`` and ``2*log(-x) + x/2 = 0`` are
        returned by this function. Though this may seem
        counter-intuitive, one must note that the ``expr`` being
        solved here has been derived from a different expression. For
        an expression like ``eq = x**2*g(x) = 1``, if we take the
        log of both sides we obtain ``log(x**2) + log(g(x)) = 0``. If
        x is positive then this simplifies to
        ``2*log(x) + log(g(x)) = 0``; the Lambert-solving routines will
        return solutions for this, but we must also consider the
        solutions for  ``2*log(-x) + log(g(x))`` since those must also
        be a solution of ``eq`` which has the same value when the ``x``
        in ``x**2`` is negated. If `g(x)` does not have even powers of
        symbol then we do not want to replace the ``x`` there with
        ``-x``. So the role of the ``t`` in the expression received by
        this function is to mark where ``+/-x`` should be inserted
        before obtaining the Lambert solutions.

        """
        nlhs, plhs = [
            expr.xreplace({t: sgn*symbol}) for sgn in (-1, 1)]
        sols = _solve_lambert(nlhs, symbol, gens)
        if plhs != nlhs:
            sols.extend(_solve_lambert(plhs, symbol, gens))
        # uniq is needed for a case like
        # 2*log(t) - log(-z**2) + log(z + log(x) + log(z))
        # where substituting t with +/-x gives all the same solution;
        # uniq, rather than list(set()), is used to maintain canonical
        # order
        return list(uniq(sols))

    nrhs, lhs = f.as_independent(symbol, as_Add=True)
    rhs = -nrhs

    lamcheck = [tmp for tmp in gens
                if (tmp.func in [exp, log] or
                (tmp.is_Pow and symbol in tmp.exp.free_symbols))]
    if not lamcheck:
        raise NotImplementedError()

    if lhs.is_Add or lhs.is_Mul:
        # replacing all even_degrees of symbol with dummy variable t
        # since these will need special handling; non-Add/Mul do not
        # need this handling
        t = Dummy('t', **symbol.assumptions0)
        lhs = lhs.replace(
            lambda i:  # find symbol**even
                i.is_Pow and i.base == symbol and i.exp.is_even,
            lambda i:  # replace t**even
                t**i.exp)

        if lhs.is_Add and lhs.has(t):
            t_indep = lhs.subs(t, 0)
            t_term = lhs - t_indep
            _rhs = rhs - t_indep
            if not t_term.is_Add and _rhs and not (
                    t_term.has(S.ComplexInfinity, S.NaN)):
                eq = expand_log(log(t_term) - log(_rhs))
                return _solve_even_degree_expr(eq, t, symbol)
        elif lhs.is_Mul and rhs:
            # this needs to happen whether t is present or not
            lhs = expand_log(log(lhs), force=True)
            rhs = log(rhs)
            if lhs.has(t) and lhs.is_Add:
                # it expanded from Mul to Add
                eq = lhs - rhs
                return _solve_even_degree_expr(eq, t, symbol)

        # restore symbol in lhs
        lhs = lhs.xreplace({t: symbol})

    lhs = powsimp(factor(lhs, deep=True))

    # make sure we have inverted as completely as possible
    r = Dummy()
    i, lhs = _invert(lhs - r, symbol)
    rhs = i.xreplace({r: rhs})

    # For the first forms:
    #
    # 1a1) B**B = R will arrive here as B*log(B) = log(R)
    #      lhs is Mul so take log of both sides:
    #        log(B) + log(log(B)) = log(log(R))
    # 1a2) B*(b*log(B) + c)**a = R will arrive unchanged so
    #      lhs is Mul, so take log of both sides:
    #        log(B) + a*log(b*log(B) + c) = log(R)
    # 1b) d*log(a*B + b) + c*B = R will arrive unchanged so
    #      lhs is Add, so isolate c*B and expand log of both sides:
    #        log(c) + log(B) = log(R - d*log(a*B + b))

    soln = []
    if not soln:
        mainlog = _mostfunc(lhs, log, symbol)
        if mainlog:
            if lhs.is_Mul and rhs != 0:
                soln = _lambert(log(lhs) - log(rhs), symbol)
            elif lhs.is_Add:
                other = lhs.subs(mainlog, 0)
                if other and not other.is_Add and [
                        tmp for tmp in other.atoms(Pow)
                        if symbol in tmp.free_symbols]:
                    if not rhs:
                        diff = log(other) - log(other - lhs)
                    else:
                        diff = log(lhs - other) - log(rhs - other)
                    soln = _lambert(expand_log(diff), symbol)
                else:
                    #it's ready to go
                    soln = _lambert(lhs - rhs, symbol)

    # For the next forms,
    #
    #     collect on main exp
    #     2a) (b*B + c)*exp(d*B + g) = R
    #         lhs is mul, so take log of both sides:
    #           log(b*B + c) + d*B = log(R) - g
    #     2b) g*exp(d*B + h) - b*B = R
    #         lhs is add, so add b*B to both sides,
    #         take the log of both sides and rearrange to give
    #           log(R + b*B) - d*B = log(g) + h

    if not soln:
        mainexp = _mostfunc(lhs, exp, symbol)
        if mainexp:
            lhs = collect(lhs, mainexp)
            if lhs.is_Mul and rhs != 0:
                soln = _lambert(expand_log(log(lhs) - log(rhs)), symbol)
            elif lhs.is_Add:
                # move all but mainexp-containing term to rhs
                other = lhs.subs(mainexp, 0)
                mainterm = lhs - other
                rhs = rhs - other
                if (mainterm.could_extract_minus_sign() and
                    rhs.could_extract_minus_sign()):
                    mainterm *= -1
                    rhs *= -1
                diff = log(mainterm) - log(rhs)
                soln = _lambert(expand_log(diff), symbol)

    # For the last form:
    #
    #  3) d*p**(a*B + g) - b*B = c
    #     collect on main pow, add b*B to both sides,
    #     take log of both sides and rearrange to give
    #       a*B*log(p) - log(b*B + c) = -log(d) - g*log(p)
    if not soln:
        mainpow = _mostfunc(lhs, Pow, symbol)
        if mainpow and symbol in mainpow.exp.free_symbols:
            lhs = collect(lhs, mainpow)
            if lhs.is_Mul and rhs != 0:
                # b*B = 0
                soln = _lambert(expand_log(log(lhs) - log(rhs)), symbol)
            elif lhs.is_Add:
                # move all but mainpow-containing term to rhs
                other = lhs.subs(mainpow, 0)
                mainterm = lhs - other
                rhs = rhs - other
                diff = log(mainterm) - log(rhs)
                soln = _lambert(expand_log(diff), symbol)

    if not soln:
        raise NotImplementedError('%s does not appear to have a solution in '
            'terms of LambertW' % f)

    return list(ordered(soln))


def bivariate_type(f, x, y, *, first=True):
    """Given an expression, f, 3 tests will be done to see what type
    of composite bivariate it might be, options for u(x, y) are::

        x*y
        x+y
        x*y+x
        x*y+y

    If it matches one of these types, ``u(x, y)``, ``P(u)`` and dummy
    variable ``u`` will be returned. Solving ``P(u)`` for ``u`` and
    equating the solutions to ``u(x, y)`` and then solving for ``x`` or
    ``y`` is equivalent to solving the original expression for ``x`` or
    ``y``. If ``x`` and ``y`` represent two functions in the same
    variable, e.g. ``x = g(t)`` and ``y = h(t)``, then if ``u(x, y) - p``
    can be solved for ``t`` then these represent the solutions to
    ``P(u) = 0`` when ``p`` are the solutions of ``P(u) = 0``.

    Only positive values of ``u`` are considered.

    Examples
    ========

    >>> from sympy import solve
    >>> from sympy.solvers.bivariate import bivariate_type
    >>> from sympy.abc import x, y
    >>> eq = (x**2 - 3).subs(x, x + y)
    >>> bivariate_type(eq, x, y)
    (x + y, _u**2 - 3, _u)
    >>> uxy, pu, u = _
    >>> usol = solve(pu, u); usol
    [sqrt(3)]
    >>> [solve(uxy - s) for s in solve(pu, u)]
    [[{x: -y + sqrt(3)}]]
    >>> all(eq.subs(s).equals(0) for sol in _ for s in sol)
    True

    """

    u = Dummy('u', positive=True)

    if first:
        p = Poly(f, x, y)
        f = p.as_expr()
        _x = Dummy()
        _y = Dummy()
        rv = bivariate_type(Poly(f.subs({x: _x, y: _y}), _x, _y), _x, _y, first=False)
        if rv:
            reps = {_x: x, _y: y}
            return rv[0].xreplace(reps), rv[1].xreplace(reps), rv[2]
        return

    p = f
    f = p.as_expr()

    # f(x*y)
    args = Add.make_args(p.as_expr())
    new = []
    for a in args:
        a = _mexpand(a.subs(x, u/y))
        free = a.free_symbols
        if x in free or y in free:
            break
        new.append(a)
    else:
        return x*y, Add(*new), u

    def ok(f, v, c):
        new = _mexpand(f.subs(v, c))
        free = new.free_symbols
        return None if (x in free or y in free) else new

    # f(a*x + b*y)
    new = []
    d = p.degree(x)
    if p.degree(y) == d:
        a = root(p.coeff_monomial(x**d), d)
        b = root(p.coeff_monomial(y**d), d)
        new = ok(f, x, (u - b*y)/a)
        if new is not None:
            return a*x + b*y, new, u

    # f(a*x*y + b*y)
    new = []
    d = p.degree(x)
    if p.degree(y) == d:
        for itry in range(2):
            a = root(p.coeff_monomial(x**d*y**d), d)
            b = root(p.coeff_monomial(y**d), d)
            new = ok(f, x, (u - b*y)/a/y)
            if new is not None:
                return a*x*y + b*y, new, u
            x, y = y, x

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\sas\sas_xport.py ===
"""
Read a SAS XPort format file into a Pandas DataFrame.

Based on code from Jack Cushman (github.com/jcushman/xport).

The file format is defined here:

https://support.sas.com/content/dam/SAS/support/en/technical-papers/record-layout-of-a-sas-version-5-or-6-data-set-in-sas-transport-xport-format.pdf
"""
from __future__ import annotations

from collections import abc
from datetime import datetime
import struct
from typing import TYPE_CHECKING
import warnings

import numpy as np

from pandas.util._decorators import Appender
from pandas.util._exceptions import find_stack_level

import pandas as pd

from pandas.io.common import get_handle
from pandas.io.sas.sasreader import ReaderBase

if TYPE_CHECKING:
    from pandas._typing import (
        CompressionOptions,
        DatetimeNaTType,
        FilePath,
        ReadBuffer,
    )
_correct_line1 = (
    "HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!"
    "000000000000000000000000000000  "
)
_correct_header1 = (
    "HEADER RECORD*******MEMBER  HEADER RECORD!!!!!!!000000000000000001600000000"
)
_correct_header2 = (
    "HEADER RECORD*******DSCRPTR HEADER RECORD!!!!!!!"
    "000000000000000000000000000000  "
)
_correct_obs_header = (
    "HEADER RECORD*******OBS     HEADER RECORD!!!!!!!"
    "000000000000000000000000000000  "
)
_fieldkeys = [
    "ntype",
    "nhfun",
    "field_length",
    "nvar0",
    "name",
    "label",
    "nform",
    "nfl",
    "num_decimals",
    "nfj",
    "nfill",
    "niform",
    "nifl",
    "nifd",
    "npos",
    "_",
]


_base_params_doc = """\
Parameters
----------
filepath_or_buffer : str or file-like object
    Path to SAS file or object implementing binary read method."""

_params2_doc = """\
index : identifier of index column
    Identifier of column that should be used as index of the DataFrame.
encoding : str
    Encoding for text data.
chunksize : int
    Read file `chunksize` lines at a time, returns iterator."""

_format_params_doc = """\
format : str
    File format, only `xport` is currently supported."""

_iterator_doc = """\
iterator : bool, default False
    Return XportReader object for reading file incrementally."""


_read_sas_doc = f"""Read a SAS file into a DataFrame.

{_base_params_doc}
{_format_params_doc}
{_params2_doc}
{_iterator_doc}

Returns
-------
DataFrame or XportReader

Examples
--------
Read a SAS Xport file:

>>> df = pd.read_sas('filename.XPT')

Read a Xport file in 10,000 line chunks:

>>> itr = pd.read_sas('filename.XPT', chunksize=10000)
>>> for chunk in itr:
>>>     do_something(chunk)

"""

_xport_reader_doc = f"""\
Class for reading SAS Xport files.

{_base_params_doc}
{_params2_doc}

Attributes
----------
member_info : list
    Contains information about the file
fields : list
    Contains information about the variables in the file
"""

_read_method_doc = """\
Read observations from SAS Xport file, returning as data frame.

Parameters
----------
nrows : int
    Number of rows to read from data file; if None, read whole
    file.

Returns
-------
A DataFrame.
"""


def _parse_date(datestr: str) -> DatetimeNaTType:
    """Given a date in xport format, return Python date."""
    try:
        # e.g. "16FEB11:10:07:55"
        return datetime.strptime(datestr, "%d%b%y:%H:%M:%S")
    except ValueError:
        return pd.NaT


def _split_line(s: str, parts):
    """
    Parameters
    ----------
    s: str
        Fixed-length string to split
    parts: list of (name, length) pairs
        Used to break up string, name '_' will be filtered from output.

    Returns
    -------
    Dict of name:contents of string at given location.
    """
    out = {}
    start = 0
    for name, length in parts:
        out[name] = s[start : start + length].strip()
        start += length
    del out["_"]
    return out


def _handle_truncated_float_vec(vec, nbytes):
    # This feature is not well documented, but some SAS XPORT files
    # have 2-7 byte "truncated" floats.  To read these truncated
    # floats, pad them with zeros on the right to make 8 byte floats.
    #
    # References:
    # https://github.com/jcushman/xport/pull/3
    # The R "foreign" library

    if nbytes != 8:
        vec1 = np.zeros(len(vec), np.dtype("S8"))
        dtype = np.dtype(f"S{nbytes},S{8 - nbytes}")
        vec2 = vec1.view(dtype=dtype)
        vec2["f0"] = vec
        return vec2

    return vec


def _parse_float_vec(vec):
    """
    Parse a vector of float values representing IBM 8 byte floats into
    native 8 byte floats.
    """
    dtype = np.dtype(">u4,>u4")
    vec1 = vec.view(dtype=dtype)
    xport1 = vec1["f0"]
    xport2 = vec1["f1"]

    # Start by setting first half of ieee number to first half of IBM
    # number sans exponent
    ieee1 = xport1 & 0x00FFFFFF

    # The fraction bit to the left of the binary point in the ieee
    # format was set and the number was shifted 0, 1, 2, or 3
    # places. This will tell us how to adjust the ibm exponent to be a
    # power of 2 ieee exponent and how to shift the fraction bits to
    # restore the correct magnitude.
    shift = np.zeros(len(vec), dtype=np.uint8)
    shift[np.where(xport1 & 0x00200000)] = 1
    shift[np.where(xport1 & 0x00400000)] = 2
    shift[np.where(xport1 & 0x00800000)] = 3

    # shift the ieee number down the correct number of places then
    # set the second half of the ieee number to be the second half
    # of the ibm number shifted appropriately, ored with the bits
    # from the first half that would have been shifted in if we
    # could shift a double. All we are worried about are the low
    # order 3 bits of the first half since we're only shifting by
    # 1, 2, or 3.
    ieee1 >>= shift
    ieee2 = (xport2 >> shift) | ((xport1 & 0x00000007) << (29 + (3 - shift)))

    # clear the 1 bit to the left of the binary point
    ieee1 &= 0xFFEFFFFF

    # set the exponent of the ieee number to be the actual exponent
    # plus the shift count + 1023. Or this into the first half of the
    # ieee number. The ibm exponent is excess 64 but is adjusted by 65
    # since during conversion to ibm format the exponent is
    # incremented by 1 and the fraction bits left 4 positions to the
    # right of the radix point.  (had to add >> 24 because C treats &
    # 0x7f as 0x7f000000 and Python doesn't)
    ieee1 |= ((((((xport1 >> 24) & 0x7F) - 65) << 2) + shift + 1023) << 20) | (
        xport1 & 0x80000000
    )

    ieee = np.empty((len(ieee1),), dtype=">u4,>u4")
    ieee["f0"] = ieee1
    ieee["f1"] = ieee2
    ieee = ieee.view(dtype=">f8")
    ieee = ieee.astype("f8")

    return ieee


class XportReader(ReaderBase, abc.Iterator):
    __doc__ = _xport_reader_doc

    def __init__(
        self,
        filepath_or_buffer: FilePath | ReadBuffer[bytes],
        index=None,
        encoding: str | None = "ISO-8859-1",
        chunksize: int | None = None,
        compression: CompressionOptions = "infer",
    ) -> None:
        self._encoding = encoding
        self._lines_read = 0
        self._index = index
        self._chunksize = chunksize

        self.handles = get_handle(
            filepath_or_buffer,
            "rb",
            encoding=encoding,
            is_text=False,
            compression=compression,
        )
        self.filepath_or_buffer = self.handles.handle

        try:
            self._read_header()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self.handles.close()

    def _get_row(self):
        return self.filepath_or_buffer.read(80).decode()

    def _read_header(self) -> None:
        self.filepath_or_buffer.seek(0)

        # read file header
        line1 = self._get_row()
        if line1 != _correct_line1:
            if "**COMPRESSED**" in line1:
                # this was created with the PROC CPORT method and can't be read
                # https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.5/movefile/p1bm6aqp3fw4uin1hucwh718f6kp.htm
                raise ValueError(
                    "Header record indicates a CPORT file, which is not readable."
                )
            raise ValueError("Header record is not an XPORT file.")

        line2 = self._get_row()
        fif = [["prefix", 24], ["version", 8], ["OS", 8], ["_", 24], ["created", 16]]
        file_info = _split_line(line2, fif)
        if file_info["prefix"] != "SAS     SAS     SASLIB":
            raise ValueError("Header record has invalid prefix.")
        file_info["created"] = _parse_date(file_info["created"])
        self.file_info = file_info

        line3 = self._get_row()
        file_info["modified"] = _parse_date(line3[:16])

        # read member header
        header1 = self._get_row()
        header2 = self._get_row()
        headflag1 = header1.startswith(_correct_header1)
        headflag2 = header2 == _correct_header2
        if not (headflag1 and headflag2):
            raise ValueError("Member header not found")
        # usually 140, could be 135
        fieldnamelength = int(header1[-5:-2])

        # member info
        mem = [
            ["prefix", 8],
            ["set_name", 8],
            ["sasdata", 8],
            ["version", 8],
            ["OS", 8],
            ["_", 24],
            ["created", 16],
        ]
        member_info = _split_line(self._get_row(), mem)
        mem = [["modified", 16], ["_", 16], ["label", 40], ["type", 8]]
        member_info.update(_split_line(self._get_row(), mem))
        member_info["modified"] = _parse_date(member_info["modified"])
        member_info["created"] = _parse_date(member_info["created"])
        self.member_info = member_info

        # read field names
        types = {1: "numeric", 2: "char"}
        fieldcount = int(self._get_row()[54:58])
        datalength = fieldnamelength * fieldcount
        # round up to nearest 80
        if datalength % 80:
            datalength += 80 - datalength % 80
        fielddata = self.filepath_or_buffer.read(datalength)
        fields = []
        obs_length = 0
        while len(fielddata) >= fieldnamelength:
            # pull data for one field
            fieldbytes, fielddata = (
                fielddata[:fieldnamelength],
                fielddata[fieldnamelength:],
            )

            # rest at end gets ignored, so if field is short, pad out
            # to match struct pattern below
            fieldbytes = fieldbytes.ljust(140)

            fieldstruct = struct.unpack(">hhhh8s40s8shhh2s8shhl52s", fieldbytes)
            field = dict(zip(_fieldkeys, fieldstruct))
            del field["_"]
            field["ntype"] = types[field["ntype"]]
            fl = field["field_length"]
            if field["ntype"] == "numeric" and ((fl < 2) or (fl > 8)):
                msg = f"Floating field width {fl} is not between 2 and 8."
                raise TypeError(msg)

            for k, v in field.items():
                try:
                    field[k] = v.strip()
                except AttributeError:
                    pass

            obs_length += field["field_length"]
            fields += [field]

        header = self._get_row()
        if not header == _correct_obs_header:
            raise ValueError("Observation header not found.")

        self.fields = fields
        self.record_length = obs_length
        self.record_start = self.filepath_or_buffer.tell()

        self.nobs = self._record_count()
        self.columns = [x["name"].decode() for x in self.fields]

        # Setup the dtype.
        dtypel = [
            ("s" + str(i), "S" + str(field["field_length"]))
            for i, field in enumerate(self.fields)
        ]
        dtype = np.dtype(dtypel)
        self._dtype = dtype

    def __next__(self) -> pd.DataFrame:
        return self.read(nrows=self._chunksize or 1)

    def _record_count(self) -> int:
        """
        Get number of records in file.

        This is maybe suboptimal because we have to seek to the end of
        the file.

        Side effect: returns file position to record_start.
        """
        self.filepath_or_buffer.seek(0, 2)
        total_records_length = self.filepath_or_buffer.tell() - self.record_start

        if total_records_length % 80 != 0:
            warnings.warn(
                "xport file may be corrupted.",
                stacklevel=find_stack_level(),
            )

        if self.record_length > 80:
            self.filepath_or_buffer.seek(self.record_start)
            return total_records_length // self.record_length

        self.filepath_or_buffer.seek(-80, 2)
        last_card_bytes = self.filepath_or_buffer.read(80)
        last_card = np.frombuffer(last_card_bytes, dtype=np.uint64)

        # 8 byte blank
        ix = np.flatnonzero(last_card == 2314885530818453536)

        if len(ix) == 0:
            tail_pad = 0
        else:
            tail_pad = 8 * len(ix)

        self.filepath_or_buffer.seek(self.record_start)

        return (total_records_length - tail_pad) // self.record_length

    def get_chunk(self, size: int | None = None) -> pd.DataFrame:
        """
        Reads lines from Xport file and returns as dataframe

        Parameters
        ----------
        size : int, defaults to None
            Number of lines to read.  If None, reads whole file.

        Returns
        -------
        DataFrame
        """
        if size is None:
            size = self._chunksize
        return self.read(nrows=size)

    def _missing_double(self, vec):
        v = vec.view(dtype="u1,u1,u2,u4")
        miss = (v["f1"] == 0) & (v["f2"] == 0) & (v["f3"] == 0)
        miss1 = (
            ((v["f0"] >= 0x41) & (v["f0"] <= 0x5A))
            | (v["f0"] == 0x5F)
            | (v["f0"] == 0x2E)
        )
        miss &= miss1
        return miss

    @Appender(_read_method_doc)
    def read(self, nrows: int | None = None) -> pd.DataFrame:
        if nrows is None:
            nrows = self.nobs

        read_lines = min(nrows, self.nobs - self._lines_read)
        read_len = read_lines * self.record_length
        if read_len <= 0:
            self.close()
            raise StopIteration
        raw = self.filepath_or_buffer.read(read_len)
        data = np.frombuffer(raw, dtype=self._dtype, count=read_lines)

        df_data = {}
        for j, x in enumerate(self.columns):
            vec = data["s" + str(j)]
            ntype = self.fields[j]["ntype"]
            if ntype == "numeric":
                vec = _handle_truncated_float_vec(vec, self.fields[j]["field_length"])
                miss = self._missing_double(vec)
                v = _parse_float_vec(vec)
                v[miss] = np.nan
            elif self.fields[j]["ntype"] == "char":
                v = [y.rstrip() for y in vec]

                if self._encoding is not None:
                    v = [y.decode(self._encoding) for y in v]

            df_data.update({x: v})
        df = pd.DataFrame(df_data)

        if self._index is None:
            df.index = pd.Index(range(self._lines_read, self._lines_read + read_lines))
        else:
            df = df.set_index(self._index)

        self._lines_read += read_lines

        return df

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\util\_decorators.py ===
from __future__ import annotations

from functools import wraps
import inspect
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    cast,
)
import warnings

from pandas._libs.properties import cache_readonly
from pandas._typing import (
    F,
    T,
)
from pandas.util._exceptions import find_stack_level

if TYPE_CHECKING:
    from collections.abc import Mapping


def deprecate(
    name: str,
    alternative: Callable[..., Any],
    version: str,
    alt_name: str | None = None,
    klass: type[Warning] | None = None,
    stacklevel: int = 2,
    msg: str | None = None,
) -> Callable[[F], F]:
    """
    Return a new function that emits a deprecation warning on use.

    To use this method for a deprecated function, another function
    `alternative` with the same signature must exist. The deprecated
    function will emit a deprecation warning, and in the docstring
    it will contain the deprecation directive with the provided version
    so it can be detected for future removal.

    Parameters
    ----------
    name : str
        Name of function to deprecate.
    alternative : func
        Function to use instead.
    version : str
        Version of pandas in which the method has been deprecated.
    alt_name : str, optional
        Name to use in preference of alternative.__name__.
    klass : Warning, default FutureWarning
    stacklevel : int, default 2
    msg : str
        The message to display in the warning.
        Default is '{name} is deprecated. Use {alt_name} instead.'
    """
    alt_name = alt_name or alternative.__name__
    klass = klass or FutureWarning
    warning_msg = msg or f"{name} is deprecated, use {alt_name} instead."

    @wraps(alternative)
    def wrapper(*args, **kwargs) -> Callable[..., Any]:
        warnings.warn(warning_msg, klass, stacklevel=stacklevel)
        return alternative(*args, **kwargs)

    # adding deprecated directive to the docstring
    msg = msg or f"Use `{alt_name}` instead."
    doc_error_msg = (
        "deprecate needs a correctly formatted docstring in "
        "the target function (should have a one liner short "
        "summary, and opening quotes should be in their own "
        f"line). Found:\n{alternative.__doc__}"
    )

    # when python is running in optimized mode (i.e. `-OO`), docstrings are
    # removed, so we check that a docstring with correct formatting is used
    # but we allow empty docstrings
    if alternative.__doc__:
        if alternative.__doc__.count("\n") < 3:
            raise AssertionError(doc_error_msg)
        empty1, summary, empty2, doc_string = alternative.__doc__.split("\n", 3)
        if empty1 or empty2 and not summary:
            raise AssertionError(doc_error_msg)
        wrapper.__doc__ = dedent(
            f"""
        {summary.strip()}

        .. deprecated:: {version}
            {msg}

        {dedent(doc_string)}"""
        )
    # error: Incompatible return value type (got "Callable[[VarArg(Any), KwArg(Any)],
    # Callable[...,Any]]", expected "Callable[[F], F]")
    return wrapper  # type: ignore[return-value]


def deprecate_kwarg(
    old_arg_name: str,
    new_arg_name: str | None,
    mapping: Mapping[Any, Any] | Callable[[Any], Any] | None = None,
    stacklevel: int = 2,
) -> Callable[[F], F]:
    """
    Decorator to deprecate a keyword argument of a function.

    Parameters
    ----------
    old_arg_name : str
        Name of argument in function to deprecate
    new_arg_name : str or None
        Name of preferred argument in function. Use None to raise warning that
        ``old_arg_name`` keyword is deprecated.
    mapping : dict or callable
        If mapping is present, use it to translate old arguments to
        new arguments. A callable must do its own value checking;
        values not found in a dict will be forwarded unchanged.

    Examples
    --------
    The following deprecates 'cols', using 'columns' instead

    >>> @deprecate_kwarg(old_arg_name='cols', new_arg_name='columns')
    ... def f(columns=''):
    ...     print(columns)
    ...
    >>> f(columns='should work ok')
    should work ok

    >>> f(cols='should raise warning')  # doctest: +SKIP
    FutureWarning: cols is deprecated, use columns instead
      warnings.warn(msg, FutureWarning)
    should raise warning

    >>> f(cols='should error', columns="can\'t pass do both")  # doctest: +SKIP
    TypeError: Can only specify 'cols' or 'columns', not both

    >>> @deprecate_kwarg('old', 'new', {'yes': True, 'no': False})
    ... def f(new=False):
    ...     print('yes!' if new else 'no!')
    ...
    >>> f(old='yes')  # doctest: +SKIP
    FutureWarning: old='yes' is deprecated, use new=True instead
      warnings.warn(msg, FutureWarning)
    yes!

    To raise a warning that a keyword will be removed entirely in the future

    >>> @deprecate_kwarg(old_arg_name='cols', new_arg_name=None)
    ... def f(cols='', another_param=''):
    ...     print(cols)
    ...
    >>> f(cols='should raise warning')  # doctest: +SKIP
    FutureWarning: the 'cols' keyword is deprecated and will be removed in a
    future version please takes steps to stop use of 'cols'
    should raise warning
    >>> f(another_param='should not raise warning')  # doctest: +SKIP
    should not raise warning

    >>> f(cols='should raise warning', another_param='')  # doctest: +SKIP
    FutureWarning: the 'cols' keyword is deprecated and will be removed in a
    future version please takes steps to stop use of 'cols'
    should raise warning
    """
    if mapping is not None and not hasattr(mapping, "get") and not callable(mapping):
        raise TypeError(
            "mapping from old to new argument values must be dict or callable!"
        )

    def _deprecate_kwarg(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Callable[..., Any]:
            old_arg_value = kwargs.pop(old_arg_name, None)

            if old_arg_value is not None:
                if new_arg_name is None:
                    msg = (
                        f"the {repr(old_arg_name)} keyword is deprecated and "
                        "will be removed in a future version. Please take "
                        f"steps to stop the use of {repr(old_arg_name)}"
                    )
                    warnings.warn(msg, FutureWarning, stacklevel=stacklevel)
                    kwargs[old_arg_name] = old_arg_value
                    return func(*args, **kwargs)

                elif mapping is not None:
                    if callable(mapping):
                        new_arg_value = mapping(old_arg_value)
                    else:
                        new_arg_value = mapping.get(old_arg_value, old_arg_value)
                    msg = (
                        f"the {old_arg_name}={repr(old_arg_value)} keyword is "
                        "deprecated, use "
                        f"{new_arg_name}={repr(new_arg_value)} instead."
                    )
                else:
                    new_arg_value = old_arg_value
                    msg = (
                        f"the {repr(old_arg_name)} keyword is deprecated, "
                        f"use {repr(new_arg_name)} instead."
                    )

                warnings.warn(msg, FutureWarning, stacklevel=stacklevel)
                if kwargs.get(new_arg_name) is not None:
                    msg = (
                        f"Can only specify {repr(old_arg_name)} "
                        f"or {repr(new_arg_name)}, not both."
                    )
                    raise TypeError(msg)
                kwargs[new_arg_name] = new_arg_value
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return _deprecate_kwarg


def _format_argument_list(allow_args: list[str]) -> str:
    """
    Convert the allow_args argument (either string or integer) of
    `deprecate_nonkeyword_arguments` function to a string describing
    it to be inserted into warning message.

    Parameters
    ----------
    allowed_args : list, tuple or int
        The `allowed_args` argument for `deprecate_nonkeyword_arguments`,
        but None value is not allowed.

    Returns
    -------
    str
        The substring describing the argument list in best way to be
        inserted to the warning message.

    Examples
    --------
    `format_argument_list([])` -> ''
    `format_argument_list(['a'])` -> "except for the arguments 'a'"
    `format_argument_list(['a', 'b'])` -> "except for the arguments 'a' and 'b'"
    `format_argument_list(['a', 'b', 'c'])` ->
        "except for the arguments 'a', 'b' and 'c'"
    """
    if "self" in allow_args:
        allow_args.remove("self")
    if not allow_args:
        return ""
    elif len(allow_args) == 1:
        return f" except for the argument '{allow_args[0]}'"
    else:
        last = allow_args[-1]
        args = ", ".join(["'" + x + "'" for x in allow_args[:-1]])
        return f" except for the arguments {args} and '{last}'"


def future_version_msg(version: str | None) -> str:
    """Specify which version of pandas the deprecation will take place in."""
    if version is None:
        return "In a future version of pandas"
    else:
        return f"Starting with pandas version {version}"


def deprecate_nonkeyword_arguments(
    version: str | None,
    allowed_args: list[str] | None = None,
    name: str | None = None,
) -> Callable[[F], F]:
    """
    Decorator to deprecate a use of non-keyword arguments of a function.

    Parameters
    ----------
    version : str, optional
        The version in which positional arguments will become
        keyword-only. If None, then the warning message won't
        specify any particular version.

    allowed_args : list, optional
        In case of list, it must be the list of names of some
        first arguments of the decorated functions that are
        OK to be given as positional arguments. In case of None value,
        defaults to list of all arguments not having the
        default value.

    name : str, optional
        The specific name of the function to show in the warning
        message. If None, then the Qualified name of the function
        is used.
    """

    def decorate(func):
        old_sig = inspect.signature(func)

        if allowed_args is not None:
            allow_args = allowed_args
        else:
            allow_args = [
                p.name
                for p in old_sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.default is p.empty
            ]

        new_params = [
            p.replace(kind=p.KEYWORD_ONLY)
            if (
                p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.name not in allow_args
            )
            else p
            for p in old_sig.parameters.values()
        ]
        new_params.sort(key=lambda p: p.kind)
        new_sig = old_sig.replace(parameters=new_params)

        num_allow_args = len(allow_args)
        msg = (
            f"{future_version_msg(version)} all arguments of "
            f"{name or func.__qualname__}{{arguments}} will be keyword-only."
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) > num_allow_args:
                warnings.warn(
                    msg.format(arguments=_format_argument_list(allow_args)),
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
            return func(*args, **kwargs)

        # error: "Callable[[VarArg(Any), KwArg(Any)], Any]" has no
        # attribute "__signature__"
        wrapper.__signature__ = new_sig  # type: ignore[attr-defined]
        return wrapper

    return decorate


def doc(*docstrings: None | str | Callable, **params) -> Callable[[F], F]:
    """
    A decorator to take docstring templates, concatenate them and perform string
    substitution on them.

    This decorator will add a variable "_docstring_components" to the wrapped
    callable to keep track the original docstring template for potential usage.
    If it should be consider as a template, it will be saved as a string.
    Otherwise, it will be saved as callable, and later user __doc__ and dedent
    to get docstring.

    Parameters
    ----------
    *docstrings : None, str, or callable
        The string / docstring / docstring template to be appended in order
        after default docstring under callable.
    **params
        The string which would be used to format docstring template.
    """

    def decorator(decorated: F) -> F:
        # collecting docstring and docstring templates
        docstring_components: list[str | Callable] = []
        if decorated.__doc__:
            docstring_components.append(dedent(decorated.__doc__))

        for docstring in docstrings:
            if docstring is None:
                continue
            if hasattr(docstring, "_docstring_components"):
                docstring_components.extend(
                    docstring._docstring_components  # pyright: ignore[reportGeneralTypeIssues]
                )
            elif isinstance(docstring, str) or docstring.__doc__:
                docstring_components.append(docstring)

        params_applied = [
            component.format(**params)
            if isinstance(component, str) and len(params) > 0
            else component
            for component in docstring_components
        ]

        decorated.__doc__ = "".join(
            [
                component
                if isinstance(component, str)
                else dedent(component.__doc__ or "")
                for component in params_applied
            ]
        )

        # error: "F" has no attribute "_docstring_components"
        decorated._docstring_components = (  # type: ignore[attr-defined]
            docstring_components
        )
        return decorated

    return decorator


# Substitution and Appender are derived from matplotlib.docstring (1.1.0)
# module https://matplotlib.org/users/license.html


class Substitution:
    """
    A decorator to take a function's docstring and perform string
    substitution on it.

    This decorator should be robust even if func.__doc__ is None
    (for example, if -OO was passed to the interpreter)

    Usage: construct a docstring.Substitution with a sequence or
    dictionary suitable for performing substitution; then
    decorate a suitable function with the constructed object. e.g.

    sub_author_name = Substitution(author='Jason')

    @sub_author_name
    def some_function(x):
        "%(author)s wrote this function"

    # note that some_function.__doc__ is now "Jason wrote this function"

    One can also use positional arguments.

    sub_first_last_names = Substitution('Edgar Allen', 'Poe')

    @sub_first_last_names
    def some_function(x):
        "%s %s wrote the Raven"
    """

    def __init__(self, *args, **kwargs) -> None:
        if args and kwargs:
            raise AssertionError("Only positional or keyword args are allowed")

        self.params = args or kwargs

    def __call__(self, func: F) -> F:
        func.__doc__ = func.__doc__ and func.__doc__ % self.params
        return func

    def update(self, *args, **kwargs) -> None:
        """
        Update self.params with supplied args.
        """
        if isinstance(self.params, dict):
            self.params.update(*args, **kwargs)


class Appender:
    """
    A function decorator that will append an addendum to the docstring
    of the target function.

    This decorator should be robust even if func.__doc__ is None
    (for example, if -OO was passed to the interpreter).

    Usage: construct a docstring.Appender with a string to be joined to
    the original docstring. An optional 'join' parameter may be supplied
    which will be used to join the docstring and addendum. e.g.

    add_copyright = Appender("Copyright (c) 2009", join='\n')

    @add_copyright
    def my_dog(has='fleas'):
        "This docstring will have a copyright below"
        pass
    """

    addendum: str | None

    def __init__(self, addendum: str | None, join: str = "", indents: int = 0) -> None:
        if indents > 0:
            self.addendum = indent(addendum, indents=indents)
        else:
            self.addendum = addendum
        self.join = join

    def __call__(self, func: T) -> T:
        func.__doc__ = func.__doc__ if func.__doc__ else ""
        self.addendum = self.addendum if self.addendum else ""
        docitems = [func.__doc__, self.addendum]
        func.__doc__ = dedent(self.join.join(docitems))
        return func


def indent(text: str | None, indents: int = 1) -> str:
    if not text or not isinstance(text, str):
        return ""
    jointext = "".join(["\n"] + ["    "] * indents)
    return jointext.join(text.split("\n"))


__all__ = [
    "Appender",
    "cache_readonly",
    "deprecate",
    "deprecate_kwarg",
    "deprecate_nonkeyword_arguments",
    "doc",
    "future_version_msg",
    "Substitution",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\distlib\index.py ===
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013-2023 Vinay Sajip.
# Licensed to the Python Software Foundation under a contributor agreement.
# See LICENSE.txt and CONTRIBUTORS.txt.
#
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
try:
    from threading import Thread
except ImportError:  # pragma: no cover
    from dummy_threading import Thread

from . import DistlibException
from .compat import (HTTPBasicAuthHandler, Request, HTTPPasswordMgr,
                     urlparse, build_opener, string_types)
from .util import zip_dir, ServerProxy

logger = logging.getLogger(__name__)

DEFAULT_INDEX = 'https://pypi.org/pypi'
DEFAULT_REALM = 'pypi'


class PackageIndex(object):
    """
    This class represents a package index compatible with PyPI, the Python
    Package Index.
    """

    boundary = b'----------ThIs_Is_tHe_distlib_index_bouNdaRY_$'

    def __init__(self, url=None):
        """
        Initialise an instance.

        :param url: The URL of the index. If not specified, the URL for PyPI is
                    used.
        """
        self.url = url or DEFAULT_INDEX
        self.read_configuration()
        scheme, netloc, path, params, query, frag = urlparse(self.url)
        if params or query or frag or scheme not in ('http', 'https'):
            raise DistlibException('invalid repository: %s' % self.url)
        self.password_handler = None
        self.ssl_verifier = None
        self.gpg = None
        self.gpg_home = None
        with open(os.devnull, 'w') as sink:
            # Use gpg by default rather than gpg2, as gpg2 insists on
            # prompting for passwords
            for s in ('gpg', 'gpg2'):
                try:
                    rc = subprocess.check_call([s, '--version'], stdout=sink,
                                               stderr=sink)
                    if rc == 0:
                        self.gpg = s
                        break
                except OSError:
                    pass

    def _get_pypirc_command(self):
        """
        Get the distutils command for interacting with PyPI configurations.
        :return: the command.
        """
        from .util import _get_pypirc_command as cmd
        return cmd()

    def read_configuration(self):
        """
        Read the PyPI access configuration as supported by distutils. This populates
        ``username``, ``password``, ``realm`` and ``url`` attributes from the
        configuration.
        """
        from .util import _load_pypirc
        cfg = _load_pypirc(self)
        self.username = cfg.get('username')
        self.password = cfg.get('password')
        self.realm = cfg.get('realm', 'pypi')
        self.url = cfg.get('repository', self.url)

    def save_configuration(self):
        """
        Save the PyPI access configuration. You must have set ``username`` and
        ``password`` attributes before calling this method.
        """
        self.check_credentials()
        from .util import _store_pypirc
        _store_pypirc(self)

    def check_credentials(self):
        """
        Check that ``username`` and ``password`` have been set, and raise an
        exception if not.
        """
        if self.username is None or self.password is None:
            raise DistlibException('username and password must be set')
        pm = HTTPPasswordMgr()
        _, netloc, _, _, _, _ = urlparse(self.url)
        pm.add_password(self.realm, netloc, self.username, self.password)
        self.password_handler = HTTPBasicAuthHandler(pm)

    def register(self, metadata):  # pragma: no cover
        """
        Register a distribution on PyPI, using the provided metadata.

        :param metadata: A :class:`Metadata` instance defining at least a name
                         and version number for the distribution to be
                         registered.
        :return: The HTTP response received from PyPI upon submission of the
                request.
        """
        self.check_credentials()
        metadata.validate()
        d = metadata.todict()
        d[':action'] = 'verify'
        request = self.encode_request(d.items(), [])
        self.send_request(request)
        d[':action'] = 'submit'
        request = self.encode_request(d.items(), [])
        return self.send_request(request)

    def _reader(self, name, stream, outbuf):
        """
        Thread runner for reading lines of from a subprocess into a buffer.

        :param name: The logical name of the stream (used for logging only).
        :param stream: The stream to read from. This will typically a pipe
                       connected to the output stream of a subprocess.
        :param outbuf: The list to append the read lines to.
        """
        while True:
            s = stream.readline()
            if not s:
                break
            s = s.decode('utf-8').rstrip()
            outbuf.append(s)
            logger.debug('%s: %s' % (name, s))
        stream.close()

    def get_sign_command(self, filename, signer, sign_password, keystore=None):  # pragma: no cover
        """
        Return a suitable command for signing a file.

        :param filename: The pathname to the file to be signed.
        :param signer: The identifier of the signer of the file.
        :param sign_password: The passphrase for the signer's
                              private key used for signing.
        :param keystore: The path to a directory which contains the keys
                         used in verification. If not specified, the
                         instance's ``gpg_home`` attribute is used instead.
        :return: The signing command as a list suitable to be
                 passed to :class:`subprocess.Popen`.
        """
        cmd = [self.gpg, '--status-fd', '2', '--no-tty']
        if keystore is None:
            keystore = self.gpg_home
        if keystore:
            cmd.extend(['--homedir', keystore])
        if sign_password is not None:
            cmd.extend(['--batch', '--passphrase-fd', '0'])
        td = tempfile.mkdtemp()
        sf = os.path.join(td, os.path.basename(filename) + '.asc')
        cmd.extend(['--detach-sign', '--armor', '--local-user',
                    signer, '--output', sf, filename])
        logger.debug('invoking: %s', ' '.join(cmd))
        return cmd, sf

    def run_command(self, cmd, input_data=None):
        """
        Run a command in a child process , passing it any input data specified.

        :param cmd: The command to run.
        :param input_data: If specified, this must be a byte string containing
                           data to be sent to the child process.
        :return: A tuple consisting of the subprocess' exit code, a list of
                 lines read from the subprocess' ``stdout``, and a list of
                 lines read from the subprocess' ``stderr``.
        """
        kwargs = {
            'stdout': subprocess.PIPE,
            'stderr': subprocess.PIPE,
        }
        if input_data is not None:
            kwargs['stdin'] = subprocess.PIPE
        stdout = []
        stderr = []
        p = subprocess.Popen(cmd, **kwargs)
        # We don't use communicate() here because we may need to
        # get clever with interacting with the command
        t1 = Thread(target=self._reader, args=('stdout', p.stdout, stdout))
        t1.start()
        t2 = Thread(target=self._reader, args=('stderr', p.stderr, stderr))
        t2.start()
        if input_data is not None:
            p.stdin.write(input_data)
            p.stdin.close()

        p.wait()
        t1.join()
        t2.join()
        return p.returncode, stdout, stderr

    def sign_file(self, filename, signer, sign_password, keystore=None):  # pragma: no cover
        """
        Sign a file.

        :param filename: The pathname to the file to be signed.
        :param signer: The identifier of the signer of the file.
        :param sign_password: The passphrase for the signer's
                              private key used for signing.
        :param keystore: The path to a directory which contains the keys
                         used in signing. If not specified, the instance's
                         ``gpg_home`` attribute is used instead.
        :return: The absolute pathname of the file where the signature is
                 stored.
        """
        cmd, sig_file = self.get_sign_command(filename, signer, sign_password,
                                              keystore)
        rc, stdout, stderr = self.run_command(cmd,
                                              sign_password.encode('utf-8'))
        if rc != 0:
            raise DistlibException('sign command failed with error '
                                   'code %s' % rc)
        return sig_file

    def upload_file(self, metadata, filename, signer=None, sign_password=None,
                    filetype='sdist', pyversion='source', keystore=None):
        """
        Upload a release file to the index.

        :param metadata: A :class:`Metadata` instance defining at least a name
                         and version number for the file to be uploaded.
        :param filename: The pathname of the file to be uploaded.
        :param signer: The identifier of the signer of the file.
        :param sign_password: The passphrase for the signer's
                              private key used for signing.
        :param filetype: The type of the file being uploaded. This is the
                        distutils command which produced that file, e.g.
                        ``sdist`` or ``bdist_wheel``.
        :param pyversion: The version of Python which the release relates
                          to. For code compatible with any Python, this would
                          be ``source``, otherwise it would be e.g. ``3.2``.
        :param keystore: The path to a directory which contains the keys
                         used in signing. If not specified, the instance's
                         ``gpg_home`` attribute is used instead.
        :return: The HTTP response received from PyPI upon submission of the
                request.
        """
        self.check_credentials()
        if not os.path.exists(filename):
            raise DistlibException('not found: %s' % filename)
        metadata.validate()
        d = metadata.todict()
        sig_file = None
        if signer:
            if not self.gpg:
                logger.warning('no signing program available - not signed')
            else:
                sig_file = self.sign_file(filename, signer, sign_password,
                                          keystore)
        with open(filename, 'rb') as f:
            file_data = f.read()
        md5_digest = hashlib.md5(file_data).hexdigest()
        sha256_digest = hashlib.sha256(file_data).hexdigest()
        d.update({
            ':action': 'file_upload',
            'protocol_version': '1',
            'filetype': filetype,
            'pyversion': pyversion,
            'md5_digest': md5_digest,
            'sha256_digest': sha256_digest,
        })
        files = [('content', os.path.basename(filename), file_data)]
        if sig_file:
            with open(sig_file, 'rb') as f:
                sig_data = f.read()
            files.append(('gpg_signature', os.path.basename(sig_file),
                         sig_data))
            shutil.rmtree(os.path.dirname(sig_file))
        request = self.encode_request(d.items(), files)
        return self.send_request(request)

    def upload_documentation(self, metadata, doc_dir):  # pragma: no cover
        """
        Upload documentation to the index.

        :param metadata: A :class:`Metadata` instance defining at least a name
                         and version number for the documentation to be
                         uploaded.
        :param doc_dir: The pathname of the directory which contains the
                        documentation. This should be the directory that
                        contains the ``index.html`` for the documentation.
        :return: The HTTP response received from PyPI upon submission of the
                request.
        """
        self.check_credentials()
        if not os.path.isdir(doc_dir):
            raise DistlibException('not a directory: %r' % doc_dir)
        fn = os.path.join(doc_dir, 'index.html')
        if not os.path.exists(fn):
            raise DistlibException('not found: %r' % fn)
        metadata.validate()
        name, version = metadata.name, metadata.version
        zip_data = zip_dir(doc_dir).getvalue()
        fields = [(':action', 'doc_upload'),
                  ('name', name), ('version', version)]
        files = [('content', name, zip_data)]
        request = self.encode_request(fields, files)
        return self.send_request(request)

    def get_verify_command(self, signature_filename, data_filename,
                           keystore=None):
        """
        Return a suitable command for verifying a file.

        :param signature_filename: The pathname to the file containing the
                                   signature.
        :param data_filename: The pathname to the file containing the
                              signed data.
        :param keystore: The path to a directory which contains the keys
                         used in verification. If not specified, the
                         instance's ``gpg_home`` attribute is used instead.
        :return: The verifying command as a list suitable to be
                 passed to :class:`subprocess.Popen`.
        """
        cmd = [self.gpg, '--status-fd', '2', '--no-tty']
        if keystore is None:
            keystore = self.gpg_home
        if keystore:
            cmd.extend(['--homedir', keystore])
        cmd.extend(['--verify', signature_filename, data_filename])
        logger.debug('invoking: %s', ' '.join(cmd))
        return cmd

    def verify_signature(self, signature_filename, data_filename,
                         keystore=None):
        """
        Verify a signature for a file.

        :param signature_filename: The pathname to the file containing the
                                   signature.
        :param data_filename: The pathname to the file containing the
                              signed data.
        :param keystore: The path to a directory which contains the keys
                         used in verification. If not specified, the
                         instance's ``gpg_home`` attribute is used instead.
        :return: True if the signature was verified, else False.
        """
        if not self.gpg:
            raise DistlibException('verification unavailable because gpg '
                                   'unavailable')
        cmd = self.get_verify_command(signature_filename, data_filename,
                                      keystore)
        rc, stdout, stderr = self.run_command(cmd)
        if rc not in (0, 1):
            raise DistlibException('verify command failed with error code %s' % rc)
        return rc == 0

    def download_file(self, url, destfile, digest=None, reporthook=None):
        """
        This is a convenience method for downloading a file from an URL.
        Normally, this will be a file from the index, though currently
        no check is made for this (i.e. a file can be downloaded from
        anywhere).

        The method is just like the :func:`urlretrieve` function in the
        standard library, except that it allows digest computation to be
        done during download and checking that the downloaded data
        matched any expected value.

        :param url: The URL of the file to be downloaded (assumed to be
                    available via an HTTP GET request).
        :param destfile: The pathname where the downloaded file is to be
                         saved.
        :param digest: If specified, this must be a (hasher, value)
                       tuple, where hasher is the algorithm used (e.g.
                       ``'md5'``) and ``value`` is the expected value.
        :param reporthook: The same as for :func:`urlretrieve` in the
                           standard library.
        """
        if digest is None:
            digester = None
            logger.debug('No digest specified')
        else:
            if isinstance(digest, (list, tuple)):
                hasher, digest = digest
            else:
                hasher = 'md5'
            digester = getattr(hashlib, hasher)()
            logger.debug('Digest specified: %s' % digest)
        # The following code is equivalent to urlretrieve.
        # We need to do it this way so that we can compute the
        # digest of the file as we go.
        with open(destfile, 'wb') as dfp:
            # addinfourl is not a context manager on 2.x
            # so we have to use try/finally
            sfp = self.send_request(Request(url))
            try:
                headers = sfp.info()
                blocksize = 8192
                size = -1
                read = 0
                blocknum = 0
                if "content-length" in headers:
                    size = int(headers["Content-Length"])
                if reporthook:
                    reporthook(blocknum, blocksize, size)
                while True:
                    block = sfp.read(blocksize)
                    if not block:
                        break
                    read += len(block)
                    dfp.write(block)
                    if digester:
                        digester.update(block)
                    blocknum += 1
                    if reporthook:
                        reporthook(blocknum, blocksize, size)
            finally:
                sfp.close()

        # check that we got the whole file, if we can
        if size >= 0 and read < size:
            raise DistlibException(
                'retrieval incomplete: got only %d out of %d bytes'
                % (read, size))
        # if we have a digest, it must match.
        if digester:
            actual = digester.hexdigest()
            if digest != actual:
                raise DistlibException('%s digest mismatch for %s: expected '
                                       '%s, got %s' % (hasher, destfile,
                                                       digest, actual))
            logger.debug('Digest verified: %s', digest)

    def send_request(self, req):
        """
        Send a standard library :class:`Request` to PyPI and return its
        response.

        :param req: The request to send.
        :return: The HTTP response from PyPI (a standard library HTTPResponse).
        """
        handlers = []
        if self.password_handler:
            handlers.append(self.password_handler)
        if self.ssl_verifier:
            handlers.append(self.ssl_verifier)
        opener = build_opener(*handlers)
        return opener.open(req)

    def encode_request(self, fields, files):
        """
        Encode fields and files for posting to an HTTP server.

        :param fields: The fields to send as a list of (fieldname, value)
                       tuples.
        :param files: The files to send as a list of (fieldname, filename,
                      file_bytes) tuple.
        """
        # Adapted from packaging, which in turn was adapted from
        # http://code.activestate.com/recipes/146306

        parts = []
        boundary = self.boundary
        for k, values in fields:
            if not isinstance(values, (list, tuple)):
                values = [values]

            for v in values:
                parts.extend((
                    b'--' + boundary,
                    ('Content-Disposition: form-data; name="%s"' %
                     k).encode('utf-8'),
                    b'',
                    v.encode('utf-8')))
        for key, filename, value in files:
            parts.extend((
                b'--' + boundary,
                ('Content-Disposition: form-data; name="%s"; filename="%s"' %
                 (key, filename)).encode('utf-8'),
                b'',
                value))

        parts.extend((b'--' + boundary + b'--', b''))

        body = b'\r\n'.join(parts)
        ct = b'multipart/form-data; boundary=' + boundary
        headers = {
            'Content-type': ct,
            'Content-length': str(len(body))
        }
        return Request(self.url, body, headers)

    def search(self, terms, operator=None):  # pragma: no cover
        if isinstance(terms, string_types):
            terms = {'name': terms}
        rpc_proxy = ServerProxy(self.url, timeout=3.0)
        try:
            return rpc_proxy.search(terms, operator or 'and')
        finally:
            rpc_proxy('close')()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\security.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Security
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network


class CertificateId(int):
    '''
    An internal certificate ID value.
    '''
    def to_json(self) -> int:
        return self

    @classmethod
    def from_json(cls, json: int) -> CertificateId:
        return cls(json)

    def __repr__(self):
        return 'CertificateId({})'.format(super().__repr__())


class MixedContentType(enum.Enum):
    '''
    A description of mixed content (HTTP resources on HTTPS pages), as defined by
    https://www.w3.org/TR/mixed-content/#categories
    '''
    BLOCKABLE = "blockable"
    OPTIONALLY_BLOCKABLE = "optionally-blockable"
    NONE = "none"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class SecurityState(enum.Enum):
    '''
    The security level of a page or resource.
    '''
    UNKNOWN = "unknown"
    NEUTRAL = "neutral"
    INSECURE = "insecure"
    SECURE = "secure"
    INFO = "info"
    INSECURE_BROKEN = "insecure-broken"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class CertificateSecurityState:
    '''
    Details about the security state of the page certificate.
    '''
    #: Protocol name (e.g. "TLS 1.2" or "QUIC").
    protocol: str

    #: Key Exchange used by the connection, or the empty string if not applicable.
    key_exchange: str

    #: Cipher name.
    cipher: str

    #: Page certificate.
    certificate: typing.List[str]

    #: Certificate subject name.
    subject_name: str

    #: Name of the issuing CA.
    issuer: str

    #: Certificate valid from date.
    valid_from: network.TimeSinceEpoch

    #: Certificate valid to (expiration) date
    valid_to: network.TimeSinceEpoch

    #: True if the certificate uses a weak signature algorithm.
    certificate_has_weak_signature: bool

    #: True if the certificate has a SHA1 signature in the chain.
    certificate_has_sha1_signature: bool

    #: True if modern SSL
    modern_ssl: bool

    #: True if the connection is using an obsolete SSL protocol.
    obsolete_ssl_protocol: bool

    #: True if the connection is using an obsolete SSL key exchange.
    obsolete_ssl_key_exchange: bool

    #: True if the connection is using an obsolete SSL cipher.
    obsolete_ssl_cipher: bool

    #: True if the connection is using an obsolete SSL signature.
    obsolete_ssl_signature: bool

    #: (EC)DH group used by the connection, if applicable.
    key_exchange_group: typing.Optional[str] = None

    #: TLS MAC. Note that AEAD ciphers do not have separate MACs.
    mac: typing.Optional[str] = None

    #: The highest priority network error code, if the certificate has an error.
    certificate_network_error: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['protocol'] = self.protocol
        json['keyExchange'] = self.key_exchange
        json['cipher'] = self.cipher
        json['certificate'] = [i for i in self.certificate]
        json['subjectName'] = self.subject_name
        json['issuer'] = self.issuer
        json['validFrom'] = self.valid_from.to_json()
        json['validTo'] = self.valid_to.to_json()
        json['certificateHasWeakSignature'] = self.certificate_has_weak_signature
        json['certificateHasSha1Signature'] = self.certificate_has_sha1_signature
        json['modernSSL'] = self.modern_ssl
        json['obsoleteSslProtocol'] = self.obsolete_ssl_protocol
        json['obsoleteSslKeyExchange'] = self.obsolete_ssl_key_exchange
        json['obsoleteSslCipher'] = self.obsolete_ssl_cipher
        json['obsoleteSslSignature'] = self.obsolete_ssl_signature
        if self.key_exchange_group is not None:
            json['keyExchangeGroup'] = self.key_exchange_group
        if self.mac is not None:
            json['mac'] = self.mac
        if self.certificate_network_error is not None:
            json['certificateNetworkError'] = self.certificate_network_error
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            protocol=str(json['protocol']),
            key_exchange=str(json['keyExchange']),
            cipher=str(json['cipher']),
            certificate=[str(i) for i in json['certificate']],
            subject_name=str(json['subjectName']),
            issuer=str(json['issuer']),
            valid_from=network.TimeSinceEpoch.from_json(json['validFrom']),
            valid_to=network.TimeSinceEpoch.from_json(json['validTo']),
            certificate_has_weak_signature=bool(json['certificateHasWeakSignature']),
            certificate_has_sha1_signature=bool(json['certificateHasSha1Signature']),
            modern_ssl=bool(json['modernSSL']),
            obsolete_ssl_protocol=bool(json['obsoleteSslProtocol']),
            obsolete_ssl_key_exchange=bool(json['obsoleteSslKeyExchange']),
            obsolete_ssl_cipher=bool(json['obsoleteSslCipher']),
            obsolete_ssl_signature=bool(json['obsoleteSslSignature']),
            key_exchange_group=str(json['keyExchangeGroup']) if 'keyExchangeGroup' in json else None,
            mac=str(json['mac']) if 'mac' in json else None,
            certificate_network_error=str(json['certificateNetworkError']) if 'certificateNetworkError' in json else None,
        )


class SafetyTipStatus(enum.Enum):
    BAD_REPUTATION = "badReputation"
    LOOKALIKE = "lookalike"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class SafetyTipInfo:
    #: Describes whether the page triggers any safety tips or reputation warnings. Default is unknown.
    safety_tip_status: SafetyTipStatus

    #: The URL the safety tip suggested ("Did you mean?"). Only filled in for lookalike matches.
    safe_url: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['safetyTipStatus'] = self.safety_tip_status.to_json()
        if self.safe_url is not None:
            json['safeUrl'] = self.safe_url
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            safety_tip_status=SafetyTipStatus.from_json(json['safetyTipStatus']),
            safe_url=str(json['safeUrl']) if 'safeUrl' in json else None,
        )


@dataclass
class VisibleSecurityState:
    '''
    Security state information about the page.
    '''
    #: The security level of the page.
    security_state: SecurityState

    #: Array of security state issues ids.
    security_state_issue_ids: typing.List[str]

    #: Security state details about the page certificate.
    certificate_security_state: typing.Optional[CertificateSecurityState] = None

    #: The type of Safety Tip triggered on the page. Note that this field will be set even if the Safety Tip UI was not actually shown.
    safety_tip_info: typing.Optional[SafetyTipInfo] = None

    def to_json(self):
        json = dict()
        json['securityState'] = self.security_state.to_json()
        json['securityStateIssueIds'] = [i for i in self.security_state_issue_ids]
        if self.certificate_security_state is not None:
            json['certificateSecurityState'] = self.certificate_security_state.to_json()
        if self.safety_tip_info is not None:
            json['safetyTipInfo'] = self.safety_tip_info.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            security_state=SecurityState.from_json(json['securityState']),
            security_state_issue_ids=[str(i) for i in json['securityStateIssueIds']],
            certificate_security_state=CertificateSecurityState.from_json(json['certificateSecurityState']) if 'certificateSecurityState' in json else None,
            safety_tip_info=SafetyTipInfo.from_json(json['safetyTipInfo']) if 'safetyTipInfo' in json else None,
        )


@dataclass
class SecurityStateExplanation:
    '''
    An explanation of an factor contributing to the security state.
    '''
    #: Security state representing the severity of the factor being explained.
    security_state: SecurityState

    #: Title describing the type of factor.
    title: str

    #: Short phrase describing the type of factor.
    summary: str

    #: Full text explanation of the factor.
    description: str

    #: The type of mixed content described by the explanation.
    mixed_content_type: MixedContentType

    #: Page certificate.
    certificate: typing.List[str]

    #: Recommendations to fix any issues.
    recommendations: typing.Optional[typing.List[str]] = None

    def to_json(self):
        json = dict()
        json['securityState'] = self.security_state.to_json()
        json['title'] = self.title
        json['summary'] = self.summary
        json['description'] = self.description
        json['mixedContentType'] = self.mixed_content_type.to_json()
        json['certificate'] = [i for i in self.certificate]
        if self.recommendations is not None:
            json['recommendations'] = [i for i in self.recommendations]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            security_state=SecurityState.from_json(json['securityState']),
            title=str(json['title']),
            summary=str(json['summary']),
            description=str(json['description']),
            mixed_content_type=MixedContentType.from_json(json['mixedContentType']),
            certificate=[str(i) for i in json['certificate']],
            recommendations=[str(i) for i in json['recommendations']] if 'recommendations' in json else None,
        )


@dataclass
class InsecureContentStatus:
    '''
    Information about insecure content on the page.
    '''
    #: Always false.
    ran_mixed_content: bool

    #: Always false.
    displayed_mixed_content: bool

    #: Always false.
    contained_mixed_form: bool

    #: Always false.
    ran_content_with_cert_errors: bool

    #: Always false.
    displayed_content_with_cert_errors: bool

    #: Always set to unknown.
    ran_insecure_content_style: SecurityState

    #: Always set to unknown.
    displayed_insecure_content_style: SecurityState

    def to_json(self):
        json = dict()
        json['ranMixedContent'] = self.ran_mixed_content
        json['displayedMixedContent'] = self.displayed_mixed_content
        json['containedMixedForm'] = self.contained_mixed_form
        json['ranContentWithCertErrors'] = self.ran_content_with_cert_errors
        json['displayedContentWithCertErrors'] = self.displayed_content_with_cert_errors
        json['ranInsecureContentStyle'] = self.ran_insecure_content_style.to_json()
        json['displayedInsecureContentStyle'] = self.displayed_insecure_content_style.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            ran_mixed_content=bool(json['ranMixedContent']),
            displayed_mixed_content=bool(json['displayedMixedContent']),
            contained_mixed_form=bool(json['containedMixedForm']),
            ran_content_with_cert_errors=bool(json['ranContentWithCertErrors']),
            displayed_content_with_cert_errors=bool(json['displayedContentWithCertErrors']),
            ran_insecure_content_style=SecurityState.from_json(json['ranInsecureContentStyle']),
            displayed_insecure_content_style=SecurityState.from_json(json['displayedInsecureContentStyle']),
        )


class CertificateErrorAction(enum.Enum):
    '''
    The action to take when a certificate error occurs. continue will continue processing the
    request and cancel will cancel the request.
    '''
    CONTINUE = "continue"
    CANCEL = "cancel"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables tracking security state changes.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Security.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables tracking security state changes.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Security.enable',
    }
    json = yield cmd_dict


def set_ignore_certificate_errors(
        ignore: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enable/disable whether all certificate errors should be ignored.

    :param ignore: If true, all certificate errors will be ignored.
    '''
    params: T_JSON_DICT = dict()
    params['ignore'] = ignore
    cmd_dict: T_JSON_DICT = {
        'method': 'Security.setIgnoreCertificateErrors',
        'params': params,
    }
    json = yield cmd_dict


def handle_certificate_error(
        event_id: int,
        action: CertificateErrorAction
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Handles a certificate error that fired a certificateError event.

    :param event_id: The ID of the event.
    :param action: The action to take on the certificate error.
    '''
    params: T_JSON_DICT = dict()
    params['eventId'] = event_id
    params['action'] = action.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Security.handleCertificateError',
        'params': params,
    }
    json = yield cmd_dict


def set_override_certificate_errors(
        override: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enable/disable overriding certificate errors. If enabled, all certificate error events need to
    be handled by the DevTools client and should be answered with ``handleCertificateError`` commands.

    :param override: If true, certificate errors will be overridden.
    '''
    params: T_JSON_DICT = dict()
    params['override'] = override
    cmd_dict: T_JSON_DICT = {
        'method': 'Security.setOverrideCertificateErrors',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Security.certificateError')
@dataclass
class CertificateError:
    '''
    There is a certificate error. If overriding certificate errors is enabled, then it should be
    handled with the ``handleCertificateError`` command. Note: this event does not fire if the
    certificate error has been allowed internally. Only one client per target should override
    certificate errors at the same time.
    '''
    #: The ID of the event.
    event_id: int
    #: The type of the error.
    error_type: str
    #: The url that was requested.
    request_url: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CertificateError:
        return cls(
            event_id=int(json['eventId']),
            error_type=str(json['errorType']),
            request_url=str(json['requestURL'])
        )


@event_class('Security.visibleSecurityStateChanged')
@dataclass
class VisibleSecurityStateChanged:
    '''
    **EXPERIMENTAL**

    The security state of the page changed.
    '''
    #: Security state information about the page.
    visible_security_state: VisibleSecurityState

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> VisibleSecurityStateChanged:
        return cls(
            visible_security_state=VisibleSecurityState.from_json(json['visibleSecurityState'])
        )


@event_class('Security.securityStateChanged')
@dataclass
class SecurityStateChanged:
    '''
    The security state of the page changed. No longer being sent.
    '''
    #: Security state.
    security_state: SecurityState
    #: True if the page was loaded over cryptographic transport such as HTTPS.
    scheme_is_cryptographic: bool
    #: Previously a list of explanations for the security state. Now always
    #: empty.
    explanations: typing.List[SecurityStateExplanation]
    #: Information about insecure content on the page.
    insecure_content_status: InsecureContentStatus
    #: Overrides user-visible description of the state. Always omitted.
    summary: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> SecurityStateChanged:
        return cls(
            security_state=SecurityState.from_json(json['securityState']),
            scheme_is_cryptographic=bool(json['schemeIsCryptographic']),
            explanations=[SecurityStateExplanation.from_json(i) for i in json['explanations']],
            insecure_content_status=InsecureContentStatus.from_json(json['insecureContentStatus']),
            summary=str(json['summary']) if 'summary' in json else None
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\security.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Security
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network


class CertificateId(int):
    '''
    An internal certificate ID value.
    '''
    def to_json(self) -> int:
        return self

    @classmethod
    def from_json(cls, json: int) -> CertificateId:
        return cls(json)

    def __repr__(self):
        return 'CertificateId({})'.format(super().__repr__())


class MixedContentType(enum.Enum):
    '''
    A description of mixed content (HTTP resources on HTTPS pages), as defined by
    https://www.w3.org/TR/mixed-content/#categories
    '''
    BLOCKABLE = "blockable"
    OPTIONALLY_BLOCKABLE = "optionally-blockable"
    NONE = "none"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class SecurityState(enum.Enum):
    '''
    The security level of a page or resource.
    '''
    UNKNOWN = "unknown"
    NEUTRAL = "neutral"
    INSECURE = "insecure"
    SECURE = "secure"
    INFO = "info"
    INSECURE_BROKEN = "insecure-broken"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class CertificateSecurityState:
    '''
    Details about the security state of the page certificate.
    '''
    #: Protocol name (e.g. "TLS 1.2" or "QUIC").
    protocol: str

    #: Key Exchange used by the connection, or the empty string if not applicable.
    key_exchange: str

    #: Cipher name.
    cipher: str

    #: Page certificate.
    certificate: typing.List[str]

    #: Certificate subject name.
    subject_name: str

    #: Name of the issuing CA.
    issuer: str

    #: Certificate valid from date.
    valid_from: network.TimeSinceEpoch

    #: Certificate valid to (expiration) date
    valid_to: network.TimeSinceEpoch

    #: True if the certificate uses a weak signature algorithm.
    certificate_has_weak_signature: bool

    #: True if the certificate has a SHA1 signature in the chain.
    certificate_has_sha1_signature: bool

    #: True if modern SSL
    modern_ssl: bool

    #: True if the connection is using an obsolete SSL protocol.
    obsolete_ssl_protocol: bool

    #: True if the connection is using an obsolete SSL key exchange.
    obsolete_ssl_key_exchange: bool

    #: True if the connection is using an obsolete SSL cipher.
    obsolete_ssl_cipher: bool

    #: True if the connection is using an obsolete SSL signature.
    obsolete_ssl_signature: bool

    #: (EC)DH group used by the connection, if applicable.
    key_exchange_group: typing.Optional[str] = None

    #: TLS MAC. Note that AEAD ciphers do not have separate MACs.
    mac: typing.Optional[str] = None

    #: The highest priority network error code, if the certificate has an error.
    certificate_network_error: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['protocol'] = self.protocol
        json['keyExchange'] = self.key_exchange
        json['cipher'] = self.cipher
        json['certificate'] = [i for i in self.certificate]
        json['subjectName'] = self.subject_name
        json['issuer'] = self.issuer
        json['validFrom'] = self.valid_from.to_json()
        json['validTo'] = self.valid_to.to_json()
        json['certificateHasWeakSignature'] = self.certificate_has_weak_signature
        json['certificateHasSha1Signature'] = self.certificate_has_sha1_signature
        json['modernSSL'] = self.modern_ssl
        json['obsoleteSslProtocol'] = self.obsolete_ssl_protocol
        json['obsoleteSslKeyExchange'] = self.obsolete_ssl_key_exchange
        json['obsoleteSslCipher'] = self.obsolete_ssl_cipher
        json['obsoleteSslSignature'] = self.obsolete_ssl_signature
        if self.key_exchange_group is not None:
            json['keyExchangeGroup'] = self.key_exchange_group
        if self.mac is not None:
            json['mac'] = self.mac
        if self.certificate_network_error is not None:
            json['certificateNetworkError'] = self.certificate_network_error
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            protocol=str(json['protocol']),
            key_exchange=str(json['keyExchange']),
            cipher=str(json['cipher']),
            certificate=[str(i) for i in json['certificate']],
            subject_name=str(json['subjectName']),
            issuer=str(json['issuer']),
            valid_from=network.TimeSinceEpoch.from_json(json['validFrom']),
            valid_to=network.TimeSinceEpoch.from_json(json['validTo']),
            certificate_has_weak_signature=bool(json['certificateHasWeakSignature']),
            certificate_has_sha1_signature=bool(json['certificateHasSha1Signature']),
            modern_ssl=bool(json['modernSSL']),
            obsolete_ssl_protocol=bool(json['obsoleteSslProtocol']),
            obsolete_ssl_key_exchange=bool(json['obsoleteSslKeyExchange']),
            obsolete_ssl_cipher=bool(json['obsoleteSslCipher']),
            obsolete_ssl_signature=bool(json['obsoleteSslSignature']),
            key_exchange_group=str(json['keyExchangeGroup']) if 'keyExchangeGroup' in json else None,
            mac=str(json['mac']) if 'mac' in json else None,
            certificate_network_error=str(json['certificateNetworkError']) if 'certificateNetworkError' in json else None,
        )


class SafetyTipStatus(enum.Enum):
    BAD_REPUTATION = "badReputation"
    LOOKALIKE = "lookalike"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class SafetyTipInfo:
    #: Describes whether the page triggers any safety tips or reputation warnings. Default is unknown.
    safety_tip_status: SafetyTipStatus

    #: The URL the safety tip suggested ("Did you mean?"). Only filled in for lookalike matches.
    safe_url: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['safetyTipStatus'] = self.safety_tip_status.to_json()
        if self.safe_url is not None:
            json['safeUrl'] = self.safe_url
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            safety_tip_status=SafetyTipStatus.from_json(json['safetyTipStatus']),
            safe_url=str(json['safeUrl']) if 'safeUrl' in json else None,
        )


@dataclass
class VisibleSecurityState:
    '''
    Security state information about the page.
    '''
    #: The security level of the page.
    security_state: SecurityState

    #: Array of security state issues ids.
    security_state_issue_ids: typing.List[str]

    #: Security state details about the page certificate.
    certificate_security_state: typing.Optional[CertificateSecurityState] = None

    #: The type of Safety Tip triggered on the page. Note that this field will be set even if the Safety Tip UI was not actually shown.
    safety_tip_info: typing.Optional[SafetyTipInfo] = None

    def to_json(self):
        json = dict()
        json['securityState'] = self.security_state.to_json()
        json['securityStateIssueIds'] = [i for i in self.security_state_issue_ids]
        if self.certificate_security_state is not None:
            json['certificateSecurityState'] = self.certificate_security_state.to_json()
        if self.safety_tip_info is not None:
            json['safetyTipInfo'] = self.safety_tip_info.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            security_state=SecurityState.from_json(json['securityState']),
            security_state_issue_ids=[str(i) for i in json['securityStateIssueIds']],
            certificate_security_state=CertificateSecurityState.from_json(json['certificateSecurityState']) if 'certificateSecurityState' in json else None,
            safety_tip_info=SafetyTipInfo.from_json(json['safetyTipInfo']) if 'safetyTipInfo' in json else None,
        )


@dataclass
class SecurityStateExplanation:
    '''
    An explanation of an factor contributing to the security state.
    '''
    #: Security state representing the severity of the factor being explained.
    security_state: SecurityState

    #: Title describing the type of factor.
    title: str

    #: Short phrase describing the type of factor.
    summary: str

    #: Full text explanation of the factor.
    description: str

    #: The type of mixed content described by the explanation.
    mixed_content_type: MixedContentType

    #: Page certificate.
    certificate: typing.List[str]

    #: Recommendations to fix any issues.
    recommendations: typing.Optional[typing.List[str]] = None

    def to_json(self):
        json = dict()
        json['securityState'] = self.security_state.to_json()
        json['title'] = self.title
        json['summary'] = self.summary
        json['description'] = self.description
        json['mixedContentType'] = self.mixed_content_type.to_json()
        json['certificate'] = [i for i in self.certificate]
        if self.recommendations is not None:
            json['recommendations'] = [i for i in self.recommendations]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            security_state=SecurityState.from_json(json['securityState']),
            title=str(json['title']),
            summary=str(json['summary']),
            description=str(json['description']),
            mixed_content_type=MixedContentType.from_json(json['mixedContentType']),
            certificate=[str(i) for i in json['certificate']],
            recommendations=[str(i) for i in json['recommendations']] if 'recommendations' in json else None,
        )


@dataclass
class InsecureContentStatus:
    '''
    Information about insecure content on the page.
    '''
    #: Always false.
    ran_mixed_content: bool

    #: Always false.
    displayed_mixed_content: bool

    #: Always false.
    contained_mixed_form: bool

    #: Always false.
    ran_content_with_cert_errors: bool

    #: Always false.
    displayed_content_with_cert_errors: bool

    #: Always set to unknown.
    ran_insecure_content_style: SecurityState

    #: Always set to unknown.
    displayed_insecure_content_style: SecurityState

    def to_json(self):
        json = dict()
        json['ranMixedContent'] = self.ran_mixed_content
        json['displayedMixedContent'] = self.displayed_mixed_content
        json['containedMixedForm'] = self.contained_mixed_form
        json['ranContentWithCertErrors'] = self.ran_content_with_cert_errors
        json['displayedContentWithCertErrors'] = self.displayed_content_with_cert_errors
        json['ranInsecureContentStyle'] = self.ran_insecure_content_style.to_json()
        json['displayedInsecureContentStyle'] = self.displayed_insecure_content_style.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            ran_mixed_content=bool(json['ranMixedContent']),
            displayed_mixed_content=bool(json['displayedMixedContent']),
            contained_mixed_form=bool(json['containedMixedForm']),
            ran_content_with_cert_errors=bool(json['ranContentWithCertErrors']),
            displayed_content_with_cert_errors=bool(json['displayedContentWithCertErrors']),
            ran_insecure_content_style=SecurityState.from_json(json['ranInsecureContentStyle']),
            displayed_insecure_content_style=SecurityState.from_json(json['displayedInsecureContentStyle']),
        )


class CertificateErrorAction(enum.Enum):
    '''
    The action to take when a certificate error occurs. continue will continue processing the
    request and cancel will cancel the request.
    '''
    CONTINUE = "continue"
    CANCEL = "cancel"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables tracking security state changes.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Security.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables tracking security state changes.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Security.enable',
    }
    json = yield cmd_dict


def set_ignore_certificate_errors(
        ignore: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enable/disable whether all certificate errors should be ignored.

    :param ignore: If true, all certificate errors will be ignored.
    '''
    params: T_JSON_DICT = dict()
    params['ignore'] = ignore
    cmd_dict: T_JSON_DICT = {
        'method': 'Security.setIgnoreCertificateErrors',
        'params': params,
    }
    json = yield cmd_dict


def handle_certificate_error(
        event_id: int,
        action: CertificateErrorAction
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Handles a certificate error that fired a certificateError event.

    :param event_id: The ID of the event.
    :param action: The action to take on the certificate error.
    '''
    params: T_JSON_DICT = dict()
    params['eventId'] = event_id
    params['action'] = action.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Security.handleCertificateError',
        'params': params,
    }
    json = yield cmd_dict


def set_override_certificate_errors(
        override: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enable/disable overriding certificate errors. If enabled, all certificate error events need to
    be handled by the DevTools client and should be answered with ``handleCertificateError`` commands.

    :param override: If true, certificate errors will be overridden.
    '''
    params: T_JSON_DICT = dict()
    params['override'] = override
    cmd_dict: T_JSON_DICT = {
        'method': 'Security.setOverrideCertificateErrors',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Security.certificateError')
@dataclass
class CertificateError:
    '''
    There is a certificate error. If overriding certificate errors is enabled, then it should be
    handled with the ``handleCertificateError`` command. Note: this event does not fire if the
    certificate error has been allowed internally. Only one client per target should override
    certificate errors at the same time.
    '''
    #: The ID of the event.
    event_id: int
    #: The type of the error.
    error_type: str
    #: The url that was requested.
    request_url: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CertificateError:
        return cls(
            event_id=int(json['eventId']),
            error_type=str(json['errorType']),
            request_url=str(json['requestURL'])
        )


@event_class('Security.visibleSecurityStateChanged')
@dataclass
class VisibleSecurityStateChanged:
    '''
    **EXPERIMENTAL**

    The security state of the page changed.
    '''
    #: Security state information about the page.
    visible_security_state: VisibleSecurityState

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> VisibleSecurityStateChanged:
        return cls(
            visible_security_state=VisibleSecurityState.from_json(json['visibleSecurityState'])
        )


@event_class('Security.securityStateChanged')
@dataclass
class SecurityStateChanged:
    '''
    The security state of the page changed. No longer being sent.
    '''
    #: Security state.
    security_state: SecurityState
    #: True if the page was loaded over cryptographic transport such as HTTPS.
    scheme_is_cryptographic: bool
    #: Previously a list of explanations for the security state. Now always
    #: empty.
    explanations: typing.List[SecurityStateExplanation]
    #: Information about insecure content on the page.
    insecure_content_status: InsecureContentStatus
    #: Overrides user-visible description of the state. Always omitted.
    summary: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> SecurityStateChanged:
        return cls(
            security_state=SecurityState.from_json(json['securityState']),
            scheme_is_cryptographic=bool(json['schemeIsCryptographic']),
            explanations=[SecurityStateExplanation.from_json(i) for i in json['explanations']],
            insecure_content_status=InsecureContentStatus.from_json(json['insecureContentStatus']),
            summary=str(json['summary']) if 'summary' in json else None
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\security.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Security
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network


class CertificateId(int):
    '''
    An internal certificate ID value.
    '''
    def to_json(self) -> int:
        return self

    @classmethod
    def from_json(cls, json: int) -> CertificateId:
        return cls(json)

    def __repr__(self):
        return 'CertificateId({})'.format(super().__repr__())


class MixedContentType(enum.Enum):
    '''
    A description of mixed content (HTTP resources on HTTPS pages), as defined by
    https://www.w3.org/TR/mixed-content/#categories
    '''
    BLOCKABLE = "blockable"
    OPTIONALLY_BLOCKABLE = "optionally-blockable"
    NONE = "none"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class SecurityState(enum.Enum):
    '''
    The security level of a page or resource.
    '''
    UNKNOWN = "unknown"
    NEUTRAL = "neutral"
    INSECURE = "insecure"
    SECURE = "secure"
    INFO = "info"
    INSECURE_BROKEN = "insecure-broken"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class CertificateSecurityState:
    '''
    Details about the security state of the page certificate.
    '''
    #: Protocol name (e.g. "TLS 1.2" or "QUIC").
    protocol: str

    #: Key Exchange used by the connection, or the empty string if not applicable.
    key_exchange: str

    #: Cipher name.
    cipher: str

    #: Page certificate.
    certificate: typing.List[str]

    #: Certificate subject name.
    subject_name: str

    #: Name of the issuing CA.
    issuer: str

    #: Certificate valid from date.
    valid_from: network.TimeSinceEpoch

    #: Certificate valid to (expiration) date
    valid_to: network.TimeSinceEpoch

    #: True if the certificate uses a weak signature algorithm.
    certificate_has_weak_signature: bool

    #: True if the certificate has a SHA1 signature in the chain.
    certificate_has_sha1_signature: bool

    #: True if modern SSL
    modern_ssl: bool

    #: True if the connection is using an obsolete SSL protocol.
    obsolete_ssl_protocol: bool

    #: True if the connection is using an obsolete SSL key exchange.
    obsolete_ssl_key_exchange: bool

    #: True if the connection is using an obsolete SSL cipher.
    obsolete_ssl_cipher: bool

    #: True if the connection is using an obsolete SSL signature.
    obsolete_ssl_signature: bool

    #: (EC)DH group used by the connection, if applicable.
    key_exchange_group: typing.Optional[str] = None

    #: TLS MAC. Note that AEAD ciphers do not have separate MACs.
    mac: typing.Optional[str] = None

    #: The highest priority network error code, if the certificate has an error.
    certificate_network_error: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['protocol'] = self.protocol
        json['keyExchange'] = self.key_exchange
        json['cipher'] = self.cipher
        json['certificate'] = [i for i in self.certificate]
        json['subjectName'] = self.subject_name
        json['issuer'] = self.issuer
        json['validFrom'] = self.valid_from.to_json()
        json['validTo'] = self.valid_to.to_json()
        json['certificateHasWeakSignature'] = self.certificate_has_weak_signature
        json['certificateHasSha1Signature'] = self.certificate_has_sha1_signature
        json['modernSSL'] = self.modern_ssl
        json['obsoleteSslProtocol'] = self.obsolete_ssl_protocol
        json['obsoleteSslKeyExchange'] = self.obsolete_ssl_key_exchange
        json['obsoleteSslCipher'] = self.obsolete_ssl_cipher
        json['obsoleteSslSignature'] = self.obsolete_ssl_signature
        if self.key_exchange_group is not None:
            json['keyExchangeGroup'] = self.key_exchange_group
        if self.mac is not None:
            json['mac'] = self.mac
        if self.certificate_network_error is not None:
            json['certificateNetworkError'] = self.certificate_network_error
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            protocol=str(json['protocol']),
            key_exchange=str(json['keyExchange']),
            cipher=str(json['cipher']),
            certificate=[str(i) for i in json['certificate']],
            subject_name=str(json['subjectName']),
            issuer=str(json['issuer']),
            valid_from=network.TimeSinceEpoch.from_json(json['validFrom']),
            valid_to=network.TimeSinceEpoch.from_json(json['validTo']),
            certificate_has_weak_signature=bool(json['certificateHasWeakSignature']),
            certificate_has_sha1_signature=bool(json['certificateHasSha1Signature']),
            modern_ssl=bool(json['modernSSL']),
            obsolete_ssl_protocol=bool(json['obsoleteSslProtocol']),
            obsolete_ssl_key_exchange=bool(json['obsoleteSslKeyExchange']),
            obsolete_ssl_cipher=bool(json['obsoleteSslCipher']),
            obsolete_ssl_signature=bool(json['obsoleteSslSignature']),
            key_exchange_group=str(json['keyExchangeGroup']) if 'keyExchangeGroup' in json else None,
            mac=str(json['mac']) if 'mac' in json else None,
            certificate_network_error=str(json['certificateNetworkError']) if 'certificateNetworkError' in json else None,
        )


class SafetyTipStatus(enum.Enum):
    BAD_REPUTATION = "badReputation"
    LOOKALIKE = "lookalike"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class SafetyTipInfo:
    #: Describes whether the page triggers any safety tips or reputation warnings. Default is unknown.
    safety_tip_status: SafetyTipStatus

    #: The URL the safety tip suggested ("Did you mean?"). Only filled in for lookalike matches.
    safe_url: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['safetyTipStatus'] = self.safety_tip_status.to_json()
        if self.safe_url is not None:
            json['safeUrl'] = self.safe_url
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            safety_tip_status=SafetyTipStatus.from_json(json['safetyTipStatus']),
            safe_url=str(json['safeUrl']) if 'safeUrl' in json else None,
        )


@dataclass
class VisibleSecurityState:
    '''
    Security state information about the page.
    '''
    #: The security level of the page.
    security_state: SecurityState

    #: Array of security state issues ids.
    security_state_issue_ids: typing.List[str]

    #: Security state details about the page certificate.
    certificate_security_state: typing.Optional[CertificateSecurityState] = None

    #: The type of Safety Tip triggered on the page. Note that this field will be set even if the Safety Tip UI was not actually shown.
    safety_tip_info: typing.Optional[SafetyTipInfo] = None

    def to_json(self):
        json = dict()
        json['securityState'] = self.security_state.to_json()
        json['securityStateIssueIds'] = [i for i in self.security_state_issue_ids]
        if self.certificate_security_state is not None:
            json['certificateSecurityState'] = self.certificate_security_state.to_json()
        if self.safety_tip_info is not None:
            json['safetyTipInfo'] = self.safety_tip_info.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            security_state=SecurityState.from_json(json['securityState']),
            security_state_issue_ids=[str(i) for i in json['securityStateIssueIds']],
            certificate_security_state=CertificateSecurityState.from_json(json['certificateSecurityState']) if 'certificateSecurityState' in json else None,
            safety_tip_info=SafetyTipInfo.from_json(json['safetyTipInfo']) if 'safetyTipInfo' in json else None,
        )


@dataclass
class SecurityStateExplanation:
    '''
    An explanation of an factor contributing to the security state.
    '''
    #: Security state representing the severity of the factor being explained.
    security_state: SecurityState

    #: Title describing the type of factor.
    title: str

    #: Short phrase describing the type of factor.
    summary: str

    #: Full text explanation of the factor.
    description: str

    #: The type of mixed content described by the explanation.
    mixed_content_type: MixedContentType

    #: Page certificate.
    certificate: typing.List[str]

    #: Recommendations to fix any issues.
    recommendations: typing.Optional[typing.List[str]] = None

    def to_json(self):
        json = dict()
        json['securityState'] = self.security_state.to_json()
        json['title'] = self.title
        json['summary'] = self.summary
        json['description'] = self.description
        json['mixedContentType'] = self.mixed_content_type.to_json()
        json['certificate'] = [i for i in self.certificate]
        if self.recommendations is not None:
            json['recommendations'] = [i for i in self.recommendations]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            security_state=SecurityState.from_json(json['securityState']),
            title=str(json['title']),
            summary=str(json['summary']),
            description=str(json['description']),
            mixed_content_type=MixedContentType.from_json(json['mixedContentType']),
            certificate=[str(i) for i in json['certificate']],
            recommendations=[str(i) for i in json['recommendations']] if 'recommendations' in json else None,
        )


@dataclass
class InsecureContentStatus:
    '''
    Information about insecure content on the page.
    '''
    #: Always false.
    ran_mixed_content: bool

    #: Always false.
    displayed_mixed_content: bool

    #: Always false.
    contained_mixed_form: bool

    #: Always false.
    ran_content_with_cert_errors: bool

    #: Always false.
    displayed_content_with_cert_errors: bool

    #: Always set to unknown.
    ran_insecure_content_style: SecurityState

    #: Always set to unknown.
    displayed_insecure_content_style: SecurityState

    def to_json(self):
        json = dict()
        json['ranMixedContent'] = self.ran_mixed_content
        json['displayedMixedContent'] = self.displayed_mixed_content
        json['containedMixedForm'] = self.contained_mixed_form
        json['ranContentWithCertErrors'] = self.ran_content_with_cert_errors
        json['displayedContentWithCertErrors'] = self.displayed_content_with_cert_errors
        json['ranInsecureContentStyle'] = self.ran_insecure_content_style.to_json()
        json['displayedInsecureContentStyle'] = self.displayed_insecure_content_style.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            ran_mixed_content=bool(json['ranMixedContent']),
            displayed_mixed_content=bool(json['displayedMixedContent']),
            contained_mixed_form=bool(json['containedMixedForm']),
            ran_content_with_cert_errors=bool(json['ranContentWithCertErrors']),
            displayed_content_with_cert_errors=bool(json['displayedContentWithCertErrors']),
            ran_insecure_content_style=SecurityState.from_json(json['ranInsecureContentStyle']),
            displayed_insecure_content_style=SecurityState.from_json(json['displayedInsecureContentStyle']),
        )


class CertificateErrorAction(enum.Enum):
    '''
    The action to take when a certificate error occurs. continue will continue processing the
    request and cancel will cancel the request.
    '''
    CONTINUE = "continue"
    CANCEL = "cancel"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables tracking security state changes.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Security.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables tracking security state changes.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Security.enable',
    }
    json = yield cmd_dict


def set_ignore_certificate_errors(
        ignore: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enable/disable whether all certificate errors should be ignored.

    :param ignore: If true, all certificate errors will be ignored.
    '''
    params: T_JSON_DICT = dict()
    params['ignore'] = ignore
    cmd_dict: T_JSON_DICT = {
        'method': 'Security.setIgnoreCertificateErrors',
        'params': params,
    }
    json = yield cmd_dict


def handle_certificate_error(
        event_id: int,
        action: CertificateErrorAction
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Handles a certificate error that fired a certificateError event.

    :param event_id: The ID of the event.
    :param action: The action to take on the certificate error.
    '''
    params: T_JSON_DICT = dict()
    params['eventId'] = event_id
    params['action'] = action.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Security.handleCertificateError',
        'params': params,
    }
    json = yield cmd_dict


def set_override_certificate_errors(
        override: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enable/disable overriding certificate errors. If enabled, all certificate error events need to
    be handled by the DevTools client and should be answered with ``handleCertificateError`` commands.

    :param override: If true, certificate errors will be overridden.
    '''
    params: T_JSON_DICT = dict()
    params['override'] = override
    cmd_dict: T_JSON_DICT = {
        'method': 'Security.setOverrideCertificateErrors',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Security.certificateError')
@dataclass
class CertificateError:
    '''
    There is a certificate error. If overriding certificate errors is enabled, then it should be
    handled with the ``handleCertificateError`` command. Note: this event does not fire if the
    certificate error has been allowed internally. Only one client per target should override
    certificate errors at the same time.
    '''
    #: The ID of the event.
    event_id: int
    #: The type of the error.
    error_type: str
    #: The url that was requested.
    request_url: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CertificateError:
        return cls(
            event_id=int(json['eventId']),
            error_type=str(json['errorType']),
            request_url=str(json['requestURL'])
        )


@event_class('Security.visibleSecurityStateChanged')
@dataclass
class VisibleSecurityStateChanged:
    '''
    **EXPERIMENTAL**

    The security state of the page changed.
    '''
    #: Security state information about the page.
    visible_security_state: VisibleSecurityState

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> VisibleSecurityStateChanged:
        return cls(
            visible_security_state=VisibleSecurityState.from_json(json['visibleSecurityState'])
        )


@event_class('Security.securityStateChanged')
@dataclass
class SecurityStateChanged:
    '''
    The security state of the page changed. No longer being sent.
    '''
    #: Security state.
    security_state: SecurityState
    #: True if the page was loaded over cryptographic transport such as HTTPS.
    scheme_is_cryptographic: bool
    #: Previously a list of explanations for the security state. Now always
    #: empty.
    explanations: typing.List[SecurityStateExplanation]
    #: Information about insecure content on the page.
    insecure_content_status: InsecureContentStatus
    #: Overrides user-visible description of the state. Always omitted.
    summary: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> SecurityStateChanged:
        return cls(
            security_state=SecurityState.from_json(json['securityState']),
            scheme_is_cryptographic=bool(json['schemeIsCryptographic']),
            explanations=[SecurityStateExplanation.from_json(i) for i in json['explanations']],
            insecure_content_status=InsecureContentStatus.from_json(json['insecureContentStatus']),
            summary=str(json['summary']) if 'summary' in json else None
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\oracle\dictionary.py ===
# dialects/oracle/dictionary.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

from .types import DATE
from .types import LONG
from .types import NUMBER
from .types import RAW
from .types import VARCHAR2
from ... import Column
from ... import MetaData
from ... import Table
from ... import table
from ...sql.sqltypes import CHAR

# constants
DB_LINK_PLACEHOLDER = "__$sa_dblink$__"
# tables
dual = table("dual")
dictionary_meta = MetaData()

# NOTE: all the dictionary_meta are aliases because oracle does not like
# using the full table@dblink for every column in query, and complains with
# ORA-00960: ambiguous column naming in select list
all_tables = Table(
    "all_tables" + DB_LINK_PLACEHOLDER,
    dictionary_meta,
    Column("owner", VARCHAR2(128), nullable=False),
    Column("table_name", VARCHAR2(128), nullable=False),
    Column("tablespace_name", VARCHAR2(30)),
    Column("cluster_name", VARCHAR2(128)),
    Column("iot_name", VARCHAR2(128)),
    Column("status", VARCHAR2(8)),
    Column("pct_free", NUMBER),
    Column("pct_used", NUMBER),
    Column("ini_trans", NUMBER),
    Column("max_trans", NUMBER),
    Column("initial_extent", NUMBER),
    Column("next_extent", NUMBER),
    Column("min_extents", NUMBER),
    Column("max_extents", NUMBER),
    Column("pct_increase", NUMBER),
    Column("freelists", NUMBER),
    Column("freelist_groups", NUMBER),
    Column("logging", VARCHAR2(3)),
    Column("backed_up", VARCHAR2(1)),
    Column("num_rows", NUMBER),
    Column("blocks", NUMBER),
    Column("empty_blocks", NUMBER),
    Column("avg_space", NUMBER),
    Column("chain_cnt", NUMBER),
    Column("avg_row_len", NUMBER),
    Column("avg_space_freelist_blocks", NUMBER),
    Column("num_freelist_blocks", NUMBER),
    Column("degree", VARCHAR2(10)),
    Column("instances", VARCHAR2(10)),
    Column("cache", VARCHAR2(5)),
    Column("table_lock", VARCHAR2(8)),
    Column("sample_size", NUMBER),
    Column("last_analyzed", DATE),
    Column("partitioned", VARCHAR2(3)),
    Column("iot_type", VARCHAR2(12)),
    Column("temporary", VARCHAR2(1)),
    Column("secondary", VARCHAR2(1)),
    Column("nested", VARCHAR2(3)),
    Column("buffer_pool", VARCHAR2(7)),
    Column("flash_cache", VARCHAR2(7)),
    Column("cell_flash_cache", VARCHAR2(7)),
    Column("row_movement", VARCHAR2(8)),
    Column("global_stats", VARCHAR2(3)),
    Column("user_stats", VARCHAR2(3)),
    Column("duration", VARCHAR2(15)),
    Column("skip_corrupt", VARCHAR2(8)),
    Column("monitoring", VARCHAR2(3)),
    Column("cluster_owner", VARCHAR2(128)),
    Column("dependencies", VARCHAR2(8)),
    Column("compression", VARCHAR2(8)),
    Column("compress_for", VARCHAR2(30)),
    Column("dropped", VARCHAR2(3)),
    Column("read_only", VARCHAR2(3)),
    Column("segment_created", VARCHAR2(3)),
    Column("result_cache", VARCHAR2(7)),
    Column("clustering", VARCHAR2(3)),
    Column("activity_tracking", VARCHAR2(23)),
    Column("dml_timestamp", VARCHAR2(25)),
    Column("has_identity", VARCHAR2(3)),
    Column("container_data", VARCHAR2(3)),
    Column("inmemory", VARCHAR2(8)),
    Column("inmemory_priority", VARCHAR2(8)),
    Column("inmemory_distribute", VARCHAR2(15)),
    Column("inmemory_compression", VARCHAR2(17)),
    Column("inmemory_duplicate", VARCHAR2(13)),
    Column("default_collation", VARCHAR2(100)),
    Column("duplicated", VARCHAR2(1)),
    Column("sharded", VARCHAR2(1)),
    Column("externally_sharded", VARCHAR2(1)),
    Column("externally_duplicated", VARCHAR2(1)),
    Column("external", VARCHAR2(3)),
    Column("hybrid", VARCHAR2(3)),
    Column("cellmemory", VARCHAR2(24)),
    Column("containers_default", VARCHAR2(3)),
    Column("container_map", VARCHAR2(3)),
    Column("extended_data_link", VARCHAR2(3)),
    Column("extended_data_link_map", VARCHAR2(3)),
    Column("inmemory_service", VARCHAR2(12)),
    Column("inmemory_service_name", VARCHAR2(1000)),
    Column("container_map_object", VARCHAR2(3)),
    Column("memoptimize_read", VARCHAR2(8)),
    Column("memoptimize_write", VARCHAR2(8)),
    Column("has_sensitive_column", VARCHAR2(3)),
    Column("admit_null", VARCHAR2(3)),
    Column("data_link_dml_enabled", VARCHAR2(3)),
    Column("logical_replication", VARCHAR2(8)),
).alias("a_tables")

all_views = Table(
    "all_views" + DB_LINK_PLACEHOLDER,
    dictionary_meta,
    Column("owner", VARCHAR2(128), nullable=False),
    Column("view_name", VARCHAR2(128), nullable=False),
    Column("text_length", NUMBER),
    Column("text", LONG),
    Column("text_vc", VARCHAR2(4000)),
    Column("type_text_length", NUMBER),
    Column("type_text", VARCHAR2(4000)),
    Column("oid_text_length", NUMBER),
    Column("oid_text", VARCHAR2(4000)),
    Column("view_type_owner", VARCHAR2(128)),
    Column("view_type", VARCHAR2(128)),
    Column("superview_name", VARCHAR2(128)),
    Column("editioning_view", VARCHAR2(1)),
    Column("read_only", VARCHAR2(1)),
    Column("container_data", VARCHAR2(1)),
    Column("bequeath", VARCHAR2(12)),
    Column("origin_con_id", VARCHAR2(256)),
    Column("default_collation", VARCHAR2(100)),
    Column("containers_default", VARCHAR2(3)),
    Column("container_map", VARCHAR2(3)),
    Column("extended_data_link", VARCHAR2(3)),
    Column("extended_data_link_map", VARCHAR2(3)),
    Column("has_sensitive_column", VARCHAR2(3)),
    Column("admit_null", VARCHAR2(3)),
    Column("pdb_local_only", VARCHAR2(3)),
).alias("a_views")

all_sequences = Table(
    "all_sequences" + DB_LINK_PLACEHOLDER,
    dictionary_meta,
    Column("sequence_owner", VARCHAR2(128), nullable=False),
    Column("sequence_name", VARCHAR2(128), nullable=False),
    Column("min_value", NUMBER),
    Column("max_value", NUMBER),
    Column("increment_by", NUMBER, nullable=False),
    Column("cycle_flag", VARCHAR2(1)),
    Column("order_flag", VARCHAR2(1)),
    Column("cache_size", NUMBER, nullable=False),
    Column("last_number", NUMBER, nullable=False),
    Column("scale_flag", VARCHAR2(1)),
    Column("extend_flag", VARCHAR2(1)),
    Column("sharded_flag", VARCHAR2(1)),
    Column("session_flag", VARCHAR2(1)),
    Column("keep_value", VARCHAR2(1)),
).alias("a_sequences")

all_users = Table(
    "all_users" + DB_LINK_PLACEHOLDER,
    dictionary_meta,
    Column("username", VARCHAR2(128), nullable=False),
    Column("user_id", NUMBER, nullable=False),
    Column("created", DATE, nullable=False),
    Column("common", VARCHAR2(3)),
    Column("oracle_maintained", VARCHAR2(1)),
    Column("inherited", VARCHAR2(3)),
    Column("default_collation", VARCHAR2(100)),
    Column("implicit", VARCHAR2(3)),
    Column("all_shard", VARCHAR2(3)),
    Column("external_shard", VARCHAR2(3)),
).alias("a_users")

all_mviews = Table(
    "all_mviews" + DB_LINK_PLACEHOLDER,
    dictionary_meta,
    Column("owner", VARCHAR2(128), nullable=False),
    Column("mview_name", VARCHAR2(128), nullable=False),
    Column("container_name", VARCHAR2(128), nullable=False),
    Column("query", LONG),
    Column("query_len", NUMBER(38)),
    Column("updatable", VARCHAR2(1)),
    Column("update_log", VARCHAR2(128)),
    Column("master_rollback_seg", VARCHAR2(128)),
    Column("master_link", VARCHAR2(128)),
    Column("rewrite_enabled", VARCHAR2(1)),
    Column("rewrite_capability", VARCHAR2(9)),
    Column("refresh_mode", VARCHAR2(6)),
    Column("refresh_method", VARCHAR2(8)),
    Column("build_mode", VARCHAR2(9)),
    Column("fast_refreshable", VARCHAR2(18)),
    Column("last_refresh_type", VARCHAR2(8)),
    Column("last_refresh_date", DATE),
    Column("last_refresh_end_time", DATE),
    Column("staleness", VARCHAR2(19)),
    Column("after_fast_refresh", VARCHAR2(19)),
    Column("unknown_prebuilt", VARCHAR2(1)),
    Column("unknown_plsql_func", VARCHAR2(1)),
    Column("unknown_external_table", VARCHAR2(1)),
    Column("unknown_consider_fresh", VARCHAR2(1)),
    Column("unknown_import", VARCHAR2(1)),
    Column("unknown_trusted_fd", VARCHAR2(1)),
    Column("compile_state", VARCHAR2(19)),
    Column("use_no_index", VARCHAR2(1)),
    Column("stale_since", DATE),
    Column("num_pct_tables", NUMBER),
    Column("num_fresh_pct_regions", NUMBER),
    Column("num_stale_pct_regions", NUMBER),
    Column("segment_created", VARCHAR2(3)),
    Column("evaluation_edition", VARCHAR2(128)),
    Column("unusable_before", VARCHAR2(128)),
    Column("unusable_beginning", VARCHAR2(128)),
    Column("default_collation", VARCHAR2(100)),
    Column("on_query_computation", VARCHAR2(1)),
    Column("auto", VARCHAR2(3)),
).alias("a_mviews")

all_tab_identity_cols = Table(
    "all_tab_identity_cols" + DB_LINK_PLACEHOLDER,
    dictionary_meta,
    Column("owner", VARCHAR2(128), nullable=False),
    Column("table_name", VARCHAR2(128), nullable=False),
    Column("column_name", VARCHAR2(128), nullable=False),
    Column("generation_type", VARCHAR2(10)),
    Column("sequence_name", VARCHAR2(128), nullable=False),
    Column("identity_options", VARCHAR2(298)),
).alias("a_tab_identity_cols")

all_tab_cols = Table(
    "all_tab_cols" + DB_LINK_PLACEHOLDER,
    dictionary_meta,
    Column("owner", VARCHAR2(128), nullable=False),
    Column("table_name", VARCHAR2(128), nullable=False),
    Column("column_name", VARCHAR2(128), nullable=False),
    Column("data_type", VARCHAR2(128)),
    Column("data_type_mod", VARCHAR2(3)),
    Column("data_type_owner", VARCHAR2(128)),
    Column("data_length", NUMBER, nullable=False),
    Column("data_precision", NUMBER),
    Column("data_scale", NUMBER),
    Column("nullable", VARCHAR2(1)),
    Column("column_id", NUMBER),
    Column("default_length", NUMBER),
    Column("data_default", LONG),
    Column("num_distinct", NUMBER),
    Column("low_value", RAW(1000)),
    Column("high_value", RAW(1000)),
    Column("density", NUMBER),
    Column("num_nulls", NUMBER),
    Column("num_buckets", NUMBER),
    Column("last_analyzed", DATE),
    Column("sample_size", NUMBER),
    Column("character_set_name", VARCHAR2(44)),
    Column("char_col_decl_length", NUMBER),
    Column("global_stats", VARCHAR2(3)),
    Column("user_stats", VARCHAR2(3)),
    Column("avg_col_len", NUMBER),
    Column("char_length", NUMBER),
    Column("char_used", VARCHAR2(1)),
    Column("v80_fmt_image", VARCHAR2(3)),
    Column("data_upgraded", VARCHAR2(3)),
    Column("hidden_column", VARCHAR2(3)),
    Column("virtual_column", VARCHAR2(3)),
    Column("segment_column_id", NUMBER),
    Column("internal_column_id", NUMBER, nullable=False),
    Column("histogram", VARCHAR2(15)),
    Column("qualified_col_name", VARCHAR2(4000)),
    Column("user_generated", VARCHAR2(3)),
    Column("default_on_null", VARCHAR2(3)),
    Column("identity_column", VARCHAR2(3)),
    Column("evaluation_edition", VARCHAR2(128)),
    Column("unusable_before", VARCHAR2(128)),
    Column("unusable_beginning", VARCHAR2(128)),
    Column("collation", VARCHAR2(100)),
    Column("collated_column_id", NUMBER),
).alias("a_tab_cols")

all_tab_comments = Table(
    "all_tab_comments" + DB_LINK_PLACEHOLDER,
    dictionary_meta,
    Column("owner", VARCHAR2(128), nullable=False),
    Column("table_name", VARCHAR2(128), nullable=False),
    Column("table_type", VARCHAR2(11)),
    Column("comments", VARCHAR2(4000)),
    Column("origin_con_id", NUMBER),
).alias("a_tab_comments")

all_col_comments = Table(
    "all_col_comments" + DB_LINK_PLACEHOLDER,
    dictionary_meta,
    Column("owner", VARCHAR2(128), nullable=False),
    Column("table_name", VARCHAR2(128), nullable=False),
    Column("column_name", VARCHAR2(128), nullable=False),
    Column("comments", VARCHAR2(4000)),
    Column("origin_con_id", NUMBER),
).alias("a_col_comments")

all_mview_comments = Table(
    "all_mview_comments" + DB_LINK_PLACEHOLDER,
    dictionary_meta,
    Column("owner", VARCHAR2(128), nullable=False),
    Column("mview_name", VARCHAR2(128), nullable=False),
    Column("comments", VARCHAR2(4000)),
).alias("a_mview_comments")

all_ind_columns = Table(
    "all_ind_columns" + DB_LINK_PLACEHOLDER,
    dictionary_meta,
    Column("index_owner", VARCHAR2(128), nullable=False),
    Column("index_name", VARCHAR2(128), nullable=False),
    Column("table_owner", VARCHAR2(128), nullable=False),
    Column("table_name", VARCHAR2(128), nullable=False),
    Column("column_name", VARCHAR2(4000)),
    Column("column_position", NUMBER, nullable=False),
    Column("column_length", NUMBER, nullable=False),
    Column("char_length", NUMBER),
    Column("descend", VARCHAR2(4)),
    Column("collated_column_id", NUMBER),
).alias("a_ind_columns")

all_indexes = Table(
    "all_indexes" + DB_LINK_PLACEHOLDER,
    dictionary_meta,
    Column("owner", VARCHAR2(128), nullable=False),
    Column("index_name", VARCHAR2(128), nullable=False),
    Column("index_type", VARCHAR2(27)),
    Column("table_owner", VARCHAR2(128), nullable=False),
    Column("table_name", VARCHAR2(128), nullable=False),
    Column("table_type", CHAR(11)),
    Column("uniqueness", VARCHAR2(9)),
    Column("compression", VARCHAR2(13)),
    Column("prefix_length", NUMBER),
    Column("tablespace_name", VARCHAR2(30)),
    Column("ini_trans", NUMBER),
    Column("max_trans", NUMBER),
    Column("initial_extent", NUMBER),
    Column("next_extent", NUMBER),
    Column("min_extents", NUMBER),
    Column("max_extents", NUMBER),
    Column("pct_increase", NUMBER),
    Column("pct_threshold", NUMBER),
    Column("include_column", NUMBER),
    Column("freelists", NUMBER),
    Column("freelist_groups", NUMBER),
    Column("pct_free", NUMBER),
    Column("logging", VARCHAR2(3)),
    Column("blevel", NUMBER),
    Column("leaf_blocks", NUMBER),
    Column("distinct_keys", NUMBER),
    Column("avg_leaf_blocks_per_key", NUMBER),
    Column("avg_data_blocks_per_key", NUMBER),
    Column("clustering_factor", NUMBER),
    Column("status", VARCHAR2(8)),
    Column("num_rows", NUMBER),
    Column("sample_size", NUMBER),
    Column("last_analyzed", DATE),
    Column("degree", VARCHAR2(40)),
    Column("instances", VARCHAR2(40)),
    Column("partitioned", VARCHAR2(3)),
    Column("temporary", VARCHAR2(1)),
    Column("generated", VARCHAR2(1)),
    Column("secondary", VARCHAR2(1)),
    Column("buffer_pool", VARCHAR2(7)),
    Column("flash_cache", VARCHAR2(7)),
    Column("cell_flash_cache", VARCHAR2(7)),
    Column("user_stats", VARCHAR2(3)),
    Column("duration", VARCHAR2(15)),
    Column("pct_direct_access", NUMBER),
    Column("ityp_owner", VARCHAR2(128)),
    Column("ityp_name", VARCHAR2(128)),
    Column("parameters", VARCHAR2(1000)),
    Column("global_stats", VARCHAR2(3)),
    Column("domidx_status", VARCHAR2(12)),
    Column("domidx_opstatus", VARCHAR2(6)),
    Column("funcidx_status", VARCHAR2(8)),
    Column("join_index", VARCHAR2(3)),
    Column("iot_redundant_pkey_elim", VARCHAR2(3)),
    Column("dropped", VARCHAR2(3)),
    Column("visibility", VARCHAR2(9)),
    Column("domidx_management", VARCHAR2(14)),
    Column("segment_created", VARCHAR2(3)),
    Column("orphaned_entries", VARCHAR2(3)),
    Column("indexing", VARCHAR2(7)),
    Column("auto", VARCHAR2(3)),
).alias("a_indexes")

all_ind_expressions = Table(
    "all_ind_expressions" + DB_LINK_PLACEHOLDER,
    dictionary_meta,
    Column("index_owner", VARCHAR2(128), nullable=False),
    Column("index_name", VARCHAR2(128), nullable=False),
    Column("table_owner", VARCHAR2(128), nullable=False),
    Column("table_name", VARCHAR2(128), nullable=False),
    Column("column_expression", LONG),
    Column("column_position", NUMBER, nullable=False),
).alias("a_ind_expressions")

all_constraints = Table(
    "all_constraints" + DB_LINK_PLACEHOLDER,
    dictionary_meta,
    Column("owner", VARCHAR2(128)),
    Column("constraint_name", VARCHAR2(128)),
    Column("constraint_type", VARCHAR2(1)),
    Column("table_name", VARCHAR2(128)),
    Column("search_condition", LONG),
    Column("search_condition_vc", VARCHAR2(4000)),
    Column("r_owner", VARCHAR2(128)),
    Column("r_constraint_name", VARCHAR2(128)),
    Column("delete_rule", VARCHAR2(9)),
    Column("status", VARCHAR2(8)),
    Column("deferrable", VARCHAR2(14)),
    Column("deferred", VARCHAR2(9)),
    Column("validated", VARCHAR2(13)),
    Column("generated", VARCHAR2(14)),
    Column("bad", VARCHAR2(3)),
    Column("rely", VARCHAR2(4)),
    Column("last_change", DATE),
    Column("index_owner", VARCHAR2(128)),
    Column("index_name", VARCHAR2(128)),
    Column("invalid", VARCHAR2(7)),
    Column("view_related", VARCHAR2(14)),
    Column("origin_con_id", VARCHAR2(256)),
).alias("a_constraints")

all_cons_columns = Table(
    "all_cons_columns" + DB_LINK_PLACEHOLDER,
    dictionary_meta,
    Column("owner", VARCHAR2(128), nullable=False),
    Column("constraint_name", VARCHAR2(128), nullable=False),
    Column("table_name", VARCHAR2(128), nullable=False),
    Column("column_name", VARCHAR2(4000)),
    Column("position", NUMBER),
).alias("a_cons_columns")

# TODO figure out if it's still relevant, since there is no mention from here
# https://docs.oracle.com/en/database/oracle/oracle-database/21/refrn/ALL_DB_LINKS.html
# original note:
# using user_db_links here since all_db_links appears
# to have more restricted permissions.
# https://docs.oracle.com/cd/B28359_01/server.111/b28310/ds_admin005.htm
# will need to hear from more users if we are doing
# the right thing here.  See [ticket:2619]
all_db_links = Table(
    "all_db_links" + DB_LINK_PLACEHOLDER,
    dictionary_meta,
    Column("owner", VARCHAR2(128), nullable=False),
    Column("db_link", VARCHAR2(128), nullable=False),
    Column("username", VARCHAR2(128)),
    Column("host", VARCHAR2(2000)),
    Column("created", DATE, nullable=False),
    Column("hidden", VARCHAR2(3)),
    Column("shard_internal", VARCHAR2(3)),
    Column("valid", VARCHAR2(3)),
    Column("intra_cdb", VARCHAR2(3)),
).alias("a_db_links")

all_synonyms = Table(
    "all_synonyms" + DB_LINK_PLACEHOLDER,
    dictionary_meta,
    Column("owner", VARCHAR2(128)),
    Column("synonym_name", VARCHAR2(128)),
    Column("table_owner", VARCHAR2(128)),
    Column("table_name", VARCHAR2(128)),
    Column("db_link", VARCHAR2(128)),
    Column("origin_con_id", VARCHAR2(256)),
).alias("a_synonyms")

all_objects = Table(
    "all_objects" + DB_LINK_PLACEHOLDER,
    dictionary_meta,
    Column("owner", VARCHAR2(128), nullable=False),
    Column("object_name", VARCHAR2(128), nullable=False),
    Column("subobject_name", VARCHAR2(128)),
    Column("object_id", NUMBER, nullable=False),
    Column("data_object_id", NUMBER),
    Column("object_type", VARCHAR2(23)),
    Column("created", DATE, nullable=False),
    Column("last_ddl_time", DATE, nullable=False),
    Column("timestamp", VARCHAR2(19)),
    Column("status", VARCHAR2(7)),
    Column("temporary", VARCHAR2(1)),
    Column("generated", VARCHAR2(1)),
    Column("secondary", VARCHAR2(1)),
    Column("namespace", NUMBER, nullable=False),
    Column("edition_name", VARCHAR2(128)),
    Column("sharing", VARCHAR2(13)),
    Column("editionable", VARCHAR2(1)),
    Column("oracle_maintained", VARCHAR2(1)),
    Column("application", VARCHAR2(1)),
    Column("default_collation", VARCHAR2(100)),
    Column("duplicated", VARCHAR2(1)),
    Column("sharded", VARCHAR2(1)),
    Column("created_appid", NUMBER),
    Column("created_vsnid", NUMBER),
    Column("modified_appid", NUMBER),
    Column("modified_vsnid", NUMBER),
).alias("a_objects")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\vector\implicitregion.py ===
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.polytools import gcd
from sympy.sets.sets import Complement
from sympy.core import Basic, Tuple, diff, expand, Eq, Integer
from sympy.core.sorting import ordered
from sympy.core.symbol import _symbol
from sympy.solvers import solveset, nonlinsolve, diophantine
from sympy.polys import total_degree
from sympy.geometry import Point
from sympy.ntheory.factor_ import core


class ImplicitRegion(Basic):
    """
    Represents an implicit region in space.

    Examples
    ========

    >>> from sympy import Eq
    >>> from sympy.abc import x, y, z, t
    >>> from sympy.vector import ImplicitRegion

    >>> ImplicitRegion((x, y), x**2 + y**2 - 4)
    ImplicitRegion((x, y), x**2 + y**2 - 4)
    >>> ImplicitRegion((x, y), Eq(y*x, 1))
    ImplicitRegion((x, y), x*y - 1)

    >>> parabola = ImplicitRegion((x, y), y**2 - 4*x)
    >>> parabola.degree
    2
    >>> parabola.equation
    -4*x + y**2
    >>> parabola.rational_parametrization(t)
    (4/t**2, 4/t)

    >>> r = ImplicitRegion((x, y, z), Eq(z, x**2 + y**2))
    >>> r.variables
    (x, y, z)
    >>> r.singular_points()
    EmptySet
    >>> r.regular_point()
    (-10, -10, 200)

    Parameters
    ==========

    variables : tuple to map variables in implicit equation to base scalars.

    equation : An expression or Eq denoting the implicit equation of the region.

    """
    def __new__(cls, variables, equation):
        if not isinstance(variables, Tuple):
            variables = Tuple(*variables)

        if isinstance(equation, Eq):
            equation = equation.lhs - equation.rhs

        return super().__new__(cls, variables, equation)

    @property
    def variables(self):
        return self.args[0]

    @property
    def equation(self):
        return self.args[1]

    @property
    def degree(self):
        return total_degree(self.equation)

    def regular_point(self):
        """
        Returns a point on the implicit region.

        Examples
        ========

        >>> from sympy.abc import x, y, z
        >>> from sympy.vector import ImplicitRegion
        >>> circle = ImplicitRegion((x, y), (x + 2)**2 + (y - 3)**2 - 16)
        >>> circle.regular_point()
        (-2, -1)
        >>> parabola = ImplicitRegion((x, y), x**2 - 4*y)
        >>> parabola.regular_point()
        (0, 0)
        >>> r = ImplicitRegion((x, y, z), (x + y + z)**4)
        >>> r.regular_point()
        (-10, -10, 20)

        References
        ==========

        - Erik Hillgarter, "Rational Points on Conics", Diploma Thesis, RISC-Linz,
          J. Kepler Universitat Linz, 1996. Available:
          https://www3.risc.jku.at/publications/download/risc_1355/Rational%20Points%20on%20Conics.pdf

        """
        equation = self.equation

        if len(self.variables) == 1:
            return (list(solveset(equation, self.variables[0], domain=S.Reals))[0],)
        elif len(self.variables) == 2:

            if self.degree == 2:
                coeffs = a, b, c, d, e, f = conic_coeff(self.variables, equation)

                if b**2 == 4*a*c:
                    x_reg, y_reg = self._regular_point_parabola(*coeffs)
                else:
                    x_reg, y_reg = self._regular_point_ellipse(*coeffs)
                return x_reg, y_reg

        if len(self.variables) == 3:
            x, y, z = self.variables

            for x_reg in range(-10, 10):
                for y_reg in range(-10, 10):
                    if not solveset(equation.subs({x: x_reg, y: y_reg}), self.variables[2], domain=S.Reals).is_empty:
                        return (x_reg, y_reg, list(solveset(equation.subs({x: x_reg, y: y_reg})))[0])

        if len(self.singular_points()) != 0:
            return list[self.singular_points()][0]

        raise NotImplementedError()

    def _regular_point_parabola(self, a, b, c, d, e, f):
            ok = (a, d) != (0, 0) and (c, e) != (0, 0) and b**2 == 4*a*c and (a, c) != (0, 0)

            if not ok:
                raise ValueError("Rational Point on the conic does not exist")

            if a != 0:
                d_dash, f_dash = (4*a*e - 2*b*d, 4*a*f - d**2)
                if d_dash != 0:
                    y_reg = -f_dash/d_dash
                    x_reg = -(d + b*y_reg)/(2*a)
                else:
                    ok = False
            elif c != 0:
                d_dash, f_dash = (4*c*d - 2*b*e, 4*c*f - e**2)
                if d_dash != 0:
                    x_reg = -f_dash/d_dash
                    y_reg = -(e + b*x_reg)/(2*c)
                else:
                    ok = False

            if ok:
                return x_reg, y_reg
            else:
                raise ValueError("Rational Point on the conic does not exist")

    def _regular_point_ellipse(self, a, b, c, d, e, f):
            D = 4*a*c - b**2
            ok = D

            if not ok:
                raise ValueError("Rational Point on the conic does not exist")

            if a == 0 and c == 0:
                K = -1
                L = 4*(d*e - b*f)
            elif c != 0:
                K = D
                L = 4*c**2*d**2 - 4*b*c*d*e + 4*a*c*e**2 + 4*b**2*c*f - 16*a*c**2*f
            else:
                K = D
                L = 4*a**2*e**2 - 4*b*a*d*e + 4*b**2*a*f

            ok = L != 0 and not(K > 0 and L < 0)
            if not ok:
                raise ValueError("Rational Point on the conic does not exist")

            K = Rational(K).limit_denominator(10**12)
            L = Rational(L).limit_denominator(10**12)

            k1, k2 = K.p, K.q
            l1, l2 = L.p, L.q
            g = gcd(k2, l2)

            a1 = (l2*k2)/g
            b1 = (k1*l2)/g
            c1 = -(l1*k2)/g
            a2 = sign(a1)*core(abs(a1), 2)
            r1 = sqrt(a1/a2)
            b2 = sign(b1)*core(abs(b1), 2)
            r2 = sqrt(b1/b2)
            c2 = sign(c1)*core(abs(c1), 2)
            r3 = sqrt(c1/c2)

            g = gcd(gcd(a2, b2), c2)
            a2 = a2/g
            b2 = b2/g
            c2 = c2/g

            g1 = gcd(a2, b2)
            a2 = a2/g1
            b2 = b2/g1
            c2 = c2*g1

            g2 = gcd(a2,c2)
            a2 = a2/g2
            b2 = b2*g2
            c2 = c2/g2

            g3 = gcd(b2, c2)
            a2 = a2*g3
            b2 = b2/g3
            c2 = c2/g3

            x, y, z = symbols("x y z")
            eq = a2*x**2 + b2*y**2 + c2*z**2

            solutions = diophantine(eq)

            if len(solutions) == 0:
                raise ValueError("Rational Point on the conic does not exist")

            flag = False
            for sol in solutions:
                syms = Tuple(*sol).free_symbols
                rep = dict.fromkeys(syms, 3)
                sol_z = sol[2]

                if sol_z == 0:
                    flag = True
                    continue

                if not isinstance(sol_z, (int, Integer)):
                    syms_z = sol_z.free_symbols

                    if len(syms_z) == 1:
                        p = next(iter(syms_z))
                        p_values = Complement(S.Integers, solveset(Eq(sol_z, 0), p, S.Integers))
                        rep[p] = next(iter(p_values))

                    if len(syms_z) == 2:
                        p, q = list(ordered(syms_z))

                        for i in S.Integers:
                            subs_sol_z = sol_z.subs(p, i)
                            q_values = Complement(S.Integers, solveset(Eq(subs_sol_z, 0), q, S.Integers))

                            if not q_values.is_empty:
                                rep[p] = i
                                rep[q] = next(iter(q_values))
                                break

                    if len(syms) != 0:
                        x, y, z = tuple(s.subs(rep) for s in sol)
                    else:
                        x, y, z =   sol
                    flag = False
                    break

            if flag:
                raise ValueError("Rational Point on the conic does not exist")

            x = (x*g3)/r1
            y = (y*g2)/r2
            z = (z*g1)/r3
            x = x/z
            y = y/z

            if a == 0 and c == 0:
                x_reg = (x + y - 2*e)/(2*b)
                y_reg = (x - y - 2*d)/(2*b)
            elif c != 0:
                x_reg = (x - 2*d*c + b*e)/K
                y_reg = (y - b*x_reg - e)/(2*c)
            else:
                y_reg = (x - 2*e*a + b*d)/K
                x_reg = (y - b*y_reg - d)/(2*a)

            return x_reg, y_reg

    def singular_points(self):
        """
        Returns a set of singular points of the region.

        The singular points are those points on the region
        where all partial derivatives vanish.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy.vector import ImplicitRegion
        >>> I = ImplicitRegion((x, y), (y-1)**2 -x**3 + 2*x**2 -x)
        >>> I.singular_points()
        {(1, 1)}

        """
        eq_list = [self.equation]
        for var in self.variables:
            eq_list += [diff(self.equation, var)]

        return nonlinsolve(eq_list, list(self.variables))

    def multiplicity(self, point):
        """
        Returns the multiplicity of a singular point on the region.

        A singular point (x,y) of region is said to be of multiplicity m
        if all the partial derivatives off to order m - 1 vanish there.

        Examples
        ========

        >>> from sympy.abc import x, y, z
        >>> from sympy.vector import ImplicitRegion
        >>> I = ImplicitRegion((x, y, z), x**2 + y**3 - z**4)
        >>> I.singular_points()
        {(0, 0, 0)}
        >>> I.multiplicity((0, 0, 0))
        2

        """
        if isinstance(point, Point):
            point = point.args

        modified_eq = self.equation

        for i, var in enumerate(self.variables):
            modified_eq = modified_eq.subs(var, var + point[i])
        modified_eq = expand(modified_eq)

        if len(modified_eq.args) != 0:
            terms = modified_eq.args
            m = min(total_degree(term) for term in terms)
        else:
            terms = modified_eq
            m = total_degree(terms)

        return m

    def rational_parametrization(self, parameters=('t', 's'), reg_point=None):
        """
        Returns the rational parametrization of implicit region.

        Examples
        ========

        >>> from sympy import Eq
        >>> from sympy.abc import x, y, z, s, t
        >>> from sympy.vector import ImplicitRegion

        >>> parabola = ImplicitRegion((x, y), y**2 - 4*x)
        >>> parabola.rational_parametrization()
        (4/t**2, 4/t)

        >>> circle = ImplicitRegion((x, y), Eq(x**2 + y**2, 4))
        >>> circle.rational_parametrization()
        (4*t/(t**2 + 1), 4*t**2/(t**2 + 1) - 2)

        >>> I = ImplicitRegion((x, y), x**3 + x**2 - y**2)
        >>> I.rational_parametrization()
        (t**2 - 1, t*(t**2 - 1))

        >>> cubic_curve = ImplicitRegion((x, y), x**3 + x**2 - y**2)
        >>> cubic_curve.rational_parametrization(parameters=(t))
        (t**2 - 1, t*(t**2 - 1))

        >>> sphere = ImplicitRegion((x, y, z), x**2 + y**2 + z**2 - 4)
        >>> sphere.rational_parametrization(parameters=(t, s))
        (-2 + 4/(s**2 + t**2 + 1), 4*s/(s**2 + t**2 + 1), 4*t/(s**2 + t**2 + 1))

        For some conics, regular_points() is unable to find a point on curve.
        To calulcate the parametric representation in such cases, user need
        to determine a point on the region and pass it using reg_point.

        >>> c = ImplicitRegion((x, y), (x  - 1/2)**2 + (y)**2 - (1/4)**2)
        >>> c.rational_parametrization(reg_point=(3/4, 0))
        (0.75 - 0.5/(t**2 + 1), -0.5*t/(t**2 + 1))

        References
        ==========

        - Christoph M. Hoffmann, "Conversion Methods between Parametric and
          Implicit Curves and Surfaces", Purdue e-Pubs, 1990. Available:
          https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=1827&context=cstech

        """
        equation = self.equation
        degree = self.degree

        if degree == 1:
            if len(self.variables) == 1:
                return (equation,)
            elif len(self.variables) == 2:
                x, y = self.variables
                y_par = list(solveset(equation, y))[0]
                return x, y_par
            else:
                raise NotImplementedError()

        point = ()

        # Finding the (n - 1) fold point of the monoid of degree
        if degree == 2:
            # For degree 2 curves, either a regular point or a singular point can be used.
            if reg_point is not None:
                # Using point provided by the user as regular point
                point = reg_point
            else:
                if len(self.singular_points()) != 0:
                    point = list(self.singular_points())[0]
                else:
                    point = self.regular_point()

        if len(self.singular_points()) != 0:
            singular_points = self.singular_points()
            for spoint in singular_points:
                syms = Tuple(*spoint).free_symbols
                rep = dict.fromkeys(syms, 2)

                if len(syms) != 0:
                    spoint = tuple(s.subs(rep) for s in spoint)

                if self.multiplicity(spoint) == degree - 1:
                    point = spoint
                    break

        if len(point) == 0:
            # The region in not a monoid
            raise NotImplementedError()

        modified_eq = equation

        # Shifting the region such that fold point moves to origin
        for i, var in enumerate(self.variables):
            modified_eq = modified_eq.subs(var, var + point[i])
        modified_eq = expand(modified_eq)

        hn = hn_1 = 0
        for term in modified_eq.args:
            if total_degree(term) == degree:
                hn += term
            else:
                hn_1 += term

        hn_1 = -1*hn_1

        if not isinstance(parameters, tuple):
            parameters = (parameters,)

        if len(self.variables) == 2:

            parameter1 = parameters[0]
            if parameter1 == 's':
                # To avoid name conflict between parameters
                s = _symbol('s_', real=True)
            else:
                s = _symbol('s', real=True)
            t = _symbol(parameter1, real=True)

            hn = hn.subs({self.variables[0]: s, self.variables[1]: t})
            hn_1 = hn_1.subs({self.variables[0]: s, self.variables[1]: t})

            x_par = (s*(hn_1/hn)).subs(s, 1) + point[0]
            y_par = (t*(hn_1/hn)).subs(s, 1) + point[1]

            return x_par, y_par

        elif len(self.variables) == 3:

            parameter1, parameter2 = parameters
            if 'r' in parameters:
                # To avoid name conflict between parameters
                r = _symbol('r_', real=True)
            else:
                r = _symbol('r', real=True)
            s = _symbol(parameter2, real=True)
            t = _symbol(parameter1, real=True)

            hn = hn.subs({self.variables[0]: r, self.variables[1]: s, self.variables[2]: t})
            hn_1 = hn_1.subs({self.variables[0]: r, self.variables[1]: s, self.variables[2]: t})

            x_par = (r*(hn_1/hn)).subs(r, 1) + point[0]
            y_par = (s*(hn_1/hn)).subs(r, 1) + point[1]
            z_par = (t*(hn_1/hn)).subs(r, 1) + point[2]

            return x_par, y_par, z_par

        raise NotImplementedError()

def conic_coeff(variables, equation):
    if total_degree(equation) != 2:
        raise ValueError()
    x = variables[0]
    y = variables[1]

    equation = expand(equation)
    a = equation.coeff(x**2)
    b = equation.coeff(x*y)
    c = equation.coeff(y**2)
    d = equation.coeff(x, 1).coeff(y, 0)
    e = equation.coeff(y, 1).coeff(x, 0)
    f = equation.coeff(x, 0).coeff(y, 0)
    return a, b, c, d, e, f

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\fetch.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Fetch
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import io
from . import network
from . import page


class RequestId(str):
    '''
    Unique request identifier.
    Note that this does not identify individual HTTP requests that are part of
    a network request.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> RequestId:
        return cls(json)

    def __repr__(self):
        return 'RequestId({})'.format(super().__repr__())


class RequestStage(enum.Enum):
    '''
    Stages of the request to handle. Request will intercept before the request is
    sent. Response will intercept after the response is received (but before response
    body is received).
    '''
    REQUEST = "Request"
    RESPONSE = "Response"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class RequestPattern:
    #: Wildcards (``'*'`` -> zero or more, ``'?'`` -> exactly one) are allowed. Escape character is
    #: backslash. Omitting is equivalent to ``"*"``.
    url_pattern: typing.Optional[str] = None

    #: If set, only requests for matching resource types will be intercepted.
    resource_type: typing.Optional[network.ResourceType] = None

    #: Stage at which to begin intercepting requests. Default is Request.
    request_stage: typing.Optional[RequestStage] = None

    def to_json(self):
        json = dict()
        if self.url_pattern is not None:
            json['urlPattern'] = self.url_pattern
        if self.resource_type is not None:
            json['resourceType'] = self.resource_type.to_json()
        if self.request_stage is not None:
            json['requestStage'] = self.request_stage.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url_pattern=str(json['urlPattern']) if 'urlPattern' in json else None,
            resource_type=network.ResourceType.from_json(json['resourceType']) if 'resourceType' in json else None,
            request_stage=RequestStage.from_json(json['requestStage']) if 'requestStage' in json else None,
        )


@dataclass
class HeaderEntry:
    '''
    Response HTTP header entry
    '''
    name: str

    value: str

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['value'] = self.value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            value=str(json['value']),
        )


@dataclass
class AuthChallenge:
    '''
    Authorization challenge for HTTP status code 401 or 407.
    '''
    #: Origin of the challenger.
    origin: str

    #: The authentication scheme used, such as basic or digest
    scheme: str

    #: The realm of the challenge. May be empty.
    realm: str

    #: Source of the authentication challenge.
    source: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['origin'] = self.origin
        json['scheme'] = self.scheme
        json['realm'] = self.realm
        if self.source is not None:
            json['source'] = self.source
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            origin=str(json['origin']),
            scheme=str(json['scheme']),
            realm=str(json['realm']),
            source=str(json['source']) if 'source' in json else None,
        )


@dataclass
class AuthChallengeResponse:
    '''
    Response to an AuthChallenge.
    '''
    #: The decision on what to do in response to the authorization challenge.  Default means
    #: deferring to the default behavior of the net stack, which will likely either the Cancel
    #: authentication or display a popup dialog box.
    response: str

    #: The username to provide, possibly empty. Should only be set if response is
    #: ProvideCredentials.
    username: typing.Optional[str] = None

    #: The password to provide, possibly empty. Should only be set if response is
    #: ProvideCredentials.
    password: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['response'] = self.response
        if self.username is not None:
            json['username'] = self.username
        if self.password is not None:
            json['password'] = self.password
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            response=str(json['response']),
            username=str(json['username']) if 'username' in json else None,
            password=str(json['password']) if 'password' in json else None,
        )


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables the fetch domain.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.disable',
    }
    json = yield cmd_dict


def enable(
        patterns: typing.Optional[typing.List[RequestPattern]] = None,
        handle_auth_requests: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables issuing of requestPaused events. A request will be paused until client
    calls one of failRequest, fulfillRequest or continueRequest/continueWithAuth.

    :param patterns: *(Optional)* If specified, only requests matching any of these patterns will produce fetchRequested event and will be paused until clients response. If not set, all requests will be affected.
    :param handle_auth_requests: *(Optional)* If true, authRequired events will be issued and requests will be paused expecting a call to continueWithAuth.
    '''
    params: T_JSON_DICT = dict()
    if patterns is not None:
        params['patterns'] = [i.to_json() for i in patterns]
    if handle_auth_requests is not None:
        params['handleAuthRequests'] = handle_auth_requests
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.enable',
        'params': params,
    }
    json = yield cmd_dict


def fail_request(
        request_id: RequestId,
        error_reason: network.ErrorReason
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Causes the request to fail with specified reason.

    :param request_id: An id the client received in requestPaused event.
    :param error_reason: Causes the request to fail with the given reason.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    params['errorReason'] = error_reason.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.failRequest',
        'params': params,
    }
    json = yield cmd_dict


def fulfill_request(
        request_id: RequestId,
        response_code: int,
        response_headers: typing.Optional[typing.List[HeaderEntry]] = None,
        binary_response_headers: typing.Optional[str] = None,
        body: typing.Optional[str] = None,
        response_phrase: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Provides response to the request.

    :param request_id: An id the client received in requestPaused event.
    :param response_code: An HTTP response code.
    :param response_headers: *(Optional)* Response headers.
    :param binary_response_headers: *(Optional)* Alternative way of specifying response headers as a \0-separated series of name: value pairs. Prefer the above method unless you need to represent some non-UTF8 values that can't be transmitted over the protocol as text.
    :param body: *(Optional)* A response body. If absent, original response body will be used if the request is intercepted at the response stage and empty body will be used if the request is intercepted at the request stage.
    :param response_phrase: *(Optional)* A textual representation of responseCode. If absent, a standard phrase matching responseCode is used.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    params['responseCode'] = response_code
    if response_headers is not None:
        params['responseHeaders'] = [i.to_json() for i in response_headers]
    if binary_response_headers is not None:
        params['binaryResponseHeaders'] = binary_response_headers
    if body is not None:
        params['body'] = body
    if response_phrase is not None:
        params['responsePhrase'] = response_phrase
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.fulfillRequest',
        'params': params,
    }
    json = yield cmd_dict


def continue_request(
        request_id: RequestId,
        url: typing.Optional[str] = None,
        method: typing.Optional[str] = None,
        post_data: typing.Optional[str] = None,
        headers: typing.Optional[typing.List[HeaderEntry]] = None,
        intercept_response: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Continues the request, optionally modifying some of its parameters.

    :param request_id: An id the client received in requestPaused event.
    :param url: *(Optional)* If set, the request url will be modified in a way that's not observable by page.
    :param method: *(Optional)* If set, the request method is overridden.
    :param post_data: *(Optional)* If set, overrides the post data in the request.
    :param headers: *(Optional)* If set, overrides the request headers. Note that the overrides do not extend to subsequent redirect hops, if a redirect happens. Another override may be applied to a different request produced by a redirect.
    :param intercept_response: **(EXPERIMENTAL)** *(Optional)* If set, overrides response interception behavior for this request.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    if url is not None:
        params['url'] = url
    if method is not None:
        params['method'] = method
    if post_data is not None:
        params['postData'] = post_data
    if headers is not None:
        params['headers'] = [i.to_json() for i in headers]
    if intercept_response is not None:
        params['interceptResponse'] = intercept_response
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.continueRequest',
        'params': params,
    }
    json = yield cmd_dict


def continue_with_auth(
        request_id: RequestId,
        auth_challenge_response: AuthChallengeResponse
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Continues a request supplying authChallengeResponse following authRequired event.

    :param request_id: An id the client received in authRequired event.
    :param auth_challenge_response: Response to  with an authChallenge.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    params['authChallengeResponse'] = auth_challenge_response.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.continueWithAuth',
        'params': params,
    }
    json = yield cmd_dict


def continue_response(
        request_id: RequestId,
        response_code: typing.Optional[int] = None,
        response_phrase: typing.Optional[str] = None,
        response_headers: typing.Optional[typing.List[HeaderEntry]] = None,
        binary_response_headers: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Continues loading of the paused response, optionally modifying the
    response headers. If either responseCode or headers are modified, all of them
    must be present.

    **EXPERIMENTAL**

    :param request_id: An id the client received in requestPaused event.
    :param response_code: *(Optional)* An HTTP response code. If absent, original response code will be used.
    :param response_phrase: *(Optional)* A textual representation of responseCode. If absent, a standard phrase matching responseCode is used.
    :param response_headers: *(Optional)* Response headers. If absent, original response headers will be used.
    :param binary_response_headers: *(Optional)* Alternative way of specifying response headers as a \0-separated series of name: value pairs. Prefer the above method unless you need to represent some non-UTF8 values that can't be transmitted over the protocol as text.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    if response_code is not None:
        params['responseCode'] = response_code
    if response_phrase is not None:
        params['responsePhrase'] = response_phrase
    if response_headers is not None:
        params['responseHeaders'] = [i.to_json() for i in response_headers]
    if binary_response_headers is not None:
        params['binaryResponseHeaders'] = binary_response_headers
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.continueResponse',
        'params': params,
    }
    json = yield cmd_dict


def get_response_body(
        request_id: RequestId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[str, bool]]:
    '''
    Causes the body of the response to be received from the server and
    returned as a single string. May only be issued for a request that
    is paused in the Response stage and is mutually exclusive with
    takeResponseBodyForInterceptionAsStream. Calling other methods that
    affect the request or disabling fetch domain before body is received
    results in an undefined behavior.
    Note that the response body is not available for redirects. Requests
    paused in the _redirect received_ state may be differentiated by
    ``responseCode`` and presence of ``location`` response header, see
    comments to ``requestPaused`` for details.

    :param request_id: Identifier for the intercepted request to get body for.
    :returns: A tuple with the following items:

        0. **body** - Response body.
        1. **base64Encoded** - True, if content was sent as base64.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.getResponseBody',
        'params': params,
    }
    json = yield cmd_dict
    return (
        str(json['body']),
        bool(json['base64Encoded'])
    )


def take_response_body_as_stream(
        request_id: RequestId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,io.StreamHandle]:
    '''
    Returns a handle to the stream representing the response body.
    The request must be paused in the HeadersReceived stage.
    Note that after this command the request can't be continued
    as is -- client either needs to cancel it or to provide the
    response body.
    The stream only supports sequential read, IO.read will fail if the position
    is specified.
    This method is mutually exclusive with getResponseBody.
    Calling other methods that affect the request or disabling fetch
    domain before body is received results in an undefined behavior.

    :param request_id:
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.takeResponseBodyAsStream',
        'params': params,
    }
    json = yield cmd_dict
    return io.StreamHandle.from_json(json['stream'])


@event_class('Fetch.requestPaused')
@dataclass
class RequestPaused:
    '''
    Issued when the domain is enabled and the request URL matches the
    specified filter. The request is paused until the client responds
    with one of continueRequest, failRequest or fulfillRequest.
    The stage of the request can be determined by presence of responseErrorReason
    and responseStatusCode -- the request is at the response stage if either
    of these fields is present and in the request stage otherwise.
    Redirect responses and subsequent requests are reported similarly to regular
    responses and requests. Redirect responses may be distinguished by the value
    of ``responseStatusCode`` (which is one of 301, 302, 303, 307, 308) along with
    presence of the ``location`` header. Requests resulting from a redirect will
    have ``redirectedRequestId`` field set.
    '''
    #: Each request the page makes will have a unique id.
    request_id: RequestId
    #: The details of the request.
    request: network.Request
    #: The id of the frame that initiated the request.
    frame_id: page.FrameId
    #: How the requested resource will be used.
    resource_type: network.ResourceType
    #: Response error if intercepted at response stage.
    response_error_reason: typing.Optional[network.ErrorReason]
    #: Response code if intercepted at response stage.
    response_status_code: typing.Optional[int]
    #: Response status text if intercepted at response stage.
    response_status_text: typing.Optional[str]
    #: Response headers if intercepted at the response stage.
    response_headers: typing.Optional[typing.List[HeaderEntry]]
    #: If the intercepted request had a corresponding Network.requestWillBeSent event fired for it,
    #: then this networkId will be the same as the requestId present in the requestWillBeSent event.
    network_id: typing.Optional[network.RequestId]
    #: If the request is due to a redirect response from the server, the id of the request that
    #: has caused the redirect.
    redirected_request_id: typing.Optional[RequestId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RequestPaused:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            request=network.Request.from_json(json['request']),
            frame_id=page.FrameId.from_json(json['frameId']),
            resource_type=network.ResourceType.from_json(json['resourceType']),
            response_error_reason=network.ErrorReason.from_json(json['responseErrorReason']) if 'responseErrorReason' in json else None,
            response_status_code=int(json['responseStatusCode']) if 'responseStatusCode' in json else None,
            response_status_text=str(json['responseStatusText']) if 'responseStatusText' in json else None,
            response_headers=[HeaderEntry.from_json(i) for i in json['responseHeaders']] if 'responseHeaders' in json else None,
            network_id=network.RequestId.from_json(json['networkId']) if 'networkId' in json else None,
            redirected_request_id=RequestId.from_json(json['redirectedRequestId']) if 'redirectedRequestId' in json else None
        )


@event_class('Fetch.authRequired')
@dataclass
class AuthRequired:
    '''
    Issued when the domain is enabled with handleAuthRequests set to true.
    The request is paused until client responds with continueWithAuth.
    '''
    #: Each request the page makes will have a unique id.
    request_id: RequestId
    #: The details of the request.
    request: network.Request
    #: The id of the frame that initiated the request.
    frame_id: page.FrameId
    #: How the requested resource will be used.
    resource_type: network.ResourceType
    #: Details of the Authorization Challenge encountered.
    #: If this is set, client should respond with continueRequest that
    #: contains AuthChallengeResponse.
    auth_challenge: AuthChallenge

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AuthRequired:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            request=network.Request.from_json(json['request']),
            frame_id=page.FrameId.from_json(json['frameId']),
            resource_type=network.ResourceType.from_json(json['resourceType']),
            auth_challenge=AuthChallenge.from_json(json['authChallenge'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\fetch.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Fetch
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import io
from . import network
from . import page


class RequestId(str):
    '''
    Unique request identifier.
    Note that this does not identify individual HTTP requests that are part of
    a network request.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> RequestId:
        return cls(json)

    def __repr__(self):
        return 'RequestId({})'.format(super().__repr__())


class RequestStage(enum.Enum):
    '''
    Stages of the request to handle. Request will intercept before the request is
    sent. Response will intercept after the response is received (but before response
    body is received).
    '''
    REQUEST = "Request"
    RESPONSE = "Response"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class RequestPattern:
    #: Wildcards (``'*'`` -> zero or more, ``'?'`` -> exactly one) are allowed. Escape character is
    #: backslash. Omitting is equivalent to ``"*"``.
    url_pattern: typing.Optional[str] = None

    #: If set, only requests for matching resource types will be intercepted.
    resource_type: typing.Optional[network.ResourceType] = None

    #: Stage at which to begin intercepting requests. Default is Request.
    request_stage: typing.Optional[RequestStage] = None

    def to_json(self):
        json = dict()
        if self.url_pattern is not None:
            json['urlPattern'] = self.url_pattern
        if self.resource_type is not None:
            json['resourceType'] = self.resource_type.to_json()
        if self.request_stage is not None:
            json['requestStage'] = self.request_stage.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url_pattern=str(json['urlPattern']) if 'urlPattern' in json else None,
            resource_type=network.ResourceType.from_json(json['resourceType']) if 'resourceType' in json else None,
            request_stage=RequestStage.from_json(json['requestStage']) if 'requestStage' in json else None,
        )


@dataclass
class HeaderEntry:
    '''
    Response HTTP header entry
    '''
    name: str

    value: str

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['value'] = self.value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            value=str(json['value']),
        )


@dataclass
class AuthChallenge:
    '''
    Authorization challenge for HTTP status code 401 or 407.
    '''
    #: Origin of the challenger.
    origin: str

    #: The authentication scheme used, such as basic or digest
    scheme: str

    #: The realm of the challenge. May be empty.
    realm: str

    #: Source of the authentication challenge.
    source: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['origin'] = self.origin
        json['scheme'] = self.scheme
        json['realm'] = self.realm
        if self.source is not None:
            json['source'] = self.source
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            origin=str(json['origin']),
            scheme=str(json['scheme']),
            realm=str(json['realm']),
            source=str(json['source']) if 'source' in json else None,
        )


@dataclass
class AuthChallengeResponse:
    '''
    Response to an AuthChallenge.
    '''
    #: The decision on what to do in response to the authorization challenge.  Default means
    #: deferring to the default behavior of the net stack, which will likely either the Cancel
    #: authentication or display a popup dialog box.
    response: str

    #: The username to provide, possibly empty. Should only be set if response is
    #: ProvideCredentials.
    username: typing.Optional[str] = None

    #: The password to provide, possibly empty. Should only be set if response is
    #: ProvideCredentials.
    password: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['response'] = self.response
        if self.username is not None:
            json['username'] = self.username
        if self.password is not None:
            json['password'] = self.password
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            response=str(json['response']),
            username=str(json['username']) if 'username' in json else None,
            password=str(json['password']) if 'password' in json else None,
        )


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables the fetch domain.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.disable',
    }
    json = yield cmd_dict


def enable(
        patterns: typing.Optional[typing.List[RequestPattern]] = None,
        handle_auth_requests: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables issuing of requestPaused events. A request will be paused until client
    calls one of failRequest, fulfillRequest or continueRequest/continueWithAuth.

    :param patterns: *(Optional)* If specified, only requests matching any of these patterns will produce fetchRequested event and will be paused until clients response. If not set, all requests will be affected.
    :param handle_auth_requests: *(Optional)* If true, authRequired events will be issued and requests will be paused expecting a call to continueWithAuth.
    '''
    params: T_JSON_DICT = dict()
    if patterns is not None:
        params['patterns'] = [i.to_json() for i in patterns]
    if handle_auth_requests is not None:
        params['handleAuthRequests'] = handle_auth_requests
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.enable',
        'params': params,
    }
    json = yield cmd_dict


def fail_request(
        request_id: RequestId,
        error_reason: network.ErrorReason
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Causes the request to fail with specified reason.

    :param request_id: An id the client received in requestPaused event.
    :param error_reason: Causes the request to fail with the given reason.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    params['errorReason'] = error_reason.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.failRequest',
        'params': params,
    }
    json = yield cmd_dict


def fulfill_request(
        request_id: RequestId,
        response_code: int,
        response_headers: typing.Optional[typing.List[HeaderEntry]] = None,
        binary_response_headers: typing.Optional[str] = None,
        body: typing.Optional[str] = None,
        response_phrase: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Provides response to the request.

    :param request_id: An id the client received in requestPaused event.
    :param response_code: An HTTP response code.
    :param response_headers: *(Optional)* Response headers.
    :param binary_response_headers: *(Optional)* Alternative way of specifying response headers as a \0-separated series of name: value pairs. Prefer the above method unless you need to represent some non-UTF8 values that can't be transmitted over the protocol as text.
    :param body: *(Optional)* A response body. If absent, original response body will be used if the request is intercepted at the response stage and empty body will be used if the request is intercepted at the request stage.
    :param response_phrase: *(Optional)* A textual representation of responseCode. If absent, a standard phrase matching responseCode is used.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    params['responseCode'] = response_code
    if response_headers is not None:
        params['responseHeaders'] = [i.to_json() for i in response_headers]
    if binary_response_headers is not None:
        params['binaryResponseHeaders'] = binary_response_headers
    if body is not None:
        params['body'] = body
    if response_phrase is not None:
        params['responsePhrase'] = response_phrase
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.fulfillRequest',
        'params': params,
    }
    json = yield cmd_dict


def continue_request(
        request_id: RequestId,
        url: typing.Optional[str] = None,
        method: typing.Optional[str] = None,
        post_data: typing.Optional[str] = None,
        headers: typing.Optional[typing.List[HeaderEntry]] = None,
        intercept_response: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Continues the request, optionally modifying some of its parameters.

    :param request_id: An id the client received in requestPaused event.
    :param url: *(Optional)* If set, the request url will be modified in a way that's not observable by page.
    :param method: *(Optional)* If set, the request method is overridden.
    :param post_data: *(Optional)* If set, overrides the post data in the request.
    :param headers: *(Optional)* If set, overrides the request headers. Note that the overrides do not extend to subsequent redirect hops, if a redirect happens. Another override may be applied to a different request produced by a redirect.
    :param intercept_response: **(EXPERIMENTAL)** *(Optional)* If set, overrides response interception behavior for this request.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    if url is not None:
        params['url'] = url
    if method is not None:
        params['method'] = method
    if post_data is not None:
        params['postData'] = post_data
    if headers is not None:
        params['headers'] = [i.to_json() for i in headers]
    if intercept_response is not None:
        params['interceptResponse'] = intercept_response
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.continueRequest',
        'params': params,
    }
    json = yield cmd_dict


def continue_with_auth(
        request_id: RequestId,
        auth_challenge_response: AuthChallengeResponse
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Continues a request supplying authChallengeResponse following authRequired event.

    :param request_id: An id the client received in authRequired event.
    :param auth_challenge_response: Response to  with an authChallenge.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    params['authChallengeResponse'] = auth_challenge_response.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.continueWithAuth',
        'params': params,
    }
    json = yield cmd_dict


def continue_response(
        request_id: RequestId,
        response_code: typing.Optional[int] = None,
        response_phrase: typing.Optional[str] = None,
        response_headers: typing.Optional[typing.List[HeaderEntry]] = None,
        binary_response_headers: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Continues loading of the paused response, optionally modifying the
    response headers. If either responseCode or headers are modified, all of them
    must be present.

    **EXPERIMENTAL**

    :param request_id: An id the client received in requestPaused event.
    :param response_code: *(Optional)* An HTTP response code. If absent, original response code will be used.
    :param response_phrase: *(Optional)* A textual representation of responseCode. If absent, a standard phrase matching responseCode is used.
    :param response_headers: *(Optional)* Response headers. If absent, original response headers will be used.
    :param binary_response_headers: *(Optional)* Alternative way of specifying response headers as a \0-separated series of name: value pairs. Prefer the above method unless you need to represent some non-UTF8 values that can't be transmitted over the protocol as text.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    if response_code is not None:
        params['responseCode'] = response_code
    if response_phrase is not None:
        params['responsePhrase'] = response_phrase
    if response_headers is not None:
        params['responseHeaders'] = [i.to_json() for i in response_headers]
    if binary_response_headers is not None:
        params['binaryResponseHeaders'] = binary_response_headers
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.continueResponse',
        'params': params,
    }
    json = yield cmd_dict


def get_response_body(
        request_id: RequestId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[str, bool]]:
    '''
    Causes the body of the response to be received from the server and
    returned as a single string. May only be issued for a request that
    is paused in the Response stage and is mutually exclusive with
    takeResponseBodyForInterceptionAsStream. Calling other methods that
    affect the request or disabling fetch domain before body is received
    results in an undefined behavior.
    Note that the response body is not available for redirects. Requests
    paused in the _redirect received_ state may be differentiated by
    ``responseCode`` and presence of ``location`` response header, see
    comments to ``requestPaused`` for details.

    :param request_id: Identifier for the intercepted request to get body for.
    :returns: A tuple with the following items:

        0. **body** - Response body.
        1. **base64Encoded** - True, if content was sent as base64.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.getResponseBody',
        'params': params,
    }
    json = yield cmd_dict
    return (
        str(json['body']),
        bool(json['base64Encoded'])
    )


def take_response_body_as_stream(
        request_id: RequestId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,io.StreamHandle]:
    '''
    Returns a handle to the stream representing the response body.
    The request must be paused in the HeadersReceived stage.
    Note that after this command the request can't be continued
    as is -- client either needs to cancel it or to provide the
    response body.
    The stream only supports sequential read, IO.read will fail if the position
    is specified.
    This method is mutually exclusive with getResponseBody.
    Calling other methods that affect the request or disabling fetch
    domain before body is received results in an undefined behavior.

    :param request_id:
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.takeResponseBodyAsStream',
        'params': params,
    }
    json = yield cmd_dict
    return io.StreamHandle.from_json(json['stream'])


@event_class('Fetch.requestPaused')
@dataclass
class RequestPaused:
    '''
    Issued when the domain is enabled and the request URL matches the
    specified filter. The request is paused until the client responds
    with one of continueRequest, failRequest or fulfillRequest.
    The stage of the request can be determined by presence of responseErrorReason
    and responseStatusCode -- the request is at the response stage if either
    of these fields is present and in the request stage otherwise.
    Redirect responses and subsequent requests are reported similarly to regular
    responses and requests. Redirect responses may be distinguished by the value
    of ``responseStatusCode`` (which is one of 301, 302, 303, 307, 308) along with
    presence of the ``location`` header. Requests resulting from a redirect will
    have ``redirectedRequestId`` field set.
    '''
    #: Each request the page makes will have a unique id.
    request_id: RequestId
    #: The details of the request.
    request: network.Request
    #: The id of the frame that initiated the request.
    frame_id: page.FrameId
    #: How the requested resource will be used.
    resource_type: network.ResourceType
    #: Response error if intercepted at response stage.
    response_error_reason: typing.Optional[network.ErrorReason]
    #: Response code if intercepted at response stage.
    response_status_code: typing.Optional[int]
    #: Response status text if intercepted at response stage.
    response_status_text: typing.Optional[str]
    #: Response headers if intercepted at the response stage.
    response_headers: typing.Optional[typing.List[HeaderEntry]]
    #: If the intercepted request had a corresponding Network.requestWillBeSent event fired for it,
    #: then this networkId will be the same as the requestId present in the requestWillBeSent event.
    network_id: typing.Optional[network.RequestId]
    #: If the request is due to a redirect response from the server, the id of the request that
    #: has caused the redirect.
    redirected_request_id: typing.Optional[RequestId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RequestPaused:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            request=network.Request.from_json(json['request']),
            frame_id=page.FrameId.from_json(json['frameId']),
            resource_type=network.ResourceType.from_json(json['resourceType']),
            response_error_reason=network.ErrorReason.from_json(json['responseErrorReason']) if 'responseErrorReason' in json else None,
            response_status_code=int(json['responseStatusCode']) if 'responseStatusCode' in json else None,
            response_status_text=str(json['responseStatusText']) if 'responseStatusText' in json else None,
            response_headers=[HeaderEntry.from_json(i) for i in json['responseHeaders']] if 'responseHeaders' in json else None,
            network_id=network.RequestId.from_json(json['networkId']) if 'networkId' in json else None,
            redirected_request_id=RequestId.from_json(json['redirectedRequestId']) if 'redirectedRequestId' in json else None
        )


@event_class('Fetch.authRequired')
@dataclass
class AuthRequired:
    '''
    Issued when the domain is enabled with handleAuthRequests set to true.
    The request is paused until client responds with continueWithAuth.
    '''
    #: Each request the page makes will have a unique id.
    request_id: RequestId
    #: The details of the request.
    request: network.Request
    #: The id of the frame that initiated the request.
    frame_id: page.FrameId
    #: How the requested resource will be used.
    resource_type: network.ResourceType
    #: Details of the Authorization Challenge encountered.
    #: If this is set, client should respond with continueRequest that
    #: contains AuthChallengeResponse.
    auth_challenge: AuthChallenge

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AuthRequired:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            request=network.Request.from_json(json['request']),
            frame_id=page.FrameId.from_json(json['frameId']),
            resource_type=network.ResourceType.from_json(json['resourceType']),
            auth_challenge=AuthChallenge.from_json(json['authChallenge'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\fetch.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Fetch
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import io
from . import network
from . import page


class RequestId(str):
    '''
    Unique request identifier.
    Note that this does not identify individual HTTP requests that are part of
    a network request.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> RequestId:
        return cls(json)

    def __repr__(self):
        return 'RequestId({})'.format(super().__repr__())


class RequestStage(enum.Enum):
    '''
    Stages of the request to handle. Request will intercept before the request is
    sent. Response will intercept after the response is received (but before response
    body is received).
    '''
    REQUEST = "Request"
    RESPONSE = "Response"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class RequestPattern:
    #: Wildcards (``'*'`` -> zero or more, ``'?'`` -> exactly one) are allowed. Escape character is
    #: backslash. Omitting is equivalent to ``"*"``.
    url_pattern: typing.Optional[str] = None

    #: If set, only requests for matching resource types will be intercepted.
    resource_type: typing.Optional[network.ResourceType] = None

    #: Stage at which to begin intercepting requests. Default is Request.
    request_stage: typing.Optional[RequestStage] = None

    def to_json(self):
        json = dict()
        if self.url_pattern is not None:
            json['urlPattern'] = self.url_pattern
        if self.resource_type is not None:
            json['resourceType'] = self.resource_type.to_json()
        if self.request_stage is not None:
            json['requestStage'] = self.request_stage.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url_pattern=str(json['urlPattern']) if 'urlPattern' in json else None,
            resource_type=network.ResourceType.from_json(json['resourceType']) if 'resourceType' in json else None,
            request_stage=RequestStage.from_json(json['requestStage']) if 'requestStage' in json else None,
        )


@dataclass
class HeaderEntry:
    '''
    Response HTTP header entry
    '''
    name: str

    value: str

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['value'] = self.value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            value=str(json['value']),
        )


@dataclass
class AuthChallenge:
    '''
    Authorization challenge for HTTP status code 401 or 407.
    '''
    #: Origin of the challenger.
    origin: str

    #: The authentication scheme used, such as basic or digest
    scheme: str

    #: The realm of the challenge. May be empty.
    realm: str

    #: Source of the authentication challenge.
    source: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['origin'] = self.origin
        json['scheme'] = self.scheme
        json['realm'] = self.realm
        if self.source is not None:
            json['source'] = self.source
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            origin=str(json['origin']),
            scheme=str(json['scheme']),
            realm=str(json['realm']),
            source=str(json['source']) if 'source' in json else None,
        )


@dataclass
class AuthChallengeResponse:
    '''
    Response to an AuthChallenge.
    '''
    #: The decision on what to do in response to the authorization challenge.  Default means
    #: deferring to the default behavior of the net stack, which will likely either the Cancel
    #: authentication or display a popup dialog box.
    response: str

    #: The username to provide, possibly empty. Should only be set if response is
    #: ProvideCredentials.
    username: typing.Optional[str] = None

    #: The password to provide, possibly empty. Should only be set if response is
    #: ProvideCredentials.
    password: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['response'] = self.response
        if self.username is not None:
            json['username'] = self.username
        if self.password is not None:
            json['password'] = self.password
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            response=str(json['response']),
            username=str(json['username']) if 'username' in json else None,
            password=str(json['password']) if 'password' in json else None,
        )


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables the fetch domain.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.disable',
    }
    json = yield cmd_dict


def enable(
        patterns: typing.Optional[typing.List[RequestPattern]] = None,
        handle_auth_requests: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables issuing of requestPaused events. A request will be paused until client
    calls one of failRequest, fulfillRequest or continueRequest/continueWithAuth.

    :param patterns: *(Optional)* If specified, only requests matching any of these patterns will produce fetchRequested event and will be paused until clients response. If not set, all requests will be affected.
    :param handle_auth_requests: *(Optional)* If true, authRequired events will be issued and requests will be paused expecting a call to continueWithAuth.
    '''
    params: T_JSON_DICT = dict()
    if patterns is not None:
        params['patterns'] = [i.to_json() for i in patterns]
    if handle_auth_requests is not None:
        params['handleAuthRequests'] = handle_auth_requests
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.enable',
        'params': params,
    }
    json = yield cmd_dict


def fail_request(
        request_id: RequestId,
        error_reason: network.ErrorReason
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Causes the request to fail with specified reason.

    :param request_id: An id the client received in requestPaused event.
    :param error_reason: Causes the request to fail with the given reason.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    params['errorReason'] = error_reason.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.failRequest',
        'params': params,
    }
    json = yield cmd_dict


def fulfill_request(
        request_id: RequestId,
        response_code: int,
        response_headers: typing.Optional[typing.List[HeaderEntry]] = None,
        binary_response_headers: typing.Optional[str] = None,
        body: typing.Optional[str] = None,
        response_phrase: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Provides response to the request.

    :param request_id: An id the client received in requestPaused event.
    :param response_code: An HTTP response code.
    :param response_headers: *(Optional)* Response headers.
    :param binary_response_headers: *(Optional)* Alternative way of specifying response headers as a \0-separated series of name: value pairs. Prefer the above method unless you need to represent some non-UTF8 values that can't be transmitted over the protocol as text.
    :param body: *(Optional)* A response body. If absent, original response body will be used if the request is intercepted at the response stage and empty body will be used if the request is intercepted at the request stage.
    :param response_phrase: *(Optional)* A textual representation of responseCode. If absent, a standard phrase matching responseCode is used.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    params['responseCode'] = response_code
    if response_headers is not None:
        params['responseHeaders'] = [i.to_json() for i in response_headers]
    if binary_response_headers is not None:
        params['binaryResponseHeaders'] = binary_response_headers
    if body is not None:
        params['body'] = body
    if response_phrase is not None:
        params['responsePhrase'] = response_phrase
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.fulfillRequest',
        'params': params,
    }
    json = yield cmd_dict


def continue_request(
        request_id: RequestId,
        url: typing.Optional[str] = None,
        method: typing.Optional[str] = None,
        post_data: typing.Optional[str] = None,
        headers: typing.Optional[typing.List[HeaderEntry]] = None,
        intercept_response: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Continues the request, optionally modifying some of its parameters.

    :param request_id: An id the client received in requestPaused event.
    :param url: *(Optional)* If set, the request url will be modified in a way that's not observable by page.
    :param method: *(Optional)* If set, the request method is overridden.
    :param post_data: *(Optional)* If set, overrides the post data in the request.
    :param headers: *(Optional)* If set, overrides the request headers. Note that the overrides do not extend to subsequent redirect hops, if a redirect happens. Another override may be applied to a different request produced by a redirect.
    :param intercept_response: **(EXPERIMENTAL)** *(Optional)* If set, overrides response interception behavior for this request.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    if url is not None:
        params['url'] = url
    if method is not None:
        params['method'] = method
    if post_data is not None:
        params['postData'] = post_data
    if headers is not None:
        params['headers'] = [i.to_json() for i in headers]
    if intercept_response is not None:
        params['interceptResponse'] = intercept_response
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.continueRequest',
        'params': params,
    }
    json = yield cmd_dict


def continue_with_auth(
        request_id: RequestId,
        auth_challenge_response: AuthChallengeResponse
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Continues a request supplying authChallengeResponse following authRequired event.

    :param request_id: An id the client received in authRequired event.
    :param auth_challenge_response: Response to  with an authChallenge.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    params['authChallengeResponse'] = auth_challenge_response.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.continueWithAuth',
        'params': params,
    }
    json = yield cmd_dict


def continue_response(
        request_id: RequestId,
        response_code: typing.Optional[int] = None,
        response_phrase: typing.Optional[str] = None,
        response_headers: typing.Optional[typing.List[HeaderEntry]] = None,
        binary_response_headers: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Continues loading of the paused response, optionally modifying the
    response headers. If either responseCode or headers are modified, all of them
    must be present.

    **EXPERIMENTAL**

    :param request_id: An id the client received in requestPaused event.
    :param response_code: *(Optional)* An HTTP response code. If absent, original response code will be used.
    :param response_phrase: *(Optional)* A textual representation of responseCode. If absent, a standard phrase matching responseCode is used.
    :param response_headers: *(Optional)* Response headers. If absent, original response headers will be used.
    :param binary_response_headers: *(Optional)* Alternative way of specifying response headers as a \0-separated series of name: value pairs. Prefer the above method unless you need to represent some non-UTF8 values that can't be transmitted over the protocol as text.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    if response_code is not None:
        params['responseCode'] = response_code
    if response_phrase is not None:
        params['responsePhrase'] = response_phrase
    if response_headers is not None:
        params['responseHeaders'] = [i.to_json() for i in response_headers]
    if binary_response_headers is not None:
        params['binaryResponseHeaders'] = binary_response_headers
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.continueResponse',
        'params': params,
    }
    json = yield cmd_dict


def get_response_body(
        request_id: RequestId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[str, bool]]:
    '''
    Causes the body of the response to be received from the server and
    returned as a single string. May only be issued for a request that
    is paused in the Response stage and is mutually exclusive with
    takeResponseBodyForInterceptionAsStream. Calling other methods that
    affect the request or disabling fetch domain before body is received
    results in an undefined behavior.
    Note that the response body is not available for redirects. Requests
    paused in the _redirect received_ state may be differentiated by
    ``responseCode`` and presence of ``location`` response header, see
    comments to ``requestPaused`` for details.

    :param request_id: Identifier for the intercepted request to get body for.
    :returns: A tuple with the following items:

        0. **body** - Response body.
        1. **base64Encoded** - True, if content was sent as base64.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.getResponseBody',
        'params': params,
    }
    json = yield cmd_dict
    return (
        str(json['body']),
        bool(json['base64Encoded'])
    )


def take_response_body_as_stream(
        request_id: RequestId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,io.StreamHandle]:
    '''
    Returns a handle to the stream representing the response body.
    The request must be paused in the HeadersReceived stage.
    Note that after this command the request can't be continued
    as is -- client either needs to cancel it or to provide the
    response body.
    The stream only supports sequential read, IO.read will fail if the position
    is specified.
    This method is mutually exclusive with getResponseBody.
    Calling other methods that affect the request or disabling fetch
    domain before body is received results in an undefined behavior.

    :param request_id:
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Fetch.takeResponseBodyAsStream',
        'params': params,
    }
    json = yield cmd_dict
    return io.StreamHandle.from_json(json['stream'])


@event_class('Fetch.requestPaused')
@dataclass
class RequestPaused:
    '''
    Issued when the domain is enabled and the request URL matches the
    specified filter. The request is paused until the client responds
    with one of continueRequest, failRequest or fulfillRequest.
    The stage of the request can be determined by presence of responseErrorReason
    and responseStatusCode -- the request is at the response stage if either
    of these fields is present and in the request stage otherwise.
    Redirect responses and subsequent requests are reported similarly to regular
    responses and requests. Redirect responses may be distinguished by the value
    of ``responseStatusCode`` (which is one of 301, 302, 303, 307, 308) along with
    presence of the ``location`` header. Requests resulting from a redirect will
    have ``redirectedRequestId`` field set.
    '''
    #: Each request the page makes will have a unique id.
    request_id: RequestId
    #: The details of the request.
    request: network.Request
    #: The id of the frame that initiated the request.
    frame_id: page.FrameId
    #: How the requested resource will be used.
    resource_type: network.ResourceType
    #: Response error if intercepted at response stage.
    response_error_reason: typing.Optional[network.ErrorReason]
    #: Response code if intercepted at response stage.
    response_status_code: typing.Optional[int]
    #: Response status text if intercepted at response stage.
    response_status_text: typing.Optional[str]
    #: Response headers if intercepted at the response stage.
    response_headers: typing.Optional[typing.List[HeaderEntry]]
    #: If the intercepted request had a corresponding Network.requestWillBeSent event fired for it,
    #: then this networkId will be the same as the requestId present in the requestWillBeSent event.
    network_id: typing.Optional[network.RequestId]
    #: If the request is due to a redirect response from the server, the id of the request that
    #: has caused the redirect.
    redirected_request_id: typing.Optional[RequestId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RequestPaused:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            request=network.Request.from_json(json['request']),
            frame_id=page.FrameId.from_json(json['frameId']),
            resource_type=network.ResourceType.from_json(json['resourceType']),
            response_error_reason=network.ErrorReason.from_json(json['responseErrorReason']) if 'responseErrorReason' in json else None,
            response_status_code=int(json['responseStatusCode']) if 'responseStatusCode' in json else None,
            response_status_text=str(json['responseStatusText']) if 'responseStatusText' in json else None,
            response_headers=[HeaderEntry.from_json(i) for i in json['responseHeaders']] if 'responseHeaders' in json else None,
            network_id=network.RequestId.from_json(json['networkId']) if 'networkId' in json else None,
            redirected_request_id=RequestId.from_json(json['redirectedRequestId']) if 'redirectedRequestId' in json else None
        )


@event_class('Fetch.authRequired')
@dataclass
class AuthRequired:
    '''
    Issued when the domain is enabled with handleAuthRequests set to true.
    The request is paused until client responds with continueWithAuth.
    '''
    #: Each request the page makes will have a unique id.
    request_id: RequestId
    #: The details of the request.
    request: network.Request
    #: The id of the frame that initiated the request.
    frame_id: page.FrameId
    #: How the requested resource will be used.
    resource_type: network.ResourceType
    #: Details of the Authorization Challenge encountered.
    #: If this is set, client should respond with continueRequest that
    #: contains AuthChallengeResponse.
    auth_challenge: AuthChallenge

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AuthRequired:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            request=network.Request.from_json(json['request']),
            frame_id=page.FrameId.from_json(json['frameId']),
            resource_type=network.ResourceType.from_json(json['resourceType']),
            auth_challenge=AuthChallenge.from_json(json['authChallenge'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\llama\llama_inputs.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache

from onnxruntime import InferenceSession, OrtValue


# Get position_ids from attention_mask
def get_position_ids(attention_mask: torch.Tensor, use_past_kv: bool):
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    if use_past_kv:
        # Shape: (batch_size, 1)
        position_ids = position_ids[:, -1].unsqueeze(-1)

    # Shape: (batch_size, sequence_length)
    return position_ids


# Inputs for first pass to get initial past_key_values
#   input_ids: (batch_size, sequence_length)
#   attention_mask: (batch_size, sequence_length)
#   position_ids: (batch_size, sequence_length)
def get_sample_inputs(
    config: AutoConfig,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    engine: str = "pt",
    return_dict: bool = False,
):
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seq_len), dtype=torch.int64)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int64)
    position_ids = get_position_ids(attention_mask, use_past_kv=False)

    # Convert inputs to NumPy (for ORT) or send to device (for PyTorch)
    input_ids = input_ids.numpy() if engine == "ort" else input_ids.to(device)
    attention_mask = attention_mask.numpy() if engine == "ort" else attention_mask.to(device)
    position_ids = position_ids.numpy() if engine == "ort" else position_ids.to(device)

    if not return_dict:
        # For export
        return (input_ids, attention_mask, position_ids)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    return inputs


# Inputs for subsequent passes with past_key_values
#   input_ids: (batch_size, 1)
#   attention_mask: (batch_size, past_sequence_length + 1)
#   position_ids: (batch_size, 1)
#   past_key: (batch_size, num_heads, past_sequence_length, head_size)
#   past_value: (batch_size, num_heads, past_sequence_length, head_size)
def get_sample_with_past_kv_inputs(
    config: AutoConfig,
    device: torch.device,
    batch_size: int,
    past_seq_len: int,
    use_fp16: bool = False,
    engine: str = "pt",
    return_dict: bool = False,
    world_size: int = 1,
):
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, 1), dtype=torch.int64)
    attention_mask = torch.ones(batch_size, past_seq_len + 1, dtype=torch.int64)
    # position_ids is of shape (batch_size, 1)
    position_ids = get_position_ids(attention_mask, use_past_kv=True)
    past_kv = get_past_kv_inputs(config, batch_size, past_seq_len, use_fp16, world_size=world_size)

    # Convert inputs to NumPy (for ORT) or send to device (for PyTorch)
    input_ids = input_ids.numpy() if engine == "ort" else input_ids.to(device)
    attention_mask = attention_mask.numpy() if engine == "ort" else attention_mask.to(device)
    position_ids = position_ids.numpy() if engine == "ort" else position_ids.to(device)
    past_kv = (
        flatten_past_kv_inputs(past_kv) if engine == "ort" else [(kv[0].to(device), kv[1].to(device)) for kv in past_kv]
    )

    if not return_dict:
        # For export
        assert isinstance(past_kv, list)
        return (input_ids, attention_mask, position_ids, past_kv)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    if engine == "ort":
        assert isinstance(past_kv, dict)
        inputs.update(past_kv)
    else:
        assert isinstance(past_kv, list)
        inputs["past_key_values"] = past_kv

    return inputs


# Inputs for all passes with past_key_values
#   input_ids: (batch_size, sequence_length)
#   attention_mask: (batch_size, past_sequence_length + sequence_length)
#   position_ids: (batch_size, sequence_length)
#   past_key: (batch_size, num_heads, kv_sequence_length, head_size)
#      For models with GQA, kv_sequence_length = max_sequence_length
#      For models without GQA, kv_sequence_length = past_sequence_length
#   past_value: (batch_size, num_heads, kv_sequence_length, head_size)
#      For models with GQA, kv_sequence_length = max_sequence_length
#      For models without GQA, kv_sequence_length = past_sequence_length
def get_merged_sample_with_past_kv_inputs(
    config: AutoConfig,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    past_seq_len: int,
    max_seq_len: int,
    use_fp16: bool = False,
    use_buffer_share: bool = False,
    engine: str = "pt",
    return_dict: bool = False,
    world_size: int = 1,
):
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seq_len), dtype=torch.int64)
    attention_mask = torch.ones(batch_size, past_seq_len + seq_len, dtype=torch.int64)
    # position_ids is of shape (batch_size, seq_len) for prompt generation, (batch_size, 1) for token generation
    position_ids = get_position_ids(attention_mask, use_past_kv=(past_seq_len != 0))
    past_kv = get_past_kv_inputs(config, batch_size, past_seq_len, use_fp16, world_size=world_size)

    # Convert inputs to NumPy (for ORT) or send to device (for PyTorch)
    input_ids = input_ids.numpy() if engine == "ort" else input_ids.to(device)
    attention_mask = attention_mask.numpy() if engine == "ort" else attention_mask.to(device)
    position_ids = position_ids.numpy() if engine == "ort" else position_ids.to(device)
    past_kv = (
        flatten_past_kv_inputs(past_kv) if engine == "ort" else [(kv[0].to(device), kv[1].to(device)) for kv in past_kv]
    )

    if not return_dict:
        # For export
        assert isinstance(past_kv, list)
        return (input_ids, attention_mask, position_ids, past_kv)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    if engine == "ort":
        assert isinstance(past_kv, dict)
        inputs.update(past_kv)

        if use_buffer_share:
            inputs = enable_past_present_share_buffer(inputs, past_seq_len, max_seq_len)

    else:
        assert isinstance(past_kv, list)
        inputs["past_key_values"] = past_kv

    return inputs


# Inputs for Microsoft export from https://github.com/microsoft/Llama-2-Onnx
def get_msft_sample_inputs(
    config: AutoConfig,
    batch_size: int,
    past_seq_len: int,
    seq_len: int,
    max_seq_len: int,
    use_fp16: bool,
    use_buffer_share: bool,
    split_kv: bool,
):
    np_dtype = np.float16 if use_fp16 else np.float32
    head_size = config.hidden_size // config.num_attention_heads

    if not split_kv:
        ort_inputs = {
            "x": np.random.rand(batch_size, seq_len, config.hidden_size).astype(np_dtype),
            "attn_mask": (-10000.0 * np.triu(np.ones((batch_size, max_seq_len, max_seq_len)), k=1)).astype(np_dtype),
            "k_cache": np.random.rand(
                batch_size, config.num_hidden_layers, past_seq_len, config.num_attention_heads, head_size
            ).astype(np_dtype),
            "v_cache": np.random.rand(
                batch_size, config.num_hidden_layers, past_seq_len, config.num_attention_heads, head_size
            ).astype(np_dtype),
            "pos": np.array(past_seq_len, dtype=np.int64),
        }
    else:
        ort_inputs = {
            "x": np.random.rand(batch_size, seq_len, config.hidden_size).astype(np_dtype),
            "attn_mask": (np.triu(np.ones((batch_size, max_seq_len, max_seq_len), dtype=np.int32), k=1) - 1).astype(
                np.int32
            ),
            "pos": np.array(past_seq_len, dtype=np.int64),
        }
        for i in range(config.num_hidden_layers):
            ort_inputs.update(
                {
                    f"k_{i}_cache": np.random.rand(
                        batch_size, config.num_attention_heads, past_seq_len, head_size
                    ).astype(np_dtype),
                    f"v_{i}_cache": np.random.rand(
                        batch_size, config.num_attention_heads, past_seq_len, head_size
                    ).astype(np_dtype),
                }
            )

        if use_buffer_share:
            ort_inputs = enable_past_present_share_buffer(ort_inputs, past_seq_len, max_seq_len)

    return ort_inputs


# Create past_key_values
# Each is of shape (batch_size, num_heads, past_sequence_length, head_size)
def get_past_kv_inputs(config: AutoConfig, batch_size: int, past_seq_len: int, use_fp16: bool, world_size: int = 1):
    num_heads = config.num_key_value_heads // world_size
    head_size = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    past_kv = [
        (
            torch.rand(batch_size, num_heads, past_seq_len, head_size, dtype=torch_dtype),
            torch.rand(batch_size, num_heads, past_seq_len, head_size, dtype=torch_dtype),
        )
        for _ in range(config.num_hidden_layers)
    ]
    return past_kv


# Convert list of past_key_values to dict of past_key and past_value
def flatten_past_kv_inputs(past_key_values: list[tuple[torch.Tensor, torch.Tensor]]):
    past_kv = {}
    for i, (past_k, past_v) in enumerate(past_key_values):
        if isinstance(past_key_values, DynamicCache):
            past_kv[f"past_key_values_key_cache_{i}"] = past_k.detach().cpu().numpy()
            past_kv[f"past_key_values_value_cache_{i}"] = past_v.detach().cpu().numpy()
        else:
            past_kv[f"past_key_values.{i}.key"] = past_k.detach().cpu().numpy()
            past_kv[f"past_key_values.{i}.value"] = past_v.detach().cpu().numpy()
    return past_kv


# Format PyTorch inputs to ONNX Runtime inputs
def convert_inputs_for_ort(
    pt_inputs: dict,
    use_buffer_share: bool = False,
    past_seq_len: int = 0,
    max_seq_len: int = 2048,
):
    ort_inputs = {}
    for k, v in pt_inputs.items():
        if isinstance(v, np.ndarray):
            ort_inputs[k] = v
        elif k == "past_key_values":
            ort_inputs.update(flatten_past_kv_inputs(v))
        else:
            ort_inputs[k] = v.detach().cpu().numpy()

    # Reshape KV caches if using past-present-share-buffer
    if use_buffer_share:
        ort_inputs = enable_past_present_share_buffer(ort_inputs, past_seq_len, max_seq_len)

    return ort_inputs


# Re-allocate KV caches from (batch_size, num_heads, past_sequence_length, head_size) to
# (batch_size, num_heads, max_sequence_length, head_size) for past-present buffer sharing
def enable_past_present_share_buffer(ort_inputs: dict, past_seq_len: int, max_seq_len: int):
    for k, v in ort_inputs.items():
        # Allocate new buffers with max_sequence_length for GQA
        if "cache" in k or "past_key_values" in k:
            # Copy v (BxSxPxH) into new_v (BxSxMxH)
            batch_size, num_heads, _, head_size = v.shape
            new_v = np.zeros((batch_size, num_heads, max_seq_len, head_size), dtype=v.dtype)
            new_v[:batch_size, :num_heads, :past_seq_len, :head_size] = v
            ort_inputs[k] = new_v
    return ort_inputs


# Verify ONNX Runtime inputs with model
def verify_ort_inputs(model: InferenceSession, ort_inputs: dict):
    # Check that all model inputs will be provided
    model_inputs = {model_input.name for model_input in model.get_inputs()}
    user_inputs = set(ort_inputs.keys())
    missing_inputs = model_inputs - user_inputs
    if len(missing_inputs):
        print(f"The following model inputs are missing: {missing_inputs}")
        raise Exception("There are missing inputs to the model. Please add them and try again.")

    # Remove unnecessary inputs from model inputs
    unnecessary_inputs = user_inputs - model_inputs
    if len(unnecessary_inputs):
        for unnecessary_input in unnecessary_inputs:
            del ort_inputs[unnecessary_input]

    return ort_inputs


# Add IO bindings for execution providers using OrtValue
# Use when you need to run inference once or twice to save memory
def add_io_bindings_as_ortvalues(
    model: InferenceSession,
    ort_inputs: dict,
    device: str,
    device_id: int,
    use_buffer_share: bool,
    kv_cache_ortvalues: dict,
):
    io_binding = model.io_binding()

    model_inputs = {i.name for i in model.get_inputs()}
    for k, v in ort_inputs.items():
        # Use this check to handle scenarios such as INT4 CUDA and FP16 CUDA models with
        # GQA + RotaryEmbedding fusion where `position_ids` is removed as an ONNX model input
        # but `position_ids` is used as a PyTorch model input
        if k not in model_inputs:
            continue

        # Bind OrtValue inputs to device
        if use_buffer_share and ("cache" in k or "past_key_values" in k):
            if k not in kv_cache_ortvalues:
                v_device = OrtValue.ortvalue_from_numpy(v, device_type=device, device_id=device_id)
                io_binding.bind_ortvalue_input(k, v_device)
                kv_cache_ortvalues[k] = v_device
            else:
                kv_cache_ortvalues[k].update_inplace(v)
                io_binding.bind_ortvalue_input(k, kv_cache_ortvalues[k])
        else:
            v_device = OrtValue.ortvalue_from_numpy(v, device_type=device, device_id=device_id)
            io_binding.bind_ortvalue_input(k, v_device)

    for output in model.get_outputs():
        name = output.name
        if use_buffer_share and ("out" in name or "present" in name):
            # Bind present KV cache outputs to past KV cache inputs in order to buffer share
            input_name = name.replace("out", "cache").replace("present", "past_key_values")
            io_binding.bind_ortvalue_output(name, kv_cache_ortvalues[input_name])
        else:
            io_binding.bind_output(name, device_type=device, device_id=device_id)

    return io_binding, kv_cache_ortvalues


# Add IO bindings for execution providers using PyTorch tensors
# Use when you need to run inference many times
def add_io_bindings_as_tensors(
    model: InferenceSession, inputs: dict, outputs: dict, use_fp16: bool, use_buffer_share: bool
):
    # Verify model inputs
    inputs = verify_ort_inputs(model, inputs)

    device = None
    pt_to_np = {
        "torch.int32": np.int32,
        "torch.int64": np.int64,
        "torch.float16": np.float16,
        "torch.float32": np.float32,
    }

    # Bind inputs/outputs to IO binding
    io_binding = model.io_binding()
    for k, v in inputs.items():
        io_binding.bind_input(
            name=k,
            device_type=v.device.type,
            device_id=0 if v.device.type == "cpu" else v.device.index,
            element_type=pt_to_np[repr(v.dtype)],
            shape=tuple(v.shape),
            buffer_ptr=v.data_ptr(),
        )
        device = v.device

    for output in model.get_outputs():
        name = output.name
        # Bind KV cache outputs to KV cache inputs
        v = (
            inputs[name.replace("present", "past_key_values")]
            if use_buffer_share and "present" in name
            else outputs[name]
        )
        io_binding.bind_output(
            name=name,
            device_type=device.type,
            device_id=0 if device.type == "cpu" else device.index,
            element_type=(np.float16 if use_fp16 else np.float32),
            shape=tuple(v.shape),
            buffer_ptr=v.data_ptr(),
        )

    return io_binding


# Get actual inputs when using real data (instead of sample data) and initialize outputs
def get_initial_inputs_and_outputs(
    config: AutoConfig,
    tokenizer: AutoTokenizer,
    requested_length: int,
    prompt: list[str],
    device: torch.device,
    use_fp16: bool,
    use_buffer_share: bool,
    engine: str,
):
    tokenizer.pad_token = tokenizer.eos_token
    encodings_dict = tokenizer.batch_encode_plus(prompt, padding=True)
    torch_dtype = torch.float16 if use_fp16 else torch.float32

    # input_ids:      pad token id is 0
    # attention_mask: pad token id is 0
    # position_ids:   pad token id is 1
    input_ids = torch.tensor(encodings_dict["input_ids"], device=device, dtype=torch.int64)
    attention_mask = torch.tensor(encodings_dict["attention_mask"], device=device, dtype=torch.int64)
    position_ids = get_position_ids(attention_mask, use_past_kv=False)

    # Check if tokenized prompt length matches the requested prompt length
    tokenized_length = input_ids.shape[-1]
    if tokenized_length > requested_length:
        # Shorten the inputs from (batch_size, tokenized_length) to (batch_size, requested_length)
        input_ids = input_ids[:, :requested_length]
        attention_mask = attention_mask[:, :requested_length]
        position_ids = get_position_ids(attention_mask, use_past_kv=False)
    elif tokenized_length < requested_length:
        # Lengthen the inputs from (batch_size, tokenized_length) to (batch_size, requested_length)
        input_ids_first_col = input_ids[:, 0].unsqueeze(0).T
        attention_mask_first_col = attention_mask[:, 0].unsqueeze(0).T
        for _ in range(requested_length - tokenized_length):
            input_ids = torch.hstack((input_ids_first_col, input_ids))
            attention_mask = torch.hstack((attention_mask_first_col, attention_mask))
        position_ids = get_position_ids(attention_mask, use_past_kv=False)

    tokenized_length = input_ids.shape[-1]
    assert tokenized_length == requested_length

    # Create inputs
    inputs = {
        "input_ids": input_ids.contiguous() if engine == "ort" else input_ids,
        "attention_mask": attention_mask.contiguous() if engine == "ort" else attention_mask,
        "position_ids": position_ids.contiguous() if engine == "ort" else position_ids,
    }
    if engine != "ort":
        inputs["past_key_values"] = []

    # Get shape of KV cache inputs
    batch_size, sequence_length = input_ids.shape
    max_sequence_length = config.max_position_embeddings
    num_heads = config.num_key_value_heads
    head_size = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads

    # Create KV cache inputs
    for i in range(config.num_hidden_layers):
        past_key = torch.zeros(
            batch_size,
            num_heads,
            max_sequence_length if use_buffer_share else 0,
            head_size,
            device=device,
            dtype=torch_dtype,
        )
        past_value = torch.zeros(
            batch_size,
            num_heads,
            max_sequence_length if use_buffer_share else 0,
            head_size,
            device=device,
            dtype=torch_dtype,
        )
        if engine == "ort":
            inputs.update(
                {
                    f"past_key_values.{i}.key": past_key.contiguous(),
                    f"past_key_values.{i}.value": past_value.contiguous(),
                }
            )
        else:
            inputs["past_key_values"].append((past_key, past_value))

    outputs = None
    if engine == "ort":
        # Create outputs
        logits = torch.zeros(batch_size, sequence_length, config.vocab_size, device=device, dtype=torch_dtype)
        outputs = {"logits": logits.contiguous()}
        if not use_buffer_share:
            for i in range(config.num_hidden_layers):
                present_key = torch.zeros(
                    batch_size, num_heads, sequence_length, head_size, device=device, dtype=torch_dtype
                )
                present_value = torch.zeros(
                    batch_size, num_heads, sequence_length, head_size, device=device, dtype=torch_dtype
                )
                outputs.update(
                    {f"present.{i}.key": present_key.contiguous(), f"present.{i}.value": present_value.contiguous()}
                )

    return inputs, outputs

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\util\ssl_.py ===
from __future__ import absolute_import

import hashlib
import hmac
import os
import sys
import warnings
from binascii import hexlify, unhexlify

from ..exceptions import (
    InsecurePlatformWarning,
    ProxySchemeUnsupported,
    SNIMissingWarning,
    SSLError,
)
from ..packages import six
from .url import BRACELESS_IPV6_ADDRZ_RE, IPV4_RE

SSLContext = None
SSLTransport = None
HAS_SNI = False
IS_PYOPENSSL = False
IS_SECURETRANSPORT = False
ALPN_PROTOCOLS = ["http/1.1"]

# Maps the length of a digest to a possible hash function producing this digest
HASHFUNC_MAP = {
    length: getattr(hashlib, algorithm, None)
    for length, algorithm in ((32, "md5"), (40, "sha1"), (64, "sha256"))
}


def _const_compare_digest_backport(a, b):
    """
    Compare two digests of equal length in constant time.

    The digests must be of type str/bytes.
    Returns True if the digests match, and False otherwise.
    """
    result = abs(len(a) - len(b))
    for left, right in zip(bytearray(a), bytearray(b)):
        result |= left ^ right
    return result == 0


_const_compare_digest = getattr(hmac, "compare_digest", _const_compare_digest_backport)

try:  # Test for SSL features
    import ssl
    from ssl import CERT_REQUIRED, wrap_socket
except ImportError:
    pass

try:
    from ssl import HAS_SNI  # Has SNI?
except ImportError:
    pass

try:
    from .ssltransport import SSLTransport
except ImportError:
    pass


try:  # Platform-specific: Python 3.6
    from ssl import PROTOCOL_TLS

    PROTOCOL_SSLv23 = PROTOCOL_TLS
except ImportError:
    try:
        from ssl import PROTOCOL_SSLv23 as PROTOCOL_TLS

        PROTOCOL_SSLv23 = PROTOCOL_TLS
    except ImportError:
        PROTOCOL_SSLv23 = PROTOCOL_TLS = 2

try:
    from ssl import PROTOCOL_TLS_CLIENT
except ImportError:
    PROTOCOL_TLS_CLIENT = PROTOCOL_TLS


try:
    from ssl import OP_NO_COMPRESSION, OP_NO_SSLv2, OP_NO_SSLv3
except ImportError:
    OP_NO_SSLv2, OP_NO_SSLv3 = 0x1000000, 0x2000000
    OP_NO_COMPRESSION = 0x20000


try:  # OP_NO_TICKET was added in Python 3.6
    from ssl import OP_NO_TICKET
except ImportError:
    OP_NO_TICKET = 0x4000


# A secure default.
# Sources for more information on TLS ciphers:
#
# - https://wiki.mozilla.org/Security/Server_Side_TLS
# - https://www.ssllabs.com/projects/best-practices/index.html
# - https://hynek.me/articles/hardening-your-web-servers-ssl-ciphers/
#
# The general intent is:
# - prefer cipher suites that offer perfect forward secrecy (DHE/ECDHE),
# - prefer ECDHE over DHE for better performance,
# - prefer any AES-GCM and ChaCha20 over any AES-CBC for better performance and
#   security,
# - prefer AES-GCM over ChaCha20 because hardware-accelerated AES is common,
# - disable NULL authentication, MD5 MACs, DSS, and other
#   insecure ciphers for security reasons.
# - NOTE: TLS 1.3 cipher suites are managed through a different interface
#   not exposed by CPython (yet!) and are enabled by default if they're available.
DEFAULT_CIPHERS = ":".join(
    [
        "ECDHE+AESGCM",
        "ECDHE+CHACHA20",
        "DHE+AESGCM",
        "DHE+CHACHA20",
        "ECDH+AESGCM",
        "DH+AESGCM",
        "ECDH+AES",
        "DH+AES",
        "RSA+AESGCM",
        "RSA+AES",
        "!aNULL",
        "!eNULL",
        "!MD5",
        "!DSS",
    ]
)

try:
    from ssl import SSLContext  # Modern SSL?
except ImportError:

    class SSLContext(object):  # Platform-specific: Python 2
        def __init__(self, protocol_version):
            self.protocol = protocol_version
            # Use default values from a real SSLContext
            self.check_hostname = False
            self.verify_mode = ssl.CERT_NONE
            self.ca_certs = None
            self.options = 0
            self.certfile = None
            self.keyfile = None
            self.ciphers = None

        def load_cert_chain(self, certfile, keyfile):
            self.certfile = certfile
            self.keyfile = keyfile

        def load_verify_locations(self, cafile=None, capath=None, cadata=None):
            self.ca_certs = cafile

            if capath is not None:
                raise SSLError("CA directories not supported in older Pythons")

            if cadata is not None:
                raise SSLError("CA data not supported in older Pythons")

        def set_ciphers(self, cipher_suite):
            self.ciphers = cipher_suite

        def wrap_socket(self, socket, server_hostname=None, server_side=False):
            warnings.warn(
                "A true SSLContext object is not available. This prevents "
                "urllib3 from configuring SSL appropriately and may cause "
                "certain SSL connections to fail. You can upgrade to a newer "
                "version of Python to solve this. For more information, see "
                "https://urllib3.readthedocs.io/en/1.26.x/advanced-usage.html"
                "#ssl-warnings",
                InsecurePlatformWarning,
            )
            kwargs = {
                "keyfile": self.keyfile,
                "certfile": self.certfile,
                "ca_certs": self.ca_certs,
                "cert_reqs": self.verify_mode,
                "ssl_version": self.protocol,
                "server_side": server_side,
            }
            return wrap_socket(socket, ciphers=self.ciphers, **kwargs)


def assert_fingerprint(cert, fingerprint):
    """
    Checks if given fingerprint matches the supplied certificate.

    :param cert:
        Certificate as bytes object.
    :param fingerprint:
        Fingerprint as string of hexdigits, can be interspersed by colons.
    """

    fingerprint = fingerprint.replace(":", "").lower()
    digest_length = len(fingerprint)
    if digest_length not in HASHFUNC_MAP:
        raise SSLError("Fingerprint of invalid length: {0}".format(fingerprint))
    hashfunc = HASHFUNC_MAP.get(digest_length)
    if hashfunc is None:
        raise SSLError(
            "Hash function implementation unavailable for fingerprint length: {0}".format(
                digest_length
            )
        )

    # We need encode() here for py32; works on py2 and p33.
    fingerprint_bytes = unhexlify(fingerprint.encode())

    cert_digest = hashfunc(cert).digest()

    if not _const_compare_digest(cert_digest, fingerprint_bytes):
        raise SSLError(
            'Fingerprints did not match. Expected "{0}", got "{1}".'.format(
                fingerprint, hexlify(cert_digest)
            )
        )


def resolve_cert_reqs(candidate):
    """
    Resolves the argument to a numeric constant, which can be passed to
    the wrap_socket function/method from the ssl module.
    Defaults to :data:`ssl.CERT_REQUIRED`.
    If given a string it is assumed to be the name of the constant in the
    :mod:`ssl` module or its abbreviation.
    (So you can specify `REQUIRED` instead of `CERT_REQUIRED`.
    If it's neither `None` nor a string we assume it is already the numeric
    constant which can directly be passed to wrap_socket.
    """
    if candidate is None:
        return CERT_REQUIRED

    if isinstance(candidate, str):
        res = getattr(ssl, candidate, None)
        if res is None:
            res = getattr(ssl, "CERT_" + candidate)
        return res

    return candidate


def resolve_ssl_version(candidate):
    """
    like resolve_cert_reqs
    """
    if candidate is None:
        return PROTOCOL_TLS

    if isinstance(candidate, str):
        res = getattr(ssl, candidate, None)
        if res is None:
            res = getattr(ssl, "PROTOCOL_" + candidate)
        return res

    return candidate


def create_urllib3_context(
    ssl_version=None, cert_reqs=None, options=None, ciphers=None
):
    """All arguments have the same meaning as ``ssl_wrap_socket``.

    By default, this function does a lot of the same work that
    ``ssl.create_default_context`` does on Python 3.4+. It:

    - Disables SSLv2, SSLv3, and compression
    - Sets a restricted set of server ciphers

    If you wish to enable SSLv3, you can do::

        from pip._vendor.urllib3.util import ssl_
        context = ssl_.create_urllib3_context()
        context.options &= ~ssl_.OP_NO_SSLv3

    You can do the same to enable compression (substituting ``COMPRESSION``
    for ``SSLv3`` in the last line above).

    :param ssl_version:
        The desired protocol version to use. This will default to
        PROTOCOL_SSLv23 which will negotiate the highest protocol that both
        the server and your installation of OpenSSL support.
    :param cert_reqs:
        Whether to require the certificate verification. This defaults to
        ``ssl.CERT_REQUIRED``.
    :param options:
        Specific OpenSSL options. These default to ``ssl.OP_NO_SSLv2``,
        ``ssl.OP_NO_SSLv3``, ``ssl.OP_NO_COMPRESSION``, and ``ssl.OP_NO_TICKET``.
    :param ciphers:
        Which cipher suites to allow the server to select.
    :returns:
        Constructed SSLContext object with specified options
    :rtype: SSLContext
    """
    # PROTOCOL_TLS is deprecated in Python 3.10
    if not ssl_version or ssl_version == PROTOCOL_TLS:
        ssl_version = PROTOCOL_TLS_CLIENT

    context = SSLContext(ssl_version)

    context.set_ciphers(ciphers or DEFAULT_CIPHERS)

    # Setting the default here, as we may have no ssl module on import
    cert_reqs = ssl.CERT_REQUIRED if cert_reqs is None else cert_reqs

    if options is None:
        options = 0
        # SSLv2 is easily broken and is considered harmful and dangerous
        options |= OP_NO_SSLv2
        # SSLv3 has several problems and is now dangerous
        options |= OP_NO_SSLv3
        # Disable compression to prevent CRIME attacks for OpenSSL 1.0+
        # (issue #309)
        options |= OP_NO_COMPRESSION
        # TLSv1.2 only. Unless set explicitly, do not request tickets.
        # This may save some bandwidth on wire, and although the ticket is encrypted,
        # there is a risk associated with it being on wire,
        # if the server is not rotating its ticketing keys properly.
        options |= OP_NO_TICKET

    context.options |= options

    # Enable post-handshake authentication for TLS 1.3, see GH #1634. PHA is
    # necessary for conditional client cert authentication with TLS 1.3.
    # The attribute is None for OpenSSL <= 1.1.0 or does not exist in older
    # versions of Python.  We only enable on Python 3.7.4+ or if certificate
    # verification is enabled to work around Python issue #37428
    # See: https://bugs.python.org/issue37428
    if (cert_reqs == ssl.CERT_REQUIRED or sys.version_info >= (3, 7, 4)) and getattr(
        context, "post_handshake_auth", None
    ) is not None:
        context.post_handshake_auth = True

    def disable_check_hostname():
        if (
            getattr(context, "check_hostname", None) is not None
        ):  # Platform-specific: Python 3.2
            # We do our own verification, including fingerprints and alternative
            # hostnames. So disable it here
            context.check_hostname = False

    # The order of the below lines setting verify_mode and check_hostname
    # matter due to safe-guards SSLContext has to prevent an SSLContext with
    # check_hostname=True, verify_mode=NONE/OPTIONAL. This is made even more
    # complex because we don't know whether PROTOCOL_TLS_CLIENT will be used
    # or not so we don't know the initial state of the freshly created SSLContext.
    if cert_reqs == ssl.CERT_REQUIRED:
        context.verify_mode = cert_reqs
        disable_check_hostname()
    else:
        disable_check_hostname()
        context.verify_mode = cert_reqs

    # Enable logging of TLS session keys via defacto standard environment variable
    # 'SSLKEYLOGFILE', if the feature is available (Python 3.8+). Skip empty values.
    if hasattr(context, "keylog_filename"):
        sslkeylogfile = os.environ.get("SSLKEYLOGFILE")
        if sslkeylogfile:
            context.keylog_filename = sslkeylogfile

    return context


def ssl_wrap_socket(
    sock,
    keyfile=None,
    certfile=None,
    cert_reqs=None,
    ca_certs=None,
    server_hostname=None,
    ssl_version=None,
    ciphers=None,
    ssl_context=None,
    ca_cert_dir=None,
    key_password=None,
    ca_cert_data=None,
    tls_in_tls=False,
):
    """
    All arguments except for server_hostname, ssl_context, and ca_cert_dir have
    the same meaning as they do when using :func:`ssl.wrap_socket`.

    :param server_hostname:
        When SNI is supported, the expected hostname of the certificate
    :param ssl_context:
        A pre-made :class:`SSLContext` object. If none is provided, one will
        be created using :func:`create_urllib3_context`.
    :param ciphers:
        A string of ciphers we wish the client to support.
    :param ca_cert_dir:
        A directory containing CA certificates in multiple separate files, as
        supported by OpenSSL's -CApath flag or the capath argument to
        SSLContext.load_verify_locations().
    :param key_password:
        Optional password if the keyfile is encrypted.
    :param ca_cert_data:
        Optional string containing CA certificates in PEM format suitable for
        passing as the cadata parameter to SSLContext.load_verify_locations()
    :param tls_in_tls:
        Use SSLTransport to wrap the existing socket.
    """
    context = ssl_context
    if context is None:
        # Note: This branch of code and all the variables in it are no longer
        # used by urllib3 itself. We should consider deprecating and removing
        # this code.
        context = create_urllib3_context(ssl_version, cert_reqs, ciphers=ciphers)

    if ca_certs or ca_cert_dir or ca_cert_data:
        try:
            context.load_verify_locations(ca_certs, ca_cert_dir, ca_cert_data)
        except (IOError, OSError) as e:
            raise SSLError(e)

    elif ssl_context is None and hasattr(context, "load_default_certs"):
        # try to load OS default certs; works well on Windows (require Python3.4+)
        context.load_default_certs()

    # Attempt to detect if we get the goofy behavior of the
    # keyfile being encrypted and OpenSSL asking for the
    # passphrase via the terminal and instead error out.
    if keyfile and key_password is None and _is_key_file_encrypted(keyfile):
        raise SSLError("Client private key is encrypted, password is required")

    if certfile:
        if key_password is None:
            context.load_cert_chain(certfile, keyfile)
        else:
            context.load_cert_chain(certfile, keyfile, key_password)

    try:
        if hasattr(context, "set_alpn_protocols"):
            context.set_alpn_protocols(ALPN_PROTOCOLS)
    except NotImplementedError:  # Defensive: in CI, we always have set_alpn_protocols
        pass

    # If we detect server_hostname is an IP address then the SNI
    # extension should not be used according to RFC3546 Section 3.1
    use_sni_hostname = server_hostname and not is_ipaddress(server_hostname)
    # SecureTransport uses server_hostname in certificate verification.
    send_sni = (use_sni_hostname and HAS_SNI) or (
        IS_SECURETRANSPORT and server_hostname
    )
    # Do not warn the user if server_hostname is an invalid SNI hostname.
    if not HAS_SNI and use_sni_hostname:
        warnings.warn(
            "An HTTPS request has been made, but the SNI (Server Name "
            "Indication) extension to TLS is not available on this platform. "
            "This may cause the server to present an incorrect TLS "
            "certificate, which can cause validation failures. You can upgrade to "
            "a newer version of Python to solve this. For more information, see "
            "https://urllib3.readthedocs.io/en/1.26.x/advanced-usage.html"
            "#ssl-warnings",
            SNIMissingWarning,
        )

    if send_sni:
        ssl_sock = _ssl_wrap_socket_impl(
            sock, context, tls_in_tls, server_hostname=server_hostname
        )
    else:
        ssl_sock = _ssl_wrap_socket_impl(sock, context, tls_in_tls)
    return ssl_sock


def is_ipaddress(hostname):
    """Detects whether the hostname given is an IPv4 or IPv6 address.
    Also detects IPv6 addresses with Zone IDs.

    :param str hostname: Hostname to examine.
    :return: True if the hostname is an IP address, False otherwise.
    """
    if not six.PY2 and isinstance(hostname, bytes):
        # IDN A-label bytes are ASCII compatible.
        hostname = hostname.decode("ascii")
    return bool(IPV4_RE.match(hostname) or BRACELESS_IPV6_ADDRZ_RE.match(hostname))


def _is_key_file_encrypted(key_file):
    """Detects if a key file is encrypted or not."""
    with open(key_file, "r") as f:
        for line in f:
            # Look for Proc-Type: 4,ENCRYPTED
            if "ENCRYPTED" in line:
                return True

    return False


def _ssl_wrap_socket_impl(sock, ssl_context, tls_in_tls, server_hostname=None):
    if tls_in_tls:
        if not SSLTransport:
            # Import error, ssl is not available.
            raise ProxySchemeUnsupported(
                "TLS in TLS requires support for the 'ssl' module"
            )

        SSLTransport._validate_ssl_context_for_tls_in_tls(ssl_context)
        return SSLTransport(sock, ssl_context, server_hostname)

    if server_hostname:
        return ssl_context.wrap_socket(sock, server_hostname=server_hostname)
    else:
        return ssl_context.wrap_socket(sock)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\fixtures\sql.py ===
# testing/fixtures/sql.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors
from __future__ import annotations

import itertools
import random
import re
import sys

import sqlalchemy as sa
from .base import TestBase
from .. import config
from .. import mock
from ..assertions import eq_
from ..assertions import ne_
from ..util import adict
from ..util import drop_all_tables_from_metadata
from ... import event
from ... import util
from ...schema import sort_tables_and_constraints
from ...sql import visitors
from ...sql.elements import ClauseElement


class TablesTest(TestBase):
    # 'once', None
    run_setup_bind = "once"

    # 'once', 'each', None
    run_define_tables = "once"

    # 'once', 'each', None
    run_create_tables = "once"

    # 'once', 'each', None
    run_inserts = "each"

    # 'each', None
    run_deletes = "each"

    # 'once', None
    run_dispose_bind = None

    bind = None
    _tables_metadata = None
    tables = None
    other = None
    sequences = None

    @config.fixture(autouse=True, scope="class")
    def _setup_tables_test_class(self):
        cls = self.__class__
        cls._init_class()

        cls._setup_once_tables()

        cls._setup_once_inserts()

        yield

        cls._teardown_once_metadata_bind()

    @config.fixture(autouse=True, scope="function")
    def _setup_tables_test_instance(self):
        self._setup_each_tables()
        self._setup_each_inserts()

        yield

        self._teardown_each_tables()

    @property
    def tables_test_metadata(self):
        return self._tables_metadata

    @classmethod
    def _init_class(cls):
        if cls.run_define_tables == "each":
            if cls.run_create_tables == "once":
                cls.run_create_tables = "each"
            assert cls.run_inserts in ("each", None)

        cls.other = adict()
        cls.tables = adict()
        cls.sequences = adict()

        cls.bind = cls.setup_bind()
        cls._tables_metadata = sa.MetaData()

    @classmethod
    def _setup_once_inserts(cls):
        if cls.run_inserts == "once":
            cls._load_fixtures()
            with cls.bind.begin() as conn:
                cls.insert_data(conn)

    @classmethod
    def _setup_once_tables(cls):
        if cls.run_define_tables == "once":
            cls.define_tables(cls._tables_metadata)
            if cls.run_create_tables == "once":
                cls._tables_metadata.create_all(cls.bind)
            cls.tables.update(cls._tables_metadata.tables)
            cls.sequences.update(cls._tables_metadata._sequences)

    def _setup_each_tables(self):
        if self.run_define_tables == "each":
            self.define_tables(self._tables_metadata)
            if self.run_create_tables == "each":
                self._tables_metadata.create_all(self.bind)
            self.tables.update(self._tables_metadata.tables)
            self.sequences.update(self._tables_metadata._sequences)
        elif self.run_create_tables == "each":
            self._tables_metadata.create_all(self.bind)

    def _setup_each_inserts(self):
        if self.run_inserts == "each":
            self._load_fixtures()
            with self.bind.begin() as conn:
                self.insert_data(conn)

    def _teardown_each_tables(self):
        if self.run_define_tables == "each":
            self.tables.clear()
            if self.run_create_tables == "each":
                drop_all_tables_from_metadata(self._tables_metadata, self.bind)
            self._tables_metadata.clear()
        elif self.run_create_tables == "each":
            drop_all_tables_from_metadata(self._tables_metadata, self.bind)

        savepoints = getattr(config.requirements, "savepoints", False)
        if savepoints:
            savepoints = savepoints.enabled

        # no need to run deletes if tables are recreated on setup
        if (
            self.run_define_tables != "each"
            and self.run_create_tables != "each"
            and self.run_deletes == "each"
        ):
            with self.bind.begin() as conn:
                for table in reversed(
                    [
                        t
                        for (t, fks) in sort_tables_and_constraints(
                            self._tables_metadata.tables.values()
                        )
                        if t is not None
                    ]
                ):
                    try:
                        if savepoints:
                            with conn.begin_nested():
                                conn.execute(table.delete())
                        else:
                            conn.execute(table.delete())
                    except sa.exc.DBAPIError as ex:
                        print(
                            ("Error emptying table %s: %r" % (table, ex)),
                            file=sys.stderr,
                        )

    @classmethod
    def _teardown_once_metadata_bind(cls):
        if cls.run_create_tables:
            drop_all_tables_from_metadata(cls._tables_metadata, cls.bind)

        if cls.run_dispose_bind == "once":
            cls.dispose_bind(cls.bind)

        cls._tables_metadata.bind = None

        if cls.run_setup_bind is not None:
            cls.bind = None

    @classmethod
    def setup_bind(cls):
        return config.db

    @classmethod
    def dispose_bind(cls, bind):
        if hasattr(bind, "dispose"):
            bind.dispose()
        elif hasattr(bind, "close"):
            bind.close()

    @classmethod
    def define_tables(cls, metadata):
        pass

    @classmethod
    def fixtures(cls):
        return {}

    @classmethod
    def insert_data(cls, connection):
        pass

    def sql_count_(self, count, fn):
        self.assert_sql_count(self.bind, fn, count)

    def sql_eq_(self, callable_, statements):
        self.assert_sql(self.bind, callable_, statements)

    @classmethod
    def _load_fixtures(cls):
        """Insert rows as represented by the fixtures() method."""
        headers, rows = {}, {}
        for table, data in cls.fixtures().items():
            if len(data) < 2:
                continue
            if isinstance(table, str):
                table = cls.tables[table]
            headers[table] = data[0]
            rows[table] = data[1:]
        for table, fks in sort_tables_and_constraints(
            cls._tables_metadata.tables.values()
        ):
            if table is None:
                continue
            if table not in headers:
                continue
            with cls.bind.begin() as conn:
                conn.execute(
                    table.insert(),
                    [
                        dict(zip(headers[table], column_values))
                        for column_values in rows[table]
                    ],
                )


class NoCache:
    @config.fixture(autouse=True, scope="function")
    def _disable_cache(self):
        _cache = config.db._compiled_cache
        config.db._compiled_cache = None
        yield
        config.db._compiled_cache = _cache


class RemovesEvents:
    @util.memoized_property
    def _event_fns(self):
        return set()

    def event_listen(self, target, name, fn, **kw):
        self._event_fns.add((target, name, fn))
        event.listen(target, name, fn, **kw)

    @config.fixture(autouse=True, scope="function")
    def _remove_events(self):
        yield
        for key in self._event_fns:
            event.remove(*key)


class ComputedReflectionFixtureTest(TablesTest):
    run_inserts = run_deletes = None

    __backend__ = True
    __requires__ = ("computed_columns", "table_reflection")

    regexp = re.compile(r"[\[\]\(\)\s`'\"]*")

    def normalize(self, text):
        return self.regexp.sub("", text).lower()

    @classmethod
    def define_tables(cls, metadata):
        from ... import Integer
        from ... import testing
        from ...schema import Column
        from ...schema import Computed
        from ...schema import Table

        Table(
            "computed_default_table",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("normal", Integer),
            Column("computed_col", Integer, Computed("normal + 42")),
            Column("with_default", Integer, server_default="42"),
        )

        t = Table(
            "computed_column_table",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("normal", Integer),
            Column("computed_no_flag", Integer, Computed("normal + 42")),
        )

        if testing.requires.schemas.enabled:
            t2 = Table(
                "computed_column_table",
                metadata,
                Column("id", Integer, primary_key=True),
                Column("normal", Integer),
                Column("computed_no_flag", Integer, Computed("normal / 42")),
                schema=config.test_schema,
            )

        if testing.requires.computed_columns_virtual.enabled:
            t.append_column(
                Column(
                    "computed_virtual",
                    Integer,
                    Computed("normal + 2", persisted=False),
                )
            )
            if testing.requires.schemas.enabled:
                t2.append_column(
                    Column(
                        "computed_virtual",
                        Integer,
                        Computed("normal / 2", persisted=False),
                    )
                )
        if testing.requires.computed_columns_stored.enabled:
            t.append_column(
                Column(
                    "computed_stored",
                    Integer,
                    Computed("normal - 42", persisted=True),
                )
            )
            if testing.requires.schemas.enabled:
                t2.append_column(
                    Column(
                        "computed_stored",
                        Integer,
                        Computed("normal * 42", persisted=True),
                    )
                )


class CacheKeyFixture:
    def _compare_equal(self, a, b, compare_values):
        a_key = a._generate_cache_key()
        b_key = b._generate_cache_key()

        if a_key is None:
            assert a._annotations.get("nocache")

            assert b_key is None
        else:
            eq_(a_key.key, b_key.key)
            eq_(hash(a_key.key), hash(b_key.key))

            for a_param, b_param in zip(a_key.bindparams, b_key.bindparams):
                assert a_param.compare(b_param, compare_values=compare_values)
        return a_key, b_key

    def _run_cache_key_fixture(self, fixture, compare_values):
        case_a = fixture()
        case_b = fixture()

        for a, b in itertools.combinations_with_replacement(
            range(len(case_a)), 2
        ):
            if a == b:
                a_key, b_key = self._compare_equal(
                    case_a[a], case_b[b], compare_values
                )
                if a_key is None:
                    continue
            else:
                a_key = case_a[a]._generate_cache_key()
                b_key = case_b[b]._generate_cache_key()

                if a_key is None or b_key is None:
                    if a_key is None:
                        assert case_a[a]._annotations.get("nocache")
                    if b_key is None:
                        assert case_b[b]._annotations.get("nocache")
                    continue

                if a_key.key == b_key.key:
                    for a_param, b_param in zip(
                        a_key.bindparams, b_key.bindparams
                    ):
                        if not a_param.compare(
                            b_param, compare_values=compare_values
                        ):
                            break
                    else:
                        # this fails unconditionally since we could not
                        # find bound parameter values that differed.
                        # Usually we intended to get two distinct keys here
                        # so the failure will be more descriptive using the
                        # ne_() assertion.
                        ne_(a_key.key, b_key.key)
                else:
                    ne_(a_key.key, b_key.key)

            # ClauseElement-specific test to ensure the cache key
            # collected all the bound parameters that aren't marked
            # as "literal execute"
            if isinstance(case_a[a], ClauseElement) and isinstance(
                case_b[b], ClauseElement
            ):
                assert_a_params = []
                assert_b_params = []

                for elem in visitors.iterate(case_a[a]):
                    if elem.__visit_name__ == "bindparam":
                        assert_a_params.append(elem)

                for elem in visitors.iterate(case_b[b]):
                    if elem.__visit_name__ == "bindparam":
                        assert_b_params.append(elem)

                # note we're asserting the order of the params as well as
                # if there are dupes or not.  ordering has to be
                # deterministic and matches what a traversal would provide.
                eq_(
                    sorted(a_key.bindparams, key=lambda b: b.key),
                    sorted(
                        util.unique_list(assert_a_params), key=lambda b: b.key
                    ),
                )
                eq_(
                    sorted(b_key.bindparams, key=lambda b: b.key),
                    sorted(
                        util.unique_list(assert_b_params), key=lambda b: b.key
                    ),
                )

    def _run_cache_key_equal_fixture(self, fixture, compare_values):
        case_a = fixture()
        case_b = fixture()

        for a, b in itertools.combinations_with_replacement(
            range(len(case_a)), 2
        ):
            self._compare_equal(case_a[a], case_b[b], compare_values)


def insertmanyvalues_fixture(
    connection, randomize_rows=False, warn_on_downgraded=False
):
    dialect = connection.dialect
    orig_dialect = dialect._deliver_insertmanyvalues_batches
    orig_conn = connection._exec_insertmany_context

    class RandomCursor:
        __slots__ = ("cursor",)

        def __init__(self, cursor):
            self.cursor = cursor

        # only this method is called by the deliver method.
        # by not having the other methods we assert that those aren't being
        # used

        @property
        def description(self):
            return self.cursor.description

        def fetchall(self):
            rows = self.cursor.fetchall()
            rows = list(rows)
            random.shuffle(rows)
            return rows

    def _deliver_insertmanyvalues_batches(
        connection,
        cursor,
        statement,
        parameters,
        generic_setinputsizes,
        context,
    ):
        if randomize_rows:
            cursor = RandomCursor(cursor)
        for batch in orig_dialect(
            connection,
            cursor,
            statement,
            parameters,
            generic_setinputsizes,
            context,
        ):
            if warn_on_downgraded and batch.is_downgraded:
                util.warn("Batches were downgraded for sorted INSERT")

            yield batch

    def _exec_insertmany_context(dialect, context):
        with mock.patch.object(
            dialect,
            "_deliver_insertmanyvalues_batches",
            new=_deliver_insertmanyvalues_batches,
        ):
            return orig_conn(dialect, context)

    connection._exec_insertmany_context = _exec_insertmany_context

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pycparser\c_generator.py ===
#------------------------------------------------------------------------------
# pycparser: c_generator.py
#
# C code generator from pycparser AST nodes.
#
# Eli Bendersky [https://eli.thegreenplace.net/]
# License: BSD
#------------------------------------------------------------------------------
from . import c_ast


class CGenerator(object):
    """ Uses the same visitor pattern as c_ast.NodeVisitor, but modified to
        return a value from each visit method, using string accumulation in
        generic_visit.
    """
    def __init__(self, reduce_parentheses=False):
        """ Constructs C-code generator

            reduce_parentheses:
                if True, eliminates needless parentheses on binary operators
        """
        # Statements start with indentation of self.indent_level spaces, using
        # the _make_indent method.
        self.indent_level = 0
        self.reduce_parentheses = reduce_parentheses

    def _make_indent(self):
        return ' ' * self.indent_level

    def visit(self, node):
        method = 'visit_' + node.__class__.__name__
        return getattr(self, method, self.generic_visit)(node)

    def generic_visit(self, node):
        if node is None:
            return ''
        else:
            return ''.join(self.visit(c) for c_name, c in node.children())

    def visit_Constant(self, n):
        return n.value

    def visit_ID(self, n):
        return n.name

    def visit_Pragma(self, n):
        ret = '#pragma'
        if n.string:
            ret += ' ' + n.string
        return ret

    def visit_ArrayRef(self, n):
        arrref = self._parenthesize_unless_simple(n.name)
        return arrref + '[' + self.visit(n.subscript) + ']'

    def visit_StructRef(self, n):
        sref = self._parenthesize_unless_simple(n.name)
        return sref + n.type + self.visit(n.field)

    def visit_FuncCall(self, n):
        fref = self._parenthesize_unless_simple(n.name)
        return fref + '(' + self.visit(n.args) + ')'

    def visit_UnaryOp(self, n):
        if n.op == 'sizeof':
            # Always parenthesize the argument of sizeof since it can be
            # a name.
            return 'sizeof(%s)' % self.visit(n.expr)
        else:
            operand = self._parenthesize_unless_simple(n.expr)
            if n.op == 'p++':
                return '%s++' % operand
            elif n.op == 'p--':
                return '%s--' % operand
            else:
                return '%s%s' % (n.op, operand)

    # Precedence map of binary operators:
    precedence_map = {
        # Should be in sync with c_parser.CParser.precedence
        # Higher numbers are stronger binding
        '||': 0,  # weakest binding
        '&&': 1,
        '|': 2,
        '^': 3,
        '&': 4,
        '==': 5, '!=': 5,
        '>': 6, '>=': 6, '<': 6, '<=': 6,
        '>>': 7, '<<': 7,
        '+': 8, '-': 8,
        '*': 9, '/': 9, '%': 9  # strongest binding
    }

    def visit_BinaryOp(self, n):
        # Note: all binary operators are left-to-right associative
        #
        # If `n.left.op` has a stronger or equally binding precedence in
        # comparison to `n.op`, no parenthesis are needed for the left:
        # e.g., `(a*b) + c` is equivalent to `a*b + c`, as well as
        #       `(a+b) - c` is equivalent to `a+b - c` (same precedence).
        # If the left operator is weaker binding than the current, then
        # parentheses are necessary:
        # e.g., `(a+b) * c` is NOT equivalent to `a+b * c`.
        lval_str = self._parenthesize_if(
            n.left,
            lambda d: not (self._is_simple_node(d) or
                      self.reduce_parentheses and isinstance(d, c_ast.BinaryOp) and
                      self.precedence_map[d.op] >= self.precedence_map[n.op]))
        # If `n.right.op` has a stronger -but not equal- binding precedence,
        # parenthesis can be omitted on the right:
        # e.g., `a + (b*c)` is equivalent to `a + b*c`.
        # If the right operator is weaker or equally binding, then parentheses
        # are necessary:
        # e.g., `a * (b+c)` is NOT equivalent to `a * b+c` and
        #       `a - (b+c)` is NOT equivalent to `a - b+c` (same precedence).
        rval_str = self._parenthesize_if(
            n.right,
            lambda d: not (self._is_simple_node(d) or
                      self.reduce_parentheses and isinstance(d, c_ast.BinaryOp) and
                      self.precedence_map[d.op] > self.precedence_map[n.op]))
        return '%s %s %s' % (lval_str, n.op, rval_str)

    def visit_Assignment(self, n):
        rval_str = self._parenthesize_if(
                            n.rvalue,
                            lambda n: isinstance(n, c_ast.Assignment))
        return '%s %s %s' % (self.visit(n.lvalue), n.op, rval_str)

    def visit_IdentifierType(self, n):
        return ' '.join(n.names)

    def _visit_expr(self, n):
        if isinstance(n, c_ast.InitList):
            return '{' + self.visit(n) + '}'
        elif isinstance(n, c_ast.ExprList):
            return '(' + self.visit(n) + ')'
        else:
            return self.visit(n)

    def visit_Decl(self, n, no_type=False):
        # no_type is used when a Decl is part of a DeclList, where the type is
        # explicitly only for the first declaration in a list.
        #
        s = n.name if no_type else self._generate_decl(n)
        if n.bitsize: s += ' : ' + self.visit(n.bitsize)
        if n.init:
            s += ' = ' + self._visit_expr(n.init)
        return s

    def visit_DeclList(self, n):
        s = self.visit(n.decls[0])
        if len(n.decls) > 1:
            s += ', ' + ', '.join(self.visit_Decl(decl, no_type=True)
                                    for decl in n.decls[1:])
        return s

    def visit_Typedef(self, n):
        s = ''
        if n.storage: s += ' '.join(n.storage) + ' '
        s += self._generate_type(n.type)
        return s

    def visit_Cast(self, n):
        s = '(' + self._generate_type(n.to_type, emit_declname=False) + ')'
        return s + ' ' + self._parenthesize_unless_simple(n.expr)

    def visit_ExprList(self, n):
        visited_subexprs = []
        for expr in n.exprs:
            visited_subexprs.append(self._visit_expr(expr))
        return ', '.join(visited_subexprs)

    def visit_InitList(self, n):
        visited_subexprs = []
        for expr in n.exprs:
            visited_subexprs.append(self._visit_expr(expr))
        return ', '.join(visited_subexprs)

    def visit_Enum(self, n):
        return self._generate_struct_union_enum(n, name='enum')

    def visit_Alignas(self, n):
        return '_Alignas({})'.format(self.visit(n.alignment))

    def visit_Enumerator(self, n):
        if not n.value:
            return '{indent}{name},\n'.format(
                indent=self._make_indent(),
                name=n.name,
            )
        else:
            return '{indent}{name} = {value},\n'.format(
                indent=self._make_indent(),
                name=n.name,
                value=self.visit(n.value),
            )

    def visit_FuncDef(self, n):
        decl = self.visit(n.decl)
        self.indent_level = 0
        body = self.visit(n.body)
        if n.param_decls:
            knrdecls = ';\n'.join(self.visit(p) for p in n.param_decls)
            return decl + '\n' + knrdecls + ';\n' + body + '\n'
        else:
            return decl + '\n' + body + '\n'

    def visit_FileAST(self, n):
        s = ''
        for ext in n.ext:
            if isinstance(ext, c_ast.FuncDef):
                s += self.visit(ext)
            elif isinstance(ext, c_ast.Pragma):
                s += self.visit(ext) + '\n'
            else:
                s += self.visit(ext) + ';\n'
        return s

    def visit_Compound(self, n):
        s = self._make_indent() + '{\n'
        self.indent_level += 2
        if n.block_items:
            s += ''.join(self._generate_stmt(stmt) for stmt in n.block_items)
        self.indent_level -= 2
        s += self._make_indent() + '}\n'
        return s

    def visit_CompoundLiteral(self, n):
        return '(' + self.visit(n.type) + '){' + self.visit(n.init) + '}'


    def visit_EmptyStatement(self, n):
        return ';'

    def visit_ParamList(self, n):
        return ', '.join(self.visit(param) for param in n.params)

    def visit_Return(self, n):
        s = 'return'
        if n.expr: s += ' ' + self.visit(n.expr)
        return s + ';'

    def visit_Break(self, n):
        return 'break;'

    def visit_Continue(self, n):
        return 'continue;'

    def visit_TernaryOp(self, n):
        s  = '(' + self._visit_expr(n.cond) + ') ? '
        s += '(' + self._visit_expr(n.iftrue) + ') : '
        s += '(' + self._visit_expr(n.iffalse) + ')'
        return s

    def visit_If(self, n):
        s = 'if ('
        if n.cond: s += self.visit(n.cond)
        s += ')\n'
        s += self._generate_stmt(n.iftrue, add_indent=True)
        if n.iffalse:
            s += self._make_indent() + 'else\n'
            s += self._generate_stmt(n.iffalse, add_indent=True)
        return s

    def visit_For(self, n):
        s = 'for ('
        if n.init: s += self.visit(n.init)
        s += ';'
        if n.cond: s += ' ' + self.visit(n.cond)
        s += ';'
        if n.next: s += ' ' + self.visit(n.next)
        s += ')\n'
        s += self._generate_stmt(n.stmt, add_indent=True)
        return s

    def visit_While(self, n):
        s = 'while ('
        if n.cond: s += self.visit(n.cond)
        s += ')\n'
        s += self._generate_stmt(n.stmt, add_indent=True)
        return s

    def visit_DoWhile(self, n):
        s = 'do\n'
        s += self._generate_stmt(n.stmt, add_indent=True)
        s += self._make_indent() + 'while ('
        if n.cond: s += self.visit(n.cond)
        s += ');'
        return s

    def visit_StaticAssert(self, n):
        s = '_Static_assert('
        s += self.visit(n.cond)
        if n.message:
            s += ','
            s += self.visit(n.message)
        s += ')'
        return s

    def visit_Switch(self, n):
        s = 'switch (' + self.visit(n.cond) + ')\n'
        s += self._generate_stmt(n.stmt, add_indent=True)
        return s

    def visit_Case(self, n):
        s = 'case ' + self.visit(n.expr) + ':\n'
        for stmt in n.stmts:
            s += self._generate_stmt(stmt, add_indent=True)
        return s

    def visit_Default(self, n):
        s = 'default:\n'
        for stmt in n.stmts:
            s += self._generate_stmt(stmt, add_indent=True)
        return s

    def visit_Label(self, n):
        return n.name + ':\n' + self._generate_stmt(n.stmt)

    def visit_Goto(self, n):
        return 'goto ' + n.name + ';'

    def visit_EllipsisParam(self, n):
        return '...'

    def visit_Struct(self, n):
        return self._generate_struct_union_enum(n, 'struct')

    def visit_Typename(self, n):
        return self._generate_type(n.type)

    def visit_Union(self, n):
        return self._generate_struct_union_enum(n, 'union')

    def visit_NamedInitializer(self, n):
        s = ''
        for name in n.name:
            if isinstance(name, c_ast.ID):
                s += '.' + name.name
            else:
                s += '[' + self.visit(name) + ']'
        s += ' = ' + self._visit_expr(n.expr)
        return s

    def visit_FuncDecl(self, n):
        return self._generate_type(n)

    def visit_ArrayDecl(self, n):
        return self._generate_type(n, emit_declname=False)

    def visit_TypeDecl(self, n):
        return self._generate_type(n, emit_declname=False)

    def visit_PtrDecl(self, n):
        return self._generate_type(n, emit_declname=False)

    def _generate_struct_union_enum(self, n, name):
        """ Generates code for structs, unions, and enums. name should be
            'struct', 'union', or 'enum'.
        """
        if name in ('struct', 'union'):
            members = n.decls
            body_function = self._generate_struct_union_body
        else:
            assert name == 'enum'
            members = None if n.values is None else n.values.enumerators
            body_function = self._generate_enum_body
        s = name + ' ' + (n.name or '')
        if members is not None:
            # None means no members
            # Empty sequence means an empty list of members
            s += '\n'
            s += self._make_indent()
            self.indent_level += 2
            s += '{\n'
            s += body_function(members)
            self.indent_level -= 2
            s += self._make_indent() + '}'
        return s

    def _generate_struct_union_body(self, members):
        return ''.join(self._generate_stmt(decl) for decl in members)

    def _generate_enum_body(self, members):
        # `[:-2] + '\n'` removes the final `,` from the enumerator list
        return ''.join(self.visit(value) for value in members)[:-2] + '\n'

    def _generate_stmt(self, n, add_indent=False):
        """ Generation from a statement node. This method exists as a wrapper
            for individual visit_* methods to handle different treatment of
            some statements in this context.
        """
        typ = type(n)
        if add_indent: self.indent_level += 2
        indent = self._make_indent()
        if add_indent: self.indent_level -= 2

        if typ in (
                c_ast.Decl, c_ast.Assignment, c_ast.Cast, c_ast.UnaryOp,
                c_ast.BinaryOp, c_ast.TernaryOp, c_ast.FuncCall, c_ast.ArrayRef,
                c_ast.StructRef, c_ast.Constant, c_ast.ID, c_ast.Typedef,
                c_ast.ExprList):
            # These can also appear in an expression context so no semicolon
            # is added to them automatically
            #
            return indent + self.visit(n) + ';\n'
        elif typ in (c_ast.Compound,):
            # No extra indentation required before the opening brace of a
            # compound - because it consists of multiple lines it has to
            # compute its own indentation.
            #
            return self.visit(n)
        elif typ in (c_ast.If,):
            return indent + self.visit(n)
        else:
            return indent + self.visit(n) + '\n'

    def _generate_decl(self, n):
        """ Generation from a Decl node.
        """
        s = ''
        if n.funcspec: s = ' '.join(n.funcspec) + ' '
        if n.storage: s += ' '.join(n.storage) + ' '
        if n.align: s += self.visit(n.align[0]) + ' '
        s += self._generate_type(n.type)
        return s

    def _generate_type(self, n, modifiers=[], emit_declname = True):
        """ Recursive generation from a type node. n is the type node.
            modifiers collects the PtrDecl, ArrayDecl and FuncDecl modifiers
            encountered on the way down to a TypeDecl, to allow proper
            generation from it.
        """
        typ = type(n)
        #~ print(n, modifiers)

        if typ == c_ast.TypeDecl:
            s = ''
            if n.quals: s += ' '.join(n.quals) + ' '
            s += self.visit(n.type)

            nstr = n.declname if n.declname and emit_declname else ''
            # Resolve modifiers.
            # Wrap in parens to distinguish pointer to array and pointer to
            # function syntax.
            #
            for i, modifier in enumerate(modifiers):
                if isinstance(modifier, c_ast.ArrayDecl):
                    if (i != 0 and
                        isinstance(modifiers[i - 1], c_ast.PtrDecl)):
                            nstr = '(' + nstr + ')'
                    nstr += '['
                    if modifier.dim_quals:
                        nstr += ' '.join(modifier.dim_quals) + ' '
                    nstr += self.visit(modifier.dim) + ']'
                elif isinstance(modifier, c_ast.FuncDecl):
                    if (i != 0 and
                        isinstance(modifiers[i - 1], c_ast.PtrDecl)):
                            nstr = '(' + nstr + ')'
                    nstr += '(' + self.visit(modifier.args) + ')'
                elif isinstance(modifier, c_ast.PtrDecl):
                    if modifier.quals:
                        nstr = '* %s%s' % (' '.join(modifier.quals),
                                           ' ' + nstr if nstr else '')
                    else:
                        nstr = '*' + nstr
            if nstr: s += ' ' + nstr
            return s
        elif typ == c_ast.Decl:
            return self._generate_decl(n.type)
        elif typ == c_ast.Typename:
            return self._generate_type(n.type, emit_declname = emit_declname)
        elif typ == c_ast.IdentifierType:
            return ' '.join(n.names) + ' '
        elif typ in (c_ast.ArrayDecl, c_ast.PtrDecl, c_ast.FuncDecl):
            return self._generate_type(n.type, modifiers + [n],
                                       emit_declname = emit_declname)
        else:
            return self.visit(n)

    def _parenthesize_if(self, n, condition):
        """ Visits 'n' and returns its string representation, parenthesized
            if the condition function applied to the node returns True.
        """
        s = self._visit_expr(n)
        if condition(n):
            return '(' + s + ')'
        else:
            return s

    def _parenthesize_unless_simple(self, n):
        """ Common use case for _parenthesize_if
        """
        return self._parenthesize_if(n, lambda d: not self._is_simple_node(d))

    def _is_simple_node(self, n):
        """ Returns True for nodes that are "simple" - i.e. nodes that always
            have higher precedence than operators.
        """
        return isinstance(n, (c_ast.Constant, c_ast.ID, c_ast.ArrayRef,
                              c_ast.StructRef, c_ast.FuncCall))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\provision.py ===
# testing/provision.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

from __future__ import annotations

import collections
import logging

from . import config
from . import engines
from . import util
from .. import exc
from .. import inspect
from ..engine import url as sa_url
from ..sql import ddl
from ..sql import schema


log = logging.getLogger(__name__)

FOLLOWER_IDENT = None


class register:
    def __init__(self, decorator=None):
        self.fns = {}
        self.decorator = decorator

    @classmethod
    def init(cls, fn):
        return register().for_db("*")(fn)

    @classmethod
    def init_decorator(cls, decorator):
        return register(decorator).for_db("*")

    def for_db(self, *dbnames):
        def decorate(fn):
            if self.decorator:
                fn = self.decorator(fn)
            for dbname in dbnames:
                self.fns[dbname] = fn
            return self

        return decorate

    def __call__(self, cfg, *arg, **kw):
        if isinstance(cfg, str):
            url = sa_url.make_url(cfg)
        elif isinstance(cfg, sa_url.URL):
            url = cfg
        else:
            url = cfg.db.url
        backend = url.get_backend_name()
        if backend in self.fns:
            return self.fns[backend](cfg, *arg, **kw)
        else:
            return self.fns["*"](cfg, *arg, **kw)


def create_follower_db(follower_ident):
    for cfg in _configs_for_db_operation():
        log.info("CREATE database %s, URI %r", follower_ident, cfg.db.url)
        create_db(cfg, cfg.db, follower_ident)


def setup_config(db_url, options, file_config, follower_ident):
    # load the dialect, which should also have it set up its provision
    # hooks

    dialect = sa_url.make_url(db_url).get_dialect()

    dialect.load_provisioning()

    if follower_ident:
        db_url = follower_url_from_main(db_url, follower_ident)
    db_opts = {}
    update_db_opts(db_url, db_opts, options)
    db_opts["scope"] = "global"
    eng = engines.testing_engine(db_url, db_opts)
    post_configure_engine(db_url, eng, follower_ident)
    eng.connect().close()

    cfg = config.Config.register(eng, db_opts, options, file_config)

    # a symbolic name that tests can use if they need to disambiguate
    # names across databases
    if follower_ident:
        config.ident = follower_ident

    if follower_ident:
        configure_follower(cfg, follower_ident)
    return cfg


def drop_follower_db(follower_ident):
    for cfg in _configs_for_db_operation():
        log.info("DROP database %s, URI %r", follower_ident, cfg.db.url)
        drop_db(cfg, cfg.db, follower_ident)


def generate_db_urls(db_urls, extra_drivers):
    """Generate a set of URLs to test given configured URLs plus additional
    driver names.

    Given:

    .. sourcecode:: text

        --dburi postgresql://db1  \
        --dburi postgresql://db2  \
        --dburi postgresql://db2  \
        --dbdriver=psycopg2 --dbdriver=asyncpg?async_fallback=true

    Noting that the default postgresql driver is psycopg2,  the output
    would be:

    .. sourcecode:: text

        postgresql+psycopg2://db1
        postgresql+asyncpg://db1
        postgresql+psycopg2://db2
        postgresql+psycopg2://db3

    That is, for the driver in a --dburi, we want to keep that and use that
    driver for each URL it's part of .   For a driver that is only
    in --dbdrivers, we want to use it just once for one of the URLs.
    for a driver that is both coming from --dburi as well as --dbdrivers,
    we want to keep it in that dburi.

    Driver specific query options can be specified by added them to the
    driver name. For example, to enable the async fallback option for
    asyncpg::

    .. sourcecode:: text

        --dburi postgresql://db1  \
        --dbdriver=asyncpg?async_fallback=true

    """
    urls = set()

    backend_to_driver_we_already_have = collections.defaultdict(set)

    urls_plus_dialects = [
        (url_obj, url_obj.get_dialect())
        for url_obj in [sa_url.make_url(db_url) for db_url in db_urls]
    ]

    for url_obj, dialect in urls_plus_dialects:
        # use get_driver_name instead of dialect.driver to account for
        # "_async" virtual drivers like oracledb and psycopg
        driver_name = url_obj.get_driver_name()
        backend_to_driver_we_already_have[dialect.name].add(driver_name)

    backend_to_driver_we_need = {}

    for url_obj, dialect in urls_plus_dialects:
        backend = dialect.name
        dialect.load_provisioning()

        if backend not in backend_to_driver_we_need:
            backend_to_driver_we_need[backend] = extra_per_backend = set(
                extra_drivers
            ).difference(backend_to_driver_we_already_have[backend])
        else:
            extra_per_backend = backend_to_driver_we_need[backend]

        for driver_url in _generate_driver_urls(url_obj, extra_per_backend):
            if driver_url in urls:
                continue
            urls.add(driver_url)
            yield driver_url


def _generate_driver_urls(url, extra_drivers):
    main_driver = url.get_driver_name()
    extra_drivers.discard(main_driver)

    url = generate_driver_url(url, main_driver, "")
    yield url

    for drv in list(extra_drivers):
        if "?" in drv:
            driver_only, query_str = drv.split("?", 1)

        else:
            driver_only = drv
            query_str = None

        new_url = generate_driver_url(url, driver_only, query_str)
        if new_url:
            extra_drivers.remove(drv)

            yield new_url


@register.init
def generate_driver_url(url, driver, query_str):
    backend = url.get_backend_name()

    new_url = url.set(
        drivername="%s+%s" % (backend, driver),
    )
    if query_str:
        new_url = new_url.update_query_string(query_str)

    try:
        new_url.get_dialect()
    except exc.NoSuchModuleError:
        return None
    else:
        return new_url


def _configs_for_db_operation():
    hosts = set()

    for cfg in config.Config.all_configs():
        cfg.db.dispose()

    for cfg in config.Config.all_configs():
        url = cfg.db.url
        backend = url.get_backend_name()
        host_conf = (backend, url.username, url.host, url.database)

        if host_conf not in hosts:
            yield cfg
            hosts.add(host_conf)

    for cfg in config.Config.all_configs():
        cfg.db.dispose()


@register.init
def drop_all_schema_objects_pre_tables(cfg, eng):
    pass


@register.init
def drop_all_schema_objects_post_tables(cfg, eng):
    pass


def drop_all_schema_objects(cfg, eng):
    drop_all_schema_objects_pre_tables(cfg, eng)

    drop_views(cfg, eng)

    if config.requirements.materialized_views.enabled:
        drop_materialized_views(cfg, eng)

    inspector = inspect(eng)

    consider_schemas = (None,)
    if config.requirements.schemas.enabled_for_config(cfg):
        consider_schemas += (cfg.test_schema, cfg.test_schema_2)
    util.drop_all_tables(eng, inspector, consider_schemas=consider_schemas)

    drop_all_schema_objects_post_tables(cfg, eng)

    if config.requirements.sequences.enabled_for_config(cfg):
        with eng.begin() as conn:
            for seq in inspector.get_sequence_names():
                conn.execute(ddl.DropSequence(schema.Sequence(seq)))
            if config.requirements.schemas.enabled_for_config(cfg):
                for schema_name in [cfg.test_schema, cfg.test_schema_2]:
                    for seq in inspector.get_sequence_names(
                        schema=schema_name
                    ):
                        conn.execute(
                            ddl.DropSequence(
                                schema.Sequence(seq, schema=schema_name)
                            )
                        )


def drop_views(cfg, eng):
    inspector = inspect(eng)

    try:
        view_names = inspector.get_view_names()
    except NotImplementedError:
        pass
    else:
        with eng.begin() as conn:
            for vname in view_names:
                conn.execute(
                    ddl._DropView(schema.Table(vname, schema.MetaData()))
                )

    if config.requirements.schemas.enabled_for_config(cfg):
        try:
            view_names = inspector.get_view_names(schema=cfg.test_schema)
        except NotImplementedError:
            pass
        else:
            with eng.begin() as conn:
                for vname in view_names:
                    conn.execute(
                        ddl._DropView(
                            schema.Table(
                                vname,
                                schema.MetaData(),
                                schema=cfg.test_schema,
                            )
                        )
                    )


def drop_materialized_views(cfg, eng):
    inspector = inspect(eng)

    mview_names = inspector.get_materialized_view_names()

    with eng.begin() as conn:
        for vname in mview_names:
            conn.exec_driver_sql(f"DROP MATERIALIZED VIEW {vname}")

    if config.requirements.schemas.enabled_for_config(cfg):
        mview_names = inspector.get_materialized_view_names(
            schema=cfg.test_schema
        )
        with eng.begin() as conn:
            for vname in mview_names:
                conn.exec_driver_sql(
                    f"DROP MATERIALIZED VIEW {cfg.test_schema}.{vname}"
                )


@register.init
def create_db(cfg, eng, ident):
    """Dynamically create a database for testing.

    Used when a test run will employ multiple processes, e.g., when run
    via `tox` or `pytest -n4`.
    """
    raise NotImplementedError(
        "no DB creation routine for cfg: %s" % (eng.url,)
    )


@register.init
def drop_db(cfg, eng, ident):
    """Drop a database that we dynamically created for testing."""
    raise NotImplementedError("no DB drop routine for cfg: %s" % (eng.url,))


def _adapt_update_db_opts(fn):
    insp = util.inspect_getfullargspec(fn)
    if len(insp.args) == 3:
        return fn
    else:
        return lambda db_url, db_opts, _options: fn(db_url, db_opts)


@register.init_decorator(_adapt_update_db_opts)
def update_db_opts(db_url, db_opts, options):
    """Set database options (db_opts) for a test database that we created."""


@register.init
def post_configure_engine(url, engine, follower_ident):
    """Perform extra steps after configuring an engine for testing.

    (For the internal dialects, currently only used by sqlite, oracle, mssql)
    """


@register.init
def follower_url_from_main(url, ident):
    """Create a connection URL for a dynamically-created test database.

    :param url: the connection URL specified when the test run was invoked
    :param ident: the pytest-xdist "worker identifier" to be used as the
                  database name
    """
    url = sa_url.make_url(url)
    return url.set(database=ident)


@register.init
def configure_follower(cfg, ident):
    """Create dialect-specific config settings for a follower database."""
    pass


@register.init
def run_reap_dbs(url, ident):
    """Remove databases that were created during the test process, after the
    process has ended.

    This is an optional step that is invoked for certain backends that do not
    reliably release locks on the database as long as a process is still in
    use. For the internal dialects, this is currently only necessary for
    mssql and oracle.
    """


def reap_dbs(idents_file):
    log.info("Reaping databases...")

    urls = collections.defaultdict(set)
    idents = collections.defaultdict(set)
    dialects = {}

    with open(idents_file) as file_:
        for line in file_:
            line = line.strip()
            db_name, db_url = line.split(" ")
            url_obj = sa_url.make_url(db_url)
            if db_name not in dialects:
                dialects[db_name] = url_obj.get_dialect()
                dialects[db_name].load_provisioning()
            url_key = (url_obj.get_backend_name(), url_obj.host)
            urls[url_key].add(db_url)
            idents[url_key].add(db_name)

    for url_key in urls:
        url = list(urls[url_key])[0]
        ident = idents[url_key]
        run_reap_dbs(url, ident)


@register.init
def temp_table_keyword_args(cfg, eng):
    """Specify keyword arguments for creating a temporary Table.

    Dialect-specific implementations of this method will return the
    kwargs that are passed to the Table method when creating a temporary
    table for testing, e.g., in the define_temp_tables method of the
    ComponentReflectionTest class in suite/test_reflection.py
    """
    raise NotImplementedError(
        "no temp table keyword args routine for cfg: %s" % (eng.url,)
    )


@register.init
def prepare_for_drop_tables(config, connection):
    pass


@register.init
def stop_test_class_outside_fixtures(config, db, testcls):
    pass


@register.init
def get_temp_table_name(cfg, eng, base_name):
    """Specify table name for creating a temporary Table.

    Dialect-specific implementations of this method will return the
    name to use when creating a temporary table for testing,
    e.g., in the define_temp_tables method of the
    ComponentReflectionTest class in suite/test_reflection.py

    Default to just the base name since that's what most dialects will
    use. The mssql dialect's implementation will need a "#" prepended.
    """
    return base_name


@register.init
def set_default_schema_on_connection(cfg, dbapi_connection, schema_name):
    raise NotImplementedError(
        "backend does not implement a schema name set function: %s"
        % (cfg.db.url,)
    )


@register.init
def upsert(
    cfg, table, returning, *, set_lambda=None, sort_by_parameter_order=False
):
    """return the backends insert..on conflict / on dupe etc. construct.

    while we should add a backend-neutral upsert construct as well, such as
    insert().upsert(), it's important that we continue to test the
    backend-specific insert() constructs since if we do implement
    insert().upsert(), that would be using a different codepath for the things
    we need to test like insertmanyvalues, etc.

    """
    raise NotImplementedError(
        f"backend does not include an upsert implementation: {cfg.db.url}"
    )


@register.init
def normalize_sequence(cfg, sequence):
    """Normalize sequence parameters for dialect that don't start with 1
    by default.

    The default implementation does nothing
    """
    return sequence

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\float16.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# This file is modified from https://github.com/microsoft/onnxconverter-common/blob/master/onnxconverter_common/float16.py
# Modifications:
# (1) Update default value of min_positive_val and max_finite_val
# (2) keep_io_types can be list of names
# (3) convert initializers if needed to preserve precision
# (4) add force_fp16_initializers option
# (5) handle Resize and GroupNorm with mixed float inputs
# (6) allow convert_float_to_float16 to accept model path

import itertools
import logging
import os
import tempfile

import numpy as np
import onnx
from onnx import AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, helper, numpy_helper
from onnx.shape_inference import infer_shapes, infer_shapes_path
from packaging import version

logger = logging.getLogger(__name__)


def _npfloat16_to_int(np_list):
    """
    Convert numpy float16 to python int.

    :param np_list: numpy float16 list
    :return int_list: python int list
    """
    return [int(bin(_.view("H"))[2:].zfill(16), 2) for _ in np_list]


def convert_np_to_float16(np_array, min_positive_val=5.96e-08, max_finite_val=65504.0):
    """
    Convert float32 numpy array to float16 without changing sign or finiteness.
    Positive values less than min_positive_val are mapped to min_positive_val.
    Positive finite values greater than max_finite_val are mapped to max_finite_val.
    Similar for negative values. NaN, 0, inf, and -inf are unchanged.
    """

    def between(a, b, c):
        return np.logical_and(a < b, b < c)

    if np_array[np.where(np_array > 0)].shape[0] > 0:
        positive_max = np_array[np.where(np_array > 0)].max()
        positive_min = np_array[np.where(np_array > 0)].min()
        if positive_max >= max_finite_val:
            logger.debug(f"the float32 number {positive_max} will be truncated to {max_finite_val}")
        if positive_min <= min_positive_val:
            logger.debug(f"the float32 number {positive_min} will be truncated to {min_positive_val}")

    if np_array[np.where(np_array < 0)].shape[0] > 0:
        negative_max = np_array[np.where(np_array < 0)].max()
        negative_min = np_array[np.where(np_array < 0)].min()
        if negative_min <= -max_finite_val:
            logger.debug(f"the float32 number {negative_min} will be truncated to {-max_finite_val}")
        if negative_max >= -min_positive_val:
            logger.debug(f"the float32 number {negative_max} will be truncated to {-min_positive_val}")

    np_array = np.where(between(0, np_array, min_positive_val), min_positive_val, np_array)
    np_array = np.where(between(-min_positive_val, np_array, 0), -min_positive_val, np_array)
    np_array = np.where(between(max_finite_val, np_array, float("inf")), max_finite_val, np_array)
    np_array = np.where(between(float("-inf"), np_array, -max_finite_val), -max_finite_val, np_array)
    return np.float16(np_array)


def convert_tensor_float_to_float16(tensor, min_positive_val=5.96e-08, max_finite_val=65504.0):
    """Convert tensor float to float16.

    Args:
        tensor (TensorProto): the tensor to convert.
        min_positive_val (float, optional): minimal positive value. Defaults to 1e-7.
        max_finite_val (float, optional): maximal finite value. Defaults to 1e4.

    Raises:
        ValueError: input type is not TensorProto.

    Returns:
        TensorProto: the converted tensor.
    """

    if not isinstance(tensor, TensorProto):
        raise ValueError(f"Expected input type is an ONNX TensorProto but got {type(tensor)}")

    if tensor.data_type == TensorProto.FLOAT:
        tensor.data_type = TensorProto.FLOAT16
        # convert float_data (float type) to float16 and write to int32_data
        if tensor.float_data:
            float16_data = convert_np_to_float16(np.array(tensor.float_data), min_positive_val, max_finite_val)
            int_list = _npfloat16_to_int(float16_data)
            tensor.int32_data[:] = int_list
            tensor.float_data[:] = []
        # convert raw_data (bytes type)
        if tensor.raw_data:
            # convert n.raw_data to float
            float32_list = np.frombuffer(tensor.raw_data, dtype="float32")
            # convert float to float16
            float16_list = convert_np_to_float16(float32_list, min_positive_val, max_finite_val)
            # convert float16 to bytes and write back to raw_data
            tensor.raw_data = float16_list.tobytes()
    return tensor


def make_value_info_from_tensor(tensor):
    shape = numpy_helper.to_array(tensor).shape
    return helper.make_tensor_value_info(tensor.name, tensor.data_type, shape)


DEFAULT_OP_BLOCK_LIST = [
    "ArrayFeatureExtractor",
    "Binarizer",
    "CastMap",
    "CategoryMapper",
    "DictVectorizer",
    "FeatureVectorizer",
    "Imputer",
    "LabelEncoder",
    "LinearClassifier",
    "LinearRegressor",
    "Normalizer",
    "OneHotEncoder",
    "RandomUniformLike",
    "SVMClassifier",
    "SVMRegressor",
    "Scaler",
    "TreeEnsembleClassifier",
    "TreeEnsembleRegressor",
    "TreeEnsemble",
    "ZipMap",
    "NonMaxSuppression",
    "TopK",
    "RoiAlign",
    "Range",
    "CumSum",
    "Min",
    "Max",
    "Upsample",
]


# Some operators has data type fixed as float for some inputs. Key is op_type, value is list of input indices
# Note that DirectML allows float16 gamma and beta in GroupNorm. Use force_fp16_inputs parameter could overwrite this.
ALWAYS_FLOAT_INPUTS = {"Resize": [2], "GroupNorm": [1, 2], "SkipGroupNorm": [1, 2]}


class InitializerTracker:
    """Class for keeping track of initializer."""

    def __init__(self, initializer: TensorProto):
        self.initializer = initializer
        self.fp32_nodes = []
        self.fp16_nodes = []

    def add_node(self, node: NodeProto, is_node_blocked):
        if is_node_blocked:
            self.fp32_nodes.append(node)
        else:
            self.fp16_nodes.append(node)


def convert_float_to_float16(
    model,
    min_positive_val=5.96e-08,
    max_finite_val=65504.0,
    keep_io_types=False,
    disable_shape_infer=False,
    op_block_list=None,
    node_block_list=None,
    force_fp16_initializers=False,
    force_fp16_inputs=None,
    use_bfloat16_as_blocked_nodes_dtype=False,
):
    """Convert tensor float type in the input ONNX model to tensor float16.

    Args:
        model (ModelProto or str): The ONNX model or path of the model to convert.
        min_positive_val (float, optional): minimal positive value. Defaults to 5.96e-08.
        max_finite_val (float, optional): maximal finite value of float16. Defaults to 65504.
        keep_io_types (Union[bool, List[str]], optional): It could be boolean or a list of float32 input/output names.
                                                          If True, model inputs/outputs should be left as float32.
                                                          Defaults to False.
        disable_shape_infer (bool, optional): Skips running onnx shape/type inference.
                                              Useful if shape inference has been done. Defaults to False.
        op_block_list (List[str], optional): List of op types to leave as float32.
                                             Defaults to None, which will use `float16.DEFAULT_OP_BLOCK_LIST`.
        node_block_list (List[str], optional): List of node names to leave as float32. Defaults to None.
        force_fp16_initializers(bool): force converting all float initializers to float16.
                                       Default to false, which will convert only the one needed to avoid precision loss.
        force_fp16_inputs(Dict[str, List[int]]): Force the conversion of the inputs of some operators to float16, even if
                                                 this script's preference it to keep them in float32.
    Raises:
        ValueError: input type is not ModelProto.

    Returns:
        ModelProto: converted model.
    """
    assert min_positive_val >= 5.96e-08, (
        "invalid min_positive_val. smallest positive float16 value: subnormal 5.96e-08, and normalized 6.104e-05"
    )
    assert max_finite_val <= float(np.finfo(np.float16).max), "invalid max_finite_val. largest float16 value: 65504"

    force_fp16_inputs_dict = {} if force_fp16_inputs is None else force_fp16_inputs

    if isinstance(model, str):
        model_path = model
        if version.parse(onnx.__version__) >= version.parse("1.8.0") and not disable_shape_infer:
            # shape_infer_model_path should be in the same folder of model_path
            with tempfile.NamedTemporaryFile(dir=os.path.dirname(model_path)) as tmpfile:
                shape_infer_model_path = tmpfile.name
                # infer_shapes_path can be used for model >2GB, and infer_shapes cannot.
                infer_shapes_path(model_path, shape_infer_model_path)
                model = onnx.load(shape_infer_model_path)
                disable_shape_infer = True
        else:
            model = onnx.load(model_path)

    if not isinstance(model, ModelProto):
        raise ValueError(f"Expected an ONNX ModelProto but got {type(model)}")

    func_infer_shape = None
    if not disable_shape_infer and version.parse(onnx.__version__) >= version.parse("1.2.0"):
        try:
            func_infer_shape = infer_shapes
        finally:
            pass

    # create blocklists
    if op_block_list is None:
        op_block_list = DEFAULT_OP_BLOCK_LIST
    if node_block_list is None:
        node_block_list = []
    op_block_list = set(op_block_list)
    node_block_list = set(node_block_list)

    logger.debug(
        f"fp16 parameters: min_positive_val={min_positive_val} max_finite_val={max_finite_val} keep_io_types={keep_io_types} disable_shape_infer={disable_shape_infer} op_block_list={op_block_list} node_block_list={node_block_list} force_fp16_initializers={force_fp16_initializers}"
    )

    # create a queue for BFS
    queue = []
    value_info_list = []
    node_list = []

    # Some operators (Like Resize or GroupNorm) have data type fixed as float for some input.
    # When it is converted to float16, there are mixed types: some inputs are float32 and some are float16.
    # This list keeps track of such nodes that are not in block list.
    mixed_float_type_node_list = []

    # type inference on input model
    if func_infer_shape is not None:
        model = func_infer_shape(model)
    queue.append(model)
    name_mapping = {}
    graph_io_to_skip = set()
    io_casts = set()

    fp32_inputs = [n.name for n in model.graph.input if n.type.tensor_type.elem_type == TensorProto.FLOAT]
    fp32_outputs = [n.name for n in model.graph.output if n.type.tensor_type.elem_type == TensorProto.FLOAT]
    if isinstance(keep_io_types, list):
        fp32_inputs = [n for n in fp32_inputs if n in keep_io_types]
        fp32_outputs = [n for n in fp32_outputs if n in keep_io_types]
    elif not keep_io_types:
        fp32_inputs = []
        fp32_outputs = []

    for i, n in enumerate(model.graph.input):
        if n.name in fp32_inputs:
            output_name = "graph_input_cast_" + str(i)
            name_mapping[n.name] = output_name
            graph_io_to_skip.add(n.name)

            node_name = "graph_input_cast" + str(i)
            new_value_info = model.graph.value_info.add()
            new_value_info.CopyFrom(n)
            new_value_info.name = output_name
            new_value_info.type.tensor_type.elem_type = TensorProto.FLOAT16
            # add Cast node (from tensor(float) to tensor(float16) after graph input
            new_node = [helper.make_node("Cast", [n.name], [output_name], to=TensorProto.FLOAT16, name=node_name)]
            model.graph.node.extend(new_node)
            value_info_list.append(new_value_info)
            io_casts.add(node_name)

    for i, n in enumerate(model.graph.output):
        if n.name in fp32_outputs:
            input_name = "graph_output_cast_" + str(i)
            name_mapping[n.name] = input_name
            graph_io_to_skip.add(n.name)

            node_name = "graph_output_cast" + str(i)
            # add Cast node (from tensor(float16) to tensor(float) before graph output
            new_value_info = model.graph.value_info.add()
            new_value_info.CopyFrom(n)
            new_value_info.name = input_name
            new_value_info.type.tensor_type.elem_type = TensorProto.FLOAT16
            new_node = [helper.make_node("Cast", [input_name], [n.name], to=1, name=node_name)]
            model.graph.node.extend(new_node)
            value_info_list.append(new_value_info)
            io_casts.add(node_name)

    fp32_initializers: dict[str, InitializerTracker] = {}
    while queue:
        next_level = []
        for q in queue:
            # if q is model, push q.graph (GraphProto)
            if isinstance(q, ModelProto):
                next_level.append(q.graph)
            # if q is model.graph, push q.node.attribute (AttributeProto)
            if isinstance(q, GraphProto):
                for n in q.initializer:  # TensorProto type
                    if n.data_type == TensorProto.FLOAT:
                        assert n.name not in fp32_initializers
                        fp32_initializers[n.name] = InitializerTracker(n)

                for n in q.node:
                    # if n is in the block list (doesn't support float16), no conversion for the node,
                    # and save the node for further processing
                    if n.name in io_casts:
                        continue
                    for i in range(len(n.input)):
                        if n.input[i] in name_mapping:
                            n.input[i] = name_mapping[n.input[i]]
                    for i in range(len(n.output)):
                        if n.output[i] in name_mapping:
                            n.output[i] = name_mapping[n.output[i]]

                    is_node_blocked = n.op_type in op_block_list or n.name in node_block_list
                    for i, input_name in enumerate(n.input):
                        if input_name in fp32_initializers:
                            # For Resize/GroupNorm, only the first input can be float16
                            use_fp32_weight = is_node_blocked or (
                                i in ALWAYS_FLOAT_INPUTS.get(n.op_type, [])
                                and i not in force_fp16_inputs_dict.get(n.op_type, [])
                            )
                            fp32_initializers[input_name].add_node(n, use_fp32_weight)

                    if is_node_blocked:
                        node_list.append(n)
                    else:
                        if n.op_type == "Cast":
                            for attr in n.attribute:
                                if attr.name == "to" and attr.i == TensorProto.FLOAT:
                                    attr.i = TensorProto.FLOAT16
                                    break

                        if n.op_type in [
                            "EyeLike",
                            "Multinomial",
                            "RandomNormal",
                            "RandomNormalLike",
                            "RandomUniform",
                            "RandomUniformLike",
                            "SequenceEmpty",
                            "Bernoulli",
                        ]:
                            has_dtype = False
                            for attr in n.attribute:
                                if attr.name == "dtype":
                                    has_dtype = True
                                    if attr.i == TensorProto.FLOAT:
                                        attr.i = TensorProto.FLOAT16

                            # The dtype attribute is optional and default is FLOAT in the following operators
                            # so we need add dtype attribute to specify the data type float16
                            if (n.op_type in ["RandomNormal", "RandomUniform", "SequenceEmpty"]) and not has_dtype:
                                n.attribute.extend([helper.make_attribute("dtype", TensorProto.FLOAT16)])

                        # For Resize/GroupNorm, attribute data type cannot be changed
                        if n.op_type not in ALWAYS_FLOAT_INPUTS or n.op_type in force_fp16_inputs_dict:
                            for attr in n.attribute:
                                next_level.append(attr)  # noqa: PERF402
                        else:
                            mixed_float_type_node_list.append(n)

            # if q is model.graph.node.attribute, push q.g and q.graphs (GraphProto)
            # and process node.attribute.t and node.attribute.tensors (TensorProto)
            if isinstance(q, AttributeProto):
                next_level.append(q.g)
                for n in q.graphs:
                    next_level.append(n)  # noqa: PERF402
                q.t.CopyFrom(convert_tensor_float_to_float16(q.t, min_positive_val, max_finite_val))
                for n in q.tensors:
                    n = convert_tensor_float_to_float16(n, min_positive_val, max_finite_val)  # noqa: PLW2901
            # if q is graph, process input, output and value_info (ValueInfoProto)
            if isinstance(q, GraphProto):
                # Note that float initializers tracked by fp32_initializers will be processed later.
                # for all ValueInfoProto with tensor(float) type in input, output and value_info, convert them to
                # tensor(float16) except map and seq(map). And save them in value_info_list for further processing
                for n in itertools.chain(q.input, q.output, q.value_info):
                    if n.type.tensor_type.elem_type == TensorProto.FLOAT:
                        if n.name not in graph_io_to_skip:
                            n.type.tensor_type.elem_type = TensorProto.FLOAT16
                            value_info_list.append(n)
                    if n.type.HasField("sequence_type"):
                        if n.type.sequence_type.elem_type.tensor_type.elem_type == TensorProto.FLOAT:
                            if n.name not in graph_io_to_skip:
                                n.type.sequence_type.elem_type.tensor_type.elem_type = TensorProto.FLOAT16
                                value_info_list.append(n)

        queue = next_level

    for value in fp32_initializers.values():
        # By default, to avoid precision loss, do not convert an initializer to fp16 when it is used only by fp32 nodes.
        if force_fp16_initializers or value.fp16_nodes:
            value.initializer = convert_tensor_float_to_float16(value.initializer, min_positive_val, max_finite_val)
            value_info_list.append(make_value_info_from_tensor(value.initializer))
            if value.fp32_nodes and not force_fp16_initializers:
                logger.info(
                    f"initializer is used by both fp32 and fp16 nodes. Consider add these nodes to block list:{value.fp16_nodes}"
                )

    # Some operators have data type fixed as float for some input. Add a float16 to float cast for those inputs.
    for node in mixed_float_type_node_list:
        for i, input_name in enumerate(node.input):
            if i not in ALWAYS_FLOAT_INPUTS[node.op_type] or i in force_fp16_inputs_dict.get(node.op_type, []):
                continue
            for value_info in value_info_list:
                if input_name == value_info.name:
                    # create new value_info for current node's new input name
                    new_value_info = model.graph.value_info.add()
                    new_value_info.CopyFrom(value_info)
                    output_name = node.name + "_input_cast_" + str(i)
                    new_value_info.name = output_name
                    new_value_info.type.tensor_type.elem_type = TensorProto.FLOAT
                    # add Cast node (from tensor(float16) to tensor(float) before current node
                    node_name = node.name + "_input_cast" + str(i)
                    new_node = [helper.make_node("Cast", [input_name], [output_name], to=1, name=node_name)]
                    model.graph.node.extend(new_node)
                    # change current node's input name
                    node.input[i] = output_name
                    break

    accuracy_type = TensorProto.BFLOAT16 if use_bfloat16_as_blocked_nodes_dtype else TensorProto.FLOAT
    # process the nodes in block list that doesn't support tensor(float16)
    for node in node_list:
        # if input's name is in the value_info_list meaning input is tensor(float16) type,
        # insert a float16 to float Cast node before the node,
        # change current node's input name and create new value_info for the new name
        for i in range(len(node.input)):
            input_name = node.input[i]
            for value_info in value_info_list:
                if input_name == value_info.name:
                    # create new value_info for current node's new input name
                    new_value_info = model.graph.value_info.add()
                    new_value_info.CopyFrom(value_info)
                    output_name = node.name + "_input_cast_" + str(i)
                    new_value_info.name = output_name
                    new_value_info.type.tensor_type.elem_type = accuracy_type
                    # add Cast node (from tensor(float16) to tensor(float) before current node
                    node_name = node.name + "_input_cast" + str(i)
                    new_node = [helper.make_node("Cast", [input_name], [output_name], to=accuracy_type, name=node_name)]
                    model.graph.node.extend(new_node)
                    # change current node's input name
                    node.input[i] = output_name
                    break
        # if output's name is in the value_info_list meaning output is tensor(float16) type, insert a float to
        # float16 Cast node after the node, change current node's output name and create new value_info for the new name
        for i in range(len(node.output)):
            output = node.output[i]
            for value_info in value_info_list:
                if output == value_info.name:
                    # create new value_info for current node's new output
                    new_value_info = model.graph.value_info.add()
                    new_value_info.CopyFrom(value_info)
                    input_name = node.name + "_output_cast_" + str(i)
                    new_value_info.name = input_name
                    new_value_info.type.tensor_type.elem_type = accuracy_type
                    # add Cast node (from tensor(float) to tensor(float16) after current node
                    node_name = node.name + "_output_cast" + str(i)
                    new_node = [helper.make_node("Cast", [input_name], [output], to=10, name=node_name)]
                    model.graph.node.extend(new_node)
                    # change current node's input name
                    node.output[i] = input_name
                    break
    return model


def float_to_float16_max_diff(tensor, min_positive_val=5.96e-08, max_finite_val=65504.0):
    """Measure the maximum absolute difference after converting a float tensor to float16."""
    if not isinstance(tensor, TensorProto):
        raise ValueError(f"Expected input type is an ONNX TensorProto but got {type(tensor)}")
    if tensor.data_type != TensorProto.FLOAT:
        raise ValueError("Expected tensor data type is float.")

    float32_data = None
    if tensor.float_data:
        float32_data = np.array(tensor.float_data)

    if tensor.raw_data:
        float32_data = np.frombuffer(tensor.raw_data, dtype="float32")

    if float32_data is None:
        raise RuntimeError("external data not loaded!")

    float16_data = convert_np_to_float16(float32_data, min_positive_val, max_finite_val)
    return np.amax(np.abs(float32_data - np.float32(float16_data)))