
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\common.py ===
"""
A module containing deprecated matrix mixin classes.

The classes in this module are deprecated and will be removed in a future
release. They are kept here for backwards compatibility in case downstream
code was subclassing them.

Importing anything else from this module is deprecated so anything here
should either not be used or should be imported from somewhere else.
"""
from __future__ import annotations
from collections import defaultdict
from collections.abc import Iterable
from inspect import isfunction
from functools import reduce

from sympy.assumptions.refine import refine
from sympy.core import SympifyError, Add
from sympy.core.basic import Atom
from sympy.core.decorators import call_highest_priority
from sympy.core.logic import fuzzy_and, FuzzyBool
from sympy.core.numbers import Integer
from sympy.core.mod import Mod
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import Abs, re, im
from sympy.utilities.exceptions import sympy_deprecation_warning
from .utilities import _dotprodsimp, _simplify
from sympy.polys.polytools import Poly
from sympy.utilities.iterables import flatten, is_sequence
from sympy.utilities.misc import as_int, filldedent
from sympy.tensor.array import NDimArray

from .utilities import _get_intermediate_simp_bool


# These exception types were previously defined in this module but were moved
# to exceptions.py. We reimport them here for backwards compatibility in case
# downstream code was importing them from here.
from .exceptions import ( # noqa: F401
    MatrixError, ShapeError, NonSquareMatrixError, NonInvertibleMatrixError,
    NonPositiveDefiniteMatrixError
)


_DEPRECATED_MIXINS = (
    'MatrixShaping',
    'MatrixSpecial',
    'MatrixProperties',
    'MatrixOperations',
    'MatrixArithmetic',
    'MatrixCommon',
    'MatrixDeterminant',
    'MatrixReductions',
    'MatrixSubspaces',
    'MatrixEigen',
    'MatrixCalculus',
    'MatrixDeprecated',
)


class _MatrixDeprecatedMeta(type):

    #
    # Override the default __instancecheck__ implementation to ensure that
    # e.g. isinstance(M, MatrixCommon) still works when M is one of the
    # matrix classes. Matrix no longer inherits from MatrixCommon so
    # isinstance(M, MatrixCommon) would now return False by default.
    #
    # There were lots of places in the codebase where this was being done
    # so it seems likely that downstream code may be doing it too. All use
    # of these mixins is deprecated though so we give a deprecation warning
    # unconditionally if they are being used with isinstance.
    #
    # Any code seeing this deprecation warning should be changed to use
    # isinstance(M, MatrixBase) instead which also works in previous versions
    # of SymPy.
    #

    def __instancecheck__(cls, instance):

        sympy_deprecation_warning(
            f"""
            Checking whether an object is an instance of {cls.__name__} is
            deprecated.

            Use `isinstance(obj, Matrix)` instead of `isinstance(obj, {cls.__name__})`.
            """,
            deprecated_since_version="1.13",
            active_deprecations_target="deprecated-matrix-mixins",
            stacklevel=3,
        )

        from sympy.matrices.matrixbase import MatrixBase
        from sympy.matrices.matrices import (
            MatrixDeterminant,
            MatrixReductions,
            MatrixSubspaces,
            MatrixEigen,
            MatrixCalculus,
            MatrixDeprecated
        )

        all_mixins = (
            MatrixRequired,
            MatrixShaping,
            MatrixSpecial,
            MatrixProperties,
            MatrixOperations,
            MatrixArithmetic,
            MatrixCommon,
            MatrixDeterminant,
            MatrixReductions,
            MatrixSubspaces,
            MatrixEigen,
            MatrixCalculus,
            MatrixDeprecated
        )

        if cls in all_mixins and isinstance(instance, MatrixBase):
            return True
        else:
            return super().__instancecheck__(instance)


class MatrixRequired(metaclass=_MatrixDeprecatedMeta):
    """Deprecated mixin class for making matrix classes."""

    rows: int
    cols: int
    _simplify = None

    def __init_subclass__(cls, **kwargs):

        # Warn if any downstream code is subclassing this class or any of the
        # deprecated mixin classes that are all ultimately subclasses of this
        # class.
        #
        # We don't want to warn about the deprecated mixins themselves being
        # created, but only about them being used as mixins by downstream code.
        # Otherwise just importing this module would trigger a warning.
        # Ultimately the whole module should be deprecated and removed but for
        # SymPy 1.13 it is premature to do that given that this module was the
        # main way to import matrix exception types in all previous versions.

        if cls.__name__ not in _DEPRECATED_MIXINS:
            sympy_deprecation_warning(
                f"""
                Inheriting from the Matrix mixin classes is deprecated.

                The class {cls.__name__} is subclassing a deprecated mixin.
                """,
                deprecated_since_version="1.13",
                active_deprecations_target="deprecated-matrix-mixins",
                stacklevel=3,
            )

        super().__init_subclass__(**kwargs)

    @classmethod
    def _new(cls, *args, **kwargs):
        """`_new` must, at minimum, be callable as
        `_new(rows, cols, mat) where mat is a flat list of the
        elements of the matrix."""
        raise NotImplementedError("Subclasses must implement this.")

    def __eq__(self, other):
        raise NotImplementedError("Subclasses must implement this.")

    def __getitem__(self, key):
        """Implementations of __getitem__ should accept ints, in which
        case the matrix is indexed as a flat list, tuples (i,j) in which
        case the (i,j) entry is returned, slices, or mixed tuples (a,b)
        where a and b are any combination of slices and integers."""
        raise NotImplementedError("Subclasses must implement this.")

    def __len__(self):
        """The total number of entries in the matrix."""
        raise NotImplementedError("Subclasses must implement this.")

    @property
    def shape(self):
        raise NotImplementedError("Subclasses must implement this.")


class MatrixShaping(MatrixRequired):
    """Provides basic matrix shaping and extracting of submatrices"""

    def _eval_col_del(self, col):
        def entry(i, j):
            return self[i, j] if j < col else self[i, j + 1]
        return self._new(self.rows, self.cols - 1, entry)

    def _eval_col_insert(self, pos, other):

        def entry(i, j):
            if j < pos:
                return self[i, j]
            elif pos <= j < pos + other.cols:
                return other[i, j - pos]
            return self[i, j - other.cols]

        return self._new(self.rows, self.cols + other.cols, entry)

    def _eval_col_join(self, other):
        rows = self.rows

        def entry(i, j):
            if i < rows:
                return self[i, j]
            return other[i - rows, j]

        return classof(self, other)._new(self.rows + other.rows, self.cols,
                                         entry)

    def _eval_extract(self, rowsList, colsList):
        mat = list(self)
        cols = self.cols
        indices = (i * cols + j for i in rowsList for j in colsList)
        return self._new(len(rowsList), len(colsList),
                         [mat[i] for i in indices])

    def _eval_get_diag_blocks(self):
        sub_blocks = []

        def recurse_sub_blocks(M):
            for i in range(1, M.shape[0] + 1):
                if i == 1:
                    to_the_right = M[0, i:]
                    to_the_bottom = M[i:, 0]
                else:
                    to_the_right = M[:i, i:]
                    to_the_bottom = M[i:, :i]
                if any(to_the_right) or any(to_the_bottom):
                    continue
                sub_blocks.append(M[:i, :i])
                if M.shape != M[:i, :i].shape:
                    recurse_sub_blocks(M[i:, i:])
                return

        recurse_sub_blocks(self)
        return sub_blocks

    def _eval_row_del(self, row):
        def entry(i, j):
            return self[i, j] if i < row else self[i + 1, j]
        return self._new(self.rows - 1, self.cols, entry)

    def _eval_row_insert(self, pos, other):
        entries = list(self)
        insert_pos = pos * self.cols
        entries[insert_pos:insert_pos] = list(other)
        return self._new(self.rows + other.rows, self.cols, entries)

    def _eval_row_join(self, other):
        cols = self.cols

        def entry(i, j):
            if j < cols:
                return self[i, j]
            return other[i, j - cols]

        return classof(self, other)._new(self.rows, self.cols + other.cols,
                                         entry)

    def _eval_tolist(self):
        return [list(self[i,:]) for i in range(self.rows)]

    def _eval_todok(self):
        dok = {}
        rows, cols = self.shape
        for i in range(rows):
            for j in range(cols):
                val = self[i, j]
                if val != self.zero:
                    dok[i, j] = val
        return dok

    def _eval_vec(self):
        rows = self.rows

        def entry(n, _):
            # we want to read off the columns first
            j = n // rows
            i = n - j * rows
            return self[i, j]

        return self._new(len(self), 1, entry)

    def _eval_vech(self, diagonal):
        c = self.cols
        v = []
        if diagonal:
            for j in range(c):
                for i in range(j, c):
                    v.append(self[i, j])
        else:
            for j in range(c):
                for i in range(j + 1, c):
                    v.append(self[i, j])
        return self._new(len(v), 1, v)

    def col_del(self, col):
        """Delete the specified column."""
        if col < 0:
            col += self.cols
        if not 0 <= col < self.cols:
            raise IndexError("Column {} is out of range.".format(col))
        return self._eval_col_del(col)

    def col_insert(self, pos, other):
        """Insert one or more columns at the given column position.

        Examples
        ========

        >>> from sympy import zeros, ones
        >>> M = zeros(3)
        >>> V = ones(3, 1)
        >>> M.col_insert(1, V)
        Matrix([
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0]])

        See Also
        ========

        col
        row_insert
        """
        # Allows you to build a matrix even if it is null matrix
        if not self:
            return type(self)(other)

        pos = as_int(pos)

        if pos < 0:
            pos = self.cols + pos
        if pos < 0:
            pos = 0
        elif pos > self.cols:
            pos = self.cols

        if self.rows != other.rows:
            raise ShapeError(
                "The matrices have incompatible number of rows ({} and {})"
                .format(self.rows, other.rows))

        return self._eval_col_insert(pos, other)

    def col_join(self, other):
        """Concatenates two matrices along self's last and other's first row.

        Examples
        ========

        >>> from sympy import zeros, ones
        >>> M = zeros(3)
        >>> V = ones(1, 3)
        >>> M.col_join(V)
        Matrix([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 1]])

        See Also
        ========

        col
        row_join
        """
        # A null matrix can always be stacked (see  #10770)
        if self.rows == 0 and self.cols != other.cols:
            return self._new(0, other.cols, []).col_join(other)

        if self.cols != other.cols:
            raise ShapeError(
                "The matrices have incompatible number of columns ({} and {})"
                .format(self.cols, other.cols))
        return self._eval_col_join(other)

    def col(self, j):
        """Elementary column selector.

        Examples
        ========

        >>> from sympy import eye
        >>> eye(2).col(0)
        Matrix([
        [1],
        [0]])

        See Also
        ========

        row
        col_del
        col_join
        col_insert
        """
        return self[:, j]

    def extract(self, rowsList, colsList):
        r"""Return a submatrix by specifying a list of rows and columns.
        Negative indices can be given. All indices must be in the range
        $-n \le i < n$ where $n$ is the number of rows or columns.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(4, 3, range(12))
        >>> m
        Matrix([
        [0,  1,  2],
        [3,  4,  5],
        [6,  7,  8],
        [9, 10, 11]])
        >>> m.extract([0, 1, 3], [0, 1])
        Matrix([
        [0,  1],
        [3,  4],
        [9, 10]])

        Rows or columns can be repeated:

        >>> m.extract([0, 0, 1], [-1])
        Matrix([
        [2],
        [2],
        [5]])

        Every other row can be taken by using range to provide the indices:

        >>> m.extract(range(0, m.rows, 2), [-1])
        Matrix([
        [2],
        [8]])

        RowsList or colsList can also be a list of booleans, in which case
        the rows or columns corresponding to the True values will be selected:

        >>> m.extract([0, 1, 2, 3], [True, False, True])
        Matrix([
        [0,  2],
        [3,  5],
        [6,  8],
        [9, 11]])
        """

        if not is_sequence(rowsList) or not is_sequence(colsList):
            raise TypeError("rowsList and colsList must be iterable")
        # ensure rowsList and colsList are lists of integers
        if rowsList and all(isinstance(i, bool) for i in rowsList):
            rowsList = [index for index, item in enumerate(rowsList) if item]
        if colsList and all(isinstance(i, bool) for i in colsList):
            colsList = [index for index, item in enumerate(colsList) if item]

        # ensure everything is in range
        rowsList = [a2idx(k, self.rows) for k in rowsList]
        colsList = [a2idx(k, self.cols) for k in colsList]

        return self._eval_extract(rowsList, colsList)

    def get_diag_blocks(self):
        """Obtains the square sub-matrices on the main diagonal of a square matrix.

        Useful for inverting symbolic matrices or solving systems of
        linear equations which may be decoupled by having a block diagonal
        structure.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x, y, z
        >>> A = Matrix([[1, 3, 0, 0], [y, z*z, 0, 0], [0, 0, x, 0], [0, 0, 0, 0]])
        >>> a1, a2, a3 = A.get_diag_blocks()
        >>> a1
        Matrix([
        [1,    3],
        [y, z**2]])
        >>> a2
        Matrix([[x]])
        >>> a3
        Matrix([[0]])

        """
        return self._eval_get_diag_blocks()

    @classmethod
    def hstack(cls, *args):
        """Return a matrix formed by joining args horizontally (i.e.
        by repeated application of row_join).

        Examples
        ========

        >>> from sympy import Matrix, eye
        >>> Matrix.hstack(eye(2), 2*eye(2))
        Matrix([
        [1, 0, 2, 0],
        [0, 1, 0, 2]])
        """
        if len(args) == 0:
            return cls._new()

        kls = type(args[0])
        return reduce(kls.row_join, args)

    def reshape(self, rows, cols):
        """Reshape the matrix. Total number of elements must remain the same.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(2, 3, lambda i, j: 1)
        >>> m
        Matrix([
        [1, 1, 1],
        [1, 1, 1]])
        >>> m.reshape(1, 6)
        Matrix([[1, 1, 1, 1, 1, 1]])
        >>> m.reshape(3, 2)
        Matrix([
        [1, 1],
        [1, 1],
        [1, 1]])

        """
        if self.rows * self.cols != rows * cols:
            raise ValueError("Invalid reshape parameters %d %d" % (rows, cols))
        return self._new(rows, cols, lambda i, j: self[i * cols + j])

    def row_del(self, row):
        """Delete the specified row."""
        if row < 0:
            row += self.rows
        if not 0 <= row < self.rows:
            raise IndexError("Row {} is out of range.".format(row))

        return self._eval_row_del(row)

    def row_insert(self, pos, other):
        """Insert one or more rows at the given row position.

        Examples
        ========

        >>> from sympy import zeros, ones
        >>> M = zeros(3)
        >>> V = ones(1, 3)
        >>> M.row_insert(1, V)
        Matrix([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0]])

        See Also
        ========

        row
        col_insert
        """
        # Allows you to build a matrix even if it is null matrix
        if not self:
            return self._new(other)

        pos = as_int(pos)

        if pos < 0:
            pos = self.rows + pos
        if pos < 0:
            pos = 0
        elif pos > self.rows:
            pos = self.rows

        if self.cols != other.cols:
            raise ShapeError(
                "The matrices have incompatible number of columns ({} and {})"
                .format(self.cols, other.cols))

        return self._eval_row_insert(pos, other)

    def row_join(self, other):
        """Concatenates two matrices along self's last and rhs's first column

        Examples
        ========

        >>> from sympy import zeros, ones
        >>> M = zeros(3)
        >>> V = ones(3, 1)
        >>> M.row_join(V)
        Matrix([
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1]])

        See Also
        ========

        row
        col_join
        """
        # A null matrix can always be stacked (see  #10770)
        if self.cols == 0 and self.rows != other.rows:
            return self._new(other.rows, 0, []).row_join(other)

        if self.rows != other.rows:
            raise ShapeError(
                "The matrices have incompatible number of rows ({} and {})"
                .format(self.rows, other.rows))
        return self._eval_row_join(other)

    def diagonal(self, k=0):
        """Returns the kth diagonal of self. The main diagonal
        corresponds to `k=0`; diagonals above and below correspond to
        `k > 0` and `k < 0`, respectively. The values of `self[i, j]`
        for which `j - i = k`, are returned in order of increasing
        `i + j`, starting with `i + j = |k|`.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(3, 3, lambda i, j: j - i); m
        Matrix([
        [ 0,  1, 2],
        [-1,  0, 1],
        [-2, -1, 0]])
        >>> _.diagonal()
        Matrix([[0, 0, 0]])
        >>> m.diagonal(1)
        Matrix([[1, 1]])
        >>> m.diagonal(-2)
        Matrix([[-2]])

        Even though the diagonal is returned as a Matrix, the element
        retrieval can be done with a single index:

        >>> Matrix.diag(1, 2, 3).diagonal()[1]  # instead of [0, 1]
        2

        See Also
        ========

        diag
        """
        rv = []
        k = as_int(k)
        r = 0 if k > 0 else -k
        c = 0 if r else k
        while True:
            if r == self.rows or c == self.cols:
                break
            rv.append(self[r, c])
            r += 1
            c += 1
        if not rv:
            raise ValueError(filldedent('''
            The %s diagonal is out of range [%s, %s]''' % (
            k, 1 - self.rows, self.cols - 1)))
        return self._new(1, len(rv), rv)

    def row(self, i):
        """Elementary row selector.

        Examples
        ========

        >>> from sympy import eye
        >>> eye(2).row(0)
        Matrix([[1, 0]])

        See Also
        ========

        col
        row_del
        row_join
        row_insert
        """
        return self[i, :]

    @property
    def shape(self):
        """The shape (dimensions) of the matrix as the 2-tuple (rows, cols).

        Examples
        ========

        >>> from sympy import zeros
        >>> M = zeros(2, 3)
        >>> M.shape
        (2, 3)
        >>> M.rows
        2
        >>> M.cols
        3
        """
        return (self.rows, self.cols)

    def todok(self):
        """Return the matrix as dictionary of keys.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix.eye(3)
        >>> M.todok()
        {(0, 0): 1, (1, 1): 1, (2, 2): 1}
        """
        return self._eval_todok()

    def tolist(self):
        """Return the Matrix as a nested Python list.

        Examples
        ========

        >>> from sympy import Matrix, ones
        >>> m = Matrix(3, 3, range(9))
        >>> m
        Matrix([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])
        >>> m.tolist()
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        >>> ones(3, 0).tolist()
        [[], [], []]

        When there are no rows then it will not be possible to tell how
        many columns were in the original matrix:

        >>> ones(0, 3).tolist()
        []

        """
        if not self.rows:
            return []
        if not self.cols:
            return [[] for i in range(self.rows)]
        return self._eval_tolist()

    def todod(M):
        """Returns matrix as dict of dicts containing non-zero elements of the Matrix

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix([[0, 1],[0, 3]])
        >>> A
        Matrix([
        [0, 1],
        [0, 3]])
        >>> A.todod()
        {0: {1: 1}, 1: {1: 3}}


        """
        rowsdict = {}
        Mlol = M.tolist()
        for i, Mi in enumerate(Mlol):
            row = {j: Mij for j, Mij in enumerate(Mi) if Mij}
            if row:
                rowsdict[i] = row
        return rowsdict

    def vec(self):
        """Return the Matrix converted into a one column matrix by stacking columns

        Examples
        ========

        >>> from sympy import Matrix
        >>> m=Matrix([[1, 3], [2, 4]])
        >>> m
        Matrix([
        [1, 3],
        [2, 4]])
        >>> m.vec()
        Matrix([
        [1],
        [2],
        [3],
        [4]])

        See Also
        ========

        vech
        """
        return self._eval_vec()

    def vech(self, diagonal=True, check_symmetry=True):
        """Reshapes the matrix into a column vector by stacking the
        elements in the lower triangle.

        Parameters
        ==========

        diagonal : bool, optional
            If ``True``, it includes the diagonal elements.

        check_symmetry : bool, optional
            If ``True``, it checks whether the matrix is symmetric.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m=Matrix([[1, 2], [2, 3]])
        >>> m
        Matrix([
        [1, 2],
        [2, 3]])
        >>> m.vech()
        Matrix([
        [1],
        [2],
        [3]])
        >>> m.vech(diagonal=False)
        Matrix([[2]])

        Notes
        =====

        This should work for symmetric matrices and ``vech`` can
        represent symmetric matrices in vector form with less size than
        ``vec``.

        See Also
        ========

        vec
        """
        if not self.is_square:
            raise NonSquareMatrixError

        if check_symmetry and not self.is_symmetric():
            raise ValueError("The matrix is not symmetric.")

        return self._eval_vech(diagonal)

    @classmethod
    def vstack(cls, *args):
        """Return a matrix formed by joining args vertically (i.e.
        by repeated application of col_join).

        Examples
        ========

        >>> from sympy import Matrix, eye
        >>> Matrix.vstack(eye(2), 2*eye(2))
        Matrix([
        [1, 0],
        [0, 1],
        [2, 0],
        [0, 2]])
        """
        if len(args) == 0:
            return cls._new()

        kls = type(args[0])
        return reduce(kls.col_join, args)


class MatrixSpecial(MatrixRequired):
    """Construction of special matrices"""

    @classmethod
    def _eval_diag(cls, rows, cols, diag_dict):
        """diag_dict is a defaultdict containing
        all the entries of the diagonal matrix."""
        def entry(i, j):
            return diag_dict[(i, j)]
        return cls._new(rows, cols, entry)

    @classmethod
    def _eval_eye(cls, rows, cols):
        vals = [cls.zero]*(rows*cols)
        vals[::cols+1] = [cls.one]*min(rows, cols)
        return cls._new(rows, cols, vals, copy=False)

    @classmethod
    def _eval_jordan_block(cls, size: int, eigenvalue, band='upper'):
        if band == 'lower':
            def entry(i, j):
                if i == j:
                    return eigenvalue
                elif j + 1 == i:
                    return cls.one
                return cls.zero
        else:
            def entry(i, j):
                if i == j:
                    return eigenvalue
                elif i + 1 == j:
                    return cls.one
                return cls.zero
        return cls._new(size, size, entry)

    @classmethod
    def _eval_ones(cls, rows, cols):
        def entry(i, j):
            return cls.one
        return cls._new(rows, cols, entry)

    @classmethod
    def _eval_zeros(cls, rows, cols):
        return cls._new(rows, cols, [cls.zero]*(rows*cols), copy=False)

    @classmethod
    def _eval_wilkinson(cls, n):
        def entry(i, j):
            return cls.one if i + 1 == j else cls.zero

        D = cls._new(2*n + 1, 2*n + 1, entry)

        wminus = cls.diag(list(range(-n, n + 1)), unpack=True) + D + D.T
        wplus = abs(cls.diag(list(range(-n, n + 1)), unpack=True)) + D + D.T

        return wminus, wplus

    @classmethod
    def diag(kls, *args, strict=False, unpack=True, rows=None, cols=None, **kwargs):
        """Returns a matrix with the specified diagonal.
        If matrices are passed, a block-diagonal matrix
        is created (i.e. the "direct sum" of the matrices).

        kwargs
        ======

        rows : rows of the resulting matrix; computed if
               not given.

        cols : columns of the resulting matrix; computed if
               not given.

        cls : class for the resulting matrix

        unpack : bool which, when True (default), unpacks a single
        sequence rather than interpreting it as a Matrix.

        strict : bool which, when False (default), allows Matrices to
        have variable-length rows.

        Examples
        ========

        >>> from sympy import Matrix
        >>> Matrix.diag(1, 2, 3)
        Matrix([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3]])

        The current default is to unpack a single sequence. If this is
        not desired, set `unpack=False` and it will be interpreted as
        a matrix.

        >>> Matrix.diag([1, 2, 3]) == Matrix.diag(1, 2, 3)
        True

        When more than one element is passed, each is interpreted as
        something to put on the diagonal. Lists are converted to
        matrices. Filling of the diagonal always continues from
        the bottom right hand corner of the previous item: this
        will create a block-diagonal matrix whether the matrices
        are square or not.

        >>> col = [1, 2, 3]
        >>> row = [[4, 5]]
        >>> Matrix.diag(col, row)
        Matrix([
        [1, 0, 0],
        [2, 0, 0],
        [3, 0, 0],
        [0, 4, 5]])

        When `unpack` is False, elements within a list need not all be
        of the same length. Setting `strict` to True would raise a
        ValueError for the following:

        >>> Matrix.diag([[1, 2, 3], [4, 5], [6]], unpack=False)
        Matrix([
        [1, 2, 3],
        [4, 5, 0],
        [6, 0, 0]])

        The type of the returned matrix can be set with the ``cls``
        keyword.

        >>> from sympy import ImmutableMatrix
        >>> from sympy.utilities.misc import func_name
        >>> func_name(Matrix.diag(1, cls=ImmutableMatrix))
        'ImmutableDenseMatrix'

        A zero dimension matrix can be used to position the start of
        the filling at the start of an arbitrary row or column:

        >>> from sympy import ones
        >>> r2 = ones(0, 2)
        >>> Matrix.diag(r2, 1, 2)
        Matrix([
        [0, 0, 1, 0],
        [0, 0, 0, 2]])

        See Also
        ========
        eye
        diagonal
        .dense.diag
        .expressions.blockmatrix.BlockMatrix
        .sparsetools.banded
       """
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.matrices.dense import Matrix
        from sympy.matrices import SparseMatrix
        klass = kwargs.get('cls', kls)
        if unpack and len(args) == 1 and is_sequence(args[0]) and \
                not isinstance(args[0], MatrixBase):
            args = args[0]

        # fill a default dict with the diagonal entries
        diag_entries = defaultdict(int)
        rmax = cmax = 0  # keep track of the biggest index seen
        for m in args:
            if isinstance(m, list):
                if strict:
                    # if malformed, Matrix will raise an error
                    _ = Matrix(m)
                    r, c = _.shape
                    m = _.tolist()
                else:
                    r, c, smat = SparseMatrix._handle_creation_inputs(m)
                    for (i, j), _ in smat.items():
                        diag_entries[(i + rmax, j + cmax)] = _
                    m = []  # to skip process below
            elif hasattr(m, 'shape'):  # a Matrix
                # convert to list of lists
                r, c = m.shape
                m = m.tolist()
            else:  # in this case, we're a single value
                diag_entries[(rmax, cmax)] = m
                rmax += 1
                cmax += 1
                continue
            # process list of lists
            for i, mi in enumerate(m):
                for j, _ in enumerate(mi):
                    diag_entries[(i + rmax, j + cmax)] = _
            rmax += r
            cmax += c
        if rows is None:
            rows, cols = cols, rows
        if rows is None:
            rows, cols = rmax, cmax
        else:
            cols = rows if cols is None else cols
        if rows < rmax or cols < cmax:
            raise ValueError(filldedent('''
                The constructed matrix is {} x {} but a size of {} x {}
                was specified.'''.format(rmax, cmax, rows, cols)))
        return klass._eval_diag(rows, cols, diag_entries)

    @classmethod
    def eye(kls, rows, cols=None, **kwargs):
        """Returns an identity matrix.

        Parameters
        ==========

        rows : rows of the matrix
        cols : cols of the matrix (if None, cols=rows)

        kwargs
        ======
        cls : class of the returned matrix
        """
        if cols is None:
            cols = rows
        if rows < 0 or cols < 0:
            raise ValueError("Cannot create a {} x {} matrix. "
                             "Both dimensions must be positive".format(rows, cols))
        klass = kwargs.get('cls', kls)
        rows, cols = as_int(rows), as_int(cols)

        return klass._eval_eye(rows, cols)

    @classmethod
    def jordan_block(kls, size=None, eigenvalue=None, *, band='upper', **kwargs):
        """Returns a Jordan block

        Parameters
        ==========

        size : Integer, optional
            Specifies the shape of the Jordan block matrix.

        eigenvalue : Number or Symbol
            Specifies the value for the main diagonal of the matrix.

            .. note::
                The keyword ``eigenval`` is also specified as an alias
                of this keyword, but it is not recommended to use.

                We may deprecate the alias in later release.

        band : 'upper' or 'lower', optional
            Specifies the position of the off-diagonal to put `1` s on.

        cls : Matrix, optional
            Specifies the matrix class of the output form.

            If it is not specified, the class type where the method is
            being executed on will be returned.

        Returns
        =======

        Matrix
            A Jordan block matrix.

        Raises
        ======

        ValueError
            If insufficient arguments are given for matrix size
            specification, or no eigenvalue is given.

        Examples
        ========

        Creating a default Jordan block:

        >>> from sympy import Matrix
        >>> from sympy.abc import x
        >>> Matrix.jordan_block(4, x)
        Matrix([
        [x, 1, 0, 0],
        [0, x, 1, 0],
        [0, 0, x, 1],
        [0, 0, 0, x]])

        Creating an alternative Jordan block matrix where `1` is on
        lower off-diagonal:

        >>> Matrix.jordan_block(4, x, band='lower')
        Matrix([
        [x, 0, 0, 0],
        [1, x, 0, 0],
        [0, 1, x, 0],
        [0, 0, 1, x]])

        Creating a Jordan block with keyword arguments

        >>> Matrix.jordan_block(size=4, eigenvalue=x)
        Matrix([
        [x, 1, 0, 0],
        [0, x, 1, 0],
        [0, 0, x, 1],
        [0, 0, 0, x]])

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Jordan_matrix
        """
        klass = kwargs.pop('cls', kls)

        eigenval = kwargs.get('eigenval', None)
        if eigenvalue is None and eigenval is None:
            raise ValueError("Must supply an eigenvalue")
        elif eigenvalue != eigenval and None not in (eigenval, eigenvalue):
            raise ValueError(
                "Inconsistent values are given: 'eigenval'={}, "
                "'eigenvalue'={}".format(eigenval, eigenvalue))
        else:
            if eigenval is not None:
                eigenvalue = eigenval

        if size is None:
            raise ValueError("Must supply a matrix size")

        size = as_int(size)
        return klass._eval_jordan_block(size, eigenvalue, band)

    @classmethod
    def ones(kls, rows, cols=None, **kwargs):
        """Returns a matrix of ones.

        Parameters
        ==========

        rows : rows of the matrix
        cols : cols of the matrix (if None, cols=rows)

        kwargs
        ======
        cls : class of the returned matrix
        """
        if cols is None:
            cols = rows
        klass = kwargs.get('cls', kls)
        rows, cols = as_int(rows), as_int(cols)

        return klass._eval_ones(rows, cols)

    @classmethod
    def zeros(kls, rows, cols=None, **kwargs):
        """Returns a matrix of zeros.

        Parameters
        ==========

        rows : rows of the matrix
        cols : cols of the matrix (if None, cols=rows)

        kwargs
        ======
        cls : class of the returned matrix
        """
        if cols is None:
            cols = rows
        if rows < 0 or cols < 0:
            raise ValueError("Cannot create a {} x {} matrix. "
                             "Both dimensions must be positive".format(rows, cols))
        klass = kwargs.get('cls', kls)
        rows, cols = as_int(rows), as_int(cols)

        return klass._eval_zeros(rows, cols)

    @classmethod
    def companion(kls, poly):
        """Returns a companion matrix of a polynomial.

        Examples
        ========

        >>> from sympy import Matrix, Poly, Symbol, symbols
        >>> x = Symbol('x')
        >>> c0, c1, c2, c3, c4 = symbols('c0:5')
        >>> p = Poly(c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + x**5, x)
        >>> Matrix.companion(p)
        Matrix([
        [0, 0, 0, 0, -c0],
        [1, 0, 0, 0, -c1],
        [0, 1, 0, 0, -c2],
        [0, 0, 1, 0, -c3],
        [0, 0, 0, 1, -c4]])
        """
        poly = kls._sympify(poly)
        if not isinstance(poly, Poly):
            raise ValueError("{} must be a Poly instance.".format(poly))
        if not poly.is_monic:
            raise ValueError("{} must be a monic polynomial.".format(poly))
        if not poly.is_univariate:
            raise ValueError(
                "{} must be a univariate polynomial.".format(poly))

        size = poly.degree()
        if not size >= 1:
            raise ValueError(
                "{} must have degree not less than 1.".format(poly))

        coeffs = poly.all_coeffs()
        def entry(i, j):
            if j == size - 1:
                return -coeffs[-1 - i]
            elif i == j + 1:
                return kls.one
            return kls.zero
        return kls._new(size, size, entry)


    @classmethod
    def wilkinson(kls, n, **kwargs):
        """Returns two square Wilkinson Matrix of size 2*n + 1
        $W_{2n + 1}^-, W_{2n + 1}^+ =$ Wilkinson(n)

        Examples
        ========

        >>> from sympy import Matrix
        >>> wminus, wplus = Matrix.wilkinson(3)
        >>> wminus
        Matrix([
        [-3,  1,  0, 0, 0, 0, 0],
        [ 1, -2,  1, 0, 0, 0, 0],
        [ 0,  1, -1, 1, 0, 0, 0],
        [ 0,  0,  1, 0, 1, 0, 0],
        [ 0,  0,  0, 1, 1, 1, 0],
        [ 0,  0,  0, 0, 1, 2, 1],
        [ 0,  0,  0, 0, 0, 1, 3]])
        >>> wplus
        Matrix([
        [3, 1, 0, 0, 0, 0, 0],
        [1, 2, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 1],
        [0, 0, 0, 0, 0, 1, 3]])

        References
        ==========

        .. [1] https://blogs.mathworks.com/cleve/2013/04/15/wilkinsons-matrices-2/
        .. [2] J. H. Wilkinson, The Algebraic Eigenvalue Problem, Claredon Press, Oxford, 1965, 662 pp.

        """
        klass = kwargs.get('cls', kls)
        n = as_int(n)
        return klass._eval_wilkinson(n)

class MatrixProperties(MatrixRequired):
    """Provides basic properties of a matrix."""

    def _eval_atoms(self, *types):
        result = set()
        for i in self:
            result.update(i.atoms(*types))
        return result

    def _eval_free_symbols(self):
        return set().union(*(i.free_symbols for i in self if i))

    def _eval_has(self, *patterns):
        return any(a.has(*patterns) for a in self)

    def _eval_is_anti_symmetric(self, simpfunc):
        if not all(simpfunc(self[i, j] + self[j, i]).is_zero for i in range(self.rows) for j in range(self.cols)):
            return False
        return True

    def _eval_is_diagonal(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if i != j and self[i, j]:
                    return False
        return True

    # _eval_is_hermitian is called by some general SymPy
    # routines and has a different *args signature.  Make
    # sure the names don't clash by adding `_matrix_` in name.
    def _eval_is_matrix_hermitian(self, simpfunc):
        mat = self._new(self.rows, self.cols, lambda i, j: simpfunc(self[i, j] - self[j, i].conjugate()))
        return mat.is_zero_matrix

    def _eval_is_Identity(self) -> FuzzyBool:
        def dirac(i, j):
            if i == j:
                return 1
            return 0

        return all(self[i, j] == dirac(i, j)
                for i in range(self.rows)
                for j in range(self.cols))

    def _eval_is_lower_hessenberg(self):
        return all(self[i, j].is_zero
                   for i in range(self.rows)
                   for j in range(i + 2, self.cols))

    def _eval_is_lower(self):
        return all(self[i, j].is_zero
                   for i in range(self.rows)
                   for j in range(i + 1, self.cols))

    def _eval_is_symbolic(self):
        return self.has(Symbol)

    def _eval_is_symmetric(self, simpfunc):
        mat = self._new(self.rows, self.cols, lambda i, j: simpfunc(self[i, j] - self[j, i]))
        return mat.is_zero_matrix

    def _eval_is_zero_matrix(self):
        if any(i.is_zero == False for i in self):
            return False
        if any(i.is_zero is None for i in self):
            return None
        return True

    def _eval_is_upper_hessenberg(self):
        return all(self[i, j].is_zero
                   for i in range(2, self.rows)
                   for j in range(min(self.cols, (i - 1))))

    def _eval_values(self):
        return [i for i in self if not i.is_zero]

    def _has_positive_diagonals(self):
        diagonal_entries = (self[i, i] for i in range(self.rows))
        return fuzzy_and(x.is_positive for x in diagonal_entries)

    def _has_nonnegative_diagonals(self):
        diagonal_entries = (self[i, i] for i in range(self.rows))
        return fuzzy_and(x.is_nonnegative for x in diagonal_entries)

    def atoms(self, *types):
        """Returns the atoms that form the current object.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import Matrix
        >>> Matrix([[x]])
        Matrix([[x]])
        >>> _.atoms()
        {x}
        >>> Matrix([[x, y], [y, x]])
        Matrix([
        [x, y],
        [y, x]])
        >>> _.atoms()
        {x, y}
        """

        types = tuple(t if isinstance(t, type) else type(t) for t in types)
        if not types:
            types = (Atom,)
        return self._eval_atoms(*types)

    @property
    def free_symbols(self):
        """Returns the free symbols within the matrix.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import Matrix
        >>> Matrix([[x], [1]]).free_symbols
        {x}
        """
        return self._eval_free_symbols()

    def has(self, *patterns):
        """Test whether any subexpression matches any of the patterns.

        Examples
        ========

        >>> from sympy import Matrix, SparseMatrix, Float
        >>> from sympy.abc import x, y
        >>> A = Matrix(((1, x), (0.2, 3)))
        >>> B = SparseMatrix(((1, x), (0.2, 3)))
        >>> A.has(x)
        True
        >>> A.has(y)
        False
        >>> A.has(Float)
        True
        >>> B.has(x)
        True
        >>> B.has(y)
        False
        >>> B.has(Float)
        True
        """
        return self._eval_has(*patterns)

    def is_anti_symmetric(self, simplify=True):
        """Check if matrix M is an antisymmetric matrix,
        that is, M is a square matrix with all M[i, j] == -M[j, i].

        When ``simplify=True`` (default), the sum M[i, j] + M[j, i] is
        simplified before testing to see if it is zero. By default,
        the SymPy simplify function is used. To use a custom function
        set simplify to a function that accepts a single argument which
        returns a simplified expression. To skip simplification, set
        simplify to False but note that although this will be faster,
        it may induce false negatives.

        Examples
        ========

        >>> from sympy import Matrix, symbols
        >>> m = Matrix(2, 2, [0, 1, -1, 0])
        >>> m
        Matrix([
        [ 0, 1],
        [-1, 0]])
        >>> m.is_anti_symmetric()
        True
        >>> x, y = symbols('x y')
        >>> m = Matrix(2, 3, [0, 0, x, -y, 0, 0])
        >>> m
        Matrix([
        [ 0, 0, x],
        [-y, 0, 0]])
        >>> m.is_anti_symmetric()
        False

        >>> from sympy.abc import x, y
        >>> m = Matrix(3, 3, [0, x**2 + 2*x + 1, y,
        ...                   -(x + 1)**2, 0, x*y,
        ...                   -y, -x*y, 0])

        Simplification of matrix elements is done by default so even
        though two elements which should be equal and opposite would not
        pass an equality test, the matrix is still reported as
        anti-symmetric:

        >>> m[0, 1] == -m[1, 0]
        False
        >>> m.is_anti_symmetric()
        True

        If ``simplify=False`` is used for the case when a Matrix is already
        simplified, this will speed things up. Here, we see that without
        simplification the matrix does not appear anti-symmetric:

        >>> print(m.is_anti_symmetric(simplify=False))
        None

        But if the matrix were already expanded, then it would appear
        anti-symmetric and simplification in the is_anti_symmetric routine
        is not needed:

        >>> m = m.expand()
        >>> m.is_anti_symmetric(simplify=False)
        True
        """
        # accept custom simplification
        simpfunc = simplify
        if not isfunction(simplify):
            simpfunc = _simplify if simplify else lambda x: x

        if not self.is_square:
            return False
        return self._eval_is_anti_symmetric(simpfunc)

    def is_diagonal(self):
        """Check if matrix is diagonal,
        that is matrix in which the entries outside the main diagonal are all zero.

        Examples
        ========

        >>> from sympy import Matrix, diag
        >>> m = Matrix(2, 2, [1, 0, 0, 2])
        >>> m
        Matrix([
        [1, 0],
        [0, 2]])
        >>> m.is_diagonal()
        True

        >>> m = Matrix(2, 2, [1, 1, 0, 2])
        >>> m
        Matrix([
        [1, 1],
        [0, 2]])
        >>> m.is_diagonal()
        False

        >>> m = diag(1, 2, 3)
        >>> m
        Matrix([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3]])
        >>> m.is_diagonal()
        True

        See Also
        ========

        is_lower
        is_upper
        sympy.matrices.matrixbase.MatrixCommon.is_diagonalizable
        diagonalize
        """
        return self._eval_is_diagonal()

    @property
    def is_weakly_diagonally_dominant(self):
        r"""Tests if the matrix is row weakly diagonally dominant.

        Explanation
        ===========

        A $n, n$ matrix $A$ is row weakly diagonally dominant if

        .. math::
            \left|A_{i, i}\right| \ge \sum_{j = 0, j \neq i}^{n-1}
            \left|A_{i, j}\right| \quad {\text{for all }}
            i \in \{ 0, ..., n-1 \}

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix([[3, -2, 1], [1, -3, 2], [-1, 2, 4]])
        >>> A.is_weakly_diagonally_dominant
        True

        >>> A = Matrix([[-2, 2, 1], [1, 3, 2], [1, -2, 0]])
        >>> A.is_weakly_diagonally_dominant
        False

        >>> A = Matrix([[-4, 2, 1], [1, 6, 2], [1, -2, 5]])
        >>> A.is_weakly_diagonally_dominant
        True

        Notes
        =====

        If you want to test whether a matrix is column diagonally
        dominant, you can apply the test after transposing the matrix.
        """
        if not self.is_square:
            return False

        rows, cols = self.shape

        def test_row(i):
            summation = self.zero
            for j in range(cols):
                if i != j:
                    summation += Abs(self[i, j])
            return (Abs(self[i, i]) - summation).is_nonnegative

        return fuzzy_and(test_row(i) for i in range(rows))

    @property
    def is_strongly_diagonally_dominant(self):
        r"""Tests if the matrix is row strongly diagonally dominant.

        Explanation
        ===========

        A $n, n$ matrix $A$ is row strongly diagonally dominant if

        .. math::
            \left|A_{i, i}\right| > \sum_{j = 0, j \neq i}^{n-1}
            \left|A_{i, j}\right| \quad {\text{for all }}
            i \in \{ 0, ..., n-1 \}

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix([[3, -2, 1], [1, -3, 2], [-1, 2, 4]])
        >>> A.is_strongly_diagonally_dominant
        False

        >>> A = Matrix([[-2, 2, 1], [1, 3, 2], [1, -2, 0]])
        >>> A.is_strongly_diagonally_dominant
        False

        >>> A = Matrix([[-4, 2, 1], [1, 6, 2], [1, -2, 5]])
        >>> A.is_strongly_diagonally_dominant
        True

        Notes
        =====

        If you want to test whether a matrix is column diagonally
        dominant, you can apply the test after transposing the matrix.
        """
        if not self.is_square:
            return False

        rows, cols = self.shape

        def test_row(i):
            summation = self.zero
            for j in range(cols):
                if i != j:
                    summation += Abs(self[i, j])
            return (Abs(self[i, i]) - summation).is_positive

        return fuzzy_and(test_row(i) for i in range(rows))

    @property
    def is_hermitian(self):
        """Checks if the matrix is Hermitian.

        In a Hermitian matrix element i,j is the complex conjugate of
        element j,i.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy import I
        >>> from sympy.abc import x
        >>> a = Matrix([[1, I], [-I, 1]])
        >>> a
        Matrix([
        [ 1, I],
        [-I, 1]])
        >>> a.is_hermitian
        True
        >>> a[0, 0] = 2*I
        >>> a.is_hermitian
        False
        >>> a[0, 0] = x
        >>> a.is_hermitian
        >>> a[0, 1] = a[1, 0]*I
        >>> a.is_hermitian
        False
        """
        if not self.is_square:
            return False

        return self._eval_is_matrix_hermitian(_simplify)

    @property
    def is_Identity(self) -> FuzzyBool:
        if not self.is_square:
            return False
        return self._eval_is_Identity()

    @property
    def is_lower_hessenberg(self):
        r"""Checks if the matrix is in the lower-Hessenberg form.

        The lower hessenberg matrix has zero entries
        above the first superdiagonal.

        Examples
        ========

        >>> from sympy import Matrix
        >>> a = Matrix([[1, 2, 0, 0], [5, 2, 3, 0], [3, 4, 3, 7], [5, 6, 1, 1]])
        >>> a
        Matrix([
        [1, 2, 0, 0],
        [5, 2, 3, 0],
        [3, 4, 3, 7],
        [5, 6, 1, 1]])
        >>> a.is_lower_hessenberg
        True

        See Also
        ========

        is_upper_hessenberg
        is_lower
        """
        return self._eval_is_lower_hessenberg()

    @property
    def is_lower(self):
        """Check if matrix is a lower triangular matrix. True can be returned
        even if the matrix is not square.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(2, 2, [1, 0, 0, 1])
        >>> m
        Matrix([
        [1, 0],
        [0, 1]])
        >>> m.is_lower
        True

        >>> m = Matrix(4, 3, [0, 0, 0, 2, 0, 0, 1, 4, 0, 6, 6, 5])
        >>> m
        Matrix([
        [0, 0, 0],
        [2, 0, 0],
        [1, 4, 0],
        [6, 6, 5]])
        >>> m.is_lower
        True

        >>> from sympy.abc import x, y
        >>> m = Matrix(2, 2, [x**2 + y, y**2 + x, 0, x + y])
        >>> m
        Matrix([
        [x**2 + y, x + y**2],
        [       0,    x + y]])
        >>> m.is_lower
        False

        See Also
        ========

        is_upper
        is_diagonal
        is_lower_hessenberg
        """
        return self._eval_is_lower()

    @property
    def is_square(self):
        """Checks if a matrix is square.

        A matrix is square if the number of rows equals the number of columns.
        The empty matrix is square by definition, since the number of rows and
        the number of columns are both zero.

        Examples
        ========

        >>> from sympy import Matrix
        >>> a = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> b = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> c = Matrix([])
        >>> a.is_square
        False
        >>> b.is_square
        True
        >>> c.is_square
        True
        """
        return self.rows == self.cols

    def is_symbolic(self):
        """Checks if any elements contain Symbols.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x, y
        >>> M = Matrix([[x, y], [1, 0]])
        >>> M.is_symbolic()
        True

        """
        return self._eval_is_symbolic()

    def is_symmetric(self, simplify=True):
        """Check if matrix is symmetric matrix,
        that is square matrix and is equal to its transpose.

        By default, simplifications occur before testing symmetry.
        They can be skipped using 'simplify=False'; while speeding things a bit,
        this may however induce false negatives.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(2, 2, [0, 1, 1, 2])
        >>> m
        Matrix([
        [0, 1],
        [1, 2]])
        >>> m.is_symmetric()
        True

        >>> m = Matrix(2, 2, [0, 1, 2, 0])
        >>> m
        Matrix([
        [0, 1],
        [2, 0]])
        >>> m.is_symmetric()
        False

        >>> m = Matrix(2, 3, [0, 0, 0, 0, 0, 0])
        >>> m
        Matrix([
        [0, 0, 0],
        [0, 0, 0]])
        >>> m.is_symmetric()
        False

        >>> from sympy.abc import x, y
        >>> m = Matrix(3, 3, [1, x**2 + 2*x + 1, y, (x + 1)**2, 2, 0, y, 0, 3])
        >>> m
        Matrix([
        [         1, x**2 + 2*x + 1, y],
        [(x + 1)**2,              2, 0],
        [         y,              0, 3]])
        >>> m.is_symmetric()
        True

        If the matrix is already simplified, you may speed-up is_symmetric()
        test by using 'simplify=False'.

        >>> bool(m.is_symmetric(simplify=False))
        False
        >>> m1 = m.expand()
        >>> m1.is_symmetric(simplify=False)
        True
        """
        simpfunc = simplify
        if not isfunction(simplify):
            simpfunc = _simplify if simplify else lambda x: x

        if not self.is_square:
            return False

        return self._eval_is_symmetric(simpfunc)

    @property
    def is_upper_hessenberg(self):
        """Checks if the matrix is the upper-Hessenberg form.

        The upper hessenberg matrix has zero entries
        below the first subdiagonal.

        Examples
        ========

        >>> from sympy import Matrix
        >>> a = Matrix([[1, 4, 2, 3], [3, 4, 1, 7], [0, 2, 3, 4], [0, 0, 1, 3]])
        >>> a
        Matrix([
        [1, 4, 2, 3],
        [3, 4, 1, 7],
        [0, 2, 3, 4],
        [0, 0, 1, 3]])
        >>> a.is_upper_hessenberg
        True

        See Also
        ========

        is_lower_hessenberg
        is_upper
        """
        return self._eval_is_upper_hessenberg()

    @property
    def is_upper(self):
        """Check if matrix is an upper triangular matrix. True can be returned
        even if the matrix is not square.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(2, 2, [1, 0, 0, 1])
        >>> m
        Matrix([
        [1, 0],
        [0, 1]])
        >>> m.is_upper
        True

        >>> m = Matrix(4, 3, [5, 1, 9, 0, 4, 6, 0, 0, 5, 0, 0, 0])
        >>> m
        Matrix([
        [5, 1, 9],
        [0, 4, 6],
        [0, 0, 5],
        [0, 0, 0]])
        >>> m.is_upper
        True

        >>> m = Matrix(2, 3, [4, 2, 5, 6, 1, 1])
        >>> m
        Matrix([
        [4, 2, 5],
        [6, 1, 1]])
        >>> m.is_upper
        False

        See Also
        ========

        is_lower
        is_diagonal
        is_upper_hessenberg
        """
        return all(self[i, j].is_zero
                   for i in range(1, self.rows)
                   for j in range(min(i, self.cols)))

    @property
    def is_zero_matrix(self):
        """Checks if a matrix is a zero matrix.

        A matrix is zero if every element is zero.  A matrix need not be square
        to be considered zero.  The empty matrix is zero by the principle of
        vacuous truth.  For a matrix that may or may not be zero (e.g.
        contains a symbol), this will be None

        Examples
        ========

        >>> from sympy import Matrix, zeros
        >>> from sympy.abc import x
        >>> a = Matrix([[0, 0], [0, 0]])
        >>> b = zeros(3, 4)
        >>> c = Matrix([[0, 1], [0, 0]])
        >>> d = Matrix([])
        >>> e = Matrix([[x, 0], [0, 0]])
        >>> a.is_zero_matrix
        True
        >>> b.is_zero_matrix
        True
        >>> c.is_zero_matrix
        False
        >>> d.is_zero_matrix
        True
        >>> e.is_zero_matrix
        """
        return self._eval_is_zero_matrix()

    def values(self):
        """Return non-zero values of self."""
        return self._eval_values()


class MatrixOperations(MatrixRequired):
    """Provides basic matrix shape and elementwise
    operations.  Should not be instantiated directly."""

    def _eval_adjoint(self):
        return self.transpose().conjugate()

    def _eval_applyfunc(self, f):
        out = self._new(self.rows, self.cols, [f(x) for x in self])
        return out

    def _eval_as_real_imag(self):  # type: ignore
        return (self.applyfunc(re), self.applyfunc(im))

    def _eval_conjugate(self):
        return self.applyfunc(lambda x: x.conjugate())

    def _eval_permute_cols(self, perm):
        # apply the permutation to a list
        mapping = list(perm)

        def entry(i, j):
            return self[i, mapping[j]]

        return self._new(self.rows, self.cols, entry)

    def _eval_permute_rows(self, perm):
        # apply the permutation to a list
        mapping = list(perm)

        def entry(i, j):
            return self[mapping[i], j]

        return self._new(self.rows, self.cols, entry)

    def _eval_trace(self):
        return sum(self[i, i] for i in range(self.rows))

    def _eval_transpose(self):
        return self._new(self.cols, self.rows, lambda i, j: self[j, i])

    def adjoint(self):
        """Conjugate transpose or Hermitian conjugation."""
        return self._eval_adjoint()

    def applyfunc(self, f):
        """Apply a function to each element of the matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(2, 2, lambda i, j: i*2+j)
        >>> m
        Matrix([
        [0, 1],
        [2, 3]])
        >>> m.applyfunc(lambda i: 2*i)
        Matrix([
        [0, 2],
        [4, 6]])

        """
        if not callable(f):
            raise TypeError("`f` must be callable.")

        return self._eval_applyfunc(f)

    def as_real_imag(self, deep=True, **hints):
        """Returns a tuple containing the (real, imaginary) part of matrix."""
        # XXX: Ignoring deep and hints...
        return self._eval_as_real_imag()

    def conjugate(self):
        """Return the by-element conjugation.

        Examples
        ========

        >>> from sympy import SparseMatrix, I
        >>> a = SparseMatrix(((1, 2 + I), (3, 4), (I, -I)))
        >>> a
        Matrix([
        [1, 2 + I],
        [3,     4],
        [I,    -I]])
        >>> a.C
        Matrix([
        [ 1, 2 - I],
        [ 3,     4],
        [-I,     I]])

        See Also
        ========

        transpose: Matrix transposition
        H: Hermite conjugation
        sympy.matrices.matrixbase.MatrixBase.D: Dirac conjugation
        """
        return self._eval_conjugate()

    def doit(self, **hints):
        return self.applyfunc(lambda x: x.doit(**hints))

    def evalf(self, n=15, subs=None, maxn=100, chop=False, strict=False, quad=None, verbose=False):
        """Apply evalf() to each element of self."""
        options = {'subs':subs, 'maxn':maxn, 'chop':chop, 'strict':strict,
                'quad':quad, 'verbose':verbose}
        return self.applyfunc(lambda i: i.evalf(n, **options))

    def expand(self, deep=True, modulus=None, power_base=True, power_exp=True,
               mul=True, log=True, multinomial=True, basic=True, **hints):
        """Apply core.function.expand to each entry of the matrix.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import Matrix
        >>> Matrix(1, 1, [x*(x+1)])
        Matrix([[x*(x + 1)]])
        >>> _.expand()
        Matrix([[x**2 + x]])

        """
        return self.applyfunc(lambda x: x.expand(
            deep, modulus, power_base, power_exp, mul, log, multinomial, basic,
            **hints))

    @property
    def H(self):
        """Return Hermite conjugate.

        Examples
        ========

        >>> from sympy import Matrix, I
        >>> m = Matrix((0, 1 + I, 2, 3))
        >>> m
        Matrix([
        [    0],
        [1 + I],
        [    2],
        [    3]])
        >>> m.H
        Matrix([[0, 1 - I, 2, 3]])

        See Also
        ========

        conjugate: By-element conjugation
        sympy.matrices.matrixbase.MatrixBase.D: Dirac conjugation
        """
        return self.T.C

    def permute(self, perm, orientation='rows', direction='forward'):
        r"""Permute the rows or columns of a matrix by the given list of
        swaps.

        Parameters
        ==========

        perm : Permutation, list, or list of lists
            A representation for the permutation.

            If it is ``Permutation``, it is used directly with some
            resizing with respect to the matrix size.

            If it is specified as list of lists,
            (e.g., ``[[0, 1], [0, 2]]``), then the permutation is formed
            from applying the product of cycles. The direction how the
            cyclic product is applied is described in below.

            If it is specified as a list, the list should represent
            an array form of a permutation. (e.g., ``[1, 2, 0]``) which
            would would form the swapping function
            `0 \mapsto 1, 1 \mapsto 2, 2\mapsto 0`.

        orientation : 'rows', 'cols'
            A flag to control whether to permute the rows or the columns

        direction : 'forward', 'backward'
            A flag to control whether to apply the permutations from
            the start of the list first, or from the back of the list
            first.

            For example, if the permutation specification is
            ``[[0, 1], [0, 2]]``,

            If the flag is set to ``'forward'``, the cycle would be
            formed as `0 \mapsto 2, 2 \mapsto 1, 1 \mapsto 0`.

            If the flag is set to ``'backward'``, the cycle would be
            formed as `0 \mapsto 1, 1 \mapsto 2, 2 \mapsto 0`.

            If the argument ``perm`` is not in a form of list of lists,
            this flag takes no effect.

        Examples
        ========

        >>> from sympy import eye
        >>> M = eye(3)
        >>> M.permute([[0, 1], [0, 2]], orientation='rows', direction='forward')
        Matrix([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]])

        >>> from sympy import eye
        >>> M = eye(3)
        >>> M.permute([[0, 1], [0, 2]], orientation='rows', direction='backward')
        Matrix([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]])

        Notes
        =====

        If a bijective function
        `\sigma : \mathbb{N}_0 \rightarrow \mathbb{N}_0` denotes the
        permutation.

        If the matrix `A` is the matrix to permute, represented as
        a horizontal or a vertical stack of vectors:

        .. math::
            A =
            \begin{bmatrix}
            a_0 \\ a_1 \\ \vdots \\ a_{n-1}
            \end{bmatrix} =
            \begin{bmatrix}
            \alpha_0 & \alpha_1 & \cdots & \alpha_{n-1}
            \end{bmatrix}

        If the matrix `B` is the result, the permutation of matrix rows
        is defined as:

        .. math::
            B := \begin{bmatrix}
            a_{\sigma(0)} \\ a_{\sigma(1)} \\ \vdots \\ a_{\sigma(n-1)}
            \end{bmatrix}

        And the permutation of matrix columns is defined as:

        .. math::
            B := \begin{bmatrix}
            \alpha_{\sigma(0)} & \alpha_{\sigma(1)} &
            \cdots & \alpha_{\sigma(n-1)}
            \end{bmatrix}
        """
        from sympy.combinatorics import Permutation

        # allow british variants and `columns`
        if direction == 'forwards':
            direction = 'forward'
        if direction == 'backwards':
            direction = 'backward'
        if orientation == 'columns':
            orientation = 'cols'

        if direction not in ('forward', 'backward'):
            raise TypeError("direction='{}' is an invalid kwarg. "
                            "Try 'forward' or 'backward'".format(direction))
        if orientation not in ('rows', 'cols'):
            raise TypeError("orientation='{}' is an invalid kwarg. "
                            "Try 'rows' or 'cols'".format(orientation))

        if not isinstance(perm, (Permutation, Iterable)):
            raise ValueError(
                "{} must be a list, a list of lists, "
                "or a SymPy permutation object.".format(perm))

        # ensure all swaps are in range
        max_index = self.rows if orientation == 'rows' else self.cols
        if not all(0 <= t <= max_index for t in flatten(list(perm))):
            raise IndexError("`swap` indices out of range.")

        if perm and not isinstance(perm, Permutation) and \
            isinstance(perm[0], Iterable):
            if direction == 'forward':
                perm = list(reversed(perm))
            perm = Permutation(perm, size=max_index+1)
        else:
            perm = Permutation(perm, size=max_index+1)

        if orientation == 'rows':
            return self._eval_permute_rows(perm)
        if orientation == 'cols':
            return self._eval_permute_cols(perm)

    def permute_cols(self, swaps, direction='forward'):
        """Alias for
        ``self.permute(swaps, orientation='cols', direction=direction)``

        See Also
        ========

        permute
        """
        return self.permute(swaps, orientation='cols', direction=direction)

    def permute_rows(self, swaps, direction='forward'):
        """Alias for
        ``self.permute(swaps, orientation='rows', direction=direction)``

        See Also
        ========

        permute
        """
        return self.permute(swaps, orientation='rows', direction=direction)

    def refine(self, assumptions=True):
        """Apply refine to each element of the matrix.

        Examples
        ========

        >>> from sympy import Symbol, Matrix, Abs, sqrt, Q
        >>> x = Symbol('x')
        >>> Matrix([[Abs(x)**2, sqrt(x**2)],[sqrt(x**2), Abs(x)**2]])
        Matrix([
        [ Abs(x)**2, sqrt(x**2)],
        [sqrt(x**2),  Abs(x)**2]])
        >>> _.refine(Q.real(x))
        Matrix([
        [  x**2, Abs(x)],
        [Abs(x),   x**2]])

        """
        return self.applyfunc(lambda x: refine(x, assumptions))

    def replace(self, F, G, map=False, simultaneous=True, exact=None):
        """Replaces Function F in Matrix entries with Function G.

        Examples
        ========

        >>> from sympy import symbols, Function, Matrix
        >>> F, G = symbols('F, G', cls=Function)
        >>> M = Matrix(2, 2, lambda i, j: F(i+j)) ; M
        Matrix([
        [F(0), F(1)],
        [F(1), F(2)]])
        >>> N = M.replace(F,G)
        >>> N
        Matrix([
        [G(0), G(1)],
        [G(1), G(2)]])
        """
        return self.applyfunc(
            lambda x: x.replace(F, G, map=map, simultaneous=simultaneous, exact=exact))

    def rot90(self, k=1):
        """Rotates Matrix by 90 degrees

        Parameters
        ==========

        k : int
            Specifies how many times the matrix is rotated by 90 degrees
            (clockwise when positive, counter-clockwise when negative).

        Examples
        ========

        >>> from sympy import Matrix, symbols
        >>> A = Matrix(2, 2, symbols('a:d'))
        >>> A
        Matrix([
        [a, b],
        [c, d]])

        Rotating the matrix clockwise one time:

        >>> A.rot90(1)
        Matrix([
        [c, a],
        [d, b]])

        Rotating the matrix anticlockwise two times:

        >>> A.rot90(-2)
        Matrix([
        [d, c],
        [b, a]])
        """

        mod = k%4
        if mod == 0:
            return self
        if mod == 1:
            return self[::-1, ::].T
        if mod == 2:
            return self[::-1, ::-1]
        if mod == 3:
            return self[::, ::-1].T

    def simplify(self, **kwargs):
        """Apply simplify to each element of the matrix.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import SparseMatrix, sin, cos
        >>> SparseMatrix(1, 1, [x*sin(y)**2 + x*cos(y)**2])
        Matrix([[x*sin(y)**2 + x*cos(y)**2]])
        >>> _.simplify()
        Matrix([[x]])
        """
        return self.applyfunc(lambda x: x.simplify(**kwargs))

    def subs(self, *args, **kwargs):  # should mirror core.basic.subs
        """Return a new matrix with subs applied to each entry.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import SparseMatrix, Matrix
        >>> SparseMatrix(1, 1, [x])
        Matrix([[x]])
        >>> _.subs(x, y)
        Matrix([[y]])
        >>> Matrix(_).subs(y, x)
        Matrix([[x]])
        """

        if len(args) == 1 and  not isinstance(args[0], (dict, set)) and iter(args[0]) and not is_sequence(args[0]):
            args = (list(args[0]),)

        return self.applyfunc(lambda x: x.subs(*args, **kwargs))

    def trace(self):
        """
        Returns the trace of a square matrix i.e. the sum of the
        diagonal elements.

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix(2, 2, [1, 2, 3, 4])
        >>> A.trace()
        5

        """
        if self.rows != self.cols:
            raise NonSquareMatrixError()
        return self._eval_trace()

    def transpose(self):
        """
        Returns the transpose of the matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix(2, 2, [1, 2, 3, 4])
        >>> A.transpose()
        Matrix([
        [1, 3],
        [2, 4]])

        >>> from sympy import Matrix, I
        >>> m=Matrix(((1, 2+I), (3, 4)))
        >>> m
        Matrix([
        [1, 2 + I],
        [3,     4]])
        >>> m.transpose()
        Matrix([
        [    1, 3],
        [2 + I, 4]])
        >>> m.T == m.transpose()
        True

        See Also
        ========

        conjugate: By-element conjugation

        """
        return self._eval_transpose()

    @property
    def T(self):
        '''Matrix transposition'''
        return self.transpose()

    @property
    def C(self):
        '''By-element conjugation'''
        return self.conjugate()

    def n(self, *args, **kwargs):
        """Apply evalf() to each element of self."""
        return self.evalf(*args, **kwargs)

    def xreplace(self, rule):  # should mirror core.basic.xreplace
        """Return a new matrix with xreplace applied to each entry.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import SparseMatrix, Matrix
        >>> SparseMatrix(1, 1, [x])
        Matrix([[x]])
        >>> _.xreplace({x: y})
        Matrix([[y]])
        >>> Matrix(_).xreplace({y: x})
        Matrix([[x]])
        """
        return self.applyfunc(lambda x: x.xreplace(rule))

    def _eval_simplify(self, **kwargs):
        # XXX: We can't use self.simplify here as mutable subclasses will
        # override simplify and have it return None
        return MatrixOperations.simplify(self, **kwargs)

    def _eval_trigsimp(self, **opts):
        from sympy.simplify.trigsimp import trigsimp
        return self.applyfunc(lambda x: trigsimp(x, **opts))

    def upper_triangular(self, k=0):
        """Return the elements on and above the kth diagonal of a matrix.
        If k is not specified then simply returns upper-triangular portion
        of a matrix

        Examples
        ========

        >>> from sympy import ones
        >>> A = ones(4)
        >>> A.upper_triangular()
        Matrix([
        [1, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1]])

        >>> A.upper_triangular(2)
        Matrix([
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]])

        >>> A.upper_triangular(-1)
        Matrix([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 1, 1]])

        """

        def entry(i, j):
            return self[i, j] if i + k <= j else self.zero

        return self._new(self.rows, self.cols, entry)


    def lower_triangular(self, k=0):
        """Return the elements on and below the kth diagonal of a matrix.
        If k is not specified then simply returns lower-triangular portion
        of a matrix

        Examples
        ========

        >>> from sympy import ones
        >>> A = ones(4)
        >>> A.lower_triangular()
        Matrix([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]])

        >>> A.lower_triangular(-2)
        Matrix([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0]])

        >>> A.lower_triangular(1)
        Matrix([
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1]])

        """

        def entry(i, j):
            return self[i, j] if i + k >= j else self.zero

        return self._new(self.rows, self.cols, entry)



class MatrixArithmetic(MatrixRequired):
    """Provides basic matrix arithmetic operations.
    Should not be instantiated directly."""

    _op_priority = 10.01

    def _eval_Abs(self):
        return self._new(self.rows, self.cols, lambda i, j: Abs(self[i, j]))

    def _eval_add(self, other):
        return self._new(self.rows, self.cols,
                         lambda i, j: self[i, j] + other[i, j])

    def _eval_matrix_mul(self, other):
        def entry(i, j):
            vec = [self[i,k]*other[k,j] for k in range(self.cols)]
            try:
                return Add(*vec)
            except (TypeError, SympifyError):
                # Some matrices don't work with `sum` or `Add`
                # They don't work with `sum` because `sum` tries to add `0`
                # Fall back to a safe way to multiply if the `Add` fails.
                return reduce(lambda a, b: a + b, vec)

        return self._new(self.rows, other.cols, entry)

    def _eval_matrix_mul_elementwise(self, other):
        return self._new(self.rows, self.cols, lambda i, j: self[i,j]*other[i,j])

    def _eval_matrix_rmul(self, other):
        def entry(i, j):
            return sum(other[i,k]*self[k,j] for k in range(other.cols))
        return self._new(other.rows, self.cols, entry)

    def _eval_pow_by_recursion(self, num):
        if num == 1:
            return self

        if num % 2 == 1:
            a, b = self, self._eval_pow_by_recursion(num - 1)
        else:
            a = b = self._eval_pow_by_recursion(num // 2)

        return a.multiply(b)

    def _eval_pow_by_cayley(self, exp):
        from sympy.discrete.recurrences import linrec_coeffs
        row = self.shape[0]
        p = self.charpoly()

        coeffs = (-p).all_coeffs()[1:]
        coeffs = linrec_coeffs(coeffs, exp)
        new_mat = self.eye(row)
        ans = self.zeros(row)

        for i in range(row):
            ans += coeffs[i]*new_mat
            new_mat *= self

        return ans

    def _eval_pow_by_recursion_dotprodsimp(self, num, prevsimp=None):
        if prevsimp is None:
            prevsimp = [True]*len(self)

        if num == 1:
            return self

        if num % 2 == 1:
            a, b = self, self._eval_pow_by_recursion_dotprodsimp(num - 1,
                    prevsimp=prevsimp)
        else:
            a = b = self._eval_pow_by_recursion_dotprodsimp(num // 2,
                    prevsimp=prevsimp)

        m     = a.multiply(b, dotprodsimp=False)
        lenm  = len(m)
        elems = [None]*lenm

        for i in range(lenm):
            if prevsimp[i]:
                elems[i], prevsimp[i] = _dotprodsimp(m[i], withsimp=True)
            else:
                elems[i] = m[i]

        return m._new(m.rows, m.cols, elems)

    def _eval_scalar_mul(self, other):
        return self._new(self.rows, self.cols, lambda i, j: self[i,j]*other)

    def _eval_scalar_rmul(self, other):
        return self._new(self.rows, self.cols, lambda i, j: other*self[i,j])

    def _eval_Mod(self, other):
        return self._new(self.rows, self.cols, lambda i, j: Mod(self[i, j], other))

    # Python arithmetic functions
    def __abs__(self):
        """Returns a new matrix with entry-wise absolute values."""
        return self._eval_Abs()

    @call_highest_priority('__radd__')
    def __add__(self, other):
        """Return self + other, raising ShapeError if shapes do not match."""
        if isinstance(other, NDimArray): # Matrix and array addition is currently not implemented
            return NotImplemented
        other = _matrixify(other)
        # matrix-like objects can have shapes.  This is
        # our first sanity check.
        if hasattr(other, 'shape'):
            if self.shape != other.shape:
                raise ShapeError("Matrix size mismatch: %s + %s" % (
                    self.shape, other.shape))

        # honest SymPy matrices defer to their class's routine
        if getattr(other, 'is_Matrix', False):
            # call the highest-priority class's _eval_add
            a, b = self, other
            if a.__class__ != classof(a, b):
                b, a = a, b
            return a._eval_add(b)
        # Matrix-like objects can be passed to CommonMatrix routines directly.
        if getattr(other, 'is_MatrixLike', False):
            return MatrixArithmetic._eval_add(self, other)

        raise TypeError('cannot add %s and %s' % (type(self), type(other)))

    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        return self * (self.one / other)

    @call_highest_priority('__rmatmul__')
    def __matmul__(self, other):
        other = _matrixify(other)
        if not getattr(other, 'is_Matrix', False) and not getattr(other, 'is_MatrixLike', False):
            return NotImplemented

        return self.__mul__(other)

    def __mod__(self, other):
        return self.applyfunc(lambda x: x % other)

    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        """Return self*other where other is either a scalar or a matrix
        of compatible dimensions.

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> 2*A == A*2 == Matrix([[2, 4, 6], [8, 10, 12]])
        True
        >>> B = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> A*B
        Matrix([
        [30, 36, 42],
        [66, 81, 96]])
        >>> B*A
        Traceback (most recent call last):
        ...
        ShapeError: Matrices size mismatch.
        >>>

        See Also
        ========

        matrix_multiply_elementwise
        """

        return self.multiply(other)

    def multiply(self, other, dotprodsimp=None):
        """Same as __mul__() but with optional simplification.

        Parameters
        ==========

        dotprodsimp : bool, optional
            Specifies whether intermediate term algebraic simplification is used
            during matrix multiplications to control expression blowup and thus
            speed up calculation. Default is off.
        """

        isimpbool = _get_intermediate_simp_bool(False, dotprodsimp)
        other = _matrixify(other)
        # matrix-like objects can have shapes.  This is
        # our first sanity check. Double check other is not explicitly not a Matrix.
        if (hasattr(other, 'shape') and len(other.shape) == 2 and
            (getattr(other, 'is_Matrix', True) or
             getattr(other, 'is_MatrixLike', True))):
            if self.shape[1] != other.shape[0]:
                raise ShapeError("Matrix size mismatch: %s * %s." % (
                    self.shape, other.shape))

        # honest SymPy matrices defer to their class's routine
        if getattr(other, 'is_Matrix', False):
            m = self._eval_matrix_mul(other)
            if isimpbool:
                return m._new(m.rows, m.cols, [_dotprodsimp(e) for e in m])
            return m

        # Matrix-like objects can be passed to CommonMatrix routines directly.
        if getattr(other, 'is_MatrixLike', False):
            return MatrixArithmetic._eval_matrix_mul(self, other)

        # if 'other' is not iterable then scalar multiplication.
        if not isinstance(other, Iterable):
            try:
                return self._eval_scalar_mul(other)
            except TypeError:
                pass

        return NotImplemented

    def multiply_elementwise(self, other):
        """Return the Hadamard product (elementwise product) of A and B

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix([[0, 1, 2], [3, 4, 5]])
        >>> B = Matrix([[1, 10, 100], [100, 10, 1]])
        >>> A.multiply_elementwise(B)
        Matrix([
        [  0, 10, 200],
        [300, 40,   5]])

        See Also
        ========

        sympy.matrices.matrixbase.MatrixBase.cross
        sympy.matrices.matrixbase.MatrixBase.dot
        multiply
        """
        if self.shape != other.shape:
            raise ShapeError("Matrix shapes must agree {} != {}".format(self.shape, other.shape))

        return self._eval_matrix_mul_elementwise(other)

    def __neg__(self):
        return self._eval_scalar_mul(-1)

    @call_highest_priority('__rpow__')
    def __pow__(self, exp):
        """Return self**exp a scalar or symbol."""

        return self.pow(exp)


    def pow(self, exp, method=None):
        r"""Return self**exp a scalar or symbol.

        Parameters
        ==========

        method : multiply, mulsimp, jordan, cayley
            If multiply then it returns exponentiation using recursion.
            If jordan then Jordan form exponentiation will be used.
            If cayley then the exponentiation is done using Cayley-Hamilton
            theorem.
            If mulsimp then the exponentiation is done using recursion
            with dotprodsimp. This specifies whether intermediate term
            algebraic simplification is used during naive matrix power to
            control expression blowup and thus speed up calculation.
            If None, then it heuristically decides which method to use.

        """

        if method is not None and method not in ['multiply', 'mulsimp', 'jordan', 'cayley']:
            raise TypeError('No such method')
        if self.rows != self.cols:
            raise NonSquareMatrixError()
        a = self
        jordan_pow = getattr(a, '_matrix_pow_by_jordan_blocks', None)
        exp = sympify(exp)

        if exp.is_zero:
            return a._new(a.rows, a.cols, lambda i, j: int(i == j))
        if exp == 1:
            return a

        diagonal = getattr(a, 'is_diagonal', None)
        if diagonal is not None and diagonal():
            return a._new(a.rows, a.cols, lambda i, j: a[i,j]**exp if i == j else 0)

        if exp.is_Number and exp % 1 == 0:
            if a.rows == 1:
                return a._new([[a[0]**exp]])
            if exp < 0:
                exp = -exp
                a = a.inv()
        # When certain conditions are met,
        # Jordan block algorithm is faster than
        # computation by recursion.
        if method == 'jordan':
            try:
                return jordan_pow(exp)
            except MatrixError:
                if method == 'jordan':
                    raise

        elif method == 'cayley':
            if not exp.is_Number or exp % 1 != 0:
                raise ValueError("cayley method is only valid for integer powers")
            return a._eval_pow_by_cayley(exp)

        elif method == "mulsimp":
            if not exp.is_Number or exp % 1 != 0:
                raise ValueError("mulsimp method is only valid for integer powers")
            return a._eval_pow_by_recursion_dotprodsimp(exp)

        elif method == "multiply":
            if not exp.is_Number or exp % 1 != 0:
                raise ValueError("multiply method is only valid for integer powers")
            return a._eval_pow_by_recursion(exp)

        elif method is None and exp.is_Number and exp % 1 == 0:
            if exp.is_Float:
                exp = Integer(exp)
            # Decide heuristically which method to apply
            if a.rows == 2 and exp > 100000:
                return jordan_pow(exp)
            elif _get_intermediate_simp_bool(True, None):
                return a._eval_pow_by_recursion_dotprodsimp(exp)
            elif exp > 10000:
                return a._eval_pow_by_cayley(exp)
            else:
                return a._eval_pow_by_recursion(exp)

        if jordan_pow:
            try:
                return jordan_pow(exp)
            except NonInvertibleMatrixError:
                # Raised by jordan_pow on zero determinant matrix unless exp is
                # definitely known to be a non-negative integer.
                # Here we raise if n is definitely not a non-negative integer
                # but otherwise we can leave this as an unevaluated MatPow.
                if exp.is_integer is False or exp.is_nonnegative is False:
                    raise

        from sympy.matrices.expressions import MatPow
        return MatPow(a, exp)

    @call_highest_priority('__add__')
    def __radd__(self, other):
        return self + other

    @call_highest_priority('__matmul__')
    def __rmatmul__(self, other):
        other = _matrixify(other)
        if not getattr(other, 'is_Matrix', False) and not getattr(other, 'is_MatrixLike', False):
            return NotImplemented

        return self.__rmul__(other)

    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return self.rmultiply(other)

    def rmultiply(self, other, dotprodsimp=None):
        """Same as __rmul__() but with optional simplification.

        Parameters
        ==========

        dotprodsimp : bool, optional
            Specifies whether intermediate term algebraic simplification is used
            during matrix multiplications to control expression blowup and thus
            speed up calculation. Default is off.
        """
        isimpbool = _get_intermediate_simp_bool(False, dotprodsimp)
        other = _matrixify(other)
        # matrix-like objects can have shapes.  This is
        # our first sanity check. Double check other is not explicitly not a Matrix.
        if (hasattr(other, 'shape') and len(other.shape) == 2 and
            (getattr(other, 'is_Matrix', True) or
             getattr(other, 'is_MatrixLike', True))):
            if self.shape[0] != other.shape[1]:
                raise ShapeError("Matrix size mismatch.")

        # honest SymPy matrices defer to their class's routine
        if getattr(other, 'is_Matrix', False):
            m = self._eval_matrix_rmul(other)
            if isimpbool:
                return m._new(m.rows, m.cols, [_dotprodsimp(e) for e in m])
            return m
        # Matrix-like objects can be passed to CommonMatrix routines directly.
        if getattr(other, 'is_MatrixLike', False):
            return MatrixArithmetic._eval_matrix_rmul(self, other)

        # if 'other' is not iterable then scalar multiplication.
        if not isinstance(other, Iterable):
            try:
                return self._eval_scalar_rmul(other)
            except TypeError:
                pass

        return NotImplemented

    @call_highest_priority('__sub__')
    def __rsub__(self, a):
        return (-self) + a

    @call_highest_priority('__rsub__')
    def __sub__(self, a):
        return self + (-a)


class MatrixCommon(MatrixArithmetic, MatrixOperations, MatrixProperties,
                  MatrixSpecial, MatrixShaping):
    """All common matrix operations including basic arithmetic, shaping,
    and special matrices like `zeros`, and `eye`."""
    _diff_wrt: bool = True


class _MinimalMatrix:
    """Class providing the minimum functionality
    for a matrix-like object and implementing every method
    required for a `MatrixRequired`.  This class does not have everything
    needed to become a full-fledged SymPy object, but it will satisfy the
    requirements of anything inheriting from `MatrixRequired`.  If you wish
    to make a specialized matrix type, make sure to implement these
    methods and properties with the exception of `__init__` and `__repr__`
    which are included for convenience."""

    is_MatrixLike = True
    _sympify = staticmethod(sympify)
    _class_priority = 3
    zero = S.Zero
    one = S.One

    is_Matrix = True
    is_MatrixExpr = False

    @classmethod
    def _new(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def __init__(self, rows, cols=None, mat=None, copy=False):
        if isfunction(mat):
            # if we passed in a function, use that to populate the indices
            mat = [mat(i, j) for i in range(rows) for j in range(cols)]
        if cols is None and mat is None:
            mat = rows
        rows, cols = getattr(mat, 'shape', (rows, cols))
        try:
            # if we passed in a list of lists, flatten it and set the size
            if cols is None and mat is None:
                mat = rows
            cols = len(mat[0])
            rows = len(mat)
            mat = [x for l in mat for x in l]
        except (IndexError, TypeError):
            pass
        self.mat = tuple(self._sympify(x) for x in mat)
        self.rows, self.cols = rows, cols
        if self.rows is None or self.cols is None:
            raise NotImplementedError("Cannot initialize matrix with given parameters")

    def __getitem__(self, key):
        def _normalize_slices(row_slice, col_slice):
            """Ensure that row_slice and col_slice do not have
            `None` in their arguments.  Any integers are converted
            to slices of length 1"""
            if not isinstance(row_slice, slice):
                row_slice = slice(row_slice, row_slice + 1, None)
            row_slice = slice(*row_slice.indices(self.rows))

            if not isinstance(col_slice, slice):
                col_slice = slice(col_slice, col_slice + 1, None)
            col_slice = slice(*col_slice.indices(self.cols))

            return (row_slice, col_slice)

        def _coord_to_index(i, j):
            """Return the index in _mat corresponding
            to the (i,j) position in the matrix. """
            return i * self.cols + j

        if isinstance(key, tuple):
            i, j = key
            if isinstance(i, slice) or isinstance(j, slice):
                # if the coordinates are not slices, make them so
                # and expand the slices so they don't contain `None`
                i, j = _normalize_slices(i, j)

                rowsList, colsList = list(range(self.rows))[i], \
                                     list(range(self.cols))[j]
                indices = (i * self.cols + j for i in rowsList for j in
                           colsList)
                return self._new(len(rowsList), len(colsList),
                                 [self.mat[i] for i in indices])

            # if the key is a tuple of ints, change
            # it to an array index
            key = _coord_to_index(i, j)
        return self.mat[key]

    def __eq__(self, other):
        try:
            classof(self, other)
        except TypeError:
            return False
        return (
            self.shape == other.shape and list(self) == list(other))

    def __len__(self):
        return self.rows*self.cols

    def __repr__(self):
        return "_MinimalMatrix({}, {}, {})".format(self.rows, self.cols,
                                                   self.mat)

    @property
    def shape(self):
        return (self.rows, self.cols)


class _CastableMatrix: # this is needed here ONLY FOR TESTS.
    def as_mutable(self):
        return self

    def as_immutable(self):
        return self


class _MatrixWrapper:
    """Wrapper class providing the minimum functionality for a matrix-like
    object: .rows, .cols, .shape, indexability, and iterability. CommonMatrix
    math operations should work on matrix-like objects. This one is intended for
    matrix-like objects which use the same indexing format as SymPy with respect
    to returning matrix elements instead of rows for non-tuple indexes.
    """

    is_Matrix     = False # needs to be here because of __getattr__
    is_MatrixLike = True

    def __init__(self, mat, shape):
        self.mat = mat
        self.shape = shape
        self.rows, self.cols = shape

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return sympify(self.mat.__getitem__(key))

        return sympify(self.mat.__getitem__((key // self.rows, key % self.cols)))

    def __iter__(self): # supports numpy.matrix and numpy.array
        mat = self.mat
        cols = self.cols

        return iter(sympify(mat[r, c]) for r in range(self.rows) for c in range(cols))


def _matrixify(mat):
    """If `mat` is a Matrix or is matrix-like,
    return a Matrix or MatrixWrapper object.  Otherwise
    `mat` is passed through without modification."""

    if getattr(mat, 'is_Matrix', False) or getattr(mat, 'is_MatrixLike', False):
        return mat

    if not(getattr(mat, 'is_Matrix', True) or getattr(mat, 'is_MatrixLike', True)):
        return mat

    shape = None

    if hasattr(mat, 'shape'): # numpy, scipy.sparse
        if len(mat.shape) == 2:
            shape = mat.shape
    elif hasattr(mat, 'rows') and hasattr(mat, 'cols'): # mpmath
        shape = (mat.rows, mat.cols)

    if shape:
        return _MatrixWrapper(mat, shape)

    return mat


def a2idx(j, n=None):
    """Return integer after making positive and validating against n."""
    if not isinstance(j, int):
        jindex = getattr(j, '__index__', None)
        if jindex is not None:
            j = jindex()
        else:
            raise IndexError("Invalid index a[%r]" % (j,))
    if n is not None:
        if j < 0:
            j += n
        if not (j >= 0 and j < n):
            raise IndexError("Index out of range: a[%s]" % (j,))
    return int(j)


def classof(A, B):
    """
    Get the type of the result when combining matrices of different types.

    Currently the strategy is that immutability is contagious.

    Examples
    ========

    >>> from sympy import Matrix, ImmutableMatrix
    >>> from sympy.matrices.matrixbase import classof
    >>> M = Matrix([[1, 2], [3, 4]]) # a Mutable Matrix
    >>> IM = ImmutableMatrix([[1, 2], [3, 4]])
    >>> classof(M, IM)
    <class 'sympy.matrices.immutable.ImmutableDenseMatrix'>
    """
    priority_A = getattr(A, '_class_priority', None)
    priority_B = getattr(B, '_class_priority', None)
    if None not in (priority_A, priority_B):
        if A._class_priority > B._class_priority:
            return A.__class__
        else:
            return B.__class__

    try:
        import numpy
    except ImportError:
        pass
    else:
        if isinstance(A, numpy.ndarray):
            return B.__class__
        if isinstance(B, numpy.ndarray):
            return A.__class__

    raise TypeError("Incompatible classes %s, %s" % (A.__class__, B.__class__))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\combinatorial\numbers.py ===
"""
This module implements some special functions that commonly appear in
combinatorial contexts (e.g. in power series); in particular,
sequences of rational numbers such as Bernoulli and Fibonacci numbers.

Factorials, binomial coefficients and related functions are located in
the separate 'factorials' module.
"""
from __future__ import annotations
from math import prod
from collections import defaultdict
from typing import Callable

from sympy.core import S, Symbol, Add, Dummy
from sympy.core.cache import cacheit
from sympy.core.containers import Dict
from sympy.core.expr import Expr
from sympy.core.function import ArgumentIndexError, DefinedFunction, expand_mul
from sympy.core.logic import fuzzy_not
from sympy.core.mul import Mul
from sympy.core.numbers import E, I, pi, oo, Rational, Integer
from sympy.core.relational import Eq, is_le, is_gt, is_lt
from sympy.external.gmpy import SYMPY_INTS, remove, lcm, legendre, jacobi, kronecker
from sympy.functions.combinatorial.factorials import (binomial,
    factorial, subfactorial)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.ntheory.factor_ import (factorint, _divisor_sigma, is_carmichael,
                                   find_carmichael_numbers_in_range, find_first_n_carmichaels)
from sympy.ntheory.generate import _primepi
from sympy.ntheory.partitions_ import _partition, _partition_rec
from sympy.ntheory.primetest import isprime, is_square
from sympy.polys.appellseqs import bernoulli_poly, euler_poly, genocchi_poly
from sympy.polys.polytools import cancel
from sympy.utilities.enumerative import MultisetPartitionTraverser
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import multiset, multiset_derangements, iterable
from sympy.utilities.memoization import recurrence_memo
from sympy.utilities.misc import as_int

from mpmath import mp, workprec
from mpmath.libmp import ifib as _ifib


def _product(a, b):
    return prod(range(a, b + 1))


# Dummy symbol used for computing polynomial sequences
_sym = Symbol('x')


#----------------------------------------------------------------------------#
#                                                                            #
#                           Carmichael numbers                               #
#                                                                            #
#----------------------------------------------------------------------------#

class carmichael(DefinedFunction):
    r"""
    Carmichael Numbers:

    Certain cryptographic algorithms make use of big prime numbers.
    However, checking whether a big number is prime is not so easy.
    Randomized prime number checking tests exist that offer a high degree of
    confidence of accurate determination at low cost, such as the Fermat test.

    Let 'a' be a random number between $2$ and $n - 1$, where $n$ is the
    number whose primality we are testing. Then, $n$ is probably prime if it
    satisfies the modular arithmetic congruence relation:

    .. math :: a^{n-1} = 1 \pmod{n}

    (where mod refers to the modulo operation)

    If a number passes the Fermat test several times, then it is prime with a
    high probability.

    Unfortunately, certain composite numbers (non-primes) still pass the Fermat
    test with every number smaller than themselves.
    These numbers are called Carmichael numbers.

    A Carmichael number will pass a Fermat primality test to every base $b$
    relatively prime to the number, even though it is not actually prime.
    This makes tests based on Fermat's Little Theorem less effective than
    strong probable prime tests such as the Baillie-PSW primality test and
    the Miller-Rabin primality test.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import find_first_n_carmichaels, find_carmichael_numbers_in_range
    >>> find_first_n_carmichaels(5)
    [561, 1105, 1729, 2465, 2821]
    >>> find_carmichael_numbers_in_range(0, 562)
    [561]
    >>> find_carmichael_numbers_in_range(0,1000)
    [561]
    >>> find_carmichael_numbers_in_range(0,2000)
    [561, 1105, 1729]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Carmichael_number
    .. [2] https://en.wikipedia.org/wiki/Fermat_primality_test
    .. [3] https://www.jstor.org/stable/23248683?seq=1#metadata_info_tab_contents
    """

    @staticmethod
    def is_perfect_square(n):
        sympy_deprecation_warning(
        """
is_perfect_square is just a wrapper around sympy.ntheory.primetest.is_square
so use that directly instead.
        """,
        deprecated_since_version="1.11",
        active_deprecations_target='deprecated-carmichael-static-methods',
        )
        return is_square(n)

    @staticmethod
    def divides(p, n):
        sympy_deprecation_warning(
        """
        divides can be replaced by directly testing n % p == 0.
        """,
        deprecated_since_version="1.11",
        active_deprecations_target='deprecated-carmichael-static-methods',
        )
        return n % p == 0

    @staticmethod
    def is_prime(n):
        sympy_deprecation_warning(
        """
is_prime is just a wrapper around sympy.ntheory.primetest.isprime so use that
directly instead.
        """,
        deprecated_since_version="1.11",
        active_deprecations_target='deprecated-carmichael-static-methods',
        )
        return isprime(n)

    @staticmethod
    def is_carmichael(n):
        sympy_deprecation_warning(
        """
is_carmichael is just a wrapper around sympy.ntheory.factor_.is_carmichael so use that
directly instead.
        """,
        deprecated_since_version="1.13",
        active_deprecations_target='deprecated-ntheory-symbolic-functions',
        )
        return is_carmichael(n)

    @staticmethod
    def find_carmichael_numbers_in_range(x, y):
        sympy_deprecation_warning(
        """
find_carmichael_numbers_in_range is just a wrapper around sympy.ntheory.factor_.find_carmichael_numbers_in_range so use that
directly instead.
        """,
        deprecated_since_version="1.13",
        active_deprecations_target='deprecated-ntheory-symbolic-functions',
        )
        return find_carmichael_numbers_in_range(x, y)

    @staticmethod
    def find_first_n_carmichaels(n):
        sympy_deprecation_warning(
        """
find_first_n_carmichaels is just a wrapper around sympy.ntheory.factor_.find_first_n_carmichaels so use that
directly instead.
        """,
        deprecated_since_version="1.13",
        active_deprecations_target='deprecated-ntheory-symbolic-functions',
        )
        return find_first_n_carmichaels(n)


#----------------------------------------------------------------------------#
#                                                                            #
#                           Fibonacci numbers                                #
#                                                                            #
#----------------------------------------------------------------------------#


class fibonacci(DefinedFunction):
    r"""
    Fibonacci numbers / Fibonacci polynomials

    The Fibonacci numbers are the integer sequence defined by the
    initial terms `F_0 = 0`, `F_1 = 1` and the two-term recurrence
    relation `F_n = F_{n-1} + F_{n-2}`.  This definition
    extended to arbitrary real and complex arguments using
    the formula

    .. math :: F_z = \frac{\phi^z - \cos(\pi z) \phi^{-z}}{\sqrt 5}

    The Fibonacci polynomials are defined by `F_1(x) = 1`,
    `F_2(x) = x`, and `F_n(x) = x*F_{n-1}(x) + F_{n-2}(x)` for `n > 2`.
    For all positive integers `n`, `F_n(1) = F_n`.

    * ``fibonacci(n)`` gives the `n^{th}` Fibonacci number, `F_n`
    * ``fibonacci(n, x)`` gives the `n^{th}` Fibonacci polynomial in `x`, `F_n(x)`

    Examples
    ========

    >>> from sympy import fibonacci, Symbol

    >>> [fibonacci(x) for x in range(11)]
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    >>> fibonacci(5, Symbol('t'))
    t**4 + 3*t**2 + 1

    See Also
    ========

    bell, bernoulli, catalan, euler, harmonic, lucas, genocchi, partition, tribonacci

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fibonacci_number
    .. [2] https://mathworld.wolfram.com/FibonacciNumber.html

    """

    @staticmethod
    def _fib(n):
        return _ifib(n)

    @staticmethod
    @recurrence_memo([None, S.One, _sym])
    def _fibpoly(n, prev):
        return (prev[-2] + _sym*prev[-1]).expand()

    @classmethod
    def eval(cls, n, sym=None):
        if n is S.Infinity:
            return S.Infinity

        if n.is_Integer:
            if sym is None:
                n = int(n)
                if n < 0:
                    return S.NegativeOne**(n + 1) * fibonacci(-n)
                else:
                    return Integer(cls._fib(n))
            else:
                if n < 1:
                    raise ValueError("Fibonacci polynomials are defined "
                       "only for positive integer indices.")
                return cls._fibpoly(n).subs(_sym, sym)

    def _eval_rewrite_as_tractable(self, n, **kwargs):
        from sympy.functions import sqrt, cos
        return (S.GoldenRatio**n - cos(S.Pi*n)/S.GoldenRatio**n)/sqrt(5)

    def _eval_rewrite_as_sqrt(self, n, **kwargs):
        from sympy.functions.elementary.miscellaneous import sqrt
        return 2**(-n)*sqrt(5)*((1 + sqrt(5))**n - (-sqrt(5) + 1)**n) / 5

    def _eval_rewrite_as_GoldenRatio(self,n, **kwargs):
        return (S.GoldenRatio**n - 1/(-S.GoldenRatio)**n)/(2*S.GoldenRatio-1)


#----------------------------------------------------------------------------#
#                                                                            #
#                               Lucas numbers                                #
#                                                                            #
#----------------------------------------------------------------------------#


class lucas(DefinedFunction):
    """
    Lucas numbers

    Lucas numbers satisfy a recurrence relation similar to that of
    the Fibonacci sequence, in which each term is the sum of the
    preceding two. They are generated by choosing the initial
    values `L_0 = 2` and `L_1 = 1`.

    * ``lucas(n)`` gives the `n^{th}` Lucas number

    Examples
    ========

    >>> from sympy import lucas

    >>> [lucas(x) for x in range(11)]
    [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123]

    See Also
    ========

    bell, bernoulli, catalan, euler, fibonacci, harmonic, genocchi, partition, tribonacci

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lucas_number
    .. [2] https://mathworld.wolfram.com/LucasNumber.html

    """

    @classmethod
    def eval(cls, n):
        if n is S.Infinity:
            return S.Infinity

        if n.is_Integer:
            return fibonacci(n + 1) + fibonacci(n - 1)

    def _eval_rewrite_as_sqrt(self, n, **kwargs):
        from sympy.functions.elementary.miscellaneous import sqrt
        return 2**(-n)*((1 + sqrt(5))**n + (-sqrt(5) + 1)**n)


#----------------------------------------------------------------------------#
#                                                                            #
#                             Tribonacci numbers                             #
#                                                                            #
#----------------------------------------------------------------------------#


class tribonacci(DefinedFunction):
    r"""
    Tribonacci numbers / Tribonacci polynomials

    The Tribonacci numbers are the integer sequence defined by the
    initial terms `T_0 = 0`, `T_1 = 1`, `T_2 = 1` and the three-term
    recurrence relation `T_n = T_{n-1} + T_{n-2} + T_{n-3}`.

    The Tribonacci polynomials are defined by `T_0(x) = 0`, `T_1(x) = 1`,
    `T_2(x) = x^2`, and `T_n(x) = x^2 T_{n-1}(x) + x T_{n-2}(x) + T_{n-3}(x)`
    for `n > 2`.  For all positive integers `n`, `T_n(1) = T_n`.

    * ``tribonacci(n)`` gives the `n^{th}` Tribonacci number, `T_n`
    * ``tribonacci(n, x)`` gives the `n^{th}` Tribonacci polynomial in `x`, `T_n(x)`

    Examples
    ========

    >>> from sympy import tribonacci, Symbol

    >>> [tribonacci(x) for x in range(11)]
    [0, 1, 1, 2, 4, 7, 13, 24, 44, 81, 149]
    >>> tribonacci(5, Symbol('t'))
    t**8 + 3*t**5 + 3*t**2

    See Also
    ========

    bell, bernoulli, catalan, euler, fibonacci, harmonic, lucas, genocchi, partition

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Generalizations_of_Fibonacci_numbers#Tribonacci_numbers
    .. [2] https://mathworld.wolfram.com/TribonacciNumber.html
    .. [3] https://oeis.org/A000073

    """

    @staticmethod
    @recurrence_memo([S.Zero, S.One, S.One])
    def _trib(n, prev):
        return (prev[-3] + prev[-2] + prev[-1])

    @staticmethod
    @recurrence_memo([S.Zero, S.One, _sym**2])
    def _tribpoly(n, prev):
        return (prev[-3] + _sym*prev[-2] + _sym**2*prev[-1]).expand()

    @classmethod
    def eval(cls, n, sym=None):
        if n is S.Infinity:
            return S.Infinity

        if n.is_Integer:
            n = int(n)
            if n < 0:
                raise ValueError("Tribonacci polynomials are defined "
                       "only for non-negative integer indices.")
            if sym is None:
                return Integer(cls._trib(n))
            else:
                return cls._tribpoly(n).subs(_sym, sym)

    def _eval_rewrite_as_sqrt(self, n, **kwargs):
        from sympy.functions.elementary.miscellaneous import cbrt, sqrt
        w = (-1 + S.ImaginaryUnit * sqrt(3)) / 2
        a = (1 + cbrt(19 + 3*sqrt(33)) + cbrt(19 - 3*sqrt(33))) / 3
        b = (1 + w*cbrt(19 + 3*sqrt(33)) + w**2*cbrt(19 - 3*sqrt(33))) / 3
        c = (1 + w**2*cbrt(19 + 3*sqrt(33)) + w*cbrt(19 - 3*sqrt(33))) / 3
        Tn = (a**(n + 1)/((a - b)*(a - c))
            + b**(n + 1)/((b - a)*(b - c))
            + c**(n + 1)/((c - a)*(c - b)))
        return Tn

    def _eval_rewrite_as_TribonacciConstant(self, n, **kwargs):
        from sympy.functions.elementary.integers import floor
        from sympy.functions.elementary.miscellaneous import cbrt, sqrt
        b = cbrt(586 + 102*sqrt(33))
        Tn = 3 * b * S.TribonacciConstant**n / (b**2 - 2*b + 4)
        return floor(Tn + S.Half)


#----------------------------------------------------------------------------#
#                                                                            #
#                           Bernoulli numbers                                #
#                                                                            #
#----------------------------------------------------------------------------#


class bernoulli(DefinedFunction):
    r"""
    Bernoulli numbers / Bernoulli polynomials / Bernoulli function

    The Bernoulli numbers are a sequence of rational numbers
    defined by `B_0 = 1` and the recursive relation (`n > 0`):

    .. math :: n+1 = \sum_{k=0}^n \binom{n+1}{k} B_k

    They are also commonly defined by their exponential generating
    function, which is `\frac{x}{1 - e^{-x}}`. For odd indices > 1,
    the Bernoulli numbers are zero.

    The Bernoulli polynomials satisfy the analogous formula:

    .. math :: B_n(x) = \sum_{k=0}^n (-1)^k \binom{n}{k} B_k x^{n-k}

    Bernoulli numbers and Bernoulli polynomials are related as
    `B_n(1) = B_n`.

    The generalized Bernoulli function `\operatorname{B}(s, a)`
    is defined for any complex `s` and `a`, except where `a` is a
    nonpositive integer and `s` is not a nonnegative integer. It is
    an entire function of `s` for fixed `a`, related to the Hurwitz
    zeta function by

    .. math:: \operatorname{B}(s, a) = \begin{cases}
              -s \zeta(1-s, a) & s \ne 0 \\ 1 & s = 0 \end{cases}

    When `s` is a nonnegative integer this function reduces to the
    Bernoulli polynomials: `\operatorname{B}(n, x) = B_n(x)`. When
    `a` is omitted it is assumed to be 1, yielding the (ordinary)
    Bernoulli function which interpolates the Bernoulli numbers and is
    related to the Riemann zeta function.

    We compute Bernoulli numbers using Ramanujan's formula:

    .. math :: B_n = \frac{A(n) - S(n)}{\binom{n+3}{n}}

    where:

    .. math :: A(n) = \begin{cases} \frac{n+3}{3} &
        n \equiv 0\ \text{or}\ 2 \pmod{6} \\
        -\frac{n+3}{6} & n \equiv 4 \pmod{6} \end{cases}

    and:

    .. math :: S(n) = \sum_{k=1}^{[n/6]} \binom{n+3}{n-6k} B_{n-6k}

    This formula is similar to the sum given in the definition, but
    cuts `\frac{2}{3}` of the terms. For Bernoulli polynomials, we use
    Appell sequences.

    For `n` a nonnegative integer and `s`, `a`, `x` arbitrary complex numbers,

    * ``bernoulli(n)`` gives the nth Bernoulli number, `B_n`
    * ``bernoulli(s)`` gives the Bernoulli function `\operatorname{B}(s)`
    * ``bernoulli(n, x)`` gives the nth Bernoulli polynomial in `x`, `B_n(x)`
    * ``bernoulli(s, a)`` gives the generalized Bernoulli function
      `\operatorname{B}(s, a)`

    .. versionchanged:: 1.12
        ``bernoulli(1)`` gives `+\frac{1}{2}` instead of `-\frac{1}{2}`.
        This choice of value confers several theoretical advantages [5]_,
        including the extension to complex parameters described above
        which this function now implements. The previous behavior, defined
        only for nonnegative integers `n`, can be obtained with
        ``(-1)**n*bernoulli(n)``.

    Examples
    ========

    >>> from sympy import bernoulli
    >>> from sympy.abc import x
    >>> [bernoulli(n) for n in range(11)]
    [1, 1/2, 1/6, 0, -1/30, 0, 1/42, 0, -1/30, 0, 5/66]
    >>> bernoulli(1000001)
    0
    >>> bernoulli(3, x)
    x**3 - 3*x**2/2 + x/2

    See Also
    ========

    andre, bell, catalan, euler, fibonacci, harmonic, lucas, genocchi,
    partition, tribonacci, sympy.polys.appellseqs.bernoulli_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bernoulli_number
    .. [2] https://en.wikipedia.org/wiki/Bernoulli_polynomial
    .. [3] https://mathworld.wolfram.com/BernoulliNumber.html
    .. [4] https://mathworld.wolfram.com/BernoulliPolynomial.html
    .. [5] Peter Luschny, "The Bernoulli Manifesto",
           https://luschny.de/math/zeta/The-Bernoulli-Manifesto.html
    .. [6] Peter Luschny, "An introduction to the Bernoulli function",
           https://arxiv.org/abs/2009.06743

    """

    args: tuple[Integer]

    # Calculates B_n for positive even n
    @staticmethod
    def _calc_bernoulli(n):
        s = 0
        a = int(binomial(n + 3, n - 6))
        for j in range(1, n//6 + 1):
            s += a * bernoulli(n - 6*j)
            # Avoid computing each binomial coefficient from scratch
            a *= _product(n - 6 - 6*j + 1, n - 6*j)
            a //= _product(6*j + 4, 6*j + 9)
        if n % 6 == 4:
            s = -Rational(n + 3, 6) - s
        else:
            s = Rational(n + 3, 3) - s
        return s / binomial(n + 3, n)

    # We implement a specialized memoization scheme to handle each
    # case modulo 6 separately
    _cache = {0: S.One, 1: Rational(1, 2), 2: Rational(1, 6), 4: Rational(-1, 30)}
    _highest = {0: 0, 1: 1, 2: 2, 4: 4}

    @classmethod
    def eval(cls, n, x=None):
        if x is S.One:
            return cls(n)
        elif n.is_zero:
            return S.One
        elif n.is_integer is False or n.is_nonnegative is False:
            if x is not None and x.is_Integer and x.is_nonpositive:
                return S.NaN
            return
        # Bernoulli numbers
        elif x is None:
            if n is S.One:
                return S.Half
            elif n.is_odd and (n-1).is_positive:
                return S.Zero
            elif n.is_Number:
                n = int(n)
                # Use mpmath for enormous Bernoulli numbers
                if n > 500:
                    p, q = mp.bernfrac(n)
                    return Rational(int(p), int(q))
                case = n % 6
                highest_cached = cls._highest[case]
                if n <= highest_cached:
                    return cls._cache[n]
                # To avoid excessive recursion when, say, bernoulli(1000) is
                # requested, calculate and cache the entire sequence ... B_988,
                # B_994, B_1000 in increasing order
                for i in range(highest_cached + 6, n + 6, 6):
                    b = cls._calc_bernoulli(i)
                    cls._cache[i] = b
                    cls._highest[case] = i
                return b
        # Bernoulli polynomials
        elif n.is_Number:
            return bernoulli_poly(n, x)

    def _eval_rewrite_as_zeta(self, n, x=1, **kwargs):
        from sympy.functions.special.zeta_functions import zeta
        return Piecewise((1, Eq(n, 0)), (-n * zeta(1-n, x), True))

    def _eval_evalf(self, prec):
        if not all(x.is_number for x in self.args):
            return
        n = self.args[0]._to_mpmath(prec)
        x = (self.args[1] if len(self.args) > 1 else S.One)._to_mpmath(prec)
        with workprec(prec):
            if n == 0:
                res = mp.mpf(1)
            elif n == 1:
                res = x - mp.mpf(0.5)
            elif mp.isint(n) and n >= 0:
                res = mp.bernoulli(n) if x == 1 else mp.bernpoly(n, x)
            else:
                res = -n * mp.zeta(1-n, x)
        return Expr._from_mpmath(res, prec)


#----------------------------------------------------------------------------#
#                                                                            #
#                                Bell numbers                                #
#                                                                            #
#----------------------------------------------------------------------------#


class bell(DefinedFunction):
    r"""
    Bell numbers / Bell polynomials

    The Bell numbers satisfy `B_0 = 1` and

    .. math:: B_n = \sum_{k=0}^{n-1} \binom{n-1}{k} B_k.

    They are also given by:

    .. math:: B_n = \frac{1}{e} \sum_{k=0}^{\infty} \frac{k^n}{k!}.

    The Bell polynomials are given by `B_0(x) = 1` and

    .. math:: B_n(x) = x \sum_{k=1}^{n-1} \binom{n-1}{k-1} B_{k-1}(x).

    The second kind of Bell polynomials (are sometimes called "partial" Bell
    polynomials or incomplete Bell polynomials) are defined as

    .. math:: B_{n,k}(x_1, x_2,\dotsc x_{n-k+1}) =
            \sum_{j_1+j_2+j_2+\dotsb=k \atop j_1+2j_2+3j_2+\dotsb=n}
                \frac{n!}{j_1!j_2!\dotsb j_{n-k+1}!}
                \left(\frac{x_1}{1!} \right)^{j_1}
                \left(\frac{x_2}{2!} \right)^{j_2} \dotsb
                \left(\frac{x_{n-k+1}}{(n-k+1)!} \right) ^{j_{n-k+1}}.

    * ``bell(n)`` gives the `n^{th}` Bell number, `B_n`.
    * ``bell(n, x)`` gives the `n^{th}` Bell polynomial, `B_n(x)`.
    * ``bell(n, k, (x1, x2, ...))`` gives Bell polynomials of the second kind,
      `B_{n,k}(x_1, x_2, \dotsc, x_{n-k+1})`.

    Notes
    =====

    Not to be confused with Bernoulli numbers and Bernoulli polynomials,
    which use the same notation.

    Examples
    ========

    >>> from sympy import bell, Symbol, symbols

    >>> [bell(n) for n in range(11)]
    [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975]
    >>> bell(30)
    846749014511809332450147
    >>> bell(4, Symbol('t'))
    t**4 + 6*t**3 + 7*t**2 + t
    >>> bell(6, 2, symbols('x:6')[1:])
    6*x1*x5 + 15*x2*x4 + 10*x3**2

    See Also
    ========

    bernoulli, catalan, euler, fibonacci, harmonic, lucas, genocchi, partition, tribonacci

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bell_number
    .. [2] https://mathworld.wolfram.com/BellNumber.html
    .. [3] https://mathworld.wolfram.com/BellPolynomial.html

    """

    @staticmethod
    @recurrence_memo([1, 1])
    def _bell(n, prev):
        s = 1
        a = 1
        for k in range(1, n):
            a = a * (n - k) // k
            s += a * prev[k]
        return s

    @staticmethod
    @recurrence_memo([S.One, _sym])
    def _bell_poly(n, prev):
        s = 1
        a = 1
        for k in range(2, n + 1):
            a = a * (n - k + 1) // (k - 1)
            s += a * prev[k - 1]
        return expand_mul(_sym * s)

    @staticmethod
    def _bell_incomplete_poly(n, k, symbols):
        r"""
        The second kind of Bell polynomials (incomplete Bell polynomials).

        Calculated by recurrence formula:

        .. math:: B_{n,k}(x_1, x_2, \dotsc, x_{n-k+1}) =
                \sum_{m=1}^{n-k+1}
                \x_m \binom{n-1}{m-1} B_{n-m,k-1}(x_1, x_2, \dotsc, x_{n-m-k})

        where
            `B_{0,0} = 1;`
            `B_{n,0} = 0; for n \ge 1`
            `B_{0,k} = 0; for k \ge 1`

        """
        if (n == 0) and (k == 0):
            return S.One
        elif (n == 0) or (k == 0):
            return S.Zero
        s = S.Zero
        a = S.One
        for m in range(1, n - k + 2):
            s += a * bell._bell_incomplete_poly(
                n - m, k - 1, symbols) * symbols[m - 1]
            a = a * (n - m) / m
        return expand_mul(s)

    @classmethod
    def eval(cls, n, k_sym=None, symbols=None):
        if n is S.Infinity:
            if k_sym is None:
                return S.Infinity
            else:
                raise ValueError("Bell polynomial is not defined")

        if n.is_negative or n.is_integer is False:
            raise ValueError("a non-negative integer expected")

        if n.is_Integer and n.is_nonnegative:
            if k_sym is None:
                return Integer(cls._bell(int(n)))
            elif symbols is None:
                return cls._bell_poly(int(n)).subs(_sym, k_sym)
            else:
                r = cls._bell_incomplete_poly(int(n), int(k_sym), symbols)
                return r

    def _eval_rewrite_as_Sum(self, n, k_sym=None, symbols=None, **kwargs):
        from sympy.concrete.summations import Sum
        if (k_sym is not None) or (symbols is not None):
            return self

        # Dobinski's formula
        if not n.is_nonnegative:
            return self
        k = Dummy('k', integer=True, nonnegative=True)
        return 1 / E * Sum(k**n / factorial(k), (k, 0, S.Infinity))


#----------------------------------------------------------------------------#
#                                                                            #
#                              Harmonic numbers                              #
#                                                                            #
#----------------------------------------------------------------------------#


class harmonic(DefinedFunction):
    r"""
    Harmonic numbers

    The nth harmonic number is given by `\operatorname{H}_{n} =
    1 + \frac{1}{2} + \frac{1}{3} + \ldots + \frac{1}{n}`.

    More generally:

    .. math:: \operatorname{H}_{n,m} = \sum_{k=1}^{n} \frac{1}{k^m}

    As `n \rightarrow \infty`, `\operatorname{H}_{n,m} \rightarrow \zeta(m)`,
    the Riemann zeta function.

    * ``harmonic(n)`` gives the nth harmonic number, `\operatorname{H}_n`

    * ``harmonic(n, m)`` gives the nth generalized harmonic number
      of order `m`, `\operatorname{H}_{n,m}`, where
      ``harmonic(n) == harmonic(n, 1)``

    This function can be extended to complex `n` and `m` where `n` is not a
    negative integer or `m` is a nonpositive integer as

    .. math:: \operatorname{H}_{n,m} = \begin{cases} \zeta(m) - \zeta(m, n+1)
            & m \ne 1 \\ \psi(n+1) + \gamma & m = 1 \end{cases}

    Examples
    ========

    >>> from sympy import harmonic, oo

    >>> [harmonic(n) for n in range(6)]
    [0, 1, 3/2, 11/6, 25/12, 137/60]
    >>> [harmonic(n, 2) for n in range(6)]
    [0, 1, 5/4, 49/36, 205/144, 5269/3600]
    >>> harmonic(oo, 2)
    pi**2/6

    >>> from sympy import Symbol, Sum
    >>> n = Symbol("n")

    >>> harmonic(n).rewrite(Sum)
    Sum(1/_k, (_k, 1, n))

    We can evaluate harmonic numbers for all integral and positive
    rational arguments:

    >>> from sympy import S, expand_func, simplify
    >>> harmonic(8)
    761/280
    >>> harmonic(11)
    83711/27720

    >>> H = harmonic(1/S(3))
    >>> H
    harmonic(1/3)
    >>> He = expand_func(H)
    >>> He
    -log(6) - sqrt(3)*pi/6 + 2*Sum(log(sin(_k*pi/3))*cos(2*_k*pi/3), (_k, 1, 1))
                           + 3*Sum(1/(3*_k + 1), (_k, 0, 0))
    >>> He.doit()
    -log(6) - sqrt(3)*pi/6 - log(sqrt(3)/2) + 3
    >>> H = harmonic(25/S(7))
    >>> He = simplify(expand_func(H).doit())
    >>> He
    log(sin(2*pi/7)**(2*cos(16*pi/7))/(14*sin(pi/7)**(2*cos(pi/7))*cos(pi/14)**(2*sin(pi/14)))) + pi*tan(pi/14)/2 + 30247/9900
    >>> He.n(40)
    1.983697455232980674869851942390639915940
    >>> harmonic(25/S(7)).n(40)
    1.983697455232980674869851942390639915940

    We can rewrite harmonic numbers in terms of polygamma functions:

    >>> from sympy import digamma, polygamma
    >>> m = Symbol("m", integer=True, positive=True)

    >>> harmonic(n).rewrite(digamma)
    polygamma(0, n + 1) + EulerGamma

    >>> harmonic(n).rewrite(polygamma)
    polygamma(0, n + 1) + EulerGamma

    >>> harmonic(n,3).rewrite(polygamma)
    polygamma(2, n + 1)/2 + zeta(3)

    >>> simplify(harmonic(n,m).rewrite(polygamma))
    Piecewise((polygamma(0, n + 1) + EulerGamma, Eq(m, 1)),
    (-(-1)**m*polygamma(m - 1, n + 1)/factorial(m - 1) + zeta(m), True))

    Integer offsets in the argument can be pulled out:

    >>> from sympy import expand_func

    >>> expand_func(harmonic(n+4))
    harmonic(n) + 1/(n + 4) + 1/(n + 3) + 1/(n + 2) + 1/(n + 1)

    >>> expand_func(harmonic(n-4))
    harmonic(n) - 1/(n - 1) - 1/(n - 2) - 1/(n - 3) - 1/n

    Some limits can be computed as well:

    >>> from sympy import limit, oo

    >>> limit(harmonic(n), n, oo)
    oo

    >>> limit(harmonic(n, 2), n, oo)
    pi**2/6

    >>> limit(harmonic(n, 3), n, oo)
    zeta(3)

    For `m > 1`, `H_{n,m}` tends to `\zeta(m)` in the limit of infinite `n`:

    >>> m = Symbol("m", positive=True)
    >>> limit(harmonic(n, m+1), n, oo)
    zeta(m + 1)

    See Also
    ========

    bell, bernoulli, catalan, euler, fibonacci, lucas, genocchi, partition, tribonacci

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Harmonic_number
    .. [2] https://functions.wolfram.com/GammaBetaErf/HarmonicNumber/
    .. [3] https://functions.wolfram.com/GammaBetaErf/HarmonicNumber2/

    """

    # This prevents redundant recalculations and speeds up harmonic number computations.
    harmonic_cache: dict[Integer, Callable[[int], Rational]] = {}

    @classmethod
    def eval(cls, n, m=None):
        from sympy.functions.special.zeta_functions import zeta
        if m is S.One:
            return cls(n)
        if m is None:
            m = S.One
        if n.is_zero:
            return S.Zero
        elif m.is_zero:
            return n
        elif n is S.Infinity:
            if m.is_negative:
                return S.NaN
            elif is_le(m, S.One):
                return S.Infinity
            elif is_gt(m, S.One):
                return zeta(m)
        elif m.is_Integer and m.is_nonpositive:
            return (bernoulli(1-m, n+1) - bernoulli(1-m)) / (1-m)
        elif n.is_Integer:
            if n.is_negative and (m.is_integer is False or m.is_nonpositive is False):
                return S.ComplexInfinity if m is S.One else S.NaN
            if n.is_nonnegative:
                if m.is_Integer:
                    if m not in cls.harmonic_cache:
                        @recurrence_memo([0])
                        def f(n, prev):
                            return prev[-1] + S.One / n**m
                        cls.harmonic_cache[m] = f
                    return cls.harmonic_cache[m](int(n))
                return Add(*(k**(-m) for k in range(1, int(n) + 1)))

    def _eval_rewrite_as_polygamma(self, n, m=S.One, **kwargs):
        from sympy.functions.special.gamma_functions import gamma, polygamma
        if m.is_integer and m.is_positive:
            return Piecewise((polygamma(0, n+1) + S.EulerGamma, Eq(m, 1)),
                    (S.NegativeOne**m * (polygamma(m-1, 1) - polygamma(m-1, n+1)) /
                    gamma(m), True))

    def _eval_rewrite_as_digamma(self, n, m=1, **kwargs):
        from sympy.functions.special.gamma_functions import polygamma
        return self.rewrite(polygamma)

    def _eval_rewrite_as_trigamma(self, n, m=1, **kwargs):
        from sympy.functions.special.gamma_functions import polygamma
        return self.rewrite(polygamma)

    def _eval_rewrite_as_Sum(self, n, m=None, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy("k", integer=True)
        if m is None:
            m = S.One
        return Sum(k**(-m), (k, 1, n))

    def _eval_rewrite_as_zeta(self, n, m=S.One, **kwargs):
        from sympy.functions.special.zeta_functions import zeta
        from sympy.functions.special.gamma_functions import digamma
        return Piecewise((digamma(n + 1) + S.EulerGamma, Eq(m, 1)),
                         (zeta(m) - zeta(m, n+1), True))

    def _eval_expand_func(self, **hints):
        from sympy.concrete.summations import Sum
        n = self.args[0]
        m = self.args[1] if len(self.args) == 2 else 1

        if m == S.One:
            if n.is_Add:
                off = n.args[0]
                nnew = n - off
                if off.is_Integer and off.is_positive:
                    result = [S.One/(nnew + i) for i in range(off, 0, -1)] + [harmonic(nnew)]
                    return Add(*result)
                elif off.is_Integer and off.is_negative:
                    result = [-S.One/(nnew + i) for i in range(0, off, -1)] + [harmonic(nnew)]
                    return Add(*result)

            if n.is_Rational:
                # Expansions for harmonic numbers at general rational arguments (u + p/q)
                # Split n as u + p/q with p < q
                p, q = n.as_numer_denom()
                u = p // q
                p = p - u * q
                if u.is_nonnegative and p.is_positive and q.is_positive and p < q:
                    from sympy.functions.elementary.exponential import log
                    from sympy.functions.elementary.integers import floor
                    from sympy.functions.elementary.trigonometric import sin, cos, cot
                    k = Dummy("k")
                    t1 = q * Sum(1 / (q * k + p), (k, 0, u))
                    t2 = 2 * Sum(cos((2 * pi * p * k) / S(q)) *
                                   log(sin((pi * k) / S(q))),
                                   (k, 1, floor((q - 1) / S(2))))
                    t3 = (pi / 2) * cot((pi * p) / q) + log(2 * q)
                    return t1 + t2 - t3

        return self

    def _eval_rewrite_as_tractable(self, n, m=1, limitvar=None, **kwargs):
        from sympy.functions.special.zeta_functions import zeta
        from sympy.functions.special.gamma_functions import polygamma
        pg = self.rewrite(polygamma)
        if not isinstance(pg, harmonic):
            return pg.rewrite("tractable", deep=True)
        arg = m - S.One
        if arg.is_nonzero:
            return (zeta(m) - zeta(m, n+1)).rewrite("tractable", deep=True)

    def _eval_evalf(self, prec):
        if not all(x.is_number for x in self.args):
            return
        n = self.args[0]._to_mpmath(prec)
        m = (self.args[1] if len(self.args) > 1 else S.One)._to_mpmath(prec)
        if mp.isint(n) and n < 0:
            return S.NaN
        with workprec(prec):
            if m == 1:
                res = mp.harmonic(n)
            else:
                res = mp.zeta(m) - mp.zeta(m, n+1)
        return Expr._from_mpmath(res, prec)

    def fdiff(self, argindex=1):
        from sympy.functions.special.zeta_functions import zeta
        if len(self.args) == 2:
            n, m = self.args
        else:
            n, m = self.args + (1,)
        if argindex == 1:
            return m * zeta(m+1, n+1)
        else:
            raise ArgumentIndexError


#----------------------------------------------------------------------------#
#                                                                            #
#                           Euler numbers                                    #
#                                                                            #
#----------------------------------------------------------------------------#


class euler(DefinedFunction):
    r"""
    Euler numbers / Euler polynomials / Euler function

    The Euler numbers are given by:

    .. math:: E_{2n} = I \sum_{k=1}^{2n+1} \sum_{j=0}^k \binom{k}{j}
        \frac{(-1)^j (k-2j)^{2n+1}}{2^k I^k k}

    .. math:: E_{2n+1} = 0

    Euler numbers and Euler polynomials are related by

    .. math:: E_n = 2^n E_n\left(\frac{1}{2}\right).

    We compute symbolic Euler polynomials using Appell sequences,
    but numerical evaluation of the Euler polynomial is computed
    more efficiently (and more accurately) using the mpmath library.

    The Euler polynomials are special cases of the generalized Euler function,
    related to the Genocchi function as

    .. math:: \operatorname{E}(s, a) = -\frac{\operatorname{G}(s+1, a)}{s+1}

    with the limit of `\psi\left(\frac{a+1}{2}\right) - \psi\left(\frac{a}{2}\right)`
    being taken when `s = -1`. The (ordinary) Euler function interpolating
    the Euler numbers is then obtained as
    `\operatorname{E}(s) = 2^s \operatorname{E}\left(s, \frac{1}{2}\right)`.

    * ``euler(n)`` gives the nth Euler number `E_n`.
    * ``euler(s)`` gives the Euler function `\operatorname{E}(s)`.
    * ``euler(n, x)`` gives the nth Euler polynomial `E_n(x)`.
    * ``euler(s, a)`` gives the generalized Euler function `\operatorname{E}(s, a)`.

    Examples
    ========

    >>> from sympy import euler, Symbol, S
    >>> [euler(n) for n in range(10)]
    [1, 0, -1, 0, 5, 0, -61, 0, 1385, 0]
    >>> [2**n*euler(n,1) for n in range(10)]
    [1, 1, 0, -2, 0, 16, 0, -272, 0, 7936]
    >>> n = Symbol("n")
    >>> euler(n + 2*n)
    euler(3*n)

    >>> x = Symbol("x")
    >>> euler(n, x)
    euler(n, x)

    >>> euler(0, x)
    1
    >>> euler(1, x)
    x - 1/2
    >>> euler(2, x)
    x**2 - x
    >>> euler(3, x)
    x**3 - 3*x**2/2 + 1/4
    >>> euler(4, x)
    x**4 - 2*x**3 + x

    >>> euler(12, S.Half)
    2702765/4096
    >>> euler(12)
    2702765

    See Also
    ========

    andre, bell, bernoulli, catalan, fibonacci, harmonic, lucas, genocchi,
    partition, tribonacci, sympy.polys.appellseqs.euler_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler_numbers
    .. [2] https://mathworld.wolfram.com/EulerNumber.html
    .. [3] https://en.wikipedia.org/wiki/Alternating_permutation
    .. [4] https://mathworld.wolfram.com/AlternatingPermutation.html

    """

    @classmethod
    def eval(cls, n, x=None):
        if n.is_zero:
            return S.One
        elif n is S.NegativeOne:
            if x is None:
                return S.Pi/2
            from sympy.functions.special.gamma_functions import digamma
            return digamma((x+1)/2) - digamma(x/2)
        elif n.is_integer is False or n.is_nonnegative is False:
            return
        # Euler numbers
        elif x is None:
            if n.is_odd and n.is_positive:
                return S.Zero
            elif n.is_Number:
                from mpmath import mp
                n = n._to_mpmath(mp.prec)
                res = mp.eulernum(n, exact=True)
                return Integer(res)
        # Euler polynomials
        elif n.is_Number:
            from sympy.core.evalf import pure_complex
            n = int(n)
            reim = pure_complex(x, or_real=True)
            if reim and all(a.is_Float or a.is_Integer for a in reim) \
                    and any(a.is_Float for a in reim):
                from mpmath import mp
                prec = min([a._prec for a in reim if a.is_Float])
                with workprec(prec):
                    res = mp.eulerpoly(n, x)
                return Expr._from_mpmath(res, prec)
            return euler_poly(n, x)

    def _eval_rewrite_as_Sum(self, n, x=None, **kwargs):
        from sympy.concrete.summations import Sum
        if x is None and n.is_even:
            k = Dummy("k", integer=True)
            j = Dummy("j", integer=True)
            n = n / 2
            Em = (S.ImaginaryUnit * Sum(Sum(binomial(k, j) * (S.NegativeOne**j *
                                                              (k - 2*j)**(2*n + 1)) /
                  (2**k*S.ImaginaryUnit**k * k), (j, 0, k)), (k, 1, 2*n + 1)))
            return Em
        if x:
            k = Dummy("k", integer=True)
            return Sum(binomial(n, k)*euler(k)/2**k*(x - S.Half)**(n - k), (k, 0, n))

    def _eval_rewrite_as_genocchi(self, n, x=None, **kwargs):
        if x is None:
            return Piecewise((S.Pi/2, Eq(n, -1)),
                             (-2**n * genocchi(n+1, S.Half) / (n+1), True))
        from sympy.functions.special.gamma_functions import digamma
        return Piecewise((digamma((x+1)/2) - digamma(x/2), Eq(n, -1)),
                         (-genocchi(n+1, x) / (n+1), True))

    def _eval_evalf(self, prec):
        if not all(i.is_number for i in self.args):
            return
        from mpmath import mp
        m, x = (self.args[0], None) if len(self.args) == 1 else self.args
        m = m._to_mpmath(prec)
        if x is not None:
            x = x._to_mpmath(prec)
        with workprec(prec):
            if mp.isint(m) and m >= 0:
                res = mp.eulernum(m) if x is None else mp.eulerpoly(m, x)
            else:
                if m == -1:
                    res = mp.pi if x is None else mp.digamma((x+1)/2) - mp.digamma(x/2)
                else:
                    y = 0.5 if x is None else x
                    res = 2 * (mp.zeta(-m, y) - 2**(m+1) * mp.zeta(-m, (y+1)/2))
                if x is None:
                    res *= 2**m
        return Expr._from_mpmath(res, prec)


#----------------------------------------------------------------------------#
#                                                                            #
#                              Catalan numbers                               #
#                                                                            #
#----------------------------------------------------------------------------#


class catalan(DefinedFunction):
    r"""
    Catalan numbers

    The `n^{th}` catalan number is given by:

    .. math :: C_n = \frac{1}{n+1} \binom{2n}{n}

    * ``catalan(n)`` gives the `n^{th}` Catalan number, `C_n`

    Examples
    ========

    >>> from sympy import (Symbol, binomial, gamma, hyper,
    ...     catalan, diff, combsimp, Rational, I)

    >>> [catalan(i) for i in range(1,10)]
    [1, 2, 5, 14, 42, 132, 429, 1430, 4862]

    >>> n = Symbol("n", integer=True)

    >>> catalan(n)
    catalan(n)

    Catalan numbers can be transformed into several other, identical
    expressions involving other mathematical functions

    >>> catalan(n).rewrite(binomial)
    binomial(2*n, n)/(n + 1)

    >>> catalan(n).rewrite(gamma)
    4**n*gamma(n + 1/2)/(sqrt(pi)*gamma(n + 2))

    >>> catalan(n).rewrite(hyper)
    hyper((-n, 1 - n), (2,), 1)

    For some non-integer values of n we can get closed form
    expressions by rewriting in terms of gamma functions:

    >>> catalan(Rational(1, 2)).rewrite(gamma)
    8/(3*pi)

    We can differentiate the Catalan numbers C(n) interpreted as a
    continuous real function in n:

    >>> diff(catalan(n), n)
    (polygamma(0, n + 1/2) - polygamma(0, n + 2) + log(4))*catalan(n)

    As a more advanced example consider the following ratio
    between consecutive numbers:

    >>> combsimp((catalan(n + 1)/catalan(n)).rewrite(binomial))
    2*(2*n + 1)/(n + 2)

    The Catalan numbers can be generalized to complex numbers:

    >>> catalan(I).rewrite(gamma)
    4**I*gamma(1/2 + I)/(sqrt(pi)*gamma(2 + I))

    and evaluated with arbitrary precision:

    >>> catalan(I).evalf(20)
    0.39764993382373624267 - 0.020884341620842555705*I

    See Also
    ========

    andre, bell, bernoulli, euler, fibonacci, harmonic, lucas, genocchi,
    partition, tribonacci, sympy.functions.combinatorial.factorials.binomial

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Catalan_number
    .. [2] https://mathworld.wolfram.com/CatalanNumber.html
    .. [3] https://functions.wolfram.com/GammaBetaErf/CatalanNumber/
    .. [4] http://geometer.org/mathcircles/catalan.pdf

    """

    @classmethod
    def eval(cls, n):
        from sympy.functions.special.gamma_functions import gamma
        if (n.is_Integer and n.is_nonnegative) or \
           (n.is_noninteger and n.is_negative):
            return 4**n*gamma(n + S.Half)/(gamma(S.Half)*gamma(n + 2))

        if (n.is_integer and n.is_negative):
            if (n + 1).is_negative:
                return S.Zero
            if (n + 1).is_zero:
                return Rational(-1, 2)

    def fdiff(self, argindex=1):
        from sympy.functions.elementary.exponential import log
        from sympy.functions.special.gamma_functions import polygamma
        n = self.args[0]
        return catalan(n)*(polygamma(0, n + S.Half) - polygamma(0, n + 2) + log(4))

    def _eval_rewrite_as_binomial(self, n, **kwargs):
        return binomial(2*n, n)/(n + 1)

    def _eval_rewrite_as_factorial(self, n, **kwargs):
        return factorial(2*n) / (factorial(n+1) * factorial(n))

    def _eval_rewrite_as_gamma(self, n, piecewise=True, **kwargs):
        from sympy.functions.special.gamma_functions import gamma
        # The gamma function allows to generalize Catalan numbers to complex n
        return 4**n*gamma(n + S.Half)/(gamma(S.Half)*gamma(n + 2))

    def _eval_rewrite_as_hyper(self, n, **kwargs):
        from sympy.functions.special.hyper import hyper
        return hyper([1 - n, -n], [2], 1)

    def _eval_rewrite_as_Product(self, n, **kwargs):
        from sympy.concrete.products import Product
        if not (n.is_integer and n.is_nonnegative):
            return self
        k = Dummy('k', integer=True, positive=True)
        return Product((n + k) / k, (k, 2, n))

    def _eval_is_integer(self):
        if self.args[0].is_integer and self.args[0].is_nonnegative:
            return True

    def _eval_is_positive(self):
        if self.args[0].is_nonnegative:
            return True

    def _eval_is_composite(self):
        if self.args[0].is_integer and (self.args[0] - 3).is_positive:
            return True

    def _eval_evalf(self, prec):
        from sympy.functions.special.gamma_functions import gamma
        if self.args[0].is_number:
            return self.rewrite(gamma)._eval_evalf(prec)


#----------------------------------------------------------------------------#
#                                                                            #
#                           Genocchi numbers                                 #
#                                                                            #
#----------------------------------------------------------------------------#


class genocchi(DefinedFunction):
    r"""
    Genocchi numbers / Genocchi polynomials / Genocchi function

    The Genocchi numbers are a sequence of integers `G_n` that satisfy the
    relation:

    .. math:: \frac{-2t}{1 + e^{-t}} = \sum_{n=0}^\infty \frac{G_n t^n}{n!}

    They are related to the Bernoulli numbers by

    .. math:: G_n = 2 (1 - 2^n) B_n

    and generalize like the Bernoulli numbers to the Genocchi polynomials and
    function as

    .. math:: \operatorname{G}(s, a) = 2 \left(\operatorname{B}(s, a) -
              2^s \operatorname{B}\left(s, \frac{a+1}{2}\right)\right)

    .. versionchanged:: 1.12
        ``genocchi(1)`` gives `-1` instead of `1`.

    Examples
    ========

    >>> from sympy import genocchi, Symbol
    >>> [genocchi(n) for n in range(9)]
    [0, -1, -1, 0, 1, 0, -3, 0, 17]
    >>> n = Symbol('n', integer=True, positive=True)
    >>> genocchi(2*n + 1)
    0
    >>> x = Symbol('x')
    >>> genocchi(4, x)
    -4*x**3 + 6*x**2 - 1

    See Also
    ========

    bell, bernoulli, catalan, euler, fibonacci, harmonic, lucas, partition, tribonacci
    sympy.polys.appellseqs.genocchi_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Genocchi_number
    .. [2] https://mathworld.wolfram.com/GenocchiNumber.html
    .. [3] Peter Luschny, "An introduction to the Bernoulli function",
           https://arxiv.org/abs/2009.06743

    """

    @classmethod
    def eval(cls, n, x=None):
        if x is S.One:
            return cls(n)
        elif n.is_integer is False or n.is_nonnegative is False:
            return
        # Genocchi numbers
        elif x is None:
            if n.is_odd and (n-1).is_positive:
                return S.Zero
            elif n.is_Number:
                return 2 * (1-S(2)**n) * bernoulli(n)
        # Genocchi polynomials
        elif n.is_Number:
            return genocchi_poly(n, x)

    def _eval_rewrite_as_bernoulli(self, n, x=1, **kwargs):
        if x == 1 and n.is_integer and n.is_nonnegative:
            return 2 * (1-S(2)**n) * bernoulli(n)
        return 2 * (bernoulli(n, x) - 2**n * bernoulli(n, (x+1) / 2))

    def _eval_rewrite_as_dirichlet_eta(self, n, x=1, **kwargs):
        from sympy.functions.special.zeta_functions import dirichlet_eta
        return -2*n * dirichlet_eta(1-n, x)

    def _eval_is_integer(self):
        if len(self.args) > 1 and self.args[1] != 1:
            return
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            return True

    def _eval_is_negative(self):
        if len(self.args) > 1 and self.args[1] != 1:
            return
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            if n.is_odd:
                return fuzzy_not((n-1).is_positive)
            return (n/2).is_odd

    def _eval_is_positive(self):
        if len(self.args) > 1 and self.args[1] != 1:
            return
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            if n.is_zero or n.is_odd:
                return False
            return (n/2).is_even

    def _eval_is_even(self):
        if len(self.args) > 1 and self.args[1] != 1:
            return
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            if n.is_even:
                return n.is_zero
            return (n-1).is_positive

    def _eval_is_odd(self):
        if len(self.args) > 1 and self.args[1] != 1:
            return
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            if n.is_even:
                return fuzzy_not(n.is_zero)
            return fuzzy_not((n-1).is_positive)

    def _eval_is_prime(self):
        if len(self.args) > 1 and self.args[1] != 1:
            return
        n = self.args[0]
        # only G_6 = -3 and G_8 = 17 are prime,
        # but SymPy does not consider negatives as prime
        # so only n=8 is tested
        return (n-8).is_zero

    def _eval_evalf(self, prec):
        if all(i.is_number for i in self.args):
            return self.rewrite(bernoulli)._eval_evalf(prec)


#----------------------------------------------------------------------------#
#                                                                            #
#                              Andre numbers                                 #
#                                                                            #
#----------------------------------------------------------------------------#


class andre(DefinedFunction):
    r"""
    Andre numbers / Andre function

    The Andre number `\mathcal{A}_n` is Luschny's name for half the number of
    *alternating permutations* on `n` elements, where a permutation is alternating
    if adjacent elements alternately compare "greater" and "smaller" going from
    left to right. For example, `2 < 3 > 1 < 4` is an alternating permutation.

    This sequence is A000111 in the OEIS, which assigns the names *up/down numbers*
    and *Euler zigzag numbers*. It satisfies a recurrence relation similar to that
    for the Catalan numbers, with `\mathcal{A}_0 = 1` and

    .. math:: 2 \mathcal{A}_{n+1} = \sum_{k=0}^n \binom{n}{k} \mathcal{A}_k \mathcal{A}_{n-k}

    The Bernoulli and Euler numbers are signed transformations of the odd- and
    even-indexed elements of this sequence respectively:

    .. math :: \operatorname{B}_{2k} = \frac{2k \mathcal{A}_{2k-1}}{(-4)^k - (-16)^k}

    .. math :: \operatorname{E}_{2k} = (-1)^k \mathcal{A}_{2k}

    Like the Bernoulli and Euler numbers, the Andre numbers are interpolated by the
    entire Andre function:

    .. math :: \mathcal{A}(s) = (-i)^{s+1} \operatorname{Li}_{-s}(i) +
            i^{s+1} \operatorname{Li}_{-s}(-i) = \\ \frac{2 \Gamma(s+1)}{(2\pi)^{s+1}}
            (\zeta(s+1, 1/4) - \zeta(s+1, 3/4) \cos{\pi s})

    Examples
    ========

    >>> from sympy import andre, euler, bernoulli
    >>> [andre(n) for n in range(11)]
    [1, 1, 1, 2, 5, 16, 61, 272, 1385, 7936, 50521]
    >>> [(-1)**k * andre(2*k) for k in range(7)]
    [1, -1, 5, -61, 1385, -50521, 2702765]
    >>> [euler(2*k) for k in range(7)]
    [1, -1, 5, -61, 1385, -50521, 2702765]
    >>> [andre(2*k-1) * (2*k) / ((-4)**k - (-16)**k) for k in range(1, 8)]
    [1/6, -1/30, 1/42, -1/30, 5/66, -691/2730, 7/6]
    >>> [bernoulli(2*k) for k in range(1, 8)]
    [1/6, -1/30, 1/42, -1/30, 5/66, -691/2730, 7/6]

    See Also
    ========

    bernoulli, catalan, euler, sympy.polys.appellseqs.andre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Alternating_permutation
    .. [2] https://mathworld.wolfram.com/EulerZigzagNumber.html
    .. [3] Peter Luschny, "An introduction to the Bernoulli function",
           https://arxiv.org/abs/2009.06743
    """

    @classmethod
    def eval(cls, n):
        if n is S.NaN:
            return S.NaN
        elif n is S.Infinity:
            return S.Infinity
        if n.is_zero:
            return S.One
        elif n == -1:
            return -log(2)
        elif n == -2:
            return -2*S.Catalan
        elif n.is_Integer:
            if n.is_nonnegative and n.is_even:
                return abs(euler(n))
            elif n.is_odd:
                from sympy.functions.special.zeta_functions import zeta
                m = -n-1
                return I**m * Rational(1-2**m, 4**m) * zeta(-n)

    def _eval_rewrite_as_zeta(self, s, **kwargs):
        from sympy.functions.elementary.trigonometric import cos
        from sympy.functions.special.gamma_functions import gamma
        from sympy.functions.special.zeta_functions import zeta
        return 2 * gamma(s+1) / (2*pi)**(s+1) * \
                (zeta(s+1, S.One/4) - cos(pi*s) * zeta(s+1, S(3)/4))

    def _eval_rewrite_as_polylog(self, s, **kwargs):
        from sympy.functions.special.zeta_functions import polylog
        return (-I)**(s+1) * polylog(-s, I) + I**(s+1) * polylog(-s, -I)

    def _eval_is_integer(self):
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            return True

    def _eval_is_positive(self):
        if self.args[0].is_nonnegative:
            return True

    def _eval_evalf(self, prec):
        if not self.args[0].is_number:
            return
        s = self.args[0]._to_mpmath(prec+12)
        with workprec(prec+12):
            sp, cp = mp.sinpi(s/2), mp.cospi(s/2)
            res = 2*mp.dirichlet(-s, (-sp, cp, sp, -cp))
        return Expr._from_mpmath(res, prec)


#----------------------------------------------------------------------------#
#                                                                            #
#                           Partition numbers                                #
#                                                                            #
#----------------------------------------------------------------------------#

class partition(DefinedFunction):
    r"""
    Partition numbers

    The Partition numbers are a sequence of integers `p_n` that represent the
    number of distinct ways of representing `n` as a sum of natural numbers
    (with order irrelevant). The generating function for `p_n` is given by:

    .. math:: \sum_{n=0}^\infty p_n x^n = \prod_{k=1}^\infty (1 - x^k)^{-1}

    Examples
    ========

    >>> from sympy import partition, Symbol
    >>> [partition(n) for n in range(9)]
    [1, 1, 2, 3, 5, 7, 11, 15, 22]
    >>> n = Symbol('n', integer=True, negative=True)
    >>> partition(n)
    0

    See Also
    ========

    bell, bernoulli, catalan, euler, fibonacci, harmonic, lucas, genocchi, tribonacci

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Partition_(number_theory%29
    .. [2] https://en.wikipedia.org/wiki/Pentagonal_number_theorem

    """
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, n):
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_negative is True:
            return S.Zero
        if n.is_zero is True or n is S.One:
            return S.One
        if n.is_Integer is True:
            return S(_partition(as_int(n)))

    def _eval_is_positive(self):
        if self.args[0].is_nonnegative is True:
            return True

    def _eval_Mod(self, q):
        # Ramanujan's congruences
        n = self.args[0]
        for p, rem in [(5, 4), (7, 5), (11, 6)]:
            if q == p and n % q == rem:
                return S.Zero


class divisor_sigma(DefinedFunction):
    r"""
    Calculate the divisor function `\sigma_k(n)` for positive integer n

    ``divisor_sigma(n, k)`` is equal to ``sum([x**k for x in divisors(n)])``

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^\omega p_i^{m_i},

    then

    .. math ::
        \sigma_k(n) = \prod_{i=1}^\omega (1+p_i^k+p_i^{2k}+\cdots
        + p_i^{m_ik}).

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import divisor_sigma
    >>> divisor_sigma(18, 0)
    6
    >>> divisor_sigma(39, 1)
    56
    >>> divisor_sigma(12, 2)
    210
    >>> divisor_sigma(37)
    38

    See Also
    ========

    sympy.ntheory.factor_.divisor_count, totient, sympy.ntheory.factor_.divisors, sympy.ntheory.factor_.factorint

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Divisor_function

    """
    is_integer = True
    is_positive = True

    @classmethod
    def eval(cls, n, k=S.One):
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        if k.is_integer is False:
            raise TypeError("k should be an integer")
        if k.is_nonnegative is False:
            raise ValueError("k should be a nonnegative integer")
        if n.is_prime is True:
            return 1 + n**k
        if n is S.One:
            return S.One
        if n.is_Integer is True:
            if k.is_zero is True:
                return Mul(*[e + 1 for e in factorint(n).values()])
            if k.is_Integer is True:
                return S(_divisor_sigma(as_int(n), as_int(k)))
            if k.is_zero is False:
                return Mul(*[cancel((p**(k*(e + 1)) - 1) / (p**k - 1)) for p, e in factorint(n).items()])


class udivisor_sigma(DefinedFunction):
    r"""
    Calculate the unitary divisor function `\sigma_k^*(n)` for positive integer n

    ``udivisor_sigma(n, k)`` is equal to ``sum([x**k for x in udivisors(n)])``

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^\omega p_i^{m_i},

    then

    .. math ::
        \sigma_k^*(n) = \prod_{i=1}^\omega (1+ p_i^{m_ik}).

    Parameters
    ==========

    k : power of divisors in the sum

        for k = 0, 1:
        ``udivisor_sigma(n, 0)`` is equal to ``udivisor_count(n)``
        ``udivisor_sigma(n, 1)`` is equal to ``sum(udivisors(n))``

        Default for k is 1.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import udivisor_sigma
    >>> udivisor_sigma(18, 0)
    4
    >>> udivisor_sigma(74, 1)
    114
    >>> udivisor_sigma(36, 3)
    47450
    >>> udivisor_sigma(111)
    152

    See Also
    ========

    sympy.ntheory.factor_.divisor_count, totient, sympy.ntheory.factor_.divisors,
    sympy.ntheory.factor_.udivisors, sympy.ntheory.factor_.udivisor_count, divisor_sigma,
    sympy.ntheory.factor_.factorint

    References
    ==========

    .. [1] https://mathworld.wolfram.com/UnitaryDivisorFunction.html

    """
    is_integer = True
    is_positive = True

    @classmethod
    def eval(cls, n, k=S.One):
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        if k.is_integer is False:
            raise TypeError("k should be an integer")
        if k.is_nonnegative is False:
            raise ValueError("k should be a nonnegative integer")
        if n.is_prime is True:
            return 1 + n**k
        if n.is_Integer:
            return Mul(*[1+p**(k*e) for p, e in factorint(n).items()])


class legendre_symbol(DefinedFunction):
    r"""
    Returns the Legendre symbol `(a / p)`.

    For an integer ``a`` and an odd prime ``p``, the Legendre symbol is
    defined as

    .. math ::
        \genfrac(){}{}{a}{p} = \begin{cases}
             0 & \text{if } p \text{ divides } a\\
             1 & \text{if } a \text{ is a quadratic residue modulo } p\\
            -1 & \text{if } a \text{ is a quadratic nonresidue modulo } p
        \end{cases}

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import legendre_symbol
    >>> [legendre_symbol(i, 7) for i in range(7)]
    [0, 1, 1, -1, 1, -1, -1]
    >>> sorted(set([i**2 % 7 for i in range(7)]))
    [0, 1, 2, 4]

    See Also
    ========

    sympy.ntheory.residue_ntheory.is_quad_residue, jacobi_symbol

    """
    is_integer = True
    is_prime = False

    @classmethod
    def eval(cls, a, p):
        if a.is_integer is False:
            raise TypeError("a should be an integer")
        if p.is_integer is False:
            raise TypeError("p should be an integer")
        if p.is_prime is False or p.is_odd is False:
            raise ValueError("p should be an odd prime integer")
        if (a % p).is_zero is True:
            return S.Zero
        if a is S.One:
            return S.One
        if a.is_Integer is True and p.is_Integer is True:
            return S(legendre(as_int(a), as_int(p)))


class jacobi_symbol(DefinedFunction):
    r"""
    Returns the Jacobi symbol `(m / n)`.

    For any integer ``m`` and any positive odd integer ``n`` the Jacobi symbol
    is defined as the product of the Legendre symbols corresponding to the
    prime factors of ``n``:

    .. math ::
        \genfrac(){}{}{m}{n} =
            \genfrac(){}{}{m}{p^{1}}^{\alpha_1}
            \genfrac(){}{}{m}{p^{2}}^{\alpha_2}
            ...
            \genfrac(){}{}{m}{p^{k}}^{\alpha_k}
            \text{ where } n =
                p_1^{\alpha_1}
                p_2^{\alpha_2}
                ...
                p_k^{\alpha_k}

    Like the Legendre symbol, if the Jacobi symbol `\genfrac(){}{}{m}{n} = -1`
    then ``m`` is a quadratic nonresidue modulo ``n``.

    But, unlike the Legendre symbol, if the Jacobi symbol
    `\genfrac(){}{}{m}{n} = 1` then ``m`` may or may not be a quadratic residue
    modulo ``n``.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import jacobi_symbol, legendre_symbol
    >>> from sympy import S
    >>> jacobi_symbol(45, 77)
    -1
    >>> jacobi_symbol(60, 121)
    1

    The relationship between the ``jacobi_symbol`` and ``legendre_symbol`` can
    be demonstrated as follows:

    >>> L = legendre_symbol
    >>> S(45).factors()
    {3: 2, 5: 1}
    >>> jacobi_symbol(7, 45) == L(7, 3)**2 * L(7, 5)**1
    True

    See Also
    ========

    sympy.ntheory.residue_ntheory.is_quad_residue, legendre_symbol

    """
    is_integer = True
    is_prime = False

    @classmethod
    def eval(cls, m, n):
        if m.is_integer is False:
            raise TypeError("m should be an integer")
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_positive is False or n.is_odd is False:
            raise ValueError("n should be an odd positive integer")
        if m is S.One or n is S.One:
            return S.One
        if (m % n).is_zero is True:
            return S.Zero
        if m.is_Integer is True and n.is_Integer is True:
            return S(jacobi(as_int(m), as_int(n)))


class kronecker_symbol(DefinedFunction):
    r"""
    Returns the Kronecker symbol `(a / n)`.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import kronecker_symbol
    >>> kronecker_symbol(45, 77)
    -1
    >>> kronecker_symbol(13, -120)
    1

    See Also
    ========

    jacobi_symbol, legendre_symbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Kronecker_symbol

    """
    is_integer = True
    is_prime = False

    @classmethod
    def eval(cls, a, n):
        if a.is_integer is False:
            raise TypeError("a should be an integer")
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if a is S.One or n is S.One:
            return S.One
        if a.is_Integer is True and n.is_Integer is True:
            return S(kronecker(as_int(a), as_int(n)))


class mobius(DefinedFunction):
    """
    Mobius function maps natural number to {-1, 0, 1}

    It is defined as follows:
        1) `1` if `n = 1`.
        2) `0` if `n` has a squared prime factor.
        3) `(-1)^k` if `n` is a square-free positive integer with `k`
           number of prime factors.

    It is an important multiplicative function in number theory
    and combinatorics.  It has applications in mathematical series,
    algebraic number theory and also physics (Fermion operator has very
    concrete realization with Mobius Function model).

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import mobius
    >>> mobius(13*7)
    1
    >>> mobius(1)
    1
    >>> mobius(13*7*5)
    -1
    >>> mobius(13**2)
    0

    Even in the case of a symbol, if it clearly contains a squared prime factor, it will be zero.

    >>> from sympy import Symbol
    >>> n = Symbol("n", integer=True, positive=True)
    >>> mobius(4*n)
    0
    >>> mobius(n**2)
    0

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/M%C3%B6bius_function
    .. [2] Thomas Koshy "Elementary Number Theory with Applications"
    .. [3] https://oeis.org/A008683

    """
    is_integer = True
    is_prime = False

    @classmethod
    def eval(cls, n):
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        if n.is_prime is True:
            return S.NegativeOne
        if n is S.One:
            return S.One
        result = None
        for m, e in (_.as_base_exp() for _ in Mul.make_args(n)):
            if m.is_integer is True and m.is_positive is True and \
               e.is_integer is True and e.is_positive is True:
                lt = is_lt(S.One, e) # 1 < e
                if lt is True:
                    result = S.Zero
                elif m.is_Integer is True:
                    factors = factorint(m)
                    if any(v > 1 for v in factors.values()):
                        result = S.Zero
                    elif lt is False:
                        s = S.NegativeOne if len(factors) % 2 else S.One
                        if result is None:
                            result = s
                        else:
                            result *= s
            else:
                return
        return result


class primenu(DefinedFunction):
    r"""
    Calculate the number of distinct prime factors for a positive integer n.

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^k p_i^{m_i},

    then ``primenu(n)`` or `\nu(n)` is:

    .. math ::
        \nu(n) = k.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import primenu
    >>> primenu(1)
    0
    >>> primenu(30)
    3

    See Also
    ========

    sympy.ntheory.factor_.factorint

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PrimeFactor.html
    .. [2] https://oeis.org/A001221

    """
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, n):
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        if n.is_prime is True:
            return S.One
        if n is S.One:
            return S.Zero
        if n.is_Integer is True:
            return S(len(factorint(n)))


class primeomega(DefinedFunction):
    r"""
    Calculate the number of prime factors counting multiplicities for a
    positive integer n.

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^k p_i^{m_i},

    then ``primeomega(n)``  or `\Omega(n)` is:

    .. math ::
        \Omega(n) = \sum_{i=1}^k m_i.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import primeomega
    >>> primeomega(1)
    0
    >>> primeomega(20)
    3

    See Also
    ========

    sympy.ntheory.factor_.factorint

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PrimeFactor.html
    .. [2] https://oeis.org/A001222

    """
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, n):
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        if n.is_prime is True:
            return S.One
        if n is S.One:
            return S.Zero
        if n.is_Integer is True:
            return S(sum(factorint(n).values()))


class totient(DefinedFunction):
    r"""
    Calculate the Euler totient function phi(n)

    ``totient(n)`` or `\phi(n)` is the number of positive integers `\leq` n
    that are relatively prime to n.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import totient
    >>> totient(1)
    1
    >>> totient(25)
    20
    >>> totient(45) == totient(5)*totient(9)
    True

    See Also
    ========

    sympy.ntheory.factor_.divisor_count

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler%27s_totient_function
    .. [2] https://mathworld.wolfram.com/TotientFunction.html
    .. [3] https://oeis.org/A000010

    """
    is_integer = True
    is_positive = True

    @classmethod
    def eval(cls, n):
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        if n is S.One:
            return S.One
        if n.is_prime is True:
            return n - 1
        if isinstance(n, Dict):
            return S(prod(p**(k-1)*(p-1) for p, k in n.items()))
        if n.is_Integer is True:
            return S(prod(p**(k-1)*(p-1) for p, k in factorint(n).items()))


class reduced_totient(DefinedFunction):
    r"""
    Calculate the Carmichael reduced totient function lambda(n)

    ``reduced_totient(n)`` or `\lambda(n)` is the smallest m > 0 such that
    `k^m \equiv 1 \mod n` for all k relatively prime to n.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import reduced_totient
    >>> reduced_totient(1)
    1
    >>> reduced_totient(8)
    2
    >>> reduced_totient(30)
    4

    See Also
    ========

    totient

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Carmichael_function
    .. [2] https://mathworld.wolfram.com/CarmichaelFunction.html
    .. [3] https://oeis.org/A002322

    """
    is_integer = True
    is_positive = True

    @classmethod
    def eval(cls, n):
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        if n is S.One:
            return S.One
        if n.is_prime is True:
            return n - 1
        if isinstance(n, Dict):
            t = 1
            if 2 in n:
                t = (1 << (n[2] - 2)) if 2 < n[2] else n[2]
            return S(lcm(int(t), *(int(p-1)*int(p)**int(k-1) for p, k in n.items() if p != 2)))
        if n.is_Integer is True:
            n, t = remove(int(n), 2)
            if not t:
                t = 1
            elif 2 < t:
                t = 1 << (t - 2)
            return S(lcm(t, *((p-1)*p**(k-1) for p, k in factorint(n).items())))


class primepi(DefinedFunction):
    r""" Represents the prime counting function pi(n) = the number
    of prime numbers less than or equal to n.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import primepi
    >>> from sympy import prime, prevprime, isprime
    >>> primepi(25)
    9

    So there are 9 primes less than or equal to 25. Is 25 prime?

    >>> isprime(25)
    False

    It is not. So the first prime less than 25 must be the
    9th prime:

    >>> prevprime(25) == prime(9)
    True

    See Also
    ========

    sympy.ntheory.primetest.isprime : Test if n is prime
    sympy.ntheory.generate.primerange : Generate all primes in a given range
    sympy.ntheory.generate.prime : Return the nth prime

    References
    ==========

    .. [1] https://oeis.org/A000720

    """
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, n):
        if n is S.Infinity:
            return S.Infinity
        if n is S.NegativeInfinity:
            return S.Zero
        if n.is_real is False:
            raise TypeError("n should be a real")
        if is_lt(n, S(2)) is True:
            return S.Zero
        try:
            n = int(n)
        except TypeError:
            return
        return S(_primepi(n))


#######################################################################
###
### Functions for enumerating partitions, permutations and combinations
###
#######################################################################


class _MultisetHistogram(tuple):
    __slots__ = ()


_N = -1
_ITEMS = -2
_M = slice(None, _ITEMS)


def _multiset_histogram(n):
    """Return tuple used in permutation and combination counting. Input
    is a dictionary giving items with counts as values or a sequence of
    items (which need not be sorted).

    The data is stored in a class deriving from tuple so it is easily
    recognized and so it can be converted easily to a list.
    """
    if isinstance(n, dict):  # item: count
        if not all(isinstance(v, int) and v >= 0 for v in n.values()):
            raise ValueError
        tot = sum(n.values())
        items = sum(1 for k in n if n[k] > 0)
        return _MultisetHistogram([n[k] for k in n if n[k] > 0] + [items, tot])
    else:
        n = list(n)
        s = set(n)
        lens = len(s)
        lenn = len(n)
        if lens == lenn:
            n = [1]*lenn + [lenn, lenn]
            return _MultisetHistogram(n)
        m = dict(zip(s, range(lens)))
        d = dict(zip(range(lens), (0,)*lens))
        for i in n:
            d[m[i]] += 1
        return _multiset_histogram(d)


def nP(n, k=None, replacement=False):
    """Return the number of permutations of ``n`` items taken ``k`` at a time.

    Possible values for ``n``:

        integer - set of length ``n``

        sequence - converted to a multiset internally

        multiset - {element: multiplicity}

    If ``k`` is None then the total of all permutations of length 0
    through the number of items represented by ``n`` will be returned.

    If ``replacement`` is True then a given item can appear more than once
    in the ``k`` items. (For example, for 'ab' permutations of 2 would
    include 'aa', 'ab', 'ba' and 'bb'.) The multiplicity of elements in
    ``n`` is ignored when ``replacement`` is True but the total number
    of elements is considered since no element can appear more times than
    the number of elements in ``n``.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import nP
    >>> from sympy.utilities.iterables import multiset_permutations, multiset
    >>> nP(3, 2)
    6
    >>> nP('abc', 2) == nP(multiset('abc'), 2) == 6
    True
    >>> nP('aab', 2)
    3
    >>> nP([1, 2, 2], 2)
    3
    >>> [nP(3, i) for i in range(4)]
    [1, 3, 6, 6]
    >>> nP(3) == sum(_)
    True

    When ``replacement`` is True, each item can have multiplicity
    equal to the length represented by ``n``:

    >>> nP('aabc', replacement=True)
    121
    >>> [len(list(multiset_permutations('aaaabbbbcccc', i))) for i in range(5)]
    [1, 3, 9, 27, 81]
    >>> sum(_)
    121

    See Also
    ========
    sympy.utilities.iterables.multiset_permutations

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Permutation

    """
    try:
        n = as_int(n)
    except ValueError:
        return Integer(_nP(_multiset_histogram(n), k, replacement))
    return Integer(_nP(n, k, replacement))


@cacheit
def _nP(n, k=None, replacement=False):

    if k == 0:
        return 1
    if isinstance(n, SYMPY_INTS):  # n different items
        # assert n >= 0
        if k is None:
            return sum(_nP(n, i, replacement) for i in range(n + 1))
        elif replacement:
            return n**k
        elif k > n:
            return 0
        elif k == n:
            return factorial(k)
        elif k == 1:
            return n
        else:
            # assert k >= 0
            return _product(n - k + 1, n)
    elif isinstance(n, _MultisetHistogram):
        if k is None:
            return sum(_nP(n, i, replacement) for i in range(n[_N] + 1))
        elif replacement:
            return n[_ITEMS]**k
        elif k == n[_N]:
            return factorial(k)/prod([factorial(i) for i in n[_M] if i > 1])
        elif k > n[_N]:
            return 0
        elif k == 1:
            return n[_ITEMS]
        else:
            # assert k >= 0
            tot = 0
            n = list(n)
            for i in range(len(n[_M])):
                if not n[i]:
                    continue
                n[_N] -= 1
                if n[i] == 1:
                    n[i] = 0
                    n[_ITEMS] -= 1
                    tot += _nP(_MultisetHistogram(n), k - 1)
                    n[_ITEMS] += 1
                    n[i] = 1
                else:
                    n[i] -= 1
                    tot += _nP(_MultisetHistogram(n), k - 1)
                    n[i] += 1
                n[_N] += 1
            return tot


@cacheit
def _AOP_product(n):
    """for n = (m1, m2, .., mk) return the coefficients of the polynomial,
    prod(sum(x**i for i in range(nj + 1)) for nj in n); i.e. the coefficients
    of the product of AOPs (all-one polynomials) or order given in n.  The
    resulting coefficient corresponding to x**r is the number of r-length
    combinations of sum(n) elements with multiplicities given in n.
    The coefficients are given as a default dictionary (so if a query is made
    for a key that is not present, 0 will be returned).

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import _AOP_product
    >>> from sympy.abc import x
    >>> n = (2, 2, 3)  # e.g. aabbccc
    >>> prod = ((x**2 + x + 1)*(x**2 + x + 1)*(x**3 + x**2 + x + 1)).expand()
    >>> c = _AOP_product(n); dict(c)
    {0: 1, 1: 3, 2: 6, 3: 8, 4: 8, 5: 6, 6: 3, 7: 1}
    >>> [c[i] for i in range(8)] == [prod.coeff(x, i) for i in range(8)]
    True

    The generating poly used here is the same as that listed in
    https://tinyurl.com/cep849r, but in a refactored form.

    """

    n = list(n)
    ord = sum(n)
    need = (ord + 2)//2
    rv = [1]*(n.pop() + 1)
    rv.extend((0,) * (need - len(rv)))
    rv = rv[:need]
    while n:
        ni = n.pop()
        N = ni + 1
        was = rv[:]
        for i in range(1, min(N, len(rv))):
            rv[i] += rv[i - 1]
        for i in range(N, need):
            rv[i] += rv[i - 1] - was[i - N]
    rev = list(reversed(rv))
    if ord % 2:
        rv = rv + rev
    else:
        rv[-1:] = rev
    d = defaultdict(int)
    for i, r in enumerate(rv):
        d[i] = r
    return d


def nC(n, k=None, replacement=False):
    """Return the number of combinations of ``n`` items taken ``k`` at a time.

    Possible values for ``n``:

        integer - set of length ``n``

        sequence - converted to a multiset internally

        multiset - {element: multiplicity}

    If ``k`` is None then the total of all combinations of length 0
    through the number of items represented in ``n`` will be returned.

    If ``replacement`` is True then a given item can appear more than once
    in the ``k`` items. (For example, for 'ab' sets of 2 would include 'aa',
    'ab', and 'bb'.) The multiplicity of elements in ``n`` is ignored when
    ``replacement`` is True but the total number of elements is considered
    since no element can appear more times than the number of elements in
    ``n``.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import nC
    >>> from sympy.utilities.iterables import multiset_combinations
    >>> nC(3, 2)
    3
    >>> nC('abc', 2)
    3
    >>> nC('aab', 2)
    2

    When ``replacement`` is True, each item can have multiplicity
    equal to the length represented by ``n``:

    >>> nC('aabc', replacement=True)
    35
    >>> [len(list(multiset_combinations('aaaabbbbcccc', i))) for i in range(5)]
    [1, 3, 6, 10, 15]
    >>> sum(_)
    35

    If there are ``k`` items with multiplicities ``m_1, m_2, ..., m_k``
    then the total of all combinations of length 0 through ``k`` is the
    product, ``(m_1 + 1)*(m_2 + 1)*...*(m_k + 1)``. When the multiplicity
    of each item is 1 (i.e., k unique items) then there are 2**k
    combinations. For example, if there are 4 unique items, the total number
    of combinations is 16:

    >>> sum(nC(4, i) for i in range(5))
    16

    See Also
    ========

    sympy.utilities.iterables.multiset_combinations

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Combination
    .. [2] https://tinyurl.com/cep849r

    """

    if isinstance(n, SYMPY_INTS):
        if k is None:
            if not replacement:
                return 2**n
            return sum(nC(n, i, replacement) for i in range(n + 1))
        if k < 0:
            raise ValueError("k cannot be negative")
        if replacement:
            return binomial(n + k - 1, k)
        return binomial(n, k)
    if isinstance(n, _MultisetHistogram):
        N = n[_N]
        if k is None:
            if not replacement:
                return prod(m + 1 for m in n[_M])
            return sum(nC(n, i, replacement) for i in range(N + 1))
        elif replacement:
            return nC(n[_ITEMS], k, replacement)
        # assert k >= 0
        elif k in (1, N - 1):
            return n[_ITEMS]
        elif k in (0, N):
            return 1
        return _AOP_product(tuple(n[_M]))[k]
    else:
        return nC(_multiset_histogram(n), k, replacement)


def _eval_stirling1(n, k):
    if n == k == 0:
        return S.One
    if 0 in (n, k):
        return S.Zero

    # some special values
    if n == k:
        return S.One
    elif k == n - 1:
        return binomial(n, 2)
    elif k == n - 2:
        return (3*n - 1)*binomial(n, 3)/4
    elif k == n - 3:
        return binomial(n, 2)*binomial(n, 4)

    return _stirling1(n, k)


@cacheit
def _stirling1(n, k):
    row = [0, 1]+[0]*(k-1) # for n = 1
    for i in range(2, n+1):
        for j in range(min(k,i), 0, -1):
            row[j] = (i-1) * row[j] + row[j-1]
    return Integer(row[k])


def _eval_stirling2(n, k):
    if n == k == 0:
        return S.One
    if 0 in (n, k):
        return S.Zero

    # some special values
    if n == k:
        return S.One
    elif k == n - 1:
        return binomial(n, 2)
    elif k == 1:
        return S.One
    elif k == 2:
        return Integer(2**(n - 1) - 1)

    return _stirling2(n, k)


@cacheit
def _stirling2(n, k):
    row = [0, 1]+[0]*(k-1) # for n = 1
    for i in range(2, n+1):
        for j in range(min(k,i), 0, -1):
            row[j] = j * row[j] + row[j-1]
    return Integer(row[k])


def stirling(n, k, d=None, kind=2, signed=False):
    r"""Return Stirling number $S(n, k)$ of the first or second (default) kind.

    The sum of all Stirling numbers of the second kind for $k = 1$
    through $n$ is ``bell(n)``. The recurrence relationship for these numbers
    is:

    .. math :: {0 \brace 0} = 1; {n \brace 0} = {0 \brace k} = 0;

    .. math :: {{n+1} \brace k} = j {n \brace k} + {n \brace {k-1}}

    where $j$ is:
        $n$ for Stirling numbers of the first kind,
        $-n$ for signed Stirling numbers of the first kind,
        $k$ for Stirling numbers of the second kind.

    The first kind of Stirling number counts the number of permutations of
    ``n`` distinct items that have ``k`` cycles; the second kind counts the
    ways in which ``n`` distinct items can be partitioned into ``k`` parts.
    If ``d`` is given, the "reduced Stirling number of the second kind" is
    returned: $S^{d}(n, k) = S(n - d + 1, k - d + 1)$ with $n \ge k \ge d$.
    (This counts the ways to partition $n$ consecutive integers into $k$
    groups with no pairwise difference less than $d$. See example below.)

    To obtain the signed Stirling numbers of the first kind, use keyword
    ``signed=True``. Using this keyword automatically sets ``kind`` to 1.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import stirling, bell
    >>> from sympy.combinatorics import Permutation
    >>> from sympy.utilities.iterables import multiset_partitions, permutations

    First kind (unsigned by default):

    >>> [stirling(6, i, kind=1) for i in range(7)]
    [0, 120, 274, 225, 85, 15, 1]
    >>> perms = list(permutations(range(4)))
    >>> [sum(Permutation(p).cycles == i for p in perms) for i in range(5)]
    [0, 6, 11, 6, 1]
    >>> [stirling(4, i, kind=1) for i in range(5)]
    [0, 6, 11, 6, 1]

    First kind (signed):

    >>> [stirling(4, i, signed=True) for i in range(5)]
    [0, -6, 11, -6, 1]

    Second kind:

    >>> [stirling(10, i) for i in range(12)]
    [0, 1, 511, 9330, 34105, 42525, 22827, 5880, 750, 45, 1, 0]
    >>> sum(_) == bell(10)
    True
    >>> len(list(multiset_partitions(range(4), 2))) == stirling(4, 2)
    True

    Reduced second kind:

    >>> from sympy import subsets, oo
    >>> def delta(p):
    ...    if len(p) == 1:
    ...        return oo
    ...    return min(abs(i[0] - i[1]) for i in subsets(p, 2))
    >>> parts = multiset_partitions(range(5), 3)
    >>> d = 2
    >>> sum(1 for p in parts if all(delta(i) >= d for i in p))
    7
    >>> stirling(5, 3, 2)
    7

    See Also
    ========
    sympy.utilities.iterables.multiset_partitions


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Stirling_numbers_of_the_first_kind
    .. [2] https://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind

    """
    # TODO: make this a class like bell()

    n = as_int(n)
    k = as_int(k)
    if n < 0:
        raise ValueError('n must be nonnegative')
    if k > n:
        return S.Zero
    if d:
        # assert k >= d
        # kind is ignored -- only kind=2 is supported
        return _eval_stirling2(n - d + 1, k - d + 1)
    elif signed:
        # kind is ignored -- only kind=1 is supported
        return S.NegativeOne**(n - k)*_eval_stirling1(n, k)

    if kind == 1:
        return _eval_stirling1(n, k)
    elif kind == 2:
        return _eval_stirling2(n, k)
    else:
        raise ValueError('kind must be 1 or 2, not %s' % k)


@cacheit
def _nT(n, k):
    """Return the partitions of ``n`` items into ``k`` parts. This
    is used by ``nT`` for the case when ``n`` is an integer."""
    # really quick exits
    if k > n or k < 0:
        return 0
    if k in (1, n):
        return 1
    if k == 0:
        return 0
    # exits that could be done below but this is quicker
    if k == 2:
        return n//2
    d = n - k
    if d <= 3:
        return d
    # quick exit
    if 3*k >= n:  # or, equivalently, 2*k >= d
        # all the information needed in this case
        # will be in the cache needed to calculate
        # partition(d), so...
        # update cache
        tot = _partition_rec(d)
        # and correct for values not needed
        if d - k > 0:
            tot -= sum(_partition_rec.fetch_item(slice(d - k)))
        return tot
    # regular exit
    # nT(n, k) = Sum(nT(n - k, m), (m, 1, k));
    # calculate needed nT(i, j) values
    p = [1]*d
    for i in range(2, k + 1):
        for m  in range(i + 1, d):
            p[m] += p[m - i]
        d -= 1
    # if p[0] were appended to the end of p then the last
    # k values of p are the nT(n, j) values for 0 < j < k in reverse
    # order p[-1] = nT(n, 1), p[-2] = nT(n, 2), etc.... Instead of
    # putting the 1 from p[0] there, however, it is simply added to
    # the sum below which is valid for 1 < k <= n//2
    return (1 + sum(p[1 - k:]))


def nT(n, k=None):
    """Return the number of ``k``-sized partitions of ``n`` items.

    Possible values for ``n``:

        integer - ``n`` identical items

        sequence - converted to a multiset internally

        multiset - {element: multiplicity}

    Note: the convention for ``nT`` is different than that of ``nC`` and
    ``nP`` in that
    here an integer indicates ``n`` *identical* items instead of a set of
    length ``n``; this is in keeping with the ``partitions`` function which
    treats its integer-``n`` input like a list of ``n`` 1s. One can use
    ``range(n)`` for ``n`` to indicate ``n`` distinct items.

    If ``k`` is None then the total number of ways to partition the elements
    represented in ``n`` will be returned.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import nT

    Partitions of the given multiset:

    >>> [nT('aabbc', i) for i in range(1, 7)]
    [1, 8, 11, 5, 1, 0]
    >>> nT('aabbc') == sum(_)
    True

    >>> [nT("mississippi", i) for i in range(1, 12)]
    [1, 74, 609, 1521, 1768, 1224, 579, 197, 50, 9, 1]

    Partitions when all items are identical:

    >>> [nT(5, i) for i in range(1, 6)]
    [1, 2, 2, 1, 1]
    >>> nT('1'*5) == sum(_)
    True

    When all items are different:

    >>> [nT(range(5), i) for i in range(1, 6)]
    [1, 15, 25, 10, 1]
    >>> nT(range(5)) == sum(_)
    True

    Partitions of an integer expressed as a sum of positive integers:

    >>> from sympy import partition
    >>> partition(4)
    5
    >>> nT(4, 1) + nT(4, 2) + nT(4, 3) + nT(4, 4)
    5
    >>> nT('1'*4)
    5

    See Also
    ========
    sympy.utilities.iterables.partitions
    sympy.utilities.iterables.multiset_partitions
    sympy.functions.combinatorial.numbers.partition

    References
    ==========

    .. [1] https://web.archive.org/web/20210507012732/https://teaching.csse.uwa.edu.au/units/CITS7209/partition.pdf

    """

    if isinstance(n, SYMPY_INTS):
        # n identical items
        if k is None:
            return partition(n)
        if isinstance(k, SYMPY_INTS):
            n = as_int(n)
            k = as_int(k)
            return Integer(_nT(n, k))
    if not isinstance(n, _MultisetHistogram):
        try:
            # if n contains hashable items there is some
            # quick handling that can be done
            u = len(set(n))
            if u <= 1:
                return nT(len(n), k)
            elif u == len(n):
                n = range(u)
            raise TypeError
        except TypeError:
            n = _multiset_histogram(n)
    N = n[_N]
    if k is None and N == 1:
        return 1
    if k in (1, N):
        return 1
    if k == 2 or N == 2 and k is None:
        m, r = divmod(N, 2)
        rv = sum(nC(n, i) for i in range(1, m + 1))
        if not r:
            rv -= nC(n, m)//2
        if k is None:
            rv += 1  # for k == 1
        return rv
    if N == n[_ITEMS]:
        # all distinct
        if k is None:
            return bell(N)
        return stirling(N, k)
    m = MultisetPartitionTraverser()
    if k is None:
        return m.count_partitions(n[_M])
    # MultisetPartitionTraverser does not have a range-limited count
    # method, so need to enumerate and count
    tot = 0
    for discard in m.enum_range(n[_M], k-1, k):
        tot += 1
    return tot


#-----------------------------------------------------------------------------#
#                                                                             #
#                          Motzkin numbers                                    #
#                                                                             #
#-----------------------------------------------------------------------------#


class motzkin(DefinedFunction):
    """
    The nth Motzkin number is the number
    of ways of drawing non-intersecting chords
    between n points on a circle (not necessarily touching
    every point by a chord). The Motzkin numbers are named
    after Theodore Motzkin and have diverse applications
    in geometry, combinatorics and number theory.

    Motzkin numbers are the integer sequence defined by the
    initial terms `M_0 = 1`, `M_1 = 1` and the two-term recurrence relation
    `M_n = \frac{2*n + 1}{n + 2} * M_{n-1} + \frac{3n - 3}{n + 2} * M_{n-2}`.


    Examples
    ========

    >>> from sympy import motzkin

    >>> motzkin.is_motzkin(5)
    False
    >>> motzkin.find_motzkin_numbers_in_range(2,300)
    [2, 4, 9, 21, 51, 127]
    >>> motzkin.find_motzkin_numbers_in_range(2,900)
    [2, 4, 9, 21, 51, 127, 323, 835]
    >>> motzkin.find_first_n_motzkins(10)
    [1, 1, 2, 4, 9, 21, 51, 127, 323, 835]


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Motzkin_number
    .. [2] https://mathworld.wolfram.com/MotzkinNumber.html

    """

    @staticmethod
    def is_motzkin(n):
        try:
            n = as_int(n)
        except ValueError:
            return False
        if n > 0:
            if n in (1, 2):
                return True

            tn1 = 1
            tn = 2
            i = 3
            while tn < n:
                a = ((2*i + 1)*tn + (3*i - 3)*tn1)/(i + 2)
                i += 1
                tn1 = tn
                tn = a

            if tn == n:
                return True
            else:
                return False

        else:
            return False

    @staticmethod
    def find_motzkin_numbers_in_range(x, y):
        if 0 <= x <= y:
            motzkins = []
            if x <= 1 <= y:
                motzkins.append(1)
            tn1 = 1
            tn = 2
            i = 3
            while tn <= y:
                if tn >= x:
                    motzkins.append(tn)
                a = ((2*i + 1)*tn + (3*i - 3)*tn1)/(i + 2)
                i += 1
                tn1 = tn
                tn = int(a)

            return motzkins

        else:
            raise ValueError('The provided range is not valid. This condition should satisfy x <= y')

    @staticmethod
    def find_first_n_motzkins(n):
        try:
            n = as_int(n)
        except ValueError:
            raise ValueError('The provided number must be a positive integer')
        if n < 0:
            raise ValueError('The provided number must be a positive integer')
        motzkins = [1]
        if n >= 1:
            motzkins.append(1)
        tn1 = 1
        tn = 2
        i = 3
        while i <= n:
            motzkins.append(tn)
            a = ((2*i + 1)*tn + (3*i - 3)*tn1)/(i + 2)
            i += 1
            tn1 = tn
            tn = int(a)

        return motzkins

    @staticmethod
    @recurrence_memo([S.One, S.One])
    def _motzkin(n, prev):
        return ((2*n + 1)*prev[-1] + (3*n - 3)*prev[-2]) // (n + 2)

    @classmethod
    def eval(cls, n):
        try:
            n = as_int(n)
        except ValueError:
            raise ValueError('The provided number must be a positive integer')
        if n < 0:
            raise ValueError('The provided number must be a positive integer')
        return Integer(cls._motzkin(n - 1))


def nD(i=None, brute=None, *, n=None, m=None):
    """return the number of derangements for: ``n`` unique items, ``i``
    items (as a sequence or multiset), or multiplicities, ``m`` given
    as a sequence or multiset.

    Examples
    ========

    >>> from sympy.utilities.iterables import generate_derangements as enum
    >>> from sympy.functions.combinatorial.numbers import nD

    A derangement ``d`` of sequence ``s`` has all ``d[i] != s[i]``:

    >>> set([''.join(i) for i in enum('abc')])
    {'bca', 'cab'}
    >>> nD('abc')
    2

    Input as iterable or dictionary (multiset form) is accepted:

    >>> assert nD([1, 2, 2, 3, 3, 3]) == nD({1: 1, 2: 2, 3: 3})

    By default, a brute-force enumeration and count of multiset permutations
    is only done if there are fewer than 9 elements. There may be cases when
    there is high multiplicity with few unique elements that will benefit
    from a brute-force enumeration, too. For this reason, the `brute`
    keyword (default None) is provided. When False, the brute-force
    enumeration will never be used. When True, it will always be used.

    >>> nD('1111222233', brute=True)
    44

    For convenience, one may specify ``n`` distinct items using the
    ``n`` keyword:

    >>> assert nD(n=3) == nD('abc') == 2

    Since the number of derangments depends on the multiplicity of the
    elements and not the elements themselves, it may be more convenient
    to give a list or multiset of multiplicities using keyword ``m``:

    >>> assert nD('abc') == nD(m=(1,1,1)) == nD(m={1:3}) == 2

    """
    from sympy.integrals.integrals import integrate
    from sympy.functions.special.polynomials import laguerre
    from sympy.abc import x
    def ok(x):
        if not isinstance(x, SYMPY_INTS):
            raise TypeError('expecting integer values')
        if x < 0:
            raise ValueError('value must not be negative')
        return True

    if (i, n, m).count(None) != 2:
        raise ValueError('enter only 1 of i, n, or m')
    if i is not None:
        if isinstance(i, SYMPY_INTS):
            raise TypeError('items must be a list or dictionary')
        if not i:
            return S.Zero
        if type(i) is not dict:
            s = list(i)
            ms = multiset(s)
        elif type(i) is dict:
            all(ok(_) for _ in i.values())
            ms = {k: v for k, v in i.items() if v}
            s = None
        if not ms:
            return S.Zero
        N = sum(ms.values())
        counts = multiset(ms.values())
        nkey = len(ms)
    elif n is not None:
        ok(n)
        if not n:
            return S.Zero
        return subfactorial(n)
    elif m is not None:
        if isinstance(m, dict):
            all(ok(i) and ok(j) for i, j in m.items())
            counts = {k: v for k, v in m.items() if k*v}
        elif iterable(m) or isinstance(m, str):
            m = list(m)
            all(ok(i) for i in m)
            counts = multiset([i for i in m if i])
        else:
            raise TypeError('expecting iterable')
        if not counts:
            return S.Zero
        N = sum(k*v for k, v in counts.items())
        nkey = sum(counts.values())
        s = None
    big = int(max(counts))
    if big == 1:  # no repetition
        return subfactorial(nkey)
    nval = len(counts)
    if big*2 > N:
        return S.Zero
    if big*2 == N:
        if nkey == 2 and nval == 1:
            return S.One  # aaabbb
        if nkey - 1 == big:  # one element repeated
            return factorial(big)  # e.g. abc part of abcddd
    if N < 9 and brute is None or brute:
        # for all possibilities, this was found to be faster
        if s is None:
            s = []
            i = 0
            for m, v in counts.items():
                for j in range(v):
                    s.extend([i]*m)
                    i += 1
        return Integer(sum(1 for i in multiset_derangements(s)))
    from sympy.functions.elementary.exponential import exp
    return Integer(abs(integrate(exp(-x)*Mul(*[
        laguerre(i, x)**m for i, m in counts.items()]), (x, 0, oo))))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\polyclasses.py ===
"""OO layer for several polynomial representations. """

from __future__ import annotations

from sympy.external.gmpy import GROUND_TYPES

from sympy.utilities.exceptions import sympy_deprecation_warning

from sympy.core.numbers import oo
from sympy.core.sympify import CantSympify
from sympy.polys.polyutils import PicklableWithSlots, _sort_factors
from sympy.polys.domains import Domain, ZZ, QQ

from sympy.polys.polyerrors import (
    CoercionFailed,
    ExactQuotientFailed,
    DomainError,
    NotInvertible,
)

from sympy.polys.densebasic import (
    ninf,
    dmp_validate,
    dup_normal, dmp_normal,
    dup_convert, dmp_convert,
    dmp_from_sympy,
    dup_strip,
    dmp_degree_in,
    dmp_degree_list,
    dmp_negative_p,
    dmp_ground_LC,
    dmp_ground_TC,
    dmp_ground_nth,
    dmp_one, dmp_ground,
    dmp_zero, dmp_zero_p, dmp_one_p, dmp_ground_p,
    dup_from_dict, dmp_from_dict,
    dmp_to_dict,
    dmp_deflate,
    dmp_inject, dmp_eject,
    dmp_terms_gcd,
    dmp_list_terms, dmp_exclude,
    dup_slice, dmp_slice_in, dmp_permute,
    dmp_to_tuple,)

from sympy.polys.densearith import (
    dmp_add_ground,
    dmp_sub_ground,
    dmp_mul_ground,
    dmp_quo_ground,
    dmp_exquo_ground,
    dmp_abs,
    dmp_neg,
    dmp_add,
    dmp_sub,
    dmp_mul,
    dmp_sqr,
    dmp_pow,
    dmp_pdiv,
    dmp_prem,
    dmp_pquo,
    dmp_pexquo,
    dmp_div,
    dmp_rem,
    dmp_quo,
    dmp_exquo,
    dmp_add_mul, dmp_sub_mul,
    dmp_max_norm,
    dmp_l1_norm,
    dmp_l2_norm_squared)

from sympy.polys.densetools import (
    dmp_clear_denoms,
    dmp_integrate_in,
    dmp_diff_in,
    dmp_eval_in,
    dup_revert,
    dmp_ground_trunc,
    dmp_ground_content,
    dmp_ground_primitive,
    dmp_ground_monic,
    dmp_compose,
    dup_decompose,
    dup_shift,
    dmp_shift,
    dup_transform,
    dmp_lift)

from sympy.polys.euclidtools import (
    dup_half_gcdex, dup_gcdex, dup_invert,
    dmp_subresultants,
    dmp_resultant,
    dmp_discriminant,
    dmp_inner_gcd,
    dmp_gcd,
    dmp_lcm,
    dmp_cancel)

from sympy.polys.sqfreetools import (
    dup_gff_list,
    dmp_norm,
    dmp_sqf_p,
    dmp_sqf_norm,
    dmp_sqf_part,
    dmp_sqf_list, dmp_sqf_list_include)

from sympy.polys.factortools import (
    dup_cyclotomic_p, dmp_irreducible_p,
    dmp_factor_list, dmp_factor_list_include)

from sympy.polys.rootisolation import (
    dup_isolate_real_roots_sqf,
    dup_isolate_real_roots,
    dup_isolate_all_roots_sqf,
    dup_isolate_all_roots,
    dup_refine_real_root,
    dup_count_real_roots,
    dup_count_complex_roots,
    dup_sturm,
    dup_cauchy_upper_bound,
    dup_cauchy_lower_bound,
    dup_mignotte_sep_bound_squared)

from sympy.polys.polyerrors import (
    UnificationFailed,
    PolynomialError)


if GROUND_TYPES == 'flint':
    import flint
    def _supported_flint_domain(D):
        return D.is_ZZ or D.is_QQ or D.is_FF and D._is_flint
else:
    flint = None
    def _supported_flint_domain(D):
        return False


class DMP(CantSympify):
    """Dense Multivariate Polynomials over `K`. """

    __slots__ = ()

    lev: int
    dom: Domain

    def __new__(cls, rep, dom, lev=None):

        if lev is None:
            rep, lev = dmp_validate(rep)
        elif not isinstance(rep, list):
            raise CoercionFailed("expected list, got %s" % type(rep))

        return cls.new(rep, dom, lev)

    @classmethod
    def new(cls, rep, dom, lev):
        # It would be too slow to call _validate_args always at runtime.
        # Ideally this checking would be handled by a static type checker.
        #
        #cls._validate_args(rep, dom, lev)
        if flint is not None:
            if lev == 0 and _supported_flint_domain(dom):
                return DUP_Flint._new(rep, dom, lev)

        return DMP_Python._new(rep, dom, lev)

    @property
    def rep(f):
        """Get the representation of ``f``. """

        sympy_deprecation_warning("""
        Accessing the ``DMP.rep`` attribute is deprecated. The internal
        representation of ``DMP`` instances can now be ``DUP_Flint`` when the
        ground types are ``flint``. In this case the ``DMP`` instance does not
        have a ``rep`` attribute. Use ``DMP.to_list()`` instead. Using
        ``DMP.to_list()`` also works in previous versions of SymPy.
        """,
            deprecated_since_version="1.13",
            active_deprecations_target="dmp-rep",
        )

        return f.to_list()

    def to_best(f):
        """Convert to DUP_Flint if possible.

        This method should be used when the domain or level is changed and it
        potentially becomes possible to convert from DMP_Python to DUP_Flint.
        """
        if flint is not None:
            if isinstance(f, DMP_Python) and f.lev == 0 and _supported_flint_domain(f.dom):
                return DUP_Flint.new(f._rep, f.dom, f.lev)

        return f

    @classmethod
    def _validate_args(cls, rep, dom, lev):
        assert isinstance(dom, Domain)
        assert isinstance(lev, int) and lev >= 0

        def validate_rep(rep, lev):
            assert isinstance(rep, list)
            if lev == 0:
                assert all(dom.of_type(c) for c in rep)
            else:
                for r in rep:
                    validate_rep(r, lev - 1)

        validate_rep(rep, lev)

    @classmethod
    def from_dict(cls, rep, lev, dom):
        rep = dmp_from_dict(rep, lev, dom)
        return cls.new(rep, dom, lev)

    @classmethod
    def from_list(cls, rep, lev, dom):
        """Create an instance of ``cls`` given a list of native coefficients. """
        return cls.new(dmp_convert(rep, lev, None, dom), dom, lev)

    @classmethod
    def from_sympy_list(cls, rep, lev, dom):
        """Create an instance of ``cls`` given a list of SymPy coefficients. """
        return cls.new(dmp_from_sympy(rep, lev, dom), dom, lev)

    @classmethod
    def from_monoms_coeffs(cls, monoms, coeffs, lev, dom):
        return cls(dict(list(zip(monoms, coeffs))), dom, lev)

    def convert(f, dom):
        """Convert ``f`` to a ``DMP`` over the new domain. """
        if f.dom == dom:
            return f
        elif f.lev or flint is None:
            return f._convert(dom)
        elif isinstance(f, DUP_Flint):
            if _supported_flint_domain(dom):
                return f._convert(dom)
            else:
                return f.to_DMP_Python()._convert(dom)
        elif isinstance(f, DMP_Python):
            if _supported_flint_domain(dom):
                return f._convert(dom).to_DUP_Flint()
            else:
                return f._convert(dom)
        else:
            raise RuntimeError("unreachable code")

    def _convert(f, dom):
        raise NotImplementedError

    @classmethod
    def zero(cls, lev, dom):
        return DMP(dmp_zero(lev), dom, lev)

    @classmethod
    def one(cls, lev, dom):
        return DMP(dmp_one(lev, dom), dom, lev)

    def _one(f):
        raise NotImplementedError

    def __repr__(f):
        return "%s(%s, %s)" % (f.__class__.__name__, f.to_list(), f.dom)

    def __hash__(f):
        return hash((f.__class__.__name__, f.to_tuple(), f.lev, f.dom))

    def __getnewargs__(self):
        return self.to_list(), self.dom, self.lev

    def ground_new(f, coeff):
        """Construct a new ground instance of ``f``. """
        raise NotImplementedError

    def unify_DMP(f, g):
        """Unify and return ``DMP`` instances of ``f`` and ``g``. """
        if not isinstance(g, DMP) or f.lev != g.lev:
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        if f.dom != g.dom:
            dom = f.dom.unify(g.dom)
            f = f.convert(dom)
            g = g.convert(dom)

        return f, g

    def to_dict(f, zero=False):
        """Convert ``f`` to a dict representation with native coefficients. """
        return dmp_to_dict(f.to_list(), f.lev, f.dom, zero=zero)

    def to_sympy_dict(f, zero=False):
        """Convert ``f`` to a dict representation with SymPy coefficients. """
        rep = f.to_dict(zero=zero)

        for k, v in rep.items():
            rep[k] = f.dom.to_sympy(v)

        return rep

    def to_sympy_list(f):
        """Convert ``f`` to a list representation with SymPy coefficients. """
        def sympify_nested_list(rep):
            out = []
            for val in rep:
                if isinstance(val, list):
                    out.append(sympify_nested_list(val))
                else:
                    out.append(f.dom.to_sympy(val))
            return out

        return sympify_nested_list(f.to_list())

    def to_list(f):
        """Convert ``f`` to a list representation with native coefficients. """
        raise NotImplementedError

    def to_tuple(f):
        """
        Convert ``f`` to a tuple representation with native coefficients.

        This is needed for hashing.
        """
        raise NotImplementedError

    def to_ring(f):
        """Make the ground domain a ring. """
        return f.convert(f.dom.get_ring())

    def to_field(f):
        """Make the ground domain a field. """
        return f.convert(f.dom.get_field())

    def to_exact(f):
        """Make the ground domain exact. """
        return f.convert(f.dom.get_exact())

    def slice(f, m, n, j=0):
        """Take a continuous subsequence of terms of ``f``. """
        if not f.lev and not j:
            return f._slice(m, n)
        else:
            return f._slice_lev(m, n, j)

    def _slice(f, m, n):
        raise NotImplementedError

    def _slice_lev(f, m, n, j):
        raise NotImplementedError

    def coeffs(f, order=None):
        """Returns all non-zero coefficients from ``f`` in lex order. """
        return [ c for _, c in f.terms(order=order) ]

    def monoms(f, order=None):
        """Returns all non-zero monomials from ``f`` in lex order. """
        return [ m for m, _ in f.terms(order=order) ]

    def terms(f, order=None):
        """Returns all non-zero terms from ``f`` in lex order. """
        if f.is_zero:
            zero_monom = (0,)*(f.lev + 1)
            return [(zero_monom, f.dom.zero)]
        else:
            return f._terms(order=order)

    def _terms(f, order=None):
        raise NotImplementedError

    def all_coeffs(f):
        """Returns all coefficients from ``f``. """
        if f.lev:
            raise PolynomialError('multivariate polynomials not supported')

        if not f:
            return [f.dom.zero]
        else:
            return list(f.to_list())

    def all_monoms(f):
        """Returns all monomials from ``f``. """
        if f.lev:
            raise PolynomialError('multivariate polynomials not supported')

        n = f.degree()

        if n < 0:
            return [(0,)]
        else:
            return [ (n - i,) for i, c in enumerate(f.to_list()) ]

    def all_terms(f):
        """Returns all terms from a ``f``. """
        if f.lev:
            raise PolynomialError('multivariate polynomials not supported')

        n = f.degree()

        if n < 0:
            return [((0,), f.dom.zero)]
        else:
            return [ ((n - i,), c) for i, c in enumerate(f.to_list()) ]

    def lift(f):
        """Convert algebraic coefficients to rationals. """
        return f._lift().to_best()

    def _lift(f):
        raise NotImplementedError

    def deflate(f):
        """Reduce degree of `f` by mapping `x_i^m` to `y_i`. """
        raise NotImplementedError

    def inject(f, front=False):
        """Inject ground domain generators into ``f``. """
        raise NotImplementedError

    def eject(f, dom, front=False):
        """Eject selected generators into the ground domain. """
        raise NotImplementedError

    def exclude(f):
        r"""
        Remove useless generators from ``f``.

        Returns the removed generators and the new excluded ``f``.

        Examples
        ========

        >>> from sympy.polys.polyclasses import DMP
        >>> from sympy.polys.domains import ZZ

        >>> DMP([[[ZZ(1)]], [[ZZ(1)], [ZZ(2)]]], ZZ).exclude()
        ([2], DMP_Python([[1], [1, 2]], ZZ))

        """
        J, F = f._exclude()
        return J, F.to_best()

    def _exclude(f):
        raise NotImplementedError

    def permute(f, P):
        r"""
        Returns a polynomial in `K[x_{P(1)}, ..., x_{P(n)}]`.

        Examples
        ========

        >>> from sympy.polys.polyclasses import DMP
        >>> from sympy.polys.domains import ZZ

        >>> DMP([[[ZZ(2)], [ZZ(1), ZZ(0)]], [[]]], ZZ).permute([1, 0, 2])
        DMP_Python([[[2], []], [[1, 0], []]], ZZ)

        >>> DMP([[[ZZ(2)], [ZZ(1), ZZ(0)]], [[]]], ZZ).permute([1, 2, 0])
        DMP_Python([[[1], []], [[2, 0], []]], ZZ)

        """
        return f._permute(P)

    def _permute(f, P):
        raise NotImplementedError

    def terms_gcd(f):
        """Remove GCD of terms from the polynomial ``f``. """
        raise NotImplementedError

    def abs(f):
        """Make all coefficients in ``f`` positive. """
        raise NotImplementedError

    def neg(f):
        """Negate all coefficients in ``f``. """
        raise NotImplementedError

    def add_ground(f, c):
        """Add an element of the ground domain to ``f``. """
        return f._add_ground(f.dom.convert(c))

    def sub_ground(f, c):
        """Subtract an element of the ground domain from ``f``. """
        return f._sub_ground(f.dom.convert(c))

    def mul_ground(f, c):
        """Multiply ``f`` by a an element of the ground domain. """
        return f._mul_ground(f.dom.convert(c))

    def quo_ground(f, c):
        """Quotient of ``f`` by a an element of the ground domain. """
        return f._quo_ground(f.dom.convert(c))

    def exquo_ground(f, c):
        """Exact quotient of ``f`` by a an element of the ground domain. """
        return f._exquo_ground(f.dom.convert(c))

    def add(f, g):
        """Add two multivariate polynomials ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._add(G)

    def sub(f, g):
        """Subtract two multivariate polynomials ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._sub(G)

    def mul(f, g):
        """Multiply two multivariate polynomials ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._mul(G)

    def sqr(f):
        """Square a multivariate polynomial ``f``. """
        return f._sqr()

    def pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
        if not isinstance(n, int):
            raise TypeError("``int`` expected, got %s" % type(n))
        return f._pow(n)

    def pdiv(f, g):
        """Polynomial pseudo-division of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._pdiv(G)

    def prem(f, g):
        """Polynomial pseudo-remainder of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._prem(G)

    def pquo(f, g):
        """Polynomial pseudo-quotient of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._pquo(G)

    def pexquo(f, g):
        """Polynomial exact pseudo-quotient of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._pexquo(G)

    def div(f, g):
        """Polynomial division with remainder of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._div(G)

    def rem(f, g):
        """Computes polynomial remainder of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._rem(G)

    def quo(f, g):
        """Computes polynomial quotient of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._quo(G)

    def exquo(f, g):
        """Computes polynomial exact quotient of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._exquo(G)

    def _add_ground(f, c):
        raise NotImplementedError

    def _sub_ground(f, c):
        raise NotImplementedError

    def _mul_ground(f, c):
        raise NotImplementedError

    def _quo_ground(f, c):
        raise NotImplementedError

    def _exquo_ground(f, c):
        raise NotImplementedError

    def _add(f, g):
        raise NotImplementedError

    def _sub(f, g):
        raise NotImplementedError

    def _mul(f, g):
        raise NotImplementedError

    def _sqr(f):
        raise NotImplementedError

    def _pow(f, n):
        raise NotImplementedError

    def _pdiv(f, g):
        raise NotImplementedError

    def _prem(f, g):
        raise NotImplementedError

    def _pquo(f, g):
        raise NotImplementedError

    def _pexquo(f, g):
        raise NotImplementedError

    def _div(f, g):
        raise NotImplementedError

    def _rem(f, g):
        raise NotImplementedError

    def _quo(f, g):
        raise NotImplementedError

    def _exquo(f, g):
        raise NotImplementedError

    def degree(f, j=0):
        """Returns the leading degree of ``f`` in ``x_j``. """
        if not isinstance(j, int):
            raise TypeError("``int`` expected, got %s" % type(j))

        return f._degree(j)

    def _degree(f, j):
        raise NotImplementedError

    def degree_list(f):
        """Returns a list of degrees of ``f``. """
        raise NotImplementedError

    def total_degree(f):
        """Returns the total degree of ``f``. """
        raise NotImplementedError

    def homogenize(f, s):
        """Return homogeneous polynomial of ``f``"""
        td = f.total_degree()
        result = {}
        new_symbol = (s == len(f.terms()[0][0]))
        for term in f.terms():
            d = sum(term[0])
            if d < td:
                i = td - d
            else:
                i = 0
            if new_symbol:
                result[term[0] + (i,)] = term[1]
            else:
                l = list(term[0])
                l[s] += i
                result[tuple(l)] = term[1]
        return DMP.from_dict(result, f.lev + int(new_symbol), f.dom)

    def homogeneous_order(f):
        """Returns the homogeneous order of ``f``. """
        if f.is_zero:
            return -oo

        monoms = f.monoms()
        tdeg = sum(monoms[0])

        for monom in monoms:
            _tdeg = sum(monom)

            if _tdeg != tdeg:
                return None

        return tdeg

    def LC(f):
        """Returns the leading coefficient of ``f``. """
        raise NotImplementedError

    def TC(f):
        """Returns the trailing coefficient of ``f``. """
        raise NotImplementedError

    def nth(f, *N):
        """Returns the ``n``-th coefficient of ``f``. """
        if all(isinstance(n, int) for n in N):
            return f._nth(N)
        else:
            raise TypeError("a sequence of integers expected")

    def _nth(f, N):
        raise NotImplementedError

    def max_norm(f):
        """Returns maximum norm of ``f``. """
        raise NotImplementedError

    def l1_norm(f):
        """Returns l1 norm of ``f``. """
        raise NotImplementedError

    def l2_norm_squared(f):
        """Return squared l2 norm of ``f``. """
        raise NotImplementedError

    def clear_denoms(f):
        """Clear denominators, but keep the ground domain. """
        raise NotImplementedError

    def integrate(f, m=1, j=0):
        """Computes the ``m``-th order indefinite integral of ``f`` in ``x_j``. """
        if not isinstance(m, int):
            raise TypeError("``int`` expected, got %s" % type(m))

        if not isinstance(j, int):
            raise TypeError("``int`` expected, got %s" % type(j))

        return f._integrate(m, j)

    def _integrate(f, m, j):
        raise NotImplementedError

    def diff(f, m=1, j=0):
        """Computes the ``m``-th order derivative of ``f`` in ``x_j``. """
        if not isinstance(m, int):
            raise TypeError("``int`` expected, got %s" % type(m))

        if not isinstance(j, int):
            raise TypeError("``int`` expected, got %s" % type(j))

        return f._diff(m, j)

    def _diff(f, m, j):
        raise NotImplementedError

    def eval(f, a, j=0):
        """Evaluates ``f`` at the given point ``a`` in ``x_j``. """
        if not isinstance(j, int):
            raise TypeError("``int`` expected, got %s" % type(j))
        elif not (0 <= j <= f.lev):
            raise ValueError("invalid variable index %s" % j)

        if f.lev:
            return f._eval_lev(a, j)
        else:
            return f._eval(a)

    def _eval(f, a):
        raise NotImplementedError

    def _eval_lev(f, a, j):
        raise NotImplementedError

    def half_gcdex(f, g):
        """Half extended Euclidean algorithm, if univariate. """
        F, G = f.unify_DMP(g)

        if F.lev:
            raise ValueError('univariate polynomial expected')

        return F._half_gcdex(G)

    def _half_gcdex(f, g):
        raise NotImplementedError

    def gcdex(f, g):
        """Extended Euclidean algorithm, if univariate. """
        F, G = f.unify_DMP(g)

        if F.lev:
            raise ValueError('univariate polynomial expected')

        if not F.dom.is_Field:
            raise DomainError('ground domain must be a field')

        return F._gcdex(G)

    def _gcdex(f, g):
        raise NotImplementedError

    def invert(f, g):
        """Invert ``f`` modulo ``g``, if possible. """
        F, G = f.unify_DMP(g)

        if F.lev:
            raise ValueError('univariate polynomial expected')

        return F._invert(G)

    def _invert(f, g):
        raise NotImplementedError

    def revert(f, n):
        """Compute ``f**(-1)`` mod ``x**n``. """
        if f.lev:
            raise ValueError('univariate polynomial expected')

        return f._revert(n)

    def _revert(f, n):
        raise NotImplementedError

    def subresultants(f, g):
        """Computes subresultant PRS sequence of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._subresultants(G)

    def _subresultants(f, g):
        raise NotImplementedError

    def resultant(f, g, includePRS=False):
        """Computes resultant of ``f`` and ``g`` via PRS. """
        F, G = f.unify_DMP(g)
        if includePRS:
            return F._resultant_includePRS(G)
        else:
            return F._resultant(G)

    def _resultant(f, g, includePRS=False):
        raise NotImplementedError

    def discriminant(f):
        """Computes discriminant of ``f``. """
        raise NotImplementedError

    def cofactors(f, g):
        """Returns GCD of ``f`` and ``g`` and their cofactors. """
        F, G = f.unify_DMP(g)
        return F._cofactors(G)

    def _cofactors(f, g):
        raise NotImplementedError

    def gcd(f, g):
        """Returns polynomial GCD of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._gcd(G)

    def _gcd(f, g):
        raise NotImplementedError

    def lcm(f, g):
        """Returns polynomial LCM of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._lcm(G)

    def _lcm(f, g):
        raise NotImplementedError

    def cancel(f, g, include=True):
        """Cancel common factors in a rational function ``f/g``. """
        F, G = f.unify_DMP(g)

        if include:
            return F._cancel_include(G)
        else:
            return F._cancel(G)

    def _cancel(f, g):
        raise NotImplementedError

    def _cancel_include(f, g):
        raise NotImplementedError

    def trunc(f, p):
        """Reduce ``f`` modulo a constant ``p``. """
        return f._trunc(f.dom.convert(p))

    def _trunc(f, p):
        raise NotImplementedError

    def monic(f):
        """Divides all coefficients by ``LC(f)``. """
        raise NotImplementedError

    def content(f):
        """Returns GCD of polynomial coefficients. """
        raise NotImplementedError

    def primitive(f):
        """Returns content and a primitive form of ``f``. """
        raise NotImplementedError

    def compose(f, g):
        """Computes functional composition of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._compose(G)

    def _compose(f, g):
        raise NotImplementedError

    def decompose(f):
        """Computes functional decomposition of ``f``. """
        if f.lev:
            raise ValueError('univariate polynomial expected')

        return f._decompose()

    def _decompose(f):
        raise NotImplementedError

    def shift(f, a):
        """Efficiently compute Taylor shift ``f(x + a)``. """
        if f.lev:
            raise ValueError('univariate polynomial expected')

        return f._shift(f.dom.convert(a))

    def shift_list(f, a):
        """Efficiently compute Taylor shift ``f(X + A)``. """
        a = [f.dom.convert(ai) for ai in a]
        return f._shift_list(a)

    def _shift(f, a):
        raise NotImplementedError

    def transform(f, p, q):
        """Evaluate functional transformation ``q**n * f(p/q)``."""
        if f.lev:
            raise ValueError('univariate polynomial expected')

        P, Q = p.unify_DMP(q)
        F, P = f.unify_DMP(P)
        F, Q = F.unify_DMP(Q)

        return F._transform(P, Q)

    def _transform(f, p, q):
        raise NotImplementedError

    def sturm(f):
        """Computes the Sturm sequence of ``f``. """
        if f.lev:
            raise ValueError('univariate polynomial expected')

        return f._sturm()

    def _sturm(f):
        raise NotImplementedError

    def cauchy_upper_bound(f):
        """Computes the Cauchy upper bound on the roots of ``f``. """
        if f.lev:
            raise ValueError('univariate polynomial expected')

        return f._cauchy_upper_bound()

    def _cauchy_upper_bound(f):
        raise NotImplementedError

    def cauchy_lower_bound(f):
        """Computes the Cauchy lower bound on the nonzero roots of ``f``. """
        if f.lev:
            raise ValueError('univariate polynomial expected')

        return f._cauchy_lower_bound()

    def _cauchy_lower_bound(f):
        raise NotImplementedError

    def mignotte_sep_bound_squared(f):
        """Computes the squared Mignotte bound on root separations of ``f``. """
        if f.lev:
            raise ValueError('univariate polynomial expected')

        return f._mignotte_sep_bound_squared()

    def _mignotte_sep_bound_squared(f):
        raise NotImplementedError

    def gff_list(f):
        """Computes greatest factorial factorization of ``f``. """
        if f.lev:
            raise ValueError('univariate polynomial expected')

        return f._gff_list()

    def _gff_list(f):
        raise NotImplementedError

    def norm(f):
        """Computes ``Norm(f)``."""
        raise NotImplementedError

    def sqf_norm(f):
        """Computes square-free norm of ``f``. """
        raise NotImplementedError

    def sqf_part(f):
        """Computes square-free part of ``f``. """
        raise NotImplementedError

    def sqf_list(f, all=False):
        """Returns a list of square-free factors of ``f``. """
        raise NotImplementedError

    def sqf_list_include(f, all=False):
        """Returns a list of square-free factors of ``f``. """
        raise NotImplementedError

    def factor_list(f):
        """Returns a list of irreducible factors of ``f``. """
        raise NotImplementedError

    def factor_list_include(f):
        """Returns a list of irreducible factors of ``f``. """
        raise NotImplementedError

    def intervals(f, all=False, eps=None, inf=None, sup=None, fast=False, sqf=False):
        """Compute isolating intervals for roots of ``f``. """
        if f.lev:
            raise PolynomialError("Cannot isolate roots of a multivariate polynomial")

        if all and sqf:
            return f._isolate_all_roots_sqf(eps=eps, inf=inf, sup=sup, fast=fast)
        elif all and not sqf:
            return f._isolate_all_roots(eps=eps, inf=inf, sup=sup, fast=fast)
        elif not all and sqf:
            return f._isolate_real_roots_sqf(eps=eps, inf=inf, sup=sup, fast=fast)
        else:
            return f._isolate_real_roots(eps=eps, inf=inf, sup=sup, fast=fast)

    def _isolate_all_roots(f, eps, inf, sup, fast):
        raise NotImplementedError

    def _isolate_all_roots_sqf(f, eps, inf, sup, fast):
        raise NotImplementedError

    def _isolate_real_roots(f, eps, inf, sup, fast):
        raise NotImplementedError

    def _isolate_real_roots_sqf(f, eps, inf, sup, fast):
        raise NotImplementedError

    def refine_root(f, s, t, eps=None, steps=None, fast=False):
        """
        Refine an isolating interval to the given precision.

        ``eps`` should be a rational number.

        """
        if f.lev:
            raise PolynomialError(
                "Cannot refine a root of a multivariate polynomial")

        return f._refine_real_root(s, t, eps=eps, steps=steps, fast=fast)

    def _refine_real_root(f, s, t, eps, steps, fast):
        raise NotImplementedError

    def count_real_roots(f, inf=None, sup=None):
        """Return the number of real roots of ``f`` in ``[inf, sup]``. """
        raise NotImplementedError

    def count_complex_roots(f, inf=None, sup=None):
        """Return the number of complex roots of ``f`` in ``[inf, sup]``. """
        raise NotImplementedError

    @property
    def is_zero(f):
        """Returns ``True`` if ``f`` is a zero polynomial. """
        raise NotImplementedError

    @property
    def is_one(f):
        """Returns ``True`` if ``f`` is a unit polynomial. """
        raise NotImplementedError

    @property
    def is_ground(f):
        """Returns ``True`` if ``f`` is an element of the ground domain. """
        raise NotImplementedError

    @property
    def is_sqf(f):
        """Returns ``True`` if ``f`` is a square-free polynomial. """
        raise NotImplementedError

    @property
    def is_monic(f):
        """Returns ``True`` if the leading coefficient of ``f`` is one. """
        raise NotImplementedError

    @property
    def is_primitive(f):
        """Returns ``True`` if the GCD of the coefficients of ``f`` is one. """
        raise NotImplementedError

    @property
    def is_linear(f):
        """Returns ``True`` if ``f`` is linear in all its variables. """
        raise NotImplementedError

    @property
    def is_quadratic(f):
        """Returns ``True`` if ``f`` is quadratic in all its variables. """
        raise NotImplementedError

    @property
    def is_monomial(f):
        """Returns ``True`` if ``f`` is zero or has only one term. """
        raise NotImplementedError

    @property
    def is_homogeneous(f):
        """Returns ``True`` if ``f`` is a homogeneous polynomial. """
        raise NotImplementedError

    @property
    def is_irreducible(f):
        """Returns ``True`` if ``f`` has no factors over its domain. """
        raise NotImplementedError

    @property
    def is_cyclotomic(f):
        """Returns ``True`` if ``f`` is a cyclotomic polynomial. """
        raise NotImplementedError

    def __abs__(f):
        return f.abs()

    def __neg__(f):
        return f.neg()

    def __add__(f, g):
        if isinstance(g, DMP):
            return f.add(g)
        else:
            try:
                return f.add_ground(g)
            except CoercionFailed:
                return NotImplemented

    def __radd__(f, g):
        return f.__add__(g)

    def __sub__(f, g):
        if isinstance(g, DMP):
            return f.sub(g)
        else:
            try:
                return f.sub_ground(g)
            except CoercionFailed:
                return NotImplemented

    def __rsub__(f, g):
        return (-f).__add__(g)

    def __mul__(f, g):
        if isinstance(g, DMP):
            return f.mul(g)
        else:
            try:
                return f.mul_ground(g)
            except CoercionFailed:
                return NotImplemented

    def __rmul__(f, g):
        return f.__mul__(g)

    def __truediv__(f, g):
        if isinstance(g, DMP):
            return f.exquo(g)
        else:
            try:
                return f.mul_ground(g)
            except CoercionFailed:
                return NotImplemented

    def __rtruediv__(f, g):
        if isinstance(g, DMP):
            return g.exquo(f)
        else:
            try:
                return f._one().mul_ground(g).exquo(f)
            except CoercionFailed:
                return NotImplemented

    def __pow__(f, n):
        return f.pow(n)

    def __divmod__(f, g):
        return f.div(g)

    def __mod__(f, g):
        return f.rem(g)

    def __floordiv__(f, g):
        if isinstance(g, DMP):
            return f.quo(g)
        else:
            try:
                return f.quo_ground(g)
            except TypeError:
                return NotImplemented

    def __eq__(f, g):
        if f is g:
            return True
        if not isinstance(g, DMP):
            return NotImplemented
        try:
            F, G = f.unify_DMP(g)
        except UnificationFailed:
            return False
        else:
            return F._strict_eq(G)

    def _strict_eq(f, g):
        raise NotImplementedError

    def eq(f, g, strict=False):
        if not strict:
            return f == g
        else:
            return f._strict_eq(g)

    def ne(f, g, strict=False):
        return not f.eq(g, strict=strict)

    def __lt__(f, g):
        F, G = f.unify_DMP(g)
        return F.to_list() < G.to_list()

    def __le__(f, g):
        F, G = f.unify_DMP(g)
        return F.to_list() <= G.to_list()

    def __gt__(f, g):
        F, G = f.unify_DMP(g)
        return F.to_list() > G.to_list()

    def __ge__(f, g):
        F, G = f.unify_DMP(g)
        return F.to_list() >= G.to_list()

    def __bool__(f):
        return not f.is_zero


class DMP_Python(DMP):
    """Dense Multivariate Polynomials over `K`. """

    __slots__ = ('_rep', 'dom', 'lev')

    @classmethod
    def _new(cls, rep, dom, lev):
        obj = object.__new__(cls)
        obj._rep = rep
        obj.lev = lev
        obj.dom = dom
        return obj

    def _strict_eq(f, g):
        if type(f) != type(g):
            return False
        return f.lev == g.lev and f.dom == g.dom and f._rep == g._rep

    def per(f, rep):
        """Create a DMP out of the given representation. """
        return f._new(rep, f.dom, f.lev)

    def ground_new(f, coeff):
        """Construct a new ground instance of ``f``. """
        return f._new(dmp_ground(coeff, f.lev), f.dom, f.lev)

    def _one(f):
        return f.one(f.lev, f.dom)

    def unify(f, g):
        """Unify representations of two multivariate polynomials. """
        # XXX: This function is not really used any more since there is
        # unify_DMP now.
        if not isinstance(g, DMP) or f.lev != g.lev:
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        if f.dom == g.dom:
            return f.lev, f.dom, f.per, f._rep, g._rep
        else:
            lev, dom = f.lev, f.dom.unify(g.dom)

            F = dmp_convert(f._rep, lev, f.dom, dom)
            G = dmp_convert(g._rep, lev, g.dom, dom)

            def per(rep):
                return f._new(rep, dom, lev)

            return lev, dom, per, F, G

    def to_DUP_Flint(f):
        """Convert ``f`` to a Flint representation. """
        return DUP_Flint._new(f._rep, f.dom, f.lev)

    def to_list(f):
        """Convert ``f`` to a list representation with native coefficients. """
        return list(f._rep)

    def to_tuple(f):
        """Convert ``f`` to a tuple representation with native coefficients. """
        return dmp_to_tuple(f._rep, f.lev)

    def _convert(f, dom):
        """Convert the ground domain of ``f``. """
        return f._new(dmp_convert(f._rep, f.lev, f.dom, dom), dom, f.lev)

    def _slice(f, m, n):
        """Take a continuous subsequence of terms of ``f``. """
        rep = dup_slice(f._rep, m, n, f.dom)
        return f._new(rep, f.dom, f.lev)

    def _slice_lev(f, m, n, j):
        """Take a continuous subsequence of terms of ``f``. """
        rep = dmp_slice_in(f._rep, m, n, j, f.lev, f.dom)
        return f._new(rep, f.dom, f.lev)

    def _terms(f, order=None):
        """Returns all non-zero terms from ``f`` in lex order. """
        return dmp_list_terms(f._rep, f.lev, f.dom, order=order)

    def _lift(f):
        """Convert algebraic coefficients to rationals. """
        r = dmp_lift(f._rep, f.lev, f.dom)
        return f._new(r, f.dom.dom, f.lev)

    def deflate(f):
        """Reduce degree of `f` by mapping `x_i^m` to `y_i`. """
        J, F = dmp_deflate(f._rep, f.lev, f.dom)
        return J, f.per(F)

    def inject(f, front=False):
        """Inject ground domain generators into ``f``. """
        F, lev = dmp_inject(f._rep, f.lev, f.dom, front=front)
        # XXX: domain and level changed here
        return f._new(F, f.dom.dom, lev)

    def eject(f, dom, front=False):
        """Eject selected generators into the ground domain. """
        F = dmp_eject(f._rep, f.lev, dom, front=front)
        # XXX: domain and level changed here
        return f._new(F, dom, f.lev - len(dom.symbols))

    def _exclude(f):
        """Remove useless generators from ``f``. """
        J, F, u = dmp_exclude(f._rep, f.lev, f.dom)
        # XXX: level changed here
        return J, f._new(F, f.dom, u)

    def _permute(f, P):
        """Returns a polynomial in `K[x_{P(1)}, ..., x_{P(n)}]`. """
        return f.per(dmp_permute(f._rep, P, f.lev, f.dom))

    def terms_gcd(f):
        """Remove GCD of terms from the polynomial ``f``. """
        J, F = dmp_terms_gcd(f._rep, f.lev, f.dom)
        return J, f.per(F)

    def _add_ground(f, c):
        """Add an element of the ground domain to ``f``. """
        return f.per(dmp_add_ground(f._rep, c, f.lev, f.dom))

    def _sub_ground(f, c):
        """Subtract an element of the ground domain from ``f``. """
        return f.per(dmp_sub_ground(f._rep, c, f.lev, f.dom))

    def _mul_ground(f, c):
        """Multiply ``f`` by a an element of the ground domain. """
        return f.per(dmp_mul_ground(f._rep, c, f.lev, f.dom))

    def _quo_ground(f, c):
        """Quotient of ``f`` by a an element of the ground domain. """
        return f.per(dmp_quo_ground(f._rep, c, f.lev, f.dom))

    def _exquo_ground(f, c):
        """Exact quotient of ``f`` by a an element of the ground domain. """
        return f.per(dmp_exquo_ground(f._rep, c, f.lev, f.dom))

    def abs(f):
        """Make all coefficients in ``f`` positive. """
        return f.per(dmp_abs(f._rep, f.lev, f.dom))

    def neg(f):
        """Negate all coefficients in ``f``. """
        return f.per(dmp_neg(f._rep, f.lev, f.dom))

    def _add(f, g):
        """Add two multivariate polynomials ``f`` and ``g``. """
        return f.per(dmp_add(f._rep, g._rep, f.lev, f.dom))

    def _sub(f, g):
        """Subtract two multivariate polynomials ``f`` and ``g``. """
        return f.per(dmp_sub(f._rep, g._rep, f.lev, f.dom))

    def _mul(f, g):
        """Multiply two multivariate polynomials ``f`` and ``g``. """
        return f.per(dmp_mul(f._rep, g._rep, f.lev, f.dom))

    def sqr(f):
        """Square a multivariate polynomial ``f``. """
        return f.per(dmp_sqr(f._rep, f.lev, f.dom))

    def _pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
        return f.per(dmp_pow(f._rep, n, f.lev, f.dom))

    def _pdiv(f, g):
        """Polynomial pseudo-division of ``f`` and ``g``. """
        q, r = dmp_pdiv(f._rep, g._rep, f.lev, f.dom)
        return f.per(q), f.per(r)

    def _prem(f, g):
        """Polynomial pseudo-remainder of ``f`` and ``g``. """
        return f.per(dmp_prem(f._rep, g._rep, f.lev, f.dom))

    def _pquo(f, g):
        """Polynomial pseudo-quotient of ``f`` and ``g``. """
        return f.per(dmp_pquo(f._rep, g._rep, f.lev, f.dom))

    def _pexquo(f, g):
        """Polynomial exact pseudo-quotient of ``f`` and ``g``. """
        return f.per(dmp_pexquo(f._rep, g._rep, f.lev, f.dom))

    def _div(f, g):
        """Polynomial division with remainder of ``f`` and ``g``. """
        q, r = dmp_div(f._rep, g._rep, f.lev, f.dom)
        return f.per(q), f.per(r)

    def _rem(f, g):
        """Computes polynomial remainder of ``f`` and ``g``. """
        return f.per(dmp_rem(f._rep, g._rep, f.lev, f.dom))

    def _quo(f, g):
        """Computes polynomial quotient of ``f`` and ``g``. """
        return f.per(dmp_quo(f._rep, g._rep, f.lev, f.dom))

    def _exquo(f, g):
        """Computes polynomial exact quotient of ``f`` and ``g``. """
        return f.per(dmp_exquo(f._rep, g._rep, f.lev, f.dom))

    def _degree(f, j=0):
        """Returns the leading degree of ``f`` in ``x_j``. """
        return dmp_degree_in(f._rep, j, f.lev)

    def degree_list(f):
        """Returns a list of degrees of ``f``. """
        return dmp_degree_list(f._rep, f.lev)

    def total_degree(f):
        """Returns the total degree of ``f``. """
        return max(sum(m) for m in f.monoms())

    def LC(f):
        """Returns the leading coefficient of ``f``. """
        return dmp_ground_LC(f._rep, f.lev, f.dom)

    def TC(f):
        """Returns the trailing coefficient of ``f``. """
        return dmp_ground_TC(f._rep, f.lev, f.dom)

    def _nth(f, N):
        """Returns the ``n``-th coefficient of ``f``. """
        return dmp_ground_nth(f._rep, N, f.lev, f.dom)

    def max_norm(f):
        """Returns maximum norm of ``f``. """
        return dmp_max_norm(f._rep, f.lev, f.dom)

    def l1_norm(f):
        """Returns l1 norm of ``f``. """
        return dmp_l1_norm(f._rep, f.lev, f.dom)

    def l2_norm_squared(f):
        """Return squared l2 norm of ``f``. """
        return dmp_l2_norm_squared(f._rep, f.lev, f.dom)

    def clear_denoms(f):
        """Clear denominators, but keep the ground domain. """
        coeff, F = dmp_clear_denoms(f._rep, f.lev, f.dom)
        return coeff, f.per(F)

    def _integrate(f, m=1, j=0):
        """Computes the ``m``-th order indefinite integral of ``f`` in ``x_j``. """
        return f.per(dmp_integrate_in(f._rep, m, j, f.lev, f.dom))

    def _diff(f, m=1, j=0):
        """Computes the ``m``-th order derivative of ``f`` in ``x_j``. """
        return f.per(dmp_diff_in(f._rep, m, j, f.lev, f.dom))

    def _eval(f, a):
        return dmp_eval_in(f._rep, f.dom.convert(a), 0, f.lev, f.dom)

    def _eval_lev(f, a, j):
        rep = dmp_eval_in(f._rep, f.dom.convert(a), j, f.lev, f.dom)
        return f.new(rep, f.dom, f.lev - 1)

    def _half_gcdex(f, g):
        """Half extended Euclidean algorithm, if univariate. """
        s, h = dup_half_gcdex(f._rep, g._rep, f.dom)
        return f.per(s), f.per(h)

    def _gcdex(f, g):
        """Extended Euclidean algorithm, if univariate. """
        s, t, h = dup_gcdex(f._rep, g._rep, f.dom)
        return f.per(s), f.per(t), f.per(h)

    def _invert(f, g):
        """Invert ``f`` modulo ``g``, if possible. """
        s = dup_invert(f._rep, g._rep, f.dom)
        return f.per(s)

    def _revert(f, n):
        """Compute ``f**(-1)`` mod ``x**n``. """
        return f.per(dup_revert(f._rep, n, f.dom))

    def _subresultants(f, g):
        """Computes subresultant PRS sequence of ``f`` and ``g``. """
        R = dmp_subresultants(f._rep, g._rep, f.lev, f.dom)
        return list(map(f.per, R))

    def _resultant_includePRS(f, g):
        """Computes resultant of ``f`` and ``g`` via PRS. """
        res, R = dmp_resultant(f._rep, g._rep, f.lev, f.dom, includePRS=True)
        if f.lev:
            res = f.new(res, f.dom, f.lev - 1)
        return res, list(map(f.per, R))

    def _resultant(f, g):
        res = dmp_resultant(f._rep, g._rep, f.lev, f.dom)
        if f.lev:
            res = f.new(res, f.dom, f.lev - 1)
        return res

    def discriminant(f):
        """Computes discriminant of ``f``. """
        res = dmp_discriminant(f._rep, f.lev, f.dom)
        if f.lev:
            res = f.new(res, f.dom, f.lev - 1)
        return res

    def _cofactors(f, g):
        """Returns GCD of ``f`` and ``g`` and their cofactors. """
        h, cff, cfg = dmp_inner_gcd(f._rep, g._rep, f.lev, f.dom)
        return f.per(h), f.per(cff), f.per(cfg)

    def _gcd(f, g):
        """Returns polynomial GCD of ``f`` and ``g``. """
        return f.per(dmp_gcd(f._rep, g._rep, f.lev, f.dom))

    def _lcm(f, g):
        """Returns polynomial LCM of ``f`` and ``g``. """
        return f.per(dmp_lcm(f._rep, g._rep, f.lev, f.dom))

    def _cancel(f, g):
        """Cancel common factors in a rational function ``f/g``. """
        cF, cG, F, G = dmp_cancel(f._rep, g._rep, f.lev, f.dom, include=False)
        return cF, cG, f.per(F), f.per(G)

    def _cancel_include(f, g):
        """Cancel common factors in a rational function ``f/g``. """
        F, G = dmp_cancel(f._rep, g._rep, f.lev, f.dom, include=True)
        return f.per(F), f.per(G)

    def _trunc(f, p):
        """Reduce ``f`` modulo a constant ``p``. """
        return f.per(dmp_ground_trunc(f._rep, p, f.lev, f.dom))

    def monic(f):
        """Divides all coefficients by ``LC(f)``. """
        return f.per(dmp_ground_monic(f._rep, f.lev, f.dom))

    def content(f):
        """Returns GCD of polynomial coefficients. """
        return dmp_ground_content(f._rep, f.lev, f.dom)

    def primitive(f):
        """Returns content and a primitive form of ``f``. """
        cont, F = dmp_ground_primitive(f._rep, f.lev, f.dom)
        return cont, f.per(F)

    def _compose(f, g):
        """Computes functional composition of ``f`` and ``g``. """
        return f.per(dmp_compose(f._rep, g._rep, f.lev, f.dom))

    def _decompose(f):
        """Computes functional decomposition of ``f``. """
        return list(map(f.per, dup_decompose(f._rep, f.dom)))

    def _shift(f, a):
        """Efficiently compute Taylor shift ``f(x + a)``. """
        return f.per(dup_shift(f._rep, a, f.dom))

    def _shift_list(f, a):
        """Efficiently compute Taylor shift ``f(X + A)``. """
        return f.per(dmp_shift(f._rep, a, f.lev, f.dom))

    def _transform(f, p, q):
        """Evaluate functional transformation ``q**n * f(p/q)``."""
        return f.per(dup_transform(f._rep, p._rep, q._rep, f.dom))

    def _sturm(f):
        """Computes the Sturm sequence of ``f``. """
        return list(map(f.per, dup_sturm(f._rep, f.dom)))

    def _cauchy_upper_bound(f):
        """Computes the Cauchy upper bound on the roots of ``f``. """
        return dup_cauchy_upper_bound(f._rep, f.dom)

    def _cauchy_lower_bound(f):
        """Computes the Cauchy lower bound on the nonzero roots of ``f``. """
        return dup_cauchy_lower_bound(f._rep, f.dom)

    def _mignotte_sep_bound_squared(f):
        """Computes the squared Mignotte bound on root separations of ``f``. """
        return dup_mignotte_sep_bound_squared(f._rep, f.dom)

    def _gff_list(f):
        """Computes greatest factorial factorization of ``f``. """
        return [ (f.per(g), k) for g, k in dup_gff_list(f._rep, f.dom) ]

    def norm(f):
        """Computes ``Norm(f)``."""
        r = dmp_norm(f._rep, f.lev, f.dom)
        return f.new(r, f.dom.dom, f.lev)

    def sqf_norm(f):
        """Computes square-free norm of ``f``. """
        s, g, r = dmp_sqf_norm(f._rep, f.lev, f.dom)
        return s, f.per(g), f.new(r, f.dom.dom, f.lev)

    def sqf_part(f):
        """Computes square-free part of ``f``. """
        return f.per(dmp_sqf_part(f._rep, f.lev, f.dom))

    def sqf_list(f, all=False):
        """Returns a list of square-free factors of ``f``. """
        coeff, factors = dmp_sqf_list(f._rep, f.lev, f.dom, all)
        return coeff, [ (f.per(g), k) for g, k in factors ]

    def sqf_list_include(f, all=False):
        """Returns a list of square-free factors of ``f``. """
        factors = dmp_sqf_list_include(f._rep, f.lev, f.dom, all)
        return [ (f.per(g), k) for g, k in factors ]

    def factor_list(f):
        """Returns a list of irreducible factors of ``f``. """
        coeff, factors = dmp_factor_list(f._rep, f.lev, f.dom)
        return coeff, [ (f.per(g), k) for g, k in factors ]

    def factor_list_include(f):
        """Returns a list of irreducible factors of ``f``. """
        factors = dmp_factor_list_include(f._rep, f.lev, f.dom)
        return [ (f.per(g), k) for g, k in factors ]

    def _isolate_real_roots(f, eps, inf, sup, fast):
        return dup_isolate_real_roots(f._rep, f.dom, eps=eps, inf=inf, sup=sup, fast=fast)

    def _isolate_real_roots_sqf(f, eps, inf, sup, fast):
        return dup_isolate_real_roots_sqf(f._rep, f.dom, eps=eps, inf=inf, sup=sup, fast=fast)

    def _isolate_all_roots(f, eps, inf, sup, fast):
        return dup_isolate_all_roots(f._rep, f.dom, eps=eps, inf=inf, sup=sup, fast=fast)

    def _isolate_all_roots_sqf(f, eps, inf, sup, fast):
        return dup_isolate_all_roots_sqf(f._rep, f.dom, eps=eps, inf=inf, sup=sup, fast=fast)

    def _refine_real_root(f, s, t, eps, steps, fast):
        return dup_refine_real_root(f._rep, s, t, f.dom, eps=eps, steps=steps, fast=fast)

    def count_real_roots(f, inf=None, sup=None):
        """Return the number of real roots of ``f`` in ``[inf, sup]``. """
        return dup_count_real_roots(f._rep, f.dom, inf=inf, sup=sup)

    def count_complex_roots(f, inf=None, sup=None):
        """Return the number of complex roots of ``f`` in ``[inf, sup]``. """
        return dup_count_complex_roots(f._rep, f.dom, inf=inf, sup=sup)

    @property
    def is_zero(f):
        """Returns ``True`` if ``f`` is a zero polynomial. """
        return dmp_zero_p(f._rep, f.lev)

    @property
    def is_one(f):
        """Returns ``True`` if ``f`` is a unit polynomial. """
        return dmp_one_p(f._rep, f.lev, f.dom)

    @property
    def is_ground(f):
        """Returns ``True`` if ``f`` is an element of the ground domain. """
        return dmp_ground_p(f._rep, None, f.lev)

    @property
    def is_sqf(f):
        """Returns ``True`` if ``f`` is a square-free polynomial. """
        return dmp_sqf_p(f._rep, f.lev, f.dom)

    @property
    def is_monic(f):
        """Returns ``True`` if the leading coefficient of ``f`` is one. """
        return f.dom.is_one(dmp_ground_LC(f._rep, f.lev, f.dom))

    @property
    def is_primitive(f):
        """Returns ``True`` if the GCD of the coefficients of ``f`` is one. """
        return f.dom.is_one(dmp_ground_content(f._rep, f.lev, f.dom))

    @property
    def is_linear(f):
        """Returns ``True`` if ``f`` is linear in all its variables. """
        return all(sum(monom) <= 1 for monom in dmp_to_dict(f._rep, f.lev, f.dom).keys())

    @property
    def is_quadratic(f):
        """Returns ``True`` if ``f`` is quadratic in all its variables. """
        return all(sum(monom) <= 2 for monom in dmp_to_dict(f._rep, f.lev, f.dom).keys())

    @property
    def is_monomial(f):
        """Returns ``True`` if ``f`` is zero or has only one term. """
        return len(f.to_dict()) <= 1

    @property
    def is_homogeneous(f):
        """Returns ``True`` if ``f`` is a homogeneous polynomial. """
        return f.homogeneous_order() is not None

    @property
    def is_irreducible(f):
        """Returns ``True`` if ``f`` has no factors over its domain. """
        return dmp_irreducible_p(f._rep, f.lev, f.dom)

    @property
    def is_cyclotomic(f):
        """Returns ``True`` if ``f`` is a cyclotomic polynomial. """
        if not f.lev:
            return dup_cyclotomic_p(f._rep, f.dom)
        else:
            return False


class DUP_Flint(DMP):
    """Dense Multivariate Polynomials over `K`. """

    lev = 0

    __slots__ = ('_rep', 'dom', '_cls')

    def __reduce__(self):
        return self.__class__, (self.to_list(), self.dom, self.lev)

    @classmethod
    def _new(cls, rep, dom, lev):
        rep = cls._flint_poly(rep[::-1], dom, lev)
        return cls.from_rep(rep, dom)

    def to_list(f):
        """Convert ``f`` to a list representation with native coefficients. """
        return f._rep.coeffs()[::-1]

    @classmethod
    def _flint_poly(cls, rep, dom, lev):
        assert _supported_flint_domain(dom)
        assert lev == 0
        flint_cls = cls._get_flint_poly_cls(dom)
        return flint_cls(rep)

    @classmethod
    def _get_flint_poly_cls(cls, dom):
        if dom.is_ZZ:
            return flint.fmpz_poly
        elif dom.is_QQ:
            return flint.fmpq_poly
        elif dom.is_FF:
            return dom._poly_ctx
        else:
            raise RuntimeError("Domain %s is not supported with flint" % dom)

    @classmethod
    def from_rep(cls, rep, dom):
        """Create a DMP from the given representation. """

        if dom.is_ZZ:
            assert isinstance(rep, flint.fmpz_poly)
            _cls = flint.fmpz_poly
        elif dom.is_QQ:
            assert isinstance(rep, flint.fmpq_poly)
            _cls = flint.fmpq_poly
        elif dom.is_FF:
            assert isinstance(rep, (flint.nmod_poly, flint.fmpz_mod_poly))
            c = dom.characteristic()
            __cls = type(rep)
            _cls = lambda e: __cls(e, c)
        else:
            raise RuntimeError("Domain %s is not supported with flint" % dom)

        obj = object.__new__(cls)
        obj.dom = dom
        obj._rep = rep
        obj._cls = _cls

        return obj

    def _strict_eq(f, g):
        if type(f) != type(g):
            return False
        return f.dom == g.dom and f._rep == g._rep

    def ground_new(f, coeff):
        """Construct a new ground instance of ``f``. """
        return f.from_rep(f._cls([coeff]), f.dom)

    def _one(f):
        return f.ground_new(f.dom.one)

    def unify(f, g):
        """Unify representations of two polynomials. """
        raise RuntimeError

    def to_DMP_Python(f):
        """Convert ``f`` to a Python native representation. """
        return DMP_Python._new(f.to_list(), f.dom, f.lev)

    def to_tuple(f):
        """Convert ``f`` to a tuple representation with native coefficients. """
        return tuple(f.to_list())

    def _convert(f, dom):
        """Convert the ground domain of ``f``. """
        if dom == QQ and f.dom == ZZ:
            return f.from_rep(flint.fmpq_poly(f._rep), dom)
        elif _supported_flint_domain(dom) and _supported_flint_domain(f.dom):
            # XXX: python-flint should provide a faster way to do this.
            return f.to_DMP_Python()._convert(dom).to_DUP_Flint()
        else:
            raise RuntimeError(f"DUP_Flint: Cannot convert {f.dom} to {dom}")

    def _slice(f, m, n):
        """Take a continuous subsequence of terms of ``f``. """
        coeffs = f._rep.coeffs()[m:n]
        return f.from_rep(f._cls(coeffs), f.dom)

    def _slice_lev(f, m, n, j):
        """Take a continuous subsequence of terms of ``f``. """
        # Only makes sense for multivariate polynomials
        raise NotImplementedError

    def _terms(f, order=None):
        """Returns all non-zero terms from ``f`` in lex order. """
        if order is None or order.alias == 'lex':
            terms = [ ((n,), c) for n, c in enumerate(f._rep.coeffs()) if c ]
            return terms[::-1]
        else:
            # XXX: InverseOrder (ilex) comes here. We could handle that case
            # efficiently by reversing the coefficients but it is not clear
            # how to test if the order is InverseOrder.
            #
            # Otherwise why would the order ever be different for univariate
            # polynomials?
            return f.to_DMP_Python()._terms(order=order)

    def _lift(f):
        """Convert algebraic coefficients to rationals. """
        # This is for algebraic number fields which DUP_Flint does not support
        raise NotImplementedError

    def deflate(f):
        """Reduce degree of `f` by mapping `x_i^m` to `y_i`. """
        # XXX: Check because otherwise this segfaults with python-flint:
        #
        #  >>> flint.fmpz_poly([]).deflation()
        #  Exception (fmpz_poly_deflate). Division by zero.
        #  Aborted (core dumped
        #
        if f.is_zero:
            return (1,), f
        g, n = f._rep.deflation()
        return (n,), f.from_rep(g, f.dom)

    def inject(f, front=False):
        """Inject ground domain generators into ``f``. """
        # Ground domain would need to be a poly ring
        raise NotImplementedError

    def eject(f, dom, front=False):
        """Eject selected generators into the ground domain. """
        # Only makes sense for multivariate polynomials
        raise NotImplementedError

    def _exclude(f):
        """Remove useless generators from ``f``. """
        # Only makes sense for multivariate polynomials
        raise NotImplementedError

    def _permute(f, P):
        """Returns a polynomial in `K[x_{P(1)}, ..., x_{P(n)}]`. """
        # Only makes sense for multivariate polynomials
        raise NotImplementedError

    def terms_gcd(f):
        """Remove GCD of terms from the polynomial ``f``. """
        # XXX: python-flint should have primitive, content, etc methods.
        J, F = f.to_DMP_Python().terms_gcd()
        return J, F.to_DUP_Flint()

    def _add_ground(f, c):
        """Add an element of the ground domain to ``f``. """
        return f.from_rep(f._rep + c, f.dom)

    def _sub_ground(f, c):
        """Subtract an element of the ground domain from ``f``. """
        return f.from_rep(f._rep - c, f.dom)

    def _mul_ground(f, c):
        """Multiply ``f`` by a an element of the ground domain. """
        return f.from_rep(f._rep * c, f.dom)

    def _quo_ground(f, c):
        """Quotient of ``f`` by a an element of the ground domain. """
        return f.from_rep(f._rep // c, f.dom)

    def _exquo_ground(f, c):
        """Exact quotient of ``f`` by an element of the ground domain. """
        q, r = divmod(f._rep, c)
        if r:
            raise ExactQuotientFailed(f, c)
        return f.from_rep(q, f.dom)

    def abs(f):
        """Make all coefficients in ``f`` positive. """
        return f.to_DMP_Python().abs().to_DUP_Flint()

    def neg(f):
        """Negate all coefficients in ``f``. """
        return f.from_rep(-f._rep, f.dom)

    def _add(f, g):
        """Add two multivariate polynomials ``f`` and ``g``. """
        return f.from_rep(f._rep + g._rep, f.dom)

    def _sub(f, g):
        """Subtract two multivariate polynomials ``f`` and ``g``. """
        return f.from_rep(f._rep - g._rep, f.dom)

    def _mul(f, g):
        """Multiply two multivariate polynomials ``f`` and ``g``. """
        return f.from_rep(f._rep * g._rep, f.dom)

    def sqr(f):
        """Square a multivariate polynomial ``f``. """
        return f.from_rep(f._rep ** 2, f.dom)

    def _pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
        return f.from_rep(f._rep ** n, f.dom)

    def _pdiv(f, g):
        """Polynomial pseudo-division of ``f`` and ``g``. """
        d = f.degree() - g.degree() + 1
        q, r = divmod(g.LC()**d * f._rep, g._rep)
        return f.from_rep(q, f.dom), f.from_rep(r, f.dom)

    def _prem(f, g):
        """Polynomial pseudo-remainder of ``f`` and ``g``. """
        d = f.degree() - g.degree() + 1
        q = (g.LC()**d * f._rep) % g._rep
        return f.from_rep(q, f.dom)

    def _pquo(f, g):
        """Polynomial pseudo-quotient of ``f`` and ``g``. """
        d = f.degree() - g.degree() + 1
        r = (g.LC()**d * f._rep) // g._rep
        return f.from_rep(r, f.dom)

    def _pexquo(f, g):
        """Polynomial exact pseudo-quotient of ``f`` and ``g``. """
        d = f.degree() - g.degree() + 1
        q, r = divmod(g.LC()**d * f._rep, g._rep)
        if r:
            raise ExactQuotientFailed(f, g)
        return f.from_rep(q, f.dom)

    def _div(f, g):
        """Polynomial division with remainder of ``f`` and ``g``. """
        if f.dom.is_Field:
            q, r = divmod(f._rep, g._rep)
            return f.from_rep(q, f.dom), f.from_rep(r, f.dom)
        else:
            # XXX: python-flint defines division in ZZ[x] differently
            q, r = f.to_DMP_Python()._div(g.to_DMP_Python())
            return q.to_DUP_Flint(), r.to_DUP_Flint()

    def _rem(f, g):
        """Computes polynomial remainder of ``f`` and ``g``. """
        return f.from_rep(f._rep % g._rep, f.dom)

    def _quo(f, g):
        """Computes polynomial quotient of ``f`` and ``g``. """
        return f.from_rep(f._rep // g._rep, f.dom)

    def _exquo(f, g):
        """Computes polynomial exact quotient of ``f`` and ``g``. """
        q, r = f._div(g)
        if r:
            raise ExactQuotientFailed(f, g)
        return q

    def _degree(f, j=0):
        """Returns the leading degree of ``f`` in ``x_j``. """
        d = f._rep.degree()
        if d == -1:
            d = ninf
        return d

    def degree_list(f):
        """Returns a list of degrees of ``f``. """
        return ( f._degree() ,)

    def total_degree(f):
        """Returns the total degree of ``f``. """
        return f._degree()

    def LC(f):
        """Returns the leading coefficient of ``f``. """
        return f._rep[f._rep.degree()]

    def TC(f):
        """Returns the trailing coefficient of ``f``. """
        return f._rep[0]

    def _nth(f, N):
        """Returns the ``n``-th coefficient of ``f``. """
        [n] = N
        return f._rep[n]

    def max_norm(f):
        """Returns maximum norm of ``f``. """
        return f.to_DMP_Python().max_norm()

    def l1_norm(f):
        """Returns l1 norm of ``f``. """
        return f.to_DMP_Python().l1_norm()

    def l2_norm_squared(f):
        """Return squared l2 norm of ``f``. """
        return f.to_DMP_Python().l2_norm_squared()

    def clear_denoms(f):
        """Clear denominators, but keep the ground domain. """
        R = f.dom
        if R.is_QQ:
            denom = f._rep.denom()
            numer = f.from_rep(f._cls(f._rep.numer()), f.dom)
            return denom, numer
        elif R.is_ZZ or R.is_FiniteField:
            return R.one, f
        else:
            raise NotImplementedError

    def _integrate(f, m=1, j=0):
        """Computes the ``m``-th order indefinite integral of ``f`` in ``x_j``. """
        assert j == 0
        if f.dom.is_Field:
            rep = f._rep
            for i in range(m):
                rep = rep.integral()
            return f.from_rep(rep, f.dom)
        else:
            return f.to_DMP_Python()._integrate(m=m, j=j).to_DUP_Flint()

    def _diff(f, m=1, j=0):
        """Computes the ``m``-th order derivative of ``f``. """
        assert j == 0
        rep = f._rep
        for i in range(m):
            rep = rep.derivative()
        return f.from_rep(rep, f.dom)

    def _eval(f, a):
        # XXX: This method is called with many different input types. Ideally
        # we could use e.g. fmpz_poly.__call__ here but more thought needs to
        # go into which types this is supposed to be called with and what types
        # it should return.
        return f.to_DMP_Python()._eval(a)

    def _eval_lev(f, a, j):
        # Only makes sense for multivariate polynomials
        raise NotImplementedError

    def _half_gcdex(f, g):
        """Half extended Euclidean algorithm. """
        s, h = f.to_DMP_Python()._half_gcdex(g.to_DMP_Python())
        return s.to_DUP_Flint(), h.to_DUP_Flint()

    def _gcdex(f, g):
        """Extended Euclidean algorithm. """
        h, s, t = f._rep.xgcd(g._rep)
        return f.from_rep(s, f.dom), f.from_rep(t, f.dom), f.from_rep(h, f.dom)

    def _invert(f, g):
        """Invert ``f`` modulo ``g``, if possible. """
        R = f.dom
        if R.is_Field:
            gcd, F_inv, _ = f._rep.xgcd(g._rep)
            # XXX: Should be gcd != 1 but nmod_poly does not compare equal to
            # other types.
            if gcd != 0*gcd + 1:
                raise NotInvertible("zero divisor")
            return f.from_rep(F_inv, R)
        else:
            # fmpz_poly does not have xgcd or invert and this is not well
            # defined in general.
            return f.to_DMP_Python()._invert(g.to_DMP_Python()).to_DUP_Flint()

    def _revert(f, n):
        """Compute ``f**(-1)`` mod ``x**n``. """
        # XXX: Use fmpz_series etc for reversion?
        # Maybe python-flint should provide revert for fmpz_poly...
        return f.to_DMP_Python()._revert(n).to_DUP_Flint()

    def _subresultants(f, g):
        """Computes subresultant PRS sequence of ``f`` and ``g``. """
        # XXX: Maybe _fmpz_poly_pseudo_rem_cohen could be used...
        R = f.to_DMP_Python()._subresultants(g.to_DMP_Python())
        return [ g.to_DUP_Flint() for g in R ]

    def _resultant_includePRS(f, g):
        """Computes resultant of ``f`` and ``g`` via PRS. """
        # XXX: Maybe _fmpz_poly_pseudo_rem_cohen could be used...
        res, R = f.to_DMP_Python()._resultant_includePRS(g.to_DMP_Python())
        return res, [ g.to_DUP_Flint() for g in R ]

    def _resultant(f, g):
        """Computes resultant of ``f`` and ``g``. """
        # XXX: Use fmpz_mpoly etc when possible...
        return f.to_DMP_Python()._resultant(g.to_DMP_Python())

    def discriminant(f):
        """Computes discriminant of ``f``. """
        # XXX: Use fmpz_mpoly etc when possible...
        return f.to_DMP_Python().discriminant()

    def _cofactors(f, g):
        """Returns GCD of ``f`` and ``g`` and their cofactors. """
        h = f.gcd(g)
        return h, f.exquo(h), g.exquo(h)

    def _gcd(f, g):
        """Returns polynomial GCD of ``f`` and ``g``. """
        return f.from_rep(f._rep.gcd(g._rep), f.dom)

    def _lcm(f, g):
        """Returns polynomial LCM of ``f`` and ``g``. """
        # XXX: python-flint should have a lcm method
        if not (f and g):
            return f.ground_new(f.dom.zero)

        l = f._mul(g)._exquo(f._gcd(g))

        if l.dom.is_Field:
            l = l.monic()
        elif l.LC() < 0:
            l = l.neg()

        return l

    def _cancel(f, g):
        """Cancel common factors in a rational function ``f/g``. """
        assert f.dom == g.dom
        R = f.dom

        # Think carefully about how to handle denominators and coefficient
        # canonicalisation if more domains are permitted...
        assert R.is_ZZ or R.is_QQ or R.is_FiniteField

        if R.is_FiniteField:
            h = f._gcd(g)
            F, G = f.exquo(h), g.exquo(h)
            return R.one, R.one, F, G

        if R.is_QQ:
            cG, F = f.clear_denoms()
            cF, G = g.clear_denoms()
        else:
            cG, F = R.one, f
            cF, G = R.one, g

        cH = cF.gcd(cG)
        cF, cG = cF // cH, cG // cH

        H = F._gcd(G)
        F, G = F.exquo(H), G.exquo(H)

        f_neg = F.LC() < 0
        g_neg = G.LC() < 0

        if f_neg and g_neg:
            F, G = F.neg(), G.neg()
        elif f_neg:
            cF, F = -cF, F.neg()
        elif g_neg:
            cF, G = -cF, G.neg()

        return cF, cG, F, G

    def _cancel_include(f, g):
        """Cancel common factors in a rational function ``f/g``. """
        cF, cG, F, G = f._cancel(g)
        return F._mul_ground(cF), G._mul_ground(cG)

    def _trunc(f, p):
        """Reduce ``f`` modulo a constant ``p``. """
        return f.to_DMP_Python()._trunc(p).to_DUP_Flint()

    def monic(f):
        """Divides all coefficients by ``LC(f)``. """
        # XXX: python-flint should add monic
        return f._exquo_ground(f.LC())

    def content(f):
        """Returns GCD of polynomial coefficients. """
        # XXX: python-flint should have a content method
        return f.to_DMP_Python().content()

    def primitive(f):
        """Returns content and a primitive form of ``f``. """
        cont = f.content()
        if f.is_zero:
            return f.dom.zero, f
        prim = f._exquo_ground(cont)
        return cont, prim

    def _compose(f, g):
        """Computes functional composition of ``f`` and ``g``. """
        return f.from_rep(f._rep(g._rep), f.dom)

    def _decompose(f):
        """Computes functional decomposition of ``f``. """
        return [ g.to_DUP_Flint() for g in f.to_DMP_Python()._decompose() ]

    def _shift(f, a):
        """Efficiently compute Taylor shift ``f(x + a)``. """
        x_plus_a = f._cls([a, f.dom.one])
        return f.from_rep(f._rep(x_plus_a), f.dom)

    def _transform(f, p, q):
        """Evaluate functional transformation ``q**n * f(p/q)``."""
        F, P, Q = f.to_DMP_Python(), p.to_DMP_Python(), q.to_DMP_Python()
        return F.transform(P, Q).to_DUP_Flint()

    def _sturm(f):
        """Computes the Sturm sequence of ``f``. """
        return [ g.to_DUP_Flint() for g in f.to_DMP_Python()._sturm() ]

    def _cauchy_upper_bound(f):
        """Computes the Cauchy upper bound on the roots of ``f``. """
        return f.to_DMP_Python()._cauchy_upper_bound()

    def _cauchy_lower_bound(f):
        """Computes the Cauchy lower bound on the nonzero roots of ``f``. """
        return f.to_DMP_Python()._cauchy_lower_bound()

    def _mignotte_sep_bound_squared(f):
        """Computes the squared Mignotte bound on root separations of ``f``. """
        return f.to_DMP_Python()._mignotte_sep_bound_squared()

    def _gff_list(f):
        """Computes greatest factorial factorization of ``f``. """
        F = f.to_DMP_Python()
        return [ (g.to_DUP_Flint(), k) for g, k in F.gff_list() ]

    def norm(f):
        """Computes ``Norm(f)``."""
        # This is for algebraic number fields which DUP_Flint does not support
        raise NotImplementedError

    def sqf_norm(f):
        """Computes square-free norm of ``f``. """
        # This is for algebraic number fields which DUP_Flint does not support
        raise NotImplementedError

    def sqf_part(f):
        """Computes square-free part of ``f``. """
        return f._exquo(f._gcd(f._diff()))

    def sqf_list(f, all=False):
        """Returns a list of square-free factors of ``f``. """
        # XXX: python-flint should provide square free factorisation.
        coeff, factors = f.to_DMP_Python().sqf_list(all=all)
        return coeff, [ (g.to_DUP_Flint(), k) for g, k in factors ]

    def sqf_list_include(f, all=False):
        """Returns a list of square-free factors of ``f``. """
        factors = f.to_DMP_Python().sqf_list_include(all=all)
        return [ (g.to_DUP_Flint(), k) for g, k in factors ]

    def factor_list(f):
        """Returns a list of irreducible factors of ``f``. """

        if f.dom.is_ZZ or f.dom.is_FF:
            # python-flint matches polys here
            coeff, factors = f._rep.factor()
            factors = [ (f.from_rep(g, f.dom), k) for g, k in factors ]

        elif f.dom.is_QQ:
            # python-flint returns monic factors over QQ whereas polys returns
            # denominator free factors.
            coeff, factors = f._rep.factor()
            factors_monic = [ (f.from_rep(g, f.dom), k) for g, k in factors ]

            # Absorb the denominators into coeff
            factors = []
            for g, k in factors_monic:
                d, g = g.clear_denoms()
                coeff /= d**k
                factors.append((g, k))

        else:
            # Check carefully when adding more domains here...
            raise RuntimeError("Domain %s is not supported with flint" % f.dom)

        # We need to match the way that polys orders the factors
        factors = f._sort_factors(factors)

        return coeff, factors

    def factor_list_include(f):
        """Returns a list of irreducible factors of ``f``. """
        # XXX: factor_list_include seems to be broken in general:
        #
        #   >>> Poly(2*(x - 1)**3, x).factor_list_include()
        #   [(Poly(2*x - 2, x, domain='ZZ'), 3)]
        #
        # Let's not try to implement it here.
        factors = f.to_DMP_Python().factor_list_include()
        return [ (g.to_DUP_Flint(), k) for g, k in factors ]

    def _sort_factors(f, factors):
        """Sort a list of factors to canonical order. """
        # Convert the factors to lists and use _sort_factors from polys
        factors = [ (g.to_list(), k) for g, k in factors ]
        factors = _sort_factors(factors, multiple=True)
        to_dup_flint = lambda g: f.from_rep(f._cls(g[::-1]), f.dom)
        return [ (to_dup_flint(g), k) for g, k in factors ]

    def _isolate_real_roots(f, eps, inf, sup, fast):
        return f.to_DMP_Python()._isolate_real_roots(eps, inf, sup, fast)

    def _isolate_real_roots_sqf(f, eps, inf, sup, fast):
        return f.to_DMP_Python()._isolate_real_roots_sqf(eps, inf, sup, fast)

    def _isolate_all_roots(f, eps, inf, sup, fast):
        # fmpz_poly and fmpq_poly have a complex_roots method that could be
        # used here. It probably makes more sense to add analogous methods in
        # python-flint though.
        return f.to_DMP_Python()._isolate_all_roots(eps, inf, sup, fast)

    def _isolate_all_roots_sqf(f, eps, inf, sup, fast):
        return f.to_DMP_Python()._isolate_all_roots_sqf(eps, inf, sup, fast)

    def _refine_real_root(f, s, t, eps, steps, fast):
        return f.to_DMP_Python()._refine_real_root(s, t, eps, steps, fast)

    def count_real_roots(f, inf=None, sup=None):
        """Return the number of real roots of ``f`` in ``[inf, sup]``. """
        return f.to_DMP_Python().count_real_roots(inf=inf, sup=sup)

    def count_complex_roots(f, inf=None, sup=None):
        """Return the number of complex roots of ``f`` in ``[inf, sup]``. """
        return f.to_DMP_Python().count_complex_roots(inf=inf, sup=sup)

    @property
    def is_zero(f):
        """Returns ``True`` if ``f`` is a zero polynomial. """
        return not f._rep

    @property
    def is_one(f):
        """Returns ``True`` if ``f`` is a unit polynomial. """
        return f._rep == f.dom.one

    @property
    def is_ground(f):
        """Returns ``True`` if ``f`` is an element of the ground domain. """
        return f._rep.degree() <= 0

    @property
    def is_linear(f):
        """Returns ``True`` if ``f`` is linear in all its variables. """
        return f._rep.degree() <= 1

    @property
    def is_quadratic(f):
        """Returns ``True`` if ``f`` is quadratic in all its variables. """
        return f._rep.degree() <= 2

    @property
    def is_monomial(f):
        """Returns ``True`` if ``f`` is zero or has only one term. """
        fr = f._rep
        return fr.degree() < 0 or not any(fr[n] for n in range(fr.degree()))

    @property
    def is_monic(f):
        """Returns ``True`` if the leading coefficient of ``f`` is one. """
        return f.LC() == f.dom.one

    @property
    def is_primitive(f):
        """Returns ``True`` if the GCD of the coefficients of ``f`` is one. """
        return f.to_DMP_Python().is_primitive

    @property
    def is_homogeneous(f):
        """Returns ``True`` if ``f`` is a homogeneous polynomial. """
        return f.to_DMP_Python().is_homogeneous

    @property
    def is_sqf(f):
        """Returns ``True`` if ``f`` is a square-free polynomial. """
        g = f._rep.gcd(f._rep.derivative())
        return g.degree() <= 0

    @property
    def is_irreducible(f):
        """Returns ``True`` if ``f`` has no factors over its domain. """
        _, factors = f._rep.factor()
        if len(factors) == 0:
            return True
        elif len(factors) == 1:
            return factors[0][1] == 1
        else:
            return False

    @property
    def is_cyclotomic(f):
        """Returns ``True`` if ``f`` is a cyclotomic polynomial. """
        if f.dom.is_QQ:
            try:
                f = f.convert(ZZ)
            except CoercionFailed:
                return False
        if f.dom.is_ZZ:
            return bool(f._rep.is_cyclotomic())
        else:
            # This is what dup_cyclotomic_p does...
            return False


def init_normal_DMF(num, den, lev, dom):
    return DMF(dmp_normal(num, lev, dom),
               dmp_normal(den, lev, dom), dom, lev)


class DMF(PicklableWithSlots, CantSympify):
    """Dense Multivariate Fractions over `K`. """

    __slots__ = ('num', 'den', 'lev', 'dom')

    def __init__(self, rep, dom, lev=None):
        num, den, lev = self._parse(rep, dom, lev)
        num, den = dmp_cancel(num, den, lev, dom)

        self.num = num
        self.den = den
        self.lev = lev
        self.dom = dom

    @classmethod
    def new(cls, rep, dom, lev=None):
        num, den, lev = cls._parse(rep, dom, lev)

        obj = object.__new__(cls)

        obj.num = num
        obj.den = den
        obj.lev = lev
        obj.dom = dom

        return obj

    def ground_new(self, rep):
        return self.new(rep, self.dom, self.lev)

    @classmethod
    def _parse(cls, rep, dom, lev=None):
        if isinstance(rep, tuple):
            num, den = rep

            if lev is not None:
                if isinstance(num, dict):
                    num = dmp_from_dict(num, lev, dom)

                if isinstance(den, dict):
                    den = dmp_from_dict(den, lev, dom)
            else:
                num, num_lev = dmp_validate(num)
                den, den_lev = dmp_validate(den)

                if num_lev == den_lev:
                    lev = num_lev
                else:
                    raise ValueError('inconsistent number of levels')

            if dmp_zero_p(den, lev):
                raise ZeroDivisionError('fraction denominator')

            if dmp_zero_p(num, lev):
                den = dmp_one(lev, dom)
            else:
                if dmp_negative_p(den, lev, dom):
                    num = dmp_neg(num, lev, dom)
                    den = dmp_neg(den, lev, dom)
        else:
            num = rep

            if lev is not None:
                if isinstance(num, dict):
                    num = dmp_from_dict(num, lev, dom)
                elif not isinstance(num, list):
                    num = dmp_ground(dom.convert(num), lev)
            else:
                num, lev = dmp_validate(num)

            den = dmp_one(lev, dom)

        return num, den, lev

    def __repr__(f):
        return "%s((%s, %s), %s)" % (f.__class__.__name__, f.num, f.den, f.dom)

    def __hash__(f):
        return hash((f.__class__.__name__, dmp_to_tuple(f.num, f.lev),
            dmp_to_tuple(f.den, f.lev), f.lev, f.dom))

    def poly_unify(f, g):
        """Unify a multivariate fraction and a polynomial. """
        if not isinstance(g, DMP) or f.lev != g.lev:
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        if f.dom == g.dom:
            return (f.lev, f.dom, f.per, (f.num, f.den), g._rep)
        else:
            lev, dom = f.lev, f.dom.unify(g.dom)

            F = (dmp_convert(f.num, lev, f.dom, dom),
                 dmp_convert(f.den, lev, f.dom, dom))

            G = dmp_convert(g._rep, lev, g.dom, dom)

            def per(num, den, cancel=True, kill=False, lev=lev):
                if kill:
                    if not lev:
                        return num/den
                    else:
                        lev = lev - 1

                if cancel:
                    num, den = dmp_cancel(num, den, lev, dom)

                return f.__class__.new((num, den), dom, lev)

            return lev, dom, per, F, G

    def frac_unify(f, g):
        """Unify representations of two multivariate fractions. """
        if not isinstance(g, DMF) or f.lev != g.lev:
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        if f.dom == g.dom:
            return (f.lev, f.dom, f.per, (f.num, f.den),
                                         (g.num, g.den))
        else:
            lev, dom = f.lev, f.dom.unify(g.dom)

            F = (dmp_convert(f.num, lev, f.dom, dom),
                 dmp_convert(f.den, lev, f.dom, dom))

            G = (dmp_convert(g.num, lev, g.dom, dom),
                 dmp_convert(g.den, lev, g.dom, dom))

            def per(num, den, cancel=True, kill=False, lev=lev):
                if kill:
                    if not lev:
                        return num/den
                    else:
                        lev = lev - 1

                if cancel:
                    num, den = dmp_cancel(num, den, lev, dom)

                return f.__class__.new((num, den), dom, lev)

            return lev, dom, per, F, G

    def per(f, num, den, cancel=True, kill=False):
        """Create a DMF out of the given representation. """
        lev, dom = f.lev, f.dom

        if kill:
            if not lev:
                return num/den
            else:
                lev -= 1

        if cancel:
            num, den = dmp_cancel(num, den, lev, dom)

        return f.__class__.new((num, den), dom, lev)

    def half_per(f, rep, kill=False):
        """Create a DMP out of the given representation. """
        lev = f.lev

        if kill:
            if not lev:
                return rep
            else:
                lev -= 1

        return DMP(rep, f.dom, lev)

    @classmethod
    def zero(cls, lev, dom):
        return cls.new(0, dom, lev)

    @classmethod
    def one(cls, lev, dom):
        return cls.new(1, dom, lev)

    def numer(f):
        """Returns the numerator of ``f``. """
        return f.half_per(f.num)

    def denom(f):
        """Returns the denominator of ``f``. """
        return f.half_per(f.den)

    def cancel(f):
        """Remove common factors from ``f.num`` and ``f.den``. """
        return f.per(f.num, f.den)

    def neg(f):
        """Negate all coefficients in ``f``. """
        return f.per(dmp_neg(f.num, f.lev, f.dom), f.den, cancel=False)

    def add_ground(f, c):
        """Add an element of the ground domain to ``f``. """
        return f + f.ground_new(c)

    def add(f, g):
        """Add two multivariate fractions ``f`` and ``g``. """
        if isinstance(g, DMP):
            lev, dom, per, (F_num, F_den), G = f.poly_unify(g)
            num, den = dmp_add_mul(F_num, F_den, G, lev, dom), F_den
        else:
            lev, dom, per, F, G = f.frac_unify(g)
            (F_num, F_den), (G_num, G_den) = F, G

            num = dmp_add(dmp_mul(F_num, G_den, lev, dom),
                          dmp_mul(F_den, G_num, lev, dom), lev, dom)
            den = dmp_mul(F_den, G_den, lev, dom)

        return per(num, den)

    def sub(f, g):
        """Subtract two multivariate fractions ``f`` and ``g``. """
        if isinstance(g, DMP):
            lev, dom, per, (F_num, F_den), G = f.poly_unify(g)
            num, den = dmp_sub_mul(F_num, F_den, G, lev, dom), F_den
        else:
            lev, dom, per, F, G = f.frac_unify(g)
            (F_num, F_den), (G_num, G_den) = F, G

            num = dmp_sub(dmp_mul(F_num, G_den, lev, dom),
                          dmp_mul(F_den, G_num, lev, dom), lev, dom)
            den = dmp_mul(F_den, G_den, lev, dom)

        return per(num, den)

    def mul(f, g):
        """Multiply two multivariate fractions ``f`` and ``g``. """
        if isinstance(g, DMP):
            lev, dom, per, (F_num, F_den), G = f.poly_unify(g)
            num, den = dmp_mul(F_num, G, lev, dom), F_den
        else:
            lev, dom, per, F, G = f.frac_unify(g)
            (F_num, F_den), (G_num, G_den) = F, G

            num = dmp_mul(F_num, G_num, lev, dom)
            den = dmp_mul(F_den, G_den, lev, dom)

        return per(num, den)

    def pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
        if isinstance(n, int):
            num, den = f.num, f.den
            if n < 0:
                num, den, n = den, num, -n
            return f.per(dmp_pow(num, n, f.lev, f.dom),
                         dmp_pow(den, n, f.lev, f.dom), cancel=False)
        else:
            raise TypeError("``int`` expected, got %s" % type(n))

    def quo(f, g):
        """Computes quotient of fractions ``f`` and ``g``. """
        if isinstance(g, DMP):
            lev, dom, per, (F_num, F_den), G = f.poly_unify(g)
            num, den = F_num, dmp_mul(F_den, G, lev, dom)
        else:
            lev, dom, per, F, G = f.frac_unify(g)
            (F_num, F_den), (G_num, G_den) = F, G

            num = dmp_mul(F_num, G_den, lev, dom)
            den = dmp_mul(F_den, G_num, lev, dom)

        return per(num, den)

    exquo = quo

    def invert(f, check=True):
        """Computes inverse of a fraction ``f``. """
        return f.per(f.den, f.num, cancel=False)

    @property
    def is_zero(f):
        """Returns ``True`` if ``f`` is a zero fraction. """
        return dmp_zero_p(f.num, f.lev)

    @property
    def is_one(f):
        """Returns ``True`` if ``f`` is a unit fraction. """
        return dmp_one_p(f.num, f.lev, f.dom) and \
            dmp_one_p(f.den, f.lev, f.dom)

    def __neg__(f):
        return f.neg()

    def __add__(f, g):
        if isinstance(g, (DMP, DMF)):
            return f.add(g)
        elif g in f.dom:
            return f.add_ground(f.dom.convert(g))

        try:
            return f.add(f.half_per(g))
        except (TypeError, CoercionFailed, NotImplementedError):
            return NotImplemented

    def __radd__(f, g):
        return f.__add__(g)

    def __sub__(f, g):
        if isinstance(g, (DMP, DMF)):
            return f.sub(g)

        try:
            return f.sub(f.half_per(g))
        except (TypeError, CoercionFailed, NotImplementedError):
            return NotImplemented

    def __rsub__(f, g):
        return (-f).__add__(g)

    def __mul__(f, g):
        if isinstance(g, (DMP, DMF)):
            return f.mul(g)

        try:
            return f.mul(f.half_per(g))
        except (TypeError, CoercionFailed, NotImplementedError):
            return NotImplemented

    def __rmul__(f, g):
        return f.__mul__(g)

    def __pow__(f, n):
        return f.pow(n)

    def __truediv__(f, g):
        if isinstance(g, (DMP, DMF)):
            return f.quo(g)

        try:
            return f.quo(f.half_per(g))
        except (TypeError, CoercionFailed, NotImplementedError):
            return NotImplemented

    def __rtruediv__(self, g):
        return self.invert(check=False)*g

    def __eq__(f, g):
        try:
            if isinstance(g, DMP):
                _, _, _, (F_num, F_den), G = f.poly_unify(g)

                if f.lev == g.lev:
                    return dmp_one_p(F_den, f.lev, f.dom) and F_num == G
            else:
                _, _, _, F, G = f.frac_unify(g)

                if f.lev == g.lev:
                    return F == G
        except UnificationFailed:
            pass

        return False

    def __ne__(f, g):
        try:
            if isinstance(g, DMP):
                _, _, _, (F_num, F_den), G = f.poly_unify(g)

                if f.lev == g.lev:
                    return not (dmp_one_p(F_den, f.lev, f.dom) and F_num == G)
            else:
                _, _, _, F, G = f.frac_unify(g)

                if f.lev == g.lev:
                    return F != G
        except UnificationFailed:
            pass

        return True

    def __lt__(f, g):
        _, _, _, F, G = f.frac_unify(g)
        return F < G

    def __le__(f, g):
        _, _, _, F, G = f.frac_unify(g)
        return F <= G

    def __gt__(f, g):
        _, _, _, F, G = f.frac_unify(g)
        return F > G

    def __ge__(f, g):
        _, _, _, F, G = f.frac_unify(g)
        return F >= G

    def __bool__(f):
        return not dmp_zero_p(f.num, f.lev)


def init_normal_ANP(rep, mod, dom):
    return ANP(dup_normal(rep, dom),
               dup_normal(mod, dom), dom)


class ANP(CantSympify):
    """Dense Algebraic Number Polynomials over a field. """

    __slots__ = ('_rep', '_mod', 'dom')

    def __new__(cls, rep, mod, dom):
        if isinstance(rep, DMP):
            pass
        elif type(rep) is dict: # don't use isinstance
            rep = DMP(dup_from_dict(rep, dom), dom, 0)
        else:
            if isinstance(rep, list):
                rep = [dom.convert(a) for a in rep]
            else:
                rep = [dom.convert(rep)]
            rep = DMP(dup_strip(rep), dom, 0)

        if isinstance(mod, DMP):
            pass
        elif isinstance(mod, dict):
            mod = DMP(dup_from_dict(mod, dom), dom, 0)
        else:
            mod = DMP(dup_strip(mod), dom, 0)

        return cls.new(rep, mod, dom)

    @classmethod
    def new(cls, rep, mod, dom):
        if not (rep.dom == mod.dom == dom):
            raise RuntimeError("Inconsistent domain")
        obj = super().__new__(cls)
        obj._rep = rep
        obj._mod = mod
        obj.dom = dom
        return obj

    # XXX: It should be possible to use __getnewargs__ rather than __reduce__
    # but it doesn't work for some reason. Probably this would be easier if
    # python-flint supported pickling for polynomial types.
    def __reduce__(self):
        return ANP, (self.rep, self.mod, self.dom)

    @property
    def rep(self):
        return self._rep.to_list()

    @property
    def mod(self):
        return self.mod_to_list()

    def to_DMP(self):
        return self._rep

    def mod_to_DMP(self):
        return self._mod

    def per(f, rep):
        return f.new(rep, f._mod, f.dom)

    def __repr__(f):
        return "%s(%s, %s, %s)" % (f.__class__.__name__, f._rep.to_list(), f._mod.to_list(), f.dom)

    def __hash__(f):
        return hash((f.__class__.__name__, f.to_tuple(), f._mod.to_tuple(), f.dom))

    def convert(f, dom):
        """Convert ``f`` to a ``ANP`` over a new domain. """
        if f.dom == dom:
            return f
        else:
            return f.new(f._rep.convert(dom), f._mod.convert(dom), dom)

    def unify(f, g):
        """Unify representations of two algebraic numbers. """

        # XXX: This unify method is not used any more because unify_ANP is used
        # instead.

        if not isinstance(g, ANP) or f.mod != g.mod:
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        if f.dom == g.dom:
            return f.dom, f.per, f.rep, g.rep, f.mod
        else:
            dom = f.dom.unify(g.dom)

            F = dup_convert(f.rep, f.dom, dom)
            G = dup_convert(g.rep, g.dom, dom)

            if dom != f.dom and dom != g.dom:
                mod = dup_convert(f.mod, f.dom, dom)
            else:
                if dom == f.dom:
                    mod = f.mod
                else:
                    mod = g.mod

            per = lambda rep: ANP(rep, mod, dom)

        return dom, per, F, G, mod

    def unify_ANP(f, g):
        """Unify and return ``DMP`` instances of ``f`` and ``g``. """
        if not isinstance(g, ANP) or f._mod != g._mod:
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        # The domain is almost always QQ but there are some tests involving ZZ
        if f.dom != g.dom:
            dom = f.dom.unify(g.dom)
            f = f.convert(dom)
            g = g.convert(dom)

        return f._rep, g._rep, f._mod, f.dom

    @classmethod
    def zero(cls, mod, dom):
        return ANP(0, mod, dom)

    @classmethod
    def one(cls, mod, dom):
        return ANP(1, mod, dom)

    def to_dict(f):
        """Convert ``f`` to a dict representation with native coefficients. """
        return f._rep.to_dict()

    def to_sympy_dict(f):
        """Convert ``f`` to a dict representation with SymPy coefficients. """
        rep = dmp_to_dict(f.rep, 0, f.dom)

        for k, v in rep.items():
            rep[k] = f.dom.to_sympy(v)

        return rep

    def to_list(f):
        """Convert ``f`` to a list representation with native coefficients. """
        return f._rep.to_list()

    def mod_to_list(f):
        """Return ``f.mod`` as a list with native coefficients. """
        return f._mod.to_list()

    def to_sympy_list(f):
        """Convert ``f`` to a list representation with SymPy coefficients. """
        return [ f.dom.to_sympy(c) for c in f.to_list() ]

    def to_tuple(f):
        """
        Convert ``f`` to a tuple representation with native coefficients.

        This is needed for hashing.
        """
        return f._rep.to_tuple()

    @classmethod
    def from_list(cls, rep, mod, dom):
        return ANP(dup_strip(list(map(dom.convert, rep))), mod, dom)

    def add_ground(f, c):
        """Add an element of the ground domain to ``f``. """
        return f.per(f._rep.add_ground(c))

    def sub_ground(f, c):
        """Subtract an element of the ground domain from ``f``. """
        return f.per(f._rep.sub_ground(c))

    def mul_ground(f, c):
        """Multiply ``f`` by an element of the ground domain. """
        return f.per(f._rep.mul_ground(c))

    def quo_ground(f, c):
        """Quotient of ``f`` by an element of the ground domain. """
        return f.per(f._rep.quo_ground(c))

    def neg(f):
        return f.per(f._rep.neg())

    def add(f, g):
        F, G, mod, dom = f.unify_ANP(g)
        return f.new(F.add(G), mod, dom)

    def sub(f, g):
        F, G, mod, dom = f.unify_ANP(g)
        return f.new(F.sub(G), mod, dom)

    def mul(f, g):
        F, G, mod, dom = f.unify_ANP(g)
        return f.new(F.mul(G).rem(mod), mod, dom)

    def pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
        if not isinstance(n, int):
            raise TypeError("``int`` expected, got %s" % type(n))

        mod = f._mod
        F = f._rep

        if n < 0:
            F, n = F.invert(mod), -n

        # XXX: Need a pow_mod method for DMP
        return f.new(F.pow(n).rem(f._mod), mod, f.dom)

    def exquo(f, g):
        F, G, mod, dom = f.unify_ANP(g)
        return f.new(F.mul(G.invert(mod)).rem(mod), mod, dom)

    def div(f, g):
        return f.exquo(g), f.zero(f._mod, f.dom)

    def quo(f, g):
        return f.exquo(g)

    def rem(f, g):
        F, G, mod, dom = f.unify_ANP(g)
        s, h = F.half_gcdex(G)

        if h.is_one:
            return f.zero(mod, dom)
        else:
            raise NotInvertible("zero divisor")

    def LC(f):
        """Returns the leading coefficient of ``f``. """
        return f._rep.LC()

    def TC(f):
        """Returns the trailing coefficient of ``f``. """
        return f._rep.TC()

    @property
    def is_zero(f):
        """Returns ``True`` if ``f`` is a zero algebraic number. """
        return f._rep.is_zero

    @property
    def is_one(f):
        """Returns ``True`` if ``f`` is a unit algebraic number. """
        return f._rep.is_one

    @property
    def is_ground(f):
        """Returns ``True`` if ``f`` is an element of the ground domain. """
        return f._rep.is_ground

    def __pos__(f):
        return f

    def __neg__(f):
        return f.neg()

    def __add__(f, g):
        if isinstance(g, ANP):
            return f.add(g)
        try:
            g = f.dom.convert(g)
        except CoercionFailed:
            return NotImplemented
        else:
            return f.add_ground(g)

    def __radd__(f, g):
        return f.__add__(g)

    def __sub__(f, g):
        if isinstance(g, ANP):
            return f.sub(g)
        try:
            g = f.dom.convert(g)
        except CoercionFailed:
            return NotImplemented
        else:
            return f.sub_ground(g)

    def __rsub__(f, g):
        return (-f).__add__(g)

    def __mul__(f, g):
        if isinstance(g, ANP):
            return f.mul(g)
        try:
            g = f.dom.convert(g)
        except CoercionFailed:
            return NotImplemented
        else:
            return f.mul_ground(g)

    def __rmul__(f, g):
        return f.__mul__(g)

    def __pow__(f, n):
        return f.pow(n)

    def __divmod__(f, g):
        return f.div(g)

    def __mod__(f, g):
        return f.rem(g)

    def __truediv__(f, g):
        if isinstance(g, ANP):
            return f.quo(g)
        try:
            g = f.dom.convert(g)
        except CoercionFailed:
            return NotImplemented
        else:
            return f.quo_ground(g)

    def __eq__(f, g):
        try:
            F, G, _, _ = f.unify_ANP(g)
        except UnificationFailed:
            return NotImplemented
        return F == G

    def __ne__(f, g):
        try:
            F, G, _, _ = f.unify_ANP(g)
        except UnificationFailed:
            return NotImplemented
        return F != G

    def __lt__(f, g):
        F, G, _, _ = f.unify_ANP(g)
        return F < G

    def __le__(f, g):
        F, G, _, _ = f.unify_ANP(g)
        return F <= G

    def __gt__(f, g):
        F, G, _, _ = f.unify_ANP(g)
        return F > G

    def __ge__(f, g):
        F, G, _, _ = f.unify_ANP(g)
        return F >= G

    def __bool__(f):
        return bool(f._rep)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\iterables.py ===
from collections import Counter, defaultdict, OrderedDict
from itertools import (
    chain, combinations, combinations_with_replacement, cycle, islice,
    permutations, product, groupby
)
# For backwards compatibility
from itertools import product as cartes # noqa: F401
from operator import gt



# this is the logical location of these functions
from sympy.utilities.enumerative import (
    multiset_partitions_taocp, list_visitor, MultisetPartitionTraverser)

from sympy.utilities.misc import as_int
from sympy.utilities.decorator import deprecated


def is_palindromic(s, i=0, j=None):
    """
    Return True if the sequence is the same from left to right as it
    is from right to left in the whole sequence (default) or in the
    Python slice ``s[i: j]``; else False.

    Examples
    ========

    >>> from sympy.utilities.iterables import is_palindromic
    >>> is_palindromic([1, 0, 1])
    True
    >>> is_palindromic('abcbb')
    False
    >>> is_palindromic('abcbb', 1)
    False

    Normal Python slicing is performed in place so there is no need to
    create a slice of the sequence for testing:

    >>> is_palindromic('abcbb', 1, -1)
    True
    >>> is_palindromic('abcbb', -4, -1)
    True

    See Also
    ========

    sympy.ntheory.digits.is_palindromic: tests integers

    """
    i, j, _ = slice(i, j).indices(len(s))
    m = (j - i)//2
    # if length is odd, middle element will be ignored
    return all(s[i + k] == s[j - 1 - k] for k in range(m))


def flatten(iterable, levels=None, cls=None):  # noqa: F811
    """
    Recursively denest iterable containers.

    >>> from sympy import flatten

    >>> flatten([1, 2, 3])
    [1, 2, 3]
    >>> flatten([1, 2, [3]])
    [1, 2, 3]
    >>> flatten([1, [2, 3], [4, 5]])
    [1, 2, 3, 4, 5]
    >>> flatten([1.0, 2, (1, None)])
    [1.0, 2, 1, None]

    If you want to denest only a specified number of levels of
    nested containers, then set ``levels`` flag to the desired
    number of levels::

    >>> ls = [[(-2, -1), (1, 2)], [(0, 0)]]

    >>> flatten(ls, levels=1)
    [(-2, -1), (1, 2), (0, 0)]

    If cls argument is specified, it will only flatten instances of that
    class, for example:

    >>> from sympy import Basic, S
    >>> class MyOp(Basic):
    ...     pass
    ...
    >>> flatten([MyOp(S(1), MyOp(S(2), S(3)))], cls=MyOp)
    [1, 2, 3]

    adapted from https://kogs-www.informatik.uni-hamburg.de/~meine/python_tricks
    """
    from sympy.tensor.array import NDimArray
    if levels is not None:
        if not levels:
            return iterable
        elif levels > 0:
            levels -= 1
        else:
            raise ValueError(
                "expected non-negative number of levels, got %s" % levels)

    if cls is None:
        def reducible(x):
            return is_sequence(x, set)
    else:
        def reducible(x):
            return isinstance(x, cls)

    result = []

    for el in iterable:
        if reducible(el):
            if hasattr(el, 'args') and not isinstance(el, NDimArray):
                el = el.args
            result.extend(flatten(el, levels=levels, cls=cls))
        else:
            result.append(el)

    return result


def unflatten(iter, n=2):
    """Group ``iter`` into tuples of length ``n``. Raise an error if
    the length of ``iter`` is not a multiple of ``n``.
    """
    if n < 1 or len(iter) % n:
        raise ValueError('iter length is not a multiple of %i' % n)
    return list(zip(*(iter[i::n] for i in range(n))))


def reshape(seq, how):
    """Reshape the sequence according to the template in ``how``.

    Examples
    ========

    >>> from sympy.utilities import reshape
    >>> seq = list(range(1, 9))

    >>> reshape(seq, [4]) # lists of 4
    [[1, 2, 3, 4], [5, 6, 7, 8]]

    >>> reshape(seq, (4,)) # tuples of 4
    [(1, 2, 3, 4), (5, 6, 7, 8)]

    >>> reshape(seq, (2, 2)) # tuples of 4
    [(1, 2, 3, 4), (5, 6, 7, 8)]

    >>> reshape(seq, (2, [2])) # (i, i, [i, i])
    [(1, 2, [3, 4]), (5, 6, [7, 8])]

    >>> reshape(seq, ((2,), [2])) # etc....
    [((1, 2), [3, 4]), ((5, 6), [7, 8])]

    >>> reshape(seq, (1, [2], 1))
    [(1, [2, 3], 4), (5, [6, 7], 8)]

    >>> reshape(tuple(seq), ([[1], 1, (2,)],))
    (([[1], 2, (3, 4)],), ([[5], 6, (7, 8)],))

    >>> reshape(tuple(seq), ([1], 1, (2,)))
    (([1], 2, (3, 4)), ([5], 6, (7, 8)))

    >>> reshape(list(range(12)), [2, [3], {2}, (1, (3,), 1)])
    [[0, 1, [2, 3, 4], {5, 6}, (7, (8, 9, 10), 11)]]

    """
    m = sum(flatten(how))
    n, rem = divmod(len(seq), m)
    if m < 0 or rem:
        raise ValueError('template must sum to positive number '
        'that divides the length of the sequence')
    i = 0
    container = type(how)
    rv = [None]*n
    for k in range(len(rv)):
        _rv = []
        for hi in how:
            if isinstance(hi, int):
                _rv.extend(seq[i: i + hi])
                i += hi
            else:
                n = sum(flatten(hi))
                hi_type = type(hi)
                _rv.append(hi_type(reshape(seq[i: i + n], hi)[0]))
                i += n
        rv[k] = container(_rv)
    return type(seq)(rv)


def group(seq, multiple=True):
    """
    Splits a sequence into a list of lists of equal, adjacent elements.

    Examples
    ========

    >>> from sympy import group

    >>> group([1, 1, 1, 2, 2, 3])
    [[1, 1, 1], [2, 2], [3]]
    >>> group([1, 1, 1, 2, 2, 3], multiple=False)
    [(1, 3), (2, 2), (3, 1)]
    >>> group([1, 1, 3, 2, 2, 1], multiple=False)
    [(1, 2), (3, 1), (2, 2), (1, 1)]

    See Also
    ========

    multiset

    """
    if multiple:
        return [(list(g)) for _, g in groupby(seq)]
    return [(k, len(list(g))) for k, g in groupby(seq)]


def _iproduct2(iterable1, iterable2):
    '''Cartesian product of two possibly infinite iterables'''

    it1 = iter(iterable1)
    it2 = iter(iterable2)

    elems1 = []
    elems2 = []

    sentinel = object()
    def append(it, elems):
        e = next(it, sentinel)
        if e is not sentinel:
            elems.append(e)

    n = 0
    append(it1, elems1)
    append(it2, elems2)

    while n <= len(elems1) + len(elems2):
        for m in range(n-len(elems1)+1, len(elems2)):
            yield (elems1[n-m], elems2[m])
        n += 1
        append(it1, elems1)
        append(it2, elems2)


def iproduct(*iterables):
    '''
    Cartesian product of iterables.

    Generator of the Cartesian product of iterables. This is analogous to
    itertools.product except that it works with infinite iterables and will
    yield any item from the infinite product eventually.

    Examples
    ========

    >>> from sympy.utilities.iterables import iproduct
    >>> sorted(iproduct([1,2], [3,4]))
    [(1, 3), (1, 4), (2, 3), (2, 4)]

    With an infinite iterator:

    >>> from sympy import S
    >>> (3,) in iproduct(S.Integers)
    True
    >>> (3, 4) in iproduct(S.Integers, S.Integers)
    True

    .. seealso::

       `itertools.product
       <https://docs.python.org/3/library/itertools.html#itertools.product>`_
    '''
    if len(iterables) == 0:
        yield ()
        return
    elif len(iterables) == 1:
        for e in iterables[0]:
            yield (e,)
    elif len(iterables) == 2:
        yield from _iproduct2(*iterables)
    else:
        first, others = iterables[0], iterables[1:]
        for ef, eo in _iproduct2(first, iproduct(*others)):
            yield (ef,) + eo


def multiset(seq):
    """Return the hashable sequence in multiset form with values being the
    multiplicity of the item in the sequence.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset
    >>> multiset('mississippi')
    {'i': 4, 'm': 1, 'p': 2, 's': 4}

    See Also
    ========

    group

    """
    return dict(Counter(seq).items())




def ibin(n, bits=None, str=False):
    """Return a list of length ``bits`` corresponding to the binary value
    of ``n`` with small bits to the right (last). If bits is omitted, the
    length will be the number required to represent ``n``. If the bits are
    desired in reversed order, use the ``[::-1]`` slice of the returned list.

    If a sequence of all bits-length lists starting from ``[0, 0,..., 0]``
    through ``[1, 1, ..., 1]`` are desired, pass a non-integer for bits, e.g.
    ``'all'``.

    If the bit *string* is desired pass ``str=True``.

    Examples
    ========

    >>> from sympy.utilities.iterables import ibin
    >>> ibin(2)
    [1, 0]
    >>> ibin(2, 4)
    [0, 0, 1, 0]

    If all lists corresponding to 0 to 2**n - 1, pass a non-integer
    for bits:

    >>> bits = 2
    >>> for i in ibin(2, 'all'):
    ...     print(i)
    (0, 0)
    (0, 1)
    (1, 0)
    (1, 1)

    If a bit string is desired of a given length, use str=True:

    >>> n = 123
    >>> bits = 10
    >>> ibin(n, bits, str=True)
    '0001111011'
    >>> ibin(n, bits, str=True)[::-1]  # small bits left
    '1101111000'
    >>> list(ibin(3, 'all', str=True))
    ['000', '001', '010', '011', '100', '101', '110', '111']

    """
    if n < 0:
        raise ValueError("negative numbers are not allowed")
    n = as_int(n)

    if bits is None:
        bits = 0
    else:
        try:
            bits = as_int(bits)
        except ValueError:
            bits = -1
        else:
            if n.bit_length() > bits:
                raise ValueError(
                    "`bits` must be >= {}".format(n.bit_length()))

    if not str:
        if bits >= 0:
            return [1 if i == "1" else 0 for i in bin(n)[2:].rjust(bits, "0")]
        else:
            return variations(range(2), n, repetition=True)
    else:
        if bits >= 0:
            return bin(n)[2:].rjust(bits, "0")
        else:
            return (bin(i)[2:].rjust(n, "0") for i in range(2**n))


def variations(seq, n, repetition=False):
    r"""Returns an iterator over the n-sized variations of ``seq`` (size N).
    ``repetition`` controls whether items in ``seq`` can appear more than once;

    Examples
    ========

    ``variations(seq, n)`` will return `\frac{N!}{(N - n)!}` permutations without
    repetition of ``seq``'s elements:

        >>> from sympy import variations
        >>> list(variations([1, 2], 2))
        [(1, 2), (2, 1)]

    ``variations(seq, n, True)`` will return the `N^n` permutations obtained
    by allowing repetition of elements:

        >>> list(variations([1, 2], 2, repetition=True))
        [(1, 1), (1, 2), (2, 1), (2, 2)]

    If you ask for more items than are in the set you get the empty set unless
    you allow repetitions:

        >>> list(variations([0, 1], 3, repetition=False))
        []
        >>> list(variations([0, 1], 3, repetition=True))[:4]
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]

    .. seealso::

       `itertools.permutations
       <https://docs.python.org/3/library/itertools.html#itertools.permutations>`_,
       `itertools.product
       <https://docs.python.org/3/library/itertools.html#itertools.product>`_
    """
    if not repetition:
        seq = tuple(seq)
        if len(seq) < n:
            return iter(())  # 0 length iterator
        return permutations(seq, n)
    else:
        if n == 0:
            return iter(((),))  # yields 1 empty tuple
        else:
            return product(seq, repeat=n)


def subsets(seq, k=None, repetition=False):
    r"""Generates all `k`-subsets (combinations) from an `n`-element set, ``seq``.

    A `k`-subset of an `n`-element set is any subset of length exactly `k`. The
    number of `k`-subsets of an `n`-element set is given by ``binomial(n, k)``,
    whereas there are `2^n` subsets all together. If `k` is ``None`` then all
    `2^n` subsets will be returned from shortest to longest.

    Examples
    ========

    >>> from sympy import subsets

    ``subsets(seq, k)`` will return the
    `\frac{n!}{k!(n - k)!}` `k`-subsets (combinations)
    without repetition, i.e. once an item has been removed, it can no
    longer be "taken":

        >>> list(subsets([1, 2], 2))
        [(1, 2)]
        >>> list(subsets([1, 2]))
        [(), (1,), (2,), (1, 2)]
        >>> list(subsets([1, 2, 3], 2))
        [(1, 2), (1, 3), (2, 3)]


    ``subsets(seq, k, repetition=True)`` will return the
    `\frac{(n - 1 + k)!}{k!(n - 1)!}`
    combinations *with* repetition:

        >>> list(subsets([1, 2], 2, repetition=True))
        [(1, 1), (1, 2), (2, 2)]

    If you ask for more items than are in the set you get the empty set unless
    you allow repetitions:

        >>> list(subsets([0, 1], 3, repetition=False))
        []
        >>> list(subsets([0, 1], 3, repetition=True))
        [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]

    """
    if k is None:
        if not repetition:
            return chain.from_iterable((combinations(seq, k)
                                        for k in range(len(seq) + 1)))
        else:
            return chain.from_iterable((combinations_with_replacement(seq, k)
                                        for k in range(len(seq) + 1)))
    else:
        if not repetition:
            return combinations(seq, k)
        else:
            return combinations_with_replacement(seq, k)


def filter_symbols(iterator, exclude):
    """
    Only yield elements from `iterator` that do not occur in `exclude`.

    Parameters
    ==========

    iterator : iterable
        iterator to take elements from

    exclude : iterable
        elements to exclude

    Returns
    =======

    iterator : iterator
        filtered iterator
    """
    exclude = set(exclude)
    for s in iterator:
        if s not in exclude:
            yield s

def numbered_symbols(prefix='x', cls=None, start=0, exclude=(), *args, **assumptions):
    """
    Generate an infinite stream of Symbols consisting of a prefix and
    increasing subscripts provided that they do not occur in ``exclude``.

    Parameters
    ==========

    prefix : str, optional
        The prefix to use. By default, this function will generate symbols of
        the form "x0", "x1", etc.

    cls : class, optional
        The class to use. By default, it uses ``Symbol``, but you can also use ``Wild``
        or ``Dummy``.

    start : int, optional
        The start number.  By default, it is 0.

    exclude : list, tuple, set of cls, optional
        Symbols to be excluded.

    *args, **kwargs
        Additional positional and keyword arguments are passed to the *cls* class.

    Returns
    =======

    sym : Symbol
        The subscripted symbols.
    """
    exclude = set(exclude or [])
    if cls is None:
        # We can't just make the default cls=Symbol because it isn't
        # imported yet.
        from sympy.core import Symbol
        cls = Symbol

    while True:
        name = '%s%s' % (prefix, start)
        s = cls(name, *args, **assumptions)
        if s not in exclude:
            yield s
        start += 1


def capture(func):
    """Return the printed output of func().

    ``func`` should be a function without arguments that produces output with
    print statements.

    >>> from sympy.utilities.iterables import capture
    >>> from sympy import pprint
    >>> from sympy.abc import x
    >>> def foo():
    ...     print('hello world!')
    ...
    >>> 'hello' in capture(foo) # foo, not foo()
    True
    >>> capture(lambda: pprint(2/x))
    '2\\n-\\nx\\n'

    """
    from io import StringIO
    import sys

    stdout = sys.stdout
    sys.stdout = file = StringIO()
    try:
        func()
    finally:
        sys.stdout = stdout
    return file.getvalue()


def sift(seq, keyfunc, binary=False):
    """
    Sift the sequence, ``seq`` according to ``keyfunc``.

    Returns
    =======

    When ``binary`` is ``False`` (default), the output is a dictionary
    where elements of ``seq`` are stored in a list keyed to the value
    of keyfunc for that element. If ``binary`` is True then a tuple
    with lists ``T`` and ``F`` are returned where ``T`` is a list
    containing elements of seq for which ``keyfunc`` was ``True`` and
    ``F`` containing those elements for which ``keyfunc`` was ``False``;
    a ValueError is raised if the ``keyfunc`` is not binary.

    Examples
    ========

    >>> from sympy.utilities import sift
    >>> from sympy.abc import x, y
    >>> from sympy import sqrt, exp, pi, Tuple

    >>> sift(range(5), lambda x: x % 2)
    {0: [0, 2, 4], 1: [1, 3]}

    sift() returns a defaultdict() object, so any key that has no matches will
    give [].

    >>> sift([x], lambda x: x.is_commutative)
    {True: [x]}
    >>> _[False]
    []

    Sometimes you will not know how many keys you will get:

    >>> sift([sqrt(x), exp(x), (y**x)**2],
    ...      lambda x: x.as_base_exp()[0])
    {E: [exp(x)], x: [sqrt(x)], y: [y**(2*x)]}

    Sometimes you expect the results to be binary; the
    results can be unpacked by setting ``binary`` to True:

    >>> sift(range(4), lambda x: x % 2, binary=True)
    ([1, 3], [0, 2])
    >>> sift(Tuple(1, pi), lambda x: x.is_rational, binary=True)
    ([1], [pi])

    A ValueError is raised if the predicate was not actually binary
    (which is a good test for the logic where sifting is used and
    binary results were expected):

    >>> unknown = exp(1) - pi  # the rationality of this is unknown
    >>> args = Tuple(1, pi, unknown)
    >>> sift(args, lambda x: x.is_rational, binary=True)
    Traceback (most recent call last):
    ...
    ValueError: keyfunc gave non-binary output

    The non-binary sifting shows that there were 3 keys generated:

    >>> set(sift(args, lambda x: x.is_rational).keys())
    {None, False, True}

    If you need to sort the sifted items it might be better to use
    ``ordered`` which can economically apply multiple sort keys
    to a sequence while sorting.

    See Also
    ========

    ordered

    """
    if not binary:
        m = defaultdict(list)
        for i in seq:
            m[keyfunc(i)].append(i)
        return m
    sift = F, T = [], []
    for i in seq:
        try:
            sift[keyfunc(i)].append(i)
        except (IndexError, TypeError):
            raise ValueError('keyfunc gave non-binary output')
    return T, F


def take(iter, n):
    """Return ``n`` items from ``iter`` iterator. """
    return [ value for _, value in zip(range(n), iter) ]


def dict_merge(*dicts):
    """Merge dictionaries into a single dictionary. """
    merged = {}

    for dict in dicts:
        merged.update(dict)

    return merged


def common_prefix(*seqs):
    """Return the subsequence that is a common start of sequences in ``seqs``.

    >>> from sympy.utilities.iterables import common_prefix
    >>> common_prefix(list(range(3)))
    [0, 1, 2]
    >>> common_prefix(list(range(3)), list(range(4)))
    [0, 1, 2]
    >>> common_prefix([1, 2, 3], [1, 2, 5])
    [1, 2]
    >>> common_prefix([1, 2, 3], [1, 3, 5])
    [1]
    """
    if not all(seqs):
        return []
    elif len(seqs) == 1:
        return seqs[0]
    i = 0
    for i in range(min(len(s) for s in seqs)):
        if not all(seqs[j][i] == seqs[0][i] for j in range(len(seqs))):
            break
    else:
        i += 1
    return seqs[0][:i]


def common_suffix(*seqs):
    """Return the subsequence that is a common ending of sequences in ``seqs``.

    >>> from sympy.utilities.iterables import common_suffix
    >>> common_suffix(list(range(3)))
    [0, 1, 2]
    >>> common_suffix(list(range(3)), list(range(4)))
    []
    >>> common_suffix([1, 2, 3], [9, 2, 3])
    [2, 3]
    >>> common_suffix([1, 2, 3], [9, 7, 3])
    [3]
    """

    if not all(seqs):
        return []
    elif len(seqs) == 1:
        return seqs[0]
    i = 0
    for i in range(-1, -min(len(s) for s in seqs) - 1, -1):
        if not all(seqs[j][i] == seqs[0][i] for j in range(len(seqs))):
            break
    else:
        i -= 1
    if i == -1:
        return []
    else:
        return seqs[0][i + 1:]


def prefixes(seq):
    """
    Generate all prefixes of a sequence.

    Examples
    ========

    >>> from sympy.utilities.iterables import prefixes

    >>> list(prefixes([1,2,3,4]))
    [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]]

    """
    n = len(seq)

    for i in range(n):
        yield seq[:i + 1]


def postfixes(seq):
    """
    Generate all postfixes of a sequence.

    Examples
    ========

    >>> from sympy.utilities.iterables import postfixes

    >>> list(postfixes([1,2,3,4]))
    [[4], [3, 4], [2, 3, 4], [1, 2, 3, 4]]

    """
    n = len(seq)

    for i in range(n):
        yield seq[n - i - 1:]


def topological_sort(graph, key=None):
    r"""
    Topological sort of graph's vertices.

    Parameters
    ==========

    graph : tuple[list, list[tuple[T, T]]
        A tuple consisting of a list of vertices and a list of edges of
        a graph to be sorted topologically.

    key : callable[T] (optional)
        Ordering key for vertices on the same level. By default the natural
        (e.g. lexicographic) ordering is used (in this case the base type
        must implement ordering relations).

    Examples
    ========

    Consider a graph::

        +---+     +---+     +---+
        | 7 |\    | 5 |     | 3 |
        +---+ \   +---+     +---+
          |   _\___/ ____   _/ |
          |  /  \___/    \ /   |
          V  V           V V   |
         +----+         +---+  |
         | 11 |         | 8 |  |
         +----+         +---+  |
          | | \____   ___/ _   |
          | \      \ /    / \  |
          V  \     V V   /  V  V
        +---+ \   +---+ |  +----+
        | 2 |  |  | 9 | |  | 10 |
        +---+  |  +---+ |  +----+
               \________/

    where vertices are integers. This graph can be encoded using
    elementary Python's data structures as follows::

        >>> V = [2, 3, 5, 7, 8, 9, 10, 11]
        >>> E = [(7, 11), (7, 8), (5, 11), (3, 8), (3, 10),
        ...      (11, 2), (11, 9), (11, 10), (8, 9)]

    To compute a topological sort for graph ``(V, E)`` issue::

        >>> from sympy.utilities.iterables import topological_sort

        >>> topological_sort((V, E))
        [3, 5, 7, 8, 11, 2, 9, 10]

    If specific tie breaking approach is needed, use ``key`` parameter::

        >>> topological_sort((V, E), key=lambda v: -v)
        [7, 5, 11, 3, 10, 8, 9, 2]

    Only acyclic graphs can be sorted. If the input graph has a cycle,
    then ``ValueError`` will be raised::

        >>> topological_sort((V, E + [(10, 7)]))
        Traceback (most recent call last):
        ...
        ValueError: cycle detected

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Topological_sorting

    """
    V, E = graph

    L = []
    S = set(V)
    E = list(E)

    S.difference_update(u for v, u in E)

    if key is None:
        def key(value):
            return value

    S = sorted(S, key=key, reverse=True)

    while S:
        node = S.pop()
        L.append(node)

        for u, v in list(E):
            if u == node:
                E.remove((u, v))

                for _u, _v in E:
                    if v == _v:
                        break
                else:
                    kv = key(v)

                    for i, s in enumerate(S):
                        ks = key(s)

                        if kv > ks:
                            S.insert(i, v)
                            break
                    else:
                        S.append(v)

    if E:
        raise ValueError("cycle detected")
    else:
        return L


def strongly_connected_components(G):
    r"""
    Strongly connected components of a directed graph in reverse topological
    order.


    Parameters
    ==========

    G : tuple[list, list[tuple[T, T]]
        A tuple consisting of a list of vertices and a list of edges of
        a graph whose strongly connected components are to be found.


    Examples
    ========

    Consider a directed graph (in dot notation)::

        digraph {
            A -> B
            A -> C
            B -> C
            C -> B
            B -> D
        }

    .. graphviz::

        digraph {
            A -> B
            A -> C
            B -> C
            C -> B
            B -> D
        }

    where vertices are the letters A, B, C and D. This graph can be encoded
    using Python's elementary data structures as follows::

        >>> V = ['A', 'B', 'C', 'D']
        >>> E = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'B'), ('B', 'D')]

    The strongly connected components of this graph can be computed as

        >>> from sympy.utilities.iterables import strongly_connected_components

        >>> strongly_connected_components((V, E))
        [['D'], ['B', 'C'], ['A']]

    This also gives the components in reverse topological order.

    Since the subgraph containing B and C has a cycle they must be together in
    a strongly connected component. A and D are connected to the rest of the
    graph but not in a cyclic manner so they appear as their own strongly
    connected components.


    Notes
    =====

    The vertices of the graph must be hashable for the data structures used.
    If the vertices are unhashable replace them with integer indices.

    This function uses Tarjan's algorithm to compute the strongly connected
    components in `O(|V|+|E|)` (linear) time.


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Strongly_connected_component
    .. [2] https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm


    See Also
    ========

    sympy.utilities.iterables.connected_components

    """
    # Map from a vertex to its neighbours
    V, E = G
    Gmap = {vi: [] for vi in V}
    for v1, v2 in E:
        Gmap[v1].append(v2)
    return _strongly_connected_components(V, Gmap)


def _strongly_connected_components(V, Gmap):
    """More efficient internal routine for strongly_connected_components"""
    #
    # Here V is an iterable of vertices and Gmap is a dict mapping each vertex
    # to a list of neighbours e.g.:
    #
    #   V = [0, 1, 2, 3]
    #   Gmap = {0: [2, 3], 1: [0]}
    #
    # For a large graph these data structures can often be created more
    # efficiently then those expected by strongly_connected_components() which
    # in this case would be
    #
    #   V = [0, 1, 2, 3]
    #   Gmap = [(0, 2), (0, 3), (1, 0)]
    #
    # XXX: Maybe this should be the recommended function to use instead...
    #

    # Non-recursive Tarjan's algorithm:
    lowlink = {}
    indices = {}
    stack = OrderedDict()
    callstack = []
    components = []
    nomore = object()

    def start(v):
        index = len(stack)
        indices[v] = lowlink[v] = index
        stack[v] = None
        callstack.append((v, iter(Gmap[v])))

    def finish(v1):
        # Finished a component?
        if lowlink[v1] == indices[v1]:
            component = [stack.popitem()[0]]
            while component[-1] is not v1:
                component.append(stack.popitem()[0])
            components.append(component[::-1])
        v2, _ = callstack.pop()
        if callstack:
            v1, _ = callstack[-1]
            lowlink[v1] = min(lowlink[v1], lowlink[v2])

    for v in V:
        if v in indices:
            continue
        start(v)
        while callstack:
            v1, it1 = callstack[-1]
            v2 = next(it1, nomore)
            # Finished children of v1?
            if v2 is nomore:
                finish(v1)
            # Recurse on v2
            elif v2 not in indices:
                start(v2)
            elif v2 in stack:
                lowlink[v1] = min(lowlink[v1], indices[v2])

    # Reverse topological sort order:
    return components


def connected_components(G):
    r"""
    Connected components of an undirected graph or weakly connected components
    of a directed graph.


    Parameters
    ==========

    G : tuple[list, list[tuple[T, T]]
        A tuple consisting of a list of vertices and a list of edges of
        a graph whose connected components are to be found.


    Examples
    ========


    Given an undirected graph::

        graph {
            A -- B
            C -- D
        }

    .. graphviz::

        graph {
            A -- B
            C -- D
        }

    We can find the connected components using this function if we include
    each edge in both directions::

        >>> from sympy.utilities.iterables import connected_components

        >>> V = ['A', 'B', 'C', 'D']
        >>> E = [('A', 'B'), ('B', 'A'), ('C', 'D'), ('D', 'C')]
        >>> connected_components((V, E))
        [['A', 'B'], ['C', 'D']]

    The weakly connected components of a directed graph can found the same
    way.


    Notes
    =====

    The vertices of the graph must be hashable for the data structures used.
    If the vertices are unhashable replace them with integer indices.

    This function uses Tarjan's algorithm to compute the connected components
    in `O(|V|+|E|)` (linear) time.


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Component_%28graph_theory%29
    .. [2] https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm


    See Also
    ========

    sympy.utilities.iterables.strongly_connected_components

    """
    # Duplicate edges both ways so that the graph is effectively undirected
    # and return the strongly connected components:
    V, E = G
    E_undirected = []
    for v1, v2 in E:
        E_undirected.extend([(v1, v2), (v2, v1)])
    return strongly_connected_components((V, E_undirected))


def rotate_left(x, y):
    """
    Left rotates a list x by the number of steps specified
    in y.

    Examples
    ========

    >>> from sympy.utilities.iterables import rotate_left
    >>> a = [0, 1, 2]
    >>> rotate_left(a, 1)
    [1, 2, 0]
    """
    if len(x) == 0:
        return []
    y = y % len(x)
    return x[y:] + x[:y]


def rotate_right(x, y):
    """
    Right rotates a list x by the number of steps specified
    in y.

    Examples
    ========

    >>> from sympy.utilities.iterables import rotate_right
    >>> a = [0, 1, 2]
    >>> rotate_right(a, 1)
    [2, 0, 1]
    """
    if len(x) == 0:
        return []
    y = len(x) - y % len(x)
    return x[y:] + x[:y]


def least_rotation(x, key=None):
    '''
    Returns the number of steps of left rotation required to
    obtain lexicographically minimal string/list/tuple, etc.

    Examples
    ========

    >>> from sympy.utilities.iterables import least_rotation, rotate_left
    >>> a = [3, 1, 5, 1, 2]
    >>> least_rotation(a)
    3
    >>> rotate_left(a, _)
    [1, 2, 3, 1, 5]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lexicographically_minimal_string_rotation

    '''
    from sympy.functions.elementary.miscellaneous import Id
    if key is None: key = Id
    S = x + x      # Concatenate string to it self to avoid modular arithmetic
    f = [-1] * len(S)     # Failure function
    k = 0       # Least rotation of string found so far
    for j in range(1,len(S)):
        sj = S[j]
        i = f[j-k-1]
        while i != -1 and sj != S[k+i+1]:
            if key(sj) < key(S[k+i+1]):
                k = j-i-1
            i = f[i]
        if sj != S[k+i+1]:
            if key(sj) < key(S[k]):
                k = j
            f[j-k] = -1
        else:
            f[j-k] = i+1
    return k


def multiset_combinations(m, n, g=None):
    """
    Return the unique combinations of size ``n`` from multiset ``m``.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset_combinations
    >>> from itertools import combinations
    >>> [''.join(i) for i in  multiset_combinations('baby', 3)]
    ['abb', 'aby', 'bby']

    >>> def count(f, s): return len(list(f(s, 3)))

    The number of combinations depends on the number of letters; the
    number of unique combinations depends on how the letters are
    repeated.

    >>> s1 = 'abracadabra'
    >>> s2 = 'banana tree'
    >>> count(combinations, s1), count(multiset_combinations, s1)
    (165, 23)
    >>> count(combinations, s2), count(multiset_combinations, s2)
    (165, 54)

    """
    from sympy.core.sorting import ordered
    if g is None:
        if isinstance(m, dict):
            if any(as_int(v) < 0 for v in m.values()):
                raise ValueError('counts cannot be negative')
            N = sum(m.values())
            if n > N:
                return
            g = [[k, m[k]] for k in ordered(m)]
        else:
            m = list(m)
            N = len(m)
            if n > N:
                return
            try:
                m = multiset(m)
                g = [(k, m[k]) for k in ordered(m)]
            except TypeError:
                m = list(ordered(m))
                g = [list(i) for i in group(m, multiple=False)]
        del m
    else:
        # not checking counts since g is intended for internal use
        N = sum(v for k, v in g)
    if n > N or not n:
        yield []
    else:
        for i, (k, v) in enumerate(g):
            if v >= n:
                yield [k]*n
                v = n - 1
            for v in range(min(n, v), 0, -1):
                for j in multiset_combinations(None, n - v, g[i + 1:]):
                    rv = [k]*v + j
                    if len(rv) == n:
                        yield rv

def multiset_permutations(m, size=None, g=None):
    """
    Return the unique permutations of multiset ``m``.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset_permutations
    >>> from sympy import factorial
    >>> [''.join(i) for i in multiset_permutations('aab')]
    ['aab', 'aba', 'baa']
    >>> factorial(len('banana'))
    720
    >>> len(list(multiset_permutations('banana')))
    60
    """
    from sympy.core.sorting import ordered
    if g is None:
        if isinstance(m, dict):
            if any(as_int(v) < 0 for v in m.values()):
                raise ValueError('counts cannot be negative')
            g = [[k, m[k]] for k in ordered(m)]
        else:
            m = list(ordered(m))
            g = [list(i) for i in group(m, multiple=False)]
        del m
    do = [gi for gi in g if gi[1] > 0]
    SUM = sum(gi[1] for gi in do)
    if not do or size is not None and (size > SUM or size < 1):
        if not do and size is None or size == 0:
            yield []
        return
    elif size == 1:
        for k, v in do:
            yield [k]
    elif len(do) == 1:
        k, v = do[0]
        v = v if size is None else (size if size <= v else 0)
        yield [k for i in range(v)]
    elif all(v == 1 for k, v in do):
        for p in permutations([k for k, v in do], size):
            yield list(p)
    else:
        size = size if size is not None else SUM
        for i, (k, v) in enumerate(do):
            do[i][1] -= 1
            for j in multiset_permutations(None, size - 1, do):
                if j:
                    yield [k] + j
            do[i][1] += 1


def _partition(seq, vector, m=None):
    """
    Return the partition of seq as specified by the partition vector.

    Examples
    ========

    >>> from sympy.utilities.iterables import _partition
    >>> _partition('abcde', [1, 0, 1, 2, 0])
    [['b', 'e'], ['a', 'c'], ['d']]

    Specifying the number of bins in the partition is optional:

    >>> _partition('abcde', [1, 0, 1, 2, 0], 3)
    [['b', 'e'], ['a', 'c'], ['d']]

    The output of _set_partitions can be passed as follows:

    >>> output = (3, [1, 0, 1, 2, 0])
    >>> _partition('abcde', *output)
    [['b', 'e'], ['a', 'c'], ['d']]

    See Also
    ========

    combinatorics.partitions.Partition.from_rgs

    """
    if m is None:
        m = max(vector) + 1
    elif isinstance(vector, int):  # entered as m, vector
        vector, m = m, vector
    p = [[] for i in range(m)]
    for i, v in enumerate(vector):
        p[v].append(seq[i])
    return p


def _set_partitions(n):
    """Cycle through all partitions of n elements, yielding the
    current number of partitions, ``m``, and a mutable list, ``q``
    such that ``element[i]`` is in part ``q[i]`` of the partition.

    NOTE: ``q`` is modified in place and generally should not be changed
    between function calls.

    Examples
    ========

    >>> from sympy.utilities.iterables import _set_partitions, _partition
    >>> for m, q in _set_partitions(3):
    ...     print('%s %s %s' % (m, q, _partition('abc', q, m)))
    1 [0, 0, 0] [['a', 'b', 'c']]
    2 [0, 0, 1] [['a', 'b'], ['c']]
    2 [0, 1, 0] [['a', 'c'], ['b']]
    2 [0, 1, 1] [['a'], ['b', 'c']]
    3 [0, 1, 2] [['a'], ['b'], ['c']]

    Notes
    =====

    This algorithm is similar to, and solves the same problem as,
    Algorithm 7.2.1.5H, from volume 4A of Knuth's The Art of Computer
    Programming.  Knuth uses the term "restricted growth string" where
    this code refers to a "partition vector". In each case, the meaning is
    the same: the value in the ith element of the vector specifies to
    which part the ith set element is to be assigned.

    At the lowest level, this code implements an n-digit big-endian
    counter (stored in the array q) which is incremented (with carries) to
    get the next partition in the sequence.  A special twist is that a
    digit is constrained to be at most one greater than the maximum of all
    the digits to the left of it.  The array p maintains this maximum, so
    that the code can efficiently decide when a digit can be incremented
    in place or whether it needs to be reset to 0 and trigger a carry to
    the next digit.  The enumeration starts with all the digits 0 (which
    corresponds to all the set elements being assigned to the same 0th
    part), and ends with 0123...n, which corresponds to each set element
    being assigned to a different, singleton, part.

    This routine was rewritten to use 0-based lists while trying to
    preserve the beauty and efficiency of the original algorithm.

    References
    ==========

    .. [1] Nijenhuis, Albert and Wilf, Herbert. (1978) Combinatorial Algorithms,
        2nd Ed, p 91, algorithm "nexequ". Available online from
        https://www.math.upenn.edu/~wilf/website/CombAlgDownld.html (viewed
        November 17, 2012).

    """
    p = [0]*n
    q = [0]*n
    nc = 1
    yield nc, q
    while nc != n:
        m = n
        while 1:
            m -= 1
            i = q[m]
            if p[i] != 1:
                break
            q[m] = 0
        i += 1
        q[m] = i
        m += 1
        nc += m - n
        p[0] += n - m
        if i == nc:
            p[nc] = 0
            nc += 1
        p[i - 1] -= 1
        p[i] += 1
        yield nc, q


def multiset_partitions(multiset, m=None):
    """
    Return unique partitions of the given multiset (in list form).
    If ``m`` is None, all multisets will be returned, otherwise only
    partitions with ``m`` parts will be returned.

    If ``multiset`` is an integer, a range [0, 1, ..., multiset - 1]
    will be supplied.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset_partitions
    >>> list(multiset_partitions([1, 2, 3, 4], 2))
    [[[1, 2, 3], [4]], [[1, 2, 4], [3]], [[1, 2], [3, 4]],
    [[1, 3, 4], [2]], [[1, 3], [2, 4]], [[1, 4], [2, 3]],
    [[1], [2, 3, 4]]]
    >>> list(multiset_partitions([1, 2, 3, 4], 1))
    [[[1, 2, 3, 4]]]

    Only unique partitions are returned and these will be returned in a
    canonical order regardless of the order of the input:

    >>> a = [1, 2, 2, 1]
    >>> ans = list(multiset_partitions(a, 2))
    >>> a.sort()
    >>> list(multiset_partitions(a, 2)) == ans
    True
    >>> a = range(3, 1, -1)
    >>> (list(multiset_partitions(a)) ==
    ...  list(multiset_partitions(sorted(a))))
    True

    If m is omitted then all partitions will be returned:

    >>> list(multiset_partitions([1, 1, 2]))
    [[[1, 1, 2]], [[1, 1], [2]], [[1, 2], [1]], [[1], [1], [2]]]
    >>> list(multiset_partitions([1]*3))
    [[[1, 1, 1]], [[1], [1, 1]], [[1], [1], [1]]]

    Counting
    ========

    The number of partitions of a set is given by the bell number:

    >>> from sympy import bell
    >>> len(list(multiset_partitions(5))) == bell(5) == 52
    True

    The number of partitions of length k from a set of size n is given by the
    Stirling Number of the 2nd kind:

    >>> from sympy.functions.combinatorial.numbers import stirling
    >>> stirling(5, 2) == len(list(multiset_partitions(5, 2))) == 15
    True

    These comments on counting apply to *sets*, not multisets.

    Notes
    =====

    When all the elements are the same in the multiset, the order
    of the returned partitions is determined by the ``partitions``
    routine. If one is counting partitions then it is better to use
    the ``nT`` function.

    See Also
    ========

    partitions
    sympy.combinatorics.partitions.Partition
    sympy.combinatorics.partitions.IntegerPartition
    sympy.functions.combinatorial.numbers.nT

    """
    # This function looks at the supplied input and dispatches to
    # several special-case routines as they apply.
    if isinstance(multiset, int):
        n = multiset
        if m and m > n:
            return
        multiset = list(range(n))
        if m == 1:
            yield [multiset[:]]
            return

        # If m is not None, it can sometimes be faster to use
        # MultisetPartitionTraverser.enum_range() even for inputs
        # which are sets.  Since the _set_partitions code is quite
        # fast, this is only advantageous when the overall set
        # partitions outnumber those with the desired number of parts
        # by a large factor.  (At least 60.)  Such a switch is not
        # currently implemented.
        for nc, q in _set_partitions(n):
            if m is None or nc == m:
                rv = [[] for i in range(nc)]
                for i in range(n):
                    rv[q[i]].append(multiset[i])
                yield rv
        return

    if len(multiset) == 1 and isinstance(multiset, str):
        multiset = [multiset]

    if not has_variety(multiset):
        # Only one component, repeated n times.  The resulting
        # partitions correspond to partitions of integer n.
        n = len(multiset)
        if m and m > n:
            return
        if m == 1:
            yield [multiset[:]]
            return
        x = multiset[:1]
        for size, p in partitions(n, m, size=True):
            if m is None or size == m:
                rv = []
                for k in sorted(p):
                    rv.extend([x*k]*p[k])
                yield rv
    else:
        from sympy.core.sorting import ordered
        multiset = list(ordered(multiset))
        n = len(multiset)
        if m and m > n:
            return
        if m == 1:
            yield [multiset[:]]
            return

        # Split the information of the multiset into two lists -
        # one of the elements themselves, and one (of the same length)
        # giving the number of repeats for the corresponding element.
        elements, multiplicities = zip(*group(multiset, False))

        if len(elements) < len(multiset):
            # General case - multiset with more than one distinct element
            # and at least one element repeated more than once.
            if m:
                mpt = MultisetPartitionTraverser()
                for state in mpt.enum_range(multiplicities, m-1, m):
                    yield list_visitor(state, elements)
            else:
                for state in multiset_partitions_taocp(multiplicities):
                    yield list_visitor(state, elements)
        else:
            # Set partitions case - no repeated elements. Pretty much
            # same as int argument case above, with same possible, but
            # currently unimplemented optimization for some cases when
            # m is not None
            for nc, q in _set_partitions(n):
                if m is None or nc == m:
                    rv = [[] for i in range(nc)]
                    for i in range(n):
                        rv[q[i]].append(i)
                    yield [[multiset[j] for j in i] for i in rv]


def partitions(n, m=None, k=None, size=False):
    """Generate all partitions of positive integer, n.

    Each partition is represented as a dictionary, mapping an integer
    to the number of copies of that integer in the partition.  For example,
    the first partition of 4 returned is {4: 1}, "4: one of them".

    Parameters
    ==========
    n : int
    m : int, optional
        limits number of parts in partition (mnemonic: m, maximum parts)
    k : int, optional
        limits the numbers that are kept in the partition (mnemonic: k, keys)
    size : bool, default: False
        If ``True``, (M, P) is returned where M is the sum of the
        multiplicities and P is the generated partition.
        If ``False``, only the generated partition is returned.

    Examples
    ========

    >>> from sympy.utilities.iterables import partitions

    The numbers appearing in the partition (the key of the returned dict)
    are limited with k:

    >>> for p in partitions(6, k=2):  # doctest: +SKIP
    ...     print(p)
    {2: 3}
    {1: 2, 2: 2}
    {1: 4, 2: 1}
    {1: 6}

    The maximum number of parts in the partition (the sum of the values in
    the returned dict) are limited with m (default value, None, gives
    partitions from 1 through n):

    >>> for p in partitions(6, m=2):  # doctest: +SKIP
    ...     print(p)
    ...
    {6: 1}
    {1: 1, 5: 1}
    {2: 1, 4: 1}
    {3: 2}

    References
    ==========

    .. [1] modified from Tim Peter's version to allow for k and m values:
           https://code.activestate.com/recipes/218332-generator-for-integer-partitions/

    See Also
    ========

    sympy.combinatorics.partitions.Partition
    sympy.combinatorics.partitions.IntegerPartition

    """
    if (n <= 0 or
        m is not None and m < 1 or
        k is not None and k < 1 or
        m and k and m*k < n):
        # the empty set is the only way to handle these inputs
        # and returning {} to represent it is consistent with
        # the counting convention, e.g. nT(0) == 1.
        if size:
            yield 0, {}
        else:
            yield {}
        return

    if m is None:
        m = n
    else:
        m = min(m, n)
    k = min(k or n, n)

    n, m, k = as_int(n), as_int(m), as_int(k)
    q, r = divmod(n, k)
    ms = {k: q}
    keys = [k]  # ms.keys(), from largest to smallest
    if r:
        ms[r] = 1
        keys.append(r)
    room = m - q - bool(r)
    if size:
        yield sum(ms.values()), ms.copy()
    else:
        yield ms.copy()

    while keys != [1]:
        # Reuse any 1's.
        if keys[-1] == 1:
            del keys[-1]
            reuse = ms.pop(1)
            room += reuse
        else:
            reuse = 0

        while 1:
            # Let i be the smallest key larger than 1.  Reuse one
            # instance of i.
            i = keys[-1]
            newcount = ms[i] = ms[i] - 1
            reuse += i
            if newcount == 0:
                del keys[-1], ms[i]
            room += 1

            # Break the remainder into pieces of size i-1.
            i -= 1
            q, r = divmod(reuse, i)
            need = q + bool(r)
            if need > room:
                if not keys:
                    return
                continue

            ms[i] = q
            keys.append(i)
            if r:
                ms[r] = 1
                keys.append(r)
            break
        room -= need
        if size:
            yield sum(ms.values()), ms.copy()
        else:
            yield ms.copy()


def ordered_partitions(n, m=None, sort=True):
    """Generates ordered partitions of integer *n*.

    Parameters
    ==========
    n : int
    m : int, optional
        The default value gives partitions of all sizes else only
        those with size m. In addition, if *m* is not None then
        partitions are generated *in place* (see examples).
    sort : bool, default: True
        Controls whether partitions are
        returned in sorted order when *m* is not None; when False,
        the partitions are returned as fast as possible with elements
        sorted, but when m|n the partitions will not be in
        ascending lexicographical order.

    Examples
    ========

    >>> from sympy.utilities.iterables import ordered_partitions

    All partitions of 5 in ascending lexicographical:

    >>> for p in ordered_partitions(5):
    ...     print(p)
    [1, 1, 1, 1, 1]
    [1, 1, 1, 2]
    [1, 1, 3]
    [1, 2, 2]
    [1, 4]
    [2, 3]
    [5]

    Only partitions of 5 with two parts:

    >>> for p in ordered_partitions(5, 2):
    ...     print(p)
    [1, 4]
    [2, 3]

    When ``m`` is given, a given list objects will be used more than
    once for speed reasons so you will not see the correct partitions
    unless you make a copy of each as it is generated:

    >>> [p for p in ordered_partitions(7, 3)]
    [[1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 2, 2]]
    >>> [list(p) for p in ordered_partitions(7, 3)]
    [[1, 1, 5], [1, 2, 4], [1, 3, 3], [2, 2, 3]]

    When ``n`` is a multiple of ``m``, the elements are still sorted
    but the partitions themselves will be *unordered* if sort is False;
    the default is to return them in ascending lexicographical order.

    >>> for p in ordered_partitions(6, 2):
    ...     print(p)
    [1, 5]
    [2, 4]
    [3, 3]

    But if speed is more important than ordering, sort can be set to
    False:

    >>> for p in ordered_partitions(6, 2, sort=False):
    ...     print(p)
    [1, 5]
    [3, 3]
    [2, 4]

    References
    ==========

    .. [1] Generating Integer Partitions, [online],
        Available: https://jeromekelleher.net/generating-integer-partitions.html
    .. [2] Jerome Kelleher and Barry O'Sullivan, "Generating All
        Partitions: A Comparison Of Two Encodings", [online],
        Available: https://arxiv.org/pdf/0909.2331v2.pdf
    """
    if n < 1 or m is not None and m < 1:
        # the empty set is the only way to handle these inputs
        # and returning {} to represent it is consistent with
        # the counting convention, e.g. nT(0) == 1.
        yield []
        return

    if m is None:
        # The list `a`'s leading elements contain the partition in which
        # y is the biggest element and x is either the same as y or the
        # 2nd largest element; v and w are adjacent element indices
        # to which x and y are being assigned, respectively.
        a = [1]*n
        y = -1
        v = n
        while v > 0:
            v -= 1
            x = a[v] + 1
            while y >= 2 * x:
                a[v] = x
                y -= x
                v += 1
            w = v + 1
            while x <= y:
                a[v] = x
                a[w] = y
                yield a[:w + 1]
                x += 1
                y -= 1
            a[v] = x + y
            y = a[v] - 1
            yield a[:w]
    elif m == 1:
        yield [n]
    elif n == m:
        yield [1]*n
    else:
        # recursively generate partitions of size m
        for b in range(1, n//m + 1):
            a = [b]*m
            x = n - b*m
            if not x:
                if sort:
                    yield a
            elif not sort and x <= m:
                for ax in ordered_partitions(x, sort=False):
                    mi = len(ax)
                    a[-mi:] = [i + b for i in ax]
                    yield a
                    a[-mi:] = [b]*mi
            else:
                for mi in range(1, m):
                    for ax in ordered_partitions(x, mi, sort=True):
                        a[-mi:] = [i + b for i in ax]
                        yield a
                        a[-mi:] = [b]*mi


def binary_partitions(n):
    """
    Generates the binary partition of *n*.

    A binary partition consists only of numbers that are
    powers of two. Each step reduces a `2^{k+1}` to `2^k` and
    `2^k`. Thus 16 is converted to 8 and 8.

    Examples
    ========

    >>> from sympy.utilities.iterables import binary_partitions
    >>> for i in binary_partitions(5):
    ...     print(i)
    ...
    [4, 1]
    [2, 2, 1]
    [2, 1, 1, 1]
    [1, 1, 1, 1, 1]

    References
    ==========

    .. [1] TAOCP 4, section 7.2.1.5, problem 64

    """
    from math import ceil, log2
    power = int(2**(ceil(log2(n))))
    acc = 0
    partition = []
    while power:
        if acc + power <= n:
            partition.append(power)
            acc += power
        power >>= 1

    last_num = len(partition) - 1 - (n & 1)
    while last_num >= 0:
        yield partition
        if partition[last_num] == 2:
            partition[last_num] = 1
            partition.append(1)
            last_num -= 1
            continue
        partition.append(1)
        partition[last_num] >>= 1
        x = partition[last_num + 1] = partition[last_num]
        last_num += 1
        while x > 1:
            if x <= len(partition) - last_num - 1:
                del partition[-x + 1:]
                last_num += 1
                partition[last_num] = x
            else:
                x >>= 1
    yield [1]*n


def has_dups(seq):
    """Return True if there are any duplicate elements in ``seq``.

    Examples
    ========

    >>> from sympy import has_dups, Dict, Set
    >>> has_dups((1, 2, 1))
    True
    >>> has_dups(range(3))
    False
    >>> all(has_dups(c) is False for c in (set(), Set(), dict(), Dict()))
    True
    """
    from sympy.core.containers import Dict
    from sympy.sets.sets import Set
    if isinstance(seq, (dict, set, Dict, Set)):
        return False
    unique = set()
    try:
        return any(True for s in seq if s in unique or unique.add(s))
    except TypeError:
        return len(seq) != len(list(uniq(seq)))


def has_variety(seq):
    """Return True if there are any different elements in ``seq``.

    Examples
    ========

    >>> from sympy import has_variety

    >>> has_variety((1, 2, 1))
    True
    >>> has_variety((1, 1, 1))
    False
    """
    for i, s in enumerate(seq):
        if i == 0:
            sentinel = s
        else:
            if s != sentinel:
                return True
    return False


def uniq(seq, result=None):
    """
    Yield unique elements from ``seq`` as an iterator. The second
    parameter ``result``  is used internally; it is not necessary
    to pass anything for this.

    Note: changing the sequence during iteration will raise a
    RuntimeError if the size of the sequence is known; if you pass
    an iterator and advance the iterator you will change the
    output of this routine but there will be no warning.

    Examples
    ========

    >>> from sympy.utilities.iterables import uniq
    >>> dat = [1, 4, 1, 5, 4, 2, 1, 2]
    >>> type(uniq(dat)) in (list, tuple)
    False

    >>> list(uniq(dat))
    [1, 4, 5, 2]
    >>> list(uniq(x for x in dat))
    [1, 4, 5, 2]
    >>> list(uniq([[1], [2, 1], [1]]))
    [[1], [2, 1]]
    """
    try:
        n = len(seq)
    except TypeError:
        n = None
    def check():
        # check that size of seq did not change during iteration;
        # if n == None the object won't support size changing, e.g.
        # an iterator can't be changed
        if n is not None and len(seq) != n:
            raise RuntimeError('sequence changed size during iteration')
    try:
        seen = set()
        result = result or []
        for i, s in enumerate(seq):
            if not (s in seen or seen.add(s)):
                yield s
                check()
    except TypeError:
        if s not in result:
            yield s
            check()
            result.append(s)
        if hasattr(seq, '__getitem__'):
            yield from uniq(seq[i + 1:], result)
        else:
            yield from uniq(seq, result)


def generate_bell(n):
    """Return permutations of [0, 1, ..., n - 1] such that each permutation
    differs from the last by the exchange of a single pair of neighbors.
    The ``n!`` permutations are returned as an iterator. In order to obtain
    the next permutation from a random starting permutation, use the
    ``next_trotterjohnson`` method of the Permutation class (which generates
    the same sequence in a different manner).

    Examples
    ========

    >>> from itertools import permutations
    >>> from sympy.utilities.iterables import generate_bell
    >>> from sympy import zeros, Matrix

    This is the sort of permutation used in the ringing of physical bells,
    and does not produce permutations in lexicographical order. Rather, the
    permutations differ from each other by exactly one inversion, and the
    position at which the swapping occurs varies periodically in a simple
    fashion. Consider the first few permutations of 4 elements generated
    by ``permutations`` and ``generate_bell``:

    >>> list(permutations(range(4)))[:5]
    [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1), (0, 3, 1, 2)]
    >>> list(generate_bell(4))[:5]
    [(0, 1, 2, 3), (0, 1, 3, 2), (0, 3, 1, 2), (3, 0, 1, 2), (3, 0, 2, 1)]

    Notice how the 2nd and 3rd lexicographical permutations have 3 elements
    out of place whereas each "bell" permutation always has only two
    elements out of place relative to the previous permutation (and so the
    signature (+/-1) of a permutation is opposite of the signature of the
    previous permutation).

    How the position of inversion varies across the elements can be seen
    by tracing out where the largest number appears in the permutations:

    >>> m = zeros(4, 24)
    >>> for i, p in enumerate(generate_bell(4)):
    ...     m[:, i] = Matrix([j - 3 for j in list(p)])  # make largest zero
    >>> m.print_nonzero('X')
    [XXX  XXXXXX  XXXXXX  XXX]
    [XX XX XXXX XX XXXX XX XX]
    [X XXXX XX XXXX XX XXXX X]
    [ XXXXXX  XXXXXX  XXXXXX ]

    See Also
    ========

    sympy.combinatorics.permutations.Permutation.next_trotterjohnson

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Method_ringing

    .. [2] https://stackoverflow.com/questions/4856615/recursive-permutation/4857018

    .. [3] https://web.archive.org/web/20160313023044/http://programminggeeks.com/bell-algorithm-for-permutation/

    .. [4] https://en.wikipedia.org/wiki/Steinhaus%E2%80%93Johnson%E2%80%93Trotter_algorithm

    .. [5] Generating involutions, derangements, and relatives by ECO
           Vincent Vajnovszki, DMTCS vol 1 issue 12, 2010

    """
    n = as_int(n)
    if n < 1:
        raise ValueError('n must be a positive integer')
    if n == 1:
        yield (0,)
    elif n == 2:
        yield (0, 1)
        yield (1, 0)
    elif n == 3:
        yield from [(0, 1, 2), (0, 2, 1), (2, 0, 1), (2, 1, 0), (1, 2, 0), (1, 0, 2)]
    else:
        m = n - 1
        op = [0] + [-1]*m
        l = list(range(n))
        while True:
            yield tuple(l)
            # find biggest element with op
            big = None, -1  # idx, value
            for i in range(n):
                if op[i] and l[i] > big[1]:
                    big = i, l[i]
            i, _ = big
            if i is None:
                break  # there are no ops left
            # swap it with neighbor in the indicated direction
            j = i + op[i]
            l[i], l[j] = l[j], l[i]
            op[i], op[j] = op[j], op[i]
            # if it landed at the end or if the neighbor in the same
            # direction is bigger then turn off op
            if j == 0 or j == m or l[j + op[j]] > l[j]:
                op[j] = 0
            # any element bigger to the left gets +1 op
            for i in range(j):
                if l[i] > l[j]:
                    op[i] = 1
            # any element bigger to the right gets -1 op
            for i in range(j + 1, n):
                if l[i] > l[j]:
                    op[i] = -1


def generate_involutions(n):
    """
    Generates involutions.

    An involution is a permutation that when multiplied
    by itself equals the identity permutation. In this
    implementation the involutions are generated using
    Fixed Points.

    Alternatively, an involution can be considered as
    a permutation that does not contain any cycles with
    a length that is greater than two.

    Examples
    ========

    >>> from sympy.utilities.iterables import generate_involutions
    >>> list(generate_involutions(3))
    [(0, 1, 2), (0, 2, 1), (1, 0, 2), (2, 1, 0)]
    >>> len(list(generate_involutions(4)))
    10

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PermutationInvolution.html

    """
    idx = list(range(n))
    for p in permutations(idx):
        for i in idx:
            if p[p[i]] != i:
                break
        else:
            yield p


def multiset_derangements(s):
    """Generate derangements of the elements of s *in place*.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset_derangements, uniq

    Because the derangements of multisets (not sets) are generated
    in place, copies of the return value must be made if a collection
    of derangements is desired or else all values will be the same:

    >>> list(uniq([i for i in multiset_derangements('1233')]))
    [[None, None, None, None]]
    >>> [i.copy() for i in multiset_derangements('1233')]
    [['3', '3', '1', '2'], ['3', '3', '2', '1']]
    >>> [''.join(i) for i in multiset_derangements('1233')]
    ['3312', '3321']
    """
    from sympy.core.sorting import ordered
    # create multiset dictionary of hashable elements or else
    # remap elements to integers
    try:
        ms = multiset(s)
    except TypeError:
        # give each element a canonical integer value
        key = dict(enumerate(ordered(uniq(s))))
        h = []
        for si in s:
            for k in key:
                if key[k] == si:
                    h.append(k)
                    break
        for i in multiset_derangements(h):
            yield [key[j] for j in i]
        return

    mx = max(ms.values())  # max repetition of any element
    n = len(s)  # the number of elements

    ## special cases

    # 1) one element has more than half the total cardinality of s: no
    # derangements are possible.
    if mx*2 > n:
        return

    # 2) all elements appear once: singletons
    if len(ms) == n:
        yield from _set_derangements(s)
        return

    # find the first element that is repeated the most to place
    # in the following two special cases where the selection
    # is unambiguous: either there are two elements with multiplicity
    # of mx or else there is only one with multiplicity mx
    for M in ms:
        if ms[M] == mx:
            break

    inonM = [i for i in range(n) if s[i] != M]  # location of non-M
    iM = [i for i in range(n) if s[i] == M]  # locations of M
    rv = [None]*n

    # 3) half are the same
    if 2*mx == n:
        # M goes into non-M locations
        for i in inonM:
            rv[i] = M
        # permutations of non-M go to M locations
        for p in multiset_permutations([s[i] for i in inonM]):
            for i, pi in zip(iM, p):
                rv[i] = pi
            yield rv
        # clean-up (and encourages proper use of routine)
        rv[:] = [None]*n
        return

    # 4) single repeat covers all but 1 of the non-repeats:
    # if there is one repeat then the multiset of the values
    # of ms would be {mx: 1, 1: n - mx}, i.e. there would
    # be n - mx + 1 values with the condition that n - 2*mx = 1
    if n - 2*mx == 1 and len(ms.values()) == n - mx + 1:
        for i, i1 in enumerate(inonM):
            ifill = inonM[:i] + inonM[i+1:]
            for j in ifill:
                rv[j] = M
            for p in permutations([s[j] for j in ifill]):
                rv[i1] = s[i1]
                for j, pi in zip(iM, p):
                    rv[j] = pi
                k = i1
                for j in iM:
                    rv[j], rv[k] = rv[k], rv[j]
                    yield rv
                    k = j
        # clean-up (and encourages proper use of routine)
        rv[:] = [None]*n
        return

    ## general case is handled with 3 helpers:
    #    1) `finish_derangements` will place the last two elements
    #       which have arbitrary multiplicities, e.g. for multiset
    #       {c: 3, a: 2, b: 2}, the last two elements are a and b
    #    2) `iopen` will tell where a given element can be placed
    #    3) `do` will recursively place elements into subsets of
    #        valid locations

    def finish_derangements():
        """Place the last two elements into the partially completed
        derangement, and yield the results.
        """

        a = take[1][0]  # penultimate element
        a_ct = take[1][1]
        b = take[0][0]  # last element to be placed
        b_ct = take[0][1]

        # split the indexes of the not-already-assigned elements of rv into
        # three categories
        forced_a = []  # positions which must have an a
        forced_b = []  # positions which must have a b
        open_free = []  # positions which could take either
        for i in range(len(s)):
            if rv[i] is None:
                if s[i] == a:
                    forced_b.append(i)
                elif s[i] == b:
                    forced_a.append(i)
                else:
                    open_free.append(i)

        if len(forced_a) > a_ct or len(forced_b) > b_ct:
            # No derangement possible
            return

        for i in forced_a:
            rv[i] = a
        for i in forced_b:
            rv[i] = b
        for a_place in combinations(open_free, a_ct - len(forced_a)):
            for a_pos in a_place:
                rv[a_pos] = a
            for i in open_free:
                if rv[i] is None:  # anything not in the subset is set to b
                    rv[i] = b
            yield rv
            # Clean up/undo the final placements
            for i in open_free:
                rv[i] = None

        # additional cleanup - clear forced_a, forced_b
        for i in forced_a:
            rv[i] = None
        for i in forced_b:
            rv[i] = None

    def iopen(v):
        # return indices at which element v can be placed in rv:
        # locations which are not already occupied if that location
        # does not already contain v in the same location of s
        return [i for i in range(n) if rv[i] is None and s[i] != v]

    def do(j):
        if j == 1:
            # handle the last two elements (regardless of multiplicity)
            # with a special method
            yield from finish_derangements()
        else:
            # place the mx elements of M into a subset of places
            # into which it can be replaced
            M, mx = take[j]
            for i in combinations(iopen(M), mx):
                # place M
                for ii in i:
                    rv[ii] = M
                # recursively place the next element
                yield from do(j - 1)
                # mark positions where M was placed as once again
                # open for placement of other elements
                for ii in i:
                    rv[ii] = None

    # process elements in order of canonically decreasing multiplicity
    take = sorted(ms.items(), key=lambda x:(x[1], x[0]))
    yield from do(len(take) - 1)
    rv[:] = [None]*n


def random_derangement(t, choice=None, strict=True):
    """Return a list of elements in which none are in the same positions
    as they were originally. If an element fills more than half of the positions
    then an error will be raised since no derangement is possible. To obtain
    a derangement of as many items as possible--with some of the most numerous
    remaining in their original positions--pass `strict=False`. To produce a
    pseudorandom derangment, pass a pseudorandom selector like `choice` (see
    below).

    Examples
    ========

    >>> from sympy.utilities.iterables import random_derangement
    >>> t = 'SymPy: a CAS in pure Python'
    >>> d = random_derangement(t)
    >>> all(i != j for i, j in zip(d, t))
    True

    A predictable result can be obtained by using a pseudorandom
    generator for the choice:

    >>> from sympy.core.random import seed, choice as c
    >>> seed(1)
    >>> d = [''.join(random_derangement(t, c)) for i in range(5)]
    >>> assert len(set(d)) != 1  # we got different values

    By reseeding, the same sequence can be obtained:

    >>> seed(1)
    >>> d2 = [''.join(random_derangement(t, c)) for i in range(5)]
    >>> assert d == d2
    """
    if choice is None:
        import secrets
        choice = secrets.choice
    def shuffle(rv):
        '''Knuth shuffle'''
        for i in range(len(rv) - 1, 0, -1):
            x = choice(rv[:i + 1])
            j = rv.index(x)
            rv[i], rv[j] = rv[j], rv[i]
    def pick(rv, n):
        '''shuffle rv and return the first n values
        '''
        shuffle(rv)
        return rv[:n]
    ms = multiset(t)
    tot = len(t)
    ms = sorted(ms.items(), key=lambda x: x[1])
  # if there are not enough spaces for the most
  # plentiful element to move to then some of them
  # will have to stay in place
    M, mx = ms[-1]
    n = len(t)
    xs = 2*mx - tot
    if xs > 0:
        if strict:
            raise ValueError('no derangement possible')
        opts = [i for (i, c) in enumerate(t) if c == ms[-1][0]]
        pick(opts, xs)
        stay = sorted(opts[:xs])
        rv = list(t)
        for i in reversed(stay):
            rv.pop(i)
        rv = random_derangement(rv, choice)
        for i in stay:
            rv.insert(i, ms[-1][0])
        return ''.join(rv) if type(t) is str else rv
  # the normal derangement calculated from here
    if n == len(ms):
      # approx 1/3 will succeed
        rv = list(t)
        while True:
            shuffle(rv)
            if all(i != j for i,j in zip(rv, t)):
                break
    else:
      # general case
        rv = [None]*n
        while True:
            j = 0
            while j > -len(ms):  # do most numerous first
                j -= 1
                e, c = ms[j]
                opts = [i for i in range(n) if rv[i] is None and t[i] != e]
                if len(opts) < c:
                    for i in range(n):
                        rv[i] = None
                    break # try again
                pick(opts, c)
                for i in range(c):
                    rv[opts[i]] = e
            else:
                return rv
    return rv


def _set_derangements(s):
    """
    yield derangements of items in ``s`` which are assumed to contain
    no repeated elements
    """
    if len(s) < 2:
        return
    if len(s) == 2:
        yield [s[1], s[0]]
        return
    if len(s) == 3:
        yield [s[1], s[2], s[0]]
        yield [s[2], s[0], s[1]]
        return
    for p in permutations(s):
        if not any(i == j for i, j in zip(p, s)):
            yield list(p)


def generate_derangements(s):
    """
    Return unique derangements of the elements of iterable ``s``.

    Examples
    ========

    >>> from sympy.utilities.iterables import generate_derangements
    >>> list(generate_derangements([0, 1, 2]))
    [[1, 2, 0], [2, 0, 1]]
    >>> list(generate_derangements([0, 1, 2, 2]))
    [[2, 2, 0, 1], [2, 2, 1, 0]]
    >>> list(generate_derangements([0, 1, 1]))
    []

    See Also
    ========

    sympy.functions.combinatorial.factorials.subfactorial

    """
    if not has_dups(s):
        yield from _set_derangements(s)
    else:
        for p in multiset_derangements(s):
            yield list(p)


def necklaces(n, k, free=False):
    """
    A routine to generate necklaces that may (free=True) or may not
    (free=False) be turned over to be viewed. The "necklaces" returned
    are comprised of ``n`` integers (beads) with ``k`` different
    values (colors). Only unique necklaces are returned.

    Examples
    ========

    >>> from sympy.utilities.iterables import necklaces, bracelets
    >>> def show(s, i):
    ...     return ''.join(s[j] for j in i)

    The "unrestricted necklace" is sometimes also referred to as a
    "bracelet" (an object that can be turned over, a sequence that can
    be reversed) and the term "necklace" is used to imply a sequence
    that cannot be reversed. So ACB == ABC for a bracelet (rotate and
    reverse) while the two are different for a necklace since rotation
    alone cannot make the two sequences the same.

    (mnemonic: Bracelets can be viewed Backwards, but Not Necklaces.)

    >>> B = [show('ABC', i) for i in bracelets(3, 3)]
    >>> N = [show('ABC', i) for i in necklaces(3, 3)]
    >>> set(N) - set(B)
    {'ACB'}

    >>> list(necklaces(4, 2))
    [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 1),
     (0, 1, 0, 1), (0, 1, 1, 1), (1, 1, 1, 1)]

    >>> [show('.o', i) for i in bracelets(4, 2)]
    ['....', '...o', '..oo', '.o.o', '.ooo', 'oooo']

    References
    ==========

    .. [1] https://mathworld.wolfram.com/Necklace.html

    .. [2] Frank Ruskey, Carla Savage, and Terry Min Yih Wang,
        Generating necklaces, Journal of Algorithms 13 (1992), 414-430;
        https://doi.org/10.1016/0196-6774(92)90047-G

    """
    # The FKM algorithm
    if k == 0 and n > 0:
        return
    a = [0]*n
    yield tuple(a)
    if n == 0:
        return
    while True:
        i = n - 1
        while a[i] == k - 1:
            i -= 1
            if i == -1:
                return
        a[i] += 1
        for j in range(n - i - 1):
            a[j + i + 1] = a[j]
        if n % (i + 1) == 0 and (not free or all(a <= a[j::-1] + a[-1:j:-1] for j in range(n - 1))):
            # No need to test j = n - 1.
            yield tuple(a)


def bracelets(n, k):
    """Wrapper to necklaces to return a free (unrestricted) necklace."""
    return necklaces(n, k, free=True)


def generate_oriented_forest(n):
    """
    This algorithm generates oriented forests.

    An oriented graph is a directed graph having no symmetric pair of directed
    edges. A forest is an acyclic graph, i.e., it has no cycles. A forest can
    also be described as a disjoint union of trees, which are graphs in which
    any two vertices are connected by exactly one simple path.

    Examples
    ========

    >>> from sympy.utilities.iterables import generate_oriented_forest
    >>> list(generate_oriented_forest(4))
    [[0, 1, 2, 3], [0, 1, 2, 2], [0, 1, 2, 1], [0, 1, 2, 0], \
    [0, 1, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 1, 0, 0], [0, 0, 0, 0]]

    References
    ==========

    .. [1] T. Beyer and S.M. Hedetniemi: constant time generation of
           rooted trees, SIAM J. Computing Vol. 9, No. 4, November 1980

    .. [2] https://stackoverflow.com/questions/1633833/oriented-forest-taocp-algorithm-in-python

    """
    P = list(range(-1, n))
    while True:
        yield P[1:]
        if P[n] > 0:
            P[n] = P[P[n]]
        else:
            for p in range(n - 1, 0, -1):
                if P[p] != 0:
                    target = P[p] - 1
                    for q in range(p - 1, 0, -1):
                        if P[q] == target:
                            break
                    offset = p - q
                    for i in range(p, n + 1):
                        P[i] = P[i - offset]
                    break
            else:
                break


def minlex(seq, directed=True, key=None):
    r"""
    Return the rotation of the sequence in which the lexically smallest
    elements appear first, e.g. `cba \rightarrow acb`.

    The sequence returned is a tuple, unless the input sequence is a string
    in which case a string is returned.

    If ``directed`` is False then the smaller of the sequence and the
    reversed sequence is returned, e.g. `cba \rightarrow abc`.

    If ``key`` is not None then it is used to extract a comparison key from each element in iterable.

    Examples
    ========

    >>> from sympy.combinatorics.polyhedron import minlex
    >>> minlex((1, 2, 0))
    (0, 1, 2)
    >>> minlex((1, 0, 2))
    (0, 2, 1)
    >>> minlex((1, 0, 2), directed=False)
    (0, 1, 2)

    >>> minlex('11010011000', directed=True)
    '00011010011'
    >>> minlex('11010011000', directed=False)
    '00011001011'

    >>> minlex(('bb', 'aaa', 'c', 'a'))
    ('a', 'bb', 'aaa', 'c')
    >>> minlex(('bb', 'aaa', 'c', 'a'), key=len)
    ('c', 'a', 'bb', 'aaa')

    """
    from sympy.functions.elementary.miscellaneous import Id
    if key is None: key = Id
    best = rotate_left(seq, least_rotation(seq, key=key))
    if not directed:
        rseq = seq[::-1]
        rbest = rotate_left(rseq, least_rotation(rseq, key=key))
        best = min(best, rbest, key=key)

    # Convert to tuple, unless we started with a string.
    return tuple(best) if not isinstance(seq, str) else best


def runs(seq, op=gt):
    """Group the sequence into lists in which successive elements
    all compare the same with the comparison operator, ``op``:
    op(seq[i + 1], seq[i]) is True from all elements in a run.

    Examples
    ========

    >>> from sympy.utilities.iterables import runs
    >>> from operator import ge
    >>> runs([0, 1, 2, 2, 1, 4, 3, 2, 2])
    [[0, 1, 2], [2], [1, 4], [3], [2], [2]]
    >>> runs([0, 1, 2, 2, 1, 4, 3, 2, 2], op=ge)
    [[0, 1, 2, 2], [1, 4], [3], [2, 2]]
    """
    cycles = []
    seq = iter(seq)
    try:
        run = [next(seq)]
    except StopIteration:
        return []
    while True:
        try:
            ei = next(seq)
        except StopIteration:
            break
        if op(ei, run[-1]):
            run.append(ei)
            continue
        else:
            cycles.append(run)
            run = [ei]
    if run:
        cycles.append(run)
    return cycles


def sequence_partitions(l, n, /):
    r"""Returns the partition of sequence $l$ into $n$ bins

    Explanation
    ===========

    Given the sequence $l_1 \cdots l_m \in V^+$ where
    $V^+$ is the Kleene plus of $V$

    The set of $n$ partitions of $l$ is defined as:

    .. math::
        \{(s_1, \cdots, s_n) | s_1 \in V^+, \cdots, s_n \in V^+,
        s_1 \cdots s_n = l_1 \cdots l_m\}

    Parameters
    ==========

    l : Sequence[T]
        A nonempty sequence of any Python objects

    n : int
        A positive integer

    Yields
    ======

    out : list[Sequence[T]]
        A list of sequences with concatenation equals $l$.
        This should conform with the type of $l$.

    Examples
    ========

    >>> from sympy.utilities.iterables import sequence_partitions
    >>> for out in sequence_partitions([1, 2, 3, 4], 2):
    ...     print(out)
    [[1], [2, 3, 4]]
    [[1, 2], [3, 4]]
    [[1, 2, 3], [4]]

    Notes
    =====

    This is modified version of EnricoGiampieri's partition generator
    from https://stackoverflow.com/questions/13131491/partition-n-items-into-k-bins-in-python-lazily

    See Also
    ========

    sequence_partitions_empty
    """
    # Asserting l is nonempty is done only for sanity check
    if n == 1 and l:
        yield [l]
        return
    for i in range(1, len(l)):
        for part in sequence_partitions(l[i:], n - 1):
            yield [l[:i]] + part


def sequence_partitions_empty(l, n, /):
    r"""Returns the partition of sequence $l$ into $n$ bins with
    empty sequence

    Explanation
    ===========

    Given the sequence $l_1 \cdots l_m \in V^*$ where
    $V^*$ is the Kleene star of $V$

    The set of $n$ partitions of $l$ is defined as:

    .. math::
        \{(s_1, \cdots, s_n) | s_1 \in V^*, \cdots, s_n \in V^*,
        s_1 \cdots s_n = l_1 \cdots l_m\}

    There are more combinations than :func:`sequence_partitions` because
    empty sequence can fill everywhere, so we try to provide different
    utility for this.

    Parameters
    ==========

    l : Sequence[T]
        A sequence of any Python objects (can be possibly empty)

    n : int
        A positive integer

    Yields
    ======

    out : list[Sequence[T]]
        A list of sequences with concatenation equals $l$.
        This should conform with the type of $l$.

    Examples
    ========

    >>> from sympy.utilities.iterables import sequence_partitions_empty
    >>> for out in sequence_partitions_empty([1, 2, 3, 4], 2):
    ...     print(out)
    [[], [1, 2, 3, 4]]
    [[1], [2, 3, 4]]
    [[1, 2], [3, 4]]
    [[1, 2, 3], [4]]
    [[1, 2, 3, 4], []]

    See Also
    ========

    sequence_partitions
    """
    if n < 1:
        return
    if n == 1:
        yield [l]
        return
    for i in range(0, len(l) + 1):
        for part in sequence_partitions_empty(l[i:], n - 1):
            yield [l[:i]] + part


def kbins(l, k, ordered=None):
    """
    Return sequence ``l`` partitioned into ``k`` bins.

    Examples
    ========

    The default is to give the items in the same order, but grouped
    into k partitions without any reordering:

    >>> from sympy.utilities.iterables import kbins
    >>> for p in kbins(list(range(5)), 2):
    ...     print(p)
    ...
    [[0], [1, 2, 3, 4]]
    [[0, 1], [2, 3, 4]]
    [[0, 1, 2], [3, 4]]
    [[0, 1, 2, 3], [4]]

    The ``ordered`` flag is either None (to give the simple partition
    of the elements) or is a 2 digit integer indicating whether the order of
    the bins and the order of the items in the bins matters. Given::

        A = [[0], [1, 2]]
        B = [[1, 2], [0]]
        C = [[2, 1], [0]]
        D = [[0], [2, 1]]

    the following values for ``ordered`` have the shown meanings::

        00 means A == B == C == D
        01 means A == B
        10 means A == D
        11 means A == A

    >>> for ordered_flag in [None, 0, 1, 10, 11]:
    ...     print('ordered = %s' % ordered_flag)
    ...     for p in kbins(list(range(3)), 2, ordered=ordered_flag):
    ...         print('     %s' % p)
    ...
    ordered = None
         [[0], [1, 2]]
         [[0, 1], [2]]
    ordered = 0
         [[0, 1], [2]]
         [[0, 2], [1]]
         [[0], [1, 2]]
    ordered = 1
         [[0], [1, 2]]
         [[0], [2, 1]]
         [[1], [0, 2]]
         [[1], [2, 0]]
         [[2], [0, 1]]
         [[2], [1, 0]]
    ordered = 10
         [[0, 1], [2]]
         [[2], [0, 1]]
         [[0, 2], [1]]
         [[1], [0, 2]]
         [[0], [1, 2]]
         [[1, 2], [0]]
    ordered = 11
         [[0], [1, 2]]
         [[0, 1], [2]]
         [[0], [2, 1]]
         [[0, 2], [1]]
         [[1], [0, 2]]
         [[1, 0], [2]]
         [[1], [2, 0]]
         [[1, 2], [0]]
         [[2], [0, 1]]
         [[2, 0], [1]]
         [[2], [1, 0]]
         [[2, 1], [0]]

    See Also
    ========

    partitions, multiset_partitions

    """
    if ordered is None:
        yield from sequence_partitions(l, k)
    elif ordered == 11:
        for pl in multiset_permutations(l):
            pl = list(pl)
            yield from sequence_partitions(pl, k)
    elif ordered == 00:
        yield from multiset_partitions(l, k)
    elif ordered == 10:
        for p in multiset_partitions(l, k):
            for perm in permutations(p):
                yield list(perm)
    elif ordered == 1:
        for kgot, p in partitions(len(l), k, size=True):
            if kgot != k:
                continue
            for li in multiset_permutations(l):
                rv = []
                i = j = 0
                li = list(li)
                for size, multiplicity in sorted(p.items()):
                    for m in range(multiplicity):
                        j = i + size
                        rv.append(li[i: j])
                        i = j
                yield rv
    else:
        raise ValueError(
            'ordered must be one of 00, 01, 10 or 11, not %s' % ordered)


def permute_signs(t):
    """Return iterator in which the signs of non-zero elements
    of t are permuted.

    Examples
    ========

    >>> from sympy.utilities.iterables import permute_signs
    >>> list(permute_signs((0, 1, 2)))
    [(0, 1, 2), (0, -1, 2), (0, 1, -2), (0, -1, -2)]
    """
    for signs in product(*[(1, -1)]*(len(t) - t.count(0))):
        signs = list(signs)
        yield type(t)([i*signs.pop() if i else i for i in t])


def signed_permutations(t):
    """Return iterator in which the signs of non-zero elements
    of t and the order of the elements are permuted and all
    returned values are unique.

    Examples
    ========

    >>> from sympy.utilities.iterables import signed_permutations
    >>> list(signed_permutations((0, 1, 2)))
    [(0, 1, 2), (0, -1, 2), (0, 1, -2), (0, -1, -2), (0, 2, 1),
    (0, -2, 1), (0, 2, -1), (0, -2, -1), (1, 0, 2), (-1, 0, 2),
    (1, 0, -2), (-1, 0, -2), (1, 2, 0), (-1, 2, 0), (1, -2, 0),
    (-1, -2, 0), (2, 0, 1), (-2, 0, 1), (2, 0, -1), (-2, 0, -1),
    (2, 1, 0), (-2, 1, 0), (2, -1, 0), (-2, -1, 0)]
    """
    return (type(t)(i) for j in multiset_permutations(t)
        for i in permute_signs(j))


def rotations(s, dir=1):
    """Return a generator giving the items in s as list where
    each subsequent list has the items rotated to the left (default)
    or right (``dir=-1``) relative to the previous list.

    Examples
    ========

    >>> from sympy import rotations
    >>> list(rotations([1,2,3]))
    [[1, 2, 3], [2, 3, 1], [3, 1, 2]]
    >>> list(rotations([1,2,3], -1))
    [[1, 2, 3], [3, 1, 2], [2, 3, 1]]
    """
    seq = list(s)
    for i in range(len(seq)):
        yield seq
        seq = rotate_left(seq, dir)


def roundrobin(*iterables):
    """roundrobin recipe taken from itertools documentation:
    https://docs.python.org/3/library/itertools.html#itertools-recipes

    roundrobin('ABC', 'D', 'EF') --> A D E B F C

    Recipe credited to George Sakkis
    """
    nexts = cycle(iter(it).__next__ for it in iterables)

    pending = len(iterables)
    while pending:
        try:
            for nxt in nexts:
                yield nxt()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))



class NotIterable:
    """
    Use this as mixin when creating a class which is not supposed to
    return true when iterable() is called on its instances because
    calling list() on the instance, for example, would result in
    an infinite loop.
    """
    pass


def iterable(i, exclude=(str, dict, NotIterable)):
    """
    Return a boolean indicating whether ``i`` is SymPy iterable.
    True also indicates that the iterator is finite, e.g. you can
    call list(...) on the instance.

    When SymPy is working with iterables, it is almost always assuming
    that the iterable is not a string or a mapping, so those are excluded
    by default. If you want a pure Python definition, make exclude=None. To
    exclude multiple items, pass them as a tuple.

    You can also set the _iterable attribute to True or False on your class,
    which will override the checks here, including the exclude test.

    As a rule of thumb, some SymPy functions use this to check if they should
    recursively map over an object. If an object is technically iterable in
    the Python sense but does not desire this behavior (e.g., because its
    iteration is not finite, or because iteration might induce an unwanted
    computation), it should disable it by setting the _iterable attribute to False.

    See also: is_sequence

    Examples
    ========

    >>> from sympy.utilities.iterables import iterable
    >>> from sympy import Tuple
    >>> things = [[1], (1,), set([1]), Tuple(1), (j for j in [1, 2]), {1:2}, '1', 1]
    >>> for i in things:
    ...     print('%s %s' % (iterable(i), type(i)))
    True <... 'list'>
    True <... 'tuple'>
    True <... 'set'>
    True <class 'sympy.core.containers.Tuple'>
    True <... 'generator'>
    False <... 'dict'>
    False <... 'str'>
    False <... 'int'>

    >>> iterable({}, exclude=None)
    True
    >>> iterable({}, exclude=str)
    True
    >>> iterable("no", exclude=str)
    False

    """
    if hasattr(i, '_iterable'):
        return i._iterable
    try:
        iter(i)
    except TypeError:
        return False
    if exclude:
        return not isinstance(i, exclude)
    return True


def is_sequence(i, include=None):
    """
    Return a boolean indicating whether ``i`` is a sequence in the SymPy
    sense. If anything that fails the test below should be included as
    being a sequence for your application, set 'include' to that object's
    type; multiple types should be passed as a tuple of types.

    Note: although generators can generate a sequence, they often need special
    handling to make sure their elements are captured before the generator is
    exhausted, so these are not included by default in the definition of a
    sequence.

    See also: iterable

    Examples
    ========

    >>> from sympy.utilities.iterables import is_sequence
    >>> from types import GeneratorType
    >>> is_sequence([])
    True
    >>> is_sequence(set())
    False
    >>> is_sequence('abc')
    False
    >>> is_sequence('abc', include=str)
    True
    >>> generator = (c for c in 'abc')
    >>> is_sequence(generator)
    False
    >>> is_sequence(generator, include=(str, GeneratorType))
    True

    """
    return (hasattr(i, '__getitem__') and
            iterable(i) or
            bool(include) and
            isinstance(i, include))


@deprecated(
    """
    Using postorder_traversal from the sympy.utilities.iterables submodule is
    deprecated.

    Instead, use postorder_traversal from the top-level sympy namespace, like

        sympy.postorder_traversal
    """,
    deprecated_since_version="1.10",
    active_deprecations_target="deprecated-traversal-functions-moved")
def postorder_traversal(node, keys=None):
    from sympy.core.traversal import postorder_traversal as _postorder_traversal
    return _postorder_traversal(node, keys=keys)


@deprecated(
    """
    Using interactive_traversal from the sympy.utilities.iterables submodule
    is deprecated.

    Instead, use interactive_traversal from the top-level sympy namespace,
    like

        sympy.interactive_traversal
    """,
    deprecated_since_version="1.10",
    active_deprecations_target="deprecated-traversal-functions-moved")
def interactive_traversal(expr):
    from sympy.interactive.traversal import interactive_traversal as _interactive_traversal
    return _interactive_traversal(expr)


@deprecated(
    """
    Importing default_sort_key from sympy.utilities.iterables is deprecated.
    Use from sympy import default_sort_key instead.
    """,
    deprecated_since_version="1.10",
active_deprecations_target="deprecated-sympy-core-compatibility",
)
def default_sort_key(*args, **kwargs):
    from sympy import default_sort_key as _default_sort_key
    return _default_sort_key(*args, **kwargs)


@deprecated(
    """
    Importing default_sort_key from sympy.utilities.iterables is deprecated.
    Use from sympy import default_sort_key instead.
    """,
    deprecated_since_version="1.10",
active_deprecations_target="deprecated-sympy-core-compatibility",
)
def ordered(*args, **kwargs):
    from sympy import ordered as _ordered
    return _ordered(*args, **kwargs)