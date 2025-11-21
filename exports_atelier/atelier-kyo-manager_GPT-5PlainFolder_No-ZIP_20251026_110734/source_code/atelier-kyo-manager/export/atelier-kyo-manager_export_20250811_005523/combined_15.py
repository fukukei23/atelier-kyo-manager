
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\matrixbase.py ===
from __future__ import annotations
from collections import defaultdict
from collections.abc import Iterable
from inspect import isfunction
from functools import reduce

from sympy.assumptions.refine import refine
from sympy.core import SympifyError, Add
from sympy.core.basic import Atom, Basic
from sympy.core.kind import UndefinedKind
from sympy.core.numbers import Integer
from sympy.core.mod import Mod
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify, _sympify
from sympy.core.function import diff
from sympy.polys import cancel
from sympy.functions.elementary.complexes import Abs, re, im
from sympy.printing import sstr
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta, LeviCivita
from sympy.core.singleton import S
from sympy.printing.defaults import Printable
from sympy.printing.str import StrPrinter
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.combinatorial.factorials import binomial, factorial

import mpmath as mp
from collections.abc import Callable
from sympy.utilities.iterables import reshape
from sympy.core.expr import Expr
from sympy.core.power import Pow
from sympy.core.symbol import uniquely_named_symbol

from .utilities import _dotprodsimp, _simplify as _utilities_simplify
from sympy.polys.polytools import Poly
from sympy.utilities.iterables import flatten, is_sequence
from sympy.utilities.misc import as_int, filldedent
from sympy.core.decorators import call_highest_priority
from sympy.core.logic import fuzzy_and, FuzzyBool
from sympy.tensor.array import NDimArray
from sympy.utilities.iterables import NotIterable

from .utilities import _get_intermediate_simp_bool

from .kind import MatrixKind

from .exceptions import (
    MatrixError, ShapeError, NonSquareMatrixError, NonInvertibleMatrixError,
)

from .utilities import _iszero, _is_zero_after_expand_mul

from .determinant import (
    _find_reasonable_pivot, _find_reasonable_pivot_naive,
    _adjugate, _charpoly, _cofactor, _cofactor_matrix, _per,
    _det, _det_bareiss, _det_berkowitz, _det_bird, _det_laplace, _det_LU,
    _minor, _minor_submatrix)

from .reductions import _is_echelon, _echelon_form, _rank, _rref

from .solvers import (
    _diagonal_solve, _lower_triangular_solve, _upper_triangular_solve,
    _cholesky_solve, _LDLsolve, _LUsolve, _QRsolve, _gauss_jordan_solve,
    _pinv_solve, _cramer_solve, _solve, _solve_least_squares)

from .inverse import (
    _pinv, _inv_ADJ, _inv_GE, _inv_LU, _inv_CH, _inv_LDL, _inv_QR,
    _inv, _inv_block)

from .subspaces import _columnspace, _nullspace, _rowspace, _orthogonalize

from .eigen import (
    _eigenvals, _eigenvects,
    _bidiagonalize, _bidiagonal_decomposition,
    _is_diagonalizable, _diagonalize,
    _is_positive_definite, _is_positive_semidefinite,
    _is_negative_definite, _is_negative_semidefinite, _is_indefinite,
    _jordan_form, _left_eigenvects, _singular_values)

from .decompositions import (
    _rank_decomposition, _cholesky, _LDLdecomposition,
    _LUdecomposition, _LUdecomposition_Simple, _LUdecompositionFF,
    _singular_value_decomposition, _QRdecomposition, _upper_hessenberg_decomposition)

from .graph import (
    _connected_components, _connected_components_decomposition,
    _strongly_connected_components, _strongly_connected_components_decomposition)


__doctest_requires__ = {
    ('MatrixBase.is_indefinite',
     'MatrixBase.is_positive_definite',
     'MatrixBase.is_positive_semidefinite',
     'MatrixBase.is_negative_definite',
     'MatrixBase.is_negative_semidefinite'): ['matplotlib'],
}


class MatrixBase(Printable):
    """All common matrix operations including basic arithmetic, shaping,
    and special matrices like `zeros`, and `eye`."""

    _op_priority = 10.01

    # Added just for numpy compatibility
    __array_priority__ = 11

    is_Matrix = True
    _class_priority = 3
    _sympify = staticmethod(sympify)
    zero = S.Zero
    one = S.One

    _diff_wrt: bool = True
    rows: int
    cols: int
    _simplify = None

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

    @classmethod
    def _eval_from_dok(cls, rows, cols, dok):
        out_flat = [cls.zero] * (rows * cols)
        for (i, j), val in dok.items():
            out_flat[i * cols + j] = val
        return cls._new(rows, cols, out_flat)

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
        dok = {divmod(i*self.cols + j, cols):
            v for (i, j), v in self.todok().items()}
        return self._eval_from_dok(rows, cols, dok)

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

    @classmethod
    def from_dok(cls, rows, cols, dok):
        """Create a matrix from a dictionary of keys.

        Examples
        ========

        >>> from sympy import Matrix
        >>> d = {(0, 0): 1, (1, 2): 3, (2, 1): 4}
        >>> Matrix.from_dok(3, 3, d)
        Matrix([
        [1, 0, 0],
        [0, 0, 3],
        [0, 4, 0]])
        """
        dok = {ij: cls._sympify(val) for ij, val in dok.items()}
        return cls._eval_from_dok(rows, cols, dok)

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

    # The RepMatrix subclass uses more efficient sparse implementations of
    # _eval_iter_values and other things.

    def _eval_iter_values(self):
        return (i for i in self if i is not S.Zero)

    def _eval_values(self):
        return list(self.iter_values())

    def _eval_iter_items(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if self[i, j]:
                    yield (i, j), self[i, j]

    def _eval_atoms(self, *types):
        values = self.values()
        if len(values) < self.rows * self.cols and isinstance(S.Zero, types):
            s = {S.Zero}
        else:
            s = set()
        return s.union(*[v.atoms(*types) for v in values])

    def _eval_free_symbols(self):
        return set().union(*(i.free_symbols for i in set(self.values())))

    def _eval_has(self, *patterns):
        return any(a.has(*patterns) for a in self.iter_values())

    def _eval_is_symbolic(self):
        return self.has(Symbol)

    # _eval_is_hermitian is called by some general SymPy
    # routines and has a different *args signature.  Make
    # sure the names don't clash by adding `_matrix_` in name.
    def _eval_is_matrix_hermitian(self, simpfunc):
        herm = lambda i, j: simpfunc(self[i, j] - self[j, i].adjoint()).is_zero
        return fuzzy_and(herm(i, j) for (i, j), v in self.iter_items())

    def _eval_is_zero_matrix(self):
        return fuzzy_and(v.is_zero for v in self.iter_values())

    def _eval_is_Identity(self) -> FuzzyBool:
        one = self.one
        zero = self.zero
        ident = lambda i, j, v: v is one if i == j else v is zero
        return all(ident(i, j, v) for (i, j), v in self.iter_items())

    def _eval_is_diagonal(self):
        return fuzzy_and(v.is_zero for (i, j), v in self.iter_items() if i != j)

    def _eval_is_lower(self):
        return all(v.is_zero for (i, j), v in self.iter_items() if i < j)

    def _eval_is_upper(self):
        return all(v.is_zero for (i, j), v in self.iter_items() if i > j)

    def _eval_is_lower_hessenberg(self):
        return all(v.is_zero for (i, j), v in self.iter_items() if i + 1 < j)

    def _eval_is_upper_hessenberg(self):
        return all(v.is_zero for (i, j), v in self.iter_items() if i > j + 1)

    def _eval_is_symmetric(self, simpfunc):
        sym = lambda i, j: simpfunc(self[i, j] - self[j, i]).is_zero
        return fuzzy_and(sym(i, j) for (i, j), v in self.iter_items())

    def _eval_is_anti_symmetric(self, simpfunc):
        anti = lambda i, j: simpfunc(self[i, j] + self[j, i]).is_zero
        return fuzzy_and(anti(i, j) for (i, j), v in self.iter_items())

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
            simpfunc = _utilities_simplify if simplify else lambda x: x

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
        sympy.matrices.matrixbase.MatrixBase.is_diagonalizable
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

        return self._eval_is_matrix_hermitian(_utilities_simplify)

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
            simpfunc = _utilities_simplify if simplify else lambda x: x

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
        return self._eval_is_upper()

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
        """Return non-zero values of self.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix([[0, 1], [2, 3]])
        >>> m.values()
        [1, 2, 3]

        See Also
        ========

        iter_values
        tolist
        flat
        """
        return self._eval_values()

    def iter_values(self):
        """
        Iterate over non-zero values of self.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix([[0, 1], [2, 3]])
        >>> list(m.iter_values())
        [1, 2, 3]

        See Also
        ========

        values
        """
        return self._eval_iter_values()

    def iter_items(self):
        """Iterate over indices and values of nonzero items.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix([[0, 1], [2, 3]])
        >>> list(m.iter_items())
        [((0, 1), 1), ((1, 0), 2), ((1, 1), 3)]

        See Also
        ========

        iter_values
        todok
        """
        return self._eval_iter_items()

    def _eval_adjoint(self):
        return self.transpose().applyfunc(lambda x: x.adjoint())

    def _eval_applyfunc(self, f):
        cols = self.cols
        size = self.rows*self.cols

        dok = self.todok()
        valmap = {v: f(v) for v in dok.values()}

        if len(dok) < size and ((fzero := f(S.Zero)) is not S.Zero):
            out_flat = [fzero]*size
            for (i, j), v in dok.items():
                out_flat[i*cols + j] = valmap[v]
            out = self._new(self.rows, self.cols, out_flat)
        else:
            fdok = {ij: valmap[v] for ij, v in dok.items()}
            out = self.from_dok(self.rows, self.cols, fdok)

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
        return self.adjoint()

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
        kwargs = {'map': map, 'simultaneous': simultaneous, 'exact': exact}

        if map:

            d = {}
            def func(eij):
                eij, dij = eij.replace(F, G, **kwargs)
                d.update(dij)
                return eij

            M = self.applyfunc(func)
            return M, d

        else:
            return self.applyfunc(lambda i: i.replace(F, G, **kwargs))

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
        return self.applyfunc(lambda x: x.simplify(**kwargs))

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

        other, T = _coerce_operand(self, other)

        if T != "is_matrix":
            return NotImplemented

        if self.shape != other.shape:
            raise ShapeError(f"Matrix size mismatch: {self.shape} + {other.shape}.")

        # Unify matrix types
        a, b = self, other
        if a.__class__ != classof(a, b):
            b, a = a, b

        return a._eval_add(b)

    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        return self * (self.one / other)

    @call_highest_priority('__rmatmul__')
    def __matmul__(self, other):
        self, other, T = _unify_with_other(self, other)

        if T != "is_matrix":
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

        self, other, T = _unify_with_other(self, other)

        if T == "possible_scalar":
            try:
                return self._eval_scalar_mul(other)
            except TypeError:
                return NotImplemented

        elif T == "is_matrix":

            if self.shape[1] != other.shape[0]:
                raise ShapeError(f"Matrix size mismatch: {self.shape} * {other.shape}.")

            m = self._eval_matrix_mul(other)

            if isimpbool:
                m = m._new(m.rows, m.cols, [_dotprodsimp(e) for e in m])

            return m

        else:
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
        return self.__add__(other)

    @call_highest_priority('__matmul__')
    def __rmatmul__(self, other):
        self, other, T = _unify_with_other(self, other)

        if T != "is_matrix":
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
        self, other, T = _unify_with_other(self, other)

        if T == "possible_scalar":
            try:
                return self._eval_scalar_rmul(other)
            except TypeError:
                return NotImplemented

        elif T == "is_matrix":
            if self.shape[0] != other.shape[1]:
                raise ShapeError("Matrix size mismatch.")

            m = self._eval_matrix_rmul(other)

            if isimpbool:
                return m._new(m.rows, m.cols, [_dotprodsimp(e) for e in m])

            return m

        else:
            return NotImplemented

    @call_highest_priority('__sub__')
    def __rsub__(self, a):
        return (-self) + a

    @call_highest_priority('__rsub__')
    def __sub__(self, a):
        return self + (-a)

    def _eval_det_bareiss(self, iszerofunc=_is_zero_after_expand_mul):
        return _det_bareiss(self, iszerofunc=iszerofunc)

    def _eval_det_berkowitz(self):
        return _det_berkowitz(self)

    def _eval_det_lu(self, iszerofunc=_iszero, simpfunc=None):
        return _det_LU(self, iszerofunc=iszerofunc, simpfunc=simpfunc)

    def _eval_det_bird(self):
        return _det_bird(self)

    def _eval_det_laplace(self):
        return _det_laplace(self)

    def _eval_determinant(self): # for expressions.determinant.Determinant
        return _det(self)

    def adjugate(self, method="berkowitz"):
        return _adjugate(self, method=method)

    def charpoly(self, x='lambda', simplify=_utilities_simplify):
        return _charpoly(self, x=x, simplify=simplify)

    def cofactor(self, i, j, method="berkowitz"):
        return _cofactor(self, i, j, method=method)

    def cofactor_matrix(self, method="berkowitz"):
        return _cofactor_matrix(self, method=method)

    def det(self, method="bareiss", iszerofunc=None):
        return _det(self, method=method, iszerofunc=iszerofunc)

    def per(self):
        return _per(self)

    def minor(self, i, j, method="berkowitz"):
        return _minor(self, i, j, method=method)

    def minor_submatrix(self, i, j):
        return _minor_submatrix(self, i, j)

    _find_reasonable_pivot.__doc__       = _find_reasonable_pivot.__doc__
    _find_reasonable_pivot_naive.__doc__ = _find_reasonable_pivot_naive.__doc__
    _eval_det_bareiss.__doc__            = _det_bareiss.__doc__
    _eval_det_berkowitz.__doc__          = _det_berkowitz.__doc__
    _eval_det_bird.__doc__            = _det_bird.__doc__
    _eval_det_laplace.__doc__            = _det_laplace.__doc__
    _eval_det_lu.__doc__                 = _det_LU.__doc__
    _eval_determinant.__doc__            = _det.__doc__
    adjugate.__doc__                     = _adjugate.__doc__
    charpoly.__doc__                     = _charpoly.__doc__
    cofactor.__doc__                     = _cofactor.__doc__
    cofactor_matrix.__doc__              = _cofactor_matrix.__doc__
    det.__doc__                          = _det.__doc__
    per.__doc__                          = _per.__doc__
    minor.__doc__                        = _minor.__doc__
    minor_submatrix.__doc__              = _minor_submatrix.__doc__

    def echelon_form(self, iszerofunc=_iszero, simplify=False, with_pivots=False):
        return _echelon_form(self, iszerofunc=iszerofunc, simplify=simplify,
                with_pivots=with_pivots)

    @property
    def is_echelon(self):
        return _is_echelon(self)

    def rank(self, iszerofunc=_iszero, simplify=False):
        return _rank(self, iszerofunc=iszerofunc, simplify=simplify)

    def rref_rhs(self, rhs):
        """Return reduced row-echelon form of matrix, matrix showing
        rhs after reduction steps. ``rhs`` must have the same number
        of rows as ``self``.

        Examples
        ========

        >>> from sympy import Matrix, symbols
        >>> r1, r2 = symbols('r1 r2')
        >>> Matrix([[1, 1], [2, 1]]).rref_rhs(Matrix([r1, r2]))
        (Matrix([
        [1, 0],
        [0, 1]]), Matrix([
        [ -r1 + r2],
        [2*r1 - r2]]))
        """
        r, _ = _rref(self.hstack(self, self.eye(self.rows), rhs))
        return r[:, :self.cols], r[:, -rhs.cols:]

    def rref(self, iszerofunc=_iszero, simplify=False, pivots=True,
            normalize_last=True):
        return _rref(self, iszerofunc=iszerofunc, simplify=simplify,
            pivots=pivots, normalize_last=normalize_last)

    echelon_form.__doc__ = _echelon_form.__doc__
    is_echelon.__doc__   = _is_echelon.__doc__
    rank.__doc__         = _rank.__doc__
    rref.__doc__         = _rref.__doc__

    def _normalize_op_args(self, op, col, k, col1, col2, error_str="col"):
        """Validate the arguments for a row/column operation.  ``error_str``
        can be one of "row" or "col" depending on the arguments being parsed."""
        if op not in ["n->kn", "n<->m", "n->n+km"]:
            raise ValueError("Unknown {} operation '{}'. Valid col operations "
                             "are 'n->kn', 'n<->m', 'n->n+km'".format(error_str, op))

        # define self_col according to error_str
        self_cols = self.cols if error_str == 'col' else self.rows

        # normalize and validate the arguments
        if op == "n->kn":
            col = col if col is not None else col1
            if col is None or k is None:
                raise ValueError("For a {0} operation 'n->kn' you must provide the "
                                 "kwargs `{0}` and `k`".format(error_str))
            if not 0 <= col < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col))

        elif op == "n<->m":
            # we need two cols to swap. It does not matter
            # how they were specified, so gather them together and
            # remove `None`
            cols = {col, k, col1, col2}.difference([None])
            if len(cols) > 2:
                # maybe the user left `k` by mistake?
                cols = {col, col1, col2}.difference([None])
            if len(cols) != 2:
                raise ValueError("For a {0} operation 'n<->m' you must provide the "
                                 "kwargs `{0}1` and `{0}2`".format(error_str))
            col1, col2 = cols
            if not 0 <= col1 < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col1))
            if not 0 <= col2 < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col2))

        elif op == "n->n+km":
            col = col1 if col is None else col
            col2 = col1 if col2 is None else col2
            if col is None or col2 is None or k is None:
                raise ValueError("For a {0} operation 'n->n+km' you must provide the "
                                 "kwargs `{0}`, `k`, and `{0}2`".format(error_str))
            if col == col2:
                raise ValueError("For a {0} operation 'n->n+km' `{0}` and `{0}2` must "
                                 "be different.".format(error_str))
            if not 0 <= col < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col))
            if not 0 <= col2 < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col2))

        else:
            raise ValueError('invalid operation %s' % repr(op))

        return op, col, k, col1, col2

    def _eval_col_op_multiply_col_by_const(self, col, k):
        def entry(i, j):
            if j == col:
                return k * self[i, j]
            return self[i, j]
        return self._new(self.rows, self.cols, entry)

    def _eval_col_op_swap(self, col1, col2):
        def entry(i, j):
            if j == col1:
                return self[i, col2]
            elif j == col2:
                return self[i, col1]
            return self[i, j]
        return self._new(self.rows, self.cols, entry)

    def _eval_col_op_add_multiple_to_other_col(self, col, k, col2):
        def entry(i, j):
            if j == col:
                return self[i, j] + k * self[i, col2]
            return self[i, j]
        return self._new(self.rows, self.cols, entry)

    def _eval_row_op_swap(self, row1, row2):
        def entry(i, j):
            if i == row1:
                return self[row2, j]
            elif i == row2:
                return self[row1, j]
            return self[i, j]
        return self._new(self.rows, self.cols, entry)

    def _eval_row_op_multiply_row_by_const(self, row, k):
        def entry(i, j):
            if i == row:
                return k * self[i, j]
            return self[i, j]
        return self._new(self.rows, self.cols, entry)

    def _eval_row_op_add_multiple_to_other_row(self, row, k, row2):
        def entry(i, j):
            if i == row:
                return self[i, j] + k * self[row2, j]
            return self[i, j]
        return self._new(self.rows, self.cols, entry)

    def elementary_col_op(self, op="n->kn", col=None, k=None, col1=None, col2=None):
        """Performs the elementary column operation `op`.

        `op` may be one of

            * ``"n->kn"`` (column n goes to k*n)
            * ``"n<->m"`` (swap column n and column m)
            * ``"n->n+km"`` (column n goes to column n + k*column m)

        Parameters
        ==========

        op : string; the elementary row operation
        col : the column to apply the column operation
        k : the multiple to apply in the column operation
        col1 : one column of a column swap
        col2 : second column of a column swap or column "m" in the column operation
               "n->n+km"
        """

        op, col, k, col1, col2 = self._normalize_op_args(op, col, k, col1, col2, "col")

        # now that we've validated, we're all good to dispatch
        if op == "n->kn":
            return self._eval_col_op_multiply_col_by_const(col, k)
        if op == "n<->m":
            return self._eval_col_op_swap(col1, col2)
        if op == "n->n+km":
            return self._eval_col_op_add_multiple_to_other_col(col, k, col2)

    def elementary_row_op(self, op="n->kn", row=None, k=None, row1=None, row2=None):
        """Performs the elementary row operation `op`.

        `op` may be one of

            * ``"n->kn"`` (row n goes to k*n)
            * ``"n<->m"`` (swap row n and row m)
            * ``"n->n+km"`` (row n goes to row n + k*row m)

        Parameters
        ==========

        op : string; the elementary row operation
        row : the row to apply the row operation
        k : the multiple to apply in the row operation
        row1 : one row of a row swap
        row2 : second row of a row swap or row "m" in the row operation
               "n->n+km"
        """

        op, row, k, row1, row2 = self._normalize_op_args(op, row, k, row1, row2, "row")

        # now that we've validated, we're all good to dispatch
        if op == "n->kn":
            return self._eval_row_op_multiply_row_by_const(row, k)
        if op == "n<->m":
            return self._eval_row_op_swap(row1, row2)
        if op == "n->n+km":
            return self._eval_row_op_add_multiple_to_other_row(row, k, row2)

    def columnspace(self, simplify=False):
        return _columnspace(self, simplify=simplify)

    def nullspace(self, simplify=False, iszerofunc=_iszero):
        return _nullspace(self, simplify=simplify, iszerofunc=iszerofunc)

    def rowspace(self, simplify=False):
        return _rowspace(self, simplify=simplify)

    # This is a classmethod but is converted to such later in order to allow
    # assignment of __doc__ since that does not work for already wrapped
    # classmethods in Python 3.6.
    def orthogonalize(cls, *vecs, **kwargs):
        return _orthogonalize(cls, *vecs, **kwargs)

    columnspace.__doc__   = _columnspace.__doc__
    nullspace.__doc__     = _nullspace.__doc__
    rowspace.__doc__      = _rowspace.__doc__
    orthogonalize.__doc__ = _orthogonalize.__doc__

    orthogonalize         = classmethod(orthogonalize)  # type:ignore

    def eigenvals(self, error_when_incomplete=True, **flags):
        return _eigenvals(self, error_when_incomplete=error_when_incomplete, **flags)

    def eigenvects(self, error_when_incomplete=True, iszerofunc=_iszero, **flags):
        return _eigenvects(self, error_when_incomplete=error_when_incomplete,
                iszerofunc=iszerofunc, **flags)

    def is_diagonalizable(self, reals_only=False, **kwargs):
        return _is_diagonalizable(self, reals_only=reals_only, **kwargs)

    def diagonalize(self, reals_only=False, sort=False, normalize=False):
        return _diagonalize(self, reals_only=reals_only, sort=sort,
                normalize=normalize)

    def bidiagonalize(self, upper=True):
        return _bidiagonalize(self, upper=upper)

    def bidiagonal_decomposition(self, upper=True):
        return _bidiagonal_decomposition(self, upper=upper)

    @property
    def is_positive_definite(self):
        return _is_positive_definite(self)

    @property
    def is_positive_semidefinite(self):
        return _is_positive_semidefinite(self)

    @property
    def is_negative_definite(self):
        return _is_negative_definite(self)

    @property
    def is_negative_semidefinite(self):
        return _is_negative_semidefinite(self)

    @property
    def is_indefinite(self):
        return _is_indefinite(self)

    def jordan_form(self, calc_transform=True, **kwargs):
        return _jordan_form(self, calc_transform=calc_transform, **kwargs)

    def left_eigenvects(self, **flags):
        return _left_eigenvects(self, **flags)

    def singular_values(self):
        return _singular_values(self)

    eigenvals.__doc__                  = _eigenvals.__doc__
    eigenvects.__doc__                 = _eigenvects.__doc__
    is_diagonalizable.__doc__          = _is_diagonalizable.__doc__
    diagonalize.__doc__                = _diagonalize.__doc__
    is_positive_definite.__doc__       = _is_positive_definite.__doc__
    is_positive_semidefinite.__doc__   = _is_positive_semidefinite.__doc__
    is_negative_definite.__doc__       = _is_negative_definite.__doc__
    is_negative_semidefinite.__doc__   = _is_negative_semidefinite.__doc__
    is_indefinite.__doc__              = _is_indefinite.__doc__
    jordan_form.__doc__                = _jordan_form.__doc__
    left_eigenvects.__doc__            = _left_eigenvects.__doc__
    singular_values.__doc__            = _singular_values.__doc__
    bidiagonalize.__doc__              = _bidiagonalize.__doc__
    bidiagonal_decomposition.__doc__   = _bidiagonal_decomposition.__doc__

    def diff(self, *args, evaluate=True, **kwargs):
        """Calculate the derivative of each element in the matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x, y
        >>> M = Matrix([[x, y], [1, 0]])
        >>> M.diff(x)
        Matrix([
        [1, 0],
        [0, 0]])

        See Also
        ========

        integrate
        limit
        """
        # XXX this should be handled here rather than in Derivative
        from sympy.tensor.array.array_derivatives import ArrayDerivative
        deriv = ArrayDerivative(self, *args, evaluate=evaluate)
        # XXX This can rather changed to always return immutable matrix
        if not isinstance(self, Basic) and evaluate:
            return deriv.as_mutable()
        return deriv

    def _eval_derivative(self, arg):
        return self.applyfunc(lambda x: x.diff(arg))

    def integrate(self, *args, **kwargs):
        """Integrate each element of the matrix.  ``args`` will
        be passed to the ``integrate`` function.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x, y
        >>> M = Matrix([[x, y], [1, 0]])
        >>> M.integrate((x, ))
        Matrix([
        [x**2/2, x*y],
        [     x,   0]])
        >>> M.integrate((x, 0, 2))
        Matrix([
        [2, 2*y],
        [2,   0]])

        See Also
        ========

        limit
        diff
        """
        return self.applyfunc(lambda x: x.integrate(*args, **kwargs))

    def jacobian(self, X):
        """Calculates the Jacobian matrix (derivative of a vector-valued function).

        Parameters
        ==========

        ``self`` : vector of expressions representing functions f_i(x_1, ..., x_n).
        X : set of x_i's in order, it can be a list or a Matrix

        Both ``self`` and X can be a row or a column matrix in any order
        (i.e., jacobian() should always work).

        Examples
        ========

        >>> from sympy import sin, cos, Matrix
        >>> from sympy.abc import rho, phi
        >>> X = Matrix([rho*cos(phi), rho*sin(phi), rho**2])
        >>> Y = Matrix([rho, phi])
        >>> X.jacobian(Y)
        Matrix([
        [cos(phi), -rho*sin(phi)],
        [sin(phi),  rho*cos(phi)],
        [   2*rho,             0]])
        >>> X = Matrix([rho*cos(phi), rho*sin(phi)])
        >>> X.jacobian(Y)
        Matrix([
        [cos(phi), -rho*sin(phi)],
        [sin(phi),  rho*cos(phi)]])

        See Also
        ========

        hessian
        wronskian
        """
        from sympy.matrices.matrixbase import MatrixBase
        if not isinstance(X, MatrixBase):
            X = self._new(X)
        # Both X and ``self`` can be a row or a column matrix, so we need to make
        # sure all valid combinations work, but everything else fails:
        if self.shape[0] == 1:
            m = self.shape[1]
        elif self.shape[1] == 1:
            m = self.shape[0]
        else:
            raise TypeError("``self`` must be a row or a column matrix")
        if X.shape[0] == 1:
            n = X.shape[1]
        elif X.shape[1] == 1:
            n = X.shape[0]
        else:
            raise TypeError("X must be a row or a column matrix")

        # m is the number of functions and n is the number of variables
        # computing the Jacobian is now easy:
        return self._new(m, n, lambda j, i: self[j].diff(X[i]))

    def limit(self, *args):
        """Calculate the limit of each element in the matrix.
        ``args`` will be passed to the ``limit`` function.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x, y
        >>> M = Matrix([[x, y], [1, 0]])
        >>> M.limit(x, 2)
        Matrix([
        [2, y],
        [1, 0]])

        See Also
        ========

        integrate
        diff
        """
        return self.applyfunc(lambda x: x.limit(*args))

    def berkowitz_charpoly(self, x=Dummy('lambda'), simplify=_utilities_simplify):
        return self.charpoly(x=x)

    def berkowitz_det(self):
        """Computes determinant using Berkowitz method.

        See Also
        ========

        det
        """
        return self.det(method='berkowitz')

    def berkowitz_eigenvals(self, **flags):
        """Computes eigenvalues of a Matrix using Berkowitz method."""
        return self.eigenvals(**flags)

    def berkowitz_minors(self):
        """Computes principal minors using Berkowitz method."""
        sign, minors = self.one, []

        for poly in self.berkowitz():
            minors.append(sign * poly[-1])
            sign = -sign

        return tuple(minors)

    def berkowitz(self):
        from sympy.matrices import zeros
        berk = ((1,),)
        if not self:
            return berk

        if not self.is_square:
            raise NonSquareMatrixError()

        A, N = self, self.rows
        transforms = [0] * (N - 1)

        for n in range(N, 1, -1):
            T, k = zeros(n + 1, n), n - 1

            R, C = -A[k, :k], A[:k, k]
            A, a = A[:k, :k], -A[k, k]

            items = [C]

            for i in range(0, n - 2):
                items.append(A * items[i])

            for i, B in enumerate(items):
                items[i] = (R * B)[0, 0]

            items = [self.one, a] + items

            for i in range(n):
                T[i:, i] = items[:n - i + 1]

            transforms[k - 1] = T

        polys = [self._new([self.one, -A[0, 0]])]

        for i, T in enumerate(transforms):
            polys.append(T * polys[i])

        return berk + tuple(map(tuple, polys))

    def cofactorMatrix(self, method="berkowitz"):
        return self.cofactor_matrix(method=method)

    def det_bareis(self):
        return _det_bareiss(self)

    def det_LU_decomposition(self):
        """Compute matrix determinant using LU decomposition.


        Note that this method fails if the LU decomposition itself
        fails. In particular, if the matrix has no inverse this method
        will fail.

        TODO: Implement algorithm for sparse matrices (SFF),
        http://www.eecis.udel.edu/~saunders/papers/sffge/it5.ps.

        See Also
        ========


        det
        berkowitz_det
        """
        return self.det(method='lu')

    def jordan_cell(self, eigenval, n):
        return self.jordan_block(size=n, eigenvalue=eigenval)

    def jordan_cells(self, calc_transformation=True):
        P, J = self.jordan_form()
        return P, J.get_diag_blocks()

    def minorEntry(self, i, j, method="berkowitz"):
        return self.minor(i, j, method=method)

    def minorMatrix(self, i, j):
        return self.minor_submatrix(i, j)

    def permuteBkwd(self, perm):
        """Permute the rows of the matrix with the given permutation in reverse."""
        return self.permute_rows(perm, direction='backward')

    def permuteFwd(self, perm):
        """Permute the rows of the matrix with the given permutation."""
        return self.permute_rows(perm, direction='forward')

    @property
    def kind(self) -> MatrixKind:
        elem_kinds = {e.kind for e in self.flat()}
        if len(elem_kinds) == 1:
            elemkind, = elem_kinds
        else:
            elemkind = UndefinedKind
        return MatrixKind(elemkind)

    def flat(self):
        """
        Returns a flat list of all elements in the matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix([[0, 2], [3, 4]])
        >>> m.flat()
        [0, 2, 3, 4]

        See Also
        ========

        tolist
        values
        """
        return [self[i, j] for i in range(self.rows) for j in range(self.cols)]

    def __array__(self, dtype=object, copy=None):
        if copy is not None and not copy:
            raise TypeError("Cannot implement copy=False when converting Matrix to ndarray")
        from .dense import matrix2numpy
        return matrix2numpy(self, dtype=dtype)

    def __len__(self):
        """Return the number of elements of ``self``.

        Implemented mainly so bool(Matrix()) == False.
        """
        return self.rows * self.cols

    def _matrix_pow_by_jordan_blocks(self, num):
        from sympy.matrices import diag, MutableMatrix

        def jordan_cell_power(jc, n):
            N = jc.shape[0]
            l = jc[0,0]
            if l.is_zero:
                if N == 1 and n.is_nonnegative:
                    jc[0,0] = l**n
                elif not (n.is_integer and n.is_nonnegative):
                    raise NonInvertibleMatrixError("Non-invertible matrix can only be raised to a nonnegative integer")
                else:
                    for i in range(N):
                        jc[0,i] = KroneckerDelta(i, n)
            else:
                for i in range(N):
                    bn = binomial(n, i)
                    if isinstance(bn, binomial):
                        bn = bn._eval_expand_func()
                    jc[0,i] = l**(n-i)*bn
            for i in range(N):
                for j in range(1, N-i):
                    jc[j,i+j] = jc [j-1,i+j-1]

        P, J = self.jordan_form()
        jordan_cells = J.get_diag_blocks()
        # Make sure jordan_cells matrices are mutable:
        jordan_cells = [MutableMatrix(j) for j in jordan_cells]
        for j in jordan_cells:
            jordan_cell_power(j, num)
        return self._new(P.multiply(diag(*jordan_cells))
                .multiply(P.inv()))

    def __str__(self):
        if S.Zero in self.shape:
            return 'Matrix(%s, %s, [])' % (self.rows, self.cols)
        return "Matrix(%s)" % str(self.tolist())

    def _format_str(self, printer=None):
        if not printer:
            printer = StrPrinter()
        # Handle zero dimensions:
        if S.Zero in self.shape:
            return 'Matrix(%s, %s, [])' % (self.rows, self.cols)
        if self.rows == 1:
            return "Matrix([%s])" % self.table(printer, rowsep=',\n')
        return "Matrix([\n%s])" % self.table(printer, rowsep=',\n')

    @classmethod
    def irregular(cls, ntop, *matrices, **kwargs):
        """Return a matrix filled by the given matrices which
        are listed in order of appearance from left to right, top to
        bottom as they first appear in the matrix. They must fill the
        matrix completely.

        Examples
        ========

        >>> from sympy import ones, Matrix
        >>> Matrix.irregular(3, ones(2,1), ones(3,3)*2, ones(2,2)*3,
        ...   ones(1,1)*4, ones(2,2)*5, ones(1,2)*6, ones(1,2)*7)
        Matrix([
            [1, 2, 2, 2, 3, 3],
            [1, 2, 2, 2, 3, 3],
            [4, 2, 2, 2, 5, 5],
            [6, 6, 7, 7, 5, 5]])
        """
        ntop = as_int(ntop)
        # make sure we are working with explicit matrices
        b = [i.as_explicit() if hasattr(i, 'as_explicit') else i
            for i in matrices]
        q = list(range(len(b)))
        dat = [i.rows for i in b]
        active = [q.pop(0) for _ in range(ntop)]
        cols = sum(b[i].cols for i in active)
        rows = []
        while any(dat):
            r = []
            for a, j in enumerate(active):
                r.extend(b[j][-dat[j], :])
                dat[j] -= 1
                if dat[j] == 0 and q:
                    active[a] = q.pop(0)
            if len(r) != cols:
                raise ValueError(filldedent('''
                    Matrices provided do not appear to fill
                    the space completely.'''))
            rows.append(r)
        return cls._new(rows)

    @classmethod
    def _handle_ndarray(cls, arg):
        # NumPy array or matrix or some other object that implements
        # __array__. So let's first use this method to get a
        # numpy.array() and then make a Python list out of it.
        arr = arg.__array__()
        if len(arr.shape) == 2:
            rows, cols = arr.shape[0], arr.shape[1]
            flat_list = [cls._sympify(i) for i in arr.ravel()]
            return rows, cols, flat_list
        elif len(arr.shape) == 1:
            flat_list = [cls._sympify(i) for i in arr]
            return arr.shape[0], 1, flat_list
        else:
            raise NotImplementedError(
                "SymPy supports just 1D and 2D matrices")

    @classmethod
    def _handle_creation_inputs(cls, *args, **kwargs):
        """Return the number of rows, cols and flat matrix elements.

        Examples
        ========

        >>> from sympy import Matrix, I

        Matrix can be constructed as follows:

        * from a nested list of iterables

        >>> Matrix( ((1, 2+I), (3, 4)) )
        Matrix([
        [1, 2 + I],
        [3,     4]])

        * from un-nested iterable (interpreted as a column)

        >>> Matrix( [1, 2] )
        Matrix([
        [1],
        [2]])

        * from un-nested iterable with dimensions

        >>> Matrix(1, 2, [1, 2] )
        Matrix([[1, 2]])

        * from no arguments (a 0 x 0 matrix)

        >>> Matrix()
        Matrix(0, 0, [])

        * from a rule

        >>> Matrix(2, 2, lambda i, j: i/(j + 1) )
        Matrix([
        [0,   0],
        [1, 1/2]])

        See Also
        ========
        irregular - filling a matrix with irregular blocks
        """
        from sympy.matrices import SparseMatrix
        from sympy.matrices.expressions.matexpr import MatrixSymbol
        from sympy.matrices.expressions.blockmatrix import BlockMatrix

        flat_list = None

        if len(args) == 1:
            # Matrix(SparseMatrix(...))
            if isinstance(args[0], SparseMatrix):
                return args[0].rows, args[0].cols, flatten(args[0].tolist())

            # Matrix(Matrix(...))
            elif isinstance(args[0], MatrixBase):
                return args[0].rows, args[0].cols, args[0].flat()

            # Matrix(MatrixSymbol('X', 2, 2))
            elif isinstance(args[0], Basic) and args[0].is_Matrix:
                return args[0].rows, args[0].cols, args[0].as_explicit().flat()

            elif isinstance(args[0], mp.matrix):
                M = args[0]
                flat_list = [cls._sympify(x) for x in M]
                return M.rows, M.cols, flat_list

            # Matrix(numpy.ones((2, 2)))
            elif hasattr(args[0], "__array__"):
                return cls._handle_ndarray(args[0])

            # Matrix([1, 2, 3]) or Matrix([[1, 2], [3, 4]])
            elif is_sequence(args[0]) \
                    and not isinstance(args[0], DeferredVector):
                dat = list(args[0])
                ismat = lambda i: isinstance(i, MatrixBase) and (
                    evaluate or isinstance(i, (BlockMatrix, MatrixSymbol)))
                raw = lambda i: is_sequence(i) and not ismat(i)
                evaluate = kwargs.get('evaluate', True)


                if evaluate:

                    def make_explicit(x):
                        """make Block and Symbol explicit"""
                        if isinstance(x, BlockMatrix):
                            return x.as_explicit()
                        elif isinstance(x, MatrixSymbol) and all(_.is_Integer for _ in x.shape):
                            return x.as_explicit()
                        else:
                            return x

                    def make_explicit_row(row):
                        # Could be list or could be list of lists
                        if isinstance(row, (list, tuple)):
                            return [make_explicit(x) for x in row]
                        else:
                            return make_explicit(row)

                    if isinstance(dat, (list, tuple)):
                        dat = [make_explicit_row(row) for row in dat]

                if len(dat) == 0:
                    rows = cols = 0
                    flat_list = []
                elif all(raw(i) for i in dat) and len(dat[0]) == 0:
                    if not all(len(i) == 0 for i in dat):
                        raise ValueError('mismatched dimensions')
                    rows = len(dat)
                    cols = 0
                    flat_list = []
                elif not any(raw(i) or ismat(i) for i in dat):
                    # a column as a list of values
                    flat_list = [cls._sympify(i) for i in dat]
                    rows = len(flat_list)
                    cols = 1 if rows else 0
                elif evaluate and all(ismat(i) for i in dat):
                    # a column as a list of matrices
                    ncol = {i.cols for i in dat if any(i.shape)}
                    if ncol:
                        if len(ncol) != 1:
                            raise ValueError('mismatched dimensions')
                        flat_list = [_ for i in dat for r in i.tolist() for _ in r]
                        cols = ncol.pop()
                        rows = len(flat_list)//cols
                    else:
                        rows = cols = 0
                        flat_list = []
                elif evaluate and any(ismat(i) for i in dat):
                    ncol = set()
                    flat_list = []
                    for i in dat:
                        if ismat(i):
                            flat_list.extend(
                                [k for j in i.tolist() for k in j])
                            if any(i.shape):
                                ncol.add(i.cols)
                        elif raw(i):
                            if i:
                                ncol.add(len(i))
                                flat_list.extend([cls._sympify(ij) for ij in i])
                        else:
                            ncol.add(1)
                            flat_list.append(i)
                        if len(ncol) > 1:
                            raise ValueError('mismatched dimensions')
                    cols = ncol.pop()
                    rows = len(flat_list)//cols
                else:
                    # list of lists; each sublist is a logical row
                    # which might consist of many rows if the values in
                    # the row are matrices
                    flat_list = []
                    ncol = set()
                    rows = cols = 0
                    for row in dat:
                        if not is_sequence(row) and \
                                not getattr(row, 'is_Matrix', False):
                            raise ValueError('expecting list of lists')

                        if hasattr(row, '__array__'):
                            if 0 in row.shape:
                                continue

                        if evaluate and all(ismat(i) for i in row):
                            r, c, flatT = cls._handle_creation_inputs(
                                [i.T for i in row])
                            T = reshape(flatT, [c])
                            flat = \
                                [T[i][j] for j in range(c) for i in range(r)]
                            r, c = c, r
                        else:
                            r = 1
                            if getattr(row, 'is_Matrix', False):
                                c = 1
                                flat = [row]
                            else:
                                c = len(row)
                                flat = [cls._sympify(i) for i in row]
                        ncol.add(c)
                        if len(ncol) > 1:
                            raise ValueError('mismatched dimensions')
                        flat_list.extend(flat)
                        rows += r
                    cols = ncol.pop() if ncol else 0

        elif len(args) == 3:
            rows = as_int(args[0])
            cols = as_int(args[1])

            if rows < 0 or cols < 0:
                raise ValueError("Cannot create a {} x {} matrix. "
                                 "Both dimensions must be positive".format(rows, cols))

            # Matrix(2, 2, lambda i, j: i+j)
            if len(args) == 3 and isinstance(args[2], Callable):
                op = args[2]
                flat_list = []
                for i in range(rows):
                    flat_list.extend(
                        [cls._sympify(op(cls._sympify(i), cls._sympify(j)))
                         for j in range(cols)])

            # Matrix(2, 2, [1, 2, 3, 4])
            elif len(args) == 3 and is_sequence(args[2]):
                flat_list = args[2]
                if len(flat_list) != rows * cols:
                    raise ValueError(
                        'List length should be equal to rows*columns')
                flat_list = [cls._sympify(i) for i in flat_list]


        # Matrix()
        elif len(args) == 0:
            # Empty Matrix
            rows = cols = 0
            flat_list = []

        if flat_list is None:
            raise TypeError(filldedent('''
                Data type not understood; expecting list of lists
                or lists of values.'''))

        return rows, cols, flat_list

    def _setitem(self, key, value):
        """Helper to set value at location given by key.

        Examples
        ========

        >>> from sympy import Matrix, I, zeros, ones
        >>> m = Matrix(((1, 2+I), (3, 4)))
        >>> m
        Matrix([
        [1, 2 + I],
        [3,     4]])
        >>> m[1, 0] = 9
        >>> m
        Matrix([
        [1, 2 + I],
        [9,     4]])
        >>> m[1, 0] = [[0, 1]]

        To replace row r you assign to position r*m where m
        is the number of columns:

        >>> M = zeros(4)
        >>> m = M.cols
        >>> M[3*m] = ones(1, m)*2; M
        Matrix([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [2, 2, 2, 2]])

        And to replace column c you can assign to position c:

        >>> M[2] = ones(m, 1)*4; M
        Matrix([
        [0, 0, 4, 0],
        [0, 0, 4, 0],
        [0, 0, 4, 0],
        [2, 2, 4, 2]])
        """
        from .dense import Matrix

        is_slice = isinstance(key, slice)
        i, j = key = self.key2ij(key)
        is_mat = isinstance(value, MatrixBase)
        if isinstance(i, slice) or isinstance(j, slice):
            if is_mat:
                self.copyin_matrix(key, value)
                return
            if not isinstance(value, Expr) and is_sequence(value):
                self.copyin_list(key, value)
                return
            raise ValueError('unexpected value: %s' % value)
        else:
            if (not is_mat and
                    not isinstance(value, Basic) and is_sequence(value)):
                value = Matrix(value)
                is_mat = True
            if is_mat:
                if is_slice:
                    key = (slice(*divmod(i, self.cols)),
                           slice(*divmod(j, self.cols)))
                else:
                    key = (slice(i, i + value.rows),
                           slice(j, j + value.cols))
                self.copyin_matrix(key, value)
            else:
                return i, j, self._sympify(value)
            return

    def add(self, b):
        """Return self + b."""
        return self + b

    def condition_number(self):
        """Returns the condition number of a matrix.

        This is the maximum singular value divided by the minimum singular value

        Examples
        ========

        >>> from sympy import Matrix, S
        >>> A = Matrix([[1, 0, 0], [0, 10, 0], [0, 0, S.One/10]])
        >>> A.condition_number()
        100

        See Also
        ========

        singular_values
        """

        if not self:
            return self.zero
        singularvalues = self.singular_values()
        return Max(*singularvalues) / Min(*singularvalues)

    def copy(self):
        """
        Returns the copy of a matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix(2, 2, [1, 2, 3, 4])
        >>> A.copy()
        Matrix([
        [1, 2],
        [3, 4]])

        """
        return self._new(self.rows, self.cols, self.flat())

    def cross(self, b):
        r"""
        Return the cross product of ``self`` and ``b`` relaxing the condition
        of compatible dimensions: if each has 3 elements, a matrix of the
        same type and shape as ``self`` will be returned. If ``b`` has the same
        shape as ``self`` then common identities for the cross product (like
        `a \times b = - b \times a`) will hold.

        Parameters
        ==========
            b : 3x1 or 1x3 Matrix

        See Also
        ========

        dot
        hat
        vee
        multiply
        multiply_elementwise
        """
        from sympy.matrices.expressions.matexpr import MatrixExpr

        if not isinstance(b, (MatrixBase, MatrixExpr)):
            raise TypeError(
                "{} must be a Matrix, not {}.".format(b, type(b)))

        if not (self.rows * self.cols == b.rows * b.cols == 3):
            raise ShapeError("Dimensions incorrect for cross product: %s x %s" %
                             ((self.rows, self.cols), (b.rows, b.cols)))
        else:
            return self._new(self.rows, self.cols, (
                (self[1] * b[2] - self[2] * b[1]),
                (self[2] * b[0] - self[0] * b[2]),
                (self[0] * b[1] - self[1] * b[0])))

    def hat(self):
        r"""
        Return the skew-symmetric matrix representing the cross product,
        so that ``self.hat() * b`` is equivalent to  ``self.cross(b)``.

        Examples
        ========

        Calling ``hat`` creates a skew-symmetric 3x3 Matrix from a 3x1 Matrix:

        >>> from sympy import Matrix
        >>> a = Matrix([1, 2, 3])
        >>> a.hat()
        Matrix([
        [ 0, -3,  2],
        [ 3,  0, -1],
        [-2,  1,  0]])

        Multiplying it with another 3x1 Matrix calculates the cross product:

        >>> b = Matrix([3, 2, 1])
        >>> a.hat() * b
        Matrix([
        [-4],
        [ 8],
        [-4]])

        Which is equivalent to calling the ``cross`` method:

        >>> a.cross(b)
        Matrix([
        [-4],
        [ 8],
        [-4]])

        See Also
        ========

        dot
        cross
        vee
        multiply
        multiply_elementwise
        """

        if self.shape != (3, 1):
            raise ShapeError("Dimensions incorrect, expected (3, 1), got " +
                             str(self.shape))
        else:
            x, y, z = self
            return self._new(3, 3, (
                 0, -z,  y,
                 z,  0, -x,
                -y,  x,  0))

    def vee(self):
        r"""
        Return a 3x1 vector from a skew-symmetric matrix representing the cross product,
        so that ``self * b`` is equivalent to  ``self.vee().cross(b)``.

        Examples
        ========

        Calling ``vee`` creates a vector from a skew-symmetric Matrix:

        >>> from sympy import Matrix
        >>> A = Matrix([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
        >>> a = A.vee()
        >>> a
        Matrix([
        [1],
        [2],
        [3]])

        Calculating the matrix product of the original matrix with a vector
        is equivalent to a cross product:

        >>> b = Matrix([3, 2, 1])
        >>> A * b
        Matrix([
        [-4],
        [ 8],
        [-4]])

        >>> a.cross(b)
        Matrix([
        [-4],
        [ 8],
        [-4]])

        ``vee`` can also be used to retrieve angular velocity expressions.
        Defining a rotation matrix:

        >>> from sympy import rot_ccw_axis3, trigsimp
        >>> from sympy.physics.mechanics import dynamicsymbols
        >>> theta = dynamicsymbols('theta')
        >>> R = rot_ccw_axis3(theta)
        >>> R
        Matrix([
        [cos(theta(t)), -sin(theta(t)), 0],
        [sin(theta(t)),  cos(theta(t)), 0],
        [            0,              0, 1]])

        We can retrieve the angular velocity:

        >>> Omega = R.T * R.diff()
        >>> Omega = trigsimp(Omega)
        >>> Omega.vee()
        Matrix([
        [                      0],
        [                      0],
        [Derivative(theta(t), t)]])

        See Also
        ========

        dot
        cross
        hat
        multiply
        multiply_elementwise
        """

        if self.shape != (3, 3):
            raise ShapeError("Dimensions incorrect, expected (3, 3), got " +
                             str(self.shape))
        elif not self.is_anti_symmetric():
            raise ValueError("Matrix is not skew-symmetric")
        else:
            return self._new(3, 1, (
                 self[2, 1],
                 self[0, 2],
                 self[1, 0]))

    @property
    def D(self):
        """Return Dirac conjugate (if ``self.rows == 4``).

        Examples
        ========

        >>> from sympy import Matrix, I, eye
        >>> m = Matrix((0, 1 + I, 2, 3))
        >>> m.D
        Matrix([[0, 1 - I, -2, -3]])
        >>> m = (eye(4) + I*eye(4))
        >>> m[0, 3] = 2
        >>> m.D
        Matrix([
        [1 - I,     0,      0,      0],
        [    0, 1 - I,      0,      0],
        [    0,     0, -1 + I,      0],
        [    2,     0,      0, -1 + I]])

        If the matrix does not have 4 rows an AttributeError will be raised
        because this property is only defined for matrices with 4 rows.

        >>> Matrix(eye(2)).D
        Traceback (most recent call last):
        ...
        AttributeError: Matrix has no attribute D.

        See Also
        ========

        sympy.matrices.matrixbase.MatrixBase.conjugate: By-element conjugation
        sympy.matrices.matrixbase.MatrixBase.H: Hermite conjugation
        """
        from sympy.physics.matrices import mgamma
        if self.rows != 4:
            # In Python 3.2, properties can only return an AttributeError
            # so we can't raise a ShapeError -- see commit which added the
            # first line of this inline comment. Also, there is no need
            # for a message since MatrixBase will raise the AttributeError
            raise AttributeError
        return self.H * mgamma(0)

    def dot(self, b, hermitian=None, conjugate_convention=None):
        """Return the dot or inner product of two vectors of equal length.
        Here ``self`` must be a ``Matrix`` of size 1 x n or n x 1, and ``b``
        must be either a matrix of size 1 x n, n x 1, or a list/tuple of length n.
        A scalar is returned.

        By default, ``dot`` does not conjugate ``self`` or ``b``, even if there are
        complex entries. Set ``hermitian=True`` (and optionally a ``conjugate_convention``)
        to compute the hermitian inner product.

        Possible kwargs are ``hermitian`` and ``conjugate_convention``.

        If ``conjugate_convention`` is ``"left"``, ``"math"`` or ``"maths"``,
        the conjugate of the first vector (``self``) is used.  If ``"right"``
        or ``"physics"`` is specified, the conjugate of the second vector ``b`` is used.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> v = Matrix([1, 1, 1])
        >>> M.row(0).dot(v)
        6
        >>> M.col(0).dot(v)
        12
        >>> v = [3, 2, 1]
        >>> M.row(0).dot(v)
        10

        >>> from sympy import I
        >>> q = Matrix([1*I, 1*I, 1*I])
        >>> q.dot(q, hermitian=False)
        -3

        >>> q.dot(q, hermitian=True)
        3

        >>> q1 = Matrix([1, 1, 1*I])
        >>> q.dot(q1, hermitian=True, conjugate_convention="maths")
        1 - 2*I
        >>> q.dot(q1, hermitian=True, conjugate_convention="physics")
        1 + 2*I


        See Also
        ========

        cross
        multiply
        multiply_elementwise
        """
        from .dense import Matrix

        if not isinstance(b, MatrixBase):
            if is_sequence(b):
                if len(b) != self.cols and len(b) != self.rows:
                    raise ShapeError(
                        "Dimensions incorrect for dot product: %s, %s" % (
                            self.shape, len(b)))
                return self.dot(Matrix(b))
            else:
                raise TypeError(
                    "`b` must be an ordered iterable or Matrix, not %s." %
                    type(b))

        if (1 not in self.shape) or (1 not in b.shape):
            raise ShapeError
        if len(self) != len(b):
            raise ShapeError(
                "Dimensions incorrect for dot product: %s, %s" % (self.shape, b.shape))

        mat = self
        n = len(mat)
        if mat.shape != (1, n):
            mat = mat.reshape(1, n)
        if b.shape != (n, 1):
            b = b.reshape(n, 1)

        # Now ``mat`` is a row vector and ``b`` is a column vector.

        # If it so happens that only conjugate_convention is passed
        # then automatically set hermitian to True. If only hermitian
        # is true but no conjugate_convention is not passed then
        # automatically set it to ``"maths"``

        if conjugate_convention is not None and hermitian is None:
            hermitian = True
        if hermitian and conjugate_convention is None:
            conjugate_convention = "maths"

        if hermitian == True:
            if conjugate_convention in ("maths", "left", "math"):
                mat = mat.conjugate()
            elif conjugate_convention in ("physics", "right"):
                b = b.conjugate()
            else:
                raise ValueError("Unknown conjugate_convention was entered."
                                 " conjugate_convention must be one of the"
                                 " following: math, maths, left, physics or right.")
        return (mat * b)[0]

    def dual(self):
        """Returns the dual of a matrix.

        A dual of a matrix is:

        ``(1/2)*levicivita(i, j, k, l)*M(k, l)`` summed over indices `k` and `l`

        Since the levicivita method is anti_symmetric for any pairwise
        exchange of indices, the dual of a symmetric matrix is the zero
        matrix. Strictly speaking the dual defined here assumes that the
        'matrix' `M` is a contravariant anti_symmetric second rank tensor,
        so that the dual is a covariant second rank tensor.

        """
        from sympy.matrices import zeros

        M, n = self[:, :], self.rows
        work = zeros(n)
        if self.is_symmetric():
            return work

        for i in range(1, n):
            for j in range(1, n):
                acum = 0
                for k in range(1, n):
                    acum += LeviCivita(i, j, 0, k) * M[0, k]
                work[i, j] = acum
                work[j, i] = -acum

        for l in range(1, n):
            acum = 0
            for a in range(1, n):
                for b in range(1, n):
                    acum += LeviCivita(0, l, a, b) * M[a, b]
            acum /= 2
            work[0, l] = -acum
            work[l, 0] = acum

        return work

    def _eval_matrix_exp_jblock(self):
        """A helper function to compute an exponential of a Jordan block
        matrix

        Examples
        ========

        >>> from sympy import Symbol, Matrix
        >>> l = Symbol('lamda')

        A trivial example of 1*1 Jordan block:

        >>> m = Matrix.jordan_block(1, l)
        >>> m._eval_matrix_exp_jblock()
        Matrix([[exp(lamda)]])

        An example of 3*3 Jordan block:

        >>> m = Matrix.jordan_block(3, l)
        >>> m._eval_matrix_exp_jblock()
        Matrix([
        [exp(lamda), exp(lamda), exp(lamda)/2],
        [         0, exp(lamda),   exp(lamda)],
        [         0,          0,   exp(lamda)]])

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Matrix_function#Jordan_decomposition
        """
        size = self.rows
        l = self[0, 0]
        exp_l = exp(l)

        bands = {i: exp_l / factorial(i) for i in range(size)}

        from .sparsetools import banded
        return self.__class__(banded(size, bands))


    def analytic_func(self, f, x):
        """
        Computes f(A) where A is a Square Matrix
        and f is an analytic function.

        Examples
        ========

        >>> from sympy import Symbol, Matrix, S, log

        >>> x = Symbol('x')
        >>> m = Matrix([[S(5)/4, S(3)/4], [S(3)/4, S(5)/4]])
        >>> f = log(x)
        >>> m.analytic_func(f, x)
        Matrix([
        [     0, log(2)],
        [log(2),      0]])

        Parameters
        ==========

        f : Expr
            Analytic Function
        x : Symbol
            parameter of f

        """

        f, x = _sympify(f), _sympify(x)
        if not self.is_square:
            raise NonSquareMatrixError
        if not x.is_symbol:
            raise ValueError("{} must be a symbol.".format(x))
        if x not in f.free_symbols:
            raise ValueError(
                "{} must be a parameter of {}.".format(x, f))
        if x in self.free_symbols:
            raise ValueError(
                "{} must not be a parameter of {}.".format(x, self))

        eigen = self.eigenvals()
        max_mul = max(eigen.values())
        derivative = {}
        dd = f
        for i in range(max_mul - 1):
            dd = diff(dd, x)
            derivative[i + 1] = dd
        n = self.shape[0]
        r = self.zeros(n)
        f_val = self.zeros(n, 1)
        row = 0

        for i in eigen:
            mul = eigen[i]
            f_val[row] = f.subs(x, i)
            if f_val[row].is_number and not f_val[row].is_complex:
                raise ValueError(
                    "Cannot evaluate the function because the "
                    "function {} is not analytic at the given "
                    "eigenvalue {}".format(f, f_val[row]))
            val = 1
            for a in range(n):
                r[row, a] = val
                val *= i
            if mul > 1:
                coe = [1 for ii in range(n)]
                deri = 1
                while mul > 1:
                    row = row + 1
                    mul -= 1
                    d_i = derivative[deri].subs(x, i)
                    if d_i.is_number and not d_i.is_complex:
                        raise ValueError(
                            "Cannot evaluate the function because the "
                            "derivative {} is not analytic at the given "
                            "eigenvalue {}".format(derivative[deri], d_i))
                    f_val[row] = d_i
                    for a in range(n):
                        if a - deri + 1 <= 0:
                            r[row, a] = 0
                            coe[a] = 0
                            continue
                        coe[a] = coe[a]*(a - deri + 1)
                        r[row, a] = coe[a]*pow(i, a - deri)
                    deri += 1
            row += 1
        c = r.solve(f_val)
        ans = self.zeros(n)
        pre = self.eye(n)
        for i in range(n):
            ans = ans + c[i]*pre
            pre *= self
        return ans


    def exp(self):
        """Return the exponential of a square matrix.

        Examples
        ========

        >>> from sympy import Symbol, Matrix

        >>> t = Symbol('t')
        >>> m = Matrix([[0, 1], [-1, 0]]) * t
        >>> m.exp()
        Matrix([
        [    exp(I*t)/2 + exp(-I*t)/2, -I*exp(I*t)/2 + I*exp(-I*t)/2],
        [I*exp(I*t)/2 - I*exp(-I*t)/2,      exp(I*t)/2 + exp(-I*t)/2]])
        """
        if not self.is_square:
            raise NonSquareMatrixError(
                "Exponentiation is valid only for square matrices")
        try:
            P, J = self.jordan_form()
            cells = J.get_diag_blocks()
        except MatrixError:
            raise NotImplementedError(
                "Exponentiation is implemented only for matrices for which the Jordan normal form can be computed")

        blocks = [cell._eval_matrix_exp_jblock() for cell in cells]
        from sympy.matrices import diag
        eJ = diag(*blocks)
        # n = self.rows
        ret = P.multiply(eJ, dotprodsimp=None).multiply(P.inv(), dotprodsimp=None)
        if all(value.is_real for value in self.values()):
            return type(self)(re(ret))
        else:
            return type(self)(ret)

    def _eval_matrix_log_jblock(self):
        """Helper function to compute logarithm of a jordan block.

        Examples
        ========

        >>> from sympy import Symbol, Matrix
        >>> l = Symbol('lamda')

        A trivial example of 1*1 Jordan block:

        >>> m = Matrix.jordan_block(1, l)
        >>> m._eval_matrix_log_jblock()
        Matrix([[log(lamda)]])

        An example of 3*3 Jordan block:

        >>> m = Matrix.jordan_block(3, l)
        >>> m._eval_matrix_log_jblock()
        Matrix([
        [log(lamda),    1/lamda, -1/(2*lamda**2)],
        [         0, log(lamda),         1/lamda],
        [         0,          0,      log(lamda)]])
        """
        size = self.rows
        l = self[0, 0]

        if l.is_zero:
            raise MatrixError(
                'Could not take logarithm or reciprocal for the given '
                'eigenvalue {}'.format(l))

        bands = {0: log(l)}
        for i in range(1, size):
            bands[i] = -((-l) ** -i) / i

        from .sparsetools import banded
        return self.__class__(banded(size, bands))

    def log(self, simplify=cancel):
        """Return the logarithm of a square matrix.

        Parameters
        ==========

        simplify : function, bool
            The function to simplify the result with.

            Default is ``cancel``, which is effective to reduce the
            expression growing for taking reciprocals and inverses for
            symbolic matrices.

        Examples
        ========

        >>> from sympy import S, Matrix

        Examples for positive-definite matrices:

        >>> m = Matrix([[1, 1], [0, 1]])
        >>> m.log()
        Matrix([
        [0, 1],
        [0, 0]])

        >>> m = Matrix([[S(5)/4, S(3)/4], [S(3)/4, S(5)/4]])
        >>> m.log()
        Matrix([
        [     0, log(2)],
        [log(2),      0]])

        Examples for non positive-definite matrices:

        >>> m = Matrix([[S(3)/4, S(5)/4], [S(5)/4, S(3)/4]])
        >>> m.log()
        Matrix([
        [         I*pi/2, log(2) - I*pi/2],
        [log(2) - I*pi/2,          I*pi/2]])

        >>> m = Matrix(
        ...     [[0, 0, 0, 1],
        ...      [0, 0, 1, 0],
        ...      [0, 1, 0, 0],
        ...      [1, 0, 0, 0]])
        >>> m.log()
        Matrix([
        [ I*pi/2,       0,       0, -I*pi/2],
        [      0,  I*pi/2, -I*pi/2,       0],
        [      0, -I*pi/2,  I*pi/2,       0],
        [-I*pi/2,       0,       0,  I*pi/2]])
        """
        if not self.is_square:
            raise NonSquareMatrixError(
                "Logarithm is valid only for square matrices")

        try:
            if simplify:
                P, J = simplify(self).jordan_form()
            else:
                P, J = self.jordan_form()

            cells = J.get_diag_blocks()
        except MatrixError:
            raise NotImplementedError(
                "Logarithm is implemented only for matrices for which "
                "the Jordan normal form can be computed")

        blocks = [
            cell._eval_matrix_log_jblock()
            for cell in cells]
        from sympy.matrices import diag
        eJ = diag(*blocks)

        if simplify:
            ret = simplify(P * eJ * simplify(P.inv()))
            ret = self.__class__(ret)
        else:
            ret = P * eJ * P.inv()

        return ret

    def is_nilpotent(self):
        """Checks if a matrix is nilpotent.

        A matrix B is nilpotent if for some integer k, B**k is
        a zero matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> a = Matrix([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        >>> a.is_nilpotent()
        True

        >>> a = Matrix([[1, 0, 1], [1, 0, 0], [1, 1, 0]])
        >>> a.is_nilpotent()
        False
        """
        if not self:
            return True
        if not self.is_square:
            raise NonSquareMatrixError(
                "Nilpotency is valid only for square matrices")
        x = uniquely_named_symbol('x', self, modify=lambda s: '_' + s)
        p = self.charpoly(x)
        if p.args[0] == x ** self.rows:
            return True
        return False

    def key2bounds(self, keys):
        """Converts a key with potentially mixed types of keys (integer and slice)
        into a tuple of ranges and raises an error if any index is out of ``self``'s
        range.

        See Also
        ========

        key2ij
        """
        islice, jslice = [isinstance(k, slice) for k in keys]
        if islice:
            if not self.rows:
                rlo = rhi = 0
            else:
                rlo, rhi = keys[0].indices(self.rows)[:2]
        else:
            rlo = a2idx(keys[0], self.rows)
            rhi = rlo + 1
        if jslice:
            if not self.cols:
                clo = chi = 0
            else:
                clo, chi = keys[1].indices(self.cols)[:2]
        else:
            clo = a2idx(keys[1], self.cols)
            chi = clo + 1
        return rlo, rhi, clo, chi

    def key2ij(self, key):
        """Converts key into canonical form, converting integers or indexable
        items into valid integers for ``self``'s range or returning slices
        unchanged.

        See Also
        ========

        key2bounds
        """
        if is_sequence(key):
            if not len(key) == 2:
                raise TypeError('key must be a sequence of length 2')
            return [a2idx(i, n) if not isinstance(i, slice) else i
                    for i, n in zip(key, self.shape)]
        elif isinstance(key, slice):
            return key.indices(len(self))[:2]
        else:
            return divmod(a2idx(key, len(self)), self.cols)

    def normalized(self, iszerofunc=_iszero):
        """Return the normalized version of ``self``.

        Parameters
        ==========

        iszerofunc : Function, optional
            A function to determine whether ``self`` is a zero vector.
            The default ``_iszero`` tests to see if each element is
            exactly zero.

        Returns
        =======

        Matrix
            Normalized vector form of ``self``.
            It has the same length as a unit vector. However, a zero vector
            will be returned for a vector with norm 0.

        Raises
        ======

        ShapeError
            If the matrix is not in a vector form.

        See Also
        ========

        norm
        """
        if self.rows != 1 and self.cols != 1:
            raise ShapeError("A Matrix must be a vector to normalize.")
        norm = self.norm()
        if iszerofunc(norm):
            out = self.zeros(self.rows, self.cols)
        else:
            out = self.applyfunc(lambda i: i / norm)
        return out

    def norm(self, ord=None):
        """Return the Norm of a Matrix or Vector.

        In the simplest case this is the geometric size of the vector
        Other norms can be specified by the ord parameter


        =====  ============================  ==========================
        ord    norm for matrices             norm for vectors
        =====  ============================  ==========================
        None   Frobenius norm                2-norm
        'fro'  Frobenius norm                - does not exist
        inf    maximum row sum               max(abs(x))
        -inf   --                            min(abs(x))
        1      maximum column sum            as below
        -1     --                            as below
        2      2-norm (largest sing. value)  as below
        -2     smallest singular value       as below
        other  - does not exist              sum(abs(x)**ord)**(1./ord)
        =====  ============================  ==========================

        Examples
        ========

        >>> from sympy import Matrix, Symbol, trigsimp, cos, sin, oo
        >>> x = Symbol('x', real=True)
        >>> v = Matrix([cos(x), sin(x)])
        >>> trigsimp( v.norm() )
        1
        >>> v.norm(10)
        (sin(x)**10 + cos(x)**10)**(1/10)
        >>> A = Matrix([[1, 1], [1, 1]])
        >>> A.norm(1) # maximum sum of absolute values of A is 2
        2
        >>> A.norm(2) # Spectral norm (max of |Ax|/|x| under 2-vector-norm)
        2
        >>> A.norm(-2) # Inverse spectral norm (smallest singular value)
        0
        >>> A.norm() # Frobenius Norm
        2
        >>> A.norm(oo) # Infinity Norm
        2
        >>> Matrix([1, -2]).norm(oo)
        2
        >>> Matrix([-1, 2]).norm(-oo)
        1

        See Also
        ========

        normalized
        """
        # Row or Column Vector Norms
        vals = list(self.values()) or [0]
        if S.One in self.shape:
            if ord in (2, None):  # Common case sqrt(<x, x>)
                return sqrt(Add(*(abs(i) ** 2 for i in vals)))

            elif ord == 1:  # sum(abs(x))
                return Add(*(abs(i) for i in vals))

            elif ord is S.Infinity:  # max(abs(x))
                return Max(*[abs(i) for i in vals])

            elif ord is S.NegativeInfinity:  # min(abs(x))
                return Min(*[abs(i) for i in vals])

            # Otherwise generalize the 2-norm, Sum(x_i**ord)**(1/ord)
            # Note that while useful this is not mathematically a norm
            try:
                return Pow(Add(*(abs(i) ** ord for i in vals)), S.One / ord)
            except (NotImplementedError, TypeError):
                raise ValueError("Expected order to be Number, Symbol, oo")

        # Matrix Norms
        else:
            if ord == 1:  # Maximum column sum
                m = self.applyfunc(abs)
                return Max(*[sum(m.col(i)) for i in range(m.cols)])

            elif ord == 2:  # Spectral Norm
                # Maximum singular value
                return Max(*self.singular_values())

            elif ord == -2:
                # Minimum singular value
                return Min(*self.singular_values())

            elif ord is S.Infinity:   # Infinity Norm - Maximum row sum
                m = self.applyfunc(abs)
                return Max(*[sum(m.row(i)) for i in range(m.rows)])

            elif (ord is None or isinstance(ord,
                                            str) and ord.lower() in
                ['f', 'fro', 'frobenius', 'vector']):
                # Reshape as vector and send back to norm function
                return self.vec().norm(ord=2)

            else:
                raise NotImplementedError("Matrix Norms under development")

    def print_nonzero(self, symb="X"):
        """Shows location of non-zero entries for fast shape lookup.

        Examples
        ========

        >>> from sympy import Matrix, eye
        >>> m = Matrix(2, 3, lambda i, j: i*3+j)
        >>> m
        Matrix([
        [0, 1, 2],
        [3, 4, 5]])
        >>> m.print_nonzero()
        [ XX]
        [XXX]
        >>> m = eye(4)
        >>> m.print_nonzero("x")
        [x   ]
        [ x  ]
        [  x ]
        [   x]

        """
        s = []
        for i in range(self.rows):
            line = []
            for j in range(self.cols):
                if self[i, j] == 0:
                    line.append(" ")
                else:
                    line.append(str(symb))
            s.append("[%s]" % ''.join(line))
        print('\n'.join(s))

    def project(self, v):
        """Return the projection of ``self`` onto the line containing ``v``.

        Examples
        ========

        >>> from sympy import Matrix, S, sqrt
        >>> V = Matrix([sqrt(3)/2, S.Half])
        >>> x = Matrix([[1, 0]])
        >>> V.project(x)
        Matrix([[sqrt(3)/2, 0]])
        >>> V.project(-x)
        Matrix([[sqrt(3)/2, 0]])
        """
        return v * (self.dot(v) / v.dot(v))

    def table(self, printer, rowstart='[', rowend=']', rowsep='\n',
              colsep=', ', align='right'):
        r"""
        String form of Matrix as a table.

        ``printer`` is the printer to use for on the elements (generally
        something like StrPrinter())

        ``rowstart`` is the string used to start each row (by default '[').

        ``rowend`` is the string used to end each row (by default ']').

        ``rowsep`` is the string used to separate rows (by default a newline).

        ``colsep`` is the string used to separate columns (by default ', ').

        ``align`` defines how the elements are aligned. Must be one of 'left',
        'right', or 'center'.  You can also use '<', '>', and '^' to mean the
        same thing, respectively.

        This is used by the string printer for Matrix.

        Examples
        ========

        >>> from sympy import Matrix, StrPrinter
        >>> M = Matrix([[1, 2], [-33, 4]])
        >>> printer = StrPrinter()
        >>> M.table(printer)
        '[  1, 2]\n[-33, 4]'
        >>> print(M.table(printer))
        [  1, 2]
        [-33, 4]
        >>> print(M.table(printer, rowsep=',\n'))
        [  1, 2],
        [-33, 4]
        >>> print('[%s]' % M.table(printer, rowsep=',\n'))
        [[  1, 2],
        [-33, 4]]
        >>> print(M.table(printer, colsep=' '))
        [  1 2]
        [-33 4]
        >>> print(M.table(printer, align='center'))
        [ 1 , 2]
        [-33, 4]
        >>> print(M.table(printer, rowstart='{', rowend='}'))
        {  1, 2}
        {-33, 4}
        """
        # Handle zero dimensions:
        if S.Zero in self.shape:
            return '[]'
        # Build table of string representations of the elements
        res = []
        # Track per-column max lengths for pretty alignment
        maxlen = [0] * self.cols
        for i in range(self.rows):
            res.append([])
            for j in range(self.cols):
                s = printer._print(self[i, j])
                res[-1].append(s)
                maxlen[j] = max(len(s), maxlen[j])
        # Patch strings together
        align = {
            'left': 'ljust',
            'right': 'rjust',
            'center': 'center',
            '<': 'ljust',
            '>': 'rjust',
            '^': 'center',
        }[align]
        for i, row in enumerate(res):
            for j, elem in enumerate(row):
                row[j] = getattr(elem, align)(maxlen[j])
            res[i] = rowstart + colsep.join(row) + rowend
        return rowsep.join(res)

    def rank_decomposition(self, iszerofunc=_iszero, simplify=False):
        return _rank_decomposition(self, iszerofunc=iszerofunc,
                simplify=simplify)

    def cholesky(self, hermitian=True):
        raise NotImplementedError('This function is implemented in DenseMatrix or SparseMatrix')

    def LDLdecomposition(self, hermitian=True):
        raise NotImplementedError('This function is implemented in DenseMatrix or SparseMatrix')

    def LUdecomposition(self, iszerofunc=_iszero, simpfunc=None,
            rankcheck=False):
        return _LUdecomposition(self, iszerofunc=iszerofunc, simpfunc=simpfunc,
                rankcheck=rankcheck)

    def LUdecomposition_Simple(self, iszerofunc=_iszero, simpfunc=None,
            rankcheck=False):
        return _LUdecomposition_Simple(self, iszerofunc=iszerofunc,
                simpfunc=simpfunc, rankcheck=rankcheck)

    def LUdecompositionFF(self):
        return _LUdecompositionFF(self)

    def singular_value_decomposition(self):
        return _singular_value_decomposition(self)

    def QRdecomposition(self):
        return _QRdecomposition(self)

    def upper_hessenberg_decomposition(self):
        return _upper_hessenberg_decomposition(self)

    def diagonal_solve(self, rhs):
        return _diagonal_solve(self, rhs)

    def lower_triangular_solve(self, rhs):
        raise NotImplementedError('This function is implemented in DenseMatrix or SparseMatrix')

    def upper_triangular_solve(self, rhs):
        raise NotImplementedError('This function is implemented in DenseMatrix or SparseMatrix')

    def cholesky_solve(self, rhs):
        return _cholesky_solve(self, rhs)

    def LDLsolve(self, rhs):
        return _LDLsolve(self, rhs)

    def LUsolve(self, rhs, iszerofunc=_iszero):
        return _LUsolve(self, rhs, iszerofunc=iszerofunc)

    def QRsolve(self, b):
        return _QRsolve(self, b)

    def gauss_jordan_solve(self, B, freevar=False):
        return _gauss_jordan_solve(self, B, freevar=freevar)

    def pinv_solve(self, B, arbitrary_matrix=None):
        return _pinv_solve(self, B, arbitrary_matrix=arbitrary_matrix)

    def cramer_solve(self, rhs, det_method="laplace"):
        return _cramer_solve(self, rhs, det_method=det_method)

    def solve(self, rhs, method='GJ'):
        return _solve(self, rhs, method=method)

    def solve_least_squares(self, rhs, method='CH'):
        return _solve_least_squares(self, rhs, method=method)

    def pinv(self, method='RD'):
        return _pinv(self, method=method)

    def inverse_ADJ(self, iszerofunc=_iszero):
        return _inv_ADJ(self, iszerofunc=iszerofunc)

    def inverse_BLOCK(self, iszerofunc=_iszero):
        return _inv_block(self, iszerofunc=iszerofunc)

    def inverse_GE(self, iszerofunc=_iszero):
        return _inv_GE(self, iszerofunc=iszerofunc)

    def inverse_LU(self, iszerofunc=_iszero):
        return _inv_LU(self, iszerofunc=iszerofunc)

    def inverse_CH(self, iszerofunc=_iszero):
        return _inv_CH(self, iszerofunc=iszerofunc)

    def inverse_LDL(self, iszerofunc=_iszero):
        return _inv_LDL(self, iszerofunc=iszerofunc)

    def inverse_QR(self, iszerofunc=_iszero):
        return _inv_QR(self, iszerofunc=iszerofunc)

    def inv(self, method=None, iszerofunc=_iszero, try_block_diag=False):
        return _inv(self, method=method, iszerofunc=iszerofunc,
                try_block_diag=try_block_diag)

    def connected_components(self):
        return _connected_components(self)

    def connected_components_decomposition(self):
        return _connected_components_decomposition(self)

    def strongly_connected_components(self):
        return _strongly_connected_components(self)

    def strongly_connected_components_decomposition(self, lower=True):
        return _strongly_connected_components_decomposition(self, lower=lower)

    _sage_ = Basic._sage_

    rank_decomposition.__doc__     = _rank_decomposition.__doc__
    cholesky.__doc__               = _cholesky.__doc__
    LDLdecomposition.__doc__       = _LDLdecomposition.__doc__
    LUdecomposition.__doc__        = _LUdecomposition.__doc__
    LUdecomposition_Simple.__doc__ = _LUdecomposition_Simple.__doc__
    LUdecompositionFF.__doc__      = _LUdecompositionFF.__doc__
    singular_value_decomposition.__doc__ = _singular_value_decomposition.__doc__
    QRdecomposition.__doc__        = _QRdecomposition.__doc__
    upper_hessenberg_decomposition.__doc__ = _upper_hessenberg_decomposition.__doc__

    diagonal_solve.__doc__         = _diagonal_solve.__doc__
    lower_triangular_solve.__doc__ = _lower_triangular_solve.__doc__
    upper_triangular_solve.__doc__ = _upper_triangular_solve.__doc__
    cholesky_solve.__doc__         = _cholesky_solve.__doc__
    LDLsolve.__doc__               = _LDLsolve.__doc__
    LUsolve.__doc__                = _LUsolve.__doc__
    QRsolve.__doc__                = _QRsolve.__doc__
    gauss_jordan_solve.__doc__     = _gauss_jordan_solve.__doc__
    pinv_solve.__doc__             = _pinv_solve.__doc__
    cramer_solve.__doc__           = _cramer_solve.__doc__
    solve.__doc__                  = _solve.__doc__
    solve_least_squares.__doc__    = _solve_least_squares.__doc__

    pinv.__doc__                   = _pinv.__doc__
    inverse_ADJ.__doc__            = _inv_ADJ.__doc__
    inverse_GE.__doc__             = _inv_GE.__doc__
    inverse_LU.__doc__             = _inv_LU.__doc__
    inverse_CH.__doc__             = _inv_CH.__doc__
    inverse_LDL.__doc__            = _inv_LDL.__doc__
    inverse_QR.__doc__             = _inv_QR.__doc__
    inverse_BLOCK.__doc__          = _inv_block.__doc__
    inv.__doc__                    = _inv.__doc__

    connected_components.__doc__   = _connected_components.__doc__
    connected_components_decomposition.__doc__ = \
        _connected_components_decomposition.__doc__
    strongly_connected_components.__doc__   = \
        _strongly_connected_components.__doc__
    strongly_connected_components_decomposition.__doc__ = \
        _strongly_connected_components_decomposition.__doc__


def _convert_matrix(typ, mat):
    """Convert mat to a Matrix of type typ."""
    from sympy.matrices.matrixbase import MatrixBase
    if getattr(mat, "is_Matrix", False) and not isinstance(mat, MatrixBase):
        # This is needed for interop between Matrix and the redundant matrix
        # mixin types like _MinimalMatrix etc. If anyone should happen to be
        # using those then this keeps them working. Really _MinimalMatrix etc
        # should be deprecated and removed though.
        return typ(*mat.shape, list(mat))
    else:
        return typ(mat)


def _has_matrix_shape(other):
    shape = getattr(other, 'shape', None)
    if shape is None:
        return False
    return isinstance(shape, tuple) and len(shape) == 2


def _has_rows_cols(other):
    return hasattr(other, 'rows') and hasattr(other, 'cols')


def _coerce_operand(self, other):
    """Convert other to a Matrix, or check for possible scalar."""

    INVALID = None, 'invalid_type'

    # Disallow mixing Matrix and Array
    if isinstance(other, NDimArray):
        return INVALID

    is_Matrix = getattr(other, 'is_Matrix', None)

    # Return a Matrix as-is
    if is_Matrix:
        return other, 'is_matrix'

    # Try to convert numpy array, mpmath matrix etc.
    if is_Matrix is None:
        if _has_matrix_shape(other) or _has_rows_cols(other):
            return _convert_matrix(type(self), other), 'is_matrix'

    # Could be a scalar but only if not iterable...
    if not isinstance(other, Iterable):
        return other, 'possible_scalar'

    return INVALID


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


def _unify_with_other(self, other):
    """Unify self and other into a single matrix type, or check for scalar."""
    other, T = _coerce_operand(self, other)

    if T == "is_matrix":
        typ = classof(self, other)
        if typ != self.__class__:
            self = _convert_matrix(typ, self)
        if typ != other.__class__:
            other = _convert_matrix(typ, other)

    return self, other, T


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


class DeferredVector(Symbol, NotIterable): # type: ignore
    """A vector whose components are deferred (e.g. for use with lambdify).

    Examples
    ========

    >>> from sympy import DeferredVector, lambdify
    >>> X = DeferredVector( 'X' )
    >>> X
    X
    >>> expr = (X[0] + 2, X[2] + 3)
    >>> func = lambdify( X, expr)
    >>> func( [1, 2, 3] )
    (3, 6)
    """

    def __getitem__(self, i):
        if i == -0:
            i = 0
        if i < 0:
            raise IndexError('DeferredVector index out of range')
        component_name = '%s[%d]' % (self.name, i)
        return Symbol(component_name)

    def __str__(self):
        return sstr(self)

    def __repr__(self):
        return "DeferredVector('%s')" % self.name

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\session.py ===
# orm/session.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Provides the Session class and related utilities."""

from __future__ import annotations

import contextlib
from enum import Enum
import itertools
import sys
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
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
import weakref

from . import attributes
from . import bulk_persistence
from . import context
from . import descriptor_props
from . import exc
from . import identity
from . import loading
from . import query
from . import state as statelib
from ._typing import _O
from ._typing import insp_is_mapper
from ._typing import is_composite_class
from ._typing import is_orm_option
from ._typing import is_user_defined_option
from .base import _class_to_mapper
from .base import _none_set
from .base import _state_mapper
from .base import instance_str
from .base import LoaderCallableStatus
from .base import object_mapper
from .base import object_state
from .base import PassiveFlag
from .base import state_str
from .context import FromStatement
from .context import ORMCompileState
from .identity import IdentityMap
from .query import Query
from .state import InstanceState
from .state_changes import _StateChange
from .state_changes import _StateChangeState
from .state_changes import _StateChangeStates
from .unitofwork import UOWTransaction
from .. import engine
from .. import exc as sa_exc
from .. import sql
from .. import util
from ..engine import Connection
from ..engine import Engine
from ..engine.util import TransactionalContext
from ..event import dispatcher
from ..event import EventTarget
from ..inspection import inspect
from ..inspection import Inspectable
from ..sql import coercions
from ..sql import dml
from ..sql import roles
from ..sql import Select
from ..sql import TableClause
from ..sql import visitors
from ..sql.base import _NoArg
from ..sql.base import CompileState
from ..sql.schema import Table
from ..sql.selectable import ForUpdateArg
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..util import IdentitySet
from ..util.typing import Literal
from ..util.typing import Protocol

if typing.TYPE_CHECKING:
    from ._typing import _EntityType
    from ._typing import _IdentityKeyType
    from ._typing import _InstanceDict
    from ._typing import OrmExecuteOptionsParameter
    from .interfaces import ORMOption
    from .interfaces import UserDefinedOption
    from .mapper import Mapper
    from .path_registry import PathRegistry
    from .query import RowReturningQuery
    from ..engine import CursorResult
    from ..engine import Result
    from ..engine import Row
    from ..engine import RowMapping
    from ..engine.base import Transaction
    from ..engine.base import TwoPhaseTransaction
    from ..engine.interfaces import _CoreAnyExecuteParams
    from ..engine.interfaces import _CoreSingleExecuteParams
    from ..engine.interfaces import _ExecuteOptions
    from ..engine.interfaces import CoreExecuteOptionsParameter
    from ..engine.result import ScalarResult
    from ..event import _InstanceLevelDispatch
    from ..sql._typing import _ColumnsClauseArgument
    from ..sql._typing import _InfoType
    from ..sql._typing import _T0
    from ..sql._typing import _T1
    from ..sql._typing import _T2
    from ..sql._typing import _T3
    from ..sql._typing import _T4
    from ..sql._typing import _T5
    from ..sql._typing import _T6
    from ..sql._typing import _T7
    from ..sql._typing import _TypedColumnClauseArgument as _TCCA
    from ..sql.base import Executable
    from ..sql.base import ExecutableOption
    from ..sql.dml import UpdateBase
    from ..sql.elements import ClauseElement
    from ..sql.roles import TypedColumnsClauseRole
    from ..sql.selectable import ForUpdateParameter
    from ..sql.selectable import TypedReturnsRows

_T = TypeVar("_T", bound=Any)

__all__ = [
    "Session",
    "SessionTransaction",
    "sessionmaker",
    "ORMExecuteState",
    "close_all_sessions",
    "make_transient",
    "make_transient_to_detached",
    "object_session",
]

_sessions: weakref.WeakValueDictionary[int, Session] = (
    weakref.WeakValueDictionary()
)
"""Weak-referencing dictionary of :class:`.Session` objects.
"""

statelib._sessions = _sessions

_PKIdentityArgument = Union[Any, Tuple[Any, ...]]

_BindArguments = Dict[str, Any]

_EntityBindKey = Union[Type[_O], "Mapper[_O]"]
_SessionBindKey = Union[Type[Any], "Mapper[Any]", "TableClause", str]
_SessionBind = Union["Engine", "Connection"]

JoinTransactionMode = Literal[
    "conditional_savepoint",
    "rollback_only",
    "control_fully",
    "create_savepoint",
]


class _ConnectionCallableProto(Protocol):
    """a callable that returns a :class:`.Connection` given an instance.

    This callable, when present on a :class:`.Session`, is called only from the
    ORM's persistence mechanism (i.e. the unit of work flush process) to allow
    for connection-per-instance schemes (i.e. horizontal sharding) to be used
    as persistence time.

    This callable is not present on a plain :class:`.Session`, however
    is established when using the horizontal sharding extension.

    """

    def __call__(
        self,
        mapper: Optional[Mapper[Any]] = None,
        instance: Optional[object] = None,
        **kw: Any,
    ) -> Connection: ...


def _state_session(state: InstanceState[Any]) -> Optional[Session]:
    """Given an :class:`.InstanceState`, return the :class:`.Session`
    associated, if any.
    """
    return state.session


class _SessionClassMethods:
    """Class-level methods for :class:`.Session`, :class:`.sessionmaker`."""

    @classmethod
    @util.deprecated(
        "1.3",
        "The :meth:`.Session.close_all` method is deprecated and will be "
        "removed in a future release.  Please refer to "
        ":func:`.session.close_all_sessions`.",
    )
    def close_all(cls) -> None:
        """Close *all* sessions in memory."""

        close_all_sessions()

    @classmethod
    @util.preload_module("sqlalchemy.orm.util")
    def identity_key(
        cls,
        class_: Optional[Type[Any]] = None,
        ident: Union[Any, Tuple[Any, ...]] = None,
        *,
        instance: Optional[Any] = None,
        row: Optional[Union[Row[Any], RowMapping]] = None,
        identity_token: Optional[Any] = None,
    ) -> _IdentityKeyType[Any]:
        """Return an identity key.

        This is an alias of :func:`.util.identity_key`.

        """
        return util.preloaded.orm_util.identity_key(
            class_,
            ident,
            instance=instance,
            row=row,
            identity_token=identity_token,
        )

    @classmethod
    def object_session(cls, instance: object) -> Optional[Session]:
        """Return the :class:`.Session` to which an object belongs.

        This is an alias of :func:`.object_session`.

        """

        return object_session(instance)


class SessionTransactionState(_StateChangeState):
    ACTIVE = 1
    PREPARED = 2
    COMMITTED = 3
    DEACTIVE = 4
    CLOSED = 5
    PROVISIONING_CONNECTION = 6


# backwards compatibility
ACTIVE, PREPARED, COMMITTED, DEACTIVE, CLOSED, PROVISIONING_CONNECTION = tuple(
    SessionTransactionState
)


class ORMExecuteState(util.MemoizedSlots):
    """Represents a call to the :meth:`_orm.Session.execute` method, as passed
    to the :meth:`.SessionEvents.do_orm_execute` event hook.

    .. versionadded:: 1.4

    .. seealso::

        :ref:`session_execute_events` - top level documentation on how
        to use :meth:`_orm.SessionEvents.do_orm_execute`

    """

    __slots__ = (
        "session",
        "statement",
        "parameters",
        "execution_options",
        "local_execution_options",
        "bind_arguments",
        "identity_token",
        "_compile_state_cls",
        "_starting_event_idx",
        "_events_todo",
        "_update_execution_options",
    )

    session: Session
    """The :class:`_orm.Session` in use."""

    statement: Executable
    """The SQL statement being invoked.

    For an ORM selection as would
    be retrieved from :class:`_orm.Query`, this is an instance of
    :class:`_sql.select` that was generated from the ORM query.
    """

    parameters: Optional[_CoreAnyExecuteParams]
    """Dictionary of parameters that was passed to
    :meth:`_orm.Session.execute`."""

    execution_options: _ExecuteOptions
    """The complete dictionary of current execution options.

    This is a merge of the statement level options with the
    locally passed execution options.

    .. seealso::

        :attr:`_orm.ORMExecuteState.local_execution_options`

        :meth:`_sql.Executable.execution_options`

        :ref:`orm_queryguide_execution_options`

    """

    local_execution_options: _ExecuteOptions
    """Dictionary view of the execution options passed to the
    :meth:`.Session.execute` method.

    This does not include options that may be associated with the statement
    being invoked.

    .. seealso::

        :attr:`_orm.ORMExecuteState.execution_options`

    """

    bind_arguments: _BindArguments
    """The dictionary passed as the
    :paramref:`_orm.Session.execute.bind_arguments` dictionary.

    This dictionary may be used by extensions to :class:`_orm.Session` to pass
    arguments that will assist in determining amongst a set of database
    connections which one should be used to invoke this statement.

    """

    _compile_state_cls: Optional[Type[ORMCompileState]]
    _starting_event_idx: int
    _events_todo: List[Any]
    _update_execution_options: Optional[_ExecuteOptions]

    def __init__(
        self,
        session: Session,
        statement: Executable,
        parameters: Optional[_CoreAnyExecuteParams],
        execution_options: _ExecuteOptions,
        bind_arguments: _BindArguments,
        compile_state_cls: Optional[Type[ORMCompileState]],
        events_todo: List[_InstanceLevelDispatch[Session]],
    ):
        """Construct a new :class:`_orm.ORMExecuteState`.

        this object is constructed internally.

        """
        self.session = session
        self.statement = statement
        self.parameters = parameters
        self.local_execution_options = execution_options
        self.execution_options = statement._execution_options.union(
            execution_options
        )
        self.bind_arguments = bind_arguments
        self._compile_state_cls = compile_state_cls
        self._events_todo = list(events_todo)

    def _remaining_events(self) -> List[_InstanceLevelDispatch[Session]]:
        return self._events_todo[self._starting_event_idx + 1 :]

    def invoke_statement(
        self,
        statement: Optional[Executable] = None,
        params: Optional[_CoreAnyExecuteParams] = None,
        execution_options: Optional[OrmExecuteOptionsParameter] = None,
        bind_arguments: Optional[_BindArguments] = None,
    ) -> Result[Any]:
        """Execute the statement represented by this
        :class:`.ORMExecuteState`, without re-invoking events that have
        already proceeded.

        This method essentially performs a re-entrant execution of the current
        statement for which the :meth:`.SessionEvents.do_orm_execute` event is
        being currently invoked.    The use case for this is for event handlers
        that want to override how the ultimate
        :class:`_engine.Result` object is returned, such as for schemes that
        retrieve results from an offline cache or which concatenate results
        from multiple executions.

        When the :class:`_engine.Result` object is returned by the actual
        handler function within :meth:`_orm.SessionEvents.do_orm_execute` and
        is propagated to the calling
        :meth:`_orm.Session.execute` method, the remainder of the
        :meth:`_orm.Session.execute` method is preempted and the
        :class:`_engine.Result` object is returned to the caller of
        :meth:`_orm.Session.execute` immediately.

        :param statement: optional statement to be invoked, in place of the
         statement currently represented by :attr:`.ORMExecuteState.statement`.

        :param params: optional dictionary of parameters or list of parameters
         which will be merged into the existing
         :attr:`.ORMExecuteState.parameters` of this :class:`.ORMExecuteState`.

         .. versionchanged:: 2.0 a list of parameter dictionaries is accepted
            for executemany executions.

        :param execution_options: optional dictionary of execution options
         will be merged into the existing
         :attr:`.ORMExecuteState.execution_options` of this
         :class:`.ORMExecuteState`.

        :param bind_arguments: optional dictionary of bind_arguments
         which will be merged amongst the current
         :attr:`.ORMExecuteState.bind_arguments`
         of this :class:`.ORMExecuteState`.

        :return: a :class:`_engine.Result` object with ORM-level results.

        .. seealso::

            :ref:`do_orm_execute_re_executing` - background and examples on the
            appropriate usage of :meth:`_orm.ORMExecuteState.invoke_statement`.


        """

        if statement is None:
            statement = self.statement

        _bind_arguments = dict(self.bind_arguments)
        if bind_arguments:
            _bind_arguments.update(bind_arguments)
        _bind_arguments["_sa_skip_events"] = True

        _params: Optional[_CoreAnyExecuteParams]
        if params:
            if self.is_executemany:
                _params = []
                exec_many_parameters = cast(
                    "List[Dict[str, Any]]", self.parameters
                )
                for _existing_params, _new_params in itertools.zip_longest(
                    exec_many_parameters,
                    cast("List[Dict[str, Any]]", params),
                ):
                    if _existing_params is None or _new_params is None:
                        raise sa_exc.InvalidRequestError(
                            f"Can't apply executemany parameters to "
                            f"statement; number of parameter sets passed to "
                            f"Session.execute() ({len(exec_many_parameters)}) "
                            f"does not match number of parameter sets given "
                            f"to ORMExecuteState.invoke_statement() "
                            f"({len(params)})"
                        )
                    _existing_params = dict(_existing_params)
                    _existing_params.update(_new_params)
                    _params.append(_existing_params)
            else:
                _params = dict(cast("Dict[str, Any]", self.parameters))
                _params.update(cast("Dict[str, Any]", params))
        else:
            _params = self.parameters

        _execution_options = self.local_execution_options
        if execution_options:
            _execution_options = _execution_options.union(execution_options)

        return self.session._execute_internal(
            statement,
            _params,
            execution_options=_execution_options,
            bind_arguments=_bind_arguments,
            _parent_execute_state=self,
        )

    @property
    def bind_mapper(self) -> Optional[Mapper[Any]]:
        """Return the :class:`_orm.Mapper` that is the primary "bind" mapper.

        For an :class:`_orm.ORMExecuteState` object invoking an ORM
        statement, that is, the :attr:`_orm.ORMExecuteState.is_orm_statement`
        attribute is ``True``, this attribute will return the
        :class:`_orm.Mapper` that is considered to be the "primary" mapper
        of the statement.   The term "bind mapper" refers to the fact that
        a :class:`_orm.Session` object may be "bound" to multiple
        :class:`_engine.Engine` objects keyed to mapped classes, and the
        "bind mapper" determines which of those :class:`_engine.Engine` objects
        would be selected.

        For a statement that is invoked against a single mapped class,
        :attr:`_orm.ORMExecuteState.bind_mapper` is intended to be a reliable
        way of getting this mapper.

        .. versionadded:: 1.4.0b2

        .. seealso::

            :attr:`_orm.ORMExecuteState.all_mappers`


        """
        mp: Optional[Mapper[Any]] = self.bind_arguments.get("mapper", None)
        return mp

    @property
    def all_mappers(self) -> Sequence[Mapper[Any]]:
        """Return a sequence of all :class:`_orm.Mapper` objects that are
        involved at the top level of this statement.

        By "top level" we mean those :class:`_orm.Mapper` objects that would
        be represented in the result set rows for a :func:`_sql.select`
        query, or for a :func:`_dml.update` or :func:`_dml.delete` query,
        the mapper that is the main subject of the UPDATE or DELETE.

        .. versionadded:: 1.4.0b2

        .. seealso::

            :attr:`_orm.ORMExecuteState.bind_mapper`



        """
        if not self.is_orm_statement:
            return []
        elif isinstance(self.statement, (Select, FromStatement)):
            result = []
            seen = set()
            for d in self.statement.column_descriptions:
                ent = d["entity"]
                if ent:
                    insp = inspect(ent, raiseerr=False)
                    if insp and insp.mapper and insp.mapper not in seen:
                        seen.add(insp.mapper)
                        result.append(insp.mapper)
            return result
        elif self.statement.is_dml and self.bind_mapper:
            return [self.bind_mapper]
        else:
            return []

    @property
    def is_orm_statement(self) -> bool:
        """return True if the operation is an ORM statement.

        This indicates that the select(), insert(), update(), or delete()
        being invoked contains ORM entities as subjects.   For a statement
        that does not have ORM entities and instead refers only to
        :class:`.Table` metadata, it is invoked as a Core SQL statement
        and no ORM-level automation takes place.

        """
        return self._compile_state_cls is not None

    @property
    def is_executemany(self) -> bool:
        """return True if the parameters are a multi-element list of
        dictionaries with more than one dictionary.

        .. versionadded:: 2.0

        """
        return isinstance(self.parameters, list)

    @property
    def is_select(self) -> bool:
        """return True if this is a SELECT operation.

        .. versionchanged:: 2.0.30 - the attribute is also True for a
           :meth:`_sql.Select.from_statement` construct that is itself against
           a :class:`_sql.Select` construct, such as
           ``select(Entity).from_statement(select(..))``

        """
        return self.statement.is_select

    @property
    def is_from_statement(self) -> bool:
        """return True if this operation is a
        :meth:`_sql.Select.from_statement` operation.

        This is independent from :attr:`_orm.ORMExecuteState.is_select`, as a
        ``select().from_statement()`` construct can be used with
        INSERT/UPDATE/DELETE RETURNING types of statements as well.
        :attr:`_orm.ORMExecuteState.is_select` will only be set if the
        :meth:`_sql.Select.from_statement` is itself against a
        :class:`_sql.Select` construct.

        .. versionadded:: 2.0.30

        """
        return self.statement.is_from_statement

    @property
    def is_insert(self) -> bool:
        """return True if this is an INSERT operation.

        .. versionchanged:: 2.0.30 - the attribute is also True for a
           :meth:`_sql.Select.from_statement` construct that is itself against
           a :class:`_sql.Insert` construct, such as
           ``select(Entity).from_statement(insert(..))``

        """
        return self.statement.is_dml and self.statement.is_insert

    @property
    def is_update(self) -> bool:
        """return True if this is an UPDATE operation.

        .. versionchanged:: 2.0.30 - the attribute is also True for a
           :meth:`_sql.Select.from_statement` construct that is itself against
           a :class:`_sql.Update` construct, such as
           ``select(Entity).from_statement(update(..))``

        """
        return self.statement.is_dml and self.statement.is_update

    @property
    def is_delete(self) -> bool:
        """return True if this is a DELETE operation.

        .. versionchanged:: 2.0.30 - the attribute is also True for a
           :meth:`_sql.Select.from_statement` construct that is itself against
           a :class:`_sql.Delete` construct, such as
           ``select(Entity).from_statement(delete(..))``

        """
        return self.statement.is_dml and self.statement.is_delete

    @property
    def _is_crud(self) -> bool:
        return isinstance(self.statement, (dml.Update, dml.Delete))

    def update_execution_options(self, **opts: Any) -> None:
        """Update the local execution options with new values."""
        self.local_execution_options = self.local_execution_options.union(opts)

    def _orm_compile_options(
        self,
    ) -> Optional[
        Union[
            context.ORMCompileState.default_compile_options,
            Type[context.ORMCompileState.default_compile_options],
        ]
    ]:
        if not self.is_select:
            return None
        try:
            opts = self.statement._compile_options
        except AttributeError:
            return None

        if opts is not None and opts.isinstance(
            context.ORMCompileState.default_compile_options
        ):
            return opts  # type: ignore
        else:
            return None

    @property
    def lazy_loaded_from(self) -> Optional[InstanceState[Any]]:
        """An :class:`.InstanceState` that is using this statement execution
        for a lazy load operation.

        The primary rationale for this attribute is to support the horizontal
        sharding extension, where it is available within specific query
        execution time hooks created by this extension.   To that end, the
        attribute is only intended to be meaningful at **query execution
        time**, and importantly not any time prior to that, including query
        compilation time.

        """
        return self.load_options._lazy_loaded_from

    @property
    def loader_strategy_path(self) -> Optional[PathRegistry]:
        """Return the :class:`.PathRegistry` for the current load path.

        This object represents the "path" in a query along relationships
        when a particular object or collection is being loaded.

        """
        opts = self._orm_compile_options()
        if opts is not None:
            return opts._current_path
        else:
            return None

    @property
    def is_column_load(self) -> bool:
        """Return True if the operation is refreshing column-oriented
        attributes on an existing ORM object.

        This occurs during operations such as :meth:`_orm.Session.refresh`,
        as well as when an attribute deferred by :func:`_orm.defer` is
        being loaded, or an attribute that was expired either directly
        by :meth:`_orm.Session.expire` or via a commit operation is being
        loaded.

        Handlers will very likely not want to add any options to queries
        when such an operation is occurring as the query should be a straight
        primary key fetch which should not have any additional WHERE criteria,
        and loader options travelling with the instance
        will have already been added to the query.

        .. versionadded:: 1.4.0b2

        .. seealso::

            :attr:`_orm.ORMExecuteState.is_relationship_load`

        """
        opts = self._orm_compile_options()
        return opts is not None and opts._for_refresh_state

    @property
    def is_relationship_load(self) -> bool:
        """Return True if this load is loading objects on behalf of a
        relationship.

        This means, the loader in effect is either a LazyLoader,
        SelectInLoader, SubqueryLoader, or similar, and the entire
        SELECT statement being emitted is on behalf of a relationship
        load.

        Handlers will very likely not want to add any options to queries
        when such an operation is occurring, as loader options are already
        capable of being propagated to relationship loaders and should
        be already present.

        .. seealso::

            :attr:`_orm.ORMExecuteState.is_column_load`

        """
        opts = self._orm_compile_options()
        if opts is None:
            return False
        path = self.loader_strategy_path
        return path is not None and not path.is_root

    @property
    def load_options(
        self,
    ) -> Union[
        context.QueryContext.default_load_options,
        Type[context.QueryContext.default_load_options],
    ]:
        """Return the load_options that will be used for this execution."""

        if not self.is_select:
            raise sa_exc.InvalidRequestError(
                "This ORM execution is not against a SELECT statement "
                "so there are no load options."
            )

        lo: Union[
            context.QueryContext.default_load_options,
            Type[context.QueryContext.default_load_options],
        ] = self.execution_options.get(
            "_sa_orm_load_options", context.QueryContext.default_load_options
        )
        return lo

    @property
    def update_delete_options(
        self,
    ) -> Union[
        bulk_persistence.BulkUDCompileState.default_update_options,
        Type[bulk_persistence.BulkUDCompileState.default_update_options],
    ]:
        """Return the update_delete_options that will be used for this
        execution."""

        if not self._is_crud:
            raise sa_exc.InvalidRequestError(
                "This ORM execution is not against an UPDATE or DELETE "
                "statement so there are no update options."
            )
        uo: Union[
            bulk_persistence.BulkUDCompileState.default_update_options,
            Type[bulk_persistence.BulkUDCompileState.default_update_options],
        ] = self.execution_options.get(
            "_sa_orm_update_options",
            bulk_persistence.BulkUDCompileState.default_update_options,
        )
        return uo

    @property
    def _non_compile_orm_options(self) -> Sequence[ORMOption]:
        return [
            opt
            for opt in self.statement._with_options
            if is_orm_option(opt) and not opt._is_compile_state
        ]

    @property
    def user_defined_options(self) -> Sequence[UserDefinedOption]:
        """The sequence of :class:`.UserDefinedOptions` that have been
        associated with the statement being invoked.

        """
        return [
            opt
            for opt in self.statement._with_options
            if is_user_defined_option(opt)
        ]


class SessionTransactionOrigin(Enum):
    """indicates the origin of a :class:`.SessionTransaction`.

    This enumeration is present on the
    :attr:`.SessionTransaction.origin` attribute of any
    :class:`.SessionTransaction` object.

    .. versionadded:: 2.0

    """

    AUTOBEGIN = 0
    """transaction were started by autobegin"""

    BEGIN = 1
    """transaction were started by calling :meth:`_orm.Session.begin`"""

    BEGIN_NESTED = 2
    """tranaction were started by :meth:`_orm.Session.begin_nested`"""

    SUBTRANSACTION = 3
    """transaction is an internal "subtransaction" """


class SessionTransaction(_StateChange, TransactionalContext):
    """A :class:`.Session`-level transaction.

    :class:`.SessionTransaction` is produced from the
    :meth:`_orm.Session.begin`
    and :meth:`_orm.Session.begin_nested` methods.   It's largely an internal
    object that in modern use provides a context manager for session
    transactions.

    Documentation on interacting with :class:`_orm.SessionTransaction` is
    at: :ref:`unitofwork_transaction`.


    .. versionchanged:: 1.4  The scoping and API methods to work with the
       :class:`_orm.SessionTransaction` object directly have been simplified.

    .. seealso::

        :ref:`unitofwork_transaction`

        :meth:`.Session.begin`

        :meth:`.Session.begin_nested`

        :meth:`.Session.rollback`

        :meth:`.Session.commit`

        :meth:`.Session.in_transaction`

        :meth:`.Session.in_nested_transaction`

        :meth:`.Session.get_transaction`

        :meth:`.Session.get_nested_transaction`


    """

    _rollback_exception: Optional[BaseException] = None

    _connections: Dict[
        Union[Engine, Connection], Tuple[Connection, Transaction, bool, bool]
    ]
    session: Session
    _parent: Optional[SessionTransaction]

    _state: SessionTransactionState

    _new: weakref.WeakKeyDictionary[InstanceState[Any], object]
    _deleted: weakref.WeakKeyDictionary[InstanceState[Any], object]
    _dirty: weakref.WeakKeyDictionary[InstanceState[Any], object]
    _key_switches: weakref.WeakKeyDictionary[
        InstanceState[Any], Tuple[Any, Any]
    ]

    origin: SessionTransactionOrigin
    """Origin of this :class:`_orm.SessionTransaction`.

    Refers to a :class:`.SessionTransactionOrigin` instance which is an
    enumeration indicating the source event that led to constructing
    this :class:`_orm.SessionTransaction`.

    .. versionadded:: 2.0

    """

    nested: bool = False
    """Indicates if this is a nested, or SAVEPOINT, transaction.

    When :attr:`.SessionTransaction.nested` is True, it is expected
    that :attr:`.SessionTransaction.parent` will be present as well,
    linking to the enclosing :class:`.SessionTransaction`.

    .. seealso::

        :attr:`.SessionTransaction.origin`

    """

    def __init__(
        self,
        session: Session,
        origin: SessionTransactionOrigin,
        parent: Optional[SessionTransaction] = None,
    ):
        TransactionalContext._trans_ctx_check(session)

        self.session = session
        self._connections = {}
        self._parent = parent
        self.nested = nested = origin is SessionTransactionOrigin.BEGIN_NESTED
        self.origin = origin

        if session._close_state is _SessionCloseState.CLOSED:
            raise sa_exc.InvalidRequestError(
                "This Session has been permanently closed and is unable "
                "to handle any more transaction requests."
            )

        if nested:
            if not parent:
                raise sa_exc.InvalidRequestError(
                    "Can't start a SAVEPOINT transaction when no existing "
                    "transaction is in progress"
                )

            self._previous_nested_transaction = session._nested_transaction
        elif origin is SessionTransactionOrigin.SUBTRANSACTION:
            assert parent is not None
        else:
            assert parent is None

        self._state = SessionTransactionState.ACTIVE

        self._take_snapshot()

        # make sure transaction is assigned before we call the
        # dispatch
        self.session._transaction = self

        self.session.dispatch.after_transaction_create(self.session, self)

    def _raise_for_prerequisite_state(
        self, operation_name: str, state: _StateChangeState
    ) -> NoReturn:
        if state is SessionTransactionState.DEACTIVE:
            if self._rollback_exception:
                raise sa_exc.PendingRollbackError(
                    "This Session's transaction has been rolled back "
                    "due to a previous exception during flush."
                    " To begin a new transaction with this Session, "
                    "first issue Session.rollback()."
                    f" Original exception was: {self._rollback_exception}",
                    code="7s2a",
                )
            else:
                raise sa_exc.InvalidRequestError(
                    "This session is in 'inactive' state, due to the "
                    "SQL transaction being rolled back; no further SQL "
                    "can be emitted within this transaction."
                )
        elif state is SessionTransactionState.CLOSED:
            raise sa_exc.ResourceClosedError("This transaction is closed")
        elif state is SessionTransactionState.PROVISIONING_CONNECTION:
            raise sa_exc.InvalidRequestError(
                "This session is provisioning a new connection; concurrent "
                "operations are not permitted",
                code="isce",
            )
        else:
            raise sa_exc.InvalidRequestError(
                f"This session is in '{state.name.lower()}' state; no "
                "further SQL can be emitted within this transaction."
            )

    @property
    def parent(self) -> Optional[SessionTransaction]:
        """The parent :class:`.SessionTransaction` of this
        :class:`.SessionTransaction`.

        If this attribute is ``None``, indicates this
        :class:`.SessionTransaction` is at the top of the stack, and
        corresponds to a real "COMMIT"/"ROLLBACK"
        block.  If non-``None``, then this is either a "subtransaction"
        (an internal marker object used by the flush process) or a
        "nested" / SAVEPOINT transaction.  If the
        :attr:`.SessionTransaction.nested` attribute is ``True``, then
        this is a SAVEPOINT, and if ``False``, indicates this a subtransaction.

        """
        return self._parent

    @property
    def is_active(self) -> bool:
        return (
            self.session is not None
            and self._state is SessionTransactionState.ACTIVE
        )

    @property
    def _is_transaction_boundary(self) -> bool:
        return self.nested or not self._parent

    @_StateChange.declare_states(
        (SessionTransactionState.ACTIVE,), _StateChangeStates.NO_CHANGE
    )
    def connection(
        self,
        bindkey: Optional[Mapper[Any]],
        execution_options: Optional[_ExecuteOptions] = None,
        **kwargs: Any,
    ) -> Connection:
        bind = self.session.get_bind(bindkey, **kwargs)
        return self._connection_for_bind(bind, execution_options)

    @_StateChange.declare_states(
        (SessionTransactionState.ACTIVE,), _StateChangeStates.NO_CHANGE
    )
    def _begin(self, nested: bool = False) -> SessionTransaction:
        return SessionTransaction(
            self.session,
            (
                SessionTransactionOrigin.BEGIN_NESTED
                if nested
                else SessionTransactionOrigin.SUBTRANSACTION
            ),
            self,
        )

    def _iterate_self_and_parents(
        self, upto: Optional[SessionTransaction] = None
    ) -> Iterable[SessionTransaction]:
        current = self
        result: Tuple[SessionTransaction, ...] = ()
        while current:
            result += (current,)
            if current._parent is upto:
                break
            elif current._parent is None:
                raise sa_exc.InvalidRequestError(
                    "Transaction %s is not on the active transaction list"
                    % (upto)
                )
            else:
                current = current._parent

        return result

    def _take_snapshot(self) -> None:
        if not self._is_transaction_boundary:
            parent = self._parent
            assert parent is not None
            self._new = parent._new
            self._deleted = parent._deleted
            self._dirty = parent._dirty
            self._key_switches = parent._key_switches
            return

        is_begin = self.origin in (
            SessionTransactionOrigin.BEGIN,
            SessionTransactionOrigin.AUTOBEGIN,
        )
        if not is_begin and not self.session._flushing:
            self.session.flush()

        self._new = weakref.WeakKeyDictionary()
        self._deleted = weakref.WeakKeyDictionary()
        self._dirty = weakref.WeakKeyDictionary()
        self._key_switches = weakref.WeakKeyDictionary()

    def _restore_snapshot(self, dirty_only: bool = False) -> None:
        """Restore the restoration state taken before a transaction began.

        Corresponds to a rollback.

        """
        assert self._is_transaction_boundary

        to_expunge = set(self._new).union(self.session._new)
        self.session._expunge_states(to_expunge, to_transient=True)

        for s, (oldkey, newkey) in self._key_switches.items():
            # we probably can do this conditionally based on
            # if we expunged or not, but safe_discard does that anyway
            self.session.identity_map.safe_discard(s)

            # restore the old key
            s.key = oldkey

            # now restore the object, but only if we didn't expunge
            if s not in to_expunge:
                self.session.identity_map.replace(s)

        for s in set(self._deleted).union(self.session._deleted):
            self.session._update_impl(s, revert_deletion=True)

        assert not self.session._deleted

        for s in self.session.identity_map.all_states():
            if not dirty_only or s.modified or s in self._dirty:
                s._expire(s.dict, self.session.identity_map._modified)

    def _remove_snapshot(self) -> None:
        """Remove the restoration state taken before a transaction began.

        Corresponds to a commit.

        """
        assert self._is_transaction_boundary

        if not self.nested and self.session.expire_on_commit:
            for s in self.session.identity_map.all_states():
                s._expire(s.dict, self.session.identity_map._modified)

            statelib.InstanceState._detach_states(
                list(self._deleted), self.session
            )
            self._deleted.clear()
        elif self.nested:
            parent = self._parent
            assert parent is not None
            parent._new.update(self._new)
            parent._dirty.update(self._dirty)
            parent._deleted.update(self._deleted)
            parent._key_switches.update(self._key_switches)

    @_StateChange.declare_states(
        (SessionTransactionState.ACTIVE,), _StateChangeStates.NO_CHANGE
    )
    def _connection_for_bind(
        self,
        bind: _SessionBind,
        execution_options: Optional[CoreExecuteOptionsParameter],
    ) -> Connection:
        if bind in self._connections:
            if execution_options:
                util.warn(
                    "Connection is already established for the "
                    "given bind; execution_options ignored"
                )
            return self._connections[bind][0]

        self._state = SessionTransactionState.PROVISIONING_CONNECTION

        local_connect = False
        should_commit = True

        try:
            if self._parent:
                conn = self._parent._connection_for_bind(
                    bind, execution_options
                )
                if not self.nested:
                    return conn
            else:
                if isinstance(bind, engine.Connection):
                    conn = bind
                    if conn.engine in self._connections:
                        raise sa_exc.InvalidRequestError(
                            "Session already has a Connection associated "
                            "for the given Connection's Engine"
                        )
                else:
                    conn = bind.connect()
                    local_connect = True

            try:
                if execution_options:
                    conn = conn.execution_options(**execution_options)

                transaction: Transaction
                if self.session.twophase and self._parent is None:
                    # TODO: shouldn't we only be here if not
                    # conn.in_transaction() ?
                    # if twophase is set and conn.in_transaction(), validate
                    # that it is in fact twophase.
                    transaction = conn.begin_twophase()
                elif self.nested:
                    transaction = conn.begin_nested()
                elif conn.in_transaction():
                    join_transaction_mode = self.session.join_transaction_mode

                    if join_transaction_mode == "conditional_savepoint":
                        if conn.in_nested_transaction():
                            join_transaction_mode = "create_savepoint"
                        else:
                            join_transaction_mode = "rollback_only"

                        if local_connect:
                            util.warn(
                                "The engine provided as bind produced a "
                                "connection that is already in a transaction. "
                                "This is usually caused by a core event, "
                                "such as 'engine_connect', that has left a "
                                "transaction open. The effective join "
                                "transaction mode used by this session is "
                                f"{join_transaction_mode!r}. To silence this "
                                "warning, do not leave transactions open"
                            )
                    if join_transaction_mode in (
                        "control_fully",
                        "rollback_only",
                    ):
                        if conn.in_nested_transaction():
                            transaction = (
                                conn._get_required_nested_transaction()
                            )
                        else:
                            transaction = conn._get_required_transaction()
                        if join_transaction_mode == "rollback_only":
                            should_commit = False
                    elif join_transaction_mode == "create_savepoint":
                        transaction = conn.begin_nested()
                    else:
                        assert False, join_transaction_mode
                else:
                    transaction = conn.begin()
            except:
                # connection will not not be associated with this Session;
                # close it immediately so that it isn't closed under GC
                if local_connect:
                    conn.close()
                raise
            else:
                bind_is_connection = isinstance(bind, engine.Connection)

                self._connections[conn] = self._connections[conn.engine] = (
                    conn,
                    transaction,
                    should_commit,
                    not bind_is_connection,
                )
                self.session.dispatch.after_begin(self.session, self, conn)
                return conn
        finally:
            self._state = SessionTransactionState.ACTIVE

    def prepare(self) -> None:
        if self._parent is not None or not self.session.twophase:
            raise sa_exc.InvalidRequestError(
                "'twophase' mode not enabled, or not root transaction; "
                "can't prepare."
            )
        self._prepare_impl()

    @_StateChange.declare_states(
        (SessionTransactionState.ACTIVE,), SessionTransactionState.PREPARED
    )
    def _prepare_impl(self) -> None:
        if self._parent is None or self.nested:
            self.session.dispatch.before_commit(self.session)

        stx = self.session._transaction
        assert stx is not None
        if stx is not self:
            for subtransaction in stx._iterate_self_and_parents(upto=self):
                subtransaction.commit()

        if not self.session._flushing:
            for _flush_guard in range(100):
                if self.session._is_clean():
                    break
                self.session.flush()
            else:
                raise exc.FlushError(
                    "Over 100 subsequent flushes have occurred within "
                    "session.commit() - is an after_flush() hook "
                    "creating new objects?"
                )

        if self._parent is None and self.session.twophase:
            try:
                for t in set(self._connections.values()):
                    cast("TwoPhaseTransaction", t[1]).prepare()
            except:
                with util.safe_reraise():
                    self.rollback()

        self._state = SessionTransactionState.PREPARED

    @_StateChange.declare_states(
        (SessionTransactionState.ACTIVE, SessionTransactionState.PREPARED),
        SessionTransactionState.CLOSED,
    )
    def commit(self, _to_root: bool = False) -> None:
        if self._state is not SessionTransactionState.PREPARED:
            with self._expect_state(SessionTransactionState.PREPARED):
                self._prepare_impl()

        if self._parent is None or self.nested:
            for conn, trans, should_commit, autoclose in set(
                self._connections.values()
            ):
                if should_commit:
                    trans.commit()

            self._state = SessionTransactionState.COMMITTED
            self.session.dispatch.after_commit(self.session)

            self._remove_snapshot()

        with self._expect_state(SessionTransactionState.CLOSED):
            self.close()

        if _to_root and self._parent:
            self._parent.commit(_to_root=True)

    @_StateChange.declare_states(
        (
            SessionTransactionState.ACTIVE,
            SessionTransactionState.DEACTIVE,
            SessionTransactionState.PREPARED,
        ),
        SessionTransactionState.CLOSED,
    )
    def rollback(
        self, _capture_exception: bool = False, _to_root: bool = False
    ) -> None:
        stx = self.session._transaction
        assert stx is not None
        if stx is not self:
            for subtransaction in stx._iterate_self_and_parents(upto=self):
                subtransaction.close()

        boundary = self
        rollback_err = None
        if self._state in (
            SessionTransactionState.ACTIVE,
            SessionTransactionState.PREPARED,
        ):
            for transaction in self._iterate_self_and_parents():
                if transaction._parent is None or transaction.nested:
                    try:
                        for t in set(transaction._connections.values()):
                            t[1].rollback()

                        transaction._state = SessionTransactionState.DEACTIVE
                        self.session.dispatch.after_rollback(self.session)
                    except:
                        rollback_err = sys.exc_info()
                    finally:
                        transaction._state = SessionTransactionState.DEACTIVE
                        transaction._restore_snapshot(
                            dirty_only=transaction.nested
                        )
                    boundary = transaction
                    break
                else:
                    transaction._state = SessionTransactionState.DEACTIVE

        sess = self.session

        if not rollback_err and not sess._is_clean():
            # if items were added, deleted, or mutated
            # here, we need to re-restore the snapshot
            util.warn(
                "Session's state has been changed on "
                "a non-active transaction - this state "
                "will be discarded."
            )
            boundary._restore_snapshot(dirty_only=boundary.nested)

        with self._expect_state(SessionTransactionState.CLOSED):
            self.close()

        if self._parent and _capture_exception:
            self._parent._rollback_exception = sys.exc_info()[1]

        if rollback_err and rollback_err[1]:
            raise rollback_err[1].with_traceback(rollback_err[2])

        sess.dispatch.after_soft_rollback(sess, self)

        if _to_root and self._parent:
            self._parent.rollback(_to_root=True)

    @_StateChange.declare_states(
        _StateChangeStates.ANY, SessionTransactionState.CLOSED
    )
    def close(self, invalidate: bool = False) -> None:
        if self.nested:
            self.session._nested_transaction = (
                self._previous_nested_transaction
            )

        self.session._transaction = self._parent

        for connection, transaction, should_commit, autoclose in set(
            self._connections.values()
        ):
            if invalidate and self._parent is None:
                connection.invalidate()
            if should_commit and transaction.is_active:
                transaction.close()
            if autoclose and self._parent is None:
                connection.close()

        self._state = SessionTransactionState.CLOSED
        sess = self.session

        # TODO: these two None sets were historically after the
        # event hook below, and in 2.0 I changed it this way for some reason,
        # and I remember there being a reason, but not what it was.
        # Why do we need to get rid of them at all?  test_memusage::CycleTest
        # passes with these commented out.
        # self.session = None  # type: ignore
        # self._connections = None  # type: ignore

        sess.dispatch.after_transaction_end(sess, self)

    def _get_subject(self) -> Session:
        return self.session

    def _transaction_is_active(self) -> bool:
        return self._state is SessionTransactionState.ACTIVE

    def _transaction_is_closed(self) -> bool:
        return self._state is SessionTransactionState.CLOSED

    def _rollback_can_be_called(self) -> bool:
        return self._state not in (COMMITTED, CLOSED)


class _SessionCloseState(Enum):
    ACTIVE = 1
    CLOSED = 2
    CLOSE_IS_RESET = 3


class Session(_SessionClassMethods, EventTarget):
    """Manages persistence operations for ORM-mapped objects.

    The :class:`_orm.Session` is **not safe for use in concurrent threads.**.
    See :ref:`session_faq_threadsafe` for background.

    The Session's usage paradigm is described at :doc:`/orm/session`.


    """

    _is_asyncio = False

    dispatch: dispatcher[Session]

    identity_map: IdentityMap
    """A mapping of object identities to objects themselves.

    Iterating through ``Session.identity_map.values()`` provides
    access to the full set of persistent objects (i.e., those
    that have row identity) currently in the session.

    .. seealso::

        :func:`.identity_key` - helper function to produce the keys used
        in this dictionary.

    """

    _new: Dict[InstanceState[Any], Any]
    _deleted: Dict[InstanceState[Any], Any]
    bind: Optional[Union[Engine, Connection]]
    __binds: Dict[_SessionBindKey, _SessionBind]
    _flushing: bool
    _warn_on_events: bool
    _transaction: Optional[SessionTransaction]
    _nested_transaction: Optional[SessionTransaction]
    hash_key: int
    autoflush: bool
    expire_on_commit: bool
    enable_baked_queries: bool
    twophase: bool
    join_transaction_mode: JoinTransactionMode
    _query_cls: Type[Query[Any]]
    _close_state: _SessionCloseState

    def __init__(
        self,
        bind: Optional[_SessionBind] = None,
        *,
        autoflush: bool = True,
        future: Literal[True] = True,
        expire_on_commit: bool = True,
        autobegin: bool = True,
        twophase: bool = False,
        binds: Optional[Dict[_SessionBindKey, _SessionBind]] = None,
        enable_baked_queries: bool = True,
        info: Optional[_InfoType] = None,
        query_cls: Optional[Type[Query[Any]]] = None,
        autocommit: Literal[False] = False,
        join_transaction_mode: JoinTransactionMode = "conditional_savepoint",
        close_resets_only: Union[bool, _NoArg] = _NoArg.NO_ARG,
    ):
        r"""Construct a new :class:`_orm.Session`.

        See also the :class:`.sessionmaker` function which is used to
        generate a :class:`.Session`-producing callable with a given
        set of arguments.

        :param autoflush: When ``True``, all query operations will issue a
           :meth:`~.Session.flush` call to this ``Session`` before proceeding.
           This is a convenience feature so that :meth:`~.Session.flush` need
           not be called repeatedly in order for database queries to retrieve
           results.

           .. seealso::

               :ref:`session_flushing` - additional background on autoflush

        :param autobegin: Automatically start transactions (i.e. equivalent to
           invoking :meth:`_orm.Session.begin`) when database access is
           requested by an operation.   Defaults to ``True``.    Set to
           ``False`` to prevent a :class:`_orm.Session` from implicitly
           beginning transactions after construction, as well as after any of
           the :meth:`_orm.Session.rollback`, :meth:`_orm.Session.commit`,
           or :meth:`_orm.Session.close` methods are called.

           .. versionadded:: 2.0

           .. seealso::

                :ref:`session_autobegin_disable`

        :param bind: An optional :class:`_engine.Engine` or
           :class:`_engine.Connection` to
           which this ``Session`` should be bound. When specified, all SQL
           operations performed by this session will execute via this
           connectable.

        :param binds: A dictionary which may specify any number of
           :class:`_engine.Engine` or :class:`_engine.Connection`
           objects as the source of
           connectivity for SQL operations on a per-entity basis.   The keys
           of the dictionary consist of any series of mapped classes,
           arbitrary Python classes that are bases for mapped classes,
           :class:`_schema.Table` objects and :class:`_orm.Mapper` objects.
           The
           values of the dictionary are then instances of
           :class:`_engine.Engine`
           or less commonly :class:`_engine.Connection` objects.
           Operations which
           proceed relative to a particular mapped class will consult this
           dictionary for the closest matching entity in order to determine
           which :class:`_engine.Engine` should be used for a particular SQL
           operation.    The complete heuristics for resolution are
           described at :meth:`.Session.get_bind`.  Usage looks like::

            Session = sessionmaker(
                binds={
                    SomeMappedClass: create_engine("postgresql+psycopg2://engine1"),
                    SomeDeclarativeBase: create_engine(
                        "postgresql+psycopg2://engine2"
                    ),
                    some_mapper: create_engine("postgresql+psycopg2://engine3"),
                    some_table: create_engine("postgresql+psycopg2://engine4"),
                }
            )

           .. seealso::

                :ref:`session_partitioning`

                :meth:`.Session.bind_mapper`

                :meth:`.Session.bind_table`

                :meth:`.Session.get_bind`


        :param \class_: Specify an alternate class other than
           ``sqlalchemy.orm.session.Session`` which should be used by the
           returned class. This is the only argument that is local to the
           :class:`.sessionmaker` function, and is not sent directly to the
           constructor for ``Session``.

        :param enable_baked_queries: legacy; defaults to ``True``.
           A parameter consumed
           by the :mod:`sqlalchemy.ext.baked` extension to determine if
           "baked queries" should be cached, as is the normal operation
           of this extension.  When set to ``False``, caching as used by
           this particular extension is disabled.

           .. versionchanged:: 1.4 The ``sqlalchemy.ext.baked`` extension is
              legacy and is not used by any of SQLAlchemy's internals. This
              flag therefore only affects applications that are making explicit
              use of this extension within their own code.

        :param expire_on_commit:  Defaults to ``True``. When ``True``, all
           instances will be fully expired after each :meth:`~.commit`,
           so that all attribute/object access subsequent to a completed
           transaction will load from the most recent database state.

            .. seealso::

                :ref:`session_committing`

        :param future: Deprecated; this flag is always True.

          .. seealso::

            :ref:`migration_20_toplevel`

        :param info: optional dictionary of arbitrary data to be associated
           with this :class:`.Session`.  Is available via the
           :attr:`.Session.info` attribute.  Note the dictionary is copied at
           construction time so that modifications to the per-
           :class:`.Session` dictionary will be local to that
           :class:`.Session`.

        :param query_cls:  Class which should be used to create new Query
          objects, as returned by the :meth:`~.Session.query` method.
          Defaults to :class:`_query.Query`.

        :param twophase:  When ``True``, all transactions will be started as
            a "two phase" transaction, i.e. using the "two phase" semantics
            of the database in use along with an XID.  During a
            :meth:`~.commit`, after :meth:`~.flush` has been issued for all
            attached databases, the :meth:`~.TwoPhaseTransaction.prepare`
            method on each database's :class:`.TwoPhaseTransaction` will be
            called. This allows each database to roll back the entire
            transaction, before each transaction is committed.

        :param autocommit: the "autocommit" keyword is present for backwards
            compatibility but must remain at its default value of ``False``.

        :param join_transaction_mode: Describes the transactional behavior to
          take when a given bind is a :class:`_engine.Connection` that
          has already begun a transaction outside the scope of this
          :class:`_orm.Session`; in other words the
          :meth:`_engine.Connection.in_transaction()` method returns True.

          The following behaviors only take effect when the :class:`_orm.Session`
          **actually makes use of the connection given**; that is, a method
          such as :meth:`_orm.Session.execute`, :meth:`_orm.Session.connection`,
          etc. are actually invoked:

          * ``"conditional_savepoint"`` - this is the default.  if the given
            :class:`_engine.Connection` is begun within a transaction but
            does not have a SAVEPOINT, then ``"rollback_only"`` is used.
            If the :class:`_engine.Connection` is additionally within
            a SAVEPOINT, in other words
            :meth:`_engine.Connection.in_nested_transaction()` method returns
            True, then ``"create_savepoint"`` is used.

            ``"conditional_savepoint"`` behavior attempts to make use of
            savepoints in order to keep the state of the existing transaction
            unchanged, but only if there is already a savepoint in progress;
            otherwise, it is not assumed that the backend in use has adequate
            support for SAVEPOINT, as availability of this feature varies.
            ``"conditional_savepoint"`` also seeks to establish approximate
            backwards compatibility with previous :class:`_orm.Session`
            behavior, for applications that are not setting a specific mode. It
            is recommended that one of the explicit settings be used.

          * ``"create_savepoint"`` - the :class:`_orm.Session` will use
            :meth:`_engine.Connection.begin_nested()` in all cases to create
            its own transaction.  This transaction by its nature rides
            "on top" of any existing transaction that's opened on the given
            :class:`_engine.Connection`; if the underlying database and
            the driver in use has full, non-broken support for SAVEPOINT, the
            external transaction will remain unaffected throughout the
            lifespan of the :class:`_orm.Session`.

            The ``"create_savepoint"`` mode is the most useful for integrating
            a :class:`_orm.Session` into a test suite where an externally
            initiated transaction should remain unaffected; however, it relies
            on proper SAVEPOINT support from the underlying driver and
            database.

            .. tip:: When using SQLite, the SQLite driver included through
               Python 3.11 does not handle SAVEPOINTs correctly in all cases
               without workarounds. See the sections
               :ref:`pysqlite_serializable` and :ref:`aiosqlite_serializable`
               for details on current workarounds.

          * ``"control_fully"`` - the :class:`_orm.Session` will take
            control of the given transaction as its own;
            :meth:`_orm.Session.commit` will call ``.commit()`` on the
            transaction, :meth:`_orm.Session.rollback` will call
            ``.rollback()`` on the transaction, :meth:`_orm.Session.close` will
            call ``.rollback`` on the transaction.

            .. tip:: This mode of use is equivalent to how SQLAlchemy 1.4 would
               handle a :class:`_engine.Connection` given with an existing
               SAVEPOINT (i.e. :meth:`_engine.Connection.begin_nested`); the
               :class:`_orm.Session` would take full control of the existing
               SAVEPOINT.

          * ``"rollback_only"`` - the :class:`_orm.Session` will take control
            of the given transaction for ``.rollback()`` calls only;
            ``.commit()`` calls will not be propagated to the given
            transaction.  ``.close()`` calls will have no effect on the
            given transaction.

            .. tip:: This mode of use is equivalent to how SQLAlchemy 1.4 would
               handle a :class:`_engine.Connection` given with an existing
               regular database transaction (i.e.
               :meth:`_engine.Connection.begin`); the :class:`_orm.Session`
               would propagate :meth:`_orm.Session.rollback` calls to the
               underlying transaction, but not :meth:`_orm.Session.commit` or
               :meth:`_orm.Session.close` calls.

          .. versionadded:: 2.0.0rc1

        :param close_resets_only: Defaults to ``True``. Determines if
          the session should reset itself after calling ``.close()``
          or should pass in a no longer usable state, disabling re-use.

          .. versionadded:: 2.0.22 added flag ``close_resets_only``.
            A future SQLAlchemy version may change the default value of
            this flag to ``False``.

          .. seealso::

            :ref:`session_closing` - Detail on the semantics of
            :meth:`_orm.Session.close` and :meth:`_orm.Session.reset`.

        """  # noqa

        # considering allowing the "autocommit" keyword to still be accepted
        # as long as it's False, so that external test suites, oslo.db etc
        # continue to function as the argument appears to be passed in lots
        # of cases including in our own test suite
        if autocommit:
            raise sa_exc.ArgumentError(
                "autocommit=True is no longer supported"
            )
        self.identity_map = identity.WeakInstanceDict()

        if not future:
            raise sa_exc.ArgumentError(
                "The 'future' parameter passed to "
                "Session() may only be set to True."
            )

        self._new = {}  # InstanceState->object, strong refs object
        self._deleted = {}  # same
        self.bind = bind
        self.__binds = {}
        self._flushing = False
        self._warn_on_events = False
        self._transaction = None
        self._nested_transaction = None
        self.hash_key = _new_sessionid()
        self.autobegin = autobegin
        self.autoflush = autoflush
        self.expire_on_commit = expire_on_commit
        self.enable_baked_queries = enable_baked_queries

        # the idea is that at some point NO_ARG will warn that in the future
        # the default will switch to close_resets_only=False.
        if close_resets_only in (True, _NoArg.NO_ARG):
            self._close_state = _SessionCloseState.CLOSE_IS_RESET
        else:
            self._close_state = _SessionCloseState.ACTIVE
        if (
            join_transaction_mode
            and join_transaction_mode
            not in JoinTransactionMode.__args__  # type: ignore
        ):
            raise sa_exc.ArgumentError(
                f"invalid selection for join_transaction_mode: "
                f'"{join_transaction_mode}"'
            )
        self.join_transaction_mode = join_transaction_mode

        self.twophase = twophase
        self._query_cls = query_cls if query_cls else query.Query
        if info:
            self.info.update(info)

        if binds is not None:
            for key, bind in binds.items():
                self._add_bind(key, bind)

        _sessions[self.hash_key] = self

    # used by sqlalchemy.engine.util.TransactionalContext
    _trans_context_manager: Optional[TransactionalContext] = None

    connection_callable: Optional[_ConnectionCallableProto] = None

    def __enter__(self: _S) -> _S:
        return self

    def __exit__(self, type_: Any, value: Any, traceback: Any) -> None:
        self.close()

    @contextlib.contextmanager
    def _maker_context_manager(self: _S) -> Iterator[_S]:
        with self:
            with self.begin():
                yield self

    def in_transaction(self) -> bool:
        """Return True if this :class:`_orm.Session` has begun a transaction.

        .. versionadded:: 1.4

        .. seealso::

            :attr:`_orm.Session.is_active`


        """
        return self._transaction is not None

    def in_nested_transaction(self) -> bool:
        """Return True if this :class:`_orm.Session` has begun a nested
        transaction, e.g. SAVEPOINT.

        .. versionadded:: 1.4

        """
        return self._nested_transaction is not None

    def get_transaction(self) -> Optional[SessionTransaction]:
        """Return the current root transaction in progress, if any.

        .. versionadded:: 1.4

        """
        trans = self._transaction
        while trans is not None and trans._parent is not None:
            trans = trans._parent
        return trans

    def get_nested_transaction(self) -> Optional[SessionTransaction]:
        """Return the current nested transaction in progress, if any.

        .. versionadded:: 1.4

        """

        return self._nested_transaction

    @util.memoized_property
    def info(self) -> _InfoType:
        """A user-modifiable dictionary.

        The initial value of this dictionary can be populated using the
        ``info`` argument to the :class:`.Session` constructor or
        :class:`.sessionmaker` constructor or factory methods.  The dictionary
        here is always local to this :class:`.Session` and can be modified
        independently of all other :class:`.Session` objects.

        """
        return {}

    def _autobegin_t(self, begin: bool = False) -> SessionTransaction:
        if self._transaction is None:
            if not begin and not self.autobegin:
                raise sa_exc.InvalidRequestError(
                    "Autobegin is disabled on this Session; please call "
                    "session.begin() to start a new transaction"
                )
            trans = SessionTransaction(
                self,
                (
                    SessionTransactionOrigin.BEGIN
                    if begin
                    else SessionTransactionOrigin.AUTOBEGIN
                ),
            )
            assert self._transaction is trans
            return trans

        return self._transaction

    def begin(self, nested: bool = False) -> SessionTransaction:
        """Begin a transaction, or nested transaction,
        on this :class:`.Session`, if one is not already begun.

        The :class:`_orm.Session` object features **autobegin** behavior,
        so that normally it is not necessary to call the
        :meth:`_orm.Session.begin`
        method explicitly. However, it may be used in order to control
        the scope of when the transactional state is begun.

        When used to begin the outermost transaction, an error is raised
        if this :class:`.Session` is already inside of a transaction.

        :param nested: if True, begins a SAVEPOINT transaction and is
         equivalent to calling :meth:`~.Session.begin_nested`. For
         documentation on SAVEPOINT transactions, please see
         :ref:`session_begin_nested`.

        :return: the :class:`.SessionTransaction` object.  Note that
         :class:`.SessionTransaction`
         acts as a Python context manager, allowing :meth:`.Session.begin`
         to be used in a "with" block.  See :ref:`session_explicit_begin` for
         an example.

        .. seealso::

            :ref:`session_autobegin`

            :ref:`unitofwork_transaction`

            :meth:`.Session.begin_nested`


        """

        trans = self._transaction
        if trans is None:
            trans = self._autobegin_t(begin=True)

            if not nested:
                return trans

        assert trans is not None

        if nested:
            trans = trans._begin(nested=nested)
            assert self._transaction is trans
            self._nested_transaction = trans
        else:
            raise sa_exc.InvalidRequestError(
                "A transaction is already begun on this Session."
            )

        return trans  # needed for __enter__/__exit__ hook

    def begin_nested(self) -> SessionTransaction:
        """Begin a "nested" transaction on this Session, e.g. SAVEPOINT.

        The target database(s) and associated drivers must support SQL
        SAVEPOINT for this method to function correctly.

        For documentation on SAVEPOINT
        transactions, please see :ref:`session_begin_nested`.

        :return: the :class:`.SessionTransaction` object.  Note that
         :class:`.SessionTransaction` acts as a context manager, allowing
         :meth:`.Session.begin_nested` to be used in a "with" block.
         See :ref:`session_begin_nested` for a usage example.

        .. seealso::

            :ref:`session_begin_nested`

            :ref:`pysqlite_serializable` - special workarounds required
            with the SQLite driver in order for SAVEPOINT to work
            correctly. For asyncio use cases, see the section
            :ref:`aiosqlite_serializable`.

        """
        return self.begin(nested=True)

    def rollback(self) -> None:
        """Rollback the current transaction in progress.

        If no transaction is in progress, this method is a pass-through.

        The method always rolls back
        the topmost database transaction, discarding any nested
        transactions that may be in progress.

        .. seealso::

            :ref:`session_rollback`

            :ref:`unitofwork_transaction`

        """
        if self._transaction is None:
            pass
        else:
            self._transaction.rollback(_to_root=True)

    def commit(self) -> None:
        """Flush pending changes and commit the current transaction.

        When the COMMIT operation is complete, all objects are fully
        :term:`expired`, erasing their internal contents, which will be
        automatically re-loaded when the objects are next accessed. In the
        interim, these objects are in an expired state and will not function if
        they are :term:`detached` from the :class:`.Session`. Additionally,
        this re-load operation is not supported when using asyncio-oriented
        APIs. The :paramref:`.Session.expire_on_commit` parameter may be used
        to disable this behavior.

        When there is no transaction in place for the :class:`.Session`,
        indicating that no operations were invoked on this :class:`.Session`
        since the previous call to :meth:`.Session.commit`, the method will
        begin and commit an internal-only "logical" transaction, that does not
        normally affect the database unless pending flush changes were
        detected, but will still invoke event handlers and object expiration
        rules.

        The outermost database transaction is committed unconditionally,
        automatically releasing any SAVEPOINTs in effect.

        .. seealso::

            :ref:`session_committing`

            :ref:`unitofwork_transaction`

            :ref:`asyncio_orm_avoid_lazyloads`

        """
        trans = self._transaction
        if trans is None:
            trans = self._autobegin_t()

        trans.commit(_to_root=True)

    def prepare(self) -> None:
        """Prepare the current transaction in progress for two phase commit.

        If no transaction is in progress, this method raises an
        :exc:`~sqlalchemy.exc.InvalidRequestError`.

        Only root transactions of two phase sessions can be prepared. If the
        current transaction is not such, an
        :exc:`~sqlalchemy.exc.InvalidRequestError` is raised.

        """
        trans = self._transaction
        if trans is None:
            trans = self._autobegin_t()

        trans.prepare()

    def connection(
        self,
        bind_arguments: Optional[_BindArguments] = None,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> Connection:
        r"""Return a :class:`_engine.Connection` object corresponding to this
        :class:`.Session` object's transactional state.

        Either the :class:`_engine.Connection` corresponding to the current
        transaction is returned, or if no transaction is in progress, a new
        one is begun and the :class:`_engine.Connection`
        returned (note that no
        transactional state is established with the DBAPI until the first
        SQL statement is emitted).

        Ambiguity in multi-bind or unbound :class:`.Session` objects can be
        resolved through any of the optional keyword arguments.   This
        ultimately makes usage of the :meth:`.get_bind` method for resolution.

        :param bind_arguments: dictionary of bind arguments.  May include
         "mapper", "bind", "clause", other custom arguments that are passed
         to :meth:`.Session.get_bind`.

        :param execution_options: a dictionary of execution options that will
         be passed to :meth:`_engine.Connection.execution_options`, **when the
         connection is first procured only**.   If the connection is already
         present within the :class:`.Session`, a warning is emitted and
         the arguments are ignored.

         .. seealso::

            :ref:`session_transaction_isolation`

        """

        if bind_arguments:
            bind = bind_arguments.pop("bind", None)

            if bind is None:
                bind = self.get_bind(**bind_arguments)
        else:
            bind = self.get_bind()

        return self._connection_for_bind(
            bind,
            execution_options=execution_options,
        )

    def _connection_for_bind(
        self,
        engine: _SessionBind,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
        **kw: Any,
    ) -> Connection:
        TransactionalContext._trans_ctx_check(self)

        trans = self._transaction
        if trans is None:
            trans = self._autobegin_t()
        return trans._connection_for_bind(engine, execution_options)

    @overload
    def _execute_internal(
        self,
        statement: Executable,
        params: Optional[_CoreSingleExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        _parent_execute_state: Optional[Any] = None,
        _add_event: Optional[Any] = None,
        _scalar_result: Literal[True] = ...,
    ) -> Any: ...

    @overload
    def _execute_internal(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        _parent_execute_state: Optional[Any] = None,
        _add_event: Optional[Any] = None,
        _scalar_result: bool = ...,
    ) -> Result[Any]: ...

    def _execute_internal(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        _parent_execute_state: Optional[Any] = None,
        _add_event: Optional[Any] = None,
        _scalar_result: bool = False,
    ) -> Any:
        statement = coercions.expect(roles.StatementRole, statement)

        if not bind_arguments:
            bind_arguments = {}
        else:
            bind_arguments = dict(bind_arguments)

        if (
            statement._propagate_attrs.get("compile_state_plugin", None)
            == "orm"
        ):
            compile_state_cls = CompileState._get_plugin_class_for_plugin(
                statement, "orm"
            )
            if TYPE_CHECKING:
                assert isinstance(
                    compile_state_cls, context.AbstractORMCompileState
                )
        else:
            compile_state_cls = None
            bind_arguments.setdefault("clause", statement)

        execution_options = util.coerce_to_immutabledict(execution_options)

        if _parent_execute_state:
            events_todo = _parent_execute_state._remaining_events()
        else:
            events_todo = self.dispatch.do_orm_execute
            if _add_event:
                events_todo = list(events_todo) + [_add_event]

        if events_todo:
            if compile_state_cls is not None:
                # for event handlers, do the orm_pre_session_exec
                # pass ahead of the event handlers, so that things like
                # .load_options, .update_delete_options etc. are populated.
                # is_pre_event=True allows the hook to hold off on things
                # it doesn't want to do twice, including autoflush as well
                # as "pre fetch" for DML, etc.
                (
                    statement,
                    execution_options,
                ) = compile_state_cls.orm_pre_session_exec(
                    self,
                    statement,
                    params,
                    execution_options,
                    bind_arguments,
                    True,
                )

            orm_exec_state = ORMExecuteState(
                self,
                statement,
                params,
                execution_options,
                bind_arguments,
                compile_state_cls,
                events_todo,
            )
            for idx, fn in enumerate(events_todo):
                orm_exec_state._starting_event_idx = idx
                fn_result: Optional[Result[Any]] = fn(orm_exec_state)
                if fn_result:
                    if _scalar_result:
                        return fn_result.scalar()
                    else:
                        return fn_result

            statement = orm_exec_state.statement
            execution_options = orm_exec_state.local_execution_options

        if compile_state_cls is not None:
            # now run orm_pre_session_exec() "for real".   if there were
            # event hooks, this will re-run the steps that interpret
            # new execution_options into load_options / update_delete_options,
            # which we assume the event hook might have updated.
            # autoflush will also be invoked in this step if enabled.
            (
                statement,
                execution_options,
            ) = compile_state_cls.orm_pre_session_exec(
                self,
                statement,
                params,
                execution_options,
                bind_arguments,
                False,
            )

        bind = self.get_bind(**bind_arguments)

        conn = self._connection_for_bind(bind)

        if _scalar_result and not compile_state_cls:
            if TYPE_CHECKING:
                params = cast(_CoreSingleExecuteParams, params)
            return conn.scalar(
                statement, params or {}, execution_options=execution_options
            )

        if compile_state_cls:
            result: Result[Any] = compile_state_cls.orm_execute_statement(
                self,
                statement,
                params or {},
                execution_options,
                bind_arguments,
                conn,
            )
        else:
            result = conn.execute(
                statement, params or {}, execution_options=execution_options
            )

        if _scalar_result:
            return result.scalar()
        else:
            return result

    @overload
    def execute(
        self,
        statement: TypedReturnsRows[_T],
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        _parent_execute_state: Optional[Any] = None,
        _add_event: Optional[Any] = None,
    ) -> Result[_T]: ...

    @overload
    def execute(
        self,
        statement: UpdateBase,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        _parent_execute_state: Optional[Any] = None,
        _add_event: Optional[Any] = None,
    ) -> CursorResult[Any]: ...

    @overload
    def execute(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        _parent_execute_state: Optional[Any] = None,
        _add_event: Optional[Any] = None,
    ) -> Result[Any]: ...

    def execute(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        _parent_execute_state: Optional[Any] = None,
        _add_event: Optional[Any] = None,
    ) -> Result[Any]:
        r"""Execute a SQL expression construct.

        Returns a :class:`_engine.Result` object representing
        results of the statement execution.

        E.g.::

            from sqlalchemy import select

            result = session.execute(select(User).where(User.id == 5))

        The API contract of :meth:`_orm.Session.execute` is similar to that
        of :meth:`_engine.Connection.execute`, the :term:`2.0 style` version
        of :class:`_engine.Connection`.

        .. versionchanged:: 1.4 the :meth:`_orm.Session.execute` method is
           now the primary point of ORM statement execution when using
           :term:`2.0 style` ORM usage.

        :param statement:
            An executable statement (i.e. an :class:`.Executable` expression
            such as :func:`_expression.select`).

        :param params:
            Optional dictionary, or list of dictionaries, containing
            bound parameter values.   If a single dictionary, single-row
            execution occurs; if a list of dictionaries, an
            "executemany" will be invoked.  The keys in each dictionary
            must correspond to parameter names present in the statement.

        :param execution_options: optional dictionary of execution options,
         which will be associated with the statement execution.  This
         dictionary can provide a subset of the options that are accepted
         by :meth:`_engine.Connection.execution_options`, and may also
         provide additional options understood only in an ORM context.

         .. seealso::

            :ref:`orm_queryguide_execution_options` - ORM-specific execution
            options

        :param bind_arguments: dictionary of additional arguments to determine
         the bind.  May include "mapper", "bind", or other custom arguments.
         Contents of this dictionary are passed to the
         :meth:`.Session.get_bind` method.

        :return: a :class:`_engine.Result` object.


        """
        return self._execute_internal(
            statement,
            params,
            execution_options=execution_options,
            bind_arguments=bind_arguments,
            _parent_execute_state=_parent_execute_state,
            _add_event=_add_event,
        )

    @overload
    def scalar(
        self,
        statement: TypedReturnsRows[Tuple[_T]],
        params: Optional[_CoreSingleExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> Optional[_T]: ...

    @overload
    def scalar(
        self,
        statement: Executable,
        params: Optional[_CoreSingleExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> Any: ...

    def scalar(
        self,
        statement: Executable,
        params: Optional[_CoreSingleExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> Any:
        """Execute a statement and return a scalar result.

        Usage and parameters are the same as that of
        :meth:`_orm.Session.execute`; the return result is a scalar Python
        value.

        """

        return self._execute_internal(
            statement,
            params,
            execution_options=execution_options,
            bind_arguments=bind_arguments,
            _scalar_result=True,
            **kw,
        )

    @overload
    def scalars(
        self,
        statement: TypedReturnsRows[Tuple[_T]],
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> ScalarResult[_T]: ...

    @overload
    def scalars(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> ScalarResult[Any]: ...

    def scalars(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> ScalarResult[Any]:
        """Execute a statement and return the results as scalars.

        Usage and parameters are the same as that of
        :meth:`_orm.Session.execute`; the return result is a
        :class:`_result.ScalarResult` filtering object which
        will return single elements rather than :class:`_row.Row` objects.

        :return:  a :class:`_result.ScalarResult` object

        .. versionadded:: 1.4.24 Added :meth:`_orm.Session.scalars`

        .. versionadded:: 1.4.26 Added :meth:`_orm.scoped_session.scalars`

        .. seealso::

            :ref:`orm_queryguide_select_orm_entities` - contrasts the behavior
            of :meth:`_orm.Session.execute` to :meth:`_orm.Session.scalars`

        """

        return self._execute_internal(
            statement,
            params=params,
            execution_options=execution_options,
            bind_arguments=bind_arguments,
            _scalar_result=False,  # mypy appreciates this
            **kw,
        ).scalars()

    def close(self) -> None:
        """Close out the transactional resources and ORM objects used by this
        :class:`_orm.Session`.

        This expunges all ORM objects associated with this
        :class:`_orm.Session`, ends any transaction in progress and
        :term:`releases` any :class:`_engine.Connection` objects which this
        :class:`_orm.Session` itself has checked out from associated
        :class:`_engine.Engine` objects. The operation then leaves the
        :class:`_orm.Session` in a state which it may be used again.

        .. tip::

            In the default running mode the :meth:`_orm.Session.close`
            method **does not prevent the Session from being used again**.
            The :class:`_orm.Session` itself does not actually have a
            distinct "closed" state; it merely means
            the :class:`_orm.Session` will release all database connections
            and ORM objects.

            Setting the parameter :paramref:`_orm.Session.close_resets_only`
            to ``False`` will instead make the ``close`` final, meaning that
            any further action on the session will be forbidden.

        .. versionchanged:: 1.4  The :meth:`.Session.close` method does not
           immediately create a new :class:`.SessionTransaction` object;
           instead, the new :class:`.SessionTransaction` is created only if
           the :class:`.Session` is used again for a database operation.

        .. seealso::

            :ref:`session_closing` - detail on the semantics of
            :meth:`_orm.Session.close` and :meth:`_orm.Session.reset`.

            :meth:`_orm.Session.reset` - a similar method that behaves like
            ``close()`` with  the parameter
            :paramref:`_orm.Session.close_resets_only` set to ``True``.

        """
        self._close_impl(invalidate=False)

    def reset(self) -> None:
        """Close out the transactional resources and ORM objects used by this
        :class:`_orm.Session`, resetting the session to its initial state.

        This method provides for same "reset-only" behavior that the
        :meth:`_orm.Session.close` method has provided historically, where the
        state of the :class:`_orm.Session` is reset as though the object were
        brand new, and ready to be used again.
        This method may then be useful for :class:`_orm.Session` objects
        which set :paramref:`_orm.Session.close_resets_only` to ``False``,
        so that "reset only" behavior is still available.

        .. versionadded:: 2.0.22

        .. seealso::

            :ref:`session_closing` - detail on the semantics of
            :meth:`_orm.Session.close` and :meth:`_orm.Session.reset`.

            :meth:`_orm.Session.close` - a similar method will additionally
            prevent re-use of the Session when the parameter
            :paramref:`_orm.Session.close_resets_only` is set to ``False``.
        """
        self._close_impl(invalidate=False, is_reset=True)

    def invalidate(self) -> None:
        """Close this Session, using connection invalidation.

        This is a variant of :meth:`.Session.close` that will additionally
        ensure that the :meth:`_engine.Connection.invalidate`
        method will be called on each :class:`_engine.Connection` object
        that is currently in use for a transaction (typically there is only
        one connection unless the :class:`_orm.Session` is used with
        multiple engines).

        This can be called when the database is known to be in a state where
        the connections are no longer safe to be used.

        Below illustrates a scenario when using `gevent
        <https://www.gevent.org/>`_, which can produce ``Timeout`` exceptions
        that may mean the underlying connection should be discarded::

            import gevent

            try:
                sess = Session()
                sess.add(User())
                sess.commit()
            except gevent.Timeout:
                sess.invalidate()
                raise
            except:
                sess.rollback()
                raise

        The method additionally does everything that :meth:`_orm.Session.close`
        does, including that all ORM objects are expunged.

        """
        self._close_impl(invalidate=True)

    def _close_impl(self, invalidate: bool, is_reset: bool = False) -> None:
        if not is_reset and self._close_state is _SessionCloseState.ACTIVE:
            self._close_state = _SessionCloseState.CLOSED
        self.expunge_all()
        if self._transaction is not None:
            for transaction in self._transaction._iterate_self_and_parents():
                transaction.close(invalidate)

    def expunge_all(self) -> None:
        """Remove all object instances from this ``Session``.

        This is equivalent to calling ``expunge(obj)`` on all objects in this
        ``Session``.

        """

        all_states = self.identity_map.all_states() + list(self._new)
        self.identity_map._kill()
        self.identity_map = identity.WeakInstanceDict()
        self._new = {}
        self._deleted = {}

        statelib.InstanceState._detach_states(all_states, self)

    def _add_bind(self, key: _SessionBindKey, bind: _SessionBind) -> None:
        try:
            insp = inspect(key)
        except sa_exc.NoInspectionAvailable as err:
            if not isinstance(key, type):
                raise sa_exc.ArgumentError(
                    "Not an acceptable bind target: %s" % key
                ) from err
            else:
                self.__binds[key] = bind
        else:
            if TYPE_CHECKING:
                assert isinstance(insp, Inspectable)

            if isinstance(insp, TableClause):
                self.__binds[insp] = bind
            elif insp_is_mapper(insp):
                self.__binds[insp.class_] = bind
                for _selectable in insp._all_tables:
                    self.__binds[_selectable] = bind
            else:
                raise sa_exc.ArgumentError(
                    "Not an acceptable bind target: %s" % key
                )

    def bind_mapper(
        self, mapper: _EntityBindKey[_O], bind: _SessionBind
    ) -> None:
        """Associate a :class:`_orm.Mapper` or arbitrary Python class with a
        "bind", e.g. an :class:`_engine.Engine` or
        :class:`_engine.Connection`.

        The given entity is added to a lookup used by the
        :meth:`.Session.get_bind` method.

        :param mapper: a :class:`_orm.Mapper` object,
         or an instance of a mapped
         class, or any Python class that is the base of a set of mapped
         classes.

        :param bind: an :class:`_engine.Engine` or :class:`_engine.Connection`
                    object.

        .. seealso::

            :ref:`session_partitioning`

            :paramref:`.Session.binds`

            :meth:`.Session.bind_table`


        """
        self._add_bind(mapper, bind)

    def bind_table(self, table: TableClause, bind: _SessionBind) -> None:
        """Associate a :class:`_schema.Table` with a "bind", e.g. an
        :class:`_engine.Engine`
        or :class:`_engine.Connection`.

        The given :class:`_schema.Table` is added to a lookup used by the
        :meth:`.Session.get_bind` method.

        :param table: a :class:`_schema.Table` object,
         which is typically the target
         of an ORM mapping, or is present within a selectable that is
         mapped.

        :param bind: an :class:`_engine.Engine` or :class:`_engine.Connection`
         object.

        .. seealso::

            :ref:`session_partitioning`

            :paramref:`.Session.binds`

            :meth:`.Session.bind_mapper`


        """
        self._add_bind(table, bind)

    def get_bind(
        self,
        mapper: Optional[_EntityBindKey[_O]] = None,
        *,
        clause: Optional[ClauseElement] = None,
        bind: Optional[_SessionBind] = None,
        _sa_skip_events: Optional[bool] = None,
        _sa_skip_for_implicit_returning: bool = False,
        **kw: Any,
    ) -> Union[Engine, Connection]:
        """Return a "bind" to which this :class:`.Session` is bound.

        The "bind" is usually an instance of :class:`_engine.Engine`,
        except in the case where the :class:`.Session` has been
        explicitly bound directly to a :class:`_engine.Connection`.

        For a multiply-bound or unbound :class:`.Session`, the
        ``mapper`` or ``clause`` arguments are used to determine the
        appropriate bind to return.

        Note that the "mapper" argument is usually present
        when :meth:`.Session.get_bind` is called via an ORM
        operation such as a :meth:`.Session.query`, each
        individual INSERT/UPDATE/DELETE operation within a
        :meth:`.Session.flush`, call, etc.

        The order of resolution is:

        1. if mapper given and :paramref:`.Session.binds` is present,
           locate a bind based first on the mapper in use, then
           on the mapped class in use, then on any base classes that are
           present in the ``__mro__`` of the mapped class, from more specific
           superclasses to more general.
        2. if clause given and ``Session.binds`` is present,
           locate a bind based on :class:`_schema.Table` objects
           found in the given clause present in ``Session.binds``.
        3. if ``Session.binds`` is present, return that.
        4. if clause given, attempt to return a bind
           linked to the :class:`_schema.MetaData` ultimately
           associated with the clause.
        5. if mapper given, attempt to return a bind
           linked to the :class:`_schema.MetaData` ultimately
           associated with the :class:`_schema.Table` or other
           selectable to which the mapper is mapped.
        6. No bind can be found, :exc:`~sqlalchemy.exc.UnboundExecutionError`
           is raised.

        Note that the :meth:`.Session.get_bind` method can be overridden on
        a user-defined subclass of :class:`.Session` to provide any kind
        of bind resolution scheme.  See the example at
        :ref:`session_custom_partitioning`.

        :param mapper:
          Optional mapped class or corresponding :class:`_orm.Mapper` instance.
          The bind can be derived from a :class:`_orm.Mapper` first by
          consulting the "binds" map associated with this :class:`.Session`,
          and secondly by consulting the :class:`_schema.MetaData` associated
          with the :class:`_schema.Table` to which the :class:`_orm.Mapper` is
          mapped for a bind.

        :param clause:
            A :class:`_expression.ClauseElement` (i.e.
            :func:`_expression.select`,
            :func:`_expression.text`,
            etc.).  If the ``mapper`` argument is not present or could not
            produce a bind, the given expression construct will be searched
            for a bound element, typically a :class:`_schema.Table`
            associated with
            bound :class:`_schema.MetaData`.

        .. seealso::

             :ref:`session_partitioning`

             :paramref:`.Session.binds`

             :meth:`.Session.bind_mapper`

             :meth:`.Session.bind_table`

        """

        # this function is documented as a subclassing hook, so we have
        # to call this method even if the return is simple
        if bind:
            return bind
        elif not self.__binds and self.bind:
            # simplest and most common case, we have a bind and no
            # per-mapper/table binds, we're done
            return self.bind

        # we don't have self.bind and either have self.__binds
        # or we don't have self.__binds (which is legacy).  Look at the
        # mapper and the clause
        if mapper is None and clause is None:
            if self.bind:
                return self.bind
            else:
                raise sa_exc.UnboundExecutionError(
                    "This session is not bound to a single Engine or "
                    "Connection, and no context was provided to locate "
                    "a binding."
                )

        # look more closely at the mapper.
        if mapper is not None:
            try:
                inspected_mapper = inspect(mapper)
            except sa_exc.NoInspectionAvailable as err:
                if isinstance(mapper, type):
                    raise exc.UnmappedClassError(mapper) from err
                else:
                    raise
        else:
            inspected_mapper = None

        # match up the mapper or clause in the __binds
        if self.__binds:
            # matching mappers and selectables to entries in the
            # binds dictionary; supported use case.
            if inspected_mapper:
                for cls in inspected_mapper.class_.__mro__:
                    if cls in self.__binds:
                        return self.__binds[cls]
                if clause is None:
                    clause = inspected_mapper.persist_selectable

            if clause is not None:
                plugin_subject = clause._propagate_attrs.get(
                    "plugin_subject", None
                )

                if plugin_subject is not None:
                    for cls in plugin_subject.mapper.class_.__mro__:
                        if cls in self.__binds:
                            return self.__binds[cls]

                for obj in visitors.iterate(clause):
                    if obj in self.__binds:
                        if TYPE_CHECKING:
                            assert isinstance(obj, Table)
                        return self.__binds[obj]

        # none of the __binds matched, but we have a fallback bind.
        # return that
        if self.bind:
            return self.bind

        context = []
        if inspected_mapper is not None:
            context.append(f"mapper {inspected_mapper}")
        if clause is not None:
            context.append("SQL expression")

        raise sa_exc.UnboundExecutionError(
            f"Could not locate a bind configured on "
            f'{", ".join(context)} or this Session.'
        )

    @overload
    def query(self, _entity: _EntityType[_O]) -> Query[_O]: ...

    @overload
    def query(
        self, _colexpr: TypedColumnsClauseRole[_T]
    ) -> RowReturningQuery[Tuple[_T]]: ...

    # START OVERLOADED FUNCTIONS self.query RowReturningQuery 2-8

    # code within this block is **programmatically,
    # statically generated** by tools/generate_tuple_map_overloads.py

    @overload
    def query(
        self, __ent0: _TCCA[_T0], __ent1: _TCCA[_T1]
    ) -> RowReturningQuery[Tuple[_T0, _T1]]: ...

    @overload
    def query(
        self, __ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2]
    ) -> RowReturningQuery[Tuple[_T0, _T1, _T2]]: ...

    @overload
    def query(
        self,
        __ent0: _TCCA[_T0],
        __ent1: _TCCA[_T1],
        __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3],
    ) -> RowReturningQuery[Tuple[_T0, _T1, _T2, _T3]]: ...

    @overload
    def query(
        self,
        __ent0: _TCCA[_T0],
        __ent1: _TCCA[_T1],
        __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3],
        __ent4: _TCCA[_T4],
    ) -> RowReturningQuery[Tuple[_T0, _T1, _T2, _T3, _T4]]: ...

    @overload
    def query(
        self,
        __ent0: _TCCA[_T0],
        __ent1: _TCCA[_T1],
        __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3],
        __ent4: _TCCA[_T4],
        __ent5: _TCCA[_T5],
    ) -> RowReturningQuery[Tuple[_T0, _T1, _T2, _T3, _T4, _T5]]: ...

    @overload
    def query(
        self,
        __ent0: _TCCA[_T0],
        __ent1: _TCCA[_T1],
        __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3],
        __ent4: _TCCA[_T4],
        __ent5: _TCCA[_T5],
        __ent6: _TCCA[_T6],
    ) -> RowReturningQuery[Tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6]]: ...

    @overload
    def query(
        self,
        __ent0: _TCCA[_T0],
        __ent1: _TCCA[_T1],
        __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3],
        __ent4: _TCCA[_T4],
        __ent5: _TCCA[_T5],
        __ent6: _TCCA[_T6],
        __ent7: _TCCA[_T7],
    ) -> RowReturningQuery[Tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7]]: ...

    # END OVERLOADED FUNCTIONS self.query

    @overload
    def query(
        self, *entities: _ColumnsClauseArgument[Any], **kwargs: Any
    ) -> Query[Any]: ...

    def query(
        self, *entities: _ColumnsClauseArgument[Any], **kwargs: Any
    ) -> Query[Any]:
        """Return a new :class:`_query.Query` object corresponding to this
        :class:`_orm.Session`.

        Note that the :class:`_query.Query` object is legacy as of
        SQLAlchemy 2.0; the :func:`_sql.select` construct is now used
        to construct ORM queries.

        .. seealso::

            :ref:`unified_tutorial`

            :ref:`queryguide_toplevel`

            :ref:`query_api_toplevel` - legacy API doc

        """

        return self._query_cls(entities, self, **kwargs)

    def _identity_lookup(
        self,
        mapper: Mapper[_O],
        primary_key_identity: Union[Any, Tuple[Any, ...]],
        identity_token: Any = None,
        passive: PassiveFlag = PassiveFlag.PASSIVE_OFF,
        lazy_loaded_from: Optional[InstanceState[Any]] = None,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
    ) -> Union[Optional[_O], LoaderCallableStatus]:
        """Locate an object in the identity map.

        Given a primary key identity, constructs an identity key and then
        looks in the session's identity map.  If present, the object may
        be run through unexpiration rules (e.g. load unloaded attributes,
        check if was deleted).

        e.g.::

            obj = session._identity_lookup(inspect(SomeClass), (1,))

        :param mapper: mapper in use
        :param primary_key_identity: the primary key we are searching for, as
         a tuple.
        :param identity_token: identity token that should be used to create
         the identity key.  Used as is, however overriding subclasses can
         repurpose this in order to interpret the value in a special way,
         such as if None then look among multiple target tokens.
        :param passive: passive load flag passed to
         :func:`.loading.get_from_identity`, which impacts the behavior if
         the object is found; the object may be validated and/or unexpired
         if the flag allows for SQL to be emitted.
        :param lazy_loaded_from: an :class:`.InstanceState` that is
         specifically asking for this identity as a related identity.  Used
         for sharding schemes where there is a correspondence between an object
         and a related object being lazy-loaded (or otherwise
         relationship-loaded).

        :return: None if the object is not found in the identity map, *or*
         if the object was unexpired and found to have been deleted.
         if passive flags disallow SQL and the object is expired, returns
         PASSIVE_NO_RESULT.   In all other cases the instance is returned.

        .. versionchanged:: 1.4.0 - the :meth:`.Session._identity_lookup`
           method was moved from :class:`_query.Query` to
           :class:`.Session`, to avoid having to instantiate the
           :class:`_query.Query` object.


        """

        key = mapper.identity_key_from_primary_key(
            primary_key_identity, identity_token=identity_token
        )

        # work around: https://github.com/python/typing/discussions/1143
        return_value = loading.get_from_identity(self, mapper, key, passive)
        return return_value

    @util.non_memoized_property
    @contextlib.contextmanager
    def no_autoflush(self) -> Iterator[Session]:
        """Return a context manager that disables autoflush.

        e.g.::

            with session.no_autoflush:

                some_object = SomeClass()
                session.add(some_object)
                # won't autoflush
                some_object.related_thing = session.query(SomeRelated).first()

        Operations that proceed within the ``with:`` block
        will not be subject to flushes occurring upon query
        access.  This is useful when initializing a series
        of objects which involve existing database queries,
        where the uncompleted object should not yet be flushed.

        """
        autoflush = self.autoflush
        self.autoflush = False
        try:
            yield self
        finally:
            self.autoflush = autoflush

    @util.langhelpers.tag_method_for_warnings(
        "This warning originated from the Session 'autoflush' process, "
        "which was invoked automatically in response to a user-initiated "
        "operation. Consider using ``no_autoflush`` context manager if this "
        "warning happended while initializing objects.",
        sa_exc.SAWarning,
    )
    def _autoflush(self) -> None:
        if self.autoflush and not self._flushing:
            try:
                self.flush()
            except sa_exc.StatementError as e:
                # note we are reraising StatementError as opposed to
                # raising FlushError with "chaining" to remain compatible
                # with code that catches StatementError, IntegrityError,
                # etc.
                e.add_detail(
                    "raised as a result of Query-invoked autoflush; "
                    "consider using a session.no_autoflush block if this "
                    "flush is occurring prematurely"
                )
                raise e.with_traceback(sys.exc_info()[2])

    def refresh(
        self,
        instance: object,
        attribute_names: Optional[Iterable[str]] = None,
        with_for_update: ForUpdateParameter = None,
    ) -> None:
        """Expire and refresh attributes on the given instance.

        The selected attributes will first be expired as they would when using
        :meth:`_orm.Session.expire`; then a SELECT statement will be issued to
        the database to refresh column-oriented attributes with the current
        value available in the current transaction.

        :func:`_orm.relationship` oriented attributes will also be immediately
        loaded if they were already eagerly loaded on the object, using the
        same eager loading strategy that they were loaded with originally.

        .. versionadded:: 1.4 - the :meth:`_orm.Session.refresh` method
           can also refresh eagerly loaded attributes.

        :func:`_orm.relationship` oriented attributes that would normally
        load using the ``select`` (or "lazy") loader strategy will also
        load **if they are named explicitly in the attribute_names
        collection**, emitting a SELECT statement for the attribute using the
        ``immediate`` loader strategy.  If lazy-loaded relationships are not
        named in :paramref:`_orm.Session.refresh.attribute_names`, then
        they remain as "lazy loaded" attributes and are not implicitly
        refreshed.

        .. versionchanged:: 2.0.4  The :meth:`_orm.Session.refresh` method
           will now refresh lazy-loaded :func:`_orm.relationship` oriented
           attributes for those which are named explicitly in the
           :paramref:`_orm.Session.refresh.attribute_names` collection.

        .. tip::

            While the :meth:`_orm.Session.refresh` method is capable of
            refreshing both column and relationship oriented attributes, its
            primary focus is on refreshing of local column-oriented attributes
            on a single instance. For more open ended "refresh" functionality,
            including the ability to refresh the attributes on many objects at
            once while having explicit control over relationship loader
            strategies, use the
            :ref:`populate existing <orm_queryguide_populate_existing>` feature
            instead.

        Note that a highly isolated transaction will return the same values as
        were previously read in that same transaction, regardless of changes
        in database state outside of that transaction.   Refreshing
        attributes usually only makes sense at the start of a transaction
        where database rows have not yet been accessed.

        :param attribute_names: optional.  An iterable collection of
          string attribute names indicating a subset of attributes to
          be refreshed.

        :param with_for_update: optional boolean ``True`` indicating FOR UPDATE
          should be used, or may be a dictionary containing flags to
          indicate a more specific set of FOR UPDATE flags for the SELECT;
          flags should match the parameters of
          :meth:`_query.Query.with_for_update`.
          Supersedes the :paramref:`.Session.refresh.lockmode` parameter.

        .. seealso::

            :ref:`session_expire` - introductory material

            :meth:`.Session.expire`

            :meth:`.Session.expire_all`

            :ref:`orm_queryguide_populate_existing` - allows any ORM query
            to refresh objects as they would be loaded normally.

        """
        try:
            state = attributes.instance_state(instance)
        except exc.NO_STATE as err:
            raise exc.UnmappedInstanceError(instance) from err

        self._expire_state(state, attribute_names)

        # this autoflush previously used to occur as a secondary effect
        # of the load_on_ident below.   Meaning we'd organize the SELECT
        # based on current DB pks, then flush, then if pks changed in that
        # flush, crash.  this was unticketed but discovered as part of
        # #8703.  So here, autoflush up front, dont autoflush inside
        # load_on_ident.
        self._autoflush()

        if with_for_update == {}:
            raise sa_exc.ArgumentError(
                "with_for_update should be the boolean value "
                "True, or a dictionary with options.  "
                "A blank dictionary is ambiguous."
            )

        with_for_update = ForUpdateArg._from_argument(with_for_update)

        stmt: Select[Any] = sql.select(object_mapper(instance))
        if (
            loading.load_on_ident(
                self,
                stmt,
                state.key,
                refresh_state=state,
                with_for_update=with_for_update,
                only_load_props=attribute_names,
                require_pk_cols=True,
                # technically unnecessary as we just did autoflush
                # above, however removes the additional unnecessary
                # call to _autoflush()
                no_autoflush=True,
                is_user_refresh=True,
            )
            is None
        ):
            raise sa_exc.InvalidRequestError(
                "Could not refresh instance '%s'" % instance_str(instance)
            )

    def expire_all(self) -> None:
        """Expires all persistent instances within this Session.

        When any attributes on a persistent instance is next accessed,
        a query will be issued using the
        :class:`.Session` object's current transactional context in order to
        load all expired attributes for the given instance.   Note that
        a highly isolated transaction will return the same values as were
        previously read in that same transaction, regardless of changes
        in database state outside of that transaction.

        To expire individual objects and individual attributes
        on those objects, use :meth:`Session.expire`.

        The :class:`.Session` object's default behavior is to
        expire all state whenever the :meth:`Session.rollback`
        or :meth:`Session.commit` methods are called, so that new
        state can be loaded for the new transaction.   For this reason,
        calling :meth:`Session.expire_all` is not usually needed,
        assuming the transaction is isolated.

        .. seealso::

            :ref:`session_expire` - introductory material

            :meth:`.Session.expire`

            :meth:`.Session.refresh`

            :meth:`_orm.Query.populate_existing`

        """
        for state in self.identity_map.all_states():
            state._expire(state.dict, self.identity_map._modified)

    def expire(
        self, instance: object, attribute_names: Optional[Iterable[str]] = None
    ) -> None:
        """Expire the attributes on an instance.

        Marks the attributes of an instance as out of date. When an expired
        attribute is next accessed, a query will be issued to the
        :class:`.Session` object's current transactional context in order to
        load all expired attributes for the given instance.   Note that
        a highly isolated transaction will return the same values as were
        previously read in that same transaction, regardless of changes
        in database state outside of that transaction.

        To expire all objects in the :class:`.Session` simultaneously,
        use :meth:`Session.expire_all`.

        The :class:`.Session` object's default behavior is to
        expire all state whenever the :meth:`Session.rollback`
        or :meth:`Session.commit` methods are called, so that new
        state can be loaded for the new transaction.   For this reason,
        calling :meth:`Session.expire` only makes sense for the specific
        case that a non-ORM SQL statement was emitted in the current
        transaction.

        :param instance: The instance to be refreshed.
        :param attribute_names: optional list of string attribute names
          indicating a subset of attributes to be expired.

        .. seealso::

            :ref:`session_expire` - introductory material

            :meth:`.Session.expire`

            :meth:`.Session.refresh`

            :meth:`_orm.Query.populate_existing`

        """
        try:
            state = attributes.instance_state(instance)
        except exc.NO_STATE as err:
            raise exc.UnmappedInstanceError(instance) from err
        self._expire_state(state, attribute_names)

    def _expire_state(
        self,
        state: InstanceState[Any],
        attribute_names: Optional[Iterable[str]],
    ) -> None:
        self._validate_persistent(state)
        if attribute_names:
            state._expire_attributes(state.dict, attribute_names)
        else:
            # pre-fetch the full cascade since the expire is going to
            # remove associations
            cascaded = list(
                state.manager.mapper.cascade_iterator("refresh-expire", state)
            )
            self._conditional_expire(state)
            for o, m, st_, dct_ in cascaded:
                self._conditional_expire(st_)

    def _conditional_expire(
        self, state: InstanceState[Any], autoflush: Optional[bool] = None
    ) -> None:
        """Expire a state if persistent, else expunge if pending"""

        if state.key:
            state._expire(state.dict, self.identity_map._modified)
        elif state in self._new:
            self._new.pop(state)
            state._detach(self)

    def expunge(self, instance: object) -> None:
        """Remove the `instance` from this ``Session``.

        This will free all internal references to the instance.  Cascading
        will be applied according to the *expunge* cascade rule.

        """
        try:
            state = attributes.instance_state(instance)
        except exc.NO_STATE as err:
            raise exc.UnmappedInstanceError(instance) from err
        if state.session_id is not self.hash_key:
            raise sa_exc.InvalidRequestError(
                "Instance %s is not present in this Session" % state_str(state)
            )

        cascaded = list(
            state.manager.mapper.cascade_iterator("expunge", state)
        )
        self._expunge_states([state] + [st_ for o, m, st_, dct_ in cascaded])

    def _expunge_states(
        self, states: Iterable[InstanceState[Any]], to_transient: bool = False
    ) -> None:
        for state in states:
            if state in self._new:
                self._new.pop(state)
            elif self.identity_map.contains_state(state):
                self.identity_map.safe_discard(state)
                self._deleted.pop(state, None)
            elif self._transaction:
                # state is "detached" from being deleted, but still present
                # in the transaction snapshot
                self._transaction._deleted.pop(state, None)
        statelib.InstanceState._detach_states(
            states, self, to_transient=to_transient
        )

    def _register_persistent(self, states: Set[InstanceState[Any]]) -> None:
        """Register all persistent objects from a flush.

        This is used both for pending objects moving to the persistent
        state as well as already persistent objects.

        """

        pending_to_persistent = self.dispatch.pending_to_persistent or None
        for state in states:
            mapper = _state_mapper(state)

            # prevent against last minute dereferences of the object
            obj = state.obj()
            if obj is not None:
                instance_key = mapper._identity_key_from_state(state)

                if (
                    _none_set.intersection(instance_key[1])
                    and not mapper.allow_partial_pks
                    or _none_set.issuperset(instance_key[1])
                ):
                    raise exc.FlushError(
                        "Instance %s has a NULL identity key.  If this is an "
                        "auto-generated value, check that the database table "
                        "allows generation of new primary key values, and "
                        "that the mapped Column object is configured to "
                        "expect these generated values.  Ensure also that "
                        "this flush() is not occurring at an inappropriate "
                        "time, such as within a load() event."
                        % state_str(state)
                    )

                if state.key is None:
                    state.key = instance_key
                elif state.key != instance_key:
                    # primary key switch. use safe_discard() in case another
                    # state has already replaced this one in the identity
                    # map (see test/orm/test_naturalpks.py ReversePKsTest)
                    self.identity_map.safe_discard(state)
                    trans = self._transaction
                    assert trans is not None
                    if state in trans._key_switches:
                        orig_key = trans._key_switches[state][0]
                    else:
                        orig_key = state.key
                    trans._key_switches[state] = (
                        orig_key,
                        instance_key,
                    )
                    state.key = instance_key

                # there can be an existing state in the identity map
                # that is replaced when the primary keys of two instances
                # are swapped; see test/orm/test_naturalpks.py -> test_reverse
                old = self.identity_map.replace(state)
                if (
                    old is not None
                    and mapper._identity_key_from_state(old) == instance_key
                    and old.obj() is not None
                ):
                    util.warn(
                        "Identity map already had an identity for %s, "
                        "replacing it with newly flushed object.   Are there "
                        "load operations occurring inside of an event handler "
                        "within the flush?" % (instance_key,)
                    )
                state._orphaned_outside_of_session = False

        statelib.InstanceState._commit_all_states(
            ((state, state.dict) for state in states), self.identity_map
        )

        self._register_altered(states)

        if pending_to_persistent is not None:
            for state in states.intersection(self._new):
                pending_to_persistent(self, state)

        # remove from new last, might be the last strong ref
        for state in set(states).intersection(self._new):
            self._new.pop(state)

    def _register_altered(self, states: Iterable[InstanceState[Any]]) -> None:
        if self._transaction:
            for state in states:
                if state in self._new:
                    self._transaction._new[state] = True
                else:
                    self._transaction._dirty[state] = True

    def _remove_newly_deleted(
        self, states: Iterable[InstanceState[Any]]
    ) -> None:
        persistent_to_deleted = self.dispatch.persistent_to_deleted or None
        for state in states:
            if self._transaction:
                self._transaction._deleted[state] = True

            if persistent_to_deleted is not None:
                # get a strong reference before we pop out of
                # self._deleted
                obj = state.obj()  # noqa

            self.identity_map.safe_discard(state)
            self._deleted.pop(state, None)
            state._deleted = True
            # can't call state._detach() here, because this state
            # is still in the transaction snapshot and needs to be
            # tracked as part of that
            if persistent_to_deleted is not None:
                persistent_to_deleted(self, state)

    def add(self, instance: object, _warn: bool = True) -> None:
        """Place an object into this :class:`_orm.Session`.

        Objects that are in the :term:`transient` state when passed to the
        :meth:`_orm.Session.add` method will move to the
        :term:`pending` state, until the next flush, at which point they
        will move to the :term:`persistent` state.

        Objects that are in the :term:`detached` state when passed to the
        :meth:`_orm.Session.add` method will move to the :term:`persistent`
        state directly.

        If the transaction used by the :class:`_orm.Session` is rolled back,
        objects which were transient when they were passed to
        :meth:`_orm.Session.add` will be moved back to the
        :term:`transient` state, and will no longer be present within this
        :class:`_orm.Session`.

        .. seealso::

            :meth:`_orm.Session.add_all`

            :ref:`session_adding` - at :ref:`session_basics`

        """
        if _warn and self._warn_on_events:
            self._flush_warning("Session.add()")

        try:
            state = attributes.instance_state(instance)
        except exc.NO_STATE as err:
            raise exc.UnmappedInstanceError(instance) from err

        self._save_or_update_state(state)

    def add_all(self, instances: Iterable[object]) -> None:
        """Add the given collection of instances to this :class:`_orm.Session`.

        See the documentation for :meth:`_orm.Session.add` for a general
        behavioral description.

        .. seealso::

            :meth:`_orm.Session.add`

            :ref:`session_adding` - at :ref:`session_basics`

        """

        if self._warn_on_events:
            self._flush_warning("Session.add_all()")

        for instance in instances:
            self.add(instance, _warn=False)

    def _save_or_update_state(self, state: InstanceState[Any]) -> None:
        state._orphaned_outside_of_session = False
        self._save_or_update_impl(state)

        mapper = _state_mapper(state)
        for o, m, st_, dct_ in mapper.cascade_iterator(
            "save-update", state, halt_on=self._contains_state
        ):
            self._save_or_update_impl(st_)

    def delete(self, instance: object) -> None:
        """Mark an instance as deleted.

        The object is assumed to be either :term:`persistent` or
        :term:`detached` when passed; after the method is called, the
        object will remain in the :term:`persistent` state until the next
        flush proceeds.  During this time, the object will also be a member
        of the :attr:`_orm.Session.deleted` collection.

        When the next flush proceeds, the object will move to the
        :term:`deleted` state, indicating a ``DELETE`` statement was emitted
        for its row within the current transaction.   When the transaction
        is successfully committed,
        the deleted object is moved to the :term:`detached` state and is
        no longer present within this :class:`_orm.Session`.

        .. seealso::

            :ref:`session_deleting` - at :ref:`session_basics`

        """
        if self._warn_on_events:
            self._flush_warning("Session.delete()")

        try:
            state = attributes.instance_state(instance)
        except exc.NO_STATE as err:
            raise exc.UnmappedInstanceError(instance) from err

        self._delete_impl(state, instance, head=True)

    def _delete_impl(
        self, state: InstanceState[Any], obj: object, head: bool
    ) -> None:
        if state.key is None:
            if head:
                raise sa_exc.InvalidRequestError(
                    "Instance '%s' is not persisted" % state_str(state)
                )
            else:
                return

        to_attach = self._before_attach(state, obj)

        if state in self._deleted:
            return

        self.identity_map.add(state)

        if to_attach:
            self._after_attach(state, obj)

        if head:
            # grab the cascades before adding the item to the deleted list
            # so that autoflush does not delete the item
            # the strong reference to the instance itself is significant here
            cascade_states = list(
                state.manager.mapper.cascade_iterator("delete", state)
            )
        else:
            cascade_states = None

        self._deleted[state] = obj

        if head:
            if TYPE_CHECKING:
                assert cascade_states is not None
            for o, m, st_, dct_ in cascade_states:
                self._delete_impl(st_, o, False)

    def get(
        self,
        entity: _EntityBindKey[_O],
        ident: _PKIdentityArgument,
        *,
        options: Optional[Sequence[ORMOption]] = None,
        populate_existing: bool = False,
        with_for_update: ForUpdateParameter = None,
        identity_token: Optional[Any] = None,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
    ) -> Optional[_O]:
        """Return an instance based on the given primary key identifier,
        or ``None`` if not found.

        E.g.::

            my_user = session.get(User, 5)

            some_object = session.get(VersionedFoo, (5, 10))

            some_object = session.get(VersionedFoo, {"id": 5, "version_id": 10})

        .. versionadded:: 1.4 Added :meth:`_orm.Session.get`, which is moved
           from the now legacy :meth:`_orm.Query.get` method.

        :meth:`_orm.Session.get` is special in that it provides direct
        access to the identity map of the :class:`.Session`.
        If the given primary key identifier is present
        in the local identity map, the object is returned
        directly from this collection and no SQL is emitted,
        unless the object has been marked fully expired.
        If not present,
        a SELECT is performed in order to locate the object.

        :meth:`_orm.Session.get` also will perform a check if
        the object is present in the identity map and
        marked as expired - a SELECT
        is emitted to refresh the object as well as to
        ensure that the row is still present.
        If not, :class:`~sqlalchemy.orm.exc.ObjectDeletedError` is raised.

        :param entity: a mapped class or :class:`.Mapper` indicating the
         type of entity to be loaded.

        :param ident: A scalar, tuple, or dictionary representing the
         primary key.  For a composite (e.g. multiple column) primary key,
         a tuple or dictionary should be passed.

         For a single-column primary key, the scalar calling form is typically
         the most expedient.  If the primary key of a row is the value "5",
         the call looks like::

            my_object = session.get(SomeClass, 5)

         The tuple form contains primary key values typically in
         the order in which they correspond to the mapped
         :class:`_schema.Table`
         object's primary key columns, or if the
         :paramref:`_orm.Mapper.primary_key` configuration parameter were
         used, in
         the order used for that parameter. For example, if the primary key
         of a row is represented by the integer
         digits "5, 10" the call would look like::

             my_object = session.get(SomeClass, (5, 10))

         The dictionary form should include as keys the mapped attribute names
         corresponding to each element of the primary key.  If the mapped class
         has the attributes ``id``, ``version_id`` as the attributes which
         store the object's primary key value, the call would look like::

            my_object = session.get(SomeClass, {"id": 5, "version_id": 10})

        :param options: optional sequence of loader options which will be
         applied to the query, if one is emitted.

        :param populate_existing: causes the method to unconditionally emit
         a SQL query and refresh the object with the newly loaded data,
         regardless of whether or not the object is already present.

        :param with_for_update: optional boolean ``True`` indicating FOR UPDATE
          should be used, or may be a dictionary containing flags to
          indicate a more specific set of FOR UPDATE flags for the SELECT;
          flags should match the parameters of
          :meth:`_query.Query.with_for_update`.
          Supersedes the :paramref:`.Session.refresh.lockmode` parameter.

        :param execution_options: optional dictionary of execution options,
         which will be associated with the query execution if one is emitted.
         This dictionary can provide a subset of the options that are
         accepted by :meth:`_engine.Connection.execution_options`, and may
         also provide additional options understood only in an ORM context.

         .. versionadded:: 1.4.29

         .. seealso::

            :ref:`orm_queryguide_execution_options` - ORM-specific execution
            options

        :param bind_arguments: dictionary of additional arguments to determine
         the bind.  May include "mapper", "bind", or other custom arguments.
         Contents of this dictionary are passed to the
         :meth:`.Session.get_bind` method.

         .. versionadded: 2.0.0rc1

        :return: The object instance, or ``None``.

        """  # noqa: E501
        return self._get_impl(
            entity,
            ident,
            loading.load_on_pk_identity,
            options=options,
            populate_existing=populate_existing,
            with_for_update=with_for_update,
            identity_token=identity_token,
            execution_options=execution_options,
            bind_arguments=bind_arguments,
        )

    def get_one(
        self,
        entity: _EntityBindKey[_O],
        ident: _PKIdentityArgument,
        *,
        options: Optional[Sequence[ORMOption]] = None,
        populate_existing: bool = False,
        with_for_update: ForUpdateParameter = None,
        identity_token: Optional[Any] = None,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
    ) -> _O:
        """Return exactly one instance based on the given primary key
        identifier, or raise an exception if not found.

        Raises :class:`_exc.NoResultFound` if the query selects no rows.

        For a detailed documentation of the arguments see the
        method :meth:`.Session.get`.

        .. versionadded:: 2.0.22

        :return: The object instance.

        .. seealso::

            :meth:`.Session.get` - equivalent method that instead
              returns ``None`` if no row was found with the provided primary
              key

        """

        instance = self.get(
            entity,
            ident,
            options=options,
            populate_existing=populate_existing,
            with_for_update=with_for_update,
            identity_token=identity_token,
            execution_options=execution_options,
            bind_arguments=bind_arguments,
        )

        if instance is None:
            raise sa_exc.NoResultFound(
                "No row was found when one was required"
            )

        return instance

    def _get_impl(
        self,
        entity: _EntityBindKey[_O],
        primary_key_identity: _PKIdentityArgument,
        db_load_fn: Callable[..., _O],
        *,
        options: Optional[Sequence[ExecutableOption]] = None,
        populate_existing: bool = False,
        with_for_update: ForUpdateParameter = None,
        identity_token: Optional[Any] = None,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
    ) -> Optional[_O]:
        # convert composite types to individual args
        if (
            is_composite_class(primary_key_identity)
            and type(primary_key_identity)
            in descriptor_props._composite_getters
        ):
            getter = descriptor_props._composite_getters[
                type(primary_key_identity)
            ]
            primary_key_identity = getter(primary_key_identity)

        mapper: Optional[Mapper[_O]] = inspect(entity)

        if mapper is None or not mapper.is_mapper:
            raise sa_exc.ArgumentError(
                "Expected mapped class or mapper, got: %r" % entity
            )

        is_dict = isinstance(primary_key_identity, dict)
        if not is_dict:
            primary_key_identity = util.to_list(
                primary_key_identity, default=[None]
            )

        if len(primary_key_identity) != len(mapper.primary_key):
            raise sa_exc.InvalidRequestError(
                "Incorrect number of values in identifier to formulate "
                "primary key for session.get(); primary key columns "
                "are %s" % ",".join("'%s'" % c for c in mapper.primary_key)
            )

        if is_dict:
            pk_synonyms = mapper._pk_synonyms

            if pk_synonyms:
                correct_keys = set(pk_synonyms).intersection(
                    primary_key_identity
                )

                if correct_keys:
                    primary_key_identity = dict(primary_key_identity)
                    for k in correct_keys:
                        primary_key_identity[pk_synonyms[k]] = (
                            primary_key_identity[k]
                        )

            try:
                primary_key_identity = list(
                    primary_key_identity[prop.key]
                    for prop in mapper._identity_key_props
                )

            except KeyError as err:
                raise sa_exc.InvalidRequestError(
                    "Incorrect names of values in identifier to formulate "
                    "primary key for session.get(); primary key attribute "
                    "names are %s (synonym names are also accepted)"
                    % ",".join(
                        "'%s'" % prop.key
                        for prop in mapper._identity_key_props
                    )
                ) from err

        if (
            not populate_existing
            and not mapper.always_refresh
            and with_for_update is None
        ):
            instance = self._identity_lookup(
                mapper,
                primary_key_identity,
                identity_token=identity_token,
                execution_options=execution_options,
                bind_arguments=bind_arguments,
            )

            if instance is not None:
                # reject calls for id in identity map but class
                # mismatch.
                if not isinstance(instance, mapper.class_):
                    return None
                return instance

            # TODO: this was being tested before, but this is not possible
            assert instance is not LoaderCallableStatus.PASSIVE_CLASS_MISMATCH

        # set_label_style() not strictly necessary, however this will ensure
        # that tablename_colname style is used which at the moment is
        # asserted in a lot of unit tests :)

        load_options = context.QueryContext.default_load_options

        if populate_existing:
            load_options += {"_populate_existing": populate_existing}
        statement = sql.select(mapper).set_label_style(
            LABEL_STYLE_TABLENAME_PLUS_COL
        )
        if with_for_update is not None:
            statement._for_update_arg = ForUpdateArg._from_argument(
                with_for_update
            )

        if options:
            statement = statement.options(*options)
        return db_load_fn(
            self,
            statement,
            primary_key_identity,
            load_options=load_options,
            identity_token=identity_token,
            execution_options=execution_options,
            bind_arguments=bind_arguments,
        )

    def merge(
        self,
        instance: _O,
        *,
        load: bool = True,
        options: Optional[Sequence[ORMOption]] = None,
    ) -> _O:
        """Copy the state of a given instance into a corresponding instance
        within this :class:`.Session`.

        :meth:`.Session.merge` examines the primary key attributes of the
        source instance, and attempts to reconcile it with an instance of the
        same primary key in the session.   If not found locally, it attempts
        to load the object from the database based on primary key, and if
        none can be located, creates a new instance.  The state of each
        attribute on the source instance is then copied to the target
        instance.  The resulting target instance is then returned by the
        method; the original source instance is left unmodified, and
        un-associated with the :class:`.Session` if not already.

        This operation cascades to associated instances if the association is
        mapped with ``cascade="merge"``.

        See :ref:`unitofwork_merging` for a detailed discussion of merging.

        :param instance: Instance to be merged.
        :param load: Boolean, when False, :meth:`.merge` switches into
         a "high performance" mode which causes it to forego emitting history
         events as well as all database access.  This flag is used for
         cases such as transferring graphs of objects into a :class:`.Session`
         from a second level cache, or to transfer just-loaded objects
         into the :class:`.Session` owned by a worker thread or process
         without re-querying the database.

         The ``load=False`` use case adds the caveat that the given
         object has to be in a "clean" state, that is, has no pending changes
         to be flushed - even if the incoming object is detached from any
         :class:`.Session`.   This is so that when
         the merge operation populates local attributes and
         cascades to related objects and
         collections, the values can be "stamped" onto the
         target object as is, without generating any history or attribute
         events, and without the need to reconcile the incoming data with
         any existing related objects or collections that might not
         be loaded.  The resulting objects from ``load=False`` are always
         produced as "clean", so it is only appropriate that the given objects
         should be "clean" as well, else this suggests a mis-use of the
         method.
        :param options: optional sequence of loader options which will be
         applied to the :meth:`_orm.Session.get` method when the merge
         operation loads the existing version of the object from the database.

         .. versionadded:: 1.4.24


        .. seealso::

            :func:`.make_transient_to_detached` - provides for an alternative
            means of "merging" a single object into the :class:`.Session`

        """

        if self._warn_on_events:
            self._flush_warning("Session.merge()")

        _recursive: Dict[InstanceState[Any], object] = {}
        _resolve_conflict_map: Dict[_IdentityKeyType[Any], object] = {}

        if load:
            # flush current contents if we expect to load data
            self._autoflush()

        object_mapper(instance)  # verify mapped
        autoflush = self.autoflush
        try:
            self.autoflush = False
            return self._merge(
                attributes.instance_state(instance),
                attributes.instance_dict(instance),
                load=load,
                options=options,
                _recursive=_recursive,
                _resolve_conflict_map=_resolve_conflict_map,
            )
        finally:
            self.autoflush = autoflush

    def _merge(
        self,
        state: InstanceState[_O],
        state_dict: _InstanceDict,
        *,
        options: Optional[Sequence[ORMOption]] = None,
        load: bool,
        _recursive: Dict[Any, object],
        _resolve_conflict_map: Dict[_IdentityKeyType[Any], object],
    ) -> _O:
        mapper: Mapper[_O] = _state_mapper(state)
        if state in _recursive:
            return cast(_O, _recursive[state])

        new_instance = False
        key = state.key

        merged: Optional[_O]

        if key is None:
            if state in self._new:
                util.warn(
                    "Instance %s is already pending in this Session yet is "
                    "being merged again; this is probably not what you want "
                    "to do" % state_str(state)
                )

            if not load:
                raise sa_exc.InvalidRequestError(
                    "merge() with load=False option does not support "
                    "objects transient (i.e. unpersisted) objects.  flush() "
                    "all changes on mapped instances before merging with "
                    "load=False."
                )
            key = mapper._identity_key_from_state(state)
            key_is_persistent = LoaderCallableStatus.NEVER_SET not in key[
                1
            ] and (
                not _none_set.intersection(key[1])
                or (
                    mapper.allow_partial_pks
                    and not _none_set.issuperset(key[1])
                )
            )
        else:
            key_is_persistent = True

        merged = self.identity_map.get(key)

        if merged is None:
            if key_is_persistent and key in _resolve_conflict_map:
                merged = cast(_O, _resolve_conflict_map[key])

            elif not load:
                if state.modified:
                    raise sa_exc.InvalidRequestError(
                        "merge() with load=False option does not support "
                        "objects marked as 'dirty'.  flush() all changes on "
                        "mapped instances before merging with load=False."
                    )
                merged = mapper.class_manager.new_instance()
                merged_state = attributes.instance_state(merged)
                merged_state.key = key
                self._update_impl(merged_state)
                new_instance = True

            elif key_is_persistent:
                merged = self.get(
                    mapper.class_,
                    key[1],
                    identity_token=key[2],
                    options=options,
                )

        if merged is None:
            merged = mapper.class_manager.new_instance()
            merged_state = attributes.instance_state(merged)
            merged_dict = attributes.instance_dict(merged)
            new_instance = True
            self._save_or_update_state(merged_state)
        else:
            merged_state = attributes.instance_state(merged)
            merged_dict = attributes.instance_dict(merged)

        _recursive[state] = merged
        _resolve_conflict_map[key] = merged

        # check that we didn't just pull the exact same
        # state out.
        if state is not merged_state:
            # version check if applicable
            if mapper.version_id_col is not None:
                existing_version = mapper._get_state_attr_by_column(
                    state,
                    state_dict,
                    mapper.version_id_col,
                    passive=PassiveFlag.PASSIVE_NO_INITIALIZE,
                )

                merged_version = mapper._get_state_attr_by_column(
                    merged_state,
                    merged_dict,
                    mapper.version_id_col,
                    passive=PassiveFlag.PASSIVE_NO_INITIALIZE,
                )

                if (
                    existing_version
                    is not LoaderCallableStatus.PASSIVE_NO_RESULT
                    and merged_version
                    is not LoaderCallableStatus.PASSIVE_NO_RESULT
                    and existing_version != merged_version
                ):
                    raise exc.StaleDataError(
                        "Version id '%s' on merged state %s "
                        "does not match existing version '%s'. "
                        "Leave the version attribute unset when "
                        "merging to update the most recent version."
                        % (
                            existing_version,
                            state_str(merged_state),
                            merged_version,
                        )
                    )

            merged_state.load_path = state.load_path
            merged_state.load_options = state.load_options

            # since we are copying load_options, we need to copy
            # the callables_ that would have been generated by those
            # load_options.
            # assumes that the callables we put in state.callables_
            # are not instance-specific (which they should not be)
            merged_state._copy_callables(state)

            for prop in mapper.iterate_properties:
                prop.merge(
                    self,
                    state,
                    state_dict,
                    merged_state,
                    merged_dict,
                    load,
                    _recursive,
                    _resolve_conflict_map,
                )

        if not load:
            # remove any history
            merged_state._commit_all(merged_dict, self.identity_map)
            merged_state.manager.dispatch._sa_event_merge_wo_load(
                merged_state, None
            )

        if new_instance:
            merged_state.manager.dispatch.load(merged_state, None)

        return merged

    def _validate_persistent(self, state: InstanceState[Any]) -> None:
        if not self.identity_map.contains_state(state):
            raise sa_exc.InvalidRequestError(
                "Instance '%s' is not persistent within this Session"
                % state_str(state)
            )

    def _save_impl(self, state: InstanceState[Any]) -> None:
        if state.key is not None:
            raise sa_exc.InvalidRequestError(
                "Object '%s' already has an identity - "
                "it can't be registered as pending" % state_str(state)
            )

        obj = state.obj()
        to_attach = self._before_attach(state, obj)
        if state not in self._new:
            self._new[state] = obj
            state.insert_order = len(self._new)
        if to_attach:
            self._after_attach(state, obj)

    def _update_impl(
        self, state: InstanceState[Any], revert_deletion: bool = False
    ) -> None:
        if state.key is None:
            raise sa_exc.InvalidRequestError(
                "Instance '%s' is not persisted" % state_str(state)
            )

        if state._deleted:
            if revert_deletion:
                if not state._attached:
                    return
                del state._deleted
            else:
                raise sa_exc.InvalidRequestError(
                    "Instance '%s' has been deleted.  "
                    "Use the make_transient() "
                    "function to send this object back "
                    "to the transient state." % state_str(state)
                )

        obj = state.obj()

        # check for late gc
        if obj is None:
            return

        to_attach = self._before_attach(state, obj)

        self._deleted.pop(state, None)
        if revert_deletion:
            self.identity_map.replace(state)
        else:
            self.identity_map.add(state)

        if to_attach:
            self._after_attach(state, obj)
        elif revert_deletion:
            self.dispatch.deleted_to_persistent(self, state)

    def _save_or_update_impl(self, state: InstanceState[Any]) -> None:
        if state.key is None:
            self._save_impl(state)
        else:
            self._update_impl(state)

    def enable_relationship_loading(self, obj: object) -> None:
        """Associate an object with this :class:`.Session` for related
        object loading.

        .. warning::

            :meth:`.enable_relationship_loading` exists to serve special
            use cases and is not recommended for general use.

        Accesses of attributes mapped with :func:`_orm.relationship`
        will attempt to load a value from the database using this
        :class:`.Session` as the source of connectivity.  The values
        will be loaded based on foreign key and primary key values
        present on this object - if not present, then those relationships
        will be unavailable.

        The object will be attached to this session, but will
        **not** participate in any persistence operations; its state
        for almost all purposes will remain either "transient" or
        "detached", except for the case of relationship loading.

        Also note that backrefs will often not work as expected.
        Altering a relationship-bound attribute on the target object
        may not fire off a backref event, if the effective value
        is what was already loaded from a foreign-key-holding value.

        The :meth:`.Session.enable_relationship_loading` method is
        similar to the ``load_on_pending`` flag on :func:`_orm.relationship`.
        Unlike that flag, :meth:`.Session.enable_relationship_loading` allows
        an object to remain transient while still being able to load
        related items.

        To make a transient object associated with a :class:`.Session`
        via :meth:`.Session.enable_relationship_loading` pending, add
        it to the :class:`.Session` using :meth:`.Session.add` normally.
        If the object instead represents an existing identity in the database,
        it should be merged using :meth:`.Session.merge`.

        :meth:`.Session.enable_relationship_loading` does not improve
        behavior when the ORM is used normally - object references should be
        constructed at the object level, not at the foreign key level, so
        that they are present in an ordinary way before flush()
        proceeds.  This method is not intended for general use.

        .. seealso::

            :paramref:`_orm.relationship.load_on_pending` - this flag
            allows per-relationship loading of many-to-ones on items that
            are pending.

            :func:`.make_transient_to_detached` - allows for an object to
            be added to a :class:`.Session` without SQL emitted, which then
            will unexpire attributes on access.

        """
        try:
            state = attributes.instance_state(obj)
        except exc.NO_STATE as err:
            raise exc.UnmappedInstanceError(obj) from err

        to_attach = self._before_attach(state, obj)
        state._load_pending = True
        if to_attach:
            self._after_attach(state, obj)

    def _before_attach(self, state: InstanceState[Any], obj: object) -> bool:
        self._autobegin_t()

        if state.session_id == self.hash_key:
            return False

        if state.session_id and state.session_id in _sessions:
            raise sa_exc.InvalidRequestError(
                "Object '%s' is already attached to session '%s' "
                "(this is '%s')"
                % (state_str(state), state.session_id, self.hash_key)
            )

        self.dispatch.before_attach(self, state)

        return True

    def _after_attach(self, state: InstanceState[Any], obj: object) -> None:
        state.session_id = self.hash_key
        if state.modified and state._strong_obj is None:
            state._strong_obj = obj
        self.dispatch.after_attach(self, state)

        if state.key:
            self.dispatch.detached_to_persistent(self, state)
        else:
            self.dispatch.transient_to_pending(self, state)

    def __contains__(self, instance: object) -> bool:
        """Return True if the instance is associated with this session.

        The instance may be pending or persistent within the Session for a
        result of True.

        """
        try:
            state = attributes.instance_state(instance)
        except exc.NO_STATE as err:
            raise exc.UnmappedInstanceError(instance) from err
        return self._contains_state(state)

    def __iter__(self) -> Iterator[object]:
        """Iterate over all pending or persistent instances within this
        Session.

        """
        return iter(
            list(self._new.values()) + list(self.identity_map.values())
        )

    def _contains_state(self, state: InstanceState[Any]) -> bool:
        return state in self._new or self.identity_map.contains_state(state)

    def flush(self, objects: Optional[Sequence[Any]] = None) -> None:
        """Flush all the object changes to the database.

        Writes out all pending object creations, deletions and modifications
        to the database as INSERTs, DELETEs, UPDATEs, etc.  Operations are
        automatically ordered by the Session's unit of work dependency
        solver.

        Database operations will be issued in the current transactional
        context and do not affect the state of the transaction, unless an
        error occurs, in which case the entire transaction is rolled back.
        You may flush() as often as you like within a transaction to move
        changes from Python to the database's transaction buffer.

        :param objects: Optional; restricts the flush operation to operate
          only on elements that are in the given collection.

          This feature is for an extremely narrow set of use cases where
          particular objects may need to be operated upon before the
          full flush() occurs.  It is not intended for general use.

        """

        if self._flushing:
            raise sa_exc.InvalidRequestError("Session is already flushing")

        if self._is_clean():
            return
        try:
            self._flushing = True
            self._flush(objects)
        finally:
            self._flushing = False

    def _flush_warning(self, method: Any) -> None:
        util.warn(
            "Usage of the '%s' operation is not currently supported "
            "within the execution stage of the flush process. "
            "Results may not be consistent.  Consider using alternative "
            "event listeners or connection-level operations instead." % method
        )

    def _is_clean(self) -> bool:
        return (
            not self.identity_map.check_modified()
            and not self._deleted
            and not self._new
        )

    def _flush(self, objects: Optional[Sequence[object]] = None) -> None:
        dirty = self._dirty_states
        if not dirty and not self._deleted and not self._new:
            self.identity_map._modified.clear()
            return

        flush_context = UOWTransaction(self)

        if self.dispatch.before_flush:
            self.dispatch.before_flush(self, flush_context, objects)
            # re-establish "dirty states" in case the listeners
            # added
            dirty = self._dirty_states

        deleted = set(self._deleted)
        new = set(self._new)

        dirty = set(dirty).difference(deleted)

        # create the set of all objects we want to operate upon
        if objects:
            # specific list passed in
            objset = set()
            for o in objects:
                try:
                    state = attributes.instance_state(o)

                except exc.NO_STATE as err:
                    raise exc.UnmappedInstanceError(o) from err
                objset.add(state)
        else:
            objset = None

        # store objects whose fate has been decided
        processed = set()

        # put all saves/updates into the flush context.  detect top-level
        # orphans and throw them into deleted.
        if objset:
            proc = new.union(dirty).intersection(objset).difference(deleted)
        else:
            proc = new.union(dirty).difference(deleted)

        for state in proc:
            is_orphan = _state_mapper(state)._is_orphan(state)

            is_persistent_orphan = is_orphan and state.has_identity

            if (
                is_orphan
                and not is_persistent_orphan
                and state._orphaned_outside_of_session
            ):
                self._expunge_states([state])
            else:
                _reg = flush_context.register_object(
                    state, isdelete=is_persistent_orphan
                )
                assert _reg, "Failed to add object to the flush context!"
                processed.add(state)

        # put all remaining deletes into the flush context.
        if objset:
            proc = deleted.intersection(objset).difference(processed)
        else:
            proc = deleted.difference(processed)
        for state in proc:
            _reg = flush_context.register_object(state, isdelete=True)
            assert _reg, "Failed to add object to the flush context!"

        if not flush_context.has_work:
            return

        flush_context.transaction = transaction = self._autobegin_t()._begin()
        try:
            self._warn_on_events = True
            try:
                flush_context.execute()
            finally:
                self._warn_on_events = False

            self.dispatch.after_flush(self, flush_context)

            flush_context.finalize_flush_changes()

            if not objects and self.identity_map._modified:
                len_ = len(self.identity_map._modified)

                statelib.InstanceState._commit_all_states(
                    [
                        (state, state.dict)
                        for state in self.identity_map._modified
                    ],
                    instance_dict=self.identity_map,
                )
                util.warn(
                    "Attribute history events accumulated on %d "
                    "previously clean instances "
                    "within inner-flush event handlers have been "
                    "reset, and will not result in database updates. "
                    "Consider using set_committed_value() within "
                    "inner-flush event handlers to avoid this warning." % len_
                )

            # useful assertions:
            # if not objects:
            #    assert not self.identity_map._modified
            # else:
            #    assert self.identity_map._modified == \
            #            self.identity_map._modified.difference(objects)

            self.dispatch.after_flush_postexec(self, flush_context)

            transaction.commit()

        except:
            with util.safe_reraise():
                transaction.rollback(_capture_exception=True)

    def bulk_save_objects(
        self,
        objects: Iterable[object],
        return_defaults: bool = False,
        update_changed_only: bool = True,
        preserve_order: bool = True,
    ) -> None:
        """Perform a bulk save of the given list of objects.

        .. legacy::

            This method is a legacy feature as of the 2.0 series of
            SQLAlchemy.   For modern bulk INSERT and UPDATE, see
            the sections :ref:`orm_queryguide_bulk_insert` and
            :ref:`orm_queryguide_bulk_update`.

            For general INSERT and UPDATE of existing ORM mapped objects,
            prefer standard :term:`unit of work` data management patterns,
            introduced in the :ref:`unified_tutorial` at
            :ref:`tutorial_orm_data_manipulation`.  SQLAlchemy 2.0
            now uses :ref:`engine_insertmanyvalues` with modern dialects
            which solves previous issues of bulk INSERT slowness.

        :param objects: a sequence of mapped object instances.  The mapped
         objects are persisted as is, and are **not** associated with the
         :class:`.Session` afterwards.

         For each object, whether the object is sent as an INSERT or an
         UPDATE is dependent on the same rules used by the :class:`.Session`
         in traditional operation; if the object has the
         :attr:`.InstanceState.key`
         attribute set, then the object is assumed to be "detached" and
         will result in an UPDATE.  Otherwise, an INSERT is used.

         In the case of an UPDATE, statements are grouped based on which
         attributes have changed, and are thus to be the subject of each
         SET clause.  If ``update_changed_only`` is False, then all
         attributes present within each object are applied to the UPDATE
         statement, which may help in allowing the statements to be grouped
         together into a larger executemany(), and will also reduce the
         overhead of checking history on attributes.

        :param return_defaults: when True, rows that are missing values which
         generate defaults, namely integer primary key defaults and sequences,
         will be inserted **one at a time**, so that the primary key value
         is available.  In particular this will allow joined-inheritance
         and other multi-table mappings to insert correctly without the need
         to provide primary key values ahead of time; however,
         :paramref:`.Session.bulk_save_objects.return_defaults` **greatly
         reduces the performance gains** of the method overall.  It is strongly
         advised to please use the standard :meth:`_orm.Session.add_all`
         approach.

        :param update_changed_only: when True, UPDATE statements are rendered
         based on those attributes in each state that have logged changes.
         When False, all attributes present are rendered into the SET clause
         with the exception of primary key attributes.

        :param preserve_order: when True, the order of inserts and updates
         matches exactly the order in which the objects are given.   When
         False, common types of objects are grouped into inserts
         and updates, to allow for more batching opportunities.

        .. seealso::

            :doc:`queryguide/dml`

            :meth:`.Session.bulk_insert_mappings`

            :meth:`.Session.bulk_update_mappings`

        """

        obj_states: Iterable[InstanceState[Any]]

        obj_states = (attributes.instance_state(obj) for obj in objects)

        if not preserve_order:
            # the purpose of this sort is just so that common mappers
            # and persistence states are grouped together, so that groupby
            # will return a single group for a particular type of mapper.
            # it's not trying to be deterministic beyond that.
            obj_states = sorted(
                obj_states,
                key=lambda state: (id(state.mapper), state.key is not None),
            )

        def grouping_key(
            state: InstanceState[_O],
        ) -> Tuple[Mapper[_O], bool]:
            return (state.mapper, state.key is not None)

        for (mapper, isupdate), states in itertools.groupby(
            obj_states, grouping_key
        ):
            self._bulk_save_mappings(
                mapper,
                states,
                isupdate=isupdate,
                isstates=True,
                return_defaults=return_defaults,
                update_changed_only=update_changed_only,
                render_nulls=False,
            )

    def bulk_insert_mappings(
        self,
        mapper: Mapper[Any],
        mappings: Iterable[Dict[str, Any]],
        return_defaults: bool = False,
        render_nulls: bool = False,
    ) -> None:
        """Perform a bulk insert of the given list of mapping dictionaries.

        .. legacy::

            This method is a legacy feature as of the 2.0 series of
            SQLAlchemy.   For modern bulk INSERT and UPDATE, see
            the sections :ref:`orm_queryguide_bulk_insert` and
            :ref:`orm_queryguide_bulk_update`.  The 2.0 API shares
            implementation details with this method and adds new features
            as well.

        :param mapper: a mapped class, or the actual :class:`_orm.Mapper`
         object,
         representing the single kind of object represented within the mapping
         list.

        :param mappings: a sequence of dictionaries, each one containing the
         state of the mapped row to be inserted, in terms of the attribute
         names on the mapped class.   If the mapping refers to multiple tables,
         such as a joined-inheritance mapping, each dictionary must contain all
         keys to be populated into all tables.

        :param return_defaults: when True, the INSERT process will be altered
         to ensure that newly generated primary key values will be fetched.
         The rationale for this parameter is typically to enable
         :ref:`Joined Table Inheritance <joined_inheritance>` mappings to
         be bulk inserted.

         .. note:: for backends that don't support RETURNING, the
            :paramref:`_orm.Session.bulk_insert_mappings.return_defaults`
            parameter can significantly decrease performance as INSERT
            statements can no longer be batched.   See
            :ref:`engine_insertmanyvalues`
            for background on which backends are affected.

        :param render_nulls: When True, a value of ``None`` will result
         in a NULL value being included in the INSERT statement, rather
         than the column being omitted from the INSERT.   This allows all
         the rows being INSERTed to have the identical set of columns which
         allows the full set of rows to be batched to the DBAPI.  Normally,
         each column-set that contains a different combination of NULL values
         than the previous row must omit a different series of columns from
         the rendered INSERT statement, which means it must be emitted as a
         separate statement.   By passing this flag, the full set of rows
         are guaranteed to be batchable into one batch; the cost however is
         that server-side defaults which are invoked by an omitted column will
         be skipped, so care must be taken to ensure that these are not
         necessary.

         .. warning::

            When this flag is set, **server side default SQL values will
            not be invoked** for those columns that are inserted as NULL;
            the NULL value will be sent explicitly.   Care must be taken
            to ensure that no server-side default functions need to be
            invoked for the operation as a whole.

        .. seealso::

            :doc:`queryguide/dml`

            :meth:`.Session.bulk_save_objects`

            :meth:`.Session.bulk_update_mappings`

        """
        self._bulk_save_mappings(
            mapper,
            mappings,
            isupdate=False,
            isstates=False,
            return_defaults=return_defaults,
            update_changed_only=False,
            render_nulls=render_nulls,
        )

    def bulk_update_mappings(
        self, mapper: Mapper[Any], mappings: Iterable[Dict[str, Any]]
    ) -> None:
        """Perform a bulk update of the given list of mapping dictionaries.

        .. legacy::

            This method is a legacy feature as of the 2.0 series of
            SQLAlchemy.   For modern bulk INSERT and UPDATE, see
            the sections :ref:`orm_queryguide_bulk_insert` and
            :ref:`orm_queryguide_bulk_update`.  The 2.0 API shares
            implementation details with this method and adds new features
            as well.

        :param mapper: a mapped class, or the actual :class:`_orm.Mapper`
         object,
         representing the single kind of object represented within the mapping
         list.

        :param mappings: a sequence of dictionaries, each one containing the
         state of the mapped row to be updated, in terms of the attribute names
         on the mapped class.   If the mapping refers to multiple tables, such
         as a joined-inheritance mapping, each dictionary may contain keys
         corresponding to all tables.   All those keys which are present and
         are not part of the primary key are applied to the SET clause of the
         UPDATE statement; the primary key values, which are required, are
         applied to the WHERE clause.


        .. seealso::

            :doc:`queryguide/dml`

            :meth:`.Session.bulk_insert_mappings`

            :meth:`.Session.bulk_save_objects`

        """
        self._bulk_save_mappings(
            mapper,
            mappings,
            isupdate=True,
            isstates=False,
            return_defaults=False,
            update_changed_only=False,
            render_nulls=False,
        )

    def _bulk_save_mappings(
        self,
        mapper: Mapper[_O],
        mappings: Union[Iterable[InstanceState[_O]], Iterable[Dict[str, Any]]],
        *,
        isupdate: bool,
        isstates: bool,
        return_defaults: bool,
        update_changed_only: bool,
        render_nulls: bool,
    ) -> None:
        mapper = _class_to_mapper(mapper)
        self._flushing = True

        transaction = self._autobegin_t()._begin()
        try:
            if isupdate:
                bulk_persistence._bulk_update(
                    mapper,
                    mappings,
                    transaction,
                    isstates=isstates,
                    update_changed_only=update_changed_only,
                )
            else:
                bulk_persistence._bulk_insert(
                    mapper,
                    mappings,
                    transaction,
                    isstates=isstates,
                    return_defaults=return_defaults,
                    render_nulls=render_nulls,
                )
            transaction.commit()

        except:
            with util.safe_reraise():
                transaction.rollback(_capture_exception=True)
        finally:
            self._flushing = False

    def is_modified(
        self, instance: object, include_collections: bool = True
    ) -> bool:
        r"""Return ``True`` if the given instance has locally
        modified attributes.

        This method retrieves the history for each instrumented
        attribute on the instance and performs a comparison of the current
        value to its previously flushed or committed value, if any.

        It is in effect a more expensive and accurate
        version of checking for the given instance in the
        :attr:`.Session.dirty` collection; a full test for
        each attribute's net "dirty" status is performed.

        E.g.::

            return session.is_modified(someobject)

        A few caveats to this method apply:

        * Instances present in the :attr:`.Session.dirty` collection may
          report ``False`` when tested with this method.  This is because
          the object may have received change events via attribute mutation,
          thus placing it in :attr:`.Session.dirty`, but ultimately the state
          is the same as that loaded from the database, resulting in no net
          change here.
        * Scalar attributes may not have recorded the previously set
          value when a new value was applied, if the attribute was not loaded,
          or was expired, at the time the new value was received - in these
          cases, the attribute is assumed to have a change, even if there is
          ultimately no net change against its database value. SQLAlchemy in
          most cases does not need the "old" value when a set event occurs, so
          it skips the expense of a SQL call if the old value isn't present,
          based on the assumption that an UPDATE of the scalar value is
          usually needed, and in those few cases where it isn't, is less
          expensive on average than issuing a defensive SELECT.

          The "old" value is fetched unconditionally upon set only if the
          attribute container has the ``active_history`` flag set to ``True``.
          This flag is set typically for primary key attributes and scalar
          object references that are not a simple many-to-one.  To set this
          flag for any arbitrary mapped column, use the ``active_history``
          argument with :func:`.column_property`.

        :param instance: mapped instance to be tested for pending changes.
        :param include_collections: Indicates if multivalued collections
         should be included in the operation.  Setting this to ``False`` is a
         way to detect only local-column based properties (i.e. scalar columns
         or many-to-one foreign keys) that would result in an UPDATE for this
         instance upon flush.

        """
        state = object_state(instance)

        if not state.modified:
            return False

        dict_ = state.dict

        for attr in state.manager.attributes:
            if (
                not include_collections
                and hasattr(attr.impl, "get_collection")
            ) or not hasattr(attr.impl, "get_history"):
                continue

            (added, unchanged, deleted) = attr.impl.get_history(
                state, dict_, passive=PassiveFlag.NO_CHANGE
            )

            if added or deleted:
                return True
        else:
            return False

    @property
    def is_active(self) -> bool:
        """True if this :class:`.Session` not in "partial rollback" state.

        .. versionchanged:: 1.4 The :class:`_orm.Session` no longer begins
           a new transaction immediately, so this attribute will be False
           when the :class:`_orm.Session` is first instantiated.

        "partial rollback" state typically indicates that the flush process
        of the :class:`_orm.Session` has failed, and that the
        :meth:`_orm.Session.rollback` method must be emitted in order to
        fully roll back the transaction.

        If this :class:`_orm.Session` is not in a transaction at all, the
        :class:`_orm.Session` will autobegin when it is first used, so in this
        case :attr:`_orm.Session.is_active` will return True.

        Otherwise, if this :class:`_orm.Session` is within a transaction,
        and that transaction has not been rolled back internally, the
        :attr:`_orm.Session.is_active` will also return True.

        .. seealso::

            :ref:`faq_session_rollback`

            :meth:`_orm.Session.in_transaction`

        """
        return self._transaction is None or self._transaction.is_active

    @property
    def _dirty_states(self) -> Iterable[InstanceState[Any]]:
        """The set of all persistent states considered dirty.

        This method returns all states that were modified including
        those that were possibly deleted.

        """
        return self.identity_map._dirty_states()

    @property
    def dirty(self) -> IdentitySet:
        """The set of all persistent instances considered dirty.

        E.g.::

            some_mapped_object in session.dirty

        Instances are considered dirty when they were modified but not
        deleted.

        Note that this 'dirty' calculation is 'optimistic'; most
        attribute-setting or collection modification operations will
        mark an instance as 'dirty' and place it in this set, even if
        there is no net change to the attribute's value.  At flush
        time, the value of each attribute is compared to its
        previously saved value, and if there's no net change, no SQL
        operation will occur (this is a more expensive operation so
        it's only done at flush time).

        To check if an instance has actionable net changes to its
        attributes, use the :meth:`.Session.is_modified` method.

        """
        return IdentitySet(
            [
                state.obj()
                for state in self._dirty_states
                if state not in self._deleted
            ]
        )

    @property
    def deleted(self) -> IdentitySet:
        "The set of all instances marked as 'deleted' within this ``Session``"

        return util.IdentitySet(list(self._deleted.values()))

    @property
    def new(self) -> IdentitySet:
        "The set of all instances marked as 'new' within this ``Session``."

        return util.IdentitySet(list(self._new.values()))


_S = TypeVar("_S", bound="Session")


class sessionmaker(_SessionClassMethods, Generic[_S]):
    """A configurable :class:`.Session` factory.

    The :class:`.sessionmaker` factory generates new
    :class:`.Session` objects when called, creating them given
    the configurational arguments established here.

    e.g.::

        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        # an Engine, which the Session will use for connection
        # resources
        engine = create_engine("postgresql+psycopg2://scott:tiger@localhost/")

        Session = sessionmaker(engine)

        with Session() as session:
            session.add(some_object)
            session.add(some_other_object)
            session.commit()

    Context manager use is optional; otherwise, the returned
    :class:`_orm.Session` object may be closed explicitly via the
    :meth:`_orm.Session.close` method.   Using a
    ``try:/finally:`` block is optional, however will ensure that the close
    takes place even if there are database errors::

        session = Session()
        try:
            session.add(some_object)
            session.add(some_other_object)
            session.commit()
        finally:
            session.close()

    :class:`.sessionmaker` acts as a factory for :class:`_orm.Session`
    objects in the same way as an :class:`_engine.Engine` acts as a factory
    for :class:`_engine.Connection` objects.  In this way it also includes
    a :meth:`_orm.sessionmaker.begin` method, that provides a context
    manager which both begins and commits a transaction, as well as closes
    out the :class:`_orm.Session` when complete, rolling back the transaction
    if any errors occur::

        Session = sessionmaker(engine)

        with Session.begin() as session:
            session.add(some_object)
            session.add(some_other_object)
        # commits transaction, closes session

    .. versionadded:: 1.4

    When calling upon :class:`_orm.sessionmaker` to construct a
    :class:`_orm.Session`, keyword arguments may also be passed to the
    method; these arguments will override that of the globally configured
    parameters.  Below we use a :class:`_orm.sessionmaker` bound to a certain
    :class:`_engine.Engine` to produce a :class:`_orm.Session` that is instead
    bound to a specific :class:`_engine.Connection` procured from that engine::

        Session = sessionmaker(engine)

        # bind an individual session to a connection

        with engine.connect() as connection:
            with Session(bind=connection) as session:
                ...  # work with session

    The class also includes a method :meth:`_orm.sessionmaker.configure`, which
    can be used to specify additional keyword arguments to the factory, which
    will take effect for subsequent :class:`.Session` objects generated. This
    is usually used to associate one or more :class:`_engine.Engine` objects
    with an existing
    :class:`.sessionmaker` factory before it is first used::

        # application starts, sessionmaker does not have
        # an engine bound yet
        Session = sessionmaker()

        # ... later, when an engine URL is read from a configuration
        # file or other events allow the engine to be created
        engine = create_engine("sqlite:///foo.db")
        Session.configure(bind=engine)

        sess = Session()
        # work with session

    .. seealso::

        :ref:`session_getting` - introductory text on creating
        sessions using :class:`.sessionmaker`.

    """

    class_: Type[_S]

    @overload
    def __init__(
        self,
        bind: Optional[_SessionBind] = ...,
        *,
        class_: Type[_S],
        autoflush: bool = ...,
        expire_on_commit: bool = ...,
        info: Optional[_InfoType] = ...,
        **kw: Any,
    ): ...

    @overload
    def __init__(
        self: "sessionmaker[Session]",
        bind: Optional[_SessionBind] = ...,
        *,
        autoflush: bool = ...,
        expire_on_commit: bool = ...,
        info: Optional[_InfoType] = ...,
        **kw: Any,
    ): ...

    def __init__(
        self,
        bind: Optional[_SessionBind] = None,
        *,
        class_: Type[_S] = Session,  # type: ignore
        autoflush: bool = True,
        expire_on_commit: bool = True,
        info: Optional[_InfoType] = None,
        **kw: Any,
    ):
        r"""Construct a new :class:`.sessionmaker`.

        All arguments here except for ``class_`` correspond to arguments
        accepted by :class:`.Session` directly.  See the
        :meth:`.Session.__init__` docstring for more details on parameters.

        :param bind: a :class:`_engine.Engine` or other :class:`.Connectable`
         with
         which newly created :class:`.Session` objects will be associated.
        :param class\_: class to use in order to create new :class:`.Session`
         objects.  Defaults to :class:`.Session`.
        :param autoflush: The autoflush setting to use with newly created
         :class:`.Session` objects.

         .. seealso::

            :ref:`session_flushing` - additional background on autoflush

        :param expire_on_commit=True: the
         :paramref:`_orm.Session.expire_on_commit` setting to use
         with newly created :class:`.Session` objects.

        :param info: optional dictionary of information that will be available
         via :attr:`.Session.info`.  Note this dictionary is *updated*, not
         replaced, when the ``info`` parameter is specified to the specific
         :class:`.Session` construction operation.

        :param \**kw: all other keyword arguments are passed to the
         constructor of newly created :class:`.Session` objects.

        """
        kw["bind"] = bind
        kw["autoflush"] = autoflush
        kw["expire_on_commit"] = expire_on_commit
        if info is not None:
            kw["info"] = info
        self.kw = kw
        # make our own subclass of the given class, so that
        # events can be associated with it specifically.
        self.class_ = type(class_.__name__, (class_,), {})

    def begin(self) -> contextlib.AbstractContextManager[_S]:
        """Produce a context manager that both provides a new
        :class:`_orm.Session` as well as a transaction that commits.


        e.g.::

            Session = sessionmaker(some_engine)

            with Session.begin() as session:
                session.add(some_object)

            # commits transaction, closes session

        .. versionadded:: 1.4


        """

        session = self()
        return session._maker_context_manager()

    def __call__(self, **local_kw: Any) -> _S:
        """Produce a new :class:`.Session` object using the configuration
        established in this :class:`.sessionmaker`.

        In Python, the ``__call__`` method is invoked on an object when
        it is "called" in the same way as a function::

            Session = sessionmaker(some_engine)
            session = Session()  # invokes sessionmaker.__call__()

        """
        for k, v in self.kw.items():
            if k == "info" and "info" in local_kw:
                d = v.copy()
                d.update(local_kw["info"])
                local_kw["info"] = d
            else:
                local_kw.setdefault(k, v)
        return self.class_(**local_kw)

    def configure(self, **new_kw: Any) -> None:
        """(Re)configure the arguments for this sessionmaker.

        e.g.::

            Session = sessionmaker()

            Session.configure(bind=create_engine("sqlite://"))
        """
        self.kw.update(new_kw)

    def __repr__(self) -> str:
        return "%s(class_=%r, %s)" % (
            self.__class__.__name__,
            self.class_.__name__,
            ", ".join("%s=%r" % (k, v) for k, v in self.kw.items()),
        )


def close_all_sessions() -> None:
    """Close all sessions in memory.

    This function consults a global registry of all :class:`.Session` objects
    and calls :meth:`.Session.close` on them, which resets them to a clean
    state.

    This function is not for general use but may be useful for test suites
    within the teardown scheme.

    .. versionadded:: 1.3

    """

    for sess in _sessions.values():
        sess.close()


def make_transient(instance: object) -> None:
    """Alter the state of the given instance so that it is :term:`transient`.

    .. note::

        :func:`.make_transient` is a special-case function for
        advanced use cases only.

    The given mapped instance is assumed to be in the :term:`persistent` or
    :term:`detached` state.   The function will remove its association with any
    :class:`.Session` as well as its :attr:`.InstanceState.identity`. The
    effect is that the object will behave as though it were newly constructed,
    except retaining any attribute / collection values that were loaded at the
    time of the call.   The :attr:`.InstanceState.deleted` flag is also reset
    if this object had been deleted as a result of using
    :meth:`.Session.delete`.

    .. warning::

        :func:`.make_transient` does **not** "unexpire" or otherwise eagerly
        load ORM-mapped attributes that are not currently loaded at the time
        the function is called.   This includes attributes which:

        * were expired via :meth:`.Session.expire`

        * were expired as the natural effect of committing a session
          transaction, e.g. :meth:`.Session.commit`

        * are normally :term:`lazy loaded` but are not currently loaded

        * are "deferred" (see :ref:`orm_queryguide_column_deferral`) and are
          not yet loaded

        * were not present in the query which loaded this object, such as that
          which is common in joined table inheritance and other scenarios.

        After :func:`.make_transient` is called, unloaded attributes such
        as those above will normally resolve to the value ``None`` when
        accessed, or an empty collection for a collection-oriented attribute.
        As the object is transient and un-associated with any database
        identity, it will no longer retrieve these values.

    .. seealso::

        :func:`.make_transient_to_detached`

    """
    state = attributes.instance_state(instance)
    s = _state_session(state)
    if s:
        s._expunge_states([state])

    # remove expired state
    state.expired_attributes.clear()

    # remove deferred callables
    if state.callables:
        del state.callables

    if state.key:
        del state.key
    if state._deleted:
        del state._deleted


def make_transient_to_detached(instance: object) -> None:
    """Make the given transient instance :term:`detached`.

    .. note::

        :func:`.make_transient_to_detached` is a special-case function for
        advanced use cases only.

    All attribute history on the given instance
    will be reset as though the instance were freshly loaded
    from a query.  Missing attributes will be marked as expired.
    The primary key attributes of the object, which are required, will be made
    into the "key" of the instance.

    The object can then be added to a session, or merged
    possibly with the load=False flag, at which point it will look
    as if it were loaded that way, without emitting SQL.

    This is a special use case function that differs from a normal
    call to :meth:`.Session.merge` in that a given persistent state
    can be manufactured without any SQL calls.

    .. seealso::

        :func:`.make_transient`

        :meth:`.Session.enable_relationship_loading`

    """
    state = attributes.instance_state(instance)
    if state.session_id or state.key:
        raise sa_exc.InvalidRequestError("Given object must be transient")
    state.key = state.mapper._identity_key_from_state(state)
    if state._deleted:
        del state._deleted
    state._commit_all(state.dict)
    state._expire_attributes(state.dict, state.unloaded)


def object_session(instance: object) -> Optional[Session]:
    """Return the :class:`.Session` to which the given instance belongs.

    This is essentially the same as the :attr:`.InstanceState.session`
    accessor.  See that attribute for details.

    """

    try:
        state = attributes.instance_state(instance)
    except exc.NO_STATE as err:
        raise exc.UnmappedInstanceError(instance) from err
    else:
        return _state_session(state)


_new_sessionid = util.counter()